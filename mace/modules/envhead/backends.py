"""Optional execution backends; saved weights remain in canonical e3nn form.

The cuEq path accelerates the frozen native MACE foundation and head's MACE
operators. Gaussian electrostatics and biased density readouts retain their
PyTorch/e3nn implementations. Backend modules are disposable execution caches,
never optimizer parameter sets or checkpoint authorities.
"""
from copy import deepcopy
import logging

import torch

from .adapters import foundation_adapter


def enable_cueq(model):
    """Enable cuEq execution on an assembled EnvMACE (CUDA only)."""
    adapter = foundation_adapter(model)
    if adapter.name not in ("mace-v1", "scale-shift-mace-v1", "mace-les-v1"):
        raise NotImplementedError(
            "EnvHead cuEq currently supports native MACE/ScaleShiftMACE "
            "(including MACE-OFF) and MACELES; custom adapters are not supported"
        )
    if next(model.parameters()).device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "EnvHead enable_cueq requires a CUDA model and working CUDA device"
        )
    if any(
        p.requires_grad for name, p in model.named_parameters()
        if not name.startswith("env_head.")
    ):
        raise ValueError("cuEq execution requires a frozen foundation")
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401
        import cuequivariance_ops_torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install the cuEquivariance Python and matching CUDA operations packages"
        ) from exc
    # Fail during setup, before allocating a training dataset or starting ASE.
    model.__dict__["_cueq_foundation_cache"] = None
    get_cueq_foundation(model)
    get_cueq_head(model)
    model._envhead_enable_cueq = True
    logging.info(
        "EnvHead backend: foundation/head MACE ops=cuEq (mul_ir); "
        "biased readouts=e3nn; electrostatics=PyTorch"
    )
    if adapter.name == "mace-les-v1":
        logging.info(
            "MACELES backend: energy/LES readouts=original; LES long-range=original "
            "PyTorch on %s; dtype=%s; foundation frozen",
            next(model.parameters()).device, next(model.parameters()).dtype,
        )
    return model


def _signature(model):
    # load_state_dict increments Tensor versions; moves change identities/device.
    # Head updates do not invalidate the frozen foundation cache.
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    return tuple(
        (name, id(value), value._version, value.device, value.dtype)
        for name, value in tensors if not name.startswith("env_head.")
    )


def get_cueq_foundation(model):
    signature = _signature(model)
    cached = model.__dict__.get("_cueq_foundation_cache")
    if cached is not None and cached[0] == signature:
        return cached[1]
    from mace.cli.convert_e3nn_cueq import transfer_weights
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    from mace.modules.models import ScaleShiftMACE
    from mace.tools.scripts_utils import extract_config_mace_model

    adapter = foundation_adapter(model)
    # Reconstruct only the native foundation view, excluding EnvHead. The native
    # converter then performs its own constructor and CG weight transformation.
    source = deepcopy(model)
    del source.env_head
    source.__class__ = adapter.model_type
    source.__dict__.pop("_envhead_enable_cueq", None)
    source.__dict__.pop("_cueq_foundation_cache", None)
    source.__dict__.pop("_cueq_head_cache", None)
    device = next(model.parameters()).device
    # Conversion initializes temporary parameters: preserve training RNG streams.
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    settings = extract_config_mace_model(source)
    is_les = adapter.name == "mace-les-v1"
    if is_les:
        # Old LES checkpoints may not store keep_last_layer_irreps. Infer the
        # actual layout from their modules, never from today's LES constructor.
        settings["keep_last_layer_irreps"] = (
            str(source.products[-1].linear.irreps_out)
            == str(source.products[0].linear.irreps_out)
        )
    # Native run() unconditionally enables fused convolution on CUDA. Its
    # segmented descriptor does not apply the requested mul_ir layout, so keep
    # gather/scatter separate and use the layout-aware cuEq tensor product.
    settings["cueq_config"] = CuEquivarianceConfig(
        enabled=True, layout="mul_ir", group="O3_e3nn",
        optimize_all=True, conv_fusion=False,
    )
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(next(model.parameters()).dtype)
        with torch.random.fork_rng(devices=devices):
            template_type = ScaleShiftMACE if is_les else adapter.model_type
            accelerated = template_type(**settings).to(device)
            if is_les:
                _validate_les_backbone(source, accelerated)
            transfer_weights(
                source, accelerated, len(settings["hidden_irreps"].slices()) - 1,
                settings["correlation"], settings["num_interactions"],
                settings.get("use_reduced_cg", True),
                settings.get("keep_last_layer_irreps", False),
            )
            if is_les:
                _validate_transferred_parameters(source, accelerated)
                # Keep the full LES object and original forward/readouts. Only
                # these three backbone components come from the cuEq template.
                for name in ("node_embedding", "interactions", "products"):
                    setattr(source, name, getattr(accelerated, name))
                accelerated = source
    finally:
        torch.set_default_dtype(old_dtype)
    accelerated.eval().requires_grad_(False)
    model.__dict__["_cueq_foundation_cache"] = (signature, accelerated)
    return accelerated


def _validate_les_backbone(source, target):
    """Reject a template that cannot represent the checkpoint's exact irreps."""
    from e3nn import o3

    def compare(left, right, label):
        if o3.Irreps(str(left)) != o3.Irreps(str(right)):
            raise ValueError(f"MACELES cuEq irreps mismatch at {label}: {left} != {right}")

    for key in ("irreps_in", "irreps_out"):
        compare(getattr(source.node_embedding.linear, key),
                getattr(target.node_embedding.linear, key), f"node_embedding.{key}")
    for group in ("interactions", "products"):
        if len(getattr(source, group)) != len(getattr(target, group)):
            raise ValueError(f"MACELES cuEq layer count mismatch: {group}")
    for index, (left, right) in enumerate(zip(source.interactions, target.interactions)):
        if type(left) is not type(right):
            raise ValueError(f"MACELES cuEq interaction type mismatch at layer {index}")
        for key in ("node_attrs_irreps", "node_feats_irreps", "edge_attrs_irreps",
                    "edge_feats_irreps", "target_irreps", "hidden_irreps", "irreps_out"):
            compare(getattr(left, key), getattr(right, key), f"interactions.{index}.{key}")
    for index, (left, right) in enumerate(zip(source.products, target.products)):
        for key in ("irreps_in", "irreps_out"):
            compare(getattr(left.linear, key), getattr(right.linear, key),
                    f"products.{index}.linear.{key}")


def _validate_transferred_parameters(source, target):
    """Native transfer may skip unmatched keys; disallow that in retained blocks.

    Symmetric contraction weights have a separate CG transformation in native
    transfer_weights; their correctness is checked by energy/derivative parity.
    """
    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    for name, value in target_parameters.items():
        if (name.startswith(("node_embedding.", "interactions.", "products."))
                and ".symmetric_contractions." not in name and value.numel()
                and name not in source_parameters):
            raise ValueError(f"Unexpected MACELES cuEq parameter: {name}")
    for name, value in source_parameters.items():
        if not name.startswith(("node_embedding.", "interactions.", "products.")):
            continue
        if ".symmetric_contractions." in name or value.numel() == 0:
            continue
        converted = target_parameters.get(name)
        if (converted is None or converted.numel() != value.numel()
                or not torch.equal(converted.reshape(-1), value.reshape(-1))):
            raise ValueError(f"MACELES cuEq parameter was not transferred exactly: {name}")


def get_cueq_head(model):
    """Construct operators and a differentiable canonical-to-cuEq weight map."""
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    from .modules.head import EnvHead

    head = model.env_head
    parameter = next(head.parameters())
    signature = (
        id(head), parameter.device, parameter.dtype,
        tuple((name, id(value), value._version) for name, value in head.named_buffers()),
    )
    cached = model.__dict__.get("_cueq_head_cache")
    if cached is not None and cached[0] == signature:
        return cached[1:]
    device = parameter.device
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    config = CuEquivarianceConfig(
        enabled=True, layout="mul_ir", group="O3_e3nn", optimize_all=True,
        conv_fusion=False,
    )
    if not config.enabled:
        raise ImportError("cuEquivariance is not available")
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(parameter.dtype)
        with torch.random.fork_rng(devices=devices):
            accelerated = EnvHead(
                head.tap_irreps, foundation_adapter(model).interaction(model),
                head.config, cueq_config=config,
            ).to(device=device, dtype=parameter.dtype)
    finally:
        torch.set_default_dtype(old_dtype)
    accelerated.requires_grad_(False)
    source = head.state_dict(keep_vars=True)
    mapping = {}
    used = set()
    ignored = set()
    for name, target in accelerated.named_parameters():
        if name in source:
            if source[name].numel() != target.numel():
                raise ValueError(f"cuEq head weight shape mismatch: {name}")
            keys = [name]
        elif name.endswith('.product.symmetric_contractions.weight'):
            prefix = name.removesuffix('.weight')
            block_index = int(name.split('.')[1])
            contractions = head.blocks[block_index].product.symmetric_contractions.contractions
            keys = []
            # Same original-MACE ordering and zero-path filtering as native
            # convert_e3nn_cueq.transfer_symmetric_contractions. EnvHead's product
            # blocks use the original (unreduced) CG basis by construction.
            for index, contraction in enumerate(contractions):
                suffixes = ['_max'] + [f'.{j}' for j in range(contraction.correlation - 1)]
                for suffix in suffixes:
                    key = f'{prefix}.contractions.{index}.weights{suffix}'
                    zeroed = f'{prefix}.contractions.{index}.weights{suffix.replace(".", "_")}_zeroed'
                    if bool(source.get(zeroed, False)):
                        ignored.add(key)
                    else:
                        keys.append(key)
            if sum(source[key].numel() for key in keys) != target.numel():
                raise ValueError(f"cuEq symmetric contraction weight mismatch: {name}")
        else:
            raise ValueError(f"Unmapped cuEq head parameter: {name}")
        mapping[name] = (keys, target.shape)
        used.update(keys)
    unmapped = set(dict(head.named_parameters())) - used - ignored
    if unmapped:
        raise ValueError(f"Canonical head parameters missing from cuEq: {sorted(unmapped)}")
    # Preserve any common nonlearned buffers; backend-only CG buffers are built
    # by cuEq in the selected O3_e3nn basis.
    with torch.no_grad():
        for name, target in accelerated.named_buffers():
            if name in source and target.shape == source[name].shape:
                target.copy_(source[name])
    model.__dict__["_cueq_head_cache"] = (signature, accelerated, mapping)
    return accelerated, mapping


def call_cueq_head(model, *args, **kwargs):
    accelerated, mapping = get_cueq_head(model)
    source = dict(model.env_head.named_parameters())
    parameters = {}
    for name, (keys, shape) in mapping.items():
        value = (
            source[keys[0]] if len(keys) == 1
            else torch.cat([source[k] for k in keys], dim=1)
        )
        parameters[name] = value.reshape(shape)
    # No detach/load_state_dict here: force-loss mixed derivatives must reach
    # the canonical parameters registered with the optimizer.
    return torch.func.functional_call(accelerated, parameters, args, kwargs)
