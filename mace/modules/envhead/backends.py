"""In-place cuEq ACE contraction and fused tensor-product conversion.

Converted modules own the trainable parameters. No detached execution copies,
parameter caches, or per-forward state-dict remapping are involved. The existing
mul_ir message layout, foundation convolutions and e3nn linears remain unchanged.
ENV channelwise tensor products use uniform_1d CUDA kernels; the full-message
comparison uses fused_tp. ACE contractions use cuEq symmetric products.
"""

from copy import deepcopy
import torch
from mace.modules.blocks import EquivariantProductBasisBlock
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.cli.convert_e3nn_cueq import transfer_symmetric_contractions


def _convert_fctp(source, device, dtype):
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    from mace.tools.cg import O3_e3nn

    target = cuet.FullyConnectedTensorProduct(
        cue.Irreps(O3_e3nn, str(source.irreps_in1)),
        cue.Irreps(O3_e3nn, str(source.irreps_in2)),
        cue.Irreps(O3_e3nn, str(source.irreps_out)),
        layout=cue.mul_ir,
        shared_weights=source.shared_weights,
        internal_weights=source.internal_weights,
        device=device,
        dtype=dtype,
        math_dtype=dtype,
        method="fused_tp",
    )
    if source.internal_weights:
        with torch.no_grad():
            target.weight.copy_(source.weight.reshape_as(target.weight))
        target.weight.requires_grad_(source.weight.requires_grad)
    return target


def _convert_channelwise(source, device, dtype):
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    from mace.tools.cg import O3_e3nn

    target = cuet.ChannelWiseTensorProduct(
        cue.Irreps(O3_e3nn, str(source.irreps_in1)),
        cue.Irreps(O3_e3nn, str(source.irreps_in2)),
        cue.Irreps(O3_e3nn, str(source.irreps_out)),
        layout=cue.mul_ir,
        shared_weights=False,
        internal_weights=False,
        device=device,
        dtype=dtype,
        math_dtype=dtype,
        method="uniform_1d",
    )
    if (
        str(target.irreps_out) != str(source.irreps_out)
        or target.weight_numel != source.weight_numel
    ):
        raise ValueError("cuEq channelwise paths differ from native MACE paths")
    return target


def _convert_product(source, reduced):
    contraction = source.symmetric_contractions
    if not hasattr(contraction, "contractions"):
        return source
    parameter = next(source.parameters())
    first = contraction.contractions[0]
    degree = first.correlation
    config = CuEquivarianceConfig(
        enabled=True, layout="mul_ir", group="O3_e3nn", optimize_symmetric=True
    )
    if not config.enabled:
        raise ImportError("cuEquivariance and its PyTorch package are required")
    target = EquivariantProductBasisBlock(
        node_feats_irreps=contraction.irreps_in,
        target_irreps=contraction.irreps_out,
        correlation=degree,
        num_elements=first.weights_max.shape[0],
        use_sc=source.use_sc,
        use_agnostic_product=getattr(source, "use_agnostic_product", False),
        use_reduced_cg=bool(reduced),
        cueq_config=config,
    ).to(device=parameter.device, dtype=parameter.dtype)
    values = {}
    transfer_symmetric_contractions(
        {"products.0." + k: v for k, v in source.state_dict().items()},
        values,
        num_product_irreps=len(contraction.irreps_out) - 1,
        products=[source],
        correlation=degree,
        num_layers=1,
        use_reduced_cg=bool(reduced),
        keep_last_layer_irreps=True,
    )
    with torch.no_grad():
        target.symmetric_contractions.weight.copy_(
            values["products.0.symmetric_contractions.weight"]
        )
    target.linear = deepcopy(source.linear)
    target.symmetric_contractions.requires_grad_(
        any(p.requires_grad for p in contraction.parameters())
    )
    target.train(source.training)
    return target


def enable_cueq(model):
    """Accelerate native/ENV ACE products on CUDA; call before optimizer creation."""
    parameter = next(model.parameters())
    if parameter.device.type != "cuda":
        raise ValueError("cuEq conversion requires a CUDA model")
    if getattr(model, "_cueq_enabled", False):
        return model
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(parameter.dtype)
        with torch.random.fork_rng(devices=[parameter.device.index or 0]):
            model.foundation.products = torch.nn.ModuleList(
                [
                    _convert_product(
                        p, getattr(model.foundation, "use_reduced_cg", False)
                    )
                    for p in model.foundation.products
                ]
            )
            for layer in model.env_layers:
                if hasattr(layer, "product"):
                    layer.product = _convert_product(layer.product, False)
                    convert = (
                        _convert_channelwise
                        if getattr(layer, "message_mode", "full") == "channelwise"
                        else _convert_fctp
                    )
                    layer.tp = convert(layer.tp, parameter.device, parameter.dtype)
                elif layer.config.backend == "e2former":
                    layer.operator.tp = _convert_fctp(
                        layer.operator.tp, parameter.device, parameter.dtype
                    )
    finally:
        torch.set_default_dtype(old_dtype)
    model._cueq_enabled = True
    model.config.cueq = True
    return model
