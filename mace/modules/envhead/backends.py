"""In-place cuEq ACE contraction and fused tensor-product conversion.

Converted modules own the trainable parameters. No detached execution copies,
parameter caches, or per-forward state-dict remapping are involved. The existing
mul_ir message layout and e3nn linears remain unchanged. Optional native
convolution fusion replaces no parameters and can upgrade existing cuEq models.
ENV channelwise tensor products use uniform_1d CUDA kernels; the full-message
comparison uses fused_tp. ACE contractions use cuEq symmetric products.
"""

from copy import deepcopy
import torch
from e3nn import o3
from mace.modules.blocks import (
    EquivariantProductBasisBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.wrapper_ops import CuEquivarianceConfig, TensorProduct
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


class NativeLayoutConvolution(torch.nn.Module):
    """Use the native indexed cuEq kernel while preserving MACE's mul_ir ABI."""

    def __init__(self, kernel, irreps_in, irreps_out):
        super().__init__()
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        from mace.tools.cg import O3_e3nn

        self.kernel = kernel
        self.input_layout = cuet.TransposeIrrepsLayout(
            cue.Irreps(O3_e3nn, str(irreps_in)),
            source=cue.mul_ir,
            target=cue.ir_mul,
            use_fallback=True,
        )
        self.output_layout = cuet.TransposeIrrepsLayout(
            cue.Irreps(O3_e3nn, str(irreps_out)),
            source=cue.ir_mul,
            target=cue.mul_ir,
            use_fallback=True,
        )

    def forward(self, features, spherical, weights, edges):
        message = self.kernel(self.input_layout(features), spherical, weights, edges)
        return self.output_layout(message)


def _convert_native_convolutions(model):
    if (
        getattr(model, "format_version", 4) != 4
        or getattr(model.config, "environment_mode", None) != "once"
    ):
        return
    for block in model.foundation.interactions:
        if isinstance(block.conv_tp, NativeLayoutConvolution):
            continue
        if hasattr(block, "conv_fusion"):
            # A model already constructed with native convolution fusion owns
            # its layout convention; do not wrap it a second time.
            continue
        source = block.conv_tp
        if not isinstance(
            block, (RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock)
        ):
            raise ValueError(
                f"Unvalidated native convolution block: {type(block).__name__}"
            )
        if not isinstance(source, o3.TensorProduct) or list(source.parameters()):
            raise ValueError(
                "Native convolution fusion requires an external-weight e3nn tensor product"
            )
        if any(mul != 1 for mul, _ in source.irreps_in2):
            raise ValueError(
                "Native convolution spherical irreps must have unit multiplicity"
            )
        config = CuEquivarianceConfig(
            enabled=True,
            layout="mul_ir",
            group="O3_e3nn",
            optimize_channelwise=True,
            conv_fusion=True,
        )
        kernel = TensorProduct(
            source.irreps_in1, source.irreps_in2, source.irreps_out, cueq_config=config
        )
        if kernel.weight_numel != source.weight_numel:
            raise ValueError("Native convolution radial weight count changed")
        block.conv_tp = NativeLayoutConvolution(
            kernel, source.irreps_in1, source.irreps_out
        ).to(next(block.parameters()))
        block.conv_fusion = True


def prepare_cueq_device(model):
    """Select cuEq's Torch reference on CPU without changing parameter bases.

    Only parameter-free polynomial and layout executors are replaced. Their descriptors,
    input layouts and trainable owners remain unchanged, including Adam order.
    A later CUDA preparation restores the original execution method.
    """
    if not getattr(model, "_cueq_enabled", False):
        return model
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    from mace.tools.cg import O3_e3nn

    parameter = next(model.parameters())
    cpu = parameter.device.type == "cpu"

    def configure(owner, name, polynomial, cuda_method):
        source = getattr(owner, name)
        # SegmentedPolynomial.__reduce__ omits custom attributes. The enclosing
        # operation retains its CUDA policy when the CPU reference is saved.
        cuda_method = getattr(source, "_envhead_cuda_method", cuda_method)
        method = "naive" if cpu else cuda_method
        if source.method == method and getattr(source.m, "math_dtype", None) == parameter.dtype:
            return
        target = cuet.SegmentedPolynomial(
            polynomial(), method=method,
            math_dtype=parameter.dtype,
        ).to(parameter)
        target.train(source.training)
        if cpu:
            target._envhead_cuda_method = cuda_method
        setattr(owner, name, target)

    def configure_layout(owner, name, irreps):
        source = getattr(owner, name)
        if cpu == getattr(source, "_envhead_cpu_reference", False):
            return
        target = cuet.TransposeIrrepsLayout(
            irreps, source=source.source, target=source.target,
            device=parameter.device, use_fallback=cpu,
        )
        target._envhead_cpu_reference = cpu
        target.train(source.training)
        setattr(owner, name, target)

    for block in model.foundation.interactions:
        if isinstance(block.conv_tp, NativeLayoutConvolution):
            configure(
                block.conv_tp.kernel, "conv_tp",
                lambda: cue.descriptors.channelwise_tensor_product(
                    cue.Irreps(O3_e3nn, str(block.edge_irreps)),
                    cue.Irreps(O3_e3nn, str(block.edge_attrs_irreps)),
                    cue.Irreps(O3_e3nn, str(block.linear.irreps_in)),
                ).flatten_coefficient_modes().squeeze_modes().polynomial,
                "uniform_1d",
            )
    for module in tuple(model.modules()):
        if isinstance(module, cuet.SymmetricContraction):
            configure(module, "f", lambda: module.etp.polynomial, module.method)
            configure_layout(module, "transpose_in", module.etp.inputs[1].irreps)
            configure_layout(module, "transpose_out", module.etp.outputs[0].irreps)
        elif isinstance(module, cuet.ChannelWiseTensorProduct):
            configure(module, "f", lambda: cue.descriptors.channelwise_tensor_product(
                module.irreps_in1, module.irreps_in2, module.irreps_out
            ).polynomial, module.method)
            configure_layout(module, "transpose_in1", module.irreps_in1)
            configure_layout(module, "transpose_in2", module.irreps_in2)
            configure_layout(module, "transpose_out", module.irreps_out)
        elif isinstance(module, cuet.FullyConnectedTensorProduct):
            configure(module, "f", lambda: cue.descriptors.fully_connected_tensor_product(
                module.irreps_in1, module.irreps_in2, module.irreps_out
            ).polynomial, module.method)
            configure_layout(module, "transpose_in1", module.irreps_in1)
            configure_layout(module, "transpose_in2", module.irreps_in2)
            configure_layout(module, "transpose_out", module.irreps_out)
    return model


def enable_cueq(model, *, conv_fusion=False):
    """Prepare CUDA operators before Adam creation or loading its state.

    conv_fusion is an explicit candidate until H20 and numerical gates pass.
    Repeated preparation is idempotent; False does not undo an existing fusion.
    """
    parameter = next(model.parameters())
    if parameter.device.type != "cuda":
        raise ValueError("cuEq conversion requires a CUDA model")
    prepare_cueq_device(model)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(parameter.dtype)
        with torch.random.fork_rng(devices=[parameter.device.index or 0]):
            if not getattr(model, "_cueq_enabled", False):
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
            if conv_fusion:
                _convert_native_convolutions(model)
    finally:
        torch.set_default_dtype(old_dtype)
    model._cueq_enabled = True
    model.config.cueq = True
    return model
