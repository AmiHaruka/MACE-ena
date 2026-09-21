"""Explicit eager foundation adapters; unknown subclasses never fall through.

Adapters are stateless Python objects, not trainable modules. The owner retains
the foundation's original state_dict keys. External adapters must be imported
and registered before construction or checkpoint loading.
"""
from dataclasses import dataclass

from mace.modules.extensions import MACELES
from mace.modules.models import MACE, ScaleShiftMACE


@dataclass(frozen=True)
class NativeMACEAdapter:
    name: str
    model_type: type

    def validate(self, foundation):
        if len(getattr(foundation, "heads", ["Default"])) != 1:
            raise ValueError("Only single-energy-head foundations are supported")
        if getattr(foundation, "compute_bec", False):
            raise NotImplementedError(
                "Disable foundation BEC computation before EnvHead adaptation"
            )

    def interaction(self, owner):
        """Native [N, channel, sum(2l+1)] pre-product layout and irreps."""
        return owner.interactions[-1]

    def evaluate(self, owner, data):
        # A scoped observer preserves the native energy forward, including LES.
        # Never detach the captured tensor. EnvMACE guards reentrant calls.
        target = owner
        if getattr(owner, "_envhead_enable_cueq", False):
            from ..backends import get_cueq_foundation

            target = get_cueq_foundation(owner)
        taps = []
        handle = self.interaction(target).register_forward_hook(
            lambda _module, _args, output: taps.append(output[0])
        )
        try:
            result = self.model_type.forward(
                target, data, training=True, compute_force=False
            )
        finally:
            handle.remove()
        if len(taps) != 1:
            raise RuntimeError("Adapter must capture exactly one final interaction")
        return result, taps[0]


_ADAPTERS = {}


def register_adapter(adapter):
    """Register one exact foundation type under a stable, versioned name."""
    if adapter.name in _ADAPTERS or any(
        a.model_type is adapter.model_type for a in _ADAPTERS.values()
    ):
        raise ValueError("Duplicate foundation adapter name or type")
    _ADAPTERS[adapter.name] = adapter
    return adapter


def foundation_adapter(model):
    name = getattr(model, "_envhead_adapter_name", None)
    if name is not None:
        if name not in _ADAPTERS:
            raise TypeError(f"Import and register foundation adapter {name!r}")
        return _ADAPTERS[name]
    for adapter in _ADAPTERS.values():
        if type(model) is adapter.model_type:
            return adapter
    raise TypeError(
        f"No EnvHead adapter registered for {type(model).__module__}.{type(model).__name__}; "
        "implement the explicit adapter contract"
    )


register_adapter(NativeMACEAdapter("mace-v1", MACE))
register_adapter(NativeMACEAdapter("scale-shift-mace-v1", ScaleShiftMACE))

register_adapter(NativeMACEAdapter("mace-les-v1", MACELES))
