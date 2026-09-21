"""Nonperiodic, fixed-source direct-delta environment extension for MACE."""
__all__ = ["EnvHeadConfig", "EnvMACE", "NativeMACEAdapter", "register_adapter"]


def __getattr__(name):
    # Running a dataset script must not eagerly import the model stack.
    from importlib import import_module

    modules = {
        "EnvHeadConfig": ".modules.config",
        "EnvMACE": ".modules.model",
        "NativeMACEAdapter": ".adapters",
        "register_adapter": ".adapters",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
