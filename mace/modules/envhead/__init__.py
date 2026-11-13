"""Environment-aware MACE models."""
__all__ = ["EnvHeadConfig", "EnvMACE"]


def __getattr__(name):
    # Running a dataset script must not eagerly import the model stack.
    from importlib import import_module

    modules = {
        "EnvHeadConfig": ".modules.config",
        "EnvMACE": ".modules.model",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
