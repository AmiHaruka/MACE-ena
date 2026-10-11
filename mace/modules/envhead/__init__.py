"""Fixed-source EnvHead: v4 single-pass models and preserved v3 models."""
__all__ = ["EnvHeadConfig", "EnvMACE", "EnvHeadConfigV4", "EnvMACEV4"]


def __getattr__(name):
    # Running a dataset script must not eagerly import the model stack.
    from importlib import import_module

    modules = {
        "EnvHeadConfig": ".modules.config",
        "EnvMACE": ".modules.model",
        "EnvHeadConfigV4": ".modules.config_v4",
        "EnvMACEV4": ".modules.model_v4",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
