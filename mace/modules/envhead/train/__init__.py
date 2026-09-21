"""Training entry point; python -m mace.modules.envhead.train remains supported."""


def main():
    from .run import main as run

    return run()


def graphs_from_atoms(*args, **kwargs):
    from ..data.loading import graphs_from_atoms as load

    return load(*args, **kwargs)
