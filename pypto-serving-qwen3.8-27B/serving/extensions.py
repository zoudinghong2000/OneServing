"""Optional OneServing engine providers. Discovery only runs on explicit opt-in."""
from importlib.metadata import entry_points
import os


def backend_factory(default_engine):
    name = os.environ.get("ONESERVING_BACKEND")
    if not name:
        return default_engine
    candidates = list(entry_points(group="oneserving.backends", name=name))
    if len(candidates) != 1:
        raise ValueError(f"Expected one installed oneserving.backends provider named {name!r}, found {len(candidates)}")
    return candidates[0].load()
