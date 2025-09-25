from functools import lru_cache

from .create_config import ConfigSingleton, PydanticConfig


# Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls. It can save time when an expensive or I/O bound function is periodically called with the same arguments.
@lru_cache(maxsize=1)
def get_config() -> PydanticConfig:
    return ConfigSingleton().config
