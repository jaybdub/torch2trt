import os


_CACHE_DIR = os.path.expanduser("~/.cache/torch2trt")


def get_cache_dir() -> str:
    global _CACHE_DIR
    return _CACHE_DIR


def make_cache_dir() -> str:
    if not os.path.exists(_CACHE_DIR):
        os.makedirs(_CACHE_DIR)


def set_cache_dir(path: str) -> str:
    global _CACHE_DIR
    _CACHE_DIR = path
