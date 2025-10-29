try:
    from ._version import __version__  # type: ignore
except Exception:
    __version__ = "0.0.0"

__all__ = ["__version__"]

