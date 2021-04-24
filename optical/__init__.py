from .converter import Annotation

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

try:
    __version__ = version(__name__)
except:  # noqa
    pass
