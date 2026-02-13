from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ccf_streamlines")
except PackageNotFoundError:
    # package is not installed
    __version__ = "uninstalled"
