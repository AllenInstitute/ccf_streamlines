from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ccf_streamlines")
except PackageNotFoundError:
    # package is not installed; use a PEP 440-parseable sentinel so consumers
    # calling packaging.version.parse(__version__) do not blow up
    __version__ = "0+unknown"
