__all__ = [
    "core",
    "algorithms",
    "tools",
    "streamer",
    "Crate",
    "CRS",
    "ReadoutModule",
    "ReadoutChannel",
    "macro",
    "algorithm",
    "HardwareMap",
    "load_dirfile",
    "load_session",
    "get_session",
    "set_session",
    "load_from_database",
    "TuberRemoteError",
]

# Version from setuptools-scm
try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

# Check version numbers
import sys

if sys.version_info < (3, 10):
    raise RuntimeError("Python >= 3.10 is required.")

import importlib.metadata  # requires Python 3.8+
from importlib.metadata import PackageNotFoundError
from packaging.version import parse

tuples = (("sqlalchemy", "2.0.0"), ("IPython", "8.0.0"))

for package, min_version in tuples:
    try:
        installed_version = importlib.metadata.version(package)
    except PackageNotFoundError as e:
        raise PackageNotFoundError(f"Package {package} is not installed!")

    if parse(installed_version) < parse(min_version):
        raise RuntimeError(
            f"Package {package} {installed_version} is too old! At least {min_version} is required."
        )

# For ipython sessions, activate awaitless
try:
    from . import awaitless
    awaitless.load_ipython_extension()
except (ImportError, RuntimeError):
    pass

# Alias from long.internal.name down to something reasonable
from .core.crs import (
    CRS,
    Crate,
    ReadoutChannel,
    ReadoutModule,
    Resonator,
    Wafer,
)

from .core.hardware_map import macro, algorithm, HardwareMap
from .core.dirfile import load_dirfile
from .core.session import load_session, get_session, set_session, load_from_database

from . import core, algorithms, streamer, tools

from .tuber import TuberRemoteError

# vim: sts=4 ts=4 sw=4 tw=78 smarttab expandtab
