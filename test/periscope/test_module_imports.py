"""Every Periscope module must import.

utils.py is the package's shared import surface: eight panels do
``from .utils import *`` and a dozen more pull names out of it by name,
so a large fraction of its imports are re-exports it never uses itself.
A linter reports those as dead, and removing one raises ImportError on
the next launch — with nothing in the test suite to catch it, because
no other test imports enough of the package.

Importing every module is the cheap check that closes that gap.  It
also catches syntax errors and circular imports in panels no other
test touches.
"""

import importlib
import os
import pkgutil

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

import rfmux.tools.periscope as periscope  # noqa: E402


def _module_names():
    return sorted(
        f"{periscope.__name__}.{m.name}"
        for m in pkgutil.iter_modules(periscope.__path__)
        # __main__ runs argument parsing at import time.
        if m.name != "__main__"
    )


@pytest.mark.parametrize("name", _module_names())
def test_module_imports(name):
    importlib.import_module(name)


def test_the_sweep_actually_covers_the_package():
    """A guard on the guard: if iter_modules ever comes back empty the
    parametrized test above silently passes zero cases."""
    names = _module_names()
    assert len(names) > 20, f"only found {len(names)} periscope modules"
    assert f"{periscope.__name__}.utils" in names
