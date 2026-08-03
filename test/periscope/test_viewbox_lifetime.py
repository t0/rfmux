"""
ClickableViewBox must not strongly reference the panel that owns it.

Panels hand their ViewBox a back-pointer (``vb.parent_window = self``) so the
double-click handler in utils.py can reach the window. Held strongly, that
closes a cycle — ViewBox -> panel -> PlotWidget -> ViewBox — and tearing the
panel down then goes through Python's *cyclic* collector, which finalizes PyQt
objects in arbitrary order and frees C++ objects out from under live wrappers.

The mild symptom is pyqtgraph writing "wrapped C/C++ object of type QComboBox
has been deleted" to stderr. The real one is a segfault: building and dropping a
single NoiseSpectrumPanel in a bare script was enough to kill the interpreter.

Two layers here:

* the weakref contract, asserted directly and cheaply — this is what fails if
  ``parent_window`` is ever turned back into a plain attribute;
* the teardown loop that actually crashed, run in a SUBPROCESS. That matters: a
  regression is a segfault, so running it in-process would take the whole pytest
  session down with it instead of failing one test.
"""

import gc
import os
import subprocess
import sys
import textwrap
import weakref

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.utils import ClickableViewBox  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_parent_window_resolves_while_the_window_lives(qt_app):
    """The back-pointer has to keep working — a weakref is only acceptable if
    reads still return the window."""
    window = QtWidgets.QWidget()
    vb = ClickableViewBox()
    vb.parent_window = window
    assert vb.parent_window is window
    # utils.py reads it via getattr; a property must stay transparent to that.
    assert getattr(vb, "parent_window", None) is window


def test_parent_window_does_not_keep_the_window_alive(qt_app):
    """The regression proper: the ViewBox must not be a strong owner.

    Turning parent_window back into a plain attribute fails right here, without
    needing to provoke the crash.
    """
    window = QtWidgets.QWidget()
    vb = ClickableViewBox()
    vb.parent_window = window

    sentinel = weakref.ref(window)
    del window
    gc.collect()

    assert sentinel() is None, \
        "ClickableViewBox is keeping its parent window alive — parent_window " \
        "must hold a weak reference, or panel teardown segfaults"
    assert vb.parent_window is None


def test_parent_window_accepts_none(qt_app):
    """Panels and teardown code may clear it; that must not raise."""
    vb = ClickableViewBox()
    vb.parent_window = None
    assert vb.parent_window is None

    window = QtWidgets.QWidget()
    vb.parent_window = window
    vb.parent_window = None
    assert vb.parent_window is None


def test_unset_parent_window_reads_as_none(qt_app):
    """A freshly built ViewBox has no back-pointer yet."""
    assert ClickableViewBox().parent_window is None


# Panels whose ViewBoxes exist right after construction. NetworkAnalysisPanel is
# absent on purpose: it builds its plots per module when data arrives, so a bare
# instance has none to tear down — the weakref contract above covers it.
_TEARDOWN_SCRIPT = textwrap.dedent(
    '''
    import gc, os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    from rfmux.tools.periscope.noise_spectrum_panel import NoiseSpectrumPanel
    from rfmux.tools.periscope.detector_digest_panel import DetectorDigestPanel
    from rfmux.tools.periscope.multisweep_panel import MultisweepPanel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    detectors = {1: {"conceptual_freq_hz": 4.0e9},
                 2: {"conceptual_freq_hz": 4.1e9}}
    cases = [
        (NoiseSpectrumPanel, dict(detector_id=1, resonance_frequency_ghz=4.0,
                                  all_detectors_data=detectors,
                                  initial_detector_idx=1)),
        (DetectorDigestPanel, dict(detector_id=1)),
        (MultisweepPanel, dict(target_module=1)),
    ]
    for cls, kwargs in cases:
        for _ in range(3):
            panel = cls(**kwargs)
            panel.close()
            del panel
            # gc.collect() is the point: it forces the cyclic collection that
            # the strong back-pointer made lethal.
            gc.collect()
            app.processEvents()
        print("survived " + cls.__name__, flush=True)
    print("ALL PANELS SURVIVED", flush=True)
    '''
)


def test_panels_survive_being_built_and_dropped(qt_app):
    """Build and drop each panel under gc.collect() in a subprocess.

    With a strong parent_window this segfaults (exit -11 / 139); the subprocess
    turns that into a readable failure instead of killing the test session.
    """
    proc = subprocess.run(
        [sys.executable, "-u", "-c", _TEARDOWN_SCRIPT],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )

    assert proc.returncode == 0, (
        f"panel teardown crashed (returncode {proc.returncode}; "
        f"-11/139 is SIGSEGV). This is the parent_window reference cycle.\n"
        f"stdout:\n{proc.stdout}\nstderr tail:\n{proc.stderr[-2000:]}"
    )
    assert "ALL PANELS SURVIVED" in proc.stdout, \
        f"teardown loop did not finish:\n{proc.stdout}\n{proc.stderr[-2000:]}"

    # The cycle also produced pyqtgraph teardown tracebacks; with it gone the
    # run should be clean, which is the cheapest signal that no new cycle crept
    # in through some other strong back-pointer.
    assert "has been deleted" not in proc.stderr, \
        f"pyqtgraph teardown noise is back:\n{proc.stderr[-2000:]}"
