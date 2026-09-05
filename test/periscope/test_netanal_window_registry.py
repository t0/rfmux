"""A closed Network Analysis must not break the next layout rebuild.

netanal_windows is a plain dict that nothing ever removed from, so closing
a Network Analysis dock destroyed the panel's C++ objects and left the
entry behind.  _build_layout walks that dict through _toggle_zoom_box_mode
on every rebuild -- changing units, toggling PSD, changing channels -- and
each one raised

    RuntimeError: wrapped C/C++ object of type ClickableViewBox
                  has been deleted

from deep inside pyqtgraph, where nothing named the actual cause.

The sibling registry, pulse_capture_windows, already prunes; this one now
does too.
"""

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PyQt6 import QtCore, QtWidgets, sip  # noqa: E402

from test.qt_helpers import spin  # noqa: E402

from rfmux.tools.periscope.app import Periscope  # noqa: E402
from rfmux.tools.periscope.utils import ClickableViewBox  # noqa: E402


def _delete(qt_app, *widgets):
    """Destroy the C++ objects the way closing a dock does.

    deleteLater + event loop, not sip.delete: hard deletion of a live
    QGraphicsView is a use-after-free waiting in pyqtgraph's scene
    bookkeeping, and on CI it segfaulted a LATER test's widget
    construction.  The state under test is the same either way -- a
    live Python wrapper over a dead C++ object.
    """
    for w in widgets:
        w.deleteLater()
    # processEvents() does not deliver DeferredDelete; only the event
    # loop proper (or this) does.
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete)
    assert all(sip.isdeleted(w) for w in widgets), "deferred deletion never ran"


def _panel():
    """A stand-in for NetworkAnalysisWindow: what the walk touches."""
    panel = QtWidgets.QWidget()
    plots = {}
    for name in ("amp_plot", "phase_plot"):
        plots[name] = pg.PlotWidget(parent=panel,
                                    viewBox=ClickableViewBox())
    panel.plots = {1: plots}
    return panel


def _periscope_with(entry):
    """Just enough Periscope to run the walk, without a CRS or a window."""
    p = Periscope.__new__(Periscope)
    p.plots = []
    p.zoom_box_mode = False
    p.netanal_windows = {"na-1": entry}
    return p


def test_a_closed_window_does_not_break_the_rebuild(qt_app):
    panel, dock = _panel(), QtWidgets.QWidget()
    p = _periscope_with({"window": panel, "dock": dock, "signals": None})

    # Alive: the walk reaches the viewboxes and sets the mode.
    p._toggle_zoom_box_mode(True)
    for plot in panel.plots[1].values():
        assert plot.getViewBox().state["mouseMode"] == pg.ViewBox.RectMode

    # Closed, the way closing the dock closes it.
    _delete(qt_app, panel, dock)

    # This is the call that raised on every layout rebuild.
    p._toggle_zoom_box_mode(False)

    # And the dead entry is gone, rather than waiting to raise again.
    assert p.netanal_windows == {}


def test_a_live_window_is_kept(qt_app):
    panel, dock = _panel(), QtWidgets.QWidget()
    p = _periscope_with({"window": panel, "dock": dock, "signals": None})

    assert set(p._live_netanal_windows()) == {"na-1"}
    assert set(p.netanal_windows) == {"na-1"}, "pruned a window that was alive"

    panel.close()
    dock.close()
    spin(qt_app)
    # close() is not deletion: the panel is still usable and must survive.
    assert set(p._live_netanal_windows()) == {"na-1"}


def test_a_half_built_entry_is_pruned(qt_app):
    """An entry with no dock is unusable, and dereferencing it raised too."""
    p = _periscope_with({"window": _panel(), "dock": None, "signals": None})
    assert p._live_netanal_windows() == {}
    assert p.netanal_windows == {}


def test_the_registry_does_not_grow_without_bound(qt_app):
    """Nothing removed entries, so a session accumulated one per open."""
    p = _periscope_with({})
    p.netanal_windows = {}
    for i in range(5):
        panel, dock = _panel(), QtWidgets.QWidget()
        p.netanal_windows[f"na-{i}"] = {"window": panel, "dock": dock}
        _delete(qt_app, panel, dock)
        p._toggle_zoom_box_mode(True)

    assert p.netanal_windows == {}


def test_a_live_panel_with_dead_viewboxes(qt_app):
    """The failure as it was actually reported.

    The traceback named ClickableViewBox, not the panel: Python attribute
    lookups all the way down to ``getViewBox()`` return stored references
    and never touch C++, so the walk got a whole plot tree's worth of live
    wrappers and only fell over on ``setMouseMode``, which emits a signal.
    Pruning by panel does not catch that -- the panel is still there -- so
    the viewbox is checked as well, exactly as the main plots are.
    """
    panel, dock = _panel(), QtWidgets.QWidget()
    p = _periscope_with({"window": panel, "dock": dock, "signals": None})

    _delete(qt_app, *[plot.getViewBox() for plot in panel.plots[1].values()])

    # The panel survives the prune, and the walk still must not raise.
    p._toggle_zoom_box_mode(True)
    assert set(p.netanal_windows) == {"na-1"}
