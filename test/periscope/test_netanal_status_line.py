"""The netanal panel's status line, and the cycle it must not close.

Routine outcomes -- how many resonances a search found, where the file went --
go here rather than into a dialog. The line clears itself on a single-shot
timer, and how that timer is connected matters: see
``test_viewbox_lifetime.py`` for why a lambda slot on one of a panel's own
widgets is a segfault waiting for the collector.
"""

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from test.qt_helpers import spin_until  # noqa: E402

from rfmux.tools.periscope.network_analysis_panel import (  # noqa: E402
    NetworkAnalysisPanel,
)


@pytest.fixture
def panel(qt_app):
    made = NetworkAnalysisPanel(module=1)
    yield made
    made.close()


def test_the_status_line_says_what_happened(panel):
    panel._show_status("7 resonances")

    assert panel.status_label.text() == "7 resonances"


def test_the_status_line_clears_itself(panel, qt_app):
    panel._show_status("7 resonances")

    panel._status_timer.setInterval(1)      # rather than wait out the real one
    panel._status_timer.start()

    assert spin_until(qt_app, lambda: panel.status_label.text() == "")


def test_the_clearing_timer_fires_at_the_label_and_not_at_the_panel(panel):
    """The receiver has to be the label's own slot.

    Qt severs a connection whose receiver is destroyed. A lambda over the panel
    is not severed -- and it closes widget -> lambda -> panel -> widget, so the
    panel outlives its C++ side and the timer then fires into a deleted label.
    That crash lands in whatever test happens to be running eight seconds
    later, which is a long way from the code that caused it.

    ``disconnect`` raises if that connection was never made, which is what
    fails if the lambda comes back.
    """
    panel._status_timer.timeout.disconnect(panel.status_label.clear)
