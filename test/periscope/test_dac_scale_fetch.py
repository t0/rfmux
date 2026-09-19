"""The blocking DAC-scale fetch returns only once its thread has ended:
a QThread destroyed while its thread still runs aborts the process."""

import time

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope import app_runtime, tasks  # noqa: E402
from test.qt_helpers import Board, bare_periscope  # noqa: E402


def test_the_blocking_fetch_outlives_its_thread(qt_app, monkeypatch):
    made = []

    class SlowTail(tasks.DACScaleFetcher):
        ended = False

        def __init__(self, crs):
            super().__init__(crs)
            made.append(self)

        def run(self):
            super().run()            # emits the scales
            time.sleep(0.2)          # and is still running after that
            self.ended = True

    monkeypatch.setattr(app_runtime, "DACScaleFetcher", SlowTail)
    p = bare_periscope(monkeypatch, crs=Board())
    scales = p.fetch_dac_scales_blocking()
    [fetcher] = made
    assert fetcher.ended and not fetcher.isRunning()
    assert scales[1] == -2.0         # the board's scale less the label offset


def test_the_rerun_dialog_finds_the_board_above_a_dock(qt_app, monkeypatch):
    """The re-run dialog shows the panel's scales at once and fetches
    fresh ones from the main window wherever that sits above the panel,
    a dock in between or not."""
    from PyQt6 import QtWidgets
    from rfmux.tools.periscope import network_analysis_dialog as nad
    from rfmux.tools.periscope.network_analysis_dialog import (
        NetworkAnalysisParamsDialog)

    started = []

    class Fetcher(tasks.DACScaleFetcher):
        def start(self):
            started.append(self.crs)

    monkeypatch.setattr(nad, "DACScaleFetcher", Fetcher)
    main = QtWidgets.QMainWindow()
    main.crs = Board()
    dock = QtWidgets.QDockWidget(main)
    panel = QtWidgets.QWidget(dock)
    dock.setWidget(panel)
    dlg = NetworkAnalysisParamsDialog(panel, {"amps": [0.01], "fmin": 1e8,
                                              "fmax": 2e8},
                                      dac_scales={1: -1.5})
    assert dlg.dac_scales == {1: -1.5}
    assert started == [main.crs]
    dlg.close()
    main.close()
