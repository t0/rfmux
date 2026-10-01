"""The blocking DAC-scale fetch returns only once its thread has ended:
a QThread destroyed while its thread still runs aborts the process."""

import time

import pytest

pytest.importorskip("PyQt6")

from rfmux.core.transferfunctions import BALUN_LOSS_DB  # noqa: E402
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
    assert scales[1] == -0.5 - BALUN_LOSS_DB   # the board's scale less the balun loss
