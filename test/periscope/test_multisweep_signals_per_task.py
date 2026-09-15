"""Test that concurrent multisweep tasks have independent progress signals."""

import pytest

pytest.importorskip("PyQt6")

from test.qt_helpers import spin  # noqa: E402

from rfmux.tools.periscope.multisweep_panel import MultisweepPanel  # noqa: E402
from rfmux.tools.periscope.tasks import MultisweepSignals  # noqa: E402


def _panel(module):
    return MultisweepPanel(target_module=module,
                           initial_params={"resonance_frequencies": [1.0e9],
                                           "amps": [0.001]},
                           dac_scales={module: -0.5})


def test_an_earlier_panel_still_gets_its_progress(qt_app):
    first, second = _panel(1), _panel(2)
    first_signals, second_signals = MultisweepSignals(), MultisweepSignals()
    first.connect_task_signals(first_signals)
    second.connect_task_signals(second_signals)   # the wiring that used to silence `first`

    first_signals.progress.emit(1, 40.0)
    second_signals.progress.emit(2, 90.0)
    spin(qt_app)

    assert first.progress_bar.value() == 40
    assert second.progress_bar.value() == 90

    first.close()
    second.close()
    spin(qt_app)
