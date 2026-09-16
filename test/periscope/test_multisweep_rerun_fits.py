"""A multisweep re-run that chooses the fitted frequencies centres each
power's sweep on that power's own fit."""

import pytest

pytest.importorskip("PyQt6")

from test.qt_helpers import spin  # noqa: E402

from rfmux.tools.periscope.tasks import sweep_centres  # noqa: E402
from rfmux.tools.periscope.multisweep_dialog import MultisweepDialog  # noqa: E402
from rfmux.tools.periscope.multisweep_panel import MultisweepPanel  # noqa: E402

CONCEPTUAL = [100.0e6, 200.0e6]


def _entry(amp, fr=None, bias=None):
    e = {"amplitude": amp, "direction": "upward",
         "original_center_frequency": 0.0}
    if fr is not None:
        e.update(skewed_fit_success=True, fit_params={"fr": fr})
    if bias is not None:
        e["bias_frequency"] = bias
    return e


def test_each_power_takes_its_own_fit_over_the_remembered_bias_point():
    """The table wins for the amplitude nearest the one swept; without
    a table the last sweep's bias point wins, then the baseline."""
    table = {0.01: [100.1e6, 200.1e6], 0.03: [100.3e6, 200.3e6]}
    remembered = lambda idx, amp: 150.0e6 if idx == 0 else None
    assert sweep_centres(0.011, CONCEPTUAL, table, remembered) == \
        [100.1e6, 200.1e6]
    assert sweep_centres(0.03, CONCEPTUAL, table, remembered) == \
        [100.3e6, 200.3e6]
    assert sweep_centres(0.03, CONCEPTUAL, None, remembered) == \
        [150.0e6, 200.0e6]


def test_the_panel_tables_the_fits_per_amplitude(qt_app):
    panel = MultisweepPanel(dark_mode=False, target_module=1,
                            initial_params={"resonance_frequencies": CONCEPTUAL})
    panel.conceptual_section_frequencies = list(CONCEPTUAL)
    panel.results_by_detector = {
        1: {0: _entry(0.01, fr=100.1e6), 1: _entry(0.03, fr=100.3e6)},
        # The second section: no fit at 0.03, so its bias point stands
        # in; nothing at all at 0.01, so the conceptual frequency does.
        2: {1: _entry(0.03, bias=200.3e6)},
    }
    table = panel._fit_frequencies_by_amp(2)
    assert table == {0.01: [100.1e6, 200.0e6], 0.03: [100.3e6, 200.3e6]}
    # The dialog is seeded with the lowest power's fits.
    assert panel._get_fit_frequencies(CONCEPTUAL) == [100.1e6, 200.0e6]
    panel.close()
    spin(qt_app)


def test_the_dialog_says_when_the_fits_were_chosen(qt_app):
    fits = [100.1e6, 200.1e6]
    dlg = MultisweepDialog(section_center_frequencies=CONCEPTUAL,
                           dac_scales={1: -0.5}, current_module=1,
                           initial_params={"amps": [0.01]},
                           fit_frequencies=fits)
    assert dlg.get_parameters()["use_fit_frequencies"] is False
    dlg.section_freq_combo.setCurrentIndex(1)
    params = dlg.get_parameters()
    assert params["use_fit_frequencies"] is True
    assert params["resonance_frequencies"] == fits
    dlg.close()
    spin(qt_app)
