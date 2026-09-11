"""The dialogs that remember, and the two that deliberately do not.

One test per dialog, each asserting the same contract from the operator's
side: what was entered and accepted is what the next dialog opens on. The
helper's own behaviour is pinned in ``test_field_memory.py``; these pin that
each dialog is actually wired to it, and that the fields read off the board
are not.
"""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.initialize_crs_dialog import (  # noqa: E402
    InitializeCRSDialog,
)
from rfmux.tools.periscope.mock_configuration_dialog import (  # noqa: E402
    MockConfigurationDialog,
)
from rfmux.tools.periscope.multisweep_dialog import MultisweepDialog  # noqa: E402
from rfmux.tools.periscope.noise_spectrum_dialog import (  # noqa: E402
    NoiseSpectrumDialog,
)
from rfmux.tools.periscope.network_analysis_dialog import (  # noqa: E402
    NetworkAnalysisDialog,
)
from rfmux.tools.periscope.streamer_config_dialog import (  # noqa: E402
    StreamerConfigDialog,
)


@pytest.fixture
def build(qt_app):
    """Builds dialogs, and closes them however the test ends."""
    made = []

    def make(factory):
        made.append(factory())
        return made[-1]

    yield make
    for widget in made:
        widget.close()


def test_network_analysis_reopens_on_the_last_sweep(build):
    first = build(lambda: NetworkAnalysisDialog(module=1))
    first.fmin_edit.setText("1234.5")
    first.points_edit.setText("4096")
    first.accept()

    second = build(lambda: NetworkAnalysisDialog(module=1))
    assert second.fmin_edit.text() == "1234.5"
    assert second.points_edit.text() == "4096"


def test_multisweep_reopens_on_the_last_sweep(build):
    first = build(lambda: MultisweepDialog(module=1))
    first.span_khz_edit.setText("250")
    first.npoints_edit.setText("301")
    first.downward_cb.setChecked(not first.downward_cb.isChecked())
    wanted = first.downward_cb.isChecked()
    first.accept()

    second = build(lambda: MultisweepDialog(module=1))
    assert second.span_khz_edit.text() == "250"
    assert second.npoints_edit.text() == "301"
    assert second.downward_cb.isChecked() == wanted


def test_multisweep_amplitude_mode_survives_as_one_choice(build):
    """The amplitude radios are exclusive; restoring must leave exactly one."""
    first = build(lambda: MultisweepDialog(module=1))
    first.ramp_radio.setChecked(True)
    first.accept()

    second = build(lambda: MultisweepDialog(module=1))
    radios = (second.single_radio, second.list_radio, second.ramp_radio)
    assert [r.isChecked() for r in radios].count(True) == 1
    assert second.ramp_radio.isChecked()


def test_multisweep_keeps_the_parameters_it_was_opened_on(build):
    """A re-run seeded by the panel shows that run, not the last dialog."""
    first = build(lambda: MultisweepDialog(module=1))
    first.span_khz_edit.setText("250")
    first.accept()

    rerun = build(lambda: MultisweepDialog(
        module=1, initial_params={"span_hz": 40e3}))
    assert rerun.span_khz_edit.text() == "40"


def test_initialize_crs_reopens_on_the_last_choice(build):
    first = build(InitializeCRSDialog)
    first.rb_sma.setChecked(True)
    first.cb_clear_channels.setChecked(False)
    first.accept()

    second = build(InitializeCRSDialog)
    assert second.rb_sma.isChecked()
    assert not second.rb_test.isChecked()
    assert not second.cb_clear_channels.isChecked()


def test_mock_configuration_reopens_on_the_last_array(build):
    first = build(MockConfigurationDialog)
    first.num_resonances_spin.setValue(42)
    first.random_seed_edit.setText("7")
    first.accept()

    second = build(MockConfigurationDialog)
    assert second.num_resonances_spin.value() == 42
    assert second.random_seed_edit.text() == "7"


def test_streamer_keeps_the_pfb_choice_but_rereads_the_board(build):
    first = build(lambda: StreamerConfigDialog(
        module=1, current_dec=6, current_short=False))
    first.pfb_check.setChecked(True)
    first.pfb_channels_edit.setText("3,4")
    first.dec_spin.setValue(2)
    first.short_check.setChecked(True)
    first.accept()

    # A board reporting a different stream: the dialog must show that
    # stream, not the one accepted above.
    second = build(lambda: StreamerConfigDialog(
        module=1, current_dec=6, current_short=False))
    assert second.pfb_check.isChecked()
    assert second.pfb_channels_edit.text() == "3,4"
    assert second.dec_spin.value() == 6
    assert not second.short_check.isChecked()


class _Board:
    """Just the serial: what the noise dialog reads a board for."""

    def __init__(self, serial):
        self.serial = serial


def test_noise_spectrum_reopens_on_the_last_settings(build):
    first = build(lambda: NoiseSpectrumDialog(
        num_resonances=3, crs=_Board("0024")))
    first.samples_edit.setText("50000")
    first.decimation_input.setValue(2)
    first.reference_input.setCurrentText("dBc")
    first.accept()

    second = build(lambda: NoiseSpectrumDialog(
        num_resonances=3, crs=_Board("0024")))
    assert second.samples_edit.text() == "50000"
    assert second.decimation_input.value() == 2
    assert second.reference_input.currentText() == "dBc"


def test_mock_does_not_inherit_a_pfb_spectrum_from_the_board(build):
    """Mock hides the PFB box, so it must not come back checked there."""
    on_board = build(lambda: NoiseSpectrumDialog(
        num_resonances=3, crs=_Board("0024")))
    on_board.pfb_checkbox.setChecked(True)
    on_board.accept()

    in_mock = build(lambda: NoiseSpectrumDialog(
        num_resonances=3, crs=_Board("0000")))
    assert not in_mock.pfb_checkbox.isChecked()
    in_mock.accept()

    # and the board's choice is still there when a board is next used
    again = build(lambda: NoiseSpectrumDialog(
        num_resonances=3, crs=_Board("0024")))
    assert again.pfb_checkbox.isChecked()
