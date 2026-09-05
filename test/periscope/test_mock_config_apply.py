"""Applying a Mock Configuration keeps Periscope's record of the mock in
step with the server.

The dialog's configuration never carries a pulse mode (the QP Pulses
toggle records that), the apply pins a random seed into the dialog's
dict, and a regenerated array invalidates the df calibrations measured
on the one it replaced.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.mock import config as mc  # noqa: E402
from rfmux.tools.periscope.app import Periscope  # noqa: E402


class _Crs:
    """Records what the server was told, the way MockCRS would take it."""

    def __init__(self):
        self.generated = []
        self.pulse_modes = []

    async def generate_resonators(self, config):
        self.generated.append(dict(config))
        return 7, None

    async def set_pulse_mode(self, mode, **kwargs):
        self.pulse_modes.append(mode)


def _periscope(crs, mock_config, pulse_mode):
    """Just enough Periscope for _apply_mock_configuration: no receiver,
    no CRS connection, no startup dialog."""
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.crs = crs
    p.module = 1
    p.mock_config = mock_config
    p.qp_pulse_mode = pulse_mode
    p.btn_qp_pulses = QtWidgets.QPushButton()
    p.df_calibrations = {1: {1: 1.0 + 1.0j}}
    p.saved = []
    p.session_manager = SimpleNamespace(is_active=True,
                                        save_mock_config=p.saved.append)
    p.calibrations_started = []
    p._start_df_calibration = p.calibrations_started.append
    return p


def _pulsing_periodic():
    """The configuration in force after the toggle turned pulses on."""
    return mc.apply_overrides({"pulse_mode": "periodic"})


def _dialog_edit(previous, **changes):
    """What the dialog returns: every key but pulse_mode, with *changes*."""
    edit = {k: v for k, v in previous.items() if k != "pulse_mode"}
    edit.update(changes)
    return edit


@pytest.fixture
def pulse_only_edit(qt_app):
    previous = _pulsing_periodic()
    p = _periscope(_Crs(), previous, "periodic")
    p._apply_mock_configuration(_dialog_edit(previous, pulse_tau_decay=1e-3),
                                previous)
    return p


def test_pulse_only_edit_keeps_the_mode(pulse_only_edit):
    p = pulse_only_edit
    assert p.qp_pulse_mode == "periodic"
    assert p.btn_qp_pulses.text() == "QP Pulses: Periodic"


def test_pulse_only_edit_keeps_the_mode_in_mock_config(pulse_only_edit):
    assert pulse_only_edit.mock_config["pulse_mode"] == "periodic"


def test_pulse_only_edit_saves_the_mode_with_the_session(pulse_only_edit):
    assert pulse_only_edit.saved[-1]["pulse_mode"] == "periodic"


def test_second_pulse_only_edit_still_pulses(pulse_only_edit):
    """The reported failure: the second edit turned the pulses off."""
    p = pulse_only_edit
    p._apply_mock_configuration(_dialog_edit(p.mock_config, pulse_tau_decay=2e-3),
                                p.mock_config)
    assert p.crs.pulse_modes[-1] == "periodic"
    assert p.qp_pulse_mode == "periodic"


@pytest.fixture
def regenerated(qt_app):
    previous = mc.apply_overrides({"pulse_mode": "periodic",
                                   "auto_bias_kids": True})
    p = _periscope(_Crs(), previous, "periodic")
    p._apply_mock_configuration(
        _dialog_edit(previous, num_resonances=previous["num_resonances"] + 1,
                     resonator_random_seed=None),
        previous)
    return p


def test_regeneration_rebuilds_with_the_mode_in_force(regenerated):
    """The rebuild sets the model's mode itself; no second call."""
    p = regenerated
    assert p.crs.generated[-1]["pulse_mode"] == "periodic"
    assert p.crs.pulse_modes == []
    assert p.qp_pulse_mode == "periodic"


def test_regeneration_records_the_seed_the_server_built_with(regenerated):
    p = regenerated
    seed = p.crs.generated[-1]["resonator_random_seed"]
    assert seed is not None
    assert p.mock_config["resonator_random_seed"] == seed


def test_regeneration_drops_the_previous_array_calibrations(regenerated):
    assert 1 not in regenerated.df_calibrations


def test_regeneration_measures_again_when_auto_bias_kids(regenerated):
    assert regenerated.calibrations_started == [1]


def test_regeneration_without_auto_bias_kids_measures_nothing(qt_app):
    previous = mc.apply_overrides({"auto_bias_kids": False})
    p = _periscope(_Crs(), previous, "none")
    p._apply_mock_configuration(
        _dialog_edit(previous, num_resonances=previous["num_resonances"] + 1),
        previous)
    assert 1 not in p.df_calibrations
    assert p.calibrations_started == []


def test_failed_apply_keeps_the_configuration_in_force(qt_app, monkeypatch):
    class _Broken(_Crs):
        async def generate_resonators(self, config):
            raise RuntimeError("no")
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        lambda *a, **k: None)
    previous = _pulsing_periodic()
    p = _periscope(_Broken(), previous, "periodic")
    p._apply_mock_configuration(
        _dialog_edit(previous, num_resonances=previous["num_resonances"] + 1),
        previous)
    assert p.mock_config is previous
    assert p.df_calibrations == {1: {1: 1.0 + 1.0j}}
