"""Offscreen tests for the Pulse Capture Settings dialog."""


import pytest


pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.pulse_capture.capture_session import (  # noqa: E402
    PulseCaptureConfig,
)
from rfmux.tools.periscope.pulse_capture_settings_dialog import (  # noqa: E402
    PulseCaptureSettingsDialog,
)


def _plain(label) -> str:
    """A derived label's text with its table markup stripped."""
    import re
    return " ".join(re.sub(r"<[^>]+>", " ", label.text()).split())



def _ok(dlg):
    return dlg.buttons.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok)


def test_derived_labels_follow_rate(qt_app):
    cfg = PulseCaptureConfig(min_pulse_ms=1.0)
    slow = PulseCaptureSettingsDialog(
        config=cfg, sample_rate=19073.486328125, mode="slow")
    assert "min pulse 19 samples" in _plain(slow.pulse_derived_label)
    slow.close()

    fast = PulseCaptureSettingsDialog(
        config=cfg, sample_rate=1220703.125, mode="fast")
    assert "min pulse 1,221 samples" in _plain(fast.pulse_derived_label) or \
        "min pulse 1221 samples" in _plain(fast.pulse_derived_label)
    fast.close()


def test_ok_gated_on_errors(qt_app):
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    assert _ok(dlg).isEnabled()
    dlg.end_spin.setValue(dlg.threshold_spin.value())  # end >= threshold
    assert not _ok(dlg).isEnabled()
    dlg.end_spin.setValue(1.5)
    assert _ok(dlg).isEnabled()
    dlg.close()


def test_roundtrip(qt_app):
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    dlg.threshold_spin.setValue(8.0)
    dlg.min_pulse_spin.setValue(0.5)
    dlg.max_pulse_spin.setValue(100.0)
    dlg.pre_pulse_spin.setValue(2.5)
    dlg.post_pulse_spin.setValue(12.0)
    dlg.pileup_check.setChecked(False)

    cfg = dlg.get_config()
    assert cfg.threshold_sigma == 8.0
    assert cfg.min_pulse_ms == 0.5
    assert cfg.max_pulse_ms == 100.0
    # The 1/f window is its own control, untouched by the pulse length.
    assert cfg.noise_train_ms == 5000.0
    assert cfg.noise_train_span_ms() == 5000.0
    assert "samples" in dlg.noise_label.text()
    assert (cfg.pre_pulse_ms, cfg.post_pulse_ms) == (2.5, 12.0)
    assert cfg.enable_pileup is False
    dlg.close()


def test_event_settings_round_trip_and_size_the_ring(qt_app):
    """The coincidence window and the dump reach the config, and the
    ring the dialog reports grows by what a dump needs."""
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    cfg = dlg.get_config()
    assert (cfg.coincidence_window_ms, cfg.dump_all_channels) == (0.0, False)
    assert dlg.coincidence_spin.text() == "off"
    before = cfg.buf_size(19073.486328125)
    dlg.coincidence_spin.setValue(2.5)
    dlg.dump_check.setChecked(True)
    cfg = dlg.get_config()
    assert (cfg.coincidence_window_ms, cfg.dump_all_channels) == (2.5, True)
    assert cfg.buf_size(19073.486328125) > before
    assert f"{cfg.buf_size(19073.486328125):,} samples" in \
        _plain(dlg.pulse_derived_label)
    dlg.close()


def test_noise_sampling_round_trips(qt_app):
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    assert dlg.noise_capture_spin.text() == "off"
    assert dlg.get_config().noise_capture_interval_s == 0.0
    dlg.noise_capture_spin.setValue(30.0)
    assert dlg.get_config().noise_capture_interval_s == 30.0
    back = PulseCaptureSettingsDialog(config=dlg.get_config(),
                                      sample_rate=19073.486328125)
    assert back.noise_capture_spin.value() == 30.0
    assert "normally distributed" in dlg.noise_capture_spin.toolTip()
    dlg.close()
    back.close()


def test_rolling_baseline_span_is_shown(qt_app):
    """No baseline controls left to get wrong — the window is the
    training span, so the dialog only reports it."""
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    assert not hasattr(dlg, "baseline_spin")
    assert not hasattr(dlg, "baseline_auto_check")
    assert "baseline median" in _plain(dlg.pulse_derived_label)
    dlg.close()


def test_derived_readouts_split_by_driving_knob(qt_app):
    """The dialog shows WHAT each primary input drives: every time
    scale under max pulse, everything statistical under threshold σ —
    including the two new derived quantities (edge lookback, hard
    stop) and the edge amplitude floor."""
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    pulse_txt = dlg.pulse_derived_label.text()
    sigma_txt = dlg.sigma_derived_label.text()
    for piece in ("ring buffer", "hard stop", "noise training",
                  "baseline median", "edge lookback"):
        assert piece in pulse_txt, piece
    for piece in ("confirmation", "accidentals", "edge jump",
                  "amplitude floor"):
        assert piece in sigma_txt, piece
    # The floor readout follows the threshold: 5σ → ≈7.1σ.
    assert "7.1σ" in sigma_txt
    dlg.threshold_spin.setValue(10.0)
    assert "14.1σ" in dlg.sigma_derived_label.text()

    # And the time scales follow max pulse: 50 ms → 60 ms hard stop
    # plus the 5 ms post-pulse time, 5 ms edge lookback (95 samples at
    # 19 kHz reads 4.98 ms).
    assert "65 ms" in pulse_txt
    assert "4.98 ms" in pulse_txt
    dlg.max_pulse_spin.setValue(500.0)
    assert "605 ms" in _plain(dlg.pulse_derived_label)
    assert "50 ms" in _plain(dlg.pulse_derived_label)
    dlg.close()

def test_max_pulse_and_the_window_are_separate_primary_controls(qt_app):
    """Max pulse and the 1/f window both sit in the main form; the
    window keeps its seconds whatever the pulse length, and the readout
    follows the window."""
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    # Not hidden behind Advanced.
    assert not dlg.adv_box.isChecked()
    assert dlg.max_pulse_spin.isVisible() or not dlg.isVisible()

    dlg.max_pulse_spin.setValue(10.0)
    assert dlg.get_config().noise_train_span_ms() == 5000.0
    first = dlg.noise_label.text()
    dlg.window_spin.setValue(8000.0)
    assert dlg.get_config().noise_train_span_ms() == 8000.0
    assert dlg.noise_label.text() != first, "readout did not follow"
    dlg.window_spin.setValue(0.0)            # derived from the pulse again
    assert dlg.get_config().noise_train_span_ms() == 200.0
    dlg.close()


def test_trigger_basis_round_trips(qt_app):
    """The basis survives the dialog, and does not disturb the rest.

    It is the one capture setting that changes what gets detected rather
    than how much of it is kept, so a dialog that silently reset it to
    the default would be worse than not offering it.
    """
    from rfmux.pulse_capture import PulseCaptureConfig

    for basis in ("iq", "df"):
        src = PulseCaptureConfig(trigger_basis=basis, threshold_sigma=7.5,
                                 max_pulse_ms=123.0, enable_pileup=False)
        out = PulseCaptureSettingsDialog(
            config=src, sample_rate=596.0).get_config()
        assert out.trigger_basis == basis
        # The other knobs come back untouched.
        assert out.threshold_sigma == pytest.approx(7.5)
        assert out.max_pulse_ms == pytest.approx(123.0)
        assert out.enable_pileup is False


def test_end_floor_is_exposed_and_round_trips(qt_app):
    """The end-confirmation floor sits under Advanced with the rest of
    the end logic, reads back into the config, and the derived readout
    says what it is in time at this rate."""
    dlg = PulseCaptureSettingsDialog(sample_rate=596.0)
    assert dlg.min_end_spin.value() == 10
    assert "end floor 10 samples" in _plain(dlg.pulse_derived_label)
    assert "16.8 ms" in _plain(dlg.pulse_derived_label)
    dlg.min_end_spin.setValue(4)
    assert dlg.get_config().min_end_samples == 4
    assert "end floor 4 samples" in _plain(dlg.pulse_derived_label)
    assert "bucket" in dlg.min_end_spin.toolTip()
    dlg.close()


def test_df_basis_needs_a_calibration(qt_app):
    """Without a df calibration the rotated basis cannot be chosen."""
    dlg = PulseCaptureSettingsDialog(
        config=PulseCaptureConfig(trigger_basis="iq"), sample_rate=596.0,
        df_available=False)
    assert dlg.basis_combo.currentIndex() == 0
    assert not dlg.basis_combo.model().item(1).isEnabled()
    assert "calibration" in dlg.basis_combo.model().item(1).toolTip()
    assert dlg.get_config().trigger_basis == "iq"
    dlg.close()

    dlg = PulseCaptureSettingsDialog(
        config=PulseCaptureConfig(trigger_basis="df"), sample_rate=596.0)
    assert dlg.basis_combo.currentIndex() == 1
    assert dlg.basis_combo.model().item(1).isEnabled()
    assert dlg.get_config().trigger_basis == "df"
    dlg.close()


def test_stored_df_basis_survives_the_dialog_without_a_calibration(qt_app):
    """The default basis is df; a round trip through the dialog before
    any channel is calibrated must not rewrite it to iq, or the panel's
    later captures would store volts where headless stores hertz.  The
    combo shows what get_config returns, and the user can still drop to
    the quadratures."""
    dlg = PulseCaptureSettingsDialog(
        config=PulseCaptureConfig(trigger_basis="df"), sample_rate=596.0,
        df_available=False)
    assert dlg.basis_combo.currentIndex() == 1
    assert dlg.get_config().trigger_basis == "df"
    dlg.basis_combo.setCurrentIndex(0)
    assert dlg.get_config().trigger_basis == "iq"
    dlg.close()


def test_saved_margins_are_shown_in_samples_at_this_rate(qt_app):
    """The times are the setting; what they come to at this stream's
    rate is derived, and follows the spin boxes."""
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    assert "95 samples before the trigger, 95 after it settled" in \
        _plain(dlg.pulse_derived_label)
    dlg.post_pulse_spin.setValue(0.0)
    assert "95 samples before the trigger, 0 after it settled" in \
        _plain(dlg.pulse_derived_label)
    dlg.close()


def test_noise_training_row_shows_the_length_actually_used(qt_app):
    """On the PFB stream the training record is memory-capped, and the
    row must agree with the derived table rather than quote the uncapped
    ratio as fact."""
    dlg = PulseCaptureSettingsDialog(sample_rate=2441406.25, mode="fast")
    d = dlg.get_config().describe(dlg.sample_rate, dlg.n_channels)
    assert d["noise_train_actual_ms"] < d["noise_train_span_ms"]
    label = dlg.noise_label.text()
    assert label.startswith("819 ms")
    assert "capped" in label
    assert "noise training 819 ms" in _plain(dlg.pulse_derived_label)
    dlg.close()
