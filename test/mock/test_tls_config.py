"""
TLS 1/f config plumbing: defaults, validation, and the dialog round-trip.

Every mock config key must survive nine hand-written touch points
(default, coercion, clamping, model seed, generate_resonators, dialog
widget, load, reset, get_configuration).  These tests walk that path.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rfmux.mock import config as mock_config  # noqa: E402

TLS_KEYS = ("tls_noise_enabled", "tls_fractional_rms", "tls_alpha",
            "tls_corner_hz")


class TestDefaults:
    def test_keys_present_and_disabled_by_default(self):
        cfg = mock_config.defaults()
        for key in TLS_KEYS:
            assert key in cfg, f"{key} missing from MOCK_DEFAULTS"
        assert cfg["tls_noise_enabled"] is False, \
            "TLS must default off so existing behaviour is unchanged"

    def test_string_values_are_coerced(self):
        cfg = mock_config.apply_overrides({
            "tls_fractional_rms": "2.5e-7",
            "tls_alpha": "0.5",
            "tls_corner_hz": "50",
        })
        assert cfg["tls_fractional_rms"] == pytest.approx(2.5e-7)
        assert cfg["tls_alpha"] == pytest.approx(0.5)
        assert cfg["tls_corner_hz"] == pytest.approx(50.0)

    def test_clamping(self):
        cfg = mock_config.apply_overrides({
            "tls_fractional_rms": -1.0,   # negative RMS is meaningless
            "tls_alpha": 9.0,             # power-law fit breaks down
            "tls_corner_hz": 1e12,        # would make the grid absurd
        })
        assert cfg["tls_fractional_rms"] == 0.0
        assert cfg["tls_alpha"] <= 2.0
        assert cfg["tls_corner_hz"] <= 1e5

        cfg = mock_config.apply_overrides({"tls_corner_hz": 0.0})
        assert cfg["tls_corner_hz"] >= 1e-3

    def test_garbage_falls_back(self):
        cfg = mock_config.apply_overrides({"tls_alpha": "not-a-number"})
        assert cfg["tls_alpha"] == pytest.approx(1.0)


class TestDialog:
    # staticmethod, not an instance method: pytest 10 drops support for
    # class-scoped fixtures defined as instance methods, because the fixture runs
    # once per class while each test gets a fresh instance — so anything it set
    # on self would be invisible to the tests. This one only yields, so the
    # behaviour was already correct; this just silences the deprecation without
    # pretending self was ever used.
    @pytest.fixture(scope="class")
    @staticmethod
    def qt_app():
        pytest.importorskip("PyQt6")
        from PyQt6 import QtWidgets
        yield QtWidgets.QApplication.instance() \
            or QtWidgets.QApplication([])

    def test_round_trip(self, qt_app):
        from rfmux.tools.periscope.mock_configuration_dialog import (
            MockConfigurationDialog,
        )
        dlg = MockConfigurationDialog(None, mock_config.defaults())
        dlg.tls_noise_enabled_cb.setChecked(True)
        dlg.tls_rms_edit.setText("3e-7")
        dlg.tls_alpha_edit.setText("0.7")
        dlg.tls_corner_edit.setText("25")

        cfg = dlg.get_configuration()
        assert cfg["tls_noise_enabled"] is True
        assert cfg["tls_fractional_rms"] == pytest.approx(3e-7)
        assert cfg["tls_alpha"] == pytest.approx(0.7)
        assert cfg["tls_corner_hz"] == pytest.approx(25.0)
        dlg.close()

    def test_reset_to_defaults_does_not_keyerror(self, qt_app):
        """_reset_to_defaults indexes cfg[...] directly, so a key that
        is missing from defaults raises — guard against that."""
        from rfmux.tools.periscope.mock_configuration_dialog import (
            MockConfigurationDialog,
        )
        dlg = MockConfigurationDialog(None, mock_config.defaults())
        dlg.tls_noise_enabled_cb.setChecked(True)
        dlg._reset_to_defaults()
        assert dlg.tls_noise_enabled_cb.isChecked() is False
        dlg.close()

    def test_load_current_values(self, qt_app):
        from rfmux.tools.periscope.mock_configuration_dialog import (
            MockConfigurationDialog,
        )
        cfg = mock_config.defaults()
        cfg.update({"tls_noise_enabled": True, "tls_corner_hz": 12.5})
        dlg = MockConfigurationDialog(None, cfg)
        assert dlg.tls_noise_enabled_cb.isChecked() is True
        assert float(dlg.tls_corner_edit.text()) == pytest.approx(12.5)
        dlg.close()
