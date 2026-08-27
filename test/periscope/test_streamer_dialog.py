"""
Offscreen tests for the Streamer Configuration dialog.

The dialog is a thin view over streamer_config.describe()/validate();
these tests check the view contract: live labels, forced short packets
below stage 3, OK disabled while errors exist, and config round-trip.
"""

import os

import pytest


pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.streamer_config_dialog import (  # noqa: E402
    StreamerConfigDialog,
)



def _ok_button(dialog):
    return dialog.buttons.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok)


def test_defaults_and_labels(qt_app):
    dlg = StreamerConfigDialog(current_dec=6, current_short=False, module=1)
    assert "596" in dlg.rate_label.text()
    assert dlg.width_label.text() == "1024"
    assert _ok_button(dlg).isEnabled()
    assert "dec 6" in dlg.current_label.text()
    dlg.close()


def test_short_forced_below_stage3(qt_app):
    dlg = StreamerConfigDialog(current_dec=6, current_short=False, module=1)
    dlg.dec_spin.setValue(2)
    assert dlg.short_check.isChecked()
    assert not dlg.short_check.isEnabled()
    assert dlg.width_label.text() == "128"
    assert _ok_button(dlg).isEnabled()

    dlg.dec_spin.setValue(4)
    assert dlg.short_check.isEnabled()
    dlg.close()


def test_over_budget_disables_ok(qt_app):
    dlg = StreamerConfigDialog(current_dec=6, current_short=False, module=1)
    dlg.dec_spin.setValue(3)
    dlg.short_check.setChecked(False)
    dlg.modules_edit.setText("1,2,3,4")  # long dec3 x4 > 1 Gbps
    assert not _ok_button(dlg).isEnabled()
    assert "Mbps" in dlg.status_label.text()

    dlg.modules_edit.setText("1")
    assert _ok_button(dlg).isEnabled()
    dlg.close()


def test_pfb_section_and_roundtrip(qt_app):
    dlg = StreamerConfigDialog(current_dec=6, current_short=False, module=2)
    assert not dlg.pfb_group.isVisible()
    dlg.pfb_check.setChecked(True)
    dlg.pfb_channels_edit.setText("1,2,3")
    dlg.pfb_module_spin.setValue(2)

    cfg = dlg.get_config()
    assert cfg.dec_stage == 6
    assert cfg.pfb_channels == [1, 2, 3]
    assert cfg.pfb_module == 2
    assert cfg.modules == [2]
    assert "PFB" in dlg.bandwidth_label.text()

    dlg.pfb_channels_edit.setText("1,2,3,4,5")  # over the 4-channel limit
    assert not _ok_button(dlg).isEnabled()
    dlg.close()


def test_bad_modules_text_is_error_not_crash(qt_app):
    dlg = StreamerConfigDialog(current_dec=6, current_short=False, module=1)
    dlg.modules_edit.setText("1,x")
    assert not _ok_button(dlg).isEnabled()
    dlg.modules_edit.setText("1")
    assert _ok_button(dlg).isEnabled()
    dlg.close()
