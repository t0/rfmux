"""Offscreen tests for the rfmux record dialog."""

import inspect
import pickle
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore  # noqa: E402

from rfmux.tools import record_dialog as rd  # noqa: E402
from rfmux.tools.record import _run  # noqa: E402


def _dialog(tmp_path, monkeypatch, running=()):
    monkeypatch.setattr(rd, "host_interfaces",
                        lambda: ["eth0", "enp2s0f0np0"])
    fake = SimpleNamespace(running_interfaces=lambda: list(running),
                           start_command=lambda i: f"sudo fastrxd -i {i}")
    monkeypatch.setattr(rd, "_fastrx", lambda: fake)
    settings = QtCore.QSettings(str(tmp_path / "record.ini"),
                                QtCore.QSettings.Format.IniFormat)
    return rd.RecordDialog(settings=settings), settings


def _session_with_bias(tmp_path, module=2, channels=(3, 7)):
    folder = tmp_path / "session_x"
    folder.mkdir()
    export = {"target_module": module, "timestamp": "2026",
              "bias_kids_output": {
                  i: {"bias_channel": c, "df_calibration": 1 + 1j}
                  for i, c in enumerate(channels)}}
    (folder / f"bias_module{module}_1.pkl").write_bytes(pickle.dumps(export))
    return folder


def test_options_are_the_runners_arguments(qt_app, tmp_path, monkeypatch):
    dlg, _ = _dialog(tmp_path, monkeypatch, running=["enp2s0f0np0"])
    dlg.serial_edit.setText("0156")
    dlg.module_spin.setValue(2)
    folder = _session_with_bias(tmp_path)
    dlg.rb_existing.setChecked(True)
    dlg.session_path_edit.setText(str(folder))
    dlg.rb_bias.setChecked(True)
    dlg.duration_spin.setValue(0.5)
    dlg.fastrx_iface_combo.setEditText("enp2s0f0np0")
    o = dlg.get_options()
    assert set(o) == set(inspect.signature(_run).parameters) - {"quiet"}
    assert (o["serial"], o["module"], o["channels"], o["session"]) == \
        ("0156", 2, None, str(folder))
    assert "bias_module2_1.pkl: 2 channels, 2 calibrated" in \
        dlg.bias_label.text()
    assert dlg.record_btn.isEnabled(), dlg.status_label.text()


def test_record_waits_for_fastrxd_and_shows_the_start_command(
        qt_app, tmp_path, monkeypatch):
    dlg, _ = _dialog(tmp_path, monkeypatch, running=[])
    dlg.serial_edit.setText("0156")
    dlg.rb_ranges.setChecked(True)
    dlg.channels_edit.setText("1-4")
    dlg.fastrx_check.setChecked(True)
    dlg.fastrx_iface_combo.setEditText("enp2s0f0np0")
    assert not dlg.record_btn.isEnabled()
    assert "sudo fastrxd -i enp2s0f0np0" in dlg.fastrx_status.text()
    dlg.fastrx_check.setChecked(False)
    assert dlg.record_btn.isEnabled(), dlg.status_label.text()


def test_the_dialog_remembers_its_values(qt_app, tmp_path, monkeypatch):
    dlg, settings = _dialog(tmp_path, monkeypatch)
    dlg.serial_edit.setText("0042")
    dlg.rb_ranges.setChecked(True)
    dlg.channels_edit.setText("5-9")
    dlg.duration_spin.setValue(7.5)
    dlg.show_combo.setCurrentIndex(1)
    dlg.capture_form.threshold_spin.setValue(6.5)
    dlg._save()
    o = rd.RecordDialog(settings=settings).get_options()
    assert (o["serial"], o["channels"], o["duration"], o["show"]) == \
        ("0042", "5-9", 7.5, "overlay")
    assert o["config"].threshold_sigma == 6.5
