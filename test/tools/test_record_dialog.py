"""Offscreen tests for the rfmux record dialog."""

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore  # noqa: E402

from rfmux.tools import record_dialog as rd  # noqa: E402
from rfmux.tools.record import _run  # noqa: E402
from test.record_helpers import bias_export  # noqa: E402


def _dialog(tmp_path, monkeypatch, running=()):
    monkeypatch.setattr(rd, "interface_speeds",
                        lambda: {"eth0": 1000, "enp2s0f0np0": 100000,
                                 "wlan0": None, "enp1s0f0": None})
    monkeypatch.setattr(rd, "_operstate",
                        {"wlan0": "up", "enp1s0f0": "down"}.get)
    monkeypatch.setattr(rd, "running_mock", lambda: None)
    fake = SimpleNamespace(running_interfaces=lambda: list(running),
                           start_command=lambda i: f"sudo fastrxd -i {i}",
                           record_stride=lambda c: (86 + 4 * c + 7) & ~7)
    monkeypatch.setattr(rd, "_fastrx", lambda: fake)
    settings = QtCore.QSettings(str(tmp_path / "record.ini"),
                                QtCore.QSettings.Format.IniFormat)
    return rd.RecordDialog(settings=settings), settings


def _session_with_bias(tmp_path, module=2, channels=(3, 7)):
    folder = tmp_path / "session_x"
    folder.mkdir(exist_ok=True)
    bias_export(folder / f"bias_module{module}_1.pkl", module, channels)
    return folder


def test_options_are_the_runners_arguments(qt_app, tmp_path, monkeypatch):
    dlg, _ = _dialog(tmp_path, monkeypatch, running=["enp2s0f0np0"])
    dlg.parser_iface_combo.setCurrentIndex(0)
    dlg.serial_edit.setText("0156")
    dlg.modules_edit.setText("2")
    folder = _session_with_bias(tmp_path)
    dlg.rb_existing.setChecked(True)
    dlg.session_path_edit.setText(str(folder))
    dlg.rb_bias.setChecked(True)
    dlg.duration_spin.setValue(0.5)
    dlg.fastrx_iface_combo.setEditText("enp2s0f0np0")
    o = dlg.get_options()
    assert set(o) == set(inspect.signature(_run).parameters) - {"quiet"}
    assert (o["serial"], o["modules"], o["channels"], o["session"]) == \
        ("0156", [2], None, str(folder))
    assert "bias_module2_1.pkl: 2 channels, 2 calibrated" in \
        dlg.bias_label.text()
    assert dlg.record_btn.isEnabled(), dlg.status_label.text()


def test_record_waits_for_fastrxd_and_shows_the_start_command(
        qt_app, tmp_path, monkeypatch):
    dlg, _ = _dialog(tmp_path, monkeypatch, running=[])
    dlg.parser_iface_combo.setCurrentIndex(0)
    dlg.serial_edit.setText("0156")
    dlg.rb_ranges.setChecked(True)
    dlg.channels_edit.setText("1-4")
    dlg.fastrx_check.setChecked(True)
    dlg.fastrx_iface_combo.setEditText("enp2s0f0np0")
    assert not dlg.record_btn.isEnabled()
    assert "sudo fastrxd -i enp2s0f0np0" in dlg.fastrx_status.text()
    dlg.fastrx_check.setChecked(False)
    assert dlg.record_btn.isEnabled(), dlg.status_label.text()


def test_the_channel_streamer_is_off_unless_asked_and_remembered(
        qt_app, tmp_path, monkeypatch):
    dlg, settings = _dialog(tmp_path, monkeypatch)
    o = dlg.get_options()
    assert (o["channel_streamer"], o["sample_trunc"]) == (False, "LOW")
    assert "±32767" in dlg.trunc_combo.toolTip()
    dlg.streamer_check.setChecked(True)
    dlg.trunc_combo.setCurrentIndex(2)
    dlg._save()
    o = rd.RecordDialog(settings=settings).get_options()
    assert (o["channel_streamer"], o["sample_trunc"]) == (True, "HIGH")
    # The choice belongs to the fastrx product.
    dlg.fastrx_check.setChecked(False)
    dlg._refresh()
    assert not dlg.streamer_check.isEnabled()


def test_the_units_choice_is_the_captures_trigger_basis(
        qt_app, tmp_path, monkeypatch):
    """One setting seen twice: the pulse file and the TOD are written
    in the same units whichever view changed it."""
    dlg, settings = _dialog(tmp_path, monkeypatch)
    assert dlg.units_combo.currentIndex() == dlg.capture_form.basis_combo.currentIndex()
    dlg.units_combo.setCurrentIndex(0)
    assert dlg.get_options()["config"].trigger_basis == "iq"
    dlg.capture_form.basis_combo.setCurrentIndex(1)
    assert dlg.units_combo.currentData() == "df"
    assert dlg.get_options()["config"].trigger_basis == "df"
    dlg._save()
    again = rd.RecordDialog(settings=settings)
    assert again.units_combo.currentData() == "df"


def test_the_tod_needs_a_stream_and_the_merge_a_pulse_file(
        qt_app, tmp_path, monkeypatch):
    dlg, _ = _dialog(tmp_path, monkeypatch)
    dlg.parser_iface_combo.setCurrentIndex(0)
    dlg.serial_edit.setText("0156")
    dlg.rb_ranges.setChecked(True)
    dlg.channels_edit.setText("1-4")
    dlg.fastrx_check.setChecked(False)
    dlg.parser_check.setChecked(True)
    dlg.tod_check.setChecked(True)
    dlg.merge_tod_check.setChecked(True)
    assert dlg.tod_check.isEnabled() and dlg.merge_tod_check.isEnabled()
    assert "fifth of real time" in dlg.tod_note.text()
    assert "grows by the whole TOD" in dlg.merge_tod_note.text()
    # Without a pulse capture the TOD is still written; there is
    # nothing to merge it into.
    dlg.capture_check.setChecked(False)
    assert dlg.tod_check.isEnabled() and not dlg.merge_tod_check.isEnabled()
    assert dlg.record_btn.isEnabled(), dlg.status_label.text()
    dlg.parser_check.setChecked(False)
    assert not dlg.tod_check.isEnabled()


def test_a_tod_without_a_bias_export_warns_but_records(
        qt_app, tmp_path, monkeypatch):
    dlg, _ = _dialog(tmp_path, monkeypatch)
    dlg.parser_iface_combo.setCurrentIndex(0)
    dlg.serial_edit.setText("0156")
    dlg.fastrx_check.setChecked(False)
    dlg.parser_check.setChecked(True)
    dlg.capture_check.setChecked(False)
    dlg.tod_check.setChecked(True)
    dlg.rb_ranges.setChecked(True)
    dlg.channels_edit.setText("2:1-4")
    assert "warning: the TOD metadata" in dlg.status_label.text()
    assert dlg.record_btn.isEnabled()
    # A session whose bias export covers the module supplies the tuning.
    folder = _session_with_bias(tmp_path)
    dlg.rb_existing.setChecked(True)
    dlg.session_path_edit.setText(str(folder))
    assert "warning" not in dlg.status_label.text(), dlg.status_label.text()
    dlg.tod_check.setChecked(False)
    dlg.rb_new.setChecked(True)
    assert "warning" not in dlg.status_label.text()


def test_the_running_mock_fills_the_hostname_for_its_serial(
        qt_app, tmp_path, monkeypatch):
    """Serial 0000 with a mock server up fills the hostname in; another
    serial takes the autofill away again, while an address the user
    typed stays."""
    dlg, settings = _dialog(tmp_path, monkeypatch)
    monkeypatch.setattr(rd, "running_mock", lambda: "127.0.0.1:9878")
    dlg.serial_edit.setText("MOCK")
    assert dlg.get_options()["hostname"] == "127.0.0.1:9878"
    dlg.serial_edit.setText("0156")
    assert dlg.get_options()["hostname"] is None
    dlg.serial_edit.setText("0000")
    assert dlg.get_options()["hostname"] == "127.0.0.1:9878"
    dlg.serial_edit.setText("0156")
    assert dlg.get_options()["hostname"] is None
    dlg.hostname_edit.setText("rfmux0156.lan")
    dlg.serial_edit.setText("0000")
    assert dlg.get_options()["hostname"] == "rfmux0156.lan"
    dlg._save()
    assert rd.RecordDialog(settings=settings).get_options()["hostname"] == \
        "rfmux0156.lan"


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


def test_interfaces_show_their_rates_and_sort_by_role(
        qt_app, tmp_path, monkeypatch):
    """The parser chooses among every interface, the 100 Gb/s one
    included, fastrx among the 100 Gb/s ones, each labelled with its
    rate; the one 100 Gb/s interface is filled in when nothing was
    chosen."""
    dlg, _ = _dialog(tmp_path, monkeypatch)
    parser = [dlg.parser_iface_combo.itemText(i)
              for i in range(dlg.parser_iface_combo.count())]
    # A wifi driver reports no rate; a port without a link is down; the
    # loopback is where a mock on this host streams.
    assert parser == ["eth0 (1 Gb/s)", "enp2s0f0np0 (100 Gb/s)",
                      "wlan0 (no rate reported)", "enp1s0f0 (down)",
                      "lo (loopback: a mock on this host)"]
    fast = [dlg.fastrx_iface_combo.itemText(i)
            for i in range(dlg.fastrx_iface_combo.count())]
    assert fast == ["enp2s0f0np0 (100 Gb/s)"]
    o = dlg.get_options()
    assert o["fastrx_interface"] == "enp2s0f0np0"
    assert o["parser_interface"] is None
    dlg.parser_iface_combo.setCurrentIndex(0)
    assert dlg.get_options()["parser_interface"] == "eth0"


def test_the_parser_interface_has_to_be_chosen(qt_app, tmp_path, monkeypatch):
    """Nothing is chosen for the user, a saved "auto" included, and
    Record waits for the choice."""
    _, settings = _dialog(tmp_path, monkeypatch)
    settings.setValue("parser_interface", "auto")
    dlg = rd.RecordDialog(settings=settings)
    dlg.serial_edit.setText("0156")
    dlg.rb_ranges.setChecked(True)
    dlg.channels_edit.setText("1-4")
    dlg.fastrx_check.setChecked(False)
    assert dlg.parser_iface_combo.currentText() == ""
    assert not dlg.record_btn.isEnabled()
    assert "interface the parser listens on" in dlg.status_label.text()
    dlg.parser_iface_combo.setCurrentIndex(1)
    assert dlg.record_btn.isEnabled(), dlg.status_label.text()
    assert dlg.get_options()["parser_interface"] == "enp2s0f0np0"
    dlg.parser_iface_combo.setEditText("")
    dlg.parser_check.setChecked(False)
    assert dlg.record_btn.isEnabled(), dlg.status_label.text()


def test_the_session_fills_in_with_the_newest_under_the_default_path(
        qt_app, tmp_path, monkeypatch):
    base = tmp_path / "outputs"
    for name in ("session_20260901_090000", "session_20260910_154331",
                 "session_20260905_120000"):
        (base / name).mkdir(parents=True)
        (base / name / "session_metadata.json").write_text("{}")
    (base / "session_20260911_000000").mkdir()      # no metadata: not one
    settings = QtCore.QSettings(str(tmp_path / "record.ini"),
                                QtCore.QSettings.Format.IniFormat)
    settings.setValue("record/session_dir", str(base))
    monkeypatch.setattr(rd, "interface_speeds", lambda: {})
    dlg = rd.RecordDialog(settings=settings)
    assert dlg.session_path_edit.text() == str(base / "session_20260910_154331")
    assert dlg.rb_existing.isChecked() and dlg.rb_bias.isChecked()


def test_several_modules_take_an_export_each_or_per_module_ranges(
        qt_app, tmp_path, monkeypatch):
    dlg, _ = _dialog(tmp_path, monkeypatch)
    dlg.parser_iface_combo.setCurrentIndex(0)
    dlg.serial_edit.setText("0156")
    folder = _session_with_bias(tmp_path, module=2, channels=(3, 7))
    _session_with_bias(tmp_path, module=3, channels=(1,))
    dlg.rb_existing.setChecked(True)
    dlg.session_path_edit.setText(str(folder))
    dlg.fastrx_check.setChecked(False)
    dlg.rb_bias.setChecked(True)
    dlg.modules_edit.setText("2,3")
    assert dlg.get_options()["modules"] == [2, 3]
    assert dlg.bias_label.text().splitlines() == [
        "bias_module2_1.pkl: 2 channels, 2 calibrated",
        "bias_module3_1.pkl: 1 channels, 1 calibrated"]
    assert dlg.record_btn.isEnabled(), dlg.status_label.text()
    dlg.modules_edit.setText("2,4")
    assert not dlg.record_btn.isEnabled()
    assert "no bias export for module 4" in dlg.status_label.text()
    dlg.rb_ranges.setChecked(True)
    dlg.channels_edit.setText("2:1-4,3:1-2")
    assert dlg.record_btn.isEnabled(), dlg.status_label.text()
    assert dlg._channels()[0] == {2: [1, 2, 3, 4], 3: [1, 2]}
    dlg.channels_edit.setText("1-4")
    assert dlg._channels()[0] == {2: [1, 2, 3, 4], 4: [1, 2, 3, 4]}
    # Modules run 1-4, the bound the parser and the wire have.
    dlg.modules_edit.setText("5")
    assert "name the modules" in dlg.status_label.text()
    dlg.modules_edit.setText("2")
    dlg.channels_edit.setText("5:1-4")
    assert "Modules run 1-4" in dlg.status_label.text()



def test_a_config_saved_with_a_retired_field_keeps_the_rest(
        qt_app, tmp_path, monkeypatch):
    """Settings saved before margin_fraction became the pre-pulse and
    post-pulse times still carry the user's other choices."""
    _, settings = _dialog(tmp_path, monkeypatch)
    settings.setValue("record/capture_config",
                      '{"threshold_sigma": 6.5, "margin_fraction": 0.2}')
    cfg = rd.RecordDialog(settings=settings).get_options()["config"]
    assert cfg.threshold_sigma == 6.5
    assert cfg.pre_pulse_ms == rd.PulseCaptureConfig().pre_pulse_ms
