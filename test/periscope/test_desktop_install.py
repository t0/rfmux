"""Periscope in the desktop: the entries --install-desktop writes, and
that --uninstall-desktop takes them away again, leaving the rest."""

import configparser
import os
import struct
import sys
from pathlib import Path

import pytest

from rfmux.tools.periscope import desktop
from rfmux.tools.periscope.utils import ICON_PATH

linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="the freedesktop entry is Linux's")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A user with their own data, config and desktop folders."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(desktop, "_desktop_dir", lambda: tmp_path / "Desktop")
    return tmp_path


def _entry(path) -> configparser.SectionProxy:
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(path)
    return cp["Desktop Entry"]


@linux
def test_install_adds_the_menu_entry_its_icon_and_open_with(home):
    written = desktop.install()
    entry_path = home / "data/applications/rfmux-periscope.desktop"
    icon = home / "data/icons/hicolor/scalable/apps/rfmux-periscope.svg"
    assert set(written) == {str(entry_path), str(icon)}
    entry = _entry(entry_path)
    assert entry["Name"] == "Periscope" and entry["Icon"] == "rfmux-periscope"
    assert entry["Exec"].endswith(" --review %f")
    assert entry["Exec"].startswith(desktop.exec_quote(desktop.launcher()[0]))
    assert entry["MimeType"] == "application/x-hdf;application/x-hdf5;"
    assert icon.read_bytes() == Path(ICON_PATH).read_bytes()
    # Neither the default nor a desktop shortcut unless asked.
    assert not (home / "config/mimeapps.list").exists()
    assert not (home / "Desktop").exists()


@linux
def test_asked_it_is_the_default_and_on_the_desktop_and_uninstall_undoes_it(
        home):
    mimeapps = home / "config/mimeapps.list"
    mimeapps.parent.mkdir(parents=True)
    mimeapps.write_text("[Default Applications]\ntext/plain=gedit.desktop\n"
                        "[Added Associations]\n"
                        "application/x-hdf=hdfview.desktop;\n")
    desktop.install(default=True, desktop_icon=True)
    cp = desktop._mimeapps()
    for mime in desktop.MIME_TYPES:
        assert cp["Default Applications"][mime] == "rfmux-periscope.desktop"
    shortcut = home / "Desktop/rfmux-periscope.desktop"
    assert shortcut.exists() and os.access(shortcut, os.X_OK)

    removed = desktop.uninstall()
    assert not (home / "data/applications/rfmux-periscope.desktop").exists()
    assert not (home / "data/icons/hicolor/scalable/apps/"
                       "rfmux-periscope.svg").exists()
    assert not shortcut.exists()
    cp = desktop._mimeapps()
    assert cp["Default Applications"]["text/plain"] == "gedit.desktop"
    assert "application/x-hdf" not in cp["Default Applications"]
    assert cp["Added Associations"]["application/x-hdf"] == "hdfview.desktop;"
    assert any("mimeapps.list" in r for r in removed)
    assert desktop.uninstall() == []


def test_exec_arguments_are_quoted_as_the_specification_asks():
    assert desktop.exec_quote("/usr/bin/periscope") == "/usr/bin/periscope"
    assert desktop.exec_quote("/home/a b/$x/periscope") == \
        '"/home/a b/\\$x/periscope"'


def test_the_windows_registry_entries_open_hdf5_files_in_review(monkeypatch):
    monkeypatch.setattr(desktop, "launcher",
                        lambda: [r"C:\env\Scripts\periscope.exe"])
    values = desktop.registry_values(r"C:\u\rfmux\periscope.ico", default=False)
    by_key = {(k, n): v for k, n, v in values}
    base = rf"Software\Classes\{desktop.PROG_ID}"
    assert by_key[(base + r"\shell\open\command", "")] == \
        r'"C:\env\Scripts\periscope.exe" --review "%1"'
    assert by_key[(base + r"\DefaultIcon", "")] == r"C:\u\rfmux\periscope.ico"
    for ext in (".h5", ".hdf5"):
        assert (rf"Software\Classes\{ext}\OpenWithProgids",
                desktop.PROG_ID) in by_key
        assert (rf"Software\Classes\{ext}", "") not in by_key
    defaults = desktop.registry_values("x.ico", default=True)
    assert (r"Software\Classes\.h5", "", desktop.PROG_ID) in defaults


def test_the_windows_icon_holds_the_svg_at_each_size(qt_app):
    pngs = desktop.render_pngs()
    ico = desktop.ico_bytes(pngs)
    reserved, kind, count = struct.unpack_from("<HHH", ico)
    assert (reserved, kind, count) == (0, 1, len(desktop.ICO_SIZES))
    for i, size in enumerate(desktop.ICO_SIZES):
        w, h, _, _, _, bpp, n, offset = struct.unpack_from(
            "<BBBBHHII", ico, 6 + 16 * i)
        assert w == h == (0 if size == 256 else size) and bpp == 32
        assert ico[offset:offset + 8] == b"\x89PNG\r\n\x1a\n"
        assert ico[offset:offset + n] == pngs[i]


@pytest.mark.skipif(sys.platform != "win32", reason="the registry is Windows'")
def test_windows_install_and_uninstall(tmp_path, monkeypatch):
    import winreg
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setattr(desktop, "_windows_desktop",
                        lambda: tmp_path / "Desktop")
    desktop.install(desktop_icon=True)
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            rf"Software\Classes\{desktop.PROG_ID}"
                            r"\shell\open\command") as key:
            assert winreg.QueryValue(key, None).endswith('--review "%1"')
        assert (tmp_path / "roaming/Microsoft/Windows/Start Menu/Programs/"
                           "Periscope.lnk").exists()
        assert (tmp_path / "Desktop/Periscope.lnk").exists()
    finally:
        desktop.uninstall()
    with pytest.raises(FileNotFoundError):
        winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                       rf"Software\Classes\{desktop.PROG_ID}")
    assert not (tmp_path / "Desktop/Periscope.lnk").exists()


def test_the_command_installs_without_starting_qt(monkeypatch, capsys):
    from rfmux.tools.periscope import __main__ as m
    calls = []
    monkeypatch.setattr(desktop, "install",
                        lambda **kw: calls.append(kw) or ["/x/entry"])
    monkeypatch.setattr(sys, "argv", ["periscope", "--install-desktop",
                                      "--default-for-hdf5"])
    monkeypatch.setattr(m.QtWidgets, "QApplication",
                        lambda *a: pytest.fail("Qt started"), raising=False)
    assert m.main() == 0
    assert calls == [{"default": True, "desktop_icon": False}]
    assert "/x/entry" in capsys.readouterr().out
