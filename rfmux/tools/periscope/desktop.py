"""Periscope in the desktop, per user and without administrator rights:
an application-menu entry with Periscope's icon that opens the startup
dialog, and "Open with Periscope" on HDF5 files, which opens one in
review mode.  Asked for, Periscope becomes the default for HDF5 files
and gets a desktop shortcut too.  Linux (the freedesktop entry every
desktop reads) and Windows (the current user's registry and Start
Menu).

    rfmux periscope --install-desktop [--default-for-hdf5] [--desktop-icon]
    rfmux periscope --uninstall-desktop

The entries start this environment's ``periscope`` by its full path,
so they keep working with the environment inactive, and point at
whichever environment ran the install.
"""

from __future__ import annotations

import configparser
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import List

from ..cli import periscope_command
from .utils import ICON_PATH

APP_ID = "rfmux-periscope"
NAME = "Periscope"
COMMENT = "Open a pulse capture or time-ordered data file in review mode"
MIME_TYPES = ("application/x-hdf", "application/x-hdf5")
EXTENSIONS = (".h5", ".hdf5")
#: Windows: the registry class HDF5 files are opened through.
PROG_ID = "rfmux.Periscope.HDF5"
#: Windows icon sizes, rendered from the SVG.
ICO_SIZES = (16, 32, 48, 256)


def launcher() -> List[str]:
    """The command that starts Periscope from this environment."""
    return periscope_command()


def install(default: bool = False, desktop_icon: bool = False) -> List[str]:
    """Install the entries; the paths and keys written."""
    if sys.platform.startswith("linux"):
        return _linux_install(default, desktop_icon)
    if sys.platform == "win32":
        return _windows_install(default, desktop_icon)
    raise RuntimeError(f"--install-desktop supports Linux and Windows, not "
                       f"{sys.platform}")


def uninstall() -> List[str]:
    """Remove what :func:`install` wrote; the paths and keys removed."""
    if sys.platform.startswith("linux"):
        return _linux_uninstall()
    if sys.platform == "win32":
        return _windows_uninstall()
    raise RuntimeError(f"--uninstall-desktop supports Linux and Windows, "
                       f"not {sys.platform}")


# ── Linux: the freedesktop entry ───────────────────────────────────

def _data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME")
                or Path.home() / ".local" / "share")


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _desktop_dir() -> Path:
    """The user's desktop folder, as xdg-user-dir names it."""
    try:
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        if out:
            return Path(out)
    except (OSError, subprocess.SubprocessError):
        pass
    return Path.home() / "Desktop"


def exec_quote(arg: str) -> str:
    """One argument of a desktop entry's Exec key: double-quoted when it
    holds a space or a reserved character, with the characters the
    specification escapes inside quotes escaped."""
    if not any(c in arg for c in ' \t\n"\'\\><~|&;$*?#()`'):
        return arg
    escaped = "".join("\\" + c if c in '"`$\\' else c for c in arg)
    return f'"{escaped}"'


def desktop_entry() -> str:
    """The entry: Periscope, opening the file it is given in review mode
    and the startup dialog without one (``--review`` with no file)."""
    command = " ".join(exec_quote(a) for a in launcher())
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={NAME}",
        f"Comment={COMMENT}",
        f"Exec={command} --review %f",
        f"Icon={APP_ID}",
        "Terminal=false",
        "MimeType=" + "".join(m + ";" for m in MIME_TYPES),
        "Categories=Science;",
        "",
    ])


def _linux_paths():
    data = _data_home()
    return (data / "applications" / f"{APP_ID}.desktop",
            data / "icons" / "hicolor" / "scalable" / "apps" / f"{APP_ID}.svg")


def _refresh(entry: Path) -> None:
    """Let the desktop see the change now; a tool that is missing only
    delays it to the next login."""
    for cmd in (["update-desktop-database", str(entry.parent)],
                ["gtk-update-icon-cache", "-q", "-t",
                 str(_data_home() / "icons" / "hicolor")]):
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass


def _mimeapps() -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str                        # MIME types keep their case
    cp.read(_config_home() / "mimeapps.list")
    return cp


def _write_mimeapps(cp: configparser.ConfigParser) -> None:
    path = _config_home() / "mimeapps.list"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        cp.write(f, space_around_delimiters=False)


def _linux_install(default: bool, desktop_icon: bool) -> List[str]:
    entry, icon = _linux_paths()
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(desktop_entry())
    icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ICON_PATH, icon)
    written = [str(entry), str(icon)]
    if default:
        cp = _mimeapps()
        if not cp.has_section("Default Applications"):
            cp.add_section("Default Applications")
        for mime in MIME_TYPES:
            cp["Default Applications"][mime] = entry.name
        _write_mimeapps(cp)
        written.append(str(_config_home() / "mimeapps.list"))
    if desktop_icon:
        shortcut = _desktop_dir() / entry.name
        shortcut.parent.mkdir(parents=True, exist_ok=True)
        shortcut.write_text(desktop_entry())
        shortcut.chmod(0o755)
        # GNOME and Cinnamon run a desktop launcher only once trusted.
        try:
            subprocess.run(["gio", "set", str(shortcut), "metadata::trusted",
                            "true"], capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
        written.append(str(shortcut))
    _refresh(entry)
    return written


def _linux_uninstall() -> List[str]:
    entry, icon = _linux_paths()
    removed = []
    for path in (entry, icon, _desktop_dir() / entry.name):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    cp = _mimeapps()
    changed = False
    for section in cp.sections():
        for mime, value in list(cp[section].items()):
            if entry.name in value.split(";"):
                kept = [v for v in value.split(";") if v and v != entry.name]
                if kept:
                    cp[section][mime] = ";".join(kept) + ";"
                else:
                    del cp[section][mime]
                changed = True
    if changed:
        _write_mimeapps(cp)
        removed.append(f"{_config_home() / 'mimeapps.list'}: {entry.name}")
    _refresh(entry)
    return removed


# ── Windows: the current user's registry and Start Menu ────────────

def _windows_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA")
                or Path.home() / "AppData" / "Local") / "rfmux"


def _start_menu() -> Path:
    return (Path(os.environ.get("APPDATA")
                 or Path.home() / "AppData" / "Roaming")
            / "Microsoft" / "Windows" / "Start Menu" / "Programs")


def _windows_desktop() -> Path:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "[Environment]::GetFolderPath('Desktop')"],
        capture_output=True, text=True, timeout=30).stdout.strip()
    return Path(out) if out else Path.home() / "Desktop"


def registry_values(ico: str, default: bool) -> List[tuple]:
    """``(subkey under HKCU, value name, value)`` the install writes: the
    class HDF5 files open through, its icon and command, the class in
    each extension's Open With list, and the extension's default when
    asked.  Value name "" is the key's default value."""
    command = " ".join(f'"{a}"' for a in launcher()) + ' --review "%1"'
    base = rf"Software\Classes\{PROG_ID}"
    values = [(base, "", "HDF5 file"),
              (base + r"\DefaultIcon", "", ico),
              (base + r"\shell\open", "", f"Open with {NAME}"),
              (base + r"\shell\open\command", "", command)]
    for ext in EXTENSIONS:
        values.append((rf"Software\Classes\{ext}\OpenWithProgids", PROG_ID, ""))
        if default:
            values.append((rf"Software\Classes\{ext}", "", PROG_ID))
    return values


def ico_bytes(pngs: List[bytes], sizes=ICO_SIZES) -> bytes:
    """A Windows icon of the given PNG images, one per size: the icon
    directory and the PNGs as its entries (Windows Vista and later)."""
    head = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries, data = b"", b""
    for size, png in zip(sizes, pngs):
        dim = 0 if size >= 256 else size         # 0 means 256
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png),
                               offset + len(data))
        data += png
    return head + entries + data


def render_pngs(sizes=ICO_SIZES) -> List[bytes]:
    """Periscope's SVG icon as PNGs of each size."""
    from PyQt6 import QtCore, QtGui, QtSvg
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    renderer = QtSvg.QSvgRenderer(ICON_PATH)
    out = []
    for size in sizes:
        image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
        painter.end()
        buf = QtCore.QBuffer()
        buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        image.save(buf, "PNG")
        out.append(bytes(buf.data()))
    del app
    return out


def shortcut_script(path: Path, ico: Path) -> str:
    """PowerShell that writes a shortcut to Periscope at *path*."""
    cmd = launcher()
    quote = lambda s: "'" + s.replace("'", "''") + "'"
    return ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
            f"{quote(str(path))}); $s.TargetPath = {quote(cmd[0])}; "
            f"$s.Arguments = {quote(' '.join(cmd[1:]))}; "
            f"$s.IconLocation = {quote(str(ico))}; "
            f"$s.Description = {quote(COMMENT)}; $s.Save()")


def _shortcut(path: Path, ico: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    shortcut_script(path, ico)], check=True,
                   capture_output=True, timeout=60)


def _notify_shell() -> None:
    """Tell Explorer the associations changed, so icons update now."""
    import ctypes
    ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)


def _windows_install(default: bool, desktop_icon: bool) -> List[str]:
    import winreg
    ico = _windows_dir() / "periscope.ico"
    ico.parent.mkdir(parents=True, exist_ok=True)
    ico.write_bytes(ico_bytes(render_pngs()))
    written = [str(ico)]
    for subkey, name, value in registry_values(str(ico), default):
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, subkey) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        written.append(rf"HKCU\{subkey}" + (f" [{name}]" if name else ""))
    links = [_start_menu() / f"{NAME}.lnk"]
    if desktop_icon:
        links.append(_windows_desktop() / f"{NAME}.lnk")
    for link in links:
        _shortcut(link, ico)
        written.append(str(link))
    _notify_shell()
    return written


def _delete_tree(winreg, root, subkey: str) -> bool:
    try:
        with winreg.OpenKey(root, subkey, 0, winreg.KEY_ALL_ACCESS) as key:
            while True:
                try:
                    child = winreg.EnumKey(key, 0)
                except OSError:
                    break
                _delete_tree(winreg, root, rf"{subkey}\{child}")
        winreg.DeleteKey(root, subkey)
        return True
    except FileNotFoundError:
        return False


def _windows_uninstall() -> List[str]:
    import winreg
    hkcu = winreg.HKEY_CURRENT_USER
    removed = []
    if _delete_tree(winreg, hkcu, rf"Software\Classes\{PROG_ID}"):
        removed.append(rf"HKCU\Software\Classes\{PROG_ID}")
    for ext in EXTENSIONS:
        for subkey, name in ((rf"Software\Classes\{ext}\OpenWithProgids",
                              PROG_ID), (rf"Software\Classes\{ext}", "")):
            try:
                with winreg.OpenKey(hkcu, subkey, 0,
                                    winreg.KEY_ALL_ACCESS) as key:
                    value, _ = winreg.QueryValueEx(key, name)
                    if name or value == PROG_ID:
                        winreg.DeleteValue(key, name)
                        removed.append(rf"HKCU\{subkey}" +
                                       (f" [{name}]" if name else ""))
            except FileNotFoundError:
                pass
    desktop = _windows_desktop()
    for path in (_start_menu() / f"{NAME}.lnk", desktop / f"{NAME}.lnk",
                 _windows_dir() / "periscope.ico"):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    _notify_shell()
    return removed
