"""Shared fixtures for the Periscope tests."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings

from rfmux.tools.periscope import settings as periscope_settings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Every test gets its own QSettings file.

    Periscope remembers preferences in the user's real
    ``~/.config/rfmux/periscope.conf``. A test that writes them would edit the
    settings of whoever ran it, and a test that reads them would pass or fail
    depending on what that person last did in the GUI. One file per test makes
    both impossible, and makes what a panel remembers between constructions
    something a test can assert on.
    """
    ini = tmp_path / "periscope.conf"
    monkeypatch.setattr(
        periscope_settings, "_get_settings",
        lambda: QSettings(str(ini), QSettings.Format.IniFormat))
    return ini
