"""UI font scaling (Ctrl+/Ctrl-/Ctrl+0) — settings + application logic."""

import os

import pytest


pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope import settings  # noqa: E402



@pytest.fixture(autouse=True)
def _restore_scale():
    original = settings.get_font_scale()
    yield
    settings.set_font_scale(original)


def test_settings_roundtrip_and_clamp():
    settings.set_font_scale(1.4)
    assert settings.get_font_scale() == pytest.approx(1.4)
    settings.set_font_scale(99.0)
    assert settings.get_font_scale() == settings.FONT_SCALE_MAX
    settings.set_font_scale(0.01)
    assert settings.get_font_scale() == settings.FONT_SCALE_MIN


def test_scale_applies_to_application_font(qt_app):
    """_set_font_scale is bound to the app font, so every widget follows."""
    from types import SimpleNamespace

    from rfmux.tools.periscope.app import Periscope

    # Plain stand-in: the methods only touch font_scale, _base_font_pt
    # and statusBar(), so no QMainWindow construction is needed.
    fake = SimpleNamespace(font_scale=1.0,
                           statusBar=lambda: _NullStatusBar())
    fake._set_font_scale = lambda s: Periscope._set_font_scale(fake, s)

    base_pt = qt_app.font().pointSizeF()
    try:
        Periscope._set_font_scale(fake, 1.5)
        assert qt_app.font().pointSizeF() == pytest.approx(base_pt * 1.5)
        assert fake.font_scale == pytest.approx(1.5)
        assert settings.get_font_scale() == pytest.approx(1.5)

        Periscope._adjust_font_scale(fake, 1 / 1.5)
        assert qt_app.font().pointSizeF() == pytest.approx(base_pt)

        Periscope._set_font_scale(fake, 99.0)     # clamped
        assert fake.font_scale == settings.FONT_SCALE_MAX
    finally:
        Periscope._set_font_scale(fake, 1.0)


class _NullStatusBar:
    def showMessage(self, *args, **kwargs):
        pass
