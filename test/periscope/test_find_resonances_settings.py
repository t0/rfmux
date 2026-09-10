"""The resonance finder's settings panel: what it sends, and what it keeps.

A parameter the panel offers and the finder does not accept is a TypeError at
the end of a network analysis. A default that drifts from the library's is a
GUI that quietly searches differently from a notebook over the same trace.
Nothing but these tests checks either.
"""

import inspect

import pytest

pytest.importorskip("PyQt6")

from test.qt_helpers import spin  # noqa: E402

from rfmux.tools.periscope.find_resonances_settings_panel import (  # noqa: E402
    DEFAULTS,
    FindResonancesSettingsPanel,
)
from rfmux.tuning.find_resonances import find_resonances  # noqa: E402


@pytest.fixture
def panel(qt_app):
    """Builds settings panels, and closes them however the test ends.

    ``isolated_settings`` in this directory's conftest gives each test its own
    QSettings file, so a second panel here reads back what the first one saved
    and nothing else.
    """
    made = []

    def build():
        made.append(FindResonancesSettingsPanel())
        return made[-1]

    yield build
    for widget in made:
        widget.close()
    spin(qt_app)


def test_the_panel_asks_for_nothing_the_finder_does_not_accept(panel):
    """Its output is splatted into the finder, so a key it does not take is a
    TypeError the moment a search runs."""
    accepted = set(inspect.signature(find_resonances).parameters)
    assert set(panel().get_parameters()) <= accepted


def test_the_panel_offers_every_threshold_the_finder_has(panel):
    """A knob missing from the panel is one a Periscope user cannot reach, and
    the finder is small enough that all of them belong here. ``label`` is the
    exception: the finder derives it from the module."""
    optional = {
        name for name, parameter in inspect.signature(find_resonances).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    assert set(panel().get_parameters()) == optional - {"label"}


def test_a_fresh_panel_asks_for_the_librarys_defaults(panel):
    """Untouched, the panel searches exactly as ``find_resonances()`` does.

    The boxes encode two of those defaults specially -- ``None`` as a zero
    reading "No limit", and a separation shown in kHz -- so this is a check on
    the encoding as much as on the values.
    """
    assert panel().get_parameters() == DEFAULTS


def test_no_limit_means_no_limit(panel):
    """Zero in a Q box is the finder's ``None``, not a Q of zero."""
    settings = panel()
    settings.min_q_spin.setValue(0.0)
    settings.max_q_spin.setValue(0.0)

    parameters = settings.get_parameters()
    assert parameters["min_Q"] is None
    assert parameters["max_Q"] is None


def test_the_collision_cut_is_typed_in_khz_and_sent_in_hz(panel):
    settings = panel()
    settings.min_separation_spin.setValue(7.5)

    assert settings.get_parameters()["min_separation_hz"] == 7500.0


def test_the_isolation_switch_reaches_the_finder(panel):
    """On by default, which is the finder's default: a tone on either member of
    a collided pair reads the other."""
    settings = panel()
    assert settings.require_isolation_check.isChecked() is True
    assert settings.get_parameters()["require_isolation"] is True

    settings.require_isolation_check.setChecked(False)
    assert settings.get_parameters()["require_isolation"] is False


def test_thresholds_outlive_the_panel(panel):
    """The point of a settings panel over a dialog: set it once, search many
    times, and find it still set next session."""
    first = panel()
    first.min_dip_depth_spin.setValue(0.4)
    first.expected_resonances_spin.setValue(64)
    first.require_isolation_check.setChecked(False)

    parameters = panel().get_parameters()
    assert parameters["min_dip_depth_db"] == 0.4
    assert parameters["expected_resonances"] == 64
    assert parameters["require_isolation"] is False


def test_reset_goes_back_to_the_library(panel):
    settings = panel()
    settings.min_dip_depth_spin.setValue(0.4)
    settings.min_q_spin.setValue(0.0)

    settings._reset()

    assert settings.get_parameters() == DEFAULTS
    assert panel().get_parameters() == DEFAULTS
