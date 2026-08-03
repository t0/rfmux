"""
Offscreen construction/navigation tests for the Noise Spectrum panel.

Replaces the old root-level test_noise_panel_smoke.py, which wrapped the
constructor in try/except and returned a bool — pytest ignores return
values, so that "test" passed even when the panel failed to build.  These
assert instead, and cover the detector navigation the panel adds on top of
plain construction.
"""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.noise_spectrum_panel import (  # noqa: E402
    NoiseSpectrumPanel,
)


@pytest.fixture
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _spin(qt_app, seconds=0.05):
    """Let Qt drain deferred deletions after closing a panel."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)


TWO_DETECTORS = {
    1: {"conceptual_freq_hz": 4.0e9},
    2: {"conceptual_freq_hz": 4.1e9},
}


def test_panel_builds_with_multiple_detectors(qt_app):
    """The panel constructs without data and enables navigation when it has
    more than one detector to walk."""
    panel = NoiseSpectrumPanel(
        detector_id=1,
        resonance_frequency_ghz=4.0,
        all_detectors_data=TWO_DETECTORS,
        initial_detector_idx=1,
    )
    assert panel.detector_indices == [1, 2]
    assert panel.current_detector_index_in_list == 0
    assert panel.prev_button.isEnabled()
    assert panel.next_button.isEnabled()
    panel.close()
    _spin(qt_app)


def test_navigation_is_disabled_for_a_lone_detector(qt_app):
    """With a single detector there is nowhere to navigate, so both buttons
    stay dead rather than wrapping onto the same detector."""
    panel = NoiseSpectrumPanel(
        detector_id=1,
        resonance_frequency_ghz=4.0,
        all_detectors_data={1: {"conceptual_freq_hz": 4.0e9}},
        initial_detector_idx=1,
    )
    assert not panel.prev_button.isEnabled()
    assert not panel.next_button.isEnabled()
    panel.close()
    _spin(qt_app)


def test_navigation_wraps_and_retitles(qt_app):
    """Next/previous move the selection modulo the detector list and pull the
    new detector's frequency through for the title."""
    panel = NoiseSpectrumPanel(
        detector_id=1,
        resonance_frequency_ghz=4.0,
        all_detectors_data=TWO_DETECTORS,
        initial_detector_idx=1,
    )

    panel._navigate_next()
    assert panel.detector_id == 2
    assert panel.resonance_frequency_ghz_title == pytest.approx(4.1)

    # Two detectors, so one more step wraps back to the first.
    panel._navigate_next()
    assert panel.detector_id == 1
    assert panel.resonance_frequency_ghz_title == pytest.approx(4.0)

    # And previous wraps the other way.
    panel._navigate_previous()
    assert panel.detector_id == 2
    panel.close()
    _spin(qt_app)


def test_initial_detector_falls_back_when_unknown(qt_app):
    """An initial_detector_idx that is not in the data must not raise; the
    panel falls back to the first detector."""
    panel = NoiseSpectrumPanel(
        detector_id=99,
        resonance_frequency_ghz=0.0,
        all_detectors_data=TWO_DETECTORS,
        initial_detector_idx=99,
    )
    assert panel.current_detector_index_in_list == 0
    panel.close()
    _spin(qt_app)
