"""The Noise Spectrum panel builds offscreen and walks its detectors."""

import pytest

from test.qt_helpers import spin


pytest.importorskip("PyQt6")

from rfmux.tools.periscope.noise_spectrum_panel import (  # noqa: E402
    NoiseSpectrumPanel,
)


TWO_DETECTORS = {
    1: {"conceptual_freq_hz": 4.0e9},
    2: {"conceptual_freq_hz": 4.1e9},
}


def test_navigation_follows_the_number_of_detectors(qt_app):
    """The panel builds without data; with one detector there is nowhere
    to navigate, so both buttons stay dead rather than wrapping onto the
    same detector."""
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

    lone = NoiseSpectrumPanel(
        detector_id=1,
        resonance_frequency_ghz=4.0,
        all_detectors_data={1: {"conceptual_freq_hz": 4.0e9}},
        initial_detector_idx=1,
    )
    assert not lone.prev_button.isEnabled()
    assert not lone.next_button.isEnabled()
    lone.close()
    spin(qt_app)


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
    spin(qt_app)


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
    spin(qt_app)
