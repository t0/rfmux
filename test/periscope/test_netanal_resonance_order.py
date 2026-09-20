"""The network analysis panel keeps a module's resonances in frequency
order however they arrive, because multisweep assigns channels in list
order: a resonance added by hand belongs between its neighbors, not at
the end of the channel list."""
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.network_analysis_panel import (  # noqa: E402
    NetworkAnalysisPanel)


def _line_positions(panel, module):
    info = panel.plots[module]
    return ([line.value() for line in info['resonance_lines_mag']],
            [line.value() for line in info['resonance_lines_phase']])


@pytest.fixture
def panel(qt_app):
    return NetworkAnalysisPanel(modules=[1], dark_mode=False)


def test_added_resonance_lands_in_frequency_order(panel):
    for freq in (5e8, 3e8, 4e8):
        panel._add_resonance(1, freq)
    assert panel.resonance_freqs[1] == [3e8, 4e8, 5e8]
    assert _line_positions(panel, 1) == ([3e8, 4e8, 5e8], [3e8, 4e8, 5e8])


def test_removal_after_an_add_takes_the_matching_lines(panel):
    for freq in (5e8, 3e8, 4e8):
        panel._add_resonance(1, freq)
    panel._remove_resonance(1, 4.01e8)
    assert panel.resonance_freqs[1] == [3e8, 5e8]
    assert _line_positions(panel, 1) == ([3e8, 5e8], [3e8, 5e8])


def test_loaded_resonances_are_sorted(panel):
    """Loading replaces the lines wholesale rather than adding one at a
    time, and sorts on its own."""
    panel._use_loaded_resonances(1, [5e8, 3e8, 4e8])
    assert panel.resonance_freqs[1] == [3e8, 4e8, 5e8]
    assert _line_positions(panel, 1)[0] == [3e8, 4e8, 5e8]
