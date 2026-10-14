"""The Fit Results tab's toolbar: which fit is drawn, and over which sweeps."""
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.fit_display_toolbar import FitDisplayToolbar  # noqa: E402
from rfmux.tools.periscope.fit_settings_panel import (  # noqa: E402
    ALL_AMPLITUDES,
    BIAS_AMPLITUDE,
)


@pytest.fixture
def toolbar(qt_app):
    """A toolbar saving into this test's own settings file."""
    return FitDisplayToolbar()


def test_nothing_fitted_is_nothing_to_draw(toolbar):
    """The combo is dead until the sweeps carry fits of something."""
    assert toolbar.get_model() is None
    assert not toolbar.model_combo.isEnabled()

    toolbar.set_models_fitted(["skewed", "nonlinear"])

    assert toolbar.get_model() == "skewed"
    assert toolbar.model_combo.isEnabled()


def test_the_model_drawn_survives_a_refit(toolbar):
    """Re-running the fits does not move the tab off what it was showing."""
    toolbar.set_models_fitted(["skewed", "nonlinear"])
    toolbar.model_combo.setCurrentIndex(toolbar.model_combo.findData("nonlinear"))

    toolbar.set_models_fitted(["skewed", "nonlinear"])

    assert toolbar.get_model() == "nonlinear"


def test_choosing_a_model_asks_for_a_redraw(toolbar):
    """Both controls change what is on screen, so both say so."""
    toolbar.set_models_fitted(["skewed", "nonlinear"])
    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES), ("Step 0", 0)])
    redraws = []
    toolbar.display_changed.connect(lambda: redraws.append(True))

    toolbar.model_combo.setCurrentIndex(toolbar.model_combo.findData("nonlinear"))
    toolbar.amplitude_combo.setCurrentIndex(toolbar.amplitude_combo.findData(0))

    assert redraws == [True, True]


def test_every_amplitude_is_the_default(toolbar):
    """A tab that has been told nothing draws everything that was fitted."""
    assert toolbar.get_amplitude() is ALL_AMPLITUDES


def test_a_step_the_next_measurement_lacks_falls_back(toolbar):
    """A step is a step of one measurement's schedule; 'all of them' is the
    answer that is always true."""
    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES),
                                   ("Step 0", 0), ("Step 1", 1)])
    toolbar.amplitude_combo.setCurrentIndex(toolbar.amplitude_combo.findData(1))

    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES), ("Step 0", 0)])

    assert toolbar.get_amplitude() is ALL_AMPLITUDES


def test_what_was_drawn_last_session_is_picked_up(qt_app):
    """The combos are empty until a measurement arrives, so the choices have to
    outlive that."""
    toolbar = FitDisplayToolbar()
    toolbar.set_models_fitted(["skewed", "nonlinear"])
    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES), ("Step 1", 1)])
    toolbar.model_combo.setCurrentIndex(toolbar.model_combo.findData("nonlinear"))
    toolbar.amplitude_combo.setCurrentIndex(toolbar.amplitude_combo.findData(1))

    next_session = FitDisplayToolbar()
    next_session.set_models_fitted(["skewed", "nonlinear"])
    next_session.set_amplitude_choices(
        [("All amplitudes", ALL_AMPLITUDES), ("Step 1", 1)])

    assert next_session.get_model() == "nonlinear"
    assert next_session.get_amplitude() == 1


def test_the_bias_amplitude_is_offered_only_when_there_is_one(toolbar):
    """'At bias' means the step a resonator was biased at, which is nothing
    until a bias has been found; the multisweep panel says when it has."""
    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES), ("Step 0", 0)])
    assert toolbar.amplitude_combo.findData(BIAS_AMPLITUDE) == -1

    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES),
                                   ("At bias amplitude", BIAS_AMPLITUDE),
                                   ("Step 0", 0)])
    toolbar.amplitude_combo.setCurrentIndex(
        toolbar.amplitude_combo.findData(BIAS_AMPLITUDE))

    assert toolbar.get_amplitude() == BIAS_AMPLITUDE


def test_histograms_default_to_bias_then_step_zero(qt_app):
    toolbar = FitDisplayToolbar(name="histograms", all_amplitudes=False)
    choices = [("All amplitudes", ALL_AMPLITUDES), ("Step 1", 1), ("Step 0", 0)]
    toolbar.set_amplitude_choices(choices)
    assert toolbar.amplitude_combo.findData(ALL_AMPLITUDES) == -1
    assert toolbar.get_amplitude() == 0
    toolbar.set_models_fitted(["skewed", "nonlinear"])
    toolbar.model_combo.setCurrentIndex(1)

    toolbar.set_amplitude_choices(choices + [("At bias amplitude", BIAS_AMPLITUDE)])
    assert toolbar.get_amplitude() == BIAS_AMPLITUDE


def test_histograms_replace_saved_all_amplitudes(qt_app):
    from rfmux.tools.periscope import settings

    settings.set_fit_display({"amplitude": ALL_AMPLITUDES}, "histograms")
    toolbar = FitDisplayToolbar(name="histograms", all_amplitudes=False)
    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES),
                                   ("At bias amplitude", BIAS_AMPLITUDE),
                                   ("Step 0", 0)])
    assert toolbar.get_amplitude() == BIAS_AMPLITUDE


def test_histograms_retain_saved_single_step(qt_app):
    from rfmux.tools.periscope import settings

    settings.set_fit_display({"amplitude": 1}, "histograms")
    toolbar = FitDisplayToolbar(name="histograms", all_amplitudes=False)
    toolbar.set_amplitude_choices([("At bias amplitude", BIAS_AMPLITUDE),
                                   ("Step 0", 0), ("Step 1", 1)])
    assert toolbar.get_amplitude() == 1


def test_frequency_scatter_sorts_and_colours_by_fitted_qr(qt_app):
    import numpy as np
    import pyqtgraph as pg
    from rfmux.tools.periscope.fit_histograms_tab import FitHistogramsTab

    tab = FitHistogramsTab()
    tab._lay_out(1)
    rows = [{"name": name, "amplitude": 0.1, "params": {"fr": fr, "Qr": qr}}
            for name, fr, qr in [("A", 300e6, 20000), ("Z", 100e6, 30000),
                                 ("B", 200e6, 10000)]]
    tab._draw_fr(tab._plots[0], rows)
    scatter = next(item for item in tab._plots[0].getPlotItem().items
                   if isinstance(item, pg.ScatterPlotItem))
    np.testing.assert_array_equal(scatter.getData()[0], [0, 1, 2])
    np.testing.assert_array_equal(scatter.getData()[1], [100, 200, 300])
    cmap = pg.colormap.get("viridis")
    assert [point.brush().color() for point in scatter.points()] == [
        cmap.map(value, mode="qcolor") for value in [1.0, 0.0, 0.5]]
    assert tab._qr_colorbar.levels() == (10000, 30000)


@pytest.mark.parametrize("qr", [20000, float("nan")])
def test_frequency_scatter_handles_uniform_or_missing_qr(qt_app, qr):
    import pyqtgraph as pg
    from rfmux.tools.periscope.fit_histograms_tab import FitHistogramsTab

    tab = FitHistogramsTab()
    tab._lay_out(1)
    rows = [{"params": {"fr": 100e6, "Qr": qr}}]
    for dark_mode in (False, True):
        tab._dark_mode = dark_mode
        tab._draw_fr(tab._plots[0], rows)
        scatter = next(item for item in tab._plots[0].getPlotItem().items
                       if isinstance(item, pg.ScatterPlotItem))
        assert len(scatter.points()) == 1
        expected = (pg.colormap.get("viridis").map(0.5, mode="qcolor")
                    if qr == qr else pg.mkColor("w" if dark_mode else "k"))
        assert scatter.points()[0].brush().color() == expected


def test_frequency_colour_bar_has_a_visible_qr_heading(qt_app):
    from rfmux.tools.periscope.fit_histograms_tab import FitHistogramsTab

    tab = FitHistogramsTab()
    tab.resize(900, 600)
    rows = [{"params": {"fr": 100e6, "Qr": 20000}}]
    for dark_mode in (False, True):
        tab._dark_mode = dark_mode
        tab._lay_out(4)
        tab._draw_fr(tab._plots[0], rows)
        tab.show()
        qt_app.processEvents()
        heading = tab._qr_colorbar.titleLabel
        assert heading.text == "Qr"
        assert heading.isVisible()
        assert tab._plots[0].sceneRect().contains(heading.sceneBoundingRect())
    tab.close()
