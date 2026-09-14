"""The tuning flow, driven end to end against the standard simulated array.

Everything else in ``test/tuning`` works on synthetic traces. This module is
where the drivers actually run: one multisweep over an amplitude schedule on the
standard array (``rfmux.mock.standard_array``), then bias finding on what came
back, then biasing the board with the result. The array and the schedule are
built once for the module — the schedule is the expensive step, about half a
minute on the simulator — and every test reads the same measurement.

These are the contracts the pre-merge ``test/algorithms`` flow tests pinned,
restated for this branch's shape: a sweep is a measurement and carries no
verdicts; the bias point, its amplitude and its calibration live on the
catalog the bias finder returns; and the board plays what the catalog says.
The nonlinear-fit checks that used to read the simulator's model directly read
the same sweeps instead.
"""

import warnings

import numpy as np
import pytest

from rfmux.core.resonators import on_grid
from rfmux.tuning import AmplitudeSchedule, find_bias_points, fit_sweeps
from rfmux.tuning.fits import BIFURCATION_A

#: The schedule the standard array is swept over: half to eight times the
#: simulator's own bias amplitude, five steps. Wide enough that every
#: resonator bifurcates inside it, and narrow enough that the answer is a step.
SCHEDULE = AmplitudeSchedule.multiplicative(0.5, 8.0, 5)
TOP = len(SCHEDULE.steps) - 1


@pytest.fixture(scope="module")
def schedule_sweeps(standard_array_board):
    """One module's multisweep over SCHEDULE, both directions."""
    loop, crs, catalog = standard_array_board
    sweeps = loop.run_until_complete(crs.multisweep(
        catalog, span_hz=100e3, npoints_per_sweep=101, nsamps=10,
        amp=SCHEDULE, sweep_direction=("upward", "downward"), save=False,
    ))
    return sweeps[crs.module[catalog.module].index()]


@pytest.fixture(scope="module")
def bias_report(schedule_sweeps):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return find_bias_points(schedule_sweeps, save=False)


def _fitted_a(schedule_sweeps, name, step):
    p = schedule_sweeps["results"][step]["upward"][name]["fits"]["nonlinear"]["params"]
    return p["a"] if p else float("nan")


@pytest.fixture(scope="module")
def nonlinear_fits(schedule_sweeps):
    report = fit_sweeps(schedule_sweeps, models=("nonlinear",))
    return report


# --- what a sweep is -------------------------------------------------------


def test_a_sweep_entry_is_a_measurement_and_nothing_else(schedule_sweeps, standard_array_board):
    """No verdicts on the entry: bifurcation, bias frequency and calibration
    are the bias finder's to decide, and it writes them on the catalog."""
    _, _, catalog = standard_array_board
    for step in range(TOP + 1):
        for direction in ("upward", "downward"):
            for name in catalog.names():
                entry = schedule_sweeps["results"][step][direction][name]
                assert {"frequencies", "iq_counts", "iq_volts", "sweep_amplitude",
                        "sweep_direction", "original_center_frequency"} <= set(entry)
                assert not {"is_bifurcated", "bias_frequency", "df_calibration",
                            "nonlinear_fit_params", "rotation_tod"} & set(entry)


# --- what bias finding hands back -------------------------------------------


def test_every_resonator_gets_a_bias_point_at_a_schedule_step(bias_report, schedule_sweeps, standard_array_board):
    _, _, catalog = standard_array_board
    assert sorted(f.name for f in bias_report.findings) == sorted(catalog.names())
    assert sorted(bias_report.catalog.names()) == sorted(catalog.names())
    for f in bias_report.findings:
        entry = schedule_sweeps["results"][f.iteration]["upward"][f.name]
        assert f.amplitude == entry["sweep_amplitude"]
        assert entry["frequencies"].min() <= f.frequency_hz <= entry["frequencies"].max()
        point = bias_report.catalog[f.name].bias
        assert point.amplitude == f.amplitude
        assert point.frequency_hz == f.frequency_hz == on_grid(f.frequency_hz)


def test_the_calibration_lives_on_the_bias_point(bias_report):
    """Hz-per-volt derivatives at the tone, read off the sweep the point came
    from, which the point keeps."""
    for name in bias_report.catalog.names():
        point = bias_report.catalog[name].bias
        assert np.isfinite(point.dI_df) and np.isfinite(point.dQ_df)
        assert (point.dI_df, point.dQ_df) != (0.0, 0.0)
        assert point.bias_sweep is not None
        assert point.bias_sweep["sweep_amplitude"] == point.amplitude


def test_bias_finding_on_a_bifurcating_schedule_raises_no_warning(bias_report):
    """Bifurcation inside the schedule is the expected outcome, not a warning;
    the fixture ran the finder with UserWarning turned into an error."""
    assert any(f.bifurcated_at is not None for f in bias_report.findings)


def test_every_resonator_bifurcates_inside_the_schedule(bias_report):
    """The array and schedule are chosen so the finder always has something to
    find. A resonator that never bifurcated would make the loudest step the
    answer by default, which is a flagged outcome, not a measured one."""
    for f in bias_report.findings:
        assert f.bifurcated_at is not None, f.name


# --- the board plays the catalog -------------------------------------------


def test_apply_bias_puts_every_tone_where_the_catalog_says(bias_report, standard_array_board):
    loop, crs, _ = standard_array_board
    catalog = bias_report.catalog
    loop.run_until_complete(crs.apply_bias(catalog))
    nco = loop.run_until_complete(crs.get_nco_frequency(module=catalog.module))
    for name in catalog.names():
        r = catalog[name]
        played = nco + loop.run_until_complete(
            crs.get_frequency(channel=r.channel, module=catalog.module))
        assert played == pytest.approx(r.bias.frequency_hz, abs=1.0), name


# --- the nonlinear fit reads the simulator's physics the right way round -----


def test_the_nonlinearity_rises_with_drive(nonlinear_fits, schedule_sweeps, standard_array_board):
    _, _, catalog = standard_array_board
    assert not nonlinear_fits.failed, nonlinear_fits.failed
    for name in catalog.names():
        assert _fitted_a(schedule_sweeps, name, 0) < _fitted_a(schedule_sweeps, name, TOP - 1), name


def test_the_schedule_brackets_bifurcation_for_every_resonator(nonlinear_fits, schedule_sweeps, standard_array_board):
    """Below the bifurcation value one step from the top, above it at the top:
    the property the schedule is chosen for."""
    _, _, catalog = standard_array_board
    for name in catalog.names():
        assert _fitted_a(schedule_sweeps, name, TOP - 1) < BIFURCATION_A < _fitted_a(schedule_sweeps, name, TOP), name


def test_the_pull_is_downward(nonlinear_fits, schedule_sweeps, standard_array_board):
    """Stored energy lowers the resonance (Swenson et al. 2013 eq. 13): driven
    hard, the transmission minimum sits below the fitted low-power fr, and
    further below it than when driven gently. At the quiet step the pull is a
    few hertz and the 1 kHz sweep grid decides which side the minimum lands
    on, so only the ordering is asserted there."""
    _, _, catalog = standard_array_board

    def pull_hz(name, step):
        entry = schedule_sweeps["results"][step]["upward"][name]
        f_min = entry["frequencies"][np.argmin(np.abs(entry["iq_counts"]))]
        return entry["fits"]["nonlinear"]["params"]["fr"] - f_min

    for name in catalog.names():
        hard = pull_hz(name, TOP - 1)
        assert hard > 1e3, (name, hard)  # more than a grid step below fr
        assert hard > pull_hz(name, 1), name


# --- the tuning record a capture stores ------------------------------------


def test_the_tuning_record_round_trips_through_a_capture_file(
        bias_report, standard_array_board, tmp_path):
    """A capture stores the catalog's bias points and gives them back.

    The one contract the pulse side rests on: what `tuning_rows` builds from
    a catalog is what `catalog_from_tuning` reads back, through the file
    format in between. Every number here was measured on the array.
    """
    h5py = pytest.importorskip("h5py")
    from rfmux.pulse_capture.hdf5 import _store_tuning
    from rfmux.tuning import catalog_from_tuning, tuning_rows

    catalog = bias_report.catalog
    rows = tuning_rows(catalog, nco_frequency_hz=1.0e9, dac_scale_dbm=-2.0,
                       nsamps=10)

    path = tmp_path / "capture.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # every field storable as it is
        with h5py.File(path, "w") as f:
            for channel in rows:
                _store_tuning(f.create_group(f"ch{channel:04d}"), rows, channel)

    from rfmux.pulse_capture.hdf5 import TUNING_JSON_FIELDS
    read_back = {}
    with h5py.File(path) as f:
        for channel in rows:
            grp = f[f"ch{channel:04d}/tuning"]
            row = {k: v for k, v in grp.attrs.items()
                   if k != TUNING_JSON_FIELDS}
            row.update({k: ds[()] for k, ds in grp.items()})
            read_back[channel] = row

    back = catalog_from_tuning(read_back, module=catalog.module)
    assert back.names() == catalog.names()
    for before, after in zip(catalog, back):
        assert after.channel == before.channel
        assert after.bias.frequency_hz == pytest.approx(before.bias.frequency_hz)
        assert after.bias.amplitude == pytest.approx(before.bias.amplitude)
        assert after.bias.df_calibration == pytest.approx(
            before.bias.df_calibration)
        np.testing.assert_allclose(after.bias.bias_sweep["frequencies"],
                                   before.bias.bias_sweep["frequencies"])
        np.testing.assert_allclose(after.bias.bias_sweep["iq_volts"],
                                   before.bias.bias_sweep["iq_volts"])
        assert (after.bias.bias_sweep["sweep_direction"]
                == before.bias.bias_sweep["sweep_direction"])


def test_a_capture_s_tuning_reads_as_the_multisweep_it_came_from(bias_report):
    """The sweeps behind the record come back in the shape a multisweep
    comes in, so a notebook and the GUI read them the same way."""
    from rfmux.tuning import multisweep_from_tuning, tuning_rows

    catalog = bias_report.catalog
    rows = tuning_rows(catalog, nsamps=10)
    container = multisweep_from_tuning(rows, catalog.module,
                                       module_id="crs0000_rmod1")
    block = container["crs0000_rmod1"]
    assert block["measurement"] == "multisweep"
    assert block["call_params"]["nsamps"] == 10

    # One iteration: each resonator at the amplitude it is biased at, which
    # is what the entries say rather than a step they share.
    [(step, by_direction)] = block["results"].items()
    assert step == 0
    drawn = {name: entry for entries in by_direction.values()
             for name, entry in entries.items()}
    assert set(drawn) == set(catalog.names())
    for r in catalog:
        entry = drawn[r.name]
        assert entry["sweep_amplitude"] == pytest.approx(r.bias.amplitude)
        np.testing.assert_allclose(entry["iq_volts"],
                                   r.bias.bias_sweep["iq_volts"])
