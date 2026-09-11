"""A capture file carries each channel's tuning row: the bias_kids entry
it was captured with, under the channel's ``tuning`` group, read back
as it was handed in."""

import numpy as np
import pytest

pytest.importorskip("h5py")

from rfmux.algorithms.measurement.df_calibration import (  # noqa: E402
    tuning_export, tuning_rows)
from rfmux.pulse_capture.detection import ChannelNoiseStats  # noqa: E402
from rfmux.pulse_capture.hdf5 import (  # noqa: E402
    DualPulseHDF5Writer, PulseHDF5Reader, PulseHDF5Writer)

ROW = {
    "bias_channel": 5, "bias_frequency": 1.2345e9, "sweep_amplitude": 0.01,
    "is_bifurcated": False, "bias_successful": True,
    "sweep_direction": "upward", "recalculation_method_applied": "max-diq",
    "df_calibration": 2.0e6 + 1.0e5j, "df_calibration_source": "measured",
    "df_calibration_fit": np.complex128(1.9e6 + 0.9e5j),
    "applied_rotation_degrees": np.float64(12.5),
    "optimal_phase_degrees": None,
    "frequencies": np.linspace(1.2344e9, 1.2346e9, 7),
    "iq_complex": np.exp(1j * np.linspace(0, 1, 7)),
    "rotation_tod": np.zeros(3, dtype=complex),
    "fit_params": {"fr": 1.2345e9, "Qr": 12000.0, "a": np.float64(0.3),
                   "note": "x", "z": 1 + 2j},
    "labels": ["a", "b"],
    "nco_frequency_hz": 1.0e9,
}


def _same(row, back):
    for name, value in row.items():
        if value is None:
            assert name not in back
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(back[name], value)
        elif name == "fit_params":
            assert back[name] == {"fr": 1.2345e9, "Qr": 12000.0, "a": 0.3,
                                  "note": "x", "z": [1.0, 2.0]}
        else:
            assert back[name] == value and type(back[name]) is type(
                value if not isinstance(value, np.generic) else value.item())


def test_the_row_round_trips_with_its_types(tmp_path):
    path = tmp_path / "t.h5"
    PulseHDF5Writer(path, [5, 9], {5: ChannelNoiseStats()},
                    {"streamer_mode": "slow"}, tuning={5: ROW}).finalize()
    with PulseHDF5Reader(path) as r:
        _same(ROW, r.tuning(5))
        assert r.df_calibration(5) == 2.0e6 + 1.0e5j
        assert r.tuning(9) == {} and r.df_calibration(9) is None
        assert set(r.f["channel_5/tuning"].attrs["json_fields"]) == {
            "fit_params", "labels"}


def test_a_dual_file_holds_the_row_on_both_streams(tmp_path):
    path = tmp_path / "d.h5"
    DualPulseHDF5Writer(path, [1], {"streamer_mode": "both"},
                        tuning={1: ROW}).finalize()
    with PulseHDF5Reader(path) as r:
        for stream in ("slow", "fast"):
            _same(ROW, r.tuning(1, stream))
            assert r.df_calibration(1, stream) == 2.0e6 + 1.0e5j


def test_a_field_that_cannot_be_stored_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "bad.h5"
    row = {"df_calibration": 1.0e6, "odd": object()}
    with pytest.warns(UserWarning, match="'odd' of channel 1 not stored"):
        PulseHDF5Writer(path, [1], {}, {"streamer_mode": "slow"},
                        tuning={1: row}).finalize()
    with PulseHDF5Reader(path) as r:
        assert r.tuning(1) == {"df_calibration": 1.0e6}


@pytest.mark.asyncio
async def test_a_bias_kids_entry_stores_whole(tmp_path):
    """The row bias_kids actually produces, every field of it, with no
    field skipped: a skip only warns, so a capture would silently lack
    the field."""
    import warnings
    from rfmux.algorithms.measurement import bias_kids as bk
    from test.algorithms.test_bias_kids_fits import _Board, _entry

    out = await bk.bias_kids(_Board(), {1: _entry()}, module=1)
    rows = tuning_rows(out, 1.0e9)
    assert set(rows) == {1}
    path = tmp_path / "real.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PulseHDF5Writer(path, [1], {}, {"streamer_mode": "slow"},
                        tuning=rows).finalize()
    with PulseHDF5Reader(path) as r:
        back = r.tuning(1)
        assert r.df_calibration(1) == rows[1]["df_calibration"]
    assert set(back) == {k for k, v in rows[1].items() if v is not None}
    assert back["bias_frequency"] == rows[1]["bias_frequency"]
    assert back["nco_frequency_hz"] == 1.0e9
    assert back["nonlinear_fit_params"] == rows[1]["nonlinear_fit_params"]
    np.testing.assert_array_equal(back["iq_complex"], rows[1]["iq_complex"])


def test_tuning_export_is_the_loaded_view_of_the_rows():
    f = np.linspace(0.999e9, 1.001e9, 5)
    rows = {3: {**ROW, "bias_channel": 3, "frequencies": f,
                "bias_frequency": 1.0e9, "dac_scale_dbm": -2.0},
            7: {**ROW, "bias_channel": 7, "frequencies": f + 1e8,
                "bias_frequency": 1.1e9, "sweep_amplitude": 0.02},
            9: {"df_calibration": 1.0}}         # no sweep: not shown
    out = tuning_export(rows, 2)
    assert out["target_module"] == 2
    params = out["initial_parameters"]
    assert params["module"] == 2 and params["amps"] == [0.01, 0.02]
    assert params["sweep_direction"] == "upward"
    assert params["span_hz"] == pytest.approx(2e6)
    res = params["resonance_frequencies"]
    assert len(res) == 7 and res[2] == 1.0e9 and res[6] == 1.1e9
    assert all(np.isnan(res[k]) for k in (0, 1, 3, 4, 5))
    assert out["dac_scales_used"] == {2: -2.0}
    assert set(out["results_by_detector"]) == {3, 7}
    assert out["results_by_detector"][7] == {0: rows[7]}
    assert out["bias_kids_output"] == {3: rows[3], 7: rows[7]}
    assert out["nco_frequency_hz"] == 1.0e9 and out["noise_data"] is None
    with pytest.raises(ValueError, match="no tuning row with a sweep"):
        tuning_export({9: rows[9]}, 2)


def test_tuning_rows_key_bias_kids_output_by_channel():
    out = {0: {"bias_channel": 3, "df_calibration": 1 + 1j},
           1: {"bias_channel": None},
           2: {"bias_channel": 7, "df_calibration": None}}
    rows = tuning_rows(out, 1.0e9)
    assert rows == {3: {"bias_channel": 3, "df_calibration": 1 + 1j,
                        "nco_frequency_hz": 1.0e9},
                    7: {"bias_channel": 7, "df_calibration": None,
                        "nco_frequency_hz": 1.0e9}}
    assert tuning_rows(out)[3] == {"bias_channel": 3, "df_calibration": 1 + 1j}
    assert tuning_rows(None) == {}
