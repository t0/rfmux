"""Invariants of the typed resonator model.

The point of these types is that certain states are unrepresentable, so this
asserts the invariants rather than the arithmetic.
"""

import pytest

from rfmux.core.transferfunctions import BASE_FREQUENCY
from rfmux.core.resonators import (
    BiasPoint,
    Resonator,
    ResonatorCatalog,
)

pytestmark = pytest.mark.portable


def a_catalog(freqs=(1.01e9, 1.03e9, 1.05e9), amplitude=0.01, **kwargs) -> ResonatorCatalog:
    """A seeded catalog: every resonator carries a BiasPoint, as all do."""
    return ResonatorCatalog.from_frequencies(freqs, module=2, amplitude=amplitude, **kwargs)


def a_resonator(name="R0001", channel=1, frequency_hz=1.01e9, amplitude=0.01) -> Resonator:
    return Resonator(name, channel=channel, bias=BiasPoint(frequency_hz, amplitude))


# ─── the data model imports clean ─────────────────────────────────────────────


@pytest.mark.xfail(
    reason="rfmux/__init__.py eagerly does `from . import ... tools`, which pulls "
    "PyQt6 and pyqtgraph, so no core module can currently be imported without "
    "the GUI stack. Flips to XPASS when that import becomes lazy.",
)
def test_data_model_imports_without_the_gui():
    """A script author importing the data model should not get Qt with it.

    Checked in a fresh interpreter: by the time this test runs, conftest has
    already imported the GUI, so in-process sys.modules proves nothing.
    """
    import subprocess
    import sys

    code = (
        "import sys; import rfmux.core.resonators; "
        "sys.exit(1 if any(m.startswith(('PyQt6', 'pyqtgraph')) "
        "for m in sys.modules) else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ─── BiasPoint ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("amplitude", [0.0, -0.5, 1.5])
def test_amplitude_must_be_normalized_dac_units(amplitude):
    with pytest.raises(ValueError, match="normalized DAC units"):
        BiasPoint(frequency_hz=1e9, amplitude=amplitude)


def test_negative_amplitude_hints_at_dbm():
    """The common mistake is passing dBm; the error should say so."""
    with pytest.raises(ValueError, match="dbm"):
        BiasPoint(frequency_hz=1e9, amplitude=-40.0)


def test_frequency_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        BiasPoint(frequency_hz=0.0, amplitude=0.01)


def test_bias_point_is_frozen():
    b = BiasPoint(frequency_hz=1e9, amplitude=0.01)
    with pytest.raises(Exception):
        b.frequency_hz = 2e9


def test_df_calibration_is_derived():
    b = BiasPoint(frequency_hz=1e9, amplitude=0.01, dI_df=1e-9, dQ_df=-2e-9)
    assert b.df_calibration == 1.0 / complex(1e-9, -2e-9)


def test_df_calibration_is_none_without_both_slopes():
    assert BiasPoint(frequency_hz=1e9, amplitude=0.01).df_calibration is None
    assert BiasPoint(frequency_hz=1e9, amplitude=0.01, dI_df=1e-9).df_calibration is None


def test_df_calibration_survives_zero_slope():
    """A degenerate fit should give None, not raise ZeroDivisionError."""
    b = BiasPoint(frequency_hz=1e9, amplitude=0.01, dI_df=0.0, dQ_df=0.0)
    assert b.df_calibration is None


def test_power_dbm():
    b = BiasPoint(frequency_hz=1e9, amplitude=0.1)
    assert b.power_dbm(dac_scale_dbm=0.0) == pytest.approx(-20.0)


# ─── quantization ─────────────────────────────────────────────────────────────


def test_quantized_lands_on_the_grid():
    b = BiasPoint(frequency_hz=1_010_000_123.456, amplitude=0.01)
    q = b.quantized()
    assert q.frequency_hz / BASE_FREQUENCY == pytest.approx(
        round(q.frequency_hz / BASE_FREQUENCY)
    )


def test_quantized_moves_less_than_half_a_grid_step():
    for offset in (0.0, 17.0, 149.0, 1234.5):
        b = BiasPoint(frequency_hz=1e9 + offset, amplitude=0.01)
        assert abs(b.quantized().frequency_hz - b.frequency_hz) <= BASE_FREQUENCY / 2


def test_quantized_keeps_calibration():
    """The shift is far smaller than a resonator width, so calibration holds."""
    b = BiasPoint(frequency_hz=1_010_000_123.456, amplitude=0.01, dI_df=1e-9, dQ_df=2e-9)
    q = b.quantized()
    assert q.df_calibration == b.df_calibration
    assert q.amplitude == b.amplitude


def test_quantized_is_idempotent():
    b = BiasPoint(frequency_hz=1_010_000_123.456, amplitude=0.01).quantized()
    assert b.quantized().frequency_hz == pytest.approx(b.frequency_hz)


def test_the_grid_is_base_frequency_and_nothing_finer():
    """One definition of the grid, and it is transferfunctions.BASE_FREQUENCY.

    A half-step offset is the shape of the bug this guards: it sits exactly on
    the finer grid that used to be hardcoded around the tree, so it must still
    be moved. Anything reintroducing a private grid constant fails here.
    """
    n = round(1e9 / BASE_FREQUENCY)
    b = BiasPoint(frequency_hz=(n + 0.5) * BASE_FREQUENCY, amplitude=0.01)
    shift = abs(b.quantized().frequency_hz - b.frequency_hz)
    assert shift == pytest.approx(BASE_FREQUENCY / 2, abs=1e-3)


# ─── a Resonator always has a bias ────────────────────────────────────────────


def test_resonator_cannot_exist_without_a_bias():
    with pytest.raises(TypeError):
        Resonator("R0001", channel=1)


# ─── set_bias: stale calibration must be unrepresentable ─────────────────────


def test_set_bias_moves_the_tone():
    r = a_resonator()
    r.set_bias(frequency_hz=1.0100003e9, amplitude=0.012)
    assert r.bias.frequency_hz == 1.0100003e9
    assert r.bias.amplitude == 0.012


def test_moving_the_frequency_drops_calibration():
    r = a_resonator()
    r.set_bias(dI_df=1e-9, dQ_df=2e-9, bifurcated_at=0.02)
    r.set_bias(frequency_hz=1.0100009e9)
    assert r.bias.df_calibration is None
    assert r.bias.dI_df is None and r.bias.bifurcated_at is None


def test_moving_the_amplitude_drops_calibration():
    r = a_resonator()
    r.set_bias(iq_rotation_deg=12.0)
    r.set_bias(amplitude=0.02)
    assert r.bias.iq_rotation_deg is None
    assert r.bias.frequency_hz == 1.01e9


def test_moving_the_tone_keeps_calibration_passed_explicitly():
    r = a_resonator()
    r.set_bias(dI_df=1e-9, dQ_df=2e-9)
    r.set_bias(frequency_hz=1.02e9, dI_df=3e-9, dQ_df=4e-9)
    assert r.bias.df_calibration == 1.0 / complex(3e-9, 4e-9)


def test_amending_only_calibration_leaves_the_tone_alone():
    r = a_resonator()
    r.set_bias(dI_df=1e-9, dQ_df=2e-9)
    assert r.bias.frequency_hz == 1.01e9
    assert r.bias.amplitude == 0.01
    assert r.bias.df_calibration is not None


# ─── ResonatorCatalog invariants ──────────────────────────────────────────────


def test_from_frequencies_sorts_and_assigns_channels():
    m = ResonatorCatalog.from_frequencies(
        [1.05e9, 1.01e9, 1.03e9], module=2, amplitude=0.01
    )
    assert [r.name for r in m] == ["R0001", "R0002", "R0003"]
    assert [r.channel for r in m] == [1, 2, 3]
    assert [r.bias.frequency_hz for r in m] == [1.01e9, 1.03e9, 1.05e9]


def test_from_frequencies_seeds_a_bias_with_no_calibration():
    """The seed is an operating point, not a measurement."""
    r = a_catalog(amplitude=0.02)["R0001"]
    assert r.bias.amplitude == 0.02
    assert r.bias.df_calibration is None
    assert r.bias.iq_rotation_deg is None and r.bias.bifurcated_at is None


def test_channels_are_one_based():
    with pytest.raises(ValueError, match="1-based"):
        ResonatorCatalog([a_resonator("R1", channel=0)], module=1)


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="Duplicate resonator name"):
        ResonatorCatalog(
            [
                a_resonator("R1", channel=1, frequency_hz=1e9),
                a_resonator("R1", channel=2, frequency_hz=2e9),
            ],
            module=1,
        )


def test_duplicate_channels_rejected():
    with pytest.raises(ValueError, match="Duplicate channel"):
        ResonatorCatalog(
            [
                a_resonator("R1", channel=1, frequency_hz=1e9),
                a_resonator("R2", channel=1, frequency_hz=2e9),
            ],
            module=1,
        )


def test_identical_frequencies_rejected():
    with pytest.raises(ValueError, match="collides"):
        ResonatorCatalog.from_frequencies([1e9, 1e9], module=1, amplitude=0.01)


def test_near_duplicate_frequencies_pass_by_default():
    """Documents the weak default: only exact collisions are caught."""
    m = ResonatorCatalog.from_frequencies([1e9, 1e9 + 1e-6], module=1, amplitude=0.01)
    assert len(m) == 2


def test_min_separation_hz_catches_split_resonances():
    with pytest.raises(ValueError, match="collides"):
        ResonatorCatalog.from_frequencies(
            [1e9, 1e9 + 500.0], module=1, amplitude=0.01, min_separation_hz=1e3
        )


def test_min_separation_hz_is_inclusive():
    """A pair exactly min_separation_hz apart collides, as in find_resonances."""
    with pytest.raises(ValueError, match="collides"):
        ResonatorCatalog.from_frequencies(
            [1e9, 1e9 + 1e3], module=1, amplitude=0.01, min_separation_hz=1e3
        )


def test_min_separation_hz_none_allows_identical_frequencies():
    """The deliberate case: two channels parked on one frequency."""
    m = ResonatorCatalog.from_frequencies(
        [1e9, 1e9], module=1, amplitude=0.01, min_separation_hz=None
    )
    assert [r.bias.frequency_hz for r in m] == [1e9, 1e9]
    assert [r.channel for r in m] == [1, 2]


def test_min_separation_hz_rejects_negative():
    with pytest.raises(ValueError, match="min_separation_hz"):
        ResonatorCatalog.from_frequencies(
            [1e9, 2e9], module=1, amplitude=0.01, min_separation_hz=-1.0
        )


def test_names_must_match_frequency_count():
    with pytest.raises(ValueError, match="2 names for 3 frequencies"):
        ResonatorCatalog.from_frequencies(
            [1e9, 2e9, 3e9], module=1, amplitude=0.01, names=["a", "b"]
        )


def test_names_pair_with_their_own_frequency_regardless_of_order():
    """Parallel lists must stay associated even though channels go by frequency."""
    m = ResonatorCatalog.from_frequencies(
        [1.03e9, 1.01e9], module=1, amplitude=0.01, names=["upper", "lower"]
    )
    assert m["upper"].bias.frequency_hz == 1.03e9
    assert m["lower"].bias.frequency_hz == 1.01e9
    # channels still follow ascending frequency
    assert m["lower"].channel == 1
    assert m["upper"].channel == 2


# ─── lookup and ordering ──────────────────────────────────────────────────────


def test_iteration_is_in_channel_order():
    m = ResonatorCatalog(
        [
            a_resonator("c", channel=3, frequency_hz=3e9),
            a_resonator("a", channel=1, frequency_hz=1e9),
            a_resonator("b", channel=2, frequency_hz=2e9),
        ],
        module=1,
    )
    assert [r.name for r in m] == ["a", "b", "c"]


def test_by_channel_round_trips():
    m = a_catalog()
    for r in m:
        assert m.by_channel(r.channel) is r


def test_by_channel_raises_for_unknown():
    with pytest.raises(KeyError, match="No resonator on channel 99"):
        a_catalog().by_channel(99)


def test_dict_like_access():
    m = a_catalog()
    assert "R0001" in m
    assert "nope" not in m
    assert len(m) == 3
    assert m["R0002"].channel == 2


# ─── copy: the threading rule ─────────────────────────────────────────────────


def test_copy_is_independent():
    m = a_catalog()
    c = m.copy()
    c["R0001"].set_bias(amplitude=0.02)
    c["R0002"].notes["worker"] = True
    assert m["R0001"].bias.amplitude == 0.01
    assert m["R0002"].notes == {}


def test_copy_preserves_catalog_metadata():
    m = ResonatorCatalog.from_frequencies(
        [1e9], module=3, amplitude=0.01, nco_frequency_hz=1.2e9
    )
    c = m.copy()
    assert c.module == 3
    assert c.nco_frequency_hz == 1.2e9


# ─── persistence ──────────────────────────────────────────────────────────────


def test_dict_round_trip():
    m = a_catalog()
    m["R0001"].set_bias(
        frequency_hz=1.0100003e9,
        amplitude=0.012,
        dI_df=1e-9,
        dQ_df=-2e-9,
        iq_rotation_deg=12.5,
        bifurcated_at=0.02,
    )
    m["R0003"].notes["flagged"] = "noisy"
    back = ResonatorCatalog.from_dict(m.to_dict())

    assert [r.name for r in back] == [r.name for r in m]
    assert [r.channel for r in back] == [r.channel for r in m]
    assert back["R0001"].bias == m["R0001"].bias
    assert back["R0001"].bias.df_calibration == m["R0001"].bias.df_calibration
    assert back["R0003"].notes == {"flagged": "noisy"}


def test_to_dict_holds_only_builtins():
    m = a_catalog()
    m["R0001"].set_bias(dI_df=1e-9)
    allowed = (str, int, float, bool, type(None), dict, list)
    d = m.to_dict()

    def walk(o, path="root"):
        assert isinstance(o, allowed), f"{path}: {type(o).__name__} is not a builtin"
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(d)


def test_to_dict_notes_are_copied_not_aliased():
    m = a_catalog()
    d = m.to_dict()
    d["resonators"][0]["notes"]["injected"] = True
    assert m["R0001"].notes == {}


def test_from_dict_rejects_unknown_schema_version():
    d = a_catalog().to_dict()
    d["schema_version"] = 999
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        ResonatorCatalog.from_dict(d)


def test_dict_round_trip_preserves_module_and_nco():
    m = ResonatorCatalog.from_frequencies(
        [1e9], module=4, amplitude=0.01, nco_frequency_hz=1.1e9
    )
    back = ResonatorCatalog.from_dict(m.to_dict())
    assert back.module == 4
    assert back.nco_frequency_hz == 1.1e9


def test_dict_round_trip_preserves_the_separation_rule():
    """Without this a duplicate-frequency catalog saves but cannot be reloaded."""
    m = ResonatorCatalog.from_frequencies(
        [1e9, 1e9], module=1, amplitude=0.01, min_separation_hz=None
    )
    back = ResonatorCatalog.from_dict(m.to_dict())
    assert back.min_separation_hz is None
    assert [r.bias.frequency_hz for r in back] == [1e9, 1e9]


def test_from_dict_defaults_the_separation_rule_for_older_files():
    d = a_catalog().to_dict()
    del d["min_separation_hz"]
    assert ResonatorCatalog.from_dict(d).min_separation_hz == 0.0


# ─── CSV ──────────────────────────────────────────────────────────────────────


def test_csv_round_trip_carries_the_operating_point():
    m = a_catalog()
    m["R0001"].set_bias(frequency_hz=1.0100003e9, amplitude=0.012)
    back = ResonatorCatalog.from_csv(m.to_csv(), module=2)
    assert [r.name for r in back] == ["R0001", "R0002", "R0003"]
    assert back["R0001"].bias.frequency_hz == pytest.approx(1.0100003e9)
    assert back["R0001"].bias.amplitude == pytest.approx(0.012)


def test_csv_columns_may_be_reordered():
    """It is a spreadsheet-editable file, so column order must not matter."""
    text = "channel,bias_amplitude,name,bias_frequency_hz\n1,0.012,R0001,1010000900.0\n"
    m = ResonatorCatalog.from_csv(text, module=2)
    assert m["R0001"].channel == 1
    assert m["R0001"].bias.amplitude == pytest.approx(0.012)


def test_csv_handles_a_name_containing_a_comma():
    m = ResonatorCatalog([a_resonator("A,B", channel=1)], module=1)
    back = ResonatorCatalog.from_csv(m.to_csv(), module=1)
    assert [r.name for r in back] == ["A,B"]


def test_csv_missing_column_is_named():
    with pytest.raises(ValueError, match="missing required column"):
        ResonatorCatalog.from_csv("name,channel\nR1,1\n", module=1)


def test_csv_blank_bias_is_rejected():
    """There is no unbiased resonator, so there is no blank bias cell."""
    text = "name,channel,bias_frequency_hz,bias_amplitude\nR0001,1,1010000900.0,\n"
    with pytest.raises(ValueError, match="are both required"):
        ResonatorCatalog.from_csv(text, module=1)


def test_csv_bad_number_reports_the_line():
    text = "name,channel,bias_frequency_hz,bias_amplitude\nR0001,1,not_a_number,0.01\n"
    with pytest.raises(ValueError, match="line 2"):
        ResonatorCatalog.from_csv(text, module=1)


def test_csv_is_lossy_by_design():
    """Calibration and notes do not survive; to_dict is the faithful path."""
    m = a_catalog()
    m["R0001"].set_bias(dI_df=1e-9, dQ_df=2e-9)
    m["R0001"].notes["x"] = 1
    back = ResonatorCatalog.from_csv(m.to_csv(), module=2)
    assert back["R0001"].bias.df_calibration is None
    assert back["R0001"].notes == {}


# ─── repr ─────────────────────────────────────────────────────────────────────


def test_repr_shows_counts_and_rows():
    text = repr(a_catalog())
    assert "module=2" in text
    assert "3 resonators" in text
    assert "R0001" in text and "R0003" in text
