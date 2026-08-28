"""How multisweep decides what to sweep, and at what amplitude.

These cover the resolution step only — the part that turns a catalog, or a bare
list of frequencies, into one normalized list of sweep targets. The measurement
loop below it needs a board and is not exercised here.
"""

import numpy as np
import pytest

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.algorithms.measurement.multisweep import (
    _amplitudes_for_frequencies,
    _resolve_amplitudes,
    _resolve_sweep_targets,
)

pytestmark = pytest.mark.portable


def a_catalog(amplitudes=(0.001, 0.002, 0.004)):
    """Three resonators, deliberately at different bias amplitudes."""
    return ResonatorCatalog(
        [
            Resonator(
                name=f"R{i + 1:04d}",
                channel=i + 1,
                bias=BiasPoint(frequency_hz=1.0e9 + i * 1e6, amplitude=a),
            )
            for i, a in enumerate(amplitudes)
        ],
        module=2,
    )


# ─── amplitude resolution ─────────────────────────────────────────────────────


def test_none_uses_each_resonators_own_bias_amplitude():
    catalog = a_catalog()
    assert _resolve_amplitudes(catalog, None) == {
        "R0001": 0.001,
        "R0002": 0.002,
        "R0003": 0.004,
    }


def test_a_number_overrides_every_resonator():
    catalog = a_catalog()
    assert _resolve_amplitudes(catalog, 0.01) == {
        "R0001": 0.01,
        "R0002": 0.01,
        "R0003": 0.01,
    }
    # …and the catalog is untouched by the override.
    assert [r.bias.amplitude for r in catalog] == [0.001, 0.002, 0.004]


def test_a_mapping_overrides_per_resonator():
    catalog = a_catalog()
    amps = {"R0001": 0.5, "R0002": 0.6, "R0003": 0.7}
    assert _resolve_amplitudes(catalog, amps) == amps


def test_a_partial_mapping_is_an_error_not_a_fallback():
    catalog = a_catalog()
    with pytest.raises(ValueError, match="R0003"):
        _resolve_amplitudes(catalog, {"R0001": 0.5, "R0002": 0.6})


def test_an_unknown_name_is_an_error():
    catalog = a_catalog()
    with pytest.raises(ValueError, match="R9999"):
        _resolve_amplitudes(
            catalog, {"R0001": 0.5, "R0002": 0.6, "R0003": 0.7, "R9999": 0.8}
        )


@pytest.mark.parametrize("sequence", [[0.1, 0.2, 0.3], (0.1, 0.2, 0.3), np.zeros(3)])
def test_a_positional_sequence_is_refused_alongside_a_catalog(sequence):
    """It would depend on catalog ordering, so it is only allowed with a
    caller-supplied frequency list."""
    with pytest.raises(TypeError, match="positional sequence"):
        _resolve_amplitudes(a_catalog(), sequence)


# ─── target resolution ────────────────────────────────────────────────────────


def test_targets_come_from_the_catalog_in_channel_order():
    catalog = a_catalog()
    targets = _resolve_sweep_targets(catalog, None, None)

    assert [t.key for t in targets] == ["R0001", "R0002", "R0003"]
    assert [t.name for t in targets] == ["R0001", "R0002", "R0003"]
    assert [t.channel for t in targets] == [1, 2, 3]
    assert [t.amplitude for t in targets] == [0.001, 0.002, 0.004]


def test_sweep_centres_are_the_catalogs_bias_frequencies():
    catalog = a_catalog()
    targets = _resolve_sweep_targets(catalog, None, None)
    assert [t.center_frequency_hz for t in targets] == [
        r.bias.frequency_hz for r in catalog
    ]


def test_the_catalogs_channels_are_used_not_a_fresh_1_to_n():
    """A resonator's channel is a permanent binding; multisweep honours it."""
    catalog = ResonatorCatalog(
        [
            Resonator(
                name="R0001",
                channel=7,
                bias=BiasPoint(frequency_hz=1.0e9, amplitude=0.001),
            ),
            Resonator(
                name="R0002",
                channel=3,
                bias=BiasPoint(frequency_hz=1.1e9, amplitude=0.001),
            ),
        ],
        module=2,
    )
    targets = _resolve_sweep_targets(catalog, None, None)
    # Channel order, so R0002 (channel 3) leads.
    assert [(t.name, t.channel) for t in targets] == [("R0002", 3), ("R0001", 7)]


def test_neither_input_is_an_error():
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_sweep_targets(None, None, 0.001)


def test_both_inputs_is_an_error():
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_sweep_targets(a_catalog(), [1.0e9], 0.001)


# ─── a bare list of frequencies ───────────────────────────────────────────────


def test_a_frequency_list_is_keyed_and_channelled_by_index():
    targets = _resolve_sweep_targets(None, [1.0e9, 1.1e9], 0.003)

    assert [t.key for t in targets] == [1, 2]
    assert [t.name for t in targets] == [None, None]
    assert [t.channel for t in targets] == [1, 2]
    assert [t.center_frequency_hz for t in targets] == [1.0e9, 1.1e9]
    assert [t.amplitude for t in targets] == [0.003, 0.003]


def test_a_frequency_list_keeps_the_order_it_was_given():
    """Unlike a catalog, this form is not sorted — index 1 is what you passed
    first, whatever its frequency."""
    targets = _resolve_sweep_targets(None, [1.5e9, 1.0e9, 1.2e9], 0.003)
    assert [t.center_frequency_hz for t in targets] == [1.5e9, 1.0e9, 1.2e9]


@pytest.mark.parametrize(
    "amps", [[0.001, 0.002, 0.003], (0.001, 0.002, 0.003), np.array([1e-3, 2e-3, 3e-3])]
)
def test_one_amplitude_per_frequency_pairs_off_positionally(amps):
    targets = _resolve_sweep_targets(None, [1.0e9, 1.1e9, 1.2e9], amps)
    assert [t.amplitude for t in targets] == [0.001, 0.002, 0.003]


def test_a_single_amplitude_applies_to_every_frequency():
    assert _amplitudes_for_frequencies([1.0e9, 1.1e9], 0.004) == [0.004, 0.004]


def test_a_mismatched_amplitude_list_is_an_error():
    with pytest.raises(ValueError, match="2 amplitudes for 3"):
        _amplitudes_for_frequencies([1.0e9, 1.1e9, 1.2e9], [0.001, 0.002])


def test_a_frequency_list_needs_an_amplitude():
    with pytest.raises(ValueError, match="amp is required"):
        _resolve_sweep_targets(None, [1.0e9], None)


def test_a_frequency_list_refuses_a_mapping():
    """There are no names for a mapping to key off."""
    with pytest.raises(TypeError, match="no names"):
        _resolve_sweep_targets(None, [1.0e9, 1.1e9], {"R0001": 0.001})
