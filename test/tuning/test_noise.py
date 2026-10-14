"""Noise-processing helper contracts."""

import copy

import numpy as np
import pytest

from rfmux.core.transferfunctions import spectrum_from_slow_tod
from rfmux.tuning import remove_common_mode, store


pytestmark = pytest.mark.portable


def noise_block(*, include_pfb: bool = True) -> dict:
    samples = np.arange(64)
    common = np.sin(2 * np.pi * samples / 8) + 0.5j * np.cos(
        2 * np.pi * samples / 8)
    residuals = (
        0.1 * np.sin(2 * np.pi * samples / 5),
        0.2j * np.cos(2 * np.pi * samples / 7),
        0.05 * np.sin(2 * np.pi * samples / 3),
    )
    gains = (1 + 0.5j, -0.25 + 2j, 0.7 - 0.3j)
    records = {}
    for index, (name, gain, residual) in enumerate(
            zip(("A", "B", "C"), gains, residuals)):
        iq = 10 * (index + 1) + gain * common + residual
        record = {"slow_data": {"iq_counts": iq}}
        if include_pfb:
            record["pfb_data"] = {"iq_counts": iq[:8].copy()}
        records[name] = record
    return {
        "measurement": "noise",
        "module": 1,
        "call_params": {"nsegments": 4, "spectrum_cutoff": 0.9},
        "results": {
            "info": {
                "iq_units": "counts",
                "reference": "absolute",
                "decimation": 6,
            },
            "resonators": records,
        },
    }


def test_removes_leading_mode_and_stores_an_exactly_reversible_decomposition():
    block = noise_block()
    records = block["results"]["resonators"]
    original = np.stack([
        records[name]["slow_data"]["iq_counts"].copy() for name in records
    ])
    pfb_before = copy.deepcopy(records["A"]["pfb_data"])

    removal = remove_common_mode(block, save=False)

    cleaned = np.stack([
        records[name]["slow_data"]["iq_counts_common_mode_removed"]
        for name in removal["resonator_names"]
    ])
    np.testing.assert_allclose(cleaned + removal["common_mode"], original)
    np.testing.assert_allclose(
        (removal["u"][:, :1] * removal["singular_values"][0])
        @ removal["vh"][:1],
        removal["common_mode"],
    )
    np.testing.assert_allclose(cleaned.mean(axis=1), original.mean(axis=1))
    assert block["common_mode_removal"] is removal
    assert all(record["common_mode_removed"] for record in records.values())
    np.testing.assert_array_equal(records["A"]["pfb_data"]["iq_counts"],
                                  pfb_before["iq_counts"])


def test_saves_psds_for_cleaned_slow_timestreams():
    block = noise_block(include_pfb=False)
    remove_common_mode(block, save=False)

    slow = block["results"]["resonators"]["B"]["slow_data"]
    iq = slow["iq_counts_common_mode_removed"]
    expected = spectrum_from_slow_tod(
        iq.real, iq.imag, 6, scaling="psd", nsegments=4,
        reference="absolute", spectrum_cutoff=0.9,
        input_units="adc_counts")
    for key in ("psd_i", "psd_q", "psd_dual_sideband"):
        np.testing.assert_array_equal(slow[f"{key}_common_mode_removed"],
                                      expected[key])


def test_updates_the_file_from_which_a_module_block_was_loaded(tmp_path):
    module_id = "crs0001_rmod1"
    path = store.save({module_id: noise_block()}, "noise", directory=tmp_path)
    loaded_block = store.load(path)[module_id]

    remove_common_mode(loaded_block, save=True)

    reloaded = store.load(path)[module_id]
    assert reloaded["common_mode_removal"]["resonator_names"] == ["A", "B", "C"]
    assert reloaded["results"]["resonators"]["A"]["common_mode_removed"]


@pytest.mark.parametrize("change, message", [
    (lambda block: block.update(measurement="multisweep"), "measure_noise"),
    (lambda block: block["results"]["resonators"].pop("C")
                   and block["results"]["resonators"].pop("B"),
     "at least two"),
    (lambda block: block["results"]["resonators"]["B"]["slow_data"]
                   .update(iq_counts=np.ones(4)), "same length"),
])
def test_rejects_incompatible_input_without_partial_results(change, message):
    block = noise_block()
    change(block)
    with pytest.raises(ValueError, match=message):
        remove_common_mode(block, save=False)
    assert "common_mode_removal" not in block
    assert not any("common_mode_removed" in record
                   for record in block["results"]["resonators"].values())
