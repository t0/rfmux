"""
The Plot field: which channels the histograms and templates draw, and
how several channels' accumulators combine into one series.
"""

import re

import numpy as np
import pytest

from rfmux.pulse_capture.analysis import (
    combine_histograms, combine_templates, plot_groups, rebin_counts)

CH = [1, 2, 3, 4, 5, 8]


def test_empty_spec_is_one_series_per_channel():
    assert plot_groups("", CH) == [(f"Ch{c}", [c]) for c in CH]


def test_channels_ranges_and_star():
    assert plot_groups("1,2,4", CH) == [("Ch1", [1]), ("Ch2", [2]),
                                        ("Ch4", [4])]
    assert plot_groups("1-5", CH) == [("Ch1-5", [1, 2, 3, 4, 5])]
    assert plot_groups("*", CH) == [("All 6 ch", CH)]
    assert plot_groups(" 2 , 3-4 , all ", CH) == [
        ("Ch2", [2]), ("Ch3-4", [3, 4]), ("All 6 ch", CH)]


def test_absent_channels_are_dropped():
    assert plot_groups("6,7", CH) == []
    assert plot_groups("4-9", CH) == [("Ch4-9", [4, 5, 8])]


@pytest.mark.parametrize("bad, token", [("x", "x"), ("3-1", "3-1"), ("1,a", "a")])
def test_bad_tokens_name_themselves(bad, token):
    with pytest.raises(ValueError, match=re.escape(token)):
        plot_groups(bad, CH)


def test_rebin_conserves_counts_and_is_exact_for_uniform_bins():
    src = np.arange(0.0, 11.0)
    counts = np.array([0, 4, 4, 8, 8, 0, 2, 2, 0, 0], float)
    dst = np.arange(0.0, 11.0, 2.0)
    out = rebin_counts(src, counts, dst)
    assert out.sum() == pytest.approx(counts.sum())
    assert out.tolist() == [4, 12, 8, 4, 0]
    # A shifted grid splits bins proportionally: [0.5, 1.5] takes half
    # of bin 0 (0) and half of bin 1 (4).
    out = rebin_counts(src, counts, src + 0.5)
    assert out[0] == pytest.approx(2.0)
    assert out.sum() == pytest.approx(counts.sum() - 0.5 * counts[-1])


def test_combine_histograms_sums_on_a_shared_grid():
    edges = np.arange(5.0)
    e, c = combine_histograms([edges, edges],
                              [np.array([1, 0, 2, 0.]), np.array([0, 1, 1, 3.])])
    assert np.array_equal(e, edges)
    assert c.tolist() == [1, 1, 3, 3]


def test_combine_histograms_rebins_differing_grids():
    edges = np.arange(5.0)
    e, c = combine_histograms([edges, edges * 2],
                              [np.array([1, 1, 1, 1.]), np.array([2, 2, 2, 2.])])
    assert e[0] == 0.0 and e[-1] == 8.0 and len(e) == 5
    assert c.sum() == pytest.approx(12.0)


def test_combine_templates_is_the_count_weighted_stack():
    means = [np.array([1.0, 1.0, np.nan]), np.array([3.0, 3.0, 5.0])]
    resids = [np.array([0.0, 1.0, np.nan]), np.array([0.0, 1.0, 2.0])]
    counts = [np.array([1, 1, 0]), np.array([3, 3, 4])]
    mean, resid, n = combine_templates(means, resids, counts)
    assert mean.tolist() == [2.5, 2.5, 5.0]
    assert n.tolist() == [4, 4, 4]
    # Bin 0: no within-channel spread, only the offsets about 2.5:
    # sqrt((1*1.5^2 + 3*0.5^2)/4) = sqrt(0.75).
    assert resid[0] == pytest.approx(np.sqrt(0.75))
    # Bin 1 adds unit spread in quadrature.
    assert resid[1] == pytest.approx(np.sqrt(0.75 + 1.0))
    assert resid[2] == pytest.approx(2.0)
