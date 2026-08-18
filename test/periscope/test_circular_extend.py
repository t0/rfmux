"""
Circular.extend must be indistinguishable from repeated add.

The display path switched to it for throughput; if the two disagree the
plotted trace silently stops matching the samples that arrived.
"""

import numpy as np
import pytest

from rfmux.tools.periscope.utils import Circular

SIZE = 64


def _by_add(values, size=SIZE):
    c = Circular(size)
    for v in values:
        c.add(v)
    return c


def _by_extend(values, chunk, size=SIZE):
    c = Circular(size)
    for i in range(0, len(values), chunk):
        c.extend(values[i:i + chunk])
    return c


@pytest.mark.parametrize("n", [0, 1, 5, SIZE - 1, SIZE, SIZE + 1, 3 * SIZE])
@pytest.mark.parametrize("chunk", [1, 7, SIZE, 1000])
def test_extend_matches_add(n, chunk):
    values = np.arange(n, dtype=float) + 0.5
    a = _by_add(values)
    b = _by_extend(values, chunk)
    assert a.count == b.count
    assert a.ptr == b.ptr
    np.testing.assert_array_equal(a.data(), b.data())


def test_extend_keeps_only_the_newest_when_overrun():
    c = Circular(SIZE)
    c.extend(np.arange(5 * SIZE, dtype=float))
    assert c.count == SIZE
    np.testing.assert_array_equal(
        c.data(), np.arange(4 * SIZE, 5 * SIZE, dtype=float))


def test_extend_wraps_across_the_seam():
    values = np.arange(SIZE + 10, dtype=float)
    a = _by_add(values)
    # A chunk that straddles the wrap point is the case the two-write
    # layout gets wrong if the halves are computed sloppily.
    b = Circular(SIZE)
    b.extend(values[:SIZE - 3])
    b.extend(values[SIZE - 3:])
    np.testing.assert_array_equal(a.data(), b.data())


def test_extend_of_nothing_is_a_no_op():
    c = Circular(SIZE)
    c.extend([])
    c.extend(np.array([]))
    assert c.count == 0 and c.ptr == 0


def test_extend_accepts_nan_timestamps():
    # t_rel is None when a packet has no recent timestamp; the batched
    # path turns that into NaN, as Circular.add always did.
    c = Circular(SIZE)
    c.extend(np.asarray([None, 1.0], dtype=float))
    assert np.isnan(c.data()[0]) and c.data()[1] == 1.0
