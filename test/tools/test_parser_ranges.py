"""The parser tool's 0-indexed ranges, built from the core grammar."""

import pytest

from rfmux.tools.parser import parse_module_channels, parse_ranges


def test_ranges_are_zero_indexed_with_consecutive_runs_merged():
    assert parse_ranges("1,5-10", 1, 100, "channel") == [range(0, 1), range(4, 10)]
    assert parse_ranges("3,1,2,5", 1, 16, "channel") == [range(0, 3), range(4, 5)]
    with pytest.raises(ValueError, match="run 1-16"):
        parse_ranges("17", 1, 16, "channel")


def test_module_channels_are_zero_indexed_per_module():
    assert parse_module_channels(["2:1-3", "3:5"]) == {1: [range(0, 3)],
                                                       2: [range(4, 5)]}
    with pytest.raises(ValueError, match="has no module"):
        parse_module_channels(["1-3"])
