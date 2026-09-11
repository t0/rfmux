"""The channel and module grammar shared by the command lines and the
dialogs.  The plain range grammar is pinned by
test/algorithms/test_channel_selection.py; this file pins the bounds
and the per-module form."""

import pytest

from rfmux.core.channels import parse_channel_spec, parse_module_channels


def test_a_bound_names_the_offending_token():
    assert parse_channel_spec("1-4", max_value=4) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="Channels run 1-1024, so '1-2000'"):
        parse_channel_spec("1-2000", max_value=1024)
    with pytest.raises(ValueError, match="Modules run 1-4"):
        parse_channel_spec("5", name="module", max_value=4)


def test_the_wildcard_can_be_refused():
    assert parse_channel_spec("all") is None
    with pytest.raises(ValueError, match="not a list"):
        parse_channel_spec("all", wildcard=False)


@pytest.mark.parametrize("spec,expected", [
    ("2:1-114,3:1-96", {2: list(range(1, 115)), 3: list(range(1, 97))}),
    ("2:1-3,7,3:5", {2: [1, 2, 3, 7], 3: [5]}),       # a prefix holds
    (["2:1-3", "3:5", "7"], {2: [1, 2, 3], 3: [5, 7]}),  # repeated option
    ("3:5, 2:9-10", {2: [9, 10], 3: [5]}),            # sorted by module
    ("2:3,2:1", {2: [1, 3]}),                         # merged per module
])
def test_module_channels_parse(spec, expected):
    assert parse_module_channels(spec) == expected


@pytest.mark.parametrize("spec,fragment", [
    ("1-3", "has no module"),
    ("1-3,2:4", "has no module"),
    ("x:1", "Could not read the module"),
    ("5:1", "Modules run 1-4"),
    ("0:1", "Modules run 1-4"),
    ("2:all", "not a list"),
    ("2:1-2000", "Channels run 1-1024"),
    ("", "No channels"),
])
def test_module_channels_reject(spec, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_module_channels(spec, max_module=4, max_channel=1024)
