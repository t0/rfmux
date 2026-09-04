"""Applying the Mock Configuration dialog: pulse settings go live,
anything else regenerates the array."""
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.app import (  # noqa: E402
    changed_keys, pulse_mode_kwargs, pulse_only_change)

BASE = {"num_resonances": 100, "pulse_mode": "none", "pulse_period": 0.25,
        "pulse_tau_decay": 25e-3, "bias_amplitude": 0.0016}


def test_turning_pulses_on_is_pulse_only():
    new = dict(BASE, pulse_mode="periodic", pulse_period=0.05)
    changed = changed_keys(BASE, new)
    assert changed == {"pulse_mode", "pulse_period"}
    assert pulse_only_change(changed)


def test_unchanged_is_pulse_only_and_empty():
    assert changed_keys(BASE, dict(BASE)) == set()
    assert pulse_only_change(set())


def test_anything_else_regenerates():
    changed = changed_keys(BASE, dict(BASE, num_resonances=50,
                                      pulse_mode="periodic"))
    assert not pulse_only_change(changed)


def test_pulse_mode_kwargs_strip_the_prefix_and_drop_the_mode():
    new = dict(BASE, pulse_mode="random", pulse_random_amp_mode="uniform")
    assert pulse_mode_kwargs(new) == {
        "period": 0.25, "tau_decay": 25e-3, "random_amp_mode": "uniform"}
