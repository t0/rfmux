"""Bringing a running mock to a new configuration, headlessly: pulse
settings go live, anything else regenerates the array."""
import asyncio

from rfmux.mock.helpers import (
    apply_mock_config, config_changes, pulse_mode_kwargs, pulse_only_change)

BASE = {"num_resonances": 100, "pulse_mode": "none", "pulse_period": 0.25,
        "pulse_tau_decay": 25e-3, "bias_amplitude": 0.0016,
        "resonator_random_seed": 7}


class _FakeCRS:
    def __init__(self):
        self.calls = []

    async def set_pulse_mode(self, mode, **kwargs):
        self.calls.append(("set_pulse_mode", mode, kwargs))

    async def generate_resonators(self, config):
        """The server's shape: the count and the resonance frequencies."""
        self.calls.append(("generate_resonators", dict(config)))
        n = config["num_resonances"]
        return n, [1.0e9] * n


def test_turning_pulses_on_is_pulse_only():
    new = dict(BASE, pulse_mode="periodic", pulse_period=0.05)
    changed = config_changes(BASE, new)
    assert changed == {"pulse_mode", "pulse_period"}
    assert pulse_only_change(changed)


def test_keys_the_dialog_does_not_carry_are_not_changes():
    """The dialog hands back None for what it does not show, and floats
    that went through a text field a rounding step off."""
    from_dialog = dict(BASE, pulse_period=0.05, scale_factor=None,
                       cache_qp_step=None, bias_amplitude=0.0016 * (1 + 1e-12))
    assert config_changes(BASE, from_dialog) == {"pulse_period"}


def test_pulse_only_apply_uses_the_merged_config():
    """set_pulse_mode must see the mode and parameters the dialog did
    not carry, from the configuration in force."""
    crs = _FakeCRS()
    prev = dict(BASE, pulse_mode="periodic")
    from_dialog = dict(prev, pulse_period=0.05, pulse_mode=None)
    outcome, _ = asyncio.run(apply_mock_config(crs, from_dialog, prev))
    assert outcome == "pulses"
    [(call, mode, kwargs)] = crs.calls
    assert (call, mode) == ("set_pulse_mode", "periodic")
    assert kwargs["period"] == 0.05 and kwargs["tau_decay"] == 25e-3


def test_anything_else_regenerates():
    changed = config_changes(BASE, dict(BASE, num_resonances=50,
                                        pulse_mode="periodic"))
    assert not pulse_only_change(changed)


def test_pulse_mode_kwargs_strip_the_prefix_and_drop_the_mode():
    new = dict(BASE, pulse_mode="random", pulse_random_amp_mode="uniform")
    assert pulse_mode_kwargs(new) == {
        "period": 0.25, "tau_decay": 25e-3, "random_amp_mode": "uniform"}


def test_apply_takes_pulses_live_and_regenerates_the_rest():
    crs = _FakeCRS()
    new = dict(BASE, pulse_mode="periodic", pulse_period=0.05)
    assert asyncio.run(apply_mock_config(crs, new, BASE)) == ("pulses", None)
    [(call, mode, kwargs)] = crs.calls
    assert (call, mode) == ("set_pulse_mode", "periodic")
    assert kwargs["period"] == 0.05 and kwargs["tau_decay"] == 25e-3

    crs = _FakeCRS()
    assert asyncio.run(apply_mock_config(crs, dict(BASE), BASE)) == \
        ("unchanged", None)
    assert crs.calls == []

    crs = _FakeCRS()
    new = dict(BASE, num_resonances=50)
    assert asyncio.run(apply_mock_config(crs, new, BASE)) == \
        ("regenerated", 50)
    assert crs.calls[0][0] == "generate_resonators"


def test_no_previous_regenerates_and_pins_a_seed():
    crs = _FakeCRS()
    cfg = dict(BASE, resonator_random_seed=None)
    outcome, count = asyncio.run(apply_mock_config(crs, cfg))
    assert outcome == "regenerated" and count == 100
    assert cfg["resonator_random_seed"] is not None
    assert crs.calls[0][1]["resonator_random_seed"] == cfg["resonator_random_seed"]


def test_clearing_the_seed_regenerates_with_a_fresh_pinned_seed():
    """An emptied seed field asks for a new random array: it is a
    change against the pinned seed, and the seed the server builds
    from is pinned back into the config."""
    crs = _FakeCRS()
    cfg = dict(BASE, resonator_random_seed=None)
    assert config_changes(BASE, cfg) == {"resonator_random_seed"}
    outcome, count = asyncio.run(apply_mock_config(crs, cfg, BASE))
    assert (outcome, count) == ("regenerated", 100)
    seed = cfg["resonator_random_seed"]
    assert seed is not None and seed != BASE["resonator_random_seed"]
    assert crs.calls[0][1]["resonator_random_seed"] == seed


def test_regeneration_sends_the_normalized_config():
    """The server gets the dialog's values through apply_overrides'
    clamps, on top of the configuration in force."""
    crs = _FakeCRS()
    cfg = dict(BASE, tls_alpha=5.0)
    del cfg["bias_amplitude"]
    asyncio.run(apply_mock_config(crs, cfg, BASE))
    sent = crs.calls[0][1]
    assert sent["tls_alpha"] == 2.0
    assert sent["bias_amplitude"] == BASE["bias_amplitude"]


def test_pulse_settings_go_live_normalized():
    crs = _FakeCRS()
    new = dict(BASE, pulse_mode="random", pulse_random_tau_min=-1.0)
    assert asyncio.run(apply_mock_config(crs, new, BASE)) == ("pulses", None)
    assert crs.calls[0][2]["random_tau_min"] == 1e-6
