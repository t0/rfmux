"""generate_resonators fills pulse_config from the config's pulse_* keys,
prefix off, so the two key sets must match one to one."""

from rfmux.mock.config import MOCK_DEFAULTS
from rfmux.mock.resonator_model import MockResonatorModel


def test_pulse_config_keys_are_the_pulse_defaults():
    model = MockResonatorModel(None)
    assert {"pulse_" + k for k in model.pulse_config} == {
        k for k in MOCK_DEFAULTS if k.startswith("pulse_")}
