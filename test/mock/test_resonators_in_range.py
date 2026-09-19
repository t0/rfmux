"""Every generated resonator lies inside the requested frequency range,
however far the capacitor variation scatters it."""
import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from rfmux.mock.config import defaults
from rfmux.mock.resonator_model import MockResonatorModel


def _frequencies(**config) -> np.ndarray:
    cfg = dict(defaults(), **config)
    model = MockResonatorModel(SimpleNamespace(_physics_config=cfg))
    with contextlib.redirect_stdout(io.StringIO()):
        model.generate_resonators(cfg["num_resonances"], cfg)
    return np.asarray(model.resonator_frequencies)


@pytest.mark.parametrize("start, end", [(1.0e9, 1.5e9), (1.0e9, 1.02e9)])
def test_resonators_lie_inside_the_requested_range(start, end):
    f = _frequencies(freq_start=start, freq_end=end, num_resonances=6,
                     resonator_random_seed=0)
    assert len(f) == 6
    assert np.all((f >= start) & (f <= end)), f


def test_a_large_capacitor_variation_stays_in_range():
    f = _frequencies(freq_start=1.0e9, freq_end=1.5e9, num_resonances=6,
                     C_variation=0.05, resonator_random_seed=0)
    assert np.all((f >= 1.0e9) & (f <= 1.5e9)), f


def test_a_dense_array_stays_in_range():
    """Three hundred in 100 MHz: the padding is a sixth of a megahertz,
    and the capacitance search has to land inside it."""
    f = _frequencies(freq_start=1.0e9, freq_end=1.1e9, num_resonances=300,
                     resonator_random_seed=0)
    assert len(f) == 300
    assert np.all((f >= 1.0e9) & (f <= 1.1e9))


def test_the_resonators_still_span_the_range():
    f = _frequencies(freq_start=1.0e9, freq_end=1.5e9, num_resonances=6,
                     resonator_random_seed=0)
    assert f.min() < 1.05e9 and f.max() > 1.45e9
