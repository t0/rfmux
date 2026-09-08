"""The small-n dispatch in jit_physics.

numba's parallel=True enters a thread-parallel region at every prange.  The
convergence solver hits three pranges per iteration for ~60 iterations, so at
the handful-of-resonators sizes the mock actually runs, that dispatch overhead
was ~98% of the runtime (measured: 1456 us/call parallel vs 23 us serial).
Both builds are kept and chosen on array length; these tests pin that down.
"""

import numpy as np
import pytest

from rfmux.mr_resonator import jit_physics as jp


def _conv_args(n):
    return (
        1.0e9, 0.001,
        np.full(n, 23.88e-9), np.full(n, 554.9e-6),
        np.full(n, 1.056e-12), np.full(n, 10.18e-15),
        np.full(n, 13.88e-9), np.full(n, 10e-9), np.zeros(n),
        20.0, complex(50.0), 1e-3, 1e-9, 500, 0.1,
    )


def _nqp_args(n):
    return (np.full(n, 52.9),) + tuple(
        np.full(n, v) for v in
        (1e9, 0.12, 3.5e-23, 1.7e10, 1e7, 20e-9, 2e-6, 1e-3, 0.0))


@pytest.mark.parametrize("n", [5, 64])
def test_convergence_builds_agree(n):
    """Serial and parallel builds must be numerically indistinguishable."""
    par = jp._converged_lekid_parameters_par(*_conv_args(n))
    ser = jp._converged_lekid_parameters_ser(*_conv_args(n))
    assert par[3] == ser[3], "iteration counts diverged"
    for a, b in zip(par[:3], ser[:3]):
        assert np.allclose(a, b, rtol=1e-12, atol=0)


def test_nqp_builds_agree():
    args = _nqp_args(7)
    R_p, Lk_p = jp._vectorized_update_params_from_nqp_par(*args)
    R_s, Lk_s = jp._vectorized_update_params_from_nqp_ser(*args)
    assert np.allclose(R_p, R_s, rtol=1e-12, atol=0)
    assert np.allclose(Lk_p, Lk_s, rtol=1e-12, atol=0)


def test_gate_picks_serial_at_mock_sizes(monkeypatch):
    """The default 5-resonator mock must not take the parallel path."""
    calls = []
    for name in ("_converged_lekid_parameters_par",
                 "_converged_lekid_parameters_ser"):
        real = getattr(jp, name)
        monkeypatch.setattr(
            jp, name,
            lambda *a, _n=name, _f=real, **k: (calls.append(_n), _f(*a, **k))[1])

    jp.converged_lekid_parameters(*_conv_args(5))
    assert calls == ["_converged_lekid_parameters_ser"]

    calls.clear()
    jp.converged_lekid_parameters(*_conv_args(jp.PARALLEL_MIN_N))
    assert calls == ["_converged_lekid_parameters_par"]


def test_serial_twin_caches_under_its_own_name():
    """Both builds share a code object; distinct qualnames keep numba's
    on-disk cache from collapsing them into one entry."""
    assert (jp._converged_lekid_parameters_ser.py_func.__qualname__
            != jp._converged_lekid_parameters_par.py_func.__qualname__)
    assert (jp._vectorized_update_params_from_nqp_ser.py_func.__qualname__
            != jp._vectorized_update_params_from_nqp_par.py_func.__qualname__)
