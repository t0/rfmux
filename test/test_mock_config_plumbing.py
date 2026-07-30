"""
Mock config plumbing regressions.

Two bugs found while designing the TLS/1-f noise model:

1. Several sites read ``mock_crs.physics_config`` (no underscore) — an
   attribute that does not exist — so the dialog's cache-tuning
   settings and ``get_samples``' scale factor were silently ignored and
   the hardcoded fallbacks always won.
2. ``update_qp_densities_for_time`` is a monotonic ratchet, and with PFB
   enabled its batches advance the clock past the slow frame — so the
   slow emitter must pass ``pulse_time`` explicitly or its samples get
   evaluated at the PFB's (later) time.
"""

import inspect

import pytest


def test_no_dead_physics_config_references():
    """The attribute is _physics_config; a bare 'physics_config' lookup
    silently returns the default forever."""
    from rfmux.mock import crs as crs_mod
    from rfmux.mock import resonator_model as model_mod

    for module in (crs_mod, model_mod):
        src = inspect.getsource(module)
        for pattern in ("'physics_config'", '"physics_config"'):
            for line in src.splitlines():
                if pattern in line and "_physics_config" not in line:
                    pytest.fail(f"{module.__name__}: dead lookup: "
                                f"{line.strip()}")


def test_cache_tuning_reads_the_real_attribute():
    """The convergence-cache steps live in _s21_lc_response_internal;
    they must read _physics_config so dialog settings take effect."""
    from rfmux.mock.resonator_model import MockResonatorModel

    src = inspect.getsource(
        MockResonatorModel._s21_lc_response_internal)
    assert "_physics_config" in src
    assert "cache_freq_step" in src
    for line in src.splitlines():
        if "physics_config" in line:
            assert "_physics_config" in line, line.strip()


def test_slow_emitter_passes_explicit_pulse_time():
    """Otherwise the slow stream inherits the PFB batches' later clock."""
    from rfmux.mock import udp_streamer

    src = inspect.getsource(udp_streamer.MockCRSStreamer._emit_slow_packet)
    assert "pulse_time=t_frame" in src, \
        "slow emitter must pin pulse_time to its own frame time"


def test_ratchet_still_guards_pulse_generation():
    """The monotonic guard stays — it protects the Poisson dt draw from
    being double-counted by out-of-order calls."""
    from rfmux.mock.resonator_model import MockResonatorModel

    src = inspect.getsource(
        MockResonatorModel.update_qp_densities_for_time)
    assert "<= self.last_update_time" in src
