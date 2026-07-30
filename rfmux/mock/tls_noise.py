"""
TLS-like 1/f frequency noise for the mock resonator model.

Real KIDs show fractional frequency noise with a ``1/f**alpha`` spectrum
(alpha ~ 0.5-1), dominated by two-level systems in the surface
dielectric.  This module synthesises that wander as a **sum of
Ornstein-Uhlenbeck processes** with log-spaced corner frequencies, which
approximates a power law across the covered band at O(n_poles) cost and
without FFT blocks.

Two properties make it usable inside the mock's streaming loops:

**Exact discrete update.**  ``x <- x*exp(-dt/tau) + sqrt(v*(1 -
exp(-2*dt/tau)))*N(0,1)`` is exact for *any* step, not a small-dt
approximation, so the grid spacing is a free choice.

**Pure function of absolute time.**  The realisation is generated once
on a coarse uniform grid and interpolated, so a query at time *t*
returns the same value no matter which stream asks, in what order, or
how many times.  That matters because the mock's slow and PFB emitters
interleave out of order (PFB batches run ahead of their slow frame) and
because the physics helper is invoked once per *(sample x tone)* — a
"step the state on each call" design would advance the process several
times per sample and decorrelate the two streams.  Both streams seeing
the *same* wander at the same absolute time is the physically correct
behaviour: it is one resonator moving.

Spectral construction: an OU process with correlation time ``tau`` and
stationary variance ``v`` has ``S(f) = 4*v*tau / (1 + (2*pi*f*tau)**2)``.
Summed over log-spaced ``tau``, the value at ``f`` is dominated by the
pole with ``tau ~ 1/(2*pi*f)``, so ``S(f) ~ v(tau)*tau``.  Requiring
``S(f) ~ f**-alpha`` with ``tau ~ 1/f`` gives ``v_i ~ tau_i**(alpha-1)``
— for alpha = 1 that is the familiar equal-weight recipe.
"""

from __future__ import annotations

import numpy as np


class TLSNoiseGenerator:
    """Per-resonator fractional frequency wander, evaluable at any time.

    Parameters
    ----------
    n_resonators : int
        Number of independent wander processes (one per resonator).
    fractional_rms : float
        Target RMS of the fractional frequency deviation ``df/f``.
    alpha : float
        Spectral slope: PSD ~ ``1/f**alpha``.  1.0 is pink; TLS is often
        quoted nearer 0.5.
    corner_hz : float
        Upper corner of the power law.  Above it the process rolls off;
        the law spans ``decades`` decades below it.
    decades : float
        Width of the power-law band below ``corner_hz``.
    n_poles : int
        Number of OU processes.  4-8 is plenty; more only smooths the
        residual ripple.
    seed : int, optional
        Seeded for reproducibility (mock runs must be repeatable).
    max_history_s : float
        Grid samples older than this are discarded to bound memory.
    """

    def __init__(
        self,
        n_resonators: int,
        fractional_rms: float = 1e-7,
        alpha: float = 1.0,
        corner_hz: float = 100.0,
        decades: float = 3.0,
        n_poles: int = 6,
        seed: int | None = None,
        max_history_s: float = 120.0,
    ):
        self.n_resonators = int(n_resonators)
        self.fractional_rms = float(fractional_rms)
        self.alpha = float(alpha)
        self.corner_hz = float(corner_hz)
        self.n_poles = max(1, int(n_poles))
        self.max_history_s = float(max_history_s)
        self._rng = np.random.default_rng(seed)

        # Pole time constants: log-spaced across the band
        f_hi = self.corner_hz
        f_lo = self.corner_hz / (10.0 ** float(decades))
        freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), self.n_poles)
        self.taus = 1.0 / (2.0 * np.pi * freqs)

        # v_i ~ tau_i**(alpha - 1), normalised to the requested RMS
        weights = self.taus ** (self.alpha - 1.0)
        weights = weights / np.sum(weights)
        self.variances = weights * (self.fractional_rms ** 2)

        # Grid resolves the fastest pole comfortably
        self.dt = float(np.min(self.taus) / 4.0)

        # OU state at the leading edge: (n_poles, n_resonators)
        self._state = self._rng.normal(
            0.0, 1.0, (self.n_poles, self.n_resonators)
        ) * np.sqrt(self.variances)[:, None]

        # Grid buffer: values[k] is the wander at t0 + k*dt
        self._t0 = 0.0
        self._values = np.sum(self._state, axis=0)[None, :]  # (1, n_res)

    # ── Generation ────────────────────────────────────────────────

    @property
    def t_end(self) -> float:
        return self._t0 + (len(self._values) - 1) * self.dt

    def _step(self, n_steps: int) -> np.ndarray:
        """Advance the OU states n_steps and return the new rows."""
        decay = np.exp(-self.dt / self.taus)[:, None]
        kick = np.sqrt(self.variances[:, None] * (1.0 - decay ** 2))
        out = np.empty((n_steps, self.n_resonators))
        state = self._state
        for k in range(n_steps):
            state = state * decay + kick * self._rng.normal(
                0.0, 1.0, state.shape)
            out[k] = np.sum(state, axis=0)
        self._state = state
        return out

    def _extend_to(self, t: float) -> None:
        """Grow the grid so it covers *t* (no-op if already covered)."""
        if t <= self.t_end:
            return
        n_steps = int(np.ceil((t - self.t_end) / self.dt))
        self._values = np.concatenate(
            (self._values, self._step(n_steps)), axis=0)
        self._trim()

    def _trim(self) -> None:
        keep = int(self.max_history_s / self.dt)
        if len(self._values) > keep > 0:
            drop = len(self._values) - keep
            self._values = self._values[drop:]
            self._t0 += drop * self.dt

    # ── Evaluation ────────────────────────────────────────────────

    def value_at(self, t: float) -> np.ndarray:
        """Fractional frequency wander per resonator at absolute *t*.

        Queries before the retained history clamp to the oldest sample;
        queries ahead of the grid extend it.  Repeated or out-of-order
        queries always return the same value.
        """
        self._extend_to(t)
        if t <= self._t0:
            return self._values[0].copy()
        pos = (t - self._t0) / self.dt
        k = int(pos)
        if k >= len(self._values) - 1:
            return self._values[-1].copy()
        frac = pos - k
        return (self._values[k] * (1.0 - frac)
                + self._values[k + 1] * frac)

    def values_at(self, times: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`value_at` — returns ``(len(times), n_res)``."""
        times = np.asarray(times, dtype=np.float64)
        if times.size == 0:
            return np.zeros((0, self.n_resonators))
        self._extend_to(float(np.max(times)))
        grid_t = self._t0 + np.arange(len(self._values)) * self.dt
        out = np.empty((times.size, self.n_resonators))
        for i in range(self.n_resonators):
            out[:, i] = np.interp(times, grid_t, self._values[:, i])
        return out
