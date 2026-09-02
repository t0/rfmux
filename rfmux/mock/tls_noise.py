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

        # One-entry memo: the physics helper queries once per TONE with
        # the same timestamp, so consecutive identical t are the common
        # case and re-interpolating them is pure overhead.
        self._memo_t: float | None = None
        self._memo_v: np.ndarray | None = None

        # Latest time anyone has asked about.  History is trimmed
        # relative to THIS, not to the grid's leading edge: extension
        # generates a chunk ahead, so trimming against t_end would
        # discard the region currently being queried and make every
        # lookup clamp to one value (a silently constant "wander").
        self._last_query: float = 0.0

    # ── Generation ────────────────────────────────────────────────

    @property
    def t_end(self) -> float:
        return self._t0 + (len(self._values) - 1) * self.dt

    def _step(self, n_steps: int) -> np.ndarray:
        """Advance the OU states n_steps and return the new rows.

        The recursion ``x_k = a*x_{k-1} + kick*n_k`` is a first-order
        IIR, so the whole block is one ``lfilter`` per pole rather than
        a Python loop over steps — bulk extension (e.g. a large jump in
        requested time) stays cheap.
        """
        from scipy.signal import lfilter

        decay = np.exp(-self.dt / self.taus)
        kick = np.sqrt(self.variances * (1.0 - decay ** 2))
        out = np.zeros((n_steps, self.n_resonators))
        new_state = np.empty_like(self._state)
        for i in range(self.n_poles):
            noise = self._rng.normal(
                0.0, 1.0, (n_steps, self.n_resonators)) * kick[i]
            # zi convention for a = [1, -decay]: zi = decay * x_{-1}
            zi = (decay[i] * self._state[i])[None, :]
            y, _ = lfilter([1.0], [1.0, -decay[i]], noise, axis=0, zi=zi)
            out += y
            new_state[i] = y[-1]
        self._state = new_state
        return out

    #: Minimum grid rows generated per extension.  ``lfilter`` costs
    #: ~10 us per call regardless of length, so extending one row at a
    #: time (what continuous streaming would otherwise do) dominates;
    #: generating a chunk ahead amortises it away.
    CHUNK = 512

    def _extend_to(self, t: float) -> None:
        """Grow the grid so it covers *t* (no-op if already covered)."""
        if t <= self.t_end:
            return
        needed = int(np.ceil((t - self.t_end) / self.dt))
        n_steps = max(needed, self.CHUNK)
        self._values = np.concatenate(
            (self._values, self._step(n_steps)), axis=0)
        self._trim()

    def _trim(self) -> None:
        """Drop rows older than max_history_s BEFORE the last query."""
        cutoff = self._last_query - self.max_history_s
        drop = int((cutoff - self._t0) / self.dt)
        if drop > 0:
            drop = min(drop, len(self._values) - 1)
            self._values = self._values[drop:]
            self._t0 += drop * self.dt

    # ── Evaluation ────────────────────────────────────────────────

    def value_at(self, t: float) -> np.ndarray:
        """Fractional frequency wander per resonator at absolute *t*.

        Queries before the retained history clamp to the oldest sample;
        queries ahead of the grid extend it.  Repeated or out-of-order
        queries always return the same value.
        """
        if self._memo_t is not None and t == self._memo_t:
            return self._memo_v
        if t > self._last_query:
            self._last_query = t
        self._extend_to(t)
        if t <= self._t0:
            value = self._values[0].copy()
        else:
            pos = (t - self._t0) / self.dt
            k = int(pos)
            if k >= len(self._values) - 1:
                value = self._values[-1].copy()
            else:
                frac = pos - k
                value = (self._values[k] * (1.0 - frac)
                         + self._values[k + 1] * frac)
        self._memo_t, self._memo_v = t, value
        return value

    def values_at(self, times: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`value_at` — returns ``(len(times), n_res)``.

        The same arithmetic as value_at, row by row, so a batch and the
        per-sample calls it replaces agree bit for bit; and the same
        memo afterwards, as if the last time had been queried alone.
        """
        times = np.asarray(times, dtype=np.float64)
        if times.size == 0:
            return np.zeros((0, self.n_resonators))
        t_max = float(np.max(times))
        if t_max > self._last_query:
            self._last_query = t_max
        self._extend_to(t_max)
        vals = self._values
        last = len(vals) - 1
        pos = (times - self._t0) / self.dt
        k = np.clip(pos.astype(np.int64), 0, max(last - 1, 0))
        frac = (pos - k)[:, None]
        interp = vals[k] * (1.0 - frac) + vals[np.minimum(k + 1, last)] * frac
        out = np.where((times <= self._t0)[:, None], vals[0][None, :],
                       np.where((pos.astype(np.int64) >= last)[:, None],
                                vals[last][None, :], interp))
        self._memo_t, self._memo_v = float(times[-1]), out[-1].copy()
        return out
