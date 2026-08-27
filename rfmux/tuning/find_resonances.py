"""
Locate resonances in a swept magnitude trace.

Two entry points, and the split between them is the point of the module:

``find_resonances(frequencies, s21, ...)``
    The search itself. Two arrays in, a :class:`ResonanceSearch` out. It knows
    nothing about netanal, the CRS, or files, so it runs equally well on a live
    sweep, a sweep loaded from disk, a simulated trace, or data from another
    instrument.

``find_resonances_in_netanal(netanal, ...)``
    A convenience wrapper over the above: it unpacks what ``crs.take_netanal()``
    returned — one module or several — and hands back results in the same shape.

The search is a dip finder. Convert ``|S21|`` to dB, give the inverted trace to
:func:`scipy.signal.find_peaks` with a prominence floor and Q-derived width
limits, then run the optional rejection passes. Every rejected candidate is
kept, with the reason, in ``ResonanceSearch.rejected``: a finder that quietly
returns fewer resonances than the array has is much harder to debug than one
that says what it dropped and why.

Nothing here imports Qt, the CRS, or matplotlib. This is analysis — Periscope
and the notebooks are callers, and plotting belongs to them.

On the reported numbers
----------------------
``data_exponent`` raises ``|S21|`` to a power before the dB conversion, which
deepens dips relative to the noise and so sharpens the prominence test. The
cost is that widths — and therefore ``q_estimate`` — are measured on the
exponentiated trace and drift from the true half-depth width as the exponent
grows. ``depth_db`` is corrected back to true dB, because raising the magnitude
to a power scales a dB *difference* exactly; the width is not, because the
half-prominence crossing does not transform that simply. Treat ``q_estimate``
as a sorting key rather than a measurement: fitting the resonance is what gives
you Q.

Departures from the previous implementation (``algorithms/measurement/fitting.py``)
----------------------------------------------------------------------------------
* **No ``distance=`` handed to** :func:`~scipy.signal.find_peaks`. See
  :func:`_separation_pass` for what replaced it and why.
* Width limits track frequency sample by sample (``frequencies / min_Q``)
  instead of being derived once from the median frequency, so a sweep spanning
  an octave no longer applies the wrong width window at its ends.
* Bad input raises instead of warning and returning empty lists.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import signal

from ..core.resonators import ResonatorCatalog

__all__ = [
    "ResonanceCandidate",
    "ResonanceSearch",
    "find_resonances",
    "find_resonances_in_netanal",
    "magnitude_db",
]


# ─── Results ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ResonanceCandidate:
    """One dip found in the trace.

    Frozen: a candidate records what the finder measured at one place in one
    sweep. The rejection passes build amended copies rather than mutating it,
    so a candidate can never be half-updated.
    """

    frequency_hz: float
    index: int  # index into the searched trace, for plotting and slicing
    depth_db: float  # prominence, corrected back to true dB
    width_hz: float  # width at half prominence of the exponentiated trace
    q_estimate: float  # frequency_hz / width_hz — rough; see module docstring
    rejected_because: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejected_because is None


@dataclass(slots=True)
class ResonanceSearch:
    """What one search over one trace found.

    Carries the processed trace as well as the candidates, so a caller can plot
    exactly what the finder looked at rather than reconstructing it.
    """

    frequencies_hz: np.ndarray  # the frequency grid that was searched
    magnitude_db: np.ndarray  # the (exponentiated, normalized) trace searched
    candidates: list[ResonanceCandidate]  # accepted, in frequency order
    rejected: list[ResonanceCandidate]  # every candidate a pass threw out
    settings: dict = field(default_factory=dict)  # how this search was run
    label: str | None = None  # e.g. "module 2", for messages

    @property
    def resonance_frequencies_hz(self) -> np.ndarray:
        """The accepted frequencies — what you feed multisweep."""
        return np.array([c.frequency_hz for c in self.candidates])

    def __len__(self) -> int:
        return len(self.candidates)

    def to_catalog(self, module: int, amplitude: float, **kwargs) -> ResonatorCatalog:
        """Seed a :class:`~rfmux.core.resonators.ResonatorCatalog` from the hits.

        The step that turns anonymous dips into named resonators on channels.
        ``amplitude`` is the probe amplitude the catalog's bias points start at;
        remaining keyword arguments go to ``ResonatorCatalog.from_frequencies``
        (``names``, ``nco_frequency_hz``, ``min_separation_hz``).
        """
        return ResonatorCatalog.from_frequencies(
            self.resonance_frequencies_hz, module=module, amplitude=amplitude, **kwargs
        )

    def __repr__(self) -> str:
        where = self.label or "trace"
        span = (
            f"{self.frequencies_hz[0] / 1e6:.3f}–{self.frequencies_hz[-1] / 1e6:.3f} MHz"
            if len(self.frequencies_hz)
            else "empty"
        )
        head = (
            f"ResonanceSearch({where}, {span}): {len(self.candidates)} found, "
            f"{len(self.rejected)} rejected"
        )
        rows = [f"  {'MHz':>12}  {'depth dB':>8}  {'est. Q':>9}"]
        shown = self.candidates[:10]
        for c in shown:
            rows.append(
                f"  {c.frequency_hz / 1e6:>12.6f}  {c.depth_db:>8.2f}  "
                f"{c.q_estimate:>9.3g}"
            )
        if len(self.candidates) > len(shown):
            rows.append(f"  ... {len(self.candidates) - len(shown)} more")
        return "\n".join([head] + rows)


# ─── The search ───────────────────────────────────────────────────────────────


def magnitude_db(s21, data_exponent: float = 1.0, reference: float | None = None):
    """``|S21|`` in dB, optionally raised to a power first.

    ``reference`` is the magnitude that maps to 0 dB; the default is the median
    of the trace, which is a robust stand-in for the off-resonance baseline.
    The choice only shifts the dB axis — prominences and widths are differences
    and so are untouched by it — but a sane 0 dB makes plots and thresholds
    readable.

    Returns ``20 * data_exponent * log10(|S21| / reference)``, which is
    algebraically the dB of ``|S21|**data_exponent`` and avoids overflowing the
    power for large exponents.
    """
    magnitude = np.abs(np.asarray(s21))  # no-op for a magnitude, modulus for I/Q

    positive = magnitude[magnitude > 0]
    if positive.size == 0:
        raise ValueError("|S21| is zero everywhere — there is nothing to search.")
    # Floor exact zeros so the log is finite. Zeros mean a dead channel, not a
    # dip of infinite depth.
    magnitude = np.maximum(magnitude, positive.min())

    if reference is None:
        reference = float(np.median(magnitude))
    if reference <= 0:
        raise ValueError(f"reference={reference}: must be a positive magnitude.")

    return 20.0 * data_exponent * np.log10(magnitude / reference)


def find_resonances(
    frequencies,
    s21,
    *,
    min_dip_depth_db: float = 1.0,
    min_Q: float | None = 1e4,
    max_Q: float | None = 1e7,
    min_separation_hz: float | None = 0.0,
    expected_resonances: int | None = None,
    data_exponent: float = 2.0,
    label: str | None = None,
) -> ResonanceSearch:
    """Find the resonance dips in one swept trace.

    Parameters
    ----------
    frequencies : array_like
        Frequency points in Hz, strictly increasing.
    s21 : array_like
        The response at each frequency: complex I/Q, or a magnitude if that is
        all you kept. ``|S21|`` is taken either way.
    min_dip_depth_db : float, optional
        Prominence floor, in true dB, for a dip to count. Lower it to 0.3–0.5
        for shallow (overcoupled or low-Q) resonators. Default 1.0.
    min_Q : float or None, optional
        Sets the *widest* dip accepted, as ``frequency / min_Q``. Lower it for
        broad resonances. ``None`` leaves dips unbounded above. Default 1e4.
    max_Q : float or None, optional
        Sets the *narrowest* dip accepted, as ``frequency / max_Q`` — this is
        what rejects single-sample noise spikes. ``None`` disables the floor,
        which is rarely what you want. Default 1e7.
    min_separation_hz : float or None, optional
        Separation below which resonances are treated as collided and **all**
        members of the group are cut — not thinned to the deepest, because a
        tone on either member of a collided pair still reads the other. Default
        0.0, which removes only candidates at identical frequencies and so
        touches nothing real. ``None`` skips the pass. See
        :func:`_separation_pass`.
    expected_resonances : int or None, optional
        If given and more candidates survive, keep the ``expected_resonances``
        deepest and reject the rest; if fewer survive, warn. For arrays whose
        count you know.
    data_exponent : float, optional
        Power applied to ``|S21|`` before the dB conversion, to deepen dips
        against the noise. Default 2.0. See the module docstring for what it
        does to the reported width and Q.
    label : str or None, optional
        Name for this trace, used in warnings and in the result's repr.

    Returns
    -------
    ResonanceSearch
        ``.candidates`` are the hits in frequency order,
        ``.resonance_frequencies_hz`` just their frequencies, and
        ``.rejected`` every candidate a pass discarded, with its reason.

    Raises
    ------
    ValueError
        On input that cannot be searched: mismatched or too-short arrays,
        unsorted frequencies, or out-of-range parameters.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    response = np.asarray(s21)
    who = f"{label}: " if label else ""

    # -- validate ------------------------------------------------------------
    if frequencies.ndim != 1:
        raise ValueError(f"{who}frequencies must be 1-D, got shape {frequencies.shape}.")
    if response.shape != frequencies.shape:
        raise ValueError(
            f"{who}frequencies has shape {frequencies.shape} but s21 has "
            f"{response.shape}; they must describe the same points."
        )
    if len(frequencies) < 3:
        raise ValueError(
            f"{who}a dip needs at least 3 points to have a width; got "
            f"{len(frequencies)}."
        )
    if not np.all(np.diff(frequencies) > 0):
        raise ValueError(
            f"{who}frequencies must be strictly increasing. Sort the sweep "
            f"first (np.argsort) — a peak finder reads the trace in order."
        )
    if min_dip_depth_db <= 0:
        raise ValueError(f"{who}min_dip_depth_db={min_dip_depth_db}: must be positive.")
    if data_exponent <= 0:
        raise ValueError(f"{who}data_exponent={data_exponent}: must be positive.")
    for name, q in (("min_Q", min_Q), ("max_Q", max_Q)):
        if q is not None and q <= 0:
            raise ValueError(f"{who}{name}={q}: must be positive or None.")
    if min_Q is not None and max_Q is not None and min_Q >= max_Q:
        raise ValueError(
            f"{who}min_Q={min_Q:g} must be below max_Q={max_Q:g}. min_Q sets the "
            f"widest dip accepted and max_Q the narrowest."
        )
    if min_separation_hz is not None and min_separation_hz < 0:
        raise ValueError(
            f"{who}min_separation_hz={min_separation_hz}: must be a separation "
            f"in Hz. 0 cuts exact duplicates only; None skips the pass."
        )
    if expected_resonances is not None and expected_resonances < 1:
        raise ValueError(
            f"{who}expected_resonances={expected_resonances}: pass None if you "
            f"don't know how many to expect."
        )

    settings = {
        "min_dip_depth_db": min_dip_depth_db,
        "min_Q": min_Q,
        "max_Q": max_Q,
        "min_separation_hz": min_separation_hz,
        "expected_resonances": expected_resonances,
        "data_exponent": data_exponent,
    }

    # -- prepare the trace ---------------------------------------------------
    trace_db = magnitude_db(response, data_exponent=data_exponent)

    # Mean spacing over the sweep. take_netanal dithers each tone by tens of Hz
    # to break up intermodulation products, so the grid is near-uniform rather
    # than exactly uniform; the mean is the right summary of it.
    point_spacing_hz = float(np.mean(np.diff(frequencies)))

    # Width limits in samples, evaluated at every point so they track frequency
    # across the sweep instead of being fixed at its middle.
    min_width_pts = _width_in_points(frequencies, max_Q, point_spacing_hz, floor=0.0)
    max_width_pts = _width_in_points(
        frequencies, min_Q, point_spacing_hz, floor=float(len(frequencies))
    )

    # The prominence floor is applied on the exponentiated trace, so scale the
    # caller's true-dB threshold into those units.
    peaks, properties = signal.find_peaks(
        -trace_db,
        prominence=min_dip_depth_db * data_exponent,
        width=(min_width_pts, max_width_pts),
    )

    candidates = []
    for i, peak in enumerate(peaks):
        width_hz = float(properties["widths"][i]) * point_spacing_hz
        frequency_hz = float(frequencies[peak])
        candidates.append(
            ResonanceCandidate(
                frequency_hz=frequency_hz,
                index=int(peak),
                depth_db=float(properties["prominences"][i]) / data_exponent,
                width_hz=width_hz,
                q_estimate=frequency_hz / width_hz if width_hz > 0 else np.inf,
            )
        )

    # -- rejection passes ----------------------------------------------------
    rejected: list[ResonanceCandidate] = []

    if min_separation_hz is not None:
        candidates, dropped = _separation_pass(candidates, min_separation_hz)
        rejected += dropped

    if expected_resonances is not None:
        candidates, dropped = _count_pass(candidates, expected_resonances, who)
        rejected += dropped

    return ResonanceSearch(
        frequencies_hz=frequencies,
        magnitude_db=trace_db,
        candidates=candidates,
        rejected=rejected,
        settings=settings,
        label=label,
    )


def _width_in_points(frequencies, q, point_spacing_hz: float, floor: float):
    """``frequencies / q`` expressed in samples, or ``floor`` if ``q`` is None."""
    if q is None:
        return np.full(len(frequencies), floor)
    return np.ceil(frequencies / q / point_spacing_hz)


# ─── Rejection passes ─────────────────────────────────────────────────────────
#
# Each takes a candidate list and returns (kept, rejected). Rejected candidates
# come back stamped with a reason, so a caller can always answer "why is this
# resonator missing?" from the result alone.


def _separation_pass(candidates, min_separation_hz: float):
    """Cut collided resonances: a candidate with a neighbour within
    ``min_separation_hz`` is removed, **and so is the neighbour**.

    This is a cut on density, not a thinning. Two resonators too close together
    to operate are not one usable resonator and one nuisance — they are two
    unusable ones. Parking a tone on either member of a collided pair still
    reads out the other's response, so keeping the deeper one would hand
    downstream tuning a detector whose sweep, bias point and timestream are all
    contaminated by a resonator we have deliberately stopped tracking. Removing
    the whole group loses two detectors and keeps the array honest.

    Because every member of a group has a neighbour inside the threshold, this
    is exactly the rule "drop any candidate with a neighbour within
    ``min_separation_hz``" — implemented by looking at each candidate's
    immediate neighbours in frequency, which are necessarily its nearest. It
    chains, and that is intended: A—B 50 kHz and B—C 50 kHz under a 60 kHz
    threshold removes all three, because each of them is in violation.
    Comparison is inclusive, so a pair exactly ``min_separation_hz`` apart is
    cut.

    The default threshold is 0 Hz, which removes only candidates at *identical*
    frequencies — the same point somehow identified twice. That is a
    can't-happen within one call (``find_peaks`` returns distinct samples), so
    the default cut is free and touches nothing real; it exists so a merged or
    concatenated candidate list cannot carry a duplicate through. Set a real
    threshold when you know the separation below which your readout cannot
    operate a detector; pass ``None`` to skip the pass entirely.

    Everything cut comes back in ``ResonanceSearch.rejected`` naming the
    neighbour that caused it, so a missing resonator is traceable rather than
    merely absent. This is also why the pass exists at all instead of
    ``find_peaks(distance=...)``, which the old implementation used: ``distance``
    is a count of samples, so one physical separation meant different things at
    different sweep resolutions; it keeps the tallest member of a close group
    rather than cutting the group; and it discards the loser silently.
    """
    ordered = sorted(candidates, key=lambda c: c.frequency_hz)
    kept: list[ResonanceCandidate] = []
    dropped: list[ResonanceCandidate] = []

    for i, c in enumerate(ordered):
        neighbours = ordered[max(i - 1, 0) : i] + ordered[i + 1 : i + 2]
        nearest = min(
            neighbours, key=lambda n: abs(n.frequency_hz - c.frequency_hz), default=None
        )
        gap = abs(nearest.frequency_hz - c.frequency_hz) if nearest else np.inf

        if gap <= min_separation_hz:
            dropped.append(
                replace(
                    c,
                    rejected_because=(
                        f"collided: {_gap_text(gap)} from the candidate at "
                        f"{nearest.frequency_hz / 1e6:.6f} MHz, within the "
                        f"{_gap_text(min_separation_hz)} separation cut, so both "
                        f"were removed"
                    ),
                )
            )
        else:
            kept.append(c)
    return kept, dropped


def _gap_text(gap_hz: float) -> str:
    """A frequency gap in whichever unit reads sensibly, for messages."""
    return f"{gap_hz:.0f} Hz" if gap_hz < 1e3 else f"{gap_hz / 1e3:.3g} kHz"


def _count_pass(candidates, expected: int, who: str):
    """Keep the ``expected`` deepest candidates; warn if there are too few."""
    if len(candidates) <= expected:
        if len(candidates) < expected:
            warnings.warn(
                f"{who}found {len(candidates)} resonances, expected {expected}. "
                f"Try lowering min_dip_depth_db, or min_Q if the dips are broad."
            )
        return candidates, []

    by_depth = sorted(candidates, key=lambda c: c.depth_db, reverse=True)
    dropped = [
        replace(
            c,
            rejected_because=(
                f"only the {expected} deepest of {len(candidates)} candidates "
                f"were kept (expected_resonances={expected})"
            ),
        )
        for c in by_depth[expected:]
    ]
    warnings.warn(
        f"{who}found {len(candidates)} candidates, kept the {expected} deepest."
    )
    return sorted(by_depth[:expected], key=lambda c: c.frequency_hz), dropped


# ─── Netanal convenience wrapper ──────────────────────────────────────────────


def find_resonances_in_netanal(netanal, *, label: str | None = None, **kwargs):
    """Run :func:`find_resonances` on the output of ``crs.take_netanal()``.

    Unpacks the netanal container and returns results in the same shape:

    * a single result dict → one :class:`ResonanceSearch`
    * a list of them (what ``take_netanal(module=[1, 2])`` returns) → a list
    * a dict keyed by module number → a dict keyed the same way, each result
      labelled by its module

    ``label`` names the trace in warnings; the multi-trace forms derive one per
    entry unless you pass your own. Remaining keyword arguments go straight
    through to :func:`find_resonances`::

        netanal = await crs.take_netanal(module=2, amp=0.001)
        found = find_resonances_in_netanal(netanal, min_dip_depth_db=0.5)
        catalog = found.to_catalog(module=2, amplitude=0.001)

    The search itself does not need this wrapper — it is here so the common case
    is one call, while the algorithm stays free of any measurement format.
    """
    if isinstance(netanal, (list, tuple)):
        return [
            find_resonances_in_netanal(
                entry, label=label or f"sweep {i}", **kwargs
            )
            for i, entry in enumerate(netanal)
        ]

    if not isinstance(netanal, dict):
        raise TypeError(
            f"Expected a netanal result dict (or a list/dict of them), got "
            f"{type(netanal).__name__}."
        )

    # A dict keyed by module number, each value a netanal result. Distinguished
    # from a single result by its keys: take_netanal names its arrays with
    # strings, so integer keys mean modules.
    if netanal and all(isinstance(k, (int, np.integer)) for k in netanal):
        return {
            module: find_resonances_in_netanal(
                entry, label=label or f"module {module}", **kwargs
            )
            for module, entry in netanal.items()
        }

    missing = {"frequencies", "iq_complex"} - set(netanal)
    if missing:
        raise KeyError(
            f"This does not look like a take_netanal result: no "
            f"{' or '.join(sorted(missing))} in it (keys: "
            f"{', '.join(repr(k) for k in netanal)}). Call find_resonances() "
            f"with the two arrays directly if your data is in another shape."
        )
    return find_resonances(
        netanal["frequencies"], netanal["iq_complex"], label=label, **kwargs
    )
