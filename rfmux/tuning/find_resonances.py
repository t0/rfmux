"""
Locate resonances in a swept magnitude trace.

Two entry points, and the split between them is the point of the module:

``find_resonances(frequencies, s21, ...)``
    The search itself. Two arrays in, a :class:`ResonanceSearch` out. It knows
    nothing about netanal, the CRS, or files, so it runs equally well on a live
    sweep, a sweep loaded from disk, a simulated trace, or data from another
    instrument.

``find_resonances_in_netanal(module_netanal, ...)``
    A convenience wrapper over the above: it unpacks one module's output out of
    what ``crs.take_netanal()`` returned — ``netanal[module_id]``, one module at
    a time as everywhere else in this package — and searches the trace in it. It
    also writes the search into that output, beside the trace it searched, so
    saving is an update to the netanal file rather than a second file to keep in
    step with it.

A third function comes at the same question from the other end, once the array
has been swept properly:

``find_sweeps_with_nearby_resonances(module_sweeps, min_separation_hz)``
    Which multisweep sections turned out to hold more than one dip. A netanal
    search can only separate what one coarse trace resolved; the finer sweep
    around each candidate is where a collided pair hiding inside a single dip
    finally comes apart, and this names the ones to cull.

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
``depth_db`` is a prominence in dB and means what it says. ``width_hz`` is the
width at half that prominence *on the dB trace*, which is not a half-power
width, so ``q_estimate = frequency / width`` is the rough figure the ``min_Q`` /
``max_Q`` window screens on and not a measurement. Fitting the resonance is what
gives you Q, and that is multisweep's business.

Departures from the previous implementation (``algorithms/measurement/fitting.py``)
----------------------------------------------------------------------------------
* **No ``distance=`` handed to** :func:`~scipy.signal.find_peaks`. See
  :func:`_separation_pass` for what replaced it and why.
* Width limits track frequency sample by sample (``frequencies / min_Q``)
  instead of being derived once from the median frequency, so a sweep spanning
  an octave no longer applies the wrong width window at its ends.
* Bad input raises instead of warning and returning empty lists.
* No ``data_exponent``. Raising ``|S21|`` to a power is a *multiplier* in dB, so
  it scaled dips and noise together; with the prominence threshold scaled to
  match, and half-prominence crossings being scale-invariant, it could not
  change a single candidate, width or depth. It was inert, and inert knobs
  invite tuning. The previous implementation left the threshold unscaled, which
  made the exponent a disguised way of dividing ``min_dip_depth_db`` by it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import signal

from ..core.resonators import ResonatorCatalog
from . import store
from .store import plain
from .sweep_results import _refuse_container

__all__ = [
    "ResonanceCandidate",
    "ResonanceSearch",
    "find_resonances",
    "find_resonances_in_netanal",
    "find_sweeps_with_nearby_resonances",
    "netanal_trace",
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
    depth_db: float  # prominence in dB against the local baseline
    width_hz: float  # width at half that prominence, on the dB trace
    q_estimate: float  # frequency_hz / width_hz — rough; see module docstring
    rejected_because: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejected_because is None

    def to_dict(self) -> dict:
        """Plain builtins only — files never contain these classes.

        No version of its own: a candidate is only ever written as part of a
        :class:`ResonanceSearch`, and one stamp on the thing that becomes a file
        is the version that matters.
        """
        return {
            "frequency_hz": float(self.frequency_hz),
            "index": int(self.index),
            "depth_db": float(self.depth_db),
            "width_hz": float(self.width_hz),
            "q_estimate": float(self.q_estimate),
            "rejected_because": self.rejected_because,
        }

    @classmethod
    def from_dict(cls, d) -> ResonanceCandidate:
        return cls(
            frequency_hz=float(d["frequency_hz"]),
            index=int(d["index"]),
            depth_db=float(d["depth_db"]),
            width_hz=float(d["width_hz"]),
            q_estimate=float(d["q_estimate"]),
            rejected_because=d.get("rejected_because"),
        )


@dataclass(slots=True)
class ResonanceSearch:
    """What one search over one trace found.

    Carries the processed trace as well as the candidates, so a caller can plot
    exactly what the finder looked at rather than reconstructing it.
    """

    # Stamped into to_dict output and required exactly by from_dict, so a file
    # from another version of this module fails loudly rather than being half
    # understood. Bump whenever the dict shape changes in a way from_dict
    # cannot absorb.
    SCHEMA_VERSION = 1

    frequencies_hz: np.ndarray  # the frequency grid that was searched
    magnitude_db: np.ndarray  # the normalized dB trace that was searched
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
        (``names``, ``min_separation_hz``).
        """
        return ResonatorCatalog.from_frequencies(
            self.resonance_frequencies_hz, module=module, amplitude=amplitude, **kwargs
        )

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain builtins and ndarrays — files never contain these classes.

        The searched trace stays an ndarray. It is measured data, and a
        five-thousand-point list of Python floats is a worse file than the array
        it came from; numpy is not what "readable without rfmux" was about.
        """
        return {
            "schema_version": self.SCHEMA_VERSION,
            "frequencies_hz": np.asarray(self.frequencies_hz),
            "magnitude_db": np.asarray(self.magnitude_db),
            "candidates": [c.to_dict() for c in self.candidates],
            "rejected": [c.to_dict() for c in self.rejected],
            "settings": plain(self.settings),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d) -> ResonanceSearch:
        version = d.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={version!r}, expected {cls.SCHEMA_VERSION}: "
                f"this dict was written by a different version of "
                f"ResonanceSearch."
            )
        return cls(
            frequencies_hz=np.asarray(d["frequencies_hz"]),
            magnitude_db=np.asarray(d["magnitude_db"]),
            candidates=[ResonanceCandidate.from_dict(c) for c in d["candidates"]],
            rejected=[ResonanceCandidate.from_dict(c) for c in d["rejected"]],
            settings=d.get("settings", {}),
            label=d.get("label"),
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


def magnitude_db(s21, reference: float | None = None):
    """``|S21|`` in dB: ``20 * log10(|S21| / reference)``.

    ``reference`` is the magnitude that maps to 0 dB; the default is the median
    of the trace, which is a robust stand-in for the off-resonance baseline.
    The choice only shifts the dB axis — prominences and widths are differences
    and so are untouched by it — but a sane 0 dB makes plots and thresholds
    readable.
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

    return 20.0 * np.log10(magnitude / reference)


def find_resonances(
    frequencies,
    s21,
    *,
    min_dip_depth_db: float = 1.0,
    min_Q: float | None = 1e4,
    max_Q: float | None = 1e7,
    min_separation_hz: float | None = 0.0,
    require_isolation: bool = True,
    expected_resonances: int | None = None,
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
        Separation below which resonances are too close to each other. What
        happens to them depends on ``require_isolation``. Default 0.0, which
        acts only on candidates at identical frequencies and so touches nothing
        real. ``None`` skips the pass.
    require_isolation : bool, optional
        What ``min_separation_hz`` promises. ``True`` (the default) cuts **all**
        members of a close group — not thinned to the deepest, because a tone on
        either member of a collided pair still reads the other — so every
        resonance returned is one nothing else is near; see
        :func:`_separation_pass`. ``False`` keeps the deepest member of each
        group and rejects the rest, the way ``find_peaks(distance=...)`` used
        to: the returned list obeys the separation, but a survivor can still
        have a real resonance beside it, namely the one that was rejected; see
        :func:`_thinning_pass`. Either way what was cut is in ``.rejected``
        with its reason.
    expected_resonances : int or None, optional
        If given and more candidates survive, keep the ``expected_resonances``
        deepest and reject the rest; if fewer survive, warn. For arrays whose
        count you know.
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
        "require_isolation": require_isolation,
        "expected_resonances": expected_resonances,
    }

    # -- prepare the trace ---------------------------------------------------
    trace_db = magnitude_db(response)

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

    peaks, properties = signal.find_peaks(
        -trace_db,
        prominence=min_dip_depth_db,
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
                depth_db=float(properties["prominences"][i]),
                width_hz=width_hz,
                q_estimate=frequency_hz / width_hz if width_hz > 0 else np.inf,
            )
        )

    # -- rejection passes ----------------------------------------------------
    rejected: list[ResonanceCandidate] = []

    if min_separation_hz is not None:
        if require_isolation:
            candidates, dropped = _separation_pass(candidates, min_separation_hz)
        else:
            candidates, dropped = _thinning_pass(candidates, min_separation_hz, trace_db)
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


def _thinning_pass(candidates, min_separation_hz: float, trace_db):
    """Thin close resonances: of any group within ``min_separation_hz`` of each
    other, keep the deepest dip and reject the rest.

    The permissive alternative to :func:`_separation_pass`, chosen with
    ``require_isolation=False``. It is what ``find_peaks(distance=...)`` did —
    the deepest peak claims its neighbourhood, then the next deepest of those
    left, and so on — expressed in Hz rather than samples, and with every
    loser returned in ``rejected`` naming the survivor that displaced it
    instead of vanishing. A survivor may therefore still have a real resonance
    beside it; that is the trade this mode makes. Comparison is inclusive, as
    in :func:`_separation_pass`.
    """
    by_depth = sorted(candidates, key=lambda c: float(trace_db[c.index]))  # deepest first
    kept: list[ResonanceCandidate] = []
    dropped: list[ResonanceCandidate] = []
    for c in by_depth:
        winner = next(
            (k for k in kept if abs(k.frequency_hz - c.frequency_hz) <= min_separation_hz),
            None,
        )
        if winner is None:
            kept.append(c)
        else:
            gap = abs(winner.frequency_hz - c.frequency_hz)
            dropped.append(
                replace(
                    c,
                    rejected_because=(
                        f"thinned: {_gap_text(gap)} from the deeper candidate at "
                        f"{winner.frequency_hz / 1e6:.6f} MHz, within the "
                        f"{_gap_text(min_separation_hz)} separation, which was "
                        f"kept instead"
                    ),
                )
            )
    kept.sort(key=lambda c: c.frequency_hz)
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


def netanal_trace(module_netanal) -> dict:
    """The one trace inside one module's netanal output.

    A netanal is one amplitude sweeping upward in frequency, so its arrays live
    at ``results[0]["upward"]`` — the same walk a sweep takes to reach a
    section, one level shorter because there is nothing to key by name. This is
    that walk, with an error worth reading when the output is not a netanal's.

    Args:
        module_netanal: one module's netanal output, ``netanal[module_id]``.

    Returns:
        dict: ``frequencies``, ``iq_counts``, ``iq_volts``, ``sweep_amplitude``
        and ``sweep_direction``.
    """
    _refuse_container(module_netanal, what="netanal", variable="netanal")

    if not isinstance(module_netanal, dict) or "results" not in module_netanal:
        got = (
            list(module_netanal) if isinstance(module_netanal, dict)
            else type(module_netanal).__name__
        )
        raise TypeError(
            f"Expected one module's netanal — what take_netanal returned, "
            f"indexed by module — got {got}."
        )
    # Strictly, rather than treating a missing 'measurement' as permission: a
    # sweep's output walks to this same depth and holds {name: section} there,
    # so a tolerant check would hand back a dict of sections dressed as a trace.
    if module_netanal.get("measurement") != "netanal":
        raise TypeError(
            f"This is a "
            f"{module_netanal.get('measurement') or 'unlabelled'} result, "
            f"not a netanal. A sweep's results are keyed by resonator — fit "
            f"them with rfmux.tuning.fit_sweeps() or read them out with "
            f"rfmux.tuning.sweep_results."
        )

    for by_direction in module_netanal["results"].values():
        for trace in by_direction.values():
            return trace
    raise ValueError("This netanal has no trace in it — nothing was measured.")


def find_resonances_in_netanal(
    module_netanal, *, label: str | None = None, save=None, **kwargs
) -> ResonanceSearch:
    """Run :func:`find_resonances` on **one module's** netanal output.

    ::

        netanal = await crs.take_netanal(module=2, amp=0.001)
        module_netanal = netanal[crs.module[2].index()]   # "crs0042_rmod2"

        search = find_resonances_in_netanal(module_netanal, min_dip_depth_db=0.5)
        catalog = search.to_catalog(module=2, amplitude=0.001)

    One module at a time, as :func:`~rfmux.tuning.fits.fit_sweeps` and
    :func:`~rfmux.tuning.bias.find_bias_points` take one module at a time; the
    whole dict keyed by module is refused with a message naming the modules it
    holds. How deep a dip has to be and how wide it may get are properties of
    the band a module looks at and the resonators sitting in it, so eight
    modules are eight decisions. Making the caller write the loop is what keeps
    those decisions visible::

        searches = {
            module_id: find_resonances_in_netanal(m, min_dip_depth_db=1.0)
            for module_id, m in netanal.items()
        }

    The search also goes *into* that module's output, beside the trace it
    searched, as ``resonance_search`` — the same move
    :func:`~rfmux.tuning.fits.fit_sweeps` makes with its fits, and for the same
    reason: a search is about one trace, so it belongs with that trace rather
    than in a file of its own that has to be kept paired with it. A trace has
    one search, so searching the same netanal again replaces what the last call
    left. Reading it back is an index and a ``from_dict``, the same as any
    stored class::

        trace = netanal_trace(module_netanal)
        search = ResonanceSearch.from_dict(trace["resonance_search"])

    ``label`` names the trace in warnings, and defaults to the module this
    output came from, so a script working through eight of them says which one
    complained. It is also the name a *first* save puts on the file — a netanal
    that has been saved already keeps the name it has, and the derived module
    label is never written to a file, only used in messages.

    ``save`` writes the netanal back out. What changed on disk is the netanal,
    so this updates the file it came from — the whole file, other modules
    included, when this output was saved as part of a container. A netanal that
    was never saved gets a new file. Defaults to
    ``rfmux.tuning.store.autosave_enabled()``.

    Remaining keyword arguments go straight through to :func:`find_resonances`.

    The search itself does not need this wrapper — it is here so the common case
    is one call, while the algorithm stays free of any measurement format.
    """
    # Every shape but one module's netanal is refused in here, container first.
    trace = netanal_trace(module_netanal)

    module = module_netanal.get("module")
    search = find_resonances(
        trace["frequencies"],
        trace["iq_counts"],
        label=label or (f"module {module}" if module is not None else None),
        **kwargs,
    )
    # In place, before the save: the netanal is what gets written, and the
    # search is now part of it.
    #
    # The whole to_dict, searched trace included, so what goes into the netanal
    # is a complete ResonanceSearch dict that ResonanceSearch.from_dict reads
    # with no help. It looks like it duplicates the arrays beside it and mostly
    # does not: the search's frequencies *are* the trace's array, which pickle
    # stores once and restores shared, so the file grows by the dB copy of the
    # magnitudes and nothing else. Cheap enough not to trade for a block that
    # only means something to a reader who knows to put the arrays back.
    trace["resonance_search"] = search.to_dict()
    # label, not the derived one: a module number belongs in a warning, not in
    # the name of a file that may hold seven other modules. No module= either —
    # a netanal's output records the module it measured.
    store.maybe_save(module_netanal, "netanal", save=save, label=label)
    return search


# ─── Collided resonances inside a multisweep section ──────────────────────────


def find_sweeps_with_nearby_resonances(
    module_sweeps,
    min_separation_hz: float,
    *,
    min_prominence_db: float = 1.0,
    min_dip_spacing_hz: float = 1e3,
    iq_key: str = "iq_counts",
    iteration: int | None = None,
    direction: str | None = None,
) -> list[str]:
    """Which multisweep sections swept a second resonance alongside their own.

    The netanal search this module starts with resolves only what one coarse
    trace could. A pair that sat inside a single dip there gets its own fine
    sweep afterwards, and comes apart in it — so this is the same collision cut
    as :func:`_separation_pass`, run again on better data.

    A section holding two dips is a section whose fit, bias frequency and bias
    amplitude are all answering the wrong question, so what to do with the names
    it returns is drop them: ``for name in culled: catalog.remove(name)``. As
    there, both members of a collided pair go — parking a tone on either one
    still reads the other.

    Parameters
    ----------
    module_sweeps : dict
        One module's sweep result — ``sweeps[module_id]``, what ``multisweep``
        returned for a single module.
    min_separation_hz : float
        The separation two dips have to clear to be allowed as distinct
        resonances; anything closer is a collision and the section is culled.
        Comparison is inclusive, as in :func:`_separation_pass`: a pair exactly
        this far apart is cut. Pass ``float("inf")`` for "any second dip
        anywhere in the sweep window disqualifies the section".
    min_prominence_db : float, optional
        Prominence floor, in dB, for a dip to count as a dip at all. Raise it to
        stop noise wiggles counting. Default 1.0.
    min_dip_spacing_hz : float, optional
        How close two minima may be and still be found as two dips, rather than
        one dip split into several. This is the finder's resolving power, so it
        also floors what *min_separation_hz* can act on: a pair closer together
        than this is never seen as a pair, and so is never culled. Keep it well
        below *min_separation_hz*. Default 1e3.
    iq_key : str, optional
        Which of the sweep's arrays to read, ``"iq_counts"`` or ``"iq_volts"``.
        Either answers the question — the dB magnitude only shifts by a constant
        between them. Default ``"iq_counts"``.
    iteration : int or None, optional
        Look only at this amplitude iteration. The default, ``None``, looks at
        all of them and culls a name if *any* amplitude shows the collision. A
        bifurcated sweep at the top of an amplitude ladder can occasionally
        split into two minima, so pass ``0`` if that turns up as a false hit.
    direction : str or None, optional
        Look only at this sweep direction. Default ``None``, both.

    Returns
    -------
    list of str
        The section names to cull, in the order they were swept. Empty if
        nothing collided.

    Raises
    ------
    ValueError
        On out-of-range parameters, or if *iteration* or *direction* selected
        nothing — a typo there would otherwise report an array with no
        collisions in it, which is the dangerous direction to be wrong in.
    """
    # -- validate ------------------------------------------------------------
    if min_separation_hz < 0:
        raise ValueError(
            f"min_separation_hz={min_separation_hz}: must be a separation in "
            f"Hz (>= 0). Use float('inf') to cull on any second dip at all."
        )
    if min_prominence_db <= 0:
        raise ValueError(f"min_prominence_db={min_prominence_db}: must be positive.")
    if min_dip_spacing_hz <= 0:
        raise ValueError(f"min_dip_spacing_hz={min_dip_spacing_hz}: must be positive.")

    _refuse_container(module_sweeps)
    try:
        results = module_sweeps["results"]
    except (TypeError, KeyError):
        raise TypeError(
            "Expected one module's sweep result (with 'results' and "
            "'call_params'), not one of its parts."
        ) from None

    culled: list[str] = []
    examined = 0

    for step, by_direction in results.items():
        if iteration is not None and step != iteration:
            continue

        for swept_direction, sections in by_direction.items():
            if direction is not None and swept_direction != direction:
                continue

            for name, sweep in sections.items():
                examined += 1
                # Already condemned by another amplitude or direction. One
                # collision is enough; the rest of its sweeps say nothing new.
                if name in culled:
                    continue
                if iq_key not in sweep:
                    raise KeyError(
                        f"{name!r} has no {iq_key!r}. Its sweep holds "
                        f"{', '.join(repr(k) for k in sweep)}."
                    )
                if _has_collided_pair(
                    sweep,
                    min_separation_hz=min_separation_hz,
                    min_prominence_db=min_prominence_db,
                    min_dip_spacing_hz=min_dip_spacing_hz,
                    iq_key=iq_key,
                ):
                    culled.append(name)

    if not examined:
        raise ValueError(
            f"iteration={iteration}, direction={direction} selected no sweeps. "
            f"This result has iterations {sorted(results)} and directions "
            f"{sorted({d for by_direction in results.values() for d in by_direction})}."
        )

    return culled


def _has_collided_pair(
    sweep, *, min_separation_hz, min_prominence_db, min_dip_spacing_hz, iq_key
) -> bool:
    """Does one sweep hold two dips within ``min_separation_hz`` of each other?

    ``distance=`` is handed to :func:`~scipy.signal.find_peaks` here, which the
    search proper deliberately does not do (see :func:`_separation_pass`). The
    objection there was that a sample count means a different frequency at every
    sweep resolution, and that ``distance`` keeps the deepest of a close group
    and silently drops the rest. Neither applies to this use: the count is
    computed from the section's own step size, so it is a frequency; and it is
    doing the opposite job — not cutting candidates, but keeping one broad dip's
    shoulders from being counted as its neighbours. The cut is the separation
    test below, on frequencies.
    """
    frequencies = np.asarray(sweep["frequencies"], dtype=float)
    iq = np.asarray(sweep[iq_key])

    # A dead channel is all zeros, which is no evidence of a collision rather
    # than a trace to search — and magnitude_db would rightly refuse it.
    if frequencies.size < 3 or not np.any(np.abs(iq) > 0):
        return False

    mag_db = magnitude_db(iq)

    # Median step, so a downward sweep (descending frequencies) or a stray
    # duplicated point does not set the scale.
    step_hz = float(np.median(np.abs(np.diff(frequencies))))
    spacing_points = max(1, int(round(min_dip_spacing_hz / step_hz)))

    # Resonances are dips in |S21|, so they are peaks in the inverted trace.
    dips, _ = signal.find_peaks(
        -mag_db, prominence=min_prominence_db, distance=spacing_points
    )
    if len(dips) < 2:
        return False

    # Sorted, so the closest pair is an adjacent pair — and sorting is what lets
    # a downward sweep be read with the same test as an upward one.
    dip_frequencies = np.sort(frequencies[dips])
    return bool(np.any(np.diff(dip_frequencies) <= min_separation_hz))
