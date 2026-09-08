"""
Choose an operating point for every resonator, from sweeps already measured.

Bias finding asks two questions about each resonator, in that order:

**Which amplitude?** The one just below where the resonator bifurcates — as
much probe power as it will take while its sweep still describes a resonance.
Answering it needs a sweep taken over an ``AmplitudeSchedule``, because "just
below" is only meaningful against amplitude steps that were actually measured,
and the answer is one of them.

**Which frequency, within that sweep?** The sweep centre is only where we
*looked*; the resonance is wherever it turned out to be, up to half a span
away.

Each question has more than one defensible method, so each is a small
dispatch. The amplitude search asks a bifurcation detector about one amplitude
step at a time and goes back one when it fires; which detector is
:data:`BIFURCATION_METHODS`. The frequency comes from
:data:`FREQUENCY_METHODS`. Both work over data — nothing here touches a board,
so a saved sweep is biased the same way a live one is.

What goes in, what comes out
----------------------------
In: **one module's** sweep result, as everything in this package takes it —
``sweeps[crs.module[m].index()]``. A ``multisweep`` that was given no schedule
is one amplitude step, which is a legitimate thing to bias off if you already
know the amplitude; the search then has nothing to go back to and says so.

Out: a :class:`BiasReport`, whose ``catalog`` is a **new**
:class:`~rfmux.core.resonators.ResonatorCatalog` carrying the operating points
that were found::

    report = find_bias_points(sweeps)
    report.catalog["BOTA"].bias.frequency_hz
    await crs.apply_bias(report.catalog)

The report also goes into the sweeps it was found from, as plain builtins,
under ``sweeps["bias_report"]`` — so an operating point travels with the data
behind it, and saving updates the sweep's own file rather than leaving a second
one beside it. ``BiasReport.from_dict(sweeps["bias_report"])`` reads it back.
One analysis is stored at a time: a second call replaces it, the way re-running
a fit replaces that model's fit.

Nothing else is modified on the way past: not the catalog that was swept, not
the sweep entries. A bias point is a claim about one analysis of one set of
sweeps, and two of them side by side — one from the derivative method, one from
hysteresis — is a comparison worth being able to make. The catalog you swept is
still the catalog you swept, and merging is
the caller's decision.

Every resonator gets a bias point
---------------------------------
There is no such thing here as a resonator that came back unbiased. The catalog
and the sweeps go together — the sweeps were taken *from* that catalog — so
every resonator has the data it needs, and the questions above always have an
answer. A missing sweep or a missing ``iq_volts`` is a mismatched pair of
arguments rather than a property of one detector, and it raises rather than
being absorbed into a per-resonator result.

What does happen is that an answer turns out to be a **default rather than a
measurement**. The quietest amplitude measured was already bifurcated, so there
was nothing below it to fall back to; or nothing bifurcated at all, so the
loudest amplitude measured is the answer only because it is the loudest; or the
resonance came out so far from the sweep centre that the tone was left where it
already was instead. Those bias points are usable and are the best the
measurement supports — but they are not the operating point the analysis set
out to find, so each one comes back with ``flagged_because`` saying which it
is. ``report.flagged`` is the list to read before applying anything.

The calibration is measured here too
------------------------------------
``dI_df`` and ``dQ_df`` (V/Hz) are evaluated at the chosen frequency, on the
chosen sweep, in the same step that chooses it — and
:attr:`~rfmux.core.resonators.BiasPoint.df_calibration`, the Hz/V factor df
units are read through, derives from them. That is not tidiness: ``BiasPoint``
is frozen precisely so a tone cannot carry a calibration measured somewhere
else, so the frequency and its calibration have to arrive together or not at
all.

The frequency lands on the hardware tone grid on the way in, as every bias
frequency does, and the derivatives are evaluated *there* rather than at the
un-quantized peak — the calibration then belongs to the tone that will actually
be played.

``iq_rotation_deg`` is deliberately left unset. It comes off a timestream
rather than a sweep, so it is not this module's to measure, and a rotation
angle measured at the previous tone would not survive the move anyway.

Not ported (yet)
----------------
* **The fitted resonance frequency as a bias-frequency method.** The fitting
  layer next door already produces ``fr``; wiring it in is a fourth entry in
  :data:`FREQUENCY_METHODS`, which is why the methods take a whole sweep entry
  rather than two arrays — a fit lives on the entry.
* **The log-arc-speed variant** of the derivative method, which exists to make
  a noisy peak stand out. Worth revisiting against real noisy data rather than
  porting on faith.
* **The diagnostic arrays** the old implementation wrote onto the selected
  sweep entry. What is scalar comes back on the report; the arrays are
  recomputable from the sweep and the settings, and a sweep entry should still
  read the way it was measured.

Attribution
-----------
The bifurcation-by-derivative test and the max-arc-speed frequency are ported
from hidfmux ``analysis/find_bias.py`` (Maclean Rouble, McGill Cosmology), by
way of ``algorithms/measurement/bias_kids.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from ..core.resonators import BiasPoint, ResonatorCatalog
from . import store
from .store import plain
from .sweep_results import _iterations, collect_amplitude_iterations_for

__all__ = [
    "BIFURCATION_METHODS",
    "FREQUENCY_METHODS",
    "HYSTERESIS_COMPARISONS",
    "BifurcationCheck",
    "AmplitudeChoice",
    "BiasFinding",
    "BiasReport",
    "find_bias_points",
    "find_bias_amplitude",
    "find_bias_frequency",
    "bifurcated_by_derivative",
    "bifurcated_by_hysteresis",
    "bifurcated_by_either",
    "iq_arc_speed",
    "normalized_arc_speed",
    "iq_derivative_splines",
    "iq_derivatives_at",
]

#: How :func:`find_bias_amplitude` decides an amplitude step is bifurcated, and
#: the default. ``"both"`` runs the other two and takes a step as bifurcated if
#: either says so, which is the most sensitive of the three and the one to
#: reach for unless you have a reason not to. It needs both sweep directions,
#: as ``"hysteresis"`` does; ``"derivative"`` works on one.
BIFURCATION_METHODS = ("both", "derivative", "hysteresis")

#: The bifurcation methods that need both sweep directions, because comparing
#: them is what they do. Checked once per call rather than once per resonator.
_NEEDS_BOTH_DIRECTIONS = ("hysteresis", "both")

#: What :func:`bifurcated_by_hysteresis` compares the two sweep directions in,
#: and the default. ``"magnitude"`` compares their ``|S21|`` against frequency;
#: ``"iq"`` measures how far apart the two traces are on the IQ plane. Same
#: test, two projections of it — see the detector for what each is blind to.
HYSTERESIS_COMPARISONS = ("magnitude", "iq")

#: How :func:`find_bias_frequency` places the tone inside the chosen sweep, and
#: the default.
FREQUENCY_METHODS = ("iq_derivative", "minimum")

#: The direction the bias frequency is measured on when an amplitude step has
#: more than one and the caller did not say.
PREFERRED_DIRECTION = "upward"

#: How many samples apart :func:`bifurcated_by_derivative` will accept its
#: up-spike and down-spike, at most. ``1`` is a jump crossed in a single
#: frequency bin, which is what a discontinuity looks like when the sweep grid
#: happens to straddle it cleanly. ``2`` also accepts the case where a sample
#: landed partway across the jump, so the trace takes two steps to get over it
#: and the peak in the arc speed is two samples wide instead of one. That is a
#: property of where the grid fell rather than of the resonator, so refusing it
#: loses real bifurcations for no reason. Beyond 2 the pattern stops describing
#: a discontinuity and starts matching the ordinary rise and fall through a
#: resonance, so this is not a knob.
MAX_SPIKE_SEPARATION = 2


# ─── Results ──────────────────────────────────────────────────────────────────


class BifurcationCheck(NamedTuple):
    """One method's verdict on one amplitude step, and the numbers behind it.

    Carries what it compared as well as the verdict, because the settings that
    produce them are knobs a user has to turn against their own array: a
    detector that only ever says yes or no gives them nothing to turn them by.

    ``metric`` is a dict with one entry per quantity the method examined, each
    named for what it is. A method that tests more than one thing has more than
    one entry, so the verdict can be read off the numbers behind it rather than
    inferred from a single figure standing in for all of them — which is what
    ``"derivative"``, testing three things, used to do. Which keys appear
    depends on the method; see the detector for what each means and what units
    it is in.

    ``threshold`` is the single bar those quantities were held to, in their own
    units. Clearing it is not on its own a positive verdict: ``"derivative"``
    reports two prominences and an ``"adjacency"`` flag, and needs all three.

    ``parts`` is empty for a check that came from one test, and holds the
    constituent checks for one that combined several — ``"both"``, whose
    ``metric`` is in multiples of each test's own bar so that the two fit on
    one set of axes. The raw numbers are then in here, each beside the
    threshold it was actually compared against, which is where to read them
    from when a threshold is what you are picking.
    """

    method: str
    bifurcated: bool
    metric: dict
    threshold: float
    parts: dict = {}  # by method name, for a combined check; else empty

    def to_dict(self) -> dict:
        """Plain builtins only — files never contain these classes."""
        return {
            "method": self.method,
            "bifurcated": bool(self.bifurcated),
            "metric": dict(self.metric),
            "threshold": float(self.threshold),
            "parts": {k: v.to_dict() for k, v in self.parts.items()},
        }

    @classmethod
    def from_dict(cls, d) -> "BifurcationCheck":
        return cls(
            method=d["method"],
            bifurcated=bool(d["bifurcated"]),
            metric=dict(d["metric"]),
            threshold=float(d["threshold"]),
            # .get, because a report written before combined checks existed
            # has no parts, and a check from a single test never does.
            parts={
                k: cls.from_dict(v) for k, v in (d.get("parts") or {}).items()
            },
        )


class AmplitudeChoice(NamedTuple):
    """Which amplitude step one resonator should be biased at.

    What :func:`find_bias_amplitude` answers with. A tuple, so it unpacks::

        iteration, amplitude, bifurcated_at, checks = find_bias_amplitude(...)

    ``checks`` holds only the steps that were examined — the search stops at
    the first bifurcated one, so the steps above it were never looked at and
    have no verdict to report.
    """

    iteration: int
    amplitude: float
    bifurcated_at: float | None  # amplitude where bifurcation was first seen
    checks: dict[int, BifurcationCheck]  # by iteration, in the order examined

    @property
    def is_bifurcated_at_bias(self) -> bool:
        """Is the *chosen* amplitude step itself bifurcated?

        True only when there was nothing below it to go back to. See
        :func:`find_bias_amplitude` for why that is still the answer.
        """
        return self.bifurcated_at is not None and self.bifurcated_at == self.amplitude

    def to_dict(self) -> dict:
        """Plain builtins only — files never contain these classes."""
        return {
            "iteration": int(self.iteration),
            "amplitude": float(self.amplitude),
            "bifurcated_at": _or_none(self.bifurcated_at),
            "checks": _checks_to_dict(self.checks),
        }

    @classmethod
    def from_dict(cls, d) -> "AmplitudeChoice":
        return cls(
            iteration=int(d["iteration"]),
            amplitude=float(d["amplitude"]),
            bifurcated_at=_or_none(d.get("bifurcated_at")),
            checks=_checks_from_dict(d["checks"]),
        )


@dataclass(frozen=True, slots=True)
class BiasFinding:
    """How one resonator's bias point was arrived at.

    The bias point itself is on the report's catalog; this is the working
    behind it. Every resonator gets one — see the module docstring for why
    there is no unbiased outcome to represent.

    ``flagged_because`` is a sentence or ``None``. It is set when the answer is
    a *default* rather than something the amplitude steps actually established:
    usable, the best available, and not what the analysis set out to find.
    """

    name: str
    iteration: int  # the amplitude step this came off
    amplitude: float
    frequency_hz: float  # on the tone grid, as it went onto the BiasPoint
    dI_df: float  # V/Hz at that frequency
    dQ_df: float
    bifurcated_at: float | None  # amplitude where bifurcation was first seen
    checks: dict[int, BifurcationCheck]  # every amplitude step examined
    flagged_because: str | None = None

    @property
    def good(self) -> bool:
        """Nothing about this bias point needs a second look."""
        return self.flagged_because is None

    def to_dict(self) -> dict:
        """Plain builtins only — files never contain these classes.

        No version of its own: a finding is only ever written as part of a
        :class:`BiasReport`, and one stamp on the thing that becomes a file is
        the version that matters.
        """
        return {
            "name": self.name,
            "iteration": int(self.iteration),
            "amplitude": float(self.amplitude),
            "frequency_hz": float(self.frequency_hz),
            "dI_df": float(self.dI_df),
            "dQ_df": float(self.dQ_df),
            "bifurcated_at": _or_none(self.bifurcated_at),
            "checks": _checks_to_dict(self.checks),
            "flagged_because": self.flagged_because,
        }

    @classmethod
    def from_dict(cls, d) -> "BiasFinding":
        return cls(
            name=d["name"],
            iteration=int(d["iteration"]),
            amplitude=float(d["amplitude"]),
            frequency_hz=float(d["frequency_hz"]),
            dI_df=float(d["dI_df"]),
            dQ_df=float(d["dQ_df"]),
            bifurcated_at=_or_none(d.get("bifurcated_at")),
            checks=_checks_from_dict(d["checks"]),
            flagged_because=d.get("flagged_because"),
        )


def _or_none(value):
    """``float(value)``, but ``None`` survives as ``None``.

    ``bifurcated_at`` is an amplitude or the statement that no amplitude step
    bifurcated, and those are different answers — coercing the second to 0.0
    would say the detector bifurcates at zero drive.
    """
    return None if value is None else float(value)


def _checks_to_dict(checks: Mapping) -> dict:
    """A ``{iteration: BifurcationCheck}`` map, as builtins.

    The keys stay integers. They are amplitude-step numbers and get compared
    and sorted as such; JSON would force them to strings, but this goes into a
    pickle, which has no such quarrel with an int.
    """
    return {int(k): v.to_dict() for k, v in checks.items()}


def _checks_from_dict(d: Mapping) -> dict:
    return {int(k): BifurcationCheck.from_dict(v) for k, v in d.items()}


@dataclass(slots=True)
class BiasReport:
    """What one call to :func:`find_bias_points` concluded, and the catalog.

    ``catalog`` is the answer; the findings are how it was reached, one per
    resonator in bias-frequency order. The settings come back here rather than being
    copied onto a thousand bias points, as with
    :class:`~rfmux.tuning.fits.FitReport` — recording them alongside the data
    is the output folder's job.
    """

    # Stamped into to_dict output and required exactly by from_dict, so a file
    # from another version of this module fails loudly rather than being half
    # understood. Bump whenever the dict shape changes in a way from_dict
    # cannot absorb. Version 2 made BifurcationCheck.metric a dict of named
    # quantities where version 1 had a single float, which from_dict cannot
    # tell apart from a legitimate value — so a version 1 report is refused
    # rather than read as though one number meant the same thing.
    SCHEMA_VERSION = 2

    catalog: ResonatorCatalog
    findings: list[BiasFinding]
    settings: dict = field(default_factory=dict)

    @property
    def flagged(self) -> list[BiasFinding]:
        """The bias points that are defaults rather than measurements.

        The list to read before applying anything: each of these is a
        resonator whose amplitude steps did not bracket its bifurcation point.
        """
        return [f for f in self.findings if not f.good]

    @property
    def good(self) -> list[BiasFinding]:
        return [f for f in self.findings if f.good]

    def __getitem__(self, name: str) -> BiasFinding:
        for f in self.findings:
            if f.name == name:
                return f
        raise KeyError(f"No finding for {name!r}.")

    def __len__(self) -> int:
        return len(self.findings)

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain builtins only — files never contain these classes.

        The catalog goes in through its own ``to_dict``, keeping its version
        stamp beside this one: the answer and the working behind it can be read
        back independently, and a catalog that outgrows this file's shape says
        so on its own terms.
        """
        return {
            "schema_version": self.SCHEMA_VERSION,
            "catalog": self.catalog.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "settings": plain(self.settings),
        }

    @classmethod
    def from_dict(cls, d) -> "BiasReport":
        version = d.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={version!r}, expected {cls.SCHEMA_VERSION}: "
                f"this dict was written by a different version of BiasReport."
            )
        return cls(
            catalog=ResonatorCatalog.from_dict(d["catalog"]),
            findings=[BiasFinding.from_dict(f) for f in d["findings"]],
            settings=d.get("settings", {}),
        )

    def __repr__(self) -> str:
        flagged = self.flagged
        head = (
            f"BiasReport: {len(self.findings)} biased, {len(flagged)} flagged"
        )
        rows = []
        for f in flagged[:5]:
            rows.append(f"  {f.name}: {f.flagged_because}")
        if len(flagged) > 5:
            rows.append(f"  ... {len(flagged) - 5} more")
        return "\n".join([head] + rows)


# ─── The entry point ──────────────────────────────────────────────────────────


def find_bias_points(
    sweeps,
    catalog: ResonatorCatalog | None = None,
    *,
    amplitude_method: str = "both",
    frequency_method: str = "iq_derivative",
    direction: str | None = None,
    spike_prominence_factor: float = 0.5,
    noise_gate_factor: float = 50.0,
    max_discrepancy: float = 0.1,
    compare: str = "magnitude",
    max_distance_hz: float | None = None,
    save=None,
    label=None,
) -> BiasReport:
    """Find an operating point for every resonator in a catalog.

    For each one: search the amplitude steps for the one below bifurcation,
    place the tone inside that step's sweep, and measure the IQ derivatives
    there. Every resonator gets a bias point; see the module docstring for what
    ``flagged_because`` means and why there is no unbiased outcome.

    Args:
        sweeps: **one module's** value out of what ``multisweep`` returned —
            ``sweeps[crs.module[m].index()]``.
            The whole container, keyed by module, is refused: a report is about
            one module, and which one is your choice to make.
        catalog: the resonators to bias, and the source of everything the new
            catalog keeps unchanged — names, channels, the module, the
            separation rule. Defaults to the catalog snapshot recorded in the
            sweep's ``call_params``, which is the usual case: you are biasing
            the array you swept. A catalog holding a resonator these sweeps do
            not cover raises, because the two were then not measured together.
        amplitude_method: which bifurcation test the amplitude search uses,
            from :data:`BIFURCATION_METHODS`. The default, ``"both"``, requires
            the sweeps to have been taken in both directions, and so does
            ``"hysteresis"`` — a one-direction sweep wants ``"derivative"``,
            which reads a single trace.
        frequency_method: where in the chosen sweep the tone goes, from
            :data:`FREQUENCY_METHODS`.
        direction: which direction's sweep to measure the bias frequency and
            the calibration on. None takes ``"upward"`` when it is there, and
            the only direction there is otherwise. The amplitude search is
            unaffected — a detector sees every direction of its own step.
        spike_prominence_factor: passed to :func:`bifurcated_by_derivative`.
        noise_gate_factor: passed to :func:`bifurcated_by_derivative` — how far
            above a sweep's own noise floor a spike has to stand before it
            counts. Lower it if real bifurcations are being missed on a noisy
            array; ``0.0`` switches it off, which is what the test did before
            the gate existed.
        max_discrepancy: passed to :func:`bifurcated_by_hysteresis`.
        compare: passed to :func:`bifurcated_by_hysteresis` — what the two
            sweep directions are compared in, from
            :data:`HYSTERESIS_COMPARISONS`.
        max_distance_hz: how far from the sweep centre a resonance may come
            out before the answer is disbelieved. Past this, the tone is left
            where the sweep was centred — the frequency it already had — and
            the finding is flagged. None, the default, believes anything, which
            is everything the trace could offer: the answer is a point of the
            trace, so it is inside the span whatever happens, and this only
            means something when it is tighter than the span.
        save: save the *sweeps*, which now carry the report. It goes into
            ``sweeps["bias_report"]`` either way; this is only whether the file
            it came from is updated to match. Sweeps that were never saved get
            a new file. Defaults to
            ``rfmux.tuning.store.autosave_enabled()``.
        label: your name for the file, used only when these sweeps are being
            written for the first time — a re-save keeps the name the file
            already has.

    Returns:
        BiasReport: a new catalog carrying the bias points, and one
        :class:`BiasFinding` per catalog resonator, in bias-frequency order, saying
        how each was arrived at. The same report, as builtins, is left in
        ``sweeps["bias_report"]``.

    Raises:
        TypeError: for the whole container rather than one module's result.
        ValueError: for an unknown method, for ``"hysteresis"`` on a sweep with
            only one direction, for a *direction* that was not swept, or for no
            catalog to bias — none passed and none recorded in the sweep.
        KeyError: for a catalog resonator these sweeps do not cover.
    """
    # Everything the caller could have got wrong about the *whole* call is
    # checked here, once, before a single resonator is analysed. A thousand
    # copies of the same complaint is worse than one.
    _check_method("amplitude_method", amplitude_method, BIFURCATION_METHODS)
    _check_method("frequency_method", frequency_method, FREQUENCY_METHODS)
    _check_method("compare", compare, HYSTERESIS_COMPARISONS)

    directions = _directions_swept(sweeps)
    if (
        amplitude_method in _NEEDS_BOTH_DIRECTIONS
        and not {"upward", "downward"} <= directions
    ):
        raise ValueError(
            f"The {amplitude_method!r} method compares an upward sweep against "
            f"a downward one, and this result holds only {sorted(directions)}. "
            f"Sweep both directions, or use amplitude_method='derivative', "
            f"which reads one sweep at a time."
        )
    if direction is not None and direction not in directions:
        raise ValueError(
            f"direction={direction!r} was not swept. This result holds "
            f"{sorted(directions)}."
        )

    # The array being biased. Falling back to the snapshot the sweep recorded
    # is the common case, and it guarantees the catalog and the data match.
    if catalog is None:
        catalog = _catalog_swept(sweeps)

    # We work on a copy and hand that back, so the catalog that was swept is
    # still the catalog that was swept. Cheap — a catalog holds only scalars.
    biased = catalog.copy()

    amplitude_settings = dict(
        method=amplitude_method,
        spike_prominence_factor=spike_prominence_factor,
        noise_gate_factor=noise_gate_factor,
        max_discrepancy=max_discrepancy,
        compare=compare,
    )

    # One finding per resonator, in bias-frequency order, because that is what
    # iterating a catalog gives you.
    findings = [
        _bias_one(
            sweeps,
            resonator,
            direction=direction,
            frequency_method=frequency_method,
            max_distance_hz=max_distance_hz,
            amplitude_settings=amplitude_settings,
        )
        for resonator in biased
    ]

    report = BiasReport(
        catalog=biased,
        findings=findings,
        settings={
            "module": sweeps.get("module"),
            "amplitude_method": amplitude_method,
            "frequency_method": frequency_method,
            "direction": direction,
            "spike_prominence_factor": spike_prominence_factor,
            "noise_gate_factor": noise_gate_factor,
            "max_discrepancy": max_discrepancy,
            "compare": compare,
            "max_distance_hz": max_distance_hz,
        },
    )
    # Into the sweeps it was found from, so that saving updates that file
    # rather than starting a second one. to_dict, not the report: a pickled
    # class records its import path and skips its constructor coming back, so
    # the file would outlive a rename only by restoring into a state BiasReport
    # would have refused to build.
    sweeps["bias_report"] = report.to_dict()
    store.maybe_save(sweeps, "multisweep", save=save, label=label)
    return report


def _bias_one( ## TODO this should be called "_find_bias_for_one", since "bias one" implies applying the bias to the resonator.
    sweeps,
    resonator,
    *,
    direction: str | None,
    frequency_method: str,
    max_distance_hz: float | None,
    amplitude_settings: dict,
) -> BiasFinding:
    """Bias one resonator of the copied catalog, in place, and say how.

    The whole of bias finding for one detector, in the order the decisions are
    made. Nothing here is caught and turned into a per-resonator failure: a
    resonator with no sweeps, or a sweep with no volts, means the catalog and
    the data did not come from the same measurement, which is the caller's
    mistake to hear about rather than this detector's problem.
    """
    # 1. Every sweep this resonator was measured at, one entry per amplitude
    #    step per direction. Raises if these sweeps do not cover it.
    iterations = collect_amplitude_iterations_for(sweeps, resonator.name)

    # 2. Which amplitude to sit at: the step below where it bifurcates.
    choice = find_bias_amplitude(iterations, **amplitude_settings)

    # 3. Of that step's sweeps, the one we take the bias point off. Both
    #    directions were tested for bifurcation, but a frequency and a
    #    calibration have to come from a single trace.
    entry = _entry_for(iterations[choice.iteration], direction)

    # 4. Where in that trace the tone belongs. The sweep centre is only where
    #    we looked; this is where the resonance turned out to be.
    measured_hz = find_bias_frequency(entry, method=frequency_method)

    # 5. Unless that is implausibly far from where the sweep was centred, in
    #    which case it is usually a neighbour in the span or noise in a trace
    #    the resonance has left — not this resonator. Leaving the tone where it
    #    already was beats moving it somewhere we do not believe, so that is
    #    what happens, and step 7 flags it.
    centre_hz = float(entry["original_center_frequency"])
    frequency_hz = (
        centre_hz
        if _too_far(measured_hz, centre_hz, max_distance_hz)
        else measured_hz
    )

    # 6. Onto the tone grid, by building the BiasPoint first. The derivatives
    #    are then read at the frequency the hardware will actually play rather
    #    than at the peak we found up to half a grid step away — and, when we
    #    fell back, at the frequency we actually settled on.
    bias = BiasPoint(frequency_hz=frequency_hz, amplitude=choice.amplitude)
    dI_df, dQ_df = iq_derivatives_at(entry, bias.frequency_hz)

    # Frequency and calibration go on together — BiasPoint is frozen so that a
    # tone can never carry a calibration measured somewhere else.
    resonator.bias = replace(
        bias, dI_df=dI_df, dQ_df=dQ_df, bifurcated_at=choice.bifurcated_at
    )

    # 7. Finally, is this an operating point we actually established, or a
    #    default we fell back to? _concern is the one place that decides.
    return BiasFinding(
        name=resonator.name,
        iteration=choice.iteration,
        amplitude=choice.amplitude,
        frequency_hz=resonator.bias.frequency_hz,
        dI_df=dI_df,
        dQ_df=dQ_df,
        bifurcated_at=choice.bifurcated_at,
        checks=choice.checks,
        flagged_because=_concern(
            choice,
            measured_hz=measured_hz,
            centre_hz=centre_hz,
            max_distance_hz=max_distance_hz,
        ),
    )


def _concern(
    choice: AmplitudeChoice,
    *,
    measured_hz: float,
    centre_hz: float,
    max_distance_hz: float | None,
) -> str | None:
    """Why this bias point is worth a second look, or None if it looks sound.

    One place, so that "good bias point" means one thing across the module and
    a reader can see the whole standard at once. Ordered worst first, and only
    the first concern is reported: a resonator whose sweeps never bifurcated
    has a bigger problem than one whose tone landed a little off centre.

    Every one of these still produces a usable bias point. What they have in
    common is that the measurement did not establish the answer, so it is a
    default that was fallen back to rather than something that was found.
    """
    if choice.is_bifurcated_at_bias:
        return (
            f"the quietest amplitude measured ({choice.amplitude:g}) was already "
            f"bifurcated, so there was nothing below it to fall back to"
        )
    if choice.bifurcated_at is None:
        return (
            f"nothing bifurcated, so this is the loudest amplitude measured "
            f"({choice.amplitude:g}) rather than a limit that was found"
        )
    if _too_far(measured_hz, centre_hz, max_distance_hz):
        return (
            f"the resonance came out {(measured_hz - centre_hz) / 1e3:+.1f} kHz "
            f"from the sweep centre, past the {max_distance_hz / 1e3:.1f} kHz "
            f"asked for — usually a neighbour in the span, or a resonance pulled "
            f"out of it — so the tone was left where the sweep was centred"
        )
    return None


def _too_far(measured_hz: float, centre_hz: float, max_distance_hz: float | None):
    """Is this frequency further from the sweep centre than we will believe?

    The one predicate behind both halves of that decision: which frequency the
    bias point gets, and what the flag says about it. ``max_distance_hz`` of
    None believes anything, which is everything the trace could offer.
    """
    return max_distance_hz is not None and abs(measured_hz - centre_hz) > max_distance_hz


def _check_method(argument: str, value: str, allowed: tuple[str, ...]) -> None:
    """Refuse an unknown method by name, the same way for every dispatch."""
    if value not in allowed:
        raise ValueError(
            f"Unknown {argument} {value!r}. Must be one of {allowed}."
        )


# ─── Which amplitude ──────────────────────────────────────────────────────────


def find_bias_amplitude(
    iterations: Mapping[int, Mapping[str, dict]],
    *,
    method: str = "both",
    spike_prominence_factor: float = 0.5,
    noise_gate_factor: float = 50.0,
    max_discrepancy: float = 0.1,
    compare: str = "magnitude",
) -> AmplitudeChoice:
    """Search one resonator's amplitude steps for the one to bias at.

    The steps are examined quietest first — in ascending *amplitude*, not in
    the order they were measured, which an ``explicit`` amplitude schedule is
    free to shuffle. Each is put to the chosen bifurcation test, and the first
    step that bifurcates ends the search: the step *below* it is the answer, as
    much drive as the resonator takes while its sweep still describes a
    resonance.

    Two ends of that, both of which still return an amplitude — the best the
    measurement supports — and are flagged by :func:`find_bias_points`:

    * If no step bifurcates, the loudest is chosen. The schedule did not reach
      the limit, so the most drive measured is the most drive known to be safe.
    * If the *quietest* step bifurcates there is nothing below it, so it is
      chosen and :attr:`AmplitudeChoice.is_bifurcated_at_bias` says so. The schedule
      started too high.

    Args:
        iterations: one resonator's sweeps, ``{iteration: {direction: entry}}``
            — what
            :func:`~rfmux.tuning.sweep_results.collect_amplitude_iterations_for`
            returns. A single ``multisweep`` gives one amplitude step, which is
            a legitimate thing to hand over.
        method: which test, from :data:`BIFURCATION_METHODS`. The default,
            ``"both"``, runs the other two and takes either verdict, so it is
            the one method that reads all three of the settings below — and
            the one that needs both sweep directions.
        spike_prominence_factor: passed to :func:`bifurcated_by_derivative`.
        noise_gate_factor: passed to :func:`bifurcated_by_derivative`.
        max_discrepancy: passed to :func:`bifurcated_by_hysteresis`.
        compare: passed to :func:`bifurcated_by_hysteresis`.

    Returns:
        AmplitudeChoice: the iteration and amplitude to bias at, where
        bifurcation was first seen, and each examined step's verdict.

    Raises:
        ValueError: for an unknown *method* or *compare*, or for no sweeps at
            all.
    """
    _check_method("method", method, BIFURCATION_METHODS)
    _check_method("compare", compare, HYSTERESIS_COMPARISONS)
    if not iterations:
        raise ValueError("No sweeps here, so there is no amplitude to choose.")

    detector = _BIFURCATION[method]
    # Each detector takes only its own settings, so a caller passing all of
    # them does not hand the hysteresis test a spike threshold it has no use
    # for. "both" runs the two tests, so it is the one that takes both sets.
    spikes = {
        "spike_prominence_factor": spike_prominence_factor,
        "noise_gate_factor": noise_gate_factor,
    }
    separation = {"max_discrepancy": max_discrepancy, "compare": compare}
    settings = {
        "derivative": spikes,
        "hysteresis": separation,
        "both": {**spikes, **separation},
    }[method]

    # What each step probed at, and the steps in ascending order of it. Every
    # direction of a step shares one amplitude, so this is one number per step.
    amplitude = {i: _amplitude_of(entries) for i, entries in iterations.items()}
    quietest_first = sorted(iterations, key=amplitude.get)

    # Walking pairs rather than indices: each step alongside the one below it,
    # and None below the quietest. `previous` is the answer whenever the step
    # in hand turns out to be bifurcated.
    checks: dict[int, BifurcationCheck] = {}
    for previous, iteration in zip([None, *quietest_first], quietest_first):
        checks[iteration] = detector(iterations[iteration], **settings)
        if not checks[iteration].bifurcated:
            continue

        # Found the limit. Bias one step below it — or here, if this is the
        # quietest amplitude we have and there is nothing below to fall back to.
        chosen = iteration if previous is None else previous
        return AmplitudeChoice(
            iteration=chosen,
            amplitude=amplitude[chosen],
            bifurcated_at=amplitude[iteration],
            checks=checks,
        )

    # Nothing bifurcated. The loudest step is as much drive as we know to be
    # safe, so it is the answer; bifurcated_at stays None to say we never
    # found the limit.
    loudest = quietest_first[-1]
    return AmplitudeChoice(
        iteration=loudest,
        amplitude=amplitude[loudest],
        bifurcated_at=None,
        checks=checks,
    )


def bifurcated_by_derivative(
    entries: Mapping[str, dict],
    *,
    spike_prominence_factor: float = 0.5,
    noise_gate_factor: float = 50.0,
) -> BifurcationCheck:
    """Is this sweep bifurcated? Ask the jumps in its IQ arc-length speed.

    A bifurcated resonance does not trace a smooth loop: the state jumps, so
    the IQ trace crosses a gap between one sweep point and the next. Differentiate
    the arc-length speed along the trace and that shows up as a positive spike
    with a negative spike immediately after it — up onto the jump, back down off
    it — which is what this looks for. Two spikes side by side, in that order.

    Working in the *speed* rather than in the trace makes the test insensitive
    to how deep or how large the loop is; I and Q are each normalized by their
    own range first, and each difference by the frequency spacing it spans, so
    what remains is shape.

    A spike has to clear **two** bars, and they ask different questions.

    *spike_prominence_factor* asks whether the spike is large compared to the
    sweep: it must stand out from its surroundings — scipy's *prominence* — by
    more than this fraction of the span of the arc-length speed. The factor
    multiplies, so it reads the way it behaves: the default of 0.5 asks a spike
    to stand half of the speed's whole range out of its own neighbourhood, and
    raising it asks for more, which is less sensitive.

    That bar alone cannot tell a jump from noise, because on a sweep with no
    visible resonance the span *is* noise. The largest excursion of a noisy
    trace and the range of that trace are both order statistics of the same
    scatter, so their ratio lands in the same place — around 0.3 to 0.6 for a
    hundred-point sweep — no matter how quiet the drive was. A bar set as a
    fraction of the span is a bar that noise clears by construction, which is
    why the quiet end of an amplitude ladder used to produce false positives
    that no choice of factor could remove.

    *noise_gate_factor* asks the other question: is this spike bigger than what
    this trace scatters by anyway? The scatter is measured as the median
    absolute deviation of the differences, scaled to read like a standard
    deviation. Median absolute deviation rather than a standard deviation on
    purpose — a jump puts two large samples into the trace and inflates a
    standard deviation by so much that the ratio barely moves, whereas a median
    is untroubled by two outliers among a hundred and goes on describing the
    noise floor rather than the noise floor plus the signal.

    The default of 50 is where the two populations separate most cleanly on the
    array this was calibrated against: the largest excursion of a noise-only
    sweep there stood under 10 median-absolute-deviations, and the weakest real
    jump stood at 152. Anything from roughly 10 to 150 behaves; below that the
    quiet steps start returning, and above it real jumps start being missed,
    which is the more expensive error. Pass ``0.0`` to switch the gate off and
    get the span bar on its own.

    A trace with no scatter at all — quantized, or constant — has a noise floor
    of zero, which switches the gate off for that trace rather than dividing by
    it. There is nothing there for a gate to measure against.

    Args:
        entries: one amplitude step, ``{direction: entry}``. Every direction
            present is tested and the step counts as bifurcated if any of them
            says so — a bifurcated resonator jumps whichever way the sweep
            runs, so needing both to agree would only lose the one that
            happened to catch it.
        spike_prominence_factor: the bar a spike has to clear, as a multiple
            of the span of the arc-length speed. Larger is less sensitive.
        noise_gate_factor: the second bar, as a multiple of the trace's own
            noise floor. Larger is less sensitive; ``0.0`` disables it.

    Returns:
        BifurcationCheck: with ``threshold`` the bar the spikes had to clear —
        **the higher of the two**, since both are prominences in the same units
        and clearing both is clearing the larger — and ``metric`` the three
        things this test asks about, from one direction:

        ``"positive_spike_prominence"``
            how far the tallest up-spike stands out of its own neighbourhood.
            ``0.0`` if the trace has no up-spike at all.
        ``"negative_spike_prominence"``
            the same for the tallest down-spike.
        ``"adjacency"``
            whether a spike that cleared the bar was followed within
            :data:`MAX_SPIKE_SEPARATION` samples by a down-spike that also
            cleared it. Its own condition, with no threshold to compare against.

        The verdict is ``True`` when both prominences clear ``threshold`` *and*
        ``"adjacency"``, so the three entries say which of those failed. A
        prominence just under the bar is a factor to lower; two prominences
        well over it with ``"adjacency"`` false is a jump the pattern-matching
        missed, which is a different problem. To find out *which* bar was
        binding, run it again with ``noise_gate_factor=0.0`` and compare the
        thresholds — or plot them, which is what
        ``Demos/example_plotting_bias.plot_bifurcation_verdict_map`` is for.

        The direction reported is the one that came closest to bifurcating —
        one that fired if any did, and otherwise the one that spiked hardest —
        so the numbers sit beside a verdict they belong to. Every direction is
        still tested.

    Raises:
        ValueError: if none of the directions holds a usable sweep.
    """
    verdict = False
    reported = None  # (rank, metric, threshold) of the closest direction so far
    for entry in _directions(entries):
        frequencies, iq = _sorted_trace(entry, "iq_counts")
        speed = _point_to_point_speed(frequencies, iq)
        if speed is None or len(speed) < 3:
            continue

        jumps = np.diff(speed)
        # Both bars measure the same thing — a prominence, in the units of
        # `jumps` — so requiring a spike to clear both is requiring it to clear
        # whichever is higher, and one number describes the bar it faced.
        prominence_threshold = max(
            float(spike_prominence_factor * (speed.max() - speed.min())),
            float(noise_gate_factor * _noise_floor(jumps)),
        )

        # Every spike with its prominence, then the bar applied as a filter,
        # rather than asking find_peaks for only the spikes that clear it. The
        # two select identically, and this way a spike that just missed still
        # has a prominence to report — which is the number the factor gets
        # turned by, so it is the one worth having when the answer was no.
        up, up_prominence = _spikes(jumps)
        down, down_prominence = _spikes(-jumps)
        cleared_up = up[up_prominence >= prominence_threshold]
        cleared_down = down[down_prominence >= prominence_threshold]

        adjacency = _paired(cleared_up, cleared_down)
        verdict = verdict or adjacency

        metric = {
            "positive_spike_prominence": _tallest(up_prominence),
            "negative_spike_prominence": _tallest(down_prominence),
            "adjacency": adjacency,
        }
        # One direction's numbers are reported, and it is the one that came
        # closest: a direction that fired outranks one that did not, and among
        # equals the harder spike wins. So the metric explains the verdict
        # rather than describing whichever sweep happened to be quieter.
        rank = (
            adjacency,
            max(
                metric["positive_spike_prominence"],
                metric["negative_spike_prominence"],
            ),
        )
        if reported is None or rank > reported[0]:
            reported = (rank, metric, prominence_threshold)

    if reported is None:
        raise ValueError(
            "No usable sweep at this amplitude: every direction is too short "
            "have a shape, or has a degenerate frequency or IQ axis."
        )
    _, metric, threshold = reported
    return BifurcationCheck(
        method="derivative", bifurcated=verdict, metric=metric, threshold=threshold
    )


def bifurcated_by_hysteresis(
    entries: Mapping[str, dict],
    *,
    max_discrepancy: float = 0.1,
    compare: str = "magnitude",
) -> BifurcationCheck:
    """Is this sweep bifurcated? Ask whether up and down agree.

    A resonator below bifurcation does not care which way it was swept: the
    upward and downward traces lie on top of each other. Above it, the state
    jumps at a different frequency going up than coming down, and the two
    traces part company in between. So the amplitude where they *begin* to
    differ is the amplitude where bifurcation set in, and this is the test that
    finds it — no assumption about what a jump looks like, just whether the two
    passes agree.

    The measure is the largest separation between the two traces anywhere in
    the sweep, in units of the trace's own scale, so it means the same thing
    for a deep resonator and a shallow one. Below bifurcation it is a noise
    figure. Above it, the traces are a good fraction of a resonance apart.

    *compare* is what "apart" is measured in, and the two answer slightly
    different questions:

    ``"magnitude"`` (the default)
        The difference between the two ``|S21|`` curves against frequency, in
        units of the upward sweep's own dip depth. Deliberately blind to phase:
        a rotation or a delay drift between the passes moves both curves
        nowhere, so what is left is whether the *depth and position of the dip*
        depended on which way the sweep ran. That is the thing bifurcation
        actually does — the two branches carry different transmission — and it
        is a narrower question than the IQ distance asks. On a real array it is
        the better-behaved of the two: the branch switch shows up as a narrow
        spike two to three orders of magnitude above an otherwise flat trace,
        so the quiet steps sit further below the bar than they do in IQ.
    ``"iq"``
        The distance between the two traces on the IQ plane, at matched
        frequency, in units of the loop's own radius. It sees every way the two
        passes can differ — which is its strength and its weakness, because
        phase is the fastest-varying thing across a resonance. A small
        frequency mis-registration between the passes, or a bit of cable-delay
        drift between them, slides one trace along the loop and reads as a
        large separation with no bifurcation anywhere in sight. This was the
        original comparison.

    Args:
        entries: one amplitude step, ``{direction: entry}``. Both directions
            are required — this test *is* the comparison.
        max_discrepancy: how far apart the traces may be, in the units
            *compare* measures in, before the step is called bifurcated. The
            default is a starting point rather than a measured number: on a
            real array it sits several times above what two agreeing passes
            leave behind, and an order of magnitude below a resonator that has
            plainly jumped, which is room enough to be wrong in
            without changing any confident answer. A *marginal* resonator can
            still fall under it. Pick it against your own array by reading
            ``"max_separation"`` across the amplitude steps of a resonator that
            is known to bifurcate, which is what it is reported for — and note
            that the two comparisons are not on the same scale, so a value
            tuned for one does not carry to the other.
        compare: what to measure the separation in, from
            :data:`HYSTERESIS_COMPARISONS`.

    Returns:
        BifurcationCheck: with ``threshold`` the *max_discrepancy* and
        ``metric`` the one quantity this test asks about:

        ``"max_separation"``
            the largest separation between the two traces — in dip depths for
            ``compare="magnitude"``, in loop radii for ``compare="iq"``.

        A dict for one number, so that a caller reading a check does not have
        to know which method produced it to know what shape it is in. The
        verdict here really is the comparison — unlike ``"derivative"``, this
        test has only the one condition.

    Raises:
        ValueError: for an unknown *compare*, if either direction is missing or
            unusable, or if the two sweeps do not cover the same frequencies.
    """
    _check_method("compare", compare, HYSTERESIS_COMPARISONS)

    missing = {"upward", "downward"} - set(entries)
    if missing:
        raise ValueError(
            f"The hysteresis test compares an upward sweep against a downward "
            f"one, and this amplitude step has no "
            f"{' or '.join(sorted(missing))} sweep."
        )

    f_up, z_up = _sorted_trace(entries["upward"], "iq_counts")
    f_down, z_down = _sorted_trace(entries["downward"], "iq_counts")
    if len(f_up) < 2 or len(f_down) < 2:
        raise ValueError("A sweep of fewer than two points has nothing to compare.")
    if f_down[0] > f_up[-1] or f_down[-1] < f_up[0]:
        raise ValueError(
            f"The two directions do not cover the same frequencies "
            f"({f_up[0] / 1e6:.6f}–{f_up[-1] / 1e6:.6f} MHz upward, "
            f"{f_down[0] / 1e6:.6f}–{f_down[-1] / 1e6:.6f} MHz downward), so "
            f"there is nothing to compare them at."
        )

    separation = _HYSTERESIS_COMPARISON[compare](f_up, z_up, f_down, z_down)
    return BifurcationCheck(
        method="hysteresis",
        bifurcated=separation > max_discrepancy,
        metric={"max_separation": separation},
        threshold=max_discrepancy,
    )


def bifurcated_by_either(
    entries: Mapping[str, dict],
    *,
    spike_prominence_factor: float = 0.5,
    noise_gate_factor: float = 50.0,
    max_discrepancy: float = 0.1,
    compare: str = "magnitude",
) -> BifurcationCheck:
    """Is this sweep bifurcated? Ask both tests, and believe whichever says yes.

    The two detectors look for different evidence of the same thing —
    :func:`bifurcated_by_derivative` for the jump inside one trace,
    :func:`bifurcated_by_hysteresis` for the disagreement between the two
    directions — and on a real array they do not fail on the same resonators.
    A resonator that jumps at nearly the same frequency going up as coming down
    is invisible to the hysteresis test and obvious to the derivative one; a
    resonator whose branch switch is spread over enough sweep points to look
    smooth is the other way around. So this asks both and takes a step as
    bifurcated if *either* fires, which is a more sensitive test than either
    alone — deliberately. The cost is the same asymmetry run the other way: a
    false positive from either test is now a false positive here, and it lands
    as a bias amplitude one step quieter than the resonator needed.

    That is usually the trade you want — biasing one step too quiet costs
    responsivity, while biasing one step too loud puts the tone on a resonance
    that is not there any more — and it is why this is the default.

    Both tests run on every step, and both have to be *able* to run: this needs
    the two sweep directions, as the hysteresis test does.

    Args:
        entries: one amplitude step, ``{direction: entry}``, with both
            ``"upward"`` and ``"downward"`` present.
        spike_prominence_factor: passed to :func:`bifurcated_by_derivative`.
        noise_gate_factor: passed to :func:`bifurcated_by_derivative`.
        max_discrepancy: passed to :func:`bifurcated_by_hysteresis`.
        compare: passed to :func:`bifurcated_by_hysteresis`.

    Returns:
        BifurcationCheck: with ``bifurcated`` the disjunction of the two
        verdicts, and ``parts`` the two checks that produced them, keyed
        ``"derivative"`` and ``"hysteresis"`` — each exactly what its own
        detector returned, raw numbers and own threshold.

        Its own ``metric`` is those numbers **in multiples of the bar each was
        held to**, one entry per quantity, named for the test it came from:

        ``"derivative_positive_spike_prominence"``
            the up-spike's prominence over the prominence it needed.
        ``"derivative_negative_spike_prominence"``
            the same for the down-spike.
        ``"derivative_adjacency"``
            the flag, carried through as it stands — a condition rather than a
            measurement, so there is nothing to divide.
        ``"hysteresis_max_separation"``
            the separation between the directions over *max_discrepancy*.

        With ``threshold`` then ``1.0`` for all of them. Two tests in units of
        their own thresholds are the one form in which they are comparable —
        it is what makes a single number tell you how close each came, and it
        puts prominences, dip depths and loop radii on one set of axes. A
        quantity over 1.0 met its condition; the verdict is still each test's
        own combination of its own conditions, which is why the checks that
        made it are kept.

        The exception is a test whose bar was zero — a completely flat
        arc-length speed, or a *spike_prominence_factor* of zero. Nothing can
        be a multiple of nothing, so a quantity that cleared such a bar is
        reported as ``inf`` and one that did not as ``0.0``.

    Raises:
        ValueError: for an unknown *compare*, or for anything either test
            refuses — a missing direction, two directions that do not overlap
            in frequency, a degenerate trace.
    """
    parts = {
        "derivative": bifurcated_by_derivative(
            entries,
            spike_prominence_factor=spike_prominence_factor,
            noise_gate_factor=noise_gate_factor,
        ),
        "hysteresis": bifurcated_by_hysteresis(
            entries, max_discrepancy=max_discrepancy, compare=compare
        ),
    }

    metric = {
        f"{name}_{key}": _in_thresholds(value, part.threshold)
        for name, part in parts.items()
        for key, value in part.metric.items()
    }
    return BifurcationCheck(
        method="both",
        bifurcated=any(part.bifurcated for part in parts.values()),
        metric=metric,
        threshold=1.0,
        parts=parts,
    )


def _in_thresholds(value, threshold: float):
    """*value* as a multiple of the bar it was held to — the unit a combined
    check reports in. A flag passes through: it has no bar to be a multiple of.

    A bar of zero has no multiples either, so clearing it is reported as
    ``inf`` rather than as a division that raises or returns a nan. That is the
    honest reading — anything at all stands out further than nothing — and it
    keeps the entry a number that plots and compares against 1.0.
    """
    if isinstance(value, bool):
        return value
    if threshold == 0:
        return float("inf") if value > 0 else 0.0
    return float(value) / threshold


def _separation_in_iq(
    f_up: np.ndarray, z_up: np.ndarray, f_down: np.ndarray, z_down: np.ndarray
) -> float:
    """How far apart the two directions are on the IQ plane, in loop radii."""
    # Onto one grid. Exact where the grids agree, which for two directions of
    # one sweep is everywhere — the interpolation is for the case where a
    # re-centring or a dropped point has moved one of them.
    on_up = np.interp(f_up, f_down, z_down.real) + 1j * np.interp(
        f_up, f_down, z_down.imag
    )

    radius = float(np.max(np.abs(z_up - z_up.mean())))
    if radius == 0:
        raise ValueError(
            "The upward sweep is a single point in IQ — no loop, so no scale "
            "to measure a discrepancy against."
        )

    return float(np.max(np.abs(z_up - on_up)) / radius)


def _separation_in_magnitude(
    f_up: np.ndarray, z_up: np.ndarray, f_down: np.ndarray, z_down: np.ndarray
) -> float:
    """How far apart the two directions' ``|S21|`` curves are, in dip depths.

    The magnitudes are interpolated, not the complex traces — the curve being
    compared is the one you would plot, so a phase difference between the
    passes cannot leak into the answer through the interpolation either.

    The scale is the upward sweep's own depth, peak to trough: the deepest the
    resonance got minus the baseline it sat on. That is the magnitude plane's
    equivalent of the loop radius, and it is what makes one threshold portable
    between a deep resonator and a shallow one.
    """
    up, down = np.abs(z_up), np.abs(z_down)
    on_up = np.interp(f_up, f_down, down)

    depth = float(np.ptp(up))
    if depth == 0:
        raise ValueError(
            "The upward sweep's |S21| is flat — no dip, so no scale to measure "
            "a discrepancy against."
        )

    return float(np.max(np.abs(up - on_up)) / depth)


def _spikes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every local maximum in *values*, and how far each stands out of its own
    neighbourhood. ``prominence=0`` keeps them all; the bar is applied after."""
    peaks, properties = find_peaks(values, prominence=0)
    return peaks, properties["prominences"]


def _tallest(prominences: np.ndarray) -> float:
    """The most prominent spike, or ``0.0`` where the trace has no spike at all.

    Zero rather than ``None`` because it is the honest answer — nothing stood
    out — and because it keeps the entry a number that can be compared against
    the threshold and plotted across amplitude steps like any other.
    """
    return float(prominences.max()) if len(prominences) else 0.0


def _noise_floor(values: np.ndarray) -> float:
    """How much *values* scatters, without the jump being allowed to say.

    The median absolute deviation, scaled by 1.4826 so that it reads as a
    standard deviation would on Gaussian noise. That scaling is what lets
    ``noise_gate_factor`` be thought of in sigmas.

    Robust is the whole point. A bifurcated sweep puts two samples of order the
    whole span into a hundred-sample trace, which lifts a standard deviation
    far enough that a jump measured against it comes out barely above where
    noise does. A median does not move for two samples out of a hundred, so
    what comes back is the floor the jump stands on rather than the floor plus
    the jump.

    ``0.0`` for a trace that does not scatter at all, which disables the gate
    rather than dividing by it.
    """
    return float(np.median(np.abs(values - np.median(values))) * 1.4826)


def _paired(up: np.ndarray, down: np.ndarray) -> bool:
    """Did an up-spike get followed closely by a down-spike? Up first.

    *Any* such pair, not the first spike of each list. Filtering by prominence
    can only ever add spikes as the bar comes down, so asking whether any pair
    exists makes the verdict monotone in the bar — lowering a threshold cannot
    take a detection away. Comparing only ``up[0]`` against ``down[0]`` does not
    have that property: admitting one more spike at a lower index displaces the
    first, and a pair that was matching stops matching.

    Within :data:`MAX_SPIKE_SEPARATION` samples rather than exactly one, so a
    jump that a sample landed partway across is still recognized.
    """
    if not (len(up) and len(down)):
        return False
    separation = down[np.newaxis, :] - up[:, np.newaxis]
    return bool(((separation >= 1) & (separation <= MAX_SPIKE_SEPARATION)).any())


# ─── Which frequency ──────────────────────────────────────────────────────────


def find_bias_frequency(entry: Mapping, *, method: str = "iq_derivative") -> float:
    """Where in this sweep the tone belongs.

    The sweep centre is where we looked; this is where the resonance turned out
    to be. Both methods return a point of the measured grid — the tone is
    quantized onto the hardware grid afterwards, by ``BiasPoint``, and that
    grid is finer than any sweep worth taking.

    ``"iq_derivative"`` (the default)
        The frequency of maximum ``|dI/df + j·dQ/df|`` — where the IQ trace
        moves fastest per hertz, which is where a small shift in the resonance
        makes the largest signal. That is the point you want to sit on, and it
        is measured off the trace itself without asking a fit to converge.
    ``"minimum"``
        The frequency of minimum ``|S21|`` — the bottom of the dip. Says
        nothing about responsivity but survives traces the derivative method
        finds noisy, and is the one to reach for when a sweep is coarse.

    Both take a whole sweep entry rather than two arrays, which is what leaves
    room for the fitted ``fr`` to join them: that method reads the entry's
    ``fits``, not its trace.

    Whether the answer is *plausible* is not asked here — the answer is always
    a point of the trace, and judging it needs the sweep centre and a tolerance.
    :func:`find_bias_points` does that, through its ``max_distance_hz``. By hand
    it is one subtraction: ``frequency_hz - entry["original_center_frequency"]``.

    Args:
        entry: one sweep, as ``multisweep`` returns it.
        method: from :data:`FREQUENCY_METHODS`.

    Returns:
        float: the frequency in Hz, un-quantized.

    Raises:
        ValueError: for an unknown *method*, or an entry without a usable trace.
    """
    _check_method("method", method, FREQUENCY_METHODS)
    return float(_FREQUENCY[method](entry))


def iq_arc_speed(entry: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """``|dI/df + j·dQ/df|`` along one sweep — what ``"iq_derivative"`` maximizes.

    A reader, in the sense :mod:`rfmux.tuning.fits` uses the word: nothing
    stores this, because it is a function of the trace, and a plot of what a
    method looked at should be the thing the method looked at rather than a
    re-derivation of it that might differ.

    Off the same splines the calibration comes from, evaluated on the sweep's
    own frequencies, in the units of ``iq_counts`` per hertz.

    Args:
        entry: one sweep, as ``multisweep`` returns it.

    Returns:
        tuple: ``(frequencies, speed)``, both ascending in frequency — which is
        the reverse of a downward sweep's own order.

    Raises:
        ValueError: for a trace too short or too degenerate to differentiate.
    """
    frequencies, iq = _sorted_trace(entry, "iq_counts")
    speed = _arc_length_speed(frequencies, iq)
    if speed is None:
        raise ValueError(
            "This sweep is too short, or its frequencies repeat, so there is "
            "no derivative to take. At least four distinct points are needed."
        )
    return frequencies, speed


def normalized_arc_speed(entry: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """How far the IQ trace moves per hertz, point to point, in units of itself.

    What :func:`bifurcated_by_derivative` differentiates and looks for spikes
    in, so plotting this — and ``np.diff`` of it — is how you see what that test
    saw, and how you pick its factor.

    One pair of sweep points at a time rather than off a spline, and with I and
    Q each divided by their own range first. See the detector for why both of
    those matter.

    Args:
        entry: one sweep, as ``multisweep`` returns it.

    Returns:
        tuple: ``(frequencies, speed)``, one shorter than the sweep. The
        frequencies are the midpoints of the point pairs, because that is where
        a difference between two points belongs.

    Raises:
        ValueError: for a trace too short or too degenerate to difference.
    """
    frequencies, iq = _sorted_trace(entry, "iq_counts")
    speed = _point_to_point_speed(frequencies, iq)
    if speed is None:
        raise ValueError(
            "This sweep is too short, or has a flat I or Q axis, or visits a "
            "frequency twice — there is no point-to-point speed to take."
        )
    return 0.5 * (frequencies[:-1] + frequencies[1:]), speed


def _frequency_by_iq_derivative(entry: Mapping) -> float:
    """The frequency of maximum IQ arc-length speed."""
    frequencies, speed = iq_arc_speed(entry)
    return float(frequencies[int(np.argmax(speed))])


def _frequency_by_minimum(entry: Mapping) -> float:
    """The frequency of minimum |S21|."""
    frequencies, iq = _sorted_trace(entry, "iq_counts")
    if len(frequencies) < 1:
        raise ValueError("This sweep has no points, so it has no minimum.")
    return float(frequencies[int(np.argmin(np.abs(iq)))])


# ─── The calibration at that frequency ────────────────────────────────────────


def iq_derivative_splines(frequencies: np.ndarray, iq: np.ndarray):
    """Cubic splines through I(f) and Q(f), differentiated.

    The building block under both the arc-length speed and the calibration, so
    that the frequency a bias point is placed at and the derivatives read off
    it come from one interpolation rather than two.

    Args:
        frequencies: Hz. Sorted internally, so a downward sweep is fine.
        iq: the complex trace, in whatever units the answer should be per-hertz
            in.

    Returns:
        tuple: ``(dI_df, dQ_df)``, each callable at a frequency or an array of
        them.

    Raises:
        ValueError: for fewer than four points, or repeated frequencies —
            either way there is no cubic spline to fit.
    """
    order = np.argsort(frequencies)
    frequencies = np.asarray(frequencies, dtype=float)[order]
    iq = np.asarray(iq)[order]

    if len(frequencies) < 4:
        raise ValueError(
            f"A cubic spline needs at least four points; this sweep has "
            f"{len(frequencies)}."
        )
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError(
            "This sweep visits the same frequency twice, so there is no "
            "single-valued I(f) to interpolate."
        )

    return (
        CubicSpline(frequencies, iq.real).derivative(),
        CubicSpline(frequencies, iq.imag).derivative(),
    )


def iq_derivatives_at(entry: Mapping, frequency_hz: float) -> tuple[float, float]:
    """``(dI_df, dQ_df)`` in V/Hz, at one frequency of one sweep.

    The calibration a bias point carries:
    :attr:`~rfmux.core.resonators.BiasPoint.df_calibration` is
    ``1/(dI_df + j·dQ_df)``, the Hz/V factor that turns a measured voltage
    excursion into a frequency shift, and it derives from these rather than
    being stored beside them.

    Reads ``iq_volts`` and nothing else. The units are the whole point here —
    counts per hertz would be a number of the right magnitude and the wrong
    meaning, and downstream has no way to tell the two apart.

    Args:
        entry: one sweep, as ``multisweep`` returns it.
        frequency_hz: where to evaluate. Normally the bias frequency *after*
            quantization, so the calibration belongs to the tone that will be
            played.

    Returns:
        tuple: ``(dI_df, dQ_df)`` in V/Hz.

    Raises:
        ValueError: if the entry has no ``iq_volts``, or no interpolable trace.
    """
    if entry.get("iq_volts") is None:
        raise ValueError(
            "This sweep has no 'iq_volts', so a calibration read off it would "
            "be in counts per hertz. Sweeps carry volts as measured; an entry "
            "without them predates that and cannot be calibrated."
        )
    frequencies, iq = _sorted_trace(entry, "iq_volts")
    dI_df, dQ_df = iq_derivative_splines(frequencies, iq)
    return float(dI_df(frequency_hz)), float(dQ_df(frequency_hz))


# ─── Reading the sweeps ───────────────────────────────────────────────────────


def _directions_swept(sweeps) -> set[str]:
    """Every sweep direction present in one module's result.

    Read once, up front, so that a request the whole call cannot satisfy —
    hysteresis on a single direction — is refused before anything is measured
    against it.
    """
    return {
        direction
        for by_direction in _iterations(sweeps).values()
        for direction in by_direction
    }


def _catalog_swept(sweeps) -> ResonatorCatalog:
    """The catalog this sweep recorded, rebuilt from its snapshot."""
    snapshot = (sweeps.get("call_params") or {}).get("catalog")
    if snapshot is None:
        raise ValueError(
            "No catalog to bias: none was passed, and this result came from a "
            "bare center_frequencies sweep, which has no resonators to put an "
            "operating point on. Pass catalog=."
        )
    return ResonatorCatalog.from_dict(snapshot)


def _directions(entries: Mapping[str, dict]) -> list[dict]:
    """The sweeps of one amplitude step, upward first when it is there."""
    return [
        entries[d]
        for d in sorted(entries, key=lambda d: (d != PREFERRED_DIRECTION, d))
    ]


def _entry_for(entries: Mapping[str, dict], direction: str | None) -> dict:
    """The one sweep of an amplitude step to measure the bias point on."""
    if direction is None:
        return _directions(entries)[0]
    if direction not in entries:
        raise ValueError(
            f"The chosen amplitude step has no {direction!r} sweep; it has "
            f"{sorted(entries)}."
        )
    return entries[direction]


def _amplitude_of(entries: Mapping[str, dict]) -> float:
    """What this step probed at. Every direction of a step shares one amplitude."""
    return float(_directions(entries)[0]["sweep_amplitude"])


def _sorted_trace(entry: Mapping, key: str) -> tuple[np.ndarray, np.ndarray]:
    """One entry's frequencies and IQ, in ascending frequency order.

    Downward sweeps arrive high-to-low, and everything here — splines,
    differences, interpolation — wants them the other way round. Sorting on the
    way in rather than asking each caller to remember is what keeps a downward
    sweep from quietly producing sign-flipped derivatives.
    """
    frequencies = entry.get("frequencies")
    iq = entry.get(key)
    if frequencies is None or iq is None:
        raise ValueError(
            f"This sweep entry has no "
            f"{'frequencies' if frequencies is None else key}. Its keys are "
            f"{sorted(entry)}."
        )

    frequencies = np.asarray(frequencies, dtype=float)
    iq = np.asarray(iq)
    if len(frequencies) != len(iq):
        raise ValueError(
            f"This sweep has {len(frequencies)} frequencies and {len(iq)} "
            f"{key} — they describe different measurements."
        )

    order = np.argsort(frequencies)
    return frequencies[order], iq[order]


def _point_to_point_speed(
    frequencies: np.ndarray, iq: np.ndarray
) -> np.ndarray | None:
    """How far the IQ trace moves per hertz, point to point, in units of itself.

    I and Q are each divided by their own range before differencing, so the
    result depends on the *shape* of the loop and not on how deep the resonance
    is or how big the readout gain was. That is what lets one spike factor mean
    the same thing on every resonator of an array.

    Deliberately finite differences rather than the spline
    :func:`_arc_length_speed` uses: a spline smooths a jump across several
    samples, which is exactly the feature this is trying to catch — and it
    smears the down-spike far enough from the up-spike that the adjacency test
    stops finding the pair.

    None when the trace is degenerate: a flat I or Q axis, or a repeated
    frequency.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    i_range = float(np.ptp(iq.real))
    q_range = float(np.ptp(iq.imag))
    if i_range == 0 or q_range == 0 or len(frequencies) < 3:
        return None

    spacing = np.diff(frequencies)
    if np.any(spacing == 0):
        return None

    return (
        np.sqrt(
            np.diff(iq.real / i_range) ** 2 + np.diff(iq.imag / q_range) ** 2
        )
        / np.abs(spacing)
    )


def _arc_length_speed(frequencies: np.ndarray, iq: np.ndarray) -> np.ndarray | None:
    """How fast the IQ trace moves per hertz, at each point of the sweep.

    ``|dI/df + j·dQ/df|`` off the splines, evaluated on the sweep's own grid,
    so a peak in it is a frequency that was actually measured — and in the same
    units as the trace, because this one is read for its position rather than
    compared against a threshold. None when the sweep cannot be splined at all.
    """
    try:
        dI_df, dQ_df = iq_derivative_splines(frequencies, iq)
    except ValueError:
        return None
    return np.abs(dI_df(frequencies) + 1j * dQ_df(frequencies))


#: Name → detector, so a new way of spotting bifurcation is a function and an
#: entry here. Each takes one amplitude step's ``{direction: entry}``, because
#: whether a test needs one direction or both is the test's business.
_BIFURCATION = {
    "derivative": bifurcated_by_derivative,
    "hysteresis": bifurcated_by_hysteresis,
    "both": bifurcated_by_either,
}

#: Name → what the hysteresis test measures the two directions apart in. Each
#: takes both traces already sorted and returns one number, normalized by the
#: scale of its own plane so that one threshold travels between resonators.
_HYSTERESIS_COMPARISON = {
    "magnitude": _separation_in_magnitude,
    "iq": _separation_in_iq,
}

#: Name → placer, each taking a whole sweep entry. The fitted ``fr`` joins here
#: when it is wired up; it reads the entry's ``fits`` rather than its trace,
#: which is why these do not take bare arrays.
_FREQUENCY = {
    "iq_derivative": _frequency_by_iq_derivative,
    "minimum": _frequency_by_minimum,
}
