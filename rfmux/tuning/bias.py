"""
Choose an operating point for every resonator, from sweeps already measured.

Bias finding asks two questions about each resonator, in that order:

**Which amplitude?** The one just below where the resonator bifurcates — as
much probe power as it will take while its sweep still describes a resonance.
Answering it needs a ``multiamp_multisweep``, because "just below" is only
meaningful against amplitude steps that were actually measured, and the answer
is one of them.

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
``sweeps[crs.module[m].index()]``, from either sweep macro. A single
``multisweep`` is one amplitude step, which is a legitimate thing to bias off
if you already know the amplitude; the search then has nothing to go back to
and says so.

Out: a :class:`BiasReport`, whose ``catalog`` is a **new**
:class:`~rfmux.core.resonators.ResonatorCatalog` carrying the operating points
that were found::

    report = find_bias_points(sweeps)
    report.catalog["R0001"].bias.frequency_hz
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
    "BifurcationCheck",
    "AmplitudeChoice",
    "BiasFinding",
    "BiasReport",
    "find_bias_points",
    "find_bias_amplitude",
    "find_bias_frequency",
    "bifurcated_by_derivative",
    "bifurcated_by_hysteresis",
    "iq_arc_speed",
    "normalized_arc_speed",
    "iq_derivative_splines",
    "iq_derivatives_at",
]

#: How :func:`find_bias_amplitude` decides an amplitude step is bifurcated,
#: and the default. ``"hysteresis"`` needs both sweep directions; ``"derivative"``
#: works on one.
BIFURCATION_METHODS = ("derivative", "hysteresis")

#: How :func:`find_bias_frequency` places the tone inside the chosen sweep, and
#: the default.
FREQUENCY_METHODS = ("iq_derivative", "minimum")

#: The direction the bias frequency is measured on when an amplitude step has
#: more than one and the caller did not say.
PREFERRED_DIRECTION = "upward"


# ─── Results ──────────────────────────────────────────────────────────────────


class BifurcationCheck(NamedTuple):
    """One method's verdict on one amplitude step, and the numbers behind it.

    Carries what it compared as well as the verdict, because the settings that
    produce them are knobs a user has to turn against their own array: a
    detector that only ever says yes or no gives them nothing to turn them by.

    ``metric`` and ``threshold`` are in whatever units the method works in —
    see the detector for what they mean, and for how closely the verdict tracks
    the comparison. Above the threshold is not on its own a positive verdict:
    ``"derivative"`` also requires the spikes it found to be adjacent, and in
    the right order.
    """

    method: str
    bifurcated: bool
    metric: float
    threshold: float

    def to_dict(self) -> dict:
        """Plain builtins only — files never contain these classes."""
        return {
            "method": self.method,
            "bifurcated": bool(self.bifurcated),
            "metric": float(self.metric),
            "threshold": float(self.threshold),
        }

    @classmethod
    def from_dict(cls, d) -> "BifurcationCheck":
        return cls(
            method=d["method"],
            bifurcated=bool(d["bifurcated"]),
            metric=float(d["metric"]),
            threshold=float(d["threshold"]),
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
    def is_bifurcated(self) -> bool:
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
    resonator in channel order. The settings come back here rather than being
    copied onto a thousand bias points, as with
    :class:`~rfmux.tuning.fits.FitReport` — recording them alongside the data
    is the output folder's job.
    """

    # Stamped into to_dict output and required exactly by from_dict, so a file
    # from another version of this module fails loudly rather than being half
    # understood. Bump whenever the dict shape changes in a way from_dict
    # cannot absorb.
    SCHEMA_VERSION = 1

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
    amplitude_method: str = "derivative",
    frequency_method: str = "iq_derivative",
    direction: str | None = None,
    spike_prominence_factor: float = 0.5,
    max_discrepancy: float = 0.25,
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
        sweeps: **one module's** value out of what ``multisweep`` or
            ``multiamp_multisweep`` returned — ``sweeps[crs.module[m].index()]``.
            The whole container, keyed by module, is refused: a report is about
            one module, and which one is your choice to make.
        catalog: the resonators to bias, and the source of everything the new
            catalog keeps unchanged — names, channels, the module, the
            separation rule. Defaults to the catalog snapshot recorded in the
            sweep's ``call_params``, which is the usual case: you are biasing
            the array you swept. A catalog holding a resonator these sweeps do
            not cover raises, because the two were then not measured together.
        amplitude_method: which bifurcation test the amplitude search uses,
            from :data:`BIFURCATION_METHODS`. ``"hysteresis"`` requires the
            sweeps to have been taken in both directions.
        frequency_method: where in the chosen sweep the tone goes, from
            :data:`FREQUENCY_METHODS`.
        direction: which direction's sweep to measure the bias frequency and
            the calibration on. None takes ``"upward"`` when it is there, and
            the only direction there is otherwise. The amplitude search is
            unaffected — a detector sees every direction of its own step.
        spike_prominence_factor: passed to :func:`bifurcated_by_derivative`.
        max_discrepancy: passed to :func:`bifurcated_by_hysteresis`.
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
        :class:`BiasFinding` per catalog resonator, in channel order, saying
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

    directions = _directions_swept(sweeps)
    if amplitude_method == "hysteresis" and not {"upward", "downward"} <= directions:
        raise ValueError(
            f"The 'hysteresis' method compares an upward sweep against a "
            f"downward one, and this result holds only {sorted(directions)}. "
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
        max_discrepancy=max_discrepancy,
    )

    # One finding per resonator, in channel order, because iterating a catalog
    # is iterating it in channel order.
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
            "max_discrepancy": max_discrepancy,
            "max_distance_hz": max_distance_hz,
        },
    )
    # Into the sweeps it was found from, so that saving updates that file
    # rather than starting a second one. to_dict, not the report: a pickled
    # class records its import path and skips its constructor coming back, so
    # the file would outlive a rename only by restoring into a state BiasReport
    # would have refused to build.
    sweeps["bias_report"] = report.to_dict()
    store.maybe_save(sweeps, _measurement_type(sweeps), save=save, label=label)
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
    if choice.is_bifurcated:
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
    method: str = "derivative",
    spike_prominence_factor: float = 0.5,
    max_discrepancy: float = 0.25,
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
      chosen and :attr:`AmplitudeChoice.is_bifurcated` says so. The schedule
      started too high.

    Args:
        iterations: one resonator's sweeps, ``{iteration: {direction: entry}}``
            — what
            :func:`~rfmux.tuning.sweep_results.collect_amplitude_iterations_for`
            returns. A single ``multisweep`` gives one amplitude step, which is
            a legitimate thing to hand over.
        method: which test, from :data:`BIFURCATION_METHODS`.
        spike_prominence_factor: passed to :func:`bifurcated_by_derivative`.
        max_discrepancy: passed to :func:`bifurcated_by_hysteresis`.

    Returns:
        AmplitudeChoice: the iteration and amplitude to bias at, where
        bifurcation was first seen, and each examined step's verdict.

    Raises:
        ValueError: for an unknown *method*, or for no sweeps at all.
    """
    _check_method("method", method, BIFURCATION_METHODS)
    if not iterations:
        raise ValueError("No sweeps here, so there is no amplitude to choose.")

    detector = _BIFURCATION[method]
    # Each detector takes only its own setting, so a caller passing both does
    # not hand the hysteresis test a spike threshold it has no use for.
    settings = (
        {"spike_prominence_factor": spike_prominence_factor}
        if method == "derivative"
        else {"max_discrepancy": max_discrepancy}
    )

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

    How big a spike has to be is set relative to the sweep itself: a spike has
    to stand out from its surroundings — scipy's *prominence* — by more than
    *spike_prominence_factor* times the span of the arc-length speed. The factor
    multiplies, so it reads the way it behaves: the default of 0.5 asks a spike
    to stand a full half of the speed's whole range out of its own
    neighbourhood, and raising it asks for more, which is less sensitive.

    It is the same bar the GUI has always applied, but **not the same number**:
    the GUI *divided* the span by a ``spike_prominence_factor`` of 2.0, so
    turning its knob up made the test more sensitive. Same argument name, the
    reciprocal value, arithmetic that no longer runs backwards. The bar itself
    has not been re-derived here.

    Scaling off the sweep is what makes one factor portable between resonators,
    and also what makes this test say yes too readily on a sweep with no
    resonance in it: the largest noise excursion is then the whole dynamic
    range, so it clears a threshold set as a fraction of itself. A sweep too
    coarse to resolve the resonance has the same problem for the opposite
    reason — a dip crossed in two samples is a discontinuity, and this test
    cannot tell that from a jump. Both show up as a ``metric`` only just over
    ``threshold``, which is why they are reported: a resonator that bifurcates
    has a jump that clears it by a wide margin, and the factor is what you
    raise when yours does not.

    Args:
        entries: one amplitude step, ``{direction: entry}``. Every direction
            present is tested and the step counts as bifurcated if any of them
            says so — a bifurcated resonator jumps whichever way the sweep
            runs, so needing both to agree would only lose the one that
            happened to catch it.
        spike_prominence_factor: the bar a spike has to clear, as a multiple
            of the span of the arc-length speed. Larger is less sensitive.

    Returns:
        BifurcationCheck: with ``metric`` the largest positive jump seen in any
        direction and ``threshold`` the bar that direction's spikes had to
        clear. The two are not quite the same quantity — a jump is measured
        from zero and a prominence from whatever the spike's own neighbourhood
        sits at — so read their ratio as the margin this sweep has, not as the
        verdict. The verdict needs more than a big spike anyway: the up-spike
        has to be followed by a down-spike, so a metric well over threshold
        with a negative verdict means the jump was there and the pattern was
        not.

    Raises:
        ValueError: if none of the directions holds a usable sweep.
    """
    verdict = False
    biggest = None  # the (metric, threshold) of the direction that jumped most
    for entry in _directions(entries):
        frequencies, iq = _sorted_trace(entry, "iq_counts")
        speed = _point_to_point_speed(frequencies, iq)
        if speed is None or len(speed) < 3:
            continue

        jumps = np.diff(speed)
        prominence_threshold = float(
            spike_prominence_factor * (speed.max() - speed.min())
        )

        up, _ = find_peaks(jumps, prominence=prominence_threshold)
        down, _ = find_peaks(-jumps, prominence=prominence_threshold)

        # look for adjacent spikes, with one up before one down:
        # TODO we are assuming there will only be one spike here - is that okay?
        # maybe filtering for multiple spikes could cut down on getting tripped up
        ## by noisy data
        if len(up) and len(down) and down[0] == up[0] + 1:
            verdict = True

        # Reported as a pair from one direction, so the metric and the
        # threshold beside it come from the same sweep.
        if biggest is None or float(jumps.max()) > biggest[0]:
            biggest = (float(jumps.max()), prominence_threshold)

    if biggest is None:
        raise ValueError(
            "No usable sweep at this amplitude: every direction is too short "
            "have a shape, or has a degenerate frequency or IQ axis."
        )
    metric, threshold = biggest
    return BifurcationCheck(
        method="derivative", bifurcated=verdict, metric=metric, threshold=threshold
    )


def bifurcated_by_hysteresis(
    entries: Mapping[str, dict], *, max_discrepancy: float = 0.25
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
    the sweep, in units of the IQ loop's own radius, so it means the same thing
    for a deep resonator and a shallow one. Below bifurcation it is a noise
    figure. Above it, the traces are a good fraction of the loop apart.

    Args:
        entries: one amplitude step, ``{direction: entry}``. Both directions
            are required — this test *is* the comparison.
        max_discrepancy: how far apart the traces may be, in loop radii, before
            the step is called bifurcated. The default is a starting point
            rather than a measured number: pick it against your own array by
            reading ``metric`` across the amplitude steps of a resonator that
            is known to bifurcate, which is what it is reported for.

    Returns:
        BifurcationCheck: with ``metric`` the largest separation, in loop
        radii, and ``threshold`` the *max_discrepancy* it was compared against.

    Raises:
        ValueError: if either direction is missing or unusable, or if the two
            sweeps do not cover the same frequencies.
    """
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

    metric = float(np.max(np.abs(z_up - on_up)) / radius)
    return BifurcationCheck(
        method="hysteresis",
        bifurcated=metric > max_discrepancy,
        metric=metric,
        threshold=max_discrepancy,
    )


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


def _measurement_type(sweeps) -> str:
    """What to call the file, for sweeps that have never been in one.

    Bias finding writes back into the sweep it read, and that sweep is usually
    in a file already — in which case this only re-stamps the type it had. What
    it is for is the sweep taken with ``save=False`` and written for the first
    time here: it should be named after what it is, and a ladder is the only
    thing that records an ``amp_schedule``.
    """
    schedule = (sweeps.get("call_params") or {}).get("amp_schedule")
    return "multiamp_multisweep" if schedule else "multisweep"


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
}

#: Name → placer, each taking a whole sweep entry. The fitted ``fr`` joins here
#: when it is wired up; it reads the entry's ``fits`` rather than its trace,
#: which is why these do not take bare arrays.
_FREQUENCY = {
    "iq_derivative": _frequency_by_iq_derivative,
    "minimum": _frequency_by_minimum,
}
