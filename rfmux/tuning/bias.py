"""Choose bias amplitudes, frequencies, and calibrations from measured sweeps.

:func:`find_bias_points` takes one module's multisweep result and returns a
:class:`BiasReport` with a new catalog. It also stores the report in the input
block's ``bias_report`` field. Check ``report.flagged`` before applying the
catalog with ``crs.apply_bias``.

The amplitude search selects the measured step below bifurcation. Frequency
selection and calibration use a sweep at that amplitude. Detector settings
and fallback rules are documented on the functions below.

The derivative bifurcation test and max-arc-speed frequency method are adapted
from hidfmux ``analysis/find_bias.py`` (Maclean Rouble, McGill Cosmology), via
``algorithms/measurement/bias_kids.py``.
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
    "NEEDS_BOTH_DIRECTIONS",
    "FLAG_KINDS",
    "FLAG_BIFURCATED_AT_QUIETEST",
    "FLAG_NEVER_BIFURCATED",
    "FLAG_OFF_CENTRE",
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
    "iq_derivatives",
    "normalized_arc_speed",
    "iq_derivative_splines",
    "iq_derivatives_at",
]

#: "both" detects bifurcation if either detector fires; it needs both directions.
BIFURCATION_METHODS = ("both", "derivative", "hysteresis")

#: Methods requiring upward and downward sweeps.
NEEDS_BOTH_DIRECTIONS = ("hysteresis", "both")

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

#: Short flag labels for plots and tables; _concern supplies explanations.
FLAG_BIFURCATED_AT_QUIETEST = "already bifurcated"
FLAG_NEVER_BIFURCATED = "No bifurcation observed in this run"
FLAG_OFF_CENTRE = "freq out of bounds"
FLAG_KINDS = (FLAG_BIFURCATED_AT_QUIETEST, FLAG_NEVER_BIFURCATED, FLAG_OFF_CENTRE)

#: Allow a jump to cross one or two bins when a sample falls partway across it.
MAX_SPIKE_SEPARATION = 2


# ─── Results ──────────────────────────────────────────────────────────────────


class BifurcationCheck(NamedTuple):
    """A detector's verdict and the metrics used to reach it.

    ``metric`` maps quantity names to values; ``threshold`` is their comparison
    threshold. See each detector for units and conditions. For ``"both"``,
    metrics are normalized by each detector's threshold and ``parts`` retains
    the original checks.
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
    """The selected bias point, checks, and any concern for one resonator.

    ``flagged_kind`` groups concerns for plotting; ``flagged_because`` explains
    the finding. Both are None when no concern was found. Calibration
    derivatives are in V/Hz at ``frequency_hz``.
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
    flagged_kind: str | None = None  # one of FLAG_KINDS

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
            "flagged_kind": self.flagged_kind,
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
            flagged_kind=d.get("flagged_kind"),
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
        """Bias points needing review before applying them."""
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
    *,
    amplitude_method: str = "derivative",
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
    """Choose a bias point and calibration for each resonator in a multisweep.

    Reads the catalog recorded in the sweep, leaving it and the measured
    entries unchanged. Returns a new catalog and stores the report in
    ``sweeps["bias_report"]``, replacing any previous report.

    ``report.flagged`` identifies a bifurcated lowest step, a drive with no
    known bifurcation above it, or a frequency outside ``max_distance_hz``.
    A prior bifurcation amplitude is retained when this run observes none.
    Clear it with ``catalog.clear_bifurcations()`` before taking new sweeps.

    Args:
        sweeps: one module's block, ``results[crs.module[m].index()]``.
        amplitude_method: ``"derivative"`` (default) works with one sweep
            direction. ``"both"`` and ``"hysteresis"`` require both.
        frequency_method: ``"iq_derivative"`` or ``"minimum"``; see
            :func:`find_bias_frequency`.
        direction: sweep used for frequency and calibration. None prefers
            ``"upward"``, otherwise uses the available direction. Amplitude
            detection still uses all directions required by its method.
        spike_prominence_factor: passed to :func:`bifurcated_by_derivative`.
        noise_gate_factor: passed to :func:`bifurcated_by_derivative`.
        max_discrepancy: passed to :func:`bifurcated_by_hysteresis`.
        compare: passed to :func:`bifurcated_by_hysteresis`.
        max_distance_hz: maximum allowed offset from the sweep centre. Beyond
            it, use the centre and flag the finding. None imposes no limit.
        save: save the sweeps with the report, updating their existing file
            or creating one. None uses ``store.autosave_enabled()``.
        label: filename label for a first save; existing filenames are kept.

    Returns:
        BiasReport: new catalog and findings in bias-frequency order. Each
        bias point carries derivatives in V/Hz at its grid-aligned frequency
        and the sweep used to compute them. IQ rotation is left unset.

    Raises:
        TypeError: the input is a whole module container.
        ValueError: a method or direction is invalid, required directions are
            missing, or the result has no catalog.
        KeyError: a catalog resonator has no sweep data.
    """
    # Everything the caller could have got wrong about the *whole* call is
    # checked here, once, before a single resonator is analysed. A thousand
    # copies of the same complaint is worse than one.
    _check_method("amplitude_method", amplitude_method, BIFURCATION_METHODS)
    _check_method("frequency_method", frequency_method, FREQUENCY_METHODS)
    _check_method("compare", compare, HYSTERESIS_COMPARISONS)

    directions = _directions_swept(sweeps)
    if (
        amplitude_method in NEEDS_BOTH_DIRECTIONS
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

    # The array being biased: the one these sweeps were taken from, which is
    # the only one they can speak for. Built fresh out of the snapshot, and it
    # is what we hand back, so the record in the file still reads as the
    # catalog that was swept.
    biased = _catalog_swept(sweeps)

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

    # Keep the calibration and its source trace at the quantized tone frequency.
    resonator.bias = replace(
        bias,
        dI_df=dI_df,
        dQ_df=dQ_df,
        bifurcated_at=(
            choice.bifurcated_at
            if choice.bifurcated_at is not None
            else resonator.bias.bifurcated_at
        ),
        bias_sweep=_stored_sweep(entry),
    )

    # 7. Finally, is this an operating point we actually established, or a
    #    default we fell back to? _concern is the one place that decides.
    flagged_kind, flagged_because = _concern(
        choice,
        known_bifurcation=resonator.bias.bifurcated_at,
        measured_hz=measured_hz,
        centre_hz=centre_hz,
        max_distance_hz=max_distance_hz,
    )
    return BiasFinding(
        name=resonator.name,
        iteration=choice.iteration,
        amplitude=choice.amplitude,
        frequency_hz=resonator.bias.frequency_hz,
        dI_df=dI_df,
        dQ_df=dQ_df,
        bifurcated_at=choice.bifurcated_at,
        checks=choice.checks,
        flagged_kind=flagged_kind,
        flagged_because=flagged_because,
    )


def _concern(
    choice: AmplitudeChoice,
    *,
    known_bifurcation: float | None = None,
    measured_hz: float,
    centre_hz: float,
    max_distance_hz: float | None,
) -> tuple[str | None, str | None]:
    """Return the first concern as ``(kind, explanation)``, or ``(None, None)``.

    Checks lowest-step bifurcation, missing bifurcation above the chosen
    amplitude, then frequency bounds. A clean sweep below a retained
    bifurcation amplitude passes the amplitude check.
    """
    if choice.is_bifurcated_at_bias:
        return FLAG_BIFURCATED_AT_QUIETEST, (
            f"the quietest amplitude measured ({choice.amplitude:g}) was already "
            f"bifurcated, so there was nothing below it to fall back to"
        )
    if choice.bifurcated_at is None and (
        known_bifurcation is None or choice.amplitude >= known_bifurcation
    ):
        return FLAG_NEVER_BIFURCATED, (
            f"No bifurcation observed in this run up to {choice.amplitude:g}, "
            f"the loudest amplitude measured"
        )
    if _too_far(measured_hz, centre_hz, max_distance_hz):
        return FLAG_OFF_CENTRE, (
            f"the resonance came out {(measured_hz - centre_hz) / 1e3:+.1f} kHz "
            f"from the sweep centre, past the {max_distance_hz / 1e3:.1f} kHz "
            f"asked for — usually a neighbour in the span, or a resonance pulled "
            f"out of it — so the tone was left where the sweep was centred"
        )
    return None, None


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
            ``"derivative"``, detects jumps within each supplied sweep.
            ``"both"`` runs the derivative and hysteresis tests and takes
            either verdict, so it needs both sweep directions.
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
    """Detect a jump from adjacent positive and negative IQ-speed spikes.

    I and Q are normalized by their ranges. Point-to-point distance divided
    by frequency spacing gives the arc speed; its differences reveal jumps.
    A positive spike must be followed by a negative one within
    ``MAX_SPIKE_SEPARATION`` samples, with both prominences meeting the
    threshold. Any usable direction can trigger the verdict.

    Args:
        entries: one amplitude step, ``{direction: sweep_entry}``.
        spike_prominence_factor: threshold as a fraction of the arc-speed
            range. Larger values are less sensitive.
        noise_gate_factor: threshold as a multiple of the robust noise floor
            of the speed differences. Larger values are less sensitive;
            zero disables this gate. See :func:`_noise_floor`.

    Returns:
        BifurcationCheck: ``threshold`` is the larger of the range and noise
        thresholds. Metrics are ``positive_spike_prominence``,
        ``negative_spike_prominence`` (both in inverse Hz), and ``adjacency``
        (whether a qualifying pair exists). Missing spikes have prominence
        zero. Metrics describe a triggering direction if any, otherwise the
        direction with the largest spike prominence.

    Raises:
        ValueError: no direction contains a usable sweep.
    """
    verdict = False
    reported = None  # (rank, metric, threshold) of the closest direction so far
    for entry in _directions(entries):
        frequencies, iq = _sorted_trace(entry, "iq_counts")
        speed = _point_to_point_speed(frequencies, iq)
        if speed is None or len(speed) < 3:
            continue

        jumps = np.diff(speed)
        # Both thresholds are prominences in inverse Hz; require the larger.
        prominence_threshold = max(
            float(spike_prominence_factor * (speed.max() - speed.min())),
            float(noise_gate_factor * _noise_floor(jumps)),
        )

        # Retain below-threshold spikes too, so a missed detection is inspectable.
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
    """Detect bifurcation from disagreement between upward and downward sweeps.

    Args:
        entries: one amplitude step with ``"upward"`` and ``"downward"``.
        max_discrepancy: largest allowed normalized separation. Tune against
            known sweeps; magnitude and IQ comparisons use different scales.
        compare: ``"magnitude"`` compares |S21| in units of the upward trace's
            dip depth, ignoring phase. ``"iq"`` compares complex IQ distance
            in loop radii and is sensitive to phase or frequency drift too.

    Returns:
        BifurcationCheck: ``metric["max_separation"]`` is the largest
        normalized separation. Bifurcation means it exceeds
        ``threshold=max_discrepancy``.

    Raises:
        ValueError: unknown comparison, missing or unusable direction, or
            frequency ranges that do not overlap.
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
    """Run both detectors and report bifurcation if either detects it.

    Both sweep directions are required. This catches either kind of evidence,
    but also accepts false positives from either detector.

    Args:
        entries: one amplitude step with ``"upward"`` and ``"downward"``.
        spike_prominence_factor: passed to :func:`bifurcated_by_derivative`.
        noise_gate_factor: passed to :func:`bifurcated_by_derivative`.
        max_discrepancy: passed to :func:`bifurcated_by_hysteresis`.
        compare: passed to :func:`bifurcated_by_hysteresis`.

    Returns:
        BifurcationCheck: ``parts`` contains both original checks.
        ``metric`` prefixes each quantity with its method and divides numeric
        values by that method's threshold; boolean adjacency is unchanged.
        The combined threshold is 1.0. For a zero source threshold, positive
        values become infinity and other values become zero.

    Raises:
        ValueError: either detector rejects the input or settings.
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
    """Divide a metric by its threshold, preserving boolean conditions.

    A zero threshold gives infinity for positive values and zero otherwise.
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
    """Return the largest prominence, or zero if there are no spikes."""
    return float(prominences.max()) if len(prominences) else 0.0


def _noise_floor(values: np.ndarray) -> float:
    """Estimate Gaussian-equivalent noise with median absolute deviation × 1.4826.

    This limits the influence of jump outliers on the noise estimate.
    A constant trace returns zero, disabling the noise gate for that trace.
    """
    return float(np.median(np.abs(values - np.median(values))) * 1.4826)


def _paired(up: np.ndarray, down: np.ndarray) -> bool:
    """Find any positive spike followed within ``MAX_SPIKE_SEPARATION`` samples
    by a negative spike.

    Checking all pairs ensures that lowering the prominence threshold cannot
    remove a detection by admitting an earlier, unrelated spike.
    """
    if not (len(up) and len(down)):
        return False
    separation = down[np.newaxis, :] - up[:, np.newaxis]
    return bool(((separation >= 1) & (separation <= MAX_SPIKE_SEPARATION)).any())


# ─── Which frequency ──────────────────────────────────────────────────────────


def find_bias_frequency(entry: Mapping, *, method: str = "iq_derivative") -> float:
    """Return a bias frequency in Hz from one sweep's measured grid.

    ``"iq_derivative"`` selects the maximum spline-derived IQ speed;
    ``"minimum"`` selects the minimum |S21|. This function does not quantize
    the result or check its distance from the sweep centre; ``find_bias_points``
    handles those steps.

    Raises ValueError for an unknown method or an unusable trace.
    """
    _check_method("method", method, FREQUENCY_METHODS)
    return float(_FREQUENCY[method](entry))


def iq_arc_speed(entry: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies, |dI/df + j*dQ/df|)`` in Hz and counts/Hz.

    Uses :func:`iq_derivatives`; both arrays follow ascending frequency.
    Raises ValueError if the trace cannot be differentiated.
    """
    frequencies, dI_df, dQ_df = iq_derivatives(entry)
    return frequencies, np.abs(dI_df + 1j * dQ_df)


def iq_derivatives(entry: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(frequencies, dI_df, dQ_df)`` in ascending frequency order.

    Derivatives are evaluated from splines through ``iq_counts`` and are in
    counts/Hz. For calibration in V/Hz, use :func:`iq_derivatives_at`.
    Raises ValueError if the trace cannot be differentiated.
    """
    frequencies, iq = _sorted_trace(entry, "iq_counts")
    try:
        dI_df, dQ_df = iq_derivative_splines(frequencies, iq)
    except ValueError as exc:
        raise ValueError(
            f"This sweep cannot be differentiated: {exc}"
        ) from exc
    return frequencies, dI_df(frequencies), dQ_df(frequencies)


def normalized_arc_speed(entry: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized point-to-point IQ speed for the derivative detector.

    I and Q are each divided by their range before differencing. Returns
    ``(frequencies, speed)`` at ascending pair midpoints, one element shorter
    than the sweep, with speed in inverse Hz. Raises ValueError for an
    unusable trace.
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
    """Evaluate ``(dI_df, dQ_df)`` in V/Hz at ``frequency_hz``.

    Splines use the entry's ``iq_volts``. Pass the quantized bias frequency to
    calibrate the tone that will be applied. ``BiasPoint.df_calibration`` is
    the reciprocal complex derivative, in Hz/V.

    Raises ValueError for missing volts or a trace that cannot be interpolated.
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


def _stored_sweep(entry: Mapping) -> dict:
    """Keep the calibration trace and context listed in ``BiasPoint.BIAS_SWEEP_KEYS``.

    Arrays are shared with the input, so a report saved alongside its sweeps
    can reuse them through pickle's memoization.
    """
    return {k: entry[k] for k in BiasPoint.BIAS_SWEEP_KEYS if k in entry}


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
    """The catalog this sweep recorded, rebuilt from its snapshot.

    A fresh object every call, which is what lets the caller be handed it: the
    snapshot in the file is a dict and stays one, so the report's catalog is
    never the record of what was swept.
    """
    snapshot = (sweeps.get("call_params") or {}).get("catalog")
    if snapshot is None:
        raise ValueError(
            "No catalog in these sweeps to bias. Every multisweep records one "
            "— a bare center_frequencies call generates one from the list — so "
            "this is a result from before that was so (schema_version 5 or "
            "earlier). Re-sweep, or build the catalog yourself and put it in "
            "call_params['catalog'] as ResonatorCatalog.to_dict() output."
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
    :func:`iq_derivatives` uses: a spline smooths a jump across several
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
