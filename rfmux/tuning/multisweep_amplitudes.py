"""Sweeping one array at several amplitudes: the amplitudes, and the answers.

The pure half of the multi-amplitude multisweep.  Two things live here, and they
are the two ends of the same contract:

* :class:`AmplitudeSchedule` decides *at what amplitude is each resonator swept,
  on each pass?*
* :func:`pack_results` and the readers under it own the shape of what comes
  back, because the schedule's own ``to_dict()`` is *inside* that dict — a
  reader resolving ``ladder[iteration]`` has to agree with the schedule about
  what a rung means, and two files agreeing about one contract is one file too
  many.

The loop between them — the only part that needs a board — is
``rfmux.algorithms.measurement.multiamp_multisweep``.  Everything in this module
can be built, printed, validated and unit-tested with no hardware and no GUI in
sight.

**A step is one amplitude.**  Steps are numbered from 0 in the order they are
measured.  A step may be swept twice — once per frequency direction — but
direction is not this module's business: it is an axis the driver adds *beneath*
a step, never fused into the step index.  So ``len(schedule)`` is a count of
amplitudes, not of sweeps.

Two fields decide every step: a **base** amplitude per resonator, and the steps
applied to it.  The base is the catalog's own ``bias.amplitude`` by default, or
one number for everything, or one per resonator by name.  Steps are either
**relative** — multiplying the base, so every resonator keeps its own scale — or
**absolute**, which *are* the amplitude and apply to all of them equally.

Which is why the absolute forms take no base: there would be nothing left for a
base to contribute.  Per-resonator *absolute* sequences are deliberately not
representable — the proportional case, which is what a bifurcation walk wants, is
``multiplicative(base={...})``, and non-proportional ones have yet to find a use
that justifies the extra state.

No iteration — one pass — is the plain constructor::

    AmplitudeSchedule()                        # each resonator's own amplitude
    AmplitudeSchedule(0.005)                   # one amplitude for all
    AmplitudeSchedule({"R0001": 0.004, ...})   # per resonator

and the iterating forms are classmethods::

    AmplitudeSchedule.multiplicative(0.5, 2.0, 5)              # × each resonator's own
    AmplitudeSchedule.multiplicative(0.5, 2.0, 5, base=0.004)  # × a base you chose
    AmplitudeSchedule.ramp(1e-3, 1e-2, 6)                      # absolute, log-spaced
    AmplitudeSchedule.explicit([1e-3, 3e-3, 1e-2])             # absolute, arbitrary

and then::

    for step in schedule.steps(catalog):
        sections = await crs.multisweep(catalog, amp=step.amplitudes, ...)

``steps`` keys its amplitudes by resonator **name**, which is what retires the
"the list must match ``res_info_dict.keys()`` order" coupling of the Periscope
dialog this replaces, and makes per-resonator amplitudes available on every step
rather than only on a single-sweep one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..core.resonators import ResonatorCatalog

__all__ = [
    "AmplitudeSchedule",
    "AmplitudeStep",
    "RESULTS_SCHEMA_VERSION",
    "pack_results",
    "collect_amplitude_iterations_for",
    "find_iteration_matching_amplitude",
    "get_amplitudes_at_iteration",
]


# Spacings a ladder can be generated with. "explicit" and "none" are what the
# constructors that don't generate one report instead.
LADDER_SPACINGS = ("log", "linear")
_SPACING_LABELS = LADDER_SPACINGS + ("explicit", "none")

# How many offenders an error message names before it summarises the rest.
_MAX_NAMED = 4


def _named(names: Sequence[str]) -> str:
    """``R0001, R0002 (and 3 more)`` — bounded, so a 500-resonator array's
    error message stays readable."""
    shown = ", ".join(names[:_MAX_NAMED])
    extra = len(names) - _MAX_NAMED
    return f"{shown} (and {extra} more)" if extra > 0 else shown


@dataclass(frozen=True, slots=True)
class AmplitudeStep:
    """One amplitude step: what every sweep section is probed at, for one pass.

    ``amplitudes`` goes straight into ``crs.multisweep(amp=...)`` — it is keyed
    by the same names the sweep sections come back under.
    """

    step: int  # execution order, 0-based
    amplitudes: dict[str, float]  # normalized DAC units, by section name
    factor: float | None  # the rung, or None when the ladder was absolute

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "amplitudes": dict(self.amplitudes),
            "factor": self.factor,
        }

    def __repr__(self) -> str:
        values = list(self.amplitudes.values())
        span = (
            f"{values[0]:g}"
            if len(set(values)) == 1
            else f"{min(values):g}…{max(values):g}"
        )
        rung = "" if self.factor is None else f", ×{self.factor:g}"
        return (
            f"AmplitudeStep(step={self.step}, {len(values)} sweep sections "
            f"at {span}{rung})"
        )


def _build_ladder(
    start: float,
    stop: float,
    nsteps: int,
    spacing: str,
    *,
    what: str,
) -> tuple[float, ...]:
    """*nsteps* rungs from *start* to *stop*, log- or linear-spaced.

    ``nsteps=1`` is only accepted when the endpoints agree.  Silently keeping
    the start and discarding the stop is how the dialog this replaces ended up
    shipping a one-step "uniform sweep" that had quietly become something else.
    """
    if spacing not in LADDER_SPACINGS:
        raise ValueError(
            f"spacing={spacing!r}: must be one of {LADDER_SPACINGS}."
        )
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError(f"nsteps={nsteps}: a schedule needs at least one step.")
    if nsteps == 1 and start != stop:
        raise ValueError(
            f"nsteps=1 with {what} running {start:g} to {stop:g}: which of the "
            f"two did you mean? Pass nsteps>1, or say it directly with one step "
            f"— explicit([{start:g}]) for an absolute amplitude, or "
            f"AmplitudeSchedule() to stay where you are."
        )
    if spacing == "log" and (start <= 0 or stop <= 0):
        raise ValueError(
            f"spacing='log' needs positive endpoints, got {start:g} to "
            f"{stop:g}. Use spacing='linear' if a rung really must be zero or "
            f"negative — though as {what} neither is likely to be meaningful."
        )
    if nsteps == 1:
        return (float(start),)
    generate = np.geomspace if spacing == "log" else np.linspace
    return tuple(float(v) for v in generate(start, stop, nsteps))


@dataclass(frozen=True, slots=True)
class AmplitudeSchedule:
    """What amplitude each resonator is probed at, on each pass.

    Two fields carry the whole answer: a **base** amplitude per resonator, and
    the **steps** applied to it. The iterating forms are built through
    :meth:`multiplicative`, :meth:`ramp` and :meth:`explicit`; the two that do
    not iterate are just the constructor::

        AmplitudeSchedule()         # one pass, at each resonator's own amplitude
        AmplitudeSchedule(0.005)    # one pass, at 0.005 for everything
        AmplitudeSchedule({"R0001": 0.004, ...})   # one pass, per resonator

    which is why *base* is the first argument: that is the only field a caller
    sets by hand with any regularity.
    """

    # Stamped into to_dict output and required exactly by from_dict, so a file
    # from another version of this module fails loudly rather than being half
    # understood. Bump whenever the dict shape changes in a way from_dict
    # cannot absorb.
    SCHEMA_VERSION = 1

    base: float | Mapping[str, float] | Sequence[float] | None = None
    ladder: tuple[float, ...] = (1.0,)
    relative: bool = True
    # Provenance only: how the steps were generated, for describe() and
    # to_dict() to report. Nothing computes with it — the ladder itself is the
    # truth — so it is excluded from equality, and two schedules that measure
    # the same thing compare equal however they were spelled.
    spacing: str = field(default="none", compare=False)

    def __post_init__(self):
        ladder = tuple(float(v) for v in self.ladder)
        if not ladder:
            raise ValueError(
                "ladder is empty: a schedule with no steps measures nothing."
            )
        if not all(math.isfinite(v) for v in ladder):
            raise ValueError(f"ladder={list(ladder)}: every rung must be finite.")
        if self.spacing not in _SPACING_LABELS:
            raise ValueError(
                f"spacing={self.spacing!r}: must be one of {_SPACING_LABELS}."
            )

        if self.relative:
            bad = [v for v in ladder if v <= 0]
            if bad:
                raise ValueError(
                    f"ladder rungs {bad} are not positive. A relative ladder "
                    f"multiplies the base amplitude, so a rung of zero silences "
                    f"the tone and a negative one is not a scaling at all."
                )
        else:
            if self.base is not None:
                raise ValueError(
                    "An absolute ladder takes no base: its rungs *are* the "
                    "amplitudes, so there is nothing for a base to contribute. "
                    "Use multiplicative(..., base=...) for a ladder that multiplies a "
                    "base you chose."
                )
            # Absolute rungs are amplitudes, so they answer to the same domain
            # BiasPoint enforces. Relative ones cannot be checked until a base
            # is known — that happens in _amplitudes_per_step.
            bad = [v for v in ladder if not 0 < v <= 1]
            if bad:
                raise ValueError(
                    f"ladder rungs {bad} are outside (0, 1]: an absolute ladder "
                    f"is in normalized DAC units. (A negative value usually "
                    f"means dBm — convert with "
                    f"amplitude = 10**((dbm - dac_scale_dbm) / 20).)"
                )

        object.__setattr__(self, "ladder", ladder)

    # ─── constructors ────────────────────────────────────────────────────────
    #
    # Only the iterating forms need one. A schedule that does not iterate is
    # the plain constructor — AmplitudeSchedule(), AmplitudeSchedule(0.005),
    # AmplitudeSchedule({...}) — which is what `base` being the first field
    # buys.

    @classmethod
    def multiplicative(
        cls,
        start: float,
        stop: float,
        nsteps: int,
        *,
        spacing: str = "log",
        base: float | Mapping[str, float] | Sequence[float] | None = None,
    ) -> AmplitudeSchedule:
        """A ladder of factors, each multiplying the base amplitude.

        Every resonator keeps its own scale, so an array biased across a spread
        of amplitudes walks that spread up and down together::

            AmplitudeSchedule.multiplicative(0.5, 2.0, 5)                 # of the catalog's
            AmplitudeSchedule.multiplicative(0.5, 2.0, 5, base=0.004)     # of one number
            AmplitudeSchedule.multiplicative(0.5, 2.0, 5, base={...})     # of your own, per name

        Args:
            start: factor of the first step.
            stop: factor of the last step.
            nsteps: how many steps, inclusive of both ends.
            spacing: ``"log"`` (the default — equal ratios, so equal steps in
                dB) or ``"linear"``.
            base: ``None`` for each resonator's own ``bias.amplitude``, one
                number for all of them, or a ``{name: amplitude}`` mapping.
        """
        return cls(
            ladder=_build_ladder(start, stop, nsteps, spacing, what="factors"),
            relative=True,
            base=base,
            spacing=spacing,
        )

    @classmethod
    def ramp(
        cls,
        start: float,
        stop: float,
        nsteps: int,
        *,
        spacing: str = "log",
    ) -> AmplitudeSchedule:
        """A ladder of absolute amplitudes, the same for every resonator.

        Args:
            start: amplitude of the first step, normalized DAC units in (0, 1].
            stop: amplitude of the last step.
            nsteps: how many steps, inclusive of both ends.
            spacing: ``"log"`` (the default) or ``"linear"``.
        """
        return cls(
            ladder=_build_ladder(start, stop, nsteps, spacing, what="amplitudes"),
            relative=False,
            spacing=spacing,
        )

    @classmethod
    def explicit(cls, levels: Sequence[float]) -> AmplitudeSchedule:
        """Absolute amplitudes, exactly as given, in the order given.

        The escape hatch for a ladder no spacing rule produces.
        """
        return cls(ladder=tuple(levels), relative=False, spacing="explicit")

    # ─── the ladder, without needing a catalog ───────────────────────────────

    @property
    def nsteps(self) -> int:
        """How many amplitude steps. Not how many sweeps — one sweep is a
        whole multisweep measurement, and the driver's ``directions``
        multiplies this to get that count."""
        return len(self.ladder)

    def __len__(self) -> int:
        return len(self.ladder)

    def __repr__(self) -> str:
        kind = "relative" if self.relative else "absolute"
        if self.base is None:
            of = "the catalog's own" if self.relative else "—"
        elif isinstance(self.base, Mapping):
            of = f"a base of {len(self.base)} named amplitudes"
        elif isinstance(self.base, (list, tuple, np.ndarray)):
            of = f"a base of {len(self.base)} positional amplitudes"
        else:
            of = f"a base of {float(self.base):g}"
        rungs = (
            f"{self.ladder[0]:g}"
            if len(self.ladder) == 1
            else f"{self.ladder[0]:g}…{self.ladder[-1]:g}, {self.spacing}"
        )
        tail = f" of {of}" if self.relative else ""
        return (
            f"AmplitudeSchedule({self.nsteps} "
            f"step{'' if self.nsteps == 1 else 's'}, {kind} {rungs}{tail})"
        )

    # ─── resolution against what is being swept ──────────────────────────────

    def _resolve_targets(
        self, target: ResonatorCatalog | Sequence[str]
    ) -> tuple[list[str], dict[str, float] | None, bool]:
        """*target* → its sweep names, the base amplitudes it can supply, and
        whether a positional base is meaningful for it.

        A catalog brings its own amplitudes and its own ordering, so a
        positional base is refused there for exactly the reason
        ``multisweep`` refuses a positional ``amp``: the pairing would depend
        on catalog order, which is not something a caller should have to know.
        A bare list of names is the caller's own ordering, so there it is fine.
        """
        if isinstance(target, ResonatorCatalog):
            return (
                [r.name for r in target],  # channel order
                {r.name: float(r.bias.amplitude) for r in target},
                False,
            )

        if isinstance(target, str):
            raise TypeError(
                f"target={target!r}: pass a ResonatorCatalog, or a sequence of "
                f"sweep names — a bare string reads as a sequence of single "
                f"characters. Did you mean [{target!r}]?"
            )
        if isinstance(target, Mapping):
            raise TypeError(
                "target must be a ResonatorCatalog or a sequence of sweep "
                "names. A mapping of amplitudes is a *base* — pass it as "
                "base= to AmplitudeSchedule() or multiplicative()."
            )

        names = list(target)
        if not names:
            raise ValueError("target is empty: there is nothing to sweep.")
        if not all(isinstance(n, str) for n in names):
            raise TypeError(
                "section names must be strings — they are the keys the sweep "
                "sections come back under."
            )
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate sweep names {duplicates}: each sweep needs its own "
                f"key, or results would overwrite each other."
            )
        return names, None, True

    def _resolve_base(
        self,
        names: list[str],
        defaults: dict[str, float] | None,
        allow_sequence: bool,
    ) -> dict[str, float]:
        """The base amplitude of every sweep, keyed by name.

        Same three-way vocabulary as ``multisweep``'s ``amp``, deliberately: a
        mapping must name every sweep, because a half-applied amplitude is the
        kind of thing that is only noticed after the data is taken.
        """
        base = self.base

        if base is None:
            if defaults is None:
                raise ValueError(
                    "A base amplitude is required when scheduling by name: "
                    "there is no catalog to take one from. Pass base= as a "
                    "single number, one per name, or a {name: amplitude} "
                    "mapping — or use ramp()/explicit(), whose rungs are "
                    "absolute amplitudes and need no base."
                )
            return dict(defaults)

        if isinstance(base, Mapping):
            unknown = sorted(set(base) - set(names))
            if unknown:
                raise ValueError(
                    f"base names {unknown} are not being swept. The names in "
                    f"play are {names[:_MAX_NAMED]}"
                    f"{' …' if len(names) > _MAX_NAMED else ''}."
                )
            missing = sorted(set(names) - set(base))
            if missing:
                raise ValueError(
                    f"base is missing an amplitude for {missing}. Pass every "
                    f"name, or a single number for all of them."
                )
            return {n: float(base[n]) for n in names}

        if isinstance(base, (list, tuple, np.ndarray)):
            if not allow_sequence:
                raise TypeError(
                    "base cannot be a positional sequence alongside a catalog "
                    "— the pairing would depend on catalog ordering, which is "
                    "not something a caller should have to know. Pass a "
                    "{name: amplitude} mapping, a single number, or None for "
                    "the catalog's own."
                )
            values = [float(v) for v in base]
            if len(values) != len(names):
                raise ValueError(
                    f"base has {len(values)} amplitudes for {len(names)} "
                    f"sections. Pass one per section, in the same order, or a "
                    f"single number for all."
                )
            return dict(zip(names, values))

        return {n: float(base) for n in names}

    def _amplitudes_per_step(
        self, target: ResonatorCatalog | Sequence[str]
    ) -> tuple[list[str], list[dict[str, float]], list[float | None]]:
        """The whole resolution, in one place: names, per-step amplitudes, and
        the rung each step came from.  Shared by :meth:`steps`,
        :meth:`validate` and :meth:`describe` so the three cannot disagree."""
        names, defaults, allow_sequence = self._resolve_targets(target)

        if not self.relative:
            # The rungs are the amplitudes; every sweep gets the same one.
            return (
                names,
                [{n: level for n in names} for level in self.ladder],
                [None] * len(self.ladder),
            )

        base = self._resolve_base(names, defaults, allow_sequence)
        return (
            names,
            [{n: base[n] * factor for n in names} for factor in self.ladder],
            list(self.ladder),
        )

    # ─── the steps ───────────────────────────────────────────────────────────

    def steps(
        self, target: ResonatorCatalog | Sequence[str]
    ) -> list[AmplitudeStep]:
        """The numbered amplitude steps, resolved against what is being swept.

        Args:
            target: a :class:`~rfmux.core.resonators.ResonatorCatalog`, or the
                names of the sweep sections when there is no catalog — for a bare
                ``center_frequencies`` sweep, the same names ``multisweep``
                will key its results by (``S0001…`` by default).

        Returns:
            list[AmplitudeStep]: one per amplitude, in measurement order.

        Raises:
            ValueError: if any resolved amplitude falls outside (0, 1], or if a
                base mapping does not name every sweep. Both are caught here,
                before the first sweep runs, rather than after some of the data
                has been taken — ``multisweep`` itself only rejects
                non-positive amplitudes, so a ladder that overshoots full scale
                would otherwise reach the hardware unchallenged.
        """
        names, per_step, factors = self._amplitudes_per_step(target)

        errors = [m for severity, m in self._range_issues(per_step) if severity == "error"]
        if errors:
            raise ValueError(" ".join(errors))

        return [
            AmplitudeStep(step=i, amplitudes=amplitudes, factor=factor)
            for i, (amplitudes, factor) in enumerate(zip(per_step, factors))
        ]

    def _range_issues(
        self, per_step: list[dict[str, float]]
    ) -> list[tuple[str, str]]:
        """Amplitudes that fall outside the (0, 1] BiasPoint enforces.

        Reported per step and per name, so the answer is "R0007 overshoots at
        step 5" rather than a failure twenty minutes into the run.
        """
        issues: list[tuple[str, str]] = []
        for i, amplitudes in enumerate(per_step):
            over = sorted(n for n, a in amplitudes.items() if a > 1)
            if over:
                worst = max(amplitudes.values())
                issues.append((
                    "error",
                    f"Step {i} puts {_named(over)} above full scale "
                    f"(largest {worst:g}, and amplitude is normalized DAC "
                    f"units in (0, 1]).",
                ))
            under = sorted(n for n, a in amplitudes.items() if a <= 0)
            if under:
                issues.append((
                    "error",
                    f"Step {i} puts {_named(under)} at or below zero "
                    f"amplitude, which is not a measurement.",
                ))
            unusable = sorted(n for n, a in amplitudes.items() if not math.isfinite(a))
            if unusable:
                issues.append((
                    "error",
                    f"Step {i} gives {_named(unusable)} a non-finite amplitude.",
                ))
        return issues

    # ─── display and checking, the PulseCaptureConfig idiom ──────────────────

    def describe(
        self,
        target: ResonatorCatalog | Sequence[str],
        n_directions: int = 1,
        dac_scale_dbm: float | None = None,
    ) -> dict:
        """Derived quantities for display, resolved against *target*.

        What a dialog or a notebook renders instead of deriving its own. Raises
        the same things :meth:`steps` does — call :meth:`validate` first if the
        input might not be sound.
        """
        names, per_step, factors = self._amplitudes_per_step(target)
        flat = [a for amplitudes in per_step for a in amplitudes.values()]

        described = {
            "nsteps": self.nsteps,
            "relative": self.relative,
            "spacing": self.spacing,
            "ladder": list(self.ladder),
            "n_sections": len(names),
            "n_directions": n_directions,
            # The number that actually predicts how long this takes.
            "n_sweeps": self.nsteps * n_directions,
            "amplitude_min": min(flat),
            "amplitude_max": max(flat),
            "amplitude_range_by_name": {
                n: (
                    min(amplitudes[n] for amplitudes in per_step),
                    max(amplitudes[n] for amplitudes in per_step),
                )
                for n in names
            },
        }
        if dac_scale_dbm is not None:
            described["power_dbm_min"] = dac_scale_dbm + 20.0 * math.log10(min(flat))
            described["power_dbm_max"] = dac_scale_dbm + 20.0 * math.log10(max(flat))
        return described

    def validate(
        self,
        target: ResonatorCatalog | Sequence[str],
        n_directions: int = 1,
    ) -> list[tuple[str, str]]:
        """``[(severity, message), ...]`` — severities error/warning/info.

        Never raises: a caller that is rendering a live preview of a
        half-entered form wants the complaint as text, not as a traceback. The
        errors here are the ones :meth:`steps` raises on.
        """
        try:
            names, per_step, factors = self._amplitudes_per_step(target)
        except (ValueError, TypeError) as exc:
            return [("error", str(exc))]

        issues = self._range_issues(per_step)

        repeated = sorted({v for v in self.ladder if self.ladder.count(v) > 1})
        if repeated:
            issues.append((
                "warning",
                f"Ladder repeats {', '.join(f'{v:g}' for v in repeated)}: those "
                f"steps measure the same thing twice.",
            ))

        issues.append((
            "info",
            f"{self.nsteps} amplitude step{'' if self.nsteps == 1 else 's'} × "
            f"{n_directions} direction{'' if n_directions == 1 else 's'} = "
            f"{self.nsteps * n_directions} sweeps of {len(names)} "
            f"section{'' if len(names) == 1 else 's'}.",
        ))
        return issues

    # ─── persistence ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Plain builtins, for the provenance block of a driver's output."""
        if isinstance(self.base, Mapping):
            base = {str(k): float(v) for k, v in self.base.items()}
        elif isinstance(self.base, (list, tuple, np.ndarray)):
            base = [float(v) for v in self.base]
        elif self.base is None:
            base = None
        else:
            base = float(self.base)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "ladder": list(self.ladder),
            "relative": self.relative,
            "base": base,
            "spacing": self.spacing,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> AmplitudeSchedule:
        version = d.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={version!r}, expected {cls.SCHEMA_VERSION}: "
                f"this dict was written by a different version of "
                f"AmplitudeSchedule."
            )
        return cls(
            ladder=tuple(d["ladder"]),
            relative=bool(d["relative"]),
            base=d.get("base"),
            spacing=d.get("spacing", "none"),
        )


# ─── the driver's output: one shape, written and read in one place ───────────
#
# multiamp_multisweep produces the dict below; the readers under it are the
# supported way to get things back out. Both live here rather than beside the
# driver because the schedule's own to_dict() is *inside* that dict — a reader
# resolving ladder[iteration] has to agree with the schedule about what a rung
# means, and two files agreeing about one contract is one file too many.

# Bumped when the packed dict changes shape in a way a reader cannot absorb.
RESULTS_SCHEMA_VERSION = 1


def pack_results(
    sweeps: Mapping[int, Mapping[str, dict]],
    *,
    module: int,
    amp_schedule: AmplitudeSchedule,
    directions: Sequence[str],
    span_hz: float,
    npoints_per_sweep: int,
    nsamps: int,
    bias_frequency_method: str | None,
    rotate_saved_data: bool,
    apply_df_calibration: bool,
    catalog=None,
    center_frequencies: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
    requested_module: int | None = None,
) -> dict:
    """Assemble what ``multiamp_multisweep`` returns.

    Args:
        sweeps: ``{iteration: {direction: one multisweep return}}``, in the
            order measured.
        module: the module actually swept — resolved, never None.
        requested_module: the ``module`` argument as the caller passed it, which
            is None whenever it came from the catalog instead. Recorded as-is,
            because *call_params* says what was asked for and not what was
            worked out from it.
        catalog: the ``ResonatorCatalog`` swept, or None in frequency-list mode.
            Snapshotted with ``to_dict`` for provenance.

    Returns:
        dict: ``schema_version``, ``module``, ``call_params`` and ``results``.

        ``results`` is keyed by amplitude iteration, numbered from 0 in the
        order measured, and an iteration holds one entry per direction swept
        and nothing else.

        Nothing is duplicated into the iteration level. What a resonator was
        probed at is already ``sweep_amplitude`` in its own entry — see
        :func:`get_amplitudes_at_iteration` — and the rung that produced it is
        ``call_params["amp_schedule"]["ladder"][iteration]``. Sweep centres are
        recorded only as passed: a later step may re-centre between amplitudes,
        at which point a top-level copy would be a lie while each sweep's own
        ``original_center_frequency`` cannot be.
    """
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "module": module,
        "call_params": {
            "catalog": catalog.to_dict() if catalog is not None else None,
            "center_frequencies": (
                [float(f) for f in center_frequencies]
                if center_frequencies is not None
                else None
            ),
            "names": list(names) if names is not None else None,
            "amp_schedule": amp_schedule.to_dict(),
            "directions": list(directions),
            "span_hz": float(span_hz),
            "npoints_per_sweep": int(npoints_per_sweep),
            "nsamps": int(nsamps),
            "bias_frequency_method": bias_frequency_method,
            "rotate_saved_data": bool(rotate_saved_data),
            "apply_df_calibration": bool(apply_df_calibration),
            "module": requested_module,
        },
        "results": {int(i): dict(by_direction) for i, by_direction in sweeps.items()},
    }


def _iterations(results: Mapping) -> dict:
    """The ``results`` block, with a useful error when handed the wrong dict."""
    try:
        return results["results"]
    except (TypeError, KeyError):
        raise TypeError(
            "Expected the dict multiamp_multisweep returned (with 'results' "
            "and 'call_params'), not one of its parts."
        ) from None


def _section_names(results: Mapping) -> list[str]:
    """Every section name that appears in the first sweep, in its order."""
    for by_direction in _iterations(results).values():
        for sections in by_direction.values():
            return list(sections)
    return []


def collect_amplitude_iterations_for(results: Mapping, name: str) -> dict:
    """Every sweep of one resonator, across the amplitude iterations.

    Args:
        results: what ``multiamp_multisweep`` returned.
        name: the resonator or section to pull out.

    Returns:
        dict: ``{iteration: {direction: sweep}}`` — the same shape as
        ``results["results"]``, one resonator deep, in the order measured.
        Measured order, not sorted by amplitude: an ``explicit`` ladder may run
        in any order, and re-sorting silently would lose the order things
        actually happened in.

    Raises:
        KeyError: if *name* was not swept.
    """
    collected = {}
    for iteration, by_direction in _iterations(results).items():
        entries = {
            direction: sections[name]
            for direction, sections in by_direction.items()
            if name in sections
        }
        if entries:
            collected[iteration] = entries

    if not collected:
        available = _section_names(results)
        raise KeyError(
            f"{name!r} was not swept. The section names in play are "
            f"{_named(available)}."
        )
    return collected


def get_amplitudes_at_iteration(results: Mapping, iteration: int) -> dict:
    """What every sweep was probed at on one iteration.

    Reads each sweep's own ``sweep_amplitude`` rather than a stored copy, which
    is why the packed dict does not carry one.

    Args:
        results: what ``multiamp_multisweep`` returned.
        iteration: which amplitude iteration.

    Returns:
        dict: ``{name: amplitude}`` in normalized DAC units.

    Raises:
        KeyError: if there is no such iteration.
    """
    iterations = _iterations(results)
    if iteration not in iterations:
        raise KeyError(
            f"No iteration {iteration}. This result has "
            f"{sorted(iterations)}."
        )

    # Every direction of one iteration was swept at the same amplitudes, so the
    # first one answers the question.
    for sections in iterations[iteration].values():
        return {name: float(s["sweep_amplitude"]) for name, s in sections.items()}
    return {}


def find_iteration_matching_amplitude(
    results: Mapping, name: str, amplitude: float | None = None
) -> int:
    """Which amplitude iteration probed *name* closest to *amplitude*.

    Args:
        results: what ``multiamp_multisweep`` returned.
        name: whose amplitudes to match against. Required, because a relative
            ladder gives every resonator its own: R0001 walking 1→2→4 µ and
            R0002 walking 3→6→12 µ share an iteration number and nothing else,
            so "the iteration at 4 µ" is only a question about one of them.
        amplitude: the amplitude to match, in normalized DAC units. Defaults to
            *name*'s own bias amplitude, read from the catalog snapshot in
            ``call_params`` — which is the usual question, "which iteration was
            taken where this resonator is actually biased?"

    Returns:
        int: the iteration number, for indexing ``results["results"]``.

    Nearest wins, and there is always a nearest — floats from a ladder rarely
    compare equal, so matching on equality would find nothing. A caller who
    needs the match to be close should check it:
    ``get_amplitudes_at_iteration(results, i)[name]``.

    Raises:
        KeyError: if *name* was not swept.
        ValueError: if *amplitude* is None and there is no catalog to take a
            bias amplitude from, or if nothing was measured.
    """
    if amplitude is None:
        amplitude = _bias_amplitude_of(results, name)

    per_iteration = {
        iteration: float(next(iter(entries.values()))["sweep_amplitude"])
        for iteration, entries in collect_amplitude_iterations_for(
            results, name
        ).items()
    }
    if not per_iteration:
        raise ValueError(f"No sweep sections for {name!r} to match against.")

    return min(per_iteration, key=lambda i: abs(per_iteration[i] - amplitude))


def _bias_amplitude_of(results: Mapping, name: str) -> float:
    """*name*'s bias amplitude, from the catalog snapshot in call_params."""
    catalog = results.get("call_params", {}).get("catalog")
    if catalog is None:
        raise ValueError(
            "No amplitude given and no catalog to take one from — this result "
            "came from a bare center_frequencies sweep, which has no bias "
            "amplitude. Pass amplitude= explicitly."
        )

    for r in catalog["resonators"]:
        if r["name"] == name:
            return float(r["bias"]["amplitude"])

    raise KeyError(
        f"{name!r} is not in the catalog this result was swept from. Its "
        f"resonators are {_named([r['name'] for r in catalog['resonators']])}."
    )
