"""Define the probe amplitude for each resonator at each multisweep step.

Pass an :class:`AmplitudeSchedule` as ``crs.multisweep(catalog, amp=schedule)``.
The plain constructor gives one step; ``multiplicative``, ``ramp``, and
``explicit`` build sequences. Amplitudes are fractions of DAC full scale.

Steps are numbered from zero. Each step can be swept in one or both frequency
directions; direction is chosen by the driver, independently of the schedule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..core.resonators import ResonatorCatalog
from ..core.transferfunctions import convert_dacunits_to_dbm

__all__ = [
    "AmplitudeSchedule",
    "AmplitudeStep",
]


# Spacings steps can be generated with. "explicit" and "none" are what the
# constructors that don't generate one report instead.
STEP_SPACINGS = ("log", "linear")
_SPACING_LABELS = STEP_SPACINGS + ("explicit", "none")

# How many offenders an error message names before it summarises the rest.
_MAX_NAMED = 4


def _named(names: Sequence[str]) -> str:
    """``BOTA, KOZR (and 3 more)`` — bounded, so a 500-resonator array's
    error message stays readable."""
    shown = ", ".join(names[:_MAX_NAMED])
    extra = len(names) - _MAX_NAMED
    return f"{shown} (and {extra} more)" if extra > 0 else shown


def resolve_amplitudes(
    names: Sequence[str],
    value: float | Sequence[float] | Mapping[str, float] | None,
    *,
    defaults: Mapping[str, float] | None,
    allow_sequence: bool,
    what: str,
    of: str = "sections",
) -> dict[str, float]:
    """Return one amplitude per name from a scalar, mapping, sequence, or None.

    None uses ``defaults`` and raises if no defaults are available. A scalar
    applies to every name. A mapping must contain exactly the supplied names.
    Sequences must match the number of names and require ``allow_sequence``;
    catalog callers disable them to avoid depending on catalog order.

    ``what`` names the argument in error messages; ``of`` describes the items.
    """
    names = list(names)

    if value is None:
        if defaults is None:
            raise ValueError(
                f"{what} is required when sweeping center_frequencies: pass a "
                f"single amplitude for all of them, one per frequency, or a "
                f"{{name: amplitude}} mapping. (A ResonatorCatalog carries an "
                f"amplitude per resonator, so there {what} is optional.)"
            )
        return {n: float(defaults[n]) for n in names}

    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise ValueError(
                f"{what} names {unknown} are not being swept. The names in play "
                f"are {names[:_MAX_NAMED]}"
                f"{' …' if len(names) > _MAX_NAMED else ''}."
            )
        missing = sorted(set(names) - set(value))
        if missing:
            raise ValueError(
                f"{what} is missing an amplitude for {missing}. Pass every "
                f"name, or a single number for all of them."
            )
        return {n: float(value[n]) for n in names}

    if isinstance(value, (list, tuple, np.ndarray)):
        if not allow_sequence:
            raise TypeError(
                f"{what} cannot be a positional sequence alongside a catalog — "
                f"a catalog is an unordered collection of resonators, so "
                f"pairing to it by position means knowing which order it was "
                f"pulled out in. Pass a {{name: amplitude}} mapping, a single "
                f"number, or None. (A positional list *is* accepted alongside "
                f"center_frequencies, where the ordering is your own.)"
            )
        values = [float(v) for v in value]
        if len(values) != len(names):
            raise ValueError(
                f"{what} has {len(values)} amplitudes for {len(names)} {of}. "
                f"Pass one each, in the same order, or a single number for all."
            )
        return dict(zip(names, values))

    return {n: float(value) for n in names}


@dataclass(frozen=True, slots=True)
class AmplitudeStep:
    """Amplitudes for one multisweep step, keyed by sweep-section name."""

    step: int  # execution order, 0-based
    amplitudes: dict[str, float]  # normalized DAC units, by section name
    factor: float | None  # the multiplier, or None for absolute amplitudes

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "amplitudes": dict(self.amplitudes),
            "factor": self.factor,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> AmplitudeStep:
        # Unversioned, like the to_dict above it: a step carries
        # provenance inside a schedule's output, not a file of its own.
        return cls(
            step=int(d["step"]),
            amplitudes={str(k): float(v) for k, v in d["amplitudes"].items()},
            factor=None if d.get("factor") is None else float(d["factor"]),
        )

    def __repr__(self) -> str:
        values = list(self.amplitudes.values())
        span = (
            f"{values[0]:g}"
            if len(set(values)) == 1
            else f"{min(values):g}…{max(values):g}"
        )
        factor = "" if self.factor is None else f", ×{self.factor:g}"
        return (
            f"AmplitudeStep(step={self.step}, {len(values)} sweep sections "
            f"at {span}{factor})"
        )


def _build_steps(
    start: float,
    stop: float,
    nsteps: int,
    spacing: str,
    *,
    what: str,
) -> tuple[float, ...]:
    """Generate logarithmic or linear steps, including both endpoints.

    A single step is allowed only when the endpoints are equal.
    """
    if spacing not in STEP_SPACINGS:
        raise ValueError(
            f"spacing={spacing!r}: must be one of {STEP_SPACINGS}."
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
            f"{stop:g}. Use spacing='linear' if a step really must be zero or "
            f"negative — though as {what} neither is likely to be meaningful."
        )
    if nsteps == 1:
        return (float(start),)
    generate = np.geomspace if spacing == "log" else np.linspace
    return tuple(float(v) for v in generate(start, stop, nsteps))


@dataclass(frozen=True, slots=True)
class AmplitudeSchedule:
    """Set the probe amplitude for each resonator at each sweep step.

    Relative schedules multiply each resonator's ``base`` by each value in
    ``steps``. Absolute schedules use the step values directly as amplitudes.
    All amplitudes are fractions of DAC full scale.

    Use the constructor for one step::

        AmplitudeSchedule()       # use each resonator's bias amplitude
        AmplitudeSchedule(0.005)  # use 0.005 for every resonator
        AmplitudeSchedule({"BOTA": 0.004, "KOZR": 0.006})

    Use :meth:`multiplicative`, :meth:`ramp`, or :meth:`explicit` for a sequence.
    """

    # Version 2 writes steps; version 1 remains readable for saved sweeps.
    SCHEMA_VERSION = 2

    base: float | Mapping[str, float] | Sequence[float] | None = None
    steps: tuple[float, ...] = (1.0,)
    relative: bool = True
    # Record how steps were generated for display and saving.
    # Spacing does not affect calculations or schedule equality.
    spacing: str = field(default="none", compare=False)

    def __post_init__(self):
        steps = tuple(float(v) for v in self.steps)
        if not steps:
            raise ValueError(
                "steps tuple is empty: a schedule with no steps measures nothing."
            )
        if not all(math.isfinite(v) for v in steps):
            raise ValueError(f"steps={list(steps)}: every step must be finite.")
        if self.spacing not in _SPACING_LABELS:
            raise ValueError(
                f"spacing={self.spacing!r}: must be one of {_SPACING_LABELS}."
            )

        if self.relative:
            bad = [v for v in steps if v <= 0]
            if bad:
                raise ValueError(
                    f"steps {bad} are not positive. A relative schedule "
                    f"multiplies the base amplitude, so a step of zero silences "
                    f"the tone and a negative one is not a scaling at all."
                )
        else:
            if self.base is not None:
                raise ValueError(
                    "An absolute schedule takes no base: its steps *are* the "
                    "amplitudes, so there is nothing for a base to contribute. "
                    "Use multiplicative(..., base=...) for a schedule that "
                    "multiplies a base you chose."
                )
            # Check absolute amplitudes now; relative amplitudes need a resolved base.
            bad = [v for v in steps if not 0 < v <= 1]
            if bad:
                raise ValueError(
                    f"steps {bad} are outside (0, 1]: an absolute schedule "
                    f"is in normalized DAC units. (A negative value usually "
                    f"means dBm — convert with "
                    f"amplitude = 10**((dbm - dac_scale_dbm) / 20).)"
                )

        object.__setattr__(self, "steps", steps)

    # ─── constructors ────────────────────────────────────────────────────────
    #
    # Use the plain constructor for one step and these methods for sequences.

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
        """A sequence of factors, each multiplying the base amplitude.

        Apply the same factors to each resonator's base amplitude::

            # Scale each resonator's catalog amplitude from 0.5x to 2x.
            AmplitudeSchedule.multiplicative(0.5, 2.0, 5)
            # Use the same base amplitude for every resonator.
            AmplitudeSchedule.multiplicative(0.5, 2.0, 5, base=0.004)

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
            steps=_build_steps(start, stop, nsteps, spacing, what="factors"),
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
        """A sequence of absolute amplitudes, the same for every resonator.

        Args:
            start: amplitude of the first step, normalized DAC units in (0, 1].
            stop: amplitude of the last step.
            nsteps: how many steps, inclusive of both ends.
            spacing: ``"log"`` (the default) or ``"linear"``.
        """
        return cls(
            steps=_build_steps(start, stop, nsteps, spacing, what="amplitudes"),
            relative=False,
            spacing=spacing,
        )

    @classmethod
    def explicit(cls, levels: Sequence[float]) -> AmplitudeSchedule:
        """Use the supplied absolute amplitudes in the supplied order."""
        return cls(steps=tuple(levels), relative=False, spacing="explicit")

    # ─── the steps, without needing a catalog ───────────────────────────────

    @property
    def nsteps(self) -> int:
        """Return the number of amplitude steps.

        Each step can run in one or both frequency directions.
        """
        return len(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

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
        steps = (
            f"{self.steps[0]:g}"
            if len(self.steps) == 1
            else f"{self.steps[0]:g}…{self.steps[-1]:g}, {self.spacing}"
        )
        tail = f" of {of}" if self.relative else ""
        return (
            f"AmplitudeSchedule({self.nsteps} "
            f"step{'' if self.nsteps == 1 else 's'}, {kind} {steps}{tail})"
        )

    # ─── resolution against what is being swept ──────────────────────────────

    def _resolve_targets(
        self, target: ResonatorCatalog | Sequence[str]
    ) -> tuple[list[str], dict[str, float] | None, bool]:
        """Return sweep names, default amplitudes, and whether sequences are allowed.

        Catalogs supply bias amplitudes and require overrides by name or scalar.
        A sequence of names defines its own order and allows positional overrides.
        """
        if isinstance(target, ResonatorCatalog):
            resonators = target.resonators(order="frequency")
            return (
                [r.name for r in resonators],
                {r.name: float(r.bias.amplitude) for r in resonators},
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
        """Resolve a base amplitude for each name using :func:`resolve_amplitudes`."""
        if self.base is None and defaults is None:
            raise ValueError(
                "A base amplitude is required when scheduling by name: "
                "there is no catalog to take one from. Pass base= as a "
                "single number, one per name, or a {name: amplitude} "
                "mapping — or use ramp()/explicit(), whose steps are "
                "absolute amplitudes and need no base."
            )
        return resolve_amplitudes(
            names,
            self.base,
            defaults=defaults,
            allow_sequence=allow_sequence,
            what="base",
        )

    def _amplitudes_per_step(
        self, target: ResonatorCatalog | Sequence[str]
    ) -> tuple[list[str], list[dict[str, float]], list[float | None]]:
        """The whole resolution, in one place: names, per-step amplitudes, and
        the multiplier for each step.  Shared by :meth:`resolve_steps`,
        :meth:`validate` and :meth:`describe` so the three cannot disagree."""
        names, defaults, allow_sequence = self._resolve_targets(target)

        if not self.relative:
            # The steps are the amplitudes; every sweep gets the same one.
            return (
                names,
                [{n: level for n in names} for level in self.steps],
                [None] * len(self.steps),
            )

        base = self._resolve_base(names, defaults, allow_sequence)
        return (
            names,
            [{n: base[n] * factor for n in names} for factor in self.steps],
            list(self.steps),
        )

    # ─── the steps ───────────────────────────────────────────────────────────

    def resolve_steps(
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
                base mapping does not name every sweep. All steps are checked
                before measurement starts.
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
        """Report non-finite amplitudes and values outside (0, 1], by step and name."""
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
        """Return step counts and amplitude ranges for display.

        Include power ranges when ``dac_scale_dbm`` is supplied. This resolves
        amplitudes but does not check their range; use :meth:`validate` to check
        settings before displaying a preview.
        """
        names, per_step, factors = self._amplitudes_per_step(target)
        flat = [a for amplitudes in per_step for a in amplitudes.values()]

        described = {
            "nsteps": self.nsteps,
            "relative": self.relative,
            "spacing": self.spacing,
            "steps": list(self.steps),
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
            described["power_dbm_min"] = float(
                convert_dacunits_to_dbm(min(flat), dac_scale_dbm))
            described["power_dbm_max"] = float(
                convert_dacunits_to_dbm(max(flat), dac_scale_dbm))
        return described

    def validate(
        self,
        target: ResonatorCatalog | Sequence[str],
        n_directions: int = 1,
    ) -> list[tuple[str, str]]:
        """Return ``(severity, message)`` pairs for errors, warnings, and step counts.

        Convert ValueError and TypeError from amplitude resolution into error
        messages. Also report invalid amplitudes and repeated steps.
        """
        try:
            names, per_step, factors = self._amplitudes_per_step(target)
        except (ValueError, TypeError) as exc:
            return [("error", str(exc))]

        issues = self._range_issues(per_step)

        repeated = sorted({v for v in self.steps if self.steps.count(v) > 1})
        if repeated:
            issues.append((
                "warning",
                f"Schedule repeats {', '.join(f'{v:g}' for v in repeated)}: those "
                f"steps measure the same thing twice.",
            ))

        issues.append((
            "info",
            f"{self.nsteps} amplitude step{'' if self.nsteps == 1 else 's'} × "
            f"{n_directions} direction{'' if n_directions == 1 else 's'} = "
            f"{self.nsteps * n_directions} "
            f"sweep{'' if self.nsteps * n_directions == 1 else 's'} of "
            f"{len(names)} section{'' if len(names) == 1 else 's'}.",
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
            "steps": list(self.steps),
            "relative": self.relative,
            "base": base,
            "spacing": self.spacing,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> AmplitudeSchedule:
        version = d.get("schema_version")
        if version not in (1, cls.SCHEMA_VERSION):
            raise ValueError(
                f"schema_version={version!r}, expected {cls.SCHEMA_VERSION}: "
                f"this dict was written by a different version of "
                f"AmplitudeSchedule."
            )
        return cls(
            steps=tuple(d["ladder"] if version == 1 else d["steps"]),
            relative=bool(d["relative"]),
            base=d.get("base"),
            spacing=d.get("spacing", "none"),
        )
