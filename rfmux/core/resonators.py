"""Named resonators and their bias points, grouped by module.

A :class:`ResonatorCatalog` holds :class:`Resonator` objects, each with a name,
hardware channel, and :class:`BiasPoint`. Bias points hold the tone frequency,
amplitude, and optional calibration. Use ``Resonator.update_bias_point`` to
retune and clear calibration fields that depend on the tone.
"""

from __future__ import annotations

import copy as _copy
import csv
import io
from dataclasses import dataclass, field, fields, replace

from collections.abc import Mapping
from typing import Callable, Iterable, Iterator, Literal, Sequence

from ..resonator_names import syllabic_names
from .transferfunctions import BASE_FREQUENCY, convert_dacunits_to_dbm


def on_grid(frequency_hz: float) -> float:
    """Round onto the hardware tone grid, ``transferfunctions.BASE_FREQUENCY``.

    The single definition every quantizing path in the tree uses. Public
    because the grid binds more than bias points: the NCO an operation parks
    its tones against has to land on it too, or every offset computed from
    that NCO is off-grid however carefully the tone was quantized.
    """
    return round(frequency_hz / BASE_FREQUENCY) * BASE_FREQUENCY


# ─── BiasPoint ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BiasPoint:
    """A bias frequency and amplitude with optional calibration at that tone.

    Frequency is snapped to the hardware tone grid unless
    ``bias_frequency_quantized=False``. Amplitude is a fraction of DAC full
    scale in (0, 1]. The object is frozen; retune with
    ``Resonator.update_bias_point`` to clear calibration fields automatically.

    ``bias_sweep`` holds the trace used for calibration: frequency in Hz,
    complex IQ in volts, sweep centre, amplitude, and direction. Its keys are
    listed in ``BIAS_SWEEP_KEYS``.
    """

    frequency_hz: float
    amplitude: float  # normalized DAC units, (0, 1]
    # Disable only when an exact, unquantized frequency is needed.
    bias_frequency_quantized: bool = True
    dI_df: float | None = None  # V/Hz at this bias point
    dQ_df: float | None = None
    iq_rotation_deg: float | None = None
    # First bifurcated amplitude in the last run that observed bifurcation.
    bifurcated_at: float | None = None
    bias_sweep: dict | None = None  # the trace dI_df/dQ_df were read off

    # Fields that describe *this* tone and are therefore invalidated by moving
    # it. Consumed by Resonator.update_bias_point.
    _CAL_FIELDS = (
        "dI_df",
        "dQ_df",
        "iq_rotation_deg",
        "bifurcated_at",
        "bias_sweep",
    )

    # Keep the calibration trace and its measurement context. Counts can be
    # recovered from volts only while VOLTS_PER_ROC is a shared constant.
    BIAS_SWEEP_KEYS = (
        "frequencies",
        "iq_volts",
        "original_center_frequency",
        "sweep_amplitude",
        "sweep_direction",
    )

    # The head of that tuple: the two arrays a calibration is computed from,
    # and the only part a stored sweep has to have. The scalars after them are
    # provenance — a missing one costs a reader context rather than an answer.
    _SWEEP_TRACES = BIAS_SWEEP_KEYS[:2]

    def __post_init__(self):
        if self.amplitude <= 0:
            raise ValueError(
                f"amplitude={self.amplitude}: must be normalized DAC units in (0, 1]. "
                f"A negative value usually means dBm — convert with "
                f"amplitude = 10**((dbm - dac_scale_dbm) / 20)."
            )
        if self.amplitude > 1:
            raise ValueError(
                f"amplitude={self.amplitude}: must be normalized DAC units in (0, 1]."
            )
        if self.frequency_hz <= 0:
            raise ValueError(f"frequency_hz={self.frequency_hz}: must be positive Hz.")
        if self.bias_frequency_quantized:
            snapped = on_grid(self.frequency_hz)
            if snapped <= 0:
                raise ValueError(
                    f"frequency_hz={self.frequency_hz}: quantizes to 0 Hz — it is "
                    f"less than half a tone-grid step ({BASE_FREQUENCY / 2:g} Hz), "
                    f"so the hardware has nowhere to put it."
                )
            object.__setattr__(self, "frequency_hz", snapped)
        if self.bias_sweep is not None:
            self._check_sweep(self.bias_sweep)

    @classmethod
    def _check_sweep(cls, sweep):
        """Reject a ``bias_sweep`` that cannot be read as a trace.

        Shallow on purpose. It is checked at all because a bias point
        validated at construction is meant to stay valid, and a sweep that
        arrives with mismatched arrays would otherwise surface much later, in
        whichever reader interpolated it.
        """
        if not isinstance(sweep, Mapping):
            raise ValueError(
                f"bias_sweep is a {type(sweep).__name__}: expected a dict of "
                f"the keys multisweep puts on a sweep entry."
            )
        missing = [k for k in cls._SWEEP_TRACES if sweep.get(k) is None]
        if missing:
            raise ValueError(
                f"bias_sweep is missing {', '.join(missing)}. A stored sweep "
                f"exists to have a calibration read off it, which needs "
                f"{' and '.join(cls._SWEEP_TRACES)}; its keys are "
                f"{sorted(sweep)}."
            )
        lengths = {k: len(sweep[k]) for k in cls._SWEEP_TRACES}
        if len(set(lengths.values())) > 1:
            described = ", ".join(f"{n} {k}" for k, n in lengths.items())
            raise ValueError(
                f"bias_sweep has {described} — they describe different "
                f"measurements."
            )

    @property
    def df_calibration(self) -> complex | None:
        """1/(dI_df + j·dQ_df) in Hz/V. Derived — can never go stale."""
        if self.dI_df is None or self.dQ_df is None:
            return None
        d = complex(self.dI_df, self.dQ_df)
        return 1.0 / d if abs(d) > 0 else None

    def power_dbm(self, dac_scale_dbm: float) -> float:
        """What this tone is driven at, against the module's DAC full scale."""
        return float(convert_dacunits_to_dbm(self.amplitude, dac_scale_dbm))

    def quantize(self) -> BiasPoint:
        """Round the frequency onto the hardware tone grid.

        Rarely needed by hand — a bias point quantizes itself at construction
        unless it was built with ``bias_frequency_quantized=False``. This is the
        one-shot for those, and a no-op for everything else. Calibration is
        kept: the shift is under half a grid step, which is small compared to a
        resonator's width. ``bias_frequency_quantized`` is policy and is left
        alone, so an opted-out point stays opted out for its next move.
        """
        return replace(self, frequency_hz=on_grid(self.frequency_hz))


def _bias_dict(bias: BiasPoint) -> dict:
    """One bias point as a dict, for :meth:`ResonatorCatalog.to_dict`.

    Field by field rather than ``dataclasses.asdict``, which deep-copies as it
    recurses. Over floats that cost nothing and nobody noticed; over a
    ``bias_sweep`` it would duplicate both arrays on every call, including the
    ``to_dict`` every multisweep does to snapshot its catalog. The sweep is
    copied one level, so the record is its own dict but shares the traces —
    they are measurement data and nothing mutates them.
    """
    d = {f.name: getattr(bias, f.name) for f in fields(bias)}
    if d.get("bias_sweep") is not None:
        d["bias_sweep"] = dict(d["bias_sweep"])
    return d


# ─── Resonator ────────────────────────────────────────────────────────────────


@dataclass(slots=True, eq=False)
class Resonator:
    """A named resonator with a hardware channel and a required bias point.

    Measurement and analysis code pass this object between tuning steps.
    ``rfmux.core.schema.HWMResonator`` is the separate hardware-map ORM type.
    """

    name: str
    channel: int  # 1-based hardware channel; permanent binding
    bias: BiasPoint
    notes: dict = field(default_factory=dict)  # explicitly the junk drawer

    def update_bias_point(self, **changes) -> BiasPoint:
        """Replace the bias point with the supplied changes and return it.

        Passing ``frequency_hz`` or ``amplitude`` clears calibration fields
        unless replacements are supplied. Frequency quantization follows the
        new point's ``bias_frequency_quantized`` setting.
        """
        if "frequency_hz" in changes or "amplitude" in changes:
            for f in BiasPoint._CAL_FIELDS:
                changes.setdefault(f, None)
        self.bias = replace(self.bias, **changes)
        return self.bias


# ─── ResonatorCatalog ─────────────────────────────────────────────────────────


class ResonatorCatalog:
    """A collection of named resonators on one module.

    Look up members by name; iteration defaults to bias-frequency order.
    ``name`` labels the catalog and defaults to ``"module <module>"``.

    ``min_separation_hz`` optionally rejects nearby bias frequencies when
    adding members. It defaults to None (no spacing check); retuning a member
    does not recheck spacing. ``from_dict`` restores the saved rule unless
    overridden.

    Catalogs may span several NCO bands. ``crs.multisweep`` measures them in
    groups; ``crs.apply_bias`` requires all tones to fit within one band.
    """

    # Stamped into to_dict output and checked by from_dict, so a file written by
    # a version of this module that shaped things differently fails loudly
    # instead of being half-understood. Bump it whenever the dict shape changes.
    SCHEMA_VERSION = 4

    # Older shapes from_dict can still read. Version 1 stored `resonators` as a
    # list of entries each carrying its own `name`; 2 keys them by name, so a
    # reader can look one up instead of scanning. That is a shape change and so
    # a version bump, but the old shape is unambiguous — no reason to strand
    # files already on disk over it.
    #
    # 3 added `bias_sweep` to a bias point: the trace its calibration was read
    # off. Reading an older file back needs nothing — the field defaults to
    # None, which is what a bias point that never had one says. The bump is for
    # the other direction, so that a file written now fails on an older reader
    # rather than arriving there as an unexpected keyword.
    #
    # 4 added `name`, the catalog's own. A file without one reads back under the
    # default, so again the bump is for the other direction: a reader that would
    # drop the name on the floor should refuse the file instead.
    READABLE_SCHEMA_VERSIONS = (1, 2, 3, 4)

    def __init__(
        self,
        resonators: Iterable[Resonator],
        module: int,
        min_separation_hz: float | None = None,
        name: str | None = None,
    ):
        """
        Args:
            resonators: the members; names and channels must be unique.
            module: the readout module these channel numbers refer to.
            min_separation_hz: reject bias frequencies this close together or
                closer. The default, ``None``, allows any spacing, including
                none at all; 0.0 rejects only exactly equal frequencies. See
                ``_check_frequency``.
            name: what this catalog is — an array, a wafer, a cooldown — for
                your own bookkeeping. Free-form. Defaults to
                ``"module <module>"``.
        """
        if min_separation_hz is not None and min_separation_hz < 0:
            raise ValueError(
                f"min_separation_hz={min_separation_hz}: must be a separation in "
                f"Hz (>= 0), or None to allow any spacing."
            )
        default_name = f"module {module}"
        if name is None:
            name = default_name
        elif not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"name={name!r}: a catalog's name is free-form text. Pass None "
                f"to take the default, {default_name!r}."
            )
        self.name = name
        self.module = module
        self.min_separation_hz = min_separation_hz
        # The one store. Channel is read off the resonators themselves rather
        # than mirrored into a second index that could fall out of step.
        self._by_name: dict[str, Resonator] = {}
        for r in resonators:
            self._add(r)

    # -- invariants -----------------------------------------------------------

    def _check_frequency(self, r: Resonator):
        """Reject a bias frequency that collides with one already present.

        The default, ``min_separation_hz=None``, skips this altogether. Nothing
        downstream depends on frequencies being distinct — the cost of two tones
        on one frequency is that the two channels read the same thing, which is
        fine when you meant it, and a caller who has already made their
        separation cut in ``find_resonances`` should not have to argue with a
        second one here.

        ``min_separation_hz=0.0`` rejects exactly equal floats. Because bias
        frequencies arrive quantized, that also catches the realistic symptom of
        ``find_resonances`` splitting one resonator: two peaks a hair apart land
        on one grid point, which is exactly what the hardware would have done
        with them. It stays a weak check past one grid step — set something
        physically motivated for anything wider.

        Comparison is inclusive — a pair exactly ``min_separation_hz`` apart
        collides — which is what makes 0.0 mean "no two tones share a
        frequency", and matches ``find_resonances``' separation pass.
        """
        threshold = self.min_separation_hz
        if threshold is None:
            return
        for other in self._by_name.values():
            gap = abs(other.bias.frequency_hz - r.bias.frequency_hz)
            if gap <= threshold:
                rule = (
                    "min_separation_hz=0.0 asks for distinct frequencies; drop "
                    "it, or pass None, to allow duplicates."
                    if threshold == 0
                    else f"This catalog requires more than {threshold:g} Hz "
                    f"between bias frequencies; these are {gap:g} Hz apart."
                )
                raise ValueError(
                    f"Bias frequency {r.bias.frequency_hz / 1e6:.6f} MHz "
                    f"({r.name!r}) collides with {other.name!r} at "
                    f"{other.bias.frequency_hz / 1e6:.6f} MHz. {rule}"
                )

    def _add(self, r: Resonator):
        if r.name in self._by_name:
            raise ValueError(f"Duplicate resonator name {r.name!r}.")
        if r.channel < 1:
            raise ValueError(
                f"channel={r.channel} ({r.name!r}): hardware channels are 1-based."
            )
        for other in self._by_name.values():
            if other.channel == r.channel:
                raise ValueError(
                    f"Duplicate channel {r.channel} ({r.name!r}); already held by "
                    f"{other.name!r}."
                )
        self._check_frequency(r)
        self._by_name[r.name] = r

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_frequencies(
        cls,
        frequencies_hz: Iterable[float],
        module: int,
        amplitude: float,
        names: list[str] | Callable[[Sequence[float]], list[str]] | None = None,
        **kwargs,
    ) -> ResonatorCatalog:
        """Seed a catalog from found resonances. Channels 1..N in frequency order.

        Each resonator gets a ``BiasPoint`` at its found frequency, carrying no
        calibration — the operating point as first guessed. Multisweep and bias
        finding move it from there. Found frequencies come off a sweep grid,
        not the tone grid, so expect them to shift by up to half a tone-grid
        step on the way in.

        ``amplitude`` is required rather than defaulted: the probe amplitude is
        a real measurement choice, and there is no value that is right for an
        arbitrary array.

        This is where a resonator's name is minted, and it is minted once: from
        here on the name is the catalog's key, it keys every result dict the
        measurement algorithms return, and it round-trips through ``to_dict``
        and ``to_csv``.

        ``names`` is either a list or a namer. A **list** is paired with
        ``frequencies_hz`` positionally *before* sorting, so parallel lists stay
        associated no matter what order they arrive in. A **namer** is a
        function of the sorted frequencies returning one name each, and the
        default is
        :func:`~rfmux.resonator_names.syllabic_names` — short made-up words like
        ``BOTA``, drawn fresh each time::

            from rfmux.resonator_names import (
                numbered_names, syllabic_names_from_frequency)

            ResonatorCatalog.from_frequencies(found, module=2, amplitude=0.01)
            ...(names=numbered_names)                   # R0001…
            ...(names=syllabic_names_from_frequency)    # stable per resonator
            ...(names=partial(numbered_names, prefix="kid"))

        A drawn name is deliberately not an index. ``R0007`` asserts a position
        in frequency order, and that assertion goes stale the first time a
        resonator is removed or retuned while still looking authoritative. The
        ordering is not lost — ``channel`` records it, and ``resonators()``
        recomputes it live from the bias frequencies, which is the version that
        stays true.
        """
        freqs = [float(f) for f in frequencies_hz]
        if names is None:
            names = syllabic_names
        if callable(names):
            ordered = sorted(freqs)
            drawn = names(ordered)
            if len(drawn) != len(ordered):
                # getattr, not .__name__: a partial() has no name, and partial
                # is what the docstring above recommends for a custom prefix.
                who = getattr(names, "__name__", repr(names))
                raise ValueError(
                    f"{who} returned {len(drawn)} names for "
                    f"{len(ordered)} frequencies."
                )
            paired = list(zip(ordered, drawn))
        else:
            if len(names) != len(freqs):
                raise ValueError(f"{len(names)} names for {len(freqs)} frequencies.")
            paired = sorted(zip(freqs, names), key=lambda p: p[0])
        return cls(
            [
                Resonator(
                    name=n,
                    channel=i + 1,
                    bias=BiasPoint(frequency_hz=f, amplitude=amplitude),
                )
                for i, (f, n) in enumerate(paired)
            ],
            module=module,
            **kwargs,
        )

    # -- dict-like ------------------------------------------------------------

    def __getitem__(self, name: str) -> Resonator:
        return self._by_name[name]

    def __iter__(self) -> Iterator[Resonator]:
        return iter(self.resonators())

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __delitem__(self, name: str):
        """``del catalog[name]`` — ``remove`` without the returned resonator."""
        self.remove(name)

    def by_channel(self, channel: int) -> Resonator:
        for r in self._by_name.values():
            if r.channel == channel:
                return r
        raise KeyError(f"No resonator on channel {channel}.")

    def resonators(
        self, order: Literal["frequency", "channel"] = "frequency"
    ) -> list[Resonator]:
        """Return members sorted by bias frequency or hardware channel.

        Frequency order is the default. Channels keep their assigned numbers
        when resonators are retuned or removed, so the orders may differ.
        """
        if order == "frequency":
            key = lambda r: r.bias.frequency_hz  # noqa: E731
        elif order == "channel":
            key = lambda r: r.channel  # noqa: E731
        else:
            raise ValueError(f"order={order!r}: expected 'frequency' or 'channel'.")
        return sorted(self._by_name.values(), key=key)

    def names(self, order: Literal["frequency", "channel"] = "frequency") -> list[str]:
        """The resonator names, low bias frequency first. See ``resonators``."""
        return [r.name for r in self.resonators(order)]

    def remove(self, name: str) -> Resonator:
        """Remove and return a named resonator, preserving other channel bindings.

        The freed channel can be reused. Raises KeyError for an unknown name.
        """
        try:
            return self._by_name.pop(name)
        except KeyError:
            # Bounded, so a 500-resonator array's error stays readable.
            known = self.names()
            shown = ", ".join(known[:5])
            if len(known) > 5:
                shown += f" (and {len(known) - 5} more)"
            raise KeyError(
                f"No resonator named {name!r}. This catalog holds {shown}."
            ) from None

    def clear_bifurcations(self) -> None:
        """Clear stored bifurcation amplitudes in place, keeping all other fields."""
        for resonator in self:
            resonator.update_bias_point(bifurcated_at=None)

    def copy(self) -> ResonatorCatalog:
        """Deep copy. THE threading rule: workers operate on ``catalog.copy()``;
        the GUI swaps its reference when the worker's completed signal fires.

        A resonator is scalars plus, once it has been biased, the one trace its
        calibration came off, so the copy costs a few kB per resonator rather
        than the sweeps it was measured in.
        """
        return _copy.deepcopy(self)

    # -- display --------------------------------------------------------------

    def __repr__(self) -> str:
        head = (
            f"ResonatorCatalog({self.name!r}, module={self.module}, "
            f"{len(self)} resonators)"
        )
        rows = [f"  {'name':<7}{'ch':>3}  {'bias MHz':>12}  {'amp':>7}"]
        for r in self:
            rows.append(
                f"  {r.name:<7}{r.channel:>3}  "
                f"{r.bias.frequency_hz / 1e6:>12.6f}  {r.bias.amplitude:>7.4f}"
            )
        return "\n".join([head] + rows)

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Builtins and arrays only — files never contain these classes.

        Everything here is a builtin except the two arrays inside a
        ``bias_sweep``, which stay arrays: ``store`` writes pickles of builtins
        and ndarrays, and a trace turned into a list of Python floats on the
        way out would be slower to read and no more portable.

        ``resonators`` is keyed by name, the same way the catalog itself is, so
        a reader that wants one resonator says ``d["resonators"]["BOTA"]``
        rather than scanning for it. The name is the key and so is not repeated
        inside the entry. Insertion is in frequency order, which dicts keep,
        but nothing needs to lean on that — ``from_dict`` takes the order back
        off the frequencies, the same as everywhere else.

        ``name`` and ``min_separation_hz`` are part of the record like every
        other field, and :meth:`from_dict` reads them back — the separation rule
        applied, the name carried.
        """
        return {
            "schema_version": self.SCHEMA_VERSION,
            "name": self.name,
            "module": self.module,
            "min_separation_hz": self.min_separation_hz,
            "resonators": {
                r.name: {
                    "channel": r.channel,
                    "bias": _bias_dict(r.bias),
                    "notes": dict(r.notes),
                }
                for r in self
            },
        }

    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> ResonatorCatalog:
        """Rebuild a catalog from ``to_dict`` output.

        The dict carries everything the object does, the separation rule
        included, so reading one back restores the catalog that was written
        instead of deciding what to make of it. The rule arrives the way it
        does in every other constructor, which means it is checked against the
        frequencies in the file — and that is a check worth having here, since
        retuning through ``Resonator.update_bias_point`` is not policed: a
        catalog whose tones were walked together after it was built fails on
        the way back in rather than coming back claiming a spacing it does
        not have.

        The name comes back the same way, and ``from_dict(d, name=...)``
        renames the catalog on the way in — a file written before catalogs had
        names carries none, and comes back under the default like any other
        unnamed catalog.

        ``from_dict(d, min_separation_hz=...)`` reads the file under a rule of
        your own instead — a tighter one to audit it with, or ``None`` to open a
        file whose rule you no longer want to be held to. A file written before
        the rule was persisted has no key at all and comes back under ``None``,
        the same as any other catalog with no rule.
        """
        version = d.get("schema_version")
        if version not in cls.READABLE_SCHEMA_VERSIONS:
            readable = ", ".join(str(v) for v in cls.READABLE_SCHEMA_VERSIONS)
            raise ValueError(
                f"Unsupported schema_version {version!r}; this module writes "
                f"{cls.SCHEMA_VERSION} and reads {readable}."
            )
        stored = d["resonators"]
        # schema_version 1 wrote a list of entries, each with its own name.
        entries = (
            stored.items()
            if isinstance(stored, dict)
            else ((rd["name"], rd) for rd in stored)
        )
        resonators = [
            Resonator(
                name=name,
                channel=rd["channel"],
                bias=BiasPoint(**rd["bias"]),
                notes=rd.get("notes", {}),
            )
            for name, rd in entries
        ]
        # The file's rule unless the caller named one of their own, so that a
        # round trip is a round trip. Anything else the file carries and
        # __init__ does not take is ignored — including a file written while the
        # catalog still carried an NCO frequency, which is why removing that
        # field did not need a schema bump: neither direction of the round trip
        # loses a resonator over it.
        kwargs.setdefault("min_separation_hz", d.get("min_separation_hz"))
        kwargs.setdefault("name", d.get("name"))
        return cls(resonators, module=d["module"], **kwargs)

    # -- CSV ------------------------------------------------------------------
    #
    # A spreadsheet-editable bias table. Deliberately lossy: it carries the
    # operating point and nothing else. `notes`, `bias_frequency_quantized` and
    # every calibration field — `df_calibration` and `bias_sweep` with them —
    # are dropped; a trace does not go in a cell anyway. So is the catalog's own
    # `name`, which is not a per-row fact: pass it, and the separation rule, to
    # `from_csv` alongside `module`, and note that a row read back comes in
    # quantized whether or not it was written that way. Use to_dict for a
    # faithful round-trip.

    CSV_COLUMNS = (
        "name",
        "channel",
        "bias_frequency_hz",
        "bias_amplitude",
    )

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=self.CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in self:
            w.writerow(
                {
                    "name": r.name,
                    "channel": r.channel,
                    "bias_frequency_hz": f"{r.bias.frequency_hz:.6f}",
                    "bias_amplitude": f"{r.bias.amplitude:.6f}",
                }
            )
        return buf.getvalue()

    @classmethod
    def from_csv(cls, text: str, module: int, **kwargs) -> ResonatorCatalog:
        """Read a bias table. Columns are matched by header name, so they may
        appear in any order.
        """
        reader = csv.DictReader(io.StringIO(text))
        missing = set(cls.CSV_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {', '.join(sorted(missing))}. "
                f"Expected a header with: {', '.join(cls.CSV_COLUMNS)}."
            )

        resonators = []
        for lineno, row in enumerate(reader, start=2):
            bias_freq = (row["bias_frequency_hz"] or "").strip()
            bias_amp = (row["bias_amplitude"] or "").strip()
            if not bias_freq or not bias_amp:
                raise ValueError(
                    f"line {lineno}: bias_frequency_hz and bias_amplitude are both "
                    f"required — every resonator has an operating point."
                )
            try:
                resonators.append(
                    Resonator(
                        name=row["name"],
                        channel=int(row["channel"]),
                        bias=BiasPoint(
                            frequency_hz=float(bias_freq),
                            amplitude=float(bias_amp),
                        ),
                    )
                )
            except ValueError as e:
                raise ValueError(f"line {lineno}: {e}") from None
        return cls(resonators, module=module, **kwargs)
