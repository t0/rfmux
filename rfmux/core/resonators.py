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
    """Round a frequency in Hz to the nearest hardware tone-grid point.

    Use this for both tones and NCOs so their frequency offsets also lie on
    multiples of ``transferfunctions.BASE_FREQUENCY``.
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

    # Fields cleared by update_bias_point when frequency or amplitude is supplied.
    _CAL_FIELDS = (
        "dI_df",
        "dQ_df",
        "iq_rotation_deg",
        "bifurcated_at",
        "bias_sweep",
    )

    # Fields retained from the sweep used to compute calibration.
    BIAS_SWEEP_KEYS = (
        "frequencies",
        "iq_volts",
        "original_center_frequency",
        "sweep_amplitude",
        "sweep_direction",
    )

    # Frequency and IQ arrays are required; the remaining sweep fields are optional.
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
        """Check that a stored sweep has frequency and IQ arrays of equal length.

        This checks the mapping and array lengths, not the values in the arrays.
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
        """Return ``1 / (dI_df + j*dQ_df)`` in Hz/V, or None for missing or zero slope."""
        if self.dI_df is None or self.dQ_df is None:
            return None
        d = complex(self.dI_df, self.dQ_df)
        return 1.0 / d if abs(d) > 0 else None

    def power_dbm(self, dac_scale_dbm: float) -> float:
        """Return this tone's drive power in dBm using the module's DAC full scale."""
        return float(convert_dacunits_to_dbm(self.amplitude, dac_scale_dbm))

    def quantize(self) -> BiasPoint:
        """Return a copy with its frequency rounded to the hardware tone grid.

        Bias points normally quantize at construction. Use this for a point built
        with ``bias_frequency_quantized=False``. Calibration and the quantization
        setting are preserved; future updates still follow that setting.
        """
        return replace(self, frequency_hz=on_grid(self.frequency_hz))


def _bias_dict(bias: BiasPoint) -> dict:
    """Return a bias point's fields as a dictionary.

    Copy the ``bias_sweep`` dictionary but share its arrays. This avoids
    copying the measured traces each time a catalog is saved. Callers must
    treat the shared arrays as read-only.
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
    notes: dict = field(default_factory=dict)  # user-supplied metadata

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

    # Version the saved catalog format; reject unsupported versions on load.
    SCHEMA_VERSION = 4

    # Version 1 uses a list of named entries; later versions key entries by name.
    # Missing bias_sweep and catalog name fields use their constructor defaults.
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
        # Store members by name; channel lookups read each resonator directly.
        self._by_name: dict[str, Resonator] = {}
        for r in resonators:
            self._add(r)

    # -- invariants -----------------------------------------------------------

    def _check_frequency(self, r: Resonator):
        """Reject a new member within ``min_separation_hz`` of an existing member.

        The comparison includes the threshold itself. None skips the check;
        0.0 rejects only equal bias frequencies. Bias points normally quantize
        at construction, so nearby input frequencies can become equal.
        This check runs when adding members, not when retuning them.
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
        """Build a catalog with channels 1..N assigned in frequency order.

        Each resonator starts with an uncalibrated ``BiasPoint`` at the supplied
        amplitude, in fractions of DAC full scale. Frequencies are rounded to the
        hardware tone grid, which can shift them by up to half a grid step.

        ``names`` may be a list or a naming function. A list is paired with the
        input frequencies before sorting. A function receives sorted frequencies
        and must return one name per frequency. The default, ``syllabic_names``,
        generates fresh names such as ``BOTA``. For example::

            from functools import partial
            from rfmux.resonator_names import numbered_names

            catalog = ResonatorCatalog.from_frequencies(
                found, module=2, amplitude=0.01,
                names=partial(numbered_names, prefix="kid"))

        Names identify resonators in the catalog, sweep results, and exports.
        They stay unchanged when a resonator is retuned or another is removed.
        Numbered names therefore need not match current frequency order.
        Additional keywords are passed to the catalog constructor.
        """
        freqs = [float(f) for f in frequencies_hz]
        if names is None:
            names = syllabic_names
        if callable(names):
            ordered = sorted(freqs)
            drawn = names(ordered)
            if len(drawn) != len(ordered):
                # Callable objects such as functools.partial may have no __name__.
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
        """Return a deep copy, including stored calibration traces and notes.

        Give workers a copy so they can update bias points independently of the
        catalog displayed by the GUI.
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
        """Return catalog fields as dictionaries, keeping sweep arrays as arrays.

        Resonators are keyed by name, for example ``d["resonators"]["BOTA"]``.
        Entries are inserted in current bias-frequency order. The catalog name,
        module, separation rule, bias points, and notes are included.

        Stored sweep arrays are shared with the catalog; use :meth:`copy` first
        if they must be independent.
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

        Restore the saved name and separation rule unless overridden by keyword
        arguments. Missing values use the constructor defaults.

        Construction checks frequency spacing again. Loading can therefore fail
        if members were retuned closer together after the catalog was built.
        Pass ``min_separation_hz=None`` to load without that check.
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
        # Explicit arguments override saved values; missing fields use defaults.
        kwargs.setdefault("min_separation_hz", d.get("min_separation_hz"))
        kwargs.setdefault("name", d.get("name"))
        return cls(resonators, module=d["module"], **kwargs)

    # -- CSV ------------------------------------------------------------------
    #
    # CSV stores only resonator names, channels, frequencies, and amplitudes.
    # Loading quantizes frequencies and leaves calibration and notes unset.
    # Pass module, catalog name, and separation rule separately to from_csv.
    # Use to_dict to retain all catalog fields.

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
