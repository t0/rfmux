"""
Typed resonator model.

Three types, in order of containment:

* ``BiasPoint`` — a tone parked on a resonator, plus the calibration valid at
  that tone. Frozen, because the tone and its calibration are one fact.
* ``Resonator`` — identity, hardware binding, and current tuning state for one
  resonator.
* ``ResonatorMap`` — the per-module collection that algorithms accept and
  return.

The map holds only what is small and canonical: identity, the operating point,
and the calibrations downstream measurements need. Sweep data is *not* stored
here. Analysis reduces a sweep to the handful of scalars that belong on a
``BiasPoint`` and the traces themselves stay with the caller, so the map is
cheap to copy, cheap to save, and cannot disagree with itself.

Nothing in this module imports Qt or CRS. It is a data model, and the
algorithms and GUI layers are both callers.
"""

from __future__ import annotations

import copy as _copy
import csv
import io
import math
from dataclasses import dataclass, field, replace, asdict

from typing import Iterable, Iterator, Literal

from .transferfunctions import COMB_SAMPLING_FREQ

# The hardware tone grid. Bias frequencies are programmed as multiples of this.
#
# NOTE: this deliberately does not use transferfunctions.BASE_FREQUENCY, which
# is COMB_SAMPLING_FREQ / 256 / 2**12 — twice this value — and carries a
# "TODO: verify still appropriate" comment. Every code path that actually
# quantizes a bias frequency uses the value below: bias_kids.py hardcodes it as
# the literal 298.0232238769531. Expressed here as a derivation rather than a
# literal so the relationship to the sampling frequency is visible. If the
# firmware's grid is ever confirmed to be BASE_FREQUENCY, this is the one line
# to change.
TONE_GRID_HZ = COMB_SAMPLING_FREQ / 256 / 2**13  # ≈ 298.023 Hz


# ─── BiasPoint ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BiasPoint:
    """A tone parked on a resonator + the calibration valid at that tone.

    Frozen on purpose: the bias tone and its calibration are one fact. To move
    the tone you build a new ``BiasPoint``; calibration fields you don't pass
    are ``None``. A frequency carrying some *other* tone's calibration is
    unrepresentable.
    """

    frequency_hz: float
    amplitude: float  # normalized DAC units, (0, 1]
    dI_df: float | None = None  # V/Hz at this bias point
    dQ_df: float | None = None
    iq_rotation_deg: float | None = None
    bifurcated_at: float | None = None  # amplitude where bifurcation first seen

    # Fields that describe *this* tone and are therefore invalidated by moving
    # it. Consumed by Resonator.set_bias.
    _CAL_FIELDS = ("dI_df", "dQ_df", "iq_rotation_deg", "bifurcated_at")

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

    @property
    def df_calibration(self) -> complex | None:
        """1/(dI_df + j·dQ_df) in Hz/V. Derived — can never go stale."""
        if self.dI_df is None or self.dQ_df is None:
            return None
        d = complex(self.dI_df, self.dQ_df)
        return 1.0 / d if abs(d) > 0 else None

    def power_dbm(self, dac_scale_dbm: float) -> float:
        return dac_scale_dbm + 20.0 * math.log10(self.amplitude)

    def quantized(self) -> BiasPoint:
        """Round the frequency onto the hardware tone grid.

        Calibration is kept: the shift is under half a grid step
        (≈ 149 Hz), which is small compared to a resonator's width.
        """
        return replace(
            self, frequency_hz=round(self.frequency_hz / TONE_GRID_HZ) * TONE_GRID_HZ
        )


# ─── Resonator ────────────────────────────────────────────────────────────────


@dataclass(slots=True, eq=False)
class Resonator:
    """Identity, hardware binding and tuning state for one resonator.

    Note this is distinct from ``rfmux.core.schema.Resonator``, which is the
    hardware-map ORM row. This one is a plain value object that measurement
    code passes around.
    """

    name: str
    channel: int  # 1-based hardware channel; permanent binding
    center_frequency_hz: float  # sweep seed; always present
    bias: BiasPoint | None = None  # None until found or assigned
    notes: dict = field(default_factory=dict)  # explicitly the junk drawer

    def set_bias(self, **changes) -> BiasPoint:
        """Build or amend this resonator's ``BiasPoint``.

        Moving the tone (``frequency_hz`` or ``amplitude``) drops calibration
        fields unless new values are passed explicitly, so stale calibration
        stays structurally impossible even through this convenience path.
        Changing only calibration leaves the tone alone.
        """
        if self.bias is None:
            missing = {"frequency_hz", "amplitude"} - changes.keys()
            if missing:
                raise TypeError(
                    f"{self.name} has no bias yet, so set_bias needs "
                    f"{' and '.join(sorted(missing))}. "
                    f"Pass frequency_hz and amplitude to establish the tone."
                )
            self.bias = BiasPoint(**changes)
        else:
            if "frequency_hz" in changes or "amplitude" in changes:
                for f in BiasPoint._CAL_FIELDS:
                    changes.setdefault(f, None)
            self.bias = replace(self.bias, **changes)
        return self.bias


# ─── ResonatorMap ─────────────────────────────────────────────────────────────


class ResonatorMap:
    """Per-module, ordered, dict-like collection of Resonators.

    The object algorithms accept and return::

        rmap = ResonatorMap.from_frequencies(found, module=2)
        await crs.multisweep(rmap, span_hz=200e3, npoints_per_sweep=101)
        find_bias_points(rmap, sweeps)
        await crs.apply_bias(rmap)

    Iteration is in channel order. Lookup is by name.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        resonators: Iterable[Resonator],
        module: int,
        nco_frequency_hz: float | None = None,
        min_separation_hz: float | None = None,
    ):
        """
        Args:
            resonators: the members; names and channels must be unique.
            module: the readout module these channel numbers refer to.
            nco_frequency_hz: NCO the channel frequencies are offset from.
            min_separation_hz: if given, reject center frequencies closer
                together than this. The default (None) rejects only exactly
                equal frequencies — see ``_check_frequency``.
        """
        self.module = module
        self.nco_frequency_hz = nco_frequency_hz
        self.min_separation_hz = min_separation_hz
        self._by_name: dict[str, Resonator] = {}
        self._by_channel: dict[int, str] = {}
        for r in resonators:
            self._add(r)

    # -- invariants -----------------------------------------------------------

    def _check_frequency(self, r: Resonator):
        """Reject a center frequency that collides with one already present.

        With ``min_separation_hz`` unset this catches only exactly equal
        floats, which is a weak check: two peaks a microhertz apart are the
        realistic symptom of ``find_resonances`` splitting one resonator, and
        they pass. Set ``min_separation_hz`` to something physically motivated
        to catch that.
        """
        threshold = self.min_separation_hz
        for other in self._by_name.values():
            gap = abs(other.center_frequency_hz - r.center_frequency_hz)
            collides = gap == 0 if threshold is None else gap < threshold
            if collides:
                raise ValueError(
                    f"Center frequency {r.center_frequency_hz / 1e6:.6f} MHz "
                    f"({r.name!r}) collides with {other.name!r} at "
                    f"{other.center_frequency_hz / 1e6:.6f} MHz. "
                    f"Each resonator needs a distinct frequency."
                )

    def _add(self, r: Resonator):
        if r.name in self._by_name:
            raise ValueError(f"Duplicate resonator name {r.name!r}.")
        if r.channel in self._by_channel:
            raise ValueError(
                f"Duplicate channel {r.channel} ({r.name!r}); already held by "
                f"{self._by_channel[r.channel]!r}."
            )
        if r.channel < 1:
            raise ValueError(
                f"channel={r.channel} ({r.name!r}): hardware channels are 1-based."
            )
        self._check_frequency(r)
        self._by_name[r.name] = r
        self._by_channel[r.channel] = r.name

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_frequencies(
        cls,
        frequencies_hz: Iterable[float],
        module: int,
        names: list[str] | None = None,
        **kwargs,
    ) -> ResonatorMap:
        """Sorted ascending; channels 1..N in frequency order.

        Supplied ``names`` are paired with ``frequencies_hz`` positionally
        *before* sorting, so parallel lists stay associated no matter what
        order they arrive in. Without names, resonators are called R0001… in
        frequency order.
        """
        freqs = [float(f) for f in frequencies_hz]
        if names is None:
            paired = [(f, None) for f in sorted(freqs)]
            paired = [(f, f"R{i + 1:04d}") for i, (f, _) in enumerate(paired)]
        else:
            if len(names) != len(freqs):
                raise ValueError(f"{len(names)} names for {len(freqs)} frequencies.")
            paired = sorted(zip(freqs, names), key=lambda p: p[0])
        return cls(
            [
                Resonator(name=n, channel=i + 1, center_frequency_hz=f)
                for i, (f, n) in enumerate(paired)
            ],
            module=module,
            **kwargs,
        )

    # -- dict-like ------------------------------------------------------------

    def __getitem__(self, name: str) -> Resonator:
        return self._by_name[name]

    def __iter__(self) -> Iterator[Resonator]:
        return iter(
            [self._by_name[self._by_channel[ch]] for ch in sorted(self._by_channel)]
        )

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def by_channel(self, channel: int) -> Resonator:
        try:
            return self._by_name[self._by_channel[channel]]
        except KeyError:
            raise KeyError(f"No resonator on channel {channel}.") from None

    def biased(self) -> list[Resonator]:
        return [r for r in self if r.bias is not None]

    def clear_biases(self):
        for r in self:
            r.bias = None

    def copy(self) -> ResonatorMap:
        """Deep copy. THE threading rule: workers operate on ``rmap.copy()``;
        the GUI swaps its reference when the worker's completed signal fires.

        Cheap because the map holds no sweep data — only scalars per resonator.
        """
        return _copy.deepcopy(self)

    # -- display --------------------------------------------------------------

    def __repr__(self) -> str:
        head = (
            f"ResonatorMap(module={self.module}, {len(self)} resonators, "
            f"{len(self.biased())} biased)"
        )
        rows = [
            f"  {'name':<7}{'ch':>3}  {'center MHz':>12}  {'bias MHz':>12}  {'amp':>7}"
        ]
        for r in self:
            b = r.bias
            rows.append(
                f"  {r.name:<7}{r.channel:>3}  {r.center_frequency_hz / 1e6:>12.6f}  "
                + (
                    f"{b.frequency_hz / 1e6:>12.6f}  {b.amplitude:>7.4f}"
                    if b
                    else f"{'—':>12}  {'—':>7}"
                )
            )
        return "\n".join([head] + rows)

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain builtins only — files never contain these classes."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "module": self.module,
            "nco_frequency_hz": self.nco_frequency_hz,
            "resonators": [
                {
                    "name": r.name,
                    "channel": r.channel,
                    "center_frequency_hz": r.center_frequency_hz,
                    "bias": asdict(r.bias) if r.bias else None,
                    "notes": dict(r.notes),
                }
                for r in self
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResonatorMap:
        if d.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {d.get('schema_version')!r}; "
                f"this module writes {cls.SCHEMA_VERSION}."
            )
        resonators = [
            Resonator(
                name=rd["name"],
                channel=rd["channel"],
                center_frequency_hz=rd["center_frequency_hz"],
                bias=BiasPoint(**rd["bias"]) if rd["bias"] else None,
                notes=rd.get("notes", {}),
            )
            for rd in d["resonators"]
        ]
        return cls(
            resonators,
            module=d["module"],
            nco_frequency_hz=d.get("nco_frequency_hz"),
        )

    # -- CSV ------------------------------------------------------------------
    #
    # A spreadsheet-editable bias table. Deliberately lossy: it carries the
    # operating point and nothing else. `notes`, `nco_frequency_hz` and every
    # calibration field (and so df_calibration) are dropped. Use to_dict for a
    # faithful round-trip.

    CSV_COLUMNS = (
        "name",
        "channel",
        "center_frequency_hz",
        "bias_frequency_hz",
        "bias_amplitude",
    )

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=self.CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in self:
            b = r.bias
            w.writerow(
                {
                    "name": r.name,
                    "channel": r.channel,
                    "center_frequency_hz": f"{r.center_frequency_hz:.6f}",
                    "bias_frequency_hz": f"{b.frequency_hz:.6f}" if b else "",
                    "bias_amplitude": f"{b.amplitude:.6f}" if b else "",
                }
            )
        return buf.getvalue()

    @classmethod
    def from_csv(cls, text: str, module: int, **kwargs) -> ResonatorMap:
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
            if bool(bias_freq) != bool(bias_amp):
                raise ValueError(
                    f"line {lineno}: bias_frequency_hz and bias_amplitude must "
                    f"either both be given or both be blank."
                )
            try:
                resonators.append(
                    Resonator(
                        name=row["name"],
                        channel=int(row["channel"]),
                        center_frequency_hz=float(row["center_frequency_hz"]),
                        bias=(
                            BiasPoint(
                                frequency_hz=float(bias_freq),
                                amplitude=float(bias_amp),
                            )
                            if bias_freq
                            else None
                        ),
                    )
                )
            except ValueError as e:
                raise ValueError(f"line {lineno}: {e}") from None
        return cls(resonators, module=module, **kwargs)
