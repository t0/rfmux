"""
Typed resonator model, and a typed collection of resonator objects.

Three types, outermost first::

    ResonatorCatalog    one per module; holds N Resonators
    └── Resonator       one per detector; holds exactly one BiasPoint
        └── BiasPoint   one tone: frequency, amplitude, and the calibration
                        measured at that tone

``BiasPoint`` is frozen and the other two are not, which is a statement about
where each one's guarantees hold. A bias point validated at construction stays
valid for as long as it exists, so a tone can never be carrying a calibration
that was measured somewhere else. A resonator's guarantee is narrower: its
identity is meant to be permanent, but its operating point moves
all through tuning, and ``set_bias`` is the chokepoint that keeps calibration
from outliving the tone it belongs to. A catalog checks its members as they
join and does
not re-check them afterwards.

The catalog holds only what is small and canonical: identity, the operating
point, and the calibrations downstream measurements need. Sweep data is *not*
stored here. Analysis reduces a sweep to the handful of scalars that belong on
a ``BiasPoint`` and the traces themselves stay with the caller, so the catalog
is cheap to copy, cheap to save, and cannot disagree with itself.

``reference-notebooks/Demos/network_analyses_find_resonances_make_resonator_catalog.md``
works through all of this against a simulated array, including what the frozen
bias point means the first time you retune a detector and find its calibration
gone. Read that before relying on the invariants here.

"""

from __future__ import annotations

import copy as _copy
import csv
import io
import math
from dataclasses import dataclass, field, replace, asdict

from typing import Iterable, Iterator

from .transferfunctions import BASE_FREQUENCY


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

        The grid is ``transferfunctions.BASE_FREQUENCY``, the single definition
        every quantizing path in the tree uses. Calibration is kept: the shift
        is under half a grid step, which is small compared to a resonator's
        width.
        """
        return replace(
            self, frequency_hz=round(self.frequency_hz / BASE_FREQUENCY) * BASE_FREQUENCY
        )


# ─── Resonator ────────────────────────────────────────────────────────────────


@dataclass(slots=True, eq=False)
class Resonator:
    """Identity, hardware binding and tuning state for one resonator.

    There is exactly one frequency per resonator and it lives on ``bias``: the
    current best estimate of where this resonator's tone belongs. It is seeded
    from ``find_resonances``, refined by multisweep and by bias finding, and
    quantized by apply-bias. There is deliberately no separate sweep-centre
    field — the sweep centre is multisweep's business, and a second frequency
    is a second thing to keep in agreement.

    ``bias`` is required. A resonator we cannot say a frequency for is not a
    resonator we know about, so there is no unbiased state to test for, clear
    to, or round-trip. Seeding a catalog from ``find_resonances`` gives every
    member an operating point immediately; everything after that moves it.

    Note this is distinct from ``rfmux.core.schema.HWMResonator``, which is the
    hardware-map ORM row. This one is a plain value object that measurement
    code passes around.
    """

    name: str
    channel: int  # 1-based hardware channel; permanent binding
    bias: BiasPoint
    notes: dict = field(default_factory=dict)  # explicitly the junk drawer

    def set_bias(self, **changes) -> BiasPoint:
        """Amend this resonator's ``BiasPoint``.

        Moving the tone (``frequency_hz`` or ``amplitude``) drops calibration
        fields unless new values are passed explicitly, so stale calibration
        stays structurally impossible even through this convenience path.
        Changing only calibration leaves the tone alone.
        """
        if "frequency_hz" in changes or "amplitude" in changes:
            for f in BiasPoint._CAL_FIELDS:
                changes.setdefault(f, None)
        self.bias = replace(self.bias, **changes)
        return self.bias


# ─── ResonatorCatalog ─────────────────────────────────────────────────────────


class ResonatorCatalog:
    """Per-module, ordered, dict-like collection of Resonators.

    The object algorithms accept and return::

        catalog = ResonatorCatalog.from_frequencies(found, module=2, amplitude=0.01)
        await crs.multisweep(catalog, span_hz=200e3, npoints_per_sweep=101)
        find_bias_points(catalog, sweeps)
        await crs.apply_bias(catalog)

    Iteration is in channel order. Lookup is by name.

    Frequency collisions are checked when a resonator joins the catalog, which
    is where a duplicate from ``find_resonances`` shows up. How close is too
    close is ``min_separation_hz``, and setting it to ``None`` turns the check
    off entirely for the deliberate case of two tones on one frequency.
    Retuning through ``Resonator.set_bias`` is not re-checked — two tones can be
    walked onto one frequency after the fact. Worth a ``validate()`` pass once
    there is a caller that retunes in bulk.
    """

    # Stamped into to_dict output and required exactly by from_dict, so a file
    # written by a future (or past) version of this module fails loudly instead
    # of being half-understood. Bump it whenever the dict shape changes in a
    # way from_dict cannot absorb.
    SCHEMA_VERSION = 1

    def __init__(
        self,
        resonators: Iterable[Resonator],
        module: int,
        nco_frequency_hz: float | None = None,
        min_separation_hz: float | None = 0.0,
    ):
        """
        Args:
            resonators: the members; names and channels must be unique.
            module: the readout module these channel numbers refer to.
            nco_frequency_hz: NCO the channel frequencies are offset from.
            min_separation_hz: reject bias frequencies this close together or
                closer. The default, 0.0, rejects only exactly equal
                frequencies; ``None`` allows any spacing, including none at
                all. See ``_check_frequency``.
        """
        if min_separation_hz is not None and min_separation_hz < 0:
            raise ValueError(
                f"min_separation_hz={min_separation_hz}: must be a separation in "
                f"Hz (>= 0), or None to allow any spacing."
            )
        self.module = module
        self.nco_frequency_hz = nco_frequency_hz
        self.min_separation_hz = min_separation_hz
        # The one store. Channel is read off the resonators themselves rather
        # than mirrored into a second index that could fall out of step.
        self._by_name: dict[str, Resonator] = {}
        for r in resonators:
            self._add(r)

    # -- invariants -----------------------------------------------------------

    def _check_frequency(self, r: Resonator):
        """Reject a bias frequency that collides with one already present.

        Two tones on one frequency is normally a hardware conflict, not a
        bookkeeping nicety, so the default threshold of 0.0 Hz still rejects
        exactly equal floats. It is a weak check: two peaks a microhertz apart
        are the realistic symptom of ``find_resonances`` splitting one
        resonator, and they pass. Set ``min_separation_hz`` to something
        physically motivated to catch that.

        Comparison is inclusive — a pair exactly ``min_separation_hz`` apart
        collides — which is what makes 0.0 mean "no two tones share a
        frequency", and matches ``find_resonances``' separation pass.

        ``min_separation_hz=None`` skips the check altogether, for the caller
        who means to park two tones on one frequency. Nothing downstream
        depends on frequencies being distinct; the cost is that the two channels
        read the same thing, which is only useful if you know that.
        """
        threshold = self.min_separation_hz
        if threshold is None:
            return
        for other in self._by_name.values():
            gap = abs(other.bias.frequency_hz - r.bias.frequency_hz)
            if gap <= threshold:
                rule = (
                    "Each resonator needs a distinct frequency; pass "
                    "min_separation_hz=None to allow duplicates."
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
        names: list[str] | None = None,
        **kwargs,
    ) -> ResonatorCatalog:
        """Seed a catalog from found resonances. Channels 1..N in frequency order.

        Each resonator gets a ``BiasPoint`` at its found frequency, carrying no
        calibration — the operating point as first guessed. Multisweep and bias
        finding move it from there.

        ``amplitude`` is required rather than defaulted: the probe amplitude is
        a real measurement choice, and there is no value that is right for an
        arbitrary array.

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
        return iter(sorted(self._by_name.values(), key=lambda r: r.channel))

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def by_channel(self, channel: int) -> Resonator:
        for r in self._by_name.values():
            if r.channel == channel:
                return r
        raise KeyError(f"No resonator on channel {channel}.")

    def copy(self) -> ResonatorCatalog:
        """Deep copy. THE threading rule: workers operate on ``catalog.copy()``;
        the GUI swaps its reference when the worker's completed signal fires.

        Cheap because the catalog holds no sweep data — only scalars per
        resonator.
        """
        return _copy.deepcopy(self)

    # -- display --------------------------------------------------------------

    def __repr__(self) -> str:
        head = f"ResonatorCatalog(module={self.module}, {len(self)} resonators)"
        rows = [f"  {'name':<7}{'ch':>3}  {'bias MHz':>12}  {'amp':>7}"]
        for r in self:
            rows.append(
                f"  {r.name:<7}{r.channel:>3}  "
                f"{r.bias.frequency_hz / 1e6:>12.6f}  {r.bias.amplitude:>7.4f}"
            )
        return "\n".join([head] + rows)

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain builtins only — files never contain these classes."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "module": self.module,
            "nco_frequency_hz": self.nco_frequency_hz,
            "min_separation_hz": self.min_separation_hz,
            "resonators": [
                {
                    "name": r.name,
                    "channel": r.channel,
                    "bias": asdict(r.bias),
                    "notes": dict(r.notes),
                }
                for r in self
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResonatorCatalog:
        if d.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {d.get('schema_version')!r}; "
                f"this module writes {cls.SCHEMA_VERSION}."
            )
        resonators = [
            Resonator(
                name=rd["name"],
                channel=rd["channel"],
                bias=BiasPoint(**rd["bias"]),
                notes=rd.get("notes", {}),
            )
            for rd in d["resonators"]
        ]
        return cls(
            resonators,
            module=d["module"],
            nco_frequency_hz=d.get("nco_frequency_hz"),
            # Absent in files written before the rule was persisted; those were
            # written under the old default, which is this one.
            min_separation_hz=d.get("min_separation_hz", 0.0),
        )

    # -- CSV ------------------------------------------------------------------
    #
    # A spreadsheet-editable bias table. Deliberately lossy: it carries the
    # operating point and nothing else. `notes`, `nco_frequency_hz`,
    # `min_separation_hz` and every calibration field (and so df_calibration)
    # are dropped — pass the separation rule to `from_csv` if the table needs
    # it. Use to_dict for a faithful round-trip.

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
