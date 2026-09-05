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

from typing import Iterable, Iterator, Literal

from .transferfunctions import BASE_FREQUENCY


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
    """A tone parked on a resonator + the calibration valid at that tone.

    Frozen on purpose: the bias tone and its calibration are one fact. To move
    the tone you build a new ``BiasPoint``; calibration fields you don't pass
    are ``None``. A frequency carrying some *other* tone's calibration is
    unrepresentable.

    The frequency is quantized onto the hardware tone grid at construction, so
    ``frequency_hz`` is what the hardware will actually play. Requested,
    recorded and set are then the same number, and no later reader has to
    wonder which of the three it is holding. ``bias_frequency_quantized=False``
    opts out for the caller who needs the exact number they asked for — a sweep
    centre they are doing arithmetic on, say — and nothing downstream
    re-quantizes for them.
    """

    frequency_hz: float
    amplitude: float  # normalized DAC units, (0, 1]
    # Snap frequency_hz onto the tone grid, always. Named for the field it
    # governs so it does not read as a past tense of `quantize()`.
    bias_frequency_quantized: bool = True
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
        if self.bias_frequency_quantized:
            snapped = on_grid(self.frequency_hz)
            if snapped <= 0:
                raise ValueError(
                    f"frequency_hz={self.frequency_hz}: quantizes to 0 Hz — it is "
                    f"less than half a tone-grid step ({BASE_FREQUENCY / 2:g} Hz), "
                    f"so the hardware has nowhere to put it."
                )
            object.__setattr__(self, "frequency_hz", snapped)

    @property
    def df_calibration(self) -> complex | None:
        """1/(dI_df + j·dQ_df) in Hz/V. Derived — can never go stale."""
        if self.dI_df is None or self.dQ_df is None:
            return None
        d = complex(self.dI_df, self.dQ_df)
        return 1.0 / d if abs(d) > 0 else None

    def power_dbm(self, dac_scale_dbm: float) -> float:
        return dac_scale_dbm + 20.0 * math.log10(self.amplitude)

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


# ─── Resonator ────────────────────────────────────────────────────────────────


@dataclass(slots=True, eq=False)
class Resonator:
    """Identity, hardware binding and tuning state for one resonator.

    There is exactly one frequency per resonator and it lives on ``bias``: the
    current best estimate of where this resonator's tone belongs. It is seeded
    from ``find_resonances`` and refined by multisweep and by bias finding; at
    every one of those steps it lands on the hardware tone grid immediately,
    because that is where the tone will go. There is deliberately no separate
    sweep-centre field — the sweep centre is multisweep's business, and a second
    frequency is a second thing to keep in agreement.

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

        A new ``frequency_hz`` is quantized on the way in like any other, so
        what you read back is what the hardware will play, not what you asked
        for.
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
        await crs.multisweep(catalog)
        find_bias_points(catalog, sweeps)
        await crs.apply_bias(catalog)

    Lookup is by name. Iteration is in bias-frequency order — the members
    themselves are an unordered collection, so ``resonators()`` and ``names()``
    take the order you want to pull them out in.

    Frequency collisions are checked when a resonator joins the catalog, if you
    ask for the check: ``min_separation_hz`` says how close is too close, and
    defaults to ``None``, which lets any spacing through, including none at all.
    A duplicate out of ``find_resonances`` is what this catches, and the
    separation cut there is the first place to make it — pass a threshold here
    when you want the catalog to hold the line as well. Every constructor takes
    it, so ``from_frequencies``, ``from_dict`` and ``from_csv`` can each be
    given one.

    Retuning through ``Resonator.set_bias`` is not re-checked — two tones can be
    walked onto one frequency after the fact. Worth a ``validate()`` pass once
    there is a caller that retunes in bulk.

    There is deliberately no NCO frequency here. A catalog is free to span more
    frequency than one NCO can carry — multisweep already re-tunes the NCO as it
    walks across such an array — so a single number recorded on the catalog
    would be a fact about one moment of one operation rather than about the
    array. The NCO in force is the board's to answer for
    (``crs.get_nco_frequency(module=...)``), and ``crs.apply_bias`` sets it from
    the frequencies it is applying. Worth adding back the day a caller turns up
    that has to record which NCO a set of measurements was taken against — but
    it should arrive with that caller, and probably alongside the measurements
    rather than on the catalog.
    """

    # Stamped into to_dict output and checked by from_dict, so a file written by
    # a version of this module that shaped things differently fails loudly
    # instead of being half-understood. Bump it whenever the dict shape changes.
    SCHEMA_VERSION = 2

    # Older shapes from_dict can still read. Version 1 stored `resonators` as a
    # list of entries each carrying its own `name`; 2 keys them by name, so a
    # reader can look one up instead of scanning. That is a shape change and so
    # a version bump, but the old shape is unambiguous — no reason to strand
    # files already on disk over it.
    READABLE_SCHEMA_VERSIONS = (1, 2)

    def __init__(
        self,
        resonators: Iterable[Resonator],
        module: int,
        min_separation_hz: float | None = None,
    ):
        """
        Args:
            resonators: the members; names and channels must be unique.
            module: the readout module these channel numbers refer to.
            min_separation_hz: reject bias frequencies this close together or
                closer. The default, ``None``, allows any spacing, including
                none at all; 0.0 rejects only exactly equal frequencies. See
                ``_check_frequency``.
        """
        if min_separation_hz is not None and min_separation_hz < 0:
            raise ValueError(
                f"min_separation_hz={min_separation_hz}: must be a separation in "
                f"Hz (>= 0), or None to allow any spacing."
            )
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
        names: list[str] | None = None,
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
        """The members as a list, low bias frequency first.

        A catalog is a collection, not a sequence — the resonators in it have
        no inherent order, and nothing is stored in one. What this does is
        *extract* them in an order you name.

        Frequency order is the array as you'd plot or tabulate it, and it is
        what iterating the catalog gives you. Channel order is what you want
        when the members have to line up with per-channel data coming back from
        the board.

        The two agree for a catalog straight out of ``from_frequencies``, which
        assigns channels 1..N in frequency order, and drift apart as soon as
        resonators are retuned or dropped.
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
        """Drop a resonator and return it. Channels are left alone.

        Removing channel 3 from 1..5 leaves 1, 2, 4, 5 — a hole, deliberately.
        Every surviving resonator keeps the channel it was measured on, so
        per-channel data you are already holding stays valid, and so does
        anything the board has been told. Nothing here requires channels to be
        contiguous. Repacking them to 1..N-1 is a separate decision and a
        separate pass; do it by rebuilding the catalog if you want it.

        The freed channel is available again — a later resonator may take it.

        Raises:
            KeyError: if no resonator goes by that name.
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
        """Plain builtins only — files never contain these classes.

        ``resonators`` is keyed by name, the same way the catalog itself is, so
        a reader that wants one resonator says ``d["resonators"]["R0007"]``
        rather than scanning for it. The name is the key and so is not repeated
        inside the entry. Insertion is in frequency order, which dicts keep,
        but nothing needs to lean on that — ``from_dict`` takes the order back
        off the frequencies, the same as everywhere else.
        """
        return {
            "schema_version": self.SCHEMA_VERSION,
            "module": self.module,
            "min_separation_hz": self.min_separation_hz,
            "resonators": {
                r.name: {
                    "channel": r.channel,
                    "bias": asdict(r.bias),
                    "notes": dict(r.notes),
                }
                for r in self
            },
        }

    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> ResonatorCatalog:
        """Rebuild a catalog from ``to_dict`` output.

        Catalog settings are taken from the file, and a keyword given here wins
        over the stored value — ``from_dict(d, min_separation_hz=1e3)`` reads an
        old catalog under a new rule, and raises if it does not meet it.
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
        # The separation rule is absent in files written before it was
        # persisted. Those went out under the old default of 0.0, but every
        # frequency in one already passed that check on the way in, so reading
        # them back under the current default of None cannot let anything
        # through that the writer would have caught.
        # kwargs.setdefault("min_separation_hz", d.get("min_separation_hz"))

        # A file written while the catalog still carried an NCO frequency has
        # that key and it is ignored, which is why removing the field did not
        # need a schema bump: neither direction of the round trip loses a
        # resonator over it.
        return cls(resonators, module=d["module"], **kwargs)

    # -- CSV ------------------------------------------------------------------
    #
    # A spreadsheet-editable bias table. Deliberately lossy: it carries the
    # operating point and nothing else. `notes`, `min_separation_hz`,
    # `bias_frequency_quantized` and every calibration field (and so
    # df_calibration) are dropped — pass the separation rule to
    # `from_csv` if the table needs it, and note that a row read back comes in
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
