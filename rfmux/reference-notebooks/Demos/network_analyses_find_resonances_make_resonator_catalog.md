---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Network analysis → find resonances → resonator catalog

The first three steps of tuning an array, from the Python API — **without
opening Periscope**. Sweep a band, locate the dips, and turn them into the
bookkeeping object every later step consumes.

The second half of the notebook is about that object on its own: a
`ResonatorCatalog` does not have to come from a sweep, and you will want to
build one by hand, save it, edit it in a spreadsheet, and load it back.

| Piece | Module |
|---|---|
| The sweep | `rfmux.algorithms.measurement.take_netanal` (`crs.take_netanal`) |
| The dip finder | `rfmux.tuning.find_resonances` |
| The netanal wrapper | `rfmux.tuning.find_resonances_in_netanal` |
| The array bookkeeping | `rfmux.core.resonators` |

The finder is deliberately two functions. `find_resonances` takes two arrays and
knows nothing about netanal, files, or the CRS, so it runs on a saved sweep or a
simulated trace as readily as on a live one;
`find_resonances_in_netanal` is a thin wrapper that unpacks what
`crs.take_netanal()` returned and calls it. Section 3 uses the wrapper and
section 4 uses the array form, on the same data.

## How to use this document

**This is a runnable notebook, not a web page.** Every grey block below is a live
code cell: put the cursor in it and press **Shift+Enter** to execute it.

- **Run the cells in order, top to bottom.** Later cells use variables the
  earlier ones defined, so skipping ahead fails with a `NameError`. *Kernel →
  Restart Kernel and Run All Cells* starts clean.
- **The outputs you see are the ones you just produced.** This file is stored as
  jupytext markdown, which keeps no saved outputs, so a cell is blank until you
  run it. Nothing here can show you a stale number from someone else's run.
- **Editing is encouraged.** Change the band, the dip-depth threshold, the Q
  limits, and re-run — that is what this document is for. The shipped copy is
  read-only, so *File → Save Notebook As…* to keep your changes.
- **Section 1 is the one exception to running everything.** It offers three ways
  to get a CRS and you run only the one that fits. Everything after it is
  identical whichever you chose.
- **Sections 6 onward need no hardware at all.** The catalog is a plain data
  model. If you only came for that, run the imports cell and skip to section 6.

```python
%matplotlib inline

import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rfmux
from rfmux.tuning import find_resonances, find_resonances_in_netanal
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import BASE_FREQUENCY

# Files written by section 7. Reference notebooks are provisioned to a
# read-only directory, so writing next to the notebook would fail for anyone
# who opened it from Periscope. Override with RFMUX_DEMO_OUTPUT.
OUTPUT_DIR = Path(os.environ.get(
    "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_catalog_demo"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODULE = 1

# The band to sweep. The simulated array in section 2 is placed inside it.
FMIN, FMAX = 0.6e9, 1.05e9
PROBE_AMPLITUDE = 0.001   # normalized DAC units, shared by the sweep and the
                          # catalog's bias points

crs = None          # set by whichever cell in section 1 or 2 you run
IS_MOCK = False     # True only if THIS notebook created the simulation

print(f"files → {OUTPUT_DIR}")
```

## 1. Connect

Everything through section 5 needs a CRS. **Run exactly one** of the three
options:

| | When to use it | Where |
|---|---|---|
| **A. Attach to a running board** | Periscope launched this notebook, or you know the board's address | below |
| **B. Your own board** | You have hardware and its serial | below |
| **C. Simulate one** | No hardware, and nothing already running | section 2 |

A network analysis needs nothing tuned beforehand — finding the resonators *is*
the first step. It does overwrite every channel's frequency and amplitude on the
module it sweeps, and zeroes them when it finishes, so do not run it on a module
someone else is using.

### A. Attach to a board that is already running

Periscope sets `RFMUX_CRS_HOSTNAME` when it launches a notebook, which is how
this cell finds the board with no configuration from you. It is not magic and
not required: paste an address into `HOSTNAME` and it works from any kernel.

```python
HOSTNAME = os.environ.get("RFMUX_CRS_HOSTNAME")   # or paste "127.0.0.1:43431"
SERIAL = os.environ.get("RFMUX_CRS_SERIAL", "0000")

if HOSTNAME:
    _s = rfmux.load_session(
        f'!HardwareMap [ !CRS {{ serial: "{SERIAL}", '
        f'hostname: "{HOSTNAME}" }} ]')
    crs = _s.query(rfmux.CRS).one()
    await crs.resolve()
    print(f"attached to CRS {SERIAL} at {HOSTNAME}")
else:
    print("Nothing advertised a board. Set HOSTNAME above, or use option B "
          "(your own board) or section 2 (simulation).")
```

### B. Your own board

```python
# SERIAL = "0042"
# _s = rfmux.load_session(f'!HardwareMap [ !CRS {{ serial: "{SERIAL}" }} ]')
# crs = _s.query(rfmux.CRS).one()
# await crs.resolve()
# print(f"connected to CRS {SERIAL}")
```

## 2. Simulate a board

**Skip this section entirely if section 1 already gave you a CRS** — the cell
below no-ops in that case.

Ten simulated LEKIDs spread across the band, with a fixed random seed so the
array is the same every run. `auto_bias_kids` is off: parking carriers is what
tuning does *after* this notebook, and leaving it off means the sweep sees the
array as it actually arrives — nothing known about it yet.

These are high-Q resonators, a few kHz wide. That matters for section 4, and it
is realistic: it is the normal condition for a survey sweep to be much coarser
than the features it is looking for.

> ⚠️ **This cell refuses to run if something is already streaming.** A network
> analysis does not use the UDP streamers, but the simulator starts them, and two
> simulations sending to one port give every reader both streams interleaved with
> nothing to say so. If Periscope is in mock mode, attach to *its* simulation
> with option 1A instead.

```python
MOCK_CONFIG = {
    "num_resonances": 10,
    "freq_start": 0.6e9,          # inside [FMIN, FMAX] so the sweep can see them
    "freq_end": 1.0e9,
    "resonator_random_seed": 42,  # same array every run
    "auto_bias_kids": False,      # nothing is tuned yet — that is the point
}

if crs is not None:
    print("already connected — skip this cell")
else:
    from rfmux.streamer import find_streamer_conflict

    if os.environ.get("RFMUX_CRS_HOSTNAME"):
        raise RuntimeError(
            "Periscope launched this notebook and is already driving a CRS.\n"
            "Run option 1A above to attach to that one.")

    conflict = find_streamer_conflict()
    if conflict:
        raise RuntimeError(
            f"Something is already using the streamer port — {conflict}.\n"
            "Attach to what is running with option 1A, or stop it, then re-run "
            "this cell.")

    from rfmux.mock.helpers import create_mock_crs

    crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG, verbose=False)
    IS_MOCK = True
    print(f"simulated CRS with {MOCK_CONFIG['num_resonances']} resonators "
          f"between {MOCK_CONFIG['freq_start']/1e9:.2f} and "
          f"{MOCK_CONFIG['freq_end']/1e9:.2f} GHz")
```

## 3. Run the network analysis

`crs.take_netanal()` measures complex S21 across a band. It does the
book-keeping a wide sweep needs — splitting the span into chunks of at most
`max_span` so each gets its own NCO setting, measuring up to `max_chans`
frequencies at a time as a comb, and rotating each chunk onto the previous one
using the single frequency they share, so the whole trace has one continuous
phase.

`npoints` is the parameter that decides what you can find, and the section-4
heading below is about why. 60,000 points across this 450 MHz band puts a sample
every 7.5 kHz. Scroll back to the simulator's output above and look at the
`Linewidth:` lines: 1.7 to 3.6 kHz. **Even this sweep steps over each resonator
in two or three samples** — and that is the normal condition for a survey sweep,
not a flaw in the example.

The sweep takes about half a minute against the simulator.

```python
netanal = await crs.take_netanal(
    amp=PROBE_AMPLITUDE,
    fmin=FMIN,
    fmax=FMAX,
    npoints=60_000,
    nsamps=10,          # averages per point
    max_chans=1023,     # frequencies measured simultaneously
    max_span=500e6,     # one NCO setting per 500 MHz of span
    module=MODULE,
)

frequencies = netanal["frequencies"]
iq = netanal["iq_complex"]

print(f"keys: {list(netanal)}")
print(f"{len(frequencies)} points, "
      f"{frequencies[0]/1e6:.1f}–{frequencies[-1]/1e6:.1f} MHz, "
      f"{np.mean(np.diff(frequencies))/1e3:.2f} kHz spacing")
```

The result is three plain arrays. Nothing about it is special — this is what
every function downstream expects, and a sweep you loaded from a file is
equally good input.

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax1.plot(frequencies / 1e6, 20 * np.log10(np.abs(iq) / np.median(np.abs(iq))), lw=0.6)
ax1.set_ylabel("|S21| [dB, normalized]")
ax2.plot(frequencies / 1e6, netanal["phase_degrees"], lw=0.6)
ax2.set_ylabel("phase [deg]"); ax2.set_xlabel("frequency [MHz]")
ax1.set_title("network analysis")
plt.tight_layout(); plt.show()
```

## 4. Find the resonances

`find_resonances_in_netanal` unpacks the sweep and searches it. The search
converts `|S21|` to dB, inverts it, and hands it to
`scipy.signal.find_peaks` with two physical constraints:

- **`min_dip_depth_db`** — how deep a dip has to be to count, as a prominence
  against the local baseline. Lower it to 0.3–0.5 dB for shallow (overcoupled or
  low-Q) resonators.
- **`min_Q` / `max_Q`** — converted to a width window. `min_Q` sets the *widest*
  dip accepted and `max_Q` the *narrowest*; the narrow end is what rejects
  single-sample noise spikes.

There is a third, `min_separation_hz`, which defaults to 0 Hz and so does nothing
here. It is a collision cut: candidates closer together than the threshold are
**all** removed, not thinned to the deepest one, because a tone parked on either
member of a collided pair still reads the other's response. Set it when you know
the separation below which your readout cannot operate a detector.

```python
found = find_resonances_in_netanal(
    netanal,
    min_dip_depth_db=1.0,
    min_Q=1e4,
    max_Q=1e7,
    label=f"module {MODULE}",
)

print(found)
```

The result is a `ResonanceSearch`, not a bare list. It carries the accepted
candidates, everything a rejection pass threw out and why, the processed trace
that was searched, and the settings used:

```python
for c in found.candidates:
    print(f"  {c.frequency_hz/1e6:11.4f} MHz   depth {c.depth_db:5.2f} dB   "
          f"width {c.width_hz/1e3:6.2f} kHz   Q ≈ {c.q_estimate:.3g}")

print(f"\n{len(found)} accepted, {len(found.rejected)} rejected")
for c in found.rejected:
    print(f"  {c.frequency_hz/1e6:11.4f} MHz — {c.rejected_because}")

if IS_MOCK:
    print(f"\nthe simulated array has {MOCK_CONFIG['num_resonances']}")
    assert len(found) > 0, "found nothing — see section 5"
```

`q_estimate` is a sorting key, not a measurement. It is `frequency / width`
where the width is read off the dB trace at half the dip's prominence, which is
not a half-power width — and `data_exponent` (default 2.0, which raises `|S21|`
to a power to deepen dips against the noise) shifts it further.

The widths printed above make the sharper point. They cluster around 8–16 kHz,
which is one to two sample spacings, because a dip narrower than the spacing
gets smeared to about one sample no matter how narrow it really is. So these
widths measure the *sweep*, not the resonators — the simulated Qs are near
3×10⁵ and the finder reports under 10⁵. Read a `q_estimate` as "resolved" or
"unresolved" and no more; fitting the resonance is what gives you Q, and that is
multisweep's job. `depth_db` *is* corrected back to true dB.

```python
searched_db = found.magnitude_db          # the trace the finder actually saw
hits = found.resonance_frequencies_hz     # just the frequencies

plt.figure(figsize=(11, 4))
plt.plot(found.frequencies_hz / 1e6, searched_db, lw=0.6, zorder=1)
plt.scatter(hits / 1e6, searched_db[[c.index for c in found.candidates]],
            s=140, facecolor="none", edgecolor="red", zorder=3, label="found")
for c in found.rejected:
    plt.axvline(c.frequency_hz / 1e6, color="darkorange", ls="--", alpha=0.6)
plt.xlabel("frequency [MHz]")
plt.ylabel(f"|S21|^{found.settings['data_exponent']:g} [dB]")
plt.title(f"{len(found)} resonances")
plt.legend(); plt.tight_layout(); plt.show()
```

### Sweep resolution decides what you can find

A survey sweep is usually much coarser than the resonators in it, and the failure
mode is quiet: the finder returns fewer resonances and nothing reports that the
sampling is why. Compare the same band at a quarter the resolution — and note
that `find_resonances` here is called on two bare arrays, no netanal dict
involved, because that is all it ever needed.

```python
coarse = await crs.take_netanal(
    amp=PROBE_AMPLITUDE, fmin=FMIN, fmax=FMAX, npoints=5_000,
    nsamps=10, max_chans=1023, module=MODULE)

for name, sweep in (("coarse", coarse), ("fine", netanal)):
    f = sweep["frequencies"]
    n = len(find_resonances(f, sweep["iq_complex"],
                            min_dip_depth_db=1.0, min_Q=1e4, max_Q=1e7))
    print(f"{name:>7}: {len(f):>6} points, "
          f"{np.mean(np.diff(f))/1e3:>6.2f} kHz spacing → {n} resonances")
```

A dip narrower than the point spacing is one or two samples deep at best, and
whether it is caught depends on where the samples happen to land. If a count
comes back low, the first thing to change is `npoints`, not the thresholds — and
if you already know roughly where the array is, sweeping a narrower band at the
same `npoints` buys the same resolution for less time.

## 5. Seed a resonator catalog

`ResonanceSearch.to_catalog()` is where anonymous dips become tracked
resonators. Each gets a name, a hardware channel, and a `BiasPoint` at its found
frequency — the operating point as first guessed. Multisweep and bias finding
move it from there.

`amplitude` is required rather than defaulted, because probe power is a real
measurement choice with no value that is right for an arbitrary array. Channels
are assigned 1..N in frequency order and are a permanent binding.

```python
catalog = found.to_catalog(module=MODULE, amplitude=PROBE_AMPLITUDE)
print(catalog)
```

That object is what the rest of tuning consumes and returns. From here you would
run a multisweep around each bias frequency, fit, and pick bias points — see
`simplified_tuning_flow.py` in this folder for the whole chain.

Everything below needs no hardware.

## 6. The catalog on its own

Three types, in order of containment:

| Type | Is | Mutable? |
|---|---|---|
| `BiasPoint` | a tone parked on a resonator, plus the calibration valid *at that tone* | frozen |
| `Resonator` | identity, hardware binding, current tuning state for one resonator | yes |
| `ResonatorCatalog` | the per-module collection algorithms accept and return | yes |

The catalog holds only what is small and canonical. **Sweep data is not stored
in it** — analysis reduces a sweep to the few scalars that belong on a
`BiasPoint`, and the traces stay with the caller. That is what makes a catalog
cheap to copy, cheap to save, and unable to disagree with itself.

`BiasPoint` is frozen on purpose: the tone and the calibration measured at it are
one fact. A frequency carrying some *other* tone's calibration is
unrepresentable.

### Building one by hand

```python
by_hand = ResonatorCatalog(
    [
        Resonator("blue", channel=1, bias=BiasPoint(1.010e9, amplitude=0.01)),
        Resonator("green", channel=2, bias=BiasPoint(1.030e9, amplitude=0.01)),
        Resonator(
            "red", channel=3,
            bias=BiasPoint(1.050e9, amplitude=0.005,
                           dI_df=1.2e-6, dQ_df=-3.4e-6,
                           iq_rotation_deg=12.0),
            notes={"wafer": "W17", "comment": "the junk drawer, explicitly"},
        ),
    ],
    module=2,
    nco_frequency_hz=1.03e9,
)
print(by_hand)
print(f"\nred's df calibration: {by_hand['red'].bias.df_calibration:.4g} Hz/V")
print(f"red at a -30 dBm DAC scale: "
      f"{by_hand['red'].bias.power_dbm(-30):.2f} dBm")
```

`df_calibration` is a property derived from `dI_df` and `dQ_df` rather than a
stored field, so it cannot go stale.

### From a list of frequencies

The same constructor the finder used. Names are optional; without them
resonators are `R0001…` in frequency order. Supplied names are paired with the
frequencies **positionally, before sorting**, so parallel lists stay associated
however they arrive:

```python
named = ResonatorCatalog.from_frequencies(
    [1.05e9, 1.01e9, 1.03e9],           # deliberately out of order
    names=["high", "low", "middle"],    # paired with the line above
    module=2,
    amplitude=0.01,
)
print(named)
```

### Reading it

Iteration is in channel order; lookup is by name.

```python
print(f"len           : {len(named)}")
print(f"by name       : {named['middle'].bias.frequency_hz/1e6:.3f} MHz")
print(f"by channel    : {named.by_channel(1).name}")
print(f"'low' present : {'low' in named}")
print(f"iteration     : {[r.name for r in named]}  (channel order)")
print(f"module        : {named.module}")
```

### Amending a bias point

`Resonator.set_bias()` builds a new `BiasPoint` rather than mutating one. Moving
the tone — changing `frequency_hz` or `amplitude` — drops the calibration fields
unless you pass new values explicitly, so stale calibration stays structurally
impossible even through this convenience path. Changing only calibration leaves
the tone alone.

```python
r = by_hand["red"]
print(f"before      : {r.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {r.bias.iq_rotation_deg}")

r.set_bias(frequency_hz=1.0505e9)        # moving the tone
print(f"tone moved  : {r.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {r.bias.iq_rotation_deg}   <- calibration dropped")

r.set_bias(iq_rotation_deg=15.0)         # calibration only
print(f"recalibrated: {r.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {r.bias.iq_rotation_deg}   <- tone untouched")
```

### Quantizing onto the tone grid

The hardware can only place a tone on a multiple of `BASE_FREQUENCY`. Applying a
bias quantizes it; `BiasPoint.quantized()` is that operation, and it keeps the
calibration, because the shift is under half a grid step and so is tiny compared
to a resonator's width.

```python
print(f"tone grid: {BASE_FREQUENCY:.6f} Hz")

# Four tenths of a step above a real grid point, so the shift is unambiguous.
# (Asking for a round 1.0 GHz would not demonstrate anything: it happens to sit
# very nearly on the grid already.)
grid_point = round(1.0e9 / BASE_FREQUENCY) * BASE_FREQUENCY
off_grid = BiasPoint(grid_point + 0.4 * BASE_FREQUENCY, amplitude=0.01,
                     iq_rotation_deg=12.0)

print(f"requested: {off_grid.frequency_hz:.6f} Hz")
print(f"on grid  : {off_grid.quantized().frequency_hz:.6f} Hz")
print(f"moved by : "
      f"{off_grid.quantized().frequency_hz - off_grid.frequency_hz:+.6f} Hz "
      f"({(off_grid.quantized().frequency_hz - off_grid.frequency_hz)/BASE_FREQUENCY:+.1f} steps)")
print(f"rotation : {off_grid.quantized().iq_rotation_deg} deg  <- calibration kept")
```

### Invariants, and what they refuse

Names and channels must be unique, channels are 1-based, and two resonators may
not sit on one frequency — two tones on one frequency is a hardware conflict,
not a bookkeeping nicety. These are checked when a resonator joins the catalog.

```python
def refused(what, thunk):
    try:
        thunk()
    except (ValueError, KeyError) as e:
        print(f"{what}:\n    {e}\n")
    else:
        print(f"{what}: accepted?!\n")

refused("duplicate channel", lambda: ResonatorCatalog(
    [Resonator("a", 1, BiasPoint(1.01e9, 0.01)),
     Resonator("b", 1, BiasPoint(1.02e9, 0.01))], module=2))

refused("identical frequencies", lambda: ResonatorCatalog(
    [Resonator("a", 1, BiasPoint(1.01e9, 0.01)),
     Resonator("b", 2, BiasPoint(1.01e9, 0.01))], module=2))

refused("amplitude in dBm, not DAC units",
        lambda: BiasPoint(1.01e9, amplitude=-30.0))

refused("no such resonator", lambda: named["nope"])
```

By default only *exactly* equal frequencies collide, which is a weak check: two
candidates a hertz apart are the realistic symptom of a split resonance and
would pass. `min_separation_hz` on the catalog makes it physical:

```python
refused("closer than the catalog's minimum separation", lambda: ResonatorCatalog(
    [Resonator("a", 1, BiasPoint(1.010000e9, 0.01)),
     Resonator("b", 2, BiasPoint(1.010002e9, 0.01))],
    module=2, min_separation_hz=10e3))
```

This is a different check from the finder's `min_separation_hz`, which cuts
collided *candidates* — both members, since a tone on either one reads the other
— before a catalog exists. The catalog's check guards the collection afterwards.

### Copying, and the one threading rule

`copy()` is a deep copy, and cheap, because the catalog holds no sweep data.
**Workers operate on `catalog.copy()`; the GUI swaps its reference when the
worker's completed signal fires.** That is the whole concurrency discipline.

```python
mine = by_hand.copy()
mine["blue"].set_bias(frequency_hz=1.0111e9)
print(f"copy     : {mine['blue'].bias.frequency_hz/1e6:.4f} MHz")
print(f"original : {by_hand['blue'].bias.frequency_hz/1e6:.4f} MHz")
```

## 7. Saving and loading

Two formats, for two purposes.

### Dictionaries — the faithful round trip

`to_dict()` returns plain builtins only, so a saved file never contains these
classes and can be read without importing rfmux. It carries everything:
calibration, `notes`, the NCO. Put it in JSON, pickle, HDF5 attributes,
wherever.

```python
d = by_hand.to_dict()
print(f"schema_version : {d['schema_version']}")
print(f"top-level keys : {list(d)}")
print(f"one resonator  : {d['resonators'][2]}")

restored = ResonatorCatalog.from_dict(d)
print(f"\nround trip: {len(restored)} resonators, "
      f"names {[r.name for r in restored]}")
print(f"calibration survived: "
      f"{restored['red'].bias.iq_rotation_deg} deg, notes {restored['red'].notes}")
```

`from_dict` requires the `schema_version` it was written with, so a file from a
future (or past) version of the module fails loudly instead of being
half-understood.

```python
import json

json_path = OUTPUT_DIR / "catalog.json"
json_path.write_text(json.dumps(by_hand.to_dict(), indent=2))
print(f"wrote {json_path} ({json_path.stat().st_size} bytes)")

from_disk = ResonatorCatalog.from_dict(json.loads(json_path.read_text()))
print(from_disk)
```

### CSV — the spreadsheet-editable bias table

**Deliberately lossy.** It carries the operating point and nothing else:
`notes`, `nco_frequency_hz` and every calibration field are dropped. Use it to
hand someone a bias table they can edit; use `to_dict` when you need everything
back.

```python
csv_path = OUTPUT_DIR / "bias_table.csv"
csv_path.write_text(by_hand.to_csv())
print(csv_path.read_text())
```

Reading takes the CSV *text* and the module — the module is not in the file,
because channel numbers are meaningless without knowing which module they belong
to. Columns are matched by header name, so they may appear in any order.

```python
reloaded = ResonatorCatalog.from_csv(csv_path.read_text(), module=2)
print(reloaded)
print(f"\ncalibration after a CSV round trip: "
      f"{reloaded['red'].bias.iq_rotation_deg}   <- dropped, as documented")
```

Hand-editing works, which is the point. Both columns of an operating point are
required — every resonator has one — so a blank cell is an error naming the
line, not a resonator with no tone:

```python
edited = "\n".join([
    "name,channel,bias_frequency_hz,bias_amplitude",
    "blue,1,1010000000.0,0.01",
    "green,2,1030000000.0,0.02",      # amplitude changed by hand
    "red,3,1050000000.0,0.005",
])
print(ResonatorCatalog.from_csv(edited, module=2))

refused("a missing bias amplitude", lambda: ResonatorCatalog.from_csv(
    "name,channel,bias_frequency_hz,bias_amplitude\nblue,1,1010000000.0,\n",
    module=2))
```

Finally, the catalog the sweep produced, saved both ways:

```python
if crs is not None:
    (OUTPUT_DIR / "found.csv").write_text(catalog.to_csv())
    (OUTPUT_DIR / "found.json").write_text(json.dumps(catalog.to_dict(), indent=2))
    print(f"wrote {len(catalog)} resonators to {OUTPUT_DIR}")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {p.name:<16} {p.stat().st_size:>7} bytes")
```

## 8. Where this maps in Periscope

| Periscope control | API equivalent |
|---|---|
| *Network Analysis* panel, **Take Netanal** | `crs.take_netanal(...)` |
| **Find Resonances** button + its dialog | `find_resonances_in_netanal(...)` |
| Expected / Min Dip Depth / Min Q / Max Q fields | the same-named arguments |
| Min Separation field | `min_separation_hz` — but see section 4: it is now a collision cut, and the GUI has not caught up |
| The red dashed markers on the plot | `ResonanceSearch.candidates` |
| The resonance list the multisweep dialog inherits | `ResonanceSearch.to_catalog(...)` |

```python
# Only tear down a simulation this notebook created. If section 1 attached to
# Periscope's CRS, stopping its streamer would kill Periscope's live plots.
if IS_MOCK:
    await crs.stop_udp_streaming()
    print("simulated streamer stopped")
elif crs is not None:
    print("left the board alone — it is not ours to stop")
```
