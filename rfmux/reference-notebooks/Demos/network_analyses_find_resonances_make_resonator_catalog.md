---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
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
`ResonatorCatalog` does not have to come from a sweep, and at some point you will probably want to
build one by hand, save it, edit it in a spreadsheet, and load it back.

| Piece | Module |
|---|---|
| The sweep | `rfmux.algorithms.measurement.take_netanal` (`crs.take_netanal`) |
| The resonance finder | `rfmux.tuning.find_resonances` |
| The convenience function to run the resonance finder on a netanal | `rfmux.tuning.find_resonances_in_netanal` |
| The array bookkeeping | `rfmux.core.resonators` |

The finder is deliberately two functions. `find_resonances` takes two arrays and
knows nothing about netanal, files, or the CRS, so it runs on a saved sweep or a
simulated trace as readily as on a live one;
`find_resonances_in_netanal` is a thin wrapper that unpacks what
`crs.take_netanal()` returned and calls it. Section 3 uses the wrapper and
the array form both, on the same data.

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
- **How you open it depends on your editor.** This file is jupytext markdown,
  not `.ipynb`. In the JupyterLab session Periscope launches it opens as a
  notebook on double-click; in a JupyterLab you started yourself, right-click →
  *Open With* → *Notebook*. **In VS Code it opens as plain text**, so pair it
  instead: with a jupytext extension installed, right-click → *Open Paired
  Notebook* (the exact wording varies by extension) creates an `.ipynb` beside
  this file and keeps the two in step — run and edit the notebook, and your
  changes flow back into the markdown. If that command does nothing, the
  extension could not find jupytext: it runs whichever interpreter VS Code
  resolved, which is often the base environment rather than the one rfmux is
  installed in. Install jupytext there, point the extension at the right
  interpreter, or skip the extension and run `jupytext --sync <this file>.md`
  from a shell that has it. The `.ipynb` is a local working copy and is
  gitignored; the markdown is the version that is kept, reviewed and tested.
- **Check which kernel you are running.** rfmux has to be importable from the
  interpreter the notebook uses, and if you have more than one checkout, it must
  be the environment installed against *this* one. Getting that wrong looks like
  a `ModuleNotFoundError` for a module you can plainly see on disk, because you
  are importing a different copy of rfmux than the one you are reading. This
  says which copy you actually got:

  ```python
  import sys, rfmux; print(sys.executable); print(rfmux.__file__)
  ```

- **Sections 5 onward need no board at all**, simulated or otherwise. The
  catalog is a plain data model. If you only came for that, run the imports cell
  and skip to section 5.

```python
%matplotlib inline

import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rfmux
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import BASE_FREQUENCY

# Files written by section 6. Reference notebooks are provisioned to a
# read-only directory, so writing next to the notebook would fail for anyone
# who opened it from Periscope. Override with RFMUX_DEMO_OUTPUT.
OUTPUT_DIR = Path(os.environ.get(
    "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_catalog_demo"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODULE = 1

# The band to sweep. The simulated array in section 1 is placed inside it.
FMIN, FMAX = 0.6e9, 1.05e9
PROBE_AMPLITUDE = 0.001   # normalized DAC units, shared by the sweep and the
                          # catalog's bias points

print(f"files → {OUTPUT_DIR}")
```

## 1. Simulate a board

We will generate 10 simulated LEKIDs spread across the band using a fixed random seed, so this
notebook produces the same array and the same numbers every time it is run.

To run the rest of the notebook against real hardware instead, replace this one
cell with a session on your board — everything after it is unchanged:

    session = rfmux.load_session('!HardwareMap [ !CRS { serial: "0042" } ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()

Note that a network analysis overwrites every channel's frequency and amplitude
on the module it sweeps, so do not point it at a module someone else is using.

```python
from rfmux.mock.helpers import create_mock_crs

MOCK_CONFIG = {
    "num_resonances": 10,
    "freq_start": 0.6e9,          # inside [FMIN, FMAX] so the sweep can see them
    "freq_end": 1.0e9,
    "resonator_random_seed": 42,  # same array every run
    "auto_bias_kids": False,      # nothing is tuned yet — that is the point
}

crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG, verbose=False)
print(f"simulated CRS with {MOCK_CONFIG['num_resonances']} resonators "
      f"between {MOCK_CONFIG['freq_start']/1e9:.2f} and "
      f"{MOCK_CONFIG['freq_end']/1e9:.2f} GHz")
```

## 2. Run the network analysis

`crs.take_netanal()` measures complex S21 across a band. 

When searching for resonances, we need to take sufficient measurement points per
frequency span that we have a good chance that one or more points falls within a 
resonance's bandwidth. This is decided with the `npoints` parameter.

```python
netanal = await crs.take_netanal(
    amp=PROBE_AMPLITUDE,
    fmin=FMIN,
    fmax=FMAX,
    npoints=60_000,
    nsamps=10,          # averages per point
    max_chans=1023,     # frequencies measured simultaneously
    module=MODULE,
)

netanal_frequencies = netanal["frequencies"]
netanal_iq_complex = netanal["iq_complex"]

print(f"keys: {list(netanal)}")
print(f"{len(netanal_frequencies)} points, "
      f"{netanal_frequencies[0]/1e6:.1f}–{netanal_frequencies[-1]/1e6:.1f} MHz, "
      f"{np.mean(np.diff(netanal_frequencies))/1e3:.2f} kHz spacing")
```

Now a quick example plotter to take a look at the data. We draft the plotters in
this notebook by hand as an exercise, but canned versions of all three of them
live in `Demos/example_plotting_netanal.py`, next to this notebook —
`plot_netanal` here, and `plot_resonance_search` and `plot_candidate_details` for
the two plots in section 3. There is one such file per topic covered in these
notebooks.

```python
fig, (magnitude_panel, phase_panel) = plt.subplots(
    2, 1, figsize=(11, 6), sharex=True
)

magnitude_db = 20 * np.log10(
    np.abs(netanal_iq_complex) / np.median(np.abs(netanal_iq_complex))
)
magnitude_panel.plot(netanal_frequencies / 1e6, magnitude_db, lw=0.6)
magnitude_panel.set_ylabel("|S21| [dB, normalized]")

phase_panel.plot(netanal_frequencies / 1e6, netanal["phase_degrees"], lw=0.6)
phase_panel.set_ylabel("phase [deg]"); phase_panel.set_xlabel("frequency [MHz]")

magnitude_panel.set_title("network analysis")
plt.tight_layout(); plt.show()
```

## 3. Find the resonances

`rfmux.tuning.find_resonances_in_netanal` unpacks the sweep and searches it.
The search
converts `|S21|` to dB, inverts it (since the peak finder algorithm expects 
positive peaks), and hands it to
`scipy.signal.find_peaks` with two physical constraints:

- **`min_dip_depth_db`** — how deep a dip has to be to count, as a prominence
  against the local baseline. Lower it for shallower resonators, and
  increase it to reduce the likelihood of picking up on noise spikes.

- **`min_Q` / `max_Q`** — converted to a width window. `min_Q` sets the *widest*
  dip accepted and `max_Q` the *narrowest*; the narrow end helps to reject
  single-sample noise spikes.

**Collision mitigation:**

There is a optional parameter, `min_separation_hz`, which defaults to 0 Hz and so does nothing
here. It is a collision cut: candidate resonances that are closer together than the threshold are
**all** removed. Set it when you know
the separation below which your readout cannot operate a detector.

```python
from rfmux.tuning import find_resonances_in_netanal

resonance_search = find_resonances_in_netanal(
    netanal,
    min_dip_depth_db=1.0,
    min_Q=1e4,
    max_Q=1e7,
    label=f"module {MODULE}",
)

print(resonance_search)
```

The result is a `rfmux.tuning.ResonanceSearch`. It carries the accepted
candidates, everything a rejection pass threw out and why, the processed trace
that was searched, and the settings used:

```python
for candidate in resonance_search.candidates:
    print(f"  {candidate.frequency_hz/1e6:11.4f} MHz   "
          f"depth {candidate.depth_db:5.2f} dB   "
          f"width {candidate.width_hz/1e3:6.2f} kHz   "
          f"Q ≈ {candidate.q_estimate:.3g}")

print(f"\n{len(resonance_search)} accepted, "
      f"{len(resonance_search.rejected)} rejected")
for candidate in resonance_search.rejected:
    print(f"  {candidate.frequency_hz/1e6:11.4f} MHz — "
          f"{candidate.rejected_because}")

print(f"\nthe simulated array has {MOCK_CONFIG['num_resonances']}")
assert len(resonance_search) > 0, "found nothing — check the sweep in section 2"
```

`q_estimate` is `frequency / width` — the rough figure the `min_Q` / `max_Q`
window screens on, which the finder applies as a width bound inside `find_peaks`
and reports back here as a Q. **It is not a measurement of the resonators' qualtiy factors.**
Determining a resonator's
real Q factors is multisweep's job: it revisits each candidate with a narrow, dense
sweep and fits it. -- we will talk about that in a separate walkthrough notebook. TODO: to be linked here when available.

```python
# the trace the finder actually saw, and just the frequencies it accepted
searched_magnitude_db = resonance_search.magnitude_db
resonance_frequencies_hz = resonance_search.resonance_frequencies_hz

candidate_indices = [c.index for c in resonance_search.candidates]

plt.figure(figsize=(11, 4))
plt.plot(resonance_search.frequencies_hz / 1e6, searched_magnitude_db,
         lw=0.6, zorder=1)
plt.scatter(resonance_frequencies_hz / 1e6,
            searched_magnitude_db[candidate_indices],
            s=140, facecolor="none", edgecolor="red", zorder=3, label="found")
for candidate in resonance_search.rejected:
    plt.axvline(candidate.frequency_hz / 1e6, color="darkorange", ls="--",
                alpha=0.6)
plt.xlabel("frequency [MHz]")
plt.ylabel("|S21| [dB, normalized]")
plt.title(f"{len(resonance_search)} resonances")
plt.legend(); plt.tight_layout(); plt.show()
```

### Look at the candidates one at a time

The overview plot shows *where* the finder placed the resonances. To see *what
it measured*, zoom in on each candidate and draw the two numbers on: the red
vertical bar is the dip depth (its prominence) and the horizontal bar is the
width, at half that depth.

This is the plot to reach for when a sweep gives you a count you did not expect.
The samples are drawn as points, so you can see how much of each dip the sweep
actually caught.

```python
def plot_candidate_details(search, ncols=5, span_widths=4.0, limit=25):
    """One panel per candidate, with the measured depth and width drawn on."""
    point_spacing_hz = float(np.mean(np.diff(search.frequencies_hz)))
    shown_candidates = search.candidates[:limit]
    nrows = int(np.ceil(len(shown_candidates) / ncols))

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(3.1 * ncols, 2.7 * nrows))
    for panel in axes.flat:
        panel.axis("off")       # panels with no candidate stay blank

    for panel, candidate in zip(axes.flat, shown_candidates):
        panel.axis("on")

        # A window a few widths wide, but never so few samples that there is
        # nothing to look at — an unresolved dip is exactly the case this plot
        # exists to show.
        half_window = max(
            int(np.ceil(span_widths * candidate.width_hz / point_spacing_hz)), 8
        )
        lo_index = max(candidate.index - half_window, 0)
        hi_index = min(candidate.index + half_window + 1,
                       len(search.frequencies_hz))
        offset_khz = (
            search.frequencies_hz[lo_index:hi_index] - candidate.frequency_hz
        ) / 1e3
        panel.plot(offset_khz, search.magnitude_db[lo_index:hi_index],
                   ".-", ms=4, lw=0.8)

        # Depth and width as the finder measured them. The width bar is drawn
        # centred; find_peaks' two crossings are usually a little asymmetric.
        dip_bottom_db = search.magnitude_db[candidate.index]
        panel.vlines(0.0, dip_bottom_db, dip_bottom_db + candidate.depth_db,
                     color="red", lw=1.5)
        panel.hlines(dip_bottom_db + candidate.depth_db / 2,
                     -candidate.width_hz / 2e3, candidate.width_hz / 2e3,
                     color="red", lw=1.5)

        panel.set_title(f"{candidate.frequency_hz / 1e6:.3f} MHz", fontsize=9)
        panel.text(0.04, 0.06,
                   f"{candidate.depth_db:.1f} dB deep\n"
                   f"{candidate.width_hz / 1e3:.1f} kHz wide",
                   transform=panel.transAxes, fontsize=8, va="bottom")
        panel.tick_params(labelsize=7)

    fig.supxlabel("offset from the candidate [kHz]", fontsize=9)
    fig.supylabel("|S21| [dB, normalized]", fontsize=9)
    fig.suptitle(
        f"{len(shown_candidates)} of {len(search)} candidates"
        if len(shown_candidates) < len(search)
        else f"{len(shown_candidates)} candidates"
    )
    fig.tight_layout()
    plt.show()


plot_candidate_details(resonance_search)
```

These are clean dips with several points down each side — a comfortable result,
and a tidier one than a real array usually gives you. On hardware, expect
shallower dips, a baseline that slopes and ripples, and candidates where the
sweep caught only a point or two. When that happens, this is the plot that shows
it, and more `npoints` is the usual answer.

### Sweep resolution decides what you can find

A survey sweep is usually much coarser than the resonators in it, and the failure
mode is quiet: the finder returns fewer resonances and nothing reports that the
sampling is why. Compare the same band at a quarter the resolution:

```python
from rfmux.tuning import find_resonances

coarse_netanal = await crs.take_netanal(
    amp=PROBE_AMPLITUDE, fmin=FMIN, fmax=FMAX, npoints=5_000,
    nsamps=10, max_chans=1023, module=MODULE)

for label, netanal_to_search in (("coarse", coarse_netanal), ("fine", netanal)):
    frequencies = netanal_to_search["frequencies"]
    n_resonances = len(find_resonances(
        frequencies, netanal_to_search["iq_complex"],
        min_dip_depth_db=1.0, min_Q=1e4, max_Q=1e7))
    print(f"{label:>7}: {len(frequencies):>6} points, "
          f"{np.mean(np.diff(frequencies))/1e3:>6.2f} kHz spacing "
          f"→ {n_resonances} resonances")
```

A dip narrower than the point spacing is one or two samples deep at best, and
whether it is caught depends on where the samples happen to land. If a count
comes back low, the first thing to change is `npoints`, not the thresholds — and
if you already know roughly where the array is, sweeping a narrower band at the
same `npoints` buys the same resolution for less time.

### The collisions this sweep cannot see

`min_separation_hz` above can only cut pairs the survey sweep managed to
separate. Two resonators closer together than its resolution arrive as a single
candidate, and nothing here can tell you that is what happened — the collision
cut has one dip to look at and no reason to complain about it.

Where they come apart is the multisweep, which sweeps a narrow span around each
candidate at a resolution the survey could not afford. So the same cut is worth
running again on that data, and
`rfmux.tuning.find_sweeps_with_nearby_resonances` is the function for it: hand it
one module's multisweep result and a `min_separation_hz`, and it hands back the
names whose section turned out to hold a second dip. Drop them from the catalog
the way section 5 does — both members, since a tone on either one still reads the
other.

It is not demonstrated here because it needs multisweep data, which this notebook
does not take; `multisweep.md` in this folder is where that data comes from.

## 4. Seed a resonator catalog

`rfmux.tuning.ResonanceSearch.to_catalog()` is where anonymous dips become tracked
resonators. Each gets a name (a string of the format of your choosing), a hardware channel, and a
`rfmux.core.resonators.BiasPoint` at its found
frequency — the operating point as first guessed. Multisweep and bias finding
will refine and update this BiasPoint as we progress through the tuning flow.

`amplitude` is required, and here is assigned automatically to be the amplitude
 used for the netanal. Channels
are assigned 1..N in frequency order.

```python
catalog = resonance_search.to_catalog(module=MODULE, amplitude=PROBE_AMPLITUDE)
print(catalog)
```

<!-- #region -->
That object is what the rest of tuning consumes and returns. From here you would
run a multisweep around each bias frequency, fit, and pick bias points — see
`simplified_tuning_flow.py` in this folder for the whole chain.


## 5. The catalog on its own

Three types, nested one inside the next:

    ResonatorCatalog    one per module; holds N Resonators
    └── Resonator       one per detector; holds exactly one BiasPoint
        └── BiasPoint   one tone: frequency, amplitude, and the calibration
                        measured at that tone

**The Catalog is the record of your array.** It is what each tuning step takes
in and hands back, what you save at the end of a session, and what you load next
time to pick up where you left off. Everything in it is small — a handful of
numbers per detector — because **sweep data is deliberately not kept here**.
Analysis reduces a sweep to the few scalar values that belong on a `BiasPoint` and the
traces stay separate. This keeps the size of this file small, and prevents mismatches
between saved fields and extraneous data. 

**The Resonator persists.** Its name is fixed the
moment the catalog is built and is meant never to change again: it will be
associated with various data products, allowing you to determine which measurement
pertains to which detector.

**A Resonator has a BiasPoint**, which can be reset as needed during tuning 
processes.
`rfmux.tuning.find_resonances` seeds its original values, and bias finding
routines refine it.
It snaps to the hardware tone grid, to avoid in-band IMD products.

**A given BiasPoint is frozen.** A tone
and the calibration measured at that tone are treated as a single indivisible
fact, so a `BiasPoint` cannot be edited — you have to replace it. The reason is that the
df calibration and the IQ rotation are only meaningful at the exact frequency
and amplitude they were measured at. If you retune the detector to a new bias 
frequency or amplitude, the bias point's calibration information will be wrong,
and a df timestream computed with them will therefore also be wrong. To avoid this
problem, we enforce that you can never modify a `BiasPoint`, only create a new one.


The catalog checks its members as they join. It verifies that each has:
- a unique name
- a unique channel number
- a unique frequency (two Resonators may not have exactly the same bias frequency)
    - TODO: is this check repeated over time? Eg if bias finding going awry, and one 
    Resonator's bias frequency migrates to its neighbours, is this flagged?
    - TODO also consider adding a bypass for this, to allow feedback testing shenanigans.

### Building a Catalog by hand

Catalogs are generated automatically by functions like the resonance finder, but can also be constructed
by hand, which may be instructive as to the layout of the object.
<!-- #endregion -->

```python
by_hand_catalog = ResonatorCatalog(
    [
        Resonator(name="blue", channel=1, bias=BiasPoint(1.010e9, amplitude=0.01)),
        Resonator(name="green", channel=2, bias=BiasPoint(1.030e9, amplitude=0.01)),
        Resonator(
            name="red", channel=3,
            bias=BiasPoint(1.050e9, amplitude=0.005,
                           dI_df=1.2e-6, dQ_df=-3.4e-6,
                           iq_rotation_deg=12.0),
            notes={"comment": "seems suss, operate with caution"},
        ),
    ],
    module=2,
)
print(by_hand_catalog)
print(f"\nred's df calibration: "
      f"{by_hand_catalog['red'].bias.df_calibration:.4g} Hz/V")
print(f"red at a -30 dBm DAC scale: "
      f"{by_hand_catalog['red'].bias.power_dbm(-30):.2f} dBm")
```

Note: `df_calibration` is a property derived from `dI_df` and `dQ_df` rather than a
stored field, so it cannot go stale.

### Making a Catalog from a list of frequencies

The same constructor the finder used. Names are optional; without them
resonators are `R0001…` in frequency order. Supplied names are paired with the
frequencies **positionally, before sorting**, so parallel lists stay associated
however they arrive:

```python
named_catalog = ResonatorCatalog.from_frequencies(
    [1.05e9, 1.01e9, 1.03e9],           # deliberately out of order
    names=["high", "low", "middle"],    # paired with the line above
    module=2,
    amplitude=0.01,
)
print(named_catalog)
```

### Reading from a Catalog

Lookup is by name. The resonators themselves are a collection rather than a
sequence — they have no inherent order, and none is stored — so pulling them
out means saying which order you want them in.

`catalog.resonators()` and `catalog.names()` do that, sorted by bias frequency
lowest first, which is the array as you would plot or tabulate it. Iterating
the catalog (`for resonator in catalog`) gives you the same thing. Pass
`order="channel"` to either one for hardware channel order instead, which is
what you want when the members have to line up with per-channel data coming
back from the board.

The two orderings agree for a catalog fresh from `from_frequencies` or
`to_catalog`, since channels are assigned 1..N in frequency order. They drift
apart once resonators are retuned or removed.

```python
print(f"len           : {len(named_catalog)}")
print(f"by name       : "
      f"{named_catalog['middle'].bias.frequency_hz/1e6:.3f} MHz")
print(f"by channel    : {named_catalog.by_channel(1).name}")
print(f"'low' present : {'low' in named_catalog}")
print(f"iteration     : {[r.name for r in named_catalog]}")
print(f"names()       : {named_catalog.names()}  (frequency order)")
print(f"names(channel): {named_catalog.names(order='channel')}")
print(f"resonators()  : {[r.channel for r in named_catalog.resonators()]}  "
      f"(their channels, in frequency order)")
print(f"module        : {named_catalog.module}")
```

<!-- #region -->
### Amending a bias point

`Resonator.set_bias()` builds a new `BiasPoint` rather than mutating one.


Changing `frequency_hz` or `amplitude` drops the calibration fields
unless you pass new values explicitly, so stale calibration stays structurally
impossible even through this convenience path. 

Changing only calibration leaves
the tone frequency and amplitude alone.
<!-- #endregion -->

```python
red_resonator = by_hand_catalog["red"]
print(f"before      : {red_resonator.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {red_resonator.bias.iq_rotation_deg}")

red_resonator.set_bias(frequency_hz=1.0505e9)        # moving the tone
print(f"tone moved  : {red_resonator.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {red_resonator.bias.iq_rotation_deg}   <- calibration dropped")

red_resonator.set_bias(iq_rotation_deg=15.0)         # calibration only
print(f"recalibrated: {red_resonator.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {red_resonator.bias.iq_rotation_deg}   <- tone untouched")
```

### Removing a resonator from the catalog

This can be done with `catalog.remove('R0001')` or `del catalog['R0001']`.

This is akin to removing a key from a dictionary, and thus everything else about
the catalog is left untouched. In particular, channel numbers are not adjusted to fill
in the missing one.

```python
pruned_catalog = by_hand_catalog.copy()
print(f"before : {pruned_catalog.names()} on channels "
      f"{[r.channel for r in pruned_catalog]}")

dropped_resonator = pruned_catalog.remove("green")
print(f"dropped: {dropped_resonator.name} from channel {dropped_resonator.channel}")

print(f"after  : {pruned_catalog.names()} on channels "
      f"{[r.channel for r in pruned_catalog]}   <- channel 2 is a hole, not reused")

del pruned_catalog["blue"]
print(f"del    : {pruned_catalog.names()}")
```

### Quantizing onto the tone grid

To avoid seeing in-band intermodulation distortion products, the hardware is only
allowed to place a tone on a multiple of
`rfmux.core.transferfunctions.BASE_FREQUENCY`.

When adding a bias frequency to a `BiasPoint`, this quantization is applied automatically,
so there can never be disagreement about what frequency will actually be output to the array.

```python
print(f"tone grid: {BASE_FREQUENCY:.6f} Hz")

# Four tenths of a step above a real grid point, so the shift is unambiguous.
# (Asking for a round 1.0 GHz would not demonstrate anything: it happens to sit
# very nearly on the grid already.)
grid_frequency_hz = round(1.0e9 / BASE_FREQUENCY) * BASE_FREQUENCY
requested_frequency_hz = grid_frequency_hz + 0.4 * BASE_FREQUENCY
quantized_bias = BiasPoint(requested_frequency_hz, amplitude=0.01,
                           iq_rotation_deg=12.0)

moved_by_hz = quantized_bias.frequency_hz - requested_frequency_hz

print(f"requested: {requested_frequency_hz:.6f} Hz")
print(f"recorded : {quantized_bias.frequency_hz:.6f} Hz")
print(f"moved by : {moved_by_hz:+.6f} Hz "
      f"({moved_by_hz/BASE_FREQUENCY:+.1f} steps)")
print(f"rotation : {quantized_bias.iq_rotation_deg} deg  <- calibration kept")
```

The shift is under half a step — far smaller than a resonator's width — so
calibration measured at the requested frequency still holds at the tone that
actually gets played. That is why quantization is not a tone move in the sense
`set_bias` cares about, and does not drop the calibration fields.

If you want the exact number you asked for — a sweep centre you are doing
arithmetic on, say — pass `bias_frequency_quantized=False`, and nothing
downstream will round it for you. `.quantize()` is the one-shot for those.

```python
unquantized_bias = BiasPoint(requested_frequency_hz, amplitude=0.01,
                             bias_frequency_quantized=False)
print(f"kept exact : {unquantized_bias.frequency_hz:.6f} Hz")
print(f".quantize(): {unquantized_bias.quantize().frequency_hz:.6f} Hz")
```

### Invariants, and what they refuse

Names and channels must be unique, and channels are 1-based. These are checked
when a resonator joins the catalog.

Bias frequencies are not policed unless you ask: `min_separation_hz` defaults to
`None`, which lets any spacing through, including two tones at exactly the same
frequency. Pass `min_separation_hz=0.0` to refuse an exact duplicate — a
resonance that `find_resonances` split in two lands there once the frequencies
are quantized — or a wider number when you know the separation below which your
readout cannot operate two detectors. Every constructor takes it, including
`from_dict` and `from_csv`.

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

refused("identical frequencies, with a separation rule asked for",
        lambda: ResonatorCatalog(
            [Resonator("a", 1, BiasPoint(1.01e9, 0.01)),
             Resonator("b", 2, BiasPoint(1.01e9, 0.01))],
            module=2, min_separation_hz=0.0))

refused("amplitude in dBm, not DAC units",
        lambda: BiasPoint(1.01e9, amplitude=-30.0))

refused("no such resonator", lambda: named_catalog["nope"])
```

## 6. Saving and loading Catalogs

To improve usability and compatibility, rfmux provides helpers to translate
Catalog objects into other standard classes and file formats.

### Dictionaries

`a_dictionary = catalog.to_dict()` reduces a catalog to plain builtins — dicts, strings and floats —
and `new_catalog_from_dict = ResonatorCatalog.from_dict(a_dictionary)` rebuilds it. Being plain builtins, the result will go into a pickle,
a JSON file or HDF5 attributes equally happily.

The `resonators` entry is itself keyed by name, like the catalog, so reading one
detector out of a saved file is `a_dictionary['resonators']['R0007']` and not a
search. `from_dict` takes the same keyword arguments as the other constructors,
so you can load under a rule the file was not written with — say
`ResonatorCatalog.from_dict(a_dictionary, min_separation_hz=100e3)`.

```python
catalog_dict = by_hand_catalog.to_dict()
print(f"schema_version : {catalog_dict['schema_version']}")
print(f"top-level keys : {list(catalog_dict)}")
print(f"resonator names: {list(catalog_dict['resonators'])}")
print(f"one resonator  : {catalog_dict['resonators']['red']}")

restored_catalog = ResonatorCatalog.from_dict(catalog_dict)
print(f"\nround trip: {len(restored_catalog)} resonators, "
      f"names {restored_catalog.names()}")
print(f"calibration survived: "
      f"{restored_catalog['red'].bias.iq_rotation_deg} deg, "
      f"notes {restored_catalog['red'].notes}")
```

<!-- #region -->


### Pickle 

Pickle is currently the main file format used for array tuning outputs. This may evolve in future.


**Pickle the dictionary, not the catalog object.** `pickle.dump(catalog, f)`
does work, BUT it is a trap. The file would then record the class's import path, so
moving or renaming `ResonatorCatalog` later makes every old file unreadable.

Instead, go
through `to_dict` / `from_dict` !

`rfmux.tuning.store` does the file handling: it picks the folder, names the
file, and stamps a `file_metadata` block into what it writes saying what the
file is and where it lives.

Generally, file saving is done automatically by the measurement algorithms, so you
probably won't need to worry about this part.
<!-- #endregion -->

```python
from rfmux.tuning import store

catalog_pkl_path = store.save(by_hand_catalog.to_dict(), "catalog",
                              label="by_hand", directory=OUTPUT_DIR)
print(f"wrote {catalog_pkl_path.name} "
      f"({catalog_pkl_path.stat().st_size} bytes)")

catalog_from_disk = ResonatorCatalog.from_dict(store.load(catalog_pkl_path))
print(catalog_from_disk)
```

`directory=` is here only because this notebook keeps its files together in
`OUTPUT_DIR`. Leave it off and the file goes where every measurement goes:
`~/rfmux_data/ipy_session_<today>/`, the folder `take_netanal` and `multisweep`
write into when they finish, since they call this same `store.save` for you.
`store.output_directory()` says where that is, and `rfmux.tuning.store`'s
docstring covers moving it — for one session, or for good.

Two things came back that you did not put in. The filename gained a date and a
time, so two catalogs saved an hour apart do not collide; and the dictionary
gained a `file_metadata` key recording what the file is and where it lives.
`ResonatorCatalog.from_dict` ignores the extra key, so a saved catalog and a
hand-built one load exactly the same way.

The usual caution applies: unpickling runs code from the file, so load `.pkl`
files you produced or trust, not ones that arrived from somewhere unknown.

### CSV 

**Note that going back and forth from CSV files is deliberately lossy.**
These files will carry the operating point and nothing else:
`notes` and every calibration field are dropped. Use it to
hand someone a bias table they can edit; use `to_dict` when you need everything
back.

```python
bias_table_csv_path = OUTPUT_DIR / "bias_table.csv"
bias_table_csv_path.write_text(by_hand_catalog.to_csv())
print(bias_table_csv_path.read_text())
```

Reading takes the CSV *text* and the module — the module is not in the file,
because channel numbers are meaningless without knowing which module they belong
to. Columns are matched by header name, so they may appear in any order.

```python
reloaded_catalog = ResonatorCatalog.from_csv(
    bias_table_csv_path.read_text(), module=2
)
print(reloaded_catalog)
print(f"\ncalibration after a CSV round trip: "
      f"{reloaded_catalog['red'].bias.iq_rotation_deg}   "
      f"<- dropped, as documented")
```

Hand-editing works. Both columns of an operating point are
required so a blank cell will throw an error naming the
line:

```python
edited_csv_text = "\n".join([
    "name,channel,bias_frequency_hz,bias_amplitude",
    "blue,1,1010000000.0,0.01",
    "green,2,1030000000.0,0.02",      # amplitude changed by hand
    "red,3,1050000000.0,0.005",
])
print(ResonatorCatalog.from_csv(edited_csv_text, module=2))

refused("a missing bias amplitude", lambda: ResonatorCatalog.from_csv(
    "name,channel,bias_frequency_hz,bias_amplitude\nblue,1,1010000000.0,\n",
    module=2))
```

Finally, the catalog the sweep produced, saved both ways:

```python
(OUTPUT_DIR / "found.csv").write_text(catalog.to_csv())
store.save(catalog.to_dict(), "catalog", label="found", directory=OUTPUT_DIR)
print(f"wrote {len(catalog)} resonators to {OUTPUT_DIR}")
for output_path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {output_path.name:<16} {output_path.stat().st_size:>7} bytes")
```

## 7. Where this maps in Periscope

TODO: revisit this once we have updated periscope to use the new code architecture

| Periscope control | API equivalent |
|---|---|
| *Network Analysis* panel, **Take Netanal** | `crs.take_netanal(...)` |
| **Find Resonances** button + its dialog | `find_resonances_in_netanal(...)` |
| Expected / Min Dip Depth / Min Q / Max Q fields | the same-named arguments |
| The red dashed markers on the plot | `ResonanceSearch.candidates` |
| The resonance list the multisweep dialog inherits | `ResonanceSearch.to_catalog(...)` |



