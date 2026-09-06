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
    display_name: rfmux-tuning
    language: python
    name: python3
---

<!-- #region -->
# Network analysis → find resonances

This notebook works through the first two steps of characterizing and tuning an
array: doing a frequency sweep (S21 vs frequency, "network analysis"-->"netanal"),
and using it to find resonances. It ends by seeding a resonator catalog, which is
a bookkeeping object that rfmux provides to keep track of your resonators and
their properties, and which every later tuning step takes in and hands back.


| Piece | Module |
|---|---|
| The sweep | `rfmux.algorithms.measurement.take_netanal` (`crs.take_netanal`) |
| The resonance finder | `rfmux.tuning.find_resonances` |
| The convenience function to run the resonance finder on one module's netanal | `rfmux.tuning.find_resonances_in_netanal` |
| The array bookkeeping | `rfmux.core.resonators` |

The finder is split into two parts: `find_resonances` takes two plain arrays, while
`find_resonances_in_netanal` is a thin wrapper that unpacks one module's output
out of what `crs.take_netanal()` returned and calls the finder. The wrapper also
puts the search back into that output, beside the trace it searched, so the
search is saved as part of the netanal rather than as a second file to keep
paired with it.

Like every other analysis in `rfmux.tuning`, the wrapper takes **one module at a
time** — `netanal[module_id]`, not the whole netanal. How deep a dip has to be
and how wide it may get depend on the band a module looks at and the resonators
in it, so eight modules are eight decisions, and writing eight calls (or a loop
you can see) is what keeps them from being made by accident.

The catalog itself — building one by hand, reading and amending it, the
invariants it enforces, and the file formats it round-trips through — is the
subject of `resonator_catalogs.md`. That notebook picks up from the netanal file
this one leaves on disk, so the two run back to back.



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

<!-- #endregion -->

```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import rfmux
from rfmux.tuning import store

MODULE = 1

# The band to sweep. The simulated array in section 1 is placed inside it.
FMIN, FMAX = 0.6e9, 1.05e9
PROBE_AMPLITUDE = 0.001   # normalized DAC units, shared by the sweep and the
                          # catalog's bias points

# Where the measurements below save themselves. Nothing in this notebook picks
# a directory: `take_netanal` calls `store.save` on its way out, and this is
# the folder it writes into — one per day, inside `store.output_directory()`.
print(f"measurements → {store.session_directory()}")
```

## 1. Simulate a board

We will generate 10 simulated LEKIDs spread across the band using a fixed random seed in rfmux's
mock mode, so this
notebook will produce the same array and the same numbers every time it is run.

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

# take_netanal returns a dict keyed by module — one entry per module swept —
# and each module's outputs contain the measured data, as well as a record of 
# how the measurement was called.
# The data is sorted by amplitude iteration index, and then by sweep direction
module_netanal_outputs = netanal[crs.module[MODULE].index()]
netanal_measured = module_netanal_outputs["results"][0]["upward"]
netanal_frequencies = netanal_measured["frequencies"]
netanal_iq_counts = netanal_measured["iq_counts"]

print(f"modules: {list(netanal)}")
print(f"called with: {module_netanal_outputs['call_params']}")
print(f"measured: {list(netanal_measured)}")
print(f"{len(netanal_frequencies)} points, "
      f"{netanal_frequencies[0]/1e6:.1f}–{netanal_frequencies[-1]/1e6:.1f} MHz, "
      f"{np.mean(np.diff(netanal_frequencies))/1e3:.2f} kHz spacing")
print(f"saved to: {store.saved_path(netanal)}")
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
    np.abs(netanal_iq_counts) / np.median(np.abs(netanal_iq_counts))
)
magnitude_panel.plot(netanal_frequencies / 1e6, magnitude_db, lw=0.6)
magnitude_panel.set_ylabel("|S21| [dB, normalized]")

phase_panel.plot(netanal_frequencies / 1e6,
                 np.degrees(np.angle(netanal_iq_counts)), lw=0.6)
phase_panel.set_ylabel("phase [deg]"); phase_panel.set_xlabel("frequency [MHz]")

magnitude_panel.set_title("network analysis")
plt.tight_layout(); plt.show()
```

## 3. Find the resonances

`rfmux.tuning.find_resonances_in_netanal` unpacks one module's sweep and
searches it.
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
**all** removed. 

```python
from rfmux.tuning import find_resonances_in_netanal

resonance_search = find_resonances_in_netanal(
    module_netanal_outputs,
    min_dip_depth_db=1.0,
    min_Q=1e4,
    max_Q=1e7,
)
print(resonance_search)
```

The search also went *into* `module_netanal_outputs`, beside the trace it
searched, under `resonance_search`.

The netanal file was rewritten in place as part of that call, so the file
`take_netanal` announced in section 2 now holds the trace *and* the search. 

You can retrieve the resonance search from the netanal dictionary and turn it back 
into a `ResonanceSearch` class using the `from_dict` method, the same as for any
class rfmux stores in a file:

```python
print(module_netanal_outputs.keys())
print(module_netanal_outputs['results'][0]['upward'].keys())
```

```python

```

```python
from rfmux.tuning import ResonanceSearch

# or, for example, if you have loaded your netanal from a file:
#   netanal = store.load(".../netanal_20260904_142231.pkl")
#   netanal_measured = netanal[crs.module[MODULE].index()]["results"][0]["upward"]
stored_search = ResonanceSearch.from_dict(netanal_measured["resonance_search"])
print(stored_search)
print(f"the file holding both: {store.saved_path(netanal)}")
```

The result is a `rfmux.tuning.ResonanceSearch`. It carries the accepted
candidates, everything a rejection pass threw out and why, the processed trace
that was searched, and the settings used:

```python
for candidate in resonance_search.candidates:
    print(f"  {candidate.frequency_hz/1e6:11.4f} MHz   "
          f"depth {candidate.depth_db:5.2f} dB   "
          f"width {candidate.width_hz/1e3:6.2f} kHz   "
          f"Q estimate ≈ {candidate.q_estimate:.3g}")

print(f"\n{len(resonance_search)} accepted, "
      f"{len(resonance_search.rejected)} rejected")
for candidate in resonance_search.rejected:
    print(f"  {candidate.frequency_hz/1e6:11.4f} MHz — "
          f"{candidate.rejected_because}")

print(f"\nthe simulated array has {MOCK_CONFIG['num_resonances']}")
assert len(resonance_search) > 0, "found nothing — check the sweep in section 2"
```

Note that `q_estimate` is `frequency / width` — the rough figure the `min_Q` / `max_Q`
window screens on. **It is not a measurement of the resonators' qualtiy factors.**
Determining a resonator's
real Q factors is done using multisweep type data -- we will talk about that in a separate walkthrough notebook.

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
sweep caught only a point or two. When that happens, these plots can be useful
for diagnosis. More `npoints` is usually the answer.

### Sweep resolution decides what you can find

A survey sweep is usually much coarser than the resonators in it, and the failure
mode is quiet: the finder simply returns fewer resonances than might truly exist. 
Compare the same array swept at a quarter the resolution:

```python
from rfmux.tuning import find_resonances

coarse_netanal = await crs.take_netanal(
    amp=PROBE_AMPLITUDE, fmin=FMIN, fmax=FMAX, npoints=5_000,
    nsamps=10, max_chans=1023, module=MODULE)

for label, netanal_to_search in (("coarse", coarse_netanal), ("fine", netanal)):
    measured = netanal_to_search[crs.module[MODULE].index()]["results"][0]["upward"]
    frequencies = measured["frequencies"]
    n_resonances = len(find_resonances(
        frequencies, measured["iq_counts"],
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

### Remaining collisions - see multisweep data

`min_separation_hz` above can only cut pairs the survey sweep managed to
distinguish as separate resonators. If two resonators are quite close together,
they may remain. Multisweep data makes these more apparent, but this is the topic 
of a separate workbook. However, we note here that rfmux provides 
`rfmux.tuning.find_sweeps_with_nearby_resonances` to prune collided resonances
using multisweep data. 

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

That object is what the rest of tuning consumes and returns. From here you would
run iterative multisweeps at various amplitudes around each bias frequency, pick bias points,
apply fits, etc — see
`multisweep.md` for more info on these.

The catalog on its own is described in more detail in
`resonator_catalogs.md`. It starts from a saved netanal exactly like the one this
notebook just wrote, so you can carry on there directly.

## 5. Where this maps in Periscope

TODO: revisit this once we have updated periscope to use the new code architecture

| Periscope control | API equivalent |
|---|---|
| *Network Analysis* panel, **Take Netanal** | `crs.take_netanal(...)` |
| **Find Resonances** button + its dialog | `find_resonances_in_netanal(...)` |
| Expected / Min Dip Depth / Min Q / Max Q fields | the same-named arguments |
| The red dashed markers on the plot | `ResonanceSearch.candidates` |
| The resonance list the multisweep dialog inherits | `ResonanceSearch.to_catalog(...)` |



