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

# Network analysis → find resonances

This notebook starts array tuning with a broad frequency sweep (a network
analysis, or “netanal”) and a search for resonance dips. It ends with a
`ResonatorCatalog` that records their names, channels, and initial bias points.

| Task | API |
|---|---|
| Measure a network analysis | `crs.take_netanal()` |
| Search frequency and IQ arrays | `rfmux.tuning.find_resonances()` |
| Search one module’s netanal and store the result | `rfmux.tuning.find_resonances_in_netanal()` |
| Manage the catalog | `rfmux.core.resonators` |

`find_resonances_in_netanal()` takes one module’s output and stores the search
beside its trace. This keeps the measurement and search in the same file.
Call it separately for each module so you can choose suitable depth and Q limits.

See `resonator_catalogs.md` for working with the catalog after this step.

## How to use this document

This is a runnable Jupytext notebook. Select a code cell and press **Shift+Enter**.

- Run cells from top to bottom. Later cells use variables defined earlier.
  Use *Kernel → Restart Kernel and Run All Cells* to start again.
- The markdown file stores no outputs. Run a cell to see its results.
- Feel free to change the band, dip-depth threshold, and Q limits and rerun the cells to explore them. The shipped
  copy is read-only; use *File → Save Notebook As…* to keep your changes.
- In Periscope's JupyterLab, double-click this file. In another JupyterLab
  session, use *Open With → Notebook*.
- VS Code opens this file as text. With a Jupytext extension, use *Open Paired
  Notebook* (the command name may vary). If pairing fails, check that the
  extension's Python environment has Jupytext installed. You can also run
  `jupytext --sync <this file>.md` in an environment with Jupytext. The paired
  `.ipynb` is a local, gitignored copy; the markdown is kept in version control.

The kernel must use the environment where this checkout of rfmux is installed.
Check the interpreter and package paths:

```python
import sys
import rfmux

print(sys.executable)
print(rfmux.__file__)
```

```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from rfmux.tuning import netanal_trace, store

MODULE = 1
```

## 1. Simulate a board

Let’s generate ten simulated LEKIDs with seed 42. The seed fixes the array;
measurement noise can still vary between runs.

For real hardware, replace the next cell with your board session:

    session = rfmux.load_session('!HardwareMap [ !CRS { serial: "0042" } ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()

A network analysis overwrites channel frequencies and amplitudes on the module
it sweeps. Use a module available for this measurement.

```python
from rfmux.mock.helpers import create_mock_crs

MOCK_CONFIG = {
    "num_resonances": 10,
    "freq_start": 0.6e9,          # inside the 0.6–1.05 GHz sweep band
    "freq_end": 1.0e9,
    "resonator_random_seed": 42,  # same array every run
    "auto_bias_kids": False,      # start without bias tones
    "pulse_mode": "none",
    "tls_noise_enabled": False,
    "nqp_noise_std_factor": 0.001,
    "T" : 0.23 # K
}

crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG, verbose=False)
```

## 2. Run the network analysis

`crs.take_netanal()` measures complex S21 across a band. Choose enough points
to sample the resonance dips. Here, 40,000 points across 450 MHz give about
11.25 kHz spacing.

```python
netanal = await crs.take_netanal(
    amp=0.001,
    fmin=0.6e9,
    fmax=1.05e9,
    npoints=40_000,
    nsamps=10,          # averages per point
    max_chans=1023,     # frequencies measured simultaneously
    module=MODULE,
)

# Select one module, then access its trace (the results dictionary).
module_id = crs.module[MODULE].index()
module_netanal_outputs = netanal[module_id]
netanal_measured = netanal_trace(module_netanal_outputs)
netanal_frequencies = netanal_measured["frequencies"]
netanal_iq_counts = netanal_measured["iq_counts"]

print(f"{len(netanal_frequencies):,} points, "
      f"{np.mean(np.diff(netanal_frequencies))/1e3:.2f} kHz spacing")
```

The returned dictionary and saved `.pkl` use the same structure, even when
only one module is measured:

```text
netanal[module_id]                  # e.g. "crs0042_rmod1"
    schema_version                 # measurement schema version (7)
    measurement                    # "netanal"
    module                         # numeric module
    call_params                    # arguments used for the measurement
    results
        frequencies                # Hz
        iq_counts                  # complex readout counts
        iq_volts                   # complex volts at the board input
        sweep_amplitude            # normalized amplitude per tone
        sweep_direction            # "upward" or "downward"
        resonance_search           # added by the resonance finder
    file_metadata                  # added when saved
```

`netanal_trace(module_netanal_outputs)` returns that module's `results`
dictionary. There are no iteration, direction, or resonator keys between
`results` and the arrays. Derive phase from `np.angle(iq_counts)`; it is not
stored separately. Select a module before calling the accessor or finder.

`sweep_direction` decides which end of the band the measurement starts at, and
defaults to `"upward"`. A downward netanal visits the same points and comes back
descending — the order it measured them in — so `frequencies[0]` is the top of
the band. One direction per call: `find_resonances_in_netanal()` reads either,
and to compare the two you call `take_netanal()` twice and keep both results.
The search stores frequencies in ascending order; candidate indices refer to
`resonance_search.frequencies_hz` and `magnitude_db`, including for a downward
sweep.

Plot magnitude and phase across the band. Magnitude is normalized by its median
to make the dips easier to compare. These cells show the plotting steps directly;
reusable versions are in `example_plotting_netanal.py`.

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
phase_panel.set_ylabel("phase [deg]")
phase_panel.set_xlabel("frequency [MHz]")

magnitude_panel.set_title("network analysis")
plt.tight_layout()
plt.show()
```

## 3. Find the resonances

The finder converts magnitude to dB, inverts the dips, and uses
`scipy.signal.find_peaks` to locate them.

- `min_dip_depth_db` sets the minimum prominence relative to the local baseline.
  Lower it to include shallower dips; raise it to reject smaller features.
- `min_Q` and `max_Q` set the accepted width range. `min_Q` limits the widest
  dips; `max_Q` limits the narrowest, helping reject single-sample spikes.
- `min_separation_hz` rejects both members of a pair this close or closer.
  The default, `0.0`, only rejects coincident frequencies. Use a positive value
  to reject nearby candidates, or `None` to disable this check.

```python
from rfmux.tuning import find_resonances_in_netanal

resonance_search = find_resonances_in_netanal(
    module_netanal_outputs,
    min_dip_depth_db=1.0,
    min_Q=1e4,
    max_Q=1e7,
)

```

The call stores a plain search dictionary at
`module_netanal_outputs["results"]["resonance_search"]`.
With autosave enabled, it also updates the measurement file.

Use `ResonanceSearch.from_dict()` to rebuild the search object. To load a saved
measurement without a board, select its module identifier from the file:

```python
from rfmux.tuning import ResonanceSearch

# For a saved file, run these lines with your path and module identifier:
# netanal = store.load("path/to/netanal.pkl")
# print(list(netanal))
# module_netanal_outputs = netanal["crs0042_rmod1"]
# netanal_measured = netanal_trace(module_netanal_outputs)
# Run the finder first if the file has no resonance_search yet.
stored_search = ResonanceSearch.from_dict(netanal_measured["resonance_search"])
```

The search contains accepted candidates, rejected candidates and their reasons,
the processed trace, and the settings used. Mark the results on that trace:



`q_estimate` is frequency divided by dip width. It is a screening estimate,
not a fitted resonator Q. Use detailed multisweeps and fits to measure Q;
see `fitting_resonators.md`.

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
for index, candidate in enumerate(resonance_search.rejected):
    plt.axvline(candidate.frequency_hz / 1e6, color="darkorange", ls="--",
                alpha=0.6, label="rejected" if index == 0 else None)
    # Rejection reasons are useful when something was actually removed.
    print(f"Rejected {candidate.frequency_hz/1e6:.4f} MHz: {candidate.rejected_because}")
plt.xlabel("frequency [MHz]")
plt.ylabel("|S21| [dB, normalized]")
plt.title(f"{len(resonance_search)} accepted, {len(resonance_search.rejected)} rejected")
plt.legend()
plt.tight_layout()
plt.show()
```

### Inspect individual candidates

Zoom in to see which samples define each dip. The vertical red bar shows depth
(prominence); the horizontal bar shows width at half that depth.
This is a useful first check when the number of resonances is unexpected.

```python
def plot_candidate_details(search: ResonanceSearch, ncols: int = 5,
                           span_widths: float = 4.0) -> None:
    """One panel per candidate, with the measured depth and width drawn on."""
    point_spacing_hz = float(np.mean(np.diff(search.frequencies_hz)))
    shown_candidates = search.candidates
    if not shown_candidates:
        print("No accepted candidates. Inspect the overview and search settings.")
        return
    nrows = int(np.ceil(len(shown_candidates) / ncols))

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(3.1 * ncols, 2.7 * nrows))
    for panel in axes.flat:
        panel.axis("off")       # panels with no candidate stay blank

    for panel, candidate in zip(axes.flat, shown_candidates):
        panel.axis("on")

        # Include a few widths and at least eight samples on either side.
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
    fig.suptitle(f"{len(shown_candidates)} candidates")
    fig.tight_layout()
    plt.show()


plot_candidate_details(resonance_search)
```

Look for several samples across each dip. Real arrays may also show shallow
dips, baseline ripples, or unresolved neighbours. If a dip has only one or two
samples, a finer sweep can help.

### Compare sweep resolution

Measure the same band with 5,000 points instead of 40,000: about 90 kHz spacing
instead of 11.25 kHz. Keep the search thresholds unchanged so we can compare the
effect of sampling alone.

The first panel compares detection counts. The close-ups show the samples around
the first two candidates found in the fine sweep.

```python
from rfmux.tuning import find_resonances

coarse_netanal = await crs.take_netanal(
    amp=0.001,
    fmin=0.6e9,
    fmax=1.05e9,
    npoints=5_000,
    nsamps=10,
    max_chans=1023,
    module=MODULE,
)
coarse_trace = netanal_trace(coarse_netanal[crs.module[MODULE].index()])
coarse_search = find_resonances(
    coarse_trace["frequencies"], coarse_trace["iq_counts"],
    min_dip_depth_db=1.0, min_Q=1e4, max_Q=1e7,
)

# Use the existing fine search; both searches used identical thresholds.
searches = [("coarse", coarse_search), ("fine", resonance_search)]
closeups = resonance_search.candidates[:2]
fig, axes = plt.subplots(1, 1 + len(closeups),
                         figsize=(4 * (1 + len(closeups)), 3.8),
                         constrained_layout=True, squeeze=False)
count_panel = axes[0, 0]
bars = count_panel.bar([label for label, search in searches],
                       [len(search) for label, search in searches],
                       color=["tab:orange", "tab:blue"])
count_panel.bar_label(bars, padding=3)
count_panel.set_ylim(0, max(1, len(coarse_search), len(resonance_search)) * 1.2)
count_panel.set_ylabel("accepted resonances")
count_panel.set_title("Same thresholds, different sampling")

for panel, candidate in zip(axes[0, 1:], closeups):
    # A 200 kHz window includes nearby coarse samples as well as the fine dip.
    for (label, search), colour in zip(searches, ["tab:orange", "tab:blue"]):
        offset_khz = (search.frequencies_hz - candidate.frequency_hz) / 1e3
        nearby = np.abs(offset_khz) <= 200
        spacing_khz = np.mean(np.diff(search.frequencies_hz)) / 1e3
        panel.plot(offset_khz[nearby], search.magnitude_db[nearby], ".-",
                   color=colour, lw=0.8, ms=5,
                   label=f"{label}: {spacing_khz:.1f} kHz spacing")
    panel.axvline(0, color="0.7", lw=0.7)
    panel.set_title(f"{candidate.frequency_hz/1e6:.3f} MHz")
    panel.set_xlabel("offset from candidate [kHz]")
    panel.set_ylabel("|S21| [dB, normalized]")
    panel.legend(fontsize=8)
plt.show()
```

A narrow dip can fall between coarse samples and be missed entirely. If the
count is low, inspect sampling before changing the thresholds. More points, or
a narrower band with the same number of points, improves frequency resolution.

### Check for unresolved neighbours

The separation cut only rejects pairs that the survey resolves as distinct dips.
Detailed multisweeps can reveal closer neighbours. See
`rfmux.tuning.find_sweeps_with_nearby_resonances` for checking those sweeps.

## 4. Build a resonator catalog

`ResonanceSearch.to_catalog()` assigns each candidate a name, a channel, and an
initial `BiasPoint`. Channels are assigned 1..N in frequency order.

Read the required module and amplitude from the measurement. Later sweeps and
bias finding refine these initial operating points.

```python
catalog = resonance_search.to_catalog(
    module=module_netanal_outputs["module"],
    amplitude=netanal_measured["sweep_amplitude"],
)
print(catalog)

# The file contains the measurement and, when autosaved, its resonance search.
print(f"Measurement file: {store.saved_path(module_netanal_outputs)}")
```

The catalog is ready for multisweeps, fitting, and bias finding. Continue with
`multisweep.md`, or explore the catalog in `resonator_catalogs.md`.

## 5. Periscope controls

The GUI wiring is being updated; this table maps the controls to the workflow
shown here, rather than documenting the current internal implementation.

| Periscope control | Workflow API |
|---|---|
| **Take Netanal** | `crs.take_netanal(...)` |
| **Find Resonances** | `find_resonances_in_netanal(...)` |
| Expected count, minimum depth, and Q limits | `expected_resonances`, `min_dip_depth_db`, `min_Q`, `max_Q` |
| Resonance markers | `ResonanceSearch.candidates` |
| Catalog for multisweep | `ResonanceSearch.to_catalog(...)` |

