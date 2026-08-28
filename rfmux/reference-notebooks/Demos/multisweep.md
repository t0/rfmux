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

# Multisweep: two ways to say what to sweep

`crs.multisweep()` measures a narrow, high-resolution sweep around each of many
frequencies at once — one hardware channel per frequency, all of them swept in
parallel. In a typical array characterization / tuning flow, it is the step after 
a network analysis: netanal finds roughly where the
resonators are, multisweep looks at each one closely enough to characterise it.

There are two ways to
tell multisweep what to sweep, and they are identical once the measurement
starts:

| You have | You pass | Results keyed by |
|---|---|---|
| Already done a netanal and used `find_resonances` to make a `ResonatorCatalog` | a `ResonatorCatalog` | resonator name (`"R0001"`) |
| A list of frequencies | `center_frequencies=` + `amp=` | section name (`"S0001"`) |

Unless you are starting completely from scratch, or looking at hardware that doesn't
have resonances, you will likely pass a catalog. A catalog already knows
each resonator's frequency, its probe amplitude and its hardware channel. 

| Piece | Module |
|---|---|
| The sweep | `rfmux.algorithms.measurement.multisweep` (`crs.multisweep`) |
| The array bookkeeping | `rfmux.core.resonators` |
| Finding resonances first | `rfmux.tuning.find_resonances` |

Seeding a catalog from a network analysis is the subject of
`network_analyses_find_resonances_make_resonator_catalog.md`. If you haven't done
that yet and are unfamiliar with the workflow, start there, then come back here.
Sweeping the same array at a *ladder* of amplitudes is a separate
layer that does not exist yet; this notebook will grow as it lands.

## How to use this document

**This is a runnable notebook, not a web page.** Every grey block below is a live
code cell: put the cursor in it and press **Shift+Enter** to execute it.

- **Run the cells in order, top to bottom.** Later cells use variables the
  earlier ones defined, so skipping ahead fails with a `NameError`. *Kernel →
  Restart Kernel and Run All Cells* starts clean.
- **The outputs you see are the ones you just produced.** This file is stored as
  jupytext markdown, which keeps no saved outputs, so a cell is blank until you
  run it. Nothing here can show you a stale number from someone else's run.
- **Editing is encouraged.** Change the span, the number of points, the
  amplitudes, and re-run — that is what this document is for. The shipped copy
  is read-only, so *File → Save Notebook As…* to keep your changes.
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
  be the environment installed against *this* one. This says which copy you
  actually got:

  ```python
  import sys, rfmux; print(sys.executable); print(rfmux.__file__)
  ```

```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import rfmux
from rfmux.tuning import find_resonances_in_netanal
from rfmux.core.resonators import ResonatorCatalog

MODULE = 1

# The band the simulated array lives in.
FMIN, FMAX = 0.6e9, 1.05e9

PROBE_AMPLITUDE = 0.001   # normalized DAC units

# One sweep's worth of settings, shared by every call below so the comparisons
# are like for like.
SPAN_HZ = 200e3           # total width of each individual sweep
NPOINTS_PER_SWEEP = 101   # points measured across that width
NSAMPS = 10               # averages per point
```

## 1. Simulate a board

Ten simulated LEKIDs on a fixed random seed, so this notebook produces the same
array and the same numbers every time it runs.

To run against real hardware instead, replace this one cell with a session on
your board — everything after it is unchanged:

    session = rfmux.load_session('!HardwareMap [ !CRS { serial: "0042" } ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()

Note that multisweep overwrites every channel's frequency and amplitude on the
module it sweeps, and zeroes them all when it finishes, so do not point it at a
module someone else is using.

```python
MOCK_CONFIG = {
    "num_resonances": 10,
    "freq_start": 0.6e9,          # inside [FMIN, FMAX] so the sweep can see them
    "freq_end": 1.0e9,
    "resonator_random_seed": 42,  # same array every run
    "auto_bias_kids": False,      # nothing is tuned yet — that is the point
}

from rfmux.mock.helpers import create_mock_crs

crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG, verbose=False)
print(f"simulated CRS with {MOCK_CONFIG['num_resonances']} resonators "
      f"between {MOCK_CONFIG['freq_start']/1e9:.2f} and "
      f"{MOCK_CONFIG['freq_end']/1e9:.2f} GHz")
```

## 2. Find the resonances, and build a catalog

The previous notebook's whole subject, in one cell: a network analysis across
the band, the dips located in it, and the result seeded into a
`ResonatorCatalog`.

`from_frequencies` sorts by frequency, numbers the resonators `R0001…` in that
order, assigns channels `1..N`, and parks every bias point at `PROBE_AMPLITUDE`.


```python
netanal = await crs.take_netanal(
    amp=PROBE_AMPLITUDE,
    fmin=FMIN,
    fmax=FMAX,
    npoints=60_000,
    nsamps=NSAMPS,
    max_chans=1023,
    module=MODULE,
)

found = find_resonances_in_netanal(netanal, min_dip_depth_db=1.0)

catalog = ResonatorCatalog.from_frequencies(
    found.resonance_frequencies_hz,
    module=MODULE,
    amplitude=PROBE_AMPLITUDE,
)

print(f"{len(found.candidates)} resonances found")
print(catalog)
```

## 3. Sweep the catalog

The catalog carries everything multisweep needs, so the call says almost nothing
beyond how finely to look:

- each resonator's **sweep centre** is its `bias.frequency_hz`
- each resonator's **probe amplitude** is its `bias.amplitude`
- each resonator's **hardware channel** is its `channel`

The catalog even knows its own module, so `module=` is optional here — pass it
only if you want the mismatch checked.

**Multisweep does not modify the catalog.** 

Everything a multisweep produces comes back in the returned dict.

```python
ms = await crs.multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
)

print(f"{len(ms)} sweep sections, keyed by resonator name: {list(ms)[:4]} …")
```

Each entry holds the sweep section itself plus the bookkeeping needed to know what it
is:

```python
entry = ms["R0001"]
for key, value in entry.items():
    if isinstance(value, np.ndarray):
        print(f"{key:<30} ndarray{value.shape} {value.dtype}")
    else:
        print(f"{key:<30} {value!r}")
```

A look at the first four, in the IQ plane and in magnitude:

```python
def plot_ms(ms, keys, title):
    fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 5.5))
    for column, key in enumerate(keys):
        s = ms[key]
        centre = s["original_center_frequency"]
        offset_khz = (s["frequencies"] - centre) / 1e3

        axes[0, column].plot(s["iq_complex"].real, s["iq_complex"].imag, lw=0.9)
        axes[0, column].set_aspect("equal", "datalim")
        axes[0, column].set_title(f"{key}\n{centre/1e6:.3f} MHz", fontsize=9)

        axes[1, column].plot(offset_khz, 20 * np.log10(np.abs(s["iq_complex"])), lw=0.9)
        axes[1, column].set_xlabel("offset [kHz]", fontsize=8)

    axes[0, 0].set_ylabel("Q")
    axes[1, 0].set_ylabel("|S21| [dB]")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_ms(ms, list(ms)[:4], "multisweep done using a catalog")
```

### Overriding the amplitude

By default, multisweep uses each resonator's bias amplitude found in the catalog.
You can also override these using the `amp` argument. Pass a number to override all
 of them for this one call, or a
`{name: amplitude}` mapping to set them individually. Note that the mapping has to name
every resonator.

The catalog is left alone either way; the amplitude actually used is reported
per resonator as `sweep_amplitude`.

```python
ms_louder = await crs.multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp=PROBE_AMPLITUDE * 2,
)

print(f"catalog bias amplitude   {catalog['R0001'].bias.amplitude}")
print(f"swept at (default)       {ms['R0001']['sweep_amplitude']}")
print(f"swept at (override)      {ms_louder['R0001']['sweep_amplitude']}")
print(f"catalog after the sweep  {catalog['R0001'].bias.amplitude}  ← unchanged")
```

Or, using a per-resonator amplitude mapping:

```python
per_resonator_amplitude_mapping = {r.name: r.bias.amplitude for r in catalog}
per_resonator_amplitude_mapping["R0001"] = PROBE_AMPLITUDE * 4
per_resonator_amplitude_mapping["R0002"] = PROBE_AMPLITUDE / 2

mixed_amplitude_ms = await crs.multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp=per_resonator_amplitude_mapping,
)

for name in list(mixed_amplitude_ms)[:4]:
    print(f"{name}  swept at {mixed_amplitude_ms[name]['sweep_amplitude']:.5f}")
```

Note that a positional *list* of amplitudes is refused if provided alongside a catalog: it
would silently depend on the catalog ordering, which is not meaningful since the resonators
are in arbitrary order within the catalog.

```python
try:
    await crs.multisweep(
        catalog,
        span_hz=SPAN_HZ,
        npoints_per_sweep=NPOINTS_PER_SWEEP,
        amp=[PROBE_AMPLITUDE] * len(catalog),
    )
except TypeError as e:
    print(f"TypeError: {e}")
```

## 4. No catalog? Multisweep using a plain list of frequencies

No catalog required. This is the form to reach for when there is nothing tuned
yet — you have a few frequencies from somewhere and you want to look at them.

Two differences from the catalog form, both consequences of the input carrying
less information:

- **`amp` is required.** There is nothing to fall back to.
- **`module` is required.** There is no catalog to read it from.
- **The sweeps are named `S0001…`** — S for section — in the order you passed
  the frequencies. Deliberately not `R0001…`, so a result dict tells you at a
  glance whether it came from a catalog or a frequency list. The position in
  the list is also the hardware channel each frequency is swept on.

```python
section_center_frequencies = [1.005e9, 1.015e9, 1.025e9]   # nothing in particular is here

no_catalog_ms = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    amp=PROBE_AMPLITUDE,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

print(f"keys: {list(no_catalog_ms)}")
for section_name, s in no_catalog_ms.items():
    print(f"{section_name}  ch {s['channel']}  "
          f"{s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']}")
```

Off-resonance, so these are flat — which is itself the point. The instrument
does not need to be pointed at a resonator for the sweep to be valid.

```python
plot_ms(no_catalog_ms, list(no_catalog_ms), "multisweep done using a plain frequency list")
```

### One amplitude per frequency

`amp` may also be a list, one value per frequency, in the same order as
`center_frequencies`. Here the pairing is positional because the ordering is
your own — you wrote both lists — which is exactly the thing that is *not* true
of a catalog.

```python
per_section_amplitude_ms = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    amp=[PROBE_AMPLITUDE, PROBE_AMPLITUDE * 2, PROBE_AMPLITUDE * 4],
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

for section_name, s in per_section_amplitude_ms.items():
    print(f"{section_name}  {s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']:.5f}")
```

A length mismatch is an error rather than a broadcast, so a list that has
drifted out of step with its frequencies is caught before the measurement:

```python
try:
    await crs.multisweep(
        center_frequencies=section_center_frequencies,
        amp=[PROBE_AMPLITUDE, PROBE_AMPLITUDE * 2],   # two, for three frequencies
        span_hz=SPAN_HZ,
        npoints_per_sweep=NPOINTS_PER_SWEEP,
        module=MODULE,
    )
except ValueError as e:
    print(f"ValueError: {e}")
```

### Naming the sections yourself

`S0001…` is only the default. Pass `names`, one per frequency in the same
order, when the sections mean something to you — and then you can address their
amplitudes by name too, exactly as you would a catalog's resonators.

```python
section_names = ["below_band", "in_band", "above_band"]

named_section_ms = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    names=section_names,
    amp={"below_band": PROBE_AMPLITUDE, "in_band": PROBE_AMPLITUDE * 2,
         "above_band": PROBE_AMPLITUDE},
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

for section_name, s in named_section_ms.items():
    print(f"{section_name:<12} ch {s['channel']}  "
          f"{s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']:.5f}")
```

### The same resonators, both ways

The two forms are the same measurement. Sweeping the catalog's own frequencies
as a bare list reproduces the catalog run — the results are simply keyed
differently.

```python
catalog_frequencies_as_a_list_ms = await crs.multisweep(
    center_frequencies=[r.bias.frequency_hz for r in catalog],
    amp=PROBE_AMPLITUDE,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

for index, resonator in enumerate(catalog, start=1):
    section_name = f"S{index:04d}"
    from_catalog = ms[resonator.name]["original_center_frequency"]
    from_list = catalog_frequencies_as_a_list_ms[section_name]["original_center_frequency"]
    print(f"{resonator.name} ↔ {section_name}   "
          f"{from_catalog/1e6:.6f} MHz  vs  {from_list/1e6:.6f} MHz   "
          f"{'✓' if from_catalog == from_list else '✗'}")
```

The catalog form is worth the extra step as soon as identity starts to matter.
`R0003` is the same detector next week, and it carries its own amplitude,
channel and calibration with it; `S0003` is only ever "the third thing in the
list I happened to pass".

## 5. What is not here yet

- **A ladder of amplitudes.** One `multisweep` call is one sweep at one
  amplitude per resonator. Walking a range of amplitudes — to find where each
  detector bifurcates, and to pick an operating point below it — is a separate
  layer being built on top of this one. See
  `tuning_multisweep_amplitudes_plan.md` in the repository root.
- **Fitting, and bias finding.** Both consume multisweep output, and both are
  what will write results back into the catalog.

The module is already quiet: multisweep zeroes every channel on its way out, so
there is nothing to clean up after it.
