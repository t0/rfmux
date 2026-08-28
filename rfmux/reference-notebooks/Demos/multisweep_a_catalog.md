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

# Multisweep: two ways to say what to sweep

`crs.multisweep()` measures a narrow, high-resolution sweep around each of many
frequencies at once — one hardware channel per frequency, all of them swept in
parallel. It is the step after a network analysis: netanal finds *where* the
resonators are, multisweep looks at each one closely enough to characterise it.

This notebook is about the front door, not the physics. There are two ways to
tell multisweep what to sweep, and they are identical once the measurement
starts:

| You have | You pass | Results keyed by |
|---|---|---|
| A tuned array | a `ResonatorCatalog` | resonator name (`"R0001"`) |
| A list of frequencies | `center_frequencies=` + `amp=` | 1-based index (`1`) |

The catalog form is what the tuning flow uses, because a catalog already knows
each resonator's frequency, its probe amplitude and its hardware channel. The
frequency-list form is for everything else: before resonances have been found,
on a system that has none, or when you simply want to point the instrument at
some frequencies and look.

| Piece | Module |
|---|---|
| The sweep | `rfmux.algorithms.measurement.multisweep` (`crs.multisweep`) |
| The array bookkeeping | `rfmux.core.resonators` |
| Finding resonances first | `rfmux.tuning.find_resonances` |

Seeding a catalog from a network analysis is the subject of
`network_analyses_find_resonances_make_resonator_catalog.md`, and this notebook
assumes it. Sweeping the same array at a *ladder* of amplitudes is a separate
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
  instead: with the jupytext extension, *Sync* creates an `.ipynb` beside this
  file and keeps the two in step — run and edit the notebook, and your changes
  flow back into the markdown. The `.ipynb` is a local working copy and is
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

Compressed to two cells, since it is the previous notebook's whole subject. A
network analysis across the band, the dips located, and the result seeded into
a `ResonatorCatalog`.

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
print(f"{len(found.candidates)} resonances found")
```

`from_frequencies` sorts by frequency, numbers the resonators `R0001…` in that
order, assigns channels `1..N`, and parks every bias point at `PROBE_AMPLITUDE`.
That amplitude is required rather than defaulted: a probe power is a real
measurement choice, and there is no value that is right for an arbitrary array.

```python
catalog = ResonatorCatalog.from_frequencies(
    found.resonance_frequencies_hz,
    module=MODULE,
    amplitude=PROBE_AMPLITUDE,
)
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

**Multisweep does not modify the catalog.** A sweep on its own has not learned
anything yet; the analyses that do — fitting, bias finding — update the catalog
themselves. Everything a sweep produces comes back in the returned dict.

```python
sweeps = await crs.multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
)

print(f"{len(sweeps)} sweeps, keyed by resonator name: {list(sweeps)[:4]} …")
```

Each entry holds the sweep itself plus the bookkeeping needed to know what it
is:

```python
entry = sweeps["R0001"]
for key, value in entry.items():
    if isinstance(value, np.ndarray):
        print(f"{key:<30} ndarray{value.shape} {value.dtype}")
    else:
        print(f"{key:<30} {value!r}")
```

A look at the first four, in the IQ plane and in magnitude:

```python
def plot_sweeps(sweeps, keys, title):
    fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 5.5))
    for column, key in enumerate(keys):
        s = sweeps[key]
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

plot_sweeps(sweeps, list(sweeps)[:4], "swept from a catalog")
```

### Overriding the amplitude

`amp` is optional with a catalog, and defaults to each resonator's own
`bias.amplitude`. Pass a number to override all of them for this one call, or a
`{name: amplitude}` mapping to set them individually. The mapping has to name
every resonator — a half-applied amplitude override is the kind of thing that is
only noticed after the data is taken.

The catalog is left alone either way; the amplitude actually used is reported
per resonator as `sweep_amplitude`.

```python
louder = await crs.multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp=PROBE_AMPLITUDE * 2,
)

print(f"catalog bias amplitude   {catalog['R0001'].bias.amplitude}")
print(f"swept at (default)       {sweeps['R0001']['sweep_amplitude']}")
print(f"swept at (override)      {louder['R0001']['sweep_amplitude']}")
print(f"catalog after the sweep  {catalog['R0001'].bias.amplitude}  ← unchanged")
```

A per-resonator mapping, for the same reason you would ever want one: two
detectors that do not want the same probe power.

```python
per_resonator = {r.name: r.bias.amplitude for r in catalog}
per_resonator["R0001"] = PROBE_AMPLITUDE * 4
per_resonator["R0002"] = PROBE_AMPLITUDE / 2

mixed = await crs.multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp=per_resonator,
)

for name in list(mixed)[:4]:
    print(f"{name}  swept at {mixed[name]['sweep_amplitude']:.5f}")
```

Note that a positional *list* of amplitudes is refused alongside a catalog: it
would silently depend on catalog ordering, which is not something a caller
should have to know. Ask by name, or pass one number for everything.

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

## 4. Sweep a bare list of frequencies

No catalog required. This is the form to reach for when there is nothing tuned
yet — you have a few frequencies from somewhere and you want to look at them.

Two differences from the catalog form, both consequences of the input carrying
less information:

- **`amp` is required.** There is nothing to fall back to.
- **`module` is required**, and results are **keyed by 1-based index** rather
  than by name, in the order you passed the frequencies. That index is also the
  hardware channel each frequency is swept on.

```python
targets = [1.005e9, 1.015e9, 1.025e9]   # nothing in particular is here

blind = await crs.multisweep(
    center_frequencies=targets,
    amp=PROBE_AMPLITUDE,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

print(f"keys: {list(blind)}")
for key, s in blind.items():
    print(f"{key}  ch {s['channel']}  "
          f"{s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']}")
```

Off-resonance, so these are flat — which is itself the point. The instrument
does not need to be pointed at a resonator for the sweep to be valid.

```python
plot_sweeps(blind, list(blind), "swept from a bare frequency list")
```

### One amplitude per frequency

`amp` may also be a list, one value per frequency, in the same order as
`center_frequencies`. Here the pairing is positional because the ordering is
your own — you wrote both lists — which is exactly the thing that is *not* true
of a catalog.

```python
ladder = await crs.multisweep(
    center_frequencies=targets,
    amp=[PROBE_AMPLITUDE, PROBE_AMPLITUDE * 2, PROBE_AMPLITUDE * 4],
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

for key, s in ladder.items():
    print(f"{key}  {s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']:.5f}")
```

A length mismatch is an error rather than a broadcast, so a list that has
drifted out of step with its frequencies is caught before the measurement:

```python
try:
    await crs.multisweep(
        center_frequencies=targets,
        amp=[PROBE_AMPLITUDE, PROBE_AMPLITUDE * 2],   # two, for three frequencies
        span_hz=SPAN_HZ,
        npoints_per_sweep=NPOINTS_PER_SWEEP,
        module=MODULE,
    )
except ValueError as e:
    print(f"ValueError: {e}")
```

### The same resonators, both ways

The two forms are the same measurement. Sweeping the catalog's own frequencies
as a bare list reproduces the catalog run — the results are simply keyed
differently.

```python
as_a_list = await crs.multisweep(
    center_frequencies=[r.bias.frequency_hz for r in catalog],
    amp=PROBE_AMPLITUDE,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

for index, resonator in enumerate(catalog, start=1):
    by_name = sweeps[resonator.name]["original_center_frequency"]
    by_index = as_a_list[index]["original_center_frequency"]
    print(f"{resonator.name} ↔ {index}   "
          f"{by_name/1e6:.6f} MHz  vs  {by_index/1e6:.6f} MHz   "
          f"{'✓' if by_name == by_index else '✗'}")
```

The catalog form is worth the extra step as soon as identity starts to matter.
`R0003` is the same detector next week, and it carries its own amplitude,
channel and calibration with it; index `3` is only ever "the third thing in the
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
