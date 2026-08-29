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

One `multisweep` call is always *one* sweep, at one amplitude per resonator, in
one direction. Sweeping the same array at a **ladder** of amplitudes is a layer
on top, `crs.multiamp_multisweep()`, and it is the subject of sections 5 to 8.

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

Note that multisweep takes over the channels it sweeps — one per resonator,
overwriting their frequency and amplitude, and zeroing them again when it
finishes. Other channels on the module are left exactly as they were, so a tone
you have parked by hand survives the call. The flip side is that multisweep does
not guarantee a quiet module: if something else is live and would intermodulate
with the sweep, clear it first with `await crs.clear_channels(module=MODULE)`.

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

## 5. A ladder of amplitudes

How hard you drive a KID changes what you measure. Too quiet and the resonance
is buried in noise; too loud and it bifurcates, the dip goes asymmetric and
snaps. Choosing an operating point means sweeping the *same* array several times
at different amplitudes and comparing.

`crs.multiamp_multisweep()` does exactly that: one `multisweep` per amplitude
step, per direction, all returned together. It decides nothing on its own — the
amplitudes come from an `AmplitudeSchedule`, which is plain Python you can build,
print and check with no board in sight.

| Piece | Module |
|---|---|
| The ladder driver | `rfmux.algorithms.measurement.multiamp_multisweep` (`crs.multiamp_multisweep`) |
| The amplitudes, and reading the results | `rfmux.tuning.multisweep_amplitudes` |

A schedule is two things: a **base** — what each resonator would be swept at
with no ladder at all — and a **ladder** of rungs applied to it.

| You want | You write | Rungs are |
|---|---|---|
| The catalog as it stands, once | `AmplitudeSchedule()` | — |
| One amplitude for everything | `AmplitudeSchedule.fixed(0.005)` | — |
| Absolute amplitudes, same for all | `AmplitudeSchedule.ramp(1e-4, 4e-3, 5)` | absolute |
| Multiples of each resonator's own | `AmplitudeSchedule.scaled(0.5, 4.0, 5)` | relative |
| Multiples of a base you choose | `AmplitudeSchedule.scaled(0.5, 4.0, 5, base=0.002)` | relative |
| Whatever you like | `AmplitudeSchedule.explicit([1e-4, 3e-4, 2e-3])` | absolute |

Spacing is logarithmic by default — equal ratios, so equal steps in dB, which is
usually what you want when walking a decade of drive power. Pass
`spacing="linear"` for evenly spaced amplitudes instead.

An **amplitude step** is one amplitude. Steps are numbered from 0 in the order
they are measured, and each one can be swept up to twice — once per direction.

```python
from rfmux.tuning import AmplitudeSchedule

ladder = AmplitudeSchedule.scaled(0.5, 4.0, 4)
print(ladder)

for step in ladder.steps(catalog):
    print(step)
```

### Absolute rungs, or relative ones

The distinction only shows itself on an array whose resonators are *not* all
biased at the same amplitude — so let's make that true, by moving two of them:

```python
catalog["R0002"].set_bias(amplitude=PROBE_AMPLITUDE * 4)
catalog["R0003"].set_bias(amplitude=PROBE_AMPLITUDE / 2)

for r in list(catalog)[:4]:
    print(f"{r.name}  bias amplitude {r.bias.amplitude:.5f}")
```

A **relative** ladder multiplies each resonator's own amplitude, so the spread
you just created is preserved and every detector walks its own range. An
**absolute** ladder ignores the catalog's amplitudes entirely and puts every
resonator on the same rung:

```python
relative = AmplitudeSchedule.scaled(1.0, 2.0, 2)
absolute = AmplitudeSchedule.ramp(1e-3, 2e-3, 2)

for label, schedule in [("scaled (relative)", relative), ("ramp (absolute)", absolute)]:
    print(f"\n{label}:  {schedule}")
    for step in schedule.steps(catalog):
        shown = {n: f"{a:.5f}" for n, a in list(step.amplitudes.items())[:4]}
        print(f"  step {step.step}  {shown}")
```

An absolute ladder takes no `base`, because its rungs already *are* the
amplitudes — there would be nothing left for a base to contribute:

```python
try:
    AmplitudeSchedule(ladder=(1e-3, 2e-3), relative=False, base=0.004)
except ValueError as e:
    print(f"ValueError: {e}")
```

## 6. Checking a schedule before you spend an hour on it

A ladder is a slow measurement, so it is worth knowing what it will do first.
`describe()` gives the derived numbers, and `validate()` returns
`(severity, message)` pairs — the same two calls a Periscope dialog renders.

```python
described = ladder.describe(catalog, n_directions=2)

for key in ("nsteps", "n_directions", "n_sweeps", "n_sweep_targets",
            "amplitude_min", "amplitude_max", "spacing"):
    print(f"{key:<18} {described[key]}")

print("\nper resonator (min, max):")
for name, (lo, hi) in list(described["amplitude_range_by_name"].items())[:4]:
    print(f"  {name}  {lo:.5f} → {hi:.5f}")
```

`validate()` is where a ladder that would overshoot full scale gets caught — and
it is caught *here*, before the first sweep, rather than partway through the
run. Amplitudes are normalized DAC units in `(0, 1]`:

```python
for severity, message in ladder.validate(catalog, n_directions=2):
    print(f"{severity:>7}: {message}")

print()
too_loud = AmplitudeSchedule.scaled(1.0, 500.0, 3)
for severity, message in too_loud.validate(catalog):
    print(f"{severity:>7}: {message}")
```

Running that one raises rather than measuring anything:

```python
try:
    await crs.multiamp_multisweep(
        catalog,
        span_hz=SPAN_HZ,
        npoints_per_sweep=NPOINTS_PER_SWEEP,
        amp_schedule=too_loud,
    )
except ValueError as e:
    print(f"ValueError: {e}")
```

## 7. Running the ladder

The call looks like `multisweep`'s, plus `amp_schedule` and `directions`. There
is no `amp` argument here — amplitude is the schedule's job and nothing else's.

`sweep_callback` fires once per completed sweep, which is what to hook a
progress bar or a live plot to. It is also why a failure part-way through does
not lose the sweeps that already finished: they have been handed to you already.

```python
def report(record):
    amplitudes = record["amplitudes"]
    print(f"  [{record['completed']}/{record['total']}] "
          f"step {record['step']} {record['direction']:<8} "
          f"R0001 at {amplitudes['R0001']:.5f}")

ladder_results = await crs.multiamp_multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp_schedule=ladder,
    sweep_callback=report,
)

print(f"\ntop-level keys: {list(ladder_results)}")
print(f"amplitude steps: {list(ladder_results['results'])}")
```

The result is one dict:

- `results` is keyed by **amplitude step**, numbered in the order measured, and
  each step holds one entry per **direction** swept and nothing else. Under a
  direction is exactly what a single `multisweep` returns.
- `call_params` records what the driver was asked for — including the schedule,
  so a saved result can say what produced it.

```python
first = ladder_results["results"][0]["upward"]
print(f"results[0]['upward'] is a normal multisweep return: {list(first)[:4]} …")
print(f"R0001 swept at {first['R0001']['sweep_amplitude']:.5f}")

print(f"\ncall_params: {list(ladder_results['call_params'])}")
print(f"schedule as stored: {ladder_results['call_params']['amp_schedule']}")
```

Note what is *not* in there: no step-level copy of the amplitudes. Each sweep
already records its own `sweep_amplitude`, so storing it twice would only create
two things to keep in step. The readers in the next section rebuild it on
demand.

## 8. Reading a ladder back

Three functions, all plain functions over the returned dict.

**One resonator, across every amplitude.** This is the usual question, and the
shape you want for plotting:

```python
from rfmux.tuning import (
    collect_amplitude_iterations_for,
    find_iteration_matching_amplitude,
    get_amplitudes_at_iteration,
)

iterations = collect_amplitude_iterations_for(ladder_results, "R0001")

for iteration, by_direction in iterations.items():
    sweep = by_direction["upward"]
    print(f"iteration {iteration}  {sweep['sweep_amplitude']:.5f}")
```

**Everything, at one amplitude step.** Rebuilt from each sweep's own
`sweep_amplitude`:

```python
for name, amplitude in list(get_amplitudes_at_iteration(ladder_results, 2).items())[:4]:
    print(f"{name}  {amplitude:.5f}")
```

**Which step was taken at a given amplitude.** With no amplitude given it finds
the step nearest that resonator's *bias* amplitude — usually the one you want,
"where was this thing measured at the power it actually runs at?"

On a relative ladder that comes out the same for everyone, and not by accident:
`scaled` includes the ×1 rung, and ×1 is each resonator's own bias amplitude
whatever that happens to be. Every detector is at its operating point on the
same step.

```python
for name in ("R0001", "R0002", "R0003"):
    at_bias = find_iteration_matching_amplitude(ladder_results, name)
    bias = catalog[name].bias.amplitude
    print(f"{name}  bias {bias:.5f}  → step {at_bias}")
```

Ask for a *fixed* amplitude instead and the three part company, which is why the
function needs a name at all. `R0001`, `R0002` and `R0003` are walking different
ranges, so the same amplitude sits at a different rung of each:

```python
print(f"{'':<8}" + "".join(f"{s:>10}" for s in ladder_results["results"]))
for name in ("R0001", "R0002", "R0003"):
    amplitudes = [
        by_direction["upward"]["sweep_amplitude"]
        for by_direction in collect_amplitude_iterations_for(ladder_results, name).values()
    ]
    print(f"{name:<8}" + "".join(f"{a:>10.5f}" for a in amplitudes))

print()
for name in ("R0001", "R0002", "R0003"):
    step = find_iteration_matching_amplitude(ladder_results, name, 0.002)
    got = get_amplitudes_at_iteration(ladder_results, step)[name]
    print(f"0.00200 for {name}  → step {step}  (actually {got:.5f})")
```

Matching is on *nearest*, not exact — a ladder's amplitudes are floating point,
and `0.001 * 4` is not reliably `0.004`. Those three happen to land exactly.

The corollary is that there is *always* a nearest, so the function answers even
when nothing is remotely close. Ask for an amplitude only `R0002` ever reaches
and the other two return their top rung regardless:

```python
for name in ("R0001", "R0002", "R0003"):
    step = find_iteration_matching_amplitude(ladder_results, name, 0.016)
    got = get_amplitudes_at_iteration(ladder_results, step)[name]
    print(f"0.01600 for {name}  → step {step}  (actually {got:.5f})")
```

So if the match has to be a good one, check it — that last column is how.

And the plot the whole exercise is for — one resonator at every amplitude,
normalized so the shapes can be compared:

```python
def plot_ladder(results, name, direction="upward"):
    iterations = collect_amplitude_iterations_for(results, name)
    fig, (ax_mag, ax_iq) = plt.subplots(1, 2, figsize=(10, 4))

    for iteration, by_direction in iterations.items():
        s = by_direction[direction]
        offset_khz = (s["frequencies"] - s["original_center_frequency"]) / 1e3
        iq = s["iq_complex"] / s["sweep_amplitude"]   # normalize out the drive
        label = f"{s['sweep_amplitude']:.5f}"

        ax_mag.plot(offset_khz, 20 * np.log10(np.abs(iq)), lw=1.0, label=label)
        ax_iq.plot(iq.real, iq.imag, lw=1.0, label=label)

    ax_mag.set_xlabel("offset [kHz]")
    ax_mag.set_ylabel("|S21| / drive [dB]")
    ax_iq.set_xlabel("I / drive")
    ax_iq.set_ylabel("Q / drive")
    ax_iq.set_aspect("equal", "datalim")
    ax_mag.legend(title="amplitude", fontsize=8)
    fig.suptitle(f"{name} swept {direction} at {len(iterations)} amplitudes")
    plt.tight_layout()
    plt.show()

plot_ladder(ladder_results, "R0001")
```

### Both directions

`directions` is an explicit sequence rather than a `"both"` flag, so every sweep
is labelled with both of its coordinates. Order matters: it is the order they
are measured in.

Steps are the outer loop and directions the inner one, so each amplitude's
up-and-down pair is measured together and the amplitude marches monotonically —
which is what you want when walking up towards bifurcation.

```python
both_ways = await crs.multiamp_multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp_schedule=AmplitudeSchedule.scaled(1.0, 2.0, 2),
    directions=("upward", "downward"),
)

for step, by_direction in both_ways["results"].items():
    for direction, sweeps in by_direction.items():
        print(f"step {step}  {direction:<9} "
              f"R0001 at {sweeps['R0001']['sweep_amplitude']:.5f}")
```

A step swept once and a step swept twice have the same shape — the directions
present are simply the ones you asked for:

```python
one_way = await crs.multiamp_multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp_schedule=AmplitudeSchedule.explicit([PROBE_AMPLITUDE]),
    directions=("downward",),
)

print(f"directions present: {list(one_way['results'][0])}")
```

### A frequency list at several amplitudes

The bare-frequency form works here too — this is how you find a sensible probe
amplitude *before* anything is tuned. There is no bias amplitude to scale, so
the schedule has to carry its own: `ramp` and `explicit` do by construction,
while `fixed` and `scaled` would need an explicit `base`.

```python
untuned_ladder = await crs.multiamp_multisweep(
    center_frequencies=section_center_frequencies,
    module=MODULE,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp_schedule=AmplitudeSchedule.ramp(PROBE_AMPLITUDE, PROBE_AMPLITUDE * 4, 3),
)

for step, by_direction in untuned_ladder["results"].items():
    amplitude = by_direction["upward"]["S0001"]["sweep_amplitude"]
    print(f"step {step}  every section at {amplitude:.5f}")

try:
    await crs.multiamp_multisweep(
        center_frequencies=section_center_frequencies,
        module=MODULE,
        span_hz=SPAN_HZ,
        npoints_per_sweep=NPOINTS_PER_SWEEP,
        amp_schedule=AmplitudeSchedule.scaled(0.5, 2.0, 3),   # relative to what?
    )
except ValueError as e:
    print(f"\nValueError: {e}")
```

## 9. What is not here yet

- **Choosing the operating amplitude.** The ladder gives you the data to see
  where each detector bifurcates; deciding which rung to bias at, and writing
  that back into the catalog, is `find_bias_points` and is not ported yet.
- **Fitting.** It consumes multisweep output, and is the other thing that will
  write results back into the catalog.
- **Saving to disk.** `pickle.dump` on the returned dict works today — it is
  plain builtins and ndarrays throughout — but a proper `store.py` with a file
  layout is still to come.

One cleanup note: multisweep silences the channels it swept, but only those. If
you parked tones on this module by hand, they are still live.
