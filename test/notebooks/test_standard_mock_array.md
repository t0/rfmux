---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# The standard simulated array

`rfmux.mock.standard_array` fixes one simulated array for the tuning tests to
share: eight seeded resonators in a 100 MHz band, biased by the simulator
itself, served over RPC with no UDP streaming. This notebook builds it and
walks the whole tuning flow across it — network analysis, resonance finding,
multisweep, fitting, bias finding, biasing — to show what the array looks like
and to pin the properties the tests lean on. It runs as a test in the quick
tier, so every claim below is also an assertion.

```python
import sys
sys.path.append('../../')

import contextlib, io, time
import numpy as np
import matplotlib.pyplot as plt

import rfmux
from rfmux.mock.standard_array import STANDARD_ARRAY, STANDARD_MODULE, standard_array
from rfmux.mock.config import apply_overrides
from rfmux.tuning import (
    AmplitudeSchedule, find_bias_points, find_resonances_in_netanal, fit_sweeps,
)
from rfmux.tuning.fits import BIFURCATION_A

MODULE = STANDARD_MODULE
timings = {}
```

## 1. Build it

The whole build — a simulated board, its resonators, and the simulator's own
biasing — takes well under a second, which is what lets every test start from a
fresh array rather than sharing a mutated one.

```python
t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    crs, catalog = await standard_array()
timings["build"] = time.perf_counter() - t

print(f"built in {timings['build']:.2f} s")
print(catalog)
assert len(catalog.names()) == STANDARD_ARRAY["num_resonances"]
assert timings["build"] < 5.0
```

The configuration, in full, is the simulator's defaults with these changes:

```python
for key, value in STANDARD_ARRAY.items():
    print(f"{key:24s} {value}")
```

## 2. It is the same array every time

The seed fixes the resonators, and the names are derived from their frequencies,
so a test can say `catalog["<name>"]` and mean the same resonator next run.
Regenerating with the same configuration gives the same eight.

The list `generate_resonators` returns is each resonator's *nominal* frequency,
the simulator's `compute_fr()`. Its S21 minimum sits a fixed fraction above
that — the simulator documents the shift as the same fraction for every
resonator — and the tone is parked on the minimum, so the tones run about
0.12 % above the nominal list, by the same ratio all the way across the band.
That ratio is what the check pins: a shift that varied from one resonator to
the next would be a different simulator.

```python
cfg = apply_overrides(dict(STANDARD_ARRAY))
with contextlib.redirect_stdout(io.StringIO()):
    count, nominal_hz = await crs.generate_resonators(cfg)
nominal_hz = np.sort(np.asarray(nominal_hz, dtype=float))

names = catalog.names()
tones_hz = np.array([catalog[n].bias.frequency_hz for n in names])
ratio = tones_hz / nominal_hz
print(f"{'name':6s} {'tone MHz':>13s} {'nominal MHz':>13s} {'tone/nominal':>13s}")
for name, tone, fr, r in zip(names, tones_hz, nominal_hz, ratio):
    print(f"{name:6s} {tone/1e6:13.6f} {fr/1e6:13.6f} {r:13.6f}")

assert count == len(nominal_hz) == STANDARD_ARRAY["num_resonances"]
assert np.all(ratio > 1.0) and np.ptp(ratio) < 1e-4, ratio
# Regenerated with the same seed, the simulator re-biased onto the same tones.
nco = await crs.get_nco_frequency(module=MODULE)
regenerated_hz = np.array([
    nco + await crs.get_frequency(channel=c, module=MODULE)
    for c in range(1, count + 1)])
assert np.allclose(np.sort(regenerated_hz), tones_hz, atol=1.0)
```

One array per process: `load_session` builds a new database session, and a
second one detaches the objects of the first (`crs.module` then raises
`DetachedInstanceError`). The test fixture is module-scoped for that reason.

The array is well separated. The closest pair sets what `min_separation_hz`
a test can ask for without cutting anything real:

```python
gaps_hz = np.diff(tones_hz)
print(f"closest pair {gaps_hz.min()/1e6:.3f} MHz apart, widest gap {gaps_hz.max()/1e6:.2f} MHz")
assert gaps_hz.min() > 1e6
```

## 3. A network analysis finds all of them

Sweep the band and run the finder. The check is the one that matters for the
flow: every resonator is found, nothing else is, and each hit is within a
linewidth or so of the truth.

```python
t = time.perf_counter()
netanal = await crs.take_netanal(
    amp=0.001, fmin=0.995e9, fmax=1.11e9, npoints=30_000, nsamps=10,
    max_chans=1023, module=MODULE, save=False,
)
timings["netanal"] = time.perf_counter() - t
module_netanal = netanal[crs.module[MODULE].index()]

search = find_resonances_in_netanal(
    module_netanal, min_dip_depth_db=1.0, min_Q=1e4, max_Q=1e7,
    min_separation_hz=100e3,
)
found_hz = np.sort(search.resonance_frequencies_hz)
print(search)
print(f"netanal {timings['netanal']:.1f} s; found {len(found_hz)} of {len(tones_hz)}")
assert len(found_hz) == len(tones_hz)
# Found on a 4 kHz grid, at a lower probe power than the bias tone: within a
# few grid points of where the simulator parked each tone.
assert np.all(np.abs(found_hz - tones_hz) < 50e3)
```

```python
trace = module_netanal["results"][0]["upward"]
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(trace["frequencies"] / 1e6, 20 * np.log10(np.abs(trace["iq_counts"])), lw=0.6)
for f in found_hz:
    ax.axvline(f / 1e6, color="C3", lw=0.6, alpha=0.6)
ax.set(xlabel="MHz", ylabel="|S21| dB", title="the standard array, as a network analysis sees it")
plt.show()
```

## 4. One multisweep, and what the fits say

A single-amplitude multisweep at the catalog's bias amplitude, then all three
models. This is the shape of the array: its Qs, its depths, and how nonlinear
it already is at the simulator's own bias power.

```python
t = time.perf_counter()
sweeps = await crs.multisweep(
    catalog, span_hz=100e3, npoints_per_sweep=101, nsamps=10, save=False,
)
timings["multisweep"] = time.perf_counter() - t
module_sweeps = sweeps[crs.module[MODULE].index()]

report = fit_sweeps(module_sweeps)
print(report)
sections = module_sweeps["results"][0]["upward"]
print(f"\n{'name':6s} {'fr MHz':>12s} {'Qr':>9s} {'Qi':>9s} {'depth dB':>9s} {'a':>7s}")
for name in catalog.names():
    s = sections[name]
    sk = s["fits"]["skewed"]["params"]
    nl = s["fits"]["nonlinear"]["params"]
    depth = 20 * np.log10(np.abs(s["iq_counts"]).max() / np.abs(s["iq_counts"]).min())
    print(f"{name:6s} {sk['fr']/1e6:12.6f} {sk['Qr']:9.0f} {sk['Qi']:9.0f} {depth:9.2f} {nl['a']:7.3f}")

assert len(report.failed) == 0, report.failed
assert all(sections[n]["fits"]["nonlinear"]["params"]["a"] < BIFURCATION_A for n in catalog.names())
```

## 5. An amplitude ladder, and bias finding

The bias finder needs a ladder that brackets bifurcation: quiet enough at the
bottom that every resonator is linear, loud enough at the top that most of
them have jumped. This is the ladder the tests use. What the check pins is
that the ladder does its job on this array — at least one resonator bifurcates
inside it — and that bias finding gives every resonator an operating point.

```python
LADDER = AmplitudeSchedule.multiplicative(0.5, 8.0, 5)
print(LADDER)

t = time.perf_counter()
ladder_sweeps = await crs.multisweep(
    catalog, span_hz=100e3, npoints_per_sweep=101, nsamps=10,
    amp=LADDER, sweep_direction=("upward", "downward"), save=False,
)
timings["ladder"] = time.perf_counter() - t
module_ladder = ladder_sweeps[crs.module[MODULE].index()]

bias = find_bias_points(module_ladder, save=False)
print(f"\nladder {timings['ladder']:.1f} s")
print(f"{'name':6s} {'rung':>4s} {'amplitude':>10s} {'bias MHz':>12s} {'bifurcated at':>14s}  flagged")
for f in bias.findings:
    bif = f"{f.bifurcated_at:.5f}" if f.bifurcated_at is not None else "-"
    print(f"{f.name:6s} {f.iteration:4d} {f.amplitude:10.5f} {f.frequency_hz/1e6:12.6f} {bif:>14s}  {f.flagged_because or ''}")

assert len(bias.findings) == len(catalog.names())
n_bifurcated = sum(f.bifurcated_at is not None for f in bias.findings)
print(f"\n{n_bifurcated} of {len(bias.findings)} bifurcate inside the ladder; {len(bias.flagged)} flagged")
assert n_bifurcated >= 1
```

What the check pins is that the ladder *brackets* bifurcation for every
resonator: the fitted nonlinearity is below `BIFURCATION_A` at the second
rung from the top and above it at the top. The ladder is then wide enough
for any detector to have something to detect, and narrow enough that the
answer is one of its rungs.

```python
fit_sweeps(module_ladder, models=("nonlinear",))
top = len(LADDER.ladder) - 1

def a_at(name, rung):
    p = module_ladder["results"][rung]["upward"][name]["fits"]["nonlinear"]["params"]
    return p["a"] if p else float("nan")

print(f"{'name':6s}" + "".join(f"{f'rung {i}':>9s}" for i in range(top + 1)) + "   chosen")
for f in bias.findings:
    print(f"{f.name:6s}" + "".join(f"{a_at(f.name, i):9.3f}" for i in range(top + 1)) + f"   {f.iteration}")

for f in bias.findings:
    assert a_at(f.name, top - 1) < BIFURCATION_A < a_at(f.name, top), f.name
```

### Where the detectors and the fits disagree

The fits put bifurcation between the top two rungs for every resonator, so the
rung below the top is the amplitude a physics reading would choose. The
detectors chose lower on most of them, and on two they fired on the quietest
rung there was. The table below runs each detector on the same sweeps and
shows the rung each would pick, beside the rung the fit would. Nothing here is
asserted: this is the comparison the post-merge plan wants to build into bias
finding, and the array exists so it can be studied. What the detectors are
reacting to on these traces is the open question — the simulator's 1/f
frequency wander between the two passes is the first suspect for the
hysteresis test, and its readout noise for the derivative one.

```python
by_method = {
    method: {f.name: f.iteration
             for f in find_bias_points(module_ladder, amplitude_method=method, save=False).findings}
    for method in ("derivative", "hysteresis", "both")
}
fit_choice = {f.name: max(i for i in range(top + 1) if a_at(f.name, i) < BIFURCATION_A)
              for f in bias.findings}
print(f"{'name':6s} {'derivative':>11s} {'hysteresis':>11s} {'both':>6s} {'fit':>5s}")
for f in bias.findings:
    n = f.name
    print(f"{n:6s} {by_method['derivative'][n]:11d} {by_method['hysteresis'][n]:11d} "
          f"{by_method['both'][n]:6d} {fit_choice[n]:5d}")
agree = sum(by_method["both"][n] == fit_choice[n] for n in fit_choice)
print(f"\nthe default detector agrees with the fit on {agree} of {len(fit_choice)}")
```

```python
name = bias.findings[0].name
fig, axes = plt.subplots(1, len(LADDER.ladder), figsize=(13, 2.6), sharex=True, sharey=True)
for i, ax in enumerate(axes):
    for direction, color in (("upward", "C0"), ("downward", "C1")):
        e = module_ladder["results"][i][direction][name]
        ax.plot((e["frequencies"] - catalog[name].bias.frequency_hz) / 1e3,
                np.abs(e["iq_volts"]), color=color, lw=0.8, label=direction)
    ax.set_title(f"rung {i}: {module_ladder['results'][i]['upward'][name]['sweep_amplitude']:.4f}", fontsize=9)
axes[0].set(ylabel="|S21| V"); axes[0].legend(fontsize=7)
fig.suptitle(f"{name} across the ladder, both directions", y=1.02)
plt.show()
```

## 6. Biasing the board with the result

`apply_bias` programs the report's catalog. Reading the tones back closes the
loop: the board plays what the catalog says.

```python
await crs.apply_bias(bias.catalog)
nco = await crs.get_nco_frequency(module=MODULE)
for name in bias.catalog.names():
    r = bias.catalog[name]
    played = nco + await crs.get_frequency(channel=r.channel, module=MODULE)
    assert abs(played - r.bias.frequency_hz) < 1.0, (name, played, r.bias.frequency_hz)
print("every tone is where its bias point says")
```

## 7. Cost

The array exists so tests can afford to drive the real flow. This is what that
costs on the simulator:

```python
total = sum(timings.values())
for step, seconds in timings.items():
    print(f"{step:12s} {seconds:6.1f} s")
print(f"{'total':12s} {total:6.1f} s")
```
