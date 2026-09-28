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

# From network analysis to a biased array

Run this notebook from top to bottom to generate an unbiased mock array, find
its resonances, measure narrow sweeps around each resonance, step the drive amplitude through
bifurcation, select operating points, program the tones, and acquire slow-stream
and PFB noise. The measurements
use the CRS API; analysis uses `rfmux.tuning`, and a `ResonatorCatalog` carries
the named resonators, channels, amplitudes, and frequencies between steps.

Measurements use `rfmux.tuning.store` and are saved by default. The processing
steps add their results to those measurement dictionaries in place and update
the same files.

This is a streamlined example workflow, which will skim over many details of the
tuning process. More details about each step can be found in the other example
.md workbooks in this folder, which treat each aspect of the tuning flow in more
detail.

The default is a fresh ten-resonator simulation. To use a real board and array,
change the connection and measurement settings in section 1. The remaining
cells are shared.

This is a runnable Jupytext notebook: open it as a notebook in JupyterLab and
use **Shift+Enter**, or **Restart Kernel and Run All Cells**. Use the Python
environment where this checkout is installed. The Markdown stores no outputs;
use **File → Save Notebook As…** to keep an executed copy of the shipped,
read-only notebook.

## 1. Connection and measurement settings

Keep `MODE = "mock"` for a standalone simulation. It uses the same compact,
seeded array as `network_analysis_find_resonances.md`, with no automatic biasing.
The seed fixes the array; measurement noise can still vary. Tuning measurements
use RPC. Section 7 starts and stops the mock UDP streamer for noise acquisition.

For a real array:

- Set `MODE = "hardware"`, `SERIAL`, and `MODULE`. Set `HOSTNAME` if discovery
  needs an explicit address; otherwise leave it `None`.
- Set `FMIN_HZ`, `FMAX_HZ`, and `NETANAL_POINTS` to cover your array with several
  samples across its narrowest resonance. The catalog applied here must fit
  within the API's allowed 500 MHz band around one module's NCO.
- Set the probe amplitude, amplitude schedule, and sweep span for your array
  and attenuation. Amplitudes are normalized DAC values, not dBm. The example
  levels are demonstration settings, not a measured limit for your array.
- Configure the board's clock/timestamp source and RF path for your setup.
  The hardware connection below leaves those settings as configured.



```python
%matplotlib inline

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rfmux

from rfmux.tuning import (
    AmplitudeSchedule, BiasReport, collect_amplitude_iterations_for,
    find_bias_amplitude, find_bias_frequency, find_bias_points,
    find_resonances_in_netanal, magnitude_db, store,
)

from rfmux.core.resonators import ResonatorCatalog

# import example plotting helpers
import sys
DEMO_DIR = Path(rfmux.__file__).resolve().parent / "reference-notebooks" / "Demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))
import example_plotting_multisweep as msplots

MODE = "mock"                 # "mock", "hardware", or "attached"
SERIAL = "0042"                # replace for hardware
HOSTNAME = None                # optional hardware address
MODULE = 1

FMIN_HZ, FMAX_HZ = 600e6, 610e6
NETANAL_POINTS = 2_000
PROBE_AMPLITUDE = 0.001
SPAN_HZ = 100e3
SWEEP_POINTS = 201             # 500 Hz spacing across 100 kHz
NSAMPS = 10
SCHEDULE = AmplitudeSchedule.ramp(0.002, 0.032, 5)

OUTPUT_DIR = store.session_directory()
print(f"rfmux: {rfmux.__file__}")
print(f"results: {OUTPUT_DIR}")
started = time.perf_counter()
```

```python
created_mock = False
if MODE == "mock":
    from rfmux.mock.config import apply_overrides

    session = rfmux.load_session('''
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
''')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()
    mock_config = apply_overrides({
        "num_resonances": 10,
        "freq_start": 601e6,
        "freq_end": 608e6,
        "C_variation": 0.0001,
        "resonator_random_seed": 42,
        "auto_bias_kids": False,
        "pulse_mode": "none",
        "tls_noise_enabled": False,
        "nqp_noise_std_factor": 0.001,
        "T": 0.23,
    })
    count, _ = await crs.generate_resonators(mock_config)
    created_mock = True
    print(f"generated {count} unbiased mock resonators")
elif MODE in ("hardware", "attached"):
    if MODE == "attached":
        HOSTNAME = os.environ["RFMUX_CRS_HOSTNAME"]
        SERIAL = os.environ.get("RFMUX_CRS_SERIAL", "0000")
    # JSON quoting also produces valid YAML strings for the hardware map.
    import json
    address = f", hostname: {json.dumps(HOSTNAME)}" if HOSTNAME else ""
    session = rfmux.load_session(
        f'!HardwareMap [ !CRS {{ serial: {json.dumps(SERIAL)}{address} }} ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()
else:
    raise ValueError(f"Unknown MODE: {MODE}")

module_id = crs.module[MODULE].index()
await crs.clear_channels(module=MODULE)
print(f"connected: {module_id}; channels cleared")
```

## 2. Network analysis and resonance catalog

The wide sweep locates dips. The mock band has about 5 kHz point spacing;
a narrower multisweep will resolve each dip in the next step. Measurement
results are keyed by module even when only one module was measured.
`module_netanal["results"]` accesses that module's arrays, including raw `iq_counts`.

```python
netanal = await crs.take_netanal(
    module=MODULE, fmin=FMIN_HZ, fmax=FMAX_HZ,
    npoints=NETANAL_POINTS, amp=PROBE_AMPLITUDE,
    nsamps=NSAMPS, max_chans=1023, save=True, label="netanal",
)
module_netanal = netanal[module_id]
trace = module_netanal["results"]

search = find_resonances_in_netanal(
    module_netanal, min_dip_depth_db=1.0, min_Q=1e4, max_Q=1e7,
    min_separation_hz=100e3, save=True,
)
print(search)
if not len(search):
    raise RuntimeError("No resonances found: inspect the band and search cuts.")
catalog = search.to_catalog(module=MODULE, amplitude=PROBE_AMPLITUDE)
print(catalog)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(search.frequencies_hz / 1e6, search.magnitude_db, lw=0.8)
for frequency in search.resonance_frequencies_hz:
    ax.axvline(frequency / 1e6, color="C1", alpha=0.6, lw=0.8)
ax.set(xlabel="frequency [MHz]", ylabel="normalized magnitude [dB]",
       title=f"Network analysis: {len(catalog)} accepted resonances")
plt.tight_layout()
plt.show()
```

Inspect the trace before continuing. The depth cut is dip prominence; the Q
bounds limit accepted widths. `min_separation_hz` rejects both members of a
close pair. See `network_analysis_find_resonances.md` for rejected candidates
and tuning those cuts. Catalog names identify resonators; channels refer to the
hardware channel that will synthesize and digitize the resonator's bias tone.
For more info on the Catalog and Resonator objects, see the `resonator_catalogs.md`
workbook.




## 3. Resolve the resonances with a multisweep

A catalog-driven multisweep measures each resonator on its assigned channel.
First take a single upward sweep at the probe amplitude and plot it, for a quick look at the array.

```python
initial_sweeps = await crs.multisweep(
    catalog, span_hz=SPAN_HZ, npoints_per_sweep=SWEEP_POINTS,
    nsamps=NSAMPS, sweep_direction="upward",
    save=True, label="initial_sweeps",
)
module_initial = initial_sweeps[module_id]
```


The structure is `results[step][direction][resonator_name]`. A section contains
`frequencies`, `iq_counts`, `iq_volts`, `sweep_amplitude`, and
`sweep_amplitude_dbm`.

You can write your own plotters for the data, but here we use some canned example
plotting helpers for brevity.
The standard `plot_magnitude_panels()` from `example_plotting_multisweep.py`
plots one panel per resonator. Once bias finding has saved a report, its dashed
vertical line is the new bias frequency, and the trace at the new bias
amplitude is thicker. Pass `show_bias_frequency=False` or
`highlight_bias_amplitude=False` to omit either decoration.

```python


msplots.plot_magnitude_panels(
    module_initial, directions="upward", normalize=True, ncols=4,
    title="Initial multisweep at the probe amplitude",
)
```

The same plotter handles multiple amplitudes and directions, with a shared
amplitude colour scale and batches of 50 resonators per figure by default.
Use `names=catalog.names()[:4]`, `iterations=0`, or `directions="upward"`
to inspect a subset. `msplots.plot_iq_panels()` accepts the same selections
for an IQ view; see `example_plotting_multisweep.py` for the full options.

Check that each sweep contains its dip. Adjust the span, point count, or
sweep centers if necessary before spending time on the amplitude scan. Pass
`center_frequencies={name: absolute_hz}` alongside the catalog to move sweep
windows without replacing bias points; the mapping must cover every name.
See `multisweep.md` for a runnable example.

## 4. Iterate multisweeps over various amplitudes

To efficiently handle iteratively calling multisweeps at multiple amplitudes,
we use the `AmplitudeSchedule` class. This provides a number of options, which
are explored in more detail in `multisweep.md`.

The `AmplitudeSchedule.ramp()` that we declared in the first cell
specifies five absolute amplitudes, logarithmically
spaced from 0.002 to 0.032. Each step measures the whole catalog in both
frequency directions. This is ten multisweep passes; the driver handles the
iteration and preserves the steps together in one result.

For per-resonator starting amplitudes, use
`AmplitudeSchedule.multiplicative(0.5, 8.0, 5)` instead: each factor multiplies
that resonator's catalog amplitude. Edit `SCHEDULE` and rerun this section to
extend or refine the range based on the bias report below.

The mock retains tone state between frequency points and can follow different
branches in upward and downward sweeps through bifurcation.
Use its traces to explore the analysis flags; a simulated threshold does not
establish a real array’s operating limit.

```python
print(SCHEDULE.describe(catalog, n_directions=2))
amplitude_sweeps = await crs.multisweep(
    catalog, amp=SCHEDULE, span_hz=SPAN_HZ,
    npoints_per_sweep=SWEEP_POINTS, nsamps=NSAMPS,
    sweep_direction=("upward", "downward"),
    save=True, label="amplitude_sweeps",
)
msplots.plot_magnitude_panels(
    amplitude_sweeps[module_id], normalize=True, ncols=4,
    title="Amplitude scan: both sweep directions",
)
```

Solid lines are upward sweeps; dashed lines are downward sweeps.

## 5. Choose the bias amplitude, then the bias frequency at that amplitude

The bias amplitude finder examines measured levels from low to high and chooses
the step below the first detected bifurcation. The bifurcation detection method `"derivative"` looks for jumps
in normalized IQ arc speed; `"hysteresis"` compares the two directions;
`"both"` flags either test. This example uses the derivative test for the mock
and both tests for the hardware example. The thresholds below are explicit so they can be
adjusted after inspecting the traces (see `bias_finding.md`).

This first cell exposes the two decisions for one resonator: first choosing the amplitude, then the frequency.
 Frequency selection
uses the **chosen amplitude's** upward trace: `"iq_derivative"` selects the
largest IQ motion per hertz. `"minimum"` is the alternative dip minimum.

```python
AMPLITUDE_METHOD = "derivative" if MODE == "mock" else "both"
BIAS_SETTINGS = dict(
    amplitude_method=AMPLITUDE_METHOD,
    frequency_method="iq_derivative", direction="upward",
    spike_prominence_factor=0.5, noise_gate_factor=50.0,
    max_discrepancy=0.1, compare="magnitude",
)
name = catalog.names()[0]
iterations = collect_amplitude_iterations_for(amplitude_sweeps[module_id], name)
choice = find_bias_amplitude(
    iterations, method=AMPLITUDE_METHOD,
    spike_prominence_factor=BIAS_SETTINGS["spike_prominence_factor"],
    noise_gate_factor=BIAS_SETTINGS["noise_gate_factor"],
    max_discrepancy=BIAS_SETTINGS["max_discrepancy"],
    compare=BIAS_SETTINGS["compare"],
)
selected = iterations[choice.iteration]["upward"]
frequency = find_bias_frequency(selected, method=BIAS_SETTINGS["frequency_method"])
print(f"{name}: step {choice.iteration}, amplitude {choice.amplitude:g}, "
      f"frequency {frequency / 1e6:.6f} MHz")

columns = min(3, len(iterations))
rows = (len(iterations) + columns - 1) // columns
fig, axes = plt.subplots(
    rows, columns, figsize=(3.8 * columns, 3.0 * rows),
    sharex=True, sharey=True, squeeze=False, constrained_layout=True,
)
for ax, (step, entries) in zip(axes.flat, iterations.items()):
    amplitude = entries["upward"]["sweep_amplitude"]
    check = choice.checks.get(step)
    verdict = f"bifurcated={check.bifurcated}" if check else "not checked"
    chosen = step == choice.iteration
    for direction, entry in entries.items():
        offset = (entry["frequencies"] - selected["original_center_frequency"]) / 1e3
        ax.plot(offset, magnitude_db(entry["iq_volts"] / entry["sweep_amplitude"]),
                ls="-" if direction == "upward" else "--", label=direction)
    ax.set_title(f"step {step}: amplitude {amplitude:g}"
                 f"{' — CHOSEN' if chosen else ''}\n{verdict}", fontsize=10)
    if chosen:
        ax.set_facecolor("#eaf5e9")
        for spine in ax.spines.values():
            spine.set_color("#287a35")
            spine.set_linewidth(2)
        ax.axvline((frequency - selected["original_center_frequency"]) / 1e3,
                   color="black", ls=":", label="bias frequency")
    ax.legend(fontsize=8)
for ax in list(axes.flat)[len(iterations):]:
    ax.set_visible(False)
fig.supxlabel("offset from sweep centre [kHz]")
fig.supylabel("magnitude [dB, sweep-amplitude-normalized]")
fig.suptitle(f"{name}: selecting amplitude and frequency")
plt.show()
```

Each panel is one measured amplitude, with both sweep directions. The green
panel is the selected step; its dotted line marks the selected frequency.
The search stops at the first detected bifurcation, so higher measured levels
can be labelled **not checked**. Each trace is divided by its recorded sweep
amplitude before conversion to dB, removing the commanded drive scaling while
preserving changes in its measured baseline.

`find_bias_points()` performs those steps for every resonator and returns a
`BiasReport` with a new catalog. This `BiasReport` is also saved into the
existing multisweep file. It rounds bias frequencies onto the tone grid
and measures IQ derivatives there from the selected sweep. Those derivatives
supply the catalog's `df_calibration`; no separate calibration measurement or
phase rotation is performed by this call. The input catalog is preserved under
`'call_params'`.

```python
bias_report = find_bias_points(amplitude_sweeps[module_id], **BIAS_SETTINGS, save=True)
print(bias_report)
print(bias_report.catalog)

```

Review the flags and selected traces before applying to a real array.
The API returns a point even for flagged resonators; it does not silently
remove them.

Note that you can run `find_bias_points` on multisweep data multiple times, using
different analysis parameters, to refine your results. You do not need to retake
the measurement.

The following panels show each resonator's selected sweep, normalized by its
recorded sweep amplitude in dB, with shared axes. Zero frequency offset marks
its bias frequency.

```python
columns = min(4, len(bias_report.findings))
rows = (len(bias_report.findings) + columns - 1) // columns
fig, axes = plt.subplots(
    rows, columns, figsize=(3.4 * columns, 2.7 * rows),
    sharex=True, sharey=True, squeeze=False, constrained_layout=True,
)
for ax, finding in zip(axes.flat, bias_report.findings):
    entry = bias_report.catalog[finding.name].bias.bias_sweep
    offset = (entry["frequencies"] - finding.frequency_hz) / 1e3
    ax.plot(offset, magnitude_db(entry["iq_volts"] / entry["sweep_amplitude"]), lw=1.2)
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.set_title(f"{finding.name}: amplitude {finding.amplitude:g}"
                 f"{' (flagged)' if finding.flagged_because else ''}", fontsize=10)
for ax in list(axes.flat)[len(bias_report.findings):]:
    ax.set_visible(False)
fig.supxlabel("offset from bias frequency [kHz]")
fig.supylabel("magnitude [dB, sweep-amplitude-normalized]")
fig.suptitle("Each resonator at its selected amplitude; dotted line = bias frequency")
plt.show()
```

## 6. Apply the bias catalog

`crs.apply_bias()` programs each catalog resonator's frequency and amplitude
on its assigned channel. It uses the catalog's module and moves the NCO only
if the current setting cannot carry the tones. It does not select operating
points: that was analysis in section 5.

```python
await crs.apply_bias(bias_report.catalog)

```

## 7. Acquire slow-stream and PFB noise

`measure_noise()` measures the configured catalog channels and returns
`{module_id: block}`, just like multisweep. It reads actual tone frequencies
and amplitudes, keeps a snapshot of the catalog, and saves through `store`.
It does not select or apply biases. `decimation=None` preserves stream settings.

These short captures use `reference="absolute"`: saved time-domain IQ is in
**volts** (`iq_volts`), and spectra are in **dBm/Hz**. Relative captures retain
**counts** (`iq_counts`); the example plotters accept either. `nsegments` sets spectral averaging;
increase sample counts to measure for longer and resolve lower frequencies.
Slow UDP and per-channel PFB RPC captures are sequential, not synchronized.
The PFB UDP streamer must be off for RPC capture; the NCO is not reset.

The **mock RPC PFB capture is uniform synthetic noise**, not detector noise.
The slow stream uses the resonator model. This example disables pulses and TLS
noise and retains a small quasiparticle-noise term.


```python
from rfmux.streamer import find_streamer_conflict

NOISE_PARAMS = dict(
    num_samples=1_000, nsegments=5, reference="absolute",
    spectrum_cutoff=0.9, pfb_samples=20_000, pfb_nsegments=5,
)
started_mock_stream = False
try:
    if created_mock:
        conflict = find_streamer_conflict()
        if conflict:
            raise RuntimeError(f"Cannot start a second mock stream: {conflict}")
        started_mock_stream = await crs.start_udp_streaming()
    noise_results = await crs.measure_noise(
        bias_report.catalog, **NOISE_PARAMS, save=True, label="noise_results")
finally:
    if started_mock_stream:
        await crs.stop_udp_streaming()
        print("notebook's mock UDP streamer stopped")

noise_path = store.saved_path(noise_results)
print(f"saved noise: {noise_path}")
```

Use the example noise plotters to take a quick look at the data.
IQ overlays use
the calibration sweep in the saved catalog; timestreams use nominal sample
spacing; PSD plots omit carrier bins and their immediate neighbors only in
the display. Saved arrays remain unchanged. See `noise_measurement.md` for
file inspection, an explicit verification multisweep, and plot options.

```python
import example_plotting_noise as noiseplots

module_noise = store.load(noise_path)[module_id]
noiseplots.plot_iq_panels(module_noise)
noiseplots.plot_timestreams(module_noise)
noiseplots.plot_psds(module_noise)
noiseplots.plot_psds(
    module_noise, stream="pfb",
    title="PFB noise (mock RPC: synthetic uniform noise)" if created_mock else None)
```

## 8. Keep and reload the results

Measurements were saved through `rfmux.tuning.store`; the resonance search
and bias report were saved back alongside their source measurements. The saved
files are pickle files, and CAN be loaded using pickle.load() but this is not advised -
when loaded with `store.load(...)`, the code reads and updates the path to where it read
from, and then will be reliably able to save updates to that file, even if it is not
in the place where it was originally generated (e.g., you sent your favourite multisweep
to your friend)

This makes it easy to reload a previous multisweep, extract its resonator
catalog, and re-tune the array from this saved starting point.

```python
for measurement in (module_netanal, module_initial, amplitude_sweeps[module_id]):
    print(store.saved_path(measurement))

saved_sweeps = store.load(store.saved_path(amplitude_sweeps[module_id]))
restored_catalog = ResonatorCatalog.from_dict(saved_sweeps[module_id]["bias_report"]['catalog'])
print(restored_catalog)
restored_noise = store.load(noise_path)
print(f"reloaded noise for {len(restored_noise[module_id]['results']['resonators'])} resonators")
# To reapply later, after connecting to the intended board:
# await crs.apply_bias(restored_catalog)
print(f"workflow completed in {time.perf_counter() - started:.1f} s")
```

The board is left biased. Section 7 stops only the mock streamer owned by this
notebook; it leaves hardware and attached-session streams running.
For further characterization, see `multisweep.md` and `bias_finding.md`;
for timestreams and pulse capture, see `pulse_capture.md`.
