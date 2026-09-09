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

# From network analysis to a biased array

Run this notebook from top to bottom to generate an unbiased mock array, find
its resonances, measure narrow sweeps, step the drive amplitude through
bifurcation, select operating points, and program the tones. The measurements
use the CRS API; analysis uses `rfmux.tuning`, and a `ResonatorCatalog` carries
the named resonators, channels, amplitudes, and frequencies between steps.

The default is a fresh ten-resonator simulation. To use a real board and array,
change the connection and measurement settings in section 1. The remaining
cells are shared. This flow changes the selected module's NCO and tones.

This is a runnable Jupytext notebook: open it as a notebook in JupyterLab and
use **Shift+Enter**, or **Restart Kernel and Run All Cells**. Use the Python
environment where this checkout is installed. The Markdown stores no outputs;
use **File → Save Notebook As…** to keep an executed copy of the shipped,
read-only notebook.

## 1. Connection and measurement settings

Keep `MODE = "mock"` for a standalone simulation. It uses the same compact,
seeded array as `network_analysis_find_resonances.md`, with no automatic biasing.
The seed fixes the array; measurement noise can still vary. RPC measurements
suffice for this workflow, so this simulation does not start a UDP streamer.

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

To attach to a CRS advertised by Periscope, use `MODE = "attached"`. This uses
`RFMUX_CRS_HOSTNAME` and `RFMUX_CRS_SERIAL` and does not generate another array.
Set the measurement band for that session's array too. Coordinate measurements
with Periscope because both clients control the same tones.

```python
%matplotlib inline

import os
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rfmux

from rfmux.tuning import (
    AmplitudeSchedule, BiasReport, collect_amplitude_iterations_for,
    find_bias_amplitude, find_bias_frequency, find_bias_points,
    find_resonances_in_netanal, fit_sweeps, netanal_trace, store,
)

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

OUTPUT_DIR = Path(os.environ.get(
    "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_tuning_flow"))
store.set_output_directory(OUTPUT_DIR)
print(f"rfmux: {rfmux.__file__}")
print(f"results: {OUTPUT_DIR}")
started = time.perf_counter()
```

```python
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
`netanal_trace()` accesses that module's arrays, including raw `iq_counts`.

```python
netanal = await crs.take_netanal(
    module=MODULE, fmin=FMIN_HZ, fmax=FMAX_HZ,
    npoints=NETANAL_POINTS, amp=PROBE_AMPLITUDE,
    nsamps=NSAMPS, max_chans=1023, save=True, label="tuning_netanal",
)
module_netanal = netanal[module_id]
trace = netanal_trace(module_netanal)

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
and tuning those cuts. Catalog names identify resonators; channels identify
where they are played. Neither is a frequency or an amplitude-step index.

If the real RF path needs cable-delay compensation, set it before the narrow
sweeps. This optional cell fits the unwrapped network-analysis phase and adds
the residual delay to the current cable length. Inspect the phase over a
suitable band before enabling it.

```python
CORRECT_CABLE_DELAY = False
if CORRECT_CABLE_DELAY:
    from rfmux.core.transferfunctions import (
        calculate_new_cable_length, fit_cable_delay,
    )
    delay = fit_cable_delay(
        trace["frequencies"], np.degrees(np.angle(trace["iq_counts"])))
    old_length = await crs.get_cable_length(module=MODULE)
    new_length = calculate_new_cable_length(old_length, delay)
    await crs.set_cable_length(length=new_length, module=MODULE)
    print(f"residual delay {delay * 1e9:+.3f} ns; cable length {new_length:.3f} m")
```

## 3. Resolve the resonances with a multisweep

A catalog-driven multisweep measures each resonator on its assigned channel.
First take a single upward sweep at the probe amplitude and fit the skewed
model for characterization. Fits are stored beside each measured section;
this fit does not choose or apply the final operating points.

```python
initial_sweeps = await crs.multisweep(
    catalog, span_hz=SPAN_HZ, npoints_per_sweep=SWEEP_POINTS,
    nsamps=NSAMPS, sweep_direction="upward",
    save=True, label="tuning_probe",
)
module_initial = initial_sweeps[module_id]
fit_report = fit_sweeps(module_initial, models=("skewed",), save=True)
print(fit_report)
```

The structure is `results[step][direction][resonator_name]`. A section contains
`frequencies`, `iq_counts`, `iq_volts`, and `sweep_amplitude`. The helper below
plots all resonators and works for both a single sweep and an amplitude scan.
Dividing magnitude by the drive amplitude lets us compare trace shapes.

```python
def plot_sweeps(module_sweeps: dict, names: list[str]) -> None:
    columns = min(4, len(names))
    rows = (len(names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 2.8 * rows),
                             squeeze=False, constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        for step, directions in module_sweeps["results"].items():
            for direction, sections in directions.items():
                entry = sections[name]
                offset = (entry["frequencies"] - entry["original_center_frequency"]) / 1e3
                magnitude = np.abs(entry["iq_counts"]) / entry["sweep_amplitude"]
                ax.plot(offset, 20 * np.log10(np.maximum(magnitude, 1e-30)),
                        color=f"C{step % 10}",
                        ls="-" if direction == "upward" else "--",
                        label=f"{entry['sweep_amplitude']:.4g}" if direction == "upward" else None)
        ax.set(title=name, xlabel="offset from sweep centre [kHz]",
               ylabel="magnitude / drive [dB counts]")
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(title="DAC amplitude", fontsize=8)
    plt.show()

plot_sweeps(module_initial, catalog.names())
```

Check that each sweep contains its dip. Adjust the span, point count, or catalog
centres if necessary before spending time on the amplitude scan.

## 4. Iterate multisweeps over drive amplitude

`AmplitudeSchedule.ramp()` specifies five absolute amplitudes, logarithmically
spaced from 0.002 to 0.032. Each step measures the whole catalog in both
frequency directions. This is ten multisweep passes; the driver handles the
iteration and preserves the steps together in one result.

For per-resonator starting amplitudes, use
`AmplitudeSchedule.multiplicative(0.5, 8.0, 5)` instead: each factor multiplies
that resonator's catalog amplitude. Edit `SCHEDULE` and rerun this section to
extend or refine the range based on the bias report below.

The mock evaluates frequency points independently and does not reproduce
physical hysteresis. It demonstrates drive-dependent traces and the analysis
flags; its detected threshold is not a validation of a real array's limit.

```python
print(SCHEDULE.describe(catalog, n_directions=2))
amplitude_sweeps = await crs.multisweep(
    catalog, amp=SCHEDULE, span_hz=SPAN_HZ,
    npoints_per_sweep=SWEEP_POINTS, nsamps=NSAMPS,
    sweep_direction=("upward", "downward"),
    save=True, label="tuning_amplitudes",
)
module_amplitudes = amplitude_sweeps[module_id]
plot_sweeps(module_amplitudes, catalog.names())
```

Solid lines are upward sweeps; dashed lines are downward sweeps.

## 5. Choose amplitude, then frequency at that amplitude

The amplitude finder examines measured levels from low to high and chooses
the step below the first detected bifurcation. `"derivative"` looks for jumps
in normalized IQ arc speed; `"hysteresis"` compares the two directions;
`"both"` flags either test. Use the derivative test for this mock, and both
for the hardware example. The thresholds below are explicit so they can be
adjusted after inspecting the traces (see `bias_finding.md`).

This first cell exposes the two decisions for one resonator. Frequency selection
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
iterations = collect_amplitude_iterations_for(module_amplitudes, name)
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
for step, check in choice.checks.items():
    print(f"step {step}: bifurcated={check.bifurcated}, metrics={check.metric}")
```

`find_bias_points()` performs those steps for every resonator and returns a
`BiasReport` with a new catalog. It rounds bias frequencies onto the tone grid
and measures IQ derivatives there from the selected sweep. Those derivatives
supply the catalog's `df_calibration`; no separate calibration measurement or
phase rotation is performed by this call. The input catalog is preserved.

```python
bias_report = find_bias_points(module_amplitudes, **BIAS_SETTINGS, save=True)
print(bias_report)
for finding in bias_report.findings:
    print(f"{finding.name}: step {finding.iteration}, "
          f"amplitude {finding.amplitude:g}, "
          f"frequency {finding.frequency_hz / 1e6:.6f} MHz, "
          f"bifurcated at {finding.bifurcated_at}; "
          f"{finding.flagged_because or 'bracketed operating point'}")
```

Review the flags and selected traces before applying to a real array:

- If nothing bifurcated, the highest measured amplitude is selected and flagged.
  Extend the range if appropriate; no upper limit was established.
- If the lowest amplitude bifurcated, that amplitude is selected and flagged.
  Measure lower levels to find a point below the detected transition.
- A coarse amplitude schedule only brackets the transition. Rerun section 4
  with more levels around it if a finer choice matters.

The API returns a point even for flagged resonators; it does not silently
remove them. This demonstration applies the entire report catalog below.
Changing the analysis settings only requires rerunning section 5, not acquiring
new data. The report is also saved inside the amplitude measurement.

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
for finding in bias_report.findings:
    resonator = bias_report.catalog[finding.name]
    entry = resonator.bias.bias_sweep
    f = entry["frequencies"]
    iq = entry["iq_volts"]
    offset = (f - finding.frequency_hz) / 1e3
    axes[0].plot(offset, np.abs(iq), label=finding.name)
    axes[1].plot(iq.real, iq.imag)
axes[0].axvline(0, color="black", ls=":", label="selected frequency")
axes[0].set(xlabel="offset from bias frequency [kHz]", ylabel="magnitude [V]",
            title="Each resonator at its selected amplitude")
axes[1].set(xlabel="I [V]", ylabel="Q [V]", title="Selected IQ sweeps")
axes[0].legend(fontsize=7, ncol=2)
plt.show()
```

## 6. Apply the bias catalog and read back the tones

`crs.apply_bias()` programs each catalog resonator's frequency and amplitude
on its assigned channel. It uses the catalog's module and moves the NCO only
if the current setting cannot carry the tones. It does not select operating
points: that was analysis in section 5.

```python
await crs.apply_bias(bias_report.catalog)
nco = await crs.get_nco_frequency(module=MODULE)
frequency_errors = []
amplitude_errors = []
for resonator in bias_report.catalog:
    frequency = nco + await crs.get_frequency(channel=resonator.channel, module=MODULE)
    amplitude = await crs.get_amplitude(channel=resonator.channel, module=MODULE)
    frequency_errors.append(abs(frequency - resonator.bias.frequency_hz))
    amplitude_errors.append(abs(amplitude - resonator.bias.amplitude))
    print(f"{resonator.name}: channel {resonator.channel}, "
          f"{frequency / 1e6:.6f} MHz, amplitude {amplitude:g}")
print(f"maximum frequency readback error: {max(frequency_errors):.3g} Hz")
print(f"maximum amplitude readback error: {max(amplitude_errors):.3g}")
```

## 7. Keep and reload the result

Measurements were saved through `rfmux.tuning.store`; the resonance search,
fits, and bias report were saved back alongside their source measurements.
Reload the amplitude file and reconstruct the catalog without a board or a
new sweep. Only load measurement files you trust, since the format is pickle.

```python
for measurement in (module_netanal, module_initial, module_amplitudes):
    print(store.saved_path(measurement))

saved_sweeps = store.load(store.saved_path(module_amplitudes))
restored_report = BiasReport.from_dict(saved_sweeps[module_id]["bias_report"])
restored_catalog = restored_report.catalog
print(restored_catalog)
# To reapply later, after connecting to the intended board:
# await crs.apply_bias(restored_catalog)
print(f"workflow completed in {time.perf_counter() - started:.1f} s")
```

The board is left biased. This notebook started no UDP streamer to tear down.
For further characterization, see `fitting_resonators.md` and `bias_finding.md`;
for timestreams and pulse capture, see `pulse_capture.md`.
