---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Full KID Tuning Guide

This is an end-to-end detector tuning guide using the Python API rather than
the Periscope GUI. The first steps are for a new chip on its first cooldown.
Once the nominal detector frequencies are known, start at section 6.

Everything here is the same code path used by the Periscope GUI. Periscope
runs these functions from `QThread` workers and draws the results; this notebook
calls them directly and plots instead.

| Step | Function |
|---|---|
| Perform a network analysis to find the resonances | `crs.take_netanal()` |
| Remove the cable delay (optional) | `fit_cable_delay`, `calculate_new_cable_length` |
| Select resonator frequencies | `find_resonances()` |
| Characterize each one in more detail | `crs.multisweep()` |
| Fit the resonances | `fit_skewed_multisweep`, `fit_nonlinear_iq_multisweep` |
| Bias the detectors | `bias_kids()` |
| Measure the noise | `crs.py_get_samples()`, `crs.py_get_pfb_samples()` |

## How to use this document

Run the cells in order; later ones use variables the earlier ones defined.
Sections 1 and 2 are the exception: run only the one option (1A, 1B or 2)
that fits.

This notebook format doesn't embed outputs like ipython noteboooks, and is shipped read-only.
It is executable and will fill with outputs like an ipython notebook.
To save it with those outputs, use *File → Save Notebook As*.

**This notebook changes the board's state.** 

```python
%matplotlib inline

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rfmux
from rfmux.core.transferfunctions import (
    calculate_new_cable_length, fit_cable_delay,
)
from rfmux.algorithms.measurement.bias_kids import bias_kids
from rfmux.algorithms.measurement.fitting import (
    find_resonances, fit_skewed_multisweep,
)
from rfmux.algorithms.measurement.fitting_nonlinear import (
    fit_nonlinear_iq_multisweep,
)

# Reference notebooks are provisioned read-only, so results go to a
# scratch directory; override it with RFMUX_DEMO_OUTPUT.
OUTPUT_DIR = Path(os.environ.get(
    "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_tuning_flow"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODULE = 1

crs = None          # set by whichever cell in section 1 or 2 you run
IS_MOCK = False     # True only if THIS notebook created the simulation

print(f"results → {OUTPUT_DIR}")
```

## 1. Connect

Everything below needs a CRS. **Run exactly one** of the three options:

| | When to use it | Where |
|---|---|---|
| **A. An existing Mock or Real Periscope session is already running** | Periscope launched the Jupyter environment you are viewing this notebook within, and it is already configured for either a real board or a mock instance | below |
| **B. Starting from scratch with real hardware** | You have a CRS and this notebook is being viewed separately from Periscope | below |
| **C. Start a new simulated environment** | No Periscope GUI instance already, and nothing already running | section 2 |

### A. Attach to the CRS Periscope is driving

Use this if you are viewing this notebook from within Periscope's embedded jupyter environment.
Periscope sets `RFMUX_CRS_HOSTNAME` when it launches this notebook, which is how
the cell finds the board with no configuration from you.

This is most important if you are currently running Periscope in mock mode, to avoid
generating a second `create_mock_crs()`, which would produce a *second, unrelated* simulation,
whose detectors are not the ones Periscope is showing you.

```python
HOSTNAME = os.environ.get("RFMUX_CRS_HOSTNAME")   # or paste "127.0.0.1:43431"
SERIAL = os.environ.get("RFMUX_CRS_SERIAL", "0000")

if HOSTNAME:
    s = rfmux.load_session(
        f'!HardwareMap [ !CRS {{ serial: "{SERIAL}", '
        f'hostname: "{HOSTNAME}" }} ]')
    crs = s.query(rfmux.CRS).one()
    await crs.resolve()
    print(f"attached to CRS {SERIAL} at {HOSTNAME}")
else:
    print("Nothing advertised a board. Set HOSTNAME above, or use option B "
          "(your own board) or section 2 (simulation).")
```

### B. Your own board, no Periscope GUI

```python
# SERIAL = "0042"
# s = rfmux.load_session(f'!HardwareMap [ !CRS {{ serial: "{SERIAL}" }} ]')
# crs = s.query(rfmux.CRS).one()
# await crs.resolve()
# await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)
# print(f"connected to CRS {SERIAL}")
```

## 2. Mock mode configuration, no Periscope GUI

**Skip this section if section 1 gave you a CRS.** The cell below does
nothing in that case.

If you have no hardware, this cell creates a simulated CRS with ten resonators
between 600 MHz and 1 GHz. They come from a physical LEKID model and respond to
drive power and temperature the way real ones do.

`auto_bias_kids` stays at its default, `False`, so the simulation leaves the
resonators unbiased and section 8 biases them. (`pulse_capture.md` sets it
`True`: the simulation then sweeps each resonator at `bias_amplitude` and
biases it at the S21 minimum it finds.)

`resonator_random_seed` fixes the array: same ten detectors on every run, so a
number that changes between runs is your change, not the simulation's.

> **This cell refuses to run if something is already streaming.** Two
> simulations send to the same UDP port and a receiver gets both interleaved,
> with no error. The message says which case you are in. If Periscope is in
> mock mode, attach to its simulation with option 1A.

```python
MOCK_CONFIG = {
    "num_resonances": 10,
    "freq_start": 0.6e9,
    "freq_end": 1.0e9,
    "resonator_random_seed": 42,
}

if crs is not None:
    print("already connected: skip this cell")
else:
    from rfmux.streamer import find_streamer_conflict

    if os.environ.get("RFMUX_CRS_HOSTNAME"):
        raise RuntimeError(
            "Periscope launched this notebook and is already driving a CRS.\n"
            "Run option 1A above to attach to that one. Creating a second "
            "simulation here would stream to the same UDP port as Periscope's, "
            "and every reader would see the two interleaved.")

    conflict = find_streamer_conflict()
    if conflict:
        raise RuntimeError(
            f"Something is already using the streamer port: {conflict}.\n"
            "A second simulation would send to that same port, and a reader "
            "would get both streams interleaved with nothing to say so.\n"
            "Attach to what is running with option 1A, or stop it, then re-run "
            "this cell.")

    from rfmux.mock.helpers import create_mock_crs
    crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG,
                                verbose=False)
    IS_MOCK = True
    print(f"simulated CRS ready: {MOCK_CONFIG['num_resonances']} resonators "
          f"between {MOCK_CONFIG['freq_start']/1e9:.1f} and "
          f"{MOCK_CONFIG['freq_end']/1e9:.1f} GHz")
```

### Confirm the connection

The rest of the notebook uses `crs`; this fails early if section 1 did not set
it, and clears channels left programmed by an earlier run.

```python
if crs is None:
    raise RuntimeError(
        "No CRS. Run option 1A (attach), 1B (your board), or the cell above "
        "(simulate one) before continuing.")

await crs.clear_channels(module=MODULE)

print(f"CRS    {crs.tuber_hostname}")
print(f"module {MODULE}, channels cleared")
print("simulation created by this notebook" if IS_MOCK
      else "pre-existing board: this notebook will not tear it down")
```

## 3. Network analysis

The first measurement is a wide sweep: step a comb of tones across the band and
record the transmitted amplitude and phase at each frequency. Resonators appear
as narrow dips in |S21|, each absorbing power at its resonant frequency.

`take_netanal` drives up to `max_chans` tones at once and re-tunes the NCO for
each `max_span`-wide chunk. The cell below sweeps 50,000 points.

Parameters:

- **`amp`**: drive amplitude in normalized DAC units. Too high drives the
  resonators nonlinear (they bifurcate and the dip is distorted); too low
  measures the amplifier's noise. 0.001 is a starting point for a first look.
- **`nsamps`**: samples averaged per point.
- **`npoints`**: sweep resolution. A resonator with too few points across it
  cannot be fitted, and at high Q the linewidth is a few kHz.
- **`max_span`**: defaults to 500 MHz, the droop-free bandwidth of one NCO
  setting.

> If Periscope is attached to the same board its plots follow along, as if you
> had driven its Network Analysis panel.

```python
NETANAL_PARAMS = {
    "amp": 0.001,
    "fmin": 0.6e9,
    "fmax": 1.1e9,
    "nsamps": 10,
    "npoints": 50000,
    "max_chans": 1023,
    "max_span": 500e6,
    "module": MODULE,
}

# progress_callback is the same hook Periscope uses to drive its progress bar.
_shown = [-25.0]
def netanal_progress(module, percentage):
    if percentage - _shown[0] >= 25.0:
        _shown[0] = percentage
        print(f"  sweeping… {percentage:.0f}%")

netanal = await crs.take_netanal(progress_callback=netanal_progress,
                                 **NETANAL_PARAMS)

frequencies = netanal["frequencies"]
iq_complex = netanal["iq_complex"]
phase_degrees = netanal["phase_degrees"]

mag_db = 20 * np.log10(np.maximum(np.abs(iq_complex), 1e-30))
print(f"\n{len(frequencies)} points, "
      f"{frequencies[0]/1e6:.0f}–{frequencies[-1]/1e6:.0f} MHz")
```

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax1.plot(frequencies / 1e6, mag_db, lw=0.6)
ax1.set_ylabel("|S21| (dB)")
ax1.set_title("Network analysis")
ax2.plot(frequencies / 1e6, phase_degrees, lw=0.6, color="#CC6633")
ax2.set_ylabel("phase (deg)"); ax2.set_xlabel("frequency (MHz)")
plt.tight_layout(); plt.show()
```

## 4. Unwrap the cable delay

A signal that takes τ seconds to travel out and back arrives with a phase that
changes linearly with frequency: `φ = -2πfτ`. Over a 500 MHz sweep with a few
metres of coax that is many full turns, and the resulting phase ramps can
swamp the phase structure of the resonances themselves.

This can be measured and compensated for in hardware.
`set_cable_length` tells the firmware how much delay to compensate, so the 
correction happens before you ever see the data.

`fit_cable_delay` measures the residual slope of the *unwrapped* phase and
converts it to a delay; `calculate_new_cable_length` adds the matching length
to the current setting and returns the new total.

```python
tau_additional = fit_cable_delay(frequencies, phase_degrees)

current_cable_length = await crs.get_cable_length(module=MODULE)
new_cable_length = calculate_new_cable_length(current_cable_length,
                                              tau_additional)
await crs.set_cable_length(length=new_cable_length, module=MODULE)

print(f"residual delay   {tau_additional*1e9:+.3f} ns")
print(f"cable length     {current_cable_length:.3f} m → {new_cable_length:.3f} m")
```

A simulated CRS has no cable: the fitted delay is close to zero and the length
barely changes.


```python
unwrapped_rad = np.unwrap(np.deg2rad(phase_degrees))
slope, intercept = np.polyfit(frequencies, unwrapped_rad, 1)
fit_deg = np.rad2deg(slope * frequencies + intercept)
residual_deg = np.rad2deg(unwrapped_rad) - fit_deg

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax1.plot(frequencies / 1e6, np.rad2deg(unwrapped_rad), lw=0.6,
         label="unwrapped phase")
ax1.plot(frequencies / 1e6, fit_deg, "k--", lw=1,
         label=f"fit: τ = {tau_additional*1e9:.3f} ns")
ax1.set_ylabel("phase (deg)"); ax1.legend()
ax1.set_title("Cable delay: the slope, and what is left without it")
ax2.plot(frequencies / 1e6, residual_deg, lw=0.6, color="#CC6633")
ax2.set_ylabel("residual (deg)"); ax2.set_xlabel("frequency (MHz)")
plt.tight_layout(); plt.show()
```

The delay is now set on the board, but the sweep in memory was taken before
that. Re-run section 3 for a corrected sweep. The resonance finding below uses
|S21|, which the delay does not affect, so the notebook continues with the
sweep it has.

## 5. Find the resonances

`find_resonances` looks for dips: it converts `|S21|**data_exponent` to dB,
runs `scipy.signal.find_peaks` on the negated trace, and keeps the peaks that
pass the depth, width and separation cuts below.

The parameters are all rejection criteria, and each one has a failure mode in
both directions:

- **`min_dip_depth_db`**: how deep a dip must be to count. Too high misses
  shallow (overcoupled or low-Q) resonators; too low finds noise. For shallow
  arrays, 0.3 to 0.5 dB.
- **`min_Q` / `max_Q`**: converted into an allowed width for the dip. A feature
  broader than `min_Q` allows is not a resonator; one narrower than `max_Q`
  allows is a spike.
- **`min_resonance_separation_hz`**: of any group of dips closer than this,
  the deepest is kept. The list obeys the separation, but a kept dip can still
  have a discarded neighbour; `require_isolation=True` drops both instead.

```python
FIND_RES_PARAMS = {
    "min_dip_depth_db": 1.0,
    "min_Q": 1e4,
    "max_Q": 1e7,
    "min_resonance_separation_hz": 100e3,
    "data_exponent": 2.0,
}

resonance_result = find_resonances(
    frequencies=frequencies,
    iq_complex=iq_complex,
    module_identifier=f"Module {MODULE}",
    **FIND_RES_PARAMS,
)

resonance_frequencies = resonance_result["resonance_frequencies"]
resonance_details = resonance_result["resonances_details"]

print(f"found {len(resonance_frequencies)} resonances\n")
for i, (freq, det) in enumerate(zip(resonance_frequencies, resonance_details), 1):
    print(f"  {i:2d}: {freq/1e6:9.3f} MHz   Q≈{det['q_estimated']:>9.0f}   "
          f"depth {det['prominence_db']:5.2f} dB   "
          f"width {det['width_hz']/1e3:6.1f} kHz")

if not resonance_frequencies:
    raise RuntimeError(
        "No resonances found. Lower min_dip_depth_db, widen the Q bounds, or "
        "check that the sweep range actually covers your array.")
```

```python
plt.figure(figsize=(11, 4))
plt.plot(frequencies / 1e6, mag_db, lw=0.6)
plt.plot(np.array(resonance_frequencies) / 1e6,
         np.interp(resonance_frequencies, frequencies, mag_db),
         "v", ms=9, color="#CC6633", label=f"{len(resonance_frequencies)} found")
plt.xlabel("frequency (MHz)"); plt.ylabel("|S21| (dB)")
plt.title("Detected resonances"); plt.legend()
plt.tight_layout(); plt.show()
```

## 6. Multisweep

The wide sweep located the resonators, but has not resolved them. At 50,000 points
across 500 MHz there is one point every 10 kHz, and a Q of 10⁵ at 1 GHz has a
linewidth of 10 kHz: the whole resonance is a couple of samples.

`multisweep` gives **one channel per resonator** and sweeps them all at once
over a narrow span. The cell below uses 101 points across 200 kHz, 2 kHz per
point: five samples across a 10 kHz linewidth instead of one. The tones are
simultaneous, so the whole array costs about the time of one sweep.

It also picks a bias frequency at this drive power. `bias_frequency_method`
decides where:

- **`"max-diq"`** (default): the point of steepest IQ motion, |d(I+jQ)/df|,
  where a small frequency shift produces the largest change in the signal.
- **`"min-s21"`**: the bottom of the dip. Not where responsivity peaks.
- **`None`**: keep the frequency you asked for.

```python
MULTISWEEP_PARAMS = {
    "span_hz": 200e3,           # Periscope's multisweep defaults: 200 kHz span,
    "npoints_per_sweep": 101,   # 101 points, 2 kHz per point
    "amp": 0.001,
    "nsamps": 10,
    "module": MODULE,
    "bias_frequency_method": "max-diq",
    "rotate_saved_data": False,
    "sweep_direction": "upward",
}

_shown[0] = -25.0
def sweep_progress(module, percentage):
    if percentage - _shown[0] >= 25.0:
        _shown[0] = percentage
        print(f"  sweeping… {percentage:.0f}%")

multisweep_results = await crs.multisweep(
    center_frequencies=resonance_frequencies,
    progress_callback=sweep_progress,
    **MULTISWEEP_PARAMS,
)

print(f"\n{len(multisweep_results)} resonances swept")
```

The result is keyed by **detector index** (1-based, matching the channel each
resonator was assigned), not by frequency. The frequencies are inside each
entry.

```python
det_ids = sorted(k for k in multisweep_results if isinstance(k, (int, np.integer)))
first = multisweep_results[det_ids[0]]

print(f"detector indices: {det_ids}")
print(f"\nkeys for detector {det_ids[0]}:")
for key in sorted(first):
    val = first[key]
    described = (repr(val) if isinstance(val, (str, bytes))
                 or not hasattr(val, "__len__")
                 else f"array{np.shape(val)}")
    print(f"  {key:<32} {described}")
```


```python
n_show = min(6, len(det_ids))
fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5))
for col, det in enumerate(det_ids[:n_show]):
    d = multisweep_results[det]
    f_off = (d["frequencies"] - d["original_center_frequency"]) / 1e3
    mag = 20 * np.log10(np.maximum(np.abs(d["iq_complex"]), 1e-30))
    axes[0, col].plot(f_off, mag, lw=1)
    axes[0, col].axvline(
        (d["bias_frequency"] - d["original_center_frequency"]) / 1e3,
        color="#CC6633", lw=1, ls="--")
    axes[0, col].set_title(f"det {det}", fontsize=9)
    axes[0, col].tick_params(labelsize=7)
    axes[1, col].plot(d["iq_complex"].real, d["iq_complex"].imag, lw=1)
    axes[1, col].set_aspect("equal", "datalim")
    axes[1, col].tick_params(labelsize=7)
axes[0, 0].set_ylabel("|S21| (dB)")
axes[1, 0].set_ylabel("Q")
fig.supxlabel("offset from center (kHz)   /   I", fontsize=9)
fig.suptitle("Multisweep: |S21| with the chosen bias point, and the IQ circle")
plt.tight_layout(); plt.show()
```

## 7. Fit the resonances

Two fits:

**The skewed Lorentzian** (`fit_skewed_multisweep`) is the standard resonator
model with a complex coupling quality factor, which makes the dip asymmetric:
real feedlines have impedance mismatches, and a symmetric model absorbs that
asymmetry into a wrong `fr`. It returns `fr`, `Qr` (loaded), `Qc` (coupling),
`Qi` (internal) and their uncertainties. `Qi` tells you about the film; `Qc` is
set by your design.

**The nonlinear fit** (`fit_nonlinear_iq_multisweep`) adds the parameter that
matters for choosing drive power: `a`, the nonlinearity. As you drive a KID
harder the resonance skews and then bifurcates: the frequency it sits at
depends on which way you swept. Above `a = 0.77` you are in that regime, and
`bias_kids` rejects such amplitudes.

```python
FIT_PARAMS = {
    "approx_Q_for_fit": 1e4,
    "fit_resonances": True,
    "center_iq_circle": True,
    "normalize_fit": True,
}

multisweep_results = fit_skewed_multisweep(multisweep_results, **FIT_PARAMS)

multisweep_results = fit_nonlinear_iq_multisweep(
    multisweep_results, fit_nonlinearity=True, n_extrema_points=5,
    verbose=False)

def fitted(det):
    """Skewed-fit params for a detector, or None if the fit failed."""
    p = multisweep_results[det].get("fit_params") or {}
    return p if p.get("fr") not in (None, "nan") else None

n_ok = sum(1 for d in det_ids if fitted(d))
print(f"skewed fits: {n_ok}/{len(det_ids)} converged\n")
print(f"{'det':>4} {'fr (MHz)':>12} {'Qr':>10} {'Qc':>10} {'Qi':>12} {'a':>7}")
for det in det_ids:
    p = fitted(det)
    if p is None:
        print(f"{det:>4}   fit failed")
        continue
    nl = multisweep_results[det].get("nonlinear_fit_params") or {}
    a = nl.get("a")
    a_str = f"{a:7.3f}" if isinstance(a, (int, float, np.floating)) else "      –"
    print(f"{det:>4} {p['fr']/1e6:12.4f} {p['Qr']:10.0f} {p['Qc']:10.0f} "
          f"{p['Qi']:12.0f} {a_str}")
```


```python
frs = np.array([fitted(d)["fr"] for d in det_ids if fitted(d)])
qrs = np.array([fitted(d)["Qr"] for d in det_ids if fitted(d)])
qcs = np.array([fitted(d)["Qc"] for d in det_ids if fitted(d)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
ax1.plot(frs / 1e6, qrs, "o", label="Qr (loaded)")
ax1.plot(frs / 1e6, qcs, "s", label="Qc (coupling)", alpha=0.7)
ax1.set_xlabel("resonance frequency (MHz)"); ax1.set_ylabel("Q")
ax1.set_yscale("log"); ax1.legend(fontsize=8)
ax1.set_title("Quality factors across the array")

spacing = np.diff(np.sort(frs)) / 1e6
ax2.bar(range(len(spacing)), spacing)
ax2.set_xlabel("gap index (sorted by frequency)")
ax2.set_ylabel("spacing (MHz)")
ax2.set_title("Spacing between neighbours")
plt.tight_layout(); plt.show()
```

## 8. Bias the KIDs

`bias_kids` biases each resonator: it picks an operating point and programs the
channel frequency and amplitude. It can also rotate the IQ basis to maximize
the signal in Q (a proxy for the df basis), with `optimize_phase=True`.

`fit_method` names the resonance fit it works from, `"nonlinear"` (default) or
`"skewed"`, and runs it on any sweep that does not already carry it. Given
sweeps at several amplitudes it chooses the **highest amplitude that is not
bifurcated and has `a` below `nonlinear_threshold`** (0.77). With one
amplitude, as here, that amplitude is used. The bias frequency is the
multisweep's `max-diq` or `min-s21` point read off the fitted curve rather than
the raw sweep grid, and the tone is programmed at the nearest multiple of the
298 Hz tone grid.

It also returns **`df_calibration`**, a complex number in hertz per volt:
multiply the IQ motion in volts by it to get frequency shift plus j times
dissipation. Pulse capture uses the same number to report pulse heights in Hz.
By default (`measure_calibration=True`) then verifies this through direct measurement:
every biased tone steps down, then up, together, by
`calibration_step` (0.05) of its fitted linewidth rounded to the tone grid.
This cell briefly moves the tones. The fit's own value is kept as `df_calibration_fit`, and
`df_calibration_source` says which one `df_calibration` is. Pass
`measure_calibration=False` to use the fit's.

```python
_shown[0] = -25.0
def bias_progress(module, percentage):
    if percentage - _shown[0] >= 25.0:
        _shown[0] = percentage
        print(f"  biasing… {percentage:.0f}%")

bias_results = await bias_kids(
    crs=crs,
    multisweep_results=multisweep_results,
    module=MODULE,
    progress_callback=bias_progress,
)

n_biased = sum(1 for d in bias_results.values() if d.get("bias_successful"))
print(f"\n{n_biased}/{len(bias_results)} detectors biased\n")
print(f"{'det':>4} {'ch':>3} {'bias freq (MHz)':>16} {'offset (kHz)':>13} "
      f"{'|df_cal| (Hz/V)':>16} {'source':>9}")
for det in sorted(bias_results):
    d = bias_results[det]
    offset = (d["bias_frequency"] - d["original_center_frequency"]) / 1e3
    cal = d.get("df_calibration")
    cal_str = f"{abs(cal):16.3e}" if cal is not None else " " * 16
    print(f"{det:>4} {d.get('bias_channel', '?'):>3} "
          f"{d['bias_frequency']/1e6:16.4f} {offset:13.2f} {cal_str} "
          f"{d.get('df_calibration_source', ''):>9}")
```

The offsets show how far `max-diq` moved the bias frequency from the dip that
`find_resonances` reported. An offset that is a large fraction of the sweep
span means the sweep did not contain its own resonance: `span_hz` is too
small, or the wide sweep mislocated it.

## 9. Noise on the biased detectors

With the detectors biased, the readout is now sensing the detector response.
`py_get_samples` collects a timestream from the slow (decimated readout) stream
and can return the spectrum with it.

`reference="absolute"` gives dBm/Hz, an absolute power spectral density, rather
than dBc/Hz relative to the carrier. `nsegments` sets the Welch averaging: more
segments, smoother spectrum, coarser frequency resolution.

```python
SAMPLE_PARAMS = {
    "num_samples": 1000,
    "return_spectrum": True,
    "scaling": "psd",
    "reference": "absolute",
    "nsegments": 5,
    "spectrum_cutoff": 0.9,
    "channel": None,        # every channel on the module
    "module": MODULE,
}

slow_data = await crs.py_get_samples(**SAMPLE_PARAMS)

freq_iq = np.asarray(slow_data.spectrum.freq_iq)
print(f"{SAMPLE_PARAMS['num_samples']} samples per channel, "
      f"spectrum to {freq_iq.max():.1f} Hz\n")
print(f"{'det':>4} {'mean I PSD':>14} {'mean Q PSD':>14}   (dBm/Hz, DC removed)")
for det in sorted(bias_results):
    idx = bias_results[det].get("bias_channel", det) - 1
    psd_i = np.asarray(slow_data.spectrum.psd_i[idx])
    psd_q = np.asarray(slow_data.spectrum.psd_q[idx])
    print(f"{det:>4} {np.mean(psd_i[2:]):14.2f} {np.mean(psd_q[2:]):14.2f}")
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
for det in sorted(bias_results)[:4]:
    idx = bias_results[det].get("bias_channel", det) - 1
    ax1.plot(np.asarray(slow_data.i[idx])[:300], lw=0.8, label=f"det {det}")
    ax2.semilogx(freq_iq[1:], np.asarray(slow_data.spectrum.psd_i[idx])[1:],
                 lw=0.9, label=f"det {det}")
ax1.set_xlabel("sample"); ax1.set_ylabel("I (V)")
ax1.set_title("Timestream"); ax1.legend(fontsize=8)
ax2.set_xlabel("frequency (Hz)"); ax2.set_ylabel("PSD (dBm/Hz)")
ax2.set_title("Noise spectrum, I"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
```

### The fast (PFB) stream

`py_get_pfb_samples` reads one channel at the full PFB rate of 2.44 MHz. At
decimation stage 6 the slow stream runs at 596 Hz, so this is 4096 times
faster, which is what resolves a fast pulse rise. It applies the PFB droop
correction before computing the spectrum.

```python
if IS_MOCK:
    print("Simulated PFB samples are uniform noise, not detector output;\n"
          "the call is exercised below, but the numbers measure nothing.")

pfb_channel = sorted(bias_results)[0]
pfb_data = await crs.py_get_pfb_samples(
    20_000 if IS_MOCK else 100_000,
    channel=bias_results[pfb_channel].get("bias_channel", pfb_channel),
    module=MODULE, binlim=1e6, trim=False, nsegments=5,
    reference="absolute", reset_NCO=False)

pfb_freq = np.asarray(pfb_data.spectrum.freq_iq)
pfb_psd_i = np.asarray(pfb_data.spectrum.psd_i)
print(f"\ndet {pfb_channel}: bandwidth to {pfb_freq.max()/1e3:.0f} kHz, "
      f"mean I PSD {np.mean(pfb_psd_i[2:]):.2f} dBm/Hz")
```

Pulse detection on these streams (triggering, per-pulse metrics, streaming
HDF5) is covered by `pulse_capture.md` in this folder. It starts where this
notebook ends: with biased detectors.

## 10. Keep the results

The tuning is on the board, but the characterization is only in this kernel.
Saving `bias_results` gives you the frequencies, the fits and the calibrations
without repeating the sweep.

```python
out_path = OUTPUT_DIR / "tuning_results.pkl"
with open(out_path, "wb") as f:
    pickle.dump({
        "netanal": netanal,
        "resonance_frequencies": resonance_frequencies,
        "multisweep_results": multisweep_results,
        "bias_results": bias_results,
    }, f)

print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

# df_calibration is what pulse capture needs to report pulse heights in Hz:
df_cals = {d.get("bias_channel", det): d["df_calibration"]
           for det, d in bias_results.items()
           if d.get("df_calibration") is not None}
print(f"df calibrations for {len(df_cals)} channels; pass these to "
      f"crs.trigger_capture(df_calibrations=…) or PulseCaptureSession")
```

## 11. Where this maps in Periscope

The Periscope panels call the same functions this notebook does:

| Periscope control | API equivalent |
|---|---|
| **Network Analysis** panel | `crs.take_netanal(...)` |
| *Unwrap Cable Delay* button | `fit_cable_delay` → `crs.set_cable_length(...)` |
| *Find Resonances* + its dialog | `find_resonances(...)` |
| **Multisweep** panel | `crs.multisweep(...)` |
| *Apply Skewed Fit* / *Apply Nonlinear Fit* in the Multisweep dialog | `fit_skewed_multisweep`, `fit_nonlinear_iq_multisweep` |
| **Bias KIDs** dialog | `bias_kids(...)` |
| *Get Noise Spectrum* button in the Multisweep panel | `crs.py_get_samples(return_spectrum=True)` |
| *Histograms* tab of the Multisweep panel | the `fit_params` distributions in section 7 |
| Progress bars | the `progress_callback=` hook on every long call |

`simplified_tuning_flow.py` in this folder runs the same sequence as a plain
script, to copy from or to run against MOCK:

    python simplified_tuning_flow.py MOCK      # simulated CRS
    python simplified_tuning_flow.py 0042      # real board

```python
# Only tear down a simulation this notebook started. If section 1 attached to
# Periscope's CRS, stopping its streamer would kill Periscope's live plots.
if IS_MOCK:
    await crs.stop_udp_streaming()
    print("simulated streamer stopped")
else:
    print("left the board as it is: biased, and not ours to tear down")
```
