# MockCRS Hardware Emulation System

This package emulates the CRS and its KIDs for testing and development without hardware.

## File Structure

```
rfmux/mock/
├── __init__.py          # Public API and flavour hook (yaml_hook)
├── config.py            # Every default, in one place
├── crs.py               # MockCRS class - core device emulation
├── server.py            # Tuber HTTP server and process management
├── resonator_model.py   # Physics-based KID resonator simulation
├── udp_streamer.py      # UDP packet streaming for real-time data
├── tls_noise.py         # TLS 1/f frequency wander
├── helpers.py           # Helper functions for resonator generation
├── standard_array.py    # The seeded array tests and notebooks share
└── README.md            # This file
```

## Module Descriptions

### `__init__.py`
Entry point for the mock flavour system. Provides `yaml_hook()` which is called when `!flavour "rfmux.mock"` is specified in a hardware map YAML file.

### `config.py`
Every MockCRS default, in one place. Contains:
- `MOCK_DEFAULTS`: Dictionary of all default parameter values
- `defaults()`: Returns a deep copy of defaults
- `apply_overrides()`: Merges user configuration with defaults

No other module defines a mock default.

### `crs.py`
The main `MockCRS` class that emulates CRS hardware. Provides:
- All CRS methods (set_frequency, get_samples, etc.)
- State management (frequencies, amplitudes, phases)
- Enum definitions (Units, Target, ClockSource, etc.)
- Integration with resonator physics and UDP streaming

### `server.py`
Handles the HTTP server for Tuber protocol communication:
- `yaml_hook()`: Sets up mock servers for each CRS in the hardware map
- `ServerProcess`: Runs the aiohttp server in a separate process
- Request routing and response serialization

### `resonator_model.py`
Physics-based simulation of Kinetic Inductance Detectors (KIDs):
- Uses `MR_LEKID` from `mr_resonator` for accurate physics
- Handles power-dependent frequency shifts
- Self-consistent iterative convergence for nonlinear effects
- Quasiparticle pulse simulation (periodic, random, manual modes)

### `udp_streamer.py`
Real-time data streaming via UDP:
- Generates packets matching real CRS format
- Supports multicast and unicast modes
- Configurable sample rates and channel counts

### `helpers.py`
Utility functions for resonator generation and configuration.

## Usage

### Basic Setup (Hardware Map YAML)
```yaml
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "MOCK0001" }
```

### Python Usage
```python
import rfmux

# Load session with mock hardware
s = rfmux.load_session("hardware_map.yaml")
crs = s.query(rfmux.CRS).one()
await crs.resolve()

# Use CRS methods as normal
await crs.set_nco_frequency(1.2e9, module=1)
await crs.set_frequency(10e6, channel=1, module=1)
await crs.set_amplitude(0.01, channel=1, module=1)
samples = await crs.get_samples(100, channel=1, module=1)
```

### Customizing Physics Configuration
```python
# Generate resonators with custom parameters
await crs.generate_resonators(config={
    'num_resonances': 10,
    'freq_start': 1.1e9,
    'freq_end': 1.5e9,
    'T': 0.12,            # Temperature [K]
    'Popt': 1e-13,        # Optical power [W]
    'auto_bias_kids': True,
})
```

### UDP Streaming
```python
# Start streaming
await crs.start_udp_streaming()   # multicast, or loopback unicast if the host cannot

# Check status
status = await crs.get_udp_streaming_status()

# Stop streaming
await crs.stop_udp_streaming()
```

### Quasiparticle Pulses
```python
# Enable periodic pulses
await crs.set_pulse_mode('periodic',
    period=2.0,
    amplitude=2.0,
    tau_decay=5e-3)
# From a config dict, with rfmux.mock.helpers.pulse_mode_kwargs:
await crs.set_pulse_mode(cfg['pulse_mode'], **pulse_mode_kwargs(cfg))

# Add manual pulse event
await crs.add_pulse_event(resonator_index=0, start_time=time.time(), amplitude=3.0)
```

## Configuration Parameters

Every parameter the simulator takes is a key of `MOCK_DEFAULTS` in
`config.py`; there is no other place a default lives. Pass a dict with the
keys you want changed and the rest keep their defaults. Four ways in:

```python
# A fresh array, headless
await crs.generate_resonators(config={"num_resonances": 20, "T": 0.2})

# A running array: pulse keys are taken live, anything else regenerates,
# and only what differs from the configuration in force counts as changed
from rfmux.mock.helpers import apply_mock_config
await apply_mock_config(crs, {"udp_noise_level": 0.0})

# The shared seeded array tests and notebooks use, with overrides on top
from rfmux.mock.standard_array import standard_array
crs, catalog = await standard_array({"tls_noise_enabled": False})

# What the array was built with
cfg = await crs.get_mock_configuration()
```

In Periscope the same keys are the fields of the Mock Configuration dialog
(noise, physics, circuit, readout and solver fields sit behind the Advanced
toggle), and the dialog's values are saved with the session so a reloaded
session rebuilds the same array.

`apply_overrides()` parses numeric strings, normalizes `pulse_resonators`,
and clamps the pulse distributions and TLS parameters to the ranges given
below. It does not reject unknown keys.

### Resonator distribution

| Key | Default | Meaning |
|---|---|---|
| `num_resonances` | 5 | How many resonators to place. |
| `freq_start`, `freq_end` | 1.0e9, 1.5e9 Hz | The band the resonances are spread across. |
| `resonator_random_seed` | None | Seed for the placement and the per-resonator spreads. None draws a fresh array each build; set an int to get the same array every time. `apply_mock_config` pins one before regenerating so client and server agree. |

### Physics

Temperature and optical power set the quasiparticle density `nqp`, which
sets the kinetic inductance and loss of every resonator, and through them
the resonant frequency and Q. Lower `T` or `Popt` means fewer quasiparticles
and higher Q: 0.12 K gives Q near 3e5 with the default geometry, 0.23 K
near 5e4.

| Key | Default | Meaning |
|---|---|---|
| `T` | 0.12 K | Bath temperature. |
| `Popt` | 1e-15 W | Optical power absorbed per resonator. |
| `material` | "Al" | Kept for the record; the physics library accepts only aluminium by name and warns on anything else. Other superconductors are described by the optional keys `material_Tc` (K), `material_N0` (µm⁻³ eV⁻¹), `material_tau0` (s) and `material_sigmaN`, which override the aluminium values when present. |
| `width`, `thickness`, `length` | 2 µm, 30 nm, 9 mm | Inductor strip geometry, which sets its volume and so how much a given `nqp` changes the inductance. |

### Circuit

| Key | Default | Meaning |
|---|---|---|
| `Lg` | 10e-9 H | Geometric inductance, in series with the kinetic inductance. |
| `Cc` | 0.01e-12 F | Coupling capacitor to the feedline; larger couples more strongly (lower Qc, deeper dip). |
| `L_junk` | 0 H | Parasitic inductance in the coupling path. |
| `C_variation`, `Cc_variation` | 0.01, 0.01 | Fractional standard deviation of the per-resonator draw of C and Cc, so the array is not eight copies of one resonator. |

### Readout chain

| Key | Default | Meaning |
|---|---|---|
| `Vin` | 1e-5 V | Drive voltage used for the reference operating point when the array is built. The response to a tone uses the tone's own amplitude. |
| `input_atten_dB` | 10 dB | Attenuation between the drive and the feedline. Applied inside the S21 calculation; do not apply it again. |
| `system_termination` | 50 Ω | Feedline impedance. |
| `ZLNA`, `GLNA` | 50 Ω, 10 dB (as a voltage ratio) | Amplifier input impedance and gain after the feedline. |
| `scale_factor` | about 2.58e6 | Converts normalized S21 × tone amplitude to ADC counts, calibrated so counts × VOLTS_PER_ROC is the physical voltage at the default DAC scale. Not usually changed. |

### Noise

Three independent sources, each switchable. With all three off the array
is noise free, which is the right setting for a test that checks a value
to many digits and the wrong one for a test that has to survive a board.

| Key | Default | Meaning |
|---|---|---|
| `nqp_noise_enabled` | True | White fluctuation of the quasiparticle density, drawn fresh at every evaluation. It scatters the resonator response sample to sample and is uncorrelated between samples and between resonators. |
| `nqp_noise_std_factor` | 0.01 | Its standard deviation as a fraction of the base `nqp`. |
| `tls_noise_enabled` | True | Two-level-system frequency wander: a fractional frequency drift with a 1/f^alpha spectrum. Unlike the QP noise it is correlated in time, so it moves the baseline rather than adding scatter. The slow and PFB streams see the same wander at the same instant, as one resonator would. |
| `tls_fractional_rms` | 1e-7 | RMS of the fractional frequency deviation df/f. Clamped to 0 or more. |
| `tls_alpha` | 1.0 | Spectral slope; 1 is pink, TLS is often quoted nearer 0.5. Clamped to [0, 2]. |
| `tls_corner_hz` | 100 Hz | Upper corner of the power law; the law spans three decades below it and rolls off above. Clamped to [1e-3, 1e5]. |
| `udp_noise_level` | 11.0 counts | Additive white readout noise, the standard deviation per slow sample in ADC counts, applied to `get_samples` and the slow stream alike. Measured on a board with no tone through a detector chain. The PFB stream scales its own sigma from this to match a board's PFB floor. |

### Bias

| Key | Default | Meaning |
|---|---|---|
| `auto_bias_kids` | False | After building, park one tone on each resonance so a caller can sweep or stream at once. |
| `bias_amplitude` | -55 dBm, stored normalized | Amplitude of those tones. `bias_amplitude_from_dbm()` in `config.py` converts; the dialog shows it in dBm. |

### Quasiparticle pulses

A pulse raises a resonator's `nqp` by `(amplitude - 1) × base` at its peak,
with an exponential rise of `tau_rise` and decay of `tau_decay`. Pulse
settings apply to a running array without rebuilding it.

| Key | Default | Meaning |
|---|---|---|
| `pulse_mode` | "none" | `"periodic"` fires every `pulse_period` seconds on each target resonator; `"random"` fires with probability `pulse_probability × dt` at each time step, a mean rate of `pulse_probability` per second per resonator; `"manual"` fires only what `add_pulse_event` adds; `"none"` fires nothing. |
| `pulse_period` | 2.0 s | Interval, periodic mode. |
| `pulse_probability` | 0.1 /s | Mean pulse rate per resonator, random mode. |
| `pulse_tau_rise` | 1e-6 s | Rise time, fixed for every pulse. |
| `pulse_tau_decay` | 5e-3 s | Decay time when the tau distribution is fixed. |
| `pulse_amplitude` | 2.0 | Peak `nqp` as a multiple of base when the amplitude distribution is fixed. Values below 1 are raised to 1: a pulse never removes quasiparticles. |
| `pulse_resonators` | "all" | Which resonators fire: `"all"`, a list of 0-based indices, or a comma-separated string of them. |

Periodic and random modes can draw each pulse's amplitude and decay time
from a distribution instead of using the fixed values:

| Key | Default | Meaning |
|---|---|---|
| `pulse_random_amp_mode` | "fixed" | `"fixed"`, `"uniform"` between the min and max, or `"lognormal"` with the given log-mean and log-sigma. |
| `pulse_random_amp_min`, `pulse_random_amp_max` | 1.1, 1.5 | Uniform bounds, each clamped to at least 1 and max to at least min. |
| `pulse_random_amp_logmean`, `pulse_random_amp_logsigma` | 0.7, 0.3 | Lognormal parameters; sigma clamped to 0 or more. |
| `pulse_random_tau_mode` | "fixed" | As above, for `tau_decay`. |
| `pulse_random_tau_min`, `pulse_random_tau_max` | 0.5 ms, 5 ms | Uniform bounds, each forced positive and max to at least min. |
| `pulse_random_tau_logmean`, `pulse_random_tau_logsigma` | -6.9, 0.5 | Lognormal parameters; -6.9 is ln(1 ms), so the median is 1 ms. |

### Solver internals

These trade accuracy for speed in the self-consistent solve of the
nonlinear response and are not knobs a user of the array needs.

| Key | Default | Meaning |
|---|---|---|
| `convergence_tolerance` | 1e-9 | Fractional tolerance of the operating-point iteration. Loosen toward 1e-5 to speed up an array of many tones. |
| `cache_freq_step`, `cache_amp_step`, `cache_qp_step` | 1e-4 Hz, 1e-8, 1e-4 | Quantization of frequency, amplitude and fractional `nqp` when looking up a converged state, so a repeat of a nearly identical evaluation reuses the last one. |
| `convergence_cache_max_size` | 1e7 | Cache entry cap. |
| `log_cache_decisions`, `cache_log_interval` | False, 100 | Print every Nth cache decision. |
| `physics_batch_mode` | "hoisted" | How a block of samples is evaluated. `"reference"` is the sample-by-sample loop the fast path is tested against. |

## Integration with Periscope

`periscope MOCK` starts the simulator behind the GUI and enables the Mock
Configuration dialog.
