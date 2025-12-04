# MockCRS Hardware Emulation System

This package provides a complete software emulation of the CRS and KIDs for testing and development without physical hardware.

## File Structure

```
rfmux/mock/
├── __init__.py          # Public API and flavour hook (yaml_hook)
├── config.py            # Single Source of Truth for all configuration
├── crs.py               # MockCRS class - core device emulation
├── server.py            # Tuber HTTP server and process management
├── resonator_model.py   # Physics-based KID resonator simulation
├── udp_streamer.py      # UDP packet streaming for real-time data
├── helpers.py           # Helper functions for resonator generation
└── README.md            # This file
```

## Module Descriptions

### `__init__.py`
Entry point for the mock flavour system. Provides `yaml_hook()` which is called when `!flavour "rfmux.mock"` is specified in a hardware map YAML file.

### `config.py`
**Single Source of Truth (SoT)** for all MockCRS configuration parameters. Contains:
- `MOCK_DEFAULTS`: Dictionary of all default parameter values
- `defaults()`: Returns a deep copy of defaults
- `apply_overrides()`: Merges user configuration with defaults

All other modules import configuration from here - no other module should define mock defaults.

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
- !CRS { serial: "0000", hostname: "127.0.0.1" }
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
crs.set_frequency(10e6, channel=1, module=1)
crs.set_amplitude(0.01, channel=1, module=1)
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
await crs.start_udp_streaming(host='127.0.0.1', port=9876)

# Check status
status = crs.get_udp_streaming_status()

# Stop streaming
await crs.stop_udp_streaming()
```

### Quasiparticle Pulses
```python
# Enable periodic pulses
await crs.set_pulse_mode('periodic', 
    pulse_period=2.0,
    pulse_amplitude=2.0,
    pulse_tau_decay=0.1)

# Add manual pulse event
await crs.add_pulse_event(resonator_index=0, start_time=time.time(), amplitude=3.0)
```

## Configuration Parameters

See `config.py` for the complete list of parameters. Key categories:

- **Resonator distribution**: num_resonances, freq_start, freq_end
- **Physics**: T (temperature), Popt (optical power), Lg, Cc
- **Readout**: Vin, input_atten_dB, system_termination
- **Convergence**: convergence_tolerance, cache settings
- **UDP streaming**: udp_noise_level, scale_factor
- **Pulses**: pulse_mode, pulse_period, pulse_amplitude, pulse_tau_decay

## Integration with Periscope

The mock system integrates with Periscope GUI for visualization. When using mock mode, Periscope will:
1. Detect mock hardware automatically
2. Enable mock configuration dialog
