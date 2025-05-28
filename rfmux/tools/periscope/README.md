# Periscope

Periscope is a real-time multi-pane viewer and network analysis tool for CRS (Control and Readout System) packets, designed for monitoring and analyzing Kinetic Inductance Detectors (KIDs).

## Overview

The Periscope application provides a comprehensive suite of visualization and analysis tools:

- **Real-time Data Visualization**:
  - Time-domain waveforms (TOD)
  - IQ visualization (density or scatter plots)
  - Fast Fourier Transform (FFT) analysis
  - Single-sideband Power Spectral Density (SSB PSD)
  - Dual-sideband Power Spectral Density (DSB PSD)

- **Network Analysis**:
  - Amplitude and phase vs frequency sweeps
  - Multi-amplitude sweeps for characterization
  - Cable delay calculation and compensation
  - Resonance frequency identification
  - Export of analysis data in various formats

- **Multisweep Analysis**:
  - High-resolution sweeps around identified resonance frequencies
  - Amplitude-dependent characterization
  - Detector parameter extraction

- **Configuration Options**:
  - Buffer sizes and refresh rates
  - Display units (counts, volts, dBm)
  - Plot scaling and zoom modes
  - Light/dark theme support
  - Multi-channel grouping and display

- **Interactive Features**:
  - Embedded iPython console for direct interaction with data
  - Customizable plot displays
  - Dynamic UI configuration

## Structure

The main components of the Periscope tool are organized within this `periscope` subpackage:

- `__main__.py`: Command-line entry point (`python -m rfmux.tools.periscope`) and programmatic entry (`raise_periscope`).
- `app.py`: Contains the main `Periscope` Qt application class that orchestrates all functionality.
- `app_runtime.py`: Contains runtime-specific functionality separated from the main app logic.
- `utils.py`: Core utilities, constants, and helper classes for Periscope's operation.
- `tasks.py`: Background worker threads for data processing and hardware interaction, implementing the concurrency model.
- `network_analysis_base.py`: Base classes and core functionality for network analysis.
- `network_analysis_window.py`: UI window for displaying network analysis results.
- `network_analysis_dialog.py`: Configuration dialog for network analysis.
- `network_analysis_export.py`: Functionality for exporting network analysis data and managing cable delays.
- `multisweep_dialog.py`: Configuration dialog for multisweep analysis.
- `multisweep_window.py`: UI window for displaying multisweep results.
- `find_resonances_dialog.py`: Dialog for configuring resonance identification parameters.
- `detector_digest_dialog.py`: Dialog for displaying detector parameters.
- `dialogs.py`: Various other `QDialog` classes for user input and configuration.
- `ui.py`: Aggregates UI components from dialog and window classes.
- `initialize_crs_dialog.py`: Dialog for CRS board initialization.
- `icons/`: Contains graphical assets like the application icon.

## Usage

### Command-Line
After installing the package with `pip install .` in the repository:
```bash
periscope <crs_board> [options]
```

Or directly as a Python module:
```bash
python -m rfmux.tools.periscope <crs_board> [options]
```

The CRS board identifier can be specified in three formats:
- A hostname in the format rfmux####.local (e.g., "rfmux0042.local")
- Just the serial number (e.g., "0042")
- A direct IP address (e.g., "192.168.2.100")

Options:
- `--module <module_num>`: Specify the module number (default: 1)
- `--channels <channel_spec>`: Channel specification, where multiple channels can be grouped with '&' and separated with ',' (default: "1")
- `--buffer <size>`: Buffer size for data acquisition (default: 5000)
- `--refresh <ms>`: GUI refresh interval in milliseconds (default: 33)
- `--dot-px <size>`: Dot diameter in pixels for IQ density display (default: 1)

Examples:
```bash
# Using hostname format
periscope rfmux0022.local --module 2 --channels "3&5,7"

# Using just the serial number
periscope 0022 --module 2 --channels "3&5,7"

# Using IP address
periscope 192.168.2.100 --module 2 --channels "3&5,7"
```

### Programmatic (e.g., in IPython/Jupyter)
```python
from rfmux.tools.periscope import raise_periscope
from rfmux import CRS # Assuming CRS object is obtained elsewhere

# Example usage:
crs_instance = CRS(...) 
await crs_instance.resolve() # If needed
await raise_periscope(crs_instance, module=2, channels="3&5")
```

## Key Features

### Network Analysis

The network analysis functionality allows for detailed characterization of resonators:

1. Click "Network Analyzer" to configure and run a frequency sweep
2. Set frequency range, sweep points, and amplitude parameters
3. View amplitude and phase response for each module
4. Use "Find Resonances" to automatically identify resonance frequencies
5. Use "Unwrap Cable Delay" to compensate for cable length effects
6. Export data in various formats for further analysis

### Multisweep Analysis

For high-resolution analysis around identified resonance frequencies:

1. First identify resonances using the Network Analysis feature
2. Click "Take Multisweep" to configure a detailed sweep around resonances
3. Analyze detector parameters and characteristics
4. Export data for detailed offline analysis

### Display Customization

Multiple customization options are available:

- Toggle between dark and light themes
- Select which plot types to display
- Choose between raw counts and real units (volts, dBm)
- Enable/disable auto-scaling for more focused analysis
- Group multiple channels for comparative visualization

## Development

When extending or modifying Periscope:

1. Follow the established coding style and patterns
2. Utilize the task system for background processing
3. Keep UI operations on the main thread
4. Implement proper error handling and user feedback
5. Maintain consistent documentation and type annotations

See the Memory Bank system patterns documentation for detailed coding guidelines.
