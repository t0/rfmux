# Periscope

Periscope is a real-time multi-pane viewer and network analysis tool for CRS packets.

## Overview

The Periscope application provides functionalities for:
- Real-time data visualization of Time-domain waveforms (TOD), IQ (density or scatter), FFT, Single-sideband PSD (SSB), and Dual-sideband PSD (DSB).
- Network Analysis (amplitude and phase vs frequency).
- Multi-channel grouping and display.
- Configuration options for buffer sizes, refresh rates, display units, and plot scaling.
- An embedded iPython console for interactive sessions.

## Structure

The main components of the Periscope tool are organized within this `periscope` subpackage:
- `__main__.py`: Command-line entry point (`python -m rfmux.tools.periscope`) and programmatic entry (`raise_periscope`).
- `app.py`: Contains the main `Periscope` Qt application class.
- `utils.py`: Core utilities, constants, and helper classes.
- `tasks.py`: Background worker threads for data processing and hardware interaction.
- `dialogs.py`: Various `QDialog` classes for user input and configuration.
- `ui.py`: Aggregates UI components from `dialogs.py`, `network_analysis_window.py`, and `multisweep_window.py`.
- `network_analysis_window.py`: Defines the `QMainWindow` for displaying network analysis results.
- `multisweep_window.py`: Defines the `QMainWindow` for displaying multisweep results.
- `icons/`: Contains graphical assets like the application icon.

## Usage

### Command-Line
```bash
python -m rfmux.tools.periscope <hostname> [options]
```
Example:
```bash
python -m rfmux.tools.periscope rfmux0022.local --module 2 --channels "3&5,7"
```
Run with `-h` or `--help` for more options.

### Programmatic (e.g., in IPython/Jupyter)
```python
from rfmux.tools.periscope import raise_periscope
from rfmux import CRS # Assuming CRS object is obtained elsewhere

# Example:
# crs_instance = CRS(...) 
# await crs_instance.resolve() # If needed
# await raise_periscope(crs_instance, module=2, channels="3&5")
