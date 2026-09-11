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
4. Use "Find Resonances" to identify resonance dips; the ⚙ beside it sets the
   thresholds, and they stay set between searches and across sessions
5. Rejected candidates are marked with a cross -- hover one for why it went
6. Use "Unwrap Cable Delay" to compensate for cable length effects
7. Save writes the measurement, and any search in it, to the session folder

### Multisweep Analysis

For high-resolution analysis around identified resonance frequencies:

1. First identify resonances using the Network Analysis feature
2. Click "Take Multisweep" to configure a detailed sweep around resonances
3. Click "Run Fit". Progress is beside the button, and what it did is said
   there in green when it finishes. The ⚙ holds the settings, which persist
   between sessions: which models to fit (skewed, nonlinear, or both), which
   sweeps -- all of them, each resonator at the amplitude it is biased at, or
   one amplitude step -- and which model the Fit Results tab draws
4. The Fit Results tab draws each resonator's measurement with one model over
   it, whichever "Which model to draw" names of the models the sweeps carry
   fits for. The measurement keeps the colour it has on the other tabs -- its drive
   -- and the model is the black or white line over it; line style is the sweep
   direction, as elsewhere. The axis is normalized to each trace's last point,
   which is the fits' own convention, so the toolbar's "Normalize Traces" does
   not apply here
5. Fits are written into the sweeps themselves, so a file saved afterwards
   carries them and reopens with them; a measurement already saved is re-saved
   where it was

### Import and Export Data
Periscope provides several ways to reuse previously captured sweeps and to archive new measurements for offline study.

- **Network analysis parameter import**
  1. In the *Network Analysis* dialog, select **Import Data** to load a saved configuration from a `.pkl`/`.pickle` file created by an earlier run.
  2. The dialog non-blockingly opens a file chooser and, once a compatible payload is selected, pre-populates all sweep parameters (modules, amplitudes, frequency span, averaging, cable length, etc.).
  3. After the file is validated, the **Load Analysis** button becomes available, allowing you to immediately reuse the imported settings without re-entering them by hand.
  4. If the resonance values are available it will also re-plot the data in the same way as displayed when it was saved.

- **Network analysis data export**
  1. In the Network Analysis window toolbar, **Export Data** opens a save dialog where you can choose Pickle (`.pkl`) or CSV (`.csv`). The picker is non-blocking so live acquisition continues once the dialog is dismissed.
  2. Pickle exports contain a timestamp, the exact sweep parameters used, and a hierarchy of per-module measurements. For every sweep Periscope stores the raw frequency grid, magnitude in counts/volts/dBm (both raw and normalized), phase in degrees, the complex IQ samples, and any resonances that were identified.
  3. If find resonance was executed, the exported file will also contain resonance frequencies.

- **Multisweep import**
  1. When launching a multisweep with "Load Multisweep" enabled, use the **Import Sweep File** button to choose a previously exported multisweep results file (`.pkl`/`.pickle`).
  2. The dialog loads the saved parameters, resonance list, and fitted frequencies. You can toggle whether to seed the run with raw sweep targets or with the fitted center frequencies captured in the file, and the amplitude and sweep settings are filled in automatically.
  3. If no file is provided one can input frequencies manually and start the sweep.


- **Multisweep data export**
  1. The Multisweep window's 💾 button writes the measurement through `rfmux.tuning.store`, into the session folder. It is the container `multisweep` returned, so the same file opens in a notebook with `store.load`, and one a notebook wrote opens here.
  2. Saving the same panel again overwrites the same file: the container carries the path it was written to. Running the fitters or Find Bias on a measurement that is already on disk re-saves it where it was, since both leave their results in the block.

- **Find Bias**
  1. **Find Bias** chooses an operating amplitude and frequency for every resonator in the sweeps on screen, in one `find_bias_points` call. The ⚙ beside it opens a settings window grouped by the test each setting belongs to: which bifurcation test decides the amplitude, the thresholds that test reads, where in the chosen sweep the tone goes, and how far from the sweep centre an answer is believed. Settings persist between sessions, and **Reset to Defaults** puts back what the library does when you say nothing.
  2. The report becomes the panel's catalog, and the sweeps come back carrying it as `bias_report`. On the sweep grids the amplitude step each resonator is biased at is drawn thick, with a line at its bias frequency; the counts, and the names of any findings that are fallbacks rather than measurements, are on the status line, where a flagged run stays until the next one.
  3. The **Bias Diagnostics** tab draws what the derivative test looks at — the point-to-point change in each sweep's normalized arc speed — with every trace divided by the bar that trace faced, so the bar is one pair of lines at ±1 and a quiet step is visible beside a loud one. It draws before anything has been found, and follows the settings as they change, which is how you choose them.

- **Apply Bias**
  1. **Apply Bias** parks a tone on every resonator in the panel's catalog, at the frequency and amplitude it carries. Which NCO carries them, and putting the frequencies on the tone grid, are `apply_bias`'s doing.
  2. On success each channel's `df_calibration` is published to the main window, so the streams can be displayed in hertz.

- **Take Noise** 
  1. Opening the Detector Digest on a resonance also provides a **Take Noise** shortcut. The digest overlays the newly acquired noise timestream with the loaded rotation data, letting you confirm phase alignment, biasing and overall noise behavior without rerunning the full sweep. One can click it multiple times to see the evolution of noise in their data stream.
  2. Noise captures are tied to the currently selected detector, making it straightforward to iterate on the bias solution and immediately see the impact on the detector’s timestream statistics.

All import dialogs validate that the selected file contains the expected data structure and will notify you if a file is missing required sections, helping prevent accidental misuse of unrelated files

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
