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
  - Amplitude and phase vs frequency sweeps, upward or downward through the band
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
2. Set the frequency range, the points and averaging, the probe amplitude, and
   which way through the band to measure
3. View amplitude and phase response for each module
4. Use "Find Resonances" to identify resonance dips; "Find Resonances Settings" sets the
   thresholds. Apply saves edits between searches and across sessions; Close
   discards unapplied edits. In the collision group, resonators at or below the
   frequency-separation threshold collide, including equality. Check **Disable
   collision cut** to skip this check entirely and grey out its controls. Apply
   remembers the disabled state. The default 0 kHz threshold only acts on
   identical frequencies; other resonance filters remain independent.
5. Rejected candidates are marked with a cross -- hover one for why it went
6. Use "Unwrap Cable Delay" to compensate for cable length effects
7. Save writes the measurement, and any search in it, to the session folder

### Multisweep Analysis

For high-resolution analysis around identified resonance frequencies:

1. First identify resonances using the Network Analysis feature
2. Click "Take Multisweep" to configure a detailed sweep around resonances
3. Click "Run Fit". Progress is beside the button, and `Fits complete` is said
   there in green when it finishes — what the fits found, and which of them did
   not converge, is on the tabs that draw them. "Fit Settings" holds the settings, which persist
   between sessions: which models to fit (skewed, nonlinear, or both) and which
   sweeps -- all of them, each resonator at the amplitude it is biased at, or
   one amplitude step. Apply saves edits; Close discards unapplied edits.
4. The Fit Results tab draws each resonator's measurement with one model over
   it. Its own toolbar says which: "Fit" offers the models the sweeps carry
   fits for, and "Amplitude" which sweeps are drawn -- all of them, one
   amplitude step, or, once Find Bias has chosen one, each resonator at the
   step it is biased at. Both are remembered between sessions. The measurement
   keeps the line it has on the other tabs -- coloured by its drive, styled by
   its direction -- and the model is a thinner black or white line over it.
   Its legend always names the two lines; with few enough drives on screen to
   label -- one step of a schedule, say -- it names each by its drive and puts
   the model's headline numbers (fr, Qr, Qi, and the nonlinear fit's a) on the
   line they came from. With more, the colorbar carries the drives and the
   legend says "Measured" and "<model> fit" once. The axis is normalized to
   each trace's last point, which is the fits' own convention, so the
   toolbar's "Normalize Traces" does not apply here
5. The Fit Histograms tab answers the same fits over the whole array rather
   than one resonator at a time: a scatter of fitted `fr` sorted in ascending
   frequency, with each dot coloured by its fitted `Qr` on a linear colour
   scale, and the quality factors binned on one shared set of log bins so they
   can be read against each other. The nonlinear model
   puts `a` beside a line at the nonlinearity where bifurcation starts. Each
   drive gets its own outline in its own colour. Choose one amplitude step
   or "At bias amplitude"; all amplitudes is not offered. The default is bias
   amplitude when available, otherwise step 0. A saved single-amplitude choice
   is retained. Its "Fit" and "Amplitude" choices are independent of Fit Results.
   It says above the plots how many fits are drawn and how many the fitters
   rejected and so are not binned
6. Fits are written into the sweeps themselves, so a file saved afterwards
   carries them and reopens with them; a measurement already saved is re-saved
   where it was

### Import and Export Data
Periscope provides several ways to reuse previously captured sweeps and to archive new measurements for offline study.

- **Network analysis parameter import**
  1. In the *Network Analysis* dialog, select **Import Data** to load a saved configuration from a `.pkl`/`.pickle` file created by an earlier run.
  2. The dialog non-blockingly opens a file chooser and, once a compatible netanal file is selected, fills the fields in with the arguments it was measured with -- amplitude, frequency range, points, averaging, max channels and span, and direction -- along with the name it was saved under. Not the module: one Periscope is one module, and the file is shown on the one it holds.
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

- **Collision Cut**
  1. In a completed or loaded Multisweep panel, open **Collision Cut**, choose the separation in kHz, and press **Run Collision Cut**. The check calls `rfmux.tuning.find_sweeps_with_nearby_resonances`: it finds pairs of dips within each fine sweep, including pairs exactly at the separation threshold. It runs only when requested. The initial separation is 100 kHz; enter `inf` to flag any second detected dip in the window.
  2. The defaults are 1 dB prominence, 10 Hz (0.01 kHz) minimum dip spacing, all amplitude steps and all directions. Keep dip spacing below the cut. Select an early amplitude step if high-drive bifurcation produces false hits. Neighbours outside the measured window cannot be detected.
  3. The **Collisions** tab shows all measured traces of the flagged resonators. **Remove collided resonators from catalog and re-sweep** opens the normal multisweep configuration with a copy of the current catalog containing only survivors. After Find Bias, this copy carries the current bias points. The previous sweep settings seed the dialog.
  4. **Start Multisweep** creates a separate measurement panel. With session auto-export enabled, it saves into the session like any multisweep; otherwise use its Save button. The original panel and saved measurement keep their catalog and data. Collision previews are not saved. Re-sweep is unavailable with no collisions, no survivors, a foreign-module file, or capture tuning.

- **Find Bias**
  1. **Find Bias** chooses an operating amplitude and frequency for every resonator in the sweeps on screen, in one `find_bias_points` call. **Find Bias Settings** opens a settings window grouped by the test each setting belongs to: which bifurcation test decides the amplitude, the thresholds that test reads, where in the chosen sweep the tone goes, and how far from the sweep centre an answer is believed. Settings persist between sessions, **Apply** saves edits, and **Close** discards unapplied edits. **Reset to Defaults** restores the default values; press **Apply** to save them.
  2. The report becomes the panel's catalog, and the sweeps come back carrying it as `bias_report`. On the sweep grids the amplitude step each resonator is biased at is drawn thick, with a dashed line at its bias frequency — dashed because solid and dotted already mean upward and downward. That line is named in the legend, `f_bias` over the drive it was chosen at in normalized DAC units — what a bias amplitude is, and what goes back into a re-run. On a resonator whose point is a fallback rather than a measurement the row names the flag, `f_bias — freq out of bounds`, and the subplot's tooltip carries the sentence behind it. The words are the library's `FLAG_KINDS`, so a plot and a notebook call a flag the same thing. The status line says `Bias found (2 of 9 flagged)`, then fades like any other outcome — nothing is on the board until Apply Bias, and which resonators are flagged is on the subplots they belong to.
  3. The **Bias: detect bifurc** tab draws what the derivative test looks at — the point-to-point change in each sweep's normalized arc speed — with every trace divided by the bar that trace faced, so the bar is one pair of lines at ±1 and a quiet step is visible beside a loud one. The band the bar encloses is shaded, because a bar is a region: everything inside it is not a spike, and a trace that stays in the shading is a trace the test passed. For the step a resonator is biased at, the gate that did *not* bind is shaded inside that, so the two shades together say how much of the bar in force is the noise gate and how much the prominence. Its legend names the two bars rather than the traces: the colorbar already says which drive a line is, and nothing else on a subplot says what ±1 is a threshold *of*. It draws before anything has been found, and redraws when you press **Apply** in **Find Bias Settings**, so you can inspect the chosen thresholds.
  4. The **Bias: frequency** tab draws what chose the frequency: each resonator's IQ arc speed — how far its trace moves per hertz — at the drive it is biased at, which is the quantity the default `iq_derivative` method maximizes. `dI/df` and `dQ/df` are drawn under it as thin green and red lines, because which of them carries the response is the other half of the reading: a speed that is almost all one component is an IQ loop that is not oriented the way it was assumed to be. The line is where the tone will actually go, on the hardware grid, so the gap between the line and the curve's peak is that quantization. Only the step the resonator is biased at is drawn, and the tab is empty until Find Bias has chosen one. With the `minimum` frequency method the line sits at the dip instead and need not be at this curve's peak.

- **Apply Bias**
  1. **Apply Bias** parks a tone on every resonator in the panel's catalog, at the frequency and amplitude it carries. Which NCO carries them, and putting the frequencies on the tone grid, are `apply_bias`'s doing.
  2. On success each channel's tuning row is published to the main window — `rfmux.tuning.tuning_rows` of the catalog, carrying the bias frequency, the amplitude, the `df_calibration` and the sweep it was read off. The main window displays the streams in hertz through each row's calibration, and a pulse capture started afterwards records the rows as its tuning.

- **Detector Digest**
  1. The **Detector Digest** tab is one resonator at the size of the panel, which is the question a grid of subplots cannot answer. Double-click any subplot on any grid tab to open the digest on that resonator; the arrow keys and the combo box walk the catalog from there.
  2. Three plots: its magnitude at every drive it was swept at, the same sweeps as IQ loops, and — on the right — the one sweep it is biased at, with the fitted model over it in cyan and a line where the tone will actually go. The left two are the measurement alone; the fit is drawn on the plot of the sweep it was fitted to. There is no drive selector, because the right-hand plot is about one drive: the one Find Bias chose. Until it has run, that plot says so.
  3. Under the plots, a column of `parameter: value` rows for each thing worth reading off this resonator: the bias point — its step, its drive, its frequency and offset, the responsivity there, the drive bifurcation was first seen at, and the flag if it carries one — and then one column per fit of the sweep it is biased at, headed by the model and the sweep direction, with each parameter beside its error. Hover a row for what the parameter means and the number the fit actually produced; the values are selectable, to be copied into a notebook. The fit columns are `collect_fit_params`'s rows, so a notebook tabulates the same thing.
  4. It measures nothing. Everything on it comes off the multisweep, the fits and the bias report the panel already holds, so it is worth opening on a file as much as on a sweep that has just finished.

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
