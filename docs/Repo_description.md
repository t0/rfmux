# rfmux Repository Map

## docs
- `Changes.rst` - Placeholder to store change logs in the repository
- `Repo_description.md` - Describing the overall structure of the repository.
## Firmware
- `firmware/CHANGES` – Detailed firmware release notes from r1.0.0 onward.
- `firmware/README.md` – How to fetch large LFS binaries and flash MicroSD cards.
- `firmware/r1.5.4/`
  - `boot.bin`, `fsbl.elf`, `u-boot.itb` – Boot artifacts.
  - `parser` – C++ data‑stream parser binary.
  - `rootfs.ext4.bz2` – Compressed root filesystem image (Git‑LFS pointer).
- `firmware/r1.5.5/` and `firmware/r1.5.6/` – Same layout; later versions, with `parser_src.tar.gz` source bundle and symlinks to previous boot files.

## Home (Jupyter Hub content)
- `home/README.md` – Notebook-style landing page for CRS boards.
- `home/Demos/`
  - `Network Analyses.ipynb`, `Noise From Parser Data.ipynb`, `Noise Spectra.ipynb` – Example notebooks.
  - `simplified_tuning_flow.py` – Script demonstrating full tuning algorithm workflow.
- `home/Release Notes/R1.5 Release Notes and Changes.ipynb` – Release documentation.
- `home/Technical Documentation/Signal Path Taps and Filtering.ipynb` – Technical note.

## Python Package `rfmux`
- `rfmux/__init__.py` – Package export list, version/dependency checks, session helpers, IPython awaitless integration.
- `rfmux/awaitless.py` – IPython AST transformer to auto‑await coroutines.
- `rfmux/streamer.py` – UDP packet format, timestamp handling, multicast helpers.

### Core (`rfmux/core`)
- `__init__.py` – Exports schema objects and mock CRS hooks.
- `crs.py` – YAML loader for hardware maps and macro/algorithm integration.
- `hardware_map.py` – SQLAlchemy session/query extensions, macro/algorithm registration, `HardwareMap` factory.
- `schema.py` – ORM models: `Crate`, `CRS`, `ReadoutModule`, `ReadoutChannel`, `Wafer`, `Resonator`, `ChannelMapping`.
- `session.py` – YAML/CSV parsing utilities, session caching.
- `dirfile.py` – Wrapper around `pygetdata` dirfiles with helper to fetch samples.
- `transferfunctions.py` – Conversion utilities (ROCs↔volts/dbm, cable delay, noise binning, IQ→df).
- Mock infrastructure:
  - `mock.py` – Public API for mock flavour.
  - `mock_crs_core.py` – Full mock CRS implementation (enums, context manager, UDP streaming control).
  - `mock_server.py` – Spawns async HTTP server backing the mock CRS.
  - `mock_udp_streamer.py` – Generates multicast UDP packets for mock data.
  - `mock_crs_helper.py` – Convenience functions to create/configure mock CRS instances.
  - `mock_resonator_model.py` – Physics model for resonators and channel responses.
  - `mock_constants.py` – Tunable parameters for mock physics, noise, and scaling.
  - `mock_constants` et al. provide strong mock support.

### Algorithms (`rfmux/algorithms`)
- `README` – Placeholder.
- `__init__.py` – Re‑exports measurement algorithms.
- `measurement/__init__.py` – Imports measurement modules.
- `measurement/bias_kids.py` – Bias KID resonators based on multisweep data.
- `measurement/fitting.py` – Skewed Lorentzian resonance fitting and IQ circle utilities.
- `measurement/fitting_nonlinear.py` – Nonlinear resonator fitting adapted from citkid.
- `measurement/multisweep.py` – High-resolution multisweep around resonances.
- `measurement/py_get_pfb_samples.py` – Client-side PFB sample acquisition and PSD computation.
- `measurement/py_get_samples.py` – Pure-Python data capture with optional spectra.
- `measurement/take_netanal.py` – Frequency sweep (network analysis) helper.

### Tools
- `rfmux/tools/__init__.py` – Conditionally exposes GUI tools (e.g., Periscope).
- `rfmux/tools/README.md` – Empty placeholder.

#### Periscope (`rfmux/tools/periscope`)
Real‑time GUI for data visualization and network analysis.
- `README.md` – Documentation of features and usage.
- `__init__.py` – Exposes `Periscope` class, CLI entry points, UI components.
- `__main__.py` – CLI/programmatic launcher (`periscope` command, `raise_periscope` function).
- `app.py` – Main `QMainWindow` application logic.
- `app_runtime.py` – Runtime helpers separated from UI logic.
- `bias_kids_dialog.py`, `detector_digest_dialog.py`, `find_resonances_dialog.py`, `initialize_crs_dialog.py`, `mock_configuration_dialog.py`, `multisweep_dialog.py`, `network_analysis_dialog.py` – PyQt dialogs for specific tasks.
- `network_analysis_base.py`, `network_analysis_window.py` – Base classes and window for sweep visualization.
- `network_analysis_export.py` – Export and cable-delay utilities.
- `multisweep_window.py` – Window for multisweep results.
- `dialogs.py` – Generic dialog classes reused across the app.
- `tasks.py` – Worker threads for UDP reception, network analysis, multisweep, etc.
- `utils.py` – Constants, helper functions, Qt/pyqtgraph setup, session creation.
- `ui.py` – Aggregated exports of windows and dialogs.
- `icons/periscope-icon.svg` – Application icon.

### Tuber (`rfmux/tuber`)
Lightweight RPC/remote-object protocol.
- `__init__.py` – Exceptions, versioning, exports client helpers.
- `client.py` – Asynchronous `TuberObject` client, resolve helpers.
- `server.py` – Registry and request handler utilities for tuber servers.
- `codecs.py` – Serialization/deserialization (JSON/CBOR/orjson) and `TuberResult` container.

## Tests (`test`)
- `__init__.py` – Empty module.
- `conftest.py` – pytest fixtures for live CRS sessions.
- `test_calls.py` – Macro/algorithm behavior tests and live board interactions.
- `test_schema.py` – Hardware map YAML/CSV parsing tests.
- `test_spotcheck.py` – Live hardware spot‑check tests (NCO, decimation, streaming).
- `test_threads.py` – Ensures SQLAlchemy session separation across threads.
- `integration/__init__.py` – Placeholder for integration tests.
- `crs_qc/`
  - `__init__.py` – Empty.
  - `conftest.py` – Interactive quality-control workflow, session setup, report generation.
  - `report_generator.py` – PDF/plot generation for QC results.
  - `style.css` – Styling for generated HTML/PDF reports.
  - `test_crs.py` – Extensive QC tests (temperature, voltage, DAC/ADC behavior, noise, etc.).

## Jupyter Settings
- `.jupyter/lab/user-settings/@jupyterlab/docmanager-extension/plugin.jupyterlab-settings` – Enables autosave and sets Markdown viewer to Jupytext.

## Other Files
- `README.md` – Overview, installation, firmware instructions, platform notes, contribution guidance.
- `README.Windows.md` – Running and testing on Windows.
- `pyproject.toml` – Package metadata and dependencies (Python ≥3.12).
- `test.sh` – Helper script to install Python versions via `pyenv` and run `tox`.
- `tox.ini` – Tox environments for multiple Python versions/SQLAlchemy.