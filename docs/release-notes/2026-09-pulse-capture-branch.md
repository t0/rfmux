# Pulse capture branch: release notes

Branch `buffer_exploration`, PR 78, September 2026.

This release adds pulse capture to rfmux: a detection engine that triggers on
a detector timestream at a threshold set from its own measured noise, records
each pulse to HDF5 as it arrives with its summary statistics, and pairs
events seen on the slow readout stream and the fast PFB stream. It runs
headlessly as the `trigger_capture` macro and interactively as a Periscope
panel, and the two share one engine, one ingest path and one file format.
Samples are stored in physical units, volts or hertz, never in ADC counts,
and the file carries the constants and calibration that produced them. The
how-to for the feature is docs/release-notes/2026-08-pulse-capture.md; this
document is the account of everything else that changed on the branch.

Around the feature, the branch reworked what it leaned on. The df calibration
moved from a spline through the multisweep into a resonance fit, and then
into a measurement `bias_kids` makes by stepping every tone; the nonlinear
resonator model was corrected to Swenson et al. 2013 eq. 13. The simulator
gained 1/f frequency wander, streams over multicast like a board, generates
its slow stream a block at a time, and biases a hundred resonators in seconds
where it took over a minute. The C++ receiver hands Periscope and the fast source
demuxed arrays instead of packets, and Periscope stopped losing packets to its
own receive path. The test suite is organised by subsystem and asked for by
tier, and CI runs every test that does not need a board on three platforms.
The C++ extension changed: rebuild after merging with
`source .venv/bin/activate && uv pip install -e .`.

## New

- `rfmux.pulse_capture`: a top-level package holding the detection engine,
  its compiled walk, the analysis helpers, the histogram and template
  accumulators, the HDF5 writers and reader, the stream sources and the
  single- and dual-stream capture sessions.
- `crs.trigger_capture(channel, module, streamer_mode, time_run, config,
  hdf5_path, df_calibrations, trigger_basis)`: one-shot capture in slow,
  fast or both modes.
- The result carries the pulses per channel, the pairs and the per-stream
  results; with `hdf5_path` the same content is written as the capture runs.
- Noise training before every capture: the threshold is `threshold_sigma`
  times a sigma measured as the samples' scatter about a block-median
  baseline, so it holds for the correlated samples the decimators produce.
  Training lasts twenty times `max_pulse_ms` and is not charged against
  `time_run`.
- Frequency-basis triggering: with a df calibration a channel is rotated so
  the pulse lies along one axis before thresholding, and stored in hertz;
  without one it stays on the quadratures in volts. The file records
  `trigger_basis`, `volts_per_count`, and per channel `stored_units` and
  `df_calibration`.
- Per-pulse timing from the packet clock: `trigger_epoch` and `trigger_utc`
  on every pulse, `time_origin_epoch` and `time_origin_utc` on the file, no
  host clock involved.
- Dual capture (`streamer_mode="both"`): slow and fast engines run together,
  triggers pair on their trigger instants, and every pair carries both
  streams over one common window (`window_t0`, `window_t1`).
- The fast stream through the C++ receiver: `PacketQueue.pop_pfb_batch`
  hands the source demuxed blocks, the lag is bounded by discarding old
  packets with a count, and a capture that receives no fast packets raises
  `TimeoutError` instead of ending silently.
- Periscope Pulse Capture panel: live capture with a pulse list, stacked I/Q
  or df/dissipation plots with the decision marks and bands, histograms,
  trigger-aligned templates, review mode for any capture file, and CSV export.
- Periscope Streamer Configuration dialog and `crs.configure_streamer`, over
  `StreamerConfig`, `describe` and `validate` in
  `rfmux.algorithms.measurement.streamer_config`: decimation, packet format,
  modules and PFB channels checked against the link budget before apply.
- `crs.get_biased_channels(module)` and `parse_channel_spec`: "all", "1,2",
  "2-19" and mixtures, in the Channels field and in scripts.
- `crs.measure_df_calibrations(channels=None, module=1, span_hz=20e3,
  resolution_hz=500)`: a lockstep sweep of every channel (all biased channels
  by default), one batched frequency write and one module read per point, the
  calibration from a resonance fitted to each sweep.
- `bias_kids(fit_method=, measure_calibration=, calibration_step=)`: the
  amplitude choice, the bias frequency and the calibration come from one fit;
  the calibration is then measured by stepping every tone and kept alongside
  the fit's as `df_calibration_fit`.
- `find_resonances(require_isolation=True)`: drop every member of a group of
  peaks closer than the separation, instead of keeping the most prominent.
- Simulator: TLS 1/f frequency wander (`tls_noise_enabled`,
  `tls_fractional_rms`, `tls_alpha`, `tls_corner_hz`); build progress
  through `get_build_progress`; pulse-cache warm-up at build; `set_pulse_mode`
  applied live; `apply_mock_config`, `config_changes`, `pulse_only_change` and
  `pulse_mode_kwargs` in `rfmux.mock.helpers`.
- Streamer helpers: `resolve_host`, `find_streamer_conflict`,
  `find_competing_receiver`, `check_multicast_loopback`, `ts_to_seconds`,
  `day_epoch`; C++ `pop_readout_batch`, `pop_pfb_batch`, `drop_pfb_before`,
  `PacketReceiver.flush_all`, `Timestamp.seconds_of_day`, and a
  `packets_missing` counter that counts packets rather than gaps.
- `rfmux.core.transferfunctions`: `PFB_NYQUIST_FREQ` beside the existing
  `PFB_SAMPLING_FREQ`, stated as not a rate; the CIC parameters as constants;
  `decimated_stream_delay_s`, `sampling_to_decimation`, `apply_iq_conversion`.
- Periscope: app-wide zoom (Ctrl+, Ctrl-, Ctrl+0, persisted); flow-layout
  toolbars that wrap to a laptop width; a framed progress window for large
  mock builds; the mock's df calibrations measured at startup.
- Two runnable jupytext notebooks, `pulse_capture.md` and
  `simplified_tuning_flow.md`, executed in CI; `pulse_capture_flow.py` as the
  script twin; the release-note how-to with reproducible screenshots.
- Test tiers by name (`pytest --tier=quick|portable|acquisition|full|hardware|all`).

## Changed

Old values are main at the merge base (e46fc41).

- Python floor: `requires-python` 3.9 to 3.10 (`rfmux/core/crs.py` uses
  `match`); tox drops py39 and installs `--group test` instead of the
  non-existent `.[dev]` extra, selecting `-m portable` instead of `-k offline`.
- h5py: undeclared to a runtime dependency. jupytext: dependency group to
  runtime dependency (it ships the JupyterLab plugin that opens `.md`
  notebooks).
- Multisweep result: `df_calibration` was the slope of a cubic spline through
  the raw sweep at the bias point, with `iq_complex_volts` and
  `calibrated_tod_df` beside it; the entries carry none of the three, and
  `apply_df_calibration` is gone. The calibration comes from `bias_kids`.
- `bias_kids`: `phase_step` removed (phase optimisation is one PCA of one
  sample set instead of a 72-phase scan); `fit_method` ("nonlinear" default,
  or "skewed"), `measure_calibration` (True) and `calibration_step` (0.05 of
  the fitted linewidth) added. The calibration was the multisweep's spline
  slope; it is now measured by a tone step, in hertz per volt.
- Nonlinear resonator model: `yg = y + a/(1+y^2)` solved by Newton, to
  `y = yg + a/(1+4y^2)` (Swenson 2013 eq. 13) solved by bisection with a
  Newton finish. Fitted `a` values from earlier releases are not comparable.
- Simulator defaults: `nqp_noise_std_factor` 0.001 to 0.01; TLS wander absent
  to on (1e-7 fractional RMS, alpha 1.0, corner 100 Hz); `pulse_tau_decay`
  0.1 s to 5 ms; `pulse_random_amp_min`/`max` 1.5/3.0 to 1.1/1.5;
  `bias_amplitude` 0.01 (about -40 dBm) to `bias_amplitude_from_dbm(-55)`,
  about 0.0016, with `DAC_SCALE_DBM` (1 dBm) and `BIAS_DBM` (-55 dBm) in
  `rfmux.mock.config`.
- Simulator transport: unicast to 127.0.0.1 with multicast TTL 1, to
  multicast on the hardware group with TTL 0, falling back to loopback unicast
  and printing the failing step when the host cannot multicast.
- Simulator auto-bias: capped at 256 channels, to as many as a packet carries.
- Simulator dip search: a 2000-point sweep over +/-10 MHz per resonator, to a
  50 kHz pass over +/-0.25% then a 200 kHz, 101-point pin.
- Simulator physics: the multi-sample path evaluated one sample at a time, to
  a hoisted batch path (`physics_batch_mode` "hoisted"; "reference" keeps the
  loop); the slow stream generated one frame per call, to about 50 ms of
  frames per call; PFB packets of 64 samples, one per physics sub-batch, to
  1000-sample packets, the hardware's size.
- Simulator kernels: numba `parallel=True` unconditionally, to parallel only
  from `PARALLEL_MIN_N` (1024) resonators.
- Simulator API: `set_analog_bank(high_bank=)` to `set_analog_bank(high=)`,
  matching the board.
- Simulator slow packet timestamps: exact, to stamped late by the CIC group
  delay the hardware imposes.
- Periscope receive thread: `receive_batch(batch_size=16)` to 2048, with the
  recvmmsg scratch allocated once.
- Periscope status bar: "net"/"gui" loss to "missed" (never reached the
  receiver) and "dropped" (reached it and was thrown away), counted per packet
  instead of per gap.
- Periscope Session Browser filters: `*.pkl, *.ipynb, .png` to include
  `*.png`, `*.h5` and `*.hdf5`.
- Periscope Jupyter: `.md` files opened as markdown, to opened as notebooks by
  default (overrides.json via app_settings_dir; the user's own settings stand).
- Periscope mock mode: the module came from the hidden spinbox, to the module
  the mock streams, not persisted.
- Periscope multisweep bias frequency method: None (keep the original centre
  frequency) to "max-diq", the headless default.
- C++ receiver build: `-ffast-math` removed (it folded a NaN select on
  clang, so an undisciplined timestamp read as a number on macOS).
- Test layout: flat files under `test/` to directories mirroring the package
  (`core/`, `streamer/`, `mock/`, `algorithms/`, `periscope/`,
  `pulse_capture/`, `notebooks/`).
- Test selection: the `offline` marker is `portable`; `--tier` and `--serial`
  are declared in a root `conftest.py` so they work from the repo root; a bare
  `pytest` applied no marker expression and now applies
  `-m "not slow_acquisition"`.
- CI: two named test files to the quick and acquisition tiers on ubuntu,
  windows and macos with the test dependency group installed; `paths-ignore`
  `**/*.md` to READMEs, `CLAUDE.md` and `CHANGELOG.md` only, so the jupytext
  demos trigger it.
- Networking guide: `rmem_max` differed across sections (64 MB and 128 MB)
  and is now one value, 268435456 (256 MB, about three seconds of the
  four-channel PFB stream).

## Fixed

Bugs present on main, with the symptom.

- Periscope lost packets to its own receive path: at stage 0 a 128-channel
  capture with one channel on screen reported 38% to 52% loss as "net", and
  widening a capture to a few hundred channels froze the window in an
  unbounded queue drain. The receive thread reads up to 2048 datagrams per
  call over a scratch allocated once, the display writes a frame at a time,
  the tap hands the capture worker whole packets, and the drain stops at a
  250 ms backstop. On the board: 128 channels at stage 0 with nothing dropped.
- Ctrl+C did not exit Periscope: the window stayed up holding UDP 9876, and
  the next launch bound the port and received nothing. SIGINT now closes the
  windows and quits; a second Ctrl+C leaves at once; the receive thread is
  joined so exit no longer core-dumps.
- A Noise Spectrum panel holding data could crash the interpreter when
  dropped: its hover handler owned the panel through the mouse-move proxy,
  so the panel waited for the cyclic collector, which finalizes Qt objects
  in arbitrary order. The Multisweep panel's unit buttons held it the same
  way through lambda slots. The handler holds the panel weakly, the buttons
  call a bound method, and the teardown test builds the Noise Spectrum
  panel with data and refuses any panel that needs the collector.
- Closing a Network Analysis dock then changing units, PSD or channels raised
  `RuntimeError: wrapped C/C++ object of type ClickableViewBox has been
  deleted` from inside pyqtgraph; the registry now forgets closed panels.
- `ClickableViewBox` held its panel strongly, so panel teardown ran through
  the cyclic collector and could segfault (exit 139); the back-pointer is a
  weakref.
- A socket that received nothing at all wedged the receive thread inside
  recvmmsg regardless of `timeout_ms` (blank viewer, 0 received, 0 dropped);
  the timeout is on the socket.
- Watching a module nothing streams read as a dead stream with zero counters
  and no message; the receiver names the module it watches and the modules
  arriving, in the status bar and on stderr.
- Another process taking the mock's unicast stream left a flat zero with no
  explanation; Periscope now says another receiver holds the port.
- The session dialogs opened at Qt's process-global last directory and never
  saved the choice; they open at and record the last session directory. A
  test run had also written a `/tmp/pytest-...` path into the user's
  `periscope.conf`; tests patch the settings store.
- `MockCRS.get_pfb_samples` returned a list of (i, q) tuples where every
  caller reads `.i` (`AttributeError: 'list' object has no attribute 'i'`),
  so the mock had no working PFB samples path; it returns `{"i", "q"}` as
  `get_fast_samples` does.
- The notebook test globbed `test*.md` relative to the working directory, so
  from the repo root, where CI runs, it collected nothing and had never run;
  the glob is relative to the file.
- Four mock sites read `physics_config` instead of `_physics_config`, so the
  Mock Configuration dialog's cache-tuning settings were ignored and
  `get_samples` always used `scale_factor=2**21`.
- Regenerating the mock array raced the streamer thread (`ValueError:
  operands could not be broadcast together`); regeneration holds the physics
  lock.
- Periscope pinned its GUI thread to one core and then forked the mock
  server, which inherited the mask: the whole simulation ran on one core. The
  child restores a full mask.
- Mock server shutdown ran one atexit handler per session in series, holding
  the streamer sockets for over a minute after a test run; they shut down
  together.
- `test_schema.py` interpolated a Windows path into a YAML scalar, so `\U` was
  an invalid escape there.
- `test_spotcheck.py` asked for `rfmux.tuber.TuberRemoteError`, a module
  removed in favour of the tuber-client package, so the hardware test errored.
- Every getting-started snippet under Common Operations raised as written
  (module imported as a function, `channel=` on `take_netanal`, `bias_kids`
  called standalone, `rfmux.core.mock`, `filter_by` on a list).
- `.venv` was not ignored, so `git add -A` after the README's setup committed
  it.
- The unimported copy of the mock defaults, `rfmux/core/mock_config.py`, was
  13 keys behind and wrong on five values; deleted.

## Simulator

- Slow stream generated in blocks of about 50 ms of frames per physics call,
  one packet per frame with its own sequence number and stamp: 100 tones ran
  at a quarter of real time and now run at real time with pulses off.
- Hoisted batch physics: pulse sum, QP noise draw, nqp to (R, Lk) kernel and
  TLS lookup evaluated once per instant of a batch, convergence-cache
  decisions in the reference order, parity with the reference loop at 1e-9.
- Coupled pairs found through the sorted tones instead of every observer/tone
  pair: a 1023-channel `get_samples` chunk from 314 ms to 29 ms.
- TLS wander is a per-resonator capacitance perturbation, a pure function of
  absolute time so the slow and PFB streams stay common-mode; a sum of
  Ornstein-Uhlenbeck processes with log-spaced corners.
- White QP noise is applied after the convergence-cache restore through a
  sensitivity linearisation, so it no longer defeats the cache.
- The dip search runs the S21 kernels over a grid (`s21_sweep`), re-converging
  at each point; the S21 minimum sits above the impedance resonance by the
  coupling shift, so the coarse pass covers +/-0.25% of the nominal frequency.
- The build runs its CPU-bound parts on a thread, so RPCs answer while it
  runs; `get_build_progress` reports generating, biasing and pulse-cache
  warm-up with counts.
- The pulse-cache warm-up runs one pulse on every resonator at build so the
  first live pulse does not stall the stream.
- A pulse-only configuration change goes to `set_pulse_mode` without
  regenerating the array; an unchanged configuration does nothing.
- The mock writes 0-indexed PFB slot fields, as the board does.
- `physics_batch_mode="reference"` keeps the per-sample loop selectable.
- `_auto_bias_kids` biases module 1 only, so a mock streams tones on module 1.

## Periscope

- Pulse Capture panel: Start/Stop, Re-estimate Noise, mode (slow/fast/both),
  Channels (with "all" and ranges), threshold and end sigma, a Settings
  dialog for the rest, a Streamer button, a Units control (counts, volts, df
  in hertz) that drives the waveforms, histograms and templates together.
- Pulse list: length, SNR, trigger time in UTC; columns size to content until
  the user drags one; Left/Right, Home/End, Space and Ctrl+E for navigation,
  tab cycling and export.
- Pulse view: I and Q, or df and dissipation, as two x-linked plots, each
  with its own baseline and bands drawn from the band the decision was made
  against; marks for the trigger, the below-threshold return and, when the
  tail is saved, the end confirmation; the info line names the quadrature
  that fired.
- Histograms of SNR, amplitude, duration and derived decay constant with
  auto-expanding ranges; a Plot field ("1,2,4", "1-5", "*") combines
  channels; templates combine as the count-weighted stack.
- Both-mode view: the fast trace in purple/red over the slow in blue/orange,
  the pair's trigger offset, per-stream bands, a status line that turns amber
  and red as the fast stream falls behind, with the cause and remedy in the
  tooltip.
- Review mode: opening a `.h5` restores the capture parameters into locked
  controls and serves pulses, pairs, histograms and templates from the file;
  double-clicking a capture in the Session Browser opens it, or focuses the
  live panel if the capture is still running.
- The capture reads the board's PFB streamer state and never sets it; a
  mismatch between streamed and requested channels fails before a socket
  opens, naming both and where to change it.
- Streamer Configuration dialog: decimation, short packets (forced below
  stage 3), modules, PFB channels; derived rate, Nyquist, channels per packet
  and Mbps against the 1 GbE budget; the validation tiers in a banner.
- Toolbars are flow layouts: the main window fits 236 px, the multisweep
  panel 389, network analysis 228, pulse capture 354.
- Mock startup: a framed progress window for arrays above 25 resonators; the
  df calibration sweep runs in a `DfCalibrationTask` while the window streams,
  reporting on the status bar; picking df units mid-sweep says so rather than
  starting a second sweep.
- Find Resonances dialog: a "require isolation" checkbox, off by default.
- Bias KIDs dialog: the fit method choice, preselected to the fit the sweeps
  carry; the nonlinearity threshold greys out under the skewed fit; the
  phase-step control is gone.
- Mock Configuration dialog: bias power shown and edited in dBm; a TLS noise
  group; pulse changes apply without a rebuild; an untouched round trip is not
  a change.
- Detector digest: both fit tables show the sweep's `is_bifurcated` flag;
  the nonlinearity parameter has three decimals.
- The Jupyter panel passes `RFMUX_CRS_HOSTNAME` and `RFMUX_CRS_SERIAL` to the
  kernel so a notebook attaches to the board Periscope is on.
- Mock mode is `periscope MOCK` or the startup dialog's mock connection.

## Algorithms and calibration

- `bias_kids` fits the sweeps that lack the chosen fit with the flow's own
  batch fitter (about 20 ms per sweep nonlinear, 4 ms skewed), writes the fit
  and the bias frequency back onto the entries, and reads the bias frequency
  (max-diq or min-s21) off the fitted curve.
- The measured calibration: every tone steps `calibration_step` of its
  fitted linewidth down and up in lockstep, two module reads, the inverse of
  the complex slope; a step is never less than one grid step and the fit
  supplies a curvature correction. Samples that do not move, or a failed
  read, fall back to the fit; one warning names detectors where fit and
  measurement disagree by more than 5 degrees or a magnitude ratio outside
  0.7 to 1.4.
- Tones stay on multiples of `TONE_GRID_HZ` (625 MHz / 2^21, about 298 Hz),
  as on main; the rounding now says why (intermodulation products land on the
  grid) and applies to the calibration step too.
- The ADC phase: with `optimize_phase=True` the principal axis of (I, Q) goes
  to Q from one sample set, and the calibration turns by minus the phase; the
  multisweep zeroes the ADC phase on the channels it sweeps.
- Bifurcation is a warning, not a refusal: `identify_bifurcation` sets
  `is_bifurcated` on the multisweep entry; `bias_kids` still biases, at the
  lowest amplitude swept when it has a choice.
- `df_calibration_for_entry` uses the nonlinear IQ fit an entry carries, else
  the skewed fit; `bias_frequency_from_fit`, `fitted_linewidth`,
  `step_slope_correction`, `ensure_fits` in
  `rfmux.algorithms.measurement.df_calibration`.
- The df calibration is a measurement the host makes, so it lives in
  algorithms; the mock has no calibration RPC of its own.
- `require_isolation` in `find_resonances` runs before the
  `expected_resonances` trimming and warns when it drops peaks; isolation is
  judged against the peaks the search returned.
- `streamer_config.validate`: stage 0-6; long packets need stage 3 or above;
  more than 1000 Mbps is an error and more than 800 Mbps (the firmware's
  derating) a warning; stage 1 and below advise on the OS buffer; more than
  one module below stage 5 is noted as unvalidated; PFB channels at most four.
- `apply_streamer_config` sends `module=` (firmware r1.6 spelling); the mock
  mirrors the firmware signature so the next rename fails in tests.
- `apply_iq_conversion` and `convert_iq_to_df` sit together in
  `transferfunctions`; `storage_transform` and `display_transform` stay in
  `pulse_capture` as that package's policy.

## Streamer and packets

- `pop_readout_batch(max_packets)`: samples as a (packets, channels) complex
  array with the packetizer gain out, seconds of day per packet (NaN when not
  disciplined), recent flag, stage, sequence numbers, and the day from the
  first disciplined stamp; one packet width per batch.
- `pop_pfb_batch(max_packets)`: one layout per batch as a (groups, samples)
  complex array, seconds of day, sequence numbers and the layout fields;
  `drop_pfb_before(t, limit)` discards by stamp without demuxing.
- The per-packet accessors remain the reference and the fallback for a
  receiver built without the getters; the batch getters share the packet's
  own accessors.
- `flush_all` releases what the reorder stage holds when a capture stops, and
  takes the mutex its per-queue sibling holds.
- `packets_missing` counts packets by unsigned sequence distance, ignoring a
  reordered packet's near-2^32 gap; `sequence_gaps` still counts bursts.
- The slow ingest (`SlowIngest`) blocks up to 256 packets or 50 ms and is
  shared by `run_slow_source` and the Periscope tap.
- Slow packets are fed in timestamp order, the newest four held back for
  stragglers at each flush; the sample clock is monotonic across a
  decimation change and the day boundary.
- The PFB source picks its channels from the packet's slot fields, so any
  subset of the streamed channels can be captured, and keeps only its
  module's packets.
- PFB packets are released in sequence order through a 64-packet window and
  fed in blocks of sixteen packets per channel.
- The PFB socket asks for the largest receive buffer the host allows
  (`net.core.rmem_max` on Linux) and warns when what it got holds under a
  second of stream.
- Past a 0.25 s lag between the slow and fast clocks the fast source discards
  packets by stamp until it is within 0.125 s, counting them as
  `flushed_packets`; `lost_packets` comes from the queue's own sequence
  accounting; `busy` is processing time over wall time.
- The dual session shifts every slow timestamp by minus the CIC group delay
  (about 2.9 slow samples at any stage, 4.99 ms at stage 6) before the
  engine, the matcher and the file; recorded as `slow_time_offset_s`, 0 when
  not applied; pass `slow_time_offset_s=0.0` to opt out.
- Pairs form on trigger instants within half the CIC2 response, three slow
  samples.
- A trigger with no partner waits the hard stop (1.2 times `max_pulse_ms`)
  plus 50 ms before going out one-sided.
- Each stream's window is taken when its own ring covers it; a stream that
  never delivers one is given `pair_window_wait_s` (3 s).
- PFB slot fields are 0-indexed on the wire, like the module field.
- The rate: the PFB stream is `PFB_SAMPLING_FREQ`, 625 MHz / 256, about
  2.44 MHz per channel; 625 MHz / 512 is its Nyquist frequency and bin spacing.

## Tests and CI

- `test/` mirrors the package: `core/`, `streamer/`, `mock/`, `algorithms/`,
  `periscope/`, `pulse_capture/`, `notebooks/`; helpers in
  `test/packet_helpers.py` and `test/qt_helpers.py`; one `qt_app` fixture in
  `test/conftest.py`.
- Tiers: `portable` (no PyQt6, no board; what tox runs), `quick` (the edit
  loop, no streaming), `acquisition` (MockCRS server plus real UDP),
  `full` (acquisition included), `hardware --serial`, `all`. Passing both
  `--tier` and `-m` is an error.
- Markers: `portable`, `slow_acquisition`, and `hardware` applied
  automatically from the `crs`, `live_session` and `serial` fixtures.
- A session guard refuses to start the acquisition tier while 9876/9877 are
  held, naming the ports and the `ss` command; `--allow-busy-streamer-ports`
  overrides. Streaming tests own their sockets from a fixture so a failure
  does not strand a reader.
- Contract tests: block ingest against per-sample ingest bitwise; the
  compiled walk against the loop and the uncompiled walk; the batched getter
  against the per-packet conversion through a real receiver on loopback; the
  CIC delay sign against a synthetically late-stamped slow stream; mock batch
  physics against the reference loop at 1e-9; the mock's carrier parity
  across decimation stages.
- Platform skips are stated: `recvmmsg` blocking (Linux only),
  `SO_REUSEPORT` (absent on Windows), `SIGINT` (Windows delivers
  `CTRL_C_EVENT`); `test_fastrx_file.py` skips unless fastrx was built.
- Both demo notebooks execute in the acquisition tier; a demo writes to a
  temp directory because the shipped copies are read-only.
- CI: `periscope-tests.yml` runs quick and acquisition on ubuntu, windows and
  macos with `fail-fast: false` and the test group installed;
  `test_mock_vs_real.py` guards its imports so a hardware-only module cannot
  abort collection; `build.yml` builds fastrx and runs `test_packets.py`.
- `test_embedded_console.py` executes code through a real in-process
  qtconsole kernel so the `ipykernel<7` pin cannot be relaxed unnoticed.
- Periscope run output lives in `outputs/` (ignored); `test/.gitignore`
  refuses `session_*/` anywhere under the test tree, so run data cannot be
  committed from either place.

## Documentation

- docs/release-notes/2026-08-pulse-capture.md, the how-to, with a README
  explaining how these notes relate to `CHANGELOG.md` and `firmware/CHANGES`.
- `rfmux/reference-notebooks/Demos/pulse_capture.md` and
  `simplified_tuning_flow.md`: jupytext notebooks with minimal headers, three
  labelled ways in (attach, own board, simulate), a streamer-conflict check
  before creating a simulation, and IS_MOCK-guarded teardown.
- `pulse_capture_flow.py` and `simplified_tuning_flow.py` are code references
  in execution order; the tuning script runs against MOCK in the acquisition
  tier, the capture script by hand.
- docs/make_release_note_screenshots.py and
  docs/make_pulse_capture_figures.py regenerate the screenshots and the
  capture-window anatomy figure; the release note's mock config is the one
  the script uses.
- docs/guides/getting-started.md: examples that run; `create_mock_crs` shown;
  one simulation per machine; the `periscope` entry point.
- docs/guides/networking.md: one `rmem_max` value and why.
- README.md: a layout that matches the repository (`mock/`, `pulse_capture/`,
  `reference-notebooks/`, `streamer/`; no `packets/`, `tuber/` or `home/`).
- `rfmux/reference-notebooks/README.md` is the landing page for notebooks
  shipped inside an installed rfmux, not for a board's JupyterLab.
- CLAUDE.md: the project reference for contributors, with the review pass
  every change gets; test/README.md: tiers, markers, layout, notebook
  conventions, the one-run-at-a-time rule and CI triggers.
- Vocabulary: docs say "bias the resonator", the API's word; "edge test",
  "end confirmation" and "capture limit" replace branded names; comments say
  what the code does, not how it got there.
- Screenshots are dark-mode captures from the mock in the frequency basis.

## Judgement calls

- `trigger_basis` defaults to "df": a channel with a calibration is rotated,
  one without stays on the quadratures, so the default is "df where
  available" by construction. Pass "iq" to threshold the raw quadratures.
- `end_sigma` defaults to 1.0, below the 1.2 white-noise break-even:
  measured on the mock, 1.0 closes every capture with no hard stops and
  unchanged pulse counts; the validation warning sits below 1.0.
- `save_to_end_confirmed` defaults to off: the window stops a margin past the
  below-threshold return, so its length is a property of the pulse, not of
  how long the baseline took to confirm.
- `trigger_samples` defaults to 0, meaning derived from the stream rate to
  hold accidentals under one per minute per channel; `noise_train_ms` 0 means
  twenty max pulses.
- `min_end_samples` (10) is exposed rather than fixed: for a short pulse it
  alone decides where the end mark lands.
- Sigma is measured about a block-median baseline, not from adjacent
  differences: thresholds on the slow stream effectively rise 1.3x to 1.6x
  at a given `threshold_sigma`, the estimate becoming honest rather than the
  sensitivity dropping.
- Samples are converted to volts or hertz on the way in, not at read time: a
  count projected onto the frequency axis means nothing alone.
- The multisweep never fits or calibrates: at thousands of resonances it
  stays the quick look; `bias_kids` fits once and writes the fit back.
- `bias_kids` measures the calibration rather than trusting the fit: the
  fit's direction is up to 4 degrees off on the simulator, and a real detector
  disagrees with the model in its own way; `measure_calibration=False`
  restores the fit's.
- The calibration step is a twentieth of the fitted linewidth: a fixed 300 Hz
  read the slope 13% low on a 2 kHz linewidth and 35% low on 1 kHz.
- Tones stay on the 298 Hz grid: intermodulation products land on the grid
  too. A resonator narrower than about 6 kHz gets one grid step and leans on
  the fit's curvature correction.
- The multisweep's bias frequency method defaults to max-dIQ, headless and in
  Periscope, read off the raw grid; `bias_kids` reads it off the fit.
- Bifurcation warns and does not withhold the bias: the sweep says so and
  `bias_kids` falls back to the lowest amplitude by default.
- `require_isolation` is off: existing callers and the tuning notebook keep
  their results.
- The mock is noisy by default (1% QP noise, TLS wander on): a simulator
  quieter than the hardware lets code pass that fails on a real detector.
- The mock's bias power is stated in dBm against `DAC_SCALE_DBM`, the unit
  the instrument is set in, and -55 dBm is the chosen power; the dip is found
  the way hardware does it, a coarse pass then multisweep's span.
- The mock multicasts on the hardware group with TTL 0 and falls back loudly:
  multicast is what people have trouble with, and a real CRS has no fallback.
- Pulse decay defaults to 5 ms: 0.1 s was 365 quasiparticle keys per tone and
  a 28 s warm-up at 100 tones.
- The framed progress window appears only for arrays above 25 resonators;
  smaller arrays build before the window would be read. The pulse-cache
  warm-up runs at every build.
- h5py and jupytext are runtime dependencies: a plain install could not
  complete a capture or open the shipped notebooks, and the h5py-optional
  guards let a capture silently persist nothing.
- The Python floor is 3.10, what the code has needed since `crs.py` used
  `match`.
- A capture never configures the PFB streamer, headless or in Periscope: set
  it first (`configure_streamer`, or the Streamer Configuration dialog), and a
  mismatch refuses the capture before a socket opens, so a capture cannot
  reconfigure the board underneath the user or tear a stream down under
  another capture. Turn the PFB streamer off yourself afterwards; the how-to
  shows the line.
- Streamer configuration is authoritative: applying the dialog, or calling
  `configure_streamer`, sets the whole streamer state it describes, so an
  unchecked PFB box (or the macro's default) disables the PFB streamer rather
  than leaving it as it was. Pass `pfb_channels=None` to leave it alone.
- `py_run_pfb_streamer` measures `time_run` from the packet timestamps and
  tolerates stale ones for five seconds, the time IRIG-B needs after
  `set_timestamp_port`, then raises naming that call, as `py_get_samples`
  does, rather than fall back to wall time or sample counts.
- The fast source bounds its lag at 0.25 s by discarding, counted: the loss
  is the same deficit either way, bounded and named this way. The kernel's
  backlog figure is shown but not acted on (UDP releases memory in steps of a
  quarter buffer).
- The PFB socket asks for the whole `rmem_max`: the kernel charges memory
  only for queued packets, so asking for less saves nothing.
- The fast stream through the C++ receiver, not Python: four channels cost
  most of a core in Python and the capture thread could not keep up.
- Pairs form on trigger instants within half the CIC2 response: one slow
  sample split matched events 55 us apart; midpoints biased the choice toward
  a fast pulse in the slow decay.
- One-sided pairs wait the hard stop plus 50 ms: a fixed 250 ms released the
  fast pulse before the slow capture's hard stop at 300 ms.
- The CIC group-delay shift applies in both mode only, marked TEMPORARY until
  the firmware stamps the decimated stream at its filter centroid;
  single-stream captures keep the board's stamps.
- The pileup split judges the deviation vector's length, not each quadrature:
  a pulse rotating in the IQ plane is one pulse.
- Past four channels (`MAX_LISTED_CHANNELS`) status lines and legends
  summarise, with the full listing in the tooltip; the Plot field combines
  channels rather than drawing 128 legend rows.
- The link budget derates to 800 Mbps, the firmware's figure; stage 0 needs
  short packets.
- Periscope taps its own receiver for the slow stream instead of opening a
  second socket: this process already holds the packets.
- `.md` files open as notebooks inside Periscope's Jupyter session, as a
  schema default the user can change, not a forced setting.
- A regeneration or a routine outcome reports on the status bar; a dialog is
  for a failure the user must act on.
- `periscope MOCK` uses the module the mock streams and does not persist it,
  so the board's module choice survives.

## Final review pass

Before this checkpoint the whole branch was reviewed against the eight
questions the project asks of every change: correctness, duplication,
simplification, docs and notebooks, comments without history, Periscope as a
thin caller, headless and Periscope parity, and the saved data products.
Thirty-seven reviewers each read one area through one lens; every finding
then went to three independent reviewers told to refute it, and survived
only when at most one could. Of 356 distinct findings 338 survived, 18 were
refuted, and 276 were fixed on the branch, one file group at a time, each fix
with the test that fails without it. The fixes that change behaviour:

- Amplitude histograms were one bar for every channel stored in volts or
  hertz: the bins were sized for ADC counts. They now span a hundred sigma of
  the measured noise in the stored units.
- Both receive threads spun on a quiet socket: `settimeout` puts the socket
  in non-blocking mode and sets no receive timeout, so `recvmmsg` returned at
  once, six hundred thousand times a second. They now set `SO_RCVTIMEO` on a
  blocking socket.
- `trigger_capture` never printed its "noise training never completed"
  warning, and `describe()` understated ring memory by half; both fixed.
- A capture of a channel beyond the slow packet width trained forever
  headlessly; it now raises naming the channels and the width, as Periscope
  refuses up front.
- Re-estimate Noise did nothing in "both" mode; a "both" capture file was
  never registered with the session; the pulse info line and the amplitude
  histogram showed stored units under the view's label; a reviewed dual
  pair never showed its decay constant. All fixed, with one
  `summary_from_attrs` helper in `rfmux.pulse_capture.analysis` shared by
  live and review paths.
- Pulse Capture refuses Start in slow and both modes when the Module spinbox
  differs from the module Periscope is receiving.
- OFFLINE mode raised once a second: `DummyReceiver` lacked the four receiver
  methods the status bar calls.
- A pulse-only edit in the Mock Configuration dialog recorded the pulse mode
  as off while the server kept pulsing, and the next edit stopped the pulses;
  the mode now travels with the merged configuration and the session file.
  Regenerating the array drops the previous array's df calibrations and
  measures again.
- Clearing the seed field in the dialog now regenerates with a fresh pinned
  seed instead of reading as no change.
- `bias_kids` restores the tones in a `finally` when the calibration step
  fails; with `fit_method="skewed"` the nonlinear threshold is not applied;
  a fit whose nonlinearity is at or above 4 sqrt(3)/9 is declined for the
  bias point and the calibration, since the model is multivalued there.
- The mock streamer has one stream clock, so a module toned after start, or
  an array with no tone on module 1, is stamped and simulated on the same
  time axis as the PFB stream; PFB frames no longer prune a block's pulses
  before the slow block sees them; a three-channel PFB request is refused,
  since the packet mode cannot express it.
- `find_competing_receiver` warns only when the stream really is unicast (the
  mock's fallback), not whenever the host is loopback.
- `packets_missing` no longer over-counts a packet that arrives late past the
  reorder window.
- A typo in the dialog's PFB channels field refuses OK instead of disabling
  the PFB streamer.

Two findings the reviewers refuted after a fix agent had acted on them were
reverted: a `ValueError` for a requested channel absent from the PFB packet
(every headless caller configures the streamer to exactly those channels
first) and a widening of `trigger_capture`'s `channel` to any sequence (the
documented contract is an int or a list).

The full ledger, with every finding's evidence, verdict and outcome, is kept
outside the repository.

## Transfer-function audit

A separate review inventoried every site where a transfer function or unit
conversion is applied, 362 of them, traced the main data paths end to end,
and refuted each suspected defect three ways. Six fixes followed; the rest,
and what only a board can settle, are in
[2026-09-hardware-checks.md](2026-09-hardware-checks.md).

- One count scale. The C++ receiver takes the packetizer's factor of 256
  out once and that is ADC counts, the scale `get_samples` reports (settled
  by inspection of a board). Periscope's main window and both pulse-capture
  sources divided by 256 a second time, so on hardware they read 48 dB low
  against the Noise Spectrum panel and a frequency-basis capture reported
  hertz 256 times too small; the simulator hid it by emitting its stream 256
  times hot. Both the second division and the simulator's multiply are gone.
  Pulse-capture files recorded on hardware before this hold volts and hertz
  256 times low; simulator files are unchanged in meaning. The simulator's
  readout-noise default (`udp_noise_level`) moves with its signal, from 10
  to 0.04 counts, so every stream consumer sees the signal-to-noise it did;
  the value is provisional until a board's floor is measured.
- One volts-to-dBm convention. Volts are peak amplitudes throughout, so
  power is the square over twice the termination. The spectrum paths in
  `spectrum_from_slow_tod` and `py_get_pfb_samples` omitted the factor of
  two and read 3 dB high; `volts_squared_to_dbm` in
  `rfmux.core.transferfunctions` is now the one place the conversion lives.
- The IQ plot in Real Units mode converted counts to volts twice; the Noise
  Spectrum panel's fast timestream was converted twice under the default
  reference. Both convert once.
- A multi-amplitude network-analysis pickle reloaded every curve on the
  wrong probe amplitude, because the display's copy of the latest sweep was
  exported at index 0 and loading paired by position. The export skips the
  copy and the loader pairs by each sweep's own amplitude tag.
- Loading a bias file no longer rotates the board's ADC phase after
  `apply_bias_output` has restored the phase each calibration was measured
  in; the refinement never rotated the calibrations to match.

## Known limitations and what was not verified

- The measurement algorithms (network analysis, multisweep, fitting,
  `bias_kids`, `measure_df_calibrations`) run against the mock only; the
  hardware tier checks plumbing, not physics. The measured calibration's
  0.7 degree figure, the fitted bias frequency's 0.7 kHz figure and the
  nonlinear model's behaviour are simulator numbers.
- The CIC group-delay sign is pinned by a synthetic test that stamps a slow
  stream late the way the board does; it has not been checked against a
  board.
- The C++ PFB receiver path is verified in mock and on loopback only; tag
  `checkpoint-validate-pfb-receiver` marks where to validate it on a board.
- Verified on a board: the receiver's loss anatomy and the drain batching
  (serial 156, module 2, stage 0); PFB slot fields 0-indexed; the pairing,
  pileup and lag fixes in the dual capture; the r1.6 `module=` spelling.
- In both mode the two engines share one event loop, so a slow host that
  cannot keep up with the fast stream also delays the slow one. The fast
  engine can still fall behind on a busy host; the source then discards and
  counts, and pairs go out without their fast windows.
- The noise training record is held whole, so on the fast stream it is
  capped at 2 M samples per channel, about 0.82 s.
- The mock generates both streams in one process, so fast and dual captures
  run slower than real time; a 100-tone slow stream with every resonator
  pulsing runs at about 0.95x real time.
- The mock biases module 1 only, so only module 1 carries tones.
- The Swenson form is not the simulator's full kinetic-inductance physics, so
  on the simulator a fitted slope's direction is up to 4 degrees off a true
  frequency step; a real detector will disagree with the model in its own way.
- A fast pulse shorter than one slow sample shows two slow samples about its
  centre in the pair view; it is detected but not resolved on the slow
  stream, and a mock decay shorter than the slow sample period looks like a
  spike.
- `test_high_sampling_rate` in the hardware tier fails on an undersized UDP
  buffer or a slow machine and reads as a code bug; set `rmem_max` first.
- No workflow runs `test_fastrx_file.py`; the fastrx build needs clang,
  libxdp, libbpf and liburing at install time on Linux.
- `pulse_capture_flow.py` is not executed by any test; run it by hand when
  its notebook changes.
- Two acquisition runs, or a run beside a Periscope holding 9876/9877, share
  or starve each other's packets and present as detector bugs; check
  `ss -ulnp` before believing one.