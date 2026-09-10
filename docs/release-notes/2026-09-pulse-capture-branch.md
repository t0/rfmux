# Pulse capture branch: release notes

Branch `buffer_exploration`, PR 78, September 2026.

This release adds pulse capture to rfmux: a detection engine that triggers on
a detector timestream at a threshold set from its own measured noise and
records each pulse to HDF5 as it arrives, with its summary statistics. It
pairs events seen on the slow readout stream and the fast PFB stream. It runs
headlessly as the `trigger_capture` macro and interactively as a Periscope
panel, and the two share one engine, one ingest path and one file format.
Samples are stored in physical units, volts or hertz, never in ADC counts,
and the file carries the constants and calibration that produced them. The
guide is [docs/guides/pulse-capture.md](../guides/pulse-capture.md). This
document is for people working on rfmux: what else changed and what to
change when upgrading.

The df calibration is measured by `bias_kids`, which steps every biased tone
and reads the slope. The nonlinear resonator model is Swenson et al. 2013
eq. 13. The simulator has 1/f frequency wander, streams over multicast like a
board and generates its slow stream a block at a time. The C++ receiver hands
Periscope and the fast source demuxed arrays. Tests are organised by
subsystem and run by tier; CI runs every test that does not need a board on
three platforms.

The C++ extension changed: rebuild after merging with
`source .venv/bin/activate && uv pip install -e .`. Pulse-capture files
recorded on hardware before this branch hold volts and hertz 256 times low;
simulator files are unchanged in meaning.

## New

- `rfmux.pulse_capture`: a top-level package holding the detection engine,
  its compiled per-sample walk (`walk.py`), the analysis helpers, the
  histogram and template accumulators, the HDF5 writers and reader, the
  stream sources and the single- and dual-stream capture sessions.
- `crs.trigger_capture(channel, module, streamer_mode="slow", time_run=10.0,
  config, threshold_sigma, end_sigma, max_pulse_ms, hdf5_path,
  df_calibrations, trigger_basis)`: one-shot capture in slow, fast or both
  modes.
- The result carries the pulses per channel, the pairs and the per-stream
  results; with `hdf5_path` the same content is written as the capture runs.
- Noise training before every capture: the threshold is `threshold_sigma`
  times a sigma measured as the samples' scatter about a block-median
  baseline, so it holds for the correlated samples the decimators produce.
  Training lasts the 1/f window, `noise_train_ms` (5 s), and is not
  charged against `time_run`.
- Triggering in the frequency basis (`trigger_basis="df"`): with a df
  calibration a channel is rotated so the pulse lies along one axis before
  thresholding, and stored in hertz; without one it stays on the quadratures
  in volts. The file records
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
  resolution_hz=500)`: a sweep of every channel at once (all biased channels
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
  `PacketReceiver.flush_all`, and a
  `packets_missing` counter that counts packets rather than gaps.
- `rfmux.core.transferfunctions`: `PFB_NYQUIST_FREQ` beside the existing
  `PFB_SAMPLING_FREQ`, stated as not a rate; the CIC parameters as constants;
  `decimated_stream_delay_s`, `sampling_to_decimation`, `apply_iq_conversion`.
- Periscope: app-wide zoom (Ctrl+, Ctrl-, Ctrl+0, persisted); flow-layout
  toolbars that wrap to a laptop width; a framed progress window for large
  mock builds; the mock's df calibrations measured at startup.
- Two runnable jupytext notebooks, `pulse_capture.md` and
  `simplified_tuning_flow.md`, executed in CI; `pulse_capture_flow.py` as the
  script twin; the pulse-capture guide with reproducible screenshots.
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
  slope; it is measured by a tone step, in hertz per volt.
- Nonlinear resonator model: `yg = y + a/(1+y^2)` solved by Newton, to
  `y = yg + a/(1+4y^2)` (Swenson 2013 eq. 13) solved by bisection with a
  Newton finish. Fitted `a` values from earlier releases are not comparable.
- Simulator defaults: `nqp_noise_std_factor` 0.001 to 0.01; TLS wander absent
  to on (1e-7 fractional RMS, alpha 1.0, corner 100 Hz); `pulse_tau_decay`
  0.1 s to 5 ms; `pulse_random_amp_min`/`max` 1.5/3.0 to 1.1/1.5.
  `bias_amplitude` 0.01 (about -40 dBm) to `bias_amplitude_from_dbm(-55)`,
  about 0.0016. `DAC_SCALE_DBM` (1 dBm) and `BIAS_DBM` (-55 dBm) live in
  `rfmux.mock.config`.
- Simulator readout floor: `udp_noise_level` 0.04 counts, a placeholder, to
  11 counts, the slow-stream sigma of board 0156 (firmware v1.7.0rc4) at
  stage 6 with no tone through a detector chain. The PFB stream's sigma is
  `PFB_NOISE_OVER_WHITE` (0.86) times the white-noise extrapolation, the
  board's ratio of 55 against 64.
- Simulator `get_pfb_samples` stub: normalized values by default, to counts,
  what the board's RPC returns (it takes no units argument).
- Simulator transport: unicast to 127.0.0.1 with multicast TTL 1, to
  multicast on the hardware group with TTL 0, falling back to loopback unicast
  and printing the failing step when the host cannot multicast.
- Simulator auto-bias: capped at 256 channels, to as many as a packet carries.
- Simulator dip search: a 2000-point sweep over +/-10 MHz per resonator, to a
  50 kHz pass over +/-0.25% then a 101-point sweep over 200 kHz.
- Simulator physics: the multi-sample path evaluated one sample at a time, to
  a batch path that evaluates the shared terms once per instant
  (`physics_batch_mode` "hoisted"; "reference" keeps the loop). The slow
  stream generated one frame per call, to about 50 ms of frames per call.
  PFB packets of 64 samples, one per physics sub-batch, to 1000-sample
  packets, the hardware's size.
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
  `pytest` applies `-m "not slow_acquisition"`.
- CI: two named test files to the quick and acquisition tiers on ubuntu,
  windows and macos with the test dependency group installed; `paths-ignore`
  covers READMEs, `AGENTS.md`, `CLAUDE.md`, `CHANGELOG.md` and `docs/**`, so the jupytext
  demos trigger CI and changes under `docs/` do not.
- Networking guide: one `rmem_max` value, 268435456 (256 MB, about three
  seconds of the four-channel PFB stream).
- Pulse capture record: the saved window runs from the pre-trigger margin to
  the sample the pulse settled on, the first of the in-band run the end
  confirmation then verifies, or to the hard stop, and a hard stop always
  flags `truncated`. `duration_ms` runs from the trigger to that settled
  sample, carried in the record as `settled_index` and `settled_time`; the
  drop below threshold stays as `below_threshold_index` and
  `below_threshold_time` and feeds the decay constant. The Pulse View marks
  all four. `save_to_end_confirmed` is gone from `PulseCaptureConfig`,
  `PulseCaptureSession`, `PulseCapture` and the Settings dialog. A file
  written with it still opens; its stored value is ignored, and its pulses
  keep the duration they had, trigger to the threshold drop.
- The 1/f window (`noise_train_ms`): derived as twenty times the max pulse,
  to its own default of 5 s. The noise fit and the rolling baseline want
  seconds whatever the pulse length, and deriving them from a short max
  pulse refreshed the baseline often enough to fall behind the stream at
  128 channels (measured: 109% of real time at 20 ms against 45% at 250
  ms, stage 1). The dialog's Noise training row is now the editable 1/f
  window, with a warning below 2 s; 0 still derives it.
- Peak-amplitude histogram: the larger of the two axis excursions, to one
  histogram per stored axis (`amplitude_i`, `amplitude_q`, shared bins),
  overlaid on the Histograms tab and named by the stored basis. A channel
  stored in the frequency basis also keeps the raw-quadrature pair
  (`amplitude_raw_i`, `amplitude_raw_q`, volts), binned from each pulse's
  waveform turned back with its calibration, and the quadrature views draw
  that pair. The per-pulse `peak_I`, `peak_Q` and `peak_amp` attributes are
  unchanged.
- `end_sigma` default: 1.5, the value the panel shipped with; 1.0 was tried
  on the mock and kept the record about a fifth longer for the same pulses.
- Streamer configuration, temporary firmware workaround: the PFB streamer
  command checks the link budget with a miscalculated slow-stream rate, so
  `apply_streamer_config` enables the fast stream with the slow stream at
  stage 6 and applies the wanted stage afterwards. Two calls again once the
  firmware that fixes the check is the minimum.
- Both mode with a partial PFB streamer: the capture takes every channel on
  the slow stream and fast data for the captured channels the PFB streamer
  carries (`pfb_streamed_channels` in `pulse_capture.sources`), warns which
  ones on the status line and console, and refuses only when none is
  streamed. `DualPulseCaptureSession` takes `fast_channels` (or
  `set_fast_channels` before start); a slow pulse on a channel without fast
  data is a one-sided pair at once; the file carries `fast_channels`;
  `PulseCaptureResult.fast_channels` reports it. A fast capture still needs
  every channel streamed.
- Periscope Template tab: its own stream selector in both mode; the
  histogram tab's selector no longer drives the templates.
- Pileup split confirmation: the rise above the pulse's own recent level
  must hold for `trigger_samples` consecutive samples, the confirmation
  length the trigger derives from the sample rate. On a tail the amplitude
  test is always satisfied, so the split had no confirmation and one noise
  sample against one sample ten back decided it, tried on every tail
  sample: about 3% of pulses split falsely on the PFB stream (on the mock at
  ten pulses per second, one-sided pairs fell from 42 to 7, the rest genuine
  doubles the slow stream cannot resolve). One sample
  at 596 Hz, so nothing changes there.
- Pre-pulse anchor: the edge tap nearest the tracked mean rather than the
  median of the three. At tens of pulses per second two taps can land on
  earlier pulses, and the median was then a pulse level, so the end band
  sat far from the baseline and the return test judged against it.
- Dual capture file: `min_pulse_ms`, `max_pulse_ms`, `noise_train_ms`,
  `trigger_samples_slow` and `trigger_samples_fast` are recorded.
- Periscope pulse list: a slow- or fast-mode row carries the marker and
  the pulse index; a both-mode row reads "slow + fast", "slow only" or
  "fast only" instead of "Pair #n", with the pileup or truncated marker
  taken from its summaries.
- Split child: dated at the onset of its rise, the sample of least
  deviation in the near window before the confirmed rise, as a trigger is
  dated to the start of its run rather than the sample that confirmed it.
  It was dated `min_end_samples` before the split, ten samples, which at
  19 kHz put the slow child half a millisecond before its fast twin and
  outside the match window; dating at the confirming sample instead left
  the slow child 0.1 to 0.2 ms after it, since the slow rise test only
  clears the level half a millisecond back a few samples in. It keeps the
  parent's pre-pulse anchor instead of the tail level at the split, since
  both pulses return to the same level; the end band in the Pulse View
  sits there.
- Pileup split test: the rise above the pulse's own recent level, and the
  decay evidence that arms it, are judged against the larger of the trained
  jump σ and the scatter measured inside the capture (a clipped average of
  the deviation magnitude's second differences over `min_end_samples`).
  At 38 kHz the mock's quasiparticle noise, which grows with the pulse,
  split every pulse's tail into fragments a few milliseconds long; the
  decision path is otherwise unchanged and the walk costs the same.

## Fixed

Bugs present on main, with the symptom.

- Periscope lost packets to its own receive path: at stage 0 a 128-channel
  capture with one channel on screen reported 38% to 52% loss as "net".
  Widening a capture to a few hundred channels froze the window in an
  unbounded read of the receive queue. The receive thread reads up to 2048
  datagrams per call over a scratch allocated once, the display writes a
  frame at a time, Periscope's packet hand-off gives the capture worker whole
  packets, and each pass over the queue stops after 250 ms. On the board
  (serial 156, module 2, stage 0): 128 channels at stage 0 with nothing
  dropped.
- Ctrl+C did not exit Periscope: the window stayed up holding UDP 9876, and
  the next launch bound the port and received nothing. SIGINT closes the
  windows and quits; a second Ctrl+C leaves at once; the receive thread is
  joined before exit.
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
  explanation; Periscope says another receiver holds the port.
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
  one packet per frame with its own sequence number and stamp: 100 tones run
  at real time with pulses off.
- Batch physics (`physics_batch_mode="hoisted"`): pulse sum, QP noise draw, nqp to (R, Lk) kernel and
  TLS lookup evaluated once per instant of a batch, convergence-cache
  decisions in the reference order, parity with the reference loop at 1e-9.
- Coupled pairs found through the sorted tones instead of every observer/tone
  pair: a 1023-channel `get_samples` chunk on the mock takes 29 ms.
- TLS wander is a per-resonator capacitance perturbation, a pure function of
  absolute time so the slow and PFB streams stay common-mode; a sum of
  Ornstein-Uhlenbeck processes with log-spaced corners.
- White QP noise is applied after the convergence-cache restore through a
  sensitivity linearisation, so the cache still applies.
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

The Pulse Capture panel is described in the how-to. Beyond it:

- Pulse view: each axis has its own baseline and bands drawn from the band
  the decision was made against; the info line names the quadrature that
  fired.
- The capture reads the board's PFB streamer state and never sets it; a
  mismatch between streamed and requested channels fails before a socket
  opens, naming both and where to change it.
- Streamer Configuration dialog: decimation, short packets (forced below
  stage 3), modules, PFB channels; derived rate, Nyquist, channels per packet
  and Mbps against the 1 GbE budget; the validation tiers in a banner.
- Toolbars are flow layouts that wrap at a laptop width.
- Mock startup: a framed progress window for arrays above 25 resonators; the
  df calibration sweep runs in a `DfCalibrationTask` while the window streams,
  reporting on the status bar; picking df units mid-sweep says so rather than
  starting a second sweep.
- Find Resonances dialog: a "require isolation" checkbox, off by default.
- Bias KIDs dialog: the fit method choice, preselected to the fit the sweeps
  carry; the nonlinearity threshold greys out under the skewed fit; the
  phase-step control is gone. A df Calibration group carries the headless
  options: measure by a tone step (on) and the step as a fraction of the
  fitted linewidth (0.05).
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
  batch fitter (on the simulator about 20 ms per sweep nonlinear, 4 ms
  skewed) and writes the fit back onto the entries. It reads the bias
  frequency (max-diq or min-s21) off the fitted curve and writes that back
  too.
- The measured calibration: every tone steps `calibration_step` of its
  fitted linewidth down and up together, two module reads, the inverse of
  the complex slope; a step is never less than one grid step and the fit
  supplies a curvature correction. Samples that do not move, or a failed
  read, fall back to the fit; one warning names detectors where fit and
  measurement disagree by more than 5 degrees or a magnitude ratio outside
  0.7 to 1.4.
- Tones stay on multiples of `TONE_GRID_HZ` (625 MHz / 2^21, about 298 Hz)
  because intermodulation products land on the grid; the calibration step is
  rounded to it too.
- The ADC phase: with `optimize_phase=True` the principal axis of (I, Q) goes
  to Q from one sample set.  The board turns samples by minus the phase it
  is given (measured on board 0156, firmware v1.7.0rc4), so the phase is
  the axis angle minus 90 degrees and the calibration turns by plus the
  phase; the simulator turns samples the same way.  The multisweep zeroes
  the ADC phase on the channels it sweeps.
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
- `streamer_config.validate`: stage 0-6; long packets need stage 3 or above.
  More than 1000 Mbps is an error and more than 800 Mbps (the firmware's
  derating) a warning. Stage 1 and below advise on the OS buffer; more than
  one module below stage 5 is noted as unvalidated; PFB channels at most four.
- `apply_streamer_config` sends `module=` (firmware r1.6 spelling); the mock
  mirrors the firmware signature so the next rename fails in tests.
- `apply_iq_conversion` and `convert_iq_to_df` sit together in
  `transferfunctions`; `storage_transform` and `display_transform` stay in
  `pulse_capture` as that package's policy.

## Streamer and packets

- `pop_readout_batch(max_packets)`: samples as a (packets, channels) complex
  array with the packetizer gain out, one packet width per batch. Beside
  them: seconds of day per packet (NaN when the timestamp is not locked to a
  source), recent flag, stage, sequence numbers, and the day from the first
  locked stamp.
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
  shared by `run_slow_source` and Periscope's capture hand-off.
- Slow packets are fed in timestamp order, the newest four held back for
  late packets at each flush; the sample clock is monotonic across a
  decimation change and the day boundary.
- The PFB source picks its channels from the packet's slot fields, so any
  subset of the streamed channels can be captured, and keeps only its
  module's packets.
- PFB packets are released in sequence order through a 64-packet window and
  fed in blocks of sixteen packets per channel.
- The PFB socket asks for the largest receive buffer the host allows
  (`net.core.rmem_max` on Linux) and reports the seconds it holds as
  `buffer_s` in the source stats.
- Past a 0.25 s lag between the slow and fast clocks the fast source discards
  packets by stamp until it is within 0.125 s, counting them as
  `flushed_packets`. `lost_packets` comes from the queue's own sequence
  accounting. `busy` is processing time over wall time.
- Every slow capture shifts its timestamps by minus the CIC group delay
  (2.8 to 3.0 slow samples at stages 3 to 6, 1.5 at stage 0; 4.99 ms at
  stage 6) before the engine, the matcher and the file, so slow and PFB
  clocks share one axis. The shift is recorded as `slow_time_offset_s`, 0
  when not applied; pass `time_offset_s=0.0` (`slow_time_offset_s=0.0` on
  the dual session) to opt out. The parser's dirfile `timebase` applies
  the same shift per packet from its `dec_stage`; the raw stamp fields
  are unchanged.
- `rfmux record --serial <NNNN> --module <M> --duration <s> --session <folder>`
  takes a slow-stream pulse capture, a parser dirfile and a fastrx
  recording of one module for the same stretch into one session folder,
  reading the board only. The parser is up before the capture starts and
  the fastrx writer starts when the capture's noise training ends; the
  channels and df calibrations come from the session's newest bias
  export. `trigger_capture` gained
  `on_noise=` for that. After the run it lists the channels that
  triggered, merges the recording into the pulse file as its fast
  stream (`--no-merge-fastrx`; `rfmux fastrx merge` for an older run)
  and opens Periscope in review mode on the file (`--show overlay` for
  the overlay viewer on the busiest channel, `--show none`). The merged
  file is a both-mode file that Periscope reviews with the recording
  under every pulse. `periscope --review <pulse.h5>` opens any capture
  file that way, offline, without the startup dialog. With no options
  the command opens a dialog with every choice, remembered between
  runs, the pulse capture settings on their own tab and the newest
  session under the default path filled in; it checks for fastrxd and
  shows the command to start it. See the 100G and 1G overlay guide.
- Pairs form on trigger instants within half the CIC2 response, three slow
  samples.
- A trigger with no partner waits the hard stop (1.2 times `max_pulse_ms`)
  plus 50 ms before it is reported without a partner.
- Each stream's window is taken when its own ring buffer covers it; a stream
  that
  never delivers one is given `pair_window_wait_s` (3 s).
- PFB slot fields are 0-indexed on the wire, like the module field.
- The rate: the PFB stream is `PFB_SAMPLING_FREQ`, 625 MHz / 256, about
  2.44 MHz per channel; 625 MHz / 512 is its Nyquist frequency and bin spacing.

## Tests and CI

Tiers, markers, layout, platform skips and CI triggers are in
`test/README.md`. Contract tests on the branch:

- block ingest against per-sample ingest, bitwise;
- the compiled per-sample walk (`walk.py`) against the Python loop and the
  uncompiled walk;
- the batched getter against the per-packet conversion through a real
  receiver on loopback;
- the CIC delay sign against a synthetically late-stamped slow stream;
- mock batch physics against the reference loop at 1e-9;
- the mock's carrier parity across decimation stages.

## Documentation

- Guide: [docs/guides/pulse-capture.md](../guides/pulse-capture.md).
- Notebooks: `rfmux/reference-notebooks/README.md`.
- Tests: `test/README.md`.
- `docs/make_pulse_capture_screenshots.py` regenerates the screenshots from
  the mock configuration it names; `docs/make_pulse_capture_figures.py`
  draws the anatomy figure from the engine's output on a synthetic pulse.
