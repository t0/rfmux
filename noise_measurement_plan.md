# Shared headless noise measurement

Status: implemented, including the Periscope migration, 2026-09-22.

## Review and first implementation

The shared acquisition boundary fits the existing helpers and persistence.
The implementation is `rfmux/algorithms/measurement/measure_noise.py`.
It reuses the measurement wrapper. The current noise layout is schema version
11; the incompatible field-name cleanup moved every packed measurement to that
shared version.

```python
noise = await crs.measure_noise(
    catalog, num_samples=10_000, nsegments=10, save=False)
block = noise[crs.module[catalog.module].index()]
record = block["results"]["resonators"][catalog.names()[0]]
volts = record["slow_data"]["iq_volts"]
frequencies = block["results"]["shared_slow"]["freq_iq"]
```

Implementation decisions from the source audit:

- Both helpers return volts for absolute TOD and counts for relative TOD.
  The routine preserves those units in `iq_volts` or `iq_counts`, with
  `info.iq_units` set to `volts` or `counts`; it copies spectra unchanged.
- Both relative spectral helpers multiply carrier-bin densities by the FFT
  bin width; this single-bin estimate omits Hann-window leakage. `spectrum_units` is `dBc/Hz`, but `carrier_bin_units` is
  `dBc`; the carrier bin is the nearest-to-zero frequency of each spectrum.
- Firmware cannot read back packet width or the streamed-module selection.
  A different explicit decimation therefore selects the requested module
  alone, short packets below stage 3 and long packets otherwise. This is a
  persistent configuration change. Omitted or unchanged decimation causes
  no configuration call. Disabled slow streaming is rejected.
- Channels outside a known short-packet constraint are rejected before
  mutation. With an unchanged setting at stage 3 or above, actual packet
  width is only available after capture; missing channels are rejected then.
- Slow segments require at least two samples. PFB segments require four
  for the helper's symmetric Hann window and a non-DC reconstructed I/Q bin.
  PFB uses the helper's documented maximum of 10,000,000 samples.
- Explicit channel names are `CH{channel:04d}`. `call_params.channel_names`
  maps names to channels. Shared slow fields are `timestamps`, `freq_iq`
  and `freq_dsb`; each stream record has `iq_volts` or `iq_counts`, `psd_i`, `psd_q` and
  `psd_dual_sideband`. PFB keeps channel-specific spectral axes in each record
  and shares its nominal `time_s` axis at the results level.
- `rfmux.tuning.noise_to_df` returns a copy-on-write module block. Calibrated
  stream records retain those native arrays and gain complex `df_hz`,
  `psd_df`, and `psd_dissipation`; their units are Hz and Hz²/Hz. It recomputes
  component spectra from the rotated timestream so I/Q correlation is not
  discarded. No display products are stored in the measurement.
- Progress is a synchronous callback receiving a dictionary after each
  capture. Cancellation propagates at existing awaits; synchronous spectral
  processing and saving are not interruptible. No latency bound is promised.

The new files are readable through `store`, but Periscope's existing noise
loaders still expect legacy dictionaries. GUI compatibility and removal of
its duplicate callers remain in stages 3–4. Single-module mock UDP acquisition
is verified by the demo runs below. Hardware calibration and multi-module mock
acquisition remain unverified.

The existing `py_get_samples` special case for serial `MOCK0001` bypasses
module filtering as well as timestamp filtering. Multi-module mock acquisition
needs that lower-level issue addressed before it can validate this wrapper's
module selection; the first-stage tests use controlled helper returns.

Verification: 36 new portable cases cover mapping, metadata, validation,
PFB helper output, progress, interrupted acquisition and persistence. The
portable suite passed with 797 passed, 4 skipped and 2 xfailed in 24.22 s
with local socket access. The sandboxed attempt had 5 failures and 13 setup
errors from `PermissionError: [Errno 1] Operation not permitted`; an initial
test fixture also hit `TypeError: Macro called with wrong types! Expected CRS,
got <class 'types.SimpleNamespace'>`, corrected by calling the unwrapped
macro for board stand-ins. The final focused run passed 43 tests in 0.30 s
(36 noise cases and 7 existing calibration cases). Production code grows by
228 lines including registration; the 250-line test module is additional.
Duplicate acquisition paths remain until caller migration.

## Demo migration and plotting

`simplified_tuning_flow.py` and its Markdown/IPython workbook now use the
public macro. The Python wrapper owns only its created mock sender; the
macro saves the completed noise container once. `noise_measurement.md`
starts with three biased mock resonators, verifies them with a multisweep,
reapplies the biases, acquires noise and reloads both files for inspection.
Paired `.ipynb` files are generated locally; Markdown remains tracked source.

`example_plotting_noise.py` provides `plot_iq_panels`, `plot_timestreams` and
`plot_psds`, each accepting one saved module block and returning figures.
IQ overlays read either an explicit verification multisweep or the catalog's
stored bias sweep, and require matching measured drive amplitudes. Plotting
uses volts by default, with a counts option. Timestream axes use nominal
sample spacing and independent capture origins. PSD plots retain the helper's
units; carrier omission is display-only, including a visible gap in signed
dual-sideband plots. This adds 199 plotting lines while the script loses 53
lines of duplicate acquisition/packing code. Periscope duplication remains.

Validation: 103 focused tests pass, including 24 new plotting cases. The
script, new noise workbook and simplified workbook all passed against mock
RPC/real loopback UDP, sequentially in 283.80 s. The script test took 112.317 s
before migration and 118.312 s after; these are entire stochastic tuning runs,
not evidence of a noise-acquisition speedup. The new workbook took 31.549 s
and the simplified workbook 132.473 s. The saved ten-tone noise file retained
5,794,888 NumPy-array bytes and occupied 5,892,401 bytes on disk; no before/after
retained-memory comparison was completed. Figures were visually inspected.
After the carrier-gap and axis-label corrections, all 61 plotting tests and
the noise workbook passed again (62 tests, 32.41 s). The four reload/plot
cells were also executed in a fresh kernel with only saved paths and imports,
without a CRS or the acquisition cell's parameter variables. Both local
notebooks contain executed outputs and match their Markdown cell sources.

During test development, a fixture failed with `KeyError: 'sweep_direction'`;
its missing sweep metadata was supplied. Mean-subtraction assertions also
reported `Not equal to tolerance rtol=1e-07, atol=0` for a 5.29e-23 V residual;
they now allow floating-point roundoff. Final runs have no test failures.

## Objective

Add one headless noise measurement routine that Periscope, scripts and
notebooks can call. It measures already configured tones, returns the
project's standard measurement container, and uses `rfmux.tuning.store`.
It does not select or apply bias points.

Reuse the existing acquisition and spectral calculations. Consolidate the
orchestration currently repeated in Periscope and the simplified tuning
demo. Command echoing and the `tcl_echo` interpreter infrastructure are a
separate project.

## Existing pieces

| Location | Responsibility today |
| --- | --- |
| `rfmux/algorithms/measurement/py_get_samples.py` | Slow-stream UDP acquisition; optional spectra and time-domain statistics. Requires a running readout stream and valid timestamps. |
| `rfmux/core/transferfunctions.py:spectrum_from_slow_tod` | Offline Welch spectra and slow-stream filter compensation. |
| `rfmux/algorithms/measurement/py_get_pfb_samples.py` | One-channel RPC capture and PFB spectral correction; no PFB UDP sender required. |
| `rfmux/tools/periscope/app_runtime.py:_collect_channel_noise` | Explicit-channel noise acquisition, packaging and GUI dispatch. |
| `rfmux/tools/periscope/multisweep_panel.py:_get_spectrum` | Multisweep-associated noise acquisition and packaging. This path indexes channels as `1..N`. |
| `rfmux/tools/periscope/noise_spectrum_dialog.py` | Acquisition settings, duration/resolution estimates and PFB options. |
| `rfmux/tools/periscope/noise_spectrum_panel.py` | Timestream and spectrum display, navigation and binning. |
| `rfmux/reference-notebooks/Demos/simplified_tuning_flow.md`, `.py` | Public noise-macro caller with mock sender ownership; the script's `_acquire_noise` is a thin sender wrapper. |
| `rfmux/reference-notebooks/Demos/example_plotting_noise.py` | Saved-block IQ overlays, timestreams and PSD plots. |

The demo is the starting point for catalog/channel mapping. The two GUI
paths supply the settings and display requirements. Neither existing
noise payload is the standard `{module_id: block}` measurement schema.

Live rolling PSD plots and pulse-capture noise training/periodic noise
windows serve different purposes and remain separate consumers of the
existing lower-level facilities.

## Public API

`measure_noise` is registered as a CRS macro in
`rfmux/algorithms/measurement/measure_noise.py` and exposed through the
measurement registration path.

Current signature:

```python
async def measure_noise(
    crs,
    catalog=None,
    *,
    channels=None,
    module=None,
    decimation=None,
    num_samples=10_000,
    nsegments=10,
    reference="absolute",
    spectrum_cutoff=0.9,
    pfb_samples=None,
    pfb_nsegments=None,
    progress_callback=None,
    save=None,
    label=None,
):
    ...
```

- A catalog provides names, actual channel bindings and the module. An
  explicit module must match it. Take a snapshot before acquisition.
- Without a catalog, require a module and an explicit channel list. Give
  these records stable channel-derived names, recording the naming rule.
  Reject simultaneous `catalog` and `channels` inputs initially.
- Support one module per call initially, matching a catalog and Periscope's
  active-module model. The returned container still uses the board/module
  identifier from `crs.module[module].index()`.
- `decimation=None` preserves the current setting. A different explicit
  value configures a supported packet mode and selects this module alone.
  The GUI passes its chosen setting explicitly.
- `pfb_samples=None` disables PFB acquisition. Otherwise acquire each
  selected channel sequentially after the slow capture.
- `pfb_nsegments=None` uses `nsegments`; keep independent segment counts
  available because the streams can have different sample counts.
- `save=None` follows `store` autosave, as the other drivers do. Periscope
  may pass `save=False` and save the returned container in its session.
- Progress is coarse acquisition progress, with an explicit stream/channel
  identity and completed/total work; do not invent per-sample progress or
  promise that fractional completion equals fractional elapsed time.

Validate the full request before changing hardware: nonempty, unique valid
channels; valid catalog bindings and module; supported decimation; integer
sample and segment counts with enough samples per segment; finite cutoff
in `(0, 1]`; supported reference. Use documented/system acquisition limits,
not GUI-only arbitrary caps.

## Acquisition and hardware contract

1. Resolve channel/name bindings and validate inputs.
2. Read relevant board settings and the module DAC scale. Record actual
   tone frequencies and amplitude fractions separately from the supplied
   catalog snapshot; a catalog is not proof of what the board is driving.
3. Configure decimation only if explicitly requested and different.
4. Use `py_get_samples(return_spectrum=True, scaling="psd", channel=None)`
   for one slow capture, retaining only requested channels in the result.
5. If requested, use `py_get_pfb_samples` for each selected channel, with
   `reset_NCO=False`, initially retaining the callers' `binlim=1e6` and
   `trim=False` settings and recording them.
6. Pack the completed measurement and save through `store` if requested.

Slow and PFB captures are sequential and must not be described as
synchronized. The function preserves tone programming and does not start
or stop a user's stream. It requires the slow UDP stream to be available.
An explicitly requested decimation change, packet width and selection of
this module alone remain in effect; document these side effects.

Keep mock stream ownership in the demo/launcher. Reuse its conflict check
and `try/finally` cleanup, stopping only a sender it started. The mock RPC
PFB capture produces synthetic uniform noise; the slow stream uses the
resonator model. Preserve that distinction in documentation and tests.

Cancellation propagates through acquisition. Do not save or publish a
partial result as a completed measurement. Verify cancellation against
the existing lower-level routines before promising interruption latency.

## Result and persistence contract

Return `{module_id: block}` with `schema_version`, `measurement="noise"`,
`module`, `dac_scale_dbm`, `call_params`, and `results`. Use the existing
packing conventions; factor out a shared wrapper only if needed, rather
than duplicating schema or file-metadata logic.

Proposed layout:

```text
module_id
  schema_version, measurement, module, dac_scale_dbm
  call_params
    catalog snapshot or explicit channel/name mapping
    requested acquisition arguments
  results
    info: resolved decimation, sample rates, reference and units
    shared_slow: shared timestamps and frequency axes
    shared_pfb: shared nominal time axis, or None
    resonators
      name
        channel, bias_frequency_hz, bias_amplitude, bias_amplitude_dbm
        slow_data: native IQ and I/Q/dual-sideband spectra; calibrated
                   df_hz and df/dissipation spectra when available
        pfb_data: IQ arrays, channel-specific frequency axes and spectra
                  plus calibrated df products when requested and available
```

Finalize exact field names and schema versioning alongside a small packer
test before migrating callers. Shared slow axes should not be copied into
every resonator. PFB axes may differ by channel and stay with each record.
Retain actual slow timestamps in a builtins/NumPy representation; generated
PFB relative times must use the sample spacing (`arange(n) / sample_rate`),
not an endpoint-inclusive interval.

Time-domain units must be explicit. Prefer retaining complex raw counts
and using established conversions for volts, consistent with the tuning
measurement contract. Audit the slow and PFB helper conversion paths before
implementing this: do not relabel volts as counts, apply a conversion twice,
or silently change the established spectral calibration. Record spectral
reference/units explicitly; retain the carrier/DC bins in saved spectra.

Store measured amplitude fractions and the DAC scale, deriving dBm labels
through the existing transfer functions. Missing board values remain
missing rather than being substituted with zero.

Use `store.save/load/maybe_save` and existing session registration. Save one
noise measurement, including the catalog snapshot when supplied, rather
than embedding a second full multisweep export. A noise file must open in
Periscope without its originating multisweep panel or a live board.

## Implementation stages

1. **Headless routine and schema.** Add registration, input validation,
   acquisition orchestration, packing and persistence. Keep spectral
   calculations in the existing helpers. Pin sparse/nonconsecutive channel
   mappings and units before changing GUI callers.
2. **Demo migration.** Replace `_acquire_noise` and the notebook's repeated
   acquisition/packing loops with the public call. Keep stream ownership
   around it. Update plotting to read the shared format and retain the
   short demo acquisition settings and mock explanation.
3. **Periscope migration.** Route both entry points through one thin worker
   calling the new routine off the GUI thread. Pass catalog bindings from
   the multisweep panel and explicit channels from the main window. Relay
   progress/completion with signals, offer cancellation, and use status
   text for routine outcomes. Remove duplicate acquisition and packaging.
4. **Display and files.** Make the noise panel read the shared block;
   update session typing, saving and loading together. Keep plot-only
   binning and carrier-bin omission out of measurement data. Remove
   positional `1..N` assumptions and redundant legacy noise dictionaries.
5. **Review and documentation.** Check units, defaults, tooltip estimates,
   file examples and test tier counts against the implementation. Report
   the resulting code size and any remaining duplicate paths.

## Verification

- Headless contract tests: catalog and explicit channels, sparse/reordered
  bindings, validation before hardware mutation, optional PFB, tone
  metadata, decimation behavior, units and schema.
- Persistence tests: builtins/NumPy payload, save/load round trip, same
  file on re-save, and one completed measurement per requested save.
- Failure/cancellation tests: no success signal or completed file on
  interrupted acquisition; existing stream ownership is preserved.
- Preserve existing spectral normalization tests, including
  `test/notebooks/test_py_get_samples.md`, carrier tests and dBm tests.
  Add focused numerical checks only for changed conversions/packing.
- Periscope tests: both entry points consume the same contract, controls
  remain responsive, nonconsecutive catalog channels are measured, and an
  exported file loads without a board.
- Compare headless and GUI results on the same settings for schema,
  channel mapping, units and metadata. Do not demand equal stochastic
  samples from independent noise captures.
- Run the simplified demo notebook and script using the existing mock/UDP
  acquisition test infrastructure. Confirm that their owned sender stops
  on failure as well as success.
- Measure a representative run's elapsed time and retained array sizes
  before/after consolidation; do not claim a performance improvement
  without measurements. Report hardware/PFB calibration checks separately
  when no physical board is available.

## Judgement calls and boundaries

- Preserve current decimation by default; the GUI supplies its existing
  default of 6 explicitly. Default normalization is absolute, matching the
  current GUI and tuning demo rather than `tcl_echo`'s relative default.
- Support both catalogs and explicit channels, but one module per call.
- Keep acquisition separate from applying bias and managing stream owners.
- Reuse the current spectral estimators and corrections. Cross spectra,
  coherence, calibrated frequency-noise/NEP spectra, synchronized slow/PFB
  capture and new detector-noise physics are outside this change.
- Keep console command echoing outside this implementation.
- Existing GUI noise pickles and demo noise dictionaries are not the new
  schema. Inventory their loaders before removing them; prefer a small
  explicit loader conversion where sufficient metadata exists. Never
  fabricate missing units, channel identities or catalog bindings. Document
  unsupported files clearly. Exact compatibility coverage remains to be
  decided from that inventory.
- Schema field names and count conversion are implemented above. Legacy-file
  compatibility remains a checkpoint for the Periscope migration.
