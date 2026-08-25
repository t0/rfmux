# Quality-of-Life Improvements

Survey of ~230 Cline task histories (Jan–Aug 2026) in
`~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/tasks/`, filtered to
work that reads as **UX / usability polish or ergonomic fixes** — things that
don't touch large structural elements of the code, nor resonator measurement,
biasing, or calibration algorithms.

**Excluded by request:** JPL pulse data, `hidfmux`, `channel_noise_plots`
(~27 tasks). Also excluded: large refactors, data-schema renames, new
measurement/calibration features, test infrastructure, memory-bank/docs updates,
and read-only Q&A sessions (~90 tasks).

**Scope note:** the list is deliberately inclusive — borderline items are kept and
flagged in [Borderline](#borderline--ux-driven-but-touches-structure-or-measurement)
so they can be pruned later. Each entry is the Cline task ID.

Totals: **101** clear QoL items, **12** borderline, **6** from a separate project.

Entries marked *(plan only)* were planned in Cline's plan mode and never
implemented — 15 of the 244 sessions never left plan mode, so they produced no
code changes. Check those against the current tree before acting on them.

---

## Plot layout, sizing & grid arrangement

| Task | What |
|---|---|
| `1769267780723` | Square IQ circle subplots — `square_axes` helper + `SquareGridLayout` in periscope `utils.py` |
| `1769273298898` | Second pass at square IQ axes in `multisweep_panel` |
| `1769275985165` | Fixed main window being forced very wide / unresizable when multisweep opens |
| `1769279147678` | Minimum of 4 columns in the sweep plot grid |
| `1769458395855` | Cache plot widgets to stop resize flicker when panels are displayed |
| `1769472526338` | Shrink batch-nav buttons, add explicit **Update** button and Size label |
| `1773520501105` | Split the crowded multisweep toolbar into two rows |
| `1774290189352` | Aggregate panels-per-page 8 → 6 for a better plot aspect ratio |
| `1775071230077` | Better plots-per-row (`ncols`) calculation in sweep, derivative and fit grids |
| `1775074093430` | Amplitude colorbar made vertical and moved to the right of the plot grid *(plan only — the tree still has a horizontal bar)* |
| `1779455877857` | Default aggregate plot count 6 → 8 |
| `1780516836504` | `batch_size` default raised 6 → 8 |
| `1780556866139` | Option to sort aggregate plots by center frequency |

## Axis labels, units, legends & colours

| Task | What |
|---|---|
| `1769279735331` | Apply **Normalize Traces** and unit modes to per-resonator plots |
| `1769445104494` | Amplitude legends and line styles on sweep and IQ plots |
| `1769458964485` | Apply trace normalization to IQ circle plots |
| `1769460885213` | Histogram legend labels match the Amplitude selector format |
| `1770136584944` | Broken **Normalize Traces** checkbox on multisweep IQ plots *(diagnosed only, no edits)* |
| `1773086578838` | Rework magnitude y-axis units and axis labels (`convert_amplitude`, `mag_axis_label`) |
| `1773326622254` | Bias-result plot styling — indicator lines, marker size, colours |
| `1773343287423` | Frequency display precision to 1 Hz; hide fit option when no fits exist |
| `1773352972271` | Fix grid x-axis label and plot-title fallback |
| `1773357389540` | Show the chosen amplitude `a=X.XXX` in the legend / mini-legend |
| `1773517649227` | Fix flat-looking IQ derivative and arc-length traces (display scaling, line style) |
| `1773518961481` | Stop exponential notation in the probe-amplitude label |
| `1773844479552` | Stable trace colours independent of iteration count |
| `1773864766124` | Split fit legend labels onto three lines (fr / Qr+Qi / Qc) |
| `1775058707304` | Move resonance count into the plot title; drop the unclickable legend entry |
| `1778896006394` | Correct PSD live-view units when Rotated IQ is selected |
| `1780516639209` | Colorbar dual labels — dBm plus normalized amplitude |
| `1781878947710` | Show probe amplitude in dBm alongside normalized units |

## Titles & tab naming

| Task | What |
|---|---|
| `1769474990070` | Plot titles show detector id and section center frequency |
| `1773856640316` | Rename multisweep tabs to clearer names |
| `1775060595039` | Measurement tabs show filenames instead of "Network analysis #1" |
| `1781876870484` | Network analysis plot title shows the filename |
| `1781877512553` | Filename / module / measurement name in multisweep plot titles |

## Dialogs — defaults, persistence, keyboard & focus

| Task | What |
|---|---|
| `1769281863849` | Enter acts as OK in the session startup dialog |
| `1769282799164` | Enter key bound to the default button in network analysis and multisweep dialogs |
| `1769293309820` | Rename / clear amplitude field labels and defaults |
| `1769436451996` | Pre-fill amplitudes, add **Clear** button, persist defaults via `QSettings` |
| `1773349030485` | Only offer bias frequencies when a bias was actually found |
| `1773356354084` | Simplify the Bias Settings panel; bind Enter to Close |
| `1773526577546` | Max derivative distance selectable as a fraction of sweep bandwidth |
| `1773538794095` | **Find bias** checkbox to auto-run bias after a multisweep |
| `1773684856702` | Persist network-analysis dialog defaults between runs |
| `1773687111497` | Replace the Find Resonances popup with a persistent settings panel |
| `1773865874098` | Amplitude selector: spinbox → dropdown in the Fit Results tab |
| `1774289677163` | Better default for expected resonances (1024 instead of `None`), silencing a confusing warning |
| `1774883868693` | Grey out "Number of steps" for a single iteration; steps default 10 → 5; dialog resized |
| `1774977370403` | Persist the Jupyter notebook library path instead of re-prompting |
| `1774987001662` | Amplitude field labels ("norm." / "dBm:") and tab order in the network analysis dialog |
| `1774989311343` | Choose which amplitude iteration Find Resonances uses (persisted) |
| `1775059209292` | Custom measurement-name suffix box in the measurement dialogs |
| `1775085518192` | Clearer placeholder text for the empty fit histogram panel |
| `1775085627321` | Fits plots always visible with an informative empty state |
| `1775086342330` | Removed confusing red field highlighting during amplitude validation |
| `1775092910099` | Sensible default of 0.005 for single-iteration global amplitude |
| `1775136506787` | Amplitude iteration radio buttons made mutually exclusive |
| `1775137123862` | Better placeholder hint values in the amplitude entry boxes |
| `1782159588043` | Note in the dialog that scaled base amplitudes come from `res_info_dict` |
| `1784877031170` | File dialogs open in the current session directory |
| `1785090389123` | Unify the load-frequencies / load-amplitudes UI across both multisweep dialogs |

## Amplitude & frequency import / re-run ergonomics

| Task | What |
|---|---|
| `1769443821872` | Save iteration metadata to simplify amplitude pre-fill |
| `1775065484288` | Editable sweep center frequencies when launching multisweep from netanal |
| `1775068368434` | Re-run netanal dialog was missing the naming fields |
| `1775071057869` | Custom suffix carried through into the multisweep re-run dialog |
| `1778030499524` | Button to load Find-Bias amplitudes into the multisweep dialog |
| `1782159846424` | Honour user-supplied center frequencies on re-run *(plan only)* |
| `1784639030285` | "Custom" re-run/load mode so typed frequencies are actually used |
| `1784788623089` | Sort imported pkl by bias frequency for consistent matching; tooltip/hint updates |
| `1784875565254` | New "Import from `res_info_dict`" button beside "Import from sweep" |
| `1785089361990` | Also populate the amplitude array when loading sweep sections *(plan only)* |
| `1785091532160` | Auto-switch the section combo to "Custom" after importing |
| `1785091973059` | Fix broken amplitude import from sweep files — `isinstance` guard in `_prepare_export_data`, wrong results key *(plan only)* |

## Sessions & file handling

| Task | What |
|---|---|
| `1769280878856` | Load-session dialog opens the parent directory with the session preselected |
| `1769463552575` | Skip / persist the Periscope config dialog when loading or creating a session |
| `1769465886909` | File browser preselects the last *active* session, not the last loaded |
| `1770132971520` | Auto-generate a mock-mode random seed when the field is left blank |
| `1773355453091` | Enable **Apply Bias** on load; auto-save bias results to the session |
| `1773684395161` | Consistent module identifier in session output filenames |
| `1780557707725` | Remember the default session directory; skip the second startup dialog |

## Status feedback, popups & log noise

| Task | What |
|---|---|
| `1769266231486` | Stop creating a duplicate aggregate plot panel |
| `1769267398576` | Don't auto-raise the detector digest panel when a multisweep starts |
| `1769459321733` | Generate histograms once when data is ready, not on every tab open |
| `1769462654067` | Remove leftover `[DEBUG]` print statements |
| `1773082429579` | Drop amplitude from progress-bar labels; assorted post-merge UI fixes |
| `1773520945681` | Replace the "bias applied" popup with a transient toolbar status label |
| `1773589344533` | Prevent Find Bias / Run Fits re-entry crashes and warning dialogs |

## Stale state & load-path annoyance fixes

| Task | What |
|---|---|
| `1769292803515` | Fix `KeyError: 'amplitude'` when loading a saved multisweep file |
| `1769473301027` | Fix stale / overlaid plots when switching plot batches |
| `1773843753537` | Make Zoom Box Mode actually work on multisweep grid plots |
| `1773853963366` | Enable **Show Bias Info** when loading a file that has bias data |
| `1774290406612` | Fit plot tab missing until the settings checkbox was toggled |
| `1775008773695` | Stale Find Resonances lines persisting on a new network analysis |
| `1775070607944` | "Direct panel display not yet implemented" when loading multisweep files |
| `1775131019695` | Spurious "res_info_dict not found" on multisweep re-run |
| `1778026513621` | Fix crash loading saved network analysis files (float guard on `dac_scales_used`) |
| `1778773135853` | Bogus "invalid file" on multisweep load; button relabelled "Run / Load Multisweep" |
| `1779468813834` | Fix Run Fit on a saved multisweep; clearer bias-mode error |
| `1780516931200` | Fix empty histograms in the Histograms tab (key mismatch) |
| `1784716772709` | IQ derivative plots not redrawn after re-running Find Bias *(plan only)* |

---

## Borderline — UX-driven but touches structure or measurement

Kept for now; prune as needed.

| Task | What | Why borderline |
|---|---|---|
| `1769206496070` | Fix aggregate plot data extraction failures and colour contrast | Half plumbing fix, half colour polish |
| `1769262107411` | Replace multisweep display with tabbed aggregate plots plus batch nav | Big UX win, but a panel restructure |
| `1769272621658` | Delete unused standalone aggregate panel files | Cleanup rather than user-visible |
| `1769278149432` | Parameter histograms added as a fourth multisweep tab | New plot surface |
| `1773167364574` | Revert a session export change, fix a double signal connection | Export plumbing |
| `1773581045662` | New **Run Fit** button plus `FitSettingsPanel` | New fitting feature with a UI face |
| `1777926112296` | Self-describing amplitude fields in export pickles (`sweep_amplitude_normalized`, `sweep_power_dbm`) | Driven by a "very clunky" complaint, but it's an output-schema change |
| `1780517130322` | Sweep amplitude recorded in normalized units and dBm in netanal output metadata, later extended to multisweep | Touches `take_netanal.py`, `core/transferfunctions.py`, `multisweep.py`; nothing user-visible |
| `1775062658776` | Noise spectrum panel revamped into tabbed PSD grid | Architectural; also left incomplete |
| `1775077166288` | NCO bandwidth strict-inequality fix, plus NCO-change warnings in dialogs | Warnings are QoL; the fix is NCO logic |
| `1778703412069` | `BiasKIDsDialog` replaces `LoadBiasDialog` / `load_bias_payload` | Structural refactor of the bias load path |
| `1784644927190` | Custom Bias widget added to the Bias Settings panel | New biasing feature |

---

## Separate project — analog signal-chain builder GUI

These are QoL items but belong to `~/code/analog` (`analog_chain_interface`),
not rfmux/periscope. Listed for completeness.

| Task | What |
|---|---|
| `1768922199255` | Move Tools menu buttons into the main menu bar |
| `1768922740554` | Show generated diagrams / summaries in the GUI with a Save button |
| `1768930527627` | Reorder main-window columns into a logical grouping |
| `1769117810452` | Embed plots and diagrams in the main window; taller default height |
| `1769126466515` | Noise / gain plot axis units and scales (dBm/Hz, dBc/Hz) |
| `1769127407131` | Move digitizer config into its own leftmost column |

---

## Method & caveats

- Each task was identified from its `focus_chain_taskid_*.md` checklist plus the
  opening user prompt in `ui_messages.json`. 244 task directories in total.
- 15 sessions have no focus chain because they never left plan mode. All 15 were
  read-only Q&A or diagnosis with no code changes, so none contributes a QoL item:
  `1769277044606`, `1769284446567`, `1769709176335`, `1770136584944`,
  `1773866910086`, `1774986573842`, `1777925917856`, `1777932440802`,
  `1779462774231`, `1779464772317`, `1779465337503`, `1780482711318`,
  `1780586382241`, `1784722408028`, `1786972687130`. Two of them describe real
  QoL-shaped complaints that may still be unfixed: `1770136584944` (Normalize
  Traces checkbox is a no-op on multisweep IQ panels) and `1777925917856`
  (confusingly named Load vs Run Multisweep buttons).
- Tasks that left focus-chain items unchecked and may be incomplete:
  `1769449800023`, `1773687111497`, `1775062658776`, `1775141568278`.
- A sample of 14 entries was checked line-by-line against its source history;
  12 were accurate as written and the 3 corrections found are already applied above.
- Duplicate/repeated themes are visible on purpose: the aggregate-panel default
  count was changed three times (`1774290189352`, `1779455877857`,
  `1780516836504`), and square IQ axes twice (`1769267780723`, `1769273298898`).
