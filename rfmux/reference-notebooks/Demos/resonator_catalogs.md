---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: rfmux-tuning
    language: python
    name: python3
---

# Resonator catalogs

A `rfmux.core.resonators.ResonatorCatalog` records the detectors in one module:
their names, hardware channels, and bias points. Save it to keep your tuning
settings, then load it when you’re ready to continue working with this array.

This notebook starts from a saved network analysis and resonance search.
Everything here runs without hardware.

| Task | Module |
|---|---|
| Manage the catalog and bias points | `rfmux.core.resonators` |
| Find resonances and build a catalog | `rfmux.tuning.find_resonances` |
| Save and load files | `rfmux.tuning.store` |

See `network_analysis_find_resonances.md` to take a network analysis and find
resonances. Here, we’ll pick up from its saved results.

## How to use this document

This is a runnable Jupytext notebook. Select a code cell and press **Shift+Enter**.

- Run cells from top to bottom. Later cells use variables defined earlier.
  Use *Kernel → Restart Kernel and Run All Cells* to start again.
- The markdown file stores no outputs. Run a cell to see its results.
- Feel free to change the names, frequencies, and separation settings as you go.
  The shipped copy is read-only; use *File → Save Notebook As…* to keep your changes.
- In Periscope’s JupyterLab, double-click this file. In another JupyterLab
  session, use *Open With → Notebook*.
- VS Code opens this file as text. With a Jupytext extension, use *Open Paired
  Notebook* (the command name may vary). If pairing fails, check that the
  extension’s Python environment has Jupytext installed. You can also run
  `jupytext --sync <this file>.md` in an environment with Jupytext. The paired
  `.ipynb` is a local, gitignored copy; the markdown is kept in version control.

The kernel must use the environment where this checkout of rfmux is installed.
Check the interpreter and package paths:

```python
import sys
import rfmux

print(sys.executable)
print(rfmux.__file__)
```

```python
from pathlib import Path

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import BASE_FREQUENCY
from rfmux.tuning import ResonanceSearch, store
```

## 1. Start from a saved network analysis

The demo file contains a network analysis of ten simulated resonators across
0.6–1.05 GHz. It also includes the saved resonance search and measurement settings.
`find_resonances_in_netanal` stored the search in the trace when it was run.

`store.load()` loads the pickle and updates its file metadata to the current
path. This lets the file be moved between machines without keeping its old path.

```python
# Find the newest matching demo file inside the installed package.
# The timestamp in its name sorts in date order.
demos = Path(rfmux.__file__).parent / "reference-notebooks" / "Demos"
netanal_path = max(demos.glob("netanal_*_demo_catalog1.pkl"))
print(f"starting from: {netanal_path.name}")
netanal = store.load(netanal_path)

# A netanal is keyed by module, one entry per module swept, with the file's own
# metadata beside them. This one swept a single module.
module_id, = [key for key in netanal if key != store.METADATA_KEY]
module_netanal = netanal[module_id]

# A network analysis measures the band once, so results is the trace itself —
# no amplitude step, no direction key and no resonator-name layer above it.
trace = module_netanal["results"]

print(f"module id  : {module_id}")
print(f"called with: {module_netanal['call_params']}")
print(f"measured   : {list(trace)}")
```

Rebuild the search object from the saved dictionary. This reads the existing
result; it does not run the search again.

```python
search = ResonanceSearch.from_dict(trace["resonance_search"])
print(search)
print(f"settings used: {search.settings}")
```

### Build the catalog

`ResonanceSearch.to_catalog()` gives each detected resonance a name, a hardware
channel, and a `BiasPoint` at its found frequency. Later sweeps and bias finding will
help refine these initial operating points.

Supply `module` and `amplitude` from the measurement. The search itself does
not store them. Channels are assigned 1..N in frequency order.

```python
# Read the module and probe amplitude directly from the saved measurement.
catalog = search.to_catalog(
    module=module_netanal["module"],
    amplitude=trace["sweep_amplitude"],
)
print(f"module {catalog.module}, probed at {trace['sweep_amplitude']} normalized DAC units\n")
print(catalog)
```

## 2. Explore the catalog

The catalog has three nested types:

    ResonatorCatalog    all the things on an array; holds Resonator objects
    └── Resonator       one detector; has a name, channel, and BiasPoint
        └── BiasPoint   tone frequency, amplitude, and calibration

The catalog stores a few values per detector. Sweep arrays stay in the measurement
results, keeping the catalog small and easy to save.

A resonator’s name identifies it across tuning steps and measurement files.
Keep the name when you retune it; replace its bias point as needed.

A `BiasPoint` is immutable. Its calibration belongs to a specific frequency and
amplitude, so changing the tone requires a new bias point. `update_bias_point()` handles
this replacement and clears old calibration when you supply a frequency or amplitude.
Bias frequencies snap to the hardware tone grid by default.

The catalog checks for unique names and channel numbers when members are added.
Frequency separation is optional; see the validation examples below.

### Build a catalog by hand

Let’s build three resonators to see where each field belongs. The third also
has calibration values and a note.

```python
by_hand_catalog = ResonatorCatalog(
    [
        Resonator(name="blue", channel=1, bias=BiasPoint(1.010e9, amplitude=0.01)),
        Resonator(name="green", channel=2, bias=BiasPoint(1.030e9, amplitude=0.01)),
        Resonator(
            name="red", channel=3,
            bias=BiasPoint(1.050e9, amplitude=0.005,
                           dI_df=1.2e-6, dQ_df=-3.4e-6,
                           iq_rotation_deg=12.0),
            notes={"comment": "check this resonator on the next sweep"},
        ),
    ],
    module=2,
)
print(by_hand_catalog)
print(f"\nred's df calibration: "
      f"{by_hand_catalog['red'].bias.df_calibration:.4g} Hz/V")
print(f"red at a -30 dBm DAC scale: "
      f"{by_hand_catalog['red'].bias.power_dbm(-30):.2f} dBm")
```

`df_calibration` is calculated from `dI_df` and `dQ_df`; it is not stored separately.

### Build a catalog from a list of frequencies

If you have a list of frequencies, you can use `from_frequencies()` to make a catalog. This is what the the resonance finder does to auto-generate a catalog based on the list of resonant frequencies it found. You can optionally provide a list of names, in the same order as the frequencies:

```python
named_catalog = ResonatorCatalog.from_frequencies(
    [1.05e9, 1.01e9, 1.03e9],           # deliberately out of order
    names=["high", "low", "middle"],    # paired with the line above
    module=2,
    amplitude=0.01,
)
print(named_catalog)
```

### Choose resonator names

If you omit `names`, each resonator gets a short generated name such as `BOTA`
or `KOZR`. New calls generate new names.

Names identify detectors; they do not track frequency rank. Use `catalog.names()` to
get a list of the names in the catalog, which by default come back in bias frequency order. You can check each resonator's `channel` for its hardware channel assignment. Numbered
names are also available, but their numbers stay unchanged after retuning or removal, so this is generally not recommended. Plus, using numbered names substantially reduces the opportunities for whimsy.

You can pass a naming function instead of a list. It receives sorted frequencies
and returns one name per frequency. Here are the three built-in options:

```python
from rfmux.resonator_names import (
    numbered_names,
    syllabic_names,
    syllabic_names_from_frequency,
)

frequencies = [1.01e9, 1.03e9, 1.05e9]

# The default. Drawn, so a second call gives different names.
print(f"syllabic_names          : {syllabic_names(frequencies)}")
print(f"    ... and again       : {syllabic_names(frequencies)}")

# Derived from frequency buckets, so the same inputs give the same names.
print(f"from_frequency          : {syllabic_names_from_frequency(frequencies)}")
print(f"    ... and again       : {syllabic_names_from_frequency(frequencies)}")

# When a number really is what you want.
print(f"numbered_names          : {numbered_names(frequencies)}")
```

`syllabic_names_from_frequency` derives names from frequency buckets. This is
useful for repeatable figures and examples. Small frequency changes usually keep
the same name, but crossing a bucket boundary can change it.

Here, adding a resonance and shifting the others by 300 Hz preserves the
original three names:

```python
stable = ResonatorCatalog.from_frequencies(
    frequencies, module=2, amplitude=0.01, names=syllabic_names_from_frequency
)
# One extra resonance, and a few hundred Hz of jitter on the others.
remeasured = ResonatorCatalog.from_frequencies(
    [f + 300 for f in frequencies] + [1.04e9],
    module=2,
    amplitude=0.01,
    names=syllabic_names_from_frequency,
)
print(f"first pass  : {stable.names()}")
print(f"re-measured : {remeasured.names()}  ← the original three kept theirs")
```

Pass the naming function as `names`. Use `functools.partial` to set its options:

```python
from functools import partial

numbered = ResonatorCatalog.from_frequencies(
    frequencies, module=2, amplitude=0.01, names=numbered_names
)
prefixed = ResonatorCatalog.from_frequencies(
    frequencies,
    module=2,
    amplitude=0.01,
    names=partial(numbered_names, prefix="kid"),
)
longer = ResonatorCatalog.from_frequencies(
    frequencies,
    module=2,
    amplitude=0.01,
    names=partial(syllabic_names, length=7),
)
print(f"numbered : {numbered.names()}")
print(f"prefixed : {prefixed.names()}")
print(f"longer   : {longer.names()}")
```

Names are assigned when the catalog is built. They key the catalog and sweep
sections, and are included in dictionary and CSV exports. Loading a saved catalog
restores its names.

### Read catalog entries

Look up a resonator by name with `catalog[name]`, or by hardware channel with
`catalog.by_channel(channel)`.

`catalog.names()`, `catalog.resonators()`, and iteration over the catalog use
current bias frequency order, lowest first. Pass `order="channel"` to either
method for channel order, for example when matching data from the board.

The orders initially agree for catalogs built from frequencies. Retuning can
change frequency order; removing a resonator leaves gaps in the channel numbers.

```python
print(f"len           : {len(named_catalog)}")
print(f"by name       : "
      f"{named_catalog['middle'].bias.frequency_hz/1e6:.3f} MHz")
print(f"by channel    : {named_catalog.by_channel(1).name}")
print(f"'low' present : {'low' in named_catalog}")
print(f"iteration     : {[r.name for r in named_catalog]}")
print(f"names()       : {named_catalog.names()}  (frequency order)")
print(f"names(channel): {named_catalog.names(order='channel')}")
print(f"resonators()  : {[r.channel for r in named_catalog.resonators()]}  "
      f"(their channels, in frequency order)")
print(f"module        : {named_catalog.module}")
```

### Update a bias point

`Resonator.update_bias_point()` creates and assigns a new `BiasPoint`.
Supplying `frequency_hz` or `amplitude` clears the calibration fields unless you
also supply new calibration values. Updating calibration alone keeps the tone unchanged.

```python
red_resonator = by_hand_catalog["red"]
print(f"before      : {red_resonator.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {red_resonator.bias.iq_rotation_deg}")

red_resonator.update_bias_point(frequency_hz=1.0505e9)        # moving the tone
print(f"tone moved  : {red_resonator.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {red_resonator.bias.iq_rotation_deg}   <- calibration dropped")

red_resonator.update_bias_point(iq_rotation_deg=15.0)         # calibration only
print(f"recalibrated: {red_resonator.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {red_resonator.bias.iq_rotation_deg}   <- tone untouched")
```

### Remove a resonator

Use `catalog.remove(name)` to remove and return a resonator, or `del catalog[name]`
to remove it. Other entries keep their names, channels, and bias points.

```python
pruned_catalog = by_hand_catalog.copy()
print(f"before : {pruned_catalog.names()} on channels "
      f"{[r.channel for r in pruned_catalog]}")

dropped_resonator = pruned_catalog.remove("green")
print(f"dropped: {dropped_resonator.name} from channel {dropped_resonator.channel}")

print(f"after  : {pruned_catalog.names()} on channels "
      f"{[r.channel for r in pruned_catalog]}   <- channel 2 is a hole, not reused")

del pruned_catalog["blue"]
print(f"del    : {pruned_catalog.names()}")
```

### Snap a frequency to the tone grid

The hardware tone grid uses multiples of
`rfmux.core.transferfunctions.BASE_FREQUENCY` to avoid in-band intermodulation
products. A `BiasPoint` rounds its frequency to this grid by default.

Let’s request a frequency between grid points and inspect the result:

```python
print(f"tone grid: {BASE_FREQUENCY:.6f} Hz")

# Start on the grid, then add 0.4 steps to make the rounding visible.
grid_frequency_hz = round(1.0e9 / BASE_FREQUENCY) * BASE_FREQUENCY
requested_frequency_hz = grid_frequency_hz + 0.4 * BASE_FREQUENCY
quantized_bias = BiasPoint(requested_frequency_hz, amplitude=0.01,
                           iq_rotation_deg=12.0)

moved_by_hz = quantized_bias.frequency_hz - requested_frequency_hz

print(f"requested: {requested_frequency_hz:.6f} Hz")
print(f"recorded : {quantized_bias.frequency_hz:.6f} Hz")

```


For array operations that need the exact requested frequency, pass
`bias_frequency_quantized=False`. **At your own peril!**

```python
unquantized_bias = BiasPoint(requested_frequency_hz, amplitude=0.01,
                             bias_frequency_quantized=False)
print(f"kept exact : {unquantized_bias.frequency_hz:.6f} Hz")
print(f".quantize(): {unquantized_bias.quantize().frequency_hz:.6f} Hz")
```

### Check invalid inputs

Names and channels must be unique, and channel numbers start at 1.
The catalog checks these when members are added.

Frequency checks depend on `min_separation_hz`:

- `None` (default): allow any spacing, including identical frequencies.
- `0.0`: reject identical frequencies, including those rounded to the same grid point.
- A positive value: reject frequencies that are this close or closer.

All catalog constructors accept this setting, including `from_dict` and `from_csv`.
Supply it when loading if you want a separation check; the saved setting is only
a record. Calling a member’s `update_bias_point()` does not rerun the catalog’s separation check.

These examples catch and print the expected errors so you can keep running the notebook:

```python
# Two resonators cannot share a hardware channel.
try:
    ResonatorCatalog(
        [Resonator("a", 1, BiasPoint(1.01e9, 0.01)),
         Resonator("b", 1, BiasPoint(1.02e9, 0.01))],
        module=2,
    )
except ValueError as e:
    print(f"duplicate channel: {e}\n")

# Identical frequencies are rejected when we request this separation rule.
try:
    ResonatorCatalog(
        [Resonator("a", 1, BiasPoint(1.01e9, 0.01)),
         Resonator("b", 2, BiasPoint(1.01e9, 0.01))],
        module=2,
        min_separation_hz=0.0,
    )
except ValueError as e:
    print(f"identical frequencies: {e}\n")

# Amplitude uses normalized DAC units, not dBm.
try:
    BiasPoint(1.01e9, amplitude=-30.0)
except ValueError as e:
    print(f"invalid amplitude: {e}\n")

# Looking up an unknown name raises KeyError, as in a dictionary.
try:
    named_catalog["nope"]
except KeyError as e:
    print(f"unknown resonator: {e}")
```

## 3. Save and load catalogs

Use a dictionary or pickle to keep the full catalog. CSV is useful for an editable
bias table, but omits calibration and notes.

### Dictionaries

`catalog.to_dict()` returns ordinary dictionaries, lists, and scalar values.
`ResonatorCatalog.from_dict()` rebuilds the catalog from them. This representation
can be serialized with pickle or JSON, or stored as suitable HDF5 attributes.

The `resonators` dictionary is keyed by name. For example,
`catalog_dict["resonators"]["red"]` contains that resonator’s fields.

To check separation while loading, pass a rule such as
`ResonatorCatalog.from_dict(catalog_dict, min_separation_hz=100e3)`.
The saved `min_separation_hz` value is not automatically applied.

```python
catalog_dict = by_hand_catalog.to_dict()
print(f"schema_version : {catalog_dict['schema_version']}")
print(f"top-level keys : {list(catalog_dict)}")
print(f"resonator names: {list(catalog_dict['resonators'])}")
print(f"one resonator  : {catalog_dict['resonators']['red']}")

restored_catalog = ResonatorCatalog.from_dict(catalog_dict)
print(f"\nround trip: {len(restored_catalog)} resonators, "
      f"names {restored_catalog.names()}")
print(f"calibration survived: "
      f"{restored_catalog['red'].bias.iq_rotation_deg} deg, "
      f"notes {restored_catalog['red'].notes}")
```

### Pickle

Pickle is the current format for tuning outputs. Save `catalog.to_dict()` so the
file does not depend on the catalog class’s import path. Rebuild it with
`from_dict()` after loading.

`store.save()` chooses a filename and adds a `file_metadata` block describing the
file and its path. Measurement routines use it to save their outputs automatically.
Here, we’ll use it to save a catalog ourselves.

Choose a writable output folder below. The default is a temporary demo folder;
set `RFMUX_DEMO_OUTPUT` to use another location.

```python
import os
import tempfile

# Keep demo files outside the read-only notebook directory.
output_dir = Path(os.environ.get(
    "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_catalog_demo"
))
output_dir.mkdir(parents=True, exist_ok=True)
print(f"saving files to: {output_dir}")

catalog_pkl_path = store.save(by_hand_catalog.to_dict(), "catalog",
                              label="by_hand", directory=output_dir)
print(f"wrote {catalog_pkl_path.name} "
      f"({catalog_pkl_path.stat().st_size} bytes)")

catalog_from_disk = ResonatorCatalog.from_dict(store.load(catalog_pkl_path))
print(catalog_from_disk)
```

Omit `directory=` to use the normal measurement folder,
`~/rfmux_data/ipy_session_<today>/`. `store.output_directory()` reports that path;
see `rfmux.tuning.store` for ways to change it.

The filename includes a timestamp. The saved dictionary also gains `file_metadata`,
which `ResonatorCatalog.from_dict()` ignores when rebuilding the catalog.

Only load pickle files you trust: unpickling can execute code from the file.

### CSV

CSV keeps the names, channels, and operating points. It drops notes and all
calibration fields. Use it to share or edit a bias table; use a dictionary or
pickle when you need the full catalog.

```python
bias_table_csv_path = output_dir / "bias_table.csv"
bias_table_csv_path.write_text(by_hand_catalog.to_csv())
print(bias_table_csv_path.read_text())
```

`from_csv()` takes CSV text and a module number. The module is not stored in the
CSV. Columns are matched by header name and may appear in any order.

```python
reloaded_catalog = ResonatorCatalog.from_csv(
    bias_table_csv_path.read_text(), module=2
)
print(reloaded_catalog)
print(f"\ncalibration after a CSV round trip: "
      f"{reloaded_catalog['red'].bias.iq_rotation_deg}   "
      f"<- dropped, as documented")
```

Try changing an amplitude in the text below. Both bias frequency and amplitude
are required; a blank value produces an error that identifies the line.

```python
edited_csv_text = "\n".join([
    "name,channel,bias_frequency_hz,bias_amplitude",
    "blue,1,1010000000.0,0.01",
    "green,2,1030000000.0,0.02",      # amplitude changed by hand
    "red,3,1050000000.0,0.005",
])
print(ResonatorCatalog.from_csv(edited_csv_text, module=2))

# Leave the amplitude blank to see the validation error.
try:
    ResonatorCatalog.from_csv(
        "name,channel,bias_frequency_hz,bias_amplitude\nblue,1,1010000000.0,\n",
        module=2,
    )
except ValueError as e:
    print(f"missing bias amplitude: {e}")
```

Finally, save the catalog built from the network analysis in both formats:

```python
(output_dir / "found.csv").write_text(catalog.to_csv())
store.save(catalog.to_dict(), "catalog", label="found", directory=output_dir)
print(f"wrote {len(catalog)} resonators to {output_dir}")
for output_path in sorted(output_dir.iterdir()):
    print(f"  {output_path.name:<16} {output_path.stat().st_size:>7} bytes")
```

You now have a catalog ready for tuning. Next, sweep around its bias frequencies,
fit the resonances, and choose bias points. See `multisweep.md` for the next step
and `simplified_tuning_flow.py` for the full workflow.

