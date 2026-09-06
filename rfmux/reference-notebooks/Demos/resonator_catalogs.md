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
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Resonator catalogs

A `rfmux.core.resonators.ResonatorCatalog` is rfmux's record of your array: which
detectors exist, what each one is called, which hardware channel it sits on, and
where it is biased. It is what each tuning step takes in and hands back, what you
save at the end of a session, and what you load next time to pick up where you
left off.

This notebook is about the catalog itself. It starts from a network analysis that
was measured earlier and saved to disk, so getting to a catalog is one load and
one call; everything after that is bookkeeping, and needs no board and no
hardware.

| Piece | Module |
|---|---|
| The catalog and its parts | `rfmux.core.resonators` |
| The search a catalog is seeded from | `rfmux.tuning.find_resonances` |
| Writing to disk and reading back | `rfmux.tuning.store` |

Taking the network analysis and finding the resonances in it is
`network_analysis_find_resonances.md`. That notebook ends where this one begins —
on a file of exactly the kind loaded in section 1 — so if the workflow is
unfamiliar, start there and come back here.


## How to use this document

**This is a runnable notebook, not a web page.** Every grey block below is a live
code cell: put the cursor in it and press **Shift+Enter** to execute it.

- **Run the cells in order, top to bottom.** Later cells use variables the
  earlier ones defined, so skipping ahead fails with a `NameError`. *Kernel →
  Restart Kernel and Run All Cells* starts clean.
- **The outputs you see are the ones you just produced.** This file is stored as
  jupytext markdown, which keeps no saved outputs, so a cell is blank until you
  run it. Nothing here can show you a stale number from someone else's run.
- **Editing is encouraged.** Change the names, the frequencies, the separation
  rule, and re-run — that is what this document is for. The shipped copy is
  read-only, so *File → Save Notebook As…* to keep your changes.
- **How you open it depends on your editor.** This file is jupytext markdown,
  not `.ipynb`. In the JupyterLab session Periscope launches it opens as a
  notebook on double-click; in a JupyterLab you started yourself, right-click →
  *Open With* → *Notebook*. **In VS Code it opens as plain text**, so pair it
  instead: with a jupytext extension installed, right-click → *Open Paired
  Notebook* (the exact wording varies by extension) creates an `.ipynb` beside
  this file and keeps the two in step — run and edit the notebook, and your
  changes flow back into the markdown. If that command does nothing, the
  extension could not find jupytext: it runs whichever interpreter VS Code
  resolved, which is often the base environment rather than the one rfmux is
  installed in. Install jupytext there, point the extension at the right
  interpreter, or skip the extension and run `jupytext --sync <this file>.md`
  from a shell that has it. The `.ipynb` is a local working copy and is
  gitignored; the markdown is the version that is kept, reviewed and tested.
- **Check which kernel you are running.** rfmux has to be importable from the
  interpreter the notebook uses, and if you have more than one checkout, it must
  be the environment installed against *this* one. Getting that wrong looks like
  a `ModuleNotFoundError` for a module you can plainly see on disk, because you
  are importing a different copy of rfmux than the one you are reading. This
  says which copy you actually got:

  ```python
  import sys, rfmux; print(sys.executable); print(rfmux.__file__)
  ```

<!-- #endregion -->

```python
import os
import tempfile
from pathlib import Path

import rfmux
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import BASE_FREQUENCY
from rfmux.tuning import ResonanceSearch, netanal_trace, store

# Files written by section 3. Reference notebooks are provisioned to a
# read-only directory, so writing next to the notebook would fail for anyone
# who opened it from Periscope. Override with RFMUX_DEMO_OUTPUT.
OUTPUT_DIR = Path(os.environ.get(
    "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_catalog_demo"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The recorded network analysis this notebook starts from, found through the
# package rather than by a relative path, so it works whichever directory the
# kernel started in. Matched by pattern rather than named outright, because
# `store` puts the date and time of the writing into every filename it makes —
# your own netanals land in `store.output_directory()` under names of exactly
# this shape.
DEMOS = Path(rfmux.__file__).parent / "reference-notebooks" / "Demos"
NETANAL_PKL = max(DEMOS.glob("netanal_*_demo_catalog1.pkl"))

print(f"starting from : {NETANAL_PKL.name}")
print(f"files →         {OUTPUT_DIR}")
```

## 1. Start from a saved network analysis

The file below is an ordinary measurement file, the kind `take_netanal` writes
for itself. It holds a sweep of a simulated ten-resonator array across
0.6–1.05 GHz, and — because `find_resonances_in_netanal` writes its result back
into the netanal and saves it again — the resonance search that was run on that
trace. One file, holding the trace, the search, and the record of how the sweep
was called, which between them is everything a catalog needs.

`store.load` is `pickle.load` plus one correction: the path the file recorded
about itself when it was written is replaced with where the file has actually
turned out to be. Demo data that shipped inside a package — as this did — can
then still save itself back to the file *you* opened rather than to a path on a
computer you may not even be on.

```python
netanal = store.load(NETANAL_PKL)

# A netanal is keyed by module, one entry per module swept, with the file's own
# metadata beside them. This one swept a single module.
module_id, = [key for key in netanal if key != store.METADATA_KEY]
module_netanal = netanal[module_id]

# `netanal_trace` is the walk down to the measured arrays —
# results[0]["upward"] — with an error worth reading if what you handed it was
# not a netanal.
trace = netanal_trace(module_netanal)

print(f"module id  : {module_id}")
print(f"called with: {module_netanal['call_params']}")
print(f"measured   : {list(trace)}")
```

The search comes back out with a `from_dict`, the same as for any class rfmux
stores in a file. Nothing re-runs: the finder's work was done once, when the
netanal was taken.

```python
search = ResonanceSearch.from_dict(trace["resonance_search"])
print(search)
print(f"settings used: {search.settings}")
```

### Seeding the catalog

`ResonanceSearch.to_catalog()` is where anonymous dips become tracked
resonators. Each gets a name (a string of the format of your choosing), a
hardware channel, and a `rfmux.core.resonators.BiasPoint` at its found frequency
— the operating point as first guessed. Multisweep and bias finding will refine
and update this BiasPoint as we progress through the tuning flow.

`module` and `amplitude` are both required. Neither is the search's to know: a
search is about one trace and does not carry the measurement around it. Both are
in the netanal, so read them off it rather than retyping numbers that have to
agree with a file. Channels are assigned 1..N in frequency order.

```python
MODULE = module_netanal["module"]
PROBE_AMPLITUDE = trace["sweep_amplitude"]

catalog = search.to_catalog(module=MODULE, amplitude=PROBE_AMPLITUDE)
print(f"module {MODULE}, probed at {PROBE_AMPLITUDE} normalized DAC units\n")
print(catalog)
```

<!-- #region -->
## 2. The catalog on its own

Three types, nested one inside the next:

    ResonatorCatalog    one per module; holds N Resonators
    └── Resonator       one per detector; holds exactly one BiasPoint
        └── BiasPoint   one tone: frequency, amplitude, and the calibration
                        measured at that tone

**The Catalog is the record of your array.** It is what each tuning step takes
in and hands back, what you save at the end of a session, and what you load next
time to pick up where you left off. Everything in it is small — a handful of
numbers per detector — because **sweep data is deliberately not kept here**.
Analysis reduces a sweep to the few scalar values that belong on a `BiasPoint` and the
traces stay separate. This keeps the size of this file small, and prevents mismatches
between saved fields and extraneous data. 

**The Resonator persists.** Its name is fixed the
moment the catalog is built and is meant never to change again: it will be
associated with various data products, allowing you to determine which measurement
pertains to which detector.

**A Resonator has a BiasPoint**, which can be reset as needed during tuning 
processes.
`rfmux.tuning.find_resonances` seeds its original values, and bias finding
routines refine it.
It snaps to the hardware tone grid, to avoid in-band IMD products.

**A given BiasPoint is frozen.** A tone
and the calibration measured at that tone are treated as a single indivisible
fact, so a `BiasPoint` cannot be edited — you have to replace it. The reason is that the
df calibration and the IQ rotation are only meaningful at the exact frequency
and amplitude they were measured at. If you retune the detector to a new bias 
frequency or amplitude, the bias point's calibration information will be wrong,
and a df timestream computed with them will therefore also be wrong. To avoid this
problem, we enforce that you can never modify a `BiasPoint`, only create a new one.


The catalog checks its members as they join. It verifies that each has:
- a unique name
- a unique channel number
- a unique frequency (two Resonators may not have exactly the same bias frequency)
    - TODO: is this check repeated over time? Eg if bias finding going awry, and one 
    Resonator's bias frequency migrates to its neighbours, is this flagged?
    - TODO also consider adding a bypass for this, to allow feedback testing shenanigans.

### Building a Catalog by hand

Catalogs are generated automatically by functions like the resonance finder, but can also be constructed
by hand, which may be instructive as to the layout of the object.
<!-- #endregion -->

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
            notes={"comment": "seems suss, operate with caution"},
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

Note: `df_calibration` is a property derived from `dI_df` and `dQ_df` rather than a
stored field, so it cannot go stale.

### Making a Catalog from a list of frequencies

The same constructor the finder used. Supplied names are paired with the
frequencies **positionally, before sorting**, so parallel lists stay associated
however they arrive:

```python
named_catalog = ResonatorCatalog.from_frequencies(
    [1.05e9, 1.01e9, 1.03e9],           # deliberately out of order
    names=["high", "low", "middle"],    # paired with the line above
    module=2,
    amplitude=0.01,
)
print(named_catalog)
```

### How resonators get their names

Names are optional. Left out, each resonator gets a short made-up but
pronounceable word — `BOTA`, `KOZR` — drawn fresh every time.

That a name says nothing about position is the point of it. A resonator is a
thing in its own right, not an index into a frequency-ordered list, and a name
like `R0007` claims otherwise: it asserts seventh-lowest, and that claim goes
quietly wrong the first time a resonator is removed or retuned. The ordering is
not lost — it lives on `channel`, and `names()` recomputes it live from the bias
frequencies, which is the version that stays true however much the array moves.

Instead of a list, `names` takes a **namer**: a function of the sorted
frequencies that returns one name each. Three come with rfmux, and you can write
your own.

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

# Derived from each frequency, so the same resonator always gets the same name.
print(f"from_frequency          : {syllabic_names_from_frequency(frequencies)}")
print(f"    ... and again       : {syllabic_names_from_frequency(frequencies)}")

# When a number really is what you want.
print(f"numbered_names          : {numbered_names(frequencies)}")
```

`syllabic_names_from_frequency` is the one to reach for when names have to be
stable across runs — a notebook whose text names a resonator out loud, or a
figure you intend to regenerate. Each name is a function of the resonator's own
frequency, so it survives re-measurement jitter and does not shift when the
array turns up one resonance more or fewer than last time:

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

Pass the namer itself, not a call to it. To vary a namer's own arguments, use
`functools.partial`:

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

Names are minted once, here, when anonymous frequencies become catalog members.
Everything downstream carries them: they key the catalog, they key every sweep
result the measurement algorithms return, and they are written into the files
`to_dict` and `to_csv` produce. Reading a catalog back with `from_dict` or
`from_csv` restores the names the file recorded rather than drawing new ones —
a saved catalog keeps its identities, which is what makes a name worth writing
down at all.

### Reading from a Catalog

Lookup is by name. The resonators themselves are a collection rather than a
sequence — they have no inherent order, and none is stored — so pulling them
out means saying which order you want them in.

`catalog.resonators()` and `catalog.names()` do that, sorted by bias frequency
lowest first, which is the array as you would plot or tabulate it. Iterating
the catalog (`for resonator in catalog`) gives you the same thing. Pass
`order="channel"` to either one for hardware channel order instead, which is
what you want when the members have to line up with per-channel data coming
back from the board.

The two orderings agree for a catalog fresh from `from_frequencies` or
`to_catalog`, since channels are assigned 1..N in frequency order. They drift
apart once resonators are retuned or removed.

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

<!-- #region -->
### Amending a bias point

`Resonator.set_bias()` builds a new `BiasPoint` rather than mutating one.


Changing `frequency_hz` or `amplitude` drops the calibration fields
unless you pass new values explicitly, so stale calibration stays structurally
impossible even through this convenience path. 

Changing only calibration leaves
the tone frequency and amplitude alone.
<!-- #endregion -->

```python
red_resonator = by_hand_catalog["red"]
print(f"before      : {red_resonator.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {red_resonator.bias.iq_rotation_deg}")

red_resonator.set_bias(frequency_hz=1.0505e9)        # moving the tone
print(f"tone moved  : {red_resonator.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {red_resonator.bias.iq_rotation_deg}   <- calibration dropped")

red_resonator.set_bias(iq_rotation_deg=15.0)         # calibration only
print(f"recalibrated: {red_resonator.bias.frequency_hz/1e6:.4f} MHz, "
      f"rotation {red_resonator.bias.iq_rotation_deg}   <- tone untouched")
```

### Removing a resonator from the catalog

This can be done with `catalog.remove('red')` or `del catalog['red']`.

This is akin to removing a key from a dictionary, and thus everything else about
the catalog is left untouched. In particular, channel numbers are not adjusted to fill
in the missing one.

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

### Quantizing onto the tone grid

To avoid seeing in-band intermodulation distortion products, the hardware is only
allowed to place a tone on a multiple of
`rfmux.core.transferfunctions.BASE_FREQUENCY`.

When adding a bias frequency to a `BiasPoint`, this quantization is applied automatically,
so there can never be disagreement about what frequency will actually be output to the array.

```python
print(f"tone grid: {BASE_FREQUENCY:.6f} Hz")

# Four tenths of a step above a real grid point, so the shift is unambiguous.
# (Asking for a round 1.0 GHz would not demonstrate anything: it happens to sit
# very nearly on the grid already.)
grid_frequency_hz = round(1.0e9 / BASE_FREQUENCY) * BASE_FREQUENCY
requested_frequency_hz = grid_frequency_hz + 0.4 * BASE_FREQUENCY
quantized_bias = BiasPoint(requested_frequency_hz, amplitude=0.01,
                           iq_rotation_deg=12.0)

moved_by_hz = quantized_bias.frequency_hz - requested_frequency_hz

print(f"requested: {requested_frequency_hz:.6f} Hz")
print(f"recorded : {quantized_bias.frequency_hz:.6f} Hz")
print(f"moved by : {moved_by_hz:+.6f} Hz "
      f"({moved_by_hz/BASE_FREQUENCY:+.1f} steps)")
print(f"rotation : {quantized_bias.iq_rotation_deg} deg  <- calibration kept")
```

The shift is under half a step — far smaller than a resonator's width — so
calibration measured at the requested frequency still holds at the tone that
actually gets played. That is why quantization is not a tone move in the sense
`set_bias` cares about, and does not drop the calibration fields.

If you want the exact number you asked for — a sweep centre you are doing
arithmetic on, say — pass `bias_frequency_quantized=False`, and nothing
downstream will round it for you. `.quantize()` is the one-shot for those.

```python
unquantized_bias = BiasPoint(requested_frequency_hz, amplitude=0.01,
                             bias_frequency_quantized=False)
print(f"kept exact : {unquantized_bias.frequency_hz:.6f} Hz")
print(f".quantize(): {unquantized_bias.quantize().frequency_hz:.6f} Hz")
```

### Invariants, and what they refuse

Names and channels must be unique, and channels are 1-based. These are checked
when a resonator joins the catalog.

Bias frequencies are not policed unless you ask: `min_separation_hz` defaults to
`None`, which lets any spacing through, including two tones at exactly the same
frequency. Pass `min_separation_hz=0.0` to refuse an exact duplicate — a
resonance that `find_resonances` split in two lands there once the frequencies
are quantized — or a wider number when you know the separation below which your
readout cannot operate two detectors. Every constructor takes it, including
`from_dict` and `from_csv`, and none of them takes it from anywhere else: a rule
applies to the catalog in front of you because you asked for it here, not
because the file you loaded was written under one.

```python
def refused(what, thunk):
    try:
        thunk()
    except (ValueError, KeyError) as e:
        print(f"{what}:\n    {e}\n")
    else:
        print(f"{what}: accepted?!\n")

refused("duplicate channel", lambda: ResonatorCatalog(
    [Resonator("a", 1, BiasPoint(1.01e9, 0.01)),
     Resonator("b", 1, BiasPoint(1.02e9, 0.01))], module=2))

refused("identical frequencies, with a separation rule asked for",
        lambda: ResonatorCatalog(
            [Resonator("a", 1, BiasPoint(1.01e9, 0.01)),
             Resonator("b", 2, BiasPoint(1.01e9, 0.01))],
            module=2, min_separation_hz=0.0))

refused("amplitude in dBm, not DAC units",
        lambda: BiasPoint(1.01e9, amplitude=-30.0))

refused("no such resonator", lambda: named_catalog["nope"])
```

## 3. Saving and loading Catalogs

To improve usability and compatibility, rfmux provides helpers to translate
Catalog objects into other standard classes and file formats.

### Dictionaries

`a_dictionary = catalog.to_dict()` reduces a catalog to plain builtins — dicts, strings and floats —
and `new_catalog_from_dict = ResonatorCatalog.from_dict(a_dictionary)` rebuilds it. Being plain builtins, the result will go into a pickle,
a JSON file or HDF5 attributes equally happily.

The `resonators` entry is itself keyed by name, like the catalog, so reading one
detector out of a saved file is `a_dictionary['resonators']['BOTA']` and not a
search. `from_dict` takes the same keyword arguments as the other constructors,
so you can load under whatever separation rule you want to hold this catalog to
— say `ResonatorCatalog.from_dict(a_dictionary, min_separation_hz=100e3)`. The
`min_separation_hz` in the file is a record of how the catalog was built and is
not applied on the way back in, so a catalog comes back unpoliced unless you ask
here.

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

<!-- #region -->


### Pickle 

Pickle is currently the main file format used for array tuning outputs. This may evolve in future.


**Pickle the dictionary, not the catalog object.** `pickle.dump(catalog, f)`
does work, BUT it is a trap. The file would then record the class's import path, so
moving or renaming `ResonatorCatalog` later makes every old file unreadable.

Instead, go
through `to_dict` / `from_dict` !

`rfmux.tuning.store` does the file handling: it picks the folder, names the
file, and stamps a `file_metadata` block into what it writes saying what the
file is and where it lives. It is the same `store.load` that opened the netanal
in section 1.

Generally, file saving is done automatically by the measurement algorithms, so you
probably won't need to worry about this part.
<!-- #endregion -->

```python
catalog_pkl_path = store.save(by_hand_catalog.to_dict(), "catalog",
                              label="by_hand", directory=OUTPUT_DIR)
print(f"wrote {catalog_pkl_path.name} "
      f"({catalog_pkl_path.stat().st_size} bytes)")

catalog_from_disk = ResonatorCatalog.from_dict(store.load(catalog_pkl_path))
print(catalog_from_disk)
```

`directory=` is here only because this notebook keeps its files together in
`OUTPUT_DIR`. Leave it off and the file goes where every measurement goes:
`~/rfmux_data/ipy_session_<today>/`, the folder `take_netanal` and `multisweep`
write into when they finish, since they call this same `store.save` for you.
`store.output_directory()` says where that is, and `rfmux.tuning.store`'s
docstring covers moving it — for one session, or for good.

Two things came back that you did not put in. The filename gained a date and a
time, so two catalogs saved an hour apart do not collide; and the dictionary
gained a `file_metadata` key recording what the file is and where it lives.
`ResonatorCatalog.from_dict` ignores the extra key, so a saved catalog and a
hand-built one load exactly the same way.

The usual caution applies: unpickling runs code from the file, so load `.pkl`
files you produced or trust, not ones that arrived from somewhere unknown.

### CSV 

**Note that going back and forth from CSV files is deliberately lossy.**
These files will carry the operating point and nothing else:
`notes` and every calibration field are dropped. Use it to
hand someone a bias table they can edit; use `to_dict` when you need everything
back.

```python
bias_table_csv_path = OUTPUT_DIR / "bias_table.csv"
bias_table_csv_path.write_text(by_hand_catalog.to_csv())
print(bias_table_csv_path.read_text())
```

Reading takes the CSV *text* and the module — the module is not in the file,
because channel numbers are meaningless without knowing which module they belong
to. Columns are matched by header name, so they may appear in any order.

```python
reloaded_catalog = ResonatorCatalog.from_csv(
    bias_table_csv_path.read_text(), module=2
)
print(reloaded_catalog)
print(f"\ncalibration after a CSV round trip: "
      f"{reloaded_catalog['red'].bias.iq_rotation_deg}   "
      f"<- dropped, as documented")
```

Hand-editing works. Both columns of an operating point are
required so a blank cell will throw an error naming the
line:

```python
edited_csv_text = "\n".join([
    "name,channel,bias_frequency_hz,bias_amplitude",
    "blue,1,1010000000.0,0.01",
    "green,2,1030000000.0,0.02",      # amplitude changed by hand
    "red,3,1050000000.0,0.005",
])
print(ResonatorCatalog.from_csv(edited_csv_text, module=2))

refused("a missing bias amplitude", lambda: ResonatorCatalog.from_csv(
    "name,channel,bias_frequency_hz,bias_amplitude\nblue,1,1010000000.0,\n",
    module=2))
```

Finally, the catalog section 1 seeded from the saved network analysis, saved both
ways:

```python
(OUTPUT_DIR / "found.csv").write_text(catalog.to_csv())
store.save(catalog.to_dict(), "catalog", label="found", directory=OUTPUT_DIR)
print(f"wrote {len(catalog)} resonators to {OUTPUT_DIR}")
for output_path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {output_path.name:<16} {output_path.stat().st_size:>7} bytes")
```

That catalog is what the rest of tuning consumes and returns. From here you would
run a multisweep around each bias frequency, fit, and pick bias points — see
`multisweep.md` for the next step, and `simplified_tuning_flow.py` in this folder
for the whole chain.
