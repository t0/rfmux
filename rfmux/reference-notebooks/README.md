---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Reference notebooks

These ship with the `rfmux` package and are provisioned **read-only**, so save
your own copies elsewhere (*File → Save Notebook As…*) before editing.

They are jupytext markdown rather than `.ipynb`. In the Jupyter session
Periscope launches they open as notebooks on double-click; in a JupyterLab you
started yourself, right-click → *Open With* → *Notebook*.

## Where to start

The first three take the tuning flow a step at a time, in this order:

- **`Demos/network_analyses_find_resonances_make_resonator_catalog.md`** — sweep
  a band, find the dips, build the resonator catalog everything downstream
  passes around. Its second half is a tour of the catalog on its own — building
  one by hand, CSV and dictionary round trips, the invariants — and needs no
  hardware.
- **`Demos/multisweep.md`** — look at each resonance closely: one narrow sweep
  per resonator, all of them in parallel, and then the same array over a ladder
  of probe amplitudes.
- **`Demos/fitting_resonators.md`** — turn those sweeps into numbers. The three
  resonator models, where their results land in the results dictionary, and what
  the fitted parameters do as you drive a detector harder.

Then:

- **`Demos/simplified_tuning_flow.md`** — the whole chain end to end: sweep,
  find and fit the resonators, park the carriers, measure the noise.
- **`Demos/pulse_capture.md`** — detect and record detector pulses, with
  streaming HDF5, histograms and matched slow+fast capture.

`simplified_tuning_flow` and `pulse_capture` have an unattended `.py`
counterpart beside them for cron jobs and smoke tests; the notebook is the
documentation, the script is the runner.

## Connecting

```python
import rfmux
```

```python
s = rfmux.load_session('!HardwareMap [ !CRS { serial: "0033" } ]') # Replace with your board serial
crs = s.query(rfmux.CRS).one()
await crs.resolve()
await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)
```

No hardware? Every demo above stands up a simulated CRS instead — see their
mock-mode sections.
