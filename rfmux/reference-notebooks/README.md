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

## Opening them

They are jupytext markdown rather than `.ipynb`. In the Jupyter session
Periscope launches they open as notebooks on double-click. In a JupyterLab you
started yourself, right-click → *Open With* → *Notebook*, or set
*Settings → Document Manager → Default Viewers* to `markdown: Jupytext
Notebook`. To convert one instead:

```bash
jupytext -o pulse_capture.ipynb pulse_capture.md
```

## Where to start

- **`Demos/simplified_tuning_flow.md`**: sweep the band, find and fit the
  resonators, park the carriers, measure the noise. Everything else assumes
  you have done this first.
- **`Demos/pulse_capture.md`**: detect and record detector pulses, with
  streaming HDF5, histograms and matched slow+fast capture.

Each has an unattended `.py` counterpart beside it for cron jobs and smoke
tests: the notebook is the documentation, the script is the runner.

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

With no hardware, both demos stand up a simulated CRS instead. See their
mock-mode sections.
