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
  resonators, bias the detectors, measure the noise. Everything else assumes
  you have done this first.
- **`Demos/pulse_capture.md`**: detect and record detector pulses, with
  streaming HDF5, histograms and matched slow+fast capture.

Each has a `.py` counterpart beside it showing the same sequence as a plain
script. This is intended as a reference for writing your own code against the API.
It is also a quick way to smoke-test a change against `MOCK`, by hand: the
notebooks are executed by the test suite, the scripts are not.

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

Both demos can also run in mock mode with simulated hardware.
