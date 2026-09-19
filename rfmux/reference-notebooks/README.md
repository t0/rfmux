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
  streaming HDF5, histograms and matched slow+fast capture, then a walk
  through the capture file: events, noise samples, units and the tuning rows.
- **`Demos/fastrx_recording.md`**: open a fastrx recording (the 100G channel
  stream on disk) offline: its times, samples, gaps and drop-outs, then the
  recording beside a pulse capture and merged into it. It needs no board and
  no fastrx extension, and writes a small recording when given none.

The first two have a `.py` counterpart beside them: the same sequence as a
plain script, to copy from. `pulse_capture_flow.py` covers the captures; the
walk through the file is in the notebook only. The notebooks and
`simplified_tuning_flow.py` run in the acquisition tier; `pulse_capture_flow.py` does not, so run it by hand after
changing its notebook.

- **`Guides/`**: the repository's guides and installation page, provisioned
  beside the notebooks.
- **`Release Notes/`**: the firmware release walkthroughs and the
  repository's release notes.

## Connecting

```python
import rfmux
```

```python
s = rfmux.load_session('!HardwareMap [ !CRS { serial: "0033" } ]') # Replace with your board serial
crs = s.query(rfmux.CRS).one()
await crs.resolve()
await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)  # the fast stream needs a timestamp source
```
