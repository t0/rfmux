"""Builders shared by the record tests: a biased multisweep as
Periscope's session writes it, and a fastrx module faked well enough for
record_streams to reach its recording window."""

import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.session_folder import register_export
from rfmux.tuning import multisweep_from_tuning, tuning_rows


def bias_export(path, module, channels, calibrated=True, timestamp=""):
    """A multisweep of *channels* on *module* at *path*, its catalog
    biased, listed in the folder's metadata as Periscope lists it.

    Where a module's bias points live: Find Bias writes its report back
    into the sweeps it read, so this is what ``record`` reads to learn
    which channels are tuned and how they convert to hertz.
    """
    f = np.linspace(0.999e9, 1.001e9, 9)
    catalog = ResonatorCatalog(
        [Resonator(name=f"R{c:04d}", channel=c,
                   bias=BiasPoint(
                       frequency_hz=1.0e9 + 1e6 * c,
                       amplitude=0.01,
                       # 1/(dI+jdQ) is the calibration each channel reports.
                       **(_derivatives(c) if calibrated else {}),
                       bias_sweep={"frequencies": f,
                                   "iq_volts": np.exp(1j * f / 1e9),
                                   "original_center_frequency": 1.0e9 + 1e6 * c,
                                   "sweep_amplitude": 0.01,
                                   "sweep_direction": "upward"}))
         for c in channels],
        module=module)
    container = multisweep_from_tuning(
        tuning_rows(catalog), module, module_id=f"crs0000_rmod{module}")
    with open(path, "wb") as fh:
        pickle.dump(container, fh)
    register_export(Path(path).parent, Path(path).name, "multisweep",
                    f"module{module}", timestamp or None)
    return Path(path)


def _derivatives(channel):
    """dI_df and dQ_df giving a df_calibration of 1e6*channel - 1e5j."""
    d = 1.0 / complex(1e6 * channel, -1e5)
    return {"dI_df": d.real, "dQ_df": d.imag}


def fake_fastrx(monkeypatch, tmp_path, *, modules_seen=0b1111, packets=0):
    """``rfmux.fastrx`` replaced for record_streams: a socket file, a
    probe that sees the modules of *modules_seen*, and a writer that
    counts *packets* by the time it is stopped."""
    import rfmux

    socket = tmp_path / "enp2s0f0np0"
    socket.touch()
    writers = []

    class Capture:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def capture(self, n, channels, module, timeout):
            return {"modules_seen": modules_seen}

    class Writer:
        def __init__(self, path, *, channels, socket):
            self.path, self.channels, self.socket = Path(path), channels, socket
            self.packets = self.overruns = self.dropouts = 0
            self.path.touch()
            writers.append(self)

        def wait(self, timeout):
            time.sleep(timeout)

        def stop(self):
            self.packets = packets

    fx = SimpleNamespace(
        socket=str(socket), writers=writers,
        resolve_socket=lambda interface, sock: sock or str(socket),
        start_command=lambda name: f"sudo fastrxd -i {name}",
        record_stride=lambda channels: (86 + 4 * channels + 7) & ~7,
        MAX_SAMPLES=128,
        PacketCapture=Capture, PacketWriter=Writer)
    monkeypatch.setattr(rfmux, "fastrx", fx, raising=False)
    return fx
