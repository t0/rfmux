"""Builders shared by the record tests: a bias export as Periscope's
session writes it, and a fastrx module faked well enough for
record_streams to reach its recording window."""

import pickle
import time
from pathlib import Path
from types import SimpleNamespace

from rfmux.core.session_folder import register_export


def bias_export(path, module, channels, calibrated=True, timestamp="",
                nco=1.0e9):
    """A Bias KIDs export of *channels* on *module* at *path*, listed
    in the folder's metadata as Periscope lists it."""
    out = {c: {"bias_channel": c,
               "df_calibration": (complex(1e6 * c, -1e5) if calibrated else None)}
           for c in channels}
    with open(path, "wb") as f:
        pickle.dump({"target_module": module, "timestamp": timestamp,
                     "bias_kids_output": out, "nco_frequency_hz": nco}, f)
    register_export(Path(path).parent, Path(path).name, "bias",
                    f"module{module}", timestamp or None)
    return Path(path)


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
