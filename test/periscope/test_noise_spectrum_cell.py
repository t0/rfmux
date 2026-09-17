"""The main panel's channel noise spectrum runs as one session cell against
a streaming mock, and the panel gets the data back. A child process, as
for the other mock-board Periscope tests."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow_acquisition

SCRIPT = r'''
import asyncio, contextlib, io, os, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6 import QtWidgets
from rfmux.mock.helpers import create_mock_crs
from rfmux.tools.periscope.app import Periscope
app = QtWidgets.QApplication([])
loop = asyncio.new_event_loop()
with contextlib.redirect_stdout(io.StringIO()):
    crs = loop.run_until_complete(create_mock_crs(
        module=1, verbose=False,
        config={"num_resonances": 2, "resonator_random_seed": 5, "auto_bias_kids": True}))
    periscope = Periscope(host="127.0.0.1", module=1, chan_str="1", buf_size=1000,
                          refresh_ms=33, dot_px=1, crs=crs, skip_startup_dialog=True)
periscope.dac_scales = {m: -0.5 for m in range(1, 9)}
periscope._collect_channel_noise({
    "channel_noise": [1, 2], "decimation": 6, "num_samples": 600, "num_segments": 2,
    "reference": "relative", "spectrum_limit": 0.9, "pfb_enabled": False,
    "time_taken": 1.0, "effective_highest_freq": 0.0, "freq_resolution": 0.0})
deadline = time.monotonic() + 90
while time.monotonic() < deadline and not periscope.channel_noise_data.get("data"):
    app.processEvents(); time.sleep(0.01)
data = periscope.channel_noise_data.get("data") or {}
print("RESULT", len(data.get("amplitudes_dbm", [])), sorted(k for k in data if k.startswith("pfb")))
print("TRANSCRIPT", periscope.jupyter_widget._control.toPlainText())
with contextlib.redirect_stdout(io.StringIO()):
    periscope.close()
    loop.run_until_complete(crs.stop_udp_streaming())
'''


def test_channel_noise_runs_as_a_console_cell():
    pytest.importorskip("qtconsole")
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    assert "RESULT 2 ['pfb_enabled']" in out.stdout
    assert ("noise_m1 = await crs.take_noise_spectrum(channels=[1, 2], decimation=6, num_samples=600, "
            "num_segments=2, reference='relative', spectrum_limit=0.9, pfb_samples=None, module=1)"
            ) in out.stdout.split("TRANSCRIPT", 1)[1]
