"""Panels run their board work as literal Python in the session: the
string the console shows is the string that executed."""

import asyncio
import subprocess
import sys
import time

import pytest

from rfmux.tools.periscope.console_kernel import Interpreter


@pytest.fixture
def interpreter():
    interp = Interpreter()
    yield interp
    interp.close()


def test_run_executes_top_level_await_in_the_namespace(interpreter):
    ns = {"asyncio": asyncio}
    interpreter.run("await asyncio.sleep(0)\nx = 6 * 7", ns).result(timeout=5)
    assert ns["x"] == 42


def test_cancelling_the_future_cancels_the_code(interpreter):
    ns = {"asyncio": asyncio, "seen": []}
    code = ("try:\n    await asyncio.sleep(10)\n"
            "except asyncio.CancelledError:\n    seen.append('cancelled')\n    raise")
    future = interpreter.run(code, ns)
    time.sleep(0.1)
    future.cancel()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not ns["seen"]:
        time.sleep(0.01)
    assert ns["seen"] == ["cancelled"]


# Periscope with the mock board runs in a child process: an InteractiveShell
# is a process-wide singleton, so whether a console kernel can exist depends
# on what ran earlier in the process.
PROLOGUE = r'''
import asyncio, contextlib, io, os, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6 import QtWidgets
from rfmux import load_session
from rfmux.core.schema import CRS
from rfmux.tools.periscope.app import Periscope
app = QtWidgets.QApplication([])
with contextlib.redirect_stdout(io.StringIO()):
    session = load_session('!HardwareMap\n- !flavour "rfmux.mock"\n- !CRS { serial: "0000", hostname: "127.0.0.1" }\n')
    crs = session.query(CRS).one(); asyncio.run(crs.resolve())
    periscope = Periscope(host="127.0.0.1", module=1, chan_str="1", buf_size=1000,
                          refresh_ms=33, dot_px=1, crs=crs, skip_startup_dialog=True)
periscope.dac_scales = {m: -0.5 for m in range(1, 9)}
'''

SWEEP_TAIL = r'''
deadline = time.monotonic() + 150
while time.monotonic() < deadline and periscope.netanal_tasks:
    app.processEvents(); time.sleep(0.01)
result = periscope.session_namespace()["netanal_0"]
print("RESULT", {amp: [r["frequencies"].size for r in per_module] for amp, per_module in result.items()})
print("HISTORY", periscope.kernel_manager.kernel.shell.history_manager.input_hist_raw[-1])
print("TRANSCRIPT", periscope.jupyter_widget._control.toPlainText())
with contextlib.redirect_stdout(io.StringIO()):
    periscope.close()
'''

SWEEP = PROLOGUE + r'''
periscope._start_network_analysis({
    "module": 1, "amps": [0.001], "fmin": 1000e6, "fmax": 1100e6, "npoints": 100,
    "nsamps": 2, "max_chans": 64, "max_span": 500e6, "cable_length": 10.0,
    "clear_channels": True})
''' + SWEEP_TAIL

SWEEP_MULTI = PROLOGUE + r'''
periscope._start_network_analysis({
    "module": [1, 2], "amps": [0.001, 0.002], "fmin": 1000e6, "fmax": 1050e6, "npoints": 40,
    "nsamps": 2, "max_chans": 64, "max_span": 500e6, "cable_length": 10.0,
    "clear_channels": False})
''' + SWEEP_TAIL

# A multisweep at two powers with a skewed fit: the second power re-centres
# on the first's bias points by name, and each fit is its own cell.
MULTISWEEP = PROLOGUE + r'''
with contextlib.redirect_stdout(io.StringIO()):
    asyncio.run(crs.generate_resonators({"num_resonances": 2, "resonator_random_seed": 5,
                                         "auto_bias_kids": True, "bias_amplitude": 0.001}))
nco = asyncio.run(crs.get_nco_frequency(module=1))
cfs = [nco + asyncio.run(crs.get_frequency(channel=ch, module=1)) for ch in (1, 2)]
periscope._start_multisweep_analysis({
    "module": 1, "resonance_frequencies": cfs, "span_hz": 200e3, "npoints_per_sweep": 21,
    "nsamps": 1, "amps": [0.001, 0.002], "sweep_direction": "upward",
    "bias_frequency_method": "max-diq", "rotate_saved_data": False,
    "apply_skewed_fit": True, "apply_nonlinear_fit": False})
panel = periscope.multisweep_windows["multisweep_0"]["window"]
deadline = time.monotonic() + 150
while time.monotonic() < deadline and len(panel.results_by_detector.get(1, {})) < 2:
    app.processEvents(); time.sleep(0.01)
print("RESULT", sorted(periscope.session_namespace()["multisweep_0"]),
      [sorted(v) for v in panel.results_by_detector.values()],
      panel.results_by_detector[1][1].get("skewed_fit_applied"))
print("TRANSCRIPT", periscope.jupyter_widget._control.toPlainText())
with contextlib.redirect_stdout(io.StringIO()):
    periscope.close()
'''

# Export and load: the file carries the named value, and loading binds it
# again as a cell under a free name, from which a second panel derives.
ROUNDTRIP = PROLOGUE + r'''
import pickle, tempfile
periscope._start_network_analysis({
    "module": 1, "amps": [0.001], "fmin": 1000e6, "fmax": 1100e6, "npoints": 100,
    "nsamps": 2, "max_chans": 64, "max_span": 500e6, "cable_length": 10.0,
    "clear_channels": True})
deadline = time.monotonic() + 150
while time.monotonic() < deadline and periscope.netanal_tasks:
    app.processEvents(); time.sleep(0.01)
panel = periscope.netanal_windows["netanal_0"]["window"]
export = panel.build_export_dict()
path = tempfile.mktemp(suffix=".pkl")
with open(path, "wb") as f:
    pickle.dump(export, f)
periscope._load_netanal_from_session(export, path)
deadline = time.monotonic() + 60
while time.monotonic() < deadline and "netanal_1" not in periscope.netanal_windows:
    app.processEvents(); time.sleep(0.01)
loaded = periscope.netanal_windows["netanal_1"]["window"]
ns = periscope.session_namespace()
print("ROUNDTRIP", export["name"], sorted(export["result"]), len(export["cells"]),
      loaded.result_name, sorted(k for k in loaded.data[1] if k != "default"),
      ns[loaded.result_name][0.001][0]["frequencies"].size)
print("TRANSCRIPT", periscope.jupyter_widget._control.toPlainText())
with contextlib.redirect_stdout(io.StringIO()):
    periscope.close()
'''

# A one-call panel action: the QP pulse toggle in mock mode.
QP_PULSES = PROLOGUE + r'''
periscope.is_mock_mode = True
periscope._toggle_qp_pulses()
deadline = time.monotonic() + 60
while time.monotonic() < deadline and periscope.qp_pulse_mode == "none":
    app.processEvents(); time.sleep(0.01)
print("MODE", periscope.qp_pulse_mode)
print("TRANSCRIPT", periscope.jupyter_widget._control.toPlainText())
with contextlib.redirect_stdout(io.StringIO()):
    periscope.close()
'''

# Raised from IPython there is no second shell: the enclosing session is the
# namespace and the transcript goes to its stdout.
HOSTED = r'''
from IPython.terminal.interactiveshell import TerminalInteractiveShell
shell = TerminalInteractiveShell.instance()
''' + PROLOGUE + r'''
periscope.run_python("answer = 6 * 7", "a check").result(timeout=10)
print("RESULT", shell.user_ns.get("answer"), shell.user_ns.get("crs") is crs, periscope.kernel_manager is None)
with contextlib.redirect_stdout(io.StringIO()):
    periscope.close()
'''


def _child(script: str) -> str:
    pytest.importorskip("qtconsole")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=180)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    return out.stdout


def test_network_analysis_runs_as_a_console_cell():
    out = _child(SWEEP)
    assert "RESULT {0.001: [100]}" in out
    cell = ("netanal_0[0.001] = await crs.take_netanal(amp=0.001, fmin=1000000000.0, fmax=1100000000.0, "
            "nsamps=2, npoints=100, max_chans=64, max_span=500000000.0, module=[1], "
            "**periscope.netanal_hooks('netanal_0', 0.001))")
    assert cell in out.split("HISTORY", 1)[1].split("TRANSCRIPT", 1)[0], "recorded in IPython history"
    assert cell in out.split("TRANSCRIPT", 1)[1], "shown in the console"
    assert "In [" in out.split("TRANSCRIPT", 1)[1]


def test_multi_module_sweep_is_one_flat_call_per_amplitude():
    out = _child(SWEEP_MULTI)
    assert "RESULT {0.001: [40, 40], 0.002: [40, 40]}" in out
    transcript = out.split("TRANSCRIPT", 1)[1]
    for amp in ("0.001", "0.002"):
        assert (f"netanal_0[{amp}] = await crs.take_netanal(amp={amp}, " in transcript
                and f"module=[1, 2], **periscope.netanal_hooks('netanal_0', {amp}))" in transcript)
    assert "asyncio.gather" not in transcript and "for amp in" not in transcript


def test_multisweep_re_centres_by_name_and_fits_in_the_call():
    out = _child(MULTISWEEP)
    assert "RESULT [(0.001, 'upward'), (0.002, 'upward')] [[0, 1], [0, 1]] True" in out
    transcript = out.split("TRANSCRIPT", 1)[1]
    assert "multisweep_0[(0.001, 'upward')] = await crs.multisweep(center_frequencies=[" in transcript
    assert ("multisweep_0[(0.002, 'upward')] = await crs.multisweep("
            "center_frequencies=bias_frequencies(multisweep_0[(0.001, 'upward')]), span_hz=200000.0, "
            "npoints_per_sweep=21, nsamps=1, bias_frequency_method='max-diq', rotate_saved_data=False, "
            "amp=0.002, sweep_direction='upward', fit_skewed=True, fit_nonlinear=False, module=1, "
            "**periscope.multisweep_hooks('multisweep_0'))") in transcript
    assert "fit_multisweep(" not in transcript


def test_export_carries_the_named_value_and_load_binds_it_again():
    out = _child(ROUNDTRIP)
    assert "ROUNDTRIP netanal_0 [0.001] 1 netanal_0_2 ['1_0.001'] 100" in out
    assert "netanal_0_2 = load_result(" in out.split("TRANSCRIPT", 1)[1]


def test_qp_pulse_toggle_is_a_console_cell():
    out = _child(QP_PULSES)
    assert "MODE periodic" in out
    assert "await crs.set_pulse_mode('periodic', " in out.split("TRANSCRIPT", 1)[1]


def test_raised_from_ipython_the_enclosing_session_is_the_namespace():
    out = _child(HOSTED)
    assert "RESULT 42 True True" in out
    assert "# a check\nanswer = 6 * 7\n" in out
