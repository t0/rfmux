"""A panel's background work and the end of the panel or the window.

A QThread destroyed while it runs aborts the process ("QThread: Destroyed
while thread is still running"), so the crash case runs in a subprocess.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope import tasks  # noqa: E402
from rfmux.tools.periscope.app_runtime import PeriscopeRuntime  # noqa: E402
from rfmux.tools.periscope.multisweep_panel import MultisweepPanel  # noqa: E402

_COLLISION_SCRIPT = textwrap.dedent('''
    import os, sys, time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, "__ROOT__")
    from PyQt6 import QtWidgets, sip
    from rfmux.tools.periscope import collision_settings_panel as csp
    from rfmux.tools.periscope.multisweep_panel import MultisweepPanel

    app = QtWidgets.QApplication([])
    csp.CollisionTask.run = lambda self: time.sleep(3)
    panel = MultisweepPanel(target_module=1)
    panel.module_sweeps = {}
    panel.catalog_editor.get_parameters = lambda: {}
    panel._run_collision_cut()
    time.sleep(0.3)
    sip.delete(panel)
    time.sleep(3.5)
    print("SURVIVED", flush=True)
''')


def test_closing_a_panel_during_its_collision_check_does_not_crash():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-u", "-c", _COLLISION_SCRIPT.replace("__ROOT__", root)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert proc.returncode == 0 and "SURVIVED" in proc.stdout, (
        f"returncode {proc.returncode} (-6 is the abort)\n"
        f"stderr tail:\n{proc.stderr[-1500:]}")


def test_closing_the_window_waits_for_apply_bias(qt_app, monkeypatch):
    """Cut off between moving the NCO and setting the tones, Apply Bias
    would leave every tone displaced."""
    finished = []

    def slow_apply(self):
        time.sleep(0.5)
        finished.append(True)

    monkeypatch.setattr(tasks.ApplyBiasTask, "run", slow_apply)
    window = QtWidgets.QMainWindow()
    panel = MultisweepPanel(window, target_module=1)
    panel._apply_bias_task = tasks.ApplyBiasTask(None, None, tasks.ApplyBiasSignals())
    panel._apply_bias_task.start()

    PeriscopeRuntime._wait_for_bias_writes(window)

    assert finished == [True]
    window.close()
