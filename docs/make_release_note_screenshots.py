#!/usr/bin/env python3
"""Regenerate the Pulse Capture panel screenshots in the release note.

Runs a short simulated capture in the frequency basis, opens the panel on
the resulting file in review mode, and grabs the two tabs the note shows.
The anatomy diagram beside them is drawn by make_pulse_capture_figures.py.

    python docs/make_release_note_screenshots.py

Both shots are taken in dark mode with the df/dissipation view in hertz,
because that is what the release is about: the axes are the frequency
basis, not the quadratures, and the amplitudes carry a real unit.
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "release-notes" / "images"
CAPTURE = Path(os.environ.get("RFMUX_SHOT_SCRATCH", "/tmp")) / "release_shots"

MODULE = 1
CHANNELS = [1, 2]

# 25 ms decay, so a pulse is sampled across tens of points rather than
# arriving as a single spike.  A shorter one is realistic but unreadable
# at screenshot scale.
TAU_S = 0.025

MOCK_CONFIG = {
    "num_resonances": len(CHANNELS),
    "resonator_random_seed": 42,
    "auto_bias_kids": True,
    "pulse_mode": "periodic",
    "pulse_period": 0.25,
    "pulse_tau_rise": 5e-3,
    "pulse_tau_decay": TAU_S,
    # A spread of pulse heights rather than one repeated event, so the
    # amplitude histogram has something to show.
    "pulse_random_amp_mode": "uniform",
    "pulse_random_amp_min": 1.1,
    "pulse_random_amp_max": 1.5,
}


async def _capture(path):
    from rfmux.algorithms.measurement.streamer_config import StreamerConfig
    from rfmux.core.transferfunctions import decimation_to_sampling
    from rfmux.mock.helpers import create_mock_crs
    from rfmux.pulse_capture import (
        PulseCaptureConfig, PulseCaptureSession, run_slow_source)
    from rfmux.streamer import find_streamer_conflict

    conflict = find_streamer_conflict()
    if conflict:
        raise SystemExit(
            f"Something is already using the streamer port — {conflict}. "
            "A second simulation would interleave with it.")

    crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG,
                                verbose=False)

    # Ten or more samples across one decay constant.
    dec = next(d for d in range(6, -1, -1)
               if decimation_to_sampling(d) >= 10.0 / TAU_S)
    cfg = StreamerConfig(dec_stage=dec, short_packets=(dec < 3),
                         modules=[MODULE])
    await crs.configure_streamer(cfg.dec_stage, short=cfg.short_packets,
                                 modules=cfg.modules)
    fs = decimation_to_sampling(dec)

    # auto_bias_kids skips the sweep-and-fit, so the simulated array has
    # no calibration; this is that measurement on its own.
    df_cals = await crs.measure_df_calibrations(channels=CHANNELS,
                                                module=MODULE)

    # Trigger in the frequency basis: the rotation happens before
    # thresholding, so a pulse lands on one axis instead of both.
    capture_config = PulseCaptureConfig(
        threshold_sigma=5.0, end_sigma=1.5, min_pulse_ms=0.2,
        max_pulse_ms=150.0, noise_train_ms=400.0, enable_pileup=True,
        trigger_basis="df")

    session = PulseCaptureSession(
        channels=CHANNELS, module=MODULE, streamer_mode="slow",
        sample_rate=fs, hdf5_path=str(path),
        df_calibrations=df_cals,
        **capture_config.session_kwargs(fs),
    )
    session.start()
    covered = await run_slow_source(session, "127.0.0.1", module=MODULE,
                                    duration_s=60.0)
    session.stop()
    print(f"{session.total_pulses} pulses over {covered:.1f} s at {fs:.0f} Hz")
    return df_cals


def _shoot(path, df_cals):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets

    from rfmux.tools.periscope import pulse_capture_panel as m

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _dark_chrome(app)
    panel = m.PulseCapturePanel(dark_mode=True,
                                df_calibrations={MODULE: df_cals})
    panel.module_spin.setValue(MODULE)
    panel.resize(1720, 950)
    panel.show()
    panel.load_from_hdf5(path)
    panel.units_combo.setCurrentText(m.UNITS_DF)
    app.processEvents()

    OUT.mkdir(parents=True, exist_ok=True)
    for tab, name in (("Pulse View", "pulse-capture-panel-review.png"),
                      ("Histograms", "pulse-capture-panel-histograms.png")):
        _select_tab(panel, tab)
        for _ in range(6):
            app.processEvents()
        panel.grab().save(str(OUT / name))
        print(f"wrote {OUT / name}")
    panel.close()


def _dark_chrome(app):
    """Dark window chrome for the screenshot.

    rfmux's dark_mode themes the plots only; the surrounding widgets
    take the desktop theme, which an offscreen render does not have.
    Setting it here keeps the shots dark wherever they are generated,
    rather than depending on whose machine ran the script.
    """
    from PyQt6 import QtGui
    from PyQt6.QtGui import QPalette

    app.setStyle("Fusion")
    window, base, text = (QtGui.QColor(53, 53, 53),
                          QtGui.QColor(35, 35, 35),
                          QtGui.QColor(220, 220, 220))
    pal = QtGui.QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Button,
                 QPalette.ColorRole.ToolTipBase):
        pal.setColor(role, window)
    for role in (QPalette.ColorRole.Base, QPalette.ColorRole.AlternateBase):
        pal.setColor(role, base)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.ToolTipText,
                 QPalette.ColorRole.BrightText):
        pal.setColor(role, text)
    pal.setColor(QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    pal.setColor(QPalette.ColorRole.HighlightedText, QtGui.QColor(0, 0, 0))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role,
                     QtGui.QColor(130, 130, 130))
    app.setPalette(pal)


def _select_tab(panel, label):
    from PyQt6 import QtWidgets

    tabs = panel.findChild(QtWidgets.QTabWidget)
    for i in range(tabs.count()):
        if tabs.tabText(i) == label:
            tabs.setCurrentIndex(i)
            return
    raise SystemExit(f"no tab named {label!r}")


def main():
    CAPTURE.mkdir(parents=True, exist_ok=True)
    path = CAPTURE / "release_demo.h5"
    df_cals = asyncio.run(_capture(path))
    _shoot(path, df_cals)


if __name__ == "__main__":
    main()
