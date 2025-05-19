#!/usr/bin/env -S uv run
"""Periscope – Real‑Time Multi‑Pane Viewer

Main application entry point. The heavy ``Periscope`` class lives
in :mod:`periscope_app` along with helper modules.
"""

from .periscope_app import Periscope
from .periscope_utils import *
from .periscope_tasks import *
from .periscope_ui import *

def main():
    """
    Entry point for command-line usage. Supports multi-channel grouping via '&',
    auto-scaling, and global I/Q/M toggles. Launches Periscope in blocking mode.
    """
    ap = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
        Periscope — real-time CRS packet visualizer with network analysis.

        Connects to a UDP/multicast stream, filters a single module, and drives a
        PyQt6 GUI with up to six linked views per channel group:

          • Time-domain waveform (TOD)
          • IQ plane (density or scatter)
          • Raw FFT
          • Single-sideband PSD (SSB)  – CIC-corrected
          • Dual-sideband PSD (DSB)    – CIC-corrected
          • Network Analysis — amplitude and phase vs frequency

        Key options
        -----------
          • Comma-separated channel list with '&' to overlay channels on one row,
            e.g. "3&5,7" → two rows: {3,5} and {7}
          • -n / --num-samples   Ring-buffer length per channel (history / FFT depth)
          • -f / --fps           Maximum GUI refresh rate (frames s⁻¹) [typically system-limited anyway]
          • -d / --density-dot   Dot size in IQ-density mode (pixels) [not typically adjusted]
          • --enable-netanal     Create CRS object to enable network analysis

        Advanced features
        -----------------
          • Real-unit conversion (counts → V, dBm/Hz, dBc/Hz) with CIC droop compensation in the PSDs
          • Welch segmentation to average PSD noise floor
          • Network analysis with real-time amplitude and phase measurements

        Example
        -------
          $ periscope rfmux0022.local --module 2 --channels "3&5,7" --enable-netanal

        Run with -h / --help for the full option list.
    """))

    ap.add_argument("hostname")
    ap.add_argument("-m", "--module", type=int, default=1)
    ap.add_argument("-c", "--channels", default="1")
    ap.add_argument("-n", "--num-samples", type=int, default=DEFAULT_BUFFER_SIZE)
    ap.add_argument("-f", "--fps", type=float, default=30.0)
    ap.add_argument("-d", "--density-dot", type=int, default=DENSITY_DOT_SIZE)
    args = ap.parse_args()

    if args.fps <= 0:
        ap.error("FPS must be positive.")
    if args.fps > 30:
        warnings.warn("FPS>30 might be unnecessary", RuntimeWarning)
    if args.density_dot < 1:
        ap.error("Density-dot size must be ≥1 pixel.")

    refresh_ms = int(round(1000.0 / args.fps))
    app = QtWidgets.QApplication(sys.argv[:1])
    app_icon = QIcon(ICON_PATH)
    app.setWindowIcon(app_icon)

    # --- Global Exception Hook ---
    def global_exception_hook(exctype, value, tb):
        # Ensure traceback is imported if not already
        import traceback as tb_module
        print("Unhandled exception in Periscope:", file=sys.stderr)
        tb_module.print_exception(exctype, value, tb, file=sys.stderr)
        # Optionally, re-raise or call sys.__excepthook__ if you want Python's default handling too
        # For a GUI app, often just logging is preferred over crashing.
        # sys.__excepthook__(exctype, value, tb)

    sys.excepthook = global_exception_hook
    # --- End Global Exception Hook ---

    # Bind only the main GUI (this) thread to a single CPU to reduce scheduling jitter
    pin_current_thread_to_core()

    # Create CRS object so we can tell it what to do
    crs = None
    try:
        # Extract serial number from hostname
        hostname = args.hostname
        if "rfmux" in hostname and ".local" in hostname:
            serial = hostname.replace("rfmux", "").replace(".local", "")
        else:
            # Default to hostname as serial
            serial = hostname
        
        # Create and resolve CRS in a synchronous way
        s = load_session(f'!HardwareMap [ !CRS {{ serial: "{serial}" }} ]')
        crs = s.query(CRS).one()
        
        # Resolve the CRS object
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(crs.resolve())
        
    except Exception as e:
        warnings.warn(f"Failed to create CRS object: {str(e)}\nNetwork analysis will be disabled.")
        crs = None

    viewer = Periscope(
        host=args.hostname,
        module=args.module,
        chan_str=args.channels,
        buf_size=args.num_samples,
        refresh_ms=refresh_ms,
        dot_px=args.density_dot,
        crs=crs  # Pass the CRS object
    )
    # Also set icon for main window
    viewer.setWindowIcon(app_icon)
    viewer.show()
    sys.exit(app.exec())

@macro(CRS, register=True)
async def raise_periscope(
    crs: CRS,
    *,
    module: int = 1,
    channels: str = "1",
    buf_size: int = DEFAULT_BUFFER_SIZE,
    fps: float = 30.0,
    density_dot: int = DENSITY_DOT_SIZE,
    blocking: bool | None = None,
):
    """
    Programmatic entry point for embedding or interactive usage of Periscope.

    This function either blocks (runs the Qt event loop) or returns control
    immediately, depending on whether a Qt event loop is already running or the
    'blocking' parameter is explicitly set.

    Parameters
    ----------
    crs : CRS object
        CRS object for hardware communication (needed for network analysis).
    module : int, optional
        Module number to filter data (default is 1).
    channels : str, optional
        A comma-separated list of channel indices, possibly with '&' to group
        multiple channels on one row (default "1").
    buf_size : int, optional
        Ring buffer size for each channel (default 5000).
    fps : float, optional
        Frames per second (default 30.0). Determines GUI update rate.
    density_dot : int, optional
        Dot dilation in pixels for IQ density mode (default 1).
    blocking : bool or None, optional
        If True, runs the Qt event loop until exit. If False, returns control
        immediately. If None, infers from the environment.

    Returns
    -------
    Periscope or (Periscope, QApplication)
        The Periscope instance, or (instance, QApplication) if non-blocking.
    """
    ip = get_ipython()
    qt_loop = is_qt_event_loop_running()

    if ip and not qt_loop:
        ip.run_line_magic("gui", "qt")
        qt_loop = True

    if blocking is None:
        blocking = not qt_loop

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app_icon = QIcon(ICON_PATH)
    app.setWindowIcon(app_icon)

    # --- Global Exception Hook (also for library use) ---
    # It's good practice to set this up if Periscope might be run programmatically
    # where the main() function isn't the entry point.
    # However, be cautious if the host application (e.g., Jupyter) has its own hook.
    if not hasattr(sys, '_periscope_excepthook_installed'): # Avoid multiple installs
        def global_exception_hook_lib(exctype, value, tb):
            import traceback as tb_module
            print("Unhandled exception in Periscope (library mode):", file=sys.stderr)
            tb_module.print_exception(exctype, value, tb, file=sys.stderr)
        
        # Check if we are in an environment that might already have a complex hook (like IPython)
        # A more robust check might be needed for various environments.
        if get_ipython() is None: # Simple check: if not in IPython, assume it's safe to set.
            sys.excepthook = global_exception_hook_lib
            sys._periscope_excepthook_installed = True # Mark as installed
    # --- End Global Exception Hook ---

    # Bind only the main GUI (this) thread to a single CPU to reduce scheduling jitter
    pin_current_thread_to_core()

    refresh_ms = int(round(1000.0 / fps))

    viewer = Periscope(
        host=crs.tuber_hostname,
        module=module,
        chan_str=channels,
        buf_size=buf_size,
        refresh_ms=refresh_ms,
        dot_px=density_dot,
        crs=crs,  # Pass the CRS object for network analysis
    )
    viewer.setWindowIcon(app_icon)
    viewer.show()

    if blocking:
        if is_running_inside_ipython():
            app.exec()
        else:
            sys.exit(app.exec())
        return viewer
    else:
        return viewer, app


if __name__ == "__main__":
    main()
