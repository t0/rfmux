#!/usr/bin/env -S uv run
# The shebang line above allows this script to be executed directly,
# using 'uv run' for optimized execution if 'uv' (a fast Python installer and runner)
# is available in the environment.
"""
Periscope – Real‑Time Multi‑Pane Viewer and Network Analysis Tool

This module serves as the primary executable entry point for the Periscope application.
It handles command-line argument parsing for launching Periscope from a terminal
and provides a programmatic function (`raise_periscope`) for embedding or
launching Periscope from other Python scripts or interactive sessions (e.g., IPython).

The core application logic and the main `Periscope` class are defined in the
`app.py` module, with UI elements in `ui.py`, utility functions in `utils.py`,
and background tasks in `tasks.py`.
"""

import argparse
import textwrap
import sys
import warnings
import asyncio

from .app import Periscope  # Core application class

# Wildcard imports are used here for convenience to bring in numerous
# utility functions, constants (like ICON_PATH, DEFAULT_BUFFER_SIZE),
# task-related classes, and UI components from their respective modules.
# While explicit imports are generally preferred for clarity and to avoid
# namespace pollution, they would require listing many individual names here.
# It's assumed that critical components like QtWidgets, QIcon, etc.,
# are provided as expected by these wildcard imports.
from .utils import *
from .tasks import *
from .ui import *
# Note: The original commented-out lines for QtWidgets and QIcon imports
# have been removed as they are expected to be covered by 'from .utils import *'.

def main():
    """
    Command-line entry point for the Periscope application.

    This function parses command-line arguments to configure and launch the
    Periscope GUI. It supports features like multi-channel grouping,
    custom buffer sizes, frame rate control, and enabling network analysis.
    The application runs in a blocking mode until the GUI is closed.
    """
    # --- Argument Parsing ---
    # Set up command-line argument parsing with a detailed description and examples.
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
          $ python -m rfmux.tools.periscope rfmux0022.local --module 2 --channels "3&5,7" --enable-netanal

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
    # Initialize the Qt application. sys.argv[:1] is used to prevent Qt
    # from trying to parse Periscope's own command-line arguments.
    app = QtWidgets.QApplication(sys.argv[:1])
    
    # Set the application icon. ICON_PATH is imported from .utils.
    app_icon = QIcon(ICON_PATH)
    app.setWindowIcon(app_icon)

    # --- Global Exception Hook Setup ---
    # Installs a global exception hook to catch any unhandled exceptions
    # that occur in the main thread, printing them to stderr. This is crucial
    # for GUI applications to provide error feedback instead of silently crashing.
    def global_exception_hook(exctype, value, tb):
        # Ensure traceback is imported if not already, to prevent import errors
        # during exception handling itself.
        import traceback as tb_module
        print("Unhandled exception in Periscope (CLI mode):", file=sys.stderr)
        tb_module.print_exception(exctype, value, tb, file=sys.stderr)
        # For a GUI app, logging is often preferred over re-raising and crashing.
        # If default Python behavior is also desired, uncomment:
        # sys.__excepthook__(exctype, value, tb)

    sys.excepthook = global_exception_hook
    # --- End Global Exception Hook ---

    # --- Performance Optimization ---
    # Bind the main GUI thread to a specific CPU core if the utility is available.
    # This can help reduce scheduling jitter and improve GUI responsiveness.
    # pin_current_thread_to_core is imported from .utils.
    pin_current_thread_to_core()

    # --- CRS Object Initialization (for Network Analysis) ---
    # Attempt to create and initialize a CRS (Control and Readout System) object.
    # This object is necessary for network analysis functionalities.
    # load_session and CRS are expected to be imported from .utils.
    crs_obj = None  # Initialize to None; will be set if successful.
    try:
        # Attempt to extract a serial number from the provided hostname
        # (e.g., "rfmux0022.local" -> "0022").
        hostname = args.hostname
        if "rfmux" in hostname and ".local" in hostname:
            serial = hostname.replace("rfmux", "").replace(".local", "")
            # Create a session and query for the CRS object based on the serial number.
            # This is done synchronously for the CLI startup.
            s = load_session(f'!HardwareMap [ !CRS {{ serial: "{serial}" }} ]')            
        else:
            # If hostname doesn't match expected pattern, use it directly as serial.
            # This might be relevant for direct IP connections or other naming schemes.
            serial = hostname
            s = load_session(f'!HardwareMap [ !CRS {{ hostname: "{serial}" }} ]')        

        crs_obj = s.query(CRS).one() # Expecting one CRS object.
        
        # Resolve the CRS object to establish connection and prepare it for use.
        # This is an asynchronous operation run synchronously here using a new event loop.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(crs_obj.resolve())
        
    except Exception as e:
        # If CRS object creation or resolution fails, issue a warning
        # and proceed without network analysis capabilities.
        warnings.warn(
            f"Failed to create or resolve CRS object: {str(e)}\n"
            "Network analysis functionalities will be disabled."
        )
        crs_obj = None  # Ensure crs_obj remains None.

    # --- Periscope Application Instantiation and Launch ---
    # Create the main Periscope window instance with configured parameters.
    viewer = Periscope(
        host=args.hostname,
        module=args.module,
        chan_str=args.channels,
        buf_size=args.num_samples,
        refresh_ms=refresh_ms,
        dot_px=args.density_dot,
        crs=crs_obj  # Pass the CRS object
    )
    # --- Application Execution ---
    # Set the icon for the main Periscope window, show it, and start the Qt event loop.
    # sys.exit(app.exec()) ensures that the application's exit code is propagated.
    viewer.setWindowIcon(app_icon)
    viewer.show()
    sys.exit(app.exec())

# Note on wildcard imports from .utils:
# The following names (and potentially others) are expected to be made available
# in the global scope of this module via 'from .utils import *':
#   - macro, CRS (decorators/classes for system integration)
#   - get_ipython, is_qt_event_loop_running, is_running_inside_ipython (IPython/Qt utilities)
#   - ICON_PATH, DEFAULT_BUFFER_SIZE, DENSITY_DOT_SIZE (application constants)
#   - pin_current_thread_to_core (performance utility)
#   - QtWidgets, QIcon (core Qt components)
# This is a common pattern in script-like modules or for very frequently used utilities,
# but for larger projects, explicit imports are generally recommended.

@macro(CRS, register=True) # Register this function with the CRS system if applicable.
async def raise_periscope(
    crs_param: CRS, # Renamed to avoid conflict with crs module
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
    crs_param : CRS object
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
    if not hasattr(sys, '_periscope_excepthook_installed'): 
        def global_exception_hook_lib(exctype, value, tb):
            import traceback as tb_module
            print("Unhandled exception in Periscope (library mode):", file=sys.stderr)
            tb_module.print_exception(exctype, value, tb, file=sys.stderr)
        
        if get_ipython() is None: 
            sys.excepthook = global_exception_hook_lib
            sys._periscope_excepthook_installed = True
    # --- End Global Exception Hook ---

    # --- Performance Optimization ---
    # Bind the current thread (which could be different from the main GUI thread
    # if called from a different context) to a specific CPU core.
    # This aims to reduce scheduling jitter.
    pin_current_thread_to_core()

    refresh_ms = int(round(1000.0 / fps))  # Calculate refresh interval in milliseconds.

    viewer = Periscope(
        host=crs_param.tuber_hostname,
        module=module,
        chan_str=channels,
        buf_size=buf_size,
        refresh_ms=refresh_ms,
        dot_px=density_dot,
        crs=crs_param,
    )
    viewer.setWindowIcon(app_icon)
    viewer.show()

    if blocking:
        if is_running_inside_ipython():
            app.exec() # Use exec() for PyQt6
        else:
            sys.exit(app.exec()) # Use exec() for PyQt6
        return viewer
    else:
        return viewer, app


if __name__ == "__main__":
    main()
