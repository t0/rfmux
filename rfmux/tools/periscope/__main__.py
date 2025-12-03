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

The application can be run in two ways:
1. Directly as a command after installation: `periscope <crs_board> [options]`
2. As a Python module: `python -m rfmux.tools.periscope <crs_board> [options]`

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
from .mock_configuration_dialog import MockConfigurationDialog

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
import platform
# Note: The original commented-out lines for QtWidgets and QIcon imports
# have been removed as they are expected to be covered by 'from .utils import *'.

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def main():
    """
    Command-line entry point for the Periscope application.

    This function parses command-line arguments to configure and launch the
    Periscope GUI. It supports features like multi-channel grouping,
    custom buffer sizes, frame rate control, and enabling network analysis.
    The application runs in a blocking mode until the GUI is closed.

    The application can be launched in two ways:
    - After installation with pip: `periscope [crs_board] [options]`
    - As a Python module: `python -m rfmux.tools.periscope [crs_board] [options]`

    The CRS board identifier can be specified in three formats:
    - A hostname in the format rfmux####.local (e.g., "rfmux0042.local")
    - Just the serial number (e.g., "0042") 
    - A direct IP address (e.g., "192.168.2.100")
    
    If no arguments are provided, a configuration dialog will be shown.
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
          After installing with `pip install .` in the repository:
          $ periscope rfmux0022.local --module 2 --channels "3&5,7"
          $ periscope 0022 --module 2 --channels "3&5,7"
          $ periscope 192.168.2.100 --module 2 --channels "3&5,7"
          
          Or directly as a Python module:
          $ python -m rfmux.tools.periscope rfmux0022.local --module 2 --channels "3&5,7"

        Run with -h / --help for the full option list.
    """))

    ap.add_argument("crs_board", metavar="CRS_BOARD", nargs='?',
                    help="CRS board identifier: can be a hostname (rfmux####.local), "
                         "a serial number (####), an IP address, or 'MOCK' for demo mode. "
                         "If omitted, a configuration dialog will be shown.")
    ap.add_argument("-m", "--module", type=int, default=1)
    ap.add_argument("-c", "--channels", default="1")
    ap.add_argument("-n", "--num-samples", type=int, default=DEFAULT_BUFFER_SIZE)
    ap.add_argument("-f", "--fps", type=float, default=30.0)
    ap.add_argument("-d", "--density-dot", type=int, default=DENSITY_DOT_SIZE)
    args = ap.parse_args()
    
    # Initialize Qt application first for the dialog
    app = QtWidgets.QApplication(sys.argv[:1])
    app_icon = QIcon(ICON_PATH)
    app.setWindowIcon(app_icon)
    
    # Import the unified startup dialog
    from .session_startup_dialog import UnifiedStartupDialog
    
    # Build pre-fill values from command-line arguments
    prefill = {}
    if args.crs_board is not None:
        crs_board = args.crs_board
        
        # Determine connection mode from CLI arg
        if crs_board.upper() == "MOCK":
            prefill['connection_mode'] = UnifiedStartupDialog.CONN_MOCK
        elif crs_board.upper() == "OFFLINE":
            prefill['connection_mode'] = UnifiedStartupDialog.CONN_OFFLINE
        else:
            prefill['connection_mode'] = UnifiedStartupDialog.CONN_HARDWARE
            # Extract serial from various formats
            if "rfmux" in crs_board and ".local" in crs_board:
                # Format: rfmux0042.local
                serial = crs_board.replace("rfmux", "").replace(".local", "")
                prefill['crs_serial'] = serial
            elif crs_board.isdigit() or (crs_board.startswith("0") and len(crs_board) > 1 and crs_board[1:].isdigit()):
                # Format: 0042 or 42
                prefill['crs_serial'] = crs_board
            else:
                # Direct hostname/IP - use as serial
                prefill['crs_serial'] = crs_board
        
        prefill['module'] = args.module
    
    # Show the startup dialog with pre-filled values
    dialog = UnifiedStartupDialog(None, prefill=prefill if prefill else None)
    
    if not dialog.exec():
        # User cancelled - exit
        sys.exit(0)
    
    # Get configuration from dialog
    config = dialog.get_configuration()
    
    # Override command-line args with dialog values
    connection_mode = config['connection_mode']
    
    if connection_mode == UnifiedStartupDialog.CONN_HARDWARE:
        args.crs_board = config.get('crs_serial', '0042')  # Use serial from dialog
        args.module = config.get('module', 1)
    elif connection_mode == UnifiedStartupDialog.CONN_MOCK:
        args.crs_board = "MOCK"
        args.module = config.get('module', 1)
    elif connection_mode == UnifiedStartupDialog.CONN_OFFLINE:
        # Offline mode - disable hardware connection
        print("[Periscope] Starting in Offline Mode")
        args.crs_board = "OFFLINE"
        args.module = 1
    
    # Store session configuration for later use
    session_mode = config['session_mode']
    session_config = {
        'mode': session_mode,
        'path': config.get('session_path'),
        'folder_name': config.get('session_folder_name')
    }

    if args.fps <= 0:
        ap.error("FPS must be positive.")
    if args.fps > 30:
        warnings.warn("FPS>30 might be unnecessary", RuntimeWarning)
    if args.density_dot < 1:
        ap.error("Density-dot size must be ≥1 pixel.")

    refresh_ms = int(round(1000.0 / args.fps))

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
    is_mock = False  # Track if we're using MockCRS
    
    try:
        # Parse the CRS board identifier - can be in four formats:
        # 1. "MOCK" - special case for demo mode
        # 2. rfmux####.local (hostname with serial number)
        # 3. #### (just the serial number)
        # 4. Any other string (treated as direct hostname or IP address)
        crs_board = args.crs_board
        
        # Special case for MOCK demo mode
        if crs_board.upper() == "MOCK":
            is_mock = True
            # Use the mock flavour to create a MockCRS instance
            s = load_session("""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
""")
            crs_obj = s.query(CRS).one()
            
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Resolve the MockCRS
            loop.run_until_complete(crs_obj.resolve())
            
            # Don't configure any channels - they should be unconfigured until "Initialize CRS" is clicked
            # The UDP streamer will still send noise for all channels even if unconfigured
            print("MockCRS initialized - channels unconfigured (streaming noise only)")
            
            # Start UDP streaming automatically
            print("Starting MockCRS UDP streaming...")
            loop.run_until_complete(crs_obj.start_udp_streaming(host='127.0.0.1', port=9876))
            
            config_dialog = MockConfigurationDialog()
            initial_mock_config = None  # Store for later
            
            if config_dialog.exec():
                # Get configuration
                mock_config = config_dialog.get_configuration()
                initial_mock_config = mock_config  # Save for viewer
                
                try:
                    # Apply configuration to the server
                    resonator_count = loop.run_until_complete(crs_obj.generate_resonators(mock_config))
                    #print(f"Mock configuration applied successfully. Generated {resonator_count} resonators.")
                except Exception as e:
                    import traceback
                    print(f"Error applying mock configuration: {e}")
                    traceback.print_exc()
                    QtWidgets.QMessageBox.critical(None, "Configuration Error", 
                                                 f"Failed to apply mock configuration:\n{str(e)}\n\n"
                                                 f"Details:\n{traceback.format_exc()}")
            else:
                # User cancelled - exit
                print("Mock configuration cancelled, exiting...")
                sys.exit(0)
            
        # Check if it's a hostname in the format rfmux####.local
        elif "rfmux" in crs_board and ".local" in crs_board:
            serial = crs_board.replace("rfmux", "").replace(".local", "")
            s = load_session(f'!HardwareMap [ !CRS {{ serial: "{serial}" }} ]')
            crs_obj = s.query(CRS).one()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(crs_obj.resolve())
        # Check if it's just a serial number (all digits, possibly with leading zeros)
        elif crs_board.isdigit() or (crs_board.startswith("0") and crs_board[1:].isdigit()):
            serial = crs_board
            s = load_session(f'!HardwareMap [ !CRS {{ serial: "{serial}" }} ]')
            crs_obj = s.query(CRS).one()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(crs_obj.resolve())
        else:
            # Treat as direct hostname or IP address
            s = load_session(f'!HardwareMap [ !CRS {{ hostname: "{crs_board}" }} ]')
            crs_obj = s.query(CRS).one()
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
    # For the UDP Receiver, we need a valid hostname, not just a serial number
    udp_hostname = args.crs_board
    
    # Special handling for MOCK mode - always use localhost
    if args.crs_board.upper() == "MOCK":
        udp_hostname = "127.0.0.1"
    elif args.crs_board.upper() == "OFFLINE":
        udp_hostname = "OFFLINE"
    elif args.crs_board.isdigit() or (args.crs_board.startswith("0") and args.crs_board[1:].isdigit()):
        # If it's just a serial number, construct the proper hostname
        udp_hostname = f"rfmux{args.crs_board}.local"
        
    # Create the main Periscope window instance with configured parameters.
    # Skip the startup dialog if session_config was already set by the launcher dialog
    viewer = Periscope(
        host=udp_hostname,
        module=args.module,
        chan_str=args.channels,
        buf_size=args.num_samples,
        refresh_ms=refresh_ms,
        dot_px=args.density_dot,
        crs=crs_obj,  # Pass the CRS object
        skip_startup_dialog=(session_config is not None),  # Skip dialog if already shown
    )
    
    # Store the initial mock configuration if in mock mode
    if is_mock and initial_mock_config is not None:
        viewer.mock_config = initial_mock_config
    
    # Apply session configuration from the startup dialog if provided
    if session_config is not None:
        from .session_startup_dialog import UnifiedStartupDialog
        session_mode = session_config['mode']
        
        if session_mode == UnifiedStartupDialog.SESS_NEW:
            # Start new session
            try:
                viewer.session_manager.start_session(
                    session_config['path'],
                    session_config['folder_name']
                )
                # Show the session browser dock
                if hasattr(viewer, 'session_browser_dock'):
                    viewer.session_browser_dock.show()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    viewer,
                    "Session Error",
                    f"Failed to start session:\n{str(e)}"
                )
        elif session_mode == UnifiedStartupDialog.SESS_LOAD:
            # Load existing session
            success = viewer.session_manager.load_session(session_config['path'])
            if success:
                # Show the session browser dock
                if hasattr(viewer, 'session_browser_dock'):
                    viewer.session_browser_dock.show()
            else:
                QtWidgets.QMessageBox.warning(
                    viewer,
                    "Load Session Failed",
                    f"Could not load session from:\n{session_config['path']}"
                )
        elif session_mode == UnifiedStartupDialog.SESS_NONE:
            # Continue without session - disable auto-export
            viewer.session_manager.auto_export_enabled = False
            viewer.auto_export_action.setChecked(False)
            print("[Session] Continuing without session (auto-export disabled)")
    
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

    # For the UDP Receiver, we need a valid hostname, not just a serial number
    # Get the hostname from the CRS object's tuber_hostname
    udp_hostname = crs_param.tuber_hostname
    
    # Check if it's just a serial number without rfmux prefix and .local suffix
    if (udp_hostname.isdigit() or (udp_hostname.startswith("0") and udp_hostname[1:].isdigit())):
        # If it's just a serial number, construct the proper hostname
        udp_hostname = f"rfmux{udp_hostname}.local"
    
    viewer = Periscope(
        host=udp_hostname,
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
