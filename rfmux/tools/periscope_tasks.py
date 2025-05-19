"""Worker threads and tasks used by the Periscope viewer."""

from .periscope_utils import *
from rfmux.algorithms.measurement import fitting # Added for MultisweepTask

class UDPReceiver(QtCore.QThread):
    """
    Receives multicast packets in a dedicated QThread and pushes them
    into a thread-safe queue.

    Parameters
    ----------
    host : str
        The multicast or UDP host address.
    module : int
        The module number to filter on.
    """

    def __init__(self, host: str, module: int) -> None:
        super().__init__()
        self.module_id = module
        self.queue = queue.Queue()
        self.sock = streamer.get_multicast_socket(host)
        self.sock.settimeout(0.2)

    def run(self):
        """
        Main loop that receives packets from the socket, parses them, and
        adds them to the internal queue. Exits upon thread interruption
        or socket error.
        """
        while not self.isInterruptionRequested():
            try:
                data = self.sock.recv(streamer.STREAMER_LEN)
                pkt = streamer.DfmuxPacket.from_bytes(data)
            except socket.timeout:
                continue
            except OSError:
                break
            if pkt.module == self.module_id - 1:
                self.queue.put(pkt)

    def stop(self):
        """
        Signal the thread to stop, and close the socket.
        """
        self.requestInterruption()
        try:
            self.sock.close()
        except OSError:
            pass

# ───────────────────────── IQ Task & Signals ─────────────────────────
class IQSignals(QObject):
    """
    Holds custom signals emitted by IQ tasks.
    """
    done = pyqtSignal(int, str, object)
    # Emitted with arguments: (row, mode_string, payload)
    #   row: row index
    #   mode_string: "density" or "scatter"
    #   payload: data from the computation

class IQTask(QRunnable):
    """
    Off-thread worker for computing IQ scatter or density histograms.

    Parameters
    ----------
    row : int
        Row index in the channel list (maps results back to the correct row).
    ch : int
        Actual channel number from which the data originates.
    I : np.ndarray
        Array of I samples.
    Q : np.ndarray
        Array of Q samples.
    dot_px : int
        Dot diameter in pixels for point dilation in density mode.
    mode : {"density", "scatter"}
        Determine whether to compute a 2D histogram or scatter points.
    signals : IQSignals
        An IQSignals instance for communicating results back to the GUI thread.
    """

    def __init__(self, row, ch, I, Q, dot_px, mode, signals: IQSignals):
        super().__init__()
        self.row = row
        self.ch = ch
        self.I = I.copy()
        self.Q = Q.copy()
        self.dot_px = dot_px
        self.mode = mode
        self.signals = signals

    def run(self):
        """
        Perform the required IQ computation off the main thread. Emit results
        via self.signals.done.
        """
        if len(self.I) < 2:
            self._handle_insufficient_data()
            return

        if self.mode == "density":
            payload = self._compute_density()
        else:  # scatter mode
            payload = self._compute_scatter()

        self.signals.done.emit(self.row, self.mode, payload)
        
    def _handle_insufficient_data(self):
        """Handle the edge case of insufficient data."""
        if self.mode == "density":
            empty = np.zeros((DENSITY_GRID, DENSITY_GRID), np.uint8)
            payload = (empty, (0, 1, 0, 1))
        else:
            payload = ([], [], [])
        self.signals.done.emit(self.row, self.mode, payload)
        
    def _compute_density(self):
        """Compute IQ density histogram."""
        g = DENSITY_GRID
        hist = np.zeros((g, g), np.uint32)

        Imin, Imax = self.I.min(), self.I.max()
        Qmin, Qmax = self.Q.min(), self.Q.max()
        if Imin == Imax or Qmin == Qmax:
            return (hist.astype(np.uint8), (Imin, Imax, Qmin, Qmax))

        # Map I/Q data to pixel indices
        ix = ((self.I - Imin) * (g - 1) / (Imax - Imin)).astype(np.intp)
        qy = ((self.Q - Qmin) * (g - 1) / (Qmax - Qmin)).astype(np.intp)

        # Base histogram
        np.add.at(hist, (qy, ix), 1)

        # Optional dot dilation
        if self.dot_px > 1:
            self._apply_dot_dilation(hist, ix, qy, g)

        # Optional smoothing & log-compression
        if gaussian_filter is not None and SMOOTH_SIGMA > 0:
            hist = gaussian_filter(hist.astype(np.float32), SMOOTH_SIGMA, mode="nearest")
        if LOG_COMPRESS:
            hist = np.log1p(hist, out=hist.astype(np.float32))

        # 8-bit normalization
        if hist.max() > 0:
            hist = (hist * (255.0 / hist.max())).astype(np.uint8)

        return (hist, (Imin, Imax, Qmin, Qmax))
        
    def _apply_dot_dilation(self, hist, ix, qy, g):
        """Apply dot dilation to the histogram."""
        r = self.dot_px // 2
        if convolve is not None:
            # Faster path if SciPy is available
            k = 2 * r + 1
            kernel = np.ones((k, k), dtype=np.uint8)
            hist = convolve(hist, kernel, mode="constant", cval=0)
        else:
            # Fallback
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ys, xs = qy + dy, ix + dx
                    mask = ((0 <= ys) & (ys < g) &
                            (0 <= xs) & (xs < g))
                    np.add.at(hist, (ys[mask], xs[mask]), 1)
                    
    def _compute_scatter(self):
        """Compute IQ scatter points."""
        N = len(self.I)
        if N > SCATTER_POINTS:
            idx = np.linspace(0, N - 1, SCATTER_POINTS, dtype=np.intp)
        else:
            idx = np.arange(N, dtype=np.intp)
        xs, ys = self.I[idx], self.Q[idx]
        rel = idx / (idx.max() if idx.size else 1)
        colors = pg.colormap.get("turbo").map(
            rel.astype(np.float32), mode="byte"
        )
        return (xs, ys, colors)

# ───────────────────────── PSD Task & Signals ─────────────────────────
class PSDSignals(QObject):
    """
    Holds custom signals emitted by PSD tasks.
    """
    done = pyqtSignal(int, str, int, object)
    # Emitted with arguments: (row, mode_string, channel, payload)

class PSDTask(QRunnable):
    """
    Off‑thread worker for single or dual sideband PSD computation.

    Parameters
    ----------
    row : int
        Row index in the channel list (used to map results back to the correct UI row).
    ch : int
        Actual channel number (data source).
    I : np.ndarray
        Array of I samples.
    Q : np.ndarray
        Array of Q samples.
    mode : {"SSB", "DSB"}
        Determines the type of PSD computation.
    dec_stage : int
        Decimation stage for the spectrum_from_slow_tod() call.
    real_units : bool
        If True, convert PSD to dBm/dBc. Otherwise, keep as raw counts²/Hz.
    psd_absolute : bool
        If True and real_units is True, uses absolute (dBm) reference. Otherwise relative (dBc).
    segments : int
        Number of segments for Welch segmentation. Data is split by nperseg = data_len // segments.
    signals : PSDSignals
        A PSDSignals instance for communicating results back to the GUI thread.
    """

    def __init__(
        self,
        row: int,
        ch: int,
        I: np.ndarray,
        Q: np.ndarray,
        mode: str,
        dec_stage: int,
        real_units: bool,
        psd_absolute: bool,
        segments: int,
        signals: PSDSignals,
    ):
        super().__init__()
        self.row = row
        self.ch = ch
        self.I = I.copy()
        self.Q = Q.copy()
        self.mode = mode
        self.dec_stage = dec_stage
        self.real_units = real_units
        self.psd_absolute = psd_absolute
        self.segments = segments
        self.signals = signals

    def run(self):
        """
        Perform PSD computations off the main thread. Emit the resulting
        frequency array(s) and PSD(s) via signals.done.
        """
        data_len = len(self.I)
        if data_len < 2:
            self._handle_insufficient_data()
            return

        # Determine reference type
        ref = self._get_reference_type()
        
        # Calculate segment size
        nper = max(1, data_len // max(1, self.segments))

        # Compute spectrum based on mode
        if self.mode == "SSB":
            payload = self._compute_ssb_psd(ref, nper)
        else:  # "DSB"
            payload = self._compute_dsb_psd(ref, nper)

        self.signals.done.emit(self.row, self.mode, self.ch, payload)

    def _handle_insufficient_data(self):
        """Handle the edge case of insufficient data."""
        if self.mode == "SSB":
            payload = ([], [], [], [], [], [], 0.0)
        else:
            payload = ([], [])
        self.signals.done.emit(self.row, self.mode, self.ch, payload)
        
    def _get_reference_type(self) -> str:
        """Determine the reference type for the spectrum computation."""
        if not self.real_units:
            return "counts"
        return "absolute" if self.psd_absolute else "relative"
        
    def _compute_ssb_psd(self, ref, nper):
        """Compute single-sideband PSD."""
        # First compute IQ spectrum
        spec_iq = spectrum_from_slow_tod(
            i_data=self.I,
            q_data=self.Q,
            dec_stage=self.dec_stage,
            scaling="psd",
            reference=ref,
            nperseg=nper,
            spectrum_cutoff=0.9,
        )
        
        # Then compute magnitude spectrum
        M_data = np.sqrt(self.I**2 + self.Q**2)
        spec_m = spectrum_from_slow_tod(
            i_data=M_data,
            q_data=np.zeros_like(M_data),
            dec_stage=self.dec_stage,
            scaling="psd",
            reference=ref,
            nperseg=nper,
            spectrum_cutoff=0.9,
        )
        
        # Return combined results
        return (
            spec_iq["freq_iq"],
            spec_iq["psd_i"],
            spec_iq["psd_q"],
            spec_m["psd_i"],
            spec_m["freq_iq"],
            spec_m["psd_i"],
            float(self.dec_stage),
        )
        
    def _compute_dsb_psd(self, ref, nper):
        """Compute dual-sideband PSD."""
        # Compute IQ spectrum
        spec_iq = spectrum_from_slow_tod(
            i_data=self.I,
            q_data=self.Q,
            dec_stage=self.dec_stage,
            scaling="psd",
            reference=ref,
            nperseg=nper,
            spectrum_cutoff=0.9,
        )
        
        # Extract dual-sideband spectrum and sort by frequency
        freq_dsb = spec_iq["freq_dsb"]
        psd_dsb = spec_iq["psd_dual_sideband"]
        order = np.argsort(freq_dsb)
        
        return (freq_dsb[order], psd_dsb[order])

# ───────────────────────── CRS Initialization Task & Signals ─────────────────────────
class CRSInitializeSignals(QObject):
    """Holds custom signals emitted by CRS initialization tasks."""
    success = pyqtSignal(str)  # success message
    error = pyqtSignal(str)    # error message

# ───────────────────────── Network Analysis ─────────────────────────
class NetworkAnalysisSignals(QObject):
    """
    Holds custom signals emitted by network analysis tasks.
    """
    progress = pyqtSignal(int, float)  # module, progress percentage
    data_update = pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray)  # module, freqs, amps, phases
    data_update_with_amp = pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray, float)  # module, freqs, amps, phases, amplitude
    completed = pyqtSignal(int)  # module
    error = pyqtSignal(str)  # error message

class DACScaleFetcher(QtCore.QThread):
    """Asynchronously fetch DAC scales for all modules."""
    dac_scales_ready = QtCore.pyqtSignal(dict)
    
    def __init__(self, crs):
        super().__init__()
        self.crs = crs
        
    def run(self):
        """Fetch DAC scales for all modules."""
        dac_scales = {}
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            self._fetch_all_dac_scales(loop, dac_scales)
        finally:
            loop.close()
            
        self.dac_scales_ready.emit(dac_scales)
        
    def _fetch_all_dac_scales(self, loop, dac_scales):
        """Fetch DAC scales for all modules using the provided event loop."""
        for module in range(1, 9):
            try:
                dac_scale = loop.run_until_complete(
                    self.crs.get_dac_scale('DBM', module=module)
                )
                dac_scales[module] = dac_scale - 1.5  # Compensation for the balun
            except Exception as e:
                # Silently handle the expected "Can't access module X with low analog banking" error
                if "Can't access module" in str(e) and "analog banking" in str(e):
                    # Just set to None without printing error
                    dac_scales[module] = None
                else:
                    # Still print unexpected ValueErrors
                    print(f"Error fetching DAC scale for module {module}: {e}")
                    dac_scales[module] = None

class NetworkAnalysisTask(QRunnable):
    """
    Off-thread worker for running network analysis.
    
    Parameters
    ----------
    crs : CRS
        CRS object from HardwareMap
    module : int
        Module number to run analysis on
    params : dict
        Network analysis parameters
    signals : NetworkAnalysisSignals
        Signals for communication with GUI thread
    amplitude : float, optional
        Specific amplitude value to use for this task
    """
    
    def __init__(self, crs: "CRS", module: int, params: dict, signals: NetworkAnalysisSignals, amplitude=None):
        super().__init__()
        self.crs = crs
        self.module = module
        self.params = params
        self.signals = signals
        self.amplitude = amplitude if amplitude is not None else params.get('amp', DEFAULT_AMPLITUDE)
        self._running = True
        self._last_update_time = 0
        self._update_interval = NETANAL_UPDATE_INTERVAL  # seconds
        self._task = None
        self._loop = None
        
    def stop(self):
        """
        Signal the task to stop and cancel any running asyncio tasks.
        """
        self._running = False
        
        # Cancel the async task if it exists
        if self._task and not self._task.done() and self._loop:
            # Schedule task cancellation in the event loop
            self._loop.call_soon_threadsafe(self._task.cancel)
            
            # Clean up channels
            if self._loop.is_running():
                try:
                    # Try to schedule cleanup in the event loop
                    cleanup_future = asyncio.run_coroutine_threadsafe(
                        self._cleanup_channels(), self._loop
                    )
                    # Wait for cleanup with a timeout
                    cleanup_future.result(timeout=2.0)
                except (asyncio.CancelledError, concurrent.futures.TimeoutError, Exception):
                    pass
    
    async def _cleanup_channels(self):
        """Clean up channels on the module being analyzed."""
        try:
            async with self.crs.tuber_context() as ctx:
                # Zero out amplitudes for all channels on this module
                for j in range(1, 1024):  # Assuming max channels is 1023
                    ctx.set_amplitude(0, channel=j, module=self.module)
                await ctx()
        except Exception:
            # Ignore errors during cleanup
            pass
    
    def run(self):
        """Execute the network analysis using the take_netanal macro."""
        # Create event loop for async operations
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._execute_network_analysis()
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            # Clean up resources
            self._cleanup_resources()
            
    def _execute_network_analysis(self):
        """Run the network analysis operation."""
        # Create callbacks that emit signals
        progress_cb = self._create_progress_callback()
        data_cb = self._create_data_callback()
        
        # Extract parameters
        params = self._extract_parameters()

        # Clear channels if requested
        if params['clear_channels'] and self._running:
            self._loop.run_until_complete(self.crs.clear_channels(module=self.module))

        # Set cable length before running the analysis
        if self._running:
            self._loop.run_until_complete(
                self.crs.set_cable_length(length=params['cable_length'], module=self.module)
            )

        # Create a task for the netanal operation
        if self._running:
            self._execute_netanal_task(params, progress_cb, data_cb)
            
    def _create_progress_callback(self):
        """Create a callback for progress updates."""
        def progress_cb(module, progress):
            if self._running:
                self.signals.progress.emit(module, progress)
        return progress_cb
        
    def _create_data_callback(self):
        """Create a callback for data updates."""
        def data_cb(module, freqs_raw, amps_raw, phases_raw):
            if self._running:
                # Sort data by frequency before displaying
                sort_idx = np.argsort(freqs_raw)
                freqs = freqs_raw[sort_idx]
                amps = amps_raw[sort_idx]
                phases = phases_raw[sort_idx]
                
                # Throttle updates to prevent GUI lag
                current_time = time.time()
                if current_time - self._last_update_time >= self._update_interval:
                    self._last_update_time = current_time                    
                
                # Emit standard update and amplitude-specific update
                self.signals.data_update.emit(module, freqs, amps, phases)
                self.signals.data_update_with_amp.emit(module, freqs, amps, phases, self.amplitude)
        return data_cb
    
    def _extract_parameters(self):
        """Extract parameters from the params dictionary with defaults."""
        return {
            'fmin': self.params.get('fmin', DEFAULT_MIN_FREQ),
            'fmax': self.params.get('fmax', DEFAULT_MAX_FREQ),
            'nsamps': self.params.get('nsamps', DEFAULT_NSAMPLES),
            'npoints': self.params.get('npoints', DEFAULT_NPOINTS),
            'max_chans': self.params.get('max_chans', DEFAULT_MAX_CHANNELS),
            'max_span': self.params.get('max_span', DEFAULT_MAX_SPAN),
            'cable_length': self.params.get('cable_length', DEFAULT_CABLE_LENGTH),
            'clear_channels': self.params.get('clear_channels', True)
        }
        
    def _execute_netanal_task(self, params, progress_cb, data_cb):
        """Execute the network analysis task and handle results."""
        netanal_coro = self.crs.take_netanal(
            amp=self.amplitude,  # Use the specific amplitude for this task
            fmin=params['fmin'],
            fmax=params['fmax'],
            nsamps=params['nsamps'],
            npoints=params['npoints'],
            max_chans=params['max_chans'],
            max_span=params['max_span'],
            module=self.module,
            progress_callback=progress_cb,
            data_callback=data_cb
        )
        
        # Create and store the task so it can be canceled
        self._task = self._loop.create_task(netanal_coro)
        
        try:
            # Run the task and get the result
            result = self._loop.run_until_complete(self._task)

            if self._running and result:
                # Unpack the dictionary returned by take_netanal
                fs_sorted = result['frequencies']
                iq_sorted = result['iq_complex']
                phase_sorted = result['phase_degrees']
                amp_sorted = np.abs(iq_sorted)

                # Emit final data update
                self.signals.data_update.emit(self.module, fs_sorted, amp_sorted, phase_sorted)
                self.signals.data_update_with_amp.emit(
                    self.module, fs_sorted, amp_sorted, phase_sorted, self.amplitude
                )
                self.signals.completed.emit(self.module)

        except asyncio.CancelledError:
            # Task was canceled, emit error signal
            # No need to print traceback here as it's an expected cancellation
            self.signals.error.emit(f"Analysis canceled for module {self.module}")
            # Make sure to clean up channels
            if self._loop and self._loop.is_running(): # Check if loop is still valid
                self._loop.run_until_complete(self._cleanup_channels())
        except KeyError as ke:
            err_msg = f"KeyError accessing network analysis results for module {self.module}: {ke}. Expected keys 'frequencies', 'iq_complex', 'phase_degrees'."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self.signals.error.emit(err_msg)
        except Exception as e:
            err_msg = f"Error processing network analysis results for module {self.module}: {type(e).__name__}: {e}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr) # This will print the full stack trace
            self.signals.error.emit(err_msg)
            
    def _cleanup_resources(self):
        """Clean up task and event loop resources."""
        if self._task and not self._task.done():
            self._task.cancel()
            
        # Clean up the event loop
        if self._loop and self._loop.is_running():
            self._loop.stop()
        if self._loop:
            self._loop.close()
            self._loop = None

class CRSInitializeTask(QRunnable):
    """
    Off-thread worker for initializing the CRS board.
    """
    def __init__(self, crs: "CRS", module: int, irig_source: Any, clear_channels: bool, signals: CRSInitializeSignals):
        super().__init__()
        self.crs = crs
        self.module = module
        self.irig_source = irig_source
        self.clear_channels = clear_channels
        self.signals = signals
        self._loop = None

    def run(self):
        """Execute the CRS initialization."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._initialize_c_r_s())
            self.signals.success.emit("CRS board initialized successfully.")
        except Exception as e:
            detailed_error = f"Error during CRS initialization: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(detailed_error)
        finally:
            if self._loop and self._loop.is_running():
                self._loop.stop()
            if self._loop:
                self._loop.close()
                self._loop = None

    async def _initialize_c_r_s(self):
        """Perform the asynchronous CRS initialization steps."""
        # It's assumed crs.TIMESTAMP_PORT.BACKPLANE, .TEST, .SMA are accessible
        # and are the actual enum members to be passed.
        await self.crs.set_timestamp_port(self.irig_source)
        
        if self.clear_channels:
            await self.crs.clear_channels(module=self.module)

# ───────────────────────── Network Analysis UI ─────────────────────────

# ───────────────────────── Multisweep Task & Signals ─────────────────────────
class MultisweepSignals(QObject):
    """
    Holds custom signals emitted by multisweep tasks.
    """
    # Emitted by the task's progress_callback wrapper
    progress = pyqtSignal(int, float)  # module, overall_progress_percentage (0-100 for one amp sweep)
    
    # Emitted by the task's data_callback wrapper during a sweep for one amplitude
    intermediate_data_update = pyqtSignal(int, float, dict) # module, amplitude, {cf: {'freqs':..., 'iq':...}}
    
    # Emitted after one amplitude sweep (and optional fitting) is complete for a module
    # The dict contains the final (possibly fitted) results for that amplitude.
    data_update = pyqtSignal(int, float, dict)  # module, amplitude, {cf: result_dict, ...}
    
    # Emitted when one amplitude sweep is fully completed for a module
    completed_amplitude = pyqtSignal(int, float) # module, amplitude
    
    # Emitted when all amplitudes for all modules are done
    all_completed = pyqtSignal()
    
    # Emitted on error
    error = pyqtSignal(int, float, str)  # module, amplitude (or -1 if general error), error_message

class MultisweepTask(QRunnable):
    """
    Off-thread worker for running multisweep analysis.
    """
    def __init__(self, crs: "CRS", resonance_frequencies: list[float], params: dict, signals: MultisweepSignals):
        super().__init__()
        self.crs = crs
        self.resonance_frequencies = resonance_frequencies
        self.params = params # Expected: span_hz, npoints_per_sweep, amps, perform_fits, module, etc.
        self.signals = signals
        self._running = True
        self._loop = None
        self._current_async_task = None
        self.current_amplitude = -1 # For error reporting

    def stop(self):
        self._running = False
        if self._current_async_task and not self._current_async_task.done() and self._loop:
            self._loop.call_soon_threadsafe(self._current_async_task.cancel)

    def _progress_callback_wrapper(self, module, progress_percentage):
        if self._running:
            self.signals.progress.emit(module, progress_percentage)

    def _data_callback_wrapper(self, module, intermediate_results):
        if self._running:
            # intermediate_results is {cf: {'frequencies': ..., 'iq_complex': ...}}
            self.signals.intermediate_data_update.emit(module, self.current_amplitude, intermediate_results)

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        module = self.params.get('module')
        if module is None:
            self.signals.error.emit(-1, -1, "Module not specified in multisweep parameters.")
            return

        try:
            amplitudes = self.params.get('amps', [DEFAULT_AMPLITUDE])
            
            for amp in amplitudes:
                if not self._running:
                    self.signals.error.emit(module, self.current_amplitude, "Multisweep canceled before starting new amplitude.")
                    return
                
                self.current_amplitude = amp

                multisweep_params_for_call = {
                    'center_frequencies': self.resonance_frequencies,
                    'span_hz': self.params['span_hz'],
                    'npoints_per_sweep': self.params['npoints_per_sweep'],
                    'amp': amp,
                    'nsamps': self.params.get('nsamps', 10), # Add nsamps
                    'module': module,
                    'progress_callback': self._progress_callback_wrapper,
                    'data_callback': self._data_callback_wrapper,
                    # Pass other relevant params like global_phase_ref_to_zero if needed
                    'global_phase_ref_to_zero': self.params.get('global_phase_ref_to_zero', True),
                    # Use the value from dialog, defaulting to True if not specified
                    'recalculate_center_frequencies': self.params.get('recalculate_center_frequencies', True),
                }

                multisweep_coro = self.crs.multisweep(**multisweep_params_for_call)
                self._current_async_task = self._loop.create_task(multisweep_coro)
                raw_results = self._loop.run_until_complete(self._current_async_task)

                if not self._running: # Check after await
                    self.signals.error.emit(module, amp, "Multisweep canceled during sweep execution.")
                    return

                processed_results = raw_results
                if raw_results and self.params.get('perform_fits', False):
                    try:
                        processed_results = fitting.process_multisweep_results(
                            raw_results,
                            approx_Q_for_fit=self.params.get('approx_Q_for_fit', 1e4),
                            fit_resonances=True,
                            center_iq_circle=self.params.get('center_iq_circle', True)
                        )
                    except Exception as fit_e:
                        self.signals.error.emit(module, amp, f"Fitting error: {fit_e}")
                        # Fallback to raw results if fitting fails
                        processed_results = raw_results
                
                if processed_results:
                    self.signals.data_update.emit(module, amp, processed_results)
                
                self.signals.completed_amplitude.emit(module, amp)

            self.signals.all_completed.emit()

        except asyncio.CancelledError:
            self.signals.error.emit(module, self.current_amplitude, "Multisweep task was canceled by user.")
        except Exception as e:
            detailed_error = f"Error in MultisweepTask (amp {self.current_amplitude:.4f}): {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(module, self.current_amplitude, detailed_error)
        finally:
            if self._loop:
                if self._loop.is_running():
                    self._loop.stop()
                self._loop.close()
            self._loop = None
            self._current_async_task = None
