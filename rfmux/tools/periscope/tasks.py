"""Worker threads and tasks used by the Periscope viewer."""

from .utils import * # Imports QtCore, QThread, QObject, pyqtSignal, QRunnable,
                     # streamer, np, asyncio, time, socket, queue, traceback, sys,
                     # DEFAULT_AMPLITUDE, NETANAL_UPDATE_INTERVAL, DENSITY_GRID,
                     # SCATTER_POINTS, spectrum_from_slow_tod, pg,
                     # gaussian_filter, convolve, SMOOTH_SIGMA, LOG_COMPRESS,
                     # DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_NSAMPLES,
                     # DEFAULT_NPOINTS, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
                     # DEFAULT_CABLE_LENGTH, concurrent, fitting (from rfmux.algorithms.measurement)

# fitting is already imported via 'from .utils import *' if utils.py imports it from rfmux.algorithms.measurement
# However, to be explicit for this module's direct dependency:
from rfmux.algorithms.measurement import fitting as fitting_module_direct # Alias to avoid conflict if utils also exports 'fitting'

class UDPReceiver(QtCore.QThread):
    """
    Receives multicast packets in a dedicated QThread and pushes them
    into a thread-safe queue.
    """
    def __init__(self, host: str, module: int) -> None:
        super().__init__()
        self.module_id = module
        self.queue = queue.Queue()
        self.sock = streamer.get_multicast_socket(host) # streamer from .utils
        self.sock.settimeout(0.2)

    def run(self):
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
        self.requestInterruption()
        try:
            self.sock.close()
        except OSError:
            pass

class IQSignals(QObject):
    done = pyqtSignal(int, str, object)

class IQTask(QRunnable):
    def __init__(self, row, ch, I, Q, dot_px, mode, signals: IQSignals):
        super().__init__()
        self.row = row; self.ch = ch; self.I = I.copy(); self.Q = Q.copy()
        self.dot_px = dot_px; self.mode = mode; self.signals = signals

    def run(self):
        if len(self.I) < 2: self._handle_insufficient_data(); return
        payload = self._compute_density() if self.mode == "density" else self._compute_scatter()
        self.signals.done.emit(self.row, self.mode, payload)
        
    def _handle_insufficient_data(self):
        # DENSITY_GRID from .utils
        empty_payload = (np.zeros((DENSITY_GRID, DENSITY_GRID), np.uint8), (0,1,0,1)) if self.mode == "density" else ([],[],[])
        self.signals.done.emit(self.row, self.mode, empty_payload)
        
    def _compute_density(self):
        # DENSITY_GRID, gaussian_filter, SMOOTH_SIGMA, LOG_COMPRESS, convolve from .utils
        g = DENSITY_GRID; hist = np.zeros((g, g), np.uint32)
        Imin, Imax = self.I.min(), self.I.max(); Qmin, Qmax = self.Q.min(), self.Q.max()
        if Imin == Imax or Qmin == Qmax: return (hist.astype(np.uint8), (Imin, Imax, Qmin, Qmax))
        ix = ((self.I - Imin) * (g - 1) / (Imax - Imin)).astype(np.intp)
        qy = ((self.Q - Qmin) * (g - 1) / (Qmax - Qmin)).astype(np.intp)
        np.add.at(hist, (qy, ix), 1)
        if self.dot_px > 1: self._apply_dot_dilation(hist, ix, qy, g)
        if gaussian_filter is not None and SMOOTH_SIGMA > 0:
            hist = gaussian_filter(hist.astype(np.float32), SMOOTH_SIGMA, mode="nearest")
        if LOG_COMPRESS: hist = np.log1p(hist, out=hist.astype(np.float32))
        if hist.max() > 0: hist = (hist * (255.0 / hist.max())).astype(np.uint8)
        return (hist, (Imin, Imax, Qmin, Qmax))
        
    def _apply_dot_dilation(self, hist, ix, qy, g):
        # convolve from .utils
        r = self.dot_px // 2
        if convolve is not None:
            k = 2 * r + 1; kernel = np.ones((k, k), dtype=np.uint8)
            hist[:] = convolve(hist, kernel, mode="constant", cval=0) # Update hist in place
        else:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ys, xs = qy + dy, ix + dx
                    mask = ((0 <= ys) & (ys < g) & (0 <= xs) & (xs < g))
                    np.add.at(hist, (ys[mask], xs[mask]), 1)
                    
    def _compute_scatter(self):
        # SCATTER_POINTS, pg from .utils
        N = len(self.I)
        idx = np.linspace(0, N - 1, SCATTER_POINTS, dtype=np.intp) if N > SCATTER_POINTS else np.arange(N, dtype=np.intp)
        xs, ys = self.I[idx], self.Q[idx]
        rel = idx / (idx.max() if idx.size else 1)
        colors = pg.colormap.get("turbo").map(rel.astype(np.float32), mode="byte")
        return (xs, ys, colors)

class PSDSignals(QObject):
    done = pyqtSignal(int, str, int, object)

class PSDTask(QRunnable):
    def __init__(self, row: int, ch: int, I: np.ndarray, Q: np.ndarray, mode: str, dec_stage: int,
                 real_units: bool, psd_absolute: bool, segments: int, signals: PSDSignals):
        super().__init__()
        self.row, self.ch, self.I, self.Q, self.mode = row, ch, I.copy(), Q.copy(), mode
        self.dec_stage, self.real_units, self.psd_absolute = dec_stage, real_units, psd_absolute
        self.segments, self.signals = segments, signals

    def run(self):
        data_len = len(self.I)
        if data_len < 2: self._handle_insufficient_data(); return
        ref = "counts" if not self.real_units else ("absolute" if self.psd_absolute else "relative")
        nper = max(1, data_len // max(1, self.segments))
        # spectrum_from_slow_tod from .utils
        payload = self._compute_ssb_psd(ref, nper) if self.mode == "SSB" else self._compute_dsb_psd(ref, nper)
        self.signals.done.emit(self.row, self.mode, self.ch, payload)

    def _handle_insufficient_data(self):
        payload = ([], [], [], [], [], [], 0.0) if self.mode == "SSB" else ([], [])
        self.signals.done.emit(self.row, self.mode, self.ch, payload)
        
    def _compute_ssb_psd(self, ref, nper):
        # spectrum_from_slow_tod from .utils
        spec_iq = spectrum_from_slow_tod(i_data=self.I, q_data=self.Q, dec_stage=self.dec_stage,
                                         scaling="psd", reference=ref, nperseg=nper, spectrum_cutoff=0.9)
        M_data = np.sqrt(self.I**2 + self.Q**2)
        spec_m = spectrum_from_slow_tod(i_data=M_data, q_data=np.zeros_like(M_data), dec_stage=self.dec_stage,
                                        scaling="psd", reference=ref, nperseg=nper, spectrum_cutoff=0.9)
        return (spec_iq["freq_iq"], spec_iq["psd_i"], spec_iq["psd_q"], spec_m["psd_i"],
                spec_m["freq_iq"], spec_m["psd_i"], float(self.dec_stage))
        
    def _compute_dsb_psd(self, ref, nper):
        # spectrum_from_slow_tod from .utils
        spec_iq = spectrum_from_slow_tod(i_data=self.I, q_data=self.Q, dec_stage=self.dec_stage,
                                         scaling="psd", reference=ref, nperseg=nper, spectrum_cutoff=0.9)
        freq_dsb, psd_dsb = spec_iq["freq_dsb"], spec_iq["psd_dual_sideband"]
        order = np.argsort(freq_dsb)
        return (freq_dsb[order], psd_dsb[order])

class CRSInitializeSignals(QObject):
    success = pyqtSignal(str); error = pyqtSignal(str)

class NetworkAnalysisSignals(QObject):
    progress = pyqtSignal(int, float)
    data_update = pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray)
    data_update_with_amp = pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray, float)
    completed = pyqtSignal(int); error = pyqtSignal(str)

class DACScaleFetcher(QtCore.QThread):
    dac_scales_ready = QtCore.pyqtSignal(dict)
    def __init__(self, crs): super().__init__(); self.crs = crs
    def run(self):
        dac_scales = {}; loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        try: self._fetch_all_dac_scales(loop, dac_scales)
        finally: loop.close()
        self.dac_scales_ready.emit(dac_scales)
    def _fetch_all_dac_scales(self, loop, dac_scales):
        for module_idx in range(1, 9): # Renamed module
            try:
                dac_scale = loop.run_until_complete(self.crs.get_dac_scale('DBM', module=module_idx))
                dac_scales[module_idx] = dac_scale - 1.5
            except Exception as e:
                if "Can't access module" in str(e) and "analog banking" in str(e):
                    dac_scales[module_idx] = None
                else:
                    print(f"Error fetching DAC scale for module {module_idx}: {e}", file=sys.stderr) # Print to stderr
                    dac_scales[module_idx] = None

class NetworkAnalysisTask(QtCore.QThread):
    """QThread subclass for performing network analysis operations without blocking the GUI."""
    def __init__(self, crs: "CRS", module: int, params: dict, signals: NetworkAnalysisSignals, amplitude=None):
        super().__init__()
        self.crs, self.module, self.params, self.signals = crs, module, params, signals
        # DEFAULT_AMPLITUDE, NETANAL_UPDATE_INTERVAL from .utils
        self.amplitude = amplitude if amplitude is not None else params.get('amp', DEFAULT_AMPLITUDE)
        self._running, self._last_update_time = True, 0
        self._update_interval = NETANAL_UPDATE_INTERVAL
        self._task, self._loop = None, None
        
    def stop(self):
        """Stop the network analysis task and cancel any ongoing async operation."""
        self._running = False
        self.requestInterruption()
    
    async def _cleanup_channels(self):
        try:
            # Direct approach to set amplitudes to zero without using async with
            for j in range(1, 1024):
                await self.crs.set_amplitude(0, channel=j, module=self.module)
        except Exception as e:
            print(f"Error in _cleanup_channels: {e}", file=sys.stderr)
            pass
    
    def run(self):
        """QThread entry point - runs in a separate thread."""
        # Create asyncio loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            progress_cb, data_cb = self._create_progress_callback(), self._create_data_callback()
            task_params = self._extract_parameters()
            
            # Setup phase: Clear channels and set cable length
            if task_params['clear_channels'] and not self.isInterruptionRequested():
                loop.run_until_complete(self.crs.clear_channels(module=self.module))
                
            if not self.isInterruptionRequested():
                loop.run_until_complete(self.crs.set_cable_length(length=task_params['cable_length'], module=self.module))
            
            # Main analysis phase
            if not self.isInterruptionRequested():
                # Combine parameters for the take_netanal call
                netanal_params = {
                    'amp': self.amplitude,
                    'fmin': task_params['fmin'],
                    'fmax': task_params['fmax'],
                    'nsamps': task_params['nsamps'],
                    'npoints': task_params['npoints'],
                    'max_chans': task_params['max_chans'],
                    'max_span': task_params['max_span'],
                    'module': self.module,
                    'progress_callback': progress_cb,
                    'data_callback': data_cb
                }
                
                # Process the network analysis asynchronously without blocking
                result = loop.run_until_complete(self._process_network_analysis(loop, netanal_params))
                
                # Process results if available and task wasn't interrupted
                if not self.isInterruptionRequested() and result:
                    fs_sorted, iq_sorted = result['frequencies'], result['iq_complex']
                    phase_sorted, amp_sorted = result['phase_degrees'], np.abs(iq_sorted)
                    self.signals.data_update.emit(self.module, fs_sorted, amp_sorted, phase_sorted)
                    self.signals.data_update_with_amp.emit(self.module, fs_sorted, amp_sorted, phase_sorted, self.amplitude)
                    self.signals.completed.emit(self.module)
            
        except asyncio.CancelledError:
            self.signals.error.emit(f"Analysis canceled for module {self.module}")
            if loop.is_running():
                loop.run_until_complete(self._cleanup_channels())
        except KeyError as ke:
            err_msg = f"KeyError accessing results for module {self.module}: {ke}. Expected 'frequencies', 'iq_complex', 'phase_degrees'."
            print(f"ERROR: {err_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self.signals.error.emit(err_msg)
        except Exception as e:
            err_msg = f"Error processing results for module {self.module}: {type(e).__name__}: {e}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self.signals.error.emit(err_msg)
        finally:
            if loop.is_running():
                loop.stop()
            loop.close()
            
    def _create_progress_callback(self):
        return lambda module_idx, prog: self.signals.progress.emit(module_idx, prog) if self._running else None # Renamed module, progress
        
    def _create_data_callback(self):
        def data_cb(module_idx, freqs_raw, amps_raw, phases_raw): # Renamed module
            if self._running:
                sort_idx = np.argsort(freqs_raw)
                freqs, amps, phases = freqs_raw[sort_idx], amps_raw[sort_idx], phases_raw[sort_idx]
                current_time = time.time()
                if current_time - self._last_update_time >= self._update_interval:
                    self._last_update_time = current_time                    
                self.signals.data_update.emit(module_idx, freqs, amps, phases)
                self.signals.data_update_with_amp.emit(module_idx, freqs, amps, phases, self.amplitude)
        return data_cb
    
    def _extract_parameters(self):
        # Constants from .utils
        return {'fmin': self.params.get('fmin', DEFAULT_MIN_FREQ), 'fmax': self.params.get('fmax', DEFAULT_MAX_FREQ),
                'nsamps': self.params.get('nsamps', DEFAULT_NSAMPLES), 'npoints': self.params.get('npoints', DEFAULT_NPOINTS),
                'max_chans': self.params.get('max_chans', DEFAULT_MAX_CHANNELS), 'max_span': self.params.get('max_span', DEFAULT_MAX_SPAN),
                'cable_length': self.params.get('cable_length', DEFAULT_CABLE_LENGTH), 'clear_channels': self.params.get('clear_channels', True)}
        
    async def _process_network_analysis(self, loop, netanal_params):
        """Process a single network analysis operation asynchronously.
        
        This method periodically yields control back to the event loop to keep the GUI responsive.
        """
        netanal_coro = self.crs.take_netanal(**netanal_params)
        task = loop.create_task(netanal_coro)
        
        # Check for interruption while the task is running
        while not task.done():
            if self.isInterruptionRequested():
                task.cancel()
                await asyncio.sleep(0.01)  # Give the cancellation a chance to process
                return None
            await asyncio.sleep(0.1)  # Short sleep to yield control back to the event loop - this is crucial for preventing GUI freezing
        
        # Get the result when the task is done
        if not task.cancelled():
            try:
                return await task
            except Exception as e:
                print(f"Error in _process_network_analysis: {e}", file=sys.stderr)
                raise
        return None

class CRSInitializeTask(QRunnable):
    def __init__(self, crs: "CRS", module: int, irig_source: Any, clear_channels: bool, signals: CRSInitializeSignals):
        super().__init__(); self.crs, self.module, self.irig_source = crs, module, irig_source
        self.clear_channels, self.signals, self._loop = clear_channels, signals, None
    def run(self):
        # traceback from .utils
        self._loop = asyncio.new_event_loop(); asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._initialize_c_r_s())
            self.signals.success.emit("CRS board initialized successfully.")
        except Exception as e:
            self.signals.error.emit(f"Error during CRS initialization: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        finally:
            if self._loop:
                if self._loop.is_running(): self._loop.stop()
                self._loop.close(); self._loop = None
    async def _initialize_c_r_s(self):
        await self.crs.set_timestamp_port(self.irig_source)
        if self.clear_channels: await self.crs.clear_channels(module=self.module)

class SetCableLengthSignals(QObject):
    """Signals for SetCableLengthTask."""
    success = pyqtSignal(int, float)  # module_id, length_set
    error = pyqtSignal(int, str)    # module_id, error_message

class SetCableLengthTask(QRunnable):
    """A QRunnable task to set the cable length on the CRS asynchronously."""
    def __init__(self, crs: "CRS", module_id: int, length: float, signals: SetCableLengthSignals):
        super().__init__()
        self.crs = crs
        self.module_id = module_id
        self.length = length
        self.signals = signals
        self._loop = None

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self.crs.set_cable_length(length=self.length, module=self.module_id))
            self.signals.success.emit(self.module_id, self.length)
        except Exception as e:
            # traceback, sys are imported from .utils
            err_msg = f"Error setting cable length for module {self.module_id} to {self.length}m: {type(e).__name__}: {str(e)}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self.signals.error.emit(self.module_id, err_msg)
        finally:
            if self._loop:
                if self._loop.is_running():
                    self._loop.stop()
                self._loop.close()
            self._loop = None

class MultisweepSignals(QObject):
    progress = pyqtSignal(int, float)
    # Updated data_update to include iteration and direction:
    # 1. results_for_plotting: {output_cf: data_dict_val} - original structure from crs.multisweep
    # 2. results_for_history: {conceptual_idx: output_cf_key} - for easy history update
    data_update = pyqtSignal(int, int, float, str, dict, dict) # module, iteration, amplitude, direction, results_for_plotting, results_for_history
    completed_iteration = pyqtSignal(int, int, float, str) # module, iteration, amplitude, direction
    starting_iteration = pyqtSignal(int, int, float, str) # module, iteration, amplitude, direction
    all_completed = pyqtSignal()
    error = pyqtSignal(int, float, str)

class MultisweepTask(QtCore.QThread):
    """QThread subclass for performing multisweep operations without blocking the GUI."""
    def __init__(self, crs: "CRS", params: dict, signals: MultisweepSignals, window: Any):
        """
        Initialize the MultisweepTask.
        
        Args:
            crs: Control and Readout System object
            params: Dictionary of parameters for the multisweep
            signals: Signal object for communication with GUI
            window: MultisweepWindow instance that will display the results
        """
        super().__init__()
        self.crs = crs
        self.params = params
        self.signals = signals
        self.window = window # Store the MultisweepWindow instance
        # Baseline frequencies are from the params passed by the window, reflecting its current configuration
        self.baseline_resonance_frequencies = list(params.get('resonance_frequencies', []))
        self._running = True
        self.current_amplitude = -1
        self.current_iteration = -1
        self.current_direction = ""
        self._task_completed = asyncio.Event()

    def stop(self):
        """Stop the multisweep task and cancel any ongoing async operation."""
        self._running = False
        self.requestInterruption()

    def _progress_callback_wrapper(self, module_idx, progress_percentage):
        """Wrapper for progress callback to emit signals."""
        if self._running: self.signals.progress.emit(module_idx, progress_percentage)

    def run(self):
        """QThread entry point - runs in a separate thread."""
        # Create asyncio loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        module_idx = self.params.get('module')
        if module_idx is None:
            self.signals.error.emit(-1, -1, "Module not specified in multisweep parameters.")
            return
        
        try:
            amplitudes = self.params.get('amps', [DEFAULT_AMPLITUDE])
            sweep_direction = self.params.get('sweep_direction', 'upward')
            conceptual_frequencies_from_window = self.window.conceptual_resonance_frequencies
            
            # Track iterations across all sweeps
            iteration_index = 0

            for amp_val in amplitudes:
                if self.isInterruptionRequested():
                    self.signals.error.emit(module_idx, self.current_amplitude, "Multisweep canceled.")
                    return
                
                self.current_amplitude = amp_val

                # Dynamically determine input CFs for this specific amplitude
                current_sweep_cfs_for_this_amp = []
                # To map results back: conceptual_idx -> input_cf_chosen_for_this_amp
                conceptual_idx_to_input_cf_map = {}

                if not conceptual_frequencies_from_window: # Should not happen if window is configured
                    self.signals.error.emit(module_idx, amp_val, "Conceptual frequencies not available from window.")
                    return

                for idx, conceptual_cf in enumerate(conceptual_frequencies_from_window):
                    remembered_cf = self.window._get_closest_remembered_cf(idx, amp_val)
                    chosen_input_cf = remembered_cf if remembered_cf is not None else self.baseline_resonance_frequencies[idx]
                    current_sweep_cfs_for_this_amp.append(chosen_input_cf)
                    conceptual_idx_to_input_cf_map[idx] = chosen_input_cf
                
                # Determine which directions to sweep
                directions_to_sweep = []
                if sweep_direction == "both":
                    directions_to_sweep = ["upward","downward"]  # Perform upward sweep first, then downward
                else:
                    directions_to_sweep = [sweep_direction]
                
                # Perform sweep(s) for each direction
                for direction in directions_to_sweep:
                    # Update current iteration and direction
                    self.current_iteration = iteration_index
                    self.current_direction = direction
                    
                    # Signal the start of this iteration to update the status bar BEFORE we start the sweep
                    self.signals.starting_iteration.emit(module_idx, iteration_index, amp_val, direction)
                    
                    multisweep_params = {
                        'center_frequencies': current_sweep_cfs_for_this_amp, # Use dynamically determined CFs
                        'span_hz': self.params['span_hz'],
                        'npoints_per_sweep': self.params['npoints_per_sweep'],
                        'amp': amp_val,
                        'nsamps': self.params.get('nsamps', 10),
                        'module': module_idx,
                        'progress_callback': self._progress_callback_wrapper,
                        'recalculate_center_frequencies': self.params.get('recalculate_center_frequencies', None),
                        'sweep_direction': direction
                    }
                    
                    # Process the multisweep asynchronously without blocking
                    raw_results_from_crs = loop.run_until_complete(self._process_multisweep(loop, multisweep_params))
                    
                    if self.isInterruptionRequested():
                        self.signals.error.emit(module_idx, amp_val, "Multisweep canceled during execution.")
                        return
                    
                    results_for_plotting = raw_results_from_crs # For plotting, use the direct structure
                    results_for_history = {} # For history: {conceptual_idx: output_cf_key}

                    if raw_results_from_crs:
                        # Map results back to conceptual_idx for history
                        for output_cf_key, data_dict_val in raw_results_from_crs.items():
                            input_cf_this_result_was_for = data_dict_val['original_center_frequency']
                            found_conceptual_idx = -1
                            for c_idx, mapped_input_cf in conceptual_idx_to_input_cf_map.items():
                                if abs(mapped_input_cf - input_cf_this_result_was_for) < 1e-3: # Tolerance for float comparison
                                    found_conceptual_idx = c_idx
                                    break
                            if found_conceptual_idx != -1:
                                results_for_history[found_conceptual_idx] = output_cf_key
                            else:
                                print(f"Warning (MultisweepTask): Could not map result for input_cf {input_cf_this_result_was_for} back to conceptual_idx for amp {amp_val}", file=sys.stderr)
                    
                    if results_for_plotting is not None: # Can be None if crs.multisweep had issues
                        # Use updated signal with iteration and direction information
                        self.signals.data_update.emit(module_idx, iteration_index, amp_val, direction, results_for_plotting, results_for_history)
                    
                    # Signal completion of this iteration
                    self.signals.completed_iteration.emit(module_idx, iteration_index, amp_val, direction)
                    
                    # Increment iteration counter for the next sweep
                    iteration_index += 1
            
            self.signals.all_completed.emit()
        except asyncio.CancelledError:
            self.signals.error.emit(module_idx, self.current_amplitude, "Multisweep task canceled by user.")
        except Exception as e:
            detailed_error = f"Error in MultisweepTask (amp {self.current_amplitude:.4f}): {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(module_idx, self.current_amplitude, detailed_error)
        finally:
            if loop.is_running():
                loop.stop()
            loop.close()

    async def _process_multisweep(self, loop, multisweep_params):
        """Process a single multisweep operation asynchronously.
        
        This method periodically yields control back to the event loop to keep the GUI responsive.
        """
        self._task_completed.clear()
        multisweep_coro = self.crs.multisweep(**multisweep_params)
        task = loop.create_task(multisweep_coro)
        
        # Check for interruption while the task is running
        while not task.done():
            if self.isInterruptionRequested():
                task.cancel()
                await asyncio.sleep(0.01)  # Give the cancellation a chance to process
                return None
            await asyncio.sleep(0.1)  # Short sleep to yield control back to the event loop
        
        # Get the result when the task is done
        if not task.cancelled():
            try:
                return await task
            except Exception as e:
                print(f"Error in _process_multisweep: {e}", file=sys.stderr)
                raise
        return None
