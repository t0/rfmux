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
from rfmux.algorithms.measurement import fitting_nonlinear # Import nonlinear fitting module

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
                if dac_scale is not None:
                    dac_scales[module_idx] = dac_scale - 1.5
                else:
                    dac_scales[module_idx] = None
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
    fitting_progress = pyqtSignal(int, str) # module, status_message
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
        self.window = window 
        self.baseline_resonance_frequencies = list(params.get('resonance_frequencies', []))
        self._running = True
        self.current_amplitude = -1
        self.current_iteration = -1
        self.current_direction = ""
        self._task_completed = asyncio.Event()

    def stop(self):
        self._running = False
        self.requestInterruption()

    def _progress_callback_wrapper(self, module_idx, progress_percentage):
        if self._running: self.signals.progress.emit(module_idx, progress_percentage)

    def _apply_fitting_analysis(self, raw_results_from_crs):
        if not raw_results_from_crs:
            return raw_results_from_crs
        
        # Initialize enhanced_results as a copy to preserve raw data if fitting is skipped or fails
        enhanced_results = {k: v.copy() for k, v in raw_results_from_crs.items()}

        apply_skewed = self.params.get('apply_skewed_fit', False)
        apply_nonlinear = self.params.get('apply_nonlinear_fit', False)

        if not apply_skewed and not apply_nonlinear:
            # If no fitting is requested, add flags indicating this
            for res_idx in enhanced_results:
                enhanced_results[res_idx]['skewed_fit_applied'] = False
                enhanced_results[res_idx]['skewed_fit_success'] = False
                enhanced_results[res_idx]['nonlinear_fit_applied'] = False
                enhanced_results[res_idx]['nonlinear_fit_success'] = False
            return enhanced_results

    #print(f"Applying fitting analysis to {len(raw_results_from_crs)} resonances...")
        
        # Temporary dictionary to hold results from skewed fit if it's applied
        skewed_fit_results_temp = enhanced_results 

        if apply_skewed:
            # Signal start of skewed fitting
            module_idx = self.params.get('module')
            if module_idx is not None and self._running:
                self.signals.fitting_progress.emit(module_idx, "Fitting in progress: Applying skewed fits")
            try:
                skewed_fit_results_temp = fitting_module_direct.fit_skewed_multisweep(
                    enhanced_results, # Start with a copy of raw or previously enhanced data
                    approx_Q_for_fit=1e4,
                    fit_resonances=True,
                    center_iq_circle=True,
                    normalize_fit=True
                )
                #print(f"Skewed fitting completed.")
                # Mark skewed fit as applied and check success for each resonance
                for res_idx in enhanced_results: # Iterate original keys to update
                    if res_idx in skewed_fit_results_temp:
                        enhanced_results[res_idx].update(skewed_fit_results_temp[res_idx])
                        enhanced_results[res_idx]['skewed_fit_applied'] = True
                        fit_p = enhanced_results[res_idx].get('fit_params', {})
                        enhanced_results[res_idx]['skewed_fit_success'] = fit_p.get('fr') is not None and fit_p.get('fr') != 'nan'
                        
                        # Generate skewed model magnitude if fit was successful
                        if enhanced_results[res_idx]['skewed_fit_success'] and fit_p:
                            frequencies = enhanced_results[res_idx].get('frequencies')
                            if frequencies is not None:
                                try:
                                    # Generate magnitude model using fitted parameters
                                    skewed_model_mag = fitting_module_direct.s21_skewed(
                                        frequencies, 
                                        fit_p['fr'], 
                                        fit_p['Qr'], 
                                        fit_p['Qcre'], 
                                        fit_p['Qcim'], 
                                        fit_p['A']
                                    )
                                    enhanced_results[res_idx]['skewed_model_mag'] = skewed_model_mag
                                except Exception as e:
                                    print(f"Warning: Failed to generate skewed model for resonance {res_idx}: {e}", file=sys.stderr)
                    else: # Should not happen if fit_skewed_multisweep returns all keys
                        enhanced_results[res_idx]['skewed_fit_applied'] = True
                        enhanced_results[res_idx]['skewed_fit_success'] = False
            except Exception as e:
                print(f"Warning: Skewed fitting failed: {e}", file=sys.stderr)
                for res_idx in enhanced_results:
                    enhanced_results[res_idx]['skewed_fit_applied'] = True
                    enhanced_results[res_idx]['skewed_fit_success'] = False
        else: # Skewed fit not applied
            for res_idx in enhanced_results:
                enhanced_results[res_idx]['skewed_fit_applied'] = False
                enhanced_results[res_idx]['skewed_fit_success'] = False
        
        # Now, `enhanced_results` contains results from skewed fitting (if applied) or is still a copy of raw.
        # `skewed_fit_results_temp` is not needed beyond this point if we update `enhanced_results` in place.

        if apply_nonlinear:
            # Signal start of nonlinear fitting
            module_idx = self.params.get('module')
            if module_idx is not None and self._running:
                self.signals.fitting_progress.emit(module_idx, "Fitting in progress: Applying non-linear fits")
            try:
                # Pass the current state of enhanced_results (which may include skewed fit data)
                nonlinear_fit_output = fitting_nonlinear.fit_nonlinear_iq_multisweep(
                    enhanced_results.copy(), # Pass a copy to avoid modifying during iteration if fit_nonlinear_iq_multisweep does
                    fit_nonlinearity=True,
                    n_extrema_points=5,
                    verbose=False
                )
                #print(f"Nonlinear fitting completed.")
                # Update enhanced_results with nonlinear fit data
                for res_idx in enhanced_results:
                    if res_idx in nonlinear_fit_output:
                        enhanced_results[res_idx].update(nonlinear_fit_output[res_idx])
                        enhanced_results[res_idx]['nonlinear_fit_applied'] = True
                        # 'nonlinear_fit_success' should be set by fit_nonlinear_iq_multisweep
                        if 'nonlinear_fit_success' not in enhanced_results[res_idx]:
                             enhanced_results[res_idx]['nonlinear_fit_success'] = False # Default if not set
                        
                        # Generate nonlinear model IQ if fit was successful
                        if enhanced_results[res_idx].get('nonlinear_fit_success', False):
                            nl_params = enhanced_results[res_idx].get('nonlinear_fit_params', {})
                            frequencies = enhanced_results[res_idx].get('frequencies')
                            if nl_params and frequencies is not None:
                                try:
                                    # Generate complex IQ model using fitted parameters
                                    nonlinear_model_iq = fitting_nonlinear.nonlinear_iq(
                                        frequencies,
                                        nl_params['fr'],
                                        nl_params['Qr'],
                                        nl_params['amp'],
                                        nl_params['phi'],
                                        nl_params['a'],
                                        nl_params['i0'],
                                        nl_params['q0']
                                    )
                                    enhanced_results[res_idx]['nonlinear_model_iq'] = nonlinear_model_iq
                                except Exception as e:
                                    print(f"Warning: Failed to generate nonlinear model for resonance {res_idx}: {e}", file=sys.stderr)
                    else: # Should not happen
                        enhanced_results[res_idx]['nonlinear_fit_applied'] = True
                        enhanced_results[res_idx]['nonlinear_fit_success'] = False
            except Exception as e:
                print(f"Warning: Nonlinear fitting failed: {e}", file=sys.stderr)
                for res_idx in enhanced_results:
                    enhanced_results[res_idx]['nonlinear_fit_applied'] = True
                    enhanced_results[res_idx]['nonlinear_fit_success'] = False
        else: # Nonlinear fit not applied
             for res_idx in enhanced_results:
                enhanced_results[res_idx]['nonlinear_fit_applied'] = False
                enhanced_results[res_idx]['nonlinear_fit_success'] = False
        
        # Final summary
        skewed_applied_count = sum(1 for r in enhanced_results.values() if r.get('skewed_fit_applied'))
        skewed_success_count = sum(1 for r in enhanced_results.values() if r.get('skewed_fit_success'))
        nonlinear_applied_count = sum(1 for r in enhanced_results.values() if r.get('nonlinear_fit_applied'))
        nonlinear_success_count = sum(1 for r in enhanced_results.values() if r.get('nonlinear_fit_success'))

        # print(f"Fitting summary: Skewed: {skewed_success_count}/{skewed_applied_count if apply_skewed else len(enhanced_results)} successful. "
        #       f"Nonlinear: {nonlinear_success_count}/{nonlinear_applied_count if apply_nonlinear else len(enhanced_results)} successful.")
        
        # Signal completion of fitting
        module_idx = self.params.get('module')
        if module_idx is not None and self._running:
            self.signals.fitting_progress.emit(module_idx, "Fitting Completed")
            
        return enhanced_results
            
        # except Exception as e: # This outer try-except might be redundant now
        #     print(f"Error in _apply_fitting_analysis: {e}", file=sys.stderr)
        #     traceback.print_exc(file=sys.stderr)
        #     # Ensure flags are set even on major error
        #     for res_idx in raw_results_from_crs: # Iterate original keys
        #         if res_idx not in enhanced_results: # If it wasn't even copied
        #             enhanced_results[res_idx] = raw_results_from_crs[res_idx].copy()
        #         enhanced_results[res_idx]['skewed_fit_applied'] = apply_skewed
        #         enhanced_results[res_idx]['skewed_fit_success'] = False
        #         enhanced_results[res_idx]['nonlinear_fit_applied'] = apply_nonlinear
        #         enhanced_results[res_idx]['nonlinear_fit_success'] = False
        #     return enhanced_results


    def run(self):
        """QThread entry point - runs in a separate thread."""
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
            iteration_index = 0

            for amp_val in amplitudes:
                if self.isInterruptionRequested():
                    self.signals.error.emit(module_idx, self.current_amplitude, "Multisweep canceled.")
                    return
                
                self.current_amplitude = amp_val
                current_sweep_cfs_for_this_amp = []
                # conceptual_idx_to_input_cf_map = {} # Not strictly needed if results are always index-keyed

                if not conceptual_frequencies_from_window:
                    self.signals.error.emit(module_idx, amp_val, "Conceptual frequencies not available from window.")
                    return

                for idx, conceptual_cf in enumerate(conceptual_frequencies_from_window):
                    remembered_cf = self.window._get_closest_remembered_cf(idx, amp_val)
                    chosen_input_cf = remembered_cf if remembered_cf is not None else self.baseline_resonance_frequencies[idx]
                    current_sweep_cfs_for_this_amp.append(chosen_input_cf)
                    # conceptual_idx_to_input_cf_map[idx] = chosen_input_cf
                
                directions_to_sweep = ["upward","downward"] if sweep_direction == "both" else [sweep_direction]
                
                for direction_val in directions_to_sweep: # Renamed 'direction' to 'direction_val' to avoid conflict
                    self.current_iteration = iteration_index
                    self.current_direction = direction_val
                    
                    self.signals.starting_iteration.emit(module_idx, iteration_index, amp_val, direction_val)
                    
                    multisweep_params = {
                        'center_frequencies': current_sweep_cfs_for_this_amp,
                        'span_hz': self.params['span_hz'],
                        'npoints_per_sweep': self.params['npoints_per_sweep'],
                        'amp': amp_val,
                        'nsamps': self.params.get('nsamps', 10),
                        'module': module_idx,
                        'progress_callback': self._progress_callback_wrapper,
                        'recalculate_center_frequencies': self.params.get('recalculate_center_frequencies', None),
                        'sweep_direction': direction_val
                    }
                    
                    raw_results_from_crs = loop.run_until_complete(self._process_multisweep(loop, multisweep_params))
                    
                    if self.isInterruptionRequested():
                        self.signals.error.emit(module_idx, amp_val, "Multisweep canceled during execution.")
                        return
                    
                    # Process bifurcation detection first
                    if raw_results_from_crs:
                        for res_key, data_dict_val in raw_results_from_crs.items():
                            if not isinstance(res_key, (int, np.integer)): continue
                            iq_data = data_dict_val.get('iq_complex')
                            if iq_data is not None:
                                try:
                                    data_dict_val['is_bifurcated'] = fitting_module_direct.identify_bifurcation(iq_data)
                                except Exception as e:
                                    print(f"Warning: Bifurcation detection failed for index {res_key}: {e}", file=sys.stderr)
                                    data_dict_val['is_bifurcated'] = False
                            else:
                                data_dict_val['is_bifurcated'] = False
                    
                    # Now apply fitting analysis using the (potentially) bifurcation-annotated data
                    # The self.params for apply_skewed_fit etc. are passed via the __init__ of MultisweepTask
                    enhanced_results = self._apply_fitting_analysis(raw_results_from_crs) 
                    results_for_plotting = enhanced_results 
                    
                    results_for_history = {}
                    if enhanced_results: 
                        for res_idx, data_dict_val in enhanced_results.items():
                            if isinstance(res_idx, (int, np.integer)):
                                bias_freq = data_dict_val.get('bias_frequency', data_dict_val.get('original_center_frequency'))
                                if bias_freq is not None: results_for_history[res_idx] = bias_freq
                                else: print(f"Warning: No bias frequency found for index {res_idx}", file=sys.stderr)
                            else: print(f"Warning (MultisweepTask): Non-integer key {res_idx} in results", file=sys.stderr)
                    
                    if results_for_plotting is not None:
                        self.signals.data_update.emit(module_idx, iteration_index, amp_val, direction_val, results_for_plotting, results_for_history)
                    
                    self.signals.completed_iteration.emit(module_idx, iteration_index, amp_val, direction_val)
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
        self._task_completed.clear()
        multisweep_coro = self.crs.multisweep(**multisweep_params)
        task = loop.create_task(multisweep_coro)
        
        while not task.done():
            if self.isInterruptionRequested():
                task.cancel()
                await asyncio.sleep(0.01) 
                return None
            await asyncio.sleep(0.1) 
        
        if not task.cancelled():
            try:
                return await task
            except Exception as e:
                print(f"Error in _process_multisweep: {e}", file=sys.stderr)
                raise
        return None
