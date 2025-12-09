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
from rfmux.core.transferfunctions import exp_bin_noise_data # Import exponential binning function

# Additional imports for async fitting with ThreadPoolExecutor
import os
import concurrent.futures
from typing import Dict, Any, Optional, Callable
import platform

class UDPReceiver(QtCore.QThread):
    """
    Receives multicast packets in a dedicated QThread and pushes them
    into a thread-safe queue.
    """
    def __init__(self, host: str, module: int) -> None:
        super().__init__()
        self.module_id = module
        self.queue = queue.PriorityQueue()
        self.sock = streamer.get_multicast_socket(host) # streamer from .utils
        if platform.system() == "Linux":
            self.sock.settimeout(0.2)
        else:
            self.sock.setblocking(False)
        self.packets_received = 0
        self.packets_dropped = 0
        self.first_packet_received = 0
        self.prev_seq = 0

    def calc_dropped_packets(self, prev_seq, seq):
        diff = seq - prev_seq
        if diff > 1:
            self.packets_dropped = self.packets_dropped + (diff - 1)
            
    def receive_counter(self):
        self.packets_received += 1

    def get_dropped_packets(self):
        return self.packets_dropped

    def get_received_packets(self):
        return self.packets_received

    def _process_received_packet(self, data):
        """Process a received UDP packet if it matches our module."""
        pkt = streamer.ReadoutPacket(data)
        if pkt.module == self.module_id - 1:
            if (self.first_packet_received == 0) or (self.first_packet_received > pkt.seq):
                self.first_packet_received = pkt.seq
                self.prev_seq = pkt.seq
            self.receive_counter()
            self.calc_dropped_packets(self.prev_seq, pkt.seq)
            self.prev_seq = pkt.seq
            self.queue.put((pkt.seq, pkt))

    def run(self):
        while not self.isInterruptionRequested():
            try:
                data = self.sock.recv(streamer.LONG_PACKET_SIZE)
                self._process_received_packet(data)
            except (socket.timeout, BlockingIOError):
                # No data available right now — just loop
                continue
            except OSError:
                break


    def stop(self):
        print(f"[UDP] UDP receiving thread stopped. Total packets received: {self.packets_received}")
        print(f"[UDP] UDP receiving thread stopped. Total packets dropped: {self.packets_dropped}")
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
                 real_units: bool, psd_absolute: bool, segments: int, signals: PSDSignals, exp_binning: bool = False, nbins: int = 1000):
        super().__init__()
        self.row, self.ch, self.I, self.Q, self.mode = row, ch, I.copy(), Q.copy(), mode
        self.dec_stage, self.real_units, self.psd_absolute = dec_stage, real_units, psd_absolute
        self.segments, self.signals = segments, signals
        self.exp_binning = exp_binning
        self.nbins = nbins

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
        # Determine input units based on whether data was already converted to volts
        input_units = "volts" if self.real_units else "adc_counts"
        
        spec_iq = spectrum_from_slow_tod(i_data=self.I, q_data=self.Q, dec_stage=self.dec_stage,
                                         scaling="psd", reference=ref, nperseg=nper, spectrum_cutoff=0.9,
                                         input_units=input_units)
        
        freq_iq = spec_iq["freq_iq"]
        psd_i = spec_iq["psd_i"]
        psd_q = spec_iq["psd_q"]
        
        # For magnitude PSD: compute in frequency domain from I and Q PSDs
        # For uncorrelated I and Q noise, magnitude PSD ≈ PSD_I + PSD_Q
        # This avoids artifacts from computing PSD of time-domain magnitude
        if ref == "counts":
            # When in counts mode, PSDs are in linear scale
            psd_m = psd_i + psd_q
        else:
            # When in dB scale (dBc or dBm), convert to linear, add, then convert back
            psd_i_linear = 10**(psd_i / 10)
            psd_q_linear = 10**(psd_q / 10)
            psd_m_linear = psd_i_linear + psd_q_linear
            psd_m = 10 * np.log10(psd_m_linear)
        
        freq_m = freq_iq  # Same frequency grid
        
        # Apply exponential binning if enabled
        if self.exp_binning and len(freq_iq) > 1:
            freq_iq_binned, psd_i_binned = exp_bin_noise_data(freq_iq, psd_i, self.nbins)
            _, psd_q_binned = exp_bin_noise_data(freq_iq, psd_q, self.nbins)
            freq_m_binned, psd_m_binned = exp_bin_noise_data(freq_m, psd_m, self.nbins)
            return (freq_iq_binned, psd_i_binned, psd_q_binned, psd_m_binned,
                    freq_m_binned, psd_m_binned, float(self.dec_stage))
        
        return (freq_iq, psd_i, psd_q, psd_m, freq_m, psd_m, float(self.dec_stage))
        
    def _compute_dsb_psd(self, ref, nper):
        # spectrum_from_slow_tod from .utils
        # Determine input units based on whether data was already converted to volts
        input_units = "volts" if self.real_units else "adc_counts"
        
        spec_iq = spectrum_from_slow_tod(i_data=self.I, q_data=self.Q, dec_stage=self.dec_stage,
                                         scaling="psd", reference=ref, nperseg=nper, spectrum_cutoff=0.9,
                                         input_units=input_units)
        freq_dsb, psd_dsb = spec_iq["freq_dsb"], spec_iq["psd_dual_sideband"]
        order = np.argsort(freq_dsb)
        freq_dsb_sorted = freq_dsb[order]
        psd_dsb_sorted = psd_dsb[order]
        
        # Apply exponential binning if enabled
        if self.exp_binning and len(freq_dsb_sorted) > 1:
            freq_dsb_binned, psd_dsb_binned = exp_bin_noise_data(freq_dsb_sorted, psd_dsb_sorted, self.nbins)
            return (freq_dsb_binned, psd_dsb_binned)
        
        return (freq_dsb_sorted, psd_dsb_sorted)

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
        
        # ThreadPool configuration for fitting operations
        self._executor = None
        # Use multiple workers for parallel fitting (reserve 1 core for GUI)
        self._max_workers = max(1, min(4, (os.cpu_count() or 1) - 1))
        self._fitting_future = None

    def stop(self):
        self._running = False
        self.requestInterruption()

    def _progress_callback_wrapper(self, module_idx, progress_percentage):
        if self._running: self.signals.progress.emit(module_idx, progress_percentage)


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
                        'bias_frequency_method': self.params.get('bias_frequency_method', 'max-diq'),
                        'rotate_saved_data': self.params.get('rotate_saved_data', False),
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
                                    data_dict_val['is_bifurcated'] = fitting_module_direct.identify_bifurcation(iq_data, threshold_factor=7)
                                except Exception as e:
                                    print(f"Warning: Bifurcation detection failed for index {res_key}: {e}", file=sys.stderr)
                                    data_dict_val['is_bifurcated'] = False
                            else:
                                data_dict_val['is_bifurcated'] = False
                    
                    # Now apply fitting analysis using the (potentially) bifurcation-annotated data
                    # Use async version with ThreadPoolExecutor for better responsiveness
                    enhanced_results = loop.run_until_complete(
                        self._process_fitting_async(raw_results_from_crs)
                    )
                    results_for_plotting = enhanced_results 
                    
                    results_for_history = {}
                    if enhanced_results: 
                        for res_idx, data_dict_val in enhanced_results.items():
                            if isinstance(res_idx, (int, np.integer)):
                                bias_freq = data_dict_val.get('bias_frequency', data_dict_val.get('original_center_frequency'))
                                if bias_freq is not None:
                                    # Convert 1-based detector index to 0-based conceptual index
                                    conceptual_idx = res_idx - 1
                                    results_for_history[conceptual_idx] = bias_freq
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
    
    async def _process_fitting_async(self, raw_results):
        """Process fitting analysis asynchronously using ThreadPoolExecutor."""
        if not raw_results:
            return raw_results
        
        # For larger datasets, use parallel processing
        return await self._process_fitting_multi_thread(raw_results)
    
    async def _process_fitting_multi_thread(self, raw_results):
        """Process fitting using multiple threads for better performance."""
        loop = asyncio.get_event_loop()
        
        # Split resonances into chunks for parallel processing
        resonance_items = list(raw_results.items())
        num_chunks = min(self._max_workers, len(resonance_items))
        chunk_size = max(1, len(resonance_items) // num_chunks)
        
        chunks = []
        for i in range(0, len(resonance_items), chunk_size):
            chunk = dict(resonance_items[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all chunks for processing
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(
                    self._process_chunk_thread_safe,
                    chunk,
                    self.params.get('apply_skewed_fit', False),
                    self.params.get('apply_nonlinear_fit', False),
                    self.params.get('module'),
                    i,
                    len(chunks)
                )
                futures.append(asyncio.wrap_future(future))
            
            # Collect results as they complete
            enhanced_results = {}
            completed = 0
            
            for future in asyncio.as_completed(futures):
                if self.isInterruptionRequested():
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    return raw_results
                
                try:
                    chunk_results = await future
                    enhanced_results.update(chunk_results)
                    completed += 1
                    
                    # Chunk completed (no progress emission to avoid clutter)
                        
                except Exception as e:
                    print(f"Error processing chunk: {e}", file=sys.stderr)
                    # Continue with other chunks
            
            return enhanced_results
    
    def _process_chunk_thread_safe(self, chunk, apply_skewed, apply_nonlinear, module_idx, chunk_idx, total_chunks):
        """Process a chunk of resonances in a thread."""
        # Process this chunk using the existing logic
        return self._apply_fitting_analysis(chunk, apply_skewed, apply_nonlinear, module_idx)
    
    def _apply_fitting_analysis(self, raw_results, apply_skewed, apply_nonlinear, module_idx):
        """Apply fitting analysis (skewed and/or nonlinear) to multisweep results.
        
        This method runs in a worker thread for parallel processing.
        Emits progress signals for GUI updates (thread-safe via Qt's queued connections).
        """
        # This runs in a separate thread, so we need to be careful about Qt signals
        
        # Initialize enhanced_results as a copy to preserve raw data if fitting is skipped or fails
        enhanced_results = {k: v.copy() for k, v in raw_results.items()}
        
        if not apply_skewed and not apply_nonlinear:
            # If no fitting is requested, add flags indicating this
            for res_idx in enhanced_results:
                enhanced_results[res_idx]['skewed_fit_applied'] = False
                enhanced_results[res_idx]['skewed_fit_success'] = False
                enhanced_results[res_idx]['nonlinear_fit_applied'] = False
                enhanced_results[res_idx]['nonlinear_fit_success'] = False
            return enhanced_results
        
        try:
            if apply_skewed:
                # Emit progress signal (thread-safe via Qt's queued connections)
                if module_idx is not None and self._running:
                    self.signals.fitting_progress.emit(module_idx, 
                        "Fitting in progress: Applying skewed fits")
                
                # Perform skewed fitting
                skewed_results = fitting_module_direct.fit_skewed_multisweep(
                    enhanced_results,
                    approx_Q_for_fit=1e4,
                    fit_resonances=True,
                    center_iq_circle=True,
                    normalize_fit=True
                )
                
                # Update results
                for res_idx in enhanced_results:
                    if res_idx in skewed_results:
                        enhanced_results[res_idx].update(skewed_results[res_idx])
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
                    else:
                        enhanced_results[res_idx]['skewed_fit_applied'] = True
                        enhanced_results[res_idx]['skewed_fit_success'] = False
            else:
                for res_idx in enhanced_results:
                    enhanced_results[res_idx]['skewed_fit_applied'] = False
                    enhanced_results[res_idx]['skewed_fit_success'] = False
            
            if apply_nonlinear:
                # Emit progress signal
                if module_idx is not None and self._running:
                    self.signals.fitting_progress.emit(module_idx, 
                        "Fitting in progress: Applying non-linear fits")
                
                # Perform nonlinear fitting
                # Disable parallel processing since we're already in a thread
                nonlinear_results = fitting_nonlinear.fit_nonlinear_iq_multisweep(
                    enhanced_results.copy(),
                    fit_nonlinearity=True,
                    n_extrema_points=5,
                    verbose=False,
                    parallel=False  # Avoid nested thread pools
                )
                
                # Update results
                for res_idx in enhanced_results:
                    if res_idx in nonlinear_results:
                        enhanced_results[res_idx].update(nonlinear_results[res_idx])
                        enhanced_results[res_idx]['nonlinear_fit_applied'] = True
                        if 'nonlinear_fit_success' not in enhanced_results[res_idx]:
                            enhanced_results[res_idx]['nonlinear_fit_success'] = False
                        
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
                    else:
                        enhanced_results[res_idx]['nonlinear_fit_applied'] = True
                        enhanced_results[res_idx]['nonlinear_fit_success'] = False
            else:
                for res_idx in enhanced_results:
                    enhanced_results[res_idx]['nonlinear_fit_applied'] = False
                    enhanced_results[res_idx]['nonlinear_fit_success'] = False
            
            # Emit completion
            if module_idx is not None and self._running:
                self.signals.fitting_progress.emit(module_idx, "Fitting Completed")
            
            return enhanced_results
            
        except Exception as e:
            print(f"Error in thread-safe fitting analysis: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Return results with error flags
            for res_idx in enhanced_results:
                enhanced_results[res_idx]['fitting_error'] = str(e)
            return enhanced_results


class BiasKidsSignals(QObject):
    """Signals for BiasKidsTask."""
    progress = pyqtSignal(int, float)  # module, progress_percentage
    completed = pyqtSignal(int, dict, dict, float)  # module, biased_results, df_calibrations, nco_frequency_hz
    error = pyqtSignal(str)  # error_message


class BiasKidsTask(QtCore.QThread):
    """QThread subclass for running the bias_kids algorithm without blocking the GUI."""
    
    def __init__(self, crs: "CRS", module: int, multisweep_results: dict, signals: BiasKidsSignals, bias_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the BiasKidsTask.
        
        Args:
            crs: Control and Readout System object
            module: Module number to bias
            multisweep_results: Multisweep results in GUI format
            signals: Signal object for communication with GUI
            bias_params: Optional dictionary of bias parameters from dialog
        """
        super().__init__()
        self.crs = crs
        self.module = module
        self.multisweep_results = multisweep_results
        self.signals = signals
        self.bias_params = bias_params or {}
        self._running = True
        
    def stop(self):
        """Stop the task."""
        self._running = False
        self.requestInterruption()
        
    def run(self):
        """QThread entry point - runs in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Progress callback
            def progress_cb(module, progress):
                if self._running:
                    self.signals.progress.emit(module, progress)
            
            # Run the bias_kids algorithm
            result = loop.run_until_complete(self._run_bias_kids(progress_cb))
            
            if self.isInterruptionRequested():
                self.signals.error.emit("Bias KIDs operation was cancelled.")
                return
                
            if result:
                # Handle both dict and list return types from bias_kids
                if isinstance(result, list):
                    # For list results (multiple modules), we're only processing one module here
                    # so this shouldn't happen, but handle it gracefully
                    if len(result) > 0:
                        result = result[0]  # Take the first module's results
                    else:
                        self.signals.error.emit("Bias KIDs operation returned empty list.")
                        return
                
                # Extract df_calibration values (result is now guaranteed to be a dict)
                df_calibrations = {}
                for det_idx, det_data in result.items():
                    if 'df_calibration' in det_data:
                        df_calibrations[det_idx] = det_data['df_calibration']
                
                # Read the NCO frequency that was used during biasing
                nco_frequency_hz = loop.run_until_complete(self.crs.get_nco_frequency(module=self.module))
                
                # Emit completion with results and NCO frequency
                self.signals.completed.emit(self.module, result, df_calibrations, float(nco_frequency_hz))
            else:
                self.signals.error.emit("Bias KIDs operation returned no results.")
                
        except asyncio.CancelledError:
            self.signals.error.emit("Bias KIDs operation was cancelled.")
        except Exception as e:
            import traceback
            error_msg = f"Error in BiasKidsTask: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Print detailed message to Console
            self.signals.error.emit(str(e))
        finally:
            if loop.is_running():
                loop.stop()
            loop.close()
            
    async def _run_bias_kids(self, progress_callback):
        """Run the bias_kids algorithm asynchronously."""
        # Import bias_kids as a regular function
        from rfmux.algorithms.measurement.bias_kids import bias_kids
        
        # Extract parameters from bias_params
        kwargs = {
            'crs': self.crs,
            'multisweep_results': self.multisweep_results,
            'module': self.module,
            'progress_callback': progress_callback
        }
        
        # Add optional parameters from dialog
        if 'nonlinear_threshold' in self.bias_params:
            kwargs['nonlinear_threshold'] = self.bias_params['nonlinear_threshold']
        if 'fallback_to_lowest' in self.bias_params:
            kwargs['fallback_to_lowest'] = self.bias_params['fallback_to_lowest']
        if 'optimize_phase' in self.bias_params:
            kwargs['optimize_phase'] = self.bias_params['optimize_phase']
        if 'bandpass_params' in self.bias_params:
            kwargs['bandpass_params'] = self.bias_params['bandpass_params']
        if 'num_phase_samples' in self.bias_params:
            kwargs['num_phase_samples'] = self.bias_params['num_phase_samples']
        if 'phase_step' in self.bias_params:
            kwargs['phase_step'] = self.bias_params['phase_step']
        
        # Call bias_kids with all parameters
        result = await bias_kids(**kwargs)
        return result
