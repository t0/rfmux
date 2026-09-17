"""Worker threads and tasks used by the Periscope viewer."""

from .utils import * # Imports QtCore, QThread, QObject, pyqtSignal, QRunnable,
                     # streamer, np, asyncio, time, socket, queue, traceback, sys,
                     # DEFAULT_AMPLITUDE, DENSITY_GRID,
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
from rfmux.pulse_capture.sources import _set_receive_timeout

# Additional imports for async fitting with ThreadPoolExecutor
import os
import concurrent.futures
from typing import Dict, Any, Optional

class UDPReceiver(QtCore.QThread):
    """
    Receives multicast packets in a dedicated QThread using the C++ ReadoutPacketReceiver.
    The C++ receiver handles packet reordering.
    """
    def __init__(self, host: str, module: int) -> None:
        super().__init__()
        self.module_id = module  # 1-indexed module ID (for Periscope)
        self.module_idx = module - 1  # 0-indexed (for packet filtering)

        # Create socket and C++ receiver
        # reorder_window=256: Maintain good packet reordering capability
        # queue_max_size=50000: Handle high data rates at FIR stage 0 (~38kHz)
        # flush_threshold=16: Flush every 16 packets for smooth updates
        # Ask BEFORE binding: the probe is a plain bind, which our own
        # socket would then fail. Loopback only -- see
        # find_competing_receiver for why a second reader is fatal for
        # the mock's unicast fallback and harmless for multicast, which
        # a board always sends and the mock sends when it can.
        self._port_conflict = streamer.find_competing_receiver(host)
        if self._port_conflict:
            print(f"[UDP] {self._port_conflict}")

        self.sock = streamer.get_multicast_socket(host)
        # A receive timeout on the SOCKET, because receive_batch's own
        # timeout_ms cannot be relied on: recvmmsg runs with
        # MSG_WAITFORONE, and the kernel only consults that timeout
        # BETWEEN datagrams. On a socket that receives nothing at all --
        # a board that is not streaming yet, a mistyped host, or another
        # process on 9876 taking the datagrams via SO_REUSEPORT -- the
        # call blocks forever. The thread then never reaches the
        # queue-discovery loop below, so Periscope draws nothing and
        # reports "0 packets received" with no error to explain it.
        # SO_RCVTIMEO on a socket left blocking: the call waits up to
        # 0.5 s, then returns EAGAIN, which receive_batch already treats
        # as "no packets this time". settimeout() would not do: it makes
        # the fd non-blocking, so an empty socket returns EAGAIN at once
        # and run() spins, retaking the GIL on every call. The 0.5 s
        # also bounds how long stop() waits for the thread.
        _set_receive_timeout(self.sock, 0.5)
        self.receiver = streamer.ReadoutPacketReceiver(self.sock,
                                                       reorder_window=256,
                                                       queue_max_size=50000,
                                                       flush_threshold=16)

        # Queue reference will be set when we first see packets for our module
        self.queue = None
        self.serial = None

        # Set while packets are arriving for some OTHER module.  See
        # _note_module_mismatch: without this the GUI cannot tell that
        # case apart from a dead stream.
        self._module_mismatch = None

        # Statistics
        self.packets_received = 0
        self.packets_dropped = 0

    def _discover_queue(self):
        """Adopt the queue for our module, or note that there isn't one."""
        streaming = []
        for serial, module, q in self.receiver.get_all_queues():
            streaming.append(module + 1)
            if module == self.module_idx:
                self.queue = q
                self.serial = serial
                self._module_mismatch = None
                print(f"[UDP] Found queue for serial={serial}, module={self.module_id}")
                return
        self._note_module_mismatch(streaming)

    def _note_module_mismatch(self, streaming_modules):
        """Record that packets are arriving, but for nobody's module.

        Every counter this class exposes reads through ``self.queue``, so
        a module that never matches reports a flat zero -- no packets, no
        loss, no error -- while the receiver is in fact working
        perfectly.  That is indistinguishable from a dead stream unless
        something says otherwise, and it is not a rare mistake: the
        startup dialog restores the last-used module, and the mock streams
        only the modules that carry a tone -- at startup, module 1 -- so
        going from hardware to mock lands here.
        """
        if not streaming_modules:
            self._module_mismatch = None   # nothing streaming yet
            return
        modules = ", ".join(str(m) for m in sorted(set(streaming_modules)))
        msg = (f"Module {self.module_id} is not streaming - packets are "
               f"arriving for module {modules}. No data will appear.")
        if msg != self._module_mismatch:
            self._module_mismatch = msg
            print(f"[UDP] {msg}")

    def get_port_conflict(self):
        """A competing receiver that may be taking our packets, or None.

        Only while we have no queue: once packets arrive, whatever else
        is bound evidently did not take them, and a stale warning would
        be worse than none.
        """
        if self.queue is not None:
            return None
        return self._port_conflict

    def get_module_mismatch(self):
        """The mismatch message, or None while healthy.

        Polled by the GUI status bar; printed once for headless callers.
        """
        return self._module_mismatch

    def get_missing_packets(self):
        """Packets that never arrived (wire or kernel socket buffer).

        Counted individually, not per discontinuity: one burst of a
        thousand lost packets is a thousand here and one in
        ``sequence_gaps``.
        """
        if self.queue is not None:
            return self.queue.get_stats().packets_missing
        return 0

    def get_queue_drops(self):
        """Packets the receiver got but the GUI never consumed.

        Queue overflow: Periscope is behind, not the network.
        """
        if self.queue is not None:
            return self.queue.get_stats().packets_dropped
        return self.packets_dropped

    def get_dropped_packets(self):
        """Everything lost, however it was lost.

        Kept for callers that just want one number; anything
        diagnosing a problem wants the two apart, because the fixes are
        unrelated -- one is a network or kernel buffer, the other is
        Periscope being too slow.
        """
        return self.get_missing_packets() + self.get_queue_drops()

    def get_received_packets(self):
        """Get cumulative received packet count from C++ queue statistics."""
        if self.queue is not None:
            stats = self.queue.get_stats()
            return stats.packets_received
        return self.packets_received

    def run(self):
        """Main reception loop - calls C++ receiver and retrieves packets."""
        while not self.isInterruptionRequested():
            try:
                # Call C++ receiver to read and process packets
                # batch_size is a CEILING, not a wait: recvmmsg runs with
                # MSG_WAITFORONE, so it returns as soon as one packet is
                # there with whatever else is already queued. A small
                # ceiling therefore costs nothing at low rates and
                # everything at high ones -- this thread has to retake
                # the GIL once per call, and at 16 packets a call that
                # is 2,400 acquisitions a second at stage 0, competing
                # with the GUI. Measured on a board at stage 0 with a
                # 128-channel capture running: 51.8% of the stream lost
                # to kernel-buffer overflow at 16, none at 2048.
                self.receiver.receive_batch(batch_size=2048, timeout_ms=50)

                # Find our module's queue
                if self.queue is None:
                    self._discover_queue()

            except Exception as e:
                if self.isInterruptionRequested():
                    break
                print(f"[UDP] Error in receive loop: {e}")
                continue

    def stop(self):
        print(f"[UDP] UDP receiving thread stopped. Total packets received: {self.get_received_packets()}")
        print(f"[UDP] UDP receiving thread stopped. Total packets dropped: {self.get_dropped_packets()}")
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

class NetworkAnalysisSignals(QObject):
    progress = pyqtSignal(int, float)
    amplitude_started = pyqtSignal(int, int, int, float)  # module, index, count, amplitude
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
        from rfmux.algorithms.measurement.bias_kids import dac_scale_dbm
        for module_idx in range(1, 9): # Renamed module
            try:
                dac_scales[module_idx] = loop.run_until_complete(
                    dac_scale_dbm(self.crs, module_idx))
            except Exception as e:
                print(f"Error fetching DAC scale for module {module_idx}: {e}", file=sys.stderr) # Print to stderr
                dac_scales[module_idx] = None

class MultisweepSignals(QObject):
    progress = pyqtSignal(int, float)
    completed_iteration = pyqtSignal(int, int, float, str) # module, iteration, amplitude, direction
    starting_iteration = pyqtSignal(int, int, float, str) # module, iteration, amplitude, direction
    fitting_progress = pyqtSignal(int, str) # module, status_message
    all_completed = pyqtSignal()
    error = pyqtSignal(int, float, str)

class BiasKidsSignals(QObject):
    """Signals for BiasKidsTask."""
    progress = pyqtSignal(int, float)  # module, progress_percentage
    completed = pyqtSignal(int, dict, float)  # module, biased_results, nco_frequency_hz
    error = pyqtSignal(str)  # error_message

class BiasKidsTask(QtCore.QThread):
    """QThread subclass for running the bias_kids algorithm without blocking the GUI."""

    def __init__(self, crs: "CRS", module: int, multisweep_results: dict, signals: BiasKidsSignals, bias_params: Optional[Dict[str, Any]] = None,
                 nco_frequency_hz: Optional[float] = None):
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
        # The NCO the sweep was taken at, to set before biasing when the
        # board is not already there (loaded data); None leaves it alone.
        self.nco_frequency_hz = nco_frequency_hz
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

                # Read the NCO frequency that was used during biasing
                nco_frequency_hz = loop.run_until_complete(self.crs.get_nco_frequency(module=self.module))

                # Emit completion with results and NCO frequency
                self.signals.completed.emit(self.module, result, float(nco_frequency_hz))
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
        if 'fit_method' in self.bias_params:
            kwargs['fit_method'] = self.bias_params['fit_method']
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
        for key in ('measure_calibration', 'calibration_step'):
            if key in self.bias_params:
                kwargs[key] = self.bias_params[key]

        # bias_kids places its tones relative to the board's NCO, so the
        # board must be where the sweep was before it runs.
        if self.nco_frequency_hz is not None:
            await self.crs.set_nco_frequency(self.nco_frequency_hz,
                                             module=self.module)
        result = await bias_kids(**kwargs)
        return result
