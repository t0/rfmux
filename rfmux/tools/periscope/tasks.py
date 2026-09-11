"""Worker threads and tasks used by the Periscope viewer."""

from .utils import * # Imports QtCore, QThread, QObject, pyqtSignal, QRunnable,
                     # streamer, np, asyncio, time, socket, queue, traceback, sys,
                     # DEFAULT_AMPLITUDE, DENSITY_GRID,
                     # SCATTER_POINTS, spectrum_from_slow_tod, pg,
                     # gaussian_filter, convolve, SMOOTH_SIGMA, LOG_COMPRESS,
                     # DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_NSAMPLES,
                     # DEFAULT_NPOINTS, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN

from rfmux.core.transferfunctions import exp_bin_noise_data # Import exponential binning function
from rfmux.pulse_capture.sources import _set_receive_timeout
from rfmux.tuning.find_resonances import find_resonances_in_netanal
from rfmux.tuning.fits import fit_sweeps, fit_sweeps_at_bias_amplitude
from rfmux.tuning.bias import find_bias_points

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

class DfCalibrationSignals(QObject):
    completed = pyqtSignal(int, dict)
    error = pyqtSignal(str)


class DfCalibrationTask(QtCore.QThread):
    """Runs one df-calibration measurement off the GUI thread.

    *measure* is a callable returning the coroutine to run; the app
    hands in crs.measure_df_calibrations for the module, tests hand in
    whatever they like.  Mock mode measures at startup
    and the sweep is seconds at many tones: it must not hold the window.
    """

    def __init__(self, measure, module: int,
                 signals: DfCalibrationSignals, parent=None):
        super().__init__(parent)
        self.measure, self.module, self.signals = measure, module, signals

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            cals = loop.run_until_complete(self.measure())
            self.signals.completed.emit(self.module, dict(cals or {}))
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            loop.close()


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
    progress = pyqtSignal(int, float)          # module, percent
    # module, trace. The trace is the module's measured arrays out of what
    # take_netanal returned: partial while the sweep runs, whole on the last
    # one, with the same keys either way.
    data_update = pyqtSignal(int, dict)
    # module, container. The container is take_netanal's whole return, keyed by
    # module identifier: what gets saved, and what the panel keeps so a save
    # after the fact writes the measurement rather than a view of it.
    completed = pyqtSignal(int, object); error = pyqtSignal(str)

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
    def __init__(self, crs: "CRS", module: int, params: dict, signals: NetworkAnalysisSignals):
        super().__init__()
        self.crs, self.module, self.params, self.signals = crs, module, params, signals
        self._running = True
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

            # No setup here. take_netanal sets the NCO it needs and zeroes its
            # own tones on the way out; anything else -- clearing channels,
            # cable length -- is the operator's, and cable length in particular
            # rotates the phase of everything the board reads.
            if not self.isInterruptionRequested():
                # Combine parameters for the take_netanal call
                netanal_params = {
                    'amp': task_params['amp'],
                    'fmin': task_params['fmin'],
                    'fmax': task_params['fmax'],
                    'nsamps': task_params['nsamps'],
                    'npoints': task_params['npoints'],
                    'max_chans': task_params['max_chans'],
                    'max_span': task_params['max_span'],
                    'module': self.module,
                    'progress_callback': progress_cb,
                    'data_callback': data_cb,
                    # Periscope writes its own session export through
                    # SessionManager. Leaving the driver's autosave on would
                    # put a second copy of every measurement in a second
                    # folder, in a second layout.
                    'save': False,
                }
                
                # Process the network analysis asynchronously without blocking
                result = loop.run_until_complete(self._process_network_analysis(loop, netanal_params))
                
                # Process results if available and task wasn't interrupted
                if not self.isInterruptionRequested() and result:
                    self.signals.data_update.emit(
                        self.module, self._trace_of(result))
                    self.signals.completed.emit(self.module, result)
            
        except asyncio.CancelledError:
            self.signals.error.emit(f"Analysis canceled for module {self.module}")
            if loop.is_running():
                loop.run_until_complete(self._cleanup_channels())
        except KeyError as ke:
            err_msg = f"Module {self.module} is not in what take_netanal returned: {ke}"
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
        
    def _trace_of(self, container):
        """The module's measured arrays, out of what take_netanal returned."""
        return container[self.crs.module[self.module].index()]['results']

    def _create_data_callback(self):
        def data_cb(module_idx, partial):
            if not self._running:
                return
            # Acquisition order is a stride pattern within each comb; the panel
            # plots a line, so hand it the sweep sorted the way the finished
            # trace is sorted.
            order = np.argsort(partial['frequencies'])
            self.signals.data_update.emit(module_idx, {
                key: value[order] for key, value in partial.items()
            })
        return data_cb
    
    def _extract_parameters(self):
        # Constants from .utils
        return {'amp': self.params.get('amp', DEFAULT_AMPLITUDE),
                'fmin': self.params.get('fmin', DEFAULT_MIN_FREQ), 'fmax': self.params.get('fmax', DEFAULT_MAX_FREQ),
                'nsamps': self.params.get('nsamps', DEFAULT_NSAMPLES), 'npoints': self.params.get('npoints', DEFAULT_NPOINTS),
                'max_chans': self.params.get('max_chans', DEFAULT_MAX_CHANNELS), 'max_span': self.params.get('max_span', DEFAULT_MAX_SPAN),
                }
        
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

class FindResonancesSignals(QObject):
    # module, ResonanceSearch. The search also went into the module's netanal
    # output, where the finder puts it, so the panel's container carries it and
    # a save writes it out with the trace it was found in.
    completed = pyqtSignal(int, object)
    error = pyqtSignal(int, str)


class FindResonancesTask(QtCore.QThread):
    """Runs the resonance finder off the GUI thread.

    Milliseconds on a netanal-sized trace -- 25 ms over 20,000 points of the
    simulator -- so this is not about speed. It is so a search reports through
    the same signals as every other measurement, and so a threshold that finds
    far more candidates than the operator meant cannot hold the window.
    """

    def __init__(self, module: int, module_netanal: dict, params: dict,
                 signals: FindResonancesSignals):
        super().__init__()
        self.module = module
        self.module_netanal = module_netanal
        self.params = params
        self.signals = signals

    def run(self):
        try:
            # save=False: the panel writes through store when it is ready to,
            # so the driver's autosave cannot put a second copy elsewhere.
            search = find_resonances_in_netanal(
                self.module_netanal, save=False, **self.params)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self.signals.error.emit(self.module, f"{type(e).__name__}: {e}")
            return
        self.signals.completed.emit(self.module, search)


class RunFitsSignals(QObject):
    progress = pyqtSignal(int, int)             # sweeps fitted, sweeps to fit
    completed = pyqtSignal(object)              # the FitReport
    error = pyqtSignal(str)


class RunFitsTask(QtCore.QThread):
    """Fits one module's sweeps off the GUI thread.

    Seconds, not milliseconds: 80 ms a sweep for all three models on the
    simulator, so a schedule over a real array is minutes. Hence a thread, a
    per-sweep progress count, and a button that goes dead while it runs.

    The fits go into the sweep entries the panel already holds -- that is what
    ``fit_sweeps`` does -- so there is nothing to hand back but the report.
    """

    def __init__(self, module_sweeps: dict, models, amplitude_choice,
                 signals: RunFitsSignals):
        super().__init__()
        self.module_sweeps = module_sweeps
        self.models = tuple(models)
        # None fits every sweep, "bias" each resonator's own bias amplitude,
        # and an integer one amplitude step.
        self.amplitude_choice = amplitude_choice
        self.signals = signals

    def run(self):
        try:
            # save=False: the panel re-saves through store, so the fitters'
            # autosave cannot put a second copy in a second place.
            if self.amplitude_choice == "bias":
                report = fit_sweeps_at_bias_amplitude(
                    self.module_sweeps, models=self.models, save=False,
                    progress_callback=self._progress)
            else:
                report = fit_sweeps(
                    self.module_sweeps, models=self.models,
                    iterations=self.amplitude_choice,
                    save=False, progress_callback=self._progress)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self.signals.error.emit(f"{type(e).__name__}: {e}")
            return
        self.signals.completed.emit(report)

    def _progress(self, completed, total):
        self.signals.progress.emit(int(completed), int(total))


class FindBiasSignals(QObject):
    completed = pyqtSignal(object)              # the BiasReport
    error = pyqtSignal(str)


class FindBiasTask(QtCore.QThread):
    """Finds one module's bias points off the GUI thread.

    Analysis, not measurement: 0.33 s for 200 resonators over five amplitude
    steps in both directions, 1.8 s for 1000. That is short enough to want no
    progress reporting -- and ``find_bias_points`` offers no callback to build
    any from -- but long enough that running it on the GUI thread would freeze
    the window, so it gets a thread and the button goes dead while it runs.

    The report carries the new catalog, and the sweeps come back carrying
    ``bias_report``; the panel does the saving.
    """

    def __init__(self, module_sweeps: dict, parameters: dict,
                 signals: FindBiasSignals):
        super().__init__()
        self.module_sweeps = module_sweeps
        self.parameters = dict(parameters)
        self.signals = signals

    def run(self):
        try:
            # save=False: the panel re-saves through store, so the finder's
            # autosave cannot put a second copy in a second place.
            report = find_bias_points(self.module_sweeps, save=False,
                                      **self.parameters)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self.signals.error.emit(f"{type(e).__name__}: {e}")
            return
        self.signals.completed.emit(report)


class ApplyBiasSignals(QObject):
    completed = pyqtSignal()
    error = pyqtSignal(str)


class ApplyBiasTask(QtCore.QThread):
    """Programs a catalog's bias points onto the board.

    One ``crs.apply_bias`` call and nothing else: which NCO to use, and putting
    the frequencies on the tone grid, are the driver's.
    """

    def __init__(self, crs, catalog, signals: ApplyBiasSignals):
        super().__init__()
        self.crs = crs
        self.catalog = catalog
        self.signals = signals

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.crs.apply_bias(self.catalog))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self.signals.error.emit(f"{type(e).__name__}: {e}")
            return
        finally:
            loop.close()
        self.signals.completed.emit()


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
    progress = pyqtSignal(int, float)          # module, percent across the whole call
    # module, partial entries, amplitude step, direction: multisweep's
    # data_callback, so a panel can draw a sweep while it is being measured.
    partial_data = pyqtSignal(int, dict, int, str)
    # One finished sweep, as sweep_callback handed it over: step, direction,
    # amplitudes, factor, completed, total, data.
    sweep_completed = pyqtSignal(dict)
    # module, container. The container is multisweep's whole return, keyed by
    # module identifier: what gets saved, and what the panel keeps.
    completed = pyqtSignal(int, object)
    error = pyqtSignal(str)


class MultisweepTask(QtCore.QThread):
    """Runs one ``crs.multisweep`` call off the GUI thread.

    One call is the whole measurement -- every amplitude step of the schedule,
    in every direction asked for -- so there is nothing here to loop over. The
    driver's three callbacks become signals, and what it returns is handed over
    whole.
    """

    def __init__(self, crs: "CRS", params: dict, signals: MultisweepSignals):
        super().__init__()
        self.crs = crs
        self.params = params
        self.signals = signals
        # The worker sweeps a copy, so the panel is free to adopt a different
        # catalog -- the one Find Bias returns, say -- while this sweep runs.
        self.catalog = params['catalog'].copy()
        # A catalog belongs to one module, which makes it the one place the
        # module can be read from.
        self.module = self.catalog.module
        self._running = True

    def stop(self):
        self._running = False
        self.requestInterruption()

    def run(self):
        """QThread entry point - runs in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # None is a cancelled sweep, which the panel that cancelled it
            # already knows about. Anything else is the whole measurement,
            # handed over even if Cancel arrived while it was being packed.
            container = loop.run_until_complete(self._process_multisweep(loop))
            if container is not None:
                self.signals.completed.emit(self.module, container)
        except asyncio.CancelledError:
            self.signals.error.emit(f"Multisweep canceled for module {self.module}")
        except Exception as e:
            err_msg = (f"Multisweep failed on module {self.module}: "
                       f"{type(e).__name__}: {e}")
            print(f"ERROR: {err_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self.signals.error.emit(err_msg)
        finally:
            if loop.is_running():
                loop.stop()
            loop.close()

    def _multisweep_params(self):
        return {
            'catalog': self.catalog,
            'span_hz': self.params['span_hz'],
            'npoints_per_sweep': self.params['npoints_per_sweep'],
            'nsamps': self.params['nsamps'],
            # A number, a mapping or an AmplitudeSchedule, as the dialog built
            # it; None sweeps each resonator at its own catalog amplitude.
            'amp': self.params.get('amp'),
            'sweep_direction': self.params.get('sweep_direction', 'upward'),
            'progress_callback': self._progress_callback,
            'data_callback': self._data_callback,
            'sweep_callback': self._sweep_callback,
            # Periscope writes its own session export through SessionManager.
            # Leaving the driver's autosave on would put a second copy of every
            # measurement in a second folder, in a second layout.
            'save': False,
        }

    def _progress_callback(self, module_idx, progress_percentage):
        if self._running:
            self.signals.progress.emit(module_idx, progress_percentage)

    def _data_callback(self, module_idx, partial, step, direction):
        if self._running:
            self.signals.partial_data.emit(module_idx, partial, step, direction)

    def _sweep_callback(self, record):
        if self._running:
            self.signals.sweep_completed.emit(record)

    async def _process_multisweep(self, loop):
        """Run the one call, yielding often enough that Cancel is answered."""
        task = loop.create_task(self.crs.multisweep(**self._multisweep_params()))

        while not task.done():
            if self.isInterruptionRequested():
                task.cancel()
                await asyncio.sleep(0.01)   # let the cancellation land
                return None
            await asyncio.sleep(0.1)

        if task.cancelled():
            return None
        return await task
