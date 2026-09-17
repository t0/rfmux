#!/usr/bin/env python3
"""
Detector tuning as a plain script: bias a set of KIDs and measure their noise.

Executes the complete measurement sequence:
1. Initialize
2. Network analysis
3. Unwrap cable delay
4. Find resonances
5. Take multisweep
6. Do fitting
7. Bias the kids
8. Get slow samples
9. Get PFB samples

For what any of these steps mean, what the parameters do and what the
data looks like at each stage, open simplified_tuning_flow.md in this
folder as a notebook (double-click it in the Jupyter panel Periscope
launches; in your own JupyterLab, right-click -> Open With -> Notebook).
It is the documentation; this is the runner.

Usage:
    python simplified_tuning_flow.py MOCK         # Run with mock CRS
    python simplified_tuning_flow.py 0042         # Run with real CRS serial 0042
"""

import asyncio
import numpy as np
from datetime import datetime
import sys

# Import required modules
import rfmux
from rfmux.core.crs import CRS
from rfmux.core.transferfunctions import (
    fit_cable_delay, calculate_new_cable_length,
    convert_amplitude_to_dbm, convert_dbm_to_amplitude)
from rfmux.algorithms.measurement.fitting import find_resonances, fit_skewed_multisweep
from rfmux.algorithms.measurement.fitting_nonlinear import fit_nonlinear_iq_multisweep
from rfmux.algorithms.measurement.bias_kids import bias_kids, dac_scale_dbm
from rfmux.streamer import find_streamer_conflict

# Import mock mode support
from rfmux.mock.helpers import create_mock_crs


async def main(serial="MOCK"):
    """Main execution flow replicating periscope algorithm.
    
    Args:
        serial: CRS serial number (e.g., "0042") or "MOCK" for mock mode
    """
    
    # Configuration parameters
    MODULE = 1  # Use Module 1

    # Tone powers, in dBm.  The algorithms take a normalized amplitude, a
    # fraction of the DAC's full scale; the conversion needs the board's
    # DAC scale, so it happens once the board is connected.
    NETANAL_POWER_DBM = -60.0
    MULTISWEEP_POWERS_DBM = [-65.0, -60.0, -55.0]   # bias_kids picks one per detector

    # Network analysis parameters ('amp' is filled in from NETANAL_POWER_DBM)
    NETANAL_PARAMS = {
        'fmin': 0.6e9,      # 600 MHz
        'fmax': 1.1e9,     # 1100 MHz  
        'nsamps': 10,
        'npoints': 50000,
        'max_chans': 1023,
        'max_span': 500e6,  # 500 MHz
        'module': MODULE
    }
    
    # Find resonances parameters
    FIND_RES_PARAMS = {
        'min_dip_depth_db': 1.0,
        'min_Q': 1e4,
        'max_Q': 1e7,
        'min_resonance_separation_hz': 100e3,
        'data_exponent': 2.0
    }
    
    # Multisweep parameters
    MULTISWEEP_PARAMS = {
        'span_hz': 200e3,             # the multisweep defaults: 200 kHz span,
        'npoints_per_sweep': 101,     # 2 kHz per point across a 10 kHz linewidth
        'nsamps': 10,                 # one sweep per MULTISWEEP_POWERS_DBM
        'module': MODULE,
        'bias_frequency_method': 'max-diq',  # Method to find optimal bias frequency
        'rotate_saved_data': False,          # Don't rotate data for df calibration consistency
        'sweep_direction': 'upward'
    }
    
    # Fitting parameters
    FIT_PARAMS = {
        'apply_skewed_fit': True,
        'apply_nonlinear_fit': True,  # Optional
        'approx_Q_for_fit': 1e4,
        'fit_resonances': True,
        'center_iq_circle': True,
        'normalize_fit': True
    }

    # Bias parameters
    BIAS_PARAMS = {
        'nonlinear_threshold': 0.77,   # highest nonlinearity `a` to bias at
        'fallback_to_lowest': True,    # no suitable power: use the lowest
        'fit_method': 'nonlinear',     # the fit the choice is read from
        'measure_calibration': True,   # step the tones to measure df_calibration
        'calibration_step': 0.05,      # that step, in fitted linewidths
        'optimize_phase': False,       # rotate the IQ basis to put the signal in Q
        'num_phase_samples': 300,      # samples that rotation is chosen from
        'bandpass_params': None,       # band-pass them first; None is 5 to 20 Hz
        'module': MODULE
    }

    SAMPLE_PARAMS = {
        'num_samples': 1000,
        'return_spectrum': True,
        'scaling': 'psd',
        'reference': 'absolute',
        'nsegments': 5,
        'spectrum_cutoff': 0.9,
        'channel': None,
        'module': MODULE
    }

    # Simulated detectors, used only when serial is "MOCK".  Placed in the
    # network-analysis band above so the sweep can actually find them.
    MOCK_CONFIG = {
        'num_resonances': 10,
        'freq_start': 0.6e9,  # 600 MHz - within network analysis range
        'freq_end': 1.0e9,    # 1 GHz - within network analysis range
        'resonator_random_seed': 42,  # same resonators every run
    }

    is_mock = serial.upper() == "MOCK"
    
    print("="*60)
    print("Simple Periscope Algorithm Flow")
    print(f"Mode: {'MOCK' if is_mock else 'REAL HARDWARE'}")
    print(f"Serial: {serial}")
    print(f"Started at: {datetime.now()}")
    print("="*60)

    crs = None
    try:
        # Step 1: Initialize CRS connection
        print("\n1. Initializing CRS...")
        
        if is_mock:
            # Refuse to be the second simulation on the port. Mock streamers
            # all send to 127.0.0.1:9876, so a reader gets both interleaved
            # and every number below is quietly wrong.
            conflict = find_streamer_conflict()
            if conflict:
                raise RuntimeError(
                    f"refusing to start a simulation — {conflict}. "
                    "Stop whatever is streaming (a Periscope in mock mode, "
                    "another run of this script, a mock server left behind "
                    "by a crashed process) and try again.")

            # Use mock CRS with simulated resonators.  Only keys that exist
            # in rfmux.mock.config.MOCK_DEFAULTS do anything — apply_overrides
            # accepts unknown ones silently, so a typo or a parameter that
            # was never implemented reads as configuration and is not.
            crs = await create_mock_crs(
                module=MODULE,
                config=MOCK_CONFIG,
                verbose=True
            )

            # For mock mode, we don't need to set timestamp port
            await crs.clear_channels(module=MODULE)
            print("   ✓ Mock CRS initialized with simulated resonators")

            # Run the algorithm flow with mock CRS.  The resonator count is
            # known here, so the run can check its own findings.
            dac_scale = await dac_scale_dbm(crs, MODULE)
            if dac_scale is None:
                raise RuntimeError(f"module {MODULE} reports no DAC scale")
            NETANAL_PARAMS['amp'] = convert_dbm_to_amplitude(NETANAL_POWER_DBM, dac_scale)
            await run_algorithm_flow(crs, MODULE, NETANAL_PARAMS, FIND_RES_PARAMS,
                                   MULTISWEEP_PARAMS, FIT_PARAMS, SAMPLE_PARAMS,
                                   expected_resonances=MOCK_CONFIG['num_resonances'],
                                   collect_pfb=True, pfb_samples=20_000,
                                   is_mock=True, BIAS_PARAMS=BIAS_PARAMS,
                                   multisweep_powers_dbm=MULTISWEEP_POWERS_DBM,
                                   dac_scale=dac_scale)

        else:
            # Use real hardware - load session with serial number
            session = rfmux.load_session(f'!HardwareMap [ !CRS {{ serial: "{serial}" }} ]')
            crs = session.query(CRS).one()
            
            # Resolve the connection
            await crs.resolve()
            print(f"   ✓ Connected to CRS serial: {serial}")
            
            # Set timestamp port
            if hasattr(crs, 'TIMESTAMP_PORT'):
                await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)
                print("   ✓ Timestamp port set to TEST")
            
            # Clear channels
            await crs.clear_channels(module=MODULE)
            print("   ✓ Channels cleared")

            # Run the algorithm flow.  No expected_resonances: how many a
            # real array has is what the sweep is there to find out.
            dac_scale = await dac_scale_dbm(crs, MODULE)
            if dac_scale is None:
                raise RuntimeError(f"module {MODULE} reports no DAC scale")
            NETANAL_PARAMS['amp'] = convert_dbm_to_amplitude(NETANAL_POWER_DBM, dac_scale)
            await run_algorithm_flow(crs, MODULE, NETANAL_PARAMS, FIND_RES_PARAMS,
                                   MULTISWEEP_PARAMS, FIT_PARAMS, SAMPLE_PARAMS,
                                   collect_pfb=True, BIAS_PARAMS=BIAS_PARAMS,
                                   multisweep_powers_dbm=MULTISWEEP_POWERS_DBM,
                                   dac_scale=dac_scale)
            
    except Exception as e:
        print(f"\n❌ Error occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Stop the simulation this script started. Irrelevant when it is run
        # as a script — the process exits and takes the streamer with it —
        # but main() is also awaited in-process by the test suite, and a mock
        # left streaming to 127.0.0.1:9876 then interleaves with whatever
        # simulation the next test creates. Both send valid packets to the
        # same port, so the reader mixes two detectors with nothing raised.
        if is_mock and crs is not None:
            try:
                await crs.stop_udp_streaming()
            except Exception:
                pass

    return 0


async def run_algorithm_flow(crs, MODULE, NETANAL_PARAMS, FIND_RES_PARAMS,
                           MULTISWEEP_PARAMS, FIT_PARAMS, SAMPLE_PARAMS, full_run = True,
                           *, expected_resonances=None, collect_pfb=False,
                           pfb_samples=100_000, is_mock=False, BIAS_PARAMS=None,
                           multisweep_powers_dbm=None, dac_scale=None):
    """Run the complete algorithm flow with the given CRS instance.

    expected_resonances : int, optional
        Number of resonances the array is known to have — only true of a
        simulation, where it is the whole point of checking. Left None on
        real hardware, where the count is the measurement, not a given.
    collect_pfb : bool
        Run step 9. Off by default so partial flows (and tests driving
        this with mocks) stop after the slow-stream noise.
    is_mock : bool
        Label the PFB numbers as synthetic. MockCRS.get_pfb_samples
        returns uniform noise rather than simulated detector output, so
        the step exercises the call and nothing more.
    """
    
    # Step 2: Network Analysis
    print("\n2. Performing network analysis...")
    print(f"   Sweeping {NETANAL_PARAMS['fmin']/1e6:.1f} - {NETANAL_PARAMS['fmax']/1e6:.1f} MHz")
    
    # Progress callback for network analysis
    def netanal_progress_callback(module, percentage):
        print(f"   Network analysis progress: {percentage:.1f}%", end='\r')
    
    NETANAL_PARAMS['progress_callback'] = netanal_progress_callback
    
    netanal_result = await crs.take_netanal(**NETANAL_PARAMS)
    
    frequencies = netanal_result['frequencies']
    iq_complex = netanal_result['iq_complex']
    phase_degrees = netanal_result['phase_degrees']
    
    print(f"   ✓ Network analysis complete: {len(frequencies)} points")
    
    # Step 3: Unwrap cable delay
    print("\n3. Unwrapping cable delay...")
    
    # Fit cable delay from phase data
    tau_additional = fit_cable_delay(frequencies, phase_degrees)
    
    # Calculate new cable length (assume current is 0m for simplicity)
    current_cable_length = await crs.get_cable_length(module=MODULE)
    new_cable_length = calculate_new_cable_length(current_cable_length, tau_additional)
    
    # Set the cable length on the CRS
    await crs.set_cable_length(length=new_cable_length, module=MODULE)
    
    print(f"   ✓ Cable delay unwrapped: delay={tau_additional*1e9:.2f} ns")
    print(f"   ✓ Cable length set to: {new_cable_length:.2f} m")
    
    # Step 4: Find resonances
    print("\n4. Finding resonances...")
    
    resonance_result = find_resonances(
        frequencies=frequencies,
        iq_complex=iq_complex,
        **FIND_RES_PARAMS,
        module_identifier=f"Module {MODULE}"
    )
    
    resonance_frequencies = resonance_result['resonance_frequencies']
    resonance_details = resonance_result['resonances_details']

    if full_run and expected_resonances is not None:
        assert len(resonance_frequencies) == expected_resonances, (
            f"found {len(resonance_frequencies)} resonances, expected "
            f"{expected_resonances} — the sweep or the detection "
            f"parameters have drifted")

    print(f"   ✓ Found {len(resonance_frequencies)} resonances:")
    for i, (freq, details) in enumerate(zip(resonance_frequencies, resonance_details)):
        print(f"     {i+1}: {freq/1e6:.3f} MHz, Q≈{details['q_estimated']:.0f}, depth={details['prominence_db']:.1f} dB")
    
    if not resonance_frequencies:
        print("   ! No resonances found. Adjust parameters and try again.")
        return
    
    # Step 5: Take multisweep
    print("\n5. Performing multisweep measurements...")
    print(f"   Sweeping around {len(resonance_frequencies)} resonances")
    
    # Set center frequencies for multisweep
    MULTISWEEP_PARAMS['center_frequencies'] = resonance_frequencies
    
    # Progress callback (optional)
    def progress_callback(module, percentage):
        print(f"   Progress: {percentage:.1f}%", end='\r')
    
    MULTISWEEP_PARAMS['progress_callback'] = progress_callback

    # One sweep of every resonator per power, keyed by detector and then
    # by power index, as bias_kids expects.  With no powers given, one
    # sweep runs at MULTISWEEP_PARAMS['amp'].
    if multisweep_powers_dbm:
        amps = [convert_dbm_to_amplitude(dbm, dac_scale)
                for dbm in multisweep_powers_dbm]
    else:
        amps = [MULTISWEEP_PARAMS.pop('amp')]
    sweeps_by_detector = {}
    for k, amp in enumerate(amps):
        print(f"   Sweep {k + 1}/{len(amps)}: amplitude {amp:.3g}"
              + (f" ({multisweep_powers_dbm[k]:g} dBm)"
                 if multisweep_powers_dbm else ""))
        sweep = await crs.multisweep(amp=amp, **MULTISWEEP_PARAMS)
        for det, entry in sweep.items():
            entry['amplitude'] = amp
            entry['direction'] = MULTISWEEP_PARAMS.get('sweep_direction', 'upward')
            sweeps_by_detector.setdefault(det, {})[k] = entry

    # The fits below are shown for the sweeps at the middle power.
    multisweep_results = {det: by_power[len(amps) // 2]
                          for det, by_power in sweeps_by_detector.items()}

    print(f"\n   ✓ Multisweep complete for {len(multisweep_results)} resonances"
          f" at {len(amps)} power{'s' if len(amps) != 1 else ''}")
    
    # Step 6: Do fitting
    print("\n6. Performing resonance fitting...")
    
    # Apply skewed Lorentzian fitting
    if FIT_PARAMS['apply_skewed_fit']:
        print("   Applying skewed Lorentzian fits...")
        multisweep_results = fit_skewed_multisweep(
            multisweep_results,
            approx_Q_for_fit=FIT_PARAMS['approx_Q_for_fit'],
            fit_resonances=FIT_PARAMS['fit_resonances'],
            center_iq_circle=FIT_PARAMS['center_iq_circle'],
            normalize_fit=FIT_PARAMS['normalize_fit']
        )
        
        # Count successful fits
        successful_fits = sum(1 for res_data in multisweep_results.values() 
                            if res_data.get('fit_params', {}).get('fr') != 'nan')
        print(f"   ✓ Skewed fitting complete: {successful_fits}/{len(multisweep_results)} successful")
    
    # Apply nonlinear fitting (optional)
    if FIT_PARAMS['apply_nonlinear_fit']:
        print("   Applying nonlinear fits...")
        multisweep_results = fit_nonlinear_iq_multisweep(
            multisweep_results,
            fit_nonlinearity=True,
            n_extrema_points=5,
            verbose=False
        )
        
        # Count successful nonlinear fits
        nl_successful = sum(1 for res_data in multisweep_results.values() 
                          if res_data.get('nonlinear_fit_success', False))
        print(f"   ✓ Nonlinear fitting complete: {nl_successful}/{len(multisweep_results)} successful")
    
    # Display fit results
    print("\n   Fit results summary:")
    for idx, res_data in multisweep_results.items():
        if isinstance(idx, (int, np.integer)):
            fit_params = res_data.get('fit_params', {})
            if fit_params.get('fr') != 'nan':
                print(f"     Resonance {idx}: fr={fit_params['fr']/1e6:.3f} MHz, "
                      f"Qr={fit_params['Qr']:.0f}, Qc={fit_params['Qc']:.0f}")
    
    # Step 7: Bias the KIDs
    print("\n7. Biasing the KIDs...")
    
    # Progress callback for bias_kids
    def bias_progress_callback(module, percentage):
        print(f"   Bias progress: {percentage:.1f}%", end='\r')
    
    # bias_kids chooses, for each detector, the highest power that is not
    # bifurcated and whose fitted nonlinearity is under the threshold.
    bias_results = await bias_kids(
        crs=crs,
        multisweep_results={'results_by_detector': sweeps_by_detector},
        progress_callback=bias_progress_callback,
        **(BIAS_PARAMS or {'module': MODULE})
    )

    print(f"\n   ✓ Bias complete for {len(bias_results)} detectors")

    # Display bias results
    print("\n   Bias results summary:")
    for det_idx, det_data in sorted(bias_results.items()):
        if 'bias_frequency' not in det_data:
            continue
        fit = det_data.get('nonlinear_fit_params') or {}
        amp = det_data.get('sweep_amplitude')
        power = ("not recorded" if amp is None
                 else f"{convert_amplitude_to_dbm(amp, dac_scale):.1f} dBm"
                 if dac_scale is not None else f"amplitude {amp:.3g}")
        print(f"     Detector {det_idx}: channel {det_data.get('bias_channel')}, "
              f"power {power}, bias_freq={det_data['bias_frequency']/1e6:.3f} MHz")
        print(f"                      a={fit.get('a', float('nan')):.2f}, "
              f"Qr={fit.get('Qr', float('nan')):.0f}"
              + (", bifurcated at one of the powers"
                 if det_data.get('bifurcation_ever_seen') else ""))
        if det_data.get('df_calibration') is not None:
            print(f"                      |df_cal|={abs(det_data['df_calibration']):.3e} Hz/V"
                  f" ({det_data.get('df_calibration_source', '')})")
    
    
    ### Step 8. Slow Noise spectrum 
    print("\n8. Collecting 1000 slow noise samples...")

    slow_data = await crs.py_get_samples(**SAMPLE_PARAMS)

    num_res = len(bias_results)

    print(f"\nSlow Noise data collected for the {num_res} biased channels")

    print("\n  Noise results summary:")

    for i in range(num_res):
        psd_i = slow_data.spectrum.psd_i[i]
        psd_q = slow_data.spectrum.psd_q[i]
        freq_iq = slow_data.spectrum.freq_iq
        freq_dsb = slow_data.spectrum.freq_dsb

        mean_psd_i = np.mean(psd_i[2:]) ### removing DC
        mean_psd_q = np.mean(psd_q[2:])

        print(f"   Channel {i+1}: Frequency Bandwidth = {max(freq_iq):.3f} Hz, Min dsb frequency = {min(freq_dsb):.3f} Hz, Max dsb frequency = {max(freq_dsb):.3f} Hz")
        print(f"                  Mean I spectrum power = {mean_psd_i:.3f} dBm/Hz, Mean Q spectrum power = {mean_psd_q:.3f} dBm/Hz\n")

    #### Step 9. PFB noise spectrum
    if not collect_pfb:
        print("\nSkipping PFB noise samples (collect_pfb=False)")
    else:
        print(f"\n9. Collecting {pfb_samples} PFB noise samples....")
        if is_mock:
            print("   ! Simulated PFB samples are uniform noise, not detector"
                  " output — the spectra below measure nothing physical.")

        for i in range(num_res):
            pfb_data = await crs.py_get_pfb_samples(pfb_samples,
                                                    channel = i + 1,
                                                    module = MODULE,
                                                    binlim = 1e6,
                                                    trim = False,
                                                    nsegments = 5,
                                                    reference = "absolute",
                                                    reset_NCO = False)
            
            # print(f"Pfb data collected for channel {i+1}")
            
            pfb_psd_i = pfb_data.spectrum.psd_i
            pfb_psd_q = pfb_data.spectrum.psd_q
            pfb_freq_iq = pfb_data.spectrum.freq_iq
            pfb_freq_dsb = pfb_data.spectrum.freq_dsb

            mean_pfb_psd_i = np.mean(pfb_psd_i[2:]) ### Removing dc bin
            mean_pfb_psd_q = np.mean(pfb_psd_q[2:])

            print(f"   Channel {i+1}: Frequency Bandwidth = {max(pfb_freq_iq):.3f} Hz, Min dsb frequency = {min(pfb_freq_dsb):.3f} Hz, Max dsb frequency = {max(pfb_freq_dsb):.3f} Hz")
            print(f"                  Mean I spectrum power = {mean_pfb_psd_i:.3f} dBm/Hz, Mean Q spectrum power = {mean_pfb_psd_q:.3f} dBm/Hz\n")
            
    print("\n" + "="*60)
    print("Algorithm flow complete!")
    print(f"Finished at: {datetime.now()}")
    print("="*60)

if __name__ == "__main__":
    # Get serial number from command line or default to MOCK
    if len(sys.argv) > 1:
        serial = sys.argv[1]
    else:
        serial = "MOCK"
        print("No serial number provided, using MOCK mode")
    
    # Run the async main function
    exit_code = asyncio.run(main(serial=serial))
    exit(exit_code)
