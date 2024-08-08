import os
import pytest
import rfmux
import time
import numpy as np
import asyncio
import pytest_asyncio
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
'''
Welcome to the 1st draft of the CRS QC. To begin testing, type into the terminal 
$pytest test_crs_integrated.py --serial=<serial> -s
'''
# Ensure the board and the computer are connected to the same network.
# Most basic check, does not use hidfmux
def test_board_bootup(results):
    print("Retrieving board status")
    os.system("timeout 1 /home/ssavchyn/mkids/crs-mkids/firmware/rfmux/r1.3/parser -i enp1s0f0 -t | grep 'Enclosure\\|Timestamp' > raw1.txt")
    try:
        with open('raw1.txt', 'r') as file:
            content = file.read()
    except FileNotFoundError:
        pytest.fail("No file was generated. Check your parser directory")
        return
    assert content, "Error: No data received. The board might not be booted correctly."
    print("Board boot status: OK")
    results['tests'].append({
        'title': 'Board Bootup Test',
        'status': 'OK'
        })
   
'''-------------------------------------HELPER FUNCTIONS---------------------------------------------'''
def transfer_function_helper(raw_sampl, AMPLITUDE, scale, attenuation):
    DB_LOSS_FACTOR = -2.56
    volts_peak_samples = (np.sqrt(2)) * np.sqrt(50 * (10 ** (-1.75 / 10)) / 1000) / 1880796.4604246316
    data_i = np.array(raw_sampl.i) * volts_peak_samples
    data_q = np.array(raw_sampl.q) * volts_peak_samples
    magnitude = np.sqrt(data_i ** 2 + data_q ** 2)
    median_magnitude = np.median(magnitude)
    median_magnitude_dbm = 10 * np.log10(((median_magnitude / np.sqrt(2)) ** 2) / 50) + 30
    DAC_output_dBm = DB_LOSS_FACTOR + 10 * np.log10((AMPLITUDE ** 2) * 10 ** (scale / 10))
    predicted_magnitude_dbm = DAC_output_dBm - attenuation
    return median_magnitude_dbm, predicted_magnitude_dbm

def value_setter_helper(d, frequency, amplitude, scale, attenuation, channel, module, nco, target_dac, target_adc, highbank):
    d.clear_channels()
    d.set_frequency(frequency, d.UNITS.HZ, channel, module)
    d.set_amplitude(amplitude, d.UNITS.NORMALIZED, target_dac, channel, module)

    if highbank == True:
        module+=4
    d.set_dac_scale(scale, d.UNITS.DBM, module)
    d._set_adc_attenuator(attenuation, d.UNITS.DB, module=module)
    d.set_nco_frequency(nco, d.UNITS.HZ, target_dac, module=module)
    d.set_nco_frequency(-nco, d.UNITS.HZ, target_adc, module=module)

def covariance_matrix(samples):
    cov_matrix = np.cov(samples, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    ratio = eigenvalues[1] / eigenvalues[0]
    return ratio
'''------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------SENSORS--------------------------------------------------------------'''
def test_temperature(hwm, results, summary_results):
    TEMPERATURE_MIN = 0.0
    TEMPERATURE_MAX = 70.0
    d = hwm.query(rfmux.CRS).one()
    errors = []
    test_status = 'Pass'

    sensors = (
        d.TEMPERATURE_SENSOR.MB_R5V0,
        d.TEMPERATURE_SENSOR.MB_R3V3A,
        d.TEMPERATURE_SENSOR.MB_R2V5,
        d.TEMPERATURE_SENSOR.MB_R1V8,
        d.TEMPERATURE_SENSOR.MB_R1V2A,
        d.TEMPERATURE_SENSOR.MB_R1V4,
        d.TEMPERATURE_SENSOR.MB_R1V2B,
        d.TEMPERATURE_SENSOR.MB_R0V85A, 
        d.TEMPERATURE_SENSOR.MB_R0V85B, 
        d.TEMPERATURE_SENSOR.RFSOC_PL, 
        d.TEMPERATURE_SENSOR.RFSOC_PS,
    )
    with d.tuber_context() as ctx:
        for s in sensors:
            ctx.get_motherboard_temperature(s)
        temps = ctx()
    for (s, v) in zip(sensors, temps):
        if not (TEMPERATURE_MIN <= v <= TEMPERATURE_MAX):
            errors.append(f"Sensor {s} with value {v} out of range {TEMPERATURE_MIN}, {TEMPERATURE_MAX}!")
            test_status = 'Fail'
        print(f"Temperature for sensor {s}: {v}")
    results['tests'].append({
        'title': 'Temperature Test',
        'sensors': [str(sensor) for sensor in sensors],
        'temperatures': temps,
        'errors': errors,
        'test_status': test_status
    })
    if 'Board Health' not in summary_results['summary']:
        summary_results['summary']['Board Health'] = {}
    summary_results['summary']['Board Health']['Temperature Test'] = test_status
    
def test_voltages(hwm, results,summary_results):
    d = hwm.query(rfmux.CRS).one()
    VOLTAGE_TOL = 0.10  # 10%
    test_status = 'Pass'
    sensors = (
        (d.VOLTAGE_SENSOR.MB_VBP, 12),  # power supply
        (d.VOLTAGE_SENSOR.MB_R0V85A, 0.85),
        (d.VOLTAGE_SENSOR.MB_R0V85B, 0.85),
        (d.VOLTAGE_SENSOR.MB_R1V2A, 1.2),
        (d.VOLTAGE_SENSOR.MB_R1V2B, 1.2),
        (d.VOLTAGE_SENSOR.MB_R1V4, 1.4),
        (d.VOLTAGE_SENSOR.MB_R1V8A, 1.8),
        (d.VOLTAGE_SENSOR.MB_R2V5, 2.5),
        (d.VOLTAGE_SENSOR.MB_R3V3A, 3.3),
        (d.VOLTAGE_SENSOR.MB_R5V0, 5.0),
        (d.VOLTAGE_SENSOR.RFSOC_PSMGTRAVCC, 0.9),
        (d.VOLTAGE_SENSOR.RFSOC_PSMGTRAVTT, 1.8),
        (d.VOLTAGE_SENSOR.RFSOC_VCCAMS, 1.8),
        (d.VOLTAGE_SENSOR.RFSOC_VCCAUX, 1.8),
        (d.VOLTAGE_SENSOR.RFSOC_VCCBRAM, 0.85),
        (d.VOLTAGE_SENSOR.RFSOC_VCCINT, 0.85),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPLAUX, 1.8),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPLINTFP, 0.85),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPLINTLP, 0.85),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSAUX, 1.8),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSDDR, 1.2),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSINTFP, 0.85),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSINTFPDDR, 0.85),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSINTLP, 0.85),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSIO0, 1.8),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSIO1, 3.3),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSIO2, 3.3),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSIO3, 3.3),
        (d.VOLTAGE_SENSOR.RFSOC_VCCVREFN, 0.05),
        (d.VOLTAGE_SENSOR.RFSOC_VCCVREFP, 1.85),
        (d.VOLTAGE_SENSOR.RFSOC_VCC_PSBATT, 3.0),
        (d.VOLTAGE_SENSOR.RFSOC_VCC_PSDDRPLL, 1.8),
        (d.VOLTAGE_SENSOR.RFSOC_VCC_PSPLL0, 1.2),
    )
    with d.tuber_context() as ctx:
        for (s, nv) in sensors:
            ctx.get_motherboard_voltage(s)
        voltages = ctx()
    for ((s, nv), v) in zip(sensors, voltages):
        min_v = nv * (1 - VOLTAGE_TOL)
        max_v = nv * (1 + VOLTAGE_TOL)
        if v > max_v:
            print(f"Sensor {s} with value {v} A above {max_v}!")
        if v < min_v:
            print(f"Sensor {s} with value {v} A below {min_v}!")

        if  v <=min_v or v>= max_v:
            f"Sensor {s} with value {v} out of range {min_v}, {max_v}!"
            test_status = 'Fail'
        #print(f"Voltage for sensor {s}: {v}")

    results['tests'].append({
        'title': 'Voltage Test',
        'sensors': [str(sensor[0]) for sensor in sensors],
        'voltages': voltages,
        'test_status': test_status
    })

    if 'Board Health' not in summary_results['summary']:
        summary_results['summary']['Board Health'] = {}
    summary_results['summary']['Board Health']['Voltage Test'] = test_status

def test_currents(hwm, results, summary_results):
    d = hwm.query(rfmux.CRS).one()
    CURRENT_LIMIT = 0.1  # 10% tolerance on the current limit
    test_status = 'Pass'
    sensors = (
        (d.CURRENT_SENSOR.MB_R0V85A, 2.5),
        (d.CURRENT_SENSOR.MB_R0V85B, 3.3),
        (d.CURRENT_SENSOR.MB_R1V2A, 2.5),
        (d.CURRENT_SENSOR.MB_R1V2B, 2.6),
        (d.CURRENT_SENSOR.MB_R1V4, 3.6),
        (d.CURRENT_SENSOR.MB_R1V8A, 3.3),
        (d.CURRENT_SENSOR.MB_R2V5, 4.1),
        (d.CURRENT_SENSOR.MB_R3V3A, 3.4),
        (d.CURRENT_SENSOR.MB_R5V0, 4.1),
        (d.CURRENT_SENSOR.MB_VBP, 4), #per molex: max 15A contact
    )
    with d.tuber_context() as ctx:
        for (s, nc) in sensors:
            ctx.get_motherboard_current(s)
        currents = ctx()
    for ((s, nc), v) in zip(sensors, currents):
        min_c = nc * (1 - CURRENT_LIMIT)
        max_c = nc * (1 + CURRENT_LIMIT)
        if v > max_c:
            print(f"Sensor {s} with value {v} A above {max_c}!")
        if v < min_c:
            print(f"Sensor {s} with value {v} A below {min_c}!")

            test_status = 'Fail'
        #print(f"Current for sensor {s}: {v}")
    results['tests'].append({
        'title': 'Current Test',
        'sensors': [str(sensor[0]) for sensor in sensors],
        'currents': currents,
        'test_status': test_status
    })
    if 'Board Health' not in summary_results['summary']:
        summary_results['summary']['Board Health'] = {}
    summary_results['summary']['Board Health']['Current Test'] = test_status
'''------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------DAC--------------------------------------------------------------'''
@pytest.mark.asyncio
async def test_dac_passband(hwm, results, summary_results, num_runs, test_highbank):

    '''This test aims to determine 2 things: if the magnitude of the output signal form the DAC 
    is within expected magnitude and if the passband stays flat throughout the predicted passband
    
    AMPLITUDE - Intentionally chosen to be 0.001 to be able to set 1000 channels at once without exceeding the full scale
 

    '''

    d= hwm.query(rfmux.CRS).one()

    AMPLITUDE = 0.001
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 100
    NCO_FREQUENCY = 625e6
    ATTENUATION = 0
    SCALE = 7
    NOMINAL_AMPLITUDE = 0.5
    NOMINAL_FREQUENCY = 200e6
    NOMINAL_CHANNEL = 1

    '''
    Measured values if ever want to compare
        test_values = np.array([-13.17, -8.58, -8.04, -8.91, -9.38, -10.03,  -14.42]) [in dBm]
        frequencies = [ -300e6, -275e6,  -150e6,  0, 150e6, 275e6, 300e6] [in HZ]
        '''
    #feel free to increase the range and density of frequencies
    frequencies = np.arange(-300e6, 310e6, 10e6)
    channels = np.arange(1, 1001)
    nruns = int(np.ceil(len(frequencies) / len(channels)))
    lenrun = min(len(frequencies), len(channels))
    test_result = {
        'title': 'DAC Passband Test',
        'modules': []
    }
    highbank_run = False #test the high analog banks is false 

    for num_run in range(0, num_runs):
        for module in range(1, 5):
            module_actual = module
            
            if highbank_run:
                module_actual +=4

            print(f'DAC_passband- Testing Module #:{module_actual}')
            
            measured_dbm = []
            predicted_dbm = []
            module_errors = []
            for nrun in range(nruns):
                start_idx = nrun * lenrun
                end_idx = min(start_idx + lenrun, len(frequencies))
                d.clear_channels()
                #First, call the function to set all the parameters that will , nconot change: attenuation, scale
                value_setter_helper(d, NOMINAL_FREQUENCY,NOMINAL_AMPLITUDE, SCALE, ATTENUATION, NOMINAL_CHANNEL, module, NCO_FREQUENCY, TARGET_DAC, TARGET_ADC, highbank_run)
                
                #Set all the nessesary channels for the run
                for idx in range(start_idx, end_idx):
                    channel = int(channels[idx - start_idx])  # Convert to int and access correct channel
                    frequency = float(frequencies[idx])  # Convert to float
                    d.set_frequency(frequency, d.UNITS.HZ, channel, module)
                    d.set_amplitude(AMPLITUDE, d.UNITS.NORMALIZED, TARGET_DAC, channel, module)
                time.sleep(0.2)

                # Get all the samples for the current run
                for idx in range(start_idx, end_idx):
                    channel = int(channels[idx - start_idx])  # Convert to int and access correct channel

                    frequency = float(frequencies[idx])  # Ensure frequency is a float
                    raw = d.get_samples(SAMPLES, channel=channel, module=module)
                    

                    measured, predicted = transfer_function_helper(raw, AMPLITUDE, SCALE, ATTENUATION)
                    measured_dbm.append((frequency, measured))
                    predicted_dbm.append((frequency, predicted))

            mag_0 = next(meas for freq, meas in measured_dbm if freq == 0)
            
            if abs(mag_0-predicted) > 3:
                error = f"At a reference frequency (625MHz), generating a power output lower than predicted. Expected value is: {predicted}. Measured value was: {mag_0:.2f}"
                print(error)
                module_errors.append(error)

            for freq, meas in measured_dbm:
                if abs(freq) <= 275e6 and meas <= mag_0 - 3:
                    error = f"Taking 0Hz as reference, there are significant gaps in the passband.Generating signals below -3dB point relative to 0Hz."
                    print(error)
                    module_errors.append(error)
                    break

            if not module_errors:
                print(f'Module {module_actual} passed the test')   

            module_result = {
                'module': module_actual,  # Ensure this is an int
                'frequencies': [freq for freq, _ in measured_dbm],  # Convert frequencies to float
                'measured_dbm': [meas for _, meas in measured_dbm],  # Ensure this is a float
                'predicted_magnitude_dbm_values': [predicted for _, predicted in predicted_dbm],  # Convert to a list of floats
                'status': 'Fail' if module_errors else 'Pass',
                'error': '\n'.join(module_errors)
            }
            test_result['modules'].append(module_result)

            if module_actual not in summary_results['summary']:
                summary_results['summary'][module_actual] = {}
            summary_results['summary'][module_actual]['Passband'] = module_result['status']

        if test_highbank:
            d.set_analog_bank(True)
            highbank_run = True
    
    d.set_analog_bank(False)
    results['tests'].append(test_result)


def test_dac_amplitude_transfer(hwm, results, summary_results, test_highbank, num_runs):
    d = hwm.query(rfmux.CRS).one()
    print('start')
    SCALE = 1
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 100
    ATTENUATION = 0
    CHANNEL = 1
    NCO_FREQUENCY = 0
    FREQUENCY = 150.5763e6
    NOMINAL_AMPLITUDE = 0
    NOMINAL_CHANNEL = 1
    test_result = {
        'title': 'DAC Amplitude Transfer Test',
        'modules': []
    }

    highbank_run = False
    amplitudes = np.arange(0.01, 1, 0.01)

    for num_run in range(0, num_runs):
        for module in range(1, 5):
            module_errors = []
            amplitude_values = []
            magnitude_dbm_values = []
            predicted_magnitude_dbm_values = []

            if highbank_run:
                print(f'DAC_amplitude - Testing Module #:{module+4}')
            else:
                print(f'DAC_amplitude - Testing Module #:{module}')
            
            d.clear_channels()
            value_setter_helper(d, FREQUENCY, NOMINAL_AMPLITUDE, SCALE, ATTENUATION, NOMINAL_CHANNEL, module, NCO_FREQUENCY, TARGET_DAC, TARGET_ADC, highbank_run)
            for amplitude in amplitudes:
                d.set_amplitude(amplitude, d.UNITS.NORMALIZED, TARGET_DAC, CHANNEL, module)
                samples = d.get_samples(SAMPLES, channel=CHANNEL, module=module)
                median_magnitude_dbm, predicted_magnitude_dbm = transfer_function_helper(samples, amplitude, SCALE, ATTENUATION)
                amplitude_values.append(amplitude)
                magnitude_dbm_values.append(median_magnitude_dbm)
                predicted_magnitude_dbm_values.append(predicted_magnitude_dbm)
            errors = np.abs(np.array(magnitude_dbm_values) - np.array(predicted_magnitude_dbm_values))
            if highbank_run:
                module +=4

            #Linearity check on the scaling: apply linear regression and check least square error
            log_amplitude_values = np.array(np.log10(amplitude_values)).reshape(-1, 1)
            regressor = LinearRegression().fit(log_amplitude_values, magnitude_dbm_values)
            linearity = r2_score(regressor.predict(log_amplitude_values), magnitude_dbm_values)
            print(linearity)

            if np.any(errors >= 3):
                error = f'Generating a power output below 3dB of expected.'
                print(f'Module {module} {error}')
                module_errors.append(error)

            if linearity<0.999:
                error = 'The output is not linear on a lin-log scale.'
                print(error)
                module_errors.append(error)

            if not module_errors:
                print(f"Module {module} passed DAC amplitude test.")
            
            module_result = {
                'module': module,
                'amplitude_values': amplitude_values,
                'magnitude_dbm_values': magnitude_dbm_values,
                'predicted_magnitude_dbm_values': predicted_magnitude_dbm_values,
                'status': 'Fail' if module_errors else 'Pass',
                'error': '\n'.join(module_errors)
            }
            test_result['modules'].append(module_result)
            if module not in summary_results['summary']:
                summary_results['summary'][module] = {}
            summary_results['summary'][module]['DAC Ampl'] = module_result['status']

        if test_highbank:
            d.set_analog_bank(True)
            highbank_run = True
            
    d.set_analog_bank(False)
    results['tests'].append(test_result)


def test_dac_scale_transfer(hwm, results, summary_results, test_highbank, num_runs):
    '''
   LATER:  check if there's an offset in the transfer funciton (set gian amplitude = 1)
    Pick 4 channels, set the same frequency-> alter scale and plot the samples first, then convert to dBm
    plot together with prediction from the transfer funciton
    '''
    d = hwm.query(rfmux.CRS).one()
    #set parameters for testing
    AMPLITUDE= 1
    NOMINAL_SCALE = 1
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 100
    FREQUENCY = 150.5763e6
    NCO_FREQUENCY = 0
    ATTENUATION = 0
    CHANNEL = 1
    test_result = {
        'title': 'DAC Scale Transfer Test',
        'modules': []
    }
    for mod in range (1,4):
        d.set_adc_calibration_mode('MODE1', mod)

    highbank_run = False
    for num_run in range (0, num_runs):
        for module in range (1,5):
            # Initialize lists to store the scale and magnitude values
            module_errors = []
            scale_values = []
            magnitude_dbm_values = []
            predicted_magnitude_dbm_values = []

            d.clear_channels()
            value_setter_helper(d, FREQUENCY, AMPLITUDE, NOMINAL_SCALE, ATTENUATION, CHANNEL, module, NCO_FREQUENCY, TARGET_DAC, TARGET_ADC, highbank_run)
            module_sample = module

            if highbank_run:
                module_set= module +4
            else:
                module_set = module
            
            print(f'DAC_scale - Testing Module #: {module_set}')

            for scale in range (0, 8):
                
                d.set_dac_scale(scale, d.UNITS.DBM, module_set)
                raw_sampl = d.get_samples(SAMPLES, channel=CHANNEL, module=module_sample)
                median_magnitude_dbm, predicted_magnitude_dbm = transfer_function_helper(raw_sampl, AMPLITUDE, scale, ATTENUATION)
                
                scale_values.append(scale)
                magnitude_dbm_values.append(median_magnitude_dbm)
                predicted_magnitude_dbm_values.append(predicted_magnitude_dbm)

            #Power check on the DAC
            errors = np.abs(np.array(magnitude_dbm_values) - np.array(predicted_magnitude_dbm_values))
            
            if np.any(errors >= 3):
                error = f'Module {module_set} is generating a power output below 3dB of expected'
                print(error)
                module_errors.append(error)
            #Linearity check on the scaling: apply linear regression and check least square error
            scale_values = np.array(scale_values).reshape(-1, 1)
            regressor = LinearRegression().fit(scale_values, magnitude_dbm_values)
            linearity = r2_score(regressor.predict(scale_values), magnitude_dbm_values)
            if linearity<0.99:
                error = 'linearity check failed'
                module_errors.append(error)

            else:
                print(f'Module {module_set} passed the DAC scaling test')
                    
            module_result = {
                'module': module_set,
                'scale_values': scale_values.flatten().tolist(),
                'magnitude_dbm_values': magnitude_dbm_values,
                'predicted_magnitude_dbm_values': predicted_magnitude_dbm_values,
                'status': 'Fail' if module_errors else 'Pass',
                'errors': f'{module_errors}'
            }

            test_result['modules'].append(module_result)
            if module_set not in summary_results['summary']:
                summary_results['summary'][module_set] = {}
            summary_results['summary'][module_set]['DAC Scale'] = module_result['status']

        if test_highbank:
            d.set_analog_bank(True)
            highbank_run = True

    d.set_analog_bank(False)
    results['tests'].append(test_result)


def test_dac_mixmodes(hwm, results, summary_results):
    d = hwm.query(rfmux.CRS).one()
    MAX_TOTAL_FREQUENCY = 2.5e9  # in Hz
    DAC_MAX_FREQUENCY = 275e6  # in Hz (max DAC frequency)
    BANDWIDTH = 550e6  # in Hz (275 MHz on each side of NCO)
    AMPLITUDE = 1
    SCALE = 1
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 100
    ATTENUATION = 0
    STEP = 50e6
    SAMPLING_FREQUENCY = 2.5e9  # in Hz
    MODULE_NYQUIST_MAP = {1: 1, 2: 2, 3: 3}
    CHANNEL = 1
    TEST_HIGHBANK = False


    reference_data = {
        'mixmode_1': {
            'mod_region_1': {
                'frequencies': [761.78e6, 1302.56e6, 1902.34e6],
                'magnitudes': [-3.42, -4.82, -6.7]
            },
            'mod_region_2': {
                'frequencies': [4338.22e6, 3697.44e6, 3097.66e6],
                'magnitudes': [-24.87, -19.00, -13.82]
            },
            'mod_region_3': {
                'frequencies': [5761.78e6, 6302.56e6, 6902.34e6],
                'magnitudes': [-31.89, -27.84, -27.87]
            }
        },
        'mixmode_2': {
            'mod_region_1': {
                'frequencies': [761.78e6, 1302.56e6, 1902.34e6],
                'magnitudes': [-16.41, -12.24, -10.21]
            },
            'mod_region_2': {
                'frequencies': [4238.22e6, 3697.44e6, 3097.66e6],
                'magnitudes': [-12.30, -11.73,-10.01 ]
            },
            'mod_region_3': {
                'frequencies': [5761.78e6, 6302.56e6, 6902.34e6],
                'magnitudes': [-19.66, -20.60, -24.93]
            }
        }
    }

    test_result = {
        'title': 'DAC Mixmodes Test',
        'mixmodes': []
    }

    for mixmode in range(1, 3):
        for module, nyquist_zone in MODULE_NYQUIST_MAP.items():
            module_errors = []
            module_magnitude_dbm_values = np.array([])
            frequencies = np.array([])
            print(f'Running module {module} in nyquist region {nyquist_zone}')

            d.set_nyquist_zone(mixmode, module=module)
            nco = 0
            frequency = -DAC_MAX_FREQUENCY

            while nco < MAX_TOTAL_FREQUENCY:
                while frequency <= DAC_MAX_FREQUENCY and nco + frequency < MAX_TOTAL_FREQUENCY:
                    if nco + frequency >= 0:
                        value_setter_helper(d, frequency, AMPLITUDE, SCALE, ATTENUATION, CHANNEL, module, nco, TARGET_DAC, TARGET_ADC, TEST_HIGHBANK)
                        samples = d.get_samples(SAMPLES, channel=CHANNEL, module=module)
                        module_magnitude_dbm, _ = transfer_function_helper(samples, AMPLITUDE, SCALE, ATTENUATION)

                        if nyquist_zone == 1:
                            frequencies = np.append(frequencies, frequency + nco)
                            module_magnitude_dbm_values = np.append(module_magnitude_dbm_values, module_magnitude_dbm)
                        elif nyquist_zone == 2:
                            frequencies = np.append(frequencies, 2* SAMPLING_FREQUENCY - (frequency + nco))
                            module_magnitude_dbm_values = np.append(module_magnitude_dbm_values, module_magnitude_dbm)
                        elif nyquist_zone == 3:
                            frequencies = np.append(frequencies, 2 * SAMPLING_FREQUENCY + frequency + nco)
                            module_magnitude_dbm_values = np.append(module_magnitude_dbm_values, module_magnitude_dbm)
                    frequency += STEP
                nco += BANDWIDTH
                frequency = -DAC_MAX_FREQUENCY

            reference_freqs = np.array(reference_data[f'mixmode_{mixmode}'][f'mod_region_{nyquist_zone}']['frequencies'])
            reference_mags = np.array(reference_data[f'mixmode_{mixmode}'][f'mod_region_{nyquist_zone}']['magnitudes'])

            for reference_freq, reference_mag in zip(reference_freqs, reference_mags):
                if len(frequencies) == 0:
                    module_errors.append(f"No frequencies measured for module {module} in Nyquist region {nyquist_zone}")
                    continue

                closest_index = np.argmin(np.abs(frequencies - reference_freq))
                obtained_magnitude = module_magnitude_dbm_values[closest_index]

                if not (0.8 * abs(reference_mag) <= abs(obtained_magnitude) <= 1.15 * abs(reference_mag)):
                    module_errors.append(f"Frequency {frequencies[closest_index] / 1e6:.2f} MHz: measured {obtained_magnitude:.2f} dBm, expected {reference_freq/1e6:.2f} to have {reference_mag} dBm")
            
            if module_errors:
                print(module_errors)
            else:
                print('Module outputs within 15 percent of expected value')

            status = 'Fail' if module_errors else 'Pass'
            test_result['mixmodes'].append({
                'mixmode': mixmode,
                'module': module,
                'frequencies': frequencies.tolist(),
                'magnitude_dbm_values': module_magnitude_dbm_values.tolist(),
                'reference_frequencies':reference_freqs.tolist(),
                'reference_magnitudes':reference_mags.tolist(), 
                'status': status
            })
            if module not in summary_results['summary']:
                summary_results['summary'][module] = {}
            summary_results['summary'][module][f'Mixmode {mixmode}'] = status
            if mixmode == 2:
                d.set_nyquist_zone(1, module=module)

    results['tests'].append(test_result)





'''---------------------------------------------------------------------------------------------------------------'''
'''--------------------------------------------------ADC----------------------------------------------------------'''

def test_adc_attenuation(hwm, results, summary_results, test_highbank, num_runs):
    d = hwm.query(rfmux.CRS).one()
    AMPLITUDE = 1
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 100
    FREQUENCY = 80.234562e6
    NCO_FREQUENCY = 0
    SCALE = 1
    CHANNEL = 1
    ATTENUATION_0 = 0

    test_result = {
        'title': 'ADC Attenuation Test',
        'modules': []
    }

    highbank_run = False
    for num_run in range (0, num_runs):
        for module in range(1, 5):
            module_errors = []
            attenuation_values = []
            magnitude_dbm_values = []
            predicted_magnitude_dbm_values = []

            d.clear_channels()
            value_setter_helper(d, FREQUENCY, AMPLITUDE, SCALE, ATTENUATION_0, CHANNEL, module, NCO_FREQUENCY, TARGET_DAC, TARGET_ADC, highbank_run)

            module_sample = module
            if highbank_run:
                module_set= module +4
            else:
                module_set = module
            
            print(f'ADC_attenuation - Testing Module #: {module_set}')

            for attenuation in range(0, 15):
                d._set_adc_attenuator(attenuation, d.UNITS.DB, module=module_set)
                raw_sampl = d.get_samples(SAMPLES, channel=CHANNEL, module=module_sample)
                median_magnitude_dbm, predicted_magnitude_dbm = transfer_function_helper(raw_sampl, AMPLITUDE, SCALE, attenuation)
                attenuation_values.append(attenuation)
                magnitude_dbm_values.append(median_magnitude_dbm)
                predicted_magnitude_dbm_values.append(predicted_magnitude_dbm)

            attenuation_values = np.array(attenuation_values).reshape(-1, 1)
            regressor = LinearRegression().fit(attenuation_values, magnitude_dbm_values)
            linearity = r2_score(magnitude_dbm_values, regressor.predict(attenuation_values))

            if linearity < 0.99:
                error = f'linearity check failed'
                module_errors.append(error)
            if np.any(np.abs(np.array(magnitude_dbm_values) - np.array(predicted_magnitude_dbm_values)) >= 1):
                error = f'Module {module_set} is generating a power output lower than expected. Check your balun and SMA connections.'
                module_errors.append(error)

            else:
                print(f"Module {module_set} passed ADC attenuation tests.")
            module_result = {
                'module': module_set,
                'attenuation_values': attenuation_values.flatten().tolist(),
                'magnitude_dbm_values': magnitude_dbm_values,
                'predicted_magnitude_dbm_values': predicted_magnitude_dbm_values,
                'status': 'Fail' if module_errors else 'Pass',
                'errors': f'{module_errors}'
            }

            test_result['modules'].append(module_result)

            if module_set not in summary_results['summary']:
                summary_results['summary'][module_set] = {}
            summary_results['summary'][module_set][f'ADC Atten'] = module_result['status']   

        if test_highbank:
            d.set_analog_bank(True)
            highbank_run = True

    d.set_analog_bank(False)
    results['tests'].append(test_result)



@pytest.mark.skip(reason="no way of currently testing this")
def test_adc_even_phase(hwm, results):
    d = hwm.query(rfmux.CRS).one()
    AMPLITUDE = 0.1
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 100
    FREQUENCY = 1000
    NCO_FREQUENCY = 0
    SCALE = 1
    CHANNEL = 1
    module = 1
    test_result = {
        'title': 'ADC Even Phase Test',
        'modules': []
    }
    for module in range(1, 5):
        value_setter_helper(d, FREQUENCY, AMPLITUDE, SCALE, 0, CHANNEL, module, NCO_FREQUENCY, TARGET_DAC, TARGET_ADC)
        x = d.get_samples(SAMPLES, channel=1, module=1)
    
        data_i_samples = np.array(x.i)
        data_q_samples = np.array(x.q)
        data_samples = np.column_stack((data_i_samples, data_q_samples))
        ratio = covariance_matrix(data_samples)
        print(f"Eigenvalue ratio: {ratio}")
        module_result = {
            'module': module,
            'module': module,
            'data_i_samples': data_i_samples.tolist(),
            'data_q_samples': data_q_samples.tolist(),
            'ratio': ratio
        }
        test_result['modules'].append(module_result)
    
    results['tests'].append(test_result)
    
    if np.isclose(ratio, 1, atol=0.1):
        print("The cluster is approximately circular.")
    else:
        print("The cluster is more oval in shape.")

def psd_helper(data_i_raw, data_q_raw):
    ADC_SAMPLING_FREQUENCY = 625e6
    volts_peak = (np.sqrt(2)) * np.sqrt(50 * (10 ** (-1.75 / 10)) / 1000) / 1880796.4604246316
    # Convert I and Q to peak volts
    tod_i = data_i_raw * volts_peak
    tod_q = data_q_raw * volts_peak

    # Define the sampling frequency for the ADC
    fir = (ADC_SAMPLING_FREQUENCY / 256 / 64 / 2 ** 6)
    # Apply Welch's method to get the signal PSD
    psdfreq, psd_i = signal.welch(tod_i/np.sqrt(4*50), fir, nperseg=len(tod_i// 4)) #tod_i / np.sqrt(4 * 50) 
    _, psd_q = signal.welch(tod_q/np.sqrt(4*50), fir, nperseg=len(tod_q// 4)) #tod_q / np.sqrt(4 * 50)

    # Convert from watts to dBm
    psd_i = 10 * np.log10(psd_i*1000)
    psd_q = 10 * np.log10(psd_q*1000)
    return psd_i, psd_q, psdfreq
       

def test_loopback_noise(hwm, results, summary_results, test_highbank, num_runs):
    '''
    1. Send a signal with no amplitude to gauge the noise floor
    2. Determine the max point at low frequency and the slope of 1/f by putting in a high power signal
    3. Determine the max point in the white noise region
    '''
    d = hwm.query(rfmux.CRS).one()
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 10000

    # Set parameters for testing white noise floor
    AMPLITUDE_0 = 0
    FREQUENCY = 277.234562e6
    NCO_FREQUENCY = 0
    SCALE = 1
    ATTENUATION = 0
    CHANNEL_WHITE = 1

    # Set parameters for 1/f slope and peak noise
    AMPLITUDE_1_F = 1
    FREQUENCY_1_F = 277.987e6
    NCO_FREQUENCY_1_F = 0
    SCALE_1_F = 1
    ATTENUATION_1_F = 0
    CHANNEL_1_F = 784
    test_result = {
        'title': 'Loopback Noise Test',
        'modules': []
    }
    highbank_run = False
    for num_run in range (0, num_runs):
        for mod in range(1,5):

            if highbank_run:
                print(f'Loopback Noise - Testing Module #:{mod+4}')
            else:
                print(f'Loopback Noise - Testing Module #:{mod}')

            floor_errors = []
            loud_errors = []

            # 1. Test the noise environment: send a signal with no amplitude and plot the PSD of the samples
            d.clear_channels()
            value_setter_helper(d, FREQUENCY, AMPLITUDE_0, SCALE, ATTENUATION, CHANNEL_WHITE, mod, NCO_FREQUENCY, TARGET_DAC, TARGET_ADC, highbank_run)
            time.sleep(0.3)
            raw_samples_white = d.get_samples(SAMPLES, channel=CHANNEL_WHITE, module=mod)
            data_i_white = np.array(raw_samples_white.i)
            data_q_white = np.array(raw_samples_white.q)
            psd_i_white, psd_q_white, psdfreq_white = psd_helper(data_i_white, data_q_white)


            # 2. Test the PSD of a large signal, establishing an expectation for 1/f slope and peak noise at low frequency
            value_setter_helper(d, FREQUENCY_1_F, AMPLITUDE_1_F, SCALE_1_F, ATTENUATION_1_F, CHANNEL_1_F, mod, NCO_FREQUENCY_1_F, TARGET_DAC, TARGET_ADC, highbank_run)
            time.sleep(0.3)
            raw_samples_psd = d.get_samples(SAMPLES, channel=CHANNEL_1_F, module=mod)
            data_i_1_f = np.array(raw_samples_psd.i)
            data_q_1_f = np.array(raw_samples_psd.q)
            psd_i_1_f, psd_q_1_f, psdfreq_1_f = psd_helper(data_i_1_f, data_q_1_f)


            # Compute the max points in the white noise and low frequency regions
            max_point_white_i = np.max(psd_i_white)
            max_point_white_q = np.max(psd_q_white)
            max_point_1_f_i = np.max(psd_i_1_f)
            max_point_1_f_q = np.max(psd_q_1_f)
            mean_point_white_i = np.mean(psd_i_white)
            mean_point_white_q = np.mean(psd_q_white)

            if highbank_run:
                mod +=4
            
            #check if any spikes are present
            if 0.95*(-163.4) < mean_point_white_i or 0.95*(-163.4) < mean_point_white_q:
                floor_errors.append('Noise floor of module {mod} is higher than expected')
            if  max_point_1_f_i > -79 or max_point_1_f_q >-79:
                loud_errors.append('Noise levels for frequencies dominated by 1/f are higher than -79dBm')
            

            # Filter the PSDs for plotting
            mask_plot = (psdfreq_1_f > 0) & (psdfreq_1_f < 150)
            psdfreq_1_f_plot = psdfreq_1_f[mask_plot]
            psd_i_1_f_plot = psd_i_1_f[mask_plot]
            psd_q_1_f_plot = psd_q_1_f[mask_plot]

            # Filter the PSDs for fitting
            mask_fit = (psdfreq_1_f > 0) & (psdfreq_1_f < 15)
            psdfreq_1_f_fit = psdfreq_1_f[mask_fit]
            psd_i_1_f_fit = psd_i_1_f[mask_fit] 
            psd_q_1_f_fit = psd_q_1_f[mask_fit]

            # Combine I and Q components for fitting
            combined_psd_1_f_fit = np.concatenate((psd_i_1_f_fit, psd_q_1_f_fit))
            combined_freqs_fit = np.concatenate((psdfreq_1_f_fit, psdfreq_1_f_fit))

            # Combine I and Q components for plotting
            combined_psd_1_f = np.concatenate((psd_i_1_f_plot, psd_q_1_f_plot))
            combined_freqs = np.concatenate((psdfreq_1_f_plot, psdfreq_1_f_plot))
            
            # Convert PSD from dBm/Hz to W/Hz for fitting
            psd_values_W_Hz_fit = 10 ** ((combined_psd_1_f_fit - 30) / 10)

            # Perform a linear fit in the log-log space
            log_frequencies_fit = np.log10(combined_freqs_fit)
            log_psd_values_fit = np.log10(psd_values_W_Hz_fit)

            # Linear regression to find the slope and intercept
            slope, intercept, r_value, p_value, std_err = linregress(log_frequencies_fit, log_psd_values_fit)

            # Calculate the fitted line in log space over the full range for plotting
            log_frequencies_plot = np.log10(combined_freqs)
            fitted_log_psd_values_plot = intercept + slope * log_frequencies_plot

            # Convert fitted values back to linear scale for plotting
            fitted_psd_values_W_Hz_plot = 10 ** fitted_log_psd_values_plot
            fitted_psd_values_dBm_Hz_plot = 10 * np.log10(fitted_psd_values_W_Hz_plot) + 30


        
            module_result = {
                'module': mod,
                'psd_i_white': psd_i_white.tolist(),
                'psd_q_white': psd_q_white.tolist(),
                'psd_frequencies_white': psdfreq_white.tolist(),
                'max_point_white_i': max_point_white_i.tolist(),
                'max_point_white_q': max_point_white_q.tolist(),
                'psd_i_1_f': psd_i_1_f.tolist(),
                'psd_q_1_f': psd_q_1_f.tolist(),
                'psd_frequencies_1_f': psdfreq_1_f.tolist(),
                'combined_psd_1_f': combined_psd_1_f.tolist(),
                'combined_freqs': combined_freqs.tolist(),  # This should be the original combined frequency list
                'max_point_1_f_i': max_point_1_f_i.tolist(),
                'max_point_1_f_q': max_point_1_f_q.tolist(),
                'fitted_psd_values_dBm_Hz': fitted_psd_values_dBm_Hz_plot.tolist(),
                'slope': slope,
                'white_status': 'Fail' if floor_errors else 'Pass',
                '1_f_status': 'Fail' if loud_errors else 'Pass'
            }

            if mod not in summary_results['summary']:
                summary_results['summary'][mod] = {}
            summary_results['summary'][mod]['Noise floor'] = module_result['white_status']
            summary_results['summary'][mod]['1/f Noise'] = module_result['1_f_status']
        
            test_result['modules'].append(module_result)
        
        if test_highbank:
            d.set_analog_bank(True)
            highbank_run = True

    d.set_analog_bank(False)
    results['tests'].append(test_result)

def test_wideband_noise(hwm, results, summary_results, test_highbank, num_runs):
    #TBD with get_samples or more fast samples
    x=1
