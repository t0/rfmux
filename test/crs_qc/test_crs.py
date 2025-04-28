import os
import pytest
import rfmux
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' for non-interactive use
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from conftest import wait_for_action
'''

Welcome to the 1st draft of the CRS QC. To begin testing, type into the terminal 
$ pytest test_crs_integrated.py --serial=<serial> -s

If you wish to run only a set of particular tests:

$ pytest test_crs_integrated.py --serial=<serial> -k "<test_test_name_1> or <test_name_2>"

'''
# Ensure the board and the computer are connected to the same network.
# Most basic check, does not use hidfmux
def test_board_bootup(results):
    print("Retrieving board status")
    os.system("timeout 1 /home/joshua/code/crs-mkids/firmware/rfmux/r1.4/parser -i enp6s0 -t | grep 'Enclosure\\|Timestamp' > boot.txt")
    try:
        with open('boot.txt', 'r') as file:
            content = file.read()
    except FileNotFoundError:
        pytest.fail("No file was generated. Check your parser directory and connection to the board")
        
    assert content, "Error: No data received. The board might not be booted correctly."
    os.remove('boot.txt')
    print("Board boot status: OK")

    results['tests'].append({
        'title': 'Board Bootup Test',
        'status': 'OK'
        })
   
'''-------------------------------------HELPER FUNCTIONS---------------------------------------------'''
def transfer_function_helper(raw_sampl, AMPLITUDE, scale, attenuation):
    '''
    Calculates the dBm value of a sample from get_samples call.
    Predicts the dBm value of the signal from the parameters passed to it
    '''
    DB_LOSS_FACTOR = -2.56
    volts_peak_samples = (np.sqrt(2)) * np.sqrt(50 * (10 ** (-1.75 / 10)) / 1000) / 1880796.4604246316
    if np.iscomplexobj(raw_sampl):
        data_i = np.array(raw_sampl.real) * volts_peak_samples
        data_q = np.array(raw_sampl.imag) * volts_peak_samples
    else:
        data_i = np.array(raw_sampl.i) * volts_peak_samples
        data_q = np.array(raw_sampl.q) * volts_peak_samples
    magnitude = np.sqrt(data_i ** 2 + data_q ** 2)
    median_magnitude = np.median(magnitude)
    median_magnitude_dbm = 10 * np.log10(((median_magnitude / np.sqrt(2)) ** 2) / 50) + 30

    DAC_output_dBm = DB_LOSS_FACTOR + 10 * np.log10((AMPLITUDE ** 2) * 10 ** (scale / 10))
    predicted_magnitude_dbm = DAC_output_dBm - attenuation
    return median_magnitude_dbm, predicted_magnitude_dbm

def value_setter_helper(d, frequency, amplitude, scale, attenuation, channel, module, nco, target_dac, target_adc, highbank):
    '''

    NOTE: All the tests below are written with the following numbering conventions for highbanks
    set_amplitude, set_frequency, get_samples, set_nyqist_zone -- accept module 1-4
    set_scale, set_attenuation, set_nco -- accept module 1-8 
    
    '''

    d.clear_channels()
    d.set_frequency(frequency, d.UNITS.HZ, channel, module)
    d.set_amplitude(amplitude, d.UNITS.NORMALIZED, target_dac, channel, module)

    if highbank == True:
        module+=4
    d.set_dac_scale(scale, d.UNITS.DBM, module)
    d.set_adc_attenuator(attenuation, d.UNITS.DB, module=module)
    d.set_nco_frequency(nco, d.UNITS.HZ, target_dac, module=module)
    d.set_nco_frequency(-nco, d.UNITS.HZ, target_adc, module=module)

#small legacy helper used in checking how circular the i-q blob is
def covariance_matrix(samples):
    cov_matrix = np.cov(samples, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    ratio = eigenvalues[1] / eigenvalues[0]
    return ratio
'''------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------SENSORS--------------------------------------------------------------'''
def test_temperature(d, results, summary_results):
    TEMPERATURE_MIN = 0.0
    TEMPERATURE_MAX = 70.0
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
    
def test_voltages(d, results,summary_results):

    VOLTAGE_TOL = 0.10  # 10%
    test_status = 'Pass'
    sensors = (
        (d.VOLTAGE_SENSOR.MB_VBP, 12),  # power supply
        (d.VOLTAGE_SENSOR.MB_R0V85A, 0.85),#
        (d.VOLTAGE_SENSOR.MB_R0V85B, 0.85),#
        (d.VOLTAGE_SENSOR.MB_R1V2A, 1.2),#
        (d.VOLTAGE_SENSOR.MB_R1V2B, 1.2),#
        (d.VOLTAGE_SENSOR.MB_R1V4, 1.4),#
        (d.VOLTAGE_SENSOR.MB_R1V8A, 1.8),#
        (d.VOLTAGE_SENSOR.MB_R2V5, 2.5),#
        (d.VOLTAGE_SENSOR.MB_R3V3A, 3.3),#
        (d.VOLTAGE_SENSOR.MB_R5V0, 5.0),#
        (d.VOLTAGE_SENSOR.RFSOC_PSMGTRAVCC, 0.9), #seen in in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_PSMGTRAVTT, 1.8), #CONFUSION top right in in rfsoc_power says 1.2 but acc output 1.8
        (d.VOLTAGE_SENSOR.RFSOC_VCCAMS, 1.8), # CONFUSION found VCCINTAMS - 0.85. but need a diff one
        (d.VOLTAGE_SENSOR.RFSOC_VCCAUX, 1.8), #in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_VCCBRAM, 0.85),# seen in rfsoc_power.sch
        (d.VOLTAGE_SENSOR.RFSOC_VCCINT, 0.85), #seen in rfsoc_power.sch
        (d.VOLTAGE_SENSOR.RFSOC_VCCPLAUX, 1.8), #
        (d.VOLTAGE_SENSOR.RFSOC_VCCPLINTFP, 0.85),#in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_VCCPLINTLP, 0.85),#in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSAUX, 1.8),#in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSDDR, 1.2), #
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSINTFP, 0.85),#in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSINTFPDDR, 0.85),#in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSINTLP, 0.85), #in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSIO0, 1.8),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSIO1, 3.3),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSIO2, 3.3),
        (d.VOLTAGE_SENSOR.RFSOC_VCCPSIO3, 3.3),
        (d.VOLTAGE_SENSOR.RFSOC_VCCVREFN, 0),
        (d.VOLTAGE_SENSOR.RFSOC_VCCVREFP, 1.85),
        (d.VOLTAGE_SENSOR.RFSOC_VCC_PSBATT, 0),#this is grounded as seen in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_VCC_PSDDRPLL, 1.8),# in rfsoc_power
        (d.VOLTAGE_SENSOR.RFSOC_VCC_PSPLL0, 1.2),# in rfsoc_power
    )
    #Get the real time data from sensors and check if it's within 10% of the allowed
    with d.tuber_context() as ctx:
        for (s, nv) in sensors:
            ctx.get_motherboard_voltage(s)
        voltages = ctx()
    for ((s, nv), v) in zip(sensors, voltages):
        min_v = nv * (1 - VOLTAGE_TOL)
        max_v = nv * (1 + VOLTAGE_TOL)

        if  round(v, 2) <min_v or round(v, 2)> max_v:
            print(f"Sensor {s} with value {v} out of range {min_v}, {max_v}!")
            test_status = 'Fail'
        print(f"Voltage for sensor {s}: {v}")

    results['tests'].append({
        'title': 'Voltage Test',
        'sensors': [str(sensor[0]) for sensor in sensors],
        'voltages': voltages,
        'test_status': test_status
    })

    if 'Board Health' not in summary_results['summary']:
        summary_results['summary']['Board Health'] = {}
    summary_results['summary']['Board Health']['Voltage Test'] = test_status

def test_currents(d, results, summary_results):
    CURRENT_LIMIT = 0.4  # 40% tolerance on the current limit
    test_status = 'Pass'
    sensors = (
        (d.CURRENT_SENSOR.MB_R0V85A, 2.5),
        (d.CURRENT_SENSOR.MB_R0V85B, 3.3),
        (d.CURRENT_SENSOR.MB_R1V2A, 2.4),
        (d.CURRENT_SENSOR.MB_R1V2B, 2.6),
        (d.CURRENT_SENSOR.MB_R1V4, 3.6),
        (d.CURRENT_SENSOR.MB_R1V8A, 3.3),
        (d.CURRENT_SENSOR.MB_R2V5, 4.1),
        (d.CURRENT_SENSOR.MB_R3V3A, 3.4),
        (d.CURRENT_SENSOR.MB_R5V0, 4.1),
        (d.CURRENT_SENSOR.MB_VBP, 4),
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
        print(f"Current for sensor {s}: {v}")
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
def test_dac_passband(d, results, summary_results, num_runs, test_highbank, results_dir):

    '''This test aims to determine 2 things: if the magnitude of the output signal from the DAC 
    is within expected margin and if the passband stays flat throughout the predicted frequency range (-300 -- 300) wrt the NCO.

    The algorithm takes the length of'frequencies' array to determine how many channels to set. If the frequencies array is >1000
    items, it records that repetitions of the process are needed (num_runs). NOTE: feel free to adjust the frequencies array to your 
    desired density. 
    Then, the 1000 channels are set to the amplitudes and frequencies -300 -- 300. Recorded sample magnitude is compared to the 
    predicted magnitude (predicted from the transfer functions). 
     
    0 Hz frequency is taken as reference. If recorded magnitude is >10% off from predicted -- fail. If the measured data on either side 
    of the 0Hz deviates by more than 3dB from center freuqency -- fail.
    
    AMPLITUDE - Intentionally chosed to be 0.001 to be able to set 1000 channels at once without exceeding the full scale
    NCO_FREQUENCY - Can be changed if desired. Was chosen as 625e6 to gauge module performance comparing it to old, measured data. 
    NOMINAL_AMPLITUDE, NOMINAL_FREQUENCY - are there to help set the attenuation, scale for each module, are completely nominal, can you any number.

    '''

    # FIXME: Some issue with the higher range of the frequencies showing a dip
    AMPLITUDE = 0.005 # FIXME: JM change by x5
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 10
    NCO_FREQUENCY = 1e9+123
    ATTENUATION = 0
    SCALE = 7
    NOMINAL_AMPLITUDE = 0.5
    NOMINAL_FREQUENCY = 200e6
    NOMINAL_CHANNEL = 1
    TITLE = 'DAC Passband Test'

   #feel free to increase the range and density of frequencies
    frequencies = np.arange(-300e6, 300e6, 10e6)
    channels = np.arange(1, 1001)
    nruns = int(np.ceil(len(frequencies) / len(channels)))
    lenrun = min(len(frequencies), len(channels))
    test_result = {
        'title': TITLE,
        'modules': []
    }

    #always start with the lowbanks. This will only change to True (and algorithm will repeat) if the passed test_highbank ==True
    highbank_run = False

    #num_runs passed from the config file depending on whether or not there will be a highbank run
    for num_run in range(0, num_runs):
        for module in range(1, 5):

            #used for saving data in the dictionary under correct number (while being able to set values and get samples following the current conventions)
            module_actual = module
            
            if highbank_run:
                module_actual +=4

            print(f'DAC_passband - Testing Module #:{module_actual}')
            
            measured_dbm = []
            predicted_dbm = []
            module_errors = []
            for nrun in range(nruns):
                start_idx = nrun * lenrun
                end_idx = min(start_idx + lenrun, len(frequencies))
                d.clear_channels()
                time.sleep(0.2)

                #First, call the function to set all the parameters that won't change: attenuation, scale, nco
                value_setter_helper(d, NOMINAL_FREQUENCY,NOMINAL_AMPLITUDE, SCALE, ATTENUATION, NOMINAL_CHANNEL, module, NCO_FREQUENCY, TARGET_DAC, TARGET_ADC, highbank_run)
                
                with d.tuber_context() as ctx:
                    #Set all the nessesary channels for the run
                    for idx in range(start_idx, end_idx):
                        channel = int(channels[idx - start_idx])  # Convert to int and access correct channel
                        frequency = float(frequencies[idx])  # Convert to float
                        ctx.set_frequency(frequency, d.UNITS.HZ, channel, module)
                        ctx.set_amplitude(AMPLITUDE, d.UNITS.NORMALIZED, TARGET_DAC, channel, module)
                    ctx()
                time.sleep(0.2)

                # Get all the samples for the current run
                raw = d.py_get_samples(SAMPLES, channel=None, module=module)

                for idx in range(start_idx, end_idx):
                    channel = int(channels[idx - start_idx])  # Convert to int and access correct channel
                    rchan = np.array(raw.i[channel+1])+1j*np.array(raw.q[channel+1])
                    frequency = float(frequencies[idx])  # Ensure frequency is a float
                    measured, predicted = transfer_function_helper(rchan, AMPLITUDE, SCALE, ATTENUATION)
                    measured_dbm.append((frequency, measured))
                    predicted_dbm.append((frequency, predicted))

            mag_0 = next(meas for freq, meas in measured_dbm if freq == 0)
            
            #if 10% error from predicted -- fail
            if mag_0 < 1.1*predicted or mag_0 > 0.9*predicted:
                error = f"At a reference frequency (625MHz), generating a power output >10% off from predicted. Expected value is: {predicted}. Measured value was: {mag_0:.2f}"
                print(error)
                module_errors.append(error)

            #if 3dBm from predicted -- fail
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
                'frequencies': [freq/ 1e6 for freq, _ in measured_dbm],  # Convert frequencies to float
                'measured_dbm': [meas for _, meas in measured_dbm],  # Ensure this is a float
                'predicted_magnitude_dbm_values': [predicted for _, predicted in predicted_dbm],  # Convert to a list of floats
                'status': 'Fail' if module_errors else 'Pass',
                'error': '\n'.join(module_errors),
                'plot_path': 'plot_path'
            }

            #Determine the graph name / address for the plot (will be accessed in report_generator)
            graph_name = TITLE + "_module_"+ f'{module_result["module"]}' +'.png'
            plot_path = os.path.join(results_dir, 'plots', graph_name)
            module_result["plot_path"] = plot_path
            
            plt.figure(figsize=(10, 6))
            plt.plot(module_result['frequencies'], module_result['measured_dbm'], '-o', label = 'measured' )
            plt.plot(module_result['frequencies'], module_result['predicted_magnitude_dbm_values'], '-o', label = 'predicted' )
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Magnitude (dBm)')
            plt.legend()
            plt.title(f'Signal at DAC of module {module_result["module"]} with increasing frequency')
            plt.grid(True)
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
            

            test_result['modules'].append(module_result)

            if module_actual not in summary_results['summary']:
                summary_results['summary'][module_actual] = {}
            summary_results['summary'][module_actual]['Passband'] = module_result['status']

        #after 1 ru is complete, determine if a second run is needed
        if test_highbank:
            d.set_analog_bank(True)
            highbank_run = True
    
    d.set_analog_bank(False)
    results['tests'].append(test_result)


def test_dac_amplitude_transfer(d, results, summary_results, test_highbank, num_runs, results_dir):
    '''
    This test aims to determine if the output of the dac is scaled appropriately with the change in amplitude. The algorithm uses only 1 channel
    at each module a set frequency, varying its amplitude. The result is compared to the transfer function prediction
    
    '''
    SCALE = 1
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 100
    ATTENUATION = 0
    CHANNEL = 1
    NCO_FREQUENCY = 0
    FREQUENCY = 150.5763e6
    NOMINAL_AMPLITUDE = 0
    TITLE = 'DAC Amplitude Transfer Test'

    test_result = {
        'title': TITLE,
        'modules': []
    }

    highbank_run = False
    #feel free to adjust the amplitude density as needed
    amplitudes = np.arange(0.05, 1, 0.05)
    d.clear_channels()

    #num_runs passed from the config file depending on whether or not there will be a highbank run
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
            
            #When starting to test a new module, clear all channels and use the value_setter to set the variables that won't change attenuation, scale, frequency etc
            value_setter_helper(d, FREQUENCY, NOMINAL_AMPLITUDE, SCALE, ATTENUATION, CHANNEL, module, NCO_FREQUENCY, TARGET_DAC, TARGET_ADC, highbank_run)
            

            for amplitude in amplitudes:
                d.set_amplitude(amplitude, d.UNITS.NORMALIZED, TARGET_DAC, CHANNEL, module)
                samples = d.py_get_samples(SAMPLES, channel=CHANNEL, module=module)
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


            #if any value is 3dB from predicted -- fail
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
                'error': '\n'.join(module_errors),
                'plot_path': 'plot_path'
            }

            graph_name = TITLE + "_module_"+ f'{module_result["module"]}' +'.png'
            plot_path = os.path.join(results_dir, 'plots', graph_name)
            module_result["plot_path"] = plot_path  

            plt.figure(figsize=(10, 6))
            plt.plot(module_result['amplitude_values'], module_result['magnitude_dbm_values'], '-o', label = 'measured' )
            plt.plot(module_result['amplitude_values'], module_result['predicted_magnitude_dbm_values'], '-o', label = 'predicted' )
            plt.xscale('log')
            plt.xlabel('Amplitude (Normalized)')
            plt.ylabel('Magnitude (dBm)')
            plt.legend()
            plt.title(f'DAC output of module {module_result["module"]} with increasing amplitude')
            plt.grid(True)
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

            test_result['modules'].append(module_result)
            if module not in summary_results['summary']:
                summary_results['summary'][module] = {}
            summary_results['summary'][module]['DAC Ampl'] = module_result['status']

        if test_highbank:
            d.set_analog_bank(True)
            highbank_run = True
        d.clear_channels()

    d.set_analog_bank(False)
    results['tests'].append(test_result)


def test_dac_scale_transfer(d, results, summary_results, test_highbank, num_runs, results_dir):
    '''
    This test aims to determine if the output of the dac is scaled appropriately with the change in 'scale' value (actually controlled 
    by varying the current in the balun). The algorithm uses only 1 channel at each module a set frequency, varying its scale. 
    The result is compared to the transfer function prediction

    Potentially add:
    Pick 4 channels, set the same frequency-> alter scale and plot the samples first, then convert to dBm
    plot together with prediction from the transfer funciton
    '''
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
    TITLE = 'DAC Scale Transfer Test'
    #dictionary just for this test will be appended to all the results at the end (could modify to avoid losing data if a pytest crash occurs)
    test_result = {
        'title': TITLE,
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
                raw_sampl = d.py_get_samples(SAMPLES, channel=CHANNEL, module=module_sample)
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
                print(error)
            if not module_errors:
                print(f'Module {module_set} passed the DAC scaling test')
                    
            module_result = {
                'module': module_set,
                'scale_values': scale_values.flatten().tolist(),
                'magnitude_dbm_values': magnitude_dbm_values,
                'predicted_magnitude_dbm_values': predicted_magnitude_dbm_values,
                'status': 'Fail' if module_errors else 'Pass',
                'errors': f'{module_errors}',
                'plot_path': 'plot_path'
            }

            #same as in above tests, the path to the plot will be used by the report_generator
            graph_name = TITLE + "_module_"+ f'{module_result["module"]}' +'.png'
            plot_path = os.path.join(results_dir, 'plots', graph_name)
            module_result["plot_path"] = plot_path  

            plt.figure(figsize=(10, 6))
            plt.plot(module_result['scale_values'], module_result['magnitude_dbm_values'], '-o', label = 'measured' )
            plt.plot(module_result['scale_values'], module_result['predicted_magnitude_dbm_values'], '-o', label = 'predicted' )
            plt.xlabel('Scale(dBm)')
            plt.ylabel('Magnitude (dBm)')
            plt.legend()
            plt.title(f'DAC output of module {module_result["module"]} with increasing scale (dBm)')
            plt.grid(True)
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

            test_result['modules'].append(module_result)
            if module_set not in summary_results['summary']:
                summary_results['summary'][module_set] = {}
            summary_results['summary'][module_set]['DAC Scale'] = module_result['status']

        if test_highbank:
            d.set_analog_bank(True)
            highbank_run = True

    d.set_analog_bank(False)
    results['tests'].append(test_result)

def test_dac_mixmodes(d, results, summary_results, test_highbank):
    '''
    This test aims to verify the overall behaviour of the nyquist zones (1(NRZ) or 2(RC)) set in the firmware as well as a sanity check for the hardware response in
    the higher nquist regions. 
    The test uses 3 modules at a time, getting the behaviour of a singular nyquist region from a singular module. The report generator then will stitch the 
    data together into one graph per nyquist zone.
    '''

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

    #obtained through a separate loopback measurement: transferfunction applied to a 1000-count get_samples call
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

    def mixmode_test_for_bank(bank_name, is_highbank):
        test_result = {
            'title': f'DAC Mixmodes Test - {bank_name} bank',
            'analog_bank': bank_name,
            'mixmodes': []
        }

        d.set_analog_bank(is_highbank)

        for mixmode in range(1, 3):
            for module, nyquist_zone in MODULE_NYQUIST_MAP.items():
                module_setter = module
                if is_highbank:
                    module+=4

                module_errors = []
                module_magnitude_dbm_values = np.array([])
                frequencies = np.array([])
                print(f'Running module {module} in mixmode #{mixmode} nyquist region #{nyquist_zone} for {bank_name} bank')

                d.set_nyquist_zone(mixmode, module=module)
                nco = 0
                frequency = -DAC_MAX_FREQUENCY

                while nco < MAX_TOTAL_FREQUENCY:
                    while frequency <= DAC_MAX_FREQUENCY and nco + frequency < MAX_TOTAL_FREQUENCY:
                        if nco + frequency >= 0:
                            value_setter_helper(d, frequency, AMPLITUDE, SCALE, ATTENUATION, CHANNEL, module_setter, nco, TARGET_DAC, TARGET_ADC, is_highbank)
                            samples = d.py_get_samples(SAMPLES, channel=CHANNEL, module=module_setter)
                            module_magnitude_dbm, _ = transfer_function_helper(samples, AMPLITUDE, SCALE, ATTENUATION)

                            if nyquist_zone == 1:
                                frequencies = np.append(frequencies, frequency + nco)
                                module_magnitude_dbm_values = np.append(module_magnitude_dbm_values, module_magnitude_dbm)
                            elif nyquist_zone == 2:
                                frequencies = np.append(frequencies, 2 * SAMPLING_FREQUENCY - (frequency + nco))
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
                    'reference_frequencies': reference_freqs.tolist(),
                    'reference_magnitudes': reference_mags.tolist(), 
                    'status': status
                })
                if module not in summary_results['summary']:
                    summary_results['summary'][module] = {}
                summary_results['summary'][module][f'Mixmode {mixmode}'] = status
                if mixmode == 2:
                    d.set_nyquist_zone(1, module=module)

        results['tests'].append(test_result)

    # Run test for low bank
    mixmode_test_for_bank('low', False)

    # If high bank needs to be tested, prompt the user to switch filters and run again
    if test_highbank:
        wait_for_action("Please switch filters from low to high banks before proceeding with high bank testing. \n \
                        1st Nyquist filter to ADC#5,\n\
                        2nd Nyquist filter to ADC#6,\n\
                        3rd Nyquist filter to ADC#7\n \
                        If already have filters connected - disregard this message")
        
        mixmode_test_for_bank('high', True)

    d.set_analog_bank(False)




'''---------------------------------------------------------------------------------------------------------------'''
'''--------------------------------------------------ADC----------------------------------------------------------'''

def test_adc_attenuation(d, results, summary_results, test_highbank, num_runs, results_dir):
    AMPLITUDE = 1
    TARGET_DAC = d.TARGET.DAC
    TARGET_ADC = d.TARGET.ADC
    SAMPLES = 100
    FREQUENCY = 80.234562e6
    NCO_FREQUENCY = 0
    SCALE = 1
    CHANNEL = 1
    ATTENUATION_0 = 0
    TITLE = 'ADC Attenuation Test'

    test_result = {
        'title': TITLE,
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
                d.set_adc_attenuator(attenuation, d.UNITS.DB, module=module_set)
                raw_sampl = d.py_get_samples(SAMPLES, channel=CHANNEL, module=module_sample)
                median_magnitude_dbm, predicted_magnitude_dbm = transfer_function_helper(raw_sampl, AMPLITUDE, SCALE, attenuation)
                attenuation_values.append(attenuation)
                magnitude_dbm_values.append(median_magnitude_dbm)
                predicted_magnitude_dbm_values.append(predicted_magnitude_dbm)

            attenuation_values = np.array(attenuation_values).reshape(-1, 1)
            regressor = LinearRegression().fit(attenuation_values, magnitude_dbm_values)
            linearity = r2_score(magnitude_dbm_values, regressor.predict(attenuation_values))

            if linearity < 0.99:
                error = f'linearity check failed'
                print(error)
                module_errors.append(error)

            if np.any(np.abs(np.array(magnitude_dbm_values) - np.array(predicted_magnitude_dbm_values)) >= 1):
                error = f'Module {module_set} is generating a power output lower than expected. Check your balun and SMA connections.'
                module_errors.append(error)
                print(error)

            else:
                print(f"Module {module_set} passed ADC attenuation tests.")
            module_result = {
                'module': module_set,
                'attenuation_values': attenuation_values.flatten().tolist(),
                'magnitude_dbm_values': magnitude_dbm_values,
                'predicted_magnitude_dbm_values': predicted_magnitude_dbm_values,
                'status': 'Fail' if module_errors else 'Pass',
                'errors': f'{module_errors}',
                'plot_path': 'plot_path'
            }

            graph_name = TITLE + "_module_"+ f'{module_result["module"]}' +'.png'
            plot_path = os.path.join(results_dir, 'plots', graph_name)
            module_result["plot_path"] = plot_path  

            plt.figure(figsize=(10, 6))
            plt.plot(module_result['attenuation_values'], module_result['magnitude_dbm_values'], '-o', label = 'measured' )
            plt.plot(module_result['attenuation_values'], module_result['predicted_magnitude_dbm_values'], '-o', label = 'predicted' )
            plt.xlabel('Attenuation(dBm)')
            plt.ylabel('Magnitude (dBm)')
            plt.legend()
            plt.title(f'Signal at ADC of module {module_result["module"]} with increasing attenuation')
            plt.grid(True)
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()


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
def test_adc_even_phase(d, results):
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
        x = d.py_get_samples(SAMPLES, channel=1, module=1)
    
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

#used in the test_loopback_noise
def psd_helper(data_i_raw, data_q_raw):
    COMB_SAMPLING_FREQUENCY = 625e6
    volts_peak = (np.sqrt(2)) * np.sqrt(50 * (10 ** (-1.75 / 10)) / 1000) / 1880796.4604246316
    # Convert I and Q to peak volts
    tod_i = data_i_raw * volts_peak
    tod_q = data_q_raw * volts_peak

    # Define the sampling frequency for the ADC
    fir = (COMB_SAMPLING_FREQUENCY / 256 / 64 / 2 ** 6)
    # Apply Welch's method to get the signal PSD
    psdfreq, psd_i = signal.welch(tod_i/np.sqrt(4*50), fir, nperseg=len(tod_i// 4)) #tod_i / np.sqrt(4 * 50) 
    _, psd_q = signal.welch(tod_q/np.sqrt(4*50), fir, nperseg=len(tod_q// 4)) #tod_q / np.sqrt(4 * 50)

    # Convert from watts to dBm
    psd_i = 10 * np.log10(psd_i*1000)
    psd_q = 10 * np.log10(psd_q*1000)
    return psd_i, psd_q, psdfreq
       

def test_loopback_noise(d, results, summary_results, test_highbank, num_runs, results_dir):
    '''
    1. Send a signal with no amplitude to gauge the noise floor, and any spikes in the noise floor
    2. Send a higher power signal to dominate the output with 1/f noise. Determine the slope of 1/f and any unusual spikes
    '''
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
    EXPECTED_SLOPE = -1
    TITLE = 'Loopback Noise Test'
    test_result = {
        'title': TITLE,
        'modules': []
    }
    highbank_run = False
    print(f'Just in case, i am not dead, just slow. At 10k samples -- 1 min/module. At 60k samples 4 min/module...')
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
            raw_samples_white = d.py_get_samples(SAMPLES, channel=CHANNEL_WHITE, module=mod)
            data_i_white = np.array(raw_samples_white.i)
            data_q_white = np.array(raw_samples_white.q)
            psd_i_white, psd_q_white, psdfreq_white = psd_helper(data_i_white, data_q_white)


            # 2. Test the PSD of a large signal, establishing an expectation for 1/f slope and peak noise at low frequency
            value_setter_helper(d, FREQUENCY_1_F, AMPLITUDE_1_F, SCALE_1_F, ATTENUATION_1_F, CHANNEL_1_F, mod, NCO_FREQUENCY_1_F, TARGET_DAC, TARGET_ADC, highbank_run)
            raw_samples_psd = d.py_get_samples(SAMPLES, channel=CHANNEL_1_F, module=mod)
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

            if slope < 1.1*EXPECTED_SLOPE or slope > 0.9*EXPECTED_SLOPE:
                loud_errors.append('1/f noise deviates from the expected value of -1')
        
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
                '1_f_status': 'Fail' if loud_errors else 'Pass',
                'plot_path_white': 'plot_path_white',
                'plot_path_1_f': 'plot_path_1_f'
            }

            graph_name_white = TITLE + "_white_"+"_module_"+ f'{module_result["module"]}' +'.png'
            plot_path_white = os.path.join(results_dir, 'plots', graph_name_white)
            module_result["plot_path_white"] = plot_path_white  

            graph_name_1_f = TITLE +"_1_f" + "_module_"+ f'{module_result["module"]}' +'.png'
            plot_path_1_f = os.path.join(results_dir, 'plots', graph_name_1_f)
            module_result["plot_path_1_f"] = plot_path_1_f

            #plot the white noise
            plt.figure(figsize=(10, 6))
            plt.plot(module_result['psd_frequencies_white'], module_result['psd_i_white'],  label = 'I' )
            plt.plot(module_result['psd_frequencies_white'], module_result['psd_q_white'], label = 'Q' )
            plt.scatter([module_result['psd_frequencies_white'][np.argmax(module_result['psd_i_white'])]], [module_result['max_point_white_i']], color='red', label=f'Max I spike: {module_result["max_point_white_i"]:.2f} dBm')
            plt.scatter([module_result['psd_frequencies_white'][np.argmax(module_result['psd_q_white'])]], [module_result['max_point_white_q']], color='green', label=f'Max Q spike: {module_result["max_point_white_q"]:.2f} dBm')
            plt.xlabel('Frequency (Hz from carrier)')
            plt.ylabel('Magnitude (dBm)')
            plt.xscale('log')
            plt.legend()
            plt.title(f'Noise floor in loopback at 277.23MHz at module {mod}')
            plt.grid(True)
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path_white), exist_ok=True)
            plt.savefig(plot_path_white)
            plt.close()

             # 1/f noise PSD
            plt.figure(figsize=(10, 6))
            plt.plot(module_result['combined_freqs'], module_result['fitted_psd_values_dBm_Hz'], 'r', label=f'fit to 1/f spectrum: n = {module_result["slope"]:.2f}')
            plt.scatter(module_result['combined_freqs'], module_result['combined_psd_1_f'], label='1/f spectrum data')
            plt.scatter([module_result['psd_frequencies_1_f'][np.argmax(module_result['psd_i_1_f'])]], [module_result['max_point_1_f_i']], color='red', label=f'Max I spike: {module_result["max_point_1_f_i"]:.2f} dBm')
            plt.scatter([module_result['psd_frequencies_1_f'][np.argmax(module_result['psd_q_1_f'])]], [module_result['max_point_1_f_q']], color='green', label=f'Max Q spike: {module_result["max_point_1_f_q"]:.2f} dBm')
            plt.xscale('log')
            plt.title('PSD of 1/f Noise')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (dBm/Hz)')
            plt.legend()
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path_1_f), exist_ok=True)
            plt.savefig(plot_path_1_f)
            plt.close()



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



def test_wideband_noise(d, results, summary_results, test_highbank, num_runs, results_dir):

    SAMPLING_FREQUENCY = 5e9  # 5 GHz
    NUM_SAMPLES = 4000  # Total number of samples
    highbank_run = False
    TITLE =  'Wideband FFT for Noise Detection'

    test_result = {
        'title': TITLE,
        'modules': []
    }

    for num_run in range(0, num_runs):
        for mod in range(1, 5):

            if highbank_run:
                print(f'Wideband FFT - Testing Module #:{mod + 4}')
            else:
                print(f'Wideband FFT - Testing Module #:{mod}')

            d.clear_channels()
            time.sleep(1)
            samples = d.get_fast_samples(NUM_SAMPLES, d.UNITS.RAW, module=mod)

            volts_peak_samples = (np.sqrt(2)) * np.sqrt(50 * (10 ** (-1.75 / 10)) / 1000) / 1880796.4604246316
            I_signal = np.array(samples.i) * volts_peak_samples
            Q_signal = np.array(samples.q) * volts_peak_samples


            psdfreq_i, psd_i = signal.welch(I_signal/np.sqrt(4*50), SAMPLING_FREQUENCY, nperseg=NUM_SAMPLES) #tod_i / np.sqrt(4 * 50) 
            psdfreq_q, psd_q = signal.welch(Q_signal/np.sqrt(4*50), SAMPLING_FREQUENCY, nperseg=NUM_SAMPLES) #tod_q / np.sqrt(4 * 50)

            # Convert from watts to dBm
            psd_i = 10 * np.log10(psd_i*1000)
            psd_q = 10 * np.log10(psd_q*1000)

            if highbank_run:
                mod+=4

            module_result = {
                'module': mod,
                'psd_i': psd_i.tolist(),
                'psd_q': psd_q.tolist(),
                'freqs_i': psdfreq_i.tolist(),
                'freqs_q': psdfreq_q.tolist(),
                'status': 'N/A',
                'plot_path': 'plot_path'
            }

            if mod not in summary_results['summary']:
                summary_results['summary'][mod] = {}
            summary_results['summary'][mod]['Wide FFT'] = module_result['status']

            graph_name = TITLE +"_module_"+ f'{module_result["module"]}' +'.png'
            plot_path = os.path.join(results_dir, 'plots', graph_name)
            module_result["plot_path"] = plot_path 

            #plot the white noise
            plt.figure(figsize=(10, 6))
            plt.plot(module_result['freqs_i'], module_result['psd_i'],  label = 'I' )
            plt.plot(module_result['freqs_q'], module_result['psd_q'], label = 'Q' )
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dBm/Hz)')
            plt.legend()
            plt.title(f'Wideband FFT at module {module_result["module"]}')
            plt.grid(True)
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

            test_result['modules'].append(module_result)

        if test_highbank:
            d.set_analog_bank(True)
            highbank_run = True

    d.set_analog_bank(False)
    results['tests'].append(test_result)
    
