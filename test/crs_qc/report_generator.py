import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib.colors as mcolors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

class PDFReport:
    def __init__(self, serial, results_dir):
        self.serial = serial
        self.results_dir = results_dir
        self.filename = os.path.join(results_dir, f"report_{serial}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        self.pdf = SimpleDocTemplate(self.filename, pagesize=letter)
        self.elements = []
        self.styles = getSampleStyleSheet()
        self.elements.append(Paragraph(f"Test Report for CRS Board #{serial}", self.styles['Title']))
        self.elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))

    def add_paragraph(self, text, style='Normal'):
        """Adds a paragraph of text to the document."""
        self.elements.append(Spacer(1, 12))
        self.elements.append(Paragraph(text, self.styles[style]))

    def add_section(self, title):
        self.elements.append(Spacer(1, 12))
        self.elements.append(Paragraph(title, self.styles['Heading1']))

    def add_plot(self, plot_path, title):
        self.elements.append(Spacer(1, 12))
        self.elements.append(Paragraph(title, self.styles['Heading2']))
        self.elements.append(Image(plot_path, width=450, height=250))

    def add_table(self, table_data, title, custom_style=None, column_width=None):
        self.elements.append(Spacer(1, 12))
        self.elements.append(Paragraph(title, self.styles['Heading2']))
        if column_width:
            table = Table(table_data, colWidths=column_width)
        else:
            table = Table(table_data)

        default_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)])
        
        if custom_style:
            table.setStyle(TableStyle(custom_style))
        else:
            table.setStyle(default_style)

        self.elements.append(table)

    def save(self):
        self.pdf.build(self.elements)
        print(f"PDF report generated: {self.filename}")

    #THIS FUNCTION is only being used by mixmodes (ran out of time to convert it). But ideally the plotting and the saving 
    #should happend as testing goes on (inside test_crs.py)
    def plot_data_and_save(self, plot_method, x_set_1, data_set_1, x_set_2 = None, data_set_2=None, label_set_1='measured', label_set_2='predicted', x_label='X-axis', y_label='Y-axis', title='Title', filename='plot.png', scale='linear', yscale='linear', mixmode=False):
        
        plt.figure(figsize=(10, 6))
        colors = list(mcolors.TABLEAU_COLORS.values())
        reference_color = 'black'

        if mixmode:
            # Special handling for mixmode data
            for i, (module, module_data) in enumerate(data_set_1.items()):
                if i == 0:
            # Plot the reference data for all modules without the label after the first plot
                    plt.plot(module_data['reference_frequencies'], module_data['reference_magnitudes'], 'o', label='Reference data', color=reference_color)
                else:
                    plt.plot(module_data['reference_frequencies'], module_data['reference_magnitudes'], 'o', color=reference_color)

                plt.scatter(module_data['frequencies'], module_data['magnitudes'], label=f'Module {module}', color=colors[i % len(colors)])
        else:
            if plot_method == 'plot':
                if x_set_1 is not None:
                    plt.plot(x_set_1, data_set_1, '-o', label=label_set_1)
                else:
                    plt.plot(data_set_1, '-o', label=label_set_1)

                if data_set_2 is not None and x_set_2 is not None:
                    plt.plot(x_set_1, data_set_2, '-o', label=label_set_2)

            elif plot_method == 'scatter':
                if x_set_1 is not None:
                    plt.scatter(x_set_1, data_set_1, label=label_set_1)
                else:
                    plt.scatter(data_set_1, label=label_set_1)

                if data_set_2  is not None and x_set_2 is not None:
                    plt.scatter(x_set_1, data_set_2, label=label_set_2)

            elif plot_method == 'plot line':
                if x_set_1 is not None:
                    plt.plot(x_set_1, data_set_1, label=label_set_1)
                else:
                    plt.plot(data_set_1, label=label_set_1)

                if data_set_2 is not None and x_set_2 is not None:
                    plt.plot(x_set_2, data_set_2, label=label_set_2)

        plt.xlabel(x_label)
        plt.xscale(scale)
        plt.yscale(yscale)
        plt.ylabel(y_label)
        plt.legend()
        plt.title(title)
        plt.grid(True)

        # Create plot directory
        plot_path = os.path.join(self.results_dir, 'plots', filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.tight_layout()
        plt.close()
        print(f"Plot saved to: {plot_path}") 
        return plot_path
    
    
    def generate_summary(self, summary_data):
        self.add_section('Summary of Test Results')

        # Extract non-module-specific tests from "Board Health"
        board_health_tests = summary_data['summary'].get('Board Health', {})

        #only create Board Health tests if they were conducted
        if board_health_tests:  
            non_module_data = [['Test', 'Status']]
            for test, status in board_health_tests.items():
                non_module_data.append([test, status])

            # Use default style for non-module-specific table
            self.add_table(non_module_data, 'Board health')


        # Generate module-specific table data
        all_tests = sorted(set(test for module, tests in summary_data['summary'].items() if module.isdigit() for test in tests))
        module_specific_data = [['Module'] + all_tests]

        for module, tests in summary_data['summary'].items():
            if not module.isdigit():
                continue  # Skip non-module-specific tests
            row = [module] + [tests.get(test, 'N/A') for test in all_tests]
            module_specific_data.append(row)
        
        page_width = letter[0] - 2 * inch  # Letter page width minus 2 inches of margins (1 inch on each side)
        num_columns = len(module_specific_data[0])
        column_width = [page_width / num_columns] * num_columns


        # Style for module-specific table
        pastel_green = colors.Color(0.85, 1.0, 0.85)
        pastel_red = colors.Color(1.0, 0.85, 0.85)
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]

        for row_index, row in enumerate(module_specific_data[1:], start=1):  # Skip header row
            for col_index, cell in enumerate(row[1:], start=1):  # Skip module column
                if cell == 'Pass':
                    style.append(('BACKGROUND', (col_index, row_index), (col_index, row_index), pastel_green))
                elif cell == 'Fail':
                    style.append(('BACKGROUND', (col_index, row_index), (col_index, row_index), pastel_red))

        if len(module_specific_data) > 1:  # Only add the table if there are module-specific tests
            self.add_table(module_specific_data, 'Summary of Module Tests', custom_style=style, column_width=column_width)

    
    
    def generate_temperature_section(self, data):
        self.add_section('Temperature Test')

        table_data = [['Sensor', 'Temperature (C)']] + list(zip(data['sensors'], data['temperatures']))
        self.add_table(table_data, 'Sensor Temperatures')

    def generate_voltage_section(self, data):
        self.add_section('Voltage Test')
        rounded_measured_voltage = [round(x, 2) for x in data['voltages']]

        table_data = [['Sensor', 'Voltage (V)']] + list(zip(data['sensors'], rounded_measured_voltage))
        self.add_table(table_data, 'Sensor Voltages')

    def generate_current_section(self, data):
        self.add_section('Current Test')
        rounded_measured_currents = [round(x, 2) for x in data['currents']]

        table_data = [['Sensor', 'Current (A)']] + list(zip(data['sensors'], rounded_measured_currents))
        self.add_table(table_data, 'Sensor Currents')



    def generate_dac_passband_section(self, data):
        self.add_section('DAC Passband Test')
        self.add_paragraph('This test aims to determine if the DAC accurately produces signals of frequencies within its passband (550MHz). \
                            A flat passband is expected within 275MHz relative to the NCO \
                           and the measured magnitudes will be compared to the reference values. The following measurement is taken at NCO = 625MHz, \
                           with Amplitude  = 0.001, and Scale = 7. This test has capacity to set and measure 1000 channels simultaneously, so you can adjust the density of the points directly in the script')


        for module in data['modules']:
            self.add_plot(f'{module["plot_path"]}', f'DAC passband at module {module["module"]}')

            if module['error']:
                self.add_paragraph(f'Module {module["module"]} does not meet the flat passband requirement.\
                                    {module["error"]}')
            else:
                self.add_paragraph(f'Module {module["module"]} meets the passband requirements')

    def generate_dac_amplitude_transfer_section(self, data):
            self.add_section('DAC Amplitude Transfer Test')
            self.add_paragraph('This test aims to determine if the DAC accurately scales the output signal with an increase in amplitude. \
                            The setting of amplitude is internal to the FPGA and represents the fraction of power outputted by the dac.\
                            The relationship is expected to be logarithmic and is predicted with a DAC Transfer function')

            for module in data['modules']:
                self.add_plot(f'{module["plot_path"]}', f'Signal power vs normalized amplitude at module {module["module"]}')

                if module['error']:
                    self.add_paragraph(f'Module {module["module"]} does not scale its output with amplitude correctly.\
                                    {module["error"]}')
                else:
                    self.add_paragraph(f'Module {module["module"]} meets the amplitude scaling requirements')

                '''
                If want to add a table, add only select values, NOT like this
                rounded_measured_dbm = [round(x, 2) for x in module['magnitude_dbm_values']]
                rounded_predicted_dbm = [round(x, 2) for x in module['predicted_magnitude_dbm_values']]
                table_data = [['Amplitude', 'Pass/Fail', 'Measured (dBm)', 'Predicted (dBm)']] + \
                            list(zip(module['amplitude_values'], \
                                    ['Pass' if 0.9*predicted > abs(measured) > 1.1 *predicted  else 'Fail' \
                                    for measured, predicted in zip(rounded_measured_dbm, rounded_predicted_dbm)],
                                    rounded_measured_dbm,\
                                    rounded_predicted_dbm))
                self.add_table(table_data, f'Magnitude vs Amplitude at module {module["module"]}')
                '''
        
    def generate_dac_scale_transfer_section(self, data):
        self.add_section('DAC Scale Transfer Test')
        self.add_paragraph('Following the test of amplitude scaling, this test will determine if the DAC converter\
                           outputs signals of expected power level. The test will repeatedly call set_dac_scale with increasing values., which\
                            will change the current input into the balun. We expect the power output to scale linearly')

        for module in data['modules']:
            self.add_plot(f'{module["plot_path"]}', f'Magnitude vs scale at module {module["module"]}')
            
            rounded_measured_dbm = [round(x, 2) for x in module['magnitude_dbm_values']]
            rounded_predicted_dbm = [round(x, 2) for x in module['predicted_magnitude_dbm_values']]
            table_data = [['Scale', 'Pass/Fail', 'Measured (dBm)', 'Predicted (dBm)']] + \
                         list(zip(module['scale_values'], \
                                  ['Pass' if abs(measured - predicted) < 3  else 'Fail' \
                                   for measured, predicted in zip(rounded_measured_dbm, rounded_predicted_dbm)],
                                   rounded_measured_dbm,\
                                   rounded_predicted_dbm))
            self.add_table(table_data, f'Magnitude vs Scale Table at module {module["module"]}')

    
    def generate_dac_mixmode_section(self, data):
        self.add_section(f'DAC in Mixmode 1 and 2 {data["analog_bank"]} bank')
        if data['analog_bank'] == 'low':
            self.add_paragraph('This test aims to determine if the DAC accurately produces signals when a particular mixmode is set.\
                                Following the selected configuration of module-filter, we obtain the data for a single Nyquist region from a corresponding module.\
                                The graphs of the mixmodes shown below are stitched together from the 3 measurements to showcase the complete behavior of the system.\
                                Furthermore, data from the benchtop measurement is used to gauge the expected power output. The expected shapes should be consistent with \
                                NRZ mode in the first mixmode and RC mode in the second mixmode.')

        # Initialize the data structure for both banks
        combined_data = {
            'low': {1: {}, 2: {}},
            'high': {1: {}, 2: {}}
        }

        # Aggregate data for each mixmode and module, separated by bank
        for mixmode in data['mixmodes']:
            mode = mixmode['mixmode']
            module = mixmode['module']
            bank = data['analog_bank']  # Determine if it's low or high bank

            if module not in combined_data[bank][mode]:
                combined_data[bank][mode][module] = {
                    'frequencies': [],
                    'magnitudes': [],
                    'reference_frequencies': [],
                    'reference_magnitudes': []
                }

            combined_data[bank][mode][module]['frequencies'].extend(mixmode['frequencies'])
            combined_data[bank][mode][module]['magnitudes'].extend(mixmode['magnitude_dbm_values'])
            combined_data[bank][mode][module]['reference_frequencies'].extend(mixmode['reference_frequencies'])
            combined_data[bank][mode][module]['reference_magnitudes'].extend(mixmode['reference_magnitudes'])

        # Plot data for each mixmode and bank
        for bank_name in ['low', 'high']:
            for mode in combined_data[bank_name]:
                if not combined_data[bank_name][mode]:
                    continue  # Skip if no data for this bank and mixmode

                # The combined data for all modules in this bank and mixmode
                data_set_1 = combined_data[bank_name][mode]

                plot_path = self.plot_data_and_save(
                    plot_method='scatter',
                    x_set_1=None,  # This is None because the data is in the dictionary
                    data_set_1=data_set_1,  # Pass the dictionary directly
                    x_set_2=None,  # No need to pass this if you're handling references separately
                    data_set_2=None,
                    label_set_1='Measured Frequencies',
                    label_set_2='Reference Frequencies',
                    x_label='Frequency (GHz)',
                    y_label='Signal Magnitude (dBm)',
                    title=f'DAC output in mixmode #{mode} for {bank_name} bank',
                    filename=f'DAC_output_in_mixmode_{mode}_{bank_name}.png',
                    mixmode=True
                )
                self.add_plot(plot_path, f'Signal power vs normalized amplitude in Mixmode #{mode} for {bank_name} bank')



    
    def generate_adc_attenuation_section(self, data):
        self.add_section('ADC Attenuation Test')
        self.add_paragraph('This test aims to determine if the attenuation factor set at the ADC \
                           accurately scales the incoming signal and is consistent with the predictions from the ADC transfer function. \
                           The internal mechanics of this test are mirror images of the scale test above.\
                           The test repeatedly calls set_adc_attenuation() with increasing factors, which changes the current input to the balun.\
                           A linear relationship is expected.')

        for module in data['modules']:
            self.add_plot(f'{module["plot_path"]}', f'Magnitude vs attenuation at module {module["module"]}')

            rounded_measured_dbm = [round(x, 2) for x in module['magnitude_dbm_values']]
            rounded_predicted_dbm = [round(x, 2) for x in module['predicted_magnitude_dbm_values']]
            table_data = [['Attenuation', 'Pass/Fail', 'Measured (dBm)', 'Predicted (dBm)']] + \
                         list(zip(module['attenuation_values'], \
                                  ['Pass' if abs(measured - predicted) < 1  else 'Fail' \
                                   for measured, predicted in zip(rounded_measured_dbm, rounded_predicted_dbm)],
                                   rounded_measured_dbm,\
                                   rounded_predicted_dbm))
            self.add_table(table_data, f'Magnitude vs Attenuation Table at module {module["module"]}')


    def generate_loopback_noise_section(self, data):
        self.add_section('Loopback Noise Test')
        self.add_paragraph('Firstly, to study the noise environment, a pure noise measurement will be taken. Setting carrier amplitude\
                            to 0 and calculating the PSD from the ADC samples yields the first graph. The expectation of the mean whilte noise level \
                           is 163dBm +- 5 percent.')
        self.add_paragraph('Secondly, we study the noise behaviour when it is entirely dominate by 1/f slope. The fitting should reveal the \
                           factor of n = -1 within an acceptable margin (now what IS that margin, still need to set).')


        for module in data['modules']:

            '''
            UNDER CONSTRUCTION. Need to add NCR calculation, use more channels, and have a graph that has a complete\
                            fit with the 1/f AND the noise floor (very close on that, figuring out final quirks of the fitting).
            '''
            self.add_plot(f'{module["plot_path_white"]}', f'Noise floor in loopback at module {module["module"]}')
                    
            if any(noise > -140 for noise in module['psd_i_white']) or any(noise > -140 for noise in module['psd_q_white']):
                self.add_paragraph(f'FAIL: module {module["module"]} produces exessive noise >-140dBm')

            else:
                self.add_paragraph(f'PASS: noise from module {module["module"]} is within expected bounds <-140dBm')

            self.add_plot(f'{module["plot_path_1_f"]}', f'IQ Standard deviation at module {module["module"]}')
            expected_slope = -1
            if abs((module['slope'] - expected_slope) / expected_slope) > 0.1:
                self.add_paragraph("Observe the slope of the 1/f noise deviates from the expected value of -1 (insert potential reason why).")

    def generate_wideband_fft_section(self, data):
        self.add_section('Wideband FFT')
        self.add_paragraph('Performed a 60000 sample measurement and took a snapshot fft of the target band 0 - 2.5GHz (Nyquist)\
                            to get an FFT with frequency resolution of 83kHz. The values on the units are a little nuts though')

        for module in data['modules']:
            self.add_plot(f'{module["plot_path"]}', f'Wideband FFT in loopback at module {module["module"]}')


def generate_report_from_data(serial, data_file, results_dir, summary_file):
    with open(data_file, 'r') as file:
        data = json.load(file)
        
    with open(summary_file, 'r') as file:
        summary_data = json.load(file)

    report = PDFReport(serial, results_dir)

    # Add summary section
    report.generate_summary(summary_data)

    for section in data['tests']:
        if section['title'] == 'Summary of Results':
            report.generate_summary(section)
        if section['title'] == 'Temperature Test':
            report.generate_temperature_section(section)
        elif section['title'] == 'Voltage Test':
            report.generate_voltage_section(section)
        elif section['title'] == 'Current Test':
            report.generate_current_section(section)
        elif section['title'] == 'Loopback Noise Test':
            report.generate_loopback_noise_section(section)
        elif section['title'] == 'DAC Scale Transfer Test':
            report.generate_dac_scale_transfer_section(section)
        elif section['title'] == 'DAC Amplitude Transfer Test':
            report.generate_dac_amplitude_transfer_section(section)
        elif section['title'].startswith('DAC Mixmodes Test'):
            report.generate_dac_mixmode_section(section)
        elif section['title'] == 'ADC Attenuation Test':
            report.generate_adc_attenuation_section(section)
        elif section['title'] == 'DAC Passband Test':
            report.generate_dac_passband_section(section)
        elif section['title'] == 'Wideband FFT for Noise Detection':
            report.generate_wideband_fft_section(section)

        #elif section['title'] == 'ADC Even Phase Test':
            # report.generate_adc_even_phase_section(section)

    report.save()


    '''
    def generate_adc_even_phase_section(self, data):
        self.add_section('ADC Even Phase Test')
        self.add_paragraph('To complete the evalution of the noise performance, raw I and Q samples will\
                           be plotted on a scatter plot. The readout is expected to have to have equal amplitude and phase noise\
                           resulting in a circular scatter plot. However, in the past this has been proven difficult with a dominant amplitude noise.')


        for module in data['modules']:
            plot_path = self.plot_data_and_save('scatter',
                module['data_i_samples'],
                module['data_q_samples'],
                None,
                'Measured at ADC SMA',
                'predicted = none',
                'I (Normalized)',
                'Q (Normalized))',
                f'I vs Q Scatter Plot for Module {module["module"]}',
                f'I vs Q Scatter Plot for Module {module["module"]}.png'
            )
            self.add_plot(plot_path, f'I vs Q Noise at module {module["module"]}')
            
            if np.isclose(module['ratio'], 1, atol=0.1):
                self.add_paragraph(f'The signal at module {module["module"]} has balanced I-Q noise')
            else:
                self.add_paragraph(f'The signal at module {module["module"]} has scewed I-Q components')

            #table_data = [['Eigenvalue Ratio']] + [[module['ratio']]]
            #self.add_table(table_data, f'Eigenvalue Ratio for Module {module["module"]}')
    '''
