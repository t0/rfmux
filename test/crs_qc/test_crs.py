#!/usr/bin/env -S pytest-3 -s
"""
Datapath tests

Please see README for hints on using and extending these test cases.
"""

import htpy
import os
import pytest
import rfmux
import time
import numpy as np
import matplotlib.pyplot as plt

from .crs_qc import render_markdown, ResultTable
from scipy import signal
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def transfer_function_helper(raw_sampl, AMPLITUDE, scale, attenuation):
    """
    Calculates the dBm value of a sample from get_samples call.
    Predicts the dBm value of the signal from the parameters passed to it
    """
    DB_LOSS_FACTOR = -2.56
    volts_peak_samples = (
        (np.sqrt(2)) * np.sqrt(50 * (10 ** (-1.75 / 10)) / 1000) / 1880796.4604246316
    )
    if np.iscomplexobj(raw_sampl):
        data_i = np.array(raw_sampl.real) * volts_peak_samples
        data_q = np.array(raw_sampl.imag) * volts_peak_samples
    else:
        data_i = np.array(raw_sampl.i) * volts_peak_samples
        data_q = np.array(raw_sampl.q) * volts_peak_samples
    magnitude = np.sqrt(data_i**2 + data_q**2)
    median_magnitude = np.median(magnitude)
    median_magnitude_dbm = (
        10 * np.log10(((median_magnitude / np.sqrt(2)) ** 2) / 50) + 30
    )

    DAC_output_dBm = DB_LOSS_FACTOR + 10 * np.log10((AMPLITUDE**2) * 10 ** (scale / 10))
    predicted_magnitude_dbm = DAC_output_dBm - attenuation
    return median_magnitude_dbm, predicted_magnitude_dbm


async def value_setter_helper(
    d, frequency, amplitude, scale, attenuation, channel, module, nco
):
    """

    NOTE: All the tests below are written with the following numbering conventions for highbanks
    set_amplitude, set_frequency, get_samples, set_nyqist_zone -- accept module 1-4
    set_scale, set_attenuation, set_nco -- accept module 1-8

    """

    async with d.tuber_context() as ctx:
        ctx.clear_channels()
        ctx.set_frequency(frequency, channel, module)
        ctx.set_amplitude(amplitude, d.UNITS.NORMALIZED, channel, module)
        ctx.set_dac_scale(scale, d.UNITS.DBM, module)
        ctx.set_adc_attenuator(attenuation, d.UNITS.DB, module=module)
        ctx.set_nco_frequency(nco, module=module)


# small legacy helper used in checking how circular the i-q blob is
def covariance_matrix(samples):
    cov_matrix = np.cov(samples, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    ratio = eigenvalues[1] / eigenvalues[0]
    return ratio


"""------------------------------------------------------------------------------------------------------------"""
"""-------------------------------------------DAC--------------------------------------------------------------"""


@pytest.mark.qc_stage2
@pytest.mark.asyncio
async def test_dac_passband(d, request, shelf, check):
    """
    ## DAC Passband Checks

    Validate:

    * Amplitude variation is less than 6 dB from nominal across the passband
    """
    render = [render_markdown(test_dac_passband.__doc__)]

    # FIXME: Some issue with the higher range of the frequencies showing a dip
    AMPLITUDE = 0.005
    SAMPLES = 10
    NCO_FREQUENCY = 1e9 + 123
    ATTENUATION = 0
    SCALE = 7

    MARGIN_DBM = 6

    # Feel free to increase the range and density of frequencies
    frequencies = np.arange(-250e6, 250e6, step=1e6)
    channels = np.arange(1, 1001)
    nruns = int(np.ceil(len(frequencies) / len(channels)))
    lenrun = min(len(frequencies), len(channels))

    # Set all parameters that won't change: attenuation, scale, NCO
    async with d.tuber_context() as ctx:
        ctx.clear_channels()
        for module in d.modules:
            ctx.set_dac_scale(SCALE, d.UNITS.DBM, module=module.module)
            ctx.set_adc_attenuator(ATTENUATION, d.UNITS.DB, module=module.module)
            ctx.set_nco_frequency(NCO_FREQUENCY, module=module.module)

    # Serialize the test over each module
    for m in d.modules:

        measured_dbm = np.zeros_like(frequencies)
        predicted_dbm = np.zeros_like(frequencies)

        for nrun in range(nruns):
            start_idx = nrun * lenrun
            end_idx = min(start_idx + lenrun, len(frequencies))
            time.sleep(0.2)

            # Set necessary channels for the run
            async with d.tuber_context() as ctx:
                for idx in range(start_idx, end_idx):
                    channel = int(channels[idx - start_idx])
                    ctx.set_frequency(frequencies[idx], channel, module=m.module)
                    ctx.set_amplitude(
                        AMPLITUDE, d.UNITS.NORMALIZED, channel, module=m.module
                    )
                for idx in range(end_idx, start_idx + lenrun):
                    ctx.set_amplitude(0, d.UNITS.NORMALIZED, channel, module=m.module)

            # Get all the samples for the current run
            raw = await d.py_get_samples(SAMPLES, channel=None, module=m.module)

            for idx in range(start_idx, end_idx):
                channel = int(channels[idx - start_idx])
                rchan = np.array(raw.i[channel + 1]) + 1j * np.array(raw.q[channel + 1])
                measured, predicted = transfer_function_helper(
                    rchan, AMPLITUDE, SCALE, ATTENUATION
                )
                measured_dbm[idx] = measured
                predicted_dbm[idx] = predicted

        mag_0 = measured_dbm[np.argmin(np.abs(frequencies))]

        rt = ResultTable("Metric", "Measurement", "Nominal", "Minimum", "Maximum")

        if check.between(mag_0, mag_0 - MARGIN_DBM, mag_0 + MARGIN_DBM):
            rt.pass_(
                "Gain at NCO Centre",
                f"{mag_0:.1f} dBm",
                f"{predicted:.1f} dBm",
                f"{predicted-MARGIN_DBM:.1f} dBm",
                f"{predicted+MARGIN_DBM:.1f} dBm",
            )
        else:
            rt.fail(
                "Gain at NCO Centre",
                f"{mag_0:.1f} dBm",
                f"{predicted:.1f} dBm",
                f"{predicted-MARGIN_DBM:.1f} dBm",
                f"{predicted+MARGIN_DBM:.1f} dBm",
            )

        errors = np.abs(measured_dbm - predicted) > MARGIN_DBM
        check.is_false(any(errors))

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, measured_dbm, "-,", label="measured")
        plt.plot(frequencies[errors], measured_dbm[errors], "ro", fillstyle="none")
        plt.plot(frequencies, predicted_dbm, "-,", label="predicted")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Magnitude (dBm)")
        plt.legend()
        # plt.title(f'Signal at DAC of module {m.module} with increasing frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{request.session.results_dir}/{request.node.name}-{m.module}.png")
        plt.close()

        render.extend(
            [
                htpy.h3[f"Module {m.module}"],
                rt,
                htpy.figure[
                    htpy.img(src=f"{request.node.name}-{m.module}.png"),
                    htpy.figcaption[f"ADC Frequency Sweep, Module {m.module}"],
                ],
            ]
        )

    with shelf as x:
        x["sections"][request.node.nodeid] = render


@pytest.mark.qc_stage2
@pytest.mark.asyncio
async def test_dac_amplitude_transfer(d, request, shelf, check):
    """
    ## DAC Amplitude Transfer Checks

    This test aims to determine if the output of the dac is scaled
    appropriately with the change in amplitude. The algorithm uses only 1
    channel at each module a set frequency, varying its amplitude. The result
    is compared to the transfer function prediction
    """
    render = [render_markdown(test_dac_amplitude_transfer.__doc__)]

    SCALE = 1
    SAMPLES = 100
    ATTENUATION = 0
    CHANNEL = 1
    NCO_FREQUENCY = 0
    FREQUENCY = 150.5763e6
    NOMINAL_AMPLITUDE = 0

    # feel free to adjust the amplitude density as needed
    amplitudes = np.arange(0.05, 1, 0.05)
    await d.clear_channels()

    for m in d.modules:
        module_errors = []
        amplitude_values = []
        magnitude_dbm_values = []
        predicted_magnitude_dbm_values = []

        # When starting to test a new module, clear all channels and use the value_setter to set the variables that won't change attenuation, scale, frequency etc
        await value_setter_helper(
            d,
            FREQUENCY,
            NOMINAL_AMPLITUDE,
            SCALE,
            ATTENUATION,
            CHANNEL,
            m.module,
            NCO_FREQUENCY,
        )

        for amplitude in amplitudes:
            await d.set_amplitude(amplitude, d.UNITS.NORMALIZED, CHANNEL, m.module)
            samples = await d.py_get_samples(SAMPLES, channel=CHANNEL, module=m.module)
            median_magnitude_dbm, predicted_magnitude_dbm = transfer_function_helper(
                samples, amplitude, SCALE, ATTENUATION
            )
            amplitude_values.append(amplitude)
            magnitude_dbm_values.append(median_magnitude_dbm)
            predicted_magnitude_dbm_values.append(predicted_magnitude_dbm)

        errors = np.abs(
            np.array(magnitude_dbm_values) - np.array(predicted_magnitude_dbm_values)
        )

        # Linearity check on the scaling: apply linear regression and check least square error
        log_amplitude_values = np.array(np.log10(amplitude_values)).reshape(-1, 1)
        regressor = LinearRegression().fit(log_amplitude_values, magnitude_dbm_values)
        linearity = r2_score(
            regressor.predict(log_amplitude_values), magnitude_dbm_values
        )

        # if any value is 3dB from predicted -- fail
        if np.any(errors >= 3):
            check.fail(f"Module {m.module}: power output deviated by > 3 dB")

        if linearity < 0.999:
            check.fail("The output is not linear on a lin-log scale.")

        plt.figure(figsize=(10, 6))
        plt.plot(
            amplitude_values,
            magnitude_dbm_values,
            "-o",
            label="measured",
        )
        plt.plot(
            amplitude_values,
            predicted_magnitude_dbm_values,
            "-o",
            label="predicted",
        )
        plt.xscale("log")
        plt.xlabel("Amplitude (Normalized)")
        plt.ylabel("Magnitude (dBm)")
        plt.legend()
        plt.title(f"DAC output of module {m.module} with increasing amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{request.session.results_dir}/{request.node.name}-{m.module}.png")
        plt.close()

        render.extend(
            [
                htpy.h3[f"Module {m.module}"],
                htpy.figure[
                    htpy.img(src=f"{request.node.name}-{m.module}.png"),
                    htpy.figcaption[f"DAC Amplitude Transfer Function, {m.module}"],
                ],
            ]
        )

    with shelf as x:
        x["sections"][request.node.nodeid] = render


@pytest.mark.qc_stage2
@pytest.mark.asyncio
async def test_dac_scale_transfer(d, request, shelf, check):
    """
    ## DAC Scale Tests

    This test aims to determine if the output of the dac is scaled
    appropriately with the change in 'scale' value (actually controlled by
    varying the current in the balun). The algorithm uses only 1 channel at
    each module a set frequency, varying its scale.  The result is compared to
    the transfer function prediction

    Potentially add:
    * Pick 4 channels, set the same frequency-> alter scale and plot the
      samples first, then convert to dBm
    * plot together with prediction from the transfer function
    """
    render = [render_markdown(test_dac_scale_transfer.__doc__)]

    # set parameters for testing
    AMPLITUDE = 1
    NOMINAL_SCALE = 1
    SAMPLES = 100
    FREQUENCY = 150.5763e6
    NCO_FREQUENCY = 0
    ATTENUATION = 0
    CHANNEL = 1
    TITLE = "DAC Scale Transfer Test"
    # dictionary just for this test will be appended to all the results at the end (could modify to avoid losing data if a pytest crash occurs)
    test_result = {"title": TITLE, "modules": []}

    for m in d.modules:
        await d.set_adc_calibration_mode("MODE1", m.module)

    for m in d.modules:
        # Initialize lists to store the scale and magnitude values
        scale_values = []
        magnitude_dbm_values = []
        predicted_magnitude_dbm_values = []

        await d.clear_channels()
        await value_setter_helper(
            d,
            FREQUENCY,
            AMPLITUDE,
            NOMINAL_SCALE,
            ATTENUATION,
            CHANNEL,
            m.module,
            NCO_FREQUENCY,
        )

        for scale in range(0, 8):

            await d.set_dac_scale(scale, d.UNITS.DBM, m.module)
            raw_sampl = await d.py_get_samples(
                SAMPLES, channel=CHANNEL, module=m.module
            )
            median_magnitude_dbm, predicted_magnitude_dbm = transfer_function_helper(
                raw_sampl, AMPLITUDE, scale, ATTENUATION
            )

            scale_values.append(scale)
            magnitude_dbm_values.append(median_magnitude_dbm)
            predicted_magnitude_dbm_values.append(predicted_magnitude_dbm)

        # Power check on the DAC
        errors = np.abs(
            np.array(magnitude_dbm_values) - np.array(predicted_magnitude_dbm_values)
        )

        if np.any(errors >= 3):
            check.fail(
                f"Module {m.module} is generating a power output below 3dB of expected"
            )

        # Linearity check on the scaling: apply linear regression and check least square error
        scale_values = np.array(scale_values).reshape(-1, 1)
        regressor = LinearRegression().fit(scale_values, magnitude_dbm_values)
        linearity = r2_score(regressor.predict(scale_values), magnitude_dbm_values)
        if linearity < 0.99:
            check.fail("linearity check failed")

        plt.figure(figsize=(10, 6))
        plt.plot(
            scale_values.flatten().tolist(),
            magnitude_dbm_values,
            "-o",
            label="measured",
        )
        plt.plot(
            scale_values.flatten().tolist(),
            predicted_magnitude_dbm_values,
            "-o",
            label="predicted",
        )
        plt.xlabel("Scale(dBm)")
        plt.ylabel("Magnitude (dBm)")
        plt.legend()
        plt.title(f"DAC output of module {m.module} with increasing scale (dBm)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{request.session.results_dir}/{request.node.name}-{m.module}.png")
        plt.close()

        render.extend(
            [
                htpy.h3[f"Module {m.module}"],
                htpy.figure[
                    htpy.img(src=f"{request.node.name}-{m.module}.png"),
                    htpy.figcaption[f"ADC Frequency Sweep, Module {m.module}"],
                ],
            ]
        )

    with shelf as x:
        x["sections"][request.node.nodeid] = render


@pytest.mark.xfail
@pytest.mark.qc_stage2
@pytest.mark.asyncio
async def test_dac_mixmodes(d, results, summary_results, test_highbank):
    """
    This test aims to verify the overall behaviour of the nyquist zones (1(NRZ) or 2(RC)) set in the firmware as well as a sanity check for the hardware response in
    the higher nquist regions.
    The test uses 3 modules at a time, getting the behaviour of a singular nyquist region from a singular module. The report generator then will stitch the
    data together into one graph per nyquist zone.
    """

    MAX_TOTAL_FREQUENCY = 2.5e9  # in Hz
    DAC_MAX_FREQUENCY = 275e6  # in Hz (max DAC frequency)
    BANDWIDTH = 550e6  # in Hz (275 MHz on each side of NCO)
    AMPLITUDE = 1
    SCALE = 1
    SAMPLES = 100
    ATTENUATION = 0
    STEP = 50e6
    SAMPLING_FREQUENCY = 2.5e9  # in Hz
    MODULE_NYQUIST_MAP = {1: 1, 2: 2, 3: 3}
    CHANNEL = 1

    # obtained through a separate loopback measurement: transferfunction applied to a 1000-count get_samples call
    reference_data = {
        "mixmode_1": {
            "mod_region_1": {
                "frequencies": [761.78e6, 1302.56e6, 1902.34e6],
                "magnitudes": [-3.42, -4.82, -6.7],
            },
            "mod_region_2": {
                "frequencies": [4338.22e6, 3697.44e6, 3097.66e6],
                "magnitudes": [-24.87, -19.00, -13.82],
            },
            "mod_region_3": {
                "frequencies": [5761.78e6, 6302.56e6, 6902.34e6],
                "magnitudes": [-31.89, -27.84, -27.87],
            },
        },
        "mixmode_2": {
            "mod_region_1": {
                "frequencies": [761.78e6, 1302.56e6, 1902.34e6],
                "magnitudes": [-16.41, -12.24, -10.21],
            },
            "mod_region_2": {
                "frequencies": [4238.22e6, 3697.44e6, 3097.66e6],
                "magnitudes": [-12.30, -11.73, -10.01],
            },
            "mod_region_3": {
                "frequencies": [5761.78e6, 6302.56e6, 6902.34e6],
                "magnitudes": [-19.66, -20.60, -24.93],
            },
        },
    }

    async def mixmode_test_for_bank(bank_name, is_highbank):
        test_result = {
            "title": f"DAC Mixmodes Test - {bank_name} bank",
            "analog_bank": bank_name,
            "mixmodes": [],
        }

        await d.set_analog_bank(is_highbank)

        for mixmode in range(1, 3):
            for module, nyquist_zone in MODULE_NYQUIST_MAP.items():
                module_setter = module
                if is_highbank:
                    module += 4

                module_errors = []
                module_magnitude_dbm_values = np.array([])
                frequencies = np.array([])
                print(
                    f"Running module {module} in mixmode #{mixmode} nyquist region #{nyquist_zone} for {bank_name} bank"
                )

                await d.set_nyquist_zone(mixmode, module=module)
                nco = 0
                frequency = -DAC_MAX_FREQUENCY

                while nco < MAX_TOTAL_FREQUENCY:
                    while (
                        frequency <= DAC_MAX_FREQUENCY
                        and nco + frequency < MAX_TOTAL_FREQUENCY
                    ):
                        if nco + frequency >= 0:
                            await value_setter_helper(
                                d,
                                frequency,
                                AMPLITUDE,
                                SCALE,
                                ATTENUATION,
                                CHANNEL,
                                module_setter,
                                nco,
                            )
                            samples = await d.py_get_samples(
                                SAMPLES, channel=CHANNEL, module=module_setter
                            )
                            module_magnitude_dbm, _ = transfer_function_helper(
                                samples, AMPLITUDE, SCALE, ATTENUATION
                            )

                            if nyquist_zone == 1:
                                frequencies = np.append(frequencies, frequency + nco)
                                module_magnitude_dbm_values = np.append(
                                    module_magnitude_dbm_values, module_magnitude_dbm
                                )
                            elif nyquist_zone == 2:
                                frequencies = np.append(
                                    frequencies,
                                    2 * SAMPLING_FREQUENCY - (frequency + nco),
                                )
                                module_magnitude_dbm_values = np.append(
                                    module_magnitude_dbm_values, module_magnitude_dbm
                                )
                            elif nyquist_zone == 3:
                                frequencies = np.append(
                                    frequencies,
                                    2 * SAMPLING_FREQUENCY + frequency + nco,
                                )
                                module_magnitude_dbm_values = np.append(
                                    module_magnitude_dbm_values, module_magnitude_dbm
                                )
                        frequency += STEP
                    nco += BANDWIDTH
                    frequency = -DAC_MAX_FREQUENCY

                reference_freqs = np.array(
                    reference_data[f"mixmode_{mixmode}"][f"mod_region_{nyquist_zone}"][
                        "frequencies"
                    ]
                )
                reference_mags = np.array(
                    reference_data[f"mixmode_{mixmode}"][f"mod_region_{nyquist_zone}"][
                        "magnitudes"
                    ]
                )

                for reference_freq, reference_mag in zip(
                    reference_freqs, reference_mags
                ):
                    if len(frequencies) == 0:
                        check.fail(
                            f"No frequencies measured for module {module} in Nyquist region {nyquist_zone}"
                        )
                        continue

                    closest_index = np.argmin(np.abs(frequencies - reference_freq))
                    obtained_magnitude = module_magnitude_dbm_values[closest_index]

                    if not (
                        0.8 * abs(reference_mag)
                        <= abs(obtained_magnitude)
                        <= 1.15 * abs(reference_mag)
                    ):
                        check.fail(
                            f"Frequency {frequencies[closest_index] / 1e6:.2f} MHz: measured {obtained_magnitude:.2f} dBm, expected {reference_freq/1e6:.2f} to have {reference_mag} dBm"
                        )

                test_result["mixmodes"].append(
                    {
                        "mixmode": mixmode,
                        "module": module,
                        "frequencies": frequencies.tolist(),
                        "magnitude_dbm_values": module_magnitude_dbm_values.tolist(),
                        "reference_frequencies": reference_freqs.tolist(),
                        "reference_magnitudes": reference_mags.tolist(),
                    }
                )
                if module not in summary_results["summary"]:
                    summary_results["summary"][module] = {}
                if mixmode == 2:
                    await d.set_nyquist_zone(1, module=module)

        results["tests"].append(test_result)

    # Run test for low bank
    await mixmode_test_for_bank("low", False)


"""---------------------------------------------------------------------------------------------------------------"""
"""--------------------------------------------------ADC----------------------------------------------------------"""


@pytest.mark.xfail
@pytest.mark.qc_stage2
@pytest.mark.asyncio
async def test_adc_attenuation(
    d, results, summary_results, test_highbank, num_runs, results_dir
):
    AMPLITUDE = 1
    SAMPLES = 100
    FREQUENCY = 80.234562e6
    NCO_FREQUENCY = 0
    SCALE = 1
    CHANNEL = 1
    ATTENUATION_0 = 0
    TITLE = "ADC Attenuation Test"

    test_result = {"title": TITLE, "modules": []}

    for num_run in range(0, num_runs):
        for m in d.modules:
            module_errors = []
            attenuation_values = []
            magnitude_dbm_values = []
            predicted_magnitude_dbm_values = []

            await d.clear_channels()
            await value_setter_helper(
                d,
                FREQUENCY,
                AMPLITUDE,
                SCALE,
                ATTENUATION_0,
                CHANNEL,
                m.module,
                NCO_FREQUENCY,
            )

            print(f"ADC_attenuation - Testing Module #{m.module}")

            for attenuation in range(0, 15):
                await d.set_adc_attenuator(attenuation, d.UNITS.DB, module=m.module)
                raw_sampl = await d.py_get_samples(
                    SAMPLES, channel=CHANNEL, module=m.module
                )
                median_magnitude_dbm, predicted_magnitude_dbm = (
                    transfer_function_helper(raw_sampl, AMPLITUDE, SCALE, attenuation)
                )
                attenuation_values.append(attenuation)
                magnitude_dbm_values.append(median_magnitude_dbm)
                predicted_magnitude_dbm_values.append(predicted_magnitude_dbm)

            attenuation_values = np.array(attenuation_values).reshape(-1, 1)
            regressor = LinearRegression().fit(attenuation_values, magnitude_dbm_values)
            linearity = r2_score(
                magnitude_dbm_values, regressor.predict(attenuation_values)
            )

            if linearity < 0.99:
                check.fail(f"linearity check failed")

            if np.any(
                np.abs(
                    np.array(magnitude_dbm_values)
                    - np.array(predicted_magnitude_dbm_values)
                )
                >= 1
            ):
                check.fail(
                    f"Module {m.module} is generating a power output lower than expected. Check your balun and SMA connections."
                )

            module_result = {
                "module": m.module,
                "attenuation_values": attenuation_values.flatten().tolist(),
                "magnitude_dbm_values": magnitude_dbm_values,
                "predicted_magnitude_dbm_values": predicted_magnitude_dbm_values,
                "plot_path": "plot_path",
            }

            graph_name = TITLE + "_module_" + f"{module_result[m.module]}" + ".png"
            plot_path = os.path.join(results_dir, "plots", graph_name)
            module_result["plot_path"] = plot_path

            plt.figure(figsize=(10, 6))
            plt.plot(
                module_result["attenuation_values"],
                module_result["magnitude_dbm_values"],
                "-o",
                label="measured",
            )
            plt.plot(
                module_result["attenuation_values"],
                module_result["predicted_magnitude_dbm_values"],
                "-o",
                label="predicted",
            )
            plt.xlabel("Attenuation(dBm)")
            plt.ylabel("Magnitude (dBm)")
            plt.legend()
            plt.title(
                f'Signal at ADC of module {module_result["module"]} with increasing attenuation'
            )
            plt.grid(True)
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

            test_result["modules"].append(module_result)

            if module_set not in summary_results["summary"]:
                summary_results["summary"][module_set] = {}
            summary_results["summary"][module_set][f"ADC Atten"] = module_result[
                "status"
            ]

    results["tests"].append(test_result)


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.qc_stage2
@pytest.mark.asyncio
async def test_adc_even_phase(d, results):
    AMPLITUDE = 0.1
    SAMPLES = 100
    FREQUENCY = 1000
    NCO_FREQUENCY = 0
    SCALE = 1
    CHANNEL = 1
    module = 1
    test_result = {"title": "ADC Even Phase Test", "modules": []}
    for module in range(1, 5):
        await value_setter_helper(
            d, FREQUENCY, AMPLITUDE, SCALE, 0, CHANNEL, module, NCO_FREQUENCY
        )
        x = await d.py_get_samples(SAMPLES, channel=1, module=1)

        data_i_samples = np.array(x.i)
        data_q_samples = np.array(x.q)
        data_samples = np.column_stack((data_i_samples, data_q_samples))
        ratio = covariance_matrix(data_samples)
        print(f"Eigenvalue ratio: {ratio}")
        module_result = {
            "module": module,
            "module": module,
            "data_i_samples": data_i_samples.tolist(),
            "data_q_samples": data_q_samples.tolist(),
            "ratio": ratio,
        }
        test_result["modules"].append(module_result)

    results["tests"].append(test_result)

    if np.isclose(ratio, 1, atol=0.1):
        print("The cluster is approximately circular.")
    else:
        print("The cluster is more oval in shape.")


@pytest.mark.qc_stage2
@pytest.mark.asyncio
async def test_wideband_noise(d, request, shelf, check):
    """
    ## Wideband Noise Checks
    """
    render = [render_markdown(test_wideband_noise.__doc__)]

    SAMPLING_FREQUENCY = 5e9  # 5 GHz
    NUM_SAMPLES = 4000  # Total number of samples

    for m in d.modules:
        await d.clear_channels()
        time.sleep(1)
        samples = await d.get_fast_samples(NUM_SAMPLES, module=m.module)

        volts_peak_samples = (
            (np.sqrt(2))
            * np.sqrt(50 * (10 ** (-1.75 / 10)) / 1000)
            / 1880796.4604246316
        )
        I_signal = np.array(samples.i) * volts_peak_samples
        Q_signal = np.array(samples.q) * volts_peak_samples

        psdfreq_i, psd_i = signal.welch(
            I_signal / np.sqrt(4 * 50), SAMPLING_FREQUENCY, nperseg=NUM_SAMPLES
        )  # tod_i / np.sqrt(4 * 50)
        psdfreq_q, psd_q = signal.welch(
            Q_signal / np.sqrt(4 * 50), SAMPLING_FREQUENCY, nperseg=NUM_SAMPLES
        )  # tod_q / np.sqrt(4 * 50)

        # Convert from watts to dBm
        psd_i = 10 * np.log10(psd_i * 1000)
        psd_q = 10 * np.log10(psd_q * 1000)

        # plot the white noise
        plt.figure(figsize=(10, 6))
        plt.plot(psdfreq_i.tolist(), psd_i.tolist(), label="I")
        plt.plot(psdfreq_q.tolist(), psd_q.tolist(), label="Q")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dBm/Hz)")
        plt.legend()
        plt.title(f"Wideband FFT at module {m.module}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{request.session.results_dir}/{request.node.name}-{m.module}.png")
        plt.close()

        render.extend(
            [
                htpy.h3[f"Module {m.module}"],
                htpy.figure[
                    htpy.img(src=f"{request.node.name}-{m.module}.png"),
                    htpy.figcaption[f"Wideband Noise, Module {m.module}"],
                ],
            ]
        )

    with shelf as x:
        x["sections"][request.node.nodeid] = render
