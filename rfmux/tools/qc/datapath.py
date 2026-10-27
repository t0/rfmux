"""
Datapath tests

Please see README for hints on using and extending these test cases.
"""

import inspect
import numpy as np
import os
import rfmux
import time

from .crs_qc import CheckTable, render_markdown, render_plot, ResultTable
from scipy import signal
from scipy.stats import linregress


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


async def dac_passband(d, report, module):
    """
    ## DAC Passband Checks

    Validate:

    * Amplitude stays within the limits below of the prediction across the
      passband
    """
    report.append(render_markdown(inspect.getdoc(dac_passband)))

    AMPLITUDE = 0.005
    SAMPLES = 10
    NCO_FREQUENCY = 1e9 + 123
    ATTENUATION = 0
    SCALE = 7

    MARGIN_DB = 6

    # Feel free to increase the range and density of frequencies
    frequencies = np.arange(-250e6, 250e6, step=1e6)
    channels = np.arange(1, 1001)
    nruns = int(np.ceil(len(frequencies) / len(channels)))
    lenrun = min(len(frequencies), len(channels))

    # Set all parameters that won't change: attenuation, scale, NCO
    async with d.tuber_context() as ctx:
        ctx.clear_channels()
        ctx.set_dac_scale(SCALE, d.UNITS.DBM, module=module)
        ctx.set_adc_attenuator(ATTENUATION, d.UNITS.DB, module=module)
        ctx.set_nco_frequency(NCO_FREQUENCY, module=module)

    report.append(checks := CheckTable())

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
                ctx.set_frequency(frequencies[idx], channel, module=module)
                ctx.set_amplitude(
                    AMPLITUDE, d.UNITS.NORMALIZED, channel, module=module
                )
            for idx in range(end_idx, start_idx + lenrun):
                ctx.set_amplitude(0, d.UNITS.NORMALIZED, channel, module=module)

        # Get all the samples for the current run
        raw = await d.py_get_samples(SAMPLES, channel=None, module=module)

        for idx in range(start_idx, end_idx):
            channel = int(channels[idx - start_idx])
            rchan = np.array(raw.i[channel - 1]) + 1j * np.array(raw.q[channel - 1])
            measured, predicted = transfer_function_helper(
                rchan, AMPLITUDE, SCALE, ATTENUATION
            )
            measured_dbm[idx] = measured
            predicted_dbm[idx] = predicted

    mag_0 = measured_dbm[np.argmin(np.abs(frequencies))]

    deviation = np.abs(measured_dbm - predicted_dbm)
    errors = deviation > MARGIN_DB
    checks.check("Gain at NCO centre", mag_0, "dBm", nominal=predicted,
                 low=predicted - MARGIN_DB, high=predicted + MARGIN_DB, module=module)
    checks.check("Largest deviation from prediction", deviation.max(), "dB",
                 high=MARGIN_DB, module=module)

    def plot_passband(data):
        import matplotlib.pyplot as plt
        import numpy as np

        frequency = np.asarray(data["frequency_mhz"])
        measured = np.asarray(data["measured_dbm"])
        errors = np.asarray(data["errors"], dtype=bool)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frequency, measured, "-,", label="measured")
        ax.plot(frequency[errors], measured[errors], "ro", fillstyle="none",
                label="outside margin")
        ax.plot(frequency, data["predicted_dbm"], "-,", label="predicted")
        ax.set_xlabel("Frequency from NCO (MHz)")
        ax.set_ylabel("Magnitude (dBm)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig

    report += [
        render_plot(plot_passband, dict(
            module=module,
            frequency_mhz=frequencies / 1e6,
            measured_dbm=measured_dbm,
            predicted_dbm=predicted_dbm,
            errors=errors,
        ), f"DAC Passband, Module {module}"),
    ]


async def dac_amplitude_transfer(d, report, module):
    """
    ## DAC Amplitude Transfer Checks

    - **Objective**: Ensure variations in DAC amplitude are reflected in readout samples.
    - **Expected Outcome**: A plot showing green circles on data points, within the grey shape showing acceptable measurements.
    - **Limitations:** Uses only channel 1
    """
    SCALE = 1
    SAMPLES = 100
    ATTENUATION = 0
    CHANNEL = 1
    NCO_FREQUENCY = 0
    FREQUENCY = 150.5763e6
    NOMINAL_AMPLITUDE = 0
    MAX_ERROR_DB = 3  # dB

    report.append(render_markdown(inspect.getdoc(dac_amplitude_transfer)))
    report.append(rt := ResultTable("Parameter", "Value"))
    rt.row("SCALE", SCALE)
    rt.row("SAMPLES", SAMPLES)
    rt.row("ATTENUATION", ATTENUATION)
    rt.row("CHANNEL", CHANNEL)
    rt.row("NCO_FREQUENCY", NCO_FREQUENCY)
    rt.row("FREQUENCY", f"{FREQUENCY/1e6:.3g} MHz")
    rt.row("NOMINAL_AMPLITUDE", NOMINAL_AMPLITUDE)
    rt.row("MAX_ERROR_DB", f"{MAX_ERROR_DB} dB")

    # feel free to adjust the amplitude density as needed
    amplitudes = np.arange(0.05, 1, 0.05)
    await d.clear_channels()

    report.append(checks := CheckTable())

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
        module,
        NCO_FREQUENCY,
    )

    for amplitude in report.track(amplitudes):
        await d.set_amplitude(amplitude, d.UNITS.NORMALIZED, CHANNEL, module)
        samples = await d.py_get_samples(SAMPLES, channel=CHANNEL, module=module)
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
    linearity = (
        linregress(log_amplitude_values.ravel(), magnitude_dbm_values).rvalue ** 2
    )

    checks.check("Largest deviation from prediction", errors.max(), "dB",
                 high=MAX_ERROR_DB, module=module)
    checks.check("Linearity (R²)", linearity, low=0.999, module=module)

    def plot_amplitude_transfer(data):
        import matplotlib.pyplot as plt
        import numpy as np

        amplitude = np.asarray(data["amplitude"])
        measured = np.asarray(data["measured_dbm"])
        predicted = np.asarray(data["predicted_dbm"])
        bad = np.abs(measured - predicted) >= data["max_error_db"]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(amplitude, predicted - data["max_error_db"],
                        predicted + data["max_error_db"], color="lightgray",
                        label="allowed")
        ax.plot(amplitude, measured, "k.-", label="measured")
        ax.plot(amplitude[bad], measured[bad], "ro", fillstyle="none")
        ax.plot(amplitude[~bad], measured[~bad], "go", fillstyle="none")
        ax.set_xscale("log")
        ax.set_xlabel("Amplitude (Normalized)")
        ax.set_ylabel("Magnitude (dBm)")
        ax.set_title(f"DAC output of module {data['module']} with increasing amplitude")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig

    report += [
        render_plot(plot_amplitude_transfer, dict(
            module=module,
            amplitude=amplitude_values,
            measured_dbm=magnitude_dbm_values,
            predicted_dbm=predicted_magnitude_dbm_values,
            max_error_db=MAX_ERROR_DB,
            linearity=linearity,
        ), f"DAC Amplitude Transfer Function, Module {module}"),
    ]


async def dac_scale_transfer(d, report, module):
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
    report.append(render_markdown(inspect.getdoc(dac_scale_transfer)))

    # set parameters for testing
    AMPLITUDE = 1
    NOMINAL_SCALE = 1
    SAMPLES = 100
    FREQUENCY = 150.5763e6
    NCO_FREQUENCY = 0
    ATTENUATION = 0
    CHANNEL = 1
    await d.set_adc_calibration_mode("MODE1", module)

    report.append(checks := CheckTable())

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
        module,
        NCO_FREQUENCY,
    )

    for scale in report.track(range(0, 8)):

        await d.set_dac_scale(scale, d.UNITS.DBM, module)
        raw_sampl = await d.py_get_samples(
            SAMPLES, channel=CHANNEL, module=module
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

    checks.check("Largest deviation from prediction", errors.max(), "dB", high=3,
                 module=module)

    # Linearity check on the scaling: apply linear regression and check least square error
    scale_values = np.array(scale_values).reshape(-1, 1)
    linearity = linregress(scale_values.ravel(), magnitude_dbm_values).rvalue ** 2
    checks.check("Linearity (R²)", linearity, low=0.99, module=module)

    def plot_scale_transfer(data):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data["scale_dbm"], data["measured_dbm"], "-o", label="measured")
        ax.plot(data["scale_dbm"], data["predicted_dbm"], "-o", label="predicted")
        ax.set_xlabel("Scale (dBm)")
        ax.set_ylabel("Magnitude (dBm)")
        ax.legend()
        ax.set_title(f"DAC output of module {data['module']} with increasing scale")
        ax.grid(True)
        fig.tight_layout()
        return fig

    report += [
        render_plot(plot_scale_transfer, dict(
            module=module,
            scale_dbm=scale_values.ravel(),
            measured_dbm=magnitude_dbm_values,
            predicted_dbm=predicted_magnitude_dbm_values,
            linearity=linearity,
        ), f"DAC Scale Transfer, Module {module}"),
    ]


async def dac_mixmodes(d, report, module):
    """
    ## DAC Nyquist Zone Checks

    Sweeps the NCO across Nyquist zones 1-4 in absolute frequency, leaving the
    firmware to choose each zone's DAC pulse shape (NRZ in zones 1 and 4, mix
    mode in zones 2 and 3) and the NCO signs.

    Validate:

    * A tone TONE_OFFSET above the NCO is found at every point (above
      FLOOR_DBM). An inverted spectrum puts it 2 × TONE_OFFSET away from the
      demodulator.
    * The level steps by less than MARGIN_DB across 5 GHz. Zones 2 and 3
      both use mix mode; NRZ in zone 3 would put the tone in its null at Fs.

    The loopback ADC samples at the DAC rate, so it sees every DAC image folded
    onto one tone: what it measures is the pulse shape filtered by the analog
    path, periodic in 5 GHz. Steps at 2.5 and 7.5 GHz, where the pulse shape
    changes between NRZ (low-pass) and mix mode (high-pass), are expected and
    depend on each module's analog path, so they are plotted but not checked.
    Mix-mode levels also depend on where within a DAC sample the ADC samples:
    near mid-sample, the pulse's two halves cancel and mix-mode zones read
    10-20 dB low while the NRZ zones stay flat to 2.5 GHz.
    """
    SCALE = 1
    SAMPLES = 100
    ATTENUATION = 0
    AMPLITUDE = 1
    CHANNEL = 1
    # below the AC-coupling corner, so NRZ has nothing to pass at 5 GHz
    TONE_OFFSET = 2e6
    # integer MHz, so the zone boundaries land exactly on the grid
    NCO_FREQUENCIES = np.arange(100, 7901, 100) * 1e6
    ZONE_BOUNDARIES = (2.5e9, 5e9, 7.5e9)
    FLOOR_DBM = -40
    MARGIN_DB = 6

    report.append(render_markdown(inspect.getdoc(dac_mixmodes)))
    report.append(rt := ResultTable("Parameter", "Value"))
    rt.row("TONE_OFFSET", f"{TONE_OFFSET/1e6:.0f} MHz")
    rt.row("FLOOR_DBM", f"{FLOOR_DBM} dBm")
    rt.row("MARGIN_DB", f"{MARGIN_DB} dB")

    checks = CheckTable()
    magnitude_dbm = np.zeros_like(NCO_FREQUENCIES)

    for i, nco in enumerate(report.track(NCO_FREQUENCIES)):
        await value_setter_helper(
            d, TONE_OFFSET, AMPLITUDE, SCALE, ATTENUATION, CHANNEL, module, nco
        )
        samples = await d.py_get_samples(SAMPLES, channel=CHANNEL, module=module)
        magnitude_dbm[i], _ = transfer_function_helper(
            samples, AMPLITUDE, SCALE, ATTENUATION
        )

    checks.check("Weakest tone", magnitude_dbm.min(), "dBm",
                 low=FLOOR_DBM, warn=10, module=module)
    i = np.flatnonzero(NCO_FREQUENCIES == 5e9)[0]
    checks.check("Step at 5.0 GHz", magnitude_dbm[i] - magnitude_dbm[i - 1], "dB",
                 low=-MARGIN_DB, warn=3, module=module)

    def plot_nyquist_zones(data):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        boundaries = data["zone_boundaries_mhz"]
        ax.axvspan(boundaries[0], boundaries[2], color="grey", alpha=0.15,
                   label="Mix mode")
        for boundary in boundaries:
            ax.axvline(boundary, color="grey", linestyle=":")
        ax.axhline(data["floor_dbm"], color="#b71c1c", linestyle="--", label="Tone floor")
        for name, level in data["levels_dbm"].items():
            ax.plot(data["frequency_mhz"], level, ".-", label=name)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude (dBm)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig

    report += [
        checks,
        render_plot(plot_nyquist_zones, dict(
            frequency_mhz=(NCO_FREQUENCIES + TONE_OFFSET) / 1e6,
            zone_boundaries_mhz=np.array(ZONE_BOUNDARIES) / 1e6,
            floor_dbm=FLOOR_DBM,
            levels_dbm={f"Module {module}": magnitude_dbm},
        ), f"DAC Nyquist zones, Module {module}"),
    ]


"""---------------------------------------------------------------------------------------------------------------"""
"""--------------------------------------------------ADC----------------------------------------------------------"""


async def adc_attenuation(d, report, module):
    """
    ## ADC Attenuation Test

    This test aims to determine if the attenuation factor set at the ADC
    accurately scales the incoming signal and is consistent with the
    predictions from the ADC transfer function. The internal mechanics of
    this test are mirror images of the scale test above. The test repeatedly
    calls set_adc_attenuation() with increasing factors, which changes the
    current input to the balun. A linear relationship is expected.
    """
    AMPLITUDE = 1
    SAMPLES = 100
    FREQUENCY = 80.234562e6
    NCO_FREQUENCY = 0
    SCALE = 1
    CHANNEL = 1
    ATTENUATION_0 = 0

    report.append(render_markdown(inspect.getdoc(adc_attenuation)))

    report.append(checks := CheckTable())

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
        module,
        NCO_FREQUENCY,
    )

    for attenuation in report.track(range(0, 15)):
        await d.set_adc_attenuator(attenuation, d.UNITS.DB, module=module)
        raw_sampl = await d.py_get_samples(
            SAMPLES, channel=CHANNEL, module=module
        )
        median_magnitude_dbm, predicted_magnitude_dbm = (
            transfer_function_helper(raw_sampl, AMPLITUDE, SCALE, attenuation)
        )
        attenuation_values.append(attenuation)
        magnitude_dbm_values.append(median_magnitude_dbm)
        predicted_magnitude_dbm_values.append(predicted_magnitude_dbm)

    attenuation_values = np.array(attenuation_values).reshape(-1, 1)
    linearity = (
        linregress(attenuation_values.ravel(), magnitude_dbm_values).rvalue ** 2
    )

    checks.check("Linearity (R²)", linearity, low=0.99, module=module)
    # a large deviation suggests the balun or SMA connections
    checks.check("Largest deviation from prediction", np.max(np.abs(
        np.array(magnitude_dbm_values) - np.array(predicted_magnitude_dbm_values))),
        "dB", high=1, module=module)

    module_result = {
        "module": module,
        "attenuation_values": attenuation_values.flatten().tolist(),
        "magnitude_dbm_values": magnitude_dbm_values,
        "predicted_magnitude_dbm_values": predicted_magnitude_dbm_values,
    }

    rt = ResultTable("Attenuation (dB)", "Measured (dBm)", "Predicted (dBm)")
    for a, measured, predicted in zip(
        module_result["attenuation_values"],
        magnitude_dbm_values,
        predicted_magnitude_dbm_values,
    ):
        if abs(measured - predicted) < 1:
            rt.pass_(a, measured, predicted)
        else:
            rt.fail(a, measured, predicted)

    def plot_attenuation(data):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data["attenuation_values"], data["magnitude_dbm_values"], "-o",
                label="measured")
        ax.plot(data["attenuation_values"], data["predicted_magnitude_dbm_values"],
                "-o", label="predicted")
        ax.set_xlabel("Attenuation (dB)")
        ax.set_ylabel("Magnitude (dBm)")
        ax.legend()
        ax.set_title(f"Signal at ADC of module {data['module']} with increasing attenuation")
        ax.grid(True)
        fig.tight_layout()
        return fig

    report += [
        rt,
        render_plot(plot_attenuation, module_result,
                    f"Magnitude vs attenuation at module {module}"),
    ]


async def wideband_noise(d, report, module):
    """
    ## Wideband Noise Checks

    - **Objective**: Plot wideband noise with no stimulus from the DAC.
    - **Expected Outcome**: A plot showing I/Q PSDs
    - **Limitations:** This test succeeds unconditionally.
    """
    SAMPLING_FREQUENCY = 5e9  # 5 GHz
    NUM_SAMPLES = 4000  # Total number of samples

    report.append(render_markdown(inspect.getdoc(wideband_noise)))
    report.append(rt := ResultTable("Parameter", "Value"))
    rt.row("NUM_SAMPLES", NUM_SAMPLES)

    await d.clear_channels()
    time.sleep(1)
    samples = await d.get_fast_samples(NUM_SAMPLES, module=module)

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

    def plot_wideband_noise(data):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data["frequency_hz"], data["psd_i_dbm_hz"], label="I")
        ax.plot(data["frequency_hz"], data["psd_q_dbm_hz"], label="Q")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dBm/Hz)")
        ax.legend()
        ax.set_title(f"Wideband FFT at module {data['module']}")
        ax.grid(True)
        fig.tight_layout()
        return fig

    report += [
        render_plot(plot_wideband_noise, dict(
            module=module, frequency_hz=psdfreq_i, psd_i_dbm_hz=psd_i,
            psd_q_dbm_hz=psd_q,
        ), f"Wideband Noise, Module {module}"),
        {"data": dict(module=module, i=list(samples.i), q=list(samples.q))},
    ]
