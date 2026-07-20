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


# used in the test_loopback_noise
def psd_helper(data_i_raw, data_q_raw):
    COMB_SAMPLING_FREQUENCY = 625e6
    volts_peak = (
        (np.sqrt(2)) * np.sqrt(50 * (10 ** (-1.75 / 10)) / 1000) / 1880796.4604246316
    )
    # Convert I and Q to peak volts
    tod_i = data_i_raw * volts_peak
    tod_q = data_q_raw * volts_peak

    # Define the sampling frequency for the ADC
    fir = COMB_SAMPLING_FREQUENCY / 256 / 64 / 2**6
    # Apply Welch's method to get the signal PSD
    psdfreq, psd_i = signal.welch(
        tod_i / np.sqrt(4 * 50), fir, nperseg=len(tod_i // 4)
    )  # tod_i / np.sqrt(4 * 50)
    _, psd_q = signal.welch(
        tod_q / np.sqrt(4 * 50), fir, nperseg=len(tod_q // 4)
    )  # tod_q / np.sqrt(4 * 50)

    # Convert from watts to dBm
    psd_i = 10 * np.log10(psd_i * 1000)
    psd_q = 10 * np.log10(psd_q * 1000)
    return psd_i, psd_q, psdfreq


@pytest.mark.asyncio
async def test_loopback_noise(d, request, shelf, check):
    """
    ## Network Analysis - Ambient Noise

    Using `take_specanal`, perform a wideband network analysis with no bias
    tones.  This is a test of the noise environment.
    """

    # Documentation contained in the DocString
    render = render_markdown(test_loopback_noise.__doc__)

    FMAX = 2500e6  # Nyquist
    MASK_FMIN = 50e6
    MASK_FMAX = 2450e6
    MASK_AMP = 1e1

    # Plot setup
    cycler = plt.rcParams["axes.prop_cycle"]
    plt.figure(figsize=(10, 6))

    rt = ResultTable("Module", "Mask Exceeded", "Max Masked Amplitude")

    for m, color in zip(d.modules, cycler.by_key()["color"]):
        await d.clear_channels()
        await d.set_nco_frequency(0, module=m.module)
        (fs, mags) = await d.take_specanal(
            nsamps=100, fmin=0, fmax=FMAX, module=m.module, scaling="ps"
        )

        # Convert V^2/Hz to dBm
        mags = 10 * np.log10(np.abs(mags)) + 30

        # Plot magnitude
        plt.semilogy(fs, np.abs(mags), "-,", color=color, label=f"Module {m.module}")

        # Check if we exceeded the mask anywhere
        errors = (np.abs(mags) > MASK_AMP) * (fs > MASK_FMIN) * (fs < MASK_FMAX)
        plt.semilogy(
            fs[errors], np.abs(mags[errors]), "o", color=color, fillstyle="none"
        )

        # Find and mark maxima within mask
        fs_masked = fs[(fs > MASK_FMIN) * (fs < MASK_FMAX)]
        mags_masked = mags[(fs > MASK_FMIN) * (fs < MASK_FMAX)]
        nmax = np.argmax(np.abs(mags_masked))
        fmax = fs_masked[nmax]
        amax = np.abs(mags_masked[nmax])
        plt.semilogy(fs_masked[nmax], amax, "o", color=color)

        if any(errors):
            check.fail(f"Mask check failed for module {m.module}")
            rt.fail(
                f"{m.module}",
                f"{100.*sum(errors)/len(errors):.1f} %",
                f"{amax:.1f} @ {fmax/1e6:.3f} MHz",
            )
        else:
            rt.pass_(
                f"{m.module}",
                f"{100.*sum(errors)/len(errors):.1f} %",
                f"{amax:.1f} @ {fmax/1e6:.3f} MHz",
            )

    # Draw the mask we're checking against
    plt.plot(
        [MASK_FMIN, MASK_FMIN, MASK_FMAX, MASK_FMAX],
        [10 * MASK_AMP, MASK_AMP, MASK_AMP, 10 * MASK_AMP],
        "k:",
        label="Mask",
    )

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (V^2/Hz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{request.session.results_dir}/{request.node.name}.png")
    plt.close()

    with shelf as x:
        x["sections"][request.node.name] = [
            render,
            rt,
            htpy.figure[
                htpy.img(src=f"{request.node.name}.png"),
                htpy.figcaption[f"Noise environment"],
            ],
        ]


"""
        # 2. Test the PSD of a large signal, establishing an expectation for 1/f slope and peak noise at low frequency
        value_setter_helper(
            d,
            FREQUENCY_1_F,
            AMPLITUDE_1_F,
            SCALE_1_F,
            ATTENUATION_1_F,
            CHANNEL_1_F,
            mod,
            NCO_FREQUENCY_1_F,
        )
        raw_samples_psd = await d.py_get_samples(
            SAMPLES, channel=CHANNEL_1_F, module=mod
        )
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

        # check if any spikes are present
        if (
            0.95 * (-163.4) < mean_point_white_i
            or 0.95 * (-163.4) < mean_point_white_q
        ):
            floor_errors.append(
                "Noise floor of module {mod} is higher than expected"
            )
        if max_point_1_f_i > -79 or max_point_1_f_q > -79:
            loud_errors.append(
                "Noise levels for frequencies dominated by 1/f are higher than -79dBm"
            )

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
        slope, intercept, r_value, p_value, std_err = linregress(
            log_frequencies_fit, log_psd_values_fit
        )

        # Calculate the fitted line in log space over the full range for plotting
        log_frequencies_plot = np.log10(combined_freqs)
        fitted_log_psd_values_plot = intercept + slope * log_frequencies_plot

        # Convert fitted values back to linear scale for plotting
        fitted_psd_values_W_Hz_plot = 10**fitted_log_psd_values_plot
        fitted_psd_values_dBm_Hz_plot = (
            10 * np.log10(fitted_psd_values_W_Hz_plot) + 30
        )

        if slope < 1.1 * EXPECTED_SLOPE or slope > 0.9 * EXPECTED_SLOPE:
            loud_errors.append("1/f noise deviates from the expected value of -1")

        module_result = {
            "module": mod,
            "psd_i_white": psd_i_white.tolist(),
            "psd_q_white": psd_q_white.tolist(),
            "psd_frequencies_white": psdfreq_white.tolist(),
            "max_point_white_i": max_point_white_i.tolist(),
            "max_point_white_q": max_point_white_q.tolist(),
            "psd_i_1_f": psd_i_1_f.tolist(),
            "psd_q_1_f": psd_q_1_f.tolist(),
            "psd_frequencies_1_f": psdfreq_1_f.tolist(),
            "combined_psd_1_f": combined_psd_1_f.tolist(),
            "combined_freqs": combined_freqs.tolist(),  # This should be the original combined frequency list
            "max_point_1_f_i": max_point_1_f_i.tolist(),
            "max_point_1_f_q": max_point_1_f_q.tolist(),
            "fitted_psd_values_dBm_Hz": fitted_psd_values_dBm_Hz_plot.tolist(),
            "slope": slope,
            "white_status": "Fail" if floor_errors else "Pass",
            "1_f_status": "Fail" if loud_errors else "Pass",
            "plot_path_white": "plot_path_white",
            "plot_path_1_f": "plot_path_1_f",
        }

        graph_name_white = (
            TITLE + "_white_" + "_module_" + f'{module_result["module"]}' + ".png"
        )
        plot_path_white = os.path.join(results_dir, "plots", graph_name_white)
        module_result["plot_path_white"] = plot_path_white

        graph_name_1_f = (
            TITLE + "_1_f" + "_module_" + f'{module_result["module"]}' + ".png"
        )
        plot_path_1_f = os.path.join(results_dir, "plots", graph_name_1_f)
        module_result["plot_path_1_f"] = plot_path_1_f

        # plot the white noise
        plt.figure(figsize=(10, 6))
        plt.plot(
            module_result["psd_frequencies_white"],
            module_result["psd_i_white"],
            label="I",
        )
        plt.plot(
            module_result["psd_frequencies_white"],
            module_result["psd_q_white"],
            label="Q",
        )
        plt.scatter(
            [
                module_result["psd_frequencies_white"][
                    np.argmax(module_result["psd_i_white"])
                ]
            ],
            [module_result["max_point_white_i"]],
            color="red",
            label=f'Max I spike: {module_result["max_point_white_i"]:.2f} dBm',
        )
        plt.scatter(
            [
                module_result["psd_frequencies_white"][
                    np.argmax(module_result["psd_q_white"])
                ]
            ],
            [module_result["max_point_white_q"]],
            color="green",
            label=f'Max Q spike: {module_result["max_point_white_q"]:.2f} dBm',
        )
        plt.xlabel("Frequency (Hz from carrier)")
        plt.ylabel("Magnitude (dBm)")
        plt.xscale("log")
        plt.legend()
        plt.title(f"Noise floor in loopback at 277.23MHz at module {mod}")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path_white), exist_ok=True)
        plt.savefig(plot_path_white)
        plt.close()

        # 1/f noise PSD
        plt.figure(figsize=(10, 6))
        plt.plot(
            module_result["combined_freqs"],
            module_result["fitted_psd_values_dBm_Hz"],
            "r",
            label=f'fit to 1/f spectrum: n = {module_result["slope"]:.2f}',
        )
        plt.scatter(
            module_result["combined_freqs"],
            module_result["combined_psd_1_f"],
            label="1/f spectrum data",
        )
        plt.scatter(
            [
                module_result["psd_frequencies_1_f"][
                    np.argmax(module_result["psd_i_1_f"])
                ]
            ],
            [module_result["max_point_1_f_i"]],
            color="red",
            label=f'Max I spike: {module_result["max_point_1_f_i"]:.2f} dBm',
        )
        plt.scatter(
            [
                module_result["psd_frequencies_1_f"][
                    np.argmax(module_result["psd_q_1_f"])
                ]
            ],
            [module_result["max_point_1_f_q"]],
            color="green",
            label=f'Max Q spike: {module_result["max_point_1_f_q"]:.2f} dBm',
        )
        plt.xscale("log")
        plt.title("PSD of 1/f Noise")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dBm/Hz)")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path_1_f), exist_ok=True)
        plt.savefig(plot_path_1_f)
        plt.close()

        if mod not in summary_results["summary"]:
            summary_results["summary"][mod] = {}
        summary_results["summary"][mod]["Noise floor"] = module_result[
            "white_status"
        ]
        summary_results["summary"][mod]["1/f Noise"] = module_result["1_f_status"]

        test_result["modules"].append(module_result)

    if m.module >= 5:
        d.set_analog_bank(high=True)

    d.set_analog_bank(high=False)

    with shelf as x:
        x["sections"][request.node.name] = [
                render,
        ]
        """
