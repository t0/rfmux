import pickle as pkl
import numpy as np
import os
from ...core.hardware_map import macro
from ...core.schema import CRS


# Module-level variable to store the cable delay in nanoseconds
CABLE_DELAY_NS = 0.0

@macro(CRS, register=True)
async def set_cable_delay(crs, meters, velocity_factor=0.66):
    """
    Set the cable delay (in nanoseconds) based on the physical length (in meters).
    
    By default, we use a typical velocity factor of 0.66 (66%)
    for many coax cables. You can change 'velocity_factor' as needed.
    
    speed_of_light_m_per_s = 3.0e8
    time_in_seconds = length / (speed_of_light * velocity_factor)
    time_in_ns = time_in_seconds * 1e9
    """
    global CABLE_DELAY_NS  # Tell Python we're modifying the module-level variable
    
    speed_of_light = 3.0e8  # m/s (approximately)
    
    # Compute one-way delay
    time_in_seconds = meters / (speed_of_light * velocity_factor)
    
    CABLE_DELAY_NS = time_in_seconds

def get_cable_delay():
    """
    Retrieve the current cable delay (in nanoseconds).
    """
    return CABLE_DELAY_NS + 2.3325e-07


def load_pickle_from_repo():
    rfmux_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pkl_path = os.path.join(rfmux_dir, "core", "utils", "phase_calibrations_r1.5.pkl")

    with open(pkl_path, "rb") as f:
        data = pkl.load(f)
    return data


final_cals = load_pickle_from_repo()
bin_centres = np.array([x * 625e6/512 for x in range(-256, 256)])

@macro(CRS, register=True)
async def py_set_frequency(crs : CRS, freq, channel, module):

    await crs.set_frequency(freq, channel=channel, module=module, disable_latency_correction=True)
    
    bin_index = np.argmin(np.abs(bin_centres - freq))-256
    
    latency = final_cals[channel-1][bin_index]['latency'] + get_cable_delay()
    offset = final_cals[channel-1][bin_index]['offset']

    phase_from_latency = np.pi * (1. - (2*latency) * (freq % (1./latency)))
    delta_phase = phase_from_latency + offset
    delta_phase = (delta_phase + np.pi) % (2*np.pi) - np.pi

    # This is set in the ADC, not DAC to avoid crest factor issues at the analog-digital interfaces.
    # If it was set at the DAC, then all sinusoids generated would be in-phase!
    await crs.set_phase(delta_phase, crs.UNITS.RADIANS, crs.TARGET.ADC, channel=channel, module=module)