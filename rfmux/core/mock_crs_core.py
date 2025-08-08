"""
Mock CRS Device - Core MockCRS class definition, state, and methods.
"""

import asyncio
import atexit
import inspect
import numpy as np
import weakref

from collections import defaultdict
from datetime import datetime
from enum import Enum

# Import schema classes
from .schema import CRS as BaseCRS
# Potentially other schema items if MockCRS directly uses them, e.g. ReadoutModule
# from .schema import ReadoutModule, Crate # etc.

# Import helper classes from their new locations
from .mock_resonator_model import MockResonatorModel
from .mock_udp_streamer import MockUDPManager # Manages the UDP streamer thread
from ..streamer import SS_PER_SECOND

# Import enhanced scaling constants
from . import mock_constants as const

# Module-level cleanup registry for MockCRS instances (to avoid JSON serialization issues)
_mock_crs_instances = weakref.WeakSet()
_cleanup_registered = False

def _cleanup_all_mock_crs_instances():
    """Emergency cleanup for all MockCRS instances at program exit"""
    for instance in list(_mock_crs_instances):
        try:
            # Stop UDP streaming synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(instance.stop_udp_streaming())
            loop.close()
            print(f"[CLEANUP] Emergency cleanup completed for MockCRS instance {id(instance)}")
        except Exception as e:
            print(f"[CLEANUP] Error during emergency cleanup: {e}")

# Enums (as defined in the original mock.py)
class ClockSource(str, Enum):
    VCXO = "VCXO"
    SMA = "SMA"
    BACKPLANE = "Backplane"

class TimestampPort(str, Enum): # This was imported from streamer, defining here for self-containment
    BACKPLANE = "BACKPLANE"
    TEST = "TEST"
    SMA = "SMA"
    # GND = "GND" # Original streamer.py had GND, mock.py used TEST as default

class FrequencyUnit(str, Enum):
    HZ = "hz"

class AmplitudeUnit(str, Enum):
    NORMALIZED = "normalized"
    RAW = "raw"

class PhaseUnit(str, Enum):
    DEGREES = "degrees"
    RADIANS = "radians"

class Target(str, Enum):
    CARRIER = "carrier"
    NULLER = "nuller"
    DEMOD = "demod"
    ADC = "adc"
    DAC = "dac"

class Units(str, Enum):
    HZ = "hz"
    RAW = "raw"
    VOLTS = "volts"
    AMPS = "amps"
    WATTS = "watts"
    DEGREES = "degrees"
    RADIANS = "radians"
    OHMS = "ohms"
    NORMALIZED = "normalized"
    DB = "db"
    DBM = "dbm"

class ADCCalibrationMode(str, Enum):
    AUTO = "AUTO"
    MODE1 = "MODE1"
    MODE2 = "MODE2"

class ADCCalibrationBlock(str, Enum):
    OCB1 = "OCB1"
    OCB2 = "OCB2"
    GCB = "GCB"
    TSCB = "TSCB"


class ClientMockCRS(BaseCRS):
    """
    Mock MKIDS device emulation class.

    This is the client-side object, and is only a thin wrapper with ORM
    properties.
    """

    # Override polymorphic identity
    __mapper_args__ = {"polymorphic_identity": "MockCRS"}

    def __init__(self, serial, slot=None, crate=None, **kwargs):
        super().__init__(serial=serial, slot=slot, crate=crate, **kwargs)

        # Register this instance for cleanup using module-level registry
        global _mock_crs_instances, _cleanup_registered
        _mock_crs_instances.add(self)

        # Register global cleanup handler (only once)
        if not _cleanup_registered:
            atexit.register(_cleanup_all_mock_crs_instances)
            _cleanup_registered = True


class ServerMockCRS:
    """Mock MKIDS device emulation class.

    This is the "server-side" object, and contains all the stuff you actually
    expect. You interact with it remotely, as a TuberObject.
    """

    _num_tuber_calls = 0 # Class attribute for server to increment

    # Expose Enums as class attributes
    Units = Units
    Target = Target
    ClockSource = ClockSource
    TimestampPort = TimestampPort
    FrequencyUnit = FrequencyUnit
    AmplitudeUnit = AmplitudeUnit
    PhaseUnit = PhaseUnit
    ADCCalibrationMode = ADCCalibrationMode
    ADCCalibrationBlock = ADCCalibrationBlock
    
    # Create properties that return simple dicts for Tuber serialization
    @property
    def TIMESTAMP_PORT(self):
        """Return a dict with TIMESTAMP_PORT enum values as strings"""
        return {
            'BACKPLANE': TimestampPort.BACKPLANE.value,
            'TEST': TimestampPort.TEST.value,
            'SMA': TimestampPort.SMA.value
        }
    
    @property
    def CLOCKSOURCE(self):
        """Return a dict with CLOCKSOURCE enum values as strings"""
        return {
            'VCXO': ClockSource.VCXO.value,
            'SMA': ClockSource.SMA.value,
            'BACKPLANE': ClockSource.BACKPLANE.value
        }
    
    @property
    def UNITS(self):
        """Return a dict with UNITS enum values as strings"""
        return {
            'HZ': Units.HZ.value,
            'RAW': Units.RAW.value,
            'VOLTS': Units.VOLTS.value,
            'AMPS': Units.AMPS.value,
            'WATTS': Units.WATTS.value,
            'DEGREES': Units.DEGREES.value,
            'RADIANS': Units.RADIANS.value,
            'OHMS': Units.OHMS.value,
            'NORMALIZED': Units.NORMALIZED.value,
            'DB': Units.DB.value,
            'DBM': Units.DBM.value
        }
    
    @property
    def TARGET(self):
        """Return a dict with TARGET enum values as strings"""
        return {
            'CARRIER': Target.CARRIER.value,
            'NULLER': Target.NULLER.value,
            'DEMOD': Target.DEMOD.value,
            'ADC': Target.ADC.value,
            'DAC': Target.DAC.value
        }

    # Static properties for units, targets, sensors, etc. (as dictionaries)
    _UNITS_DICT = { # Renamed to avoid conflict with Enum
        "HZ": "hz", "RAW": "raw", "VOLTS": "volts", "AMPS": "amps",
        "WATTS": "watts", "DEGREES": "degrees", "RADIANS": "radians",
        "OHMS": "ohms", "NORMALIZED": "normalized", "DB": "db", "DBM": "dbm"
    }
    _TARGET_DICT = {
        "CARRIER": "carrier", "NULLER": "nuller", "DEMOD": "demod",
        "DAC" : "dac", "ADC" : "adc"
    }
    _CLOCK_SOURCE_DICT = {
        "XTAL": "XTAL", "SMA": "SMA", "BACKPLANE": "Backplane"
    }
    _TIMESTAMP_PORT_DICT = {
        "BACKPLANE": "BACKPLANE", "SMA": "SMA", "TEST": "TEST"
    }
    _TEMPERATURE_SENSOR_DICT = {
        "MOTHERBOARD_POWER": "MOTHERBOARD_TEMPERATURE_POWER",
        "MOTHERBOARD_ARM": "MOTHERBOARD_TEMPERATURE_ARM",
        "MOTHERBOARD_FPGA": "MOTHERBOARD_TEMPERATURE_FPGA",
        "MOTHERBOARD_PHY": "MOTHERBOARD_TEMPERATURE_PHY",
    }
    _RAIL_DICT = {
        "MB_VCC3V3": "MOTHERBOARD_RAIL_VCC3V3", "MB_VCC12V0": "MOTHERBOARD_RAIL_VCC12V0",
        "MB_VCC5V5": "MOTHERBOARD_RAIL_VCC5V5", "MB_VCC1V0_GTX": "MOTHERBOARD_RAIL_VCC1V0_GTX",
        "MB_VCC1V0": "MOTHERBOARD_RAIL_VCC1V0", "MB_VCC1V2": "MOTHERBOARD_RAIL_VCC1V2",
        "MB_VCC1V5": "MOTHERBOARD_RAIL_VCC1V5", "MB_VCC1V8": "MOTHERBOARD_RAIL_VCC1V8",
        "MB_VADJ": "MOTHERBOARD_RAIL_VADJ",
    }
    _ADCCALIBRATIONMODE_DICT = {
        "AUTO": "AUTO", "MODE1": "MODE1", "MODE2": "MODE2"
    }
    _ADCCALIBRATIONBLOCK_DICT = {
        "OCB1": "OCB1", "OCB2": "OCB2", "GCB": "GCB", "TSCB": "TSCB"
    }

    def __init__(self, serial, slot=None, crate=None, **kwargs):
        self._serial = serial
        self._slot = slot
        self._crate = crate

        self._clock_source = None
        self._clock_priority = []
        self._timestamp_port = None
        self._timestamp = dict(y=2025, d=1, h=0, m=0, s=0, ss=0, recent=False)
        self._max_samples_per_channel = 100000
        self._max_samples_per_module = 30000

        self._frequencies = defaultdict(lambda: 0.0) # (module, channel) -> frequency
        self._amplitudes = defaultdict(lambda: 0.0) # (module, channel) -> amplitude
        self._phases = defaultdict(lambda: 0.0) # (module, channel) -> phase
        self._tuning_results = {} # Placeholder

        self._active_modules = [1, 2, 3, 4]  # FIXME: add set_analog_bank() support
        self._fir_stage = 6     # Default FIR stage
        self._streamed_modules = [1, 2, 3, 4]
        self._short_packets = False

        self._temperature_sensors = {
            "MOTHERBOARD_TEMPERATURE_POWER": 30.0, "MOTHERBOARD_TEMPERATURE_ARM": 35.0,
            "MOTHERBOARD_TEMPERATURE_FPGA": 40.0, "MOTHERBOARD_TEMPERATURE_PHY": 45.0,
        }
        self._rails = {
            "MOTHERBOARD_RAIL_VCC3V3": {"voltage": 3.3, "current": 1.0},
            "MOTHERBOARD_RAIL_VCC12V0": {"voltage": 12.0, "current": 0.5},
            "MOTHERBOARD_RAIL_VCC5V5": {"voltage": 5.5, "current": 0.8},
            "MOTHERBOARD_RAIL_VCC1V0_GTX": {"voltage": 1.0, "current": 0.3},
            "MOTHERBOARD_RAIL_VCC1V0": {"voltage": 1.0, "current": 0.4},
            "MOTHERBOARD_RAIL_VCC1V2": {"voltage": 1.2, "current": 0.6},
            "MOTHERBOARD_RAIL_VCC1V5": {"voltage": 1.5, "current": 0.7},
            "MOTHERBOARD_RAIL_VCC1V8": {"voltage": 1.8, "current": 0.9},
            "MOTHERBOARD_RAIL_VADJ": {"voltage": 2.5, "current": 0.2},
        }

        self._rfdc_initialized = False
        self._nco_frequencies = {} # module -> nco_frequency
        self._adc_attenuators = {m: {"amplitude": 0.0, "units": self.Units.DB} for m in range(1, 5)}
        self._dac_scales = {m: {"amplitude": 1.0, "units": self.Units.DBM} for m in range(1, 5)}
        self._adc_autocal = {m: True for m in range(1, 5)}
        self._adc_calibration_mode = {m: self.ADCCalibrationMode.AUTO for m in range(1, 5)}
        self._adc_calibration_coefficients = {} # module -> {block: [coeffs]}
        self._nyquist_zones = {} # module -> {target: zone}
        self._hmc7044_registers = {} # address -> value
        self._cable_lengths = {}  # module -> cable length in meters

        # Store physics configuration (can be updated via generate_resonators)
        self._physics_config = {}  # Will store active physics configuration
        
        # Instantiate helpers
        self._resonator_model = MockResonatorModel(self) # Pass self for state access
        self._udp_manager = MockUDPManager(self)         # Pass self for state access

        # Initialize resonator parameters via the model
        self._resonator_model.generate_lc_resonances() # Default LC model init

    async def generate_resonators(self, config=None):
        """Generate/regenerate resonators with current or provided parameters.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters to use. If not provided, uses defaults.
            Keys can include:
            - num_resonances
            - freq_start
            - freq_end
            - kinetic_inductance_fraction
            - kinetic_inductance_variation
            - q_min
            - q_max
            - q_variation
            - coupling_min
            - coupling_max
            etc.
        """
        try:
            # Initialize resonator model if needed
            if not hasattr(self, '_resonator_model') or self._resonator_model is None:
                self._resonator_model = MockResonatorModel(self)
                
            # Ensure lists are initialized
            if not hasattr(self._resonator_model, 'lc_resonances'):
                self._resonator_model.lc_resonances = []
            if self._resonator_model.lc_resonances is None:
                self._resonator_model.lc_resonances = []
                
            if not hasattr(self._resonator_model, 'kinetic_inductance_fractions'):
                self._resonator_model.kinetic_inductance_fractions = []
            if self._resonator_model.kinetic_inductance_fractions is None:
                self._resonator_model.kinetic_inductance_fractions = []
                
            # Clear existing resonators
            self._resonator_model.lc_resonances = []
            self._resonator_model.kinetic_inductance_fractions = []
            
            # If config provided, store it permanently in the instance
            if config:
                # Update our physics configuration (permanent storage)
                self._physics_config = config.copy()
                
                # Generate with new parameters - the model will use self._physics_config
                self._resonator_model.generate_lc_resonances()
            else:
                # Generate with defaults
                self._resonator_model.generate_lc_resonances()
            
            # Validate result
            if self._resonator_model.lc_resonances is None:
                self._resonator_model.lc_resonances = []
                
            return len(self._resonator_model.lc_resonances)
            
        except Exception as e:
            print(f"Error in generate_resonators: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to send error back to client

    def _validate_enum_member(self, value, enum_class, name):
        """Validate that value is a member of enum_class, converting strings if necessary."""
        if isinstance(value, str):
            try:
                return enum_class(value) # Return the enum member
            except ValueError:
                valid_values = [e.value for e in enum_class]
                raise ValueError(f"Invalid {name} '{value}'. Must be one of {valid_values}")
        elif not isinstance(value, enum_class):
            raise TypeError(f"{name.capitalize()} must be a {enum_class.__name__} enum member or a valid string.")
        return value # Already an enum member

    # --- Core CRS Methods ---
    async def set_frequency(self, frequency, channel=None, module=None):
        assert isinstance(frequency, (int, float)), "Frequency must be a number"
        min_freq_hz = -313.5e6
        max_freq_hz = 313.5e6
        if not (min_freq_hz <= frequency <= max_freq_hz):
            raise ValueError(f"Frequency must be between -313.5 MHz and +313.5 MHz.")
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        self._frequencies[(module, channel)] = frequency

    async def get_frequency(self, channel=None, module=None):
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        return self._frequencies.get((module, channel))

    async def set_amplitude(self, amplitude, channel=None, module=None):
        assert isinstance(amplitude, (int, float)), "Amplitude must be a number"
        if not (-1.0 <= amplitude <= 1.0): # Normalized
            raise ValueError("Amplitude must be between -1.0 and +1.0.")
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        self._amplitudes[(module, channel)] = amplitude

    async def get_amplitude(self, channel=None, module=None):
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        return self._amplitudes.get((module, channel))

    async def set_phase(self, phase, units='DEGREES', target=None, channel=None, module=None):
        assert isinstance(phase, (int, float)), "Phase must be a number"
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        
        # Validate units if provided as string
        if isinstance(units, str):
            units = units.upper()
            if units not in ['DEGREES', 'RADIANS']:
                raise ValueError(f"Invalid phase units '{units}'. Must be 'DEGREES' or 'RADIANS'")
        
        # Convert to degrees for internal storage
        if units == 'RADIANS' or (hasattr(units, 'value') and units.value == 'radians'):
            phase_degrees = phase * 180.0 / np.pi
        else:
            phase_degrees = phase
            
        self._phases[(module, channel)] = phase_degrees

    async def get_phase(self, units='DEGREES', target=None, channel=None, module=None):
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        
        # Get phase in degrees (internal storage)
        phase_degrees = self._phases.get((module, channel))
        if phase_degrees is None:
            return None
            
        # Validate units if provided as string
        if isinstance(units, str):
            units = units.upper()
            if units not in ['DEGREES', 'RADIANS']:
                raise ValueError(f"Invalid phase units '{units}'. Must be 'DEGREES' or 'RADIANS'")
        
        # Convert based on requested units
        if units == 'RADIANS' or (hasattr(units, 'value') and units.value == 'radians'):
            return phase_degrees * np.pi / 180.0
        else:
            return phase_degrees

    async def set_timestamp_port(self, port):
        port = self._validate_enum_member(port, TimestampPort, "timestamp port")
        self._timestamp_port = port

    async def get_timestamp_port(self):
        return self._timestamp_port

    async def get_analog_bank(self):
        return False

    async def set_analog_bank(self, high=False):
        assert high == False, "High analog banking is not yet mocked up!"

    async def set_clock_source(self, clock_source=None):
        if clock_source is not None:
            clock_source = self._validate_enum_member(clock_source, ClockSource, "clock source")
        self._clock_source = clock_source

    async def get_clock_source(self):
        return self._clock_source
    
    async def set_clock_priority(self, clock_priority=None):
        if clock_priority is not None:
            clock_priority = [self._validate_enum_member(cs, ClockSource, "clock source") for cs in clock_priority]
        self._clock_priority = clock_priority

    async def get_clock_priority(self):
        return self._clock_priority

    async def get_motherboard_temperature(self, sensor):
        # sensor should be a string key from _TEMPERATURE_SENSOR_DICT
        if sensor not in self._TEMPERATURE_SENSOR_DICT.values(): # Check against values
             raise ValueError(f"Invalid sensor '{sensor}'. Must be one of {list(self._TEMPERATURE_SENSOR_DICT.values())}")
        return self._temperature_sensors.get(sensor)


    async def get_motherboard_voltage(self, rail):
        if rail not in self._RAIL_DICT.values():
            raise ValueError(f"Invalid rail '{rail}'. Must be one of {list(self._RAIL_DICT.values())}")
        return self._rails.get(rail, {}).get("voltage")

    async def get_motherboard_current(self, rail):
        if rail not in self._RAIL_DICT.values():
            raise ValueError(f"Invalid rail '{rail}'. Must be one of {list(self._RAIL_DICT.values())}")
        return self._rails.get(rail, {}).get("current")

    async def set_decimation(self, stage: int=6,
                       short_packets: bool=False,
                       modules: list[int] | None=None):

        assert isinstance(stage, int) and 0 <= stage <= 6, \
                "FIR stage must be an integer between 0 and 6 (inclusive)"

        if modules is None:
            modules = list(self._active_modules)

        if not isinstance(modules, list) or set(modules) > set(self._active_modules):
            raise ValueError("Invalid 'modules' argument to set_decimation!")

        self._fir_stage = stage
        self._short_packets = short_packets
        self._streamed_modules = modules

    async def get_decimation(self):
        return None if len(self._streamed_modules)==0 else self._fir_stage

    async def _thread_lock_acquire(self): # Keep for server compatibility
        await asyncio.sleep(0.001) # Minimal sleep

    def _thread_lock_release(self): # Keep for server compatibility
        pass

    async def get_timestamp(self):
        # Simulate time passing or return a fixed test timestamp
        now = datetime.now()
        self._timestamp = dict(y=now.year % 100,
                              d=now.timetuple().tm_yday,
                              h=now.hour,
                              m=now.minute,
                              s=now.second,
                              ss=int(now.microsecond / 1e6 * SS_PER_SECOND),
                              c=0,
                              sbs=0,
                              source=TimestampPort.TEST,
                              recent=True)
        return self._timestamp

    async def get_pfb_samples(self, num_samples, units=Units.NORMALIZED, channel=None, module=1):
        assert isinstance(num_samples, int) and num_samples > 0
        assert channel is not None and isinstance(channel, int)
        assert 0 <= channel < 1024 # Assuming 0-indexed for internal consistency
        assert module in [1, 2, 3, 4]
        units = self._validate_enum_member(units, Units, "units")
        assert units in [Units.NORMALIZED, Units.RAW]
        await asyncio.sleep(0.001) # Simulate delay

        i_vals, q_vals = [], []
        if units == Units.NORMALIZED:
            i_vals = np.random.uniform(-1.0, 1.0, num_samples).tolist()
            q_vals = np.random.uniform(-1.0, 1.0, num_samples).tolist()
        else: # RAW
            max_val = 2**23 - 1
            i_vals = np.random.randint(-max_val-1, max_val+1, num_samples).tolist()
            q_vals = np.random.randint(-max_val-1, max_val+1, num_samples).tolist()
        return list(zip(i_vals, q_vals))

    async def get_fast_samples(self, num_samples, units=Units.NORMALIZED, module=1):
        assert isinstance(num_samples, int) and num_samples > 0
        assert module in [1, 2, 3, 4]
        units = self._validate_enum_member(units, Units, "units")
        assert units in [Units.NORMALIZED, Units.RAW]
        await asyncio.sleep(0.001)

        i_vals, q_vals = [], []
        if units == Units.NORMALIZED:
            i_vals = np.random.uniform(-1.0, 1.0, num_samples).tolist()
            q_vals = np.random.uniform(-1.0, 1.0, num_samples).tolist()
        else: # RAW
            max_val = 2**15 - 1
            i_vals = np.random.randint(-max_val-1, max_val+1, num_samples).tolist()
            q_vals = np.random.randint(-max_val-1, max_val+1, num_samples).tolist()
        return {"i": i_vals, "q": q_vals}

    async def set_nco_frequency(self, frequency, module=None):
        """Set NCO frequency - ASYNC method as it's called directly with await."""
        assert isinstance(frequency, (int, float))
        assert module is not None and isinstance(module, int)
        
        # Simulate hardware communication delay
        await asyncio.sleep(0.001)
        
        # Set NCO frequency for the module
        self._nco_frequencies[module] = frequency

    async def get_nco_frequency(self, module=None, target=None): # target unused but for API compat
        assert module is not None and isinstance(module, int)
        return self._nco_frequencies.get(module)

    async def set_adc_attenuator(self, attenuation, module=None):
        assert isinstance(attenuation, int) and 0 <= attenuation <= 27
        assert module is not None and isinstance(module, int)
        self._adc_attenuators[module] = {"amplitude": float(attenuation), "units": self.Units.DB}


    async def get_adc_attenuator(self, module=None):
        assert module is not None and isinstance(module, int)
        return self._adc_attenuators.get(module, {}).get("amplitude")


    async def set_dac_scale(self, amplitude, units='DBM', module=None):
        assert isinstance(amplitude, (int, float))
        # units = self._validate_enum_member(units, Units, "units") # Original didn't validate strictly
        assert module is not None and isinstance(module, int)
        self._dac_scales[module] = {"amplitude": amplitude, "units": units}


    async def get_dac_scale(self, units='DBM', module=None):
        # units = self._validate_enum_member(units, Units, "units") # Original didn't validate
        assert module is not None and isinstance(module, int)
        return self._dac_scales.get(module, {}).get("amplitude")

    async def set_adc_autocal(self, autocal, module=None):
        assert isinstance(autocal, bool)
        assert module is not None and isinstance(module, int)
        self._adc_autocal[module] = autocal

    async def get_adc_autocal(self, module=None):
        assert module is not None and isinstance(module, int)
        return self._adc_autocal.get(module, False) # Default to False if not set

    async def set_adc_calibration_mode(self, mode, module=None):
        mode = self._validate_enum_member(mode, ADCCalibrationMode, "ADC calibration mode")
        assert module is not None and isinstance(module, int)
        self._adc_calibration_mode[module] = mode

    async def get_adc_calibration_mode(self, module=None):
        assert module is not None and isinstance(module, int)
        return self._adc_calibration_mode.get(module)

    async def set_adc_calibration_coefficients(self, coefficients, block, module=None):
        assert isinstance(coefficients, list)
        block = self._validate_enum_member(block, ADCCalibrationBlock, "ADC calibration block")
        assert module is not None and isinstance(module, int)
        if module not in self._adc_calibration_coefficients:
            self._adc_calibration_coefficients[module] = {}
        self._adc_calibration_coefficients[module][block] = coefficients

    async def get_adc_calibration_coefficients(self, block, module=None):
        block = self._validate_enum_member(block, ADCCalibrationBlock, "ADC calibration block")
        assert module is not None and isinstance(module, int)
        return self._adc_calibration_coefficients.get(module, {}).get(block, [])

    async def clear_adc_calibration_coefficients(self, block, module=None):
        block = self._validate_enum_member(block, ADCCalibrationBlock, "ADC calibration block")
        assert module is not None and isinstance(module, int)
        if module in self._adc_calibration_coefficients:
            self._adc_calibration_coefficients[module].pop(block, None)

    async def set_nyquist_zone(self, zone, target=None, module=None):
        assert isinstance(zone, int) and zone in [1, 2, 3]
        if target is not None: # target is optional
            target = self._validate_enum_member(target, Target, "target")
        assert module is not None and isinstance(module, int)
        if module not in self._nyquist_zones:
            self._nyquist_zones[module] = {}
        self._nyquist_zones[module][target] = zone


    async def get_nyquist_zone(self, target=None, module=None):
        if target is not None: # target is optional
            target = self._validate_enum_member(target, Target, "target")
        assert module is not None and isinstance(module, int)
        return self._nyquist_zones.get(module, {}).get(target)

    async def get_firmware_release(self):
        return {"version": "MOCK.1.0.0", "build_date": "2024-01-01", "commit_hash": "mockcommit"}

    async def display(self, message=None):
        if message:
            print(f"MockCRS Display: {message}")
        else:
            print("MockCRS Display Cleared")

    async def hmc7044_peek(self, address):
        assert isinstance(address, int)
        return self._hmc7044_registers.get(address, 0x00) # Default to 0 if not poked

    async def hmc7044_poke(self, address, value):
        assert isinstance(address, int)
        assert isinstance(value, int) # Assuming byte value
        self._hmc7044_registers[address] = value

    async def get_samples(self, num_samples, channel=None, module=1, average=False):
        """Get sample data for a specific module, delegating to resonator model with enhanced scaling."""
        assert isinstance(num_samples, int), "Number of samples must be an integer"
        assert channel is None or isinstance(channel, int), "Channel must be None or an integer"
        assert module in [1, 2, 3, 4], "Invalid module number"
        await asyncio.sleep(0.0001) # Simulate hardware delay

        overrange = False # Mock flags
        overvoltage = False
        
        # Get sample rate based on decimation
        fir_stage = await self.get_decimation()
        if fir_stage is None:
            fir_stage = 6  # Default only if None, not if 0
        fs = 625e6 / 256 / 64 / (2**fir_stage)  # Actual sample rate based on decimation
        t_list = (np.arange(num_samples) / fs).tolist()

        nco_freq = await self.get_nco_frequency(module=module) or 0
        
        # Scaling constants
        scale_factor = const.SCALE_FACTOR  # Base scaling
        noise_level = const.UDP_NOISE_LEVEL  # Noise level

        if channel is not None:
            # Single channel logic - use coupled calculation for time-varying signals
            # Get the coupled response (which includes beat frequencies)
            module_responses = self._resonator_model.calculate_module_response_coupled(
                module, num_samples=num_samples, sample_rate=fs
            )
            
            if channel in module_responses:
                response = module_responses[channel]
                
                # Handle both single values and arrays
                if isinstance(response, np.ndarray):
                    # Time-varying signal (beat frequencies)
                    i_list = (response.real * scale_factor + np.random.normal(0, noise_level, num_samples)).tolist()
                    q_list = (response.imag * scale_factor + np.random.normal(0, noise_level, num_samples)).tolist()
                else:
                    # Single value - repeat for all samples
                    i_base = response.real * scale_factor
                    q_base = response.imag * scale_factor
                    i_list = [i_base + np.random.normal(0, noise_level) for _ in range(num_samples)]
                    q_list = [q_base + np.random.normal(0, noise_level) for _ in range(num_samples)]
            else:
                # Channel not configured
                i_list = [np.random.normal(0, noise_level) for _ in range(num_samples)]
                q_list = [np.random.normal(0, noise_level) for _ in range(num_samples)]
            
            return {
                "i": i_list, "q": q_list, "ts": t_list,
                "flags": {"overrange": overrange, "overvoltage": overvoltage}
            }
        else:
            # All channels logic - use coupled calculation
            channels_per_module = 1024 # As per NUM_CHANNELS in streamer
            i_data, q_data = [], []
            
            # Get all channel responses at once (includes coupling)
            module_responses = self._resonator_model.calculate_module_response_coupled(
                module, num_samples=num_samples, sample_rate=fs
            )
            
            # Process each channel
            for ch_idx in range(1, channels_per_module + 1): # 1-based channel
                if ch_idx in module_responses:
                    response = module_responses[ch_idx]
                    
                    # Handle both single values and arrays
                    if isinstance(response, np.ndarray):
                        # Time-varying signal
                        i_ch_samples = (response.real * scale_factor + np.random.normal(0, noise_level, num_samples)).tolist()
                        q_ch_samples = (response.imag * scale_factor + np.random.normal(0, noise_level, num_samples)).tolist()
                    else:
                        # Single value - repeat for all samples
                        i_base = response.real * scale_factor
                        q_base = response.imag * scale_factor
                        i_ch_samples = [i_base + np.random.normal(0, noise_level) for _ in range(num_samples)]
                        q_ch_samples = [q_base + np.random.normal(0, noise_level) for _ in range(num_samples)]
                else:
                    # Channel not configured - just noise
                    i_ch_samples = [np.random.normal(0, noise_level) for _ in range(num_samples)]
                    q_ch_samples = [np.random.normal(0, noise_level) for _ in range(num_samples)]
                
                i_data.append(i_ch_samples)
                q_data.append(q_ch_samples)

            if average:
                mean_i = [np.mean(ch_samples) for ch_samples in i_data]
                mean_q = [np.mean(ch_samples) for ch_samples in q_data]
                return {
                    "mean": {"i": mean_i, "q": mean_q},
                    "i": i_data, "q": q_data, "ts": t_list,
                    "flags": {"overrange": overrange, "overvoltage": overvoltage}
                }
            else:
                return {
                    "i": i_data, "q": q_data, "ts": t_list,
                    "flags": {"overrange": overrange, "overvoltage": overvoltage}
                }

    async def clear_channels(self, module=None):
        modules_to_clear = range(1, 5) if module is None else [module]
        keys_to_delete_freq = []
        keys_to_delete_amp = []
        keys_to_delete_phase = []

        for m in modules_to_clear:
            for ch in range(1, 1025): # Channels 1-1024
                if (m, ch) in self._frequencies: keys_to_delete_freq.append((m,ch))
                if (m, ch) in self._amplitudes: keys_to_delete_amp.append((m,ch))
                if (m, ch) in self._phases: keys_to_delete_phase.append((m,ch))
        
        for key in keys_to_delete_freq: self._frequencies.pop(key, None)
        for key in keys_to_delete_amp: self._amplitudes.pop(key, None)
        for key in keys_to_delete_phase: self._phases.pop(key, None)


    async def set_cable_length(self, length, module):
        assert isinstance(length, (int, float))
        assert isinstance(module, int)
        # Store the cable length for the module
        self._cable_lengths[module] = length
        return None # Important for Tuber serialization
    
    async def get_cable_length(self, module):
        """Get the cable length for a module."""
        assert isinstance(module, int)
        # Return stored cable length or default to 0.0
        return self._cable_lengths.get(module, 0.0)

    def raise_periscope(self, module=1, channels="1", 
                        buf_size=5000, fps=30.0, 
                        density_dot=1, blocking=None):
        from ..tools.periscope.__main__ import raise_periscope as base_raise_periscope
        
        # base_raise_periscope is async, need to run it in an event loop if called synchronously
        # For simplicity, assuming this MockCRS method might be called from async context
        # or the caller handles the event loop.
        # If this method itself needs to be async: `async def raise_periscope...`
        # and then `await base_raise_periscope(...)`

        # If this method must remain synchronous but call an async one:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError: # No running event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            created_loop = True
        else:
            created_loop = False

        result = loop.run_until_complete(base_raise_periscope(
            self, module=module, channels=channels,
            buf_size=buf_size, fps=fps, 
            density_dot=density_dot, blocking=blocking
        ))

        if created_loop:
            loop.close()
        return result

            
    # --- UDP Streaming Control ---
    async def start_udp_streaming(self, host='127.0.0.1', port=9876):
        return await self._udp_manager.start_udp_streaming(host, port)

    async def stop_udp_streaming(self):
        return await self._udp_manager.stop_udp_streaming()

    async def get_udp_streaming_status(self): # This was not async in original
        return self._udp_manager.get_udp_streaming_status()
    
# Export MockCRS as CRS for the flavour system
CRS = ClientMockCRS
