"""
Mock CRS Device - Core MockCRS class definition, state, and methods.
"""
import asyncio
import numpy as np
from enum import Enum
from datetime import datetime
import contextlib
import atexit
import weakref
import dataclasses
import time
import threading

# Import schema classes
from ..core.schema import CRS as BaseCRS

# Import helper classes from this package
from .resonator_model import MockResonatorModel
from .udp_streamer import MockUDPManager

from ..tuber.codecs import TuberResult
from ..streamer import LONG_PACKET_CHANNELS, SHORT_PACKET_CHANNELS, Timestamp, TimestampSource

_mock_crs_instances = weakref.WeakSet()
_cleanup_registered = False


def _cleanup_all_mock_crs_instances():
    """Emergency cleanup for all ServerMockCRS instances at program exit"""
    for instance in list(_mock_crs_instances):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(instance.stop_udp_streaming())
            loop.close()
            print(f"[CLEANUP] Shutdown completed for ServerMockCRS instance {id(instance)}")
        except Exception as e:
            print(f"[CLEANUP] Error during emergency cleanup: {e}")

# Enums
class ClockSource(str, Enum):
    VCXO = "VCXO"
    SMA = "SMA"
    BACKPLANE = "Backplane"

class TimestampPort(str, Enum):
    BACKPLANE = "BACKPLANE"
    TEST = "TEST"
    SMA = "SMA"

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
    ADC_COUNTS = "adc_counts"
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
    """Client-side Mock CRS - thin ORM wrapper.

    This is the client-side object that inherits from the ORM base class.
    It's a thin shim that communicates with ServerMockCRS via the mock tuber server.
    No cleanup needed - the server process handles that.
    """

    __mapper_args__ = {"polymorphic_identity": "MockCRS"}

    def __init__(self, serial, slot=None, crate=None, **kwargs):
        super().__init__(serial=serial, slot=slot, crate=crate, **kwargs)


class ServerMockCRS:
    """Server-side Mock CRS - full implementation.

    This is the server-side object that contains all the actual implementation.
    It does NOT inherit from BaseCRS and is interacted with remotely as a TuberObject.
    """
    _num_tuber_calls = 0

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

    @property
    def TIMESTAMP_PORT(self):
        return {
            'BACKPLANE': TimestampPort.BACKPLANE.value,
            'TEST': TimestampPort.TEST.value,
            'SMA': TimestampPort.SMA.value
        }

    @property
    def CLOCKSOURCE(self):
        return {
            'VCXO': ClockSource.VCXO.value,
            'SMA': ClockSource.SMA.value,
            'BACKPLANE': ClockSource.BACKPLANE.value
        }

    @property
    def UNITS(self):
        return {
            'HZ': Units.HZ.value, 'RAW': Units.RAW.value,
            'ADC_COUNTS': Units.ADC_COUNTS.value, 'VOLTS': Units.VOLTS.value,
            'AMPS': Units.AMPS.value, 'WATTS': Units.WATTS.value,
            'DEGREES': Units.DEGREES.value, 'RADIANS': Units.RADIANS.value,
            'OHMS': Units.OHMS.value, 'NORMALIZED': Units.NORMALIZED.value,
            'DB': Units.DB.value, 'DBM': Units.DBM.value
        }

    @property
    def TARGET(self):
        return {
            'CARRIER': Target.CARRIER.value, 'NULLER': Target.NULLER.value,
            'DEMOD': Target.DEMOD.value, 'ADC': Target.ADC.value,
            'DAC': Target.DAC.value
        }

    # Static properties for units, targets, sensors, etc. (as dictionaries)
    UNITS_DICT = {
        "HZ": "hz", "RAW": "raw", "ADC_COUNTS": "adc_counts", "VOLTS": "volts",
        "AMPS": "amps", "WATTS": "watts", "DEGREES": "degrees", "RADIANS": "radians",
        "OHMS": "ohms", "NORMALIZED": "normalized", "DB": "db", "DBM": "dbm"
    }
    TARGET_DICT = {
        "CARRIER": "carrier", "NULLER": "nuller", "DEMOD": "demod",
        "DAC": "dac", "ADC": "adc"
    }
    CLOCK_SOURCE_DICT = {"XTAL": "XTAL", "SMA": "SMA", "BACKPLANE": "Backplane"}
    TIMESTAMP_PORT_DICT = {"BACKPLANE": "BACKPLANE", "SMA": "SMA", "TEST": "TEST"}
    TEMPERATURE_SENSOR_DICT = {
        "MOTHERBOARD_POWER": "MOTHERBOARD_TEMPERATURE_POWER",
        "MOTHERBOARD_ARM": "MOTHERBOARD_TEMPERATURE_ARM",
        "MOTHERBOARD_FPGA": "MOTHERBOARD_TEMPERATURE_FPGA",
        "MOTHERBOARD_PHY": "MOTHERBOARD_TEMPERATURE_PHY",
    }
    RAIL_DICT = {
        "MB_VCC3V3": "MOTHERBOARD_RAIL_VCC3V3", "MB_VCC12V0": "MOTHERBOARD_RAIL_VCC12V0",
        "MB_VCC5V5": "MOTHERBOARD_RAIL_VCC5V5", "MB_VCC1V0_GTX": "MOTHERBOARD_RAIL_VCC1V0_GTX",
        "MB_VCC1V0": "MOTHERBOARD_RAIL_VCC1V0", "MB_VCC1V2": "MOTHERBOARD_RAIL_VCC1V2",
        "MB_VCC1V5": "MOTHERBOARD_RAIL_VCC1V5", "MB_VCC1V8": "MOTHERBOARD_RAIL_VCC1V8",
        "MB_VADJ": "MOTHERBOARD_RAIL_VADJ",
    }
    ADCCALIBRATIONMODE_DICT = {"AUTO": "AUTO", "MODE1": "MODE1", "MODE2": "MODE2"}
    ADCCALIBRATIONBLOCK_DICT = {"OCB1": "OCB1", "OCB2": "OCB2", "GCB": "GCB", "TSCB": "TSCB"}

    def __init__(self, serial, slot=None, crate=None, **kwargs):
        self._serial = serial
        self._slot = slot
        self._crate = crate

        global _mock_crs_instances, _cleanup_registered
        _mock_crs_instances.add(self)

        if not _cleanup_registered:
            atexit.register(_cleanup_all_mock_crs_instances)
            _cleanup_registered = True

        self._clock_source = None
        self._clock_priority = []
        self._timestamp_port = None
        self._timestamp = {"y": 2024, "d": 1, "h": 0, "m": 0, "s": 0, "c": 0}
        self._last_timestamp = Timestamp(
            y=2024, d=1, h=0, m=0, s=0, ss=0, c=0, sbs=0,
            source=TimestampSource.TEST, recent=True
        )
        self._max_samples_per_channel = 100000
        self._max_samples_per_module = 30000

        self._frequencies = {}
        self._amplitudes = {}
        self._phases = {}
        self._tuning_results = {}

        self._active_modules = [1, 2, 3, 4]
        self._high_bank = False
        self._fir_stage = 6
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
        self._nco_frequencies = {}
        self._adc_attenuators = {m: {"amplitude": 0.0, "units": self.Units.DB} for m in range(1, 5)}
        self._dac_scales = {m: {"amplitude": 1.0, "units": self.Units.DBM} for m in range(1, 5)}
        self._adc_autocal = {m: True for m in range(1, 5)}
        self._adc_calibration_mode = {m: self.ADCCalibrationMode.AUTO for m in range(1, 5)}
        self._adc_calibration_coefficients = {}
        self._nyquist_zones = {}
        self._hmc7044_registers = {}
        self._cable_lengths = {}

        # Store physics configuration
        from .config import defaults
        self._physics_config = defaults()

        self._prune_channels_over_limit()
        self.mock_start_time = time.time()
        self._config_lock = threading.RLock()

        # Instantiate helpers
        self._resonator_model = MockResonatorModel(self)
        self._udp_manager = MockUDPManager(self)

    def channels_per_module(self):
        if self._short_packets:
            return SHORT_PACKET_CHANNELS
        else:
            return LONG_PACKET_CHANNELS

    def _prune_channels_over_limit(self):
        max_channels = self.channels_per_module()
        for store in (self._frequencies, self._amplitudes, self._phases):
            for key in [k for k in store.keys() if k[1] > max_channels]:
                store.pop(key, None)

    def channels_per_module(self):
        if self._short_packets:
            return SHORT_PACKET_CHANNELS
        else:
            return LONG_PACKET_CHANNELS


    def _prune_channels_over_limit(self):
        max_channels = self.channels_per_module()
        for store in (self._frequencies, self._amplitudes, self._phases):
            for key in [k for k in store.keys() if k[1] > max_channel]:
                store.pop(key, None)
    
    async def generate_resonators(self, config=None):
        """Generate/regenerate resonators with current or provided parameters."""
        try:
            if not hasattr(self, 'resonator_model') or self._resonator_model is None:
                self._resonator_model = MockResonatorModel(self)

            if config:
                self._physics_config.update(config)

            active_config = self._physics_config
            num_resonances = active_config.get('num_resonances', 2)

            self._resonator_model.generate_resonators(
                num_resonances=num_resonances,
                config=active_config
            )

            resonator_count = len(self._resonator_model.mr_lekids)
            resonance_frequencies = self._resonator_model.resonator_frequencies.copy()

            auto_bias = active_config.get('auto_bias_kids', True)
            if auto_bias and resonance_frequencies:
                await self._auto_bias_kids(active_config, resonance_frequencies)

            return resonator_count, resonance_frequencies

        except Exception as e:
            print(f"Error in generate_resonators: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _auto_bias_kids(self, config, resonance_frequencies, amplitude=None):
        """Automatically configure channels at resonator frequencies."""
        try:
            if amplitude is None:
                amplitude = config.get('bias_amplitude', 0.01)
            module = 1

            print(f"[MockCRS] Auto-biasing {len(resonance_frequencies)} KIDs with amplitude {amplitude}")

            nco_freq = np.mean(resonance_frequencies)
            print(f"[MockCRS] Setting NCO frequency to {nco_freq:.3e} Hz")
            await self.set_nco_frequency(nco_freq, module=module)

            chan_limit = min(self.channels_per_module(), 256)
            configured_count = 0
            for i, freq_Hz in enumerate(resonance_frequencies[:chan_limit]):
                channel = i + 1
                relative_freq = freq_Hz - nco_freq
                await self.set_frequency(relative_freq, channel=channel, module=module)
                await self.set_amplitude(amplitude, channel=channel, module=module)
                await self.set_phase(0, channel=channel, module=module)
                configured_count += 1

            print(f"[MockCRS] Configured {configured_count} channels with automatic KID biasing")

        except Exception as e:
            print(f"[MockCRS] Error in auto-bias KIDs: {e}")
            import traceback
            traceback.print_exc()

    async def _auto_bias_kids(self, config, resonance_frequencies, amplitude=None):
        """Automatically configure channels at resonator frequencies (bias KIDs).
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing bias parameters
        resonance_frequencies : list
            List of resonator frequencies in Hz
        """
        try:
            # Get configuration parameters
            if amplitude is None:
                amplitude = config.get('bias_amplitude', 0.01)  # Normalized units
            module = 1  # Always use module 1 for mock
            
            print(f"[MockCRS] Auto-biasing {len(resonance_frequencies)} KIDs with amplitude {amplitude}")
            
            # Set NCO to the mean of resonance frequencies
            nco_freq = np.mean(resonance_frequencies)
            print(f"[MockCRS] Setting NCO frequency to {nco_freq:.3e} Hz")
            await self.set_nco_frequency(nco_freq, module=module)
            
            # Configure channels for each resonator (up to 256 channels per module)
            chan_limit = min(self.channels_per_module(), 256)
            configured_count = 0
            for i, freq_Hz in enumerate(resonance_frequencies[:chan_limit]):
                channel = i + 1  # Channels are 1-indexed
                
                # Set frequency relative to NCO
                relative_freq = freq_Hz - nco_freq
                
                # Configure the channel (these are synchronous methods)
                self.set_frequency(relative_freq, channel=channel, module=module)
                self.set_amplitude(amplitude, channel=channel, module=module)
                self.set_phase(0, channel=channel, module=module)
                
                configured_count += 1
                
            print(f"[MockCRS] Configured {configured_count} channels with automatic KID biasing")
            
        except Exception as e:
            print(f"[MockCRS] Error in auto-bias KIDs: {e}")
            import traceback
            traceback.print_exc()
            # Don't re-raise - this is a convenience feature, not critical

    def validate_enum_member(self, value, enum_class, name):
        if isinstance(value, str):
            try:
                return enum_class(value)
            except ValueError:
                valid_values = [e.value for e in enum_class]
                raise ValueError(f"Invalid {name} '{value}'. Must be one of {valid_values}")
        elif not isinstance(value, enum_class):
            raise TypeError(f"{name.capitalize()} must be a {enum_class.__name__} enum member or a valid string.")
        return value

    # --- Core CRS Methods ---
    async def set_frequency(self, frequency, channel=None, module=None):
        assert isinstance(frequency, (int, float)), "Frequency must be a number"
        min_freq_hz = -313.5e6
        max_freq_hz = 313.5e6
        if not (min_freq_hz <= frequency <= max_freq_hz):
            raise ValueError(f"The set frequency must be between -313.5 MHz and +313.5 MHz of the NCO frequency.")
        max_channel = self.channels_per_module()
        if not 1 <= channel <= max_channel:
            raise ValueError(f"Channel must be between 1 and {max_channel} for the current packet length.")
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        with self._config_lock:
            self._frequencies[(module, channel)] = frequency

    async def get_frequency(self, channel=None, module=None):
        assert channel is not None and isinstance(channel, int)
        assert module is not None and isinstance(module, int)
        return self._frequencies.get((module, channel))

    async def set_amplitude(self, amplitude, channel=None, module=None):
        assert isinstance(amplitude, (int, float)), "Amplitude must be a number"
        if not (-1.0 <= amplitude <= 1.0):
            raise ValueError("Amplitude must be between -1.0 and +1.0.")
        max_channel = self.channels_per_module()
        if not 1 <= channel <= max_channel:
            raise ValueError(f"Channel must be between 1 and {max_channel} for the current packet length.")
        assert channel is not None and isinstance(channel, int)
        assert module is not None and isinstance(module, int)
        with self._config_lock:
            self._amplitudes[(module, channel)] = amplitude

    async def get_amplitude(self, channel=None, module=None):
        assert channel is not None and isinstance(channel, int)
        assert module is not None and isinstance(module, int)
        return self._amplitudes.get((module, channel))

    async def set_phase(self, phase, units='DEGREES', target=None, channel=None, module=None):
        assert isinstance(phase, (int, float)), "Phase must be a number"
        assert channel is not None and isinstance(channel, int)
        assert module is not None and isinstance(module, int)
        max_channel = self.channels_per_module()
        if not 1 <= channel <= max_channel:
            raise ValueError(f"Channel must be between 1 and {max_channel} for the current packet length.")
        if isinstance(units, str):
            units = units.upper()
            if units not in ['DEGREES', 'RADIANS']:
                raise ValueError(f"Invalid phase units '{units}'. Must be 'DEGREES' or 'RADIANS'")
        if units == 'RADIANS' or (hasattr(units, 'value') and units.value == 'radians'):
            phase_degrees = phase * 180.0 / np.pi
        else:
            phase_degrees = phase
        with self._config_lock:
            self._phases[(module, channel)] = phase_degrees

    async def get_phase(self, units='DEGREES', target=None, channel=None, module=None):
        assert channel is not None and isinstance(channel, int)
        assert module is not None and isinstance(module, int)
        phase_degrees = self._phases.get((module, channel))
        if phase_degrees is None:
            return None
        if isinstance(units, str):
            units = units.upper()
            if units not in ['DEGREES', 'RADIANS']:
                raise ValueError(f"Invalid phase units '{units}'. Must be 'DEGREES' or 'RADIANS'")
        if units == 'RADIANS' or (hasattr(units, 'value') and units.value == 'radians'):
            return phase_degrees * np.pi / 180.0
        else:
            return phase_degrees

    async def set_timestamp_port(self, port):
        port = self.validate_enum_member(port, TimestampPort, "timestamp port")
        self._timestamp_port = port

    async def get_timestamp_port(self):
        return self._timestamp_port

    async def set_clock_source(self, clock_source=None):
        if clock_source is not None:
            clock_source = self.validate_enum_member(clock_source, ClockSource, "clock source")
        self._clock_source = clock_source

    async def get_clock_source(self):
        return self._clock_source

    async def set_clock_priority(self, clock_priority=None):
        if clock_priority is not None:
            clock_priority = [self.validate_enum_member(cs, ClockSource, "clock source") for cs in clock_priority]
        self._clock_priority = clock_priority

    async def get_clock_priority(self):
        return self._clock_priority

    async def get_motherboard_temperature(self, sensor):
        if sensor not in self.TEMPERATURE_SENSOR_DICT.values():
            raise ValueError(f"Invalid sensor '{sensor}'. Must be one of {list(self.TEMPERATURE_SENSOR_DICT.values())}")
        return self._temperature_sensors.get(sensor)

    async def get_motherboard_voltage(self, rail):
        if rail not in self.RAIL_DICT.values():
            raise ValueError(f"Invalid rail '{rail}'. Must be one of {list(self.RAIL_DICT.values())}")
        return self._rails.get(rail, {}).get("voltage")

    async def get_motherboard_current(self, rail):
        if rail not in self.RAIL_DICT.values():
            raise ValueError(f"Invalid rail '{rail}'. Must be one of {list(self.RAIL_DICT.values())}")
        return self._rails.get(rail, {}).get("current")

    async def set_decimation(self, stage: int = 6, short: bool = False, module: int | list[int] | None = None):
        assert isinstance(stage, int) and 0 <= stage <= 6
        if module is None:
            module = list(self._active_modules)
        if isinstance(module, int):
            module = [module]
        if (not isinstance(module, list) or not all(isinstance(m, int) for m in module)
                or not set(module) <= set(self._active_modules)):
            raise ValueError("Invalid 'module' argument to set_decimation!")
        if (stage <= 3) and (short == False):
            print(f"[Decimation<=3] Streaming only 128 channels")
            short = True
        self._fir_stage = stage
        self._short_packets = short
        self._streamed_modules = module

    async def get_decimation(self):
        return None if len(self._streamed_modules) == 0 else self._fir_stage

    async def set_analog_bank(self, high_bank: bool):
        self._high_bank = bool(high_bank)

    async def get_analog_bank(self):
        return bool(self.__dict__.get("_high_bank", False))

    async def _thread_lock_acquire(self):
        await asyncio.sleep(0.001)

    def _thread_lock_release(self):
        pass

    async def get_timestamp(self):
        return dict(self._last_timestamp)

    async def get_pfb_samples(self, num_samples, units=Units.NORMALIZED, channel=None, module=1):
        assert isinstance(num_samples, int) and num_samples > 0
        assert channel is not None and isinstance(channel, int)
        assert 0 <= channel < 1024
        assert module in [1, 2, 3, 4]
        units = self.validate_enum_member(units, Units, "units")
        assert units in [Units.NORMALIZED, Units.RAW]
        await asyncio.sleep(0.001)
        if units == Units.NORMALIZED:
            i_vals = np.random.uniform(-1.0, 1.0, num_samples).tolist()
            q_vals = np.random.uniform(-1.0, 1.0, num_samples).tolist()
        else:
            max_val = 2 ** 23 - 1
            i_vals = np.random.randint(-max_val - 1, max_val + 1, num_samples).tolist()
            q_vals = np.random.randint(-max_val - 1, max_val + 1, num_samples).tolist()
        return list(zip(i_vals, q_vals))

    async def get_fast_samples(self, num_samples, units=Units.NORMALIZED, module=1):
        assert isinstance(num_samples, int) and num_samples > 0
        assert module in [1, 2, 3, 4]
        units = self.validate_enum_member(units, Units, "units")
        assert units in [Units.NORMALIZED, Units.RAW]
        await asyncio.sleep(0.001)
        if units == Units.NORMALIZED:
            i_vals = np.random.uniform(-1.0, 1.0, num_samples).tolist()
            q_vals = np.random.uniform(-1.0, 1.0, num_samples).tolist()
        else:
            max_val = 2 ** 15 - 1
            i_vals = np.random.randint(-max_val - 1, max_val + 1, num_samples).tolist()
            q_vals = np.random.randint(-max_val - 1, max_val + 1, num_samples).tolist()
        return {"i": i_vals, "q": q_vals}

    async def set_nco_frequency(self, frequency, module=None):
        assert isinstance(frequency, (int, float))
        assert module is not None and isinstance(module, int)
        await asyncio.sleep(0.001)
        self._nco_frequencies[module] = frequency

    async def get_nco_frequency(self, module=None, target=None):
        assert module is not None and isinstance(module, int)
        return self._nco_frequencies.get(module, 0)

    async def set_adc_attenuator(self, attenuation, module=None):
        assert isinstance(attenuation, int) and 0 <= attenuation <= 27
        assert module is not None and isinstance(module, int)
        self._adc_attenuators[module] = {"amplitude": float(attenuation), "units": self.Units.DB}

    async def get_adc_attenuator(self, module=None):
        assert module is not None and isinstance(module, int)
        return self._adc_attenuators.get(module, {}).get("amplitude")

    async def set_dac_scale(self, amplitude, units='DBM', module=None):
        assert isinstance(amplitude, (int, float))
        assert module is not None and isinstance(module, int)
        self._dac_scales[module] = {"amplitude": amplitude, "units": units}

    async def get_dac_scale(self, units='DBM', module=None):
        assert module is not None and isinstance(module, int)
        return self._dac_scales.get(module, {}).get("amplitude")

    async def set_adc_autocal(self, autocal, module=None):
        assert isinstance(autocal, bool)
        assert module is not None and isinstance(module, int)
        self._adc_autocal[module] = autocal

    async def get_adc_autocal(self, module=None):
        assert module is not None and isinstance(module, int)
        return self._adc_autocal.get(module, False)

    async def set_adc_calibration_mode(self, mode, module=None):
        mode = self.validate_enum_member(mode, ADCCalibrationMode, "ADC calibration mode")
        assert module is not None and isinstance(module, int)
        self._adc_calibration_mode[module] = mode

    async def get_adc_calibration_mode(self, module=None):
        assert module is not None and isinstance(module, int)
        return self._adc_calibration_mode.get(module)

    async def set_adc_calibration_coefficients(self, coefficients, block, module=None):
        assert isinstance(coefficients, list)
        block = self.validate_enum_member(block, ADCCalibrationBlock, "ADC calibration block")
        assert module is not None and isinstance(module, int)
        if module not in self._adc_calibration_coefficients:
            self._adc_calibration_coefficients[module] = {}
        self._adc_calibration_coefficients[module][block] = coefficients

    async def get_adc_calibration_coefficients(self, block, module=None):
        block = self.validate_enum_member(block, ADCCalibrationBlock, "ADC calibration block")
        assert module is not None and isinstance(module, int)
        return self._adc_calibration_coefficients.get(module, {}).get(block, [])

    async def clear_adc_calibration_coefficients(self, block, module=None):
        block = self.validate_enum_member(block, ADCCalibrationBlock, "ADC calibration block")
        assert module is not None and isinstance(module, int)
        if module in self._adc_calibration_coefficients:
            self._adc_calibration_coefficients[module].pop(block, None)

    async def set_nyquist_zone(self, zone, target=None, module=None):
        assert isinstance(zone, int) and zone in [1, 2, 3]
        if target is not None:
            target = self.validate_enum_member(target, Target, "target")
        assert module is not None and isinstance(module, int)
        if module not in self._nyquist_zones:
            self._nyquist_zones[module] = {}
        self._nyquist_zones[module][target] = zone

    async def get_nyquist_zone(self, target=None, module=None):
        if target is not None:
            target = self.validate_enum_member(target, Target, "target")
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
        return self._hmc7044_registers.get(address, 0x00)

    async def hmc7044_poke(self, address, value):
        assert isinstance(address, int)
        assert isinstance(value, int)
        self._hmc7044_registers[address] = value

    async def get_samples(self, num_samples, channel=None, module=1, average=False):
        """Get sample data for a specific module using direct physics calculations."""
        assert isinstance(num_samples, int)
        assert channel is None or isinstance(channel, int)
        assert module in [1, 2, 3, 4]
        await asyncio.sleep(0.0001)

        overrange = False
        overvoltage = False

        fir_stage = await self.get_decimation()
        if fir_stage is None:
            fir_stage = 6
        fs = 625e6 / 256 / 64 / (2 ** fir_stage)
        t_list = (np.arange(num_samples) / fs).tolist()

        cfg = self._physics_config if hasattr(self, 'physics_config') else {}
        scale_factor = cfg.get('scale_factor', 2 ** 21)
        noise_level = cfg.get('udp_noise_level', 10.0)
        current_time = time.time() - self.mock_start_time

        if average:
            effective_num_samples = 1
        else:
            effective_num_samples = num_samples

        module_responses = self._resonator_model.calculate_module_response_coupled(
            module, num_samples=effective_num_samples, sample_rate=fs, start_time=int(current_time)
        )

        if channel is not None:
            if channel in module_responses:
                # Get the S21 response with QP noise for this channel
                response = module_responses[channel]
                if isinstance(response, np.ndarray):
                    i_list = (response.real * scale_factor).tolist()
                    q_list = (response.imag * scale_factor).tolist()
                else:
                    i_base = response.real * scale_factor
                    q_base = response.imag * scale_factor
                    i_list = [i_base] * num_samples
                    q_list = [q_base] * num_samples
            else:
                i_list = np.random.normal(0, noise_level, num_samples).tolist()
                q_list = np.random.normal(0, noise_level, num_samples).tolist()

            return {
                "i": i_list, "q": q_list, "ts": t_list,
                "flags": {"overrange": overrange, "overvoltage": overvoltage}
            }
        else:
            channels_per_module = self.channels_per_module()
            i_data, q_data = [], []

            for ch_idx in range(1, channels_per_module + 1):
                if ch_idx in module_responses:
                    response = module_responses[ch_idx]
                    if isinstance(response, np.ndarray):
                        i_ch_samples = (response.real * scale_factor).tolist()
                        q_ch_samples = (response.imag * scale_factor).tolist()
                    else:
                        i_base = response.real * scale_factor
                        q_base = response.imag * scale_factor
                        i_ch_samples = [i_base] * num_samples
                        q_ch_samples = [q_base] * num_samples
                else:
                    i_ch_samples = np.random.normal(0, noise_level, num_samples).tolist()
                    q_ch_samples = np.random.normal(0, noise_level, num_samples).tolist()
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

        with self._config_lock:
            for m in modules_to_clear:
                for ch in range(1, 1025):
                    if (m, ch) in self._frequencies:
                        keys_to_delete_freq.append((m, ch))
                    if (m, ch) in self._amplitudes:
                        keys_to_delete_amp.append((m, ch))
                    if (m, ch) in self._phases:
                        keys_to_delete_phase.append((m, ch))

            for key in keys_to_delete_freq:
                self._frequencies.pop(key, None)
            for key in keys_to_delete_amp:
                self._amplitudes.pop(key, None)
            for key in keys_to_delete_phase:
                self._phases.pop(key, None)

    async def set_cable_length(self, length, module):
        assert isinstance(length, (int, float))
        assert isinstance(module, int)
        self._cable_lengths[module] = length
        return None

    async def get_cable_length(self, module):
        assert isinstance(module, int)
        return self._cable_lengths.get(module, 0.0)

    async def raise_periscope(self, module=1, channels="1", buf_size=5000, fps=30.0, density_dot=1, blocking=None):
        from ..tools.periscope.__main__ import raise_periscope as base_raise_periscope
        from .. import streamer

        original_get_local_ip = streamer.get_local_ip

        def mock_get_local_ip(crs_hostname):
            if any(host_part in str(crs_hostname) for host_part in ["127.0.0.1", "localhost", "::1"]):
                return "127.0.0.1"
            return original_get_local_ip(crs_hostname)

        try:
            streamer.get_local_ip = mock_get_local_ip
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
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
        finally:
            streamer.get_local_ip = original_get_local_ip

    # --- UDP Streaming Control ---
    async def start_udp_streaming(self, host='127.0.0.1', port=9876):
        return await self._udp_manager.start_udp_streaming(host, port)

    async def stop_udp_streaming(self):
        return await self._udp_manager.stop_udp_streaming()

    async def get_udp_streaming_status(self):
        return self._udp_manager.get_udp_streaming_status()

    # --- Quasiparticle Pulse Control ---
    async def set_pulse_mode(self, mode, **kwargs):
        if not hasattr(self, 'resonator_model') or self._resonator_model is None:
            raise RuntimeError("Resonator model not available.")
        if not hasattr(self._resonator_model, 'set_pulse_mode'):
            raise RuntimeError("Resonator model does not support pulse functionality.")
        await asyncio.sleep(0.001)
        return self._resonator_model.set_pulse_mode(mode, **kwargs)

    async def add_pulse_event(self, resonator_index, start_time, amplitude=None):
        if not hasattr(self, 'resonator_model') or self._resonator_model is None:
            raise RuntimeError("Resonator model not available.")
        if not hasattr(self._resonator_model, 'add_pulse_event'):
            raise RuntimeError("Resonator model does not support pulse functionality.")
        await asyncio.sleep(0.001)
        return self._resonator_model.add_pulse_event(resonator_index, start_time, amplitude)

    async def get_pulse_status(self):
        if not hasattr(self, 'resonator_model') or self._resonator_model is None:
            return {'mode': 'unavailable', 'active_pulses': 0, 'error': 'Resonator model not available'}
        if not hasattr(self._resonator_model, 'pulse_config'):
            return {'mode': 'unsupported', 'active_pulses': 0, 'error': 'Pulse functionality not supported'}
        await asyncio.sleep(0.001)
        return {
            'mode': self._resonator_model.pulse_config.get('mode', 'unknown'),
            'config': self._resonator_model.pulse_config.copy(),
            'active_pulses': len(getattr(self._resonator_model, 'pulse_events', [])),
            'error': None
        }



# Export ClientMockCRS as CRS for the ORM/flavour system
CRS = ClientMockCRS
