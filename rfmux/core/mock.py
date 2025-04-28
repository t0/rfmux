"""
Mock CRS Device - Core structure, YAML hook, class properties, and methods.
"""

__all__ = [
    "MockCRS"
]

import asyncio
import json
import socket
import time
import logging
import multiprocessing
from aiohttp import web
import atexit
from enum import Enum
import numpy as np
import types

from .schema import CRS

mp_ctx = multiprocessing.get_context("fork")

def yaml_hook(hwm):
    """Patch up the HWM using mock Dfmuxes instead of real ones.

    To do so, we alter the hostname associated with the Dfmux objects
    in the HWM and redirect HTTP requests to a local server. Each Dfmux
    gets a distinct port, which is used to route requests to a distinct
    model class.
    """

    # These objects are hardware models of the dfmuxes, indexed by the port number
    # used to connect with them over HTTP.
    models = {}

    # Find all Dfmux objects in the database and patch up their hostnames to
    # something local.
    sockets = []
    for d in hwm.query(CRS):

        # Create a socket to be shared with the server process.
        s = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        s.bind(("localhost", 0))
        (hostname, port) = s.getsockname()

        sockets.append(s)
        d.hostname = f"{hostname}:{port}"
        models[port] = MockCRS(
            serial=d.serial if d.serial else ("%05d" % port),
            slot=d.slot if d.crate else None,
            crate=d.crate.serial if d.crate else None,
        )

    hwm.commit()

    l = mp_ctx.Semaphore(0)

    p = ServerProcess(sockets=sockets, models=models, lock=l)
    p.start()
    l.acquire()

    atexit.register(p.terminate)

    # In the client process, we do not need the sockets -- in fact, we don't
    # want a reference hanging around.
    for s in sockets:
        s.close()


# Start up a web server. This is a distinct process, so COW semantics.
class ServerProcess(mp_ctx.Process):
    daemon = True

    def __init__(self, sockets, models, lock):
        self.sockets = sockets
        self.models = models
        self.lock = lock
        super().__init__()

    def run(self):
        loop = asyncio.new_event_loop()

        app = web.Application()
        app.add_routes([web.post("/tuber", self.post_handler)])

        for s in self.sockets:
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())
            site = web.SockSite(runner, s)
            loop.run_until_complete(site.start())

        self.lock.release()
        loop.run_forever()

    async def post_handler(self, request):
        port = request.url.port
        model = self.models[port]

        body = await request.text()

        await model._thread_lock_acquire()

        try:
            request_data = json.loads(body)
            model._num_tuber_calls += 1

            if isinstance(request_data, list):
                response = [await self.__single_handler(model, r) for r in request_data]
            elif isinstance(request_data, dict):
                response = await self.__single_handler(model, request_data)
            else:
                response = {"error": "Didn't know what to do!"}

            # Convert to serializable format
            serializable_response = convert_to_serializable(response)

            return web.Response(text=json.dumps(serializable_response), content_type='application/json')

        except Exception as e:
            raise e
        finally:
            model._thread_lock_release()

    async def __single_handler(self, model, request):
        """Handle a single Tuber request"""

        if "method" in request and hasattr(model, request["method"]):
            m = getattr(model, request["method"])
            a = request.get("args", [])
            k = request.get("kwargs", {})
            r = e = None
            try:
                if asyncio.iscoroutinefunction(m):
                    r = await m(*a, **k)
                else:
                    r = m(*a, **k)
            except Exception as oops:
                e = {"message": "%s: %s" % (oops.__class__.__name__, str(oops))}
            return {"result": r, "error": e}

        elif "property" in request and hasattr(model, request["property"]):
            prop = getattr(model, request["property"])
            if callable(prop):
                return {
                    "result": {
                        "name": request["property"],
                        "args": [],
                        "explanation": "(...)",
                    }
                }
            else:
                return {"result": getattr(model, request["property"])}

        elif "object" in request:
            # We need to provide some metadata to TuberObject so it can populate
            # properties and methods on the client-side object.
            illegal_prefixes = ("__", "_MockCRS__", "_thread_lock_")
            names = set(
                filter(lambda s: not s.startswith(illegal_prefixes), dir(model))
            )
            methods = list(filter(lambda n: callable(getattr(model, n)), names))
            properties = list(
                filter(lambda n: not callable(getattr(model, n)), names)
            )

            return {
                "result": {
                    "name": "TuberObject",
                    "summary": "",
                    "explanation": "",
                    "properties": properties,
                    "methods": methods,
                }
            }

        else:
            return {"error": "Didn't know what to do!"}



def convert_to_serializable(obj):
    """Recursively convert NumPy arrays in the object to lists."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class ClockSource(str, Enum):
    VCXO = "VCXO"
    SMA = "SMA"
    BACKPLANE = "Backplane"

class TimestampPort(str, Enum):
    BACKPLANE = "BACKPLANE"
    SMA = "SMA"
    TEST = "TEST"

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

class MockCRS:
    """Mock MKIDS device emulation class."""
    _num_tuber_calls = 0

    # Enums
    Units = Units
    Target = Target
    ClockSource = ClockSource
    TimestampPort = TimestampPort
    FrequencyUnit = FrequencyUnit
    AmplitudeUnit = AmplitudeUnit
    PhaseUnit = PhaseUnit
    ADCCalibrationMode = ADCCalibrationMode
    ADCCalibrationBlock = ADCCalibrationBlock

    # Static properties for units, targets, sensors, etc.
    UNITS = {
        "HZ": "hz",
        "RAW": "raw",
        "VOLTS": "volts",
        "AMPS": "amps",
        "WATTS": "watts",
        "DEGREES": "degrees",
        "RADIANS": "radians",
        "OHMS": "ohms",
        "NORMALIZED": "normalized",
        "DB": "db",
        "DBM": "dbm",
        "WATTS": "watts"
    }

    TARGET = {
        "CARRIER": "carrier",
        "NULLER": "nuller",
        "DEMOD": "demod",
        "DAC" : "dac",
        "ADC" : "adc",
    }

    CLOCK_SOURCE = {
        "XTAL": "XTAL",
        "SMA": "SMA",
        "BACKPLANE": "Backplane",
    }

    TIMESTAMP_PORT = {
        "BACKPLANE": "BACKPLANE",
        "SMA": "SMA",
        "TEST": "TEST",
    }

    TEMPERATURE_SENSOR = {
        "MOTHERBOARD_POWER": "MOTHERBOARD_TEMPERATURE_POWER",
        "MOTHERBOARD_ARM": "MOTHERBOARD_TEMPERATURE_ARM",
        "MOTHERBOARD_FPGA": "MOTHERBOARD_TEMPERATURE_FPGA",
        "MOTHERBOARD_PHY": "MOTHERBOARD_TEMPERATURE_PHY",
    }

    RAIL = {
        "MB_VCC3V3": "MOTHERBOARD_RAIL_VCC3V3",
        "MB_VCC12V0": "MOTHERBOARD_RAIL_VCC12V0",
        "MB_VCC5V5": "MOTHERBOARD_RAIL_VCC5V5",
        "MB_VCC1V0_GTX": "MOTHERBOARD_RAIL_VCC1V0_GTX",
        "MB_VCC1V0": "MOTHERBOARD_RAIL_VCC1V0",
        "MB_VCC1V2": "MOTHERBOARD_RAIL_VCC1V2",
        "MB_VCC1V5": "MOTHERBOARD_RAIL_VCC1V5",
        "MB_VCC1V8": "MOTHERBOARD_RAIL_VCC1V8",
        "MB_VADJ": "MOTHERBOARD_RAIL_VADJ",
    }

    ADCCALIBRATIONMODE = {
        "AUTO": "AUTO",
        "MODE1": "MODE1",
        "MODE2": "MODE2",
    }

    ADCCALIBRATIONBLOCK = {
        "OCB1": "OCB1",
        "OCB2": "OCB2",
        "GCB": "GCB",
        "TSCB": "TSCB",
    }

    def __init__(self, serial, slot=None, crate=None):
        self.serial = serial
        self.slot = slot
        self.crate = crate
        self.clock_source = None
        self.clock_priority = []
        self.timestamp_port = None
        self.timestamp = {
            "y": 2024,
            "d": 280,
            "h": 12,
            "m": 0,
            "s": 0,
            "c": 0
        }
        self.max_samples_per_channel = 100000
        self.max_samples_per_module = 30000

        # Initialize resonator parameters
        self.resonator_frequencies = []
        self.resonator_Q_factors = []

        # Initialize dynamic resonance parameters
        self.ff_factor = 0.01  # Frequency shift scaling factor (ffs)
        self.p_sky = 0        # External fixed power source (ps)
        self.resonator_current_f0s = None  # To be set in generate_resonators
        self.resonator_pe = None            # Power estimates for each resonator

        num_resonators = 10
        f_start = 1e9     # 1 MHz
        f_end = 2e9     # 10 MHz
        nominal_Q = 1000
        min_spacing = 0.3e6  # 100 kHz

        # # Generate resonators
        self.generate_resonators(num_resonators, f_start, f_end, nominal_Q, min_spacing)
        self.initialize_dynamic_resonators()

        self.frequencies = {}
        self.amplitudes = {}
        self.phases = {}
        self.tuning_results = {}
        self.fir_stage = 6

        # Initialize temperature sensors with unique default values
        self.temperature_sensors = {
            "MOTHERBOARD_TEMPERATURE_POWER": 30.0,
            "MOTHERBOARD_TEMPERATURE_ARM": 35.0,
            "MOTHERBOARD_TEMPERATURE_FPGA": 40.0,
            "MOTHERBOARD_TEMPERATURE_PHY": 45.0,
        }

        # Initialize rails with unique default voltage and current values
        self.rails = {
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

        # New attributes for the additional methods
        self.rfdc_initialized = False
        self.nco_frequencies = {}
        # Initialize ADC attenuators for all modules to 0 dB
        self.adc_attenuators = {}
        for module in range(1, 5):  # Assuming modules are numbered from 1 to 4
            self.adc_attenuators[module] = {
                "amplitude": 0.0,
                "units": self.UNITS["DB"]
            }
        # Initialize DAC scales for all modules to 1 dBm
        self.dac_scales = {}
        for module in range(1, 5):
            self.dac_scales[module] = {
                "amplitude": 1.0,
                "units": self.UNITS["DBM"]
            }
        # Initialize ADC auto calibration to True for all modules
        self.adc_autocal = {}
        for module in range(1, 5):
            self.adc_autocal[module] = True

        # Initialize ADC calibration mode to 'AUTO' for all modules
        self.adc_calibration_mode = {}
        for module in range(1, 5):
            self.adc_calibration_mode[module] = self.ADCCALIBRATIONMODE['AUTO']

        self.adc_calibration_coefficients = {}
        self.nyquist_zones = {}
        self.hmc7044_registers = {}

    def validate_enum_member(self, value, enum_class, name):
        """Validate that value is a member of enum_class, converting strings if necessary."""
        if isinstance(value, str):
            try:
                value = enum_class(value)
            except ValueError:
                valid_values = [e.value for e in enum_class]
                raise ValueError(f"Invalid {name} '{value}'. Must be one of {valid_values}")
        elif not isinstance(value, enum_class):
            raise TypeError(f"{name.capitalize()} must be a {enum_class.__name__} enum member or a valid string.")
        return value

    def set_frequency(self, frequency, units=FrequencyUnit.HZ, channel=None, module=None):
        """Set frequency for a specific channel and module."""
        assert isinstance(frequency, (int, float)), "Frequency must be a number"
        units = self.validate_enum_member(units, FrequencyUnit, "frequency unit")

    # Validate frequency range (-313.5 MHz to +313.5 MHz)
        min_freq_hz = -313.5e6
        max_freq_hz = 313.5e6
        if not (min_freq_hz <= frequency <= max_freq_hz):
            raise ValueError(f"Frequency must be between -313.5 MHz and +313.5 MHz.")


        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        self.frequencies[(module, channel)] = {
            "value": frequency,
            "units": units
        }

    def get_frequency(self, units=FrequencyUnit.HZ, channel=None, module=None):
        """Get frequency for a specific channel and module."""
        units = self.validate_enum_member(units, FrequencyUnit, "frequency unit")
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        freq_info = self.frequencies.get((module, channel))
        if freq_info is None:
            return None
        return freq_info["value"]

    def set_amplitude(self, amplitude, units=AmplitudeUnit.NORMALIZED, target=Target.CARRIER, channel=None, module=None):
        """Set amplitude for a specific channel and module."""
        assert isinstance(amplitude, (int, float)), "Amplitude must be a number"
        units = self.validate_enum_member(units, AmplitudeUnit, "amplitude unit")
        target = self.validate_enum_member(target, Target, "target")
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        # Validate amplitude range when units are normalized
        if units == self.AmplitudeUnit.NORMALIZED:
            if not (-1.0 <= amplitude <= 1.0):
                raise ValueError("Amplitude must be between -1.0 and +1.0 when units are 'normalized'.")


        self.amplitudes[(module, channel, target)] = {
            "value": amplitude,
            "units": units
        }

    def get_amplitude(self, units=AmplitudeUnit.NORMALIZED, target=Target.CARRIER, channel=None, module=None):
        """Get amplitude for a specific channel and module."""
        units = self.validate_enum_member(units, AmplitudeUnit, "amplitude unit")
        target = self.validate_enum_member(target, Target, "target")
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        amp_info = self.amplitudes.get((module, channel, target))
        if amp_info is None:
            return None
        return amp_info["value"]

    def set_phase(self, phase, units=PhaseUnit.DEGREES, target=Target.CARRIER, channel=None, module=None):
        """Set phase for a specific channel and module."""
        assert isinstance(phase, (int, float)), "Phase must be a number"
        units = self.validate_enum_member(units, PhaseUnit, "phase unit")
        target = self.validate_enum_member(target, Target, "target")
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        self.phases[(module, channel, target)] = {
            "value": phase,
            "units": units
        }

    def get_phase(self, units=PhaseUnit.DEGREES, target=Target.CARRIER, channel=None, module=None):
        """Get phase for a specific channel and module."""
        units = self.validate_enum_member(units, PhaseUnit, "phase unit")
        target = self.validate_enum_member(target, Target, "target")
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        phase_info = self.phases.get((module, channel, target))
        if phase_info is None:
            return None
        return phase_info["value"]

    def set_timestamp_port(self, port):
        """Set the timestamp port."""
        port = self.validate_enum_member(port, TimestampPort, "timestamp port")
        self.timestamp_port = port

    def get_timestamp_port(self):
        """Get the current timestamp port."""
        return self.timestamp_port

    def set_clock_source(self, clock_source=None):
        """Set the clock source."""
        if clock_source is not None:
            clock_source = self.validate_enum_member(clock_source, ClockSource, "clock source")
        self.clock_source = clock_source

    def get_clock_source(self):
        """Get the current clock source."""
        return self.clock_source

    def set_clock_priority(self, clock_priority=None):
        """Set clock priority."""
        if clock_priority is not None:
            clock_priority = [self.validate_enum_member(cs, ClockSource, "clock source") for cs in clock_priority]
        self.clock_priority = clock_priority

    def get_clock_priority(self):
        """Get the current clock priority."""
        return self.clock_priority

    def get_motherboard_temperature(self, sensor):
        """Get the motherboard temperature for a specific sensor."""
        if sensor not in self.temperature_sensors:
            raise ValueError(f"Invalid sensor '{sensor}'. Must be one of {list(self.temperature_sensors.keys())}")
        return self.temperature_sensors[sensor]

    def get_motherboard_voltage(self, rail):
        """Get the voltage of a specific motherboard rail."""
        if rail not in self.rails:
            raise ValueError(f"Invalid rail '{rail}'. Must be one of {list(self.rails.keys())}")
        return self.rails[rail]["voltage"]

    def get_motherboard_current(self, rail):
        """Get the current of a specific motherboard rail."""
        if rail not in self.rails:
            raise ValueError(f"Invalid rail '{rail}'. Must be one of {list(self.rails.keys())}")
        return self.rails[rail]["current"]

    def set_fir_stage(self, stage):
        """Set the FIR stage."""
        assert isinstance(stage, int), "FIR stage must be an integer"
        self.fir_stage = stage

    def get_fir_stage(self):
        """Get the FIR stage."""
        return self.fir_stage

    async def _thread_lock_acquire(self):
        """Simulate thread lock acquisition."""
        await asyncio.sleep(0.01)

    def _thread_lock_release(self):
        """Simulate thread lock release."""
        pass

    def get_timestamp(self):
        """Get the current timestamp."""
        # For simulation purposes, return a static or incremented timestamp
        # You can enhance this method to return more realistic timestamps
        return self.timestamp


    async def get_pfb_samples(self, num_samples, units=Units.NORMALIZED, channel=None, module=1):
        """Get PFB samples for a specific channel and module."""
        # Validate inputs
        assert isinstance(num_samples, int) and num_samples > 0, "Number of samples must be a positive integer"
        assert channel is not None and isinstance(channel, int), "Channel must be specified and be an integer"
        assert 0 <= channel < 1024, "Invalid channel number"
        assert module in [1, 2, 3, 4], "Invalid module number"
        units = self.validate_enum_member(units, Units, "units")
        assert units in [Units.NORMALIZED, Units.RAW], "Units must be 'normalized' or 'raw'"

        # Simulate data retrieval delay
        await asyncio.sleep(0.01)

        # Generate sample data
        if units == Units.NORMALIZED:
            # Generate normalized i and q values between -1.0 and 1.0
            i_values = np.random.uniform(-1.0, 1.0, num_samples)
            q_values = np.random.uniform(-1.0, 1.0, num_samples)
        else:
            # Generate raw i and q integer values (24-bit signed integers)
            max_int_value = 8388607  # 2^23 - 1
            i_values = np.random.randint(-max_int_value - 1, max_int_value + 1, num_samples)
            q_values = np.random.randint(-max_int_value - 1, max_int_value + 1, num_samples)

        # Create a list of tuples (i, q)
        samples = list(zip(i_values.tolist(), q_values.tolist()))
        return samples

    async def get_fast_samples(self, num_samples, units=Units.NORMALIZED, module=1):
        """Get fast samples for a specific module."""
        # Validate inputs
        assert isinstance(num_samples, int) and num_samples > 0, "Number of samples must be a positive integer"
        assert module in [1, 2, 3, 4], "Invalid module number"
        units = self.validate_enum_member(units, Units, "units")
        assert units in [Units.NORMALIZED, Units.RAW], "Units must be 'normalized' or 'raw'"

        # Simulate data retrieval delay
        await asyncio.sleep(0.01)

        # Generate sample data
        if units == Units.NORMALIZED:
            # Generate normalized i and q values between -1.0, 1.0
            i_values = np.random.uniform(-1.0, 1.0, num_samples).tolist()
            q_values = np.random.uniform(-1.0, 1.0, num_samples).tolist()
        else:
            # Generate raw i and q integer values (16-bit signed integers)
            max_int_value = 32767  # 2^15 - 1
            i_values = np.random.randint(-max_int_value - 1, max_int_value + 1, num_samples).tolist()
            q_values = np.random.randint(-max_int_value - 1, max_int_value + 1, num_samples).tolist()

        # Return a dictionary with "i" and "q" arrays
        return {
            "i": i_values,
            "q": q_values
        }

    # Additional methods
    def set_nco_frequency(self, frequency, units=Units.HZ, target=None, module=None):
        """Set NCO frequency."""
        assert isinstance(frequency, (int, float)), "Frequency must be a number"
        units = self.validate_enum_member(units, Units, "units")
        if target is not None:
            target = self.validate_enum_member(target, Target, "target")
        assert module is not None and isinstance(module, int), "Module must be an integer"
        if target is None:
            self.nco_frequencies[(module, Target.ADC)] = {"frequency": frequency, "units": units}
            self.nco_frequencies[(module, Target.DAC)] = {"frequency": frequency, "units": units}

        else:
            self.nco_frequencies[(module, target)] = {"frequency": frequency, "units": units}


    def get_nco_frequency(self, units=Units.HZ, target=None, module=None):
        """Get NCO frequency."""
        units = self.validate_enum_member(units, Units, "units")
        if target is not None:
            target = self.validate_enum_member(target, Target, "target")
        assert module is not None and isinstance(module, int), "Module must be an integer"
        freq_info = self.nco_frequencies.get((module, target))
        if freq_info is None:
            return None
        return freq_info["frequency"]

    def set_adc_attenuator(self, attenuation, units=Units.DB, module=None):
        """Set ADC attenuator."""
        assert isinstance(attenuation, int) and 0 <= attenuation <= 27, "Amplitude must be an integer between 0 and 27"
        units = self.validate_enum_member(units, Units, "units")
        assert units == Units.DB, "Units must be 'db' for ADC attenuator"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        self.adc_attenuators[module] = {"amplitude": attenuation, "units": units}

    def get_adc_attenuator(self, units=Units.DB, module=None):
        """Get ADC attenuator."""
        units = self.validate_enum_member(units, Units, "units")
        assert units == Units.DB, "Units must be 'db' for ADC attenuator"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        attenuator_info = self.adc_attenuators.get(module)
        if attenuator_info is None:
            return None
        return attenuator_info["amplitude"]

    def set_dac_scale(self, amplitude, units=Units.DBM, module=None):
        """Set DAC scale."""
        assert isinstance(amplitude, (int, float)), "Amplitude must be a number"
        units = self.validate_enum_member(units, Units, "units")
        assert units in [Units.DBM, Units.WATTS, Units.AMPS], "Units must be 'dbm', 'watts', or 'amps'"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        self.dac_scales[module] = {"amplitude": amplitude, "units": units}

    def get_dac_scale(self, units=Units.DBM, module=None):
        """Get DAC scale."""
        units = self.validate_enum_member(units, Units, "units")
        assert units in [Units.DBM, Units.WATTS, Units.AMPS], "Units must be 'dbm', 'watts', or 'amps'"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        scale_info = self.dac_scales.get(module)
        if scale_info is None:
            return None
        return scale_info["amplitude"]

    def set_adc_autocal(self, autocal, module=None):
        """Set ADC auto calibration."""
        assert isinstance(autocal, bool), "Autocal must be a boolean"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        self.adc_autocal[module] = autocal

    def get_adc_autocal(self, module=None):
        """Get ADC auto calibration."""
        assert module is not None and isinstance(module, int), "Module must be an integer"
        return self.adc_autocal.get(module, False)

    def set_adc_calibration_mode(self, mode, module=None):
        """Set ADC calibration mode."""
        mode = self.validate_enum_member(mode, ADCCalibrationMode, "ADC calibration mode")
        assert module is not None and isinstance(module, int), "Module must be an integer"
        self.adc_calibration_mode[module] = mode

    def get_adc_calibration_mode(self, module=None):
        """Get ADC calibration mode."""
        assert module is not None and isinstance(module, int), "Module must be an integer"
        return self.adc_calibration_mode.get(module, None)

    def set_adc_calibration_coefficients(self, coefficients, block, module=None):
        """Set ADC calibration coefficients."""
        assert isinstance(coefficients, list), "Coefficients must be a list of integers"
        block = self.validate_enum_member(block, ADCCalibrationBlock, "ADC calibration block")
        assert module is not None and isinstance(module, int), "Module must be an integer"
        if module not in self.adc_calibration_coefficients:
            self.adc_calibration_coefficients[module] = {}
        self.adc_calibration_coefficients[module][block] = coefficients

    def get_adc_calibration_coefficients(self, block, module=None):
        """Get ADC calibration coefficients."""
        block = self.validate_enum_member(block, ADCCalibrationBlock, "ADC calibration block")
        assert module is not None and isinstance(module, int), "Module must be an integer"
        return self.adc_calibration_coefficients.get(module, {}).get(block, [])

    def clear_adc_calibration_coefficients(self, block, module=None):
        """Clear ADC calibration coefficients."""
        block = self.validate_enum_member(block, ADCCalibrationBlock, "ADC calibration block")
        assert module is not None and isinstance(module, int), "Module must be an integer"
        if module in self.adc_calibration_coefficients:
            if block in self.adc_calibration_coefficients[module]:
                del self.adc_calibration_coefficients[module][block]

    def set_nyquist_zone(self, zone, target=None, module=None):
        """Set Nyquist zone."""
        assert isinstance(zone, int), "Zone must be an integer"
        assert zone in [1, 2, 3], "Zone must be 1, 2, or 3"
        if target is not None:
            target = self.validate_enum_member(target, Target, "target")
        assert module is not None and isinstance(module, int), "Module must be an integer"
        if module not in self.nyquist_zones:
            self.nyquist_zones[module] = {}
        self.nyquist_zones[module][target] = zone

    def get_nyquist_zone(self, target=None, module=None):
        """Get Nyquist zone."""
        if target is not None:
            target = self.validate_enum_member(target, Target, "target")
        assert module is not None and isinstance(module, int), "Module must be an integer"
        return self.nyquist_zones.get(module, {}).get(target, None)

    def get_firmware_release(self):
        """Get firmware release information."""
        return {
            "version": "1.0.0",
            "build_date": "2024-10-07",
            "commit_hash": "abcdef1234567890"
        }

    def display(self, message=None):
        """Display a message."""
        if message:
            print(f"Display message: {message}")
        else:
            print("Clearing display")

    def hmc7044_peek(self, address):
        """Peek at HMC7044 register."""
        assert isinstance(address, int), "Address must be an integer"
        return self.hmc7044_registers.get(address, 0x00)

    def hmc7044_poke(self, address, value):
        """Poke HMC7044 register."""
        assert isinstance(address, int), "Address must be an integer"
        assert isinstance(value, int), "Value must be an integer"
        self.hmc7044_registers[address] = value


    def initialize_dynamic_resonators(self):
        """Initialize dynamic resonance parameters based on generated resonators."""
        if not self.resonator_frequencies.size or not self.resonator_Q_factors.size:
            # Avoid initializing before resonators are generated
            self.resonator_current_f0s = np.array([])
            self.resonator_pe = np.array([])
            return
        
        self.num_resonators = len(self.resonator_frequencies)
        self.resonator_current_f0s = np.copy(self.resonator_frequencies)
        self.resonator_pe = np.zeros(self.num_resonators)
        
        # Initialize history tracking for multiple resonators
        self.resonator_f0_history = [[] for _ in range(self.num_resonators)]

    def generate_resonators(self, num_resonators, f_start, f_end, nominal_Q, min_spacing):
        """Generate random resonator frequencies and Q factors."""
        frequencies = []
        Q_factors = []  # Correct initialization

        # Calculate the total frequency range
        freq_range = f_end - f_start

        # Ensure that the total bandwidth can accommodate the resonators with the minimum spacing
        max_resonators = int(freq_range / min_spacing)
        assert num_resonators <= max_resonators, "Cannot fit resonators with the given minimum spacing."

        # Generate frequencies with minimum spacing
        attempts = 0
        max_attempts = num_resonators * 100  # Limit to prevent infinite loops
        while len(frequencies) < num_resonators and attempts < max_attempts:
            freq = np.random.uniform(f_start, f_end)
            if all(abs(freq - f) >= min_spacing for f in frequencies):
                frequencies.append(freq)
                # Assign a Q factor with some random variation
                Q = nominal_Q * np.random.uniform(0.9, 1.1)
                Q_factors.append(Q)
            attempts += 1

        if len(frequencies) < num_resonators:
            raise ValueError("Could not generate the required number of resonators with the given constraints.")

        # Sort frequencies for consistency
        frequencies, Q_factors = zip(*sorted(zip(frequencies, Q_factors)))

        # Store the resonator parameters
        self.resonator_frequencies = np.array(frequencies)
        self.resonator_Q_factors = np.array(Q_factors)
        
        # Initialize dynamic resonance parameters
        self.initialize_dynamic_resonators()

    def s21_response(self, frequency):
        """Compute the combined S21 response of the resonator comb at a given frequency."""
        # Initialize S21 to 1 (no attenuation)
        s21 = 1.0 + 0j  # Complex number

        # For each resonator, compute its contribution based on current f0
        for idx in range(self.num_resonators):
            f0 = self.resonator_current_f0s[idx]
            Q = self.resonator_Q_factors[idx]
            delta_f = frequency - f0
            # Compute the complex S21 response for the inverted resonance
            s21_resonator = 1 - (1 / (1 + 2j * Q * delta_f / f0))
            # Multiply the responses (assuming serial resonators)
            s21 *= s21_resonator

        # Return magnitude and phase
        s21_mag = abs(s21)
        s21_phase = np.angle(s21)
        return s21_mag, s21_phase  # Magnitude and phase in radians

    def update_resonator_frequencies(self, s21_mag_array):
        """
        Update resonance frequencies based on power estimates.

        Args:
            s21_mag_array (np.ndarray): Array of S21 magnitudes for each resonator.
        """

        # Power estimate pe is proportional to |1 - S21_resonator|
        #pe = (1.0 - s21_mag_array) ** 2
        pe = (s21_mag_array) ** 2

        self.resonator_pe = pe

        # Update resonance frequencies based on pe
        self.resonator_current_f0s *= (1 - (self.resonator_pe + self.p_sky) * self.ff_factor)
        #self.resonator_current_f0s -= (self.resonator_pe + self.p_sky) * self.ff_factor


    async def get_samples(self, num_samples, channel=None, module=1):
        """Get sample data for a specific module."""
        # Validate inputs
        assert isinstance(num_samples, int), "Number of samples must be an integer"
        assert channel is None or isinstance(channel, int), "Channel must be None or an integer"
        assert module in [1, 2, 3, 4], "Invalid module number"
        await asyncio.sleep(0.01)

        # Simulate overrange and overvoltage flags (fixed for simplicity)
        overrange = False
        overvoltage = False

        # Sample rate for time vector
        fs = 1e6  # 1 MHz sample rate
        t = np.arange(num_samples) / fs  # Time vector

        nco_freq = self.get_nco_frequency(units=Units.HZ, target='adc', module=module)
        # Initialize t_list once
        t_list = t.tolist()

        if channel is not None:
            # Handle single channel
            # Retrieve the frequency, amplitude, and phase for the channel
            chan_freq = self.frequencies.get((module, channel))
            amp_info = self.amplitudes.get((module, channel, self.TARGET['CARRIER']))
            phase_info = self.phases.get((module, channel, self.TARGET['CARRIER']))

            if chan_freq is None or amp_info is None or phase_info is None:
                raise ValueError("Frequency, amplitude, or phase not set for channel")

            frequency = chan_freq['value'] + nco_freq    # Frequency in Hz            
            amplitude = amp_info['value']   # Amplitude (normalized)
            phase = np.deg2rad(phase_info['value'])  # Convert phase to radians

            # Initialize lists to store samples
            i_samples = []
            q_samples = []

            # Converge S21 response:
            for i in range(1000):
                # Compute S21 response
                s21_mag, s21_phase = self.s21_response(frequency)
                # Update resonance frequencies based on power estimates
                self.update_resonator_frequencies(np.array([amplitude*s21_mag]))  # Single resonator

            for sample_idx in range(num_samples):

                # Total phase includes channel phase and S21 phase
                total_phase = phase + s21_phase

                # Generate I and Q samples
                i = amplitude * s21_mag * np.cos(total_phase)
                q = amplitude * s21_mag * np.sin(total_phase)

                # Add random noise
                noise_level = 0.0  # Adjust noise level as needed
                i += np.random.normal(0, noise_level)
                q += np.random.normal(0, noise_level)

                # Append samples
                i_samples.append(i)
                q_samples.append(q)

            # Convert to lists for JSON serialization
            i_list = i_samples  # Already a Python list
            q_list = q_samples  # Already a Python list

            return {
                "i": i_list,
                "q": q_list,
                "ts": t_list,
                "flags": {"overrange": overrange, "overvoltage": overvoltage}
            }
        else:
            # Handle all channels
            channels_per_module = 1024

            # Initialize lists to store samples
            i_data = []
            q_data = []

            for ch in range(channels_per_module):
                chan_freq = self.frequencies.get((module, ch))
                amp_info = self.amplitudes.get((module, ch, self.TARGET['CARRIER']))
                phase_info = self.phases.get((module, ch, self.TARGET['CARRIER']))

                if chan_freq is None or amp_info is None or phase_info is None:
                    # If not set, default to zero amplitude
                    i = np.zeros(num_samples)
                    q = np.zeros(num_samples)
                else:
                    frequency = chan_freq['value']+nco_freq    # Frequency in Hz
                    amplitude = amp_info['value']   # Amplitude (normalized)
                    phase = np.deg2rad(phase_info['value'])  # Convert phase to radians


                    # Converge S21 response:
                    for i in range(1000):
                        # Compute S21 response
                        s21_mag, s21_phase = self.s21_response(frequency)
                        # Update resonance frequencies based on power estimates
                        self.update_resonator_frequencies(np.array([amplitude*s21_mag]))  # Single resonator

                    # Generate I and Q samples
                    total_phase = phase + s21_phase
                    i = amplitude * s21_mag * np.cos(total_phase) * np.ones(num_samples)
                    q = amplitude * s21_mag * np.sin(total_phase) * np.ones(num_samples)

                    # Add random noise
                    noise_level = 0.0  # Adjust noise level as needed
                    i += np.random.normal(0, noise_level, size=np.shape(i) if np.shape(i) else None)
                    q += np.random.normal(0, noise_level, size=np.shape(q) if np.shape(q) else None)
                # Convert to lists for JSON serialization
                i_list = i.tolist()
                q_list = q.tolist()

                i_data.append(i_list)
                q_data.append(q_list)

            return {
                "i": i_data,
                "q": q_data,
                "ts": t_list,
                "flags": {"overrange": overrange, "overvoltage": overvoltage}
            }