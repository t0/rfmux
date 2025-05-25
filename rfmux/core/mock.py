"""
Mock CRS Device - Core structure, YAML hook, class properties, and methods.
"""

__all__ = [
    "CRS",  # This will be MockCRS
    "Crate",
    "ReadoutModule",
    "ReadoutChannel",
    "Wafer",
    "Resonator",
    "ChannelMapping",
    "yaml_hook"
]

import asyncio
import json
import socket
import struct
import time
import logging
import multiprocessing
from aiohttp import web
import atexit
from enum import Enum
import numpy as np
import types
import threading
import array
from datetime import datetime
import contextlib
from typing import Dict, List, Tuple, Optional

# Import all schema classes we need to re-export
from .schema import (
    CRS as BaseCRS,
    Crate,
    ReadoutModule,
    ReadoutChannel,
    Wafer,
    Resonator,
    ChannelMapping
)
from ..streamer import (
    DfmuxPacket, Timestamp, TimestampPort,
    STREAMER_MAGIC, STREAMER_VERSION, NUM_CHANNELS, SS_PER_SECOND
)

# Import measurement algorithms EARLY to ensure they're registered on CRS class
# This must happen before yaml_hook creates MockCRS instances
try:
    from ..algorithms.measurement import take_netanal
    # Import other algorithms here as needed
    # from ..algorithms.measurement import multisweep
except ImportError:
    pass  # Algorithms are optional

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

    # Find all CRS objects in the database and patch up their hostnames to
    # something local.
    sockets = []
    for d in hwm.query(BaseCRS):

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
        
        # Force reload of algorithm modules to re-execute @macro decorators in server process
        self._reload_algorithm_modules()

        app = web.Application()
        app.add_routes([web.post("/tuber", self.post_handler)])

        for s in self.sockets:
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())
            site = web.SockSite(runner, s)
            loop.run_until_complete(site.start())

        self.lock.release()
        loop.run_forever()

    def _reload_algorithm_modules(self):
        """
        Force reload of algorithm modules to re-execute @macro decorators in server process.
        
        This method discovers and reloads all algorithm modules to ensure that
        @macro decorated functions are properly registered on the server side.
        """
        import importlib
        import sys
        import pkgutil
        
        try:
            import rfmux.algorithms
            
            # Discover and reload all algorithm modules
            for importer, modname, ispkg in pkgutil.walk_packages(
                path=rfmux.algorithms.__path__, 
                prefix=rfmux.algorithms.__name__ + "."
            ):
                if modname in sys.modules:
                    print(f"[RELOAD] Reloading algorithm module: {modname}")
                    importlib.reload(sys.modules[modname])
                else:
                    try:
                        print(f"[RELOAD] Importing algorithm module: {modname}")
                        __import__(modname)
                    except ImportError as e:
                        print(f"[RELOAD] Failed to import {modname}: {e}")
                        pass  # Skip modules that can't be imported
                        
        except ImportError:
            print("[RELOAD] No algorithms package found, skipping algorithm reload")
            pass  # algorithms package not available

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

            return web.Response(body=json.dumps(serializable_response).encode('utf-8'), content_type='application/json')

        except Exception as e:
            raise e
        finally:
            model._thread_lock_release()

    async def __single_handler(self, model, request):
        """Handle a single Tuber request"""
        
        # Debug logging
        print(f"[DEBUG] Tuber request: {request}")
        
        # Handle resolve requests first
        if request.get("resolve", False):
            # Return object metadata
            return await self.__single_handler(model, {"object": request.get("object", "Dfmux")})

        if "method" in request and request["method"] is not None:
            method_name = request["method"]
            object_name = request.get("object")
            
            # Debug log
            print(f"[DEBUG] Method request: method={method_name}, object={object_name}")
            
            # Look for the method on the instance first, then on the class
            m = None
            if hasattr(model, method_name):
                m = getattr(model, method_name)
            elif hasattr(type(model), method_name):
                # Method might be on the class (e.g., macro-registered algorithms)
                class_method = getattr(type(model), method_name)
                # Bind it to the instance
                m = class_method.__get__(model, type(model))
            
            if m is None:
                return {"error": {"message": f"Method '{method_name}' not found"}}
            
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

        elif "property" in request:
            prop_name = request["property"]
            if hasattr(model, prop_name):
                prop = getattr(model, prop_name)
                if callable(prop):
                    # Debug logging for method property requests
                    print(f"[DEBUG] Property request for callable '{prop_name}'")
                    
                    # Return metadata for methods
                    import inspect
                    sig = None
                    try:
                        sig = str(inspect.signature(prop))
                    except:
                        sig = "(...)"
                    
                    doc = inspect.getdoc(prop) or f"Method {prop_name}"
                    
                    # Return a TuberResult-like structure for method metadata
                    return {
                        "result": {
                            "__name__": prop_name,
                            "__signature__": sig,
                            "__doc__": doc,
                        }
                    }
                else:
                    return {"result": prop}
            else:
                # Property doesn't exist
                return {"error": {"message": f"Property '{prop_name}' not found"}}

        elif "object" in request:
            obj_name = request["object"]
            
            # CRITICAL FIX: Handle case where object is a list like ['get_frequency']
            # This happens when Tuber client calls methods - it puts the method name in object field
            if isinstance(obj_name, list) and len(obj_name) == 1:
                method_name = obj_name[0]
                # Check if this is actually a method on our model
                if hasattr(model, method_name) and callable(getattr(model, method_name)):
                    # This is actually a method call! Redirect to method handler
                    print(f"[DEBUG] Redirecting object request {obj_name} to method call")
                    return await self.__single_handler(model, {
                        "method": method_name,
                        "args": request.get("args", []),
                        "kwargs": request.get("kwargs", {})
                    })
            
            # Normal object metadata request
            # We need to provide some metadata to TuberObject so it can populate
            # properties and methods on the client-side object.
            illegal_prefixes = ("__", "_MockCRS__", "_thread_lock_", "_resolve", "_sa_")
            # Also exclude Tuber-specific methods and SQLAlchemy properties that shouldn't be exposed
            exclude_methods = {
                "tuber_resolve", "tuber_context", "object_factory", 
                "_resolve_meta", "_resolve_method", "_resolve_object",
                "_context_class", "reconstruct", "to_query"
            }
            exclude_properties = {
                "metadata", "registry", "_sa_class_manager", "_sa_instance_state",
                "_sa_registry", "is_container", "modules", "module", "crate",
                "hwm", "tuber_hostname"
            }
            
            names = set(
                filter(lambda s: not any(s.startswith(p) for p in illegal_prefixes) 
                                 and s not in exclude_methods
                                 and s not in exclude_properties, 
                       dir(model))
            )
            
            methods = []
            properties = []
            
            for name in names:
                try:
                    attr = getattr(model, name)
                    if callable(attr):
                        # Only include methods that are not coroutines from parent classes
                        if not (asyncio.iscoroutinefunction(attr) and 
                                name in ['tuber_resolve', 'resolve']):
                            methods.append(name)
                    else:
                        # Double-check it's not in exclude list
                        if name not in exclude_properties:
                            properties.append(name)
                except:
                    # Skip attributes that can't be accessed
                    pass

            return {
                "result": {
                    "name": "TuberObject",
                    "summary": "",
                    "explanation": "",
                    "properties": sorted(properties),
                    "methods": sorted(methods),
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


def _dfmuxpacket_to_bytes(self: DfmuxPacket) -> bytes:
    """
    Serialize a DfmuxPacket into 8240 bytes, matching streamer.py 'from_bytes()'.
    Layout:
      - 16-byte header (<IHHBBBBI)
      - 8192-byte channel data (NUM_CHANNELS*2 int32)
      - 32-byte timestamp (<8I)
    """
    import struct
    
    c_masked = self.ts.c & 0x1FFFFFFF
    # encode source
    source_map = {
        TimestampPort.BACKPLANE: 0,
        TimestampPort.TEST: 1,
        TimestampPort.SMA: 2,
        TimestampPort.TEST: 3,  # Use TEST as default instead of GND
    }
    source_val = source_map.get(self.ts.source, 0)
    c_masked |= (source_val << 29)
    # if recent => bit31
    if self.ts.recent:
        c_masked |= 0x80000000

    # 1) Header
    hdr_struct = struct.Struct("<IHHBBBBI")
    hdr = hdr_struct.pack(
        self.magic,
        self.version,
        self.serial,
        self.num_modules,
        self.block,
        self.fir_stage,
        self.module,
        self.seq,
    )

    # 2) Channel data
    body_bytes = self.s.tobytes()
    if len(body_bytes) != NUM_CHANNELS*2*4:
        raise ValueError(f"Channel data must be 1024*2=2048 int => 8192 bytes.")
    
    # 3) Timestamp
    ts_struct = struct.Struct("<8I")
    ts_data = ts_struct.pack(
        self.ts.y,
        self.ts.d,
        self.ts.h,
        self.ts.m,
        self.ts.s,
        self.ts.ss,
        c_masked,
        self.ts.sbs,
    )

    packet = hdr + body_bytes + ts_data
    if len(packet) != 8240:  # STREAMER_LEN
        raise ValueError(f"Packet length mismatch: {len(packet)} != 8240")
    return packet


# Monkey-patch the to_bytes method onto DfmuxPacket
DfmuxPacket.to_bytes = _dfmuxpacket_to_bytes


class MockCRSContext:
    """Context manager for batched CRS operations"""
    
    def __init__(self, mock_crs):
        self.mock_crs = mock_crs
        self.pending_ops = []
    
    def _add_call(self, **kwargs):
        """Add a call to the context (for Tuber compatibility)"""
        if 'resolve' in kwargs and kwargs['resolve']:
            # This is a resolve operation
            self.pending_ops.append(('_resolve', {}))
        elif 'method' in kwargs:
            # This is a method call
            method_name = kwargs['method']
            args = kwargs.get('args', [])
            call_kwargs = kwargs.get('kwargs', {})
            self.pending_ops.append((method_name, call_kwargs))
    
    def set_frequency(self, frequency, channel=None, module=None):
        """Queue a frequency setting operation"""
        self.pending_ops.append(('set_frequency', {
            'frequency': frequency,
            'channel': channel,
            'module': module
        }))
    
    def set_amplitude(self, amplitude, channel=None, module=None):
        """Queue an amplitude setting operation"""
        self.pending_ops.append(('set_amplitude', {
            'amplitude': amplitude,
            'channel': channel,
            'module': module
        }))
    
    def set_phase(self, phase, channel=None, module=None):
        """Queue a phase setting operation"""
        self.pending_ops.append(('set_phase', {
            'phase': phase,
            'channel': channel,
            'module': module
        }))
    
    async def __call__(self):
        """Execute all pending operations"""
        # Check if this is a resolve operation
        for op in self.pending_ops:
            if op == ('_resolve', {}):
                # Return metadata for the MockCRS object
                return [{"result": {
                    "name": "MockCRS",
                    "summary": "Mock CRS device",
                    "explanation": "Mock implementation of CRS for testing",
                    "properties": [],  # Will be populated by server
                    "methods": []      # Will be populated by server
                }}]
        
        # Execute normal operations
        for op_name, kwargs in self.pending_ops:
            method = getattr(self.mock_crs, op_name)
            method(**kwargs)
        return []


class MockCRSUDPStreamer(threading.Thread):
    """Streams UDP packets with S21 data based on MockCRS state"""
    
    def __init__(self, mock_crs, host='127.0.0.1', port=9876):
        super().__init__(daemon=True)
        self.mock_crs = mock_crs
        self.host = host
        self.port = port
        
        # For MockCRS, use unicast to avoid multicast issues
        # Real hardware uses multicast, but for testing unicast is more reliable
        from ..streamer import STREAMER_PORT
        self.unicast_host = host if host != '239.192.0.2' else '127.0.0.1'
        self.unicast_port = port if port != 9876 else STREAMER_PORT
        
        # Create UDP socket for unicast
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)
        
        self.running = True
        self.seq_counters = {m: 0 for m in range(1, 5)}  # Per-module sequence numbers
        self.packets_sent = 0
        print(f"[UDP] MockCRSUDPStreamer initialized - unicast to {self.unicast_host}:{self.unicast_port}")
        
    def stop(self):
        """Stop the streaming thread"""
        print("[UDP] Stopping UDP streamer...")
        self.running = False
        
    def run(self):
        """Stream UDP packets at the configured sample rate"""
        print("[UDP] UDP streaming thread started")
        packet_count = 0
        last_log_time = time.time()
        
        while self.running:
            start_time = time.perf_counter()
            
            # Generate and send packet for each module
            for module in range(1, 5):
                try:
                    packet = self.generate_packet(module, self.seq_counters[module])
                    # Send unicast for MockCRS (more reliable than multicast on loopback)
                    bytes_sent = self.socket.sendto(packet, (self.unicast_host, self.unicast_port))
                    self.seq_counters[module] += 1
                    self.packets_sent += 1
                    packet_count += 1
                    
                    # Log first few packets in detail
                    if self.packets_sent <= 5:
                        print(f"[UDP] Sent unicast packet #{self.packets_sent}: module={module}, seq={self.seq_counters[module]-1}, size={bytes_sent} bytes to {self.unicast_host}:{self.unicast_port}")
                    
                except Exception as e:
                    print(f"[UDP] ERROR generating/sending packet for module {module}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Log periodically (every 1 second)
            current_time = time.time()
            if current_time - last_log_time >= 1.0:
                print(f"[UDP] Status: {packet_count} packets sent in last second, total={self.packets_sent}")
                packet_count = 0
                last_log_time = current_time
            
            # Sleep to maintain sample rate
            fir_stage = self.mock_crs.get_fir_stage()
            sample_rate = 625e6 / (256 * 64 * 2**fir_stage)
            frame_time = 1.0 / sample_rate
            
            elapsed = time.perf_counter() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        print(f"[UDP] UDP streaming thread stopped. Total packets sent: {self.packets_sent}")
    
    def generate_packet(self, module, seq):
        """Generate a DfmuxPacket with current channel states"""
        # Create channel data array
        arr = array.array("i", [0] * (NUM_CHANNELS * 2))
        
        non_zero_channels = 0
        for channel in range(NUM_CHANNELS):
            s21_complex = self.mock_crs.calculate_channel_response(module, channel+1)
            
            # Convert to int32 range
            # Note: Real hardware uses different scaling, but this works for testing
            scale = 32767  # Use 16-bit range for better dynamic range
            i_val = np.clip(s21_complex.real * scale, -2147483648, 2147483647)
            q_val = np.clip(s21_complex.imag * scale, -2147483648, 2147483647)
            
            arr[channel * 2] = int(i_val)
            arr[channel * 2 + 1] = int(q_val)
            
            if i_val != 0 or q_val != 0:
                non_zero_channels += 1
        
        # Log first packet details
        if seq == 0:
            print(f"[UDP] First packet for module {module}: {non_zero_channels} non-zero channels")
        
        # Create timestamp
        now = datetime.now()
        ts = Timestamp(
            y=now.year % 100,
            d=now.timetuple().tm_yday,
            h=now.hour,
            m=now.minute,
            s=now.second,
            ss=int(now.microsecond * SS_PER_SECOND / 1e6),
            c=0,
            sbs=0,
            source=TimestampPort.TEST,
            recent=True
        )
        
        # Build packet
        packet = DfmuxPacket(
            magic=STREAMER_MAGIC,
            version=STREAMER_VERSION,
            serial=int(self.mock_crs.serial) if self.mock_crs.serial.isdigit() else 0,
            num_modules=1,
            block=0,
            fir_stage=self.mock_crs.get_fir_stage(),
            module=module-1,  # 0-indexed in packet
            seq=seq,
            s=arr,
            ts=ts
        )
        
        packet_bytes = packet.to_bytes()
        
        # Verify packet size on first packet
        if seq == 0:
            print(f"[UDP] Packet size check for module {module}: {len(packet_bytes)} bytes (expected 8240)")
            
        return packet_bytes


class ClockSource(str, Enum):
    VCXO = "VCXO"
    SMA = "SMA"
    BACKPLANE = "Backplane"

# TimestampPort is imported from streamer module

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

class MockCRS(BaseCRS):
    """Mock MKIDS device emulation class."""
    _num_tuber_calls = 0
    
    # Override polymorphic identity
    __mapper_args__ = {"polymorphic_identity": "MockCRS"}

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

    def __init__(self, serial, slot=None, crate=None, **kwargs):
        # Call parent __init__ to handle SQLAlchemy setup
        super().__init__(serial=serial, slot=slot, crate=crate, **kwargs)
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
        
        # UDP streaming attributes
        self.udp_streaming = False
        self.udp_thread = None
        self.lc_resonances = []
        
        # Initialize with LC resonances instead of original resonators
        self.generate_lc_resonances()

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

    def set_frequency(self, frequency, channel=None, module=None):
        """Set frequency for a specific channel and module."""
        assert isinstance(frequency, (int, float)), "Frequency must be a number"

        # Validate frequency range (-313.5 MHz to +313.5 MHz)
        min_freq_hz = -313.5e6
        max_freq_hz = 313.5e6
        if not (min_freq_hz <= frequency <= max_freq_hz):
            raise ValueError(f"Frequency must be between -313.5 MHz and +313.5 MHz.")

        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        self.frequencies[(module, channel)] = frequency

    def get_frequency(self, channel=None, module=None):
        """Get frequency for a specific channel and module."""
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        return self.frequencies.get((module, channel))

    def set_amplitude(self, amplitude, channel=None, module=None):
        """Set amplitude for a specific channel and module."""
        assert isinstance(amplitude, (int, float)), "Amplitude must be a number"
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        # Validate amplitude range (normalized)
        if not (-1.0 <= amplitude <= 1.0):
            raise ValueError("Amplitude must be between -1.0 and +1.0.")

        self.amplitudes[(module, channel)] = amplitude

    def get_amplitude(self, channel=None, module=None):
        """Get amplitude for a specific channel and module."""
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        return self.amplitudes.get((module, channel))

    def set_phase(self, phase, channel=None, module=None):
        """Set phase for a specific channel and module."""
        assert isinstance(phase, (int, float)), "Phase must be a number"
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        self.phases[(module, channel)] = phase

    def get_phase(self, channel=None, module=None):
        """Get phase for a specific channel and module."""
        assert channel is not None and isinstance(channel, int), "Channel must be an integer"
        assert module is not None and isinstance(module, int), "Module must be an integer"

        return self.phases.get((module, channel))

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
    def set_nco_frequency(self, frequency, module=None):
        """Set NCO frequency."""
        assert isinstance(frequency, (int, float)), "Frequency must be a number"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        
        # Set NCO frequency for the module
        self.nco_frequencies[module] = frequency

    def get_nco_frequency(self, module=None, target=None):
        """Get NCO frequency."""
        assert module is not None and isinstance(module, int), "Module must be an integer"
        # Ignore target parameter for compatibility but don't require it
        return self.nco_frequencies.get(module)

    def set_adc_attenuator(self, attenuation, module=None):
        """Set ADC attenuator."""
        assert isinstance(attenuation, int) and 0 <= attenuation <= 27, "Amplitude must be an integer between 0 and 27"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        self.adc_attenuators[module] = {"amplitude": attenuation, "units": "db"}

    def get_adc_attenuator(self, module=None):
        """Get ADC attenuator."""
        assert module is not None and isinstance(module, int), "Module must be an integer"
        attenuator_info = self.adc_attenuators.get(module)
        if attenuator_info is None:
            return None
        return attenuator_info["amplitude"]

    def set_dac_scale(self, amplitude, units='DBM', module=None):
        """Set DAC scale."""
        assert isinstance(amplitude, (int, float)), "Amplitude must be a number"
        assert module is not None and isinstance(module, int), "Module must be an integer"
        # Accept string units for compatibility but don't validate
        self.dac_scales[module] = {"amplitude": amplitude, "units": units}

    def get_dac_scale(self, units='DBM', module=None):
        """Get DAC scale."""
        # Accept string units for compatibility
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
        if (isinstance(self.resonator_frequencies, list) and not self.resonator_frequencies) or \
           (hasattr(self.resonator_frequencies, 'size') and not self.resonator_frequencies.size) or \
           (isinstance(self.resonator_Q_factors, list) and not self.resonator_Q_factors) or \
           (hasattr(self.resonator_Q_factors, 'size') and not self.resonator_Q_factors.size):
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


    async def get_samples(self, num_samples, channel=None, module=1, average=False):
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

        nco_freq = self.get_nco_frequency(module=module)
        if nco_freq is None:
            nco_freq = 0  # Default NCO frequency to 0 if not set
        # Initialize t_list once
        t_list = t.tolist()

        if channel is not None:
            # Handle single channel
            # Retrieve the frequency, amplitude, and phase for the channel
            chan_freq = self.frequencies.get((module, channel))
            chan_amp = self.amplitudes.get((module, channel))
            chan_phase = self.phases.get((module, channel))

            if chan_freq is None or chan_amp is None or chan_phase is None:
                raise ValueError("Frequency, amplitude, or phase not set for channel")

            frequency = chan_freq + nco_freq    # Frequency in Hz
            amplitude = chan_amp   # Amplitude (normalized)
            phase = np.deg2rad(chan_phase)  # Convert phase to radians

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
                chan_freq = self.frequencies.get((module, ch+1))  # 1-based channel numbers
                chan_amp = self.amplitudes.get((module, ch+1))
                chan_phase = self.phases.get((module, ch+1))

                if chan_freq is None or chan_amp is None or chan_phase is None:
                    # If not set, default to zero amplitude
                    i = np.zeros(num_samples)
                    q = np.zeros(num_samples)
                else:
                    frequency = chan_freq + nco_freq    # Frequency in Hz
                    amplitude = chan_amp   # Amplitude (normalized)
                    phase = np.deg2rad(chan_phase)  # Convert phase to radians

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

            # Handle averaging if requested
            if average:
                # Create mean structure for compatibility with network analysis
                mean_i = [np.mean([i_data[ch][s] for s in range(num_samples)]) for ch in range(channels_per_module)]
                mean_q = [np.mean([q_data[ch][s] for s in range(num_samples)]) for ch in range(channels_per_module)]
                
                # Return dictionary structure compatible with network analysis expectations
                # This will be converted to TuberResult on the client side
                return {
                    "mean": {
                        "i": mean_i,
                        "q": mean_q
                    },
                    "i": i_data,
                    "q": q_data,
                    "ts": t_list,
                    "flags": {"overrange": overrange, "overvoltage": overvoltage}
                }
            else:
                return {
                    "i": i_data,
                    "q": q_data,
                    "ts": t_list,
                    "flags": {"overrange": overrange, "overvoltage": overvoltage}
                }
    # LC resonance methods
    def generate_lc_resonances(self, num_resonances=50, f_start=1e9, f_end=2e9):
        """
        Generate simple LC resonances distributed across spectrum.
        
        Parameters
        ----------
        num_resonances : int
            Number of resonances to generate
        f_start : float
            Starting frequency in Hz
        f_end : float
            Ending frequency in Hz
        """
        self.lc_resonances = []
        
        # Generate random frequencies with minimum spacing
        min_spacing = (f_end - f_start) / (num_resonances * 2)
        frequencies = []
        
        while len(frequencies) < num_resonances:
            f = np.random.uniform(f_start, f_end)
            # Ensure minimum spacing
            if all(abs(f - existing) > min_spacing for existing in frequencies):
                frequencies.append(f)
        
        frequencies.sort()
        
        # Generate resonance parameters
        for f0 in frequencies:
            Q = np.random.uniform(5000, 20000)  # Q factor
            coupling = np.random.uniform(0.05, 0.2)  # Coupling coefficient
            
            self.lc_resonances.append({
                'f0': f0,
                'Q': Q,
                'coupling': coupling
            })
    
    def s21_lc_response(self, frequency, amplitude=1.0):
        """
        Calculate S21 response including amplitude dependence.
        
        Parameters
        ----------
        frequency : float
            Probe frequency in Hz
        amplitude : float
            Probe amplitude (normalized)
            
        Returns
        -------
        complex
            Complex S21 response
        """
        s21_linear = 1.0 + 0j
        
        for res in self.lc_resonances:
            f0 = res['f0']
            Q = res['Q']
            k = res['coupling']
            
            # Amplitude-dependent effects
            # 1. Q degradation at high power (saturation)
            Q_eff = Q / (1 + amplitude**2 * 0.1)
            
            # 2. Frequency shift at high power (nonlinearity)
            f0_eff = f0 * (1 + amplitude**2 * 0.001)
            
            # 3. Coupling changes
            k_eff = k * (1 + amplitude * 0.05)
            
            # Calculate resonance response
            delta = (frequency - f0_eff) / f0_eff
            denom = 1 + 2j * Q_eff * delta
            
            s21_res = 1 - k_eff / denom
            s21_linear *= s21_res
        
        # Add measurement noise
        noise_level = 0.001 * (1 + amplitude * 0.1)
        noise = noise_level * (np.random.randn() + 1j * np.random.randn())
        
        return s21_linear + noise
    
    def calculate_channel_response(self, module, channel):
        """
        Calculate complex S21 for current channel settings.
        
        Parameters
        ----------
        module : int
            Module number (1-based)
        channel : int
            Channel number (1-based)
            
        Returns
        -------
        complex
            Complex response value
        """
        # Get channel settings
        freq = self.frequencies.get((module, channel), 0)
        amp = self.amplitudes.get((module, channel), 0)
        phase_deg = self.phases.get((module, channel), 0)
        
        # Get NCO frequency
        nco_freq = self.get_nco_frequency(module=module) or 0
        
        if freq == 0 and amp == 0:
            return 0 + 0j
        
        # Total frequency = channel freq + NCO freq
        total_freq = freq + nco_freq
        
        if amp == 0:
            return 0 + 0j
        
        # Calculate S21 response
        s21 = self.s21_lc_response(total_freq, amp)
        
        # Apply commanded phase
        phase_rad = np.deg2rad(phase_deg)
        s21 *= np.exp(1j * phase_rad)
        
        # Scale by amplitude
        return s21 * amp
    
    async def start_udp_streaming(self, host='127.0.0.1', port=9876):
        """
        Start UDP packet streaming.
        
        Parameters
        ----------
        host : str
            Host address to stream to (for MockCRS, uses unicast)
        port : int
            UDP port number
        """
        # Use a simple flag check
        if not hasattr(self, '_udp_streaming_active'):
            self._udp_streaming_active = False
            
        if not self._udp_streaming_active:
            self._udp_streaming_active = True
            self.udp_streaming = True
            # For MockCRS, use unicast to localhost
            self.udp_thread = MockCRSUDPStreamer(self, '127.0.0.1', port)
            self.udp_thread.start()
            # Wait a moment to ensure thread is started
            await asyncio.sleep(0.1)
            # Return a simple boolean, not the object
            return True
        return False
    
    async def stop_udp_streaming(self):
        """Stop UDP packet streaming"""
        if hasattr(self, '_udp_streaming_active') and self._udp_streaming_active:
            self._udp_streaming_active = False
            self.udp_streaming = False
            if hasattr(self, 'udp_thread') and self.udp_thread:
                self.udp_thread.stop()
                self.udp_thread.join(timeout=1.0)
                self.udp_thread = None
            return True
        return False
    
    def get_udp_streaming_status(self):
        """Get UDP streaming status without returning the thread object"""
        return {
            "active": getattr(self, '_udp_streaming_active', False),
            "streaming": getattr(self, 'udp_streaming', False),
            "thread_alive": hasattr(self, 'udp_thread') and self.udp_thread and self.udp_thread.is_alive()
        }
    
    def clear_channels(self, module=None):
        """
        Clear all channel frequencies and amplitudes.
        
        Parameters
        ----------
        module : int, optional
            Module to clear. If None, clears all modules.
        """
        if module is None:
            # Clear all modules
            modules = [1, 2, 3, 4]
        else:
            modules = [module]
        
        for mod in modules:
            # Clear all channels for this module
            for channel in range(1, 1025):  # 1024 channels
                # Remove frequency entries
                self.frequencies.pop((mod, channel), None)
                # Remove amplitude entries
                self.amplitudes.pop((mod, channel), None)
                # Remove phase entries
                self.phases.pop((mod, channel), None)
    
    def set_cable_length(self, length, module):
        """
        Set cable length compensation (placeholder).
        
        Parameters
        ----------
        length : float
            Cable length
        module : int
            Module number
        """
        # This is a placeholder - in real hardware this would compensate
        # for phase delays based on cable length
        # Just validate the parameters
        assert isinstance(length, (int, float)), "Length must be a number"
        assert isinstance(module, int), "Module must be an integer"
        # Return None to ensure clean serialization
        return None
    
    def raise_periscope(self, module=1, channels="1", 
                        buf_size=5000, fps=30.0, 
                        density_dot=1, blocking=None):
        """
        Launch Periscope GUI for MockCRS.
        
        Special handling for localhost to avoid multicast issues.
        This is a synchronous method that launches the GUI.
        """
        # Import here to avoid circular imports
        from ..tools.periscope.__main__ import raise_periscope as base_raise_periscope
        from .. import streamer
        import asyncio
        
        # Check if we're using localhost
        if self.hostname and any(host in self.hostname for host in ["127.0.0.1", "localhost", "::1"]):
            # For localhost, we need to handle this specially
            original_get_local_ip = streamer.get_local_ip
            
            def mock_get_local_ip(crs_hostname):
                # For localhost, just return localhost
                if any(host in str(crs_hostname) for host in ["127.0.0.1", "localhost", "::1"]):
                    return "127.0.0.1"
                return original_get_local_ip(crs_hostname)
            
            try:
                # Temporarily replace the function
                streamer.get_local_ip = mock_get_local_ip
                
                # Call the base implementation
                # Note: base_raise_periscope is async, so we need to run it
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(base_raise_periscope(
                    self, module=module, channels=channels,
                    buf_size=buf_size, fps=fps, 
                    density_dot=density_dot, blocking=blocking
                ))
            finally:
                # Restore the original function
                streamer.get_local_ip = original_get_local_ip
                
            return result
        else:
            # For non-localhost, use the normal implementation
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(base_raise_periscope(
                self, module=module, channels=channels,
                buf_size=buf_size, fps=fps, 
                density_dot=density_dot, blocking=blocking
            ))


# Export MockCRS as CRS for the flavour system
CRS = MockCRS
