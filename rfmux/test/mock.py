# TODO: Implement this

# """
# Mock CRS, for performance testing and debugging at scale.

# This !flavour of CRS stuff is activated by using

#     !flavour rfmux.test.mock

# ...in your hardware map. After loading the HWM, a local webserver is launched
# and CRS HTTP requests are redirected to it.

# There is only a cursory attempt at mimicing the CRS API -- no signal-path
# simulation is included, and the API is missing big chunks. Please feel free
# to expand it if it is useful.
# """

# import asyncio
# import multiprocessing
# import socket
# import json
# import time
# import atexit

# from aiohttp import web

# from ..core.crs import CRS

# mp_ctx = multiprocessing.get_context("fork")

# def yaml_hook(hwm):
#     """Patch up the HWM using mock CRSes instead of real ones.

#     To do so, we alter the hostname associated with the CRS objects
#     in the HWM and redirect HTTP requests to a local server. Each CRS
#     gets a distinct port, which is used to route requests to a distinct
#     model class.
#     """

#     # These objects are hardware models of the crses, indexed by the port number
#     # used to connect with them over HTTP.
#     models = {}

#     # Find all CRS objects in the database and patch up their hostnames to
#     # something local.
#     sockets = []
#     for d in hwm.query(CRS):

#         # Create a socket to be shared with the server process.
#         s = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
#         s.bind(("localhost", 0))
#         (hostname, port) = s.getsockname()

#         sockets.append(s)
#         d.hostname = f"{hostname}:{port}"
#         models[port] = CRSModel(
#             serial=d.serial if d.serial else ("%05d" % port),
#             slot=d.slot if d.crate else None,
#             crate=d.crate.serial if d.crate else None,
#         )

#     hwm.commit()

#     l = mp_ctx.Semaphore(0)

#     p = ServerProcess(sockets=sockets, models=models, lock=l)
#     p.start()
#     l.acquire()

#     atexit.register(p.terminate)

#     # In the client process, we do not need the sockets -- in fact, we don't
#     # want a reference hanging around.
#     for s in sockets:
#         s.close()


# # Start up a web server. This is a distinct process, so COW semantics.
# class ServerProcess(mp_ctx.Process):
#     daemon = True

#     def __init__(self, sockets, models, lock):
#         self.sockets = sockets
#         self.models = models
#         self.lock = lock
#         super().__init__()

#     def run(self):
#         loop = asyncio.new_event_loop()

#         app = web.Application()
#         app.add_routes([web.post("/tuber", self.post_handler)])

#         for s in self.sockets:
#             runner = web.AppRunner(app)
#             loop.run_until_complete(runner.setup())
#             site = web.SockSite(runner, s)
#             loop.run_until_complete(site.start())

#         self.lock.release()
#         loop.run_forever()

#     async def post_handler(self, request):
#         port = request.url.port
#         model = self.models[port]

#         body = await request.text()

#         await model._thread_lock_acquire()

#         try:
#             request = json.loads(body)
#             model._num_tuber_calls += 1

#             if isinstance(request, list):
#                 r = [await self.__single_handler(model, r) for r in request]
#             elif isinstance(request, dict):
#                 r = await self.__single_handler(model, request)
#             else:
#                 r = {"error": "Didn't know what to do!"}

#             return web.Response(text=json.dumps(r))

#         except Exception as e:
#             raise e
#         finally:
#             model._thread_lock_release()

#     async def __single_handler(self, model, request):
#         """Handle a single Tuber request"""

#         if "method" in request and hasattr(model, request["method"]):
#             m = getattr(model, request["method"])
#             a = request.get("args", [])
#             k = request.get("kwargs", {})
#             r = e = None
#             try:
#                 if asyncio.iscoroutine(m):
#                     r = await m(*a, **k)
#                 else:
#                     r = m(*a, **k)
#             except Exception as oops:
#                 e = {"message": "%s: %s" % (oops.__class__.__name__, str(oops))}
#             return {"result": r, "error": e}

#         elif "property" in request and hasattr(model, request["property"]):
#             prop = getattr(model, request["property"])
#             if callable(prop):
#                 return {
#                     "result": {
#                         "name": request["property"],
#                         "args": [],
#                         "explanation": "(...)",
#                     }
#                 }
#             else:
#                 return {"result": getattr(model, request["property"])}

#         elif "object" in request:
#             # We need to provide some metadata to TuberObject so it can populate
#             # properties and methods on the client-side object.
#             illegal_prefixes = ("__", "_CRSModel__", "_thread_lock_")
#             names = set(
#                 filter(lambda s: not s.startswith(illegal_prefixes), dir(model))
#             )
#             methods = list(filter(lambda n: callable(getattr(model, n)), names))
#             properties = list(
#                 filter(lambda n: not callable(getattr(model, n)), names)
#             )

#             return {
#                 "result": {
#                     "name": "TuberObject",
#                     "summary": "",
#                     "explanation": "",
#                     "properties": properties,
#                     "methods": methods,
#                 }
#             }

#         else:
#             return {"error": "Didn't know what to do!"}


# class CRSModel:
#     _num_tuber_calls = 0

#     def __init__(self, serial, slot=None, crate=None):
#         self.__serial = serial
#         self.__slot = slot
#         self.__crate = crate

#         # To simulate limited number of simultaneous board connections
#         self.__thread_lock = asyncio.Semaphore(value=4)

#         # FIR stage
#         self.__fir_stage = 5

#         # Motherboard temperatures
#         self.__mb_temps = {
#             "MOTHERBOARD_TEMPERATURE_POWER": 0,
#             "MOTHERBOARD_TEMPERATURE_ARM": 0,
#             "MOTHERBOARD_TEMPERATURE_FPGA": 0,
#             "MOTHERBOARD_TEMPERATURE_PHY": 0,
#         }

#         # Mezzanine temperatures
#         self.__mezz_temps = [0]

#         # Mezzanine power
#         self.__mezz_power = [False]

#         # Mezzanine serials
#         self.__mezz_serial = ["0001"]

#         # Tuning results
#         self.__tuning_results = [[[None] * 1024 for i in range(2)] for j in range(1)]

#         # Frequency
#         self.__frequencies = [[[0.0] * 1024 for i in range(2)] for j in range(1)]

#         # Amplitude
#         self.__amplitudes = {
#             k: [[[0.0] * 1024 for i in range(2)] for j in range(1)]
#             for k in ["carrier", "nuller"]
#         }

#         # Phase
#         self.__phases = {
#             k: [[[0.0] * 1024 for i in range(2)] for j in range(1)]
#             for k in ["carrier", "nuller", "demod"]
#         }

#         # timestamp port
#         self.__timestamp_port = self.TIMESTAMP_PORT["TEST"]

#         # routing
#         self.__dmfd_routing = [[['ADC'] for i in range(2)] for j in range(1)]

#     async def _thread_lock_acquire(self):
#         await self.__thread_lock.acquire()

#     def _thread_lock_release(self):
#         self.__thread_lock.release()

#     NUM_MEZZANINES = 1

#     UNITS = {
#         "HZ": "Hz",
#         "RAW": "RAW",
#         "VOLTS": "Volts",
#         "AMPS": "Amps",
#         "WATTS": "Watts",
#         "DAC_COUNTS": "DAC Counts",
#         "ADC_COUNTS": "ADC Counts",
#         "NORMALIZED": "Normalized",
#         "DEGREES": "Degrees",
#         "RADIANS": "Radians",
#         "OHMS": "Ohms",
#     }

#     TARGET = {
#         "CARRIER": "carrier",
#         "NULLER": "nuller",
#         "DEMOD": "demod",
#         "RAWDUMP_COUNTER": "rawdump_counter",
#     }

#     TEMPERATURE_SENSOR = {
#         "MB_POWER": "MOTHERBOARD_TEMPERATURE_POWER",
#         "MB_ARM": "MOTHERBOARD_TEMPERATURE_ARM",
#         "MB_FPGA": "MOTHERBOARD_TEMPERATURE_FPGA",
#         "MB_PHY": "MOTHERBOARD_TEMPERATURE_PHY",
#     }

#     RAIL = {
#         "MB_VCC3V3": "MOTHERBOARD_RAIL_VCC3V3",
#         "MB_VCC12V0": "MOTHERBOARD_RAIL_VCC12V0",
#         "MB_VCC5V5": "MOTHERBOARD_RAIL_VCC5V5",
#         "MB_VCC1V0_GTX": "MOTHERBOARD_RAIL_VCC1V0_GTX",
#         "MB_VCC1V0": "MOTHERBOARD_RAIL_VCC1V0",
#         "MB_VCC1V2": "MOTHERBOARD_RAIL_VCC1V2",
#         "MB_VCC1V5": "MOTHERBOARD_RAIL_VCC1V5",
#         "MB_VCC1V8": "MOTHERBOARD_RAIL_VCC1V8",
#         "MB_VADJ": "MOTHERBOARD_RAIL_VADJ",
#     }

#     CLOCK_SOURCE = {
#         "XTAL": "CLOCK_SOURCE_XTAL",
#         "SMA": "CLOCK_SOURCE_SMA",
#         "BP": "CLOCK_SOURCE_BP",
#     }

#     TIMESTAMP_PORT = {
#         "BACKPLANE": "BACKPLANE",
#         "SMA_B": "SMA_B",
#         "TEST": "TEST",
#         "GND": "GND",
#     }

#     ROUTING = {
#         "ADC": "routing_adc",
#         "CARRIER": "routing_car",
#         "NULLER": "routing_nul",
#         "SUM": "routing_sum",
#         "TSCAL": "routing_tscal",
#     }

#     STREAMER_TYPE = {"READOUT": "streamer_readout"}

#     def get_motherboard_serial(self):
#         return self.__serial

#     def get_backplane_slot(self):
#         if self.__slot is None:
#             raise ValueError("This crs is not in a crate")
#         return self.__slot

#     def _get_backplane_serial(self):
#         if self.__crate is None:
#             raise ValueError("This crs is not in a crate")
#         return self.__crate

#     # timestamp port
#     def set_timestamp_port(self, port):
#         self.__timestamp_port = port

#     def get_timestamp_port(self):
#         return self.__timestamp_port

#     def get_timestamp(self):
#         t = time.gmtime()
#         return {
#             "y": t.tm_year,
#             "d": t.tm_yday,
#             "h": t.tm_hour,
#             "m": t.tm_min,
#             "s": t.tm_sec,
#             "c": 0,
#             "ss": 0,
#             "sbs": 0,
#         }

#     # FIR stage
#     def set_fir_stage(self, fir_stage):
#         self.__fir_stage = fir_stage

#     def get_fir_stage(self):
#         return self.__fir_stage

#     # Motherboard temperatures
#     def get_motherboard_temperature(self, sensor):
#         return self.__mb_temps[sensor]

#     # Mezzanine temperatures
#     def get_mezzanine_temperature(self, mezzanine):
#         return self.__mezz_temps[mezzanine - 1]

#     # Mezzanine power
#     def is_mezzanine_present(self, mezzanine):
#         return True

#     def _get_mezzanine_serial(self, mezzanine):
#         return self.__mezz_serial[mezzanine - 1]

#     def set_mezzanine_power(self, power, mezzanine):
#         self.__mezz_power[mezzanine - 1] = power

#     def get_mezzanine_power(self, mezzanine):
#         return self.__mezz_power[mezzanine - 1]

#     # routing
#     def set_dmfd_routing(self, target, module, mezzanine):
#         assert target in self.ROUTING
#         self.__dmfd_routing[mezzanine - 1][module - 1] = target

#     def get_dmfd_routing(self, module, mezzanine):
#         return self.__dmfd_routing[mezzanine - 1][module - 1]

#     # Tuning results
#     def get_tuning_result(self, channel, module, mezzanine=1):
#         assert channel < 1025, "Invalid channel %d" % channel
#         assert module < 3, "Invalid module %d" % module
#         return self.__tuning_results[mezzanine - 1][module - 1][channel - 1]

#     def set_tuning_result(self, value, channel, module, mezzanine=1):
#         assert channel < 1025, "Invalid channel %d" % channel
#         assert module < 3, "Invalid module %d" % module
#         self.__tuning_results[mezzanine - 1][module - 1][channel - 1] = value

#     # get_samples
#     async def get_samples(self, num_samples, channel=None, module=0, mezzanine=0):

#         if channel is None:
#             channel = list(range(1024))

#         assert module in [1, 2]
#         assert mezzanine in [1]

#         # This call normally takes some time
#         await asyncio.sleep(num_samples / 150.0)

#         if isinstance(channel, list):
#             return {
#                 "i": [[0] * num_samples] * len(channel),
#                 "q": [[0] * num_samples] * len(channel),
#                 "ts": [{}] * num_samples,
#             }
#         else:
#             return {
#                 "i": [0] * num_samples,
#                 "q": [0] * num_samples,
#                 "ts": [{}] * num_samples,
#             }

#     async def get_fast_samples(
#         self,
#         num_samples=None,
#         units="ADC_COUNTS",
#         average=False,
#         target="DEMOD",
#         module=0,
#         mezzanine=0,
#     ):

#         if num_samples is None:
#             num_samples = 333333

#         assert module in [1, 2]
#         assert mezzanine in [1]

#         return [0] * num_samples

#     # Frequency
#     def get_frequency(self, units, channel, module, mezzanine):
#         assert units.lower() == "hz"
#         return self.__frequencies[mezzanine - 1][module - 1][channel - 1]

#     def set_frequency(self, frequency, units, channel, module, mezzanine):
#         assert units.lower() == "hz"
#         self.__frequencies[mezzanine - 1][module - 1][channel - 1] = frequency

#     # Amplitude
#     def get_amplitude(self, units, target, channel, module, mezzanine):
#         return self.__amplitudes[target.lower()][mezzanine - 1][module - 1][channel - 1]

#     def set_amplitude(
#         self, amplitude, units, target, channel, module, mezzanine, steps=1
#     ):
#         self.__amplitudes[target.lower()][mezzanine - 1][module - 1][
#             channel - 1
#         ] = amplitude

#     # Phase
#     def get_phase(self, units, target, channel, module, mezzanine):
#         return self.__phases[target.lower()][mezzanine - 1][module - 1][channel - 1]

#     def set_phase(self, phase, units, target, channel, module, mezzanine, steps=1):
#         self.__phases[target.lower()][mezzanine - 1][module - 1][channel - 1] = phase

#     def _clear_num_tuber_calls(self):
#         self._num_tuber_calls = 0

#     def _get_num_tuber_calls(self):
#         return self._num_tuber_calls

#     def _dump_housekeeping(self):
#         mezzanines = [
#             {
#                 "present": self.is_mezzanine_present(mezz),
#                 "power": self.get_mezzanine_power(mezz),
#                 "temperature": self.get_mezzanine_temperature(mezz),
#                 "modules": [
#                     {
#                         "routing": self.get_dmfd_routing(mod, mezz),
#                         "channels": [
#                             {
#                                 "frequency": self.get_frequency("Hz", chan, mod, mezz),
#                                 "carrier_amplitude": self.get_amplitude("normalized", "carrier", chan, mod, mezz),
#                                 "nuller_amplitude": self.get_amplitude("normalized", "nuller", chan, mod, mezz),
#                                 "carrier_phase": self.get_phase("degrees", "carrier", chan, mod, mezz),
#                                 "nuller_phase": self.get_phase("degrees", "nuller", chan, mod, mezz),
#                                 "demod_phase": self.get_phase("degrees", "demod", chan, mod, mezz),
#                                 "tuning": self.get_tuning_result(chan, mod, mezz),
#                             }
#                             for chan in range(1, 1025)
#                         ],
#                     }
#                     for mod in range(1, 3)
#                 ],
#             }
#             for mezz in range(1, 2)
#         ]

#         return {
#             "firmware_name": "mkids_iceboard_mock",
#             "firmware_version": "1.0.0",
#             "fir_stage": self.get_fir_stage(),
#             "serial": self.get_motherboard_serial(),
#             "temperatures" : {
#                 k: self.get_motherboard_temperature(v) for k, v in self.TEMPERATURE_SENSOR.items()
#             },
#             "voltages": {k: 0 for k in self.RAIL},
#             "currents": {k: 0 for k in self.RAIL},
#             "mezzanines": mezzanines,
#             "timestamp_port": self.get_timestamp_port(),
#             "timestamp": self.get_timestamp(),
#         }
