"""
Mock CRS Device - Server Process and YAML Hook.
Handles the mock server setup, process management, and request handling for Tuber.
"""
import asyncio
import json
import socket
import multiprocessing
from aiohttp import web
import atexit
import signal
import numpy as np

# Import MockCRS from its new location
from .mock_crs_core import MockCRS
# Import BaseCRS for type hinting or direct use if necessary
from .schema import CRS as BaseCRS

# DO NOT import algorithms on the server side
# Algorithms should only run on the client side

mp_ctx = multiprocessing.get_context()
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
    for d in hwm.query(BaseCRS): # Query for BaseCRS, as MockCRS might not be in DB yet

        # Create a socket to be shared with the server process.
        s = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        s.bind(("localhost", 0))
        (hostname, port) = s.getsockname()

        sockets.append(s)
        d.hostname = f"{hostname}:{port}"
        # Instantiate MockCRS from mock_crs_core
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

    def cleanup_server_process():
        """Properly shutdown the MockCRS server process"""
        if p.is_alive():
            print("[MockCRS] Shutting down server process...")
            # First try graceful termination
            p.terminate()
            try:
                # Wait up to 5 seconds for graceful shutdown
                p.join(timeout=2.0)
            except:
                pass
            
            # If still alive, force kill
            if p.is_alive():
                print("[MockCRS] Force killing server process...")
                p.kill()
                try:
                    p.join(timeout=2.0)
                except:
                    pass
            
            print("[MockCRS] Server process shutdown complete")

    atexit.register(cleanup_server_process)

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
        asyncio.set_event_loop(loop)
        
        # Set up signal handlers for graceful shutdown
        shutdown_event = asyncio.Event()
        
        def signal_handler():
            print("[MockCRS Server] Received shutdown signal")
            shutdown_event.set()
        
        # Register signal handlers
        try:
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
            loop.add_signal_handler(signal.SIGINT, signal_handler)
        except (ValueError, NotImplementedError):
            # Signal handling may not be available in all contexts
            pass
        
        # Do NOT load algorithms on server side - they should only run on client

        app = web.Application()
        app.add_routes([web.post("/tuber", self.post_handler)])

        runners = []
        sites = []
        
        for s in self.sockets:
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())
            runners.append(runner)
            site = web.SockSite(runner, s)
            loop.run_until_complete(site.start())
            sites.append(site)

        self.lock.release()
        
        # Wait for shutdown signal instead of running forever
        try:
            loop.run_until_complete(shutdown_event.wait())
        except KeyboardInterrupt:
            print("[MockCRS Server] Received KeyboardInterrupt")
        
        # Clean shutdown
        print("[MockCRS Server] Shutting down...")
        
        # Stop UDP streaming for all models
        for model in self.models.values():
            if hasattr(model, 'udp_manager') and model.udp_manager:
                try:
                    loop.run_until_complete(model.udp_manager.stop_udp_streaming())
                except Exception as e:
                    print(f"[MockCRS Server] Error stopping UDP streaming: {e}")
        
        # Clean up web sites and runners
        for site in sites:
            try:
                loop.run_until_complete(site.stop())
            except Exception as e:
                print(f"[MockCRS Server] Error stopping site: {e}")
        
        for runner in runners:
            try:
                loop.run_until_complete(runner.cleanup())
            except Exception as e:
                print(f"[MockCRS Server] Error cleaning up runner: {e}")
        
        loop.close()
        print("[MockCRS Server] Shutdown complete")

    async def post_handler(self, request):
        port = request.url.port
        model = self.models[port] # This is an instance of MockCRS

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
            # It's better to log the exception on the server and return a generic error
            # to the client, or a specific error structure Tuber expects.
            print(f"Error in post_handler: {e}") # Basic logging
            import traceback
            traceback.print_exc()
            # Re-raise to let aiohttp handle it or return a structured error
            # For now, re-raising to see default aiohttp behavior
            raise e
        finally:
            model._thread_lock_release()

    async def __single_handler(self, model, request):
        """Handle a single Tuber request"""
        
        # Debug logging
        #print(f"[DEBUG] Tuber request: {request}")
        
        # Handle resolve requests first
        if request.get("resolve", False):
            # Return object metadata
            return await self.__single_handler(model, {"object": request.get("object", "Dfmux")})

        if "method" in request and request["method"] is not None:
            method_name = request["method"]
            object_name = request.get("object")
            
            # Debug log
            #print(f"[DEBUG] Method request: method={method_name}, object={object_name}")
            
            # List of algorithm methods that should NOT be executed on the server
            # These should only run on the client side
            algorithm_methods = {
                'take_netanal', 'multisweep'
                # Add other algorithm names here as needed
            }
            
            # Check if this is an algorithm method
            if method_name in algorithm_methods:
                return {"error": {"message": f"Method '{method_name}' is an algorithm that should run on the client side, not the server"}}
            
            # Look for the method on the instance
            m = None
            if hasattr(model, method_name):
                m = getattr(model, method_name)
            
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
                import traceback
                print(f"Error executing method {method_name}: {oops}")
                traceback.print_exc()
                e = {"message": "%s: %s" % (oops.__class__.__name__, str(oops))}
            return {"result": r, "error": e}

        elif "property" in request:
            prop_name = request["property"]
            if hasattr(model, prop_name):
                prop = getattr(model, prop_name)
                if callable(prop):
                    # Debug logging for method property requests
                    #print(f"[DEBUG] Property request for callable '{prop_name}'")
                    
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
                    #print(f"[DEBUG] Redirecting object request {obj_name} to method call")
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
                "_context_class", "reconstruct", "to_query",
                # Exclude algorithms that should run on client side
                "take_netanal", "multisweep"
            }
            exclude_properties = {
                "metadata", "registry", "_sa_class_manager", "_sa_instance_state",
                "_sa_registry", "is_container", "modules", "module", "crate",
                "hwm", "tuber_hostname", "keys", "values", "items",
                # Exclude dictionary attributes that cause issues with Tuber resolution
                "frequencies", "amplitudes", "phases", "tuning_results",
                "temperature_sensors", "rails", "nco_frequencies",
                "adc_attenuators", "dac_scales", "adc_autocal",
                "adc_calibration_mode", "adc_calibration_coefficients",
                "nyquist_zones", "hmc7044_registers",
                # Exclude helper objects that can't be serialized
                "resonator_model", "udp_manager", "timestamp"
            }
            
            # Special enum properties that should be included even though they're callables
            special_enum_properties = {
                "TIMESTAMP_PORT", "CLOCKSOURCE", "UNITS", "TARGET"
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
                    # Special handling for enum properties
                    if name in special_enum_properties:
                        properties.append(name)
                    elif callable(attr):
                        # Only include methods that are not coroutines from parent classes
                        if not (asyncio.iscoroutinefunction(attr) and 
                                name in ['tuber_resolve', 'resolve']):
                            methods.append(name)
                    else:
                        # Double-check it's not in exclude list and not a dict
                        if name not in exclude_properties and not isinstance(attr, dict):
                            properties.append(name)
                except:
                    # Skip attributes that can't be accessed
                    pass

            return {
                "result": {
                    "name": "TuberObject", # Or model.__class__.__name__ if more specific
                    "summary": "",
                    "explanation": "",
                    "properties": sorted(properties),
                    "methods": sorted(methods),
                }
            }

        else:
            return {"error": "Didn't know what to do!"}


def convert_to_serializable(obj):
    """Recursively convert NumPy arrays to JSON-serializable formats."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert numpy numbers to Python native types
        return obj.item()
    elif obj.__class__.__name__ == "TuberResult":
    # flatten it if it has a to_dict() or __dict__
        if hasattr(obj, "to_dict"):
            return convert_to_serializable(obj.to_dict())
        elif hasattr(obj, "__dict__"):
            return convert_to_serializable(obj.__dict__)
        else:
            return str(obj)
    # Add other non-serializable types here if needed
    else:
        return obj
