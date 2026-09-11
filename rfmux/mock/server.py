"""
Mock CRS Device - Server Process and YAML Hook.
Handles the mock server setup, process management, and request handling for Tuber.
"""
import asyncio
import json
import os
import socket
import time
import multiprocessing
import threading
from aiohttp import web
import atexit
import signal
import numpy as np

# Import MockCRS from the crs module within this package
from .crs import ServerMockCRS
# Import BaseCRS for type hinting or direct use if necessary
from ..core.schema import CRS as BaseCRS

# DO NOT import algorithms on the server side
# Algorithms should only run on the client side

mp_ctx = multiprocessing.get_context()

# Interpreter exit is a fallback for sessions that were not explicitly closed.
# Stop the fleet concurrently so shutdown time does not grow with its size.
_server_processes: list["ServerProcess"] = []

# Grace period for the whole fleet to exit before SIGKILL; one deadline, not
# one per process.
_SHUTDOWN_GRACE_S = 2.0


def _shutdown_servers(processes: list) -> None:
    """Stop the selected servers against a shared deadline."""
    alive = [p for p in processes if p.is_alive()]
    for p in processes:
        if p in _server_processes and not p.is_alive():
            _server_processes.remove(p)
    if not alive:
        return

    print(f"[MockCRS] Shutting down {len(alive)} server process(es)...")

    # Ask them all to stop before waiting on any of them; the grace periods
    # then run concurrently instead of stacking up.
    for p in alive:
        try:
            p.terminate()
        except Exception:
            pass

    deadline = time.monotonic() + _SHUTDOWN_GRACE_S
    for p in alive:
        try:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            pass

    # Anything still up after the shared grace period gets killed, again
    # all-then-wait.
    stubborn = [p for p in alive if p.is_alive()]
    if stubborn:
        print(f"[MockCRS] Force killing {len(stubborn)} server process(es)...")
        for p in stubborn:
            try:
                p.kill()
            except Exception:
                pass
        deadline = time.monotonic() + _SHUTDOWN_GRACE_S
        for p in stubborn:
            try:
                p.join(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                pass

    for p in alive:
        if not p.is_alive() and p in _server_processes:
            _server_processes.remove(p)
    if any(p.is_alive() for p in alive):
        raise RuntimeError("Mock CRS server did not exit after kill")
    print("[MockCRS] Server shutdown complete")


def _shutdown_all_servers() -> None:
    _shutdown_servers(list(_server_processes))


atexit.register(_shutdown_all_servers)


def yaml_hook(hwm):
    """Start one mock server and attach its cleanup to the hardware map."""
    # Build the models in the child: their locks cannot be pickled for spawn.
    model_configs = {}

    # Find all CRS objects in the database and patch up their hostnames to
    # something local.
    sockets = []
    p = None
    try:
        for crs in hwm.query(BaseCRS):

            # Create a socket to be shared with the server process.
            s = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            sockets.append(s)
            s.bind(("localhost", 0))
            (hostname, port) = s.getsockname()

            crs.hostname = f"{hostname}:{port}"
            # Store configuration for MockCRS instantiation in subprocess
            model_configs[port] = {
                'serial': crs.serial if crs.serial else ("%05d" % port),
                'slot': crs.slot if crs.crate else None,
                'crate': crs.crate.serial if crs.crate else None,
            }

        hwm.commit()

        if not sockets:
            return
        ready = mp_ctx.Semaphore(0)
        p = ServerProcess(sockets=sockets, model_configs=model_configs,
                          lock=ready)
        p.start()
        _server_processes.append(p)
        hwm().on_close(lambda: _shutdown_servers([p]))
        while not ready.acquire(timeout=0.1):
            if not p.is_alive():
                raise RuntimeError(
                    f"Mock CRS server failed to start (exit code {p.exitcode})")
    except BaseException:
        if p is not None and p.pid is not None:
            _shutdown_servers([p])
        raise
    finally:
        for s in sockets:
            s.close()


class ServerProcess(mp_ctx.Process):
    """Local RPC server owned by the process that loaded the hardware map."""

    daemon = True

    def __init__(self, sockets, model_configs, lock):
        self.sockets = sockets
        self.model_configs = model_configs
        self.lock = lock
        super().__init__()

    def run(self):
        # Undo any CPU pinning inherited across fork().  Periscope pins
        # its GUI thread to a single core (periscope/utils.py
        # pin_current_thread_to_core) BEFORE this process is forked, and
        # fork inherits the affinity mask — which would confine the
        # whole simulation, streamer thread and every numba worker, to
        # one core.  The pin is meant for the Qt event loop, not for a
        # compute-bound child.
        try:
            n_cpus = os.cpu_count() or 1
            os.sched_setaffinity(0, set(range(n_cpus)))
        except (AttributeError, OSError):
            pass  # not Linux, or not permitted — harmless

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        shutdown_event = asyncio.Event()
        finished = threading.Event()
        parent = multiprocessing.parent_process()

        def watch_parent() -> None:
            # getppid also detects death when forked siblings inherited a
            # copy of multiprocessing's parent-sentinel pipe.
            while not finished.wait(0.2):
                if (not parent.is_alive()
                        or (os.name == "posix" and os.getppid() != parent.pid)):
                    loop.call_soon_threadsafe(shutdown_event.set)
                    if not finished.wait(2 * _SHUTDOWN_GRACE_S):
                        os._exit(1)  # a blocked RPC must not orphan this server
                    return

        threading.Thread(target=watch_parent, daemon=True,
                         name="mock-parent-watch").start()
        self.models = {}
        runners = []
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, shutdown_event.set)
                except (ValueError, NotImplementedError):
                    pass

            for port, config in self.model_configs.items():
                self.models[port] = ServerMockCRS(**config)

            app = web.Application()
            app.add_routes([web.post("/tuber", self.post_handler)])
            runner = web.AppRunner(app, shutdown_timeout=_SHUTDOWN_GRACE_S)
            runners.append(runner)
            loop.run_until_complete(runner.setup())
            for s in self.sockets:
                site = web.SockSite(runner, s)
                loop.run_until_complete(site.start())

            self.lock.release()
            loop.run_until_complete(shutdown_event.wait())
        finally:
            try:
                for model in self.models.values():
                    try:
                        loop.run_until_complete(model.stop_udp_streaming())
                    except Exception as exc:
                        print(f"[MockCRS Server] Error stopping stream: {exc}")
                for runner in runners:
                    loop.run_until_complete(runner.cleanup())
            finally:
                finished.set()
                for s in self.sockets:
                    s.close()
                loop.close()

    async def post_handler(self, request):
        port = request.url.port
        model = self.models[port]  # This is an instance of ServerMockCRS

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
            print(f"Error in post_handler: {e}")
            import traceback
            traceback.print_exc()
            raise e
        finally:
            model._thread_lock_release()

    async def __single_handler(self, model, request):
        """Handle a single Tuber request"""

        # Handle resolve requests first
        if request.get("resolve", False):
            # Return object metadata
            return await self.__single_handler(model, {"object": request.get("object", "Dfmux")})

        if "method" in request and request["method"] is not None:
            method_name = request["method"]
            object_name = request.get("object")

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

            # Block access to private attributes (server-side implementation details)
            if prop_name.startswith('_'):
                return {"error": {"message": f"Property '{prop_name}' is private and not accessible via RPC"}}

            if hasattr(model, prop_name):
                prop = getattr(model, prop_name)
                if callable(prop):
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
                    return await self.__single_handler(model, {
                        "method": method_name,
                        "args": request.get("args", []),
                        "kwargs": request.get("kwargs", {})
                    })

            # Normal object metadata request
            # We need to provide some metadata to TuberObject so it can populate
            # properties and methods on the client-side object.
            # Block all private attributes (single underscore) from RPC access
            # These are server-side implementation details
            illegal_prefixes = ("_",)
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
                "resonator_model", "udp_manager", "timestamp",
                "_config_lock"
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
    else:
        return obj
