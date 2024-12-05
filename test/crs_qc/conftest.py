import sys
sys.path.append('.')

import pytest
import os
import json
from datetime import datetime
import time
import rfmux
import asyncio
import inspect
import pytest_asyncio
from report_generator import generate_report_from_data

def run_async_if_needed(func):
    """Run the given function synchronously, but if it's a coroutine, run it asynchronously."""
    def wrapper(*args, **kwargs):
        if inspect.iscoroutinefunction(func):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.ensure_future(func(*args, **kwargs))
            else:
                return loop.run_until_complete(func(*args, **kwargs))
        else:
            return func(*args, **kwargs)
    return wrapper

class SyncContextWrapper:
    def __init__(self, async_context):
        self.async_context = async_context

    def __enter__(self):
        loop = asyncio.get_event_loop()
        self.context = loop.run_until_complete(self.async_context.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_context.__aexit__(exc_type, exc_val, exc_tb))

    def __call__(self, *args, **kwargs):
        # Automatically await the __call__ method
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.context(*args, **kwargs))

    def __getattr__(self, name):
        # Delegate attribute access to the underlying context
        return getattr(self.context, name)


# Patching function for d object
def patch_tuber_context(d):
    original_tuber_context = d.tuber_context

    def patched_tuber_context(*args, **kwargs):
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # If we're in an async context, return the original async context
            return original_tuber_context(*args, **kwargs)
        except RuntimeError:
            # If we're in a sync context, return the SyncContextWrapper
            return SyncContextWrapper(original_tuber_context(*args, **kwargs))

    # Override the tuber_context method with the patched version
    setattr(d, 'tuber_context', patched_tuber_context)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    session.user_test_highbank = input("Would you like to run the test on both low (1-4) and high (5-8) banks? Enter Y/N  ")
    if session.user_test_highbank == 'Y':
        session.test_highbank = True
        session.num_runs = 2
        wait_for_action("Connect the modules 1-8 in loopback.(DAC#--ADC#): (1--1), (2--2), etc. Don't cross connect")
    else:
        session.test_highbank = False
        session.num_runs = 1
        wait_for_action("Connect the modules 1-4 in loopback.(DAC#--ADC#): (1--1), (2--2), etc. Don't cross connect")

    session.results = {
        'serial': session.config.getoption("--serial"),
        'tests': []
    }
    session.summary_results = {
        'serial': session.config.getoption("--serial"),
        'summary': {}
    }
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    day = datetime.now().strftime('%Y%m%d')
    serial = session.config.getoption("--serial")
    session.results_dir = f'CRS_test_results/{day}/CRS_test_{serial}_{timestamp}'
    os.makedirs(session.results_dir, exist_ok=True)


@pytest.fixture(scope="session")
def serial(pytestconfig):
    serial = pytestconfig.getoption("--serial")
    if serial is None:
        pytest.fail("--serial number not provided! Can't talk to the CRS board.")
    return serial


@pytest.fixture(scope="session")
def d(pytestconfig):
    serial = pytestconfig.getoption("--serial")
    if serial is None:
        pytest.fail("--serial number not provided! Can't talk to the CRS board.")

    hwm = rfmux.load_session(f'!HardwareMap [ !CRS {{serial: "{serial}"}} ]')
    d = hwm.query(rfmux.CRS).one()

    # Wrap the 'resolve' method with run_async_if_needed since we need it to get the other methods
    setattr(d, "resolve", run_async_if_needed(getattr(d, "resolve")) ) # Wrap all methods with run_async_if_needed

    # Resolve the device to get all the methods    
    d.resolve()

    # Wrap all callable methods, async or not, with run_async_if_needed
    for attr in dir(d):
        if not attr.startswith('__'):  # Skip internal methods
            method = getattr(d, attr)
            if callable(method):  # Check if it's callable
                setattr(d, attr, run_async_if_needed(method))  # Wrap all methods with run_async_if_needed

    # Patch tuber_context to seamlessly switch between async and sync contexts
    patch_tuber_context(d)

    d.set_timestamp_port(d.TIMESTAMP_PORT.TEST)

    return d  # Return the wrapped 'd' object


@pytest.fixture(scope="session")
def user_test_highbank(request):
    return request.session.user_test_highbank

@pytest.fixture(scope="session")
def test_highbank(request):
    return request.session.test_highbank

@pytest.fixture(scope="session")
def num_runs(request):
    return request.session.num_runs

@pytest.fixture(scope="session")
def results_dir(request):
    return request.session.results_dir

@pytest.fixture(scope="session")
def results(request):
    return request.session.results

@pytest.fixture(scope="session")
def summary_results(request):
    return request.session.summary_results

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session):
    results_dir = session.results_dir
    results = session.results
    summary_results = session.summary_results
    results_file = save_results(results, results_dir)
    summary_file = save_summary_results(summary_results, results_dir)
    serial = results['serial']
    generate_report_from_data(serial, results_file, results_dir, summary_file)

def save_results(results, results_dir):
    results_file = os.path.join(results_dir, 'results.json')
    with open(results_file, 'w') as file:
        json.dump(results, file, indent=4)
    print(f"Results saved to '{results_file}'")  # Debug print
    return results_file

def save_summary_results(summary_results, results_dir):
    summary_file = os.path.join(results_dir, 'summary_results.json')
    with open(summary_file, 'w') as file:
        json.dump(summary_results, file, indent=4)
    print(f"Summary results saved to '{summary_file}'")
    return summary_file

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if item.name == "test_dac_mixmodes":
        wait_for_action("Please attach the filters to the board before proceeding with the test_dac_mixmodes test.\
                              \n1st Nyquist filter to ADC#1, \n2nd Nyquist filter to ADC#2,\n3rd Nyquist filter to ADC#3 ")

@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    if item.name == "test_dac_mixmodes":
        wait_for_action("Please remove the filters from the board after completing the test_dac_mixmodes test.")

def wait_for_action(action):
    input(f"{action}\nPress Enter to continue...")
