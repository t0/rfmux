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


@pytest_asyncio.fixture(scope="function")
async def d(pytestconfig):
    serial = pytestconfig.getoption("--serial")
    if serial is None:
        pytest.fail("--serial number not provided! Can't talk to the CRS board.")

    hwm = rfmux.load_session(f'!HardwareMap [ !CRS {{serial: "{serial}"}} ]')
    d = hwm.query(rfmux.CRS).one()

    # Resolve the device to get all the methods    
    await d.resolve()
    await d.set_timestamp_port(d.TIMESTAMP_PORT.TEST)

    return d


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
