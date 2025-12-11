"""
Pytest suite for comparing attributes and function signatures
between the real CRS board and the mock framework.

Run with:
pytest test_mock_vs_real.py --serial 0013
"""

import sys
import importlib
import inspect
import asyncio
import pytest
import rfmux
import pytest_check as check


def compare_attributes(attr_mock, attr_real):
    diff_real = list(set(attr_real) - set(attr_mock))
    diff_mock = list(set(attr_mock) - set(attr_real))
    same_elements = list(set(attr_mock) & set(attr_real))
    return sorted(diff_mock), sorted(diff_real), sorted(same_elements)


def get_attribute_names(doc):
    if not doc:
        return []

    args = doc.split("(", 1)[1].split(")", 1)[0].split(",")
    names = []

    for arg in args:
        arg = arg.strip()
        if ":" in arg:
            arg = arg.split(":", 1)[0].strip()

        if "=" in arg:
            arg = arg.split("=", 1)[0].strip()

        if any(x in arg for x in ("int", "float", "self", "str")):
            continue

        if arg:
            names.append(arg)

    return names


def load_fresh_rfmux():
    """Reload rfmux to ensure fresh module import."""
    for name in list(sys.modules.keys()):
        if name.startswith("rfmux"):
            del sys.modules[name]
    return importlib.import_module("rfmux")


def pytest_addoption(parser):
    parser.addoption(
        "--serial",
        action="store",
        required=True,
        help="Serial number of the real CRS board."
    )


@pytest.fixture(scope="session")
def serial(request):
    return request.config.getoption("--serial")


@pytest.fixture(scope="session")
def crs_mock():
    """Load mock CRS session."""
    s_mock = rfmux.load_session("""
    !HardwareMap
    - !flavour "rfmux.mock"
    - !CRS { serial: "0000", hostname: "127.0.0.1" }
    """)
    crs = s_mock.query(rfmux.CRS).one()
    asyncio.run(crs.resolve())
    return crs


@pytest.fixture(scope="session")
def crs_real(serial):
    """Load real CRS session using provided serial."""
    r_new = load_fresh_rfmux()

    yaml_map = f"""
    !HardwareMap
    - !CRS {{ serial: "{serial}" }}
    """

    s_real = r_new.load_session(yaml_map)
    crs = s_real.query(r_new.CRS).one()
    asyncio.run(crs.resolve())
    return crs


# ---------------------------------------------------------------------
#                         Tests
# ---------------------------------------------------------------------

def test_attribute_sets_match(crs_mock, crs_real):
    mock_dir = dir(crs_mock)
    real_dir = dir(crs_real)

    mock_diff, real_diff, same = compare_attributes(mock_dir, real_dir)

    check.is_true(
        not real_diff,
        "Attributes present in REAL but missing in MOCK:\n" + "\n".join(real_diff)
    )

    check.is_true(
        not mock_diff,
        "Attributes present in MOCK but missing in REAL:\n" + "\n".join(mock_diff)
    )


def test_function_signatures_match(crs_mock, crs_real):
    mock_dir = dir(crs_mock)
    real_dir = dir(crs_real)
    _, _, same = compare_attributes(mock_dir, real_dir)

    mismatches = []

    for name in sorted(same):
        try:
            func_real = getattr(crs_real, name)
            func_mock = getattr(crs_mock, name)

            doc_real = inspect.getdoc(func_real)
            doc_mock = inspect.getdoc(func_mock)

            real_args = get_attribute_names(doc_real)
            mock_args = get_attribute_names(doc_mock)

            if real_args != mock_args:
                mismatches.append(
                    f"Function: {name}\n"
                    f"REAL: {real_args}\n"
                    f"MOCK: {mock_args}\n"
                )
        except Exception:
            continue

    check.is_true(
        not mismatches,
        "Function signature differences detected:\n\n" + "\n".join(mismatches)
    )
