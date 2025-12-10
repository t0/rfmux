'''
This script compares the attributes between real board and the mock framework. It should be run whenever new firmware is deployed.
How to run: python mock_vs_real.py --serial 0013 (or serial number of your board)
'''

import sys
import importlib
import rfmux
import inspect
import io
import os
import asyncio
import argparse

def compare_attributes(attr_mock, attr_real):
    diff_real = list(set(attr_real) - set(attr_mock))
    diff_mock = list(set(attr_mock) - set(attr_real))
    same_elements = list(set(attr_mock) & set(attr_real))
    return sorted(diff_mock), sorted(diff_real), sorted(same_elements)

def get_attribute_names(doc):
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

        if arg:  # ignore empty strings
            names.append(arg)

    return names

def get_function_attributes(same_un, crs_mock, crs_real):
    same = sorted(same_un)
    print("\nFunction calls that are different in mock and real mode\n")
    for ob in same:
        try:
            func_real = getattr(crs_real, ob)
            doc_real = inspect.getdoc(func_real)
            real = get_attribute_names(doc_real)

            func_mock = getattr(crs_mock, ob)
            doc_mock = inspect.getdoc(func_mock)
            mock = get_attribute_names(doc_mock)
        
            if real != mock:
                print(f"FUNCTION:{ob}")   
                print("REAL:", real)
                print("MOCK:", mock)
                print("\n")
        except:
            continue


def load_fresh_rfmux():
    for name in list(sys.modules.keys()):
        if name.startswith("rfmux"):
            del sys.modules[name]
    return importlib.import_module("rfmux")

def main():
    # --- load mock session ---

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serial",
        required=True,
        help="Serial number of the real crs board."
    )
    
    args = parser.parse_args()
    
    s_mock = rfmux.load_session("""
    !HardwareMap
    - !flavour "rfmux.mock"
    - !CRS { serial: "0000", hostname: "127.0.0.1" }
    """)
    
    crs_mock = s_mock.query(rfmux.CRS).one()
    asyncio.run(crs_mock.resolve())
    
    mock_dir = dir(crs_mock)
    print("MOCK_LEN:", len(mock_dir))
    
    rfmux.set_session(None)
    
    r_new = load_fresh_rfmux()
    
    serial = args.serial
    # serial = "0013" ### This can be user input ####
    
    yaml_map = f"""
    !HardwareMap
    - !CRS {{ serial: "{serial}" }}
    """
    
    s_real = r_new.load_session(yaml_map)
    crs_real = s_real.query(r_new.CRS).one()
    asyncio.run(crs_real.resolve())
    
    real_dir = dir(crs_real)
    print("REAL_LEN:", len(real_dir))
    
    mock_diff, real_diff, same = compare_attributes(mock_dir, real_dir)
    
    print("\nAttributes - REAL_NOT_IN_MOCK\n")
    for r in real_diff:
        print(r)
    print("\nAttributes - MOCK_NOT_IN_REAL\n")
    for m in mock_diff:
        print(m)
    
    get_function_attributes(same, crs_mock, crs_real)

if __name__ == "__main__":
    main()