import sys
import importlib
import rfmux
import inspect
import io
import os
import asyncio

def compare_attributes(attr_mock, attr_real):
    diff_real = list(set(attr_real) - set(attr_mock))
    diff_mock = list(set(attr_mock) - set(attr_real))
    same_elements = list(set(attr_mock) & set(attr_real))
    return diff_mock, diff_real, same_elements

def get_attribute_names(doc, mock=True):
    brac = (doc.split('(')[1]).split(')')[0]
    args = brac.split(',')
    calls = []
    for arg in args:
        if mock:
            cut = (arg.split('=')[0]).split()[0]
        else:
            cut = (arg.split(':')[0]).split()[0]
        if cut == "self":
            continue
        else:
            calls.append(cut)
    return calls

def get_function_attributes(same_un, crs_mock, crs_real):
    same = sorted(same_un)
    print("\nFunction calls that are different in mock and real mode\n")
    for ob in same:
        try:
            func_real = getattr(crs_real, ob)
            doc_real = inspect.getdoc(func_real)
            real = get_attribute_names(doc_real, False)

            func_mock = getattr(crs_mock, ob)
            doc_mock = inspect.getdoc(func_mock)
            mock = get_attribute_names(doc_mock, True)
        
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
    s_mock = rfmux.load_session("""
    !HardwareMap
    - !flavour "rfmux.core.mock"
    - !CRS { serial: "0000", hostname: "127.0.0.1" }
    """)
    
    crs_mock = s_mock.query(rfmux.CRS).one()
    asyncio.run(crs_mock.resolve())
    
    mock_dir = dir(crs_mock)
    print("MOCK_LEN:", len(mock_dir))
    
    rfmux.set_session(None)
    
    r_new = load_fresh_rfmux()
    
    serial = "0013" ### This can be user input ####
    
    yaml_map = f"""
    !HardwareMap
    - !CRS {{ serial: "{serial}" }}
    """
    
    s_real = r_new.load_session(yaml_map)
    crs_real = s_real.query(r_new.CRS).one()
    asyncio.run(crs_real.resolve())
    
    real_dir = dir(crs_real)
    print("REAL_LEN:", len(real_dir))
    
    result = compare_attributes(mock_dir, real_dir)
    if result is None:
        print("Mock and real attribute lengths are equal")
    else:
        mock_diff, real_diff, same = result
    
    print("\nREAL_NOT_IN_MOCK\n", sorted(real_diff))
    print("\nMOCK_NOT_IN_REAL\n", sorted(mock_diff))
    
    get_function_attributes(same, crs_mock, crs_real)

if __name__ == "__main__":
    main()