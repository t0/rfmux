#!/usr/bin/env python3

"""
Schema tests.

This test script can be invoked in three ways:

- By itself, in your PC's current Python environment, without relying on a
  running CRS board:

      ~/rfmux/test$ PYTHONPATH=.. ./test_schema.py 

- By itself, in your PC's current Python environment, with a CRS board running
  alongside: (note the CRS_SERIAL environment variable!)

      ~/rfmux/test$ CRS_SERIAL=0024 PYTHONPATH=.. ./test_schema.py

- As part of a complete regression test environment (exercising all test
  scripts and in a variety of different Python environments), without relying
  on a running CRS board:

      ~/rfmux$ ./test.sh
"""

import rfmux
import pytest
import os
import textwrap


def test_hardware_map_with_single_board():
    s = rfmux.load_session(
        """
        !HardwareMap
        - !CRS { serial: "0024" }
        """
    )
    d = s.query(rfmux.CRS).one()
    assert d.serial == "0024"


def test_hardware_map_with_single_crate():
    s = rfmux.load_session(
        """
        !HardwareMap
        - !Crate { serial: "001" }
        """
    )
    d = s.query(rfmux.Crate).one()
    assert d.serial == "001"


def test_hardware_map_with_crate_slots_indexed_by_list():
    s = rfmux.load_session(
        """
        !HardwareMap
        - !Crate
          serial: "001"
          slots:
            - !CRS { serial: "0024" }
            - !CRS { serial: "0025" }
            - !CRS { serial: "0026" }
        """
    )

    c = s.query(rfmux.Crate).one()
    (d1, d2, d3) = s.query(rfmux.CRS).all()

    # Slots are 1-indexed, so there's some fixup to avoid starting at 0 like a
    # Python array naturally would.
    assert set(c.slots.slot) == {1, 2, 3}
    assert c.slot[1].serial == "0024"
    assert c.slot[2].serial == "0025"
    assert c.slot[3].serial == "0026"

    # Ensure dfmux objects have the expected serial numbers
    assert {d1.serial, d2.serial, d3.serial} == {"0024", "0025", "0026"}

    # Repeat test, but retrieve the serials at ORM level
    assert set(s.query(rfmux.CRS).serial) == {"0024", "0025", "0026"}

    # Ensure everyone agrees on crate serials
    assert {c.serial, d1.crate.serial, d2.crate.serial, d3.crate.serial} == {"001"}


def test_hardware_map_with_crate_slots_indexed_by_dictionary():
    s = rfmux.load_session(
        """
        !HardwareMap
        - !Crate
          serial: "001"
          slots:
            1: !CRS { serial: "0024" }
            2: !CRS { serial: "0025" }
            3: !CRS { serial: "0026" }
        """
    )

    c = s.query(rfmux.Crate).one()
    assert c.serial == "001"

    # Here, slots are explicitly given their indices and should match
    assert c.slot[1].serial == "0024"
    assert c.slot[2].serial == "0025"
    assert c.slot[3].serial == "0026"


def test_hardware_map_with_wafer_and_resonator_csv(tmp_path):
    csvfile = tmp_path / "test.csv"

    # Create a CSV file describing a few Resonators. We'll load this below in
    # the HWM.
    csvfile.write_text(
        textwrap.dedent(
            f"""
                name\tbias_freq\tbias_amplitude
                steve\t100e6\t0.1
                nancy\t101e6\t0.2
            """
        ).strip()
    )

    s = rfmux.load_session(
        f"""
        !HardwareMap
        - !Wafer
          name: some_wafer
          resonators: !Resonators "{str(csvfile)}"
        """
    )

    # Query the resonators, sorted by bias amplitude.
    r1, r2 = s.query(rfmux.Resonator).order_by(rfmux.Resonator.bias_amplitude).all()

    # Ensure we picked them up with the correct values. Note that type
    # conversion happens implicitly here - the CSV is just a bunch of strings.
    assert r1.name == "steve"
    assert r1.bias_freq == 100e6
    assert r1.bias_amplitude == 0.1
    assert r1.wafer.name == "some_wafer"

    assert r2.name == "nancy"
    assert r2.bias_freq == 101e6
    assert r2.bias_amplitude == 0.2
    assert r2.wafer.name == "some_wafer"


@pytest.fixture
def live_session():
    if "CRS_SERIAL" not in os.environ:
        pytest.skip(
            "Set the CRS_SERIAL environment variable to match your CRS board to run this test."
        )

    return rfmux.load_session(
        f"""
        !HardwareMap
        - !CRS {{ serial: "{os.environ['CRS_SERIAL']}" }}
        """
    )


def test_simple_live_board_interaction(live_session):
    d = live_session.query(rfmux.CRS).one()
    d.resolve()
    d.get_frequency(d.UNITS.HZ, channel=1, module=1)


def test_live_board_interaction_with_orm(live_session):
    ds = live_session.query(rfmux.CRS)
    ds.resolve()

    d = live_session.query(rfmux.CRS).one()
    f = d.get_frequency(d.UNITS.HZ, channel=1, module=1)
    fs = ds.get_frequency(d.UNITS.HZ, channel=1, module=1)

    assert {f} == set(fs)


if __name__ == "__main__":
    pytest.main([__file__])
