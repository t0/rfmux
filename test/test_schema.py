#!/usr/bin/env -S PYTHONPATH=.. pytest-3 -v

"""
Schema tests.

This test script can be invoked in two ways:

- By itself, in your PC's current Python environment:

      ./test_schema.py

- As part of a complete regression test environment (exercising all test
  scripts and in a variety of different Python environments), without relying
  on a running CRS board:

      ~/rfmux$ ./test.sh
"""

import rfmux
import pytest
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


def test_hardware_map_with_channel_mappings(tmp_path):

    # Create a CSV file describing a few Resonators. We'll load this below in
    # the HWM.
    mapping = tmp_path / "channel_mapping.csv"
    mapping.write_text(
        textwrap.dedent(
            f"""
                resonator\treadout_channel
                some_wafer/steve\t0024/1/1
                some_wafer/nancy\t0025/1/1
                some_wafer/george\t003/1/1/2
                some_wafer/georgina\t003/2/1/2
            """
        ).strip()
    )

    # Create a CSV file describing a few Resonators. We'll load this below in
    # the HWM.
    resonators = tmp_path / "resonators.csv"
    resonators.write_text(
        textwrap.dedent(
            f"""
                name\tbias_freq\tbias_amplitude
                steve\t100e6\t0.1
                nancy\t101e6\t0.2
                george\t102e6\t0.3
                georgina\t103e6\t0.4
            """
        ).strip()
    )

    s = rfmux.load_session(
        f"""
        !HardwareMap
        - !Crate
          serial: "003"
          slots:
            1: !CRS {{ serial: "0024" }}
            2: !CRS {{ serial: "0025" }}

        - !Wafer
          name: some_wafer
          resonators: !Resonators "{str(resonators)}"

        - !ChannelMappings "{str(mapping)}"
        """
    )

    # Query the resonators, sorted by bias amplitude.
    r1, r2, r3, r4 = (
        s.query(rfmux.Resonator).order_by(rfmux.Resonator.bias_amplitude).all()
    )

    assert r1.name == "steve"
    assert r1.readout_channel.module.crs.serial == "0024"
    assert r1.readout_channel.channel == 1

    assert r2.name == "nancy"
    assert r2.readout_channel.module.crs.serial == "0025"
    assert r2.readout_channel.channel == 1

    assert r3.name == "george"
    assert r3.readout_channel.module.crs.serial == "0024"
    assert r3.readout_channel.channel == 2

    assert r4.name == "georgina"
    assert r4.readout_channel.module.crs.serial == "0025"
    assert r4.readout_channel.channel == 2


if __name__ == "__main__":
    pytest.main([__file__])
