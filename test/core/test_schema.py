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


@pytest.mark.portable
def test_hardware_map_with_single_board():
    s = rfmux.load_session(
        """
        !HardwareMap
        - !CRS { serial: "0024" }
        """
    )
    d = s.query(rfmux.CRS).one()
    assert d.serial == "0024"


@pytest.mark.portable
def test_readout_module_index():
    s = rfmux.load_session(
        """
        !HardwareMap
        - !CRS { serial: "0024" }
        """
    )
    d = s.query(rfmux.CRS).one()

    assert d.index() == "crs0024"
    assert d.module[1].index() == "crs0024_rmod1"
    assert d.module[4].index() == "crs0024_rmod4"


@pytest.mark.portable
def test_index_prefers_serial_over_hostname():
    """The serial is the board's identity; the hostname is only a route to it.

    This is the opposite precedence to `tuber_hostname`, deliberately — an
    explicit hostname overriding a serial for *connecting* does not make the
    board a different board. The standard mock map carries both, and should
    name itself after the serial sitting right there.
    """
    s = rfmux.load_session(
        """
        !HardwareMap
        - !CRS { serial: "0000", hostname: "127.0.0.1" }
        """
    )
    d = s.query(rfmux.CRS).one()

    assert d.index() == "crs0000"


@pytest.mark.portable
@pytest.mark.parametrize(
    "hostname, expected",
    [
        ("rfmux0042.local", "hostrfmux0042-local"),
        ("192.168.2.100", "host192-168-2-100"),
    ],
)
def test_index_falls_back_to_hostname(hostname, expected):
    """A board with no serial still names itself, rather than saying "None".

    Reachable in practice: Periscope's "just type an IP or hostname" path
    builds exactly this (`tools/periscope/__main__.py`). Punctuation is
    flattened to dashes so the result stays usable as a dict key and as a
    filename component.

    Built without a session on purpose. `rfmux.mock`'s flavour rewrites every
    CRS hostname in the HWM to `localhost:<ephemeral port>` and stays applied
    for the rest of the process, so a session-based version of this test would
    pass or fail depending on whether a mock test ran first.
    """
    d = rfmux.CRS(hostname=hostname)

    assert d.index() == expected
    assert "None" not in d.index()


@pytest.mark.xfail(
    strict=True,
    reason="A CRS in a crate slot cannot be constructed at all — the same "
           "tuber-client limitation the crate tests below are marked for. So "
           "index()'s crate/slot fallback, which mirrors tuber_hostname's, is "
           "unreachable for now. strict=True so that fixing the crate "
           "limitation fails here and prompts someone to check this form "
           "too, rather than becoming an XPASS nobody notices.")
@pytest.mark.portable
def test_index_falls_back_to_crate_and_slot():
    s = rfmux.load_session(
        """
        !HardwareMap
        - !Crate
          serial: "001"
          slots:
            3: !CRS { hostname: "10.0.0.7" }
        """
    )
    d = s.query(rfmux.CRS).one()

    assert d.index() == "crate001_slot3"


@pytest.mark.portable
def test_hardware_map_with_single_crate():
    s = rfmux.load_session(
        """
        !HardwareMap
        - !Crate { serial: "001" }
        """
    )
    d = s.query(rfmux.Crate).one()
    assert d.serial == "001"


@pytest.mark.xfail(
    strict=True,
    reason="Crate slot indexing is unsupported. "
           "SimpleTuberObject.__getattr__ in the tuber-client package "
           "intercepts _items, so len() and "
           "iteration on a Dfmux proxy raise instead of reaching the "
           "mapping. A settled limitation, not a flake — strict=True so "
           "that fixing it fails here instead of quietly becoming an XPASS "
           "nobody notices.")
@pytest.mark.portable
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


@pytest.mark.xfail(
    strict=True,
    reason="Crate slot indexing is unsupported. "
           "SimpleTuberObject.__getattr__ in the tuber-client package "
           "intercepts _items, so len() and "
           "iteration on a Dfmux proxy raise instead of reaching the "
           "mapping. A settled limitation, not a flake — strict=True so "
           "that fixing it fails here instead of quietly becoming an XPASS "
           "nobody notices.")
@pytest.mark.portable
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


@pytest.mark.portable
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
          hwm_resonators: !HWMResonators "{csvfile.as_posix()}"
        """
    )

    # Query the resonators, sorted by bias amplitude.
    r1, r2 = s.query(rfmux.HWMResonator).order_by(rfmux.HWMResonator.bias_amplitude).all()

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


@pytest.mark.xfail(
    strict=True,
    reason="Crate slot indexing is unsupported. "
           "SimpleTuberObject.__getattr__ in the tuber-client package "
           "intercepts _items, so len() and "
           "iteration on a Dfmux proxy raise instead of reaching the "
           "mapping. A settled limitation, not a flake — strict=True so "
           "that fixing it fails here instead of quietly becoming an XPASS "
           "nobody notices.")
@pytest.mark.portable
def test_hardware_map_with_channel_mappings(tmp_path):

    # Create a CSV file describing a few Resonators. We'll load this below in
    # the HWM.
    mapping = tmp_path / "channel_mapping.csv"
    mapping.write_text(
        textwrap.dedent(
            f"""
                hwm_resonator\treadout_channel
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
          hwm_resonators: !HWMResonators "{resonators.as_posix()}"

        - !ChannelMappings "{str(mapping)}"
        """
    )

    # Query the resonators, sorted by bias amplitude.
    r1, r2, r3, r4 = (
        s.query(rfmux.HWMResonator).order_by(rfmux.HWMResonator.bias_amplitude).all()
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
