#!/usr/bin/env -S uv run pytest -s
"""
Housekeeping Tests.

Please see README for hints on using and extending these test cases.
"""

import pytest
from dataclasses import dataclass

from .crs_qc import render_markdown, ResultTable


@dataclass
class SensorReading:
    name: str
    nom: float
    min: float
    max: float

    @classmethod
    def relative(cls, name: str, nom: float, rel: float):
        return cls(name, nom=nom, min=(1 - rel) * nom, max=(1 + rel) * nom)


@pytest.mark.qc_stage1
@pytest.mark.asyncio
async def test_housekeeping_temperature(d, request, shelf, check):
    """
    ## Temperatures

    Nominal values are fairly arbitrary. Selected maximum operating
    temperatures taken from component datasheets are as follows:

    * RFSoC: 100C
    * PLLs: 85C (HMC7044, AD9574)
    * VCXO: 70C (CVHD-950, standard option)
    * Buck converters: 125C (LTM4638, LTM4636-1)

    Given the board's thermal design and hot spots, it is likely that the VCXO
    is the part with the lowest thermal margin.
    """

    NOM = 50.0
    MIN = 0.0
    MAX = 70.0

    sensors = (
        SensorReading(d.TEMPERATURE_SENSOR.MB_R5V0, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.MB_R3V3A, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.MB_R2V5, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.MB_R1V8, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.MB_R1V2A, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.MB_R1V4, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.MB_R1V2B, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.MB_R0V85A, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.MB_R0V85B, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.RFSOC_PL, nom=NOM, min=MIN, max=MAX),
        SensorReading(d.TEMPERATURE_SENSOR.RFSOC_PS, nom=NOM, min=MIN, max=MAX),
    )

    async with d.tuber_context() as ctx:
        for s in sensors:
            ctx.get_motherboard_temperature(s.name)
        temps = await ctx()

    rt = ResultTable(
        "Sensor", r"Reading (C)", "Nominal (C)", "Minimum (C)", r"Maximum (C)"
    )
    for s, v in zip(sensors, temps):
        if check(s.min <= v <= s.max):
            rt.pass_(s.name, v, s.nom, s.min, s.max)
        else:
            rt.fail(s.name, v, s.nom, s.min, s.max)

    with shelf as x:
        x["sections"][request.node.nodeid] = [
            render_markdown(test_housekeeping_temperature.__doc__),
            rt,
        ]


@pytest.mark.qc_stage1
@pytest.mark.asyncio
async def test_housekeeping_voltages(d, request, shelf, check):
    """
    ## Voltages

    There are, in general, two kinds of on-board voltage measurements:

    * Point-of-supply (POS) measurements, provided by I2C peripherals that are
      physically close to the regulators. Since we typically use remote-sense
      configurations for regulators, voltages measured here are higher than the
      set-point where PDN resistance causes a voltage drop.
    * Point-of-load (POL) measurements, provided inside or near a load IC.

    Maximum and minimum values for these rails are set as follows:

    * Point-of-supply (POS) measurements are allowed a relatively loose (10%)
      regulation tolerance, on the assumption that remote sense and POL
      measurements give us better constraints
    * Point-of-load (POL) measurements take their min/max values verbatim from
      the relevant component datasheets (primarily, DS926).
    """

    VOLTAGE_TOL = 0.10  # 10%
    sensors = (
        # The VBP rail is primarily limited by a "hard max" for the LTM4638
        # buck converter of 16V. This part's recommendex maximum is 15V.
        #
        # For rev4 and newer CRSes:
        # - VBP is measured at -ve node of F1
        # - voltage at LTM4638 is reduced by PDN and IR drop across L1
        # - over/undervoltage protection provided by U8
        #
        # For rev3 and older CRSes:
        # - VBP is measured at +ve node of F1
        # - voltage at LTM4638 is reduced by PDN and IR drop across F1 and Q13
        # - no overvoltage protection
        # - undervoltage protection (of a kind) provided by Q2/D3 - but this
        #   seems to make the system repeatedly cycle through its reset state
        SensorReading(d.VOLTAGE_SENSOR.MB_VBP, nom=12.0, min=8.0, max=16.0),
        # These rails are measured at the power supply, which is in turn
        # remotely sensed at point of load - so a sloppy margin here is OK,
        # provided we have tighter margins at the POL.
        SensorReading.relative(d.VOLTAGE_SENSOR.MB_R0V85A, nom=0.85, rel=0.1),
        SensorReading.relative(d.VOLTAGE_SENSOR.MB_R0V85B, nom=0.85, rel=0.1),
        SensorReading.relative(d.VOLTAGE_SENSOR.MB_R1V2A, nom=1.2, rel=0.1),
        SensorReading.relative(d.VOLTAGE_SENSOR.MB_R1V2B, nom=1.2, rel=0.1),
        SensorReading.relative(d.VOLTAGE_SENSOR.MB_R1V4, nom=1.4, rel=0.1),
        SensorReading.relative(d.VOLTAGE_SENSOR.MB_R1V8A, nom=1.8, rel=0.1),
        SensorReading.relative(d.VOLTAGE_SENSOR.MB_R2V5, nom=2.5, rel=0.1),
        SensorReading.relative(d.VOLTAGE_SENSOR.MB_R3V3A, nom=3.3, rel=0.1),
        SensorReading.relative(d.VOLTAGE_SENSOR.MB_R5V0, nom=5.0, rel=0.1),
        # These rails are measured at the RFSoC, and stricter margins are good
        # practice. These numbers typically come from DS926 Table 2.
        SensorReading(
            d.VOLTAGE_SENSOR.RFSOC_PSMGTRAVCC, nom=0.85, min=0.825, max=0.875
        ),
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_PSMGTRAVTT, nom=1.8, min=1.746, max=1.854),
        # VCCAMS is an internal name referring to VCCADC @ 1.8v, not VCCINT_AMS @ 0.85 v
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_VCCAMS, nom=1.8, min=1.710, max=1.890),
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_VCCAUX, nom=1.8, min=1.710, max=1.890),
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_VCCBRAM, nom=0.85, min=0.825, max=0.876),
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_VCCINT, nom=0.85, min=0.825, max=0.876),
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_VCCPLAUX, nom=1.8, min=1.746, max=1.854),
        SensorReading(
            d.VOLTAGE_SENSOR.RFSOC_VCCPLINTFP, nom=0.85, min=0.825, max=0.876
        ),
        SensorReading(
            d.VOLTAGE_SENSOR.RFSOC_VCCPLINTLP, nom=0.85, min=0.825, max=0.876
        ),
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_VCCPSAUX, nom=1.8, min=1.710, max=1.890),
        SensorReading.relative(d.VOLTAGE_SENSOR.RFSOC_VCCPSDDR, nom=1.2, rel=0.05),
        SensorReading(
            d.VOLTAGE_SENSOR.RFSOC_VCCPSINTFP, nom=0.85, min=0.808, max=0.892
        ),
        SensorReading(
            d.VOLTAGE_SENSOR.RFSOC_VCCPSINTFPDDR, nom=0.85, min=0.808, max=0.892
        ),
        SensorReading(
            d.VOLTAGE_SENSOR.RFSOC_VCCPSINTLP, nom=0.85, min=0.808, max=0.892
        ),
        SensorReading.relative(d.VOLTAGE_SENSOR.RFSOC_VCCPSIO0, nom=1.8, rel=0.05),
        SensorReading.relative(d.VOLTAGE_SENSOR.RFSOC_VCCPSIO1, nom=3.3, rel=0.05),
        SensorReading.relative(d.VOLTAGE_SENSOR.RFSOC_VCCPSIO2, nom=3.3, rel=0.05),
        SensorReading.relative(d.VOLTAGE_SENSOR.RFSOC_VCCPSIO3, nom=3.3, rel=0.05),
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_VCCVREFN, nom=0.0, min=-0.1, max=0.1),
        SensorReading.relative(d.VOLTAGE_SENSOR.RFSOC_VCCVREFP, nom=1.85, rel=0.05),
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_VCC_PSBATT, nom=0.0, min=-0.1, max=0.1),
        SensorReading(
            d.VOLTAGE_SENSOR.RFSOC_VCC_PSDDRPLL, nom=1.8, min=1.710, max=1.890
        ),
        SensorReading(d.VOLTAGE_SENSOR.RFSOC_VCC_PSPLL0, nom=1.2, min=1.164, max=1.236),
    )

    # Retrieve sensor values
    async with d.tuber_context() as ctx:
        for s in sensors:
            ctx.get_motherboard_voltage(s.name)
        voltages = await ctx()

    # Evaluate pass/fail
    rt = ResultTable("Sensor", "Reading", "Nominal", "Minimum", "Maximum")
    for s, v in zip(sensors, voltages):
        if check(s.min < v < s.max):
            rt.pass_(s.name, v, s.nom, s.min, s.max)
        else:
            rt.fail(s.name, v, s.nom, s.min, s.max)

    with shelf as x:
        x["sections"][request.node.nodeid] = [
            render_markdown(test_housekeeping_voltages.__doc__),
            rt,
        ]


@pytest.mark.qc_stage1
@pytest.mark.asyncio
async def test_housekeeping_currents(d, request, shelf, check):
    """
    ## Currents

    Because current measurements rely on parasitic elements, these tests are
    barely constrained and are mostly useful to gather historical measurements
    of nominal values.
    """

    # These are sloppy measurements - give them a huge margin.
    # Nominal readings aren't reported and are arbitrary.
    sensors = (
        SensorReading(d.CURRENT_SENSOR.MB_R0V85A, nom=1.0, min=0.1, max=10.0),
        SensorReading(d.CURRENT_SENSOR.MB_R0V85B, nom=1.0, min=0.1, max=10.0),
        SensorReading(d.CURRENT_SENSOR.MB_R1V2A, nom=1.0, min=0.1, max=10.0),
        SensorReading(d.CURRENT_SENSOR.MB_R1V2B, nom=1.0, min=0.1, max=10.0),
        SensorReading(d.CURRENT_SENSOR.MB_R1V4, nom=1.0, min=0.1, max=10.0),
        SensorReading(d.CURRENT_SENSOR.MB_R1V8A, nom=1.0, min=0.1, max=10.0),
        SensorReading(d.CURRENT_SENSOR.MB_R2V5, nom=1.0, min=0.1, max=10.0),
        SensorReading(d.CURRENT_SENSOR.MB_R3V3A, nom=1.0, min=0.1, max=10.0),
        SensorReading(d.CURRENT_SENSOR.MB_R5V0, nom=1.0, min=0.1, max=10.0),
        SensorReading(d.CURRENT_SENSOR.MB_VBP, nom=1.0, min=0.1, max=10.0),
    )
    async with d.tuber_context() as ctx:
        for s in sensors:
            ctx.get_motherboard_current(s.name)
        currents = await ctx()

    rt = ResultTable("Sensor", "Reading (A)", "Minimum (A)", "Maximum (A)")
    for s, v in zip(sensors, currents):
        if check.between(v, s.min, s.max):
            rt.pass_(s.name, v, s.min, s.max)
        else:
            rt.fail(s.name, v, s.min, s.max)

    with shelf as x:
        x["sections"][request.node.nodeid] = [
            render_markdown(test_housekeeping_currents.__doc__),
            rt,
        ]
