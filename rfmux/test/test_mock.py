#!/usr/bin/env python3
"""
Mock rfmux Tests
~~~~~~~~~~~~~~~~

The "mock rfmux" is a Pure-Python implementation of a back-end TuberObject, with
enough support in the HWM code to act as a transparent replacement for real
hardware.

The following test cases ensure the mock dfmux is operating properly.
"""

import unittest
import rfmux
import numpy
import pytest

class MockDfmuxTestCase(unittest.TestCase):
    def test_set_random_frequencies(self):
        """Assign random frequencies and validate"""

        # This test is most useful with several mock boards
        s = rfmux.load_session(
            """
            !HardwareMap
            - !flavour "rfmux.test.mock"
            - !CRS { serial: "0354" }
            - !CRS { serial: "0355" }
            - !CRS { serial: "0356" }
        """
        )

        d = s.query(rfmux.CRS).first()
        bs = s.query(rfmux.CRS)
        bs.resolve()

        num_boards = bs.count()
        mod_per_board = 4
        cpm = 1024

        # Create a set of random frequencies, unique on a per-channel basis.
        freqs = numpy.random.uniform(
            0, 500e6, (num_boards, mod_per_board, cpm)
        )

        # Set every channel in the HWM to these frequencies
        set_values = []
        for board, fs in enumerate(freqs):
            with bs[board].tuber_context() as ctx:
                for mezz, fs in enumerate(fs):
                    for mod, fs in enumerate(fs):
                        for chan, f in enumerate(fs):
                            ctx.set_frequency(
                                f, d.UNITS.HZ, chan + 1, mod + 1, mezz + 1
                            )
                            set_values.append(f)

        # Retrieve the set values
        get_values = []
        for board, fs in enumerate(freqs):
            with bs[board].tuber_context() as ctx:
                for mezz, fs in enumerate(fs):
                    for mod, fs in enumerate(fs):
                        for chan, f in enumerate(fs):
                            ctx.get_frequency(d.UNITS.HZ, chan + 1, mod + 1, mezz + 1)

                get_values.extend(ctx())

        # Ensure they matched.
        for c, (fs, fg) in enumerate(zip(set_values, get_values)):
            assert fs == fg, f"Index {c} frequency {fg} didn't match expected {fs}"


if __name__ == "__main__":
    unittest.main()
