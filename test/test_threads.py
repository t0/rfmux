#!/usr/bin/env -S PYTHONPATH=.. pytest-3 -v

"""
Threading tests.

SQLAlchemy has a rational threading model, with helpers that easily provide a
session-per-thread policy. These tests are intended to ensure we stay on the
straight and narrow path.

In hidfmux, HardwareMap objects were thinly wrapped SQLAlchemy Session objects.
However, this broke the threading model - a single Session object couldn't be
passed to threaded code.

Luckily, SQLAlchemy provides [1] a scoped_session object that defers
construction of the underlying Session and keeps a cached copy in thread-local
storage.  This should give us a single Session object per thread.

[1]: https://docs.sqlalchemy.org/en/20/orm/contextual.html#sqlalchemy.orm.scoped_session
"""

import threading
import rfmux

def test_are_sessions_distinct_between_threads():
    '''Check that threads each get their own underlying Session / HardwareMap'''

    s = rfmux.load_session(
        """
        !HardwareMap
        - !CRS { serial: "0024" }
        """
    )

    results = []

    CHECKS = 10  # really only need 2 here

    def call(hwm):
        # Convert the HWM from a scoped_session proxy into an actual
        # session. Otherwise, every thread gets the same proxy and our
        # checks fail even if the underlying calls would have been proxied.
        hwm = hwm()

        # Now pass this back to the main thread so we can check identities
        results.append(hwm)

    threads = [ threading.Thread(target=call, args=(s, )) for n in range(CHECKS) ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert len(results) == CHECKS
    for j, x in enumerate(results):
        for k, y in enumerate(results):
            assert (j==k) ^ (x is not y), "Identical Session/HardwareMap objects found in different Threads!"
