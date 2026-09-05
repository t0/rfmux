"""The skip shared by every streamer probe test that needs SO_REUSEPORT.

A plain module rather than conftest so it can be imported by name (see
``test/qt_helpers.py``).
"""

import socket

import pytest

#: Port sharing between readers is a property of SO_REUSEPORT, which
#: Windows does not have.  Skipping is the honest outcome there: the
#: option is absent, so the behaviour it produces cannot arise.
requires_reuseport = pytest.mark.skipif(
    not hasattr(socket, "SO_REUSEPORT"),
    reason="SO_REUSEPORT is POSIX-only; Windows has no equivalent")
