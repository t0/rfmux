"""In-process IPython kernel for Periscope's embedded console."""

import threading

from ipykernel.inprocess.ipkernel import InProcessKernel
from qtconsole.inprocess import QtInProcessKernelManager


class _ShellChannel:
    """Stands in for the shell channel thread a ZMQ kernel owns.

    ipykernel 7 decides whether an ``await`` cell may install its SIGINT
    handler by comparing the current thread with the shell channel's
    parent thread. The in-process kernel has no shell channel thread, so
    the comparison raises. Only that attribute is served here.
    """

    def __init__(self):
        self.parent_thread = threading.current_thread()


class ConsoleKernel(InProcessKernel):
    def __init__(self, **traits):
        super().__init__(**traits)
        self.shell_channel_thread = _ShellChannel()

    # Subshells need a real shell channel thread. With the stand-in present,
    # ipykernel would otherwise route every request through its manager.
    _supports_kernel_subshells = property(lambda self: False)


class ConsoleKernelManager(QtInProcessKernelManager):
    def start_kernel(self, **kwds):
        self.kernel = ConsoleKernel(parent=self, session=self.session)
