"""
Dirfile (libgetdata) interface

Where possible, we try to just expose the underlying pygetdata API - it's not
terribly hard to use, and attempting to wrap it in something "better" is a
fairly weak position.
"""


def load_dirfile(name):
    return Dirfile(name)


class Dirfile:
    """
    Augment pygetdata.dirfile with rfmux-specific features

    This is implemented using __getattr__ shenanigans because we can't directly
    inherit from a pygetdata.dirfile() instance - inheriting from a C/C++ base
    class isn't possible.
    """

    def __init__(self, dirfile_path: str):
        import pygetdata

        self._df = pygetdata.dirfile(dirfile_path)

    def __dir__(self):
        # Allow tab-completion on underlying pygetdata.dirfile() instance
        return list(set(self._df.__dir__() + super().__dir__()))

    def __getattr__(self, name: str):
        # Can't use pygetdata.dirfile as a base class, so here we are instead
        return getattr(self._df, name)

    def get_samples(self, serial, module: int, channel: int):
        """
        Friendly API shim to retrieve rfmux sample data from a dirfile.

        This:

        >>> df.get_samples("0024", module=1, channel=1)

        is just a wrapper around the pygetdata API:

        >>> df.getdata("serial_0024.m01_c0001")
        """

        if isinstance(serial, int):
            serial = f"{serial:04i}"
        elif not isinstance(serial, str):
            raise TypeError("Unexpected type for argument 'serial')")

        if not isinstance(module, int) or not isinstance(channel, int):
            raise TypeError("Unexpected integers for module/channel arguments")

        return self.getdata(f"serial_{serial}.m{module:02d}_c{channel:04d}")
