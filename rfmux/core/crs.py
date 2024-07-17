"""
Objects model for the CRS
=========================

"""

__all__ = [
    "CRS",
    "Crate",
    "ReadoutModule",
    "ReadoutChannel",
    "Wafer",
    "Resonator",
    "ChannelMapping",
]

from .hardware_map import algorithm
from .schema import Crate, CRS, ReadoutModule, ReadoutChannel
from .session import HWMConstructor


from . import session

import sys


class YAMLLoader(session.YAMLLoader):

    # This is the default flavour
    flavour = sys.modules[__name__]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Readout chain
        self.add_constructor(
            "!Crate",
            HWMConstructor(
                lambda loader: getattr(loader.flavour, "Crate"),
                AttributeMappingTouchup("slots", "slot"),
            ),
        )

        self.add_constructor(
            "!CRS",
            HWMConstructor(lambda loader: getattr(loader.flavour, "CRS")),
        )


class AttributeMappingTouchup:
    """Correctly assign indexes for SQLAlchemy attribute_mapped_collections.

    HWM objects like ReadoutModule come with index columns like "module",
    which indicate their position in a collection (crs.modules) starting
    from 1. This idiom is convenient in ORM-land, but awkward to support in
    YAML serialization.
    """

    def __init__(self, group_attribute, member_attribute):
        self._group_attribute = group_attribute
        self._member_attribute = member_attribute

    def __call__(self, loader, mapping):

        if self._group_attribute in mapping:
            values = mapping[self._group_attribute]

            if isinstance(values, list):
                # We've been provided a list. Start numbering at 1.
                for index, value in enumerate(values):
                    if not value:
                        continue

                    # Don't clobber an existing numbering.
                    if getattr(value, self._member_attribute) is not None:
                        continue

                    setattr(value, self._member_attribute, index + 1)

                mapping[self._group_attribute] = [v for v in values if v]

            elif isinstance(values, dict):
                # We've been provided a dictionary. Assume the keys
                # provide the numbering.
                for key, value in values.items():
                    if not value:
                        continue
                    setattr(value, self._member_attribute, key)

                mapping[self._group_attribute] = [v for v in values.values() if v]

            else:
                raise TypeError(f"Expected a list, got '{values}'!")


# Tell session to use our YAMLLoader.
session.set_yaml_loader_class(YAMLLoader)

# vim: sts=4 ts=4 sw=4 tw=78 smarttab expandtab
