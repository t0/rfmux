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
from .schema import (
    CRS,
    ChannelMapping,
    Crate,
    ReadoutChannel,
    ReadoutModule,
    Resonator,
    Wafer,
)
from .session import HWMConstructor, HWMCSVConstructor, read_csv


from . import session

import sys
import os
import yaml


class YAMLLoader(session.YAMLLoader):

    # This is the default flavour
    flavour = sys.modules[__name__]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Plumbing
        self.add_constructor(u"!flavour", yaml_flavour_constructor)

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

        # Cryogenic chain
        self.add_constructor(
            "!Wafer", HWMConstructor(lambda loader: getattr(loader.flavour, "Wafer"))
        )

        self.add_constructor(
            "!Resonators",
            HWMCSVConstructor(lambda loader: getattr(loader.flavour, "Resonator")),
        )

        self.add_constructor("!ChannelMappings", ChannelMappingCSVConstructor)


def yaml_flavour_constructor(loader, node):
    """!flavour tag for YAML.

    This tag must occur within a HardwareMap directive. Note that it's
    swallowed by the HardwareMap, which ignores its children after
    instantiating them.  (The children add themselves in their instantiation
    methods.)
    """

    name = loader.construct_scalar(node)
    __import__(name)

    flavour = loader.flavour = sys.modules[name]

    if hasattr(flavour, "yaml_hook"):
        loader.register_finalization_hook(flavour.yaml_hook)

    return sys.modules[name]



def ChannelMappingCSVConstructor(loader, node):
    """
    Expect the following attributes:

    resonator:
        A path specifying a resonator (wafer/name or name)

    channel:
        The channel number used to index ReadoutChannel
    """

    fn = os.path.join(os.path.dirname(loader.name), node.value)
    dr = read_csv(fn)

    to_add = []

    for mapping in dr:
        # Parse 'readout_channel' entry into a ReadoutChannel mapping
        if "readout_channel" not in mapping:
            raise yaml.YAMLError("Missing 'readout_channel' mapping in ChannelMapping CSV")

        match mapping["readout_channel"].split("/"):
            case (serial, mod, channel):
                mapping["readout_channel"] = (
                    loader.hwm.query(ReadoutChannel)
                    .join(ReadoutModule)
                    .join(CRS)
                    .filter(
                        ReadoutChannel.channel == channel,
                        ReadoutModule.module == mod,
                        CRS.serial == serial,
                    )
                    .one()
                )

            case (crate, slot, mod, channel):
                mapping["readout_channel"] = (
                    loader.hwm.query(ReadoutChannel)
                    .join(ReadoutModule)
                    .join(CRS)
                    .join(Crate)
                    .filter(
                        ReadoutChannel.channel == channel,
                        ReadoutModule.module == mod,
                        CRS.slot == slot,
                        Crate.serial == crate,
                    )
                    .one()
                )

            case _:
                raise yaml.YAMLError(f"Unable to parse {mapping['readout_channel']} into a ReadoutChannel")

        # Parse 'resonator' entry into a Resonator mapping
        if "resonator" not in mapping:
            raise yaml.YAMLError("Missing 'resonator' mapping in ChannelMapping CSV")

        match mapping["resonator"].split("/"):
            case (wafer, resonator):
                mapping["resonator"] = (
                    loader.hwm.query(Resonator)
                    .join(Wafer)
                    .filter(Resonator.name == resonator, Wafer.name == wafer)
                    .one()
                )

            case _:
                raise yaml.YAMLError(f"Unable to parse {mapping['resonator']} into Resonator mapping!")

    loader.hwm.add_all([ChannelMapping(**x) for x in dr])


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
