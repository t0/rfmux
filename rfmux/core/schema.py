"""Base object for CRS objects.

To specialize a CRS object for a particular experiment, you're
encouraged to create a subclass.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint, Float
from sqlalchemy.orm import relationship, backref
from sqlalchemy.orm.collections import attribute_mapped_collection
from .hardware_map import Boolean

from . import hardware_map, tuber, tworoutine

import base64
import asyncio

from packaging import version
import sqlalchemy


from .hardware_map import HWMResource, HWMQuery

from packaging import version
assert version.parse(sqlalchemy.__version__) >= version.parse('1.2')

from . import tuber

if version.parse(sqlalchemy.__version__) >= version.parse("1.4"): #TODO : is this still relevant?
    def overlaps(val):
        return dict(overlaps=val)
else:
    def overlaps(val):
        return dict()


@tuber.TuberCategory("Backplane", lambda b: b.slots.first())
class Crate(hardware_map.HWMResource):
    __tablename__ = "crates"
    __table_args__ = (UniqueConstraint("serial"),)
    __mapper_args__ = {"polymorphic_identity": __package__}

    _pk = Column(Integer, primary_key=True)
    serial = Column(String, doc="The serial number written on the board (verbatim!)")

    slots = relationship(
        "CRS",
        lazy="dynamic",
        query_class=hardware_map.HWMQuery,
        doc="""A SQLAlchemy subquery corresponding to this Crate's
            CRS boards. If you want to index this array using slot index,
            you should use the 'crs' object instead.""",
        **overlaps("slot"),
    )

    slot = relationship(
        "CRS",
        backref=backref("crate", **overlaps("slots")),
        collection_class=attribute_mapped_collection("slot"),
        doc="The Crate's CRS boards, indexed as you would expect.",
    )

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.serial)


class CRS(hardware_map.HWMResource, tuber.TuberObject):
    __tablename__ = "crs"
    __table_args__ = (UniqueConstraint("serial"),)
    __mapper_args__ = {"polymorphic_identity": "CRS", "polymorphic_on": "_cls"}

    _pk = Column(Integer, primary_key=True)
    _cls = Column(String, nullable=False)
    _crate_pk = Column(Integer, ForeignKey("crates._pk"), index=True)

    hostname = Column(String, doc="The hostname (or IP) to use for this resource.")
    serial = Column(String, doc="The serial number written on the board (verbatim!)")
    slot = Column(Integer, doc="The Crate slot occupied by this board")

    backplane_serial = Column(
        String, doc="The serial number of the Crate occupied by this board"
    )

    online = Column(Boolean, default=True)

    def __init__(self, *args, **kwargs):
        if "module" not in kwargs and "modules" not in kwargs:
            self.modules = [ReadoutModule(module=m+1) for m in range(4)]
        super().__init__(*args, **kwargs)


    modules = relationship(
        "ReadoutModule",
        lazy="dynamic",
        query_class=HWMQuery,
        doc="This is a SQLAlchemy subquery; you probably want 'module'",
        **overlaps("module"),
    )


    module = relationship(
        "ReadoutModule",
        backref=backref("crs", **overlaps("modules")),
        collection_class=attribute_mapped_collection("module"),
        doc="""Readout modules associated with this CRS board.

            You can use these modules to write more legible function calls.
            For example, the following two calls are equivalent::

                >>> x = d.get_fast_samples(100, d.UNITS.ADC_COUNTS,
                ...                        average=False,
                ...                        target=d.TARGET.DEMOD,
                ...                        module=1)

                >>> mod = d.module[1]
                >>> mod.get_fast_samples(100, d.UNITS.ADC_COUNTS,
                ...                      average=False,
                ...                      target=d.TARGET.DEMOD)

            When writing Python code that involves multiple operations in
            sequence to a single hardware element, the second form is often
            much clearer.""",
    )

    def __repr__(self):
        if self.crate:
            repstring =  "%r.%s(%s)" % (self.crate, self.__class__.__name__, "slot=%s" % self.slot)
        else:

            repstring = "%s(%s)" % (self.__class__.__name__,"%s" % self.serial)
        return repstring

    def set_fpga_bitstream(self, buf):
        """
        Configures the FPGA with the specified buffer.

        The buffer is an ordinary string object or similar, and
        contains an already loaded .BIT or .BIN file.
        """
        b64_string = base64.b64encode(buf)
        self._set_fpga_bitstream_base64(b64_string)


    @property
    def tuber_uri(self):
        """Smarter, CRS-aware tuber_uri.

        The version of 'tuber_uri' in tuber.py doesn't know about calculating
        hostnames from serials, for instance.
        """

        if self.hostname:
            # We have a hostname; just use it.
            return "http://{}/tuber".format(self.hostname)

        if self.serial:
            # We have a serial number; compute the hostname.
            return "http://crs{}.local/tuber".format(self.serial)

        if self.slot and self.crate:
            # We have a slot and crate,
            # we can use the crate-based hostname (i.e. slot3.crate001.local).
            return "http://slot{0}.crate{1}.local/tuber".format(
                self.slot, self.crate.serial
            )

        raise NameError(
            "Couldn't figure out a Tuber URI for this object! "
            "I need serial or crate information."
        )

    @tworoutine.tworoutine
    async def resolve(self):
        await (~self._tuber_get_meta)()

@tuber.TuberCategory(
    "ReadoutModule",
    lambda m: m.crs,
    module=lambda m: m.module,
)
class ReadoutModule(HWMResource):
    """ReadoutModule provides a home for module-scope functions."""

    __tablename__ = "readout_modules"
    __mapper_args__ = {"polymorphic_identity": __package__, "polymorphic_on": "_cls"}

    def __init__(self, *args, **kwargs):
        if "channel" not in kwargs and "channels" not in kwargs:
            kwargs["channels"] = [ReadoutChannel(channel=c + 1) for c in range(1024)]

        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "%s.%s(%r)" % (self.crs.__repr__(), self.__class__.__name__, self.module)

    def index(self):
        '''
        A shorthand string representation for this readout module, in the form:
        'crs0030_rmod1'
        '''
        
        return 'crs%s_rmod%d' % (self.crs.serial, self.module)

    # Boilerplate
    _cls = Column(String, nullable=False)
    _pk = Column(Integer, primary_key=True)
    _crs_pk = Column(Integer, ForeignKey("crs._pk"), index=True)
    module = Column(Integer, index=True, doc="Module number (1-4)")


    channel = relationship(
        "ReadoutChannel",
        backref=backref("module", **overlaps("channels")),
        collection_class=attribute_mapped_collection("channel"),
        doc="""Readout channels associated with this module.

            You can use these modules to write more legible function calls.
            For example, the following two calls are equivalent::

                >>> x = d.get_frequency(d.UNITS.HZ, d.TARGET.DEMOD,
                ...                        channel=1,
                ...                        module=1
                ...                        )

                >>> c = d.module[1].channel[1]
                >>> c.get_frequency(d.UNITS.HZ, d.TARGET.DEMOD)

            When writing Python code that involves multiple operations in
            sequence to a single hardware element, the second form is often
            much clearer.""",
    )

    channels = relationship(
        "ReadoutChannel",
        lazy="dynamic",
        query_class=HWMQuery,
        doc="This is a SQLAlchemy subquery; you probably want 'channel'",
        **overlaps("channel"),
    )


@tuber.TuberCategory(
    "ReadoutChannel",
    lambda c: c.crs,
    module=lambda c: c.module.module,
    channel=lambda c: c.channel,
)
class ReadoutChannel(HWMResource):
    """ReadoutChannel provides a home for channel-scope functions."""

    __tablename__ = "readout_channels"
    __mapper_args__ = {"polymorphic_identity": __package__, "polymorphic_on": "_cls"}

    def __repr__(self):
        return "%r.%s(%r)" % (self.module, self.__class__.__name__, self.channel)

    # Boilerplate
    _cls = Column(String, nullable=False)
    _pk = Column(Integer, primary_key=True)
    _mod_pk = Column(Integer, ForeignKey("readout_modules._pk"), index=True)

    channel = Column(Integer, index=True, doc="Channel number (1-1024)")

    frequency = Column(
        Float,
        doc="Frequency",
    )

    crs = property(
        lambda x: x.module.crs if x.module else None,
        doc="Shortcut back to the CRS board",
    )


@hardware_map.algorithm(CRS, register=True)
async def resolve(boards):
    await asyncio.gather(*[(~d.resolve)() for d in boards])

# vim: sts=4 ts=4 sw=4 tw=78 smarttab expandtab
