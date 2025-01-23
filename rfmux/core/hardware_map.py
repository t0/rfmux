"""Hardware mapper.

This HardwareMapper actually returns a SQLAlchemy session object, which we've
extended slightly.
"""

__all__ = [
    "Base",
    "HWMQueryException",
    "HWMQuery",
    "HWMResource",
    "macro",
    "algorithm",
    "HardwareMap",
    "Boolean",
    "Session",
]

import logging
import os
import time
import asyncio
import functools

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.types
import sqlalchemy.exc
import sqlite3

from .. import tuber

from sqlalchemy.inspection import inspect as sqlinspect
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class HWMQueryException(Exception):
    pass


class HWMQuery(sqlalchemy.orm.Query):
    """HWMQuery object: A parallel-call extension to Query objects.

    This is also pretty well internal; you shouldn't have to use it directly.
    Please look at the DocStrings for HardwareMap instead.

    This is an extension to SQLAlchemy's Query object. A HWMQuery can be
    used to dispatch method calls on every class it contains.
    """

    _algorithm_registry = {}
    _attribute_registry = {}

    @property
    def hwm(self):
        """
        Return the hardware map associated with this query object.
        """
        return self.session

    def count_distinct(self):
        """
        Return a count of distinct entries in this query. This is consistent
        with len(query.all()), but much more efficient, since it avoids having
        to create each of the ORM objects in the query.
        """
        return self.distinct().count()

    def __getattr__(self, name):
        """Teach a collection of query results how to parallelize.

        You would ordinarily retrieve a bunch of query results as
        follows (assuming "serial" is some property of the underlying
        object):

            >>> objects = hwm.query(ObjectClass)
            >>> for o in objects:
            ...     o.do_something()

        Since objects like "o" tend to be network resources, and calls
        like "o.do_something()" tend to be I/O-limited, we are
        interested in running this kind of function call in a
        multi-threaded context. We use green threads to avoid the
        overhead from threading -- which seems minimal, until we pile
        on the SQLAlchemy housekeeping that's required.

        This call allows you to do exactly that, as follows:

            >>> objects.do_something()

        It does so by trapping calls it doesn't recognize itself, and
        delegating them to each object in the Query results. It returns
        an array of results corresponding to each underlying object.
        """

        # Since this is a SQLAlchemy "Query" subclass, we can use it
        # as a collection and call things like "count()" on it.

        # Refuse to __getattr__ a couple of special names used elsewhere.
        if tuber.client.attribute_blacklisted(name):
            raise AttributeError()

        query_type = self.column_descriptions[0]["type"]

        # First, try algorithms from the registry.
        for cls in self._algorithm_registry:
            if not issubclass(query_type, cls):
                continue
            algs = self._algorithm_registry[cls]
            if name in algs:
                return functools.partial(algs[name], self)

        # Next, try attributes/methods from contained objects.
        # Special case: if the query returned no results, we can't reliably
        # ask it for attributes. Treat this as an error (we can sometimes
        # do the right thing, but not always.)
        if self.first() is None:
            raise AttributeError(
                f"Can't request attribute '{name}' from a Query with no results!"
            )

        # Generate an exception if only some of the objects have the
        # desired attribute. (If none of the objects have the attribute,
        # we want to raise the ordinary Python AttributeError. This
        # happens below.)
        attr_present = [hasattr(obj, name) for obj in self]
        if not any(attr_present):
            raise AttributeError(
                f"Attribute '{name}' does not exist on any element of the query"
            )
        elif not all(attr_present):
            raise AttributeError(
                f"Attribute '{name}' does not exist on *ALL* elements of the query"
            )

        # Get the specified attribute from all objects.
        attrs = [getattr(x, name) for x in self]

        # Ensure attributes are always, or never, callable.
        if len(set([callable(x) for x in attrs])) != 1:
            raise HWMQueryException("Called with mismatching objects!")

        # For non-callable attributes: treat like a property.
        if not callable(attrs[0]):
            return attrs

        # Bind everything with a self
        return self._call_proto(attrs, [[]] * len(attrs))

    def __dir__(self):
        """Retrieve and cache a list of interesting attributes.

        This method allows inspection and tab-completion of the attributes of
        a query object that depend on the type of ORM objects in the query.

        When the `dir()` function is called on a non-empty HWMQuery instance,
        a list is created of (1) attributes of the query, (2) attributes of
        the objects it contains, and (3) any macros or algorithms that are
        associated with the object type.  The list is created once and cached
        in the class-level HWMQuery._attribute_registry dictionary. In an
        interactive IPython session, this registry is automatically populated
        as a hardware map is constructed (loaded), and includes items for all
        ORM object types that are found in the hardware map.
        """

        # empty query
        if not len(self.column_descriptions):
            return dir(HWMQuery)

        # query that does not return ORM objects
        if not isinstance(self.column_descriptions[0]["type"], type):
            return dir(HWMQuery)

        # Check cache first
        for cls in self._attribute_registry:
            if issubclass(self.column_descriptions[0]["type"], cls):
                return self._attribute_registry[cls]

        s = set(dir(HWMQuery))
        s.update(dir(super()))

        # Add methods/properties from sub-objects
        empty = True
        for x in self:
            s.update(dir(x.__class__))
            s.update(dir(x))
            empty = False
            break

        # Add algorithms from the registry.
        for cls in self._algorithm_registry:
            if issubclass(self.column_descriptions[0]["type"], cls):
                s.update(list(self._algorithm_registry[cls]))

        # Add to cache
        s = list(s)
        s.sort()
        if not empty:
            HWMQuery._attribute_registry[self.column_descriptions[0]["type"]] = s

        return s

    def _call_proto(self, funcs, zip_args, *args, **kwargs):
        """Generate a function suitable for queueing a call and return it."""

        # Short-circuit for empty queries.
        if self.count() == 0:
            return lambda *x, **y: []

        async def acall(*sub_args, **sub_kwargs):
            # Note that we have two sets of (*args, **kwargs): the ones
            # provided directly to the prototype function, and the ones
            # given to _call_proto.  Ensure only one set is non-null.
            if args and sub_args:
                raise ValueError(f"Multiple argument lists! ({args}, {sub_args})")
            a = list(args or sub_args)

            if kwargs and sub_kwargs:
                raise ValueError(
                    f"Multiple argument dictionaries! ({kwargs}, {sub_kwargs})"
                )
            k = (kwargs or sub_kwargs).copy()

            return await asyncio.gather(
                *[f(*(za + a), **k) for f, za in zip(funcs, zip_args)]
            )

        return acall


class HWMResource(Base):
    """Base class for Hardware Mapper resources to share.

    You should inherit from this class in order to create a Hardware-Mapped
    resource. It's declared for convenience, since "Base" is an instantiated
    class (see the top of this file.)
    """

    __abstract__ = True

    @property
    def hwm(self):
        """Retrieve the :class:`HardwareMap` that stores this object."""
        return sqlalchemy.orm.object_session(self)

    def to_query(self):
        """Convert the HWMResource into a HWMQuery instance that contains this object"""
        hwm = self.hwm
        pk = sqlinspect(self.__class__).primary_key[0].key
        query = hwm.query(self.__class__).filter(
            getattr(self.__class__, pk) == getattr(self, pk)
        )
        query.one()
        return query

    def __dir__(self):
        """Retrieve a list of resource attributes"""

        s = set(dir(HWMResource))
        s.update(dir(super()))
        s.update(dir(self.__class__))

        if isinstance(self, tuber.TuberObject):
            s.update(tuber.TuberObject.__dir__(self))

        s = list(s)
        s.sort()

        return s


class macro:
    """Decorator for "macros" that performs some rudimentary typechecking.

    Macros are functions used with query objects that are parallelized
    directly, i.e.

    >>> @macro(ReadoutChannel)
    ... def print_channel(c):
    ...     print c.channel

    >>> hwm.query(ReadoutChannel).call_with(print_channel)

    The "print_channel" function ends up executing once for each
    ReadoutChannel in the query. See the "algorithm" decorator for an
    alternative.

    You do not need to use this macro to get this behaviour; it's the
    default case. We encourage use of the decorator anyway, for
    typechecking and to give context for the function being called.
    """

    def __init__(self, cls, register=False):
        # Accept either a class or a list of classes
        try:
            iter(cls)
        except TypeError:
            cls = (cls,)

        self._valid_classes = cls
        self._register = register

    def __call__(self, func):

        if not asyncio.iscoroutinefunction(func):
            raise RuntimeError(
                "The @macro decorator is intended for 'async def' functions only!"
            )

        vcs = self._valid_classes

        @functools.wraps(func)
        async def acall(obj, *args, **kwargs):

            # Check that the macro was called with an allowed class
            if vcs and not issubclass(obj.__class__, vcs):
                raise TypeError(
                    "Macro called with wrong types! Expected %s, got %s"
                    % (", ".join([cls.__name__ for cls in vcs]), obj.__class__)
                )

            # Say something about the call
            l = logging.getLogger(__name__)
            l.debug(f"{obj}: Invoking {func.__name__}(...)")

            # Invoke the macro.
            return await func(obj, *args, **kwargs)

        # Register this algorithm with the class it's used on.
        if self._register:
            for vc in vcs:
                setattr(vc, func.__name__, acall)

        return acall


class algorithm:
    """Decorator for "algorithms".

    This decorator is used as follows:

    >>> @algorithm(ReadoutChannel)
    ... async def print_channel(cs):
    ...     print cs.channel

    >>> hwm.query(ReadoutChannel).call_with(print_channel)

    The "print_channel" function is called *once*, and is passed in a HWMQuery
    of ReadoutChannels that can be iterated over. When we bump into the
    "print" statement, an array of integers is assembled.
    """

    def __init__(dec, cls, register=True):
        # Accept either a class or a list of classes
        try:
            iter(cls)
        except TypeError:
            cls = (cls,)

        dec.__valid_classes = cls
        dec.__register = register

    def __call__(dec, func):

        name = func.__name__

        if not asyncio.iscoroutinefunction(func):
            raise RuntimeError(
                "The @algorithm decorator is intended for "
                "'async def' functions only!"
            )

        async def acall(objs, *args, **kwargs):

            vcs = dec.__valid_classes
            if vcs and not all([issubclass(x.__class__, vcs) for x in objs]):
                raise TypeError(
                    "Algorithm called with wrong types! "
                    "Expected %s, got %s"
                    % (
                        ", ".join([cls.__name__ for cls in vcs]),
                        ", ".join(set([x.__class__.__name__ for x in objs])),
                    )
                )

            return await func(objs, *args, **kwargs)

        # Register this algorithm with the HWMQuery.
        if dec.__register:
            for vc in dec.__valid_classes:
                try:
                    HWMQuery._algorithm_registry[vc][name] = acall
                except KeyError:
                    HWMQuery._algorithm_registry[vc] = {name: acall}

        return acall


class Boolean(sqlalchemy.types.TypeDecorator):
    """A Boolean lookalike for column definitions, accepting "true"/"false".

    Use this instead of sqlalchemy.Boolean when the column might be initialized
    with string values, e.g. from CSV entries.
    """

    impl = sqlalchemy.types.Boolean

    def process_bind_param(self, value, dialect):
        if isinstance(value, str):
            if value.lower() == "true":
                return True
            if value.lower() == "false":
                return False
        return value


class Session(sqlalchemy.orm.Session):
    """Subclass SQLAlchemy's 'Session' object.

    This subclass is not used here, but it provides a way for experiment code
    (dfmux.py) to extend it."""


def HardwareMap(uri="sqlite:///:memory:", echo=False, data=None, *args, **kwargs):
    """Create a HardwareMap object (which is really a SQLAlchemy Session).

    By default, this function creates an empty in-memory SQLAlchemy session.

    If the input URI points to an existing hardware map file on disk, this will
    load the hardware map into an in-memory SQLALchemy session, and close the
    original connection to the on-disk database.  Thus, changes to the active
    session will *not* be stored to disk.  Use `hwm.dump()` to write the changed
    hardware map to disk manually.

    An IOError is raised if the input URI points to a hardware map file path
    that does not exist.

    If the `data` argument is supplied, it is assumed to be the output of a
    call to `hwm.dumps()` and is loaded into the database on connect.
    """

    path = uri.replace("sqlite:///", "")
    if not path.startswith(":memory:") and not os.path.exists(path):
        raise IOError(f"Cannot find hardware map database {path}")

    # connection function that retries connections to the database
    # a few times with exponential backoff
    def connect():
        delay = 0.01
        tries = 10
        for i in range(tries):
            try:
                # copy database from file into memory
                if not path.startswith(":memory:"):
                    db = sqlite3.connect(path)
                    q = "".join([line for line in db.iterdump()])
                    db.close()
                    conn = sqlite3.connect(":memory:")
                    conn.executescript(q)
                else:
                    conn = sqlite3.connect(path)
                    if data is not None:
                        conn.executescript(data)
                return conn
            except sqlite3.OperationalError as e:
                if i < tries - 1:
                    logging.exception(
                        f"Caught OperationalError on connect, "
                        f"retrying {i+1}/{tries}"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e

    # Connect to the database
    if ":memory:" not in path:
        # always use an in-memory connection, having pre-loaded the database
        # from disk if necessary above.
        uri = "sqlite:///:memory:"
    e = sqlalchemy.create_engine(uri, echo=echo, creator=connect)

    # It's possible we're operating on an empty, in-memory database
    # (that's one of the use cases we anticipate) -- so ensure all
    # of the relevant tables have been created.
    Base.metadata.create_all(e)

    return sqlalchemy.orm.scoped_session(
        sqlalchemy.orm.sessionmaker(
            bind=e, query_cls=HWMQuery, class_=Session, *args, **kwargs
        )
    )


# vim: sts=4 ts=4 sw=4 tw=78 smarttab expandtab
