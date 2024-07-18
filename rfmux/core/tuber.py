"""
Tuber object interface
"""

import asyncio
import aiohttp
import textwrap
import types

# Simplejson is now mandatory (it's faster enough to insist)
import simplejson

__all__ = [
    "TuberError",
    "TuberRemoteError",
    "TuberCategory",
    "TuberObject",
]


class TuberError(Exception):
    pass


class TuberStateError(TuberError):
    pass


class TuberNetworkError(TuberError):
    pass


class TuberRemoteError(TuberError):
    pass


class TuberResult:
    def __init__(self, d):
        self.__dict__.update(d)

    def __iter__(self):
        return iter(self.__dict__.values())

    def __repr__(self):
        "Return a nicely formatted representation string"
        return "TuberResult({0})".format(
            ",".join(
                "{0}={1!r}".format(name, val) for name, val in self.__dict__.items()
            )
        )


def valid_dynamic_attr(name):
    """
    Return True if the input attribute name can be assigned to a dynamically
    created attribute (i.e. an attribute that would be returned by a call to
    `_tuber_get_meta()`), False otherwise.  See documentation of the
    `TuberObject.__getattr__` method for a discussion of why this function
    is necessary.
    """
    # These are mostly hints for SQLAlchemy or IPython.

    if name.startswith(
        (
            "__",
            "_sa",
            "_tuber",
            "_repr",
            "_ipython",
            "_orm",
            "_calls",
        )
    ):
        return False

    if name in {"trait_names", "_getAttributeNames", "getdoc"}:
        return False

    return True


class Context:
    """A context container for TuberCalls. Permits calls to be aggregated.

    Using this interface, you can write code like:

        #>>> from rfmux import CRS, macro, asynchronously, Return

        #>>> @rfmux.macro(CRS)
        #... def my_macro(d):
        #...     with d.tuber_context() as ctx:
        #...         p = ctx.get_mezzanine_power(2)
        #...         f = yield ctx.get_frequency(ctx.UNITS.HZ, 1, 1, 1)
        #...         ctx.set_frequency(f+1, ctx.UNITS.HZ, 1, 1, 1)
        #...         yield asynchronously(ctx)
        #...     raise Return((yield p))

        #>>> my_macro(d)
        #True

    Commands are dispatched to the board strictly in-order, but are
    automatically bundled up to reduce traffic. In this example, the first two
    calls are dispatched together, since the result 'p' is not used until
    later.

    There are a couple of important considerations:

        * Calls made on 'ctx' return Futures, which can be converted into
          their results via 'yield'.

        * The final 'yield asynchronously(ctx)' ensures the context queue is
          flushed. It's only necessary if you don't yield the results of the
          final call in the context. If you need this yield and leave it out,
          you'll see a warning and your code will dispatch synchronously (i.e.
          other work in the asynchronous framework won't get done in the
          meantime.)

    Note that you will *not* catch exceptions unless you check for them
    explicitly. For example, in this code:

        #>>> with d.tuber_context() as ctx:
        #...     p = ctx.get_mezzanine_power(3) # TODO: Better example

    the function call is executed, and generates an exception. The exception is
    embedded in 'p' and will not be raised unless you call 'p.result()'!

    Adjust the `connect_timeout` and `request_timeout` attributes of the ctx
    object to change the connect and request timeouts. The default value
    (1800 seconds) allows calls to the board to be quite slow.
    """

    def __init__(self, obj, **ctx_kwargs):
        self.calls = []
        self.connect_timeout = (
            1800  # TODO: these are too long. Come up with something rational here.
        )
        self.request_timeout = 1800
        self.obj = obj
        self.ctx_kwargs = ctx_kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.calls:
            self()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure the context is flushed."""

        if self.calls:
            await self()

    def _add_call(self, **request):

        future = asyncio.Future()

        # Ensure call is made in the correct context
        objname = request.setdefault("object", self.obj.tuber_objname)
        assert (
            objname == self.obj.tuber_objname
        ), f"Got call to {objname} in context for {self.obj.tuber_objname}"

        self.calls.append((request, future))

        return future

    async def __call__(self):
        """Break off a set of calls and return them for execution."""

        calls = []
        futures = []
        while self.calls:
            (c, f) = self.calls.pop(0)

            calls.append(c)
            futures.append(f)

        if calls:
            # initialize the client session
            # Configure the max clients to be 4 per TuberObject.
            loop = asyncio.get_running_loop()
            if not hasattr(loop, "_tuber_session"):
                connector = aiohttp.TCPConnector(
                    limit=0, limit_per_host=4
                )  # TODO: is this the right limit for CRS boards?
                loop._tuber_session = aiohttp.ClientSession(
                    json_serialize=simplejson.dumps,
                    connector=connector,
                    headers={"Accept": "application/json"},
                )

                # Ensure that ClientSession.close() is called when the loop is
                # closed.  ClientSession.__del__ does not close the session, so it
                # is not sufficient to simply attach the session to the loop to
                # ensure garbage collection.
                loop_close = loop.close

                def close(self):
                    if hasattr(self, "_tuber_session"):
                        if not self.is_closed():
                            self.run_until_complete(self._tuber_session.close())
                        del self._tuber_session
                    loop_close()

                loop.close = types.MethodType(close, loop)

            cs = loop._tuber_session

            # Create a HTTP request to complete the call. This is a coroutine,
            # so we queue the call and then suspend execution (via 'yield')
            # until it's complete.
            try:
                async with cs.post(self.obj.tuber_uri, json=calls) as resp:
                    json_out = await resp.json(
                        loads=simplejson.JSONDecoder(object_hook=TuberResult).decode,
                        content_type=None,
                    )
            except aiohttp.ClientConnectorError as e:
                raise TuberNetworkError(e)

            # Resolve futures
            results = []
            for f, r in zip(futures, json_out):
                if hasattr(r, "error") and r.error:
                    f.set_exception(TuberRemoteError(r.error.message))
                else:
                    results.append(r.result)
                    f.set_result(r.result)

            # Return a list of results
            return [await x for x in futures]

    def __getattr__(self, name):

        # Refuse to __getattr__ a couple of special names used elsewhere.
        if not valid_dynamic_attr(name):
            raise AttributeError("'%s' is not a valid method or property!" % name)

        # Queue methods calls.
        def caller(*args, **kwargs):

            # Add extra arguments where they're provided
            kwargs = kwargs.copy()
            kwargs.update(self.ctx_kwargs)

            # ensure that a new unique future is returned
            # each time this function is called
            future = self._add_call(method=name, args=args, kwargs=kwargs)

            return future

        setattr(self, name, caller)
        return caller


# TODO: Is this description still valid with the CRS?
class TuberCategory:
    """Pull Tuber functions into ORM objects based on categories.

    Here's an example. If "crs" is a CRS object, and you have the
    ordinary IceBoard function crs.set_timestamp_port(d.TIMESTAMP_PORT.TEST), you can run the
    following:

        >>> crs.d.set_timestamp_port(d.TIMESTAMP_PORT.TEST)

    You also have the following ORM object:

        >>> mod = crs.module[1]
        >>> print mod.module_number
        1

    Rather than accessing module methods through the CRS, it
    seems more logical to do things like this:

        >>> mod.set_dac_scale(scale, d.UNITS.DBM)

    The TuberCategory decorator allows this kind of call.
    Borrowing from FMCMezzanine again, we invoke the TuberCategory decorator as follows:

        @tuber.TuberCategory("Module", lambda m: m.crs,
            {"module": lambda m: m.module_number })
        class Module(HWMResource):
            [...]

    The decorator intercepts Tuber functions that claim to be members
    of the "Module" category, and using the Module object's
    module_number property, fills in "module" parameters before
    dispatching the function call back to the module's "crs"
    property.

    TODO: Update this example to one with an actual module field
    # "Categories" are exported by C code. For example, try the following:

    #     $ curl -d '{"object":"CRS","property":"set_timestamp_port"}' \
    #             http://crs004.local/tuber|json_pp

    # This shell command asks the CRS to describe its 'set_timestamp_port'
    # call. The response includes:

    # "result" : {
    #   "__doc__" : "set_timestamp_port(self: libmkids.Dfmux, port: TimestampPort) -> None"}
    """

    def __init__(self, category, getobject, **arg_mappers):
        self.category = category
        self.getobject = getobject
        self.arg_mappers = arg_mappers

    def __call__(decorator, cls):

        def tuber_context(self):
            obj = decorator.getobject(self)
            kwargs = {n: f(self) for (n, f) in decorator.arg_mappers.items()}
            return Context(obj, **kwargs)

        cls.tuber_context = tuber_context

        def __getattr__(self, name):
            """This is a fall-through replacement for __getattr__.

            We assume we're capturing a function call that's missing
            arguments. We fill in these arguments and dispatch the call.

            See `TuberObject.__getattr__` for details.
            """

            # Refuse to __getattr__ a couple of special names used elsewhere.
            if not valid_dynamic_attr(name):
                raise AttributeError("'%s' is not a valid method or property!" % name)

            parent = decorator.getobject(self)

            # Raise an Attribute error if the parent board isn't set
            if parent is None:
                raise AttributeError("'%s' is not a valid method or property!" % name)
            m = getattr(parent, name)

            async def acall(*args, **kwargs):
                mapped_args = {n: f(self) for (n, f) in decorator.arg_mappers.items()}

                return await m(*args, **kwargs, **mapped_args)

            return acall

        cls.__getattr__ = __getattr__

        def __dir__(self):
            """Retrieve a list of class properties/methods that are relevant.

            We try to grab the original list of attributes from the ORM,
            and then augment it with any Tuber functions in our category.

            See `TuberObject.__dir__` for more details.
            """

            d = set(dir(self.__class__))
            d.update(dir(super()))

            o = decorator.getobject(self)
            # Return just the static attributes if the board object isn't set
            # This can occur during construction of the ORM.
            if o is None:
                return sorted(d)
            (meta, metap, metam) = o._tuber_get_meta()

            for m in meta.methods:
                if (
                    hasattr(metam[m], "categories")
                    and decorator.category in metam[m].categories
                ):
                    d.add(m)
            return sorted(d)

        cls.__dir__ = __dir__

        return cls


class TuberObject:
    """A base class for TuberObjects.

    This is a great way of using Python to correspond with network resources
    over a HTTP tunnel. It hides most of the gory details and makes your
    networked resource look and behave like a local Python object.

    To use it, you should subclass this TuberObject.
    """

    _tuber_meta = {}
    _tuber_meta_properties = {}
    _tuber_meta_methods = {}

    def tuber_context(self):
        return Context(self)

    @property
    def tuber_uri(self):
        """Retrieve the URI associated with this TuberResource."""
        raise NotImplementedError("Subclass needs to define tuber_uri!")

    @property
    def tuber_objname(self):
        """Retrieve the Tuber Object associated with this TuberResource."""
        # TODO: Hack for now, to maintain compatibility with hidfmux the boards only
        # know "dfmux", even thought we are changing the ORM to 'CRS'. This will
        # be fixed in the future

        # return self.__class__.__name__
        return "Dfmux"

    @property
    def __doc__(self):
        """Construct DocStrings using metadata from the underlying resource."""

        (meta, _, _) = self._tuber_get_meta()

        return f"{meta.name}:\t{meta.summary}\n\n{meta.explanation}"

    def __dir__(self):
        """Provide a list of what's here. (Used for tab-completion.)

        This function calls the `_tuber_get_meta()` method to get a list of
        methods and properties stored on the board. If an error occurs in
        communicating with the board, then this function returns an empty list,
        and future calls to `_tuber_get_meta()` do not attempt to access the
        board.  Use the `set_tuber_inspect(True)` module-level function to
        re-enable communication with the board.
        """

        attrs = dir(super())
        (meta, _, _) = self._tuber_get_meta()

        return sorted(attrs + meta.properties + meta.methods)

    async def _tuber_get_meta(self):
        """Retrieve metadata associated with the remote network resource.

        This data isn't strictly needed to construct "blind" JSON-RPC calls,
        except for user-friendliness:

           * tab-completion requires knowledge of what the board does, and
           * docstrings are useful, but must be retrieved and attached.

        This class retrieves object-wide metadata, which can be used to build
        up properties and values (with tab-completion and docstrings)
        on-the-fly as they're needed.

        If tuber inspection has been disabled (either automatically when
        an error was encountered while calling this function in an attempt
        at tab-completion, or manually by calling `set_tuber_inspect(False)`),
        then this function returns empty lists.
        """

        if self.tuber_uri not in self._tuber_meta:
            async with self.tuber_context() as ctx:
                ctx._add_call()
                meta = await ctx()
                meta = meta[0]

                for p in meta.properties:
                    ctx._add_call(property=p)
                prop_list = await ctx()

                for m in meta.methods:
                    ctx._add_call(property=m)
                meth_list = await ctx()

                props = dict(zip(meta.properties, prop_list or []))
                methods = dict(zip(meta.methods, meth_list or []))

            self._tuber_meta_properties[self.tuber_uri] = props
            self._tuber_meta_methods[self.tuber_uri] = methods
            self._tuber_meta[self.tuber_uri] = meta

        return (
            self._tuber_meta[self.tuber_uri],
            self._tuber_meta_properties[self.tuber_uri],
            self._tuber_meta_methods[self.tuber_uri],
        )

    def __getattr__(self, name):
        """Remote function call magic.

        This function is called to get attributes (e.g. class variables and
        functions) that don't exist on "self". Since we build up a cache of
        descriptors for things we've seen before, we don't need to avoid
        round-trips to the board for metadata in the following code.

        This function is only called for attributes that aren't yet bound
        to the instance (e.g. in the class definition or set using `setattr`).
        Thus, when an attribute is requested that has not yet been bound,
        it is first filtered through the `valid_dynamic_attr()` function
        to determine whether it is likely that the attribute is one that
        comes from the board.  If not, an AttributeError is raised; this
        is important to avoid making tuber calls to the board for attributes
        that are not bound by construction, such as attributes that
        sqlalchemy or ipython checks for to determine how it should interact
        with a given object.  If the input name is not first checked with
        `valid_dynamic_attr()`, it is possible to trigger nasty recursion
        depth errors by trying to access an attribute that does not
        exist on the board.
        """

        # Refuse to __getattr__ a couple of special names used elsewhere.
        if not valid_dynamic_attr(name):
            raise AttributeError(f"'{name}' is not a valid method or property!")

        # Make sure this request corresponds to something in the underlying
        # TuberObject.
        try:
            (meta, metap, metam) = (
                self._tuber_meta[self.tuber_uri],
                self._tuber_meta_properties[self.tuber_uri],
                self._tuber_meta_methods[self.tuber_uri],
            )
        except KeyError as e:
            raise TuberStateError(
                e,
                "Attempt to retrieve metadata on TuberObject that doesn't have it yet! Did you forget to call resolve()?",
            )

        if name not in meta.methods and name not in meta.properties:
            raise AttributeError(f"'{name}' is not a valid method or property!")

        if name in meta.properties:
            # Fall back on properties.
            setattr(self, name, metap[name])
            return getattr(self, name)

        if name in meta.methods:
            # Generate a callable prototype
            async def invoke(self, *args, **kwargs):
                async with self.tuber_context() as ctx:
                    result = getattr(ctx, name)(*args, **kwargs)
                return result.result()

            # if hasattr(metam[name], "__doc__"):
            #    # When available, prefer direct annotations via '__doc__' to
            #    # overstructured metadata.
            #    invoke.__call__.__doc__ = metam[name].__doc__
            # else:
            #    invoke.__call__.__doc__ = textwrap.dedent(
            #        """
            #        {name}({args_short})

            #        {args_long}

            #        {explanation}"""
            #    ).format(
            #        name=name,
            #        args_short=", ".join([a.name for a in metam[name].args]),
            #        args_long="\n".join(
            #            [
            #                "    {:<16} {}".format(arg.name + ":", arg.description)
            #                for arg in metam[name].args
            #            ]
            #        ),
            #        explanation="\n".join(textwrap.wrap(metam[name].explanation)),
            #    )

            # Associate as a class method.
            setattr(self.__class__, name, invoke)
            return getattr(self, name)


# vim: sts=4 ts=4 sw=4 tw=78 smarttab expandtab
