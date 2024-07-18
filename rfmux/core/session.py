"""
YAML session loader
~~~~~~~~~~~~~~~~~~~

Using this code, we can create a hardware map and associated data.
"""

__all__ = [
    "HWMCSVConstructor",
    "HWMConstructor",
    "IncludedYAMLValue",
    "YAMLLoader",
    "load_from_database",
    "load_session",
    "read_csv",
    "set_yaml_loader_class",
]

import yaml
import json
import csv
import os
import time
import mimetypes
import logging
from . import hardware_map
import re


def read_csv(filename):
    with open(filename, "r") as f:
        # First pass: look for fields containing spaces and ensure they're
        # quoted.
        for l, line in enumerate(f):
            if line.startswith("#"):
                continue
            for field in re.findall(r"([^\t]* [^\t]*)", line.rstrip("\n")):
                if field[0] != '"' or field[-1] != '"':
                    raise ValueError(
                        f"File {filename} (line {l+1}). Unescaped space in field: "
                        f"'{field}'. Spaces are only legal in fields that begin and "
                        f"end with double-quote blocks. Fields should be "
                        f"tab-delimited."
                    )

        # Second pass: Actually read the file
        f.seek(0)
        dr = csv.DictReader(
            [row for row in f if not row.startswith("#")], dialect="excel-tab"
        )
        dr = list(dr)

        # Third pass: remove empty entries
        for row in dr:
            for k, v in list(row.items()):
                if v == "" or v is None:
                    row.pop(k)

    return dr


class HWMCSVConstructor:
    """When parsing a !tagged CSV filename, generate instances of 'cls'.

    The arguments to 'cls' are taken from columns in the CSV, after applying
    any transforms. For a description of transforms, see the
    HWMConstructor DocStrings.
    """

    def __init__(self, constructor, *transforms):
        self._constructor = constructor
        self._transforms = transforms

    def __call__(self, loader, node):
        fn = os.path.join(os.path.dirname(loader.name), node.value)

        classes = []

        dr = read_csv(fn)
        for m in dr:
            for t in self._transforms:
                t(loader, m)
            c = self._constructor(loader)(**m)
            classes.append(c)

        if hasattr(loader, "hwm"):
            loader.hwm.add_all(classes)

        return classes


class HWMConstructor:
    """When parsing a !tagged YAML dictionary, generate an instance of 'cls'.

    The arguments to 'cls' are taken from the YAML dictionary, after applying
    any transforms. Transforms are selected by keys in the YAML dictionary, and
    are functions of the form:

        def transform(x):
            x['foo'] = x['foo'] + 1

    This transform cause the YAML

        !some_tag { foo: 1 }

    to be instantiated as 'cls(foo=2)'. We use this to work around impedance
    mismatches between sensibly serialized HWM and the ORM.
    """

    def __init__(self, constructor, *transforms):
        self._constructor = constructor
        self._transforms = transforms

    def __call__(self, loader, node):
        # 'deep=True' is crucial and pretty much undocumented.
        m = loader.construct_mapping(node, deep=True)

        # Apply transformations, if any.
        for t in self._transforms:
            t(loader, m)

        c = self._constructor(loader)(**m)

        if hasattr(loader, "hwm"):
            loader.hwm.add(c)

        return c


class IncludedYAMLValue:
    """Container for YAML data coming from an !include snippet.

    There are two intended uses for this class:

    1. To allow structured, human-written HardwareMaps, by allowing one
       file to include another. In this mode, the save() method should
       not be used since it strips formatting from the file (e.g.
       comments).

    2. To allow bits of the hardware map to be updated by software,
       using the save() method. These files don't contain comments and
       can be reformatted and rewritten. The save() method does this.
    """

    def __new__(cls, filename, value):
        """Produce an IncludedYAMLValue class with the correct inheritence.

        Note there are two IncludedYAMLValue classes; the one above is
        user-facing, but a thin wrapper around the one below.
        """

        class IncludedYAMLValue(value.__class__):
            __doc__ = cls.__doc__  # preserve DocStrings

            def __repr__(self):
                return "IncludedYAMLValue(%r)" % value

            @property
            def filename(self):
                return filename

            def save(self):
                """Save the data contained in this file back to JSON/YAML.

                Note that comments (and other unparsed information) are
                clobbered. This method is not suitable for round-tripping
                complete HWMs.
                """
                with open(self.filename, "w") as f:

                    mimetype = mimetypes.guess_type(self.filename[0])

                    if mimetype == "application/json":
                        json.dump(value.__class__(self), f, indent=4)
                    else:
                        yaml.dump(
                            value.__class__(self), f, indent=4, default_flow_style=False
                        )

        # Instantiate and return an IncludedYAMLValue with the right value.
        return IncludedYAMLValue(value)


def yaml_include_constructor(loader, node):
    """!include tag for YAML.

    Since YAML is a superset of JSON, this tag works for JSON data too.
    """

    try:
        o = loader.construct_mapping(node, deep=True)
        retval = IncludedYAMLValue(node.value, o)

    except yaml.loader.ConstructorError:

        fn = os.path.join(os.path.dirname(loader.name), node.value)
        with open(fn) as f:
            o = yaml.safe_load(f)
        retval = IncludedYAMLValue(fn, o)

    if "save_database" in o:
        dbfile = o["save_database"]
        if not os.path.isabs(dbfile):
            dbfile = os.path.join(os.path.dirname(loader.name), dbfile)
        loader.database_path = dbfile
    # Instantiate an instance of the wrapper class with the correct value and
    # return it.
    return retval


def hwm_constructor(loader, node):
    """A YAML !HardwareMap constructor for core.HardwareMap objects.

    The node annotated with this tag must be an array of objects to be added.
    """

    # Because parts of the YAML script can query the HWM, we need to populate
    # the HWM greedily rather than building a list of objects and adding them
    # all at the end. To do so, we embed a HWM in the loader and add elements
    # as they're built.
    loader.hwm = hardware_map.HardwareMap()

    # 'deep=True' is crucial and pretty much undocumented.
    loader.construct_sequence(node, deep=True)

    return loader.hwm


def hwm_lookup_constructor(loader, node):
    """Permit YAML to make direct HWM accesses."""

    s = loader.construct_sequence(node)
    obj = s.pop(0)

    for lookup in s:
        if not isinstance(lookup, dict) or len(lookup) != 1:
            raise yaml.YAMLError(
                "HWM lookups require single key:value pairs!" "Got %r" % lookup
            )

        key = lookup.keys()[0]
        value = lookup[key]

        # Do an ORM lookup.
        try:
            obj = getattr(obj, key)[value]
        except KeyError:
            raise KeyError("%r has no %s %s" % (obj, key, value))

    return obj


class YAMLLoader(yaml.SafeLoader):

    # We allow some hooks to be queued when the document
    # is finished loading.
    __finalize_hooks = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Plumbing
        self.add_constructor("!include", yaml_include_constructor)
        self.add_constructor("!HardwareMap", hwm_constructor)
        self.add_constructor("!HWMLookup", hwm_lookup_constructor)

    def register_finalization_hook(self, hook):
        self.__finalize_hooks.append(hook)

    def construct_document(self, *args, **kwargs):
        """
        Make sure SQL database is up to date when the loader is finished.
        """
        t1 = time.time()
        data = super().construct_document(*args, **kwargs)
        # Write any pending transactions to the database
        if hasattr(self, "hwm"):
            self.hwm.commit()
            # Store constructed HWM to disk
            if hasattr(self, "database_path"):
                self.hwm.dump(self.database_path, backup=True)

        # Fire any callbacks
        for hook in self.__finalize_hooks:
            hook(getattr(self, "hwm", None))

        t2 = time.time()
        if hasattr(self, "hwm"):
            logging.info("Hardware map loaded in %.2f seconds" % (t2 - t1))

        return data


def set_yaml_loader_class(cls):
    """Override the YAMLLoader class used to create sessions."""

    global __yaml_loader_class
    __yaml_loader_class = cls


__yaml_loader_class = YAMLLoader


def load_session(stream, store=True):
    """Load a YAML document into a Session object."""
    y = yaml.load(stream, Loader=__yaml_loader_class)
    if hasattr(stream, "name") and "hardware_map" in y:
        y["hwm_dir"] = os.path.abspath(stream.name)
    if store:
        set_session(y)
    return y


def set_session(session):
    """Store a Session object somewhere it's globally accessible."""
    global __session_handle
    __session_handle = session


def get_session():
    """Retrieve the session stored via set_session()"""
    return __session_handle


__session_handle = None


def load_from_database(path):
    """
    To use within yaml hwm:
    output_database : !include
        save_database : '/tmp/rfmux.db'
    """

    hwm = hardware_map.HardwareMap("sqlite:///{path}")

    # preload attributes for tab-completion in interactive sessions
    hwm.cache_attributes()

    return hwm


# vim: sts=4 ts=4 sw=4 tw=80 smarttab expandtab
