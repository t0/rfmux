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
    "ChannelMapping"
]

from .hardware_map import algorithm
from .schema import Crate, CRS, ReadoutModule, ReadoutChannel
from .session import HWMConstructor


import sqlalchemy

from packaging import version
assert version.parse(sqlalchemy.__version__) >= version.parse('1.2')

from . import session

import sys
import logging


logger = logging.getLogger()

# A list of all branches of the ORM
orm_joins = [
    [Crate, CRS, ReadoutModule, ReadoutChannel],
]
# A list of all of the types in the ORM
orm_types = tuple(set().union(*orm_joins))
_join_cache = {}



class YAMLLoader(session.YAMLLoader):

    # This is the default flavour
    flavour = sys.modules[__name__]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # Readout chain
        self.add_constructor(
            u"!Crate",
            HWMConstructor(
                lambda loader: getattr(loader.flavour, "Crate"),
                AttributeMappingTouchup("slots", "slot"),
            ),
        )

        self.add_constructor(
            u"!CRS",
            HWMConstructor(
                lambda loader: getattr(loader.flavour, "CRS")
            ),
        )


def _get_joins(query_type, obj_type):
    """
    Return the list of joins between the query type and the object type.
    """
    cache_key = (query_type, obj_type)

    # check cache first
    if cache_key in _join_cache:
        return list(_join_cache[cache_key]) # ensure copy
    else:
        inv_key = (obj_type, query_type)
        if inv_key in _join_cache:
            return list(_join_cache[inv_key])[::-1] # ensure copy

    # Find join path indices of the source query and requested object type
    query_index = []
    obj_index = []
    for i, path in enumerate(orm_joins):
        for j, tp in enumerate(path):
            if issubclass(query_type, tp):
                query_index += [(i, j)]
            if issubclass(obj_type, tp):
                obj_index += [(i, j)]

    # Raise errors if either type is not in the list
    if not len(query_index):
        raise TypeError("Unrecognized query type %s" % (str(query_type)))
    if not len(obj_index):
        raise TypeError("Unrecognized object type %s" % (str(obj_type)))

    # Requested type is the same as the query type
    if query_index == obj_index:
        _join_cache[cache_key] = []
        return []

    # Try to find a path between types on the same branch
    if len(query_index) > 1 or len(obj_index) > 1:
        qb = [x[0] for x in query_index]
        ob = [x[0] for x in obj_index]
        common_branches = set(qb) & set(ob)
        if len(common_branches):
            # Use the first common path
            branch = sorted(common_branches)[0]
            query_index = [query_index[qb.index(branch)]]
            obj_index = [obj_index[ob.index(branch)]]

    # Use the first path found
    query_index = query_index[0]
    obj_index = obj_index[0]

    # Construct the join list between query and object types
    if query_index[0] == obj_index[0]:
        # Requested type is on the same join branch as the query type,
        # so we can just slice the branch to get the join list
        branch = orm_joins[query_index[0]]
        if query_index[1] < obj_index[1]:
            joins = branch[query_index[1] + 1:obj_index[1]][::-1]
        else:
            joins = branch[obj_index[1] + 1:query_index[1]]
    else:
        # Requested type is on a different branch, so we merge branches
        # to create the join list.  The last entry in each branch is
        # the same (ChannelMapping), so we merge branches by appending
        # the items in the query branch in reverse order to the object
        # branch.
        obj_branch = orm_joins[obj_index[0]]
        joins = obj_branch[obj_index[1] + 1:]
        query_branch = orm_joins[query_index[0]]
        joins += query_branch[query_index[1] + 1:-1][::-1]

    _join_cache[cache_key] = list(joins) # ensure copy
    return joins


@algorithm(orm_types, register=True, allow_async=False)
def get_objects(query, obj_type):
    """
    Return a subquery of the requested object type that join with
    any of the entries in this query.  The returned query is distinct,
    meaning that all items in the query are unique.
    """

    # Get input query type
    query_type = query.column_descriptions[0]['type']

    # Requested type is the same as the query type
    if query_type == obj_type:
        return query

    # Construct the join list
    joins = _get_joins(query_type, obj_type)

    # Construct the output query by joining a query of all objects of
    # the requested type with the objects in this query
    q = query.hwm.query(obj_type)
    for j in joins:
        q = q.join(j)
    return q.join(query.subquery()).distinct()

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
                for (index, value) in enumerate(values):
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
                for (key, value) in values.items():
                    if not value:
                        continue
                    setattr(value, self._member_attribute, key)

                mapping[self._group_attribute] = [v for v in values.values() if v]

            else:
                raise TypeError("Expected a list, got '%r'!" % values)


# Tell session to use our YAMLLoader.
session.set_yaml_loader_class(YAMLLoader)

# vim: sts=4 ts=4 sw=4 tw=78 smarttab expandtab
