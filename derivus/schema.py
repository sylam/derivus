########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of Derivus.
#
# Derivus is free for noncommercial use under the terms of the PolyForm
# Noncommercial License 1.0.0. You should have received a copy of the license
# along with Derivus. If not, see
# <https://polyformproject.org/licenses/noncommercial/1.0.0>.
#
# Derivus is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
########################################################################

"""Per-class field declarations, and the legacy `fields.mapping` view emitted from them.

`fields.py` keys one descriptor per field NAME by convention, so two deals needing different valid
values for the same field must invent a key and carry the real name elsewhere - the 21 entries in
`ALIASED_KEYS`. A class that owns its own fields has no such collision.

The schema's inheritance is composition of named field GROUPS, not the class hierarchy: `FXAdmin`
is shared by eight deals with no common base, and `Admin` by all 47. Groups are therefore ordinary
module-level constants a class lists, not something recovered from the MRO.

Nothing here is read at valuation time. `construct_instrument` takes the raw JSON and `Deal.__init__`
stores it unfiltered, so this is authoring-time metadata: the UI, the docs generator and the Excel
add-in. `emit_instrument()` reproduces the legacy shape exactly so none of them need to change.
"""


class F(object):
    """One field of one deal.

    `type` is semantic (Text/Float/Integer/Date/Choice/Table/Container); the WIDGET name is the
    front end's business and is only reintroduced by the legacy emitter. `json_name` is the escape
    hatch the name-keyed dict needed constantly and a per-class list needs almost never - the two
    cashflow shapes that genuinely share a JSON key.
    """
    WIDGET = {'Text': 'Text', 'Float': 'Float', 'Integer': 'Integer', 'Date': 'DatePicker',
              'Choice': 'Dropdown', 'Table': 'Table', 'Container': 'Container'}

    __slots__ = ('name', 'type', 'default', 'description', 'values', 'obj', 'columns',
                 'column_types', 'sub_fields', 'json_name')

    def __init__(self, name, type, default=None, description=None, values=None, obj=None,
                 columns=None, column_types=None, sub_fields=None, json_name=None):
        self.name = name
        self.type = type
        self.default = default
        # the UI recovers the JSON key from the description (`description.replace(' ', '_')`), so
        # it is a key as well as a label - derive it rather than let the two drift
        self.description = description if description is not None else name.replace('_', ' ')
        self.values = values
        self.obj = obj
        self.columns = columns
        self.column_types = column_types
        self.sub_fields = sub_fields
        self.json_name = json_name

    @property
    def key(self):
        """The descriptor key the legacy store uses - the alias when there is one."""
        return self.name

    def descriptor(self):
        """This field as a legacy `fields.mapping[...]['fields']` entry."""
        d = {'widget': self.WIDGET[self.type], 'description': self.description,
             'value': self.default}
        if self.json_name is not None:
            d = {'name': self.json_name, **d}
        if self.values is not None:
            d['values'] = self.values
        if self.obj is not None:
            d['obj'] = self.obj
        if self.columns is not None:
            d['col_names'] = self.columns
        if self.column_types is not None:
            d['sub_types'] = self.column_types
        if self.sub_fields is not None:
            d['sub_fields'] = self.sub_fields
        return d


class Group(object):
    """A named, reusable block of fields - what the legacy schema called a section.

    Shared blocks (`Admin`, `FXAdmin`) are module-level constants; a class's own block is named
    `<ClassName>.Fields` by convention and built by `own()`.
    """
    __slots__ = ('name', 'fields')

    def __init__(self, name, fields):
        self.name = name
        self.fields = list(fields)


def own(cls_name, fields, role='Fields'):
    """A class's own block, named the way the legacy sections are."""
    return Group('{}.{}'.format(cls_name, role), fields)


def emit_instrument(declared):
    """The legacy `mapping['Instrument']` sub-tree, rebuilt from per-class declarations.

    `declared` maps deal type -> ordered list of Groups. Returns (types, sections, fields) with
    declaration order preserved: the UI lays panels out in `types[T]` order and widgets within a
    panel in section order, and a set-backed view scrambles both silently.
    """
    types, sections, fields = {}, {}, {}
    for deal_type, groups in declared.items():
        types[deal_type] = [g.name for g in groups]
        for g in groups:
            sections.setdefault(g.name, [f.key for f in g.fields])
            for f in g.fields:
                fields.setdefault(f.key, f.descriptor())
    return types, sections, fields
