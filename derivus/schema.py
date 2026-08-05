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

"""Per-class field declarations, and the `fields.mapping` view emitted from them.

`fields.py` keys one descriptor per field NAME by convention, so two deals needing different valid
values for the same field must invent a key and carry the real name elsewhere - the 21 entries in
`ALIASED_KEYS`. A class that owns its own fields has no such collision.

The schema's inheritance is composition of named field GROUPS, not the class hierarchy: `FXAdmin`
is shared by eight deals with no common base, and `Admin` by all 47. Groups are therefore ordinary
module-level constants a class lists, not something recovered from the MRO.

Nothing here is read at valuation time. `construct_instrument` takes the raw JSON and `Deal.__init__`
stores it unfiltered, so this is authoring-time metadata: the UI, the docs generator and the Excel
add-in. `emit_instrument()` reproduces `fields.mapping` exactly so none of them need to change.
"""


#: `default=REQUIRED` - the author must supply it. Distinct from a default of None, which is a
#: field the engine reads with `.get` and is content to find missing.
REQUIRED = object()

#: How `fields.mapping` renders each type for Handsontable. Rendering only: derived on the way out
#: and never declared, which is the point - a front end that is not Handsontable ignores all of it.
WIDGET_FORMAT = {
    'Date': {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
    'Float': {'type': 'numeric', 'numericFormat': {'pattern': '0,0.00'}},
    'Basis': {'type': 'numeric', 'numericFormat': {'pattern': '0,0.00'}},
    'Percent': {'type': 'numeric', 'numericFormat': {'pattern': '0.00 %'}},
    'Integer': {'type': 'numeric', 'numericFormat': {'pattern': '0.'}},
    'Text': {}, 'Period': {}, 'Table': {},
}

#: The `obj` token `fields.mapping` uses per type, for a table declaring its columns positionally.
OBJ_TOKEN = {'Date': 'DatePicker', 'Float': 'Float', 'Integer': 'Integer', 'Text': 'Text',
              'Percent': 'Percent', 'Basis': 'Basis', 'Period': 'Period', 'Table': 'ResetArray'}


class Row(object):
    """The ordered fields of one table row.

    A table's columns were two parallel lists - names in `col_names`, rendering specs in
    `sub_types` - matched by position, with no per-column default, type or required-ness. That is
    why a column could not be declared nullable and why 21 columns could sit in a table of which
    four are read: there was nothing to declare them ON.
    """
    __slots__ = ('fields',)

    def __init__(self, fields):
        self.fields = list(fields)


class F(object):
    """One field of one deal.

    `type` is semantic (Text/Float/Integer/Date/Percent/Basis/Period/Table/Container); the WIDGET
    name is the front end's business and is only reintroduced when emitting `fields.mapping`. A
    choice list is not a type - it is a Text whose `values` are a fixed set, and the dropdown falls
    out of that.
    `json_name` is the escape hatch the name-keyed dict needed constantly and a per-class list
    needs almost never - the cashflow shapes that genuinely share a JSON key.

    A Table declares its columns as a `Row`; `tag` names the utils container the wire form uses
    (`DateList`, `DateValueList`, `CreditSupportList`), absent for a plain array of rows.
    """
    WIDGET = {'Text': 'Text', 'Float': 'Float', 'Integer': 'Integer', 'Date': 'DatePicker',
              'Percent': 'Float', 'Basis': 'Float', 'Period': 'Text',
              'Table': 'Table', 'Container': 'Container'}

    __slots__ = ('name', 'type', 'default', 'description', 'values', 'row', 'tag',
                 'sub_fields', 'json_name', 'obj')

    def __init__(self, name, type, default=None, description=None, values=None, row=None,
                 tag=None, sub_fields=None, json_name=None, obj=None):
        self.name = name
        self.type = type
        self.default = default
        # the UI recovers the JSON key from the description (`description.replace(' ', '_')`), so
        # it is a key as well as a label - derive it rather than let the two drift
        self.description = description if description is not None else name.replace('_', ' ')
        self.values = values
        self.row = row
        self.tag = tag
        self.sub_fields = sub_fields
        self.json_name = json_name
        # parse token on SCALARS only ('Tuple' = a dotted factor reference); tables no longer
        # carry it, `row` and `tag` say it properly
        self.obj = obj

    @property
    def key(self):
        """The key `fields.mapping` files this descriptor under - the alias when there is one."""
        return self.name

    def descriptor(self):
        """This field as a `fields.mapping[...]['fields']` entry.

        `col_names`, `sub_types` and `obj` are DERIVED from the row rather than stored: they are
        Handsontable's rendering vocabulary, and a front end that is not Handsontable wants none
        of it. Deriving them is also what stops the two parallel lists drifting apart.
        """
        widget = 'Dropdown' if self.values is not None else self.WIDGET[self.type]
        d = {'widget': widget, 'description': self.description, 'value': self.default}
        if self.json_name is not None:
            d = {'name': self.json_name, **d}
        if self.values is not None:
            d['values'] = self.values
        if self.obj is not None:
            d['obj'] = self.obj
        if self.row is not None:
            d['obj'] = self.tag if self.tag else [OBJ_TOKEN[f.type] for f in self.row.fields]
            d['col_names'] = [f.name for f in self.row.fields]
            d['sub_types'] = [{'type': 'dropdown', 'source': f.values} if f.values is not None
                              else dict(WIDGET_FORMAT[f.type]) for f in self.row.fields]
        if self.sub_fields is not None:
            d['sub_fields'] = [f.key for f in self.sub_fields]
        return d


class Group(object):
    """A named, reusable block of fields - what `fields.mapping` calls a section.

    Shared blocks (`Admin`, `FXAdmin`) are module-level constants; a class's own block is named
    `<ClassName>.Fields` by convention and built by `own()`.
    """
    __slots__ = ('name', 'fields')

    def __init__(self, name, fields):
        self.name = name
        self.fields = list(fields)


def own(cls_name, fields, role='Fields'):
    """A class's own block, named the way `fields.mapping`'s sections are."""
    return Group('{}.{}'.format(cls_name, role), fields)


def emit_instrument(module):
    """The `types` / `sections` / `fields` of `fields.mapping['Instrument']`, from the classes.

    Scans `module` for classes declaring their own `fields` list. Own-attr only, so a subclass that
    inherits its parent's declaration does not re-emit it as a second deal type. Declaration order
    is preserved: the UI lays panels out in `types[T]` order and widgets within a panel in section
    order, and a set-backed view scrambles both silently, with no exception.
    """
    def register(f, fields):
        # a container's children and a table's columns are fields in their own right; the flat
        # store needs them by key, which is also why an alias like FloatItems has to exist there
        fields.setdefault(f.key, f.descriptor())
        for child in (f.sub_fields or []):
            register(child, fields)

    types, sections, fields = {}, {}, {}
    for deal_type, cls in vars(module).items():
        groups = cls.__dict__.get('fields') if isinstance(cls, type) else None
        if not isinstance(groups, list):
            continue
        types[deal_type] = [g.name for g in groups]
        for g in groups:
            sections.setdefault(g.name, [f.key for f in g.fields])
            for f in g.fields:
                register(f, fields)
    return types, sections, fields
