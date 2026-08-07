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

A deal's schema is composition of named field GROUPS, not the class hierarchy: `FXAdmin` is shared
by eight deals with no common base, and `Admin` by all 47. Groups are therefore ordinary
module-level constants a class lists, not something recovered from the MRO. A price factor has no
such structure - its JSON block is one flat dict - so a factor class declares a flat list and
`emit_factor` files the descriptors per TYPE.

Nothing here is read at valuation time. `construct_instrument` takes the raw JSON and `Deal.__init__`
stores it unfiltered; `construct_factor` hands the raw block to the factor class. So this is
authoring-time metadata: the UI, the docs generator and the Excel add-in. A front end needs no other
source - a type's entry IS its descriptors, each keyed by the JSON key an author writes, so a panel,
its defaults and the write-back key all come from one lookup.
"""


#: `default=REQUIRED` - the author must supply it. Distinct from a default of None, which is a
#: field the engine reads with `.get` and is content to find missing.
REQUIRED = object()

#: The blank FORM of a shape-valued field. A curve or a surface has no empty value - the blank is
#: one degenerate knot - so these are what "no default" means for those types.
BLANK = {
    'Curve': '[{"label":"None", "data":[[0.0,0.0]]}]',
    'Surface': '[[0.0,1.0], [1.0,0.0]]',
    'Space': '{"0.0":[[0.0,0.0],[0.0,0.0]]}',
}

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
    """One field of one deal or one price factor.

    `type` is semantic (Text/Float/Integer/Date/Percent/Basis/Period/Table/Container, plus the
    market-data shapes Curve/Surface/Space); the WIDGET name is the front end's business and is
    only reintroduced when emitting `fields.mapping`. A choice list is not a type - it is a Text
    whose `values` are a fixed set, and the dropdown falls out of that.
    `json_name` is the escape hatch the name-keyed dict needed constantly and a per-class list
    needs almost never - the cashflow shapes that genuinely share a JSON key.

    A Table declares its columns as a `Row`; `tag` names the utils container the wire form uses
    (`DateList`, `DateValueList`, `CreditSupportList`), absent for a plain array of rows.

    `description` is free text: the JSON key is the key the descriptor is FILED under, so a front
    end reads it from the store rather than reconstructing it from a label.
    """
    WIDGET = {'Text': 'Text', 'Float': 'Float', 'Integer': 'Integer', 'Date': 'DatePicker',
              'Percent': 'Float', 'Basis': 'Float', 'Period': 'Text',
              'Table': 'Table', 'Container': 'Container',
              'Curve': 'Flot', 'Surface': 'Three', 'Space': 'Three'}

    __slots__ = ('name', 'type', 'default', 'description', 'values', 'row', 'tag',
                 'sub_fields', 'json_name', 'obj', 'bounds')

    def __init__(self, name, type, default=None, description=None, values=None, row=None,
                 tag=None, sub_fields=None, json_name=None, obj=None, bounds=None):
        self.name = name
        self.type = type
        self.default = BLANK.get(type) if default is None else default
        self.description = description if description is not None else name.replace('_', ' ')
        self.values = values
        self.row = row
        self.tag = tag
        self.sub_fields = sub_fields
        self.json_name = json_name
        # parse token on SCALARS only ('Tuple' = a dotted factor reference); tables no longer
        # carry it, `row` and `tag` say it properly
        self.obj = obj
        # (min, max) on a Float the author cannot sensibly exceed - a recovery rate is a fraction
        self.bounds = bounds

    @property
    def key(self):
        """The key an author writes in the JSON, which is what the descriptor is filed under.

        This used to be the descriptor's own name with `json_name` carried alongside as an alias,
        because a store keyed by field name across all 47 deals cannot hold two `Cashflows`. Filed
        per SECTION, each one holds its own, and the alias has nothing left to do.
        """
        return self.json_name or self.name

    def descriptor(self):
        """This field as a `fields.mapping[...]['fields']` entry.

        `col_names`, `sub_types` and `obj` are DERIVED from the row rather than stored: they are
        Handsontable's rendering vocabulary, and a front end that is not Handsontable wants none
        of it. Deriving them is also what stops the two parallel lists drifting apart.
        """
        widget = ('Dropdown' if self.values is not None else
                  'BoundedFloat' if self.bounds is not None else self.WIDGET[self.type])
        # a required field has no default to offer - it is blank until the author supplies it
        d = {'widget': widget, 'description': self.description,
             'value': '' if self.default is REQUIRED else self.default}
        if self.default is REQUIRED:
            d['required'] = True
        if self.values is not None:
            d['values'] = self.values
        if self.bounds is not None:
            d['min'], d['max'] = self.bounds
        if self.obj is not None:
            d['obj'] = self.obj
        if self.row is not None:
            d['obj'] = self.tag if self.tag else [OBJ_TOKEN[f.type] for f in self.row.fields]
            d['col_names'] = [f.name for f in self.row.fields]
            d['sub_types'] = [{'type': 'dropdown', 'source': f.values} if f.values is not None
                              else dict(WIDGET_FORMAT[f.type]) for f in self.row.fields]
        if self.sub_fields is not None:
            # a container holds its children, rather than naming entries in a store beside it
            d['sub_fields'] = {f.key: f.descriptor() for f in self.sub_fields}
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


def required_fields(cls):
    """Every field a class declares REQUIRED, inherited declarations included."""
    return [f.key for group in getattr(cls, 'fields', []) for f in group.fields
            if f.default is REQUIRED]


def validate_instrument(deal):
    """Authoring-time messages for one constructed deal; empty when it has nothing to say.

    Two layers. The declarations give the REQUIRED fields for free. A rule spanning several fields
    cannot be declared on any one of them, so a class states those in its own `validate()` - as
    code, because the predicates have no common shape: a value being non-zero, two fields being
    alternatives, one column of a table row implying another.

    Missing means FALSY, not absent. Optional fields are declared with an empty default, so a UI
    writes the key present and empty where hand-written JSON omits it, and every fallback in the
    engine already tests the value for exactly that reason.

    `validate` is looked up normally rather than own-attr-only, unlike `fields`: an alias subclass
    inherits the `calc_dependencies` whose reads these rules describe, so it inherits the rules.

    Nothing in the valuation path calls this, and a message never stops a deal pricing. The engine
    still fails exactly where it failed before - this says so first, naming the field.
    """
    messages = ['{} is required'.format(name) for name in required_fields(type(deal))
                if not deal.field.get(name)]
    own = getattr(deal, 'validate', None)
    return messages + (list(own()) if own else [])


def emit_instrument(module):
    """The `types` and `sections` of `fields.mapping['Instrument']`, from the classes.

    Scans `module` for classes declaring their own `fields` list. Own-attr only, so a subclass that
    inherits its parent's declaration does not re-emit it as a second deal type. Declaration order
    is preserved: the UI lays panels out in `types[T]` order and widgets within a panel in section
    order, and a set-backed view scrambles both silently, with no exception.

    A section OWNS its descriptors. The store used to key them by field name across every deal,
    which admits one descriptor per name - so a field needing different valid values in two deals
    had to invent a key and carry the real one as an alias. `Payment_Timing` is `Touch`/`Expiry` on
    a one-touch and `End`/`Begin`/`Discounted` on a cashflow leg, and both are right, because the
    JSON is per-deal and only the flat view was ambiguous.
    """
    types, sections = {}, {}
    for deal_type, cls in vars(module).items():
        groups = cls.__dict__.get('fields') if isinstance(cls, type) else None
        if not isinstance(groups, list):
            continue
        types[deal_type] = [g.name for g in groups]
        for g in groups:
            sections.setdefault(g.name, {f.key: f.descriptor() for f in g.fields})
    return types, sections


def emit_factor(module):
    """The `types` of `fields.mapping['Factor']` - each factor TYPE holding its own descriptors.

    A price factor has no sections to compose: its `Price Factors` block is one flat dict, so a
    class declares a flat list and the type IS that list's descriptors, keyed by the JSON key.

    Own-attr only, matching `emit_instrument`. `ForwardRate` subclasses `ForwardPrice` and carries
    no `Fixings`, so it declares its own; a subclass that inherits the declaration is an alias for
    the same block rather than a second factor type.
    """
    return {factor_type: {f.key: f.descriptor() for f in cls.__dict__['fields']}
            for factor_type, cls in vars(module).items()
            if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)}
