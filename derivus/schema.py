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

"""Field declarations, the emitters that read them off the classes, and `mapping`.

A deal's schema composes named field GROUPS rather than the class hierarchy (`FXAdmin` is shared by
eight deals with no common base, `Admin` by all 47), so a group is a module-level constant a class
lists. A price factor's block is one flat dict, so a factor class declares a flat list. `System` and
the create-deal menu are the only hand-written stores.

Authoring-time metadata for the UI, the docs generator and the Excel add-in - a type's entry IS its
descriptors, keyed by the JSON key an author writes. `default=` is the exception the engine reads:
`declared_defaults` completes a calculation's params from it and `DealFields` answers a deal's read
by name, for `COMPLETABLE` fields only. `bind=` says which fields a job may change without
recompiling - see `partition_factor`.

`mapping` is assembled at the bottom, because the declaring modules import `F` from here.
"""

import copy
import logging
import numbers
import re

import numpy as np
import pandas as pd

from . import utils


#: `default=REQUIRED` - the author must supply it. Distinct from `default=None`, a field the engine
#: reads with `.get` and is content to find missing.
REQUIRED = object()

#: The blank FORM of a shape-valued field: one degenerate knot, a curve having no empty value.
BLANK = {
    'Curve': '[{"label":"None", "data":[[0.0,0.0]]}]',
    'Surface': '[[0.0,1.0], [1.0,0.0]]',
    'Space': '{"0.0":[[0.0,0.0],[0.0,0.0]]}',
}

#: The types whose content is a coordinate grid plus ONE value column - `[[tenor, rate], ...]`,
#: `[[moneyness, expiry, vol], ...]`. Only the last column is content, so `bind` splits the field.
SHAPED = tuple(BLANK)

#: `{factor_type: {json_key: F}}`, filled by `emit_factor`. The partition and the emitted store read
#: the same declarations, so neither can drift.
FACTOR_FIELDS = {}

#: The VALUE keys of a `Market Prices` quote row, for every family at once. Read by both
#: `update_market_quote`'s tick guard and `partition_market_price`'s projection.
MARKET_QUOTE_VALUES = ('Quoted_Market_Value', 'Quoted_Bid', 'Quoted_Ask', 'Timestamp')

#: Which of those a row cannot be without: a mid is MOVED and never removed, so a patch clearing it
#: refuses. Sides and stamp are absent whenever the source has no print for them.
MARKET_QUOTE_REQUIRED = ('Quoted_Market_Value',)

#: Which table on a block carries the value plane - a row that DECLARES the four value keys,
#: whatever the table is called. Derived by `quote_containers` at the bottom of this file; empty
#: until then, and read only at call time. `partition_market_price` is the one reader.
MARKET_QUOTE_CONTAINERS = ()

#: How `mapping` renders each type for Handsontable. Rendering only, derived on the way out.
WIDGET_FORMAT = {
    'Date': {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
    'Float': {'type': 'numeric', 'numericFormat': {'pattern': '0,0.00'}},
    'Basis': {'type': 'numeric', 'numericFormat': {'pattern': '0,0.00'}},
    'Percent': {'type': 'numeric', 'numericFormat': {'pattern': '0.00 %'}},
    'Integer': {'type': 'numeric', 'numericFormat': {'pattern': '0.'}},
    'Boolean': {'type': 'checkbox'},
    'Text': {}, 'Period': {}, 'Table': {},
}

#: The `obj` token `mapping` uses per type, for a table declaring its columns positionally.
OBJ_TOKEN = {'Date': 'DatePicker', 'Float': 'Float', 'Integer': 'Integer', 'Text': 'Text',
              'Percent': 'Percent', 'Basis': 'Basis', 'Period': 'Period', 'Boolean': 'Boolean',
              'Table': 'ResetArray'}

#: What one declared field's value must BE once `decode_wire` has run, and the wire form to name
#: where it is not: keyed by the `obj` token a field declares, else by its `type`. A declaration
#: absent here is one nothing checkable can be said about from the descriptor alone - a Text, a
#: table, a shape, and a Period, which is a `DateOffset` on a deal and the bare string a desk posts
#: a convention as. A scaled number is a number: the tag says which unit it was authored in.
AUTHORED = {'Date': ((pd.Timestamp,), '{".Timestamp": "2027-01-15"}'),
            'Float': ((numbers.Real, utils.Scaled), 'a number'),
            'Integer': ((numbers.Real, utils.Scaled), 'a number')}


class Row(object):
    """The ordered fields of one table row - each column a full `F`."""
    __slots__ = ('fields',)

    def __init__(self, fields):
        self.fields = list(fields)


class F(object):
    """One field of one deal or one price factor.

    `type` is semantic (Text/Float/Integer/Date/Percent/Basis/Period/Table/Container, plus the
    market-data shapes Curve/Surface/Space); the widget name is reintroduced only when emitting
    `mapping`. A choice list is a Text with fixed `values`. `json_name` overrides the key the
    descriptor is filed under. A Table declares its columns as a `Row`; `tag` names the utils
    container the wire form uses, absent for a plain array of rows.

    `bind` is STRUCTURAL by default. `bind='value'` says the engine reads this field's CONTENT and
    that nothing about discovery, tenor grids, process wiring, correlation or the code paths depends
    on it - see `partition_factor`. Declare it only from the consumption site: a wrong structural
    costs a recompile, a wrong value corrupts a plan silently.

    `Boolean` is a bare JSON `true`/`false`, not the `'Yes'`/`'No'` a flag is spelled with -
    `bool(param.get(...))` takes `'No'` as true, so the two cannot share a descriptor.
    """
    # 'Surface' covers BOTH shaped types (a Space is a tenor-keyed surface), so a renderer branches
    # on the value's row arity, never on the token.
    WIDGET = {'Text': 'Text', 'Float': 'Float', 'Integer': 'Integer', 'Date': 'DatePicker',
              'Percent': 'Float', 'Basis': 'Float', 'Period': 'Text', 'Boolean': 'Checkbox',
              'Table': 'Table', 'Container': 'Container',
              'Curve': 'Curve', 'Surface': 'Surface', 'Space': 'Surface'}

    __slots__ = ('name', 'type', 'default', 'description', 'values', 'row', 'tag',
                 'sub_fields', 'json_name', 'obj', 'bounds', 'bind')

    def __init__(self, name, type, default=None, description=None, values=None, row=None,
                 tag=None, sub_fields=None, json_name=None, obj=None, bounds=None, bind=None):
        self.name = name
        self.type = type
        self.default = BLANK.get(type) if default is None else default
        self.description = description if description is not None else name.replace('_', ' ')
        self.values = values
        self.row = row
        self.tag = tag
        self.sub_fields = sub_fields
        self.json_name = json_name
        # parse token, scalars only ('Tuple' = a dotted factor reference); a table uses `row`/`tag`
        self.obj = obj
        # (min, max) on a Float the author cannot sensibly exceed - a recovery rate is a fraction
        self.bounds = bounds
        self.bind = bind

    @property
    def key(self):
        """The key an author writes in the JSON, which is what the descriptor is filed under."""
        return self.json_name or self.name

    def descriptor(self):
        """This field as a `mapping[...]['fields']` entry.

        `col_names`, `sub_types` and `obj` are derived from the row rather than stored, which keeps
        the parallel lists from drifting apart.
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
        if self.bind is not None:
            d['bind'] = self.bind
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
    """A named, reusable block of fields - what `mapping` calls a section.

    Shared blocks (`Admin`, `FXAdmin`) are module-level constants; a class's own is
    `<ClassName>.Fields` by convention, built by `own()`.
    """
    __slots__ = ('name', 'fields')

    def __init__(self, name, fields):
        self.name = name
        self.fields = list(fields)


def own(cls_name, fields, role='Fields'):
    """A class's own block, named the way `mapping`'s sections are."""
    return Group('{}.{}'.format(cls_name, role), fields)


def declared_fields(cls):
    """Every field a class declares, by the key an author writes - inherited declarations included,
    which is the reading `mapping` composes a type's sections from."""
    return {f.key: f for group in getattr(cls, 'fields', []) for f in group.fields}


def required_fields(cls):
    """Every field a class declares REQUIRED, inherited declarations included."""
    return [key for key, f in declared_fields(cls).items() if f.default is REQUIRED]


def instrument_fields(deal_type):
    """Every JSON key one deal TYPE declares, off the emitted store - the reading a wire block's
    `Object` reaches, and the descriptors `describe_instrument_type` publishes."""
    return sorted(key for section in mapping['Instrument']['types'].get(deal_type, ())
                  for key in mapping['Instrument']['sections'][section])


def declared_defaults(cls, params):
    """A block completed by its own declarations - the `F` default under every key the author
    omitted, so a reader may index directly.

    Mutable defaults are deep-copied per call. `REQUIRED` and `None` are skipped: the first has
    nothing to offer, the second declares a field the engine is content to find missing."""
    merged = {f.key: copy.deepcopy(f.default) for f in cls.fields
              if f.default is not REQUIRED and f.default is not None}
    merged.update(params)
    return merged


#: A blank Table by the `utils` container its `tag` names. `'null'` is what a widget writes for an
#: empty table, so it is the declared blank of every Table on a deal.
BLANK_TABLE = {'DateList': lambda: utils.DateList({}),
               'DateEqualList': lambda: utils.DateEqualList([]),
               'CreditSupportList': lambda: utils.CreditSupportList([]),
               'DateValueList': list, None: list}

#: `{deal class: {key: engine-form default}}`, filled by `deal_defaults`.
_DEAL_DEFAULTS = {}

#: The fields a DEAL completes from its declaration, by name. A `default=` is what a blank panel
#: shows, NOT an economic statement: answering `FXBarrierOption.Strike_Price` 0.0 turns a
#: schema-invalid block into a plausible wrong number - 741.53 against the 78.93 the author meant.
#: So completion is an ALLOWLIST of fields whose declared value IS the engine's own fallback; every
#: other omission keeps its `KeyError` and the named skip that makes it visible.
COMPLETABLE = frozenset(['Barrier_Monitoring_Frequency', 'Barrier_Observation', 'Cash_Rebate'])


def engine_default(field):
    """One declared default in the form the ENGINE reads, not the form a widget shows.

    A blank table is the string `'null'`, a period `'3M'`, a rate a whole number of percent, so a
    default reaching a pricer is converted exactly as the loader converts that field's wire form. A
    blank Text or Date stays blank: `Expiry_Date` declares `''` and reaches a comparison as a `str`.
    """
    if field.type == 'Table':
        return BLANK_TABLE[field.tag]() if field.default == 'null' else copy.deepcopy(field.default)
    if field.obj == 'Period':
        return utils.parse_period(field.default)
    if field.obj == 'Percent':
        return utils.Percent(field.default)
    if field.obj == 'Basis':
        return utils.Basis(field.default)
    return copy.deepcopy(field.default)


def deal_defaults(cls):
    """Every field a deal class declares a default for, in engine form.

    Built once per class and never handed out directly - `DealFields` copies what it reads.
    """
    if cls not in _DEAL_DEFAULTS:
        _DEAL_DEFAULTS[cls] = {f.key: engine_default(f)
                               for group in getattr(cls, 'fields', []) or []
                               for f in group.fields
                               if f.default is not REQUIRED and f.default is not None}
    return _DEAL_DEFAULTS[cls]


class DealFields(dict):
    """A deal's authored block, completed on a READ BY NAME from its own class's declarations.

    `field[key]` falls through to `deal_defaults` for a `COMPLETABLE` key the author omitted. Every
    other omission still raises, because completing one silently prices a schema-invalid block.
    Everything else is exactly what the author wrote - `get`, `in`, iteration, `len`, the JSON round
    trip, and therefore `plan_hash` and the factor universe. A default answers a read; it does not
    enter the program.
    """

    def __init__(self, params=(), cls=None):
        super(DealFields, self).__init__(params)
        self.declared = deal_defaults(cls) if cls is not None else {}
        self.furnished = {}

    def __missing__(self, key):
        if key not in COMPLETABLE or key not in self.declared:
            raise KeyError(key)
        # deep-copied on first read, so a mutable default is this deal's own and never the class's
        return self.furnished.setdefault(key, copy.deepcopy(self.declared[key]))


def value_message(field, value):
    """What one authored value gets wrong against its own declaration, or None.

    A wire tag is the author's spelling and the loader has already turned it into what the engine
    reads, so a bare string where a date belongs survives decoding unchanged - which is what naming
    the wire form is for. A `bool` is not a number, `True` pricing as 1.0, and a field declaring
    `values` accepts nothing outside them. A FALSY value states nothing and is read against
    nothing: an optional field declares an empty default, and a required one missing is a
    different message.
    """
    if not value:
        return None
    if field.values is not None and value not in field.values:
        return '{} is {!r}, not one of {}'.format(field.key, value, ', '.join(field.values))
    wanted = AUTHORED.get(field.obj or field.type)
    if wanted is None or (not isinstance(value, bool) and isinstance(value, wanted[0])):
        return None
    return '{} must be {}, not {!r}'.format(field.key, wanted[1], value)


def unreadable(deal):
    """Whether any value this deal carries is not what its own declaration says it must be.

    What a walk over a deal's dates and amounts cannot be taken over: a date that stayed a string
    compares against no Timestamp, a string amount concatenates. A REQUIRED field left out is not
    one of these - it is absent, and every reader tests the value rather than the key.
    """
    declared = declared_fields(type(deal))
    return any(value_message(declared[key], value)
               for key, value in deal.field.items() if key in declared)


def validate_instrument(deal):
    """Authoring-time messages for one constructed deal; empty when it has nothing to say.

    The declarations give the REQUIRED fields and what every authored value has to BE; a rule
    spanning several fields is code in the class's own `validate()`. Missing means FALSY, not
    absent - optional fields declare an empty default and every fallback tests the value rather
    than the key - and a falsy value is checked against nothing else, having nothing to state.
    `validate` is looked up normally rather than own-attr-only, unlike `fields`, so an alias
    subclass inherits the rules. Nothing in the valuation path calls this; a message never stops a
    deal.
    """
    messages = []
    for key, field in declared_fields(type(deal)).items():
        value = deal.field.get(key)
        message = ('{} is required'.format(key) if not value and field.default is REQUIRED
                   else value_message(field, value))
        if message:
            messages.append(message)
    own = getattr(deal, 'validate', None)
    return messages + (list(own()) if own else [])


def emit_instrument(module):
    """The `types`, `sections` and `containers` of `mapping['Instrument']`, from the classes.

    Scans `module` for classes declaring their own `fields` list. Own-attr only, so a subclass
    inheriting its parent's declaration is not a second deal type. Declaration order is the UI's
    layout order. A section OWNS its descriptors, so `Payment_Timing` is `Touch`/`Expiry` on a
    one-touch and `End`/`Begin`/`Discounted` on a cashflow leg and both are right. `containers` is
    read off `Deal.accepts_children`, so a client renders it without importing the engine.
    """
    types, sections, containers = {}, {}, []
    for deal_type, cls in vars(module).items():
        groups = cls.__dict__.get('fields') if isinstance(cls, type) else None
        if not isinstance(groups, list):
            continue
        types[deal_type] = [g.name for g in groups]
        if getattr(cls, 'accepts_children', False):
            containers.append(deal_type)
        for g in groups:
            sections.setdefault(g.name, {f.key: f.descriptor() for f in g.fields})
    return types, sections, sorted(containers)


def emit_factor(module):
    """The `types` of `mapping['Factor']` - each factor TYPE holding its own descriptors.

    A `Price Factors` block is one flat dict, so the type IS that list's descriptors, keyed by the
    JSON key. Own-attr only, matching `emit_instrument`. The declarations are also recorded in
    `FACTOR_FIELDS`, which `partition_factor` reads - one scan, one source.
    """
    declared = {factor_type: {f.key: f for f in cls.__dict__['fields']}
                for factor_type, cls in vars(module).items()
                if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)}
    FACTOR_FIELDS.update(declared)
    return {factor_type: {key: f.descriptor() for key, f in fields.items()}
            for factor_type, fields in declared.items()}


def emit_process(module, factor_types):
    """The `types` of `mapping['Process']`, and the `Process_factor_map` beside it.

    A process is a class declaring `fields` AND `factor_types`; the calibration classes share this
    module and declare `fields` and `model_type` - see `emit_calibration`. Own-attr only.

    `Process_factor_map` is the same declaration read the other way round. Every factor type is a
    key, including the ones no process drives, because a missing key is a KeyError not an empty menu.
    """
    declared = [(name, cls) for name, cls in vars(module).items()
                if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)
                and 'factor_types' in cls.__dict__]
    factor_map = {factor_type: [] for factor_type in factor_types}
    for name, cls in declared:
        for factor_type in cls.__dict__['factor_types']:
            factor_map[factor_type].append(name)
    types = {name: {f.key: f.descriptor() for f in cls.__dict__['fields']} for name, cls in declared}
    return types, factor_map


def emit_interpolation(module):
    """The `Interpolation_factor_map` beside the Factor store, from the factor declarations.

    `Interpolation` is NOT a key an author writes: `construct_factor` reads it out of the `Price
    Factor Interpolation` section and injects it, for the opted-in types only, so a class declares
    which METHODS it can be set to rather than an `F` for the block. Every `Factor1D` honours one,
    but only a routed type is given one, so publishing the rest would offer a setting the engine
    drops.
    """
    return {factor_type: list(cls.__dict__['interpolation_methods'])
            for factor_type, cls in vars(module).items()
            if isinstance(cls, type) and 'interpolation_methods' in cls.__dict__}


def emit_calculation(module):
    """The `types` of `mapping['Calculation']` - each calculation TYPE holding its own.

    Keyed by the `Object` string a job document writes, which is NOT the class name (`run_job`
    branches on `CreditMonteCarlo`, the class is `Credit_Monte_Carlo`), so the class states its own
    as `calc_type`.
    """
    return {cls.__dict__['calc_type']: {f.key: f.descriptor() for f in cls.__dict__['fields']}
            for cls in vars(module).values()
            if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)}


def emit_market_prices(module):
    """The `types` of `mapping['MarketPrices']` - each price FAMILY holding its own.

    Keyed by the `Market Prices` type string the engine selects work by, declared as
    `market_factor_type`: the block carries a `Prices` suffix and the class a `Parameters` one, so
    neither name recovers the other. Own-attr only, so a base declaring nothing is not a family.
    """
    return {cls.__dict__['market_factor_type']: {f.key: f.descriptor() for f in cls.__dict__['fields']}
            for cls in vars(module).values()
            if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)
            and 'market_factor_type' in cls.__dict__}


def emit_configuration(module, interpolation_default):
    """`mapping['Configuration']` - one declaration per market-data SECTION a book states its
    bootstrap in, keyed by the section's own name.

    A `Bootstrapper Configuration` entry is keyed by what its family WRITES, the key the registry
    resolves, and its dials are `Prices` - the stem of the `Market Prices` type it routes on - plus
    every scalar the family declares. A Table or a Container is the quote BLOCK's (the ladders and
    the instrument definitions), never the section's. `aliases` is what else the registry answers
    to for that family, so an older book's class-name key files under the right entry.

    `Price Factor Interpolation` REFERENCES the menu beside it rather than restating it, names both
    `ModelParams` halves - the `entry` one method per routed type is written in and the `rules` one
    factor's own scheme is, keyed by `key` off its dotted name - and states what a routed factor is
    built with where the section names neither.
    """
    entries = {}
    for cls in module.FAMILIES:
        stem = cls.market_factor_type[:-len('Prices')]
        dials = {module.PRICES_KEY: F(module.PRICES_KEY, 'Text', default=stem,
                                      description='The Market Prices stem this entry routes on'
                                      ).descriptor()}
        dials.update({f.key: f.descriptor() for f in cls.fields
                      if f.type not in ('Table', 'Container')})
        entries[cls.price_factor_type] = {
            'aliases': sorted(name for name, family in module.WRITERS.items()
                              if family is cls and name != cls.price_factor_type),
            'fields': dials}
    return {'Bootstrapper Configuration': {'types': entries},
            'Price Factor Interpolation': {'entry': 'modeldefaults', 'rules': 'modelfilters',
                                           'key': 'id', 'menu': 'Interpolation_factor_map',
                                           'value': interpolation_default}}


def emit_calibration(module):
    """The `types` of `mapping['Calibration']` - each PROCESS holding its tuning block.

    Keyed by the stochastic-process class name, what a `Calibrations` entry is filed under, while
    the entry's own `Method` is the CALIBRATION class `construct_calibration_config` dispatches on.
    A class states the process it calibrates with `model_type`. `Method` is stamped from the class
    name rather than declared, so the dispatch key cannot drift from the class it dispatches to.
    """
    return {cls.__dict__['model_type']: dict(
        {'Method': F('Method', 'Text', default=name,
                     description='The calibration class to run for this model').descriptor()},
        **{f.key: f.descriptor() for f in cls.__dict__['fields']})
        for name, cls in vars(module).items()
        if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)
        and 'model_type' in cls.__dict__}


def emit_structures(module):
    """The `types` of `mapping['Structure']` - each SALES structure holding what it is made of.

    Keyed by the class name, the registry key `structures.structure_named` dispatches on, so a menu
    and the runner pricing the choice read the same word. `vernacular`, `fields` as descriptors and
    `recipe` as the readable step list are published for every structure, and then EITHER `legs`,
    where it is dealt one way, OR `variations` - each with the side of the pair its client buys, the
    parameters only it takes and its own legs - where it is dealt more. Never both, so a consumer
    reads which shape it has off the entry. A leg NAMES a declared `Instrument` type rather than
    expanding its schema, so the two cannot drift.

    A SELECTOR is published as one, with no `value`: it chooses which variation is being quoted
    rather than filling a leg, so a front end must not ask a client to state it like a parameter.
    It is the declared field OBJECT that says so and never its key, or a structure's own parameter
    that merely shares the name would be published as a selector - stripped of its value, and
    optional where it was declared required. Whether one MUST be stated is computed from the
    declarations rather than listed: `"required"` where some variation's own parameters do not tell
    it apart from another's (a strip's two forms name one level, so only the client's side can say
    which is dealt) and `"optional"` where naming the level is enough.

    Own-attr only, gated on `vernacular` rather than `fields` alone, so the module's own vocabulary
    classes do not emit as empty structures.
    """
    entries = {}
    for name, cls in vars(module).items():
        if not (isinstance(cls, type) and 'vernacular' in cls.__dict__
                and isinstance(cls.__dict__.get('fields'), list)):
            continue
        variations = cls.__dict__.get('variations')
        own = [{f.key for f in variation.fields} for variation in (variations or {}).values()]
        needed = any(one <= other for index, one in enumerate(own)
                     for other in own[index + 1:] + own[:index])
        fields = {}
        for f in cls.__dict__['fields']:
            published = f.descriptor()
            if f in module.SELECTORS:
                published = dict({k: v for k, v in published.items() if k != 'value'},
                                 selector='required' if needed else 'optional')
            fields[f.key] = published
        entry = {'vernacular': cls.__dict__['vernacular'], 'fields': fields,
                 'recipe': [step.describe() for step in cls.__dict__['recipe']]}
        if 'variations' in cls.__dict__:
            entry['variations'] = {word: variation.descriptor()
                                   for word, variation in variations.items()}
        else:
            entry['legs'] = {leg.role: leg.descriptor() for leg in cls.__dict__['legs']}
        entries[name] = entry
    return entries


def partition_factor(type_name, block):
    """Split one `Price Factors` block into `(structural, values)`.

    STRUCTURAL is the PLAN's half: everything discovery, the tenor grids, process wiring,
    correlation and the code paths read - every field unless its declaration says `bind='value'`, an
    undeclared field included, since the safe answer costs only a recompile.

    A shape-valued field splits INSIDE itself: structural keeps the coordinate columns, values takes
    the last one. A scalar shadows to `None`, which still says the key is THERE - the key SET is
    structural, so adding or dropping a field is a new plan.

    Exact inverse of `apply_values`: `apply_values(t, *partition_factor(t, block)) == block`.
    """
    declared = FACTOR_FIELDS.get(type_name, {})
    structural, values = dict(block), {}
    for key, content in block.items():
        field = declared.get(key)
        if field is None or field.bind != 'value':
            continue
        if field.type in SHAPED:
            structural[key] = utils.Curve(content.meta, content.array[:, :-1])
            values[key] = content.array[:, -1].tolist()
        else:
            structural[key] = None
            values[key] = content
    return structural, values


def apply_values(type_name, structural, values):
    """Put a values patch back onto a structural projection, returning the whole block.

    The caller owns the check that `values` names only value-bound fields - it holds the factor name.
    """
    declared = FACTOR_FIELDS[type_name]
    for key, f in declared.items():
        # a value-bound field absent from `values` stays a coordinate shell
        if f.bind == 'value' and key in structural and key not in values:
            logging.warning('%s.%s reconstructed without its values - left as a coordinate shell',
                            type_name, key)
    block = dict(structural)
    for key, content in values.items():
        if declared[key].type in SHAPED:
            coords = structural[key]
            content = utils.Curve(coords.meta, np.column_stack((coords.array, content)))
        block[key] = content
    return block


def quote_containers(module):
    """The instrument keys whose ROWS travel the value plane - `MARKET_QUOTE_CONTAINERS`, read off
    the market-price families' own declarations.

    A field is a quote container when its row declares EVERY `MARKET_QUOTE_VALUES` key - a
    DECLARATION rather than a list of table names, so a family carrying only a mid stays wholly
    plan-side and one that grows the block joins by declaring it. Tables and containers alike, since
    the two `Points` families declare the same row two ways.
    """
    keys = set(MARKET_QUOTE_VALUES)
    return tuple(sorted({
        f.key for cls in vars(module).values()
        if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)
        and 'market_factor_type' in cls.__dict__
        for f in cls.__dict__['fields']
        if keys <= {c.key for c in (f.row.fields if f.row is not None else f.sub_fields or [])}}))


def quote_table(block, market_price, key='Points'):
    """The quote ROWS of one `Market Prices` block, refused by name where the block carries none.

    A Table's declared blank is the string `'null'`, so a block completed from its declarations
    rather than authored carries that where a document carries rows, and every reader that walks
    the table walks the string instead. One read for all of them.
    """
    rows = block.get(key)
    if not isinstance(rows, list):
        found = repr(rows)
        raise ValueError(
            '{0} carries no {1} table, so there is nothing to fit: the block reads {2} where a '
            'list of quote rows belongs. Author the quotes under {1}, or drop the block'.format(
                market_price or 'A Market Prices block', key,
                found if len(found) < 48 else found[:45] + '...'))
    return rows


def quote_rows(instrument):
    """`(container key, rows)` for the one quote table on this block that carries values, or
    `(None, None)` where the family quotes somewhere the value plane does not reach.

    The block names its own container - no family declares two - so this is a lookup, not a choice.
    A LIST or nothing: a table's declared default is the string `'null'`, and a block completed from
    its declarations rather than authored carries that string where a document carries rows.
    """
    for key in MARKET_QUOTE_CONTAINERS:
        rows = instrument.get(key)
        if rows and isinstance(rows, list):
            return key, rows
    return None, None


def partition_market_price(block):
    """Split one `Market Prices` block into `(structural, values)`.

    A block is the `{"instrument": {...}}` shape both the wire and `cfg.params['Market Prices']`
    carry. STRUCTURAL is everything but `MARKET_QUOTE_VALUES` on each quote row, because a moved
    node is a plan of its own; VALUES is one dict per quote row carrying exactly the value keys that
    row HAS a number for. Row ORDER aligns the two halves, so nothing is padded. Which table is the
    quote table is `MARKET_QUOTE_CONTAINERS`.

    A `null` is an ABSENCE, matching how `quote_delta` reads a null in a patch, so the round trip is
    an identity on every block whose value keys hold numbers. A family whose rows declare no value
    keys has an EMPTY values half and stays wholly plan-side.

    Value keys are DROPPED rather than shadowed to `None` - a deliberate divergence from
    `partition_factor`: a pillar that starts or stops being quoted two-sided is the same node of the
    same plan, which makes key-presence itself value-plane here.

    `apply_market_values` is the exact inverse, with the value keys landing last in each row.
    """
    container, points = quote_rows(block['instrument'])
    if not points:
        return dict(block), []
    values = [{key: point[key] for key in MARKET_QUOTE_VALUES if point.get(key) is not None}
              for point in points]
    structural = dict(block, instrument=dict(block['instrument'], **{container: [
        {key: content for key, content in point.items() if key not in MARKET_QUOTE_VALUES}
        for point in points]}))
    # a block no row of which carries a value key contributes nothing, exactly as one with no table
    return structural, values if any(values) else []


def apply_market_values(structural, values):
    """Put a values patch back onto a structural projection, returning the whole block.

    The caller owns the check that `values` names only `MARKET_QUOTE_VALUES` - it holds the block
    name. Held here: row order is the only thing pairing the two halves, so a `zip` over mismatched
    lengths would drop quotes with no refusal anywhere.
    """
    if not values:
        return dict(structural)
    instrument = structural['instrument']
    container, points = quote_rows(instrument)
    if len(values) != len(points or []):
        raise ValueError('a values half of {} row(s) against {} quote row(s) - row ORDER is what '
                         'pairs the two halves, so the caller must align them'.format(
                             len(values), len(points or [])))
    return dict(structural, instrument=dict(instrument, **{container: [
        dict(point, **row) for point, row in zip(points, values)]}))


# Shared field blocks - the groups a class lists rather than inherits.
CASHFLOWLISTDEAL = Group('CashflowListDeal.Fields', [
    F('Repo_Rate', 'Text', default='', obj='Tuple'),
    F('Recovery_Rate', 'Text', default='', obj='Tuple'),
    F('Description', 'Text', default=''),
    F('Survival_Probability', 'Text', default='', obj='Tuple'),
    F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell']),
    F('Settlement_Date', 'Date', default=''),
    F('Settlement_Rate', 'Text', default=''),
    F('Currency', 'Text', default=''),
    F('Discount_Rate', 'Text', default='', obj='Tuple'),
    F('Investment_Horizon', 'Date', default=''),
    F('Issuer', 'Text', default='', obj='Tuple')
])

EQUITYOPTIONBASE = Group('EquityOptionBase.Fields', [
    F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell']),
    F('Currency', 'Text', default=''),
    F('Discount_Rate', 'Text', default='', obj='Tuple'),
    F('Equity', 'Text', default='', obj='Tuple'),
    F('Equity_Volatility', 'Text', default='', obj='Tuple'),
    F('Expiry_Date', 'Date', default=''),
    F('Option_Type', 'Text', default='Call', values=['Call', 'Put']),
    F('Payoff_Currency', 'Text', default=''),
    F('Strike_Price', 'Float', default=0.0),
    F('Dividends', 'Text', default='', obj='Tuple')
])

QEDI_CUSTOMAUTOCALLSWAP = Group('QEDI_CustomAutoCallSwap.Fields', [
    F('Price_Fixing', 'Table', default='null', row=Row([F('Date', 'Date'), F('Value', 'Float')]), tag='DateValueList'),
    F('Settlement_Style', 'Text', default='Physical', values=['Physical', 'Cash']),
    F('Option_On_Forward', 'Text', default='No', values=['Yes', 'No']),
    F('Barrier', 'Float', default=0),
    F('Option_Style', 'Text', default='European', values=['European', 'American']),
    F('Units', 'Float', default=0.0),
    F('Barrier_Dates', 'Table', default='null', row=Row([F('Date', 'Date')])),
    F('Autocall_Coupons', 'Table', default='null', row=Row([F('Date', 'Date'), F('Value', 'Float')]), tag='DateValueList'),
    F('Coupon_Observations', 'Table', default='null',
      row=Row([F('Coupon', 'Date'), F('Observation', 'Date')]),
      description='Which price fixing each coupon is observed on, one row per coupon. Absent - '
                  'the default - the pairing is DERIVED from the schedule: a coupon observes '
                  'every fixing after its predecessor up to and including its own date'),
    F('Autocall_Thresholds', 'Table', default='null', row=Row([F('Date', 'Date'), F('Value', 'Float')]), tag='DateValueList'),
    F('Payoff_Type', 'Text', default='Standard', values=['Standard', 'Quanto', 'Compo']),
    F('Barrier_Observation', 'Text', default='Spot', values=['Spot', 'Average'],
      description="Where the put barrier is observed - the barrier date's own spot, or the "
                  "arithmetic average of the coupon window it falls in")
])

QEDI_CUSTOMSWAP = Group('QEDI_CustomSwap.Fields', [
    F('Forecast_Rate', 'Text', default='', obj='Tuple'),
    F('Floating_Margin', 'Float', default=0.0),
    F('Reset_Frequency', 'Text', default='3M', obj='Period'),
    F('Autocall_Floating', 'Table', default='null', row=Row([F('Date', 'Date'), F('Value', 'Float')]), tag='DateValueList')
])

ADMIN = Group('Admin', [
    F('Object', 'Text', default=''),
    F('Reference', 'Text', default=''),
    F('Tags', 'Text', default=''),
    F('MtM', 'Text', default=''),
    F('Sales_Margin', 'Float', default=0,
      description='What the desk charged for this ticket over the mid, in Sales_Margin_Currency '
                  'and positive when the desk earns it. A RECORD of what was agreed: the charge '
                  'is already inside the terms, so nothing is priced off this field. Every deal '
                  'declares it, a margin being a property of the ticket rather than of an asset '
                  'class'),
    F('Sales_Margin_Currency', 'Text', default='',
      description='The currency Sales_Margin is stated in. A margin is an AMOUNT rather than a '
                  'rate, and it need not be a currency this deal settles or discounts in')
])

FX_ADMIN = Group('FXAdmin', [
    F('Trade_Date', 'Date', default=''),
    F('Delivery_Date', 'Date', default=''),
    F('Structure_Reference', 'Text', default='')
])

#: The columns every option quote carries, whatever the family.
OPTION_QUOTE = [F('Expiry_Date', 'Date'), F('Strike', 'Float', description='0 reads the forward'),
                F('Option_Type', 'Text', values=['Call', 'Put']), F('Units', 'Float'),
                F('Weight', 'Float', description='Relative weight in the objective'),
                F('Quoted_Market_Value', 'Float',
                  description='The quote, read per Quote_Type; 0 reads the vol surface. The one '
                              'value key a patch cannot clear (MARKET_QUOTE_REQUIRED): a mid is '
                              'moved, never removed')]

#: The EVIDENCE beside a mid: the two-way the print was dealt on and its clock. A row carrying these
#: declares all four `MARKET_QUOTE_VALUES`, which is what `quote_containers` reads to put that table
#: on the value plane. Read by NOTHING in any fit - the mid is what every objective is posed
#: against; they are declared so a machine-fetched quote has somewhere to put what it saw.
QUOTE_TWO_WAY = [
    F('Quoted_Bid', 'Float',
      description='The bid side of this quote, in the same unit as the mid. QUOTE-LAYER data: the '
                  'fit reads Quoted_Market_Value alone. Optional, because a contract the source '
                  'quotes no two-way for stays mid-only rather than borrowing a spread'),
    F('Quoted_Ask', 'Float',
      description='The offer side, the pair of Quoted_Bid, and optional on the same terms'),
    F('Timestamp', 'Date', default='',
      description='When this quote was seen - the contract\'s own last print, which is what says a '
                  'listed strike is still a market. Stored and reported, never read by the fit: '
                  'what counts as too old is the consumer\'s policy')]


# ---------------------------------------------------------------------------------------
# The wire-form document tree: the verbs that read and edit a JOB DOCUMENT as it sits on
# disk, before any of it becomes an engine object. They read this module's own vocabulary
# and nothing else, and every reader of them is outside config.

def job_children(document):
    """The root `Children` list of a job document in wire form - the ONE read of the deal tree's
    root, so a document that is not a job refuses by name rather than raising on a missing key."""
    try:
        children = document['Calc']['Deals']['Deals']['Children']
    except (KeyError, TypeError):
        children = None
    if not isinstance(children, list):
        raise ValueError('not a job document - no Calc.Deals.Deals.Children')
    return children


def walk_job_deals(children, path=()):
    """Every deal node of a wire-form job, as `(deal_path, node)`, from the document or from a
    `Children` list. The positional path ('0/2/1') is the node's identity, because References are
    not unique in a book."""
    if isinstance(children, dict):
        children = job_children(children)
    for position, node in enumerate(children):
        deal_path = path + (position,)
        yield '/'.join(map(str, deal_path)), node
        yield from walk_job_deals(node.get('Children', []), deal_path)


def splice_deal(document, deal, parent_reference=None):
    """Append `deal` to a wire-form job document IN PLACE and return the new node's `deal_path`.

    The node is `{'Instrument': {'.Deal': deal}}`, gaining an empty `Children` when the booked type
    is itself a container. A COMPOSED deal - one arriving with node-shaped legs under its own
    `Children`, the way `structures.quote` hands a structure back - has them lifted onto the node,
    because the engine walks `node['Children']` and never inside the deal block. The insertion point
    is the root, or the single node whose Reference is `parent_reference`; an unknown, ambiguous or
    non-container parent refuses by name.
    """
    children = job_children(document)
    containers = mapping['Instrument']['containers']
    parent_path = ''
    if parent_reference is not None:
        found = [(deal_path, node) for deal_path, node in walk_job_deals(children)
                 if node['Instrument']['.Deal'].get('Reference') == parent_reference]
        if len(found) != 1:
            raise ValueError('{} deals carry Reference {!r} - a parent must be unique'.format(
                len(found) or 'no', parent_reference))
        parent_path, parent = found[0]
        parent_type = parent['Instrument']['.Deal'].get('Object')
        if parent_type not in containers:
            raise ValueError('{!r} is a {}, which takes no children'.format(
                parent_reference, parent_type))
        children = parent.setdefault('Children', [])
    deal = dict(deal)
    composed = deal.pop('Children', None)
    node = {'Instrument': {'.Deal': deal}}
    if composed:
        node['Children'] = composed
    elif deal.get('Object') in containers:
        node['Children'] = []
    children.append(node)
    position = str(len(children) - 1)
    return '{}/{}'.format(parent_path, position) if parent_path else position


def _positions(deal_path):
    """A `deal_path` as index steps. Negative positions refuse rather than silently resolving
    from the end - a wrong path must never quietly name a different deal."""
    positions = [int(p) for p in str(deal_path).split('/')]
    if any(p < 0 for p in positions):
        raise ValueError
    return positions


def deal_at(document, deal_path):
    """The node at a positional `deal_path` - a live reference into the document, which is what
    an amendment edits in place."""
    children = job_children(document)
    try:
        node = None
        for position in _positions(deal_path):
            node = children[position]
            children = node.get('Children', [])
        return node
    except (ValueError, IndexError):
        raise ValueError('no deal at path {!r}'.format(deal_path))


def remove_deal(document, deal_path):
    """Remove and return the node at a positional `deal_path`, in place - the whole subtree goes
    with it, which is what deleting a structure means."""
    children = job_children(document)
    try:
        positions = _positions(deal_path)
        for position in positions[:-1]:
            children = children[position]['Children']
        return children.pop(positions[-1])
    except (ValueError, KeyError, IndexError):
        raise ValueError('no deal at path {!r}'.format(deal_path))


def quote_plan(block):
    """A `Market Prices` block less its value plane - the half a tick may NOT move, and what two
    blocks are compared on to tell a re-quote from a re-authoring."""
    instrument = dict(block['instrument'])
    container, points = quote_rows(instrument)
    if container is not None:
        instrument[container] = [{key: value for key, value in point.items()
                                  if key not in MARKET_QUOTE_VALUES} for point in points]
    return instrument


def update_market_quote(document, name, block):
    """Install or update one `Market Prices` block in a wire-form job document, in place.

    An update is VALUE-ONLY, for every family at once: everything except each quote row's
    `Quoted_Market_Value`, `Quoted_Bid`/`Quoted_Ask` and `Timestamp` must stand, because the pillar
    set, the expiries, the strikes, the conventions and the tolerances are STRUCTURE - a plan and a
    pinned grid hang off them, so a moved node is a re-authoring, never a tick. A two-way is
    value-side for the reason the mid is. That line is `MARKET_QUOTE_VALUES` and which table
    carries it is `quote_rows`, both read here rather than kept as a second copy of the split
    `plan_hash` and `market_patch` take. Returns 'installed' or 'updated'.
    """
    if not isinstance(block, dict) or 'instrument' not in block:
        raise ValueError('{}: a Market Prices block is {{"instrument": {{...}}}}'.format(name))
    prices = document['Calc']['MergeMarketData']['ExplicitMarketData'].setdefault(
        'Market Prices', {})
    if name in prices:
        if quote_plan(prices[name]) != quote_plan(block):
            raise ValueError('{}: structure differs from the installed block - a moved node is a '
                             'new plan; re-author it deliberately'.format(name))
        prices[name] = block
        return 'updated'
    prices[name] = block
    return 'installed'


def sniff_indent(text, default=2):
    """A JSON file's own indent, so a rewrite is a diff of the change and nothing else."""
    found = re.search(r'\n( +)"', text)
    return len(found.group(1)) if found else default


def tables_of(results):
    """Every table in a `Results` tree, flat, under the path that names it.

    `cashflows` and `scenarios` are dicts of tables, so they arrive as `cashflows/ZAR` and
    `scenarios/FxRate.ZAR`. The flat form is what a per-RESULT-CLASS tolerance is declared against.
    """
    tables = {}
    for name, value in results.items():
        if isinstance(value, dict) and '.DataFrame' not in value:
            tables.update({'{}/{}'.format(name, path): table
                           for path, table in tables_of(value).items()})
        else:
            tables[name] = value
    return tables


# The declaring modules import `F` from here, so the assembly must come after the vocabulary above.
from . import bootstrappers, calculation, instruments, riskfactors, stochasticprocess  # noqa: E402
# structures is last: its legs name Instrument types, meaningful only beside an emitted store
from . import structures  # noqa: E402

#: Filled from the declarations now that the families are imported - see `quote_containers`.
MARKET_QUOTE_CONTAINERS = quote_containers(bootstrappers)

_types, _sections, _containers = emit_instrument(instruments)
_factor_types = emit_factor(riskfactors)
_process_types, _process_factor_map = emit_process(stochasticprocess, _factor_types)

#: The blank value of a table COLUMN, keyed by its declared `obj` token. A shape is never a column,
#: so a blank curve or surface is in `BLANK` instead, keyed by type.
default = {
    'Integer': 0,
    'Float': 0.0,
    'Percent': 0.0,
    'Text': '',
    'DateList': 'null',
    'CreditSupportList': '[[0,1]]',
    'DatePicker': ''
}

#: Every JSON store a front end renders from. Only `System` and the create-deal menu are
#: hand-written; the rest is emitted from the declarations on the classes.
mapping = {
    # keyed by the PROCESS; its `Method` names the calibration class the engine dispatches on
    'Calibration': {'types': emit_calibration(stochasticprocess)},
    # keyed by the `Object` string a job document writes
    'Calculation': {'types': emit_calculation(calculation)},
    # a factor TYPE holds its own descriptors, and so does a process type
    'Factor': {'types': _factor_types},
    'Process': {'types': _process_types},
    # a price FAMILY holds its own, keyed by the type string the engine selects work by; `values`
    # is the quote-row plane `update_market_quote` refuses against, so a client ticks what the
    # engine calls a value without spelling one of the four names
    'MarketPrices': {'types': emit_market_prices(bootstrappers),
                     'values': list(MARKET_QUOTE_VALUES)},
    # the two market-data sections a book states its bootstrap in, and what each entry may carry
    'Configuration': emit_configuration(bootstrappers, riskfactors.INTERPOLATION_DEFAULT),
    # a SALES structure holds its vernacular, parameters, legs and recipe
    'Structure': {'types': emit_structures(structures)},
    # the UI's two menus, the same declarations read the other way round
    'Process_factor_map': _process_factor_map,
    'Interpolation_factor_map': emit_interpolation(riskfactors),
    # hand-written: its one "type" is a UI panel, and `System Parameters` is consumed by `Config`
    'System': {
        'fields': {
            'Base_Currency': {'widget': 'Text', 'description': 'Base Currency', 'value': ''},
            'Base_Date': {'widget': 'DatePicker', 'description': 'Base Date',
                          'value': default['DatePicker']},
            'Exclude_Deals_With_Missing_Market_Data': {
                'widget': 'Dropdown', 'value': 'Yes', 'values': ['Yes', 'No'],
                'description': 'Exclude Deals With Missing Market Data'},
            'Correlations_Healing_Method': {
                'widget': 'Dropdown', 'value': 'Eigenvalue_Raising',
                'values': ['Eigenvalue_Raising', 'Alternating_Projections'],
                'description': 'Correlations Healing Method'}
        },
        'types': {
            'Config':
                ['Base_Currency', 'Base_Date', 'Exclude_Deals_With_Missing_Market_Data',
                 'Correlations_Healing_Method']
        }
    },
    'Instrument': {
        # the create-deal menu, the one hand-kept part of the store; whether a type holds children
        # is `Deal.accepts_children`, emitted as `containers` below
        'groups': {
            'New Structure': ['NettingCollateralSet', 'StructuredDeal'],
            'New Interest Rate Derivative':
                ['FixedCashflowDeal', 'CFFixedListDeal', 'CFFixedInterestListDeal',
                 'CFFloatingInterestListDeal', 'DepositDeal', 'CapDeal', 'FRADeal',
                 'FloorDeal', 'SwapInterestDeal', 'SwaptionDeal',
                 'YieldInflationCashflowListDeal', 'CashAccountDeal'],
            'New FX Derivative':
                ['FXNonDeliverableForward', 'FXForwardDeal', 'FXOptionDeal', 'FXBinaryOption',
                 'FXDiscreteExplicitAsianOption', 'FXOneTouchOption',
                 'FXBarrierOption', 'FXSwapDeal',
                 'MtMCrossCurrencySwapDeal', 'FXTARFOptionDeal', 'FXAccumulatorOptionDeal',
                 'FXExtendableForwardDeal',
                 'FXDiscreteExplicitDoubleAsianOption', 'FXPartialTimeBarrierOption'],
            'New Energy Derivative':
                ['FloatingEnergyDeal', 'FixedEnergyDeal', 'EnergySingleOption',
                 'CommodityForwardDeal', 'CommodityFutureDeal',
                 'CommodityAveragePriceSwapDeal'],
            'New Equity Derivative':
                ['EquityDeal', 'EquitySwapLeg', 'EquityForwardDeal',
                 'EquityOptionDeal', 'EquityBinaryOption',
                 'EquityOneTouchOption', 'QEDI_CustomAutoCallSwap',
                 'QEDI_CustomAutoCallSwap_V2', 'EquitySwapletListDeal',
                 'EquityBarrierOption', 'EquityBarrierBinaryOption',
                 'EquityDiscreteExplicitAsianOption'],
            'New Credit Derivative': ['DealDefaultSwap', 'CreditNthToDefault']
        },
        'sections': _sections,
        'types': _types,
        'containers': _containers
    }
}
