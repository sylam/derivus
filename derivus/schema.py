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

"""The field-declaration vocabulary, the emitters that read it off the classes, and `mapping`.

A deal's schema is composition of named field GROUPS rather than the class hierarchy - `FXAdmin`
is shared by eight deals with no common base and `Admin` by all 47 - so a group is a module-level
constant a class lists. A price factor's JSON block is one flat dict, so a factor class declares a
flat list and `emit_factor` files the descriptors per type. `System` and the create-deal menu are
the only hand-written stores left.

Almost nothing here is read at valuation time: `construct_factor` hands the raw block to the
factor class, and this is authoring-time metadata for the UI, the docs generator and the Excel
add-in - a type's entry IS its descriptors, each keyed by the JSON key an author writes, so a
panel, its defaults and the write-back key are one lookup. The one exception is `default=`, which
`declared_defaults` completes a calculation's params with and `DealFields` answers a deal's read
by name from - for the `COMPLETABLE` fields only, a declared default being what a blank panel
shows rather than what an engine means.

`bind=` adds the second axis a front end needs: which fields a job may change without recompiling.
See `partition_factor`.

`mapping` is assembled at the bottom of this file, because the declaring modules import `F` from
here and the edge only runs one way.
"""

import copy
import logging

import numpy as np

from . import utils


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

#: The types whose content is a coordinate grid plus ONE value column - `[[tenor, rate], ...]`,
#: `[[moneyness, expiry, vol], ...]`. The coordinates size `all_tenors` when the factor is
#: constructed; only the last column is content, so `bind` on these splits the field.
SHAPED = tuple(BLANK)

#: `{factor_type: {json_key: F}}`, filled by `emit_factor`. The partition and the emitted store
#: read the same declarations, so neither can drift from the other.
FACTOR_FIELDS = {}

#: The VALUE keys of a `Market Prices` quote row - the plan/values line for that whole section,
#: for every family at once rather than per family. Both `config.update_market_quote`'s tick guard
#: and `partition_market_price`'s projection read this one tuple.
MARKET_QUOTE_VALUES = ('Quoted_Market_Value', 'Quoted_Bid', 'Quoted_Ask', 'Timestamp')

#: Which of those a row cannot be without. A mid is MOVED and never removed, so a patch clearing
#: it refuses; the two-way sides and the stamp are absent whenever the source has no print for
#: them. A fifth value key has to be classified here rather than becoming null-clearable.
MARKET_QUOTE_REQUIRED = ('Quoted_Market_Value',)

#: WHICH TABLE ON A BLOCK CARRIES THE VALUE PLANE, derived from the families' own declarations by
#: `quote_containers` at the bottom of this file - a quote row is a row that DECLARES the four
#: value keys, whatever the table is called. Empty until then, and read only at call time.
#: `partition_market_price` is the one reader, which is what keeps the tick guard, `plan_hash`,
#: `market_patch`/`patch_market` and the artifact slot on one declaration.
MARKET_QUOTE_CONTAINERS = ()

#: How `mapping` renders each type for Handsontable. Rendering only: derived on the way out and
#: never declared, so a front end that is not Handsontable ignores all of it.
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


class Row(object):
    """The ordered fields of one table row - each column a full `F`, carrying its own type,
    default, valid values and required-ness."""
    __slots__ = ('fields',)

    def __init__(self, fields):
        self.fields = list(fields)


class F(object):
    """One field of one deal or one price factor.

    `type` is semantic (Text/Float/Integer/Date/Percent/Basis/Period/Table/Container, plus the
    market-data shapes Curve/Surface/Space); the widget name is the front end's business and is
    only reintroduced when emitting `mapping`. A choice list is not a type - it is a Text whose
    `values` are a fixed set. `json_name` overrides the key the descriptor is filed under, for the
    cashflow shapes that genuinely share a JSON key.

    A Table declares its columns as a `Row`; `tag` names the utils container the wire form uses
    (`DateList`, `DateValueList`, `CreditSupportList`), absent for a plain array of rows.
    `description` is free text a front end reads from the store rather than rebuilding it.

    `bind` is STRUCTURAL by default. `bind='value'` says the engine reads this field's CONTENT and
    that nothing about discovery, tenor grids, process wiring, correlation or the code paths
    depends on it - see `partition_factor`. Declare it only from the consumption site: a wrong
    structural costs a recompile, a wrong value corrupts a plan silently.

    `Boolean` is a bare JSON `true`/`false`, not the `'Yes'`/`'No'` string the rest of the
    vocabulary spells a flag with - a calibration reading `bool(param.get(...))` takes `'No'` as
    true, so the two cannot share a descriptor.
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
        # parse token, scalars only ('Tuple' = a dotted factor reference); a table says it as
        # `row` and `tag` instead
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

        `col_names`, `sub_types` and `obj` are derived from the row rather than stored: they are
        Handsontable's rendering vocabulary, and deriving them is what keeps the parallel lists
        from drifting apart.
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

    Shared blocks (`Admin`, `FXAdmin`) are module-level constants; a class's own block is named
    `<ClassName>.Fields` by convention and built by `own()`.
    """
    __slots__ = ('name', 'fields')

    def __init__(self, name, fields):
        self.name = name
        self.fields = list(fields)


def own(cls_name, fields, role='Fields'):
    """A class's own block, named the way `mapping`'s sections are."""
    return Group('{}.{}'.format(cls_name, role), fields)


def required_fields(cls):
    """Every field a class declares REQUIRED, inherited declarations included."""
    return [f.key for group in getattr(cls, 'fields', []) for f in group.fields
            if f.default is REQUIRED]


def declared_defaults(cls, params):
    """A calculation block completed by its own declarations - the `F` default under every key the
    author omitted, so the schema is the single source of an engine default and reads inside
    `execute` may index directly.

    Mutable defaults are deep-copied per call, so a run edits its params without writing into the
    class declaration. `REQUIRED` and `None` defaults are skipped: the first has nothing to offer,
    the second declares a field the engine is content to find missing."""
    merged = {f.key: copy.deepcopy(f.default) for f in cls.fields
              if f.default is not REQUIRED and f.default is not None}
    merged.update(params)
    return merged


#: A blank Table by the `utils` container its `tag` names. `'null'` is what a widget writes for an
#: empty table, so it is the DECLARED blank of every Table on a deal.
BLANK_TABLE = {'DateList': lambda: utils.DateList({}),
               'DateEqualList': lambda: utils.DateEqualList([]),
               'CreditSupportList': lambda: utils.CreditSupportList([]),
               'DateValueList': list, None: list}

#: The period grammar, built on first use. `get_grid_grammar` is the one place periods are parsed;
#: reaching it needs `config`, which imports this module, so the import is deferred to call time.
_PERIOD = []

#: `{deal class: {key: engine-form default}}`, filled by `deal_defaults`.
_DEAL_DEFAULTS = {}

#: The fields a DEAL completes from its declaration, by name. A declaration is authoring metadata
#: and its `default=` is what a blank panel shows, NOT an economic statement: `FXBarrierOption`
#: declares `Strike_Price` 0.0 and `Barrier_Type` 'Down_And_In', and answering either would turn a
#: schema-invalid block into a plausible wrong number - measured, 741.53 for the strike and the
#: Down_And_In value 6.37 for the type, against 78.93 for the deal the author meant. So completion
#: is an ALLOWLIST of fields whose declared value IS the engine's own fallback and whose repair is
#: gated; every other omission keeps its `KeyError` and the named skip that makes it visible. The
#: list grows by measurement, one field at a time.
COMPLETABLE = frozenset(['Barrier_Monitoring_Frequency', 'Cash_Rebate'])


def _period_offset(period):
    """`'3M'` -> a `DateOffset`, through the grammar a job's own periods are parsed with."""
    if not _PERIOD:
        from .config import get_grid_grammar
        _PERIOD.append(get_grid_grammar()[1])
    return _PERIOD[0].parseString(period)[0]


def engine_default(field):
    """One declared default in the form the ENGINE reads, not the form a widget shows.

    A declaration is authoring metadata - a blank table is the string `'null'`, a period is `'3M'`,
    a rate is a whole number of percent - so a default reaching a pricer is converted exactly as
    the loader converts that field's wire form. A blank Text or Date stays blank, which is NOT the
    same as an omitted field meaning falsy: `Expiry_Date` declares `''` and a blank date reaches a
    comparison as a `str`. Which of these a deal may answer with is `COMPLETABLE`'s question.
    """
    if field.type == 'Table':
        return BLANK_TABLE[field.tag]() if field.default == 'null' else copy.deepcopy(field.default)
    if field.obj == 'Period':
        return _period_offset(field.default)
    if field.obj == 'Percent':
        return utils.Percent(field.default)
    if field.obj == 'Basis':
        return utils.Basis(field.default)
    return copy.deepcopy(field.default)


def deal_defaults(cls):
    """Every field a deal class declares a default for, in engine form.

    Inherited declarations included, `REQUIRED` and `None` skipped on `declared_defaults`' terms.
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

    `field[key]` falls through to `deal_defaults` for a `COMPLETABLE` key the author omitted - the
    read that used to raise `KeyError` inside `calc_dependencies` and skip the deal to a silent
    zero. Every other omission still raises, because a declared default is what a blank panel
    shows and not what the engine means: completing one silently prices a schema-invalid block.

    Everything else is exactly what the author wrote: `get`, `in`, iteration, `len` and the JSON
    round trip, and therefore `plan_hash` and the factor universe `get_fieldname` discovers. A
    default answers a read; it does not enter the program. An explicit `get(key, fallback)` is the
    reader's own statement of what an omitted field means and keeps saying it.
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


def validate_instrument(deal):
    """Authoring-time messages for one constructed deal; empty when it has nothing to say.

    Two layers: the declarations give the REQUIRED fields for free, and a rule spanning several
    fields is stated as code in the class's own `validate()`, because those predicates have no
    common shape.

    Missing means FALSY, not absent - optional fields are declared with an empty default, and
    every fallback in the engine tests the value rather than the key.

    `validate` is looked up normally rather than own-attr-only, unlike `fields`, so an alias
    subclass inherits the rules along with the `calc_dependencies` they describe. Nothing in the
    valuation path calls this, and a message never stops a deal pricing.
    """
    messages = ['{} is required'.format(name) for name in required_fields(type(deal))
                if not deal.field.get(name)]
    own = getattr(deal, 'validate', None)
    return messages + (list(own()) if own else [])


def emit_instrument(module):
    """The `types`, `sections` and `containers` of `mapping['Instrument']`, from the classes.

    Scans `module` for classes declaring their own `fields` list. Own-attr only, so a subclass that
    inherits its parent's declaration does not re-emit it as a second deal type. Declaration order
    is preserved: the UI lays panels out in `types[T]` order and widgets within a panel in section
    order.

    A section OWNS its descriptors, so `Payment_Timing` is `Touch`/`Expiry` on a one-touch and
    `End`/`Begin`/`Discounted` on a cashflow leg and both are right. `containers` names the types
    that can hold other deals, read off `Deal.accepts_children`, so a client rendering from this
    store answers that without importing the engine.
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

    A price factor has no sections to compose: its `Price Factors` block is one flat dict, so a
    class declares a flat list and the type IS that list's descriptors, keyed by the JSON key.
    Own-attr only, matching `emit_instrument`, so a subclass inheriting the declaration is an alias
    for the same block rather than a second factor type.

    The declarations themselves are also recorded in `FACTOR_FIELDS`, which is what
    `partition_factor` reads - one scan, one source.
    """
    declared = {factor_type: {f.key: f for f in cls.__dict__['fields']}
                for factor_type, cls in vars(module).items()
                if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)}
    FACTOR_FIELDS.update(declared)
    return {factor_type: {key: f.descriptor() for key, f in fields.items()}
            for factor_type, fields in declared.items()}


def emit_process(module, factor_types):
    """The `types` of `mapping['Process']`, and the `Process_factor_map` beside it.

    A process's `Price Models` block is one flat dict, like a price factor's, so a class declares a
    flat list and the type IS its descriptors. Own-attr only, matching the other emitters. A
    process is a class declaring `fields` AND `factor_types`; the calibration classes share this
    module and declare `fields` and `model_type` - see `emit_calibration`.

    `Process_factor_map` is the same declaration read the other way round: a class names the price
    factors it can drive, the UI wants a menu per factor. Every factor type is a key, including the
    ones no process drives, because a missing key is a KeyError rather than an empty menu.
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

    `Interpolation` is NOT a key an author writes in a `Price Factors` block: `construct_factor`
    reads it out of the `Price Factor Interpolation` section and injects it, and only for the
    factor types that opt in. So a class declares which METHODS it can be set to rather than an
    `F` for the block, and the map is that read straight off.

    Every `Factor1D` honours an `Interpolation` in its block, but only a routed type is ever given
    one, so publishing the menu for the rest would offer a setting the engine drops.
    """
    return {factor_type: list(cls.__dict__['interpolation_methods'])
            for factor_type, cls in vars(module).items()
            if isinstance(cls, type) and 'interpolation_methods' in cls.__dict__}


def emit_calculation(module):
    """The `types` of `mapping['Calculation']` - each calculation TYPE holding its own.

    Keyed by the `Object` string a job document writes, which is NOT the class name: `run_job`
    branches on `CreditMonteCarlo` / `BaseValuation` / `HedgeMonteCarlo` while the classes are
    `Credit_Monte_Carlo` / `Base_Revaluation` / `HedgeMonteCarlo`. The class states its own with
    `calc_type` because no rule recovers one of those names from the other.
    """
    return {cls.__dict__['calc_type']: {f.key: f.descriptor() for f in cls.__dict__['fields']}
            for cls in vars(module).values()
            if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)}


def emit_market_prices(module):
    """The `types` of `mapping['MarketPrices']` - each price FAMILY holding its own.

    Keyed by the `Market Prices` type string the engine selects work by, which the class declares
    as `market_factor_type`: the block is named for the model with a `Prices` suffix and the class
    with a `Parameters` one, so neither name recovers the other. Own-attr only, matching the other
    emitters, so a base declaring nothing is not a family of its own.
    """
    return {cls.__dict__['market_factor_type']: {f.key: f.descriptor() for f in cls.__dict__['fields']}
            for cls in vars(module).values()
            if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)
            and 'market_factor_type' in cls.__dict__}


def emit_calibration(module):
    """The `types` of `mapping['Calibration']` - each PROCESS holding its tuning block.

    Keyed by the stochastic-process class name, because that is what a `Calibrations` entry is
    filed under, while the entry's own `Method` is the CALIBRATION class
    `construct_calibration_config` dispatches on. A class states the process it calibrates with
    `model_type`, since no rule recovers one of those names from the other.

    `Method` is stamped from the class name rather than declared, so the dispatch key cannot drift
    from the class it dispatches to and the process/calibration wiring needs no map of its own.
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

    Keyed by the class name, which is the registry key `structures.structure_named` dispatches on,
    so a front end offering a menu and the runner pricing the choice read the same word.

    All four of a structure's declarations are published: `vernacular` (what a desk calls it),
    `fields` as descriptors, `legs` as a deal type plus the block the structure pins and the
    parameter slots it maps, and `recipe` as the readable step list. Legs are NOT expanded into
    deal schemas - a leg names a declared `Instrument` type, whose entry in the Instrument store
    IS its field schema, referenced so the two cannot drift.

    Own-attr only, matching the other emitters, and gated on `vernacular` rather than on `fields`
    alone, so the module's own vocabulary classes do not emit as empty structures.
    """
    return {name: {'vernacular': cls.__dict__['vernacular'],
                   'fields': {f.key: f.descriptor() for f in cls.__dict__['fields']},
                   'legs': {leg.role: leg.descriptor() for leg in cls.__dict__['legs']},
                   'recipe': [step.describe() for step in cls.__dict__['recipe']]}
            for name, cls in vars(module).items()
            if isinstance(cls, type) and 'vernacular' in cls.__dict__
            and isinstance(cls.__dict__.get('fields'), list)}


def partition_factor(type_name, block):
    """Split one `Price Factors` block into `(structural, values)`.

    STRUCTURAL is the PLAN's half: everything discovery, the tenor grids, process wiring,
    correlation and the code paths read. That is every field unless its declaration says
    `bind='value'`, an undeclared field included - the safe answer costs only a recompile.

    A shape-valued field splits INSIDE itself rather than falling to one side: the structural half
    keeps the coordinate columns, the value half is the last one. A scalar shadows to `None`, which
    still says the key is THERE - the key SET is structural even where the content is not, so
    adding or dropping a field is a new plan.

    `apply_values` is the exact inverse: `apply_values(t, *partition_factor(t, block)) == block`.
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

    The caller owns the check that `values` names only value-bound fields - it is the one holding
    the factor name, which is what a message has to say.
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

    A field is a quote container when its row declares EVERY `MARKET_QUOTE_VALUES` key: the mid,
    the two-way sides and the stamp. That is the whole rule, and it is a DECLARATION rather than a
    list of table names - a family whose row carries only a mid (an energy option) or quotes under
    another name entirely (a swaption's `Market_Volatility`) declares no such row and stays wholly
    plan-side, and a family that grows one joins the plane by declaring it.

    Tables and containers alike, since the two `Points` families declare the same row two ways -
    one as a `Row` of columns, the other as a container's `sub_fields`.
    """
    keys = set(MARKET_QUOTE_VALUES)
    return tuple(sorted({
        f.key for cls in vars(module).values()
        if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)
        and 'market_factor_type' in cls.__dict__
        for f in cls.__dict__['fields']
        if keys <= {c.key for c in (f.row.fields if f.row is not None else f.sub_fields or [])}}))


def quote_rows(instrument):
    """`(container key, rows)` for the one quote table on this block that carries values, or
    `(None, None)` where the family quotes somewhere the value plane does not reach.

    The block names its own container - no family declares two - so this is a lookup rather than a
    choice, and it is what lets every reader of the split take a block and nothing else.

    A LIST or nothing: a table's declared default is the string `'null'` (what a blank panel
    renders), and a block completed from its declarations rather than authored carries that string
    where a document carries rows. It has no quotes in it, which is the answer here.
    """
    for key in MARKET_QUOTE_CONTAINERS:
        rows = instrument.get(key)
        if rows and isinstance(rows, list):
            return key, rows
    return None, None


def partition_market_price(block):
    """Split one `Market Prices` block into `(structural, values)`.

    A block is the `{"instrument": {...}}` shape both the wire and `cfg.params['Market Prices']`
    carry. STRUCTURAL is everything but `MARKET_QUOTE_VALUES` on each quote row - the pillars, the
    expiries, the strikes, the conventions, the `Deal` a quote is a price for, the weights, the
    solver knobs and the lifecycle switches - because a moved node is a re-authoring and a plan of
    its own. VALUES is one dict per quote row, carrying exactly the value keys that row HAS a
    number for: row ORDER is structural and is what aligns the two halves, so nothing is padded.

    WHICH TABLE IS THE QUOTE TABLE is `MARKET_QUOTE_CONTAINERS`, derived from the declarations by
    `quote_containers`: a curve strip and an FX smile quote in `Points`, the Heston-Nandi families
    in `European_Options`, and the rule is the row's own declared columns rather than the table's
    name.

    A `null` in the document is an ABSENCE rather than a value that happens to be nothing, matching
    how `quote_delta` reads a null in a patch. So the round trip is an identity on every block
    whose value keys hold numbers, and a canonicalisation on one spelling an absence as a null.

    A family whose rows declare no value keys has an EMPTY values half and stays wholly plan-side:
    a tick on such a block is a new plan.

    Value keys are DROPPED rather than shadowed to `None` - a deliberate divergence from
    `partition_factor`, because a pillar that starts or stops being quoted two-sided is the same
    node of the same plan, which makes key-presence itself value-plane here.

    `apply_market_values` is the exact inverse: `apply_market_values(*partition_market_price(b))`
    is `b` again, with the value keys landing last in each row rather than where they were.
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

    The caller owns the check that `values` names only `MARKET_QUOTE_VALUES` and the named refusal
    for a row count that moved - it is the one holding the block name. What is held here is that a
    short values half cannot land silently: row order is the only thing pairing the two halves, so
    a `zip` over mismatched lengths would drop quotes with no refusal anywhere.
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


# Shared field blocks - the groups a class lists rather than inherits, so metadata spanning
# classes lives here beside the vocabulary.
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
    F('Autocall_Thresholds', 'Table', default='null', row=Row([F('Date', 'Date'), F('Value', 'Float')]), tag='DateValueList'),
    F('Payoff_Type', 'Text', default='Standard', values=['Standard', 'Quanto', 'Compo'])
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
    F('MtM', 'Text', default='')
])

FX_ADMIN = Group('FXAdmin', [
    F('Trade_Date', 'Date', default=''),
    F('Delivery_Date', 'Date', default=''),
    F('Sales_Margin', 'Float', default=0),
    F('Structure_Reference', 'Text', default='')
])

#: The columns every option quote carries, whatever the family: the contract, what it is worth,
#: and how much the fit cares.
OPTION_QUOTE = [F('Expiry_Date', 'Date'), F('Strike', 'Float', description='0 reads the forward'),
                F('Option_Type', 'Text', values=['Call', 'Put']), F('Units', 'Float'),
                F('Weight', 'Float', description='Relative weight in the objective'),
                F('Quoted_Market_Value', 'Float',
                  description='The quote, read per Quote_Type; 0 reads the vol surface. The one '
                              'value key a patch cannot clear (MARKET_QUOTE_REQUIRED): a mid is '
                              'moved, never removed')]

#: The EVIDENCE beside a mid: the two-way the print was dealt on and the print's own clock. A row
#: carrying these declares all four `MARKET_QUOTE_VALUES`, which is what `quote_containers` reads
#: to put that table on the value plane - so a family adding this block admits the value-only
#: re-tick, and one quoting the mid alone stays plan-side.
#:
#: Read by NOTHING in any fit: the mid is what every objective is posed against. They are declared
#: so a machine-fetched quote has somewhere to put what it saw.
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


# The declaring modules import `F` from here, so the edge runs one way and the assembly below has
# to come after the vocabulary above it.
from . import bootstrappers, calculation, instruments, riskfactors, stochasticprocess  # noqa: E402
# structures is last: it names Instrument types in its legs, so the store it publishes is only
# meaningful beside one already emitted
from . import structures  # noqa: E402

#: Filled from the declarations now that the families are imported - see `quote_containers`. The
#: name is bound at module scope rather than passed around, because `partition_market_price` takes
#: a block and nothing else and every reader of the split goes through it.
MARKET_QUOTE_CONTAINERS = quote_containers(bootstrappers)

_types, _sections, _containers = emit_instrument(instruments)
_factor_types = emit_factor(riskfactors)
_process_types, _process_factor_map = emit_process(stochasticprocess, _factor_types)

#: The blank value of a table COLUMN, keyed by the `obj` token that column declares. A shape is
#: never a column, so a blank curve or surface is in `BLANK` instead, keyed by type.
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
    # keyed by the PROCESS a `Calibrations` entry is filed under, while its `Method` names the
    # calibration class the engine dispatches on
    'Calibration': {'types': emit_calibration(stochasticprocess)},
    # keyed by the `Object` string a job document writes
    'Calculation': {'types': emit_calculation(calculation)},
    # a factor TYPE holds its own descriptors, and so does a process type
    'Factor': {'types': _factor_types},
    'Process': {'types': _process_types},
    # a price FAMILY holds its own, keyed by the `Market Prices` type string the engine selects
    # work by
    'MarketPrices': {'types': emit_market_prices(bootstrappers)},
    # a SALES structure holds its vernacular, parameters, legs and recipe, keyed by the name the
    # runner dispatches on
    'Structure': {'types': emit_structures(structures)},
    # the UI's two menus: a valid-processes-per-factor one and a valid-interpolations-per-factor
    # one, both the same declarations read the other way round
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
        # the create-deal menu, the one hand-kept part of the Instrument store. Whether a type can
        # hold children is NOT here: it is `Deal.accepts_children`, emitted as `containers` below.
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
