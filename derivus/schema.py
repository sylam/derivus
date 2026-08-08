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

This is what `fields.py` was: 1,931 lines of hand-written stores keyed one descriptor per field
NAME, so two deals needing different valid values for the same field had to invent a key and carry
the real name elsewhere - 21 entries in `ALIASED_KEYS` at its worst. A class that owns its own
fields has no such collision, and the alias list is empty. What is left hand-written is `System`,
whose consumer is `Config` itself, and the create-deal menu.

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

`bind=` adds the second axis a front end needs: which fields a job may change without recompiling.
See `partition_factor`.

`mapping` is assembled at the BOTTOM of this file, because the declaring modules import `F` from
here and the edge only runs one way.
"""

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
#: `[[moneyness, expiry, vol], ...]`, `[[moneyness, expiry, tenor, vol], ...]`. The coordinates
#: size `all_tenors` when the factor is constructed; only the last column is content. So `bind` on
#: these splits the field rather than choosing it, which is why `BLANK` and this share a key set.
SHAPED = tuple(BLANK)

#: `{factor_type: {json_key: F}}`, filled by `emit_factor`. The partition and the emitted store
#: read the same declarations, so neither can drift from the other.
FACTOR_FIELDS = {}

#: How `fields.mapping` renders each type for Handsontable. Rendering only: derived on the way out
#: and never declared, which is the point - a front end that is not Handsontable ignores all of it.
WIDGET_FORMAT = {
    'Date': {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
    'Float': {'type': 'numeric', 'numericFormat': {'pattern': '0,0.00'}},
    'Basis': {'type': 'numeric', 'numericFormat': {'pattern': '0,0.00'}},
    'Percent': {'type': 'numeric', 'numericFormat': {'pattern': '0.00 %'}},
    'Integer': {'type': 'numeric', 'numericFormat': {'pattern': '0.'}},
    'Boolean': {'type': 'checkbox'},
    'Text': {}, 'Period': {}, 'Table': {},
}

#: The `obj` token `fields.mapping` uses per type, for a table declaring its columns positionally.
OBJ_TOKEN = {'Date': 'DatePicker', 'Float': 'Float', 'Integer': 'Integer', 'Text': 'Text',
              'Percent': 'Percent', 'Basis': 'Basis', 'Period': 'Period', 'Boolean': 'Boolean',
              'Table': 'ResetArray'}


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

    `bind` is STRUCTURAL by default. `bind='value'` says the engine reads this field's CONTENT and
    nothing about discovery, tenor grids, process wiring, correlation or the code paths depends on
    it - see `partition_factor`. Declare it only from the consumption site, and leave it alone when
    unsure: a wrong structural costs a recompile, a wrong value corrupts a plan silently.

    `Boolean` is a bare JSON `true`/`false`, which is NOT the `'Yes'`/`'No'` string the rest of the
    vocabulary spells a flag with: a calibration reading `bool(param.get(...))` takes `'No'` as
    true, so the two cannot share a descriptor.
    """
    WIDGET = {'Text': 'Text', 'Float': 'Float', 'Integer': 'Integer', 'Date': 'DatePicker',
              'Percent': 'Float', 'Basis': 'Float', 'Period': 'Text', 'Boolean': 'Checkbox',
              'Table': 'Table', 'Container': 'Container',
              'Curve': 'Flot', 'Surface': 'Three', 'Space': 'Three'}

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
        # parse token on SCALARS only ('Tuple' = a dotted factor reference); tables no longer
        # carry it, `row` and `tag` say it properly
        self.obj = obj
        # (min, max) on a Float the author cannot sensibly exceed - a recovery rate is a fraction
        self.bounds = bounds
        self.bind = bind

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

    It also records the declarations themselves in `FACTOR_FIELDS`, which is what `partition_factor`
    reads - one scan, one source.
    """
    declared = {factor_type: {f.key: f for f in cls.__dict__['fields']}
                for factor_type, cls in vars(module).items()
                if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)}
    FACTOR_FIELDS.update(declared)
    return {factor_type: {key: f.descriptor() for key, f in fields.items()}
            for factor_type, fields in declared.items()}


def emit_process(module, factor_types):
    """The `types` of `fields.mapping['Process']`, and the `Process_factor_map` beside it.

    A process's `Price Models` block is one flat dict, like a price factor's, so a class declares a
    flat list and the type IS its descriptors. Own-attr only, matching the other two emitters:
    `CSImpliedForwardPriceModel` subclasses `CSForwardPriceModel` and takes its parameters from the
    implied factor instead, so it declares its own empty list rather than re-emitting the parent's.

    The map is the same declaration read the other way round. A class names the price factors it can
    drive; the UI wants the inverse, a menu per factor. Every factor type is a key, including the
    ones no process drives, because the Workbench indexes it by the type of the factor in front of
    it - a missing key is a KeyError, not an empty menu.

    A process is a class declaring `fields` AND `factor_types`; the calibration classes share this
    module and declare `fields` and `model_type` - see `emit_calibration`.
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
    factor types that opt in. So the class declares which METHODS it can be set to rather than an
    `F` for the block, and the map is that read straight off - the same shape as a process naming
    the `factor_types` it drives.

    The restriction the two hand-written rows carried is real and is exactly this opt-in. Every
    `Factor1D` honours an `Interpolation` in its block, but only a routed type is ever given one,
    so publishing the menu for the rest would offer a setting the engine drops on the floor.
    """
    return {factor_type: list(cls.__dict__['interpolation_methods'])
            for factor_type, cls in vars(module).items()
            if isinstance(cls, type) and 'interpolation_methods' in cls.__dict__}


def emit_calculation(module):
    """The `types` of `fields.mapping['Calculation']` - each calculation TYPE holding its own.

    Keyed by the `Object` string a job document writes, which is NOT the class name: `run_job`
    branches on `CreditMonteCarlo` / `BaseValuation` / `HedgeMonteCarlo` while the classes are
    `Credit_Monte_Carlo` / `Base_Revaluation` / `HedgeMonteCarlo`. The class states its own with
    `calc_type` rather than the emitter unmangling underscores, because `Base_Revaluation` and
    `BaseValuation` are not the same word and no rule recovers one from the other.
    """
    return {cls.__dict__['calc_type']: {f.key: f.descriptor() for f in cls.__dict__['fields']}
            for cls in vars(module).values()
            if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)}


def emit_market_prices(module):
    """The `types` of `fields.mapping['MarketPrices']` - each price FAMILY holding its own.

    Keyed by the `Market Prices` type string the engine selects work by, which the class declares
    as `market_factor_type` rather than the emitter recovering it from the class name: the block is
    named for the model with a `Prices` suffix and the class with a `Parameters` one, and one
    family is declared without being built at all.

    Own-attr only, matching the other emitters. `HullWhite2FactorModelParameters` subclasses
    `RiskNeutralInterestRateModel`, which declares nothing and is not a family of its own.
    """
    return {cls.__dict__['market_factor_type']: {f.key: f.descriptor() for f in cls.__dict__['fields']}
            for cls in vars(module).values()
            if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)
            and 'market_factor_type' in cls.__dict__}


def emit_calibration(module):
    """The `types` of `fields.mapping['Calibration']` - each PROCESS holding its tuning block.

    Keyed by the stochastic-process class name, because that is what a `Calibrations` entry is filed
    under, while the entry's own `Method` is the CALIBRATION class `construct_calibration_config`
    dispatches on. The class states the process it calibrates with `model_type` for the reason a
    calculation states its `calc_type`: `HWInterestRateCalibration` calibrates
    `HullWhite1FactorInterestRateModel` and no rule recovers one of those names from the other.

    `Method` is stamped from the class NAME rather than declared, so the dispatch key cannot drift
    from the class it dispatches to, and the process/calibration wiring needs no map of its own -
    it is this store's key paired with its `Method`.
    """
    return {cls.__dict__['model_type']: dict(
        {'Method': F('Method', 'Text', default=name,
                     description='The calibration class to run for this model').descriptor()},
        **{f.key: f.descriptor() for f in cls.__dict__['fields']})
        for name, cls in vars(module).items()
        if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)
        and 'model_type' in cls.__dict__}


def partition_factor(type_name, block):
    """Split one `Price Factors` block into `(structural, values)`.

    STRUCTURAL is the PLAN's half: everything discovery, the tenor grids, process wiring,
    correlation and the code paths read. That is every field unless its declaration says
    `bind='value'`, and an undeclared field is structural for the same reason a blank one is - the
    safe answer costs a recompile.

    A shape-valued field splits INSIDE itself rather than falling to one side. A curve's knots size
    `all_tenors` when the factor is constructed while its rate column is content, so the structural
    half keeps the coordinate columns and the value half is the last one. A scalar shadows to
    `None`, which still says the key is THERE: the key SET is structural even where the content is
    not, so adding or dropping a field is a new plan.

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
    the factor NAME, which is what a message has to say.
    """
    declared = FACTOR_FIELDS[type_name]
    for key, f in declared.items():
        # a value-bound field absent from `values` stays a coordinate shell - reconstructible
        # only on purpose, so say so where the caller can hear it
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


# The declaring modules import `F` from here, so the edge only runs one way and the assembly below
# has to come after the vocabulary above it. Any `import derivus.<anything>` initialises the
# package first, and `derivus/__init__` imports this module before any declaring one.
from . import bootstrappers, calculation, instruments, riskfactors, stochasticprocess  # noqa: E402

_types, _sections = emit_instrument(instruments)
_factor_types = emit_factor(riskfactors)
_process_types, _process_factor_map = emit_process(stochasticprocess, _factor_types)

#: Object-list defaults, keyed by WIDGET. The shape-valued ones come from the declaration
#: vocabulary so a blank curve has one definition.
default = {
    'Integer': 0,
    'Float': 0.0,
    'Percent': 0.0,
    'Text': '',
    'Flot': BLANK['Curve'],
    'Surface': BLANK['Surface'],
    'Space': BLANK['Space'],
    'DateList': 'null',
    'CreditSupportList': '[[0,1]]',
    'DatePicker': ''
}

#: Every JSON store a front end renders from, and the last of it that is hand-written is `System`
#: and the create-deal menu. Everything else is emitted from the declarations on the classes.
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
    # the UI's two menus: a valid-processes-per-factor one and a valid-interpolations-per-factor
    # one, both the same declarations read the other way round
    'Process_factor_map': _process_factor_map,
    'Interpolation_factor_map': emit_interpolation(riskfactors),
    # `System` stays hand-written. Its one "type" is a UI panel name rather than anything the JSON
    # dispatches on, and the class that consumes `System Parameters` is `Config` itself - the whole
    # configuration object, so giving it a `fields` list would make "a class that declares fields
    # IS a type" mean something else in that module.
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
        # hold children is NOT here: it is `Deal.accepts_children`, a property of the deal.
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
                 'MtMCrossCurrencySwapDeal', 'FXTARFOptionDeal',
                 'FXDiscreteExplicitDoubleAsianOption', 'FXPartialTimeBarrierOption'],
            'New Energy Derivative':
                ['FloatingEnergyDeal', 'FixedEnergyDeal', 'EnergySingleOption',
                 'CommodityForwardDeal', 'CommodityFutureDeal'],
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
        'types': _types
    }
}
