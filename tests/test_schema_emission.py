"""`schema.mapping['Instrument']`, `['Factor']`, `['Process']` and `['Calculation']` are generated
from the per-class `fields` declarations, and a SECTION (deals) or a TYPE (everything else) owns its
descriptors - so there is no round-trip to gate.

Three defect classes are unreachable by construction: a type naming no class (a type IS a class that
declares fields), a section naming a field with no descriptor or a descriptor no section reaches (a
section IS its descriptors), and one field name resolving to another deal's descriptor.

What remains gateable is what the declarations can still get wrong: a malformed descriptor, a
section declared two ways, a shared group copied instead of shared, a deal type no create-menu
offers, a process no factor menu offers, a calculation type `run_job` cannot dispatch, and a
switch a run reads declared somewhere other than where it reads it.

The census is per store and per family: each test reads every declaration of its kind, so it is
fast and drives no document - a store equal to its declaration IS the outcome.
"""
import ast
import inspect
import os
import sys
import textwrap
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import derivus
from derivus import (bootstrappers, calculation, instruments, riskfactors, schema,
                     stochasticprocess)

INSTRUMENT = schema.mapping['Instrument']
FACTOR = schema.mapping['Factor']
PROCESS = schema.mapping['Process']
PROCESS_FACTOR_MAP = schema.mapping['Process_factor_map']
CALCULATION = schema.mapping['Calculation']
CALIBRATION = schema.mapping['Calibration']
INTERPOLATION_MAP = schema.mapping['Interpolation_factor_map']
MARKET_PRICES = schema.mapping['MarketPrices']

# riskfactors classes that legitimately declare no schema row of their own.
UNDECLARED_FACTORS = {
    'Factor0D': 'dimension base', 'Factor1D': 'dimension base',
    'Factor2D': 'dimension base', 'Factor3D': 'dimension base',
    'CurveModelParameters': 'the base of a spot model whose parameters carry term structures, never a type',
    'InterestRateJacobian': 'its block is keyed by BENCHMARK name, so it has no fixed field set',
}

# How many columns each table tag's deserializer consumes, read off `set_repr` in derivus_jupyter:
# `DateList`/`CreditSupportList` unpack a PAIR per row, `DateEqualList` keys on [0] and keeps [1:],
# `DateValueList` only re-types [0], and `ResetArray` zips against ten hardcoded field types.
TAG_ARITY = {'DateList': (2, 2), 'CreditSupportList': (2, 2), 'DateEqualList': (2, 99),
             'DateValueList': (1, 99), 'ResetArray': (1, 10)}


def declared_classes():
    """Deal classes carrying their own `fields`. Own-attr only, matching `emit_instrument`: a
    subclass inheriting its parent's declaration is an alias, not a second deal type."""
    return {n: c.__dict__['fields'] for n, c in vars(instruments).items()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)}


def factor_classes():
    """The same for price factors, whose declaration is one flat list rather than groups."""
    return {n: c.__dict__['fields'] for n, c in vars(riskfactors).items()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)}


def process_classes():
    """Stochastic-process classes carrying their own `fields`, own-attr only as `emit_process`
    reads them. `CSImpliedForwardPriceModel` subclasses `CSForwardPriceModel` and declares its own
    empty list, so it is a type in its own right rather than an alias re-emitting the parent's.

    `factor_types` is what separates a process from a calibration: both declare `fields` in this
    module, and a process names the factors it drives while a calibration names the process it
    calibrates."""
    return {n: c.__dict__['fields'] for n, c in vars(stochasticprocess).items()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)
            and 'factor_types' in c.__dict__}


def concrete_processes():
    """Every process class DEFINED here bar the abstract base - the set a declared type comes
    from, and the set every declared type has to come back to."""
    return {n: c for n, c in vars(stochasticprocess).items()
            if isinstance(c, type) and issubclass(c, stochasticprocess.StochasticProcess)
            and c is not stochasticprocess.StochasticProcess}


def riskfactor_classes():
    """Every class DEFINED in riskfactors - the set a declared type has to come from."""
    return {n: c for n, c in vars(riskfactors).items()
            if isinstance(c, type) and c.__module__ == riskfactors.__name__}


def every_field(group):
    """A group's fields including nested container children and table columns."""
    def walk(f):
        yield f
        for child in (f.sub_fields or []):
            yield from walk(child)
    for top in group.fields:
        yield from walk(top)


def test_every_store_is_the_emitted_view():
    """Every store a front end renders from is what its emitter builds off the declarations, and
    none is empty - every gate below is vacuously true over an empty declaration set, so an import
    error or a filter bug would otherwise read as a green schema. Nor does a flat name-keyed store
    sit beside the per-type ones.

    Killing mutation: a hand-written entry beside the emitted Factor types."""
    process_types, factor_map = schema.emit_process(stochasticprocess, FACTOR['types'])
    emitted = {
        'Instrument': (INSTRUMENT['types'], schema.emit_instrument(instruments)[0]),
        'Factor': (FACTOR['types'], schema.emit_factor(riskfactors)),
        'Process': (PROCESS['types'], process_types),
        'Process_factor_map': (PROCESS_FACTOR_MAP, factor_map),
        'Calculation': (CALCULATION['types'], schema.emit_calculation(calculation)),
        'MarketPrices': (MARKET_PRICES['types'], schema.emit_market_prices(bootstrappers)),
        'Calibration': (CALIBRATION['types'], schema.emit_calibration(stochasticprocess)),
        'Interpolation_factor_map': (INTERPOLATION_MAP, schema.emit_interpolation(riskfactors))}
    for name, (store, view) in emitted.items():
        assert store, f'the {name} store is empty - every gate over it is vacuous'
        assert store == view, f'the {name} store is not the emitted view - a hand-written copy'
    for store in (FACTOR, PROCESS, CALCULATION, CALIBRATION):
        assert 'fields' not in store, 'a flat name-keyed store has come back beside the types'
    assert set(MARKET_PRICES) == {'types', 'values'}, (
        f'sub-stores have come back beside the types: {sorted(set(MARKET_PRICES) - {"types", "values"})}')


def test_every_deal_type_carries_its_own_plain_names():
    """A model asks for "an NDF" or "a cap", so every emitted deal type carries the names a desk
    says for it, published as `vernacular` - its OWN, read off the class and never its parent's, so
    a binary is never called the option it subclasses. The scratch pair holds the own-attr rule
    where every real type already names itself.

    Killing mutation: the name read through `getattr`, which hands a type declaring none its
    parent's words - the scratch child then reads 'a parent'.
    """
    named = INSTRUMENT['vernacular']
    assert set(named) == set(INSTRUMENT['types'])
    assert [deal_type for deal_type in sorted(named) if not named[deal_type]
            or named[deal_type] != vars(getattr(instruments, deal_type)).get('vernacular')] == []
    scratch = types.ModuleType('scratch')
    scratch.Parent = type('Parent', (), {'fields': [instruments.ADMIN], 'vernacular': 'a parent'})
    scratch.Child = type('Child', (scratch.Parent,), {'fields': [instruments.ADMIN]})
    assert schema.emit_instrument(scratch)[3] == {'Parent': 'a parent', 'Child': None}


#: The deal types whose own declarations name no day they settle, expire or mature on: containers
#: dated by what they hold, and schedules no field of theirs declares as settling.
UNDATED = ('CFFixedInterestListDeal', 'CFFixedListDeal', 'CFFloatingInterestListDeal',
           'CashAccountDeal', 'EnergySingleOption', 'EquityDeal', 'EquitySwapletListDeal',
           'FixedEnergyDeal', 'FloatingEnergyDeal', 'NettingCollateralSet', 'StructuredDeal',
           'YieldInflationCashflowListDeal')


def test_every_deal_type_names_the_days_its_tenor_is_read_off():
    """A TENOR IS READ OFF DECLARATIONS (`schema.tenor_fields`): the dates a type declares it
    settles on and the fields it declares for an expiry or a maturity. Every emitted type answers
    one, or is named here as declaring none - so a type added with neither is one a tenor cap
    refuses by name, and somebody decided it would be.

    Killing mutation: the settlement dates left out of the tenor, which dates a forward, a
    cashflow and every strip by nothing.
    """
    assert sorted(deal_type for deal_type in INSTRUMENT['types']
                  if not schema.tenor_fields(getattr(instruments, deal_type))) == list(UNDATED)
    assert schema.tenor_fields(instruments.FXForwardDeal)[0][0] == 'Settlement_Date'


def test_the_fields_shim_serves_the_same_objects():
    """`derivus.fields` is deprecated for one release and holds nothing of its own. `fields.mapping`
    was the documented surface and the package is on PyPI, so an external caller that bound it keeps
    working - on the same object, not a copy, which is the whole point of the retirement.

    Killing mutation: the shim binding a copy of the store."""
    assert derivus.fields.mapping is schema.mapping
    assert derivus.fields.default is schema.default
    src = inspect.getsource(derivus.fields)
    assert 'mapping = {' not in src, 'the shim has grown a store of its own'


FAMILIES = ('Instrument', 'Factor', 'Process', 'Calculation', 'MarketPrices', 'Calibration')


@pytest.mark.parametrize('family', FAMILIES)
def test_descriptor_shape(family):
    """The descriptor is a tagged union on `widget` and consumers destructure it positionally: a
    `Table` missing `sub_types` raises in the UI's cell renderer, a `Dropdown` missing `values`
    renders an empty list. Containers are recursed - the CVA, FVA, CollVA and initial-margin
    blocks, the quote container and the generation parameters are where the nesting is - and the
    shapeless arrays are pinned separately, below.

    Deals read the DECLARATIONS class by class: sections are shared objects, so walking the store
    visits `Admin` once however many types list it.

    Killing mutation: a Dropdown's `values` left out of its descriptor."""
    if family == 'Instrument':
        for cls_name, groups in declared_classes().items():
            for group in groups:
                for f in every_field(group):
                    check_descriptor(f'{cls_name}.{group.name}.{f.key}', f.descriptor())
    elif family == 'Factor':
        for cls_name, fields in factor_classes().items():
            for f in fields:
                check_descriptor(f'{cls_name}.{f.key}', f.descriptor())
    else:
        store = {'Process': PROCESS, 'Calculation': CALCULATION, 'MarketPrices': MARKET_PRICES,
                 'Calibration': CALIBRATION}[family]
        for type_name, descriptors in store['types'].items():
            for key, d in descriptors.items():
                check_shape(f'{type_name}.{key}', d)


# Descriptors whose JSON value is an array or map whose SHAPE is an OUTPUT - an NxN transition
# matrix, a length-N regime vector, a list of per-regime dicts, a deal map keyed by Object then
# Reference, or a whole deal whose TYPE a sibling names. `Table` declares fixed columns and
# `Container` fixed named children, so the vocabulary cannot state any of them and the Workbench
# raises. The pinning gate fails both when one appears and when one is fixed. Keyed (type, key).
SHAPELESS = {
    ('MarkovHMMSpotModel', 'States'),
    ('MarkovHMMSpotModel', 'Transition_Matrix'),
    ('MarkovHMMSpotModel', 'Initial_State_Probs'),
    # The GARCH drift chain reuses the HMM vocabulary; same output-shaped arrays.
    ('GARCHSpotModel', 'Drift_States'),
    ('GARCHSpotModel', 'Drift_Transition_Matrix'),
    ('GARCHSpotModel', 'Drift_Initial_Probs'),
    ('QuadraticCarryCurveModel', 'Reference_Tenors'),
    ('BasisLinkedSpotModel', 'Sigma_By_State'),
    ('CreditMonteCarlo', 'Credit_Valuation_Adjustment.CDS_Tenors'),
    # what remains is a map keyed by a NAME the author invents (deal Object then Reference;
    # instrument or commodity) or a plain list of factor names - shapes the vocabulary cannot state
    ('HedgeMonteCarlo', 'Hedging_Problem.Tradable_Instruments'),
    ('HedgeMonteCarlo', 'Hedging_Problem.Liabilities'),
    ('HedgeMonteCarlo', 'Hedging_Problem.Portfolio_State'),
    ('HedgeMonteCarlo', 'Scenario_Factors'),
    ('InterestRatePrices', 'Points.Deal'),
}


def is_shapeless(d):
    """A Container with no children or a Table with no columns - the two ways a descriptor can
    name a shape the vocabulary cannot state."""
    return ((d['widget'] == 'Container' and 'sub_fields' not in d)
            or (d['widget'] == 'Table' and 'col_names' not in d))


def check_shape(key, d):
    """`check_descriptor` skipping the pinned shapeless set, so the gate covers everything else."""
    if (key.split('.')[0], key.split('.', 1)[1]) not in SHAPELESS:
        check_descriptor(key, d)
    for sub_key, sub in d.get('sub_fields', {}).items():
        check_shape(f'{key}.{sub_key}', sub)


def check_descriptor(key, d):
    assert {'widget', 'description', 'value'} <= set(d), f'{key} is missing a required descriptor key'
    assert ('values' in d) == (d['widget'] == 'Dropdown'), f'{key}: values must appear iff Dropdown'
    assert ('sub_fields' in d) == (d['widget'] == 'Container'), (
        f'{key}: sub_fields must appear iff Container')
    assert ('col_names' in d) == ('sub_types' in d) == (d['widget'] == 'Table'), (
        f'{key}: col_names and sub_types must both appear, iff Table')
    if d['widget'] == 'Table':
        assert len(d['col_names']) == len(d['sub_types']), (
            f'{key}: col_names and sub_types are matched by POSITION and disagree in length')
        if isinstance(d['obj'], list):
            assert len(d['obj']) == len(d['col_names']), f'{key}: one obj token per column'
        else:
            lo, hi = TAG_ARITY[d['obj']]
            assert lo <= len(d['col_names']) <= hi, (
                f'{key}: tag {d["obj"]} consumes {lo}..{hi} columns, the row declares '
                f'{len(d["col_names"])}')


def every_descriptor(node=None, key='mapping'):
    """Every descriptor in the published mapping, `sub_fields` included - a dict is a descriptor
    iff it carries `widget` and `value`, which is what the consumers destructure on."""
    node = schema.mapping if node is None else node
    if isinstance(node, dict):
        if 'widget' in node and 'value' in node:
            yield key, node
            for sub_key, sub in node.get('sub_fields', {}).items():
                yield from every_descriptor(sub, f'{key}.{sub_key}')
        else:
            for k, v in node.items():
                yield from every_descriptor(v, f'{key}.{k}')


def test_the_widget_vocabulary_names_no_plotting_library():
    """'Flot' (jQuery-flot) and 'Three' (three.js/k3d) are plotting libraries, not types. The tokens
    denote a curve OBJECT and a surface OBJECT, and the store says so: 'Curve', and 'Surface'
    covering both shaped types, so a renderer branches on row arity rather than on the token. The
    legacy spellings live in the front end's `LEGACY_WIDGET` map.

    Killing mutation: the Curve type's widget spelt `Flot` again."""
    widgets = set()
    for key, d in every_descriptor():
        assert d['widget'] not in ('Flot', 'Three'), key
        widgets.add(d['widget'])
    # non-vacuous: the new tokens are actually emitted, so a rename AWAY from them also fails here
    assert {'Curve', 'Surface'} <= widgets


def test_the_column_default_map_is_keyed_by_a_column_token():
    """`default` supplies the blank cell of a table COLUMN, and a column is never a shape. Keyed by
    widget, 'Surface' would mean both the token every Space field carries and the Surface type, so
    `default[F.WIDGET['Space']]` would hand a Space field a 2-column blank where its blank is a
    tenor-keyed map. `BLANK`, keyed by TYPE, is the one definition of a shape's blank.

    The last assert is the jupyter hazard: `set_value_from_widget` reads `obj or widget` into ONE
    variable and picks the decoder off it, so a column token spelled like a shape widget would
    decode a table as a curve.

    Killing mutation: a `Curve` blank in the column map."""
    column_tokens = set(schema.OBJ_TOKEN.values()) | set(TAG_ARITY)
    assert set(schema.default) <= column_tokens, sorted(set(schema.default) - column_tokens)
    shape_widgets = {schema.F.WIDGET[t] for t in schema.SHAPED}
    assert set(schema.default).isdisjoint(shape_widgets)
    emitted = set()
    for _, d in every_descriptor():
        obj = d.get('obj')
        emitted.update(obj if isinstance(obj, list) else [] if obj is None else [obj])
    assert emitted.isdisjoint(shape_widgets), sorted(emitted & shape_widgets)


def test_no_class_is_hidden_from_the_create_menu():
    """`groups` is the Workbench's create-deal menu and stays hand-curated, being presentation. So
    it is the one part of the store that can still drift from the classes: a deal type absent from
    every group is fully declared, fully priceable and unreachable from the UI.

    Killing mutation: `StructuredDeal` dropped from its menu group."""
    menued = {t for members in INSTRUMENT['groups'].values() for t in members}
    assert not sorted(set(INSTRUMENT['types']) - menued), (
        f'declared deal types in no menu group: {sorted(set(INSTRUMENT["types"]) - menued)}')


def test_one_name_may_carry_two_descriptors_in_different_types():
    """The capability the per-section and per-type stores exist for, pinned so a return to a flat
    one fails - the JSON is per deal, per process and per factor type, so only a store keyed by
    field name across all of them was ambiguous, one silently winning and the loser carrying an
    alias.

    `Payment_Timing` is `Touch`/`Expiry` on a one-touch and `End`/`Begin`/`Discounted` on a
    cashflow leg; `Sigma` a scalar on the OU/hazard/Clewlow-Strickland models and a term-structure
    curve on Hull-White (once filed as `sigma`); `Surface` a (moneyness, expiry, vol) triple list on
    a `VolatilityGrid` and a quad list on the three vol SPACES (once filed as `Space`).

    Killing mutation: the Instrument emitter filing each descriptor under its field name across
    every section, the first declaration winning."""
    sections = INSTRUMENT['sections']
    seen = {tuple(d['values']) for s in sections.values()
            for k, d in s.items() if k == 'Payment_Timing'}
    assert seen == {('Touch', 'Expiry'), ('End', 'Begin', 'Discounted')}, seen

    # and each type sees only its own
    def values_for(deal_type):
        return [d['values'] for g in INSTRUMENT['types'][deal_type]
                for k, d in sections[g].items() if k == 'Payment_Timing']
    assert values_for('FXOneTouchOption') == [['Touch', 'Expiry']]
    assert values_for('CapDeal') == [['End', 'Begin', 'Discounted']]

    processes = PROCESS['types']
    assert processes['LogOUSpotModel']['Sigma']['widget'] == 'Float'
    assert processes['HullWhite1FactorInterestRateModel']['Sigma']['widget'] == 'Curve'
    assert processes['BasisLinkedSpotModel']['Phi']['widget'] == 'Float'
    assert not any('sigma' in d for d in processes.values()), 'the lowercase alias key is back'

    assert FACTOR['types']['VolatilityGrid']['Surface']['value'] == schema.BLANK['Surface']
    for space in ('InterestYieldVol', 'InterestRateVol', 'ForwardPriceVol'):
        assert FACTOR['types'][space]['Surface']['value'] == schema.BLANK['Space']


@pytest.mark.parametrize('family', FAMILIES)
def test_no_type_declares_a_key_twice(family):
    """A section or a type is a dict keyed by the JSON name, so a name declared twice in one loses a
    descriptor outright. `Net_Cashflows` was declared twice verbatim on `StructuredDeal`, and the
    option family builds its eight factor-reference fields from `factor_types`, which is exactly
    the shape that can produce one. No cross-type version: two SECTIONS may key the same name
    differently, which is the point.

    Killing mutation: `Greeks` declared twice on the base valuation."""
    keys = {
        'Instrument': lambda: {f'{c}.{g.name}': [f.key for f in g.fields]
                               for c, groups in declared_classes().items() for g in groups},
        'Factor': lambda: {c: [f.key for f in fields] for c, fields in factor_classes().items()},
        'Process': lambda: {c: [f.key for f in fields] for c, fields in process_classes().items()},
        'Calculation': lambda: {t: [f.key for f in c.__dict__['fields']]
                                for t, c in calculation_classes().items()},
        'MarketPrices': lambda: {t: [f.key for f in c.__dict__['fields']]
                                 for t, c in market_price_classes().items()},
        'Calibration': lambda: {t: ['Method'] + [f.key for f in c.__dict__['fields']]
                                for t, c in calibration_classes().items()}}[family]()
    dupes = {name: sorted({k for k in found if found.count(k) > 1}) for name, found in keys.items()}
    assert not {name: d for name, d in dupes.items() if d}, f'{family} keys declared twice'


def test_no_section_is_declared_two_ways():
    """The same hazard one level up: `sections` is keyed by group NAME, so two groups sharing a name
    and differing in fields silently collapse to one panel.

    Killing mutation: the one-touch listing an `FXAdmin` one field short."""
    seen = {}
    for cls_name, groups in declared_classes().items():
        for g in groups:
            seen.setdefault(g.name, {}).setdefault(tuple(f.key for f in g.fields), []).append(cls_name)
    clashing = {k: {c[0] for c in v.values()} for k, v in seen.items() if len(v) > 1}
    assert not clashing, f'one section name declared with differing fields: {clashing}'


def test_shared_groups_are_shared_not_copied():
    """`FXAdmin` is one object listed by eight classes. Copying it per class would let the copies
    drift - which is the whole defect the name-keyed dict has, reintroduced one level down.

    Killing mutation: the one-touch listing a copy of `FXAdmin`, field for field."""
    users = [f for f in declared_classes().values() if any(g.name == 'FXAdmin' for g in f)]
    assert len(users) > 1, 'FXAdmin is declared by fewer than two classes - nothing to share'
    groups = {id(g) for f in users for g in f if g.name == 'FXAdmin'}
    assert len(groups) == 1, f'FXAdmin exists as {len(groups)} distinct objects, not one'


def test_the_factor_types_are_the_riskfactor_classes():
    """Both directions. `construct_factor` does `globals().get(factor.type)(block)`, so a declared
    type naming no class is `None(block)` - a TypeError at compile time rather than a logged miss;
    `ConvenienceYield` sat in that state, declared with two fields and no class. And a factor class
    no schema declares cannot be authored or documented. The exemptions are the four dimension
    bases and the curve-model base, which are never a `Factor.type`, and the Jacobian, whose block
    is keyed by benchmark instrument rather than by a fixed field set.

    Killing mutation: `ForwardPriceSample`'s declaration withdrawn."""
    undispatchable = sorted(set(FACTOR['types']) - set(riskfactor_classes()))
    assert not undispatchable, f'schema offers factor types with no class: {undispatchable}'
    missing = sorted(set(riskfactor_classes()) - set(FACTOR['types']) - set(UNDECLARED_FACTORS))
    assert not missing, f'riskfactors classes no schema can author: {missing}'


def test_the_process_types_are_the_process_classes():
    """Both directions. `construct_process` does `globals().get(sp_type)(factor, param,
    implied_factor)`, so a declared type naming no class is a TypeError as the scenario engine
    builds; and a process class no schema declares cannot be authored from the Workbench or found
    in the JSON reference - `GARCHSpotModel` sat there, calibrated, shipped in a fixture,
    documented, and absent from both the Price Models panel and every process menu.

    Killing mutation: `LogOUSpotModel`'s declaration withdrawn."""
    undispatchable = sorted(set(PROCESS['types']) - set(concrete_processes()))
    assert not undispatchable, f'schema offers process types with no class: {undispatchable}'
    missing = sorted(set(concrete_processes()) - set(PROCESS['types']))
    assert not missing, f'process classes no schema can author: {missing}'


def test_the_process_menu_is_every_factor_type_by_the_declared_processes():
    """The Workbench indexes the map by the type of the factor in front of it
    (`possible_risk_process[factor.type]`), so a factor type with no entry is a KeyError that takes
    the whole Price Factors page down; the panel looks an offered process's descriptors up by name,
    so an entry naming no declared type is a KeyError one click later; and a process no factor's
    menu offers cannot be selected at all - three implied models were in that state, plus
    `GARCHSpotModel`.

    Killing mutation: `HWHazardRateModel` declaring no factor type."""
    assert set(PROCESS_FACTOR_MAP) == set(FACTOR['types']), (
        f'process menu and factor types disagree: '
        f'{sorted(set(PROCESS_FACTOR_MAP) ^ set(FACTOR["types"]))}')
    offered = {p for members in PROCESS_FACTOR_MAP.values() for p in members}
    assert offered == set(PROCESS['types']), (
        f'offered but undeclared: {sorted(offered - set(PROCESS["types"]))}; declared but in no '
        f'menu: {sorted(set(PROCESS["types"]) - offered)}')


def calculation_classes():
    """Calculation classes carrying their own `fields`, keyed by the `Object` string a job writes
    rather than by the class name - `Base_Revaluation` is authored as `BaseValuation`."""
    return {c.__dict__['calc_type']: c for c in vars(calculation).values()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)}


def dispatched_calculations():
    """The `Object` strings `Context.run_job` actually branches on, read off the source.

    Parsed rather than listed, for the reason `bootstrapped_market_factor_types` is: a hand-kept
    list here would be a fourth store of the same knowledge and would drift the same way."""
    found = set()
    src = inspect.getsource(derivus.Context.run_job)
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        # `if self.current_cfg.deals['Calculation']['Object'] == 'X':`
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Subscript) \
                and isinstance(node.left.slice, ast.Constant) and node.left.slice.value == 'Object':
            found.update(c.value for c in node.comparators
                         if isinstance(c, ast.Constant) and isinstance(c.value, str))
    return found


def test_the_calculation_types_are_what_run_job_dispatches():
    """Both directions. `run_job` branches on the `Object` string and RAISES on a miss, so a declared
    type naming no branch is a calculation the create menu offers and the engine refuses to run;
    and the converse was where the drift was - `HedgeMonteCarlo` had a `run_job` branch, a
    documented contract and two shipped fixtures, and no schema row, so opening a job that used it
    raised KeyError in `CalculationPage.load_items`. The type is the `Object` string, not the class
    name (`Base_Revaluation` is authored as `BaseValuation`), which the class states with
    `calc_type` because no rule recovers one word from the other.

    Killing mutation: `run_job`'s HedgeMonteCarlo branch keyed on another string."""
    undispatchable = sorted(set(CALCULATION['types']) - dispatched_calculations())
    assert not undispatchable, f'schema offers calculations run_job cannot dispatch: {undispatchable}'
    undeclared = sorted(dispatched_calculations() - set(CALCULATION['types']))
    assert not undeclared, f'calculations run_job dispatches that no schema declares: {undeclared}'


def calculation_fallback_reads(cls_name, module_ast):
    """`{key: fallback}` for every calc knob read as `params.get('Key', <constant>)`.

    Scoped to the class AND its bases in this module, the way `fallback_reads` is: the exposure
    engine reads most of its knobs on `Credit_Monte_Carlo` and a subclass declaring one of them
    would otherwise publish two defaults with nothing holding them together.
    """
    classes = {n.name: n for n in module_ast.body if isinstance(n, ast.ClassDef)}
    nodes = [classes[cls_name]] + [classes[ast.unparse(b)] for b in classes[cls_name].bases
                                   if ast.unparse(b) in classes]
    locals_ = ('params', 'self.params', 'calc_params')
    return {n.args[0].value: n.args[1].value
            for node in nodes for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'get' and ast.unparse(n.func.value) in locals_
            and len(n.args) == 2 and all(isinstance(a, ast.Constant) for a in n.args)}


@pytest.mark.parametrize('calc_type', sorted(calculation_classes()))
def test_a_declared_calculation_default_is_the_default_the_engine_falls_back_to(calc_type):
    """A knob read with `.get(key, fallback)` publishes TWO defaults, and they have to be one: the
    panel gives the DECLARED value, an omitted key gives the FALLBACK, and where they disagree the
    same job means two things with nothing raising.

    Scoped to keys the type DECLARES. A knob read but never declared is a different defect - the
    panel cannot write it - and `HedgeMonteCarlo` reads a dozen without declaring any.

    Killing mutation: the credit Monte Carlo's `Keep_Tensor` declared `Yes` where the engine
    falls back to `No`.
    """
    module_ast = ast.parse(inspect.getsource(calculation))
    cls_name = calculation_classes()[calc_type].__name__
    declared = declared_values(CALCULATION['types'][calc_type])
    drift = {key: (declared[key], fallback)
             for key, fallback in calculation_fallback_reads(cls_name, module_ast).items()
             if key in declared and declared[key] != fallback}
    assert not drift, f'{calc_type} declares one default and falls back to another: {drift}'


def declared_menus(descriptors):
    """`{key: [choice, ...]}` for every declared knob that offers a menu, containers flattened."""
    out = {}
    for key, d in descriptors.items():
        if 'values' in d:
            out[key] = d['values']
        out.update(declared_menus(d.get('sub_fields', {})))
    return out


def compared_constants(cls_name, module_ast):
    """`{key: {constant, ...}}` for every calc knob the engine tests against a string LITERAL.

    Both spellings of the test, because the engine uses both: `params['Key'] == 'X'` and
    `params.get('Key', d) == 'X'`, plus the `in ('X', 'Y')` form. Scoped to the class and its
    bases in this module, exactly as `calculation_fallback_reads` is.
    """
    classes = {n.name: n for n in module_ast.body if isinstance(n, ast.ClassDef)}
    nodes = [classes[cls_name]] + [classes[ast.unparse(b)] for b in classes[cls_name].bases
                                   if ast.unparse(b) in classes]
    locals_ = ('params', 'self.params', 'calc_params')

    def key_read(node):
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) in locals_ \
                and isinstance(node.slice, ast.Constant):
            return node.slice.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == 'get' and ast.unparse(node.func.value) in locals_ \
                and node.args and isinstance(node.args[0], ast.Constant):
            return node.args[0].value
        return None

    out = {}
    for node in nodes:
        for n in ast.walk(node):
            if not (isinstance(n, ast.Compare) and len(n.ops) == 1):
                continue
            key, right = key_read(n.left), n.comparators[0]
            if key is None:
                continue
            if isinstance(n.ops[0], (ast.Eq, ast.NotEq)) and isinstance(right, ast.Constant) \
                    and isinstance(right.value, str):
                out.setdefault(key, set()).add(right.value)
            elif isinstance(n.ops[0], ast.In) and isinstance(right, (ast.List, ast.Tuple, ast.Set)):
                out.setdefault(key, set()).update(
                    e.value for e in right.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str))
    return out


@pytest.mark.parametrize('calc_type', sorted(calculation_classes()))
def test_every_value_the_engine_tests_for_is_one_the_menu_offers(calc_type):
    """A `values` list IS the menu, so a setting the engine acts on but the menu does not offer is
    UNREACHABLE from the schema - no panel, no validator, no schema-authored job.

    This is how the second-order block sat dormant: `BaseValuation.Greeks` declared
    `['First', 'No']` while `__init_shared_mem` tests `== 'All'`, so the whole `Greeks_Second` path
    had no way in. Hand-written JSON could still say `'All'`, which is why it was a silent gap
    rather than a broken run.

    Scoped to keys the type declares WITH a menu, for the default gate's reason.

    Killing mutation: `'All'` dropped from the `Greeks` menu again.
    """
    module_ast = ast.parse(inspect.getsource(calculation))
    cls_name = calculation_classes()[calc_type].__name__
    menus = declared_menus(CALCULATION['types'][calc_type])
    unreachable = {key: sorted(set(constants) - set(menus[key]))
                   for key, constants in compared_constants(cls_name, module_ast).items()
                   if key in menus and not set(constants) <= set(menus[key])}
    assert not unreachable, (
        f'{calc_type} acts on settings its menu cannot author: {unreachable}')


#: Switches a run reads, each declared where it reads it and nowhere else, at the default and menu
#: the engine means: `(type, key): (default, menu)`, and `(type, key): None` for a calculation that
#: must NOT declare one.
DECLARED_SWITCHES = {
    ('CreditMonteCarlo', 'Time_Grid'): ('0d 2d 1w(1w) 3m(1m) 2y(3m)', None),
    ('HedgeMonteCarlo', 'Time_Grid'): ('0d 1d(1d) 4m', None),
    ('CreditMonteCarlo', 'Base_Time_Grid'): None,
    ('BaseValuation', 'Greeks'): ('No', ['All', 'First', 'No']),
    ('BaseValuation', 'Recompute_Inner_MC'): ('No', ['Yes', 'No']),
    ('CreditMonteCarlo', 'Recompute_Inner_MC'): ('No', ['Yes', 'No']),
    ('BaseValuation', 'Branch_And_Weight'): ('Yes', ['Yes', 'No']),
    ('CreditMonteCarlo', 'Branch_And_Weight'): None,
    ('HedgeMonteCarlo', 'Branch_And_Weight'): None,
}


def test_a_switch_is_declared_where_the_run_reads_it():
    """A framework feature ships behind a JSON switch, declared on the calculation that reads it:
    the grid is `Time_Grid` (the store once declared `Base_Time_Grid` beside it, which a Workbench
    run wrote and nobody read); second derivatives are on the base valuation's `Greeks` menu at
    `All` and off by default; `Recompute_Inner_MC` is off by default on both valuations, the trade
    being the machine's; `Branch_And_Weight` is on by default on the base valuation alone, so the
    crisp path's exposure, cashflow and collateral semantics have no key an author could write
    to reach them. And a TARF's knock-in `Barrier`, which `pv_MC_Tarf` reads by name, is a field
    of its section, so a schema-authored TARF can switch its leveraged leg on.

    Killing mutations: `Greeks` defaulting to `All`; the TARF's `Barrier` withdrawn from its
    section; `Recompute_Inner_MC` defaulting to `Yes` on the credit Monte Carlo."""
    for (calc_type, key), declared in DECLARED_SWITCHES.items():
        found = CALCULATION['types'][calc_type].get(key)
        if declared is None:
            assert found is None, (calc_type, key, 'declared where nothing reads it')
        else:
            assert (found['value'], found.get('values')) == declared, (calc_type, key, found)
    assert 'Barrier' in INSTRUMENT['sections']['FXTARFOptionDeal.Fields']


def market_price_classes():
    """Bootstrapper classes carrying their own `fields`, keyed by the `Market Prices` type string
    they select their work by. Own-attr only: `HullWhite2FactorModelParameters` subclasses
    `RiskNeutralInterestRateModel`, which declares nothing and is no family of its own."""
    return {c.__dict__['market_factor_type']: c for c in vars(bootstrappers).values()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)
            and 'market_factor_type' in c.__dict__}


def test_the_quote_value_plane_is_published_beside_the_families():
    """A client ticks a quote by moving its VALUE columns and nothing else, and which columns those
    are is not a name it may spell: the store publishes the one tuple `update_market_quote` refuses
    against, so the screen that edits them cannot drift from the guard that admits them.

    The second half is the predicate a client finds a block's ladder BY - the declared field whose
    row carries every value key - held to `quote_containers`' own reading of the same declarations,
    because a table a client cannot find is a quote it cannot tick.

    Killing mutation: the published plane one key short of the tuple.
    """
    assert MARKET_PRICES['values'] == list(schema.MARKET_QUOTE_VALUES), (
        'the published value plane is not the tuple the tick guard reads')
    found = {key for section in MARKET_PRICES['types'].values() for key, d in section.items()
             if set(MARKET_PRICES['values']) <= set(d.get('col_names')
                                                    or d.get('sub_fields') or ())}
    assert found == set(schema.MARKET_QUOTE_CONTAINERS), (
        'the published declarations name {} as quote tables, the engine {}'.format(
            sorted(found), sorted(schema.MARKET_QUOTE_CONTAINERS)))


# The locals a bootstrapper binds from its own quote block, and therefore the reads that have to be
# declared: `implied_params['instrument']` IS the block, `instrument`/`block` are aliases, and
# `option`/`x`/`point` are one quote row. A family binding the block under a name not here is not
# gated at all, which is why this is a list rather than a guess.
QUOTE_LOCALS = ("implied_params['instrument']", 'instrument', 'block', 'option', 'x', 'point')


def quote_reads(cls_name, module_ast):
    """Every key a price family reads off its own quote block, hard-keyed or `.get`.

    Scoped to the class, its bases in this module, and the module-level helpers any of them call -
    which is how the Hull-White row reaches `create_market_swaps`, where its columns are consumed.

    A key ASSIGNED anywhere in that scope is computed by the bootstrapper rather than authored
    (`option['Premium']`, `option['T']`), and drops out. Subtraction is by key NAME, not by
    (base, key): the same row is `option` where it is written and `x` where it is read back.
    """
    classes = {n.name: n for n in module_ast.body if isinstance(n, ast.ClassDef)}
    funcs = {n.name: n for n in module_ast.body if isinstance(n, ast.FunctionDef)}
    nodes = [classes[cls_name]] + [classes[ast.unparse(b)] for b in classes[cls_name].bases
                                   if ast.unparse(b) in classes]
    nodes += [funcs[n.func.id] for node in list(nodes) for n in ast.walk(node)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs]
    read, written = set(), set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                    and isinstance(n.slice.value, str):
                if isinstance(n.ctx, ast.Store):
                    written.add(n.slice.value)
                elif ast.unparse(n.value) in QUOTE_LOCALS:
                    read.add(n.slice.value)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr == 'get' and ast.unparse(n.func.value) in QUOTE_LOCALS \
                    and n.args and isinstance(n.args[0], ast.Constant):
                read.add(n.args[0].value)
    return read - written


def declared_keys(descriptors):
    """Every JSON key a family's block can carry - table columns and container children too."""
    out = set()
    for key, d in descriptors.items():
        out.add(key)
        out |= set(d.get('col_names', []))
        out |= declared_keys(d.get('sub_fields', {}))
    return out


@pytest.mark.parametrize('market_type', sorted(market_price_classes()))
def test_the_quote_block_declares_what_the_bootstrapper_reads(market_type):
    """The declaration IS the quote block, held to the reads.

    Two live defects: the Hull-White row declared `Day_Count` while `create_market_swaps` hard-reads
    `Floating_Day_Count` and `Fixed_Day_Count`, so a schema-authored block raised KeyError before
    the first swaption priced; and `Weight` is read by the Clewlow-Strickland objective and declared
    only by the option family, so an energy option quote had no weight column.

    One direction only: the converse would need the option family's factor references, which `resolve`
    reads with a COMPUTED key, and `Generate_Instruments` / `Generation_Parameters`, declared as
    unbuilt functionality.

    Killing mutation: the FX smile family's `Grid_Tolerance` declared under another name."""
    module_ast = ast.parse(inspect.getsource(bootstrappers))
    cls_name = market_price_classes()[market_type].__name__
    undeclared = sorted(quote_reads(cls_name, module_ast) -
                        declared_keys(MARKET_PRICES['types'][market_type]))
    assert not undeclared, (
        f'{market_type} reads quote keys no schema-authored block can carry: {undeclared}')


def test_the_configuration_store_is_the_families_own_declarations():
    """A `Bootstrapper Configuration` entry's dials ARE the family's declarations - the same
    descriptor the MarketPrices store publishes, at the same default - less the Tables and
    Containers, which are the quote BLOCK's ladders and instrument definitions and belong to no
    section. Every family in the registry has an entry under the factor it writes, and every
    spelling the registry answers to reaches one, which is what lets a client file an older book's
    class-name key without knowing what a price family is.

    Killing mutation: a family's Containers published as dials."""
    store = schema.mapping['Configuration']
    entries = store['Bootstrapper Configuration']['types']
    assert set(entries) == {cls.price_factor_type for cls in bootstrappers.FAMILIES}

    for cls in bootstrappers.FAMILIES:
        declared = {f.key: f for f in cls.fields}
        dials = dict(entries[cls.price_factor_type]['fields'])
        stem = dials.pop(schema.PRICES_KEY)
        assert stem['value'] + 'Prices' == cls.market_factor_type, (
            f'{cls.price_factor_type} routes on {stem["value"]!r}, which is not its block')
        assert set(dials) == {key for key, f in declared.items()
                              if f.type not in ('Table', 'Container')}, (
            f'{cls.price_factor_type} publishes dials its family does not declare, or drops one')
        for key, descriptor in dials.items():
            assert descriptor == declared[key].descriptor(), f'{key} is not the declared field'
            assert descriptor['widget'] not in ('Table', 'Container'), f'{key} is a table'

    spellings = {name for key, entry in entries.items() for name in [key] + entry['aliases']}
    assert spellings == set(bootstrappers.WRITERS)
    for name in spellings:
        assert bootstrappers.family_class(name).price_factor_type in entries


def test_the_interpolation_section_references_the_menu_and_the_engines_own_default():
    """The second section states no methods of its own: it NAMES the menu beside it, and its value
    is what a routed factor is actually built with where the section declares nothing for it - read
    off a constructed factor, because a store agreeing with a constant it was built from would
    agree with the wrong one just as happily.

    Killing mutation: the section stating a default the factor is not built with."""
    from derivus.config import ModelParams

    declared = schema.mapping['Configuration']['Price Factor Interpolation']
    assert schema.mapping[declared['menu']] is INTERPOLATION_MAP
    factor = derivus.utils.Factor('InterestRate', ('USD',))
    built = riskfactors.construct_factor(factor, {'InterestRate.USD': {
        'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': derivus.utils.Curve([], [[0.0, 0.02], [5.0, 0.02]])}}, ModelParams())
    assert built.param['Interpolation'] == declared['value']


def declared_values(descriptors):
    """Every declared key's default value, containers flattened the way `declared_keys` flattens."""
    out = {}
    for key, d in descriptors.items():
        out[key] = d['value']
        out.update(declared_values(d.get('sub_fields', {})))
    return out


def fallback_reads(cls_name, module_ast):
    """`{key: fallback}` for every quote-block knob read as `<block>.get('Key', <constant>)`.

    Scoped to the class AND its bases in this module, the way `quote_reads` is: the swaption
    family's closure lives on `RiskNeutralInterestRateModel`, so a knob read there and declared on
    the subclass would otherwise publish two defaults with nothing holding them together.
    """
    classes = {n.name: n for n in module_ast.body if isinstance(n, ast.ClassDef)}
    nodes = [classes[cls_name]] + [classes[ast.unparse(b)] for b in classes[cls_name].bases
                                   if ast.unparse(b) in classes]
    return {n.args[0].value: n.args[1].value
            for node in nodes for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'get' and ast.unparse(n.func.value) in QUOTE_LOCALS
            and len(n.args) == 2 and all(isinstance(a, ast.Constant) for a in n.args)}


@pytest.mark.parametrize('market_type', sorted(market_price_classes()))
def test_a_declared_default_is_the_default_the_engine_falls_back_to(market_type):
    """A knob read with `.get(key, fallback)` publishes TWO defaults, and they have to be one: the
    panel gives the DECLARED value, an omitted key the FALLBACK, and nothing raises - the solve
    just runs to a different tolerance. The same mismatch class put a wrong grid KEY in the
    Calculation store beside the one the engine read.

    Killing mutation: the Hull-White block's `Simulations` read with a fallback of its own.
    """
    module_ast = ast.parse(inspect.getsource(bootstrappers))
    cls_name = market_price_classes()[market_type].__name__
    declared = declared_values(MARKET_PRICES['types'][market_type])
    drift = {key: (declared.get(key), fallback)
             for key, fallback in fallback_reads(cls_name, module_ast).items()
             if declared.get(key) != fallback}
    assert not drift, f'{market_type} declares one default and falls back to another: {drift}'


def test_a_quote_type_means_different_things_to_different_families():
    """The capability the per-type store exists for, pinned so a return to a flat one fails.

    The flat store published `ATM` / `Implied_Volatility` / `Premium` for every family. The
    Clewlow-Strickland bootstrapper supports only `Implied_Volatility`, the option family takes that
    or `Premium`, and an interest-rate quote is a par rate - three questions sharing one name, which
    is right because the JSON is per family.

    `InterestRatePrices` declares the one convention it implements: a value the solve does not
    implement is the same defect as a field nothing reads.

    `ATM` is a real quote type to exactly one family and is declared on a TABLE COLUMN, which the
    search below had to grow a leg to see - a gate that cannot see a declaration holds it to
    nothing.

    Killing mutation: `BF` dropped from the FX smile family's quote types."""
    def find(descriptors):
        for key, d in descriptors.items():
            if key == 'Quote_Type':
                return d['values']
            # a table declares its columns positionally: `sub_types` carries a dropdown's choices
            # under `source`, matched to `col_names` by index
            for name, sub in zip(d.get('col_names', []), d.get('sub_types', [])):
                if name == 'Quote_Type':
                    return sub['source']
            found = d.get('sub_fields') and find(d['sub_fields'])
            if found:
                return found

    quote_type = {t: find(d) for t, d in MARKET_PRICES['types'].items()}
    assert quote_type == {'CSForwardPriceModelPrices': ['Implied_Volatility'],
                          'LogVar2FJModelPrices': ['Implied_Volatility', 'Premium'],
                          'GBMAssetPriceTSModelPrices': None,
                          'HullWhite2FactorModelPrices': None,
                          'FXVolPrices': ['ATM', 'RR', 'BF'],
                          'InterestRatePrices': ['Par_Rate']}, quote_type
    assert [t for t, v in quote_type.items() if 'ATM' in (v or ())] == ['FXVolPrices'], (
        'ATM belongs to the FX smile family and to no other')


def interpolated_factor_types():
    """The factor types `construct_factor` routes through the `Price Factor Interpolation` section,
    read off the source.

    Parsed rather than listed, for the reason `dispatched_calculations` is: a hand-kept list here
    would be a second store of the same knowledge and would drift the same way."""
    src = ast.parse(inspect.getsource(riskfactors))
    fn = next(n for n in src.body
              if isinstance(n, ast.FunctionDef) and n.name == 'construct_factor')
    # `if factor.type in ['InterestRate', 'InflationRate']:`
    return {c.value for n in ast.walk(fn) if isinstance(n, ast.Compare)
            and any(isinstance(o, ast.In) for o in n.ops)
            for comp in n.comparators if isinstance(comp, (ast.List, ast.Tuple))
            for c in comp.elts if isinstance(c, ast.Constant)}


def test_the_interpolation_menu_is_the_routed_types_and_their_implemented_methods():
    """`Interpolation` is not a `Price Factors` key an author writes - `construct_factor` reads it
    out of the `Price Factor Interpolation` section and injects it, only for the types listed there.
    Every `Factor1D` honours the key once it has one, so a factor type outside that opt-in offers
    the author a setting the engine drops on the floor. And `Factor1D.check_interpolation` falls
    through to `Linear` for anything it does not know, so a method offered but not implemented is
    not an error - it is a curve silently interpolated the wrong way. The authored value also has
    to survive `factor_interp_map`, which is what `construct_factor` looks it up in.

    Killing mutations: `DividendRate` routed with no menu of its own; a method offered that
    `check_interpolation` does not know."""
    routed = interpolated_factor_types()
    assert set(INTERPOLATION_MAP) == routed, (
        f'interpolation menu and the routed types disagree: '
        f'{sorted(set(INTERPOLATION_MAP) ^ routed)}')
    src = ast.parse(textwrap.dedent(inspect.getsource(riskfactors.Factor1D.check_interpolation)))
    implemented = {n.value for n in ast.walk(src)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for factor_type, methods in INTERPOLATION_MAP.items():
        assert not set(methods) - implemented, (
            f'{factor_type} offers methods check_interpolation does not implement: '
            f'{sorted(set(methods) - implemented)}')
        assert not set(methods) - set(riskfactors.factor_interp_map), (
            f'{factor_type} offers methods factor_interp_map drops: '
            f'{sorted(set(methods) - set(riskfactors.factor_interp_map))}')


def calibration_classes():
    """Calibration classes carrying their own `fields`, keyed by the PROCESS they calibrate rather
    than by the class name - `HWInterestRateCalibration` calibrates
    `HullWhite1FactorInterestRateModel`."""
    return {c.__dict__['model_type']: c for c in vars(stochasticprocess).values()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)
            and 'model_type' in c.__dict__}


def calibration_source():
    """Every class in the module whose name says it is a calibration. Named rather than typed
    because the classes share no base - `calibrate()` is their whole surface - and `globals()`
    dispatch means a new one is discoverable the moment it is defined."""
    return {n: c for n, c in vars(stochasticprocess).items()
            if isinstance(c, type) and n.endswith('Calibration')}


def test_every_calibration_class_is_declared_under_a_process_and_dispatches_to_itself():
    """A calibration class with no schema row cannot be configured from the UI or found in the
    docs, however well it fits; the store used to describe two PROCESSES and no calibration class
    at all. An entry is filed under the PROCESS it configures - `Config.parse_json` keys
    `calibration_process_map` by it - so a type naming no process is an entry no factor ever
    reaches (the converse does not hold: the implied processes are bootstrapped, not calibrated).
    And `construct_calibration_config` does `globals().get(param['Method'])(model, param)`, so a
    `Method` naming no class is a TypeError as the config loads - which is why it is stamped from
    the class name.

    Killing mutations: `GBMAssetPriceCalibration`'s declaration withdrawn; its `model_type` naming
    no process."""
    missing = sorted(set(calibration_source()) -
                     {c.__name__ for c in calibration_classes().values()})
    assert not missing, f'calibration classes no schema can configure: {missing}'
    unknown = sorted(set(CALIBRATION['types']) - set(PROCESS['types']))
    assert not unknown, f'calibration types naming no declared process: {unknown}'
    undispatchable = sorted(
        model for model, d in CALIBRATION['types'].items()
        if not isinstance(getattr(stochasticprocess, d['Method']['value'], None), type))
    assert not undispatchable, f'calibration Methods that dispatch to no class: {undispatchable}'


def param_reads(cls):
    """Every `param` key a calibration class reads, hard-keyed or `.get`.

    Follows a local alias (`p = self.param`), which is how the Kalman calibration reads its ten
    knobs. Nothing else in these classes subscripts `self.param`."""
    src = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    aliases = {'self.param'} | {t.id for n in ast.walk(src) if isinstance(n, ast.Assign)
                                and ast.unparse(n.value) == 'self.param'
                                for t in n.targets if isinstance(t, ast.Name)}
    read = set()
    for n in ast.walk(src):
        if isinstance(n, ast.Subscript) and ast.unparse(n.value) in aliases \
                and isinstance(n.slice, ast.Constant):
            read.add(n.slice.value)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == 'get' and ast.unparse(n.func.value) in aliases \
                and isinstance(n.args[0], ast.Constant):
            read.add(n.args[0].value)
    return read


@pytest.mark.parametrize('model_type', sorted(calibration_classes()))
def test_the_declared_tuning_keys_are_the_ones_the_class_reads(model_type):
    """The declaration IS the tuning contract, held to the reads in both directions.

    Every knob the fourteen classes take was undeclared and nineteen descriptors were read by
    nothing - a whole `MLE_Parameters` tree, `Data_Retrieval_Parameters` and
    `Use_Pre_Computed_Statistics` - so the panel offered fields the fit ignores and none of the
    fields it honours. `Method` is exempt: it is stamped from the class name and read by
    `construct_calibration_config`, not by the class.

    Killing mutation: `GBMAssetPriceCalibration` declaring a knob it never reads."""
    cls = calibration_classes()[model_type]
    declared = {f.key for f in cls.__dict__['fields']}
    read = param_reads(cls)
    assert declared == read, (
        f'{cls.__name__} declares {sorted(declared - read)} it never reads and reads '
        f'{sorted(read - declared)} it never declares')


def test_the_descriptors_with_no_widget_are_exactly_these():
    """The known defect, pinned in both directions: a new shapeless descriptor fails here, and so
    does fixing one without updating the list.

    `define_input` reads `element['col_names']` for a Table and `element['sub_fields']` for a
    Container without checking, so every entry below raises KeyError the moment the Workbench
    renders it - which is every process in the platinum world, and the hedging problem itself. The
    declarations are not wrong: the shape of a transition matrix, a regime vector or a deal map
    keyed by Object then Reference is an OUTPUT, and the vocabulary has no way to say that.
    Migrating the two stores is what made the defect expressible; the fix wants a widget.

    Killing mutation: a Container's `sub_fields` left out of its descriptor."""
    found = set()

    def walk(type_name, key, d):
        if is_shapeless(d):
            found.add((type_name, key))
        for sub_key, sub in d.get('sub_fields', {}).items():
            walk(type_name, f'{key}.{sub_key}', sub)
    for store in (PROCESS['types'], CALCULATION['types'], FACTOR['types'], CALIBRATION['types'],
                  MARKET_PRICES['types']):
        for type_name, descriptors in store.items():
            for key, d in descriptors.items():
                walk(type_name, key, d)
    for section in INSTRUMENT['sections'].values():
        for key, d in section.items():
            walk('Instrument', key, d)
    assert found == SHAPELESS, (
        f'appeared: {sorted(found - SHAPELESS)}; fixed or gone: {sorted(SHAPELESS - found)}')
