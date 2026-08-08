"""`fields.mapping['Instrument']`, `['Factor']`, `['Process']` and `['Calculation']` are generated
from the per-class `fields` declarations, and a SECTION (deals) or a TYPE (everything else) owns
its descriptors. So there is no round-trip to gate - the old test diffed the emitted dict against a
hand-written one, and both sides are now the same object.

Three defect classes are unreachable by construction rather than merely absent, which is why the
gates that guarded them are gone:

  - a type naming no class (`SwapBasisDeal`, `SwapCurrencyDeal`: the UI offered them and
    `construct_instrument` logged and returned `{}`) - a type IS a class that declares fields
  - a section naming a field with no descriptor, and a descriptor no section reaches - a section
    IS its descriptors, and a descriptor exists only inside one
  - one field name silently resolving to another deal's descriptor - each section holds its own

What remains gateable is what the declarations can still get wrong: a malformed descriptor, a
section declared two ways by two classes, a shared group copied instead of shared, a deal type that
no create-menu offers, a process no factor menu offers, and a calculation type `run_job` cannot
dispatch.
"""
import ast
import inspect
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import derivus
from derivus import calculation, fields, instruments, riskfactors, schema, stochasticprocess

INSTRUMENT = fields.mapping['Instrument']
FACTOR = fields.mapping['Factor']
PROCESS = fields.mapping['Process']
PROCESS_FACTOR_MAP = fields.mapping['Process_factor_map']
CALCULATION = fields.mapping['Calculation']

# riskfactors classes that legitimately declare no schema row of their own.
UNDECLARED_FACTORS = {
    'Factor0D': 'dimension base', 'Factor1D': 'dimension base',
    'Factor2D': 'dimension base', 'Factor3D': 'dimension base',
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
    empty list, so it is a type in its own right rather than an alias re-emitting the parent's."""
    return {n: c.__dict__['fields'] for n, c in vars(stochasticprocess).items()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)}


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


def test_the_store_is_generated():
    """Guards the gate itself: every assertion below is vacuously true over an empty declaration
    set, so an import error or a filter bug would read as a green schema."""
    assert declared_classes(), 'no class declares `fields` - these gates are vacuous'
    assert INSTRUMENT['types'] == schema.emit_instrument(instruments)[0], (
        'the Instrument store is not the emitted view - a hand-written copy has come back')


@pytest.mark.parametrize('cls_name', sorted(declared_classes()))
def test_descriptor_shape(cls_name):
    """The descriptor is a tagged union on `widget`, and the consumers destructure it positionally
    without checking. A `Table` missing `sub_types` raises in the UI's cell renderer; a `Dropdown`
    missing `values` renders an empty list the author cannot pick from.

    Parametrized over CLASSES and reading the DECLARATIONS, not the emitted store. Sections are
    shared objects, so walking the store visits `Admin` once however many types list it; walking
    the classes checks every declaration in the place an author would edit it.
    """
    for group in declared_classes()[cls_name]:
        for f in every_field(group):
            check_descriptor(f'{cls_name}.{group.name}.{f.key}', f.descriptor())


# Descriptors whose JSON value is an array or a map whose SHAPE is an OUTPUT - an NxN transition
# matrix, a length-N regime vector, a list of per-regime dicts, a deal map keyed by Object then by
# Reference. `Table` declares fixed columns and `Container` fixed named children, so the
# vocabulary cannot state any of them, and the Workbench raises on all of them. Pinned by
# `test_the_descriptors_with_no_widget_are_exactly_these`, which is the known-defect gate: it
# fails both when one appears and when one is fixed. Keyed (type, dotted key).
SHAPELESS = {
    ('MarkovSwitchingLogOUSpotModel', 'States'),
    ('MarkovSwitchingLogOUSpotModel', 'Transition_Matrix'),
    ('MarkovSwitchingLogOUSpotModel', 'Initial_State_Probs'),
    ('MarkovHMMSpotModel', 'States'),
    ('MarkovHMMSpotModel', 'Transition_Matrix'),
    ('MarkovHMMSpotModel', 'Initial_State_Probs'),
    ('VARMixedFactorInterestRateModel', 'Mean'),
    ('VARMixedFactorInterestRateModel', 'Phi'),
    ('VARMixedFactorInterestRateModel', 'Sigma'),
    ('VARMixedFactorInterestRateModel', 'Calibration_Tenors'),
    ('BasisLinkedSpotModel', 'Sigma_By_State'),
    ('CreditMonteCarlo', 'Credit_Valuation_Adjustment.CDS_Tenors'),
    ('HedgeMonteCarlo', 'Hedging_Problem.Tradable_Instruments'),
    ('HedgeMonteCarlo', 'Hedging_Problem.Liabilities'),
    ('HedgeMonteCarlo', 'Hedging_Problem.Portfolio_State'),
    ('HedgeMonteCarlo', 'Hedging_Problem.Objective'),
    ('HedgeMonteCarlo', 'Hedging_Problem.Evaluator'),
    ('HedgeMonteCarlo', 'Hedging_Problem.Solver'),
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


def test_no_class_is_hidden_from_the_create_menu():
    """`groups` is the Workbench's create-deal menu and stays hand-curated, being presentation. So
    it is the one part of the store that can still drift from the classes: a deal type absent from
    every group is fully declared, fully priceable and unreachable from the UI."""
    menued = {t for members in INSTRUMENT['groups'].values() for t in members}
    assert not sorted(set(INSTRUMENT['types']) - menued), (
        f'declared deal types in no menu group: {sorted(set(INSTRUMENT["types"]) - menued)}')


def test_one_name_may_carry_two_descriptors_in_different_sections():
    """The capability the per-section store exists for, pinned so a return to a flat one fails.

    `Payment_Timing` is `Touch`/`Expiry` on a one-touch and `End`/`Begin`/`Discounted` on a
    cashflow leg. Both are right - the JSON is per-deal, so the data is never ambiguous and only a
    store keyed by field name across every deal was. Under that store one of these silently won,
    and the loser had to invent a descriptor key (`Option_Payment_Timing`) and carry its real name
    as an alias.
    """
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


@pytest.mark.parametrize('cls_name', sorted(declared_classes()))
def test_no_group_declares_a_key_twice(cls_name):
    """A section is now a dict keyed by the JSON name, so a name declared twice in one group loses
    a descriptor outright rather than merely shadowing one. `Net_Cashflows` was declared twice
    verbatim on `StructuredDeal` and the duplicate was invisible.

    The cross-class version of this check is gone on purpose - two SECTIONS may key the same name
    differently, which is the point of the per-section store."""
    for group in declared_classes()[cls_name]:
        keys = [f.key for f in group.fields]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        assert not dupes, f'{cls_name}.{group.name} declares {dupes} more than once'


def test_no_section_is_declared_two_ways():
    """The same hazard one level up: `sections` is keyed by group NAME, so two groups sharing a name
    and differing in fields silently collapse to one panel."""
    seen = {}
    for cls_name, groups in declared_classes().items():
        for g in groups:
            seen.setdefault(g.name, {}).setdefault(tuple(f.key for f in g.fields), []).append(cls_name)
    clashing = {k: {c[0] for c in v.values()} for k, v in seen.items() if len(v) > 1}
    assert not clashing, f'one section name declared with differing fields: {clashing}'


def test_shared_groups_are_shared_not_copied():
    """`FXAdmin` is one object listed by eight classes. Copying it per class would let the copies
    drift - which is the whole defect the name-keyed dict has, reintroduced one level down."""
    users = [f for f in declared_classes().values() if any(g.name == 'FXAdmin' for g in f)]
    assert len(users) > 1, 'FXAdmin is declared by fewer than two classes - nothing to share'
    groups = {id(g) for f in users for g in f if g.name == 'FXAdmin'}
    assert len(groups) == 1, f'FXAdmin exists as {len(groups)} distinct objects, not one'


def test_the_factor_store_is_generated():
    """The same guard as above, for factors: every Factor assertion here is vacuously true over an
    empty declaration set."""
    assert factor_classes(), 'no riskfactors class declares `fields` - these gates are vacuous'
    assert FACTOR['types'] == schema.emit_factor(riskfactors), (
        'the Factor store is not the emitted view - a hand-written copy has come back')
    assert 'fields' not in FACTOR, 'a flat name-keyed store has come back beside the types'


def test_every_declared_factor_type_is_constructible():
    """`construct_factor` does `globals().get(factor.type)(block)`, so a declared type naming no
    class is not a logged miss like a deal - it is `None(block)`, a TypeError at compile time.
    `ConvenienceYield` sat in exactly that state, declared with two fields and a process-map row
    and with no class in this repo. A type IS a class that declares fields, so it is unreachable."""
    undispatchable = sorted(set(FACTOR['types']) - set(riskfactor_classes()))
    assert not undispatchable, f'schema offers factor types with no class: {undispatchable}'


def test_every_riskfactor_class_is_declarable():
    """The converse: a factor class no schema declares cannot be authored or documented. The
    exemptions are the four dimension bases, which are never a `Factor.type`, and the Jacobian,
    whose block is keyed by benchmark instrument rather than by a fixed field set."""
    missing = sorted(set(riskfactor_classes()) - set(FACTOR['types']) - set(UNDECLARED_FACTORS))
    assert not missing, f'riskfactors classes no schema can author: {missing}'


@pytest.mark.parametrize('cls_name', sorted(factor_classes()))
def test_factor_descriptor_shape(cls_name):
    """`check_descriptor` again, over the factor declarations - same tagged union, same
    consumers."""
    for f in factor_classes()[cls_name]:
        check_descriptor(f'{cls_name}.{f.key}', f.descriptor())


@pytest.mark.parametrize('cls_name', sorted(factor_classes()))
def test_no_factor_declares_a_key_twice(cls_name):
    """A factor type is one dict keyed by the JSON name, so a name declared twice loses a
    descriptor outright."""
    keys = [f.key for f in factor_classes()[cls_name]]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f'{cls_name} declares {dupes} more than once'


def test_the_process_store_is_generated():
    """The same guard again, for processes - every Process assertion here is vacuous over an empty
    declaration set."""
    assert process_classes(), 'no stochasticprocess class declares `fields` - these gates are vacuous'
    types, factor_map = schema.emit_process(stochasticprocess, FACTOR['types'])
    assert PROCESS['types'] == types, (
        'the Process store is not the emitted view - a hand-written copy has come back')
    assert PROCESS_FACTOR_MAP == factor_map, 'the process/factor map is not the emitted view'
    assert 'fields' not in PROCESS, 'a flat name-keyed store has come back beside the types'


def test_every_declared_process_type_is_dispatchable():
    """`construct_process` does `globals().get(sp_type)(factor, param, implied_factor)`, so a
    declared type naming no class is `None(...)` - a TypeError as the scenario engine builds,
    after the market data has loaded and the deals have compiled."""
    undispatchable = sorted(set(PROCESS['types']) - set(concrete_processes()))
    assert not undispatchable, f'schema offers process types with no class: {undispatchable}'


def test_every_process_class_is_declarable():
    """The converse: a process class no schema declares cannot be authored from the Workbench or
    found in the JSON reference, however well it simulates. `GARCHSpotModel` sat in exactly that
    state - calibrated, shipped in a fixture, documented in the theory pages, and absent from both
    the Price Models panel and every factor's process menu."""
    missing = sorted(set(concrete_processes()) - set(PROCESS['types']))
    assert not missing, f'process classes no schema can author: {missing}'


@pytest.mark.parametrize('cls_name', sorted(process_classes()))
def test_process_descriptor_shape(cls_name):
    """`check_descriptor` again, over the process declarations - same tagged union, same
    consumers. The shapeless arrays are pinned separately, below."""
    for f in process_classes()[cls_name]:
        check_shape(f'{cls_name}.{f.key}', f.descriptor())


@pytest.mark.parametrize('cls_name', sorted(process_classes()))
def test_no_process_declares_a_key_twice(cls_name):
    """A process type is one dict keyed by the JSON name, so a name declared twice loses a
    descriptor outright."""
    keys = [f.key for f in process_classes()[cls_name]]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f'{cls_name} declares {dupes} more than once'


def test_every_factor_type_has_a_process_menu():
    """The Workbench indexes the map by the type of the factor in front of it
    (`possible_risk_process[factor.type]`), so a factor type with no entry is a KeyError that takes
    the whole Price Factors page down - not an empty dropdown. Emitting the keys from the factor
    declarations is what makes that unreachable."""
    assert set(PROCESS_FACTOR_MAP) == set(FACTOR['types']), (
        f'process menu and factor types disagree: '
        f'{sorted(set(PROCESS_FACTOR_MAP) ^ set(FACTOR["types"]))}')


def test_every_mapped_process_is_a_declared_type():
    """The menu offers a process by name and the panel then looks its descriptors up by that name,
    so an entry naming no declared type is a KeyError one click later."""
    offered = {p for members in PROCESS_FACTOR_MAP.values() for p in members}
    assert not offered - set(PROCESS['types']), (
        f'process menu offers undeclared types: {sorted(offered - set(PROCESS["types"]))}')


def test_every_process_reaches_a_factor_menu():
    """The converse, which is the one that was drifting: a process the engine constructs but no
    factor's menu offers cannot be selected in the Workbench at all. Three implied models were in
    that state (`CSImpliedForwardPriceModel`, `HullWhite2FactorImpliedInterestRateModel`, and
    `GBMAssetPriceTSModelImplied` on equity, which its own `calc_references` handles), plus
    `GARCHSpotModel`, which was in no store at all."""
    offered = {p for members in PROCESS_FACTOR_MAP.values() for p in members}
    assert not set(PROCESS['types']) - offered, (
        f'declared processes no factor menu offers: {sorted(set(PROCESS["types"]) - offered)}')


def test_one_name_may_carry_two_shapes_in_different_processes():
    """The capability the per-type store exists for, pinned so a return to a flat one fails.

    Three names carry two shapes each. `Sigma` is a scalar on the OU/hazard/Clewlow-Strickland
    models and a term-structure curve on Hull-White - under the flat store the scalar had to be
    filed as `sigma` and carry `Sigma` as an alias, which is the last Process entry in
    `ALIASED_KEYS`. `Phi` is a 3x3 VAR transition matrix on `VARMixedFactorInterestRateModel` and a
    scalar AR(1) coefficient on `BasisLinkedSpotModel`; the flat store rendered the basis
    coefficient as a matrix table. And `VARMixedFactorInterestRateModel.Sigma` is a length-3 vector,
    which the flat store rendered as the Hull-White curve widget."""
    types = PROCESS['types']
    assert types['LogOUSpotModel']['Sigma']['widget'] == 'Float'
    assert types['HullWhite1FactorInterestRateModel']['Sigma']['widget'] == 'Flot'
    assert types['VARMixedFactorInterestRateModel']['Sigma']['widget'] == 'Container'
    assert types['VARMixedFactorInterestRateModel']['Phi']['widget'] == 'Table'
    assert types['BasisLinkedSpotModel']['Phi']['widget'] == 'Float'
    assert not any('sigma' in d for d in types.values()), 'the lowercase alias key is back'


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


def test_the_calculation_store_is_generated():
    """The same guard again - every Calculation assertion here is vacuous over an empty
    declaration set."""
    assert calculation_classes(), 'no calculation class declares `fields` - these gates are vacuous'
    assert CALCULATION['types'] == schema.emit_calculation(calculation), (
        'the Calculation store is not the emitted view - a hand-written copy has come back')
    assert 'fields' not in CALCULATION, 'a flat name-keyed store has come back beside the types'


def test_every_declared_calculation_type_is_dispatchable():
    """`run_job` branches on the `Object` string and RAISES on a miss, so a declared type naming no
    branch is a calculation the Workbench offers in its create menu and the engine refuses to run.

    The type is the `Object` string, which is not the class name: `Base_Revaluation` is authored as
    `BaseValuation`. The class states its own with `calc_type` rather than the emitter unmangling
    underscores, because no rule recovers one of those words from the other."""
    undispatchable = sorted(set(CALCULATION['types']) - dispatched_calculations())
    assert not undispatchable, f'schema offers calculations run_job cannot dispatch: {undispatchable}'


def test_every_dispatchable_calculation_is_declared():
    """The converse, which is where the drift was: `HedgeMonteCarlo` had a `run_job` branch, a
    documented `Hedging_Problem` contract and two shipped fixtures, and no schema row at all. So
    the Workbench's create menu did not offer it, and opening a job that used it raised KeyError in
    `CalculationPage.load_items` - the store is indexed by the block's own `Object`."""
    undeclared = sorted(dispatched_calculations() - set(CALCULATION['types']))
    assert not undeclared, f'calculations run_job dispatches that no schema declares: {undeclared}'


@pytest.mark.parametrize('calc_type', sorted(calculation_classes()))
def test_calculation_descriptor_shape(calc_type):
    """`check_descriptor` over the calculation declarations, containers recursed - the CVA, FVA,
    CollVA and initial-margin blocks are where the nesting is."""
    for key, d in CALCULATION['types'][calc_type].items():
        check_shape(f'{calc_type}.{key}', d)


@pytest.mark.parametrize('calc_type', sorted(calculation_classes()))
def test_no_calculation_declares_a_key_twice(calc_type):
    """One dict per type keyed by the JSON name, so a name declared twice loses a descriptor."""
    keys = [f.key for f in calculation_classes()[calc_type].__dict__['fields']]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f'{calc_type} declares {dupes} more than once'


def test_the_calculation_time_grid_is_the_key_the_engine_reads():
    """The drift this migration fixed. The store declared `Base_Time_Grid`; `run_cmc` and
    `run_hedgemontecarlo` read `calc_params.get('Time_Grid', ...)` and every fixture and doc
    writes `Time_Grid`, so the Workbench's grid field wrote a key nobody reads and a
    Workbench-authored run silently took the hardcoded default grid."""
    assert 'Time_Grid' in CALCULATION['types']['CreditMonteCarlo']
    assert 'Time_Grid' in CALCULATION['types']['HedgeMonteCarlo']
    assert 'Base_Time_Grid' not in CALCULATION['types']['CreditMonteCarlo']
    src = inspect.getsource(derivus)
    assert "calc_params.get('Time_Grid'" in src and 'Base_Time_Grid' not in src


def test_the_descriptors_with_no_widget_are_exactly_these():
    """The known defect, pinned in both directions: a new shapeless descriptor fails here, and so
    does fixing one without updating the list.

    `define_input` reads `element['col_names']` for a Table and `element['sub_fields']` for a
    Container without checking, so every entry below raises KeyError the moment the Workbench
    renders it - which is every process in the platinum world, and the hedging problem itself. The
    declarations are not wrong: the shape of a transition matrix, a regime vector or a deal map
    keyed by Object then Reference is an OUTPUT, and the vocabulary has no way to say that.
    Migrating the two stores is what made the defect expressible; the fix wants a widget."""
    found = set()

    def walk(type_name, key, d):
        if is_shapeless(d):
            found.add((type_name, key))
        for sub_key, sub in d.get('sub_fields', {}).items():
            walk(type_name, f'{key}.{sub_key}', sub)
    for store in (PROCESS['types'], CALCULATION['types'], FACTOR['types']):
        for type_name, descriptors in store.items():
            for key, d in descriptors.items():
                walk(type_name, key, d)
    for section in INSTRUMENT['sections'].values():
        for key, d in section.items():
            walk('Instrument', key, d)
    assert found == SHAPELESS, (
        f'appeared: {sorted(found - SHAPELESS)}; fixed or gone: {sorted(SHAPELESS - found)}')


def test_a_2d_and_a_3d_surface_may_both_be_called_surface():
    """The capability the per-type store exists for, pinned so a return to a flat one fails.

    `Surface` is a (moneyness, expiry, vol) triple list on a `VolatilityGrid` and a quad list on
    the three vol SPACES, and both are right - the JSON is per factor type. Under a flat store one
    of them had to be filed as `Space` and carry `Surface` as an alias."""
    assert FACTOR['types']['VolatilityGrid']['Surface']['value'] == schema.BLANK['Surface']
    for space in ('InterestYieldVol', 'InterestRateVol', 'ForwardPriceVol'):
        assert FACTOR['types'][space]['Surface']['value'] == schema.BLANK['Space']
