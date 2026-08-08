"""`schema.mapping['Instrument']`, `['Factor']`, `['Process']` and `['Calculation']` are generated
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
import subprocess
import sys
import textwrap

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


def test_the_store_is_generated():
    """Guards the gate itself: every assertion below is vacuously true over an empty declaration
    set, so an import error or a filter bug would read as a green schema."""
    assert declared_classes(), 'no class declares `fields` - these gates are vacuous'
    assert INSTRUMENT['types'] == schema.emit_instrument(instruments)[0], (
        'the Instrument store is not the emitted view - a hand-written copy has come back')


def test_the_fields_shim_serves_the_same_objects():
    """`derivus.fields` is deprecated for one release and holds nothing of its own. `fields.mapping`
    was the documented surface and the package is on PyPI, so an external caller that bound it keeps
    working - on the same object, not a copy, which is the whole point of the retirement."""
    assert derivus.fields.mapping is schema.mapping
    assert derivus.fields.default is schema.default
    src = inspect.getsource(derivus.fields)
    assert 'mapping = {' not in src, 'the shim has grown a store of its own'


def test_the_store_survives_a_declaring_module_being_imported_first():
    """`schema.py` assembles `mapping` at the BOTTOM, after the vocabulary its declaring modules
    import from it. That is a one-way edge with an ordering question attached: a declaring module
    initialised first would have `emit_*` read a half-initialised module, and an emitter that found
    nothing would return an EMPTY store rather than raising - which every gate in this file would
    pass right through, since they compare the store to the emitter and both would be empty.

    In this package the wrong order raises instead, because `stochasticprocess` imports NAMES from
    `instruments` and a partially initialised module has none - but that is a property of the
    import graph rather than of the design, so it is not what this leans on. A submodule import
    always initialises the package first, `derivus/__init__` imports schema before any declaring
    module, and this holds that fixed. In a subprocess: by the time this module is collected the
    answer is already cached in `sys.modules`."""
    out = subprocess.run(
        [sys.executable, '-c', 'import derivus.instruments, derivus;'
         'print(len(derivus.schema.mapping["Instrument"]["types"]))'],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    assert out.returncode == 0, out.stderr
    assert int(out.stdout) == len(INSTRUMENT['types']), (
        'importing a declaring module first yields a different store')


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
# Reference, or a whole deal whose TYPE a sibling field names. `Table` declares fixed columns and
# `Container` fixed named children, so the vocabulary cannot state any of them, and the Workbench
# raises on all of them. Pinned by `test_the_descriptors_with_no_widget_are_exactly_these`, which
# is the known-defect gate: it fails both when one appears and when one is fixed. Keyed
# (type, dotted key).
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


def market_price_classes():
    """Bootstrapper classes carrying their own `fields`, keyed by the `Market Prices` type string
    they select their work by. Own-attr only: `HullWhite2FactorModelParameters` subclasses
    `RiskNeutralInterestRateModel`, which declares nothing and is no family of its own."""
    return {c.__dict__['market_factor_type']: c for c in vars(bootstrappers).values()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)
            and 'market_factor_type' in c.__dict__}


def test_the_market_prices_store_is_generated():
    """The same guard again - every MarketPrices assertion here is vacuous over an empty
    declaration set."""
    assert market_price_classes(), 'no bootstrapper declares `fields` - these gates are vacuous'
    assert MARKET_PRICES['types'] == schema.emit_market_prices(bootstrappers), (
        'the MarketPrices store is not the emitted view - a hand-written copy has come back')
    assert set(MARKET_PRICES) == {'types'}, (
        f'sub-stores have come back beside the types: {sorted(set(MARKET_PRICES) - {"types"})}')


@pytest.mark.parametrize('market_type', sorted(market_price_classes()))
def test_market_price_descriptor_shape(market_type):
    """`check_descriptor` over the price-family declarations - same tagged union, same consumers.
    The quote container and the generation parameters are where the nesting is."""
    for key, d in MARKET_PRICES['types'][market_type].items():
        check_shape(f'{market_type}.{key}', d)


@pytest.mark.parametrize('market_type', sorted(market_price_classes()))
def test_no_market_price_declares_a_key_twice(market_type):
    """One dict per type keyed by the JSON name, so a name declared twice loses a descriptor. The
    Heston-Nandi block builds its eight factor-reference fields from `factor_types`, which is
    exactly the shape that can produce one."""
    keys = [f.key for f in market_price_classes()[market_type].__dict__['fields']]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f'{market_type} declares {dupes} more than once'


# The locals a bootstrapper binds from its own quote block, and therefore the reads that have to be
# declared. `implied_params['instrument']` IS the block, `instrument` and `block` are its aliases,
# and `option`, `x` and `point` are one quote row - the loop and comprehension variables over the
# quote tables. A family binding the block under a name that is not here is not gated at all, which
# is why the list is a comment rather than a guess.
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

    Three live defects came out of this store and two of them are exactly this gate. The
    Hull-White row declared `Day_Count` while `create_market_swaps` hard-reads
    `Floating_Day_Count` and `Fixed_Day_Count`, so a block authored from the schema - or copied
    from the JSON reference's own example - raised KeyError before the first swaption priced. And
    `Weight` is read by the Clewlow-Strickland objective and declared only by Heston-Nandi, so an
    energy option quote had no weight column at all.

    One direction only. The converse would need the four Heston-Nandi factor references and their
    four `_Type` siblings, which `resolve` reads with a COMPUTED key off `factor_types` - the same
    attribute the declarations are built from - and `Generate_Instruments` /
    `Generation_Parameters`, which stay declared as unbuilt functionality."""
    module_ast = ast.parse(inspect.getsource(bootstrappers))
    cls_name = market_price_classes()[market_type].__name__
    undeclared = sorted(quote_reads(cls_name, module_ast) -
                        declared_keys(MARKET_PRICES['types'][market_type]))
    assert not undeclared, (
        f'{market_type} reads quote keys no schema-authored block can carry: {undeclared}')


def declared_values(descriptors):
    """Every declared key's default value, containers flattened the way `declared_keys` flattens."""
    out = {}
    for key, d in descriptors.items():
        out[key] = d['value']
        out.update(declared_values(d.get('sub_fields', {})))
    return out


def fallback_reads(cls_name, module_ast):
    """`{key: fallback}` for every quote-block knob read as `<block>.get('Key', <constant>)`."""
    classes = {n.name: n for n in module_ast.body if isinstance(n, ast.ClassDef)}
    return {n.args[0].value: n.args[1].value
            for n in ast.walk(classes[cls_name])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'get' and ast.unparse(n.func.value) in QUOTE_LOCALS
            and len(n.args) == 2 and all(isinstance(a, ast.Constant) for a in n.args)}


@pytest.mark.parametrize('market_type', sorted(market_price_classes()))
def test_a_declared_default_is_the_default_the_engine_falls_back_to(market_type):
    """A knob read with `.get(key, fallback)` publishes TWO defaults, and they have to be one.

    An author who fills the panel in gets the DECLARED value; a block that omits the key gets the
    engine's FALLBACK. Where those disagree the same job means two different things depending on
    whether it was hand-written or authored from the schema, and nothing raises - the solve just
    runs to a different tolerance. This is the mismatch class that put `Base_Time_Grid` in the
    Calculation store beside a `Time_Grid` the engine read: the key was wrong there rather than the
    value, and the same gate catches both.
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
    Clewlow-Strickland bootstrapper logs `quote_type ... not supported yet` for anything but
    `Implied_Volatility`, Heston-Nandi takes that or `Premium`, and an interest-rate quote is a par
    rate - three different questions sharing one name, which is right, because the JSON is per
    family.

    `InterestRatePrices` declares the one convention it implements. It used to offer `Rate` and
    `Price` as well, which nothing read: a futures price and a money-market rate on a different
    basis are conventions the family would have to author differently, and a value the solve does
    not implement is the same defect as a field nothing reads."""
    def find(descriptors):
        for key, d in descriptors.items():
            if key == 'Quote_Type':
                return d['values']
            found = d.get('sub_fields') and find(d['sub_fields'])
            if found:
                return found

    quote_type = {t: find(d) for t, d in MARKET_PRICES['types'].items()}
    assert quote_type == {'CSForwardPriceModelPrices': ['Implied_Volatility'],
                          'HestonNandiModelPrices': ['Implied_Volatility', 'Premium'],
                          'GBMAssetPriceTSModelPrices': None,
                          'HullWhite2FactorModelPrices': None,
                          'InterestRatePrices': ['Par_Rate']}, quote_type
    assert not any('ATM' in (v or ()) for v in quote_type.values()), (
        'ATM is back, and no family takes it')


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


def test_the_interpolation_map_is_generated():
    """The same guard again - the two Interpolation assertions below are vacuous over an empty
    declaration set."""
    assert INTERPOLATION_MAP, 'no factor class declares `interpolation_methods`'
    assert INTERPOLATION_MAP == schema.emit_interpolation(riskfactors), (
        'the interpolation menu is not the emitted view - a hand-written copy has come back')


def test_the_interpolation_menu_is_the_types_the_engine_routes():
    """`Interpolation` is not a `Price Factors` key an author writes - `construct_factor` reads it
    out of the `Price Factor Interpolation` section and injects it, and only for the types listed
    there. Every `Factor1D` honours the key once it has one, so the restriction the two
    hand-written rows carried was never about the interpolation code: it is this opt-in, and a
    factor type outside it offers the author a setting the engine drops on the floor."""
    routed = interpolated_factor_types()
    assert set(INTERPOLATION_MAP) == routed, (
        f'interpolation menu and the routed types disagree: '
        f'{sorted(set(INTERPOLATION_MAP) ^ routed)}')


def test_every_offered_interpolation_method_is_implemented():
    """`Factor1D.check_interpolation` falls through to `Linear` for anything it does not know, so a
    method offered but not implemented is not an error - it is a curve silently interpolated the
    wrong way. The authored value also has to survive `factor_interp_map`, which is what
    `construct_factor` looks it up in."""
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


def test_the_calibration_store_is_generated():
    """The same guard again - every Calibration assertion here is vacuous over an empty
    declaration set."""
    assert calibration_classes(), 'no calibration class declares `fields` - these gates are vacuous'
    assert CALIBRATION['types'] == schema.emit_calibration(stochasticprocess), (
        'the Calibration store is not the emitted view - a hand-written copy has come back')
    assert 'fields' not in CALIBRATION, 'a flat name-keyed store has come back beside the types'


def test_every_calibration_class_is_declarable():
    """A calibration class with no schema row cannot be configured from the UI or found in the
    docs, however well it fits. The store used to describe two PROCESSES and no calibration class
    at all: it was keyed by the process while `construct_calibration_config` dispatches on the
    entry's own `Method`, so the type, the block and the class carried three different names."""
    missing = sorted(set(calibration_source()) -
                     {c.__name__ for c in calibration_classes().values()})
    assert not missing, f'calibration classes no schema can configure: {missing}'


def test_every_calibration_type_names_a_declared_process():
    """A `Calibrations` entry is filed under the PROCESS it configures - `Config.parse_json` keys
    `calibration_process_map` by it and `fetch_all_calibration_factors` looks a factor's model up
    in that map - so a type naming no process is an entry no factor ever reaches.

    The converse does NOT hold and is not gated: the implied/risk-neutral processes are
    bootstrapped from market prices rather than calibrated from an archive, so they have no
    calibration class and want none."""
    unknown = sorted(set(CALIBRATION['types']) - set(PROCESS['types']))
    assert not unknown, f'calibration types naming no declared process: {unknown}'


def test_every_calibration_method_is_dispatchable():
    """`construct_calibration_config` does `globals().get(param['Method'])(model, param)`, so a
    `Method` naming no class is `None(...)` - a TypeError as the calibration config loads. The
    descriptor's value is stamped from the class name for exactly that reason."""
    undispatchable = sorted(
        model for model, d in CALIBRATION['types'].items()
        if not isinstance(getattr(stochasticprocess, d['Method']['value'], None), type))
    assert not undispatchable, f'calibration Methods that dispatch to no class: {undispatchable}'


@pytest.mark.parametrize('model_type', sorted(calibration_classes()))
def test_calibration_descriptor_shape(model_type):
    """`check_descriptor` over the calibration declarations - same tagged union, same consumers."""
    for key, d in CALIBRATION['types'][model_type].items():
        check_shape(f'{model_type}.{key}', d)


@pytest.mark.parametrize('model_type', sorted(calibration_classes()))
def test_no_calibration_declares_a_key_twice(model_type):
    """One dict per type keyed by the JSON name, so a name declared twice loses a descriptor."""
    keys = ['Method'] + [f.key for f in calibration_classes()[model_type].__dict__['fields']]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f'{model_type} declares {dupes} more than once'


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

    Every knob the thirteen classes take was undeclared and nineteen descriptors were read by
    nothing - a whole `MLE_Parameters` tree, `Data_Retrieval_Parameters` and
    `Use_Pre_Computed_Statistics` - so the panel offered fields the fit ignores and none of the
    fields it honours. `Method` is exempt: it is stamped from the class name and read by
    `construct_calibration_config`, not by the class."""
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
    Migrating the two stores is what made the defect expressible; the fix wants a widget."""
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


def test_a_2d_and_a_3d_surface_may_both_be_called_surface():
    """The capability the per-type store exists for, pinned so a return to a flat one fails.

    `Surface` is a (moneyness, expiry, vol) triple list on a `VolatilityGrid` and a quad list on
    the three vol SPACES, and both are right - the JSON is per factor type. Under a flat store one
    of them had to be filed as `Space` and carry `Surface` as an alias."""
    assert FACTOR['types']['VolatilityGrid']['Surface']['value'] == schema.BLANK['Surface']
    for space in ('InterestYieldVol', 'InterestRateVol', 'ForwardPriceVol'):
        assert FACTOR['types'][space]['Surface']['value'] == schema.BLANK['Space']
