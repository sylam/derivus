"""`fields.mapping` is a third store of field knowledge, keyed by name by CONVENTION, and nothing
checks that convention against the classes it describes.

The engine never reads `fields.py` - `construct_instrument` takes the raw JSON - so every drift here
is invisible to the valuation suite and shows up only as an authoring failure: a deal type the UI
offers but `globals()` cannot dispatch (logged, then dropped as `{}`), a field a pricer reads but no
schema-authored deal can emit, a widget that reads a JSON key nobody writes.

Each test below is one leg of the class<->schema correspondence, and each exemption list names WHY
a case is deliberate. The lists are the interesting part: `ALIASED_KEYS` in particular enumerates
every field whose descriptor key had to be invented because a name-keyed dict cannot give two deals
different descriptors for the same field name - which is the cost the per-class `fields` declaration
exists to remove.
"""
import inspect

import pytest

from derivus import calculation, fields, instruments, riskfactors, stochasticprocess

MAPPING = fields.mapping
INSTRUMENT = MAPPING['Instrument']

# Deal subclasses that legitimately have no schema row of their own.
UNDECLARED_DEALS = {
    'Deal': 'abstract base',
    'FXEuropeanOption': 'alias subclass of FXOptionDeal - no own fields, authored as the parent',
    'StructuredDealBreakClause': 'alias subclass of StructuredDeal - no own fields',
}

# Descriptor keys that are NOT the JSON field name, per store -> the key they actually read.
# A name-keyed dict admits one descriptor per name, so a field needing different widgets or
# different valid values in two deals must invent a key and carry the real name elsewhere:
# `Barrier_Type` is Up/Down for a one-touch and Down_And_In/... for a barrier, so `Barrier_Type_One`
# exists. `sigma` is aliased in Process but genuine in Factor, which is why this is keyed by store.
# A per-class declaration gives each class its own descriptor and every entry here disappears.
ALIASED_KEYS = {
    'Instrument': {
        'FloatItems': 'Items', 'RealYieldItems': 'Items', 'FixedItems': 'Items',
        'FixedSimpleItems': 'Items', 'EnergyItems': 'Items', 'EquityItems': 'Items',
        'EnergyFixedItems': 'Items',
        'Float_Cashflows': 'Cashflows', 'Equity_Cashflows': 'Cashflows',
        'Fixed_Cashflows': 'Cashflows', 'Fixed_Simple_Cashflows': 'Cashflows',
        'Real_Yield_Cashflows': 'Cashflows',
        'Energy_Cashflows': 'Payments', 'Energy_Fixed_Cashflows': 'Payments',
        'Index_Reference_Type': 'Reference_Type', 'Adjustment_Method': 'Rate_Adjustment_Method',
        'Barrier_Type_One': 'Barrier_Type', 'Option_Payment_Timing': 'Payment_Timing',
    },
    'Factor': {'Space': 'Surface'},
    'Process': {'sigma': 'Sigma'},
    'Calibration': {'Number_PCA_Factors': 'Number_Of_PCA_Factors'},
}


def deal_classes():
    return {n: c for n, c in vars(instruments).items()
            if inspect.isclass(c) and issubclass(c, instruments.Deal)}


def declared_fields(deal_type):
    """Every field name a schema-authored deal of this type can carry."""
    return {f for section in INSTRUMENT['types'][deal_type] for f in INSTRUMENT['sections'][section]}


@pytest.mark.xfail(strict=True, reason='DepositDeal / SwapBasisDeal / SwapCurrencyDeal are declared '
                                       'but unwritten - wanted for bootstrapping and the calibration '
                                       'Jacobians. Implementing them flips this to XPASS.')
def test_every_declared_deal_type_is_dispatchable():
    """`construct_instrument` does `globals().get(param['Object'])`, and on a miss LOGS AN ERROR and
    returns `{}` - the deal vanishes from the portfolio rather than raising. So a schema type naming
    no class is a deal the UI offers, the docs document, and the engine silently drops."""
    undispatchable = sorted(set(INSTRUMENT['types']) - set(deal_classes()))
    assert not undispatchable, (
        f'schema offers deal types that instruments.globals() cannot construct: {undispatchable}')


def test_every_deal_class_is_declarable():
    """The converse: a Deal subclass with no schema row cannot be authored from the UI or the docs
    at all, however well the pricer works."""
    missing = sorted(set(deal_classes()) - set(INSTRUMENT['types']) - set(UNDECLARED_DEALS))
    assert not missing, f'Deal subclasses no schema can author: {missing}'


def test_every_group_member_is_a_declared_type():
    """`groups` drives the UI's create-deal context menu. A member naming no declared type offers a
    deal the panel cannot build. This caught a missing comma between two adjacent entries, which
    Python concatenated into `FXSwapDealMtMCrossCurrencySwapDeal` - removing BOTH from the menu."""
    unknown = {name: [m for m in members if m not in INSTRUMENT['types']]
               for name, (_, members) in INSTRUMENT['groups'].items()}
    unknown = {k: v for k, v in unknown.items() if v}
    assert not unknown, f'context-menu entries naming no declared type: {unknown}'


def test_every_section_field_has_a_descriptor():
    """A section listing a name with no descriptor is skipped by the doc generator and raises a
    KeyError in the UI's `load_fields`."""
    undescribed = sorted({f for fl in INSTRUMENT['sections'].values() for f in fl
                          if f not in INSTRUMENT['fields']})
    assert not undescribed, f'sections list fields that have no descriptor: {undescribed}'


@pytest.mark.xfail(strict=True, reason='15 unreachable descriptors awaiting a ruling: most name '
                                       'real unbuilt features (double/memory barrier, pivot TARF, '
                                       'dual strike), so wiring beats deleting.')
def test_no_dead_descriptors():
    """A descriptor no section reaches is unreachable metadata - it cannot be rendered, documented
    or authored. Left in place it reads as coverage that does not exist."""
    used = {f for fl in INSTRUMENT['sections'].values() for f in fl}
    nested = {s for d in INSTRUMENT['fields'].values() for s in d.get('sub_fields', [])}
    dead = sorted(set(INSTRUMENT['fields']) - used - nested)
    assert not dead, f'descriptors no section references: {dead}'


def test_factor_fields_are_declared():
    """`factor_fields` is how discovery finds a deal's price factors, so every key in it is a field
    the deal genuinely reads. One absent from the schema is the `Barrier_Price` defect: the pricer
    reads it with a hard key while no schema-authored deal can emit it."""
    undeclared = [(n, k) for n, c in sorted(deal_classes().items())
                  if n in INSTRUMENT['types']
                  for k in c.__dict__.get('factor_fields', {})
                  if (k if isinstance(k, str) else k[0]) not in declared_fields(n)]
    assert not undeclared, f'factor references no schema-authored deal can carry: {undeclared}'


@pytest.mark.parametrize('mapping_key', sorted(k for k, v in MAPPING.items() if 'fields' in v))
def test_aliasing_is_declared_not_inferred(mapping_key):
    """The UI used to recover the JSON key by reconstructing it from the DESCRIPTION
    (`description.replace(' ', '_')`, six sites), which made that one string load-bearing as a key
    and free text at the same time: prose written into it pointed the widget at a key nobody writes,
    so the field silently showed its default and edits landed somewhere else. `name` now carries the
    key explicitly and `description` is free text.

    Aliasing is legitimate but must be DECLARED, so this holds `ALIASED_KEYS` to exactly the set
    that needs it - an alias added later without a listing fails here rather than going quiet."""
    aliased = ALIASED_KEYS.get(mapping_key, {})
    for key, meta in MAPPING[mapping_key]['fields'].items():
        assert meta.get('name', key) == aliased.get(key, key), (
            f'{mapping_key}.{key} reads {meta.get("name", key)!r}; ALIASED_KEYS expects '
            f'{aliased.get(key, key)!r}')
