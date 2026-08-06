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
import ast
import inspect
import pathlib
import re

import pytest

from derivus import (bootstrappers, calculation, fields, instruments, riskfactors, stochasticprocess,
                      utils)

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


def test_every_declared_deal_type_is_dispatchable():
    """`construct_instrument` does `globals().get(param['Object'])`, and on a miss LOGS AN ERROR and
    returns `{}` - the deal vanishes from the portfolio rather than raising. So a schema type naming
    no class is a deal the UI offers, the docs document, and the engine silently drops.

    `SwapBasisDeal` and `SwapCurrencyDeal` sat in exactly that state, offered under two menus with
    128 descriptors between them and no class in this repo or the one it came from. A type is now a
    class that declares `fields`, so the state is unreachable rather than merely absent."""
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


def test_no_dead_descriptors():
    """A descriptor no section reaches is unreachable metadata - it cannot be rendered, documented
    or authored. Left in place it reads as coverage that does not exist.

    Fifteen sat here (double/memory barrier, pivot TARF, dual strike - real unbuilt features). They
    were reachable only because a flat dict admits an entry no section names; a descriptor now
    exists only by being declared on a class, so building one of those features means declaring its
    fields on the deal that reads them."""
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


# MarketPrices types that name no bootstrapper branch on purpose.
UNMATCHED_MARKET_PRICES = {
    'InterestRatePrices': 'the FRA/Swap/Deposit quote family, declared but not yet bootstrapped',
    'quote': 'the shape of one quote inside a price block, not a market-factor type',
}


def bootstrapped_market_factor_types():
    """The `Market Prices` type strings the engine actually matches, read off the source.

    Parsed rather than listed, because a hand-kept list here would be a fourth store of the same
    knowledge and would drift the same way the first three did."""
    found = set()
    for node in ast.walk(ast.parse(inspect.getsource(bootstrappers))):
        # `if market_factor.type == 'X':`
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute) \
                and node.left.attr == 'type':
            found.update(c.value for c in node.comparators
                         if isinstance(c, ast.Constant) and isinstance(c.value, str))
        # `self.market_factor_type = 'X'`
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and any(isinstance(t, ast.Attribute) and t.attr == 'market_factor_type'
                        for t in node.targets):
            found.add(node.value.value)
    return found


def test_declared_market_prices_are_bootstrapped():
    """A bootstrapper selects work by `market_factor.type == '<literal>'` with no else, so a schema
    name that matches no literal is a quote block the engine walks straight past - the config looks
    authored, the calibration never runs, and only a downstream `wrote no *.* price factor` error
    hints at it. This drift had reached the published docs: `GBMTSModelPrices` and
    `HullWhite2FactorInterestRateModelPrices` were documented while the engine matched
    `GBMAssetPriceTSModelPrices` and `HullWhite2FactorModelPrices`."""
    matched = bootstrapped_market_factor_types()
    declared = set(MAPPING['MarketPrices']['types']) - set(UNMATCHED_MARKET_PRICES)
    assert not declared - matched, (
        f'declared MarketPrices types no bootstrapper matches: {sorted(declared - matched)}')


def test_bootstrapped_market_prices_are_declared():
    """The converse: a type the engine bootstraps but no schema declares cannot be authored from the
    UI or found in the docs. `CSForwardPriceModelPrices` sat in that state."""
    undeclared = bootstrapped_market_factor_types() - set(MAPPING['MarketPrices']['types'])
    assert not undeclared, f'bootstrapped types absent from the schema: {sorted(undeclared)}'


RETIRED_VOL_TYPES = ('FXVol', 'EquityPriceVol', 'CommodityPriceVol')


def test_the_retired_vol_types_stay_retired():
    """Three empty `Factor2D` subclasses whose bodies differed only in a docstring, and three schema
    declarations that had drifted apart - FX and commodity could not author the SVI/Skew surfaces
    `Factor2D.get_subtype` has always supported, and only commodity declared Currency.

    What varies is the SUBTYPE, not the asset class: asset class belongs to the UNDERLYING, whose
    types (FxRate / EquityPrice / CommodityPrice) stay distinct and are what the bootstrapper reads
    it from. One `VolatilityGrid` replaces all three, in the classes, the schema, the discovery
    registries and the deals' `factor_fields`."""
    assert hasattr(riskfactors, 'VolatilityGrid')
    back = [n for n in RETIRED_VOL_TYPES if hasattr(riskfactors, n)]
    assert not back, f'retired vol classes are back in riskfactors: {back}'

    declared = {n for n in RETIRED_VOL_TYPES
                if n in MAPPING['Factor']['types'] or n in MAPPING['Process_factor_map']}
    assert not declared, f'retired vol types are back in the schema: {sorted(declared)}'

    referenced = sorted({f'{n}.{f}' for n, c in deal_classes().items()
                         for f, types in getattr(c, 'factor_fields', {}).items()
                         if set(types) & set(RETIRED_VOL_TYPES)})
    assert not referenced, f'deals still reference a retired vol type: {referenced}'
    assert utils.TwoDimensionalFactors == ['VolatilityGrid'], (
        f'the 2D factor registry disagrees: {utils.TwoDimensionalFactors}')


def test_discountrate_stays_retired():
    """`DiscountRate` was a factor type whose entire body was one field pointing at another factor,
    and `get_discount_factor` existed only to follow it: look up the wrapper, ask it for the name,
    resolve THAT as an InterestRate. Every block in the repo mapped a name to itself.

    The tell that it was redundant is that `add_rates_for_factor` self-healed this type and no
    other - it could synthesise the block precisely because the block held nothing the engine could
    not already derive. A deal's `Discount_Rate` field now names an InterestRate directly, which
    loses no expressiveness: discounting on a different curve was always done by naming one."""
    assert not hasattr(riskfactors, 'DiscountRate'), 'the wrapper class is back'
    assert not hasattr(instruments, 'get_discount_factor'), 'the name hop is back'
    assert 'DiscountRate' not in MAPPING['Factor']['types'], 'back in the Factor store'
    assert 'DiscountRate' not in MAPPING['Process_factor_map'], 'back in the process map'
    assert 'DiscountRate' not in utils.DimensionLessFactors, 'back in the dimensionless registry'

    referenced = sorted({f'{n}.{f}' for n, c in deal_classes().items()
                         for f, types in getattr(c, 'factor_fields', {}).items()
                         if 'DiscountRate' in types})
    assert not referenced, f'deals still reference the retired type: {referenced}'


def test_docs_publish_only_real_market_price_names():
    """The drift reached the DOCS, which is where it did its damage: a reader copying the example
    authored a block the engine walks past. Neither the schema nor the suite looked at prose, so the
    two pages disagreed with the engine, and with each other, for as long as they existed."""
    docs = pathlib.Path(__file__).parent.parent / 'docs_src'
    known = bootstrapped_market_factor_types() | set(MAPPING['MarketPrices']['types'])
    published = {(p.relative_to(docs), n) for p in docs.rglob('*.md')
                 for n in re.findall(r'\b\w*ModelPrices\b', p.read_text())}
    unknown = sorted(f'{p}: {n}' for p, n in published if n not in known)
    assert not unknown, f'docs publish market-price names the engine never matches: {unknown}'
