"""The market data splits in two: what a PLAN pins and what a values patch carries.

`bind='value'` on a declaration says the engine reads that field's CONTENT and nothing about
discovery, tenor grids, process wiring, correlation or the code paths depends on it. The split is
not per FIELD, though - a curve's knots size `all_tenors` when the factor is constructed while its
rate column is content, so a shape-valued field splits inside itself and only the last column
travels.

The gates below are the three things that can go wrong. The partition can lose information, which
`test_reconstruction` catches per declared type over shapes the framework never sees in a fixture.
The patch verb can accept something it should refuse, which the fail-loud gates catch. And - the
one no structural gate can see - the patch can be correct on paper and never reach a number, which
is what the reprice gates are for: each bumps through `patch_market` and again by hand, and holds
the two to the same MTM.

Three fixtures, because a field is only bindable from its consumption site and the three sites are
different: a cashflow for a curve rate and a spot, a quanto equity option for `Correlation.Value`,
and a CDS for `SurvivalProb.Recovery_Rate`. The last two used to be read at COMPILE, which is why
they were declined; the reprice gates are what says they are read at eval now, and their base MTMs
are pinned to the commit that still baked them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import schema, utils
from derivus.config import Config, CustomJsonEncoder
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
RATE = 0.02
SPOT = 18.5
AMOUNT = 1_000_000.0
VOL = 0.25
EQ_SPOT = 100.0
RHO = 0.3
RHO_BUMPED = -0.5
RECOVERY = 0.4
RECOVERY_BUMPED = 0.2
CORRELATION = 'Correlation.EquityPrice.EQ/FxRate.USD.ZAR'

# What the parent commit (92dad5c) produced, when `get_implied_correlation` and `get_recovery_rate`
# still baked their numbers into `field_index` at `calc_dependencies`. Reading them at eval instead
# is a change of WHEN, so all four have to survive it - the bumped pair authored the only way that
# commit could author it, by editing the block before the run.
QUANTO_MTM = 11.548002479638967
QUANTO_MTM_BUMPED = 9.827092105505875
CDS_MTM = 54332.429663776406
CDS_MTM_BUMPED = 81391.73851161855

# Content for each shape, authored HERE: the coordinate arity is what the partition splits on, and
# a fixture only ever carries the shapes its own market data happens to use.
SHAPE_SAMPLE = {
    'Curve': [[0.0, 0.011], [1.0, 0.022], [5.0, 0.033]],
    'Surface': [[-0.1, 1.0, 0.21], [0.0, 1.0, 0.22], [0.1, 2.0, 0.23]],
    'Space': [[-0.1, 1.0, 2.0, 0.21], [0.0, 1.0, 2.0, 0.22], [0.1, 2.0, 5.0, 0.23]],
}

# Semantic types whose whole content is one value, so shadowing them to None loses nothing. A
# Table or a Container carries structure of its own and would be silently dropped.
SCALAR_TYPES = ('Text', 'Float', 'Integer', 'Date', 'Percent', 'Basis', 'Period')


def as_json(obj):
    """The comparison the round trip means by identical - `.Curve` serialises meta plus data."""
    return json.dumps(obj, sort_keys=True, cls=CustomJsonEncoder)


def sample_block(factor_type):
    """A block carrying every field the type declares, so the partition sees all of them."""
    return {key: utils.Curve([], SHAPE_SAMPLE[f.type]) if f.type in schema.SHAPED else f.default
            for key, f in schema.FACTOR_FIELDS[factor_type].items()}


def _two_currency_cfg():
    """USD reporting over a ZAR leg - the smallest world every fixture here is built on."""
    cfg = Config()
    cfg.params['System Parameters']['Base_Currency'] = 'USD'
    cfg.params['System Parameters']['Base_Date'] = BASE
    cfg.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR', 'Priority': 1,
                       'Spot': SPOT},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
        'InterestRate.ZAR': {'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
    }
    cfg.params['Price Models'] = {}
    cfg.params['Valuation Configuration'] = {}
    return cfg


def _in_context(cfg, deal, reference):
    cfg.deals = {'Attributes': {'Reference': reference, 'Tag_Titles': ''},
                 'Deals': {'Children': [{'Instrument': construct_instrument(deal, {})}]},
                 'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    context = derivus.Context()
    context.current_cfg = cfg
    return context


def _context():
    """A ZAR cashflow reported in USD: its MTM is amount x discount factor x spot, so one deal is
    sensitive to a Curve's rate column and a 0D Spot at once."""
    return _in_context(_two_currency_cfg(), {
        'Object': 'FixedCashflowDeal', 'Reference': 'CF1', 'Currency': 'ZAR',
        'Discount_Rate': 'ZAR', 'Calendars': None, 'Amount': AMOUNT,
        'Payment_Date': BASE + pd.DateOffset(years=2)}, 'patch')


def _quanto_context():
    """A ZAR equity option settling in USD, which is what makes `Correlation.Value` reachable.

    The correlation is named on the SORTED currency pair, so a ZAR deal paying USD reads it with
    the reverse-pair sign - which is why raising rho LOWERS this option. A fixture the other way
    round would price identically whether or not the sign survived.
    """
    cfg = _two_currency_cfg()
    cfg.params['Price Factors'].update({
        'EquityPrice.EQ': {'Spot': EQ_SPOT, 'Currency': 'ZAR', 'Interest_Rate': 'ZAR',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'ZAR', 'Floor': None,
                            'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, VOL] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
        'VolatilityGrid.USD.ZAR': {
            'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[m, t, 0.15] for m in (0.8, 1.0, 1.2)
                                        for t in (0.02, 2.0)])},
        CORRELATION: {'Value': RHO},
    })
    return _in_context(cfg, {
        'Object': 'EquityOptionDeal', 'Reference': 'QUANTO1', 'Currency': 'ZAR',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'ZAR',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': EQ_SPOT, 'Units': 1.0, 'Payoff_Type': 'Quanto',
        'Expiry_Date': BASE + pd.DateOffset(years=1)}, 'quanto')


def _cds_context():
    """A three-year CDS on one name: the only deal whose price reads `SurvivalProb.Recovery_Rate`.
    Protection is the (1 - R) leg, so the recovery moves the MTM without touching the premium."""
    cfg = _two_currency_cfg()
    cfg.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': RECOVERY, 'Minimum_Recovery_Rate': None, 'Issuer': '',
        'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.25]])}
    return _in_context(cfg, {
        'Object': 'DealDefaultSwap', 'Reference': 'CDS1', 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Name': 'CPTY', 'Buy_Sell': 'Buy', 'Principal': AMOUNT,
        'Pay_Frequency': pd.DateOffset(months=3), 'Pay_Rate': utils.Percent(1.0),
        'Accrual_Day_Count': 'ACT_365', 'Effective_Date': BASE,
        'Maturity_Date': BASE + pd.DateOffset(years=3), 'Amortisation': None,
        'Penultimate_Coupon_Date': None, 'First_Coupon_Date': None, 'Upfront': 0.0,
        'Upfront_Date': None, 'Survival_Probability': 'CPTY', 'Calendars': None,
        'ISDA_Standard': 'ISDA_03', 'Accrue_Fee': 'No',
        'Protection_Paid_At_Maturity': 'No'}, 'cds')


def price(context, reference):
    _, out = derivus.run_baseval(context.current_cfg, prec=DTYPE,
                                 overrides={'MCMC_Simulations': 1, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == reference]['Value'].iloc[0])


def test_the_declarations_bind_something():
    """Guards the gates themselves: every assertion here is vacuous over an empty value set."""
    bound = {(t, k) for t, d in schema.FACTOR_FIELDS.items() for k, f in d.items()
             if f.bind == 'value'}
    assert bound, 'no field declares bind=value - these gates are vacuous'


@pytest.mark.parametrize('factor_type', sorted(schema.FACTOR_FIELDS))
def test_reconstruction(factor_type):
    """`structural` union `values` is the block, exactly, for every declared type.

    Also that the shadow is a SHADOW: a scalar reduced to None, a shape to its coordinate columns
    alone. A partition that returned the field intact on both sides would reconstruct perfectly and
    pin nothing."""
    block = sample_block(factor_type)
    structural, values = schema.partition_factor(factor_type, block)
    assert as_json(schema.apply_values(factor_type, structural, values)) == as_json(block)

    for key, field in schema.FACTOR_FIELDS[factor_type].items():
        if field.bind != 'value':
            assert key not in values, f'{factor_type}.{key} is structural but travels as a value'
        elif field.type in schema.SHAPED:
            assert structural[key].array.shape[1] == block[key].array.shape[1] - 1, (
                f'{factor_type}.{key}: the structural half still carries the value column')
            assert len(values[key]) == len(block[key].array)
        else:
            assert structural[key] is None, (
                f'{factor_type}.{key}: the structural half still carries the value')


def test_every_value_field_has_a_shadow_rule():
    """No silent fall-through. A shape splits on its last column and a scalar shadows to None;
    anything else - a Table, a Container - would be shadowed to None as well, dropping structure
    the partition never looked at."""
    unruled = sorted(f'{t}.{k} ({f.type})' for t, d in schema.FACTOR_FIELDS.items()
                     for k, f in d.items()
                     if f.bind == 'value' and f.type not in schema.SHAPED + SCALAR_TYPES)
    assert not unruled, f'value-bound fields with no shadow rule for their type: {unruled}'


def test_the_descriptor_publishes_the_binding():
    """A front end has to know which fields it may offer for a patch, and the store is all it
    reads. Structural is the default, so only a value-bound field carries the key."""
    types = derivus.schema.mapping['Factor']['types']
    assert types['FxRate']['Spot']['bind'] == 'value'
    assert 'bind' not in types['FxRate']['Interest_Rate']


def test_round_trip_leaves_the_market_data_identical():
    context = _context()
    before = as_json(context.current_cfg.params['Price Factors'])
    context.patch_market(context.market_patch())
    assert as_json(context.current_cfg.params['Price Factors']) == before


def test_market_patch_carries_the_values_and_only_the_values():
    patch = _context().market_patch()
    assert patch == {'FxRate.USD': {'Spot': 1.0}, 'FxRate.ZAR': {'Spot': SPOT},
                     'InterestRate.USD': {'Curve': [RATE, RATE]},
                     'InterestRate.ZAR': {'Curve': [RATE, RATE]}}


def test_a_structural_field_in_a_patch_raises_naming_it():
    context = _context()
    with pytest.raises(ValueError, match=r'InterestRate\.ZAR: Day_Count is structural'):
        context.patch_market({'InterestRate.ZAR': {'Day_Count': 'ACT_360'}})
    # a field the block does not carry is structural for the same reason: the key SET is the plan
    with pytest.raises(ValueError, match=r'FxRate\.ZAR: Jump_Level is structural'):
        context.patch_market({'FxRate.ZAR': {'Jump_Level': 0.1}})


def test_an_unknown_factor_in_a_patch_raises_naming_it():
    with pytest.raises(KeyError, match=r'InterestRate\.GBP is not a price factor'):
        _context().patch_market({'InterestRate.GBP': {'Curve': [0.05]}})


def test_a_patched_curve_and_spot_reprice():
    """The gate the others cannot stand in for: the patch reaches the NUMBER.

    A structurally perfect patch that the engine never reads reconstructs, round-trips and prices
    exactly as before. So bump twice - once through `patch_market`, once by editing the blocks the
    way a market-data file would - and hold the two to the same MTM.
    """
    base = price(_context(), 'CF1')

    patched = _context()
    patch = patched.market_patch()
    patch['InterestRate.ZAR']['Curve'] = [RATE + 0.01, RATE + 0.01]
    patch['FxRate.ZAR']['Spot'] = SPOT * 1.1
    patched.patch_market(patch)
    by_patch = price(patched, 'CF1')

    authored = _context()
    factors = authored.current_cfg.params['Price Factors']
    factors['InterestRate.ZAR']['Curve'] = utils.Curve(
        [], [[0.0, RATE + 0.01], [5.0, RATE + 0.01]])
    factors['FxRate.ZAR']['Spot'] = SPOT * 1.1
    by_hand = price(authored, 'CF1')

    assert by_patch != pytest.approx(base, rel=1e-9), 'the patch never reached the number'
    assert by_patch == pytest.approx(by_hand, rel=1e-12)
    # 10% up on spot, 100bp up on a 2y discount: up on net, and each leg the right sign
    assert by_patch == pytest.approx(
        base * 1.1 * np.exp(-0.01 * ((BASE + pd.DateOffset(years=2)) - BASE).days / 365.0),
        rel=1e-6)


def test_the_two_eval_time_reads_carry_the_number_they_used_to_bake():
    """Reading a field at eval rather than at compile changes WHEN, so nothing may move.

    Both fixtures are pinned twice: unbumped, and bumped the only way the parent commit could bump
    them - by editing the block before the run, which that commit read at `calc_dependencies`.
    Pinning only the unbumped pair would be satisfied by a factor reference nobody ever reads.
    """
    assert price(_quanto_context(), 'QUANTO1') == pytest.approx(QUANTO_MTM, rel=1e-12)
    assert price(_cds_context(), 'CDS1') == pytest.approx(CDS_MTM, rel=1e-12)

    quanto = _quanto_context()
    quanto.current_cfg.params['Price Factors'][CORRELATION]['Value'] = RHO_BUMPED
    assert price(quanto, 'QUANTO1') == pytest.approx(QUANTO_MTM_BUMPED, rel=1e-12)

    cds = _cds_context()
    cds.current_cfg.params['Price Factors']['SurvivalProb.CPTY'][
        'Recovery_Rate'] = RECOVERY_BUMPED
    assert price(cds, 'CDS1') == pytest.approx(CDS_MTM_BUMPED, rel=1e-12)


def test_a_patched_implied_correlation_reprices():
    """What binding `Correlation.Value` bought: the quanto adjustment follows a patch.

    The value half now carries it, so the same bump through `patch_market` has to land on the same
    MTM as authoring it - and on the DOWN side, because this deal reads the pair in reverse.
    """
    base = price(_quanto_context(), 'QUANTO1')

    patched = _quanto_context()
    patch = patched.market_patch()
    assert patch[CORRELATION] == {'Value': RHO}
    patch[CORRELATION]['Value'] = RHO_BUMPED
    patched.patch_market(patch)
    by_patch = price(patched, 'QUANTO1')

    assert by_patch != pytest.approx(base, rel=1e-9), 'the patch never reached the number'
    assert by_patch == pytest.approx(QUANTO_MTM_BUMPED, rel=1e-12)
    assert by_patch < base, 'raising rho raised the option - the reverse-pair sign was dropped'


def test_a_patched_recovery_rate_reprices():
    """What binding `SurvivalProb.Recovery_Rate` bought: the protection leg follows a patch.

    Protection pays (1 - R) and the premium leg does not see R at all, so the MTM is affine in R -
    which a third recovery pins without any reference outside this fixture. A patch that reached
    only part of the pricer would land on the right direction and the wrong line.
    """
    base = price(_cds_context(), 'CDS1')

    patched = _cds_context()
    patch = patched.market_patch()
    assert patch['SurvivalProb.CPTY']['Recovery_Rate'] == RECOVERY
    patch['SurvivalProb.CPTY']['Recovery_Rate'] = RECOVERY_BUMPED
    patched.patch_market(patch)
    by_patch = price(patched, 'CDS1')

    zero_recovery = _cds_context()
    patch_zero = zero_recovery.market_patch()
    patch_zero['SurvivalProb.CPTY']['Recovery_Rate'] = 0.0
    zero_recovery.patch_market(patch_zero)
    by_patch_zero = price(zero_recovery, 'CDS1')

    assert by_patch != pytest.approx(base, rel=1e-9), 'the patch never reached the number'
    assert by_patch == pytest.approx(CDS_MTM_BUMPED, rel=1e-12)
    assert by_patch_zero - base == pytest.approx(
        (by_patch - base) * RECOVERY / (RECOVERY - RECOVERY_BUMPED), rel=1e-9)


def test_a_patch_is_a_delta():
    """A field the patch names is replaced; a value field it omits keeps its current content.

    Without the merge, `apply_values` put back only the named fields and a factor's OTHER value
    fields stayed coordinate shells - `SurvivalProb` is the two-value-field case, so patching only
    its Recovery_Rate used to leave `Curve` as a one-column shell that failed on the next read."""
    cx = _context()
    cx.current_cfg.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Currency': 'USD',
        'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    original = cx.current_cfg.params['Price Factors']['SurvivalProb.CPTY']['Curve'].array.copy()
    cx.patch_market({'SurvivalProb.CPTY': {'Recovery_Rate': 0.55}})
    block = cx.current_cfg.params['Price Factors']['SurvivalProb.CPTY']
    assert block['Recovery_Rate'] == 0.55
    assert np.array_equal(block['Curve'].array, original), 'the unnamed value field was dropped'


def test_an_empty_per_factor_patch_is_a_no_op():
    """The degenerate delta: naming a factor with no fields changes nothing."""
    cx = _context()
    before = cx.current_cfg.params['Price Factors']['InterestRate.USD']['Curve'].array.copy()
    cx.patch_market({'InterestRate.USD': {}})
    after = cx.current_cfg.params['Price Factors']['InterestRate.USD']['Curve'].array
    assert np.array_equal(after, before)


def test_apply_values_warns_when_a_value_field_is_missing(caplog):
    """The low-level guard: `apply_values` called directly with an incomplete values dict leaves a
    coordinate shell, which is reconstructible only on purpose - so it says so, naming the field.
    `patch_market` merges before calling, so the public verb never triggers this."""
    import logging as _logging
    block = {'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.01], [10.0, 0.02]]),
             'Currency': 'USD'}
    structural, values = schema.partition_factor('SurvivalProb', block)
    del values['Curve']
    with caplog.at_level(_logging.WARNING):
        schema.apply_values('SurvivalProb', structural, values)
    assert any('SurvivalProb.Curve' in r.getMessage() and 'coordinate shell' in r.getMessage()
               for r in caplog.records), caplog.records
