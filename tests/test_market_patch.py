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
is what `test_a_patched_curve_and_spot_reprice` is for: it bumps through `patch_market` and again
by hand, and holds the two to the same MTM.
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


def _context():
    """A ZAR cashflow reported in USD: its MTM is amount x discount factor x spot, so one deal is
    sensitive to a Curve's rate column and a 0D Spot at once."""
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
    deal = {'Object': 'FixedCashflowDeal', 'Reference': 'CF1', 'Currency': 'ZAR',
            'Discount_Rate': 'ZAR', 'Calendars': None, 'Amount': AMOUNT,
            'Payment_Date': BASE + pd.DateOffset(years=2)}
    cfg.deals = {'Attributes': {'Reference': 'patch', 'Tag_Titles': ''},
                 'Deals': {'Children': [{'Instrument': construct_instrument(deal, {})}]},
                 'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    context = derivus.Context()
    context.current_cfg = cfg
    return context


def price(context):
    _, out = derivus.run_baseval(context.current_cfg, prec=DTYPE,
                                 overrides={'MCMC_Simulations': 1, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'CF1']['Value'].iloc[0])


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
    types = derivus.fields.mapping['Factor']['types']
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
    base = price(_context())

    patched = _context()
    patch = patched.market_patch()
    patch['InterestRate.ZAR']['Curve'] = [RATE + 0.01, RATE + 0.01]
    patch['FxRate.ZAR']['Spot'] = SPOT * 1.1
    patched.patch_market(patch)
    by_patch = price(patched)

    authored = _context()
    factors = authored.current_cfg.params['Price Factors']
    factors['InterestRate.ZAR']['Curve'] = utils.Curve(
        [], [[0.0, RATE + 0.01], [5.0, RATE + 0.01]])
    factors['FxRate.ZAR']['Spot'] = SPOT * 1.1
    by_hand = price(authored)

    assert by_patch != pytest.approx(base, rel=1e-9), 'the patch never reached the number'
    assert by_patch == pytest.approx(by_hand, rel=1e-12)
    # 10% up on spot, 100bp up on a 2y discount: up on net, and each leg the right sign
    assert by_patch == pytest.approx(
        base * 1.1 * np.exp(-0.01 * ((BASE + pd.DateOffset(years=2)) - BASE).days / 365.0),
        rel=1e-6)
