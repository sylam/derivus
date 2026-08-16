"""`cx.validate()` - can this job run, and if not, what is missing.

Two things were knowable before a run and reachable only by running. A deal that breaks an
authoring rule: `schema.validate_instrument` has said so per deal since the rules were declared,
but nothing walked the book and asked. And a price factor with no market-data block, which is
worse - discovery logs two lines and skips it, the factor is never built, and the deal quietly
leaves the portfolio at pricing time. The verb returns both as data.

The want-list has to be assembled from two places, and a fixture naming only one kind of factor
would hide half of it. A type carrying dependants (`FxRate`, `EquityPrice`, `CommodityPrice`)
raises `KeyError` on its own missing block and discovery SKIPS it, so it is absent from
`dependent_factors` and only `discover_factors` can report it; a type with no dependants
(`InterestRate`) is discovered happily and blows up much later, in `construct_factor`, so it is
the set difference against `Price Factors`. Both kinds are tested, and each fails if the other
half is dropped.

Every gate pairs a violating job with a conforming one: a `validate` that complained about
everything would satisfy the first assertion of each test and none of the second.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.config import Config
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64

#: A conforming deal, and the same option written as a binary with no `Cash_Payoff` - which IS its
#: notional, so the declaration makes it REQUIRED.
OPTION = {'Object': 'EquityOptionDeal', 'Reference': 'OPT1', 'Currency': 'USD',
          'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
          'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Units': 1.0,
          'Strike_Price': 100.0, 'Expiry_Date': BASE + pd.Timedelta(days=365)}
BINARY = {'Object': 'EquityBinaryOption', 'Reference': 'BIN1', 'Currency': 'USD',
          'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
          'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
          'Strike_Price': 100.0, 'Expiry_Date': BASE + pd.Timedelta(days=365),
          'Settlement_Date': BASE + pd.Timedelta(days=365)}
STRUCTURE = {'Object': 'StructuredDeal', 'Reference': 'STR1', 'Currency': 'USD',
             'Net_Cashflows': 'Yes'}
NO_CLASS = {'Object': 'SwapBasisDeal', 'Reference': 'GONE'}


def _node(deal, children=None, **node_fields):
    node = dict({'Instrument': construct_instrument(deal, {})}, **node_fields)
    if children is not None:
        node['Children'] = children
    return node


def _cfg(*nodes):
    """An equity world whose every factor has a block, so an empty want-list is the clean answer.

    `EquityPrice` resolves to an IMPLIED process whose `GBMAssetPriceTSModelParameters` block is
    present while `Price Models` has no entry for it - the one configuration the old `find_models`
    wrote a dummy entry for, and therefore the fixture that catches any code that starts writing
    into params again (`test_validate_leaves_the_job_exactly_as_it_found_it`)."""
    c = Config()
    c.params['System Parameters'].update({'Base_Currency': 'USD', 'Base_Date': BASE})
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.02], [10.0, 0.02]])},
        'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, 0.2] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
        'GBMAssetPriceTSModelParameters.EQ': {
            'Vol': utils.Curve([], [[0.0, 0.2], [5.0, 0.2]]),
            'Quanto_FX_Volatility': None, 'Quanto_FX_Correlation': None},
    }
    c.params['Price Models'] = {}
    c.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceTSModelImplied')
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': list(nodes)},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def test_a_job_that_can_run_says_nothing():
    """The one that guards every other test here: they all assert the ABSENCE of a message or a
    want-list entry for a conforming job, and a validate that complained about everything - about
    every declared field, or about every discovered factor - would satisfy none of them."""
    assert _cfg(_node(OPTION)).validate() == {'deals': {}, 'factors': []}


def test_a_deal_that_cannot_price_is_named_by_its_reference():
    """The authoring layer, reached through the book rather than one deal at a time."""
    assert _cfg(_node(BINARY)).validate()['deals'] == {'BIN1': ['Cash_Payoff is required']}
    assert _cfg(_node(dict(BINARY, Cash_Payoff=1e6))).validate()['deals'] == {}


@pytest.mark.parametrize('field,factor', [
    ('Discount_Rate', utils.Factor('InterestRate', ('ZAR',))),
    ('Equity', utils.Factor('EquityPrice', ('NOSUCH',)))])
def test_a_factor_with_no_market_data_block_is_a_want_list_entry(field, factor):
    """The two ways a factor goes missing - see the module docstring. `InterestRate` carries no
    dependants and is discovered, `EquityPrice` does and is skipped, and they are collected in
    different places. The want-list spells both the way `check_tuple_name` does, which is the
    spelling of the `Price Factors` key an author has to add."""
    name = utils.check_tuple_name(factor)
    assert name in _cfg(_node(dict(OPTION, **{field: factor.name[0]}))).validate()['factors']
    assert name not in _cfg(_node(OPTION)).validate()['factors']


def test_a_deal_under_children_is_still_walked():
    """Any deal can carry `Children` and the engine recurses on their PRESENCE, never on the type,
    so validate has to as well. Both halves descend: the child's message and the child's factor."""
    result = _cfg(_node(STRUCTURE, children=[
        _node(dict(BINARY, Equity='NOSUCH'))])).validate()
    assert result['deals'] == {'BIN1': ['Cash_Payoff is required']}
    assert utils.check_tuple_name(utils.Factor('EquityPrice', ('NOSUCH',))) in result['factors']


def test_an_ignored_node_is_skipped_whole():
    """`walk_groups` semantics: `Ignore` takes the node and its subtree out of the job, so neither
    its messages nor its factors are anything this job needs."""
    ignored = _cfg(_node(STRUCTURE, Ignore='True', children=[
        _node(dict(BINARY, Equity='NOSUCH'))])).validate()
    assert ignored == {'deals': {}, 'factors': []}
    assert _cfg(_node(STRUCTURE, children=[
        _node(dict(BINARY, Equity='NOSUCH'))])).validate() != ignored


def test_a_node_that_never_became_a_deal_is_reported_and_the_walk_carries_on():
    """`construct_instrument` logs an unknown `Object` and returns `{}`, taking the payload with
    it, so the position is the only thing left to name the node by. It used to be an
    `AttributeError` from inside discovery instead - the one config most in need of validating was
    the one it could not survive."""
    result = _cfg(_node(NO_CLASS), _node(dict(OPTION, Discount_Rate='ZAR'))).validate()
    assert result['deals'] == {'#0': ['Object names no deal type']}
    assert result['factors'] == [utils.check_tuple_name(utils.Factor('InterestRate', ('ZAR',)))], (
        'the walk stopped at the node it could not build')


def test_references_that_do_not_identify_a_deal_fall_back_to_the_walk_position():
    """A `Reference` is what an author searches for, so it is the key - but nothing enforces that
    it is present or unique, and a dict silently keeps the last of anything keyed twice."""
    result = _cfg(_node(BINARY), _node(BINARY), _node(dict(BINARY, Reference=''))).validate()
    assert sorted(result['deals']) == ['#2', 'BIN1', 'BIN1#1']


def test_the_result_is_json():
    """VALIDATE answers over the same wire the job arrived on."""
    import json
    assert json.loads(json.dumps(_cfg(_node(BINARY), _node(NO_CLASS)).validate()))['deals'] == {
        'BIN1': ['Cash_Payoff is required'], '#1': ['Object names no deal type']}


def _mtm(c):
    _, out = derivus.run_baseval(c, prec=DTYPE, overrides={'MCMC_Simulations': 1, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'OPT1']['Value'].iloc[0])


def test_validate_leaves_the_job_exactly_as_it_found_it():
    """validate writes nothing — and since `find_models` stopped minting implied dummies,
    neither does a full run: the implied model prices with NO `Price Models` entry at all.
    The after-the-run arm is what holds that: this fixture's implied model is exactly the
    one the old code wrote a dummy for, so any code that starts writing into params again
    fails here, and the valuation itself proves the entry was never needed."""
    c = _cfg(_node(OPTION))
    factors = list(c.params['Price Factors'].items())

    c.validate()
    c.validate()
    assert c.params['Price Models'] == {}
    assert [(k, v) for k, v in c.params['Price Factors'].items()] == factors

    assert _mtm(c) == _mtm(_cfg(_node(OPTION))), 'the valuation moved'
    assert c.params['Price Models'] == {}, 'the run wrote into Price Models'
    assert [(k, v) for k, v in c.params['Price Factors'].items()] == factors
