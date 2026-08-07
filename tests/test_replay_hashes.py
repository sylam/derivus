"""The replay identity: `(plan_hash, values_hash, engine_version, seed)`.

A reported number is reproducible from four coordinates, and the first two are content hashes of the
two halves the market data already splits into. `plan_hash` is the PROGRAM - `params` and `deals`
with every `bind='value'` field and `Random_Seed` taken out - and `values_hash` is exactly what
`cx.market_patch()` emits. Both are pure: they run nothing and change nothing.

Three things can go wrong, and there is a gate for each. The hashes can be unstable, which makes
every replay claim false - two independent loads of one job must agree. The split can leak, which is
the whole point of having two hashes rather than one - so every value-bound patch must move the
values hash ALONE, and every structural edit the plan hash alone. And the seed, being its own
coordinate, must move neither.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import derivus
from derivus import utils
import test_market_patch as patch

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'tests', 'fixtures', 'policy_test_simulate_only.json')


def _context():
    """The quanto fixture plus what a plan hash has to see and a values patch must not touch: a
    survival curve, a simulation correlation matrix, and a Calculation block carrying the batch
    shape and the seed."""
    context = patch._quanto_context()
    cfg = context.current_cfg
    cfg.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': patch.RECOVERY, 'Minimum_Recovery_Rate': None, 'Issuer': '',
        'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.25]])}
    cfg.params['Correlations'] = {('FxRate.ZAR', 'EquityPrice.EQ'): 0.25}
    cfg.deals['Calculation'].update(
        {'Batch_Size': 1024, 'Simulation_Batches': 1, 'Random_Seed': 42})
    return context


def _day_count(cfg):
    cfg.params['Price Factors']['InterestRate.ZAR']['Day_Count'] = 'ACT_360'


def _curve_knot(cfg):
    """The knot MOVES and the rate column does not: a curve splits inside itself, so this is the
    edit that separates the coordinate half from the value half."""
    cfg.params['Price Factors']['InterestRate.ZAR']['Curve'] = utils.Curve(
        [], [[0.0, patch.RATE], [7.0, patch.RATE]])


def _deal_field(cfg):
    cfg.deals['Deals']['Children'][0]['Instrument'].field['Strike_Price'] = 110.0


def _batch_size(cfg):
    cfg.deals['Calculation']['Batch_Size'] = 2048


def _correlation_matrix(cfg):
    cfg.params['Correlations'][('FxRate.ZAR', 'EquityPrice.EQ')] = 0.5


def test_two_loads_of_one_job_hash_the_same():
    """Determinism is the whole claim. Two Contexts, so the market data is parsed twice rather than
    served from one context's cache - a cache hit would agree with itself for free."""
    first, second = derivus.Context().load_json(FIXTURE), derivus.Context().load_json(FIXTURE)
    assert first.plan_hash() == second.plan_hash()
    assert first.values_hash() == second.values_hash()
    assert first.plan_hash() != first.values_hash()


def test_two_identically_built_contexts_hash_the_same():
    """The same claim without a file: nothing in either hash reads an address or an insertion
    accident."""
    assert _context().plan_hash() == _context().plan_hash()
    assert _context().values_hash() == _context().values_hash()


def test_the_replay_tuple_is_reachable_from_the_public_surface():
    """Four coordinates, no internal import: two verbs, the shipped version, and the seed off the
    Calculation block."""
    context = _context()
    plan, values = context.plan_hash(), context.values_hash()
    assert len(plan) == 64 and len(values) == 64
    assert derivus.__version__ == '.'.join(map(str, derivus.version_info))
    assert context.current_cfg.deals['Calculation']['Random_Seed'] == 42


@pytest.mark.parametrize('factor,field,value', [
    ('FxRate.ZAR', 'Spot', 19.0),
    ('InterestRate.ZAR', 'Curve', [0.05, 0.05]),
    (patch.CORRELATION, 'Value', patch.RHO_BUMPED),
    ('SurvivalProb.CPTY', 'Recovery_Rate', patch.RECOVERY_BUMPED),
])
def test_a_value_patch_moves_the_values_hash_alone(factor, field, value):
    """The separation, from the values side. The last two rows are what the eval-time reads bought:
    an implied correlation and a recovery rate were plan content until they were read at eval."""
    base = _context()
    patched = _context()
    values = patched.market_patch()
    values[factor][field] = value
    patched.patch_market(values)

    assert patched.plan_hash() == base.plan_hash()
    assert patched.values_hash() != base.values_hash()


@pytest.mark.parametrize('mutate', [
    _day_count, _curve_knot, _deal_field, _batch_size, _correlation_matrix])
def test_a_structural_change_moves_the_plan_hash_alone(mutate):
    """The separation, from the plan side - a factor's day count, a curve's knots, a deal field, the
    batch shape and the simulation correlation matrix are all program, not market values."""
    base = _context()
    changed = _context()
    mutate(changed.current_cfg)

    assert changed.plan_hash() != base.plan_hash()
    assert changed.values_hash() == base.values_hash()


def test_the_seed_belongs_to_neither_hash():
    """It is the fourth coordinate. Folding it into the plan would make every reseeded run a
    different program and every plan cache useless."""
    base = _context()
    reseeded = _context()
    reseeded.current_cfg.deals['Calculation']['Random_Seed'] = 987

    assert reseeded.plan_hash() == base.plan_hash()
    assert reseeded.values_hash() == base.values_hash()
