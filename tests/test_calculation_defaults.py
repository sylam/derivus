"""The F declaration is the single source of a calculation field's default.

RULED (2026-08-12): three fields ran on `.get` fallbacks that disagreed with their declarations -
`Random_Seed` 1 against a declared 5120, `Dynamic_Scenario_Dates` and `Generate_Cashflows` 'No'
against a declared 'Yes' - so a job authored from the schema and a job hand-written without the
key ran two different ways and nothing raised. The ruling is that the declaration wins, and the
mechanism is ONE seam rather than corrected fallbacks: `schema.declared_defaults` completes the
params dict at the top of every `execute`, reads index directly, and the fallback that could lie
is gone. `run_*` inject only runtime-derived keys (`Run_Date`, `Time_grid`).

`MCMC_Simulations` on base valuation is the fourth find, previously masked: the store said 2048,
the `.get` said 32768, and `run_baseval` injected 32768 unconditionally - so 32768 is what every
run has ever used, and the declaration now records it rather than changing it.

The AST gate is what keeps the rule: any surviving `params.get(key, fallback)` on a declared key
inside a calculation class must agree with the declaration, so a new disagreement cannot land
silently. Mutation-checked by editing a fallback and watching it fail.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import derivus
from derivus import calculation, run_baseval
from derivus.schema import REQUIRED, declared_defaults
import test_boundary_tarf_events as tarf

DTYPE = torch.float64
CALCS = (calculation.Credit_Monte_Carlo, calculation.Base_Revaluation, calculation.HedgeMonteCarlo)


# ---------------------------------------------------------------- the seam

def test_the_declaration_fills_every_omitted_field():
    """The ruled values, by name: the seam is what makes 5120 and the two 'Yes' flags land."""
    cmc = declared_defaults(calculation.Credit_Monte_Carlo, {})
    assert cmc['Random_Seed'] == 5120
    assert cmc['Dynamic_Scenario_Dates'] == 'Yes'
    assert cmc['Generate_Cashflows'] == 'Yes'
    bv = declared_defaults(calculation.Base_Revaluation, {})
    assert bv['Random_Seed'] == 5120
    assert bv['MCMC_Simulations'] == 4096 * 8, 'the declaration must record what run_baseval always injected'
    hmc = declared_defaults(calculation.HedgeMonteCarlo, {})
    assert hmc['Random_Seed'] == 5120 and hmc['Scenario_Factors'] == [] and hmc['Tenor_Offset'] == 0.0
    for cls in CALCS:
        filled = declared_defaults(cls, {})
        for f in cls.fields:
            if f.default is not REQUIRED and f.default is not None:
                assert f.key in filled, f'{cls.__name__}.{f.key} declared a default the seam dropped'


def test_the_author_wins_over_the_declaration():
    filled = declared_defaults(calculation.Credit_Monte_Carlo, {'Random_Seed': 7, 'Generate_Cashflows': 'No'})
    assert filled['Random_Seed'] == 7 and filled['Generate_Cashflows'] == 'No'


def test_a_mutable_default_is_a_copy_and_never_the_declaration_object():
    """A run edits its params freely - a container default handed out by reference would let the
    first run rewrite the schema for every run after it."""
    first = declared_defaults(calculation.Credit_Monte_Carlo, {})
    first['Credit_Valuation_Adjustment']['Calculate'] = 'CORRUPTED'
    second = declared_defaults(calculation.Credit_Monte_Carlo, {})
    assert second['Credit_Valuation_Adjustment']['Calculate'] == 'No'
    declared = next(f for f in calculation.Credit_Monte_Carlo.fields
                    if f.key == 'Credit_Valuation_Adjustment').default
    assert declared['Calculate'] == 'No', 'the class declaration itself was written through'


# ---------------------------------------------------------------- the rule, held by AST

def test_no_get_fallback_in_a_calculation_disagrees_with_its_declaration():
    """Any `params.get(key, fallback)` a calculation still carries on a DECLARED key must agree
    with the declaration - a fallback that disagrees is exactly the defect the ruling closed, and
    this is what stops one landing again. Undeclared keys (`DealLevel`, `Greeks` on HMC) and
    no-fallback `.get`s are outside the rule by decision."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'derivus', 'calculation.py')).read()
    tree = ast.parse(src)
    spans = {node.name: (node.lineno, node.end_lineno) for node in ast.walk(tree)
             if isinstance(node, ast.ClassDef)}
    violations = []
    for cls in CALCS:
        declared = {f.key: f.default for f in cls.fields}
        lo, hi = spans[cls.__name__]
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'get' and lo <= node.lineno <= hi):
                continue
            base = node.func.value
            on_params = (isinstance(base, ast.Name) and base.id == 'params') or (
                isinstance(base, ast.Attribute) and base.attr == 'params'
                and isinstance(base.value, ast.Name) and base.value.id == 'self')
            if not (on_params and node.args and isinstance(node.args[0], ast.Constant)):
                continue
            key = node.args[0].value
            if key not in declared or len(node.args) < 2:
                continue
            try:
                fallback = ast.literal_eval(node.args[1])
            except ValueError:
                violations.append((cls.__name__, node.lineno, key, ast.unparse(node.args[1])))
                continue
            if fallback != declared[key]:
                violations.append((cls.__name__, node.lineno, key, fallback))
    assert not violations, f'fallbacks disagreeing with their declaration: {violations}'


# ---------------------------------------------------------------- the engine obeys it

def _baseval_price(deal, **extra):
    overrides = dict({'MCMC_Simulations': 64, 'Greeks': 'No'}, **extra)
    _, out = run_baseval(tarf._cfg(deal, tarf.SPOT), prec=DTYPE, overrides=overrides)
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'TARF1']['Value'].iloc[0])


def test_an_omitted_random_seed_is_the_declared_5120_not_the_old_1():
    """Priced three ways on the TARF world, whose inner Monte Carlo consumes the seed: omitted
    equals an explicit 5120 exactly, and an explicit 1 - the fallback the engine used to take -
    prices differently, so the gate cannot pass on a fixture the seed never reaches."""
    omitted = _baseval_price(tarf.KNOCK_IN)
    explicit = _baseval_price(tarf.KNOCK_IN, Random_Seed=5120)
    old_fallback = _baseval_price(tarf.KNOCK_IN, Random_Seed=1)
    assert omitted == explicit
    assert omitted != old_fallback, 'the fixture cannot see the seed - the gate is a placebo'


def _cmc_run(**extra):
    overrides = dict({
        'Run_Date': tarf.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2m(2m)', 'Batch_Size': 64,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'MCMC_Simulations': 16,
        'Deflation_Interest_Rate': 'USD',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'No'}}, **extra)
    _, out = derivus.run_cmc(
        tarf._cfg(tarf.KNOCK_IN_CMC, tarf.SPOT, counterparty=True, simulate_fx=True),
        prec=DTYPE, overrides=overrides)
    return out


def test_omitted_generate_cashflows_now_generates_them():
    """Declared 'Yes', engine used to fall back to 'No': an omitted key must produce the cashflow
    tables an explicit 'Yes' produces, and an explicit 'No' must not - the third run is what makes
    a fixture with no cashflows at all unable to fake the gate."""
    omitted = _cmc_run()
    explicit_no = _cmc_run(Generate_Cashflows='No')
    assert 'cashflows' in omitted['Results'] and omitted['Results']['cashflows'], \
        'the declared Yes did not reach the engine'
    assert 'cashflows' not in explicit_no['Results']


def test_omitted_dynamic_scenario_dates_now_runs_dynamic():
    """Omitted must equal an explicit 'Yes' bit for bit, and differ from an explicit 'No'. The
    flag decides whether the SCENARIO grid carries the deal's own dates - simulated exactly there
    - or only the parsed base grid, with factor values interpolated to the deal dates. The mtm
    frame keeps its shape either way; the values move, deterministically under one seed."""
    omitted = _cmc_run()
    explicit_yes = _cmc_run(Dynamic_Scenario_Dates='Yes')
    explicit_no = _cmc_run(Dynamic_Scenario_Dates='No')
    assert np.array_equal(omitted['Results']['mtm'].values, explicit_yes['Results']['mtm'].values)
    assert not np.array_equal(omitted['Results']['mtm'].values, explicit_no['Results']['mtm'].values), \
        'the fixture cannot see the flag - the gate is a placebo'
