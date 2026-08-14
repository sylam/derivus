"""A per-path series a process PUBLISHES forks with the path, exactly as its factor does.

`_run_inner_mc_at_t` republishes each FACTOR grid as a `ScenarioSource` — the outer-realized past
at `Batch_Size`, then the forked rows at `Batch_Size x Inner_Sub_Batch` — so a pricer reading row
`t` inside a fork lands on the fork's own row. A `(key, kind)` series (`BasisLinkedSpotModel`'s
`basis_mu` / `basis_sig2`) is read through the SAME `calc_time_grid_spot_rate` seam and was NOT
republished: it stayed the fork's own two rows at `(B, B2)`, and a pricer asking it for outer row
`t` was handed a two-row tensor. The note at `instruments.get_observed_basis_decay` named it,
named the route in — a `Hedge_Monte_Carlo` liability that projects a composed spot, i.e. this
world — and left it unmeasured. It is measured here, and it is not a wrong number: **the pre-fix
engine cannot price this liability inside a fork at all** (`IndexError: index 100 is out of bounds
for dimension 0 with size 2`, swallowed by the canonical deal guard and caught by the fork's own
shape check).

THE IDENTITY THIS FILE TURNS ON. `CommodityAveragePriceSwapDeal` takes no decision on simulated
state, so a fork adds NO information to it: its mark at the fork row is a function of the state at
that row, and the fork's row `t` IS the outer path's row `t`. So the fork's `L_t` must equal the
outer path's liability mark at `t`, for every inner draw, EXACTLY — no Monte Carlo error, no
tolerance to argue about. Measured: **$0, bitwise, on CUDA**; **$0.625 on a $4.15M notional
(1.5e-07) on CPU**, which is float32 reassociation between a 136-row gather and a 2-row one.

WHAT THE FIX IS NOT, and the brief's own hypothesis refuted with a number. The state is per-outer-
path at the fork ROW, so "broadcast the fork row's value across the inner horizon" looks like the
whole answer. It is exactly half of it: `state[t]` is what step t->t+1 CONSUMES, so the fork's row
`t+1` is what the step produced from the INNER draw — a simulated quantity, and the inner
`generate` already computes it (its `_recursion_seed` seeds row 0 from `mu0_inner`, which is the
outer state at the fork row, so the broadcast half is already right BY CONSTRUCTION). Freezing the
series at the fork row instead leaves `L_t` bit-identical — the broadcast half really is right —
and moves the one-step labels the solver bootstraps from by **1.97e-02 mean / 1.475 max relative
(CPU), 3.61e-03 mean / 6.81e-02 max (CUDA)**, and the solved `V_0` by 1.0% (CPU) / 3.4% (CUDA).

ONE EXPRESSION, and that is the point. The factor path and the published series go through the
SAME publication, so there is no second spelling to get wrong — and a mutation of it necessarily
hits both, which is why the `past off by one row` mutant below is not aimed at `basis_mu` alone.

MUTATION MATRIX — every one RUN.

| mutant | what it spells | result |
|---|---|---|
| publish factor paths only | HEAD | RAISES `inner-fork liability pricing degenerated`, cause `index 100 is out of bounds for dimension 0 with size 2` |
| the past block starts one row late | a misaligned logical grid | identity dies: max|L_t - outer| **$81,911 = 1.97e-02 of notional** (shipped: 1.5e-07) |
| the published series is frozen at the fork row | "it's a broadcast, not a simulation" | `L_t` bit-identical (the broadcast half IS right), `L_t1` moves 1.97e-02 mean, `V_0` moves |

ANTI-PLACEBO — the fixture property each gate needs.

| property | value | what goes blind without it |
|---|---|---|
| the slow mean is ON | `Slow_Mean_Lambda` 0.96875, `Mu_0` 6.25 | with it off there is no `(key, kind)` series at all and every gate here passes vacuously |
| `Mu_0` != the basis `Spot` | 6.25 vs 9.56 | at `Mu_0 == b_0` the AR term `phi^n (b_t - mu_t)` is zero at row 0 and a wrong mean is worth nothing |
| forks at MID-LIFE rows | fork rows 100..133, last fixing at row 120 | past the last fixing the projected half is multiplied by `1 - past = 0`: the mean is read and discarded, and every mutant here survives. Asserted, not assumed. |
| the fork reaches BELOW its own cutoff | cutoff 100 of a 136-row grid | at `cutoff_idx == 0` there is no past block and the geometry gate is vacuous |
| more than one inner draw | `Inner_Sub_Batch` 4 | at 1 draw the flat width equals the outer width and a missing batch map is invisible |

WHY THIS FILE RUNS ON THE CPU. The pre-fix publication makes the pricer gather row `t` out of a
two-row tensor. On CPU that is a clean `IndexError`; on CUDA it is an ASYNCHRONOUS device-side
assert that poisons the context for every later test in the process. `prec` in
`test_average_price_swap.py` is the precedent: the device is a harness knob, not a deal input. The
identity gate's number is reported for both devices above, and the CUDA one is the exact one.
"""
import copy
import inspect
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import derivus as rf
from derivus import calculation, pricing, stochasticprocess, utils
from derivus.calculation import HedgeMonteCarlo

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
APS_WORLD = os.path.join(FIXTURES, 'commodity_aps_world.json')
POLICY = os.path.join(FIXTURES, 'policy_test_simulate_only.json')
TS = lambda d: {'.Timestamp': str(d)[:10]}
NOTIONAL = 2500.0 * 1661.4839212399793      # Units x spot: the scale the identity is read against
LAST_FIXING_ROW = 120                       # 2026-05-15, the row past which the projection is dead


@pytest.fixture(autouse=True)
def _on_the_cpu(monkeypatch):
    """See the module docstring: one mutant's wrong read is an out-of-bounds gather, which CUDA
    reports asynchronously and which poisons the context for the rest of the process."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


# ---------------------------------------------------------------------------
# the world: the APS market data, the shipped hedging problem, neither file edited
# ---------------------------------------------------------------------------

def world(t_min=100, batch=8, inner=4, batches=2):
    """A `Hedge_Monte_Carlo` whose LIABILITY is the average-price swap — the only route to the
    published-series read, and the configuration the note at `get_observed_basis_decay` named.

    Market data, deal and models are `commodity_aps_world.json`'s (the reconciled platinum blocks,
    slow mean ON); the hedging problem is `policy_test_simulate_only.json`'s shape with the
    tradables repointed at the martingale PRIMARY. Both fixtures are read, never written."""
    aps = json.load(open(APS_WORLD))
    md = aps['Calc']['MergeMarketData']['ExplicitMarketData']
    deal = aps['Calc']['Deals']['Deals']['Children'][0]['Children'][0]['Instrument']['.Deal']
    hp = copy.deepcopy(json.load(open(POLICY))['Calc']['Calculation']['Hedging_Problem'])

    base = pd.Timestamp('2026-01-15')
    p0 = md['Price Factors']['CommodityPrice.PLATINUM_CME']['Spot']
    futs, pos, setts, margin, limits = {}, {}, {}, {}, {}
    for i, mat in enumerate([pd.Timestamp(x) for x in ('2026-03-30', '2026-05-28', '2026-07-29')], 1):
        name = f'PL_M{i}'
        futs[name] = {'Maturity_Date': TS(mat), 'Currency': 'USD', 'Carry': 'PLATINUM_CARRY',
                      'Repo_Rate': 'USD-REPO', 'Commodity': 'PLATINUM_CME', 'Contract_Size': 50}
        pos[name] = 0
        setts[name] = round(p0 * float(np.exp(0.019 * (mat - base).days / 365.25)), 4)
        margin[name] = {'Method': 'per_contract', 'Amount': 8500.0}
        limits[name] = {'Min_Position': -50, 'Max_Position': 0}
    hp['Tradable_Instruments']['CommodityFutureDeal'] = futs
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH']['Investment_Horizon'] = TS('2026-05-29')
    state = hp['Portfolio_State']
    state['Positions'], state['Settlement_Prices'], state['Initial_Margin'] = pos, setts, margin
    hp['Evaluator']['Position_Limits'] = limits
    # realised primary spot, strictly before the base date (History_Lookback_Business_Days rows)
    hist = pd.bdate_range(end=base - pd.Timedelta(days=1), periods=30)
    state['Spot_Price_History'] = {'CommodityPrice.PLATINUM_CME': {
        'Dates': [TS(d) for d in hist],
        'Prices': [float(x) for x in p0 * (1.0 + 0.004 * np.sin(np.arange(30) * 0.7))]}}
    hp['Liabilities'] = {'CommodityAveragePriceSwapDeal': {
        'APS1': {k: v for k, v in deal.items() if k not in ('Object', 'Reference')}}}
    hp['Randomize_Initial_State'] = 'No'
    hp['Solver'] = {'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 3,
                    'Training_Action_Chunk_Size': 64, 'T_Min': t_min, 'DiffV2_Fit_Iters': 2}
    return {'Calc': {
        'Calculation': {
            'Object': 'HedgeMonteCarlo', 'Base_Date': TS(base), 'Time_Grid': '0d 1d(1d)',
            'Currency': 'USD', 'Calendar': 'Chicago', 'Batch_Size': batch,
            'Simulation_Batches': batches, 'Random_Seed': 1234, 'Execution_Mode': 'solve_hedge',
            'Inner_MC_Enabled': 'Yes', 'Inner_Sub_Batch': inner, 'Hedging_Problem': hp},
        'MergeMarketData': {'ExplicitMarketData': md},
        'CalendDataFile': './tests/fixtures/data/calendars.cal'}}


def run(cfg, name='fork_state'):
    """One JSON-only run, returning `(V_0, [(bundle, t, L_t, L_t1), ...])` — every fork paired with
    the bundle it forked FROM, because each streamed batch forks from its own."""
    bundles, forks = [], []
    attach, fork = HedgeMonteCarlo._attach_inner_mc, HedgeMonteCarlo._run_inner_mc_at_t

    def _attach(self, bundle, *args, **kwargs):
        bundles.append(bundle)
        return attach(self, bundle, *args, **kwargs)

    def _fork(self, t, *args, **kwargs):
        out = fork(self, t, *args, **kwargs)
        forks.append((bundles[-1], t, out['L_t'].detach().clone(), out['L_t1'].detach().clone()))
        return out

    HedgeMonteCarlo._attach_inner_mc, HedgeMonteCarlo._run_inner_mc_at_t = _attach, _fork
    try:
        cx = rf.Context()
        cx.load_json((json.dumps(cfg), f'{name}.json'))
        _, result = cx.run_job()
        v_0 = ((result.evaluation_summary or {}).get('diagnostics') or {}).get('V_0')
        return v_0, forks
    finally:
        HedgeMonteCarlo._attach_inner_mc, HedgeMonteCarlo._run_inner_mc_at_t = attach, fork


def identity_gap(forks):
    """max |fork's mark at its own row - the outer path's mark at that row|, in dollars."""
    return max(float((L_t - bundle.liability_sim[t].reshape(-1, 1)).abs().max())
               for bundle, t, L_t, _ in forks)


# ---------------------------------------------------------------------------
# 1. the published series is on the factor's logical grid
# ---------------------------------------------------------------------------

def test_a_published_series_is_published_the_way_its_factor_is():
    """Read at the READ SITE: what the pricer is handed for `(key, 'basis_mu')` and what it is
    handed for the basis path itself must be the same kind of object with the same geometry — a
    bare tensor in the outer loop, and inside a fork the same two blocks at the same rows and the
    same widths. Anything else means the two are on different logical grids, which is the defect."""
    seen = []
    priced = pricing.pv_average_price_swap

    def spy(shared, time_grid, deal_data):
        code = deal_data.Factor_dep['Basis_Mu']
        assert code, 'the fixture has the slow mean OFF — every gate in this file is vacuous'
        seen.append((shared.t_Scenario_Buffer[code[0][utils.FACTOR_INDEX_Offset]],
                     shared.t_Scenario_Buffer[deal_data.Factor_dep['Basis'][0][
                         utils.FACTOR_INDEX_Offset]]))
        return priced(shared, time_grid, deal_data)

    pricing.pv_average_price_swap = spy
    try:
        run(world(), 'geometry')
    finally:
        pricing.pv_average_price_swap = priced

    forked = [(mu, b) for mu, b in seen if isinstance(b, utils.ScenarioSource)]
    assert forked, 'no fork published a block sequence — the gate never reached its subject'
    assert len(forked) < len(seen), 'no OUTER pricing happened; the bare-tensor arm is unmeasured'
    for mu, basis in seen:
        assert type(mu) is type(basis), 'the mean and its own factor are on different grids'
        if not isinstance(basis, utils.ScenarioSource):
            continue
        assert len(mu.blocks) == len(basis.blocks) == 2, 'the fork did not reach below its cutoff'
        for got, want in zip(mu.blocks, basis.blocks):
            assert got.first_row == want.first_row and got.n_rows == want.n_rows
            assert got.tensor.shape[-1] == want.tensor.shape[-1]
            assert (got.batch_index is None) == (want.batch_index is None)
            if got.batch_index is not None:
                assert torch.equal(got.batch_index, want.batch_index)
        assert mu.shape == basis.shape
        # the past really is the OUTER width, so the batch map is load-bearing rather than identity
        assert mu.blocks[0].tensor.shape[-1] < mu.blocks[-1].tensor.shape[-1]


# ---------------------------------------------------------------------------
# 2. the identity: a fork adds no information to a deal with no optionality
# ---------------------------------------------------------------------------

def test_the_swap_marks_the_same_inside_a_fork_as_on_the_outer_path():
    """THE gate. `L_t` is the liability at the fork's OWN row, so it cannot depend on the inner
    draws and it cannot differ from the outer path's mark there. Exact on CUDA; the tolerance here
    is float32 reassociation between a full-grid gather and a two-row one, and it is 1.5e-07 of
    the deal's notional against a mutant at 1.97e-02."""
    v_0, forks = run(world(), 'identity')
    assert v_0 is not None and forks, 'the solve produced no forks'
    rows = sorted({t for _, t, _, _ in forks})
    assert min(rows) < LAST_FIXING_ROW, (
        f'every fork row {rows[0]}..{rows[-1]} is past the last fixing (row {LAST_FIXING_ROW}): '
        f'the projected half is multiplied by zero and this gate is blind')
    spread = max(float((L_t.max(dim=1).values - L_t.min(dim=1).values).abs().max())
                 for _, _, L_t, _ in forks)
    assert spread == 0.0, f'the mark at the fork row varies across inner draws by ${spread}'
    gap = identity_gap(forks)
    assert gap / NOTIONAL < 1e-6, f'fork/outer mark disagree by ${gap} ({gap / NOTIONAL:.3e})'


# ---------------------------------------------------------------------------
# 3. the mutants
# ---------------------------------------------------------------------------

def _dedent_to(block):
    """`inspect.getsource` of a method is dedented once, so a pattern written at file indentation
    has to follow it down."""
    return ''.join(ln[4:] if ln.startswith('    ') else ln for ln in block.splitlines(True))


def _mutate(monkeypatch, owner, name, old, new, module):
    """Exec a one-token edit of the method's OWN source onto its module and bind it back. Same
    idiom as `test_average_price_swap._mutate`: everything about a mutant except the edit is the
    shipped code, and its globals ARE the module's."""
    src = textwrap.dedent(inspect.getsource(getattr(owner, name)))
    old, new = _dedent_to(old), _dedent_to(new)
    assert src.count(old) == 1, f'mutation target not unique: {src.count(old)}'
    exec(compile(src.replace(old, new).replace(f'def {name}', 'def _mutant'), '<mutant>', 'exec'),
         module.__dict__)
    monkeypatch.setattr(owner, name, module.__dict__.pop('_mutant'))


PUBLISH_ALL = """                for key in [k for k, v in shared_mem.t_Scenario_Buffer.items()
                            if v is not outer_entries.get(k) and k in outer_scenario_buffer]:"""
PUBLISH_FACTORS_ONLY = """                for key in [k for k in self.stoch_factors_inner
                            if k.type not in utils.DimensionLessFactors]:"""


def test_publishing_the_factor_paths_alone_cannot_price_the_liability_at_all(monkeypatch):
    """HEAD, run. The two-row series is handed to a pricer asking for an outer row, the canonical
    deal guard swallows the `IndexError` into a scalar-0 mark, and the fork's own shape check turns
    that into the loud stop it is. Not a wrong number — an unrunnable configuration."""
    _mutate(monkeypatch, HedgeMonteCarlo, '_run_inner_mc_at_t',
            PUBLISH_ALL, PUBLISH_FACTORS_ONLY, calculation)
    with pytest.raises(RuntimeError, match='liability pricing degenerated'):
        run(world(), 'head')


def test_a_past_block_starting_one_row_late_dies_on_the_identity(monkeypatch):
    """The logical grid misaligned by one row — the failure mode the `first_row` / past-block pair
    exists to prevent. One expression publishes the factor and the series, so this mutant moves
    both; that is the design, not a gap in the mutant."""
    _mutate(monkeypatch, HedgeMonteCarlo, '_run_inner_mc_at_t',
            "                    past = [utils.ScenarioBlock("
            "outer_scenario_buffer[key][:cutoff_idx],\n",
            "                    past = [utils.ScenarioBlock("
            "outer_scenario_buffer[key][1:cutoff_idx + 1],\n",
            calculation)
    _, forks = run(world(), 'offbyone')
    gap = identity_gap(forks)
    assert gap / NOTIONAL > 1e-3, f'the misaligned grid survives the identity (${gap})'


FROZEN_OLD = ("            if path is not None:\n"
              "                shared_mem.t_Scenario_Buffer[(self.factor_key, kind)] = path\n")
FROZEN_NEW = ("            if path is not None:\n"
              "                shared_mem.t_Scenario_Buffer[(self.factor_key, kind)] = (\n"
              "                    path[:1].expand_as(path) if path.ndim == 3 else path)\n")


def test_freezing_the_series_at_the_fork_row_is_right_at_t_and_wrong_at_t_plus_one(monkeypatch):
    """The brief's own hypothesis — "the state is per-outer-path and constant across inner draws,
    likely a broadcast" — run as a mutant, at the one place that owns the series.

    It is half right, and the halves are separated here: `L_t` is BIT-IDENTICAL (the fork's row 0
    already is the outer state at the fork row, seeded by `inner_fork_seed`), while `L_t1` — the
    one-step label the whole fork exists to produce — moves, because `state[t]` is what the step
    t->t+1 consumes and the inner draw is what produced it."""
    v_0, shipped = run(world(), 'shipped')
    _mutate(monkeypatch, stochasticprocess.BasisLinkedSpotModel, '_publish_recursions',
            FROZEN_OLD, FROZEN_NEW, stochasticprocess)
    v_0_frozen, frozen = run(world(), 'frozen')

    assert [t for _, t, _, _ in shipped] == [t for _, t, _, _ in frozen], 'the runs diverged'
    assert all(torch.equal(a[2], b[2]) for a, b in zip(shipped, frozen)), (
        'the mark at the FORK ROW moved — the broadcast half was not already right')
    rel = [float(((a[3] - b[3]).abs() / b[3].abs().clamp_min(1e-9)).max())
           for a, b in zip(shipped, frozen)]
    assert max(rel) > 1e-3, f'freezing the state is a no-op on the one-step labels ({max(rel):.3e})'
    assert v_0 != v_0_frozen, 'the solved value is indifferent to the labels it was fitted on'
