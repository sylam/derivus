"""Sharding one Credit Monte Carlo across workers, deterministically in their number.

THE ORIGINAL DEFECT. `Credit_Monte_Carlo(runparallel=True)` passed SEVEN positional arguments to
six-parameter `run_cmc`, so every child raised TypeError before running and the parent blocked
forever on `results.get()`. The stray was a `False` in the `job_id` slot. Both spellings are
byte-identical at every commit back to the root: the path had never run once.

DETERMINISM IN n IS THE POINT - sharding that changes the answer is a second model. The old scheme
seeded once per WORKER and consumed that stream sequentially, so a batch drew whatever was left
after the batches before it IN ITS PROCESS. Under `runparallel` the seeding is PER BATCH: batch b
seeds from its GLOBAL index, workers take CONTIGUOUS ranges, and the parent merges by worker. The
pooled result is BIT-IDENTICAL across worker counts, asserted by SHA-256 over the pooled `mtm`.

THE PER-BATCH SEED IS MIXED, NOT ADDED. `calculation.batch_seed` is a SplitMix64 round over
`(Random_Seed, b)`: reseeding per batch asks for as many seeds as the job has batches, and
consecutive integers are the weakest input a generator's initialization takes. CUDA's Philox is
counter-based and does not care; CPU MT19937 expands its state by a linear recurrence, which is the
case the literature declines to bless. One mix closes it for both, spelled once.

THE UNSHARDED DEFAULT IS UNTOUCHED: `deterministic_batches` is off everywhere but the `runparallel`
dispatch, verified against a hash taken before this landed. So a sharded run and an unsharded one
do NOT agree bitwise - two valid path sets over one document - and are compared statistically.

WORKER COUNT IS DECOUPLED FROM DEVICE COUNT. `runparallel` is the only knob: `True` is one worker
per visible CUDA device, an int is exactly that many, and worker j lands on
`cuda:(j % device_count)` with the surplus sharing a device. Where there is no CUDA every worker
runs on `cpu` and the same determinism holds, which is why the CPU arm carries no skip marker.

Bit-identity is per device TYPE, not across types: `manual_seed` drives different generators, so a
CPU shard and a CUDA shard are different path sets. Within a type the two RTX 3090s in this box
were measured byte-for-byte equal, which `test_the_two_devices_agree_bitwise` pins.

BOTH STREAMS ARE ANCHORED. A world whose outer path draws quasi-random numbers reads a Sobol
sequence that is reproducible but POSITION-dependent, and position is what sharding moves.
`set_quasi_batch` hands the quasi stream the same global index the generator gets, and the anchored
arm takes batch b's draw from absolute position `1024 + b * sample_size` - which is where the
historical engine already stands on an unsharded run, pinned directly against `SobolEngine`.

THERE IS ONE `quasi_rng`, NOT TWO: the draw, the memo, the clamp and the inverse-CDF happen once
and the arms differ only in the index and position they supply. The historical arm reads its
engine's STANDING position rather than recomputing it, because a dimension drawn at two sample
sizes shares one engine and interleaves. The narrowed refusal covers one shape: two draws of the
same `(dimension, sample_size)` INSIDE a batch, which have no distinct position between them.
An inner Monte Carlo's Sobol rows are the canonical inner block's, a function of the path count
alone, so they need no anchor and an inner-MC book shards byte for byte too.

THE BAND, MEASURED, for the unsharded-vs-sharded comparison only. Over 20 seeds the two estimates
have seed relative sds of 1.10% and 0.99%, so their difference carries sd ~1.56%; the observed max
gap was 2.89%, 1.85 sd, and `BAND` is 8%. Paired with a noise-FREE check: the t=0 row is
deterministic and the two agreed there to 0.0 across all 20 seeds.

Only the LINEAR columns pool by averaging, so everything here pools the `mtm` and re-summarizes -
which is also what makes the equality exact, one reduction over the same columns in the same order.

THIS BOX: two RTX 3090s, GPU 1 driving the display and so behind the 2s TDR watchdog. The document
is sized so no kernel goes near it - 512 paths x 4 batches over an 8-row grid, 0.03-0.04s a run.

A SECOND BLOCKER, ALSO CLOSED. `Config` held a pyparsing grammar whose parse actions are local
closures, so it could not be pickled into a spawned child. The grammar is derived state, so
`__getstate__` drops it and `__setstate__` rebuilds it; under `fork` neither hook runs.
"""
import ast
import hashlib
import inspect
import os
import re
import sys
import textwrap
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.multiprocessing as mp
from torch.overrides import TorchFunctionMode

import derivus
import rates_world
from derivus import calculation, utils
from derivus.config import Config
from derivus.instruments import construct_instrument

DEVICE_COUNT = torch.cuda.device_count()

needs_two_devices = pytest.mark.skipif(
    DEVICE_COUNT < 2,
    reason='this gate reads a SECOND CUDA device - worker j runs on cuda:(j %% device_count), so '
           'showing that two of them carry the shard needs at least 2 visible devices and this '
           'box reports {}. The determinism itself is device-agnostic and is gated on CPU '
           'without a marker.'.format(DEVICE_COUNT))

needs_cuda = pytest.mark.skipif(
    not DEVICE_COUNT,
    reason='pins the unsharded CUDA stream by hash, which needs a visible CUDA device to produce - '
           'this box has none, and the CPU pin gates the same property on the same code. Keyed on '
           'the device COUNT rather than `is_available()`, which reports True with nothing visible')

BASE = pd.Timestamp('2024-06-28')
SPOT, VOL, RATE = 100.0, 0.20, 0.02
#: 4 batches divides exactly by the 1, 2 and 4 worker counts the equality arms compare.
BATCH_SIZE, SIMULATION_BATCHES = 512, 4
GRID = '0d 6m(3m)'
SEED = 1
#: ~5 sd of the measured 1.56% difference sd. Unsharded-vs-sharded ONLY: the sharded-vs-sharded
#: comparisons assert equality and carry no tolerance at all.
BAND = 0.08

CALL = {
    'Object': 'EquityOptionDeal', 'Reference': 'EQCALL', 'Currency': 'USD',
    'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
    'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
    'Option_Style': 'European', 'Strike_Price': 100.0, 'Units': 100.0,
    'Expiry_Date': BASE + pd.DateOffset(years=2), 'Settlement_Style': 'Cash',
    'Option_On_Forward': 'No', 'Payoff_Type': 'Standard'}

PUT = dict(CALL, Reference='EQPUT', Option_Type='Put', Strike_Price=95.0, Units=80.0)


def job_document():
    """Two vanilla European options on one GBM equity in a flat-rate USD world. BOUGHT, so the
    exposure is one-sided and the band's statistic keeps away from a near-zero denominator. GBM
    draws from the torch generator rather than Sobol and a vanilla prices in closed form, which is
    what puts this document inside the determinism boundary.
    """
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, VOL] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 3.0)])},
    }
    c.params['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': VOL, 'Drift': RATE}}
    c.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceModel')
    c.deals = {'Attributes': {'Reference': 'multigpu'},
               'Deals': {'Children': [{'Instrument': construct_instrument(CALL, {})},
                                      {'Instrument': construct_instrument(PUT, {})}]},
               'Calculation': {'Object': 'CreditMonteCarlo', 'Base_Date': BASE,
                               'Currency': 'USD'}}
    return c


def overrides(seed=SEED):
    return {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': GRID,
            'Batch_Size': BATCH_SIZE, 'Simulation_Batches': SIMULATION_BATCHES,
            'Random_Seed': seed, 'Currency': 'USD', 'Tenor_Offset': 0.0,
            'Deflation_Interest_Rate': 'USD', 'Percentile': '95',
            'Generate_Cashflows': 'No', 'Dynamic_Scenario_Dates': 'No'}


def sha(array):
    """A hash, so an equality failure prints two short strings rather than two 8x2048 matrices."""
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()[:16]


def shard(job_id, num_jobs, device=None, seed=SEED, deterministic=True):
    """One shard, run HERE. `run_cmc` takes the same six positional arguments the dispatch passes
    it, and the two keyword ones the dispatch passes by name."""
    calc, out = derivus.run_cmc(job_document(), torch.float32, overrides(seed),
                               job_id, num_jobs, None,
                               deterministic_batches=deterministic, device=device)
    return str(calc.device), out


def pooled(n, device=None, seed=SEED):
    """All n shards in this process, pooled in WORKER order. In-process because what these gates
    measure is the NUMBERS and the arithmetic is the same either way; the spawned path is exercised
    by `sharded` and the end-to-end tests. Pooling concatenates the `mtm` and re-summarizes, so the
    reduction reads the same columns in the same order an unsharded run does.
    """
    devices, frames = [], []
    for job_id in range(n):
        device_used, out = shard(job_id, n, device, seed)
        devices.append(device_used)
        frames.append(out['Results']['mtm'])
    matrix = np.concatenate([f.values for f in frames], axis=1)
    return {'devices': devices, 'mtm': matrix, 'frames': frames,
            'profile': derivus.summarize_data(matrix, '95')}


# ------------------------------------------------------------------ the call shape is repaired

def test_the_worker_spawn_binds_the_signature():
    """The old seven-argument shape is gone and what the dispatch passes now BINDS. Read off the
    AST so the assertion is about the CALL and not its layout: the six positional arguments are
    still six - the determinism work added its two by KEYWORD so the fragile tuple did not grow
    again - and `signature.bind` is the check the interpreter makes.
    """
    source = textwrap.dedent(inspect.getsource(derivus.Context.Credit_Monte_Carlo))
    spawns = [node for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'Process']
    assert len(spawns) == 1, 'expected exactly one worker spawn, found %d' % len(spawns)
    keywords = {kw.arg: kw.value for kw in spawns[0].keywords}
    assert 'args' in keywords, 'the spawn no longer passes its arguments positionally through args='
    passed = [ast.unparse(e) for e in keywords['args'].elts]
    assert len(passed) == 6, (
        'the spawn passes %d positional arguments - the arity defect is back: %s'
        % (len(passed), passed))

    signature = inspect.signature(derivus.run_cmc)
    by_name = {ast.unparse(k): ast.unparse(v)
               for k, v in zip(keywords['kwargs'].keys, keywords['kwargs'].values)} \
        if 'kwargs' in keywords else {}
    bound = signature.bind(*passed, **{k.strip("'"): v for k, v in by_name.items()})
    assert bound.arguments['context'] == 'self.current_cfg'
    assert bound.arguments['job_id'] == 'i'
    assert bound.arguments['num_jobs'] == 'num_workers'
    assert bound.arguments['res_queue'] == 'results'
    assert bound.arguments['deterministic_batches'] == 'True', (
        'the dispatch no longer asks for per-batch seeding, so it is no longer deterministic in n')

    # the shape that was there before genuinely does not bind, which is why it never ran
    with pytest.raises(TypeError):
        signature.bind('cfg', 'prec', 'ov', False, 'i', 'n', 'q', 'extra', 'extra2')


def test_a_config_survives_the_pickle_a_spawn_puts_it_through():
    """The `__getstate__`/`__setstate__` pair, needing no device. The grammar is dropped and rebuilt
    on the far side, so what matters is that the REBUILT parsers work: a Config arriving without
    them raises `AttributeError` on the first `parse_grid`, deep inside a child.
    """
    import pickle

    revived = pickle.loads(pickle.dumps(job_document()))
    assert hasattr(revived, 'gridparser') and hasattr(revived, 'periodparser'), (
        'the parsers were dropped on the way out and never rebuilt on the way in')
    assert revived.parse_period('3M') == pd.DateOffset(months=3)
    assert len(revived.parse_grid(BASE, BASE + pd.DateOffset(years=1), GRID)) > 1
    assert revived.params['System Parameters']['Base_Currency'] == 'USD'
    assert len(revived.deals['Deals']['Children']) == 2


def test_the_worker_count_knob_is_runparallel_itself():
    """`True` means one per device and an int means that many - no second knob and no cap. The CPU
    fallback is load-bearing: `device_count()` is 0 without CUDA, and a zero-worker run would leave
    the parent blocked on a queue nothing writes to.
    """
    assert derivus.worker_count(True) == (DEVICE_COUNT or 1)
    assert derivus.worker_count(True) >= 1, 'True must never resolve to zero workers'
    for n in (1, 2, 3, 7):
        assert derivus.worker_count(n) == n, 'an int worker count is not capped by the devices'
    for bad in (0, -1):
        with pytest.raises(ValueError):
            derivus.worker_count(bad)


# ------------------------------------------------------------------ deterministic in n, on CPU
# No skip marker on this arm: the determinism is device-agnostic, so it is gated on every box.

def test_cpu_sharding_is_bit_identical_in_the_worker_count():
    """Shard the same job 1, 2 and 4 ways on CPU and the pooled `mtm` is byte-for-byte the same
    matrix. Equality, not a tolerance."""
    one = pooled(1, device='cpu')
    assert one['devices'] == ['cpu']
    reference = sha(one['mtm'])
    assert reference == '268a30bfa45063ec', 'the sharded CPU stream moved: %s' % reference

    for n in (2, 4):
        many = pooled(n, device='cpu')
        assert many['devices'] == ['cpu'] * n
        assert many['mtm'].shape == one['mtm'].shape
        assert sha(many['mtm']) == reference, (
            'cpu n=%d pooled mtm %s against n=1 %s - the shard count moved the answer'
            % (n, sha(many['mtm']), reference))
        assert np.array_equal(many['mtm'], one['mtm'])
        # and the profile the caller actually reads
        assert np.array_equal(many['profile'].values, one['profile'].values)


def test_cpu_shards_carry_their_own_contiguous_batch_range():
    """Worker j owns `[j*k, (j+1)*k)`, so the shards partition the paths rather than repeat them."""
    per_worker = SIMULATION_BATCHES // 2
    frames = pooled(2, device='cpu')['frames']
    assert [f.shape[1] for f in frames] == [BATCH_SIZE * per_worker] * 2
    # different global batch indices means different seeds means different paths
    assert not np.array_equal(frames[0].values, frames[1].values), (
        'the two shards priced the same paths - the batch ranges overlap')
    # except at t=0, which is deterministic on every path
    assert np.array_equal(frames[0].values[0], frames[1].values[0])


def hmm_document():
    """A commodity future on a `MarkovHMMSpotModel` - the cheapest world whose OUTER path draws
    quasi-random numbers. `generate` calls `quasi_rng` unconditionally once per batch, the regime
    being the model, where `GARCHSpotModel` needs an optional `Drift_States` chain and
    `BasisLinkedSpotModel` a parent commodity. The carry factor carries no model entry, so exactly
    ONE stochastic process draws from the Sobol stream.
    """
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
        'CommodityPrice.PLAT': {'Spot': 950.0, 'Currency': 'USD', 'Interest_Rate': 'USD',
                                'Forward_Rate': 'PLAT_CARRY'},
        'ForwardRate.PLAT_CARRY': {'Currency': 'USD',
                                   'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
    }
    c.params['Price Models'] = {'MarkovHMMSpotModel.PLAT': {
        'States': [{'Mu': 0.02, 'Sigma': 0.20}, {'Mu': -0.05, 'Sigma': 0.45}],
        'Transition_Matrix': [[0.98, 0.02], [0.05, 0.95]],
        'Initial_State_Probs': [0.7, 0.3],
        'Calibration_DT_Years': 1.0 / 252.0}}
    c.params['Model Configuration'].append('CommodityPrice', (), 'MarkovHMMSpotModel')
    # ten contracts: `Units` is the future's declared size, read since 2026-09-26 - the four pins
    # below were re-taken then, each matrix exactly ten times the one it replaced
    future = {'Object': 'CommodityFutureDeal', 'Reference': 'FUT', 'Commodity': 'PLAT',
              'Currency': 'USD', 'Repo_Rate': 'USD', 'Carry': 'PLAT_CARRY',
              'Maturity_Date': BASE + pd.DateOffset(years=1), 'Units': 10.0,
              'Payoff_Currency': 'USD'}
    c.deals = {'Attributes': {'Reference': 'hmm'},
               'Deals': {'Children': [{'Instrument': construct_instrument(future, {})}]},
               'Calculation': {'Object': 'CreditMonteCarlo', 'Base_Date': BASE,
                               'Currency': 'USD'}}
    return c


def hmm_pooled(n, device=None, deterministic=True):
    """The HMM world's n shards, pooled in worker order."""
    frames = []
    for job_id in range(n):
        _, out = derivus.run_cmc(hmm_document(), torch.float32, overrides(),
                                 job_id, n, None,
                                 deterministic_batches=deterministic, device=device)
        frames.append(out['Results']['mtm'].values)
    return np.concatenate(frames, axis=1)


def test_the_hmm_world_really_does_draw_from_the_quasi_stream():
    """The premise the coverage gate rests on: a world that quietly stopped drawing would make it
    pass for the wrong reason, testing the generator path twice."""
    from derivus.stochasticprocess import MarkovHMMSpotModel

    body = inspect.getsource(MarkovHMMSpotModel.generate)
    assert 'quasi_rng' in body, 'MarkovHMMSpotModel no longer draws quasi-random numbers'
    # and the document stands up and prices
    _, out = derivus.run_cmc(hmm_document(), torch.float32, overrides(), 0, 1, None, device='cpu')
    mtm = out['Results']['mtm']
    assert np.isfinite(mtm.values).all()
    assert mtm.values.std(axis=1)[1:].min() > 0.0, 'the future was skipped rather than priced'


def test_a_sobol_consuming_world_is_bit_identical_in_the_worker_count():
    """THE COVERAGE GATE: the quasi stream is anchored rather than refused, so a world that draws
    from it shards byte-for-byte. Unanchored, worker 1 of a two-way shard would start its engine at
    position zero and read global batch 2 out of the points batch 0 should have had - and the
    pooled matrix would move with n while every worker still agreed with itself.
    """
    reference = sha(hmm_pooled(1, device='cpu'))
    assert reference == 'a9e2f5e42834078d', 'the sharded HMM CPU stream moved: %s' % reference
    for n in (2, 4):
        assert sha(hmm_pooled(n, device='cpu')) == reference, (
            'the Sobol-consuming world moved at n=%d - the quasi stream is reading its position '
            'from history rather than from the global batch index' % n)


def inner_pooled(n, device='cpu'):
    """A book priced by an inner Monte Carlo - a discretely monitored down-and-out call on the GBM
    equity, 32 paths a batch and 64 inner, so the Sobol arm - its shards pooled in worker order."""
    deal = dict(CALL, Object='EquityBarrierOption', Reference='EQDAO', Barrier_Type='Down_And_Out',
                Barrier_Price=85.0, Cash_Rebate=0.0, Expiry_Date=BASE + pd.DateOffset(years=1),
                Barrier_Dates=[BASE + pd.DateOffset(months=m) for m in range(1, 13)],
                Barrier_Monitoring_Frequency=pd.DateOffset(days=0))
    frames = []
    for job_id in range(n):
        config = job_document()
        config.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
        _, out = derivus.run_cmc(config, torch.float32,
                                 dict(overrides(), Batch_Size=32, MCMC_Simulations=64),
                                 job_id, n, None, deterministic_batches=True, device=device)
        frames.append(out['Results']['mtm'].values)
    return np.concatenate(frames, axis=1)


def test_an_inner_monte_carlo_book_is_bit_identical_in_the_worker_count():
    """An inner-MC book shards byte for byte: its Sobol rows are the canonical inner block's, a
    function of the path count alone, so no inner draw has a position to anchor.

    Killing mutation: the block's anchor advancing with each batch the process runs - a position
    reintroduced - moves the pooled matrix at n = 2 and 4."""
    reference = sha(inner_pooled(1))
    assert reference == 'fb1ba147026afcca', 'the sharded inner-MC CPU stream moved: %s' % reference
    for n in (2, 4):
        assert sha(inner_pooled(n)) == reference, (
            'the inner-MC book moved at n=%d - an inner draw reads a position' % n)


@needs_two_devices
def test_a_sobol_consuming_world_is_bit_identical_across_devices():
    """The same, across the two real devices, where worker j also changes silicon."""
    reference = sha(hmm_pooled(1))
    assert reference == '8532978a799719ce', 'the sharded HMM CUDA stream moved: %s' % reference
    for n in (2, 4):
        assert sha(hmm_pooled(n)) == reference, (
            'the Sobol-consuming world moved at n=%d on CUDA' % n)


def test_the_unsharded_hmm_path_is_untouched():
    """The quasi anchoring is reachable ONLY through `set_quasi_batch`, so an ordinary caller keeps
    the free-running engine. Pinned by hash; the GBM pin below was taken before this work and is
    the real before/after evidence, where this one pins the Sobol world going forward.
    """
    _, cpu = derivus.run_cmc(hmm_document(), torch.float32, overrides(), 0, 1, None,
                             deterministic_batches=False, device='cpu')
    assert sha(cpu['Results']['mtm'].values) == 'cd34f5838cab3d1d', (
        'the unsharded HMM path moved on cpu: %s' % sha(cpu['Results']['mtm'].values))
    # and it is NOT the sharded answer: anchoring moves which points a batch reads and the
    # per-batch reseed moves the generator, so the two are different valid path sets
    assert sha(cpu['Results']['mtm'].values) != sha(hmm_pooled(1, device='cpu'))


@needs_cuda
def test_the_unsharded_hmm_path_is_untouched_on_cuda():
    """The same pin on the device, where a different generator backend feeds the same stream."""
    _, out = derivus.run_cmc(hmm_document(), torch.float32, overrides(), 0, 1, None,
                             deterministic_batches=False, device=None)
    assert sha(out['Results']['mtm'].values) == 'fbf7c38c5c63fc60', (
        'the unsharded HMM path moved on cuda: %s' % sha(out['Results']['mtm'].values))


def test_the_anchored_position_is_the_unsharded_one():
    """The arithmetic the anchoring rests on, against `SobolEngine` itself: batch b's draw sits at
    `1024 + b * sample_size`, which is where the historical free-running engine stands after b
    batches of one draw each. Off by anything the sharded numbers would still be self-consistent,
    so only a comparison against the UNSHARDED stream catches it.
    """
    from torch.quasirandom import SobolEngine

    dimension, size, anchor, seed = 9, 512, 1024, 1234

    running = SobolEngine(dimension=dimension, scramble=True, seed=seed)
    running.fast_forward(anchor)
    historical = [running.draw(size, dtype=torch.float64) for _ in range(5)]

    for b, expected in enumerate(historical):
        fresh = SobolEngine(dimension=dimension, scramble=True, seed=seed)
        fresh.fast_forward(anchor + b * size)
        assert torch.equal(fresh.draw(size, dtype=torch.float64), expected), (
            'anchored batch %d does not land on the position the unsharded run reads' % b)

    # a worker that skips to its own slice, then advances forward only, stays on the rails
    skipped = SobolEngine(dimension=dimension, scramble=True, seed=seed)
    skipped.fast_forward(anchor + 3 * size)
    assert torch.equal(skipped.draw(size, dtype=torch.float64), historical[3])
    assert torch.equal(skipped.draw(size, dtype=torch.float64), historical[4])


def _quasi_state():
    """A cold `CMC_State` carrying only what the quasi stream reads, on the unsharded arm."""
    from derivus.calculation import CMC_State
    state = CMC_State.__new__(CMC_State)
    state.one = torch.zeros(1, dtype=torch.float64)
    state.t_quasi_rng, state.t_quasi_rng_batch = {}, {}
    state.sobol_position, state.sobol_engine = {}, {}
    state.quasi_batch = None
    return state


def test_the_historical_arm_is_still_one_free_running_engine():
    """Deduping the two arms into one `quasi_rng` must not move the default path. The historical arm
    reads its engine's STANDING position, so it is the same engine advancing draw after draw -
    including the case the anchored index arithmetic cannot express, where ONE dimension drawn at
    two sample sizes interleaves on a single engine. Neither hash-gated world does that, so the
    unsharded pins would not have caught it.
    """
    from derivus.utils import Calculation_State
    QUASI_ANCHOR, QUASI_SEED = Calculation_State.QUASI_ANCHOR, Calculation_State.QUASI_SEED
    from torch.quasirandom import SobolEngine

    state = _quasi_state()

    # one dimension, two sample sizes, interleaved - and a repeat of the first shape after
    plan = [(7, 64), (7, 32), (7, 64), (7, 32)]

    reference = SobolEngine(dimension=7, scramble=True, seed=QUASI_SEED)
    reference.fast_forward(QUASI_ANCHOR)

    for dimension, size in plan:
        expected = reference.draw(size, dtype=torch.float64)
        u = state.quasi_rng(dimension, size)[1]
        margin = 1.0e-6
        assert torch.equal(u, expected.clamp(min=margin, max=1.0 - margin)), (
            'the historical arm left the free-running engine at (%d, %d)' % (dimension, size))


def test_a_draw_wider_than_the_engine_is_its_chunks_at_successive_positions():
    """`SobolEngine` caps its dimension at 21201, and `oss_uniforms` asks for the PATH COUNT as the
    dimension - so every OSS pricer refused above the cap, `Deal.calculate` swallowed it into a
    `CRITICAL ... skipped` line and the run died downstream on a collapsed frame. 20480 ran; 21248
    and 32768 did not.

    A wider draw is now taken in chunks of at most the cap at successive positions and
    concatenated, each chunk advancing the stream by `sample_size` exactly as one draw of that
    width does. Held here against `SobolEngine` itself, both sides of the cap: 21201 is one engine
    at the anchor, 32768 is 21201 at the anchor plus 11567 one sample_size on.
    """
    from torch.quasirandom import SobolEngine
    from derivus.utils import Calculation_State
    QUASI_ANCHOR, QUASI_SEED = Calculation_State.QUASI_ANCHOR, Calculation_State.QUASI_SEED
    SOBOL_MAX_DIMENSION = SobolEngine.MAXDIM

    margin, size = 1.0e-6, 8

    def raw(dimension, position):
        engine = SobolEngine(dimension=dimension, scramble=True, seed=QUASI_SEED)
        engine.fast_forward(position)
        return engine.draw(size, dtype=torch.float64)

    at_cap = _quasi_state().quasi_rng(SOBOL_MAX_DIMENSION, size)[1]
    assert torch.equal(at_cap, raw(SOBOL_MAX_DIMENSION, QUASI_ANCHOR).clamp(
        min=margin, max=1.0 - margin)), 'a draw AT the cap is no longer the single engine own bytes'

    wide = 32768
    over = _quasi_state().quasi_rng(wide, size)[1]
    assert over.shape == (size, wide), over.shape
    expected = torch.cat([raw(SOBOL_MAX_DIMENSION, QUASI_ANCHOR),
                          raw(wide - SOBOL_MAX_DIMENSION, QUASI_ANCHOR + size)], dim=1)
    assert torch.equal(over, expected.clamp(min=margin, max=1.0 - margin)), (
        'the chunked draw is not its chunks at successive positions')


def test_a_wide_draw_advances_the_stream_by_every_chunk_it_took():
    """A chunked draw consumes `span * sample_size` of the stream, and BOTH arms have to step by
    that or two draws of one shape return the same points.

    The historical arm reads a standing position keyed by the width it was ASKED for, which above
    the cap is never a width any engine has - left keyed by the chunks, it stood at the anchor
    for ever and a second draw repeated the first byte for byte. The anchored arm reads
    `QUASI_ANCHOR + batch * span * sample_size`; at one stride per batch, chunk 1 of batch b and
    chunk 0 of batch b+1 are the same 21201 points.

    Both are held against the property that survives either arm: successive draws differ, and the
    two arms agree draw for draw, which is what makes anchoring a repositioning rather than a
    second model.
    """
    from torch.quasirandom import SobolEngine
    SOBOL_MAX_DIMENSION = SobolEngine.MAXDIM

    size = 8
    for dimension in (1024, SOBOL_MAX_DIMENSION, SOBOL_MAX_DIMENSION + 1, 32768, 63603):
        historical, previous = _quasi_state(), None
        for batch in range(3):
            walked = historical.quasi_rng(dimension, size)[1]
            anchored = _quasi_state()
            anchored.set_quasi_batch(batch)
            assert torch.equal(anchored.quasi_rng(dimension, size)[1], walked), (
                'the two arms part at dimension %d, batch %d' % (dimension, batch))
            if batch:
                assert not torch.equal(walked, previous), (
                    'dimension %d repeated its draw: the stream did not advance' % dimension)
            previous = walked


def test_a_second_draw_in_one_batch_is_refused_by_name():
    """The narrowed refusal: anchoring gives a draw the position of its BATCH, so two draws of one
    `(dimension, sample_size)` within a batch have no distinct position between them.
    """
    state = _quasi_state()
    state.set_quasi_batch(2)

    state.quasi_rng(4, 64)
    with pytest.raises(RuntimeError, match='second draw of one quasi-random stream'):
        state.quasi_rng(4, 64)

    # a different shape is a different stream and is fine
    assert state.quasi_rng(4, 32)[0].shape[0] == 32


def test_the_anchored_stream_is_a_function_of_the_batch_alone():
    """Two states standing at different points in their own history draw the SAME batch the same
    way - which is the whole property sharding needs and the one the historical path lacks."""
    # one state walks batches 0..3, as an unsharded worker would
    walker = _quasi_state()
    walked = {}
    for b in range(4):
        walker.set_quasi_batch(b)
        walked[b] = walker.quasi_rng(6, 128)[1].clone()

    # another starts cold at batch 2, as the second worker of a two-way shard does
    latecomer = _quasi_state()
    latecomer.set_quasi_batch(2)
    assert torch.equal(latecomer.quasi_rng(6, 128)[1], walked[2])
    latecomer.set_quasi_batch(3)
    assert torch.equal(latecomer.quasi_rng(6, 128)[1], walked[3])

    # and the batches genuinely differ from one another - the anchor is not pinning them together
    assert not torch.equal(walked[0], walked[1])


def _block_state(batch, dtype=torch.float32):
    """A cold `CMC_State` carrying only what the inner block reads."""
    from derivus.calculation import CMC_State
    state = CMC_State.__new__(CMC_State)
    state.one, state.simulation_batch, state.t_inner_block = torch.ones(1, dtype=dtype), batch, {}
    return state


def test_the_inner_block_is_a_function_of_its_rows_and_paths_alone():
    """`CMC_State.inner_block` is pure in `(rows, sims)`: states of different batch sizes asking in
    either order read the same rows, a shorter request is a longer one's prefix, chunk c is
    dimensions `64c` on of `SobolEngine` `64(c + 1)` wide at `QUASI_ANCHOR` - one point per path
    over all its rows - the double halves sum to exactly one and each float32 half is its double
    rounded once.

    Killing mutations: the complement taken after the cast breaks the float32 half; the block keyed
    on the batch breaks the two states' agreement; chunk c read `c * sims` points further along one
    64-dimensional engine breaks chunk 1."""
    from derivus.calculation import CMC_State
    INNER_CHUNK, INNER_MARGIN = CMC_State.INNER_CHUNK, CMC_State.INNER_MARGIN
    QUASI_ANCHOR, QUASI_SEED = CMC_State.QUASI_ANCHOR, CMC_State.QUASI_SEED
    from torch.quasirandom import SobolEngine

    sims, rows = 1024, 2 * INNER_CHUNK + 2
    wide, narrow = _block_state(512, torch.float64), _block_state(512)
    first, short = narrow.inner_block(rows, sims), narrow.inner_block(10, sims)
    other = _block_state(7)
    late_short, late = other.inner_block(10, sims), other.inner_block(rows, sims)
    exact = wide.inner_block(rows, sims)
    for a, b, c, d, e in zip(first, short, late_short, late, exact):
        assert torch.equal(a[:10], b) and torch.equal(b, c) and torch.equal(a, d), (
            'the block depends on the batch or on the order it was asked in')
        assert torch.equal(a, e.float()), 'a float32 half is not its exact double rounded once'
    assert (exact[0] + exact[1] == 1.0).all(), 'the double halves are not exact complements'
    margin = INNER_MARGIN * 2.0 ** -30
    for chunk in range(3):
        engine = SobolEngine(dimension=INNER_CHUNK * (chunk + 1), scramble=True, seed=QUASI_SEED)
        engine.fast_forward(QUASI_ANCHOR)
        expected = engine.draw(sims, dtype=torch.float64)[:, chunk * INNER_CHUNK:].T.clamp(
            margin, 1.0 - margin)
        rows_here = exact[0][chunk * INNER_CHUNK:(chunk + 1) * INNER_CHUNK]
        assert torch.equal(rows_here, expected[:len(rows_here)]), (
            'chunk %d is not dimensions %d on of the engine %d wide at QUASI_ANCHOR' % (
                chunk, chunk * INNER_CHUNK, INNER_CHUNK * (chunk + 1)))


# ------------------------------------------------------------------ deterministic in n, on CUDA

@needs_two_devices
def test_cuda_sharding_is_bit_identical_in_the_worker_count():
    """The same equality across real devices, and past the device count: at n=4 on a 2-device box
    two workers share cuda:0 and two share cuda:1, and the pooled matrix does not move."""
    one = pooled(1)
    assert one['devices'] == ['cuda:0']
    reference = sha(one['mtm'])
    assert reference == 'fbc3ebac89d5399a', 'the sharded CUDA stream moved: %s' % reference

    expected_devices = {2: ['cuda:0', 'cuda:1'],
                        4: ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1']}
    for n in (2, 4):
        many = pooled(n)
        assert many['devices'] == expected_devices[n], (
            'worker j must land on cuda:(j %% %d) - got %s' % (DEVICE_COUNT, many['devices']))
        assert sha(many['mtm']) == reference, (
            'cuda n=%d pooled mtm %s against n=1 %s - the shard count moved the answer'
            % (n, sha(many['mtm']), reference))
        assert np.array_equal(many['profile'].values, one['profile'].values)


@needs_two_devices
def test_the_two_devices_agree_bitwise():
    """The claim the equality above rests on, isolated. Same global batch, same seed, one device
    each: the per-batch seeding fixes the stream from the batch index alone, so the two must
    produce the identical matrix if the devices are bitwise equal on this path. They are on this
    box, and if a future box's pair are not, this says so by name.
    """
    _, zero = shard(0, 1, device='cuda:0')
    _, one = shard(0, 1, device='cuda:1')
    a, b = zero['Results']['mtm'].values, one['Results']['mtm'].values

    assert a.shape == b.shape
    if not np.array_equal(a, b):                      # pragma: no cover - not this box
        gap = np.abs(a - b)
        pytest.fail(
            'the two devices are NOT bitwise equal on this path: sha %s vs %s, max abs %.3e, '
            'max rel %.3e, %d of %d entries differ. The n-invariance gates above rest on this, '
            'so they cannot hold on this hardware.'
            % (sha(a), sha(b), gap.max(), (gap / np.maximum(np.abs(a), 1e-30)).max(),
               int((gap != 0).sum()), gap.size))
    assert sha(a) == sha(b)


# ------------------------------------------------------------------ the spawned path, both devices

def _shard_worker(job_id, num_jobs, seed, device, lib_queue, probe_queue):
    """THE CHILD, spawned as the dispatch spawns it: `lib_queue` lands on `res_queue`, so the
    library's own merge payload is produced by its own code path. The Config is built HERE - one
    fewer thing between the seed and the numbers - and the probe payload is what the parent cannot
    otherwise learn, a parent being unable to see a child's CUDA allocation.
    """
    calc, out = derivus.run_cmc(job_document(), torch.float32, overrides(seed),
                               job_id, num_jobs, lib_queue,
                               deterministic_batches=True, device=device)
    mtm = out['Results']['mtm']
    probe_queue.put({
        'job_id': job_id,
        'device': str(calc.device),
        'allocated_on_own_device': (torch.cuda.memory_allocated(calc.device)
                                    if calc.device.type == 'cuda' else None),
        'batches_actually_run': out['Stats']['Simulation_Batches'],
        'mtm_shape': tuple(mtm.shape),
        'mtm_finite': bool(np.isfinite(mtm.values).all()),
        # row 0 is t=0, where every path still sits at spot and the sd is legitimately zero
        'min_cross_path_sd': float(mtm.values.std(axis=1)[1:].min()),
        'index': [str(x) for x in mtm.index],
        'sha': sha(mtm.values),
        'EE': np.asarray(out['Results']['exposure_profile']['EE'], dtype=np.float64)})


@pytest.fixture(scope='module')
def sharded():
    """One spawn round on CUDA. Both queues are drained BEFORE any join - the dispatch's own
    ordering, and the only one that cannot deadlock on a full pipe."""
    n = DEVICE_COUNT
    lib_queue, probe_queue = mp.Queue(), mp.Queue()
    workers = [mp.Process(target=_shard_worker, args=(i, n, SEED, None, lib_queue, probe_queue))
               for i in range(n)]
    for w in workers:
        w.start()
    try:
        library = [lib_queue.get(timeout=300) for _ in range(n)]
        probes = [probe_queue.get(timeout=300) for _ in range(n)]
    finally:
        for w in workers:
            w.join(timeout=300)
            if w.is_alive():          # pragma: no cover - a hung child is a failed run
                w.terminate()
    assert all(w.exitcode == 0 for w in workers), (
        'a worker exited non-zero: %s' % [w.exitcode for w in workers])
    return {'library': sorted(library, key=lambda p: p['Job']),
            'probes': sorted(probes, key=lambda p: p['job_id'])}


@needs_two_devices
def test_both_devices_ran_their_own_shard(sharded):
    """Evidence, not assumption: each worker reports the device it was given, from inside its own
    process, with a live allocation on it."""
    probes = sharded['probes']
    assert [p['job_id'] for p in probes] == list(range(DEVICE_COUNT))
    devices = [p['device'] for p in probes]
    assert devices == ['cuda:%d' % i for i in range(DEVICE_COUNT)], (
        'worker j must run on cuda:(j %% %d) - got %s' % (DEVICE_COUNT, devices))
    assert len(set(devices)) == DEVICE_COUNT, 'the workers collapsed onto one device: %s' % devices
    for p in probes:
        assert p['allocated_on_own_device'] > 0, (
            'worker %d reports no allocation on %s, so nothing ran there'
            % (p['job_id'], p['device']))


@needs_two_devices
def test_the_spawned_shards_match_the_in_process_ones(sharded):
    """The spawn changes nothing about the numbers, which is what lets every equality gate above run
    in-process and still describe the real dispatch."""
    here = pooled(DEVICE_COUNT)
    assert [p['sha'] for p in sharded['probes']] == [sha(f.values) for f in here['frames']], (
        'the spawned shards and the in-process shards disagree')


@needs_two_devices
def test_the_results_are_finite_and_correctly_shaped(sharded):
    """Shape, finiteness and the DISPERSION guard: a deal whose pricer raised is swallowed and
    contributes zeros, which reads as a valid deep-OTM profile unless dispersion is checked."""
    per_worker = SIMULATION_BATCHES // DEVICE_COUNT
    rows = len(sharded['probes'][0]['index'])
    assert rows > 1, 'the profile collapsed to a single row'
    for p in sharded['probes']:
        assert p['mtm_finite'], 'worker %d returned a non-finite mtm' % p['job_id']
        assert p['mtm_shape'] == (rows, BATCH_SIZE * per_worker)
        assert p['batches_actually_run'] == per_worker
        assert p['min_cross_path_sd'] > 0.0, (
            'worker %d has a stochastic grid row with zero dispersion - a deal was skipped'
            % p['job_id'])
        assert np.isfinite(p['EE']).all()
        assert (p['EE'] > 0.0).all(), 'bought options must carry a positive exposure'


@needs_two_devices
def test_the_library_merge_payload_comes_back_keyed_by_worker(sharded):
    """`run_cmc`'s `res_queue` branch, carrying the five keys the parent's merge reads - `Job` among
    them, which is what lets the parent order a race-ordered queue. BOTH batch counts report what
    this worker RAN: `Calculation.execute` rebinds `params` through `declared_defaults` before
    `//= num_jobs`, so `Params` carried the document's request and a consumer pooling off it
    over-counted the paths behind each worker by `num_jobs`.
    """
    per_worker = SIMULATION_BATCHES // DEVICE_COUNT
    assert [p['Job'] for p in sharded['library']] == list(range(DEVICE_COUNT))
    for payload in sharded['library']:
        assert set(payload) == {'Results', 'Stats', 'Params', 'Reference', 'Job'}
        assert payload['Reference'] == 'multigpu'
        assert payload['Stats']['Simulation_Batches'] == per_worker
        assert payload['Params']['Simulation_Batches'] == per_worker, (
            'Params reports %d batches where the worker ran %d'
            % (payload['Params']['Simulation_Batches'], per_worker))
        assert np.isfinite(payload['Results']['exposure_profile'].values).all()


# ------------------------------------------------------------------ the whole path, end to end

def _end_to_end(runparallel, device):
    cx = derivus.Context()
    cx.current_cfg = job_document()
    return cx.Credit_Monte_Carlo(overrides(), runparallel, device=device)


def _check_merged(merged, n, reference):
    assert set(merged) == {'Results', 'Stats', 'Params', 'Reference', 'Job'}
    assert all(len(v) == n for v in merged.values()), (
        'the merge lost a worker: %s' % {k: len(v) for k, v in merged.items()})
    assert merged['Job'] == list(range(n)), (
        'the merge is in completion order, not worker order: %s' % merged['Job'])
    matrix = np.concatenate([r['mtm'].values for r in merged['Results']], axis=1)
    assert sha(matrix) == sha(reference), (
        'the end-to-end pooled mtm %s does not match the in-process reference %s'
        % (sha(matrix), sha(reference)))
    return matrix


def test_the_cpu_context_path_shards_end_to_end():
    """`Credit_Monte_Carlo(runparallel=2, device='cpu')` - the real spawned dispatch, pooled
    bit-identically against the in-process n=1 CPU reference. The gate that needed all of it: the
    arity, or the child raises TypeError and the parent blocks forever; `Config.__getstate__`, or
    `start()` raises before a child exists; per-batch seeding, or two workers do not reproduce one;
    the `Job` key, or the merge is a race.
    """
    merged = _end_to_end(2, 'cpu')
    _check_merged(merged, 2, pooled(1, device='cpu')['mtm'])
    for stats in merged['Stats']:
        assert stats['Simulation_Batches'] == SIMULATION_BATCHES // 2


def test_the_cpu_path_is_deterministic_past_the_device_count():
    """Four workers, no devices at all involved - the worker count is its own knob."""
    merged = _end_to_end(4, 'cpu')
    _check_merged(merged, 4, pooled(1, device='cpu')['mtm'])


@needs_two_devices
def test_the_cuda_context_path_shards_end_to_end():
    """The same call on the real devices, `runparallel=True` resolving to one worker per device."""
    merged = _end_to_end(True, None)
    _check_merged(merged, DEVICE_COUNT, pooled(1)['mtm'])


# ------------------------------------------------------------------ sharded vs unsharded

@needs_two_devices
def test_the_sharded_estimate_agrees_with_the_unsharded_one():
    """The distribution comparison, and the only tolerance in this file. A sharded run and an
    unsharded one are two DIFFERENT valid path sets over one document, so they agree in
    distribution and not bitwise; the band is measured. Pooling concatenates the `mtm` and
    re-summarizes, because `EE` pools by averaging equal shards and `PFE_<p>` does not.
    """
    _, unsharded = shard(0, 1, deterministic=False)
    reference = np.asarray(
        unsharded['Results']['exposure_profile']['EE'], dtype=np.float64)
    estimate = np.asarray(pooled(DEVICE_COUNT)['profile']['EE'], dtype=np.float64)
    assert estimate.shape == reference.shape

    # the deterministic row carries no MC noise at all, so it is not a band question
    assert estimate[0] == pytest.approx(reference[0], rel=1e-9), (
        'the deterministic t=0 row disagrees (%r vs %r) - the two are not pricing the same '
        'document' % (estimate[0], reference[0]))

    got = abs(estimate.mean() - reference.mean()) / abs(reference.mean())
    assert got < BAND, (
        'sharded mean-EE %.4f against unsharded %.4f is a %.4f relative gap, outside the measured '
        '%.2f band (~5 sd of the 1.56%% seed spread)'
        % (estimate.mean(), reference.mean(), got, BAND))


@needs_two_devices
def test_the_unsharded_default_did_not_move():
    """The historical stream, pinned by hash: `deterministic_batches` defaults off, so an ordinary
    caller draws what it drew before the determinism work landed. This hash was taken from the tree
    before those changes.
    """
    _, out = shard(0, 1, deterministic=False)
    assert sha(out['Results']['mtm'].values) == '2df61471b2970c5e', (
        'the unsharded path moved: %s' % sha(out['Results']['mtm'].values))


# ------------------------------------------------------------------ host traffic per batch

#: The per-batch count `HostTraffic` takes on `traffic_document`, CPU and CUDA alike: 801 while
#: deal-static arrays went up and wing decisions came back every batch, 54 once they do not.
HOST_TRAFFIC_PER_BATCH = 54

SKEW_TENORS = [0.02, 0.25, 0.5, 0.75, 1.0, 2.0]
SKEW = {'ATM_Vol': 0.25, 's': -0.15, 'L': 0.30, 'R': 0.20, 'C': -0.35, 'D': 0.35, 'lam': 0.5,
        'rho': 0.5}
QUARTERS = [BASE + pd.DateOffset(months=3 * k) for k in range(1, 5)]
ON_SKEW = {'Equity': 'SK', 'Dividends': 'SK', 'Discount_Rate': 'USD', 'Equity_Volatility': 'SK',
           'Currency': 'USD', 'Payoff_Currency': 'USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
           'Strike_Price': 100.0, 'Expiry_Date': QUARTERS[-1]}
TRAFFIC_DEALS = [
    dict(ON_SKEW, Object='EquityBarrierOption', Reference='SKB', Units=1.0, Cash_Rebate=0.0,
         Barrier_Type='Down_And_Out', Barrier_Price=85.0,
         Barrier_Monitoring_Frequency=pd.DateOffset(days=0),
         Barrier_Dates=[[d, ''] for d in QUARTERS]),
    dict(ON_SKEW, Object='QEDI_CustomAutoCallSwap', Reference='SKAC', Units=10.0,
         Settlement_Style='Cash', Option_On_Forward='No', Option_Style='European', Barrier=0.0,
         Payoff_Type='Standard', Price_Fixing=[[d, 0.0] for d in QUARTERS],
         Autocall_Coupons=[[d, 0.02] for d in QUARTERS],
         Autocall_Thresholds=[[d, 1.0] for d in QUARTERS], Barrier_Dates=[], Autocall_Floating=[]),
    dict(rates_world.par_swap('SWAP', 'USD', 'USD', 'USD', 2, 2.0), Effective_Date=BASE,
         Maturity_Date=BASE + pd.DateOffset(years=2))]


def traffic_document():
    """`job_document`'s two vanillas on a flat surface, beside a second GBM equity on a Skew
    surface carrying a discretely monitored barrier and a one-step-survival autocall, and a swap."""
    c = job_document()
    c.params['Price Factors'].update({
        'EquityPrice.SK': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.SK': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.0, 0.01], [5.0, 0.01]])},
        'EquityPriceVol.SK': dict(
            {'Surface_Type': 'Skew', 'Moneyness_Rule': 'Sticky_Moneyness', 'Currency': 'USD',
             'ATM_Ref': utils.Curve([], [[t, SPOT] for t in SKEW_TENORS])},
            **{k: utils.Curve([], [[t, v] for t in SKEW_TENORS]) for k, v in SKEW.items()})})
    c.params['Price Models']['GBMAssetPriceModel.SK'] = {'Vol': 0.25, 'Drift': RATE}
    c.deals['Deals']['Children'] += [{'Instrument': construct_instrument(deal, {})}
                                     for deal in TRAFFIC_DEALS]
    return c


class HostTraffic(TorchFunctionMode):
    """Counts the torch calls that read a tensor's value on the host or hand a tensor host data,
    by name and innermost derivus line: each is a stream sync on the card, and a call on the CPU."""

    READS = {'__bool__', 'item', 'tolist', 'numpy', 'cpu', '__float__', '__int__', '__index__',
             'nonzero', 'equal', 'allclose', 'is_nonzero', 'masked_select', 'unique'}

    def __init__(self):
        super().__init__()
        self.sites = Counter()

    @staticmethod
    def host(value):
        return isinstance(value, (np.ndarray, list, tuple, float, int)) and not isinstance(
            value, bool)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, '__name__', '')
        if name in self.READS:
            hit = bool(args) and torch.is_tensor(args[0])
        elif name == '__getitem__':
            # a gather by a device index is no sync; by numpy, another device or a mask it is
            index = args[1] if isinstance(args[1], tuple) else (args[1],)
            hit = any(isinstance(i, (np.ndarray, list)) or torch.is_tensor(i) and (
                i.device != args[0].device or i.dtype == torch.bool) for i in index)
        elif name in ('new', 'new_tensor'):
            hit = len(args) > 1 and self.host(args[1])
        elif name in ('tensor', 'as_tensor'):
            hit = 'device' in kwargs and bool(args) and self.host(args[0])
        else:
            hit = any(isinstance(a, np.ndarray) and a.ndim for a in args)
        if hit:
            inner = [f for f in traceback.extract_stack()[:-1]
                     if os.sep + 'derivus' + os.sep in os.path.normpath(f.filename)]
            self.sites['{} {}'.format(name, '{}:{}'.format(
                os.path.basename(inner[-1].filename), inner[-1].lineno) if inner else '')] += 1
        return func(*args, **kwargs)


def host_traffic(batches, device):
    """`HostTraffic`'s sites over one run of `traffic_document`, and the deals it skipped."""
    mode = HostTraffic()
    with mode:
        calc, _ = derivus.run_cmc(traffic_document(), torch.float32, dict(
            overrides(), Batch_Size=32, MCMC_Simulations=64, Simulation_Batches=batches),
            device=device)
    return mode.sites, calc.calc_stats.get('Deals Skipped', 0)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=needs_cuda)])
def test_a_batch_reads_back_and_uploads_only_what_it_simulated(device):
    """What a batch hands between host and device and recomputes from static data is the count at
    three batches less the count at one, halved: 801 before, 54 after, on CPU and CUDA alike. A
    surface's grid and expiry, its wings' zero test, a curve's scenario rows, a step's length and
    a leg's index arrays are fixed for the run, so they go up or come back once a calculation.

    Killed by any one reverted: the Skew read's numpy-indexed gathers and wing bools (+675), the
    curve read's per-batch row upload (+19), the autocall walk's step test on the device (+23),
    the flat surface's plan and the strip's and float leg's per-batch uploads (+29)."""
    once, once_skipped = host_traffic(1, device)
    thrice, thrice_skipped = host_traffic(3, device)
    assert once_skipped == thrice_skipped == 0, 'a deal was skipped, so its reads went uncounted'
    per_batch = (sum(thrice.values()) - sum(once.values())) / 2
    assert per_batch <= HOST_TRAFFIC_PER_BATCH, (
        '%s host reads and uploads a batch on %s against %d, over two batches: %s' % (
            per_batch, device, HOST_TRAFFIC_PER_BATCH, (thrice - once).most_common(12)))


# ------------------------------------------------------------------ a batch that would page

class OnASmallCard(calculation.Credit_Monte_Carlo):
    """A credit Monte Carlo told its device has 16 KiB free, wherever it runs."""

    def device_free_bytes(self):
        return 2 ** 14


def small_card_run(batch, calc=OnASmallCard):
    """`job_document` at `batch` paths, one batch, on the host."""
    job = job_document()
    return calc(job, prec=torch.float32, device=torch.device('cpu')).execute(
        dict(job.deals['Calculation'], **dict(overrides(), Batch_Size=batch, Simulation_Batches=1)))


def test_a_batch_that_would_page_is_refused_naming_the_width_that_fits():
    """A `Batch_Size` past the device's free memory is refused before its first batch naming the
    largest that fits, which runs while one path more is refused; the host checks nothing. Killed
    by: the guard not called; the width that fits off by a path; the host checked."""
    with pytest.raises(ValueError, match='Batch_Size %d needs' % BATCH_SIZE) as refused:
        small_card_run(BATCH_SIZE)
    fits = int(re.search(r'Batch_Size (\d+) fits', str(refused.value)).group(1))
    assert 0 < fits < BATCH_SIZE, str(refused.value)
    small_card_run(fits)
    with pytest.raises(ValueError, match='Batch_Size %d needs' % (fits + 1)):
        small_card_run(fits + 1)
    small_card_run(BATCH_SIZE, calculation.Credit_Monte_Carlo)
