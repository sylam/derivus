"""Sharding one Credit Monte Carlo across workers, deterministically in their number.

THE ORIGINAL DEFECT. `Context.Credit_Monte_Carlo(runparallel=True)` spawns workers and shards the
simulation across them. It could not: the spawn passed SEVEN positional arguments to six-parameter
`run_cmc`, so every child raised TypeError before running and the parent then blocked forever on
`results.get()`. The stray was a `False` sitting in the `job_id` slot, which pushed `i`,
`num_workers` and `results` each one position right and left `results` unbindable. `ef5ac31` is the
repo's ROOT commit, and there `run_cmc` already carried exactly six parameters while the call site
already passed seven, both byte-identical at every commit since - the two were never consistent in
this repository's history, and the path had never run once. Dropping the `False` is the fix; the
signature had lost nothing.

DETERMINISM IN n IS THE POINT. Sharding that changes the answer is not parallelism, it is a second
model. The old scheme seeded once per WORKER - `CMC_State.__init__` calls
`manual_seed(seed + job_id)` and the batch loop then consumes that stream sequentially - so a batch
drew whatever was left after the batches before it in ITS process, and running the same job over
one worker or two covered different paths. Under `runparallel` the seeding is now PER BATCH: batch
b seeds from its GLOBAL index, workers take CONTIGUOUS ranges (worker j owns `[j*k, (j+1)*k)`), and
the parent merges by worker rather than by completion. Batch b is therefore the same batch drawing
the same numbers whichever worker runs it and wherever that worker sits, and the pooled result is
BIT-IDENTICAL across worker counts - asserted here by SHA-256 over the pooled `mtm`, not by a
tolerance.

THE PER-BATCH SEED IS MIXED, NOT ADDED. `calculation.batch_seed` is a SplitMix64 round over
`(Random_Seed, b)` rather than `Random_Seed + b`, because reseeding per batch asks for as many
seeds as the job has batches - a 1024-batch job wants 1024 of them - and consecutive integers are
the weakest input a generator's initialization takes. CUDA's Philox is counter-based and
key-independent by construction and does not care; CPU MT19937 expands its state from the seed by a
linear recurrence, which is the case the literature declines to bless rather than one it endorses.
One mix closes it for both without answering the question per backend, and there is exactly one
spelling of it so a sharded run and an unsharded-but-batch-seeded one cannot derive a batch's seed
two different ways.

The measured readings on this box: pooled `mtm` over 4 batches x 512 paths is byte-for-byte equal
at n = 1, 2 and 4 workers, on CUDA and on CPU independently, for BOTH a world that draws only from
the generator (two vanilla options on a GBM equity) and one that also draws quasi-random numbers
(a commodity future on a `MarkovHMMSpotModel`).

THE UNSHARDED DEFAULT IS UNTOUCHED. `deterministic_batches` is off everywhere except the
`runparallel` dispatch, so every ordinary caller keeps the historical stream bit for bit - verified
against a hash taken before any of this landed, and `tests/test_hn_barrier_cmc.py` is the standing
regression gate. A sharded run and an unsharded one therefore do NOT agree bitwise: they are two
different valid path sets over the same document, and `test_the_sharded_estimate_agrees_with_the_
unsharded_one` compares them statistically, in the band measured below.

WORKER COUNT IS DECOUPLED FROM DEVICE COUNT. `runparallel` is the knob and there is no second one:
`True` keeps the historical meaning, one worker per visible CUDA device, and an int asks for
exactly that many. Workers are NOT capped by the device count - worker j lands on
`cuda:(j % device_count)` and the surplus share a device - so a four-way shard runs on this
two-device box, and where there is no CUDA at all every worker runs on `cpu` and the same per-batch
determinism holds. That is why the CPU arm below carries NO skip marker: the deterministic sharding
is gated on every box, and only the tests that read a SECOND device are conditional.

Bit-identity is per device TYPE, not across types. `manual_seed` drives different generators on CPU
and CUDA, so a CPU shard and a CUDA shard of the same document are different path sets - a real
difference in the paths, not a rounding gap, and nothing here claims otherwise. Within a type the
two RTX 3090s in this box were measured byte-for-byte equal on this path, which
`test_the_two_devices_agree_bitwise` pins by hash.

BOTH STREAMS ARE ANCHORED. A world whose outer path draws quasi-random numbers
(`MarkovHMMSpotModel`, `GARCHSpotModel`, `BasisLinkedSpotModel`) reads a Sobol sequence that is
perfectly reproducible but POSITION-dependent, and position is precisely what sharding moves: the
historical path advances one engine through every draw before it IN THIS PROCESS, so a worker whose
slice starts at global batch 6 reads the points batch 0 should have had. `set_quasi_batch` hands the
quasi stream the same global index the generator gets, and the anchored arm of `quasi_rng` then
takes batch b's draw from absolute position `1024 + b * sample_size` however many draws the asking
worker has already made. On an unsharded run that IS where the historical engine already stands -
one draw per batch advances it by exactly `sample_size` - which is why the default path is
untouched, and `test_the_anchored_position_is_the_unsharded_one` pins that arithmetic directly
against `SobolEngine` rather than taking it on trust.

THERE IS ONE `quasi_rng`, NOT TWO. The draw, the memo, the clamp and the inverse-CDF happen once;
the arms differ only in what they supply as the index and the position - the historical one its
own process-local draw count and wherever its engine has reached, the sharded one the global batch
and that batch's own place in the sequence. The historical arm deliberately reads its engine's
STANDING position rather than recomputing it from the index, because a dimension drawn at two
different sample sizes shares one engine and interleaves, and there the two would part company;
byte-identity of the default path is a property of the code, not only of the two worlds gated here.

The narrowed refusal that remains covers one shape only: two draws of the same
`(dimension, sample_size)` INSIDE a single batch, which have no distinct batch position between
them. Two factors sharing a shape, or an inner Monte Carlo pricer drawing beside the outer path,
are the sources; a memoized re-read after `reset_qrg` is the inner-MC replay idiom, is not a second
draw, and returns the same tensor by identity. Inner-MC Sobol is out of scope for n-invariance and
this is where that is said.

THE BAND, MEASURED, for the unsharded-vs-sharded comparison only. Over 20 seeds the unsharded
mean-EE estimate has a seed relative sd of 1.10% and the sharded one 0.99%, so the difference of
two such estimates carries sd ~= sqrt(2) * 1.10% = 1.56%. The observed max
|sharded - unsharded| / unsharded over those seeds was 2.89%, i.e. 1.85 sd. `BAND` is 8%, ~5 sd.
Paired with it is a noise-FREE check: the t=0 report row is deterministic (every path still sits at
spot) and the two agreed there to 0.000e+00 across all 20 seeds.

Only the LINEAR columns pool by averaging - `EE` and `ENE` are means over paths, `PFE_<p>` is a
percentile and is not - so everything here pools the `mtm` and re-summarizes rather than averaging
profiles, which is also what makes the equality exact: one reduction over the same columns in the
same order, rather than a mean of means.

THIS BOX. Two RTX 3090s, and GPU 1 drives the display, so it sits behind the Windows TDR watchdog
at its 2s default. The document is sized so no kernel goes near that: 512 paths x 4 batches over an
8-row grid, which measures at 0.03-0.04s per run. Do not raise these without re-checking that.

A SECOND BLOCKER, ALSO CLOSED. `Config` held a pyparsing grammar whose parse actions are
`_trim_arity.<locals>.wrapper` closures, so it could not be pickled into a spawned child and
`w.start()` raised `AttributeError: Can't get local object` on every spawn platform. The grammar is
pure derived state, so `Config.__getstate__` drops it and `__setstate__` rebuilds it from
`get_grid_grammar()`; under `fork` neither hook runs.
"""
import ast
import hashlib
import inspect
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.multiprocessing as mp

import derivus
from derivus import utils
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
#: ~5 sd of the measured 1.56% difference sd; see the module docstring. Unsharded-vs-sharded ONLY -
#: the sharded-vs-sharded comparisons assert equality and carry no tolerance at all.
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
    """Two vanilla European options on one GBM equity in a flat-rate USD world.

    Bought options, so the exposure is one-sided and EE carries the whole profile - it keeps the
    statistic the band is measured on away from a near-zero denominator. GBM draws from the torch
    generator rather than the Sobol stream and a vanilla prices in closed form rather than through
    an inner Monte Carlo, which is what puts this document inside the determinism boundary.
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
    c.deals = {'Attributes': {'Reference': 'multigpu', 'Tag_Titles': ''},
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
    """All n shards in this process, pooled in WORKER order - the column order one worker produces.

    In-process because what these gates measure is the NUMBERS, and the sharding arithmetic is the
    same whether the shards run here or in n children; the spawned path is exercised separately by
    `sharded` and by the end-to-end tests. Pooling concatenates the `mtm` and re-summarizes rather
    than averaging profiles, so the reduction reads the same columns in the same order an unsharded
    run does - which is what makes the equality exact rather than close.
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
    """The old seven-argument shape is gone, and what the dispatch passes now binds.

    Read off the AST rather than the source text so the assertion is about the CALL, not its
    layout. The six POSITIONAL arguments are still six - the determinism work added its two
    arguments by KEYWORD precisely so the fragile positional tuple did not grow again - and
    `signature.bind` is the same check the interpreter makes.
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

    # and the shape that was there before genuinely does not bind, which is why it never ran
    with pytest.raises(TypeError):
        signature.bind('cfg', 'prec', 'ov', False, 'i', 'n', 'q', 'extra', 'extra2')


def test_a_config_survives_the_pickle_a_spawn_puts_it_through():
    """The `__getstate__`/`__setstate__` pair, on its own and without needing any device.

    The grammar is dropped rather than pickled and rebuilt on the far side, so what matters is that
    the REBUILT parsers work - a Config that arrives without them raises `AttributeError` on the
    first `parse_grid`, one frame deep inside a child where nobody is watching.
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
    """`True` means one per device and an int means that many. No second knob, and no cap.

    The CPU fallback is the load-bearing case: `torch.cuda.device_count()` is 0 without CUDA, and a
    zero-worker run would leave the parent blocked on a queue nothing ever writes to.
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
    """THE RULING, on the device every box has: shard the same job 1, 2 and 4 ways on CPU and the
    pooled `mtm` is byte-for-byte the same matrix. Equality, not a tolerance."""
    one = pooled(1, device='cpu')
    assert one['devices'] == ['cpu']
    reference = sha(one['mtm'])
    assert reference == 'f11f3e223dd243b7', 'the sharded CPU stream moved: %s' % reference

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
    # ... except at t=0, which is deterministic on every path
    assert np.array_equal(frames[0].values[0], frames[1].values[0])


def hmm_document():
    """A commodity future on a `MarkovHMMSpotModel` - the cheapest world whose OUTER path draws
    quasi-random numbers, which is the whole point of it being here.

    `MarkovHMMSpotModel.generate` calls `quasi_rng(T + 1, Batch_Size)` unconditionally and exactly
    once per batch - the regime IS the model, so there is no switch that turns the draw off and no
    companion factor to stand up. `GARCHSpotModel` reaches the same stream only once an optional
    `Drift_States` chain is configured, and `BasisLinkedSpotModel` needs a parent commodity and a
    `Band_Mixture` reversion, so both cost more document for the same coverage. The carry factor
    carries no model entry deliberately: it stays deterministic, leaving exactly ONE stochastic
    process drawing from the Sobol stream.
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
    future = {'Object': 'CommodityFutureDeal', 'Reference': 'FUT', 'Commodity': 'PLAT',
              'Currency': 'USD', 'Repo_Rate': 'USD', 'Carry': 'PLAT_CARRY',
              'Maturity_Date': BASE + pd.DateOffset(years=1), 'Units': 10.0,
              'Payoff_Currency': 'USD'}
    c.deals = {'Attributes': {'Reference': 'hmm', 'Tag_Titles': ''},
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
    """The premise the coverage gate rests on. A world that quietly stopped drawing would make the
    gate below pass for the wrong reason - it would be testing the generator path twice."""
    from derivus.stochasticprocess import MarkovHMMSpotModel

    body = inspect.getsource(MarkovHMMSpotModel.generate)
    assert 'quasi_rng' in body, 'MarkovHMMSpotModel no longer draws quasi-random numbers'
    # and the document stands up and prices
    _, out = derivus.run_cmc(hmm_document(), torch.float32, overrides(), 0, 1, None, device='cpu')
    mtm = out['Results']['mtm']
    assert np.isfinite(mtm.values).all()
    assert mtm.values.std(axis=1)[1:].min() > 0.0, 'the future was skipped rather than priced'


def test_a_sobol_consuming_world_is_bit_identical_in_the_worker_count():
    """THE COVERAGE GATE. The quasi stream is anchored, not refused: a world that draws from it
    shards byte-for-byte, on the device every box has.

    This is what the anchoring buys. Unanchored, worker 1 of a two-way shard would start its
    engine at position zero and read global batch 2 out of the points batch 0 should have had, and
    the pooled matrix would move with n while every worker still agreed with itself.
    """
    reference = sha(hmm_pooled(1, device='cpu'))
    assert reference == 'bd4f479854a56fcc', 'the sharded HMM CPU stream moved: %s' % reference
    for n in (2, 4):
        assert sha(hmm_pooled(n, device='cpu')) == reference, (
            'the Sobol-consuming world moved at n=%d - the quasi stream is reading its position '
            'from history rather than from the global batch index' % n)


@needs_two_devices
def test_a_sobol_consuming_world_is_bit_identical_across_devices():
    """The same, across the two real devices, where worker j also changes silicon."""
    reference = sha(hmm_pooled(1))
    assert reference == '16d1acb3658a91b3', 'the sharded HMM CUDA stream moved: %s' % reference
    for n in (2, 4):
        assert sha(hmm_pooled(n)) == reference, (
            'the Sobol-consuming world moved at n=%d on CUDA' % n)


def test_the_unsharded_hmm_path_is_untouched():
    """The quasi anchoring is reachable ONLY through `set_quasi_batch`, so an ordinary caller keeps
    the free-running engine. Pinned by hash on both devices, alongside the GBM pin below - that one
    was taken before any of this work and is the real before/after evidence; this one pins the
    Sobol world going forward.
    """
    _, cpu = derivus.run_cmc(hmm_document(), torch.float32, overrides(), 0, 1, None,
                             deterministic_batches=False, device='cpu')
    assert sha(cpu['Results']['mtm'].values) == '571f7d2265552420', (
        'the unsharded HMM path moved on cpu: %s' % sha(cpu['Results']['mtm'].values))
    # and it is NOT the sharded answer - anchoring moves which points a batch reads, and the
    # per-batch reseed moves the generator, so the two are different valid path sets
    assert sha(cpu['Results']['mtm'].values) != sha(hmm_pooled(1, device='cpu'))


@needs_cuda
def test_the_unsharded_hmm_path_is_untouched_on_cuda():
    """The same pin on the device, where a different generator backend feeds the same stream."""
    _, out = derivus.run_cmc(hmm_document(), torch.float32, overrides(), 0, 1, None,
                             deterministic_batches=False, device=None)
    assert sha(out['Results']['mtm'].values) == 'eb1d50954f7eb319', (
        'the unsharded HMM path moved on cuda: %s' % sha(out['Results']['mtm'].values))


def test_the_anchored_position_is_the_unsharded_one():
    """The arithmetic the quasi anchoring rests on, checked against `SobolEngine` itself.

    The anchored arm claims batch b's draw sits at absolute position
    `1024 + b * sample_size`, and that this is exactly where the historical free-running engine
    stands after b batches of one draw each. If that were off by anything the sharded numbers
    would still be self-consistent - every worker would agree with every other - and only a
    comparison against the UNSHARDED stream would catch it, which is what this is.
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

    # and a worker that skips to its own slice, then advances forward only, stays on the rails
    skipped = SobolEngine(dimension=dimension, scramble=True, seed=seed)
    skipped.fast_forward(anchor + 3 * size)
    assert torch.equal(skipped.draw(size, dtype=torch.float64), historical[3])
    assert torch.equal(skipped.draw(size, dtype=torch.float64), historical[4])


def test_the_historical_arm_is_still_one_free_running_engine():
    """Deduping the two arms into one `quasi_rng` must not have moved the default path.

    The historical arm reads its engine's STANDING position, so it is the same engine advancing
    draw after draw it always was - including the case the anchored index arithmetic cannot
    express, where ONE dimension is drawn at two different sample sizes and the two draws
    interleave on a single engine. That case is why the arms supply position differently rather
    than both computing `anchor + index * size`, and the unsharded hash gates would not have caught
    it, because neither gated world does it.
    """
    from derivus.calculation import QUASI_ANCHOR, QUASI_SEED, CMC_State
    from torch.quasirandom import SobolEngine

    state = CMC_State.__new__(CMC_State)
    state.one = torch.zeros(1, dtype=torch.float64)
    state.t_quasi_rng, state.t_quasi_rng_batch, state.sobol_position = {}, {}, {}
    state.quasi_batch = None                      # the default, unsharded arm

    # one dimension, two sample sizes, interleaved - and a repeat of the first shape after
    plan = [(7, 64), (7, 32), (7, 64), (7, 32)]

    reference = SobolEngine(dimension=7, scramble=True, seed=QUASI_SEED)
    reference.fast_forward(QUASI_ANCHOR)

    for dimension, size in plan:
        expected = reference.draw(size, dtype=torch.float64)
        _, u = state.quasi_rng(dimension, size)
        margin = 1.0e-6
        assert torch.equal(u, expected.clamp(min=margin, max=1.0 - margin)), (
            'the historical arm left the free-running engine at (%d, %d)' % (dimension, size))


def test_the_anchored_memo_does_not_grow_with_the_batch_count():
    """The memo is a WITHIN-batch convenience on the anchored arm, and is dropped between batches.

    It cannot be on the historical arm: there a draw's position is wherever the engine has crawled
    to, so a dropped entry is unrecoverable and the memo has to live for the whole run. Anchored,
    position comes entirely from the key, so a re-draw is bit-identical and keeping the previous
    batch's entries only grows a dict nothing will read - 20,480 bytes a batch, measured, on a
    two-state HMM.

    Both halves are asserted: that it stays bounded, AND that bounding it moved no number.
    """
    from derivus.calculation import CMC_State

    def bare():
        state = CMC_State.__new__(CMC_State)
        state.one = torch.zeros(1, dtype=torch.float64)
        state.t_quasi_rng, state.t_quasi_rng_batch, state.sobol_position = {}, {}, {}
        state.quasi_batch = None
        return state

    walker = bare()
    drawn, sizes = [], []
    for b in range(8):
        walker.set_quasi_batch(b)
        drawn.append(walker.quasi_rng(6, 128)[1].clone())
        sizes.append(len(walker.t_quasi_rng))
    assert sizes == [1] * 8, (
        'the anchored memo grows with the batch count: %s' % sizes)

    # the entries a batch makes DO stand within it - the replay idiom reads them
    walker.set_quasi_batch(99)
    walker.quasi_rng(6, 128)
    walker.quasi_rng(6, 64)
    assert len(walker.t_quasi_rng) == 2, 'within-batch entries were dropped underneath the batch'

    # and dropping the earlier batches changed nothing: each is still what a cold state draws
    for b, expected in enumerate(drawn):
        cold = bare()
        cold.set_quasi_batch(b)
        assert torch.equal(cold.quasi_rng(6, 128)[1], expected), (
            'batch %d moved when the memo stopped carrying it' % b)


def test_a_second_draw_in_one_batch_is_refused_by_name():
    """The narrowed refusal: inner-MC Sobol is out of scope for n-invariance, and says so.

    Anchoring gives a draw the position of its BATCH, so two draws of one
    `(dimension, sample_size)` within a batch have no distinct position between them. The refusal
    fires on the second DRAW, not on a memoized re-read - `reset_qrg` followed by the same request
    is the inner-MC replay idiom and must still return the identical tensor.
    """
    from derivus.calculation import CMC_State

    state = CMC_State.__new__(CMC_State)          # the quasi stream needs none of the rest
    state.one = torch.zeros(1, dtype=torch.float64)
    state.t_quasi_rng, state.t_quasi_rng_batch, state.sobol_position = {}, {}, {}
    state.quasi_batch = None
    state.set_quasi_batch(2)

    first = state.quasi_rng(4, 64)
    with pytest.raises(RuntimeError, match='second draw of one quasi-random stream'):
        state.quasi_rng(4, 64)

    # the replay idiom is not a second draw: same tensor, by identity
    state.reset_qrg()
    assert state.quasi_rng(4, 64)[0] is first[0]

    # a different shape is a different stream and is fine
    assert state.quasi_rng(4, 32)[0].shape[0] == 32


def test_the_anchored_stream_is_a_function_of_the_batch_alone():
    """Two states standing at different points in their own history draw the SAME batch the same
    way - which is the whole property sharding needs and the one the historical path lacks."""
    from derivus.calculation import CMC_State

    def fresh():
        state = CMC_State.__new__(CMC_State)
        state.one = torch.zeros(1, dtype=torch.float64)
        state.t_quasi_rng, state.t_quasi_rng_batch, state.sobol_position = {}, {}, {}
        state.quasi_batch = None
        return state

    # one state walks batches 0..3, as an unsharded worker would
    walker = fresh()
    walked = {}
    for b in range(4):
        walker.set_quasi_batch(b)
        walked[b] = walker.quasi_rng(6, 128)[1].clone()

    # another starts cold at batch 2, as the second worker of a two-way shard does
    latecomer = fresh()
    latecomer.set_quasi_batch(2)
    assert torch.equal(latecomer.quasi_rng(6, 128)[1], walked[2])
    latecomer.set_quasi_batch(3)
    assert torch.equal(latecomer.quasi_rng(6, 128)[1], walked[3])

    # and the batches genuinely differ from one another - the anchor is not pinning them together
    assert not torch.equal(walked[0], walked[1])


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
    """The claim the equality above rests on, isolated and hex-compared.

    Same global batch, same seed, one device each. `manual_seed(Random_Seed + b)` fixes the stream
    from the batch index alone, so running batch 0 as worker 0 of a 1-way shard and then forcing
    the same batch onto cuda:1 must produce the identical matrix if the two 3090s are bitwise
    equal on this path. They are, on this box - and if a future box's pair are not, this is the
    gate that says so by name rather than the equality gates failing obscurely.
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
    """THE CHILD. Spawned as the dispatch spawns it, handing `run_cmc` the same six positional
    arguments and the same two keyword ones - `lib_queue` lands on `res_queue`, so the library's
    own merge payload is produced by the library's own code path.

    The Config is built HERE rather than passed in: it pickles fine now, but a worker that builds
    its own document is one fewer thing between the seed and the numbers. The probe payload is what
    the parent cannot otherwise learn - a parent process cannot see a child's CUDA allocation, so
    the device each worker ran on has to be reported by the worker itself.
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
    """One spawn round on CUDA. Both queues are drained BEFORE any join, which is the ordering the
    dispatch itself uses and the only one that cannot deadlock on a full pipe."""
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
    """The spawn changes nothing about the numbers - which is what lets every equality gate above
    run in-process and still describe the real dispatch."""
    here = pooled(DEVICE_COUNT)
    assert [p['sha'] for p in sharded['probes']] == [sha(f.values) for f in here['frames']], (
        'the spawned shards and the in-process shards disagree')


@needs_two_devices
def test_the_results_are_finite_and_correctly_shaped(sharded):
    """Shape, finiteness, and the dispersion guard - a deal whose pricer raised is swallowed and
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
    """`run_cmc`'s own `res_queue` branch, carrying the five keys the parent's merge reads - `Job`
    among them, which is what lets the parent order a race-ordered queue.

    BOTH batch counts report what this worker RAN. They did not agree until `run_cmc` copied the
    post-division figure back onto `params_mc`: `Calculation.execute` rebinds `params` through
    `declared_defaults` before applying `//= num_jobs`, so `Params` carried the document's request
    and a consumer pooling off it over-counted the paths behind each worker by `num_jobs`.
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
    """`Context.Credit_Monte_Carlo(runparallel=2, device='cpu')` - the real spawned dispatch, on a
    device every box has, pooled bit-identically against the in-process n=1 CPU reference.

    This is the gate that needed all of it: the arity, or the child raises TypeError and the parent
    blocks forever; `Config.__getstate__`, or `w.start()` raises before a child exists; per-batch
    seeding, or two workers do not reproduce one; and the `Job` key, or the merge is a race.
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
    """The distribution comparison, and the only tolerance in this file.

    A sharded run and an unsharded one are two DIFFERENT valid path sets over one document - the
    unsharded default still seeds once and consumes that stream sequentially, which is exactly what
    was left untouched - so they agree in distribution and not bitwise. The band is measured, not
    guessed; see the module docstring. Pooling concatenates the `mtm` and re-summarizes, because
    `EE` pools by averaging equal shards but `PFE_<p>` does not.
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
    """The historical stream, pinned by hash. `deterministic_batches` defaults off, so an ordinary
    caller draws exactly what it drew before any of the determinism work landed - this hash was
    taken from the tree before those changes and `tests/test_hn_barrier_cmc.py` is the standing
    regression gate for the same property.
    """
    _, out = shard(0, 1, deterministic=False)
    assert sha(out['Results']['mtm'].values) == '2df61471b2970c5e', (
        'the unsharded path moved: %s' % sha(out['Results']['mtm'].values))
