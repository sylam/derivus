"""Sharding one Credit Monte Carlo across workers, deterministically in their number.

THE ORIGINAL DEFECT. `Credit_Monte_Carlo(runparallel=True)` passed SEVEN positional arguments to
six-parameter `run_cmc`, so every child raised TypeError before running and the parent blocked
forever on `results.get()`. The stray was a `False` in the `job_id` slot. Both spellings are
byte-identical at every commit back to the root: the path had never run once. The CPU end-to-end
gate is the spawned dispatch itself, so the arity, the `Config` pickle and the merge order all
die there.

DETERMINISM IN n IS THE POINT - sharding that changes the answer is a second model. The old scheme
seeded once per WORKER and consumed that stream sequentially, so a batch drew whatever was left
after the batches before it IN ITS PROCESS. Under `runparallel` the seeding is PER BATCH: batch b
seeds from its GLOBAL index, workers take CONTIGUOUS ranges, and the parent merges by worker. The
pooled result is BIT-IDENTICAL across worker counts, asserted by SHA-256 over the pooled `mtm`.

THE PER-BATCH SEED IS MIXED, NOT ADDED. `calculation.batch_seed` is a SplitMix64 round over
`(Random_Seed, b)`: reseeding per batch asks for as many seeds as the job has batches, and
consecutive integers are the weakest input a generator's initialization takes. One mix closes it
for CUDA's Philox and CPU MT19937 alike, spelled once.

THE UNSHARDED DEFAULT IS UNTOUCHED: `deterministic_batches` is off everywhere but the `runparallel`
dispatch, pinned by hash on both device types. A sharded run and an unsharded one are two valid
path sets over one document and do NOT agree bitwise; over 20 seeds their EE means differed by at
most 2.89% (1.85 sd of the 1.56% difference sd), the t=0 row exactly.

WORKER COUNT IS DECOUPLED FROM DEVICE COUNT. `runparallel` is the only knob: `True` is one worker
per visible CUDA device, an int is exactly that many, and worker j lands on
`cuda:(j % device_count)` with the surplus sharing a device. Where there is no CUDA every worker
runs on `cpu` and the same determinism holds, which is why the CPU arm carries no skip marker.
Bit-identity is per device TYPE: `manual_seed` drives different generators, so a CPU shard and a
CUDA shard are different path sets; the two RTX 3090s in this box were measured byte-for-byte
equal.

BOTH STREAMS ARE ANCHORED. A world whose outer path draws quasi-random numbers reads a Sobol
sequence that is reproducible but POSITION-dependent, and position is what sharding moves.
`set_quasi_batch` hands the quasi stream the same global index the generator gets, and the anchored
arm takes batch b's draw from absolute position `1024 + b * sample_size` - where the historical
engine already stands on an unsharded run. An inner Monte Carlo's Sobol rows are the canonical
inner block's, a function of the path count alone, so they need no anchor and an inner-MC book
shards byte for byte too.

Only the LINEAR columns pool by averaging, so everything here pools the `mtm` and re-summarizes -
which is also what makes the equality exact, one reduction over the same columns in the same order.

THIS BOX: two RTX 3090s, GPU 1 driving the display and so behind the 2s TDR watchdog. The document
is sized so no kernel goes near it - 512 paths x 4 batches over an 8-row grid, 0.03-0.04s a run.
"""
import hashlib
import logging
import os
import sys
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
    exposure is one-sided. GBM draws from the torch generator rather than Sobol and a vanilla prices
    in closed form, which is what puts this document inside the determinism boundary.
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
    measure is the NUMBERS and the arithmetic is the same either way; the spawned path is the
    end-to-end tests'. Pooling concatenates the `mtm` and re-summarizes, so the reduction reads
    the same columns in the same order an unsharded run does.
    """
    devices, frames = [], []
    for job_id in range(n):
        device_used, out = shard(job_id, n, device, seed)
        devices.append(device_used)
        frames.append(out['Results']['mtm'])
    matrix = np.concatenate([f.values for f in frames], axis=1)
    return {'devices': devices, 'mtm': matrix, 'frames': frames,
            'profile': derivus.summarize_data(matrix, '95')}


# ------------------------------------------------------------------ deterministic in n, on CPU
# No skip marker on this arm: the determinism is device-agnostic, so it is gated on every box.

def test_cpu_sharding_is_bit_identical_in_the_worker_count():
    """Shard the same job 1, 2 and 4 ways on CPU and the pooled `mtm` is byte-for-byte the same
    matrix - worker j owning `[j*k, (j+1)*k)`, so the shards partition the paths rather than
    repeat them. Equality, not a tolerance.

    Killing mutation: a batch seeded from its index within its own worker rather than the job."""
    one = pooled(1, device='cpu')
    assert one['devices'] == ['cpu']
    reference = sha(one['mtm'])
    assert reference == '268a30bfa45063ec', 'the sharded CPU stream moved: %s' % reference

    for n in (2, 4):
        many = pooled(n, device='cpu')
        assert many['devices'] == ['cpu'] * n
        assert [f.shape[1] for f in many['frames']] == [BATCH_SIZE * SIMULATION_BATCHES // n] * n
        assert sha(many['mtm']) == reference, (
            'cpu n=%d pooled mtm %s against n=1 %s - the shard count moved the answer'
            % (n, sha(many['mtm']), reference))
        assert np.array_equal(many['mtm'], one['mtm'])
        # and the profile the caller actually reads
        assert np.array_equal(many['profile'].values, one['profile'].values)


def hmm_document():
    """A commodity future on a `MarkovHMMSpotModel` - the cheapest world whose OUTER path draws
    quasi-random numbers. `generate` calls `quasi_rng` unconditionally once per batch, the regime
    being the model. The carry factor carries no model entry, so exactly ONE stochastic process
    draws from the Sobol stream.
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
    # ten contracts: `Units` is the future's declared size - every pin below prices ten
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


def test_a_sobol_consuming_world_is_bit_identical_in_the_worker_count():
    """THE COVERAGE GATE: the quasi stream is anchored rather than refused, so a world that draws
    from it shards byte-for-byte. Unanchored, worker 1 of a two-way shard would start its engine at
    position zero and read global batch 2 out of the points batch 0 should have had - and the
    pooled matrix would move with n while every worker still agreed with itself.

    Killing mutation: the quasi stream left unanchored under deterministic batches.
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

    Killing mutation: the block's anchor advancing with the global batch it was first built in -
    a position reintroduced - moves the pooled matrix at n = 2 and 4."""
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


@pytest.mark.parametrize('device,pin', [('cpu', 'cd34f5838cab3d1d'),
                                        pytest.param(None, 'fbf7c38c5c63fc60', marks=needs_cuda)])
def test_the_unsharded_hmm_path_is_untouched(device, pin):
    """The quasi anchoring is reachable ONLY through `set_quasi_batch`, so an ordinary caller keeps
    the free-running engine: pinned by hash on CPU and on the device, a different generator backend
    feeding the same stream. It is NOT the sharded answer - anchoring moves which points a batch
    reads and the per-batch reseed moves the generator, two valid path sets.

    Killing mutation: the per-batch reseed and anchor taken on every run, sharded or not."""
    _, out = derivus.run_cmc(hmm_document(), torch.float32, overrides(), 0, 1, None,
                             deterministic_batches=False, device=device)
    assert sha(out['Results']['mtm'].values) == pin, (
        'the unsharded HMM path moved on %s: %s' % (device, sha(out['Results']['mtm'].values)))
    if device == 'cpu':
        assert pin != sha(hmm_pooled(1, device='cpu'))


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


# ------------------------------------------------------------------ the spawned path, both devices

def _shard_worker(job_id, num_jobs, seed, device, lib_queue, probe_queue):
    """THE CHILD, spawned as the dispatch spawns it: `lib_queue` lands on `res_queue`, so the
    library's own merge payload is produced by its own code path. The probe payload is what the
    parent cannot otherwise learn, a parent being unable to see a child's CUDA allocation.
    """
    calc, out = derivus.run_cmc(job_document(), torch.float32, overrides(seed),
                               job_id, num_jobs, lib_queue,
                               deterministic_batches=True, device=device)
    probe_queue.put({
        'job_id': job_id,
        'device': str(calc.device),
        'allocated_on_own_device': (torch.cuda.memory_allocated(calc.device)
                                    if calc.device.type == 'cuda' else None)})


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
    for p in probes:
        assert p['allocated_on_own_device'] > 0, (
            'worker %d reports no allocation on %s, so nothing ran there'
            % (p['job_id'], p['device']))


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
    assert merged['Reference'] == ['multigpu'] * n
    # BOTH batch counts are what the worker RAN: `Calculation.execute` rebinds `params` through
    # `declared_defaults` before `//= num_jobs`, so `Params` carried the document's request
    for stats, params in zip(merged['Stats'], merged['Params']):
        assert stats['Simulation_Batches'] == params['Simulation_Batches'] == SIMULATION_BATCHES // n
    matrix = np.concatenate([r['mtm'].values for r in merged['Results']], axis=1)
    assert sha(matrix) == sha(reference), (
        'the end-to-end pooled mtm %s does not match the in-process reference %s'
        % (sha(matrix), sha(reference)))
    return matrix


def test_the_cpu_context_path_shards_end_to_end():
    """`Credit_Monte_Carlo(runparallel=2, device='cpu')` - the real spawned dispatch, pooled
    bit-identically against the in-process n=1 CPU reference, past a device count of none. The gate
    that needed all of it: the arity, or the child raises TypeError and the parent blocks forever;
    `Config.__getstate__`, or `start()` raises before a child exists; per-batch seeding, or two
    workers do not reproduce one; the `Job` key, or the merge is a race.

    Killing mutations: the dispatch spawning its workers without `deterministic_batches`; a
    worker's `Params` reporting the document's batch count.
    """
    _check_merged(_end_to_end(2, 'cpu'), 2, pooled(1, device='cpu')['mtm'])


@needs_two_devices
def test_the_cuda_context_path_shards_end_to_end():
    """The same call on the real devices, `runparallel=True` resolving to one worker per device."""
    _check_merged(_end_to_end(True, None), DEVICE_COUNT, pooled(1)['mtm'])


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
    the flat surface's plan and the strip's and float leg's per-batch uploads (+29); and by
    `Calculation_State.upload` copying on every call."""
    once, once_skipped = host_traffic(1, device)
    thrice, thrice_skipped = host_traffic(3, device)
    assert once_skipped == thrice_skipped == 0, 'a deal was skipped, so its reads went uncounted'
    per_batch = (sum(thrice.values()) - sum(once.values())) / 2
    assert per_batch <= HOST_TRAFFIC_PER_BATCH, (
        '%s host reads and uploads a batch on %s against %d, over two batches: %s' % (
            per_batch, device, HOST_TRAFFIC_PER_BATCH, (thrice - once).most_common(12)))


# ------------------------------------------------------------------ what a batch held

class OnASmallCard(calculation.Credit_Monte_Carlo):
    """A credit Monte Carlo that measured a peak past its device, wherever it runs."""

    def device_memory(self):
        return {'Peak_GB': 2.0, 'Device_GB': 1.0}


def test_a_run_reports_what_it_held_and_warns_where_it_passed_the_device(caplog):
    """A credit Monte Carlo on a CUDA device reports its peak reserved memory beside the device's
    under `Stats.Device_Memory`, the peak at or under the device and nothing warned; a peak past
    the device - which the driver pages over the bus rather than refuses - is a warning naming the
    batch; the host reports nothing. Killed by: the stat not written; the warning on every run;
    the host reporting a device."""
    job = job_document()
    calc = OnASmallCard(job, prec=torch.float32, device=torch.device('cpu'))
    stats = {'Batch_Size': BATCH_SIZE}
    with caplog.at_level(logging.WARNING):
        calc.report_device_memory(stats)
    assert stats['Device_Memory'] == {'Peak_GB': 2.0, 'Device_GB': 1.0}
    assert 'Batch_Size %d held 2.00 GB at its peak on a 1.00 GB device' % BATCH_SIZE in caplog.text
    caplog.clear()
    plain = calculation.Credit_Monte_Carlo(job, prec=torch.float32, device=torch.device('cpu'))
    plain.report_device_memory(stats := {})
    assert stats == {} and not caplog.text
    if torch.cuda.is_available():
        _, out = derivus.run_cmc(job_document(), torch.float32, dict(overrides(), Simulation_Batches=1),
                                 device='cuda')
        memory = out['Stats']['Device_Memory']
        assert 0.0 < memory['Peak_GB'] <= memory['Device_GB'], memory
        assert 'held' not in caplog.text
