"""Does the boundary estimate hold still over a range of bandwidths, at the operating point?

`pricing.boundary_weights` says local-linear weights cancel the local-constant estimator's
O(bandwidth) bias "so the estimate holds still over a range of bandwidths", and that is the only
acceptance criterion the docstring offers. `Boundary_AAD_Bandwidth`'s own field note puts the
default 0.01 at "roughly 32768 paths to populate the near-boundary band", and this script reads it
there and at the declared 16384.

THE READING IS THE REPORTED NUMBER. `Boundary_AAD_Bandwidth` is the only JSON knob that reaches the
estimator, so the ladder is run end to end and read off `grad_cva`'s equity-spot entry - what a desk
sees, not an internal coefficient. The correction is isolated at the public seam
`test_swaption_exercise_boundary.py::test_the_registration_moves_the_cva_gradient_and_not_the_cva`
already uses: at 1e-12 the kernel underflows on every scenario, `boundary_weights` lands on its
empty-kernel branch, and the correction contributes an exact zero. Nothing is patched, and the
registration still happens and is still scored.

ONE SUBJECT, the roadmap row's own re-baselined gate, ridden as a fixture rather than rerun as a
test: the monthly discrete down-and-out of `test_boundary_pricer_events.DISCRETE_BARRIER`, whose
correction is 24% of its reported gradient. A second subject under a declared spot model, whose
knocked-out counterfactual is the model-free zeros branch, read 4.7% before the Heston-Nandi
families were retired and has no fixture today.

SEEDS ARE THE YARDSTICK AND THAT IS THE WHOLE METHOD. "Holds still" is meaningless in the absolute:
the estimate is a Monte Carlo functional, so the question is whether moving the bandwidth over a
decade moves it by LESS than re-seeding does. Every ladder is run on `SEEDS` (six), the
per-bandwidth seed spread is the floor, and a plateau is the widest contiguous bandwidth window
whose spread sits under it. Spread is peak-to-peak over the median, the statistic
`crn_ladder.Ladder.flatness` reads, so a bandwidth ladder and a bump ladder are quoted on one scale.

THE SEED FLOOR IS A FLOOR ON PART OF THE NOISE, NOT ALL OF IT. `Random_Seed` moves the
pseudo-random half only: `calculation.CMC_State.quasi_rng` states that its Sobol stream is "a fixed
scrambled Sobol sequence starting at a fixed offset, neither derived from the job's `Random_Seed`",
so every seed here reads the same quasi-random points and the floor is measured against a
re-randomisation of what is left. It is a floor, and the plateau sits an order under it, but it
understates the sampling noise rather than bounding it.

`Recompute_Inner_MC: 'Yes'` is what keeps the inner-MC tape off a 24GB device - 16384 paths at 32
inner sims peaks at 4.30 GB against 15.93 taped, and 128 inner sims taped does not fit at all. It is
a declared switch gated bit-identical in cva, profile, cashflows and the whole gradient on both
these pricers (`tests/test_recompute_inner_mc.py`, `tests/test_recompute_equity_pricers.py`).

THE READING, six seeds, seed MEAN of the isolated correction:

                   0.005..0.04 (x8)   0.005..0.08 (x16)   0.0025..0.08 (x32)   seed floor
    16384 paths          1.98%              1.97%                5.33%           14.36%
    32768 paths          3.36%              3.35%                3.45%           10.64%

and of the REPORTED CVA delta, which is the number a desk sees:

    16384 paths          0.50%              0.50%                1.35%            3.90%
    32768 paths          0.85%              0.85%                0.88%            2.17%

SO IT HOLDS STILL AT BOTH COUNTS: the widest seed-mean window under the averaged floor is
0.0025..0.08 at 16384 and 0.00125..0.08 at 32768 (0.0025..0.08 on the delta), with the declared 0.01
inside it, and the seed floor is what says that means something - the same quantity moves 10% to
14% when nothing but the seed changes. At 16384 the two lowest rungs read 5% and 7% below the rest,
inside two standard errors; at 32768 they do not - the narrow kernel is populated by the path
count, and `BOUNDARY_MAX_AMPLIFICATION` is where a starved two-point local-linear solve lands.

THREE SEEDS CANNOT SEE THIS, AND ONE SEES NOTHING. Seeds 1 to 3 alone read 2.4% over 0.005..0.04 at
16384 and 6.1% at 32768, a single seed 3% to 14% over 0.005..0.08, and the seeds do not agree on
the SIGN of the drift, which is what says the per-seed drift is sampling and not the O(bandwidth)
bias local-linear weights cancel. The plateau is a statement about the estimator, readable only on
the average of six - no spread here is a plateau WIDTH.

THE PATH COUNT IS WHAT BUYS THE PLATEAU, which the same script says by being run under it: at 2048
paths on the barrier, two seeds, the seed-mean correction falls MONOTONICALLY by 23.76% across
0.0025..0.02 and no window of any width holds still to the seed floor - the control that keeps the
reading above from being a property of the statistic rather than of the estimator.

RUN IT IN CHUNKS, and that is not a style choice. A run peaks at 4.3 GB at 16384 paths and 7.8 GB
at 32768, and each further run inside one interpreter holds a run's worth more - 1 GB and 2 GB -
until the caching allocator reaches the whole card and the driver starts paging to host memory.
Three bandwidths per process is inside that budget at 16384 and two at 32768; the readings are
CRN-deterministic in (subject, paths, seed, bandwidth, mcmc), so chunks merge exactly - re-running
one reproduces it to the last digit.

Run:  CUDA_VISIBLE_DEVICES=0 python gates/boundary_bandwidth_plateau.py --seeds 1 --paths 16384 \
          --mcmc 32 --ladder 0.005,0.01,0.02
      ... --seeds 1 --paths 4096      (a cheap smoke of the same ladder)
"""
import argparse
import gc
import logging
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))

import numpy as np
import torch

import derivus
from derivus import utils
from derivus.instruments import construct_instrument

import test_barrier_bridge as bb
import test_boundary_pricer_events as bpe

#: the declared default, and the operating point `Boundary_AAD_Bandwidth`'s own field note names
PATHS = (16384, 32768)
DECLARED = 0.01
#: a factor-64 ladder in steps of 2, so the declared window below sits four rungs wide inside it
LADDER = (0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
#: the factor-8 window the declared bandwidth sits in the middle of
DECLARED_WINDOW = (0.0025, 0.02)
#: the empty-kernel branch, reached through the declared field rather than through a patch
SUPPRESSED = 1e-12
SEEDS = (1, 2, 3, 4, 5, 6)
#: inner OSS sims per scenario - 32 for the discrete barrier, the most the subject fits at these
#: path counts. Common random numbers across the ladder - one seed draws one
#: set of inner paths and every bandwidth reads it - so this sets the level the whole ladder shares
#: and cannot manufacture or hide spread ACROSS it. The estimator's sample count is the outer paths.
MCMC = 32


class SkipWatch(logging.Handler):
    """`Deal.calculate` swallows a pricer's refusal into a CRITICAL log line and carries on, so the
    reason a run produced nothing is only ever in the log record."""

    def __init__(self):
        logging.Handler.__init__(self, logging.CRITICAL)
        self.seen = []

    def emit(self, record):
        self.seen.append(record.getMessage())


def _cva_gradient(config, seed, bandwidth, paths, mcmc, recompute, extra=None, batches=1):
    """The CVA gradient's equity-spot entry, and the CVA it is the derivative of.

    `batches` is TOTAL paths and not the estimator's: `shared.boundary_sets` is cleared per
    simulation batch, so the kernel always sees `paths` samples however many batches are averaged
    over it."""
    overrides = {
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': paths,
        'Simulation_Batches': batches, 'Random_Seed': seed, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD',
        'Gradient_Variables': 'Factors', 'Boundary_AAD_Bandwidth': bandwidth,
        'Recompute_Inner_MC': recompute,
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}}
    overrides.update(extra or {})
    watch = SkipWatch()
    logging.getLogger().addHandler(watch)
    try:
        _, out = derivus.run_cmc(config, prec=bb.DTYPE, overrides=overrides)
    except Exception as exc:
        # the ValueError that surfaces is three layers downstream of the collapsed frame; the
        # pricer's actual refusal only ever reached the log
        raise RuntimeError('the deal was skipped and the run died on the collapsed frame (%s: %s)'
                           '; the refusal was: %s' % (
                               type(exc).__name__, exc, ' | '.join(watch.seen) or 'not logged'))
    finally:
        logging.getLogger().removeHandler(watch)
    if watch.seen:
        raise RuntimeError('the deal was skipped: ' + ' | '.join(watch.seen))
    g = out['Results']['grad_cva']['Gradient']
    rows = [i for i in g.index if 'EquityPrice' in str(i[0])]
    if not rows:
        raise RuntimeError('the run reported no EquityPrice sensitivity, so there is no estimate '
                           'to read - the gradient support is empty')
    reading = float(out['Results']['cva']), float(g.loc[rows[0]])
    # an autograd graph is a reference CYCLE, so the run's tape is unreachable but not yet
    # collected and `empty_cache` has nothing to hand back. Without the collect this loop climbs
    # from 4.3 GB to the whole device inside twenty runs and starts paging to host memory.
    del out, g
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reading


def _with_counterparty(config):
    """A counterparty is what gives the barrier an exposure profile, which is where the touch state
    accumulates and the only place the correction has anything to correct."""
    config.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    return config


def discrete_barrier():
    """`test_boundary_pricer_events.DISCRETE_BARRIER` - the monthly down-and-out whose already-hit
    latch is the correction's own subject, uncollateralised."""
    c = _with_counterparty(bb._cfg())
    c.deals['Deals']['Children'] = [
        {'Instrument': construct_instrument(bpe.DISCRETE_BARRIER, {})}]
    return c, {'Generate_Cashflows': 'Yes'}


SUBJECTS = {'discrete': discrete_barrier}


def spread(values):
    """Peak-to-peak relative to the median."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.ptp(v) / max(abs(np.median(v)), 1e-30))


def window_spread(rungs, values, lo, hi):
    """Spread over the rungs between `lo` and `hi` inclusive."""
    inside = [v for r, v in zip(rungs, values) if lo <= r <= hi]
    return spread(inside) if len(inside) > 1 else float('nan')


def trend(rungs, values, lo, hi):
    """``(end-to-end relative change, monotone)`` over the rungs between `lo` and `hi`.

    This is what tells the two failure modes apart, and it is the same distinction
    `tests/crn_ladder.py` draws between agreement and flatness: a MONOTONE march across the window
    is the O(bandwidth) bias local-linear weights are supposed to have cancelled, while scatter of
    the same size is Monte Carlo noise and says nothing about the estimator."""
    inside = [v for r, v in zip(rungs, values) if lo <= r <= hi]
    if len(inside) < 2:
        return float('nan'), False
    d = np.diff(inside)
    return ((inside[-1] - inside[0]) / max(abs(inside[0]), 1e-30),
            bool(np.all(d > 0) or np.all(d < 0)))


def widest_plateau(rungs, values, tol):
    """The widest contiguous bandwidth window whose spread stays under `tol`, as
    ``(lo, hi, factor, spread)`` - or None if no two adjacent rungs do."""
    best = None
    for i in range(len(rungs)):
        for j in range(i + 2, len(rungs) + 1):
            s = spread(values[i:j])
            if s <= tol and (best is None or rungs[j - 1] / rungs[i] > best[2]):
                best = (rungs[i], rungs[j - 1], rungs[j - 1] / rungs[i], s)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subjects', default='discrete')
    ap.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    ap.add_argument('--paths', default=','.join(str(p) for p in PATHS))
    ap.add_argument('--mcmc', type=int, default=MCMC)
    ap.add_argument('--recompute', default='Yes', choices=['Yes', 'No'])
    ap.add_argument('--batches', type=int, default=1,
                    help='simulation batches: TOTAL paths, never the estimator sample count')
    ap.add_argument('--ladder', default=','.join(repr(b) for b in LADDER))
    ap.add_argument('--memory-fraction', type=float, default=0.65,
                    help='cap the caching allocator; 0 leaves it alone')
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)     # tests/conftest.py's hygiene, outside pytest
    if torch.cuda.is_available() and args.memory_fraction:
        # a run's peak is 4.3 GB and the caching allocator's RESERVE still walks to the whole card
        # inside one ladder; capping it makes the allocator recycle rather than reach for the
        # host-memory fallback, which is what turns a 3-second run into a 45-second one
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    seeds = [int(s) for s in args.seeds.split(',')]
    paths_list = [int(p) for p in args.paths.split(',')]
    ladder = [float(b) for b in args.ladder.split(',')]
    names = [s.strip() for s in args.subjects.split(',')]
    print('mcmc %d   recompute %s   batches %d   seeds %s   declared bandwidth %g' % (
        args.mcmc, args.recompute, args.batches, seeds, DECLARED))
    print('ladder %s   suppressed at %g (empty-kernel branch)\n' % (ladder, SUPPRESSED))

    readings = {}
    for paths in paths_list:
        for name in names:
            extra = SUBJECTS[name]()[1]
            for seed in seeds:
                t = time.time()
                base_cva, base = _cva_gradient(SUBJECTS[name]()[0], seed, SUPPRESSED, paths,
                                               args.mcmc, args.recompute, extra, args.batches)
                print('%-8s %6d paths seed %d   suppressed %+.9e   cva %.10f   (%.0fs)' % (
                    name, paths, seed, base, base_cva, time.time() - t), flush=True)
                grads, cvas = [], []
                for b in ladder:
                    t = time.time()
                    cva, g = _cva_gradient(SUBJECTS[name]()[0], seed, b, paths, args.mcmc,
                                           args.recompute, extra, args.batches)
                    grads.append(g)
                    cvas.append(cva)
                    print('    bandwidth %8.6f   gradient %+.9e   correction %+.6e  %7.2f%% of '
                          'it   (%.0fs)' % (
                              b, g, g - base, 100.0 * (g - base) / max(abs(g), 1e-30),
                              time.time() - t), flush=True)
                readings[(paths, name, seed)] = (base, base_cva, grads, cvas)

    print('\n%s\nPLATEAU\n%s' % ('=' * 78, '=' * 78))
    for paths in paths_list:
        for name in names:
            rows = [readings[(paths, name, s)] for s in seeds if (paths, name, s) in readings]
            if not rows:
                continue
            print('\n%s at %d paths' % (name, paths))
            moved = [(b, c) for b, c in zip(ladder, rows[0][3]) if c != rows[0][1]]
            print('  the reported CVA across the whole ladder: %s' % (
                'bit-identical at %.17g' % rows[0][1] if not moved else
                'MOVED, so the ladder is not comparing one number: %s' % moved))
            seed_floor = [spread([r[2][i] for r in rows]) for i in range(len(ladder))]
            corr_floor = [spread([r[2][i] - r[0] for r in rows]) for i in range(len(ladder))]
            print('  %-11s %-18s %-18s %-14s' % (
                'bandwidth', 'grad seed spread', 'corr seed spread', 'corr / grad'))
            for i, b in enumerate(ladder):
                share = float(np.mean([(r[2][i] - r[0]) / max(abs(r[2][i]), 1e-30) for r in rows]))
                print('  %-11.6f %-18s %-18s %-14s' % (
                    b, '%.2f%%' % (100.0 * seed_floor[i]), '%.2f%%' % (100.0 * corr_floor[i]),
                    '%.2f%%' % (100.0 * share)))
            floor = float(np.median(seed_floor))
            print('  seed floor (median grad seed spread over the ladder): %.2f%%'
                  % (100.0 * floor))
            for label, series in (('gradient', lambda r: r[2]),
                                  ('correction', lambda r: [g - r[0] for g in r[2]])):
                for seed, r in zip(seeds, rows):
                    win = widest_plateau(ladder, series(r), floor)
                    drift, mono = trend(ladder, series(r), *DECLARED_WINDOW)
                    print('  %-10s seed %d  declared window %g..%g: %.2f%% spread, %+.2f%% '
                          'end-to-end%s   widest: %s' % (
                              label, seed, DECLARED_WINDOW[0], DECLARED_WINDOW[1],
                              100.0 * window_spread(ladder, series(r), *DECLARED_WINDOW),
                              100.0 * drift, ' MONOTONE' if mono else '',
                              'none - no two adjacent rungs hold still to the seed floor'
                              if win is None else '%g..%g (factor %g) at %.2f%%' % (
                                  win[0], win[1], win[2], 100.0 * win[3])))
                print('  %-10s full ladder spread per seed: %s' % (
                    label, ', '.join('%.2f%%' % (100.0 * spread(series(r))) for r in rows)))
                # the seed MEAN carries 1/sqrt(len(seeds)) of one seed's noise, so what survives
                # averaging is the estimator's bandwidth dependence and not the sampling
                mean = [float(np.mean([series(r)[i] for r in rows])) for i in range(len(ladder))]
                drift, mono = trend(ladder, mean, *DECLARED_WINDOW)
                win = widest_plateau(ladder, mean, floor / np.sqrt(len(rows)))
                print('  %-10s seed MEAN  %s' % (
                    label, '  '.join('%.4g' % v for v in mean)))
                print('  %-10s seed MEAN  declared window %.2f%% spread, %+.2f%% end-to-end%s   '
                      'widest under the averaged floor (%.2f%%): %s' % (
                          label, 100.0 * window_spread(ladder, mean, *DECLARED_WINDOW),
                          100.0 * drift, ' MONOTONE' if mono else '',
                          100.0 * floor / np.sqrt(len(rows)),
                          'none' if win is None else '%g..%g (factor %g) at %.2f%%' % (
                              win[0], win[1], win[2], 100.0 * win[3])))


if __name__ == '__main__':
    main()
