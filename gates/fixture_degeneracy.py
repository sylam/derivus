"""FIXTURE DEGENERACY CENSUS - a fixture must not zero the quantity its gate is sensitive to.

WHY. A +1432% error in the already-hit barrier leg survived a green suite because THREE
independent fixture degeneracies each hid it on their own: every fixture set ``r = q = 0`` (which
zeroed the missing-``dt`` half of the arithmetic), every HN barrier gate ran BASE VALUATION (one
MTM row, so the ``all_hit`` mask is all-False at row 0 and the leg is never executed), and the one
barrier on an exposure grid was ``Down_And_Out`` (whose leg is the model-free zeros branch). None
of the three is a wrong assertion. Each is a fixture that zeroed the quantity its gate was
supposed to be sensitive to, and prose alone has now failed to catch this twice.

WHAT THIS IS. A pytest plugin that OBSERVES a real run rather than parsing test source: it rebinds
``run_baseval`` / ``run_cmc`` / ``run_hedgemontecarlo`` (which is every entry point - ``Context``'s
methods reach them as module globals) and for every call records what the ENGINE actually saw -
the market data, the deal fields, the resulting time grid - as a vector of degeneracy axes. Rows
are attributed to the DEAL TYPES priced in that run, so there is no hand-maintained module->family
map and a new deal type appears in the table the day its first test runs.

THREE KINDS OF AXIS.
  MARKET axes ask "was this quantity zeroed?" of the run (rates, carry, smile, correlation, the
      number of MTM rows). BLIND when no run of that deal type ever made it non-zero.
  DEAL axes ask the same of ONE deal (rebate, position sign), so a mixed-portfolio run cannot
      charge a forward for the barrier's rebate sitting next to it.
  VARY axes ask "which value did it take?" BLIND when the family only ever observed ONE -
      the ``Down_And_Out``-only column is exactly this.

APPLICABILITY IS TRACKED. An axis returns ``None`` where the quantity does not exist (no vol
surface, no barrier field), so a deal type is never charged for a knob it does not have. An axis
applicable somewhere and non-degenerate NOWHERE is the finding.

MARGINAL COVERAGE IS NOT JOINT COVERAGE, which is what ``JOINT`` is for and is the sharp end: the
barrier defect needed a grid AND a knock-IN AND a live carry in ONE run, and the suite had each of
the three separately. Verified against the row set with the fix's own gates removed - the
requirement is UNMET there and met on this tree.

USE
    python gates/fixture_degeneracy.py tests/...    # self-driving; --degeneracy-json PATH
    ... --degeneracy-strict                         # also fail on undeclared blind cells
By default only an unmet ``JOINT`` or a STALE exemption fails - the blind-spot table is a report
until someone triages it into ``ACCEPTED``, because 45 red cells on day one is a gate nobody reads.
Measured on the 14 barrier/option/HN/TARF/autocall modules: 6:39 without, 7:12 with, on a shared
GPU. It rides an existing suite invocation rather than being a run of its own.

WHAT IT CANNOT DO. It reads FIXTURES, never assertions: a gate whose fixture varies everything and
asserts nothing reads clean. Placebo detection is a separate discipline (mutate, then verify).

MAINTENANCE. The axis list is the whole surface - a dozen extractors of a few lines each, reading
fields the engine already reads. ``ACCEPTED`` is the declared-blind-spot list; every entry needs a
reason, and an entry that stops being blind FAILS, because a stale exemption is its own placebo.
"""
import json
import os
import tempfile
from collections import defaultdict

import numpy as np

# ======================================================================================
# declared blind spots: (deal_type, axis) -> reason. Kept SHORT on purpose.
# An entry that is no longer blind fails the census - exemptions do not get to go stale.
# ======================================================================================
ACCEPTED = {}

# ======================================================================================
# JOINT requirements - the sharp end. Marginal coverage is NOT joint coverage: the barrier
# defect needed a grid AND a knock-IN in the SAME run, and the suite had each separately
# (test_barrier_bridge ran a grid, but Down_And_Out; the HN gates ran Up_And_In, but base
# valuation). Each entry names one pricer branch and the conjunction that reaches it.
#   (deal_type, {axis: required_value, ...}, why)
# ======================================================================================
JOINT = [
    ('EquityBarrierOption', {'single_mtm_row': False, 'barrier_dir': 'In', 'carry_zero': False},
     'the already-hit vanilla leg. All THREE at once: a post-t0 row or all_hit is empty and the '
     'leg never executes; a knock-IN or the leg is the model-free zeros branch; a live carry or '
     'the missing-dt drift has nothing to multiply. Two of three was the pre-fix state and it '
     'shipped a +1432% mark - VERIFIED unmet on the HEAD~1 row set, met on HEAD.'),
    ('QEDI_CustomAutoCallSwap', {'single_mtm_row': False, 'carry_zero': False},
     'the autocall is the OTHER adopter of the same OSS seam and its called/knocked state is the '
     'same shape of outer path-state override. It has never been priced at a non-zero carry.'),
]

_VOL_FACTORS = ('VolatilityGrid', 'FXVol', 'SurfaceVol', 'ForwardPriceVol')


def _values(block, key='Curve'):
    """Every numeric value carried by a price-factor curve/surface block, flat."""
    c = block.get(key)
    a = getattr(c, 'array', c)
    if a is None:
        return np.array([])
    return np.asarray(a, dtype=np.float64).reshape(-1, np.asarray(a).shape[-1])[:, -1]


def _factors(pf, kind):
    return {k: v for k, v in pf.items() if k.split('.')[0] == kind}


# ======================================================================================
# MARKET axes - run-level quantities a fixture can set to nothing
# ======================================================================================

def _ax_rates_zero(pf, calc):
    """Every discount/forecast rate in the market data is identically zero, so anything that
    multiplies a rate - a carry, a discount factor, a missing `dt` on a rate - reads as correct."""
    v = np.concatenate([_values(b) for b in _factors(pf, 'InterestRate').values()] or [np.array([])])
    return None if not v.size else bool(np.all(v == 0.0))


def _ax_carry_zero(pf, calc):
    """r - q == 0 for every equity in the market data. This is the exact hole the barrier defect
    lived in: the missing-dt drift term is proportional to the carry, so r == q shrinks a +1432%
    error to +17.5%."""
    out = []
    for name, eq in _factors(pf, 'EquityPrice').items():
        u = name.split('.', 1)[1]
        r, q = pf.get('InterestRate.{}'.format(eq.get('Interest_Rate'))), pf.get('DividendRate.' + u)
        if r is None or q is None:
            continue
        rv, qv = _values(r), _values(q)
        out.append(rv.size and qv.size and np.allclose(rv.mean(), qv.mean()))
    return None if not out else bool(all(out))


def _ax_vol_flat(pf, calc):
    """Every vol surface is a single number - no smile, no term structure. A pricer that reads the
    wrong moneyness or the wrong expiry slice is indistinguishable from one that reads the right
    one."""
    v = [_values(b, 'Surface') for k in _VOL_FACTORS for b in _factors(pf, k).values()]
    v = np.concatenate(v) if v else np.array([])
    return None if not v.size else bool(np.unique(np.round(v, 12)).size == 1)


def _ax_corr_zero(pf, calc):
    """Every declared correlation is zero, so any cross-term (quanto carry, multi-factor
    diffusion) is unobservable."""
    c = [float(b.get('Value', 0.0)) for b in _factors(pf, 'Correlation').values()]
    return None if not c else bool(all(x == 0.0 for x in c))


def _ax_single_mtm_row(pf, calc):
    """The run has ONE time row (base valuation). Every path-state override that only switches on
    after t0 - the already-hit barrier leg, an exercised swaption, a knocked TARF - is never
    reached, so its arithmetic is not asserted, it is not EXECUTED."""
    tg = getattr(calc, 'time_grid', None)
    d = getattr(tg, 'mtm_dates', None)
    return None if d is None else bool(len(d) <= 1)


def _ax_one_netting_set(pf, calc):
    """One netting set, so every cross-set term - collateral thresholds, the boundary correction's
    scoping, aggregation itself - is a no-op that cannot be wrong."""
    n = len(getattr(calc, 'netting_sets', None).sub_structures) if hasattr(
        getattr(calc, 'netting_sets', None), 'sub_structures') else None
    return None if n is None else bool(n <= 1)


def _ax_one_batch(pf, calc):
    """A single simulation batch. Anything the engine folds ACROSS batches - a running statistic,
    a bundle, a per-batch seed stride - never executes its fold."""
    b = getattr(calc, 'params', {}).get('Simulation_Batches')
    return None if b is None else bool(int(b) <= 1)


MARKET_AXES = {
    'rates_zero': _ax_rates_zero,
    'carry_zero': _ax_carry_zero,
    'vol_flat': _ax_vol_flat,
    'corr_zero': _ax_corr_zero,
    'single_mtm_row': _ax_single_mtm_row,
    'one_netting_set': _ax_one_netting_set,
    'one_batch': _ax_one_batch,
}

# ======================================================================================
# DEAL-LEVEL ZERO axes - same "blind when never non-zero" semantics, but read off ONE deal so a
# mixed-portfolio run cannot charge a forward for the barrier's rebate sitting beside it.
# ======================================================================================

def _dz_rebate(d):
    """Cash_Rebate is zero, so the knock branch pays nothing and a wrong knock time or a wrong
    discount on it costs nothing."""
    return None if 'Cash_Rebate' not in d else float(d['Cash_Rebate'] or 0.0) == 0.0


def _dz_units_sign(d):
    """Position is long. A sign error in a leg, or a max/min that should be a min/max, needs the
    other sign to show. (Buy_Sell is the other half of this and is a VARY axis.)"""
    u = d.get('Units')
    return None if u is None else float(u) > 0


DEAL_AXES = {'rebate_zero': _dz_rebate, 'long_only': _dz_units_sign}
ZERO_AXES = dict(MARKET_AXES, **DEAL_AXES)

# ======================================================================================
# VARY axes - a deal field whose family must span more than one value
# ======================================================================================

def _vx_barrier_dir(d):
    b = d.get('Barrier_Type')
    return None if not b else ('In' if str(b).endswith('_In') else 'Out')


def _vx_barrier_side(d):
    b = d.get('Barrier_Type')
    return None if not b else str(b).split('_')[0]


VARY_AXES = {
    'buy_sell': lambda d: d.get('Buy_Sell'),
    'option_type': lambda d: d.get('Option_Type'),
    'barrier_dir': _vx_barrier_dir,
    'barrier_side': _vx_barrier_side,
    'payoff_type': lambda d: d.get('Payoff_Type') or ('Plain' if 'Payoff_Currency' in d else None),
}

AXES = list(ZERO_AXES) + list(VARY_AXES)

# ======================================================================================
# observation
# ======================================================================================

_ROWS = []
_NODE = ['<setup>']


def observe(context, calc):
    """One row per calculation run: the axis vector, attributed to the deal types priced."""
    deals = [i.field for i in context.walk_deals()]
    pf = context.params.get('Price Factors', {})
    zero = {k: f(pf, calc) for k, f in MARKET_AXES.items()}
    vary = defaultdict(lambda: defaultdict(set))
    dzero = defaultdict(lambda: defaultdict(set))
    for d in deals:
        t = d.get('Object')
        if t is None:
            continue
        vary[t]                                         # a type with no VARY field still gets a row
        for k, f in VARY_AXES.items():
            v = f(d)
            if v is not None:
                vary[t][k].add(str(v))
        for k, f in DEAL_AXES.items():
            v = f(d)
            if v is not None:
                dzero[t][k].add(bool(v))
    _ROWS.append({'test': _NODE[0], 'zero': zero,
                  'vary': {t: {k: sorted(v) for k, v in ax.items()} for t, ax in vary.items()},
                  'dzero': {t: {k: sorted(v) for k, v in ax.items()} for t, ax in dzero.items()}})


def _census():
    """Fold the rows into deal_type x axis. Value is 'blind' / 'ok' / '-' (not applicable)."""
    zero_seen = defaultdict(lambda: defaultdict(set))   # type -> axis -> {True/False}
    vary_seen = defaultdict(lambda: defaultdict(set))   # type -> axis -> {values}
    tests = defaultdict(set)
    for row in _ROWS:
        for t in row['vary']:
            tests[t].add(row['test'])
            for k, v in row['zero'].items():
                if v is not None:
                    zero_seen[t][k].add(v)
            for k, vals in row.get('dzero', {}).get(t, {}).items():
                zero_seen[t][k].update(vals)
            for k, vals in row['vary'].get(t, {}).items():
                vary_seen[t][k].update(vals)
    table = {}
    for t in sorted(set(zero_seen) | set(vary_seen)):
        cells = {}
        for k in ZERO_AXES:
            s = zero_seen[t].get(k)
            cells[k] = '-' if not s else ('blind' if s == {True} else 'ok')
        for k in VARY_AXES:
            s = vary_seen[t].get(k)
            cells[k] = '-' if not s else ('blind:' + list(s)[0] if len(s) == 1 else 'ok')
        table[t] = {'cells': cells, 'n_tests': len(tests[t]), 'n_runs':
                    sum(1 for r in _ROWS if t in r['vary'])}
    return table


def render(table):
    w = max([len(t) for t in table] + [9])
    head = '{:<{w}}  {:>5} {}'.format('deal type', 'runs', ' '.join(
        '{:>14}'.format(a[:14]) for a in AXES), w=w)
    out = [head, '-' * len(head)]
    for t, r in table.items():
        out.append('{:<{w}}  {:>5} {}'.format(t, r['n_runs'], ' '.join(
            '{:>14}'.format(r['cells'][a]) for a in AXES), w=w))
    return '\n'.join(out)


def blind_spots(table):
    return sorted((t, a) for t, r in table.items() for a in AXES if r['cells'][a].startswith('blind'))


def unmet_joints():
    """A joint requirement is met when ONE observed run satisfies every condition at once.

    A deal type absent from this selection is NOT a finding - that is a scoped run, not a blind
    fixture - so it is skipped rather than failed.
    """
    out = []
    seen = {t for row in _ROWS for t in row['vary']}
    for deal, cond, why in JOINT:
        if deal not in seen:
            continue
        hit = any(
            deal in row['vary']
            and all((row['zero'].get(k) == v if k in MARKET_AXES else
                     [v] == row.get('dzero', {}).get(deal, {}).get(k) if k in DEAL_AXES else
                     v in row['vary'][deal].get(k, []))
                    for k, v in cond.items())
            for row in _ROWS)
        if not hit:
            out.append((deal, cond, why))
    return out


# ======================================================================================
# pytest plugin
# ======================================================================================

def pytest_addoption(parser):
    g = parser.getgroup('degeneracy')
    g.addoption('--degeneracy-census', action='store_true', help='run the fixture degeneracy census')
    g.addoption('--degeneracy-strict', action='store_true',
                help='also fail on undeclared blind cells, not only unmet JOINT requirements')
    # the raw rows are a regenerable artifact, so they default OUTSIDE the repo
    g.addoption('--degeneracy-json',
                default=os.path.join(tempfile.gettempdir(), 'derivus_degeneracy_census.json'))


def pytest_configure(config):
    if not config.getoption('degeneracy_census'):
        return
    import derivus

    def wrap(fn):
        def inner(context, *a, **kw):
            out = fn(context, *a, **kw)
            try:
                observe(context, out[0] if isinstance(out, tuple) else None)
            except Exception as e:                      # a census must never fail a real gate
                _ROWS.append({'test': _NODE[0], 'zero': {}, 'vary': {}, 'error': repr(e)})
            return out
        return inner

    # Context.run_job / Base_Valuation / Credit_Monte_Carlo reach these as module globals, so
    # rebinding them here covers the Context entry points too.
    for name in ('run_baseval', 'run_cmc', 'run_hedgemontecarlo'):
        setattr(derivus, name, wrap(getattr(derivus, name)))


def pytest_runtest_setup(item):
    _NODE[0] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption('degeneracy_census'):
        return
    table = _census()
    path = session.config.getoption('degeneracy_json')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'rows': _ROWS, 'table': table}, f, indent=1, default=str)
    bad = [b for b in blind_spots(table) if b not in ACCEPTED]
    print('\n\nFIXTURE DEGENERACY CENSUS  ({} runs observed, rows -> {})\n'.format(len(_ROWS), path))
    print(render(table))
    print('\nblind spots: {} undeclared, {} accepted'.format(len(bad), len(ACCEPTED)))
    for t, a in bad:
        print('  BLIND  {:<28} {}'.format(t, a))
    stale = [k for k in ACCEPTED if k not in blind_spots(table)]
    for k in stale:
        print('  STALE EXEMPTION  {} {}'.format(*k))
    joints = unmet_joints()
    seen = {t for row in _ROWS for t in row['vary']}
    print('\nunmet joint requirements: {} of {} ({} not exercised by this selection)'.format(
        len(joints), len(JOINT), sum(1 for d, _, _ in JOINT if d not in seen)))
    for deal, cond, why in joints:
        print('  UNMET  {:<24} {}\n         {}'.format(deal, cond, why))
    # The blind-spot table is a REPORT until someone triages it into ACCEPTED - 45 red cells on day
    # one is a gate nobody reads. What ENFORCES by default is the short, hand-declared list: an
    # unmet JOINT requirement, and an exemption that has gone stale.
    if joints or stale or (bad and session.config.getoption('degeneracy_strict')):
        session.exitstatus = 1


if __name__ == '__main__':
    import sys

    import pytest
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = sys.argv[1:] or ['tests/']
    raise SystemExit(pytest.main(['-q', '--degeneracy-census'] + args, plugins=[sys.modules[__name__]]))
