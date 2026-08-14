"""`CommodityAveragePriceSwapDeal` - the platinum campaign's liability, priced closed form on the
path against a spot the world composes (`CommodityPrice.PLATINUM_CME` + `ObservedBasis…LBMA`) and a
carry curve the world simulates (`QuadraticCarryCurveModel.PLATINUM_CARRY`).

THE MID-LIFE MANDATE. The campaign's standing lesson is the already-hit barrier leg: an
inception-only fixture leaves the lookback term identically zero, so the gate runs, passes, and
measures nothing. Every gate here therefore prices a world with **two fixings already declared,
four still to come, and reporting rows on both sides of every one of them** - `tests/fixtures/
commodity_aps_world.json`, whose 16-row grid realises fixings at rows 3, 6, 9 and 12. Gate 2 is
stated per PATH at rows with 2, 3, 4, 5 and 6 fixings realised, which is the only shape that can
tell a lookback reading the right rows from one reading its neighbours.

WHAT EACH GATE HOLDS

  1. INCEPTION, closed form, exact. `V_0 = D(0,T_pay) N w (sum_j w_j S~_j - K)` rebuilt from the
     market data alone - the carry's own closed form, the repo integral, and the basis AR - on a
     genuine inception (no fixing before the base date) AND on the shipped world (two of them),
     both directions of `Buy_Sell`.
  2. MID-LIFE, per path, exact. The reported row equals realised-half + projected-half rebuilt
     from the CAPTURED scenario paths, with the slow mean re-derived from the basis path by the
     recursion the model documents rather than read back from it. This is the gate that pins every
     read: the fixing row, the decay exponent, the composition, the weights.
  3. TERMINAL + SETTLE-ONCE. Past the last fixing the value is `D (A - K)` with `A` entirely
     realised, and the cashflow is booked exactly once, at the settlement row, at its undiscounted
     amount.
  4. MARTINGALE / two-clock. With the basis extension off and a ZERO carry and repo the projected
     average is a martingale and `E[A_t] = A_0` within MC error; with the carry LIVE the same test
     fails by a stated number of standard errors, which is the E[dF] != 0 statement made checkable
     rather than assumed.
  5. AAD. `Greeks='First'` on a base valuation: the gradient's factor set is exactly
     {spot, basis, both carry knots, repo, discount} - no vol node, no model-parameter block, no
     FX - and every entry is finite.
  6. CMC end to end through `run_job`, collateral off: the exposure profile is not flat, its
     dispersion RISES while fixings remain, and past the last one it moves by exactly the discount
     - the shape an average of conditional expectations has, which a placebo does not reproduce.
  7. MUTATIONS - the matrix below, every one run.
  8. SCHEMA - the declaration, the cross-field rule, and the create menu.

ANTI-PLACEBO - the fixture property each gate needs, and what goes blind without it.

| property | value | what goes blind without it |
|---|---|---|
| fixings on both sides of the base date | 2 past (declared), 4 future | at zero past fixings the whole `known_resets` half is unexecuted; at zero future ones the projection is |
| reporting rows between fixings | 16 rows, 2-weekly, fixings at 4 of them | with a row only at each fixing, `past` is never a strict subset in the middle of the schedule and an off-by-one lookback lands on a row that happens to hold the same value |
| the scenario grid is NOT calendar-daily | 2-weekly + fixing dates | at one step per calendar day `phi^steps == phi^days` and the decay-clock mutant is a NO-OP |
| basis level | 9.56 on a 1661 spot | a zero basis makes the dropped-composition mutant invisible; it is 0.58% of the spot here, ~50x the MC noise on a t0 mark |
| `Mu_0` != the basis `Spot` | 6.25 vs 9.56 | at `Mu_0 == b_0` the AR term `phi^n (b_t - mu_t)` is identically zero and every decay mutant survives (measured: the two agree to 0 ULP at row 0) |
| weights | 1,2,3,1,2,3 - not equal | equal weights hide a weight/fixing misalignment, and make the `sum w_j S_j` half of the K-placement question degenerate |
| carry | two knots, 0.0191 -> 0.0105, sloped | a flat carry is affine-degenerate: the knot pair stops being identified and a wrong knot read cancels |
| repo != discount != 0 | 1.2-1.9% vs 3.5-4.1% | a shared curve cannot show the forward reading the primary's repo and the payoff reading the deal's discount; a zero one kills both |
| `Buy` and `Sell` | both run | a one-sided fixture cannot see a sign folded into the wrong half |

MUTATION MATRIX - every one RUN against every gate in this file, by exec'ing a one-token edit of
`pricing.pv_average_price_swap`'s own source onto the module (so the deal's `generate` picks it up
unchanged). Control: twelve gates, zero failures.

| mutant | killed by | count |
|---|---|---|
| the lookback reads scenario row `+1` | mid-life, seam, terminal, settled amount, profile | 5 |
| realised fixings re-read from the CURRENT row's spot | inception x2, baseval gap, mid-life, seam, terminal, settled amount, profile | 8 |
| `E[b]` decays in CALENDAR DAYS instead of simulation steps | inception x3, mid-life, seam | 5 |
| the realised read drops the basis component (primary alone) | mid-life, seam, terminal, settled amount, profile | 5 |
| the PROJECTED leg puts the composed spot under the carry exponential | inception x3, baseval gap, mid-life, seam, profile | 7 |
| `K` charged per fixing without its weight (`A - n K`) | everything but the two martingale arms and the greek set | 9 |
| `K` applied INSIDE the weights (`sum_j w_j (S_j - K)`) | nothing - a reported NO-OP, below | 0 |
| the settlement fires per reporting row | the settle-once spy ONLY, below | 1 |

WHAT THE MATRIX SAYS OUT LOUD, and it is not what the gate names suggest.

**Inception cannot see either lookback mutant.** At row 0 the only realised fixings are the two
DECLARED ones, which come out of `known_resets` rather than out of the buffer - so shifting the
scenario row, or dropping the basis from the composed read, moves nothing at t=0. Both die on the
mid-life rows and nowhere earlier. That is the mid-life mandate stated as a measurement rather than
as a principle: a file with only gate 1 in it would have shipped both defects.

**Neither martingale arm kills anything**, and that is correct rather than a weak gate. Its world
is zero-carry, extension-off and 8192 paths: the decay term is inert there by construction, and a
lookback shifted by one row still reads a martingale, so its mean is unmoved. An MC gate on a mean
cannot see a per-path defect that is mean-preserving - which is exactly why gate 2 is stated per
PATH and exactly.

**The greek gate kills nothing either.** It asserts the factor SET, and every mutant here is
arithmetic on the same factors. It is aimed at a different failure - a factor appearing that the
payoff has no business reading - and a matrix that showed it killing these would mean it was
measuring something else.

TWO MUTANTS THAT ARE NOT WHAT THEY LOOK LIKE, both run and both reported.

`K` applied BEFORE the weights is a NO-OP, and provably so: `make_sampling_data` divides every
weight by their sum, so the weights sum to one by construction and `sum_j w_j (S_j - K)` IS
`sum_j w_j S_j - K`. Zero kills out of twelve gates, measured. It is run and asserted
BIT-IDENTICAL rather than left out - if the weights ever stopped summing to one, that test is what
fails - and the neighbouring spelling that is genuinely different (`K` charged once per fixing) is
the one carried in the matrix, at nine kills.

The settlement fired per reporting row changes NO reported number either, for a different reason:
`cash_settle` books only into an index the currency map holds, and `reset` registers exactly one
settlement date, so fourteen of the fifteen calls fall on the floor and the fifteenth is the row
the correct code settles at, with the same value. Only a spy on the call COUNT can see it. That is
recorded rather than papered over - the gate is on the discipline, and the number it protects is
gated separately (the settled amount against the closed form).

A THIRD THING THE SCORING RUN FOUND, in the harness rather than the pricer: a mutant exec'd into a
COPY of the module dict freezes every name it calls, so a `cash_settle` spy installed afterwards is
invisible to it and every mutant reads as caught by the settle-once gate. `_mutate` therefore execs
into `pricing.__dict__` itself. The first scoring pass scored six false kills that way.
"""
import inspect
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus as rf
from derivus import instruments, pricing, schema, utils

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'commodity_aps_world.json')
DEAL_TYPE = 'CommodityAveragePriceSwapDeal'


# ---------------------------------------------------------------------------
# the world, and the variants that are overrides ON it rather than copies OF it
# ---------------------------------------------------------------------------

def _load():
    return json.load(open(FIXTURE))


def _parts(cfg):
    """`(Calculation, the deal block, Price Factors, Price Models)` of a loaded world."""
    calc = cfg['Calc']['Calculation']
    md = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
    deal = cfg['Calc']['Deals']['Deals']['Children'][0]['Children'][0]['Instrument']['.Deal']
    return calc, deal, md['Price Factors'], md['Price Models']


def world(buy='Buy', inception=False, extensions=True, zero_carry=False, carry_scale=1.0, **calc):
    """The shipped world, plus the overrides the gates need. The FILE is never edited."""
    cfg = _load()
    calculation, deal, factors, models = _parts(cfg)
    calculation.update(calc)
    if carry_scale != 1.0:
        for knot in factors['ForwardRate.PLATINUM_CARRY']['Curve']['.Curve']['data']:
            knot[1] *= carry_scale
    deal['Buy_Sell'] = buy
    if inception:
        # a genuine inception: keep only the fixings after the base date, so nothing is realised
        deal['Sampling_Data'] = [r for r in deal['Sampling_Data']
                                 if pd.Timestamp(r[0]['.Timestamp']) > base_date(cfg)]
    if not extensions:
        for k in ('Slow_Mean_Lambda', 'Mu_0'):
            models['BasisLinkedSpotModel.PLATINUM_CME.LBMA'].pop(k)
    if zero_carry:
        # F(t,T) = S(t): the only world in which a martingale SPOT makes a martingale FORWARD.
        # The carry goes STATIC rather than zero-vol - `QuadraticCarryCurveModel` refuses a zero
        # Sigma at construction, and `NoModel='Constant'` holds an unmodelled factor flat, which
        # is the same curve with one fewer moving part.
        cfg['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Model Configuration']['.ModelParams']['modeldefaults'].pop('ForwardRate')
        models.pop('QuadraticCarryCurveModel.PLATINUM_CARRY')
        for curve in ('ForwardRate.PLATINUM_CARRY', 'InterestRate.USD-REPO'):
            for knot in factors[curve]['Curve']['.Curve']['data']:
                knot[1] = 0.0
    return cfg


def base_date(cfg):
    return pd.Timestamp(cfg['Calc']['Calculation']['Base_Date']['.Timestamp'])


def run(cfg, prec=torch.float64, **overrides):
    """`load_json` + the calculation the JSON names. `prec` is a harness knob, not a deal input:
    the exact gates need float64 and `Context.run_job` fixes CMC at float32."""
    cx = rf.Context()
    cx.load_json((json.dumps(cfg), 'commodity_aps_world.json'))
    if prec is None:
        return cx.run_job(overrides or None)
    return rf.run_cmc(cx.current_cfg, prec=prec, overrides=overrides or None)


# ---------------------------------------------------------------------------
# the independent value, rebuilt from the market data and the captured paths
# ---------------------------------------------------------------------------

class Reference(object):
    """The average-price swap written a second time, off the JSON and the scenario buffers.

    Deliberately NOT the pricer's shape: a python loop over fixings, `np.interp` for every curve
    read, and the slow mean re-derived by the recursion `BasisLinkedSpotModel` documents
    (`mu[t] = lam mu[t-1] + (1-lam) b[t]`, seeded at `Mu_0`) rather than read back out of the
    buffer the pricer reads. So agreement is two routes meeting, not one route restated.
    """

    def __init__(self, cfg, out):
        calculation, deal, factors, models = _parts(cfg)
        self.base = base_date(cfg)
        self.grid = list(out['Results']['mtm'].index)
        self.sofr = np.array(factors['InterestRate.USD-SOFR']['Curve']['.Curve']['data'])
        self.repo = np.array(factors['InterestRate.USD-REPO']['Curve']['.Curve']['data'])
        self.units = deal['Units'] * (1.0 if deal['Buy_Sell'] == 'Buy' else -1.0)
        self.strike = deal['Fixed_Price']
        self.settle = pd.Timestamp(deal['Settlement_Date']['.Timestamp'])
        rows = deal['Sampling_Data']
        self.dates = [pd.Timestamp(r[0]['.Timestamp']) for r in rows]
        self.known = [r[1] for r in rows]
        self.weight = np.array([r[2] for r in rows]) / sum(r[2] for r in rows)

        basis_model = models['BasisLinkedSpotModel.PLATINUM_CME.LBMA']
        self.phi = basis_model['Phi']
        self.lam = basis_model.get('Slow_Mean_Lambda', 0.0)
        scen = out['Results']['scenarios']
        self.spot = scen['CommodityPrice.PLATINUM_CME'].xs(0.0, level='tenor').values.T
        self.basis = scen['ObservedBasis.PLATINUM_CME.LBMA'].xs(0.0, level='tenor').values.T
        if 'ForwardRate.PLATINUM_CARRY' in scen:
            carry = scen['ForwardRate.PLATINUM_CARRY']
            self.knots = list(carry.index.get_level_values('tenor').unique())
            self.z = np.stack([carry.xs(k, level='tenor').values.T for k in self.knots], axis=-1)
        else:
            # a world whose carry has no process: the knots are the market's, on every row
            declared = np.array(factors['ForwardRate.PLATINUM_CARRY']['Curve']['.Curve']['data'])
            self.knots = list(declared[:, 0])
            self.z = np.broadcast_to(declared[:, 1], self.spot.shape + (2,))
        # the slow mean, by its own recursion off the realised basis path
        self.mu = np.zeros_like(self.basis)
        if self.lam:
            self.mu[0] = basis_model['Mu_0']
            for t in range(1, len(self.mu)):
                self.mu[t] = self.lam * self.mu[t - 1] + (1.0 - self.lam) * self.basis[t]

    def _rate(self, curve, years):
        return np.interp(years, curve[:, 0], curve[:, 1])

    def sample(self, t, j):
        """The `j`th fixing as row `t` sees it - realised level or conditional expectation."""
        date = self.dates[j]
        if date < self.base:
            return np.full(self.spot.shape[1], self.known[j])
        if date < self.grid[t]:
            i = self.grid.index(date)
            return self.spot[i] + self.basis[i]
        days = (date - self.grid[t]).days
        alpha = np.interp((date - utils.excel_offset).days, self.knots, [0.0, 1.0])
        z = self.z[t, :, 0] * (1.0 - alpha) + self.z[t, :, 1] * alpha
        forward = self.spot[t] * np.exp(
            z * days / utils.DAYS_IN_YEAR + self._rate(self.repo, days / 365.0) * days / 365.0)
        n = self.grid.index(date) - t
        return forward + self.mu[t] + self.phi ** n * (self.basis[t] - self.mu[t])

    def average(self, t):
        return sum(w * self.sample(t, j) for j, w in enumerate(self.weight))

    def discount(self, t):
        tau = (self.settle - self.grid[t]).days / 365.0
        return np.exp(-self._rate(self.sofr, tau) * tau)

    def value(self, t):
        return self.discount(t) * self.units * (self.average(t) - self.strike)

    def realised_count(self, t):
        return sum(1 for d in self.dates if d < self.grid[t])

    @property
    def rows(self):
        """The rows the deal is alive on. The mtm frame runs one row past the settlement date -
        the grid's own last date - where an expired deal marks a hard zero."""
        return [t for t, d in enumerate(self.grid) if d <= self.settle]


def priced(cfg, **overrides):
    """`(reported mtm frame, Reference)` on one run - the paths the reference rebuilds from are
    the paths that were priced, so nothing here depends on the RNG being reproducible."""
    _, out = run(cfg, **dict({'Batch_Size': 64, 'Calc_Scenarios': 'All'}, **overrides))
    return out['Results']['mtm'], Reference(cfg, out)


# ---------------------------------------------------------------------------
# 1. inception, closed form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('buy', ['Buy', 'Sell'])
@pytest.mark.parametrize('inception', [True, False])
def test_the_inception_value_is_the_closed_form_off_the_market_data(buy, inception):
    """Row 0 against the deal's own algebra, rebuilt from the JSON: every future fixing is the
    carry curve's closed form plus the basis AR, every past one its declared price, and the whole
    thing discounted from the settlement date.

    Row 0 is DETERMINISTIC - the state is the market's at t=0 - so this is an equality rather than
    an average, and it is asserted per path so a pricer that let one path differ cannot hide in a
    mean. `inception=True` drops the two declared fixings, which is the shape the brief names;
    `False` keeps them, which is the shape that has a lookback at all."""
    mtm, ref = priced(world(buy=buy, inception=inception))
    got, want = mtm.iloc[0].values, ref.value(0)
    assert ref.realised_count(0) == (0 if inception else 2)
    assert np.abs(got / want - 1.0).max() < 1e-12, (
        f'inception mark {got[0]:.6f} vs closed form {want[0]:.6f}')
    assert got.std() == 0.0, 'the t=0 mark is not path-independent'
    # the fixture has something to see: the basis is a live part of the mark
    assert abs(ref.basis[0, 0]) > 5.0 and abs(ref.mu[0, 0] - ref.basis[0, 0]) > 1.0


def test_the_projection_is_the_only_thing_between_a_base_valuation_and_the_simulated_row_zero():
    """Base valuation prices the SAME deal in a world with no processes in it, and the framework's
    answer for a factor with no process is `NoModel='Constant'` - so the basis sits at its observed
    level and does not decay. `get_observed_basis_decay` returns `phi = 1` for exactly that case,
    which is that rule written in the projection's own arithmetic instead of as a branch.

    The consequence is a real, measurable gap between the two engines on one world, and it is
    pinned here rather than discovered later: it is the decay term and nothing else."""
    cfg = world()
    calculation, deal, _, _ = _parts(cfg)
    cfg['Calc']['Calculation'] = {'Object': 'BaseValuation', 'Currency': 'USD',
                                  'Base_Date': calculation['Base_Date'], 'Greeks': 'No'}
    cx = rf.Context()
    cx.load_json((json.dumps(cfg), 'aps_base.json'))
    _, base = cx.run_job()
    flat = base['Results']['mtm'].set_index('Reference').loc['APS1', 'Value']

    mtm, ref = priced(world())
    # the same reference with the AR replaced by "the basis stays where it is"
    ref.phi, ref.lam, ref.mu = 1.0, 0.0, np.zeros_like(ref.mu)
    assert flat == pytest.approx(ref.value(0)[0], rel=1e-12), (
        f'base valuation {flat:.6f} is not the no-decay mark {ref.value(0)[0]:.6f}')
    gap = flat - mtm.iloc[0, 0]
    assert abs(gap) > 1e3, f'the decay term is worth {gap:.2f} - this fixture cannot see it'


# ---------------------------------------------------------------------------
# 2. mid-life, per path
# ---------------------------------------------------------------------------

def test_the_mid_life_row_decomposes_into_realised_and_remaining_per_path():
    """The gate the mid-life mandate exists for. At every row of the grid - which includes rows
    with 2, 3, 4, 5 and 6 of the six fixings realised - the reported value must equal, PER PATH,
    the realised half read at the fixing rows plus the remaining half projected from this row.

    Every read the pricer makes is pinned by this and by nothing else in the file: which scenario
    row a realised fixing comes from, which weight it carries, how far the basis has to decay, and
    that the composition is summed on both halves.
    """
    mtm, ref = priced(world())
    counts = set()
    for t in ref.rows:                     # the deal grid stops at settlement
        got, want = mtm.iloc[t].values, ref.value(t)
        assert np.abs(got / want - 1.0).max() < 1e-11, (
            f'row {t} ({ref.grid[t].date()}, {ref.realised_count(t)} realised) off by '
            f'{np.abs(got / want - 1.0).max():.3e}')
        counts.add(ref.realised_count(t))
    assert counts == {2, 3, 4, 5, 6}, f'the grid never realises a fixing mid-life: {counts}'
    # ...and the paths are genuinely apart, or this is an equality between two constants
    assert mtm.iloc[6].std() > 0.05 * abs(mtm.iloc[6].mean())


def test_a_realised_fixing_and_the_projection_of_it_agree_on_its_own_row():
    """The seam is CONTINUOUS, and that is a property rather than a coincidence: the split is
    `fixing_day < row`, so a fixing lands in the PROJECTED half on its own row - where the forward
    has zero tenor and the basis has zero steps to decay, making the projection the composed spot
    itself. A pricer whose two halves disagreed would show a jump at exactly those rows."""
    mtm, ref = priced(world())
    rows = [t for t in ref.rows if ref.grid[t] in ref.dates]
    assert len(rows) == 4, rows
    for t in rows:
        j = ref.dates.index(ref.grid[t])
        i = ref.grid.index(ref.dates[j])
        assert np.abs(ref.sample(t, j) / (ref.spot[i] + ref.basis[i]) - 1.0).max() < 1e-14
        assert np.abs(mtm.iloc[t].values / ref.value(t) - 1.0).max() < 1e-11


# ---------------------------------------------------------------------------
# 3. terminal + the settle-once discipline
# ---------------------------------------------------------------------------

def test_past_the_last_fixing_the_value_is_the_realised_average_discounted():
    """No projection is left, so the value is arithmetic on data: `D (A - K) N`, with `A` the
    weighted realised average and `D` running to the settlement date. Asserted on BOTH post-fixing
    rows, so the discount is exercised at a live tenor and at zero."""
    mtm, ref = priced(world())
    tail = [t for t in ref.rows if ref.realised_count(t) == len(ref.dates)]
    assert len(tail) == 2, tail
    for t in tail:
        realised = sum(w * ref.sample(t, j) for j, w in enumerate(ref.weight))
        want = ref.discount(t) * ref.units * (realised - ref.strike)
        assert np.abs(mtm.iloc[t].values / want - 1.0).max() < 1e-12
    assert ref.discount(tail[0]) < 0.9995 and ref.discount(tail[-1]) == 1.0


def test_the_settlement_is_booked_once_at_its_undiscounted_amount(monkeypatch):
    """Two statements, because they fail differently. The COUNT is the discipline - one call per
    pricing pass - and the AMOUNT is the number that discipline protects.

    A spy is the only instrument that can see the count: `cash_settle` books into the currency
    map, `reset` registers exactly one settlement date on it, and every call at another row is
    discarded. So a settlement fired per reporting row reports identical cashflows (measured) and
    is visible only here."""
    calls = []
    settle = pricing.cash_settle
    monkeypatch.setattr(pricing, 'cash_settle', lambda shared, ccy, index, value: (
        calls.append((ccy, index)), settle(shared, ccy, index, value))[1])

    cfg = world()
    _, out = run(cfg, Batch_Size=64, Calc_Scenarios='All', Generate_Cashflows='Yes')
    ref = Reference(cfg, out)
    assert len(calls) == 1, f'the settlement fired {len(calls)} times: {calls}'

    cash = out['Results']['cashflows']['USD']
    booked = cash.loc[ref.settle].values
    realised = sum(w * ref.sample(ref.rows[-1], j) for j, w in enumerate(ref.weight))
    want = ref.units * (realised - ref.strike)
    assert np.abs(booked / want - 1.0).max() < 1e-12, 'the settled amount is not the payoff'
    assert abs(want).min() > 1.0, 'the settled amount is zero - this gate is reading nothing'


# ---------------------------------------------------------------------------
# 4. the martingale / two-clock statement
# ---------------------------------------------------------------------------

def _average_path(cfg, **overrides):
    """`(E[A_t], se, A_0)` on the exposure grid, with `A` recovered from the reported value by
    dividing out the deterministic discount - so this measures the AVERAGE the pricer projects
    rather than the discounting, which gates 1-3 already hold exactly."""
    mtm, ref = priced(cfg, **overrides)
    a = np.stack([mtm.iloc[t].values / (ref.discount(t) * ref.units) + ref.strike
                  for t in ref.rows])
    # row 0 is deterministic, so its standard error is float noise and the ratio is 0/0; gate 1
    # holds that row exactly and this one starts at row 1.
    return a[1:].mean(axis=1), a[1:].std(axis=1) / np.sqrt(a.shape[1]), a[0, 0]


def test_the_projected_average_is_a_martingale_when_the_forward_is_one():
    """With the slow mean off the basis AR reverts to zero, the spot is a martingale by the GARCH
    block's convexity correction, and a ZERO carry and repo make `F(t,T) = S(t)` - so every term of
    `A_t` is a martingale and `E[A_t] = A_0` at every row, whichever side of a fixing it is on.

    That is the tower property of the pricer's own split: a realised fixing enters at its observed
    level and an unrealised one at a conditional mean, and the two have to agree in expectation or
    the split is losing something at the seam."""
    got, se, a0 = _average_path(world(extensions=False, zero_carry=True), Batch_Size=8192)
    off = np.abs(got - a0) / se
    assert off.max() < 4.0, (
        f'E[A_t] wanders {off.max():.1f} se from A_0 = {a0:.4f}: {np.round(got, 3)}')
    assert (se > 0.05).all(), 'the paths do not disperse - gate blind'


def test_a_live_carry_breaks_that_martingale_and_by_how_much():
    """The other half of the two-clock discipline, and the reason the gate above needs its own
    world rather than the shipped one. The spot is a martingale; the FORWARD is not, because
    `F(t,T) = S(t) exp(c (T-t))` gives up its carry as `t` runs out. So with a carry on, the same
    test must FAIL - and if it ever stopped, the gate above would have become a tautology about a
    curve that is flat.

    IT NEEDS AN EXAGGERATED CARRY, AND THAT IS THE FINDING. At the SHIPPED carry (1.9% falling to
    1.1%, plus a 1.2-1.9% repo) the drift is -4.90 on `A_0 = 1649.02` - 0.30%, and **2.2 standard
    errors at 8192 paths**. The E[dF] bias this deal carries is real and is BELOW MONTE CARLO POWER
    at any path count a gate can afford, so a test written at the shipped carry would be measuring
    noise and reading as a pass either way. At ten times the carry the same runs land at -39.4 and
    17.4 se, which is the mechanism made visible rather than assumed."""
    got, se, a0 = _average_path(world(extensions=False, carry_scale=10.0), Batch_Size=8192)
    off = np.abs(got - a0) / se
    assert off.max() > 10.0, (
        f'a live carry leaves E[A_t] within {off.max():.1f} se of A_0 - the carry is not live')
    assert (got - a0).max() < 0.0, 'the drift is not the sign a decaying forward has'


# ---------------------------------------------------------------------------
# 5. AAD
# ---------------------------------------------------------------------------

def test_the_gradient_names_exactly_the_factors_this_payoff_reads():
    """The absent-factor discipline. An average-price swap has no optionality, so no volatility
    surface and no model-parameter block may appear in its gradient - and every factor it DOES
    read must, including both carry knots, which is what says the two-knot representation is
    identified rather than one knot doing all the work."""
    cfg = world()
    calculation, _, _, _ = _parts(cfg)
    cfg['Calc']['Calculation'] = {'Object': 'BaseValuation', 'Currency': 'USD',
                                  'Base_Date': calculation['Base_Date'], 'Greeks': 'First'}
    cx = rf.Context()
    cx.load_json((json.dumps(cfg), 'aps_greeks.json'))
    _, out = cx.run_job()
    greeks = out['Results']['Greeks_First']
    names = set(greeks.index.get_level_values('Rate'))
    assert names == {'CommodityPrice.PLATINUM_CME', 'ObservedBasis.PLATINUM_CME.LBMA',
                     'ForwardRate.PLATINUM_CARRY', 'InterestRate.USD-REPO',
                     'InterestRate.USD-SOFR'}, sorted(names)
    values = greeks['root'].astype(float)
    assert np.isfinite(values.values).all(), values
    knots = values.xs('ForwardRate.PLATINUM_CARRY')
    assert len(knots) == 2 and (knots.abs() > 1.0).all(), knots
    assert abs(float(values.xs('CommodityPrice.PLATINUM_CME').iloc[0])) > 1.0
    assert abs(float(values.xs('ObservedBasis.PLATINUM_CME.LBMA').iloc[0])) > 1.0


# ---------------------------------------------------------------------------
# 6. CMC end to end
# ---------------------------------------------------------------------------

def test_the_exposure_profile_runs_through_run_job_and_freezes_when_the_last_fixing_is_in():
    """The JSON is the contract: `load_json` + `run_job`, no overrides, collateral off.

    The anti-placebo half is a SHAPE, and the shape is the opposite of the intuitive one. `A_t` is
    a conditional expectation, so its cross-path dispersion GROWS as information arrives - it does
    not collapse. What it does at the last fixing is STOP: from there `A` is a realised number and
    the only thing still moving is the discount, so the spread's remaining growth is exactly the
    discount's, to the digit. That is a two-sided statement - the profile must rise while fixings
    remain and must stop rising once they do not - and a value ignoring the schedule reproduces
    neither half.

    Measured on this world: 0 -> 5.31e5 over the twelve live rows, then 5.306e5 / 5.310e5 / 5.314e5
    on the three rows past the last fixing, whose ratios ARE D(t)/D(12) to 1e-9.
    """
    cfg = world()
    _, out = run(cfg, prec=None)
    mtm = out['Results']['mtm']
    profile = out['Results']['exposure_profile']
    assert len(profile) == len(mtm), (len(profile), len(mtm))
    assert (profile.std() > 0.0).all() and np.isfinite(profile.values).all(), (
        'the exposure profile is flat')

    spread = mtm.std(axis=1).values
    last_fix = 12                                     # 2026-05-15, the sixth sampling date
    assert spread[0] == 0.0, 'the t0 mark disperses'
    assert (np.diff(spread[:last_fix + 1]) > 0.0).all(), (
        f'the profile stops growing before the last fixing: {np.round(spread)}')
    assert spread[last_fix] > 1e5, spread[last_fix]

    # past the last fixing only the discount moves, and it moves the spread by exactly its ratio
    calculation, deal, factors, _ = _parts(cfg)
    sofr = np.array(factors['InterestRate.USD-SOFR']['Curve']['.Curve']['data'])
    grid, settle = list(mtm.index), pd.Timestamp(deal['Settlement_Date']['.Timestamp'])
    tau = np.array([(settle - grid[t]).days / 365.0 for t in range(last_fix, 15)])
    discount = np.exp(-np.interp(tau, sofr[:, 0], sofr[:, 1]) * tau)
    assert np.abs(spread[last_fix:15] / spread[last_fix] - discount / discount[0]).max() < 1e-6, (
        f'the frozen tail is not pure discounting: {spread[last_fix:15]}')


# ---------------------------------------------------------------------------
# 7. the mutants
# ---------------------------------------------------------------------------

def _mutate(monkeypatch, old, new):
    """Exec a one-token edit of the pricer's own source onto `pricing`, which is where the deal's
    `generate` looks it up. Editing the SOURCE rather than writing a second pricer is what makes a
    mutant a mutant: everything else about it is the shipped code.

    The mutant's globals ARE the module's dict, not a snapshot of it - a snapshot freezes every
    name the pricer calls at mutation time, so a spy installed afterwards (`cash_settle`) would be
    invisible to the mutant and the mutant would score as killed for the wrong reason. Measured:
    every mutant reads as caught by the settle-once gate under a snapshot."""
    src = textwrap.dedent(inspect.getsource(pricing.pv_average_price_swap))
    assert src.count(old) == 1, f'mutation target not unique: {old!r}'
    exec(compile(src.replace(old, new).replace('def pv_average_price_swap', 'def _mutant'),
                 '<mutant>', 'exec'), pricing.__dict__)
    monkeypatch.setattr(pricing, 'pv_average_price_swap', pricing.__dict__.pop('_mutant'))


#: `(name, old, new)`. Each is the smallest edit that spells the named defect.
MUTANTS = [
    ('lookback off by one',
     'sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)',
     'sim_samples[:, :utils.RESET_INDEX_Scenario + 1] + np.array([0.0, 0.0, 1.0]), shared)'),
    ('realised read from the current row',
     'torch.sum(realised.unsqueeze(0) * weight[:, :n] * past[:, :n], dim=1))',
     'torch.sum(utils.calc_time_grid_spot_rate(factor_dep[\'Commodity\'], deal_time, shared)'
     '.unsqueeze(1) * weight[:, :n] * past[:, :n], dim=1))'),
    ('decay in calendar days',
     "factor_dep['Basis_Phi'] ** steps", "factor_dep['Basis_Phi'] ** tau"),
    ('realised read drops the basis',
     "factor_dep['Commodity'], sim_samples", "factor_dep['Spot'], sim_samples"),
    ('composed spot under the carry exponential',
     "spot = utils.calc_time_grid_spot_rate(factor_dep['Spot'], deal_time, shared)",
     "spot = utils.calc_time_grid_spot_rate(factor_dep['Commodity'], deal_time, shared)"),
    ('K charged per fixing, unweighted',
     "(average - factor_dep['Strike'])",
     "(average - factor_dep['Strike'] * weight.shape[1])"),
]


@pytest.mark.parametrize('name,old,new', MUTANTS, ids=[m[0] for m in MUTANTS])
def test_every_mutant_dies(monkeypatch, name, old, new):
    """One run per mutant, scored on the three exact gates at once: inception, every mid-life row,
    and the terminal row. The assertion is that the SHIPPED reference and the mutant's marks
    separate somewhere - which is the same statement the gates above make, with the reference held
    fixed and the pricer moved."""
    cfg = world()
    clean, ref = priced(cfg)
    _mutate(monkeypatch, old, new)
    mutated, _ = priced(cfg)
    rows = ref.rows
    delta = np.array([np.abs(mutated.iloc[t].values / clean.iloc[t].values - 1.0).max()
                      for t in rows])
    assert delta.max() > 1e-9, f'{name}: survives every row (max rel move {delta.max():.3e})'
    # ...and name the rows it moves, so a later fixture change that blinds one is visible
    assert np.abs(mutated.iloc[0].values / ref.value(0) - 1.0).max() > 1e-9 or delta[1:].max() > 1e-9


def test_charging_the_strike_inside_the_weights_is_the_same_number(monkeypatch):
    """The mutant the brief names, run and reported as a NO-OP with its reason.

    `make_sampling_data` normalises the weights, so `sum_j w_j (S_j - K) == sum_j w_j S_j - K`
    identically. Asserting BIT equality rather than skipping it is what makes the claim checkable:
    if the weights ever stopped summing to one, this test - not a value gate - is what fails."""
    cfg = world()
    clean, ref = priced(cfg)
    assert ref.weight.sum() == pytest.approx(1.0, abs=1e-15)
    _mutate(monkeypatch, "(average - factor_dep['Strike'])",
            "(average - factor_dep['Strike'] * weight.sum())")
    mutated, _ = priced(cfg)
    assert np.array_equal(clean.values, mutated.values), (
        'the two K placements differ - the weights no longer sum to one')


def test_settling_on_every_reporting_row_is_caught_by_the_count_and_by_nothing_else(monkeypatch):
    """The other reported survivor. Fifteen calls instead of one, identical cashflows: the
    currency map holds exactly one settlement index, so the extra calls are discarded and the
    surviving one lands on the row the shipped code settles at, with the same value."""
    cfg = world()
    _, clean = run(cfg, Batch_Size=64, Generate_Cashflows='Yes')
    calls = []
    settle = pricing.cash_settle
    monkeypatch.setattr(pricing, 'cash_settle', lambda shared, ccy, index, value: (
        calls.append(index), settle(shared, ccy, index, value))[1])
    _mutate(monkeypatch,
            "    cash_settle(shared, factor_dep['SettleCurrency'],\n"
            "                deal_data.Time_dep.deal_time_grid[-1], cash[-1])",
            "    [cash_settle(shared, factor_dep['SettleCurrency'], r, cash[i])\n"
            "     for i, r in enumerate(deal_data.Time_dep.deal_time_grid)]")
    _, mutated = run(cfg, Batch_Size=64, Generate_Cashflows='Yes')
    assert len(calls) > 10, f'the per-row mutant fired {len(calls)} times'
    assert np.array_equal(clean['Results']['cashflows']['USD'].values,
                          mutated['Results']['cashflows']['USD'].values), (
        'the per-row settlement moved a cashflow - update the docstring, it is now value-visible')


# ---------------------------------------------------------------------------
# 8. schema
# ---------------------------------------------------------------------------

def _validate(**field):
    return schema.validate_instrument(instruments.construct_instrument(
        dict({'Object': DEAL_TYPE, 'Reference': 'X'}, **field), {}))


def test_the_deal_is_declared_and_offered_in_exactly_one_menu():
    """A declared type in no create-menu group is a deal no author can reach; in two, a UI shows
    it twice. `test_schema_emission` holds the first globally - this holds both for this type."""
    groups = [g for g, members in schema.mapping['Instrument']['groups'].items()
              if DEAL_TYPE in members]
    assert groups == ['New Energy Derivative'], groups
    assert schema.mapping['Instrument']['types'][DEAL_TYPE] == [
        'Admin', '{}.Fields'.format(DEAL_TYPE)]


def test_the_fields_the_deal_cannot_price_without_are_required():
    """`calc_dependencies` reads each of these unguarded, so an author who omits one gets a
    `KeyError` out of the compile rather than a message naming the field."""
    missing = _validate()
    for name in ('Commodity', 'Carry', 'Currency', 'Settlement_Date', 'Sampling_Data'):
        assert '{} is required'.format(name) in missing, name
    assert set(schema.required_fields(instruments.CommodityAveragePriceSwapDeal)) == {
        'Commodity', 'Carry', 'Currency', 'Settlement_Date', 'Sampling_Data'}


def test_a_settlement_inside_the_sampling_window_is_refused():
    """The one cross-field rule this deal introduces, paired with a conforming deal so the gate is
    not passing on a validator that complains about everything."""
    rows = [[pd.Timestamp('2026-02-16'), 0.0, 1.0], [pd.Timestamp('2026-05-15'), 0.0, 1.0]]
    bad = _validate(Sampling_Data=rows, Settlement_Date=pd.Timestamp('2026-04-30'))
    good = _validate(Sampling_Data=rows, Settlement_Date=pd.Timestamp('2026-05-15'))
    message = 'Settlement_Date must be on or after the last Sampling_Data date'
    assert message in bad and message not in good
