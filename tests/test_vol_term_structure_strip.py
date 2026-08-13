"""AN IMPLIED VOL IS CUMULATIVE, AND A MONITORED INTERVAL WANTS THE DIFFERENCE.

``pv_discrete_barrier_option`` read ONE implied vol per MTM row - the surface at the strike's
moneyness and the EXPIRY tenor - and ``sim_spot_oss`` applied it to every monitoring interval as
``var = sigma^2 * dt``. ``sigma(T)^2 * T`` is the TOTAL variance to ``T``, so the variance of
``[T_j-1, T_j]`` is a DIFFERENCE of cumulative variances and not ``sigma(T)^2 * dt_j``.
``pv_MC_AutoCallSwap`` carried the identical expression in both of its branches.

WHY THE WHOLE SUITE WAS GREEN, and it is the same mechanism as the carry strip one layer up: the
wrong allocation TELESCOPES to exactly the right total variance. The terminal distribution is
right, so every European limit, in-out parity and every CRN gradient ladder are right - only the
path-dependent MONITORING is biased. And a FLAT surface makes wrong and right identical to the last
bit, which is what every barrier, TARF and autocall fixture in this repo is.

MEASURED. Two surfaces carrying the SAME 1y implied vol - flat 0.2479, against a term structure
running 0.10 at the short end to 0.32 at 2y with ``sigma(1y) = 0.2479`` - priced
``Down_And_Out``/``Down_And_In``/``Up_And_Out`` BITWISE IDENTICALLY, while the true interval strip
ran 0.111 -> 0.336 against the single 0.2479 applied to all of it, each interval's dispersion off
by x0.45 to x1.36. Against the oracle below, the sloped world read

    Down_And_Out  1004.910 -> 1021.304   oracle 1019.81 +/- 1.25    (-1.46% before)
    Down_And_In    159.053 ->  142.659   oracle  142.63 +/- 0.41    (+11.53% before)
    Up_And_Out      66.422 ->   74.708   oracle   74.69 +/- 0.19    (-11.07% before)

and the flat world did not move where the strip is the whole story: 4 ULP on a ``Down_And_Out``
exposure profile and 99 on an ``Up_And_Out`` (the differencing amplifies the rounding of a
cumulative time by ``T_j/dt_j``, which is why nothing here uses ``torch.equal`` on a price).

ONE FLAT-SURFACE NUMBER DID MOVE, and it is not the strip. The in-out-parity vanilla inside
``sim_spot_oss`` valued its European at ``sqrt(sum_j clamp(sigma^2 dt_j, 1e-4))`` - the SIMULATED
strip's total, including the ``1e-4`` variance floor a ZERO-LENGTH interval collects, which is
every MTM row that IS an observation date. It now takes ``sigma(K, tau) * sqrt(tau)``, the same
European quote the already-hit leg marks with. Measured on a monthly exposure grid: knock-IN rows
that sit on an observation date move by -0.08% of the profile mean and -0.71% at the worst element
(``Down_And_In`` 236.378 -> 236.197, worst row 359.51 -> 359.34); every other row is ULP.

AND THE FLOOR ITSELF WOKE UP UNDER THE FIX, which is the second thing this file gates. Under the
defect every interval carried ``sigma(T)^2``, the LARGEST vol on an upward surface, so the ``1e-4``
clamp essentially never bound; a correct forward-variance strip hands it genuinely small numbers
and it binds wherever ``sigma_fwd < 0.01/sqrt(dt)`` - 19.1% annualised at daily monitoring. It is
now conditioned on the ZERO-LENGTH interval it exists for (``dt == 0``, elementwise), and the
drift is taken from the same variance the vol is, so every interval is a martingale under the law
it is actually drawn from. Measured on an upward 0.12 -> 0.24 surface with 114 of 365 intervals
floored, eight seeds against a floor-free oracle:

    daily             Down_And_Out   Up_And_Out
    shipped spelling      +1.584%      -6.870%    unconditional floor, drift off the unclamped var
    floor alone           +0.049%      -5.576%    the two halves partly cancel on the Down_And_Out
    conditional + one var +0.183%      +0.403%    1.7 se and 0.9 se

Weekly (0 of 53) and monthly (0 of 13) are BITWISE unchanged under all three, and so is the monthly
exposure grid in `test_recompute_equity_pricers` - value, CVA and all 13 CVA-gradient entries.
What the ``dt == 0`` clamp actually protects is measured too, by deleting it: every VALUE stays
finite (that exposure profile moves -0.046%) and 11 of the 13 CVA-gradient entries go NaN, 7 of 13
on the averaging autocall (``sqrt`` has an infinite derivative at zero and the variance carries the
surface's graph), which is what the autocall's own "prevent gradients from blowing up" comment was
about without saying which intervals needed it.

THE ORACLE is a fine-step Monte Carlo under the instantaneous vol term structure the surface's own
total variance implies - ``dV(t) = d(sigma(t)^2 t)`` per sub-step - with the same discrete
observation dates, and it shares no code with the pricer: no `derivus` import inside it, its own
RNG, its own reflection-free brute-force monitoring. Sub-stepping is not what makes it an oracle
(under DISCRETE monitoring only the observation-date marginals matter, so 1 and 4 sub-steps agree
inside the Monte Carlo error); what makes it one is that the variance ALLOCATION across intervals
is written independently, which is the quantity under test.

THE TWO VOL QUANTITIES ARE BOTH LEGITIMATE and this file asserts both. The SIMULATION takes the
interval strip; the EUROPEAN legs - the already-hit KI mark and the in-out-parity vanilla - take
``sigma(K, tau) * sqrt(tau)``, the surface's own quote for the option being valued.
``test_the_never_knocking_limit_is_black_at_the_expiry_vol`` is the invariant that must NOT move:
push the barrier out of reach on any of the four surfaces and the pricer must reproduce Black at
the expiry read, which the defect also did - it is here because the fix must not break it. They
are two READS of one surface, not two surfaces, so they have to agree on the MONEYNESS: the strip
takes the deal's declared ``use_forwards``, and reading it at the fixings' own forwards instead
broke that European by +0.948% and in-out parity by -11.03 on a smiley surface.

THE THIRD ADOPTER, `pv_MC_AutoCallSwap`, is gated here too, and it had nothing before: every
autocall fixture in the repo is flat-surface AND ``r = q = 0``, so its half of this correction
shipped unmeasured. On the same two surfaces with quarterly coupons and a 1.02 threshold the
pre-port pricer returns 0.034026629996 on BOTH (``array_equal`` True), against a brute-force
oracle of 0.037094 on the sloped one - -8.27% and 207 standard errors - where the strip reads
0.037064, -0.081% and 2.0 se. The two surfaces separate by +8.93%, twice the barrier's four
monthly percent, because a quarterly digital trigger is all monitoring and no terminal payoff.

AND THE AUTOCALL CAN SETTLE THE MONEYNESS QUESTION EXACTLY, which the barrier cannot. A ONE-COUPON
autocall is a closed-form digital - ``sim_spot``'s survival probability is ``norm_cdf`` of it and
no draw ever advances ``Sj`` - so its European limit is asserted to 0 ULP with no Monte Carlo error
anywhere. It reads the deal's DECLARED (spot-moneyness) quote exactly, 0.024490310460; a strip read
at the fixing's own FORWARD moneyness is 0.024428368300, -0.2529% away. The same statement on the
barrier is +0.948% and 8.3 standard errors of a never-knocking ``Down_And_Out`` against Black, with
in-out parity going -11.03 -> -0.19. INTERNAL CONSISTENCY DECIDED IT: a pricer whose simulation and
whose European legs read one surface at two moneynesses cannot reproduce its own quote. The
per-fixing SMILE that costs is a real modelling question - a desk quoting sticky-forward moneyness
wants the other convention - and it is open in `roadmap.md` with these numbers on it. The
term-structure half is separable and untouched by the choice: alternating only that flag in one
process on the smile-free surfaces leaves the prices agreeing to 1.1e-15 relative.

MUTATION MATRIX (each applied from a scratch runner - an attribute rebind on `pricing` for the two
strip functions, a re-exec of mutated source for the three inside `sim_spot_oss`'s closure - run,
discarded, no edit to the tree. `sep` = the two-surface separation gates, `alg` = the two algebra
gates, `decl` = the declared-moneyness gate, `black`/`parity` = the European invariant and in-out
parity on the flat/sloped x plain/smiley surfaces, `dens` = the daily-density gate against the
oracle, `dnsE` = the daily-density EUROPEAN limit against Black, `ac-sep` = the autocall
separation, `ac-eur` = the autocall's exact one-coupon digital):

`black+parity` names the ARM of its 2x2 that dies, by the feature the arm carries: `flat` is the
flat smile-free surface, which no mutation here reaches; `slope` the two SLOPED arms (both smiles);
`smile` the two SMILEY ones (both slopes). The two overlap on sloped-and-smiley and no row needs
them to disagree there.

    mutation                                          sep  alg  decl black+parity  dens dnsE ac-sep ac-eur
                                                                     flat slope smile
    (1) strip collapsed to the expiry vol (the defect) DIED DIED pass pass pass  pass  DIED pass DIED  pass
    (2) differencing dropped: sigma_j^2 * dt_j         DIED DIED pass pass DIED  pass  DIED DIED DIED  pass
    (3) every read taken at the EXPIRY tenor           DIED pass pass pass pass  pass  DIED pass DIED  pass
    (4) moneyness hard-coded back to the FORWARD       pass pass DIED pass pass  DIED  pass pass pass  DIED
    (5) the 1e-4 variance floor made unconditional     pass pass pass pass pass  pass  DIED DIED pass  pass
    (6) the floor deleted, including at dt == 0        pass pass pass pass pass  pass  pass pass pass  pass
    (7) the drift left on the UNCLAMPED variance       pass pass pass pass pass  pass  pass pass pass  pass
    (7b) BOTH, which is the spelling that shipped      pass pass pass pass pass  pass  DIED DIED pass  pass
    (8) CONTROL: the fix itself                        PASS PASS PASS PASS PASS  PASS  PASS PASS PASS  PASS

WHICH ARM of the density gate dies is itself a reading, and it is not the same one twice. (1), (3)
and (7b) die on `Down_And_Out`, (2) and (7b) on `Up_And_Out`, (5) on `Up_And_Out` ALONE - the floor
adds variance to the short end, which knocks an up-and-out out (-5.6%) far harder than it knocks a
down-and-out out (-0.13%, inside the oracle's own bar). Both arms are load-bearing and neither is
redundant. (5) and (7) are ALSO the pair that cancel: the floor alone reads +0.049% on the
`Down_And_Out` - CLOSER to the oracle than the fix - and only with the incoherent drift on top of it
(7b, what actually shipped) does that arm reach +1.584% and die.

`dnsE` is that same density with the barrier pushed out of reach, and its reference is ARITHMETIC:
a floored interval is variance added to the TERMINAL law, so the European limit sees the floor with
no oracle at all. It kills (5) at 37.5 standard errors and (7b) at 51.9 against Black, where the
oracle arms need 8e6 paths to reach 12 and 6, and it is the only gate here that catches (2) on the
upward surface. It cannot see the moneyness half - this surface carries no smile - so it replaces
neither the `Up_And_Out` arm's oracle nor `decl`.

Read the `black+parity` columns twice. (1) is THE ORIGINAL DEFECT and it passes every European,
because one implied vol on every interval telescopes to exactly the right total variance - that is
the whole reason it shipped, and a matrix in which the defect died everywhere would mean the
fixture was wrong. (2) does not telescope, so a European catches it on the SLOPED surface and
nothing catches it on the flat one. (3) is the half no smile can see and (4) the half no term
structure can see. (5) is invisible to every gate in this file that is not DENSE - monthly and
weekly fixtures are bitwise identical with the floor on or off, which is why the density gate had
to be written rather than inherited.

ROWS (6) AND (7) ARE GREEN HERE AND THAT IS THE FINDING, not an omission: every fixture in this
file reports on `0d` with the base date OFF the observation strip, so no row of it has a
zero-length first interval and the `dt == 0` clamp is never reached. They are measured on
`test_recompute_equity_pricers`'s exposure grid instead, where 13 of 37 rows sit ON an observation
date. (6) leaves every VALUE finite and moves the barrier profile -0.046% and its CVA -0.041%, and
turns 11 of the 13 CVA-gradient entries NaN (7 of 13 on the averaging autocall, whose value moves
+0.20%) - SIX gates die, five in that file and `test_digital_terminal_step_is_integrated_not_sampled`
in `test_barrier_bridge`, so what the clamp buys is the GRADIENT and nothing else. (7) - the
incoherence, which shipped inside (7b) - is caught by NO gate in the repo: it moves the barrier
profile +0.0215% and its CVA +0.0240%, every CVA-gradient entry by up to 7.5e-4 relative, and the
averaging autocall's gradient SUM by 5.1%, with nothing asserting any of it. THE ATTRIBUTION IS
EXACT: (5) alone is bit-exact on that grid in value AND gradient - the floor binds there only at
`dt == 0`, where it still applies - and (7) applied to the fixed tree reproduces HEAD's profile and
CVA BITWISE on both deals, so the drift is the whole of what this seam moved in value there and the
strip is the whole of what it moved in gradient. It is closed here by construction: `drift`
and `vol` read one `var`, so the two cannot disagree.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import pricing, utils
from derivus.config import Config
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
SPOT, STRIKE, UNITS = 100.0, 100.0, 100.0
#: both non-zero and DIFFERENT - r = q kills the forward and with it the per-fixing moneyness,
#: r = 0 kills the discount. See conventions.md#fixture-degeneracy.
R, Q = 0.05, 0.01
EXPIRY_DAY = 365
OBS_DAYS = [30 * k for k in range(1, 13)]
MONEYNESS = (0.8, 1.0, 1.2)

#: the term structure. `sigma(1y)` is pinned to FLAT_VOL so the two surfaces are indistinguishable
#: to every European limit and only the monitoring can separate them.
TS_KNOTS = np.array([0.02, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
TS_VOLS = np.array([0.10, 0.14, 0.18, 0.2215, 0.2479, 0.29, 0.32])
FLAT_VOL = 0.2479
SMILE = 0.35                      # vol = base + SMILE * (moneyness - 1)^2
#: a SECOND upward term structure, 0.12 -> 0.24 over the year, for the density gate: it puts the
#: short end's forward vol BELOW `sqrt(1e-4/dt)` = 19.1% at daily monitoring, which is the only
#: place the variance floor can be seen. `sigma(1y)` is deliberately NOT `FLAT_VOL` - this surface
#: separates nothing, it is monitored at three densities against its own oracle.
UP_VOLS = np.array([0.12, 0.15, 0.18, 0.21, 0.24, 0.26, 0.28])
TS = {True: (TS_KNOTS, TS_VOLS), 'up': (TS_KNOTS, UP_VOLS)}

#: the oracle read on a SEPARATE 4e6-path numpy implementation (scratch, seed 7, same allocation),
#: as (value, se). `_oracle` here is the torch one at 1e6 paths and must land on these - which is
#: what stops the oracle drifting into agreement with the pricer instead of with the definition.
ORACLE = {('Down_And_Out', True): (1019.81, 1.25), ('Down_And_In', True): (142.63, 0.41),
          ('Up_And_Out', True): (74.69, 0.19), ('Down_And_Out', False): (1002.38, 1.25),
          ('Down_And_In', False): (159.19, 0.43), ('Up_And_Out', False): (66.23, 0.17)}


def implied_vol(t, sloped, smile=0.0, moneyness=1.0):
    """The surface's own bilinear read, rebuilt OUTSIDE the engine. `gather_flat_surface`
    interpolates linearly in VOL on both axes for a non-Malz explicit surface, and the knots below
    are the ones `_surface` writes, so this is the same function the pricer will query.

    LINEARLY between the moneyness knots, not the quadratic that generated them - the surface is
    three points, and off a knot the engine returns the chord. The grid is a full product, so the
    bilinear read separates into `interp(t) + interp(m)`. At `smile = 0` the second term is
    identically zero and this is the old one-line form.

    `sloped` selects the term structure: False flat, True the separation surface, 'up' the density
    gate's 0.12 -> 0.24 one.
    """
    v = np.interp(t, *TS[sloped]) if sloped else FLAT_VOL
    return v + np.interp(moneyness, MONEYNESS, [smile * (m - 1.0) ** 2 for m in MONEYNESS])


def _surface(sloped, smile):
    tenors = TS[sloped][0] if sloped else np.array([0.02, 2.0])
    return utils.Curve([], [[m, float(t), float(implied_vol(t, sloped, smile, m))]
                            for m in MONEYNESS for t in tenors])


def _cfg(barrier_type, barrier, sloped, smile=0.0, strike=STRIKE, model_vol=0.0,
         obs_days=OBS_DAYS):
    """A monthly-monitored equity barrier by default. `model_vol = 0` holds every scenario path at
    SPOT, so base valuation reports the pricer's own inner Monte Carlo and nothing else - the vol
    under test is the PRICING surface, which is a different object from the scenario diffusion.

    `obs_days` is the monitoring density and it is a parameter because every barrier fixture in
    this repo is monthly, which is ten times clear of the variance floor - see the density gate."""
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': strike, 'Expiry_Date': BASE + pd.Timedelta(days=EXPIRY_DAY),
        'Units': UNITS, 'Barrier_Type': barrier_type, 'Barrier_Price': barrier,
        'Cash_Rebate': 0.0,
        'Barrier_Dates': [BASE + pd.Timedelta(days=d) for d in obs_days],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=1),
    }
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, R], [5.0, R]])},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.01, Q], [5.0, Q]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': _surface(sloped, smile)},
    }
    c.params['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': model_vol, 'Drift': 0.0}}
    c.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceModel')
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(field, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _price(c, mcmc=65536, seed=1):
    _, out = derivus.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d', 'Batch_Size': 1,
        'Simulation_Batches': 1, 'Random_Seed': seed, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'Deflation_Interest_Rate': 'USD', 'MCMC_Simulations': mcmc})
    return float(out['Results']['mtm'].values[0].mean())


# ------------------------------------------------------------------ the independent oracle

def _oracle(barrier_type, barrier, sloped, smile=0.0, strike=STRIKE, n_paths=1_000_000, sub=4,
            seed=7, device='cpu', obs_days=OBS_DAYS):
    """Fine-step MC under the instantaneous vol term structure the surface's total variance implies.

    Nothing from `derivus` is in this loop. The variance of each sub-interval is
    `V(t_k+1) - V(t_k)` with `V(t) = implied_vol(t)^2 * t` - the DEFINITION of an implied vol, and
    the quantity the pricer had wrong. Monitoring is a brute-force endpoint test on the observation
    dates only; expiry is an observation DATE (instruments.py unions it into the strip) but not a
    monitoring one (`Barrier_Dates` flags it -1), which the pricer does too and which a naive oracle
    gets wrong by 12% on an up-and-out.

    IT HAS NO VARIANCE FLOOR, which is the second thing it adjudicates: each interval is drawn at
    its own `dV`, however small, so a pricer that clamps one gets caught the moment the monitoring
    is dense enough for the clamp to bind.

    The surface is read at the DEAL'S DECLARED moneyness - `S/K`, one number for the whole strip -
    because that is the convention `forward_vol_strip` takes from the pricer. It is only ever
    called at `smile = 0`, where every convention reads the same number, so it cannot adjudicate
    that choice; the closed-form digital below does.
    """
    obs = np.union1d(np.array(obs_days, float), [EXPIRY_DAY])
    edges, monitor, prev = [0.0], [], 0.0
    for d in obs:
        edges.extend(prev + (d - prev) * k / sub for k in range(1, sub + 1))
        if d != EXPIRY_DAY:
            monitor.append(len(edges) - 1)
        prev = d
    t = np.array(edges) / 365.0
    sig = np.array([implied_vol(x, sloped, smile, SPOT / strike) for x in t])
    dV = np.diff(sig ** 2 * t)
    assert (dV > 0).all(), 'declining total variance - the fixture is arbitrageable'
    db = (R - Q) * np.diff(t)

    g = torch.Generator(device=device).manual_seed(seed)
    up, knock_in = 'Up' in barrier_type, 'In' in barrier_type
    logS = torch.full((n_paths,), float(np.log(SPOT)), dtype=DTYPE, device=device)
    hit = torch.zeros(n_paths, dtype=torch.bool, device=device)
    monitor = set(monitor)
    for k in range(len(dV)):
        z = torch.randn(n_paths, generator=g, dtype=DTYPE, device=device)
        logS += float(db[k] - 0.5 * dV[k]) + float(np.sqrt(dV[k])) * z
        if k + 1 in monitor:
            hit |= (logS > np.log(barrier)) if up else (logS < np.log(barrier))
    v = (logS.exp() - strike).clamp(min=0.0) * (hit if knock_in else ~hit).to(DTYPE)
    df = float(np.exp(-R * EXPIRY_DAY / 365.0))
    return (UNITS * df * float(v.mean()),
            UNITS * df * float(v.std()) / np.sqrt(n_paths))


def _black(sigma, strike=STRIKE, T=EXPIRY_DAY / 365.0):
    from math import erfc, exp, log, sqrt
    F, sd = SPOT * exp((R - Q) * T), sigma * sqrt(T)
    d1 = (log(F / strike) + 0.5 * sd * sd) / sd
    N = lambda x: 0.5 * erfc(-x / sqrt(2.0))
    return UNITS * exp(-R * T) * (F * N(d1) - strike * N(d1 - sd))


# ------------------------------------------------------------------------------- the gates

@pytest.mark.parametrize('barrier_type,barrier', [('Down_And_Out', 90.0), ('Down_And_In', 90.0),
                                                  ('Up_And_Out', 115.0)])
def test_two_surfaces_with_one_expiry_vol_must_separate(barrier_type, barrier):
    """THE GATE THE DEFECT DIES ON. Two surfaces agreeing at the deal's expiry must give DIFFERENT
    monitored barrier prices, and each must land on its own oracle.

    The separation is the whole statement: before the fix these two priced bitwise identically,
    because the single expiry vol is the only thing either surface was asked for. The oracle
    assertion is what stops the separation being satisfied by any wrong number that merely differs
    - a fix that separated them the WRONG way would pass a difference test and fail here.

    The tolerance is 4 combined standard errors of (oracle MC + pricer MC), not a fixed relative
    band: 4 seeds at 65536 inner paths against 1e6 oracle paths. The defect sits 12x (Down_And_In)
    and 44x (Up_And_Out) above that bar on the sloped surface, and the FLAT arm - where the defect
    is exactly correct - is scored too, so the gate cannot be passed by breaking the flat case.
    """
    for sloped in (False, True):
        ref, se = _oracle(barrier_type, barrier, sloped)
        pinned, pinned_se = ORACLE[(barrier_type, sloped)]
        assert abs(ref - pinned) < 4.0 * np.hypot(se, pinned_se), (
            'the oracle itself moved: {:.4f} +/- {:.4f} against the pinned {:.2f} +/- {:.2f} from '
            'an independent numpy implementation'.format(ref, se, pinned, pinned_se))
        seeds = [_price(_cfg(barrier_type, barrier, sloped), seed=s) for s in (1, 2, 3, 4)]
        got = float(np.mean(seeds))
        pricer_se = float(np.std(seeds, ddof=1)) / 2.0
        bar = 4.0 * np.hypot(se, pricer_se)
        assert abs(got - ref) < bar, (
            '{} on the {} surface: pricer {:.4f} vs oracle {:.4f} +/- {:.4f} (pricer se {:.4f}), '
            'off by {:.3%} against a {:.4f} bar'.format(
                barrier_type, 'sloped' if sloped else 'flat', got, ref, se, pricer_se,
                got / ref - 1.0, bar))

    flat = float(np.mean([_price(_cfg(barrier_type, barrier, False), seed=s) for s in (1, 2, 3, 4)]))
    slope = float(np.mean([_price(_cfg(barrier_type, barrier, True), seed=s) for s in (1, 2, 3, 4)]))
    assert abs(slope / flat - 1.0) > 0.01, (
        '{}: the two surfaces priced within {:.4%} of each other. They carry the same 1y implied '
        'vol and different term structures, so a pricer that reads the surface only at expiry '
        'CANNOT tell them apart - which is the defect.'.format(barrier_type, slope / flat - 1.0))


def test_a_flat_surfaces_forward_variance_is_sigma_squared_dt():
    """THE CONTROL. Differencing a flat strip must give back `sigma^2 * dt` - but NOT bitwise.

    `forward_vol_rate` computes `(sigma^2 T_j - sigma^2 T_j-1) / dt_j`, and the rounding of the
    cumulative times is amplified by `T_j / dt_j` in that difference. On this fixture's strip -
    twelve monthly observations and then five days to expiry - that ratio reaches 73 on the last
    interval, so the bound asserted here is 2 * max(T_j/dt_j) steps and not zero. MEASURED: the
    strip runs 0, 0, 0, 0, 1, 2, 1, 2, 2, 4, 2, 2, 10 float64 steps from sigma against a bound of
    146, worst on exactly the interval with the largest amplification. In float32 the same strip
    reads 5 steps, which is the same relative error - this is a property of the arithmetic, not of
    the precision.

    A gate written with `torch.equal` here would be a placebo in the other direction - it would
    fail on correct code the day a fixture's dates stop dividing evenly.
    """
    days = np.array(OBS_DAYS + [EXPIRY_DAY], float)
    cum = torch.tensor(days / 365.0, dtype=DTYPE)
    dt = torch.cat([cum[:1], cum.diff()])
    vols = torch.full((len(days), 8), FLAT_VOL, dtype=DTYPE)

    got = pricing.forward_vol_rate(vols, cum, dt)
    bound = 2.0 * float((cum / dt).max())
    steps = ((got - vols).abs() / torch.tensor(np.spacing(FLAT_VOL), dtype=DTYPE)).max()
    assert bound > 1.0, 'the fixture has no amplification - the bound below is vacuous'
    assert float(steps) <= bound, (
        'a flat strip differenced to {} float64 steps from sigma, bound {:.1f} (max T_j/dt_j = '
        '{:.1f})'.format(float(steps), bound, float((cum / dt).max())))
    assert not torch.equal(got, vols) or float((cum / dt).max()) < 2.0, (
        'the flat strip came back BITWISE equal, so this fixture cannot demonstrate the '
        'amplification the docstring claims and the bound above is untested')


def test_the_interval_vol_strip_is_the_difference_of_cumulative_variances():
    """`forward_vol_rate` against the definition, with no pricer and no market data.

    An implied vol is cumulative, so the interval's variance is `(sigma_j^2 T_j - sigma_j-1^2
    T_j-1)` and its ANNUALISED vol is that over `dt_j`. Two statements are asserted: the strip is
    that difference, and its variance SUMS BACK to the total `sigma_N^2 T_N` - the telescoping
    property that is exactly why the defect was invisible to every European gate in the repo.

    Rank-polymorphic, on the same rule as `forward_carry_rate`: one MTM row and a whole block must
    be the same expression, which is what lets one function serve the barrier's block-shaped call
    site and the TARF's.
    """
    g = torch.Generator().manual_seed(5)
    dt = torch.rand(4, 9, generator=g, dtype=DTYPE) * 0.2 + 0.02
    cum = dt.cumsum(dim=-1)
    # an ARBITRAGE-FREE surface: total variance rises, so the clamp is not exercised and the
    # reference below is a real square root rather than a floor. The clamp has its own gate.
    total = (torch.rand(4, 9, 6, generator=g, dtype=DTYPE) * 0.04 + 0.01).cumsum(dim=-2)
    vols = torch.sqrt(total / cum.unsqueeze(-1))

    block = pricing.forward_vol_rate(vols, cum, dt)
    assert block.shape == vols.shape
    for row in range(4):
        assert torch.equal(block[row], pricing.forward_vol_rate(vols[row], cum[row], dt[row])), (
            'row %d: the block form and the row form are different expressions' % row)

    cum_var = vols ** 2 * cum.unsqueeze(-1)
    want = torch.sqrt(cum_var.diff(dim=-2) / dt[..., 1:].unsqueeze(-1))
    assert torch.allclose(block[..., 1:, :], want, rtol=0, atol=1e-15), (
        'the strip is not the difference of cumulative variances')
    assert torch.equal(block[..., :1, :], vols[..., :1, :]), (
        'the FIRST interval is the cumulative window itself and must be read straight off')
    summed = (block ** 2 * dt.unsqueeze(-1)).sum(dim=-2)
    assert torch.allclose(summed, cum_var[..., -1, :], rtol=1e-14, atol=0.0), (
        'the strip does not telescope back to the total variance - which is the property that '
        'made the defect invisible, so losing it would be a different defect')

    # a DECLINING cumulative variance is floored, not square-rooted negative. `pv_MC_Tarf` has
    # carried this clamp since it was written and the hoist preserves it term for term.
    falling = vols.clone()
    falling[..., 4, :] *= 3.0
    out = pricing.forward_vol_rate(falling, cum, dt)
    assert torch.isfinite(out).all(), 'a declining total variance produced a NaN rather than a floor'
    assert float(out[..., 5, :].max()) <= float(np.sqrt(np.finfo(np.float64).eps /
                                                        float(dt[..., 5].min()))), (
        'the floored interval is not at the eps floor')

    # zero-length intervals - a fixing the row has already observed - divide by one, not by zero
    dt0 = dt.clone()
    dt0[..., 3] = 0.0
    assert torch.isfinite(pricing.forward_vol_rate(vols, cum, dt0)).all(), (
        'a zero-length interval divided by zero')


def test_the_strip_reads_the_moneyness_the_deal_declares():
    """THE HALF A TERM-STRUCTURE FIXTURE CANNOT SEE, and the convention it settles.

    The strip used to read every fixing at its own FORWARD moneyness, hard-coded, mirroring
    `pv_MC_Tarf`. This pricer is not the TARF: it HAS a European limit and an in-out-parity leg,
    both of which read the deal's DECLARED moneyness (`use_forwards = False` on these fixtures), so
    a forward read made the simulation step a law the pricer's own European legs disagreed with -
    +0.948% and 8.3 standard errors on the never-knocking limit, -11.03 on parity. The strip now
    takes the deal's flag; the per-fixing smile read is the open modelling question in `roadmap.md`.

    THE STATEMENT HERE IS EXACT, not statistical. `SPOT == STRIKE` and the scenario spot is held
    there (`model_vol = 0`), so the declared moneyness is 1.0 - a knot of this surface - at EVERY
    fixing, and a smiley surface must therefore price BITWISE IDENTICALLY to the smile-free one.
    Measured: 0.0 difference on all ten seeds for both barrier types. Under the forward read the
    same paired difference is +5.876 +/- 0.016 and -1.110 +/- 0.003, so the mutation dies by 370
    standard errors and this gate needs no tolerance at all.

    `r = q` would freeze the forward on the strike and make the whole statement vacuous, which is
    why the fixture's carry is asserted before its price. The POSITIVE half - that the strip reads
    the surface at all, and at the right value - is
    `test_the_never_knocking_limit_is_black_at_the_expiry_vol`, which is parametrised on the smile
    for exactly this reason, and the separation gates above.
    """
    assert float(np.exp((R - Q) * EXPIRY_DAY / 365.0)) > 1.03, (
        'r = q would freeze the moneyness and this gate would read nothing')
    assert SPOT == STRIKE, 'the declared moneyness is not on a knot and the exactness below is lost'

    seeds = tuple(range(1, 11))
    for barrier_type, barrier in [('Down_And_Out', 90.0), ('Up_And_Out', 115.0)]:
        d = np.array([_price(_cfg(barrier_type, barrier, False, smile=SMILE), seed=s) -
                      _price(_cfg(barrier_type, barrier, False), seed=s) for s in seeds])
        assert not d.any(), (
            '{}: the smile moved the paired price by up to {:.4f}. This deal declares '
            '`use_forwards = False` and sits exactly on the m = 1 knot, so the strip must read the '
            'same vol on both surfaces; at the fixing\'s own FORWARD moneyness it reads {:+.3f}, '
            'which is the convention this gate pins.'.format(
                barrier_type, np.abs(d).max(), 5.876 if barrier < SPOT else -1.110))


@pytest.mark.parametrize('sloped', [False, True])
@pytest.mark.parametrize('smile', [0.0, SMILE])
def test_the_never_knocking_limit_is_black_at_the_expiry_vol(sloped, smile):
    """THE EUROPEAN INVARIANT THAT MUST NOT MOVE. Put the barrier out of reach and the pricer is
    valuing a plain call, which must be Black at the surface's own DECLARED `(K, T)` read - on all
    four surfaces, the two term structures agreeing there by construction.

    This is what makes the two-quantity split a claim rather than a hope: the simulation steps a
    strip whose entries differ from the expiry vol by up to a factor of 1.36, and the terminal
    distribution has to come back to the same total variance anyway. The term-structure defect
    passed this test too, which is the point - it is here to catch the FIX breaking a European.

    PARAMETRISED ON THE SMILE, which it was not, and that omission is how the moneyness convention
    shipped unmeasured for a release: on a smiley surface a strip read at the fixings' forwards
    puts the terminal law at sigma(F/K, T) = 0.250757 while every European leg of the same pricer
    marks at sigma(S/K, T) = 0.247900, and the pricer reads 1174.80 against Black's 1163.96 -
    +0.948%, 8.3 standard errors. The smile-free arms cannot see that at all.

    THE BAR IS THE PRICER'S OWN MONTE CARLO ERROR, measured over the ten seeds it reports, not a
    fixed relative band: at 65536 inner paths the standard error of the mean is 1.30 on a value of
    1164 (1.1e-3 relative), and a four-seed reading pinned to a 2e-4 band failed on noise alone.
    Measured agreement is +0.016% (flat) and +0.018% (sloped), 0.1 standard errors, on both smiles.
    """
    ref = _black(implied_vol(EXPIRY_DAY / 365.0, sloped, smile, SPOT / STRIKE))
    p = np.array([_price(_cfg('Down_And_Out', 20.0, sloped, smile=smile), seed=s)
                  for s in range(1, 11)])
    got, se = float(p.mean()), float(p.std(ddof=1)) / np.sqrt(len(p))
    assert abs(got - ref) < 4.0 * se, (
        'never-knocking Down_And_Out on the {} surface (smile {}): {:.5f} +/- {:.5f} vs Black at '
        'the DECLARED sigma(S/K, 1y) = {:.6f} -> {:.5f} ({:+.4%}, {:.1f} standard errors). At the '
        'fixings\' own forward moneyness this reads {:.5f}.'.format(
            'sloped' if sloped else 'flat', smile, got, se,
            implied_vol(1.0, sloped, smile, SPOT / STRIKE), ref, got / ref - 1.0,
            abs(got - ref) / se,
            _black(implied_vol(1.0, sloped, smile, SPOT * np.exp(R - Q) / STRIKE))))


@pytest.mark.parametrize('sloped', [False, True])
@pytest.mark.parametrize('smile', [0.0, SMILE])
def test_in_out_parity_holds_on_every_surface(sloped, smile):
    """KNOCK-OUT + KNOCK-IN = THE VANILLA, on all four surfaces, and it is two statements.

    The ANALYTIC half is free: `sim_spot_oss` prices BARRIER_IN by parity off `sd_to_expiry`, so
    KO + KI reproduces Black at the declared quote to 2.2e-12 absolute (1.9e-15 relative) on the
    same seed, whatever the strip does. That is asserted because it is the leg's construction -
    it dies the moment the KI vanilla is marked at anything other than the deal's own European
    quote - and it carries no Monte Carlo error at all.

    The MONTE CARLO half is the one with content: the third leg is the SIMULATED vanilla, the same
    deal with its barrier pushed out of reach, which prices the terminal law the STRIP produces.
    Parity therefore says the strip's total variance must come back to the European quote, and the
    forward-moneyness read broke it on the smiley surfaces by -11.03 on ten seeds, 8.4 standard
    errors, where it reads -0.19 +/- 1.30 here. It is the same defect the never-knocking gate above
    catches, reached through the KI leg's own construction rather than through Black.
    """
    seeds = tuple(range(1, 11))
    ko = np.array([_price(_cfg('Down_And_Out', 90.0, sloped, smile=smile), seed=s) for s in seeds])
    ki = np.array([_price(_cfg('Down_And_In', 90.0, sloped, smile=smile), seed=s) for s in seeds])
    van = np.array([_price(_cfg('Down_And_Out', 20.0, sloped, smile=smile), seed=s) for s in seeds])
    black = _black(implied_vol(EXPIRY_DAY / 365.0, sloped, smile, SPOT / STRIKE))

    resid = ko + ki - black
    assert np.abs(resid).max() < 1e-9 * black, (
        'KO + KI is not the declared European: {:+.4e} on a value of {:.4f}. The knock-in leg is '
        'priced BY parity off `sd_to_expiry`, so this is arithmetic, not Monte Carlo.'.format(
            float(np.abs(resid).max()), black))

    d = ko + ki - van
    got, se = float(d.mean()), float(d.std(ddof=1)) / np.sqrt(len(seeds))
    assert abs(got) < 4.0 * se, (
        'in-out parity against the SIMULATED vanilla on the {} surface (smile {}): {:+.4f} +/- '
        '{:.4f}, {:.1f} standard errors. The strip\'s terminal variance has to telescope back to '
        'the quote the parity leg marks at.'.format(
            'sloped' if sloped else 'flat', smile, got, se, abs(got) / se))


# ------------------------------------------------------------------ the monitoring DENSITY gate
#: daily monitoring, expiry excluded - `instruments.py` unions the expiry into the observation
#: strip and flags it as a non-monitoring date, which is what `_oracle` reproduces.
DAILY_DAYS = list(range(1, EXPIRY_DAY))
WEEKLY_DAYS = list(range(7, EXPIRY_DAY, 7))
#: the daily oracle, pinned as (value, se) from 8e6 paths. Two INDEPENDENT streams stand behind
#: each: CUDA (906.906 / 9.6050) and CPU (907.677 / 9.5764), 0.9 and 1.1 standard errors apart, so
#: the gate below cannot be satisfied by a device.
DENSITY_ORACLE = {('Down_And_Out', 90.0): (907.29, 0.60), ('Up_And_Out', 110.0): (9.591, 0.026)}


def floored_intervals(obs_days, sloped='up'):
    """How many of this strip's intervals carry `sigma_fwd^2 * dt < 1e-4`, from the SURFACE alone.

    No engine, no Monte Carlo: the forward variance of `[T_j-1, T_j]` is the difference of the
    surface's cumulative variances, which is the same definition `forward_vol_rate` implements and
    `_oracle` steps. It is the anti-placebo statement of the gate below - a fixture whose intervals
    never reach the floor cannot see the floor, and EVERY barrier fixture in this repo is one.
    """
    t = np.union1d(np.array(obs_days, float), [EXPIRY_DAY]) / 365.0
    cum_var = np.array([implied_vol(x, sloped, 0.0, SPOT / STRIKE) ** 2 * x for x in t])
    var = np.diff(np.r_[0.0, cum_var])
    return int((var < 1e-4).sum()), len(var)


@pytest.mark.parametrize('barrier_type,barrier', [('Down_And_Out', 90.0), ('Up_And_Out', 110.0)])
def test_a_daily_monitored_barrier_is_not_priced_at_the_variance_floor(barrier_type, barrier):
    """THE DENSITY GATE. `sim_spot_oss` floors every interval's variance at 1e-4, which binds
    whenever `sigma_fwd < 0.01/sqrt(dt)` - 19.1% annualised at daily monitoring - so a correct
    forward-variance strip is silently re-inflated on the short end of any upward term structure.

    THE FIXTURE THAT PROVES THE STRIP HIDES THIS. Under the pre-strip defect every interval carried
    `sigma(T)^2`, the LARGEST vol on an upward surface, and the floor essentially never bound; the
    corrected strip hands it genuinely small forward variances. And every barrier fixture in this
    repo - including the ones above - is MONTHLY, which is ten times clear of the floor. Measured
    on this surface: 114 of 365 daily intervals are floored, 0 of 53 weekly and 0 of 13 monthly,
    and at those two densities the pricer is BITWISE identical with the floor conditional,
    unconditional, or removed. The gate had to be given a density before it could see anything.

    WHAT IT COSTS, against the floor-free oracle, eight seeds: `Down_And_Out` +1.584% and
    `Up_And_Out` -6.870% with the spelling that shipped (unconditional floor AND the drift taken
    from the unclamped variance), +0.183% and +0.403% with the floor conditioned on the zero-length
    step and both consumers reading one `var`.

    BOTH ARMS ARE LOAD-BEARING AND THEY DO NOT DIE TOGETHER. The floor is a variance ADDED to the
    short end, which knocks an up-and-out out much harder than a down-and-out: the floor ALONE reads
    -5.576% here and only -0.13% there, inside this gate's own bar - so the `Up_And_Out` arm is the
    one that kills an unconditional floor, and the `Down_And_Out` arm is the one that kills the
    incoherent drift on top of it (+1.584%) and the collapsed-strip mutants. Neither arm is
    redundant and the two halves of the shipped defect partly CANCEL on the down-and-out, which is
    the reason a one-armed version of this gate would have scored the floor as harmless.

    THE FLOOR STAYS AT `dt == 0`, and that is not a compromise: it is what the floor is FOR. A
    reporting row that is itself an observation date has a zero-length first interval, and removing
    the clamp there leaves every VALUE finite and turns 11 of the 13 CVA-gradient entries NaN on
    `test_recompute_equity_pricers`'s fixture (7 of 13 on its averaging autocall) - `sqrt` has an
    infinite derivative at zero and the variance carries the surface's graph. The autocall's own
    comment said so ("to prevent gradients from blowing up") without saying which intervals needed
    it. These do not: at `dt > 0` that same fixture is bit-exact with the floor unconditional.

    The bar is 4 combined standard errors, oracle plus pricer, as everywhere else in this file.
    """
    n_floored, n_total = floored_intervals(DAILY_DAYS)
    assert n_floored > 50, (
        'only {} of {} daily intervals reach the 1e-4 floor - this fixture cannot see the '
        'quantity under test'.format(n_floored, n_total))
    assert floored_intervals(WEEKLY_DAYS)[0] == 0 and floored_intervals(OBS_DAYS)[0] == 0, (
        'the weekly/monthly strips now reach the floor too, so the density claim above is not what '
        'this fixture measures any more')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ref, se = _oracle(barrier_type, barrier, 'up', n_paths=8_000_000, sub=1,
                      obs_days=DAILY_DAYS, device=device)
    pinned, pinned_se = DENSITY_ORACLE[(barrier_type, barrier)]
    assert abs(ref - pinned) < 4.0 * np.hypot(se, pinned_se), (
        'the daily oracle itself moved: {:.4f} +/- {:.4f} against the pinned {:.3f} +/- '
        '{:.3f}'.format(ref, se, pinned, pinned_se))

    seeds = tuple(range(1, 9))
    p = np.array([_price(_cfg(barrier_type, barrier, 'up', obs_days=DAILY_DAYS), seed=s)
                  for s in seeds])
    got, pricer_se = float(p.mean()), float(p.std(ddof=1)) / np.sqrt(len(seeds))
    bar = 4.0 * np.hypot(se, pricer_se)
    assert abs(got - ref) < bar, (
        '{} monitored daily on the upward term structure: pricer {:.4f} +/- {:.4f} vs a floor-free '
        'oracle {:.4f} +/- {:.4f}, off by {:+.3%} against a {:.4f} bar. {} of {} intervals carry '
        'less than 1e-4 of variance, so an unconditional floor re-inflates them.'.format(
            barrier_type, got, pricer_se, ref, se, got / ref - 1.0, bar, n_floored, n_total))


def test_a_daily_monitored_european_limit_is_black_at_the_expiry_vol():
    """THE DENSITY GATE'S EUROPEAN ARM, and its reference is ARITHMETIC rather than an oracle.

    A floored interval is variance ADDED to the terminal law, so pushing the barrier out of reach
    at a density where the floor binds turns the whole question into a European: the pricer must
    still be Black at the surface's own `(K, T)` quote. From this surface alone, with no Monte
    Carlo anywhere, the 114 floored intervals add 0.003654 to a total variance of 0.057600 - 6.34%,
    which is terminal vol 0.247496 against the declared 0.240000 and Black +2.506%.

    WHY IT IS HERE AND THE ORACLE ARM ABOVE IS NOT ENOUGH. That gate's `Down_And_Out` arm passes an
    unconditional floor at +0.049%, because on a down-and-out the floor's added variance (-0.13%)
    and the incoherent drift (+1.7%) partly cancel; only its `Up_And_Out` arm kills the floor, at
    -5.576%. This one kills BOTH spellings on the same fixture the two-surface gates already use,
    against an exact reference and with no 8e6-path oracle to run. MEASURED over the same ten
    seeds: the fix +0.020% (0.3 se), an unconditional floor with a coherent drift +2.536% (37.5
    se), and the spelling that shipped - unconditional floor AND drift off the unclamped variance -
    +3.516% (51.9 se). Dropping the DIFFERENCING reads -18.897% (405 se) here, where the monthly
    European gates cannot see it at all; collapsing the strip to the expiry vol passes at 0.7 se,
    which is right and is the whole reason that defect shipped.

    THE ANTI-PLACEBO IS ASSERTED, not described: the floored census and the Black price at the
    inflated variance are both computed here, and the gate refuses to run if the second is not
    many bars away from the first - a fixture whose intervals never reach the floor would make
    this test agree with everything.
    """
    n_floored, n_total = floored_intervals(DAILY_DAYS)
    t = np.union1d(np.array(DAILY_DAYS, float), [EXPIRY_DAY]) / 365.0
    var = np.diff(np.r_[0.0, np.array([implied_vol(x, 'up') ** 2 * x for x in t])])
    added = float(np.where(var < 1e-4, 1e-4 - var, 0.0).sum())
    total = float(var.sum())

    ref = _black(implied_vol(EXPIRY_DAY / 365.0, 'up'))
    inflated = _black(np.sqrt(total + added))
    p = np.array([_price(_cfg('Down_And_Out', 20.0, 'up', obs_days=DAILY_DAYS), seed=s)
                  for s in range(1, 11)])
    got, se = float(p.mean()), float(p.std(ddof=1)) / np.sqrt(len(p))

    assert n_floored > 50 and abs(inflated - ref) > 20.0 * se, (
        'this fixture cannot see the floor: {} of {} intervals floored, and the {:.4%} of variance '
        'a floor would add moves Black by {:.4f} against a standard error of {:.4f}'.format(
            n_floored, n_total, added / total, inflated - ref, se))
    assert abs(got - ref) < 4.0 * se, (
        'never-knocking Down_And_Out monitored daily on the upward term structure: {:.4f} +/- '
        '{:.4f} vs Black at sigma(S/K, 1y) = {:.6f} -> {:.4f} ({:+.4%}, {:.1f} standard errors). '
        '{} of {} intervals carry less than 1e-4 of variance; flooring them puts the terminal law '
        'at {:.6f} and Black at {:.4f}, which is {:+.4%}.'.format(
            got, se, implied_vol(1.0, 'up'), ref, got / ref - 1.0, abs(got - ref) / se,
            n_floored, n_total, np.sqrt(total + added), inflated, inflated / ref - 1.0))


# ------------------------------------------------------- the third adopter: pv_MC_AutoCallSwap
#: quarterly coupons on the same year and the same two surfaces the barrier separates on. A
#: threshold ABOVE the forward is the case with vega in it: an at-the-money trigger is nearly
#: vol-blind and would make every autocall gate below a placebo.
AC_DAYS = [91, 182, 273, EXPIRY_DAY]
AC_THRESHOLD, AC_COUPON = 1.02, 0.05


def _ac_cfg(days, threshold, sloped, smile=0.0, coupon=AC_COUPON):
    """The barrier fixture's market with an autocall in place of the barrier - same surface, same
    r and q, same spot. `no_averaging` is selected by one `Price_Fixing` per `Autocall_Coupon`,
    which is what puts this on the one-step-survival branch."""
    dates = [BASE + pd.Timedelta(days=d) for d in days]
    c = _cfg('Down_And_Out', 90.0, sloped, smile)
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument({
        'Object': 'QEDI_CustomAutoCallSwap', 'Reference': 'AC1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': STRIKE, 'Expiry_Date': dates[-1], 'Units': 1.0,
        'Settlement_Style': 'Cash', 'Option_On_Forward': 'No', 'Option_Style': 'European',
        'Barrier': 0.0, 'Payoff_Type': None, 'Barrier_Dates': [], 'Autocall_Floating': [],
        'Price_Fixing': [[d, 0.0] for d in dates],
        'Autocall_Coupons': [[d, coupon] for d in dates],
        'Autocall_Thresholds': [[d, threshold] for d in dates]}, {})}]
    return c


def _ac_price(c, mcmc=65536, seed=1):
    _, out = derivus.run_baseval(c, prec=DTYPE, overrides={
        'MCMC_Simulations': mcmc, 'Random_Seed': seed, 'Greeks': 'No'})
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'AC1']['Value'].iloc[0])


def _ac_oracle(days, threshold, sloped, smile=0.0, coupon=AC_COUPON, n_paths=2_000_000, sub=8,
               seed=7):
    """Brute-force autocall MC, sharing nothing with the pricer: no `derivus` in the loop, no
    one-step survival, no inverse-CDF truncated draw. Paths are stepped and the FIRST fixing that
    closes at or above its threshold redeems the note.

    The law is the surface's own - `V(T_j) = sigma(F_j/K, T_j)^2 T_j` read once per fixing, linearly
    interpolated in between and DIFFERENCED per sub-step, which is the quantity under test written
    independently. The observation-date marginals are exact for any `sub` (the interpolation is
    within an interval and the endpoints are the reads), so sub-stepping buys realism of the path,
    not accuracy of the answer.

    IT TAKES THE DEAL'S DECLARED MONEYNESS, `S/K`, which is what `forward_vol_strip` now reads -
    but it is only ever called at `smile = 0`, where the surface is flat in moneyness and every
    convention reads the same number, so it cannot adjudicate that choice. The moneyness half is
    settled by the closed-form digital below, not here.
    """
    T = np.array(days, float) / 365.0
    V_fix = np.array([implied_vol(x, sloped, smile, SPOT / STRIKE) ** 2 * x for x in T])
    edges, prev = [0.0], 0.0
    for x in T:
        edges.extend(prev + (x - prev) * k / sub for k in range(1, sub + 1))
        prev = x
    grid = np.array(edges)
    dV = np.diff(np.interp(grid, np.r_[0.0, T], np.r_[0.0, V_fix]))
    assert (dV > 0).all(), 'declining total variance - the fixture is arbitrageable'
    db = (R - Q) * np.diff(grid)

    g = torch.Generator().manual_seed(seed)
    logS = torch.full((n_paths,), float(np.log(SPOT)), dtype=DTYPE)
    alive, pv = torch.ones(n_paths, dtype=DTYPE), torch.zeros(n_paths, dtype=DTYPE)
    for k in range(len(dV)):
        z = torch.randn(n_paths, generator=g, dtype=DTYPE)
        logS += float(db[k] - 0.5 * dV[k]) + float(np.sqrt(dV[k])) * z
        if (k + 1) % sub == 0:
            j = (k + 1) // sub - 1
            fire = (logS >= np.log(threshold * STRIKE)).to(DTYPE) * alive
            pv += fire * coupon * float(np.exp(-R * T[j]))
            alive = alive - fire
    return float(pv.mean()), float(pv.std()) / np.sqrt(n_paths)


def _cash_digital(sigma, threshold, coupon=AC_COUPON, T=EXPIRY_DAY / 365.0):
    """`coupon * P(S_T >= threshold*K) * DF` under the same lognormal law the pricer's `norm_cdf`
    survival probability is."""
    from math import erfc, exp, log, sqrt
    K = threshold * STRIKE
    d2 = (log(SPOT / K) + (R - Q - 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    return coupon * 0.5 * erfc(-d2 / sqrt(2.0)) * exp(-R * T)


def test_the_autocalls_two_surfaces_must_separate():
    """THE SIBLING GATE. `pv_MC_AutoCallSwap` is the seam's third adopter and had NO fixture that
    could see the strip at all - every autocall fixture in this repo is flat-surface and `r = q = 0`.

    Same two surfaces, same statement: they agree at the deal's expiry, so the pre-port pricer -
    one implied vol at the expiry tenor on every interval - priced them BITWISE IDENTICALLY.
    MEASURED on this fixture, four seeds at 65536 inner paths: defect 0.034026629996 on both
    surfaces (`array_equal` True), against a sloped oracle of 0.037094, i.e. **-8.27%** and 207
    standard errors off. The strip reads 0.037064, -0.081% and 2.0 se. The separation itself is
    +8.93%, which is where the barrier's four monthly-monitored percent go when the monitoring is
    quarterly and the payoff is a digital rather than a knock-out on a call.

    Both arms are scored, so a fix that separated them by breaking the FLAT case fails here: on the
    flat surface the defect is exactly correct and reads -0.046% (1.0 se).
    """
    got = {}
    for sloped in (False, True):
        ref, se = _ac_oracle(AC_DAYS, AC_THRESHOLD, sloped)
        seeds = [_ac_price(_ac_cfg(AC_DAYS, AC_THRESHOLD, sloped), seed=s) for s in (1, 2, 3, 4)]
        got[sloped] = float(np.mean(seeds))
        pricer_se = float(np.std(seeds, ddof=1)) / 2.0
        bar = 4.0 * np.hypot(se, pricer_se)
        assert abs(got[sloped] - ref) < bar, (
            'autocall on the {} surface: pricer {:.6f} vs oracle {:.6f} +/- {:.6f} (pricer se '
            '{:.6f}), off by {:.3%} against a {:.6f} bar'.format(
                'sloped' if sloped else 'flat', got[sloped], ref, se, pricer_se,
                got[sloped] / ref - 1.0, bar))

    assert abs(got[True] / got[False] - 1.0) > 0.02, (
        'the two surfaces priced the autocall within {:.4%} of each other. They carry the same 1y '
        'implied vol and different term structures, so a pricer reading the surface only at expiry '
        'CANNOT tell them apart - which is the defect.'.format(got[True] / got[False] - 1.0))


@pytest.mark.parametrize('smile', [0.0, SMILE])
def test_the_autocalls_one_coupon_limit_is_the_closed_form_digital(smile):
    """THE AUTOCALL'S OWN EUROPEAN LIMIT, and it carries NO MONTE CARLO ERROR.

    One coupon, on the expiry date. `sim_spot`'s survival probability is `norm_cdf` of a closed
    form, and with a single fixing no draw ever advances `Sj` - so the reported value IS a
    cash-or-nothing digital, exactly, and the vol the strip reads is measured to the last bit
    rather than through a Monte Carlo. `forward_vol_rate`'s `j == 0` branch takes that read
    straight, so this pins the READ and nothing else.

    WHICH READ, MEASURED. On a surface flat in tenor the term-structure half is a no-op and only
    the moneyness convention is left. The strip reads at the moneyness the DEAL declares, and this
    deal declares `use_forwards = False` - the SPOT read, the same quote `pv_MC_AutoCallSwap` marks
    its own European with. At `r - q = 4%` over the year the two candidate reads are 0.247900 and
    0.250757, and the two prices they give are exact:

        declared (spot moneyness) 0.024490310460   <- what ships, exact to 0 ULP
        strip at the FORWARD       0.024428368300   <- what shipped before the threading, exact
        difference                     -0.2529%

    Both are asserted: the pricer must equal the DECLARED read bitwise on both smiles, and on the
    smiley surface the forward read must be the measured distance away - a distance that exists in
    this file so the open modelling question in `roadmap.md` can be taken with a number in hand,
    and that dies the moment the strip is hard-coded back to the forward.

    THE FORWARD READ IS NOT WRONG, it is a different model: a desk quoting sticky-FORWARD moneyness
    wants it. What it cannot be is silently inconsistent with the European quote the same pricer
    marks with, which is what internal consistency decided here.
    """
    T = EXPIRY_DAY / 365.0
    sigma_fwd = implied_vol(T, False, smile, SPOT * np.exp((R - Q) * T) / STRIKE)
    sigma_spot = implied_vol(T, False, smile, SPOT / STRIKE)
    got = _ac_price(_ac_cfg([EXPIRY_DAY], 1.0, False, smile), mcmc=64)

    declared = _cash_digital(sigma_spot, 1.0)
    assert got == declared, (
        'the one-coupon autocall is not the closed-form digital at the DECLARED read: {:.12f} vs '
        '{:.12f} at sigma = {:.6f}'.format(got, declared, sigma_spot))
    forward = _cash_digital(sigma_fwd, 1.0)
    gap = forward / declared - 1.0
    if smile:
        assert sigma_fwd > sigma_spot + 1e-4, 'the fixture has no smile to read'
        assert -0.0026 < gap < -0.0024, (
            'a forward-moneyness strip would be {:+.6%} from the deal\'s DECLARED European quote, '
            'where -0.252925% was measured. That distance is the open convention decision in '
            'roadmap.md, not noise - if it moved, say which way and why.'.format(gap))
    else:
        assert forward == declared, (
            'with no smile the forward read and the spot read are the same number, so the two '
            'must agree BITWISE: {:.17g} vs {:.17g}'.format(forward, declared))
