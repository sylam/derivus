"""`QuadraticCarryCurveModel`: the two-factor continuous carry curve, and the claim it is built on
- that two knots holding the AVERAGE carry to maturity reproduce the quadratic log-futures curve
through the forward reader that already exists, with no new read code.

What this file gates:

  * THE REPRESENTATION. Three synthetic listed futures -> (S, two z-knots) -> `DerivedForwardCurve`
    reproduces all three at their true tenors and an off-knot tenor besides, through the real
    gather stack (`calc_time_grid_curve_rate` + `TensorBlock.interpolate_curve`), with a live repo
    leg beside the carry so the two cannot be reading each other.
  * ...AND WHERE IT STOPS. `utils.CurveTenor.get_index` CLIPS a query to the knot range, so the
    read outside the bracket is FLAT in z rather than the linear extrapolation the algebra would
    give. That is pinned as behaviour, with its measured size, because it is the one constraint the
    market data has to honour: the knots must bracket every (row date, query date) pair.
  * THE DYNAMICS, against an independently-structured reference loop in float64 - `torch.equal`,
    not `allclose` (see `_reference` for why it is torch and not numpy).
  * `num_factors() == len(correlation_name[1])`, the invariant nothing asserted.
  * THE INITIAL STATE is the market curve, recovered rather than declared, and row 0 IS that curve.
  * THE FORK: the state a fork recovers at row t is the outer path's state at row t. This process
    publishes no private buffer key and its fork verbs are the base no-ops, so that recovery IS the
    fork protocol and this is the gate that holds it.
  * NEAR-UNIT-ROOT HONESTY: phi_L = 0.9962 is 2.3 s.e. from a unit root, so the documented caveat
    that nothing may lean on carry reversion for value is made checkable - the conditional mean
    gives up 21% of a level deviation over a quarter, and the simulated mean agrees.
  * The calibration recovers a block it generated, and takes its `Reference_Tenors` from the
    archive column labels rather than a constant of its own.

ANTI-PLACEBO - the fixture property each gate needs, and what goes blind without it.

| property | value | what goes blind without it |
|---|---|---|
| `a` (curve curvature) | -0.0022 (D = 0.0068 <> 0) | at `a = 0` z is CONSTANT in tau, the average carry and the instantaneous slope coincide, and the whole knot-convention question disappears - the labels-bug mutant survives every gate |
| `Gamma` | -0.451 | the sample's own t-MLE gives -0.019 (see the calibration docstring: the loading is a TAIL phenomenon and its size is a function of nu). At -0.019 the dropped-Gamma mutant moves the shape by ~1e-5 and reads as noise; the fixture uses the Gaussian-loss value so the mutation is visible |
| `Mu_L`, `Mu_D` | 0.0169, -0.00054, both <> the initial state | at `Mu = L_0` the reversion term is identically zero and a mutant that drops the mean survives |
| `L_0`, `D_0` | 0.0329, 0.0068, both <> Mu | the same statement from the other side: at `L_0 = Mu_L` the "initial state ignored" mutant is invisible |
| `Phi_L` <> `Phi_D` | 0.9962 vs 0.9468 | one phi for both factors and a swapped-coefficient mutant survives |
| `Calibration_DT_Years` | 1/252 against a CALENDAR daily grid | at f = 1 the clock rescale is the identity and a mutant that ignores `Calibration_DT_Years` is a no-op |
| knots | dated, and NOT at the reference tenors | knots AT tau_A/tau_B give `k[0] = (-1/2, +1/2)`, where L is the plain average of the two knots whatever the shape coordinate is - so a wrong k SCALE (which is exactly the instantaneous-convention mutant) would leave the level exact and only the shape wrong, halving what every gate here can see |
| rows | T = 40, both modes, B > 1 | one row leaves every recursion at its seed; one mode leaves the middle-axis broadcast untested |
| repo leg | 0.04, non-zero | a zero repo cannot show that the carry gather is not silently reading it, nor that the two legs use different day counts (365.25 vs 365) |

MUTATION MATRIX - every one RUN, by monkeypatching a parameterised copy of `generate` /
`precalculate` onto the class (and of `arx1_t_mle` for the two calibration mutants) and scoring the
mutant on this whole file plus the inner-MC contract test. Control: 51 passed, 0 failing.

| mutant | killed by | count |
|---|---|---|
| `Gamma` dropped from the shape recursion | the two reference gates (outer + inner) | 2 |
| the ARX reads `dL_{t+1}` (lookahead) instead of the same step's `dL` | the same two | 2 |
| knots published in the INSTANTANEOUS convention, `r(tau) = c + 2 a tau` | the market-continuation gate, at **187 standard errors**, and both reference gates | 3 |
| the initial state broadcast on the LAST axis in inner mode (init collides with the B2 fan-out) | the inner reference gate, the fork gate, the inner-MC contract row | 3 |
| the initial state ignored - seeded at (`Mu_L`, `Mu_D`) instead of the market curve | the market-continuation gate at **664 standard errors**, both reference gates, the near-unit-root gate | 4 |
| `Calibration_DT_Years` ignored (f = 1, no clock rescale) | both reference gates, the clock gate, the near-unit-root gate | 4 |
| the calibration drops the ΔL regressor | the round trip, the collinearity gate, the reference-tenor gate | 3 |
| the calibration lags the ΔL regressor by one row | the round trip, the collinearity gate | 2 |

TWO THINGS THE MATRIX SAYS OUT LOUD. The `generate` mutants cannot reach the calibration gates and
the calibration mutants cannot reach the simulation gates - they are separate code paths, and a
matrix that showed one killing the other would mean a gate was reading the wrong thing. And the
INSTANTANEOUS-convention mutant is killed by exactly ONE gate that is not the reference loop, which
is why that gate exists: both conventions are AFFINE in tau and the reader is exact on any affine
curve, so a process publishing the wrong one reproduces its own closed form at every row and every
gate that compares the output to a curve rebuilt from that same output passes it. Only lining the
first simulated row up against the MARKET curve it started from can see it.
"""
import os
import types

import numpy as np
import pandas as pd
import pytest
import torch

from derivus import utils
from derivus.calculation import CMC_State, CMC_State_Inner
from derivus.stochasticprocess import QuadraticCarryCurveCalibration, QuadraticCarryCurveModel

DEVICE = torch.device('cpu')
REF_DATE = pd.Timestamp('2026-04-10')
EXCEL0 = (REF_DATE - utils.excel_offset).days
DT_C = 1.0 / 252.0
DIY = utils.DAYS_IN_YEAR

#: `QuadraticCarryCurveCalibration` on data/plat_archive_sync.csv, with `Gamma` at the Gaussian-loss
#: value the study reported (-0.451) rather than the t-MLE's -0.019 - see the anti-placebo table.
FITTED = {'Phi_L': 0.9962, 'Mu_L': 0.0169, 'Sigma_L': 0.00148,
          'Phi_D': 0.9468, 'Mu_D': -0.00054, 'Sigma_D': 0.00308,
          'Gamma': -0.451, 'Nu': 3.0, 'Reference_Tenors': [0.5, 1.0],
          'Calibration_DT_Years': DT_C}

#: The last row of the sample: the level far above its mean, the shape far below the level's.
L0, D0 = 0.0329, 0.0068

#: Dated knots bracketing a one-year book, deliberately NOT at the reference tenors.
KNOT_DAYS = np.array([0.0, 400.0])


def _time_grid(T, start=0):
    """A daily CALENDAR grid - dt = 1/365.25 against a 1/252 calibration step, so the clock
    rescale is live in every gate here."""
    days = np.arange(start, start + T, dtype=np.float64)
    tg = types.SimpleNamespace()
    tg.scen_time_grid = days
    tg.time_grid_years = days / DIY
    tg.CurrencyMap = {}
    scen = np.zeros((T, 3))
    scen[:, utils.TIME_GRID_MTM] = days
    scen[:, utils.TIME_GRID_ScenarioPriorIndex] = np.arange(T)
    tg.scenario_grid = scen
    return tg


def _shared(B, T, seed=42, sub=None, dtype=torch.float64, rho=-0.2245, start=0):
    one = torch.ones(1, 1, dtype=dtype, device=DEVICE)
    chol = torch.tensor([[1.0, 0.0], [rho, np.sqrt(1.0 - rho * rho)]], dtype=dtype)
    kw = dict(cholesky=chol, static_buffer={}, batch_size=B, one=one, mcmc_sims=0,
              report_currency=None, seed=seed, job_id=0, num_jobs=1)
    tg = _time_grid(T, start)
    if sub is None:
        s = CMC_State(**kw)
        s.reset(num_factors=2, time_grid=tg)
    else:
        s = CMC_State_Inner(simulation_sub_batch=sub, **kw)
        s.reset_inner(num_factors=2, time_grid=tg)
    return s


def _factor(knot_days=KNOT_DAYS):
    return types.SimpleNamespace(get_tenor=lambda: EXCEL0 + np.asarray(knot_days, dtype=np.float64))


def _curve(state, k, dtype=torch.float64):
    """The two knot values a state `(L, D)` implies at shape coordinates `k` - the market curve a
    world carrying that state would publish."""
    return torch.tensor([state[0] + state[1] * float(ki) for ki in k], dtype=dtype)


def _k(knot_days, tenors, day=0):
    """Shape coordinate of each dated knot as seen from `day`, in the reference-tenor frame."""
    tau_a, tau_b = tenors
    tau = (np.asarray(knot_days, dtype=np.float64) - day) / DIY
    return (tau - 0.5 * (tau_a + tau_b)) / (tau_b - tau_a)


def _process(param, shared, T, tensor, knot_days=KNOT_DAYS, ref_date=REF_DATE, start=0):
    p = QuadraticCarryCurveModel(factor=_factor(knot_days), param=dict(param))
    p.factor_key = utils.Factor('ForwardRate', ('PLATINUM_CARRY',))
    p.precalculate(ref_date, _time_grid(T, start), tensor, shared, process_ofs=0)
    return p


def _state_from_curve(z, k):
    """(L, D) from two knot values at shape coordinates `k` - the explicit 2x2, written the other
    way round from the process's constant-matrix inverse."""
    D = (z[..., 1, :] - z[..., 0, :]) / (k[1] - k[0]) if z.dim() > 1 else (z[1] - z[0]) / (k[1] - k[0])
    return (z[..., 0, :] if z.dim() > 1 else z[0]) - D * k[0], D


# ---------------------------------------------------------------------------
# The representation: two z-knots ARE the quadratic curve, to the forward reader
# ---------------------------------------------------------------------------

REPO_RATE = 0.04


def _derived_forward(z_knots, knot_days, spot, shared):
    """A `DerivedForwardCurve` over a STATIC carry curve holding `z_knots` at `knot_days`, with a
    live flat repo leg beside it - the object `pv_energy_cashflows` builds when a deal prices off
    forward-curve components."""
    carry_key = utils.Factor('ForwardRate', ('PLATINUM_CARRY',))
    repo_key = utils.Factor('InterestRate', ('USD-SOFR',))
    carry_code = [(False, carry_key, None,
                   utils.tenor_diff(EXCEL0 + np.asarray(knot_days, dtype=np.float64)),
                   lambda d: utils.get_day_count_accrual(REF_DATE, d, utils.DAYCOUNT_None))]
    repo_tenors = np.array([0.0, 1.0, 5.0, 30.0])
    repo_code = [(False, repo_key, None, utils.tenor_diff(repo_tenors, 'Linear'),
                  lambda d: utils.get_day_count_accrual(REF_DATE, d, utils.DAYCOUNT_ACT365))]
    shared.t_Static_Buffer[carry_key] = z_knots
    shared.t_Static_Buffer[repo_key] = torch.full((repo_tenors.size,), REPO_RATE, dtype=torch.float64)
    tg = np.zeros((1, 3))
    return utils.DerivedForwardCurve(
        spot, utils.calc_time_grid_curve_rate(carry_code, tg, shared),
        utils.calc_time_grid_curve_rate(repo_code, tg, shared),
        np.array([float(EXCEL0)]), tg)


def _closed_form(spot, c, a, tau_days):
    """F(t,T) = S exp(c*tau + a*tau^2) with the repo leg the reader adds on its own day count."""
    tau = tau_days / DIY
    return spot * np.exp(c * tau + a * tau * tau + REPO_RATE * tau_days / 365.0)


@pytest.mark.parametrize('knot_taus', [(0.10, 0.60), (0.0, 1.0)])
def test_three_listed_futures_reconstruct_exactly_off_two_z_knots(knot_taus):
    """The claim the whole representation rests on, through the real gather stack.

    Three synthetic futures at PL1/PL2/PL3 tenors identify (S, c, a) exactly; z(tau) = c + a tau is
    AFFINE, so the two knots that carry it reproduce every one of them - and an off-knot tenor
    besides - because the reader multiplies the gathered z by tau and the gather interpolates
    linearly in the query DATE, which is affine in tau at a fixed row.

    Both parametrisations bracket the futures: the tight one puts the knots ON the first and last
    listed contract, the wide one at the base date and a year out. Neither is at the REFERENCE
    tenors (0.5, 1.0) the state is defined at - the two are separate choices and this is the gate
    that says so."""
    spot, c, a = 950.0, 0.0163, -0.0022
    fut_taus = np.array([0.10, 0.35, 0.60])
    knot_days = np.asarray(knot_taus) * DIY
    z = torch.tensor(c + a * np.asarray(knot_taus), dtype=torch.float64)

    shared = _shared(1, 2)
    dfc = _derived_forward(z, knot_days, torch.full((1, 1), spot, dtype=torch.float64), shared)

    query = np.append(fut_taus, 0.42) * DIY                             # + one strictly off-knot
    got = dfc.gather_weighted_curve(shared, (EXCEL0 + query).reshape(1, -1)).numpy()[0, :, 0]
    want = _closed_form(spot, c, a, query)
    rel = np.abs(got / want - 1.0)
    assert rel.max() < 1e-14, f'reconstruction off the closed form by {rel.max():.3e}: {got} vs {want}'

    # ...and the curvature is live, so "z is constant" is not what made this pass.
    assert abs(a) > 0.001 and abs(want[-1] / want[0] - 1.0) > 0.005


def test_the_read_outside_the_knot_bracket_is_flat_in_z_not_linear():
    """Where the algebra stops, pinned as BEHAVIOUR with its size.

    `utils.CurveTenor.get_index` clips a query to [first knot, last knot], so beyond the bracket
    the gathered z is the nearest knot's and the log-carry continues LINEARLY instead of
    quadratically. The premise "linear interpolation between AND extrapolation beyond the two
    knots" is therefore only half true, and the market data has to place the knots to bracket every
    (row date, query date) pair the book reaches.

    Measured here on knots at tau 0.5 and 1.0: 0 ULP inside, -2.5e-5 relative at tau = 0.05 and
    +8.3e-4 at tau = 1.5 - and the clipped read is reproduced exactly, which is what makes this a
    property of the reader rather than a tolerance."""
    spot, c, a = 950.0, 0.0163, -0.0011
    knot_taus = np.array([0.5, 1.0])
    z_np = c + a * knot_taus
    shared = _shared(1, 2)
    dfc = _derived_forward(torch.tensor(z_np, dtype=torch.float64), knot_taus * DIY,
                           torch.full((1, 1), spot, dtype=torch.float64), shared)

    taus = np.array([0.05, 0.25, 0.5, 0.7, 1.0, 1.5])
    got = dfc.gather_weighted_curve(shared, (EXCEL0 + taus * DIY).reshape(1, -1)).numpy()[0, :, 0]
    inside = (taus >= knot_taus[0]) & (taus <= knot_taus[1])
    exact = _closed_form(spot, c, a, taus * DIY)
    assert np.abs(got[inside] / exact[inside] - 1.0).max() < 1e-15, 'the bracket is not exact'

    # outside: z is the CLIPPED knot value, still multiplied by the true tau
    z_clipped = c + a * np.clip(taus, *knot_taus)
    flat = spot * np.exp(z_clipped * taus + REPO_RATE * taus * DIY / 365.0)
    assert np.abs(got / flat - 1.0).max() < 1e-15, 'the outside read is not the flat-clip read'
    err = got[~inside] / exact[~inside] - 1.0
    assert err.min() < -1e-5 < 1e-5 < err.max(), f'the bracket costs nothing measurable: {err}'


def test_the_simulated_curve_is_the_quadratic_the_reader_prices_at_every_row():
    """The ageing half: the knots are DATED, so their tenors shrink with sim time and the process
    publishes z at whatever tenor each has aged to. Read the simulated curve at a later row through
    the same stack and it must still be the quadratic implied by that row's own (L, D)."""
    T, B, t = 12, 4, 7
    shared = _shared(B, T)
    p = _process(FITTED, shared, T, _curve((L0, D0), _k(KNOT_DAYS, FITTED['Reference_Tenors'])))
    torch.manual_seed(5)
    path = p.generate(shared)                                            # (T, 2, B)

    row_excel = EXCEL0 + t
    L, D = _state_from_curve(path[t], _k(KNOT_DAYS, FITTED['Reference_Tenors'], day=t))
    tau_a, tau_b = FITTED['Reference_Tenors']
    a = D.numpy() / (tau_b - tau_a)
    c = L.numpy() - D.numpy() * 0.5 * (tau_a + tau_b) / (tau_b - tau_a)

    # one row of the deal grid, sitting at scenario row t
    tg = np.zeros((1, 3))
    tg[:, utils.TIME_GRID_MTM] = float(t)
    tg[:, utils.TIME_GRID_ScenarioPriorIndex] = t
    carry_key = utils.Factor('ForwardRate', ('PLATINUM_CARRY',))
    repo_key = utils.Factor('InterestRate', ('USD-SOFR',))
    repo_tenors = np.array([0.0, 1.0, 5.0, 30.0])
    shared.t_Scenario_Buffer[carry_key] = path
    shared.t_Static_Buffer[repo_key] = torch.full((repo_tenors.size,), REPO_RATE, dtype=torch.float64)
    carry = utils.calc_time_grid_curve_rate(
        [(True, carry_key, None, utils.tenor_diff(EXCEL0 + KNOT_DAYS),
          lambda d: utils.get_day_count_accrual(REF_DATE, d, utils.DAYCOUNT_None))], tg, shared)
    repo = utils.calc_time_grid_curve_rate(
        [(False, repo_key, None, utils.tenor_diff(repo_tenors, 'Linear'),
          lambda d: utils.get_day_count_accrual(REF_DATE, d, utils.DAYCOUNT_ACT365))], tg, shared)
    spot = torch.full((1, B), 950.0, dtype=torch.float64)
    dfc = utils.DerivedForwardCurve(spot, carry, repo, np.array([float(row_excel)]), tg)

    query_days = np.array([30.0, 120.0, 250.0])                          # all inside the bracket
    got = dfc.gather_weighted_curve(shared, (row_excel + query_days).reshape(1, -1)).numpy()[0]
    want = np.stack([_closed_form(950.0, c, a, d) for d in query_days])
    assert np.abs(got / want - 1.0).max() < 1e-13, (
        f'aged curve off the closed form by {np.abs(got / want - 1.0).max():.3e}')


def test_the_first_simulated_row_continues_the_market_curve_in_the_same_convention():
    """The one gate that can see the KNOT CONVENTION, and the reason the two gates above cannot.

    Both z(tau) = c + a*tau and the instantaneous slope r(tau) = c + 2 a tau are AFFINE in tau, and
    the reader is exact on any affine curve - so a process publishing the wrong one is perfectly
    self-consistent and reproduces its own closed form at every row. What it cannot do is line up
    with the MARKET curve it started from: row 0 is the market's own knots, so a convention that
    changes at row 1 shows as a step in the gathered carry that has nothing to do with the
    innovations.

    Measured against the exact one-step conditional mean, in gathered-carry units at a fixed
    delivery date, with the sample mean over B paths and its own standard error - so this is a
    number, not a smell: the archive-labels convention lands 187 standard errors away, and a run
    that ignores the initial state 664."""
    T, B, delivery = 3, 4096, 250.0
    shared = _shared(B, T, seed=13)
    z0 = _curve((L0, D0), _k(KNOT_DAYS, FITTED['Reference_Tenors']))
    p = _process(FITTED, shared, T, z0)
    torch.manual_seed(29)
    path = p.generate(shared)

    tau_a, tau_b = FITTED['Reference_Tenors']
    k1 = float(_k([EXCEL0 + delivery], [tau_a, tau_b], day=EXCEL0 + 1)[0])
    phi_L, phi_D = (float(p.phi_L[1]), float(p.phi_D[1]))
    L1 = FITTED['Mu_L'] + phi_L * (L0 - FITTED['Mu_L'])
    D1 = FITTED['Mu_D'] + phi_D * (D0 - FITTED['Mu_D']) + FITTED['Gamma'] * (L1 - L0)
    predicted = L1 + D1 * k1

    tg = np.zeros((1, 3))
    tg[:, utils.TIME_GRID_MTM] = 1.0
    tg[:, utils.TIME_GRID_ScenarioPriorIndex] = 1
    carry_key = utils.Factor('ForwardRate', ('PLATINUM_CARRY',))
    shared.t_Scenario_Buffer[carry_key] = path
    block = utils.calc_time_grid_curve_rate(
        [(True, carry_key, None, utils.tenor_diff(EXCEL0 + KNOT_DAYS),
          lambda d: utils.get_day_count_accrual(REF_DATE, d, utils.DAYCOUNT_None))], tg, shared)
    z1 = block.gather_weighted_curve(
        shared, np.array([[EXCEL0 + delivery]]), multiply_by_time=False)[0, 0]     # (B,)

    se = float(z1.std()) / np.sqrt(B)
    off = abs(float(z1.mean()) - predicted) / se
    assert off < 5.0, (f'the first simulated row is {off:.0f} se off the conditional mean of the '
                       f'curve it started from: {float(z1.mean()):.6f} vs {predicted:.6f}')
    # the fixture has something to see: the shape term is a live part of the level at this tenor
    assert abs(D1 * k1) > 20.0 * se, 'the shape contributes less than the noise - gate blind'


# ---------------------------------------------------------------------------
# The dynamics, exactly
# ---------------------------------------------------------------------------

def _reference(p, shared, seed, param, knot_days=KNOT_DAYS, start=0):
    """The two recursions as the docstring states them: a row-at-a-time loop over python state, the
    knot geometry and the clock rescale rebuilt from the FACTOR and the params rather than read off
    `p`, and the mode never mentioned - every expression broadcasts.

    In torch rather than numpy, and that is a measurement rather than a preference: `torch.sqrt`
    and tensor division on float64 disagree with numpy's by up to 7.1e-15 on identical inputs, so a
    numpy transcription could only be compared with a tolerance, and a tolerant recursion gate is
    strictly weaker than an exact one. `p.state0` is taken as given here - the recovery is its own
    gate below, and this one is about the RECURSION.

    Returns the level path, the shape path and the curve they imply."""
    Z = shared.t_random_numbers[0:2, :p.scenario_horizon]
    T = Z.shape[1]
    torch.manual_seed(seed)
    W = torch.distributions.Chi2(shared.one.new_tensor(param['Nu'])).sample(Z.shape[1:]).clamp_min(1.0e-6)
    eps = Z * torch.sqrt((param['Nu'] - 2.0) / W)

    days = np.arange(start, start + T, dtype=np.float64)
    k = torch.tensor(_k(knot_days, param['Reference_Tenors'], day=days.reshape(-1, 1)))
    years = _time_grid(T, start).time_grid_years                         # the grid's own clock
    f = np.diff(np.hstack(([years[0]], years))) / param['Calibration_DT_Years']
    coeff = {}
    for nm in ('L', 'D'):
        phi, sigma = param[f'Phi_{nm}'], param[f'Sigma_{nm}']
        phi_f = phi ** f
        coeff[nm] = (torch.tensor(phi_f),
                     torch.tensor(sigma * np.sqrt((1.0 - phi_f * phi_f) / (1.0 - phi * phi))))

    inner = Z.dim() == 4
    Ls = [(p.state0[0].unsqueeze(-1) if inner else p.state0[0]) + torch.zeros_like(Z[0, 0])]
    Ds = [(p.state0[1].unsqueeze(-1) if inner else p.state0[1]) + torch.zeros_like(Z[0, 0])]
    for t in range(1, T):
        phi_L, sig_L = coeff['L']
        phi_D, sig_D = coeff['D']
        Ls.append(param['Mu_L'] + phi_L[t] * (Ls[-1] - param['Mu_L']) + sig_L[t] * eps[0, t])
        Ds.append(param['Mu_D'] + phi_D[t] * (Ds[-1] - param['Mu_D'])
                  + param['Gamma'] * (Ls[-1] - Ls[-2]) + sig_D[t] * eps[1, t])
    L, D = torch.stack(Ls), torch.stack(Ds)
    curve = torch.stack([L + D * k[:, i].reshape(-1, *([1] * (L.dim() - 1))) for i in range(2)], dim=1)
    return L, D, curve


@pytest.mark.parametrize('sub', [None, 6])
def test_the_dynamics_match_an_independent_reference_exactly(sub):
    """float64 end to end so "exact" means exact: the same elementwise operators in the order the
    documented equations name them, off the same seeded draws, in both simulation modes."""
    T, B, seed = 40, 8, 11
    shared = _shared(B, T, sub=sub)
    z0 = _curve((L0, D0), _k(KNOT_DAYS, FITTED['Reference_Tenors']))
    tensor = z0 if sub is None else z0.unsqueeze(-1).expand(2, B).contiguous()
    p = _process(FITTED, shared, T, tensor)

    torch.manual_seed(seed)
    got = p.generate(shared)
    L, D, want = _reference(p, shared, seed, FITTED)
    assert got.shape == want.shape == ((T, 2, B) if sub is None else (T, 2, B, sub))
    assert torch.equal(got, want), (
        f'carry curve path off the reference recursion, max |d| = {(got - want).abs().max():.3e}')
    # the fixture keeps both recursions live: neither factor sits at its seed or its mean
    assert (L[-1] - L[0]).abs().max() > 1e-3 and (D[-1] - D[0]).abs().max() > 1e-3
    assert (L - FITTED['Mu_L']).abs().mean() > 1e-3, 'the level sits on its mean - reversion inert'


def test_the_clock_rescale_is_grid_invariant():
    """`Calibration_DT_Years` is READ, and this is the property that says so rather than the
    spelling `_reference` shares with the process.

    The recursions are calibrated per business day and the sim grid is CALENDAR daily, so a step is
    f = dt/dt_c ~ 0.69 calibration steps. The exact stationary aggregation phi_f = phi^f,
    sigma_f = sigma*sqrt((1-phi_f^2)/(1-phi^2)) leaves TWO things invariant to the grid, and both
    are checked here on three different grids: the stationary variance sigma^2/(1-phi^2), and the
    reversion RATE log(phi)/dt. A process that ignored the clock (f = 1) would keep neither."""
    stationary = {nm: FITTED[f'Sigma_{nm}'] ** 2 / (1 - FITTED[f'Phi_{nm}'] ** 2) for nm in 'LD'}
    rate = {nm: np.log(FITTED[f'Phi_{nm}']) / DT_C for nm in 'LD'}
    z0 = _curve((L0, D0), _k(KNOT_DAYS, FITTED['Reference_Tenors']))
    for dt_c in (DT_C, 1.0 / 365.25, 1.0 / 52.0):
        param = dict(FITTED, Calibration_DT_Years=dt_c,
                     # hold the CONTINUOUS-time dynamics fixed as the calibration step moves
                     Phi_L=float(np.exp(rate['L'] * dt_c)), Phi_D=float(np.exp(rate['D'] * dt_c)))
        for nm in 'LD':
            param[f'Sigma_{nm}'] = float(np.sqrt(stationary[nm] * (1 - param[f'Phi_{nm}'] ** 2)))
        p = _process(param, _shared(2, 4), 4, z0)
        for nm, phi, sig in (('L', p.phi_L, p.sig_L), ('D', p.phi_D, p.sig_D)):
            step = 1.0 / DIY                                             # the grid's own step, years
            assert float(torch.log(phi[1])) / step == pytest.approx(rate[nm], rel=1e-12), (
                f'{nm}: reversion rate moved with the calibration step')
            assert float(sig[1] ** 2 / (1 - phi[1] ** 2)) == pytest.approx(stationary[nm], rel=1e-12), (
                f'{nm}: stationary variance moved with the calibration step')
    # ...and the rescale is not the identity on this grid, or none of the above could fail
    p = _process(FITTED, _shared(2, 4), 4, z0)
    assert abs(float(p.phi_L[1]) - FITTED['Phi_L']) > 1e-4


def test_num_factors_matches_the_correlation_name():
    """The invariant every process has to hold and nothing asserted: `num_factors()` is how many
    substreams of `t_random_numbers` the process reads, `correlation_name[1]` is how many columns
    the global cholesky gives it, and they are declared in two places."""
    p = QuadraticCarryCurveModel(factor=None, param=dict(FITTED))
    name, addons = p.correlation_name
    assert p.num_factors() == len(addons) == 2, (name, addons)
    assert [a for a in addons] == [('L',), ('D',)]


def test_the_initial_state_is_the_market_curve_and_row_zero_is_that_curve():
    """No declared `L_0`/`D_0`: the state IS the factor's curve, so `precalculate` recovers it and
    `generate` publishes the curve itself at row 0. A declared pair would be a second source for a
    number the market data already carries, and the market has to win - otherwise the t=0 forwards
    a run prices are not the ones the world was built with."""
    T, B = 6, 4
    shared = _shared(B, T)
    k0 = _k(KNOT_DAYS, FITTED['Reference_Tenors'])
    z0 = _curve((L0, D0), k0)
    p = _process(FITTED, shared, T, z0)

    assert float(p.state0[0]) == pytest.approx(L0, rel=1e-14)
    assert float(p.state0[1]) == pytest.approx(D0, rel=1e-13)
    assert not np.allclose(k0, [-0.5, 0.5]), 'knots at the reference tenors make this symmetric'

    torch.manual_seed(3)
    path = p.generate(shared)
    assert torch.equal(path[0], z0.unsqueeze(-1).expand(2, B)), 'row 0 is not the market curve'
    # and the state that was recovered is far from the means, so "seeded at Mu" cannot pass
    assert abs(L0 - FITTED['Mu_L']) > 0.01 and abs(D0 - FITTED['Mu_D']) > 0.005


def test_the_fork_recovers_the_outer_state_at_the_fork_row():
    """Fork coherence, and the reason this process publishes no fork seed at all.

    `Credit_Monte_Carlo._run_inner_mc_at_t` hands a fork `outer_scenario_buffer[key][t]` as its
    initial curve and a time grid rebased to the fork date. Both knots have aged by exactly those
    days, so the fork's own recovery returns the outer path's (L_t, D_t) - the state travels in the
    curve, and `inner_fork_seed` has nothing to add."""
    T, B, B2, t = 30, 5, 4, 13
    shared = _shared(B, T)
    z0 = _curve((L0, D0), _k(KNOT_DAYS, FITTED['Reference_Tenors']))
    outer = _process(FITTED, shared, T, z0)
    torch.manual_seed(21)
    path = outer.generate(shared)

    # what the outer path's state IS at row t, read off the published curve
    L_t, D_t = _state_from_curve(path[t], _k(KNOT_DAYS, FITTED['Reference_Tenors'], day=t))

    fork_shared = _shared(B, T - t, sub=B2, seed=99, start=0)
    fork = _process(FITTED, fork_shared, T - t, path[t].detach(),
                    ref_date=REF_DATE + pd.Timedelta(days=t))
    # the fork's constant-matrix recovery against the explicit 2x2 written the other way round -
    # the same number by two routes, so a few ulp rather than `torch.equal`
    assert torch.allclose(fork.state0[0], L_t, rtol=1e-13, atol=0.0), (fork.state0[0], L_t)
    assert torch.allclose(fork.state0[1], D_t, rtol=1e-12, atol=0.0), (fork.state0[1], D_t)
    assert fork.inner_fork_seed(fork.factor_key, shared.t_Scenario_Buffer, t) == {}
    assert fork.outer_reseed() == {}

    torch.manual_seed(31)
    inner = fork.generate(fork_shared)
    assert inner.shape == (T - t, 2, B, B2)
    assert torch.equal(inner[0], path[t].unsqueeze(-1).expand(2, B, B2)), (
        'the fork does not start on the curve it was forked from')


def test_the_level_does_not_mean_revert_materially_over_a_quarter():
    """The documented modelling caveat, made checkable. phi_L = 0.9962 is 2.3 s.e. from a unit root,
    so nothing may lean on carry reversion for value: over a quarter the conditional mean gives up
    only 1 - phi^(sum f) of a level deviation. The threshold is COMPUTED from the declared phi and
    the grid, not written down, and the simulated mean has to agree with it."""
    T, B, horizon = 92, 16384, 91
    shared = _shared(B, T, seed=7)
    z0 = _curve((L0, D0), _k(KNOT_DAYS, FITTED['Reference_Tenors']))
    p = _process(FITTED, shared, T, z0)
    torch.manual_seed(17)
    path = p.generate(shared)

    f_total = horizon / DIY / DT_C                                       # business days elapsed
    retained = FITTED['Phi_L'] ** f_total
    assert retained > 0.75, (
        f'the level gave up {1 - retained:.1%} of its deviation over a quarter - this model is '
        f'documented as a near-unit root and nothing may lean on its reversion')

    L, _ = _state_from_curve(path[horizon], _k(KNOT_DAYS, FITTED['Reference_Tenors'], day=horizon))
    expected = FITTED['Mu_L'] + retained * (L0 - FITTED['Mu_L'])
    phi_f = FITTED['Phi_L'] ** (1.0 / DIY / DT_C)
    sig_f = FITTED['Sigma_L'] * np.sqrt((1 - phi_f ** 2) / (1 - FITTED['Phi_L'] ** 2))
    var = sig_f ** 2 * np.sum(phi_f ** (2 * np.arange(horizon)))
    se = np.sqrt(var / B)
    assert abs(float(L.mean()) - expected) < 6.0 * se, (
        f'simulated E[L_T] = {float(L.mean()):.6f} vs closed form {expected:.6f} '
        f'({abs(float(L.mean()) - expected) / se:.1f} se)')


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _archive(n=4000, seed=5, param=None, rho=0.0):
    """An archive the MODEL generated: the same two recursions in numpy, published as the two
    average-carry columns the calibration reads, so the round trip below is a genuine recovery.

    `rho = 0` by default because Gamma is NOT identified against a correlated innovation pair -
    see `test_gamma_and_the_innovation_correlation_are_one_coupling_split_two_ways`."""
    param = param or FITTED
    rng = np.random.default_rng(seed)
    nu = param['Nu']
    w = rng.chisquare(nu, n)
    z = rng.multivariate_normal([0, 0], [[1.0, rho], [rho, 1.0]], n)
    eps = z * np.sqrt((nu - 2.0) / w)[:, None]
    L, D = np.empty(n), np.empty(n)
    L[0], D[0] = L0, D0
    for i in range(1, n):
        L[i] = param['Mu_L'] + param['Phi_L'] * (L[i - 1] - param['Mu_L']) + param['Sigma_L'] * eps[i, 0]
        D[i] = (param['Mu_D'] + param['Phi_D'] * (D[i - 1] - param['Mu_D'])
                + param['Gamma'] * (L[i] - L[i - 1]) + param['Sigma_D'] * eps[i, 1])
    tau_a, tau_b = param['Reference_Tenors']
    return pd.DataFrame({f'ForwardRate.PLATINUM_CARRY,{tau_a}': L - 0.5 * D,
                         f'ForwardRate.PLATINUM_CARRY,{tau_b}': L + 0.5 * D},
                        index=pd.bdate_range('2010-01-01', periods=n))


def test_the_calibration_recovers_the_block_that_generated_the_archive():
    """The round trip, on 4000 rows of the model's own output. `Gamma` is the coefficient the whole
    ARX exists for and it is recovered inside 10%, which is what makes the dropped-Gamma mutant a
    calibration failure as well as a simulation one."""
    out = QuadraticCarryCurveCalibration(model=None, param={}).calibrate(_archive(), 0.0)
    assert list(out.param) == ['Phi_L', 'Mu_L', 'Sigma_L', 'Phi_D', 'Mu_D', 'Sigma_D',
                               'Gamma', 'Nu', 'Reference_Tenors', 'Calibration_DT_Years']
    assert out.param['Phi_L'] == pytest.approx(FITTED['Phi_L'], abs=0.01)
    assert out.param['Phi_D'] == pytest.approx(FITTED['Phi_D'], abs=0.03)
    assert out.param['Sigma_L'] == pytest.approx(FITTED['Sigma_L'], rel=0.15)
    assert out.param['Sigma_D'] == pytest.approx(FITTED['Sigma_D'], rel=0.15)
    assert out.param['Gamma'] == pytest.approx(FITTED['Gamma'], rel=0.10)
    assert out.param['Reference_Tenors'] == FITTED['Reference_Tenors'], (
        'the reference tenors must come from the archive column labels, not a constant')
    # delta: two standardised innovation columns, which is what the correlation consolidation needs
    assert list(out.delta.columns) == ['ForwardRate.PLATINUM_CARRY,L', 'ForwardRate.PLATINUM_CARRY,D']
    assert abs(float(out.delta.values.std()) - 1.0) < 0.35
    assert np.array_equal(out.correlation, np.eye(2))
    # ...and the stamped block runs in the model it was stamped for
    shared = _shared(4, 10)
    p = _process(out.param, shared, 10, _curve((L0, D0), _k(KNOT_DAYS, out.param['Reference_Tenors'])))
    torch.manual_seed(1)
    assert torch.isfinite(p.generate(shared)).all()


def test_gamma_and_the_innovation_correlation_are_one_coupling_split_two_ways():
    """Γ and ρ(ε_L, ε_D) are COLLINEAR, and this is the gate that says the split does not matter.

    The one-step covariance the two channels produce is Cov(ΔL, ΔD) = Γ·σ_L² + ρ·σ_L·σ_D, one
    equation in two unknowns: a pure-Γ world and a pure-ρ world with the same covariance are the
    same joint law, and no regression can tell them apart. That is why the sample's own Γ swings
    from -0.02 to -0.45 with the tail weight (the calibration docstring's ν sweep) - the fit is
    moving along that line, not finding a different market.

    What has to hold, and does: the coupling ARRIVES intact whichever world produced the data. Two
    archives are generated with the same Cov(ΔL, ΔD) - one carrying it entirely in Γ with
    independent innovations, one entirely in ρ with Γ = 0 - and the calibration returns the SAME Γ
    from both, to 15%, with the residual correlation the framework consolidates left at zero in
    BOTH. ΔL is endogenous in the ρ world, so the conditional-mean fit absorbs the correlation into
    Γ; the simulator therefore never double counts the coupling and never loses it.

    The invariant is stated on Γ rather than on Cov(ΔL, ΔD) = Γ·σ_L² deliberately: at ν = 3 the
    scale MLE carries ~10% sampling error on 4000 rows, which the square doubles, so the covariance
    form would need a 35% tolerance to hold and would be measuring σ̂_L rather than the split.

    Measured on the real archive the fit lands on the OTHER end: at ν = 3 the tail observations
    where the coupling lives are downweighted in the conditional mean, so Γ comes back -0.019 while
    the residual correlation the framework consolidates is -0.22. Same line, other point."""
    g, s_L, s_D = FITTED['Gamma'], FITTED['Sigma_L'], FITTED['Sigma_D']
    rho = g * s_L / s_D                                    # the same Cov(dL, dD), as pure rho
    fitted = {}
    for name, param, r in (('gamma', FITTED, 0.0), ('rho', dict(FITTED, Gamma=0.0), rho)):
        out = QuadraticCarryCurveCalibration(model=None, param={}).calibrate(
            _archive(param=param, rho=r), 0.0)
        rho_hat = float(out.delta.corr().values[0, 1])
        assert abs(rho_hat) < 0.05, f'{name}: the fit left {rho_hat:.3f} of coupling in the residual'
        fitted[name] = out.param['Gamma']
    assert fitted['gamma'] == pytest.approx(fitted['rho'], rel=0.15), fitted
    assert fitted['rho'] == pytest.approx(g, rel=0.15), (fitted, g)


def test_the_reference_tenors_follow_the_archive_labels():
    """The state definition is data, not a constant: relabel the columns and the stamped
    `Reference_Tenors` follow, so a curve identified at other tenors cannot be silently read in the
    0.5/1.0 frame."""
    param = dict(FITTED, Reference_Tenors=[0.25, 0.75])
    out = QuadraticCarryCurveCalibration(model=None, param={}).calibrate(_archive(param=param), 0.0)
    assert out.param['Reference_Tenors'] == [0.25, 0.75]


ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'plat_archive_sync.csv')


@pytest.mark.skipif(not os.path.exists(ARCHIVE), reason='platinum archive not present (data/ is untracked)')
def test_the_platinum_archive_carries_the_average_carry_convention():
    """The relabel: `data/plat_archive_sync.csv` used to label these two columns 0.25 / 0.5, as the
    INSTANTANEOUS slope r(tau) = c + 2 a tau. The identity r(tau) == z(2 tau) makes those the same
    NUMBERS as z(0.5) and z(1.0), so the values were always right and only the labels were wrong
    for the convention this model reads. Nothing but the header changed."""
    df = pd.read_csv(ARCHIVE, index_col=0, nrows=5)
    carry = [c for c in df.columns if c.startswith('ForwardRate.PLATINUM_CARRY')]
    assert [c.split(',', 1)[1] for c in carry] == ['0.5', '1.0'], carry
