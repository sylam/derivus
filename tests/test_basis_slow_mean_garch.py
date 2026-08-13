"""`BasisLinkedSpotModel`'s two optional extensions: the slow observable mean and the basis's
own GARCH(1,1) innovation vol.

Both are deterministic recursions on the realised path — the `GARCHSpotModel` observable-`h`
idiom — so neither draws, and both are OFF at their 0.0 defaults. What the file gates:

  * OFF is the SHIPPED path bitwise, outer and inner, against a frozen reproduction of
    `generate` as of 59a7f36 held here as source rather than a stored array.
  * The two recursions, and their interaction, against an independently written reference in
    float64 — exact, not tolerant, on a seeded path. (`_reference` records why that reference
    cannot be numpy and still be exact.)
  * `Sigma_By_State` still wins: the documented precedence, with the GARCH fields declared
    beside it and inert.
  * The inner fork continues the OUTER path's mean / variance state at the fork row, with a
    negative control (an unseeded fork starts from the calibrated seeds instead).
  * The burn-in reseed, and the replay hook's refusal.
  * `BasisLinkedSpotCalibration` stamps exactly the shipped block at its defaults, and the two
    new field groups when asked.

ANTI-PLACEBO — the fixture property each gate needs, and why. A gate here is worthless if the
fixture zeroes the term it measures, and three of these are one keystroke from doing so:

| property | value | what goes blind without it |
|---|---|---|
| `A` | -0.609 (the shipping fixture's) | at `A = 0` the ΔS term is dead, so every gate stops seeing the linked path and a mutation that mis-indexes it survives |
| `Phi` | 0.633 | **at `Phi = 1` the slow mean CANCELS** — `mu + ds + 1*(b-mu) + eta` is the shipped expression — so the whole mean half would read as a no-op; at `Phi = 0` the AR term dies |
| `Slow_Mean_Lambda` | 0.96875 | at 0 the recursion is off by definition; at 1 `mu` is frozen at `Mu_0` and the recursion never runs |
| `Mu_0` | 9.3786 (≠ 0, ≠ `b0`) | at 0 the mean term contributes nothing at t=1 and "the seed is ignored" survives |
| `Sig2_0` | 18.1455 | the GARCH's own unconditional variance here is **0.953**, 19x smaller — start it at the unconditional and the mutant is loud; start it AT the unconditional in the fixture and the mutant is invisible |
| `G_Alpha` | 0.066 | at 0 the recursion never reads an innovation at all, so `eta_t` vs `eta_{t-1}` is unobservable |
| linked path | GARCH-primary path, moves every step | a flat linked path kills `ds` exactly as `A = 0` does |
| rows | T = 40, both modes | one row leaves every recursion at its seed |

MUTATION MATRIX — every one RUN, by monkeypatching a parameterised copy of `generate` (and of
`_recursion_seed`) onto the class and scoring the mutant on the whole file. Control: 0 failing.

| mutant | killed by | count |
|---|---|---|
| `lam = 0.0` hard-coded inside the recursion | both ON recursion gates, inner-mode, fork | 4 |
| GARCH lookahead: `sig2_t` absorbs the CURRENT row's innovation before it is drawn | the `garch`/`both` recursion gates, inner-mode | 3 |
| GARCH lag-2: `sig2_t` absorbs `eta_{t-2}` | the same three | 3 |
| GARCH update hoisted above the draw (so row 1 never updates) | the same three | 3 |
| `Sig2_0` ignored — seeded at the unconditional `omega/(1-alpha-beta)` | the `garch`/`both` gates, inner-mode, fork | 4 |
| the OFF guard removed (`slow_mean`/`garch` forced True) | all four OFF gates, the regime OFF gate, precedence, two recursion gates | 8 |
| the fork seed dropped (`_recursion_seed` returns the calibrated scalar always) | the fork gate and the burn-in reseed gate | 2 |

One mutant is recorded here because it SURVIVED and should have: moving the `sig2` update from
below the draw to above it, still consuming the previous row's innovation, is a pure reordering —
nothing between the two statements reads `sig2` — and it produces a bitwise identical path. "Read
`eta_t` instead of `eta_{t-1}`" has to be spelled as a lookahead or as a lag to be a defect at
all, which is why both spellings are in the table above.
"""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from derivus import utils
from derivus.calculation import CMC_State, CMC_State_Inner
from derivus.stochasticprocess import BasisLinkedSpotCalibration, BasisLinkedSpotModel

DEVICE = torch.device('cpu')
REF_DATE = pd.Timestamp('2026-04-10')
DT_C = 1.0 / 252.0

#: `BasisLinkedSpotModel.PLATINUM_CME.LME_CME` out of tests/fixtures/data/MarketDataRF_platinum_garch.json
#: — the shipping world's own flat-Sigma block, so the OFF gate is that world's arithmetic.
PLATINUM = {'A': -0.6090393403931138, 'Phi': 0.35795333829682124, 'Nu': 5.3936611012885685,
            'Mu': 0.0, 'Sigma': 8.079302632269696, 'Calibration_DT_Years': DT_C}

#: The completed basis study's fitted numbers (data/plat_marketdata_draft.json), plus the two
#: seeds `BasisLinkedSpotCalibration` stamps for them on data/plat_archive_sync.csv. `A` is the
#: shipping fixture's rather than the study's 0.0, which would zero the ΔS term (see above).
FITTED = dict(PLATINUM, Phi=0.633, Nu=5.31,
              Slow_Mean_Lambda=1.0 - 2.0 / 64.0, Mu_0=9.3786,
              G_Omega=0.0448, G_Alpha=0.066, G_Beta=0.887, Sig2_0=18.1455)

B0 = 12.0                                                    # observed initial basis, ≠ Mu_0


def _time_grid(T):
    days = np.arange(1, T + 1, dtype=np.float64)
    tg = types.SimpleNamespace()
    tg.scen_time_grid = days
    tg.time_grid_years = days * DT_C
    tg.CurrencyMap = {}
    scen = np.zeros((T, 3), dtype=np.float64)
    scen[:, utils.TIME_GRID_MTM] = days
    scen[:, utils.TIME_GRID_ScenarioPriorIndex] = np.arange(T)
    tg.scenario_grid = scen
    return tg


def _shared(B, T, seed=42, sub=None, dtype=torch.float32):
    one = torch.ones(1, 1, dtype=dtype, device=DEVICE)
    tg = _time_grid(T)
    kw = dict(cholesky=torch.eye(1, dtype=dtype), static_buffer={}, batch_size=B, one=one,
              mcmc_sims=0, report_currency=None, seed=seed, job_id=0, num_jobs=1)
    if sub is None:
        s = CMC_State(**kw)
        s.reset(num_factors=1, time_grid=tg)
    else:
        s = CMC_State_Inner(simulation_sub_batch=sub, **kw)
        s.reset_inner(num_factors=1, time_grid=tg)
    return s


def _linked(shape, dtype, seed=123):
    """A positive linked-spot path that MOVES every step — `ds` is live in every gate here."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    steps = torch.randn(shape, generator=g, dtype=dtype, device=DEVICE)
    return 1700.0 + steps.cumsum(0) * 6.0


def _basis(param, shared, T, b0, dtype=torch.float32, regimes=None):
    p = BasisLinkedSpotModel(factor=None, param=dict(param))
    p.factor_key = utils.Factor('ObservedBasis', ('PLATINUM_CME', 'LME_CME'))
    p.linked_key = utils.Factor('CommodityPrice', ('PLATINUM_CME',))
    p.precalculate(REF_DATE, _time_grid(T), b0, shared, process_ofs=0)
    Z = shared.t_random_numbers[0, :T]
    shared.t_Scenario_Buffer[p.linked_key] = _linked(Z.shape, dtype)
    if regimes is not None:
        shared.t_Scenario_Buffer[(p.linked_key, 'regimes')] = regimes
    return p


def _head_generate(p, shared_mem):
    """`BasisLinkedSpotModel.generate` verbatim as of 59a7f36 (the two-loop, mode-branched form),
    kept as SOURCE so the OFF gate reads as the two expressions side by side rather than as a
    hash of a stored array."""
    Z = shared_mem.t_random_numbers[p.z_offset, :p.scenario_horizon]
    device, dtype = Z.device, Z.dtype
    linked_path = shared_mem.t_Scenario_Buffer[p.linked_key]
    if p.sigma_by_state is not None:
        sigma_t = p.sigma_by_state[shared_mem.t_Scenario_Buffer[(p.linked_key, 'regimes')]]
    else:
        sigma_t = p.sigma_flat
    nu, phi, a = p.Nu, p.Phi, p.A
    if Z.ndim == 2:
        T, B = Z.shape
        W = torch.distributions.Chi2(shared_mem.one.new_tensor(nu)).sample((T, B)).clamp_min(1.0e-6)
        eta = sigma_t * Z * torch.sqrt((nu - 2.0) / W)
        out = torch.empty((T, B), device=device, dtype=dtype)
        out[0] = p.b0.expand(B)
    else:
        T, B, B2 = Z.shape
        W = torch.distributions.Chi2(shared_mem.one.new_tensor(nu)).sample((T, B, B2)).clamp_min(1.0e-6)
        eta = sigma_t * Z * torch.sqrt((nu - 2.0) / W)
        out = torch.empty((T, B, B2), device=device, dtype=dtype)
        out[0] = p.b0.unsqueeze(-1).expand(B, B2)
    for t in range(1, T):
        out[t] = a * (linked_path[t] - linked_path[t - 1]) + phi * out[t - 1] + eta[t]
    return out


def _reference(p, shared, seed):
    """The two recursions written from the class docstring's equations as a row-at-a-time loop
    over python state — no preallocated buffer, no fused branch, the mode never mentioned — off
    the SAME seeded draws. Returns `(b, mu, sig2)` stacked over t, `[t]` the state the step
    t→t+1 consumes and `[0]` the seeds.

    In torch rather than numpy, and that is a MEASUREMENT rather than a preference: on this build
    `torch.sqrt` and scalar/tensor division on float64 disagree with numpy's by up to 7.1e-15 on
    identical inputs (`(nu-2)/W` and `sqrt` both), so a numpy transcription could only be compared
    with a tolerance — and a tolerant recursion gate is a strictly weaker gate than an exact one.
    Every operator below is the one the documented equation names, in the order it names it."""
    Z = shared.t_random_numbers[0, :p.scenario_horizon]
    torch.manual_seed(seed)
    W = torch.distributions.Chi2(shared.one.new_tensor(p.Nu)).sample(Z.shape).clamp_min(1.0e-6)
    scale = torch.sqrt((p.Nu - 2.0) / W)
    lp = shared.t_Scenario_Buffer[p.linked_key]
    sigma = float(p.param.get('Sigma', 0.0))
    zero = torch.zeros_like(Z[0])
    bs = [zero + (p.b0.unsqueeze(-1) if Z.ndim == 3 else p.b0)]
    mus = [zero + float(p.param.get('Mu_0', 0.0))]
    s2s = [zero + float(p.param.get('Sig2_0', 0.0))]
    for t in range(1, Z.shape[0]):
        eta = (s2s[-1].sqrt() if p.garch else sigma) * Z[t] * scale[t]
        m, prev = mus[-1], bs[-1]
        bs.append(m + p.A * (lp[t] - lp[t - 1]) + p.Phi * (prev - m) + eta if p.slow_mean
                  else p.A * (lp[t] - lp[t - 1]) + p.Phi * prev + eta)
        mus.append(p.lam * m + (1.0 - p.lam) * bs[-1])
        s2s.append(p.g_omega + p.g_alpha * eta * eta + p.g_beta * s2s[-1])
    return torch.stack(bs), torch.stack(mus), torch.stack(s2s)


# ---------------------------------------------------------------------------
# OFF is the shipped path, bitwise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('explicit_defaults', [False, True])
@pytest.mark.parametrize('sub', [None, 8])
def test_off_reproduces_the_shipped_path_bitwise(explicit_defaults, sub):
    """The shipping world's own basis block, with the six new fields absent and then present at
    their declared 0.0 defaults, against the frozen pre-change `generate`. Outer AND inner: the
    platinum HMC world forks this process at every decision step."""
    T, B = 40, 64
    param = dict(PLATINUM)
    if explicit_defaults:
        param.update(Slow_Mean_Lambda=0.0, Mu_0=0.0,
                     G_Omega=0.0, G_Alpha=0.0, G_Beta=0.0, Sig2_0=0.0)
    sh = _shared(B, T, sub=sub)
    b0 = torch.tensor([B0]) if sub is None else torch.full((B,), B0)
    p = _basis(param, sh, T, b0)
    assert not p.slow_mean and not p.garch, 'defaults did not read as OFF'

    torch.manual_seed(99)
    expected = _head_generate(p, sh)
    torch.manual_seed(99)
    got = p.generate(sh)
    assert torch.equal(got, expected), 'OFF basis path moved off the shipped expression'
    assert (p.factor_key, 'basis_mu') not in sh.t_Scenario_Buffer
    assert (p.factor_key, 'basis_sig2') not in sh.t_Scenario_Buffer
    assert p.outer_reseed() == {} and p.inner_fork_seed(p.factor_key, sh.t_Scenario_Buffer, 3) == {}
    p.reseed_from_path(got, sh)                                       # OFF: still the base no-op


def test_off_is_the_regime_path_bitwise_too():
    """The other shipped innovation form — `Sigma_By_State` off a regime path — takes the same
    else arms."""
    T, B = 40, 64
    sh = _shared(B, T)
    regimes = torch.randint(0, 3, (T, B), generator=torch.Generator().manual_seed(5))
    param = dict(PLATINUM)
    param.pop('Sigma')
    param['Sigma_By_State'] = [6.76, 8.73, 13.39]
    p = _basis(param, sh, T, torch.tensor([B0]), regimes=regimes)
    torch.manual_seed(7)
    expected = _head_generate(p, sh)
    torch.manual_seed(7)
    assert torch.equal(p.generate(sh), expected), 'OFF regime path moved'


# ---------------------------------------------------------------------------
# The recursions, exact against numpy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('on', ['mean', 'garch', 'both'])
def test_the_recursions_match_the_reference_exactly(on):
    """float64 end to end so "exact" means exact: the same elementwise ops in the same order on
    the same seeded draws. `both` is the interaction — the GARCH innovation feeds the mean the AR
    reverts to, and nothing feeds back the other way, which is the claim that the two recursions
    compose without either one changing the other's arithmetic."""
    T, B, seed = 40, 64, 11
    param = dict(FITTED)
    if on == 'mean':
        for k in ('G_Omega', 'G_Alpha', 'G_Beta', 'Sig2_0'):
            param.pop(k)
    if on == 'garch':
        param['Slow_Mean_Lambda'] = 0.0
    sh = _shared(B, T, dtype=torch.float64)
    p = _basis(param, sh, T, torch.tensor([B0], dtype=torch.float64), dtype=torch.float64)
    assert p.slow_mean == (on != 'garch') and p.garch == (on != 'mean')

    torch.manual_seed(seed)
    got = p.generate(sh)
    b_ref, mu_ref, sig2_ref = _reference(p, sh, seed)
    assert torch.equal(got, b_ref), 'basis path off the reference recursion'
    if p.slow_mean:
        mu = sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')]
        assert torch.equal(mu, mu_ref), 'published basis_mu off the reference recursion'
        assert mu[0, 0].item() == FITTED['Mu_0'], 'Mu_0 is not the state the first step reverts to'
        assert not torch.equal(mu[1], mu[2]), 'the mean never moved — recursion inert'
    if p.garch:
        s2 = sh.t_Scenario_Buffer[(p.factor_key, 'basis_sig2')]
        assert torch.equal(s2, sig2_ref), 'published basis_sig2 off the reference recursion'
        assert s2[0, 0].item() == FITTED['Sig2_0'], 'Sig2_0 is not the first innovation variance'
        lr = FITTED['G_Omega'] / (1.0 - FITTED['G_Alpha'] - FITTED['G_Beta'])
        assert s2[0, 0].item() / lr > 10.0, 'fixture Sig2_0 sits on the unconditional — gate blind'


def test_the_inner_mode_recursions_match_the_reference_too():
    """The (T, B, B2) branch runs the same loop; the seeds broadcast across the fan-out."""
    T, B, B2, seed = 25, 8, 4, 13
    sh = _shared(B, T, sub=B2, dtype=torch.float64)
    p = _basis(FITTED, sh, T, torch.full((B,), B0, dtype=torch.float64), dtype=torch.float64)
    torch.manual_seed(seed)
    got = p.generate(sh)
    assert got.shape == (T, B, B2)
    b_ref, mu_ref, sig2_ref = _reference(p, sh, seed)
    assert torch.equal(got, b_ref)
    assert torch.equal(sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')], mu_ref)
    assert torch.equal(sh.t_Scenario_Buffer[(p.factor_key, 'basis_sig2')], sig2_ref)


def test_sigma_by_state_still_takes_precedence_over_the_garch_fields():
    """The documented precedence, and the reason the calibration stamps a flat `Sigma` when it
    fits the GARCH: declared beside `Sigma_By_State`, the GARCH fields are inert."""
    T, B = 30, 32
    sh = _shared(B, T)
    regimes = torch.randint(0, 3, (T, B), generator=torch.Generator().manual_seed(5))
    param = dict(FITTED)
    param.pop('Sigma')
    param['Sigma_By_State'] = [6.76, 8.73, 13.39]
    p = _basis(param, sh, T, torch.tensor([B0]), regimes=regimes)
    assert p.g_omega > 0.0 and not p.garch, 'GARCH fields overrode the regime form'
    torch.manual_seed(3)
    p.generate(sh)
    assert (p.factor_key, 'basis_sig2') not in sh.t_Scenario_Buffer
    assert (p.factor_key, 'basis_mu') in sh.t_Scenario_Buffer, 'the mean half must still run'


# ---------------------------------------------------------------------------
# Forking
# ---------------------------------------------------------------------------

def test_the_inner_fork_continues_the_outer_mean_and_variance_state():
    """`inner_fork_seed` at outer row t must hand the inner run the state the outer path's step
    t→t+1 consumes — which is the inner run's own step 0→1 — and the inner recursion must
    CONTINUE from it. The negative control is the same fork with the seeds withheld."""
    T, B, B2, t = 40, 8, 4, 17
    sh = _shared(B, T, dtype=torch.float64)
    p = _basis(FITTED, sh, T, torch.tensor([B0], dtype=torch.float64), dtype=torch.float64)
    torch.manual_seed(21)
    outer = p.generate(sh)
    mu_out = sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')]
    s2_out = sh.t_Scenario_Buffer[(p.factor_key, 'basis_sig2')]

    seeds = p.inner_fork_seed(p.factor_key, sh.t_Scenario_Buffer, t)
    assert torch.equal(seeds[(p.factor_key, 'mu0_inner')], mu_out[t])
    assert torch.equal(seeds[(p.factor_key, 'sig20_inner')], s2_out[t])
    assert not seeds[(p.factor_key, 'mu0_inner')].requires_grad, 'fork seed leaks the outer tape'

    inner_sh = _shared(B, T, sub=B2, dtype=torch.float64, seed=99)
    inner = _basis(FITTED, inner_sh, T, outer[t].detach(), dtype=torch.float64)
    for k, v in seeds.items():
        inner_sh.t_Scenario_Buffer[k] = v
    torch.manual_seed(31)
    inner_path = inner.generate(inner_sh)
    mu_in = inner_sh.t_Scenario_Buffer[(inner.factor_key, 'basis_mu')]
    s2_in = inner_sh.t_Scenario_Buffer[(inner.factor_key, 'basis_sig2')]

    assert torch.equal(mu_in[0], mu_out[t].unsqueeze(-1).expand(B, B2)), 'fork mu not at the outer state'
    assert torch.equal(s2_in[0], s2_out[t].unsqueeze(-1).expand(B, B2)), 'fork sig2 not at the outer state'
    assert torch.equal(inner_path[0], outer[t].unsqueeze(-1).expand(B, B2))
    # …and the recursion continues from it rather than merely starting there.
    lam = FITTED['Slow_Mean_Lambda']
    assert torch.equal(mu_in[1], lam * mu_in[0] + (1.0 - lam) * inner_path[1])

    # Negative control: withhold the seeds and the fork restarts at the calibrated pair, which
    # the fixture keeps far from the state at t — so the gate above is not measuring a tie.
    bare_sh = _shared(B, T, sub=B2, dtype=torch.float64, seed=99)
    bare = _basis(FITTED, bare_sh, T, outer[t].detach(), dtype=torch.float64)
    torch.manual_seed(31)
    bare.generate(bare_sh)
    assert (bare_sh.t_Scenario_Buffer[(bare.factor_key, 'basis_mu')][0] == FITTED['Mu_0']).all()
    assert not torch.equal(bare_sh.t_Scenario_Buffer[(bare.factor_key, 'basis_mu')][0], mu_in[0])
    assert not torch.equal(bare_sh.t_Scenario_Buffer[(bare.factor_key, 'basis_sig2')][0], s2_in[0])


def test_outer_reseed_carries_the_terminal_state_into_the_next_run():
    """The diff-ML burn-in carries the terminal basis LEVEL into the next outer run; this carries
    the state that level was generated under with it."""
    T, B = 30, 16
    sh = _shared(B, T, dtype=torch.float64)
    p = _basis(FITTED, sh, T, torch.tensor([B0], dtype=torch.float64), dtype=torch.float64)
    torch.manual_seed(41)
    p.generate(sh)
    mu_out = sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')]
    seeds = p.outer_reseed()
    assert torch.equal(seeds[(p.factor_key, 'mu0_outer')], mu_out[-1])

    nxt = _shared(B, T, dtype=torch.float64, seed=7)
    q = _basis(FITTED, nxt, T, torch.tensor([B0], dtype=torch.float64), dtype=torch.float64)
    for k, v in seeds.items():
        nxt.t_Scenario_Buffer[k] = v
    torch.manual_seed(43)
    q.generate(nxt)
    assert torch.equal(nxt.t_Scenario_Buffer[(q.factor_key, 'basis_mu')][0], mu_out[-1])


def test_observed_path_replay_refuses_the_two_recursions():
    """`reseed_from_path` has no spelling for either recursion, and the platinum walk-forward DOES
    replay this factor — so it raises rather than leave the simulated path's state published."""
    T, B = 20, 16
    sh = _shared(B, T)
    p = _basis(FITTED, sh, T, torch.tensor([B0]))
    torch.manual_seed(51)
    path = p.generate(sh)
    with pytest.raises(ValueError, match='Slow_Mean_Lambda'):
        p.reseed_from_path(path, sh)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _archive(n=1200, seed=3):
    """A basis with a slow-moving mean and clustered innovation vol, against a moving spot — so
    both switches have something to find."""
    rng = np.random.default_rng(seed)
    spot = 1700.0 + np.cumsum(rng.normal(0.0, 12.0, n))
    mu = 4.0 + 6.0 * np.sin(np.arange(n) / 250.0)
    b = np.empty(n)
    h = np.empty(n)
    b[0], h[0] = mu[0], 1.0
    for i in range(1, n):
        h[i] = 0.05 + 0.10 * (b[i - 1] - mu[i - 1]) ** 2 * 0.1 + 0.88 * h[i - 1]
        b[i] = mu[i] + 0.63 * (b[i - 1] - mu[i - 1]) + np.sqrt(h[i]) * rng.standard_t(6.0)
    return pd.DataFrame({'ObservedBasis.PLATINUM_CME.LBMA': b,
                         'CommodityPrice.PLATINUM_CME': spot},
                        index=pd.bdate_range('2020-01-01', periods=n))


def test_calibration_at_its_defaults_stamps_exactly_the_shipped_block():
    out = BasisLinkedSpotCalibration(model=None, param={}).calibrate(_archive(), 0.0)
    assert list(out.param) == ['A', 'Phi', 'Nu', 'Mu', 'Sigma_By_State', 'Calibration_DT_Years']


def test_calibration_stamps_the_slow_mean_when_the_span_is_declared():
    """λ is the span's EWMA decay and `Mu_0` the mean the NEXT observation reverts to — the
    strictly-lagged recursion the simulator runs, not a lookahead."""
    df = _archive()
    out = BasisLinkedSpotCalibration(model=None, param={'Slow_Mean_Span': 63}).calibrate(df, 0.0)
    assert out.param['Slow_Mean_Lambda'] == 1.0 - 2.0 / 64.0
    b = df['ObservedBasis.PLATINUM_CME.LBMA'].values
    assert out.param['Mu_0'] == pytest.approx(
        pd.Series(b).ewm(span=63, adjust=False).mean().values[-1], rel=1e-12)
    # The slow mean is what it is for: the deviation reverts faster than the level does.
    flat = BasisLinkedSpotCalibration(model=None, param={}).calibrate(df, 0.0)
    assert out.param['Phi'] < flat.param['Phi'], 'the slow mean absorbed no persistence'


def test_calibration_stamps_the_garch_innovation_and_drops_the_regime_form():
    """`Sigma_By_State` would take precedence over the fit, so the GARCH branch stamps the flat
    `Sigma` instead — and the stamped block must run in the model it was stamped for."""
    df = _archive()
    out = BasisLinkedSpotCalibration(
        model=None, param={'GARCH_Innovation': 'Yes'}).calibrate(df, 0.0)
    assert 'Sigma_By_State' not in out.param and out.param['Sigma'] > 0.0
    assert out.param['G_Omega'] > 0.0 and out.param['G_Alpha'] >= 0.0
    assert out.param['G_Alpha'] + out.param['G_Beta'] < 1.0, 'non-stationary innovation variance'
    assert out.param['Sig2_0'] > 0.0
    # Standardised residual, not the raw one — the correlation consolidation must not inherit
    # the heteroskedasticity (the same statement GARCHSpotCalibration's `delta` makes).
    assert abs(float(out.delta.values.std()) - 1.0) < 0.35

    T, B = 20, 16
    sh = _shared(B, T)
    p = _basis(dict(out.param, Calibration_DT_Years=DT_C), sh, T, torch.tensor([B0]))
    assert p.garch
    torch.manual_seed(61)
    assert torch.isfinite(p.generate(sh)).all()
