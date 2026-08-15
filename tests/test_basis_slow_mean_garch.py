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
  * The burn-in reseed, and the OBSERVED-PATH REPLAY: replaying the path a seeded run produced
    rebuilds that run's own state, replaying a different one publishes THAT path's, and the
    fork-index convention survives both.
  * `BasisLinkedSpotCalibration` stamps exactly the shipped block at its defaults — byte-for-byte,
    against the estimator written out here as source — and fits ONE likelihood once either switch
    is declared, with the A-versus-R-matrix identifiability measured rather than assumed.

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
| rows | T = 40 (125 for the replay), both modes | one row leaves every recursion at its seed |
| replay: a SECOND path | a different seed's, from a different `b0` | replaying the path that was just simulated cannot tell "rebuilt it" from "kept it" |
| replay: a second LINKED path | `_linked(seed=999)` | with the generating linked path in the buffer, dropping `A·ΔS` from the recovery is invisible |
| calibration: split coupling | the real archive, where A carries 49.6% and the innovation 50.4% | a fixture with the whole coupling in one channel cannot see a double count |
| calibration: `Slow_Mean_Span` vs φ | 63, generating φ = 0.65 against a level persistence of 0.99 | equal reversion rates mean the mean term is doing nothing and every mean mutant survives |

FITTED is deliberately NOT the reconciled block. The reconciled `G_Omega/G_Alpha/G_Beta` put the
unconditional variance at 11.0 against a stamped `Sig2_0` of 19.4 — a ratio of 1.8, where the
fixture's is 19 — so re-emitting the draft into this file would quietly halve the "`Sig2_0` ignored"
mutant's signal. The fixture's job is sensitivity, not currency.

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

REPLAY AND RECONCILIATION MUTANTS — thirteen more, every one RUN as a pytest plugin patching the
engine and scoring the whole file. Control: 0 failing. Zero survivors.

| mutant | killed by | count |
|---|---|---|
| HEAD's replay: raise rather than publish (what this increment replaces) | all five replay gates AND the end-to-end job | 6 |
| replay is a no-op (the state stays the DISCARDED simulation's) | all five replay gates | 5 |
| the recursions are seeded but never advanced | the same five | 5 |
| off by one: `state[t]` is what the step t−1→t consumed | the same five | 5 |
| η recovered without the `A·ΔS` term | four (not the fork-index gate, which reads only μ) | 4 |
| `_advance` re-derives `b` as `mean + η` instead of taking the level | the replayed-linked-spot gate ONLY | 1 |
| `delta` from an A-free fit beside the stamped A (the draft's own pairing) | the one-fit gate on all three extended paths, plus both byte-identity gates | 5 |
| the mean fitted as a free intercept instead of the observable EWMA | the one-fit gate ×3, the round trip, the mean gate | 5 |
| the mean read one row ahead (contemporaneous de-meaning) | the one-fit gate on the two slow-mean paths | 2 |
| two-stage: OLS mean, then a GARCH-t on its residual (the path this replaces) | nine, including both byte-identity gates | 9 |
| the GARCH fitted on the DEVIATION rather than on η | ten | 10 |
| `Sig2_0` at the unconditional variance instead of the end of the sample | the round trip and the one-fit gate ×2 | 3 |
| the defaults routed through the joint likelihood | both byte-identity gates and two more | 4 |

The mean-lookahead mutant is the weakest kill in the table and the entry that says most about where
the strength actually lives. Shifting a span-63 EWMA by one row barely moves any fitted number — the
round trip does not notice — and it is caught only because the one-fit gate rebuilds η with the
mean the SIMULATOR will run and finds it no longer reconciles with the stamped GARCH. A recovery
gate cannot see a filtration error that small; a consistency gate can.

The one-gate kill is the entry worth reading twice. `_advance` taking the level from the innovation
supplier rather than rebuilding it as `mean + η` is invisible on the gate that replays the path the
simulation just produced — there the rebuild happens to be exact to the bit — and shows up only
where the fixture supplies a linked spot the path was NOT generated against, which makes the
conditional mean land far enough from the level that the reconstruction rounds. One fixture
property, one gate, one mutant: the file's own instance of "marginal coverage is not joint
coverage".
"""
import os
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
            'Sigma': 8.079302632269696, 'Calibration_DT_Years': DT_C}

#: The completed basis study's fitted numbers (data/plat_marketdata_draft.json), plus the two
#: seeds `BasisLinkedSpotCalibration` stamps for them on data/plat_archive_sync.csv. `A` is the
#: shipping fixture's rather than the study's 0.0, which would zero the ΔS term (see above).
FITTED = dict(PLATINUM, Phi=0.633, Nu=5.31,
              Slow_Mean_Lambda=1.0 - 2.0 / 64.0, Mu_0=9.3786,
              G_Omega=0.0448, G_Alpha=0.066, G_Beta=0.887, Sig2_0=18.1455)

B0 = 12.0                                                    # observed initial basis, ≠ Mu_0

#: The synchronized 2010-2026 study archive the whole reconciliation is measured on. Untracked
#: (regenerable), so the gates that read it skip rather than fail on a clean checkout.
ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'plat_archive_sync.csv')


def _platinum_archive():
    if not os.path.exists(ARCHIVE):
        pytest.skip('platinum archive not present (data/ is untracked)')
    df = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
    return df[['ObservedBasis.PLATINUM_CME.LBMA', 'CommodityPrice.PLATINUM_CME']].dropna()


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
    assert (p.factor_key, 'basis_mu') not in sh.t_Scenario_Buffer
    assert (p.factor_key, 'basis_sig2') not in sh.t_Scenario_Buffer


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


# ---------------------------------------------------------------------------
# Observed-path replay
# ---------------------------------------------------------------------------

def _replay_reference(p, path, linked, lam):
    """The two recursions along a GIVEN path, written from the class's equations rather than from
    `_advance`: η_t is the realised deviation from the conditional mean at t, the mean recursion
    consumes the OBSERVED level, and nothing else is read. `[t]` is the state the step t→t+1
    consumes, `[0]` the seeds — the published fork-index convention."""
    zero = torch.zeros_like(path[0])
    mus = [zero + float(p.param['Mu_0'])]
    s2s = [zero + float(p.param['Sig2_0'])]
    for t in range(1, path.shape[0]):
        mean = mus[-1] + p.A * (linked[t] - linked[t - 1]) + p.Phi * (path[t - 1] - mus[-1])
        eta = path[t] - mean
        mus.append(lam * mus[-1] + (1.0 - lam) * path[t])
        s2s.append(p.g_omega + p.g_alpha * eta * eta + p.g_beta * s2s[-1])
    return torch.stack(mus), torch.stack(s2s)


def _replay(param, T, B, dtype, seed=51, path=None, linked=None):
    """Generate, then replay `path` (default: the generated one) through `reseed_from_path`, having
    first wiped what `generate` published — so what comes back cannot be the simulated run's."""
    sh = _shared(B, T, dtype=dtype)
    p = _basis(param, sh, T, torch.tensor([B0], dtype=dtype), dtype=dtype)
    torch.manual_seed(seed)
    sim = p.generate(sh)
    out = {k: sh.t_Scenario_Buffer.pop((p.factor_key, k)).clone()
           for k in ('basis_mu', 'basis_sig2')}
    if linked is not None:
        sh.t_Scenario_Buffer[p.linked_key] = linked
    p.reseed_from_path(sim if path is None else path, sh)
    return p, sh, sim, out


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_replaying_a_simulated_path_reproduces_that_run_s_own_state(dtype):
    """The invariant the walk-forward needs: hand the replay the path a seeded simulation produced
    and it must rebuild that simulation's own μ and σ².

    EXACTNESS IS ASYMMETRIC, and this is the measurement rather than an assumption. μ comes back
    BITWISE, structurally: it is a function of the observed LEVEL, and `_advance` takes the level
    from the innovation supplier rather than rebuilding it, so replay consumes the same floats the
    simulation did. σ² CANNOT: it is a function of η, and `fl(b − mean)` differs from the η that was
    drawn by the rounding error of the forward pass's own final addition — 1.0e-15 relative in
    float64, 5.9e-7 in float32, measured here on 125 rows. No spelling of this replay recovers η
    exactly from a rounded sum, so the σ² arm is asserted at 1e-12 / 1e-5 and the mutation matrix
    above is what makes that tolerance mean something: every mutant is orders of magnitude louder."""
    T, B = 125, 64
    p, sh, sim, sim_state = _replay(FITTED, T, B, dtype)
    mu = sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')]
    s2 = sh.t_Scenario_Buffer[(p.factor_key, 'basis_sig2')]
    assert torch.equal(mu, sim_state['basis_mu']), 'replayed mean is not the simulation\'s own'
    rel = ((s2 - sim_state['basis_sig2']).abs() / sim_state['basis_sig2']).max().item()
    assert rel < (1.0e-12 if dtype == torch.float64 else 1.0e-5), rel
    # …and the fixture is not degenerate: both recursions move, far, over the replayed rows.
    assert (mu[-1] - mu[0]).abs().max().item() > 1.0
    assert (s2.max() / s2.min()).item() > 5.0


def test_the_replay_is_the_replayed_path_s_recursion_not_the_simulated_one_s():
    """The hole the raise stood in for: replay a DIFFERENT path — a second seed's — and the
    published state must be THAT path's recursion, matched against an independent reference, with
    the discarded simulation's state nowhere in it. This is the walk-forward's actual shape: the
    driver's realised basis replaces a simulated one that was never anything but a placeholder."""
    T, B, dtype = 125, 64, torch.float64
    sh2 = _shared(B, T, dtype=dtype, seed=8)
    q = _basis(FITTED, sh2, T, torch.tensor([B0 + 4.0], dtype=dtype), dtype=dtype)
    torch.manual_seed(123)
    other = q.generate(sh2)

    p, sh, sim, sim_state = _replay(FITTED, T, B, dtype, path=other)
    assert not torch.equal(other, sim), 'the two paths coincide — gate blind'
    mu_ref, s2_ref = _replay_reference(
        p, other, sh.t_Scenario_Buffer[p.linked_key], FITTED['Slow_Mean_Lambda'])
    assert torch.equal(sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')], mu_ref)
    assert torch.equal(sh.t_Scenario_Buffer[(p.factor_key, 'basis_sig2')], s2_ref)
    # negative control: the state the DISCARDED simulation published is not what is there now
    assert not torch.equal(sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')], sim_state['basis_mu'])
    assert not torch.equal(sh.t_Scenario_Buffer[(p.factor_key, 'basis_sig2')], sim_state['basis_sig2'])
    # `outer_reseed` follows it, so a continuing replay carries the replayed terminal state
    assert torch.equal(p.outer_reseed()[(p.factor_key, 'mu0_outer')], mu_ref[-1])
    assert torch.equal(p.outer_reseed()[(p.factor_key, 'sig20_outer')], s2_ref[-1])


def test_the_replay_recovers_eta_against_the_replayed_linked_spot():
    """η_t is `b_t` minus the model's own conditional mean, and that mean carries `A·ΔS`. A world
    replaying the parent spot too hands this factor a DIFFERENT ΔS, and the recovered innovation
    must follow it — the calc publishes each factor's path before the next process generates, so
    the buffer read is the replayed one. Dropping the ΔS term from the recovery moves σ² by more
    than 5% on this fixture, which is the negative control below."""
    T, B, dtype = 125, 64, torch.float64
    other_linked = _linked((T, B), dtype, seed=999)
    p, sh, sim, _ = _replay(FITTED, T, B, dtype, linked=other_linked)
    mu_ref, s2_ref = _replay_reference(p, sim, other_linked, FITTED['Slow_Mean_Lambda'])
    assert torch.equal(sh.t_Scenario_Buffer[(p.factor_key, 'basis_sig2')], s2_ref)
    # the mean recursion reads only the basis path, so it is INVARIANT to the linked substitution —
    # which is what makes the σ² arm above a measurement of the ΔS term and nothing else
    mu_same, _ = _replay_reference(p, sim, sh.t_Scenario_Buffer[p.linked_key], FITTED['Slow_Mean_Lambda'])
    assert torch.equal(mu_ref, mu_same)
    # the mutant, spelled as data: a FLAT linked path is "η recovered without the A·ΔS term"
    _, s2_no_ds = _replay_reference(p, sim, torch.full_like(other_linked, 1700.0),
                                    FITTED['Slow_Mean_Lambda'])
    assert (s2_no_ds / s2_ref - 1.0).abs().max().item() > 0.05


SHIPPING = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fixtures', 'platinum_hedge_shipping.json')
CALIBRATED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'fixtures', 'data', 'MarketDataRF_platinum_calibrated_cme.json')


def _replay_job(tmp_path, extensions, seed=4):
    """The platinum walk-forward's SHAPE, small: `Observed_Scenario` substitutes a realised basis
    path into the shipping fixture. The fixture is a template — the extended block is an in-memory
    override, and `Sigma_By_State` has to give way to a flat `Sigma` or the documented precedence
    leaves the GARCH inert and the run gates half of what it claims to."""
    import json
    import derivus as rf
    cfg = json.load(open(SHIPPING))
    calc = cfg['Calc']['Calculation']
    calc.update({'Execution_Mode': 'simulate_only', 'Batch_Size': 32, 'Simulation_Batches': 1,
                 'Random_Seed': 1})
    calc['Hedging_Problem'].pop('Solver', None)
    rng = np.random.default_rng(seed)
    npz = str(tmp_path / 'observed.npz')
    np.savez(npz, **{'ObservedBasis.PLATINUM_CME.LME_CME':
                     15.5 + np.cumsum(rng.normal(0.0, 1.5, 500))})
    calc['Observed_Scenario'] = npz
    if extensions:
        blk = dict(json.load(open(CALIBRATED))['MarketData']['Price Models'][
                       'BasisLinkedSpotModel.PLATINUM_CME.LME_CME'],
                   Slow_Mean_Lambda=0.96875, Mu_0=9.3786,
                   G_Omega=0.0448, G_Alpha=0.066, G_Beta=0.887, Sig2_0=18.1455)
        blk['Sigma'] = float(np.mean(blk.pop('Sigma_By_State')))
        cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Models'] = {
            'BasisLinkedSpotModel.PLATINUM_CME.LME_CME': blk}
    cx = rf.Context()
    cx.load_json((json.dumps(cfg, default=str), 'wf_replay.json'))
    return cx.run_job()[1]


def test_a_walk_forward_replay_runs_the_extended_basis_and_prices_it_identically(tmp_path,
                                                                                 monkeypatch):
    """END TO END, and the gate the hole needed: `experiments/production_walk_forward.py` puts this
    factor in `Observed_Scenario`, so the extended block reaching production would have RAISED on
    the first roll. Nothing above sees that — every replay gate here calls `reseed_from_path`
    itself, and the wiring (substitute, then ask the process to re-derive) lives in the calc.

    Under replay the basis path is DATA, and neither recursion consumes randomness, so the two
    extensions are pure state bookkeeping: the priced liability must be bitwise identical with them
    on and off. That is a strong statement precisely because it is not about the state — it says the
    replay changed the published state and nothing else.

    Anti-placebo: the same run with `reseed_from_path` mined to raise MUST fail, or the assertion
    above is being made about a code path the calc never reaches."""
    off = _replay_job(tmp_path, False)
    on = _replay_job(tmp_path, True)
    assert torch.equal(off.bundle.liability_mtm, on.bundle.liability_mtm)

    def mined(self, simulated, shared_mem):
        raise ValueError('reseed_from_path reached')

    monkeypatch.setattr(BasisLinkedSpotModel, 'reseed_from_path', mined)
    with pytest.raises(ValueError, match='reached'):
        _replay_job(tmp_path, True)


def test_the_replayed_state_keeps_the_fork_index_convention():
    """`state[t]` is what the step t→t+1 consumes — the property `inner_fork_seed` reads as a plain
    index. Asserted directly on the replayed arrays against the observed levels, which is the
    off-by-one gate: `state[0]` is the seed and `state[t]` has already consumed `b_t`."""
    T, B, dtype = 60, 32, torch.float64
    p, sh, sim, _ = _replay(FITTED, T, B, dtype)
    mu = sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')]
    lam = FITTED['Slow_Mean_Lambda']
    assert (mu[0] == FITTED['Mu_0']).all(), 'row 0 is not the seed'
    assert torch.equal(mu[1:], lam * mu[:-1] + (1.0 - lam) * sim[1:]), 'mean is off by a row'
    seeds = p.inner_fork_seed(p.factor_key, sh.t_Scenario_Buffer, 17)
    assert torch.equal(seeds[(p.factor_key, 'mu0_inner')], mu[17])


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


def _model_archive(n=2500, seed=7, a=-0.03, phi=0.65, span=63, nu=6.0, rho=0.0,
                   omega=0.08, alpha=0.11, beta=0.88, sigma=None, ds_sigma=24.0):
    """An archive the MODEL generated — its own two recursions in numpy against a moving spot, with
    the spot/basis coupling placed in `a`, in `rho`, or split between them. `sigma` (a float) fixes
    the innovation std instead of running the GARCH, which is what makes the identifiability algebra
    below EXACT rather than Jensen-approximate. One χ² shared by both innovations, so the pair is
    elliptical t — the law `generate` draws from."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], n)
    eps = z * np.sqrt((nu - 2.0) / rng.chisquare(nu, n))[:, None]
    ds = ds_sigma * eps[:, 0]
    lam = 1.0 - 2.0 / (span + 1.0)
    b, mu, sig2 = np.zeros(n), 0.0, (sigma ** 2 if sigma else omega / (1.0 - alpha - beta))
    for i in range(1, n):
        eta = np.sqrt(sig2) * eps[i, 1]
        b[i] = mu + a * ds[i] + phi * (b[i - 1] - mu) + eta
        mu = lam * mu + (1.0 - lam) * b[i]
        if sigma is None:
            sig2 = omega + alpha * eta * eta + beta * sig2
    return pd.DataFrame({'ObservedBasis.PLATINUM_CME.LBMA': b,
                         'CommodityPrice.PLATINUM_CME': 1700.0 + np.cumsum(ds)},
                        index=pd.bdate_range('2010-01-01', periods=n))


def _coupling(param, df, delta):
    """The two channels the spot/basis coupling can ride, in the units of the one-step covariance:
    `A·Var(ΔS)` through the conditional mean, and `Cov(η, ΔS)` through the innovation the framework
    consolidates into the R matrix. η is recomputed from the STAMPED block, because that is what the
    simulator will produce; the third return is the OBSERVED total. One equation in two unknowns —
    exactly the carry curve's Γ-versus-ρ. The last pair is the consistency the draft JSON broke:
    `delta` must be that same η up to scale, or the two ends came from different fits."""
    b = df['ObservedBasis.PLATINUM_CME.LBMA'].astype(np.float64).values
    ds = np.diff(df['CommodityPrice.PLATINUM_CME'].astype(np.float64).values)
    lam = param.get('Slow_Mean_Lambda', 0.0)
    mu = (pd.Series(b).ewm(span=2.0 / (1.0 - lam) - 1.0, adjust=False).mean().values[:-1]
          if lam else 0.0)
    ahead = (b[1:] - mu) - param['Phi'] * (b[:-1] - mu)       # b_t less its mean-reverting part
    eta = ahead - param['A'] * ds
    return (param['A'] * ds.var(ddof=1), np.cov(eta, ds)[0, 1], np.cov(ahead, ds)[0, 1],
            np.corrcoef(eta, ds)[0, 1], np.corrcoef(delta.values[:, 0], ds)[0, 1])


def _head_calibrate(df):
    """`BasisLinkedSpotCalibration.calibrate` at its DEFAULTS as of 68368f1 — OLS for (a, φ), a
    moment-matched ν off the residual's excess kurtosis, per-regime σ from rolling-vol terciles —
    kept as SOURCE so the byte-identity gate reads as the two estimators side by side. Every world
    in `tests/fixtures` was calibrated with this one; the reconciliation is not allowed to move it."""
    from scipy import stats as scipy_stats
    b = df['ObservedBasis.PLATINUM_CME.LBMA'].astype(np.float64).values
    dlme = np.diff(df['CommodityPrice.PLATINUM_CME'].astype(np.float64).values)
    y, X = b[1:], np.column_stack([dlme, b[:-1]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    eta = y - X @ coef
    nu = float(np.clip(4.0 + 6.0 / max(float(scipy_stats.kurtosis(eta, fisher=True)), 1.0e-3),
                       3.0, 50.0))
    rv = pd.Series(dlme).rolling(21, min_periods=21).std().values
    valid = ~np.isnan(rv)
    q = np.quantile(rv[valid], [1.0 / 3, 2.0 / 3])
    terc = np.where(rv > q[1], 2, np.where(rv > q[0], 1, 0))
    terc[~valid] = 1
    sbs = [float(eta[terc == s].std()) if (terc == s).sum() > 1 else float(eta.std())
           for s in range(3)]
    return {'A': float(coef[0]), 'Phi': float(coef[1]), 'Nu': nu,
            'Sigma_By_State': sbs, 'Calibration_DT_Years': 1.0 / 252.0}, eta


def test_calibration_at_its_defaults_stamps_exactly_the_shipped_block():
    out = BasisLinkedSpotCalibration(model=None, param={}).calibrate(_archive(), 0.0)
    assert list(out.param) == ['A', 'Phi', 'Nu', 'Sigma_By_State', 'Calibration_DT_Years']


@pytest.mark.parametrize('archive', ['synthetic', 'platinum'])
def test_the_default_estimator_is_byte_identical_to_the_one_the_shipped_worlds_used(archive):
    """FORBIDDEN to move: with neither switch declared the block is the OLS-plus-moment-matched-ν
    one every calibrated platinum world carries, to the last bit — parameters AND the `delta`
    column the correlation consolidation reads. The joint likelihood is the EXTENDED path's, and
    that separation is the whole of what makes this a reconciliation rather than a revaluation."""
    df = _archive() if archive == 'synthetic' else _platinum_archive()
    head, eta = _head_calibrate(df)
    out = BasisLinkedSpotCalibration(model=None, param={}).calibrate(df, 0.0)
    assert list(out.param) == list(head)
    for k, v in head.items():
        assert out.param[k] == v, k
    assert np.array_equal(out.delta.values[:, 0], eta)


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


# ---------------------------------------------------------------------------
# The reconciled estimator: one likelihood, and the identifiability it buys
# ---------------------------------------------------------------------------

CANON = {'Slow_Mean_Span': 63, 'GARCH_Innovation': 'Yes'}


def test_the_joint_fit_recovers_the_block_that_generated_the_archive():
    """The round trip, on 2500 rows of the model's own output: one likelihood recovers (a, φ, ω, α,
    β, ν) together, plus the two seeds. It is a genuine recovery — the archive is built by the two
    recursions `generate` runs, not by the estimator's own algebra."""
    gen = dict(a=-0.03, phi=0.65, nu=6.0, omega=0.08, alpha=0.11, beta=0.88)
    out = BasisLinkedSpotCalibration(model=None, param=CANON).calibrate(_model_archive(**gen), 0.0)
    assert out.param['A'] == pytest.approx(gen['a'], rel=0.15)
    assert out.param['Phi'] == pytest.approx(gen['phi'], abs=0.03)
    assert out.param['Nu'] == pytest.approx(gen['nu'], rel=0.25)
    assert out.param['G_Alpha'] == pytest.approx(gen['alpha'], rel=0.35)
    assert out.param['G_Beta'] == pytest.approx(gen['beta'], abs=0.04)
    assert out.param['G_Omega'] == pytest.approx(gen['omega'], rel=0.60)
    assert out.param['Slow_Mean_Lambda'] == 1.0 - 2.0 / 64.0
    # …and the stamped seeds are TODAY's state, not the sample's average: Sig2_0 is the variance of
    # the NEXT observation, so it must track the end of the sample rather than the unconditional.
    lr = out.param['G_Omega'] / (1.0 - out.param['G_Alpha'] - out.param['G_Beta'])
    assert out.param['Sig2_0'] != pytest.approx(lr, rel=0.02), 'Sig2_0 collapsed to the LR variance'


def test_the_loading_and_the_innovation_correlation_are_one_coupling_split_two_ways():
    """A and ρ(η, ΔS) are COLLINEAR, and this is the gate that says the split does not matter.

    The one-step covariance the two channels produce is Cov(b_t − E_{t-1}[b_t], ΔS) = A·Var(ΔS) +
    Cov(η, ΔS): one equation in two unknowns, so a pure-A world and a pure-ρ world with the same
    covariance are the same joint law and no regression can tell them apart. It is the carry
    curve's Γ-versus-ρ one factor over, and it is why the scratch study's A = 0 beside a −0.358
    spot/basis correlation and the class's A ≠ 0 beside a near-zero one are BOTH right — and why
    taking one from each is the defect.

    Two archives are generated with the same coupling, one carrying it entirely in A with
    independent innovations and one entirely in ρ with A = 0, and the calibration returns the same
    A from both. Constant-σ worlds, so Cov(η, ΔS) = ρ·σ·σ_ΔS holds exactly rather than up to
    Jensen on E[σ_t].

    The measured gap is 11%, and it does NOT shrink with the sample (21% at 2500 rows, 10.6% at
    6000, 11.7% at 12000): the ρ world is misspecified for a model that assumes an independent
    innovation, so the t-likelihood's pseudo-true loading sits a little above the covariance ratio.
    What does hold in both, and is the half that matters, is that the residual left for the
    framework's R matrix is empty — so the simulator gets the coupling once."""
    a0, sigma, ds_sigma = -0.03, 3.0, 24.0
    rho = a0 * ds_sigma / sigma                                        # the same Cov(η+A·ΔS, ΔS)
    fitted = {}
    for name, a, r in (('A', a0, 0.0), ('rho', 0.0, rho)):
        df = _model_archive(n=8000, a=a, rho=r, sigma=sigma, ds_sigma=ds_sigma)
        out = BasisLinkedSpotCalibration(
            model=None, param={'Slow_Mean_Span': 63}).calibrate(df, 0.0)
        a_ch, rho_ch, obs, corr_eta, corr_delta = _coupling(out.param, df, out.delta)
        assert a_ch + rho_ch == pytest.approx(obs, rel=1.0e-9), name
        assert abs(corr_eta) < 0.05, f'{name}: the fit left {corr_eta:.3f} of coupling behind'
        fitted[name] = out.param['A']
    assert fitted['A'] == pytest.approx(fitted['rho'], rel=0.20), fitted
    assert fitted['rho'] == pytest.approx(a0, rel=0.20), (fitted, a0)


@pytest.mark.parametrize('param', [CANON, {'Slow_Mean_Span': 63}, {'GARCH_Innovation': 'Yes'}])
def test_the_stamped_loading_and_the_stamped_delta_come_from_one_fit(param):
    """THE ANTI-DOUBLE-COUNT GATE. The framework builds the spot/basis correlation from `delta`, so
    the two ends of the coupling are stamped in two different PLACES — the block and the R matrix —
    and they are consistent only if they are two halves of one fit. The gate reconstructs the whole
    fit from the stamped block alone: η from (A, φ, λ), then σ_t = η/`delta`, and that σ_t must
    satisfy the stamped GARCH recursion σ²_t = ω + α·η²_{t-1} + β·σ²_{t-1} — which pins A, φ, λ, ω,
    α, β and `delta` to ONE estimation, to 4.4e-13. Nothing outside a single joint fit can pass it.

    Measured on the real platinum archive, where the split is nearly even: A carries −13.06 of the
    −26.36 and the innovation carries −13.29. Take the class's A and pair it with an A = 0 fit's
    residual — which is what `data/plat_marketdata_draft.json` did across its two blocks — and the
    simulated coupling is 150% of the observed one."""
    df = _platinum_archive()
    out = BasisLinkedSpotCalibration(model=None, param=param).calibrate(df, 0.0)
    a_ch, rho_ch, obs, corr_eta, corr_delta = _coupling(out.param, df, out.delta)
    assert a_ch + rho_ch == pytest.approx(obs, rel=1.0e-9), (a_ch, rho_ch, obs)
    assert abs(a_ch / obs) > 0.2 and abs(rho_ch / obs) > 0.2, (
        'the fixture put the whole coupling in one channel — the gate cannot see a double count')

    b = df['ObservedBasis.PLATINUM_CME.LBMA'].astype(np.float64).values
    ds = np.diff(df['CommodityPrice.PLATINUM_CME'].astype(np.float64).values)
    lam = out.param.get('Slow_Mean_Lambda', 0.0)
    mu = (pd.Series(b).ewm(span=2.0 / (1.0 - lam) - 1.0, adjust=False).mean().values[:-1]
          if lam else 0.0)
    eta = (b[1:] - mu) - out.param['Phi'] * (b[:-1] - mu) - out.param['A'] * ds
    delta = out.delta.values[:, 0]
    if 'G_Omega' not in out.param:
        assert delta == pytest.approx(eta, rel=1e-9), 'delta is not the stamped fit\'s innovation'
        return
    h = (eta / delta) ** 2                                  # σ²_t implied by the two stamped ends
    pred = out.param['G_Omega'] + out.param['G_Alpha'] * eta[:-1] ** 2 + out.param['G_Beta'] * h[:-1]
    assert pred == pytest.approx(h[1:], rel=1.0e-9), 'delta and the GARCH block are two fits'
    # Sig2_0 is `h[-1]`, the variance OF the last observation - `garch11_t_mle`'s documented
    # one-step-stale convention, shared with `GARCHSpotCalibration`'s H0. Advancing one more step
    # would give 17.25 against the stamped 19.36, so this pins WHICH end of the sample is stamped.
    assert out.param['Sig2_0'] == pytest.approx(h[-1], rel=1.0e-9)


def test_the_mean_the_fit_sees_is_the_mean_the_simulator_will_run():
    """μ_t is an OBSERVABLE, not a parameter: the fit de-means with the same strictly-lagged EWMA
    recursion `generate` runs, under the same likelihood as the AR — which is the whole content of
    "one likelihood". Two things follow and are asserted here: `Mu_0` is the mean the NEXT
    observation reverts to, and the fitted φ is the DEVIATION's persistence rather than the level's,
    which on this archive are 0.65 and 0.99."""
    df = _model_archive(phi=0.65, span=63)
    b = df['ObservedBasis.PLATINUM_CME.LBMA'].values
    out = BasisLinkedSpotCalibration(model=None, param=CANON).calibrate(df, 0.0)
    assert out.param['Mu_0'] == pytest.approx(
        pd.Series(b).ewm(span=63, adjust=False).mean().values[-1], rel=1e-12)
    assert out.param['Phi'] == pytest.approx(0.65, abs=0.03)
    level = BasisLinkedSpotCalibration(
        model=None, param={'GARCH_Innovation': 'Yes'}).calibrate(df, 0.0)
    assert level.param['Phi'] > out.param['Phi'] + 0.2, (
        'the level and the deviation revert at the same rate — the fixture has no slow mean')


def test_the_loss_moves_the_split_but_not_the_total():
    """THE RECONCILIATION, in one measurement, on the real archive. The scratch study and the class
    disagreed about A because they were minimising different things, and the collinear line above
    is where that disagreement lives: the Gaussian loss and the Student-t loss land on different
    POINTS of it and on the same TOTAL.

    OLS puts the entire coupling in the loading — A = −0.0461, and its residual is orthogonal to ΔS
    by construction, so the R-matrix channel is exactly zero. The joint t-GARCH likelihood
    downweights the tail days where the ΔS coupling is largest and halves it — A = −0.0227, ratio
    2.03 — leaving the other half for the innovation. Channels −26.514/−0.000 against
    −13.063/−13.294; totals −26.514 and −26.358, 0.6% apart. Every number a hedge sees comes from
    the total, which is why the split being unidentified is a property of the data and not a defect,
    and why BOTH ends have to be stamped by the same fit."""
    df = _platinum_archive()
    b = df['ObservedBasis.PLATINUM_CME.LBMA'].astype(np.float64).values
    ds = np.diff(df['CommodityPrice.PLATINUM_CME'].astype(np.float64).values)
    mu = pd.Series(b).ewm(span=63, adjust=False).mean().values[:-1]
    X = np.column_stack([ds, b[:-1] - mu])
    coef = np.linalg.lstsq(X, b[1:] - mu, rcond=None)[0]
    gauss = {'A': float(coef[0]), 'Phi': float(coef[1]), 'Slow_Mean_Lambda': 1.0 - 2.0 / 64.0}
    g_a, g_rho, g_obs, _, _ = _coupling(
        gauss, df, pd.DataFrame({'x': (b[1:] - mu) - X @ coef}))
    out = BasisLinkedSpotCalibration(model=None, param=CANON).calibrate(df, 0.0)
    t_a, t_rho, t_obs, _, _ = _coupling(out.param, df, out.delta)

    assert abs(g_rho / g_obs) < 0.001, 'the Gaussian fit did not put the whole coupling in A'
    assert abs(t_rho / t_obs) > 0.4, 'the t fit did not move any of it into the innovation'
    assert abs(gauss['A'] / out.param['A']) > 1.8, (gauss['A'], out.param['A'])
    assert g_obs == pytest.approx(t_obs, rel=0.02), (g_obs, t_obs)
