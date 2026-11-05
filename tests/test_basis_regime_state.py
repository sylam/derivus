"""The band basis's stress regime as world state: `BasisLinkedSpotModel` under `Band_Mixture` carries
P(stress | the innovations so far) as a recursion beside its slow mean and variance, forward and on a
replay alike, and under `Reveal_Regime` 'Yes' shows it to the hedging value net before the level.
`BasisLinkedSpotCalibration` fits the chain by Baum-Welch.

The fit is a calibration document through `Config.calibrate_factors` (`test_chained_basis._fit_band`);
the reveal and the fork are the canonical hedging fixture (`test_action_space_and_artifact._cfg`)
replaying an observed scenario, the belief read off the replay's `BASIS_REPLAY` debug line and each
fork's start off `BASIS_FORK`.
"""
import ast
import logging

import numpy as np
import pandas as pd
import torch

from test_action_space_and_artifact import BAND_BASIS, LAST_STEP, _cfg, _run
from test_chained_basis import _band_basis, _fit_band

#: The row of the observed basis's one large innovation, among the last decision steps.
JUMP = LAST_STEP - 4


def _chain(n, stay_quiet, stay_stress, sds, rng):
    """`n` innovations of a two-state Markov chain started quiet, Gaussian at `sds` = (quiet, stress)."""
    state, u = np.zeros(n, int), rng.random(n)
    for t in range(1, n):
        state[t] = u[t] >= stay_quiet if state[t - 1] == 0 else u[t] < stay_stress
    return np.asarray(sds)[state] * rng.standard_normal(n)


def _observed(tmp_path):
    """An observed scenario of 160 daily rows: a random-walk spot and a `BAND_BASIS` basis drawn by its
    own law from its `Mu_0` on innovations of sd 0.4, but for one of 12 at `JUMP`: its path and the basis."""
    rng = np.random.default_rng(3)
    spot = 2055.0 + np.cumsum(rng.normal(0.0, 25.0, 160))
    eta = rng.normal(0.0, 0.4, 160)
    eta[JUMP] = 12.0
    lam, mu, basis = BAND_BASIS['Slow_Mean_Lambda'], BAND_BASIS['Mu_0'], np.full(160, BAND_BASIS['Mu_0'])
    for t in range(1, 160):
        d = basis[t - 1] - mu
        pull = BAND_BASIS['Band_Beta'] * np.sign(d) * max(abs(d) - BAND_BASIS['Band_Kappa'], 0.0)
        basis[t] = basis[t - 1] + BAND_BASIS['A'] * (spot[t] - spot[t - 1]) - pull + eta[t]
        mu = lam * mu + (1.0 - lam) * basis[t]
    path = str(tmp_path / 'observed.npz')
    np.savez(path, **{'CommodityPrice.PLATINUM_LME': spot, 'ObservedBasis.PLATINUM_LME.LME_CME': basis})
    return path, basis


def _replayed(caplog, path, reveal, t_min=LAST_STEP, save=None):
    """`_cfg` on two paths a batch replaying `path` from the calibrated t0, swept from `t_min`, its basis
    `BAND_BASIS` stating `Reveal_Regime` `reveal`, saving its value fn to `save`: the result, the replay's
    per-row recursions on its first path and each fork's starting belief per outer path."""
    cfg = _cfg(batch=4, save=save)
    calc = cfg['Calc']['Calculation']
    calc['Observed_Scenario'] = path
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'No'
    hp['Solver']['T_Min'] = t_min
    cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Models'] = {
        'BasisLinkedSpotModel.PLATINUM_LME.LME_CME': dict(BAND_BASIS, Reveal_Regime=reveal)}
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        result = _run(cfg)
    lines = [r.getMessage() for r in caplog.records]
    replay = ast.literal_eval(next(m for m in lines if m.startswith('BASIS_REPLAY')).split(' ', 2)[2])
    forks = [ast.literal_eval(m.split(' ', 1)[1]) for m in lines if m.startswith('BASIS_FORK')]
    return result, replay, forks


def test_the_fit_recovers_a_persistent_stress_chain(tmp_path):
    """A calibration document on 3,000 business days of a `Band_Mixture` basis drawn from the fit's own
    law (A 0.05, a zero-width band, pull 0.3, the 63-day slow mean) on innovations of a two-state
    Markov chain staying quiet with 0.97 and in stress with 0.9, at sds 1 and 4, stationary in stress
    with 0.03 / 0.13 = 0.2308: the Baum-Welch fit reads `Mix_Stay_Stress` within 0.05 of 0.9 (0.8982),
    `Mix_Q_Stress` within 0.05 of 0.2308 (0.2217) and the two sds within 15% (1.0002, 4.0878), and
    writes `Reveal_Regime` as the calibration states it, 'No' by default. 6 s.

    Killing mutation: the static mixture by EM with the stress persistence read as the share of
    posterior-stress days followed by another, the fit Baum-Welch replaced (0.4548)."""
    rng = np.random.default_rng(11)
    n = 3000
    dates = pd.bdate_range('2014-01-01', periods=n)
    spot = 1000.0 + np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 15.0, n - 1)]))
    eta = _chain(n, 0.97, 0.9, (1.0, 4.0), rng)
    fitted = _fit_band(tmp_path, dates, spot, _band_basis(spot, eta, kappa=0.0))
    q, stay = fitted['Mix_Q_Stress'], fitted['Mix_Stay_Stress']
    assert abs(stay - 0.9) < 0.05 and abs(q - 0.03 / 0.13) < 0.05, (stay, q)
    assert abs(fitted['Mix_Sigma_Quiet'] - 1.0) < 0.15 and abs(fitted['Mix_Sigma_Stress'] / 4.0 - 1.0) < 0.15, fitted
    assert fitted['Reveal_Regime'] == 'No', fitted


def test_the_revealed_belief_rises_on_a_large_innovation_and_decays_over_quiet_rows(tmp_path, caplog):
    """The fixture replaying a band basis whose innovations are quiet but for one large one, swept at its
    last decision step: under `Reveal_Regime` 'Yes' the value net reads one market column more than under
    'No' (9 against 8), the frame the 'No' one with the replay's published belief at the step inserted
    before the level, to the bit; and the replayed belief is 1.0 in float32 on the row of the large
    innovation against 0.0071 the row before, then falls on each of the four quiet rows after it
    (0.531, 0.162, 0.042, 0.015). 4 s.

    Killing mutations: the belief held at its seed (one value on every row); the belief left out of the
    reveal (8 against 8)."""
    path, basis = _observed(tmp_path)
    shown, hidden = str(tmp_path / 'value_fn_shown.pt'), str(tmp_path / 'value_fn_hidden.pt')
    yes, replay, _ = _replayed(caplog, path, 'Yes', save=shown)
    no, _, _ = _replayed(caplog, path, 'No', save=hidden)
    md = [r.evaluation_summary['diagnostics']['market_dim'] for r in (no, yes)]
    before, after = (torch.load(c, map_location='cpu', weights_only=False)['m_mean'].tolist() for c in (hidden, shown))
    belief = replay['belief']
    col = before.index(float(np.float32(basis[LAST_STEP])))
    assert md[1] == md[0] + 1, md
    assert after[:col] + after[col + 1:] == before and after[col] == belief[LAST_STEP], (after, before)
    assert belief[JUMP] > 0.9 > 0.1 > belief[JUMP - 1], belief[JUMP - 2:JUMP + 5]
    assert all(belief[t + 1] < belief[t] for t in range(JUMP, JUMP + 4)), belief[JUMP:JUMP + 5]


def test_a_fork_starts_from_the_outer_path_s_belief_at_the_fork_row(tmp_path, caplog):
    """The fixture replaying the band basis of the large innovation, swept from six rows before its last
    decision step under `Reveal_Regime` 'Yes', forks each outer path from the replay's published belief
    at the fork row: the forks' starting beliefs are, to the bit, the replay's at the four decision
    rows, Thursday to Tuesday over a weekend, the large innovation's Friday (1.0) among them. 1 s.

    Killing mutations: the fork's seed unread, every fork starting at `Mix_P0_Stress` (0.1); the belief
    read a row early (Wednesday's 0.00735 for Thursday's 0.00714, Thursday's for Friday's 1.0, Sunday's
    0.162 for Monday's 0.042)."""
    path, _ = _observed(tmp_path)
    _, replay, forks = _replayed(caplog, path, 'Yes', t_min=LAST_STEP - 5)
    base = pd.Timestamp(_cfg()['Calc']['Calculation']['Base_Date']['.Timestamp'])
    decisions = [t for t in range(LAST_STEP - 5, LAST_STEP + 1) if (base + pd.Timedelta(days=t)).dayofweek < 5]
    started = {b for fork in forks for b in fork}
    assert len(decisions) == 4 and started == {replay['belief'][t] for t in decisions} and 1.0 in started,         (sorted(started), decisions)
