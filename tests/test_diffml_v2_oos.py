"""DiffSolverV2 (clean-room framework-native diff-ML hedger) — bounded value + OUT-OF-SAMPLE
hedging gate on the platinum deal.

Two claims:
  • the value stays BOUNDED at depth — `max|Y_boot|` and V_0 small;
  • the greedy policy HEDGES out-of-sample — on held-out paths it beats no-hedge on the
    objective E[u(W_T)].
The verdict rolls on the held-out BATCH, which no fit step saw, so a policy that merely overfits
the fitted paths fails this.

JSON-is-the-contract: load_json + run_job only. The inner-MC fork is single-pass at
`Batch_Size x Inner_Sub_Batch`, so there is no partition to cover. A policy trained inside a
`Total_Position_Schedule` is `test_action_space_and_artifact`'s.
"""
import json as jsonlib
import os

import derivus as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')


def _cfg(inner_antithetic='Yes'):
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'], calc['Simulation_Batches'] = 24, 2   # 24 fitted, 24 held out
    calc['Inner_Sub_Batch'] = 8
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = inner_antithetic
    calc['Random_Seed'] = 1234
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64,
        'T_Min': 100,                       # ~17-step bounded sweep (fast); full depth in build notes
        'DiffV2_Fit_Iters': 30,
        # defaults apply: DiffV2_Weight_Decay=0 (the twin-loss gradient match is the regularizer),
        # DiffV2_Lambda_Grad=1, DiffV2_Hidden=32. Multi-seed robustness is validated at
        # B_outer=4095.
    }
    return cfg


def test_diffsolverv2_bounded_and_hedges_oos():
    """The antithetic inner fold (mirrored (z, -z) pairs on the inner axis), seed 1234, read on
    the held-out batch: V_0 0.186 and max|Y_boot| 0.195 against a bound of 1; the April contract
    has expired before the sweep window and carries no position; greedy E[u] 0.100 against
    no-hedge 0.077. The plain Sobol fold reads 0.180, 0.192 and 0.115 - the same gates.

    Killing mutations: the expired-contract guard off (`live` all ones in `_inner_step`) - V_0 and
    max|Y_boot| 10.37, 41.9 contracts parked on the dead April leg, greedy no better than no-hedge;
    the greedy book inverted - E[u] 0.053 against no-hedge 0.077."""
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(_cfg()), 'diffml_v2_oos.json'))
    _, result = cx.run_job()
    diag = result.evaluation_summary['diagnostics']

    v0 = float(diag['V_0'])
    assert abs(v0) < 1.0 and diag['bounded'] is True, f'V_0 not bounded: {v0}'
    assert float(diag['max_abs_Y_boot']) < 1.0, \
        f"max|Y_boot|={diag['max_abs_Y_boot']} — value inflating (expired-dF guard regressed?)"

    assert diag['verdict_is_oos'] is True, 'verdict must be on held-out paths'
    v = diag['verdict']
    assert v['greedy_mean_abs_q'][0] == 0.0, \
        f"the expired April leg carries {v['greedy_mean_abs_q'][0]} contracts — live mask off?"
    g_u, nh_u = v['greedy']['u_mean'], v['nohedge']['u_mean']
    assert g_u > nh_u, f'greedy does not beat no-hedge OOS (u {g_u:.4f} against {nh_u:.4f})'
