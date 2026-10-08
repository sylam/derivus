"""End-to-end test for the interactive `BundleStepper`: a no-trade rollout driven from client code
through `result.create_stepper()` lands on the framework's own no-trade baseline in
`evaluation_summary.reference.no_trade.metrics`.

JSON in, run_job, drive the stepper. No internal imports.
"""
import json as jsonlib
import os

import torch

import derivus as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures',
                       'policy_test_simulate_only.json')


def test_no_trade_stepper_matches_framework_baseline():
    """The terminal P&L of a zero-trade client rollout - its mean and its worst path - is the
    reported no-trade baseline to a dollar.

    Killing mutation: the baseline's `worst_net_pnl` read off the best path."""
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(jsonlib.load(open(FIXTURE))), 'stepper_test.json'))
    _, result = cx.run_job()
    stepper = result.create_stepper()
    last = None
    while not stepper.done:
        last = stepper.step(None)  # zero trades every step
    pnl = (last['transition_pnl_excess'] + last['transition_liability_value']).to(dtype=torch.float64)
    stepper_mean = float(pnl.mean().item())
    stepper_worst = float(pnl.min().item())

    baseline = result.evaluation_summary['reference']['no_trade']['metrics']
    fw_mean = float(baseline['average_net_pnl'])
    fw_worst = float(baseline['worst_net_pnl'])

    assert abs(stepper_mean - fw_mean) < 1.0, f"stepper mean ${stepper_mean:,.0f} ≠ framework ${fw_mean:,.0f}"
    assert abs(stepper_worst - fw_worst) < 1.0, f"stepper worst ${stepper_worst:,.0f} ≠ framework ${fw_worst:,.0f}"
