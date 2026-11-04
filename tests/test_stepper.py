"""End-to-end test for the interactive `BundleStepper`: a no-trade rollout driven from client code
through `result.create_stepper()` lands on the framework's own no-trade baseline in
`evaluation_summary.metrics`.

JSON in, run_job, drive the stepper. No internal imports.
"""
import json as jsonlib
import os

import pandas as pd
import torch

import derivus as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures',
                       'policy_test_simulate_only.json')


def test_no_trade_stepper_matches_framework_baseline(tmp_path):
    """The terminal P&L of a zero-trade client rollout - its mean and its worst path - is the
    reported no-trade baseline to the bit, both read off the money the stepper carries, and the
    stepper's summary CSV prints them to the cent.

    Killing mutations: the baseline's `worst_net_pnl` read off the best path; the baseline read off
    a float32 cast of the net (its mean 118,250.78 against the CSV's 118,250.77)."""
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(jsonlib.load(open(FIXTURE))), 'stepper_test.json'))
    _, result = cx.run_job()
    stepper = result.create_stepper()
    last = None
    while not stepper.done:
        last = stepper.step(None)  # zero trades every step
    pnl = last['transition_net_pnl'].to(dtype=torch.float64)
    stepper_mean = float(pnl.mean().item())
    stepper_worst = float(pnl.min().item())

    baseline = result.evaluation_summary['metrics']
    fw_mean = float(baseline['average_net_pnl'])
    fw_worst = float(baseline['worst_net_pnl'])

    assert (stepper_mean, stepper_worst) == (fw_mean, fw_worst), (
        stepper_mean, fw_mean, stepper_worst, fw_worst)
    stepper.write_diagnostic_csvs(str(tmp_path), 'nt')
    summary = pd.read_csv(tmp_path / 'nt_summary.csv').set_index('metric')['net_pnl']
    assert (summary['mean'], summary['min']) == (round(fw_mean, 2), round(fw_worst, 2)), summary
