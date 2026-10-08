"""The utility scale `c` a symlog objective reads, through `cx.run_job()` on a JSON fixture: an
`Objective.Utility_Scale_Explicit` is honoured exactly, and a misspelt `Utility_Scale_Mode` is
refused by name rather than read as the default formula. The formula itself is
`test_spot_history_optional`'s.

NO monkey-patching. NO internal imports. Just JSON in, result out.
"""
import json as jsonlib
import os

import pytest

import derivus as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'policy_test_simulate_only.json')


def _load(**objective_overrides):
    """Load the base fixture and apply Objective overrides."""
    data = jsonlib.load(open(FIXTURE))
    obj = data['Calc']['Calculation']['Hedging_Problem']['Objective']
    obj.update(objective_overrides)
    return data


def _run(data):
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(data), 'smoke.json'))
    _, result = cx.run_job()
    return result


def test_unknown_utility_scale_mode_fails_loud():
    """A typo in Utility_Scale_Mode raises at bundle build, naming the spelling.

    Killing mutation: the mode check dropped, the typo read as the default formula."""
    data = _load(Object='AsymmetricUtility_Symlog', Floor_Penalty=10.0,
                 Utility_Scale_Mode='vol_scled_notional')  # typo
    with pytest.raises(ValueError, match='vol_scled_notional'):
        _run(data)


def test_explicit_utility_scale_override():
    """Utility_Scale_Explicit overrides the formula, exactly.

    Killing mutation: the explicit branch skipped, the formula's c reported."""
    data = _load(Object='AsymmetricUtility_Symlog', Floor_Penalty=10.0,
                 Utility_Scale_Explicit=5_000_000.0)
    assert float(_run(data).bundle.utility_scale) == 5_000_000.0
