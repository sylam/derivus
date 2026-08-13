"""`SpotModel` is a CAPABILITY the deal class declares, and the declaration is enforced.

`Valuation Configuration` carries `SpotModel` per deal TYPE, and `config.py`'s EquityPrice /
FxRate conditionals pull `<SpotModel>ModelParameters.<underlying>` into the factor universe for
any instrument carrying the field - regardless of whether that deal type has a non-GBM pricer.
Before `Deal.spot_models`, only the four types that call `get_spot_model_params_factor` ever
looked at the switch, so `{'EquityOptionDeal': {'SpotModel': 'HestonNandi'}}` loaded the HN
parameters and priced Black, silently: measured MTM 9.94764497 with the switch on, 9.94764497
with it off - bit-identical - against 9.412848 for the same option under the HN closed form with
those parameters (+5.7% delivered under a model the JSON did not ask for, with no log line).

The rule is the symmetric half of the either/or ruling: a deal that DECLARED a model must not be
handed another one. The class declares what it honours (`spot_models`, the base being GBM-only),
and `Deal.__init__` refuses anything else naming the deal type and the model. Because the switch
is keyed by deal TYPE, a bad value is not one bad deal - it mis-declares the model for every deal
of that type in the book - so the refusal is a construction-time raise, not a per-deal skip.

MUTATION KILL MATRIX (source edit, run, revert BY HAND):

    mutant                                                       killed by
    drop the `if spot_model not in self.spot_models` raise       test_unsupported_deal_type_refuses[3 cases],
                                                                 test_unknown_value_refuses_on_a_supported_type
    `Deal.spot_models = ('None', 'HestonNandi')` (base opens up) test_unsupported_deal_type_refuses[3 cases],
                                                                 test_declaration_matches_the_pricers
    drop `spot_models` from EquityBarrierOption                   test_supported_deal_types_accept[EquityBarrierOption],
                                                                 test_declaration_matches_the_pricers
    `spot_models` on EquityOptionDeal (declared, no pricer)       test_declaration_matches_the_pricers
"""
import inspect
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from derivus import instruments
from derivus.instruments import Deal, construct_instrument

BASE = pd.Timestamp('2024-06-28')

# minimal constructible nodes - construction stores `field` and (barrier) reads Payoff_Currency
_SUPPORTED = {
    'EquityBarrierOption': {'Currency': 'USD', 'Payoff_Currency': 'USD'},
    'EquityBarrierBinaryOption': {'Currency': 'USD', 'Payoff_Currency': 'USD'},
    'QEDI_CustomAutoCallSwap': {'Currency': 'USD'},
    'QEDI_CustomAutoCallSwap_V2': {'Currency': 'USD'},   # inherits the declaration
    'FXTARFOptionDeal': {'Currency': 'USD', 'Underlying_Currency': 'AUD'},
}
_UNSUPPORTED = {
    'EquityOptionDeal': {'Currency': 'USD', 'Payoff_Currency': 'USD', 'Equity': 'EQ',
                         'Option_Style': 'European'},
    'FXOptionDeal': {'Currency': 'USD', 'Underlying_Currency': 'AUD'},
    'EquityForwardDeal': {'Currency': 'USD', 'Equity': 'EQ'},
}


def _build(obj, fields, spot_model=None):
    val = {obj: {'SpotModel': spot_model}} if spot_model is not None else {}
    return construct_instrument(dict(fields, Object=obj, Reference='D1'), val)


@pytest.mark.parametrize('obj', sorted(_UNSUPPORTED))
def test_unsupported_deal_type_refuses(obj):
    """The V6 defect: a deal type with no non-GBM pricer must not accept the switch. The refusal
    names the deal type and the model - both are what the user has to change."""
    with pytest.raises(ValueError) as e:
        _build(obj, _UNSUPPORTED[obj], 'HestonNandi')
    assert obj in str(e.value) and 'HestonNandi' in str(e.value)


@pytest.mark.parametrize('obj', sorted(_UNSUPPORTED))
@pytest.mark.parametrize('spot_model', [None, 'None'])
def test_unsupported_deal_type_still_builds_under_gbm(obj, spot_model):
    """The other direction: absent and the explicit 'None' are the GBM contract every deal type
    honours, so neither may be refused. Without this the check could be `raise` unconditionally."""
    assert isinstance(_build(obj, _UNSUPPORTED[obj], spot_model), Deal)


@pytest.mark.parametrize('obj', sorted(_SUPPORTED))
@pytest.mark.parametrize('spot_model', [None, 'None', 'HestonNandi'])
def test_supported_deal_types_accept(obj, spot_model):
    """No false refusal on the five types that DO honour HN (V2 by inheriting the declaration)."""
    assert isinstance(_build(obj, _SUPPORTED[obj], spot_model), Deal)


def test_unknown_value_refuses_on_a_supported_type():
    """A typo'd value on a type that does honour a model is the same refusal, and still carries
    the offending value and the accepted set (it used to be raised in calc_dependencies - see
    test_hn_tarf_pricer.test_unknown_spot_model_fails_loudly)."""
    with pytest.raises(ValueError) as e:
        _build('FXTARFOptionDeal', _SUPPORTED['FXTARFOptionDeal'], 'Heston_Nandi')
    assert 'Heston_Nandi' in str(e.value) and "('None', 'HestonNandi')" in str(e.value)


def test_declaration_matches_the_pricers():
    """The declaration is the only place the capability is written down, so it must stay in step
    with the code that reads a SpotModel: a class declares `spot_models` of its own IFF its own
    `calc_dependencies` resolves the parameters. Catches both drifts - a new deal type that wires
    up the pricer without declaring (it would refuse a model it can price), and a declaration
    with no pricer behind it (the V6 defect, re-introduced through the front door)."""
    declares, resolves = set(), set()
    for name, cls in vars(instruments).items():
        if not (inspect.isclass(cls) and issubclass(cls, Deal) and cls is not Deal
                and cls.__module__ == instruments.__name__):
            continue
        if 'spot_models' in cls.__dict__:
            declares.add(name)
        dep = cls.__dict__.get('calc_dependencies')
        if dep is not None and 'get_spot_model_params_factor' in inspect.getsource(dep):
            resolves.add(name)
    assert declares == resolves, 'spot_models declarations {0} vs pricers {1}'.format(
        sorted(declares), sorted(resolves))
    assert declares == set(_SUPPORTED) - {'QEDI_CustomAutoCallSwap_V2'}
    assert Deal.spot_models == ('None',), 'the base must stay GBM-only or every type opts in'
