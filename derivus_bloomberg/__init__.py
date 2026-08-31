########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of Derivus.
#
# Derivus is free for noncommercial use under the terms of the PolyForm
# Noncommercial License 1.0.0. You should have received a copy of the license
# along with Derivus. If not, see
# <https://polyformproject.org/licenses/noncommercial/1.0.0>.
#
# Derivus is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
########################################################################
"""The Bloomberg adapters, re-exported - and the pandas half of them re-exported LAZILY.

`equity_chain` is imported eagerly because it costs the standard library and this package's own
`errors`, and that is a claim its own docstring makes about `import derivus_bloomberg.equity_chain`.
An eager `from .fxvol import ...` here would falsify it from outside the module: `fxvol` and `types`
carry pandas, importing a submodule imports its package first, and the chain emitter would land
numpy and pandas on a workstation that only wanted to read a listed chain. So the pandas-carrying
names resolve on FIRST ACCESS (PEP 562), which leaves `from derivus_bloomberg import fetch_fx_vol`
and `derivus_bloomberg.fetch_fx_vol` working exactly as before - the import is deferred, not
removed, and the FX adapters pay for their own dependency at the moment somebody asks for one.
"""
import importlib

from .equity_chain import (ChainContract, EquityChain, EquityForward, EquityLadder,
                           equity_hn_block, fetch_equity_chain, screen_chain, select_rungs)

#: `{exported name: the submodule that owns it}` for the names whose modules carry pandas. Read by
#: `__getattr__` below and by `__dir__`, so this mapping IS the lazy half of `__all__`.
#:
#: THE TWO RATES EMITTERS ARE HERE RATHER THAN ABOVE, and not because they carry pandas themselves:
#: `ir_curve` and `swaption_vol` reach `discover` for its ticker grammar, `discover` reaches
#: `security_map`, and that reaches `types`. So an EAGER re-export of either would land pandas on
#: `import derivus_bloomberg.equity_chain` behind that module's back - which is precisely the claim
#: its own gate measures in a fresh interpreter. Deferring them keeps the chain emitter's budget
#: what it says it is and costs a caller who wants a curve nothing it was not already paying.
_LAZY = {
    'fetch_fx_vol': '.fxvol', 'install_fx_vol_snapshot': '.fxvol', 'normalize_fx_vol': '.fxvol',
    'to_market_prices_block': '.fxvol', 'update_fx_vol_snapshot': '.fxvol',
    'FXQuoteSecurity': '.types', 'FXVolDefinition': '.types', 'FXVolPoint': '.types',
    'FXVolSnapshot': '.types', 'RawBloombergObservation': '.types',
    'CurveConventions': '.ir_curve', 'CurveScreen': '.ir_curve', 'CurveStrip': '.ir_curve',
    'RatePrint': '.ir_curve', 'curve_conventions': '.ir_curve',
    'fetch_curve_strip': '.ir_curve', 'ir_curve_block': '.ir_curve', 'screen_strip': '.ir_curve',
    # ONE `reauthor`, owned by `ir_curve` and reached by `swaption_vol` - the drop-and-re-install
    # both emitters need, for the two reasons its docstring states apart. It was documented as "the
    # only way a re-quoted ladder reaches a book" and then left out of this mapping, so
    # `from derivus_bloomberg import reauthor` raised AttributeError against its own documentation.
    'reauthor': '.ir_curve',
    'SwaptionConventions': '.swaption_vol', 'SwaptionLadder': '.swaption_vol',
    'SwaptionQuote': '.swaption_vol', 'SwaptionScreen': '.swaption_vol',
    'fetch_swaption_ladder': '.swaption_vol', 'hw2f_block': '.swaption_vol',
    'screen_ladder': '.swaption_vol', 'swaption_conventions': '.swaption_vol',
}

__all__ = [
    'ChainContract', 'CurveConventions', 'CurveScreen', 'CurveStrip', 'EquityChain',
    'EquityForward', 'EquityLadder', 'FXQuoteSecurity', 'FXVolDefinition', 'FXVolPoint',
    'FXVolSnapshot', 'RatePrint', 'RawBloombergObservation', 'SwaptionConventions',
    'SwaptionLadder', 'SwaptionQuote', 'SwaptionScreen', 'curve_conventions', 'equity_hn_block',
    'fetch_curve_strip', 'fetch_equity_chain', 'fetch_fx_vol', 'fetch_swaption_ladder',
    'hw2f_block', 'install_fx_vol_snapshot', 'ir_curve_block', 'normalize_fx_vol', 'reauthor',
    'screen_chain', 'screen_ladder', 'screen_strip', 'select_rungs', 'swaption_conventions',
    'to_market_prices_block', 'update_fx_vol_snapshot',
]


def __getattr__(name):
    """The deferred half of the re-export - imported once, then cached in the module's own globals
    so a second access is an ordinary attribute lookup and a patch to it stays patched."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
