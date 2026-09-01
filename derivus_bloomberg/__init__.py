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
"""The Bloomberg adapters, re-exported - the pandas-carrying half of them LAZILY.

`equity_chain` is eager and costs only the standard library plus this package's `errors`. Every
name whose module reaches pandas resolves on FIRST ACCESS (PEP 562) instead, so importing the
chain emitter does not land numpy and pandas on a caller that only wanted a listed chain. Both
`from derivus_bloomberg import <name>` and attribute access work: the import is deferred, not
removed.
"""
import importlib

from .equity_chain import (ChainContract, EquityChain, EquityForward, EquityLadder,
                           equity_hn_block, fetch_equity_chain, screen_chain, select_rungs)

#: `{exported name: the submodule that owns it}` - the lazy half of `__all__`, read by
#: `__getattr__` and `__dir__`. The rates emitters belong here too: `ir_curve` and `swaption_vol`
#: reach `discover` -> `security_map` -> `types`, so an eager re-export would land pandas.
_LAZY = {
    'fetch_fx_vol': '.fxvol', 'install_fx_vol_snapshot': '.fxvol', 'normalize_fx_vol': '.fxvol',
    'to_market_prices_block': '.fxvol', 'update_fx_vol_snapshot': '.fxvol',
    'FXQuoteSecurity': '.types', 'FXVolDefinition': '.types', 'FXVolPoint': '.types',
    'FXVolSnapshot': '.types', 'RawBloombergObservation': '.types',
    'CurveConventions': '.ir_curve', 'CurveScreen': '.ir_curve', 'CurveStrip': '.ir_curve',
    'RatePrint': '.ir_curve', 'curve_conventions': '.ir_curve',
    'fetch_curve_strip': '.ir_curve', 'ir_curve_block': '.ir_curve', 'screen_strip': '.ir_curve',
    # One `reauthor`, owned by `ir_curve` and reached by `swaption_vol`.
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
    """Resolve a lazily re-exported name, caching it in module globals so later accesses are
    ordinary attribute lookups and a patch applied to one stays patched."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
