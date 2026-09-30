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

"""The collateral call at one date - the engine's own recursion, read on a close.

A CSA asks for the credit support its dials make of the exposure: the independent amount, plus the
exposure's excess over the received threshold where it is above it, plus its excess below the
posted threshold where it is below it - the engine's `At`, thresholds and independent amount stated
in the agreement currency and crossed from it. The CALL is that less the balance held, each asset
valued after its `Haircut_Posted`, the one haircut the engine reads, and it moves only where it
clears the minimum transfer on its side STRICTLY, as `scan_collateral_balance` transfers: the bank
calls where it is short by more than `Minimum_Received`, and posts where it holds more than
`Minimum_Posted` over what is required. Nothing is rounded, the netting set declaring no rounding.
COLLATERAL AND MARGIN ARE TWO BALANCES: the call reads the collateral alone.

Pure functions over plain data, so the service and the oracle run one spelling.
"""
from .verbs import HELD

#: The two sides of a call, from the bank's: it calls collateral in, or posts it out.
CALL, POST = 'call', 'post'
#: What `call` answers - the fields a call nobody can work out carries as nulls.
WORKED = ('required', 'balance', 'call', 'direction', 'minimum_transfer')
#: The dials a CSA states under `Credit_Support_Amounts`, each a `CreditSupportList`.
DIALS = ('Independent_Amount', 'Received_Threshold', 'Posted_Threshold', 'Minimum_Received',
         'Minimum_Posted')
#: What an agreement nothing moved under holds: two empty balances.
NOTHING = dict.fromkeys(HELD, {})


def csa(terms):
    """The CSA a netting set's `terms` declare, as the engine reads them - or None where they
    collateralise nothing, the engine then holding no collateral at all.

    `{Agreement_Currency, Balance_Currency, <each of DIALS>, haircuts}`: every `CreditSupportList`
    at its first value, None where the terms state none - the independent amount excepted, which
    the engine reads as nothing - the balance currency the agreement's where unstated, and
    `haircuts` `{currency: Haircut_Posted}` off the cash rows of `Collateral_Assets`, a currency
    being the one asset a close's spots value.
    """
    if terms.get('Collateralized', 'False') != 'True':
        return None
    support = terms.get('Credit_Support_Amounts') or {}
    dials = {dial: _first(support.get(dial)) for dial in DIALS}
    rows = (terms.get('Collateral_Assets') or {}).get('Cash_Collateral') or []
    return dict(dials, Independent_Amount=dials['Independent_Amount'] or 0.0,
                Agreement_Currency=terms.get('Agreement_Currency'),
                Balance_Currency=terms.get('Balance_Currency') or terms.get('Agreement_Currency'),
                haircuts={row['Currency']: _percent(row.get('Haircut_Posted')) for row in rows})


def held(movements, until=None):
    """`{agreement: {kind: {asset: amount}}}` - the collateral and the margin held under every
    agreement, kept apart, in one pass over the `cash` fold's standing movements: each held kind
    summed per asset, as of the value date `until` where one is named. Signed from the bank's side,
    so what it posted is negative; an agreement nothing moved under holds `NOTHING`."""
    balances = {}
    for row in movements:
        if row['kind'] in HELD and (until is None or row['effective_time'][:10] <= until):
            kinds = balances.setdefault(row['subject'], {kind: {} for kind in HELD})
            kinds[row['kind']][row['asset']] = (kinds[row['kind']].get(row['asset'], 0.0)
                                                + row['amount'])
    return balances


def valued(holding, csa, fx):
    """What `holding` (`{asset: amount}`) is worth to the CSA in the currency `fx` values every
    asset in: each amount crossed, less its `Haircut_Posted` on whichever side it is held - the
    engine reads no other - and an asset held at nothing worth nothing, crossed or not."""
    return sum((amount * fx[asset] * (1.0 - csa['haircuts'].get(asset, 0.0))
                for asset, amount in sorted(holding.items()) if amount), 0.0)


def required(exposure, csa, fx):
    """The credit support the CSA asks for at `exposure`, in the currency `exposure` is in and
    `fx` values every currency in - the engine's `At`, term for term: positive is support the bank
    should hold, negative support it should have posted."""
    agreement = fx[csa['Agreement_Currency']]
    above, below = csa['Received_Threshold'] * agreement, csa['Posted_Threshold'] * agreement
    return (csa['Independent_Amount'] * agreement + (exposure - above) * (exposure > above)
            + (exposure - below) * (exposure < below))


def call(exposure, holding, csa, fx):
    """The call at `exposure` with the collateral `holding` held: `{required, balance, call,
    direction, minimum_transfer}` in the currency `fx` values in. `call` is the signed amount the
    settlement moves - received positive - and nothing where the difference clears neither
    minimum, `direction` saying which the bank does, and `minimum_transfer` the minimum on the side
    the difference falls."""
    needed, balance = required(exposure, csa, fx), valued(holding, csa, fx)
    agreement, short = fx[csa['Agreement_Currency']], needed - balance
    received = csa['Minimum_Received'] * agreement
    posted = csa['Minimum_Posted'] * agreement
    direction = CALL if short > received else POST if -short > posted else None
    return {'required': needed, 'balance': balance, 'call': short if direction else 0.0,
            'direction': direction, 'minimum_transfer': received if short >= 0 else posted}


def _first(table):
    """A `CreditSupportList` at its first value - a dict over its `[rating, amount]` rows, as the
    engine reads one - or None where it states none."""
    rows = table.get('.CreditSupportList') if isinstance(table, dict) else None
    return float(next(iter(dict(rows).values()))) if rows else None


def _percent(value):
    """A haircut as the fraction the engine reads its `Percent` as, nothing where unstated."""
    return value['.Percent'] / 100.0 if isinstance(value, dict) else float(value or 0.0)
