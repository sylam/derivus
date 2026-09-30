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

"""The collateral calls on a marked close - what each agreement's CSA asks of the exposure under
it, less what is held under it.

THE EXPOSURE IS THE ONE THE NETTING SET'S OWN RECURSION READS, off the P&L's numbers: the value
`pnl.pnl` gives each position under the agreement at the close - its quantity times its unit mark
less what the unit paid that day, a position held through its last day as the set holds it until
its cash moves - with that day's payment added back where the set holds it (the engine's
`Exclude_Paid_Today` off, its default), crossed into the agreement currency at the close's own
spots; a mark nobody has is named rather than read as zero. THE ARITHMETIC IS THE RECORD'S:
`derivus_spine.collateral`, one formula the service and the oracle run and the engine's own
recursion at one date.

Pure over plain data: the P&L over the close alone, the agreements declared, the `cash` rows.
"""
from .pnl import _times
from .spine import package, visible

#: What a call nobody can work out answers in place of its arithmetic.
UNWORKED = dict.fromkeys(('required', 'balance', 'call', 'direction', 'minimum_transfer'))


def paid_today(document):
    """Whether the netting sets of `document` hold the day's payments in the exposure their
    recursion reads: `Exclude_Paid_Today` off - the engine's default - as a set reads it off the
    job's `Valuation Configuration`."""
    options = (document['Calc']['MergeMarketData'].get('ExplicitMarketData') or {}).get(
        'Valuation Configuration') or {}
    return not options.get('NettingCollateralSet', {}).get('Exclude_Paid_Today', False)


def exposure(agreement, valued, rates, currency, today):
    """What the positions under `agreement` are worth at a close in `currency`: the value the P&L
    gives each there (`valued`, `pnl.pnl`'s answer over that close alone) and, where the set holds
    the day's payments (`today`), what each paid that day, crossed at the close's `rates`
    (`pnl.rates`) - None where one of them is not known or the close carries no spot for it."""
    rows = [row for row in valued['rows'] if row['agreement'] == agreement]
    values = [row['value_end'] for row in rows] + [
        _times(row['quantity_end'], row['paid_end']) for row in rows if today]
    return None if None in values or currency not in rates else sum(values, 0.0) / rates[currency]


def collateralised(agreements):
    """The declared `agreements` whose terms collateralise - the ones a close has calls on."""
    return [row for row in agreements if package().collateral.csa(row['terms']) is not None]


def seen(agreements, positions, sight, verb=None):
    """The `agreements` a seat whose `sight` this is reads: those a position standing among
    `positions` sits under at a node it holds `verb` at - any verb where none is named - and every
    one where nothing narrows the seat, one nothing is held under included (`spine.visible`)."""
    held = {row['agreement'] for row in visible(positions, sight, verb=verb) if row['quantity']}
    # a row sitting nowhere is seen only where nothing narrows the seat
    everywhere = bool(visible([{}], sight, verb=verb))
    return [row for row in agreements if everywhere or row['agreement'] in held]


def calls(close, valued, agreements, movements, until):
    """`[{agreement, entity, currency, exposure, held, margin, required, balance, call, direction,
    minimum_transfer, unknown}]` - the call of each of the `collateralised` agreements on the marks
    `close` (`{day, rates, document}`), in the agreement's currency: `valued` is the P&L over that
    close alone, `movements` the `cash` rows and `until` the value date a balance is held as of -
    every movement filed, whatever its value date, where None.

    `held` and `margin` are the two balances under it per asset, and the call reads the first
    alone. What nobody can know - a mark, a spot the close carries none of, a dial the terms state
    none of - is named under `unknown`, and the call is null over it.
    """
    record, rates, answer = package().collateral, close['rates'], []
    balances, today = record.held(movements, until), paid_today(close['document'])
    for declared in agreements:
        dials = record.csa(declared['terms'])
        agreement, currency = declared['agreement'], dials['Agreement_Currency']
        held = balances.get(agreement, record.NOTHING)
        fx = {} if currency not in rates else {
            other: rate / rates[currency] for other, rate in rates.items()}
        mine = {row['instrument'] for row in valued['rows'] if row['agreement'] == agreement}
        # an asset held at nothing needs no spot to be worth nothing, nor a currency never named
        uncrossed = {currency} - set(fx) - {None, ''} | {
            asset for asset, amount in held['collateral'].items() if amount and asset not in fx}
        unknown = [entry for entry in valued['unknown'] if entry['instrument'] in mine] + [
            {'instrument': None, 'what': 'the close carries no spot for {}'.format(other)}
            for other in sorted(uncrossed)] + [
            {'instrument': None, 'what': 'the terms of {} state no {}'.format(agreement, dial)}
            for dial in record.DIALS + ('Agreement_Currency',) if dials[dial] in (None, '')]
        value = exposure(agreement, valued, rates, currency, today)
        answer.append(dict(
            UNWORKED if unknown else record.call(value, held['collateral'], dials, fx),
            agreement=agreement, entity=declared['entity'], currency=currency, exposure=value,
            held=held['collateral'], margin=held['margin'], unknown=unknown))
    return answer
