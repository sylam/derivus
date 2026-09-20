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

"""The diary - every payment, fixing and expiry the book's own compile announces, as JSON rows.

THE COMPILE'S SCHEDULE RE-EMITTED. The deals are constructed and their schedules bound exactly as a
valuation binds them and then READ rather than priced, reached by `utils.walk_schedules` - the one
walk the binding itself takes - so the diary and the pricer cannot disagree about a payment: they
are the same schedule. Nothing here learns about users, workflow or storage; a row is plain JSON,
the derived key is asked of the one seam that knows the record exists, and a box without the record
installed carries `key: null` and everything else.

A NULL AMOUNT IS NEVER WRITTEN AS 0.0. A floating leg's amount is not determined until its resets
fix, and a settlement file carrying 0.0 for one is an instruction to pay nothing - so the row says
`determined: false`, and `export_settlements` refuses it by name rather than exporting a zero.
"""
from collections import namedtuple

import numpy as np
import pandas as pd

from .spine import SpineRefused, cashflow_key
from .pricing import fixed_payments
from .instruments import Deal, barrier_monitoring_rows
from .utils import (CASHFLOW_INDEX_FixedAmt, CASHFLOW_INDEX_FixedRate, CASHFLOW_INDEX_Nominal,
                    CASHFLOW_INDEX_Pay_Day, CASHFLOW_INDEX_Year_Frac, RESET_INDEX_Reset_Day,
                    RESET_INDEX_Value, TensorCashFlows, TensorResets, check_rate_name,
                    walk_schedules)

#: What a row announces. A `barrier` row is a monitoring date rather than a due thing: what the
#: level did is read off the fixing that satisfies it, and a close never waits on one.
PAYMENT, FIXING, EXPIRY, BARRIER = 'payment', 'fixing', 'expiry', 'barrier'

#: What a deal the compile could not read announces. Not a due thing - a reading of the book that
#: says the book could not be read - and a close is never legal while one stands.
UNREADABLE = 'unreadable'

#: Where a row stands. The compile answers these three; `settled` is the record's own answer and is
#: stamped by whoever holds the log.
DUE, OBSERVED, EXPIRED = 'due', 'observed', 'expired'
SETTLED = 'settled'

#: The legs that are not a schedule: the deal's own settlement ladder, and its last day.
SETTLEMENT, EXPIRY_LEG = 'Settlement', 'Expiry'

#: A physically settled option DELIVERS, and delivering is an act somebody takes - so its expiry
#: vests a choice and the diary says so.
PHYSICAL = 'Physical'
ELECTION = 'election'

#: What a deal type declares to the record. `table` is the deal's own observation table and
#: `column` the cell a print is written into; `index` and `family` name the price factor an
#: observation is OF; `elects` is the field whose `Physical` vests a choice at expiry; `expires`
#: is the day the terms are fixed on; and `pays`/`amount` are the fields a deal with NO schedule
#: of its own settles by - the date falling back to the expiry where the type names no other.
Terms = namedtuple('Terms', 'table column index family elects expires pays amount',
                   defaults=(None,) * 8)

TERMS = {
    'EquityBarrierOption': Terms('Barrier_Dates', 1, 'Equity', 'EquityPrice',
                                 expires='Expiry_Date'),
    'EquityBarrierBinaryOption': Terms('Barrier_Dates', 1, 'Equity', 'EquityPrice',
                                       expires='Expiry_Date', pays='Settlement_Date'),
    'QEDI_CustomAutoCallSwap': Terms('Price_Fixing', 1, 'Equity', 'EquityPrice'),
    'QEDI_CustomAutoCallSwap_V2': Terms('Price_Fixing', 1, 'Equity', 'EquityPrice'),
    'EquityOptionDeal': Terms(index='Equity', family='EquityPrice', elects='Settlement_Style',
                              expires='Expiry_Date'),
    'FXOptionDeal': Terms(index='Underlying_Currency', family='FxRate',
                          elects='Settlement_Style', expires='Expiry_Date',
                          pays='Delivery_Date'),
    'FXBinaryOption': Terms(index='Underlying_Currency', family='FxRate',
                            expires='Expiry_Date', pays='Delivery_Date'),
    'SwapInterestDeal': Terms(index='Interest_Rate', family='InterestRate'),
    'CFFloatingInterestListDeal': Terms(index='Forecast_Rate', family='InterestRate'),
    'FixedCashflowDeal': Terms(pays='Payment_Date', amount='Amount'),
}


def index_named(fields, terms):
    """The price factor an observation of this deal is OF, spelled as the engine spells a factor
    name - `EquityPrice.SPX`, `InterestRate.ZAR` - or None where the type names no index."""
    named = fields.get(terms.index) if terms is not None and terms.index else None
    return '{}.{}'.format(terms.family, '.'.join(check_rate_name(named))) if named else None


def schedule_of(context, instruments=None):
    """Every payment, fixing and expiry this job's deals carry, as JSON rows - the COMPILE's own
    schedule, re-emitted.

    `instruments` maps a deal's `Reference` to the instrument address the record books it under,
    which is what gives a row its key; a reference it does not name carries `instrument: null` and
    `key: null`, so the diary reads on a book nothing was ever booked from.

    A deal the compile LEFT OUT is read off the loaded tree instead: expired, and it announces what
    a live one of its type announces, or unreadable, and it announces why. The compiled tree is
    matched to the loaded one by reference, so two deals sharing one - which the book's own
    positional path is the answer to - are read as the live one.
    """
    calc, base_date = _compiled(context)
    named = instruments or {}
    rows, live = [], set()
    for deal_data in calc.netting_sets.deals():
        live.add(deal_data.Instrument.field.get('Reference'))
        rows.extend(_deal_rows(deal_data.Instrument, deal_data.Factor_dep, base_date, named))
    for path, deal in _leaves(calc.config.deals['Deals']['Children']):
        if deal.field.get('Reference') in live:
            continue
        rows.extend(_dropped_rows(deal, named) if _expired(calc, base_date, deal)
                    else [_unreadable_row(calc, base_date, path, deal, named)])
    return sorted(rows, key=lambda row: (row['due_date'] or '', row['kind'], row['leg'],
                                         row['schedule_index'], row['instrument'] or ''))


def export_settlements(rows, official_values_hash, due_before):
    """The settlement file for ONE day: every payment due on or before `due_before` that is not
    already settled, against one market.

    Takes the diary, the hash of the market it is struck on and the day it settles, and nothing
    else - no book, no service, no executor. `due_before` has no default: a settlement file is
    struck FOR a date, and one that exported every future payment would instruct the whole book.
    A row whose amount is UNDETERMINED refuses by name rather than exporting a zero: a floating
    amount is not known until its resets fix, and instructing a payment of 0.0 for one is a wrong
    payment rather than a missing one.
    """
    payments = [row for row in rows if row['kind'] == PAYMENT and row['state'] != SETTLED
                and row['due_date'] and row['due_date'] <= due_before]
    undetermined = [row for row in payments if not row['determined']]
    if undetermined:
        raise ValueError(
            'these {} rows are due and their amount is not determined, so nothing is exported for '
            'them: {}. A floating amount fixes when its resets do - file the observations and '
            'export again, or export the determined rows by asking for them'.format(
                len(undetermined), ', '.join(
                    '{} {} {}[{}] on {}'.format(row['key'], row['instrument'], row['leg'],
                                                row['schedule_index'], row['due_date'])
                    for row in undetermined[:8])))
    unnamed = [row for row in payments if not row['currency']]
    if unnamed:
        raise ValueError(
            'these {} rows name no currency, so nothing is exported for them: {}. A settlement '
            'file totals BY currency, and a total under none is money going nowhere - name the '
            "deal's own Currency and export again".format(
                len(unnamed), ', '.join('{} {}[{}]'.format(row['key'], row['leg'],
                                                           row['schedule_index'])
                                        for row in unnamed[:8])))
    totals = {}
    for row in payments:
        totals[row['currency']] = totals.get(row['currency'], 0.0) + row['amount']
    return {'market': official_values_hash, 'due_before': due_before, 'totals': totals,
            'rows': [{field: row[field] for field in
                      ('key', 'instrument', 'leg', 'schedule_index', 'due_date', 'currency',
                       'amount')} for row in payments]}


# ------------------------------------------------------------------------------------------------
# The compile, and the rows read off it.

def _compiled(context):
    """The job's deals COMPILED and nothing priced: the calculation with its market built and every
    schedule bound, and the date it was built on.

    The compile half of a base valuation, stopped before the structure resolves. The precision and
    the device are the defaults rather than a valuation's: the diary reads the numpy half of a
    schedule, which is authoritative until the bind and unmoved by it.
    """
    from .calculation import Aggregation, DealStructure, construct_calculation
    from .schema import declared_defaults

    config = context.current_cfg
    declared = config.deals['Calculation']
    calc = construct_calculation('Base_Revaluation', config)
    params = declared_defaults(type(calc), dict(
        declared, Run_Date=declared['Base_Date'].strftime('%Y-%m-%d')))
    base_date = pd.Timestamp(params['Run_Date'])
    calc.params = params
    shared = calc.update_factors(params, base_date)
    calc.netting_sets = DealStructure(Aggregation('root'), store_results=True)
    calc.set_deal_structures(config.deals['Deals']['Children'], calc.netting_sets,
                             shared.one, deal_level_mtm=True)
    return calc, base_date


def _leaves(children, path=()):
    """Every LEAF deal of a compiled config's tree, as `(deal path, deal)`. A container is its
    structure and carries no schedule of its own."""
    for position, node in enumerate(children):
        if node.get('Ignore') == 'True' or not isinstance(node.get('Instrument'), Deal):
            continue
        if node.get('Children'):
            yield from _leaves(node['Children'], path + (position,))
        else:
            yield '/'.join(map(str, path + (position,))), node['Instrument']


def _expired(calc, base_date, deal):
    """Whether the compile left this deal out because it has EXPIRED - the engine's own expiry
    answer, and the only one used."""
    from .calculation import DealStructure

    return DealStructure.calc_time_dependency(base_date, deal, calc.time_grid) is None


def _dropped_rows(deal, instruments):
    """What a deal the compile dropped as EXPIRED still announces: its expiry, and the fixing and
    the settlement its type declares.

    An option whose expiry is already behind the base date is exactly the deal a catch-up rule
    exists for - one that fell due and nobody cleared - and the two declared rows are field reads
    that need no schedule, so the expired branch says what the live one says.
    """
    fields = deal.field
    terms = TERMS.get(fields.get('Object'))
    instrument = instruments.get(fields.get('Reference'))
    rows = [_expiry_row(deal, instruments, EXPIRED)]
    rows.extend(_expiry_fixing(fields, terms, index_named(fields, terms), instrument, rows))
    rows.extend(_declared_payment(fields, terms, deal.get_settlement_currencies(), instrument))
    return rows


def _unreadable_row(calc, base_date, path, deal, instruments):
    """The one row a deal the compile COULD NOT READ leaves: where it sits and what the engine said.

    A deal skipped for a price factor the market data has no block for has no schedule, no expiry
    and, without this, no trace at all - and a close check answering `legal` on a book it could not
    read is a clean bill nobody earned. The dependency build is asked once more, for this deal
    alone, so the row carries the engine's own sentence rather than a paraphrase of it.
    """
    try:
        deal.calc_dependencies(base_date, calc.static_factors, calc.stoch_factors,
                               calc.all_factors, calc.all_tenors, calc.time_grid,
                               calc.config.holidays)
        said = 'the compile left this deal out and reading it again raised nothing'
    except Exception as error:
        said = '{}'.format(error)
    return _row(UNREADABLE, instruments.get(deal.field.get('Reference')), path, 0, None,
                state=UNREADABLE, reason='{} ({}): {}'.format(
                    deal.field.get('Reference'), deal.field.get('Object'), said))


def _deal_rows(deal, compiled, base_date, instruments):
    """One deal's whole diary: its schedules, its monitoring table, the payment a deal with no
    schedule declares, and its expiry."""
    fields = deal.field
    terms = TERMS.get(fields.get('Object'))
    instrument = instruments.get(fields.get('Reference'))
    index = index_named(fields, terms)
    settlement = deal.get_settlement_currencies()
    rows = []
    for leg, schedule in walk_schedules(compiled):
        currency = _currency(fields, settlement)
        if isinstance(schedule, TensorCashFlows):
            rows.extend(_payment_rows(schedule, _compiled_at(compiled, leg), leg, base_date,
                                      currency, instrument))
            if schedule.Resets is not None:
                rows.extend(_fixing_rows(schedule.Resets, leg + '.Resets', base_date, index,
                                         instrument))
        elif isinstance(schedule, TensorResets):
            rows.extend(_fixing_rows(schedule, leg, base_date, index, instrument))
    rows.extend(_barrier_rows(fields, terms, index, instrument))
    rows.extend(_expiry_fixing(fields, terms, index, instrument, rows))
    if not any(row['kind'] == PAYMENT for row in rows):
        rows.extend(_declared_payment(fields, terms, settlement, instrument))
    rows.append(_expiry_row(deal, instruments, DUE))
    return rows


def _payment_rows(schedule, compiled, leg, base_date, currency, instrument):
    """A leg's PAYMENTS - one per pay day, not one per schedule row.

    The amount is `pricing.fixed_payments`', the line the pricer discounts: the rate coupon and the
    fixed amount of every row sharing that day, summed, compounded where the leg's terms compound.
    A schedule carrying RESETS is not determined and says so - its amount is a floating pricer's
    and the diary does not spell a second one.
    """
    determined = schedule.Resets is None
    rows = schedule.schedule
    days, index, counts = np.unique(rows[:, CASHFLOW_INDEX_Pay_Day], return_index=True,
                                    return_counts=True)
    amounts = fixed_payments(
        rows[:, CASHFLOW_INDEX_FixedRate] * rows[:, CASHFLOW_INDEX_Year_Frac],
        rows[:, CASHFLOW_INDEX_Nominal], rows[:, CASHFLOW_INDEX_FixedAmt], index, counts,
        bool(compiled.get('Compounding', False)))
    return [_row(PAYMENT, instrument, leg, position,
                 base_date + pd.Timedelta(days=int(day)), currency=currency,
                 notional=float(rows[index[position], CASHFLOW_INDEX_Nominal]),
                 amount=float(amounts[position]) if determined else None, determined=determined)
            for position, day in enumerate(days)]


def _compiled_at(compiled, leg):
    """The compiled block a leg's schedule sits in, which is where its `Compounding` flag is."""
    for step in leg.split('.')[:-1]:
        compiled = compiled[step] if isinstance(compiled, dict) else compiled[int(step)]
    return compiled if isinstance(compiled, dict) else {}


def _fixing_rows(resets, leg, base_date, index, instrument):
    """A reset schedule's observations, each one already fixed where its value is on the row."""
    rows = []
    for position, row in enumerate(resets.schedule):
        observed = float(row[RESET_INDEX_Value])
        due = base_date + pd.Timedelta(days=int(row[RESET_INDEX_Reset_Day]))
        rows.append(_row(FIXING, instrument, leg, position, due, index=index,
                         observed=observed or None, state=OBSERVED if observed else DUE))
    return rows


def _barrier_rows(fields, terms, index, instrument):
    """The deal's own monitoring table: a date and the close it fixed at, through the one reader
    that already tolerates both shapes of the row."""
    if terms is None or terms.table is None:
        return []
    return [_row(BARRIER, instrument, terms.table, position, date, index=index, observed=observed,
                 state=OBSERVED if observed is not None else DUE)
            for position, (date, observed)
            in enumerate(barrier_monitoring_rows(fields.get(terms.table) or []))]


def _declared_payment(fields, terms, settlement, instrument):
    """The payment a deal with NO schedule declares: its type's own settlement date, or its expiry
    where the type names no other, and the amount where a field holds one.

    `get_settlement_currencies()` is the reval-date accumulator - a barrier registers its
    monitoring days in it - so it is not a payment ladder and is never read as one. AN OPTION'S
    PAYOFF IS NOT IN A FIELD (`Units` times an intrinsic nobody has fixed), so its row is due with
    `amount: null`: the money still moves, and a close waits for the transition that moved it.
    """
    if terms is None:
        return []
    date = fields.get(terms.pays) if terms.pays else None
    date = date if date is not None else (fields.get(terms.expires) if terms.expires else None)
    if date is None:
        return []
    amount = fields.get(terms.amount) if terms.amount else None
    return [_row(PAYMENT, instrument, SETTLEMENT, 0, date,
                 currency=_currency(fields, settlement),
                 amount=None if amount is None else float(amount),
                 determined=amount is not None)]


def _expiry_fixing(fields, terms, index, instrument, rows):
    """The observation a deal's EXPIRY needs, where the compile builds no reset schedule for it.

    A European option's payoff is its underlying's print on the expiry day, and nothing in the
    compiled tree announces it - so a close after expiry would wait for nothing at all. A deal
    whose own schedules already announce a fixing on that day announces it once.
    """
    day = fields.get(terms.expires) if terms is not None and terms.expires else None
    if day is None or index is None:
        return []
    date = pd.Timestamp(day).date().isoformat()
    if any(row['kind'] == FIXING and row['due_date'] == date for row in rows):
        return []
    return [_row(FIXING, instrument, EXPIRY_LEG, 0, day, index=index)]


def _expiry_row(deal, instruments, state):
    """One row per deal at the last day it can pay, carrying what its expiry leaves to an actor:
    `election` where the terms vest a choice, and null where a fixing determines the payoff."""
    fields = deal.field
    terms = TERMS.get(fields.get('Object'))
    dates = deal.get_reval_dates()
    elects = terms is not None and terms.elects and fields.get(terms.elects) == PHYSICAL
    return _row(EXPIRY, instruments.get(fields.get('Reference')), EXPIRY_LEG, 0,
                max(dates) if dates else None, needs=ELECTION if elects else None, state=state)


def _row(kind, instrument, leg, position, due, **named):
    """One diary row: every declared field present, and the derived key stamped."""
    date = None if due is None else pd.Timestamp(due).date().isoformat()
    row = {'key': _key(instrument, leg, kind, date), 'instrument': instrument, 'leg': leg,
           'schedule_index': position, 'kind': kind, 'currency': None, 'amount': None,
           'determined': False, 'notional': None, 'index': None, 'source': None, 'observed': None,
           'needs': None, 'reason': None, 'state': DUE, 'due_date': date}
    row.update(named)
    return row


def _key(instrument, leg, kind, date):
    """The row's derived key, or None where nothing booked the instrument or the record is not
    installed on this box - the diary being a reading of the book either way.

    The row's own DATE and never its position: a schedule drops the rows it has paid as the book
    rolls, so a position renumbers under a settlement reference already filed while the date it
    named does not move.
    """
    if instrument is None or date is None:
        return None
    try:
        return cashflow_key(instrument, leg, kind, date)
    except SpineRefused:
        return None


def _currency(fields, settlement):
    """The currency a deal settles in: its OWN field, else the one currency its reval dates hold.

    Never a guess at which currency's date set carries the day: a two-currency deal's legs share a
    maturity, so that guess labels one leg with the other's currency and a settlement file nets the
    two. A deal whose legs really do settle apart is a container whose children are the legs, and
    each child names its own.
    """
    currency = fields.get('Currency')
    if isinstance(currency, str) and currency:
        return currency
    return next(iter(settlement)) if len(settlement) == 1 else None
