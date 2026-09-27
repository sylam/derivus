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

"""The desk's P&L - what the book made between two marked closes, or since the last one.

A close is MARKED by one standing run valuing ONE UNIT of every instrument the book holds or traded
since the last marks, each as written, on the close's own values: a position's value is its
quantity times its unit mark, any grouping of the book is a sum, and a past close replays from the
job the run attested rather than from a book file that has moved since. Between two marks, per
position,

    P&L = value at the end - value at the start + premiums + payments + fees

in the reporting currency: the premiums at each fill's consideration, crossed at the rate its
booking filed; the payments at what the diary determines, falling due in the window, else what the
settlements FILED in the window moved, and the fees as filed, each at the official close standing on
its own day - so a late filing lands in the day it was filed, a day already struck never moves, and
a month is the sum of its days; the realised half at average cost, a position's open basis released
in the window its last day falls in. A number nobody can know - a fill booked with no price, a
payment neither the diary nor a settlement states, a day no close stands on - is NAMED, never read
as zero.
"""
import json
from copy import deepcopy

from . import content_hash
from .diary import SETTLED
from .schema import instrument_of, job_children, mapping, tables_of
from .spine import book_name

#: The book a marks job files under - a name no book carries, so the compile writes no position
#: into it and every instrument prices as the one unit it was written as.
MARKS = 'marks:{}'

#: `{(job address, book): day}` - the day a marks job valued its book as of, or None for a job that
#: marks no book of that name. A job is immutable, so each is opened once per process.
DAYS = {}

#: A row's figures, in the order a reader sees them, and the ones a total sums.
FIGURES = ('value_start', 'value_end', 'premiums', 'payments', 'fees', 'existing', 'trading',
           'pnl', 'realised', 'unrealised')


def unit_node(address, terms):
    """One unit of the instrument `terms`, as written, filed under its own `address`."""
    node = {'Instrument': {'.Deal': dict(
        {key: value for key, value in terms.items() if key != 'Children'}, Reference=address)}}
    if terms.get('Children'):
        node['Children'] = deepcopy(terms['Children'])
    return node


def held_terms(document, wanted, stored):
    """`{address: terms}` for every instrument in `wanted` - read off the first node of the book
    file carrying it, a structure being ONE instrument with its legs inside its terms, and off the
    record's own copy of the terms where the file no longer holds one."""
    containers, found = mapping['Instrument']['containers'], {}

    def walk(children):
        for node in children:
            address = content_hash(instrument_of(node))
            if address in wanted:
                found.setdefault(address, instrument_of(node))
            elif node['Instrument']['.Deal'].get('Object') in containers:
                walk(node.get('Children', []))

    walk(job_children(document))
    for address in sorted(set(wanted) - set(found)):
        found[address] = json.loads(stored(address).decode('utf-8'))
    return found


def marks_job(document, terms):
    """The job marking `terms` ({address: terms}): one unit of each on `document`'s market and
    calculation, as a base valuation under the marks book."""
    calc = dict(document['Calc'])
    calc['Calculation'] = {key: value for key, value in dict(
        calc['Calculation'], Object='BaseValuation').items() if key != 'Greeks'}
    calc['Deals'] = dict(calc['Deals'], Reference=MARKS.format(book_name(document)), Deals={
        'Children': [unit_node(address, terms[address]) for address in sorted(terms)]})
    return {'Calc': calc}


def unit_marks(results):
    """`{address: value}` off a marks run's results - its top-level rows, each one unit of the
    instrument filed under its own address. An instrument the run could not price, or that had
    expired, has no row."""
    table = tables_of(results)['mtm']['.DataFrame']
    column = {name: position for position, name in enumerate(table['columns'])}
    root = table['data'][0][column['Reference']] if table['data'] else None
    return {row[column['Reference']]: float(row[column['Value']] or 0.0)
            for row in table['data'][1:] if row[column['Parent']] == root}


def marked(book, closes, attestations, stored):
    """`{day: marks}` - the LATEST standing marks run of `book` on each of `closes`, keyed by the
    day it valued the book as of: its position, the values it stood on, and the addresses of the
    job it attested and the result. `loaded` reads a marks' numbers."""
    from .structures import timestamp

    values, found = {row['values_hash'] for row in closes}, {}
    for row in attestations:
        if row['values_hash'] not in values:
            continue
        if (row['job'], book) not in DAYS:
            job = json.loads(stored(row['job']).decode('utf-8'))
            DAYS[(row['job'], book)] = None if job['Calc']['Deals'].get(
                'Reference') != MARKS.format(book) else timestamp(
                job['Calc']['Calculation']['Base_Date']).strftime('%Y-%m-%d')
        day = DAYS[(row['job'], book)]
        if day is not None and (day not in found or row['lsn'] > found[day]['lsn']):
            found[day] = {'day': day, 'lsn': row['lsn'], 'values_hash': row['values_hash'],
                          'job': row['job'], 'result': row['result']}
    return found


def loaded(marks, stored):
    """`marks` with what a P&L reads off them: the `document` of the job it attested and the unit
    marks of its result."""
    def read(digest):
        return json.loads(stored(digest).decode('utf-8'))

    return dict(marks, document=read(marks['job']), units=unit_marks(read(marks['result'])))


def within(row, portfolio=None, agreement=None, clients=None):
    """Whether a position row sits in the scope asked for: under a portfolio path, under one
    agreement, or with a counterparty among `clients` - the whole book where nothing is named."""
    return ((portfolio is None or row['portfolio'] == portfolio
             or row['portfolio'].startswith(portfolio + '/'))
            and (agreement is None or row['agreement'] == agreement)
            and (clients is None or row.get('counterparty') in clients))


def rates(values, reporting, base):
    """`{currency: units of the reporting currency per unit}` off a values vector's spots - the
    engine's axis, every spot priced in `base`, which carries none of its own and is one."""
    spots = {name[len('FxRate.'):]: fields['Spot'] for name, fields in values.items()
             if name.startswith('FxRate.') and '.' not in name[len('FxRate.'):]
             and 'Spot' in fields}
    if base:
        spots[base] = 1.0
    return {} if not spots.get(reporting) else {
        currency: spot / spots[reporting] for currency, spot in spots.items() if spot}


def rates_on(closes, stored, reporting, base):
    """`day -> rates` - what one unit of each currency is worth in the reporting one at the official
    close STANDING on a day: the last declared for that day or before it, a restatement standing
    over the close it restates. None where no close comes on or before the day. This is the rate a
    movement of cash is booked at, so a month's cash is the sum of its days'."""
    ordered, answered = sorted(closes, key=lambda close: (close['date'], close['lsn'])), {}

    def on(day):
        found = None
        for close in ordered:
            if close['date'] > day:
                break
            found = close
        if found is None:
            return None
        if found['values_hash'] not in answered:
            answered[found['values_hash']] = rates(
                json.loads(stored(found['values_hash']).decode('utf-8')), reporting, base)
        return answered[found['values_hash']]

    return on


def filed(before, after):
    """The movements of money filed between two positions of the record, off the `cash` rows at
    each: a new one whole, and a restated one as the filing it corrects taken back and the
    correction put in its place, each on its own value date."""
    standing, moved = {row['reference']: row for row in before}, []
    for row in after:
        old = standing.get(row['reference'])
        if old is None or old['lsn'] != row['lsn']:
            moved.extend(([dict(old, amount=-old['amount'])] if old else []) + [row])
    return moved


def recorded(cut, costs, fills):
    """`day -> {held, through, dealt}` - what each position held before a day's trading and through
    it, and what it traded on it, as the record stood at the marks: `cut` is every marks' `(lsn,
    day)`, `costs(lsn)` the `costs` rows there and `fills(after, until)` the fills between two
    positions. A movement dated before its window is placed by its own day through this."""
    marked, folded, answered = dict(cut), {}, {}

    def held(lsn):
        if lsn not in folded:
            folded[lsn] = {} if lsn is None else {
                (row['instrument'], row['agreement'], row['portfolio']): row['quantity']
                for row in costs(lsn)}
        return folded[lsn]

    def on(day):
        if day not in answered:
            before = max((lsn for lsn, at in cut if at < day), default=None)
            through = max((lsn for lsn, at in cut if at <= day), default=None)
            dealt = {}
            for fill in fills(before or 0, through) if marked.get(through) == day else ():
                dealt[_where(fill)] = dealt.get(_where(fill), 0.0) + abs(float(fill['quantity']))
            answered[day] = {'held': held(before), 'through': held(through), 'dealt': dealt}
        return answered[day]

    return on


def pnl(start, end, costs_start, costs_end, fills, diary, last_days, movements, positions_end,
        scope, rates_on, days):
    """The P&L between two marks, per position and in total.

    `start` and `end` are marks (`{day, lsn, units}`); `costs_*` the `costs` rows at their two
    positions; `fills` every fill between them, each with the business `day` it was traded on;
    `diary` every payment row the instruments held in the window announce from the start on, per
    unit, keyed as the record keys a settlement and answered as it stands at the end; `last_days`
    the last day of every instrument that has one; `movements` the money `filed` between the two
    marks; `positions_end` the `positions` rows at the end, which carry the counterparty; `scope`
    the predicate a position must satisfy; `rates_on(day)` what a currency is worth in the
    reporting one at the close standing on a day; and `days` the record on a day before the window
    (`recorded`), so a movement dated back is placed by its own day.

    A MOVEMENT COUNTS IN THE WINDOW IT WAS FILED IN, at the close standing on its own value date:
    a late filing lands in the day it was filed and a day already struck never moves.

    A position whose last day falls in the window CLOSES at its settlement value there: its mark
    at the end is nothing, its open basis is released to realised and its payoff arrives as the
    payment it is. One whose last day came on or before the start is worth nothing, and it and one
    holding nothing stand only through the money they still move - a payoff settling at T+2, a fee
    filed after the position closed.
    """
    def key_of(row):
        return row['instrument'], row['agreement'], row['portfolio']

    counterparties = {key_of(row): row.get('counterparty') for row in positions_end}
    before, after = ({key_of(row): row for row in rows} for rows in (costs_start, costs_end))
    window = {}
    for fill in fills:
        window.setdefault(_where(fill), []).append(fill)
    settled = {}
    for movement in movements:
        settled.setdefault((movement['kind'], movement['subject']), []).append(movement)
    every = set(before) | set(after) | set(window)
    held = {}
    for key in every:
        held.setdefault(key[0], []).append(key)
    flows = {'diary': diary, 'settled': settled, 'held': held, 'before': before,
             'window': window, 'rates_on': rates_on, 'days': days, 'start': start['day'],
             'end': end['day']}

    # a settlement against a row no instrument here announces cannot be placed on a position
    carried = {row['key'] for announced in diary.values() for row in announced}
    unknown = [{'instrument': None, 'what': 'settlement {} was filed against a payment this '
                "window's diary does not announce".format(movement['reference'])}
               for (kind, subject), moved in settled.items() if kind == 'payment'
               and subject not in carried for movement in moved]
    rows = []
    for key in sorted(every):
        instrument, agreement, portfolio = key
        q_start = before.get(key, {}).get('quantity', 0.0)
        q_end = after.get(key, {}).get('quantity', 0.0)
        traded = window.get(key, [])
        last = last_days.get(instrument)
        if not within({'agreement': agreement, 'portfolio': portfolio,
                       'counterparty': counterparties.get(key)}, **scope):
            continue
        payments = _payments(key, flows, unknown)
        fees = _fees(key, flows, unknown)
        closed = last is not None and last <= start['day']
        # one holding nothing, or past its last day, stands only through the money it still moves
        if (closed or not (q_start or q_end)) and not traded and payments == 0.0 and fees == 0.0:
            continue
        ended = last is not None and last <= end['day']
        u_start, u_end = start['units'].get(instrument), end['units'].get(instrument)
        # a mark is UNKNOWN where the position needs one the run did not answer, save where the
        # instrument's last day has come, which leaves it worth nothing
        mark_start = 0.0 if closed else None if q_start and u_start is None else u_start or 0.0
        mark_end = None if (q_end or traded) and u_end is None and not ended else (
            0.0 if ended else u_end or 0.0)
        if mark_start is None:
            unknown.append({'instrument': instrument, 'what': 'no mark at the start'})
        if mark_end is None:
            unknown.append({'instrument': instrument, 'what': 'no mark at the end - the run '
                            'could not price it'})
        value_start = None if mark_start is None else q_start * mark_start
        value_end = None if mark_end is None else q_end * mark_end
        premiums, dealt = 0.0, sum(float(fill['quantity']) for fill in traded)
        for fill in traded:
            if fill.get('price') is None:
                unknown.append({'instrument': instrument, 'what': 'fill {} was booked with no '
                                'price'.format(fill['execution_reference'])})
                premiums = None
            elif premiums is not None:
                premiums -= float(fill['quantity']) * float(fill['price']) * float(
                    fill.get('rate', 1.0))
        # the window its last day falls in is where a position closes, at its settlement value
        released = (after.get(key, {}).get('basis', 0.0) if ended and last > start['day']
                    else 0.0)
        moved = _less(value_end, value_start)
        earned = None if mark_end is None else dealt * mark_end
        total = _sum(moved, premiums, payments, fees)
        realised = _sum(_less(after.get(key, {}).get('realised', 0.0),
                              before.get(key, {}).get('realised', 0.0)), payments, fees,
                        _less(0.0, released))
        rows.append({
            'instrument': instrument, 'agreement': agreement, 'portfolio': portfolio,
            'counterparty': counterparties.get(key), 'quantity_start': q_start,
            'quantity_end': q_end, 'unit_start': u_start, 'unit_end': u_end,
            'value_start': value_start, 'value_end': value_end, 'premiums': premiums,
            'payments': payments, 'fees': fees,
            # what the book held moved, and what was traded in the window earned against the end
            'existing': _less(moved, earned), 'trading': _sum(earned, premiums),
            'pnl': total, 'realised': realised, 'unrealised': _less(total, realised)})
    unknown = [json.loads(named) for named in sorted({json.dumps(row, sort_keys=True)
                                                      for row in unknown})]
    return {'rows': rows, 'unknown': unknown, 'realised_method': 'average cost',
            'total': {figure: sum(row[figure] for row in rows if row[figure] is not None)
                      for figure in FIGURES},
            'complete': not unknown and all(row[figure] is not None
                                            for row in rows for figure in FIGURES)}


def _where(fill):
    """The position a fill moved: its instrument, under its agreement - its netting set where it
    names none - in its portfolio, else its book."""
    return (fill['instrument'], fill.get('agreement') or fill['netting_set'],
            fill.get('portfolio') or fill['book'] or '')


def _held(position, day, flows, through):
    """What `position` held on `day`, before the day's trading or through it where `through`: the
    start's quantity and the window's fills, or the record as it stood where the day came on or
    before the start."""
    if day <= flows['start']:
        return flows['days'](day)['through' if through else 'held'].get(position, 0.0)
    return flows['before'].get(position, {}).get('quantity', 0.0) + sum(
        float(fill['quantity']) for fill in flows['window'].get(position, [])
        if fill['day'] < day or through and fill['day'] == day)


def _dealt(position, day, flows):
    """What `position` traded on `day`: the window's fills of that business day, or the record's
    where the day came on or before the start."""
    if day <= flows['start']:
        return flows['days'](day)['dealt'].get(position, 0.0)
    return sum(abs(float(fill['quantity'])) for fill in flows['window'].get(position, [])
               if fill['day'] == day)


def _payments(key, flows, unknown):
    """What this position was paid in the window: the diary's amount times what it held the day
    before, for every payment the diary determines falling due in the window; and its share - by
    what each position held the day before the row fell due - of what every settlement FILED in the
    window moved against a row the diary leaves undetermined, each at the close standing on its own
    day. An undetermined payment falling due that the record has not settled by the window's end is
    UNKNOWN, named; one settled with no amount moved nothing."""
    total = 0.0
    for row in flows['diary'].get(key[0], []):
        due, moved = row['due_date'], flows['settled'].get(('payment', row['key']), [])
        falls = flows['start'] < due <= flows['end']
        if not (falls or moved and not row['determined']):
            continue
        mine = _held(key, due, flows, False)
        if row['determined']:
            if mine:
                total = _sum(total, _converted(mine * row['amount'], row['currency'], due, flows,
                                               key[0], unknown))
            continue
        if mine and falls and row['state'] != SETTLED:
            unknown.append({'instrument': key[0], 'what': 'the {} payment due {} is not '
                            'determined and nothing settled it'.format(row['leg'], due)})
            return None
        for movement in moved:
            among = _booked(flows['held'][key[0]], movement)
            net = sum(_held(other, due, flows, False) for other in among)
            if mine and net and key in among:
                total = _sum(total, _converted(
                    movement['amount'] * mine / net, movement['asset'],
                    movement['effective_time'][:10], flows, key[0], unknown))
    return total


def _booked(positions, movement):
    """The positions a movement is shared across: those under the book it was filed for, every
    one where it names none."""
    book = movement.get('book')
    return [key for key in positions
            if not book or key[2] == book or key[2].startswith(book + '/')]


def _fees(key, flows, unknown):
    """This position's share of the fees FILED on its instrument in the window, among the
    positions under the book each was filed for - by what each traded on the fee's value date
    where the instrument traded that day, else by what each held on it - at the close standing on
    that day. Whose a fee is depends on its own day and never on the window it lands in."""
    total = 0.0
    for movement in flows['settled'].get(('fee', key[0]), []):
        day, among = movement['effective_time'][:10], _booked(flows['held'][key[0]], movement)
        dealt = {other: _dealt(other, day, flows) for other in among}
        weights = dealt if any(dealt.values()) else {
            other: abs(_held(other, day, flows, True)) for other in among}
        share = weights.get(key, 0.0) / sum(weights.values()) if sum(weights.values()) else 0.0
        total = _sum(total, _converted(share * movement['amount'], movement['asset'], day, flows,
                                       key[0], unknown))
    return total


def _converted(amount, currency, day, flows, instrument, unknown):
    """`amount` of `currency` in the reporting currency at the close standing on `day`, or UNKNOWN
    - named - where no close stands on it or it carries no spot for the currency."""
    rates = flows['rates_on'](day)
    if rates is not None and currency in rates:
        return amount * rates[currency]
    unknown.append({'instrument': instrument, 'what': (
        'no official close on or before {} to report what moved in {}'.format(day, currency)
        if rates is None else 'the close standing on {} carries no spot for {}'.format(
            day, currency))})
    return None


def _sum(*parts):
    """A total that stays UNKNOWN once any part is."""
    return None if any(part is None for part in parts) else sum(parts)


def _less(held, less):
    """A difference that stays UNKNOWN once either side is."""
    return None if held is None or less is None else held - less
