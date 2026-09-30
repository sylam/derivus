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
its own day as the record stood at the end of the business day it counts in - so a late filing
lands in the day it was filed, a day already struck never moves, and a month is the sum of its days;
the realised half at average cost, a position's open basis released in the window its last day
falls in. The engine values a payment into the marks of the day it falls due, so a position's value
at a close is its quantity times its unit mark LESS what that day paid - the close is the end of the
day and the payment is cash by then. A number nobody can know - a fill booked with no price, a
payment neither the diary nor a settlement states, a day no close stands on - is NAMED, never read
as zero.
"""
import json
from copy import deepcopy

from . import content_hash
from .diary import SETTLED
from .schema import instrument_of, job_children, mapping, tables_of
from .spine import book_name, under, visible

#: The book a marks job files under - a name no book carries, so the compile writes no position
#: into it and every instrument prices as the one unit it was written as.
MARKS = 'marks:{}'

#: `{(job address, book): day}` - the day a marks job valued its book as of, or None for a job that
#: marks no book of that name. A job is immutable, so each is opened once per process.
DAYS = {}

#: A row's figures, in the order a reader sees them, and the ones a total sums.
FIGURES = ('value_start', 'value_end', 'premiums', 'payments', 'fees', 'existing', 'trading',
           'pnl', 'realised', 'unrealised')
#: The new-deal split, which needs a unit mark at the end for a position traded to nothing in the
#: window - one the end's marks may not carry - and so can be unknown while the P&L is not.
SPLIT = ('existing', 'trading')


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


def held_legs(children, known):
    """`{leg address: holder address}` for every leg inside an instrument of `known` - a structure
    held whole is ONE instrument, so what its legs pay, and the day the last of them ends, are
    its own. `children` is a deal tree, the netting sets it nests positions in walked through; a
    node is the instrument its terms hash to, or a unit a marks job files under that address."""
    containers, found = mapping['Instrument']['containers'], {}

    def walk(nodes):
        for node in nodes:
            address = content_hash(instrument_of(node))
            filed = node['Instrument']['.Deal'].get('Reference')
            holder = address if address in known else filed if filed in known else None
            if holder is not None:
                found.update((content_hash(instrument_of(leg)), holder)
                             for leg in _leaves(node.get('Children', [])))
            elif node['Instrument']['.Deal'].get('Object') in containers:
                walk(node.get('Children', []))

    walk(children)
    return found


def apart(units):
    """`units` in as few groups as leave every leaf to one unit of its group: a leaf two units
    carry - the same terms, or one reference over other terms, as a restruck structure's legs
    beside the ones it replaced - is read once per unit, each unit in a group of its own."""
    groups = []
    for unit in units:
        leaves = {part for leaf in _leaves([unit]) for part in (
            leaf['Instrument']['.Deal'].get('Reference'), content_hash(instrument_of(leaf)))}
        group = next((group for group in groups if not group[1] & leaves), None)
        if group is None:
            groups.append(([unit], leaves))
        else:
            group[0].append(unit)
            group[1].update(leaves)
    return [group for group, _ in groups]


def _leaves(nodes):
    """Every node under `nodes` with no children of its own - a node with none being its own."""
    for node in nodes:
        if node.get('Children'):
            yield from _leaves(node['Children'])
        else:
            yield node


def marks_job(document, terms, cut=None):
    """The job marking `terms` ({address: terms}): one unit of each on `document`'s market and
    calculation, as a base valuation under the marks book - which names `cut`, the position of the
    record the terms were read at, where the marks close a business day there."""
    calc = dict(document['Calc'])
    calc['Calculation'] = {key: value for key, value in dict(
        calc['Calculation'], Object='BaseValuation').items() if key != 'Greeks'}
    book = MARKS.format(book_name(document)) + ('' if cut is None else '@{}'.format(cut))
    calc['Deals'] = dict(calc['Deals'], Reference=book, Deals={
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
    day it valued the book as of: the position of the record it read the book at, which closes
    that business day, the values it stood on, and the addresses of the job it attested and the
    result. `loaded` reads a marks' numbers."""
    from .structures import timestamp

    values, found = {row['values_hash'] for row in closes}, {}
    for row in attestations:
        if row['values_hash'] not in values:
            continue
        if (row['job'], book) not in DAYS:
            job = json.loads(stored(row['job']).decode('utf-8'))
            reference = job['Calc']['Deals'].get('Reference') or ''
            # the cut follows the LAST '@', which a book's own name may carry too
            name, _, cut = reference.rpartition('@')
            cut = int(cut) if name == MARKS.format(book) and cut.isdigit() else None
            DAYS[(row['job'], book)] = None if cut is None and reference != MARKS.format(
                book) else (timestamp(job['Calc']['Calculation']['Base_Date']).strftime(
                    '%Y-%m-%d'), cut)
        if DAYS[(row['job'], book)] is None:
            continue
        day, cut = DAYS[(row['job'], book)]
        if day not in found or row['lsn'] > found[day]['attested']:
            # a marks job naming no cut read the book where it was attested
            found[day] = {'day': day, 'lsn': row['lsn'] if cut is None else cut,
                          'attested': row['lsn'], 'values_hash': row['values_hash'],
                          'job': row['job'], 'result': row['result']}
    return found


def loaded(marks, stored):
    """`marks` with what a P&L reads off them: the `document` of the job it attested and the unit
    marks of its result."""
    def read(digest):
        return json.loads(stored(digest).decode('utf-8'))

    return dict(marks, document=read(marks['job']), units=unit_marks(read(marks['result'])))


def within(row, portfolio=None, agreement=None, clients=None, sight=None):
    """Whether a position row sits in the scope asked for: under a portfolio path, under one
    agreement, with a counterparty among `clients`, and where a seat whose `sight` this is sees it
    (`spine.visible`) - the whole book where nothing is named."""
    return ((portfolio is None or under(row['portfolio'], portfolio))
            and (agreement is None or row['agreement'] == agreement)
            and (clients is None or row.get('counterparty') in clients)
            and bool(visible([row], sight)))


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
    """`(day, lsn) -> rates` - what one unit of each currency is worth in the reporting one at the
    official close STANDING on a day, as the record stood at `lsn`: the last declared for that day
    or before it among those filed by then, a restatement standing over the close it restates.
    None where none stands. A movement of cash is booked at it as the record stood at the end of
    the business day the movement counts in, so a window read again later, or inside a longer one,
    books it at the same rate and a month stays the sum of its days."""
    ordered, answered = sorted(closes, key=lambda close: (close['date'], close['lsn'])), {}

    def on(day, lsn):
        found = None
        for close in ordered:
            if close['date'] > day:
                break
            if close['lsn'] <= lsn:
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


def recorded(cut, costs, fills, amendments, known=None):
    """`day -> {held, through, dealt}` - what each position held before a day's trading and through
    it, and what it traded on it, as the record stood at the marks: `cut` is every marks' `(lsn,
    day)` and the end's, oldest first, `costs(lsn)` the `costs` rows there - `known` the ones
    already read - and `fills(after, until)` and `amendments(after, until)` what was filed between
    two positions. An amendment restrikes the trade rather than trading it, so what it moved is
    held on its new terms from the START of the business day it was filed in, whatever that day
    paid on them."""
    marked, answered = dict(cut), {}
    folded = {lsn: {(row['instrument'], row['agreement'], row['portfolio']): row['quantity']
                    for row in rows} for lsn, rows in (known or {}).items()}

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
            dealt, opening = {}, held(before)
            if marked.get(through) == day:
                for fill in fills(before or 0, through):
                    dealt[_where(fill)] = dealt.get(_where(fill), 0.0) + abs(
                        float(fill['quantity']))
                opening = _restruck(opening, amendments(before or 0, through))
            answered[day] = {'held': opening, 'through': held(through), 'dealt': dealt}
        return answered[day]

    return on


def _restruck(held, amendments):
    """`held` with every position of the terms each of `amendments` restruck moved onto the terms
    they became, in the order they were filed."""
    held = dict(held)
    for amendment in amendments:
        for key in [key for key in held if key[0] == amendment['instrument'] and held[key]]:
            onto = (amendment['amended_to'],) + key[1:]
            held[onto] = held.get(onto, 0.0) + held.pop(key)
    return held


def pnl(start, end, record, scope):
    """The P&L between two marks, per position and in total.

    `start` and `end` are marks (`{day, lsn, units, rates}`), `rates` what one unit of each
    currency was worth in the reporting one at the close they stood on. `record` is what the
    record says of the window: `costs_start` and `costs_end`, `cash_start` and `cash_end`, the
    `costs` and `cash` rows at the two marks; `fills`, every fill between them with the business
    `day` it was traded on; `diary`, every payment the instruments held in the window announce,
    per unit, keyed as a settlement names it and answered as the record stands at the end;
    `last_days`, the last day of every instrument that has one; `positions_end`, the `positions`
    rows at the end, which carry the counterparty; `cut`, every marks' `(lsn, day)` and the end's,
    oldest first; `days`, the record on a day (`recorded`); `rates(day, lsn)`, the close standing
    on a day as filed by a position; and `fills_between(after, until)`. `scope` narrows the rows.

    A MOVEMENT COUNTS IN THE WINDOW IT WAS FILED IN, and a payment the diary determines in the one
    it falls due in, each read as the record stood at the end of that business day: its rate, and
    whose it is. A late filing lands in the day it was filed and a day already struck never moves.

    THE CLOSE IS THE END OF THE DAY. The marks value a payment into the day it falls due, so a
    position's value at a close is its quantity times its unit mark less what the unit PAID that
    day: every payment the diary determines, and every one it does not that a settlement filed by
    the marks moved. One nothing settled by then is still the book's and stays in its value, and a
    window it leaves without the money is UNKNOWN, named.

    A position whose last day falls in the window CLOSES at its settlement value there: its mark
    at the end is nothing, its open basis is released to realised and its payoff arrives as the
    payment it is. One whose last day came on or before the start is worth nothing, realises only
    the money it moves, and it and one holding nothing stand only through that money - a payoff
    settling at T+2, a fee filed after the position closed.
    """
    def key_of(row):
        return row['instrument'], row['agreement'], row['portfolio']

    counterparties = {key_of(row): row.get('counterparty') for row in record['positions_end']}
    before, after = ({key_of(row): row for row in rows}
                     for rows in (record['costs_start'], record['costs_end']))
    window = {}
    for fill in record['fills']:
        window.setdefault(_where(fill), []).append(fill)
    settled, standing = {}, {'start': {}, 'end': {}}
    for movement in filed(record['cash_start'], record['cash_end']):
        settled.setdefault((movement['kind'], movement['subject']), []).append(movement)
    for side in standing:
        for row in record['cash_' + side]:
            standing[side].setdefault((row['kind'], row['subject']), []).append(row)
    every = set(before) | set(after) | set(window)
    held = {}
    for key in every:
        held.setdefault(key[0], []).append(key)
    carried = {}
    for instrument, announced in record['diary'].items():
        for row in announced:
            carried.setdefault(row['key'], []).append((instrument, row))
    flows = dict(record, start=start['day'], end=end['day'], settled=settled, standing=standing,
                 held=held, carried=carried, owners={}, paid={})

    unknown = []
    for (kind, subject), moved in sorted(settled.items()):
        for movement in moved:
            # a settlement no row here announces, one against a payment nobody held, and a fee
            # nobody traded or held, belong to nothing
            if kind == 'payment' and subject not in carried:
                unknown.append({'instrument': None, 'what': 'settlement {} was filed against a '
                                "payment this window's diary does not announce".format(
                                    movement['reference'])})
            elif kind == 'payment':
                instrument, row = carried[subject][0]
                held_then = _holdings(row, movement, flows)
                if not any(held_then.get(other) for other in _carrying(subject, movement, flows)):
                    unknown.append({'instrument': instrument, 'what': 'settlement {} moved a '
                                    'payment no position held when it fell due'.format(
                                        movement['reference'])})
            if kind == 'fee' and not _owners(movement, flows):
                unknown.append({'instrument': subject, 'what': 'fee {} falls to no position - '
                                'none under the book traded or held what it was filed on'.format(
                                    movement['reference'])})
    rows = []
    for key in sorted(every):
        instrument, agreement, portfolio = key
        if not within({'agreement': agreement, 'portfolio': portfolio,
                       'counterparty': counterparties.get(key)}, **scope):
            continue
        q_start = before.get(key, {}).get('quantity', 0.0)
        q_end = after.get(key, {}).get('quantity', 0.0)
        traded = window.get(key, [])
        last = flows['last_days'].get(instrument)
        payments = _payments(key, flows, unknown)
        fees = _fees(key, flows, unknown)
        closed = last is not None and last <= start['day']
        # one holding nothing, or past its last day, stands only through the money it still moves
        if (closed or not (q_start or q_end)) and not traded and payments == 0.0 and fees == 0.0:
            continue
        ended = last is not None and last <= end['day']
        u_start, u_end = start['units'].get(instrument), end['units'].get(instrument)
        paid_start = (0.0, None) if closed else _paid(instrument, start, 'start', flows)
        paid_end = (0.0, None) if ended else _paid(instrument, end, 'end', flows)
        # a unit at a close, less what it paid that day; nothing once its last day has come
        mark_start = 0.0 if closed else _worth(u_start, paid_start[0])
        mark_end = 0.0 if ended else _worth(u_end, paid_end[0])
        value_start = 0.0 if not q_start else _times(q_start, mark_start)
        value_end = 0.0 if not q_end else _times(q_end, mark_end)
        for value, unit, paid, side in ((value_start, u_start, paid_start, 'start'),
                                        (value_end, u_end, paid_end, 'end')):
            if value is None:
                unknown.append({'instrument': instrument, 'what': paid[1] if unit is not None
                                else 'no mark at the {}{}'.format(side, (
                                    ' - the run could not price it' if side == 'end' else ''))})
        premiums, dealt = 0.0, sum(float(fill['quantity']) for fill in traded)
        for fill in traded:
            if fill.get('price') is None:
                unknown.append({'instrument': instrument, 'what': 'fill {} was booked with no '
                                'price'.format(fill['execution_reference'])})
                premiums = None
            elif premiums is not None:
                premiums -= float(fill['quantity']) * float(fill['price']) * float(
                    fill.get('rate', 1.0))
        moved = _less(value_end, value_start)
        # what the window's trades earned against the end: nothing where they net to nothing
        earned = 0.0 if not dealt else _times(dealt, mark_end)
        total = _sum(moved, premiums, payments, fees)
        rows.append({
            'instrument': instrument, 'agreement': agreement, 'portfolio': portfolio,
            'counterparty': counterparties.get(key), 'quantity_start': q_start,
            'quantity_end': q_end, 'unit_start': u_start, 'unit_end': u_end,
            'paid_start': paid_start[0], 'paid_end': paid_end[0],
            'value_start': value_start, 'value_end': value_end, 'premiums': premiums,
            'payments': payments, 'fees': fees,
            # what the book held moved, and what was traded in the window earned against the end
            'existing': _less(moved, earned), 'trading': _sum(earned, premiums),
            'pnl': total, 'realised': _realised(key, closed, ended and last > start['day'],
                                                (before, after), (premiums, payments, fees),
                                                unknown)})
        rows[-1]['unrealised'] = _less(total, rows[-1]['realised'])
    unknown = [json.loads(named) for named in sorted({json.dumps(row, sort_keys=True)
                                                      for row in unknown})]
    return {'rows': rows, 'unknown': unknown, 'realised_method': 'average cost',
            'total': {figure: _sum(*(row[figure] for row in rows)) for figure in FIGURES},
            'complete': not unknown and all(row[figure] is not None for row in rows
                                            for figure in FIGURES if figure not in SPLIT)}


def explain(rows, last_days, start, end, run):
    """WHY the positions a window started with made what they made - the P&L `rows` of one scope
    between the days `start` and `end` - never assembling the P&L out of its pieces.

    `existing` is theirs, and it is the CARRY of the start's book rolled to the end's day at the
    start's own quotes - a position whose last day fell in the window closing at nothing, which
    time alone did - plus the MARKET, every risk factor's move between the two closes times the
    start's own sensitivity to it, plus the RESIDUAL nothing here explains. `run(quantities)` values
    the positions still standing at the end - `{instrument: quantity}` - and answers `(opened,
    closed, rolled, note)`: the start's value and sensitivities, the end's levels of the same
    factors, the start's value rolled to the end's day and why its quotes did not connect, where
    they did not. The runs value a day's payments into that day as the marks do, and the carry
    takes out what a row's values take out. Reserves are not carried, so none is explained.
    """
    quantities, carry, existing, paid, unknown = {}, 0.0, 0.0, {}, []
    for row in rows:
        existing = _sum(existing, row['existing'])
        if row['existing'] is None:
            unknown.append({'instrument': row['instrument'], 'what': 'what it held moved is not '
                            'known, so neither is what explains it'})
        last = last_days.get(row['instrument'])
        if not row['quantity_start'] or last is not None and last <= start:
            continue
        if last is not None and last <= end:
            carry = _sum(carry, row['existing'])
        else:
            quantities[row['instrument']] = (quantities.get(row['instrument'], 0.0)
                                             + row['quantity_start'])
            paid[row['instrument']] = _less(row['paid_start'], row['paid_end'])
    answer = {'existing': existing, 'carry': carry, 'market': 0.0, 'residual': None,
              'reserves': None, 'factors': [], 'unknown': unknown, 'note': None}
    quantities = {instrument: held for instrument, held in quantities.items() if held}
    if quantities:
        try:
            opened, closed, rolled, answer['note'] = run(quantities)
        except Exception as refused:
            answer['unknown'].append({'instrument': None, 'what': 'the positions would not '
                                      'value for the explain: {}'.format(refused)})
            answer['carry'] = answer['market'] = None
            return answer
        taken = _sum(*(_times(quantities[instrument], paid[instrument])
                       for instrument in sorted(quantities)))
        if taken is None:
            answer['unknown'].append({'instrument': None, 'what': 'what a position paid on the '
                                      'day of a marks is not known, so neither is its carry'})
        answer['carry'] = _sum(carry, rolled - opened['value'], taken)
        answer['factors'], moved = _moved(opened, closed)
        answer['unknown'].extend(moved)
        answer['market'] = sum(row['pnl'] for row in answer['factors'] if row['pnl'] is not None)
    answer['residual'] = _less(_less(existing, answer['carry']), answer['market'])
    return answer


def _moved(opened, closed):
    """`(rows, unknown)` - every risk factor the start's sensitivities read that moved, with its
    move to the end and what that move made: per quote the factors are built from, and per factor
    for the rest - the factors a quote stands for being explained by the quote. Largest first."""
    rows, unknown = [], []
    ends = {(row['block'], row['quote']): row['level'] for row in closed['quotes']}
    for row in opened['quotes']:
        rows.append(dict({'block': row['block'], 'quote': row['quote']}, **_move(
            row, ends.get((row['block'], row['quote'])))))
    ends = {(row['factor'], tuple(row.get('tenor', ()))): row['level'] for row in closed['greeks']}
    for row in opened['greeks']:
        if row['factor'] not in opened['quoted']:
            named = {'factor': row['factor'], 'tenor': row.get('tenor', [])}
            rows.append(dict(named, **_move(row, ends.get((row['factor'],
                                                           tuple(named['tenor']))))))
    # a factor that did not move explains nothing, and one whose move is unknown is named
    rows = [row for row in rows if row['delta'] and row['move'] != 0.0]
    for row in rows:
        if row['pnl'] is None:
            unknown.append({'instrument': None, 'what': 'no level at the end for {}'.format(
                ' '.join(str(part) for part in (row.get('block') or row['factor'],
                                                row.get('quote') or row['tenor']) if part))})
    return sorted(rows, key=lambda row: (-abs(row['pnl'] or 0.0), json.dumps(row))), unknown


def _move(row, end):
    """One factor's sensitivity at the start, its level at each end and what the move made."""
    move = None if end is None or row['level'] is None else end - row['level']
    return {'delta': row['value'], 'start': row['level'], 'end': end, 'move': move,
            'pnl': None if move is None else row['value'] * move}


def _where(fill):
    """The position a fill moved: its instrument, under its agreement - its netting set where it
    names none - in its portfolio, else its book."""
    return (fill['instrument'], fill.get('agreement') or fill['netting_set'],
            fill.get('portfolio') or fill['book'] or '')


def _filed(lsn, flows):
    """`(lsn, day)` of the business day a filing at `lsn` counts in: the first marks at or after
    it, else the window's end."""
    return next(((at, day) for at, day in flows['cut'] if at >= lsn), flows['cut'][-1])


def _falls(day, flows):
    """The position of the marks of the business day `day` falls in: the first marked on or after
    it, else the window's end."""
    return min(((at_day, at) for at, at_day in flows['cut'] if at_day >= day),
               default=(None, flows['cut'][-1][0]))[1]


def _realised(key, closed, ending, costs, cash, unknown):
    """What a position realised in the window, at average cost: what its reductions realised, what
    it was paid and charged, and the open basis its last day released - or, past its last day, the
    money it moved alone, the basis having been released on that day. UNKNOWN, named, where a
    reduction closed at no price or against a lot booked with none, or the basis released was one.
    """
    premiums, payments, fees = cash
    if closed:
        return _sum(premiums, payments, fees)
    before, after = (rows.get(key, {}) for rows in costs)
    if after.get('unpriced_reductions', 0) != before.get('unpriced_reductions', 0):
        unknown.append({'instrument': key[0], 'what': 'a reduction closed at no price, or against '
                        'a lot booked with none, so what it realised is not known'})
        return None
    released = after.get('basis', 0.0) if ending else 0.0
    if released is None:
        unknown.append({'instrument': key[0], 'what': 'what it held at its last day was booked '
                        'with no price, so the basis released there is not known'})
    return _sum(after.get('realised', 0.0) - before.get('realised', 0.0), payments, fees,
                _less(0.0, released))


def _paid(instrument, marks, side, flows):
    """`(per unit, why)` - what one unit of `instrument` paid on the day `marks` value it, which the
    unit mark still carries, in the reporting currency at the close the marks stood on: every
    payment the diary determines falling due that day, and every one it does not that a settlement
    filed by the marks moved, per unit held when it fell due. None, with why, where that is not
    known."""
    if (instrument, side) not in flows['paid']:
        total, why, day = 0.0, None, marks['day']
        for row in flows['diary'].get(instrument, []):
            if row['determined']:
                moved = [(row['amount'], row['currency'])] if row['due_date'] == day else []
            elif row['due_date'] < day:
                moved = []
            else:
                moved = []
                for movement in flows['standing'][side].get(('payment', row['key']), []):
                    among = _carrying(row['key'], movement, flows)
                    held = _holdings(row, movement, flows)
                    if not any(held.get(other) for other in among):
                        # nobody here held it when it fell due: named in the window it was filed
                        continue
                    net = sum(held.get(other, 0.0) for other in among)
                    if not net:
                        why = 'settlement {} moved a payment of positions netting to nothing, so ' \
                              'what one unit was paid is not known'.format(movement['reference'])
                        break
                    moved.append((movement['amount'] / net, movement['asset']))
            for amount, currency in moved:
                if currency not in marks['rates']:
                    why = 'the close the {} marks stood on carries no spot for {}'.format(
                        day, currency)
                    continue
                total += amount * marks['rates'][currency]
            if why is not None:
                break
        flows['paid'][(instrument, side)] = (None, why) if why else (total, None)
    return flows['paid'][(instrument, side)]


def _worth(unit, paid):
    """A unit at a close less what it paid that day, or None where either is not known."""
    return None if unit is None or paid is None else unit - paid


def _holdings(row, movement, flows):
    """What each position held when the payment `row` fell due - before that day's trading - as
    the record stood when `movement` settled it: through the day it was filed where it was filed
    before it fell due."""
    _, day = _filed(movement['lsn'], flows)
    return (flows['days'](row['due_date'])['held'] if row['due_date'] <= day
            else flows['days'](day)['through'])


def _payments(key, flows, unknown):
    """What this position was paid in the window: the diary's amount times what it held before the
    day it fell due, for every payment the diary determines falling due in the window; and its
    share - by what each position held when the row fell due - of what every settlement FILED in
    the window moved against a row the diary leaves undetermined. Each at the close standing on its
    own day as the record stood at the end of the business day it counts in.

    An undetermined payment is the book's until something settles it: it leaves the book's value
    the day after it falls due, or on the position's last day where it is due then or after, and a
    window it leaves before anything settled it is UNKNOWN, named. A settlement over positions
    netting to nothing cannot be shared, a movement naming no agreement, and is named too."""
    total, last = 0.0, flows['last_days'].get(key[0])
    for row in flows['diary'].get(key[0], []):
        due = row['due_date']
        if row['determined']:
            if flows['start'] < due <= flows['end']:
                mine = flows['days'](due)['held'].get(key, 0.0)
                if mine:
                    total = _sum(total, _converted(mine * row['amount'], row['currency'], due,
                                                   _falls(due, flows), flows, key[0], unknown))
            continue
        leaves = (flows['start'] <= due < flows['end'] if last is None or due < last
                  else flows['start'] < due <= flows['end'])
        if (leaves and row['state'] != SETTLED
                and flows['days'](due)['held'].get(key, 0.0)):
            unknown.append({'instrument': key[0], 'what': 'the {} payment due {} is not '
                            'determined and nothing settled it'.format(row['leg'], due)})
            return None
        for movement in flows['settled'].get(('payment', row['key']), []):
            among = _carrying(row['key'], movement, flows)
            held = _holdings(row, movement, flows)
            mine, net = held.get(key, 0.0), sum(held.get(other, 0.0) for other in among)
            if not mine or key not in among:
                continue
            if not net:
                unknown.append({'instrument': key[0], 'what': 'settlement {} moved a payment of '
                                'positions netting to nothing - a settlement names no agreement, '
                                'so whose it is is not known'.format(movement['reference'])})
                return None
            total = _sum(total, _converted(
                movement['amount'] * mine / net, movement['asset'],
                movement['effective_time'][:10], _filed(movement['lsn'], flows)[0], flows,
                key[0], unknown))
    return total


def _booked(positions, movement):
    """The positions a movement is shared across: those under the book it was filed for, every
    one where it names none."""
    book = movement.get('book')
    return [key for key in positions if not book or under(key[2], book)]


def _carrying(key, movement, flows):
    """The positions a movement against the row `key` is shared across: those of every instrument
    whose diary carries the row - a restruck structure and the one it replaced carry an unchanged
    leg alike - under the book it was filed for."""
    return _booked([position for instrument, _ in flows['carried'].get(key, ())
                    for position in flows['held'].get(instrument, [])], movement)


def _owners(movement, flows):
    """`{position: weight}` - the positions a fee falls to, among those under the book it was filed
    for: those that traded what it was filed on, on its own day - its value date, or the day it was
    filed where it is dated after that - else those holding it through that day, else those that
    traded it on the record up to its filing, else those holding it when it was filed - terms an
    amendment struck being traded by no fill. Empty where none did."""
    found = (movement['reference'], movement['lsn'])
    if found not in flows['owners']:
        among = _booked(flows['held'].get(movement['subject'], []), movement)
        filed_on, weights = _filed(movement['lsn'], flows)[1], {}
        if among:
            record = flows['days'](min(movement['effective_time'][:10], filed_on))
            weights = {other: record['dealt'].get(other, 0.0) for other in among}
            if not any(weights.values()):
                weights = {other: abs(record['through'].get(other, 0.0)) for other in among}
            if not any(weights.values()):
                weights = {}
                for fill in flows['fills_between'](0, movement['lsn']):
                    if _where(fill) in among:
                        weights[_where(fill)] = weights.get(_where(fill), 0.0) + abs(
                            float(fill['quantity']))
            if not any(weights.values()):
                weights = {other: abs(flows['days'](filed_on)['through'].get(other, 0.0))
                           for other in among}
        flows['owners'][found] = {other: weight for other, weight in weights.items() if weight}
    return flows['owners'][found]


def _fees(key, flows, unknown):
    """This position's share of the fees FILED on its instrument in the window (`_owners`), at the
    close standing on each fee's value date as the record stood at the end of the business day it
    was filed in. Whose a fee is depends on its own day and its filing, never on the window it
    lands in."""
    total = 0.0
    for movement in flows['settled'].get(('fee', key[0]), []):
        weights = _owners(movement, flows)
        if weights.get(key):
            total = _sum(total, _converted(
                weights[key] / sum(weights.values()) * movement['amount'], movement['asset'],
                movement['effective_time'][:10], _filed(movement['lsn'], flows)[0], flows,
                key[0], unknown))
    return total


def _converted(amount, currency, day, lsn, flows, instrument, unknown):
    """`amount` of `currency` in the reporting currency at the close standing on `day` as filed by
    `lsn`, or UNKNOWN - named - where no close stands on it or it carries no spot for the currency.
    """
    rates = flows['rates'](day, lsn)
    if rates is not None and currency in rates:
        return amount * rates[currency]
    unknown.append({'instrument': instrument, 'what': (
        'no official close on or before {} to report what moved in {}'.format(day, currency)
        if rates is None else 'the close standing on {} carries no spot for {}'.format(
            day, currency))})
    return None


def _times(quantity, unit):
    """A quantity of units, or UNKNOWN where the unit is."""
    return None if unit is None else quantity * unit


def _sum(*parts):
    """A total that stays UNKNOWN once any part is."""
    return None if any(part is None for part in parts) else sum(parts)


def _less(held, less):
    """A difference that stays UNKNOWN once either side is."""
    return None if held is None or less is None else held - less
