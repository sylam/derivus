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
"""A workstation's verified zero-coupon inflation swaps on one index as one `InflationRatePrices`
block, and the index's prints as the `PriceIndex` factor the block reads.

The family quotes a CONTRACT: each row is the block's standard swap at the row's whole-year tenor,
so the emitter says what number the terminal quoted - the breakeven in percent on `PX_LAST`, as it
prints - and the rule the swaps reference the index by. The currency, the discount curve, the day
count, the reference rule and the publication period are seed-declared and emitted beside the
rows, and a seed entry carrying no `conventions` refuses by name with every missing field listed.
A row names no date and its `Deal` carries no number, so a re-tick on any day passes
`schema.update_market_quote` as a values patch.

THE INDEX COMES OFF THE SAME TRIP, since the swaps reference prints the market must carry: its
monthly prints from the first month the base date's reference reads, `months_lag` back, to the
last print, the knots' origin, and the terminal's next release date. A swap's base reference is
the base date's and every later reference projects off the last print, so those are the months
the family reads.

FRESHNESS FIRST, the tick's rule: a swap print older than `stale_days`, or an index whose next
release is on or before the as-of, refuses the trip by name before the history is asked, a dead
series answering a plausible number. A swap `ir_curve.screen_strip` does not believe, its band in
percent, or one answering nothing is held out with `Use` No by name and carries no number.

IMPORTS: the standard library and this package's own modules; nothing here imports `derivus`.
"""
import collections.abc
import datetime
from dataclasses import dataclass

from . import discover
from .errors import BloombergConfigurationError, IncompleteStrip, InvalidQuote
from .ir_curve import (BATCH, QUOTE_FIELDS, CurveScreen, CurveStrip, RatePrint, add_months, probe,
                       read_date, read_number, read_prints, screen_strip, wire_timestamp)

#: The family this emitter writes for, and the one quote type it takes.
FAMILY = 'InflationRatePrices'
QUOTE_TYPE = 'Breakeven'

#: What the index is asked beside its history: the day its next print is released.
RELEASE_FIELDS = ('ECO_RELEASE_DT',)
HISTORY_FIELDS = ('PX_LAST',)

#: The screen a breakeven is held to: a rate print's band in percent, which refuses a
#: decimal-shifted print, and a floor of one quote, which is a flat growth curve.
SCREEN = CurveScreen(minimum_points=1)

#: `utils.excel_offset`, the day a `PriceIndex` counts its prints from, re-spelled because this
#: package imports no engine module.
EXCEL_ORIGIN = datetime.date(1899, 12, 30)

#: The conventions a seeded index must declare before its block can be authored, and the one that
#: carries a default: a blank discount curve is the currency's own.
REQUIRED_CONVENTIONS = ('currency', 'months_lag', 'reference_type', 'day_count', 'publication')
OPTIONAL_CONVENTIONS = {'discount_rate': ''}


@dataclass(frozen=True)
class InflationConventions:
    """What an index's swaps are quoted against, read off the seed, never inferred from a ticker."""
    currency: str
    months_lag: int
    reference_type: str
    day_count: str
    publication: str
    discount_rate: str = OPTIONAL_CONVENTIONS['discount_rate']


@dataclass(frozen=True)
class IndexPrints:
    """An index's monthly prints, `((month start, level), ...)` oldest first, the security they
    came off and the day the next one is released."""
    security: str
    prints: tuple
    next_release: datetime.date


def inflation_conventions(seed, name):
    """The declared conventions of one seeded index, or the refusal naming EVERY absent field."""
    spec = seed.get('inflation', {}).get(name)
    if spec is None:
        raise BloombergConfigurationError(
            'the seed names no inflation entry for {} - an index this workstation never seeded '
            'has no swaps to fetch. Add it to `seed.json` and re-run `DV_Bloomberg discover`'.format(
                name))
    declared = spec.get('conventions')
    if not isinstance(declared, collections.abc.Mapping):
        raise BloombergConfigurationError(
            '{} carries no `conventions` block - a breakeven is not a swap until something says '
            'how it references the index. Declare {} on the {} entry in your seed'.format(
                name, ', '.join(REQUIRED_CONVENTIONS), name))
    missing = [field for field in REQUIRED_CONVENTIONS if declared.get(field) is None]
    if missing:
        raise BloombergConfigurationError(
            '{} declares no {} - the full set this emitter reads is {}. Fix the {} entry in your '
            'seed'.format(name, ', '.join(missing), ', '.join(REQUIRED_CONVENTIONS), name))
    return InflationConventions(**{field: declared[field] for field in REQUIRED_CONVENTIONS},
                                **{field: declared[field] for field in OPTIONAL_CONVENTIONS
                                   if field in declared})


def inflation_entries(document, seed, name):
    """`(index, wanted, ledger)` - one index's verified print and its verified swaps as
    `[(tenor, security)]` in the seed's tenor order, walked off the grammar, `unverified` on the
    ledger for each swap the map lacks. An index the map lacks refuses by name: every swap
    references its prints."""
    index, wanted, ledger = None, [], {}
    for candidate in discover.inflation_candidates(name, seed['inflation'][name]):
        entry = document.get('blocks', {})
        for part in candidate.path:
            entry = entry.get(part, {}) if isinstance(entry, collections.abc.Mapping) else {}
        if 'security' not in entry:
            ledger[candidate.security] = 'unverified'
        elif candidate.path[2] == 'index':
            index = entry['security']
        else:
            wanted.append((candidate.path[-1], entry['security']))
    if index is None:
        raise BloombergConfigurationError(
            'the map carries no verified print for {} - its swaps reference the index, so there is '
            'no block without it. Re-run `DV_Bloomberg discover`'.format(name))
    return index, wanted, ledger


def fetch_inflation_strip(source, document, seed, name, as_of, screen=SCREEN, batch=BATCH,
                          on_batch=None):
    """`(strip, index)` - one seeded index's verified swaps, screened, as a `CurveStrip` whose
    prints are the rows in the seed's tenor order, and its prints as `IndexPrints`.

    ONE TRIP: the swaps asked their print, two-way and date, the index its next release and then
    its history from the first month the base date's reference reads to `as_of`. A late print
    refuses the trip by name before the history is asked; a swap the screen does not believe or
    that answered nothing is held out. A history missing a month refuses by name, the base
    reference reading every month it spans.
    """
    conventions = inflation_conventions(seed, name)
    index, wanted, ledger = inflation_entries(document, seed, name)
    report = probe(source, [security for _, security in wanted], QUOTE_FIELDS, batch, on_batch)
    released = read_date(probe(source, [index], RELEASE_FIELDS)[index]['fields'].get(
        'ECO_RELEASE_DT'))
    prints, rejected = read_prints(report, [(tenor, QUOTE_TYPE, security)
                                            for tenor, security in wanted])
    late = ['{} last printed {}'.format(item.security, item.last_update) for item in prints
            if read_date(item.last_update) is not None
            and (as_of - read_date(item.last_update)).days > screen.stale_days]
    if released is None or released <= as_of:
        late.append('{} next releases on {}, not after {}'.format(index, released, as_of))
    if late:
        raise InvalidQuote(
            '{} refused: {} - a dead series keeps answering a plausible number, so one late print '
            'stops the trip. Re-run `DV_Bloomberg discover` if a security has gone dead'.format(
                name, '; '.join(late)))

    start = add_months(as_of, -conventions.months_lag).replace(day=1)
    history = tuple((read_date(day).replace(day=1), read_number(values.get('PX_LAST')))
                    for day, values in source.historical_data(index, HISTORY_FIELDS, start, as_of))
    months = [add_months(start, step) for step in range(len(history))]
    if not history or [month for month, _ in history] != months or None in dict(history).values():
        raise IncompleteStrip(
            '{} answered {} where the base reference reads every month from {:%Y-%m} to the last '
            'print'.format(index, ', '.join('{:%Y-%m} {}'.format(*row) for row in history)
                           or 'no prints', start))

    accepted, screened = screen_strip(prints, as_of, screen)
    rejected.update(screened)
    believed = {item.security: item for item in accepted}
    rows = tuple(believed.get(security) or RatePrint(tenor, QUOTE_TYPE, security, None, use='No')
                 for tenor, security in wanted)
    rejected.update(ledger)
    return (CurveStrip(currency=conventions.currency, curve=name, as_of=as_of,
                       conventions=conventions, prints=rows, rejected=rejected),
            IndexPrints(index, history, released))


def author_row(item, name):
    """One `Points` row: the tenor the standard swap runs to, the breakeven, the security it came
    off and the evidence beside it, a held-out row carrying no number."""
    row = {'Use': item.use, 'Tenor': item.label, 'DealType': 'StructuredDeal',
           'Quote_Type': QUOTE_TYPE, 'Security': item.security,
           'Descriptor': '{} {} ({})'.format(name, item.label, item.security), 'Deal': {}}
    row.update({key: value for key, value in (('Quoted_Market_Value', item.value),
                                              ('Quoted_Bid', item.bid), ('Quoted_Ask', item.ask))
                if value is not None})
    if item.last_update:
        row['Timestamp'] = wire_timestamp(read_date(item.last_update))
    return row


def inflation_block(strip, index, screen=SCREEN):
    """`((Market Prices name, block), (Price Factors name, factor))` - one screened strip as ONE
    `InflationRatePrices` block, the conventions beside the rows in the seed's tenor order, and the
    index's prints as the `PriceIndex` factor its `Price_Index` names, dated in excel days."""
    used = [item for item in strip.prints if item.use == 'Yes']
    if len(used) < screen.minimum_points:
        raise IncompleteStrip(
            '{} screened to {} believed quote(s) against a floor of {} - the terminal refused {} '
            '({}). Re-run `DV_Bloomberg discover` if the swaps have gone dead'.format(
                strip.curve, len(used), screen.minimum_points, len(strip.rejected),
                ', '.join('{} {}'.format(count, verdict)
                          for verdict, count in sorted(strip.census.items())) or 'nothing'))
    conventions = strip.conventions
    block = {'instrument': {
        'Currency': conventions.currency, 'Discount_Rate': conventions.discount_rate,
        'Day_Count': conventions.day_count, 'Price_Index': strip.curve,
        'Months_Lag': conventions.months_lag, 'Reference_Type': conventions.reference_type,
        'Points': [author_row(item, strip.curve) for item in strip.prints]}}
    factor = {
        'Index': {'.Curve': {'meta': [], 'data': [[float((month - EXCEL_ORIGIN).days), level]
                                                  for month, level in index.prints]}},
        'Last_Period_Start': wire_timestamp(index.prints[-1][0]),
        'Next_Publication_Date': wire_timestamp(index.next_release),
        'Publication_Period': conventions.publication, 'Currency': conventions.currency}
    return ('{}.{}'.format(FAMILY, strip.curve), block), ('PriceIndex.' + strip.curve, factor)
