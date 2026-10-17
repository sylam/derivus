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
"""A workstation's verified CDS quotes on one name as one `SurvivalProbPrices` block.

The family quotes a CONTRACT rather than a number: each row is the block's standard default swap
at the row's tenor, so the emitter's job is to say what kind of number the terminal quoted. A
default swap prints its conventional spread in basis points on `PX_LAST`/`PX_BID`/`PX_ASK`, against
the standard coupon `SW_SPREAD`, and `CDS_QUOTED_PRICE` the price per hundred of principal where it
quotes one. A row is an `Upfront` where the price answers - a hundred less the price, paid by the
buyer at the declared coupon, its spread sides being another unit and left off - and a
`Par_Spread` where the spread alone does, the conventional spread stated as a par spread being an
approximation the row's type says it is. `PTS_UPFRONT`, `CDS_PTS_UPFRONT_MID`,
`QUOTED_PRICE_TYPE`, `CDS_RECOVERY_RATE` and `SW_CURVE_QUOTE_TYPE` are not fields the terminal
carries, and nothing asks them.

THE BLOCK IS THE CURVE'S DEFINITION: the currency, the discount curve, the recovery, the coupon,
the frequency and the day count are seed-declared and emitted beside the rows, and a seed entry
carrying no `conventions` refuses by name with every missing field listed. Securities come from
`discover.credit_candidates` walked against the workstation's own verified map, so the ticker
grammar is spelled once in this package. A row names no date and its `Deal` carries no number, the
quote riding in `Quoted_Market_Value`, so a re-tick on any day passes `schema.update_market_quote`
as a values patch.

IMPORTS: the standard library and this package's own modules; nothing here imports `derivus`.
"""
import collections.abc
from dataclasses import dataclass, replace

from . import discover
from .errors import BloombergConfigurationError, IncompleteStrip
from .ir_curve import (BATCH, CurveScreen, CurveStrip, probe, read_date, read_number, read_prints,
                       screen_strip, wire_period, wire_timestamp)

#: What every contract is asked: its spread and the two-way, the price an upfront is read off, the
#: coupon both are quoted against, and the print's own date.
QUOTE_FIELDS = ('PX_LAST', 'PX_BID', 'PX_ASK', 'CDS_QUOTED_PRICE', 'SW_SPREAD', 'LAST_UPDATE_DT')

#: The family this emitter writes for.
FAMILY = 'SurvivalProbPrices'

#: The principal a CDS price is quoted per, so an upfront in percent is this less the price.
PAR = 100.0

#: The screen a spread print is held to: the band IN BASIS POINTS, which refuses a decimal-shifted
#: print, and a floor of one quote, which is a flat hazard rather than no curve.
SCREEN = CurveScreen(rate_band=(0.0, 1e4), minimum_points=1)

#: The conventions a seeded name must declare before its block can be authored, and the one that
#: carries a default: a blank discount curve is the currency's own.
REQUIRED_CONVENTIONS = ('currency', 'coupon', 'recovery_rate', 'pay_frequency', 'accrual_day_count')
OPTIONAL_CONVENTIONS = {'discount_rate': ''}


@dataclass(frozen=True)
class CreditConventions:
    """What a name's quotes are quoted against - read off the seed, never inferred from a ticker."""
    currency: str
    coupon: float
    recovery_rate: float
    pay_frequency: str
    accrual_day_count: str
    discount_rate: str = OPTIONAL_CONVENTIONS['discount_rate']


def credit_conventions(seed, name):
    """The declared conventions of one seeded name, or the refusal naming EVERY absent field."""
    spec = seed.get('credit', {}).get(name)
    if spec is None:
        raise BloombergConfigurationError(
            'the seed names no credit entry for {} - a name this workstation never seeded has no '
            'quotes to fetch. Add it to `seed.json` and re-run `DV_Bloomberg discover`'.format(name))
    declared = spec.get('conventions')
    if not isinstance(declared, collections.abc.Mapping):
        raise BloombergConfigurationError(
            '{} carries no `conventions` block - a CDS quote is not a contract until something says '
            'what it is quoted against. Declare {} on the {} entry in your seed'.format(
                name, ', '.join(REQUIRED_CONVENTIONS), name))
    missing = [field for field in REQUIRED_CONVENTIONS if declared.get(field) is None]
    if missing:
        raise BloombergConfigurationError(
            '{} declares no {} - the full set this emitter reads is {}. Fix the {} entry in your '
            'seed'.format(name, ', '.join(missing), ', '.join(REQUIRED_CONVENTIONS), name))
    return CreditConventions(**{field: declared[field] for field in REQUIRED_CONVENTIONS},
                             **{field: declared[field] for field in OPTIONAL_CONVENTIONS
                                if field in declared})


def credit_entries(document, seed, name):
    """`(wanted, ledger)` - the verified contracts of one name as `[(tenor, security)]` in the seed's
    tenor order, walked off the grammar, and `unverified` on the ledger for each the map lacks."""
    wanted, ledger = [], {}
    for candidate in discover.credit_candidates(name, seed['credit'][name]):
        entry = document.get('blocks', {})
        for part in candidate.path:
            entry = entry.get(part, {}) if isinstance(entry, collections.abc.Mapping) else {}
        if 'security' in entry:
            wanted.append((candidate.path[-1], entry['security']))
        else:
            ledger[candidate.security] = 'unverified'
    return wanted, ledger


def fetch_cds_strip(source, document, seed, name, as_of, screen=SCREEN, batch=BATCH,
                    on_batch=None):
    """One seeded name's verified quotes, screened - a `CurveStrip` whose prints are rows.

    ONE ROUND TRIP over the contracts the map believed. Each spread print is held to
    `ir_curve.screen_strip`'s order of distrust, and then one quoted against a coupon other than
    the declared one is `off-coupon`, an upfront being meaningless against another coupon. A
    believed print is an `Upfront` where the price answered and a `Par_Spread` otherwise.
    """
    conventions = credit_conventions(seed, name)
    wanted, ledger = credit_entries(document, seed, name)
    report = probe(source, [security for _, security in wanted], QUOTE_FIELDS, batch, on_batch)
    prints, rejected = read_prints(report, [(tenor, 'Par_Spread', security)
                                            for tenor, security in wanted])
    accepted, screened = screen_strip(prints, as_of, screen)
    rejected.update(ledger)
    rejected.update(screened)
    order = [tenor for tenor, _ in wanted]
    quoted = []
    for item in sorted(accepted, key=lambda item: order.index(item.label)):
        answered = report[item.security]['fields']
        coupon, price = (read_number(answered.get(field)) for field in ('SW_SPREAD',
                                                                        'CDS_QUOTED_PRICE'))
        if coupon is not None and coupon != conventions.coupon:
            rejected[item.security] = 'off-coupon'
        elif price is not None:
            quoted.append(replace(item, kind='Upfront', value=PAR - price, bid=None, ask=None))
        else:
            quoted.append(item)
    return CurveStrip(currency=conventions.currency, curve=name, as_of=as_of,
                      conventions=conventions, prints=tuple(quoted), rejected=rejected)


def author_row(item, name, as_of):
    """One `Points` row: the tenor the standard contract runs to, what kind of number is quoted,
    the number, the security it came off and the evidence beside it."""
    row = {'Use': item.use, 'Tenor': item.label, 'DealType': 'DealDefaultSwap',
           'Quote_Type': item.kind, 'Quoted_Market_Value': item.value, 'Security': item.security,
           'Descriptor': '{} {} ({})'.format(name, item.label, item.security), 'Deal': {}}
    if item.bid is not None:
        row['Quoted_Bid'] = item.bid
    if item.ask is not None:
        row['Quoted_Ask'] = item.ask
    row['Timestamp'] = wire_timestamp(read_date(item.last_update) or as_of)
    return row


def cds_block(strip, screen=SCREEN):
    """`(Market Prices name, block)` - one screened strip as ONE `SurvivalProbPrices` block, the
    conventions beside the rows in the seed's tenor order, which is the knot grid's."""
    used = [item for item in strip.prints if item.use == 'Yes']
    if len(used) < screen.minimum_points:
        raise IncompleteStrip(
            '{} screened to {} believed quote(s) against a floor of {} - the terminal refused {} '
            '({}). Re-run `DV_Bloomberg discover` if the name has gone dead'.format(
                strip.curve, len(used), screen.minimum_points, len(strip.rejected),
                ', '.join('{} {}'.format(count, verdict)
                          for verdict, count in sorted(strip.census.items())) or 'nothing'))
    conventions = strip.conventions
    return '{}.{}'.format(FAMILY, strip.curve), {'instrument': {
        'Currency': conventions.currency, 'Discount_Rate': conventions.discount_rate,
        'Recovery_Rate': conventions.recovery_rate, 'Standard_Coupon': conventions.coupon,
        'Pay_Frequency': wire_period(conventions.pay_frequency),
        'Accrual_Day_Count': conventions.accrual_day_count,
        'Points': [author_row(item, strip.curve, strip.as_of) for item in strip.prints]}}
