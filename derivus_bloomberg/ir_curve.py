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
"""A workstation's verified swap strip as one `InterestRatePrices` block.

The family quotes an instrument rather than a number: each `Points` row carries a `Deal` block in
the instrument's own conventions and the solve holds it at PV zero, so the emitter's job is to say
what the instrument IS, from declared data.

THE BLOCK IS THE CURVE'S DEFINITION, so the conventions the rows were authored under are emitted
beside them and each row carries the `Tenor` it came from. They are seed-declared - `USOSFR10` is
annual/annual ACT/360 compounded overnight, `SASW10` quarterly/quarterly ACT/365 against 3M JIBAR,
and the same number under the other convention is a different curve by basis points. A seed entry
carrying no `conventions` block refuses by name with the missing fields listed. `front` names which
verified entry seeds the short end (a SOFR OIS curve's is the overnight print, a JIBAR-3M curve's
the 3M JIBAR fixing) and the remaining seeded fixings are ledgered `not-a-benchmark`. Securities
come from `discover.strip_candidates` walked against the workstation's own verified map, so the
ticker grammar is spelled once in this package and a strip the terminal never verified cannot enter
a block on a seed's say-so.

The quote is not authored into the deal: `QUOTE_WRITERS` is where a number lands, off
`Quoted_Market_Value`, and every rate-carrying field is authored at a neutral zero, so a value-only
re-tick passes `schema.update_market_quote` as 'updated' rather than refusing as a moved plan.

The four shapes a row's `Tenor` names: `3M` at the front is a deposit, `1Mx4M` a FRA, `3Y` a spot
swap and `6M1M` a swap starting in six months. An OIS strip is the same swap with `Compounding` OIS
and ONE reset spanning each coupon - at t0 that prices the par swap the daily fixing list does, to
the ninth decimal, and a two-year benchmark is 1.4 KB rather than the 285 KB its fixings encode to.

A DESK STATES THE SAME THING. `author_block` authors a block off DECLARED rows - a tenor, a
security and a number - reading each row's shape off its own tenor rather than off a map path, and
`block_conventions` reads a block back as the definition it is, which is what re-rolls a strip on a
later date without the seed that started it.

SCOPE: `Quote_Type` `Par_Rate`, discounting on the curve itself or on one the definition names. No
FX-forward outrights, no cross-currency, no projection curve.

IMPORTS: the standard library and this package's own modules. `discover` is reached for its grammar
and carries pandas, so unlike `equity_chain` this module makes no pandas-free claim; nothing here
imports `derivus`, the block being emitted as wire JSON.
"""
import collections.abc
import datetime
import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Protocol, Sequence

from . import discover
from .errors import BloombergConfigurationError, IncompleteStrip, InvalidQuote

#: What every strip candidate is asked: the value, both sides of the two-way, and the EVIDENCE that
#: any of it still means anything. The date is asked again at fetch time because a map records when
#: a quote was VERIFIED and a tick needs when it last PRINTED.
QUOTE_FIELDS = ('PX_LAST', 'PX_BID', 'PX_ASK', 'LAST_UPDATE_DT')

#: One request per chunk - `discover.BATCH`, for its reason.
BATCH = 50

#: The family this emitter writes for, and the `Quote_Type` it declares. `Par_Rate` is the single
#: convention `InterestRateCurveParameters` builds: the solve holds every benchmark at PV zero.
FAMILY = 'InterestRatePrices'
QUOTE_TYPE = 'Par_Rate'

#: What a swap row IS - the `SwapInterestDeal` field of the same name. `OIS` marks a benchmark
#: against an overnight index; the coupon carries one reset spanning itself either way.
COMPOUNDING = ('None', 'OIS')

#: The day counts a seed may declare - ACT/365 and ACT/360 and no more, because the engine's own
#: schedule generation answers ACT_365_ISDA and ACT_ACT_ICMA as days/365 behind a TODO and the two
#: 30/360 conventions need date arithmetic this module does not author.
DAY_COUNTS = ('ACT_365', 'ACT_360')

#: `riskfactors.INTERPOLATION_METHODS`, re-spelled because this package imports no engine module.
#: A gate reads the two declarations and requires them equal, so neither drifts alone.
INTERPOLATIONS = ('HermiteRT', 'Hermite', 'LinearRT', 'Linear')

#: The convention fields a seeded curve must declare before its strip can be authored, and the
#: ones that carry a default. A missing one is a NAMED refusal listing all of them at once, because
#: a desk filling in a seed wants the whole list rather than one field per run.
REQUIRED_CONVENTIONS = ('curve_day_count', 'spot_days', 'front', 'front_day_count', 'compounding',
                        'fixed_frequency', 'float_frequency', 'fixed_day_count', 'float_day_count')
OPTIONAL_CONVENTIONS = {'notional': 1000000.0, 'quote_scale': 1.0, 'calendar': '',
                        'near_interpolation': '', 'near_tenor': ''}


class ReferenceDataSource(Protocol):
    """The tolerant reader alone - one request carries the value, both sides of the two-way and the
    print's own date, and the strict policy is applied HERE, per print, by the screen."""

    def reference_data_report(self, securities: Sequence[str],
                              fields: Sequence[str]) -> Mapping[str, Mapping[str, object]]:
        ...


# ---------------------------------------------------------------------------------------------
# declared data
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CurveConventions:
    """What a curve's strip IS - read off the seed, never inferred from a ticker.

    Every field here is a market convention somebody owns. They are validated on construction (a
    day count this module does not write, a compounding rule no swap carries, half a near split, a
    negative settlement lag) so a bad declaration refuses at the seed rather than inside a cashflow.
    """
    curve_day_count: str
    spot_days: int
    front: str
    front_day_count: str
    compounding: str
    fixed_frequency: str
    float_frequency: str
    fixed_day_count: str
    float_day_count: str
    notional: float = OPTIONAL_CONVENTIONS['notional']
    #: What multiplies the terminal's print to reach PERCENT, the unit a rate benchmark is quoted
    #: in here: `DepositDeal` divides its pinned schedule by 100, `SwapInterestDeal` divides
    #: `Swap_Rate`, and a fixed leg's `Rate` is a `Percent`. Both seeded strips print percent
    #: already and declare 1.0; a family printing decimals is then a seed edit.
    quote_scale: float = OPTIONAL_CONVENTIONS['quote_scale']
    #: The holiday calendar the dates are rolled against, named in the job's calendar file. The
    #: DATES come from an iterable the caller hands in; this is what the block declares it used.
    calendar: str = OPTIONAL_CONVENTIONS['calendar']
    #: Where the near end is quoted in another instrument, the scheme it carries up to `near_tenor`
    #: - a ZARONIA curve is monthly MPC-dated to 18M and ordinary swaps beyond.
    near_interpolation: str = OPTIONAL_CONVENTIONS['near_interpolation']
    near_tenor: str = OPTIONAL_CONVENTIONS['near_tenor']

    def __post_init__(self):
        if self.compounding not in COMPOUNDING:
            raise BloombergConfigurationError(
                'compounding {!r} is not what a swap row can be - it is {}. Fix the curve\'s '
                '`conventions` in the seed'.format(self.compounding, ' or '.join(
                    repr(value) for value in COMPOUNDING)))
        for name in ('curve_day_count', 'front_day_count', 'fixed_day_count', 'float_day_count'):
            _check_day_count(getattr(self, name), name)
        for name in ('fixed_frequency', 'float_frequency'):
            # POSITIVE, not merely readable: a zero-length coupon period is a schedule the engine's
            # own generation never advances along, and a hang carries no message
            if read_tenor(getattr(self, name), name)[0] <= 0:
                raise BloombergConfigurationError(
                    '{} is {!r} - a leg frequency has to be a positive period, or the coupon '
                    'schedule never advances'.format(name, getattr(self, name)))
        if self.near_interpolation and self.near_interpolation not in INTERPOLATIONS:
            raise BloombergConfigurationError(
                'near_interpolation {!r} is not a scheme the engine interpolates a curve with - it '
                'is one of {}, or blank for one scheme over the whole curve'.format(
                    self.near_interpolation, ', '.join(INTERPOLATIONS)))
        if bool(self.near_interpolation) != bool(self.near_tenor):
            raise BloombergConfigurationError(
                'near_interpolation is {!r} against a near_tenor of {!r} - a near scheme is a '
                'scheme AND where it stops, and half of it would be read nowhere. Declare both, or '
                'neither'.format(self.near_interpolation, self.near_tenor))
        if self.near_tenor:
            read_tenor(self.near_tenor, 'near_tenor')
        if not isinstance(self.spot_days, int) or self.spot_days < 0:
            raise BloombergConfigurationError(
                'spot_days must be a whole number of business days at or above zero, not {!r} - it '
                'is the settlement lag the strip\'s Effective_Date is placed at'.format(
                    self.spot_days))
        if not math.isfinite(self.notional) or self.notional <= 0.0:
            raise BloombergConfigurationError('notional must be positive and finite')
        if not math.isfinite(self.quote_scale) or self.quote_scale == 0.0:
            raise BloombergConfigurationError('quote_scale must be finite and non-zero')


def curve_conventions(seed: Mapping, curve: str, stated: Mapping = ()) -> CurveConventions:
    """The declared conventions of one seeded curve, or the refusal naming EVERY absent field at
    once - a desk extending a seed wants the whole questionnaire, not one field per run.

    `stated` is what a REQUEST says, and it wins over the seed field by field: a desk setting a
    curve up completes its entry rather than editing the file, and one that states the whole
    questionnaire needs no entry at all.
    """
    spec = seed.get('rates', {}).get(curve)
    if spec is None and not stated:
        raise BloombergConfigurationError(
            'the seed names no rates entry for {} - a curve this workstation never seeded has no '
            'strip to fetch and no conventions to author one in. Add it to `seed.json` and re-run '
            '`DV_Bloomberg discover`'.format(curve))
    declared = (spec or {}).get('conventions')
    if stated:
        declared = dict(declared if isinstance(declared, collections.abc.Mapping) else {}, **stated)
    if not isinstance(declared, collections.abc.Mapping):
        raise BloombergConfigurationError(
            '{} carries no `conventions` block - a par swap rate is not an instrument until '
            'something says what it accrues on, and this emitter reads that rather than guessing '
            'it. Declare {} on the {} entry in your seed (see derivus_bloomberg/seed.json for the '
            'shipped USD and ZAR declarations)'.format(
                curve, ', '.join(REQUIRED_CONVENTIONS), curve))
    missing = [name for name in REQUIRED_CONVENTIONS if declared.get(name) is None]
    if missing:
        raise BloombergConfigurationError(
            '{} declares no {} - the full set this emitter reads is {}, and a convention block '
            'filled in half way authors an instrument nobody stated. Fix the {} entry in your '
            'seed'.format(curve, ', '.join(missing), ', '.join(REQUIRED_CONVENTIONS), curve))
    unknown = sorted(set(declared) - set(REQUIRED_CONVENTIONS) - set(OPTIONAL_CONVENTIONS))
    if unknown:
        raise BloombergConfigurationError(
            '{} declares {} which this emitter reads nothing of - a convention nobody reads is a '
            'convention that is not applied, which is worse than one that is missing. Remove it, or '
            'spell it as one of {}'.format(
                curve, ', '.join(unknown),
                ', '.join(sorted(set(REQUIRED_CONVENTIONS) | set(OPTIONAL_CONVENTIONS)))))
    # `front` is a PATH INTO THE SEED, so it is checked HERE rather than in `__post_init__`, which
    # cannot see one: a path that names nothing does not fail, it aims elsewhere. `front:
    # 'strip/1Y'` would author the 1Y par swap as a one-day deposit labelled overnight.
    admissible = _seeded_fronts(seed, curve)
    if spec is not None and declared['front'] not in admissible:
        raise BloombergConfigurationError(
            '{} declares its front as {!r}, which is not an entry its seed could name - the '
            'admissible spellings are {}. The front is what seeds the short end, and a `front` '
            'aimed at the swap strip would author a par swap as a one-day overnight deposit and '
            'name it `overnight` in the Descriptor rather than refuse. Fix the {} entry in your '
            'seed'.format(curve, declared['front'],
                          ', '.join(admissible) or 'none: the entry seeds neither an `overnight` '
                          'print nor any `fixings`, so it can carry no front at all', curve))
    return CurveConventions(**{name: declared[name] for name in REQUIRED_CONVENTIONS},
                            **{name: declared[name] for name in OPTIONAL_CONVENTIONS
                               if name in declared})


@dataclass(frozen=True)
class CurveScreen:
    """The screens a print is held to, every one of them a parameter with a stated default."""
    #: `discover.STALE_DAYS`, for its reason: a retired benchmark keeps answering a plausible price
    #: and the update date is the only thing that says so (SAONIA read 8.855 nineteen years on).
    stale_days: int = discover.STALE_DAYS
    #: The band a par rate is believed inside, IN PERCENT. Not a market view - a 50% band admits
    #: every currency anyone has quoted a swap in and refuses a decimal-shifted print, which is the
    #: failure this catches: the same feed that answers 8.855 can answer 8855 with no error.
    rate_band: tuple = (-5.0, 50.0)
    #: How many believed prints a block needs. Two is the floor a CURVE means anything at: one knot
    #: is a flat curve quoted once, and the family's own knot rule puts one knot per used quote.
    minimum_points: int = 2

    def __post_init__(self):
        object.__setattr__(self, 'rate_band', tuple(self.rate_band))
        if len(self.rate_band) != 2 or not all(math.isfinite(edge) for edge in self.rate_band) \
                or not self.rate_band[0] < self.rate_band[1]:
            raise BloombergConfigurationError('rate_band must be (low, high) with low < high')
        if self.minimum_points < 1:
            raise BloombergConfigurationError('minimum_points must be at least one')


# ---------------------------------------------------------------------------------------------
# what a strip is
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RatePrint:
    """One quoted point of a strip as the terminal answered for it - raw, unjudged, not believed.

    It is also the row a DESK declares, which is the same thing stated instead of fetched: `use` is
    the family's own hold-out flag and `verdict` is why the screen did not believe the print, so a
    dead ticker leaves the row standing and named rather than taking the curve with it.
    """
    label: str
    kind: str
    security: str
    value: float | None
    bid: float | None = None
    ask: float | None = None
    last_update: str | None = None
    use: str = 'Yes'
    verdict: str = ''


@dataclass(frozen=True)
class CurveStrip:
    """A screened strip: what survived, what did not and why, and the curve it is quoted for.

    `rejected` is the LEDGER - `{security: verdict}` for every candidate that did not make it. On a
    strip, a point silently dropped is the difference between a short curve and a wrong one.
    """
    currency: str
    curve: str
    as_of: datetime.date
    conventions: CurveConventions
    prints: tuple
    rejected: Mapping[str, str] = field(default_factory=dict)
    #: What the quotes DISCOUNT on. Blank is the self-discounting curve a fetched strip builds; a
    #: definition naming another curve makes the two a multi-curve set, which `Discount_Rate` orders.
    discount_rate: str = ''

    @property
    def census(self) -> dict:
        census = {}
        for verdict in self.rejected.values():
            census[verdict] = census.get(verdict, 0) + 1
        return census


# ---------------------------------------------------------------------------------------------
# reading the terminal
# ---------------------------------------------------------------------------------------------

def read_number(value):
    """A terminal value as a finite float, or None. Absent, blank, unparseable and non-finite all
    read as ABSENT rather than as a number manufactured out of a blank."""
    if value is None or value == '':
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_date(value):
    """A terminal date as a `datetime.date`, or None. Bloomberg answers dates as `datetime.date`
    through blpapi and as ISO strings through every canned fixture, so both are read."""
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if value in (None, ''):
        return None
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def read_word(value):
    return '' if value in (None, '') else str(value).strip()


def probe(source, securities, fields=QUOTE_FIELDS, batch=BATCH, on_batch=None):
    """Every candidate asked in bounded chunks - `discover.probe`'s contract. `on_batch(done,
    total)` counts names REPLIED ABOUT, so a caller can watch a slow terminal."""
    report = {}
    for start in range(0, len(securities), batch):
        report.update(source.reference_data_report(securities[start:start + batch], list(fields)))
        if on_batch is not None:
            on_batch(min(start + batch, len(securities)), len(securities))
    return report


#: What a candidate's own map path says the row IS - the grammar's third segment, and the shape
#: `author_point` writes. `fixings` is the only one that is not a benchmark on its own.
KINDS = {'strip': 'swap', 'fra': 'fra', 'forward': 'forward'}


def strip_entries(document, seed, curve):
    """`(wanted, ledger)` - the verified securities of one curve's strip, walked off the GRAMMAR.

    `discover.strip_candidates` supplies the candidates and each one's `path` is looked up in the
    workstation's own map, so this module spells no ticker and no map path. A candidate the map did
    not verify is `unverified` on the ledger by name; a seeded fixing that is not the declared
    front point is `not-a-benchmark`, since a 6M JIBAR print is an index rather than an instrument
    this block holds at par.

    `wanted` is `[(label, kind, security)]` with `kind` one of `front` / `swap` / `fra` /
    `forward`, in the grammar's own order - the emitter sorts by maturity later, that being a
    property of the calendar.
    """
    conventions = curve_conventions(seed, curve)
    blocks = document.get('blocks', {})
    front_path = ('rates', curve) + tuple(part for part in conventions.front.split('/') if part)
    wanted, ledger, found_front = [], {}, False
    for candidate in discover.strip_candidates(curve, seed['rates'][curve]):
        entry = blocks
        for part in candidate.path:
            entry = entry.get(part) if isinstance(entry, collections.abc.Mapping) else None
            if entry is None:
                break
        security = entry.get('security') if isinstance(entry, collections.abc.Mapping) else None
        if security is None:
            ledger[candidate.security] = 'unverified'
            continue
        if candidate.path == front_path:
            wanted.append((front_tenor(conventions.front), 'front', security))
            found_front = True
        elif candidate.path[2] in KINDS:
            wanted.append((candidate.path[-1], KINDS[candidate.path[2]], security))
        else:
            ledger[security] = 'not-a-benchmark'
    if not found_front:
        raise BloombergConfigurationError(
            '{} declares its front point as {!r} and the map carries no verified entry there - the '
            'front is what seeds the short end of the curve, and a strip quoted from its first swap '
            'alone leaves everything under {} unidentified. Re-run `DV_Bloomberg discover` (the '
            'entry may have gone dead), or declare a `front` the map verified: the seeded ones are '
            '{}'.format(curve, conventions.front, wanted[0][0] if wanted else 'its first knot',
                        ', '.join(_seeded_fronts(seed, curve)) or 'none'))
    return tuple(wanted), ledger


def _seeded_fronts(seed, curve):
    """The `front` spellings a curve's seed entry could name - what a refusal offers."""
    spec = seed.get('rates', {}).get(curve, {})
    return (['overnight'] if spec.get('overnight') else []) + [
        'fixings/{}'.format(label) for label in sorted(spec.get('fixings', {}))]


def front_tenor(front):
    """The front point's TENOR as a label. A named fixing carries its own (`fixings/3M` is a 3M
    deposit); an overnight print has none to carry, so `ON` is the label and the deposit's span is
    worked out from the calendar at authoring time."""
    label, _, tenor = read_word(front).partition('/')
    return tenor if label == 'fixings' else 'ON'


def row_kind(tenor, front='ON'):
    """What a DECLARED row is, read off its own tenor - the grammar `strip_entries` reads off a map
    path instead, and the one thing a desk never has to state.

    The declared front and `ON` are the deposit; an `x` is a FRA (`1Mx4M`); two tenors run together
    are a forward-starting swap (`6M1M`, the terminal's own spelling); everything else is a spot
    swap. A label neither this nor `read_tenor` can read refuses when the dates are rolled.
    """
    text = read_word(tenor).upper()
    if text in (read_word(front).upper(), 'ON'):
        return 'front'
    if 'X' in text:
        return 'fra'
    return 'forward' if sum(letter in 'DWMY' for letter in text) > 1 else 'swap'


def fetch_curve_strip(source, document, seed, key, as_of, curve=None, screen=None,
                      batch=BATCH, on_batch=None):
    """One seeded curve's verified strip, screened - a `CurveStrip`.

    ONE ROUND TRIP over the securities the map believed, asking each the value, both sides of its
    two-way and its own last print. The tolerant reader makes the request and the strict policy is
    applied CLIENT-SIDE, per print: one dead point is a curve with one fewer knot, where a strip
    refused whole is no curve at all.

    THE SEED IS KEYED BY CURVE and the entry declares the `currency` it is quoted in, an entry
    without one being its own currency: `USD` is the USD curve, `ZAR-ZARONIA` a second ZAR one.
    `curve` names the `InterestRate` factor this strip builds and defaults to that key, so the block
    key is `InterestRatePrices.<curve>` and the deals project off it.
    """
    screen = screen or CurveScreen()
    conventions = curve_conventions(seed, key)
    currency = seed['rates'][key].get('currency', key)
    wanted, ledger = strip_entries(document, seed, key)
    report = probe(source, [security for _, _, security in wanted], batch=batch, on_batch=on_batch)

    prints, rejected = read_prints(report, wanted, conventions.quote_scale)
    rejected.update(ledger)
    accepted, screened = screen_strip(prints, as_of, screen)
    rejected.update(screened)
    return CurveStrip(currency=currency, curve=curve or key, as_of=as_of,
                      conventions=conventions, prints=accepted, rejected=rejected)


def _scaled(value, quote_scale):
    number = read_number(value)
    return None if number is None else number * quote_scale


def read_prints(report, wanted, quote_scale=1.0):
    """`(prints, rejected)` - one `RatePrint` per `(label, kind, security)` the terminal answered
    about, and `invalid` on the ledger for the ones it answered nothing for."""
    prints, rejected = [], {}
    for label, kind, security in wanted:
        answered = (report.get(security) or {}).get('fields') or {}
        if not answered:
            rejected[security] = 'invalid'
            continue
        prints.append(RatePrint(
            label=label, kind=kind, security=security,
            value=_scaled(answered.get('PX_LAST'), quote_scale),
            bid=_scaled(answered.get('PX_BID'), quote_scale),
            ask=_scaled(answered.get('PX_ASK'), quote_scale),
            last_update=read_word(answered.get('LAST_UPDATE_DT')) or None))
    return prints, rejected


def price_rows(source, rows, as_of, conventions, screen=None, batch=BATCH, on_batch=None):
    """DECLARED rows re-priced off the terminal - one batched round trip over the securities the
    used rows name, screened per print the way a strip is.

    A believed print moves the row's mid, both sides and its stamp; one the screen refuses leaves
    the row's number where it stood and puts the VERDICT on it, so a caller either holds that row
    out by name (a tick) or refuses naming it (an authoring). A row with no security keeps what it
    carries - a benchmark quoted by hand is nothing the terminal knows about.
    """
    wanted = [row for row in rows if _asked(row)]
    report = probe(source, [row.security for row in wanted], batch=batch, on_batch=on_batch)
    prints, rejected = read_prints(report, [(row.label, row.kind, row.security) for row in wanted],
                                   conventions.quote_scale)
    accepted, screened = screen_strip(prints, as_of, screen)
    rejected.update(screened)
    priced = {item.security: item for item in accepted}
    moved = []
    for row in rows:
        item = priced.get(row.security)
        if not _asked(row):
            moved.append(row)
        elif item is not None:
            moved.append(replace(row, value=item.value, bid=item.bid, ask=item.ask,
                                 last_update=item.last_update))
        else:
            moved.append(replace(row, verdict=rejected.get(row.security, 'invalid')))
    return tuple(moved)


def _asked(row):
    """A row the terminal is asked about: one the block uses, quoted off a security it names."""
    return row.use == 'Yes' and bool(row.security)


def hold_out(rows):
    """Rows the screen did not believe, marked `Use: No`. One dead ticker is one knot fewer, never
    a curve refused; the verb is what puts a held-out row back."""
    return tuple(replace(row, use='No') if row.verdict else row for row in rows)


def screen_strip(prints, as_of, screen=None):
    """`(accepted, rejected)` - the trust boundary, in the ORDER OF DISTRUST.

    `discover.verify`'s own shape, read for a rate rather than for a candidate: what the print IS
    before how well it is known, and how well it is known before whether it still means anything.

      unpriced   no PX_LAST at all - a knot cannot be identified by a blank
      off-market a rate outside the declared band: the feed that answers 8.855 answers 8855 too,
                 and a decimal-shifted print passes every check that is not a band
      crossed    bid above ask - a stale side left standing against a live one, which takes the
                 mid's credibility with it
      undated    no readable LAST_UPDATE_DT: a print that cannot evidence its own time, and the
                 row's `Timestamp` would otherwise carry it into the block
      stale      a last update older than `stale_days`
      live       believed

    `rejected` is `{security: verdict}` for every one of them, BY NAME.
    """
    screen = screen or CurveScreen()
    accepted, rejected = [], {}
    for item in sorted(prints, key=lambda print_: print_.security):
        verdict = _verdict(item, as_of, screen)
        if verdict == 'live':
            accepted.append(item)
        else:
            rejected[item.security] = verdict
    return tuple(accepted), rejected


def _verdict(item, as_of, screen):
    if item.value is None:
        return 'unpriced'
    low, high = screen.rate_band
    if not low <= item.value <= high:
        return 'off-market'
    if item.bid is not None and item.ask is not None and item.bid > item.ask:
        return 'crossed'
    stamp = read_date(item.last_update)
    if stamp is None:
        return 'undated'
    if (as_of - stamp).days > screen.stale_days:
        return 'stale'
    return 'live'


# ---------------------------------------------------------------------------------------------
# the calendar, in the standard library
# ---------------------------------------------------------------------------------------------

def read_tenor(label, what='tenor'):
    """`(count, unit)` off a strip label - `1W`, `3M`, `10Y`, `2D`, the vocabulary
    `discover.strip_candidates` labels a strip with. An unreadable label REFUSES: it is a seed
    nobody can author from rather than something to skip."""
    text = read_word(label).upper()
    if len(text) < 2 or text[-1] not in 'DWMY' or not text[:-1].lstrip('+').isdigit():
        raise BloombergConfigurationError(
            '{!r} is not a tenor this emitter can read as a {} - it spells `<count><D|W|M|Y>`, '
            'which is the vocabulary `discover.strip_candidates` labels a strip with'.format(
                label, what))
    return int(text[:-1]), text[-1]


def _add_months(date, months):
    """A date moved whole months, CLAMPED to the end of the shorter month - `pd.DateOffset`'s own
    rule, re-spelled here because this package constructs no pandas offset. 31 January plus a month
    is 28 February, and 29 February plus a year is 28 February."""
    total = (date.year * 12 + date.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    day = min(date.day, _days_in_month(year, month))
    return datetime.date(year, month, day)


def _days_in_month(year, month):
    if month == 12:
        return 31
    return (datetime.date(year + (month // 12), (month % 12) + 1, 1) -
            datetime.date(year, month, 1)).days


def _add_tenor(date, label):
    count, unit = read_tenor(label)
    if unit == 'D':
        return date + datetime.timedelta(days=count)
    if unit == 'W':
        return date + datetime.timedelta(weeks=count)
    return _add_months(date, count * (12 if unit == 'Y' else 1))


def split_tenors(label, what='tenor pair'):
    """`1Mx4M` and `6M1M` as their two tenors - a FRA's two dates, and a forward-starting swap's
    start and term. The `x` separates where it is written; without one the first unit letter does,
    which is how the terminal spells a forward start."""
    text = read_word(label).upper()
    left, _, right = text.partition('X')
    if not right:
        cut = next((i for i, letter in enumerate(text) if letter in 'DWMY'), len(text) - 1)
        left, right = text[:cut + 1], text[cut + 1:]
    read_tenor(left, what), read_tenor(right, what)
    return left, right


def roll(date, holidays=(), modified=False):
    """A date moved onto a business day - Following, or Modified Following where the roll would
    leave the month.

    THE ONE CALENDAR RULE. A business day is Monday to Friday and not in `holidays`, an iterable of
    dates the caller reads out of the job's calendar file, so no holidays handed in is the weekday
    rule this module has always applied. Deposits roll Following, a swap's two dates and a FRA's
    Modified Following, the market's convention; the coupon dates the engine generates between a
    swap's two dates are not rolled.
    """
    moved = date
    while moved.weekday() > 4 or moved in holidays:
        moved += datetime.timedelta(days=1)
    if modified and moved.month != date.month:
        moved = date
        while moved.weekday() > 4 or moved in holidays:
            moved -= datetime.timedelta(days=1)
    return moved


def _next_business_day(date, holidays=()):
    return roll(date + datetime.timedelta(days=1), holidays)


def _add_business_days(date, count, holidays=()):
    moved = roll(date, holidays)
    for _ in range(count):
        moved = _next_business_day(moved, holidays)
    return moved


def _check_day_count(day_count, what='day count'):
    if day_count not in DAY_COUNTS:
        raise BloombergConfigurationError(
            '{!r} is not a {} a seed may declare here - it writes {}. The engine answers '
            'ACT_365_ISDA and ACT_ACT_ICMA as days/365 behind a TODO and the 30/360 conventions '
            'need date arithmetic this module does not author, so they are refused rather than '
            'declared on trust'.format(day_count, what, ' or '.join(sorted(DAY_COUNTS))))


# ---------------------------------------------------------------------------------------------
# the wire spellings
# ---------------------------------------------------------------------------------------------

def wire_timestamp(date):
    """A date in the WIRE spelling the engine's own decoder reads - `{'.Timestamp': 'YYYY-MM-DD'}`.

    The wire form rather than a `pandas.Timestamp`: a block is posted as JSON, and spelling it here
    keeps this module free of the engine. It is what `CustomJsonEncoder` writes too, so a block
    emitted here and a block read back off disk are the same bytes.
    """
    if date is None:
        raise InvalidQuote(
            'a date this block carries cannot be absent - every emitted point has been screened '
            'for a readable print date, so reaching here is a bug rather than a market')
    return {'.Timestamp': date.isoformat()}


def wire_period(label):
    """A tenor in the wire spelling both decoders parse - `{'.DateOffset': '3M'}`, the string form
    `CustomJsonEncoder` writes. `Config.parse_json` also accepts a kwargs dict under this key, for
    bytes already on disk; nothing writes that form."""
    read_tenor(label)
    return {'.DateOffset': label}


def read_period(period):
    """A wire tenor read back as its label - `{'.DateOffset': '3M'}` is `3M`, and a blank stays
    blank. What lets a block declare a frequency AND be re-authored from what it declares."""
    return period.get('.DateOffset', '') if isinstance(period, dict) else read_word(period)


def wire_percent(value):
    """`utils.Percent`'s wire form. The number is in PERCENT - `{'.Percent': 4.28}` is 4.28%, and
    the decoded object's `.amount` is 0.0428."""
    return {'.Percent': value}


def wire_date_list(pairs):
    return {'.DateList': [[date.isoformat(), value] for date, value in pairs]}


# ---------------------------------------------------------------------------------------------
# authoring an instrument
# ---------------------------------------------------------------------------------------------

def _deposit(reference, currency, curve, effective, maturity, tenor, day_count, notional,
             calendar):
    """A money-market deposit - the strip's FRONT point.

    The rate is pinned through `Interest_Rate_Schedule`, which keeps a front quote off the forecast
    curve entirely: `DepositDeal.reset` drops the `Interest_Rate` dependency when the schedule
    covers every accrual start, so the point cannot depend on the curve it identifies. The schedule
    is authored EMPTY because `QUOTE_WRITERS['DepositDeal']` writes it from the quote - an authored
    one would put a rate in the block's plan half and make every re-tick a re-authoring.
    """
    return {
        'Object': 'DepositDeal', 'Reference': reference, 'Currency': currency,
        'Interest_Rate': curve,
        'Effective_Date': wire_timestamp(effective), 'Maturity_Date': wire_timestamp(maturity),
        'Payment_Frequency': wire_period(tenor), 'Interest_Frequency': wire_period(tenor),
        'Accrual_Day_Count': day_count, 'Amount': notional, 'Amortisation': None,
        'Compounding': 'No', 'Payment_Timing': 'End', 'Payment_Offset': 0,
        'Accrual_Calendars': calendar, 'Payment_Calendars': calendar,
        'First_Coupon_Date': None, 'Penultimate_Coupon_Date': None,
        'Rate_Currency': '', 'FX_Reset_Offset': 0, 'Known_FX_Rates': None,
        'Interest_Rate_Schedule': wire_date_list(())}


def _fra(reference, currency, curve, effective, maturity, conventions):
    """A forward rate agreement - the front of a curve between its deposit and its swaps.

    The reset is AT the effective date rather than a fixing lag before it, which is what holds the
    benchmark at par: the quote identifies the forward the curve carries over `[effective,
    maturity]`, and a lag would price a rate fixed off a date the strip says nothing about.

    `FRA_Rate` is authored at ZERO and the print rides in `Quoted_Market_Value`, where
    `QUOTE_WRITERS['FRADeal']` puts it.
    """
    return {
        'Object': 'FRADeal', 'Reference': reference, 'Currency': currency,
        'Interest_Rate': curve,
        'Effective_Date': wire_timestamp(effective), 'Maturity_Date': wire_timestamp(maturity),
        'Reset_Date': wire_timestamp(effective), 'Day_Count': conventions.float_day_count,
        'Principal': conventions.notional, 'FRA_Rate': 0.0, 'Borrower_Lender': 'Borrower',
        'Use_Known_Rate': 'No', 'Known_Rate': 0.0, 'Payment_Timing': 'End',
        'Calendars': conventions.calendar or None}


def _swap(reference, currency, curve, effective, maturity, conventions):
    """A par interest-rate swap - fixed against a single-reset floating leg, and the ONE shape
    every swap row of a strip is authored in, spot-starting or forward-starting, term index or
    overnight.

    `Index_Tenor` of zero months makes each coupon carry ONE reset spanning its own accrual period.
    For a leg paying at the index's own frequency that IS the index - a quarterly leg on 3M JIBAR -
    and for an overnight benchmark it is the compounded rate over the coupon, which at t0 is what
    the daily fixing list prices: the compounded forwards read off a curve telescope to the period
    forward. So `Compounding` OIS is a `Compounding_Method` on this deal and not a cashflow list of
    one item per business day, which is two orders of magnitude of JSON for the same number and
    prices NaN when a coupon spans several resets.

    `Swap_Rate` is authored at ZERO and the print rides in `Quoted_Market_Value`:
    `QUOTE_WRITERS['SwapInterestDeal']` writes it, so a re-tick moves the value plane alone.
    """
    calendar = conventions.calendar or None
    return {
        'Object': 'SwapInterestDeal', 'Reference': reference, 'Currency': currency,
        'Interest_Rate': curve,
        'Effective_Date': wire_timestamp(effective), 'Maturity_Date': wire_timestamp(maturity),
        'Pay_Rate_Type': 'Fixed', 'Pay_Frequency': wire_period(conventions.fixed_frequency),
        'Pay_Day_Count': conventions.fixed_day_count,
        'Pay_Interest_Frequency': wire_period(conventions.fixed_frequency),
        'Pay_Timing': 'End', 'Pay_Payment_Offset': 0, 'Pay_Accrual_Calendars': calendar,
        'Pay_Payment_Calendars': calendar, 'Pay_First_Coupon_Date': None,
        'Pay_Penultimate_Coupon_Date': None,
        'Receive_Frequency': wire_period(conventions.float_frequency),
        'Receive_Day_Count': conventions.float_day_count,
        'Receive_Interest_Frequency': wire_period('0M'), 'Receive_Timing': 'End',
        'Receive_Payment_Offset': 0, 'Receive_Accrual_Calendars': calendar,
        'Receive_Payment_Calendars': calendar, 'Receive_First_Coupon_Date': None,
        'Receive_Penultimate_Coupon_Date': None,
        'Index_Tenor': wire_period('0M'), 'Index_Day_Count': conventions.float_day_count,
        'Index_Frequency': wire_period('0M'), 'Index_Offset': 0,
        'Index_Calendars': None, 'Index_Publication_Calendars': None,
        'Reset_Type': 'Standard', 'Rate_Multiplier': 1.0, 'Rate_Constant': wire_percent(0.0),
        'Floating_Margin': 0.0, 'Fixed_Compounding': 'No',
        'Compounding_Method': conventions.compounding,
        'Known_Rates': None, 'Amortisation': None, 'Swap_Rate': 0.0,
        'Principal': conventions.notional,
        'Interest_Rate_Volatility': '', 'Discount_Rate_Volatility': ''}

# ---------------------------------------------------------------------------------------------
# the block
# ---------------------------------------------------------------------------------------------

def market_price_name(curve):
    """`InterestRatePrices.<curve>` - the `Market Prices` key, whose tail is the `InterestRate`
    factor the family writes. Unlike the other four families this one writes an ordinary
    `InterestRate` rather than a factor named for its own class."""
    return '{}.{}'.format(FAMILY, curve)


def strip_dates(tenor, kind, as_of, conventions, holidays=()):
    """`(effective, maturity)` for one row - the calendar, applied once.

    THE ROLL IS THE SHAPE'S. A deposit rolls Following, a swap's maturity and a FRA's two dates
    Modified Following; spot is `spot_days` business days on. An `ON` print starts at t0 and matures
    the next business day, where a named fixing is a spot-starting deposit of its own tenor: a 3M
    JIBAR deposit is a spot-start three-month instrument, an overnight print is not. A FRA's and a
    forward swap's dates are both measured off SPOT before either is rolled, so a rolled start does
    not drag the end with it.
    """
    spot = _add_business_days(as_of, conventions.spot_days, holidays)
    if kind == 'front':
        if tenor == 'ON':
            return as_of, _next_business_day(as_of, holidays)
        return spot, roll(_add_tenor(spot, tenor), holidays)
    if kind == 'swap':
        return spot, roll(_add_tenor(spot, tenor), holidays, True)
    start, term = split_tenors(tenor)
    effective = _add_tenor(spot, start)
    end = _add_tenor(spot, term) if kind == 'fra' else _add_tenor(effective, term)
    return roll(effective, holidays, True), roll(end, holidays, True)


def author_point(item, as_of, currency, curve, conventions, holidays=()):
    """One `Points` row: an authored instrument, what kind of number is quoted, and the number.

    THE ROW CARRIES WHAT IT WAS AUTHORED FROM. `Tenor` is the label the block's conventions and the
    calendar turn into dates, so a strip re-rolled on a later date is the same plan re-read rather
    than a new one guessed; `Security` is where the number came off.

    `Deal` carries the block with neither `Object` nor `Discount_Rate` on it - the point names the
    type in `DealType` and the family stamps the discount curve from the block it belongs to, so
    neither is authored twice. `Use` is Yes and `Quote_Type` is `Par_Rate`.

    `Quoted_Bid`, `Quoted_Ask` and `Timestamp` ride BESIDE the mid where the terminal answered
    them. They are `schema.MARKET_QUOTE_VALUES` - the value plane `schema.update_market_quote` lets
    a tick move - so the two-way and the print's own clock land as declared evidence.
    """
    effective, maturity = strip_dates(item.label, item.kind, as_of, conventions, holidays)
    reference = '{}_{}'.format(currency, item.label.replace('/', '_'))
    if item.kind == 'front':
        # the overnight's own span is the calendar's - a Friday print is a 3D period, not three
        span = '{}D'.format((maturity - effective).days) if item.label == 'ON' else item.label
        deal = _deposit(reference, currency, curve, effective, maturity, span,
                        conventions.front_day_count, conventions.notional,
                        conventions.calendar or None)
    elif item.kind == 'fra':
        deal = _fra(reference, currency, curve, effective, maturity, conventions)
    else:
        deal = _swap(reference, currency, curve, effective, maturity, conventions)
    row = {
        'Use': item.use,
        'DealType': deal['Object'],
        'Quote_Type': QUOTE_TYPE,
        'Quoted_Market_Value': item.value,
        'Tenor': item.label,
        'Security': item.security,
        'Descriptor': '{} {}{}'.format(currency, item.label,
                                       ' ({})'.format(item.security) if item.security else ''),
        'Deal': {key: value for key, value in deal.items() if key != 'Object'},
    }
    if item.bid is not None:
        row['Quoted_Bid'] = item.bid
    if item.ask is not None:
        row['Quoted_Ask'] = item.ask
    # a DECLARED row has no print to date itself by, so the day it was authored is its stamp
    row['Timestamp'] = wire_timestamp(read_date(item.last_update) or as_of)
    return maturity, row


def ir_curve_block(strip, screen=None, holidays=()):
    """`(Market Prices name, block)` - one verified strip as ONE `InterestRatePrices` block.

    THE BLOCK IS THE CURVE'S DEFINITION: the conventions the rows were authored under are emitted
    beside them, so a reader has the instrument AND the rule that made it, and a later re-roll needs
    nothing this block does not carry. `holidays` is an iterable of dates for the calendar the
    conventions NAME - `Config.parse_calendar_file` answers `{Location: {'holidays': {...}}}` and
    the caller hands the dates over - and none handed in is the Monday-to-Friday rule.

    THE ORDER IS THE CALENDAR'S. Points are emitted by maturity, so the block reads as a strip and
    the knot grid the family builds - one knot per used quote, at that benchmark's last cashflow
    date - comes out ascending without anything having to sort it afterwards.

    TWO BENCHMARKS MATURING ON THE SAME DAY REFUSE BY NAME. The knot rule is what makes the
    bootstrap square: two instruments maturing between the same pair of knots leave the curve
    under-determined between them, and a seed quoting 4W beside 1M does exactly that. It would
    otherwise reach the solve as a singular Jacobian rather than as a sentence.

    `Discount_Rate` is the strip's own - blank for the self-discounting configuration a fetched
    strip builds, and the harder solve, since the unknown appears on both sides.
    """
    screen = screen or CurveScreen()
    used = [item for item in strip.prints if item.use == 'Yes']
    if len(used) < screen.minimum_points:
        raise IncompleteStrip(
            '{} screened to {} believed point{} against a floor of {} - the terminal was asked '
            'about {} securities and refused {} ({}). One knot is a flat curve quoted once, so '
            'there is no strip to solve. Widen the screen the census names, re-run `DV_Bloomberg '
            'discover` if the strip has gone dead, or quote a currency this workstation is '
            'entitled to'.format(
                strip.currency, len(used), '' if len(used) == 1 else 's',
                screen.minimum_points, len(strip.prints) + len(strip.rejected),
                len(strip.rejected),
                ', '.join('{} {}'.format(count, verdict)
                          for verdict, count in sorted(strip.census.items())) or 'nothing refused'))

    dated = [author_point(item, strip.as_of, strip.currency, strip.curve, strip.conventions,
                          holidays) for item in strip.prints]
    knots = {}
    for maturity, row in dated:
        # a held-out row carries no knot, so two of them on one day are no clash at all
        if row['Use'] != 'Yes':
            continue
        if maturity in knots:
            raise IncompleteStrip(
                '{} and {} both mature on {} - the family puts ONE knot per used quote at that '
                'benchmark\'s last cashflow date, so two benchmarks maturing between the same pair '
                'of knots leave the curve under-determined between them and the solve is singular '
                'rather than wrong. Drop one of the two from the currency\'s seeded strip'.format(
                    knots[maturity], row['Descriptor'], maturity.isoformat()))
        knots[maturity] = row['Descriptor']

    conventions = strip.conventions
    return market_price_name(strip.curve), {'instrument': {
        'Currency': strip.currency,
        'Day_Count': conventions.curve_day_count,
        'Discount_Rate': strip.discount_rate,
        'Calendar': conventions.calendar,
        'Spot_Days': conventions.spot_days,
        'Fixed_Frequency': wire_period(conventions.fixed_frequency),
        'Float_Frequency': wire_period(conventions.float_frequency),
        'Fixed_Day_Count': conventions.fixed_day_count,
        'Float_Day_Count': conventions.float_day_count,
        'Front_Day_Count': conventions.front_day_count,
        'Compounding': conventions.compounding,
        'Near_Interpolation': conventions.near_interpolation,
        'Near_Tenor': wire_period(conventions.near_tenor) if conventions.near_tenor else '',
        'Points': [row for _, row in sorted(
            dated, key=lambda item: (item[0], item[1]['Descriptor']))]}}


def author_block(definition, as_of, holidays=(), screen=None):
    """`(Market Prices name, block)` for a curve a desk DECLARED - `ir_curve_block` over stated
    rows rather than over a workstation's verified strip.

    `definition` is `{curve, currency, conventions, rows, discount_rate?}`, `rows` being the
    `RatePrint`s the caller collected. Each row's SHAPE is read off its own tenor, so a desk states
    a tenor, a ticker and a number and never an instrument.

    A used row carrying no number REFUSES BY NAME, carrying the screen's verdict where a print was
    asked for and not believed: a benchmark with no quote identifies no knot, and an unquoted block
    cannot bootstrap.
    """
    conventions = definition['conventions']
    front = front_tenor(conventions.front)
    rows = tuple(replace(row, kind=row_kind(row.label, front)) for row in definition['rows'])
    blank = ['{}{}{}'.format(row.label, ' ({})'.format(row.security) if row.security else '',
                             ': ' + row.verdict if row.verdict else '')
             for row in rows if row.use == 'Yes' and row.value is None]
    if blank:
        raise InvalidQuote(
            '{} carries no quote for {} - a benchmark with no number identifies no knot, so the '
            'block cannot bootstrap. State a `quote` on the row, hold it out with `use` No, or '
            'name a `security` this workstation\'s terminal prices'.format(
                definition['curve'], ', '.join(blank)))
    return ir_curve_block(CurveStrip(
        currency=definition['currency'], curve=definition['curve'], as_of=as_of,
        conventions=conventions, prints=rows,
        discount_rate=definition.get('discount_rate') or ''), screen, holidays)


def block_rows(instrument):
    """A block's `Points` read back as the rows they were authored from - the tenor, the security
    the number came off, the number, its two-way and whether the block uses it. `Timestamp` is read
    in its WIRE spelling, a block on the book being what a tick and a re-roll both hold."""
    return tuple(RatePrint(
        label=row['Tenor'], kind='', security=row.get('Security', ''),
        value=row.get('Quoted_Market_Value'), bid=row.get('Quoted_Bid'), ask=row.get('Quoted_Ask'),
        last_update=(row.get('Timestamp') or {}).get('.Timestamp'), use=row.get('Use', 'Yes'))
        for row in instrument['Points'])


def block_conventions(instrument, seed=None, curve=''):
    """The `CurveConventions` a block DECLARES - what a strip re-rolled on a later date is authored
    under, read off the block rather than off whatever seeded it.

    `front` is the deposit row the strip is fronted by and `notional` its principal, both read off
    the rows. `quote_scale` is the one convention a block does not declare, being a property of the
    FEED and not of the curve, so it comes off the seed's entry where one names this curve.
    """
    rows = instrument['Points']
    front = next(('overnight' if row['Tenor'] == 'ON' else 'fixings/' + row['Tenor']
                  for row in rows if row['DealType'] == 'DepositDeal'), 'overnight')
    principal = next((row['Deal'].get('Principal', row['Deal'].get('Amount')) for row in rows), None)
    declared = ((seed or {}).get('rates', {}).get(curve) or {}).get('conventions') or {}
    return CurveConventions(
        curve_day_count=instrument['Day_Count'], spot_days=instrument['Spot_Days'], front=front,
        front_day_count=instrument['Front_Day_Count'], compounding=instrument['Compounding'],
        fixed_frequency=read_period(instrument['Fixed_Frequency']),
        float_frequency=read_period(instrument['Float_Frequency']),
        fixed_day_count=instrument['Fixed_Day_Count'],
        float_day_count=instrument['Float_Day_Count'],
        notional=principal or OPTIONAL_CONVENTIONS['notional'],
        quote_scale=declared.get('quote_scale', OPTIONAL_CONVENTIONS['quote_scale']),
        calendar=instrument.get('Calendar', ''),
        near_interpolation=instrument.get('Near_Interpolation', ''),
        near_tenor=read_period(instrument.get('Near_Tenor')))


def seeded_rows(seed, curve):
    """The rows a desk could set a seeded curve up with: `strip_candidates`' own labels and
    tickers, the declared front among them and every other fixing left out. NONE of them is
    verified - `strip_entries` is this same walk against a workstation's own map."""
    front = curve_conventions(seed, curve).front
    rows = []
    for candidate in discover.strip_candidates(curve, seed['rates'][curve]):
        path = '/'.join(candidate.path[2:])
        if path == front:
            rows.append({'tenor': front_tenor(front), 'security': candidate.security})
        elif candidate.path[2] in KINDS:
            rows.append({'tenor': candidate.path[-1], 'security': candidate.security})
    return rows


def reauthor(market_prices, name, block):
    """Drop this block and re-install it - the route a block takes when a re-tick is a RE-AUTHORING.

    One spelling for both emitters, `swaption_vol` reaching it too, the mechanism being one thing
    and only the reason differing:

      the swaption ladder  `schema.partition_market_price` gives a values half only to a table
                           whose ROW declares the value keys, and
                           `HullWhite2FactorModelPrices` quotes in `Instrument_Definitions`, whose
                           row declares a `Market_Volatility` and no `Quoted_Market_Value`. A moved
                           vol is therefore not a value at all and every re-quote is a new plan.
      the curve strip      this family DOES have a values half and a same-day re-tick passes as
                           'updated'. What a tick cannot carry is a ROLLED DATE: `Effective_Date`
                           and `Maturity_Date` are structure.

    `market_prices` is the section itself. Returns 'installed' or 'reauthored', so a caller can
    tell a first fetch from a re-quote in a log.
    """
    if not isinstance(block, collections.abc.Mapping) or 'instrument' not in block:
        raise BloombergConfigurationError(
            '{}: a Market Prices block is {{"instrument": {{...}}}}'.format(name))
    existed = name in market_prices
    market_prices.pop(name, None)
    market_prices[name] = block
    return 'reauthored' if existed else 'installed'


def quote_census(strip):
    """The strip's own account of what the terminal served, for a caller with a screen or a report.

    DECLARED LIMITATION: it is not written into the block. `InterestRateCurveParameters` declares
    no `Quote_Source` and no `Quote_Timestamp` where the option and HW2F families do, so the
    only block-level provenance a curve block carries is the per-point `Descriptor`, which names
    the ticker and nothing about the census. The per-point EVIDENCE is declared: `Quoted_Bid`,
    `Quoted_Ask` and `Timestamp` are columns of `Points`.
    """
    return {'currency': strip.currency, 'curve': strip.curve, 'as_of': strip.as_of.isoformat(),
            'asked': len(strip.prints) + len(strip.rejected), 'believed': len(strip.prints),
            'refused': dict(strip.rejected), 'census': strip.census,
            'securities': [item.security for item in strip.prints]}
