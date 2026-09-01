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

`fxvol` writes a surface and `equity_chain` writes a premium ladder; this writes the third thing a
terminal answers for - a CURVE, quoted as the instruments a desk actually deals. The family it
writes for quotes an instrument rather than a number: each `Points` row carries a `Deal` block
authored in the instrument's own conventions, and the solve holds it at PV zero. So the emitter's
whole job is to say what the instrument IS, and to say it from DECLARED DATA rather than from a
guess made in code.

THE CONVENTIONS ARE SEED-DECLARED. A par swap rate means nothing without the accrual it pays on:
`USOSFR10` is annual/annual ACT/360 compounded overnight, `SASW10` is quarterly/quarterly ACT/365
against 3M JIBAR, and the same number under the other convention is a different curve by basis
points. Each seeded currency carries a `conventions` block and this module READS it, rather than
putting a market convention in code where nobody who owns it can see it. A currency whose entry
lacks one REFUSES BY NAME with the missing fields listed, and a WRONG convention is then a data fix
the owner makes in `seed.json`.

NO SECOND SPELLING OF THE TICKER GRAMMAR. The securities come from `discover.strip_candidates`
walked against the workstation's own verified map: the emitter asks the grammar for its candidates
and keeps the ones the map believed, so a ticker exists in exactly one place in this package and a
strip the terminal never verified cannot enter a block on the strength of being in a seed.

THE FRONT POINT IS DECLARED TOO, and it is where a basis error would otherwise be free. A SOFR OIS
curve's front is the overnight print; a JIBAR-3M curve's front is the 3M JIBAR fixing, and putting
ZARONIA there instead would seed a JIBAR curve with an overnight rate that is not on it. So
`front` names which verified entry seeds the short end, per currency, and the rest of the seeded
fixings are ledgered `not-a-benchmark` rather than silently dropped.

THE QUOTE IS NOT AUTHORED INTO THE DEAL, and that is what makes a re-tick a tick. `QUOTE_WRITERS`
is where a number lands in an instrument - the family stamps `Swap_Rate`, the fixed leg's `Rate`
column and the deposit's pinned schedule itself, off `Quoted_Market_Value`. So every rate-carrying
field here is authored at a NEUTRAL zero and the print rides in `Quoted_Market_Value` alone: the
`Deal` half of a row is a function of the calendar and the conventions and of nothing that moves
between prints, so a value-only re-tick passes `config.update_market_quote` as 'updated' instead of
refusing as a moved plan.

V1 SCOPE, stated rather than discovered: a self-discounting single curve (blank `Discount_Rate`),
the declared front point and the swap strip as seeded. No FRAs, no FX-forward outrights, no
cross-currency, no projection curve. `Quote_Type` is `Par_Rate`, which is the one convention the
family builds.

IMPORTS: the standard library, this package's own modules, and no engine. `discover` is reached for
its grammar, which costs the pandas the package's map layer already carries, so unlike
`equity_chain` this module makes NO pandas-free claim. What it does claim is that nothing here
imports `derivus`: the block is emitted as WIRE JSON (`{'.Timestamp': ...}`, `{'.DateOffset':
'3M'}`, `{'.Percent': 0.0}`), which is what `Config.read_json` and `CustomJsonEncoder` spell
between them, so no engine type is ever constructed out here.
"""
import collections.abc
import datetime
import math
from dataclasses import dataclass, field
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

#: The two authoring shapes a swap strip is written in, and the DealType each one names. `OIS` is a
#: container over an OIS-compounded floating leg and a fixed leg, which the compounding rule
#: requires: `pv_float_cashflow_list` compounds geometrically only when the reset count differs
#: from the cashflow count, and a `SwapInterestDeal`'s generated legs never reach that compression.
#: `Swap` is the vanilla single-reset par swap, one deal.
AUTHORING = {'OIS': 'StructuredDeal', 'Swap': 'SwapInterestDeal'}

#: The day counts this emitter computes an accrual in - ACT/365 and ACT/360 and no more, because an
#: authored `Accrual_Year_Fraction` is used verbatim by `make_float_cashflows`. ACT_365_ISDA and
#: ACT_ACT_ICMA (which the engine answers as days/365 behind a TODO) and the two 30/360 conventions
#: refuse by name rather than being reproduced on trust.
DAY_COUNTS = {'ACT_365': 365.0, 'ACT_360': 360.0}

#: The convention fields a seeded currency must declare before its strip can be authored, and the
#: ones that carry a default. A missing one is a NAMED refusal listing all of them at once, because
#: a desk filling in a seed wants the whole list rather than one field per run.
REQUIRED_CONVENTIONS = ('curve_day_count', 'spot_days', 'front', 'front_day_count', 'authoring',
                        'fixed_frequency', 'float_frequency', 'fixed_day_count', 'float_day_count')
OPTIONAL_CONVENTIONS = {'notional': 1000000.0, 'quote_scale': 1.0}


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
    """What a currency's strip IS - read off the seed, never inferred from a ticker.

    Every field here is a market convention somebody owns. They are validated on construction
    (a day count this module cannot compute, an authoring shape it does not write, a negative
    settlement lag) so a bad declaration refuses at the seed rather than inside a cashflow.
    """
    curve_day_count: str
    spot_days: int
    front: str
    front_day_count: str
    authoring: str
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

    def __post_init__(self):
        if self.authoring not in AUTHORING:
            raise BloombergConfigurationError(
                'authoring {!r} is not a shape this emitter writes - a swap strip is authored as '
                '{}. Fix the currency\'s `conventions` in the seed'.format(
                    self.authoring, ' or '.join(
                        '{!r} (a {})'.format(key, value) for key, value in sorted(AUTHORING.items()))))
        for name in ('curve_day_count', 'front_day_count', 'fixed_day_count', 'float_day_count'):
            _day_count_factor(getattr(self, name), name)
        for name in ('fixed_frequency', 'float_frequency'):
            # POSITIVE, not merely readable: on a zero-length coupon period `_dates_backward` walks
            # forever rather than refusing, and a hang carries no message
            if read_tenor(getattr(self, name), name)[0] <= 0:
                raise BloombergConfigurationError(
                    '{} is {!r} - a leg frequency has to be a positive period, or the coupon '
                    'schedule never advances'.format(name, getattr(self, name)))
        # V1 AUTHORS BOTH OIS LEGS ON ONE SCHEDULE: `_ois_swap` rolls the coupon dates once off
        # `fixed_frequency`, so a differing `float_frequency` would be declared and never read. The
        # `Swap` path reads both - the engine generates the legs there - and is left alone.
        if self.authoring == 'OIS' and self.float_frequency != self.fixed_frequency:
            raise BloombergConfigurationError(
                'float_frequency is {!r} against a fixed_frequency of {!r} on an OIS declaration, '
                'and v1 authors BOTH OIS legs on ONE schedule rolled off fixed_frequency - so the '
                'float frequency would be declared here and read nowhere. Declare the two equal, '
                'or declare `authoring: "Swap"`, where the engine generates each leg on its own '
                'frequency'.format(self.float_frequency, self.fixed_frequency))
        if not isinstance(self.spot_days, int) or self.spot_days < 0:
            raise BloombergConfigurationError(
                'spot_days must be a whole number of business days at or above zero, not {!r} - it '
                'is the settlement lag the strip\'s Effective_Date is placed at'.format(
                    self.spot_days))
        if not math.isfinite(self.notional) or self.notional <= 0.0:
            raise BloombergConfigurationError('notional must be positive and finite')
        if not math.isfinite(self.quote_scale) or self.quote_scale == 0.0:
            raise BloombergConfigurationError('quote_scale must be finite and non-zero')

    @property
    def deal_type(self) -> str:
        return AUTHORING[self.authoring]


def curve_conventions(seed: Mapping, currency: str) -> CurveConventions:
    """The declared conventions of one seeded currency, or the refusal naming EVERY absent field at
    once - a desk extending a seed wants the whole questionnaire, not one field per run.
    """
    spec = seed.get('rates', {}).get(currency)
    if spec is None:
        raise BloombergConfigurationError(
            'the seed names no rates entry for {} - a currency this workstation never seeded has no '
            'strip to fetch and no conventions to author one in. Add it to `seed.json` and re-run '
            '`DV_Bloomberg discover`'.format(currency))
    declared = spec.get('conventions')
    if not isinstance(declared, collections.abc.Mapping):
        raise BloombergConfigurationError(
            '{} carries no `conventions` block - a par swap rate is not an instrument until '
            'something says what it accrues on, and this emitter reads that rather than guessing '
            'it. Declare {} on the {} entry in your seed (see derivus_bloomberg/seed.json for the '
            'shipped USD and ZAR declarations)'.format(
                currency, ', '.join(REQUIRED_CONVENTIONS), currency))
    missing = [name for name in REQUIRED_CONVENTIONS if declared.get(name) is None]
    if missing:
        raise BloombergConfigurationError(
            '{} declares no {} - the full set this emitter reads is {}, and a convention block '
            'filled in half way authors an instrument nobody stated. Fix the {} entry in your '
            'seed'.format(currency, ', '.join(missing), ', '.join(REQUIRED_CONVENTIONS), currency))
    unknown = sorted(set(declared) - set(REQUIRED_CONVENTIONS) - set(OPTIONAL_CONVENTIONS))
    if unknown:
        raise BloombergConfigurationError(
            '{} declares {} which this emitter reads nothing of - a convention nobody reads is a '
            'convention that is not applied, which is worse than one that is missing. Remove it, or '
            'spell it as one of {}'.format(
                currency, ', '.join(unknown),
                ', '.join(sorted(set(REQUIRED_CONVENTIONS) | set(OPTIONAL_CONVENTIONS)))))
    # `front` is a PATH INTO THE SEED, so it is checked HERE rather than in `__post_init__`, which
    # cannot see one: a path that names nothing does not fail, it aims elsewhere. `front:
    # 'strip/1Y'` would author the 1Y par swap as a one-day deposit labelled overnight.
    admissible = _seeded_fronts(seed, currency)
    if declared['front'] not in admissible:
        raise BloombergConfigurationError(
            '{} declares its front as {!r}, which is not an entry its seed could name - the '
            'admissible spellings are {}. The front is what seeds the short end, and a `front` '
            'aimed at the swap strip would author a par swap as a one-day overnight deposit and '
            'name it `overnight` in the Descriptor rather than refuse. Fix the {} entry in your '
            'seed'.format(currency, declared['front'],
                          ', '.join(admissible) or 'none: the entry seeds neither an `overnight` '
                          'print nor any `fixings`, so it can carry no front at all', currency))
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
    #: How many daily fixings the whole block may author, across every OIS benchmark in it.
    #:
    #: A SIZE BOUND. An OIS floating leg is one authored item per business-day fixing (see
    #: `_ois_swap`), so a 30Y benchmark is about 7,800 items and the shipped USD strip about 25,700
    #: - roughly fourteen megabytes of JSON, which the default admits. What the cap catches is a
    #: seed reaching further, and it refuses with the count rather than with a MemoryError.
    maximum_fixings: int = 50000

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
    """One quoted point of a strip as the terminal answered for it - raw, unjudged, not believed."""
    label: str
    kind: str
    security: str
    value: float | None
    bid: float | None = None
    ask: float | None = None
    last_update: str | None = None


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


def strip_entries(document, seed, currency):
    """`(wanted, ledger)` - the verified securities of one currency's strip, walked off the GRAMMAR.

    `discover.strip_candidates` supplies the candidates and each one's `path` is looked up in the
    workstation's own map, so this module spells no ticker and no map path. A candidate the map did
    not verify is `unverified` on the ledger by name; a seeded fixing that is not the declared
    front point is `not-a-benchmark`, since a 6M JIBAR print is an index rather than an instrument
    this block holds at par.

    `wanted` is `[(label, kind, security)]` with `kind` one of `front` / `swap`, in the grammar's
    own order - the emitter sorts by maturity later, that being a property of the calendar.
    """
    conventions = curve_conventions(seed, currency)
    blocks = document.get('blocks', {})
    front_path = ('rates', currency) + tuple(part for part in conventions.front.split('/') if part)
    wanted, ledger, found_front = [], {}, False
    for candidate in discover.strip_candidates(currency, seed['rates'][currency]):
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
            wanted.append((_front_label(candidate.path), 'front', security))
            found_front = True
        elif candidate.path[2] == 'strip':
            wanted.append((candidate.path[-1], 'swap', security))
        else:
            ledger[security] = 'not-a-benchmark'
    if not found_front:
        raise BloombergConfigurationError(
            '{} declares its front point as {!r} and the map carries no verified entry there - the '
            'front is what seeds the short end of the curve, and a strip quoted from its first swap '
            'alone leaves everything under {} unidentified. Re-run `DV_Bloomberg discover` (the '
            'entry may have gone dead), or declare a `front` the map verified: the seeded ones are '
            '{}'.format(currency, conventions.front, wanted[0][0] if wanted else 'its first knot',
                        ', '.join(_seeded_fronts(seed, currency)) or 'none'))
    return tuple(wanted), ledger


def _seeded_fronts(seed, currency):
    """The `front` spellings a currency's seed entry could name - what a refusal offers."""
    spec = seed.get('rates', {}).get(currency, {})
    return (['overnight'] if spec.get('overnight') else []) + [
        'fixings/{}'.format(label) for label in sorted(spec.get('fixings', {}))]


def _front_label(path):
    """The front point's TENOR as a label. A named fixing carries its own (`fixings/3M` is a 3M
    deposit); an overnight print has none to carry, so `overnight` is the label and the deposit's
    span is worked out from the calendar at authoring time."""
    return path[-1] if path[2] == 'fixings' else 'overnight'


def fetch_curve_strip(source, document, seed, currency, as_of, curve=None, screen=None,
                      batch=BATCH, on_batch=None):
    """One currency's verified strip, screened - a `CurveStrip`.

    ONE ROUND TRIP over the securities the map believed, asking each the value, both sides of its
    two-way and its own last print. The tolerant reader makes the request and the strict policy is
    applied CLIENT-SIDE, per print: one dead point is a curve with one fewer knot, where a strip
    refused whole is no curve at all.

    `curve` names the `InterestRate` factor this strip builds and defaults to the CURRENCY, the
    single-curve V1's own name and what an `FxRate`'s `Interest_Rate` points at. A desk running a
    multi-curve set names its curves itself (`USD-OIS`, `ZAR-JIBAR-3M`); the block key is
    `InterestRatePrices.<curve>` and the deals project off it.
    """
    screen = screen or CurveScreen()
    conventions = curve_conventions(seed, currency)
    wanted, ledger = strip_entries(document, seed, currency)
    report = probe(source, [security for _, _, security in wanted], batch=batch, on_batch=on_batch)

    prints, rejected = [], dict(ledger)
    for label, kind, security in wanted:
        row = report.get(security, {'ok': False, 'error': 'no answer in the response', 'fields': {}})
        answered = row.get('fields') or {}
        if not answered:
            rejected[security] = 'invalid'
            continue
        prints.append(RatePrint(
            label=label, kind=kind, security=security,
            value=_scaled(answered.get('PX_LAST'), conventions.quote_scale),
            bid=_scaled(answered.get('PX_BID'), conventions.quote_scale),
            ask=_scaled(answered.get('PX_ASK'), conventions.quote_scale),
            last_update=read_word(answered.get('LAST_UPDATE_DT')) or None))
    accepted, screened = screen_strip(prints, as_of, screen)
    rejected.update(screened)
    return CurveStrip(currency=currency, curve=curve or currency, as_of=as_of,
                      conventions=conventions, prints=accepted, rejected=rejected)


def _scaled(value, quote_scale):
    number = read_number(value)
    return None if number is None else number * quote_scale


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


def _is_business_day(date):
    """Monday to Friday, and NO HOLIDAY CALENDAR. The authored deals carry `Accrual_Calendars:
    None` and `Payment_Calendars: None`, so the engine adjusts nothing either - one convention on
    both sides of the boundary. A desk needing a real settlement calendar is read here."""
    return date.weekday() < 5


def _next_business_day(date):
    moved = date + datetime.timedelta(days=1)
    while not _is_business_day(moved):
        moved += datetime.timedelta(days=1)
    return moved


def _add_business_days(date, count):
    moved = date
    for _ in range(count):
        moved = _next_business_day(moved)
    return moved


def _business_days(start, end):
    """Every business day in `[start, end)` - the days an overnight leg takes a fixing on.

    HALF OPEN AT THE END, so the last day found accrues to the coupon's own end rather than past
    it. A `start` on a weekend is not a business day and does not appear, so the days between it
    and the first Monday belong to no window returned here: TILING THE COUPON IS THE CALLER'S JOB,
    and `_ois_swap` does it by putting the coupon's own start in front of what comes back.
    """
    days, moved = [], start
    while moved < end:
        if _is_business_day(moved):
            days.append(moved)
        moved += datetime.timedelta(days=1)
    return days


def _dates_backward(end, start, frequency):
    """The coupon dates of one leg, rolled BACKWARD from maturity and clipped at the effective date
    - `instruments.generate_dates_backward`, re-spelled.

    BACKWARD RATHER THAN FORWARD, so the stub is at the FRONT. That is the market's roll and what
    the engine does generating a `SwapInterestDeal`'s legs, so both authoring shapes put the stub
    in the same place: an 18M OIS pays at +6M and +18M under either.
    """
    count, unit = read_tenor(frequency, 'frequency')
    if count <= 0:
        raise BloombergConfigurationError(
            'a coupon schedule cannot roll by {!r} - the loop below would never reach its own '
            'start'.format(frequency))
    dates, index, moved = [end], 1, end
    while moved > start:
        moved = max(start, _step_back(end, count * index, unit))
        dates.append(moved)
        index += 1
    dates.reverse()
    return dates


def _step_back(end, count, unit):
    if unit == 'D':
        return end - datetime.timedelta(days=count)
    if unit == 'W':
        return end - datetime.timedelta(weeks=count)
    return _add_months(end, -count * (12 if unit == 'Y' else 1))


def _day_count_factor(day_count, what='day count'):
    factor = DAY_COUNTS.get(day_count)
    if factor is None:
        raise BloombergConfigurationError(
            '{!r} is not a {} this emitter computes an accrual in - it writes {}. An authored '
            '`Accrual_Year_Fraction` is used verbatim by the engine, so ACT_365_ISDA and '
            'ACT_ACT_ICMA are refused rather than reproduced (the engine answers both as days/365 '
            'behind a TODO) and the 30/360 conventions need date arithmetic this module does not '
            'author'.format(day_count, what, ' or '.join(sorted(DAY_COUNTS))))
    return factor


def _accrual(start, end, day_count):
    """`(end - start).days / N` - `utils.get_day_count_accrual`'s ACT/N branch, the only one this
    module authors into a cashflow."""
    return (end - start).days / _day_count_factor(day_count)


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


def wire_percent(value):
    """`utils.Percent`'s wire form. The number is in PERCENT - `{'.Percent': 4.28}` is 4.28%, and
    the decoded object's `.amount` is 0.0428."""
    return {'.Percent': value}


def wire_basis(value):
    return {'.Basis': value}


def wire_date_list(pairs):
    return {'.DateList': [[date.isoformat(), value] for date, value in pairs]}


# ---------------------------------------------------------------------------------------------
# authoring an instrument
# ---------------------------------------------------------------------------------------------

def _deposit(reference, currency, curve, effective, maturity, tenor, day_count, notional):
    """A money-market deposit - the strip's FRONT point, and the one shape both authorings share.

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
        'Accrual_Calendars': None, 'Payment_Calendars': None,
        'First_Coupon_Date': None, 'Penultimate_Coupon_Date': None,
        'Rate_Currency': '', 'FX_Reset_Offset': 0, 'Known_FX_Rates': None,
        'Interest_Rate_Schedule': wire_date_list(())}


def _par_swap(reference, currency, curve, effective, maturity, conventions):
    """A vanilla par interest-rate swap - fixed against a single-reset floating leg.

    `Index_Tenor` of zero months makes each coupon carry ONE reset spanning its own accrual period,
    which for a leg paying at the index's own frequency IS the index: a quarterly leg on 3M JIBAR.
    A leg whose payment frequency differs from its index tenor is a different instrument and V1
    does not author it.

    `Swap_Rate` is authored at ZERO and the print rides in `Quoted_Market_Value`:
    `QUOTE_WRITERS['SwapInterestDeal']` writes it, so a re-tick moves the value plane alone.
    """
    return {
        'Object': 'SwapInterestDeal', 'Reference': reference, 'Currency': currency,
        'Interest_Rate': curve,
        'Effective_Date': wire_timestamp(effective), 'Maturity_Date': wire_timestamp(maturity),
        'Pay_Rate_Type': 'Fixed', 'Pay_Frequency': wire_period(conventions.fixed_frequency),
        'Pay_Day_Count': conventions.fixed_day_count,
        'Pay_Interest_Frequency': wire_period(conventions.fixed_frequency),
        'Pay_Timing': 'End', 'Pay_Payment_Offset': 0, 'Pay_Accrual_Calendars': None,
        'Pay_Payment_Calendars': None, 'Pay_First_Coupon_Date': None,
        'Pay_Penultimate_Coupon_Date': None,
        'Receive_Frequency': wire_period(conventions.float_frequency),
        'Receive_Day_Count': conventions.float_day_count,
        'Receive_Interest_Frequency': wire_period('0M'), 'Receive_Timing': 'End',
        'Receive_Payment_Offset': 0, 'Receive_Accrual_Calendars': None,
        'Receive_Payment_Calendars': None, 'Receive_First_Coupon_Date': None,
        'Receive_Penultimate_Coupon_Date': None,
        'Index_Tenor': wire_period('0M'), 'Index_Day_Count': conventions.float_day_count,
        'Index_Frequency': wire_period('0M'), 'Index_Offset': 0,
        'Index_Calendars': None, 'Index_Publication_Calendars': None,
        'Reset_Type': 'Standard', 'Rate_Multiplier': 1.0, 'Rate_Constant': wire_percent(0.0),
        'Floating_Margin': 0.0, 'Fixed_Compounding': 'No', 'Compounding_Method': 'None',
        'Known_Rates': None, 'Amortisation': None, 'Swap_Rate': 0.0,
        'Principal': conventions.notional,
        'Interest_Rate_Volatility': '', 'Discount_Rate_Volatility': ''}


def _ois_swap(reference, currency, curve, effective, maturity, conventions):
    """An OIS swap as a CONTAINER over two legs - the shape the compounding rule requires.

    `pv_float_cashflow_list` compounds an accrual period geometrically when the reset count differs
    from the cashflow count, a reshape set up by `compress_no_compounding(groupsize=-1)` under
    `Compounding_Method='OIS'`. So the floating leg is ONE ITEM PER FIXING, every item of a coupon
    sharing that coupon's payment date: the compression merges them into one cashflow carrying
    every reset at `Weight` 1, and only then compounds. A leg authored as one item with many resets
    arrives weighted `1/n` and compounds at a fraction of the rate - the AVERAGING legs' arithmetic
    - and a `SwapInterestDeal`'s generated legs never reach the compression at all.

    The fixed leg carries the quote on every row of its schedule
    (`QUOTE_WRITERS['CFFixedInterestListDeal']`), so every `Rate` here is authored at ZERO percent
    and the print rides in `Quoted_Market_Value` alone.

    THE FIXING WINDOWS PARTITION THE COUPON, which is what puts the two legs on one convention.
    Both are rolled off the SAME coupon dates, so they accrue the same span only if the float leg's
    windows tile `[coupon_start, coupon_end]` exactly - and the coupon's own start is a boundary
    whatever weekday it falls on, since a fixing accrues THROUGH a weekend at a coupon boundary
    exactly as it does inside one. Starting the leg at the first BUSINESS day instead drops days:
    on the USD 5Y OIS effective 2026-09-02, the coupon starting Saturday 2028-09-02 accrued
    1.00833333 of a year against the fixed leg's 1.01388889. So the coupon start goes in front of
    whatever `_business_days` returns.
    """
    coupons = _dates_backward(maturity, effective, conventions.fixed_frequency)
    float_items, fixed_items = [], []
    for start, end in zip(coupons[:-1], coupons[1:]):
        fixings = _business_days(start, end)
        if not fixings or fixings[0] != start:
            fixings.insert(0, start)
        for fixing, following in zip(fixings, fixings[1:] + [end]):
            accrual = _accrual(fixing, following, conventions.float_day_count)
            float_items.append({
                'Payment_Date': wire_timestamp(end), 'Notional': conventions.notional,
                'Accrual_Start_Date': wire_timestamp(fixing), 'Accrual_End_Date': wire_timestamp(following),
                'Accrual_Day_Count': conventions.float_day_count,
                'Accrual_Year_Fraction': accrual,
                'Resets': [[wire_timestamp(fixing), wire_timestamp(fixing), wire_timestamp(following), accrual,
                            wire_period('1D'), conventions.float_day_count, '0D', 0.0, 'No',
                            wire_percent(0.0)]],
                'Margin': wire_basis(0.0), 'Fixed_Amount': 0.0,
                'FX_Reset_Date': None, 'Known_FX_Rate': 0.0})
        fixed_items.append({
            'Payment_Date': wire_timestamp(end), 'Notional': conventions.notional,
            'Rate': wire_percent(0.0),
            'Accrual_Start_Date': wire_timestamp(start), 'Accrual_End_Date': wire_timestamp(end),
            'Accrual_Day_Count': conventions.fixed_day_count,
            'Accrual_Year_Fraction': _accrual(start, end, conventions.fixed_day_count),
            'Fixed_Amount': 0.0, 'Discounted': 'No',
            'FX_Reset_Date': None, 'Known_FX_Rate': 0.0})

    return {
        'Object': 'StructuredDeal', 'Reference': reference, 'Currency': currency,
        'Net_Cashflows': 'Yes', 'Children': [
            _cashflow_leg('CFFloatingInterestListDeal', reference + '_FLOAT', currency, 'Buy',
                          {'Compounding_Method': 'OIS', 'Averaging_Method': 'Average_Interest',
                           'Properties': [], 'Items': float_items},
                          Forecast_Rate=curve, Rate_Adjustment_Method='None',
                          Rate_Sticky_Month_End='Yes', Rate_Offset=0, Rate_Calendars=None,
                          Accrual_Calendars=None, Forecast_Rate_Cap_Volatility='',
                          Forecast_Rate_Swaption_Volatility='', Discount_Rate_Cap_Volatility='',
                          Discount_Rate_Swaption_Volatility=''),
            _cashflow_leg('CFFixedInterestListDeal', reference + '_FIXED', currency, 'Sell',
                          {'Compounding': 'No', 'Items': fixed_items},
                          Calendars=None, Rate_Currency='')]}


def _cashflow_leg(object_type, reference, currency, buy_sell, cashflows, **extra):
    """The `CashflowListDeal` block both interest-cashflow legs share, plus the type's own fields.

    `Discount_Rate` is absent here and on every other authored deal: `author_quote` stamps it on
    the node and recurses into `Children`, because what an instrument PROJECTS off is its own
    business while what the quote set DISCOUNTS on is a property of the curve set.
    """
    return dict({
        'Object': object_type, 'Reference': reference, 'Currency': currency,
        'Buy_Sell': buy_sell, 'Description': '',
        'Settlement_Date': None, 'Settlement_Amount': 0.0, 'Settlement_Style': 'Physical',
        'Settlement_Amount_Is_Clean': 'Yes', 'Is_Defaultable': 'No', 'Repo_Rate': '',
        'Recovery_Rate': '', 'Survival_Probability': '', 'Investment_Horizon': None,
        'Issuer': '', 'Settlement_Rate': '', 'Cashflows': cashflows}, **extra)


# ---------------------------------------------------------------------------------------------
# the block
# ---------------------------------------------------------------------------------------------

def market_price_name(curve):
    """`InterestRatePrices.<curve>` - the `Market Prices` key, whose tail is the `InterestRate`
    factor the family writes. Unlike the other four families this one writes an ordinary
    `InterestRate` rather than a factor named for its own class."""
    return '{}.{}'.format(FAMILY, curve)


def strip_dates(item, as_of, conventions):
    """`(effective, maturity, tenor label)` for one print - the calendar, applied once.

    The front point starts at t0 and matures on the next business day when it is an OVERNIGHT rate,
    and starts at spot like every swap when it is a named fixing: a 3M JIBAR deposit is a spot-start
    three-month instrument, an O/N print is not. The tenor label doubles as the deposit's own
    payment frequency, so the pinned schedule is ONE period - an overnight spanning a weekend is a
    `3D` period rather than three of them.
    """
    spot = _add_business_days(as_of, conventions.spot_days)
    if item.kind != 'front':
        return spot, _add_tenor(spot, item.label), item.label
    if item.label == 'overnight':
        maturity = _next_business_day(as_of)
        return as_of, maturity, '{}D'.format((maturity - as_of).days)
    return spot, _add_tenor(spot, item.label), item.label


def author_point(item, as_of, currency, curve, conventions):
    """One `Points` row: an authored instrument, what kind of number is quoted, and the number.

    `Deal` carries the block with neither `Object` nor `Discount_Rate` on it - the point names the
    type in `DealType` and the family stamps the discount curve from the block it belongs to, so
    neither is authored twice. `Use` is Yes, `Quote_Type` is `Par_Rate`, and `Descriptor` names the
    ticker the number came off, which is the only place in the block a security lands.

    `Quoted_Bid`, `Quoted_Ask` and `Timestamp` ride BESIDE the mid where the terminal answered
    them. They are `schema.MARKET_QUOTE_VALUES` - the value plane `config.update_market_quote` lets
    a tick move - and `InterestRateCurveParameters.Points` declares all three among its nine
    sub-fields, so the two-way and the print's own clock land as declared evidence.
    """
    effective, maturity, tenor = strip_dates(item, as_of, conventions)
    reference = '{}_{}'.format(currency, item.label.replace('/', '_'))
    if item.kind == 'front':
        deal = _deposit(reference, currency, curve, effective, maturity, tenor,
                        conventions.front_day_count, conventions.notional)
    elif conventions.authoring == 'OIS':
        deal = _ois_swap(reference, currency, curve, effective, maturity, conventions)
    else:
        deal = _par_swap(reference, currency, curve, effective, maturity, conventions)
    row = {
        'Use': 'Yes',
        'DealType': deal['Object'],
        'Quote_Type': QUOTE_TYPE,
        'Quoted_Market_Value': item.value,
        'Descriptor': '{} {} ({})'.format(currency, item.label, item.security),
        'Deal': {key: value for key, value in deal.items() if key != 'Object'},
    }
    if item.bid is not None:
        row['Quoted_Bid'] = item.bid
    if item.ask is not None:
        row['Quoted_Ask'] = item.ask
    row['Timestamp'] = wire_timestamp(read_date(item.last_update))
    return maturity, row


def ir_curve_block(strip, screen=None):
    """`(Market Prices name, block)` - one verified strip as ONE `InterestRatePrices` block.

    THE ORDER IS THE CALENDAR'S. Points are emitted by maturity, so the block reads as a strip and
    the knot grid the family builds - one knot per used quote, at that benchmark's last cashflow
    date - comes out ascending without anything having to sort it afterwards.

    TWO BENCHMARKS MATURING ON THE SAME DAY REFUSE BY NAME. The knot rule is what makes the
    bootstrap square: two instruments maturing between the same pair of knots leave the curve
    under-determined between them, and a seed quoting 4W beside 1M does exactly that. It would
    otherwise reach the solve as a singular Jacobian rather than as a sentence.

    `Discount_Rate` is blank - the self-discounting single-curve configuration, and the harder
    solve, since the unknown appears on both sides. V1 authors no multi-curve case.
    """
    screen = screen or CurveScreen()
    if len(strip.prints) < screen.minimum_points:
        raise IncompleteStrip(
            '{} screened to {} believed point{} against a floor of {} - the terminal was asked '
            'about {} securities and refused {} ({}). One knot is a flat curve quoted once, so '
            'there is no strip to solve. Widen the screen the census names, re-run `DV_Bloomberg '
            'discover` if the strip has gone dead, or quote a currency this workstation is '
            'entitled to'.format(
                strip.currency, len(strip.prints), '' if len(strip.prints) == 1 else 's',
                screen.minimum_points, len(strip.prints) + len(strip.rejected),
                len(strip.rejected),
                ', '.join('{} {}'.format(count, verdict)
                          for verdict, count in sorted(strip.census.items())) or 'nothing refused'))

    dated = [author_point(item, strip.as_of, strip.currency, strip.curve, strip.conventions)
             for item in strip.prints]
    knots = {}
    for maturity, row in dated:
        if maturity in knots:
            raise IncompleteStrip(
                '{} and {} both mature on {} - the family puts ONE knot per used quote at that '
                'benchmark\'s last cashflow date, so two benchmarks maturing between the same pair '
                'of knots leave the curve under-determined between them and the solve is singular '
                'rather than wrong. Drop one of the two from the currency\'s seeded strip'.format(
                    knots[maturity], row['Descriptor'], maturity.isoformat()))
        knots[maturity] = row['Descriptor']

    points = [row for _, row in sorted(dated, key=lambda item: (item[0], item[1]['Descriptor']))]
    fixings = sum(len(child['Cashflows']['Items'])
                  for row in points if row['DealType'] == 'StructuredDeal'
                  for child in row['Deal']['Children']
                  if child['Object'] == 'CFFloatingInterestListDeal')
    if fixings > screen.maximum_fixings:
        raise IncompleteStrip(
            '{} authors {} daily fixings across its {} OIS benchmarks, past the declared cap of {}. '
            'An OIS floating leg is ONE ITEM PER BUSINESS-DAY FIXING - that is what makes the leg '
            'compound geometrically rather than average - so the block grows with the SUM of the '
            'strip\'s tenors rather than with its point count, and a strip reaching {} would encode '
            'to something no reader downstream should be handed. Cut the currency\'s seeded `years` '
            'back, or raise CurveScreen.maximum_fixings deliberately'.format(
                strip.currency, fixings,
                sum(1 for row in points if row['DealType'] == 'StructuredDeal'),
                screen.maximum_fixings, max(knots).isoformat()))
    return market_price_name(strip.curve), {'instrument': {
        'Currency': strip.currency,
        'Day_Count': strip.conventions.curve_day_count,
        'Discount_Rate': '',
        'Points': points}}


def reauthor(market_prices, name, block):
    """Drop this block and re-install it - the route a block takes when a re-tick is a RE-AUTHORING.

    One spelling for both emitters, `swaption_vol` reaching it, because the mechanism is one thing
    and only the reason differs:

      the swaption ladder  `schema.partition_market_price` gives every family whose quotes do not
                           live in `Points` rows an EMPTY values half, and
                           `HullWhite2FactorModelPrices` quotes in `Instrument_Definitions`. So a
                           moved vol is not a value AT ALL and every re-quote is a new plan.
      the curve strip      this family DOES have a values half and a same-day re-tick passes as
                           'updated'. What a tick cannot carry is a ROLLED DATE: `Effective_Date`
                           and `Maturity_Date` are structure, so tomorrow's strip of the same
                           benchmarks is a different plan and rightly refuses.

    `market_prices` is the section itself - `cfg.params['Market Prices']`, or a wire document's
    `Calc/MergeMarketData/ExplicitMarketData/Market Prices`. Returns 'installed' or 'reauthored',
    so a caller can tell a first fetch from a re-quote in a log.
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
    no `Quote_Source` and no `Quote_Timestamp` where the Heston-Nandi and HW2F families do, so the
    only block-level provenance a curve block carries is the per-point `Descriptor`, which names
    the ticker and nothing about the census. The per-point EVIDENCE is declared: `Quoted_Bid`,
    `Quoted_Ask` and `Timestamp` are columns of `Points`.
    """
    return {'currency': strip.currency, 'curve': strip.curve, 'as_of': strip.as_of.isoformat(),
            'asked': len(strip.prints) + len(strip.rejected), 'believed': len(strip.prints),
            'refused': dict(strip.rejected), 'census': strip.census,
            'securities': [item.security for item in strip.prints]}
