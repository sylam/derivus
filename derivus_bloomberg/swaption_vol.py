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
"""A workstation's verified ATM swaption grid as one `HullWhite2FactorModelPrices` block.

`ir_curve` writes the curve the model diffuses; this writes what identifies its diffusion. The
family fits five parameters - two sigma term structures, two mean reversions and a correlation - to
a set of ATM swaptions, and each benchmark is a FORWARD STARTING SWAP described by eight columns:
its start, its tenor, both leg frequencies, both leg day counts, the quoted vol and the objective
weight. That is the whole block, and this emitter fills it from a verified grid.

TWO COLUMNS, NOT ONE, AND THE ROADMAP RECORDS WHY. `create_market_swaps` reads
`Floating_Day_Count` and `Fixed_Day_Count` SEPARATELY - the float leg's generates the par swap rate
and the fixed leg's only exists on the unequal-frequency branch - and an authored block spelling
one `Day_Count` for both used to die downstream of that. `INSTRUMENT_COLUMNS` below is held against
the committed declaration by a gate, as DATA, so the day the family's row changes the gate says so
instead of a fetch failing in a cashflow generator.

THE CONVENTIONS ARE SEED-DECLARED, exactly as the curve's are: the forward swap a `SASN` cell is a
vol OF is quarterly/quarterly ACT/365 against 3M JIBAR, and nothing in the ticker says so. A
currency whose `swaption` entry carries no `conventions` block REFUSES BY NAME with the missing
fields listed, and a wrong convention is then a data fix the owner makes in `seed.json`.

THE QUOTED DISTRIBUTION IS DECLARED AND THE CALIBRATION READS IT. `SASN` is `ZAR SWPT NVOL`: a
NORMAL (Bachelier) vol in basis points. `create_market_swaps` builds each benchmark's market
premium under the referenced surface's `Distribution_Type` - Bachelier for `Normal`, displaced
Black for `Lognormal` - the same convention `Factor3D.get_subtype` carries into every deal's
`Volatility` dependency, so a ladder is fitted and marked under one convention. This emitter
transcribes what the terminal quoted, scales it into the family's `Percent` column, and states
the distribution in `Quote_Source` and on the returned ladder; the referenced `InterestYieldVol`
surface must declare the matching `Distribution_Type` for the fit to read it.

A RE-TICK IS A RE-AUTHORING, and that is structural rather than a preference.
`schema.partition_market_price` gives every family whose quotes do not live in `Points` rows an
EMPTY values half, and this family quotes in `Instrument_Definitions`
(`tests/test_market_prices_partition.py` asserts it by name). So a moved vol is 'structure differs'
to `config.update_market_quote` and there is no tick that reaches it: `reauthor` drops the block
before re-installing it, exactly as `POST /book/hn` does with a Heston-Nandi ladder and for the
same reason.

IMPORTS: the standard library, this package's own modules, and no engine. The block is emitted as
WIRE JSON (`{'.DateOffset': '1Y'}`, `{'.Percent': 1.45}`, `{'.Timestamp': ...}`), which is what
`Config.read_json` reads and `CustomJsonEncoder` writes.
"""
import collections.abc
import datetime
import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from . import discover
from .errors import BloombergConfigurationError, IncompleteLadder
# the package's ONE spelling of the wire form and of a terminal answer, reached rather than
# re-spelled: `{'.Percent': 1.45}` has to mean the same thing in both emitters or a desk reading two
# blocks side by side is reading two conventions
from .ir_curve import (BATCH, QUOTE_FIELDS, probe, read_date, read_number, read_tenor, read_word,
                       wire_percent, wire_period, wire_timestamp)
# RE-EXPORTED RATHER THAN RE-SPELLED. A drop-and-re-install is one mechanism and both emitters need
# it - this family because its values half is EMPTY so no tick can reach it, the curve strip because
# a rolled date is structure. `swaption_vol.reauthor` therefore still resolves and still means what
# this module's docstring says it does; there is simply one of it. See its own docstring for both
# reasons stated apart.
from .ir_curve import reauthor  # noqa: F401

#: The family this emitter writes for.
FAMILY = 'HullWhite2FactorModelPrices'

#: The `Instrument_Definitions` row, in the committed schema's own order. Spelled here because the
#: package may not import the engine, and held against `HullWhite2FactorModelParameters.fields` by a
#: gate that reads the COMMITTED declaration and compares it as data - the
#: whitelist-against-declaration pattern `equity_chain.HN_REFERENCE_TYPES` already rides on.
INSTRUMENT_COLUMNS = ('Start', 'Tenor', 'Floating_Frequency', 'Fixed_Frequency',
                      'Floating_Day_Count', 'Fixed_Day_Count', 'Market_Volatility', 'Weight')

#: The value-plane keys a row carries beside its vol. NOT DECLARED by the family's row - the eight
#: columns above are all of it - so they ride as undeclared keys `create_market_swaps` reads past.
#: Carried anyway, because the two-way and the print's own clock are the EVIDENCE, and named as a
#: finding rather than dropped to fit eight columns. `equity_chain.QUOTE_VALUE_KEYS`, one family
#: over, and with the same consequence: this family's quote column is `Market_Volatility` rather
#: than `Quoted_Market_Value`, so `schema.MARKET_QUOTE_VALUES` cannot see the mid either.
QUOTE_VALUE_KEYS = ('Quoted_Bid', 'Quoted_Ask', 'Timestamp')

#: The distributions a seeded grid may declare its quotes in. Bloomberg spells the two families
#: `{CCY}SN` (normal, basis points) and `{CCY}SV` (lognormal, percent) - the README's own note off
#: the 2026-08-27 terminal session - and which one a prefix is cannot be read off the prefix.
DISTRIBUTIONS = ('Normal', 'Lognormal')

#: The convention fields a seeded currency's swaption entry must declare, and the ones defaulted.
REQUIRED_CONVENTIONS = ('fixed_frequency', 'float_frequency', 'fixed_day_count', 'float_day_count',
                        'distribution', 'quote_scale')
OPTIONAL_CONVENTIONS = {'weight': 1.0}

#: The day counts the family's own row declares. Unlike `ir_curve.DAY_COUNTS` this is a passthrough
#: rather than an arithmetic: nothing here computes an accrual, `create_market_swaps` does, so the
#: whole declared list is admissible and a spelling outside it refuses at the seed.
DAY_COUNTS = ('ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA')


class ReferenceDataSource(Protocol):
    """The tolerant reader alone - one request carries the vol, both sides of the two-way and the
    print's own date, and the strict policy is applied HERE, per cell, by the screen."""

    def reference_data_report(self, securities: Sequence[str],
                              fields: Sequence[str]) -> Mapping[str, Mapping[str, object]]:
        ...


# ---------------------------------------------------------------------------------------------
# declared data
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SwaptionConventions:
    """What the forward swap under a quoted cell IS - read off the seed, never inferred."""
    fixed_frequency: str
    float_frequency: str
    fixed_day_count: str
    float_day_count: str
    #: `Normal` or `Lognormal` - what the terminal's number MEANS. See the module docstring: the
    #: family prices lognormal and reads no distribution, so this travels into `Quote_Source` and
    #: into the ladder rather than into a column the engine consults.
    distribution: str
    #: What multiplies the terminal's print to reach the `Percent` column's own number, which is in
    #: PERCENT: a `SASN` normal vol of 145 basis points is 1.45 percent, so ZAR declares 0.01. A
    #: percent-quoted lognormal family (`{CCY}SV`) would declare 1.0.
    quote_scale: float
    #: The objective weight every row carries. FLAT ONE is V1's declaration and it is a declaration
    #: rather than a default: a vega weight over a swaption grid needs the annuity, which needs the
    #: curve, which this emitter deliberately does not have. An unweighted least squares over ATM
    #: normal vols is at least a stated objective; a weight invented out here would not be.
    weight: float = OPTIONAL_CONVENTIONS['weight']

    def __post_init__(self):
        if self.distribution not in DISTRIBUTIONS:
            raise BloombergConfigurationError(
                'distribution {!r} is not one this emitter carries - a swaption grid is quoted '
                '{}, and which one a ticker prefix is cannot be read off the prefix. Fix the '
                'currency\'s swaption `conventions` in the seed'.format(
                    self.distribution, ' or '.join(DISTRIBUTIONS)))
        for name in ('fixed_day_count', 'float_day_count'):
            value = getattr(self, name)
            if value not in DAY_COUNTS:
                raise BloombergConfigurationError(
                    '{} {!r} is not a day count HullWhite2FactorModelParameters declares - the row '
                    'takes {}. Fix the currency\'s swaption `conventions` in the seed'.format(
                        name, value, ', '.join(DAY_COUNTS)))
        for name in ('fixed_frequency', 'float_frequency'):
            # POSITIVE, not merely readable - `create_market_swaps` rolls both legs with
            # `generate_dates_backward`, whose loop advances by the frequency it is given
            if read_tenor(getattr(self, name), name)[0] <= 0:
                raise BloombergConfigurationError(
                    '{} is {!r} - a leg frequency has to be a positive period, or the forward '
                    'swap\'s coupon schedule never advances'.format(name, getattr(self, name)))
        if not math.isfinite(self.quote_scale) or self.quote_scale == 0.0:
            raise BloombergConfigurationError('quote_scale must be finite and non-zero')
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise BloombergConfigurationError(
                'weight must be positive and finite - a benchmark weighing zero is a benchmark the '
                'objective cannot see, which is a row that should not have been emitted')


def swaption_conventions(seed: Mapping, currency: str) -> SwaptionConventions:
    """The declared swaption conventions of one seeded currency, or the refusal naming what is
    missing - `ir_curve.curve_conventions`' own shape, and the refusal lists every absent field at
    once for its reason."""
    spec = seed.get('swaption', {}).get(currency)
    if spec is None:
        raise BloombergConfigurationError(
            'the seed names no swaption entry for {} - a currency this workstation never seeded has '
            'no grid to fetch and no conventions to author one in. Add it to `seed.json` and re-run '
            '`DV_Bloomberg discover`'.format(currency))
    declared = spec.get('conventions')
    if not isinstance(declared, collections.abc.Mapping):
        raise BloombergConfigurationError(
            '{} carries no swaption `conventions` block - an ATM vol is not a benchmark until '
            'something says what forward swap it is a vol of, and what distribution the number is '
            'in. Declare {} on the {} swaption entry in your seed (see derivus_bloomberg/seed.json '
            'for the shipped ZAR declaration)'.format(
                currency, ', '.join(REQUIRED_CONVENTIONS), currency))
    missing = [name for name in REQUIRED_CONVENTIONS if declared.get(name) is None]
    if missing:
        raise BloombergConfigurationError(
            '{} declares no {} - the full set this emitter reads is {}, and a convention block '
            'filled in half way authors a benchmark nobody stated. Fix the {} swaption entry in '
            'your seed'.format(currency, ', '.join(missing), ', '.join(REQUIRED_CONVENTIONS),
                               currency))
    unknown = sorted(set(declared) - set(REQUIRED_CONVENTIONS) - set(OPTIONAL_CONVENTIONS))
    if unknown:
        raise BloombergConfigurationError(
            '{} declares {} which this emitter reads nothing of - a convention nobody reads is a '
            'convention that is not applied. Remove it, or spell it as one of {}'.format(
                currency, ', '.join(unknown),
                ', '.join(sorted(set(REQUIRED_CONVENTIONS) | set(OPTIONAL_CONVENTIONS)))))
    return SwaptionConventions(**{name: declared[name] for name in REQUIRED_CONVENTIONS},
                               **{name: declared[name] for name in OPTIONAL_CONVENTIONS
                                  if name in declared})


@dataclass(frozen=True)
class SwaptionScreen:
    """The screens a quoted cell is held to, every one of them a parameter with a stated default."""
    #: `discover.STALE_DAYS`, for its reason - and a swaption grid is where it bites hardest: the
    #: back corners of a ragged grid go days without a print while the front keeps ticking.
    stale_days: int = discover.STALE_DAYS
    #: The band a vol is believed inside, in the EMITTED percent units - so a 145bp normal vol reads
    #: as 1.45 and a 20% lognormal as 20.0. One band over both distributions because it is a
    #: sanity bound and not a market view: a hundred percent admits every swaption anyone has
    #: quoted and refuses a decimal-shifted print.
    vol_band: tuple = (0.0, 100.0)
    #: Cells the ladder must survive screening with. FIVE is the family's own parameter count -
    #: sigma_1, sigma_2, alpha_1, alpha_2, rho - so below it the fit is interpolating and reports a
    #: stationarity it did not earn. It is a FLOOR and not a sufficiency: J has 23 columns (two mean
    #: reversions, a correlation and two ten-knot sigma term structures), so a grid under 23 cells
    #: leaves a null space no linear algebra can invent, which `quote_sensitivities.md` says is the
    #: problem rather than a defect. The emitter reports the count; it refuses only under the floor.
    minimum_rows: int = 5

    def __post_init__(self):
        object.__setattr__(self, 'vol_band', tuple(self.vol_band))
        if len(self.vol_band) != 2 or not all(math.isfinite(edge) for edge in self.vol_band) \
                or not self.vol_band[0] < self.vol_band[1]:
            raise BloombergConfigurationError('vol_band must be (low, high) with low < high')
        if self.minimum_rows < 1:
            raise BloombergConfigurationError('minimum_rows must be at least one')


# ---------------------------------------------------------------------------------------------
# what a ladder is
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SwaptionQuote:
    """One cell of the grid as the terminal answered for it - raw, unjudged, not believed."""
    expiry: str
    tenor: str
    security: str
    value: float | None
    bid: float | None = None
    ask: float | None = None
    last_update: str | None = None


@dataclass(frozen=True)
class SwaptionLadder:
    """A screened grid: what survived, what did not and why, and the surface it is quoted for.

    `rejected` is the LEDGER, `{security: verdict}`, `discover.build_map`'s discipline transferred:
    a ragged grid is the ordinary case here - entitlements differ by cell - so what was refused and
    why is most of what a desk needs to read.
    """
    currency: str
    surface: str
    as_of: datetime.date
    conventions: SwaptionConventions
    quotes: tuple
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

def swaption_entries(document, seed, currency):
    """`(wanted, ledger)` - the verified cells of one currency's grid, walked off the GRAMMAR.

    `discover.swaption_candidates` is asked for its candidates and each one's `path` is looked up in
    the workstation's own map, so this module spells no ticker: the two-character expiry code and
    the unpadded tenor (SASN011 is 1Y into 1Y) stay in `discover` where they were verified. A cell
    the map did not verify is `unverified` on the ledger by name - which on a per-entitlement grid
    is the normal shape rather than an alarm.

    `wanted` is `[(expiry label, tenor label, security)]`, both labels being PERIODS the block
    carries straight into `Start` and `Tenor`: the map path a candidate carries is spelled
    `'{expiry} x {tenor}Y'` by the grammar, so the two halves are read back off it rather than
    re-derived from the seed's nested loops.
    """
    swaption_conventions(seed, currency)
    blocks = document.get('blocks', {})
    wanted, ledger = [], {}
    for candidate in discover.swaption_candidates(currency, seed['swaption'][currency]):
        entry = blocks
        for part in candidate.path:
            entry = entry.get(part) if isinstance(entry, collections.abc.Mapping) else None
            if entry is None:
                break
        security = entry.get('security') if isinstance(entry, collections.abc.Mapping) else None
        if security is None:
            ledger[candidate.security] = 'unverified'
            continue
        wanted.append(_cell(candidate) + (security,))
    return tuple(wanted), ledger


def _cell(candidate):
    """`(expiry label, tenor label)` off a candidate's map path - the grammar's own
    `'{expiry} x {tenor}Y'`. An unreadable one refuses rather than being skipped: a cell dropped for
    being unparseable is indistinguishable from one the terminal never answered."""
    parts = str(candidate.path[-1]).split(' x ')
    if len(parts) != 2:
        raise BloombergConfigurationError(
            '{} is mapped at {!r}, which is not the `<expiry> x <tenor>` a swaption candidate is '
            'pathed with - the map was written by a different grammar than this one reads'.format(
                candidate.security, candidate.path[-1]))
    expiry, tenor = (part.strip() for part in parts)
    read_tenor(expiry, 'swaption expiry')
    read_tenor(tenor, 'swap tenor')
    return expiry, tenor


def fetch_swaption_ladder(source, document, seed, currency, as_of, surface=None, screen=None,
                          batch=BATCH, on_batch=None):
    """One currency's verified ATM grid, screened - `SwaptionLadder`.

    ONE ROUND TRIP over the securities the map believed, in `discover.BATCH`-sized chunks, asking
    each the vol, both sides of its two-way and its own last print. The tolerant reader makes the
    request and the strict policy is applied CLIENT-SIDE, per cell: on a ragged grid one refused
    ticker is the FINDING and not the failure.
    """
    screen = screen or SwaptionScreen()
    conventions = swaption_conventions(seed, currency)
    wanted, ledger = swaption_entries(document, seed, currency)
    report = probe(source, [security for _, _, security in wanted], fields=QUOTE_FIELDS,
                   batch=batch, on_batch=on_batch)

    quotes, rejected = [], dict(ledger)
    for expiry, tenor, security in wanted:
        row = report.get(security, {'ok': False, 'error': 'no answer in the response', 'fields': {}})
        answered = row.get('fields') or {}
        if not answered:
            rejected[security] = 'invalid'
            continue
        quotes.append(SwaptionQuote(
            expiry=expiry, tenor=tenor, security=security,
            value=_scaled(answered.get('PX_LAST'), conventions.quote_scale),
            bid=_scaled(answered.get('PX_BID'), conventions.quote_scale),
            ask=_scaled(answered.get('PX_ASK'), conventions.quote_scale),
            last_update=read_word(answered.get('LAST_UPDATE_DT')) or None))
    accepted, screened = screen_ladder(quotes, as_of, screen)
    rejected.update(screened)
    return SwaptionLadder(currency=currency, surface=surface or currency, as_of=as_of,
                          conventions=conventions, quotes=accepted, rejected=rejected)


def _scaled(value, quote_scale):
    number = read_number(value)
    return None if number is None else number * quote_scale


def screen_ladder(quotes, as_of, screen=None):
    """`(accepted, rejected)` - the trust boundary, in the ORDER OF DISTRUST.

      unpriced   no PX_LAST at all
      zero       a vol of exactly zero. This screen was built when it was a TRAP rather than a
                 sanity check - `create_market_swaps`'s rule was
                 `if instrument['Market_Volatility'].amount:`, so a zero column silently meant "read
                 the swaption surface's ATM instead" and a blank cell emitted as zero calibrated
                 against whatever the book's surface happened to hold, under the name of a quote the
                 terminal never gave. The engine refuses that row by name since 2026-09-01; this
                 screen stays where it is, because a ladder is refused HERE where a desk can see
                 which cell went dark, rather than at the far end of a book
      off-market a vol outside the declared band - the decimal-shifted print no other check sees
      crossed    bid above ask: a stale side standing against a live one
      undated    no readable LAST_UPDATE_DT - a print that cannot evidence its own time
      stale      a last update older than `stale_days`; on a ragged grid the back corners are where
                 this fires, and they are exactly the cells that identify the long sigma knots
      live       believed

    `rejected` is `{security: verdict}` for every one of them, BY NAME.
    """
    screen = screen or SwaptionScreen()
    accepted, rejected = [], {}
    for quote in sorted(quotes, key=lambda item: item.security):
        verdict = _verdict(quote, as_of, screen)
        if verdict == 'live':
            accepted.append(quote)
        else:
            rejected[quote.security] = verdict
    return tuple(accepted), rejected


def _verdict(quote, as_of, screen):
    if quote.value is None:
        return 'unpriced'
    if quote.value == 0.0:
        return 'zero'
    low, high = screen.vol_band
    if not low <= quote.value <= high:
        return 'off-market'
    if quote.bid is not None and quote.ask is not None and quote.bid > quote.ask:
        return 'crossed'
    stamp = read_date(quote.last_update)
    if stamp is None:
        return 'undated'
    if (as_of - stamp).days > screen.stale_days:
        return 'stale'
    return 'live'


# ---------------------------------------------------------------------------------------------
# the block
# ---------------------------------------------------------------------------------------------

def market_price_name(curve):
    """`HullWhite2FactorModelPrices.<curve>` - the `Market Prices` key. Its tail is the
    `InterestRate` curve the calibration diffuses, which is what `bootstrap` resolves the process
    and the benchmark swaps off; the written parameters land at
    `HullWhite2FactorModelParameters.<curve>`."""
    return '{}.{}'.format(FAMILY, curve)


def _period_years(label):
    count, unit = read_tenor(label)
    return count * {'D': 1.0 / 365.0, 'W': 7.0 / 365.0, 'M': 1.0 / 12.0, 'Y': 1.0}[unit]


def instrument_row(quote, conventions):
    """One `Instrument_Definitions` row - the eight declared columns, then the evidence.

    `Start` and `Tenor` are the grid's own coordinates as `Period`s, which is what the family
    declares them as: `create_market_swaps` does `effective = base_date + instrument['Start']` and
    `maturity = effective + instrument['Tenor']`, so a 1Y x 10Y cell is a ten-year swap starting in
    a year and no date is computed out here at all.

    `Market_Volatility` is a `Percent`, so the wire number is in PERCENT and the decoded `.amount`
    is the fraction the pricer reads. The distribution that number is IN is declared on the ladder
    and stated in the block's `Quote_Source`; there is no ROW column for it and there is not meant
    to be - the convention is a property of the surface the block names, which is where
    `create_market_swaps` reads it.
    """
    row = {
        'Start': wire_period(quote.expiry),
        'Tenor': wire_period(quote.tenor),
        'Floating_Frequency': wire_period(conventions.float_frequency),
        'Fixed_Frequency': wire_period(conventions.fixed_frequency),
        'Floating_Day_Count': conventions.float_day_count,
        'Fixed_Day_Count': conventions.fixed_day_count,
        'Market_Volatility': wire_percent(quote.value),
        'Weight': conventions.weight,
    }
    if quote.bid is not None:
        row['Quoted_Bid'] = quote.bid
    if quote.ask is not None:
        row['Quoted_Ask'] = quote.ask
    row['Timestamp'] = wire_timestamp(read_date(quote.last_update))
    return row


def hw2f_block(ladder, curve=None, screen=None):
    """`(Market Prices name, block)` - one verified grid as ONE `HullWhite2FactorModelPrices` block.

    THE ROWS ARE ORDERED BY THE GRID, expiry then tenor, so the block reads as the ladder a desk
    would look at and two emissions off the same answers are the same bytes.

    WHAT IS WRITTEN AND WHAT IS NOT. `Swaption_Volatility` and `Instrument_Definitions` are the
    quote; every other declared field on this family - `Objective`, `Simulations`, `Batches`,
    `Random_Seed`, `Quote_Sensitivity`, `Jacobian_Rcond`, `Stationarity_Tol` - is a property of the
    SOLVE rather than of the market, and each is read by the engine with its declared default. An
    emitter that stated them would be deciding a job's optimizer from a market-data fetch. (The
    Heston-Nandi emitter states `Steps_Per_Year` for the opposite reason: there the step clock is
    what the fitted parameters MEAN.)

    SO AN EMITTED LADDER FOLLOWS THE FAMILY'S DEFAULT WHEREVER IT GOES, which is the point of not
    stating it and is worth saying once: `Objective` flipped to `Analytic` on 2026-08-31, so a grid
    fetched here now solves through Schrager-Pelsser rather than through the simulation. Nothing
    about this emission moved - the bytes are the same bytes - and the change is entirely the
    engine's, which is the behaviour a field left to its default is asking for.

    THE SEED'S `distribution: Normal` NOW HAS AN ENGINE THAT HONOURS IT, and that is the second such
    change and the one this emitter was waiting for. `SASN` is a normal vol in basis points, the
    seed has declared it as one since this package shipped, and `create_market_swaps` priced every
    benchmark with a lognormal Black whatever the surface said - so an emitted ZAR ladder could be
    fetched, screened and booked honestly and still be fitted under the wrong convention. Since
    2026-09-01 that calibration reads the named `InterestYieldVol`'s `Distribution_Type` and prices
    a Normal ladder with Bachelier. NOTHING HERE MOVED FOR IT except the `Quote_Source` note, which
    said the opposite: what makes the fit honest is the SURFACE's declaration, which this emitter
    does not author - so a desk pointing `Swaption_Volatility` at a lognormally-declared factor
    still gets a lognormal fit of normal quotes, and the block's own line is where that is stated.

    `Quote_Source` IS NOW A DECLARED FIELD of this family, and that is the finding this emitter
    raised, closed on 2026-09-01. It used to ride as an undeclared key that `bootstrap` read past
    and `as_json` preserved - both Heston-Nandi families declared `Quote_Source` and
    `Quote_Timestamp` and this one declared neither, so a machine-fetched ladder had nowhere inside
    its own block to say where it came from. HW2F declares both now, on their shape. NOTHING HERE
    MOVED FOR IT EITHER: the same key, the same line, the same bytes - what changed is that a
    schema-driven front end can see it, and a reader of the declarations can find out that this is
    the only place a block says which convention its numbers are in.
    """
    screen = screen or SwaptionScreen()
    if not ladder.surface:
        raise BloombergConfigurationError(
            'the block names no Swaption_Volatility surface - the family declares it REQUIRED and '
            'the calibration reads that surface\'s Distribution_Type to know which convention to '
            'price these quotes in, so an unnamed surface is a calibration that resolves nothing. '
            'Name the InterestYieldVol factor')
    if len(ladder.quotes) < screen.minimum_rows:
        raise IncompleteLadder(
            '{} screened to {} believed cell{} against a floor of {} - the terminal was asked about '
            '{} securities and refused {} ({}). The family fits five parameters (sigma_1, sigma_2, '
            'alpha_1, alpha_2, rho), so under that floor the fit interpolates and reports a '
            'stationarity it did not earn. Widen the screen the census names, re-run '
            '`DV_Bloomberg discover` if the grid has gone dead, or quote a currency this '
            'workstation is entitled to'.format(
                ladder.currency, len(ladder.quotes), '' if len(ladder.quotes) == 1 else 's',
                screen.minimum_rows, len(ladder.quotes) + len(ladder.rejected),
                len(ladder.rejected),
                ', '.join('{} {}'.format(count, verdict)
                          for verdict, count in sorted(ladder.census.items()))
                or 'nothing refused'))

    rows = [instrument_row(quote, ladder.conventions) for quote in sorted(
        ladder.quotes, key=lambda item: (_period_years(item.expiry), _period_years(item.tenor),
                                         item.security))]
    return market_price_name(curve or ladder.currency), {'instrument': {
        'Swaption_Volatility': ladder.surface,
        'Quote_Source': quote_source(ladder),
        'Instrument_Definitions': rows}}


def quote_source(ladder):
    """The block's own account of where its quotes came from, in one line beside the parameters they
    produce - `equity_chain.quote_source`'s job, on a family that declares no field for it.

    THE DISTRIBUTION IS THE FIRST THING IT SAYS, because it is the one thing about this block a
    reader cannot recover from the numbers: 1.45 in the `Market_Volatility` column is an ordinary
    lognormal vol read one way and a 145 basis point normal vol read the other. Since 2026-09-01
    `create_market_swaps` reads the surface's own `Distribution_Type` and prices a Normal ladder
    with Bachelier, so this line and the engine now agree - and it still has to be SAID here,
    because the block itself declares no field to say it in and the declaration it depends on lives
    on a factor this emitter does not author.
    """
    census = ', '.join('{} {}'.format(count, verdict)
                       for verdict, count in sorted(ladder.census.items())) or 'nothing refused'
    return (
        '{} ATM cells quoted as {} vols off the {} swaption grid as at {}, {} believed of {} asked '
        '({}); the forward swaps are {}/{} {} vs {}, weighted flat at {:g}. NOTE: '
        'create_market_swaps prices each benchmark in the convention the named surface DECLARES - '
        'Normal vols reach a Bachelier premium since 2026-09-01 - so this ladder is priced as it '
        'is quoted only where that InterestYieldVol declares Distribution_Type {}'.format(
            len(ladder.quotes), ladder.conventions.distribution.upper(), ladder.currency,
            ladder.as_of.isoformat(), len(ladder.quotes),
            len(ladder.quotes) + len(ladder.rejected), census,
            ladder.conventions.fixed_frequency, ladder.conventions.float_frequency,
            ladder.conventions.fixed_day_count, ladder.conventions.float_day_count,
            ladder.conventions.weight, ladder.conventions.distribution))


def quote_census(ladder):
    """The ladder's own account of what the terminal served, for a caller with a screen or a report
    - the same numbers `quote_source` renders into the block, as data."""
    return {'currency': ladder.currency, 'surface': ladder.surface,
            'as_of': ladder.as_of.isoformat(), 'distribution': ladder.conventions.distribution,
            'asked': len(ladder.quotes) + len(ladder.rejected), 'believed': len(ladder.quotes),
            'refused': dict(ladder.rejected), 'census': ladder.census,
            'cells': ['{} x {}'.format(quote.expiry, quote.tenor) for quote in ladder.quotes]}
