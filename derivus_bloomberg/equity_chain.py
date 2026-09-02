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
"""The listed equity option CHAIN, read off a terminal and emitted as a Heston-Nandi quote block.

Equities calibrate TO THE CHAIN and quote PREMIUMS rather than implied vols: a listed price is a
print while its implied vol is a convention (which forward, which discounting, which exercise), and
any equity vol surface is already somebody's fit. The Heston-Nandi families accept `Quote_Type`
Premium, which is what this emitter writes.

It is `fxvol`'s sibling - reach a terminal, screen what came back, write a `{"instrument": {...}}`
block - and it prices nothing, fits nothing, builds no surface, and imports no engine nor pandas,
which is why `__init__` re-exports `fxvol` and `types` lazily.

EVERY FOREIGN ANSWER IS UNTRUSTED. Half of any index chain is dead strikes: contracts that resolve
and carry a plausible `PX_LAST` but have not traded in a month, or are quoted one-sided, crossed,
or at zero open interest. `screen_chain` classifies every contract in an order of distrust and
ledgers every refusal by name. AMERICAN EXERCISE REFUSES BY NAME (`UnsupportedExerciseStyle`) - v1
is indices only - and a contract stating no exercise style is refused too, absence not proving
European.

THE LADDER REACHES THE PRODUCT HORIZON: pillars default to 3M/6M/1Y/2Y/3Y where the FX ladder stops
at 1Y, because equity autocalls run three to five years and one omega cannot hold a multi-year ATM
term structure. The target family is therefore `HestonNandiComponentModelPrices`; the plain
spelling is emitted from the same selection, the two blocks differing only in their header.

THE FORWARD IS DECLARED, NOT DISCOVERED - the strikes and every weight hang off it, so an emitter
forward disagreeing with the pricer's fits the calibration at coordinates the pricer never visits.
`EquityForward` names the references the fit resolves and the numbers the emitter placed its
strikes with, and both travel into the block, the curve that GROWS the forward and the curve the
PREMIUM discounts on named separately. The chain's own parity-implied dividend yield is reported
beside the declared one, never averaged in.
"""
import collections.abc
import datetime
import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .errors import (BloombergConfigurationError, IncompleteChain, InvalidQuote,
                     UnsupportedExerciseStyle, raise_response_error)

NORMAL = statistics.NormalDist()

#: What the terminal is asked about the UNDERLYING: what it is, what it is worth, when it last
#: printed - `discover.FIELDS` exactly. All three are READ: the spot refuses on a blank, and its
#: `LAST_UPDATE_DT` is screened against `stale_days` as every listed contract's is.
UNDERLYING_FIELDS = ('NAME', 'PX_LAST', 'LAST_UPDATE_DT')

#: The BULK field naming a listed chain's members. Read through
#: `BloombergSession.bulk_reference_data_report`, never the scalar reader - `getValue()` on an
#: array element answers row ZERO and says nothing about the two thousand it dropped.
CHAIN_FIELD = 'OPT_CHAIN'

#: The sub-field of an `OPT_CHAIN` row that carries the member's ticker. Tried first; a row that
#: does not carry it falls back to its single string value, and a row with neither is ledgered.
CHAIN_MEMBER_FIELD = 'Security Description'

#: What every candidate contract is asked. The first four are the CONTRACT (what it is), the next
#: three are the PRICE (what it is worth, and from which side), and the last three are the
#: EVIDENCE that any of it still means anything.
CONTRACT_FIELDS = ('OPT_STRIKE_PX', 'OPT_EXPIRE_DT', 'OPT_PUT_CALL', 'OPT_EXER_TYP',
                   'PX_BID', 'PX_ASK', 'PX_LAST', 'OPEN_INT', 'VOLUME', 'LAST_UPDATE_DT')

#: One request per chunk - `discover.BATCH`, and for its reason: a bounded request keeps a partial
#: outage partial. A full SPX chain is thousands of names.
BATCH = 50

#: The five reference fields the Heston-Nandi families declare, and the factor TYPE this emitter
#: names each with for an equity underlying. Spelled here because the package may not import the
#: engine; held against `HestonNandiModelParameters.factor_types` by a gate.
HN_REFERENCE_TYPES = {'Underlying': 'EquityPrice', 'Volatility': 'EquityPriceVol',
                      'Discount_Rate': 'InterestRate', 'Yield': 'DividendRate',
                      'Funding_Rate': 'InterestRate'}

#: The two spellings of the target family, emitted off ONE selection: the component family for the
#: multi-year ATM term structure its L curve holds, the plain one for the five-parameter fit.
COMPONENT_FAMILY = 'HestonNandiComponentModelPrices'
PLAIN_FAMILY = 'HestonNandiModelPrices'
FAMILIES = (COMPONENT_FAMILY, PLAIN_FAMILY)

#: The two switches the emitter STATES rather than lets fall through, at the families' own
#: declared defaults. The STEP CLOCK is what the fitted parameters mean, so a deal's
#: `Steps_Per_Year` has to be this number or it simulates a different model.
STEPS_PER_YEAR = 252.0
QUADRATURE_PANELS = 64

#: The extra header the COMPONENT family declares and the plain one does not, at that family's own
#: declared defaults. `Rho` is a pin the block has to state; `Quote_Sensitivity` Yes is REFUSED by
#: this family.
COMPONENT_HEADER = {'Rho': 0.99, 'Quote_Sensitivity': 'No'}

#: The value-plane keys an option quote row carries beside its mid - the two-way the print was
#: dealt on and its own clock, which are the evidence.
#:
#: The Heston-Nandi families declare all three on their option row (`schema.QUOTE_TWO_WAY`), so
#: `European_Options` is a `schema.MARKET_QUOTE_CONTAINERS` table and these travel the VALUE plane
#: as an FX smile's `Points` do: a re-quoted chain on the same contracts moves `values_hash` with
#: `plan_hash` bit-identical. A moved strike or expiry is still a re-authoring.
QUOTE_VALUE_KEYS = ('Quoted_Bid', 'Quoted_Ask', 'Timestamp')

#: How many two-sided strikes the chain's own parity carry is MEDIANED over, and the band that
#: carry is believed inside where the caller declared none.
#:
#: Five and a median, because one pair is one print: a single fat-fingered near-money quote passing
#: every per-contract screen moves a parity-read carry by percentage points, and the carry moves
#: the pillar's whole forward. -5% to +15% covers a hard-to-borrow basket quoting through zero and
#: an emerging index quoting a full year of dividends, and nothing beyond a bad chain.
PARITY_STRIKES = 5
PARITY_BAND = (-0.05, 0.15)


# ---------------------------------------------------------------------------------------------
# what a chain is
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainContract:
    """One listed option as the terminal answered for it - raw, unjudged, and NOT yet believed.

    The two-way, the last print, the open interest and the last-update date are all OPTIONAL: a
    dead strike answers half of them, and saying which half is the screen's job.
    """
    security: str
    strike: float | None
    expiry: datetime.date | None
    option_type: str
    exercise: str
    bid: float | None
    ask: float | None
    last: float | None
    open_interest: float | None
    volume: float | None
    last_update: str | None

    @property
    def mid(self) -> float | None:
        """The two-way's midpoint, or the last print where the terminal quoted no two-way. The
        screen refuses a one-sided contract anyway; the fallback is what lets it tell `unpriced`
        and `one-sided` apart."""
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        return self.last

    @property
    def spread(self) -> float | None:
        """The quoted spread as a FRACTION OF MID, or None where there is no two-way to read it
        off. Both the screen's cap and the rung's weight are read off this number."""
        mid = self.mid
        if self.bid is None or self.ask is None or not mid or mid <= 0.0:
            return None
        return (self.ask - self.bid) / mid


@dataclass(frozen=True)
class EquityChain:
    """A screened chain: what survived, what did not and why, and the underlying it hangs off.

    `rejected` is the LEDGER - `{security: verdict}` for every candidate that did not make it,
    because a candidate silently dropped is indistinguishable from one never asked about.
    """
    underlying: str
    name: str
    spot: float
    as_of: datetime.date
    contracts: tuple[ChainContract, ...]
    rejected: Mapping[str, str]
    #: WHEN THE SPOT ITSELF LAST PRINTED, and it travels into `Quote_Source` beside the spot.
    #: `fetch_equity_chain` screens it against `stale_days` and refuses by name, so a fetched chain
    #: always carries one; optional only because the selection is drivable on a hand-built chain.
    spot_as_of: datetime.date | None = None

    @property
    def expiries(self) -> tuple[datetime.date, ...]:
        """The expiries the SURVIVING contracts carry, sorted - what a refusal names."""
        return tuple(sorted({contract.expiry for contract in self.contracts}))


@dataclass(frozen=True)
class EquityForward:
    """WHICH CURVE FEEDS THE CARRY, declared.

    The strikes and every weight hang off the forward, and the FIT rebuilds it from the curves this
    names: `spot * exp((f - q) t)`, `f` the funding curve and `q` the `Yield` one. So the NAMES
    travel into the block for the fit to resolve, and the NUMBERS the emitter used travel into
    `Quote_Source` beside them.

    TWO CURVES, TWO JOBS. `funding_rate` names what GROWS the forward - the equity's own repo curve,
    what `utils.calc_eq_forward` integrates - and `discount_rate` what the PREMIUM discounts on.
    Left blank, `funding_rate` is `discount_rate`, the index with no repo spread.

    `rate` is the FORWARD's number, the funding rate the emitter carried the spot at, which places
    every strike. Where a repo spread is declared the parity read and the vega discount at the same
    number, biasing a chain-implied carry by the spread; declare `dividend_yield` and the reading
    is reported rather than fitted. `dividend_yield` None takes it from the CHAIN by put-call
    parity.
    """
    underlying_factor: str
    volatility_factor: str
    discount_rate: str
    dividend_reference: str
    rate: float
    dividend_yield: float | None = None
    funding_rate: str = ''
    underlying_type: str = HN_REFERENCE_TYPES['Underlying']
    volatility_type: str = HN_REFERENCE_TYPES['Volatility']
    discount_rate_type: str = HN_REFERENCE_TYPES['Discount_Rate']
    dividend_type: str = HN_REFERENCE_TYPES['Yield']
    funding_rate_type: str = HN_REFERENCE_TYPES['Funding_Rate']

    def __post_init__(self):
        for name, value in (('underlying_factor', self.underlying_factor),
                            ('volatility_factor', self.volatility_factor),
                            ('discount_rate', self.discount_rate),
                            ('dividend_reference', self.dividend_reference)):
            if not value:
                raise BloombergConfigurationError(
                    'EquityForward.{} is blank - the block names its inputs BY NAME and the fit '
                    'resolves them out of the book\'s Price Factors, so an unnamed reference is a '
                    'block that skips its own calibration with one line in a log. Name the '
                    'factor'.format(name))
        if not math.isfinite(self.rate):
            raise BloombergConfigurationError('EquityForward.rate must be finite')
        if self.dividend_yield is not None and not math.isfinite(self.dividend_yield):
            raise BloombergConfigurationError('EquityForward.dividend_yield must be finite or None')


@dataclass(frozen=True)
class EquityLadder:
    """The ladder, its screens and its floors - every one of them a parameter with a stated default.

    THE PILLARS REACH THE PRODUCT HORIZON: 3M/6M/1Y/2Y/3Y, where the FX ladder stops at 1Y,
    because equity autocalls run three to five years and one omega cannot hold a multi-year ATM
    term structure. The component family's L curve can, and the pillars are what identify it.

    THE WINGS ARE ONE PILLAR SHORT of the ATM ladder, four expiries as
    `HestonNandiComponentModelParameters` wants: the ATM rungs are SPENT on the L pillars, so what
    identifies the five free globals is what is left. The 3Y wing is dropped rather than the 3M
    one because that is where a listed chain thins out first.
    """
    pillars: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0)
    wing_pillars: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    #: The wing's delta, placed as a MONEYNESS BAND off the forward under the chain's own ATM
    #: implied vol (see `select_rungs`). A listed chain has no delta axis to read, so the pillar is
    #: a target the rung then SNAPS off - it names where to look, never what was quoted.
    wing_delta: float = 0.25
    #: The quoted spread, as a fraction of mid, past which a print is not a market. A quarter of
    #: mid is wide for an index ATM and ordinary for a far wing, so it is one cap over the whole
    #: chain; it ALSO enters the weight, halving a contract that barely survived against a locked one.
    spread_cap: float = 0.25
    #: OI > 0. Zero open interest is a listed contract nobody holds.
    minimum_open_interest: float = 1.0
    #: `discover.STALE_DAYS`, for its reason: a strike whose last update is three weeks old still
    #: answers a plausible price, and the date is the only thing that says otherwise.
    stale_days: int = 5
    #: DISTINCT contracts the ladder must survive snapping with -
    #: `HestonNandiComponentModelParameters.fx_minimum_contracts`: the ATM rungs are consumed by
    #: the L bootstrap, so what identifies the globals is what is left.
    minimum_contracts: int = 8
    #: How far PAST the ladder's longest rung a listed expiry may still be snapped to. A month,
    #: because that is the width of the same quarterly listing rolled once - not a second expiry.
    expiry_tolerance: float = 31.0 / 365.0
    #: How far a rung may MOVE, in log-expiry, before it is dropped instead of snapped. On a ladder
    #: spanning 3M to 3Y an unconditional argmin would land the 3Y rung on a stub chain's 3M
    #: listing. Half a log-unit is the 1Y rung reaching 6M or 1.6Y.
    pillar_band: float = 0.5
    #: The day count the emitter measures its OWN year fractions in, for placing strikes and
    #: weighting. The FIT recomputes `t` off `Expiry_Date` through the discount curve's day count:
    #: the two agree exactly under ACT_365 and differ by a day count otherwise, which moves the
    #: WEIGHT and never the contract, since the contract was snapped to a listing rather than
    #: computed.
    days_per_year: float = 365.0
    #: The step clock and the inversion width the block STATES. Read off the field declarations.
    steps_per_year: float = STEPS_PER_YEAR
    quadrature_panels: int = QUADRATURE_PANELS
    #: The flat vol the wing bands fall back to where an expiry's own ATM contract yields no
    #: admissible Black implied vol. A seed for a coordinate, never a price.
    reference_vol: float = 0.20
    #: How many two-sided strikes the chain's own parity carry is MEDIANED over, and the band it is
    #: BELIEVED inside where the caller declared no dividend yield - see `PARITY_STRIKES` /
    #: `PARITY_BAND`. An undeclared carry is evidence like any other and is screened like it.
    parity_strikes: int = PARITY_STRIKES
    parity_band: tuple[float, float] = PARITY_BAND

    def __post_init__(self):
        # coerced to tuples, so a caller who hands in a list gets a frozen ladder rather than a
        # frozen handle on something still mutable
        object.__setattr__(self, 'pillars', tuple(self.pillars))
        object.__setattr__(self, 'wing_pillars', tuple(self.wing_pillars))
        if not self.pillars:
            raise BloombergConfigurationError('the ladder names no pillars')
        if len(set(self.pillars)) != len(self.pillars):
            raise BloombergConfigurationError('ladder pillars must be unique')
        if any(not math.isfinite(pillar) or pillar <= 0.0 for pillar in self.pillars):
            raise BloombergConfigurationError('ladder pillars must be positive and finite')
        extra = sorted(set(self.wing_pillars) - set(self.pillars))
        if extra:
            raise BloombergConfigurationError(
                'the wing pillars {} are not ATM pillars - a wing is placed off its own expiry\'s '
                'ATM implied vol, so a wing with no ATM rung beneath it has nothing to be a wing '
                'of'.format(extra))
        if not 0.0 < self.wing_delta < 0.5:
            raise BloombergConfigurationError('wing_delta must be strictly between 0 and 0.5')
        if not self.spread_cap > 0.0:
            raise BloombergConfigurationError('spread_cap must be positive')
        if self.minimum_contracts < 1:
            raise BloombergConfigurationError('minimum_contracts must be at least one')
        object.__setattr__(self, 'parity_band', tuple(self.parity_band))
        if self.parity_strikes < 1:
            raise BloombergConfigurationError('parity_strikes must be at least one')
        if len(self.parity_band) != 2 or not all(
                math.isfinite(edge) for edge in self.parity_band) \
                or not self.parity_band[0] < self.parity_band[1]:
            raise BloombergConfigurationError(
                'parity_band must be (low, high) with low < high - it is the band an UNDECLARED '
                'dividend yield read off the chain is believed inside, and a band that is not one '
                'would either refuse every chain or screen nothing')


@dataclass(frozen=True)
class Rung:
    """One snapped rung: the listed contract a pillar landed on, and everything the block needs."""
    kind: str
    pillar: float
    contract: ChainContract
    tau: float
    forward: float
    implied_vol: float
    vega: float
    weight: float


# ---------------------------------------------------------------------------------------------
# the Black arithmetic - a WEIGHT, never a price
# ---------------------------------------------------------------------------------------------

def black_price(forward, strike, rate, vol, tau, is_call):
    """The Black-76 value of one unit. Used to INVERT a quoted premium into an implied vol and to
    place a wing coordinate - never to value anything. The premium in the block is the terminal's
    own print, and nothing here ever touches it."""
    if tau <= 0.0 or strike <= 0.0 or forward <= 0.0:
        return 0.0
    discount = math.exp(-rate * tau)
    if vol <= 0.0:
        return discount * max(forward - strike, 0.0) if is_call else \
            discount * max(strike - forward, 0.0)
    stddev = vol * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * stddev * stddev) / stddev
    d2 = d1 - stddev
    if is_call:
        return discount * (forward * NORMAL.cdf(d1) - strike * NORMAL.cdf(d2))
    return discount * (strike * NORMAL.cdf(-d2) - forward * NORMAL.cdf(-d1))


def black_vega(forward, strike, rate, vol, tau):
    """`exp(-r t) F n(d1) sqrt(t)`, shared by puts and calls - `HestonNandiModelParameters
    .fx_black_vega`.

    THE WEIGHT, before the liquidity factor and before normalisation. It is what makes the
    objective scale-free across a term structure running to three years: a 3M premium is a
    fraction of a 3Y one, and an unweighted least squares would fit the back end alone.
    """
    if tau <= 0.0 or vol <= 0.0 or forward <= 0.0 or strike <= 0.0:
        return 0.0
    stddev = vol * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * stddev * stddev) / stddev
    return math.exp(-rate * tau) * forward * NORMAL.pdf(d1) * math.sqrt(tau)


def implied_vol(price, forward, strike, rate, tau, is_call, iterations=100,
                low=1e-6, high=5.0):
    """The Black vol of a quoted premium, or None where the premium is not one.

    Bisection over `[low, high]`, the price being monotone in the vol. A bracket that does not
    straddle the price is the ANSWER rather than a starting point - a mid below intrinsic is not a
    premium any vol reaches - and the caller falls back to the ladder's flat proxy, which a WEIGHT
    may do and a price may not.
    """
    if tau <= 0.0 or price is None or price <= 0.0 or forward <= 0.0 or strike <= 0.0:
        return None
    if black_price(forward, strike, rate, low, tau, is_call) > price:
        return None
    if black_price(forward, strike, rate, high, tau, is_call) < price:
        return None
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if black_price(forward, strike, rate, middle, tau, is_call) < price:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------------------------
# reading the terminal
# ---------------------------------------------------------------------------------------------

class ChainDataSource(Protocol):
    """The two readers a chain needs: the BULK one for the chain's own membership, the scalar one
    for everything asked of a member. Both are `BloombergSession`'s and both are TOLERANT - on a
    chain of two thousand names a refused ticker is a finding, and the screen applies the policy."""

    def bulk_reference_data_report(self, securities: Sequence[str],
                                   fields: Sequence[str]) -> Mapping[str, Mapping[str, object]]:
        ...

    def reference_data_report(self, securities: Sequence[str],
                              fields: Sequence[str]) -> Mapping[str, Mapping[str, object]]:
        ...


def _number(value):
    """A terminal value as a finite float, or None. Absent, blank, unparseable and non-finite all
    read as ABSENT rather than as a number manufactured out of a blank."""
    if value is None or value == '':
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _date(value):
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


def _word(value):
    return '' if value in (None, '') else str(value).strip()


def _exercise(value):
    """The exercise style, normalised to `European`/`American`/`''`. Bloomberg spells it several
    ways ('European', 'EUROPEAN', 'Euro'); anything unrecognised is the EMPTY string, which the
    screen refuses - an unrecognised style is not a European one."""
    word = _word(value).upper()
    if word.startswith('EUR'):
        return 'European'
    if word.startswith('AMER'):
        return 'American'
    return ''


def _option_type(value):
    word = _word(value).upper()
    if word.startswith('C'):
        return 'Call'
    if word.startswith('P'):
        return 'Put'
    return ''


def probe(source, securities, fields, batch=BATCH, on_batch=None):
    """Every candidate asked in bounded chunks - `discover.probe`, re-spelled here so this module
    stays free of `security_map` and therefore of pandas. `on_batch(done, total)` counts names
    REPLIED ABOUT, never names sent."""
    report = {}
    for start in range(0, len(securities), batch):
        report.update(source.reference_data_report(securities[start:start + batch], fields))
        if on_batch is not None:
            on_batch(min(start + batch, len(securities)), len(securities))
    return report


def chain_members(row, chain_field=CHAIN_FIELD):
    """`(members, unreadable)` off one bulk `OPT_CHAIN` answer.

    A bulk row is a dict of Bloomberg's own sub-field names: `Security Description` is tried by
    name first, then the row's only string value, since the sub-field's spelling is Bloomberg's
    and not a contract this package can pin. A row neither route reads a ticker out of comes back
    as `unreadable` rather than skipped, so it reaches the ledger like every other refusal.
    """
    members, unreadable = [], []
    for index, member in enumerate(row.get(chain_field) or ()):
        if isinstance(member, str):
            found = member.strip()
        elif isinstance(member, collections.abc.Mapping):
            found = _word(member.get(CHAIN_MEMBER_FIELD))
            if not found:
                strings = [_word(value) for value in member.values() if isinstance(value, str)]
                found = strings[0] if len(strings) == 1 else ''
        else:
            found = ''
        if found:
            members.append(found)
        else:
            unreadable.append('{}[{}]'.format(chain_field, index))
    return members, unreadable


def read_contract(security, fields):
    """One candidate contract off the terminal's own answer - RAW, unjudged. Nothing here refuses
    anything; `screen_chain` owns the order of distrust."""
    return ChainContract(
        security=security,
        strike=_number(fields.get('OPT_STRIKE_PX')),
        expiry=_date(fields.get('OPT_EXPIRE_DT')),
        option_type=_option_type(fields.get('OPT_PUT_CALL')),
        exercise=_exercise(fields.get('OPT_EXER_TYP')),
        bid=_number(fields.get('PX_BID')),
        ask=_number(fields.get('PX_ASK')),
        last=_number(fields.get('PX_LAST')),
        open_interest=_number(fields.get('OPEN_INT')),
        volume=_number(fields.get('VOLUME')),
        last_update=_word(fields.get('LAST_UPDATE_DT')) or None)


def screen_chain(contracts, as_of, ladder=None, spot=None):
    """`(accepted, rejected)` - the trust boundary, and the whole reason a chain can be believed.

    THE ORDER IS THE ORDER OF DISTRUST, `discover.verify`'s own shape: what the contract IS before
    what it is worth, and what it is worth before how well anyone knows it. The order is what makes
    a verdict actionable - a screen that checked the spread first would report `wide` where
    `american` is true.

      malformed          no strike, no expiry, or no put/call - not a contract
      expired            an expiry at or before the as-of - not a quote
      unstated-exercise  no exercise style: absence cannot prove EUROPEAN
      american           an American premium is not the European one the fit prices
      unpriced           no two-way and no last print
      one-sided          a side missing or non-positive: a spread nobody quoted is not a spread
      crossed            bid above ask - a stale side left standing against a live one
      off-market         the two MODEL-FREE bounds a chain can be held to with no curve in sight:
                         a call is never worth more than the underlying, a put never more than its
                         strike. Anything sharper needs the forward the EMITTER declares
      wide               a spread past the declared cap: quoted, but not a market
      no-open-interest   OI absent or zero - a listed contract nobody holds
      undated            no LAST_UPDATE_DT: a print that cannot evidence its own time
      stale              a last update older than `stale_days`
      live               believed

    `rejected` is `{security: verdict}` for every one of them, BY NAME. `spot` is optional only
    because the screen is drivable on canned rows without one; a fetch always has it.
    """
    ladder = ladder or EquityLadder()
    accepted, rejected = [], {}
    for contract in sorted(contracts, key=lambda item: item.security):
        verdict = _verdict(contract, as_of, ladder, spot)
        if verdict == 'live':
            accepted.append(contract)
        else:
            rejected[contract.security] = verdict
    return tuple(accepted), rejected


def _verdict(contract, as_of, ladder, spot=None):
    if contract.strike is None or contract.strike <= 0.0 or contract.expiry is None \
            or not contract.option_type:
        return 'malformed'
    if contract.expiry <= as_of:
        return 'expired'
    if not contract.exercise:
        return 'unstated-exercise'
    if contract.exercise != 'European':
        return 'american'
    if contract.mid is None or contract.mid <= 0.0:
        return 'unpriced'
    if contract.bid is None or contract.ask is None or contract.bid <= 0.0 or contract.ask <= 0.0:
        return 'one-sided'
    if contract.bid > contract.ask:
        return 'crossed'
    if spot is not None and contract.mid > (spot if contract.option_type == 'Call'
                                            else contract.strike):
        return 'off-market'
    spread = contract.spread
    if spread is None or spread > ladder.spread_cap:
        return 'wide'
    if contract.open_interest is None or contract.open_interest < ladder.minimum_open_interest:
        return 'no-open-interest'
    # PARSED, not merely present: an unparseable `LAST_UPDATE_DT` would otherwise ride into the
    # block as the row's `Timestamp` and refuse at the decoder instead of here
    stamp = _date(contract.last_update)
    if stamp is None:
        return 'undated'
    if (as_of - stamp).days > ladder.stale_days:
        return 'stale'
    return 'live'


def fetch_equity_chain(source, underlying, as_of, ladder=None, batch=BATCH, on_batch=None,
                       chain_field=CHAIN_FIELD):
    """The listed chain of one index underlying, screened - `underlying` in, `EquityChain` out.

    TWO ROUND TRIPS AND NO MORE VOCABULARY THAN THAT. The first asks the underlying what it IS,
    what it is worth and when it last printed, and asks for its chain in the same request; the
    second asks every member the ten questions a quote needs, in `discover.BATCH`-sized chunks. No
    ticker is spelled by this package at any point - a listed chain's membership is the terminal's
    to state, which is why the bulk route exists and why the SCREEN is the trust boundary.

    The spot refuses BY NAME on anything that is not a positive number and on a print date it
    cannot read or that is older than `stale_days`: every strike, forward and weight below hangs
    off it, and it is the one price nothing downstream can cross-check. The date it passed on
    travels into the block.
    """
    ladder = ladder or EquityLadder()
    head = source.bulk_reference_data_report(
        [underlying], list(UNDERLYING_FIELDS) + [chain_field])
    row = head.get(underlying, {'ok': False, 'error': 'no answer in the response', 'fields': {}})
    if not row.get('ok') and row.get('error'):
        raise_response_error('{}: {}'.format(underlying, row['error']))
    fields = row.get('fields', {})
    spot = _number(fields.get('PX_LAST'))
    if spot is None or spot <= 0.0:
        raise InvalidQuote(
            '{} returned {!r} for PX_LAST - every strike, forward and weight in the ladder hangs '
            'off the spot, so a chain cannot be built around a blank. Check the underlying '
            'ticker and this workstation\'s entitlement for it'.format(
                underlying, fields.get('PX_LAST')))
    printed = _date(fields.get('LAST_UPDATE_DT'))
    if printed is None:
        raise InvalidQuote(
            '{} answered {!r} for LAST_UPDATE_DT beside a spot of {:.6g} - a price that cannot '
            'evidence its own time is the `undated` verdict every listed contract in this chain is '
            'refused on, and the number the whole ladder is placed with cannot be the one thing '
            'exempt from it. Check the underlying ticker and this workstation\'s entitlement for '
            'it'.format(underlying, fields.get('LAST_UPDATE_DT'), spot))
    if (as_of - printed).days > ladder.stale_days:
        raise InvalidQuote(
            '{} last printed {}, {} days before the as-of {}, against a stale_days of {} - the '
            'spot is stale and every strike, forward and weight in the ladder would be placed on '
            'it. A retired ticker keeps answering a plausible price and the update date is the '
            'only thing that says so. Quote the chain as at {}, or check the underlying ticker and '
            'this workstation\'s entitlement for it'.format(
                underlying, printed.isoformat(), (as_of - printed).days, as_of.isoformat(),
                ladder.stale_days, printed.isoformat()))

    members, unreadable = chain_members(fields, chain_field)
    if not members:
        raise IncompleteChain(
            '{} answered no {} members - either this workstation is not entitled to the listed '
            'chain, or the ticker names something with no options on it. Ask for an index with a '
            'listed chain (SPX Index, SX5E Index)'.format(underlying, chain_field))

    # a row with ANY field in it is read and the SCREEN judges it; only an empty row is refused.
    # `ok: False` covers a mere fieldException too - an untraded contract carries no VOLUME - and
    # gating on it cost 1,855 of 8,000 contracts on a measured live SPX chain
    report = probe(source, sorted(set(members)), CONTRACT_FIELDS, batch=batch, on_batch=on_batch)
    contracts, rejected = [], {name: 'unreadable-chain-row' for name in unreadable}
    for security in sorted(set(members)):
        answer = report.get(security, {'ok': False, 'error': 'no answer in the response',
                                       'fields': {}})
        answered = answer.get('fields') or {}
        if not answered:
            rejected[security] = 'invalid'
            continue
        contracts.append(read_contract(security, answered))
    accepted, screened = screen_chain(contracts, as_of, ladder, spot)
    rejected.update(screened)
    return EquityChain(underlying=underlying, name=_word(fields.get('NAME')), spot=spot,
                       as_of=as_of, contracts=accepted, rejected=rejected, spot_as_of=printed)


# ---------------------------------------------------------------------------------------------
# the selection
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ParityCarry:
    """The carry a chain implies at one expiry, and the EVIDENCE it was read off.

    The strikes and the per-strike readings travel with the median because a refusal naming only
    the number would tell a desk its chain is wrong without saying where to look.
    """
    value: float
    strikes: tuple[float, ...]
    readings: tuple[float, ...]


def parity_dividend_yield(contracts, spot, rate, tau, strikes=PARITY_STRIKES):
    """The dividend yield the CHAIN ITSELF implies at one expiry, by put-call parity, or None.

    `C - P = (F - K) exp(-r t)` wherever BOTH legs survived the screen, so `F = K + (C - P) exp(r
    t)` and `q = r - ln(F/S)/t`. Parity holds at every strike, so this is one reading per two-sided
    pair rather than a fit, and what comes back is their MEDIAN over the `strikes` pairs nearest
    the forward.

    A MEDIAN, NOT ONE PAIR: where the caller declared no dividend yield this number IS the pillar's
    carry, and one fat-fingered near-money print believed by every per-contract screen moves a
    single-pair read by percentage points.

    NEAREST THE FORWARD, IN TWO PASSES: the strike nearest the SPOT has one deep-ish in-the-money
    leg, which a chain quotes worst, so the neighbourhood is chosen around the spot and re-read
    around the forward that pass implies. Not iteration - parity is exact at every strike and the
    second pass is about QUOTE QUALITY.

    Where a dividend yield was declared this stays a CHECK, reported beside it, never averaged in.
    """
    if tau <= 0.0 or spot <= 0.0:
        return None
    pairs = {}
    for contract in contracts:
        pairs.setdefault(contract.strike, {})[contract.option_type] = contract.mid
    both = [(strike, legs['Call'], legs['Put']) for strike, legs in pairs.items()
            if legs.get('Call') and legs.get('Put')]
    if not both:
        return None
    around_spot = _parity_median(both, spot, rate, tau, spot, strikes)
    if around_spot is None:
        return None
    level = spot * math.exp((rate - around_spot.value) * tau)
    return _parity_median(both, spot, rate, tau, level, strikes) or around_spot


def _parity_median(both, spot, rate, tau, around, strikes):
    """The median parity carry over the `strikes` two-sided pairs nearest `around`, with the
    strikes it read and every reading it took - the evidence a refusal names."""
    nearest = sorted(both, key=lambda item: (abs(math.log(item[0] / around)), item[0]))[:strikes]
    measured = []
    for strike, call, put in sorted(nearest):
        forward = strike + (call - put) * math.exp(rate * tau)
        if forward > 0.0:
            measured.append((strike, rate - math.log(forward / spot) / tau))
    if not measured:
        return None
    return ParityCarry(statistics.median(reading for _, reading in measured),
                       tuple(strike for strike, _ in measured),
                       tuple(reading for _, reading in measured))


def assign_expiries(expiries, ladder):
    """`({pillar: expiry}, {pillar: why})` - the ladder's pillars matched to LISTED expiries, ONE
    EXPIRY TO ONE PILLAR, and the reason by name for every pillar left with nothing.

    THREE RULES, AND THE THIRD IS WHY THIS IS A MATCHING RATHER THAN AN ARGMIN PER PILLAR.

    The CAP: an argmin has no ceiling, so without it a chain listing only two months answers every
    rung of a three-year ladder and the fit is a 2M fit wearing a 3Y label. The BAND: 3M and 3Y are
    a log-unit and a half apart, and half a log-unit is the 1Y rung reaching 6M or 1.6Y.

    ONE EXPIRY, ONE PILLAR is the third. An ordinary board lists quarterlies out a year then jumps
    to LEAPS, so a 1Y and a 2Y pillar both land on one January listing. Allowing it would emit ONE
    contract as TWO equations at double weight - which the two family spellings read differently -
    and write an L strip with fewer knots than the ladder declares pillars. So a pillar left with
    nothing is DROPPED by name.

    NEAREST CLAIM WINS, not shortest-first: pairs are taken in order of log-distance, so a 3Y
    listing does not enter as the 2Y rung while the 3Y pillar it IS gets dropped. Ties break on the
    pillar then the date, so the assignment is a function of the chain rather than of dict order.

    ONE METRIC, LOG-EXPIRY throughout. A ladder spanning 3M to 3Y cannot judge nearness in years -
    a month's error is most of the front rung and a rounding error on the back one.
    """
    cap = max(ladder.pillars) + ladder.expiry_tolerance
    pairs = sorted((abs(math.log(tau / pillar)), pillar, expiry)
                   for pillar in ladder.pillars for expiry, tau in expiries.items()
                   if tau <= cap and abs(math.log(tau / pillar)) <= ladder.pillar_band)
    assigned, claimed = {}, {}
    for _, pillar, expiry in pairs:
        if pillar not in assigned and expiry not in claimed:
            assigned[pillar], claimed[expiry] = expiry, pillar
    dropped = {}
    for pillar in sorted(set(ladder.pillars) - set(assigned)):
        taken = sorted(expiry for expiry in claimed
                       if expiries[expiry] <= cap
                       and abs(math.log(expiries[expiry] / pillar)) <= ladder.pillar_band)
        dropped[pillar] = (
            'every listed expiry within {:g} log-units of it is a nearer pillar\'s rung ({})'.format(
                ladder.pillar_band,
                ', '.join('{} is the {:g}y'.format(expiry.isoformat(), claimed[expiry])
                          for expiry in taken))
            if taken else
            'no listed expiry within {:g} log-units of it at or under {:g}'.format(
                ladder.pillar_band, cap))
    return assigned, dropped


def _snap_strike(candidates, forward, target):
    """The listed contract nearest a target strike, taking the OTM leg at whatever strike wins.

    The OTM leg is the one a desk deals and the fit is blind to the choice (both families price
    puts by parity off the call), so the type follows the STRIKE rather than the rung's intention:
    a put wing snapping above the forward emits the call quoted at that strike.
    """
    best, distance = None, None
    for contract in candidates:
        wanted = 'Call' if contract.strike >= forward else 'Put'
        if contract.option_type != wanted:
            continue
        moved = abs(math.log(contract.strike / target))
        if distance is None or moved < distance or (
                moved == distance and contract.security < best.security):
            best, distance = contract, moved
    return best


def select_rungs(chain, forward, ladder=None):
    """`(rungs, notes, readings)` - the selection, and the one both family spellings are emitted
    from. Nothing below this line knows which family it is writing for.

    THE FORMULA, stated once:

      for each pillar T:
        E(T)  the listed expiry nearest T, at or under max(pillars) + expiry_tolerance and within
              pillar_band of T in log-expiry, and CLAIMED BY NO OTHER PILLAR (`assign_expiries`:
              nearest claim wins); a pillar with no such expiry is DROPPED with a note
        t     (E(T) - as_of) / days_per_year
        q(T)  the DECLARED dividend yield, or the MEDIAN parity carry over the parity_strikes
              two-sided pairs nearest the forward where none was declared - measured either way and
              reported beside the declared number, and screened against parity_band where it is
              the carry
        F(T)  spot * exp((r - q(T)) * t)
        ATM rung   target strike F(T)
        sigma(T)   the Black implied vol of the SNAPPED ATM contract's own mid, off F(T)
        wing rungs, at each pillar in wing_pillars, target strikes
                   K+ = F exp(+sigma^2 t/2 + z sigma sqrt(t))   the delta-call wing
                   K- = F exp(+sigma^2 t/2 - z sigma sqrt(t))   the delta-put wing
                   z = Phi^-1(1 - wing_delta), i.e. 0.6745 at the 25 delta
        every rung SNAPS to the listed contract at E(T) minimising |log(K_listed / K_target)|,
        taking the OTM leg at that strike

    The wing is a place to LOOK, the band read off the chain's own ATM implied vol: a listed chain
    has no delta axis, so the band names a coordinate rather than asserting a delta.

    THE WEIGHT, and it is a weight rather than a price:

        w = vega(F, K, r, sigma_K, t) * sqrt(OI) / (1 + spread / spread_cap)
        Weight = w / sum(w)

    Vega is at the contract's own implied vol off the declared forward, which is what makes the
    objective scale-free across a term structure running to three years. `sqrt(OI)` because open
    interest is evidence of attention and not a precision. The spread factor runs from 1 at a locked
    market to 1/2 at the cap, so the tightest print is worth twice the widest survivor and nothing
    that survived is worth zero.
    """
    ladder = ladder or EquityLadder()
    if not chain.contracts:
        raise IncompleteChain(
            '{} screened to no believed contract at all - all {} candidates were refused ({}). '
            'There is no ladder to select, so nothing below this line would mean anything. Widen '
            'the screen the census names, or quote a chain that trades'.format(
                chain.underlying, len(chain.rejected),
                ', '.join('{} {}'.format(count, verdict) for verdict, count in sorted(
                    _census(chain).items())) or 'nothing was asked'))
    by_expiry = {}
    for contract in chain.contracts:
        by_expiry.setdefault(contract.expiry, []).append(contract)
    expiries = {expiry: (expiry - chain.as_of).days / ladder.days_per_year
                for expiry in by_expiry}
    z = NORMAL.inv_cdf(1.0 - ladder.wing_delta)
    assigned, dropped = assign_expiries(expiries, ladder)

    rungs, notes, readings = [], [], {}
    for pillar in sorted(ladder.pillars):
        if pillar not in assigned:
            notes.append('{:g}y DROPPED - {}'.format(pillar, dropped[pillar]))
            continue
        expiry = assigned[pillar]
        tau = expiries[expiry]
        candidates = by_expiry[expiry]
        implied_q = parity_dividend_yield(candidates, chain.spot, forward.rate, tau,
                                          ladder.parity_strikes)
        carry = forward.dividend_yield if forward.dividend_yield is not None else \
            _believed_carry(implied_q, pillar, expiry, forward, ladder)
        level = chain.spot * math.exp((forward.rate - carry) * tau)

        atm = _snap_strike(candidates, level, level)
        if atm is None:
            notes.append('{:g}y DROPPED - {} carries no contract on the deal side of its '
                         'forward'.format(pillar, expiry.isoformat()))
            continue
        atm_vol = implied_vol(atm.mid, level, atm.strike, forward.rate, tau,
                              atm.option_type == 'Call') or ladder.reference_vol
        readings[pillar] = {'expiry': expiry, 'tau': tau, 'forward': level, 'atm_vol': atm_vol,
                            'declared_dividend': carry,
                            'implied_dividend': None if implied_q is None else implied_q.value,
                            'implied_strikes': () if implied_q is None else implied_q.strikes}
        if not _close(expiries[expiry], pillar):
            notes.append('{:g}y -> {} ({:.4g}y)'.format(pillar, expiry.isoformat(), tau))
        rungs.append(_rung('ATM', pillar, atm, tau, level, forward.rate, ladder))

        if pillar not in ladder.wing_pillars:
            continue
        stddev = atm_vol * math.sqrt(tau)
        for side, label in ((1.0, '{:g}d call'.format(ladder.wing_delta * 100)),
                            (-1.0, '{:g}d put'.format(ladder.wing_delta * 100))):
            target = level * math.exp(0.5 * stddev * stddev + side * z * stddev)
            wing = _snap_strike(candidates, level, target)
            if wing is None:
                notes.append('{} {:g}y DROPPED - {} carries no contract near {:.6g}'.format(
                    label, pillar, expiry.isoformat(), target))
                continue
            rungs.append(_rung(label, pillar, wing, tau, level, forward.rate, ladder))

    total = sum(rung.weight for rung in rungs)
    if not total > 0.0:
        raise IncompleteChain(
            '{} priced every selected contract at zero vega times liquidity, so there is no weight '
            'to normalise and nothing the fit would be sensitive to. The ladder landed on {} '
            'contract(s); check the chain\'s open interest and its spreads'.format(
                chain.underlying, len(rungs)))
    rungs = tuple(Rung(rung.kind, rung.pillar, rung.contract, rung.tau, rung.forward,
                       rung.implied_vol, rung.vega, rung.weight / total) for rung in rungs)
    return rungs, tuple(notes), readings


def _believed_carry(implied, pillar, expiry, forward, ladder):
    """The chain's own parity carry, SCREENED against `parity_band` before it becomes a pillar's
    carry, refusing with the pillar, the number and the strikes it was read off.

    A declared `EquityForward.dividend_yield` never reaches this function. With nothing declared the
    parity reading IS the carry, placing and weighing every strike on the pillar, so it is screened
    like any other evidence: the median has already thrown out one bad print and the band catches a
    neighbourhood that is wrong together.
    """
    if implied is None:
        raise IncompleteChain(
            'the {:g}y pillar ({}) declares no dividend yield and the chain implies none - '
            'put-call parity needs one strike whose CALL and PUT both survived the screen, and '
            'this expiry carries none. Declare EquityForward.dividend_yield off the book\'s '
            'own {} curve, or quote an expiry with a two-sided pair on it'.format(
                pillar, expiry.isoformat(), forward.dividend_reference))
    low, high = ladder.parity_band
    if not low <= implied.value <= high:
        raise IncompleteChain(
            'the {:g}y pillar ({}) declares no dividend yield and the carry its own chain implies, '
            '{:.4%}, is outside the declared band {:.4%}..{:.4%} - the median of ({}) read at '
            'strikes {}. An undeclared carry is not a reading beside the block, it IS the pillar\'s '
            'forward: it places every strike on that pillar and weighs every one of them, so it is '
            'screened like any other foreign answer. Declare EquityForward.dividend_yield off the '
            'book\'s own {} curve, or widen EquityLadder.parity_band if this chain really carries '
            'that carry'.format(
                pillar, expiry.isoformat(), implied.value, low, high,
                ', '.join('{:.4%}'.format(reading) for reading in implied.readings),
                '/'.join('{:g}'.format(strike) for strike in implied.strikes),
                forward.dividend_reference))
    return implied.value


def collapse_rungs(rungs):
    """`(rows, notes)` - ONE ROW PER DISTINCT LISTED CONTRACT, the colliding rungs' weights SUMMED.

    A repeated contract is a WEIGHT and not a second equation, and the collapse happens HERE because
    the two family spellings do not agree about a duplicate row: the component bootstrap
    deduplicates a repeated ATM by strike within an expiry and DISCARDS that row's weight, the plain
    family applies every row. One block that reads two ways is not one selection with two spellings.

    `assign_expiries` has already made the cross-pillar collision impossible; what survives is the
    collision WITHIN a pillar. The note names which rung was absorbed into which, a rung that
    quietly vanished being unauditable against the ladder.
    """
    order, merged, notes = [], {}, []
    for rung in rungs:
        key = (rung.contract.expiry, rung.contract.strike, rung.contract.option_type)
        if key not in merged:
            order.append(key)
            merged[key] = [rung, rung.weight]
            continue
        first = merged[key][0]
        merged[key][1] += rung.weight
        notes.append(
            '{} {:g}y MERGED into the {} {:g}y rung - both landed on {} {:g} {}, and a repeated '
            'contract is a weight rather than a second equation'.format(
                rung.kind, rung.pillar, first.kind, first.pillar, rung.contract.expiry.isoformat(),
                rung.contract.strike, rung.contract.option_type))
    return tuple((merged[key][0], merged[key][1]) for key in order), tuple(notes)


def _close(left, right, tolerance=1e-12):
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _census(chain):
    """`{verdict: count}` over the ledger - what the report reads and what a refusal names."""
    census = {}
    for verdict in chain.rejected.values():
        census[verdict] = census.get(verdict, 0) + 1
    return census


def _rung(kind, pillar, contract, tau, level, rate, ladder):
    """One rung with its RAW weight - `vega * sqrt(OI) / (1 + spread/cap)`, normalised by the
    caller over the ladder as emitted. The vega is taken at the contract's own implied vol, or at
    the ladder's flat proxy where the mid yields none, so a screened contract never weighs zero
    for want of an inversion."""
    vol = implied_vol(contract.mid, level, contract.strike, rate, tau,
                      contract.option_type == 'Call') or ladder.reference_vol
    vega = black_vega(level, contract.strike, rate, vol, tau)
    spread = contract.spread if contract.spread is not None else ladder.spread_cap
    liquidity = math.sqrt(max(contract.open_interest or 0.0, 0.0))
    weight = vega * liquidity / (1.0 + spread / ladder.spread_cap)
    return Rung(kind, pillar, contract, tau, level, vol, vega, weight)


# ---------------------------------------------------------------------------------------------
# the block
# ---------------------------------------------------------------------------------------------

def _timestamp(value):
    """A date in the WIRE spelling the engine's own decoder reads - `{'.Timestamp': 'YYYY-MM-DD'}`.

    The wire form rather than a `pandas.Timestamp`, since a block is posted as JSON and spelling it
    here keeps this module free of pandas. Whatever the terminal's own spelling was, what is
    written is a parsed ISO date, so the block is the same bytes whichever way the field came back.
    """
    stamp = _date(value)
    if stamp is None:
        raise InvalidQuote(
            '{!r} is not a date this block can carry - every emitted contract has been screened '
            'for a readable print date, so reaching here is a bug rather than a market'.format(
                value))
    return {'.Timestamp': stamp.isoformat()}


def market_price_name(family, forward):
    """`<family>.<underlying factor>` - `fxvol._market_price_name`'s shape. The written parameters
    then land at `<family minus Prices, plus Parameters>.<underlying>`, which is the factor a deal
    resolves by naming convention off its own equity."""
    return '{}.{}'.format(family, forward.underlying_factor)


def equity_hn_block(chain, forward, ladder=None, family=COMPONENT_FAMILY):
    """`(Market Prices name, block)` - the chain as ONE Heston-Nandi quote block.

    ONE SELECTION, TWO SPELLINGS. `select_rungs` runs the same way for both families and the
    `European_Options` table is byte-identical between them; only the header each family declares
    differs. `collapse_rungs` keeps that true of the two families' READING of the table as well.
    The component spelling additionally states `Rho` and `Quote_Sensitivity` No.

    PREMIUMS, NOT VOLS. `Quote_Type` is Premium and `Quoted_Market_Value` the mid of the terminal's
    two-way, in the underlying's own units, with `QUOTE_VALUE_KEYS` beside it as DECLARED value
    columns - the two-way and the print's own clock are the evidence, and what lets a chain tick.

    `Use_Forward` and `Invert_Moneyness` are written at their declared defaults and are INERT: both
    exist to look a vol surface up AT A STRIKE, and under `Quote_Type` Premium none is read. The
    block names a `Volatility` reference anyway, since a chain-sourced fit still marks against a
    surface downstream.

    Refuses by name, with the remedy, on: exercise style killing the ladder, a ladder collapsing
    below `minimum_contracts` distinct contracts, an expiry with no admissible dividend evidence or
    an undeclared carry outside `parity_band`, and a ladder with no weight in it.
    """
    ladder = ladder or EquityLadder()
    if family not in FAMILIES:
        raise BloombergConfigurationError(
            '{!r} is not a Heston-Nandi quote family - this emitter writes {}'.format(
                family, ' or '.join(FAMILIES)))
    _refuse_american(chain, ladder)

    rungs, selection_notes, readings = select_rungs(chain, forward, ladder)
    rows, merged = collapse_rungs(rungs)
    notes = selection_notes + merged
    contracts = {(rung.contract.expiry, rung.contract.strike) for rung, _ in rows}
    if len(contracts) < ladder.minimum_contracts:
        raise IncompleteChain(
            '{} lists expiries {} - the ladder (ATM {}, {:g}d wings {}) collapses onto {} distinct '
            'contract{} on it, and {} do not identify five free globals off the smile: the ATM '
            'term structure is spent on the L pillars, which are bootstrapped rather than fitted. '
            'Quote the chain at more expiries and strikes (at least {} distinct contracts), or '
            'author the {} block by hand. What each rung did: {}'.format(
                chain.underlying, '/'.join(expiry.isoformat() for expiry in chain.expiries),
                '/'.join('{:g}'.format(pillar) for pillar in sorted(ladder.pillars)),
                ladder.wing_delta * 100,
                '/'.join('{:g}'.format(pillar) for pillar in sorted(ladder.wing_pillars)),
                len(contracts), '' if len(contracts) == 1 else 's', len(contracts),
                ladder.minimum_contracts, family,
                ', '.join(notes) or 'every rung landed on the pillar it was asked for'))

    quotes = [{
        'Expiry_Date': _timestamp(rung.contract.expiry),
        'Strike': rung.contract.strike,
        'Option_Type': rung.contract.option_type,
        'Units': 1.0,
        'Weight': weight,
        'Quoted_Market_Value': rung.contract.mid,
        'Quoted_Bid': rung.contract.bid,
        'Quoted_Ask': rung.contract.ask,
        'Timestamp': _timestamp(rung.contract.last_update)} for rung, weight in rows]

    instrument = {
        'Underlying': forward.underlying_factor, 'Underlying_Type': forward.underlying_type,
        'Volatility': forward.volatility_factor, 'Volatility_Type': forward.volatility_type,
        'Discount_Rate': forward.discount_rate,
        'Discount_Rate_Type': forward.discount_rate_type,
        'Yield': forward.dividend_reference, 'Yield_Type': forward.dividend_type,
        # the funding curve is written only where one was DECLARED: blank, the family funds the
        # forward off Discount_Rate, and a field spelling that out would state a name nobody chose
        **({'Funding_Rate': forward.funding_rate,
            'Funding_Rate_Type': forward.funding_rate_type} if forward.funding_rate else {}),
        'Quote_Type': 'Premium',
        'Use_Forward': 'No', 'Invert_Moneyness': 'No',
        'Steps_Per_Year': ladder.steps_per_year,
        'Quadrature_Panels': ladder.quadrature_panels,
        'Quote_Timestamp': _timestamp(chain.as_of),
        'Quote_Source': quote_source(chain, forward, ladder, rungs, rows, notes, readings),
        'European_Options': quotes}
    if family == COMPONENT_FAMILY:
        instrument.update(COMPONENT_HEADER)
        # the option table stays LAST whichever family this is, so the two spellings differ only
        # in their header
        instrument['European_Options'] = instrument.pop('European_Options')
    return market_price_name(family, forward), {'instrument': instrument}


def _refuse_american(chain, ladder):
    """Refuse a chain that exercise style, not thin quoting, took below the contract floor.

    Fires when the chain cannot reach `minimum_contracts` AT ALL and more candidates were refused
    on style than survived the whole screen. Below the floor with the refusals elsewhere, the
    floor's own refusal is the true one and names its own census.

    The per-contract screen stays either way: a mixed board - a flex or weekly American listing
    beside European ones - drops those individually at `_verdict` and the European ones calibrate.
    """
    styles = sorted({verdict for verdict in chain.rejected.values()
                     if verdict in ('american', 'unstated-exercise')})
    refused = sum(1 for verdict in chain.rejected.values() if verdict in styles)
    if not refused or len(chain.contracts) >= ladder.minimum_contracts \
            or refused <= len(chain.contracts):
        return
    raise UnsupportedExerciseStyle(
        '{} screened to {} believed contract{} against a floor of {}, and {} of its {} candidates '
        'were refused on exercise style ({}) - so what the chain IS killed the ladder, not how '
        'thinly it is quoted. An AMERICAN premium is not the European premium a Heston-Nandi fit '
        'prices against: the early-exercise right is worth something the closed form does not '
        'carry, so fitting one would put the wrong number in the objective under the right name. '
        'Quote an index chain with European exercise (SPX Index, SX5E Index). The whole census: '
        '{}'.format(
            chain.underlying, len(chain.contracts), '' if len(chain.contracts) == 1 else 's',
            ladder.minimum_contracts, refused, len(chain.rejected), '/'.join(styles),
            ', '.join('{} {}'.format(count, verdict)
                      for verdict, count in sorted(_census(chain).items())) or 'nothing refused'))


def quote_source(chain, forward, ladder, rungs, rows, notes, readings):
    """The block's own account of where its quotes came from, in one line beside the parameters
    they produce - `fx_surface_block`'s `Quote_Source` plus what a chain has that a surface does
    not: the census, the spot's own print date, the declared carry and the carry the chain implies.

    THE DIVIDEND DISAGREEMENT IS REPORTED RATHER THAN RESOLVED: a declared yield placed the strikes
    and is what the fit rebuilds its forward from, while the parity-implied number beside it is the
    chain's own opinion. A reading outside `parity_band` is NAMED here rather than refused, the band
    screening nothing where a yield is declared.

    The two curves are named separately where they differ: `r` is the FUNDING rate the forward grew
    at, and the premium's discount curve is named after it. One curve says so by saying nothing.

    The rows are counted beside the rungs because two rungs landing on one listed contract are
    emitted once at their summed weight.
    """
    census = _census(chain)
    low, high = ladder.parity_band
    dividends = ', '.join(
        '{:g}y declared {:.4%}{}'.format(
            pillar, reading['declared_dividend'],
            '' if reading['implied_dividend'] is None else
            ' / chain implies {:.4%}{}'.format(
                reading['implied_dividend'],
                '' if low <= reading['implied_dividend'] <= high
                else ' OUTSIDE the declared band {:.4%}..{:.4%}'.format(low, high)))
        for pillar, reading in sorted(readings.items()))
    source = (
        '{} rungs ({}) on {} distinct contracts off the listed {} chain as at {}, {} contracts '
        'believed of {} asked ({}); premiums are the terminal\'s own two-way mids. Forward: spot '
        '{:.6g} (last printed {}) carried at r={:.4%} on {} against {}{} [{}]'.format(
            len(rungs), '/'.join(sorted({rung.kind for rung in rungs})), len(rows),
            chain.underlying, chain.as_of.isoformat(), len(chain.contracts),
            len(chain.contracts) + len(chain.rejected),
            ', '.join('{} {}'.format(count, verdict)
                      for verdict, count in sorted(census.items())) or 'nothing refused',
            chain.spot,
            'undated' if chain.spot_as_of is None else chain.spot_as_of.isoformat(),
            forward.rate, forward.funding_rate or forward.discount_rate,
            forward.dividend_reference,
            # named only where the two curves differ: one curve is what the line already says
            ', premiums discounting on {}'.format(forward.discount_rate)
            if forward.funding_rate and forward.funding_rate != forward.discount_rate else '',
            dividends or 'no pillar priced'))
    if notes:
        source += '; rungs the chain does not list, moved or dropped: {}'.format(', '.join(notes))
    return source
