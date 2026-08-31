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

THE SOURCE FLIPS, WHICH IS THE WHOLE ROW. FX calibrates off the desk's own built surface because
that surface IS the market - the OTC delta quotes are what a desk deals. An equity's market is the
LISTED CHAIN, and any equity vol surface is already somebody's fit to it, so calibrating to a
surface would fit a fit. Equities therefore calibrate TO THE CHAIN and quote PREMIUMS, not implied
vols: a listed price is a PRINT, while its implied vol is a convention (which forward, which
discounting, which exercise) and two desks disagree about it before either has been wrong. The
Heston-Nandi families already accept `Quote_Type` **Premium**; this is the emitter that was waiting
for.

WHAT THIS MODULE IS AND IS NOT. It is `fxvol`'s sibling: it reaches a terminal, screens what came
back, and writes a `{"instrument": {...}}` block. It prices nothing, fits nothing, builds no
surface, and - like every module in this package - IMPORTS NO ENGINE. It goes further than its
sibling and imports no pandas either, so `import derivus_bloomberg.equity_chain` costs the standard
library and this package's own `errors`. That is a claim about the IMPORT and not merely about this
file, and the file alone cannot hold it: importing a submodule imports its package first, and
`fxvol` and `types` carry pandas. So the package's `__init__` re-exports those two LAZILY, and two
gates hold the whole claim - one reads this source, one measures a fresh interpreter's `sys.modules`
and would have caught the package dragging pandas in behind the module's back.

EVERY FOREIGN ANSWER IS UNTRUSTED, and a listed chain is where that stops being a slogan. Half of
any index chain is dead strikes: contracts that resolve, carry a plausible `PX_LAST` and have not
traded in a month; contracts quoted one-sided, or crossed, or with an open interest of zero. That
is the same trap `discover` exists for (a retired benchmark still answers a price - SAONIA read
8.855 nineteen years after its last print), so the same discipline applies: `screen_chain`
classifies every contract in an ORDER OF DISTRUST and puts every refusal on a ledger BY NAME,
because a contract silently dropped is indistinguishable from one never asked about.

AMERICAN EXERCISE REFUSES BY NAME. V1 is indices only. An American premium is not the European
premium the fit assumes - it carries the early-exercise right the closed form does not price - so
an American contract is refused from candidacy and a chain with no European contract at all raises
`UnsupportedExerciseStyle` naming the underlying and the remedy. A contract that states NO exercise
style is refused too, and the asymmetry with `discover._is_stale` ("absence cannot prove
staleness") is deliberate: absence cannot prove DEATH, but it cannot prove EUROPEAN either, and the
cost of the two mistakes is not the same.

THE LADDER REACHES THE PRODUCT HORIZON. Equity autocalls run three to five years, so the pillars
default to 3M/6M/1Y/2Y/3Y rather than FX's "nothing past 1Y" - and a multi-year ATM term structure
is exactly what one omega cannot hold and the component family's L curve can, which is why the
target family is `HestonNandiComponentModelPrices`. The plain spelling is emitted from the SAME
selection, so the two blocks differ only in the header their families declare.

THE FORWARD IS DECLARED, NOT DISCOVERED. This is the genuinely new work beside FX. The strikes hang
off the forward and so does every weight, and if the emitter's forward disagrees with the pricer's
then the calibration is fitted at coordinates the pricer will never visit. So `EquityForward` names
the two references the fit resolves - the curve that funds the carry and the dividend/borrow
reference - AND the numbers the emitter placed its strikes with, and both travel into the block.
The chain's own parity-implied dividend yield is measured beside the declared one and reported, so
a desk sees a disagreement rather than inheriting it.
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
#: printed - `discover.FIELDS` exactly, because the same three questions answer the same doubt. All
#: three are READ: the spot refuses on a blank, and `LAST_UPDATE_DT` is screened against
#: `stale_days` exactly as every listed contract's is, because the spot is the one number every
#: strike, forward and weight in the block hangs off and the anchor cannot be the one thing exempt
#: from the discipline this module invokes by name.
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

#: One request per chunk - `discover.BATCH`, and for its reason: the Desktop API takes large
#: batches, and a bounded request keeps a partial outage partial. A full SPX chain is thousands of
#: names, so this one actually gets used.
BATCH = 50

#: The four reference fields the Heston-Nandi families declare, and the factor TYPE this emitter
#: names each of them with for an equity underlying. Spelled here because the package may not
#: import the engine, and held against `HestonNandiModelParameters.factor_types` by a gate - the
#: whitelist-against-declaration pattern `fxvol._structure` already rides on.
HN_REFERENCE_TYPES = {'Underlying': 'EquityPrice', 'Volatility': 'EquityPriceVol',
                      'Discount_Rate': 'InterestRate', 'Yield': 'DividendRate'}

#: The two spellings of the target family. ONE SELECTION, TWO NAMES: the component family is the
#: one the roadmap ratified (a multi-year ATM term structure is what the L curve is for), and the
#: plain one is emitted off the same rungs for a desk that wants the five-parameter fit.
COMPONENT_FAMILY = 'HestonNandiComponentModelPrices'
PLAIN_FAMILY = 'HestonNandiModelPrices'
FAMILIES = (COMPONENT_FAMILY, PLAIN_FAMILY)

#: The declared defaults of the two switches the emitter STATES rather than lets fall through -
#: `fx_surface_block`'s own discipline, and for its reason: the STEP CLOCK is what the fitted
#: parameters mean, so a deal's `Steps_Per_Year` has to be this number or it simulates a different
#: model. Gated against the families' own field declarations.
STEPS_PER_YEAR = 252.0
QUADRATURE_PANELS = 64

#: The extra header the COMPONENT family declares and the plain one does not. `Rho` is a PIN and a
#: pin the block does not state is a pin nobody can see; `Quote_Sensitivity` is REFUSED by that
#: family, so saying No is honesty rather than decoration. Both are the fields' own declared
#: defaults, read off the declaration by the same gate that reads the two above.
COMPONENT_HEADER = {'Rho': 0.99, 'Quote_Sensitivity': 'No'}

#: The value-plane keys an option quote row carries beside its mid. NOT DECLARED BY `OPTION_QUOTE`
#: today - the engine's option tables carry six columns and none of these three - so they ride as
#: undeclared keys that `bootstrap` reads past and `as_json` preserves. Carried anyway, because the
#: two-way and the print's own clock are the EVIDENCE, and a quote layer that drops them has thrown
#: away the desk's spread to fit a schema. Named as a finding, gated as a complement.
#:
#: AND THE CONSEQUENCE, STATED RATHER THAN DISCOVERED: because these three are not declared on the
#: option row, `schema.partition_market_price` reads this family's whole block as PLAN and gives it
#: an EMPTY value half - so a chain that merely RE-TICKS cannot be value-updated at all. Moving a
#: premium is 'structure differs' to `config.update_market_quote`, and a re-quoted chain has to be
#: RE-AUTHORED rather than ticked. That is engine-side today and it is what the follow-up
#: book/service verb has to decide; the gate asserts the refusal in its current shape so the day
#: `OPTION_QUOTE` gains these columns, the gate flips and says so.
QUOTE_VALUE_KEYS = ('Quoted_Bid', 'Quoted_Ask', 'Timestamp')

#: How many two-sided strikes the chain's own parity carry is MEDIANED over, and the band that
#: carry is believed inside where the caller declared none. Both are ladder parameters with these
#: as their defaults; they are spelled here beside the other declared constants.
#:
#: FIVE STRIKES AND A MEDIAN, because one pair is one print. A single fat-fingered near-money quote
#: that passes every per-contract screen - tight, deep, dated today - moves a parity-read carry by
#: percentage points, and the carry moves the pillar's whole forward: every strike placed on it and
#: every weight computed there. A median over the strikes nearest the forward is the cheapest
#: estimator that ignores one bad print entirely.
#:
#: THE BAND IS THE SCREEN, because a median cannot save a neighbourhood that is wrong together.
#: Minus five points to plus fifteen is a listed index's carry with room on both sides - a
#: hard-to-borrow basket quoting through zero, an emerging index quoting a full year of dividends -
#: and it is nowhere near a two-figure negative carry, which is a bad chain rather than a market.
PARITY_STRIKES = 5
PARITY_BAND = (-0.05, 0.15)


# ---------------------------------------------------------------------------------------------
# what a chain is
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainContract:
    """One listed option as the terminal answered for it - raw, unjudged, and NOT yet believed.

    `expiry` is a `datetime.date` and `strike` a float because those are what a contract IS; the
    two-way, the last print, the open interest and the last-update date are all OPTIONAL, because
    a dead strike answers half of them and the screen's whole job is to say which half.
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
        screen refuses a one-sided contract before this ever matters, so the fallback exists for
        the ledger's sake - `unpriced` and `one-sided` are different findings."""
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        return self.last

    @property
    def spread(self) -> float | None:
        """The quoted spread as a FRACTION OF MID - the market's own statement of how well it
        knows this price, and the number both the screen's cap and the weight are read off."""
        mid = self.mid
        if self.bid is None or self.ask is None or not mid or mid <= 0.0:
            return None
        return (self.ask - self.bid) / mid


@dataclass(frozen=True)
class EquityChain:
    """A screened chain: what survived, what did not and why, and the underlying it hangs off.

    `rejected` is the LEDGER - `{security: verdict}` for every candidate that did not make it, the
    `build_map` discipline transferred: a candidate silently dropped is indistinguishable from one
    never asked about, and on a chain of two thousand names that difference is the whole report.
    """
    underlying: str
    name: str
    spot: float
    as_of: datetime.date
    contracts: tuple[ChainContract, ...]
    rejected: Mapping[str, str]
    #: WHEN THE SPOT ITSELF LAST PRINTED. `fetch_equity_chain` screens it against `stale_days` on
    #: the way in and refuses by name, so a fetched chain always carries a date the screen believed;
    #: it is optional only because the selection is drivable on a hand-built chain. It travels into
    #: `Quote_Source` beside the spot, because a believed chain should say how old its anchor is.
    spot_as_of: datetime.date | None = None

    @property
    def expiries(self) -> tuple[datetime.date, ...]:
        """The expiries the SURVIVING contracts carry - what a refusal names, because the chain's
        own listed dates are what a desk would have to go and look at."""
        return tuple(sorted({contract.expiry for contract in self.contracts}))


@dataclass(frozen=True)
class EquityForward:
    """WHICH CURVE FEEDS THE CARRY, declared - the row's genuinely new work.

    The strikes hang off the forward and so does every weight, and the FIT rebuilds that forward
    from the two curves this names: `spot * exp((r - q) t)` with `r` the `Discount_Rate` factor and
    `q` the `Yield` one. If the emitter placed its ladder on a different forward than the fit
    prices on, the calibration is fitted at coordinates the pricer never visits - the
    `Steps_Per_Year` mismatch one axis over. So the NAMES travel into the block for the fit to
    resolve, and the NUMBERS the emitter actually used travel into `Quote_Source` beside them.

    THE PRICER'S EQUITY FORWARD IS REPO MINUS DIVIDEND. `calc_eq_forward` reads the `Equity_Zero`
    curve - which is `EquityPrice.Interest_Rate`, falling back to the equity's `Currency` - against
    a `DividendRate`. So `discount_rate` should name THAT curve rather than the payoff's discount
    curve, or the calibrated forward and the priced one part company. Where an index carries a repo
    spread the two are not the same curve, and the family has ONE `Discount_Rate` reference doing
    both jobs (it funds the forward AND discounts the premium) - which is finding #2 of this build,
    not something an emitter can fix from out here.

    `dividend_yield` None means TAKE IT FROM THE CHAIN by put-call parity, which is the honest
    default where the book carries no dividend curve yet; a declared number is used as declared,
    with the parity-implied one measured beside it and reported.
    """
    underlying_factor: str
    volatility_factor: str
    discount_rate: str
    dividend_reference: str
    rate: float
    dividend_yield: float | None = None
    underlying_type: str = HN_REFERENCE_TYPES['Underlying']
    volatility_type: str = HN_REFERENCE_TYPES['Volatility']
    discount_rate_type: str = HN_REFERENCE_TYPES['Discount_Rate']
    dividend_type: str = HN_REFERENCE_TYPES['Yield']

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

    THE PILLARS REACH THE PRODUCT HORIZON. Equity autocalls run three to five years, so the ATM
    rungs default to 3M/6M/1Y/2Y/3Y where the FX ladder stops at 1Y. That is not a preference: one
    omega cannot hold a multi-year ATM term structure, the component family's L curve can, and the
    pillars are what identify it.

    THE WINGS ARE WIDENED, one pillar short of the ATM ladder. `HestonNandiComponentModelParameters`
    widened FX's two wing expiries to four for the reason that transfers exactly: the ATM rungs are
    SPENT on the L pillars, so what identifies the five free globals is what is left, and four wing
    expiries is what that wants. The 3Y wing is dropped rather than the 3M one because that is
    where a listed chain thins out first.
    """
    pillars: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0)
    wing_pillars: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    #: The wing's delta, placed as a MONEYNESS BAND off the forward under the chain's own ATM
    #: implied vol (see `select_rungs`). A listed chain has no delta axis to read, so the pillar is
    #: a target the rung then SNAPS off - it names where to look, never what was quoted.
    wing_delta: float = 0.25
    #: The quoted spread, as a fraction of mid, past which a print is not a market. A quarter of
    #: mid is wide for an index ATM and ordinary for a far wing, which is why it is one cap over a
    #: whole chain rather than a per-rung judgement - and why it ALSO enters the weight, so a
    #: contract that barely survived is worth half of one that is locked.
    spread_cap: float = 0.25
    #: OI > 0, the roadmap's own wording. Zero open interest is a listed contract nobody holds.
    minimum_open_interest: float = 1.0
    #: `discover.STALE_DAYS`, for its reason. A strike whose last update is three weeks old still
    #: answers a plausible price, and the date is the only thing that says otherwise.
    stale_days: int = 5
    #: DISTINCT contracts the ladder must survive snapping with -
    #: `HestonNandiComponentModelParameters.fx_minimum_contracts`, transferred verbatim: the ATM
    #: rungs are consumed by the L bootstrap, so what identifies the globals is what is left.
    minimum_contracts: int = 8
    #: How far PAST the ladder's longest rung a listed expiry may still be snapped to. A month,
    #: because that is the width of the same quarterly listing rolled once - not a second expiry.
    expiry_tolerance: float = 31.0 / 365.0
    #: How far a rung may MOVE, in log-expiry, before it is dropped instead of snapped. FX relies
    #: on the cap plus the distinct-contract floor alone; on a ladder that spans 3M to 3Y an
    #: unconditional argmin would land the 3Y rung on the 3M listing of a stub chain and call it a
    #: term structure. Half a log-unit is the 1Y rung reaching 6M or 1.6Y.
    pillar_band: float = 0.5
    #: The day count the emitter measures its OWN year fractions in, for placing strikes and
    #: weighting. The FIT recomputes `t` off `Expiry_Date` through the discount curve's day count;
    #: the two agree exactly under ACT_365 and differ by a day count otherwise - which moves the
    #: weight and NEVER the contract, because the contract is a listed one and was snapped, not
    #: computed. That is strictly better than the FX emitter can do, where the strike itself is a
    #: function of the accrual.
    days_per_year: float = 365.0
    #: The step clock and the inversion width the block STATES. Read off the field declarations.
    steps_per_year: float = STEPS_PER_YEAR
    quadrature_panels: int = QUADRATURE_PANELS
    #: The flat vol the wing bands fall back to where an expiry's own ATM contract yields no
    #: admissible Black implied vol. A seed for a coordinate, never a price.
    reference_vol: float = 0.20
    #: How many two-sided strikes the chain's own parity carry is MEDIANED over, and the band it is
    #: BELIEVED inside where the caller declared no dividend yield. See `PARITY_STRIKES` /
    #: `PARITY_BAND`: an undeclared carry is read off the chain, and a number read off the chain is
    #: evidence like any other - one print may not move a pillar's forward, and a neighbourhood that
    #: implies a two-figure negative carry is a bad chain rather than a market.
    parity_strikes: int = PARITY_STRIKES
    parity_band: tuple[float, float] = PARITY_BAND

    def __post_init__(self):
        # coerced to tuples the way `FXVolDefinition` coerces its own, so a caller who hands in a
        # list gets a frozen ladder rather than a frozen handle on something still mutable
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
    """`exp(-r t) F n(d1) sqrt(t)` - `HestonNandiModelParameters.fx_black_vega`, transferred.

    THE WEIGHT, before the liquidity factor and before normalisation. Vega is what a desk's risk in
    a quote actually IS, and it is what makes the objective scale-free across a term structure that
    now runs to three years: a 3M premium is a fraction of a 3Y one, and an unweighted least
    squares would fit the back end and leave the front to fend for itself. Puts and calls share it.
    """
    if tau <= 0.0 or vol <= 0.0 or forward <= 0.0 or strike <= 0.0:
        return 0.0
    stddev = vol * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * stddev * stddev) / stddev
    return math.exp(-rate * tau) * forward * NORMAL.pdf(d1) * math.sqrt(tau)


def implied_vol(price, forward, strike, rate, tau, is_call, iterations=100,
                low=1e-6, high=5.0):
    """The Black vol of a quoted premium, or None where the premium is not one.

    Bisection, because the price is monotone in the vol and a bracket that does not straddle is
    the ANSWER rather than a starting point: a mid below the forward's own intrinsic is not a
    premium any vol reaches. The caller falls back to the ladder's flat proxy, which is what a
    WEIGHT may do and a price may not - the screen's own `off-market` verdict is model-free and
    has already refused the prints that cannot be prices at all.
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
    """The two readers a chain needs: the BULK one for the chain's own membership, the tolerant
    scalar one for everything asked of a member. Both are `BloombergSession`'s, and both are
    tolerant, because on a chain of two thousand names one refused ticker is the FINDING and not
    the failure - the strict policy is applied here, per contract, by the screen."""

    def bulk_reference_data_report(self, securities: Sequence[str],
                                   fields: Sequence[str]) -> Mapping[str, Mapping[str, object]]:
        ...

    def reference_data_report(self, securities: Sequence[str],
                              fields: Sequence[str]) -> Mapping[str, Mapping[str, object]]:
        ...


def _number(value):
    """A terminal value as a finite float, or None. Absent, blank, unparseable and non-finite all
    read as ABSENT - `fxvol._scaled_side`'s rule, and for its reason: the one thing this must never
    do is manufacture a number out of a blank."""
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
    ways per asset class ('European', 'EUROPEAN', 'Euro'); anything it does not recognise is the
    EMPTY string, which the screen refuses - an unrecognised style is not a European one."""
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
    stays free of `security_map` and therefore of pandas (the `security_map.home()` precedent: one
    pattern, deliberately re-spelled per boundary rather than imported across one). `on_batch(done,
    total)` counts names REPLIED ABOUT, never names sent, because a full index chain is thousands
    of names over a terminal that takes its time."""
    report = {}
    for start in range(0, len(securities), batch):
        report.update(source.reference_data_report(securities[start:start + batch], fields))
        if on_batch is not None:
            on_batch(min(start + batch, len(securities)), len(securities))
    return report


def chain_members(row, chain_field=CHAIN_FIELD):
    """`(members, unreadable)` off one bulk `OPT_CHAIN` answer.

    A bulk row is a dict of Bloomberg's own sub-field names, and `Security Description` is the one
    that carries the ticker - tried by name first, then the row's only string value, because the
    sub-field's spelling is Bloomberg's and not a contract this package can pin. A row neither
    route can read a ticker out of is returned as `unreadable` rather than skipped, so it reaches
    the ledger like every other refusal.
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
    anything: `screen_chain` owns the order of distrust, and keeping the two apart is what lets the
    gates drive the screen on canned rows with no session in sight."""
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
    what it is worth, and what it is worth before how well anyone knows it. Reading them in this
    order is what makes a verdict actionable - `american` and `wide` are different instructions to
    a desk, and a screen that checked the spread first would report the second where the first is
    true.

      malformed          no strike, no expiry, or no put/call - not a contract
      expired            an expiry at or before the as-of - not a quote
      unstated-exercise  no exercise style: absence cannot prove EUROPEAN
      american           the roadmap's V1 ruling - an American premium is not the European one
      unpriced           no two-way and no last print
      one-sided          a side missing or non-positive: a spread nobody quoted is not a spread
      crossed            bid above ask - a stale side left standing against a live one
      off-market         a price that cannot be one, on the two MODEL-FREE bounds a chain can be
                         held to with no curve in sight: a call is never worth more than the
                         underlying, a put never more than its strike. Everything sharper than
                         that needs a forward, and a forward is the EMITTER's declared business
      wide               a spread past the declared cap: quoted, but not a market
      no-open-interest   OI absent or zero - a listed contract nobody holds
      undated            no LAST_UPDATE_DT: a print that cannot evidence its own time
      stale              a last update older than `stale_days` - the SAONIA rule, per strike
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
    # PARSED, not merely present: a `LAST_UPDATE_DT` of 'N/A' evidences a print's time exactly as
    # well as a blank one does, and it would otherwise ride into the block as the row's `Timestamp`
    # and refuse at the decoder instead of here
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
    to state, which is exactly why the bulk route exists and why the SCREEN, not a grammar, is
    where the trust boundary sits.

    The spot refuses BY NAME on anything that is not a positive number, which is
    `security_map.fetch_fx_spot`'s own policy: every strike, every forward and every weight below
    hangs off it, and a ladder built on a blank is worse than no ladder.

    AND ON ITS OWN CLOCK, which is the same policy read to the end. A blank spot is the loud
    failure; the quiet one is a spot that answers a plausible number nineteen years after it last
    printed - `discover`'s SAONIA, one axis over. Every listed contract in this chain is screened
    against `stale_days` and refused `undated` if it cannot evidence its own time; the number the
    whole ladder is placed with is held to exactly that bar, and the date it passed on travels into
    the block. The asymmetry with `discover._is_stale` ("absence cannot prove staleness") is the
    same one the contract screen already makes and for the same reason: absence cannot prove death,
    but this is the one price nothing downstream can cross-check.
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

    # THE POLICY IS APPLIED CLIENT-SIDE, and this is `fxvol._value_of`'s lesson paid for a second
    # time. The tolerant reader answers `ok: False` on ANY per-security trouble - a securityError
    # AND a mere fieldException - and on a chain that is the ordinary case rather than the broken
    # one: a listed contract that has not traded today carries no VOLUME, which is a field
    # exception, which would have thrown the whole contract away with it. MEASURED on a live SPX
    # chain: reading `ok` cost 1,855 of 8,000 contracts, most of them answering every field the
    # emitter needs. So a row with ANY field in it is read, and what judges it is the SCREEN -
    # which is where the judging belongs. A row with NOTHING in it is the genuine refusal.
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

    The strikes and the per-strike readings travel with the number because this is the one foreign
    answer in the module that is not a per-contract quote: where the caller declared no dividend
    yield it becomes the pillar's carry, and a refusal that named only the median would be telling
    a desk that its chain is wrong without saying where to look.
    """
    value: float
    strikes: tuple[float, ...]
    readings: tuple[float, ...]


def parity_dividend_yield(contracts, spot, rate, tau, strikes=PARITY_STRIKES):
    """The dividend yield the CHAIN ITSELF implies at one expiry, by put-call parity, or None.

    `C - P = (F - K) exp(-r t)` wherever BOTH legs survived the screen, so `F = K + (C - P) exp(r
    t)` and `q = r - ln(F/S)/t`. Parity holds at every strike, so this is one reading per two-sided
    pair rather than a fit - and what comes back is their MEDIAN, over the `strikes` pairs nearest
    the forward.

    A MEDIAN OVER SEVERAL, NOT ONE PAIR, and that is the difference between a check and a hazard.
    Where the caller declared no dividend yield this number IS the pillar's carry: it places every
    strike on that pillar and weighs every one of them. One fat-fingered near-money print - tight,
    deep, dated today, and therefore BELIEVED by every per-contract screen there is - moves a
    single-pair read by percentage points and takes the whole pillar's forward with it. A median
    ignores it entirely; that is the whole reason for the estimator.

    NEAREST THE FORWARD, IN TWO PASSES. The strike nearest the SPOT has one deep-ish in-the-money
    leg, which is the leg a chain quotes worst and the one `_snap_strike` never emits. But the
    forward is what this function is measuring, so the neighbourhood is chosen from a first pass
    around the spot and then re-read around the forward that pass implies - two medians and no
    iteration, because parity is exact at every strike and the second pass is about QUOTE QUALITY
    rather than convergence.

    It stays a CHECK wherever a dividend yield was declared: the declared number places the ladder
    and this one is reported beside it, because a book's dividend curve disagreeing with the chain
    by a point is a fact a desk should be told rather than have quietly averaged away.
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

    The CAP is FX's: an argmin has no ceiling, so without it a chain whose only listings are two
    months long would answer every rung of a three-year ladder and the fit would be a 2M fit
    wearing a 3Y label. The BAND is this ladder's own: FX's rungs sit inside one year of each other
    and lean on the distinct-contract floor to catch a collapse, but 3M and 3Y are a log-unit and a
    half apart, and a 3Y rung landing on the 3M listing would pass a floor counted over strikes.
    Half a log-unit is the 1Y rung reaching 6M or 1.6Y.

    ONE EXPIRY, ONE PILLAR is the third, and a per-pillar argmin cannot state it. Two pillars whose
    nearest listing is the SAME date is not a hypothetical: an ordinary board lists quarterlies out
    a year and then jumps to LEAPS, so a 1Y and a 2Y pillar both land on one January listing, and
    where a 2Y LEAP is simply absent the 2Y and 3Y pillars both reach the 3Y one (0.41 log-units,
    inside the band). Letting them would emit ONE listed contract as TWO equations at double weight
    - which the two family spellings then read differently, the component bootstrap dropping a
    duplicate ATM by strike and discarding its weight while the plain family applies every row it is
    given - and would write an L strip carrying FEWER knots than the ladder declares pillars, which
    is the one thing the component family is here for. So the pillars and the listings are MATCHED,
    and a pillar left with nothing is DROPPED by name, the way a pillar outside the band already is.

    NEAREST CLAIM WINS, not shortest-first. The pairs are taken in order of log-distance, so the
    listing goes to the pillar it actually belongs to. Handing it to the shorter pillar instead
    would put a 3Y listing into the block as the 2Y rung, 0.41 log-units from its label, and drop
    the 3Y pillar the listing IS - a name the chain contradicts, in place of an honest gap. Ties
    break on the pillar and then the date, so the assignment is a function of the chain rather than
    of dictionary order.

    ONE METRIC, and it is LOG-EXPIRY throughout. A ladder spanning 3M to 3Y cannot judge nearness
    in years: a month's error is most of the front rung and a rounding error on the back one.
    Choosing linearly and banding logarithmically would also let the two disagree - the
    linear-nearest listing can be the log-farther of two admissible ones - so everything here reads
    the same distance.
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

    The OTM leg is the one a desk deals, and the fit is blind to the choice either way (both
    families price puts by parity off the call) - so the type follows the STRIKE rather than the
    rung's intention. A put wing that snaps above the forward on a coarse chain therefore emits the
    call at that strike, which is the contract that was actually quoted there.
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
    """`(rungs, notes, readings)` - THE SELECTION, and the one both family spellings are emitted
    from. Nothing below this line knows which family it is writing for.

    THE FORMULA, stated once:

      for each pillar T:
        E(T)  the listed expiry nearest T, at or under max(pillars) + expiry_tolerance and within
              pillar_band of T in log-expiry, and CLAIMED BY NO OTHER PILLAR (`assign_expiries`:
              nearest claim wins); a pillar with no such expiry is DROPPED with a note
        t     (E(T) - as_of) / days_per_year
        q(T)  the DECLARED dividend yield, or the MEDIAN parity carry over the parity_strikes
              two-sided pairs nearest the forward where none was declared - measured either way and
              reported beside the declared number, and where it IS the carry it is screened against
              parity_band rather than believed for being arithmetic
        F(T)  spot * exp((r - q(T)) * t)
        ATM rung   target strike F(T)
        sigma(T)   the Black implied vol of the SNAPPED ATM contract's own mid, off F(T)
        wing rungs, at each pillar in wing_pillars, target strikes
                   K+ = F exp(+sigma^2 t/2 + z sigma sqrt(t))   the delta-call wing
                   K- = F exp(+sigma^2 t/2 - z sigma sqrt(t))   the delta-put wing
                   z = Phi^-1(1 - wing_delta), i.e. 0.6745 at the 25 delta
        every rung SNAPS to the listed contract at E(T) minimising |log(K_listed / K_target)|,
        taking the OTM leg at that strike

    WHY A MONEYNESS BAND AND NOT A DELTA SOLVE. FX inverts the surface's own delta because the
    surface WAS built by inverting it - the strike found that way is the strike the quote sits at.
    A listed chain has no delta axis at all: the wing is a place to LOOK, and what enters the block
    is whichever listed contract was found there. So the band is read off the chain's own ATM
    implied vol (the flat proxy only where that expiry's ATM yields none), and it names a
    coordinate rather than asserting a delta.

    THE WEIGHT, and it is a WEIGHT rather than a price:

        w = vega(F, K, r, sigma_K, t) * sqrt(OI) / (1 + spread / spread_cap)
        Weight = w / sum(w)

      * vega at the contract's OWN implied vol, off the declared forward - `fx_black_vega`,
        transferred, and for its reason: it is what makes the objective scale-free across a term
        structure that now runs to three years.
      * sqrt(OI) because open interest is evidence of ATTENTION and not a precision - a contract
        with a hundred times the open interest is worth ten times the weight, not a hundred, and a
        linear read would let one deep-liquid strike own the objective.
      * the spread factor runs from 1 at a locked market to 1/2 at the cap, so the tightest print
        is worth twice the widest one that survived the screen and NOTHING that survived is worth
        zero. Liquidity joins the vega weight because half a chain is dead strikes - the roadmap's
        own addition to the FX rule.
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
    """The chain's own parity carry, SCREENED before it becomes a pillar's carry - the order of
    distrust applied to the one foreign answer here that is not a per-contract quote.

    A DECLARED `EquityForward.dividend_yield` never reaches this function: a declared number is the
    caller's to own, and the chain's own reading is measured beside it and reported. With nothing
    declared the parity reading IS the carry - it places every strike on the pillar and weighs every
    one of them - so it is screened like any other evidence rather than believed for being
    arithmetic. The median has already thrown out one bad print (see `parity_dividend_yield`); the
    band is what catches a whole neighbourhood that is wrong together, and it refuses with the
    pillar, the number and the strikes it was read off, because "the chain is wrong" without a
    coordinate is not something a desk can act on.
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

    A REPEATED CONTRACT IS A WEIGHT, NOT A SECOND EQUATION - and the collapse happens HERE, before
    emission, precisely because the two family spellings do not agree about it. The component
    bootstrap deduplicates a repeated ATM by strike within an expiry and DISCARDS that row's weight;
    the plain family applies every row it is given. So one duplicated print is dropped by one family
    and double-counted by the other off the same block - a block that reads two ways is not one
    selection with two spellings, which is the promise this emitter makes.

    `assign_expiries` has already made the cross-pillar collision impossible. What survives is the
    collision WITHIN a pillar: a wing whose moneyness band falls inside the listed grid's own
    spacing, or a wing whose own strike the screen refused and which snapped back onto the ATM
    contract. Summing is what the objective would have done with the two rows anyway; the note names
    which rung was absorbed into which, because a rung that quietly vanished is a rung nobody can
    audit against the ladder that asked for it.
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
    caller over the ladder as emitted. The vega is taken at the contract's own implied vol where
    the mid yields one and at the ladder's flat proxy where it does not, because a contract that
    survived the screen still has a spread and an open interest and should not silently weigh
    nothing."""
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

    The wire form rather than a `pandas.Timestamp` on purpose: a block is posted as JSON (through
    `/book/market`, through `config.update_market_quote`, through a file), the decoder turns this
    into the `Timestamp` the fit subtracts a base date from, and spelling it here is what keeps
    this module free of pandas. Whatever the terminal's own spelling was, what is written is a
    parsed ISO date - so the block is the same bytes whichever way the field came back.
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
    `European_Options` table it produces is identical between them, byte for byte; what differs is
    only the header each family DECLARES. `collapse_rungs` is what keeps that true of the two
    families' READING of the table as well: a contract two rungs landed on is emitted ONCE at their
    summed weight, because the component family and the plain one do different things with a
    duplicate row. The component spelling additionally states `Rho` (a pin
    nobody can see is not a declared pin) and `Quote_Sensitivity` No (that family refuses Yes, and
    saying so is honesty), because those are its own fields and the plain family declares neither.

    PREMIUMS, NOT VOLS. `Quote_Type` is Premium and `Quoted_Market_Value` is the mid of the
    terminal's two-way - the print, in the underlying's own units. `Quoted_Bid`, `Quoted_Ask` and
    `Timestamp` ride beside it: they are `schema.MARKET_QUOTE_VALUES` on every other family's quote
    row and the engine's `OPTION_QUOTE` declares none of the three, so they travel as undeclared
    keys that `bootstrap` reads past. Carried anyway, and named as a finding, because the two-way
    and the print's own clock ARE the evidence - an emitter that dropped them to fit a six-column
    schema would have thrown away the desk's spread on the way in.

    `Use_Forward` and `Invert_Moneyness` are written at their declared defaults and are INERT here:
    both exist to look a vol surface up AT A STRIKE, and under `Quote_Type` Premium no surface is
    read at all. The block still has to NAME a `Volatility` factor, which is finding #1.

    Refuses by name, with the remedy, on: a chain whose census says exercise style is what killed
    the ladder (`UnsupportedExerciseStyle`), a ladder that collapses below the component family's
    floor of eight DISTINCT contracts (`IncompleteChain`, naming the chain's own expiries), an
    expiry with no admissible dividend evidence or an undeclared carry outside `parity_band`, and a
    ladder with no weight in it.
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
        'Quote_Type': 'Premium',
        'Use_Forward': 'No', 'Invert_Moneyness': 'No',
        'Steps_Per_Year': ladder.steps_per_year,
        'Quadrature_Panels': ladder.quadrature_panels,
        'Quote_Timestamp': _timestamp(chain.as_of),
        'Quote_Source': quote_source(chain, forward, ladder, rungs, rows, notes, readings),
        'European_Options': quotes}
    if family == COMPONENT_FAMILY:
        instrument.update(COMPONENT_HEADER)
        # the option table stays LAST whichever family this is, so the two spellings read as one
        # block with a different header rather than as two differently shaped documents
        instrument['European_Options'] = instrument.pop('European_Options')
    return market_price_name(family, forward), {'instrument': instrument}


def _refuse_american(chain, ladder):
    """The roadmap's V1 ruling, fired off the CENSUS rather than off an empty chain.

    THE PER-CONTRACT SCREEN IS RIGHT AND STAYS. A mixed index board - a flex or weekly American
    listing beside European ones - drops those contracts individually at `_verdict` and the European
    ones still calibrate. That is not what this function is for.

    WHAT THE CENSUS ADDS IS THE NEAR MISS. Gating this on `contracts` being EMPTY meant a chain that
    was American except for a handful of survivors raised the distinct-contract floor instead: 191
    American listings and one European survivor produced "the ladder collapses onto 1 distinct
    contract ... quote the chain at more expiries and strikes" about a chain listing six expiries
    and sixteen strikes. That names the symptom and prescribes the wrong remedy - the exact thing
    this refusal exists to avoid - and one survivor was all it took.

    So it fires when the chain cannot reach the floor AT ALL and exercise style is what took it
    there: more candidates refused on style than survived the whole screen. Below the floor with the
    refusals elsewhere, the floor's own refusal is the true one and names its own census.
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
    they produce - `fx_surface_block`'s `Quote_Source`, carrying what a chain has that a surface
    does not: the census, the spot's own print date, the declared carry, and the carry the chain
    itself implies.

    THE DIVIDEND DISAGREEMENT IS REPORTED RATHER THAN RESOLVED. Where the caller declared a yield,
    that is the one the strikes were placed on and the one the fit will rebuild its forward from;
    the parity-implied number beside it is the chain's own opinion, and the gap between them is a
    fact about the book's dividend curve that belongs in the record rather than in an average. A
    parity reading outside `parity_band` is NAMED here rather than refused: where the yield was
    declared, the band screens nothing (the declared number placed the ladder) and an out-of-band
    chain reading is exactly the disagreement a desk is owed.

    THE ROWS ARE COUNTED BESIDE THE RUNGS because they are not always the same number: two rungs
    landing on one listed contract are emitted once at their summed weight, and a report that said
    "13 rungs" over an eleven-row table would be describing a block that does not exist.
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
        '{:.6g} (last printed {}) carried at r={:.4%} on {} against {} [{}]'.format(
            len(rungs), '/'.join(sorted({rung.kind for rung in rungs})), len(rows),
            chain.underlying, chain.as_of.isoformat(), len(chain.contracts),
            len(chain.contracts) + len(chain.rejected),
            ', '.join('{} {}'.format(count, verdict)
                      for verdict, count in sorted(census.items())) or 'nothing refused',
            chain.spot,
            'undated' if chain.spot_as_of is None else chain.spot_as_of.isoformat(),
            forward.rate, forward.discount_rate, forward.dividend_reference,
            dividends or 'no pillar priced'))
    if notes:
        source += '; rungs the chain does not list, moved or dropped: {}'.format(', '.join(notes))
    return source
