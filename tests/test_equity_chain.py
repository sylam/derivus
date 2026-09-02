"""The listed equity option CHAIN as a Heston-Nandi quote block - `derivus_bloomberg.equity_chain`.

Everything but the last gate runs on ONE canned chain: 192 listed contracts over six expiries,
mixed liquidity, both exercise styles, and a poison table of dead prints authored on purpose. No
monkeypatching - the canned terminal is a `BloombergSession` subclass whose event walks yield rows
(`test_bloomberg_discover.Walked`), and the engine is imported and never touched.

  the budget       `equity_chain` imports the standard library, this package's own modules and a
                   LAZY blpapi - read off the source and proved again in a fresh interpreter
  the screen       the order of distrust, one contract per verdict, and a census of the whole
                   canned chain by name: a candidate silently dropped is indistinguishable from
                   one never asked about
  the policy       applied CLIENT-SIDE: a contract the tolerant reader flagged for a FIELD
                   exception is still read, and only a row with nothing in it is `invalid`
  American         a chain that screened to nothing but American exercise refuses BY NAME with the
                   remedy - an American premium is not the European premium the fit prices against
  the selection    the ATM rung is the listed strike nearest its own forward, in the metric the
                   ENGINE re-derives the ATM/wing split with; the wings are 25-delta-equivalent
                   bands under the chain's own ATM implied vol; a dead print at a wing moves the
                   rung to the next listed strike
  the floor        eight DISTINCT contracts, the component family's own number
  the block        every key is a field the family DECLARES, every `_Type` one of its candidates,
                   and the option row's value keys are `MARKET_QUOTE_VALUES` less the mid
  the round trip   `update_market_quote` installs, updates, takes a VALUE-ONLY re-tick and refuses
                   a moved strike; the tick moves `values_hash` with `plan_hash` hex-identical
  the weights      normalised vega x sqrt(open interest) / (1 + spread/cap)
  determinism      the same canned chain emits the same bytes
  end to end       the component bootstrap off a real JSON document in a book with NO authored
                   surface - the circularity gone
  the refusals     a reference a quote type requires and the block does not name refuses BY NAME,
                   writes no factor, and lands no 'skipping' line
  the forward      the fit's forward off the `Strike` column it fills in, against
                   `calc_eq_forward`'s through an `EquityForwardDeal` in the same job
"""
import ast
import datetime
import json
import math
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_bloomberg import equity_chain
from derivus_bloomberg.equity_chain import (ChainContract, EquityForward, EquityLadder,
                                            black_price, equity_hn_block, fetch_equity_chain,
                                            screen_chain, select_rungs)
from derivus_bloomberg.errors import (BloombergConfigurationError, IncompleteChain, InvalidQuote,
                                      UnsupportedExerciseStyle)
from derivus_bloomberg.session import BloombergSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AS_OF = datetime.date(2026, 8, 31)
UNDERLYING = 'SPX Index'
SPOT = 5000.0
RATE, DIVIDEND = 0.04, 0.015

#: Six listed expiries: five on the ladder's pillars (3M/6M/1Y/2Y/3Y) and a front one on none of
#: them, because a listed expiry the ladder never asks for is part of what a chain looks like.
EXPIRIES = (datetime.date(2026, 9, 30), datetime.date(2026, 11, 30), datetime.date(2027, 2, 28),
            datetime.date(2027, 8, 31), datetime.date(2028, 8, 31), datetime.date(2029, 8, 31))

#: Sixteen strikes per expiry per side, finer near the money. 16 x 2 x 6 = 192 contracts.
RATIOS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975,
          1.00, 1.025, 1.05, 1.075, 1.10, 1.15, 1.20, 1.30)


# =============================================================================================
# the canned chain
# =============================================================================================

def tau_of(expiry):
    return (expiry - AS_OF).days / 365.0


def forward_of(expiry):
    return SPOT * math.exp((RATE - DIVIDEND) * tau_of(expiry))


def chain_vol(expiry, strike):
    """The vol the fixture's prices are generated FROM: a rising ATM term structure with a steep,
    one-signed, slightly convex skew - the index shape, and the case the component family's
    declining-variance guard does not refuse."""
    moneyness = math.log(strike / forward_of(expiry))
    return 0.16 + 0.02 * math.sqrt(tau_of(expiry)) - 0.35 * moneyness + 0.6 * moneyness ** 2


def security_of(expiry, strike, option_type):
    return 'SPX {} {}{:g} Index'.format(expiry.strftime('%m/%d/%y'),
                                        'C' if option_type == 'Call' else 'P', strike)


def _tick(value):
    """A listed price sits on a tick; a fixture quoting eleven decimals is not a print."""
    return round(value * 20.0) / 20.0


def _row(expiry, strike, option_type):
    """One clean listed contract: a Black price off the fixture's vol, ticked, with a two-way that
    tightens toward the money and an open interest that thins away from it."""
    tau, forward = tau_of(expiry), forward_of(expiry)
    mid = _tick(black_price(forward, strike, RATE, chain_vol(expiry, strike), tau,
                            option_type == 'Call'))
    moneyness = abs(math.log(strike / forward))
    half = max(_tick(mid * (0.004 + 0.05 * moneyness)), 0.05)
    interest = max(int(round(40000.0 * math.exp(-12.0 * moneyness * moneyness))), 25)
    return {'OPT_STRIKE_PX': strike, 'OPT_EXPIRE_DT': expiry.isoformat(),
            'OPT_PUT_CALL': option_type, 'OPT_EXER_TYP': 'European',
            'PX_BID': round(mid - half, 2), 'PX_ASK': round(mid + half, 2), 'PX_LAST': mid,
            'OPEN_INT': interest, 'VOLUME': max(int(interest / 20), 1),
            'LAST_UPDATE_DT': '2026-08-31'}


#: THE POISON TABLE, one entry per verdict the screen can reach, keyed by
#: `(expiry index, ratio, side)`. Every one is a shape a real chain carries.
POISON = {
    # crossed - a stale bid left standing above a live offer
    (1, 0.80, 'Put'): {'PX_BID': 40.0, 'PX_ASK': 30.0},
    (3, 1.20, 'Call'): {'PX_BID': 60.0, 'PX_ASK': 55.0},
    # one-sided - the terminal quoted a bid and no offer
    (2, 0.70, 'Put'): {'PX_ASK': None},
    (4, 1.30, 'Call'): {'PX_ASK': ''},
    (0, 0.70, 'Put'): {'PX_BID': None},
    # no open interest - a listed strike nobody holds
    (5, 0.70, 'Put'): {'OPEN_INT': 0},
    (5, 1.30, 'Call'): {'OPEN_INT': 0},
    (4, 0.70, 'Put'): {'OPEN_INT': ''},
    # stale and undated - the SAONIA rule, per strike. The undated one sits NEAR THE MONEY on
    # purpose: the order of distrust means a far wing would have been refused for its spread long
    # before anything looked at its clock, and a verdict no fixture can reach is not gated.
    (3, 0.75, 'Put'): {'LAST_UPDATE_DT': '2026-07-15'},
    (4, 0.75, 'Put'): {'LAST_UPDATE_DT': '2026-06-02'},
    (2, 1.05, 'Call'): {'LAST_UPDATE_DT': ''},
    # wide - quoted, but not a market
    (1, 1.30, 'Call'): {'PX_BID': 0.20, 'PX_ASK': 1.60, 'PX_LAST': 0.90},
    (0, 1.20, 'Call'): {'PX_BID': 0.05, 'PX_ASK': 0.60, 'PX_LAST': 0.30},
    # unpriced - it resolves, it has an open interest, and it is worth nothing anyone will say
    (3, 0.975, 'Put'): {'PX_BID': None, 'PX_ASK': None, 'PX_LAST': None},
    # exercise style: AMERICAN, and one that states none at all
    (5, 0.85, 'Put'): {'OPT_EXER_TYP': 'American'},
    (5, 0.90, 'Put'): {'OPT_EXER_TYP': 'AMERICAN'},
    (5, 1.15, 'Call'): {'OPT_EXER_TYP': 'American'},
    (4, 1.20, 'Call'): {'OPT_EXER_TYP': 'Amer'},
    (3, 0.70, 'Put'): {'OPT_EXER_TYP': ''},
    (2, 1.20, 'Call'): {'OPT_EXER_TYP': 'Bermudan'},
    # malformed and expired - not a contract, and not a quote
    (1, 0.75, 'Put'): {'OPT_STRIKE_PX': ''},
    (2, 0.75, 'Put'): {'OPT_EXPIRE_DT': '2026-08-15'},
    # off-market - a call marked above the index itself, which is the model-free bound and needs
    # no curve to see: a right to buy the underlying is never worth more than the underlying
    (1, 0.70, 'Call'): {'PX_BID': 5100.0, 'PX_ASK': 5200.0, 'PX_LAST': 5150.0},
}


def canned_rows(poison=None, expiries=EXPIRIES, ratios=RATIOS):
    """`{security: fields}` for the whole canned chain, the poison table applied on top."""
    poison = POISON if poison is None else poison
    rows = {}
    for index, expiry in enumerate(expiries):
        for ratio in ratios:
            strike = round(SPOT * ratio, 2)
            for option_type in ('Call', 'Put'):
                fields = _row(expiry, strike, option_type)
                fields.update(poison.get((index, ratio, option_type), {}))
                rows[security_of(expiry, strike, option_type)] = fields
    return rows


class Walked(BloombergSession):
    """A session whose event walks are canned rows - `test_bloomberg_discover.Walked` with the BULK
    walk canned too, so the tolerance, the per-name filling and the batching all run."""

    def __init__(self, rows, chain=None, underlying=None, errors=None):
        super().__init__()
        self._api = self._session = self._service = object()  # started, as far as the guard cares
        self.rows = rows
        #: `{security: Bloomberg's own text}` for names the terminal flagged while STILL answering
        #: their fields - a fieldException, which the tolerant reader reports as `ok: False`
        self.errors = errors or {}
        self.chain = list(rows) if chain is None else chain
        self.underlying = underlying if underlying is not None else {
            'NAME': 'S&P 500 INDEX', 'PX_LAST': SPOT, 'LAST_UPDATE_DT': '2026-08-31',
            equity_chain.CHAIN_FIELD: [{'Security Description': name} for name in self.chain]}
        self.batches = []

    def _walk(self, securities, fields):
        self.batches.append(tuple(securities))
        for security in securities:
            if security in self.rows:
                yield security, self.errors.get(security), dict(self.rows[security])
            else:
                yield security, 'Unknown/Invalid Security', {}

    def _walk_bulk(self, securities, fields):
        for security in securities:
            yield security, None, dict(self.underlying)


def canned_chain(**kwargs):
    """The canned chain fetched through the real reader, screened - what every gate below starts
    from."""
    return fetch_equity_chain(Walked(canned_rows(**kwargs)), UNDERLYING, AS_OF)


FORWARD = EquityForward(
    underlying_factor='SPX', volatility_factor='SPX', discount_rate='USD',
    dividend_reference='SPX', rate=RATE, dividend_yield=DIVIDEND)


# =============================================================================================
# 1  the dependency budget
# =============================================================================================

def imported_names(source):
    """The top-level names a file imports, however deep the import sits. Relative imports are
    skipped: they resolve inside the package and carry no module name to judge."""
    names = set()
    with open(source, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.add((node.module or '').split('.')[0])
    return names


def test_the_chain_emitter_imports_the_standard_library_and_nothing_else():
    """A STRICTER budget than the package's own - `fxvol` and `types` carry pandas and this module
    does not - so `equity_chain` is held to the standard library, this package's own modules, and a
    blpapi imported LAZILY or not at all. An import that never executes is still a dependency, so
    this reads the SOURCE.
    """
    imported = imported_names(os.path.join(ROOT, 'derivus_bloomberg', 'equity_chain.py'))
    # the package's own modules are reached relatively, which `imported_names` skips;
    # `derivus_bloomberg` is allowed so an absolute intra-package import passes too
    assert imported <= {'collections', 'datetime', 'math', 'statistics', 'dataclasses', 'typing',
                        'derivus_bloomberg'}, sorted(imported)
    # blpapi is on this list on purpose: the package has to import on a machine with no terminal
    assert imported.isdisjoint({'derivus', 'torch', 'pandas', 'numpy', 'scipy', 'blpapi'}), \
        sorted(imported)
    # non-vacuous: a module that imported nothing would pass both assertions above
    assert imported


def in_a_fresh_interpreter(statements):
    """What `sys.modules` holds after a fresh interpreter runs `statements`."""
    code = ('import json, sys; {}; '
            'print(json.dumps(sorted({{name.split(".")[0] for name in sys.modules}})))'.format(
                statements))
    done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    assert done.returncode == 0, done.stderr
    return set(json.loads(done.stdout))


def test_importing_the_chain_emitter_lands_no_engine_no_blpapi_and_no_pandas():
    """The source gate's answer measured, because the parser reads ONE FILE while an import runs a
    PACKAGE: an eager `from .fxvol import ...` in `derivus_bloomberg/__init__.py` would land numpy
    and pandas behind this module's back with every line of `equity_chain.py` still innocent. The
    package re-exports the pandas-carrying names lazily, and blpapi is never imported at all.
    """
    landed = in_a_fresh_interpreter('import derivus_bloomberg.equity_chain')
    assert 'derivus_bloomberg' in landed, 'the module did not import'
    assert landed.isdisjoint({'derivus', 'torch', 'blpapi', 'pandas', 'numpy'}), sorted(
        landed & {'derivus', 'torch', 'blpapi', 'pandas', 'numpy'})

    # the re-export still WORKS, and the emitter is reachable off the package by name
    assert 'derivus_bloomberg' in in_a_fresh_interpreter(
        'from derivus_bloomberg import equity_hn_block, EquityLadder')
    # non-vacuous: asking the package for an FX name is what pays for pandas, so the gate above
    # measures a deferral rather than an absence
    assert 'pandas' in in_a_fresh_interpreter(
        'import derivus_bloomberg; derivus_bloomberg.fetch_fx_vol')


# =============================================================================================
# 2  the screen - the order of distrust
# =============================================================================================

def contract(**kwargs):
    base = dict(security='X', strike=5000.0, expiry=datetime.date(2027, 8, 31),
                option_type='Call', exercise='European', bid=99.0, ask=101.0, last=100.0,
                open_interest=1000.0, volume=10.0, last_update='2026-08-31')
    base.update(kwargs)
    return ChainContract(**base)


def test_the_screen_classifies_off_the_terminals_own_answers():
    """One contract per verdict, in the order the screen reads them. The ORDER is the claim as much
    as the verdicts: `american` and `wide` are different instructions to a desk, and a screen that
    checked the spread first would report the second where the first is true."""
    cases = {
        'malformed': contract(strike=None),
        'expired': contract(expiry=datetime.date(2026, 8, 15)),
        'unstated-exercise': contract(exercise=''),
        'american': contract(exercise='American'),
        'unpriced': contract(bid=None, ask=None, last=None),
        'one-sided': contract(ask=None),
        'crossed': contract(bid=110.0, ask=90.0),
        'wide': contract(bid=40.0, ask=160.0),
        'no-open-interest': contract(open_interest=0.0),
        'undated': contract(last_update=None),
        'stale': contract(last_update='2026-07-15'),
    }
    named = [contract(security=verdict, **{
        field: getattr(item, field) for field in
        ('strike', 'expiry', 'option_type', 'exercise', 'bid', 'ask', 'last', 'open_interest',
         'last_update')}) for verdict, item in cases.items()]
    live = contract(security='live')
    accepted, rejected = screen_chain(named + [live], AS_OF)

    assert [item.security for item in accepted] == ['live']
    assert rejected == {verdict: verdict for verdict in cases}
    # the ORDER where it bites: an American contract that is ALSO crossed and dead reads as
    # american, because that is the finding a desk can act on
    both = contract(security='both', exercise='American', bid=110.0, ask=90.0,
                    open_interest=0.0, last_update='2007-03-26')
    assert screen_chain([both], AS_OF)[1] == {'both': 'american'}

    # a print date PRESENT and unreadable evidences a print's time no better than a blank one, and
    # would otherwise ride into the block as that row's `Timestamp`
    assert screen_chain([contract(security='na', last_update='N/A')], AS_OF)[1] == {'na': 'undated'}

    # the model-free bounds, which need no curve - and need the spot, without which the screen
    # simply does not make the claim
    rich = contract(security='rich', bid=5100.0, ask=5200.0)
    assert screen_chain([rich], AS_OF, spot=SPOT)[1] == {'rich': 'off-market'}
    assert screen_chain([rich], AS_OF)[1] == {}


def test_the_canned_chain_is_believed_by_census():
    """The whole 192-contract fixture through the real reader, counted by verdict and named per
    family of refusal. A candidate silently dropped is indistinguishable from one never asked
    about, and on a chain this size that difference IS the report."""
    chain = canned_chain()
    assert len(chain.contracts) + len(chain.rejected) == len(RATIOS) * 2 * len(EXPIRIES) == 192
    census = {}
    for verdict in chain.rejected.values():
        census[verdict] = census.get(verdict, 0) + 1
    assert census == {'american': 4, 'crossed': 2, 'expired': 1, 'malformed': 1,
                      'no-open-interest': 3, 'off-market': 1, 'one-sided': 3,
                      'stale': 2, 'undated': 1, 'unpriced': 3, 'unstated-exercise': 2, 'wide': 6}
    assert len(chain.contracts) == 192 - sum(census.values()) == 163
    # six of the nine wide-or-unpriced are NOT in the poison table: far wings whose minimum tick is
    # a third of their mid, or whose price rounds to nothing. A chain carries dead strikes whether
    # anyone authored one or not
    natural = {security for security, verdict in chain.rejected.items()
               if verdict in ('wide', 'unpriced')}
    assert len(natural) == 9 and sum('09/30/26' in security for security in natural) == 5

    # named, so a re-tuned screen has to come here and say so
    assert chain.rejected[security_of(EXPIRIES[5], 4250.0, 'Put')] == 'american'
    assert chain.rejected[security_of(EXPIRIES[2], 6000.0, 'Call')] == 'unstated-exercise'
    assert chain.rejected[security_of(EXPIRIES[1], 4000.0, 'Put')] == 'crossed'
    assert chain.rejected[security_of(EXPIRIES[3], 3750.0, 'Put')] == 'stale'
    assert chain.rejected[security_of(EXPIRIES[2], 5250.0, 'Call')] == 'undated'
    assert chain.rejected[security_of(EXPIRIES[1], 3500.0, 'Call')] == 'off-market'
    assert chain.spot == SPOT and chain.name == 'S&P 500 INDEX'
    assert chain.expiries == EXPIRIES


def test_the_fetch_asks_the_bulk_reader_for_the_chain_and_batches_its_members():
    """TWO ROUND TRIPS AND NO SPELLED TICKERS. Membership comes off the BULK reader, because the
    scalar one answers row zero of an array and says nothing about the two thousand it dropped;
    the members batch through the tolerant scalar reader, so one refused ticker in a batch of fifty
    is the finding rather than the failure."""
    session = Walked(canned_rows())
    chain = fetch_equity_chain(session, UNDERLYING, AS_OF, batch=50)
    assert [len(batch) for batch in session.batches] == [50, 50, 50, 42]
    assert sum(len(batch) for batch in session.batches) == 192

    # the bulk reader's own contract: a LIST of rows, not row zero
    report = session.bulk_reference_data_report([UNDERLYING], [equity_chain.CHAIN_FIELD])
    members = report[UNDERLYING]['fields'][equity_chain.CHAIN_FIELD]
    assert isinstance(members, list) and len(members) == 192

    # a member the terminal refuses lands on the ledger BY NAME rather than vanishing
    rows = canned_rows()
    ghost = security_of(EXPIRIES[3], 5000.0, 'Call')
    del rows[ghost]
    refused = fetch_equity_chain(Walked(rows, chain=list(canned_rows())), UNDERLYING, AS_OF)
    assert refused.rejected[ghost] == 'invalid'
    assert chain.underlying == UNDERLYING


def test_a_field_exception_does_not_throw_the_contract_away_with_it():
    """THE POLICY IS APPLIED CLIENT-SIDE. The tolerant reader answers `ok: False` on ANY
    per-security trouble, and a fieldException is trouble: a contract that has not traded today
    carries no `VOLUME`, and reading `ok` would throw away a contract that answered everything
    else. On a real SPX chain that refused 1,855 of 8,000. So a row with ANY field in it is READ
    and the SCREEN judges it; only a row with nothing in it is `invalid`.
    """
    rows = canned_rows()
    flagged = security_of(EXPIRIES[3], 5125.0, 'Put')  # the 1y ATM rung
    chain = fetch_equity_chain(
        Walked(rows, errors={flagged: 'fieldExceptions: VOLUME - Field Not Applicable'}),
        UNDERLYING, AS_OF)
    assert flagged not in chain.rejected
    assert any(item.security == flagged for item in chain.contracts)
    # and the rung it carries is still selected, so the block is the one the clean chain writes
    assert equity_hn_block(chain, FORWARD)[1] == equity_hn_block(canned_chain(), FORWARD)[1]

    # a row with NO fields at all is the genuine refusal, error text or not
    empty = fetch_equity_chain(
        Walked({name: ({} if name == flagged else fields) for name, fields in rows.items()}),
        UNDERLYING, AS_OF)
    assert empty.rejected[flagged] == 'invalid'


def test_a_blank_spot_refuses_before_a_ladder_is_built_on_it():
    """Every strike, forward and weight hangs off the spot, so a chain cannot be built around a
    blank - and the refusal names the underlying rather than dying three functions later."""
    session = Walked(canned_rows())
    session.underlying = dict(session.underlying, PX_LAST=None)
    with pytest.raises(InvalidQuote, match='SPX Index'):
        fetch_equity_chain(session, UNDERLYING, AS_OF)

    session = Walked(canned_rows())
    session.underlying = dict(session.underlying, **{equity_chain.CHAIN_FIELD: []})
    with pytest.raises(IncompleteChain, match='OPT_CHAIN'):
        fetch_equity_chain(session, UNDERLYING, AS_OF)


def test_a_stale_spot_refuses_the_way_a_stale_contract_does():
    """THE ANCHOR IS SCREENED LIKE EVERYTHING ELSE. A blank spot is the loud failure; the quiet one
    is a spot answering a plausible number nineteen years after it last printed, while every
    contract placed against it is screened on exactly that field. Both halves: a spot older than
    `stale_days`, and a print date PRESENT and unreadable. The believed date travels into the
    block, so a chain says how old its anchor is.
    """
    session = Walked(canned_rows())
    session.underlying = dict(session.underlying, LAST_UPDATE_DT='2007-03-26')
    with pytest.raises(InvalidQuote) as refusal:
        fetch_equity_chain(session, UNDERLYING, AS_OF)
    message = str(refusal.value)
    assert 'SPX Index' in message and '2007-03-26' in message and 'stale' in message

    session = Walked(canned_rows())
    session.underlying = dict(session.underlying, LAST_UPDATE_DT='N/A')
    with pytest.raises(InvalidQuote, match='LAST_UPDATE_DT'):
        fetch_equity_chain(session, UNDERLYING, AS_OF)

    chain = canned_chain()
    assert chain.spot_as_of == AS_OF
    assert 'spot 5000 (last printed 2026-08-31)' in \
        equity_hn_block(chain, FORWARD)[1]['instrument']['Quote_Source']


def test_an_american_chain_refuses_by_name_with_its_remedy():
    """An American premium is not the European premium a Heston-Nandi fit prices against, so a
    single-name chain refuses rather than fitting the wrong number under the right name. The
    refusal names the underlying, the count, the style and the remedy; "eight distinct contracts"
    would name the symptom."""
    american = {(index, ratio, side): {'OPT_EXER_TYP': 'American'}
                for index in range(len(EXPIRIES)) for ratio in RATIOS
                for side in ('Call', 'Put')}
    chain = canned_chain(poison=american)
    assert chain.contracts == ()
    with pytest.raises(UnsupportedExerciseStyle) as refusal:
        equity_hn_block(chain, FORWARD)
    message = str(refusal.value)
    assert 'SPX Index' in message and 'american' in message
    assert 'European exercise' in message and 'SX5E Index' in message


def test_a_chain_that_is_american_but_for_one_survivor_still_refuses_on_exercise():
    """THE NEAR MISS, where a refusal gated on an EMPTY chain named the wrong cause: one European
    contract among 191 American listings fired the floor, which asks a desk to quote more of a
    chain already listing six expiries and sixteen strikes, and said neither `american` nor
    `exercise`.

    So the refusal reads the CENSUS - the chain cannot reach the floor and more candidates were
    refused on exercise style than survived. The per-contract screen is untouched: the second half
    asserts a mixed board still calibrates, so this cannot be "fixed" into refusing every chain
    with a flex listing on it.
    """
    survivor = (3, 1.00, 'Call')
    american = {(index, ratio, side): {'OPT_EXER_TYP': 'American'}
                for index in range(len(EXPIRIES)) for ratio in RATIOS
                for side in ('Call', 'Put') if (index, ratio, side) != survivor}
    chain = canned_chain(poison=american)
    assert len(chain.contracts) == 1 and len(chain.rejected) == 191

    with pytest.raises(UnsupportedExerciseStyle) as refusal:
        equity_hn_block(chain, FORWARD)
    message = str(refusal.value)
    assert 'SPX Index' in message and 'exercise style' in message and 'american' in message
    assert 'European exercise' in message and 'SX5E Index' in message
    assert '191 of its 191 candidates' in message
    # what it must NOT say: the floor's message and its wrong remedy
    assert 'distinct contract' not in message and 'more expiries and strikes' not in message

    # and the mixed board still calibrates, per contract, off what is left
    mixed = canned_chain()
    assert sum(1 for verdict in mixed.rejected.values()
               if verdict in ('american', 'unstated-exercise')) == 6
    assert len(equity_hn_block(mixed, FORWARD)[1]['instrument']['European_Options']) == 13


# =============================================================================================
# 3  the selection
# =============================================================================================

def test_the_selection_picks_the_documented_contracts():
    """THE LADDER, contract by contract: five ATM rungs on the ladder's pillars plus 25-delta wings
    at the first four, thirteen in all, and each one a LISTED contract rather than a coordinate.
    The ATM rung is the strike nearest its own forward with the type following the strike (the
    out-of-the-money leg is the one a desk deals); each wing is the listed strike nearest the
    moneyness band drawn under that expiry's OWN at-the-money implied vol."""
    chain = canned_chain()
    rungs, notes, readings = select_rungs(chain, FORWARD)

    assert [(rung.kind, rung.pillar, rung.contract.strike, rung.contract.option_type)
            for rung in rungs] == [
        ('ATM', 0.25, 5000.0, 'Put'),
        ('25d call', 0.25, 5375.0, 'Call'), ('25d put', 0.25, 4750.0, 'Put'),
        ('ATM', 0.5, 5125.0, 'Call'),
        ('25d call', 0.5, 5500.0, 'Call'), ('25d put', 0.5, 4750.0, 'Put'),
        ('ATM', 1.0, 5125.0, 'Put'),
        ('25d call', 1.0, 5750.0, 'Call'), ('25d put', 1.0, 4625.0, 'Put'),
        ('ATM', 2.0, 5250.0, 'Put'),
        ('25d call', 2.0, 5750.0, 'Call'), ('25d put', 2.0, 4500.0, 'Put'),
        ('ATM', 3.0, 5375.0, 'Put')]

    # the front listing the ladder never asks for is not used, and the pillars that MOVED say so
    assert {rung.contract.expiry for rung in rungs} == set(EXPIRIES[1:])
    assert [note for note in notes if 'DROPPED' in note] == []
    assert any('0.25y -> 2026-11-30' in note for note in notes)
    assert len({(rung.contract.expiry, rung.contract.strike) for rung in rungs}) == 13

    # the ATM rung is the one the ENGINE will call ATM: the component family re-derives the split
    # per expiry off the row nearest its own forward, so agreement has to hold in that metric
    for rung in rungs:
        if rung.kind != 'ATM':
            continue
        peers = [item for item in rungs if item.contract.expiry == rung.contract.expiry]
        assert rung.contract.strike == min(
            (item.contract.strike for item in peers),
            key=lambda strike: abs(strike / rung.forward - 1.0))
        assert readings[rung.pillar]['forward'] == pytest.approx(
            forward_of(rung.contract.expiry), rel=1e-12)
        # and the type follows the STRIKE - the out-of-the-money leg is the one a desk deals
        assert rung.contract.option_type == (
            'Call' if rung.contract.strike >= rung.forward else 'Put')


def test_a_dead_print_at_a_wing_moves_the_rung_to_the_next_listed_strike():
    """THE SCREEN AND THE SNAP ARE ONE MECHANISM. A crossed print at the strike the 25-delta band
    lands on neither enters the objective nor drops the rung: the contract was refused from
    CANDIDACY, so the argmin never saw it and the rung landed on the next listed strike."""
    clean = select_rungs(canned_chain(), FORWARD)[0]
    wing = next(rung for rung in clean if rung.kind == '25d put' and rung.pillar == 0.25)
    assert wing.contract.strike == 4750.0

    poisoned = dict(POISON)
    poisoned[(1, 0.95, 'Put')] = {'PX_BID': 200.0, 'PX_ASK': 150.0}  # 4750, crossed
    chain = canned_chain(poison=poisoned)
    assert chain.rejected[security_of(EXPIRIES[1], 4750.0, 'Put')] == 'crossed'
    moved = select_rungs(chain, FORWARD)[0]
    assert next(rung for rung in moved
                if rung.kind == '25d put' and rung.pillar == 0.25).contract.strike == 4875.0
    assert all(rung.contract.security != security_of(EXPIRIES[1], 4750.0, 'Put')
               for rung in moved)


def test_the_weights_are_positive_normalised_and_carry_their_liquidity():
    """WEIGHT = normalised vega x sqrt(open interest) / (1 + spread/cap), every factor read here.
    Vega makes the objective scale-free to three years; the square root is why one deep-liquid
    strike cannot own it; the spread factor runs 1 to 1/2, so nothing that survived is worth zero."""
    rungs, _, _ = select_rungs(canned_chain(), FORWARD)
    assert all(rung.weight > 0.0 for rung in rungs)
    assert sum(rung.weight for rung in rungs) == pytest.approx(1.0, rel=1e-12)
    assert all(rung.vega > 0.0 and 0.05 < rung.implied_vol < 1.0 for rung in rungs)

    # the formula, read off two contracts that differ in ONE factor at a time
    ladder = EquityLadder()
    base = contract(strike=5000.0, open_interest=10000.0)
    ten_times = contract(strike=5000.0, open_interest=100000.0)
    wide = contract(strike=5000.0, open_interest=10000.0, bid=95.0, ask=105.0)
    weights = {}
    for label, item in (('base', base), ('ten', ten_times), ('wide', wide)):
        weights[label] = equity_chain._rung(
            'ATM', 1.0, item, 1.0, 5100.0, RATE, ladder).weight
    assert weights['ten'] / weights['base'] == pytest.approx(math.sqrt(10.0), rel=1e-12)
    # base spreads 2/100 of mid, wide spreads 10/100, against a cap of 0.25
    assert weights['wide'] / weights['base'] == pytest.approx(
        (1.0 + 0.02 / 0.25) / (1.0 + 0.10 / 0.25), rel=1e-12)


def test_the_distinct_contract_floor_fires_naming_the_chains_own_expiries():
    """THIRTEEN RUNGS ARE NOT THIRTEEN QUOTES. DISTINCT contracts are counted after snapping, and a
    ladder below the component family's floor of eight refuses by name with the chain's listed
    expiries and the per-rung notes in the message.

    THE BAND IS ASSERTED HERE, because the floor cannot substitute for it: the three long pillars
    have an unclaimed 0.08y listing they COULD reach by argmin, and only `pillar_band` stops them.
    The distinct count is the same either way, so only the notes say which chain was read - and a
    2M fit wearing a 3Y label is what `assign_expiries` exists to forbid.
    """
    sparse = canned_rows(poison={}, expiries=EXPIRIES[:3], ratios=(0.95, 1.0, 1.05))
    chain = fetch_equity_chain(Walked(sparse), UNDERLYING, AS_OF)
    # the short listing IS believed and IS reachable by an unbanded argmin, or the gate below
    # measures an absent expiry rather than the band that refused it
    assert EXPIRIES[0] in chain.expiries

    rungs, notes, _ = select_rungs(chain, FORWARD)
    dropped = [note for note in notes if 'DROPPED' in note]
    assert [note.split(' ')[0] for note in dropped] == ['1y', '2y', '3y']
    for note in dropped:
        assert 'no listed expiry within 0.5 log-units of it' in note
    assert max(rung.pillar for rung in rungs) == 0.5
    assert {rung.contract.expiry for rung in rungs} == set(EXPIRIES[1:3])

    with pytest.raises(IncompleteChain) as refusal:
        equity_hn_block(chain, FORWARD)
    message = str(refusal.value)
    assert '2026-11-30' in message and '2027-02-28' in message
    assert 'distinct contract' in message and 'at least 8' in message
    assert 'HestonNandiComponentModelPrices' in message
    # the notes ride into the refusal, so it says what each rung DID
    assert '1y DROPPED' in message

    # the floor is a PARAMETER with a stated default, so a shorter ladder is taken deliberately
    _, block = equity_hn_block(chain, FORWARD, EquityLadder(
        pillars=(0.25, 0.5), wing_pillars=(0.25, 0.5), minimum_contracts=4))
    assert len({(row['Expiry_Date']['.Timestamp'], row['Strike'])
                for row in block['instrument']['European_Options']}) >= 4


def test_two_pillars_cannot_claim_one_listed_expiry():
    """ONE EXPIRY, ONE PILLAR - and the board that breaks a per-pillar argmin is an ORDINARY one:
    quarterlies to a year, a 1.4y LEAP, a 3y LEAP and no 2y LEAP, with both LEAPS inside
    `pillar_band` of the 2Y pillar. An argmin per pillar gives 2028-01-25 to the 1Y and the 2Y
    alike, so one contract enters `European_Options` twice, the two families read that differently,
    and the L strip comes out with FEWER knots than the ladder declares pillars.

    So pillars and listings are MATCHED and the pillar left with nothing is dropped BY NAME.
    NEAREST CLAIM WINS: the 3y listing stays the 3Y rung, where giving it to the 2Y pillar 0.41
    log-units away would put a name in the block the chain contradicts, in place of an honest gap.
    """
    board = (EXPIRIES[1], EXPIRIES[2], datetime.date(2028, 1, 25), EXPIRIES[5])
    chain = fetch_equity_chain(Walked(canned_rows(poison={}, expiries=board)), UNDERLYING, AS_OF)
    expiries = {expiry: (expiry - AS_OF).days / 365.0 for expiry in chain.expiries}
    ladder = EquityLadder()

    # the collision is REACHABLE on this board, or the assertion below is vacuous
    assert abs(math.log(expiries[board[2]] / 1.0)) < ladder.pillar_band
    assert abs(math.log(expiries[board[2]] / 2.0)) < ladder.pillar_band
    assert abs(math.log(expiries[board[3]] / 2.0)) < ladder.pillar_band

    assigned, dropped = equity_chain.assign_expiries(expiries, ladder)
    assert assigned == {0.25: board[0], 0.5: board[1], 1.0: board[2], 3.0: board[3]}
    assert set(dropped) == {2.0}
    assert '2028-01-25 is the 1y' in dropped[2.0] and '2029-08-31 is the 3y' in dropped[2.0]

    rungs, notes, _ = select_rungs(chain, FORWARD)
    assert [note for note in notes if 'DROPPED' in note] == [
        '2y DROPPED - {}'.format(dropped[2.0])]
    assert not [rung for rung in rungs if rung.pillar == 2.0]
    assert next(rung for rung in rungs if rung.pillar == 3.0).contract.expiry == board[3]

    _, block = equity_hn_block(chain, FORWARD)
    rows = block['instrument']['European_Options']
    keys = [(row['Expiry_Date']['.Timestamp'], row['Strike'], row['Option_Type']) for row in rows]
    assert len(keys) == len(set(keys)) == len(rungs) == 10
    assert sum(row['Weight'] for row in rows) == pytest.approx(1.0, rel=1e-12)
    # one knot per emitted expiry, one per SURVIVING pillar: a lost pillar costs a knot honestly,
    # a duplicated contract loses one silently
    assert len({key[0] for key in keys}) == 4


def test_two_rungs_on_one_contract_are_one_row_at_the_summed_weight():
    """A REPEATED CONTRACT IS A WEIGHT, NOT A SECOND EQUATION, and the emitter says so rather than
    leaving it to two families that disagree.

    What survives the injective expiry assignment is the collision WITHIN a pillar: on a coarse
    strike grid both wings snap back onto the ATM contract. Emitting it three times would put ONE
    print in as three equations at triple weight, which the component family drops one of and the
    plain family counts. So rows are collapsed onto distinct contracts with their weights summed,
    and the note names which rung was absorbed into which.
    """
    coarse = canned_rows(poison={}, expiries=EXPIRIES[1:3], ratios=(0.70, 1.0, 1.30))
    chain = fetch_equity_chain(Walked(coarse), UNDERLYING, AS_OF)
    ladder = EquityLadder(pillars=(0.25, 0.5), wing_pillars=(0.25, 0.5), minimum_contracts=2)

    rungs, _, _ = select_rungs(chain, FORWARD, ladder)
    rows, merged = equity_chain.collapse_rungs(rungs)
    assert len(rungs) == 6 and len(rows) == 2 and len(merged) == 4
    assert all('MERGED into the ATM' in note for note in merged)
    assert all(note.endswith('a repeated contract is a weight rather than a second equation')
               for note in merged)
    for rung, weight in rows:
        assert weight == pytest.approx(sum(
            item.weight for item in rungs
            if item.contract.security == rung.contract.security), rel=1e-12)
        assert weight > max(item.weight for item in rungs
                            if item.contract.security != rung.contract.security)
    assert sum(weight for _, weight in rows) == pytest.approx(1.0, rel=1e-12)

    _, block = equity_hn_block(chain, FORWARD, ladder)
    emitted_rows = block['instrument']['European_Options']
    keys = [(row['Expiry_Date']['.Timestamp'], row['Strike'], row['Option_Type'])
            for row in emitted_rows]
    assert len(keys) == len(set(keys)) == 2
    assert sum(row['Weight'] for row in emitted_rows) == pytest.approx(1.0, rel=1e-12)
    source = block['instrument']['Quote_Source']
    assert '6 rungs' in source and 'on 2 distinct contracts' in source and 'MERGED' in source


def test_the_floor_and_the_defaults_are_the_families_own_numbers():
    """The emitter cannot import the engine, so every number it hard-codes is held against the
    engine's own DECLARATION here. A default that moves on either side has to move on both."""
    from derivus.bootstrappers import (HestonNandiComponentModelParameters,
                                       HestonNandiModelParameters)

    assert EquityLadder().minimum_contracts == \
        HestonNandiComponentModelParameters.fx_minimum_contracts == 8
    declared = {field.name: field.default for field in HestonNandiModelParameters.fields}
    assert equity_chain.STEPS_PER_YEAR == declared['Steps_Per_Year']
    assert equity_chain.QUADRATURE_PANELS == declared['Quadrature_Panels']
    component = {field.name: field.default
                 for field in HestonNandiComponentModelParameters.fields}
    assert equity_chain.COMPONENT_HEADER == {
        'Rho': component['Rho'], 'Quote_Sensitivity': component['Quote_Sensitivity']}
    assert equity_chain.HN_REFERENCE_TYPES.keys() == \
        HestonNandiModelParameters.factor_types.keys()
    for field, spelled in equity_chain.HN_REFERENCE_TYPES.items():
        assert spelled in HestonNandiModelParameters.factor_types[field], field

    # the declared ladder: the product horizon, and the component family's widened wings
    assert EquityLadder().pillars == (0.25, 0.5, 1.0, 2.0, 3.0)
    assert len(EquityLadder().wing_pillars) == len(
        HestonNandiComponentModelParameters.fx_wing_expiries) == 4
    assert EquityLadder().wing_delta == HestonNandiModelParameters.fx_wing_pillar == 0.25


def test_a_ladder_that_contradicts_itself_refuses_at_construction():
    """A wing with no ATM rung beneath it has nothing to be a wing OF - the band is drawn under that
    expiry's own ATM implied vol - so it refuses where it is declared."""
    with pytest.raises(BloombergConfigurationError, match='not ATM pillars'):
        EquityLadder(pillars=(0.25, 0.5), wing_pillars=(0.25, 1.0))
    with pytest.raises(BloombergConfigurationError, match='wing_delta'):
        EquityLadder(wing_delta=0.75)
    with pytest.raises(BloombergConfigurationError, match='volatility_factor is blank'):
        EquityForward(underlying_factor='SPX', volatility_factor='', discount_rate='USD',
                      dividend_reference='SPX', rate=RATE)


# =============================================================================================
# 4  the forward, declared
# =============================================================================================

def test_the_forward_is_declared_and_the_chain_is_measured_against_it():
    """Every strike and weight hangs off the forward, and the FIT rebuilds that forward out of the
    two curves the block NAMES - so the emitter places its ladder on the same one or the
    calibration sits at coordinates the pricer never visits. The DECLARED carry builds the ladder;
    the chain's parity-implied dividend is measured beside it and REPORTED. Non-vacuous both ways:
    the chain implies the declared dividend back to 5e-5, and a caller who declares nothing gets
    the chain's own number."""
    chain = canned_chain()
    _, _, readings = select_rungs(chain, FORWARD)
    for pillar, reading in readings.items():
        assert reading['declared_dividend'] == DIVIDEND
        assert reading['implied_dividend'] == pytest.approx(DIVIDEND, abs=5e-5), pillar

    undeclared = EquityForward(
        underlying_factor='SPX', volatility_factor='SPX', discount_rate='USD',
        dividend_reference='SPX', rate=RATE)
    _, block = equity_hn_block(chain, undeclared)
    source = block['instrument']['Quote_Source']
    assert 'chain implies' in source and 'carried at r=4.0000% on USD against SPX' in source

    # and the block DECLARES both references, in the fields the family resolves them through
    instrument = block['instrument']
    assert (instrument['Discount_Rate'], instrument['Discount_Rate_Type']) == (
        'USD', 'InterestRate')
    assert (instrument['Yield'], instrument['Yield_Type']) == ('SPX', 'DividendRate')


UNDECLARED = EquityForward(underlying_factor='SPX', volatility_factor='SPX', discount_rate='USD',
                           dividend_reference='SPX', rate=RATE)


def test_the_parity_carry_is_a_median_and_one_fat_finger_does_not_move_it():
    """THE UNDECLARED CARRY IS THE ONE FOREIGN ANSWER THAT IS NOT A PER-CONTRACT QUOTE, and it is
    screened like one. Off a single pair it is the softest number here: a fat-fingered near-money
    call - 1899/1901, forty thousand open, dated today, and therefore BELIEVED by every screen -
    moves the 2Y parity carry from +1.5% to -9.7% and the forward from 5257 to 6580, taking every
    strike and weight on that pillar with it. Nothing per-contract catches it.

    A MEDIAN over the strikes nearest the forward does, and both halves are asserted: the bad
    prints ARE read, by name, and they are OUTVOTED. THREE strikes are poisoned, each closing a
    route a single pair could take - 5000 is nearest the SPOT, 5250 nearest the FORWARD, and 6500
    is where a one-pair read that believed 5000 runs away to. Two of five is the honest stress: a
    median needing four clean prints of five would be a mean with extra steps.
    """
    fat = dict(POISON)
    for ratio in (1.00, 1.05, 1.30):
        fat[(4, ratio, 'Call')] = {'PX_BID': 1899.0, 'PX_ASK': 1901.0, 'PX_LAST': 1900.0}
    chain = canned_chain(poison=fat)
    refused = {security_of(EXPIRIES[4], strike, 'Call') for strike in (5000.0, 5250.0, 6500.0)}
    assert not refused & set(chain.rejected), 'the fixture is not poison if the screen catches it'

    # what ONE pair implies at each anchor, off the believed mids - the numbers the median refuses
    tau = tau_of(EXPIRIES[4])
    mid_of = lambda strike, side: next(
        item.mid for item in chain.contracts
        if item.security == security_of(EXPIRIES[4], strike, side))
    for strike in (5000.0, 5250.0, 6500.0):
        one_pair = RATE - math.log(
            (strike + (mid_of(strike, 'Call') - mid_of(strike, 'Put'))
             * math.exp(RATE * tau)) / SPOT) / tau
        assert one_pair < -0.09, strike

    rungs, _, readings = select_rungs(chain, UNDECLARED)
    # the window is read AROUND THE FORWARD, where both legs of a pair are near the money and
    # quoted best; anchored on the spot its top strike would be 250 points in the money
    assert readings[2.0]['implied_strikes'] == (5000.0, 5125.0, 5250.0, 5375.0, 5500.0)
    assert {5000.0, 5250.0} <= set(readings[2.0]['implied_strikes']), 'the bad prints were not read'
    assert readings[2.0]['implied_dividend'] == pytest.approx(DIVIDEND, abs=5e-4)
    assert readings[2.0]['forward'] == pytest.approx(forward_of(EXPIRIES[4]), rel=1e-4)

    # the control is the same chain with the carry DECLARED - same contracts, same prints, and the
    # only difference is where the carry came from
    on_the_pillar = lambda items: [(item.kind, item.contract.security) for item in items
                                   if item.pillar == 2.0]
    assert on_the_pillar(rungs) == on_the_pillar(select_rungs(chain, FORWARD)[0])


def test_an_undeclared_carry_outside_the_band_refuses_by_name():
    """THE BAND IS WHAT A MEDIAN CANNOT DO: a median cannot save a neighbourhood that is wrong
    together. Every call at one expiry marked a thousand points rich implies a carry of -7.9% that
    parity reproduces at EVERY strike, so the estimator agrees with itself and is still wrong.

    Where the carry was DECLARED this is a reading: the ladder stands on the declared number and
    the disagreement is named in the record. Where nothing was declared the reading IS the forward,
    so it refuses with the pillar, the number, the band and the strikes it was read off.
    """
    expiry, fat = EXPIRIES[4], dict(POISON)
    for ratio in RATIOS:
        strike = round(SPOT * ratio, 2)
        mid = _tick(black_price(forward_of(expiry), strike, RATE, chain_vol(expiry, strike),
                                tau_of(expiry), True)) + 1000.0
        fat[(4, ratio, 'Call')] = {'PX_BID': mid - 2.0, 'PX_ASK': mid + 2.0, 'PX_LAST': mid}
    chain = canned_chain(poison=fat)

    with pytest.raises(IncompleteChain) as refusal:
        equity_hn_block(chain, UNDECLARED)
    message = str(refusal.value)
    assert 'the 2y pillar (2028-08-31)' in message
    assert 'outside the declared band' in message and 'EquityLadder.parity_band' in message
    assert '-7.8' in message and '6500' in message
    assert 'EquityForward.dividend_yield' in message

    # declared, the same chain emits - and says so
    _, block = equity_hn_block(chain, FORWARD)
    source = block['instrument']['Quote_Source']
    assert '2y declared 1.5000% / chain implies -7.8' in source
    assert 'OUTSIDE the declared band -5.0000%..15.0000%' in source
    # and the front pillars, whose expiries the feed did not touch, are not accused of anything
    assert '0.25y declared 1.5000% / chain implies 1.5' in source

    # the band is a PARAMETER with a stated default, so a desk that really carries that carry says so
    widened = EquityLadder(parity_band=(-0.20, 0.20))
    assert equity_hn_block(chain, UNDECLARED, widened)[1]['instrument']['European_Options']
    with pytest.raises(BloombergConfigurationError, match='parity_band'):
        EquityLadder(parity_band=(0.20, -0.20))


# =============================================================================================
# 5  the block
# =============================================================================================

def emitted(family=equity_chain.COMPONENT_FAMILY, **kwargs):
    return equity_hn_block(canned_chain(), FORWARD, EquityLadder(**kwargs), family=family)


def test_the_block_writes_only_fields_the_family_declares():
    """The emitter cannot import the family to check its own schema, so the gate does: every header
    key is a declared field, every `_Type` one of that field's candidates, `Premium` a declared
    `Quote_Type`, and the option row's nine columns are `OPTION_QUOTE`'s six plus the two-way and
    the stamp - read as an EQUALITY against `MARKET_QUOTE_VALUES` rather than as a gap."""
    from derivus import schema
    from derivus.bootstrappers import (HestonNandiComponentModelParameters,
                                       HestonNandiModelParameters)

    for family, klass in ((equity_chain.COMPONENT_FAMILY, HestonNandiComponentModelParameters),
                          (equity_chain.PLAIN_FAMILY, HestonNandiModelParameters)):
        name, block = emitted(family=family)
        assert name == '{}.SPX'.format(family)
        declared = {field.name: field for field in klass.fields}
        instrument = block['instrument']
        assert set(instrument) <= set(declared), sorted(set(instrument) - set(declared))
        for key, value in instrument.items():
            if declared[key].values:
                assert value in declared[key].values, (family, key, value)
        assert instrument['Quote_Type'] == 'Premium'
        assert 'Premium' in declared['Quote_Type'].values

        rows = instrument['European_Options']
        columns = {field.name for field in declared['European_Options'].row.fields}
        assert columns == {'Expiry_Date', 'Strike', 'Option_Type', 'Units', 'Weight',
                           'Quoted_Market_Value', 'Quoted_Bid', 'Quoted_Ask', 'Timestamp'}
        for row in rows:
            assert columns <= set(row), 'a row is missing a column the family declares'
            # the three-way identity: the keys the row carries beside `OPTION_QUOTE`'s six, the
            # keys the emitter declares, and the house's value plane are ONE SET - which is what
            # puts `European_Options` in `MARKET_QUOTE_CONTAINERS`. A column added on any side has
            # to appear on the other two.
            assert set(row) - {field.name for field in schema.OPTION_QUOTE} == \
                set(equity_chain.QUOTE_VALUE_KEYS) == \
                set(schema.MARKET_QUOTE_VALUES) - {'Quoted_Market_Value'}
        assert 'European_Options' in schema.MARKET_QUOTE_CONTAINERS, (
            'the option row declares the value keys and the value plane does not know it')


def test_one_selection_writes_both_family_spellings():
    """ONE SELECTION, TWO NAMES: the same option table row for row and byte for byte, differing
    only in the header each family declares - the component one states its pinned `Rho` and its
    refused `Quote_Sensitivity`, and the plain family declares neither."""
    component_name, component = emitted(family=equity_chain.COMPONENT_FAMILY)
    plain_name, plain = emitted(family=equity_chain.PLAIN_FAMILY)

    assert (component_name, plain_name) == ('HestonNandiComponentModelPrices.SPX',
                                            'HestonNandiModelPrices.SPX')
    assert component['instrument']['European_Options'] == plain['instrument']['European_Options']
    assert json.dumps(component['instrument']['European_Options'], sort_keys=True) == \
        json.dumps(plain['instrument']['European_Options'], sort_keys=True)
    difference = set(component['instrument']) - set(plain['instrument'])
    assert difference == set(equity_chain.COMPONENT_HEADER)
    assert {key: component['instrument'][key] for key in difference} == \
        equity_chain.COMPONENT_HEADER
    assert {key: value for key, value in component['instrument'].items()
            if key not in difference} == plain['instrument']

    with pytest.raises(BloombergConfigurationError, match='not a Heston-Nandi quote family'):
        equity_hn_block(canned_chain(), FORWARD, family='FXVolPrices')


def test_the_two_way_is_carried_and_the_crossed_print_never_reaches_it():
    """The mid is what the fit prices against and the two-way is what a desk dealt, so both travel
    with the print's own clock. No row of the block is crossed, one-sided, dead or American -
    none of those were ever candidates."""
    chain = canned_chain()
    _, block = emitted()
    believed = {item.security: item for item in chain.contracts}
    for row in block['instrument']['European_Options']:
        assert row['Quoted_Bid'] < row['Quoted_Market_Value'] < row['Quoted_Ask']
        assert row['Quoted_Market_Value'] == pytest.approx(
            0.5 * (row['Quoted_Bid'] + row['Quoted_Ask']), rel=1e-12)
        assert row['Timestamp'] == {'.Timestamp': '2026-08-31'}
        assert row['Units'] == 1.0 and row['Weight'] > 0.0
    assert sum(row['Weight'] for row in block['instrument']['European_Options']) == \
        pytest.approx(1.0, rel=1e-12)

    # every emitted contract is one the screen believed, by name
    for rung in select_rungs(chain, FORWARD)[0]:
        assert rung.contract.security in believed


def test_the_same_canned_chain_emits_the_same_bytes():
    """DETERMINISM: the only clock in sight is the chain's own as-of, a parameter and not a wall
    clock, so two emissions off the same canned answers are byte-identical."""
    first = json.dumps(emitted()[1], sort_keys=True)
    second = json.dumps(emitted()[1], sort_keys=True)
    assert first == second
    # a moved market moves the bytes, or the claim above is vacuous - and the contract moved is one
    # the ladder EMITS, which is what "the market moved" means for a block of listed premiums
    bumped = dict(POISON)
    bumped[(3, 1.025, 'Put')] = {'PX_BID': 371.0, 'PX_ASK': 375.0, 'PX_LAST': 373.0}
    moved = equity_hn_block(canned_chain(poison=bumped), FORWARD)[1]
    assert json.dumps(moved, sort_keys=True) != first
    assert [row['Quoted_Bid'] for row in moved['instrument']['European_Options']].count(371.0) == 1

    # a print the ladder does NOT emit leaves the block alone, which is the selection saying what
    # it is a function of - it moved the block while a single parity pair placed the forward
    ignored = dict(POISON)
    ignored[(3, 1.00, 'Call')] = {'PX_BID': 401.0, 'PX_ASK': 405.0, 'PX_LAST': 403.0}
    assert json.dumps(equity_hn_block(canned_chain(poison=ignored), FORWARD)[1],
                      sort_keys=True) == first


# =============================================================================================
# 6  the engine seam - read-only
# =============================================================================================

#: The flat surface the world carries when it carries one. A `Premium` block reads no surface, so
#: the end-to-end gate builds its world WITHOUT this - and a block naming a surface the book does
#: carry still has to resolve, which is the other half of the same statement.
SURFACE = {'EquityPriceVol.SPX': {
    'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
    'Surface': {'.Curve': {'meta': [], 'data': [
        [0.8, 0.1, 0.20], [0.8, 3.5, 0.20], [1.0, 0.1, 0.18], [1.0, 3.5, 0.18],
        [1.2, 0.1, 0.16], [1.2, 3.5, 0.16]]}}}}


def job_document(market_prices=None, factors=None, surface=True, repo=None):
    """A real wire-form job document over the world this chain is quoted around: spot 5000, a 4% USD
    curve, a 1.5% dividend rate, and a flat surface unless `surface` says otherwise. `repo` names a
    SECOND curve as the equity's own `Interest_Rate` - the one `utils.calc_eq_forward` integrates -
    so a world with a genuine funding spread is one argument away. `factors` is merged last.
    """
    curve = lambda value: {'.Curve': {'meta': [], 'data': [[0.0, value], [5.0, value]]}}
    price_factors = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': curve(RATE)},
        'EquityPrice.SPX': {'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0,
                            'Currency': 'USD', 'Interest_Rate': repo or 'USD', 'Spot': SPOT},
        'DividendRate.SPX': {'Currency': 'USD', 'Curve': curve(DIVIDEND)}}
    if surface:
        price_factors.update(SURFACE)
    price_factors.update(factors or {})
    return {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': {'.Timestamp': AS_OF.isoformat()},
                        'Currency': 'USD', 'MCMC_Simulations': 1, 'Random_Seed': 1},
        'Deals': {'Tag_Titles': '', 'Reference': 'chain', 'Deals': {'Children': []}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD',
                                  'Base_Date': {'.Timestamp': AS_OF.isoformat()}},
            'Price Factors': price_factors,
            'Bootstrapper Configuration': {},
            'Market Prices': market_prices or {}}}}}


def value_tick(block, moves):
    """The block a TICK SOURCE posts: the same rows, `moves` merged at each index, nothing else.

    A RE-EMISSION IS NOT A TICK, by the emitter's own arithmetic: every `Weight` is a normalised
    vega x liquidity, so one moved print re-weighs the whole block, and `Weight` is structure. What
    the value plane admits is a source that moves quotes and leaves the weights - the shape
    `Context.market_patch` publishes.
    """
    rows = [dict(row) for row in block['instrument']['European_Options']]
    for index, fields in moves.items():
        rows[index].update(fields)
    return {'instrument': dict(block['instrument'], European_Options=rows)}


def test_the_block_installs_and_updates_through_the_engines_own_guard():
    """`config.update_market_quote` is the contract every quote source posts against: 'installed'
    the first time, 'updated' the second, and 'updated' on a VALUE-ONLY re-tick - which used to
    refuse with 'structure differs', because the guard's value plane was `Points` rows while these
    families quote in `European_Options`. A moved STRIKE still refuses, which makes that a line
    rather than a waiver, and so does a re-EMITTED chain, because `Weight` is a normalised vega and
    one moved print re-weighs the ladder. Both readings asserted.
    """
    from derivus.config import update_market_quote

    name, block = emitted()
    document = job_document()
    assert update_market_quote(document, name, block) == 'installed'
    prices = document['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices']
    assert prices[name] is block

    again = emitted()[1]
    assert again == block and again is not block
    assert update_market_quote(document, name, again) == 'updated'

    # the value-only re-tick: one wing's mid, two-way and stamp move, and nothing else
    row = block['instrument']['European_Options'][-1]
    ticked = value_tick(block, {len(block['instrument']['European_Options']) - 1: {
        'Quoted_Market_Value': row['Quoted_Market_Value'] + 4.0,
        'Quoted_Bid': row['Quoted_Bid'] + 4.0, 'Quoted_Ask': row['Quoted_Ask'] + 4.0,
        'Timestamp': {'.Timestamp': '2026-09-01'}}})
    assert ticked != block, 'the tick moved nothing - this gate is vacuous'
    assert update_market_quote(document, name, ticked) == 'updated'

    # a MOVED CONTRACT on the same block is still a re-authoring
    moved = value_tick(block, {0: {'Strike': block['instrument'][
        'European_Options'][0]['Strike'] + 25.0}})
    with pytest.raises(ValueError, match='structure differs'):
        update_market_quote(document, name, moved)

    # the emitter's own re-emission: the contracts stand, the WEIGHTS do not
    bumped = dict(POISON)
    bumped[(4, 1.15, 'Call')] = {'PX_BID': 611.0, 'PX_ASK': 619.0, 'PX_LAST': 615.0}
    reticked = equity_hn_block(canned_chain(poison=bumped), FORWARD)[1]
    node = lambda item: [(row['Expiry_Date'], row['Strike'], row['Option_Type'])
                         for row in item['instrument']['European_Options']]
    assert node(reticked) == node(block), 'the re-emission moved a contract, not a value'
    assert [row['Weight'] for row in reticked['instrument']['European_Options']] != \
        [row['Weight'] for row in block['instrument']['European_Options']], (
        'a moved print left the normalised ladder alone - the weight claim below is vacuous')
    with pytest.raises(ValueError, match='structure differs'):
        update_market_quote(document, name, reticked)

    with pytest.raises(ValueError, match='a Market Prices block is'):
        update_market_quote(document, name, block['instrument'])


def test_a_premium_tick_moves_the_values_hash_and_leaves_the_chains_plan_bit_identical():
    """THE PARTITION, on a chain block loaded through the real decoder. A premium, its two-way and
    its stamp move `values_hash` and leave `plan_hash` HEX-IDENTICAL, so a re-quoted chain is a
    tick rather than a recompile; a moved strike or weight refuses as structural. `market_patch`
    publishes the ladder's value rows under its own table name and `patch_market` takes them back.
    """
    import derivus
    from derivus.config import CustomJsonEncoder

    name, block = emitted()
    context = derivus.Context().load_json(
        (json.dumps(job_document({name: block}), cls=CustomJsonEncoder), 'chain-tick'))
    prices = context.current_cfg.params['Market Prices']
    rows = prices[name]['instrument']['European_Options']
    assert len(rows) > 1, 'the ladder arrived with no quotes - nothing below means anything'

    # the values half is keyed by the family's OWN table, one entry per row
    patch = context.market_patch()[name]
    assert set(patch) == {'European_Options'} and len(patch['European_Options']) == len(rows)
    assert all(set(row) == {'Quoted_Market_Value', 'Quoted_Bid', 'Quoted_Ask', 'Timestamp'}
               for row in patch['European_Options'])

    before = (context.plan_hash(), context.values_hash())
    context.patch_market(context.market_patch())
    assert (context.plan_hash(), context.values_hash()) == before, (
        'the values-plane identity moved a hash')

    context.patch_market({name: {'European_Options': [
        {'Quoted_Market_Value': rows[0]['Quoted_Market_Value'] + 1.5}] + [{} for _ in rows[1:]]}})
    assert context.plan_hash() == before[0], 'a moved premium moved the chain block\'s plan'
    assert context.values_hash() != before[1], 'a moved premium left the values hash alone'

    # the other direction: a moved strike is a re-authoring and says so
    with pytest.raises(ValueError, match='Strike is structural, not a value'):
        context.patch_market({name: {'European_Options': [
            {'Strike': 4000.0}] + [{} for _ in rows[1:]]}})
    with pytest.raises(ValueError, match='Weight is structural, not a value'):
        context.patch_market({name: {'European_Options': [
            {'Weight': 0.5}] + [{} for _ in rows[1:]]}})


#: A SHORT ladder and a low evaluation cap: the reading is about the FIT'S ARITHMETIC over a chain
#: block, not about the ladder. The component fit re-bootstraps the whole L strip per outer iterate,
#: so a five-pillar ladder to three years is 756 daily steps a price and hours of wall clock -
#: which is why the floor is taken down DELIBERATELY here and nowhere else.
E2E_LADDER = EquityLadder(pillars=(0.25, 0.5), wing_pillars=(0.25, 0.5), minimum_contracts=4)


def test_the_component_family_fits_the_chain_block_with_no_authored_surface(caplog):
    """END TO END WITH NO SURFACE IN THE BOOK AT ALL - the circularity gone.

    A `Premium` block reads no surface: `quote_type_references` says `Implied_Volatility` reads one
    and `Premium` does not, so the `Volatility` NAME the block carries is never resolved. The
    reference used to be declared REQUIRED and resolved before the branch that never reads it, and
    a book without a surface got `Unable to bootstrap ... - skipping` - no factor, no exception,
    and a chain-sourced fit that could not run unless someone had already fitted the same chain.

    What is measured: the block resolves the references its quote type actually reads, the L
    bootstrap brackets its pillars against LISTED premiums, and the family writes a positive `H0`,
    `Beta` and `Gamma_1` (the equity leverage sign) with one L knot per ATM pillar plus the anchor
    at tenor zero. The ATM residual and the wall clock are RECORDED.
    """
    import logging as _logging
    import time

    import derivus
    from derivus.config import CustomJsonEncoder

    name, block = equity_hn_block(canned_chain(), FORWARD, E2E_LADDER)
    rows = block['instrument']['European_Options']
    assert len({(row['Expiry_Date']['.Timestamp'], row['Strike']) for row in rows}) >= 4

    block['instrument']['Max_Iterations'] = 8
    document = job_document({name: block}, surface=False)
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    assert not [factor for factor in market['Price Factors'] if factor.startswith('EquityPriceVol')]
    assert block['instrument']['Volatility'] == 'SPX', (
        'the block does not name a surface, so nothing is being said about naming one it lacks')
    market['Bootstrapper Configuration'] = {'HestonNandiComponentModelParameters': {}}

    config = derivus.Context().load_json(
        (json.dumps(document, cls=CustomJsonEncoder), 'equity_chain')).current_cfg
    started = time.time()
    with caplog.at_level(_logging.INFO):
        config.bootstrap()
    elapsed = time.time() - started

    written = config.params['Price Factors'].get('HestonNandiComponentModelParameters.SPX')
    assert written is not None, (
        'the component family wrote no factor off a premium-quoted chain block in a surface-free '
        'book - the circularity is still standing')
    assert not [record for record in caplog.records if 'skipping' in record.getMessage()], (
        'a reference was skipped rather than read or refused')
    for key in ('Alpha', 'Beta', 'Gamma_1', 'Rho', 'Phi', 'Gamma_2', 'H0'):
        assert key in written and math.isfinite(float(written[key])), key
    assert written['Rho'] == pytest.approx(equity_chain.COMPONENT_HEADER['Rho'])
    assert written['H0'] > 0.0 and written['Beta'] > 0.0
    # a positive Gamma_1 is the equity leverage sign
    assert written['Gamma_1'] > 0.0, 'a falling index smile fitted with the FX leverage sign'
    # the L curve: a knot at tenor zero anchoring q0 = L(0), then one per ATM pillar
    from derivus import utils
    curve = written[utils.HN_COMPONENT_CURVE_NAME]
    assert len(curve.array) == 1 + len(E2E_LADDER.pillars) and curve.array[0][0] == 0.0
    assert all(level > 0.0 for _, level in curve.array), curve.array

    # the reading, recorded: the bootstrap's own ATM residual off its report, and the wall clock
    reported = [record.getMessage() for record in caplog.records if 'ATM residual' in
                record.getMessage()]
    assert reported, 'the family fitted and reported nothing about it'
    print('\nfitted off the chain block, no surface in the book: {}\nL (annualised vol): {}\n{}\n'
          'bootstrap wall clock {:.1f}s'.format(
              {key: float(written[key]) for key in utils.HN_COMPONENT_PARAM_NAMES},
              [(float(knot), round(float(math.sqrt(level * 252.0)), 4))
               for knot, level in curve.array],
              reported[-1].strip(), elapsed))


def test_a_surface_the_book_does_carry_is_still_read_where_the_quote_type_reads_one():
    """The other half, so "optional" does not read as "ignored": under `Quote_Type` Premium the
    surface is INERT, so the same block fits to the SAME parameters with and without one - which
    says the surface stopped being an input rather than merely stopped being fetched.
    """
    import derivus
    from derivus.config import CustomJsonEncoder
    from derivus import utils

    def fitted(surface):
        name, block = equity_hn_block(canned_chain(), FORWARD, E2E_LADDER)
        block['instrument']['Max_Iterations'] = 8
        document = job_document({name: block}, surface=surface)
        document['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Bootstrapper Configuration'] = {'HestonNandiComponentModelParameters': {}}
        config = derivus.Context().load_json(
            (json.dumps(document, cls=CustomJsonEncoder), 'equity_chain')).current_cfg
        config.bootstrap()
        written = config.params['Price Factors']['HestonNandiComponentModelParameters.SPX']
        return [float(written[key]) for key in utils.HN_COMPONENT_PARAM_NAMES]

    with_surface, without = fitted(True), fitted(False)
    assert with_surface == without, (
        'a Premium fit read the surface it declares it does not: {} against {}'.format(
            with_surface, without))


# =============================================================================================
# 7  a missing reference refuses; the skip is dead
# =============================================================================================

def bootstrapped(block, family='HestonNandiComponentModelPrices', **world):
    """`(config, refusal or None)` - one block through the REAL bootstrap, the config returned
    either way so a gate can ask what was written."""
    import derivus
    from derivus.config import CustomJsonEncoder

    name = '{}.SPX'.format(family)
    document = job_document({name: block}, **world)
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Bootstrapper Configuration'] = {
        family.replace('Prices', 'Parameters'): {}}
    config = derivus.Context().load_json(
        (json.dumps(document, cls=CustomJsonEncoder), 'refusal')).current_cfg
    try:
        config.bootstrap()
    except Exception as refusal:
        return config, refusal
    return config, None


def blanked(field, quote_type='Premium'):
    """The canned chain's block with one reference BLANKED - the shape a block takes when its author
    leaves a panel empty, which is the case that used to skip."""
    _, block = equity_hn_block(canned_chain(), FORWARD, E2E_LADDER)
    block['instrument'][field] = ''
    block['instrument']['Quote_Type'] = quote_type
    block['instrument']['Max_Iterations'] = 2
    return block


@pytest.mark.parametrize('quote_type,field,required', [
    ('Premium', 'Underlying', 'Underlying/Discount_Rate'),
    ('Premium', 'Discount_Rate', 'Underlying/Discount_Rate'),
    ('Implied_Volatility', 'Volatility', 'Underlying/Volatility/Discount_Rate'),
    ('Implied_Volatility', 'Underlying', 'Underlying/Volatility/Discount_Rate')])
def test_a_missing_required_reference_refuses_by_name_and_never_skips(
        quote_type, field, required, caplog):
    """THE SKIP IS DEAD, in three parts: the exception REACHES THE CALLER, NO factor is written,
    and no 'skipping' line lands anywhere. What stood here caught every resolution failure, logged
    `Unable to bootstrap ... - skipping`, and moved on, so a job whose block named nothing the book
    carried completed and told its caller nothing.

    Both quote types ENUMERATE what they require in the message, because the remedy differs: a
    missing `Volatility` under `Implied_Volatility` is fixed by naming a surface OR by quoting
    premiums, and the refusal says both.
    """
    import logging as _logging

    with caplog.at_level(_logging.ERROR):
        config, refusal = bootstrapped(blanked(field, quote_type), surface=True)

    assert refusal is not None, 'a blank {} under {} still skipped'.format(field, quote_type)
    message = str(refusal)
    assert 'HestonNandiComponentModelPrices.SPX' in message, 'the refusal does not name the block'
    assert field in message and required in message, message
    assert quote_type in message, 'the refusal does not name the quote type doing the requiring'
    if field == 'Volatility':
        assert 'Quote_Type Premium' in message, 'the second remedy is not offered'
    assert not [factor for factor in config.params['Price Factors']
                if factor.startswith('HestonNandiComponentModelParameters.')], (
        'a refused block wrote a price factor anyway')
    assert not [record for record in caplog.records if 'skipping' in record.getMessage()], (
        'the refusal still logged a skip')


def test_a_reference_the_book_does_not_carry_refuses_naming_the_factor_it_looked_for():
    """The field is NAMED and the book has nothing under that name. The refusal spells out the
    factor it looked for, because the block's name and the `Price Factors` key differ by a TYPE -
    which is what `<field>_Type` is for. A `Quote_Type` this family does not fit refuses first."""
    _, block = equity_hn_block(canned_chain(), FORWARD, E2E_LADDER)
    block['instrument']['Discount_Rate'] = 'ZAR'
    block['instrument']['Max_Iterations'] = 2

    config, refusal = bootstrapped(block, surface=False)
    assert refusal is not None, 'a named-but-absent curve still skipped'
    assert 'InterestRate.ZAR' in str(refusal) and 'Discount_Rate' in str(refusal), str(refusal)
    assert not [factor for factor in config.params['Price Factors']
                if factor.startswith('HestonNandiComponentModelParameters.')]

    # and it refuses before anything is resolved at all
    block['instrument'].update({'Discount_Rate': 'USD', 'Quote_Type': 'Mid'})
    _, refused = bootstrapped(block, surface=False)
    assert refused is not None and 'Quote_Type' in str(refused)
    assert 'Implied_Volatility' in str(refused) and 'Premium' in str(refused)


# =============================================================================================
# 8  the forward the fit builds is the forward the pricer builds
# =============================================================================================

#: THE REPO CURVE and the spread that makes this gate non-vacuous: 125bp, so a forward built on the
#: wrong curve misses by 191 index points at the 3y rung.
REPO = 'USD-REPO'
REPO_SPREAD = 0.0125
#: A curve at ZERO, so the probe deal's discount factor is exactly 1.0 and its MtM IS the forward.
ZERO_CURVE = 'USD-ZERO'
#: The pillars the identity is measured at, in whole days off the base date.
FORWARD_DAYS = (91, 182, 365, 730, 1095)


def flat(value):
    return {'.Curve': {'meta': [], 'data': [[0.0, value], [5.0, value]]}}


def repo_world(market_prices=None, deals=()):
    """The index world with a GENUINE REPO SPREAD - the equity funds on `USD-REPO` at 5.25% while
    its deals discount on `USD` at 4% - plus a curve at zero for the probe deal."""
    document = job_document(market_prices, surface=False, repo=REPO, factors={
        'InterestRate.{}'.format(REPO): {'Currency': 'USD', 'Day_Count': 'ACT_365',
                                         'Sub_Type': None, 'Curve': flat(RATE + REPO_SPREAD)},
        'InterestRate.{}'.format(ZERO_CURVE): {'Currency': 'USD', 'Day_Count': 'ACT_365',
                                               'Sub_Type': None, 'Curve': flat(0.0)}})
    document['Calc']['Deals']['Deals']['Children'] = [
        {'Instrument': {'.Deal': deal}} for deal in deals]
    return document


def probe_block(funding=REPO, days=FORWARD_DAYS):
    """A `HestonNandiModelPrices.SPX` block whose every quote leaves `Strike` at ZERO - the declared
    meaning of which is `0 reads the forward`, and the family fills the row in in place. So after a
    real bootstrap each row's `Strike` IS the fit's own forward, read out of the engine.

    `Steps_Per_Year` is 12 because a 3y rung at 252 is 756 sequential recursions per price. The
    premiums are plausible ATM prints; the fit's quality is not what is measured.
    """
    expiries = [AS_OF + datetime.timedelta(days=day) for day in days]
    quotes = []
    for expiry, day in zip(expiries, days):
        tau = day / 365.0
        level = SPOT * math.exp((RATE + REPO_SPREAD - DIVIDEND) * tau)
        quotes.append({'Expiry_Date': {'.Timestamp': expiry.isoformat()}, 'Strike': 0.0,
                       'Option_Type': 'Call', 'Units': 1.0, 'Weight': 1.0 / len(days),
                       'Quoted_Market_Value': black_price(level, level, RATE, 0.20, tau, True)})
    instrument = {'Underlying': 'SPX', 'Underlying_Type': 'EquityPrice',
                  'Volatility': 'SPX', 'Volatility_Type': 'EquityPriceVol',
                  'Discount_Rate': 'USD', 'Discount_Rate_Type': 'InterestRate',
                  'Yield': 'SPX', 'Yield_Type': 'DividendRate',
                  'Quote_Type': 'Premium', 'Use_Forward': 'No', 'Invert_Moneyness': 'No',
                  'Steps_Per_Year': 12.0, 'Quadrature_Panels': 16,
                  'Quote_Timestamp': '', 'Quote_Source': 'the forward-identity probe',
                  'European_Options': quotes}
    if funding:
        instrument.update({'Funding_Rate': funding, 'Funding_Rate_Type': 'InterestRate'})
    return expiries, {'instrument': instrument}


def fitted_forwards(block, deals=()):
    """`(forwards, the five fitted parameters, the run output)` - one real bootstrap of `block` in
    the repo world, with `deals` priced against the same market in the same job."""
    import derivus
    from derivus import run_baseval, utils
    from derivus.config import CustomJsonEncoder

    name = 'HestonNandiModelPrices.SPX'
    document = repo_world({name: block}, deals)
    document['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Bootstrapper Configuration'] = {'HestonNandiModelParameters': {}}
    config = derivus.Context().load_json(
        (json.dumps(document, cls=CustomJsonEncoder), 'forward-identity')).current_cfg
    config.bootstrap()
    rows = config.params['Market Prices'][name]['instrument']['European_Options']
    written = config.params['Price Factors']['HestonNandiModelParameters.SPX']
    out = run_baseval(config)[1] if deals else None
    return ([row['Strike'] for row in rows],
            [float(written[key]) for key in utils.HN_PARAM_NAMES], out)


def probe_deals(expiries):
    """One `EquityForwardDeal` per expiry, struck at ZERO on the zero curve, so its base-date MtM is
    `utils.calc_eq_forward` itself. THE PRICER'S FORWARD AND NOT A REPLICA: the deal reads
    `EquityPrice.SPX`'s own `Interest_Rate` against `DividendRate.SPX` through the compiled factor
    path, the same call every equity option makes."""
    return [{'Object': 'EquityForwardDeal', 'Reference': 'FWD{}'.format(index), 'Tags': '',
             'MtM': '', 'Forward_Price': 0.0, 'Buy_Sell': 'Buy', 'Payoff_Type': 'Standard',
             'Equity_Volatility': '', 'Maturity_Date': {'.Timestamp': expiry.isoformat()},
             'Equity': 'SPX', 'Units': 1.0, 'Currency': 'USD',
             'Discount_Rate': ZERO_CURVE, 'Payoff_Currency': 'USD'}
            for index, expiry in enumerate(expiries)]


def test_the_calibrated_forward_is_the_priced_forward_at_every_pillar():
    """One `Discount_Rate` used to FUND the calibrated forward and DISCOUNT the premium, while
    `utils.calc_eq_forward` grows the priced forward on the equity's own repo curve - so on an
    index with a borrow spread the two parted and the fit sat where the pricer never visits.

    The block declares `Funding_Rate` now, and both forwards are read out of the ENGINE: the fit's
    off the `Strike` column it filled in, the pricer's off an `EquityForwardDeal`'s MtM in the same
    job. They agree to 1e-13. Non-vacuous by construction: with no `Funding_Rate` the same probe is
    the old arithmetic, off by exactly the spread's own carry (over 0.3%) at every pillar.
    """
    expiries, block = probe_block()
    deals = probe_deals(expiries)
    calibrated, _, out = fitted_forwards(block, deals)
    frame = out['Results']['mtm']
    priced = [float(frame[frame['Reference'] == deal['Reference']]['Value'].iloc[0])
              for deal in deals]

    worst = max(abs(fit / price - 1.0) for fit, price in zip(calibrated, priced))
    assert worst < 1e-13, (
        'the calibrated forward is not the priced one: {} against {} (worst {:.3e})'.format(
            calibrated, priced, worst))

    # the world is one where it MATTERS: with no funding curve the fit falls back to the discount
    # curve, which is the old arithmetic
    _, unfunded = probe_block(funding=None)
    fallback = fitted_forwards(unfunded)[0]
    assert fallback != calibrated
    for fit, price, day in zip(fallback, priced, FORWARD_DAYS):
        assert fit == pytest.approx(price * math.exp(-REPO_SPREAD * day / 365.0), rel=1e-13), (
            'the fallback is not spot exp((r-q)t) on the discount curve')
    assert max(abs(fit / price - 1.0) for fit, price in zip(fallback, priced)) > 0.003, (
        'the repo spread moves nothing on this world - the gate above is vacuous')
    print('\nforward identity at {} pillars, {:g}bp repo spread: worst relative miss {:.3e}; '
          'undeclared funding misses by up to {:.2%}'.format(
              len(FORWARD_DAYS), REPO_SPREAD * 10000, worst,
              max(abs(fit / price - 1.0) for fit, price in zip(fallback, priced))))


def test_no_funding_curve_is_the_arithmetic_this_family_always_had():
    """THE BIT-IDENTITY BAR, stated structurally rather than against a golden file. The funding
    basis rides in as a shift on `q`: `effective_yield` adds `r - f`, and with no `Funding_Rate`
    the term is not evaluated at all, so every expression downstream is the same bits. Where a
    funding curve names the discount curve, `r - f` is exactly 0.0.

    Both halves in HEX rather than to a tolerance, at the function and through a real fit.
    """
    from derivus.bootstrappers import HestonNandiModelParameters as HN

    class Flat:
        def __init__(self, level):
            self.level = level

        def current_value(self, t):
            return self.level * (1.0 + t)

    carry, discount = Flat(0.0173), Flat(0.041)
    for t in (0.25, 1.0, 3.0):
        rate = float(discount.current_value(t))
        assert HN.effective_yield(rate, None, carry, t).hex() == \
            float(carry.current_value(t)).hex(), 'an undeclared funding curve moved the carry'
        assert HN.effective_yield(rate, discount, carry, t).hex() == \
            float(carry.current_value(t)).hex(), 'funding ON the discount curve moved the carry'

    # the same statement through a REAL fit, over the probe world, both ways
    _, unfunded = probe_block(funding=None)
    _, self_funded = probe_block(funding='USD')
    plain_fwd, plain_fit, _ = fitted_forwards(unfunded)
    funded_fwd, funded_fit, _ = fitted_forwards(self_funded)
    hexed = lambda values: [value.hex() for value in values]
    assert hexed(plain_fwd) == hexed(funded_fwd), (
        'declaring the discount curve as the funding curve moved the forward')
    assert hexed(plain_fit) == hexed(funded_fit), (
        'declaring the discount curve as the funding curve moved the fitted parameters')


def test_the_chain_emitter_declares_the_funding_curve_it_placed_its_strikes_with():
    """The emitter's half: `EquityForward` names the funding curve, the block carries it in the
    field the family resolves, and `Quote_Source` says which curve did which job. Blank, the block
    writes no `Funding_Rate` at all rather than a declaration nobody made."""
    from derivus.bootstrappers import HestonNandiModelParameters

    spread = EquityForward(
        underlying_factor='SPX', volatility_factor='SPX', discount_rate='USD',
        dividend_reference='SPX', rate=RATE + REPO_SPREAD, dividend_yield=DIVIDEND,
        funding_rate=REPO)
    instrument = equity_hn_block(canned_chain(), spread, E2E_LADDER)[1]['instrument']

    assert (instrument['Funding_Rate'], instrument['Funding_Rate_Type']) == (REPO, 'InterestRate')
    assert instrument['Discount_Rate'] == 'USD'
    assert 'Funding_Rate' in {field.name for field in HestonNandiModelParameters.fields}
    assert instrument['Funding_Rate_Type'] in \
        HestonNandiModelParameters.factor_types['Funding_Rate']
    source = instrument['Quote_Source']
    assert 'carried at r=5.2500% on {} against SPX'.format(REPO) in source
    assert 'premiums discounting on USD' in source

    # blank, and the block says nothing rather than something wrong
    plain = equity_hn_block(canned_chain(), FORWARD, E2E_LADDER)[1]['instrument']
    assert 'Funding_Rate' not in plain and 'Funding_Rate_Type' not in plain
    assert 'discounting on' not in plain['Quote_Source']


def test_a_live_terminal_answers_or_the_smoke_skips_by_name():
    """LIVE SMOKE; a workstation with no terminal SKIPS by name. What is asserted is the ROUTE and
    not the market: the bulk chain read answers, its members batch through the scalar reader, and
    every one comes back as a contract or a NAMED refusal. The census is PRINTED and never
    asserted - a chain read after the close screens to almost nothing on `one-sided` alone.

    Slow because an index chain is thousands of contracts at ten fields each, fifty at a time;
    `on_batch` prints progress so a quiet ten minutes is legible.
    """
    import time

    from derivus_bloomberg.errors import BloombergFXError
    from derivus_bloomberg.session import blpapi_module

    try:
        blpapi_module()
    except BloombergFXError as absent:
        pytest.skip('no Bloomberg SDK on this workstation: {}'.format(absent))
    seen, started = [], time.time()
    try:
        with BloombergSession(timeout_ms=60000, connect_timeout_ms=5000) as session:
            chain = fetch_equity_chain(
                session, 'SPX Index', datetime.date.today(),
                on_batch=lambda done, total: seen.append((done, total)))
    except BloombergFXError as refused:
        pytest.skip('no Bloomberg terminal answering on this workstation: {}'.format(refused))

    census = {}
    for verdict in chain.rejected.values():
        census[verdict] = census.get(verdict, 0) + 1
    print('\nSPX Index ({}), spot {}: {} listed, {} believed, {} refused ({}) in {:.0f}s'.format(
        chain.name, chain.spot, len(chain.contracts) + len(chain.rejected),
        len(chain.contracts), len(chain.rejected),
        ', '.join('{} {}'.format(count, verdict) for verdict, count in sorted(census.items()))
        or 'nothing refused', time.time() - started))

    assert seen and seen[-1][0] == seen[-1][1], 'the member probe did not run to its own total'
    # every member the terminal named is accounted for one way or the other - the only thing about
    # a live chain that does not depend on the hour. The ledger can hold MORE than the probe asked
    # about: a chain row nobody could read a ticker out of is refused before it becomes a member
    assert len(chain.contracts) + len(chain.rejected) >= seen[-1][1] > 0
