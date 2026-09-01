"""The listed equity option CHAIN as a Heston-Nandi quote block - `derivus_bloomberg.equity_chain`.

Nothing here opens a socket except the last gate, which skips by name where this workstation has no
terminal. Everything else runs on ONE canned chain - 192 listed contracts over six expiries, mixed
liquidity, both exercise styles, and a poison table of dead prints authored on purpose - driven
through the package's real readers (`BloombergSession`'s own event walk, canned rows) and the real
engine seam (`config.update_market_quote`, both Heston-Nandi bootstraps, `Context`'s two hashes and
a base valuation over an `EquityForwardDeal`) - the engine is imported and never touched.

WHAT IS HELD:

  the budget       `equity_chain` imports the standard library, this package's own modules and a
                   LAZY blpapi - read off the source, and proved a second way in a fresh
                   interpreter, which also shows blpapi is not imported by importing the module
  the screen       the order of distrust, one contract per verdict, and the census of the whole
                   canned chain by name - a candidate silently dropped is indistinguishable from
                   one never asked about
  the policy       applied CLIENT-SIDE, `fxvol._value_of`'s lesson: a contract the tolerant reader
                   flagged for a FIELD exception (no VOLUME, because it has not traded today) is
                   still read, and only a row with nothing in it is `invalid`
  American         a chain that screened to nothing but American exercise refuses BY NAME, with
                   the underlying and the remedy, because an American premium is not the European
                   premium the fit prices against
  the selection    the documented contracts, exactly: the ATM rung is the listed strike nearest
                   its own forward - and nearest it in the metric the ENGINE re-derives the
                   ATM/wing split with - the wings are the 25-delta-equivalent moneyness bands
                   under the chain's own ATM implied vol, and a dead print AT a wing moves the
                   rung to the next listed strike rather than into the objective
  the floor        eight DISTINCT contracts, which is the component family's own number, and the
                   refusal names the chain's own expiries
  the block        every key it writes is a field the family DECLARES, the `_Type` values are the
                   family's own candidate lists, and the three value keys the option row carries
                   beside its mid are exactly `MARKET_QUOTE_VALUES` less the mid - now read as an
                   equality the family's own row satisfies rather than as a gap
  the round trip   `update_market_quote` installs it, updates it, takes a VALUE-ONLY re-tick, and
                   refuses a moved strike; the tick moves `values_hash` with `plan_hash`
                   hex-identical and `market_patch`/`patch_market` round-trip it
  the weights      positive, normalised, and the stated formula - two contracts alike but for
                   their open interest weigh in the ratio of the SQUARE ROOTS
  determinism      the same canned chain emits the same bytes
  end to end       the component family's own bootstrap, off a real JSON document, in a book with
                   NO authored surface - the circularity gone, on a short ladder and a low
                   evaluation cap
  the refusals     a reference a quote type requires and the block does not name refuses BY NAME,
                   writes no factor, and lands no 'skipping' line
  the forward      the fit's own forward, read off the `Strike` column it fills in, against
                   `calc_eq_forward`'s through an `EquityForwardDeal` in the same job - on an index
                   with a genuine repo spread, and hex-identical to the old arithmetic without one

NO MONKEYPATCHING. The canned terminal is a `BloombergSession` subclass whose event walk yields
rows, which is `test_bloomberg_discover.Walked` verbatim; the engine is imported and never touched.
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

#: Six listed expiries. Five sit on the ratified pillars (3M/6M/1Y/2Y/3Y to the day count's own
#: rounding) and the front one does not sit on any of them - a listed expiry the ladder never asks
#: for is part of what a chain looks like, and a selection that quietly used it would be wrong.
EXPIRIES = (datetime.date(2026, 9, 30), datetime.date(2026, 11, 30), datetime.date(2027, 2, 28),
            datetime.date(2027, 8, 31), datetime.date(2028, 8, 31), datetime.date(2029, 8, 31))

#: Sixteen listed strikes per expiry per side, finer near the money and coarse in the wings, which
#: is what an index chain actually looks like. 16 x 2 x 6 = 192 contracts.
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
    """The vol the fixture's prices are generated FROM - a rising ATM term structure with a steep,
    one-signed, slightly convex skew. That is the index shape the roadmap names (the positive
    `Gamma_Star` box's home market), and a rising term structure is the case the component family's
    declining-variance guard does not refuse - the humped one is `test_hn_component`'s gate, not
    this one's."""
    moneyness = math.log(strike / forward_of(expiry))
    return 0.16 + 0.02 * math.sqrt(tau_of(expiry)) - 0.35 * moneyness + 0.6 * moneyness ** 2


def security_of(expiry, strike, option_type):
    return 'SPX {} {}{:g} Index'.format(expiry.strftime('%m/%d/%y'),
                                        'C' if option_type == 'Call' else 'P', strike)


def _tick(value):
    """A listed price sits on a tick, and a fixture that quotes eleven decimals is not a print."""
    return round(value * 20.0) / 20.0


def _row(expiry, strike, option_type):
    """One clean listed contract: a Black price off the fixture's own vol, ticked, with a two-way
    that tightens toward the money and an open interest that thins away from it."""
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


#: THE POISON TABLE, authored on purpose, one entry per verdict the screen can reach. Every one of
#: these is a shape a real chain actually carries: a wing quoted one-sided into the close, a stale
#: side left standing against a live one, a listed strike nobody holds, a benchmark that has not
#: printed in six weeks, a single-name-style American listing, and a deep in-the-money call marked
#: below its own intrinsic. Keyed by `(expiry index, ratio, side)`.
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
    """A session whose event walks are canned rows - `test_bloomberg_discover.Walked`, with the
    BULK walk canned beside the scalar one. Driving the real session means the tolerance, the
    per-name filling and the batching are all gated through the code a terminal would run."""

    def __init__(self, rows, chain=None, underlying=None, errors=None):
        super().__init__()
        self._api = self._session = self._service = object()  # started, as far as the guard cares
        self.rows = rows
        #: `{security: Bloomberg's own text}` for names the terminal flagged while STILL answering
        #: their fields - a fieldException, which is what a contract that has not traded today
        #: raises on VOLUME. The tolerant reader reports those as `ok: False`.
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
    """The top-level names a file imports, however deep in it the import sits. Relative imports
    are skipped: they resolve inside the package by construction and carry no module name to
    judge - `test_spine_imports.imported_names`, which is the pattern this extends."""
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
    """A NEW gate rather than an edit to the old one, and a STRICTER budget than the package's own:
    `fxvol` and `types` carry pandas, and this module deliberately does not - so `equity_chain` is
    held to the standard library, this package's own modules, and a blpapi that is imported LAZILY
    or not at all. An import that never executes is still a dependency, so this reads the SOURCE.
    """
    imported = imported_names(os.path.join(ROOT, 'derivus_bloomberg', 'equity_chain.py'))
    # the package's own modules are reached RELATIVELY (`from .errors import ...`), which
    # `imported_names` skips by construction; `derivus_bloomberg` is allowed so an absolute
    # intra-package import would pass too, and `derivus` is not, which is the rule that matters
    assert imported <= {'collections', 'datetime', 'math', 'statistics', 'dataclasses', 'typing',
                        'derivus_bloomberg'}, sorted(imported)
    # blpapi is on this list on purpose: it must be imported LAZILY (`session.blpapi_module`) or
    # not at all, because the package has to import on a workstation that has never seen a terminal
    assert imported.isdisjoint({'derivus', 'torch', 'pandas', 'numpy', 'scipy', 'blpapi'}), \
        sorted(imported)
    # non-vacuous: a module that imported nothing at all would pass both assertions above
    assert imported


def in_a_fresh_interpreter(statements):
    """What `sys.modules` holds after a fresh interpreter runs `statements` - the source gate's
    answer measured rather than parsed."""
    code = ('import json, sys; {}; '
            'print(json.dumps(sorted({{name.split(".")[0] for name in sys.modules}})))'.format(
                statements))
    done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    assert done.returncode == 0, done.stderr
    return set(json.loads(done.stdout))


def test_importing_the_chain_emitter_lands_no_engine_no_blpapi_and_no_pandas():
    """The source gate's answer proved a second way, because the first one trusts the parser AND
    because the parser reads ONE FILE while an import runs a package.

    blpapi is the claim that matters as much as the engine - the whole package rests on being
    importable on a workstation that has no terminal, and `session.blpapi_module` is what makes that
    true. PANDAS is the claim this file's own docstring makes and the one a source gate CANNOT hold:
    importing a submodule imports its package first, so an eager `from .fxvol import ...` in
    `derivus_bloomberg/__init__.py` would land numpy and pandas behind this module's back with every
    line of `equity_chain.py` still innocent. The package re-exports the pandas-carrying names
    lazily; this is what says so.
    """
    landed = in_a_fresh_interpreter('import derivus_bloomberg.equity_chain')
    assert 'derivus_bloomberg' in landed, 'the module did not import'
    assert landed.isdisjoint({'derivus', 'torch', 'blpapi', 'pandas', 'numpy'}), sorted(
        landed & {'derivus', 'torch', 'blpapi', 'pandas', 'numpy'})

    # the re-export still WORKS, and the emitter is reachable off the package by name
    assert 'derivus_bloomberg' in in_a_fresh_interpreter(
        'from derivus_bloomberg import equity_hn_block, EquityLadder')
    # non-vacuous, and the cost is where it belongs: asking the package for an FX name is what
    # pays for pandas, so the gate above is measuring a deferral rather than an absence
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
    as the verdicts are: `american` and `wide` are different instructions to a desk, and a screen
    that checked the spread first would report the second where the first is true."""
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
    # the ORDER, asserted where it bites: an American contract that is ALSO crossed and dead reads
    # as american, because that is the finding a desk can act on
    both = contract(security='both', exercise='American', bid=110.0, ask=90.0,
                    open_interest=0.0, last_update='2007-03-26')
    assert screen_chain([both], AS_OF)[1] == {'both': 'american'}

    # a print date that is PRESENT and unreadable evidences a print's time exactly as well as a
    # blank one does - and it would otherwise ride into the block as that row's own `Timestamp`
    # and refuse at the engine's decoder instead of here
    assert screen_chain([contract(security='na', last_update='N/A')], AS_OF)[1] == {'na': 'undated'}

    # the model-free bounds, which need no curve: a call is never worth more than the underlying
    # and a put never more than its strike. Both need the spot, and without one the screen simply
    # does not make the claim
    rich = contract(security='rich', bid=5100.0, ask=5200.0)
    assert screen_chain([rich], AS_OF, spot=SPOT)[1] == {'rich': 'off-market'}
    assert screen_chain([rich], AS_OF)[1] == {}


def test_the_canned_chain_is_believed_by_census():
    """The whole 192-contract fixture through the real reader, counted by verdict and named where
    it matters. The ledger is the point: a candidate silently dropped is indistinguishable from one
    never asked about, and on a chain this size that difference IS the report."""
    chain = canned_chain()
    assert len(chain.contracts) + len(chain.rejected) == len(RATIOS) * 2 * len(EXPIRIES) == 192
    census = {}
    for verdict in chain.rejected.values():
        census[verdict] = census.get(verdict, 0) + 1
    assert census == {'american': 4, 'crossed': 2, 'expired': 1, 'malformed': 1,
                      'no-open-interest': 3, 'off-market': 1, 'one-sided': 3,
                      'stale': 2, 'undated': 1, 'unpriced': 3, 'unstated-exercise': 2, 'wide': 6}
    assert len(chain.contracts) == 192 - sum(census.values()) == 163
    # SIX OF THE NINE WIDE-OR-UNPRICED ARE NOT IN THE POISON TABLE - they are far wings whose
    # minimum tick is a third of their own mid, or whose price rounds to nothing at all. That is
    # not fixture noise; it is the thing the roadmap's liquidity ruling exists for. Half a chain
    # is dead strikes even when nobody authored one.
    natural = {security for security, verdict in chain.rejected.items()
               if verdict in ('wide', 'unpriced')}
    assert len(natural) == 9 and sum('09/30/26' in security for security in natural) == 5

    # named, one per family of refusal, so a re-tuned screen has to come here and say so
    assert chain.rejected[security_of(EXPIRIES[5], 4250.0, 'Put')] == 'american'
    assert chain.rejected[security_of(EXPIRIES[2], 6000.0, 'Call')] == 'unstated-exercise'
    assert chain.rejected[security_of(EXPIRIES[1], 4000.0, 'Put')] == 'crossed'
    assert chain.rejected[security_of(EXPIRIES[3], 3750.0, 'Put')] == 'stale'
    assert chain.rejected[security_of(EXPIRIES[2], 5250.0, 'Call')] == 'undated'
    assert chain.rejected[security_of(EXPIRIES[1], 3500.0, 'Call')] == 'off-market'
    assert chain.spot == SPOT and chain.name == 'S&P 500 INDEX'
    assert chain.expiries == EXPIRIES


def test_the_fetch_asks_the_bulk_reader_for_the_chain_and_batches_its_members():
    """TWO ROUND TRIPS AND NO SPELLED TICKERS. The membership comes off the BULK reader, because
    the scalar one answers row zero of an array and says nothing about the two thousand it dropped;
    the members are then asked in `discover.BATCH`-sized chunks off the tolerant scalar reader, so
    one refused ticker in a batch of fifty is the finding rather than the failure."""
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
    """THE POLICY IS APPLIED CLIENT-SIDE - `fxvol._value_of`'s lesson, paid for a second time and
    measured on a live chain.

    The tolerant reader answers `ok: False` on ANY per-security trouble, and a mere fieldException
    is trouble: a listed contract that has not traded today carries no `VOLUME`, and reading `ok`
    would throw away a contract that answered its strike, its expiry, its two-way, its open
    interest and its print date. MEASURED against a real SPX chain, which is where this was found:
    reading `ok` refused 1,855 of 8,000 contracts. So a row with ANY field in it is READ and the
    SCREEN judges it; only a row with nothing in it is `invalid`.
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
    """THE ANCHOR IS SCREENED LIKE EVERYTHING ELSE, which is the SAONIA rule read to the end. A
    blank spot is the loud failure and it already refused; the quiet one is a spot that answers a
    plausible number nineteen years after it last printed - and every listed contract in this chain
    is screened on exactly that field while the number all of them are placed against was not.

    Both halves are here: a spot older than `stale_days`, and a spot whose print date is PRESENT
    and unreadable, which evidences a print's time exactly as well as a blank one does. And the
    date a believed spot passed on travels into the block, so a chain that was believed says how old
    its anchor is instead of leaving a reader to assume it was today's.
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
    """THE ROADMAP'S V1 RULING, raised where it bites. An American premium is not the European
    premium a Heston-Nandi fit prices against - the early-exercise right is worth something the
    closed form does not carry - so a single-name chain refuses rather than fitting the wrong
    number under the right name. The refusal names the underlying, the count, the style and the
    remedy; saying "eight distinct contracts" about it would name the symptom."""
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
    """THE NEAR MISS, which is where a refusal gated on an EMPTY chain named the wrong cause. One
    European contract among 191 American listings is not a mixed board that calibrates thinly - it
    is the same chain the gate above refuses, one print away from being empty - and firing the floor
    there produced "the ladder collapses onto 1 distinct contract ... quote the chain at more
    expiries and strikes" about a chain listing six expiries and sixteen strikes. The symptom named,
    the wrong remedy prescribed, and neither the word `american` nor the word `exercise` anywhere in
    it.

    So the refusal reads the CENSUS: the chain cannot reach the floor at all, and more candidates
    were refused on exercise style than survived the whole screen. The per-contract screen is
    untouched and stays right - a mixed board drops its American listings individually and the
    European ones still calibrate, which the second half asserts so this cannot be "fixed" into
    refusing every chain with a flex listing on it.
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
    # what it must NOT say: the floor's message, which asks a desk to go and quote more of a chain
    # that is already quoted at six expiries and sixteen strikes
    assert 'distinct contract' not in message and 'more expiries and strikes' not in message

    # and the mixed board still calibrates: the standard fixture refuses six contracts on exercise
    # style, per contract, and emits its ladder off what is left
    mixed = canned_chain()
    assert sum(1 for verdict in mixed.rejected.values()
               if verdict in ('american', 'unstated-exercise')) == 6
    assert len(equity_hn_block(mixed, FORWARD)[1]['instrument']['European_Options']) == 13


# =============================================================================================
# 3  the selection
# =============================================================================================

def test_the_selection_picks_the_documented_contracts():
    """THE LADDER, contract by contract. Five ATM rungs on the ratified pillars plus 25-delta
    wings at the first four - thirteen rungs, and each one is a LISTED contract rather than a
    coordinate: the ATM rung is the strike nearest its own forward (a PUT at 5000 against a
    forward of 5031, because the type follows the strike and the out-of-the-money leg is the one a
    desk deals), and each wing is the listed strike nearest the moneyness band drawn under that
    expiry's OWN at-the-money implied vol."""
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

    # THE ATM RUNG IS THE ONE THE ENGINE WILL CALL ATM. `HestonNandiComponentModelParameters`
    # re-derives the split off the block by taking, per expiry, the row nearest its own forward -
    # "the emitter's ordering is not a marker anything may depend on" - so the property that has to
    # hold is agreement in the ENGINE's own metric, over the rows actually emitted.
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
    """THE SCREEN AND THE SNAP ARE ONE MECHANISM, and this is where that pays. A crossed print at
    the strike the 25-delta band lands on does not enter the objective and does not drop the rung
    either: the contract was refused from CANDIDACY, so the argmin never saw it and the rung landed
    on the next listed strike. That is the difference between a screen and a filter applied
    afterwards."""
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
    """WEIGHT = normalised vega x sqrt(open interest) / (1 + spread/cap), and every factor of it is
    read here. Vega is what makes the objective scale-free over a term structure that now runs to
    three years; the square root is why one deep-liquid strike cannot own the objective; and the
    spread factor runs from 1 at a locked market to 1/2 at the cap, so nothing that survived the
    screen is worth zero."""
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
    """THIRTEEN RUNGS ARE NOT THIRTEEN QUOTES - the FX rule, transferred with its arithmetic. A
    chain listing three strikes at three short expiries answers only the front of the ladder, and
    what it does answer lands on too few distinct contracts to identify anything: the DISTINCT
    contracts are counted after snapping and a ladder below the component family's own floor of
    eight refuses by name, with the chain's listed expiries in the message.

    AND THE BAND IS ASSERTED HERE, because the floor cannot substitute for it. The three long
    pillars have a listing they COULD reach by argmin - 2026-09-30, unclaimed, 0.08y - and what
    stops the 1Y, 2Y and 3Y rungs landing on a five-week contract is `pillar_band` alone. The
    distinct count is the same either way, so the floor's refusal is identical with the band and
    without it; only the notes say which chain was read. A 2M fit wearing a 3Y label is the thing
    `assign_expiries` spends a paragraph forbidding, and this is where the forbidding is measured.
    """
    sparse = canned_rows(poison={}, expiries=EXPIRIES[:3], ratios=(0.95, 1.0, 1.05))
    chain = fetch_equity_chain(Walked(sparse), UNDERLYING, AS_OF)
    # the short listing IS believed and IS reachable by an unbanded argmin - without this the gate
    # below would be measuring an absent expiry rather than the band that refused it
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
    # the notes ride into the refusal, so the message says what each rung DID and not only that
    # there were too few of them
    assert '1y DROPPED' in message

    # the floor is a PARAMETER with a stated default, so a reading that wants a shorter ladder can
    # take one deliberately - and the same chain then emits
    _, block = equity_hn_block(chain, FORWARD, EquityLadder(
        pillars=(0.25, 0.5), wing_pillars=(0.25, 0.5), minimum_contracts=4))
    assert len({(row['Expiry_Date']['.Timestamp'], row['Strike'])
                for row in block['instrument']['European_Options']}) >= 4


def test_two_pillars_cannot_claim_one_listed_expiry():
    """ONE EXPIRY, ONE PILLAR - and the board that breaks a per-pillar argmin is an ORDINARY one.

    Quarterlies out to a year, a 1.4y LEAP, a 3y LEAP and NO 2y LEAP is a listing pattern, not a
    pathology, and both LEAPS sit inside `pillar_band` of the 2Y pillar. An argmin per pillar gives
    2028-01-25 to the 1Y and the 2Y alike: the same listed contract enters `European_Options` TWICE,
    which the two family spellings then read differently (the component bootstrap drops a duplicate
    ATM by strike and discards its weight, the plain family applies every row), and the L strip
    comes out with FEWER knots than the ladder declares pillars - the one thing the component family
    is here for.

    So the pillars and the listings are MATCHED and the pillar left with nothing is dropped BY NAME.
    NEAREST CLAIM WINS is asserted too: the 3y listing stays the 3Y rung. Handing it to the 2Y
    pillar 0.41 log-units away - which is what "the shorter pillar has first claim" would do - would
    put a name into the block that the chain contradicts, in place of an honest gap.
    """
    board = (EXPIRIES[1], EXPIRIES[2], datetime.date(2028, 1, 25), EXPIRIES[5])
    chain = fetch_equity_chain(Walked(canned_rows(poison={}, expiries=board)), UNDERLYING, AS_OF)
    expiries = {expiry: (expiry - AS_OF).days / 365.0 for expiry in chain.expiries}
    ladder = EquityLadder()

    # the collision is REACHABLE on this board, which is what makes the assertion below non-vacuous
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
    # one knot per emitted expiry is what the L strip will carry, and it is one per SURVIVING
    # pillar - a lost pillar costs a knot honestly, a duplicated contract loses one silently
    assert len({key[0] for key in keys}) == 4


def test_two_rungs_on_one_contract_are_one_row_at_the_summed_weight():
    """A REPEATED CONTRACT IS A WEIGHT, NOT A SECOND EQUATION, and the emitter says so rather than
    leaving it to two families that disagree about it.

    What survives the injective expiry assignment is the collision WITHIN a pillar: on a coarse
    strike grid the 25-delta band falls inside the listing's own spacing and both wings snap back
    onto the ATM contract. Emitting that contract three times would put ONE print into the objective
    as three equations at triple weight - and the component family would silently drop one of them
    while the plain family counted all three. So the rows are collapsed onto distinct contracts with
    their weights summed, which is what the objective would have done with them anyway, and the note
    names which rung was absorbed into which so the ladder is still auditable.
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
    """The emitter cannot import the engine, so every number it hard-codes is a WHITELIST held
    against the engine's own DECLARATION here - the `fxvol._structure` pattern. A default that
    moves on either side now has to move on both."""
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

    # the ratified ladder: the product horizon, and the component family's widened wings
    assert EquityLadder().pillars == (0.25, 0.5, 1.0, 2.0, 3.0)
    assert len(EquityLadder().wing_pillars) == len(
        HestonNandiComponentModelParameters.fx_wing_expiries) == 4
    assert EquityLadder().wing_delta == HestonNandiModelParameters.fx_wing_pillar == 0.25


def test_a_ladder_that_contradicts_itself_refuses_at_construction():
    """A wing with no ATM rung beneath it has nothing to be a wing OF - the band is drawn under
    that expiry's own at-the-money implied vol - so it refuses where it is declared rather than
    producing a ladder nobody asked for."""
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
    """THE GENUINELY NEW WORK BESIDE FX. The strikes hang off the forward and so does every weight,
    and the FIT rebuilds that forward out of the two curves the block NAMES - so the emitter has to
    place its ladder on the same one or the calibration sits at coordinates the pricer never
    visits. The declared carry is what the ladder is built on; the chain's own parity-implied
    dividend is measured beside it and REPORTED, so a disagreement is a fact a desk is told rather
    than one it inherits.

    Non-vacuous both ways: on this fixture the chain implies the declared dividend back to eight
    figures (put-call parity is exact on prices generated from one forward), and a caller who
    declares nothing gets the chain's own number and the ladder that follows from it."""
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
    screened like one.

    Read off a single pair it is the softest number in the module: a fat-fingered near-money call -
    1899/1901, forty thousand open, dated today, and therefore BELIEVED by every screen there is -
    moves the 2Y parity carry from +1.5% to -9.7%, the forward from 5257 to 6580, and takes every
    strike and every weight on that pillar with it. Nothing per-contract can catch it, because
    nothing about that print is wrong on its own terms.

    A MEDIAN over the strikes nearest the forward is what catches it, and the gate asserts both
    halves: the bad prints ARE read (they are two of the five strikes, by name) and they are
    OUTVOTED, so the pillar's forward and its whole ladder are the clean chain's.

    THREE STRIKES ARE POISONED, and each one closes a route a SINGLE pair could take. 5000 is the
    strike nearest the SPOT, which is where a one-pair read starts; 5250 is the strike nearest the
    FORWARD, which is where it looks second; and 6500 is the strike a one-pair read RUNS AWAY TO -
    believing 5000 puts the forward at 6580, and 6500 is what is nearest that. So a single pair is
    wrong whichever anchor it chooses, while five strikes and a vote never see more than two bad
    ones at once. Two of five is also the honest stress: a median that needed four clean prints out
    of five would be a mean with extra steps.
    """
    fat = dict(POISON)
    for ratio in (1.00, 1.05, 1.30):
        fat[(4, ratio, 'Call')] = {'PX_BID': 1899.0, 'PX_ASK': 1901.0, 'PX_LAST': 1900.0}
    chain = canned_chain(poison=fat)
    refused = {security_of(EXPIRIES[4], strike, 'Call') for strike in (5000.0, 5250.0, 6500.0)}
    assert not refused & set(chain.rejected), 'the fixture is not poison if the screen catches it'

    # what ONE pair implies at each of the three anchors, computed here off the believed mids - the
    # numbers the median refuses, and the -9.7% that moved this pillar's forward to 6580 before it
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
    # the window is read AROUND THE FORWARD (5257) and not around the spot, which is where both
    # legs of a pair are near the money and quoted best - anchored on the spot it would have been
    # (4750, 4875, 5000, 5125, 5250), whose top strike is 250 points in the money on one side
    assert readings[2.0]['implied_strikes'] == (5000.0, 5125.0, 5250.0, 5375.0, 5500.0)
    assert {5000.0, 5250.0} <= set(readings[2.0]['implied_strikes']), 'the bad prints were not read'
    assert readings[2.0]['implied_dividend'] == pytest.approx(DIVIDEND, abs=5e-4)
    assert readings[2.0]['forward'] == pytest.approx(forward_of(EXPIRIES[4]), rel=1e-4)

    # THE CONTROL IS THE SAME CHAIN WITH THE CARRY DECLARED - same contracts, same prints, and the
    # only difference is where the carry came from. A ladder read off the chain that matches the
    # ladder placed on a declared 1.5% is the claim stated exactly.
    on_the_pillar = lambda items: [(item.kind, item.contract.security) for item in items
                                   if item.pillar == 2.0]
    assert on_the_pillar(rungs) == on_the_pillar(select_rungs(chain, FORWARD)[0])


def test_an_undeclared_carry_outside_the_band_refuses_by_name():
    """THE BAND IS WHAT A MEDIAN CANNOT DO, because a median cannot save a neighbourhood that is
    wrong together. Every call at one expiry marked a thousand points rich is a bad feed rather than
    a market, and it implies a carry of -7.9% that parity reproduces at EVERY strike - so the
    estimator agrees with itself and is still wrong.

    Where the carry was DECLARED this is a reading and stays one: the ladder was placed on the
    declared number, the chain's opinion is reported beside it, and the disagreement is named in the
    record rather than averaged away. Where nothing was declared the reading IS the forward, so it
    refuses with the pillar, the number, the band and the strikes it was read off - "the chain is
    wrong" without a coordinate is not something a desk can act on.
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

    # the band is a PARAMETER with a stated default, so a desk that really carries that carry can
    # say so - and then the same chain places its ladder on the chain's own number
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
    """The block is THIS family's schema in THIS family's conventions, and the emitter cannot
    import the family to check - so the gate does. Every header key is a declared field, every
    `_Type` value is one of that field's own candidates, `Premium` is a declared `Quote_Type`, and
    the option row's NINE declared columns are `OPTION_QUOTE`'s six plus the two-way and the stamp -
    which is the third finding CLOSED, read as an equality where it used to be read as a gap."""
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
            # THE THREE-WAY IDENTITY, unchanged in shape and now read the other way round: the keys
            # the row carries beside `OPTION_QUOTE`'s six, the keys the emitter DECLARES it carries,
            # and the value plane the rest of the house agrees on are ONE SET - and the family's own
            # row declares them, which is what puts `European_Options` in
            # `schema.MARKET_QUOTE_CONTAINERS`. A column added on any of the three sides now has to
            # appear on the other two or this fails.
            assert set(row) - {field.name for field in schema.OPTION_QUOTE} == \
                set(equity_chain.QUOTE_VALUE_KEYS) == \
                set(schema.MARKET_QUOTE_VALUES) - {'Quoted_Market_Value'}
        assert 'European_Options' in schema.MARKET_QUOTE_CONTAINERS, (
            'the option row declares the value keys and the value plane does not know it')


def test_one_selection_writes_both_family_spellings():
    """ONE SELECTION, TWO NAMES. The two blocks carry the SAME option table, row for row and byte
    for byte; what differs is only the header each family declares - the component one states its
    pinned `Rho` (a pin nobody can see is not a declared pin) and its refused `Quote_Sensitivity`,
    and the plain family declares neither field."""
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
    """The mid is what the fit prices against and the two-way is what a desk dealt - so both
    travel, with the print's own clock beside them. And the screen's work shows up HERE: no row of
    the block is crossed, one-sided, dead or American - none of those were ever candidates."""
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
    """DETERMINISM, and the only clock in sight is the chain's own as-of - which is a parameter,
    not a wall clock. Two emissions off the same canned answers are byte-identical, so a block that
    changed is a market that moved."""
    first = json.dumps(emitted()[1], sort_keys=True)
    second = json.dumps(emitted()[1], sort_keys=True)
    assert first == second
    # and a moved market moves the bytes, or the claim above would be vacuous. The contract moved
    # is one the ladder EMITS - the 1y ATM rung - because that is what "the market moved" means for
    # a block of listed premiums: the mid, the two-way and every weight normalised beside it
    bumped = dict(POISON)
    bumped[(3, 1.025, 'Put')] = {'PX_BID': 371.0, 'PX_ASK': 375.0, 'PX_LAST': 373.0}
    moved = equity_hn_block(canned_chain(poison=bumped), FORWARD)[1]
    assert json.dumps(moved, sort_keys=True) != first
    assert [row['Quoted_Bid'] for row in moved['instrument']['European_Options']].count(371.0) == 1

    # a print the ladder does NOT emit leaves the block alone, which is the selection saying what
    # it is a function of: one near-money call bumped 30 points off-market moved the block only
    # while a single parity pair placed the pillar's forward, and it no longer does
    ignored = dict(POISON)
    ignored[(3, 1.00, 'Call')] = {'PX_BID': 401.0, 'PX_ASK': 405.0, 'PX_LAST': 403.0}
    assert json.dumps(equity_hn_block(canned_chain(poison=ignored), FORWARD)[1],
                      sort_keys=True) == first


# =============================================================================================
# 6  the engine seam - read-only
# =============================================================================================

#: The FLAT SURFACE the world carries when it carries one at all. A `Premium` block reads no
#: surface, so the end-to-end gate builds its world WITHOUT this - but a block that names a surface
#: the book does carry has to keep resolving, which is the other half of the same statement.
SURFACE = {'EquityPriceVol.SPX': {
    'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
    'Surface': {'.Curve': {'meta': [], 'data': [
        [0.8, 0.1, 0.20], [0.8, 3.5, 0.20], [1.0, 0.1, 0.18], [1.0, 3.5, 0.18],
        [1.2, 0.1, 0.16], [1.2, 3.5, 0.16]]}}}}


def job_document(market_prices=None, factors=None, surface=True, repo=None):
    """A real wire-form job document over the equity world this chain is quoted around: spot 5000,
    a 4% USD curve, a 1.5% dividend rate, and a flat surface unless `surface` says otherwise.
    Authored here rather than borrowed, because the fixture chain is priced off exactly these
    numbers.

    `repo` names a SECOND curve as the equity's own `Interest_Rate` - the repo curve
    `utils.calc_eq_forward` integrates - so a world with a genuine funding spread is one argument
    away from the one every other gate runs on. `factors` is merged last, for a world that needs a
    deal or a curve nothing else does.
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
    """The block a TICK SOURCE posts: the same rows, with `moves` merged onto the row at each index
    and nothing else touched.

    A RE-EMISSION IS NOT A TICK, and that is the emitter's own arithmetic rather than a gap: every
    `Weight` is a vega times a liquidity NORMALISED over the ladder, so one moved print re-weighs
    every row of the block. `Weight` is structure - the fit's objective is posed with it - so an
    emitted block always re-authors. What the value plane admits is a source that moves quotes and
    leaves the weights, which is exactly the shape `Context.market_patch` publishes.
    """
    rows = [dict(row) for row in block['instrument']['European_Options']]
    for index, fields in moves.items():
        rows[index].update(fields)
    return {'instrument': dict(block['instrument'], European_Options=rows)}


def test_the_block_installs_and_updates_through_the_engines_own_guard():
    """`config.update_market_quote` is the contract every quote source posts against, and this
    block passes it BOTH ways: 'installed' the first time, 'updated' when the same chain is emitted
    again.

    AND THE SECOND HALF IS THE ROW THAT CLOSED. A moved premium used to refuse with 'structure
    differs', because the guard's value plane was `Points` rows and the Heston-Nandi families quote
    in `European_Options`; the option row declares the four value keys now, `European_Options` is a
    `schema.MARKET_QUOTE_CONTAINERS` table, and a VALUE-ONLY re-tick is 'updated'. A moved STRIKE on
    the same block still refuses, which is what makes that a line rather than a waiver.

    THE THIRD CLAIM IS THE EMITTER'S OWN, and it is why a re-tick is built by hand here rather than
    by re-emitting: `Weight` is a normalised vega, so one moved print re-weighs the whole ladder and
    a re-EMITTED chain is a re-authoring by arithmetic. Both readings are asserted.
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

    # THE VALUE-ONLY RE-TICK: one wing's mid, its two-way and its stamp move, and nothing else -
    # same expiries, same strikes, same option types, same weights, same row order
    row = block['instrument']['European_Options'][-1]
    ticked = value_tick(block, {len(block['instrument']['European_Options']) - 1: {
        'Quoted_Market_Value': row['Quoted_Market_Value'] + 4.0,
        'Quoted_Bid': row['Quoted_Bid'] + 4.0, 'Quoted_Ask': row['Quoted_Ask'] + 4.0,
        'Timestamp': {'.Timestamp': '2026-09-01'}}})
    assert ticked != block, 'the tick moved nothing - this gate is vacuous'
    assert update_market_quote(document, name, ticked) == 'updated'

    # and a MOVED CONTRACT on the same block is still a re-authoring
    moved = value_tick(block, {0: {'Strike': block['instrument'][
        'European_Options'][0]['Strike'] + 25.0}})
    with pytest.raises(ValueError, match='structure differs'):
        update_market_quote(document, name, moved)

    # the emitter's own re-emission of a moved market: the contracts stand, the WEIGHTS do not
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
    """THE PARTITION, on a chain block loaded through the real decoder - the statement the roadmap
    row is closed by, in the coordinates the two staleness dimensions are named in.

    A premium, its two-way and its stamp move `values_hash` and leave `plan_hash` HEX-IDENTICAL, so
    a re-quoted chain is a tick rather than a recompile; a moved strike moves the plan, because the
    contract itself changed. `market_patch` publishes the ladder's own value rows keyed by its own
    table name and `patch_market` takes them straight back, which is the values-plane identity this
    family now has and did not.
    """
    import derivus
    from derivus.config import CustomJsonEncoder

    name, block = emitted()
    context = derivus.Context().load_json(
        (json.dumps(job_document({name: block}), cls=CustomJsonEncoder), 'chain-tick'))
    prices = context.current_cfg.params['Market Prices']
    rows = prices[name]['instrument']['European_Options']
    assert len(rows) > 1, 'the ladder arrived with no quotes - nothing below means anything'

    # the values half is keyed by the family's OWN table, and it is one entry per row
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

    # the other direction, on the same block: a moved strike is a re-authoring and says so
    with pytest.raises(ValueError, match='Strike is structural, not a value'):
        context.patch_market({name: {'European_Options': [
            {'Strike': 4000.0}] + [{} for _ in rows[1:]]}})
    with pytest.raises(ValueError, match='Weight is structural, not a value'):
        context.patch_market({name: {'European_Options': [
            {'Weight': 0.5}] + [{} for _ in rows[1:]]}})


#: A SHORT ladder and a low evaluation cap - the reading is about the FIT'S ARITHMETIC over a chain
#: block, not about the ladder, exactly as `test_service.hand_authored_hn_block` says of its own.
#: The component fit re-bootstraps the whole L strip per outer iterate and every price derives its
#: own quadrature bound, so a five-pillar ladder to three years is 756 daily steps a price and
#: hours of wall clock. Two pillars and one wing expiry make four contracts, which is why the floor
#: is taken down DELIBERATELY here and nowhere else.
E2E_LADDER = EquityLadder(pillars=(0.25, 0.5), wing_pillars=(0.25, 0.5), minimum_contracts=4)


def test_the_component_family_fits_the_chain_block_with_no_authored_surface(caplog):
    """END TO END, AND WITH NO SURFACE IN THE BOOK AT ALL - the standing gate the emitter build
    deferred, and the whole point of the landing.

    THE WORLD CARRIES NO `EquityPriceVol.SPX`. A `Premium` block reads none: the family's
    `quote_type_references` says `Implied_Volatility` reads the surface and `Premium` does not, so
    the `Volatility` NAME the block still carries (the surface its marks will be read off) is never
    resolved. Before this landing the reference was declared REQUIRED and resolved before the
    branch that never reads it, and a book without a surface got
    `Unable to bootstrap ... - skipping` - no factor, no exception, and a chain-sourced fit that
    could not run unless somebody had already fitted the same chain. That is the circularity, and
    the assertion below is that it is gone.

    What this measures, on this workstation: whether the block resolves the references its quote
    type actually reads, whether the L bootstrap brackets its pillars against LISTED premiums (the
    FX fixtures bracket against premiums synthesised off a surface, which is a smoother target),
    and what the family writes when it does. The ATM residual and the wall clock are RECORDED.
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
    # the equity leverage shape: index skew is steep, stable and ONE-SIGNED, and a positive
    # Gamma_1 is what says the model fitted the shape the chain actually carries
    assert written['Gamma_1'] > 0.0, 'a falling index smile fitted with the FX leverage sign'
    # the L curve is the reading this family exists for: a knot at tenor zero anchoring q0 = L(0),
    # then one per ATM pillar, every level a positive variance
    from derivus import utils
    curve = written[utils.HN_COMPONENT_CURVE_NAME]
    assert len(curve.array) == 1 + len(E2E_LADDER.pillars) and curve.array[0][0] == 0.0
    assert all(level > 0.0 for _, level in curve.array), curve.array

    # THE READING, recorded: the bootstrap's own ATM residual off its report, and the wall clock
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
    """The other half of the same statement, so "optional" does not read as "ignored".

    Under `Quote_Type` Premium the surface is INERT whether the book carries one or not - the fit
    is posed against the chain's own prints. So the same chain block fits to the SAME parameters in
    a world with a surface and in one without, which is what says the surface stopped being an
    input rather than merely stopped being fetched.
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
    """`(config, refusal or None)` - one block through the REAL bootstrap in a real world, with
    whatever it raised. The config is returned either way, so a gate can ask what was written."""
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
    """The canned chain's own block with one reference BLANKED - the shape a block takes when its
    author leaves a panel empty, which is the case that used to skip."""
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
    """THE SKIP IS DEAD, and this is the three-part assertion that says so: the exception REACHES
    THE CALLER, NO factor is written, and no 'skipping' line lands anywhere.

    What stood here caught every resolution failure, logged
    `Unable to bootstrap ... - skipping`, and moved on - so a job whose block named nothing the book
    carried completed, wrote no price factor, and told its caller precisely nothing. That is the
    hollow-container failure mode in bootstrap clothing, and it is the reason a chain-sourced fit
    could not run: the reference it tripped on was the one its quote type never reads.

    BOTH QUOTE TYPES ENUMERATE WHAT THEY REQUIRE in the message, because the remedy differs between
    them - a missing `Volatility` under `Implied_Volatility` is fixed by naming a surface OR by
    quoting premiums, and the refusal says both.
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
    """The second half: the field is NAMED and the book has nothing under that name. A caller needs
    the factor it looked for spelled out, because the name in the block and the key in `Price
    Factors` differ by a TYPE the block never states - that is what `<field>_Type` is for."""
    _, block = equity_hn_block(canned_chain(), FORWARD, E2E_LADDER)
    block['instrument']['Discount_Rate'] = 'ZAR'
    block['instrument']['Max_Iterations'] = 2

    config, refusal = bootstrapped(block, surface=False)
    assert refusal is not None, 'a named-but-absent curve still skipped'
    assert 'InterestRate.ZAR' in str(refusal) and 'Discount_Rate' in str(refusal), str(refusal)
    assert not [factor for factor in config.params['Price Factors']
                if factor.startswith('HestonNandiComponentModelParameters.')]

    # and a Quote_Type this family does not fit refuses before anything is resolved at all
    block['instrument'].update({'Discount_Rate': 'USD', 'Quote_Type': 'Mid'})
    _, refused = bootstrapped(block, surface=False)
    assert refused is not None and 'Quote_Type' in str(refused)
    assert 'Implied_Volatility' in str(refused) and 'Premium' in str(refused)


# =============================================================================================
# 8  the forward the fit builds is the forward the pricer builds
# =============================================================================================

#: THE REPO CURVE and the spread that makes this gate non-vacuous. 125bp is a hard-to-borrow
#: basket rather than an index, chosen so a forward built on the wrong curve misses by whole index
#: points at every pillar - at the 3y rung, 5000 exp(0.0125*3) is 191 points of forward.
REPO = 'USD-REPO'
REPO_SPREAD = 0.0125
#: A curve at ZERO, so the probe deal's discount factor is exactly 1.0 and its MtM IS the forward.
ZERO_CURVE = 'USD-ZERO'
#: The pillars the identity is measured at, in whole days off the base date.
FORWARD_DAYS = (91, 182, 365, 730, 1095)


def flat(value):
    return {'.Curve': {'meta': [], 'data': [[0.0, value], [5.0, value]]}}


def repo_world(market_prices=None, deals=()):
    """The index world with a GENUINE REPO SPREAD: the equity funds on `USD-REPO` at 5.25% while
    its deals discount on `USD` at 4%, which is the world where one reference doing two jobs parts
    the calibrated forward from the priced one. Plus a curve at zero, for the probe deal."""
    document = job_document(market_prices, surface=False, repo=REPO, factors={
        'InterestRate.{}'.format(REPO): {'Currency': 'USD', 'Day_Count': 'ACT_365',
                                         'Sub_Type': None, 'Curve': flat(RATE + REPO_SPREAD)},
        'InterestRate.{}'.format(ZERO_CURVE): {'Currency': 'USD', 'Day_Count': 'ACT_365',
                                               'Sub_Type': None, 'Curve': flat(0.0)}})
    document['Calc']['Deals']['Deals']['Children'] = [
        {'Instrument': {'.Deal': deal}} for deal in deals]
    return document


def probe_block(funding=REPO, days=FORWARD_DAYS):
    """A `HestonNandiModelPrices.SPX` block whose every quote leaves `Strike` at ZERO.

    THE BLOCK IS A PROBE. `0 reads the forward` is the declared meaning of that column, and the
    family fills the row in with the forward it built - in place, on the block the config holds. So
    after a real bootstrap each row's `Strike` IS the fit's own forward at that expiry, read out of
    the engine rather than recomputed beside it.

    `Steps_Per_Year` is 12 because `n` is the only thing it moves here and a 3y rung at 252 is 756
    sequential recursions per price. The premiums are plausible ATM prints so the seed inversion has
    something to invert; the fit's quality is not what is being measured.
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
    """`(the fit's forwards, the five fitted parameters, the full run output)` - one real bootstrap
    of `block` in the repo world, with `deals` priced against the same market in the same job."""
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
    """One `EquityForwardDeal` per expiry, struck at ZERO and discounted on the zero curve, so its
    base-date MtM is `utils.calc_eq_forward` itself: units x (forward - 0) x 1.0 x 1.0.

    THIS IS THE PRICER'S FORWARD AND NOT A REPLICA OF IT. The deal reads `EquityPrice.SPX`'s own
    `Interest_Rate` (the repo curve) against `DividendRate.SPX` through the compiled factor path -
    the same call every equity option in the library makes."""
    return [{'Object': 'EquityForwardDeal', 'Reference': 'FWD{}'.format(index), 'Tags': '',
             'MtM': '', 'Forward_Price': 0.0, 'Buy_Sell': 'Buy', 'Payoff_Type': 'Standard',
             'Equity_Volatility': '', 'Maturity_Date': {'.Timestamp': expiry.isoformat()},
             'Equity': 'SPX', 'Units': 1.0, 'Currency': 'USD',
             'Discount_Rate': ZERO_CURVE, 'Payoff_Currency': 'USD'}
            for index, expiry in enumerate(expiries)]


def test_the_calibrated_forward_is_the_priced_forward_at_every_pillar():
    """THE ROW THAT CLOSED, measured: one `Discount_Rate` used to FUND the calibrated forward and
    DISCOUNT the premium, while `utils.calc_eq_forward` grows the priced forward on the equity's own
    repo curve. On an index with a borrow spread the two forwards parted, and the fit sat at
    coordinates the pricer never visits.

    The block declares `Funding_Rate` now - a reference `resolve` reads like any other, `_Type`
    discipline included - and the fit grows its forward on it while the premium still discounts on
    `Discount_Rate`. Both forwards are read out of the ENGINE: the fit's off the `Strike` column it
    filled in, the pricer's off an `EquityForwardDeal`'s own MtM in the same job on the same market.

    NON-VACUOUS BY CONSTRUCTION: the same probe with no `Funding_Rate` is measured too, and it is
    the old arithmetic - off by exactly the spread's own carry at every pillar.
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

    # and the world is one where it MATTERS: with no funding curve declared the fit falls back to
    # the discount curve, which is the old arithmetic and misses by the spread's own carry
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
    """THE BIT-IDENTITY BAR, stated structurally rather than against a golden file.

    The funding basis rides in as a shift on `q`: `effective_yield` adds `r - f` to the carry, and
    with no `Funding_Rate` the term is not evaluated at all - so `q` is the float the family always
    computed and every expression downstream (the forward, the per-step carry, the `exp(-qt)`
    rescale, the seed inversion) is the same bits. Where a funding curve IS declared and names the
    discount curve, `r - f` is exactly 0.0 and `q + 0.0` is `q`.

    Both halves are asserted in HEX, not to a tolerance: a fit run twice over one probe world - once
    with no `Funding_Rate`, once with it naming the discount curve - writes the same five
    parameters to the bit, and the forwards it filled in match to the bit.
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

    # and the same statement through a REAL fit, over the probe world, both ways
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
    writes no `Funding_Rate` at all - a field spelling out a name nobody chose would be a
    declaration nobody made."""
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
    """LIVE SMOKE, and a workstation with no terminal is a SKIP rather than a failure - the whole
    package is built to import and gate on a machine that has never seen blpapi.

    WHAT IS ASSERTED IS THE ROUTE, not the market. The claim is that the bulk chain read answers,
    that its members batch through the scalar reader, and that every one of them comes back as a
    contract or as a NAMED refusal - so the census is PRINTED and never asserted: a chain read
    after the close screens to almost nothing on `one-sided` alone, and a gate that failed on that
    would be measuring the clock rather than the code.

    IT IS SLOW BECAUSE THE CHAIN IS BIG. An index chain is thousands of listed contracts and the
    fetch asks every one of them ten fields, fifty at a time, over a Desktop API that takes its
    time; `on_batch` prints the progress so a quiet ten minutes is legible rather than alarming.
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
    # every member the terminal named is accounted for, one way or the other - which is the ledger
    # discipline, and the only thing about a live chain that does not depend on the hour. The
    # ledger can hold MORE than the probe asked about, because a chain row nobody could read a
    # ticker out of is refused before it becomes a member at all
    assert len(chain.contracts) + len(chain.rejected) >= seen[-1][1] > 0
