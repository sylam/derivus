"""The verified swap strip as an `InterestRatePrices` block - `derivus_bloomberg.ir_curve`.

Nothing here opens a socket except the last gate, which skips by name where this workstation has no
terminal. Everything else runs on ONE canned world - a seeded USD OIS strip and a seeded ZAR JIBAR
strip, both walked through the package's real discovery grammar into a real verified map, with a
poison table of dead prints authored on purpose - driven through the package's own reader
(`BloombergSession`'s event walk, canned rows) and the real engine seams
(`config.update_market_quote` and `bootstrappers.quote_nodes` / `quote_knots`, all imported
READ-ONLY).

WHAT IS HELD:

  the budget       `ir_curve` imports the standard library, this package's own modules and a LAZY
                   blpapi - read off the source and proved again in a fresh interpreter, which also
                   shows the chain emitter's own no-pandas claim survives the two new modules
  the declaration  the SHIPPED conventions, spelled out as data so the owner checks a gate rather
                   than a paragraph; a currency with no `conventions` block refuses naming every
                   missing field at once; a convention nobody reads refuses too
  the selection    the documented prints and the documented rejects BY NAME - unverified, invalid,
                   unpriced, off-market, crossed, undated, stale - one canned print per verdict
  the conventions  land on the authored `Deal` blocks exactly as the seed declares them: the USD
                   strip is an OIS-compounded `StructuredDeal` with one float item per fixing
                   window, the ZAR strip a quarterly `SwapInterestDeal`, and the front point is the
                   one the seed named rather than whichever overnight print was in the map
  the partition    the OIS fixing windows tile EVERY coupon exactly, so the two legs accrue the same
                   span per coupon - including at a coupon boundary that lands on a weekend, which
                   is where the float leg used to lose two days of a one-year accrual
  the quote       is NOT authored into the deal - every rate-carrying field is a neutral zero and
                   the print rides in `Quoted_Market_Value`, which is WHY a re-tick is a tick
  the knot rule    one knot per used quote at its last cashflow date; two benchmarks maturing on one
                   day refuse by name rather than reaching the solve as a singular Jacobian
  the round trip   `update_market_quote` installs the block and UPDATES a value-only re-tick; a
                   moved convention refuses as a new plan, and a ROLLED DATE reaches the book
                   through `reauthor`
  the fields       every authored deal key is a field the COMMITTED instrument schema declares, and
                   every declared key it omits is one the engine stamps or an admin key - which is
                   what `construct_instrument` does NOT check, so nothing else would say so
  the compile      the engine's own `quote_nodes` / `quote_knots` construct the authored deals and
                   read their last cashflow dates - no solve, and no schema check either
  determinism      the same canned answers emit the same bytes
  the taxonomy     a broken seed is a CONFIGURATION refusal and never a no-terminal skip
  live smoke       one real strip off this workstation's terminal, or a skip by name

NO MONKEYPATCHING. The canned terminal is a `BloombergSession` subclass whose event walk yields
rows, which is `test_bloomberg_discover.Walked` verbatim; the engine is imported and never touched.
"""
import ast
import copy
import datetime
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_bloomberg import discover, ir_curve
from derivus_bloomberg.errors import (BloombergConfigurationError, BloombergFXError,
                                      BloombergRequestError, BloombergUnavailable, IncompleteStrip)
from derivus_bloomberg.ir_curve import (CurveScreen, curve_conventions, fetch_curve_strip,
                                        ir_curve_block, reauthor, screen_strip)
from derivus_bloomberg.session import BloombergSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AS_OF = datetime.date(2026, 8, 31)        # a Monday
YESTERDAY = '2026-08-28'                  # the Friday before it - a live print
LONG_AGO = '2026-07-15'                   # past any sane stale bound

#: WHAT "NO TERMINAL ON THIS WORKSTATION" ACTUALLY LOOKS LIKE - and the reason the live smokes catch
#: this tuple rather than `BloombergFXError`.
#:
#: `BloombergConfigurationError` hangs off that base with everything else (one taxonomy, one base -
#: `errors.py` says why, and `derivus.service` and `derivus_mcp` catch it), so a smoke gate catching
#: the BASE reads a BROKEN SEED as an absent terminal and skips green. A workstation whose
#: `seed.json` had lost its `conventions` block would report "no Bloomberg terminal answering" while
#: the terminal answered perfectly well - the one failure a smoke gate exists to surface, reported as
#: the one thing it is allowed to skip for. The hierarchy is left alone and the CATCH SITES are
#: narrowed instead: these two are the transport, and nothing else is.
NO_TERMINAL = (BloombergUnavailable, BloombergRequestError)


def no_terminal_reason(error):
    """Why a live smoke is skipping, NAMED FOR WHAT ACTUALLY HAPPENED.

    The two members of `NO_TERMINAL` are not the same event and a desk does different things about
    them. `BloombergUnavailable` is the absent terminal - no SDK, no session, nothing listening, and
    the gate is simply not runnable here. `BloombergRequestError` is a terminal that ANSWERED AND
    REFUSED: a timeout, or the one this workstation meets most often -

        responseError = { code = -4001  category = "LIMIT"
                          message = "Daily capacity reached."
                          subcategory = "DAILY_CAPACITY_REACHED" }

    - the daily data quota, spent, which is a thing to wait out or go and raise rather than a
    workstation without a terminal. Reporting that as "no Bloomberg terminal answering" is the same
    mistake as reporting a broken seed that way, one step milder: a skip has to name the thing
    somebody would go and fix, or the census it lands in says the wrong thing about the desk.
    """
    if isinstance(error, BloombergUnavailable):
        return 'no Bloomberg terminal answering on this workstation: {}'.format(error)
    return 'this workstation\'s terminal answered and refused the request ({}): {}'.format(
        type(error).__name__, error)

#: The SHIPPED declarations, restated here as data. This is the gate the owner checks: a convention
#: is market fact rather than code, so the seed is where it lives and this is where a change to it
#: has to be agreed. USD SOFR OIS settles T+2 and pays annual/annual on ACT/360 with an overnight
#: compounded float leg; ZAR SASW settles same day and pays quarterly/quarterly on ACT/365 against
#: 3M JIBAR, whose own fixing - not ZARONIA - is the front of a JIBAR curve.
SHIPPED = {
    'USD': {'curve_day_count': 'ACT_365', 'spot_days': 2, 'front': 'overnight',
            'front_day_count': 'ACT_360', 'authoring': 'OIS',
            'fixed_frequency': '1Y', 'float_frequency': '1Y',
            'fixed_day_count': 'ACT_360', 'float_day_count': 'ACT_360',
            'notional': 1000000.0, 'quote_scale': 1.0},
    'ZAR': {'curve_day_count': 'ACT_365', 'spot_days': 0, 'front': 'fixings/3M',
            'front_day_count': 'ACT_365', 'authoring': 'Swap',
            'fixed_frequency': '3M', 'float_frequency': '3M',
            'fixed_day_count': 'ACT_365', 'float_day_count': 'ACT_365',
            'notional': 1000000.0, 'quote_scale': 1.0},
}


def packaged_seed():
    return json.load(open(os.path.join(ROOT, 'derivus_bloomberg', 'seed.json'), encoding='utf-8'))


#: The canned world's own seed - the shipped vocabulary cut down to two currencies and a handful of
#: tenors, with the SHIPPED conventions carried across unchanged so what the gates author is what a
#: desk would get.
SEED = {
    'fx_vol': {}, 'fx_spot': {},
    'rates': {
        'USD': {'prefix': 'USOSFR', 'expect': 'USD OIS', 'weeks': True,
                'years': [1, 2, 5],
                'overnight': {'security': 'SOFRRATE Index', 'expect': 'SOFR'},
                'conventions': SHIPPED['USD']},
        'ZAR': {'prefix': 'SASW', 'expect': 'ZAR SWAP QTR', 'years': [1, 2, 3, 5, 10],
                'overnight': {'security': 'ZARONIA Index',
                              'expect': 'South African Overnight'},
                'fixings': {'1M': {'security': 'JIBA1M Index', 'expect': 'Johannesburg'},
                            '3M': {'security': 'JIBA3M Index', 'expect': 'Johannesburg'},
                            '6M': {'security': 'JIBA6M Index', 'expect': 'Johannesburg'}},
                'conventions': SHIPPED['ZAR']},
    },
}

#: THE POISON TABLE, one entry per verdict the screen documents, authored as a terminal really
#: answers them. `None` means the map never verified the ticker at all.
POISON = {
    'USOSFR3Z BGN Curncy': None,                                          # unverified
    'USOSFR2Z BGN Curncy': {'PX_BID': 4.40, 'PX_ASK': 4.30},              # crossed
    'USOSFR5 BGN Curncy': {'LAST_UPDATE_DT': LONG_AGO},                   # stale
    'USOSFR1 BGN Curncy': {'PX_LAST': None},                              # unpriced
    'SASW10 BGN Curncy': {'PX_LAST': 8825.0},                             # off-market
    'SASW2 BGN Curncy': {'LAST_UPDATE_DT': 'N/A'},                        # undated
    'SASW5 BGN Curncy': {},                                               # invalid: nothing at all
}

#: What a clean point answers. Levels are invented and only have to be plausibly shaped: nothing
#: below prices anything off them.
CLEAN = {'SOFRRATE Index': 4.33, 'USOSFR1Z BGN Curncy': 4.31, 'USOSFR2Z BGN Curncy': 4.30,
         'USOSFR3Z BGN Curncy': 4.29, 'USOSFR1 BGN Curncy': 4.02, 'USOSFR2 BGN Curncy': 3.88,
         'USOSFR5 BGN Curncy': 3.79, 'ZARONIA Index': 7.02, 'JIBA1M Index': 7.28,
         'JIBA3M Index': 7.41, 'JIBA6M Index': 7.55, 'SASW1 BGN Curncy': 7.62,
         'SASW2 BGN Curncy': 7.94, 'SASW3 BGN Curncy': 8.21, 'SASW5 BGN Curncy': 8.55,
         'SASW10 BGN Curncy': 8.83}


# =============================================================================================
# the canned world
# =============================================================================================

class Walked(BloombergSession):
    """A session whose event walk is canned rows - `test_bloomberg_discover.Walked` verbatim, so
    the emitters are driven through the package's REAL readers with no socket and no patching."""

    def __init__(self, rows):
        super().__init__()
        self._api = self._session = self._service = object()
        self.rows = rows

    def _walk(self, securities, fields):
        for security in securities:
            row = self.rows.get(security)
            if row is not None:
                yield security, row['error'], row['fields']


def candidates(seed=None):
    return list(discover.candidates_from_seed(seed or SEED))


def verified_map(seed=None, poison=None):
    """A real security map, built by the package's own discovery off canned NAME answers - so what
    the emitters read is the artifact a workstation would actually hold rather than a hand-built
    dict. A poisoned entry of `None` never verifies, which is how a dead ticker reaches the strip
    emitter as `unverified` rather than as a hole."""
    seed = seed or SEED
    poison = POISON if poison is None else poison
    report = {}
    for candidate in candidates(seed):
        if poison.get(candidate.security, False) is None:
            report[candidate.security] = {'ok': False, 'error': 'Unknown/Invalid Security',
                                          'fields': {}}
            continue
        report[candidate.security] = {'ok': True, 'error': None, 'fields': {
            'NAME': ' '.join(candidate.expect) + ' ' + candidate.path[-1],
            'PX_LAST': CLEAN.get(candidate.security, 5.0), 'LAST_UPDATE_DT': YESTERDAY}}
    verdicts = discover.verify(candidates(seed), report, AS_OF)
    return discover.build_map(seed, verdicts, AS_OF.isoformat())


def canned_session(seed=None, poison=None):
    """The terminal answering the strip: a clean two-way and yesterday's print everywhere, with the
    poison table overriding whichever field each dead case is dead in."""
    seed = seed or SEED
    poison = POISON if poison is None else poison
    rows = {}
    for candidate in candidates(seed):
        override = poison.get(candidate.security, {})
        if override is None:
            continue
        if override == {} and candidate.security in poison:
            rows[candidate.security] = {'error': 'Unknown/Invalid Security', 'fields': {}}
            continue
        value = CLEAN.get(candidate.security, 5.0)
        fields = {'PX_LAST': value, 'PX_BID': round(value - 0.01, 4),
                  'PX_ASK': round(value + 0.01, 4), 'LAST_UPDATE_DT': YESTERDAY}
        fields.update(override)
        rows[candidate.security] = {'error': None,
                                    'fields': {key: item for key, item in fields.items()
                                               if item is not None}}
    return Walked(rows)


def strip_of(currency, seed=None, poison=None, screen=None, as_of=AS_OF, curve=None):
    # the poison table is resolved HERE and passed on explicitly, so the map and the session are
    # always built from the SAME one - a cell that is dead in one half and live in the other is a
    # fixture that gates nothing
    seed, poison = seed or SEED, POISON if poison is None else poison
    return fetch_curve_strip(canned_session(seed, poison), verified_map(seed, poison), seed,
                             currency, as_of, curve=curve, screen=screen)


def block_of(currency, **kwargs):
    return ir_curve_block(strip_of(currency, **kwargs))


# =============================================================================================
# 1  the dependency budget
# =============================================================================================

def imported_names(source):
    """The top-level names a file imports, however deep in it the import sits - relative imports are
    skipped, since they resolve inside the package by construction. `test_equity_chain`'s own
    helper, which is `test_spine_imports`' before it."""
    names = set()
    with open(source, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.add((node.module or '').split('.')[0])
    return names


def test_the_curve_emitter_imports_the_standard_library_and_nothing_else():
    """The package's own budget, extended to a new module: the standard library, this package's own
    modules, and NO ENGINE. The chain emitter's stricter no-pandas budget is deliberately NOT
    claimed here - `ir_curve` reaches `discover` for its ticker grammar rather than spelling a
    second one, `discover` reaches `security_map`, and that carries pandas. Reusing the grammar is
    worth a dependency the package's map layer already pays for; inventing a second spelling of a
    Bloomberg ticker would not be."""
    imported = imported_names(os.path.join(ROOT, 'derivus_bloomberg', 'ir_curve.py'))
    assert imported <= {'collections', 'datetime', 'math', 'dataclasses', 'typing',
                        'derivus_bloomberg'}, sorted(imported)
    assert imported.isdisjoint({'derivus', 'torch', 'pandas', 'numpy', 'scipy', 'blpapi'}), \
        sorted(imported)
    assert imported


def in_a_fresh_interpreter(statements):
    code = ('import json, sys; {}; '
            'print(json.dumps(sorted({{name.split(".")[0] for name in sys.modules}})))'.format(
                statements))
    done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    assert done.returncode == 0, done.stderr
    return set(json.loads(done.stdout))


def test_importing_the_curve_emitter_lands_no_engine_and_no_blpapi():
    """The source gate's answer proved a second way, because the first trusts the parser and reads
    ONE FILE while an import runs a package.

    The two claims are asymmetric ON PURPOSE. `derivus` and `blpapi` must not land: the package has
    to import on a workstation that has never seen a terminal, and it must never reach across into
    the engine. `pandas` DOES land, through the map layer, and is asserted POSITIVELY - a gate that
    quietly allowed it would not be stating a budget, it would be omitting one.

    And the last two lines are the reason these emitters are re-exported LAZILY: an eager
    `from .ir_curve import ...` in the package's `__init__` would land pandas on
    `import derivus_bloomberg.equity_chain` behind that module's back, falsifying its own gate from
    outside it.
    """
    landed = in_a_fresh_interpreter('import derivus_bloomberg.ir_curve')
    assert 'derivus_bloomberg' in landed, 'the module did not import'
    assert landed.isdisjoint({'derivus', 'torch', 'blpapi'}), sorted(
        landed & {'derivus', 'torch', 'blpapi'})
    assert 'pandas' in landed, 'the stated budget says the grammar costs pandas'

    chain = in_a_fresh_interpreter('import derivus_bloomberg.equity_chain')
    assert chain.isdisjoint({'derivus', 'torch', 'blpapi', 'pandas', 'numpy'}), sorted(
        chain & {'derivus', 'torch', 'blpapi', 'pandas', 'numpy'})
    assert 'derivus_bloomberg' in in_a_fresh_interpreter(
        'from derivus_bloomberg import fetch_curve_strip, ir_curve_block')


# =============================================================================================
# 2  the declaration
# =============================================================================================

def test_the_shipped_conventions_are_the_declared_ones():
    """THE OWNER'S GATE. Every convention this package will author a USD or ZAR benchmark in, as
    data, read off the seed the wheel ships - so a change to a market convention is a diff here and
    a conversation, rather than a number that moved inside a cashflow.

    The two that matter most are the ones a ticker cannot tell you: USD OIS accrues ACT/360 on both
    legs where the curve's own tenors are ACT/365, and the ZAR front point is the 3M JIBAR fixing
    rather than the ZARONIA print sitting beside it in the same map - a JIBAR curve seeded with an
    overnight rate is a basis error nothing downstream would report.
    """
    shipped = packaged_seed()['rates']
    for currency, declared in SHIPPED.items():
        assert shipped[currency]['conventions'] == declared, currency
        conventions = curve_conventions(packaged_seed(), currency)
        assert conventions.authoring == declared['authoring']
        assert conventions.deal_type == {'OIS': 'StructuredDeal',
                                         'Swap': 'SwapInterestDeal'}[declared['authoring']]
    assert curve_conventions(packaged_seed(), 'USD').front == 'overnight'
    assert curve_conventions(packaged_seed(), 'ZAR').front == 'fixings/3M'
    # and the currencies the shipped seed maps but does NOT declare conventions for are the ones a
    # desk has to fill in - they must refuse rather than inherit a neighbour's
    for currency in ('EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD'):
        with pytest.raises(BloombergConfigurationError, match='carries no `conventions` block'):
            curve_conventions(packaged_seed(), currency)


def test_a_currency_without_its_conventions_refuses_naming_every_missing_field():
    """A convention block filled in half way authors an instrument nobody stated, so the refusal
    lists the WHOLE questionnaire at once - a desk extending a seed should not learn it one terminal
    round trip at a time."""
    seed = copy.deepcopy(SEED)
    del seed['rates']['USD']['conventions']['fixed_day_count']
    del seed['rates']['USD']['conventions']['spot_days']
    with pytest.raises(BloombergConfigurationError) as refused:
        curve_conventions(seed, 'USD')
    message = str(refused.value)
    assert 'declares no spot_days, fixed_day_count' in message
    for name in ir_curve.REQUIRED_CONVENTIONS:
        assert name in message

    with pytest.raises(BloombergConfigurationError, match='the seed names no rates entry for GBP'):
        curve_conventions(seed, 'GBP')

    # a convention nobody reads is a convention that is not APPLIED, which is worse than a missing
    # one: it reads as a declaration and does nothing
    seed = copy.deepcopy(SEED)
    seed['rates']['ZAR']['conventions']['payment_lag'] = 2
    with pytest.raises(BloombergConfigurationError, match='payment_lag'):
        curve_conventions(seed, 'ZAR')


def test_a_declaration_this_emitter_cannot_author_refuses_at_the_seed():
    """The three ways a declaration can be wrong rather than absent - and each refuses where a desk
    can fix it, not inside a cashflow generator.

    The day-count refusal is the sharpest one: an authored `Accrual_Year_Fraction` is used VERBATIM
    by `make_float_cashflows`, so a day count this module cannot compute must never be quietly
    approximated. `ACT_365_ISDA` and `ACT_ACT_ICMA` are days/365 in the engine behind a `# TODO`,
    and reproducing that on trust is how a known-wrong number gets a second home.
    """
    for field, value, expected in (
            ('authoring', 'Bootstrap', 'is not a shape this emitter writes'),
            ('float_day_count', 'ACT_ACT_ICMA', 'is not a float_day_count this emitter'),
            ('fixed_day_count', '_30_360', 'is not a fixed_day_count this emitter'),
            ('spot_days', -1, 'spot_days must be a whole number'),
            ('fixed_frequency', '1', 'is not a tenor this emitter can read'),
            # a zero-length coupon period is the one bad declaration a refusal cannot be read out
            # of: the roll would walk towards a maturity it never reaches
            ('float_frequency', '0M', 'has to be a positive period')):
        seed = copy.deepcopy(SEED)
        seed['rates']['USD']['conventions'][field] = value
        with pytest.raises(BloombergConfigurationError, match=expected):
            curve_conventions(seed, 'USD')


def test_a_front_the_seed_could_not_name_refuses_before_it_mis_authors_the_short_end():
    """`front` IS THE ONE REQUIRED CONVENTION WHOSE WRONG VALUE IS SILENTLY AUTHORABLE, which is why
    it is validated against the seed rather than against itself.

    Every other convention refuses on its own value - a day count this module cannot compute, an
    authoring shape it does not write. This one is a PATH INTO THE SEED, and a path that names
    nothing does not fail, it aims somewhere else: `front: 'strip/1Y'` matches the 1Y par swap's own
    map path, `_front_label` finds no `fixings` at the path's second part and manufactures the label
    `overnight`, and `strip_dates` then authors that 1Y par swap as a ONE-DAY DepositDeal which the
    Descriptor calls overnight. The block comes out well formed, the strip is short by a knot, the
    short end is seeded with an instrument nobody quoted, and nothing downstream can tell.

    So the refusal names the field, the value declared and the whole admissible set - which is what
    a desk needs to fix it, and it is `_seeded_fronts`' own list rather than a second spelling of it.
    """
    seed = copy.deepcopy(SEED)
    seed['rates']['ZAR']['conventions']['front'] = 'strip/1Y'
    with pytest.raises(BloombergConfigurationError) as refused:
        curve_conventions(seed, 'ZAR')
    message = str(refused.value)
    assert "ZAR declares its front as 'strip/1Y'" in message
    assert 'one-day overnight deposit' in message
    for admissible in ('overnight', 'fixings/1M', 'fixings/3M', 'fixings/6M'):
        assert admissible in message

    # the refusal is at the SEED, so it fires before anything is fetched or authored
    with pytest.raises(BloombergConfigurationError, match='not an entry its seed could name'):
        strip_of('ZAR', seed=seed)

    # USD seeds no fixings at all, so `overnight` is the only front it could name
    seed = copy.deepcopy(SEED)
    seed['rates']['USD']['conventions']['front'] = 'fixings/3M'
    with pytest.raises(BloombergConfigurationError) as refused:
        curve_conventions(seed, 'USD')
    assert 'the admissible spellings are overnight.' in str(refused.value)

    # and both SHIPPED declarations pass the check that catches those - a validation nothing real
    # survives is a validation nobody can use
    assert curve_conventions(packaged_seed(), 'USD').front == 'overnight'
    assert curve_conventions(packaged_seed(), 'ZAR').front == 'fixings/3M'


def test_an_unverified_front_point_refuses_and_offers_the_ones_the_seed_names():
    """The SECOND of the two front refusals, and they are different failures. Above, the seed could
    never have named that front; here it names one the seed CAN spell and the map has no verified
    entry for - a seeded fixing that went dead. The front is what seeds the short end, so that
    cannot be skipped into a shorter curve: it has to say so, and say which fronts the seed offers.
    """
    seed = copy.deepcopy(SEED)
    seed['rates']['ZAR']['fixings']['12M'] = {'security': 'JIBA12M Index',
                                              'expect': 'Johannesburg'}
    seed['rates']['ZAR']['conventions']['front'] = 'fixings/12M'
    poison = dict(POISON, **{'JIBA12M Index': None})       # seeded, and never verified
    with pytest.raises(BloombergConfigurationError) as refused:
        ir_curve.strip_entries(verified_map(seed, poison), seed, 'ZAR')
    message = str(refused.value)
    assert "front point as 'fixings/12M'" in message
    assert 'fixings/1M' in message and 'fixings/3M' in message and 'overnight' in message


def test_an_ois_declaration_whose_float_frequency_nobody_reads_refuses():
    """A CONVENTION NOBODY READS IS A CONVENTION THAT IS NOT APPLIED, and this module says so in its
    own refusal text - so it is held to it.

    `_ois_swap` rolls the coupon dates ONCE, off `fixed_frequency`, and hangs both the fixed items
    and the compounded fixing windows on those same boundaries. `float_frequency` is required and
    validated and then read by nothing on that path: declaring USD at `6M` used to emit a
    BYTE-IDENTICAL block, so the seed said one thing and the instrument was another and no gate
    anywhere could see the difference. V1 authors both OIS legs on one schedule, so the declaration
    that says otherwise refuses rather than being quietly ignored.
    """
    seed = copy.deepcopy(SEED)
    seed['rates']['USD']['conventions']['float_frequency'] = '6M'
    with pytest.raises(BloombergConfigurationError) as refused:
        curve_conventions(seed, 'USD')
    message = str(refused.value)
    assert "float_frequency is '6M' against a fixed_frequency of '1Y'" in message
    assert 'BOTH OIS legs on ONE schedule' in message
    with pytest.raises(BloombergConfigurationError, match='read nowhere'):
        strip_of('USD', seed=seed)

    # equal frequencies still emit, which is what makes this a boundary rather than a ban
    assert curve_conventions(SEED, 'USD').float_frequency == '1Y'
    assert len(block_of('USD')[1]['instrument']['Points']) == 3

    # and the `Swap` path READS BOTH - the engine generates each leg on its own frequency there - so
    # an unequal declaration is authored rather than refused. The refusal is about what v1 writes,
    # not about what a swap may be
    swapped = copy.deepcopy(SEED)
    swapped['rates']['ZAR']['conventions']['float_frequency'] = '6M'
    deal = points_of(block_of('ZAR', seed=swapped)[1])['1Y']['Deal']
    assert deal['Pay_Frequency'] == {'.DateOffset': '3M'}
    assert deal['Receive_Frequency'] == {'.DateOffset': '6M'}


# =============================================================================================
# 3  the selection
# =============================================================================================

def test_the_screen_classifies_off_the_terminals_own_answers():
    """The order of distrust, one canned print per verdict. Reading them in this order is what makes
    a verdict actionable: `crossed` and `stale` are different instructions to a desk, and a screen
    that checked the date first would report the second where the first is true."""
    def rate(**extra):
        return ir_curve.RatePrint(**dict(
            {'label': '5Y', 'kind': 'swap', 'security': 'x', 'value': 4.0, 'bid': 3.99,
             'ask': 4.01, 'last_update': YESTERDAY}, **extra))

    cases = {
        'unpriced': rate(value=None),
        'off-market': rate(value=8825.0),
        'crossed': rate(bid=4.10, ask=4.00),
        'undated': rate(last_update='N/A'),
        'stale': rate(last_update=LONG_AGO),
        'live': rate(),
    }
    named = [ir_curve.RatePrint(**dict(item.__dict__, security=verdict))
             for verdict, item in cases.items()]
    accepted, rejected = screen_strip(named, AS_OF)
    assert [item.security for item in accepted] == ['live']
    assert rejected == {verdict: verdict for verdict in cases if verdict != 'live'}

    # a mid-only print is BELIEVED - a two-way the terminal never quoted is not a spread, and the
    # mid is what a curve is built off. `fxvol._scaled_side`'s rule, one family over
    assert screen_strip([rate(security='mid-only', bid=None, ask=None)], AS_OF)[1] == {}


def test_the_canned_strip_is_believed_by_census():
    """Every candidate accounted for, one way or the other - the ledger discipline, and the whole
    reason a short strip is legible rather than alarming. A candidate silently dropped is
    indistinguishable from one never asked about."""
    usd, zar = strip_of('USD'), strip_of('ZAR')
    assert [item.label for item in usd.prints] == ['overnight', '1W', '2Y']
    assert usd.rejected == {
        'USOSFR3Z BGN Curncy': 'unverified', 'USOSFR2Z BGN Curncy': 'crossed',
        'USOSFR1 BGN Curncy': 'unpriced', 'USOSFR5 BGN Curncy': 'stale'}

    assert [item.label for item in zar.prints] == ['3M', '1Y', '3Y']
    assert zar.rejected == {
        'ZARONIA Index': 'not-a-benchmark', 'JIBA1M Index': 'not-a-benchmark',
        'JIBA6M Index': 'not-a-benchmark', 'SASW2 BGN Curncy': 'undated',
        'SASW5 BGN Curncy': 'invalid', 'SASW10 BGN Curncy': 'off-market'}
    # the ZARONIA print IS verified and IS live - it is refused for being the wrong INSTRUMENT on a
    # JIBAR curve, which is the declaration doing its job rather than the screen
    assert zar.census == {'not-a-benchmark': 3, 'undated': 1, 'invalid': 1, 'off-market': 1}

    # and the census a report reads is the same numbers as data, so a caller with a screen and a
    # caller with a log see one account of the fetch rather than two
    counted = ir_curve.quote_census(usd)
    assert (counted['asked'], counted['believed']) == (7, 3)
    assert counted['securities'] == [item.security for item in usd.prints]
    assert counted['refused'] == usd.rejected and counted['curve'] == 'USD'


def test_a_strip_below_its_floor_refuses_naming_what_the_terminal_served():
    """One knot is a flat curve quoted once, so a strip that screened away has to say what it was
    asked and what came back - "no curve" with no census is not something a desk can act on."""
    poison = dict(POISON, **{security: {'PX_LAST': None} for security in
                             ('SASW1 BGN Curncy', 'SASW3 BGN Curncy')})
    with pytest.raises(IncompleteStrip) as refused:
        block_of('ZAR', poison=poison)
    message = str(refused.value)
    assert 'ZAR screened to 1 believed point against a floor of 2' in message
    assert '2 unpriced' in message and '3 not-a-benchmark' in message


# =============================================================================================
# 4  the conventions on the authored deals
# =============================================================================================

def points_of(block):
    return {row['Descriptor'].split()[1]: row for row in block['instrument']['Points']}


def test_the_ois_strip_is_authored_as_the_compounding_rule_requires():
    """A USD OIS benchmark is a CONTAINER over an OIS-compounded floating leg and a fixed leg, with
    ONE FLOAT ITEM PER FIXING WINDOW sharing its coupon's payment date.

    That shape is the whole point and it is not decoration: `pv_float_cashflow_list` compounds
    geometrically only when the reset count differs from the cashflow count, and the reshape that
    makes it so is `compress_no_compounding(groupsize=-1)` under `Compounding_Method='OIS'` - which
    merges a payment date's items into ONE cashflow carrying all their resets, each still at
    `Weight` 1. A leg authored as one item with many resets arrives weighted `1/n` and compounds at
    a fraction of the rate, which is the AVERAGING legs' arithmetic.
    """
    row = points_of(block_of('USD')[1])['2Y']
    assert row['DealType'] == 'StructuredDeal'
    deal = row['Deal']
    assert deal['Net_Cashflows'] == 'Yes'
    floating, fixed = deal['Children']
    assert floating['Object'] == 'CFFloatingInterestListDeal'
    assert floating['Cashflows']['Compounding_Method'] == 'OIS'
    assert floating['Forecast_Rate'] == 'USD'
    assert fixed['Object'] == 'CFFixedInterestListDeal'

    # ANNUAL coupons rolled BACKWARD from maturity, which is where the market puts a stub and what
    # `generate_dates_backward` does when the engine generates a swap's own legs.
    #
    # AND UNADJUSTED, which is the stated limitation rather than an oversight: the second payment
    # falls on a Saturday and stays there. The authored deals carry `Accrual_Calendars: None` and
    # `Payment_Calendars: None`, so the engine rolls nothing either - one convention applied on both
    # sides of the boundary. A desk that needs a settlement calendar has to give the deals one, and
    # a weekend roll invented out here would then disagree with the legs the engine generates itself
    # for the `Swap` authoring, where only Effective and Maturity come from this module.
    payments = sorted({item['Payment_Date']['.Timestamp'] for item in fixed['Cashflows']['Items']})
    assert payments == ['2027-09-02', '2028-09-02']
    assert datetime.date.fromisoformat(payments[1]).weekday() == 5
    assert deal['Children'][0]['Cashflows']['Items'][0]['Accrual_Start_Date']['.Timestamp'] == \
        '2026-09-02', 'the strip starts at T+2, which is what spot_days declares'

    # every float item carries exactly ONE reset, and they tile their coupon exactly
    items = [item for item in floating['Cashflows']['Items']
             if item['Payment_Date']['.Timestamp'] == payments[0]]
    assert all(len(item['Resets']) == 1 for item in items)
    windows = sorted((item['Accrual_Start_Date']['.Timestamp'],
                      item['Accrual_End_Date']['.Timestamp']) for item in items)
    assert windows[0][0] == '2026-09-02' and windows[-1][1] == payments[0]
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))
    # INSIDE a coupon a window starts on a business day; the coupon's OWN start is a boundary
    # whatever weekday it falls on, which is what makes the two legs accrue the same span. This
    # coupon starts on a Wednesday so the two rules agree here and nothing distinguishes them -
    # the weekend boundary is `test_the_ois_fixing_windows_partition_every_coupon`, below, which is
    # where the asymmetry lived and why this gate could not see it
    assert all(datetime.date.fromisoformat(start).weekday() < 5 for start, _ in windows[1:])

    # the DECLARED day count, on the item the engine reads verbatim
    item = items[0]
    assert item['Accrual_Day_Count'] == 'ACT_360'
    span = (datetime.date.fromisoformat(item['Accrual_End_Date']['.Timestamp']) -
            datetime.date.fromisoformat(item['Accrual_Start_Date']['.Timestamp'])).days
    assert item['Accrual_Year_Fraction'] == pytest.approx(span / 360.0)
    assert item['Resets'][0][3] == item['Accrual_Year_Fraction']


def _wire_date(value):
    return datetime.date.fromisoformat(value['.Timestamp'])


def test_the_ois_fixing_windows_partition_every_coupon():
    """THE TWO LEGS ACCRUE THE SAME SPAN, PER COUPON - which is the whole content of "one
    convention on both legs" on a swap the solve holds at PV zero.

    Both legs roll off the SAME coupon dates, so they can only accrue over the same span if the
    float leg's fixing windows tile `[coupon_start, coupon_end]` exactly. A FIXING ACCRUES THROUGH A
    WEEKEND - Friday's rate is what a Saturday-to-Monday gap earns - and it does so at a coupon
    boundary exactly as it does inside a coupon, so the coupon's own start is a window boundary
    whatever weekday it falls on.

    STARTING THE LEG AT THE FIRST BUSINESS DAY INSTEAD IS WHERE THIS WAS WRONG, and the size of it
    is measurable. On this canned USD 5Y (effective 2026-09-02) the coupon starting Saturday
    2028-09-02 accrued 1.00833333 of a year on the float leg against the fixed leg's 1.01388889 -
    two days of a one-year coupon dropped on one side - and the coupon starting Sunday 2029-09-02
    accrued 1.01111111 against the same 1.01388889, losing one. Both now read 1.01388889.

    The 5Y is what carries the weekend boundaries, so the poison table lets it through here: the
    2Y's coupons start on a Wednesday and a Thursday, which is why the gate beside this one could
    run green against a leg that was losing accrual.
    """
    poison = {security: value for security, value in POISON.items()
              if security != 'USOSFR5 BGN Curncy'}
    points = points_of(block_of('USD', poison=poison)[1])
    assert '5Y' in points, 'the 5Y is the benchmark whose coupons land on a weekend'

    weekend = []
    for label, row in sorted(points.items()):
        if row['DealType'] != 'StructuredDeal':
            continue
        floating, fixed = row['Deal']['Children']
        windows = {}
        for item in floating['Cashflows']['Items']:
            windows.setdefault(item['Payment_Date']['.Timestamp'], []).append(
                (_wire_date(item['Accrual_Start_Date']), _wire_date(item['Accrual_End_Date']),
                 item['Accrual_Year_Fraction'], item['Accrual_Day_Count']))
        for coupon in fixed['Cashflows']['Items']:
            start, end = _wire_date(coupon['Accrual_Start_Date']), \
                _wire_date(coupon['Accrual_End_Date'])
            spans = sorted(windows[coupon['Payment_Date']['.Timestamp']])
            # THE PARTITION: the windows start where the coupon starts, end where it ends, and each
            # one begins exactly where the last one finished - no gap, no overlap
            assert spans[0][0] == start, (label, start)
            assert spans[-1][1] == end, (label, end)
            assert all(left[1] == right[0] for left, right in zip(spans, spans[1:])), label
            # so on the same day count the two legs accrue the same span, exactly - stated in DAYS
            # first, because that equality is an integer one and cannot be a rounding
            assert sum((stop - begin).days for begin, stop, _, _ in spans) == (end - start).days
            assert {day_count for _, _, _, day_count in spans} == {coupon['Accrual_Day_Count']} \
                == {'ACT_360'}, 'USD declares ACT/360 on both legs'
            float_accrual = sum(fraction for _, _, fraction, _ in spans)
            assert float_accrual == pytest.approx(coupon['Accrual_Year_Fraction']), (label, start)
            if start.weekday() >= 5:
                weekend.append((label, start.isoformat(), round(float_accrual, 8)))

    # THE MEASURED CASE, named rather than merely covered - so a strip that stopped carrying a
    # weekend boundary would fail here rather than pass this gate vacuously
    assert weekend == [('5Y', '2028-09-02', 1.01388889), ('5Y', '2029-09-02', 1.01388889)]

    # and the first window of such a coupon starts ON the weekend day, which is the change: it is
    # the coupon boundary that is unadjusted, not the fixing calendar
    five = points['5Y']['Deal']['Children'][0]['Cashflows']['Items']
    boundary = sorted(_wire_date(item['Accrual_Start_Date']) for item in five
                      if _wire_date(item['Accrual_Start_Date']).weekday() >= 5)
    assert [date.isoformat() for date in boundary] == ['2028-09-02', '2029-09-02']


def test_the_jibar_strip_is_a_vanilla_swap_on_its_declared_conventions():
    """ZAR declares `Swap` authoring, so a point is ONE `SwapInterestDeal` and the engine generates
    both legs: quarterly against quarterly on ACT/365, with `Index_Tenor` zero months so each
    coupon carries one reset spanning its own accrual - which for a quarterly leg IS 3M JIBAR."""
    row = points_of(block_of('ZAR')[1])['1Y']
    assert row['DealType'] == 'SwapInterestDeal'
    deal = row['Deal']
    assert deal['Pay_Frequency'] == {'.DateOffset': '3M'}
    assert deal['Receive_Frequency'] == {'.DateOffset': '3M'}
    assert deal['Pay_Day_Count'] == 'ACT_365' and deal['Receive_Day_Count'] == 'ACT_365'
    assert deal['Index_Tenor'] == {'.DateOffset': '0M'}
    assert deal['Index_Day_Count'] == 'ACT_365'
    assert deal['Compounding_Method'] == 'None'
    assert deal['Principal'] == 1000000.0
    # spot_days is ZERO for ZAR, so the strip starts at the as-of itself
    assert deal['Effective_Date'] == {'.Timestamp': '2026-08-31'}
    assert deal['Maturity_Date'] == {'.Timestamp': '2027-08-31'}


def test_the_front_point_is_the_one_the_seed_declared():
    """A basis error that would otherwise be free. The ZAR map carries a live, verified ZARONIA
    print AND a live 3M JIBAR fixing; the seed says which of the two is a JIBAR curve's front, and
    the emitter reads that rather than taking the overnight one because it is shorter."""
    zar = points_of(block_of('ZAR')[1])
    assert 'JIBA3M Index' in zar['3M']['Descriptor']
    assert zar['3M']['DealType'] == 'DepositDeal'
    assert zar['3M']['Deal']['Accrual_Day_Count'] == 'ACT_365'
    assert zar['3M']['Deal']['Payment_Frequency'] == {'.DateOffset': '3M'}
    assert all('ZARONIA' not in row['Descriptor'] for row in zar.values())

    # USD declares the overnight one, and an O/N deposit is T+0 to the NEXT BUSINESS DAY - its own
    # payment frequency is that span, so the pinned schedule is one period rather than three
    usd = points_of(block_of('USD')[1])
    assert 'SOFRRATE Index' in usd['overnight']['Descriptor']
    assert usd['overnight']['Deal']['Effective_Date'] == {'.Timestamp': '2026-08-31'}
    assert usd['overnight']['Deal']['Maturity_Date'] == {'.Timestamp': '2026-09-01'}
    assert usd['overnight']['Deal']['Payment_Frequency'] == {'.DateOffset': '1D'}
    assert usd['overnight']['Deal']['Accrual_Day_Count'] == 'ACT_360'


def test_the_quote_is_never_authored_into_the_deal():
    """THE CAUSE OF THE TICK, gated as the property it is. `QUOTE_WRITERS` is where a number lands
    in an instrument, so every rate-carrying field the emitter writes is a NEUTRAL zero and the
    print rides in `Quoted_Market_Value` alone: the `Deal` half of a row is a function of the
    calendar and the conventions and of nothing that moves between prints.

    Author the quote in and the row would be structurally different at every tick, which
    `config.update_market_quote` would refuse by name and be right to.
    """
    usd, zar = points_of(block_of('USD')[1]), points_of(block_of('ZAR')[1])
    assert zar['1Y']['Deal']['Swap_Rate'] == 0.0
    assert zar['1Y']['Quoted_Market_Value'] == 7.62
    assert zar['3M']['Deal']['Interest_Rate_Schedule'] == {'.DateList': []}
    assert zar['3M']['Quoted_Market_Value'] == 7.41
    fixed = usd['2Y']['Deal']['Children'][1]
    assert {item['Rate']['.Percent'] for item in fixed['Cashflows']['Items']} == {0.0}
    assert usd['2Y']['Quoted_Market_Value'] == 3.88

    # and neither `Object` nor `Discount_Rate` is authored twice: the point NAMES the type and the
    # family stamps the discount curve from the block it belongs to
    for row in list(usd.values()) + list(zar.values()):
        assert 'Object' not in row['Deal'] and 'Discount_Rate' not in row['Deal']
        for child in row['Deal'].get('Children', ()):
            assert 'Discount_Rate' not in child


def test_the_two_way_and_the_stamp_ride_beside_the_mid():
    """`Quoted_Bid`, `Quoted_Ask` and `Timestamp` are `schema.MARKET_QUOTE_VALUES` - the plane a
    tick may move - and `InterestRateCurveParameters.Points` now DECLARES all three, on the shape
    `FXVolPrices` already declared them in: optional columns on the value side, read by nothing in
    the solve. This gate said the opposite while that was a finding (2026-09-01 closed it); what it
    holds is unchanged, because the emitter's bytes never depended on the declaration - the two-way
    and the print's own clock ARE the evidence, whether or not a schema had a column for them."""
    row = points_of(block_of('ZAR')[1])['1Y']
    assert (row['Quoted_Bid'], row['Quoted_Ask']) == (7.61, 7.63)
    assert row['Timestamp'] == {'.Timestamp': YESTERDAY}
    assert row['Use'] == 'Yes' and row['Quote_Type'] == 'Par_Rate'

    # a mid-only print carries no sides at all rather than a manufactured spread
    poison = dict(POISON)
    poison['SASW1 BGN Curncy'] = {'PX_BID': None, 'PX_ASK': None}
    lonely = points_of(block_of('ZAR', poison=poison)[1])['1Y']
    assert 'Quoted_Bid' not in lonely and 'Quoted_Ask' not in lonely
    assert lonely['Quoted_Market_Value'] == 7.62


def test_the_block_writes_only_fields_the_family_declares():
    """Every BLOCK-level key is a declared field of `InterestRateCurveParameters`, read off the
    COMMITTED declaration rather than off the working tree - the solve's own knobs (`N_Iter`, `Tol`,
    `Damping_Halvings`) and the three lifecycle switches are deliberately NOT written, because they
    are properties of a job rather than of a market and the engine reads each with its declared
    default.

    THE POINT KEYS ARE NOW A SUBSET TOO, and that half reads the WORKING TREE (`at=None`): the
    `Quoted_Bid`/`Quoted_Ask`/`Timestamp` columns are declared by this very change, so HEAD cannot
    be asked about them. Nothing else in this file reads the tree, and the claim itself is what
    changed rather than the emitter - this gate used to subtract the three by name, which is the
    shape of a gate documenting a gap.
    """
    declared = committed_fields('InterestRateCurveParameters')
    instrument = block_of('ZAR')[1]['instrument']
    assert set(instrument) <= set(declared), sorted(set(instrument) - set(declared))
    assert set(instrument) == {'Currency', 'Day_Count', 'Discount_Rate', 'Points'}
    assert instrument['Discount_Rate'] == '', 'V1 builds a self-discounting single curve'
    assert instrument['Day_Count'] == 'ACT_365'
    # the block key names the curve the strip BUILDS, which defaults to the currency and which a
    # multi-curve desk names itself - the deals project off exactly that name
    assert block_of('ZAR')[0] == 'InterestRatePrices.ZAR'
    named = ir_curve_block(strip_of('ZAR', curve='ZAR-JIBAR-3M'))
    assert named[0] == 'InterestRatePrices.ZAR-JIBAR-3M'
    assert {row['Deal'].get('Interest_Rate') for row in named[1]['instrument']['Points']} == {
        'ZAR-JIBAR-3M'}
    # and every POINT key is a declared sub-field - no undeclared extra rides beside the mid
    points = committed_fields('InterestRateCurveParameters', table='Points', at=None)
    assert set(points) == {'Use', 'Deal', 'Descriptor', 'DealType', 'Quote_Type',
                           'Quoted_Market_Value', 'Quoted_Bid', 'Quoted_Ask', 'Timestamp'}, points
    for row in instrument['Points']:
        assert set(row) <= set(points), sorted(set(row) - set(points))
        # the mid and the structure are on every row; the two-way and the stamp only where printed
        assert {'Use', 'Deal', 'Descriptor', 'DealType', 'Quote_Type',
                'Quoted_Market_Value'} <= set(row)


def committed_fields(class_name, table=None, at='HEAD'):
    """The field names one bootstrapper class declares, read off the COMMITTED `bootstrappers.py`
    via `git show HEAD` and parsed as an AST - never imported, and by default never off the working
    tree.

    Reading the committed state is what lets this file gate an engine declaration while another
    workflow is mid-edit in the same tree; parsing rather than importing is what lets it do so
    without the engine's own import cost. `table`, when given, descends into that field's
    `row=Row([...])` or `sub_fields=[...]` and returns the COLUMNS of the row instead.

    `at=None` reads the working tree, and there is exactly one thing it is for: a declaration this
    repository is LANDING. A gate on a field that does not exist at HEAD yet cannot be asked about
    HEAD, and a gate that reads the tree says so at its own call site.
    """
    tree = ast.parse(_committed('derivus/bootstrappers.py', at), filename='bootstrappers.py')
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                        getattr(target, 'id', None) == 'fields' for target in statement.targets):
                    return _field_names(statement.value, table)
    raise AssertionError('{} declares no `fields` in the committed bootstrappers'.format(class_name))


def _field_names(node, table):
    names = []
    for entry in node.elts:
        name = entry.args[0].value
        if table is None:
            names.append(name)
            continue
        if name != table:
            continue
        for keyword in entry.keywords:
            # a Table declares its columns as `row=Row([...])`, a Container its children as
            # `sub_fields=[...]` - two spellings of "the fields inside this one"
            if keyword.arg == 'row':
                return [row.args[0].value for row in keyword.value.args[0].elts]
            if keyword.arg == 'sub_fields':
                return [child.args[0].value for child in keyword.value.elts]
    if table is not None:
        raise AssertionError('no {} table in the committed declaration'.format(table))
    return names


def _committed(path, at='HEAD'):
    """One file as `at` has it, or as the WORKING TREE has it when `at` is None.

    Every schema comparison in this file goes through here rather than reading the tree, which is
    what lets these gates hold an ENGINE declaration while a parallel workflow is mid-edit in the
    same checkout. The tree read is for the one case that rule cannot cover - a declaration landing
    in this very change, which HEAD does not have yet.
    """
    if at is None:
        with open(os.path.join(ROOT, path), 'rt', encoding='utf-8') as source:
            return source.read()
    return subprocess.run(['git', 'show', '{}:{}'.format(at, path)], cwd=ROOT,
                          stdout=subprocess.PIPE, universal_newlines=True,
                          encoding='utf-8').stdout


#: `{deal type: declared JSON keys}`, parsed once - `git show` is a process per file.
_DEAL_FIELDS = {}


def committed_deal_fields(deal_type):
    """The JSON keys one INSTRUMENT type declares, off the committed `instruments.py` and
    `schema.py`, parsed as an AST - never imported, and never read off the working tree.

    `json_name` IS HONOURED WHERE A FIELD DECLARES ONE, and that is the whole subtlety of reading
    this declaration as data: both cashflow legs declare their container as `Fixed_Cashflows` /
    `Float_Cashflows` and write it to JSON as `Cashflows`, so a comparison made on declared names
    alone would report the emitter's own correct key as an undeclared extra. Group references
    (`ADMIN`, `CASHFLOWLISTDEAL`) are resolved by name out of `schema.py`, which is where the shared
    halves of a declaration live.
    """
    if not _DEAL_FIELDS:
        groups = {}
        for node in ast.parse(_committed('derivus/schema.py')).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and getattr(node.value.func, 'id', None) == 'Group':
                groups[node.targets[0].id] = _json_names(node.value.args[1])
        for node in ast.walk(ast.parse(_committed('derivus/instruments.py'))):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                        getattr(target, 'id', None) == 'fields' for target in statement.targets):
                    _DEAL_FIELDS[node.name] = _declared_keys(statement.value, groups)
    assert deal_type in _DEAL_FIELDS, \
        '{} declares no `fields` in the committed instruments'.format(deal_type)
    return _DEAL_FIELDS[deal_type]


def _declared_keys(node, groups):
    """One `fields = [ADMIN, own('X', [...])]` declaration flattened to the JSON keys it names."""
    names = set()
    for entry in node.elts:
        if isinstance(entry, ast.Name):
            names |= groups[entry.id]
        elif getattr(entry.func, 'id', None) == 'own':
            names |= _json_names(entry.args[1])
        else:
            names |= _json_names(ast.List(elts=[entry]))
    return names


def _json_names(node):
    names = set()
    for entry in node.elts:
        declared = {keyword.arg: keyword.value for keyword in entry.keywords}
        json_name = declared.get('json_name')
        names.add(json_name.value if json_name is not None else entry.args[0].value)
    return names


def test_every_authored_deal_key_is_one_the_committed_schema_declares():
    """`construct_instrument` VALIDATES NOTHING, which is why this gate exists rather than the
    compile gate below covering it.

    An unknown field on an authored deal is not an error to the engine - it is read past, and the
    instrument is built from the fields it did recognise. So a key spelled `Accrual_Daycount` would
    construct, reset, generate cashflows and produce a knot, and the whole seam gate would run green
    on a deal accruing under a default nobody chose. The only thing that can catch that is a
    comparison against the DECLARATION, so here it is, parsed as data off the committed schema.

    WHAT IS MISSING IS AS DECLARED AS WHAT IS EXTRA, and each absence has one reason:

      Object              named in `DealType` instead, on the top-level deal - the child legs DO
                          carry it, because they are deal-tree nodes rather than Points rows
      Tags, MtM           the ADMIN group: a position's own bookkeeping, not a benchmark's
      Discount_Rate       stamped by `author_quote`, which recurses into `Children`. What an
                          instrument PROJECTS off is its own business; what the quote set DISCOUNTS
                          on is a property of the curve set, and authoring it would state the same
                          thing twice with the second one going stale

    `Children` is the only authored key that is not a declared field, and it is not one: it is the
    deal TREE's own key. `StructuredDeal` declares `Currency` and `Net_Cashflows` and nothing else,
    and its legs hang off the tree rather than off its field list.
    """
    seen = set()
    for currency in ('USD', 'ZAR'):
        for row in block_of(currency)[1]['instrument']['Points']:
            nodes = [(row['DealType'], row['Deal'], False)]
            nodes += [(child['Object'], child, True) for child in row['Deal'].get('Children', ())]
            for deal_type, node, is_child in nodes:
                declared = committed_deal_fields(deal_type)
                extra = set(node) - declared
                assert sorted(extra) == (['Children'] if deal_type == 'StructuredDeal' else []), \
                    (currency, deal_type, sorted(extra))
                missing = declared - set(node)
                assert missing == ({'MtM', 'Tags'}
                                   | (set() if is_child else {'Object'})
                                   | ({'Discount_Rate'} if 'Discount_Rate' in declared else set())), \
                    (currency, deal_type, sorted(missing))
                seen.add(deal_type)
    assert seen == {'DepositDeal', 'SwapInterestDeal', 'StructuredDeal',
                    'CFFloatingInterestListDeal', 'CFFixedInterestListDeal'}, sorted(seen)


# =============================================================================================
# 5  the knot rule and determinism
# =============================================================================================

def test_two_benchmarks_on_one_maturity_refuse_by_name():
    """The knot rule is what makes the bootstrap SQUARE: one knot per used quote at that
    benchmark's last cashflow date. A seed quoting 4W beside 1M puts two instruments between the
    same pair of knots and leaves the curve under-determined between them - which reaches the solve
    as a singular Jacobian rather than as a sentence, so the emitter says it first."""
    seed = copy.deepcopy(SEED)
    # two spellings of the same three weeks: the seeded 3W point and a 21-day 'strip' entry
    seed['rates']['USD']['years'] = [1, 2]
    poison = {security: value for security, value in POISON.items()
              if security != 'USOSFR3Z BGN Curncy'}
    poison['USOSFR1 BGN Curncy'] = {}
    strip = strip_of('USD', seed=seed, poison=poison)
    twin = ir_curve.RatePrint(label='21D', kind='swap', security='TWIN Curncy', value=4.29,
                              bid=4.28, ask=4.30, last_update=YESTERDAY)
    doubled = ir_curve.CurveStrip(
        currency=strip.currency, curve=strip.curve, as_of=strip.as_of,
        conventions=strip.conventions, prints=strip.prints + (twin,), rejected=strip.rejected)
    with pytest.raises(IncompleteStrip) as refused:
        ir_curve_block(doubled)
    message = str(refused.value)
    assert 'both mature on' in message and 'USOSFR3Z' in message and 'TWIN' in message
    assert 'ONE knot per used quote' in message


def test_the_fixing_cap_names_the_arithmetic_rather_than_dying_on_it():
    """An OIS block grows with the SUM of its strip's tenors rather than with its point count - one
    authored item per business-day fixing - so the shipped USD strip is tens of thousands of items
    and tens of megabytes. The default admits that, because it is a block a real terminal produces;
    what the bound catches is a seed reaching further, and it refuses with the count."""
    with pytest.raises(IncompleteStrip) as refused:
        ir_curve_block(strip_of('USD'), CurveScreen(maximum_fixings=10))
    message = str(refused.value)
    assert 'daily fixings across its 2 OIS benchmarks, past the declared cap of 10' in message
    assert 'ONE ITEM PER BUSINESS-DAY FIXING' in message
    # the seeded strip itself is inside the shipped default, which is what makes the cap a bound
    # rather than a blockage
    assert ir_curve_block(strip_of('USD'))[1]['instrument']['Points']


def test_the_same_canned_strip_emits_the_same_bytes():
    """DETERMINISM, and the only clock in sight is the as-of, which is a parameter. Two emissions
    off the same canned answers are byte-identical - including the timestamps, which come off the
    prints rather than off a wall clock - so a block that changed is a market that moved."""
    first = json.dumps(block_of('ZAR')[1], sort_keys=True)
    assert first == json.dumps(block_of('ZAR')[1], sort_keys=True)

    moved = dict(POISON)
    moved['SASW3 BGN Curncy'] = {'PX_LAST': 8.30, 'PX_BID': 8.29, 'PX_ASK': 8.31}
    second = json.dumps(block_of('ZAR', poison=moved)[1], sort_keys=True)
    assert second != first
    assert '8.3' in second


# =============================================================================================
# 6  the engine seam - read-only
# =============================================================================================

def job_document(market_prices=None):
    """A wire-form job document with a `Market Prices` section - the shape
    `config.update_market_quote` writes into and `Config.read_json` reads."""
    return {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': {'.Timestamp': AS_OF.isoformat()},
                        'Currency': 'ZAR'},
        'Deals': {'Tag_Titles': '', 'Reference': 'strip', 'Deals': {'Children': []}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'ZAR',
                                  'Base_Date': {'.Timestamp': AS_OF.isoformat()}},
            'Price Factors': {}, 'Bootstrapper Configuration': {},
            'Market Prices': market_prices or {}}}}}


def test_the_block_installs_and_a_value_only_retick_updates():
    """`config.update_market_quote` is the contract every quote source posts against, and unlike the
    Heston-Nandi chain this block passes it BOTH WAYS - because `InterestRatePrices` quotes in
    `Points` rows, which is exactly what `schema.partition_market_price` gives a values half to.

    A moved RATE is a tick: the mid, the two sides and the stamp all live on the value plane and the
    `Deal` beside them does not move, which is what authoring the quote OUTSIDE the deal buys. A
    moved CONVENTION is a new plan and refuses by name, which is what it should be.
    """
    from derivus.config import update_market_quote

    name, block = block_of('ZAR')
    document = job_document()
    assert update_market_quote(document, name, block) == 'installed'
    prices = document['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices']
    assert prices[name] is block

    again = block_of('ZAR')[1]
    assert again == block and again is not block
    assert update_market_quote(document, name, again) == 'updated'

    # the market moves: a new rate, a new two-way, a fresh stamp, and NOTHING structural
    moved = dict(POISON)
    moved['SASW1 BGN Curncy'] = {'PX_LAST': 7.71, 'PX_BID': 7.70, 'PX_ASK': 7.72,
                                 'LAST_UPDATE_DT': AS_OF.isoformat()}
    reticked = block_of('ZAR', poison=moved)[1]
    assert reticked != block
    deals = lambda item: [row['Deal'] for row in item['instrument']['Points']]
    assert deals(reticked) == deals(block), 'a re-tick moved a deal, not a value'
    assert update_market_quote(document, name, reticked) == 'updated'

    # a moved CONVENTION is a re-authoring and refuses - the guard reading the plan half
    seed = copy.deepcopy(SEED)
    seed['rates']['ZAR']['conventions']['fixed_frequency'] = '6M'
    with pytest.raises(ValueError, match='structure differs'):
        update_market_quote(document, name, block_of('ZAR', seed=seed)[1])

    with pytest.raises(ValueError, match='a Market Prices block is'):
        update_market_quote(document, name, block['instrument'])


def test_a_rolled_date_strip_reaches_a_book_through_reauthor():
    """THE OTHER HALF OF THE ROUND TRIP, and the reason `ir_curve` has a `reauthor` of its own.

    The gate above is the tick: a moved RATE is a value and passes as 'updated'. This is the case
    that is not a tick and could never be one - TOMORROW'S STRIP. `Effective_Date` and
    `Maturity_Date` are structure, so the same benchmarks fetched a day later are a different plan
    and `update_market_quote` refuses by name. It is RIGHT to refuse: the guard cannot tell a rolled
    date from a mis-authored one, and that is exactly what it is for.

    So a next-day curve reaches the book the way a re-quoted swaption ladder does - dropped and
    re-installed. One function does both (`ir_curve.reauthor`, reached by `swaption_vol`), because
    the mechanism is one thing and only the reason differs: there the values half is EMPTY and no
    tick exists at all, here the values half works fine and the date is simply not in it.
    """
    from derivus.config import update_market_quote

    name, block = block_of('ZAR')
    document = job_document()
    assert update_market_quote(document, name, block) == 'installed'
    prices = document['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices']

    tomorrow = block_of('ZAR', as_of=AS_OF + datetime.timedelta(days=1))[1]
    dates = lambda item: {row['Deal']['Maturity_Date']['.Timestamp']
                          for row in item['instrument']['Points']}
    assert dates(tomorrow) != dates(block), 'the roll is what makes this a new plan'
    assert '2027-09-01' in dates(tomorrow) and '2027-08-31' in dates(block)
    with pytest.raises(ValueError, match='structure differs'):
        update_market_quote(document, name, tomorrow)

    assert reauthor(prices, name, tomorrow) == 'reauthored'
    assert prices[name] is tomorrow
    assert update_market_quote(document, name, tomorrow) == 'updated', \
        're-installed, so the guard now compares the rolled strip with itself'

    # a first fetch says so rather than reporting a re-authoring of nothing
    assert reauthor({}, name, block) == 'installed'
    with pytest.raises(BloombergConfigurationError, match='a Market Prices block is'):
        reauthor(prices, name, block['instrument'])

    # and it is REACHABLE off the package, which is what the swaption emitter's docstring always
    # claimed of it and the lazy re-export did not carry
    assert 'derivus_bloomberg' in in_a_fresh_interpreter(
        'from derivus_bloomberg import reauthor; assert callable(reauthor)')
    import derivus_bloomberg

    assert 'reauthor' in derivus_bloomberg.__all__ and 'reauthor' in dir(derivus_bloomberg)
    assert derivus_bloomberg.reauthor is reauthor


def test_a_broken_seed_is_a_configuration_refusal_and_never_a_no_terminal_skip():
    """A BROKEN WORKSTATION MUST NOT READ AS AN ABSENT ONE, which is a property of the CATCH SITES
    rather than of the taxonomy.

    `BloombergConfigurationError` hangs off `BloombergFXError` with everything else - one base, one
    taxonomy, and `errors.py` says why the name stays historical. The live smokes used to catch that
    BASE and skip 'no Bloomberg terminal answering', so a workstation whose `seed.json` had lost its
    `conventions` block would have reported an absent terminal while the terminal answered fine: the
    one failure a smoke gate exists to surface, disguised as the one thing it may skip for.

    The hierarchy is untouched (`derivus.service` and `derivus_mcp` catch the base and must keep
    working) and the smokes now catch `NO_TERMINAL`. This gate is what holds them to it: widen that
    tuple back and the last assertion goes red.
    """
    doctored = copy.deepcopy(SEED)
    del doctored['rates']['USD']['conventions']
    with pytest.raises(BloombergConfigurationError) as refused:
        strip_of('USD', seed=doctored)
    assert 'carries no `conventions` block' in str(refused.value)

    stripped = copy.deepcopy(SEED)
    del stripped['rates']['ZAR']['conventions']['front']
    with pytest.raises(BloombergConfigurationError, match='declares no front'):
        strip_of('ZAR', seed=stripped)

    # the taxonomy is UNCHANGED - which is exactly why the catch site had to be the thing that moved
    assert issubclass(BloombergConfigurationError, BloombergFXError)
    assert all(issubclass(error, BloombergFXError) for error in NO_TERMINAL)

    # ...and this is the property the two live smokes rely on
    assert not isinstance(refused.value, NO_TERMINAL), \
        'a seed refusal would be skipped as an absent terminal'
    assert BloombergConfigurationError not in NO_TERMINAL
    assert not any(issubclass(BloombergConfigurationError, error) for error in NO_TERMINAL)

    # and the two skips a smoke MAY make are named apart, because they are different events and a
    # desk does different things about them. This workstation meets the second one for real: its
    # terminal answers and refuses with `DAILY_CAPACITY_REACHED` once the daily data quota is spent,
    # which is a thing to wait out - not a workstation without a terminal
    absent = no_terminal_reason(BloombergUnavailable('no blpapi'))
    refused_request = no_terminal_reason(BloombergRequestError('DAILY_CAPACITY_REACHED'))
    assert absent.startswith('no Bloomberg terminal answering')
    assert 'answered and refused' in refused_request and 'DAILY_CAPACITY_REACHED' in refused_request
    assert 'no Bloomberg terminal answering' not in refused_request


def test_the_engine_builds_the_authored_deals_and_reads_their_knots():
    """READ-ONLY, AND NO SOLVE. The block is decoded by the engine's own JSON reader and every point
    is turned into a benchmark deal node by `bootstrappers.quote_nodes` - which authors the quote
    through `QUOTE_WRITERS`, constructs each instrument and recurses into `Children` - and then
    `quote_knots` resets each leaf and reads its last cashflow date.

    WHAT THIS PROVES, STATED NARROWLY. The authored blocks CONSTRUCT: every point becomes an
    instrument, every leg resets, every benchmark produces a last cashflow date, and the knot grid
    comes out ASCENDING and strictly positive - which is the curve contract
    (`Factor1D.interpolate` divides by the tenor, so a zero knot is NaN). The quote reaches the
    field the family's own writer puts it in.

    WHAT IT DOES NOT PROVE is that the authored field NAMES are declared ones. `construct_instrument`
    validates nothing: an unknown key is read past and the instrument is built from the rest, so a
    misspelled day count would reach this gate and pass it. That claim belongs to
    `test_every_authored_deal_key_is_one_the_committed_schema_declares`, which compares against the
    declaration instead. The fit-through belongs to the composition harness; this is the seam
    under it.
    """
    import pandas as pd
    from derivus.bootstrappers import quote_knots, quote_nodes
    from derivus.config import Config

    name, block = block_of('ZAR')
    usd_name, usd_block = block_of('USD')
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_strip_probe.json')
    try:
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(job_document({name: block, usd_name: usd_block}), handle)
        data = Config().read_json(path)
    finally:
        if os.path.isfile(path):
            os.remove(path)
    prices = data['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices']

    for market_price, currency, expected in ((name, 'ZAR', 3), (usd_name, 'USD', 3)):
        instrument = prices[market_price]['instrument']
        assert len(instrument['Points']) == expected
        nodes = quote_nodes(instrument['Points'], currency)
        knots = quote_knots(nodes, pd.Timestamp(AS_OF), instrument['Day_Count'], {})
        assert len(knots) == expected
        assert list(knots) == sorted(knots), knots
        assert all(knot > 0.0 for knot in knots), knots

    # the quote reached the instrument through the family's own writer, on the type that carries it
    zar = prices[name]['instrument']
    swap = [point for point in zar['Points'] if point['DealType'] == 'SwapInterestDeal'][0]
    node = quote_nodes([swap], 'ZAR')[0]
    assert float(node['Instrument'].field['Swap_Rate']) == pytest.approx(
        swap['Quoted_Market_Value'])


# =============================================================================================
# 7  live smoke
# =============================================================================================

def test_a_live_terminal_answers_the_strip_or_the_smoke_skips_by_name():
    """LIVE SMOKE, and a workstation with no terminal is a SKIP rather than a failure.

    WHAT IS ASSERTED IS THE ROUTE, not the market: that this workstation's own map and seed reach a
    strip, that every candidate comes back as a print or a NAMED refusal, and that whatever
    survives authors a block. The census is PRINTED and never asserted - a strip read out of hours
    screens differently from one read at noon, and a gate that failed on that would be measuring the
    clock.

    THE WORKSTATION SEED IS READ FOR ITS VOCABULARY AND NOT NECESSARILY FOR ITS CONVENTIONS. A desk
    whose `DV_HOME/seed.json` predates this build has the tickers but not the declarations, and this
    gate does not edit a desk's own file: it takes the conventions from the PACKAGED seed where the
    workstation's carries none, says so in the census, and leaves the file alone. Which currencies
    exist is still the workstation's map, so nothing is smuggled in.
    """
    import time

    from derivus_bloomberg import security_map
    from derivus_bloomberg.session import blpapi_module

    try:
        blpapi_module()
    except NO_TERMINAL as absent:
        pytest.skip('no Bloomberg SDK on this workstation: {}'.format(absent))
    provisioned = discover.provisioned()
    if provisioned is None:
        pytest.skip('this workstation has no security map - run `DV_Bloomberg discover` first')
    document = security_map.load(provisioned)
    seed_path = os.path.join(security_map.home(), 'seed.json')
    seed = json.load(open(seed_path, encoding='utf-8')) if os.path.isfile(seed_path) \
        else packaged_seed()

    currencies = [currency for currency in ('USD', 'ZAR')
                  if currency in document.get('blocks', {}).get('rates', {})
                  and currency in seed.get('rates', {})]
    if not currencies:
        pytest.skip('this workstation\'s map verified no USD or ZAR strip')
    borrowed = []
    for currency in currencies:
        if 'conventions' not in seed['rates'][currency]:
            seed['rates'][currency]['conventions'] = \
                packaged_seed()['rates'][currency]['conventions']
            borrowed.append(currency)
    print('\nconventions borrowed from the packaged seed for {}'.format(
        ', '.join(borrowed) or 'nothing - the workstation seed declares its own'))

    # NO_TERMINAL AND NOT `BloombergFXError`: a configuration or seed refusal is this gate's whole
    # point and must FAIL here, not skip green as an absent terminal - see
    # `test_a_broken_seed_is_a_configuration_refusal_and_never_a_no_terminal_skip`
    started = time.time()
    try:
        with BloombergSession(timeout_ms=60000, connect_timeout_ms=5000) as session:
            strips = [fetch_curve_strip(session, document, seed, currency,
                                        datetime.date.today()) for currency in currencies]
    except NO_TERMINAL as refused:
        pytest.skip(no_terminal_reason(refused))

    for strip in strips:
        asked = len(strip.prints) + len(strip.rejected)
        print('\n{} strip as at {}: {} asked, {} believed ({}), {} refused ({}) in {:.0f}s'.format(
            strip.currency, strip.as_of.isoformat(), asked, len(strip.prints),
            ', '.join(item.label for item in strip.prints) or 'nothing',
            len(strip.rejected),
            ', '.join('{} {}'.format(count, verdict)
                      for verdict, count in sorted(strip.census.items())) or 'nothing refused',
            time.time() - started))
        assert asked > 0, 'the map carried no candidate for {}'.format(strip.currency)
        if len(strip.prints) >= CurveScreen().minimum_points:
            name, block = ir_curve_block(strip)
            print('  {} -> {} points, {} bytes'.format(
                name, len(block['instrument']['Points']), len(json.dumps(block))))
