"""The verified swap strip as an `InterestRatePrices` block - `derivus_bloomberg.ir_curve`.

Only the last gate opens a socket, and it skips by name where this workstation has no terminal.
Everything else runs on ONE canned world - a seeded USD OIS strip and a seeded ZAR JIBAR strip,
walked through the package's real discovery grammar into a real verified map with a poison table of
dead prints - driven through the package's own reader and the real engine seams
(`schema.update_market_quote`, `bootstrappers.quote_nodes` / `quote_knots`), all imported READ-ONLY.

WHAT IS HELD:

  the budget       `ir_curve` imports the standard library, this package and a LAZY blpapi
  the declaration  the SHIPPED conventions as data; a currency with no `conventions` block refuses
                   naming every missing field at once; a convention nobody reads refuses too
  the selection    the documented prints and rejects BY NAME - unverified, invalid, unpriced,
                   off-market, crossed, undated, stale - one canned print per verdict
  the conventions  land on the authored `Deal` blocks as the seed declares: USD an OIS-compounded
                   `StructuredDeal` with one float item per fixing window, ZAR a quarterly
                   `SwapInterestDeal`, and the front point the one the seed named
  the partition    the OIS fixing windows tile EVERY coupon exactly, so the two legs accrue the same
                   span - including at a coupon boundary on a weekend, where the float leg used to
                   lose two days of a one-year accrual
  the quote        is NOT authored into the deal - every rate-carrying field is a neutral zero and
                   the print rides in `Quoted_Market_Value`, which is why a re-tick is a tick
  the knot rule    one knot per used quote at its last cashflow date; two benchmarks maturing on one
                   day refuse by name rather than reaching the solve as a singular Jacobian
  the definition   every emitted block, read back through `block_conventions` and `block_rows` and
                   authored again at its own date, is the SAME BYTES - nothing off the seed
  the grammar      a DECLARED row's shape comes off its own tenor, a held-out row drops its knot
                   and keeps its place, and a used row with no number refuses by name
  the round trip   `update_market_quote` installs and UPDATES a value-only re-tick; a moved
                   convention refuses as a new plan, and a ROLLED DATE goes through `reauthor`
  the fields       every authored deal key is a field the COMMITTED instrument schema declares -
                   which `construct_instrument` does NOT check, so nothing else would say so
  the compile      `quote_nodes` / `quote_knots` construct the deals and read their last cashflow
                   dates - no solve, no schema check
  determinism      the same canned answers emit the same bytes
  the taxonomy     a broken seed is a CONFIGURATION refusal and never a no-terminal skip
  live smoke       one real strip off this workstation's terminal, or a skip by name

NO MONKEYPATCHING: the canned terminal is a `BloombergSession` subclass whose event walk yields
rows, and the engine is imported and never touched.
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
                                      BloombergRequestError, BloombergUnavailable, IncompleteStrip,
                                      InvalidQuote)
from derivus_bloomberg.ir_curve import (CurveScreen, curve_conventions, fetch_curve_strip,
                                        ir_curve_block, reauthor, screen_strip)
from derivus_bloomberg.session import BloombergSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AS_OF = datetime.date(2026, 8, 31)        # a Monday
YESTERDAY = '2026-08-28'                  # the Friday before it - a live print
LONG_AGO = '2026-07-15'                   # past any sane stale bound

#: The transport failures, and nothing else - why the live smokes catch this tuple rather than
#: `BloombergFXError`. `BloombergConfigurationError` hangs off that base with everything else, so a
#: smoke catching the BASE reads a BROKEN SEED as an absent terminal and skips green. The hierarchy
#: is left alone; the CATCH SITES are narrowed.
NO_TERMINAL = (BloombergUnavailable, BloombergRequestError)


def no_terminal_reason(error):
    """Why a live smoke is skipping, NAMED FOR WHAT ACTUALLY HAPPENED.

    `BloombergUnavailable` is the absent terminal - no SDK, no session, nothing listening.
    `BloombergRequestError` is a terminal that ANSWERED AND REFUSED: a timeout, or this
    workstation's usual `DAILY_CAPACITY_REACHED`, which is a thing to wait out rather than a
    workstation without a terminal. A skip has to name the thing somebody would go and fix.
    """
    if isinstance(error, BloombergUnavailable):
        return 'no Bloomberg terminal answering on this workstation: {}'.format(error)
    return 'this workstation\'s terminal answered and refused the request ({}): {}'.format(
        type(error).__name__, error)

def ois(spot_days, day_count, frequency='1Y'):
    """One overnight-index curve's declaration: what it settles at, what its swap accrues on and
    how often that pays, over the eleven fields every OIS entry shares."""
    return {'curve_day_count': 'ACT_365', 'spot_days': spot_days, 'front': 'overnight',
            'front_day_count': day_count, 'compounding': 'OIS',
            'fixed_frequency': frequency, 'float_frequency': frequency,
            'fixed_day_count': day_count, 'float_day_count': day_count,
            'notional': 1000000.0, 'quote_scale': 1.0}


#: The SHIPPED declarations as data - the gate the owner checks, since a convention is market fact.
#: USD SOFR OIS settles T+2, annual/annual on ACT/360 against a compounded overnight index and the
#: six other OIS curves differ from it in those three facts alone; ZAR SASW settles same day,
#: quarterly/quarterly on ACT/365 against 3M JIBAR, whose own fixing - not ZARONIA - is the front of
#: a JIBAR curve; the ZARONIA OIS curve is a third, annual/annual on ACT/365 and quoted monthly to a
#: year and then at two, which is what its near split is for.
SHIPPED = {
    'USD': ois(2, 'ACT_360'),                # SOFR
    'EUR': ois(2, 'ACT_360'),                # ESTR
    'GBP': ois(0, 'ACT_365'),                # SONIA, dealt for same-day settlement
    'JPY': ois(2, 'ACT_365'),                # TONA
    'CHF': ois(2, 'ACT_360'),                # SARON
    'CAD': ois(0, 'ACT_365', '6M'),          # CORRA, semi-annual coupons
    'AUD': ois(1, 'ACT_365'),                # AONIA, T+1
    'ZAR': {'curve_day_count': 'ACT_365', 'spot_days': 0, 'front': 'fixings/3M',
            'front_day_count': 'ACT_365', 'compounding': 'None',
            'fixed_frequency': '3M', 'float_frequency': '3M',
            'fixed_day_count': 'ACT_365', 'float_day_count': 'ACT_365',
            'notional': 1000000.0, 'quote_scale': 1.0},
    'ZAR-ZARONIA': dict(ois(0, 'ACT_365'), near_interpolation='LinearRT', near_tenor='2Y'),
}

#: The benchmarks each seeded entry spells - the ladder a set-up authors, counted so that a seed
#: edit dropping a tenor is a diff here rather than a curve quietly short of a knot.
LADDERS = {'USD': 24, 'EUR': 24, 'GBP': 24, 'JPY': 21, 'CHF': 21, 'CAD': 21, 'AUD': 21,
           'ZAR': 21, 'ZAR-ZARONIA': 26, 'ZAR-ZARONIA-FWD': 24}


def packaged_seed():
    return json.load(open(os.path.join(ROOT, 'derivus_bloomberg', 'seed.json'), encoding='utf-8'))


#: The canned world's seed - the shipped vocabulary cut to three CURVES over two currencies and a
#: handful of tenors, with the SHIPPED conventions carried across unchanged. Between them the four
#: authored shapes are covered: a deposit and a term swap on ZAR, a FRA strip beside them, an
#: overnight front and an OIS swap on USD and ZARONIA, and the forward-starting swaps on the
#: fourth key - the family the ZARONIA entry declares and does not use.
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
                'fras': {'prefix': 'SAFR', 'expect': 'ZAR FRA', 'source': '',
                         'tenors': {'1Mx4M': '0AD', '6Mx9M': '0FI'}},
                'conventions': SHIPPED['ZAR']},
        'ZAR-ZARONIA': {'currency': 'ZAR', 'prefix': 'SAOIAA', 'expect': 'ZAR OIS', 'source': '',
                        'months': ['3M', '9M'], 'long_months': ['18M'], 'years': [3, 5],
                        'overnight': {'security': 'ZARONIA Index',
                                      'expect': 'South African Overnight'},
                        'conventions': SHIPPED['ZAR-ZARONIA']},
        'ZAR-ZARONIA-FWD': {'currency': 'ZAR',
                            'overnight': {'security': 'ZARONIA Index',
                                          'expect': 'South African Overnight'},
                            'forwards': {'prefix': 'SAFOM', 'expect': 'FW SWP(ZARONIA)',
                                         'source': '',
                                         'tenors': {'1M1M': 'AA', '6M1M': 'FA', '15M1M': 'OA'}},
                            'conventions': {
                                key: value for key, value in SHIPPED['ZAR-ZARONIA'].items()
                                if not key.startswith('near_')}},
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

#: What a clean point answers. Levels are invented and only have to be plausibly shaped.
CLEAN = {'SOFRRATE Index': 4.33, 'USOSFR1Z BGN Curncy': 4.31, 'USOSFR2Z BGN Curncy': 4.30,
         'USOSFR3Z BGN Curncy': 4.29, 'USOSFR1 BGN Curncy': 4.02, 'USOSFR2 BGN Curncy': 3.88,
         'USOSFR5 BGN Curncy': 3.79, 'ZARONIA Index': 7.02, 'JIBA1M Index': 7.28,
         'JIBA3M Index': 7.41, 'JIBA6M Index': 7.55, 'SASW1 BGN Curncy': 7.62,
         'SASW2 BGN Curncy': 7.94, 'SASW3 BGN Curncy': 8.21, 'SASW5 BGN Curncy': 8.55,
         'SASW10 BGN Curncy': 8.83, 'SAFR0AD Curncy': 7.35, 'SAFR0FI Curncy': 7.48,
         'SAOIAAC Curncy': 7.05, 'SAOIAAI Curncy': 7.12, 'SAOIAA1F Curncy': 7.24,
         'SAOIAA3 Curncy': 7.56, 'SAOIAA5 Curncy': 7.81, 'SAFOMAA Curncy': 7.04,
         'SAFOMFA Curncy': 7.18, 'SAFOMOA Curncy': 7.33}


# =============================================================================================
# the canned world
# =============================================================================================

class Walked(BloombergSession):
    """A session whose event walk is canned rows, so the emitters are driven through the package's
    REAL readers with no socket and no patching."""

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
    """A real security map, built by the package's own discovery off canned NAME answers rather than
    hand-built. A poisoned entry of `None` never verifies, which is how a dead ticker reaches the
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


def strip_of(key, seed=None, poison=None, screen=None, as_of=AS_OF, curve=None):
    # the poison table is resolved HERE and passed on explicitly, so the map and the session are
    # built from the SAME one - a cell dead in one half and live in the other gates nothing
    seed, poison = seed or SEED, POISON if poison is None else poison
    return fetch_curve_strip(canned_session(seed, poison), verified_map(seed, poison), seed,
                             key, as_of, curve=curve, screen=screen)


def block_of(key, holidays=(), **kwargs):
    return ir_curve_block(strip_of(key, **kwargs), holidays=holidays)


# =============================================================================================
# 1  the dependency budget
# =============================================================================================

def imported_names(source):
    """The top-level names a file imports, however deep the import sits. Relative imports are
    skipped - they resolve inside the package by construction."""
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
    """The standard library, this package's own modules, and NO ENGINE. The chain emitter's stricter
    no-pandas budget is deliberately NOT claimed: `ir_curve` reaches `discover` for its ticker
    grammar, `discover` reaches `security_map`, and that carries pandas. Reusing the grammar is
    worth a dependency the map layer already pays for."""
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
    """The source gate's answer proved a second way - the first trusts the parser and reads ONE
    FILE, while an import runs a package.

    Asymmetric on purpose: `derivus` and `blpapi` must not land, since the package has to import on
    a workstation that has never seen a terminal. `pandas` DOES land through the map layer and is
    asserted POSITIVELY - allowing it quietly would omit a budget rather than state one.

    The last two lines are why these emitters are re-exported LAZILY: an eager
    `from .ir_curve import ...` would land pandas on `import derivus_bloomberg.equity_chain` behind
    that module's back.
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
    """THE OWNER'S GATE. Every convention this package will author a seeded benchmark in, as data
    off the shipped seed, so a change to a market convention is a diff here rather than a number
    that moved inside a cashflow - and every seeded entry authors its own ladder from it.

    The three a ticker cannot tell you: USD OIS accrues ACT/360 on both legs where the curve's
    tenors are ACT/365, the ZAR front is the 3M JIBAR fixing rather than the ZARONIA print beside
    it - a JIBAR curve seeded with an overnight rate is a basis error nothing downstream reports -
    and the ZARONIA curve is quoted monthly to a year and then at two, the near split reaching the
    two-year knot rather than a tenor grid.
    """
    shipped = packaged_seed()['rates']
    for curve, declared in SHIPPED.items():
        assert shipped[curve]['conventions'] == declared, curve
        conventions = curve_conventions(packaged_seed(), curve)
        assert conventions.compounding == declared['compounding']
    assert curve_conventions(packaged_seed(), 'USD').front == 'overnight'
    assert curve_conventions(packaged_seed(), 'ZAR').front == 'fixings/3M'
    # THE SEED IS KEYED BY CURVE: two ZAR curves, each naming its own currency, and an entry
    # naming none is its own - which is what keeps USD, EUR and the rest where they were
    assert {curve: spec.get('currency', curve) for curve, spec in shipped.items()
            if spec.get('currency')} == {'ZAR-ZARONIA': 'ZAR', 'ZAR-ZARONIA-FWD': 'ZAR'}
    assert 'currency' not in shipped['USD'] and 'currency' not in shipped['ZAR']
    near = curve_conventions(packaged_seed(), 'ZAR-ZARONIA')
    assert (near.near_interpolation, near.near_tenor) == ('LinearRT', '2Y')
    assert curve_conventions(packaged_seed(), 'ZAR-ZARONIA-FWD').near_interpolation == ''
    # EVERY seeded entry AUTHORS, which is what makes a currency set-up-able at all: its own
    # vocabulary's rows, read off the seed and emitted at a flat quote, with no terminal anywhere
    seed = packaged_seed()
    for curve, points in LADDERS.items():
        rows = [ir_curve.RatePrint(label=row['tenor'], kind='', security=row['security'],
                                   value=7.0) for row in ir_curve.seeded_rows(seed, curve)]
        _, block = ir_curve.author_block(
            {'curve': curve, 'currency': shipped[curve].get('currency', curve),
             'conventions': curve_conventions(seed, curve), 'rows': rows}, AS_OF)
        assert len(block['instrument']['Points']) == points, curve


def test_the_interpolation_menu_cannot_drift_from_the_engines_declaration():
    """This package imports no engine module, so the near-end schemes it admits are re-spelled
    here - and a scheme the engine stopped declaring would otherwise reach a block and be silently
    read as `Linear` by `factor_interp_map.get`."""
    source = _committed('derivus/riskfactors.py', at=None)
    declared = next(line for line in source.splitlines()
                    if line.startswith('INTERPOLATION_METHODS'))
    assert list(ir_curve.INTERPOLATIONS) == list(
        ast.literal_eval(declared.split('=', 1)[1].strip()))


def test_a_currency_without_its_conventions_refuses_naming_every_missing_field():
    """A half-filled convention block authors an instrument nobody stated, so the refusal lists the
    WHOLE questionnaire at once rather than one terminal round trip at a time."""
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

    # a convention nobody reads is not APPLIED, which is worse than a missing one: it reads as a
    # declaration and does nothing
    seed = copy.deepcopy(SEED)
    seed['rates']['ZAR']['conventions']['payment_lag'] = 2
    with pytest.raises(BloombergConfigurationError, match='payment_lag'):
        curve_conventions(seed, 'ZAR')


def test_a_declaration_this_emitter_cannot_author_refuses_at_the_seed():
    """The ways a declaration can be wrong rather than absent, each refusing where a desk can fix it
    rather than inside a cashflow generator.

    The day-count refusal is the sharpest: an authored `Accrual_Year_Fraction` is used VERBATIM by
    `TensorCashFlows.float`, so a day count this module cannot compute must never be approximated.
    `ACT_365_ISDA` and `ACT_ACT_ICMA` are days/365 in the engine behind a `# TODO`.
    """
    for field, value, expected in (
            ('compounding', 'Flat', 'is not what a swap row can be'),
            ('float_day_count', 'ACT_ACT_ICMA', 'is not a float_day_count a seed may declare'),
            ('fixed_day_count', '_30_360', 'is not a fixed_day_count a seed may declare'),
            ('spot_days', -1, 'spot_days must be a whole number'),
            ('fixed_frequency', '1', 'is not a tenor this emitter can read'),
            # a zero-length coupon period would roll towards a maturity it never reaches
            ('float_frequency', '0M', 'has to be a positive period'),
            # half a near split is a scheme with no end, or an end with no scheme
            ('near_interpolation', 'LinearRT', 'a near scheme is a scheme AND where it stops'),
            ('near_tenor', '18M', 'a near scheme is a scheme AND where it stops')):
        seed = copy.deepcopy(SEED)
        seed['rates']['USD']['conventions'][field] = value
        with pytest.raises(BloombergConfigurationError, match=expected):
            curve_conventions(seed, 'USD')

    # a scheme the engine does not interpolate with refuses too, naming the menu
    seed = copy.deepcopy(SEED)
    seed['rates']['ZAR-ZARONIA']['conventions']['near_interpolation'] = 'Cubic'
    with pytest.raises(BloombergConfigurationError, match='HermiteRT'):
        curve_conventions(seed, 'ZAR-ZARONIA')


def test_a_front_the_seed_could_not_name_refuses_before_it_mis_authors_the_short_end():
    """`front` IS THE ONE REQUIRED CONVENTION WHOSE WRONG VALUE IS SILENTLY AUTHORABLE, which is why
    it is validated against the seed rather than against itself.

    Every other convention refuses on its own value. This one is a PATH INTO THE SEED, and a path
    that names nothing aims somewhere else: `front: 'strip/1Y'` matches the 1Y par swap's map path,
    `_front_label` finds no `fixings` and manufactures the label `overnight`, and `strip_dates`
    authors that 1Y par swap as a ONE-DAY DepositDeal. The block is well formed, the strip is short
    by a knot, and nothing downstream can tell.

    So the refusal names the field, the declared value and the whole admissible set - `_seeded_
    fronts`' own list rather than a second spelling of it.
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

    # both SHIPPED declarations pass - a validation nothing real survives is one nobody can use
    assert curve_conventions(packaged_seed(), 'USD').front == 'overnight'
    assert curve_conventions(packaged_seed(), 'ZAR').front == 'fixings/3M'


def test_an_unverified_front_point_refuses_and_offers_the_ones_the_seed_names():
    """The SECOND front refusal, a different failure. Above the seed could never have named that
    front; here it names one the seed CAN spell and the map has no verified entry for - a seeded
    fixing that went dead. The front seeds the short end, so it cannot be skipped into a shorter
    curve: it says so, and says which fronts the seed offers.
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


def test_both_leg_frequencies_are_read_on_every_swap_row():
    """EVERY SWAP ROW IS ONE `SwapInterestDeal` and the engine generates each leg on its own
    frequency, OIS rows included - so a declaration that differs is authored rather than refused.

    It used to be refused on an OIS declaration, because the fixing-list authoring rolled ONE
    schedule off `fixed_frequency` and hung both legs on it: declaring USD at `6M` emitted a
    byte-identical block, a convention validated and then read by nothing. The term authoring has
    no schedule of its own to disagree with.
    """
    for curve, fixed in (('ZAR', '3M'), ('USD', '1Y')):
        swapped = copy.deepcopy(SEED)
        swapped['rates'][curve]['conventions']['float_frequency'] = '6M'
        deal = read_of(block_of(curve, seed=swapped)[1], '2Y' if curve == 'USD' else '1Y')
        assert deal['Pay_Frequency'] == {'.DateOffset': fixed}
        assert deal['Receive_Frequency'] == {'.DateOffset': '6M'}
    assert curve_conventions(SEED, 'USD').float_frequency == '1Y'


# =============================================================================================
# 3  the selection
# =============================================================================================

def test_the_screen_classifies_off_the_terminals_own_answers():
    """The order of distrust, one canned print per verdict. `crossed` and `stale` are different
    instructions to a desk, and a screen checking the date first reports the second where the first
    is true."""
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

    # a mid-only print is BELIEVED: a two-way the terminal never quoted is not a spread, and the mid
    # is what a curve is built off
    assert screen_strip([rate(security='mid-only', bid=None, ask=None)], AS_OF)[1] == {}


def test_the_canned_strip_is_believed_by_census():
    """Every candidate accounted for either way - which is what makes a short strip legible. A
    candidate silently dropped is indistinguishable from one never asked about."""
    usd, zar = strip_of('USD'), strip_of('ZAR')
    assert [item.label for item in usd.prints] == ['ON', '1W', '2Y']
    assert usd.rejected == {
        'USOSFR3Z BGN Curncy': 'unverified', 'USOSFR2Z BGN Curncy': 'crossed',
        'USOSFR1 BGN Curncy': 'unpriced', 'USOSFR5 BGN Curncy': 'stale'}

    assert [item.label for item in zar.prints] == ['3M', '1Mx4M', '6Mx9M', '1Y', '3Y']
    assert zar.rejected == {
        'ZARONIA Index': 'not-a-benchmark', 'JIBA1M Index': 'not-a-benchmark',
        'JIBA6M Index': 'not-a-benchmark', 'SASW2 BGN Curncy': 'undated',
        'SASW5 BGN Curncy': 'invalid', 'SASW10 BGN Curncy': 'off-market'}
    # the ZARONIA print IS verified and live - refused for being the wrong INSTRUMENT on a JIBAR
    # curve, which is the declaration doing its job rather than the screen
    assert zar.census == {'not-a-benchmark': 3, 'undated': 1, 'invalid': 1, 'off-market': 1}

    # the census a report reads is the same numbers as data, so a caller with a screen and one with
    # a log see one account of the fetch
    counted = ir_curve.quote_census(usd)
    assert (counted['asked'], counted['believed']) == (7, 3)
    assert counted['securities'] == [item.security for item in usd.prints]
    assert counted['refused'] == usd.rejected and counted['curve'] == 'USD'


def test_a_strip_below_its_floor_refuses_naming_what_the_terminal_served():
    """One knot is a flat curve quoted once, so a strip that screened away says what it was asked and
    what came back - "no curve" with no census is not something a desk can act on."""
    poison = dict(POISON, **{security: {'PX_LAST': None} for security in
                             ('SASW1 BGN Curncy', 'SASW3 BGN Curncy',
                              'SAFR0AD Curncy', 'SAFR0FI Curncy')})
    with pytest.raises(IncompleteStrip) as refused:
        block_of('ZAR', poison=poison)
    message = str(refused.value)
    assert 'ZAR screened to 1 believed point against a floor of 2' in message
    assert '4 unpriced' in message and '3 not-a-benchmark' in message


# =============================================================================================
# 4  the conventions on the authored deals
# =============================================================================================

def points_of(block):
    return {row['Tenor']: row for row in block['instrument']['Points']}


def read_of(block, tenor):
    """One row's deal as the ENGINE reads it: what the block STATES over what its own declarations
    already say. A block states its terms and only the conventions that differ from the schema's,
    so a gate on a convention reads it here - `ir_curve.DECLARED` is the other half, and
    `test_a_convention_the_block_leaves_unsaid_reads_as_the_declaration` holds it to the engine."""
    return dict(ir_curve.DECLARED, **points_of(block)[tenor]['Deal'])


def test_an_ois_row_is_a_term_swap_carrying_the_compounding_rule():
    """AN OIS BENCHMARK IS ONE `SwapInterestDeal`, `Compounding_Method` OIS, with `Index_Tenor` and
    `Receive_Interest_Frequency` at zero months - ONE RESET SPANNING EACH COUPON.

    It used to be a `StructuredDeal` over a floating cashflow LIST carrying one item per
    business-day fixing, because `pv_float_cashflow_list` compounds geometrically only where the
    reset count differs from the cashflow count. That spelling is retired: at t0 the compounded
    forwards read off a curve telescope to the period forward, so the term swap prices the same par
    rate (`test_a_term_ois_benchmark_prices_the_fixing_list_it_replaces` measures it), and a list
    authored one item per COUPON - the shape a generated leg has - pays one over n of the interest.

    THIS CANNED USD BLOCK IS 290,967 BYTES ON MAIN AND 4,470 HERE, the same three benchmarks with
    531 cashflow items gone; the shipped thirty-year strip was some 26,000 items and 14 MB.
    """
    row = points_of(block_of('USD')[1])['2Y']
    assert row['DealType'] == 'SwapInterestDeal'
    deal = row['Deal']
    assert 'Children' not in deal
    assert deal['Compounding_Method'] == 'OIS'
    assert deal['Index_Tenor'] == {'.DateOffset': '0M'}
    assert deal['Receive_Interest_Frequency'] == {'.DateOffset': '0M'}
    assert deal['Interest_Rate'] == 'USD'
    assert deal['Pay_Frequency'] == deal['Receive_Frequency'] == {'.DateOffset': '1Y'}
    assert deal['Pay_Day_Count'] == deal['Receive_Day_Count'] == 'ACT_360'
    assert deal['Principal'] == 1000000.0
    # T+2 on a Monday as-of, and the 2Y maturity rolls off the weekend it lands on
    assert deal['Effective_Date'] == {'.Timestamp': '2026-09-02'}
    assert deal['Maturity_Date'] == {'.Timestamp': '2028-09-04'}

    # a whole USD block is now small enough to read - the fixing list was 25,700 items shipped
    assert len(json.dumps(block_of('USD')[1])) < 20000


def test_a_fra_row_is_a_fradeal_spanning_the_two_tenors_its_label_names():
    """`1Mx4M` IS THE INSTRUMENT: both dates measured off spot, each rolled Modified Following, and
    the reset AT the effective date - the forward the curve carries over that window, which is what
    the solve holds at par. `FRA_Rate` is authored at zero; `QUOTE_WRITERS['FRADeal']` puts the
    print in."""
    row = points_of(block_of('ZAR')[1])['6Mx9M']
    assert row['DealType'] == 'FRADeal'
    # 2027-02-28 is a Sunday AND the end of February, so Modified Following goes BACK to the 26th
    assert row['Deal']['Effective_Date'] == {'.Timestamp': '2027-02-26'}
    assert row['Deal']['Maturity_Date'] == {'.Timestamp': '2027-05-31'}
    deal = read_of(block_of('ZAR')[1], '1Mx4M')
    assert deal['Effective_Date'] == {'.Timestamp': '2026-09-30'}
    assert deal['Maturity_Date'] == {'.Timestamp': '2026-12-31'}
    assert deal['Reset_Date'] == deal['Effective_Date']
    assert deal['Day_Count'] == 'ACT_365' and deal['Principal'] == 1000000.0
    assert deal['FRA_Rate'] == 0.0 and points_of(block_of('ZAR')[1])['1Mx4M'][
        'Quoted_Market_Value'] == 7.35
    assert deal['Borrower_Lender'] == 'Borrower' and deal['Payment_Timing'] == 'End'
    assert deal['Interest_Rate'] == 'ZAR'

    # the end is measured off SPOT, not off the rolled start: 9M from 2026-08-31 is 2027-05-31, and
    # walking three months off the rolled 2027-02-26 would have said the 26th of May instead
    assert ir_curve.strip_dates('6Mx9M', 'fra', AS_OF, curve_conventions(SEED, 'ZAR')) == (
        datetime.date(2027, 2, 26), datetime.date(2027, 5, 31))


def test_a_forward_starting_row_starts_at_spot_plus_its_first_tenor():
    """`6M1M` is a swap EFFECTIVE at spot + 6M running one month on - the MPC-dated shape a ZARONIA
    curve is quoted in inside 18 months, and the one row whose maturity is measured off its own
    effective date rather than off spot."""
    row = points_of(block_of('ZAR-ZARONIA-FWD')[1])['6M1M']
    assert row['DealType'] == 'SwapInterestDeal'
    deal = row['Deal']
    assert deal['Effective_Date'] == {'.Timestamp': '2027-02-26'}
    assert deal['Maturity_Date'] == {'.Timestamp': '2027-03-29'}
    assert deal['Compounding_Method'] == 'OIS'
    assert deal['Index_Tenor'] == {'.DateOffset': '0M'}
    # one coupon: the leg frequency is annual and the swap is a month long, so the engine's own
    # backward roll puts a single period between the two dates
    assert deal['Pay_Frequency'] == {'.DateOffset': '1Y'}
    assert row['Security'] == 'SAFOMFA Curncy'

    # the END is measured off the UNROLLED effective, so a start rolled back three days does not
    # shorten the swap: 2027-02-28 plus a month is 2027-03-28, a Sunday, rolling on to the 29th
    assert ir_curve.strip_dates('15M1M', 'forward', AS_OF,
                                curve_conventions(SEED, 'ZAR-ZARONIA-FWD')) == (
        datetime.date(2027, 11, 30), datetime.date(2027, 12, 30))


def test_the_roll_moves_a_date_onto_a_business_day_and_stays_in_its_month():
    """ONE CALENDAR RULE, and the two cases that separate Following from Modified Following.

    A Saturday maturity rolls FORWARD to Monday; a month-end that would roll into the next month
    rolls BACK instead, which is what keeps a monthly strip's knots in the months they are quoted
    for. With no holidays handed in the rule is Monday to Friday, which is what this module has
    always applied; a holiday set moves a date the weekday rule leaves alone.
    """
    saturday, sunday = datetime.date(2028, 9, 2), datetime.date(2029, 9, 2)
    assert ir_curve.roll(saturday) == datetime.date(2028, 9, 4)
    assert ir_curve.roll(saturday, modified=True) == datetime.date(2028, 9, 4)
    assert ir_curve.roll(sunday, modified=True) == datetime.date(2029, 9, 3)

    # 2026-05-31 is a Sunday and the last day of May: Following leaves May, Modified goes back
    month_end = datetime.date(2026, 5, 31)
    assert ir_curve.roll(month_end) == datetime.date(2026, 6, 1)
    assert ir_curve.roll(month_end, modified=True) == datetime.date(2026, 5, 29)

    # a business day is not moved, with or without a calendar
    assert ir_curve.roll(datetime.date(2026, 9, 2)) == datetime.date(2026, 9, 2)
    holidays = {datetime.date(2026, 9, 2), datetime.date(2026, 9, 3)}
    assert ir_curve.roll(datetime.date(2026, 9, 2), holidays) == datetime.date(2026, 9, 4)
    assert ir_curve.roll(datetime.date(2026, 9, 30), holidays={datetime.date(2026, 9, 30)},
                         modified=True) == datetime.date(2026, 9, 29)

    # and it reaches the authored dates: a ZAR holiday on the 1Y maturity moves that knot alone
    plain = points_of(block_of('ZAR')[1])
    holiday = points_of(block_of('ZAR', holidays={datetime.date(2027, 8, 31)})[1])
    assert plain['1Y']['Deal']['Maturity_Date'] == {'.Timestamp': '2027-08-31'}
    assert holiday['1Y']['Deal']['Maturity_Date'] == {'.Timestamp': '2027-08-30'}
    assert holiday['3Y']['Deal']['Maturity_Date'] == plain['3Y']['Deal']['Maturity_Date']
    # spot itself moves where the holiday is the as-of, which is the settlement lag reading the
    # same calendar as the maturity
    assert ir_curve.strip_dates('1Y', 'swap', AS_OF, curve_conventions(SEED, 'ZAR'),
                                {AS_OF})[0] == datetime.date(2026, 9, 1)


def test_the_jibar_strip_is_a_vanilla_swap_on_its_declared_conventions():
    """ZAR declares `None` compounding, so its swap rows carry a TERM index: quarterly against
    quarterly on ACT/365, with `Index_Tenor` zero months so each coupon carries one reset spanning
    its own accrual - which for a quarterly leg IS 3M JIBAR."""
    row = points_of(block_of('ZAR')[1])['1Y']
    assert row['DealType'] == 'SwapInterestDeal'
    deal = read_of(block_of('ZAR')[1], '1Y')
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
    """A basis error that would otherwise be free. The ZAR map carries a live ZARONIA print AND a
    live 3M JIBAR fixing; the seed says which is a JIBAR curve's front, and the emitter reads that
    rather than taking the overnight one because it is shorter."""
    zar = points_of(block_of('ZAR')[1])
    assert 'JIBA3M Index' in zar['3M']['Descriptor']
    assert zar['3M']['DealType'] == 'DepositDeal'
    assert read_of(block_of('ZAR')[1], '3M')['Accrual_Day_Count'] == 'ACT_365'
    assert read_of(block_of('ZAR')[1], '3M')['Payment_Frequency'] == {'.DateOffset': '3M'}
    assert all('ZARONIA' not in row['Descriptor'] for row in zar.values())

    # USD declares the overnight one: an O/N deposit is T+0 to the NEXT BUSINESS DAY, so its payment
    # frequency is that span and the pinned schedule is one period
    usd = points_of(block_of('USD')[1])
    assert 'SOFRRATE Index' in usd['ON']['Descriptor'] and usd['ON']['Security'] == 'SOFRRATE Index'
    assert usd['ON']['Deal']['Effective_Date'] == {'.Timestamp': '2026-08-31'}
    assert usd['ON']['Deal']['Maturity_Date'] == {'.Timestamp': '2026-09-01'}
    assert usd['ON']['Deal']['Payment_Frequency'] == {'.DateOffset': '1D'}
    assert usd['ON']['Deal']['Accrual_Day_Count'] == 'ACT_360'

    # a Friday as-of spans the weekend, so the pinned schedule is ONE 3D period, not three of one
    friday = points_of(block_of('USD', as_of=datetime.date(2026, 8, 28))[1])
    assert friday['ON']['Deal']['Payment_Frequency'] == {'.DateOffset': '3D'}
    assert friday['ON']['Deal']['Maturity_Date'] == {'.Timestamp': '2026-08-31'}


def test_the_quote_is_never_authored_into_the_deal():
    """THE CAUSE OF THE TICK. `QUOTE_WRITERS` is where a number lands in an instrument, so every
    rate-carrying field the emitter writes is a NEUTRAL zero and the print rides in
    `Quoted_Market_Value` alone: a row's `Deal` half is a function of the calendar and the
    conventions and of nothing that moves between prints. Author the quote in and every tick is
    structurally different, which `update_market_quote` refuses by name and is right to.
    """
    usd, zar = points_of(block_of('USD')[1]), points_of(block_of('ZAR')[1])
    assert zar['1Y']['Deal']['Swap_Rate'] == 0.0
    assert zar['1Y']['Quoted_Market_Value'] == 7.62
    assert zar['3M']['Deal']['Interest_Rate_Schedule'] == {'.DateList': []}
    assert zar['3M']['Quoted_Market_Value'] == 7.41
    assert zar['1Mx4M']['Deal']['FRA_Rate'] == 0.0
    assert zar['1Mx4M']['Quoted_Market_Value'] == 7.35
    assert usd['2Y']['Deal']['Swap_Rate'] == 0.0
    assert usd['2Y']['Quoted_Market_Value'] == 3.88

    # neither `Object` nor `Discount_Rate` is authored twice: the point NAMES the type and the
    # family stamps the discount curve from the block it belongs to
    for row in list(usd.values()) + list(zar.values()):
        assert 'Object' not in row['Deal'] and 'Discount_Rate' not in row['Deal']


def test_the_two_way_and_the_stamp_ride_beside_the_mid():
    """`Quoted_Bid`, `Quoted_Ask` and `Timestamp` are `schema.MARKET_QUOTE_VALUES` - the plane a
    tick may move - and `InterestRateCurveParameters.Points` declares all three as optional columns
    on the value side, read by nothing in the solve."""
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
    declaration. The solve's knobs (`N_Iter`, `Tol`, `Damping_Halvings`) and the three lifecycle
    switches are deliberately NOT written - properties of a job rather than of a market, each read
    by the engine with its declared default.

    THE CONVENTIONS ARE. The block is the curve's definition, so the calendar, the settlement lag,
    both legs' frequency and day count, the front's day count, the compounding rule and the near
    split are emitted beside the rows - and each row carries the `Tenor` it was authored from and
    the `Security` it was quoted off.

    Read off the WORKING TREE (`at=None`), the declarations being this change's own.
    """
    declared = committed_fields('InterestRateCurveParameters', at=None)
    instrument = block_of('ZAR')[1]['instrument']
    assert set(instrument) <= set(declared), sorted(set(instrument) - set(declared))
    assert set(instrument) == {'Currency', 'Day_Count', 'Discount_Rate', 'Calendar', 'Spot_Days',
                               'Fixed_Frequency', 'Float_Frequency', 'Fixed_Day_Count',
                               'Float_Day_Count', 'Front_Day_Count', 'Compounding',
                               'Near_Interpolation', 'Near_Tenor', 'Points'}
    assert instrument['Discount_Rate'] == '', 'the emitter builds a self-discounting single curve'
    assert (instrument['Day_Count'], instrument['Front_Day_Count']) == ('ACT_365', 'ACT_365')
    assert (instrument['Calendar'], instrument['Spot_Days']) == ('', 0)
    assert instrument['Fixed_Frequency'] == instrument['Float_Frequency'] == {'.DateOffset': '3M'}
    assert instrument['Compounding'] == 'None'
    assert (instrument['Near_Interpolation'], instrument['Near_Tenor']) == ('', '')

    # a curve quoted monthly at the front declares the split it carries, in the block
    zaronia = block_of('ZAR-ZARONIA')[1]['instrument']
    assert zaronia['Near_Interpolation'] == 'LinearRT'
    assert zaronia['Near_Tenor'] == {'.DateOffset': '2Y'}
    assert zaronia['Compounding'] == 'OIS' and zaronia['Currency'] == 'ZAR'

    # the block key names the curve the strip BUILDS - the seed's own key, or a name the caller
    # gives - and the deals project off exactly that name
    assert block_of('ZAR')[0] == 'InterestRatePrices.ZAR'
    assert block_of('ZAR-ZARONIA')[0] == 'InterestRatePrices.ZAR-ZARONIA'
    named = ir_curve_block(strip_of('ZAR', curve='ZAR-JIBAR-3M'))
    assert named[0] == 'InterestRatePrices.ZAR-JIBAR-3M'
    assert {row['Deal'].get('Interest_Rate') for row in named[1]['instrument']['Points']} == {
        'ZAR-JIBAR-3M'}
    # and every POINT key is a declared sub-field - no undeclared extra rides beside the mid
    points = committed_fields('InterestRateCurveParameters', table='Points', at=None)
    assert set(points) == {'Use', 'Deal', 'Descriptor', 'Tenor', 'Security', 'DealType',
                           'Quote_Type', 'Quoted_Market_Value', 'Quoted_Bid', 'Quoted_Ask',
                           'Timestamp'}, points
    for row in instrument['Points']:
        assert set(row) <= set(points), sorted(set(row) - set(points))
        # the mid and the structure are on every row; the two-way and the stamp only where printed
        assert {'Use', 'Deal', 'Descriptor', 'Tenor', 'Security', 'DealType', 'Quote_Type',
                'Quoted_Market_Value'} <= set(row)


def committed_fields(class_name, table=None, at='HEAD'):
    """The field names one bootstrapper class declares, read off the COMMITTED `bootstrappers.py`
    via `git show HEAD` and parsed as an AST - never imported.

    Reading the committed state lets this file gate an engine declaration while another workflow is
    mid-edit in the same tree; parsing rather than importing avoids the engine's import cost.
    `table` descends into that field's `row=Row([...])` or `sub_fields=[...]` and returns its
    COLUMNS. `at=None` reads the working tree, for the one case HEAD cannot answer: a declaration
    not yet committed.
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
            # a Table declares columns as `row=Row([...])`, a Container children as `sub_fields=`
            if keyword.arg == 'row':
                return [row.args[0].value for row in keyword.value.args[0].elts]
            if keyword.arg == 'sub_fields':
                return [child.args[0].value for child in keyword.value.elts]
    if table is not None:
        raise AssertionError('no {} table in the committed declaration'.format(table))
    return names


def _committed(path, at='HEAD'):
    """One file as `at` has it, or as the WORKING TREE has it when `at` is None - which is for the
    one case the committed read cannot cover, a declaration not yet committed.
    """
    if at is None:
        with open(os.path.join(ROOT, path), 'rt', encoding='utf-8') as source:
            return source.read()
    return subprocess.run(['git', 'show', '{}:{}'.format(at, path)], cwd=ROOT,
                          stdout=subprocess.PIPE, universal_newlines=True,
                          encoding='utf-8').stdout


#: `{deal type: {declared JSON key: convention}}`, parsed once.
_DEAL_FIELDS = {}


def declared_deal_fields(deal_type):
    """`{JSON key: whether its default is a CONVENTION}` for one INSTRUMENT type, read off
    `instruments.py` and `schema.py` as an AST - never imported, which is the point: the emitter is
    compared against the DECLARATION rather than against the engine's own reading of it.

    `json_name` IS HONOURED where a field declares one: both cashflow legs declare their container
    as `Fixed_Cashflows` / `Float_Cashflows` and write it as `Cashflows`, so a comparison on
    declared names alone would report the emitter's correct key as an undeclared extra. Group
    references (`ADMIN`, `CASHFLOWLISTDEAL`) are resolved by name out of `schema.py`.
    """
    if not _DEAL_FIELDS:
        groups = {}
        for node in ast.parse(_committed('derivus/schema.py', None)).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and getattr(node.value.func, 'id', None) == 'Group':
                groups[node.targets[0].id] = _json_names(node.value.args[1])
        for node in ast.walk(ast.parse(_committed('derivus/instruments.py', None))):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                        getattr(target, 'id', None) == 'fields' for target in statement.targets):
                    _DEAL_FIELDS[node.name] = _declared_keys(statement.value, groups)
    assert deal_type in _DEAL_FIELDS, \
        '{} declares no `fields` in the instruments'.format(deal_type)
    return _DEAL_FIELDS[deal_type]


def _declared_keys(node, groups):
    """One `fields = [ADMIN, own('X', [...])]` declaration flattened to `{JSON key: convention}`."""
    names = {}
    for entry in node.elts:
        if isinstance(entry, ast.Name):
            names.update(groups[entry.id])
        elif getattr(entry.func, 'id', None) == 'own':
            names.update(_json_names(entry.args[1]))
        else:
            names.update(_json_names(ast.List(elts=[entry])))
    return names


def _json_names(node):
    """`{JSON key: whether the declaration calls its default a convention}` for one field list."""
    names = {}
    for entry in node.elts:
        declared = {keyword.arg: keyword.value for keyword in entry.keywords}
        json_name = declared.get('json_name')
        key = json_name.value if json_name is not None else entry.args[0].value
        names[key] = getattr(declared.get('convention'), 'value', False) is True
    return names


def test_every_authored_deal_key_is_one_the_committed_schema_declares():
    """`construct_instrument` VALIDATES NOTHING, which is why this gate exists rather than the
    compile gate below covering it: an unknown field is read past and the instrument is built from
    the rest, so a key spelled `Accrual_Daycount` would construct, reset, generate cashflows and
    produce a knot under a default nobody chose. Only a comparison against the DECLARATION catches
    that.

    WHAT IS MISSING IS AS DECLARED AS WHAT IS EXTRA - three things the block never carries, and
    otherwise only fields the COMMITTED schema calls conventions:

      Object              named in `DealType` instead, on the top-level deal - the child legs DO
                          carry it, being deal-tree nodes rather than Points rows
      Discount_Rate       stamped by `author_quote`. What an instrument PROJECTS off is its own
                          business; what the quote set DISCOUNTS on is the curve set's
      a convention        the declaration already says it, so the block states only what DIFFERS -
                          the ADMIN bookkeeping a benchmark has none of, the null calendars, the
                          3M frequencies a quarterly strip does not restate. A PLACEHOLDER can
                          never be missing here: the terms, the dates and the amounts are what a
                          benchmark IS.

    Every one of the three authored types is covered, which is what the closing set says.
    """
    seen = set()
    for curve in ('USD', 'ZAR', 'ZAR-ZARONIA-FWD'):
        for row in block_of(curve)[1]['instrument']['Points']:
            deal_type, node = row['DealType'], row['Deal']
            declared = declared_deal_fields(deal_type)
            assert not set(node) - set(declared), (
                curve, deal_type, sorted(set(node) - set(declared)))
            stated = [key for key in set(declared) - set(node)
                      if key not in ('Object', 'Discount_Rate') and not declared[key]]
            assert not stated, (curve, deal_type, stated)
            seen.add(deal_type)
    assert seen == {'DepositDeal', 'FRADeal', 'SwapInterestDeal'}, sorted(seen)


# =============================================================================================
# 5  the knot rule and determinism
# =============================================================================================

def test_two_benchmarks_on_one_maturity_refuse_by_name():
    """The knot rule makes the bootstrap SQUARE: one knot per used quote at that benchmark's last
    cashflow date. Two instruments between one pair of knots leave the curve under-determined, which
    reaches the solve as a singular Jacobian rather than a sentence - so the emitter says it first."""
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


def test_the_knots_of_every_shape_are_the_maturities_the_labels_name():
    """ONE KNOT PER USED QUOTE at its own last cashflow, for all four shapes - a deposit, a FRA, a
    term swap and a forward-starting one - and the block comes out in maturity order, so the grid
    the family builds is ascending without anything sorting it afterwards.

    A FRA and a forward start are the two whose knot is NOT `spot + label`: each lands at the end of
    its own window, which is what puts a monthly forward strip's knots a month apart."""
    zar = block_of('ZAR')[1]['instrument']['Points']
    assert [row['Tenor'] for row in zar] == ['3M', '1Mx4M', '6Mx9M', '1Y', '3Y']
    assert [row['Deal']['Maturity_Date']['.Timestamp'] for row in zar] == [
        '2026-11-30', '2026-12-31', '2027-05-31', '2027-08-31', '2029-08-31']
    forwards = block_of('ZAR-ZARONIA-FWD')[1]['instrument']['Points']
    assert [row['Tenor'] for row in forwards] == ['ON', '1M1M', '6M1M', '15M1M']
    assert [row['Deal']['Maturity_Date']['.Timestamp'] for row in forwards] == [
        '2026-09-01', '2026-10-30', '2027-03-29', '2027-12-30']


def test_the_same_canned_strip_emits_the_same_bytes():
    """DETERMINISM: the only clock is the as-of, which is a parameter. Two emissions off the same
    canned answers are byte-identical, timestamps included (they come off the prints), so a block
    that changed is a market that moved."""
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

#: The canned world's curves and how many points each screens to - what the engine section runs
#: over, so a shape that stopped being authored fails a count rather than passing unexercised.
CURVES = {'USD': 3, 'ZAR': 5, 'ZAR-ZARONIA': 6, 'ZAR-ZARONIA-FWD': 4}


def decoded_blocks(curves, restated=False):
    """Every named curve's block through the ENGINE'S OWN JSON reader - the wire timestamps,
    periods and percents turned into what a pricer indexes.

    `restated` writes every convention back onto each row's deal before decoding, which is the
    block the emitter used to author - the one a completed read has to agree with."""
    from derivus.config import Config
    blocks = dict(block_of(curve) for curve in curves)
    if restated:
        for block in blocks.values():
            for row in block['instrument']['Points']:
                declared = declared_deal_fields(row['DealType'])
                row['Deal'] = dict({key: value for key, value in ir_curve.DECLARED.items()
                                    if key in declared}, **row['Deal'])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_strip_probe.json')
    try:
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(job_document(blocks), handle)
        data = Config().read_json(path)
    finally:
        if os.path.isfile(path):
            os.remove(path)
    return data['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices']


def job_document(market_prices=None):
    """A wire-form job document with a `Market Prices` section - what `update_market_quote` writes
    into and `Config.read_json` reads."""
    return {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': {'.Timestamp': AS_OF.isoformat()},
                        'Currency': 'ZAR'},
        'Deals': {'Tag_Titles': '', 'Reference': 'strip', 'Deals': {'Children': []}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'ZAR',
                                  'Base_Date': {'.Timestamp': AS_OF.isoformat()}},
            'Price Factors': {}, 'Bootstrapper Configuration': {},
            'Market Prices': market_prices or {}}}}}


def test_a_convention_the_block_leaves_unsaid_reads_as_the_declaration():
    """THE EMITTER'S RULE, held against the engine it cannot import. A benchmark is a deal, so it
    states its terms and only the conventions that DIFFER from the declaration - and `ir_curve` has
    no engine to ask, so `DECLARED` is its own spelling of what an omitted key already means.

    Held by READING, not by comparing spellings: every row is constructed twice, once off the block
    as emitted and once with every convention written back on, and the two deals answer every key
    the type declares the same. A wrong row in `DECLARED` drops a key whose meaning moved, and one
    of the two reads changes. Each dropped key is also checked to be a field the schema declares a
    convention, so nothing an author must state can be folded away here.

    THE SAME means as a READER sees it - a scaled rate by its amount, a table by its rows - and
    equal or both NOTHING. This module writes `None` where a declaration writes `''`, an empty
    table or a zero percent: an unstated calendar reaches `calendars.get(...)` as either, and
    `TensorCashFlows.periods` reads `amort.data.items() if amort else []`, the same empty array for
    a null and for an empty `DateList`.
    """
    from derivus.instruments import construct_instrument

    def read(value):
        return getattr(value, 'amount', getattr(value, 'data', value))

    def same(one, other):
        return read(one) == read(other) or not (read(one) or read(other))

    emitted, restated = decoded_blocks(CURVES), decoded_blocks(CURVES, restated=True)
    checked = 0
    for name, block in emitted.items():
        rows = zip(block['instrument']['Points'], restated[name]['instrument']['Points'])
        for row, full in rows:
            declared = declared_deal_fields(row['DealType'])
            for key in set(declared) - set(row['Deal']):
                assert declared[key] or key in ('Object', 'Discount_Rate'), (name, key)
            lean = construct_instrument(dict(row['Deal'], Object=row['DealType']), {})
            whole = construct_instrument(dict(full['Deal'], Object=row['DealType']), {})
            for key in declared:
                if key in whole.field:
                    assert same(lean.field[key], whole.field[key]), (name, row['Tenor'], key)
                    checked += 1
    assert checked > 200, checked


def test_the_block_installs_and_a_value_only_retick_updates():
    """`update_market_quote` is the contract every quote source posts against, and this block passes
    it BOTH WAYS because `InterestRatePrices` quotes in `Points` rows, which is what
    `schema.partition_market_price` gives a values half to.

    A moved RATE is a tick - mid, both sides and stamp on the value plane, `Deal` unmoved, which is
    what authoring the quote OUTSIDE the deal buys. A moved CONVENTION is a new plan and refuses.
    """
    from derivus.schema import update_market_quote

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

    # a moved CONVENTION is a re-authoring and refuses: the guard reading the plan half
    seed = copy.deepcopy(SEED)
    seed['rates']['ZAR']['conventions']['fixed_frequency'] = '6M'
    with pytest.raises(ValueError, match='structure differs'):
        update_market_quote(document, name, block_of('ZAR', seed=seed)[1])

    with pytest.raises(ValueError, match='a Market Prices block is'):
        update_market_quote(document, name, block['instrument'])


def test_a_standing_block_stating_every_convention_is_reauthored_once():
    """THE ONE THING THAT MOVES. A book written before a block stated only what differs
    carries every convention on every row; the same strip emitted now is the same instruments in
    fewer keys, and the plan guard cannot tell that from a mis-authoring - so the first tick drops
    and re-installs the block, and every tick after it is a tick again.

    Nothing priced moves across that re-authoring: `test_a_convention_the_block_leaves_unsaid_reads
    _as_the_declaration` is the reading, and this is the WRITE side of the same fact.
    """
    from derivus.schema import update_market_quote

    name, block = block_of('ZAR')
    standing = copy.deepcopy(block)
    for row in standing['instrument']['Points']:
        declared = declared_deal_fields(row['DealType'])
        row['Deal'] = dict({key: value for key, value in ir_curve.DECLARED.items()
                            if key in declared}, **row['Deal'])
    assert sum(len(row['Deal']) for row in standing['instrument']['Points']) == 133
    assert sum(len(row['Deal']) for row in block['instrument']['Points']) == 45

    document = job_document()
    assert update_market_quote(document, name, standing) == 'installed'
    prices = document['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices']
    with pytest.raises(ValueError, match='structure differs'):
        update_market_quote(document, name, block)
    assert reauthor(prices, name, block) == 'reauthored'
    assert update_market_quote(document, name, block_of('ZAR')[1]) == 'updated'


def test_a_rolled_date_strip_reaches_a_book_through_reauthor():
    """THE OTHER HALF OF THE ROUND TRIP: TOMORROW'S STRIP, which could never be a tick.
    `Effective_Date` and `Maturity_Date` are structure, so the same benchmarks fetched a day later
    are a different plan and `update_market_quote` refuses - rightly, since the guard cannot tell a
    rolled date from a mis-authored one.

    So a next-day curve reaches the book dropped and re-installed, through the one function that
    also serves a re-quoted swaption ladder: there the values half is EMPTY and no tick exists at
    all, here the values half works fine and the date is simply not in it.
    """
    from derivus.schema import update_market_quote

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

    # and it is REACHABLE off the package, which the lazy re-export did not carry
    assert 'derivus_bloomberg' in in_a_fresh_interpreter(
        'from derivus_bloomberg import reauthor; assert callable(reauthor)')
    import derivus_bloomberg

    assert 'reauthor' in derivus_bloomberg.__all__ and 'reauthor' in dir(derivus_bloomberg)
    assert derivus_bloomberg.reauthor is reauthor


def declared_of(block, curve, as_of=AS_OF, seed=None):
    """A block read back as the definition it is and authored again - the round trip a rolled date
    takes, and the one a desk's own rows take when it sets a curve up."""
    instrument = block['instrument']
    return ir_curve.author_block(
        {'curve': curve, 'currency': instrument['Currency'],
         'conventions': ir_curve.block_conventions(instrument, seed or SEED, curve),
         'rows': ir_curve.block_rows(instrument),
         'discount_rate': instrument['Discount_Rate']}, as_of)


def test_a_block_carries_everything_it_would_be_re_authored_from():
    """THE BLOCK IS THE CURVE'S DEFINITION, and this is the claim measured: every block the fetch
    emits, read back through `block_conventions` and `block_rows` and authored again at its OWN
    date, is the SAME BYTES. Nothing is read off the seed but the print scale, which is a property
    of the feed and not of the curve.

    That is what makes a rolled date a re-roll rather than a guess: at a later date the same rows
    and the same conventions produce the same benchmarks on new dates, which is what
    `update_market_quote` refuses and `reauthor` then installs.
    """
    for curve in CURVES:
        name, block = block_of(curve)
        assert declared_of(block, curve) == (name, block), curve

    name, block = block_of('ZAR')
    rolled = declared_of(block, 'ZAR', as_of=AS_OF + datetime.timedelta(days=1))[1]
    dates = lambda item: [row['Deal']['Maturity_Date'] for row in item['instrument']['Points']]

    assert dates(rolled) != dates(block)
    assert [row['Tenor'] for row in rolled['instrument']['Points']] == [
        row['Tenor'] for row in block['instrument']['Points']]
    # the conventions came off the BLOCK, so a seed that no longer names the curve changes nothing
    assert declared_of(block, 'ZAR', seed={'rates': {}}) == (name, block)


def test_a_declared_row_reads_its_shape_off_its_own_tenor():
    """A DESK STATES A TENOR AND A NUMBER, never an instrument. The declared front and `ON` author
    a deposit, an `x` a FRA, two tenors run together a forward-starting swap and anything else a
    spot swap - the same four shapes the map's own grammar reaches, off the label alone.

    A row held out with `Use` No keeps its place in the block and drops its KNOT with it, so two
    held-out benchmarks on one maturity are no clash; a used row with no number refuses BY NAME,
    carrying the verdict where a print was asked for and not believed, because an unquoted
    benchmark identifies no knot and the block cannot bootstrap.
    """
    def rows(*declared):
        return [ir_curve.RatePrint(label=label, kind='', security='', value=value, use=use)
                for label, value, use in declared]

    def author(declared, front='fixings/3M'):
        conventions = ir_curve.CurveConventions(**dict(SHIPPED['ZAR'], front=front))
        return ir_curve.author_block(
            {'curve': 'ZAR', 'currency': 'ZAR', 'conventions': conventions, 'rows': declared},
            AS_OF)[1]['instrument']

    block = author(rows(('3M', 7.41, 'Yes'), ('1Mx4M', 7.35, 'Yes'), ('6M1M', 7.5, 'Yes'),
                        ('3Y', 8.21, 'Yes')))
    overnight = author(rows(('ON', 7.02, 'Yes'), ('3Y', 8.21, 'Yes')), front='overnight')

    assert [row['DealType'] for row in block['Points']] == [
        'DepositDeal', 'FRADeal', 'SwapInterestDeal', 'SwapInterestDeal']
    assert block['Points'][2]['Deal']['Effective_Date'] != block['Points'][3]['Deal'][
        'Effective_Date'], 'a forward-starting swap that started at spot'
    assert overnight['Points'][0]['DealType'] == 'DepositDeal'
    assert overnight['Points'][0]['Deal']['Payment_Frequency'] == {'.DateOffset': '1D'}
    # no security, so no parenthetical - and the authoring date is the row's own stamp
    assert overnight['Points'][0]['Descriptor'] == 'ZAR ON'
    assert overnight['Points'][0]['Timestamp'] == {'.Timestamp': AS_OF.isoformat()}

    held = author(rows(('3M', 7.41, 'Yes'), ('1Y', 7.62, 'Yes'), ('4W', 7.3, 'No')))
    assert [row['Use'] for row in held['Points']] == ['No', 'Yes', 'Yes']

    with pytest.raises(InvalidQuote, match=r'2Y'):
        author(rows(('3M', 7.41, 'Yes'), ('1Y', 7.62, 'Yes'), ('2Y', None, 'Yes')))
    with pytest.raises(BloombergConfigurationError, match="'3Q'"):
        author(rows(('3M', 7.41, 'Yes'), ('3Q', 7.62, 'Yes')))


def test_a_broken_seed_is_a_configuration_refusal_and_never_a_no_terminal_skip():
    """A BROKEN WORKSTATION MUST NOT READ AS AN ABSENT ONE - a property of the CATCH SITES rather
    than of the taxonomy.

    `BloombergConfigurationError` hangs off `BloombergFXError` with everything else, and the live
    smokes used to catch that BASE: a workstation whose `seed.json` had lost its `conventions` block
    reported an absent terminal while the terminal answered fine. The hierarchy is untouched
    (`derivus.service` and `derivus_mcp` catch the base) and the smokes catch `NO_TERMINAL`; widen
    that tuple back and the last assertion goes red.
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

    # the two skips a smoke MAY make are named apart: this workstation meets the second for real,
    # its terminal refusing with `DAILY_CAPACITY_REACHED` once the daily quota is spent
    absent = no_terminal_reason(BloombergUnavailable('no blpapi'))
    refused_request = no_terminal_reason(BloombergRequestError('DAILY_CAPACITY_REACHED'))
    assert absent.startswith('no Bloomberg terminal answering')
    assert 'answered and refused' in refused_request and 'DAILY_CAPACITY_REACHED' in refused_request
    assert 'no Bloomberg terminal answering' not in refused_request


def test_the_engine_builds_the_authored_deals_and_reads_their_knots():
    """READ-ONLY, AND NO SOLVE. The block is decoded by the engine's JSON reader, every point turned
    into a benchmark deal node by `quote_nodes`, and `quote_knots` resets each leaf and reads its
    last cashflow date.

    WHAT THIS PROVES: all four authored shapes CONSTRUCT, every leg resets, every benchmark
    produces a last cashflow date, and the knot grid is ASCENDING and strictly positive
    (`Factor1D.interpolate` divides by the tenor, so a zero knot is NaN). A FRA's and a forward
    swap's knots land at their OWN last cashflow rather than at the label's outer tenor, which is
    what puts a monthly forward strip's knots a month apart. The quote reaches the field the
    family's writer puts it in.

    WHAT IT DOES NOT PROVE is that the authored field NAMES are declared ones - a misspelled day
    count reaches this gate and passes it. That belongs to
    `test_every_authored_deal_key_is_one_the_committed_schema_declares`.
    """
    import pandas as pd
    from derivus.bootstrappers import quote_knots, quote_nodes

    prices = decoded_blocks(CURVES)
    for curve, expected in CURVES.items():
        instrument = prices[ir_curve.market_price_name(curve)]['instrument']
        assert len(instrument['Points']) == expected
        nodes = quote_nodes(instrument['Points'], instrument['Currency'])
        knots = quote_knots(nodes, pd.Timestamp(AS_OF), instrument['Day_Count'], {})
        assert len(knots) == expected
        assert list(knots) == sorted(knots), (curve, knots)
        assert all(knot > 0.0 for knot in knots), (curve, knots)
        # the knot IS the maturity the row was authored to, in the block's own day count
        for point, knot in zip(instrument['Points'], knots):
            assert knot == pytest.approx(
                (point['Deal']['Maturity_Date'] - pd.Timestamp(AS_OF)).days / 365.0), point['Tenor']

    # the quote reached each instrument through the family's own writer, per type
    zar = prices['InterestRatePrices.ZAR']['instrument']
    for point in zar['Points']:
        node = quote_nodes([point], 'ZAR')[0]
        if point['DealType'] == 'SwapInterestDeal':
            assert float(node['Instrument'].field['Swap_Rate']) == pytest.approx(
                point['Quoted_Market_Value'])
        elif point['DealType'] == 'FRADeal':
            assert float(node['Instrument'].field['FRA_Rate']) == pytest.approx(
                point['Quoted_Market_Value'])


def test_every_authored_shape_solves_to_par():
    """THE SHAPES PRICE. Each canned block is bootstrapped the way `Config.bootstrap` runs the
    family, and every benchmark it was solved from is then repriced off the solved curve and has to
    come back at PV zero - a deposit, a FRA, a term swap, an OIS swap and a forward-starting swap.

    This is what a knot grid being square looks like from outside: one knot per used quote at its
    own last cashflow, so the residual vector and the unknowns are the same length and the damped
    Newton has a root to find. A shape whose dates the emitter got wrong reprices away from par
    here even though it constructs.
    """
    import pandas as pd
    import torch
    from derivus.bootstrappers import (BenchmarkInstruments, InterestRateCurveParameters,
                                       author_quote, completed, quote_node)
    from derivus.config import ModelParams

    base, device = pd.Timestamp(AS_OF), torch.device('cpu')
    prices = decoded_blocks(CURVES)
    for curve in CURVES:
        name = ir_curve.market_price_name(curve)
        block = prices[name]['instrument']
        currency = block['Currency']
        price_factors = {'FxRate.{}'.format(currency): {
            'Domestic_Currency': None, 'Interest_Rate': curve, 'Priority': 1, 'Spot': 1.0}}
        InterestRateCurveParameters({}, device, torch.float32).bootstrap(
            {'Base_Date': base, 'Base_Currency': currency}, {}, price_factors, ModelParams(),
            {name: {'instrument': copy.deepcopy(block), 'Children': []}}, {})
        assert 'InterestRate.{}'.format(curve) in price_factors

        nodes = []
        for point in block['Points']:
            deal = completed(copy.deepcopy(dict(point['Deal'], Object=point['DealType'])))
            author_quote(deal, point['Quoted_Market_Value'], curve)
            nodes.append(quote_node(deal, {}))
        priced = BenchmarkInstruments(nodes, price_factors, ModelParams(), base, currency, {}, [],
                                      device)({}).detach().numpy()
        # a notional of a million, so 1e-6 is a thousandth of a basis point of the principal
        assert abs(priced).max() < 1e-6, (curve, priced)




# =============================================================================================
# 7  live smoke
# =============================================================================================

def test_a_live_terminal_answers_the_strip_or_the_smoke_skips_by_name():
    """LIVE SMOKE; no terminal is a SKIP rather than a failure.

    WHAT IS ASSERTED IS THE ROUTE, not the market: this workstation's map and seed reach a strip,
    every candidate comes back as a print or a NAMED refusal, and whatever survives authors a block.
    The census is PRINTED and never asserted - a strip read out of hours screens differently from
    one read at noon.

    THE WORKSTATION SEED IS READ FOR ITS VOCABULARY, not necessarily its conventions: where
    `DV_HOME/seed.json` predates this build the conventions come from the PACKAGED seed, said in the
    census, with the desk's file left alone. Which currencies exist is still the workstation's map.
    """
    import time

    from derivus_bloomberg import security_map
    from derivus_bloomberg.session import blpapi_module

    # a live pull is a decision, never a side effect of running the suite on a workstation with
    # a terminal; it is taken by setting the variable
    if not os.environ.get('DERIVUS_LIVE_BLOOMBERG'):
        pytest.skip('live Bloomberg smoke runs only with DERIVUS_LIVE_BLOOMBERG set')
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

    # NO_TERMINAL AND NOT `BloombergFXError`: a configuration or seed refusal must FAIL here rather
    # than skip green as an absent terminal
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
