"""The verified ATM swaption grid as a `HullWhite2FactorModelPrices` block -
`derivus_bloomberg.swaption_vol`.

Nothing here opens a socket except the last gate, which skips by name where this workstation has no
terminal. Everything else runs on ONE canned ZAR grid - five expiries against three tenors, walked
through the package's real discovery grammar into a real verified map, with a poison table of dead
cells authored on purpose - driven through the package's own reader and the real engine seams
(`config.update_market_quote` and `schema.partition_market_price`, both imported READ-ONLY). The
HW2F calibration itself is NOT run: the fit-through is the composition harness's reading, and what
is held here is the block under it.

WHAT IS HELD:

  the budget       `swaption_vol` imports the standard library, this package's own modules and a
                   LAZY blpapi - read off the source and proved again in a fresh interpreter
  the columns      `INSTRUMENT_COLUMNS` equals the COMMITTED `HullWhite2FactorModelParameters`
                   declaration, read via `git show HEAD` and compared AS DATA - two day-count
                   columns and no `Day_Count`, which is the defect authored blocks used to die on
  the declaration  the SHIPPED ZAR conventions spelled out as data, and every way a convention can
                   be absent, unread or unauthorable refusing at the seed
  the screen       the order of distrust, one canned cell per verdict - and `zero` is the one that
                   matters most, because a zero `Market_Volatility` USED TO BE a silent instruction
                   to read the surface's ATM rather than a bad number. The engine refuses that row
                   by name now; this screen stays, because a ladder is refused here where a desk can
                   see which cell went dark
  the row          the seed's declared conventions on every row, the vol scaled into the family's
                   `Percent` column, `Weight` flat at v1's declaration, and the two-way and the
                   stamp riding beside them as undeclared keys
  the distribution declared, carried into `Quote_Source`, and READ BY THE ENGINE since 2026-09-01 -
                   the finding this file gated in its open shape, now gated closed: the premium
                   construction is held to the Bachelier pair as source AND as behaviour, and the
                   block's own line is held to saying which convention its numbers are in, because
                   the declaration the engine reads lives on a SURFACE this emitter does not author
  the partition    this family has an EMPTY values half, so `update_market_quote` refuses a re-tick
                   and `reauthor` is the only route a re-quoted grid reaches a book by
  determinism      the same canned grid emits the same bytes
  live smoke       one real ZAR ladder off this workstation's terminal, or a skip by name

NO MONKEYPATCHING. The canned terminal is the curve gate's own `BloombergSession` subclass, whose
event walk yields rows; the engine is imported and never touched.
"""
import copy
import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_bloomberg import discover, swaption_vol
from derivus_bloomberg.errors import BloombergConfigurationError, IncompleteLadder
from derivus_bloomberg.session import BloombergSession
from derivus_bloomberg.swaption_vol import (SwaptionScreen, fetch_swaption_ladder, hw2f_block,
                                            reauthor, screen_ladder, swaption_conventions)

from test_curve_strip_emitter import (AS_OF, LONG_AGO, NO_TERMINAL, ROOT, YESTERDAY, Walked,
                                      committed_fields, imported_names, in_a_fresh_interpreter,
                                      no_terminal_reason, packaged_seed, verified_map)

#: The SHIPPED declaration, restated as data - the owner's gate. A `SASN` cell is a NORMAL vol in
#: basis points on a forward swap that pays quarterly against quarterly on ACT/365, which is the
#: same `SASW` convention the curve strip authors. `quote_scale` 0.01 is what takes 145 basis points
#: to the 1.45 the family's `Percent` column carries.
SHIPPED = {'fixed_frequency': '3M', 'float_frequency': '3M', 'fixed_day_count': 'ACT_365',
           'float_day_count': 'ACT_365', 'distribution': 'Normal', 'quote_scale': 0.01,
           'weight': 1.0}

SEED = {
    'fx_vol': {}, 'fx_spot': {}, 'rates': {},
    'swaption': {'ZAR': {
        'prefix': 'SASN', 'expect': 'ZAR SWPT NVOL',
        'expiries': {'1M': '0A', '1Y': '01', '2Y': '02', '5Y': '05', '10Y': '10'},
        'tenor_years': [1, 5, 10],
        'conventions': SHIPPED}},
}

#: THE POISON TABLE, one entry per verdict the screen documents, as a terminal really answers them.
#: `None` means the map never verified the cell at all - which on a per-entitlement swaption grid is
#: the ordinary shape rather than an alarm.
POISON = {
    'SASN0A1 Curncy': None,                                    # unverified
    'SASN0A5 Curncy': {'PX_LAST': 0.0},                        # zero - the silent surface fallback
    'SASN0A10 Curncy': {'PX_LAST': None},                      # unpriced
    'SASN011 Curncy': {'PX_BID': 150.0, 'PX_ASK': 140.0},      # crossed
    'SASN015 Curncy': {'LAST_UPDATE_DT': LONG_AGO},            # stale
    'SASN0110 Curncy': {'LAST_UPDATE_DT': 'N/A'},              # undated
    'SASN051 Curncy': {'PX_LAST': 14500.0},                    # off-market: a decimal-shifted print
}

#: Normal vols in basis points, the shape a ZAR grid has: falling with expiry, falling with tenor.
CLEAN = {'SASN021 Curncy': 141.0, 'SASN025 Curncy': 133.5, 'SASN0210 Curncy': 126.0,
         'SASN055 Curncy': 128.0, 'SASN0510 Curncy': 120.5,
         'SASN101 Curncy': 118.0, 'SASN105 Curncy': 112.5, 'SASN1010 Curncy': 106.0,
         'SASN0A1 Curncy': 152.0, 'SASN0A5 Curncy': 146.0, 'SASN0A10 Curncy': 138.0,
         'SASN011 Curncy': 145.0, 'SASN015 Curncy': 137.0, 'SASN0110 Curncy': 130.0,
         'SASN051 Curncy': 124.0}


# =============================================================================================
# the canned world
# =============================================================================================

def canned_session(seed=None, poison=None):
    seed = seed or SEED
    poison = POISON if poison is None else poison
    rows = {}
    for candidate in discover.candidates_from_seed(seed):
        override = poison.get(candidate.security, {})
        if override is None:
            continue
        value = CLEAN.get(candidate.security, 130.0)
        fields = {'PX_LAST': value, 'PX_BID': value - 0.5, 'PX_ASK': value + 0.5,
                  'LAST_UPDATE_DT': YESTERDAY}
        fields.update(override)
        rows[candidate.security] = {'error': None,
                                    'fields': {key: item for key, item in fields.items()
                                               if item is not None}}
    return Walked(rows)


def ladder_of(seed=None, poison=None, screen=None, surface='ZAR-SWAPTION', as_of=AS_OF):
    # the poison table is resolved HERE and passed on explicitly, so the map and the session are
    # built from the same one - `verified_map` lives in the curve gate and would otherwise fall back
    # to that module's table, which is how a cell can be `invalid` in one half and live in the other
    seed, poison = seed or SEED, POISON if poison is None else poison
    return fetch_swaption_ladder(canned_session(seed, poison), verified_map(seed, poison), seed,
                                 'ZAR', as_of, surface=surface, screen=screen)


def block_of(**kwargs):
    curve = kwargs.pop('curve', 'ZAR-SWAP')
    screen = kwargs.pop('block_screen', None)
    return hw2f_block(ladder_of(**kwargs), curve=curve, screen=screen)


def rows_of(block):
    return {'{} x {}'.format(row['Start']['.DateOffset'], row['Tenor']['.DateOffset']): row
            for row in block['instrument']['Instrument_Definitions']}


# =============================================================================================
# 1  the dependency budget
# =============================================================================================

def test_the_swaption_emitter_imports_the_standard_library_and_nothing_else():
    """The package's budget, extended to the second new module: the standard library, this
    package's own modules, and NO ENGINE. Like `ir_curve` and for the same reason it makes no
    pandas-free claim - it reaches `discover` for the two-character expiry code rather than
    spelling a second one."""
    imported = imported_names(os.path.join(ROOT, 'derivus_bloomberg', 'swaption_vol.py'))
    assert imported <= {'collections', 'datetime', 'math', 'dataclasses', 'typing',
                        'derivus_bloomberg'}, sorted(imported)
    assert imported.isdisjoint({'derivus', 'torch', 'pandas', 'numpy', 'scipy', 'blpapi'}), \
        sorted(imported)
    assert imported


def test_importing_the_swaption_emitter_lands_no_engine_and_no_blpapi():
    """Proved a second way, in a fresh interpreter - and the last line is why both emitters are
    re-exported LAZILY: reaching one off the package must not cost the chain emitter its own
    measured budget.

    THE CLAIMS ARE ASYMMETRIC, exactly as the curve twin's are. `derivus` and `blpapi` must not
    land; `pandas` DOES, through the map layer this module reaches for its ticker grammar, and it is
    asserted POSITIVELY - a gate that quietly allowed it would not be stating a budget, it would be
    omitting one.
    """
    landed = in_a_fresh_interpreter('import derivus_bloomberg.swaption_vol')
    assert 'derivus_bloomberg' in landed, 'the module did not import'
    assert landed.isdisjoint({'derivus', 'torch', 'blpapi'}), sorted(
        landed & {'derivus', 'torch', 'blpapi'})
    assert 'pandas' in landed, 'the stated budget says the grammar costs pandas'
    assert 'derivus_bloomberg' in in_a_fresh_interpreter(
        'from derivus_bloomberg import fetch_swaption_ladder, hw2f_block, reauthor')


# =============================================================================================
# 2  the columns, against the committed declaration
# =============================================================================================

def test_the_row_is_the_committed_schemas_own_declaration():
    """THE GATE THIS BUILD WAS ASKED FOR, and it reads the COMMITTED state rather than the working
    tree, so it holds while another workflow is mid-edit in `bootstrappers.py`.

    `INSTRUMENT_COLUMNS` is a whitelist spelled in a package that may not import the engine, so it
    can only be trusted if something compares it to the declaration - and it is compared AS DATA,
    in order, off an AST parse of `git show HEAD:derivus/bootstrappers.py`.

    TWO DAY-COUNT COLUMNS AND NO `Day_Count`. `create_market_swaps` reads `Floating_Day_Count` to
    generate the float leg that gives the par swap rate and `Fixed_Day_Count` only on the
    unequal-frequency branch; an authored block spelling one `Day_Count` for both is the defect the
    roadmap records, and it dies in a cashflow generator rather than at the schema. This is that
    defect turned into a comparison.
    """
    declared = committed_fields('HullWhite2FactorModelParameters', table='Instrument_Definitions')
    assert tuple(declared) == swaption_vol.INSTRUMENT_COLUMNS, declared
    assert 'Day_Count' not in declared
    assert declared.count('Floating_Day_Count') == 1 and declared.count('Fixed_Day_Count') == 1

    # and what the emitter actually writes IS that row, in that order, plus exactly the evidence
    # keys the family does not declare
    row = list(rows_of(block_of()[1]).values())[0]
    assert tuple(key for key in row if key in declared) == swaption_vol.INSTRUMENT_COLUMNS
    assert set(row) - set(declared) == set(swaption_vol.QUOTE_VALUE_KEYS)

    # the BLOCK-level keys, and every one of them is now DECLARED. `Quote_Source` used to be the
    # one key this family had no column for - the emitter wrote it anyway because provenance is the
    # evidence, and `bootstrap` read past it - which is the gap this half asserted. HW2F declares
    # `Quote_Source` and `Quote_Timestamp` since 2026-09-01, on the shape both Heston-Nandi
    # families already had, so the subtraction is empty. Read off the WORKING TREE, because the
    # declaration lands in this same change and HEAD cannot be asked about it.
    instrument = block_of()[1]['instrument']
    block_fields = committed_fields('HullWhite2FactorModelParameters', at=None)
    assert set(instrument) - set(block_fields) == set(), sorted(set(instrument) - set(block_fields))
    assert 'Quote_Source' in block_fields and 'Quote_Timestamp' in block_fields
    assert set(instrument) == {'Swaption_Volatility', 'Instrument_Definitions', 'Quote_Source'}
    # and the row columns are unmoved by it: the block gained two fields, the ladder none
    assert 'Quote_Source' not in declared and 'Quote_Timestamp' not in declared


def test_the_engine_reads_the_declared_distribution_and_this_block_says_which():
    """THE FINDING, CLOSED, AND GATED IN ITS NEW SHAPE. This gate was
    `test_the_engine_prices_a_lognormal_black_and_reads_no_distribution` and it asserted the GAP:
    `SASN` is a NORMAL vol in basis points, `create_market_swaps` priced every benchmark's premium
    with `utils.black_european_option_price` whatever the surface declared, and `InterestYieldVol`'s
    `Distribution_Type` reached the deal path and nothing else. It said "the day
    `create_market_swaps` reads a distribution this gate fails and says so", and that day was
    2026-09-01. It is rewritten rather than deleted, holding the same seam from the other side.

    TWO HALVES, AND THE SECOND ONE IS BEHAVIOURAL. The engine's premium construction is READ, as
    text, off the working tree rather than off `git show HEAD` - the committed-source trick the
    column gate uses is right for a DECLARATION, which has to hold while another workflow edits the
    module, and wrong for a behaviour that only exists once the edit lands. So the source half is a
    statement about what is running, and the run below is what proves it: the same ladder's own
    numbers priced under the two declarations come out an ORDER OF MAGNITUDE apart, which is what
    reading `Distribution_Type` is worth and what ignoring it cost.

    This package may not import the engine anywhere but here, and it does not: the import is inside
    this gate, which is a test of the engine's seam rather than of the emitter's budget - the budget
    gates one section up are what hold that line.
    """
    import inspect

    from derivus import bootstrappers, riskfactors, utils

    body = inspect.getsource(bootstrappers.create_market_swaps)
    assert 'get_subtype' in body, (
        'create_market_swaps no longer reads the surface\'s declared convention - the roadmap row '
        'this gate closes is open again')
    assert 'PREMIUM_CONVENTIONS' in body and 'displacement' in body, body[:400]
    assert bootstrappers.PREMIUM_CONVENTIONS['Normal'] == (
        utils.bachelier_european_option_price, utils.bachelier_european_option), (
        'a Normal surface has to reach the Bachelier pair, numpy premium and tensor twin alike')
    assert bootstrappers.PREMIUM_CONVENTIONS['Lognormal'] == (
        utils.black_european_option_price, utils.black_european_option)
    declared = next(f for f in riskfactors.InterestYieldVol.fields
                    if f.name == 'Distribution_Type')
    assert sorted(declared.values) == sorted(bootstrappers.PREMIUM_CONVENTIONS), (
        'the surface declares {} and the calibration prices {} - a value a block can author and '
        'the engine cannot price is the same defect one layer over'.format(
            declared.values, sorted(bootstrappers.PREMIUM_CONVENTIONS)))
    assert SHIPPED['distribution'] in bootstrappers.PREMIUM_CONVENTIONS, (
        'the seed declares a distribution this engine cannot price')

    # the behavioural read: this ladder's own quotes, priced under each declaration
    row = list(rows_of(block_of()[1]).values())[0]
    quote, expiry = row['Market_Volatility']['.Percent'] / 100.0, 2.0
    premium = {name: pricer(0.09, 0.09, 0.0, quote, expiry, 1.0, 1.0)
               for name, (pricer, _) in bootstrappers.PREMIUM_CONVENTIONS.items()}
    assert premium['Normal'] / premium['Lognormal'] > 5.0, (
        'the two conventions price {} within {:.3g}x of each other - a normal vol read as a '
        'lognormal one is the defect this gate exists for'.format(
            row['Market_Volatility'], premium['Normal'] / premium['Lognormal']))

    # and the block still says which, because the family declares no field for it and the surface
    # this emitter does not author is what the engine actually reads
    source_line = block_of()[1]['instrument']['Quote_Source']
    assert 'NORMAL vols' in source_line
    assert 'the convention the named surface DECLARES' in source_line
    assert 'Distribution_Type Normal' in source_line
    assert 'LOGNORMAL Black' not in source_line and 'reads no Distribution_Type' not in source_line


# =============================================================================================
# 3  the declaration
# =============================================================================================

def test_the_shipped_swaption_conventions_are_the_declared_ones():
    """THE OWNER'S GATE. Everything about a `SASN` cell that its ticker does not say, as data, read
    off the seed the wheel ships."""
    assert packaged_seed()['swaption']['ZAR']['conventions'] == SHIPPED
    conventions = swaption_conventions(packaged_seed(), 'ZAR')
    assert conventions.distribution == 'Normal'
    assert conventions.quote_scale == 0.01, '145 basis points is 1.45 in a Percent column'
    assert conventions.weight == 1.0
    assert (conventions.fixed_frequency, conventions.float_frequency) == ('3M', '3M')
    assert (conventions.fixed_day_count, conventions.float_day_count) == ('ACT_365', 'ACT_365')


def test_a_grid_without_its_conventions_refuses_naming_every_missing_field():
    """The same refusal shape the curve emitter makes, and for the same reason - a desk extending a
    seed wants the whole questionnaire at once."""
    seed = copy.deepcopy(SEED)
    del seed['swaption']['ZAR']['conventions']['distribution']
    del seed['swaption']['ZAR']['conventions']['quote_scale']
    with pytest.raises(BloombergConfigurationError) as refused:
        swaption_conventions(seed, 'ZAR')
    message = str(refused.value)
    assert 'declares no distribution, quote_scale' in message
    for name in swaption_vol.REQUIRED_CONVENTIONS:
        assert name in message

    with pytest.raises(BloombergConfigurationError,
                       match='the seed names no swaption entry for USD'):
        swaption_conventions(seed, 'USD')

    stripped = copy.deepcopy(SEED)
    del stripped['swaption']['ZAR']['conventions']
    with pytest.raises(BloombergConfigurationError, match='carries no swaption `conventions`'):
        swaption_conventions(stripped, 'ZAR')

    extra = copy.deepcopy(SEED)
    extra['swaption']['ZAR']['conventions']['shift'] = 3.0
    with pytest.raises(BloombergConfigurationError, match='shift'):
        swaption_conventions(extra, 'ZAR')


def test_a_declaration_this_emitter_cannot_author_refuses_at_the_seed():
    """The ways a declaration can be wrong rather than absent. The day-count list is the FAMILY's
    own - nothing here computes an accrual, `create_market_swaps` does - so the whole declared set
    passes and a spelling outside it refuses where a desk can fix it."""
    for field, value, expected in (
            ('distribution', 'Bachelier', 'is not one this emitter carries'),
            ('fixed_day_count', 'ACT_364', 'is not a day count HullWhite2FactorModelParameters'),
            ('float_frequency', 'quarterly', 'is not a tenor this emitter can read'),
            ('weight', 0.0, 'weight must be positive'),
            ('float_frequency', '0M', 'has to be a positive period'),
            ('quote_scale', 0.0, 'quote_scale must be finite and non-zero')):
        seed = copy.deepcopy(SEED)
        seed['swaption']['ZAR']['conventions'][field] = value
        with pytest.raises(BloombergConfigurationError, match=expected):
            swaption_conventions(seed, 'ZAR')

    # the whole declared list IS admissible, which is what makes the refusal above a boundary
    for day_count in swaption_vol.DAY_COUNTS:
        seed = copy.deepcopy(SEED)
        seed['swaption']['ZAR']['conventions']['fixed_day_count'] = day_count
        assert swaption_conventions(seed, 'ZAR').fixed_day_count == day_count


# =============================================================================================
# 4  the screen
# =============================================================================================

def test_the_screen_classifies_off_the_terminals_own_answers():
    """The order of distrust, one canned cell per verdict.

    `zero` sits second on purpose. Every other verdict here refuses a number that is WRONG; this
    one refuses a number that is an INSTRUCTION: `create_market_swaps` reads
    `if instrument['Market_Volatility'].amount:` and falls through to the swaption surface's own ATM
    when it is false. A blank cell emitted as zero would therefore calibrate against whatever the
    book's surface happened to hold, under the name of a quote the terminal never gave - and
    nothing anywhere would say so.
    """
    def cell(**extra):
        return swaption_vol.SwaptionQuote(**dict(
            {'expiry': '1Y', 'tenor': '5Y', 'security': 'x', 'value': 1.45, 'bid': 1.44,
             'ask': 1.46, 'last_update': YESTERDAY}, **extra))

    cases = {
        'unpriced': cell(value=None),
        'zero': cell(value=0.0),
        'off-market': cell(value=145.0),
        'crossed': cell(bid=1.50, ask=1.40),
        'undated': cell(last_update='N/A'),
        'stale': cell(last_update=LONG_AGO),
        'live': cell(),
    }
    named = [swaption_vol.SwaptionQuote(**dict(item.__dict__, security=verdict))
             for verdict, item in cases.items()]
    accepted, rejected = screen_ladder(named, AS_OF)
    assert [item.security for item in accepted] == ['live']
    assert rejected == {verdict: verdict for verdict in cases if verdict != 'live'}

    # a mid-only cell is BELIEVED: a two-way the terminal never quoted is not a spread
    assert screen_ladder([cell(security='mid-only', bid=None, ask=None)], AS_OF)[1] == {}


def test_the_canned_grid_is_believed_by_census():
    """Every candidate accounted for, one way or the other. A swaption grid is ragged by
    entitlement, so what was refused and why is most of what a desk needs to read."""
    ladder = ladder_of()
    assert ladder.rejected == {
        'SASN0A1 Curncy': 'unverified', 'SASN0A5 Curncy': 'zero',
        'SASN0A10 Curncy': 'unpriced', 'SASN011 Curncy': 'crossed',
        'SASN015 Curncy': 'stale', 'SASN0110 Curncy': 'undated',
        'SASN051 Curncy': 'off-market'}
    assert ladder.census == {'unverified': 1, 'zero': 1, 'unpriced': 1, 'crossed': 1,
                             'stale': 1, 'undated': 1, 'off-market': 1}
    assert len(ladder.quotes) == 8
    assert sorted('{} x {}'.format(quote.expiry, quote.tenor) for quote in ladder.quotes) == [
        '10Y x 10Y', '10Y x 1Y', '10Y x 5Y', '2Y x 10Y', '2Y x 1Y', '2Y x 5Y',
        '5Y x 10Y', '5Y x 5Y']

    # the census a report reads is the same numbers as data, and it carries the DISTRIBUTION -
    # which is the one thing about this ladder a reader cannot recover from the vols themselves
    counted = swaption_vol.quote_census(ladder)
    assert (counted['asked'], counted['believed']) == (15, 8)
    assert counted['distribution'] == 'Normal' and counted['surface'] == 'ZAR-SWAPTION'
    assert counted['refused'] == ladder.rejected


def test_a_ladder_below_its_floor_refuses_naming_what_the_terminal_served():
    """Five is the family's own parameter count, so under it the fit interpolates and reports a
    stationarity it did not earn. The refusal names the count, the floor and the census - "not
    enough quotes" with no coordinates is not something a desk can act on."""
    poison = dict(POISON, **{security: {'PX_LAST': None} for security in
                             ('SASN021 Curncy', 'SASN025 Curncy', 'SASN0210 Curncy',
                              'SASN055 Curncy')})
    with pytest.raises(IncompleteLadder) as refused:
        block_of(poison=poison)
    message = str(refused.value)
    assert 'ZAR screened to 4 believed cells against a floor of 5' in message
    assert '5 unpriced' in message and '1 zero' in message
    assert 'sigma_1, sigma_2, alpha_1, alpha_2, rho' in message

    # and a block naming no surface refuses too: the family declares Swaption_Volatility REQUIRED
    # and reads a cell's ATM off it wherever Market_Volatility is zero, so an unnamed surface is a
    # calibration that resolves nothing
    import dataclasses

    with pytest.raises(BloombergConfigurationError, match='names no Swaption_Volatility surface'):
        hw2f_block(dataclasses.replace(ladder_of(), surface=''), curve='ZAR')


# =============================================================================================
# 5  the row
# =============================================================================================

def test_every_row_carries_the_seeds_declared_conventions_and_the_scaled_vol():
    """What the block says a benchmark IS, against what the seed declared it is - and the vol in the
    family's own units, which is the one arithmetic this emitter does.

    `Start` and `Tenor` are `Period`s and no date is computed here at all: `create_market_swaps`
    does `effective = base_date + Start` and `maturity = effective + Tenor`, so a 5Y x 10Y cell is a
    ten-year swap starting in five years and the calendar is the engine's.
    """
    rows = rows_of(block_of()[1])
    assert set(rows) == {'2Y x 1Y', '2Y x 5Y', '2Y x 10Y', '5Y x 5Y', '5Y x 10Y',
                         '10Y x 1Y', '10Y x 5Y', '10Y x 10Y'}
    row = rows['5Y x 10Y']
    assert row['Start'] == {'.DateOffset': '5Y'} and row['Tenor'] == {'.DateOffset': '10Y'}
    assert row['Floating_Frequency'] == {'.DateOffset': '3M'}
    assert row['Fixed_Frequency'] == {'.DateOffset': '3M'}
    assert row['Floating_Day_Count'] == 'ACT_365' and row['Fixed_Day_Count'] == 'ACT_365'
    # 120.5 basis points of NORMAL vol is 1.205 percent, whose decoded `.amount` is 0.01205
    assert row['Market_Volatility'] == {'.Percent': pytest.approx(1.205)}
    assert row['Weight'] == 1.0, 'flat one is v1\'s stated declaration, not a fallthrough'
    assert row['Quoted_Bid'] == pytest.approx(1.20) and row['Quoted_Ask'] == pytest.approx(1.21)
    assert row['Timestamp'] == {'.Timestamp': YESTERDAY}

    # the rows are ORDERED by the grid - expiry then tenor - so the block reads as a ladder
    assert list(rows) == ['2Y x 1Y', '2Y x 5Y', '2Y x 10Y', '5Y x 5Y', '5Y x 10Y',
                          '10Y x 1Y', '10Y x 5Y', '10Y x 10Y']

    # a mid-only cell carries no sides rather than a manufactured spread
    lonely = dict(POISON)
    lonely['SASN055 Curncy'] = {'PX_BID': None, 'PX_ASK': None}
    thin = rows_of(block_of(poison=lonely)[1])['5Y x 5Y']
    assert 'Quoted_Bid' not in thin and 'Quoted_Ask' not in thin
    assert thin['Market_Volatility'] == {'.Percent': pytest.approx(1.28)}


def test_the_wire_form_decodes_to_the_types_the_family_reads():
    """READ-ONLY, AND NO CALIBRATION. The block goes through the engine's own JSON reader, and what
    comes back are the types `create_market_swaps` indexes: `Start` and `Tenor` as `DateOffset`s it
    adds to a base date, `Market_Volatility` as a `Percent` whose `.amount` is the fraction, and
    `Weight` as a plain float.

    That is the wire claim proved rather than described. The fit-through - what the model does with
    a normal vol read as a lognormal one - is the composition harness's reading and is deliberately
    not run here.
    """
    import pandas as pd
    from derivus.config import Config

    name, block = block_of()
    document = {'Calc': {'Calculation': {}, 'Deals': {}, 'MergeMarketData': {
        'MarketDataFile': '', 'ExplicitMarketData': {'Market Prices': {name: block}}}}}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ladder_probe.json')
    try:
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(document, handle)
        data = Config().read_json(path)
    finally:
        if os.path.isfile(path):
            os.remove(path)

    assert name == 'HullWhite2FactorModelPrices.ZAR-SWAP'
    instrument = data['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices'][name][
        'instrument']
    row = instrument['Instrument_Definitions'][4]           # 5Y x 10Y
    base = pd.Timestamp(AS_OF)
    assert (base + row['Start']) == pd.Timestamp('2031-08-31')
    assert (base + row['Start'] + row['Tenor']) == pd.Timestamp('2041-08-31')
    assert row['Market_Volatility'].amount == pytest.approx(0.01205)
    assert row['Weight'] == 1.0
    assert instrument['Swaption_Volatility'] == 'ZAR-SWAPTION'


# =============================================================================================
# 6  the partition - read-only
# =============================================================================================

def test_this_family_has_an_empty_values_half_so_a_retick_is_a_reauthoring():
    """STRUCTURAL, not a preference, and it is the reason `reauthor` exists.

    `schema.partition_market_price` gives every family whose quotes do not live in `Points` rows an
    EMPTY values half - "a tick on such a block is a new plan, which is what it has always been" -
    and this family quotes in `Instrument_Definitions`. So a moved vol is not a value at all:
    `config.update_market_quote` sees the whole block as structure and refuses. `POST /book/hn`
    drops its Heston-Nandi block before re-installing for a weaker reason (its strikes are a
    function of the surface); here there is no tick that could reach the block in the first place.
    """
    from derivus import schema
    from derivus.config import update_market_quote

    name, block = block_of()
    structural, values = schema.partition_market_price(block)
    assert values == [], 'this family is wholly plan-side'
    assert structural == block

    document = {'Calc': {'MergeMarketData': {'ExplicitMarketData': {'Market Prices': {}}}}}
    assert update_market_quote(document, name, block) == 'installed'
    again = block_of()[1]
    assert again == block and again is not block
    assert update_market_quote(document, name, again) == 'updated'

    moved = dict(POISON)
    moved['SASN055 Curncy'] = {'PX_LAST': 129.5, 'PX_BID': 129.0, 'PX_ASK': 130.0}
    reticked = block_of(poison=moved)[1]
    node = lambda item: [(row['Start'], row['Tenor'])
                         for row in item['instrument']['Instrument_Definitions']]
    assert node(reticked) == node(block), 'the re-tick moved a cell, not a value'
    assert reticked != block
    with pytest.raises(ValueError, match='structure differs'):
        update_market_quote(document, name, reticked)

    # so the route is a DROP and a re-install, which is what `reauthor` is
    prices = document['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices']
    assert reauthor(prices, name, reticked) == 'reauthored'
    assert prices[name] is reticked
    assert update_market_quote(document, name, reticked) == 'updated', \
        're-installed, so the guard now compares the re-quoted block with itself'
    assert reauthor({}, name, block) == 'installed'
    with pytest.raises(BloombergConfigurationError, match='a Market Prices block is'):
        reauthor(prices, name, block['instrument'])

    # ONE `reauthor`, owned by `ir_curve` and reached from here rather than re-spelled: the drop and
    # re-install is a single mechanism, and only the REASON differs between the two families. This
    # family has no values half at all; a curve strip has one that a rolled date is simply not in
    from derivus_bloomberg import ir_curve

    assert reauthor is ir_curve.reauthor


def test_the_same_canned_grid_emits_the_same_bytes():
    """DETERMINISM, and the only clock in sight is the as-of, which is a parameter."""
    first = json.dumps(block_of()[1], sort_keys=True)
    assert first == json.dumps(block_of()[1], sort_keys=True)

    moved = dict(POISON)
    moved['SASN101 Curncy'] = {'PX_LAST': 121.0, 'PX_BID': 120.5, 'PX_ASK': 121.5}
    second = json.dumps(block_of(poison=moved)[1], sort_keys=True)
    assert second != first and '1.21' in second


# =============================================================================================
# 7  live smoke
# =============================================================================================

def test_a_live_terminal_answers_the_ladder_or_the_smoke_skips_by_name():
    """LIVE SMOKE, and a workstation with no terminal is a SKIP rather than a failure.

    WHAT IS ASSERTED IS THE ROUTE: that this workstation's own map reaches a ZAR grid, that every
    verified cell comes back as a quote or a NAMED refusal, and that whatever survives authors a
    block whose columns are the family's own. The census is PRINTED and never asserted - a grid read
    out of hours screens differently from one read at noon.

    The conventions come from the PACKAGED seed where the workstation's carries none, exactly as the
    curve smoke does, and this gate never edits a desk's own file.
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
    if 'ZAR' not in seed.get('swaption', {}) or 'ZAR' not in document.get('blocks', {}).get(
            'swaption', {}):
        pytest.skip('this workstation\'s map verified no ZAR swaption grid')
    borrowed = 'conventions' not in seed['swaption']['ZAR']
    if borrowed:
        seed['swaption']['ZAR']['conventions'] = \
            packaged_seed()['swaption']['ZAR']['conventions']
    print('\nconventions borrowed from the packaged seed: {}'.format(borrowed))

    # NO_TERMINAL AND NOT `BloombergFXError`: a doctored seed refuses with a
    # `BloombergConfigurationError`, which hangs off that base, and catching the base would report
    # a broken workstation as an absent one - see the curve gate's taxonomy test, which holds the
    # tuple both smokes share
    started = time.time()
    try:
        with BloombergSession(timeout_ms=60000, connect_timeout_ms=5000) as session:
            ladder = fetch_swaption_ladder(session, document, seed, 'ZAR',
                                           datetime.date.today(), surface='ZAR-SWAPTION')
    except NO_TERMINAL as refused:
        pytest.skip(no_terminal_reason(refused))

    asked = len(ladder.quotes) + len(ladder.rejected)
    print('ZAR swaption grid as at {}: {} asked, {} believed, {} refused ({}) in {:.0f}s'.format(
        ladder.as_of.isoformat(), asked, len(ladder.quotes), len(ladder.rejected),
        ', '.join('{} {}'.format(count, verdict)
                  for verdict, count in sorted(ladder.census.items())) or 'nothing refused',
        time.time() - started))
    assert asked > 0, 'the map carried no ZAR swaption candidate'
    if len(ladder.quotes) >= SwaptionScreen().minimum_rows:
        name, block = hw2f_block(ladder, curve='ZAR')
        rows = block['instrument']['Instrument_Definitions']
        print('  {} -> {} rows, vols {:.4g}..{:.4g} percent, cells {}'.format(
            name, len(rows), min(row['Market_Volatility']['.Percent'] for row in rows),
            max(row['Market_Volatility']['.Percent'] for row in rows),
            ', '.join(sorted(set(rows_of(block))))))
        assert all(tuple(key for key in row if key not in swaption_vol.QUOTE_VALUE_KEYS)
                   == swaption_vol.INSTRUMENT_COLUMNS for row in rows)
