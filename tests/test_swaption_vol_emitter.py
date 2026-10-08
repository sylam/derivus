"""The verified ATM swaption grid as a `HullWhite2FactorModelPrices` block -
`derivus_bloomberg.swaption_vol`.

Only the last gate opens a socket, and it skips by name where this workstation has no terminal.
Everything else runs on ONE canned ZAR grid - five expiries against three tenors, walked through the
package's real discovery grammar into a real verified map with a poison table of dead cells - driven
through the package's own reader and the real engine seams (`schema.update_market_quote`,
`schema.partition_market_price`), imported READ-ONLY. The HW2F calibration is NOT run: the
fit-through is the composition harness's reading.

WHAT IS HELD:

  the declaration  the SHIPPED ZAR conventions as data, and every way a convention can be absent,
                   unread or unauthorable refusing at the seed
  the screen       the order of distrust, one canned cell per verdict - `zero` matters most,
                   because a zero `Market_Volatility` used to be a silent instruction to read the
                   surface's ATM rather than a bad number
  the row          the seed's declared conventions on every row, the vol scaled into the family's
                   `Percent` column, `Weight` flat, and the two-way and stamp beside them
  the distribution declared ON THE BLOCK, the convention the terminal quoted
  the partition    this family has an EMPTY values half, so `update_market_quote` refuses a re-tick
                   and `reauthor` is the only route a re-quoted grid reaches a book by
  live smoke       one real ZAR ladder off this workstation's terminal, or a skip by name

NO MONKEYPATCHING: the canned terminal is the curve gate's `BloombergSession` subclass, and the
engine is imported and never touched.
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
                                            reauthor, swaption_conventions)

from test_curve_strip_emitter import (AS_OF, LONG_AGO, NO_TERMINAL, YESTERDAY, Walked,
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
    # built from the same one - `verified_map` lives in the curve gate and would otherwise fall
    # back to that module's table
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
# 3  the declaration
# =============================================================================================

def test_the_shipped_swaption_conventions_are_the_declared_ones():
    """THE OWNER'S GATE. Everything about a `SASN` cell that its ticker does not say, as data, read
    off the seed the wheel ships.

    Killing mutation: the shipped seed's quote scale moved."""
    assert packaged_seed()['swaption']['ZAR']['conventions'] == SHIPPED
    conventions = swaption_conventions(packaged_seed(), 'ZAR')
    assert conventions.distribution == 'Normal'
    assert conventions.quote_scale == 0.01, '145 basis points is 1.45 in a Percent column'
    assert conventions.weight == 1.0
    assert (conventions.fixed_frequency, conventions.float_frequency) == ('3M', '3M')
    assert (conventions.fixed_day_count, conventions.float_day_count) == ('ACT_365', 'ACT_365')


def test_a_grid_without_its_conventions_refuses_naming_every_missing_field():
    """The same refusal shape the curve emitter makes, and for the same reason - a desk extending a
    seed wants the whole questionnaire at once.

    Killing mutation: the refusal naming the first missing field alone."""
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
    passes and a spelling outside it refuses where a desk can fix it.

    Killing mutation: the distribution left unchecked against the two this emitter carries."""
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

def test_the_canned_grid_is_believed_by_census():
    """Every candidate accounted for, one way or the other, one canned cell per verdict in the order
    of distrust. A swaption grid is ragged by entitlement, so what was refused and why is most of
    what a desk needs to read. `zero` matters most: every other verdict refuses a number that is
    WRONG, this one a number that is an INSTRUCTION - `create_market_swaps` used to read a zero
    `Market_Volatility` as the surface's own ATM.

    Killing mutation: a zero cell believed."""
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
    enough quotes" with no coordinates is not something a desk can act on.

    Killing mutation: the floor at four.
    """
    poison = dict(POISON, **{security: {'PX_LAST': None} for security in
                             ('SASN021 Curncy', 'SASN025 Curncy', 'SASN0210 Curncy',
                              'SASN055 Curncy')})
    with pytest.raises(IncompleteLadder) as refused:
        block_of(poison=poison)
    message = str(refused.value)
    assert 'ZAR screened to 4 believed cells against a floor of 5' in message
    assert '5 unpriced' in message and '1 zero' in message
    assert 'sigma_1, sigma_2, alpha_1, alpha_2, rho' in message

    # a block naming no surface refuses too: the family declares Swaption_Volatility REQUIRED and
    # reads a cell's ATM off it wherever Market_Volatility is zero
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
    ten-year swap starting in five years and the calendar is the engine's. The block DECLARES the
    convention its numbers are in, in the family's own spelling, and the same canned grid emits the
    same bytes - the only clock in sight is the as-of, a parameter - where one moved cell moves them.

    Killing mutation: the print written unscaled - 120.5 where the family reads 1.205.
    """
    instrument = block_of()[1]['instrument']
    assert instrument['Distribution_Type'] == SHIPPED['distribution']
    assert 'NORMAL vols' in instrument['Quote_Source']
    assert json.dumps(block_of()[1], sort_keys=True) == json.dumps(block_of()[1], sort_keys=True)
    moved = dict(POISON)
    moved['SASN101 Curncy'] = {'PX_LAST': 121.0, 'PX_BID': 120.5, 'PX_ASK': 121.5}
    assert '1.21' in json.dumps(block_of(poison=moved)[1], sort_keys=True)
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
    """READ-ONLY, AND NO CALIBRATION. The block goes through the engine's JSON reader and comes back
    as the types `create_market_swaps` indexes: `Start` and `Tenor` as `DateOffset`s,
    `Market_Volatility` as a `Percent` whose `.amount` is the fraction, `Weight` a plain float. The
    fit-through is the composition harness's reading and is deliberately not run here.

    Killing mutation: `Start` written as the bare tenor rather than a wire period.
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
    """STRUCTURAL, not a preference, and the reason `reauthor` exists.
    `schema.partition_market_price` gives every family whose quotes do not live in `Points` rows an
    EMPTY values half, and this family quotes in `Instrument_Definitions` - so a moved vol is not a
    value at all and `update_market_quote` sees the whole block as structure and refuses.

    Killing mutation: `reauthor` keeping the standing block.
    """
    from derivus import schema
    from derivus.schema import update_market_quote

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


# =============================================================================================
# 7  live smoke
# =============================================================================================

def test_a_live_terminal_answers_the_ladder_or_the_smoke_skips_by_name():
    """LIVE SMOKE; no terminal is a SKIP rather than a failure.

    WHAT IS ASSERTED IS THE ROUTE: this workstation's map reaches a ZAR grid, every verified cell
    comes back as a quote or a NAMED refusal, and whatever survives authors a block whose columns
    are the family's own. The census is PRINTED and never asserted. The conventions come from the
    PACKAGED seed where the workstation's carries none, and this gate never edits a desk's file.
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

    # NO_TERMINAL AND NOT `BloombergFXError`: catching the base would report a broken workstation
    # as an absent one - the curve gate's taxonomy test holds the tuple both smokes share
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
