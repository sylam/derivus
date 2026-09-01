"""Is a quote a VALUE now, and is it one for every family by the same rule?

`Market Prices` used to be wholly plan-side, so a vol tick moved `plan_hash` and left `values_hash`
alone - a recompile coordinate for a number that changed nothing about the program. The split is
now `schema.MARKET_QUOTE_VALUES`: per quote row, the mid, its two-way sides and its `Timestamp` are
values, and every other key of the row and everything else on the block is structure. That is one
declaration, stated once and read by four call sites, and these gates are what holds them to it.

WHICH TABLE IS THE QUOTE TABLE is the second declaration, and it is DERIVED rather than listed:
`schema.quote_containers` reads the families' own rows and calls a table a quote table when its row
declares all four value keys, which `schema.MARKET_QUOTE_CONTAINERS` holds. So an FX smile and a
curve strip quote in `Points`, the two Heston-Nandi families in `European_Options`, and a family
joins the value plane by DECLARING the columns rather than by being named somewhere.

The rule is applied UNIFORMLY, which is the part worth reading twice. Three of the seven declared
families still have an EMPTY values half - `HullWhite2FactorModelPrices` quotes a
`Market_Volatility` in `Instrument_Definitions`, `CSForwardPriceModelPrices` a mid and no two-way in
`Energy_Futures_Options`, `GBMAssetPriceTSModelPrices` nothing at all - so a tick on one of them is
still a new plan. That is not an exemption granted per family; it is what the one rule says about a
block whose rows declare no value keys, and it is asserted here BY NAME so a family that later
declares them cannot arrive unpartitioned - and so the two that just did cannot quietly un-declare
them.

The projection DROPS the four keys where `partition_factor` shadows a value to `None`. The
divergence is deliberate and it is the tick guard's own ruling read back: a pillar that starts or
stops being quoted two-sided is the same node of the same plan, so `Quoted_Bid` key-PRESENCE is
value-plane, and the mid's key goes with it for one uniform projection equal to the guard's.

ONE WIRE SPELLING PER TOKEN is the other thing a block has to survive, and the last section is
where `.DateOffset` lands: `CustomJsonEncoder` writes it as a STRING (`{'.DateOffset': '3M'}`) and
`Config.parse_json` wanted a kwargs DICT, so a `Market Prices` block written by this engine could
not be read back by one of this engine's own two decoders. Both read the string now, through one
spelling of the parse.

Every block here is a real one: the ZAR strip `test_interest_rate_prices` authors off a known curve,
the USDZAR smile `derivus_bloomberg` normalises out of canned terminal observations, and the two
Heston-Nandi ladders the engine EMITS off the surface that smile builds. The three families with no
fixture anywhere in this repo are held to their own declarations instead, which is the stronger
statement for them: a family declaring no `Points` field has no block that could have a values half.
"""
import ast
import copy
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest

import derivus
from derivus import bootstrappers, schema, utils
from derivus.bootstrappers import (FXVolSurfaceParameters, HestonNandiComponentModelParameters,
                                   HestonNandiModelParameters, InterestRateCurveParameters)
from derivus.config import Config, CustomJsonEncoder, ModelParams, update_market_quote

from rates_world import BASE as RATES_BASE
from test_interest_rate_prices import authored_world
from test_quote_propagation import BLOCK as ZAR_BLOCK, CCY, bootstrapped, market, run, with_deals
from test_service import CLIENT, JSON, desk, desk_smile, dump, fx_vol_quotes, job  # noqa: F401

FX_BLOCK = 'FXVolPrices.USD.ZAR'
HN_BLOCK = 'HestonNandiModelPrices.ZAR'

#: The family enumeration is `emit_market_prices`' own predicate, so this dict cannot fall behind
#: the store: a seventh family arriving with no row here fails the first gate rather than partitioning
#: silently.
FAMILIES = {cls.__dict__['market_factor_type']: cls
            for cls in vars(bootstrappers).values()
            if isinstance(cls, type) and isinstance(cls.__dict__.get('fields'), list)
            and 'market_factor_type' in cls.__dict__}

#: Where each family's quotes actually live, spelled BY NAME and confirmed against every fixture
#: below. FOUR of these tables carry the value plane - the two `Points` families and the two
#: Heston-Nandi option tables - and the other three have an empty values half and stay wholly
#: plan-side, which is the one rule and not three exceptions to it.
QUOTE_CONTAINER = {'CSForwardPriceModelPrices': 'Energy_Futures_Options',
                   'FXVolPrices': 'Points',
                   'GBMAssetPriceTSModelPrices': None,
                   'HestonNandiComponentModelPrices': 'European_Options',
                   'HestonNandiModelPrices': 'European_Options',
                   'HullWhite2FactorModelPrices': 'Instrument_Definitions',
                   'InterestRatePrices': 'Points'}

#: Which of those tables the value plane reaches, spelled here rather than read off
#: `schema.MARKET_QUOTE_CONTAINERS` - the derivation is what these gates are holding, so a gate
#: that asked it which families it covers would be asking the answer to grade itself.
VALUE_PLANE = ('European_Options', 'Points')

_BLOCKS = {}


def canonical(obj):
    """The encoder every hash in the engine is taken through. A round trip that survives this is
    byte-identical where it counts, whatever order the keys came back in."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), cls=CustomJsonEncoder)


def loaded(market_prices, deals=None):
    """A `Context` over a real job carrying `market_prices` - posted as text and rebuilt by the
    decoder, so the blocks these gates hash are the ones a job actually holds."""
    document = job(sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}},
                             'Market Prices': market_prices})
    return derivus.Context().load_json((dump(document), 'partition'))


def family_blocks():
    """One REAL block per declared family, built once.

    Four come from fixtures the suite already builds: the ZAR strip authored off a known curve, the
    USDZAR smile normalised out of canned terminal observations, and the two Heston-Nandi ladders
    the engine EMITS off the `FXVol` surface that smile bootstraps - real entry points throughout,
    nothing hand-written to pass.

    `CSForwardPriceModelPrices`, `GBMAssetPriceTSModelPrices` and `HullWhite2FactorModelPrices` have
    no fixture in this repo at all. What stands for them is their own declarations completed by
    `schema.declared_defaults`, which is the stronger statement anyway: none of the three declares a
    `Points` field, so no block of them can carry a values half.

    Built ONCE and handed out as a DEEP COPY, because a cached live reference is a fixture the
    session can edit: a gate that shortens a row list, or a family block that reaches a document by
    reference, would leave the next gate reading something no fixture ever built. The block count
    each one carries is asserted here rather than downstream, so a fixture that arrives empty fails
    at the source instead of turning a refusal gate VACUOUS - a row-count refusal over zero rows
    asserts nothing and passes.
    """
    if not _BLOCKS:
        market_prices, _, _ = authored_world('zar')
        _BLOCKS['InterestRatePrices'] = market_prices[ZAR_BLOCK]
        _BLOCKS['FXVolPrices'] = fx_vol_quotes()[FX_BLOCK]

        # the smile a desk posts, bootstrapped into the surface both ladders are authored off
        surface = loaded(desk_smile()).current_cfg
        surface.bootstrap()
        assert 'FXVol.USD.ZAR' in surface.params['Price Factors'], 'the surface did not build'
        for family in (HestonNandiModelParameters, HestonNandiComponentModelParameters):
            _BLOCKS[family.market_factor_type] = family.fx_surface_block(
                'USD.ZAR', surface.params['Price Factors'], surface.params['System Parameters'],
                surface.params['Price Factor Interpolation'])[1]

        # everything standing here came from a fixture; the three completed from declarations below
        # carry an empty table by construction and are asserted as that by the round-trip gate
        for name in sorted(_BLOCKS):
            assert len(_BLOCKS[name]['instrument'].get(QUOTE_CONTAINER[name]) or []) > 1, (
                '{}: the fixture reached this file with no quotes in {} - every gate below would '
                'be measuring an empty table'.format(name, QUOTE_CONTAINER[name]))

        for name, cls in FAMILIES.items():
            _BLOCKS.setdefault(name, {'instrument': schema.declared_defaults(cls, {})})
    return copy.deepcopy(_BLOCKS)


# ---------------------------------------------------------------------------------------------
# The projection - one rule, every family, and its exact inverse
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('family', sorted(QUOTE_CONTAINER))
def test_every_family_partitions_by_the_one_rule_and_round_trips_exactly(family):
    """ROUND TRIP per family, and the values half named for what it is.

    `apply_market_values(*partition_market_price(b))` re-encodes byte-identically through the
    canonical encoder - the same encoder `content_hash` takes every plan and every values hash
    through, so a round trip that survives it is one no hash can tell from the original.

    The second half is the uniformity claim, and it is stated in each family's own table. A family
    whose quote row DECLARES the four value keys has a values half with one entry per row - the two
    `Points` families and, since the option row grew its two-way, the two Heston-Nandi ladders. A
    family whose row does not declare them has an EMPTY one and is asserted as that BY NAME, because
    "this family has no values half" is a statement about the rule rather than a gap in it. Three of
    the seven are in that position and the enumeration is `emit_market_prices`' own, so neither a
    family that grows a declared quote table nor one that drops one can pass without moving a name
    here.
    """
    assert set(FAMILIES) == set(QUOTE_CONTAINER), (
        'a family arrived or left without this enumeration moving: {}'.format(
            set(FAMILIES) ^ set(QUOTE_CONTAINER)))
    container = QUOTE_CONTAINER[family]
    block = family_blocks()[family]
    structural, values = schema.partition_market_price(block)

    assert canonical(schema.apply_market_values(structural, values)) == canonical(block), (
        '{}: the partition is not invertible'.format(family))
    if container in VALUE_PLANE:
        assert schema.quote_rows(block['instrument'])[0] == container, (
            '{}: the partition found its quotes somewhere other than {}'.format(family, container))
        assert len(values) == len(block['instrument'][container]), (
            '{}: the values half does not line up row for row'.format(family))
        assert all('Quoted_Market_Value' in row for row in values), (
            '{}: a quoted row reached the values half without its mid'.format(family))
        assert not any(set(row) - set(schema.MARKET_QUOTE_VALUES) for row in values), (
            '{}: something structural is on the values side'.format(family))
        # a PARTITION, not two overlapping projections: every key of every row is on exactly one
        # side of the line, which is what makes the round trip above an identity rather than a merge
        assert all(set(row).isdisjoint(kept) and set(row) | set(kept) == set(point)
                   for row, kept, point in zip(values, structural['instrument'][container],
                                               block['instrument'][container])), (
            '{}: the two halves do not partition the row'.format(family))
    else:
        assert values == [], (
            '{}: quotes live in {}, whose row declares no value keys, so this family has no values '
            'half - and has one'.format(family, container))
        assert schema.quote_rows(block['instrument']) == (None, None), (
            '{}: {} reached the value plane without declaring the value columns'.format(
                family, container))
        assert canonical(structural) == canonical(block), (
            '{}: a wholly plan-side family lost something to the projection'.format(family))


def test_the_fx_authored_ladders_name_no_funding_curve():
    """THE BIT-IDENTITY BAR ON THE FX ROUTE, as the one thing that could move it.

    The Heston-Nandi families grew a `Funding_Rate` reference so an equity's forward can grow on its
    own repo curve while the premium discounts on another. An FX pair needs none: `fx_surface_block`
    already names the pair's OWN two curves - `Discount_Rate` is the domestic and `Yield` the
    foreign, which is exactly what `utils.calc_fx_forward` builds the forward from - so the emitter
    writes no funding curve and the fit's basis term is not evaluated at all. Every number an FX
    ladder has ever produced is the same bits.

    Asserted on the blocks the engine EMITS off a real built surface, both spellings, so an emitter
    that started declaring one would have to come through here.
    """
    for family in ('HestonNandiModelPrices', 'HestonNandiComponentModelPrices'):
        instrument = family_blocks()[family]['instrument']
        assert 'Funding_Rate' not in instrument and 'Funding_Rate_Type' not in instrument, (
            '{} declares a funding curve, so its forward is no longer the one it always built'
            .format(family))
        assert instrument['Discount_Rate'] and instrument['Yield'], (
            '{}: the pair\'s two curves are what the forward is built from'.format(family))
    assert 'Funding_Rate' in {f.key for f in HestonNandiModelParameters.fields}, (
        'the reference this gate says the FX route does not use is not declared at all')


def test_a_row_that_starts_being_quoted_two_sided_is_the_same_node_of_the_same_plan():
    """The divergence from `partition_factor`, stated as the property it buys.

    A price factor's key SET is structural - a value is shadowed to `None` there, because a field
    appearing or disappearing costs a recompile. A quote row's four value keys are DROPPED, because
    the tick guard already ruled that a pillar which starts or stops being quoted two-sided is the
    same node of the same plan: a spread widens between one print and the next. So adding a
    `Quoted_Bid` to a row that had none leaves the structural half bit-identical, and it is the
    values half that moves.
    """
    block = copy.deepcopy(family_blocks()['FXVolPrices'])
    plain = schema.partition_market_price(block)

    two_way = copy.deepcopy(block)
    for point in two_way['instrument']['Points']:
        point['Quoted_Bid'] = point['Quoted_Market_Value'] - 0.002
        point['Quoted_Ask'] = point['Quoted_Market_Value'] + 0.002
    widened = schema.partition_market_price(two_way)

    assert canonical(widened[0]) == canonical(plain[0]), (
        'a pillar that started being quoted two-sided re-authored the plan')
    assert canonical(widened[1]) != canonical(plain[1]), 'the sides reached neither half'
    assert canonical(schema.apply_market_values(*widened)) == canonical(two_way)


def test_a_short_values_half_refuses_rather_than_dropping_the_rows_it_cannot_pair():
    """Row ORDER is the only thing pairing the two halves, so a values half of the wrong length is
    not a short patch - it is a block that would come back shorter than the one it was projected
    from, with quotes gone and nothing said.

    `quote_delta` refuses first for every live caller, naming the block and both lengths, and that
    is the message a caller reads. This is the seam itself refusing, so silent row loss is
    unrepresentable rather than merely unreached: `apply_market_values` is public and its own
    docstring used to hand the length check to a caller two modules away. MUTANT: the bare
    `zip(instrument['Points'], values)` it shipped with returns 11 Points from a 12-row block on
    this fixture, no refusal and no log line anywhere.
    """
    block = family_blocks()['FXVolPrices']
    structural, values = schema.partition_market_price(block)
    assert len(values) > 1, 'this fixture cannot be shortened - nothing is being tested'

    with pytest.raises(ValueError) as refusal:
        schema.apply_market_values(structural, values[:-1])
    assert str(len(values)) in str(refusal.value) and str(len(values) - 1) in str(refusal.value), (
        'the refusal must name both lengths - a caller has no other way to find the row it lost')


# ---------------------------------------------------------------------------------------------
# Both directions: what moves `values_hash`, and what moves `plan_hash`
# ---------------------------------------------------------------------------------------------

def hashed():
    """One job carrying two families and two quote tables: the desk's USDZAR smile, which quotes in
    `Points`, and the `HestonNandiModelPrices` ladder the engine authors off the surface it builds,
    which quotes in `European_Options`. Both rows declare the four value keys, so BOTH tables are on
    the value plane - the same rule reaching two differently shaped blocks in one hash.

    The world states its own preconditions, because the gates below it degrade to VACUOUS rather
    than to red when it arrives empty: a row-count refusal over zero `Points` compares 0 against 0
    and passes, asserting nothing about a refusal. So the rows that came back through the decoder
    are counted against the rows that were posted before any gate reads them.
    """
    smile = desk_smile()
    context = loaded(dict(smile, **{HN_BLOCK: family_blocks()['HestonNandiModelPrices']}))
    prices = context.current_cfg.params['Market Prices']

    assert len(prices[FX_BLOCK]['instrument']['Points']) == len(
        smile[FX_BLOCK]['instrument']['Points']) > 1, (
        'the loaded smile is not the smile that was posted - this world is contaminated')
    assert len(prices[HN_BLOCK]['instrument']['European_Options']) > 1, (
        'the option ladder arrived with no quotes - there is nothing to move either way')
    return context, prices


def moves(context, before, edit):
    """`(plan moved, values moved)` after `edit` - both hashes read the same way for every case."""
    edit()
    return context.plan_hash() != before[0], context.values_hash() != before[1]


def test_a_quote_tick_moves_the_values_hash_and_leaves_the_plan_bit_identical():
    """The whole point of the change, in the coordinates the two staleness dimensions are named in.

    A mid, a bid and a `Timestamp` each move `values_hash` alone - which is what lets two-hash quote
    firmness read them as different questions. A `Quoted_Bid` appearing on a row that had none is on
    the same side, per the guard's own ruling. The plan is bit-identical across all of it, which is
    the assertion that used to run the other way.

    DRIVEN OVER BOTH QUOTE TABLES, because the split is a rule about a declared row and not about a
    key called `Points`: the same three edits on the Heston-Nandi ladder's `European_Options` row
    land the same way, and that row's mid is the one that used to be plan-side.
    """
    # what each fixture's first row ALREADY carries, so a case is known to be a MOVE or an ARRIVAL
    # rather than whichever it happened to be: the smile is posted with a stamp and no two-way, the
    # surface-authored ladder with the mid alone
    carried = {FX_BLOCK: {'Quoted_Market_Value', 'Timestamp'},
               HN_BLOCK: {'Quoted_Market_Value'}}
    for block, container in ((FX_BLOCK, 'Points'), (HN_BLOCK, 'European_Options')):
        for field, value in (('Quoted_Market_Value', 0.16), ('Quoted_Bid', 0.15),
                             ('Timestamp', pd.Timestamp('2024-06-28 17:45'))):
            context, prices = hashed()
            before = (context.plan_hash(), context.values_hash())
            row = prices[block]['instrument'][container][0]
            assert (field in row) == (field in carried[block]), (
                'this fixture does not test what the case name says it does: {} {}'.format(
                    block, field))

            def edit():
                row[field] = value

            assert moves(context, before, edit) == (False, True), (
                '{}: {} is not on the values plane alone'.format(block, field))


@pytest.mark.parametrize('case,plan_too,values_too',
                         [('pillar', True, False), ('expiry', True, False), ('use', True, False),
                          ('weight', True, False), ('strike', True, False),
                          ('option quote', False, True), ('new row', True, True)])
def test_re_authoring_a_quote_set_moves_the_plan(case, plan_too, values_too):
    """The other direction, and it is the same rule read from the other end.

    A moved pillar or expiry, a moved STRIKE on the option ladder, a quote held out with `Use`, a
    row appended: each re-authors the instrument set the solve is posed over, so each is a plan of
    its own. `Weight` is the one worth having in one gate beside them - it is a number, and it is
    not a value, because the line is drawn at the DECLARED value columns of a quote row rather than
    at whether a thing looks like a price.

    `option quote` is the case that FLIPPED. The Heston-Nandi ladder's mid was plan-side while its
    row declared no two-way, so a re-quoted chain was a recompile; the row declares the four value
    keys now, so a moved premium is a value on `European_Options` exactly as it is on `Points` -
    while the strike beside it stays a re-authoring, which is what makes this a line and not an
    exemption.

    Appending a row is the one case that moves BOTH, and honestly so: a new quote is a new node AND
    a number nobody had. Pinned as both rather than waved past, because a gate demanding the values
    hash stand there would be asserting that a quote arrived without a price.
    """
    context, prices = hashed()
    before = (context.plan_hash(), context.values_hash())
    points = prices[FX_BLOCK]['instrument']['Points']
    ladder = prices[HN_BLOCK]['instrument']['European_Options']
    edits = {'pillar': lambda: points[1].__setitem__('Pillar', 0.1),
             'expiry': lambda: points[0].__setitem__('Expiry', 0.3),
             'use': lambda: points[0].__setitem__('Use', 'No'),
             'weight': lambda: ladder[0].__setitem__('Weight', 0.5),
             'strike': lambda: ladder[0].__setitem__('Strike', 19.0),
             'new row': lambda: points.append(dict(points[0])),
             'option quote': lambda: ladder[0].__setitem__('Quoted_Market_Value', 0.19)}

    assert moves(context, before, edits[case]) == (plan_too, values_too), (
        '{} did not land where the plan/values line puts it'.format(case))


# ---------------------------------------------------------------------------------------------
# The patch: delta semantics, and every refusal by name
# ---------------------------------------------------------------------------------------------

def patchable():
    context, prices = hashed()
    return context, prices[FX_BLOCK]['instrument']['Points']


def rows(points, **fields):
    """A `Points` patch naming `fields` on the FIRST row and nothing on the rest - which is the
    delta a streaming tick actually sends."""
    return {'Points': [dict(fields)] + [{} for _ in points[1:]]}


def test_a_quote_patch_replaces_what_it_names_and_keeps_what_it_omits():
    """DELTA semantics, the same ones a price factor's patch has: a named field replaces, an
    omitted one keeps its current content. So one moved pillar is a complete, valid patch and a tick
    never has to resend the surface."""
    context, points = patchable()
    was = copy.deepcopy(points[1])

    context.patch_market({FX_BLOCK: rows(points, Quoted_Market_Value=0.19, Quoted_Bid=0.188)})
    patched = context.current_cfg.params['Market Prices'][FX_BLOCK]['instrument']['Points']

    assert patched[0]['Quoted_Market_Value'] == 0.19 and patched[0]['Quoted_Bid'] == 0.188
    assert patched[0]['Timestamp'] == points[0]['Timestamp'], 'an omitted field did not keep'
    assert canonical(patched[1]) == canonical(was), 'an unnamed row moved'


def first_row(context):
    """Row zero of the patched block as it stands in the context NOW - `patch_market` replaces the
    block rather than editing it, so a gate reading a row it captured earlier reads the old one."""
    return context.current_cfg.params['Market Prices'][FX_BLOCK]['instrument']['Points'][0]


def test_a_null_clears_a_two_way_side_and_refuses_on_the_mid():
    """`null` is what a source says "this pillar is no longer quoted two-sided" with, and the guard
    already ruled that is a VALUE. So it clears a side or a `Timestamp` - the key leaves the row -
    and it refuses on `Quoted_Market_Value`, which has no such reading: a mid is moved, never
    removed, and a block with a hole where a quote was is not a market.

    WHICH keys those are is read off `schema.MARKET_QUOTE_REQUIRED` rather than spelled here, and
    every declared value key is driven through both readings, so the gate asserts the DECLARATION
    and not a copy of it. That is the shape the rule needed: "the mid is mandatory, the sides and
    the stamp are optional" used to live as `field == 'Quoted_Market_Value'` inside `quote_delta`
    and in prose in three other modules, so a fifth value key became null-clearable with nobody
    deciding. TWO MUTANTS, both measured here: `quote_delta` back to a hand-named set that
    disagrees with the declaration (`('Quoted_Market_Value', 'Quoted_Ask')`) dies on the clear
    branch, and a mis-DECLARATION (`MARKET_QUOTE_REQUIRED` gaining `Quoted_Bid`) dies on the
    survival assertion, because clearing the ask must leave every required key standing.
    """
    seeds = {'Quoted_Market_Value': 0.19, 'Quoted_Bid': 0.188, 'Quoted_Ask': 0.192,
             'Timestamp': pd.Timestamp('2024-06-28 17:45')}
    assert set(seeds) == set(schema.MARKET_QUOTE_VALUES), (
        'a declared value key with no reading here: {}'.format(
            set(seeds) ^ set(schema.MARKET_QUOTE_VALUES)))
    assert set(schema.MARKET_QUOTE_REQUIRED) <= set(schema.MARKET_QUOTE_VALUES), (
        'a key declared required is not a value key at all')

    for field in schema.MARKET_QUOTE_VALUES:
        context, points = patchable()
        # seeded first, so a clear is a key LEAVING the row rather than an absence staying absent
        context.patch_market({FX_BLOCK: rows(points, **{field: seeds[field]})})
        assert first_row(context)[field] == seeds[field], '{}: the seed never landed'.format(field)

        if field in schema.MARKET_QUOTE_REQUIRED:
            with pytest.raises(ValueError, match='{} cannot be cleared'.format(field)):
                context.patch_market({FX_BLOCK: rows(points, **{field: None})})
            assert first_row(context)[field] == seeds[field], (
                '{}: the refused clear moved the row anyway'.format(field))
        else:
            context.patch_market({FX_BLOCK: rows(points, **{field: None})})
            assert field not in first_row(context), '{}: null did not clear it'.format(field)
            assert set(schema.MARKET_QUOTE_REQUIRED) <= set(first_row(context)), (
                '{}: clearing it took a required key with it'.format(field))

    # and the two-way side plus its stamp in ONE patch, which is the shape a source that stopped
    # printing a spread actually sends
    context, points = patchable()
    context.patch_market({FX_BLOCK: rows(points, Quoted_Bid=0.13)})
    context.patch_market({FX_BLOCK: rows(points, Quoted_Bid=None, Timestamp=None)})
    cleared = first_row(context)

    assert 'Quoted_Bid' not in cleared and 'Timestamp' not in cleared
    assert 'Quoted_Market_Value' in cleared, 'clearing a side took the mid with it'


@pytest.mark.parametrize('field', ['Quoted_Bid', 'Quoted_Market_Value'])
def test_a_null_in_a_document_is_an_absence_and_the_values_plane_is_an_identity_over_it(field):
    """`patch_market(market_patch())` is the values-plane IDENTITY, and a `null` in the document is
    the shape that used to break it in both directions at once.

    A source with no print for a pillar posts the key holding `null` - `update_market_quote`
    installs it, because both keys are value-side and the guard has no objection. Read as a value
    present, `market_patch` published the null verbatim and `quote_delta` read it back as a CLEAR:
    the round trip DROPPED a `Quoted_Bid` and `values_hash` moved for a no-op, and on a null MID it
    raised `Quoted_Market_Value cannot be cleared` against a patch `market_patch` itself produced.

    So a null is an absence on the values plane, stated once in `partition_market_price`. The
    values half skips it; the row keeps it, because a patch that does not NAME a field keeps that
    field's current content and null is what the document says the content is. Both hashes stand.
    """
    context, points = patchable()
    points[0][field] = None
    before = (context.plan_hash(), context.values_hash())

    emitted = context.market_patch()[FX_BLOCK]['Points'][0]
    assert field not in emitted, 'a null was published as a value the plane has to carry'

    context.patch_market(context.market_patch())
    landed = first_row(context)

    assert (context.plan_hash(), context.values_hash()) == before, (
        'the values-plane identity moved a hash')
    assert field in landed and landed[field] is None, (
        'a field the patch never named did not keep its content')
    assert canonical(landed) == canonical(points[0]), 'the identity re-authored the row'


def test_a_patch_that_changes_the_row_count_refuses_naming_both_lengths():
    """A quote added or dropped re-authors the instrument set, so it is a new plan and not a short
    patch - and the refusal has to say WHICH two lengths disagree, because a caller streaming a
    surface has no other way to find the row it lost.

    The row count is asserted BEFORE the refusal is asked for: over an empty block this gate
    compares zero against zero, takes no refusal, and passes - degrading to vacuous rather than to
    red, which is the failure mode a refusal gate cannot afford.
    """
    context, points = patchable()
    assert len(points) > 1, 'a short patch over this block is not short - nothing is being refused'
    with pytest.raises(ValueError) as refusal:
        context.patch_market({FX_BLOCK: {'Points': [{} for _ in points[1:]]}})

    assert str(len(points)) in str(refusal.value) and str(len(points) - 1) in str(refusal.value)
    assert 'new plan' in str(refusal.value)


@pytest.mark.parametrize('patch,named', [({'Grid_Tolerance': 1e-6}, 'Grid_Tolerance'),
                                         ({'Points': [{'Pillar': 0.1}]}, 'Pillar'),
                                         ({'Points': [{'Use': 'No'}]}, 'Use')])
def test_a_structural_key_refuses_by_name(patch, named):
    """Every key that is not one of the four is structure, block-level and row-level alike, and the
    refusal names the key rather than the block: a caller sending a whole row back needs to know
    which field of it was the plan."""
    context, points = patchable()
    if 'Points' in patch:
        patch = {'Points': patch['Points'] + [{} for _ in points[1:]]}
    with pytest.raises(ValueError, match='{} is structural, not a value'.format(named)):
        context.patch_market({FX_BLOCK: patch})


def test_a_name_in_neither_section_refuses_naming_both():
    """`patch_market` now spans two sections, so a name it cannot place has to say both - a caller
    that mistyped a factor and a caller that mistyped a quote block get the same message and it
    tells each of them where to look."""
    context, _ = patchable()
    with pytest.raises(KeyError, match='neither a price factor nor a market price'):
        context.patch_market({'FXVolPrices.EUR.USD': {'Points': []}})


# ---------------------------------------------------------------------------------------------
# The guard and the partition are one declaration
# ---------------------------------------------------------------------------------------------

def test_the_tick_guard_and_the_partition_draw_the_same_line():
    """GUARD PARITY, driven both ways with the same doctored blocks.

    `update_market_quote` is where this split was first written, as a refusal; the partition is the
    same split as a projection. They now read one tuple, and this is what says so without asserting
    it about the tuple: a value-only re-post passes the guard AND leaves the structural half
    identical, and a moved pillar refuses the guard AND moves the structural half. Two statements
    that could disagree, on one pair of blocks.

    THE TICKED BLOCK IS BUILT OFF THE DECLARATION, one mover per member, because a gate that names
    the fields by hand tests the fields it happens to name. MUTANT: `structure()` back to its own
    inline tuple with one field misspelled. Measured on the hand-named version, which moved the mid
    and added a bid: `Quoted_Market_Value` and `Quoted_Bid` died here, `Quoted_Ask` survived this
    file and died in `test_service`, and `Timestamp` survived this file, `test_service` and
    `test_mcp` together - 118 passed with the guard silently reading a key nothing posts. Every
    member is moved here now, and asserted to have MOVED, so a fifth value key cannot arrive
    untested and a stamp nobody re-posts cannot go uncovered.
    """
    document = json.loads(dump(job()))
    quotes = json.loads(dump(fx_vol_quotes()))[FX_BLOCK]
    assert update_market_quote(document, FX_BLOCK, quotes) == 'installed'
    installed = copy.deepcopy(quotes)

    # through the real encoder, because the guard compares WIRE forms and a Timestamp reaches it as
    # the token a posted document carries rather than as an object
    later = json.loads(dump({'stamp': pd.Timestamp('2024-06-28 17:45')}))['stamp']
    # each mover takes the row's NEW mid, so the two-way straddles the mid it is quoted around
    movers = {'Quoted_Market_Value': lambda mid: mid,
              'Quoted_Bid': lambda mid: mid - 0.002,
              'Quoted_Ask': lambda mid: mid + 0.002,
              'Timestamp': lambda mid: later}
    assert set(movers) == set(schema.MARKET_QUOTE_VALUES), (
        'a declared value key this gate never moves: {}'.format(
            set(movers) ^ set(schema.MARKET_QUOTE_VALUES)))

    ticked = copy.deepcopy(quotes)
    for point in ticked['instrument']['Points']:
        ticked_mid = point['Quoted_Market_Value'] + 0.01
        point.update({field: mover(ticked_mid) for field, mover in movers.items()})
    assert all(was.get(field) != now.get(field) for field in schema.MARKET_QUOTE_VALUES
               for was, now in zip(installed['instrument']['Points'],
                                   ticked['instrument']['Points'])), (
        'a value field was posted back at the content it already held - the guard cannot see it')
    assert update_market_quote(document, FX_BLOCK, ticked) == 'updated'
    assert (canonical(schema.partition_market_price(ticked)[0]) ==
            canonical(schema.partition_market_price(installed)[0])), (
        'the guard took a tick the partition reads as a re-authoring')

    moved = copy.deepcopy(ticked)
    for point in moved['instrument']['Points']:
        point['Pillar'] = 0.1
    with pytest.raises(ValueError, match='structure differs'):
        update_market_quote(document, FX_BLOCK, moved)
    assert (canonical(schema.partition_market_price(moved)[0]) !=
            canonical(schema.partition_market_price(ticked)[0])), (
        'the guard refused a block the partition reads as the same plan')


def test_the_artifact_slot_projects_away_exactly_the_declared_value_fields():
    """THE THIRD SIBLING. `CalibrationArtifact`'s slot was shadowing quote numbers out of its key
    with a copy of this rule; it now calls the partition, which DROPS them, and this is the parity
    gate that holds it there - asserted against the declaration rather than against prose. Only the
    `lifecycle_fields` are still shadowed, and the two words are not interchangeable: a shadow keeps
    the key and a projection does not, which is exactly what lets a row gain a `Quoted_Bid` without
    moving its slot.

    Every key a `Points` row carries, plus the four the declaration names, is moved in turn: the
    slot has to survive exactly the four and move on everything else. The pairing is the mutation -
    a `plan_key` that ignored the whole row would pass the first half and fail the second, and one
    that hashed the row whole would fail the first.

    `Quoted_Bid`, `Quoted_Ask` and `Timestamp` are not on an `InterestRatePrices` row at all, so
    those three cases ADD a key - which is the drop-not-shadow rule reaching this sibling: a strip
    that starts carrying two-way quotes keeps riding the operator it was fitted with, because the
    solve reads none of them.
    """
    block = authored_world('zar')[0][ZAR_BLOCK]['instrument']

    def slot(candidate):
        return InterestRateCurveParameters.plan_key(
            [(ZAR_BLOCK, candidate)], ModelParams(), RATES_BASE)

    base = slot(block)
    for field in sorted(set(block['Points'][0]) | set(schema.MARKET_QUOTE_VALUES)):
        doctored = copy.deepcopy(block)
        doctored['Points'][0][field] = 'MOVED'
        assert (slot(doctored) == base) == (field in schema.MARKET_QUOTE_VALUES), (
            '{} is on the wrong side of the artifact slot'.format(field))


def test_the_bloomberg_snapshot_guard_cannot_drift_from_the_engines_declaration():
    """THE FOURTH SIBLING, and the one that cannot be unified: `derivus_bloomberg` enforces this
    same split snapshot-side and CANNOT import the engine - blpapi lives on a terminal workstation
    and the package has to stand alone. So it keeps its copy, and the copy is gated.

    Read as source rather than imported, because what is being compared is a WHITELIST against a
    blacklist: `_structure` names the fields that are structure, this file names the fields that are
    values, and the declaration is what says the two are complements. A field added to the block on
    either side now has to appear on the other or this fails.
    """
    root = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    structure = next(
        node for node in ast.walk(ast.parse((root / 'derivus_bloomberg' / 'fxvol.py').read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == '_structure')
    named = {node.value for node in ast.walk(structure)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)}

    declared = {f.key for f in FXVolSurfaceParameters.fields}
    row = {f.key for f in next(f for f in FXVolSurfaceParameters.fields
                               if f.key == 'Points').row.fields}

    assert row - named == set(schema.MARKET_QUOTE_VALUES), (
        'the snapshot guard and MARKET_QUOTE_VALUES disagree on a quote row: {}'.format(
            (row - named) ^ set(schema.MARKET_QUOTE_VALUES)))
    assert declared - named == set(), (
        'a block field the engine declares is not structure snapshot-side: {}'.format(
            declared - named))


# ---------------------------------------------------------------------------------------------
# What a quote patch does NOT do
# ---------------------------------------------------------------------------------------------

def test_a_patched_quote_reprices_nothing_and_says_so_in_the_hashes():
    """NO-BOOTSTRAP HONESTY. A quote patch moves the quotes and nothing else: the price factors the
    last bootstrap wrote stand exactly where they are, so a job that does not ride reports the same
    numbers to the bit.

    That is not a gap being papered over - it is what `values_hash` is FOR. The board that moved is
    recorded, the marks that did not move are the marks, and the consumer that turns one into the
    other is the ride. A patch that quietly re-bootstrapped would make a values patch a plan
    operation and cost the tick the whole solve it was written to avoid.
    """
    context = derivus.Context()
    context.current_cfg = with_deals(bootstrapped(market(False)))
    before = (context.plan_hash(), context.values_hash())
    curve = context.current_cfg.params['Price Factors']['InterestRate.ZAR-JIBAR-3M']['Curve']
    tables = canonical(run(with_deals(context.current_cfg))[1]['Results'])

    points = context.current_cfg.params['Market Prices'][ZAR_BLOCK]['instrument']['Points']
    context.patch_market({ZAR_BLOCK: {'Points': [
        {'Quoted_Market_Value': point['Quoted_Market_Value'] + 0.25} for point in points]}})

    assert context.plan_hash() == before[0], 'a moved quote moved the plan'
    assert context.values_hash() != before[1], 'a moved quote left the values hash alone'
    assert context.current_cfg.params['Price Factors'][
        'InterestRate.ZAR-JIBAR-3M']['Curve'] is curve, 'the patch re-bootstrapped'
    assert canonical(run(with_deals(context.current_cfg))[1]['Results']) == tables, (
        'the marks moved without a bootstrap or a ride between them')


def test_the_live_book_refuses_a_quote_as_a_values_patch(desk):
    """The one place the book is STRICTER than the engine, and why.

    `patch_market` takes a quote because an EXECUTE that rides re-derives its curve from it - the
    patch and the consumer are in the same call. A live book has no such step: a patched quote would
    leave the price factors on disk standing against quotes they no longer solve, and the book's
    whole invariant is that its marks and its quotes came out of one atomic write. So the values
    path refuses it and names the path that does bootstrap.
    """
    installed = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes()}),
                            headers=JSON).json()
    assert installed['installed'] == [FX_BLOCK] and installed['written'] is True

    before = desk.read_bytes()
    refused = CLIENT.post('/book/market', content=dump(
        {'patch': {FX_BLOCK: {'Points': [{'Quoted_Market_Value': 0.2}]}}}), headers=JSON)

    assert refused.status_code == 422
    assert FX_BLOCK in refused.json()['detail']
    assert '`quotes`' in refused.json()['detail'], 'the refusal must name the remedy'
    assert desk.read_bytes() == before, 'the refused patch still touched the book'


# ---------------------------------------------------------------------------------------------
# One wire spelling of `.DateOffset`, and both decoders read it
# ---------------------------------------------------------------------------------------------

def offsets_in(node, path=''):
    """`[(path, value)]` for every `pd.DateOffset` in a decoded structure, in a stable order.

    Walked rather than named: the ZAR strip carries its tenors inside authored `Deal` blocks
    several levels down, and a gate that named the paths would be gating the paths it happened to
    name. A `.DateOffset` the decoder left as a raw string or dict is NOT a `DateOffset` and so is
    collected as itself, which is what makes the type assertion below able to fail.
    """
    if isinstance(node, dict):
        if set(node) & {'.DateOffset'}:
            return [(path, node)]
        return [item for key in sorted(node) for item in offsets_in(node[key], path + '/' + key)]
    if isinstance(node, list):
        return [item for i, entry in enumerate(node)
                for item in offsets_in(entry, '{}[{}]'.format(path, i))]
    return [(path, node)] if isinstance(node, pd.DateOffset) else []


def test_one_dateoffset_wire_spelling_and_both_decoders_read_it(tmp_path):
    """THE ROW THIS CLOSES: `CustomJsonEncoder` writes `{'.DateOffset': '3M'}` and
    `Config.parse_json` did `DateOffset(**dct['.DateOffset'])`, which wants a kwargs dict - so a
    `MarketData.json` this engine WROTE could not be read back through `parse_json` at all, and the
    emitters (which write the encoder's spelling, because the service path is `read_json`'s) were
    emitting bytes one of the two decoders would die on.

    ONE SPELLING OF THE PARSE, not a second copy: `parse_json` routes the string through
    `Config.parse_period`, which is the expression `read_json` already used, so the two cannot drift
    again. The kwargs dict is still accepted and that is deliberate - reading a spelling nobody
    writes any more is free, writing two of them was the defect.

    THE BLOCK IS A REAL ONE, the same ZAR strip the partition gates run on, and it carries 38
    `.DateOffset` sites through authored `Deal` blocks. Whole-block equality is asserted through
    `canonical` rather than `==` for one reason worth recording: `utils.DateList` defines no
    `__eq__`, so a `==` over the whole structure compares two decoded schedules by IDENTITY and is
    False for any two decodes of anything. The offsets themselves ARE compared with `==`, which is
    the comparison this row is about.
    """
    block = family_blocks()['InterestRatePrices']
    market = {'System Parameters': {'Base_Date': RATES_BASE, 'Base_Currency': 'ZAR'},
              'Price Factors': {}, 'Correlations': {},
              'Market Prices': {ZAR_BLOCK: block}}
    text = json.dumps({'MarketData': market, 'Version': ['JSONVersion', '22.05.30']},
                      cls=CustomJsonEncoder)
    assert '"' + '.DateOffset": "' in text, 'the encoder no longer writes the string form'
    assert '.DateOffset": {' not in text, 'the encoder writes a kwargs dict somewhere'

    path = tmp_path / 'MarketData.json'
    path.write_text(text, encoding='utf-8')
    parsed = Config()
    parsed.parse_json(str(path))
    read = Config().read_json((text, 'wire'))['MarketData']

    through_parse = offsets_in(parsed.params['Market Prices'])
    through_read = offsets_in(read['Market Prices'])
    assert len(through_parse) == text.count('.DateOffset') == 38, through_parse
    assert [where for where, _ in through_parse] == [where for where, _ in through_read]
    assert all(isinstance(value, pd.DateOffset) for _, value in through_parse), (
        'parse_json left a .DateOffset undecoded: {}'.format(
            [item for item in through_parse if not isinstance(item[1], pd.DateOffset)]))
    assert [value for _, value in through_parse] == [value for _, value in through_read]
    assert canonical(parsed.params['Market Prices']) == canonical(read['Market Prices'])
    assert canonical(parsed.params['Market Prices']) == canonical({ZAR_BLOCK: block}), (
        'the block did not survive its own encoder')


def test_the_kwargs_dict_still_reads_because_old_bytes_are_on_disk(tmp_path):
    """The legacy spelling `parse_json` was written for keeps working, and it decodes to the SAME
    object the string does - which is what makes accepting both a compatibility statement rather
    than a second convention. Nothing in the tree writes it: `CustomJsonEncoder` is the only writer
    of this key and the emitters copy its output, so this arm is about bytes already on disk.

    A SEPARATE FINDING, named here because this gate is where it surfaced and it is NOT fixed:
    `CustomJsonEncoder` builds the string by walking `DateOffset.kwds`, whose key order for a
    MULTI-UNIT period is a set iteration and therefore varies with the interpreter's hash seed -
    `DateOffset(months=6, days=2)` encodes `'6M2D'` in one process and `'2D6M'` in the next
    (measured 4:1 over five fresh interpreters). Both parse back to the same offset, so nothing
    reads wrong; what is not byte-stable across processes is `write_marketdata_json`'s output and
    any hash taken over such a block. Every offset the emitters write is single-unit, which is why
    no determinism gate in the repo has seen it. This gate asserts the set of parts rather than
    their order, so it states what is true rather than pinning a coin flip."""
    legacy = json.dumps({'MarketData': {
        'System Parameters': {}, 'Price Factors': {}, 'Correlations': {},
        'Market Prices': {'legacy': {'instrument': {
            'Start': {'.DateOffset': {'years': 1}},
            'Tenor': {'.DateOffset': {'months': 6, 'days': 2}}}}}},
        'Version': ['JSONVersion', '22.05.30']})
    path = tmp_path / 'Legacy.json'
    path.write_text(legacy, encoding='utf-8')
    old = Config()
    old.parse_json(str(path))
    instrument = old.params['Market Prices']['legacy']['instrument']

    assert instrument['Start'] == pd.DateOffset(years=1)
    assert instrument['Tenor'] == pd.DateOffset(months=6, days=2)
    # and the string form of the same two offsets, through the same decoder, is the same object
    written = json.dumps({'MarketData': {
        'System Parameters': {}, 'Price Factors': {}, 'Correlations': {},
        'Market Prices': {'legacy': {'instrument': instrument}}},
        'Version': ['JSONVersion', '22.05.30']}, cls=CustomJsonEncoder)
    assert '{".DateOffset": "1Y"}' in written, written
    assert json.loads(written)['MarketData']['Market Prices']['legacy']['instrument'][
        'Tenor']['.DateOffset'] in ('6M2D', '2D6M'), written
    again = tmp_path / 'Written.json'
    again.write_text(written, encoding='utf-8')
    new = Config()
    new.parse_json(str(again))
    assert new.params['Market Prices'] == old.params['Market Prices']
