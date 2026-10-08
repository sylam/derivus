"""A position prices at its units of the deal as written.

`spine.scaled` writes a node HELD q times: every field its type declares `sized` - its legs'
included - multiplied, and a negative q taken as the MIRROR, every `side` its type declares
flipped with the sizes at q's magnitude, or the sizes signed where it declares none. A position is units of
the instrument, so the trial is the identity that defines it, one per type: held at half a unit a
deal marks exactly half, and held short exactly minus. Both are asserted to the bit - a power of
two and a sign commute with every operation a payoff linear in its amounts makes, whatever the
pricer, a Monte Carlo strip included - and every type a position can be held in declares its sizes
beside its trial. The node scaled is the WIRE form, which is what the compile reads off a book file.

The trials are filed by family, one module each (`trial_<family>.py`): `DEALS`, one deal of every
type the family declares with every amount stated and live, `FACTORS` and `CONFIGURATION` the world
they need beyond the declared-defaults book's, and `UNMARKED`, the rows a parent values.
"""

import copy
import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

import derivus
import rates_world
import test_declared_defaults as book
import trial_commodity
import trial_credit
import trial_equity
import trial_equity_swaps
import trial_fx
import trial_rates
from derivus import instruments, schema, spine
from derivus.config import CustomJsonEncoder

E, B = book.WORLD_EXPIRY, book.WORLD_BASE

#: Priced beside the declared-defaults book, in its world: one of every declared type that book
#: does not already carry, and a structure of two of them.
EXTRA = [
    {'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Amount': 250_000.0, 'Payment_Date': E},
    {'Object': 'FXForwardDeal', 'Reference': 'FWD', 'Buy_Currency': 'EUR', 'Sell_Currency': 'USD',
     'Buy_Amount': 1e6, 'Sell_Amount': 1.2e6, 'Settlement_Date': E, 'Buy_Discount_Rate': 'EUR',
     'Sell_Discount_Rate': 'USD'},
    {'Object': 'FXNonDeliverableForward', 'Reference': 'NDF', 'Buy_Currency': 'EUR',
     'Sell_Currency': 'USD', 'Buy_Amount': 1e6, 'Sell_Amount': 1.2e6, 'Settlement_Date': E,
     'Settlement_Currency': 'USD', 'Discount_Rate': 'USD'},
    {'Object': 'FXSwapDeal', 'Reference': 'FXSW', 'Near_Settlement_Date': B + pd.DateOffset(months=1),
     'Far_Settlement_Date': E, 'Near_Buy_Far_Sell_Ccy': 'EUR', 'Near_Sell_Far_Buy_Ccy': 'USD',
     'Near_Buy_Far_Sell_Discount_Rate': 'EUR', 'Near_Sell_Far_Buy_Discount_Rate': 'USD',
     'Near_Buy_Amount': 1e6, 'Near_Sell_Amount': 1.2e6, 'Far_Buy_Amount': 1.21e6,
     'Far_Sell_Amount': 1e6},
    rates_world._cashflow_leg(
        'CFFloatingInterestListDeal', 'FLOAT', 'USD', 'USD', 'Buy',
        {'Compounding_Method': 'None', 'Averaging_Method': 'Average_Rate', 'Properties': [],
         'Items': book.float_items()},
        Forecast_Rate='USD-PROJ', Rate_Adjustment_Method='None', Rate_Sticky_Month_End='Yes',
        Rate_Offset=0, Rate_Calendars=None, Accrual_Calendars=None,
        Forecast_Rate_Cap_Volatility='', Forecast_Rate_Swaption_Volatility='',
        Discount_Rate_Cap_Volatility='', Discount_Rate_Swaption_Volatility=''),
    {'Object': 'EquityOptionDeal', 'Reference': 'EQO', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Equity': 'EQ', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Put',
     'Strike_Price': 95.0, 'Expiry_Date': E, 'Units': 100.0},
    {'Object': 'EquityForwardDeal', 'Reference': 'EQF', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Equity': 'EQ', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Forward_Price': 98.0,
     'Maturity_Date': E, 'Units': 100.0},
    book.fx_leg('FXBinaryOption', 'BIN', Expiry_Date=E, Payoff=10_000.0),
    # a collar: its container is no option on its legs, so each leg is itself held and flips alone
    {'Object': 'StructuredDeal', 'Reference': 'COLLAR', 'Currency': 'USD', 'Children': [
        book.fx_leg('FXOptionDeal', 'COLLAR_PUT', Expiry_Date=E, Option_Type='Put',
                    Strike_Price=1.15),
        book.fx_leg('FXOptionDeal', 'COLLAR_CALL', Expiry_Date=E, Buy_Sell='Sell',
                    Strike_Price=1.35)]},
]

#: The declared-defaults book less what holds no position of its own - a netting set, whose CASHFLOW
#: LIST is held here in its place - with the types above beside it.
CORE = SimpleNamespace(
    DEALS=[deal for deal in book.BOOK if deal['Object'] != 'NettingCollateralSet'] + [
        leg for deal in book.BOOK if deal['Object'] == 'NettingCollateralSet'
        for leg in deal['Children']] + EXTRA,
    FACTORS={}, CONFIGURATION={},
    # a swaption's legs are valued by the swaption's own `post_process`, so what they hold here is
    # their scaling, which the swaption's own row carries
    UNMARKED={'SWPT_FIXED', 'SWPT_FLOAT'})

FAMILIES = {'core': CORE, 'rates': trial_rates, 'fx': trial_fx, 'equity': trial_equity,
            'equity_swaps': trial_equity_swaps, 'commodity': trial_commodity,
            'credit': trial_credit}

#: Containers whose legs carry the whole of their size, declaring none of their own. A netting set
#: holds no position at all: it is the agreement positions are held under.
HELD_BY_LEGS = {'StructuredDeal', 'MtMCrossCurrencySwapDeal'}


def node_of(deal):
    block = {key: value for key, value in deal.items() if key != 'Children'}
    node = {'Instrument': {'.Deal': block}}
    if deal.get('Children'):
        node['Children'] = [node_of(child) for child in deal['Children']]
    return node


def deal_of(node):
    deal = dict(node['Instrument']['.Deal'])
    if node.get('Children'):
        deal['Children'] = [deal_of(child) for child in node['Children']]
    return deal


def held(deal, quantity):
    return deal_of(spine.scaled(node_of(deal), quantity))


def references(deals):
    return [deal['Reference'] for deal in deals] + [
        reference for deal in deals for reference in references(deal.get('Children', []))]


def document(family):
    """The family's deals in the declared-defaults world and its own, as the WIRE JSON a book file
    carries."""
    job = book.book(family.DEALS)
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors'].update(family.FACTORS)
    market['Valuation Configuration'].update(family.CONFIGURATION)
    return json.loads(json.dumps(job, cls=CustomJsonEncoder))


def holding(job, quantity):
    """`job` with every deal it holds held `quantity` times, as the compile writes a position."""
    job = copy.deepcopy(job)
    nodes = job['Calc']['Deals']['Deals']['Children']
    nodes[:] = [spine.scaled(node, quantity) for node in nodes]
    return job


def marks(job):
    """`{reference: mark as hex}` for one wire document."""
    context = derivus.Context()
    context.load_json((json.dumps(job), 'trial'))
    _, answer = context.run_job()
    frame = answer['Results']['mtm']
    return {str(reference): float(value).hex()
            for reference, value in zip(frame['Reference'], frame['Value'])}


def nodes(children):
    for node in children:
        yield node
        yield from nodes(node.get('Children', []))


def declared_amounts(field, path=()):
    """Every part of a field its declaration sizes - the field, a sub-field, a table column."""
    path = path + (field.key,)
    if field.sized:
        return {path}
    return set().union(set(), *(declared_amounts(sub, path) for sub in field.sub_fields or ()),
                       *({path + (column.key,)} for column in (field.row.fields if field.row else ())
                         if column.sized))


def stated_amounts(field, value, path=()):
    """The parts of a wire `value` its declaration sizes that it states, and states non-zero."""
    path = path + (field.key,)
    if field.sized:
        return {path} if isinstance(value, (int, float)) and value else set()
    stated = set().union(set(), *(stated_amounts(sub, value[sub.key], path)
                                  for sub in field.sub_fields or ()
                                  if isinstance(value, dict) and sub.key in value))
    rows = value
    if isinstance(value, dict) and len(value) == 1 and next(iter(value)).startswith('.'):
        rows = next(iter(value.values()))
    for position, column in enumerate(field.row.fields if field.row and isinstance(rows, list)
                                      else ()):
        cells = [row.get(column.key) if isinstance(row, dict) else
                 row[position] if isinstance(row, list) and position < len(row) else None
                 for row in rows]
        if column.sized and any(isinstance(cell, (int, float)) and cell for cell in cells):
            stated.add(path + (column.key,))
    return stated


@pytest.mark.parametrize('name', sorted(FAMILIES))
def test_half_a_unit_marks_half_and_short_marks_minus_to_the_bit(name):
    """Every type a position can be held in, one family at a time: the half-size and short readings
    off the as-written one, bit for bit.

    Killing mutations: a declared size left out - the FX forward's sold amount unscaled, which
    halves its euro leg and not its dollar one - and a type declaring no side never signed, which
    leaves a swap's short reading equal to its long one. A swaption's row is valued off its own
    terms, so a flip wrongly carried into its legs is the rule's own test to see.
    """
    family = FAMILIES[name]
    assert family.DEALS, 'the {} family holds no trial'.format(name)
    job = document(family)
    whole, half, short = marks(job), marks(holding(job, 0.5)), marks(holding(job, -1.0))
    for reference in set(references(family.DEALS)) - family.UNMARKED:
        value = float.fromhex(whole[reference])
        assert math.isfinite(value) and value != 0.0, (reference, value)
        assert float.fromhex(half[reference]) == value / 2, (reference, half[reference], value)
        assert float.fromhex(short[reference]) == -value, (reference, short[reference], value)


def test_every_type_a_position_is_held_in_declares_its_sizes_beside_its_trial():
    """THE CENSUS. Every type the store publishes but the netting set either declares its sizes or
    is a container whose legs carry the whole of them, and every one is held in a trial. A size is
    a number, which is all `spine.scaled` moves, and a side is a field of two values.

    Killing mutations: a type's sizes left undeclared, a type no family trials, and a Percent
    declared a size, which the wire spells as a token no scaling reaches.
    """
    def parts(field):
        yield field
        for part in list(field.sub_fields or ()) + (field.row.fields if field.row else []):
            yield from parts(part)

    types = set(schema.mapping['Instrument']['types']) - {'NettingCollateralSet'}
    fields = {name: [part for field in schema.declared_fields(getattr(instruments, name)).values()
                     for part in parts(field)] for name in types}
    declared = {name for name in types if any(part.sized for part in fields[name])}
    trialled = {node['Instrument']['.Deal']['Object'] for family in FAMILIES.values()
                for node in nodes(document(family)['Calc']['Deals']['Deals']['Children'])}
    assert declared == types - HELD_BY_LEGS, sorted(declared ^ (types - HELD_BY_LEGS))
    assert types <= trialled, sorted(types - trialled)
    assert all(part.type in ('Float', 'Integer') for name in types for part in fields[name]
               if part.sized)
    assert all(len(part.values or ()) == 2 for name in types for part in fields[name] if part.side)


def test_every_declared_amount_is_stated_and_non_zero_in_a_trial():
    """A trial sees a declaration only where the amount it sizes is there to move, so every part a
    type declares sized - a scalar, a sub-field, a table column - is stated non-zero in some deal of
    that type, a leg included. The trial deal is where it must also be LIVE: an amount stated and
    never read reaches this and no mark.

    Killing mutation: an amortisation table left null on every deal carrying one.
    """
    declared, stated = {}, {}
    for family in FAMILIES.values():
        for node in nodes(document(family)['Calc']['Deals']['Deals']['Children']):
            deal = node['Instrument']['.Deal']
            fields = schema.declared_fields(getattr(instruments, deal['Object']))
            for key, field in fields.items():
                declared.setdefault(deal['Object'], set()).update(declared_amounts(field))
                stated.setdefault(deal['Object'], set()).update(
                    stated_amounts(field, deal[key]) if key in deal else set())
    missing = {name: sorted(paths - stated[name]) for name, paths in declared.items()
               if paths - stated[name]}
    assert not missing, missing


def test_a_swaption_holds_its_legs_as_terms():
    """Where the mirror is taken on a swaption. Held short it flips ITS side and keeps its
    underlying's legs as they were written, sized at the quantity's magnitude - the one placement
    no mark can see, the swaption's row being valued off its own terms; a collar's and a cap's
    legs, and a swap's signed principal and unsigned amortisation, mark through the half and short
    readings above.

    Killing mutation: the flip carried into a swaption's legs, which writes the short swaption on a
    receiver underlying (every mark unmoved: measured green on the half-and-short gate).
    """
    swaption = next(deal for deal in CORE.DEALS if deal['Reference'] == 'SWPT')
    short = held(swaption, -2.0)
    assert (short['Buy_Sell'], short['Principal']) == ('Sell', 2 * swaption['Principal'])
    assert [leg['Buy_Sell'] for leg in short['Children']] == [
        leg['Buy_Sell'] for leg in swaption['Children']]
    assert short['Children'][0]['Cashflows']['Items'][0]['Notional'] == \
        2 * swaption['Children'][0]['Cashflows']['Items'][0]['Notional']


def test_a_deal_stating_none_of_its_amounts_refuses_a_position_other_than_one_by_name():
    """A block stating none of the amounts its type declares - an option carrying no
    `Underlying_Amount` - would price one unit whatever the position is, so a position of anything
    else refuses naming the type and the deal, a leg inside a structure included. An empty frame
    declares no amount and holds nothing to size, so it does not refuse; a leaf declaring none -
    a type the store does not know - is one unit, so a count of it does.

    Killing mutations: the refusal dropped, which prices half such an option as a whole one; the
    refusal taken for an empty structure, whose half position would refuse every read; and every
    type declaring no amount let through, which prices two units as one.
    """
    empty = {'Object': 'StructuredDeal', 'Reference': 'EMPTY', 'Currency': 'USD'}
    assert deal_of(spine.scaled(node_of(empty), 0.5)) == empty
    unknown = {'Object': 'NoSuchDeal', 'Reference': 'ODD'}
    with pytest.raises(ValueError, match="NoSuchDeal 'ODD'"):
        spine.scaled(node_of(unknown), 2.0)
    bare = {key: value for key, value in book.fx_leg('FXOptionDeal', 'BARE', Expiry_Date=E).items()
            if key != 'Underlying_Amount'}
    for quantity in (0.5, -1.0):
        with pytest.raises(ValueError, match="FXOptionDeal 'BARE'"):
            spine.scaled(node_of(bare), quantity)
    structure = {'Object': 'StructuredDeal', 'Reference': 'MIXED', 'Currency': 'USD',
                 'Children': [book.fx_leg('FXOptionDeal', 'LEG', Expiry_Date=E), bare]}
    with pytest.raises(ValueError, match="FXOptionDeal 'BARE'"):
        spine.scaled(node_of(structure), 0.5)
