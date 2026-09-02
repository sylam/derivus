"""`derivus.quote_sheet` end to end: an outcome and a book in, one workbook out.

The fixture is a hand-authored zero-cost collar on USDZAR plus a book in the wire form a job file
carries, asserted loadable by the engine's own `load_json`. Nothing is stubbed; the workbook is
read back with `zipfile` and `ElementTree`.

Four claims, each a way a sheet can lie.

THE STRIKE AXIS. The trade is negotiated at USDZAR 15.50 and priced on a deal holding 1/15.50.
The ticket carries 15.50 and never 0.0645; the Legs sheet, the deal about to be booked, carries
the reciprocal.

FROZEN VALUES. `strings_to_*` off is the mechanism; an `<f>` element in the cell XML beside a
reference like `=SUM(A1:A2)` is the falsification.

DETERMINISM. `created` is stamped with the book's `Base_Date`, not the clock. Bytes cannot say
this - the zip records a mod time per member - so the gate compares the SHEET XML and
`docProps/core.xml` across two archives.

THE SILENT SKIP. `Worksheet.write` truncates past 32767 characters and reports only in a return
code. Here it is a named refusal that leaves no file behind.
"""
import json
import os
import sys
import zipfile
from collections import namedtuple
from xml.etree import ElementTree

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import derivus
from derivus import utils
from derivus.quote_sheet import CELL_LIMIT, QuoteSheetError, write_sheet

BASE = '2024-06-28'
EXPIRY = '2025-06-30'
SPOT = 18.5
FLOOR = 15.50
CAP = 18.75
NOTIONAL = 1_000_000.0
PROTECTION_PREMIUM = 41_250.0
FINANCING_PREMIUM = -39_875.0
NET = PROTECTION_PREMIUM + FINANCING_PREMIUM
QUOTE_ID = 'a4f1c0de9b7e2d3155ce0817'
#: `structures.compose` names the container after the structure and the head of the quote id.
DEAL_REFERENCE = 'ZeroCostCollar-' + QUOTE_ID[:8]

#: The book, in the wire form the file on disk holds - `.Timestamp` and `.Curve` tokens and all.
#: `FxRate` spots are the value of one unit in the BASE currency, so ZAR is quoted at 1/18.5 and
#: the market number the desk says out loud, USDZAR 18.5, is the ratio of the two.
BOOK = {'Calc': {
    'Calculation': {'Object': 'BaseValuation', 'Base_Date': {'.Timestamp': BASE},
                    'Currency': 'USD', 'MCMC_Simulations': 1, 'Random_Seed': 1},
    'Deals': {'Tag_Titles': '', 'Reference': 'desk', 'Deals': {'Children': [
        {'Instrument': {'.Deal': {
            'Object': 'FixedCashflowDeal', 'Reference': 'CF1', 'Currency': 'ZAR',
            'Discount_Rate': 'ZAR', 'Calendars': None, 'Amount': NOTIONAL,
            'Payment_Date': {'.Timestamp': '2026-06-28'}}}}]}},
    'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
        'System Parameters': {'Base_Currency': 'USD', 'Base_Date': {'.Timestamp': BASE}},
        'Price Factors': {
            'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
            'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR',
                           'Spot': 1.0 / SPOT},
            'InterestRate.USD': {
                'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.0525], [5.0, 0.048]]}}},
            'InterestRate.ZAR': {
                'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.0825], [5.0, 0.091]]}}},
            'FXVol.USD.ZAR': {
                'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                'Currency': 'USD',
                # authored in the order a `utils.Curve` sorts into, which is the order the encoder
                # writes back - a file that came off a real bootstrap looks like this
                'Surface': {'.Curve': {'meta': [], 'data': [
                    [0.9, 0.25, 0.165], [0.9, 1.0, 0.172], [1.0, 0.25, 0.152],
                    [1.0, 1.0, 0.161], [1.1, 0.25, 0.158], [1.1, 1.0, 0.1665]]}}}}}}}}


def option(reference, buy_sell, option_type, strike_market):
    """One leg's deal, as the runner composes it: the strike on the ENGINE axis, which for an FX
    option whose underlying is the quote currency is the reciprocal of the market number."""
    return {'Object': 'FXOptionDeal', 'Reference': reference, 'Currency': 'USD',
            'Underlying_Currency': 'ZAR', 'Underlying_Amount': NOTIONAL,
            'Strike_Price': 1.0 / strike_market, 'Buy_Sell': buy_sell,
            'Option_Type': option_type, 'Option_Style': 'European',
            'Expiry_Date': {'.Timestamp': EXPIRY}, 'FX_Volatility': 'USD.ZAR',
            'Discount_Rate': 'USD'}


def outcome(financing='COLLAR_FINANCING', **extra):
    """A zero-cost collar's outcome in the shape `structures.quote` returns: floor given, cap
    solved to the protection's premium, every leg strike on the engine axis with the market number
    beside it, and the composed `StructuredDeal`. `financing` names the second leg - where the
    frozen-values and cell-limit gates put their awkward text; `extra` adds keys the runner does
    not send today.
    """
    protection = option('COLLAR_PROTECTION', 'Buy', 'Put', FLOOR)
    sold = option(financing, 'Sell', 'Call', CAP)
    return dict({
        'quote_id': QUOTE_ID,
        'structure': 'ZeroCostCollar',
        'params': {'pair': 'USDZAR', 'expiry': EXPIRY, 'notional': NOTIONAL,
                   'notional_currency': 'USD', 'floor': FLOOR},
        'legs': [
            {'reference': 'COLLAR_PROTECTION', 'role': 'protection', 'deal_type': 'FXOptionDeal',
             'buy_sell': 'Buy', 'strike_market': FLOOR, 'premium': PROTECTION_PREMIUM,
             'solved': None},
            {'reference': financing, 'role': 'financing', 'deal_type': 'FXOptionDeal',
             'buy_sell': 'Sell', 'strike_market': CAP, 'premium': FINANCING_PREMIUM,
             'solved': {'Strike_Price': 1.0 / CAP}}],
        'net': NET,
        # `compose`'s own shape: the container's fields, with the legs as Children beneath them
        'deal': {'Object': 'StructuredDeal', 'Reference': DEAL_REFERENCE, 'Currency': 'USD',
                 'Net_Cashflows': 'Yes',
                 'Children': [{'Instrument': {'.Deal': protection}},
                              {'Instrument': {'.Deal': sold}}]}}, **extra)


MAIN = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
DOCUMENT_RELATION = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'

#: One cell as the archive holds it: where, what kind, the text, and whether a formula element is
#: sitting in it - which is the whole frozen-values question.
Cell = namedtuple('Cell', 'reference kind text formula')


def parts(archive):
    """`{sheet name: archive member}`, resolved the way a reader resolves it - the workbook names
    its sheets and points at them by relationship id, and the rels part says where they live."""
    workbook = ElementTree.fromstring(archive.read('xl/workbook.xml'))
    relations = ElementTree.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
    target = {node.get('Id'): node.get('Target') for node in relations}
    found = {}
    for node in workbook.find(MAIN + 'sheets'):
        where = target[node.get(DOCUMENT_RELATION + 'id')]
        found[node.get('name')] = where[1:] if where.startswith('/') else 'xl/' + where
    return found


def strings(archive):
    """The shared string table a cell of type `s` indexes into."""
    if 'xl/sharedStrings.xml' not in archive.namelist():
        return []
    table = ElementTree.fromstring(archive.read('xl/sharedStrings.xml'))
    return [''.join(node.text or '' for node in item.iter(MAIN + 't')) for item in table]


def cells(path, name):
    """Every written cell of one named sheet, read with the standard library alone."""
    with zipfile.ZipFile(str(path)) as archive:
        table = strings(archive)
        sheet = ElementTree.fromstring(archive.read(parts(archive)[name]))
    found = []
    for cell in sheet.iter(MAIN + 'c'):
        kind, value = cell.get('t'), cell.find(MAIN + 'v')
        if kind == 's':
            text = table[int(value.text)]
        elif kind == 'inlineStr':
            text = ''.join(node.text or '' for node in cell.iter(MAIN + 't'))
        else:
            text = '' if value is None else (value.text or '')
        found.append(Cell(cell.get('r'), kind, text, cell.find(MAIN + 'f') is not None))
    return found


def said(path, name):
    """The text of every string cell on a sheet."""
    return [cell.text for cell in cells(path, name) if cell.kind in ('s', 'inlineStr')]


def numbers(path, name):
    """The value of every number cell on a sheet."""
    return [float(cell.text) for cell in cells(path, name)
            if cell.kind is None and cell.text not in ('', None)]


def holds(values, wanted):
    """Whether a number is among them, to the precision a float round trip through XML keeps."""
    return any(value == pytest.approx(wanted, rel=1e-12, abs=1e-12) for value in values)


def written(tmp_path, name='quote.xlsx', **kwargs):
    """One sheet written for real, from the fixture book, at a real path."""
    path = tmp_path / name
    write_sheet(path, outcome(**kwargs), BOOK)
    return path


def loaded(document):
    """The same book as the OBJECTS a loaded job holds - `.Curve` as a `utils.Curve`, `.Timestamp`
    as a stamp - authored with the library's own types. The writer reads a book by shape rather
    than by importing the engine, so both forms have to render one sheet."""
    if isinstance(document, dict):
        if list(document) == ['.Curve']:
            return utils.Curve(document['.Curve']['meta'], document['.Curve']['data'])
        if list(document) == ['.Timestamp']:
            return pd.Timestamp(document['.Timestamp'])
        return {key: loaded(value) for key, value in document.items()}
    if isinstance(document, list):
        return [loaded(item) for item in document]
    return document


def test_the_fixture_book_is_a_book_the_engine_loads():
    """The fixture read with the decoder a job file is read with, and validated: a drift from the
    wire form - a token misspelt, a section moved, a factor the deals need - fails HERE."""
    context = derivus.Context().load_json((json.dumps(BOOK), 'fixture'))
    factors = context.current_cfg.params['Price Factors']

    assert context.validate() == {'deals': {}, 'factors': []}
    assert 'FXVol.USD.ZAR' in factors and 'InterestRate.ZAR' in factors
    assert factors['FxRate.ZAR']['Spot'] == pytest.approx(1.0 / SPOT)
    # the token really is a curve to the engine, which is the shape the sheet writer renders
    assert isinstance(factors['InterestRate.ZAR']['Curve'], utils.Curve)


def test_the_workbook_holds_the_three_sheets_by_name(tmp_path):
    """Quote, Legs, Market - in that order, because it is the order the conversation goes in."""
    path = written(tmp_path)

    with zipfile.ZipFile(str(path)) as archive:
        assert list(parts(archive)) == ['Quote', 'Legs', 'Market']


def test_the_ticket_carries_the_structure_every_leg_and_the_net(tmp_path):
    """The Quote sheet IS the ticket: nothing a dealer reads off it may be missing."""
    path = written(tmp_path)
    text, value = said(path, 'Quote'), numbers(path, 'Quote')

    assert 'ZeroCostCollar' in text
    assert QUOTE_ID in text
    assert BASE in text
    assert 'USDZAR' in text and 'floor' in text

    for column in ('Role', 'Reference', 'Deal Type', 'Buy/Sell', 'Strike (market)', 'Premium',
                   'Solved'):
        assert column in text
    for leg in ('protection', 'financing', 'COLLAR_PROTECTION', 'COLLAR_FINANCING'):
        assert leg in text
    assert text.count('FXOptionDeal') == 2 and 'Buy' in text and 'Sell' in text
    # a solved leg says WHAT was solved; the given leg's cell stays empty
    assert any(item.startswith('Strike_Price=') for item in text)

    assert holds(value, PROTECTION_PREMIUM) and holds(value, FINANCING_PREMIUM)
    assert 'Net' in text and holds(value, NET)


def test_the_sales_names_ride_the_ticket_when_the_outcome_carries_them(tmp_path):
    """The row is written when the outcome carries a vernacular and left out when it does not - a
    ticket never shows an empty label. The runner sends none today."""
    plain, named = written(tmp_path, 'plain.xlsx'), written(
        tmp_path, 'named.xlsx', vernacular='zero-cost collar, range forward, cylinder')

    assert 'Vernacular' not in said(plain, 'Quote')
    assert 'Vernacular' in said(named, 'Quote')
    assert 'zero-cost collar, range forward, cylinder' in said(named, 'Quote')


def test_the_ticket_quotes_the_strike_the_trade_was_negotiated_in(tmp_path):
    """A collar struck at USDZAR 15.50 is quoted at 15.50. The engine's 1/15.50 belongs to the
    deal, so it is on the Legs sheet and nowhere on the ticket."""
    path = written(tmp_path)
    ticket, legs = numbers(path, 'Quote'), numbers(path, 'Legs')

    assert holds(ticket, FLOOR) and holds(ticket, CAP)
    assert not holds(ticket, 1.0 / FLOOR)
    assert holds(legs, 1.0 / FLOOR) and holds(legs, 1.0 / CAP)


def test_a_reference_that_looks_like_a_formula_stays_text(tmp_path):
    """`strings_to_formulas` off is the mechanism; this is the falsification - a counterparty's
    reference lands as a string cell with no `<f>` element on the sheet at all."""
    typed = '=SUM(A1:A2)'
    path = written(tmp_path, financing=typed)

    for sheet in ('Quote', 'Legs'):
        matched = [cell for cell in cells(path, sheet) if cell.text == typed]
        assert matched, 'the reference never reached the {} sheet'.format(sheet)
        for cell in matched:
            assert cell.kind in ('s', 'inlineStr')
            assert not cell.formula
        assert not any(cell.formula for cell in cells(path, sheet))


def test_two_writes_of_one_quote_hold_the_same_cells(tmp_path):
    """Same inputs, same workbook - down to the created stamp, which is the book's base date and
    not the clock. Read on the sheet XML, the string table and the core properties, because the
    zip records a mod time per member."""
    first, second = written(tmp_path, 'first.xlsx'), written(tmp_path, 'second.xlsx')

    for sheet in ('Quote', 'Legs', 'Market'):
        assert cells(first, sheet) == cells(second, sheet)

    with zipfile.ZipFile(str(first)) as one, zipfile.ZipFile(str(second)) as two:
        for member in list(parts(one).values()) + ['xl/sharedStrings.xml', 'docProps/core.xml']:
            assert one.read(member) == two.read(member), member
        assert BASE in one.read('docProps/core.xml').decode('utf-8')


def test_a_cell_excel_cannot_hold_is_named_not_skipped(tmp_path):
    """`Worksheet.write` truncates past 32767 characters and reports it in a return code. The
    helper raises instead, names the sheet and the cell, and leaves no file."""
    path = tmp_path / 'refused.xlsx'

    with pytest.raises(QuoteSheetError) as refusal:
        write_sheet(path, outcome(financing='X' * (CELL_LIMIT + 1)), BOOK)

    message = str(refusal.value)
    assert 'Quote!B' in message and '32767' in message
    assert str(CELL_LIMIT + 1) in message
    assert not path.exists(), 'a refused sheet left a half workbook behind'


def test_the_legs_sheet_carries_the_deal_that_will_be_booked(tmp_path):
    """Field for field, so the sheet and the booking are checkable against each other. The
    composed deal is the authority - not the ticket row, which says less on purpose."""
    path = written(tmp_path)
    text, value = said(path, 'Legs'), numbers(path, 'Legs')

    for field in ('Object', 'Reference', 'Underlying_Currency', 'Underlying_Amount',
                  'Strike_Price', 'Option_Type', 'Option_Style', 'Expiry_Date', 'FX_Volatility',
                  'Discount_Rate'):
        assert field in text
    assert 'Put' in text and 'Call' in text
    # the wire's timestamp token is rendered as the date it means, not as its json
    assert EXPIRY in text
    # the container the legs hang off is said too - it is what /book/quote will write
    assert DEAL_REFERENCE in text and 'StructuredDeal' in text
    assert holds(value, NOTIONAL)


def test_the_tree_is_said_as_sections_not_as_json_in_one_cell(tmp_path):
    """A container's `Children` ARE the leg sections above it, so the subtree as json in one cell
    would say it twice - and on a three-legged structure walk a good quote into the 32767-character
    refusal."""
    text = said(written(tmp_path), 'Legs')

    assert 'Children' not in text and 'Instrument' not in text
    assert not any('"Instrument"' in item or '".Deal"' in item for item in text)
    # the container's OWN fields still land, which is the half that has to survive the skip
    assert 'Net_Cashflows' in text and 'Yes' in text
    assert max(len(item) for item in text) < 200


def test_the_market_sheet_carries_the_book_it_was_priced_off(tmp_path):
    """The spots, the rate they imply in market terms, the vol surface's own rows and every rate
    curve's tenors."""
    path = written(tmp_path)
    text, value = said(path, 'Market'), numbers(path, 'Market')

    assert 'FxRate.USD' in text and 'FxRate.ZAR' in text
    assert holds(value, 1.0) and holds(value, 1.0 / SPOT)
    # the pair the right way up: the ratio of two base-currency spots is what the desk quotes
    assert 'USDZAR' in text and 'market terms' in text
    assert holds(value, 1.0 / (1.0 / SPOT))

    assert 'FXVol.USD.ZAR' in text and 'Explicit' in text
    for column in ('Moneyness', 'Expiry', 'Volatility'):
        assert column in text
    assert holds(value, 0.152) and holds(value, 0.1665)

    assert 'InterestRate.USD' in text and 'InterestRate.ZAR' in text
    assert 'Tenor' in text and 'Rate' in text
    assert holds(value, 0.0525) and holds(value, 0.091)


def test_a_book_loaded_in_process_renders_the_same_sheet(tmp_path):
    """The wire form and the loaded form are the same book, so they are the same sheet. The writer
    reads a curve by its shape rather than by importing the engine to recognise one."""
    wire, live = written(tmp_path, 'wire.xlsx'), tmp_path / 'live.xlsx'
    write_sheet(live, outcome(), loaded(BOOK))

    for sheet in ('Quote', 'Legs', 'Market'):
        assert cells(wire, sheet) == cells(live, sheet)
