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

"""The quote as the sheet a desk sends - the pending trade, frozen.

`write_sheet(path, outcome, document)` is a pure function of the structure runner's outcome and
the book it was priced against. It is the SOLE importer of `xlsxwriter` in the codebase, which is
what lets the engine install without it: `derivus.service` imports this module inside the job, so
a desk with no `xlsxwriter` still gets its quote and is told which install it is missing, rather
than having a quote refused over a spreadsheet.

Three sheets, and each answers a different question a counterparty asks. `Quote` is the ticket -
what the structure is, what was asked for, what each leg costs and what the package nets, with
every strike in the MARKET terms the trade was negotiated in rather than the reciprocal the engine
prices on. `Legs` is the deal about to be booked, field for field, so the sheet and the booking
are checkable against each other. `Market` is what it was priced off: the pair's spots and the
rate they imply, the vol surface's own rows, and each rate curve's tenors - because a quote
without its market data is a number with no date on it.

Two rules make it a RECORD rather than a spreadsheet program, and both are gated.

Values only. `strings_to_formulas`, `strings_to_urls` and `strings_to_numbers` are all off, so a
reference a client typed as `=SUM(A1:A2)` is that text and not a calculation, and an account code
with leading zeros keeps them. Nothing in the workbook recomputes: what the desk quoted is what
the sheet still says a year later.

Every write is checked. `Worksheet.write` SKIPS a cell it cannot write - a string past Excel's
32767 characters is truncated - and reports it only in a return code nobody reads. So one helper
owns every cell and turns a non-zero code into a named refusal. A quote sheet that quietly lost a
field would be worse than no sheet at all.

Determinism follows from the same instinct: `created` is stamped with the book's own `Base_Date`,
never the wall clock, so two writes of one quote differ in nothing but zip metadata and the sheet
can be diffed against the quote it came from.
"""
import datetime
import json

import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

#: What makes the workbook a record: text stays text, and the whole thing is built in memory so a
#: refused sheet leaves no half file behind in the desk's tmp directory.
WORKBOOK_OPTIONS = {'strings_to_formulas': False, 'strings_to_urls': False,
                    'strings_to_numbers': False, 'in_memory': True}

#: `Worksheet.write`'s return codes said in words. It writes what it can and returns the rest.
WRITE_REFUSALS = {-1: 'the row or column is outside the worksheet',
                  -2: 'the string is longer than the 32767 characters an Excel cell holds',
                  -3: 'the url could not be written'}

#: What a price factor's curve columns MEAN, by the field that carries them. A shape not named
#: here is still rendered - its columns are numbered rather than titled.
CURVE_COLUMNS = {'Curve': ('Tenor', 'Rate'),
                 'Surface': ('Moneyness', 'Expiry', 'Volatility'),
                 'Delta_Surface': ('Delta', 'Expiry', 'Volatility'),
                 'ATM_Vol': ('Tenor', 'Volatility'),
                 'ATM_Ref': ('Tenor', 'Level')}

#: The ticket's columns, in the order a dealer reads them.
#: `Note` is last because it is usually blank: it carries what the runner had to decide about a
#: leg that the parameters did not say - today, that the book has no calibration for the model the
#: leg asked for and it was priced GBM. A dealer who has to know that has to read it on the ticket.
LEG_COLUMNS = ('Role', 'Reference', 'Deal Type', 'Buy/Sell', 'Strike (market)', 'Premium',
               'Solved', 'Note')

#: The keys of a deal block that are the TREE rather than a field of it. Each subtree is already
#: said as its own section, so repeating it as json in one cell is noise - and on a three-legged
#: structure it is noise long enough to pass Excel's cell limit and refuse a quote that was fine.
TREE_KEYS = ('Children', 'Instrument', '.Deal')

#: Excel's own cell limit, which is also the length a rendering this module INVENTED is elided to.
CELL_LIMIT = 32767


class QuoteSheetError(Exception):
    """The sheet could not be written as asked - a cell the workbook would have skipped, or a book
    with no base date to stamp it as of. Named rather than swallowed: the sheet is a record."""


def write_sheet(path, outcome, document):
    """Write `outcome` at `path` as a three-sheet workbook - `Quote`, `Legs`, `Market`.

    `outcome` is what the structure runner returned: `quote_id`, `structure`, `params`, the
    per-leg rows (`role`, `reference`, `deal_type`, `buy_sell`, `strike_market`, `premium`,
    `solved`), the `net`, and `deal` - the composed deal in wire form, which is the authority on
    what each leg actually holds. `document` is the book it was priced against, also in wire form,
    so `.Curve` and `.Timestamp` arrive as the tokens a job file carries; a document loaded in
    process is read by the same code, by shape, because this module never imports the engine to
    render a book.

    Nothing reaches the disk until every cell has been accepted: the workbook is assembled in
    memory and a refusal abandons it, so `path` either holds a whole sheet or holds nothing.
    """
    base_date = _base_date(document)
    workbook = xlsxwriter.Workbook(str(path), dict(WORKBOOK_OPTIONS))
    try:
        # the created stamp is the BOOK's date, not now: a quote sheet is a statement about a
        # valuation date, and a clock in the metadata would move the file on every rewrite
        workbook.set_properties({'title': '{} quote'.format(outcome.get('structure')),
                                 'subject': str(outcome.get('quote_id')),
                                 'created': base_date})
        formats = {'title': workbook.add_format({'bold': True, 'font_size': 12}),
                   'head': workbook.add_format({'bold': True}),
                   'rate': workbook.add_format({'num_format': '0.0000'}),
                   'strike': workbook.add_format({'num_format': '0.0000'}),
                   'premium': workbook.add_format({'num_format': '0.00'})}
        _quote_sheet(_Sheet(workbook, 'Quote', formats), outcome, base_date)
        _legs_sheet(_Sheet(workbook, 'Legs', formats), outcome)
        _market_sheet(_Sheet(workbook, 'Market', formats), outcome, document, base_date)
    except BaseException:
        # in_memory means nothing has reached the disk yet; the flag stops any destructor
        # finishing a workbook this refusal abandoned
        workbook.fileclosed = 1
        raise
    workbook.close()


class _Sheet:
    """One worksheet, the row it is up to, and the only place a value reaches xlsxwriter."""

    def __init__(self, workbook, name, formats):
        self.name = name
        self.sheet = workbook.add_worksheet(name)
        self.formats = formats
        self.row = 0

    def cell(self, column, value, style=None):
        """Write one cell, or name the refusal. `Worksheet.write` returns a code and writes what it
        can - an unchecked call is how a sheet loses a field without anyone being told."""
        code = self.sheet.write(self.row, column, value, self.formats.get(style))
        if code:
            size = ' (the value is {} characters)'.format(
                len(value)) if isinstance(value, str) else ''
            raise QuoteSheetError('quote sheet: xlsxwriter skipped {}!{} - {} (code {}){}'.format(
                self.name, xl_rowcol_to_cell(self.row, column),
                WRITE_REFUSALS.get(code, 'unknown write code'), code, size))

    def line(self, cells, style=None):
        """One row of cells, each a value or a `(value, style)` pair, then move down."""
        for column, entry in enumerate(cells):
            value, cell_style = entry if isinstance(entry, tuple) else (entry, style)
            self.cell(column, value, cell_style)
        self.row += 1

    def blank(self):
        """Leave a row empty - the sheet's only punctuation."""
        self.row += 1

    def widths(self, widths):
        """Column widths, left to right, so the sheet opens readable rather than in ####."""
        for column, width in enumerate(widths):
            self.sheet.set_column(column, column, width)


def _quote_sheet(sheet, outcome, base_date):
    """The ticket: what was asked for, what each leg costs, what the package nets."""
    sheet.widths((26, 24, 18, 12, 16, 16, 34, 48))
    sheet.line([('Quote', 'title')])
    sheet.blank()
    sheet.line([('Structure', 'head'), _stated(outcome.get('structure'))])
    if outcome.get('vernacular'):
        sheet.line([('Vernacular', 'head'), _stated(outcome['vernacular'])])
    sheet.line([('Quote ID', 'head'), _stated(outcome.get('quote_id'))])
    sheet.line([('Base Date', 'head'), base_date.strftime('%Y-%m-%d')])
    sheet.blank()
    sheet.line([('Parameters', 'head')])
    for name, value in (outcome.get('params') or {}).items():
        sheet.line([name, _stated(value)])
    sheet.blank()
    sheet.line([(title, 'head') for title in LEG_COLUMNS])
    for leg in outcome.get('legs') or []:
        # the strike here is the MARKET number the trade was negotiated in; the engine's
        # reciprocal lives on the Legs sheet, where it belongs to the deal rather than to the
        # conversation
        sheet.line([_stated(leg.get('role')), _stated(leg.get('reference')),
                    _stated(leg.get('deal_type')), _stated(leg.get('buy_sell')),
                    (_number(leg.get('strike_market')), 'strike'),
                    (_number(leg.get('premium')), 'premium'),
                    _solved(leg.get('solved')), _stated(leg.get('note'))])
    sheet.blank()
    sheet.line([('Net', 'head'), None, None, None, None, (_number(outcome.get('net')), 'premium')])


def _legs_sheet(sheet, outcome):
    """Every field of the deal about to be booked, so the sheet and the booking are checkable
    against each other rather than merely consistent-looking."""
    sheet.widths((32, 48))
    sheet.line([('Legs', 'title')])
    sheet.blank()
    blocks, said = {}, set()
    for fields in _deals(outcome.get('deal')):
        reference = fields.get('Reference')
        if isinstance(reference, str):
            blocks.setdefault(reference, fields)
    for leg in outcome.get('legs') or []:
        reference = leg.get('reference')
        sheet.line([('{} - {}'.format(_stated(leg.get('role')), _stated(reference)), 'head')])
        # the composed deal is the authority on a leg's fields; a leg it does not carry is still
        # SAID, out of the outcome's own row, rather than dropped from the sheet
        fields = blocks.get(reference)
        said.add(reference)
        _fields_block(sheet, leg if fields is None else fields)
    for reference, fields in blocks.items():
        if reference not in said:
            # the container the legs hang off, and anything else the composer put in the tree
            sheet.line([('{} - {}'.format(_stated(fields.get('Object')), reference), 'head')])
            _fields_block(sheet, fields)


def _fields_block(sheet, fields):
    """One deal's own fields as key/value rows, then a blank. What hangs BELOW it in the tree is
    skipped: those deals get their own sections, and a cell holding the whole subtree as json says
    the same thing twice at a length Excel may refuse."""
    for name, value in fields.items():
        if name not in TREE_KEYS:
            sheet.line([name, _stated(value)])
    sheet.blank()


def _market_sheet(sheet, outcome, document, base_date):
    """What the quote was priced off, read out of the book itself."""
    market = _market_data(document)
    factors = market.get('Price Factors') or {}
    system = market.get('System Parameters') or {}
    deals = _deals(outcome.get('deal'))

    sheet.widths((30, 22, 20, 18))
    sheet.line([('Market', 'title')])
    sheet.blank()
    sheet.line([('Base Currency', 'head'), _stated(system.get('Base_Currency'))])
    sheet.line([('Base Date', 'head'), base_date.strftime('%Y-%m-%d')])
    sheet.blank()

    pair = _currencies(outcome, deals)
    every = [name for name in sorted(factors) if name.startswith('FxRate.')]
    chosen = [name for name in ['FxRate.' + code for code in pair or ()] if name in factors]
    sheet.line([('FX Rates', 'head')])
    sheet.line([(title, 'head')
                for title in ('Factor', 'Currency', 'Spot (base currency per unit)')])
    for name in chosen or every:
        sheet.line([name, name.split('.', 1)[-1],
                    (_number((factors.get(name) or {}).get('Spot')), 'strike')])
    if pair:
        sheet.line([('{}{}'.format(*pair), 'head'), 'market terms',
                    (_number(_market_rate(factors, *pair)), 'strike')])
    sheet.blank()

    referenced = sorted({'FXVol.' + deal['FX_Volatility'] for deal in deals
                         if isinstance(deal.get('FX_Volatility'), str)}.intersection(factors))
    sheet.line([('FX Volatility', 'head')])
    for name in referenced or [name for name in sorted(factors) if name.startswith('FXVol.')]:
        _factor_block(sheet, name, factors.get(name))
    sheet.blank()

    sheet.line([('Interest Rates', 'head')])
    for name in sorted(factors):
        if name.startswith('InterestRate.'):
            _factor_block(sheet, name, factors.get(name))


def _factor_block(sheet, name, factor):
    """One price factor: its stated fields as key/value rows, then every curve it carries as its
    own tenor/value rows. The curve is the point - a surface quoted as a name tells a counterparty
    nothing about what the number was read off."""
    sheet.line([(name, 'head')])
    curves = []
    for field, value in (factor or {}).items():
        rendered = _curve_data(value)
        if rendered is None:
            sheet.line([field, _stated(value)])
        else:
            curves.append((field, rendered))
    for field, (meta, rows) in curves:
        width = max([len(row) for row in rows] or [0])
        columns = CURVE_COLUMNS.get(field, ())
        sheet.line([(field, 'head')])
        if meta:
            sheet.line(['meta', _pretty(list(meta))])
        sheet.line([(columns[index] if index < len(columns) else 'Column {}'.format(index + 1),
                     'head') for index in range(width)])
        for row in rows:
            sheet.line([(_number(value), 'rate') for value in row])
        sheet.blank()


def _base_date(document):
    """The book's base date, as a `datetime` to stamp the workbook created with.

    xlsxwriter stamps `dcterms:created` off the wall clock unless it is handed a datetime, and a
    sheet whose metadata moves on every write cannot be diffed against the quote it came from. A
    book naming no base date is refused rather than stamped now: the numbers in the sheet were
    priced AS OF something, and a sheet that will not say what is not a record.
    """
    calc = document.get('Calc') or {}
    stamp = (calc.get('Calculation') or {}).get('Base_Date')
    if stamp is None:
        stamp = ((calc.get('MergeMarketData') or {}).get('ExplicitMarketData') or {}).get(
            'System Parameters', {}).get('Base_Date')
    text = stamp.get('.Timestamp') if isinstance(stamp, dict) else stamp
    if hasattr(text, 'year'):
        # a book loaded in process carries a real Timestamp where the wire carries the token
        return datetime.datetime(text.year, text.month, text.day)
    if isinstance(text, str) and text:
        try:
            return datetime.datetime.fromisoformat(text)
        except ValueError:
            raise QuoteSheetError(
                'quote sheet: the book Base_Date {!r} is not an ISO date'.format(text))
    raise QuoteSheetError('quote sheet: the book names no Base_Date to stamp the workbook with')


def _market_data(document):
    """The book's explicit market data block, or nothing when the document carries none."""
    return ((document.get('Calc') or {}).get('MergeMarketData') or {}).get(
        'ExplicitMarketData') or {}


def _deals(node, found=None):
    """Every deal field block in a composed deal, in the order the tree holds them.

    A wire deal is `{'Instrument': {'.Deal': fields}}` with `Children` hanging off the container,
    so the walk follows those keys and collects anything naming an `Object`. Written as a walk
    rather than an unpack because the composer owns the tree's shape, not this module.
    """
    found = [] if found is None else found
    if isinstance(node, list):
        for item in node:
            _deals(item, found)
    elif isinstance(node, dict):
        if isinstance(node.get('Object'), str):
            found.append(node)
        for key in ('Instrument', '.Deal', 'Deals', 'Children'):
            if key in node:
                _deals(node[key], found)
    return found


def _currencies(outcome, deals):
    """The two currencies the quote is on, first over second, or None when nothing says.

    The parameters name the pair the trade was asked for; a quote whose parameters do not is read
    off the legs themselves, where an FX option's own two currencies are the same pair.
    """
    pair = (outcome.get('params') or {}).get('pair')
    if isinstance(pair, str):
        letters = ''.join(character for character in pair.upper() if character.isalpha())
        if len(letters) == 6:
            return letters[:3], letters[3:]
    for deal in deals:
        first, second = deal.get('Currency'), deal.get('Underlying_Currency')
        if isinstance(first, str) and isinstance(second, str) and first != second:
            return first, second
    return None


def _market_rate(factors, first, second):
    """`second` per unit of `first` - the number the market quotes the pair as.

    An `FxRate` spot is the value of one unit in the BASE currency, so the ratio of the two is the
    pair the right way up. It is the same inversion the runner does to a strike, which is why the
    sheet carries 15.50 beside a deal holding 1/15.50 and both are true.
    """
    one = (factors.get('FxRate.' + first) or {}).get('Spot')
    two = (factors.get('FxRate.' + second) or {}).get('Spot')
    if isinstance(one, (int, float)) and isinstance(two, (int, float)) and two:
        return one / two
    return None


def _curve_data(value):
    """`(meta, rows)` of a curve, or None when the value is not one.

    A book off the wire carries `{'.Curve': {'meta': [...], 'data': [...]}}`; a book loaded in
    process carries a `utils.Curve`. Both are read here by SHAPE, so rendering a sheet never drags
    the engine in behind an optional spreadsheet dependency.
    """
    if isinstance(value, dict) and '.Curve' in value:
        token = value['.Curve'] or {}
        meta, data = token.get('meta') or [], token.get('data') or []
    elif hasattr(value, 'array') and hasattr(value, 'meta'):
        array = value.array
        meta = value.meta or []
        data = array.tolist() if hasattr(array, 'tolist') else list(array)
    else:
        return None
    return list(meta), [list(row) if isinstance(row, (list, tuple)) else [row] for row in data]


def _stated(value):
    """A value a cell can hold.

    Scalars pass straight through, a date says its own date, and a one-key wire token says its
    payload - keeping the token's NAME where the name carries the unit, so a reader is never shown
    a bare 2.5 that was a percent. Anything else - a nested block, a grid of grids - is
    pretty-printed. A shape this module has no rendering for is worth showing badly; it is not
    worth failing a quote over.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d')
    if isinstance(value, dict) and len(value) == 1:
        token, payload = next(iter(value.items()))
        if isinstance(token, str) and token.startswith('.') and isinstance(
                payload, (str, int, float)):
            return payload if token in ('.Timestamp', '.DateOffset') else '{}({})'.format(
                token[1:], payload)
    return _pretty(value)


def _pretty(value):
    """A shape said as text, elided to what a cell holds.

    A rendering this module INVENTED may be shortened; a value the caller stated is never silently
    shortened - that one earns the named refusal from the cell writer instead.
    """
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= CELL_LIMIT else text[:CELL_LIMIT - 3] + '...'


def _number(value):
    """A number for the column's format to apply to, or the value said some other way - a premium
    the runner could not state shows as what it IS rather than as a convincing zero."""
    if isinstance(value, bool):
        return _stated(value)
    return value if isinstance(value, (int, float)) else _stated(value)


def _solved(solved):
    """What a recipe step moved, as one cell: `Strike_Price=0.0645`. Blank is a leg that was given
    rather than solved - and that difference is the whole reason a structure has a recipe."""
    if not isinstance(solved, dict):
        return _stated(solved)
    return ', '.join('{}={}'.format(field, _stated(value))
                     for field, value in solved.items()) or None
