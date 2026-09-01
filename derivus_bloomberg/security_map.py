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
"""Consume a verified security map - the artifact `DV_Bloomberg discover` writes.

The map is the caller's and lives outside any repo. Every entry carries its EVIDENCE: the NAME
the terminal answered, the quote's last print, and when it was verified; `load` refuses an entry
missing any of it by name, so a hand-edited ticker cannot ride into `fetch_fx_vol` on the
strength of being in the file.

This module talks to no service and prices nothing: it turns map blocks into the definitions
`fxvol` consumes, and answers the freshness questions the fetch path does not ask (`fetch_fx_vol`
stamps every point with the retrieval clock, so a dead series arrives wearing today's time -
check `stale` beside a tick).
"""
import datetime
import json
import math
import os

from .errors import BloombergConfigurationError, InvalidQuote
from .types import FXQuoteSecurity, FXVolDefinition

SCHEMA = 'derivus-bloomberg-map/1'


def home():
    """The user-data directory the DV_* tools share: `$DV_HOME`, defaulting to `~/.derivus` -
    where a desk's own files (`book.json`, `security_map.json`, `seed.json`) live, outside any
    repo."""
    return os.path.expanduser(os.environ.get('DV_HOME', os.path.join('~', '.derivus')))


def packaged_seed():
    """The path to the questionnaire the package ships - the starting vocabulary of pairs, curve
    prefixes and grids, copied to `$DV_HOME/seed.json` on first use. The seed names CANDIDATES
    only; the trust boundary is `load`, which refuses an entry carrying no terminal evidence."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seed.json')


def load(path=None):
    """The map - `$DV_HOME/security_map.json` unless named - refused unless every entry still
    carries its evidence."""
    path = path or os.path.join(home(), 'security_map.json')
    with open(path, encoding='utf-8') as handle:
        document = json.load(handle)
    if document.get('schema') != SCHEMA:
        raise BloombergConfigurationError(
            '{} is not a security map this package reads (schema {!r}, wanted {!r})'.format(
                path, document.get('schema'), SCHEMA))
    for entry_path, entry in entries(document):
        for evidence in ('name', 'verified'):
            if not entry.get(evidence):
                raise BloombergConfigurationError(
                    '{} ({}) carries no {} - an entry without its terminal evidence is not '
                    'trusted; re-run `DV_Bloomberg discover`'.format(
                        '/'.join(entry_path), entry['security'], evidence))
    return document


def entries(document):
    """Every leaf entry of the map's blocks as `(path, entry)` - a leaf is a dict naming a
    `security`; anything else is a container (or metadata like `expiries`) and is walked past."""
    def walk(node, path):
        if isinstance(node, dict):
            if 'security' in node:
                yield path, node
            else:
                for key, value in node.items():
                    yield from walk(value, path + (key,))
    yield from walk(document.get('blocks', {}), ())


def fx_vol_definition(document, pair, expiries=None, pillars=(0.25,), quote_sensitivity=False):
    """An `FXVolDefinition` built from the map's verified entries - the caller still owns scope
    (which expiries, which pillars) and the refusals are by name: a pillar the map never carried
    (the 35-delta grid stops at 5Y) is a named refusal, not a KeyError out of a dict.

    The surface name and quote currency derive from the pair - `USDZAR` is the base-then-quote
    Bloomberg spelling, so the surface is `USD.ZAR` and the quote (domestic) currency `ZAR`.
    Vols are quoted in percent, `FXVolDefinition`'s own `quote_scale` default.
    """
    block = document.get('blocks', {}).get('fx_vol', {}).get(pair)
    if block is None:
        raise BloombergConfigurationError('the map carries no fx_vol block for {}'.format(pair))
    labels = list(expiries) if expiries else list(block['expiries'])
    missing = [label for label in labels if label not in block['expiries']]
    if missing:
        raise BloombergConfigurationError(
            '{} was not verified at {}'.format(pair, ', '.join(missing)))

    securities = {}
    for label in labels:
        quotes = block['quotes'][label]
        if 'ATM' not in quotes:
            # a tenor stays in the map when ANY of its quotes verified, so a dead ATM beside a
            # live wing is reachable - and a smile with no ATM is not a smile
            raise BloombergConfigurationError(
                '{} carries no verified ATM at {}'.format(pair, label))
        securities[(label, 'ATM', None)] = FXQuoteSecurity(quotes['ATM']['security'], 'PX_LAST')
        for pillar in pillars:
            key = '{:.2f}'.format(pillar)
            for quote_type, prefix in (('RR', 'RR_'), ('BF', 'BF_')):
                if prefix + key not in quotes:
                    raise BloombergConfigurationError(
                        '{} carries no {:.0f}-delta {} at {}'.format(
                            pair, pillar * 100, quote_type, label))
                securities[(label, quote_type, pillar)] = FXQuoteSecurity(
                    quotes[prefix + key]['security'], 'PX_LAST')

    return FXVolDefinition(
        pair=pair, surface_name=pair[:3] + '.' + pair[3:], currency=pair[3:],
        expiries={label: block['expiries'][label] for label in labels},
        pillars=tuple(pillars), securities=securities, quote_sensitivity=quote_sensitivity)


def fx_spot_route(document, currency, base_currency):
    """`(pair, security)` - the verified market pair whose cross prices one unit of `currency` in
    `base_currency` units, or `(None, None)` where the currency IS the base and prices itself.

    Both spellings a market pair could carry are tried - `USDZAR` for ZAR against a USD base,
    `EURUSD` for EUR against it. Triangulating through a third currency is a market VIEW rather
    than a lookup, so an unverified pair refuses by name and the caller keeps the spot it had.
    """
    if currency == base_currency:
        return None, None
    block = document.get('blocks', {}).get('fx_spot', {})
    for pair in (currency + base_currency, base_currency + currency):
        if pair in block:
            return pair, block[pair]['security']
    raise BloombergConfigurationError(
        'the map verified no spot for {} against {} - neither {} nor {}'.format(
            currency, base_currency, currency + base_currency, base_currency + currency))


def fetch_fx_spot(source, securities):
    """`{security: last price}` for verified spot tickers - ONE request off a live session.

    The tolerant reader makes the request and the strict policy is applied client-side: a name
    that did not answer, or answered with something that is not a positive finite number, raises
    `InvalidQuote` naming it rather than reaching a book.
    """
    wanted = sorted(set(securities))
    report = source.reference_data_report(wanted, ('PX_LAST',))
    values = {}
    for security in wanted:
        row = report.get(security, {'error': 'no answer in the response', 'fields': {}})
        raw = row.get('fields', {}).get('PX_LAST')
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = None
        if value is None or not math.isfinite(value) or value <= 0.0:
            raise InvalidQuote('{} returned {!r}{}'.format(
                security, raw, ' - {}'.format(row['error']) if row.get('error') else ''))
        values[security] = value
    return values


def freshness(source, securities):
    """`{security: LAST_UPDATE_DT}` off a live session - the question `fetch_fx_vol` does not
    ask, since it requests only each quote's value field and stamps the retrieval clock."""
    report = source.reference_data_report(list(securities), ('LAST_UPDATE_DT',))
    return {name: row['fields'].get('LAST_UPDATE_DT') for name, row in report.items()}


def stale(source, securities, as_of=None, stale_days=5):
    """The securities whose last print is older than `stale_days` - BY NAME, with the date, so a
    refusal reads as what it is. A missing LAST_UPDATE_DT is flagged too: a quote that cannot
    evidence freshness should not tick a book unremarked."""
    as_of = as_of or datetime.date.today()
    late = {}
    for name, stamp in freshness(source, securities).items():
        if stamp in (None, ''):
            late[name] = 'no LAST_UPDATE_DT'
        elif (as_of - datetime.date.fromisoformat(str(stamp)[:10])).days > stale_days:
            late[name] = str(stamp)
    return late
