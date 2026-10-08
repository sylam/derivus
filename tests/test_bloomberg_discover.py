"""Discovery: 'verify every security' as a property rather than an instruction.

Nothing here opens a socket: every terminal is a `BloombergSession` whose event walk is canned rows,
the subclass seam the session declares for its offline gates. Gated: a candidate is believed only when the
terminal's own NAME says it is what it claims; a dead benchmark is refused on its update date
however sane its price reads (the SAONIA trap - 8.855, nineteen years after its last print); an
entry stripped of its evidence refuses to load by name; and the strict and tolerant readers are
one walk with two policies. First use is its own claim: the shipped questionnaire lands where the
desk can cut it down BEFORE the terminal is asked, so a refused probe leaves a seed and no
half-map, and a map already on disk is loaded rather than rebuilt.
"""
import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_bloomberg import discover, security_map
from derivus_bloomberg.errors import (BloombergConfigurationError, BloombergEntitlementError,
                                      BloombergRequestError)
from derivus_bloomberg.session import BloombergSession
from derivus_bloomberg.types import FXQuoteSecurity

AS_OF = datetime.date(2026, 8, 27)

SEED = {
    'fx_vol': {'pairs': ['USDZAR'], 'expiries': {'1M': 1.0 / 12.0, '1Y': 1.0},
               'pillars': [0.25, 0.1]},
    'fx_spot': {'pairs': ['USDZAR']},
    'rates': {'ZAR': {'prefix': 'SASW', 'expect': 'ZAR SWAP QTR', 'years': [1, 5],
                      'overnight': {'security': 'ZARONIA Index',
                                    'expect': 'South African Overnight'}},
              'USD': {'prefix': 'USOSFR', 'expect': 'USD OIS', 'weeks': ['1W'],
                      'months': ['1M', '11M'], 'years': [10]}},
    'swaption': {'ZAR': {'prefix': 'SASN', 'expect': 'ZAR SWPT NVOL',
                         'expiries': {'1Y': '01'}, 'tenor_years': [1, 10]}},
}


def answered(name, px_last=1.0, last_update='2026-08-26'):
    return {'ok': True, 'error': None,
            'fields': {'NAME': name, 'PX_LAST': px_last, 'LAST_UPDATE_DT': last_update}}


def full_report():
    """Every SEED candidate answered as itself, live - the baseline the mutations below break. The
    names are spelled as the terminal spells them: an EURAUD-style double space on the 1M ATM, the
    10Y spelling's dropped space on the 1Y risk reversals."""
    report = {'USDZAR BGN Curncy': answered('USD-ZAR X-RATE'),
              'ZARONIA Index': answered('South African Overnight Index'),
              'SASW1 BGN Curncy': answered('ZAR SWAP QTR (VS 3M) 1Y'),
              'SASW5 BGN Curncy': answered('ZAR SWAP QTR (VS 3M) 5Y'),
              'USOSFR1Z BGN Curncy': answered('USD OIS 1WK'),
              'USOSFRA BGN Curncy': answered('USD OIS 1M'),
              'USOSFRK BGN Curncy': answered('USD OIS 11M'),
              'USOSFR10 BGN Curncy': answered('USD OIS 10Y'),
              'SASN011 Curncy': answered('ZAR SWPT NVOL 1Y1Y'),
              'SASN0110 Curncy': answered('ZAR SWPT NVOL 1Y10Y')}
    for tenor in ('1M', '1Y'):
        report['USDZARV{} BGN Curncy'.format(tenor)] = answered(
            'USD-ZAR OPT VOL {}{}'.format(' ' if tenor == '1M' else '', tenor))
        for code in ('25', '10'):
            report['USDZAR{}R{} BGN Curncy'.format(code, tenor)] = answered(
                'USD-ZAR RR {}D{}{}'.format(code, '' if tenor == '1Y' else ' ', tenor))
            report['USDZAR{}B{} BGN Curncy'.format(code, tenor)] = answered(
                'USD-ZAR BFY {}D {}'.format(code, tenor))
    return report


def rows(report):
    """A canned report as the event walk's `(security, error, fields)` rows."""
    return [(security, None if row['ok'] else row['error'], row['fields'])
            for security, row in report.items()]


def discover_with(report):
    return discover.discover(SEED, Walked(rows(report)), AS_OF)


def test_the_terminals_answers_verify_the_grammar_and_ledger_the_rest():
    """The seed's vocabulary spelled as the terminal verified it, ticker and NAME both - the OIS
    suffix 1Z for weeks and bare letters for months, an unpadded swaption tenor (SASN011 is 1Y into
    1Y), the 10-delta wings beside the 25 - and the order of distrust, one candidate each: refused
    by Bloomberg, answering as something else, resolving but priceless, priced but long dead (the
    SAONIA shape, a plausible level nineteen years old) and live. Only live candidates enter the
    map, each carrying its evidence; the rest land on the `rejected` ledger by name.

    Killing mutations: the week suffix spelled `1W`; the update date unread - the dead print enters
    the map.
    """
    report = full_report()
    report['USDZAR25B1Y BGN Curncy'] = {'ok': False, 'error': 'Unknown/Invalid Security',
                                        'fields': {}}
    report['SASW5 BGN Curncy'] = answered('SOMETHING ELSE ENTIRELY')
    report['SASN0110 Curncy'] = answered('ZAR SWPT NVOL 1Y10Y', px_last=None)
    report['ZARONIA Index'] = answered('South African Overnight Index', px_last=8.855,
                                       last_update='2007-03-26')
    document, _ = discover_with(report)
    blocks = document['blocks']
    quotes = blocks['fx_vol']['USDZAR']['quotes']
    assert quotes['1M']['ATM'] == {'security': 'USDZARV1M BGN Curncy', 'name': 'USD-ZAR OPT VOL  1M',
                                   'last_update': '2026-08-26', 'verified': AS_OF.isoformat()}
    assert quotes['1Y']['RR_0.25']['security'] == 'USDZAR25R1Y BGN Curncy'
    assert quotes['1M']['RR_0.10']['security'] == 'USDZAR10R1M BGN Curncy'
    assert 'BF_0.25' not in quotes['1Y']
    # the expiries meta only names tenors that actually landed quotes
    assert set(blocks['fx_vol']['USDZAR']['expiries']) == {'1M', '1Y'}
    assert {label: entry['security'] for label, entry in blocks['rates']['USD']['strip'].items()} == {
        '1W': 'USOSFR1Z BGN Curncy', '1M': 'USOSFRA BGN Curncy', '11M': 'USOSFRK BGN Curncy',
        '10Y': 'USOSFR10 BGN Curncy'}
    assert blocks['swaption']['ZAR'] == {'1Y x 1Y': dict(
        security='SASN011 Curncy', name='ZAR SWPT NVOL 1Y1Y', last_update='2026-08-26',
        verified=AS_OF.isoformat())}
    assert {security: row['verdict'] for security, row in document['rejected'].items()} == {
        'USDZAR25B1Y BGN Curncy': 'invalid', 'SASW5 BGN Curncy': 'mismatch',
        'SASN0110 Curncy': 'unpriced', 'ZARONIA Index': 'dead'}


def test_a_missing_seed_is_an_instruction_not_a_traceback(tmp_path, monkeypatch):
    """The seed is the one file no tool writes, so discovery without one says where the seed goes
    and where the starting one is, before any session is attempted.

    Killing mutation: the seed's presence unchecked - the CLI dies opening it."""
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['DV_Bloomberg', 'discover'])
    with pytest.raises(SystemExit, match='questionnaire'):
        discover.main()


def test_a_desk_seed_that_declares_no_conventions_reads_the_packaged_entry(
        tmp_path, monkeypatch, caplog):
    """THE DESK'S SEED IS AUTHORITATIVE WHERE IT DECLARES SOMETHING, and read-only either way.

    A desk file written before the conventions existed names a currency and says nothing about what
    it accrues on; one written before a curve was keyed has no entry at all. Both read the PACKAGED
    declaration, said at INFO so the desk knows whose numbers it got, and neither writes the desk's
    own file - a seed is the one file no tool here edits. An entry that does declare its
    conventions wins, packaged spelling included.

    Killing mutation: the desk's seed taken whole whether or not its entry declares conventions.
    """
    import logging

    monkeypatch.setenv('DV_HOME', str(tmp_path))
    desk = tmp_path / 'seed.json'
    desk.write_text(json.dumps({'rates': {
        'ZAR': {'prefix': 'SASW', 'expect': 'ZAR SWAP QTR', 'years': [1, 5]},
        'MINE': {'prefix': 'X', 'expect': 'X', 'conventions': {'front': 'overnight'}}}}),
        encoding='utf-8', newline='\n')
    before = desk.read_bytes()

    with caplog.at_level(logging.INFO):
        stale = security_map.curve_seed('ZAR')
        unkeyed = security_map.curve_seed('ZAR-ZARONIA')
    declared = security_map.curve_seed('MINE')
    rates = security_map.seeded_rates()

    assert stale['rates']['ZAR']['conventions']['front'] == 'fixings/3M'
    assert unkeyed['rates']['ZAR-ZARONIA']['conventions']['compounding'] == 'OIS'
    assert declared['rates']['MINE'] == {'prefix': 'X', 'expect': 'X',
                                         'conventions': {'front': 'overnight'}}
    assert sum(str(desk) in record.getMessage() for record in caplog.records) == 2
    assert rates['MINE'] == declared['rates']['MINE'], "the desk's own entry did not win"
    assert rates['ZAR']['conventions'] and 'ZAR-ZARONIA' in rates
    assert desk.read_bytes() == before, 'the desk seed was written'


def test_a_map_entry_without_evidence_is_refused_by_name(tmp_path):
    """The map is trusted BECAUSE each entry records what the terminal answered - so a hand-edited
    entry with the evidence stripped refuses to load, naming the entry.

    Killing mutation: the evidence check dropped from `load`."""
    document, _ = discover_with(full_report())
    path = tmp_path / 'map.json'
    path.write_text(json.dumps(document, indent=1), encoding='utf-8', newline='\n')
    loaded = security_map.load(str(path))
    assert loaded['blocks']['fx_spot']['USDZAR']['security'] == 'USDZAR BGN Curncy'

    document['blocks']['fx_vol']['USDZAR']['quotes']['1M']['ATM'].pop('name')
    path.write_text(json.dumps(document, indent=1), encoding='utf-8', newline='\n')
    with pytest.raises(BloombergConfigurationError, match='USDZARV1M'):
        security_map.load(str(path))


def test_a_definition_builds_from_the_map_and_a_missing_pillar_refuses_by_name():
    """The map is consumable exactly where the package always started - an `FXVolDefinition` -
    and scope the terminal never verified (the 35-delta grid stops at 5Y) refuses naming the
    pair, the pillar and the tenor, never a KeyError out of a dict lookup.

    Killing mutation: a smile with no verified ATM read as one."""
    document, _ = discover_with(full_report())
    definition = security_map.fx_vol_definition(document, 'USDZAR', expiries=['1M', '1Y'],
                                                pillars=(0.25,))
    assert definition.surface_name == 'USD.ZAR' and definition.currency == 'ZAR'
    assert definition.expiries == {'1M': 1.0 / 12.0, '1Y': 1.0}
    assert definition.securities[('1M', 'ATM', None)] == FXQuoteSecurity(
        'USDZARV1M BGN Curncy', 'PX_LAST')
    assert definition.securities[('1Y', 'RR', 0.25)] == FXQuoteSecurity(
        'USDZAR25R1Y BGN Curncy', 'PX_LAST')

    with pytest.raises(BloombergConfigurationError, match='35-delta'):
        security_map.fx_vol_definition(document, 'USDZAR', expiries=['1M'], pillars=(0.35,))
    with pytest.raises(BloombergConfigurationError, match='10Y'):
        security_map.fx_vol_definition(document, 'USDZAR', expiries=['10Y'])
    with pytest.raises(BloombergConfigurationError, match='EURUSD'):
        security_map.fx_vol_definition(document, 'EURUSD')

    # a dead ATM beside a live wing is a reachable map (build_map keeps a tenor when ANY of its
    # quotes verified), and a smile with no ATM must refuse by name, not KeyError
    document['blocks']['fx_vol']['USDZAR']['quotes']['1M'].pop('ATM')
    with pytest.raises(BloombergConfigurationError, match='no verified ATM at 1M'):
        security_map.fx_vol_definition(document, 'USDZAR', expiries=['1M'])


class Walked(BloombergSession):
    """A session whose event walk is canned rows - so the two READERS are gated as two policies
    over one walk, which is the refactor's whole claim."""

    def __init__(self, rows):
        super().__init__()
        self._api = self._session = self._service = object()  # started, as far as the guard cares
        self.rows = rows

    def _walk(self, securities, fields):
        yield from self.rows


def test_the_two_session_readers_share_one_walk_and_differ_only_in_policy():
    """`reference_data` refuses the whole batch on one bad name - a production tick built from a
    partial answer is a wrong market - while `reference_data_report` records the same walk's rows
    as per-security outcomes, filling in the names the response never answered. An entitlement
    text still types the strict refusal.

    Killing mutation: the strict reader recording a bad name instead of refusing the batch."""
    rows = [('GOOD Curncy', None, {'NAME': 'GOOD-NAME'}),
            ('BAD Curncy', 'securityError = Unknown/Invalid', {})]
    with pytest.raises(BloombergRequestError, match='BAD Curncy: securityError'):
        Walked(rows).reference_data(['GOOD Curncy', 'BAD Curncy'], ('NAME',))

    report = Walked(rows).reference_data_report(['GOOD Curncy', 'BAD Curncy', 'SILENT Curncy'],
                                                ('NAME',))
    assert report['GOOD Curncy'] == {'ok': True, 'error': None, 'fields': {'NAME': 'GOOD-NAME'}}
    assert report['BAD Curncy']['ok'] is False and 'Unknown' in report['BAD Curncy']['error']
    assert report['SILENT Curncy'] == {'ok': False, 'error': 'no answer in the response',
                                       'fields': {}}

    with pytest.raises(BloombergEntitlementError):
        Walked([('SEC Curncy', 'NOT_ENTITLED', {})]).reference_data(['SEC Curncy'], ('NAME',))


def test_staleness_reads_the_terminals_date_never_the_wall_clock():
    """`stale` answers off LAST_UPDATE_DT at an explicit as-of: the nineteen-year SAONIA is
    flagged with its own date, a quote that cannot evidence freshness is flagged as exactly
    that, and yesterday's print passes - no wall clock enters the arithmetic.

    Killing mutation: a missing LAST_UPDATE_DT passed as fresh."""
    source = Walked([('SAONIA Index', None, {'LAST_UPDATE_DT': '2007-03-26'}),
                     ('ZARONIA Index', None, {'LAST_UPDATE_DT': '2026-08-26'}),
                     ('MUTE Index', None, {})])
    late = security_map.stale(source, ['SAONIA Index', 'ZARONIA Index', 'MUTE Index'],
                              as_of=AS_OF)
    assert late == {'SAONIA Index': '2007-03-26', 'MUTE Index': 'no LAST_UPDATE_DT'}


class Answering(Walked):
    """The terminal answering every SEED candidate as itself, live - what a first-use
    provisioning run meets on a workstation that has its entitlements."""

    def __init__(self):
        super().__init__([(security, None, row['fields'])
                          for security, row in full_report().items()])


class Refusing(Walked):
    """A session that refuses the walk - no terminal, no entitlement, a timeout mid-request -
    and so the only way to assert that a call probed NOTHING at all."""

    def __init__(self):
        super().__init__([])

    def _walk(self, securities, fields):
        raise BloombergRequestError('the terminal was asked about {} names'.format(
            len(securities)))


def provisioning_home(tmp_path, monkeypatch):
    """`DV_HOME` on a folder holding the desk's own seed - `SEED`, which `provision` reads in place
    of the packaged questionnaire it copies only where no seed is laid."""
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('DV_HOME', str(home))
    (home / 'seed.json').write_bytes(json.dumps(SEED, indent=1).encode('utf-8'))
    return home


def test_first_use_provisions_once_and_the_second_run_probes_nothing(tmp_path, monkeypatch):
    """First run: provisioning writes the map its probe evidenced to `$DV_HOME/security_map.json`,
    which `load()` with no path then reads - every DV_* tool agrees on the file without passing a
    path. The second run LOADS rather than rebuilds - a session that raises the moment it is walked
    passes it, and `created` says so.

    Killing mutation: an existing map ignored - the second run probes and the refusal surfaces."""
    home = provisioning_home(tmp_path, monkeypatch)
    document, created = discover.provision(Answering(), AS_OF)
    assert created is True
    assert document['blocks']['fx_spot']['USDZAR']['security'] == 'USDZAR BGN Curncy'
    assert json.loads((home / 'security_map.json').read_text(encoding='utf-8')) == document
    assert security_map.home() == str(home) and security_map.load() == document

    again, created = discover.provision(Refusing(), AS_OF)
    assert created is False
    assert again == document


def test_a_refused_probe_leaves_the_packaged_seed_behind_and_no_half_map(tmp_path, monkeypatch):
    """First use with no DV_HOME at all: the folder and a byte-for-byte copy of the shipped
    questionnaire are laid down BEFORE the terminal is asked, so a refusal leaves the seed to cut
    down and NO partial map to distrust.

    Killing mutation: the packaged seed not copied in - provision dies reading a seed it never
    laid."""
    home = tmp_path / 'home'
    monkeypatch.setenv('DV_HOME', str(home))
    with pytest.raises(BloombergRequestError):
        discover.provision(Refusing(), AS_OF)
    with open(security_map.packaged_seed(), 'rb') as handle:
        assert (home / 'seed.json').read_bytes() == handle.read()
    assert not (home / 'security_map.json').exists()


def test_progress_counts_answered_names_and_lands_exactly_on_the_total(tmp_path, monkeypatch):
    """`on_batch(done, total)` fires per chunk: the counts only rise, and the last chunk clamps to
    the total instead of overshooting by the batch remainder. Provisioning threads the same
    callback through.

    Killing mutation: the last chunk counted at its full batch width."""
    securities = [candidate.security for candidate in discover.candidates_from_seed(SEED)]
    total = len(securities)
    progress = []
    discover.probe(Answering(), securities, batch=5,
                   on_batch=lambda done, count: progress.append((done, count)))
    counted = [done for done, _ in progress]
    assert [count for _, count in progress] == [total] * len(progress)
    assert counted == sorted(set(counted)) and counted[-1] == total
    assert counted[0] == 5 and len(counted) == 1 + (total - 1) // 5

    provisioning_home(tmp_path, monkeypatch)
    threaded = []
    discover.provision(Answering(), AS_OF,
                       on_batch=lambda done, count: threaded.append((done, count)))
    assert threaded and threaded[-1] == (total, total)


class Element:
    """The two shapes `_request` touches on a blpapi request: an ARRAY it appends to and a ROW it
    sets named values on. Enough of the SDK's element to record what was assembled, no more."""

    def __init__(self):
        self.values, self.rows, self.named = [], [], {}

    def appendValue(self, value):
        self.values.append(value)

    def appendElement(self):
        row = Element()
        self.rows.append(row)
        return row

    def setElement(self, name, value):
        self.named[name] = value


class Request:
    def __init__(self):
        self.elements = {}

    def getElement(self, name):
        return self.elements.setdefault(name, Element())


class Assembling(BloombergSession):
    """A started session whose service hands out recording requests and whose event loop answers
    one empty RESPONSE - so what reaches `sendRequest` is the assertion."""

    def __init__(self):
        super().__init__()
        self.sent = []
        self._api = type('api', (), {'Event': type('Event', (), {'TIMEOUT': 0, 'RESPONSE': 1})})
        self._service = type('service', (), {'createRequest': staticmethod(
            lambda name: Request())})
        self._session = self

    def sendRequest(self, request):
        self.sent.append(request)

    def nextEvent(self, timeout_ms):
        return type('event', (), {'eventType': lambda self=None: 1, '__iter__':
                                  lambda self=None: iter(())})()


def test_an_override_rides_the_request_and_its_absence_sends_no_element():
    """`IVOL_MATURITY` and its kind are REQUEST parameters, not fields: without an `overrides`
    array a field whose name carries a tenor answers at the service's default maturity. The
    element is assembled only when one is asked for, so every existing caller sends the request
    it always sent.

    Killing mutation: the overrides array assembled on every request."""
    plain = Assembling()
    plain.reference_data_report(['NKY Index'], ('3MTH_IMPVOL_100.0%MNY_DF',))
    assert 'overrides' not in plain.sent[0].elements
    # and a canned two-argument walk, which is what every offline gate in this package overrides,
    # still answers under the new signature
    assert Walked([('GOOD Curncy', None, {'NAME': 'GOOD-NAME'})]).reference_data_report(
        ['GOOD Curncy'], ('NAME',))['GOOD Curncy']['ok'] is True

    session = Assembling()
    session.reference_data_report(['NKY Index'], ('12MTH_IMPVOL_100.0%MNY_DF', 'IVOL_MATURITY'),
                                  {'IVOL_MATURITY': '12M', 'IVOL_MONEYNESS': 100})
    request = session.sent[0]
    assert request.elements['securities'].values == ['NKY Index']
    assert request.elements['fields'].values == ['12MTH_IMPVOL_100.0%MNY_DF', 'IVOL_MATURITY']
    assert [row.named for row in request.elements['overrides'].rows] == [
        {'fieldId': 'IVOL_MATURITY', 'value': '12M'},
        {'fieldId': 'IVOL_MONEYNESS', 'value': '100'}]

    bulk = Assembling()
    bulk.bulk_reference_data_report(['NKY Index'], ('OPT_CHAIN',), {'SINGLE_DATE_OVERRIDE': 1})
    assert [row.named for row in bulk.sent[0].elements['overrides'].rows] == [
        {'fieldId': 'SINGLE_DATE_OVERRIDE', 'value': '1'}]


class Counting(Answering):
    """The answering terminal, remembering every name it was asked about."""

    def __init__(self):
        super().__init__()
        self.asked = []

    def _walk(self, securities, fields):
        self.asked.extend(securities)
        yield from super()._walk(securities, fields)


def test_extending_a_map_probes_only_the_names_it_has_never_heard_of(tmp_path, monkeypatch):
    """A seed that gains a curve costs the terminal that curve's names alone. Every entry the map
    already carries keeps its evidence untouched and is never re-asked; a new name the terminal
    does not answer lands on the ledger by name; a seed with nothing new probes nothing.

    Killing mutation: the known names not subtracted - the whole seed is asked again."""
    provisioning_home(tmp_path, monkeypatch)
    document, _ = discover.provision(Answering(), AS_OF)
    before = json.loads(json.dumps(document))
    seed = json.loads(json.dumps(SEED))
    seed['rates']['NEW-CURVE'] = {'currency': 'ZAR', 'prefix': 'NEWX', 'expect': 'NEW',
                                  'years': [1, 2]}
    terminal = Counting()
    grown, verdicts = discover.extend(document, seed, terminal, AS_OF)

    assert set(terminal.asked) == {'NEWX1 BGN Curncy', 'NEWX2 BGN Curncy'}
    assert grown['blocks']['fx_spot'] == before['blocks']['fx_spot']
    assert {item.candidate.security for item in verdicts} == set(terminal.asked)
    assert set(terminal.asked) <= set(grown['rejected'])

    again = Counting()
    assert discover.extend(grown, seed, again, AS_OF)[1] == [] and again.asked == []
