"""Discovery turns the README's 'verify every security' from an instruction into a property.

Nothing here opens a socket: `verify` and `build_map` are pure over canned terminal answers -
the `normalize_fx_vol` seam - and the two session readers are driven through a stubbed event
walk. What is gated is the trust chain a map rides on: a candidate is believed only when the
terminal's own NAME says it is what it claims, a dead benchmark is refused on its update date
however sane its price reads (the SAONIA trap: 8.855 nineteen years after its last print), an
entry stripped of its evidence refuses to load by name, and the strict and tolerant readers are
one walk with two policies rather than two walks that drift.
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
               'pillars': [0.25]},
    'fx_spot': {'pairs': ['USDZAR']},
    'rates': {'ZAR': {'prefix': 'SASW', 'expect': 'ZAR SWAP QTR', 'years': [1, 5],
                      'overnight': {'security': 'ZARONIA Index',
                                    'expect': 'South African Overnight'}}},
    'swaption': {'ZAR': {'prefix': 'SASN', 'expect': 'ZAR SWPT NVOL',
                         'expiries': {'1Y': '01'}, 'tenor_years': [1, 10]}},
}


def answered(name, px_last=1.0, last_update='2026-08-26'):
    return {'ok': True, 'error': None,
            'fields': {'NAME': name, 'PX_LAST': px_last, 'LAST_UPDATE_DT': last_update}}


def full_report():
    """Every SEED candidate answered as itself, live - the baseline the mutations below break."""
    report = {'USDZAR BGN Curncy': answered('USD-ZAR X-RATE'),
              'ZARONIA Index': answered('South African Overnight Index'),
              'SASW1 BGN Curncy': answered('ZAR SWAP QTR (VS 3M) 1Y'),
              'SASW5 BGN Curncy': answered('ZAR SWAP QTR (VS 3M) 5Y'),
              'SASN011 Curncy': answered('ZAR SWPT NVOL 1Y1Y'),
              'SASN0110 Curncy': answered('ZAR SWPT NVOL 1Y10Y')}
    for tenor in ('1M', '1Y'):
        report['USDZARV{} BGN Curncy'.format(tenor)] = answered(
            'USD-ZAR OPT VOL {}'.format(tenor))
        report['USDZAR25R{} BGN Curncy'.format(tenor)] = answered(
            'USD-ZAR RR 25D {}'.format(tenor))
        report['USDZAR25B{} BGN Curncy'.format(tenor)] = answered(
            'USD-ZAR BFY 25D {}'.format(tenor))
    return report


def test_the_fx_grammar_spells_the_verified_shapes():
    """The spellings this grammar emits are the ones the terminal verified (session 2026-08-27),
    ticker and expected NAME both - so a reworded candidate turns up here before it turns up as
    a silent mismatch against the live terminal."""
    spelled = {candidate.security: candidate
               for candidate in discover.fx_vol_candidates('USDZAR', {'1M': 1 / 12}, [0.25, 0.1])}
    assert set(spelled) == {'USDZARV1M BGN Curncy', 'USDZAR25R1M BGN Curncy',
                            'USDZAR25B1M BGN Curncy', 'USDZAR10R1M BGN Curncy',
                            'USDZAR10B1M BGN Curncy'}
    atm = spelled['USDZARV1M BGN Curncy']
    assert atm.path == ('fx_vol', 'USDZAR', 'quotes', '1M', 'ATM')
    assert discover._matches('USD-ZAR OPT VOL 1M', atm.expect)
    assert discover._matches('USD-ZAR OPT VOL  1M', atm.expect)  # EURAUD-style double space
    assert not discover._matches('USD-ZAR RR 25D 1M', atm.expect)
    wing = spelled['USDZAR25R1M BGN Curncy']
    assert wing.path[-1] == 'RR_0.25'
    assert discover._matches('USD-ZAR RR 25D1M', wing.expect)  # the 10Y spelling drops its space


def test_the_strip_and_swaption_grammars_spell_their_suffixes():
    """The two encodings that cost a night to find: the OIS suffix is 1Z/2Z/3Z for weeks, bare
    letters A..K for months, plain integers for years; a swaption tenor is NEVER zero-padded -
    SASN011 is 1Y into 1Y, and the padded spelling resolves nothing."""
    strip = {candidate.path[-1]: candidate.security for candidate in discover.strip_candidates(
        'USD', {'prefix': 'USOSFR', 'expect': 'USD OIS', 'weeks': True, 'months': True,
                'years': [1, 10]})}
    assert strip['1W'] == 'USOSFR1Z BGN Curncy'
    assert strip['1M'] == 'USOSFRA BGN Curncy'
    assert strip['11M'] == 'USOSFRK BGN Curncy'
    assert strip['10Y'] == 'USOSFR10 BGN Curncy'

    swaption = {candidate.path[-1]: candidate.security for candidate in
                discover.swaption_candidates('ZAR', SEED['swaption']['ZAR'])}
    assert swaption == {'1Y x 1Y': 'SASN011 Curncy', '1Y x 10Y': 'SASN0110 Curncy'}


def test_verification_classifies_off_the_terminals_own_answers():
    """The order of distrust, one candidate each: refused by Bloomberg, answering as something
    else, resolving but priceless, priced but long dead, and live. The dead fixture is the
    SAONIA shape - a plausible level, an update date nineteen years old - which no price check
    can see and the date check must."""
    candidates = [
        discover.Candidate('GOOD Curncy', ('fx_spot', 'GOOD'), ('GOOD-NAME',)),
        discover.Candidate('GONE Curncy', ('fx_spot', 'GONE'), ('GONE-NAME',)),
        discover.Candidate('OTHER Curncy', ('fx_spot', 'OTHER'), ('OTHER-NAME',)),
        discover.Candidate('BLANK Curncy', ('fx_spot', 'BLANK'), ('BLANK-NAME',)),
        discover.Candidate('SAONIA Index', ('rates', 'ZAR', 'overnight'), ('South Africa',)),
    ]
    report = {
        'GOOD Curncy': answered('GOOD-NAME'),
        'GONE Curncy': {'ok': False, 'error': 'Unknown/Invalid Security', 'fields': {}},
        'OTHER Curncy': answered('SOMETHING ELSE ENTIRELY'),
        'BLANK Curncy': answered('BLANK-NAME', px_last=None),
        'SAONIA Index': answered('South Africa Overnight Avg', px_last=8.855,
                                 last_update='2007-03-26'),
    }
    verdicts = {item.candidate.security: item.verdict
                for item in discover.verify(candidates, report, AS_OF)}
    assert verdicts == {'GOOD Curncy': 'live', 'GONE Curncy': 'invalid',
                        'OTHER Curncy': 'mismatch', 'BLANK Curncy': 'unpriced',
                        'SAONIA Index': 'dead'}


def test_only_verified_entries_enter_the_map_and_the_rest_are_ledgered():
    """A candidate the terminal did not confirm lands on the `rejected` ledger by name, never
    silently dropped - and a live entry carries all three pieces of evidence."""
    report = full_report()
    report['USDZAR25B1Y BGN Curncy'] = {'ok': False, 'error': 'Unknown/Invalid Security',
                                        'fields': {}}
    document, verdicts = discover_with(report)
    entry = document['blocks']['fx_vol']['USDZAR']['quotes']['1M']['ATM']
    assert entry == {'security': 'USDZARV1M BGN Curncy', 'name': 'USD-ZAR OPT VOL 1M',
                     'last_update': '2026-08-26', 'verified': AS_OF.isoformat()}
    assert document['rejected']['USDZAR25B1Y BGN Curncy']['verdict'] == 'invalid'
    assert 'BF_0.25' not in document['blocks']['fx_vol']['USDZAR']['quotes']['1Y']
    # the expiries meta only names tenors that actually landed quotes
    assert set(document['blocks']['fx_vol']['USDZAR']['expiries']) == {'1M', '1Y'}


def discover_with(report):
    candidates = list(discover.candidates_from_seed(SEED))
    verdicts = discover.verify(candidates, report, AS_OF)
    return discover.build_map(SEED, verdicts, AS_OF.isoformat()), verdicts


def test_a_map_entry_without_evidence_is_refused_by_name(tmp_path):
    """The map is trusted BECAUSE each entry records what the terminal answered - so a
    hand-edited entry with the evidence stripped refuses to load, naming the entry, rather than
    riding into a fetch on the strength of being in the file."""
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
    pair, the pillar and the tenor, never a KeyError out of a dict lookup."""
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
    text still types the strict refusal."""
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
    that, and yesterday's print passes - no wall clock enters the arithmetic."""
    source = Walked([('SAONIA Index', None, {'LAST_UPDATE_DT': '2007-03-26'}),
                     ('ZARONIA Index', None, {'LAST_UPDATE_DT': '2026-08-26'}),
                     ('MUTE Index', None, {})])
    late = security_map.stale(source, ['SAONIA Index', 'ZARONIA Index', 'MUTE Index'],
                              as_of=AS_OF)
    assert late == {'SAONIA Index': '2007-03-26', 'MUTE Index': 'no LAST_UPDATE_DT'}
