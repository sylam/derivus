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
"""Build and re-verify a workstation's Bloomberg security map - `DV_Bloomberg`.

The package still ships NO ticker vocabulary: which pairs, which curve prefixes, which grids is
a SEED file the caller owns (the README carries a starting one), and nothing enters a map that
this workstation's terminal did not answer for. Discovery asks Bloomberg what each candidate IS
(`NAME`), when it last printed (`LAST_UPDATE_DT`) and whether it prices (`PX_LAST`), and writes
only the candidates whose answers match - so a map ENTRY is evidence, and `security_map.load`
refuses one that carries none. The dead trap this exists for is real: a retired benchmark keeps
returning a plausible `PX_LAST` with no error, and the update date is the only thing that says
so (SAONIA read 8.855 nineteen years after its last print).

What stays in code is only the spelling grammar - how a vol ticker composes from a pair, a
pillar and a tenor, how an OIS strip suffixes weeks, months and years - because a spelling is
verified per candidate against `NAME` before it is believed. `verify` and `build_map` are pure,
so the gates drive them on canned terminal answers, the `normalize_fx_vol` seam.
"""
import datetime
import json
import os
import shutil
from collections import namedtuple

from . import security_map
from .errors import BloombergConfigurationError
from .security_map import SCHEMA, entries, load

#: What discovery asks of every candidate: what it is, whether it prices, when it last did.
FIELDS = ('NAME', 'PX_LAST', 'LAST_UPDATE_DT')

#: A LAST_UPDATE_DT older than this many days marks a quote dead however sane its price looks.
STALE_DAYS = 5

#: One request per chunk - the Desktop API takes large batches, but a bounded request keeps a
#: partial outage partial.
BATCH = 50

#: The OIS-strip tenor suffix has three forms in one family: integer+Z for weeks, a bare letter
#: A..K for months 1..11, and the plain year number - USOSFR1Z / USOSFRA / USOSFR1.
WEEK_SUFFIX = {'1W': '1Z', '2W': '2Z', '3W': '3Z'}
MONTH_SUFFIX = {'{}M'.format(month + 1): code for month, code in enumerate('ABCDEFGHIJK')}

Candidate = namedtuple('Candidate', 'security path expect')
Verdict = namedtuple('Verdict', 'candidate verdict name last_update error')


def _dashed(pair):
    return pair[:3] + '-' + pair[3:]


def _pillar_code(pillar):
    return str(round(pillar * 100))


def fx_vol_candidates(pair, expiries, pillars):
    """The broker-grid spelling, checked against Bloomberg's own naming of it: `{PAIR}V{TENOR}`
    is named `XXX-YYY OPT VOL {TENOR}`, the wings `RR {DD}D` / `BFY {DD}D` - so a candidate that
    resolves but is not the quote it claims to be reads as a mismatch, never as a number."""
    dashed = _dashed(pair)
    for tenor in expiries:
        yield Candidate('{}V{} BGN Curncy'.format(pair, tenor),
                        ('fx_vol', pair, 'quotes', tenor, 'ATM'),
                        (dashed, 'OPT VOL', tenor))
        for pillar in pillars:
            code = _pillar_code(pillar)
            key = '{:.2f}'.format(pillar)
            yield Candidate('{}{}R{} BGN Curncy'.format(pair, code, tenor),
                            ('fx_vol', pair, 'quotes', tenor, 'RR_' + key),
                            (dashed, 'RR {}D'.format(code), tenor))
            yield Candidate('{}{}B{} BGN Curncy'.format(pair, code, tenor),
                            ('fx_vol', pair, 'quotes', tenor, 'BF_' + key),
                            (dashed, 'BFY {}D'.format(code), tenor))


def fx_spot_candidates(pairs):
    for pair in pairs:
        yield Candidate('{} BGN Curncy'.format(pair), ('fx_spot', pair),
                        (_dashed(pair), 'X-RATE'))


def strip_candidates(currency, spec):
    """A swap strip from a seed-supplied prefix. The expected NAME is the seed's `expect`
    fragment alone - strip names spell their tenors too inconsistently to grammar ('1WK',
    'SASW10' truncating - unlike the FX grid, which is checked tenor and all)."""
    prefix, expect = spec['prefix'], (spec['expect'],)
    suffixes = {}
    if spec.get('weeks'):
        suffixes.update(WEEK_SUFFIX)
    if spec.get('months'):
        suffixes.update(MONTH_SUFFIX)
    suffixes.update({'{}Y'.format(year): str(year) for year in spec.get('years', [])})
    for label, suffix in suffixes.items():
        yield Candidate('{}{} BGN Curncy'.format(prefix, suffix),
                        ('rates', currency, 'strip', label), expect)
    overnight = spec.get('overnight')
    if overnight:
        yield Candidate(overnight['security'], ('rates', currency, 'overnight'),
                        (overnight['expect'],))
    for label, fixing in spec.get('fixings', {}).items():
        yield Candidate(fixing['security'], ('rates', currency, 'fixings', label),
                        (fixing['expect'],))


def swaption_candidates(currency, spec):
    """ATM swaption vols: expiry is a TWO-character code (0A=1M, 0C=3M, 01=1Y, 10=10Y), the swap
    tenor the plain year number with no padding - SASN011 is 1Y into 1Y, and a zero-padded tenor
    silently matches nothing."""
    prefix, expect = spec['prefix'], (spec['expect'],)
    for expiry, code in spec['expiries'].items():
        for tenor in spec['tenor_years']:
            yield Candidate('{}{}{} Curncy'.format(prefix, code, tenor),
                            ('swaption', currency, '{} x {}Y'.format(expiry, tenor)), expect)


def candidates_from_seed(seed):
    fx_vol = seed.get('fx_vol', {})
    for pair in fx_vol.get('pairs', []):
        yield from fx_vol_candidates(pair, fx_vol.get('expiries', {}), fx_vol.get('pillars', []))
    yield from fx_spot_candidates(seed.get('fx_spot', {}).get('pairs', []))
    for currency, spec in seed.get('rates', {}).items():
        yield from strip_candidates(currency, spec)
    for currency, spec in seed.get('swaption', {}).items():
        yield from swaption_candidates(currency, spec)


def _squash(text):
    return ''.join(str(text).split()).upper()


def _matches(name, expect):
    squashed = _squash(name)
    return all(_squash(fragment) in squashed for fragment in expect)


def _is_stale(last_update, as_of, stale_days):
    if last_update in (None, ''):
        return False  # absence cannot prove staleness; the date, when present, can
    return (as_of - datetime.date.fromisoformat(str(last_update)[:10])).days > stale_days


def verify(candidates, report, as_of, stale_days=STALE_DAYS):
    """Classify every candidate off the terminal's own answers - pure, no socket. The order of
    the checks is the order of distrust: a name Bloomberg refused, a name that answers as
    something else, a security that resolves but prices nothing, a price whose update date says
    it stopped meaning anything, and only then a live quote."""
    verdicts = []
    for candidate in candidates:
        row = report.get(candidate.security, {'ok': False, 'error': 'not probed', 'fields': {}})
        fields = row['fields']
        name = fields.get('NAME')
        if not row['ok']:
            verdict = 'invalid'
        elif not _matches(name or '', candidate.expect):
            verdict = 'mismatch'
        elif fields.get('PX_LAST') in (None, ''):
            verdict = 'unpriced'
        elif _is_stale(fields.get('LAST_UPDATE_DT'), as_of, stale_days):
            verdict = 'dead'
        else:
            verdict = 'live'
        last_update = fields.get('LAST_UPDATE_DT')
        verdicts.append(Verdict(candidate, verdict, name,
                                None if last_update in (None, '') else str(last_update),
                                row['error']))
    return verdicts


def build_map(seed, verdicts, generated):
    """Only `live` verdicts become map entries, each carrying its evidence - the NAME the
    terminal answered, the quote's own last print, and when this map verified it. Everything
    else lands on the `rejected` ledger BY NAME, because a candidate silently dropped is
    indistinguishable from one never asked about."""
    blocks, rejected = {}, {}
    for item in verdicts:
        if item.verdict != 'live':
            rejected[item.candidate.security] = {
                'verdict': item.verdict, 'name': item.name,
                'last_update': item.last_update, 'error': item.error}
            continue
        node = blocks
        for part in item.candidate.path[:-1]:
            node = node.setdefault(part, {})
        node[item.candidate.path[-1]] = {'security': item.candidate.security, 'name': item.name,
                                         'last_update': item.last_update, 'verified': generated}
    for pair, block in blocks.get('fx_vol', {}).items():
        quoted = set(block.get('quotes', {}))
        block['expiries'] = {label: fraction for label, fraction
                             in seed.get('fx_vol', {}).get('expiries', {}).items()
                             if label in quoted}
    return {'schema': SCHEMA, 'generated': generated, 'blocks': blocks, 'rejected': rejected}


def probe(session, securities, fields=FIELDS, batch=BATCH, on_batch=None):
    """Every candidate asked in bounded chunks. `on_batch(done, total)`, when given, is called
    after each chunk ANSWERS - a full seed is several hundred names over a terminal that takes
    its time, so `done` counts names replied about, never names sent."""
    report = {}
    for start in range(0, len(securities), batch):
        report.update(session.reference_data_report(securities[start:start + batch], fields))
        if on_batch is not None:
            on_batch(min(start + batch, len(securities)), len(securities))
    return report


def discover(seed, session, as_of, stale_days=STALE_DAYS, on_batch=None):
    """Seed in, verified map out: candidates spelled from the seed's vocabulary, probed in
    batches, classified, and assembled with their evidence. `on_batch` is the probe's own
    progress, passed through untouched so a caller with a screen has something to show."""
    candidates = list(candidates_from_seed(seed))
    if not candidates:
        raise BloombergConfigurationError('the seed names nothing to discover')
    report = probe(session, [candidate.security for candidate in candidates], on_batch=on_batch)
    verdicts = verify(candidates, report, as_of, stale_days)
    return build_map(seed, verdicts, as_of.isoformat()), verdicts


def provisioned(home=None):
    """The map on disk, or None - the question `provision` answers first, asked on its own.

    A ROUTINE fetch (a cadence, a cron) must never provision: verifying a workstation's whole
    vocabulary is minutes of terminal time and an interactive act. So it asks this before it
    opens a session at all, and refuses by name on None rather than discovering its way into a
    map nobody was watching being built.
    """
    path = os.path.join(home or security_map.home(), 'security_map.json')
    return path if os.path.isfile(path) else None


def provision(session, as_of, home=None, stale_days=STALE_DAYS, on_batch=None):
    """First use, once: `(document, created)` for `$DV_HOME/security_map.json`.

    A map that is already there is LOADED and returned with `created` False - no probe, no
    write - so a desk that has cut its seed down and verified it keeps that map until it asks
    for another. With no map, the home folder and the seed are laid down FIRST: the packaged
    questionnaire is copied in byte for byte, so what the user meets is a real file to edit
    rather than an instruction to go find one. Only then is the terminal asked.

    That order is the point of the ordering: a Bloomberg failure propagates AFTER the folder
    and seed exist, so a refusal leaves the user with the seed to cut down and NO half-written
    map - and the retry starts from an edited seed instead of from a map nobody trusts.
    """
    home = home or security_map.home()
    map_path = provisioned(home)
    if map_path:
        return load(map_path), False
    map_path = os.path.join(home, 'security_map.json')
    os.makedirs(home, exist_ok=True)
    seed_path = os.path.join(home, 'seed.json')
    if not os.path.isfile(seed_path):
        shutil.copyfile(security_map.packaged_seed(), seed_path)
    with open(seed_path, encoding='utf-8') as handle:
        seed = json.load(handle)
    document, _ = discover(seed, session, as_of, stale_days, on_batch=on_batch)
    with open(map_path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(document, handle, indent=1)
    return document, True


def recheck(document, session, as_of, stale_days=STALE_DAYS):
    """Re-verify an existing map against the terminal standing now: each entry is expected to
    still BE what its recorded evidence says. Drift - renamed, unpriced, gone stale, gone
    entirely - is reported per entry; a quiet map is exit 0, so this runs from cron."""
    recorded = list(entries(document))
    report = probe(session, [entry['security'] for _, entry in recorded])
    drifted = {}
    for path, entry in recorded:
        row = report.get(entry['security'], {'ok': False, 'error': 'not probed', 'fields': {}})
        fields = row['fields']
        if not row['ok']:
            drift = 'invalid: {}'.format(row['error'])
        elif _squash(fields.get('NAME')) != _squash(entry['name']):
            drift = 'renamed: {!r} was verified as {!r}'.format(fields.get('NAME'), entry['name'])
        elif fields.get('PX_LAST') in (None, ''):
            drift = 'unpriced'
        elif _is_stale(fields.get('LAST_UPDATE_DT'), as_of, stale_days):
            drift = 'stale since {}'.format(fields.get('LAST_UPDATE_DT'))
        else:
            continue
        drifted['/'.join(path)] = {'security': entry['security'], 'drift': drift}
    return drifted


def main():
    """`DV_Bloomberg` - build a map from a seed, or re-verify one, on a terminal workstation."""
    import argparse

    from .session import BloombergSession

    parser = argparse.ArgumentParser(
        description='Build and re-verify a Bloomberg security map for derivus market data.')
    from .security_map import home

    verbs = parser.add_subparsers(dest='verb', required=True)
    discovering = verbs.add_parser('discover', help='probe a seed and write a verified map')
    discovering.add_argument('--seed', default=os.path.join(home(), 'seed.json'),
                             help='the vocabulary file you own (default: DV_HOME/seed.json)')
    discovering.add_argument('--out', default=os.path.join(home(), 'security_map.json'),
                             help='where the map lands (default: DV_HOME/security_map.json)')
    checking = verbs.add_parser('verify', help='re-probe an existing map and report drift')
    checking.add_argument('--map', default=os.path.join(home(), 'security_map.json'),
                          dest='map_path', help='the map to re-probe (default: DV_HOME/security_map.json)')
    for verb in (discovering, checking):
        verb.add_argument('--stale-days', type=int, default=STALE_DAYS,
                          help='a LAST_UPDATE_DT older than this marks a quote dead')
    args = parser.parse_args()

    as_of = datetime.date.today()
    if args.verb == 'discover':
        # the CLI keeps the seed deliberate where `provision` copies it: a hand-run discovery
        # refuses on a missing seed so the desk cuts its scope FIRST, and the refusal names the
        # packaged questionnaire to start from
        if not os.path.isfile(args.seed):
            raise SystemExit(
                'no seed at {} - copy the packaged questionnaire ({}) there and cut it to the '
                'scope your desk quotes, or name one with --seed'.format(
                    args.seed, security_map.packaged_seed()))
        with open(args.seed, encoding='utf-8') as handle:
            seed = json.load(handle)
        with BloombergSession(timeout_ms=30000) as session:
            document, verdicts = discover(seed, session, as_of, args.stale_days)
        with open(args.out, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(document, handle, indent=1)
        counts = {}
        for item in verdicts:
            counts[item.verdict] = counts.get(item.verdict, 0) + 1
        print('{} candidates: {}'.format(len(verdicts), ', '.join(
            '{} {}'.format(count, verdict) for verdict, count in sorted(counts.items()))))
        for security, entry in sorted(document['rejected'].items()):
            print('  {:<10} {} {}'.format(entry['verdict'], security, entry['error'] or ''))
        print('written to {}'.format(args.out))
        return 0

    document = load(args.map_path)
    with BloombergSession(timeout_ms=30000) as session:
        drifted = recheck(document, session, as_of, args.stale_days)
    for path, entry in sorted(drifted.items()):
        print('DRIFT {} ({}): {}'.format(path, entry['security'], entry['drift']))
    print('{} entries, {} drifted'.format(len(list(entries(document))), len(drifted)))
    return 1 if drifted else 0


if __name__ == '__main__':
    raise SystemExit(main())
