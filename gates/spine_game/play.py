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

"""Play the day: stand a hub and its followers up, let the seats work, and hold the record to it.

`python -m gates.spine_game.play [--red] [--faults] [--replicas N] [--out <dir>]`. The hub is a real
service on an EPHEMERAL PORT with a real home behind it, the followers are real replicas pulling
real frames over a socket, and the binding is bound to the hub the way a host binds it. What lands
under `--out` is the day: the book it was played on, the SCRIPT of what was asked, and the oracle's
report over each copy. The exit code is the report - non-zero where an invariant did not hold.

WITH `--red` OFF THIS IS THE DEMO: one desk, one client, one structure quoted, accepted, signed and
booked, the day's settlement file struck and its payment filed, the close declared over attested
numbers, and three copies of the record that agree. With it on, the same day carries an adversary.

The book is the suite's own fixture plus a client node: a day needs a book with a counterparty in
it, and authoring a second one here would be a second book to keep in step with the first.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for reachable in (ROOT, ROOT / 'tests'):
    if str(reachable) not in sys.path:
        sys.path.insert(0, str(reachable))

from derivus_mcp import server as binding  # noqa: E402
from derivus_spine import SpineLog, init_home, oracle, replica  # noqa: E402
from derivus_spine.custody import CLASS_KEY_FILE  # noqa: E402
from derivus_spine.vocabulary import FIRM_CLASS  # noqa: E402

from gates.spine_game import faults, red, roles  # noqa: E402

#: The client the day quotes for, and the only node the suite's book does not already carry.
CLIENT_NODE = {'Object': 'NettingCollateralSet', 'Reference': roles.CLIENT, 'Netted': 'True',
               'Collateralized': 'False', 'Agreement_Currency': 'USD', 'Balance_Currency': 'USD',
               'Funding_Rate': 'USD', 'Liquidation_Period': 0.0, 'Settlement_Period': 0.0,
               'Credit_Support_Amounts': {'Counterparty': 'CPTY_A'}}

#: Where a run lands when nobody says. Never tracked, like every other generated thing.
OUT = ROOT / 'artifacts' / 'spine_game'

#: What a day moves in the environment and puts back: the record's home, the seat the poll paths
#: run under, and the desk's own directory.
HOMES = ('DV_SPINE_HOME', 'DV_SPINE_ACTOR', 'DV_HOME')


def book_at(path):
    """Write the day's book - the suite's job document, its deals left out, with an empty client
    node under it - and answer the path.

    THE FILE STARTS WHERE THE RECORD DOES. A book carrying a deal nobody booked is a DIVERGENCE
    `/book/reconcile` names, and that reading is the day's own control against a second writer, so
    a day that began already tripping it could never say anything with it. The position the desk
    opens holding is BOOKED through the hub instead, which is how a legacy trade reaches this
    record at all - there is no import verb.
    """
    from test_service import dump, job

    document = json.loads(dump(job(
        deals=(), sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}})))
    document['Calc']['Deals']['Deals']['Children'].append(
        {'Instrument': {'.Deal': CLIENT_NODE}, 'Children': []})
    path.write_text(json.dumps(document, indent=2), newline='\n')
    return path


def hub_at(out):
    """Mint the day's home, declare what scopes it, write its book, and serve it on an ephemeral
    port. Answers `(table, the pieces to shut down)`.

    The environment is set before the first call that reads it: the service resolves its home per
    call rather than at import, so this is configuration and not a patch.
    """
    from derivus import service
    from test_spine_doorbell import serving

    home, desk = out / 'hub', out / 'desk'
    desk.mkdir(parents=True, exist_ok=True)
    init_home(home, roles.CONTROL)
    log = SpineLog(home)
    try:
        roles.declare(log, roles.CONTROL)
    finally:
        log.close()

    os.environ['DV_SPINE_HOME'] = str(home)
    os.environ['DV_SPINE_ACTOR'] = roles.CONTROL
    os.environ['DV_HOME'] = str(desk)
    service.BOOK = service.Book(str(book_at(out / 'book.json')))
    url, server, running, held = serving(service.app)
    return roles.Table(url, home), (service, server, running, held)


def followers(table, out, count):
    """`count` replicas of the hub, each a real home pulling over the socket. Answers their paths.

    ONE OF THEM IS ENTITLED - its class key put in place as a file copy, which is what a
    materialized replica looks like on disk - so the oracle can ask the five questions that live
    inside a body. The rest stay chain-only, which is the posture that says what a keyless copy
    cannot assess.
    """
    mirrors = []
    for at in range(count):
        home = out / 'replica-{}'.format(at + 1)
        for part in ('log', 'blobs'):
            (home / part).mkdir(parents=True, exist_ok=True)
        if not at:
            (home / 'keys').mkdir(exist_ok=True)
            (home / 'keys' / CLASS_KEY_FILE.format(FIRM_CLASS)).write_bytes(
                (table.home / 'keys' / CLASS_KEY_FILE.format(FIRM_CLASS)).read_bytes())
        mirrors.append(home)
    return mirrors


def catch_up(table, mirrors):
    """Pull every replica up to the hub's head, the first one with the blobs its chain cites."""
    caught = []
    for at, home in enumerate(mirrors):
        mirror = SpineLog(home)
        try:
            caught.append(replica.catch_up(
                mirror, replica.Hub(table.url, actor=roles.CONTROL), blobs=not at, bounded=True,
                whole=True))
        finally:
            mirror.close()
    return caught


def play(out, with_red=False, with_faults=False, replicas=2):
    """Play one day and answer `(script, {copy: the oracle's report})`.

    The order is the day's: the seats work, then the adversary, then the machinery goes wrong -
    because a fault that tore the record before the desk used it would be a day nobody played.
    """
    out.mkdir(parents=True, exist_ok=True)
    # the environment is where the desk's two homes are named, so it is put back: a day is an act
    # this function performs and undoes, not a setting it leaves behind
    stood = dict((named, os.environ.get(named)) for named in HOMES)
    table, standing = hub_at(out)
    service, server, running, held = standing
    mirrors = followers(table, out, replicas)
    played = {}
    try:
        for _, _, act in roles.DAY:
            act(table)
        if with_red:
            played['objectives'] = red.play(table, out)
        if with_faults:
            played['faults'] = faults.play(table, out, mirrors)
        caught = catch_up(table, mirrors)
        # the diary is a COMPILE of the book and not a fold of the record, so the seventh
        # invariant's keys are read here, in the hub's own process, and handed in as data
        keys = [row['key'] for row in binding.book_diary()['rows'] if row['key']]
    finally:
        server.should_exit = True
        running.join(timeout=faults.WIRE_SECONDS)
        held.close()
        service.BOOK = None
        binding.SERVICE = None
        for named, was in stood.items():
            os.environ.pop(named, None) if was is None else os.environ.update({named: was})

    table.did('the deployment', 'catch_up', replicas=[caught['head_lsn'] for caught in caught])
    script = dict(table.script('the desk, played'),
                  replicas=[str(home) for home in mirrors], **played)
    (out / 'script.json').write_text(json.dumps(script, indent=2, sort_keys=True), newline='\n')
    reports = {'hub': oracle.report(table.home, script=script, against=mirrors[0],
                                    diary_keys=keys),
               'replica-1': oracle.report(mirrors[0], script=script, against=table.home,
                                          diary_keys=keys)}
    reports.update(('replica-{}'.format(at + 2), oracle.report(home, against=table.home))
                   for at, home in enumerate(mirrors[1:]))
    (out / 'oracle.json').write_text(json.dumps(reports, indent=2, sort_keys=True), newline='\n')
    return script, reports


def main(argv=None):
    """Play the day named on the command line and report; 1 where an invariant did not hold."""
    parser = argparse.ArgumentParser(
        prog='spine_game', description='Play a day on the desk against a real hub and hold the '
                                       'record to what was asked.')
    parser.add_argument('--red', action='store_true',
                        help='play the adversary beside the desk; with it off this is the demo')
    parser.add_argument('--faults', action='store_true',
                        help='break the machinery beside the desk: kill the writer, partition a '
                             'replica, skew a clock, close twice, act twice')
    parser.add_argument('--replicas', type=int, default=2,
                        help='how many followers pull from the hub; the first is entitled')
    parser.add_argument('--out', type=str, default=str(OUT),
                        help='where the book, the script and the oracle\'s reports land')
    args = parser.parse_args(argv)

    script, reports = play(Path(args.out), args.red, args.faults, args.replicas)
    json.dump({'acts': len(script['acts']),
               'objectives': len(script.get('objectives', ())),
               'faults': len(script.get('faults', ())),
               'reports': dict((copy, oracle.failed(found)) for copy, found in reports.items())},
              sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write('\n')
    return 1 if any(oracle.failed(found) for found in reports.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
