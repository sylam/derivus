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

"""Play the bank: found it, stand a hub and its followers up, let the seats work, and hold the
record to it.

`python -m gates.spine_game.play [--red] [--faults] [--replicas N] [--out <dir>]`. The bank is
founded through the CLI in a clean folder; the hub is a real service on an EPHEMERAL PORT with that
home behind it, the followers are real replicas pulling real frames over a socket, and the binding
is bound to the hub the way a host binds it. What lands under `--out` is the day: the documents the
bank was founded with, the book it was played on, the SCRIPT of what was asked, the diary's rows
and the P&L the engine answered, and the oracle's report over each copy. The exit code is the
report - non-zero where an invariant did not hold.

WITH `--red` OFF THIS IS THE DEMO: three desks and the back office through three closes - the book
carried in and marked, a day's trading signed where the workflow wants a second seat, the day's
payments settled and confirmed, collateral called and posted, the P&L read and explained - and
three copies of the record that agree. With it on, the same days carry an adversary.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for reachable in (ROOT, ROOT / 'tests'):
    if str(reachable) not in sys.path:
        sys.path.insert(0, str(reachable))

from derivus_mcp import server as binding  # noqa: E402
from derivus_spine import SpineLog, oracle, replica  # noqa: E402
from derivus_spine.custody import CLASS_KEY_FILE  # noqa: E402
from derivus_spine.vocabulary import FIRM_CLASS  # noqa: E402

from gates.spine_game import faults, red, roles  # noqa: E402

#: Where a run lands when nobody says. Never tracked, like every other generated thing.
OUT = ROOT / 'artifacts' / 'spine_game'

#: What a day moves in the environment and puts back: the record's home, the seat the poll paths
#: run under, and the desk's own directory.
HOMES = ('DV_SPINE_HOME', 'DV_SPINE_ACTOR', 'DV_HOME')


def book_at(path):
    """Write the bank's book - the suite's market on the first day, a metal's forward curve and
    each client's credit curve beside it, and no deal - and answer the path.

    THE FILE STARTS WHERE THE RECORD DOES. A book carrying a deal nobody booked is a DIVERGENCE
    `/book/reconcile` names, so every position reaches the file through the hub, and each client's
    netting set arrives with its first booking out of the paper legal declared.
    """
    import pandas as pd

    from derivus import utils
    from test_service import FACTORS, dump, job

    def curve(*knots):
        return utils.Curve([], [list(knot) for knot in knots])

    factors = dict(FACTORS, **{
        'CommodityPrice.METAL': {'Spot': 2300.0, 'Currency': 'USD', 'Interest_Rate': 'USD',
                                 'Forward_Rate': 'METAL_CARRY'},
        'ForwardRate.METAL_CARRY': {'Currency': 'USD', 'Curve': curve((45400.0, 0.010),
                                                                      (46200.0, 0.015))},
        'ReferencePrice.METAL': {'Fixing_Curve': curve((40000, 40000), (60000, 60000)),
                                 'ForwardPrice': 'METAL'},
        'ForwardPrice.METAL': {'Currency': 'USD', 'Curve': curve((45400.0, 2300.0),
                                                                 (46200.0, 2400.0))}},
        **{'SurvivalProb.' + client[3]: {'Recovery_Rate': 0.4, 'Curve': curve((0.0, 0.0),
                                                                              (10.0, 0.2))}
           for client in roles.CLIENTS})
    first = pd.Timestamp(roles.D0)
    document = json.loads(dump(job(
        deals=(), factors=factors, Base_Date=first,
        sections={'Bootstrapper Configuration': {
            'FXVolSurfaceParameters': {'Prices': 'FXVol'}}})))
    document['Calc']['Deals']['Reference'] = roles.BANK
    document['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters'][
        'Base_Date'] = json.loads(dump({'day': first}))['day']
    path.write_text(json.dumps(document, indent=2), newline='\n')
    return path


def term_sheets():
    """`{reference: deal}` - the trades the desks book, in the wire form a host sends them in."""
    import pandas as pd

    import rates_world
    from derivus import utils
    from test_service import dump

    d0, d1, d2 = (pd.Timestamp(day) for day in (roles.D0, roles.D1, roles.D2))
    months = pd.DateOffset
    return dict((deal['Reference'], json.loads(dump(deal))) for deal in (
        {'Object': 'FixedCashflowDeal', 'Reference': 'CARRIED', 'Currency': 'ZAR',
         'Discount_Rate': 'ZAR', 'Calendars': None, 'Amount': 1_000_000.0, 'Payment_Date': d2},
        {'Object': 'FXForwardDeal', 'Reference': 'FWD-1', 'Buy_Currency': 'USD',
         'Buy_Amount': 200_000.0, 'Buy_Discount_Rate': 'USD', 'Sell_Currency': 'ZAR',
         'Sell_Amount': 10_810.0, 'Sell_Discount_Rate': 'ZAR', 'Settlement_Date': d2},
        dict(rates_world.deposit('DEP-1', 'ZAR', 'ZAR', 3, 2.0), Effective_Date=d2,
             Maturity_Date=d2 + months(months=3), Amount=5_000_000.0,
             Interest_Rate_Schedule=utils.DateList({d2: 2.0})),
        dict(rates_world.fra('FRA-1', 'ZAR', 'ZAR', 'ZAR', 3, 6, 2.0),
             Effective_Date=d1 + months(months=3), Maturity_Date=d1 + months(months=6),
             Reset_Date=d1 + months(months=3), Principal=50_000_000.0),
        dict(rates_world.par_swap('SWAP-1', 'ZAR', 'ZAR', 'ZAR', 2, 2.0), Effective_Date=d1,
             Maturity_Date=d1 + months(years=2), Principal=100_000_000.0),
        {'Object': 'CommodityForwardDeal', 'Reference': 'METAL-1', 'Commodity': 'METAL',
         'Currency': 'USD', 'Discount_Rate': 'USD', 'Buy_Sell': 'Buy', 'Reference_Type': 'METAL',
         'Forward_Date': d0 + months(months=6), 'Maturity_Date': d0 + months(months=6),
         'Units': 500.0}))


def hub_at(out):
    """Found the bank on a fresh home, write its book, and serve it on an ephemeral port.
    Answers `(table, the setup's acts)`.

    The environment is set before the first call that reads it: the service resolves its home per
    call rather than at import, so this is configuration and not a patch.
    """
    from derivus import service

    home, desk = out / 'hub', out / 'desk'
    desk.mkdir(parents=True, exist_ok=True)
    founded = roles.found(home, out / 'setup')
    os.environ['DV_SPINE_HOME'] = str(home)
    os.environ['DV_SPINE_ACTOR'] = roles.HUB
    os.environ['DV_HOME'] = str(desk)
    service.BOOK = service.Book(str(book_at(out / 'book.json')))
    table = roles.Table(home, term_sheets())
    table.serve()
    return table, founded


def followers(table, out, count):
    """`count` replicas of the hub, each a real home pulling over the socket. Answers their paths.

    ONE OF THEM IS ENTITLED - its class key put in place as a file copy, which is what a
    materialized replica looks like on disk - so the oracle can ask the questions that live inside
    a body. The rest stay chain-only, which is the posture that says what a keyless copy cannot
    assess.
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
    """Pull every replica up to the hub's head, the first with the blobs its chain cites under
    audit's seat."""
    caught = []
    for at, home in enumerate(mirrors):
        mirror = SpineLog(home)
        try:
            caught.append(replica.catch_up(
                mirror, replica.Hub(table.url, actor=roles.AUDIT), blobs=not at, bounded=True,
                whole=True))
        finally:
            mirror.close()
    return caught


def play(out, with_red=False, with_faults=False, replicas=2):
    """Play the bank's days and answer `(script, {copy: the oracle's report})`.

    The order is the days': the seats work, then the adversary, then the machinery goes wrong -
    because a fault that tore the record before the desk used it would be a day nobody played.
    """
    from derivus import service

    out.mkdir(parents=True, exist_ok=True)
    # the environment is where the desk's two homes are named, so it is put back: a day is an act
    # this function performs and undoes, not a setting it leaves behind
    stood = dict((named, os.environ.get(named)) for named in HOMES)
    began = time.monotonic()
    table, founded = hub_at(out)
    mirrors = followers(table, out, replicas)
    played = {}
    try:
        for _, _, act in roles.DAY:
            act(table)
        played['played_seconds'] = round(time.monotonic() - began, 1)
        if with_red:
            played['objectives'] = red.play(table, out)
        if with_faults:
            played['faults'] = faults.play(table, out, mirrors)
        caught = catch_up(table, mirrors)
    finally:
        table.stop()
        service.BOOK = None
        binding.SERVICE = None
        for named, was in stood.items():
            os.environ.pop(named, None) if was is None else os.environ.update({named: was})

    table.did('the deployment', 'catch_up', replicas=[caught['head_lsn'] for caught in caught])
    # the diary and the P&L are COMPILES of the book, not folds of the record, so what the engine
    # answered is handed to the oracle as data - the diary's rows over every day it was read on
    diary, pnl = dict(sorted(table.standing['diary'].items())), table.standing['pnl']
    script = dict(table.script('the bank, played'), founded=founded,
                  replicas=[str(home) for home in mirrors], **played)
    timed = time.monotonic()
    reports = {'hub': oracle.report(table.home, script=script, against=mirrors[0],
                                    diary_keys=diary, pnl=pnl),
               'replica-1': oracle.report(mirrors[0], script=script, against=table.home,
                                          diary_keys=diary, pnl=pnl)}
    reports.update(('replica-{}'.format(at + 2), oracle.report(home, against=table.home))
                   for at, home in enumerate(mirrors[1:]))
    script['oracle_seconds'] = round(time.monotonic() - timed, 2)
    for name, written in (('script', script), ('diary', diary), ('pnl', pnl),
                          ('oracle', reports)):
        (out / '{}.json'.format(name)).write_text(json.dumps(written, indent=2, sort_keys=True),
                                                 newline='\n')
    return script, reports


def main(argv=None):
    """Play the days named on the command line and report; 1 where an invariant did not hold."""
    parser = argparse.ArgumentParser(
        prog='spine_game', description='Found a bank, play its days against a real hub and hold '
                                       'the record to what was asked.')
    parser.add_argument('--red', action='store_true',
                        help='play the adversary beside the desks; with it off this is the demo')
    parser.add_argument('--faults', action='store_true',
                        help='break the machinery beside the desks: kill the hub, partition a '
                             'replica, skew a clock, close twice, act twice, settle twice')
    parser.add_argument('--replicas', type=int, default=2,
                        help='how many followers pull from the hub; the first is entitled')
    parser.add_argument('--out', type=str, default=str(OUT),
                        help='where the setup, the book, the script and the oracle\'s reports '
                             'land')
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
