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

"""What goes wrong with the machinery rather than with anybody's intent, and what converges anyway.

Six faults, none of them a patch. The HUB is a real process and is killed with the operating
system, so the torn-tail rule is observed on its platter rather than asserted about one. A replica
is partitioned by not being told, and resumes by asking. A clock is skewed by a fact carrying a
truth-time older than the one standing, which is data the record is built to hold. A late print is
answered by a SECOND close that supersedes the first rather than correcting it, and marked again.
And an act repeated - a mark, a settlement under its reference - is one fact, because a tuple
carries no clock of the writer's own.

Every fault answers `{fault, expected, answer}` and none of them leaves the record unreadable: what
a fault proves is that the day carries on.
"""
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from derivus_spine import SpineLog, replica, verify_home
from derivus_spine.projections import PROJECTORS, fold
from derivus_mcp import server as binding
from mcp.server.mcpserver.exceptions import ToolError

from gates.spine_game import roles

ROOT = Path(__file__).resolve().parents[2]

#: How long a fault waits on a socket, a process or a beat before it says so rather than hanging.
WIRE_SECONDS = 20.0
#: How long the hub, served as a process of its own, has to import the engine and answer.
STARTUP_SECONDS = 300.0
#: How long the kill waits once an append was acknowledged - long enough for several to land and
#: short of the stream's end, so it falls while product control is still marking.
KILL_AFTER = 0.5
#: The index a print no deal names carries, and the two truth-times the skew is measured between.
#: An index nothing reads is nothing to a compile, which is why a heartbeat may be printed at all.
HEARTBEAT = 'GAME.HEARTBEAT'
LATE, EARLY = '2024-06-28T17:00:00.000000Z', '2024-06-28T09:00:00.000000Z'
#: Where the late print puts the spot, which is what makes the restated close a different board.
LATE_SPOT = 18.6


def play(table, out, mirrors):
    """Every fault in turn, each answering `{fault, expected, answer}`."""
    return [kill_the_hub(table, out), partition_a_replica(table, mirrors),
            skew_a_clock(table), a_late_fixing_after_the_close(table), the_same_act_twice(table),
            the_same_settlement_twice(table)]


def kill_the_hub(table, out):
    """Kill the hub's own process between two appends, and serve it again on the tail it left.

    THE TORN-TAIL RULE ON RESTART: a final line with no terminating newline was never durable and
    is truncated when the home is next opened, while a newline-terminated line that will not parse
    is a durable line somebody altered. So the hub is served as a process of its own - `DV_Service`
    on the day's book - and killed by the operating system while product control's marks stream
    into it. Where the kill fell between two lines rather than inside one, the half line a kill
    mid-write leaves is put on its platter as DATA. The hub served again opens its home on that
    tail, every append it acknowledged is still there, the chain re-derives whole, and the next act
    lands on it - nothing in this process writing while the hub's own process holds the home.
    """
    table.stop()
    stood, acked = _head(table), []
    hub = _served(table, out, 'hub-killed')

    def stream():
        for at in range(1000):
            try:
                acked.append(_beat(at)['recorded']['lsn'])
            except ToolError:
                return

    marking = threading.Thread(target=stream)
    marking.start()
    deadline = time.monotonic() + WIRE_SECONDS
    while not acked and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(KILL_AFTER)
    hub.kill()
    hub.wait(timeout=WIRE_SECONDS)
    marking.join(timeout=WIRE_SECONDS)
    assert acked, 'the hub was killed before it acknowledged an append, so nothing could tear'

    tail = sorted((table.home / 'log').glob('segment-*.jsonl'))[-1]
    platter = tail.read_bytes()
    torn = not platter.endswith(b'\n')
    if not torn:
        last = platter[:-1].rsplit(b'\n', 1)[-1]
        with open(tail, 'ab') as handle:
            handle.write(last[:len(last) // 2])
    hub = _served(table, out, 'hub-restarted')
    try:
        restarted = binding.book_markets()['lsn']
        resumed = _beat('restarted')['recorded']['lsn']
    finally:
        hub.kill()
        hub.wait(timeout=WIRE_SECONDS)
    checked = verify_home(table.home, entitled=False)
    truncated = [line.strip() for line in (out / 'hub-restarted.log').read_text(
        errors='replace').splitlines() if 'torn byte' in line]
    table.serve()
    table.did('the machinery', 'the hub was killed mid-append and served again', stood=stood,
              acked=acked, torn=torn, restarted_at=restarted, resumed_at=resumed,
              verified=checked['head_lsn'], truncated=truncated)
    return _row('kill the hub between two appends',
                'the restarted hub truncates the torn tail, keeps every acknowledged append, '
                're-derives the chain and takes the next act',
                'stood at {}, {} append(s) acknowledged to LSN {}, killed; {} - the hub served '
                'again {}, stood at {}, verified to {}, resumed at {}'.format(
                    stood, len(acked), acked[-1],
                    'the kill tore the tail' if torn else 'half a line put on the platter',
                    'logged "{}"'.format(truncated[0]) if truncated else 'logged no truncation',
                    restarted, checked['head_lsn'], resumed))


def partition_a_replica(table, mirrors):
    """Stop telling a replica, write, and let it ask again.

    A beat is a NOTIFICATION and never a delivery, so a replica that heard nothing is behind rather
    than wrong: one catch-up covers however long it was away, and the head it lands on is the hub's.
    """
    home = mirrors[-1]
    mirror = SpineLog(home)
    try:
        replica.catch_up(mirror, replica.Hub(table.url), bounded=True, whole=True)
        behind = mirror.head()[0]
        for at in range(3):
            table.file(roles.PRODUCT_CONTROL, 'fixing_observed',
                       {'index': HEARTBEAT, 'date': '2024-07-{:02d}'.format(at + 2),
                        'source': 'GAME', 'value': float(at)})
        caught = replica.catch_up(mirror, replica.Hub(table.url), bounded=True)
    finally:
        mirror.close()
    table.did('the machinery', 'a replica was partitioned and resumed',
              behind=behind, caught=caught['frames'])
    return _row('partition a replica', 'one catch-up reaches the hub\'s head',
                'behind at {}, {} frame(s) pulled, standing at {} ({})'.format(
                    behind, caught['frames'], caught['head_lsn'], caught['head_hash'][:12]))


def skew_a_clock(table):
    """File a print whose truth-time is older than the one standing.

    Supersession is by `(effective_time, lsn)` under the whole key, so a backdated republication
    does not win by arriving last - and the print it did not beat stays ON THE ROW, because the
    record holds it and a projection may not hide it.
    """
    for stamp in (LATE, EARLY):
        table.file(roles.PRODUCT_CONTROL, 'fixing_observed',
                   {'index': HEARTBEAT, 'date': '2024-08-01', 'source': 'GAME',
                    'value': 1.0 if stamp == LATE else 9.9}, effective_time=stamp)
    standing = table.read(lambda log: fold(log, PROJECTORS['lifecycle'])['fixings'])
    row = standing[HEARTBEAT]['2024-08-01']['GAME']
    table.did('the machinery', 'a backdated print was filed', standing=row['value'],
              superseded=[beaten['value'] for beaten in row['supersedes']])
    return _row('skew a clock', 'the later truth-time stands and the backdated print is on the row',
                '{} stands over {}'.format(row['value'],
                                           [beaten['value'] for beaten in row['supersedes']]))


def a_late_fixing_after_the_close(table):
    """A print arrives after the day was closed, so the desk closes it again and marks it again.

    A close is SUPERSEDED and never corrected: the second names the position the first stood at,
    and a fold taken as at the first still answers what it answered. The mark moves with the
    print, because a market's identity is its NUMBERS - a close restating the same vector is one
    fact, and a day that really moved is two - and the last day marked is marked again, so the
    P&L reads the board the firm now stands on.
    """
    table.file(roles.PRODUCT_CONTROL, 'fixing_observed',
               {'index': HEARTBEAT, 'date': roles.D2, 'source': 'GAME', 'value': LATE_SPOT})
    binding.patch_market_values({'FxRate.ZAR': {'Spot': LATE_SPOT}})
    restated = binding.declare_close(date=roles.D2, actor=roles.PRODUCT_CONTROL)
    table.did(roles.PRODUCT_CONTROL, 'declare_close (restated)',
              recorded=restated['recorded']['lsn'], supersedes_lsn=restated['supersedes_lsn'])
    # the worklist lists a close by its day, and this day is marked: the desk marks it again itself
    roles.mark(table, roles.D2)
    return _row('a late fixing after the close', 'a second close supersedes the first',
                'LSN {} supersedes LSN {}'.format(restated['recorded']['lsn'],
                                                  restated['supersedes_lsn']))


def the_same_act_twice(table):
    """Say one thing twice.

    A retry is the same fact by construction - the semantic tuple carries no clock of the writer's
    own - so the second mark coalesces onto the position the first has and the head does not move.
    """
    first = binding.declare_market(roles.OFFICIAL, actor=roles.PRODUCT_CONTROL)
    stood = _head(table)
    again = binding.declare_market(roles.OFFICIAL, actor=roles.PRODUCT_CONTROL)
    table.did(roles.PRODUCT_CONTROL, 'declare_market official (twice)',
              recorded=[first['recorded']['lsn'], again['recorded']['lsn']], head=stood)
    return _row('the same act twice', 'one fact at one LSN, and the head unmoved',
                'LSN {} then {}, head {} then {}'.format(
                    first['recorded']['lsn'], again['recorded']['lsn'], stood, _head(table)))


def the_same_settlement_twice(table):
    """Settlements files the day's last payment again, under its own reference: the reference is
    the movement, so the second filing is the first fact at its LSN, and the money moved once."""
    paid, filed = table.standing['paid']
    again = binding.file_status(**paid)
    moved = [row for row in table.read(lambda log: PROJECTORS['cash'].rows(
        fold(log, PROJECTORS['cash']))) if row['reference'] == paid['reference']]
    table.did(roles.SETTLEMENTS, 'file_status payment (twice)',
              recorded=[filed, again['recorded']['lsn']], movements=len(moved))
    return _row('the same settlement filed twice', 'one fact at one LSN, one movement',
                'LSN {} then {}, {} movement(s) under {}'.format(
                    filed, again['recorded']['lsn'], len(moved), paid['reference']))


# ------------------------------------------------------------------------------------------------
# The pieces the faults above are made of.

def _row(fault, expected, answer):
    """One fault's row: what was broken, what was expected of the record, and what it did."""
    return {'fault': fault, 'expected': expected, 'answer': answer}


def _head(table):
    """Where the hub's record stands right now."""
    return table.read(lambda log: log.head()[0])


def _beat(at):
    """Product control marks the live board under a private name of its own - one append a beat,
    on a board no process resolves."""
    return binding.declare_market('private/{}/beat-{}'.format(roles.PRODUCT_CONTROL, at),
                                  actor=roles.PRODUCT_CONTROL)


def _served(table, out, name):
    """The hub served as a process of its own - `DV_Service` on the day's book and the home the
    environment names, on a port the box chose - once it answers, every seat pointed at it. Its
    output is `<name>.log` under `out`."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with open(out / '{}.log'.format(name), 'wb') as said:
        hub = subprocess.Popen(
            [sys.executable, '-m', 'derivus.service', '--book', str(out / 'book.json'), '--port',
             str(port)], cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
            stdout=said, stderr=subprocess.STDOUT)
    table.point('http://127.0.0.1:{}'.format(port))
    deadline = time.monotonic() + STARTUP_SECONDS
    while True:
        try:
            binding.book_markets()
            return hub
        except ToolError:
            if hub.poll() is not None or time.monotonic() > deadline:
                hub.kill()
                raise AssertionError('the hub process never answered - {}.log says why'.format(
                    name))
            time.sleep(0.25)
