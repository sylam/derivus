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

Five faults, none of them a patch. The writer is a REAL PROCESS and is killed with the operating
system, so the torn-tail rule is observed on a platter rather than asserted about one. A replica is
partitioned by not being told, and resumes by asking. A clock is skewed by a fact carrying a
truth-time older than the one standing, which is data the record is built to hold. A late print is
answered by a SECOND close that supersedes the first rather than correcting it. And an act repeated
is one fact, because a tuple carries no clock of the writer's own.

Every fault answers `{fault, expected, answer}` and none of them leaves the record unreadable: what
a fault proves is that the day carries on.
"""
import subprocess
import sys
import time
from pathlib import Path

from derivus_spine import SpineLog, replica, verify_home
from derivus_spine.projections import PROJECTORS, fold
from derivus_mcp import server as binding

from gates.spine_game import roles

ROOT = Path(__file__).resolve().parents[2]

#: How long a fault waits on a socket, a process or a beat before it says so rather than hanging.
WIRE_SECONDS = 20.0
#: The index the killed writer prints, and the two truth-times the skew is measured between. An
#: index no deal names is nothing to a compile, which is why a heartbeat may be printed at all.
HEARTBEAT = 'GAME.HEARTBEAT'
LATE, EARLY = '2024-06-28T17:00:00.000000Z', '2024-06-28T09:00:00.000000Z'
#: Where the late print puts the spot, which is what makes the restated close a different board.
LATE_SPOT = 18.6
#: How many prints the killed writer is told to make, and how long it is left running. Long enough
#: for several to be fsynced and short enough that it is still mid-loop when the kill lands.
PRINTS = 40
KILL_AFTER = 0.15

#: The writer, as a process of its own: it claims the home, prints, and is killed mid-loop. Run as
#: `-c` rather than as a file so there is no second module to keep in step with this one.
WRITER = """
import sys
from derivus_spine import SpineLog
log = SpineLog(sys.argv[1])
for at in range({prints}):
    log.append('fixing_observed',
               {{'index': {index!r}, 'date': '2024-06-{{:02d}}'.format(at % 28 + 1),
                 'source': 'GAME', 'value': float(at)}},
               actor=sys.argv[2])
    print(at, flush=True)
"""


def play(table, out, mirrors):
    """Every fault in turn, each answering `{fault, expected, answer}`."""
    return [kill_the_writer(table, out), partition_a_replica(table, mirrors),
            skew_a_clock(table), a_late_fixing_after_the_close(table), the_same_act_twice(table)]


def kill_the_writer(table, out):
    """Kill the process holding the home between two appends.

    THE TORN-TAIL RULE ON RESTART: a final line with no terminating newline was never durable and
    is truncated when the home is next opened, while a newline-terminated line that will not parse
    is a durable line somebody altered. So a killed writer costs at most the append it was making,
    the chain re-derives whole, and the desk writes the next frame onto it.
    """
    stood = _head(table)
    writing = subprocess.Popen(
        [sys.executable, '-c', WRITER.format(prints=PRINTS, index=HEARTBEAT),
         str(table.home), roles.CONTROL],
        cwd=str(ROOT), stdout=subprocess.PIPE, universal_newlines=True)
    try:
        # the first print is the first append's own fsync, so waiting for it is what makes the kill
        # land on a writer that has WRITTEN; the gate holds the head to it
        assert writing.stdout.readline(), 'the killed writer never printed, so it never appended'
        time.sleep(KILL_AFTER)
    finally:
        writing.kill()
        writing.wait(timeout=WIRE_SECONDS)
    checked = verify_home(table.home, entitled=False)
    landed = table.file(roles.CONTROL, 'fixing_observed',
                        {'index': HEARTBEAT, 'date': '2024-07-01', 'source': 'GAME',
                         'value': 1.0})
    table.did('the machinery', 'the writer was killed mid-append', stood=stood,
              killed_at=checked['head_lsn'], resumed_at=landed['lsn'])
    return _row('kill the hub between two appends',
                'the chain re-derives whole on restart and the next frame lands on it',
                'stood at {}, verified to {}, resumed at {}'.format(
                    stood, checked['head_lsn'], landed['lsn']))


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
            table.file(roles.CONTROL, 'fixing_observed',
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
        table.file(roles.CONTROL, 'fixing_observed',
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
    """A print arrives after the day was closed, so the desk re-marks and closes again.

    A close is SUPERSEDED and never corrected: the second names the position the first stood at, and
    a fold taken as at the first still answers what it answered. The mark moves with the print,
    because a market's identity is its NUMBERS - a close restating the same vector is one fact, and
    a day that really moved is two.
    """
    table.file(roles.CONTROL, 'fixing_observed',
               {'index': HEARTBEAT, 'date': roles.CLOSE_DATE, 'source': 'GAME', 'value': LATE_SPOT})
    binding.patch_market_values({'FxRate.ZAR': {'Spot': LATE_SPOT}})
    restated = binding.declare_close(date=roles.CLOSE_DATE, actor=roles.CONTROL)
    table.did(roles.CONTROL, 'declare_close (restated)', recorded=restated['recorded']['lsn'],
              supersedes_lsn=restated['supersedes_lsn'])
    return _row('a late fixing after the close', 'a second close supersedes the first',
                'LSN {} supersedes LSN {}'.format(restated['recorded']['lsn'],
                                                  restated['supersedes_lsn']))


def the_same_act_twice(table):
    """Say one thing twice.

    A retry is the same fact by construction - the semantic tuple carries no clock of the writer's
    own - so the second mark coalesces onto the position the first has and the head does not move.
    """
    first = binding.declare_market(roles.OFFICIAL, actor=roles.CONTROL)
    stood = _head(table)
    again = binding.declare_market(roles.OFFICIAL, actor=roles.CONTROL)
    table.did(roles.CONTROL, 'declare_market official (twice)',
              recorded=[first['recorded']['lsn'], again['recorded']['lsn']], head=stood)
    return _row('the same act twice', 'one fact at one LSN, and the head unmoved',
                'LSN {} then {}, head {} then {}'.format(
                    first['recorded']['lsn'], again['recorded']['lsn'], stood, _head(table)))


# ------------------------------------------------------------------------------------------------
# The pieces the faults above are made of.

def _row(fault, expected, answer):
    """One fault's row: what was broken, what was expected of the record, and what it did."""
    return {'fault': fault, 'expected': expected, 'answer': answer}


def _head(table):
    """Where the hub's record stands right now."""
    return table.read(lambda log: log.head()[0])
