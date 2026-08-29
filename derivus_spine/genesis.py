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

"""Minting a home - four events that make every later question answerable from inside the log.

Genesis is not setup. It is the first four FACTS, and each one exists because of a question a
replica would otherwise have to ask a human.

LSN 1, the admin grant: who may govern this deployment. Authorization replays like everything
else - "could X approve in March" is a fold over policy declarations - and a fold needs a first
row.

LSN 2, the break-glass grant: declared BESIDE the first admin grant rather than after the accident.
A policy declaration that would strand the last admin or brick writes is recoverable only through a
path that already existed, and its use is itself an appended fact - so the recovery is in the
record rather than in someone's memory of a weekend.

LSN 3, the checkpoint verifying key, published as a firm-class blob and cited by hash. This is what
turns checkpoints from self-assertion into authenticity: a replica reads the key out of the log's
own history and checks every signature from checkpoint one, holding nothing secret.

LSN 4, the first checkpoint - signed over the head after LSN 3, so the three grants above are
covered by a signature from the moment the home exists.

The blob goes in BEFORE the event citing it: durability ordering is law, and genesis obeys it like
every other write. A second genesis over a directory that already holds a log is `HomeExists` - a
home is minted once, and forking the record is not a thing a retyped command should be able to do.
"""
from pathlib import Path

from .checkpoint import write_checkpoint
from .errors import HomeExists
from .log import SpineLog
from .seal import Keys
from .store import BlobStore

#: The three genesis policy bodies, spelled here rather than in a caller: they are the shape a
#: replica folds, so they belong to the mint.
GENESIS_POLICY = 'genesis'
BREAK_GLASS_POLICY = 'break_glass'
VERIFYING_KEY_POLICY = 'checkpoint_verifying_key'


def init_home(home, actor):
    """Mint the spine home at `home`, attributed to `actor`. Answers a summary of what was written.

    Directories, then keys, then the published verifying key, then the four events through the
    ordinary writer - nothing here is a special path into the log.
    """
    home = Path(home)
    log_dir = home / 'log'
    if log_dir.is_dir() and any(log_dir.glob('segment-*.jsonl')):
        raise HomeExists(
            '{} already holds a log: a home is initialised once, and a second genesis would fork '
            'the record - verify the home that is there, or mint the new one in a directory of '
            'its own'.format(home))
    for part in ('log', 'blobs', 'keys'):
        (home / part).mkdir(parents=True, exist_ok=True)

    keys = Keys.generate(home)
    # Put and fsync the key blob first: no event may cite a blob that is not yet on the platter.
    blob = BlobStore(home).put(keys.verifying_key())

    log = SpineLog(home)
    log.append('policy_declared',
               {'policy': GENESIS_POLICY, 'grants': [{'subject': actor, 'scope': 'admin'}]},
               actor=actor)
    log.append('policy_declared',
               {'policy': BREAK_GLASS_POLICY, 'grant': {'subject': actor}},
               actor=actor)
    log.append('policy_declared',
               {'policy': VERIFYING_KEY_POLICY, 'blob': blob},
               actor=actor, blob_refs=(blob,))
    write_checkpoint(log, actor=actor)

    lsn, head = log.head()
    # Genesis is done writing, so it lets go of the home: the next writer is whoever the deployment
    # runs, not the process that happened to mint the directory.
    log.close()
    return {'home': str(home), 'actor': actor, 'events': lsn, 'head_lsn': lsn, 'head_hash': head,
            'checkpoint_verifying_key': blob}
