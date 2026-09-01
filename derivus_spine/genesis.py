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

"""Minting a home - the first four events, so every later question is answerable from the log.

LSN 1 is the admin grant: who may govern this deployment, and the first row of the fold that
answers "could X approve in March". LSN 2 is the break-glass grant, declared beside the first admin
grant so a policy that strands the last admin is recoverable through a path that already existed.
LSN 3 publishes the checkpoint verifying key as a firm-class blob cited by hash, which is what lets
a replica check every signature from checkpoint one holding nothing secret. LSN 4 is the first
checkpoint, signed over the head after LSN 3 so the three grants are covered from the start.

The key blob is fsynced before the event citing it, and a second genesis over a directory that
already holds a log raises `HomeExists`.
"""
from pathlib import Path

from .checkpoint import write_checkpoint
from .errors import HomeExists
from .log import SpineLog
from .seal import Keys
from .store import BlobStore

#: The three genesis policy names, spelled at the mint rather than in a caller: they are the shape
#: a replica folds.
GENESIS_POLICY = 'genesis'
BREAK_GLASS_POLICY = 'break_glass'
VERIFYING_KEY_POLICY = 'checkpoint_verifying_key'


def init_home(home, actor):
    """Mint the spine home at `home`, attributed to `actor`, and return a summary of what was
    written.

    Directories, then keys, then the published verifying key blob, then the four genesis events
    through the ordinary writer. Raises `HomeExists` if `home` already holds a log.
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
    # Release the writer lock: the next writer is whoever the deployment runs.
    log.close()
    return {'home': str(home), 'actor': actor, 'events': lsn, 'head_lsn': lsn, 'head_hash': head,
            'checkpoint_verifying_key': blob}
