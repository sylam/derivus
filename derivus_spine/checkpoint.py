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

"""Checkpoint events: the deployment's signature over its own log position.

A chain proves only internal consistency; a checkpoint proves authenticity. It carries the
deployment's Ed25519 signature over the canonicalised pair `{"event_hash", "lsn"}` and nothing
else, and the verifying key is published at genesis as a firm-class policy blob, so a replica
holding no secret can check it. The append is ordinary - validated by the closed vocabulary,
sealed and chained like any other event, and covered by the checkpoints that follow it.
"""
from .canon import canonical_bytes
from .errors import HomeMissing


def write_checkpoint(log, actor=None):
    """Sign `log`'s current head and append the checkpoint event, returning its envelope.

    `actor` defaults to the seat that minted the home, read off LSN 1. Raises `HomeMissing` on an
    empty log.
    """
    lsn, head = log.head()
    if lsn == 0:
        raise HomeMissing(
            'there is nothing to checkpoint in {}: the log holds no events - mint the home with '
            '`DV_Spine init` so genesis and its first checkpoint exist'.format(log.home))
    signature = log.keys.sign_checkpoint(canonical_bytes({'event_hash': head, 'lsn': lsn}))
    return log.append('checkpoint', {'event_hash': head, 'lsn': lsn, 'signature': signature},
                      actor=actor or log.genesis_actor)
