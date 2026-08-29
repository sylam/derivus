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

"""The deployment signing where it stands - the one thing in the log a replica cannot forge.

A chain proves internal consistency and nothing else: hand someone a log and they can re-derive
every link in it, including a log rewritten from genesis. A CHECKPOINT is the deployment's Ed25519
signature over the pair `(lsn, event_hash)`, and the verifying key is published at genesis as a
firm-class policy blob, so a replica asserts AUTHENTICITY - this really is the history that hub
wrote, up to here - from checkpoint one, holding no secret at all.

The append is ordinary on purpose. A checkpoint is a fact like any other: validated by the closed
vocabulary, tagged, sealed, chained, and covered by the checkpoints that come after it. Nothing in
the writer knows this event is special, which is what keeps the mechanism honest - the signature is
the point, and it is data inside a body rather than a privilege in the log.

The signed payload is deliberately thin: `{"event_hash", "lsn"}` canonicalised, and nothing else.
An external anchor hook exports exactly those two fields; anything richer would be a second
statement about the log that could disagree with the log.
"""
from .canon import canonical_bytes
from .errors import HomeMissing


def write_checkpoint(log, actor=None):
    """Sign `log`'s current head and append the checkpoint. Answers the envelope.

    `actor` defaults to the seat that minted the home - read off LSN 1's envelope - because a
    periodic checkpoint is the deployment attesting to its own position, not a person acting. A
    deployment that wants an operator's name on it passes one.
    """
    lsn, head = log.head()
    if lsn == 0:
        raise HomeMissing(
            'there is nothing to checkpoint in {}: the log holds no events - mint the home with '
            '`DV_Spine init` so genesis and its first checkpoint exist'.format(log.home))
    signature = log.keys.sign_checkpoint(canonical_bytes({'event_hash': head, 'lsn': lsn}))
    return log.append('checkpoint', {'event_hash': head, 'lsn': lsn, 'signature': signature},
                      actor=actor or log.genesis_actor)
