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

"""The derivus trading spine - the append-only book of record the engine is priced against.

A sibling package like `derivus_mcp` and `derivus_bloomberg`: shipped in the same wheel, holding
none of the engine, and importing STDLIB PLUS `cryptography` and nothing else - not `derivus`, not
torch, not requests (a gate in `tests/test_spine_imports.py` reads the source and says so). The
truth layer is files: fsynced append-only JSONL segments chained from genesis, and a
content-addressed blob tree beside them. Everything else - positions, the blotter, lifecycle
state - is a fold, and a wrong fold is fixed by fixing the projector and replaying, never by
editing a row.

Nothing here is ever edited in place. The writer refuses; it does not repair.
"""

from .canon import canonical_bytes, content_hash
from .errors import (
    SpineRefusal,
    CanonRefusal,
    UnknownEventType,
    MalformedEvent,
    CollisionRefusal,
    MissingBlobRefusal,
    ChainBroken,
    SealedBodyUnreadable,
    CheckpointInvalid,
    WriterBusy,
    HomeExists,
    HomeMissing,
)
from .store import BlobStore
from .log import SpineLog
from .checkpoint import write_checkpoint
from .genesis import init_home
from .verify import verify_home

# One import for a caller, and a surface small enough to read: mint a home, open its log, verify it,
# sign its head, address bytes by content. Everything else - the keys, the vocabulary, the wire
# format's own helpers - is reached module by module by the code that has business with it.
__all__ = [
    'canonical_bytes',
    'content_hash',
    'BlobStore',
    'SpineLog',
    'init_home',
    'verify_home',
    'write_checkpoint',
    'SpineRefusal',
    'CanonRefusal',
    'UnknownEventType',
    'MalformedEvent',
    'CollisionRefusal',
    'MissingBlobRefusal',
    'ChainBroken',
    'SealedBodyUnreadable',
    'CheckpointInvalid',
    'WriterBusy',
    'HomeExists',
    'HomeMissing',
]
