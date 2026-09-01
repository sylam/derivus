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

A sibling package to `derivus_mcp` and `derivus_bloomberg`: shipped in the same wheel, holding
none of the engine, and importing stdlib plus `cryptography` and nothing else. Truth is files -
fsynced append-only JSONL segments chained from genesis, and a content-addressed blob tree beside
them. Positions, the blotter and lifecycle state are folds over that log; a wrong fold is fixed by
fixing the projector and replaying, never by editing a row. Nothing here is edited in place.
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
    IdentityRefused,
    CapabilityDenied,
    CustodyRefusal,
    ReplayRefused,
    QuoteNotFirm,
)
from .store import BlobStore
from .log import SpineLog
from .checkpoint import write_checkpoint
from .genesis import init_home
from .verify import verify_home

# The small surface: mint a home, open its log, verify it, sign its head, address bytes by
# content. The keys, the vocabulary and the wire helpers are reached module by module.
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
    'IdentityRefused',
    'CapabilityDenied',
    'CustodyRefusal',
    'ReplayRefused',
    'QuoteNotFirm',
]
