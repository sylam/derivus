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

"""Every way the spine says no, typed - because a refusal is the design working, not the design
failing.

The record never repairs: nothing is edited in place, a duplicate that is not identical is not
deduplicated, an event citing a blob nobody fsynced does not append. That discipline is only real
if the writer has a name for each refusal and the caller can catch the one it means to handle, so
these are the vocabulary of the whole package and every module raises from this list rather than
inventing its own. `SpineRefusal` is the one thing to catch at a boundary - the CLI turns it into
exit 1 and prints the refusal's own wording, which is why every message here must NAME the thing
that was wrong and the remedy that fixes it.

They are all deliberately thin. The message carries the case (this LSN, this hash, this field);
the class carries the kind, so callers branch on the kind and operators read the case.
"""


class SpineRefusal(Exception):
    """The spine declining to do something it will not do. Base of every refusal below."""


class CanonRefusal(SpineRefusal):
    """A value that has no RFC 8785 spelling - a non-finite float, an integer past the
    double-exact range, a type JSON does not have."""


class UnknownEventType(SpineRefusal):
    """An event type outside the closed vocabulary. Consequence-shaped types land here: a knock
    is a projection, never a fact."""


class MalformedEvent(SpineRefusal):
    """A body that does not satisfy its type's validator - a missing field, a wrong shape, or a
    surplus key where the type does not admit one."""


class CollisionRefusal(SpineRefusal):
    """Two different byte strings under one hash. Verify-then-dedup: a duplicate that does not
    byte-compare equal is refused out loud, never silently swapped."""


class MissingBlobRefusal(SpineRefusal):
    """An event citing a blob the store does not hold. Durability ordering is law - the blob is
    fsynced before the event referencing it may append."""


class ChainBroken(SpineRefusal):
    """The chain does not verify at some LSN: a broken prev-hash link, a gap in the sequence, a
    recomputed event hash that disagrees, or a sealed body whose AAD no longer matches its
    envelope."""


class SealedBodyUnreadable(SpineRefusal):
    """A body that cannot be opened - the class key is absent (crypto-shredding is exactly this
    state) or the ciphertext does not authenticate."""


class CheckpointInvalid(SpineRefusal):
    """A checkpoint signature that does not verify under the verifying key published at genesis,
    or that covers an (lsn, event_hash) pair the log does not have."""


class WriterBusy(SpineRefusal):
    """A second writer reaching for a home one writer already holds. One deployment, one log, one
    writer: the second would assign an LSN that is already taken, and nothing here may edit or
    remove the line that results."""


class HomeExists(SpineRefusal):
    """Genesis run against a directory that already holds a log. A home is initialised once; a
    second genesis would fork the record."""


class HomeMissing(SpineRefusal):
    """A home that is not there, or is there without the log/blobs/keys a home is made of."""
