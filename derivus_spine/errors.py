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

"""Every way the spine says no, typed.

These are the refusal vocabulary of the whole package: every module raises from this list rather
than inventing its own, and `SpineRefusal` is the one type to catch at a boundary (the CLI turns
it into exit 1 and prints the refusal's own wording, so every message must name the thing that was
wrong and the remedy that fixes it). The class carries the kind, so callers branch on it; the
message carries the case - this LSN, this hash, this field.
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
    writer: the second would assign an LSN that is already taken."""


class HomeExists(SpineRefusal):
    """Genesis run against a directory that already holds a log. A home is initialised once; a
    second genesis would fork the record."""


class HomeMissing(SpineRefusal):
    """A home that is not there, or is there without the log/blobs/keys a home is made of."""


class IdentityRefused(SpineRefusal):
    """A token that does not prove who it claims - a signature that does not verify under the
    deployment's published JWKS, a wrong audience or issuer, an expired claim, or an algorithm
    outside the allowlist (the `none` and HMAC confusions land here by design)."""


class CapabilityDenied(SpineRefusal):
    """An actor without the (verb, book) scope the event demands, under the capability policy in
    force at this LSN. The denial is itself appended to the log."""


class ReplayRefused(SpineRefusal):
    """A replay claim the record will not attest: the re-execution ran at another engine version,
    or the bytes it produced do not reproduce the ones claimed within the declared tolerance - or
    no tolerance policy is declared at all, so there is no standard to hold the claim to."""


class QuoteNotFirm(SpineRefusal):
    """A quote whose pins no longer stand: the market moved or aged under it (the VALUES
    dimension), or the book moved or aged under it (the PLAN dimension). The message names which
    dimension - the two have different remedies."""


class CustodyRefusal(SpineRefusal):
    """Key custody declining - a wrap that does not open under the seat's own private key, an
    enrollment the log does not carry, or an escrow recovery asked of a home that declared no
    escrow key."""
