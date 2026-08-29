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

"""The four keys a home is made of - one that blinds, one that seals, and a pair that signs.

The log is CLASSIFIED: the envelope is firm-visible and the body is sealed, so the three jobs a
naive design gives to one plaintext hash are split across three keys held here.

`blind.key` answers idempotency. A raw plaintext hash sitting in a firm-visible envelope is a
dictionary oracle - an approval over a plan hash visible elsewhere, a status transition, any
low-entropy body could be CONFIRMED by hashing candidates - so the envelope carries an HMAC of the
canonical plaintext instead, and the writer, which holds every key, enforces uniqueness on that.
The tag is therefore key-dependent by construction: the same fact appended into two homes wears two
tags, which is what makes the no-keyless-check assertion in the gate true rather than hopeful. The
blind key never leaves the hub and never couples to revocation rotation.

`class_firm.key` seals bodies under AES-256-GCM from genesis, even with one class and one desk,
for two reasons that outrank stdlib purity: genesis-era bodies must be crypto-shreddable (a
determination's reasoning, a relayed client utterance - erasure of the body inside an untouched
chain is this system's answer to erasure regimes), and phase 1 is already multi-seat, so
sealed-at-rest is what makes the share-and-backup posture true on day one. Destroying this ONE
file is the crypto-shred: the chain still verifies over ciphertext nobody can read, which is why
its absence is `SealedBodyUnreadable` - a state the design has a name for - while a missing blind
or signing key is `HomeMissing`, a home that is not equipped to write.

`checkpoint_signing.key` / `checkpoint_verify.key` are the deployment's Ed25519 pair. Signatures
rather than an HMAC economization, because the sealing dependency already ships them and a
signature is the only thing a REPLICA can check: the verifying key is published as a firm-class
policy blob at genesis, so every replica asserts checkpoint AUTHENTICITY from checkpoint one
instead of merely re-walking a chain it was handed.

Keys are read from disk at every use rather than cached at construction. A key destroyed at 14:02
is destroyed at 14:02, and a `Keys` held open across that moment must not be the thing that
remembers it.
"""
import hashlib
import hmac
import os
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat)

from .errors import CheckpointInvalid, HomeExists, HomeMissing, SealedBodyUnreadable

#: AES-256 and HMAC-SHA256 both take 32 bytes, and an Ed25519 key is 32 bytes raw - one width for
#: every file in `keys/`, so a truncated key is a length check rather than a subtle failure later.
KEY_BYTES = 32
#: GCM's nonce, prepended to the ciphertext: 12 bytes is the size the mode is defined over, and a
#: fresh random one per body is what keeps two identical payloads from looking identical at rest.
NONCE_BYTES = 12
#: GCM's tag, the minimum a sealed body can carry beyond its nonce.
TAG_BYTES = 16

BLIND = 'blind.key'
FIRM = 'class_firm.key'
SIGNING = 'checkpoint_signing.key'
VERIFYING = 'checkpoint_verify.key'


class Keys:
    """The `keys/` directory of one spine home, read on demand.

    Construction never touches the disk and never refuses: a replica that holds no key at all still
    builds one of these, opens its log, and verifies its chain over ciphertext. The refusals happen
    where a key is actually USED, which is the only place the difference between "unentitled" and
    "broken" is knowable.
    """

    def __init__(self, home):
        self.home = Path(home)
        self.keys = self.home / 'keys'

    def __repr__(self):
        return 'Keys({!r})'.format(str(self.home))

    @classmethod
    def generate(cls, home):
        """Mint a home's four keys and answer the `Keys` over them.

        Called once, by genesis. It refuses to overwrite: a second mint would strand every body
        already sealed under the key it replaced, which is crypto-shredding by accident.
        """
        keys = cls(home)
        keys.keys.mkdir(parents=True, exist_ok=True)
        signing = Ed25519PrivateKey.generate()
        minted = (
            (BLIND, os.urandom(KEY_BYTES)),
            (FIRM, os.urandom(KEY_BYTES)),
            (SIGNING, signing.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())),
            (VERIFYING, signing.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)),
        )
        for name, material in minted:
            path = keys.keys / name
            if path.exists():
                raise HomeExists(
                    'keys/{} already exists at {}: minting over it would strand every body sealed '
                    'under the key it replaces - move the directory aside and initialise a fresh '
                    'home, or keep the one that is there'.format(name, keys.keys))
            keys._write(path, material)
        return keys

    def has_blind(self):
        """Whether the tag can be recomputed here. A verifier that cannot recompute tags says so
        rather than reporting a check it did not run."""
        return (self.keys / BLIND).is_file()

    def has_firm(self):
        """Whether bodies can be opened here. False is the crypto-shredded state, not an error."""
        return (self.keys / FIRM).is_file()

    def verifying_key(self):
        """The raw 32-byte Ed25519 public key, for genesis to publish as a blob. Verification
        elsewhere reads the BLOB, never this file - a replica has the blob and does not have
        `keys/`."""
        return self._require(VERIFYING, HomeMissing)

    def blind_tag(self, canonical):
        """HMAC-SHA256 of the canonical semantic bytes under `blind.key`, hex.

        This is the idempotency key the writer enforces uniqueness on, and the reason an unentitled
        replica holding every envelope has no computable check against a candidate plaintext.
        """
        return hmac.new(self._require(BLIND, HomeMissing), canonical, hashlib.sha256).hexdigest()

    def seal(self, plaintext, aad):
        """AES-256-GCM under the firm class key: `nonce || ciphertext+tag`.

        `aad` is the envelope the body is being bound to. It is authenticated but not encrypted, so
        an envelope field edited after the fact - an actor, a book - makes the body stop opening:
        the seal is what makes the plaintext envelope tamper-evident, not merely the chain.
        """
        nonce = os.urandom(NONCE_BYTES)
        return nonce + AESGCM(self._require(FIRM, SealedBodyUnreadable)).encrypt(
            nonce, plaintext, aad)

    def open(self, sealed, aad):
        """The plaintext of `sealed` under `aad`, or `SealedBodyUnreadable`.

        Every failure lands on that one refusal - an absent class key, a truncated body, a flipped
        ciphertext byte, an envelope that no longer matches the AAD it was sealed against. The
        caller that can tell the cases apart is the one that knows which key it holds; verification
        checks the key first and reads a failure here as tampering.
        """
        key = self._require(FIRM, SealedBodyUnreadable)
        if not isinstance(sealed, (bytes, bytearray)) or len(sealed) < NONCE_BYTES + TAG_BYTES:
            raise SealedBodyUnreadable(
                'the sealed body is {} bytes, short of the {} a nonce and a GCM tag take: the '
                'frame is truncated - restore it from a verified replica'.format(
                    len(sealed) if sealed is not None else 0, NONCE_BYTES + TAG_BYTES))
        sealed = bytes(sealed)
        try:
            return AESGCM(key).decrypt(sealed[:NONCE_BYTES], sealed[NONCE_BYTES:], aad)
        except Exception:
            # GCM does not say WHICH of the two failed, and neither will this: either the
            # ciphertext moved or the envelope it was bound to did.
            raise SealedBodyUnreadable(
                'the sealed body does not authenticate under its envelope: the ciphertext or one '
                'of the nine envelope fields it is bound to has been altered - restore the frame '
                'from a verified replica')

    def sign_checkpoint(self, payload):
        """Ed25519 over the canonical (event_hash, lsn) payload, hex."""
        return Ed25519PrivateKey.from_private_bytes(
            self._require(SIGNING, HomeMissing)).sign(payload).hex()

    @staticmethod
    def verify_checkpoint(payload, sig_hex, verify_key_bytes, where='a checkpoint'):
        """Assert `sig_hex` is `verify_key_bytes`' signature over `payload`; `CheckpointInvalid`
        otherwise, naming `where`.

        Static because this is the REPLICA's check: it needs the published verifying key and
        nothing out of `keys/`, so the same code runs on a machine that could never have produced
        the signature it is checking.
        """
        try:
            signature = bytes.fromhex(sig_hex)
        except (TypeError, ValueError):
            raise CheckpointInvalid(
                '{}: the signature is not hex ({!r}): the body was written by something that is '
                'not this writer - verify the home against a replica'.format(where, sig_hex))
        try:
            Ed25519PublicKey.from_public_bytes(bytes(verify_key_bytes)).verify(signature, payload)
        except Exception:
            raise CheckpointInvalid(
                '{}: the signature does not verify under the verifying key published at genesis - '
                'the checkpoint, the head it covers, or the published key has been altered; '
                'compare the home against a replica before trusting any position in it'.format(
                    where))

    def _require(self, name, refusal):
        """The key material behind `name`, or `refusal` naming the file and the remedy."""
        path = self.keys / name
        try:
            material = path.read_bytes()
        except (IOError, OSError):
            raise refusal(
                'keys/{0} is not in this home ({1}): {2}'.format(
                    name, self.keys,
                    'the class key was destroyed, so these bodies are unreadable here forever - '
                    'that is what crypto-shredding is; verify the chain with entitled=False, or '
                    'read the bodies on a replica that still holds the key'
                    if name == FIRM else
                    'a home writes with its own keys - restore keys/ from the custodian\'s copy, '
                    'or treat this home as the read-only replica it now is'))
        if len(material) != KEY_BYTES:
            raise refusal(
                'keys/{} is {} bytes, not {}: the file is truncated or is not a key - restore it '
                'from the custodian\'s copy'.format(name, len(material), KEY_BYTES))
        return material

    @staticmethod
    def _write(path, material):
        """One key file, fsynced, and 0600 where the platform has an opinion."""
        with open(str(path), 'wb') as handle:
            handle.write(material)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(str(path), 0o600)
        except (IOError, OSError):
            # Best effort: Windows ACLs are not this mode, and a key the owner can read is the
            # posture either way. Custody is increment 2's business, not a chmod's.
            pass
