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

The log is classified: the envelope is firm-visible and the body is sealed, so three jobs are split
across three keys rather than resting on one plaintext hash.

`blind.key` answers idempotency. A raw plaintext hash in a firm-visible envelope would be a
dictionary oracle over low-entropy bodies, so the envelope carries an HMAC of the canonical
plaintext instead and the writer enforces uniqueness on that. The tag is key-dependent: the same
fact appended into two homes wears two tags. This key never leaves the hub.

`class_firm.key` seals bodies under AES-256-GCM from genesis, so bodies are crypto-shreddable and
sealed at rest across seats. Destroying this one file is the crypto-shred - the chain still
verifies over ciphertext nobody can read, which is why its absence raises `SealedBodyUnreadable`
while a missing blind or signing key raises `HomeMissing`.

`checkpoint_signing.key` / `checkpoint_verify.key` are the deployment's Ed25519 pair. The verifying
key is published as a firm-class policy blob at genesis, so a replica can assert checkpoint
authenticity rather than merely re-walk a chain it was handed.

Keys are read from disk at every use rather than cached at construction, so a key destroyed at
14:02 is destroyed at 14:02 even for a `Keys` held open across that moment.
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
#: GCM's nonce, prepended to the ciphertext. Fresh and random per body, so two identical payloads
#: do not look identical at rest.
NONCE_BYTES = 12
#: GCM's tag, the minimum a sealed body can carry beyond its nonce.
TAG_BYTES = 16

BLIND = 'blind.key'
FIRM = 'class_firm.key'
SIGNING = 'checkpoint_signing.key'
VERIFYING = 'checkpoint_verify.key'


class Keys:
    """The `keys/` directory of one spine home, read on demand.

    Construction never touches the disk and never refuses, so a replica holding no key still builds
    one of these and verifies its chain over ciphertext. Refusals happen where a key is used.
    """

    def __init__(self, home):
        self.home = Path(home)
        self.keys = self.home / 'keys'

    def __repr__(self):
        return 'Keys({!r})'.format(str(self.home))

    @classmethod
    def generate(cls, home):
        """Mint a home's four keys and return the `Keys` over them.

        Called once, by genesis. Raises `HomeExists` rather than overwrite: a second mint would
        strand every body already sealed under the key it replaced.
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
        """Whether `blind.key` is present, so idempotency tags can be recomputed here."""
        return (self.keys / BLIND).is_file()

    def has_firm(self):
        """Whether bodies can be opened here. False is the crypto-shredded state, not an error."""
        return (self.keys / FIRM).is_file()

    def verifying_key(self):
        """The raw 32-byte Ed25519 public key, for genesis to publish as a blob. Verification
        elsewhere reads that blob, never this file - a replica has no `keys/`."""
        return self._require(VERIFYING, HomeMissing)

    def blind_tag(self, canonical):
        """HMAC-SHA256 of the canonical semantic bytes under `blind.key`, hex.

        The idempotency tag the writer enforces uniqueness on. Being keyed is what denies an
        unentitled holder of the envelopes any check against a candidate plaintext.
        """
        return hmac.new(self._require(BLIND, HomeMissing), canonical, hashlib.sha256).hexdigest()

    def seal(self, plaintext, aad):
        """Seal `plaintext` under the firm class key: `nonce || ciphertext+tag`, AES-256-GCM.

        `aad` is the envelope the body binds to - authenticated but not encrypted, so an envelope
        field edited afterwards makes the body stop opening.
        """
        nonce = os.urandom(NONCE_BYTES)
        return nonce + AESGCM(self._require(FIRM, SealedBodyUnreadable)).encrypt(
            nonce, plaintext, aad)

    def open(self, sealed, aad):
        """The plaintext of `sealed` under `aad`.

        Every failure raises `SealedBodyUnreadable` - an absent class key, a truncated body, a
        flipped ciphertext byte, an envelope that no longer matches the AAD it was sealed against.
        A caller that needs to tell those apart checks key presence first.
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
            # GCM does not say which of the two failed - the ciphertext moved or the envelope did.
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
        """Assert `sig_hex` is `verify_key_bytes`' signature over `payload`, raising
        `CheckpointInvalid` naming `where` otherwise.

        Static because it is the replica's check: it needs the published verifying key and nothing
        out of `keys/`.
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
        """The key material behind `name`, raising `refusal` if it is absent or the wrong width."""
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
        """Write `material` to `path`, fsynced, and 0600 where the platform has an opinion."""
        with open(str(path), 'wb') as handle:
            handle.write(material)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(str(path), 0o600)
        except (IOError, OSError):
            # Best effort: Windows ACLs are not this mode, and custody is enforced elsewhere.
            pass
