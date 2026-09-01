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

"""The content-addressed blob store - write-once bytes under their own SHA-256.

Every reference in the spine is a hash rather than a path: surfaces, curves, plans, values vectors
and raw tape live here once and are spoken of by hash, while the log inlines only a value that is
itself the fact and is tiny. Two disciplines make the address trustworthy.

Writes land atomically. Bytes go to `blobs/tmp`, are flushed and fsynced, then `os.replace`d onto
their address, so a reader sees the whole blob or nothing. Durability ordering is law: no event may
cite a blob that is not yet on the platter.

Dedup verifies. An address that already exists is byte-compared against the incoming bytes - equal
dedups, different raises `CollisionRefusal` - and the same check runs on the way out, so the store
never swaps content under an address.

There is no deletion verb here, public or private; retention arrives as a logged event.
"""
import hashlib
import os
from pathlib import Path
from typing import Iterator

from .errors import CollisionRefusal, MissingBlobRefusal

#: SHA-256 everywhere, as the brief says once; changing it is a logged `rehash_declared` policy
#: event with history dual-addressed under the successor, never an edit here.
HEX_LENGTH = 64
HEX_DIGITS = frozenset('0123456789abcdef')


class BlobStore:
    """The blob tree under `root` (the spine home): `blobs/<h[:2]>/<h[2:4]>/<h>`, two levels of
    256 so no directory holds the whole store, plus `blobs/tmp` for scratch.

    Only `put` creates directories, so a store pointed at a home that does not exist reads as empty
    rather than conjuring one.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / 'blobs'
        self.tmp = self.blobs / 'tmp'

    def __repr__(self):
        return 'BlobStore({!r})'.format(str(self.root))

    def put(self, data: bytes) -> str:
        """File `data` under its SHA-256 and return the hex id.

        An address that already exists is byte-compared: equal dedups, different raises
        `CollisionRefusal`. Non-bytes input is a `TypeError`.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(
                'BlobStore.put files bytes, not {}: canonicalise the object (canon.canonical_bytes)'
                ' or encode the text before storing it'.format(type(data).__name__))
        data = bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(digest)
        if path.exists():
            stored = path.read_bytes()
            if stored != data:
                raise CollisionRefusal(
                    'blob {} already holds different bytes ({} stored, {} offered): the store does '
                    'not swap content under an address - keep both payloads out of band and raise '
                    'the collision, or restore the file from a verified replica'.format(
                        digest, len(stored), len(data)))
            return digest
        self.tmp.mkdir(parents=True, exist_ok=True)
        scratch = self.tmp / os.urandom(16).hex()
        try:
            with open(str(scratch), 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(scratch), str(path))
        except BaseException:
            # Scratch never had an address and nothing can cite it, so clearing it is not deletion.
            if scratch.exists():
                os.unlink(str(scratch))
            raise
        self._fsync_dir(path.parent)
        return digest

    def get(self, digest: str) -> bytes:
        """The bytes filed under `digest`, re-hashed on the way out.

        Raises `MissingBlobRefusal` if the address does not resolve, `CollisionRefusal` if the file
        no longer hashes to the name it was filed under.
        """
        if not self.has(digest):
            raise MissingBlobRefusal(
                'blob {} is not in the store at {}: pull it from any replica that holds it - a '
                'blob is self-verifying by hash, so any source will do - before citing it'.format(
                    digest, self.blobs))
        data = self._path(digest).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise CollisionRefusal(
                'blob {} hashes to {} on disk: the file has been altered under its own name - '
                'restore it from a verified replica'.format(digest, actual))
        return data

    def has(self, digest: str) -> bool:
        """Whether `digest` resolves here - the referential-closure check the writer runs before
        appending an event that cites a blob."""
        return self._is_id(digest) and self._path(digest).is_file()

    def walk(self) -> Iterator[str]:
        """Every blob id on disk, in shard order - the walk that rebuilds the manifest projection.

        Only well-formed ids sitting at their own address are yielded; `blobs/tmp` and anything
        else under the tree is skipped.
        """
        if not self.blobs.is_dir():
            return
        for shard in sorted(self.blobs.iterdir()):
            if shard.name == 'tmp' or not shard.is_dir():
                continue
            for sub in sorted(shard.iterdir()):
                if not sub.is_dir():
                    continue
                for blob in sorted(sub.iterdir()):
                    digest = blob.name
                    if (self._is_id(digest) and digest[:2] == shard.name
                            and digest[2:4] == sub.name and blob.is_file()):
                        yield digest

    def _path(self, digest):
        return self.blobs / digest[:2] / digest[2:4] / digest

    @staticmethod
    def _is_id(digest):
        """Whether `digest` is a well-formed store address: 64 lowercase hex characters."""
        return (isinstance(digest, str) and len(digest) == HEX_LENGTH
                and HEX_DIGITS.issuperset(digest))

    @staticmethod
    def _fsync_dir(path):
        """Sync `path` so the rename into it is durable. POSIX only - Windows offers no directory
        handle to sync and `os.replace` is atomic there regardless, so it is skipped."""
        if os.name != 'posix':
            return
        handle = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
