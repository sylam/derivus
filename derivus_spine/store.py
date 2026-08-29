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

"""The content-addressed blob store - write-once bytes under their own SHA-256, and no way back.

A blob is named by what it IS, so the same bytes put twice are one file and one answer, and every
reference anywhere in the spine is a hash rather than a path: surfaces, curves, plans, values
vectors and raw tape live here once and are spoken of by hash forever, while the log inlines only
a value that IS the fact and tiny. Two disciplines make that name trustworthy.

Writes land ATOMICALLY. Bytes go to `blobs/tmp`, are flushed and fsynced, and only then are
`os.replace`d onto their address, so a reader sees the whole blob or nothing and a crash leaves
scratch rather than a half blob under a name that promises the rest. Durability ordering is law -
no event may cite a blob that is not yet on the platter - and that promise is only worth what the
fsync makes it.

Dedup VERIFIES. A path that already exists is byte-compared against the incoming bytes: equal is
the dedup, different is a named refusal. The store never swaps content under an address, so even a
cryptanalytic surprise degrades into a loud stop rather than a silent substitution - and the same
suspicion runs on the way out, since bytes that no longer hash to the name they were filed under
are not the blob that was asked for.

There is NO deletion verb here, public or private. Retention arrives in a later increment as a
logged event - a blob class reduces because the record says so - and until then the absence of the
method IS the guarantee: nothing in this package can be asked to forget.
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

    Reads never provision - only `put` makes directories - so a store pointed at a home that does
    not exist answers "empty" instead of conjuring one, and the home's own refusals stay the ones
    that fire.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / 'blobs'
        self.tmp = self.blobs / 'tmp'

    def __repr__(self):
        return 'BlobStore({!r})'.format(str(self.root))

    def put(self, data: bytes) -> str:
        """File `data` under its SHA-256 and answer the hex id.

        The existing-path branch is the collision discipline: byte-compare first, dedup only on
        equality, refuse by name otherwise.
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
            # Scratch is not a blob - it never had an address, and nothing can cite it - so
            # clearing a failed write is hygiene rather than retention.
            if scratch.exists():
                os.unlink(str(scratch))
            raise
        self._fsync_dir(path.parent)
        return digest

    def get(self, digest: str) -> bytes:
        """The bytes filed under `digest`, re-hashed on the way out.

        The record never trusts what it can re-derive, and a blob is the cheapest case of it: the
        store hands back what was asked for or it refuses.
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
        """Whether `digest` resolves here. The referential-closure check the writer runs before it
        may append an event that cites a blob."""
        return self._is_id(digest) and self._path(digest).is_file()

    def walk(self) -> Iterator[str]:
        """Every blob id on disk, shard order. The manifest is a projection and this is the walk
        that rebuilds it.

        Only well-formed ids sitting at their OWN address are content: `blobs/tmp` is scratch, and
        anything else under the tree is unaddressable by construction, so neither is yielded.
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
        """A store address is 64 lowercase hex. Anything else cannot be present, so `has` says no
        and `get` refuses by the name it was given."""
        return (isinstance(digest, str) and len(digest) == HEX_LENGTH
                and HEX_DIGITS.issuperset(digest))

    @staticmethod
    def _fsync_dir(path):
        """POSIX only: a rename is durable once its directory is synced. Windows has no directory
        handle to open for that and `os.replace` is atomic there regardless, so it is skipped
        rather than faked."""
        if os.name != 'posix':
            return
        handle = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
