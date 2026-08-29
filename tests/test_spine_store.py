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

"""What the blob store promises, asserted against real files in a real directory.

Every address here is recomputed by the gate itself - `hashlib.sha256(...).hexdigest()` and the
`blobs/<h[:2]>/<h[2:4]>/<h>` layout spelled out by hand - so an implementation that agreed with
itself but not with the wire format turns this file red. Nothing is patched: the collision fault
is injected the only way the house allows, by DOCTORING BYTES ON DISK at an address the gate
computes, and then asking the store to do the thing that must refuse.

The last gate is the one that asserts an absence. Retention is a logged event in a later
increment, so today the store must carry no verb for forgetting at all - not public, not private,
not module level - because until the record can say a blob went away, nothing may make one go
away.
"""
import ast
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine.errors import CollisionRefusal, MissingBlobRefusal
from derivus_spine.store import BlobStore

#: Verbs that would make retention representable without a logged event. Dunders are object's own
#: protocol (`__delattr__` is not a store verb) and are exempted by name, not by accident.
FORGETTING = ('delete', 'remove', 'expire', 'unlink', 'purge', 'evict', 'prune', 'discard', 'drop')

PLAN = b'{"instrument":"a canonical plan","version":1}'
TAPE = b'\x00\x01\x02 raw tape is evidence, not derivation \xff\xfe'


def address(root, data):
    """The path the wire format says these bytes live at, computed from the RFC-shaped rule rather
    than asked of the store."""
    digest = hashlib.sha256(data).hexdigest()
    return digest, root / 'blobs' / digest[:2] / digest[2:4] / digest


def test_a_blob_round_trips_under_the_hash_this_gate_computes_itself(tmp_path):
    """put answers the SHA-256 of the bytes, get answers the bytes, has answers yes - and the file
    is where the layout says, not merely somewhere the store can find it again."""
    store = BlobStore(tmp_path)
    digest, path = address(tmp_path, PLAN)
    assert store.put(PLAN) == digest
    assert path.is_file(), 'blob did not land at blobs/{}/{}/{}'.format(
        digest[:2], digest[2:4], digest)
    assert path.read_bytes() == PLAN
    assert store.get(digest) == PLAN
    assert store.has(digest)
    # Bulk and empty are the same discipline: the empty blob is a legitimate object with a real
    # address, and nothing here reads zero bytes as absence.
    assert store.get(store.put(TAPE)) == TAPE
    empty = store.put(b'')
    assert empty == hashlib.sha256(b'').hexdigest()
    assert store.get(empty) == b'' and store.has(empty)
    assert set(store.walk()) == {digest, hashlib.sha256(TAPE).hexdigest(), empty}


def test_the_same_bytes_put_twice_are_one_file_and_one_answer(tmp_path):
    """Dedup is the whole point of addressing by content: booking the same surface twice must not
    cost a second copy, and the second put must answer the same name the first did."""
    store = BlobStore(tmp_path)
    first = store.put(PLAN)
    second = store.put(bytearray(PLAN))
    assert first == second
    blobs = [p for p in (tmp_path / 'blobs').rglob('*') if p.is_file()]
    assert len(blobs) == 1, blobs
    assert list(store.walk()) == [first]


def test_nothing_is_left_in_scratch_after_a_write(tmp_path):
    """Atomicity leaves a trace when it is done right: bytes go to `blobs/tmp` and are renamed onto
    their address, so a store at rest has an EMPTY scratch directory. A leftover here is a
    half-written blob waiting to be mistaken for one."""
    store = BlobStore(tmp_path)
    for data in (PLAN, TAPE, PLAN, b'', b'a values vector'):
        store.put(data)
    assert list((tmp_path / 'blobs' / 'tmp').iterdir()) == []


def test_a_hash_that_arrives_with_different_bytes_is_refused_by_name(tmp_path):
    """The collision fault, injected as data: wrong bytes are written BY HAND at the address the
    true bytes hash to, and the true bytes are then offered. Verify-then-dedup means the store
    byte-compares and refuses - a silent dedup here would swap one object for another everywhere
    it is cited, forever."""
    store = BlobStore(tmp_path)
    digest, path = address(tmp_path, PLAN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'not the plan it claims to be')
    with pytest.raises(CollisionRefusal) as refusal:
        store.put(PLAN)
    assert digest in str(refusal.value), refusal.value
    # A refusal writes nothing: the doctored file is still what it was, and no scratch survives.
    assert path.read_bytes() == b'not the plan it claims to be'
    scratch = tmp_path / 'blobs' / 'tmp'
    assert not scratch.exists() or list(scratch.iterdir()) == []


def test_bytes_altered_under_their_own_name_never_come_back_out(tmp_path):
    """The same suspicion on the read side. A blob is re-hashed as it leaves, so a file edited in
    place under a name that promises other content is a refusal naming the hash, never a payload
    the caller goes on to trust."""
    store = BlobStore(tmp_path)
    digest = store.put(PLAN)
    _, path = address(tmp_path, PLAN)
    path.write_bytes(PLAN + b' amended in place')
    with pytest.raises(CollisionRefusal) as refusal:
        store.get(digest)
    assert digest in str(refusal.value), refusal.value


def test_an_absent_blob_is_a_named_refusal_not_an_empty_answer(tmp_path):
    """Referential closure begins here: a hash that does not resolve is a stop with a name in it,
    so the writer can refuse an event citing it instead of appending a dangling reference."""
    store = BlobStore(tmp_path)
    absent = hashlib.sha256(b'never stored').hexdigest()
    assert not store.has(absent)
    with pytest.raises(MissingBlobRefusal) as refusal:
        store.get(absent)
    assert absent in str(refusal.value), refusal.value
    # Not an address at all: `has` says no rather than guessing at a path, and `get` still refuses
    # by the name it was handed.
    for bad in ('', 'deadbeef', absent.upper(), absent + '00', None, 42):
        assert not store.has(bad)
    with pytest.raises(MissingBlobRefusal):
        store.get('deadbeef')
    with pytest.raises(TypeError):
        store.put('a canonical plan, but as text')


def test_a_read_never_provisions_a_home_that_is_not_there(tmp_path):
    """Reads answer questions; they do not create trees. A store pointed at a home that does not
    exist yet says "empty" and leaves the directory absent, so the home's own refusals are the ones
    a caller sees."""
    store = BlobStore(tmp_path / 'no_such_home')
    assert not store.has(hashlib.sha256(PLAN).hexdigest())
    assert list(store.walk()) == []
    assert not (tmp_path / 'no_such_home').exists()


def test_walk_yields_what_the_store_can_address_and_nothing_else(tmp_path):
    """The manifest is a projection rebuilt by this walk, so it must report CONTENT: scratch is not
    content, and a file sitting somewhere its own name does not resolve to is not addressable and
    therefore is not a blob."""
    store = BlobStore(tmp_path)
    stored = {store.put(PLAN), store.put(TAPE)}
    (tmp_path / 'blobs' / 'tmp' / 'half-written').write_bytes(b'crashed mid-put')
    misplaced = hashlib.sha256(b'filed under the wrong shard').hexdigest()
    (tmp_path / 'blobs' / 'ff' / 'ff').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'blobs' / 'ff' / 'ff' / misplaced).write_bytes(b'filed under the wrong shard')
    (tmp_path / 'blobs' / 'ff' / 'ff' / 'README').write_bytes(b'not an address')
    assert set(store.walk()) == stored


def test_the_store_carries_no_verb_for_forgetting(tmp_path):
    """Retention is a logged event in a later increment. Until it exists, the guarantee IS this
    absence: no method on the store - public, private, or module level - can be asked to reduce or
    expire a blob, so 'expire the tape without a retention event' is not a thing anyone can spell.
    """
    store = BlobStore(tmp_path)
    surface = [name for name in dir(store) if not (name.startswith('__') and name.endswith('__'))]
    assert {'put', 'get', 'has', 'walk'}.issubset(surface), surface
    for name in surface:
        assert not any(verb in name.lower() for verb in FORGETTING), \
            'BlobStore.{} makes retention representable without a logged event'.format(name)
    # The class surface is the contract, but a module-level helper would be the same hole, so the
    # source is read too - definitions only, since the atomic write's own scratch cleanup is
    # hygiene over a file that never had an address.
    from derivus_spine import store as module
    tree = ast.parse(open(module.__file__).read())
    defined = [node.name for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for name in defined:
        assert not any(verb in name.lower() for verb in FORGETTING), name
