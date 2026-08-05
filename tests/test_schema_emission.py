"""`fields.mapping['Instrument']` is now GENERATED from the per-class `fields` declarations, so the
old round-trip gate - emit the dict, diff it against the hand-written one - compares the view against
itself and asserts nothing. These are the gates that survive the flip.

What the flip removed structurally: a schema type naming no class (`SwapBasisDeal`,
`SwapCurrencyDeal` - the UI offered them, `construct_instrument` logged and returned `{}`) and a
descriptor no section reaches (15 of them). Neither is expressible when the classes are the source,
which is why the two strict xfails guarding them in `test_schema_class_correspondence` are gone.

What the flip did NOT remove is the flat store itself. `emit_instrument` folds 45 classes into one
name-keyed dict with `setdefault`, so two classes declaring the same key with different content
still silently resolve to whichever class was defined first - the exact defect the per-class
declaration exists to end, still live for as long as anything consumes the flat view.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from derivus import fields, instruments, schema

INSTRUMENT = fields.mapping['Instrument']

# How many columns each table tag's deserializer consumes, read off `set_repr` in derivus_jupyter:
# `DateList`/`CreditSupportList` unpack a PAIR per row, `DateEqualList` keys on [0] and keeps [1:],
# `DateValueList` only re-types [0], and `ResetArray` zips against ten hardcoded field types.
TAG_ARITY = {'DateList': (2, 2), 'CreditSupportList': (2, 2), 'DateEqualList': (2, 99),
             'DateValueList': (1, 99), 'ResetArray': (1, 10)}


def declared_classes():
    """Deal classes carrying their own `fields`. Own-attr only, matching `emit_instrument`: a
    subclass inheriting its parent's declaration is an alias, not a second deal type."""
    return {n: c.__dict__['fields'] for n, c in vars(instruments).items()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)}


def every_field(group):
    """A group's fields including nested container children and table columns."""
    def walk(f):
        yield f
        for child in (f.sub_fields or []):
            yield from walk(child)
    for top in group.fields:
        yield from walk(top)


def test_the_store_is_generated():
    """Guards the gate itself: every assertion below is vacuously true over an empty declaration
    set, so an import error or a filter bug would read as a green schema."""
    assert declared_classes(), 'no class declares `fields` - these gates are vacuous'
    assert INSTRUMENT['types'] == schema.emit_instrument(instruments)[0], (
        'the Instrument store is not the emitted view - a hand-written copy has come back')


@pytest.mark.parametrize('cls_name', sorted(declared_classes()))
def test_descriptor_shape(cls_name):
    """The descriptor is a tagged union on `widget`, and the consumers destructure it positionally
    without checking. A `Table` missing `sub_types` raises in the UI's cell renderer; a `Dropdown`
    missing `values` renders an empty list the author cannot pick from.

    Parametrized over CLASSES, not over the merged store's keys. `emit_instrument` folds 45 classes
    into one name-keyed dict with `setdefault`, so a descriptor read back from there is whichever
    class was defined first - checking the merged view tests the winner and cannot see a defect in
    any of the other declarations of the same key.
    """
    for group in declared_classes()[cls_name]:
        for f in every_field(group):
            check_descriptor(f'{cls_name}.{group.name}.{f.key}', f.descriptor())


def check_descriptor(key, d):
    assert {'widget', 'description', 'value'} <= set(d), f'{key} is missing a required descriptor key'
    assert ('values' in d) == (d['widget'] == 'Dropdown'), f'{key}: values must appear iff Dropdown'
    assert ('sub_fields' in d) == (d['widget'] == 'Container'), (
        f'{key}: sub_fields must appear iff Container')
    assert ('col_names' in d) == ('sub_types' in d) == (d['widget'] == 'Table'), (
        f'{key}: col_names and sub_types must both appear, iff Table')
    if d['widget'] == 'Table':
        assert len(d['col_names']) == len(d['sub_types']), (
            f'{key}: col_names and sub_types are matched by POSITION and disagree in length')
        if isinstance(d['obj'], list):
            assert len(d['obj']) == len(d['col_names']), f'{key}: one obj token per column'
        else:
            lo, hi = TAG_ARITY[d['obj']]
            assert lo <= len(d['col_names']) <= hi, (
                f'{key}: tag {d["obj"]} consumes {lo}..{hi} columns, the row declares '
                f'{len(d["col_names"])}')


def test_sub_fields_resolve():
    """A container names its children by key, and the UI looks each one up. A name with no
    descriptor is a KeyError in `load_fields`, not a missing widget."""
    missing = sorted({s for d in INSTRUMENT['fields'].values() for s in d.get('sub_fields', [])
                      if s not in INSTRUMENT['fields']})
    assert not missing, f'containers name children that have no descriptor: {missing}'


def test_no_class_is_hidden_from_the_create_menu():
    """`groups` is the Workbench's create-deal menu and stays hand-curated, being presentation. So
    it is the one part of the store that can still drift from the classes: a deal type absent from
    every group is fully declared, fully priceable and unreachable from the UI."""
    menued = {t for _, members in INSTRUMENT['groups'].values() for t in members}
    assert not sorted(set(INSTRUMENT['types']) - menued), (
        f'declared deal types in no menu group: {sorted(set(INSTRUMENT["types"]) - menued)}')


def test_no_key_is_declared_two_ways():
    """`emit_instrument` folds every class into one name-keyed dict with `setdefault`, so a key two
    classes declare differently resolves to whichever class `vars(module)` yields first - i.e. to
    source order. The loser renders the winner's widget with no error anywhere."""
    seen = {}
    for cls_name, groups in declared_classes().items():
        for f in (f for g in groups for f in every_field(g)):
            seen.setdefault(f.key, {}).setdefault(repr(f.descriptor()), []).append(cls_name)
    clashing = {k: {c[0] for c in v.values()} for k, v in seen.items() if len(v) > 1}
    assert not clashing, f'one descriptor key declared with differing content: {clashing}'


def test_no_section_is_declared_two_ways():
    """The same hazard one level up: `sections` is keyed by group NAME, so two groups sharing a name
    and differing in fields silently collapse to one panel."""
    seen = {}
    for cls_name, groups in declared_classes().items():
        for g in groups:
            seen.setdefault(g.name, {}).setdefault(tuple(f.key for f in g.fields), []).append(cls_name)
    clashing = {k: {c[0] for c in v.values()} for k, v in seen.items() if len(v) > 1}
    assert not clashing, f'one section name declared with differing fields: {clashing}'


def test_shared_groups_are_shared_not_copied():
    """`FXAdmin` is one object listed by eight classes. Copying it per class would let the copies
    drift - which is the whole defect the name-keyed dict has, reintroduced one level down."""
    users = [f for f in declared_classes().values() if any(g.name == 'FXAdmin' for g in f)]
    assert len(users) > 1, 'FXAdmin is declared by fewer than two classes - nothing to share'
    groups = {id(g) for f in users for g in f if g.name == 'FXAdmin'}
    assert len(groups) == 1, f'FXAdmin exists as {len(groups)} distinct objects, not one'
