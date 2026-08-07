"""`fields.mapping['Instrument']` and `['Factor']` are generated from the per-class `fields`
declarations, and a SECTION (deals) or a TYPE (factors) owns its descriptors. So there is no
round-trip to gate - the old test diffed the emitted dict against a hand-written one, and both
sides are now the same object.

Three defect classes are unreachable by construction rather than merely absent, which is why the
gates that guarded them are gone:

  - a type naming no class (`SwapBasisDeal`, `SwapCurrencyDeal`: the UI offered them and
    `construct_instrument` logged and returned `{}`) - a type IS a class that declares fields
  - a section naming a field with no descriptor, and a descriptor no section reaches - a section
    IS its descriptors, and a descriptor exists only inside one
  - one field name silently resolving to another deal's descriptor - each section holds its own

What remains gateable is what the declarations can still get wrong: a malformed descriptor, a
section declared two ways by two classes, a shared group copied instead of shared, and a deal type
that no create-menu offers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from derivus import fields, instruments, riskfactors, schema

INSTRUMENT = fields.mapping['Instrument']
FACTOR = fields.mapping['Factor']

# riskfactors classes that legitimately declare no schema row of their own.
UNDECLARED_FACTORS = {
    'Factor0D': 'dimension base', 'Factor1D': 'dimension base',
    'Factor2D': 'dimension base', 'Factor3D': 'dimension base',
    'InterestRateJacobian': 'its block is keyed by BENCHMARK name, so it has no fixed field set',
}

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


def factor_classes():
    """The same for price factors, whose declaration is one flat list rather than groups."""
    return {n: c.__dict__['fields'] for n, c in vars(riskfactors).items()
            if isinstance(c, type) and isinstance(c.__dict__.get('fields'), list)}


def riskfactor_classes():
    """Every class DEFINED in riskfactors - the set a declared type has to come from."""
    return {n: c for n, c in vars(riskfactors).items()
            if isinstance(c, type) and c.__module__ == riskfactors.__name__}


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

    Parametrized over CLASSES and reading the DECLARATIONS, not the emitted store. Sections are
    shared objects, so walking the store visits `Admin` once however many types list it; walking
    the classes checks every declaration in the place an author would edit it.
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


def test_no_class_is_hidden_from_the_create_menu():
    """`groups` is the Workbench's create-deal menu and stays hand-curated, being presentation. So
    it is the one part of the store that can still drift from the classes: a deal type absent from
    every group is fully declared, fully priceable and unreachable from the UI."""
    menued = {t for members in INSTRUMENT['groups'].values() for t in members}
    assert not sorted(set(INSTRUMENT['types']) - menued), (
        f'declared deal types in no menu group: {sorted(set(INSTRUMENT["types"]) - menued)}')


def test_one_name_may_carry_two_descriptors_in_different_sections():
    """The capability the per-section store exists for, pinned so a return to a flat one fails.

    `Payment_Timing` is `Touch`/`Expiry` on a one-touch and `End`/`Begin`/`Discounted` on a
    cashflow leg. Both are right - the JSON is per-deal, so the data is never ambiguous and only a
    store keyed by field name across every deal was. Under that store one of these silently won,
    and the loser had to invent a descriptor key (`Option_Payment_Timing`) and carry its real name
    as an alias.
    """
    sections = INSTRUMENT['sections']
    seen = {tuple(d['values']) for s in sections.values()
            for k, d in s.items() if k == 'Payment_Timing'}
    assert seen == {('Touch', 'Expiry'), ('End', 'Begin', 'Discounted')}, seen

    # and each type sees only its own
    def values_for(deal_type):
        return [d['values'] for g in INSTRUMENT['types'][deal_type]
                for k, d in sections[g].items() if k == 'Payment_Timing']
    assert values_for('FXOneTouchOption') == [['Touch', 'Expiry']]
    assert values_for('CapDeal') == [['End', 'Begin', 'Discounted']]


@pytest.mark.parametrize('cls_name', sorted(declared_classes()))
def test_no_group_declares_a_key_twice(cls_name):
    """A section is now a dict keyed by the JSON name, so a name declared twice in one group loses
    a descriptor outright rather than merely shadowing one. `Net_Cashflows` was declared twice
    verbatim on `StructuredDeal` and the duplicate was invisible.

    The cross-class version of this check is gone on purpose - two SECTIONS may key the same name
    differently, which is the point of the per-section store."""
    for group in declared_classes()[cls_name]:
        keys = [f.key for f in group.fields]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        assert not dupes, f'{cls_name}.{group.name} declares {dupes} more than once'


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


def test_the_factor_store_is_generated():
    """The same guard as above, for factors: every Factor assertion here is vacuously true over an
    empty declaration set."""
    assert factor_classes(), 'no riskfactors class declares `fields` - these gates are vacuous'
    assert FACTOR['types'] == schema.emit_factor(riskfactors), (
        'the Factor store is not the emitted view - a hand-written copy has come back')
    assert 'fields' not in FACTOR, 'a flat name-keyed store has come back beside the types'


def test_every_declared_factor_type_is_constructible():
    """`construct_factor` does `globals().get(factor.type)(block)`, so a declared type naming no
    class is not a logged miss like a deal - it is `None(block)`, a TypeError at compile time.
    `ConvenienceYield` sat in exactly that state, declared with two fields and a process-map row
    and with no class in this repo. A type IS a class that declares fields, so it is unreachable."""
    undispatchable = sorted(set(FACTOR['types']) - set(riskfactor_classes()))
    assert not undispatchable, f'schema offers factor types with no class: {undispatchable}'


def test_every_riskfactor_class_is_declarable():
    """The converse: a factor class no schema declares cannot be authored or documented. The
    exemptions are the four dimension bases, which are never a `Factor.type`, and the Jacobian,
    whose block is keyed by benchmark instrument rather than by a fixed field set."""
    missing = sorted(set(riskfactor_classes()) - set(FACTOR['types']) - set(UNDECLARED_FACTORS))
    assert not missing, f'riskfactors classes no schema can author: {missing}'


@pytest.mark.parametrize('cls_name', sorted(factor_classes()))
def test_factor_descriptor_shape(cls_name):
    """`check_descriptor` again, over the factor declarations - same tagged union, same
    consumers."""
    for f in factor_classes()[cls_name]:
        check_descriptor(f'{cls_name}.{f.key}', f.descriptor())


@pytest.mark.parametrize('cls_name', sorted(factor_classes()))
def test_no_factor_declares_a_key_twice(cls_name):
    """A factor type is one dict keyed by the JSON name, so a name declared twice loses a
    descriptor outright."""
    keys = [f.key for f in factor_classes()[cls_name]]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f'{cls_name} declares {dupes} more than once'


def test_a_2d_and_a_3d_surface_may_both_be_called_surface():
    """The capability the per-type store exists for, pinned so a return to a flat one fails.

    `Surface` is a (moneyness, expiry, vol) triple list on a `VolatilityGrid` and a quad list on
    the three vol SPACES, and both are right - the JSON is per factor type. Under a flat store one
    of them had to be filed as `Space` and carry `Surface` as an alias."""
    assert FACTOR['types']['VolatilityGrid']['Surface']['value'] == schema.BLANK['Surface']
    for space in ('InterestYieldVol', 'InterestRateVol', 'ForwardPriceVol'):
        assert FACTOR['types'][space]['Surface']['value'] == schema.BLANK['Space']
