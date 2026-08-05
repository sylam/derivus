"""Per-class `fields` declarations must reproduce `fields.mapping['Instrument']` exactly.

This is the gate the migration runs behind. Each class that has been migrated declares its own
ordered field list; `schema.emit_instrument` rebuilds the legacy three-level shape from those
declarations, and it has to come back byte-identical - because the Workbench, the docs generator
and the Excel add-in all read that shape and none of them is being changed.

Any difference that shows up here is NOT migration breakage. It is pre-existing drift between the
class and the hand-written dict, finally visible, and it belongs on a findings list rather than
being papered over by editing the declaration to match.

Two things beyond equality matter. Declaration ORDER is load-bearing: the Workbench lays panels out
in `types[T]` order and widgets within a panel in section order, so a set-backed view scrambles the
UI silently, with no exception. And the schema's inheritance is composition of named GROUPS, not the
class hierarchy - `FXAdmin` is shared by eight deals with no common base - which is why groups are
module-level constants a class lists rather than anything recovered from the MRO.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inspect

import pytest

from derivus import fields, instruments, schema

INSTRUMENT = fields.mapping['Instrument']


def declared_classes():
    """Deal classes migrated to a per-class `fields` declaration. Own-attr only: a subclass that
    has not been migrated must not silently inherit its parent's declaration and appear done."""
    return {n: c.__dict__['fields'] for n, c in vars(instruments).items()
            if inspect.isclass(c) and issubclass(c, instruments.Deal) and 'fields' in c.__dict__}


EMITTED = schema.emit_instrument(declared_classes())


def test_some_classes_are_migrated():
    """Guards the gate itself: every assertion below is vacuously true over an empty declaration
    set, so an import error or a filter bug would read as a green migration."""
    assert declared_classes(), 'no class declares `fields` - the round-trip gate is vacuous'


@pytest.mark.parametrize('deal_type', sorted(declared_classes()))
def test_emitted_type_sections_match(deal_type):
    """`types[T]` is a list, and the Workbench builds one panel per entry IN ORDER."""
    assert EMITTED[0][deal_type] == INSTRUMENT['types'][deal_type]


@pytest.mark.parametrize('section', sorted(EMITTED[1]))
def test_emitted_section_fields_match(section):
    """Field order within a section is the widget order on screen."""
    assert EMITTED[1][section] == INSTRUMENT['sections'][section]


@pytest.mark.parametrize('field', sorted(EMITTED[2]))
def test_emitted_descriptor_matches(field):
    """The descriptor the UI renders and the docs generator publishes."""
    assert EMITTED[2][field] == INSTRUMENT['fields'][field]


def test_shared_groups_are_shared_not_copied():
    """`FXAdmin` is one object listed by eight classes. Copying it per class would let the copies
    drift - which is the whole defect the name-keyed dict has, reintroduced one level down."""
    users = [f for f in declared_classes().values()
             if any(g.name == 'FXAdmin' for g in f)]
    assert len(users) > 1, 'FXAdmin is declared by fewer than two classes - nothing to share'
    groups = {id(g) for f in users for g in f if g.name == 'FXAdmin'}
    assert len(groups) == 1, f'FXAdmin exists as {len(groups)} distinct objects, not one'
