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

"""The canonicaliser is the one thing in the spine that must agree with strangers, and this is the
file that says so.

Every hash the record keeps is taken over `canonical_bytes`, so a second implementation - another
language, another decade, an auditor's own script - has to reproduce these bytes or the record is
unverifiable. That makes RFC 8785's own vectors the acceptance bar rather than anything invented
here: the RFC's sample document with its published canonical output, the RFC's key-order document
with its astral-versus-BMP trap, and an ECMAScript number table read off the specification's
ToString rules rather than off CPython.

THE TRAP THIS FILE EXISTS FOR is in the number table. `repr` and `json.dumps` write `1e+16` and
`1e-07`; the RFC writes `10000000000000000` and `1e-7`. An implementation that forwards Python's
spelling looks right on every hand-written example and is wrong on the wire, so a repr passthrough
turns the table's two rows red.
"""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import canonical_bytes, content_hash
from derivus_spine.errors import CanonRefusal


def u_escapes(text):
    """`#` back to a JSON `\\u` escape.

    The RFC's documents below are quoted verbatim and are made almost entirely of unicode escapes;
    writing those tokens literally in a Python source file invites every layer between here and
    disk - editors, formatters, patch tools - to decode one of them, which would silently turn a
    published test vector into a different document. `#` appears nowhere else in either.
    """
    return text.replace('#', chr(92) + 'u')


# RFC 8785 section 3.2.3: the sample document, and the canonical form the RFC publishes for it.
RFC_SAMPLE_INPUT = u_escapes(r'''{
  "numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],
  "string": "#20ac$#000F#000aA'#0042#0022#005c\\\"\/",
  "literals": [null, true, false]
}''')
RFC_SAMPLE_OUTPUT = u_escapes(
    r'''{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'''
    r'''"string":"€$#000f\nA'B\"\\\\\"/"}''')

# The RFC's second document, whose whole point is the sort order.
RFC_KEYS_INPUT = u_escapes(r'''{
  "#20ac": "Euro Sign",
  "\r": "Carriage Return",
  "#000a": "Newline",
  "1": "One",
  "#0080": "Control#007f",
  "#d83d#de02": "Smiley",
  "#00f6": "Latin Small Letter O With Diaeresis",
  "#fb33": "Hebrew Letter Dalet With Dagesh",
  "</script>": "Browser Challenge"
}''')
RFC_KEYS_ORDER = [
    '\n', '\r', '1', '</script>', '\x80', '\xf6',   # LF, CR, DIGIT ONE, '<', PAD, o-diaeresis
    chr(0x20ac),                                    # EURO SIGN
    '\U0001f602',                                   # FACE WITH TEARS OF JOY - astral, leads 0xd83d
    chr(0xfb33),                                    # HEBREW LETTER DALET WITH DAGESH - BMP, 0xfb33
]

#: ECMAScript `Number::toString` as RFC 8785 requires it, row by row. The expectations come from
#: the specification's own rules - fixed notation while the decimal exponent sits in (-7, 21],
#: exponential outside it, `0` for both zeros - and several rows are the RFC's appendix B doubles.
ES_NUMBERS = [
    (0, '0'),
    (0.0, '0'),
    (-0.0, '0'),                                        # ToString(-0) keeps no sign
    (1, '1'),
    (-1, '-1'),
    (1.0, '1'),
    (100.0, '100'),
    (4.50, '4.5'),
    (123.456, '123.456'),
    (2e-3, '0.002'),
    (1.0 / 3.0, '0.3333333333333333'),
    (333333333.33333329, '333333333.3333333'),
    (1e16, '10000000000000000'),                        # repr says 1e+16
    (1e20, '100000000000000000000'),                    # still fixed: the exponent is exactly 21
    (1e21, '1e+21'),                                    # one past the boundary, exponential
    (2.9514790517935283e20, '295147905179352830000'),   # RFC appendix B
    (9.999999999999997e22, '9.999999999999997e+22'),    # RFC appendix B
    (1e23, '1e+23'),                                    # RFC appendix B
    (1.0000000000000001e23, '1.0000000000000001e+23'),  # RFC appendix B
    (1e30, '1e+30'),
    (1.7976931348623157e308, '1.7976931348623157e+308'),
    (-1.7976931348623157e308, '-1.7976931348623157e+308'),
    (1e-6, '0.000001'),                                 # the fixed side of the small boundary
    (9.999999999999997e-7, '9.999999999999997e-7'),     # RFC appendix B, just under it
    (1e-7, '1e-7'),                                     # the exponential side; repr says 1e-07
    (1e-27, '1e-27'),
    (2.2250738585072014e-308, '2.2250738585072014e-308'),
    (1e-323, '1e-323'),
    (5e-324, '5e-324'),                                 # the smallest subnormal
    (-5e-324, '-5e-324'),
    (2 ** 53, '9007199254740992'),                      # the largest integer that may be recorded
    (-(2 ** 53), '-9007199254740992'),
    (float(2 ** 53), '9007199254740992'),
    (2 ** 53 - 1, '9007199254740991'),
    (1000000, '1000000'),
]


def canonical_text(obj):
    """`canonical_bytes` read back as text - the assertions are about bytes, the failures read
    better as strings."""
    return canonical_bytes(obj).decode('utf-8')


def test_rfc_sample_document_matches_the_published_output():
    """The RFC's own example, byte for byte. Everything else in this file is detail of this.

    Killing mutation: the exponent's `+` dropped, which writes `1e30` where the RFC writes `1e+30`.
    """
    assert canonical_bytes(json.loads(RFC_SAMPLE_INPUT)) == RFC_SAMPLE_OUTPUT.encode('utf-8')


@pytest.mark.parametrize('value,expected', ES_NUMBERS, ids=[repr(v) for v, _ in ES_NUMBERS])
def test_es_number_serialization(value, expected):
    """One number, canonicalised alone - an array wrapper would only hide the spelling.

    Killing mutation: the fixed-notation ceiling moved from an exponent of 21 to 16, which writes
    `1e20` as `1e+20`.
    """
    assert canonical_text(value) == expected


def test_keys_sort_by_utf16_code_unit_not_code_point():
    """The RFC's key document. U+1F602 is astral - UTF-16 leads it with 0xd83d, so it sorts BEFORE
    U+FB33, while a plain code-point sort puts it after. That single inversion is the whole test.

    Killing mutation: the keys sorted by code point, which puts U+1F602 after U+FB33.
    """
    order = list(json.loads(canonical_text(json.loads(RFC_KEYS_INPUT))).keys())
    assert order == RFC_KEYS_ORDER
    naive = sorted(json.loads(RFC_KEYS_INPUT))
    assert naive.index('\U0001f602') > naive.index(chr(0xfb33))
    assert order.index('\U0001f602') < order.index(chr(0xfb33))


def test_nested_objects_sort_at_every_level():
    """Every object is sorted wherever it sits, the empty ones included, and a number inside a
    document is spelled as it is alone.

    Killing mutation: only the top level sorted, which the RFC's sample - holding no nested object -
    cannot see.
    """
    obj = {'b': {'z': 1, 'a': {'€': 0, '$': 0}}, 'a': [{'y': 1, 'x': 2}],
           'c': {'': {'': ''}, 'e': [], 'o': {}, 'values': [1e16, 1e-7]}}
    assert canonical_text(obj) == (
        '{"a":[{"x":2,"y":1}],"b":{"a":{"$":0,"€":0},"z":1},'
        '"c":{"":{"":""},"e":[],"o":{},"values":[10000000000000000,1e-7]}}')
    assert json.loads(canonical_bytes(obj)) == obj


def test_string_escaping_is_the_rfc_set_and_nothing_more():
    """The seven mandated escapes, a lowercase `u00xx` escape for the other C0 controls, and every
    other character raw UTF-8 - DEL and the solidus included, which JavaScript-flavoured encoders
    like to escape.

    Killing mutation: the control escape written in uppercase hex, `\\u001F`.
    """
    value = '"' + chr(92) + '\b\t\n\f\r' + '\x00\x01\x0b\x1f' + '\x7fé€\U0001f602/'
    assert canonical_text(value) == (
        '"' + r'\"' + chr(92) * 2 + r'\b\t\n\f\r' + u_escapes('#0000#0001#000b#001f')
        + '\x7fé€\U0001f602/' + '"')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_numbers_are_refused_by_name(value):
    """Killing mutation: the finiteness check dropped, which reaches the digit parser with `nan`."""
    with pytest.raises(CanonRefusal) as refusal:
        canonical_bytes({'body': {'value': value}})
    message = str(refusal.value)
    assert repr(value) in message                        # names the offender
    assert '$.body.value' in message                     # and where it sat
    assert 'string' in message                           # and the remedy


def test_integers_past_the_safe_range_are_refused():
    """2**53 is recorded and one past it refuses by name - and canonical output that large reparses
    as an `int` the record refuses again, loudly rather than as a hash that quietly disagrees.

    Killing mutation: the bound doubled, which records 2**53 + 1.
    """
    assert canonical_text(2 ** 53) == '9007199254740992'
    with pytest.raises(CanonRefusal) as refusal:
        canonical_bytes({'lsn': 2 ** 53 + 1})
    message = str(refusal.value)
    assert str(2 ** 53 + 1) in message and '2**53' in message
    assert 'string' in message
    with pytest.raises(CanonRefusal):
        canonical_bytes(-(2 ** 53) - 1)
    with pytest.raises(CanonRefusal):
        canonical_bytes(json.loads(canonical_text(2.9514790517935283e20)))


def test_non_json_types_are_refused_by_name():
    """Killing mutation: a set taken as an array."""
    with pytest.raises(CanonRefusal) as refusal:
        canonical_bytes({'tape': b'\x00\x01'})
    message = str(refusal.value)
    assert 'bytes' in message and '$.tape' in message
    assert 'blob store' in message                       # bulk bytes have somewhere else to be
    for offender in [set([1, 2]), complex(1, 2), object()]:
        with pytest.raises(CanonRefusal):
            canonical_bytes([offender])
    with pytest.raises(CanonRefusal) as refusal:
        canonical_bytes({1: 'one'})
    assert 'int' in str(refusal.value)


def test_semantically_equal_documents_hash_identically():
    """Canonical identity: key order and number spelling are not facts about a document, and `1`
    and `1.0` are one number.

    Killing mutation: `-0.0` spelled with its sign.
    """
    assert content_hash({'a': 1, 'b': [2, 3]}) == content_hash({'b': [2, 3], 'a': 1})
    assert content_hash({'x': 1e2}) == content_hash({'x': 100})
    assert content_hash({'quantity': 5}) == content_hash({'quantity': 5.0})
    assert content_hash({'x': -0.0}) == content_hash({'x': 0.0})
    assert content_hash({'a': 1}) != content_hash({'a': 2})


def test_content_hash_is_sha256_of_the_canonical_bytes():
    """Killing mutation: the address taken under SHA3-256."""
    obj = {'type': 'checkpoint', 'lsn': 3}
    assert content_hash(obj) == hashlib.sha256(canonical_bytes(obj)).hexdigest()
    assert len(content_hash(obj)) == 64
