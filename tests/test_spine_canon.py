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

THE TRAP THIS FILE EXISTS FOR is the last test. `repr` and `json.dumps` write `1e+16` and `1e-07`;
the RFC writes `10000000000000000` and `1e-7`. An implementation that forwards Python's spelling
looks right on every hand-written example and is wrong on the wire, so the gate asserts the two
spellings DIFFER from Python's - a repr passthrough turns this file red.
"""
import hashlib
import json
import math
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


def test_the_rfc_vectors_survived_transcription():
    """Before anything is asserted ABOUT the vectors, assert they are the RFC's. The escapes above
    are assembled at import, so a broken assembly would quietly weaken every test below."""
    sample = json.loads(RFC_SAMPLE_INPUT)
    assert sorted(sample) == ['literals', 'numbers', 'string']
    assert sample['string'] == '€$\x0f\nA\'B"' + chr(92) * 2 + '"/'
    assert len(sample['numbers']) == 5
    keys = json.loads(RFC_KEYS_INPUT)
    assert len(keys) == 9 and '\U0001f602' in keys and chr(0xfb33) in keys
    assert sorted(keys) == sorted(RFC_KEYS_ORDER)


def test_rfc_sample_document_matches_the_published_output():
    """The RFC's own example, byte for byte. Everything else in this file is detail of this."""
    assert canonical_bytes(json.loads(RFC_SAMPLE_INPUT)) == RFC_SAMPLE_OUTPUT.encode('utf-8')


def test_rfc_sample_output_is_utf8_and_reparses():
    canonical = canonical_bytes(json.loads(RFC_SAMPLE_INPUT))
    assert canonical.decode('utf-8')                     # no BOM, no surrogates, nothing lost
    assert json.loads(canonical) == json.loads(RFC_SAMPLE_INPUT)


@pytest.mark.parametrize('value,expected', ES_NUMBERS, ids=[repr(v) for v, _ in ES_NUMBERS])
def test_es_number_serialization(value, expected):
    """One number, canonicalised alone - an array wrapper would only hide the spelling."""
    assert canonical_text(value) == expected


def test_es_number_table_straddles_both_boundaries():
    """The table earns its place only by sitting on both sides of the two places the rule turns."""
    spellings = set(expected for _, expected in ES_NUMBERS)
    assert '10000000000000000' in spellings and '1e+21' in spellings   # the 21 boundary, both sides
    assert '0.000001' in spellings and '1e-7' in spellings             # the -7 boundary, both sides
    assert len(ES_NUMBERS) >= 25


def test_integers_and_floats_of_equal_value_share_a_spelling():
    """RFC 8785 has one number type, so `1` and `1.0` are one fact and hash as one."""
    assert canonical_text(1) == canonical_text(1.0) == '1'
    assert content_hash({'quantity': 5}) == content_hash({'quantity': 5.0})


def test_keys_sort_by_utf16_code_unit_not_code_point():
    """The RFC's key document. U+1F602 is astral - UTF-16 leads it with 0xd83d, so it sorts BEFORE
    U+FB33, while a plain code-point sort puts it after. That single inversion is the whole test.
    """
    order = list(json.loads(canonical_text(json.loads(RFC_KEYS_INPUT))).keys())
    assert order == RFC_KEYS_ORDER
    naive = sorted(json.loads(RFC_KEYS_INPUT))
    assert naive != order, (
        'the chosen keys cannot separate UTF-16 order from code-point order - pick an astral '
        'character against a BMP character above U+E000 (this document has U+1F602 and U+FB33)')
    assert naive.index('\U0001f602') > naive.index(chr(0xfb33))
    assert order.index('\U0001f602') < order.index(chr(0xfb33))


def test_a_purely_bmp_document_sorts_the_same_either_way():
    """The counterexample to the counterexample: with no astral key the two orders agree, and the
    canonicaliser must not invent a difference to prove a point."""
    keys = {'A': 1, 'a': 2, '\xf6': 3, chr(0xe000): 4}
    order = list(json.loads(canonical_text(keys)).keys())
    assert order == sorted(keys) == ['A', 'a', '\xf6', chr(0xe000)]


def test_nested_objects_sort_at_every_level():
    obj = {'b': {'z': 1, 'a': {'€': 0, '$': 0}}, 'a': [{'y': 1, 'x': 2}]}
    assert canonical_text(obj) == '{"a":[{"x":2,"y":1}],"b":{"a":{"$":0,"€":0},"z":1}}'


def test_string_escaping_is_the_rfc_set_and_nothing_more():
    """The seven mandated escapes, a lowercase `u00xx` escape for the other C0 controls, and every
    other character raw UTF-8 - DEL included, which JavaScript-flavoured encoders like to escape."""
    value = '"' + chr(92) + '\b\t\n\f\r' + '\x00\x01\x1f' + '\x7fé€\U0001f602/'
    assert canonical_text(value) == (
        '"' + r'\"' + chr(92) * 2 + r'\b\t\n\f\r' + u_escapes('#0000#0001#001f')
        + '\x7fé€\U0001f602/' + '"')


def test_control_escapes_are_lowercase_hex():
    assert canonical_text('\x1f') == '"' + u_escapes('#001f') + '"'
    assert canonical_text('\x0b') == '"' + u_escapes('#000b') + '"'
    assert u_escapes('#001F') not in canonical_text('\x1f')


def test_solidus_and_del_travel_raw():
    """Two characters the RFC explicitly does NOT escape - a canonicaliser that escapes either
    produces bytes no other implementation will reproduce."""
    assert canonical_text('a/b') == '"a/b"'
    assert canonical_text('Control\x7f') == '"Control\x7f"'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_numbers_are_refused_by_name(value):
    with pytest.raises(CanonRefusal) as refusal:
        canonical_bytes({'body': {'value': value}})
    message = str(refusal.value)
    assert repr(value) in message                        # names the offender
    assert '$.body.value' in message                     # and where it sat
    assert 'string' in message                           # and the remedy


def test_integers_past_the_safe_range_are_refused():
    assert canonical_text(2 ** 53) == '9007199254740992'
    with pytest.raises(CanonRefusal) as refusal:
        canonical_bytes({'lsn': 2 ** 53 + 1})
    message = str(refusal.value)
    assert str(2 ** 53 + 1) in message and '2**53' in message
    assert 'string' in message
    with pytest.raises(CanonRefusal):
        canonical_bytes(-(2 ** 53) - 1)


def test_non_json_types_are_refused_by_name():
    with pytest.raises(CanonRefusal) as refusal:
        canonical_bytes({'tape': b'\x00\x01'})
    message = str(refusal.value)
    assert 'bytes' in message and '$.tape' in message
    assert 'blob store' in message                       # bulk bytes have somewhere else to be
    for offender in [set([1, 2]), complex(1, 2), object()]:
        with pytest.raises(CanonRefusal):
            canonical_bytes([offender])


def test_non_string_object_keys_are_refused():
    with pytest.raises(CanonRefusal) as refusal:
        canonical_bytes({1: 'one'})
    assert 'int' in str(refusal.value)


def test_a_refusal_returns_nothing_at_all():
    """A refusal is not a half-written record - the caller gets an exception, never a prefix."""
    with pytest.raises(CanonRefusal):
        canonical_bytes({'a': 1, 'b': float('nan'), 'c': 2})


def test_round_trip_through_json():
    """Canonical bytes are still JSON - an auditor reads them with any parser."""
    fixtures = [
        {'type': 'fill',
         'body': {'quantity': -1000000.5, 'counterparty': 'LEI:5493001KJTIIGC8Y1R12',
                  'netting_set': 'NS/1', 'execution_reference': 'XETR-88213'}},
        {'grants': [{'subject': 'sub-1', 'scope': 'admin'}, {'subject': 'sub-2', 'scope': 'mark'}]},
        [None, True, False, 0, 1e-7, 'ünïcødé \U0001f602', {'nested': [[[{'deep': 1}]]]}],
        {'': {'': ''}},
        [],
        {},
        'a bare string',
        42,
    ]
    for fixture in fixtures:
        assert json.loads(canonical_bytes(fixture)) == fixture


def test_large_magnitudes_reparse_loudly_rather_than_silently():
    """The declared asymmetry, pinned so nobody meets it as a mystery.

    Canonical output past 2**53 is a plain integer literal, and `json.loads` hands it back as an
    `int` the record then refuses - which is the RIGHT failure (a refusal naming the value) rather
    than the wrong one (a hash that quietly disagrees). The remedy is `parse_int`, and it is
    asserted here so the fix is in the gate rather than in somebody's memory.
    """
    canonical = canonical_text(2.9514790517935283e20)      # an RFC appendix B double
    assert canonical == '295147905179352830000'
    assert isinstance(json.loads(canonical), int)          # the parser hands back an int...
    with pytest.raises(CanonRefusal):                      # ...and the record says no, out loud
        canonical_bytes(json.loads(canonical))

    def keep_doubles(text):
        return float(text) if abs(int(text)) > 2 ** 53 else int(text)

    assert canonical_bytes(json.loads(canonical, parse_int=keep_doubles)) == canonical.encode()


def test_semantically_equal_documents_hash_identically():
    """Canonical identity: key order and number spelling are not facts about a document."""
    assert content_hash({'a': 1, 'b': [2, 3]}) == content_hash({'b': [2, 3], 'a': 1})
    assert content_hash({'x': 1e2}) == content_hash({'x': 100})
    assert content_hash({'x': -0.0}) == content_hash({'x': 0.0})
    assert content_hash({'a': 1}) != content_hash({'a': 2})


def test_content_hash_is_sha256_of_the_canonical_bytes():
    obj = {'type': 'checkpoint', 'lsn': 3}
    assert content_hash(obj) == hashlib.sha256(canonical_bytes(obj)).hexdigest()
    assert len(content_hash(obj)) == 64


def test_repr_passthrough_would_be_wrong():
    """THE TRAP. `repr` and `json.dumps` disagree with the RFC on exactly these two values, so an
    implementation that forwards Python's spelling fails here and only here - which is why the
    serializer is vendored instead of reaching for `json.dumps(sort_keys=True)`.
    """
    assert canonical_text(1e16) == '10000000000000000'
    assert repr(1e16) == '1e+16'
    assert json.dumps(1e16) == '1e+16'
    assert canonical_text(1e16) != repr(1e16)

    assert canonical_text(1e-7) == '1e-7'
    assert repr(1e-7) == '1e-07'
    assert json.dumps(1e-7) == '1e-07'
    assert canonical_text(1e-7) != repr(1e-7)

    # And the same disagreement inside a real document, which is where it would actually bite.
    document = {'values': [1e16, 1e-7]}
    assert canonical_text(document) == '{"values":[10000000000000000,1e-7]}'
    assert canonical_text(document) != json.dumps(document, sort_keys=True, separators=(',', ':'))


def test_the_canonicaliser_holds_no_state():
    """Two calls, one answer - a canonicaliser with memory would hash the same fact two ways."""
    obj = {'b': 1, 'a': [float(2 ** 53), -0.0, math.pi]}
    assert canonical_bytes(obj) == canonical_bytes(obj)
    assert canonical_bytes(obj) == canonical_bytes(json.loads(canonical_bytes(obj)))
