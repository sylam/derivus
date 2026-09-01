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

"""RFC 8785 (JSON Canonicalization Scheme), vendored - one spelling per value, so a hash means
something.

Every hash in the spine is taken over the bytes this module emits: the content hash that is an
instrument's id, the idempotency tag the writer blinds, the interior binding sealed inside a body,
the payload a checkpoint signs. The RFC is adopted outright, not approximated. `json.dumps` is not
this: the RFC serialises numbers the way ECMAScript does, so `repr`'s `1e+16` must be written
`10000000000000000` and `1e-07` must be written `1e-7`. Keys sort by UTF-16 CODE UNIT rather than
code point, which puts an astral character ahead of U+E000..U+FFFF where a plain `sorted()` puts
it after.

Declared limitation: canonical bytes are not a fixpoint of `json.loads` for magnitudes at or past
2**53. Such a value canonicalises to a plain integer literal, and re-parsing yields a Python `int`,
which this module then refuses. The failure is loud, never a differing hash; anything re-parsing
canonical bytes passes `parse_int` to map oversized literals back to `float`.

Stdlib only, importing nothing from the spine but its refusal type.
"""

import hashlib
import math
import re

from .errors import CanonRefusal

#: Beyond this an integer no longer round-trips through the IEEE-754 double the RFC's number rule
#: is written in - the record refuses rather than quietly losing a digit.
MAX_SAFE_INTEGER = 2 ** 53

#: `repr` of a positive finite float, taken apart. CPython's shortest round-tripping digits are
#: the `k`-minimal digit string ECMAScript's ToString is defined over, so those digits are
#: re-dressed under the RFC's formatting rules, never re-derived.
_REPR = re.compile(r'^(\d+)(?:\.(\d+))?(?:[eE]([-+]?\d+))?$')

#: The two-character escapes RFC 8785 mandates. Everything else below U+0020 goes out as
#: `\u00xx` in LOWERCASE hex; DEL, U+2028 and the rest travel raw as UTF-8.
_ESCAPES = {
    '"': '\\"',
    '\\': '\\\\',
    '\b': '\\b',
    '\f': '\\f',
    '\n': '\\n',
    '\r': '\\r',
    '\t': '\\t',
}


def canonical_bytes(obj):
    """The RFC 8785 canonical UTF-8 encoding of `obj`.

    Accepts the JSON types and nothing else: `dict` with string keys, `list`/`tuple`, `str`,
    `bool`, `int`, `float`, `None`. Anything else raises `CanonRefusal` naming the path it sat at.
    """
    out = []
    _emit(obj, out, '$')
    text = ''.join(out)
    try:
        return text.encode('utf-8')
    except UnicodeEncodeError as exc:
        # A lone surrogate in a str - it has no UTF-8 form, so it has no canonical form either.
        raise CanonRefusal(
            'canonical_bytes: the value contains a lone surrogate at position {0} and has no '
            'UTF-8 encoding - repair the string at the caller (it did not come from valid '
            'JSON)'.format(exc.start))


def content_hash(obj):
    """SHA-256 of `canonical_bytes(obj)`, hex. The spine's one address for a value."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _emit(value, out, path):
    """Append `value`'s canonical text to `out`. `path` is carried only to make refusals nameable."""
    if value is None:
        out.append('null')
    elif value is True:
        out.append('true')
    elif value is False:
        out.append('false')
    elif isinstance(value, str):
        out.append(_string(value))
    elif isinstance(value, int):
        # bool is an int, and was already taken above.
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonRefusal(
                '{0}: the integer {1} is past 2**53 and cannot round-trip through the IEEE-754 '
                'double RFC 8785 numbers are defined over - carry it as a decimal string'.format(
                    path, value))
        out.append(_number(float(value), path))
    elif isinstance(value, float):
        out.append(_number(value, path))
    elif isinstance(value, dict):
        out.append('{')
        first = True
        for key in _sorted_keys(value, path):
            if not first:
                out.append(',')
            first = False
            out.append(_string(key))
            out.append(':')
            _emit(value[key], out, '{0}.{1}'.format(path, key))
        out.append('}')
    elif isinstance(value, (list, tuple)):
        out.append('[')
        for index, item in enumerate(value):
            if index:
                out.append(',')
            _emit(item, out, '{0}[{1}]'.format(path, index))
        out.append(']')
    else:
        raise CanonRefusal(
            '{0}: {1} is not a JSON type - encode it before it reaches the record (bulk bytes '
            'belong in the blob store, spoken of by hash)'.format(path, type(value).__name__))


def _sorted_keys(mapping, path):
    """`mapping`'s keys in UTF-16 code-unit order, refusing any non-string key.

    The big-endian UTF-16 encoding is the sort key: it compares bytewise exactly as the code units
    compare, placing an astral character ahead of U+E000..U+FFFF.
    """
    for key in mapping:
        if not isinstance(key, str):
            raise CanonRefusal(
                '{0}: the object key {1!r} is {2} - JSON object keys are strings; stringify it at '
                'the caller so the record says which spelling it meant'.format(
                    path, key, type(key).__name__))
    return sorted(mapping, key=lambda k: k.encode('utf-16-be', 'surrogatepass'))


def _number(value, path='$'):
    """ECMAScript `Number::toString` of `value`, which is what RFC 8785 means by a number.

    Fixed notation while the decimal exponent sits in (-7, 21], exponential outside it, `0` for
    both zeros. Non-finite values raise `CanonRefusal`.
    """
    if not math.isfinite(value):
        raise CanonRefusal(
            '{0}: {1!r} is not a JSON number - RFC 8785 admits only finite values, so record the '
            'reason as a string field rather than a non-finite number'.format(path, value))
    if value == 0.0:
        return '0'  # -0.0 lands here too: ToString(-0) is "0", sign and all.
    sign = '-' if value < 0.0 else ''
    digits, n = _digits(abs(value))
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + '0' * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + '.' + digits[n:]
    if -6 < n <= 0:
        return sign + '0.' + '0' * (-n) + digits
    exponent = n - 1
    tail = 'e' + ('+' if exponent >= 0 else '-') + str(abs(exponent))
    if k == 1:
        return sign + digits + tail
    return sign + digits[0] + '.' + digits[1:] + tail


def _digits(magnitude):
    """`(digits, n)` for a positive finite float, with magnitude == 0.<digits> * 10**n and
    `digits` carrying no leading or trailing zero - the triple ECMAScript's ToString is written in.
    """
    match = _REPR.match(repr(magnitude))
    integer, fraction, exponent = match.group(1), match.group(2) or '', match.group(3)
    digits = integer + fraction
    n = len(integer) + (int(exponent) if exponent else 0)
    lead = digits.lstrip('0')
    n -= len(digits) - len(lead)
    return lead.rstrip('0'), n


def _string(value):
    """A JSON string literal under the RFC's escaping: the seven mandated escapes, `\\u00xx` in
    lowercase for the remaining C0 controls, and every other character raw."""
    out = ['"']
    for char in value:
        escape = _ESCAPES.get(char)
        if escape is not None:
            out.append(escape)
        elif char < ' ':
            out.append('\\u%04x' % ord(char))
        else:
            out.append(char)
    out.append('"')
    return ''.join(out)
