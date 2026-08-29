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

"""Who the actor is - verified rather than asserted - and the one mutable place a human name lives.

Identity is BOUGHT, not built. The deployment already runs an IdP; this module does the one thing
the spine cannot delegate, which is to CHECK the token that IdP issued before an actor reference
gets stamped into a fact that outlives everyone in the room. There is no HTTP here, no discovery
document, no token acquisition and no JWKS fetching: the key set arrives as a `dict` the deployment
hands in as data, which is what keeps the truth layer's import surface at stdlib plus
`cryptography` and keeps a verifier honest on a machine with no network at all.

The allowlist is the whole security posture in one line: RS256 and ES256, and nothing else, ever.
`alg: none` and every HMAC family are refused BY NAME rather than falling through some default,
because the classic forgery against a JWT verifier is not a broken signature - it is a token that
declares HS256 and is "signed" with the RSA public key the JWKS PUBLISHES, which a naive verifier
happily checks with the attacker's own material. Refusing on the declared algorithm before a key is
ever selected is what makes that attack unrepresentable instead of merely unlikely.

Order matters as much as the checks do. Segments are split, the header is read, the algorithm is
allowlisted, a key is selected by `kid`, the SIGNATURE is verified - and only then is the payload
parsed and its claims read. Nothing in this module makes a decision on unverified bytes, which is
why one altered payload byte lands as a signature refusal rather than as a claim the verifier went
on to reason about. One encoding trap is written into the contract because it silently half-works
otherwise: a JWS ES256 signature is the RAW fixed-width pair `R || S` (64 bytes, RFC 7518 3.4)
while `cryptography` verifies the DER SEQUENCE that OpenSSL emits, so a DER signature arriving here
is refused on its length rather than converted for the sender's convenience.

What comes back is a SUBJECT REFERENCE - the token's `sub`, verbatim - and the log gets nothing
else. The record stays pseudonymous by rule, and nothing secret is written anywhere by anything in
this file: no token, no claim set, no key material reaches the disk on the verification path, which
holds no path at all. Display names are the counterpart to that rule and the exception that proves
it: they live in `<home>/names.json`, a mutable, unhashed side table beside the log and never
inside it, so an erasure request is a dict key going away rather than a chain to rewrite. That is
the whole reason the side table exists - a name in a sealed body would be crypto-shreddable, but a
name in the ENVELOPE would be permanent, and erasure regimes do not accept "the hash chain says no"
as an answer.
"""
import base64
import binascii
import json
import math
import os
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from .errors import IdentityRefused

#: The allowlist, as `alg -> the JWK key type that alg is signed with`. Two asymmetric algorithms
#: and no third: a symmetric alg in an ID token means the verifier holds the signing secret, which
#: is exactly the confusion this map exists to refuse.
ALGORITHMS = {'RS256': 'RSA', 'ES256': 'EC'}
#: The only curve ES256 is defined over.
ES256_CURVE = 'P-256'
#: RFC 7518 3.4: the ES256 signature is `R || S`, each coordinate a fixed 32 bytes big-endian.
ES256_SIGNATURE_BYTES = 64
P256_COORDINATE_BYTES = 32
#: Clock skew between the IdP and this box. Sixty seconds is slack, not policy - a token an hour
#: dead is dead here, and `now` is injectable so a gate never has to ask the wall clock what time
#: it is.
LEEWAY_SECONDS = 60.0
#: The side table, at the home's ROOT. Never under `log/` - the log directory is what a replica
#: file-copies and what the chain covers, and an erasable attribute may not live in either.
NAMES_FILE = 'names.json'
#: base64url's alphabet, checked before decoding so a segment carrying `+` or `/` is a named
#: refusal rather than something Python's decoder quietly tolerates.
B64URL_ALPHABET = frozenset(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_')
#: The padding character, which is not in the alphabet and is read only where padding may be.
B64URL_PAD = '='


def verify_id_token(token: str, jwks: dict, issuer: str, audience: str, now: float = None) -> dict:
    """Verify an OIDC ID token against `jwks` and answer `{subject, issuer, claims}`.

    Every way this can fail is an `IdentityRefused` naming what was wrong and what fixes it. The
    sequence is the security property: parse, allowlist the algorithm, select the key, VERIFY, and
    only then believe a claim. `now` is seconds since the epoch and defaults to the wall clock;
    callers that need a deterministic answer - gates, as-of replays - pass it.
    """
    header, segments = _parse(token)
    algorithm = _allowed_algorithm(header.get('alg'))
    signature = _b64url_decode(segments[2], 'the JWS signature')
    signing_input = '.'.join(segments[:2]).encode('ascii')

    _verify_signature(_candidates(jwks, header, algorithm), algorithm, signing_input, signature)

    claims = _json_object(_b64url_decode(segments[1], 'the JWS payload'), 'the JWS payload')
    _check_claims(claims, issuer, audience, time.time() if now is None else now)
    # The subject reference and nothing else: the log is pseudonymous by rule, so the caller gets
    # the claims to decide with and the RECORD gets `subject`.
    return {'subject': claims['sub'], 'issuer': claims['iss'], 'claims': dict(claims)}


def display_names(home) -> dict:
    """The `subject -> display name` side table of the home at `home`, or `{}` where there is none.

    A read never provisions: a home with no side table has no names, which is not an error and not
    a reason to write a file.
    """
    try:
        raw = (Path(home) / NAMES_FILE).read_bytes()
    except (IOError, OSError):
        return {}
    try:
        table = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        raise IdentityRefused(
            '{}/{} is not UTF-8 JSON: the side table is an ordinary editable file and something '
            'has truncated or mangled it - restore it from a backup, or delete it and re-enter the '
            'names (nothing in the log depends on it)'.format(home, NAMES_FILE))
    if not isinstance(table, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in table.items()):
        raise IdentityRefused(
            '{}/{} is not an object of subject -> display name: the side table maps a subject '
            'reference to one string - correct the file, or delete it and re-enter the '
            'names'.format(home, NAMES_FILE))
    return table


def set_display_name(home, subject, name) -> dict:
    """Name `subject` in the home's side table. Answers the whole table as it now stands.

    The subject is the reference the log carries; the name is the erasable attribute that must
    never go near it.
    """
    subject = _subject_reference(subject)
    if not isinstance(name, str) or not name.strip():
        raise IdentityRefused(
            'the display name for {} is {!r}, which is not a name: supply a non-empty string, or '
            'call erase_display_name to remove the entry - a blank name is not how something is '
            'forgotten here'.format(subject, name))
    table = display_names(home)
    table[subject] = name
    return _write_names(home, table)


def erase_display_name(home, subject) -> dict:
    """Forget `subject`'s display name. Answers the whole table as it now stands.

    This is the erasure path, and it is deliberately the ONLY one: the name is a dict key going
    away in a file no hash covers, so not one byte of the log, the chain, or the blob store moves -
    which is the property a gate asserts by hashing the segments either side of this call. Erasing
    a name that is not there is already true, so it is not an error.
    """
    table = display_names(home)
    table.pop(_subject_reference(subject), None)
    return _write_names(home, table)


def _parse(token):
    """The three segments of a compact JWS and its parsed header.

    Only the header is read here. The payload stays raw text until a signature has vouched for it.
    """
    if not isinstance(token, str) or not token:
        raise IdentityRefused(
            'the ID token is {}, not a compact JWS string: hand the token the IdP issued in '
            'exactly as it arrived - this verifier reads the serialization, not a decoded dict and '
            'not an `Authorization: Bearer ...` header'.format(
                'empty' if isinstance(token, str) else type(token).__name__))
    segments = token.split('.')
    if len(segments) != 3:
        raise IdentityRefused(
            'the ID token has {} dot-separated segments, not the 3 of a compact JWS '
            '(header.payload.signature): a 5-segment JWE or a truncated paste is not something '
            'this verifier can check - re-acquire the token from the IdP'.format(len(segments)))
    for segment, what in zip(segments, ('the JWS header', 'the JWS payload', 'the JWS signature')):
        _check_alphabet(segment, what)
    return _json_object(_b64url_decode(segments[0], 'the JWS header'), 'the JWS header'), segments


def _check_alphabet(segment, what):
    """base64url and nothing else, with `=` tolerated where padding lives and nowhere else.

    Emptiness is allowed here and refused at the decode, so that an unsigned token is refused for its
    ALGORITHM rather than for its missing bytes. The trailing `=` is tolerated for the same reason
    `_b64url_decode` re-pads: JWS strips padding, and most copy-pastes and a few issuers do not, so a
    padded token is a paste to read rather than a corruption to report. Inside a segment `=` is still
    a refusal - that is a re-encoded token, which is what this check exists to catch.
    """
    if not isinstance(segment, str) or not B64URL_ALPHABET.issuperset(segment.rstrip(B64URL_PAD)):
        raise IdentityRefused(
            '{} is not base64url: the segment carries characters outside [A-Za-z0-9_-], with `=` '
            'read only as trailing padding - the token has been re-encoded or corrupted in '
            'transport, so re-acquire it from the IdP'.format(what))


def _b64url_decode(segment, what):
    """The bytes behind one base64url segment, padding tolerated (JWS strips it; some issuers and
    most copy-pastes do not) and the encoding required to be CANONICAL.

    Canonical means the trailing character's unused bits are zero, which base64's own arithmetic
    leaves them: a 256-byte RSA signature is 342 characters plus 2 significant bits in the last one,
    so four spellings of that character decode to identical bytes. A decoder that accepts all four
    accepts four spellings of one signed token - and a gate flipping that character to prove tamper
    detection silently proves nothing. JWS mandates the canonical spelling; this is where it is
    enforced, by re-encoding what came out and requiring it to be what went in.
    """
    if isinstance(segment, str):
        segment = segment.rstrip(B64URL_PAD)
    if not segment:
        raise IdentityRefused(
            '{} is empty: every segment of a compact JWS carries bytes - an empty signature is the '
            '`alg: none` forgery and an empty header or payload is a truncated token; re-acquire '
            'it from the IdP'.format(what))
    try:
        material = base64.urlsafe_b64decode(segment + B64URL_PAD * (-len(segment) % 4))
    except (binascii.Error, ValueError):
        raise IdentityRefused(
            '{} does not decode as base64url: the token is truncated or was mangled in transport - '
            're-acquire it from the IdP'.format(what))
    if base64.urlsafe_b64encode(material).rstrip(b'=').decode('ascii') != segment:
        raise IdentityRefused(
            '{} is not the CANONICAL base64url of the bytes it decodes to: the final character '
            'carries bits that base64 leaves zero, so this spelling is one of several that decode '
            'alike and JWS admits only one of them - the token has been rewritten, so re-acquire it '
            'from the IdP'.format(what))
    return material


def _json_object(raw, what):
    """A JSON object, or a refusal naming which segment was not one."""
    try:
        value = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        raise IdentityRefused(
            '{} is not UTF-8 JSON: what decoded out of the segment is not a JOSE object at all - '
            'the token is corrupt or is not an ID token; re-acquire it from the IdP'.format(what))
    if not isinstance(value, dict):
        raise IdentityRefused(
            '{} decodes to {}, not a JSON object: a JOSE header and a claim set are both objects - '
            're-acquire the token from the IdP'.format(what, type(value).__name__))
    return value


def _allowed_algorithm(alg):
    """`alg` itself once it is on the allowlist, or the refusal that closes the confusion.

    This runs BEFORE any key is selected, which is the point. The HMAC families are refused here
    because a verifier that accepts one will check an attacker's forgery against the very public
    key the JWKS publishes, and `none` is refused here because the alternative is a signature check
    that is skipped rather than failed.
    """
    if not isinstance(alg, str) or not alg:
        raise IdentityRefused(
            'the JWS header declares no alg: an unlabelled token cannot be checked, and guessing '
            'the algorithm from the key is the confusion this verifier exists to refuse - '
            'reconfigure the IdP to sign ID tokens with RS256 or ES256')
    if alg not in ALGORITHMS:
        raise IdentityRefused(
            'alg {!r} is outside the allowlist {}: `none` means the signature is not checked at '
            'all, and an HMAC alg (HS256 and family) means a token can be forged under the PUBLIC '
            'key the JWKS publishes - the alg-confusion attack; reconfigure the IdP to sign ID '
            'tokens with RS256 or ES256'.format(alg, ' and '.join(sorted(ALGORITHMS))))
    return alg


def _candidates(jwks, header, algorithm):
    """The JWKS entries this token may be checked against, selected by `kid` where it has one.

    A token with a `kid` is checked against that key and no other. A token without one is checked
    against every key of the right type, which is the rotation case an IdP that omits `kid` leaves
    a verifier in - it is a wider door, so it is narrowed by key type, declared `alg` and `use`
    rather than left open.
    """
    key_type = ALGORITHMS[algorithm]
    if not isinstance(jwks, dict) or not isinstance(jwks.get('keys'), list):
        raise IdentityRefused(
            'the JWKS is not a JWK Set ({{"keys": [...]}}) but {}: the deployment hands the key '
            'set in as DATA - nothing here fetches one - so fix the JWKS file the deployment '
            'provides and pass it in parsed'.format(type(jwks).__name__))
    keys = [key for key in jwks['keys'] if isinstance(key, dict)]
    kid = header.get('kid')
    if kid is not None and not isinstance(kid, str):
        raise IdentityRefused(
            'the JWS header names kid {!r}, which is not a string: a key id is text - the token is '
            'malformed, so re-acquire it from the IdP'.format(kid))

    if kid is None:
        chosen, where = keys, 'the JWKS holds no {} key at all'.format(key_type)
    else:
        chosen = [key for key in keys if key.get('kid') == kid]
        where = 'kid {!r} is not in the JWKS, which publishes {}'.format(
            kid, ', '.join(repr(key.get('kid')) for key in keys) or 'no keys at all')
    usable = [key for key in chosen if key.get('kty') == key_type
              and key.get('use', 'sig') == 'sig' and key.get('alg', algorithm) == algorithm]
    if not usable:
        raise IdentityRefused(
            'no key in the JWKS can check this {} token: {} - the IdP has rotated its signing key, '
            'or this token was issued by another deployment; refresh the JWKS file the deployment '
            'provides and verify again'.format(algorithm, where))
    return usable


def _verify_signature(candidates, algorithm, signing_input, signature):
    """Check `signature` over `signing_input` under each candidate; refuse if none vouches for it.

    Nothing downstream of this runs on bytes it did not authenticate, so a single altered payload
    byte stops here rather than surfacing as a claim.

    An UNUSABLE candidate does not end the sweep. A kidless token is checked against every key of the
    right type, which is the rotation case an IdP that omits `kid` leaves a verifier in - and a real
    key set carries entries this build cannot turn into a key: a P-384 EC entry beside the live P-256
    one, a JWK missing a member. Letting the first of those out of the loop would sink every kidless
    token behind it and report the wrong reason for it. So they are collected instead, and one is
    re-raised only when EVERY candidate was unusable - which is the case where naming the broken
    member is the most specific true thing, and is what a single named `kid` resolving to a malformed
    entry still gets.
    """
    if algorithm == 'ES256':
        signature = _der_from_raw(signature)
    unusable = []
    for key in candidates:
        try:
            public_key = _public_key(key, algorithm)
            if algorithm == 'RS256':
                public_key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
            else:
                public_key.verify(signature, signing_input, ec.ECDSA(hashes.SHA256()))
            return key
        except IdentityRefused as broken:
            unusable.append(broken)
            continue
        except InvalidSignature:
            continue
    if len(unusable) == len(candidates):
        raise unusable[0]
    raise IdentityRefused(
        'the {} signature does not verify under {} of the JWKS: the token has been altered in '
        'transit, or it was signed by a key this deployment does not publish - re-acquire the '
        'token, and refresh the JWKS file if the IdP has rotated'.format(
            algorithm,
            'the key' if len(candidates) == 1 else 'any of the {} keys'.format(len(candidates))))


def _der_from_raw(signature):
    """The DER SEQUENCE `cryptography` verifies, built from the raw `R || S` pair JWS carries.

    The trap, stated out loud: RFC 7518 3.4 defines the ES256 signature as two fixed-width
    32-byte integers concatenated, while every OpenSSL-shaped library - this one included - signs
    and verifies the DER encoding, which for P-256 is 70 or 71 bytes of tag-length-value. They are
    the same signature in two spellings and neither library will tell you which one it got, so the
    encoding is part of this verifier's contract and a DER blob is refused on its LENGTH.
    """
    if len(signature) != ES256_SIGNATURE_BYTES:
        raise IdentityRefused(
            'the ES256 signature is {} bytes, not the {} of a raw R||S pair: JWS ES256 is the '
            'fixed-width pair of RFC 7518 3.4, not the DER SEQUENCE that OpenSSL and '
            '`cryptography` emit - convert at the signer (decode_dss_signature, then two 32-byte '
            'big-endian writes) rather than sending DER'.format(
                len(signature), ES256_SIGNATURE_BYTES))
    return encode_dss_signature(
        int.from_bytes(signature[:P256_COORDINATE_BYTES], 'big'),
        int.from_bytes(signature[P256_COORDINATE_BYTES:], 'big'))


def _public_key(key, algorithm):
    """One JWK turned into a `cryptography` public key - `kty`/`n`/`e` for RSA, `crv`/`x`/`y` for
    P-256 - or a refusal naming the field that was not a key."""
    if algorithm == 'RS256':
        modulus = _b64url_int(key, 'n')
        exponent = _b64url_int(key, 'e')
        try:
            return rsa.RSAPublicNumbers(exponent, modulus).public_key()
        except (ValueError, TypeError):
            raise IdentityRefused(
                'the JWKS entry for kid {!r} is not an RSA public key: n and e decode but do not '
                'form one - the key set is corrupt, so refresh the JWKS file the deployment '
                'provides'.format(key.get('kid')))
    curve = key.get('crv')
    if curve != ES256_CURVE:
        raise IdentityRefused(
            'the JWKS entry for kid {!r} declares crv {!r}: ES256 is defined over {} alone - a key '
            'on another curve cannot check this token, so refresh the JWKS file or reconfigure the '
            'IdP'.format(key.get('kid'), curve, ES256_CURVE))
    coordinates = []
    for field in ('x', 'y'):
        raw = _b64url_decode(_field(key, field), 'the JWKS entry\'s {}'.format(field))
        if len(raw) != P256_COORDINATE_BYTES:
            raise IdentityRefused(
                'the JWKS entry for kid {!r} has a {}-byte {} coordinate, not {}: a {} coordinate '
                'is a FIXED-WIDTH octet string, left-padded with zeros - refresh the JWKS file the '
                'deployment provides'.format(
                    key.get('kid'), len(raw), field, P256_COORDINATE_BYTES, ES256_CURVE))
        coordinates.append(int.from_bytes(raw, 'big'))
    try:
        return ec.EllipticCurvePublicNumbers(
            coordinates[0], coordinates[1], ec.SECP256R1()).public_key()
    except (ValueError, TypeError):
        raise IdentityRefused(
            'the JWKS entry for kid {!r} is not a point on {}: x and y decode but do not lie on '
            'the curve - the key set is corrupt, so refresh the JWKS file the deployment '
            'provides'.format(key.get('kid'), ES256_CURVE))


def _field(key, name):
    """One required JWK member, or the refusal naming it."""
    value = key.get(name)
    if not isinstance(value, str) or not value:
        raise IdentityRefused(
            'the JWKS entry for kid {!r} has no {}: a {} key is spelled by its members, and one is '
            'missing - refresh the JWKS file the deployment provides'.format(
                key.get('kid'), name, key.get('kty')))
    return value


def _b64url_int(key, name):
    """A JWK's big-endian unsigned integer member (`n`, `e`, ...)."""
    return int.from_bytes(
        _b64url_decode(_field(key, name), 'the JWKS entry\'s {}'.format(name)), 'big')


def _check_claims(claims, issuer, audience, now):
    """`iss`, `aud`, `exp`, `nbf` and a subject, each its own named refusal.

    Ordered from the identity questions to the clock ones, so an operator reading a refusal learns
    the most specific true thing: a token for another deployment says so, rather than saying it
    expired.
    """
    subject = claims.get('sub')
    if not isinstance(subject, str) or not subject:
        raise IdentityRefused(
            'the ID token carries no sub: the subject reference is what gets stamped into every '
            'fact this actor writes, and there is nothing here to stamp - configure the IdP to '
            'issue a subject claim')
    if claims.get('iss') != issuer:
        raise IdentityRefused(
            'the ID token was issued by {!r}, not {!r}: a token from another issuer is another '
            'deployment\'s token however well it is signed - point the caller at this '
            'deployment\'s IdP, or correct the issuer this home was configured with'.format(
                claims.get('iss'), issuer))
    stated = claims.get('aud')
    audiences = [stated] if isinstance(stated, str) else stated
    if not isinstance(audiences, list) or not all(isinstance(one, str) for one in audiences):
        raise IdentityRefused(
            'the ID token\'s aud is {!r}, which is neither a string nor a list of them: the '
            'audience is who the token was minted FOR - the token is malformed, so re-acquire '
            'it'.format(stated))
    if audience not in audiences:
        raise IdentityRefused(
            'the ID token was minted for {}, not {!r}: a token addressed to another client is not '
            'a credential here even from the right issuer - request a token for this audience from '
            'the IdP'.format(', '.join(repr(one) for one in audiences) or 'no audience at all',
                             audience))
    _check_authorized_party(claims, audiences, audience)

    expiry = claims.get('exp')
    if not _finite(expiry):
        raise IdentityRefused(
            'the ID token has no usable exp ({!r}): a credential without an expiry never stops '
            'being one - configure the IdP to issue an expiry claim'.format(expiry))
    if now > expiry + LEEWAY_SECONDS:
        raise IdentityRefused(
            'the ID token expired {:.0f} s ago (exp {:.0f}, now {:.0f}, {:.0f} s of clock leeway '
            'allowed): acquire a fresh token from the IdP - the spine does not refresh '
            'credentials'.format(now - expiry, expiry, now, LEEWAY_SECONDS))
    not_before = claims.get('nbf')
    if not_before is not None:
        if not _finite(not_before):
            raise IdentityRefused(
                'the ID token\'s nbf is {!r}, which is not a time: the token is malformed, so '
                're-acquire it from the IdP'.format(not_before))
        if now < not_before - LEEWAY_SECONDS:
            raise IdentityRefused(
                'the ID token is not valid for another {:.0f} s (nbf {:.0f}, now {:.0f}, {:.0f} s '
                'of clock leeway allowed): the token was minted for later, or this box\'s clock is '
                'behind the IdP\'s - check the clock before retrying'.format(
                    not_before - now, not_before, now, LEEWAY_SECONDS))


def _check_authorized_party(claims, audiences, audience):
    """`azp`, which is who the token was minted FOR when `aud` names more than one client.

    Membership in `aud` is not enough on a multi-audience token, and OIDC Core 3.1.3.7 says so in
    two rules this implements as one. A token minted for another client of the same IdP, co-audienced
    to this one, is a token that client HOLDS - so accepting it lets any application the deployment's
    IdP will co-audience speak for an actor in this record, and the subject it speaks under is
    stamped into facts that outlive everyone in the room. So a multi-audience token must name its
    authorized party and that party must be us; and wherever `azp` is present at all it must be us,
    single audience or not, because a present `azp` naming somebody else is the IdP saying out loud
    who this was issued to.
    """
    party = claims.get('azp')
    if party is None:
        if len(audiences) > 1:
            raise IdentityRefused(
                'the ID token names {} audiences ({}) and no azp: a multi-audience ID token must '
                'say which client it was minted FOR, or membership in the list is all that stands '
                'between this deployment and any other client the IdP will co-audience - configure '
                'the IdP to issue azp, or to issue single-audience ID tokens'.format(
                    len(audiences), ', '.join(repr(one) for one in audiences)))
        return
    if not isinstance(party, str) or party != audience:
        raise IdentityRefused(
            'the ID token names azp {!r}, not {!r}: the authorized party is the client the IdP '
            'minted this token for, so a token issued to another client is that client\'s '
            'credential however many audiences it lists - request a token for this audience from '
            'the IdP'.format(party, audience))


def _finite(value):
    """A real point on the timeline. `True` is an int in Python and is not one of these."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _subject_reference(subject):
    """The subject reference a side-table verb was handed, or the refusal naming it."""
    if not isinstance(subject, str) or not subject:
        raise IdentityRefused(
            'the side table is keyed by subject reference and was handed {!r}: use the `subject` '
            'that verify_id_token answered - the same reference the log carries - rather than a '
            'display name or an email'.format(subject))
    return subject


def _write_names(home, table):
    """The side table, replaced atomically. Answers the table it wrote.

    tmp-then-`os.replace`, like every other write in this package: a reader sees the whole table or
    the previous one, never half of an erasure. The scratch file sits at the home's root beside the
    table, so nothing this function does can put a byte under `log/`.
    """
    home = Path(home)
    if not home.is_dir():
        raise IdentityRefused(
            'there is no home at {}: the side table lives beside a log, not on its own - mint the '
            'home first (init_home), then name its subjects'.format(home))
    scratch = home / ('.{}-{}.tmp'.format(NAMES_FILE, os.urandom(8).hex()))
    payload = json.dumps(table, sort_keys=True, indent=2, ensure_ascii=False).encode('utf-8')
    try:
        with open(str(scratch), 'wb') as handle:
            handle.write(payload + b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(scratch), str(home / NAMES_FILE))
    except BaseException:
        # Scratch is not the side table - nothing reads it and nothing names it - so clearing a
        # failed write is hygiene, and the table that was there is untouched.
        if scratch.exists():
            os.unlink(str(scratch))
        raise
    return table
