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

"""Verifying who an actor is, and the one mutable place a human name lives.

The deployment runs its own IdP; this module checks the token that IdP issued before an actor
reference is stamped into a fact. There is no HTTP here, no discovery document, no token
acquisition and no JWKS fetching - the key set arrives as a `dict` the deployment hands in as data,
so the import surface stays stdlib plus `cryptography` and a verifier works with no network.

The algorithm allowlist is the security posture: RS256 and ES256, nothing else. `alg: none` and
every HMAC family are refused by name before a key is ever selected, which is what makes the
alg-confusion forgery - a token declaring HS256 and "signed" with the RSA public key the JWKS
publishes - unrepresentable rather than merely unlikely.

Order is part of the contract: split the segments, read the header, allowlist the algorithm, select
a key by `kid`, verify the signature, and only then parse the payload and read its claims. Nothing
here decides on unverified bytes. One encoding trap is stated because it half-works silently: a JWS
ES256 signature is the raw fixed-width pair `R || S` (64 bytes, RFC 7518 3.4) while `cryptography`
verifies the DER SEQUENCE OpenSSL emits, so a DER signature is refused on its length rather than
converted.

Verification returns the subject reference - the token's `sub`, verbatim - and writes nothing to
disk. Display names are the counterpart: they live in `<home>/names.json`, a mutable unhashed side
table beside the log and never inside it, so an erasure is a dict key going away rather than a
chain to rewrite. A name in an envelope would be permanent, which erasure regimes do not accept.
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
#: and no third: a symmetric alg would mean the verifier holds the signing secret.
ALGORITHMS = {'RS256': 'RSA', 'ES256': 'EC'}
#: The only curve ES256 is defined over.
ES256_CURVE = 'P-256'
#: RFC 7518 3.4: the ES256 signature is `R || S`, each coordinate a fixed 32 bytes big-endian.
ES256_SIGNATURE_BYTES = 64
P256_COORDINATE_BYTES = 32
#: Clock skew allowed between the IdP and this box, in seconds. Slack rather than policy.
LEEWAY_SECONDS = 60.0
#: The display-name side table, at the home's root. Never under `log/`, which is what a replica
#: file-copies and what the chain covers; an erasable attribute may live in neither.
NAMES_FILE = 'names.json'
#: base64url's alphabet, checked before decoding so a segment carrying `+` or `/` is a named
#: refusal rather than something Python's decoder tolerates.
B64URL_ALPHABET = frozenset(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_')
#: The padding character, which is not in the alphabet and is read only where padding may be.
B64URL_PAD = '='


def verify_id_token(token: str, jwks: dict, issuer: str, audience: str, now: float = None) -> dict:
    """Verify an OIDC ID token against `jwks` and return `{subject, issuer, claims}`.

    The sequence is the security property: parse, allowlist the algorithm, select the key, verify,
    and only then read a claim. `now` is seconds since the epoch and defaults to the wall clock.
    Every failure raises `IdentityRefused`.
    """
    header, segments = _parse(token)
    algorithm = _allowed_algorithm(header.get('alg'))
    signature = _b64url_decode(segments[2], 'the JWS signature')
    signing_input = '.'.join(segments[:2]).encode('ascii')

    _verify_signature(_candidates(jwks, header, algorithm), algorithm, signing_input, signature)

    claims = _json_object(_b64url_decode(segments[1], 'the JWS payload'), 'the JWS payload')
    _check_claims(claims, issuer, audience, time.time() if now is None else now)
    # The record takes `subject` alone; the caller gets the claims to decide with.
    return {'subject': claims['sub'], 'issuer': claims['iss'], 'claims': dict(claims)}


def display_names(home) -> dict:
    """The `subject -> display name` side table at `home`, or `{}` where there is none.

    A read never provisions. A table that is present but not an object of string to string raises
    `IdentityRefused`.
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
    """Name `subject` in the home's side table and return the whole table as it now stands.

    `subject` is the reference the log carries; `name` must be a non-empty string - use
    `erase_display_name` to remove an entry.
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
    """Forget `subject`'s display name and return the whole table as it now stands.

    The only erasure path: a dict key going away in a file no hash covers, so no byte of the log,
    the chain or the blob store moves. Erasing a name that is not there is not an error.
    """
    table = display_names(home)
    table.pop(_subject_reference(subject), None)
    return _write_names(home, table)


def _parse(token):
    """`(header, segments)` for a compact JWS. Only the header is parsed - the payload stays raw
    text until a signature has vouched for it."""
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
    """Assert `segment` is base64url, `=` tolerated as trailing padding and nowhere else.

    Emptiness passes here and is refused at the decode, so an unsigned token is refused for its
    algorithm rather than for its missing bytes.
    """
    if not isinstance(segment, str) or not B64URL_ALPHABET.issuperset(segment.rstrip(B64URL_PAD)):
        raise IdentityRefused(
            '{} is not base64url: the segment carries characters outside [A-Za-z0-9_-], with `=` '
            'read only as trailing padding - the token has been re-encoded or corrupted in '
            'transport, so re-acquire it from the IdP'.format(what))


def _b64url_decode(segment, what):
    """The bytes behind one base64url segment, trailing padding tolerated.

    The encoding is required to be canonical - the trailing character's unused bits zero - by
    re-encoding the result and requiring it to match. Otherwise several spellings of one signed
    token would decode alike, and a flipped final character would go undetected.
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
    """`raw` parsed as a UTF-8 JSON object, raising `IdentityRefused` naming `what` if it is not
    one."""
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
    """`alg` itself once it is on the allowlist, otherwise `IdentityRefused`.

    Runs before any key is selected: `none` would skip the signature check rather than fail it, and
    an HMAC alg would check a forgery against the public key the JWKS publishes.
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
    """The JWKS entries this token may be checked against.

    A token carrying a `kid` is checked against that key alone. One without is checked against every
    key of the right type - the rotation case an IdP omitting `kid` leaves a verifier in - narrowed
    by key type, declared `alg` and `use`.
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
    """Check `signature` over `signing_input` under each candidate, returning the key that vouches
    for it and raising `IdentityRefused` if none does.

    An unusable candidate - a P-384 entry beside the live P-256 one, a JWK missing a member - does
    not end the sweep: those refusals are collected and one is re-raised only if every candidate was
    unusable, so a malformed entry cannot sink the kidless tokens behind it.
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

    RFC 7518 3.4 defines the ES256 signature as two fixed-width 32-byte integers concatenated,
    while OpenSSL-shaped libraries verify the DER encoding (70-71 bytes for P-256). A signature of
    any other length is refused rather than converted.
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
    """One JWK as a `cryptography` public key - `n`/`e` for RS256, `crv`/`x`/`y` for ES256 - or
    `IdentityRefused` naming the member that was not a key."""
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
    """The required JWK member `name` as a non-empty string, or `IdentityRefused` naming it."""
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
    """Check `sub`, `iss`, `aud`, `azp`, `exp` and `nbf`, each with its own named refusal.

    Ordered identity questions first and clock questions second, so a token belonging to another
    deployment says so rather than reporting that it expired.
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
    """Check `azp` - who the token was minted for - per OIDC Core 3.1.3.7.

    Membership in `aud` is not enough on a multi-audience token, which must name an authorized
    party equal to `audience`. Wherever `azp` is present at all it must equal `audience`, single
    audience or not: a present `azp` naming another client is the IdP saying who holds this token.
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
    """Whether `value` is a finite point on the timeline. `True` is an int in Python and is not
    one."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _subject_reference(subject):
    """`subject` asserted to be a non-empty subject reference, the one the log carries."""
    if not isinstance(subject, str) or not subject:
        raise IdentityRefused(
            'the side table is keyed by subject reference and was handed {!r}: use the `subject` '
            'that verify_id_token answered - the same reference the log carries - rather than a '
            'display name or an email'.format(subject))
    return subject


def _write_names(home, table):
    """Replace the side table with `table` atomically and return it.

    tmp-then-`os.replace`, so a reader sees the whole table or the previous one, never half of an
    erasure. The scratch file sits at the home's root, so nothing here writes under `log/`.
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
        # Nothing reads or names the scratch file, and the table that was there is untouched.
        if scratch.exists():
            os.unlink(str(scratch))
        raise
    return table
