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

"""The token verifier attacked with real keys, and the side table proved to be outside the chain.

Nothing here is a fixture string copied out of a JWT tutorial. Every gate MINTS an RSA-2048 pair
and a P-256 pair with `cryptography` inside the test, spells the JWKS by hand out of the public
numbers (`n`/`e`, `crv`/`x`/`y`, base64url with the padding stripped the way JOSE strips it), and
signs a genuine token over the real signing input - so a verifier that agreed with itself but not
with the wire format turns this file red. The forgeries are minted the same way, because a gate
that patches the module it is testing proves the patch works.

Three attacks earn their own gate. `alg: none` is the signature that is skipped rather than failed.
HS256-under-the-public-key is the alg-confusion forgery: the attacker HMACs the signing input with
the very key material the JWKS PUBLISHES, and a verifier that dispatches on the token's declared
algorithm checks the forgery against the attacker's own secret and says yes. And one flipped
character in the payload segment stands in for every edited claim there is - it must land on the
SIGNATURE, not on a claim the verifier reasoned about first, which is an assertion about the ORDER
of the checks rather than about any one of them.

The last gate is the erasure claim, asserted the only way it can be: a real home is minted, real
facts are appended, every file under `log/` is hashed, a name is set and then erased, and the
hashes are compared. The side table is erasable precisely because no hash covers it; if a single
segment byte moved, the display name was never outside the record and the design does not work.
Every clock in this file is INJECTED - `now` is a number the gate names - so nothing here can pass
in the morning and fail at midnight.
"""
import base64
import hashlib
import json
import os
import sys
from collections import namedtuple

import pytest
from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine.errors import IdentityRefused
from derivus_spine.genesis import init_home
from derivus_spine.identity import (
    display_names, erase_display_name, set_display_name, verify_id_token)
from derivus_spine.log import SpineLog

#: The deployment's IdP, its client, and one pseudonymous subject. The subject is a reference and
#: looks like one: the log never learns that this is a person with a name.
ISSUER = 'https://idp.desk.example/'
AUDIENCE = 'derivus-spine'
SUBJECT = '9f3c1a7e-4d21-4f0b-9a2f-6c5d8e0b1234'
ADMIN = 'c0ffee11-2233-4455-6677-8899aabbccdd'

#: A fixed instant. Every expiry question in this file is asked against this number, never against
#: the wall clock - a gate that reads the clock is a gate that fails on the wrong afternoon.
NOW = 1830000000.0
LIFETIME = 300.0

RSA_KID = 'idp-rsa-2026-08'
EC_KID = 'idp-ec-2026-08'
DECOY_KID = 'idp-rsa-2026-02'

Idp = namedtuple('Idp', 'rsa_key ec_key decoy jwks')


#: base64url's alphabet in index order, which is what makes "a character that decodes to the same
#: bytes" computable rather than guessed at.
B64URL_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'


def b64url(data):
    """base64url with the padding stripped, which is how JOSE spells every segment."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def flip(segment):
    """One character of `segment` rewritten, in the MIDDLE where every character is data.

    Never the last one. base64 packs three bytes into four characters, so a segment whose byte count
    is not a multiple of three ends in a character carrying padding bits that decode to nothing -
    rewrite that and the bytes can come back identical, which is a tamper gate that asserts nothing
    on some fraction of its runs. The middle is always a byte.
    """
    at = len(segment) // 2
    return segment[:at] + ('B' if segment[at] != 'B' else 'C') + segment[at + 1:]


def to_bytes(value):
    """A JWK's big-endian unsigned integer, in the minimum octets that hold it."""
    return value.to_bytes((value.bit_length() + 7) // 8 or 1, 'big')


def rsa_jwk(public_key, kid):
    """An RSA JWK spelled from the public numbers by hand - `kty`, `n`, `e`, and the labels."""
    numbers = public_key.public_numbers()
    return {'kty': 'RSA', 'kid': kid, 'alg': 'RS256', 'use': 'sig',
            'n': b64url(to_bytes(numbers.n)), 'e': b64url(to_bytes(numbers.e))}


def ec_jwk(public_key, kid):
    """A P-256 JWK by hand. The coordinates are FIXED-WIDTH 32 bytes, zero-padded on the left -
    the minimum-length spelling that works for `n` is wrong here, and the gate spells it right."""
    numbers = public_key.public_numbers()
    return {'kty': 'EC', 'kid': kid, 'crv': 'P-256', 'alg': 'ES256', 'use': 'sig',
            'x': b64url(numbers.x.to_bytes(32, 'big')), 'y': b64url(numbers.y.to_bytes(32, 'big'))}


def claims(**overrides):
    """A well-formed OIDC claim set at `NOW`, with whatever this gate wants to break in it."""
    body = {'iss': ISSUER, 'sub': SUBJECT, 'aud': AUDIENCE, 'exp': NOW + LIFETIME,
            'iat': NOW - 10.0, 'nbf': NOW - 10.0}
    body.update(overrides)
    return {key: value for key, value in body.items() if value is not None}


def signing_input(header, body):
    """The two segments and the ASCII bytes a signature is actually taken over."""
    head = b64url(json.dumps(header, sort_keys=True).encode('utf-8'))
    payload = b64url(json.dumps(body, sort_keys=True).encode('utf-8'))
    return head, payload, (head + '.' + payload).encode('ascii')


def rs256(private_key, body, kid=RSA_KID, header=None):
    """A real RS256 token: PKCS1v15 over SHA-256, exactly what the verifier must check."""
    head, payload, signed = signing_input(header or {'alg': 'RS256', 'typ': 'JWT', 'kid': kid},
                                          body)
    return '.'.join([head, payload,
                     b64url(private_key.sign(signed, padding.PKCS1v15(), hashes.SHA256()))])


def es256(private_key, body, kid=EC_KID, der=False):
    """A real ES256 token. `cryptography` signs DER, so the gate does the conversion the wire
    format demands - and `der=True` skips it, which is the trap this contract names."""
    head, payload, signed = signing_input({'alg': 'ES256', 'typ': 'JWT', 'kid': kid}, body)
    signature = private_key.sign(signed, ec.ECDSA(hashes.SHA256()))
    if not der:
        r, s = decode_dss_signature(signature)
        signature = r.to_bytes(32, 'big') + s.to_bytes(32, 'big')
    return '.'.join([head, payload, b64url(signature)])


@pytest.fixture(scope='module')
def idp():
    """The deployment's IdP: two live keys, one retired decoy, and the JWKS that publishes all
    three. Module-scoped because RSA-2048 keygen is the slowest thing in this file and the keys are
    never mutated - every forgery is built from them rather than in them."""
    signing_rsa = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    decoy_rsa = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signing_ec = ec.generate_private_key(ec.SECP256R1())
    jwks = {'keys': [rsa_jwk(decoy_rsa.public_key(), DECOY_KID),
                     rsa_jwk(signing_rsa.public_key(), RSA_KID),
                     ec_jwk(signing_ec.public_key(), EC_KID)]}
    return Idp(signing_rsa, signing_ec, decoy_rsa, jwks)


def verify(idp, token, **overrides):
    """The verifier under this file's fixed clock, issuer and audience unless a gate says otherwise.
    """
    call = {'issuer': ISSUER, 'audience': AUDIENCE, 'now': NOW}
    call.update(overrides)
    return verify_id_token(token, idp.jwks, **call)


def refusal(idp, token, **overrides):
    """The `IdentityRefused` a token earns, as a lowercase string to read assertions out of."""
    with pytest.raises(IdentityRefused) as raised:
        verify(idp, token, **overrides)
    return str(raised.value).lower()


def test_a_real_rs256_token_verifies_against_a_jwks_built_by_hand(idp):
    """The happy path, end to end and nothing stubbed: a token signed by an RSA-2048 key minted in
    this process, checked against a JWK spelled out of that key's public numbers. What comes back
    is the SUBJECT REFERENCE verbatim - the pseudonymous thing the log is allowed to carry - beside
    the issuer and the claims the caller decides with."""
    verified = verify(idp, rs256(idp.rsa_key, claims()))

    assert verified['subject'] == SUBJECT
    assert verified['issuer'] == ISSUER
    assert verified['claims']['aud'] == AUDIENCE
    # The subject is the token's `sub` verbatim - not a hash of it, not an email pulled out of a
    # profile claim - because a reference the IdP does not recognise is a reference nobody can
    # resolve when the record is read in 2056.
    assert verified['claims']['sub'] == verified['subject']


def test_a_real_es256_token_verifies_and_its_signature_is_the_raw_pair(idp):
    """The same claim for the EC half of the allowlist, and the encoding assertion beside it: the
    token this gate builds carries 64 raw bytes, so the verifier is provably doing the R||S to DER
    conversion rather than being handed something `cryptography` already understood."""
    token = es256(idp.ec_key, claims())

    signature = token.split('.')[2]
    assert len(base64.urlsafe_b64decode(signature + '=' * (-len(signature) % 4))) == 64
    assert verify(idp, token)['subject'] == SUBJECT

    # An audience list rather than a string is the other legal spelling, and membership is the test -
    # with the authorized party named, which is what a multi-audience token owes (see below).
    listed = es256(idp.ec_key, claims(aud=['some-other-client', AUDIENCE], azp=AUDIENCE))
    assert verify(idp, listed)['subject'] == SUBJECT


def test_an_expired_token_refuses_and_sixty_seconds_is_all_the_slack_there_is(idp):
    """Expiry against an INJECTED now. The leeway is clock skew between the IdP and this box, so it
    is asserted from both sides: a second past the sixty still verifies, and a second past that is
    a refusal naming the expiry - never a shrug, and never a refresh."""
    token = rs256(idp.rsa_key, claims())
    expiry = NOW + LIFETIME

    assert verify(idp, token, now=expiry + 59.0)['subject'] == SUBJECT
    said = refusal(idp, token, now=expiry + 61.0)
    assert 'expired' in said and 'exp' in said

    # nbf is the same rule read the other way: a token minted for later is not a credential yet.
    early = rs256(idp.rsa_key, claims(nbf=NOW + 600.0, exp=NOW + 900.0))
    assert 'nbf' in refusal(idp, early)
    # ...and a token with no expiry at all is a credential that never stops being one.
    assert 'exp' in refusal(idp, rs256(idp.rsa_key, claims(exp=None)))


def test_a_token_minted_for_another_audience_refuses(idp):
    """`aud` is who the token was minted FOR. A perfectly signed, perfectly fresh token addressed
    to another client of the same IdP is not a credential here - accepting one lets any application
    the IdP serves speak for an actor in this record."""
    said = refusal(idp, rs256(idp.rsa_key, claims(aud='some-other-client')))
    assert 'aud' in said or 'minted for' in said
    assert 'some-other-client' in said

    # A list that does not contain us is the same refusal: membership, never intersection-with-luck.
    assert 'minted for' in refusal(idp, rs256(idp.rsa_key, claims(aud=['a-portal', 'a-batch-job'])))


def test_a_co_audienced_token_belonging_to_another_client_refuses_on_azp(idp):
    """Membership in `aud` is not enough, and this is the attack it is not enough against.

    An IdP that co-audiences a token names both clients in `aud` and names the one it minted it FOR
    in `azp`. The token is that client's credential: they hold it, it is correctly signed, it is
    fresh, and this deployment is listed. Accepting it lets any application the IdP will co-audience
    speak for an actor in this record, under a subject reference that gets stamped into facts nobody
    will be able to correct in 2056. OIDC Core 3.1.3.7 says the client verifies `azp` where it is
    present and requires it where the audience is plural; both halves are asserted here, and so is
    the ordinary single-audience path staying green.
    """
    foreign = rs256(idp.rsa_key, claims(aud=['attacker-client', AUDIENCE], azp='attacker-client'))
    said = refusal(idp, foreign)
    assert 'azp' in said and 'attacker-client' in said

    # plural audience and NO azp: the token does not say who it was minted for, so it is refused
    silent = refusal(idp, rs256(idp.rsa_key, claims(aud=['attacker-client', AUDIENCE])))
    assert 'azp' in silent and 'multi-audience' in silent

    # a present azp is checked even where the audience is a single string - the IdP said out loud
    # which client this belongs to, and it is not this one
    assert 'azp' in refusal(idp, rs256(idp.rsa_key, claims(azp='attacker-client')))

    # and the two legal shapes still verify: one audience, and one audience with our own azp on it
    assert verify(idp, rs256(idp.rsa_key, claims()))['subject'] == SUBJECT
    assert verify(idp, rs256(idp.rsa_key, claims(azp=AUDIENCE)))['subject'] == SUBJECT
    assert verify(idp, rs256(idp.rsa_key, claims(
        aud=['a-portal', AUDIENCE], azp=AUDIENCE)))['subject'] == SUBJECT


def test_a_token_from_another_issuer_refuses(idp):
    """Issuer equality, and it is equality: a token from another deployment is another
    deployment's, however well it is signed. The gate flips it both ways - a foreign `iss` in the
    token, and this token checked against a home configured for a different IdP."""
    said = refusal(idp, rs256(idp.rsa_key, claims(iss='https://idp.attacker.example/')))
    assert 'idp.attacker.example' in said

    assert 'issue' in refusal(idp, rs256(idp.rsa_key, claims()),
                              issuer='https://idp.somewhere-else.example/')


def test_alg_none_refuses_by_name_before_a_key_is_ever_selected(idp):
    """The unsigned token. `none` is not a weak signature, it is the absence of one, and a verifier
    that dispatches on the declared algorithm without an allowlist will happily accept it. The
    refusal must name the algorithm - so an operator reads "this token was not signed" rather than
    "something did not verify"."""
    head, payload, _ = signing_input({'alg': 'none', 'typ': 'JWT'}, claims())

    # The classic shape: a header saying none and an empty signature segment.
    said = refusal(idp, head + '.' + payload + '.')
    assert 'none' in said and 'allowlist' in said
    # And with junk in the signature segment, in case emptiness was doing the refusing.
    assert 'none' in refusal(idp, '.'.join([head, payload, b64url(b'not a signature')]))


def test_hs256_signed_with_the_published_public_key_refuses_by_name(idp):
    """The alg-confusion forgery, minted for real. The attacker takes the RSA public key the JWKS
    PUBLISHES, uses its serialized bytes as an HMAC secret, and declares HS256; a verifier that
    looks up the key by kid and then asks "which algorithm did the token say?" checks the forgery
    against the attacker's own secret and admits them as this subject. The allowlist is what makes
    that unrepresentable, and it fires before any key is selected at all."""
    public_bytes = idp.rsa_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    head, payload, signed = signing_input({'alg': 'HS256', 'typ': 'JWT', 'kid': RSA_KID}, claims())
    forger = crypto_hmac.HMAC(public_bytes, hashes.SHA256())
    forger.update(signed)
    forged = '.'.join([head, payload, b64url(forger.finalize())])

    said = refusal(idp, forged)
    assert 'hs256' in said and 'allowlist' in said
    # The refusal is the ONLY outcome: nothing about the forgery resolved into a subject.
    with pytest.raises(IdentityRefused):
        verify(idp, forged)


def test_one_altered_payload_byte_lands_on_the_signature_not_on_a_claim(idp):
    """Tamper detection, and an assertion about ORDER. A segment has one character changed, which is
    every edited claim there is - a promoted subject, a stretched expiry. It must refuse on the
    SIGNATURE, because a verifier that parsed and judged the payload first has already reasoned
    about bytes nothing vouched for.

    The edited character is in the MIDDLE of the segment, and that is the whole care of this gate.
    base64 packs three bytes into four characters, so the LAST character of a segment carries the
    leftover data bits and then padding bits that decode to nothing: an RSA-2048 signature is 256
    bytes, 256 % 3 == 1, so its final character holds two significant bits and four that do not
    exist. Rewriting that character is a one-in-four chance of editing nothing at all - the same
    bytes, the same valid signature, and a tamper gate that quietly asserted nothing on those runs.
    A character in the middle is always a different byte.
    """
    head, payload, _ = signing_input({'alg': 'RS256', 'typ': 'JWT', 'kid': RSA_KID}, claims())
    token = rs256(idp.rsa_key, claims())

    said = refusal(idp, '.'.join([head, flip(payload), token.split('.')[2]]))
    assert 'signature' in said and 'verify' in said

    # The signature segment itself, one character over: same refusal, same reason.
    signature = token.split('.')[2]
    assert 'signature' in refusal(idp, '.'.join([head, payload, flip(signature)]))

    # A token signed by a key the deployment never published - a whole valid signature, wrong hand.
    assert 'signature' in refusal(idp, rs256(idp.decoy, claims(), kid=RSA_KID))


def test_a_segment_spelled_with_non_zero_padding_bits_is_not_this_token(idp):
    """The other half of the gate above, and the reason it can be written at all: JWS admits ONE
    spelling of a segment, and the verifier enforces it.

    The last character of the signature segment is rewritten to a different character that decodes
    to byte-identical signature bytes - four of the sixty-four do, because four of its six bits are
    padding. Under a decoder that tolerates them this token verifies and answers the same subject:
    one signed credential with four legal spellings, which is a token that can be rewritten in
    transit without breaking. It refuses instead, naming the canonical spelling.

    Kills the mutant that drops the re-encode check in `identity._b64url_decode` (returning the
    decoded bytes without requiring `urlsafe_b64encode(material).rstrip(b'=') == segment`), which is
    also the mutant under which the trailing-character edit this file used to make asserted nothing.
    """
    token = rs256(idp.rsa_key, claims())
    head, payload, signature = token.split('.')
    assert len(base64.urlsafe_b64decode(signature + '==')) == 256, 'RSA-2048 signs 256 bytes'

    # index % 16 == 0 is the canonical spelling of two data bits; +1 is a twin that decodes alike
    position = B64URL_ALPHABET.index(signature[-1])
    assert position % 16 == 0, 'the encoder left the padding bits set'
    twin = signature[:-1] + B64URL_ALPHABET[position + 1]
    assert base64.urlsafe_b64decode(twin + '==') == base64.urlsafe_b64decode(signature + '=='), \
        'the twin is meant to decode to the SAME bytes'

    assert 'canonical' in refusal(idp, '.'.join([head, payload, twin]))


def test_a_token_pasted_back_with_its_padding_restored_still_verifies(idp):
    """The padding tolerance the contract pins, gated rather than documented.

    JWS strips base64 padding; a copy-paste through a tool that re-encodes, and a few issuers,
    restore it. `=` is not in the base64url alphabet, so a verifier that checks the alphabet over the
    raw segment refuses a perfectly good token as re-encoded - fail-closed, but it misdiagnoses a
    paste as a corruption and the tolerance its own decoder documents can never run.

    Where the tolerance is REACHABLE is the point. The header and payload are the signing input, so
    padding them changes the bytes the IdP signed and no verifier can accept that - a fact about JWS
    rather than about this one. The signature segment is not signed over, and neither are the JWKS
    members, so those are where a padded paste arrives and those are what this asserts. An INTERIOR
    `=` stays the refusal it always was, which is what keeps this a tolerance rather than a hole.
    """
    token = rs256(idp.rsa_key, claims())
    head, payload, signature = token.split('.')
    padded = signature + '=' * (-len(signature) % 4)
    assert padded != signature, 'a 256-byte signature pads, and this one did not'

    assert verify(idp, '.'.join([head, payload, padded]))['subject'] == SUBJECT

    # the same tolerance on the deployment's own file: a JWKS whose members carry their padding
    entry = dict(rsa_jwk(idp.rsa_key.public_key(), RSA_KID))
    entry['n'] = entry['n'] + '=' * (-len(entry['n']) % 4)
    entry['e'] = entry['e'] + '=' * (-len(entry['e']) % 4)
    assert verify_id_token(token, {'keys': [entry]}, ISSUER, AUDIENCE, NOW)['subject'] == SUBJECT

    interior = head[:-1] + '=' + head[-1]
    assert 'base64url' in refusal(idp, '.'.join([interior, payload, signature]))


def test_a_kid_the_jwks_does_not_carry_refuses_and_a_kidless_token_tries_every_key(idp):
    """Key selection, both branches. A `kid` names one key and only that key, so a rotated-away id
    is a refusal naming it rather than a hopeful sweep over whatever else is published. A token
    with NO kid is the rotation case an IdP that omits the field leaves a verifier in: every key of
    the right type is tried, which is exactly why the decoy sits in this JWKS - the right key is
    the second RSA entry, so a verifier that gave up after the first is red here."""
    said = refusal(idp, rs256(idp.rsa_key, claims(), kid='idp-rsa-2027-01'))
    assert 'idp-rsa-2027-01' in said and RSA_KID in said

    kidless = rs256(idp.rsa_key, claims(), header={'alg': 'RS256', 'typ': 'JWT'})
    assert verify(idp, kidless)['subject'] == SUBJECT
    # And when none of them verifies, it is a signature refusal rather than a silent pass.
    assert 'signature' in refusal(
        idp, rs256(rsa.generate_private_key(public_exponent=65537, key_size=2048), claims(),
                   header={'alg': 'RS256', 'typ': 'JWT'}))

    # An UNUSABLE entry ahead of the good one does not end the sweep. The decoy above is a
    # well-formed key that simply does not verify, which proves only that a wrong key is skipped; a
    # JWKS entry this build cannot turn into a key at all - a member missing, a P-384 curve where
    # ES256 needs P-256 - is the ordinary shape of a real key set, and letting its refusal out of
    # the loop sinks every kidless token published behind it.
    good = rsa_jwk(idp.rsa_key.public_key(), RSA_KID)
    headless = {'kty': 'RSA', 'use': 'sig', 'n': good['n']}
    assert verify_id_token(kidless, {'keys': [headless, good]},
                           ISSUER, AUDIENCE, NOW)['subject'] == SUBJECT, \
        'the sweep stopped at a JWKS entry it could not turn into a key'

    # the same on the EC half, which is where an ordinary key set does this to you: a P-384 entry
    # with no `alg` sits in the candidate list beside the live P-256 key and cannot check anything.
    ec_kidless = es256(idp.ec_key, claims(), kid=None)
    off_curve = dict(ec_jwk(idp.ec_key.public_key(), 'idp-ec-p384'), crv='P-384')
    del off_curve['alg']
    assert verify_id_token(ec_kidless, {'keys': [off_curve, ec_jwk(idp.ec_key.public_key(), EC_KID)]},
                           ISSUER, AUDIENCE, NOW)['subject'] == SUBJECT, \
        'one unusable curve sank every kidless ES256 token behind it'

    # ...and where EVERY candidate is unusable, the refusal names the member rather than the
    # signature: that is the most specific true thing, and it is what a single named kid still gets.
    with pytest.raises(IdentityRefused) as raised:
        verify_id_token(kidless, {'keys': [headless]}, ISSUER, AUDIENCE, NOW)
    assert 'no e' in str(raised.value)

    # A kid that resolves to the wrong KIND of key cannot check this token either: the EC entry is
    # published, present, and useless for RS256.
    assert 'no key in the jwks' in refusal(idp, rs256(idp.rsa_key, claims(), kid=EC_KID))


def test_an_es256_signature_in_der_refuses_because_the_encoding_is_the_contract(idp):
    """The trap, gated. `cryptography` signs and verifies the DER SEQUENCE; JWS carries the raw
    fixed-width R||S pair. They are the same signature in two spellings, and a verifier that hands
    the wire bytes straight to the library rejects every legitimate token while one that hands them
    over unchecked would accept a DER blob no IdP will ever send. The length is the discriminator
    and the refusal says so."""
    der_token = es256(idp.ec_key, claims(), der=True)

    signature = der_token.split('.')[2]
    raw = base64.urlsafe_b64decode(signature + '=' * (-len(signature) % 4))
    assert raw[0] == 0x30 and len(raw) != 64, 'the gate did not produce a DER signature'

    said = refusal(idp, der_token)
    assert 'der' in said and '64' in said
    # The raw spelling of the SAME signature over the same claims verifies, which is what makes the
    # refusal above about the encoding rather than about the key.
    assert verify(idp, es256(idp.ec_key, claims()))['subject'] == SUBJECT


def test_a_token_that_is_not_a_compact_jws_refuses_before_anything_else(idp):
    """The parse. Everything that is not three base64url segments is refused by name and none of it
    reaches a key: a bearer header someone pasted whole, a 5-segment JWE, a decoded dict, an empty
    string. A verifier whose first act is a `split('.')[1]` is the one that crashes on these."""
    for bad in ('', 'Bearer ' + rs256(idp.rsa_key, claims()), 'a.b', 'a.b.c.d.e',
                rs256(idp.rsa_key, claims()) + '.extra'):
        with pytest.raises(IdentityRefused):
            verify(idp, bad)
    for not_a_token in (None, 42, {'alg': 'RS256'}, b'bytes are not the serialization'):
        with pytest.raises(IdentityRefused):
            verify(idp, not_a_token)

    # And a JWKS that is not a JWK Set is the deployment's file being wrong, said as much: this
    # module fetches nothing, so the remedy is always a file someone controls.
    with pytest.raises(IdentityRefused) as raised:
        verify_id_token(rs256(idp.rsa_key, claims()), {'k': []}, ISSUER, AUDIENCE, NOW)
    assert 'jwks' in str(raised.value).lower()


def test_a_token_with_no_subject_refuses_because_there_is_nothing_to_stamp(idp):
    """The subject reference is the whole product of this module - it is what gets stamped into
    every fact the actor writes and read back in 2056. A signed, fresh, correctly addressed token
    with no `sub` is a valid token and an unusable credential, and the refusal says which."""
    assert 'sub' in refusal(idp, rs256(idp.rsa_key, claims(sub=None)))
    assert 'sub' in refusal(idp, rs256(idp.rsa_key, claims(sub='')))


def test_the_side_table_round_trips_beside_the_log_and_never_inside_it(tmp_path):
    """The display-name table: mutable, unhashed, and at the home's ROOT. A read of a home that has
    no table answers empty and provisions nothing - the store's posture, for the same reason - and
    a write lands atomically with no scratch left behind."""
    home = tmp_path / 'spine'
    home.mkdir()

    assert display_names(home) == {}
    assert not (home / 'names.json').exists(), 'a read provisioned a side table'

    assert set_display_name(home, SUBJECT, 'Ada Lovelace') == {SUBJECT: 'Ada Lovelace'}
    set_display_name(home, ADMIN, 'The Custodian')
    assert display_names(home) == {SUBJECT: 'Ada Lovelace', ADMIN: 'The Custodian'}
    # Renaming is ordinary: this table is the one place in the system where editing is the point.
    assert set_display_name(home, SUBJECT, 'Ada, King of Diamonds')[SUBJECT] == \
        'Ada, King of Diamonds'

    assert (home / 'names.json').is_file()
    assert not (home / 'log').exists(), 'the side table conjured a log directory'
    assert [path.name for path in home.iterdir()] == ['names.json'], 'scratch survived a write'

    # It is a plain JSON object an operator can read and an editor can fix, not a frame.
    assert json.loads((home / 'names.json').read_text(encoding='utf-8')) == display_names(home)

    for bad in ('', None, 42):
        with pytest.raises(IdentityRefused):
            set_display_name(home, bad, 'a name')
        with pytest.raises(IdentityRefused):
            erase_display_name(home, bad)
    with pytest.raises(IdentityRefused):
        set_display_name(home, SUBJECT, '   ')
    # A home that is not there is a refusal naming it, never a directory conjured to hold a name.
    with pytest.raises(IdentityRefused):
        set_display_name(tmp_path / 'no-such-home', SUBJECT, 'Ada Lovelace')
    assert not (tmp_path / 'no-such-home').exists()

    # A doctored table is data on disk, and the refusal names the file rather than the caller.
    (home / 'names.json').write_bytes(b'{"a subject": ["not a name"]}')
    with pytest.raises(IdentityRefused):
        display_names(home)
    (home / 'names.json').write_bytes(b'not json at all')
    with pytest.raises(IdentityRefused):
        display_names(home)


def log_fingerprint(home):
    """Every file under `log/`, by relative path, hashed. This is the "not one byte moved" claim in
    the only form that can be asserted."""
    return {str(path.relative_to(home)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((home / 'log').rglob('*')) if path.is_file()}


def test_erasing_a_display_name_leaves_every_byte_of_the_chain_untouched(tmp_path):
    """The erasure claim, on a real home with real facts in it.

    This is why the side table exists. A display name inside the log would be either sealed - and
    so crypto-shreddable, which the design does have an answer for - or in the ENVELOPE, where it
    would be permanent, because erasing it means rewriting a hash chain and no erasure regime
    accepts that as an answer. So the name lives in a file no hash covers, and the proof is
    arithmetic: hash every segment before, erase, hash every segment after, compare. The subject
    REFERENCE is still in the log afterwards and must be - the fact did not go away, the name did.
    """
    home = tmp_path / 'spine'
    init_home(str(home), actor=ADMIN)
    log = SpineLog(home)
    try:
        log.append('determination', {'subject': 'the 09:00 EURUSD touch', 'ruling': 'touched'},
                   actor=SUBJECT, book='FX-DESK')
        log.append('status_transition', {'subject': 'ticket-4471', 'status': 'confirmed'},
                   actor=SUBJECT, book='FX-DESK')
    finally:
        log.close()

    before = log_fingerprint(home)
    assert before, 'the home minted no segments to compare'

    set_display_name(home, SUBJECT, 'Ada Lovelace')
    set_display_name(home, ADMIN, 'The Custodian')
    assert erase_display_name(home, SUBJECT) == {ADMIN: 'The Custodian'}
    # Erasing what is already gone is already true, so it is not an error and not a second write.
    assert erase_display_name(home, SUBJECT) == {ADMIN: 'The Custodian'}

    assert log_fingerprint(home) == before, 'an erasure moved a byte of the chain'
    assert (home / 'names.json').is_file()
    assert not (home / 'log' / 'names.json').exists(), 'the side table is inside the log'
    assert SUBJECT not in (home / 'names.json').read_text(encoding='utf-8')

    # The record is untouched in the sense that matters: the fact and its actor reference are still
    # there and still readable. Erasure removed an attribute, not a row.
    log = SpineLog(home)
    try:
        actors = {frame['actor'] for frame in log.frames()}
        assert log.head()[0] >= 6
    finally:
        log.close()
    assert SUBJECT in actors, 'the erasure took the subject reference with it'


def test_verification_writes_nothing_and_the_home_learns_only_the_name(tmp_path, idp):
    """Nothing secret is stored anywhere, ever. The verifier is handed no path and must leave none
    behind, so a home is fingerprinted whole across a verification; and when the subject is then
    NAMED, the only thing the home learns is the name - the token, its signature and its claim set
    appear in no file under the home, because a credential at rest is a credential someone can
    replay."""
    home = tmp_path / 'spine'
    init_home(str(home), actor=ADMIN)

    def whole_home():
        return {str(path.relative_to(home)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(home.rglob('*')) if path.is_file()}

    token = rs256(idp.rsa_key, claims())
    before = whole_home()
    verified = verify(idp, token)
    assert whole_home() == before, 'verifying a token wrote something to the home'

    set_display_name(home, verified['subject'], 'Ada Lovelace')
    written = {path: path.read_bytes() for path in home.rglob('*') if path.is_file()}
    for path, data in written.items():
        for secret in (token, token.split('.')[2], json.dumps(verified['claims'], sort_keys=True)):
            assert secret.encode('utf-8') not in data, '{} carries token material'.format(path)
    assert display_names(home) == {verified['subject']: 'Ada Lovelace'}
