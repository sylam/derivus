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

"""`DV_Spine` - the home verbs and the identity verbs.

A home is a directory, never a service: `log/` segments, `blobs/`, `keys/`. The home verbs are
`init` (mint one), `verify` (re-derive every hash from the bytes on disk), `checkpoint` (sign the
head) and `status` (read it). The identity verbs are `enroll` (mint a seat keypair), `grant`
(declare a capabilities document), `rewrap` (wrap the class key to whoever the document now
admits), `name` (the mutable display-name side table) and `whoami` (verify an OIDC token against a
JWKS the deployment hands in as a file). Nothing here fetches, listens or stores a secret.

Which home a verb works on: `--home`, else `DV_SPINE_HOME`, else `~/.derivus_spine`, resolved at
the call rather than captured at import. The four home verbs are spelled out individually since
each carries its own flags; the five identity verbs register from `IDENTITY_VERBS`.

`custody` and `identity` are imported inside the verbs that need them, so a missing module for one
verb cannot stop the CLI loading for the others.

This module is a mouth, not a mechanism: every verb is one call into the package's public surface,
and the answer is that call's own dict as JSON on stdout. A refusal is the library's own sentence
on stderr with exit 1, verbatim - rewording one would make the CLI a second source of truth. Usage
errors stay argparse's exit 2.

Run: `DV_Spine init` (`python -m derivus_spine.cli init` from a source tree).
"""
import argparse
import getpass
import json
import os
import sys

from derivus_spine import SpineLog, SpineRefusal, init_home, verify_home, write_checkpoint
from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document
from derivus_spine.errors import CapabilityDenied, CustodyRefusal

HOME_HELP = ('the spine home to work on; defaults to DV_SPINE_HOME, else ~/.derivus_spine')


def spine_home(named):
    """The absolute home path: `named`, else `DV_SPINE_HOME`, else `~/.derivus_spine`. Read on
    every call rather than captured at import."""
    named = named or os.environ.get('DV_SPINE_HOME') or os.path.join('~', '.derivus_spine')
    return os.path.abspath(os.path.expanduser(named))


def local_actor():
    """Who an append is attributed to when nobody said: `DV_SPINE_ACTOR`, else this account's
    name, else `'local'`."""
    try:
        return os.environ.get('DV_SPINE_ACTOR') or getpass.getuser()
    except (KeyError, OSError):
        # No account name to read (a bare container, a service seat): name that rather than fail
        # the mint, since the alternative is a home nobody can create.
        return 'local'


def report(payload):
    """Write `payload` to stdout as one JSON object, sorted and indented, and return exit code 0.
    Sorted so the output is stable under repetition and diffable."""
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write('\n')
    return 0


def do_init(args):
    """Mint a home and report what genesis wrote. A second run raises `HomeExists`."""
    return report(init_home(spine_home(args.home), args.actor or local_actor()))


def do_verify(args):
    """Re-derive the whole chain and report it. `--chain-only` runs the unentitled posture, leaving
    bodies sealed; the report names which mode it read in."""
    return report(verify_home(spine_home(args.home), entitled=not args.chain_only))


def do_checkpoint(args):
    """Append a signed checkpoint over the current head and report its envelope."""
    return report(write_checkpoint(SpineLog(spine_home(args.home))))


def do_status(args):
    """Report the head position in four fields and no fold. `bodies_readable` says whether the
    class key is present - a crypto-shredded home still verifies its chain."""
    home = spine_home(args.home)
    lsn, event_hash = SpineLog(home).head()
    return report({'home': home, 'head_lsn': lsn, 'head_hash': event_hash,
                   'bodies_readable': os.path.isfile(
                       os.path.join(home, 'keys', 'class_firm.key'))})


def read_json(path, what):
    """The JSON at `path`, read as data. Unreadable or non-JSON raises `CapabilityDenied` naming
    the path and `what` it was meant to be."""
    try:
        with open(path, 'rb') as handle:
            return json.loads(handle.read().decode('utf-8'))
    except (IOError, OSError) as missing:
        raise CapabilityDenied(
            '{} cannot be read as the {} ({}) - point --file or --jwks at the file the deployment '
            'is handing in'.format(path, what, missing))
    except (UnicodeDecodeError, ValueError) as broken:
        raise CapabilityDenied(
            '{} is not JSON, so it is not the {} ({}) - fix the file; nothing here guesses at what '
            'a malformed policy meant'.format(path, what, broken))


def do_enroll(args):
    """Mint a seat and report the enrollment.

    With `--public-key` (32 raw X25519 bytes as hex) the seat generated its own keypair, so the hub
    publishes a public key and stores no private one. Without it the hub mints both and keeps the
    private half - the bootstrap case, and a file to hand over and shred.
    """
    from derivus_spine import custody
    public_key = None
    if args.public_key is not None:
        try:
            public_key = bytes.fromhex(args.public_key)
        except (AttributeError, TypeError, ValueError):
            raise CustodyRefusal(
                '--public-key is {!r}, which is not hex: pass the seat\'s raw X25519 public key as '
                '64 hex characters - what `public_bytes(Encoding.Raw, PublicFormat.Raw)` answers on '
                'the machine that generated it'.format(args.public_key))
    log = SpineLog(spine_home(args.home))
    try:
        return report(custody.enroll(log, args.subject, args.actor, public_key=public_key))
    finally:
        log.close()


def do_grant(args):
    """Declare the capabilities document in `--file` and report the declaration plus the key drift
    it owes.

    The document is checked and canonicalised before it is stored, so one policy is one blob however
    the operator spelled their JSON. A grant that adds a read row leaves the subject entitled with no
    wrap in the store; that drift is computed against the document now in force and named here, and
    `rewrap` is what clears it.
    """
    from derivus_spine import custody
    document = read_json(args.file, 'capabilities document')
    raw = canonical_document(document, args.file)
    log = SpineLog(spine_home(args.home))
    try:
        blob = log.store.put(raw)
        envelope = log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
                              actor=args.actor, blob_refs=(blob,))
        drift = custody.wrap_drift(log)
    finally:
        log.close()
    return report(dict(envelope, policy=CAPABILITIES_POLICY, blob=blob,
                       entitled=drift['entitled'], wrapped=drift['current'],
                       rewrap_owed=[subject for subject, _ in drift['pending']],
                       unenrolled=drift['unenrolled'], unresolved=drift['unresolved']))


def do_rewrap(args):
    """Wrap the class key to every seat the document in force now admits, and report what was
    written. Idempotent: a second run emits nothing."""
    from derivus_spine import custody
    log = SpineLog(spine_home(args.home))
    try:
        return report(custody.rewrap(log, args.actor))
    finally:
        log.close()


def do_name(args):
    """Read, set or erase a display name, and report the name standing afterwards.

    The side table is mutable and lives outside the log, so an erasure touches no chain byte. With
    neither `--display` nor `--erase` this reads rather than writes.
    """
    from derivus_spine import identity
    home = spine_home(args.home)
    if args.erase:
        identity.erase_display_name(home, args.subject)
    elif args.display is not None:
        identity.set_display_name(home, args.subject, args.display)
    return report({'home': home, 'subject': args.subject,
                   'display': identity.display_names(home).get(args.subject)})


def do_whoami(args):
    """Verify an OIDC id token against a JWKS file and report the pseudonymous subject reference.

    No fetching and no token written anywhere: the JWKS is data the deployment hands in. A token
    that does not prove who it claims raises `IdentityRefused`.
    """
    from derivus_spine import identity
    return report(identity.verify_id_token(
        args.token, read_json(args.jwks, 'JWKS'), args.issuer, args.audience))


#: flag -> how it is declared, spelled once so `--actor` means the same thing on every verb that
#: appends.
ARGUMENTS = {
    'subject': {'help': 'the pseudonymous subject reference to name'},
    '--subject': {'required': True, 'help': 'the pseudonymous subject reference to enroll'},
    '--public-key': {'default': None,
                     'help': 'the seat\'s own raw X25519 public key as 64 hex characters; with it '
                             'the hub publishes a public key and stores no private one. Omitted, '
                             'the hub mints the pair and keeps the private half - the bootstrap '
                             'case, and a file to hand over and shred'},
    '--actor': {'required': True,
                'help': 'the subject reference this append is attributed to; it needs the verb '
                        'the event demands once a capabilities document is in force'},
    '--file': {'required': True,
               'help': 'the capabilities document to declare: {"grants": [...], "read": [...]}, '
                       'canonicalised on the way into the store'},
    '--display': {'default': None, 'help': 'the display name to write into the side table'},
    '--erase': {'action': 'store_true',
                'help': 'erase this subject\'s display name; the log is not touched'},
    '--token': {'required': True, 'help': 'the OIDC id token to verify'},
    '--jwks': {'required': True,
               'help': 'the JWKS file the deployment provides; nothing here fetches one'},
    '--issuer': {'required': True, 'help': 'the issuer the token must claim'},
    '--audience': {'required': True, 'help': 'the audience the token must carry'},
}

#: `(verb, help, arguments, runner)` for the identity verbs. A nested tuple of flags declares a
#: mutually exclusive group.
IDENTITY_VERBS = (
    ('enroll', 'mint a seat keypair, publish its public key and record the enrollment',
     ('--subject', '--actor', '--public-key'), do_enroll),
    ('grant', 'declare a capabilities document from a policy file',
     ('--file', '--actor'), do_grant),
    ('rewrap', 'wrap the class key to every entitled seat the document now admits',
     ('--actor',), do_rewrap),
    ('name', 'read, set or erase a display name in the side table outside the log',
     ('subject', ('--display', '--erase')), do_name),
    ('whoami', 'verify an OIDC id token against a JWKS file and print the subject reference',
     ('--token', '--jwks', '--issuer', '--audience'), do_whoami),
)


def build_parser():
    """The `DV_Spine` argument parser.

    Built per call rather than held at module scope, so two invocations in one process cannot
    accumulate state between them.
    """
    common = argparse.ArgumentParser(add_help=False)
    # --home hangs off the subcommands, so `DV_Spine init --home X` reads as typed.
    common.add_argument('--home', type=str, default=None, help=HOME_HELP)

    parser = argparse.ArgumentParser(
        prog='DV_Spine',
        description='Mint, verify and checkpoint a derivus spine home, and seat who may write '
                    'to it.')
    verbs = parser.add_subparsers(dest='verb', required=True)

    minted = verbs.add_parser('init', parents=[common],
                              help='mint a spine home: keys, genesis grants, the published '
                                   'verifying key and the first checkpoint')
    minted.add_argument('--actor', type=str, default=None,
                        help='the subject reference genesis is attributed to; defaults to '
                             'DV_SPINE_ACTOR, else this account\'s name')
    minted.set_defaults(run=do_init)

    checked = verbs.add_parser('verify', parents=[common],
                               help='re-derive every hash, linkage and checkpoint signature in '
                                    'the home from its own bytes')
    checked.add_argument('--chain-only', action='store_true',
                         help='verify as an unentitled replica: the chain over ciphertext, no '
                              'body opened, checkpoint authenticity reported as not assessed')
    checked.set_defaults(run=do_verify)

    verbs.add_parser('checkpoint', parents=[common],
                     help='append a signed checkpoint over the current head'
                     ).set_defaults(run=do_checkpoint)
    verbs.add_parser('status', parents=[common],
                     help='report the head position and whether this home can open its bodies'
                     ).set_defaults(run=do_status)

    for verb, help_text, arguments, runner in IDENTITY_VERBS:
        seated = verbs.add_parser(verb, parents=[common], help=help_text)
        for argument in arguments:
            if isinstance(argument, tuple):
                exclusive = seated.add_mutually_exclusive_group()
                for member in argument:
                    exclusive.add_argument(member, **ARGUMENTS[member])
            else:
                seated.add_argument(argument, **ARGUMENTS[argument])
        seated.set_defaults(run=runner)
    return parser


def main(argv=None):
    """Dispatch one `DV_Spine` verb, print its answer, and return the exit code: 0, or 1 for a
    refusal."""
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except SpineRefusal as refusal:
        # The library's own wording, unedited: it already names the thing and the remedy.
        sys.stderr.write('{}\n'.format(refusal))
        return 1


if __name__ == '__main__':
    sys.exit(main())
