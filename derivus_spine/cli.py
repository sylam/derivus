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

"""`DV_Spine` - the home verbs, the identity verbs, and no third kind.

A home is a DIRECTORY, never a service: `log/` segments, `blobs/`, `keys/`. Nothing here holds
state, listens on anything, or repairs anything - `init` mints a home, `verify` re-derives every
hash in one from the bytes on disk, `checkpoint` signs the head, `status` reads it. Which home is
answered the way `DV_HOME` answers it one level over: `--home`, else `DV_SPINE_HOME`, else
`~/.derivus_spine`, resolved at the call rather than captured at import, so a script that moves
the variable moves the next verb with it.

Increment 2 adds five more, and they are the same kind of thing: `enroll` mints a seat's keypair,
`grant` declares a capabilities document, `rewrap` wraps the class key to whoever the document now
admits, `name` writes the mutable display-name side table, and `whoami` verifies an OIDC token
against a JWKS the deployment hands in as a FILE. That last one is the whole identity posture in a
sentence: identity is bought, not built, and nothing here fetches, listens or stores a secret. The
four home verbs keep their own spelling because each carries its own flags; the five identity verbs
register from one table, because they are uniform - a home, an actor where they append, and their
own two or three arguments.

`enroll` and `rewrap` reach `derivus_spine.custody` INSIDE their functions rather than at import.
The mouth must not be the reason the package fails to load on a machine that has not built custody
yet, and the same holds for `name` and `whoami` against `derivus_spine.identity`: a CLI that cannot
run `status` because an unrelated verb's module is missing is a CLI with a dependency it does not
have.

This module is a MOUTH, not a mechanism. Every verb is one call into the package's public
surface, and the answer is that call's own dict as JSON on stdout - what reads a spine CLI is a
script or an auditor, and both want the report rather than a paragraph about it. A refusal is the
library's own sentence on stderr and exit 1, verbatim: a CLI that rewords a refusal becomes a
second source of truth about what went wrong, which is the one thing this whole workstream exists
to forbid. Usage errors stay argparse's (exit 2) - they are about the command line, not the log.

Verification is local by construction here: `verify` reads files and recomputes, so it needs no
network, no writer and, in `--chain-only`, no key it is entitled to. That is the replica's posture
run from a terminal.

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
    """Which home a verb works on. Read on every call rather than captured at import, for the
    reason `dv_home` gives one level over: the home a script names and the home the environment
    names are the same setting answered at two different moments."""
    named = named or os.environ.get('DV_SPINE_HOME') or os.path.join('~', '.derivus_spine')
    return os.path.abspath(os.path.expanduser(named))


def local_actor():
    """Who genesis is attributed to when nobody said. Increment 2 buys identity; until it does,
    the local account name is the honest stand-in - stamped into the first events, never inferred
    afterwards - and `DV_SPINE_ACTOR` is how a deployment seats its own subject reference from
    event one instead of renaming history later."""
    try:
        return os.environ.get('DV_SPINE_ACTOR') or getpass.getuser()
    except (KeyError, OSError):
        # no account name to read (a bare container, a service seat): say so as the actor rather
        # than fail the mint, since the alternative is a home nobody can create
        return 'local'


def report(payload):
    """One JSON object per run, sorted and indented - the answer is a document a script parses
    and a human diffs, so it is stable under repetition rather than pretty."""
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write('\n')
    return 0


def do_init(args):
    """Mint a home. `HomeExists` is the refusal that makes this safe to retype."""
    return report(init_home(spine_home(args.home), args.actor or local_actor()))


def do_verify(args):
    """Re-derive the whole chain. `--chain-only` is the UNENTITLED posture - bodies stay sealed
    and the report says which mode it read in, because a verification that quietly skipped
    something is worse than one that refused."""
    return report(verify_home(spine_home(args.home), entitled=not args.chain_only))


def do_checkpoint(args):
    """Sign the head. An ordinary event through the ordinary writer - the signature is the point,
    the append is not special."""
    return report(write_checkpoint(SpineLog(spine_home(args.home))))


def do_status(args):
    """Where the log stands, in four fields and no fold. `bodies_readable` is the crypto-shred
    question asked plainly: a home whose firm key was destroyed still verifies its chain, and a
    terminal should be able to see which of the two it is holding."""
    home = spine_home(args.home)
    lsn, event_hash = SpineLog(home).head()
    return report({'home': home, 'head_lsn': lsn, 'head_hash': event_hash,
                   'bodies_readable': os.path.isfile(
                       os.path.join(home, 'keys', 'class_firm.key'))})


def read_json(path, what):
    """One JSON file, read as data. A file the operator names is the deployment handing something
    in - a policy document, a JWKS - and a file that is not the shape it claims is a refusal that
    names the path, because the path is what the operator can go and fix."""
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
    """Mint a seat. The keypair, the published public key and the `seat_enrolled` fact are custody's
    business; this verb is the mouth that names the subject and the actor doing the enrolling.

    `--public-key` is the posture, and its absence is the bootstrap. Hand in 32 raw bytes as hex and
    the seat generated its own keypair on its own machine, so the hub publishes a public key and
    stores no private one; omit it and the hub mints both and keeps the private half, which is the
    declared residual custody's header names and a file that should be handed over and shredded.
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
    """Declare a capabilities document - the policy-file editor the non-goals allow, and nothing
    more than that.

    The file is CANONICALISED before it is stored, which is the whole difference between an editor
    and a second source of truth: one policy is one blob however the operator spelled their JSON,
    so the same decision declared twice is the same hash and history stays addressable. The
    document is checked here too, so a malformed grant is met while the operator still has the file
    open rather than by a home that can no longer write.

    And the report says what the declaration OWES. A grant that adds a READ row creates key drift the
    moment it lands - the subject is entitled and the store holds no wrap for it - and the brief's
    "rewrap on grant change" is a property of the system or it is a line in a runbook nobody read.
    So the drift is computed against the document that is now in force and named here, where the
    operator is still at the keyboard, rather than discovered by an entitled seat whose body will not
    open. `rewrap` is what clears it, and this verb says how much of it there is.
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
    """Wrap the class key to whoever the document in force now admits. Idempotent by custody's own
    law - a second run emits nothing - so this is safe to put in a runbook after every grant, and
    `grant`'s own report names what it will find when it gets there."""
    from derivus_spine import custody
    log = SpineLog(spine_home(args.home))
    try:
        return report(custody.rewrap(log, args.actor))
    finally:
        log.close()


def do_name(args):
    """Read, set or erase a display name.

    The side table is MUTABLE and lives outside the log by design: the record is pseudonymous, so
    the erasable attributes are here, where an erasure touches no chain byte. With neither flag this
    reads rather than writes, which is what makes it safe to type first.
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
    """Verify an OIDC id token against a JWKS the deployment hands in as a FILE and print the
    subject reference.

    No fetching, no network, no token written anywhere: the JWKS is data, the answer is a
    pseudonymous reference, and a token that does not prove who it claims is `IdentityRefused` on
    stderr with exit 1 like every other refusal.
    """
    from derivus_spine import identity
    return report(identity.verify_id_token(
        args.token, read_json(args.jwks, 'JWKS'), args.issuer, args.audience))


#: flag -> how it is declared. Spelled once so that `--actor` means the same thing on every verb
#: that appends, and so the identity verbs stay a table rather than five near-copies of each other.
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

#: `(verb, help, arguments, runner)` for the identity verbs. A nested tuple of flags is a mutually
#: exclusive group - `name` either writes a display name or erases one, and asking for both is a
#: command-line mistake rather than a question about the record.
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
    """The argument table, and the only logic in this file.

    Built rather than held at module scope: a parser is cheap, and one constructed per call cannot
    accumulate state between the CLI's own gates and a runbook's two invocations in one process.
    """
    common = argparse.ArgumentParser(add_help=False)
    # every verb takes --home, and it hangs off the SUBcommands so that `DV_Spine init --home X`
    # reads the way anyone would type it
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
    """`DV_Spine` - dispatch one verb, print its answer, and translate a refusal into an exit
    code."""
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except SpineRefusal as refusal:
        # the library's own wording, unedited: it already names the thing and the remedy, and
        # anything added here would be a paraphrase competing with it
        sys.stderr.write('{}\n'.format(refusal))
        return 1


if __name__ == '__main__':
    sys.exit(main())
