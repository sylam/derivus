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

"""`DV_Spine` - the home verbs, the identity verbs and the policy verbs.

A home is a directory, never a service: `log/` segments, `blobs/`, `keys/`. The home verbs are
`init` (mint one), `verify` (re-derive every hash from the bytes on disk), `checkpoint` (sign the
head), `status` (read it), `follow` (pull a hub's frames into this home, the one verb that
speaks to a network and the one that only ever reads at the far end), `oracle` (answer the nine
invariants over what this home holds, against what a day's script says was asked) and `seed`
(mint the folds at an official close into a folder a reader starts from).

The identity verbs are `enroll` (mint a seat keypair), `grant` (declare a capabilities document),
`rewrap` (wrap the class key to whoever the document now admits), `name` (the mutable display-name
side table) and `whoami` (verify an OIDC token against a JWKS the deployment hands in as a file).
The policy verbs are `declare` (put any reserved policy document on the record from a JSON file)
and `policy` (report what is in force). Nothing here listens or stores a secret.

Which home a verb works on: `--home`, else `DV_SPINE_HOME`, else `~/.derivus_spine`, resolved at
the call rather than captured at import. The home verbs are spelled out individually since each
carries its own flags; the rest register from `IDENTITY_VERBS` and `POLICY_VERBS`.

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
import re
import sys

from derivus_spine import SpineLog, SpineRefusal, init_home, verify_home, write_checkpoint
from derivus_spine import policy as policies
from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document
from derivus_spine.errors import CapabilityDenied, CustodyRefusal, MalformedEvent

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


def do_follow(args):
    """Pull the hub's frames into this home until it stands where the hub does, and report each
    catch-up that took something.

    `--once` catches up to THE HEAD THE FIRST PULL REPORTED and stops - an operator's one-shot
    sync, and a cron's whole job, which against a desk in session is a hub that never stops
    writing. Without it the verb follows: the hub's doorbell where it rings one, a beat every
    `--interval` seconds where it does not or where the stream drops, and the same catch-up either
    way. `--blobs` pulls the bytes the frames cite, which wants a home that can open its bodies; a
    chain-only replica pulls frames alone. `--verify` re-derives the whole chain on every beat
    rather than the range that beat landed, which the first catch-up does regardless.
    """
    from derivus_spine import replica
    hub = replica.Hub(args.url, actor=args.actor)
    log = SpineLog(spine_home(args.home))
    try:
        if args.once:
            return report(replica.catch_up(log, hub, blobs=args.blobs, bounded=True, whole=True))
        beats = replica.beating(
            hub, replica.INTERVAL if args.interval is None else args.interval)
        for caught in replica.follow(log, hub, beats, blobs=args.blobs, whole=args.verify):
            if caught['frames']:
                report(caught)
    finally:
        log.close()
    return 0


def do_oracle(args):
    """Answer the nine invariants over this home and print the report; exit 1 where one did not
    hold.

    `--script` is what was ASKED - the acts a day put to the record - which two of the nine hold the
    record against; `--against` is a second copy to compare with. The diary one is not assessed
    here: it is a compile of the book rather than a fold of the record.
    """
    from derivus_spine import oracle
    answers = oracle.report(spine_home(args.home), against=args.against,
                            script=read_json(args.script, 'game script') if args.script else None)
    report(answers)
    return 1 if oracle.failed(answers) else 0


def do_seed(args):
    """Mint a seed for every projector but the strip at one official close, into `--out` (the
    home's own `seeds/` where none is named), and report what was written.

    The close is `--at`: an LSN, a day (`YYYY-MM-DD`, the last close true on or before it), or the
    last close there is where nothing is named. When this runs and where the folder is are the
    deployment's own - an end of day dropping its seeds where every seat reads them, or a seat
    standing at a day of its own choosing.
    """
    from derivus_spine import projections
    log = SpineLog(spine_home(args.home))
    try:
        close_lsn = named_close(log, args.at)
        seeds = [projections.seed_at(log, projections.PROJECTORS[name], close_lsn, args.out)
                 for name in sorted(projections.PROJECTORS) if name not in projections.UNSEEDED]
    finally:
        log.close()
    return report({'close_lsn': close_lsn,
                   'folder': os.path.abspath(args.out) if args.out
                   else os.path.join(spine_home(args.home), projections.SEEDS),
                   'seeds': [{'projector': seed['projector'], 'version': seed['version'],
                              'state_hash': seed['state_hash']} for seed in seeds]})


def named_close(log, at):
    """The close `at` names: an LSN as given, a `YYYY-MM-DD` day's last close, or the last close
    there is. `MalformedEvent` where it names none."""
    from derivus_spine import projections
    if at is not None and at.isdigit():
        return int(at)
    if at is not None and not re.match(r'^\d{4}-\d{2}-\d{2}$', at):
        raise MalformedEvent(
            '--at {!r} is neither an LSN nor a day: name the close as its position in the log, or '
            'as YYYY-MM-DD for the last close true on or before that day'.format(at))
    found = projections.close_on(log, at)
    if found is None:
        raise MalformedEvent(
            'this home holds no official close{} - a seed stands at a close, the one position '
            'where the desk already agrees what the day was'.format(
                '' if at is None else ' on or before {}'.format(at)))
    return found


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


def do_declare(args):
    """Declare a reserved policy document from a JSON file and report the declaration.

    The document is parsed and canonicalised on the way into the store, so one policy is one blob
    however the operator spelled their JSON and a document nobody could read back never lands. The
    capabilities document is `grant`'s: it is the one policy this module does not parse.
    """
    log = SpineLog(spine_home(args.home))
    try:
        return report(policies.declare(
            log, args.actor, args.policy,
            read_json(args.file, '{} policy document'.format(args.policy))))
    finally:
        log.close()


def do_policy(args):
    """Report the reserved policies in force, each with the blob it is stored under and the LSN of
    the declaration that put it there.

    All three come off ONE fold and so cannot disagree: `policy_declared` is open-bodied, and a
    declaration under a reserved name carrying no blob is a fact this reading steps over rather
    than a position it borrows. Named, it is one policy; unnamed, every reserved name - one nobody
    declared standing as nulls, so silence is never mistaken for absence.
    """
    if args.name is not None and args.name not in policies.PARSERS:
        raise MalformedEvent(
            '{!r} is not a policy this verb reads - it reads {}, and the capabilities document is '
            '`DV_Spine grant`\'s own file'.format(
                args.name, ', '.join(sorted(policies.PARSERS))))
    log = SpineLog(spine_home(args.home))
    try:
        return report(dict(
            (name, dict(zip(('blob', 'document', 'lsn'), policies.in_force(log, name))))
            for name in ([args.name] if args.name else sorted(policies.PARSERS))))
    finally:
        log.close()


def do_whoami(args):
    """Verify an OIDC id token against a JWKS file and report the pseudonymous subject reference.

    No fetching and no token written anywhere: the JWKS is data the deployment hands in. A token
    that does not prove who it claims raises `IdentityRefused`.
    """
    from derivus_spine import identity
    return report(identity.verify_id_token(
        args.token, read_json(args.jwks, 'JWKS'), args.issuer, args.audience))


#: argument -> how it is declared, spelled once so `--actor` means the same thing on every verb
#: that appends. A name carrying no dashes is a positional.
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
    'policy': {'help': 'the reserved policy this document is declared under: one of {}'.format(
        ', '.join(sorted(policies.PARSERS)))},
    'file': {'help': 'the JSON policy document, parsed and canonicalised into the store'},
    'name': {'nargs': '?', 'default': None,
             'help': 'the reserved policy to report; every one of them where no name is given'},
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

#: The same four-tuple for the policy verbs - the policy-file editor this deployment has, in its
#: CLI form: one verb that declares a document and one that reports what is standing.
POLICY_VERBS = (
    ('declare', 'declare a reserved policy document from a JSON file',
     ('policy', 'file', '--actor'), do_declare),
    ('policy', 'report the reserved policies in force with the blob and LSN each stands at',
     ('name',), do_policy),
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

    pulled = verbs.add_parser('follow', parents=[common],
                              help='pull a hub\'s frames into this home and keep pulling: a '
                                   'replica writes what the hub chained and submits nothing')
    pulled.add_argument('url', type=str, help='where the hub serves /spine/frames')
    pulled.add_argument('--once', action='store_true',
                        help='catch up to the head the first pull reports and stop, rather than '
                             'following a hub that may never stop writing')
    pulled.add_argument('--verify', action='store_true',
                        help='re-derive the whole chain on every beat rather than the range that '
                             'beat landed; the first catch-up does it regardless')
    pulled.add_argument('--interval', type=float, default=None,
                        help='seconds between beats where the hub rings no doorbell, or where the '
                             'stream drops')
    pulled.add_argument('--blobs', action='store_true',
                        help='pull the bytes the frames cite too; wants a home that can open its '
                             'bodies, a chain-only replica needing none of them')
    pulled.add_argument('--actor', type=str, default=None,
                        help='the subject reference a blob read is served under; a chain-only '
                             'follower needs none')
    pulled.set_defaults(run=do_follow)

    asked = verbs.add_parser('oracle', parents=[common],
                             help='answer the nine invariants over this home and say which of '
                                  'them a copy standing here could not assess')
    asked.add_argument('--script', type=str, default=None,
                       help='the JSON record of what a day ASKED, which the refusals and the '
                            'attestation lanes are held against; without it those two are not '
                            'assessed')
    asked.add_argument('--against', type=str, default=None,
                       help='a second copy of this record to compare heads and folds with at the '
                            'shallower of the two heads')
    asked.set_defaults(run=do_oracle)

    minted_seeds = verbs.add_parser('seed', parents=[common],
                                    help='mint the folds at one official close into a folder, '
                                         'so a reader starts at that close instead of genesis')
    minted_seeds.add_argument('--at', type=str, default=None,
                              help='the close: an LSN, or YYYY-MM-DD for the last close true on '
                                   'or before that day; the last close there is where omitted')
    minted_seeds.add_argument('--out', type=str, default=None,
                              help='the folder to file the seeds in - one every seat reads, or a '
                                   'seat\'s own; the home\'s seeds/ where omitted')
    minted_seeds.set_defaults(run=do_seed)

    for verb, help_text, arguments, runner in IDENTITY_VERBS + POLICY_VERBS:
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
