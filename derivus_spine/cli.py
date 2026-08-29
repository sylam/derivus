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

"""`DV_Spine` - four verbs over one spine home, and no fifth.

A home is a DIRECTORY, never a service: `log/` segments, `blobs/`, `keys/`. Nothing here holds
state, listens on anything, or repairs anything - `init` mints a home, `verify` re-derives every
hash in one from the bytes on disk, `checkpoint` signs the head, `status` reads it. Which home is
answered the way `DV_HOME` answers it one level over: `--home`, else `DV_SPINE_HOME`, else
`~/.derivus_spine`, resolved at the call rather than captured at import, so a script that moves
the variable moves the next verb with it.

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


def main(argv=None):
    """`DV_Spine` - dispatch one verb, print its answer, and translate a refusal into an exit
    code. The only logic in this file is the argument table."""
    common = argparse.ArgumentParser(add_help=False)
    # every verb takes --home, and it hangs off the SUBcommands so that `DV_Spine init --home X`
    # reads the way anyone would type it
    common.add_argument('--home', type=str, default=None, help=HOME_HELP)

    parser = argparse.ArgumentParser(
        prog='DV_Spine', description='Mint, verify and checkpoint a derivus spine home.')
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

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except SpineRefusal as refusal:
        # the library's own wording, unedited: it already names the thing and the remedy, and
        # anything added here would be a paraphrase competing with it
        sys.stderr.write('{}\n'.format(refusal))
        return 1


if __name__ == '__main__':
    sys.exit(main())
