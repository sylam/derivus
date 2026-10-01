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

"""The one seam between the engine and the book of record - lazy and off by default - and the desk's
readings over it: the diary keyed to the record, the settlement file, the marks and the P&L (`PnL`)
and the collateral calls (`Collateral`).

`derivus_spine` is the append-only log the desk's numbers are recorded in, importing stdlib and
`cryptography` and never the engine, so a book of record can be verified on a machine that cannot
price a trade. This module is the seam from the other side and the only place under `derivus/` that
knows the spine exists. Five rules shape it.

The engine stays a pure function from facts to numbers, so every verb below is a delegator: it
canonicalises through the engine's own encoder, hands the spine plain bytes and numbers, and holds
no storage logic. `pin_result` hands the spine a callable rather than letting it reach for a pricer.

The spine is an extra, so the import is lazy and inside the function - `import derivus` must not
need `cryptography` any more than it needs `fastapi`. A tree without it refuses by name with the
install line.

No spine home configured means bit-identical behaviour. `DV_SPINE_HOME` is the whole switch, read on
every call, and it does not fall back to the CLI's `~/.derivus_spine` default - a box that once ran
`DV_Spine init` must not silently start recording. Unset, every call site here is a no-op; set but
not minted is a named refusal rather than a quiet fall-back.

Every event carries a pseudonymous subject reference, so the actor is the seat a request's bearer
token proves where the deployment checks tokens, else what the caller names - never the writer's
reserved one - and, on a record declaring no capabilities document, `DV_SPINE_ACTOR` where the
caller named none; under a document an unnamed act is refused, since inventing a seat would put a
name in the record nobody chose. `hub()` is the deployment's own seat, which the poll paths and a
read's own compile run under: a requester's seat narrows what a read answers and nothing else.

One writer: the spine claims its home exclusively at the first append, so `writing()` holds a
process-wide lock for the length of one act and closes the handle after it. And every `SpineRefusal`
is re-raised as `SpineRefused`, a `ValueError` carrying the spine's sentence unedited, so the book
verbs' `except ValueError -> 422` handlers surface the library's own wording.
"""
import contextlib
import contextvars
import hashlib
import json
import numbers
import os
import threading
from copy import deepcopy

from . import Context, content_hash, instruments
from ._version import __version__
from .calculation import Base_Revaluation, Diary, construct_calculation
from .schema import (OBSERVES, declared_fields, index_named, instrument_of, job_children, mapping,
                     tables_of, walk_job_deals, without_clocks)
from .config import Config, CustomJsonEncoder, as_json
from .structures import timestamp

#: The whole switch, the actor beside it, and the folder a reader's folds start from. Read per
#: call, like `DV_HOME` one module over.
SPINE_HOME = 'DV_SPINE_HOME'
SPINE_ACTOR = 'DV_SPINE_ACTOR'
SPINE_SEEDS = 'DV_SPINE_SEEDS'
#: Where a deployment checks who is asking: a JWKS file, and the issuer and audience its tokens
#: name. Unset, nothing is checked and a request's seat is what it says.
SPINE_JWKS, SPINE_ISSUER, SPINE_AUDIENCE = 'DV_SPINE_JWKS', 'DV_SPINE_ISSUER', 'DV_SPINE_AUDIENCE'

#: The subject a request's verified bearer token names, for that request alone: None on a box
#: checking no token, and on every thread no request started - the metronome, the compute worker.
REQUESTER = contextvars.ContextVar('requester', default=None)

#: The three attestation lanes, respelled here so a call site can name one without paying for the
#: spine import. The refusal wording still comes from `derivus_spine.verbs.check_lane`.
TELEMETRY = 'telemetry'
CURIOSITY = 'curiosity'
STANDING = 'standing'
LANES = (TELEMETRY, CURIOSITY, STANDING)

#: What the standing lane MINTS, respelled beside the lanes for the same reason: a standing job the
#: queue turns away is recorded as a refusal of this type.
STANDING_TYPE = 'run_completed'

#: The processes a market is designated for - `derivus_spine.policy.DESIGNATED_PROCESSES`,
#: respelled here so the service can name them without paying for the spine import: the one a
#: settlement file is struck under, and the one the book is marked under at each close.
SETTLEMENT_EXPORT = 'settlement_export'
PNL = 'pnl'

#: Where a diary row stands once the record answers it, stamped by whoever holds the log.
SETTLED = 'settled'

#: What a tree without the extra is told, with the line that fixes it.
NO_PACKAGE = ('the book of record is not installed on this box ({}) - {} names a spine home, so '
              'this verb records to it; `pip install derivus[enterprise]`, or unset {} to run the '
              'edge exactly as it ran before')
#: What a configured home with no actor is told. An unconfigured home says nothing at all: that is
#: the default posture rather than an error.
NO_ACTOR = ('no actor for this append: every event carries the pseudonymous subject reference that '
            'submitted it, so name one on the request or set {} - a record that invented an actor '
            'would be a record naming somebody who never spoke')
#: What an act naming no seat is told once a capabilities document is in force.
UNNAMED = ('no actor for this act, and a capabilities document is in force here: every act is '
           'admitted and filed under the seat the request names, so name one - the deployment\'s '
           'own seat, {}, is the one the poll paths no request signs run under')
#: What an act naming the writer's reserved actor is told.
RESERVED = ('{!r} is the record\'s own voice and never a seat: the writer alone files under it - '
            'name the seat that is acting')
#: What an act naming another seat than its bearer token is told.
NOT_THE_BEARER = ('this request names the seat {!r} and its bearer token is {!r}\'s: a request '
                  'acts as the seat its token proves - drop the name, or sign in as that seat')
#: What a request carrying no bearer token is told where the deployment checks them.
NO_BEARER = ('this service checks who is asking and the request carries no `Authorization: '
             'Bearer` token - send the ID token the deployment\'s identity provider issued')
#: What a deployment naming a key set that does not read is told.
NO_KEYSET = ('{} names {}, which does not read as a key set ({}): every request\'s token is '
             'checked against it, so nothing is served without one - point it at the JWKS file '
             'the identity provider publishes, or unset it to check nothing')

#: One writer, one act at a time. The spine answers `WriterBusy` to a second holder of the home;
#: this lock keeps a request thread and the compute worker from meeting that over their own book.
_WRITER = threading.Lock()

#: The key set as last read: `[path, mtime, document]`, so a request stats the file and a rotated
#: set is read again rather than every request reading it.
_KEYSET = [None, None, None]

#: The folds a BOOKING advances rather than re-walking, as the `(lsn, state)` pair `projections.fold`
#: takes: projector name -> that pair, under the one HISTORY they were folded on. A box serves one
#: record, so a history that changes drops the lot rather than growing a map nobody evicts.
#: Assignment is the only write and the fold copies before it advances, so a second thread costs at
#: worst a pair one position older.
_ADVANCED, _ADVANCED_HOME = {}, None


class SpineRefused(ValueError):
    """The book of record declining, in the engine's own exception vocabulary.

    A `ValueError` on purpose: the book verbs already map one to a 422 carrying its message
    verbatim, so a refusal reaches a desk in the spine's own wording, and catching it needs no
    import of the spine's exception tree.
    """


def home():
    """The configured spine home, or None where the deployment configured none.

    No default: `DV_Spine` falls back to `~/.derivus_spine` because a person typing a verb means
    that home, but an engine that fell back would record on any box where somebody once ran `init`.
    """
    named = os.environ.get(SPINE_HOME)
    return os.path.abspath(os.path.expanduser(named)) if named else None


def configured():
    """Whether this box records - the question every call site asks first. False means the edge
    behaves exactly as it did before this module existed."""
    return home() is not None


def package():
    """`derivus_spine`, imported here and now, or `SpineRefused` naming the install line.

    Each submodule read below is imported by name because a submodule is only an attribute of its
    package once something has imported it, and the package's own surface stays the truth layer.
    """
    try:
        import derivus_spine.capability
        import derivus_spine.firmness
        import derivus_spine.oracle
        import derivus_spine.policy
        import derivus_spine.projections
        import derivus_spine.tiers
        import derivus_spine.verbs
        import derivus_spine.vocabulary
    except ImportError as absent:
        raise SpineRefused(NO_PACKAGE.format(absent, SPINE_HOME, SPINE_HOME))
    return derivus_spine


def actor(named=None, log=None):
    """Who this act is attributed to: the seat a request's token proves, a different name refused;
    else the seat the caller named - else, where no capabilities document is in force,
    `DV_SPINE_ACTOR` - and never the writer's reserved name, whoever says it. `log` is a handle
    already open on the home, whose state says what is in force."""
    spine = package()
    signed = REQUESTER.get()
    if signed is not None and named and named != signed:
        raise SpineRefused(NOT_THE_BEARER.format(named, signed))
    named = named or signed
    if not named and (folded(spine.capability.state_at) if log is None
                      else log.capabilities())[0] is not None:
        raise SpineRefused(UNNAMED.format(SPINE_ACTOR))
    subject = named or hub()
    if subject is None:
        raise SpineRefused(NO_ACTOR.format(SPINE_ACTOR))
    if subject == spine.vocabulary.WRITER:
        raise SpineRefused(RESERVED.format(subject))
    return subject


def hub():
    """The deployment's own seat, `DV_SPINE_ACTOR`, or None where it names none - what the jobs no
    request signs run under: the diary read, the tick and the securities verification."""
    return os.environ.get(SPINE_ACTOR) or None


@contextlib.contextmanager
def deployed():
    """The deployment's own work on a request's thread, for the length of the block: the requester
    set aside, so the seat is `hub()` exactly as on a thread no request started."""
    held = REQUESTER.set(None)
    try:
        yield hub()
    finally:
        REQUESTER.reset(held)


def keyset():
    """The key set `DV_SPINE_JWKS` names, read once per version of its file, or None where the
    deployment checks no token. One that does not read refuses by name, which the service asks
    before it serves anything."""
    keys = os.environ.get(SPINE_JWKS)
    if not keys:
        return None
    try:
        stamp = os.stat(keys).st_mtime_ns
        if _KEYSET[:2] != [keys, stamp]:
            with open(keys, encoding='utf-8') as handle:
                _KEYSET[:] = [keys, stamp, json.load(handle)]
    except (OSError, ValueError) as unreadable:
        raise SpineRefused(NO_KEYSET.format(SPINE_JWKS, keys, unreadable))
    return _KEYSET[2]


def bearer(authorization):
    """The subject the `Authorization: Bearer` header value proves, verified against the key set
    `DV_SPINE_JWKS` names for `DV_SPINE_ISSUER` and `DV_SPINE_AUDIENCE` - or None where no key set
    is configured, a box checking no token. A missing token refuses by name and a bad one in the
    verifier's own words; verifying is local, the key set being data the deployment hands in."""
    published = keyset()
    if published is None:
        return None
    scheme, _, token = (authorization or '').partition(' ')
    if scheme.lower() != 'bearer' or not token.strip():
        raise SpineRefused(NO_BEARER)
    from derivus_spine.identity import verify_id_token

    with translating():
        return verify_id_token(token.strip(), published, os.environ.get(SPINE_ISSUER),
                               os.environ.get(SPINE_AUDIENCE))['subject']


def where():
    """The home to work on, or `SpineRefused` saying the deployment configured none."""
    name = home()
    if name is None:
        raise SpineRefused(
            'no spine home is configured, so there is nothing to record to: set {} to the home '
            '`DV_Spine init` minted, or leave it unset and the edge runs exactly as it always '
            'has'.format(SPINE_HOME))
    return name


@contextlib.contextmanager
def translating():
    """Every `SpineRefusal` raised inside leaves as `SpineRefused` carrying the same sentence, so a
    desk reads the library's own words rather than a paraphrase."""
    try:
        yield
    except package().SpineRefusal as refusal:
        raise SpineRefused(str(refusal)) from None


def check_lane(lane):
    """`lane` checked against the record's own vocabulary.

    The names are respelled in this module, but the refusal is `derivus_spine.verbs.check_lane`'s -
    a service wording it itself would be a second source of truth about what a lane means.
    """
    with translating():
        return package().verbs.check_lane(lane)


@contextlib.contextmanager
def writing():
    """The home's log, opened for one act, closed after it, and serialised against this process.

    A context manager rather than a held handle: a home is a directory, and the claim on it belongs
    to whoever is writing now. The lock is taken before the log is opened, so a second thread of one
    service waits rather than meeting `WriterBusy`, which is a refusal meant for a second process.

    The open is inside the `translating` block too, since a configured home that is not a home is
    the commonest of these refusals.
    """
    spine = package()
    name = where()
    with _WRITER:
        with translating():
            log = spine.SpineLog(name)
            try:
                yield log
            finally:
                log.close()


def folded(fold):
    """Run the read-only `fold` over the home's log and return its answer.

    The read side of `writing`. Reading never claims the home, so this takes no lock and can run
    while another thread writes - asking the record a question never queues behind an append.
    """
    with translating():
        log = package().SpineLog(where())
        try:
            return fold(log)
        finally:
            log.close()


@contextlib.contextmanager
def watching(listener):
    """`listener(home, lsn, event_hash)` called once per frame this process lands, for the length
    of the block.

    The record's own publication: `SpineLog` announces where its head went at the moment the bytes
    are durable, so a doorbell rings off the WRITE rather than off a poll of the log. What is
    published is a position and never a fact, and the registration is this process's own and is
    dropped at the end of the block, so a client that went away leaves nothing behind it.
    """
    watchers = package().log.WATCHERS
    watchers.append(listener)
    try:
        yield listener
    finally:
        watchers.remove(listener)


def blob(digest, actor_name=None):
    """The bytes the record holds at `digest`, for a seat it admits to READ them.

    `admit`'s shape on the reading side, and the one read a replica needs beyond the frames: no
    capabilities document in force serves everyone, as every other enforcement here does, and under
    one a seat outside the `read` rows - or a read nobody signed - is refused by name. The class is
    the firm's while classification is dormant, which is why this asks the document rather than the
    blob. The store re-hashes on the way out, so what a replica pulls is self-verifying whatever
    served it.
    """
    spine = package()

    def read(log):
        document, _ = log.capabilities()
        if document is not None:
            subject = actor(actor_name, log)
            if subject not in spine.capability.read_subjects(document):
                raise SpineRefused(
                    'actor {0!r} is not admitted to read the {1} class here, so these bytes are '
                    'not served: a replica pulls the blobs its seat may open, and the CHAIN it '
                    'pulls beside them is re-derived over ciphertext and needs no key at all. '
                    'Declare a read row ({{"read": [{{"class": "{1}", "subject": {0!r}}}]}}) '
                    'through `DV_Spine grant --file`, or follow this record chain-only'.format(
                        subject, spine.vocabulary.FIRM_CLASS))
        return log.store.get(digest)

    return folded(read)


def seeds():
    """The folder a reader starts its folds from: `DV_SPINE_SEEDS`, else the home's own `seeds/`.

    Where a deployment files its seeds - a folder every seat reads after the end of day, or a
    seat's own for a day it wants to stand at - is its own to say; a reader finding none there
    folds from genesis and answers the same.
    """
    named = os.environ.get(SPINE_SEEDS)
    return os.path.abspath(os.path.expanduser(named)) if named else None


def fold_from(log, projector, lsn=None):
    """`projector`'s state at `lsn`, started from the newest verified seed in `seeds()`."""
    return package().projections.fold_from(log, projector, lsn=lsn, folder=seeds())


def _rows(projector, lsn=None):
    """One projector's rows at `lsn`, folded off the home - what every read below is made of."""
    def fold(log):
        named = package().projections.PROJECTORS[projector]
        return named.rows(fold_from(log, named, lsn))

    return folded(fold)


def advancing(log, projector, lsn=None):
    """`projector`'s state at `lsn`, ADVANCED from the position this process last folded it to.

    The strip's own pattern moved onto the booking path: a fold takes an `(lsn, state)` pair, so a
    booking pays for the events since the last one rather than for every decision the desk has ever
    filed - which is the difference between a constant and a number that grows with exactly the rows
    this verb mints. The fold is PURE in that pair, copying the state before it advances it, so what
    is held here is never what a caller reads.

    Bounded at the position it answers rather than at the end of the platter, so the pair says
    exactly what it covers and a frame appended mid-fold is not applied twice. A position BEHIND the
    pair, or one no pair is held for, folds from the newest seed at or behind it (genesis where
    there is none), and a history that is not the one the pairs were taken on drops them
    - keyed on the GENESIS EVENT HASH rather than on the path, the way `read_seed` checks a close's
    own hash, so a home re-minted where the last one stood is a different record and not a fold
    that answers its predecessor's. What comes back is the pair's own state, so a caller READS it
    and does not edit it - and a caller at the position the pair stands at pays nothing.
    """
    global _ADVANCED_HOME
    projections = package().projections
    at = log.head()[0] if lsn is None else lsn
    genesis = log.frame_at(1)['event_hash']
    if _ADVANCED_HOME != genesis:
        _ADVANCED.clear()
        _ADVANCED_HOME = genesis
    held = _ADVANCED.get(projector.name)
    if held and held['lsn'] == at:
        return held['state']
    state = projections.fold(log, projector, lsn=at,
                             seed=held if held and held['lsn'] <= at
                             else projections.latest_seed(log, projector, seeds(), at))
    _ADVANCED[projector.name] = {'projector': projector.name, 'version': projector.version,
                                 'lsn': at, 'state': state}
    return state


def canonical(obj):
    """The bytes the engine hashes `obj` as: sorted keys, tight separators, `CustomJsonEncoder`.

    This is why an engine hash can be a blob address: a blob is named by the SHA-256 of its bytes
    and the engine's hashes are the SHA-256 of exactly these, so `values_hash` and the address of
    the values vector are one number. The spine canonicalises its own objects under RFC 8785 and
    addresses whatever bytes it is handed, so the two schemes compose.
    """
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      cls=CustomJsonEncoder).encode('utf-8')


def cashflow_key(instrument_hash, leg, kind, date):
    """The stable id of one diary row: the content hash of `{instrument, leg, kind, date}`.

    Taken through the record's own canonicaliser, so it is 64 lowercase hex and passes
    `vocabulary.is_hash` - a `status_transition` naming one settlement needs no new field kind and
    the vocabulary does not grow for it. A row IS one `(leg, kind, date)`, so the four are unique;
    and the date is the thing the terms fixed, so a book rolled past a coupon renumbers nothing.
    """
    return package().content_hash(
        {'instrument': instrument_hash, 'leg': leg, 'kind': kind, 'date': date})


def ticket(plan_hash, trade):
    """What an approval of one trade signs: the content hash of the plan the book has once it
    lands and the trade's own fields as its event files them - so the trade retried is the same
    ticket, its event coalescing onto the LSN it already has, and one differing in any field is
    another."""
    return package().content_hash({'plan_hash': plan_hash, 'trade': trade})


def fill_key(instrument, execution_reference):
    """The stable id of one fill, which a confirmation is filed against - `verbs.fill_key`."""
    return package().verbs.fill_key(instrument, execution_reference)


def under(path, node):
    """Whether `path` is the node `node` or a path below it - the record's own reading."""
    return package().capability.under(path, node)


def sight(named=None):
    """Where the seat `named` - or the one a request's token names - acts: `{verb: nodes}`, a verb
    it holds over `*` answering None. None throughout where nothing narrows a read: no home, no
    capabilities document, or a read no seat signs on a box checking no token, which is the
    deployment's own view - a filter a self-declared name could lift protects nothing it
    withholds."""
    named = named or REQUESTER.get()
    if not named or not configured():
        return None

    def read(log):
        spine = package()
        document, _ = log.capabilities()
        subject = actor(named, log)
        return None if document is None else {verb: spine.capability.nodes(
            document, subject, verb) for verb in spine.vocabulary.VERBS}

    return folded(read)


def visible(rows, sight, key='portfolio', verb=None):
    """The `rows` a seat whose `sight` this is sees: every one where nothing narrows it, else those
    whose `key` sits at or under a node it holds `verb` at - any verb where none is named, a seat
    that may act at a node seeing the rows it acts on. PRESENTATION while classification is
    dormant: one class key opens every body, so a seat holding a `read` row reads the frames
    whole."""
    held = None if sight is None else [sight[verb]] if verb else list(sight.values())
    if held is None or None in held:
        return rows
    return [row for row in rows if row.get(key) is not None and any(
        under(row[key], node) for nodes in held for node in nodes)]


def pin():
    """The record's head as the book file's pin - `{lsn, head, hydrated_at}` - or None where no
    home is configured.

    A SIBLING of `Calc` and never inside it: `Context.load_json` reads `Calc` alone and
    `Context.plan_hash` hashes `params` and `deals`, so what is stamped here cannot move a plan.
    """
    import datetime

    lsn, head = folded(lambda log: log.head())
    return {'lsn': lsn, 'head': head, 'hydrated_at': datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}


def fixings(lsn=None, indices=None):
    """`{(index, date): {value, source, effective_time, lsn}}` - the fixing in force per index and
    date at `lsn`, resolved across sources by the order the `fixings` policy declares.

    `indices` are the names the caller is COMPILING against, and the only ones an undeclared source
    order refuses for - a plan may not read a fixing nobody vouched for. A READ asks `observations`
    instead, which refuses nothing.
    """
    return folded(lambda log: package().projections.fixings_at(log, lsn=lsn, indices=indices))


def observations(lsn=None, indices=None):
    """`(resolved, unresolved)` for `indices` at `lsn` - what a READ of the book gets.

    A read refuses NOTHING: the policy in force is asked for first and only the indices it orders
    are resolved, so an index whose authority nobody declared comes back under `unresolved` and the
    row that names it reads due rather than taking the verb down. A print for an index nothing here
    names is nothing to this read at all.
    """
    def fold(log):
        spine = package()
        blob, document, _ = spine.policy.in_force(log, spine.policy.FIXINGS_POLICY, lsn)
        ordered = {} if blob is None else document[spine.policy.FIXINGS_SECTION]
        named = set(indices or ())
        return (spine.projections.fixings_at(log, lsn=lsn, sources=ordered,
                                             indices=named & set(ordered)),
                sorted(named - set(ordered)))

    return folded(fold)


def compiled_job(document, lsn=None, strict=True, runs=None):
    """The job the plan hashes, with every position the record holds at `lsn` written at its net
    and every declared observation filled. Unchanged where no home is configured.

    A position is units of the instrument, so the node carrying it prices at its NET - the fills
    summed over every portfolio under its agreement's set - through `scaled`: a net of one
    is the deal as written, of nothing is no deal, and a second node of the same terms under the
    same set is the first one's clip, ignored rather than priced twice. Ignored rather than
    removed, so every deal path a verb resolved against the file still names the node it named. A
    node the record holds nothing for prices as written.

    `strict` is what separates a PLAN from a READ. A plan may not read a fixing nobody vouched for,
    so an index it compiles against that no `fixings` policy orders refuses by name. A read fills
    what the declared orders can fill and leaves the rest of the document as the desk wrote it -
    the row that named the unresolved index says so on itself, and no GET fails for it.

    The plan is terms PLUS the observations the record holds, so an auditor recompiles from what
    was printed rather than from what somebody typed. A deal type's own table is where its
    observations live and its own `observes` declares which - a fixing the record holds
    for a date the deal names overwrites the cell, and a date it holds nothing for is left standing.

    NOTHING AFTER THE BASE DATE is filled. A `fixing_observed` carries its date as text, so a print
    dated forward is a legal fact; writing one onto a monitoring row would price a barrier as
    already observed on a day that has not happened.

    A collateralised netting set the record holds collateral under is written its BALANCE
    (`_balances`) where the job is compiled for a run that reads one - `runs` the calculation it
    runs as, the document's own where None. A base valuation runs no collateral recursion, so it is
    written none and a settlement moves neither its plan nor what it caches; it still refuses a
    balance its spots cannot cross, as every valuation does. `runs` False is a job that values
    nothing - a diary's schedules - and reads no balance at all. The margin is never written, nor
    a movement dated after the base date.
    """
    if not configured():
        return document
    try:
        job_children(document)
    except ValueError:
        return document
    held = _nets(lsn, book_name(document))
    balances = {} if runs is False else _balances(document, lsn)
    if (runs or document['Calc']['Calculation'].get('Object')) == Base_Revaluation.calc_type:
        balances = {}
    if not held and not balances and not _observing(document):
        return document
    filled = deepcopy(document)
    _hold(filled, held)
    for _, node in walk_job_deals(filled):
        deal = node['Instrument']['.Deal']
        if deal.get('Object') == 'NettingCollateralSet' and deal.get('Reference') in balances:
            deal['Opening_Balance'] = balances[deal['Reference']]
    observing = [(deal, terms, index_named(deal, terms)) for deal, terms in _observing(filled)]
    named = {index for _, _, index in observing if index}
    observed = (fixings(lsn, indices=named) if strict else observations(lsn, named)[0])
    base = _base_day(filled)
    for deal, terms, index in observing:
        deal[terms.table] = [_observed(row, terms, observed.get((index, _day(row)))
                                       if _day(row) <= base else None)
                             for row in deal[terms.table]]
    return filled


def book_name(document):
    """The book a fill is attributed to: the job document's own `Deals.Reference`, or None.

    Capability grants are (verb x book), so a desk scoped over one book must not reach another. A
    document naming none files a firm-level fact, which only a `*` grant reaches.
    """
    named = (document.get('Calc', {}).get('Deals', {}) or {}).get('Reference')
    return named if isinstance(named, str) and named else None


def _nets(lsn=None, book=None):
    """`{(instrument, agreement): net}` - every position `book` holds at `lsn`, summed over its
    portfolios, the agreement being the set the file materialises it under."""
    nets = {}
    for row in positions(lsn):
        if row.get('book') == book:
            key = (row['instrument'], row['agreement'])
            nets[key] = nets.get(key, 0.0) + row['quantity']
    return nets


def _balances(document, lsn=None):
    """`{agreement: balance}` - the `Opening_Balance` of each collateralised netting set of
    `document` the record holds collateral under at `lsn`: what it holds by the job's base date,
    after haircuts (`CSA.valued`), in the set's `Balance_Currency`, every other asset crossed at
    the job's own spots (`PnL.rates`). The `cash` fold is ADVANCED, and folded only where such a
    set exists. A set holding an asset those spots cannot cross REFUSES by name, a plan and a read
    alike: the book lacks that market, and nothing is valued on a zero."""
    sets = {deal.get('Reference'): deal for deal in (
        node['Instrument']['.Deal'] for _, node in walk_job_deals(document))
        if deal.get('Object') == 'NettingCollateralSet' and deal.get('Collateralized') == 'True'}
    if not sets:
        return {}
    spine = package()
    cash = spine.projections.PROJECTORS['cash']
    held = spine.projections.CSA.held(folded(lambda log: cash.rows(advancing(log, cash, lsn))),
                                      _base_day(document))
    market = document['Calc'].get('MergeMarketData', {}).get('ExplicitMarketData') or {}
    balances = {}
    for agreement, terms in sets.items():
        holding = held.get(agreement, spine.projections.CSA.NOTHING)[spine.verbs.COLLATERAL]
        dials = spine.projections.CSA.of(terms)
        currency = dials['Balance_Currency']
        fx = dict(PnL.rates(market.get('Price Factors') or {}, currency, market.get(
            'System Parameters', {}).get('Base_Currency')), **{currency: 1.0})
        uncrossed = sorted(asset for asset, amount in holding.items() if amount and asset not in fx)
        if uncrossed:
            raise SpineRefused(
                'the record holds {} as collateral under {!r} and this book carries no spot for '
                'it into {}, its Balance_Currency: a balance in an asset the book carries no spot '
                'for is a market the book lacks; install the spot'.format(
                    ', '.join(uncrossed), agreement, currency))
        if holding:
            balances[agreement] = spine.projections.CSA.valued(holding, dials, fx)
    return balances


def _hold(document, nets):
    """Every node of `document` the record holds a position in, IN PLACE at its net: scaled, the
    same terms' second node under one set ignored, and a net of nothing ignored. A container the
    record holds nothing for is walked through, a netting set naming the agreement below it.

    A structure held whole is ONE instrument, its legs inside its own terms, so a position the
    record holds in one of those legs has no node of its own to price at: it refuses by name
    rather than being priced as the structure's."""
    containers = mapping['Instrument']['containers']
    carried, inside = set(), {}

    def legs(children, agreement, holder):
        for node in children:
            inside[(content_hash(instrument_of(node)), agreement)] = (node, holder)
            legs(node.get('Children', []), agreement, holder)

    def walk(children, agreement):
        for position, node in enumerate(children):
            deal = node['Instrument']['.Deal']
            if node.get('Ignore') == 'True':
                continue
            key = (content_hash(instrument_of(node)), agreement)
            if key not in nets:
                if deal.get('Object') in containers:
                    walk(node.get('Children', []), deal.get('Reference') if deal.get('Object')
                         == 'NettingCollateralSet' else agreement)
                continue
            legs(node.get('Children', []), agreement, deal)
            if key in carried or not nets[key]:
                node['Ignore'] = 'True'
            elif nets[key] != 1:
                try:
                    children[position] = scaled(node, nets[key])
                except ValueError as refused:
                    raise SpineRefused(str(refused))
            carried.add(key)

    walk(job_children(document), None)
    for key, (node, holder) in inside.items():
        if nets.get(key) and key not in carried:
            leg = node['Instrument']['.Deal']
            raise SpineRefused(
                'the record holds a position in {} {!r} and one in {} {!r}, the structure it sits '
                'in: a structure is held whole or through its legs, never both - close one of the '
                'two'.format(leg.get('Object'), leg.get('Reference'), holder.get('Object'),
                             holder.get('Reference')))


def _scaled_value(field, value, factor):
    """`value` with every part `field` declares sized multiplied by `factor` - a scalar, a table
    column in rows keyed by name or by position, a container's sub-field - and how many it moved."""
    number = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if field.sized and number:
        return value * (abs(factor) if field.sized == 'magnitude' else factor), 1
    if field.sub_fields and isinstance(value, dict):
        declared = {sub.key: sub for sub in field.sub_fields}
        moved = {key: _scaled_value(declared[key], item, factor) if key in declared else (item, 0)
                 for key, item in value.items()}
        return ({key: item for key, (item, _) in moved.items()},
                sum(count for _, count in moved.values()))
    columns = [column for column in field.row.fields if column.sized] if field.row else []
    token = next(iter(value)) if isinstance(value, dict) and len(value) == 1 else None
    rows = value[token] if token and token.startswith('.') else value
    if not columns or not isinstance(rows, list):
        return value, 0
    positions = [field.row.fields.index(column) for column in columns]
    count, scaled = 0, []
    for row in rows:
        row = dict(row) if isinstance(row, dict) else list(row) if isinstance(row, list) else row
        for column, position in zip(columns, positions):
            cell = column.key if isinstance(row, dict) else position
            if isinstance(row, (dict, list)) and (cell in row if isinstance(row, dict)
                                                  else cell < len(row)):
                row[cell], moved = _scaled_value(column, row[cell], factor)
                count += moved
        scaled.append(row)
    return ({token: scaled} if rows is not value else scaled), count


def _sizes(field):
    """Whether `field` declares an amount anywhere in it - itself, a sub-field, a column."""
    return bool(field.sized) or any(_sizes(part) for part in list(field.sub_fields or ()) + (
        field.row.fields if field.row else []))


def scaled(node, quantity):
    """`node` HELD `quantity` times, as a new node: every field its type declares `sized` - its
    legs' included - multiplied, and a negative quantity taken as the MIRROR, every `side` the type
    declares flipped with the sizes at the quantity's magnitude, or the sizes signed where it
    declares none. The engine prices amounts as written and never learns a quantity.

    The legs of a deal written as an OPTION ON ITS CHILDREN - a swaption's underlying - are its
    terms, so they are sized and never flipped; every other deal's legs - a structure's, a cap's
    caplets - are themselves held, and decide. An amount the block omits is what its declaration
    says omission means, so a convention's value is written at the size. A deal stating none of the
    amounts its type declares refuses by name: it would price one unit whatever the position is.
    A frame declaring none - an empty structure - holds nothing to size; a LEAF declaring none is
    one unit whatever the position, and refuses too.
    """
    deal = node['Instrument']['.Deal']
    cls = getattr(instruments, str(deal.get('Object')), None)
    declared = declared_fields(cls) if isinstance(cls, type) else {}
    sides = [field for field in declared.values() if field.side]
    flip = quantity < 0 and bool(sides)
    factor = -quantity if flip else quantity
    count, terms = 0, {}
    for key, value in deal.items():
        terms[key], moved = (_scaled_value(declared[key], value, factor) if key in declared
                             else (value, 0))
        count += moved
    for key, field in declared.items():
        if key not in deal and field.sized and field.convention and field.default:
            terms[key], count = field.default * factor, count + 1
    for side in sides if flip else ():
        held = deal.get(side.key, side.default)
        terms[side.key] = side.values[1] if held == side.values[0] else side.values[0]
    held = dict(node, Instrument=dict(node['Instrument'], **{'.Deal': terms}))
    if node.get('Children'):
        legs = factor if getattr(cls, 'option_on_children', False) else quantity
        held['Children'] = [scaled(child, legs) for child in node['Children']]
    elif not count and (any(_sizes(field) for field in declared.values())
                        or not getattr(cls, 'accepts_children', False)):
        raise ValueError('a position of {} in {} {!r} cannot be priced: it states none of the '
                         'amounts its type declares, so it would price one unit as written - '
                         'state them, or hold it in whole units'.format(
                             quantity, deal.get('Object'), deal.get('Reference')))
    return held


def _base_day(document):
    """The ISO day this job is valued as of - what a print may not be dated after."""
    return _day([document['Calc']['Calculation']['Base_Date']])


def _observing(document):
    """Every `(deal block, terms)` of this job whose type declares an observation table the
    document actually carries rows in. A document that is not a job observes nothing."""
    try:
        nodes = list(walk_job_deals(document))
    except (KeyError, TypeError, ValueError):
        return []
    deals = [node['Instrument']['.Deal'] for _, node in nodes]
    return [(deal, OBSERVES[deal['Object']]) for deal in deals
            if OBSERVES.get(deal.get('Object')) is not None
            and OBSERVES[deal['Object']].table and deal.get(OBSERVES[deal['Object']].table)]


def _day(row):
    """The ISO day a table row is dated at, whatever wire form the date arrived in."""
    date = row[0] if isinstance(row, (list, tuple)) else row
    if isinstance(date, dict):
        date = next(iter(date.values()))
    return (date if isinstance(date, str) else date.strftime('%Y-%m-%d'))[:10]


def _observed(row, terms, fixing):
    """One table row with the record's print written into the column that deal type declares it in.
    A row the record holds no print for is left exactly as the desk wrote it."""
    if fixing is None:
        return row
    cells = list(row) if isinstance(row, (list, tuple)) else [row]
    cells.extend([None] * (terms.column + 1 - len(cells)))
    cells[terms.column] = fixing['value']
    return cells


def replay(context):
    """The four coordinates `context`'s numbers replay from. Hashed, they are also its `result_id`,
    which is what makes an identical submission one execution."""
    return {'plan_hash': context.plan_hash(), 'values_hash': context.values_hash(),
            'engine_version': __version__,
            'seed': context.current_cfg.deals['Calculation'].get('Random_Seed')}


def values_of(context):
    """The values vector this context would run against, as the bytes its `values_hash` names.

    The same clock projection `values_hash` takes, so the stored vector's ADDRESS is the pinned
    hash and `verbs._values` can assert the two are one number.
    """
    return canonical(without_clocks(context.market_patch()))


def result_of(out):
    """A run's results as the bytes a claim is compared byte for byte against.

    `Stats` is excluded: timings and batch counts are facts about a machine, so a result document
    carrying them would never reproduce on a second box even when every number did.

    Flattened through `tables_of`, the shape the result store keeps and a client pages, so a
    per-result-class tolerance is declared against a table path a desk knows and an already-executed
    run's bytes are recoverable as `canonical(result['tables'])`.
    """
    return canonical(tables_of(as_json(out['Results'])))


def result_stored(result):
    """The same bytes as `result_of`, off a result the executor already holds.

    Content addressing means a job whose numbers exist is not run twice, so a standing submission
    can arrive at a tuple this box has already computed and still owes an attestation.
    `result['tables']` is what `result_of` canonicalises, so the two answer the same bytes.
    """
    return canonical(result['tables'])


def read_values(values):
    """A stored values vector back as the objects `patch_market` takes.

    Through `Config.read_json`, the decoder that reads a job file, so a `.Curve` comes back a Curve
    and there is no second parser to keep in step with the first.
    """
    return Config().read_json((bytes(values).decode('utf-8'), 'values'))


def executor(kind=None):
    """The callable `pin_result` re-executes a claim through: `(job, values, version)` in,
    `(version, result bytes)` out. `kind` overrides the `Context` class it runs on.

    The injection seam: the spine cannot import a pricer, so re-execution arrives as a function over
    real documents. The version is checked here before anything runs, so a claim at another build
    costs a refusal rather than a Monte Carlo; the spine checks the version it gets back regardless.
    """
    def execute(job, values, engine_version):
        if engine_version != __version__:
            raise SpineRefused(
                'this build is engine {} and the claim is at {}: a replay claim is a claim AT the '
                'recorded version, so nothing is re-executed here - pin it on a build of {}, or '
                'file the run this build produces as its own attestation'.format(
                    __version__, engine_version, engine_version))
        context = (kind or Context)().load_json((bytes(job).decode('utf-8'), 'pinned'))
        context.patch_market(read_values(values))
        _, out = context.run_job()
        return __version__, result_of(out)

    return execute


# ------------------------------------------------------------------------------------------------
# The verbs. Each one is: canonicalise, open the home, delegate, close.

def book(deal, quantity, counterparty, netting_set, execution_reference,
         actor_name=None, book_name=None, effective_time=None, price=None, agreement=None,
         portfolio=None, currency=None, rate=None, plan=None, terms=None, quote=None,
         once=False):
    """Book a fill against the canonical instrument `deal`, answering what the record now says
    (`routed`) - or, with `once`, answering the fill already standing under its key (`_booked`):
    a booking sent again.

    `deal` is canonicalised through the engine's encoder and its hash is the instrument id, so
    booking the same strike twice registers one instrument and files two events against it.
    `execution_reference` has no default: it is what makes a retry the same fact. `quantity` is the
    position change in units of the instrument; `agreement`, `portfolio` and `price` are filed
    where they are stated, a price in another currency with that `currency` and the `rate` the
    booking crossed it at. Once any node under the book is declared, a portfolio below the book is
    one of them - read off the tree ADVANCED on this handle. `quote` is an acceptance's quote,
    filed before the fill under the fill's own ticket.
    """
    spine = package()
    with writing() as log:
        subject = actor(actor_name, log)
        if portfolio not in (None, book_name):
            declared = sorted(path for path in advancing(log, spine.projections.PROJECTORS[
                'portfolios']) if spine.capability.under(path, book_name))
            if declared and portfolio not in declared:
                raise SpineRefused(
                    'this booking names portfolio {!r}, and book {!r} books into itself or a node '
                    'declared under it - {} - so declare the node first, at its parent'.format(
                        portfolio, book_name, ', '.join(declared)))
        body = spine.verbs.fill(log, canonical(deal), quantity, counterparty, netting_set,
                                execution_reference, price=price, agreement=agreement,
                                portfolio=portfolio, currency=currency, rate=rate)
        return once and _booked(log, body) or routed(
            log, 'fill', body, subject, book_name, effective_time, plan, terms, quote)


def _booked(log, body):
    """`{booked: {lsn, instrument}}` for the fill standing under `body`'s key - its instrument
    and execution reference - where it files the fields `body` does, the rate its price was
    crossed at aside, `instrument` being the terms its position stands in now; None where none
    stands; and refused by name where one stands with other fields."""
    key = (body['instrument'], body['execution_reference'])
    held = next(((instrument, clip['fill']) for instrument, agreements in advancing(
        log, package().projections.PROJECTORS['positions']).items()
        for rows in agreements.values() for row in rows.values() for clip in row['tickets']
        if (clip['instrument'], clip['execution_reference']) == key), None)
    if held is None:
        return None
    filed = log.open_body(log.frame_at(held[1]))
    moved = sorted(field for field in set(filed) | set(body)
                   if field not in ('ticket', 'rate') and filed.get(field) != body.get(field))
    if moved:
        raise SpineRefused(
            'execution {!r} is booked at LSN {} with other terms - {} - so this is not that '
            'booking sent again: a second execution wants its own reference'.format(
                key[1], held[1], ', '.join('{} {!r} where it filed {!r}'.format(
                    field, body.get(field), filed.get(field)) for field in moved)))
    return {'booked': {'lsn': held[1], 'instrument': held[0]}}


def routed(log, event_type, body, subject, book_name, effective_time, plan, terms, quote=None):
    """File `body` - a fill or a restrike - as its ticket routes it, in ONE act of the writer:
    `{recorded, ticket, accepted?, tier?, waits_on?}`, or `{ticket, accepted?, refused}` with
    nothing booked.

    THE TICKET IS THE TRADE: where `plan` - the plan the book has once it lands - is handed in, the
    event carries `ticket` over that plan and its own fields, so a retry of the act lands every
    event on the LSN it already has. And EVERY REFUSAL IT CAN MEET COMES FIRST: the caller built the
    body, which asserted its shape, and `SpineLog.admits` asks the vocabulary and the seat's scope
    with nothing written, so the hub signs nothing a refusal then strands. Then, with a tiers policy
    in force (`terms`, what it reads of the trade), the FIRST tier covering where it books whose
    every check passes applies: an automatic one is the hub's own approval, filed before the event;
    one under four eyes lets it land PENDING, `waits_on` saying on whom; and one no tier admits
    answers `refused` with every sentence the route collected. A crash between the approval and the
    event is the one way an approval stands alone, which the oracle names until the act is
    retried. A seat the scope refuses is answered by the append itself, which lands the denial.
    """
    admitted = log.admits(event_type, body, subject, book_name, effective_time)
    answer = {}
    if plan is not None:
        body['ticket'] = answer['ticket'] = ticket(plan, dict(body))
    if quote is not None:
        answer['accepted'] = package().verbs.file_quote(
            log, subject, ticket=body.get('ticket'), book=book_name,
            portfolio=body.get('portfolio'), **quote)
    route = None if terms is None or not admitted else _route(
        log, body.get('ticket'), dict(terms, portfolio=body.get('portfolio') or book_name), subject)
    if route is not None:
        if route['tier'] is None:
            return dict(answer, refused=route['refusals'])
        answer['tier'] = {'name': route['tier'], 'four_eyes': bool(route['four_eyes']),
                          'approval_lsn': route['approval_lsn']}
        if not route['four_eyes']:
            answer['tier']['approval_lsn'] = log.own(
                'approval', {'plan_hash': body['ticket']}, book=book_name)['lsn']
        elif route['approval_lsn'] is None:
            answer['waits_on'] = route['waits_on']
    answer['recorded'] = dict(log.append(event_type, body, subject, book=book_name,
                                         effective_time=effective_time),
                              **{field: body[field] for field in package().vocabulary.BLOB_FIELDS[
                                  event_type]})
    return answer


def declare_entity(entity, name, parent=None, actor_name=None, effective_time=None):
    """Declare a legal entity the book may trade with. `document` scope, which the writer enforces;
    who holds it is the deployment's grant."""
    verbs = package().verbs
    with writing() as log:
        return verbs.declare_entity(log, actor(actor_name, log), entity, name, parent=parent,
                                    effective_time=effective_time)


def declare_agreement(agreement, entity, kind, terms, actor_name=None, effective_time=None):
    """Declare an agreement with a legal entity, its `terms` the netting set its positions are
    compiled into, canonicalised through the engine's encoder and filed by address."""
    verbs = package().verbs
    with writing() as log:
        return verbs.declare_agreement(log, actor(actor_name, log), agreement, entity, kind,
                                       canonical(terms), effective_time=effective_time)


def declare_portfolio(path, actor_name=None, effective_time=None):
    """Declare a node of the desk's tree, judged at its parent - `admin` there, or over `*` for a
    book itself."""
    verbs = package().verbs
    with writing() as log:
        return verbs.declare_portfolio(log, actor(actor_name, log), path,
                                       effective_time=effective_time)


def portfolios(lsn=None):
    """Every node of the desk's tree declared at `lsn`: its path, and who declared it when."""
    return _rows('portfolios', lsn)


def positions(lsn=None):
    """Every position the record holds at `lsn`, keyed instrument by agreement by portfolio: the net
    quantity in units of the instrument, the clips behind it and the amendment that moved it."""
    return _rows('positions', lsn)


def entities(lsn=None):
    """Every legal entity the record declares at `lsn`: its name and the parent it is grouped
    under."""
    return _rows('entities', lsn)


def agreements(lsn=None):
    """Every agreement the record declares at `lsn`, each with the terms it was declared under read
    back out of the store."""
    def fold(log):
        projections = package().projections
        named = projections.PROJECTORS['agreements']
        rows = named.rows(fold_from(log, named, lsn))
        return [dict(row, terms=json.loads(log.store.get(row['terms']).decode('utf-8')),
                     terms_hash=row['terms']) for row in rows]

    return folded(fold)


def amend(deal, amended_to, actor_name=None, book_name=None, effective_time=None, plan=None,
          terms=None):
    """Record that these terms became those terms - a new instrument hash linked to the old one -
    judged at the deepest node holding every position in them under the book, read off the
    positions ADVANCED on this handle and never stated by a caller. A restrike is a trade of its
    own, ticketed and routed as `book` routes a fill (`routed`), and one that already landed is
    answered `{booked: {lsn}}` rather than filed again (`_restruck`)."""
    spine = package()
    was, now = (hashlib.sha256(canonical(terms)).hexdigest() for terms in (deal, amended_to))
    with writing() as log:
        subject = actor(actor_name, log)
        held = advancing(log, spine.projections.PROJECTORS['positions'])
        landed = _restruck(held, was, now)
        if landed is not None:
            return {'booked': {'lsn': landed}}
        body = spine.verbs.amendment(
            log, canonical(deal), canonical(amended_to), portfolio=spine.capability.deepest(
                spine.capability.holders(held, was, book_name)))
        return routed(log, 'amendment', body, subject, book_name, effective_time, plan, terms)


def _restruck(positions, was, now):
    """The LSN of the restrike that already moved `was` onto `now`, or None: the terms emptied
    of every clip and naming `now` as where they went, or - `was` being `now`, the file already
    carrying the change - clips a restrike moved onto them. `positions` is the fold's state."""
    rows = [row for held in positions.get(was, {}).values() for row in held.values()]
    if was != now:
        landed = [row['last_lsn'] for row in rows if row['amended_to'] == now]
        return max(landed) if landed and not any(row['clips'] for row in rows) else None
    return max((clip['lsn'] for row in rows for clip in row['tickets']
                if clip['instrument'] != now), default=None)


def apply_lifecycle(event_type, body, actor_name=None, book_name=None, effective_time=None):
    """File an election, a fixing observation or a determination. Anything consequence-shaped is
    refused - see `derivus_spine.verbs.apply_lifecycle`."""
    verbs = package().verbs
    with writing() as log:
        return verbs.apply_lifecycle(log, actor(actor_name, log), event_type, body,
                                     book=book_name, effective_time=effective_time)


def transition(subject, status, actor_name=None, book_name=None, effective_time=None,
               amount=None, asset=None, kind=None, reference=None):
    """Move the state a party put a subject in - a settlement paid, a confirmation matched. The
    subject is a derived cashflow key or an instrument address; the diary's rows are a fold's
    question and not this verb's. A settlement that moved money states the amount, the asset, its
    kind and the settlement system's reference - see `derivus_spine.verbs.transition`."""
    verbs = package().verbs
    with writing() as log:
        return verbs.transition(log, actor(actor_name, log), subject, status, book=book_name,
                                effective_time=effective_time, amount=amount, asset=asset,
                                kind=kind, reference=reference)


def lifecycle(lsn=None):
    """The `lifecycle` fold at `lsn`: every print, election and ruling, and the status standing
    under each subject a transition names."""
    return _rows('lifecycle', lsn)


def cash(lsn=None):
    """Every movement of money the record's settlements filed at `lsn`, under the settlement
    system's own reference - a restated one standing in place of the filing it corrects."""
    return _rows('cash', lsn)


def cash_standing():
    """`(lsn, rows)` - the head, and the `cash` fold's rows there ADVANCED from where this process
    last folded it: what a collateral call reads on every ask."""
    def read(log):
        cash = package().projections.PROJECTORS['cash']
        head = log.head()[0]
        return head, cash.rows(advancing(log, cash, head))

    return folded(read)


def costs(lsn=None):
    """What every position cost at `lsn`, at average cost, and what its reductions realised - keyed
    where the position sits, a split nobody priced reading null."""
    return _rows('costs', lsn)


def attestations():
    """Every standing attestation, oldest first: the replay tuple and the job, values and result
    it names."""
    return _rows('attestations')


def closes(market):
    """Every official close declared on `market`, superseded ones included, in the order they were
    filed: the position, the values it stands on, and the day it is FOR - the one it states, else
    the day it is true on."""
    def walk(log):
        found = []
        for frame in log.frames():
            if frame['event_type'] == 'official_close_declared':
                body = log.open_body(frame)
                if body['market'] == market:
                    found.append({'lsn': frame['lsn'], 'values_hash': body['values_hash'],
                                  'date': body.get('date') or (
                                      frame['effective_time'] or frame['record_time'])[:10]})
        return found

    return folded(walk)


def fills(after=None, until=None):
    """Every fill filed after LSN `after` and at or before `until`, each with where it sits in the
    record, who booked it and when it is true."""
    def walk(log):
        return [dict(log.open_body(frame), lsn=frame['lsn'], book=frame['book'],
                     actor=frame['actor'], as_of=frame['effective_time'] or frame['record_time'])
                for frame in log.frames(start_lsn=(after or 0) + 1, end_lsn=until)
                if frame['event_type'] == 'fill']

    return folded(walk)


#: What a position's tickets read together, the first a desk acts on first: one rejected, one
#: awaiting a second seat, every one signed, none under a workflow.
STATUSES = ('rejected', 'pending', 'approved', 'unticketed')


def standing(lsn=None):
    """Every position the record holds at `lsn` with what its tickets read: each ticket its own
    `status` (`status_of`), the position the first of `STATUSES` among them and `pending` the
    quantity awaiting a second seat. The positions fold beside the decisions fold, both ADVANCED
    in one open of the log - no fill is reopened."""
    def read(log):
        projections = package().projections
        decisions = advancing(log, projections.PROJECTORS['decisions'], lsn)
        rows = projections.PROJECTORS['positions'].rows(
            advancing(log, projections.PROJECTORS['positions'], lsn))
        for row in rows:
            row['tickets'] = [dict(entry, status=status) for entry, status in zip(
                row['tickets'], status_of(row['tickets'], decisions))]
            said = {entry['status'] for entry in row['tickets']}
            row.update(status=next((word for word in STATUSES if word in said), STATUSES[-1]),
                       pending=sum((entry['quantity'] for entry in row['tickets']
                                    if entry['status'] == 'pending'), 0.0))
        return rows

    return folded(read)


def status_of(tickets, decisions):
    """What each of a position's `tickets` reads: `approved`, `rejected` or `pending` by the
    verdict standing over it, and `unticketed` where it carries none or no tiers policy stood when
    it landed. `decisions` is the decisions fold's state.

    A tier is automatic unless it declares `four_eyes`, and an automatic one is the hub's own
    approval filed before the trade, so a ticket the hub never signed is under four eyes: a seat
    other than its booker signs it, the booker's own verdicts not read.
    """
    spine = package()
    since = decisions['policies'].get(spine.policy.TIERS_POLICY, {}).get('since')
    read = []
    for entry in tickets:
        verdicts = decisions['plans'].get(entry['ticket'], [])
        if entry['ticket'] is None or since is None or since > entry['lsn']:
            read.append('unticketed')
            continue
        standing = spine.tiers.standing_verdict({'four_eyes': not any(
            verdict['verdict'] == 'approval' and verdict['actor'] == spine.vocabulary.WRITER
            for verdict in verdicts)}, entry['actor'], verdicts) or {}
        read.append({'approval': 'approved', 'rejection': 'rejected'}.get(
            standing.get('verdict'), 'pending'))
    return read


def amendments(after=None, until=None):
    """Every amendment filed after LSN `after` and at or before `until`: the terms restruck and the
    terms they became, with where it sits in the record."""
    def walk(log):
        return [dict(log.open_body(frame), lsn=frame['lsn'], book=frame['book'])
                for frame in log.frames(start_lsn=(after or 0) + 1, end_lsn=until)
                if frame['event_type'] == 'amendment']

    return folded(walk)


def stored(digest):
    """The bytes the store holds at `digest` - a job, a result or a values vector a reading cites,
    read on this box's own record."""
    return folded(lambda log: log.store.get(digest))


def approve(plan_hash, actor_name=None, book_name=None, effective_time=None, portfolio=None):
    """Sign a plan hash, judged at the `portfolio` it books into where one is named. One seat
    approving one plan twice is one fact, coalescing onto the LSN it already has; an amended plan
    is a different hash and so is a different signature."""
    verbs = package().verbs
    with writing() as log:
        return verbs.approve(log, actor(actor_name, log), plan_hash, book=book_name,
                             effective_time=effective_time, portfolio=portfolio)


def reject(plan_hash, reason, actor_name=None, book_name=None, effective_time=None,
           portfolio=None):
    """Refuse a plan hash with the reason on the row, judged where `approve` is."""
    verbs = package().verbs
    with writing() as log:
        return verbs.reject(log, actor(actor_name, log), plan_hash, reason, book=book_name,
                            effective_time=effective_time, portfolio=portfolio)


def declare_market(name, values, actor_name=None, effective_time=None):
    """Point the market `name` at a values vector. `official` demands `mark` scope, which the writer
    enforces - a check here would be a second place to get it wrong."""
    verbs = package().verbs
    with writing() as log:
        return verbs.declare_market(log, actor(actor_name, log), name, values,
                                    effective_time=effective_time)


def declare_close(market, values, actor_name=None, effective_time=None, date=None):
    """Declare the official close on `market` for the day `date` over a values vector. A second
    close supersedes the first rather than editing it, so a day restated is two facts and both stay
    readable."""
    verbs = package().verbs
    with writing() as log:
        return verbs.declare_close(log, actor(actor_name, log), market, values,
                                   effective_time=effective_time, date=date)


def file_quote(quote_id, structure, plan_hash, values, solved, edge, request=None, ticket=None,
               actor_name=None, book_name=None, effective_time=None):
    """File a quote with both hashes pinned - the values vector it was struck on and the book plan
    its marginal charge was solved against - and, where the caller computed one, the `ticket` plan
    an approval of this quote would sign."""
    verbs = package().verbs
    with writing() as log:
        return verbs.file_quote(log, actor(actor_name, log), quote_id, structure, plan_hash,
                                values, solved, edge, request=request, ticket=ticket,
                                book=book_name, effective_time=effective_time)


def complete_run(claim, lane, job, values, result, book_name=None):
    """Attest a standing run at birth, in the writer's own voice: the hub that ran it says so. A
    telemetry or curiosity lane mints nothing and should not reach here; the verb refuses one that
    does rather than dropping it quietly."""
    verbs = package().verbs
    with writing() as log:
        return verbs.complete_run(log, lane, claim, job, values, result, book=book_name)


def pin_result(claim, job, values, result, actor_name=None, book_name=None, effective_time=None,
               execute=None):
    """Promote a replay claim this hub did not witness, through re-execution or a cache hit."""
    verbs = package().verbs
    with writing() as log:
        return verbs.pin_result(log, actor(actor_name, log), claim, job, values, result,
                                execute or executor(), book=book_name,
                                effective_time=effective_time)


def tiers_policy():
    """The tiers document standing in the record, or None where this home declared none.

    A fold like every other policy read, so what a ticket would be routed by is answerable without
    queueing behind a booking.
    """
    return folded(package().policy.tiers_in_force)


def tolerances(lsn=None):
    """`{result class: epsilon}` the tolerance policy standing at `lsn` declares, or nothing where
    none stands - every difference then a departure (`policy.compare`)."""
    def read(log):
        policy = package().policy
        return (policy.in_force(log, policy.TOLERANCE_POLICY, lsn)[1] or {}).get(
            policy.TOLERANCE_SECTION, {})

    return folded(read)


def quote_at(lsn, quote_id):
    """The quote filed at `lsn`, as the body the record holds, or a refusal naming what is there.

    ONE FRAME AND ONE BODY. A decision names the quote it is about and the acceptance wrote down
    where that quote sits, so this is `frame_at`'s seek by byte offset rather than a fold over every
    quote the desk has ever struck - the one read whose cost would otherwise grow with the desk's
    own activity. The id is checked against the body rather than assumed: a position remembered
    somewhere else is not evidence about what stands at it.
    """
    def sought(log):
        frame = log.frame_at(lsn)
        if frame['event_type'] != 'quote_filed':
            raise SpineRefused(
                'LSN {} holds a {} in this record and not the quote {!r} - the acceptance files '
                'the quote and writes down where, so a position holding something else is a '
                'pending file edited or copied from another home. Accept the quote '
                'again'.format(lsn, frame['event_type'], quote_id))
        body = log.open_body(frame)
        if body.get('quote_id') != quote_id:
            raise SpineRefused(
                'LSN {} holds the quote {!r} in this record and not {!r} - a pending file names '
                'the position its own acceptance was filed at, so this one was edited or copied '
                'from another home. Accept the quote again'.format(
                    lsn, body.get('quote_id'), quote_id))
        return dict(body, lsn=frame['lsn'], actor=frame['actor'], book=frame['book'])

    return folded(sought)


def verdicts(plan_hash, lsn=None):
    """The verdicts filed against `plan_hash` at `lsn`, oldest first.

    An empty list is a plan nobody has ruled on, which is not the answer a rejected plan gives - the
    two have different remedies, so the caller reads the list rather than a boolean.
    """
    return _verdicts(_rows('decisions', lsn), plan_hash)


def _verdicts(rows, plan_hash):
    """The verdicts the `decisions` rows hold against `plan_hash`, oldest first."""
    return next((row['verdicts'] for row in rows['plans'] if row['plan_hash'] == plan_hash), [])


def route_ticket(plan_hash, terms, booker, lsn=None):
    """Which tier a ticket falls in here and whether it is already signed, or None where this home
    declared no tiers - which is the flow that books. `_route` on a handle of its own."""
    return folded(lambda log: _route(log, plan_hash, terms, booker, lsn))


def _route(log, plan_hash, terms, booker, lsn=None):
    """Which tier a ticket falls in on `log` and whether it is already signed, or None where this
    home declared no tiers.

    `plan_hash` is the ticket ITSELF, the plan a decision is filed over; `terms` is what the checks
    read, `{notional_in, tenor_years, values_hash, portfolio}`, as `tiers.assess` takes it. The
    composition nothing else performs: the `tiers` document in force, the evaluator over it, the
    market names standing, and - for a tier under four eyes - the verdicts standing over that plan.
    Both folds ADVANCE (`advancing`) rather than re-walk, and a home whose decisions name no tiers
    policy walks nothing more: this runs on every booking, inside the writer's act.

    `{tier, four_eyes, refusals, approval_lsn, waits_on}`: a `tier` of None with the route's own
    sentences under `refusals` is a ticket no tier admits; a tier without `four_eyes` is automatic,
    the hub's own approval; `approval_lsn` is the verdict that stands and `waits_on` names what is
    missing where none does.
    """
    spine = package()
    decisions = advancing(log, spine.projections.PROJECTORS['decisions'], lsn)
    document = (spine.policy.tiers_in_force(log, lsn)
                if spine.policy.TIERS_POLICY in decisions['policies'] else None)
    if document is None:
        return None
    markets = advancing(log, spine.projections.PROJECTORS['markets'], lsn)['names']
    verdict = spine.tiers.assess(
        document, terms, dict((name, row['values_hash']) for name, row in markets.items()))
    tier = next((row for row in document[spine.policy.TIERS_SECTION]
                 if row['name'] == verdict['tier']), {})
    answer = {'tier': verdict['tier'], 'four_eyes': tier.get('four_eyes'),
              'refusals': verdict['refusals'], 'approval_lsn': None, 'waits_on': None}
    if tier.get('four_eyes'):
        answer['approval_lsn'], answer['waits_on'] = spine.tiers.standing_approval(
            tier, booker, decisions['plans'].get(plan_hash, []))
    return answer


def resolve_market(name, actor=None, process=None):
    """The values vector standing under the market `name`: `{name, values_hash, lsn, values}`.

    The composition nothing performed before - the `markets` fold names it, the store holds the
    bytes, `read_values` turns them back into what `patch_market` takes. A name nobody declared
    REFUSES rather than falling back on the book's own market, which is the whole point of binding a
    process to a market by name.

    A NAME IS RESOLVED TO ITS LATEST DECLARATION, across the declarations of the name and the
    official close standing on it alike - a close is one way of declaring what a market stands on,
    and a reader that took the name's own row would answer yesterday's board on a market the desk
    has since closed. The close standing is the market's latest BY DAY, so a past day restated
    never unseats a later day's board; the two are merged by the fold's own as-of key,
    `(effective_time, lsn)`, so a close backdated behind the mark in force does not displace it.

    `private/<subject>/<name>` is one seat's own and resolves for that SUBJECT alone - the name
    carries its owner and `verbs.declare_market` refuses a private name whose subject is not the
    seat declaring it, so ownership reads the same off the name and off the fold. The surveillance
    and admin read of one is an entitlement class this record does not yet classify, so nobody else
    reaches it here. The two rules are checked independently: with `process` named, the market must
    ALSO be the one the tiers policy designates for it, which no private name ever is.
    """
    asking = actor or hub()

    def fold(log):
        spine = package()
        # the fold's own state rather than its rows: the rows drop the as-of key the merge orders by
        markets = spine.projections.fold(log, spine.projections.PROJECTORS['markets'])
        standing = [rows[name] for rows in (markets['names'], markets['closes']) if name in rows]
        if not standing:
            raise SpineRefused(
                'no market is declared under the name {!r} in this record: a process bound to a '
                'market by name may not price on whatever market happens to be loaded, so this '
                'refuses rather than falling back on the book\'s own. The names standing are '
                '{}'.format(name, ', '.join(sorted(
                    set(markets['names']) | set(markets['closes']))) or '(none)'))
        owner = name.split('/')[1] if name.startswith(spine.policy.PRIVATE_MARKET) else None
        if owner is not None and asking != owner:
            raise SpineRefused(
                '{!r} is {!r}\'s own market and this read names {!r}: a private market resolves for '
                'the seat that declared it and for nobody else here, the surveillance and admin '
                'read of one being an entitlement class this record does not yet classify. Ask '
                'that seat, or declare your own'.format(name, owner, asking))
        if process is not None:
            designations = (spine.policy.tiers_in_force(log) or {}).get(
                spine.policy.DESIGNATIONS_SECTION, {})
            if designations.get(process) != name:
                raise SpineRefused(
                    'the process {!r} resolves {} here and this asked for the market {!r}: a '
                    'designated process prices on the market the {} policy designates for it and '
                    'never on another - and never on a {} market, which is one seat\'s own. '
                    'Designate it ({{"{}": {{{!r}: "<market>"}}}}), or resolve the market without '
                    'naming a process'.format(
                        process, repr(designations[process]) if process in designations
                        else 'nothing at all', name, spine.policy.TIERS_POLICY,
                        spine.policy.PRIVATE_MARKET, spine.policy.DESIGNATIONS_SECTION, process))
        latest = max(standing, key=lambda row: (row['as_of'], row['lsn']))
        # the POSITION of the declaration in force, so a file struck on this board names where the
        # board was declared and replays to it rather than to whatever the name answers later
        return {'name': name, 'values_hash': latest['values_hash'], 'lsn': latest['lsn'],
                'values': read_values(log.store.get(latest['values_hash']))}

    return folded(fold)


def designated_market(process):
    """The market this record DESIGNATES for `process`, resolved - `resolve_market`'s answer.

    Which board a process prices on is a governance decision the desk declares once, never a
    caller's parameter, so the name comes off the `tiers` policy and resolving another through a
    designated process is unrepresentable rather than merely refused. A home designating nothing is
    told what to declare; a designated name nothing stands under refuses where the name resolves.
    """
    spine = package()
    named = designation(process)
    if not named:
        raise SpineRefused(
            'this record designates no market for {0!r}, so there is nothing to strike it on: '
            'which board a designated process prices on is declared once and read by name, never '
            'chosen per call. Declare it on the {1} policy ({{"{2}": {{"{0}": "official"}}}}) '
            'through `DV_Spine declare {1} <file.json>`'.format(
                process, spine.policy.TIERS_POLICY, spine.policy.DESIGNATIONS_SECTION))
    return resolve_market(named, process=process)


def designation(process):
    """The market name the `tiers` policy in force designates for `process`, or None."""
    return (tiers_policy() or {}).get(package().policy.DESIGNATIONS_SECTION, {}).get(process)


def admit(lane, actor_name=None, book=None):
    """Let this seat put work on the hub's compute, or refuse in the record's own words.

    THE QUEUE IS THE ONE WAY IN. `pin_result` has no HTTP verb, so a submission is the whole of what
    an unscoped actor could make this box pay for, and the question is asked once where every job
    passes rather than in each verb. A STANDING run is numbers a fact is about to cite - the hub
    attests them in its own voice - so it asks `mark` over the book the job marks; every other lane
    mints nothing and asks `validate` over the book or any node under it, a seat working one node
    of a book being admitted to price the book.

    Enforcement activates BY DECLARATION, as it does at the writer: with no capabilities document in
    force every job is admitted and this box is the single-user instrument it was. Under one, an
    unnamed actor is refused by name - a job nobody signed for is one the record could not attribute
    - and a refused seat lands its `capability_denied` in the writer's voice before the raise, a
    repeat coalescing onto the LSN it already has.
    """
    standing = lane == STANDING
    # the type the lane would have filed, or the lane itself where it files none
    entitled(package().vocabulary.MARK if standing else package().vocabulary.VALIDATE, actor_name,
             book, STANDING_TYPE if standing else (lane or CURIOSITY), 'this job is not queued '
             'and nothing runs: the hub\'s compute is reached through the queue and the queue asks '
             'first, so work this seat is not scoped for is work this box does not pay for',
             anywhere=not standing)


def entitled(verb, actor_name, book, attempted, refused, anywhere=False):
    """Let this seat act under `verb` over `book` - over any node under it where `anywhere` - or
    refuse in the record's own words, saying `refused`, the denial of `attempted` landed first.
    Every seat where no capabilities document is in force."""
    spine = package()

    def asked(log):
        document, genesis = log.capabilities()
        if document is None:
            return None
        subject = actor(actor_name, log)
        held = (spine.capability.holds_any(document, subject, verb, book) if anywhere
                else spine.capability.evaluate(document, genesis, subject, verb, book))
        return None if held else subject

    subject = folded(asked)
    if subject is None:
        return
    with writing() as log:
        denial = log.refuse(subject, verb, book, attempted)
    raise SpineRefused(
        'actor {0!r} holds no {1} scope over {2!r}, so {3}. Declare a document granting ({0!r}, '
        '{1}, {2!r}) through `DV_Spine grant --file`. The refusal is itself recorded at LSN '
        '{4}'.format(subject, verb, book if book is not None else spine.capability.ANY_BOOK,
                     refused, denial['lsn']))


def firmness_policy():
    """The staleness window standing in the record, or the empty document a home declaring none
    runs on.

    A fold rather than a write, so asking what the window is never queues behind a booking or turns
    an acceptance into a `WriterBusy`.
    """
    return folded(package().policy.firmness_in_force)


def check_firmness(pinned, current, pillar_age, quote_id=None, policy=None):
    """Whether this quote may be booked: the verdict, or a refusal naming what failed and its
    remedy.

    A pure call into `derivus_spine.firmness` over plain data. `policy` is read out of the record
    when the caller hands none in, which is the ordinary case - the window is policy data, so it
    lives in the log rather than in a constant here.
    """
    window = firmness_policy() if policy is None else policy
    with translating():
        return package().firmness.check(pinned, current, pillar_age, window, quote_id=quote_id)


def _key(instrument, leg, kind, date):
    """The row's derived key, or None where nothing booked the instrument or the record is not
    installed on this box - the diary being a reading of the book either way.

    The row's own DATE and never its position: a schedule drops the rows it has paid as the book
    rolls, so a position renumbers under a settlement reference already filed while the date it
    named does not move.
    """
    if instrument is None or date is None:
        return None
    try:
        return cashflow_key(instrument, leg, kind, date)
    except SpineRefused:
        return None


def _times(quantity, unit):
    """A quantity of units, or UNKNOWN where the unit is."""
    return None if unit is None else quantity * unit


def _sum(*parts):
    """A total that stays UNKNOWN once any part is."""
    return None if any(part is None for part in parts) else sum(parts)


def _less(held, less):
    """A difference that stays UNKNOWN once either side is."""
    return None if held is None or less is None else held - less


def diary(context, instruments=None):
    """The `Diary` calculation's rows over `context`, sorted, each with its instrument and key.

    `instruments` maps a deal's `Reference` to the instrument address the record books it under,
    which is what gives a row its key; a reference it does not name carries `instrument: null` and
    `key: null`, so the diary reads on a book nothing was ever booked from. The precision and the
    device are the defaults rather than a valuation's: the diary reads the numpy half of a
    schedule, which is authoritative until the bind and unmoved by it.
    """
    config = context.current_cfg
    declared = config.deals['Calculation']
    named, rows = instruments or {}, []
    for row in construct_calculation('Diary', config).execute(dict(
            declared, Run_Date=declared['Base_Date'].strftime('%Y-%m-%d'))):
        instrument = named.get(row.pop('reference'))
        rows.append(dict(key=_key(instrument, row['leg'], row['kind'], row['due_date']),
                         instrument=instrument, **row))
    return sorted(rows, key=lambda row: (row['due_date'] or '', row['kind'], row['leg'],
                                         row['schedule_index'], row['instrument'] or ''))


def export_settlements(rows, official_values_hash, due_before):
    """The settlement file for ONE day: every payment due on or before `due_before` that is not
    already settled, against one market.

    Takes the diary, the hash of the market it is struck on and the day it settles, and nothing else
    - no book, no service, no executor. The hash comes back as `values_hash`, the BOARD the file was
    struck on, leaving `market` free for the name a caller resolved it under. `due_before` has no
    default: a settlement file is struck FOR a date, and one that exported every future payment
    would instruct the whole book. A row whose amount is UNDETERMINED refuses by name rather than
    exporting a zero: a floating amount is not known until its resets fix, and instructing a payment
    of 0.0 for one is a wrong payment rather than a missing one.
    """
    payments = [row for row in rows if row['kind'] == Diary.PAYMENT and row['state'] != SETTLED
                and row['due_date'] and row['due_date'] <= due_before]
    undetermined = [row for row in payments if not row['determined']]
    if undetermined:
        raise ValueError(
            'these {} rows are due and their amount is not determined, so nothing is exported for '
            'them: {}. A floating amount fixes when its resets do - file the observations and '
            'export again, or export the determined rows by asking for them'.format(
                len(undetermined), ', '.join(
                    '{} {} {}[{}] on {}'.format(row['key'], row['instrument'], row['leg'],
                                                row['schedule_index'], row['due_date'])
                    for row in undetermined[:8])))
    unnamed = [row for row in payments if not row['currency']]
    if unnamed:
        raise ValueError(
            'these {} rows name no currency, so nothing is exported for them: {}. A settlement '
            'file totals BY currency, and a total under none is money going nowhere - name the '
            "deal's own Currency and export again".format(
                len(unnamed), ', '.join('{} {}[{}]'.format(row['key'], row['leg'],
                                                           row['schedule_index'])
                                        for row in unnamed[:8])))
    totals = {}
    for row in payments:
        totals[row['currency']] = totals.get(row['currency'], 0.0) + row['amount']
    return {'values_hash': official_values_hash, 'due_before': due_before, 'totals': totals,
            'rows': [{field: row[field] for field in
                      ('key', 'instrument', 'leg', 'schedule_index', 'due_date', 'currency',
                       'amount')} for row in payments]}


class PnL:
    """The desk's P&L - what the book made between two marked closes, or since the last one.

    A close is MARKED by one standing run valuing ONE UNIT of every instrument the book holds or
    traded since the last marks, each as written, on the close's own values: a position's value is
    its quantity times its unit mark, any grouping of the book is a sum, and a past close replays
    from the job the run attested rather than from a book file that has moved since. Between two
    marks, per position,

        P&L = value at the end - value at the start + premiums + payments + fees

    in the reporting currency: the premiums at each fill's consideration, crossed at the rate its
    booking filed; the payments at what the diary determines, falling due in the window, else what
    the settlements FILED in the window moved, and the fees as filed, each at the official close
    standing on its own day as the record stood at the end of the business day it counts in - so a
    late filing lands in the day it was filed, a day already struck never moves, and a month is the
    sum of its days; the realised half at average cost, a position's open basis released in the
    window its last day falls in. The engine values a payment into the marks of the day it falls
    due, so a position's value at a close is its quantity times its unit mark LESS what that day
    paid - the close is the end of the day and the payment is cash by then. A number nobody can
    know - a fill booked with no price, a payment neither the diary nor a settlement states, a day
    no close stands on - is NAMED, never read as zero.
    """

    #: `{(job address, book): day}` - the day a marks job valued its book as of, or None for a job
    #: that marks no book of that name. A job is immutable, so each is opened once per process.
    DAYS = {}

    #: A row's figures, in the order a reader sees them, and the ones a total sums.
    FIGURES = ('value_start', 'value_end', 'premiums', 'payments', 'fees', 'existing', 'trading',
               'pnl', 'realised', 'unrealised')
    #: The new-deal split, which needs a unit mark at the end for a position traded to nothing in
    #: the window - one the end's marks may not carry - and so can be unknown while the P&L is not.
    SPLIT = ('existing', 'trading')

    @staticmethod
    def unit_node(address, terms):
        """One unit of the instrument `terms`, as written, filed under its own `address`."""
        node = {'Instrument': {'.Deal': dict(
            {key: value for key, value in terms.items() if key != 'Children'}, Reference=address)}}
        if terms.get('Children'):
            node['Children'] = deepcopy(terms['Children'])
        return node

    @staticmethod
    def held_terms(document, wanted, stored):
        """`{address: terms}` for every instrument in `wanted` - read off the first node of the book
        file carrying it, a structure being ONE instrument with its legs inside its terms, and off
        the record's own copy of the terms where the file no longer holds one."""
        containers, found = mapping['Instrument']['containers'], {}

        def walk(children):
            for node in children:
                address = content_hash(instrument_of(node))
                if address in wanted:
                    found.setdefault(address, instrument_of(node))
                elif node['Instrument']['.Deal'].get('Object') in containers:
                    walk(node.get('Children', []))

        walk(job_children(document))
        for address in sorted(set(wanted) - set(found)):
            found[address] = json.loads(stored(address).decode('utf-8'))
        return found

    @classmethod
    def held_legs(cls, children, known):
        """`{leg address: holder address}` for every leg inside an instrument of `known` - a
        structure held whole is ONE instrument, so what its legs pay, and the day the last of them
        ends, are its own. `children` is a deal tree, the netting sets it nests positions in walked
        through; a node is the instrument its terms hash to, or a unit a marks job files under that
        address."""
        containers, found = mapping['Instrument']['containers'], {}

        def walk(nodes):
            for node in nodes:
                address = content_hash(instrument_of(node))
                filed = node['Instrument']['.Deal'].get('Reference')
                holder = address if address in known else filed if filed in known else None
                if holder is not None:
                    found.update((content_hash(instrument_of(leg)), holder)
                                 for leg in cls._leaves(node.get('Children', [])))
                elif node['Instrument']['.Deal'].get('Object') in containers:
                    walk(node.get('Children', []))

        walk(children)
        return found

    @classmethod
    def apart(cls, units):
        """`units` in as few groups as leave every leaf to one unit of its group: a leaf two units
        carry - the same terms, or one reference over other terms, as a restruck structure's legs
        beside the ones it replaced - is read once per unit, each unit in a group of its own."""
        groups = []
        for unit in units:
            leaves = {part for leaf in cls._leaves([unit]) for part in (
                leaf['Instrument']['.Deal'].get('Reference'), content_hash(instrument_of(leaf)))}
            group = next((group for group in groups if not group[1] & leaves), None)
            if group is None:
                groups.append(([unit], leaves))
            else:
                group[0].append(unit)
                group[1].update(leaves)
        return [group for group, _ in groups]

    @classmethod
    def _leaves(cls, nodes):
        """Every node under `nodes` with no children of its own - a node with none being its own."""
        for node in nodes:
            if node.get('Children'):
                yield from cls._leaves(node['Children'])
            else:
                yield node

    @classmethod
    def marks_job(cls, document, terms, cut=None):
        """The job marking `terms` ({address: terms}): one unit of each on `document`'s market and
        calculation, as a base valuation under the marks book - which names `cut`, the position of
        the record the terms were read at, where the marks close a business day there."""
        calc = dict(document['Calc'])
        calc['Calculation'] = {key: value for key, value in dict(
            calc['Calculation'], Object='BaseValuation').items() if key != 'Greeks'}
        book = package().verbs.marks_name(book_name(document), cut)
        calc['Deals'] = dict(calc['Deals'], Reference=book, Deals={
            'Children': [cls.unit_node(address, terms[address]) for address in sorted(terms)]})
        return {'Calc': calc}

    @staticmethod
    def unit_marks(results):
        """`{address: value}` off a marks run's results - its top-level rows, each one unit of the
        instrument filed under its own address. An instrument the run could not price, or that had
        expired, has no row."""
        table = tables_of(results)['mtm']['.DataFrame']
        column = {name: position for position, name in enumerate(table['columns'])}
        root = table['data'][0][column['Reference']] if table['data'] else None
        return {row[column['Reference']]: float(row[column['Value']] or 0.0)
                for row in table['data'][1:] if row[column['Parent']] == root}

    @classmethod
    def marked(cls, book, closes, attestations, stored):
        """`{day: marks}` - the LATEST standing marks run of `book` on each of `closes`, keyed by
        the day it valued the book as of: the position of the record it read the book at, which
        closes that business day, the values it stood on, and the addresses of the job it attested
        and the result. `loaded` reads a marks' numbers."""
        values, found = {row['values_hash'] for row in closes}, {}
        for row in attestations:
            if row['values_hash'] not in values:
                continue
            if (row['job'], book) not in cls.DAYS:
                job = json.loads(stored(row['job']).decode('utf-8'))
                named = package().verbs.marks_of(job['Calc']['Deals'].get('Reference'))
                cls.DAYS[(row['job'], book)] = None if named is None or named[0] != book else (
                    timestamp(job['Calc']['Calculation']['Base_Date']).strftime('%Y-%m-%d'),
                    named[1])
            if cls.DAYS[(row['job'], book)] is None:
                continue
            day, cut = cls.DAYS[(row['job'], book)]
            if day not in found or row['lsn'] > found[day]['attested']:
                # a marks job naming no cut read the book where it was attested
                found[day] = {'day': day, 'lsn': row['lsn'] if cut is None else cut,
                              'attested': row['lsn'], 'values_hash': row['values_hash'],
                              'job': row['job'], 'result': row['result']}
        return found

    @classmethod
    def loaded(cls, marks, stored):
        """`marks` with what a P&L reads off them: the `document` of the job it attested and the
        unit marks of its result."""
        def read(digest):
            return json.loads(stored(digest).decode('utf-8'))

        return dict(marks, document=read(marks['job']), units=cls.unit_marks(read(marks['result'])))

    @staticmethod
    def within(row, portfolio=None, agreement=None, clients=None, sight=None):
        """Whether a position row sits in the scope asked for: under a portfolio path, under one
        agreement, with a counterparty among `clients`, and where a seat whose `sight` this is sees
        it (`spine.visible`) - the whole book where nothing is named."""
        return ((portfolio is None or under(row['portfolio'], portfolio))
                and (agreement is None or row['agreement'] == agreement)
                and (clients is None or row.get('counterparty') in clients)
                and bool(visible([row], sight)))

    @staticmethod
    def rates(values, reporting, base):
        """`{currency: units of the reporting currency per unit}` off a values vector's spots - the
        engine's axis, every spot priced in `base`, which carries none of its own and is one."""
        spots = {name[len('FxRate.'):]: fields['Spot'] for name, fields in values.items()
                 if name.startswith('FxRate.') and '.' not in name[len('FxRate.'):]
                 and 'Spot' in fields}
        if base:
            spots[base] = 1.0
        return {} if not spots.get(reporting) else {
            currency: spot / spots[reporting] for currency, spot in spots.items() if spot}

    @classmethod
    def rates_on(cls, closes, stored, reporting, base):
        """`(day, lsn) -> rates` - what one unit of each currency is worth in the reporting one at
        the official close STANDING on a day, as the record stood at `lsn`: the last declared for
        that day or before it among those filed by then, a restatement standing over the close it
        restates. None where none stands. A movement of cash is booked at it as the record stood at
        the end of the business day the movement counts in, so a window read again later, or inside
        a longer one, books it at the same rate and a month stays the sum of its days."""
        ordered, answered = sorted(closes, key=lambda close: (close['date'], close['lsn'])), {}

        def on(day, lsn):
            found = None
            for close in ordered:
                if close['date'] > day:
                    break
                if close['lsn'] <= lsn:
                    found = close
            if found is None:
                return None
            if found['values_hash'] not in answered:
                answered[found['values_hash']] = cls.rates(
                    json.loads(stored(found['values_hash']).decode('utf-8')), reporting, base)
            return answered[found['values_hash']]

        return on

    @staticmethod
    def filed(before, after):
        """The movements of money filed between two positions of the record, off the `cash` rows at
        each: a new one whole, and a restated one as the filing it corrects taken back and the
        correction put in its place, each on its own value date."""
        standing, moved = {row['reference']: row for row in before}, []
        for row in after:
            old = standing.get(row['reference'])
            if old is None or old['lsn'] != row['lsn']:
                moved.extend(([dict(old, amount=-old['amount'])] if old else []) + [row])
        return moved

    @classmethod
    def recorded(cls, cut, costs, fills, known=None):
        """`day -> {through, dealt}` - what each position held at the END of a day, which is what
        it is paid what falls due that day on, and what it traded on it, as the record stood at the
        marks: `cut` is every marks' `(lsn, day)` and the end's, oldest first, `costs(lsn)` the
        `costs` rows there - `known` the ones already read - and `fills(after, until)` what was
        filed between two positions. A trade's price carries what its own day pays, so a seller
        that day is paid nothing and a buyer all of it, and an amendment restrikes the position for
        the whole of the business day it was filed in."""
        marked, answered = dict(cut), {}
        folded = {lsn: {(row['instrument'], row['agreement'], row['portfolio']): row['quantity']
                        for row in rows} for lsn, rows in (known or {}).items()}

        def held(lsn):
            if lsn not in folded:
                folded[lsn] = {} if lsn is None else {
                    (row['instrument'], row['agreement'], row['portfolio']): row['quantity']
                    for row in costs(lsn)}
            return folded[lsn]

        def on(day):
            if day not in answered:
                before = max((lsn for lsn, at in cut if at < day), default=None)
                through = max((lsn for lsn, at in cut if at <= day), default=None)
                dealt = {}
                if marked.get(through) == day:
                    for fill in fills(before or 0, through):
                        dealt[cls._where(fill)] = dealt.get(cls._where(fill), 0.0) + abs(
                            float(fill['quantity']))
                answered[day] = {'through': held(through), 'dealt': dealt}
            return answered[day]

        return on

    @classmethod
    def pnl(cls, start, end, record, scope):
        """The P&L between two marks, per position and in total.

        `start` and `end` are marks (`{day, lsn, units, rates}`), `rates` what one unit of each
        currency was worth in the reporting one at the close they stood on. `record` is what the
        record says of the window: `costs_start` and `costs_end`, `cash_start` and `cash_end`, the
        `costs` and `cash` rows at the two marks; `fills`, every fill between them with the business
        `day` it was traded on; `diary`, every payment the instruments held in the window announce,
        per unit, keyed as a settlement names it and answered as the record stands at the end;
        `last_days`, the last day of every instrument that has one; `positions_end`, the `positions`
        rows at the end, which carry the counterparty; `cut`, every marks' `(lsn, day)` and the
        end's, oldest first; `days`, the record on a day (`recorded`); `rates(day, lsn)`, the close
        standing on a day as filed by a position; `fills_between(after, until)`; and `tolerances`,
        the epsilons the record declares. `scope` narrows the rows, and below the whole book the
        unknowns to those naming an instrument of them or none, and the breaks to theirs.

        A PAYMENT THE DIARY DETERMINES IS BOOKED AT ITS AMOUNT: a settlement filed against it that
        moved another, beyond the `pnl` tolerance or in another currency, is a BREAK -
        `{key, instrument, determined, settled, difference, reference, value_date}` under `breaks` -
        and never an unknown.

        A MOVEMENT COUNTS IN THE WINDOW IT WAS FILED IN, and a payment the diary determines in the
        one it falls due in, each read as the record stood at the end of that business day: its
        rate, and whose it is. A late filing lands in the day it was filed and a day already struck
        never moves.

        THE CLOSE IS THE END OF THE DAY. The marks value a payment into the day it falls due, so a
        position's value at a close is its quantity times its unit mark less what the unit PAID that
        day: every payment the diary determines, and every one it does not that a settlement filed
        by the marks moved. One nothing settled by then is still the book's and stays in its value,
        and a window it leaves without the money is UNKNOWN, named.

        A position whose last day falls in the window CLOSES at its settlement value there: its mark
        at the end is nothing, its open basis is released to realised and its payoff arrives as the
        payment it is. One whose last day came on or before the start is worth nothing, realises
        only the money it moves, and it and one holding nothing stand only through that money - a
        payoff settling at T+2, a fee filed after the position closed.
        """
        def key_of(row):
            return row['instrument'], row['agreement'], row['portfolio']

        counterparties = {key_of(row): row.get('counterparty') for row in record['positions_end']}
        before, after = ({key_of(row): row for row in rows}
                         for rows in (record['costs_start'], record['costs_end']))
        window = {}
        for fill in record['fills']:
            window.setdefault(cls._where(fill), []).append(fill)
        settled, standing = {}, {'start': {}, 'end': {}}
        for movement in cls.filed(record['cash_start'], record['cash_end']):
            settled.setdefault((movement['kind'], movement['subject']), []).append(movement)
        for side in standing:
            for row in record['cash_' + side]:
                standing[side].setdefault((row['kind'], row['subject']), []).append(row)
        every = set(before) | set(after) | set(window)
        held = {}
        for key in every:
            held.setdefault(key[0], []).append(key)
        carried = {}
        for instrument, announced in record['diary'].items():
            for row in announced:
                carried.setdefault(row['key'], []).append((instrument, row))
        flows = dict(record, start=start['day'], end=end['day'], settled=settled, standing=standing,
                     held=held, carried=carried, owners={}, paid={})

        unknown, breaks = [], []
        for (kind, subject), moved in sorted(settled.items()):
            for movement in moved:
                # a settlement no row here announces, one against a payment nobody held, and a fee
                # nobody traded or held, belong to nothing
                if kind == 'payment' and subject not in carried:
                    unknown.append({'instrument': None, 'what': 'settlement {} was filed against a '
                                    "payment this window's diary does not announce".format(
                                        movement['reference'])})
                elif kind == 'payment':
                    instrument, row = carried[subject][0]
                    held_then = cls._holdings(row, movement, flows)
                    among = cls._carrying(subject, movement, flows)
                    if not any(held_then.get(other) for other in among):
                        unknown.append({'instrument': instrument, 'what': 'settlement {} moved a '
                                        'payment no position held when it fell due'.format(
                                            movement['reference'])})
                    elif row['determined'] and movement in standing['end'].get((kind, subject), ()):
                        # what the diary determines is booked; a settlement moving else is a break
                        due = row['amount'] * sum(held_then.get(other, 0.0) for other in among)
                        same, spine = movement['asset'] == row['currency'], package()
                        if not same or spine.policy.compare(
                                {spine.oracle.PNL: due}, {spine.oracle.PNL: movement['amount']},
                                record['tolerances']):
                            breaks.append({
                                'key': subject, 'instrument': instrument, 'determined': due,
                                'settled': movement['amount'],
                                'difference': movement['amount'] - due if same else None,
                                'reference': movement['reference'],
                                'value_date': movement['effective_time'][:10]})
                if kind == 'fee' and not cls._owners(movement, flows):
                    unknown.append({'instrument': subject, 'what': 'fee {} falls to no position - '
                                    'none under the book traded or held what it was filed '
                                    'on'.format(movement['reference'])})
        rows = []
        for key in sorted(every):
            instrument, agreement, portfolio = key
            if not cls.within({'agreement': agreement, 'portfolio': portfolio,
                               'counterparty': counterparties.get(key)}, **scope):
                continue
            q_start = before.get(key, {}).get('quantity', 0.0)
            q_end = after.get(key, {}).get('quantity', 0.0)
            traded = window.get(key, [])
            last = flows['last_days'].get(instrument)
            payments = cls._payments(key, flows, unknown)
            fees = cls._fees(key, flows, unknown)
            closed = last is not None and last <= start['day']
            # one holding nothing, or past its last day, stands only through money it still moves
            if ((closed or not (q_start or q_end)) and not traded and payments == 0.0
                    and fees == 0.0):
                continue
            ended = last is not None and last <= end['day']
            u_start, u_end = start['units'].get(instrument), end['units'].get(instrument)
            paid_start = (0.0, None) if closed else cls._paid(instrument, start, 'start', flows)
            paid_end = (0.0, None) if ended else cls._paid(instrument, end, 'end', flows)
            # a unit at a close, less what it paid that day; nothing once its last day has come
            mark_start = 0.0 if closed else cls._worth(u_start, paid_start[0])
            mark_end = 0.0 if ended else cls._worth(u_end, paid_end[0])
            value_start = 0.0 if not q_start else _times(q_start, mark_start)
            value_end = 0.0 if not q_end else _times(q_end, mark_end)
            for value, unit, paid, side in ((value_start, u_start, paid_start, 'start'),
                                            (value_end, u_end, paid_end, 'end')):
                if value is None:
                    unknown.append({'instrument': instrument, 'what': paid[1] if unit is not None
                                    else 'no mark at the {}{}'.format(side, (
                                        ' - the run could not price it' if side == 'end' else ''))})
            premiums, dealt = 0.0, sum(float(fill['quantity']) for fill in traded)
            for fill in traded:
                if fill.get('price') is None:
                    unknown.append({'instrument': instrument, 'what': 'fill {} was booked with no '
                                    'price'.format(fill['execution_reference'])})
                    premiums = None
                elif premiums is not None:
                    premiums -= float(fill['quantity']) * float(fill['price']) * float(
                        fill.get('rate', 1.0))
            moved = _less(value_end, value_start)
            # what the window's trades earned against the end: nothing where they net to nothing
            earned = 0.0 if not dealt else _times(dealt, mark_end)
            total = _sum(moved, premiums, payments, fees)
            rows.append({
                'instrument': instrument, 'agreement': agreement, 'portfolio': portfolio,
                'counterparty': counterparties.get(key), 'quantity_start': q_start,
                'quantity_end': q_end, 'unit_start': u_start, 'unit_end': u_end,
                'paid_start': paid_start[0], 'paid_end': paid_end[0],
                'value_start': value_start, 'value_end': value_end, 'premiums': premiums,
                'payments': payments, 'fees': fees,
                # what the book held moved, and what the window's trades earned against the end
                'existing': _less(moved, earned), 'trading': _sum(earned, premiums),
                'pnl': total, 'realised': cls._realised(
                    key, closed, ended and last > start['day'], (before, after),
                    (premiums, payments, fees), unknown)})
            rows[-1]['unrealised'] = _less(total, rows[-1]['realised'])
        # below the book an unknown of another node's instrument is the book's; one naming none
        # cannot be placed, so it stays on every read
        if not cls.within({'portfolio': (scope.get('portfolio') or '').split('/')[0],
                           'agreement': None, 'counterparty': None}, **scope):
            mine = {row['instrument'] for row in rows} | {None}
            unknown = [entry for entry in unknown if entry['instrument'] in mine]
            breaks = [entry for entry in breaks if entry['instrument'] in mine]
        unknown = [json.loads(named) for named in sorted({json.dumps(row, sort_keys=True)
                                                          for row in unknown})]
        return {'rows': rows, 'unknown': unknown, 'breaks': breaks,
                'realised_method': 'average cost',
                'total': {figure: _sum(*(row[figure] for row in rows)) for figure in cls.FIGURES},
                'complete': not unknown and all(
                    row[figure] is not None for row in rows
                    for figure in cls.FIGURES if figure not in cls.SPLIT)}

    @classmethod
    def explain(cls, rows, last_days, start, end, run):
        """WHY the positions a window started with made what they made - the P&L `rows` of one scope
        between the days `start` and `end` - never assembling the P&L out of its pieces.

        `existing` is theirs, and it is the CARRY of the start's book rolled to the end's day at the
        start's own quotes - a position whose last day fell in the window closing at nothing, which
        time alone did - plus the MARKET, every risk factor's move between the two closes times the
        start's own sensitivity to it, plus the RESIDUAL nothing here explains. `run(quantities)`
        values the positions still standing at the end - `{instrument: quantity}` - and answers
        `(opened, closed, rolled, note)`: the start's value and sensitivities, the end's levels of
        the same factors, the start's value rolled to the end's day and why its quotes did not
        connect, where they did not. The runs value a day's payments into that day as the marks do,
        and the carry takes out what a row's values take out. Reserves are not carried, so none is
        explained.
        """
        quantities, carry, existing, paid, unknown = {}, 0.0, 0.0, {}, []
        for row in rows:
            existing = _sum(existing, row['existing'])
            if row['existing'] is None:
                unknown.append({'instrument': row['instrument'], 'what': 'what it held moved is '
                                'not known, so neither is what explains it'})
            last = last_days.get(row['instrument'])
            if not row['quantity_start'] or last is not None and last <= start:
                continue
            if last is not None and last <= end:
                carry = _sum(carry, row['existing'])
            else:
                quantities[row['instrument']] = (quantities.get(row['instrument'], 0.0)
                                                 + row['quantity_start'])
                paid[row['instrument']] = _less(row['paid_start'], row['paid_end'])
        answer = {'existing': existing, 'carry': carry, 'market': 0.0, 'residual': None,
                  'reserves': None, 'factors': [], 'unknown': unknown, 'note': None}
        quantities = {instrument: held for instrument, held in quantities.items() if held}
        if quantities:
            try:
                opened, closed, rolled, answer['note'] = run(quantities)
            except Exception as refused:
                answer['unknown'].append({'instrument': None, 'what': 'the positions would not '
                                          'value for the explain: {}'.format(refused)})
                answer['carry'] = answer['market'] = None
                return answer
            taken = _sum(*(_times(quantities[instrument], paid[instrument])
                           for instrument in sorted(quantities)))
            if taken is None:
                answer['unknown'].append({'instrument': None, 'what': 'what a position paid on the '
                                          'day of a marks is not known, so neither is its carry'})
            answer['carry'] = _sum(carry, rolled - opened['value'], taken)
            answer['factors'], moved = cls._moved(opened, closed)
            answer['unknown'].extend(moved)
            answer['market'] = sum(row['pnl'] for row in answer['factors']
                                   if row['pnl'] is not None)
        answer['residual'] = _less(_less(existing, answer['carry']), answer['market'])
        return answer

    @classmethod
    def _moved(cls, opened, closed):
        """`(rows, unknown)` - every risk factor the start's sensitivities read that moved, with its
        move to the end and what that move made: per quote the factors are built from, and per
        factor for the rest - the factors a quote stands for being explained by the quote. Largest
        first."""
        rows, unknown = [], []
        ends = {(row['block'], row['quote']): row['level'] for row in closed['quotes']}
        for row in opened['quotes']:
            rows.append(dict({'block': row['block'], 'quote': row['quote']}, **cls._move(
                row, ends.get((row['block'], row['quote'])))))
        ends = {(row['factor'], tuple(row.get('tenor', ()))): row['level']
                for row in closed['greeks']}
        for row in opened['greeks']:
            if row['factor'] not in opened['quoted']:
                named = {'factor': row['factor'], 'tenor': row.get('tenor', [])}
                rows.append(dict(named, **cls._move(row, ends.get((row['factor'],
                                                                   tuple(named['tenor']))))))
        # a factor that did not move explains nothing, and one whose move is unknown is named
        rows = [row for row in rows if row['delta'] and row['move'] != 0.0]
        for row in rows:
            if row['pnl'] is None:
                unknown.append({'instrument': None, 'what': 'no level at the end for {}'.format(
                    ' '.join(str(part) for part in (row.get('block') or row['factor'],
                                                    row.get('quote') or row['tenor']) if part))})
        return sorted(rows, key=lambda row: (-abs(row['pnl'] or 0.0), json.dumps(row))), unknown

    @staticmethod
    def _move(row, end):
        """One factor's sensitivity at the start, its level at each end and what the move made."""
        move = None if end is None or row['level'] is None else end - row['level']
        return {'delta': row['value'], 'start': row['level'], 'end': end, 'move': move,
                'pnl': None if move is None else row['value'] * move}

    @staticmethod
    def _where(fill):
        """The position a fill moved: its instrument, under its agreement - its netting set where it
        names none - in its portfolio, else its book."""
        return (fill['instrument'], fill.get('agreement') or fill['netting_set'],
                fill.get('portfolio') or fill['book'] or '')

    @staticmethod
    def _filed(lsn, flows):
        """`(lsn, day)` of the business day a filing at `lsn` counts in: the first marks at or after
        it, else the window's end."""
        return next(((at, day) for at, day in flows['cut'] if at >= lsn), flows['cut'][-1])

    @staticmethod
    def _falls(day, flows):
        """The position of the marks of the business day `day` falls in: the first marked on or
        after it, else the window's end."""
        return min(((at_day, at) for at, at_day in flows['cut'] if at_day >= day),
                   default=(None, flows['cut'][-1][0]))[1]

    @staticmethod
    def _realised(key, closed, ending, costs, cash, unknown):
        """What a position realised in the window, at average cost: what its reductions realised,
        what it was paid and charged, and the open basis its last day released - or, past its last
        day, the money it moved alone, the basis having been released on that day. UNKNOWN, named,
        where a reduction closed at no price or against a lot booked with none, or the basis
        released was one.
        """
        premiums, payments, fees = cash
        if closed:
            return _sum(premiums, payments, fees)
        before, after = (rows.get(key, {}) for rows in costs)
        if after.get('unpriced_reductions', 0) != before.get('unpriced_reductions', 0):
            unknown.append({'instrument': key[0], 'what': 'a reduction closed at no price, or '
                            'against a lot booked with none, so what it realised is not known'})
            return None
        released = after.get('basis', 0.0) if ending else 0.0
        if released is None:
            unknown.append({'instrument': key[0], 'what': 'what it held at its last day was booked '
                            'with no price, so the basis released there is not known'})
        return _sum(after.get('realised', 0.0) - before.get('realised', 0.0), payments, fees,
                    _less(0.0, released))

    @classmethod
    def _paid(cls, instrument, marks, side, flows):
        """`(per unit, why)` - what one unit of `instrument` paid on the day `marks` value it, which
        the unit mark still carries, in the reporting currency at the close the marks stood on:
        every payment the diary determines falling due that day, and every one it does not that a
        settlement filed by the marks moved, per unit held when it fell due. None, with why, where
        that is not known."""
        if (instrument, side) not in flows['paid']:
            total, why, day = 0.0, None, marks['day']
            for row in flows['diary'].get(instrument, []):
                if row['determined']:
                    moved = [(row['amount'], row['currency'])] if row['due_date'] == day else []
                elif row['due_date'] < day:
                    moved = []
                else:
                    moved = []
                    for movement in flows['standing'][side].get(('payment', row['key']), []):
                        among = cls._carrying(row['key'], movement, flows)
                        held = cls._holdings(row, movement, flows)
                        if not any(held.get(other) for other in among):
                            # held by nobody here when it fell due: named in the window filed
                            continue
                        net = sum(held.get(other, 0.0) for other in among)
                        if not net:
                            why = 'settlement {} moved a payment of positions netting to ' \
                                  'nothing, so what one unit was paid is not known'.format(
                                      movement['reference'])
                            break
                        moved.append((movement['amount'] / net, movement['asset']))
                for amount, currency in moved:
                    if currency not in marks['rates']:
                        why = 'the close the {} marks stood on carries no spot for {}'.format(
                            day, currency)
                        continue
                    total += amount * marks['rates'][currency]
                if why is not None:
                    break
            flows['paid'][(instrument, side)] = (None, why) if why else (total, None)
        return flows['paid'][(instrument, side)]

    @staticmethod
    def _worth(unit, paid):
        """A unit at a close less what it paid that day, or None where either is not known."""
        return None if unit is None or paid is None else unit - paid

    @classmethod
    def _holdings(cls, row, movement, flows):
        """What each position held at the end of the day the payment `row` fell due, as the record
        stood when `movement` settled it: of the day it was filed where it was filed before it fell
        due."""
        _, day = cls._filed(movement['lsn'], flows)
        return flows['days'](min(row['due_date'], day))['through']

    @classmethod
    def _payments(cls, key, flows, unknown):
        """What this position was paid in the window: the diary's amount times what it held at the
        end of the day it fell due, for every payment the diary determines falling due in the
        window; and its share - by what each position held then - of what every settlement FILED in
        the window moved against a row the diary leaves undetermined. Each at the close standing on
        its own day as the record stood at the end of the business day it counts in.

        An undetermined payment is the book's until something settles it: it leaves the book's value
        the day after it falls due, or on the position's last day where it is due then or after, and
        a window it leaves before anything settled it is UNKNOWN, named. A settlement over positions
        netting to nothing cannot be shared, a movement naming no agreement, and is named too."""
        total, last = 0.0, flows['last_days'].get(key[0])
        for row in flows['diary'].get(key[0], []):
            due = row['due_date']
            if row['determined']:
                if flows['start'] < due <= flows['end']:
                    mine = flows['days'](due)['through'].get(key, 0.0)
                    if mine:
                        total = _sum(total, cls._converted(
                            mine * row['amount'], row['currency'], due, cls._falls(due, flows),
                            flows, key[0], unknown))
                continue
            leaves = (flows['start'] <= due < flows['end'] if last is None or due < last
                      else flows['start'] < due <= flows['end'])
            if (leaves and row['state'] != SETTLED
                    and flows['days'](due)['through'].get(key, 0.0)):
                unknown.append({'instrument': key[0], 'what': 'the {} payment due {} is not '
                                'determined and nothing settled it'.format(row['leg'], due)})
                return None
            for movement in flows['settled'].get(('payment', row['key']), []):
                among = cls._carrying(row['key'], movement, flows)
                held = cls._holdings(row, movement, flows)
                mine, net = held.get(key, 0.0), sum(held.get(other, 0.0) for other in among)
                if not mine or key not in among:
                    continue
                if not net:
                    unknown.append({'instrument': key[0], 'what': 'settlement {} moved a payment '
                                    'of positions netting to nothing - a settlement names no '
                                    'agreement, so whose it is is not known'.format(
                                        movement['reference'])})
                    return None
                total = _sum(total, cls._converted(
                    movement['amount'] * mine / net, movement['asset'],
                    movement['effective_time'][:10], cls._filed(movement['lsn'], flows)[0], flows,
                    key[0], unknown))
        return total

    @staticmethod
    def _booked(positions, movement):
        """The positions a movement is shared across: those under the book it was filed for, every
        one where it names none."""
        book = movement.get('book')
        return [key for key in positions if not book or under(key[2], book)]

    @classmethod
    def _carrying(cls, key, movement, flows):
        """The positions a movement against the row `key` is shared across: those of every
        instrument whose diary carries the row - a restruck structure and the one it replaced carry
        an unchanged leg alike - under the book it was filed for."""
        return cls._booked([position for instrument, _ in flows['carried'].get(key, ())
                            for position in flows['held'].get(instrument, [])], movement)

    @classmethod
    def _owners(cls, movement, flows):
        """`{position: weight}` - the positions a fee falls to, among those under the book it was
        filed for: those that traded what it was filed on, on its own day - its value date, or the
        day it was filed where it is dated after that - else those holding it through that day, else
        those that traded it on the record up to its filing, else those holding it when it was filed
        - terms an amendment struck being traded by no fill. Empty where none did."""
        found = (movement['reference'], movement['lsn'])
        if found not in flows['owners']:
            among = cls._booked(flows['held'].get(movement['subject'], []), movement)
            filed_on, weights = cls._filed(movement['lsn'], flows)[1], {}
            if among:
                record = flows['days'](min(movement['effective_time'][:10], filed_on))
                weights = {other: record['dealt'].get(other, 0.0) for other in among}
                if not any(weights.values()):
                    weights = {other: abs(record['through'].get(other, 0.0)) for other in among}
                if not any(weights.values()):
                    weights = {}
                    for fill in flows['fills_between'](0, movement['lsn']):
                        if cls._where(fill) in among:
                            weights[cls._where(fill)] = weights.get(cls._where(fill), 0.0) + abs(
                                float(fill['quantity']))
                if not any(weights.values()):
                    weights = {other: abs(flows['days'](filed_on)['through'].get(other, 0.0))
                               for other in among}
            flows['owners'][found] = {other: weight for other, weight in weights.items() if weight}
        return flows['owners'][found]

    @classmethod
    def _fees(cls, key, flows, unknown):
        """This position's share of the fees FILED on its instrument in the window (`_owners`), at
        the close standing on each fee's value date as the record stood at the end of the business
        day it was filed in. Whose a fee is depends on its own day and its filing, never on the
        window it lands in."""
        total = 0.0
        for movement in flows['settled'].get(('fee', key[0]), []):
            weights = cls._owners(movement, flows)
            if weights.get(key):
                total = _sum(total, cls._converted(
                    weights[key] / sum(weights.values()) * movement['amount'], movement['asset'],
                    movement['effective_time'][:10], cls._filed(movement['lsn'], flows)[0], flows,
                    key[0], unknown))
        return total

    @staticmethod
    def _converted(amount, currency, day, lsn, flows, instrument, unknown):
        """`amount` of `currency` in the reporting currency at the close standing on `day` as filed
        by `lsn`, or UNKNOWN - named - where no close stands on it or it carries no spot for the
        currency.
        """
        rates = flows['rates'](day, lsn)
        if rates is not None and currency in rates:
            return amount * rates[currency]
        unknown.append({'instrument': instrument, 'what': (
            'no official close on or before {} to report what moved in {}'.format(day, currency)
            if rates is None else 'the close standing on {} carries no spot for {}'.format(
                day, currency))})
        return None


class Collateral:
    """The collateral calls on a marked close - what each agreement's CSA asks of the exposure under
    it, less what is held under it.

    THE EXPOSURE IS THE ONE THE NETTING SET'S OWN RECURSION READS, off the P&L's numbers: the value
    `PnL.pnl` gives each position under the agreement at the close - its quantity times its unit
    mark less what the unit paid that day, a position held through its last day as the set holds it
    until its cash moves - with that day's payment added back where the set holds it (the engine's
    `Exclude_Paid_Today` off, its default), crossed into the agreement currency at the close's own
    spots; a mark nobody has is named rather than read as zero. THE ARITHMETIC IS THE RECORD'S:
    `derivus_spine.projections.CSA`, one formula the service and the oracle run and the engine's own
    recursion at one date.

    Pure over plain data: the P&L over the close alone, the agreements declared, the `cash` rows.
    """

    @staticmethod
    def paid_today(document):
        """Whether the netting sets of `document` hold the day's payments in the exposure their
        recursion reads: `Exclude_Paid_Today` off - the engine's default - as a set reads it off the
        job's `Valuation Configuration`."""
        options = (document['Calc']['MergeMarketData'].get('ExplicitMarketData') or {}).get(
            'Valuation Configuration') or {}
        return not options.get('NettingCollateralSet', {}).get('Exclude_Paid_Today', False)

    @staticmethod
    def exposure(agreement, valued, rates, currency, today):
        """What the positions under `agreement` are worth at a close in `currency`: the value the
        P&L gives each there (`valued`, `PnL.pnl`'s answer over that close alone) and, where the set
        holds the day's payments (`today`), what each paid that day, crossed at the close's `rates`
        (`PnL.rates`) - None where one of them is not known or the close carries no spot for it."""
        rows = [row for row in valued['rows'] if row['agreement'] == agreement]
        values = [row['value_end'] for row in rows] + [
            _times(row['quantity_end'], row['paid_end']) for row in rows if today]
        return (None if None in values or currency not in rates
                else sum(values, 0.0) / rates[currency])

    @staticmethod
    def collateralised(agreements):
        """The declared `agreements` whose terms collateralise - the ones a close has calls on."""
        return [row for row in agreements
                if package().projections.CSA.of(row['terms']) is not None]

    @staticmethod
    def seen(agreements, positions, sight, verb=None):
        """The `agreements` a seat whose `sight` this is reads: those a position standing among
        `positions` sits under at a node it holds `verb` at - any verb where none is named - and
        every one where nothing narrows the seat, one nothing is held under included
        (`spine.visible`)."""
        held = {row['agreement'] for row in visible(positions, sight, verb=verb) if row['quantity']}
        # a row sitting nowhere is seen only where nothing narrows the seat
        everywhere = bool(visible([{}], sight, verb=verb))
        return [row for row in agreements if everywhere or row['agreement'] in held]

    @classmethod
    def calls(cls, close, valued, agreements, movements, until):
        """`[{agreement, entity, currency, exposure, fx, held, margin, required, balance, call,
        direction, minimum_transfer, unknown}]` - the call of each of the `collateralised`
        agreements on the marks `close` (`{day, rates, document}`), in the agreement's currency:
        `valued` is the P&L over that close alone, `movements` the `cash` rows and `until` the value
        date a balance is held as of - every movement filed, whatever its value date, where None.

        `held` and `margin` are the two balances under it per asset, and the call reads the first
        alone; `fx` is the close's spots in the agreement's currency, so with the exposure a call
        replays off the record. What nobody can know - a mark, a spot the close carries none of, a
        dial the terms state none of - is named under `unknown`, and the call is null over it.
        """
        record, rates, answer = package().projections.CSA, close['rates'], []
        balances, today = record.held(movements, until), cls.paid_today(close['document'])
        for declared in agreements:
            dials = record.of(declared['terms'])
            agreement, currency = declared['agreement'], dials['Agreement_Currency']
            held = balances.get(agreement, record.NOTHING)
            fx = {} if currency not in rates else {
                other: rate / rates[currency] for other, rate in rates.items()}
            mine = {row['instrument'] for row in valued['rows'] if row['agreement'] == agreement}
            # an asset held at nothing needs no spot to be worth nothing, nor a currency never named
            uncrossed = {currency} - set(fx) - {None, ''} | {
                asset for asset, amount in held['collateral'].items() if amount and asset not in fx}
            unknown = [entry for entry in valued['unknown'] if entry['instrument'] in mine] + [
                {'instrument': None, 'what': 'the close carries no spot for {}'.format(other)}
                for other in sorted(uncrossed)] + [
                {'instrument': None, 'what': 'the terms of {} state no {}'.format(agreement, dial)}
                for dial in record.DIALS + ('Agreement_Currency',) if dials[dial] in (None, '')]
            value = cls.exposure(agreement, valued, rates, currency, today)
            answer.append(dict(
                dict.fromkeys(record.WORKED) if unknown else record.call(
                    value, held['collateral'], dials, fx),
                agreement=agreement, entity=declared['entity'], currency=currency, exposure=value,
                fx=fx, held=held['collateral'], margin=held['margin'], unknown=unknown))
        return answer
