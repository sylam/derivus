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

"""The one seam between the engine and the book of record - thin, lazy, and off by default.

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

Every event carries a pseudonymous subject reference and the service has no auth of its own, so the
actor is what the caller names, else `DV_SPINE_ACTOR`, else a refusal - inventing one would put a
name in the record nobody chose.

One writer: the spine claims its home exclusively at the first append, so `writing()` holds a
process-wide lock for the length of one act and closes the handle after it. And every `SpineRefusal`
is re-raised as `SpineRefused`, a `ValueError` carrying the spine's sentence unedited, so the book
verbs' `except ValueError -> 422` handlers surface the library's own wording.
"""
import contextlib
import json
import os
import threading

from ._version import __version__
from .schema import tables_of
from .config import Config, CustomJsonEncoder, as_json

#: The whole switch, and the actor beside it. Read per call, like `DV_HOME` one module over.
SPINE_HOME = 'DV_SPINE_HOME'
SPINE_ACTOR = 'DV_SPINE_ACTOR'

#: The three attestation lanes, respelled here so a call site can name one without paying for the
#: spine import. The refusal wording still comes from `derivus_spine.verbs.check_lane`.
TELEMETRY = 'telemetry'
CURIOSITY = 'curiosity'
STANDING = 'standing'
LANES = (TELEMETRY, CURIOSITY, STANDING)

#: What a tree without the extra is told, with the line that fixes it.
NO_PACKAGE = ('the book of record is not installed on this box ({}) - {} names a spine home, so '
              'this verb records to it; `pip install derivus[enterprise]`, or unset {} to run the '
              'edge exactly as it ran before')
#: What a configured home with no actor is told. An unconfigured home says nothing at all: that is
#: the default posture rather than an error.
NO_ACTOR = ('no actor for this append: every event carries the pseudonymous subject reference that '
            'submitted it, so name one on the request or set {} - a record that invented an actor '
            'would be a record naming somebody who never spoke')

#: One writer, one act at a time. The spine answers `WriterBusy` to a second holder of the home;
#: this lock keeps a request thread and the compute worker from meeting that over their own book.
_WRITER = threading.Lock()


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

    `firmness`, `policy` and `verbs` are imported by name because a submodule is only an attribute
    of its package once something has imported it, and the package's own surface stays the truth
    layer.
    """
    try:
        import derivus_spine
        from derivus_spine import (capability, firmness, policy,  # noqa: F401  attribute
                                   projections, verbs)            # noqa: F401  access below
    except ImportError as absent:
        raise SpineRefused(NO_PACKAGE.format(absent, SPINE_HOME, SPINE_HOME))
    return derivus_spine


def actor(named=None):
    """Who this append is attributed to: the caller's name, else `DV_SPINE_ACTOR`, else a refusal."""
    subject = named or os.environ.get(SPINE_ACTOR)
    if not subject:
        raise SpineRefused(NO_ACTOR.format(SPINE_ACTOR))
    return subject


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


def _rows(projector, lsn=None):
    """One projector's rows at `lsn`, folded off the home - what every read below is made of."""
    def fold(log):
        projections = package().projections
        named = projections.PROJECTORS[projector]
        return named.rows(projections.fold(log, named, lsn=lsn))

    return folded(fold)


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


def compiled_job(document, lsn=None, strict=True):
    """The job the plan hashes, with every declared observation filled from the record at `lsn`.
    Unchanged where no home is configured.

    `strict` is what separates a PLAN from a READ. A plan may not read a fixing nobody vouched for,
    so an index it compiles against that no `fixings` policy orders refuses by name. A read fills
    what the declared orders can fill and leaves the rest of the document as the desk wrote it -
    the row that named the unresolved index says so on itself, and no GET fails for it.

    The plan is terms PLUS the observations the record holds, so an auditor recompiles from what
    was printed rather than from what somebody typed. A deal type's own table is where its
    observations live and `derivus.diary` is the one table saying which - a fixing the record holds
    for a date the deal names overwrites the cell, and a date it holds nothing for is left standing.

    NOTHING AFTER THE BASE DATE is filled. A `fixing_observed` carries its date as text, so a print
    dated forward is a legal fact; writing one onto a monitoring row would price a barrier as
    already observed on a day that has not happened.
    """
    import copy

    from .diary import index_named

    if not configured() or not _observing(document):
        return document
    filled = copy.deepcopy(document)
    observing = [(deal, terms, index_named(deal, terms)) for deal, terms in _observing(filled)]
    named = {index for _, _, index in observing if index}
    observed = (fixings(lsn, indices=named) if strict else observations(lsn, named)[0])
    base = _base_day(filled)
    for deal, terms, index in observing:
        deal[terms.table] = [_observed(row, terms, observed.get((index, _day(row)))
                                       if _day(row) <= base else None)
                             for row in deal[terms.table]]
    return filled


def _base_day(document):
    """The ISO day this job is valued as of - what a print may not be dated after."""
    return _day([document['Calc']['Calculation']['Base_Date']])


def _observing(document):
    """Every `(deal block, terms)` of this job whose type declares an observation table the
    document actually carries rows in. A document that is not a job observes nothing."""
    from .diary import TERMS
    from .schema import walk_job_deals

    try:
        nodes = list(walk_job_deals(document))
    except (KeyError, TypeError, ValueError):
        return []
    deals = [node['Instrument']['.Deal'] for _, node in nodes]
    return [(deal, TERMS[deal['Object']]) for deal in deals
            if TERMS.get(deal.get('Object')) is not None
            and TERMS[deal['Object']].table and deal.get(TERMS[deal['Object']].table)]


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
    """The values vector this context would run against, as the bytes its `values_hash` names."""
    return canonical(context.market_patch())


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
    from . import Context

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
         actor_name=None, book_name=None, effective_time=None):
    """Book a fill against the canonical instrument `deal`, returning the spine's envelope.

    `deal` is canonicalised through the engine's encoder and its hash is the instrument id, so
    booking the same strike twice registers one instrument and files two events against it.
    `execution_reference` has no default: it is what makes a retry the same fact.
    """
    verbs = package().verbs
    with writing() as log:
        return verbs.book(log, actor(actor_name), canonical(deal), quantity, counterparty,
                          netting_set, execution_reference, book=book_name,
                          effective_time=effective_time)


def amend(deal, amended_to, actor_name=None, book_name=None, effective_time=None):
    """Record that these terms became those terms - a new instrument hash linked to the old one."""
    verbs = package().verbs
    with writing() as log:
        return verbs.amend(log, actor(actor_name), canonical(deal), canonical(amended_to),
                           book=book_name, effective_time=effective_time)


def apply_lifecycle(event_type, body, actor_name=None, book_name=None, effective_time=None):
    """File an election, a fixing observation or a determination. Anything consequence-shaped is
    refused - see `derivus_spine.verbs.apply_lifecycle`."""
    verbs = package().verbs
    with writing() as log:
        return verbs.apply_lifecycle(log, actor(actor_name), event_type, body, book=book_name,
                                     effective_time=effective_time)


def approve(plan_hash, actor_name=None, book_name=None, effective_time=None):
    """Sign a plan hash. One seat approving one plan twice is one fact, coalescing onto the LSN it
    already has; an amended plan is a different hash and so is a different signature."""
    verbs = package().verbs
    with writing() as log:
        return verbs.approve(log, actor(actor_name), plan_hash, book=book_name,
                             effective_time=effective_time)


def reject(plan_hash, reason, actor_name=None, book_name=None, effective_time=None):
    """Refuse a plan hash with the reason on the row - the check, what it measured and the bound."""
    verbs = package().verbs
    with writing() as log:
        return verbs.reject(log, actor(actor_name), plan_hash, reason, book=book_name,
                            effective_time=effective_time)


def declare_market(name, values, actor_name=None, effective_time=None):
    """Point the market `name` at a values vector. `official` demands `mark` scope, which the writer
    enforces - a check here would be a second place to get it wrong."""
    verbs = package().verbs
    with writing() as log:
        return verbs.declare_market(log, actor(actor_name), name, values,
                                    effective_time=effective_time)


def declare_close(market, values, actor_name=None, effective_time=None):
    """Declare the official close on `market` over a values vector. A second close supersedes the
    first rather than editing it, so a day restated is two facts and both stay readable."""
    verbs = package().verbs
    with writing() as log:
        return verbs.declare_close(log, actor(actor_name), market, values,
                                   effective_time=effective_time)


def file_quote(quote_id, structure, plan_hash, values, solved, edge, request=None, ticket=None,
               actor_name=None, book_name=None, effective_time=None):
    """File a quote with both hashes pinned - the values vector it was struck on and the book plan
    its marginal charge was solved against - and, where the caller computed one, the `ticket` plan
    an approval of this quote would sign."""
    verbs = package().verbs
    with writing() as log:
        return verbs.file_quote(log, actor(actor_name), quote_id, structure, plan_hash, values,
                                solved, edge, request=request, ticket=ticket, book=book_name,
                                effective_time=effective_time)


def complete_run(claim, lane, job, values, result, actor_name=None, book_name=None):
    """Attest a standing run at birth. A telemetry or curiosity lane mints nothing and should not
    reach here; the verb refuses one that does rather than dropping it quietly."""
    verbs = package().verbs
    with writing() as log:
        return verbs.complete_run(log, actor(actor_name), lane, claim, job, values, result,
                                  book=book_name)


def pin_result(claim, job, values, result, actor_name=None, book_name=None, effective_time=None,
               execute=None):
    """Promote a replay claim this hub did not witness, through re-execution or a cache hit."""
    verbs = package().verbs
    with writing() as log:
        return verbs.pin_result(log, actor(actor_name), claim, job, values, result,
                                execute or executor(), book=book_name,
                                effective_time=effective_time)


def tiers_policy():
    """The tiers document standing in the record, or None where this home declared none.

    A fold like every other policy read, so what a ticket would be routed by is answerable without
    queueing behind a booking.
    """
    return folded(package().policy.tiers_in_force)


def quotes(lsn=None):
    """Every quote the record holds at `lsn`, oldest first: who struck it, what it pinned, what it
    solved, and the ticket plan an approval of it would sign."""
    return _rows('quotes', lsn)


def verdicts(plan_hash, lsn=None):
    """The verdicts filed against `plan_hash` at `lsn`, oldest first.

    An empty list is a plan nobody has ruled on, which is not the answer a rejected plan gives - the
    two have different remedies, so the caller reads the list rather than a boolean.
    """
    filed = _rows('decisions', lsn)['plans']
    return next((row['verdicts'] for row in filed if row['plan_hash'] == plan_hash), [])


def resolve_market(name, actor=None, process=None):
    """The values vector standing under the market `name`: `{name, values_hash, values}`.

    The composition nothing performed before - the `markets` fold names it, the store holds the
    bytes, `read_values` turns them back into what `patch_market` takes. A name nobody declared
    REFUSES rather than falling back on the book's own market, which is the whole point of binding a
    process to a market by name.

    A NAME IS RESOLVED TO ITS LATEST DECLARATION, across the declarations of the name and the
    official closes of it alike - a close is one way of declaring what a market stands on, and a
    reader that took the name's own row would answer yesterday's board on a market the desk has
    since closed. Latest by the fold's own as-of key, `(effective_time, lsn)`, so a close backdated
    behind the mark in force is on the platter and does not displace it.

    `private/<subject>/<name>` is one seat's own and resolves for that SUBJECT alone - the name
    carries its owner and `verbs.declare_market` refuses a private name whose subject is not the
    seat declaring it, so ownership reads the same off the name and off the fold. The surveillance
    and admin read of one is an entitlement class this record does not yet classify, so nobody else
    reaches it here. The two rules are checked independently: with `process` named, the market must
    ALSO be the one the tiers policy designates for it, which no private name ever is.
    """
    asking = actor or os.environ.get(SPINE_ACTOR)

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
        return {'name': name, 'values_hash': latest['values_hash'],
                'values': read_values(log.store.get(latest['values_hash']))}

    return folded(fold)


def firmness_policy():
    """The two staleness windows standing in the record - declared, or the spine's stated defaults.

    A fold rather than a write, so asking what the windows are never queues behind a booking or
    turns an approval into a `WriterBusy`.
    """
    return folded(package().policy.firmness_in_force)


def check_firmness(pinned, current, ages, quote_id=None, policy=None):
    """Whether this quote is still firm in both dimensions: the verdict, or a refusal naming each
    dimension that failed and its remedy.

    A pure call into `derivus_spine.firmness` over plain data. `policy` is read out of the record
    when the caller hands none in, which is the ordinary case - the windows are policy data, so they
    live in the log rather than in a constant here.
    """
    windows = firmness_policy() if policy is None else policy
    with translating():
        return package().firmness.check(pinned, current, ages, windows, quote_id=quote_id)
