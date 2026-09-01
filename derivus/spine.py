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
from .config import Config, CustomJsonEncoder, as_json, tables_of

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
        from derivus_spine import firmness, policy, verbs  # noqa: F401  attribute access below
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


def canonical(obj):
    """The bytes the engine hashes `obj` as: sorted keys, tight separators, `CustomJsonEncoder`.

    This is why an engine hash can be a blob address: a blob is named by the SHA-256 of its bytes
    and the engine's hashes are the SHA-256 of exactly these, so `values_hash` and the address of
    the values vector are one number. The spine canonicalises its own objects under RFC 8785 and
    addresses whatever bytes it is handed, so the two schemes compose.
    """
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      cls=CustomJsonEncoder).encode('utf-8')


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


def declare_market(name, values, actor_name=None, effective_time=None):
    """Point the market `name` at a values vector. `official` demands `mark` scope, which the writer
    enforces - a check here would be a second place to get it wrong."""
    verbs = package().verbs
    with writing() as log:
        return verbs.declare_market(log, actor(actor_name), name, values,
                                    effective_time=effective_time)


def file_quote(quote_id, structure, plan_hash, values, solved, edge, request=None,
               actor_name=None, book_name=None, effective_time=None):
    """File a quote with both hashes pinned - the values vector it was struck on and the book plan
    its marginal charge was solved against."""
    verbs = package().verbs
    with writing() as log:
        return verbs.file_quote(log, actor(actor_name), quote_id, structure, plan_hash, values,
                                solved, edge, request=request, book=book_name,
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
