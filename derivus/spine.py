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

`derivus_spine` is the append-only log the desk's numbers are recorded in. It imports stdlib and
`cryptography` and nothing else - never the engine - which is what lets a book of record be
verified on a machine that cannot price a trade. This module is the seam from the other side, and
it is deliberately the ONLY place under `derivus/` that knows the spine exists.

THREE RULES SHAPE IT AND ALL THREE ARE VISIBLE IN THE CODE.

The engine stays a pure function from facts to numbers. So every verb below is a DELEGATOR: it
canonicalises through the engine's own encoder, hands the spine plain bytes and plain numbers, and
holds no storage logic of its own. The reconciliation with the brief's "booking verbs on Context"
is dependency injection - the logic lives in `derivus_spine.verbs`, and `pin_result` hands that
layer a callable rather than letting it reach for a pricer it must never import.

The spine is an EXTRA, so the import is lazy and inside the function. `import derivus` must not need
`cryptography`, exactly as it must not need `fastapi` - the `service.py` precedent, applied for the
same reason. A tree without the extra refuses BY NAME with the install line in it.

NO SPINE HOME CONFIGURED MEANS BIT-IDENTICAL. `DV_SPINE_HOME` is the whole switch, it is read on
every call rather than captured at import, and it does NOT fall back to the CLI's `~/.derivus_spine`
default - a desk box that once ran `DV_Spine init` must not silently start recording, so the engine
activates only where a deployment said so out loud. Unset, every call site below is a no-op and the
edge behaves to the byte as it did before this module existed. Set but not minted is a NAMED
refusal, never a quiet fall-back, because "the deployment configured a home that is not there" and
"the deployment configured none" are different facts.

WHO IS WRITING. Every event carries a pseudonymous subject reference, and the service has no auth
(it is a trusted-network deployment behind something that terminates one). So the actor is what the
caller names, else `DV_SPINE_ACTOR`, else a refusal - a home that is configured and an actor that is
not is a booking nobody could be held to, and inventing one would put a name in the record that
nobody chose.

ONE WRITER. The spine takes an exclusive claim on its home at the first append, so two threads of
one service reaching for it would meet `WriterBusy`. `writing()` holds a process-wide lock for the
length of one act and closes the handle after it, which is the same discipline the CLI's verbs use:
a home is a directory, not a session.

REFUSALS TRAVEL VERBATIM. Every `SpineRefusal` is re-raised as `SpineRefused`, which is a
`ValueError` carrying the spine's own sentence unedited - so the book verbs' existing
`except ValueError -> 422` handlers surface the library's wording rather than a paraphrase
competing with it, which is the rule the CLI already follows on stderr.
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

#: The three attestation lanes, spelled here because the engine names them at call sites that must
#: not pay for the spine import to do it - `capability.py`'s reason for spelling the genesis policy
#: names itself, met from the other side. A gate pins these three against
#: `derivus_spine.verbs.LANES` so the two spellings cannot drift.
TELEMETRY = 'telemetry'
CURIOSITY = 'curiosity'
STANDING = 'standing'
LANES = (TELEMETRY, CURIOSITY, STANDING)

#: What a tree without the extra is told, with the line that fixes it.
NO_PACKAGE = ('the book of record is not installed on this box ({}) - {} names a spine home, so '
              'this verb records to it; `pip install derivus[enterprise]`, or unset {} to run the '
              'edge exactly as it ran before')
#: And what a box with the package but no configuration is told - which is nothing at all, because
#: an unconfigured home is the default posture rather than an error.
NO_ACTOR = ('no actor for this append: every event carries the pseudonymous subject reference that '
            'submitted it, so name one on the request or set {} - a record that invented an actor '
            'would be a record naming somebody who never spoke')

#: One writer, one act at a time. The spine enforces this with a byte-range claim on the home and
#: answers `WriterBusy` to the second holder; this lock is what keeps a request thread and the
#: compute worker from meeting that refusal over their own service's book.
_WRITER = threading.Lock()


class SpineRefused(ValueError):
    """The book of record declining, in the engine's own exception vocabulary.

    A `ValueError` on purpose: the book verbs already map one to a 422 carrying its message
    verbatim, so a spine refusal reaches a desk in the spine's own wording - naming the thing and
    the remedy - rather than in a paraphrase this module invented. It is also what keeps the engine
    from having to import the spine's exception tree to catch it.
    """


def home():
    """The configured spine home, or None where the deployment configured none.

    No default. `DV_Spine` falls back to `~/.derivus_spine` because a person typing a verb means
    that home; an engine that fell back would start recording on any box where somebody once ran
    `init`, which is the opposite of a switch.
    """
    named = os.environ.get(SPINE_HOME)
    return os.path.abspath(os.path.expanduser(named)) if named else None


def configured():
    """Whether this box records. The one question every call site asks first, and a False answer
    means the edge behaves exactly as it did before this module existed."""
    return home() is not None


def package():
    """`derivus_spine`, imported here and now, or the refusal naming the install line.

    The three increment-3 modules are named explicitly because a submodule is only an attribute of
    its package once something has imported it, and `derivus_spine/__init__.py` deliberately keeps
    its surface to the truth layer - the verbs, the policies and the firmness check are reached
    module by module by the code that has business with them, and this is that code.
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
    """The home to work on, or the refusal saying the deployment configured none."""
    name = home()
    if name is None:
        raise SpineRefused(
            'no spine home is configured, so there is nothing to record to: set {} to the home '
            '`DV_Spine init` minted, or leave it unset and the edge runs exactly as it always '
            'has'.format(SPINE_HOME))
    return name


@contextlib.contextmanager
def translating():
    """Every `SpineRefusal` raised inside leaves as `SpineRefused` carrying the same sentence.

    One place, because the rule is one rule: a desk reads the library's own words - naming the
    thing and the remedy - rather than a paraphrase this module invented, and the book verbs'
    existing `except ValueError -> 422` handlers are what carry it to them.
    """
    try:
        yield
    except package().SpineRefusal as refusal:
        raise SpineRefused(str(refusal)) from None


def check_lane(lane):
    """The declared attestation lane, checked against the RECORD's own vocabulary.

    The three names are spelled in this module so a call site need not pay for the import to name
    one, but the refusal is `derivus_spine.verbs.check_lane`'s: the lanes are the record's
    vocabulary, and a service that wrote its own wording for them would be a second source of truth
    about what a lane means.
    """
    with translating():
        return package().verbs.check_lane(lane)


@contextlib.contextmanager
def writing():
    """The home's log, opened for one act, closed after it, and serialised against this process.

    A context manager rather than a held handle, for the reason the CLI opens and closes around
    every verb: a home is a directory and the claim on it belongs to whoever is writing right now.
    The lock is taken BEFORE the log is opened, so a second thread of one service WAITS rather than
    meeting `WriterBusy` - that refusal is for a second process, which is a real deployment mistake,
    and turning one service's own two threads into it would be this module's mistake instead.

    Every `SpineRefusal` raised inside the block leaves as `SpineRefused` carrying the same
    sentence - and the OPEN is inside that translation too, because a configured home that is not
    a home is the commonest of these refusals and it must reach a desk in the same wording as the
    rest.
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
    """Run a read-only fold over the home's log and answer it.

    Reading never claims the home - a replica must not be locked out of its own log by whoever is
    writing to it - so this takes no lock and can be called while another thread writes. It is the
    read side of `writing` and exists so that asking the record a question never has to queue behind
    somebody answering one.
    """
    with translating():
        log = package().SpineLog(where())
        try:
            return fold(log)
        finally:
            log.close()


def canonical(obj):
    """The bytes the engine hashes an object as: sorted keys, tight separators, and the one encoder
    the codebase already has for what JSON has no form for.

    This is the whole reason the engine's own hashes can be blob ADDRESSES in the spine's store: a
    blob is named by the SHA-256 of its bytes, and `content_hash` is the SHA-256 of exactly these,
    so `values_hash` and the address of the values vector are one number rather than two that could
    disagree. The spine canonicalises its own objects under RFC 8785 and addresses whatever bytes it
    is handed, which is what makes the two schemes compose instead of collide.
    """
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      cls=CustomJsonEncoder).encode('utf-8')


def replay(context):
    """The four coordinates a reported number replays from. Hashed, they are also its `result_id`,
    which is what makes an identical submission one execution: the tuple names the numbers."""
    return {'plan_hash': context.plan_hash(), 'values_hash': context.values_hash(),
            'engine_version': __version__,
            'seed': context.current_cfg.deals['Calculation'].get('Random_Seed')}


def values_of(context):
    """The values vector this context would run against, as the bytes its `values_hash` names."""
    return canonical(context.market_patch())


def result_of(out):
    """A run's results as the bytes a claim is compared byte for byte against.

    `Stats` is deliberately absent. It carries timings and counts - how long the run took, how many
    batches it drew - which are facts about a MACHINE rather than about the numbers, and a result
    document carrying them would never reproduce on a second box even when every number in it did.

    FLATTENED THROUGH `tables_of`, which is the same shape the result store already keeps and the
    same shape a client pages. Two things follow and both are load-bearing: a per-result-class
    tolerance is declared against the table path a desk already knows, and the bytes of a run that
    has ALREADY been executed are recoverable from the store as `canonical(result['tables'])`
    rather than needing the tree kept beside it.
    """
    return canonical(tables_of(as_json(out['Results'])))


def result_stored(result):
    """The same bytes, off a result the executor already holds.

    Content addressing means a job whose numbers exist is not run twice, so a standing submission
    can arrive at a tuple this box has already computed. The attestation still has to be made - the
    lane is about STANDING, not about arithmetic - and this is what makes that possible without
    keeping a second copy of every result in memory: `result['tables']` IS what `result_of`
    canonicalises, so the two answer the same bytes by construction.
    """
    return canonical(result['tables'])


def read_values(values):
    """A stored values vector back as the objects `patch_market` takes.

    Through `Config.read_json`, which is the decoder that reads a job file - so a `.Curve` comes
    back a Curve and a `.Timestamp` a Timestamp, and there is no second parser to keep in step with
    the first.
    """
    return Config().read_json((bytes(values).decode('utf-8'), 'values'))


def executor(kind=None):
    """The callable `pin_result` re-executes a claim through: `(job, values, version)` in,
    `(version, result bytes)` out.

    THE INJECTION SEAM, and the reason it is a seam at all: the spine cannot import a pricer - the
    truth layer never depends on the thing it records - so re-execution arrives as a function. It
    is a real function over real documents, which is also why a gate can exercise the whole path
    without patching a single line of library code.

    The version is checked HERE too, before anything runs, so a claim at another build costs a
    refusal rather than a Monte Carlo. The spine checks the version it gets BACK regardless, because
    the record never trusts what it can re-derive - two checks of one thing, and the outer one is
    the one that matters.
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
    """Book a fill against the canonical instrument `deal`. Answers the spine's own envelope.

    The deal block is canonicalised through the engine's encoder and its hash IS the instrument id,
    so booking the same strike twice registers one instrument and files two events against it. The
    execution reference is required by the verb below and has no default here either: it is what
    makes a retry the same fact and two identical clips two facts.
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
    refused with the closure stated - see `derivus_spine.verbs.apply_lifecycle`."""
    verbs = package().verbs
    with writing() as log:
        return verbs.apply_lifecycle(log, actor(actor_name), event_type, body, book=book_name,
                                     effective_time=effective_time)


def declare_market(name, values, actor_name=None, effective_time=None):
    """Point the market name at a values vector. `official` demands `mark` scope and the writer is
    what enforces it - a second check here would be a second place to get it wrong."""
    verbs = package().verbs
    with writing() as log:
        return verbs.declare_market(log, actor(actor_name), name, values,
                                    effective_time=effective_time)


def file_quote(quote_id, structure, plan_hash, values, solved, edge, request=None,
               actor_name=None, book_name=None, effective_time=None):
    """File a quote with BOTH hashes pinned - the values vector it was struck on and the book plan
    its marginal charge was solved against."""
    verbs = package().verbs
    with writing() as log:
        return verbs.file_quote(log, actor(actor_name), quote_id, structure, plan_hash, values,
                                solved, edge, request=request, book=book_name,
                                effective_time=effective_time)


def complete_run(claim, lane, job, values, result, actor_name=None, book_name=None):
    """Attest a standing run at birth. A telemetry or curiosity lane mints NOTHING and never
    reaches this function; the verb refuses one that does, rather than dropping it quietly."""
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

    A read, so it folds rather than writes: asking what the windows are must never queue behind
    somebody booking, and it must never be the thing that turns an approval into a `WriterBusy`.
    """
    return folded(package().policy.firmness_in_force)


def check_firmness(pinned, current, ages, quote_id=None, policy=None):
    """Is this quote still firm, in BOTH dimensions? Answers the verdict, or refuses naming each
    dimension that failed and its own remedy.

    A pure call into `derivus_spine.firmness` over plain data. The policy is read out of the record
    when the caller does not hand one in, which is the ordinary case - `firmness tolerances` is
    policy data by the brief's own list, so where it lives is the log rather than a constant here.
    """
    windows = firmness_policy() if policy is None else policy
    with translating():
        return package().firmness.check(pinned, current, ages, windows, quote_id=quote_id)
