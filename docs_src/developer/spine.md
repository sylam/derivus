# The Spine

`derivus_spine/` is the append-only book of record being built around the engine — the center a desk
box is the edge of. The full seven-increment design lives in the owner's brief outside the tree; this
page documents what is BUILT, which is **increments 1, 2, 3, 4, 5a and 5b**: the log, the blob store
and the chain (riding on them: identity, capability enforcement and key custody), on top of those the
booking verbs, the attestation lanes and the firmness check, over all of it the
projections, the diary, the book file's pin and the desk's own readers of them, beside those the
tier policy, its evaluator and the verbs that file a decision and a close, and through all of it the
quote lifecycle: a price recorded when the client accepts it, routed through the desk's own workflow
before it books. No network — a library, a CLI, eight delegators on `Context`, five read verbs and
three write verbs on the service, and 318 gates.
Nothing here imports the engine, and exactly one module under `derivus/` imports `derivus_spine`:
`derivus/spine.py`.

## The package, and the one dependency

A sibling package on the `derivus_mcp` terms: in the wheel, never importing the engine. Its import
surface is **stdlib plus `cryptography`** (AES-GCM sealing, Ed25519 checkpoint signatures) and nothing
else, held by an AST gate over every module and a subprocess gate proving `import derivus_spine` pulls
no torch and no `derivus.*`. The extra is `pip install derivus[enterprise]`, orthogonal to `desk`.
`DV_Spine init | verify [--chain-only] | checkpoint | status` is the console script; the home is
`--home`, else `DV_SPINE_HOME`, else `~/.derivus_spine` — the spine is the CENTER's store and
deliberately not `DV_HOME`, which is the edge's.

## The truth layer is files

```
DV_SPINE_HOME/
  log/segment-00000001.jsonl   append-only frames, fsync per append, roll at 64 MiB
  blobs/ab/cd/<sha256>         content-addressed, write-once: tmp + fsync + os.replace
  keys/                        blind.key, class_firm.key, the Ed25519 checkpoint pair
  seeds/<projector>-<v>-<lsn>  a fold cached at a close: derivable, and deletable
```

The manifest is a PROJECTION — presence is the tree itself, and `verify` rebuilds what it needs by
walking it — so "the record never trusts what it can re-derive" holds from day one. DuckDB belongs to a
reading plane this package does not have and is forbidden inside it by name; the projections of
increment 4 fold the log itself. A blob whose bytes do not match its hash refuses on read; a put
colliding with different bytes at the same hash is a NAMED REFUSAL, never a dedup; and the
store has **no verb for forgetting** — retention arrives later as a logged event, and the absence of a
delete method is gated as the increment-1 form of that law.

## Two hashes, and why the envelope carries a tag instead of one of them

Every event is a sealed body under a firm-visible envelope. `content_hash` is SHA-256 over the RFC 8785
canonicalization of the semantic tuple (type, version, effective_time, actor, book, body), and it rides
INSIDE the sealed body, because a plaintext hash in a public envelope is a dictionary oracle for
low-entropy bodies. The envelope instead carries the `idempotency_tag`: an HMAC of the same canonical
bytes under a writer-held blind key, so no keyless confirmation exists — gated with a toy key, and by
the same tuple landing different tags in different homes. `event_hash` is SHA-256 over
(idempotency_tag, ciphertext_hash, prev_hash, record_time): the chain, from a genesis `prev_hash` of
sixty-four zeros. LSN is positional and outside both.

The canonicaliser is VENDORED (stdlib only) and held to the RFC's own vectors. The trap is ECMAScript
number serialization, where Python's `repr` has the right digits in the wrong clothing (`1e+16` where
ES writes `10000000000000000`, `1e-07` where ES writes `1e-7`); a repr-passthrough implementation turns
the gate file red by design. Keys sort by UTF-16 code unit; ints past 2^53 and non-finite floats refuse
by name.

Sealing is AES-256-GCM under the firm class key with a fresh 96-bit nonce, AAD over the nine pre-LSN
envelope fields — so no envelope field can move without the body refusing to open — and the interior
binding `{content_hash, payload}` closing the loop. Classification ships DORMANT: one class, everyone
holds it, and the day desk two arrives walls are a classification decision, not a redesign. Sealing
itself is not deferred with the classes: genesis-era bodies must be crypto-shreddable, and the shred is
gated — delete the class key and the chain still verifies while every body is unreadable.

## The writer

One writer, enforced rather than asserted: the first `append` takes an exclusive byte-range claim on
`log/.writer.lock` (msvcrt/fcntl per platform) and a second writer refuses with `WriterBusy` naming the
lock. Verify never takes it, so replicas read freely. The append flow is ordered so nothing partial can
land: validate against the closed vocabulary → canonical bytes → tag → duplicate-tag check (on a hit the
writer DECRYPTS the stored body and byte-compares canonical plaintexts — equal is the safe retry,
coalesced onto the existing LSN; different is `CollisionRefusal` naming both) → every blob the body
cites must already be in the store (`vocabulary.BLOB_FIELDS` declares which fields name stored bytes) →
stamp, seal, hash, one fsynced line. A retry with no caller-passed `effective_time` coalesces because
the tuple carries `null` there — the writer's own clock is not part of the fact — and `as_of_key`
resolves null to `record_time` at read time.

The vocabulary is CLOSED: sixteen trading fact types (`checkpoint` among them), the two custody types,
increment 3's three provenance types and the writer's own reserved one, with validators naming the
missing or surplus field. A consequence-shaped submission ("knocked_out") is refused with the closure
stated — knocks and expiries are projections, never events.

The torn-tail rule is exact: a final line with no terminating newline was never durable (the write is
one call, newline last) and is truncated on open; a newline-TERMINATED line that will not parse is a
durable line that was altered, and is `ChainBroken`. That distinction is what stops the recovery path
being usable to roll a log back through its own checkpoints.

## Checkpoints, and verification as a replica

`checkpoint` events sign (lsn, head event_hash) with the deployment's Ed25519 key. The verifying key is
published at genesis as a firm-class policy blob, so a replica asserts checkpoint AUTHENTICITY — not
merely chain integrity — from checkpoint one, against the log's own contents rather than any local
file: the gates prove it by deleting the local `checkpoint_verify.key` (verification still passes) and
by deleting the published blob (refusal naming it). Rotation is a fold: every
`checkpoint_verifying_key` declaration is recorded with its LSN and each checkpoint verifies under the
key in force at ITS position, so a logged rotation neither bricks history nor lets a later declaration
retro-invalidate genesis.

`verify_home` runs in two modes. **Entitled**: chain, seals, interior bindings, blinded tags (where the
blind key is present), checkpoint signatures, referential closure over every cited blob. **Chain-only**
— the keyless replica posture, a home of `log/` and `blobs/` alone — verifies the whole chain over
ciphertext it cannot read and says honestly that checkpoint authenticity and bodies were not assessed,
rather than skipping silently. Frames refuse surplus fields in both modes: there is no unauthenticated
channel into the record.

## The gates

105 in four files (`test_spine.py`, `test_spine_canon.py`, `test_spine_imports.py`,
`test_spine_store.py`; the glob `tests/test_spine*.py` is the wider eleven-file set worth 293 of the
318 above, and `tests/test_diary.py` carries the rest),
all real stores in temp dirs, every fault injected by doctoring DATA on disk. The shapes worth naming:
three tampers on three copies, each caught by a different layer (body byte by the chain, envelope field
by the AAD, record_time by a keyless replica); a re-forged tail caught by the interior binding AND its
dual caught by the stale tag, each proven independently load-bearing in the posture where the other is
absent; the design's synthetic book (late booking, backdated amendment, republished fixing under one
(index, date, source) key, superseding close) driving all sixteen fact types through the writer with
as-of provably departing from as-at; and restore as file-copy, both replica shapes re-verified on the
far side.

## Increment 2 — identity, attribution, key custody

**Identity is bought, not built.** `identity.py` VERIFIES an OIDC ID token against a JWKS the deployment
hands in as data — nothing is fetched, verification stays local — under an RS256/ES256 allowlist that
refuses `none` and every HMAC by name before any key is selected (the alg-confusion attacks are gates,
not warnings), with `kid` selection across key types, the ES256 raw-`R||S` signature contract,
`exp`/`nbf` on an injectable clock, and OIDC Core's `azp` rule so a co-audienced client cannot replay
its own token as a spine credential. The subject reference is the token's `sub`, pseudonymous by the
the design's rule; display names live in `names.json`, a mutable side table OUTSIDE the log whose erasure is
gated to leave every chain byte identical.

**Capabilities are one document and one pure function.** The document — grants of (verb × book) over
`draft | validate | book | approve | mark | admin`, plus READ over entitlement classes — is a hashed
blob declared through the ordinary writer (`policy_declared` with the RESERVED policy name
`capabilities`), each declaration a complete replacement, resolved by fold-at-LSN so "could X do Y in
March" replays from the log like every other question. Enforcement activates BY DECLARATION: a home with
no document runs as the single-user instrument it is — which is why every increment-1 home and gate is
bit-for-bit untouched — and once one is in force the writer refuses an unscoped append AND logs the
refusal as a `capability_denied` fact under the writer's reserved actor, because a decision is a fact. A
document whose blob has been doctored or lost folds to UNREADABLE rather than raising: every verb
refuses by name, custody hands out no key, and the break-glass walk leads OUT — the condition the
recovery grant exists for cannot brick the recovery itself. `break_glass_used` is gated on the genesis
grant from event one, the admin it restores is revocable by the next capabilities declaration, and
`vocabulary.classify` derives the entitlement class the envelope carries ("firm" for everything until
desk two — one function instead of one constant).

**Custody.** Per-seat X25519 keypairs at enrollment; the firm class key wrapped per READ-entitled
subject — ephemeral ECDH, HKDF, AES-GCM with the (class, subject) pair bound into the AAD, so a wrap
re-addressed to another seat refuses to open — plus an escrow wrap under a declared escrow key. `rewrap`
is idempotent and driven by the document in force; `grant` reports the rewrap it now owes, so "rewrap on
grant change" is a printed obligation rather than operator memory; `materialize` turns a chain-only
replica entitled off its wrap without overwriting anything; escrow recovery is gated on a
crypto-shredded copy. **THREE residuals are declared**: the design's two (forward-only revocation,
traffic shape) and this increment's own — a hub-minted seat key is a bootstrap the hub has seen, stated
in `custody.py`'s docstring, with seat-generated `--public-key` enrollment as the form that eliminates
it.

The CLI grows `enroll | grant | rewrap | name | whoami`. Of the 64 new gates the two shapes worth
naming are the strand-and-recover walk (a document stranding the last admin, every later declaration
refused, the genesis seat walking out, the recovered admin later revoked) and the stale-fold check (a
revocation lands and the same open handle answers off the platter, never a snapshot).

## Increment 3 — booking verbs, lanes, `result_pinned`, quote firmness

**The vocabulary grew by three and changed none.** `run_completed`, `result_pinned` and `quote_filed`
are a fourth part of the closed set (`PROVENANCE_TYPES`), because they are said by a fourth mouth: the
thing that PRODUCED the numbers, rather than the desk that books against them. Every validator that
existed before them validates exactly what it did before. Their scopes are three different authorities:
an attestation and a quote take `book`, a promotion takes `approve` — giving standing to a tuple this
hub never witnessed is a second pair of eyes on somebody else's claim. Two field kinds were added for
two fields: a nullable seed (a job declaring no `Random_Seed` is hashed with a null there, and a
substituted zero would name a tuple no result was ever filed under) and an object of name-to-number for
a quote's solved coordinates.

**The verbs live in the spine and the engine gets delegators.** The build order says "booking
verbs on `Context`"; the house's first law says no module under `derivus/` learns about users, workflow
or storage. Both hold because the LOGIC is `derivus_spine/verbs.py` — plain functions over plain data —
and `Context` gains five one-line delegators (`book | amend | apply_lifecycle | declare_market |
pin_result`) that canonicalise through the engine's own encoder and hand bytes across.
`derivus/spine.py` is the whole seam: it imports `derivus_spine` LAZILY inside the function (the
`service.py`/fastapi precedent), refuses by name where the extra is absent or where `DV_SPINE_HOME`
names no home, and re-raises every `SpineRefusal` as a `ValueError` carrying the spine's own sentence
unedited — so the book verbs' existing 422 handlers surface the library's wording rather than a
paraphrase.

**`DV_SPINE_HOME` unset means BIT-IDENTICAL.** There is deliberately no fall-back to the CLI's
`~/.derivus_spine` default: a desk box that once ran `DV_Spine init` must not silently start recording.
Unset, a lane is accepted and inert, no pending file grows a field, nothing is written, and the Context
verbs refuse by name — asserted as the first gate of the increment, with `tests/test_service.py`,
`test_mcp.py` and `test_structures.py` green untouched beside it.

**Attestation is by LANE, and the rule is one sentence: a run is recorded iff its output will be cited
by a fact.** `telemetry` is the blotter's repaint, superseded before anything could cite it; `curiosity`
is a what-if; `standing` is a run a fact is about to name. Only standing mints. `/execute` takes the
lane (unknown refuses BY NAME where a home is configured, inert where none is), defaulting to curiosity
because a caller who has not said their output will be cited has not said it; `/book/price` is curiosity
and not a parameter; the Bloomberg tick is telemetry; and `/book/structure` is CURIOSITY too, because a
quote is not cited until a client accepts it — 5b moved it, and the head does not move on a quote.
Three consequences are gated rather than assumed, and the
first two are one sentence read carefully — **content addressing dedupes NUMBERS, and the lane is about
STANDING**:

- a standing run whose numbers ALREADY exist still attests, on the request thread, because the worker
  will never revisit that result;
- a standing submission that coalesces onto a run still QUEUED or RUNNING is PROMOTED onto it inside
  `ComputeExecutor.submit`, because the job the worker dequeued carries the first caller's lane. The
  promotion and the publication of the result are one transition under one lock, so the request thread's
  arm and the worker's arm are exhaustive rather than merely likely;
- a standing run whose attestation is REFUSED fails, because serving numbers that did not acquire
  standing is the unbacked citation the rule exists to prevent.

**`pin_result` re-executes through an INJECTED executor.** Re-execution is the one thing the truth layer
cannot learn to do — it would mean depending on the thing it records — so the caller hands in
`executor(job, values, version) -> (version, result bytes)` and the record checks what came back rather
than trusting who ran it. Four paths: a CACHE HIT where a prior `run_completed` carries the same four
coordinates (nothing executed, counted rather than believed; a prior attestation naming a different
result is refused, because one replay tuple cannot have two); bit-equality as the fast path; a
per-result-class tolerance comparison; and refusal by name for a version that is not the recorded one,
bytes that will not reproduce, or a result class the policy does not name. The fold reads
`run_completed` and never `result_pinned`, so one unverified promotion can never become the evidence for
the next. **The spine admits no tolerance of its own**: the only floating-point comparison in the
package is `policy.compare`, it runs on numbers that came out of the engine, and every epsilon it uses
was declared by a deployment in a hashed policy blob — a home that has declared no tolerance policy pins
nothing at all.

**A quote pins TWO hashes, and what they are asked is 5b's.** `quote_filed` carries the values vector
the quote was struck on and the book plan its marginal charge was solved against, beside the solved
coordinates, the edge, and — optionally — the relayed client request as an erasable field, which needs
no mechanism of its own because every body here is sealed. The two hashes are the BOOK's, taken before
the live spot lands on the quote's copy, because what a booking asks is whether the market and the book
this trade would LAND against have moved. The two planes are disjoint by MEASUREMENT — a vol tick moves
`values_hash` and leaves `plan_hash` bit-identical, and the gate asserts that on a fixture that ticks a
vol with no booking in sight. Increment 5b is where they are read: the PLAN refuses, the MARKET is
reported, and the one staleness window a desk declares is the age of the board the quote was struck on.

**The dual write has a declared ORDER.** Under a spine home the event goes first and the book file
follows — the durability law applied to the pair, so a refused booking leaves the file byte-identical.
The file's formal rehoming as an LSN-pinned PROJECTION is increment 4's; until then it is the interim
stand-in and the log is what is true. Two fields become required that were optional before, because a
fill's body carries them: a signed `quantity` and an `execution_reference` (the venue exec id, or a
quote id for an approval), plus a `NettingCollateralSet` above the trade to name the counterparty. A
`delete` records nothing: the fact that ends a trade is an election, an observation or a status
transition, never an inference from a row leaving a cache.

**58 new gates in two files.** `tests/test_spine_verbs.py` drives the spine half with no engine anywhere
near it — the executors are ordinary functions written in the file. `tests/test_spine_engine.py` drives
the seam through both at once: real homes, a real book, a real `BaseValuation`, the real partition. The
shapes worth naming: the plan hash RE-DERIVED by recompiling the blob-stored job document at the
recorded LSN and required to equal the recorded tuple, with the result reproducing to the byte beside
it; the tick sequence whose four market ticks and four what-ifs move the head not once (absence asserted
on the head rather than on a filtered count); the fixtures aged BY DECLARATION, a firmness policy with
zero-second windows in the record rather than a sleep; and the in-flight coalescing gate, which backs
the one worker up behind a barrier job so that "a standing submission observed while the same tuple is
still queued" is a fact rather than a timing window.

## Increment 4a — the folds, the seeds, and the fixings policy

**A projection is a pure fold, and a consequence is never a fact.** `derivus_spine/projections.py` is
seven projectors and one driver: `fold(log, projector, lsn=None, seed=None)` streams the frames a
projector NAMES, opens only those bodies, and answers state nobody edited. `positions`, `blotter`,
`lifecycle`, `markets`, `attestations`, `decisions` and `activity` each carry three members and no
state of their own — `initial`, `apply`, `rows` — so what a reader sees is sorted canonicalisable JSON
and a golden replay is a byte comparison. A knock, an expiry and an accrual are read OFF that state
(`knocked` is the first crossing over `fixings_at`'s answer, with the terms handed in by the caller),
because a record holding one would be a second source of truth about whether the barrier fired.
An amendment carries the position FORWARD onto the instrument the amended
terms hash to, the old row standing at zero naming where it went, because that is the hash the book
file's deal node now has; and a row is never dropped, since a position closed out is a fact about the
book rather than an absence. `lifecycle` files a status transition under the subject its body names
and nowhere else — an instrument hash today, a cashflow key when the diary has one — so a status is
read where it was filed rather than inferred. `activity` opens no body at all and reads EVERY type:
its summary is a declared table of type to one sentence, and a type the table does not know renders
its own name, so the strip renders complete on a replica holding no key, including one replicating a
newer hub.

**Nothing here caches, and a seed is verified rather than trusted.** The caller holds
`(head_lsn, state)` and advances it, and a SEED is that pair written down: `seed_at` mints one only AT
an `official_close_declared` — the position where the desk already agrees what the day was, located on
the PLATTER rather than in a handle's own index — as canonical JSON at
`seeds/<projector>-<version>-<lsn>.json` beside `log/` and `blobs/`, written scratch-fsync-rename the
way the store writes. It carries that close's event hash AND its state's own address, so a seed
minted over another home's history, one of another projector or version, a torn file, and one whose
state was edited beneath an intact close each refuse BY NAME where they are read, and the projector
and version again where the state is folded. The fold is pure in it: the state is copied before it is
advanced, the caller's seed is left where it was, and a position BEHIND the seed refuses rather than
answering the state in front of it. A seed is derivable and disposable: both verification modes
answer exactly what they answered before one was minted, and deleting the directory costs one refold.
A caller seeds `positions` and `blotter`, never the `activity` strip, whose state IS its history —
copying that state costs 219 ms where folding it from genesis costs 38. Seed equivalence is gated
byte for byte on every projector, at a close the synthetic book restates AFTER.

**Supersession is `(effective_time, lsn)` under the whole key.** A republished print of one
`(index, date, source)` wins by its as-of key rather than by arriving last, and the print it beat stays
on the row — the record holds it, so the projection may not hide it. A second administrator's print of
the same index and date is a second row, never a supersession, and `fixings_at` resolves ACROSS sources
by a declared order: the third reserved policy name, `fixings`, is `{"sources": {"FxRate.ZAR": ["ECB",
"BFIX"]}}`, one ordered list per index, validated at the declaration like the other two. The first
named source holding a print is the fixing in force. A home declaring no policy at all has named an
authority for nothing and resolves nothing; once one is declared, an index it does not name REFUSES BY
NAME where the caller asked for that index — a fixing whose authority nobody declared is not a fixing
a plan may use — and is left alone where it did not, so one stray print cannot refuse a whole compile.
The prints are the `lifecycle` fold's rather than a second walk for them; the policy itself is read
the way every policy is, by `in_force`.

**Twelve gates** (`tests/test_spine_projections.py`), eleven on the design's own synthetic book,
imported rather than written again: a committed golden per projector and for the strip's table of
sentences, seed equivalence with the fold pure in its seed, the seeded fold seeing the restatement
behind it, supersession in both directions and under the whole key, a knock derived while
`apply_lifecycle` refuses to file one, LSN order under a shared truth-time, a v2 projector folding v1
bodies while a seed of another version, another home or a torn file refuses, a fold that never claims
the home and sees the writer's next frame, the fixings refusal and the declared order, `seeds/`
invisible to verification, and the import surface with the gate's glob widened to every `.py` at any
depth so a subpackage cannot smuggle one past it. The twelfth is a second fixture for what the first
cannot say: a position closed to zero, one replay tuple attested twice (the FIRST stands, as
`verbs.attestation` answers), one policy declared twice (the LAST stands, as `policy.in_force`
answers), a print superseded twice, and two verdicts read in the order they were filed.

## Increment 4b — the diary, the LSN pin, and the plan as a fold

**The diary is the compile's own schedule, re-emitted.** `derivus/diary.py`'s `schedule_of(context)`
runs the COMPILE half of a base valuation — the market built, every deal constructed and its
schedules bound — and stops before the structure resolves; then it reads what it just bound through
`utils.walk_schedules`, which is the walk `utils.bind_schedules` itself takes. One walk, so the leg
names a settlement reference is built from cannot drift from the binding.

**ONE SPELLING OF A PAYMENT.** A row is one payment per `(leg, pay day)`, never one per schedule
row, and its amount is `pricing.fixed_payments` — the line `pv_fixed_cashflows` discounts, factored
out so the two cannot be two numbers: the rate coupon and the fixed amount of every row sharing that
day, summed, and compounded where the leg's own terms compound. A bond repaying its principal with
its last coupon announces the sum, and two accrual sub-periods paying on one day announce one
payment. A schedule carrying RESETS determines nothing — its amount is a floating pricer's and the
diary does not spell a second one — so its rows read `amount: null, determined: false`: **a null
amount is never written as 0.0**, and `export_settlements(rows, official_values_hash, due_before)` —
which takes the diary, one market hash and the day the file settles, and reaches nothing else —
refuses an undetermined row, and a row naming no currency, BY NAME rather than instructing a wrong
payment. `due_before` has no default: a settlement file is struck FOR a day.

An `expiry` row carries `needs`, which is `election` where the deal type's terms vest an exercise in
an actor and null where a fixing determines the payoff. A deal whose compile announces no payment at
all pays on the settlement date its TYPE declares in the deal-type table, falling back to its
expiry: `get_settlement_currencies()` is the reval-date accumulator — a barrier registers its
monitoring days in it — so it is not a payment ladder and is never read as one. AN OPTION WAITS FOR
TWO THINGS after its expiry, and the table names both: a `fixing` row on the underlying, because a
European option compiles no reset schedule and its payoff is that day's print; and a `payment` row
at its settlement date with `amount: null`, because no field holds `Units × max(S−K, 0)` and the
money still moves. A close does not pass over an unsettled payoff.

**The derived key is the row's own DATE, and position was refused.**
`derivus/spine.py`'s `cashflow_key(instrument_hash, leg, kind, date)` is the content hash of those
four through the record's own canonicaliser, so it is 64 lowercase hex and passes
`vocabulary.is_hash`: a `status_transition` names one settlement with NO new field kind, and the
vocabulary does not grow for it. The four are unique because a row IS one `(leg, kind, date)`. The
POSITION was refused: a schedule drops the rows it has paid as the book rolls, so positions renumber
under the settlement references already filed and a transition settling February would read, after
the roll, as settling August. A date does not move. `derivus/diary.py` asks the seam for the key and
imports nothing of the record itself, so on a box without the extra a row carries `key: null` and
everything else. A book rolled PAST a coupon keeps every surviving row's key and loses only the row
that was paid.

**The book file is the fold's SUBJECT, not its output.** It keeps its content — the market data is
not in the fold at all, and a hand edit is the desk's own act — and gains `Spine: {lsn, head,
hydrated_at}`, a sibling of `Calc` stamped by `Book._land` under a configured home and written
nowhere without one. `Context.load_json` reads `Calc` alone and `Context.plan_hash` hashes `params`
and `deals`, so the pin cannot move a plan; `risk_etag` reads the three `Calc` sections, so it does
not bust the risk cache; `Book._current` hashes the whole text, so it does move the poll etag, which
is right — a new LSN is a new state of the book. `GET /book` answers the `lsn` beside the etag,
`GET /book/status` gains a `spine` block with the pin and how far the record has moved since — in
events, and in the fills and amendments among them — and NOTHING that folds: counting a divergence
is linear in the history and that is the verb a client starts with. `GET /book/reconcile` is where a
desk asks, and it pays in full: a trade the record holds
that the file lost, a deal the file holds that nobody booked, and an instrument the two count a
different number of clips of — the file carries terms and never a signed quantity, which lives only
in the record, so clips are what the two can disagree about and the record's own quantity is
reported beside them, compared as VALUES rather than as types since a seed round-trips a float to
an int. **A divergence is a READING and never a refusal** — the desk books what
it books at this stage, and the rules tighten when the spine is full. Every comparison is by
instrument address rather than by reference, so a renamed deal is two divergences rather than a clean
reconcile, an amendment chain is compared at its head, and a node the engine is told to `Ignore` is
still a node somebody booked. **The fold is taken AT THE HEAD**, never at the pin: a booking whose
file write did not land is the one failure this verb exists to name and it lives entirely past the
pin. `events_behind` counts every event since the file was written and `positions_behind` the fills
and amendments among them — a policy or an attestation moves the first and no row here.

**The plan is terms plus the observations the record holds.** `spine.compiled_job(document, lsn)`
walks the job's deals, and for each one whose type declares an observation table — the deal-type
table in `derivus/diary.py`, read by the compiler and the diary alike and never spelled twice —
writes the fixing in force at `lsn` into the cell that type declares it in, for a day ON OR BEFORE
the job's own base date: a print carries its date as text, so a forward-dated one is a legal fact
and writing it would price a barrier as observed on a day that has not happened. `/execute` and
`/prepare` both compile before they hash, so a plan named once and ticked as a delta is the plan
that runs; without a home, and for a document whose deals declare no observation table, `compiled_job`
returns its argument and the edge is what it always was. Which source is authoritative is POLICY:
`fixings_at` resolves across sources by the order the reserved `fixings` policy declares, and an
index A PLAN COMPILES AGAINST that the policy does not name refuses by name — a fixing whose
authority nobody vouched for is not a fixing a plan may use, and the refusal reaches a desk as a 422
in the record's own words. A READING refuses nothing, its COMPILE HALF INCLUDED: `compiled_job` takes a
`strict` flag, and the two read verbs pass it false — the book is compiled as written plus whatever
the declared orders can fill, an index the policy does not order reads unresolved with the reason on
the row and is outstanding, and a print for an index no deal here names is nothing to them at all.
A read that cannot compile at all answers its refusal ONCE: the result store keeps successes only,
so a desk that declares the missing order is answered on its next ask rather than told the same
thing until the book file moves.

**The close check is the catch-up rule as a read.** `GET /book/close/check?date=` answers whether a
close on that day is legal and what it waits on: a `fixing` row no declared source has printed, a
`payment` row no settlement transition was filed against its key, and an `expiry` row whose terms
vest a choice nobody has elected. An expiry a fixing determines never blocks a close — the fixing
and payment rows already carry it. Nothing is declared here; declaring the close stays a verb.
`?date=` is PARSED to a calendar day — a time, a year or a garbage string refuses 422 by name, since
a string compare would answer `legal` for an empty one. `GET /book/diary` serves the rows, cached
under the etag of what a COMPILE reads — the deals and the calculation, not the market, so a tick
that moves every value keeps the compile — and computed as a job on the compute queue at a base
valuation's cost class, so a diary never rides the poll path and two asks over an unmoved book are
one compile. Both verbs compile the book through `compiled_job` first, so a barrier the record makes
priceable is readable. The three reads are `book_reconcile`, `book_diary` and `close_check` in the
MCP binding.

**Two boundaries, declared rather than discovered.** `xva.json` is NOT a fold and does not become
one here: `run_completed`'s body carries no netting set, the vocabulary does not grow in this
increment, and a row that cannot name its set is not a projection. And FULL HYDRATION — the deal tree
regenerated from the positions fold rather than compared against it — waits until a netting set's own
node is in the record, its CSA terms being what no fact carries today; until then the file is the
materialisation and reconcile is how it is checked.

## Increment 4c — the record's readers on the desk

**Two reads, one seam each, and both are READINGS.** `GET /book/activity` is the strip: one line
per event, newest last, beside the `lsn` to ask again from. **A page walks the record FORWARD**:
with `?since=` it is the OLDEST `?limit=` rows after that position and the cursor is THE LAST ROW
DELIVERED, so a reader asking again with the cursor it was given reaches every event in turn and a
capped page loses none; with no `?since=` it is the NEWEST rows — a strip's first paint — and the
cursor is the head the fold reached, never the head the handle opened at, so an append during the
fold cannot make a client skip an event. The fold advances the strip's own `(lsn, state)` pair —
the pair `fold` takes, a strip's state being its history, and never a seed FILE, which is minted at
a close and is not what a page wants. **A page saves the rows and not the read**: `Log.frames`
reaches `start_lsn` by skipping LSNs rather than by seeking to a byte offset, so the log parses its
segments either way. At the design's synthetic home — 21 events — the fold is 0.24 ms whole and
0.25 ms from the head, against 0.41 ms to open the handle at all; at 2,004 events one HTTP read is
25 ms and 40 KB for the capped strip against 21 ms and 22 bytes for a page that answers nothing
new, and one beat of the web's poll is 47 ms at 2,005 events. Seeking would be a change inside the
log's own reader, and is not one this made. `GET /book/markets` is the `markets` fold at the head —
the official close standing per market with the LSN of the close it restated, the declared names,
the snapshots. **Neither asks for a book**: a replica carrying a home and no book file at all is
exactly the posture a strip that opens no body is for, so the two check the home and never
`DV_HOME/book.json`, where `/book/reconcile` compares against the file and needs one. A home whose
class key is gone answers the record's own sentence as a 422 on every read that opens a body — a
crypto-shredded home is an entitlement fact, not a fault in the verb. The binding gains
`book_activity` and `book_markets` beside `book_reconcile`, `book_diary` and `close_check`.

**Three consumers on the desk, one pure module.** `web/src/spine.ts` holds the arithmetic and
`web/scripts/spine_check.mjs` drives it: the strip's merge of a page onto the rows already held
(LSN order, no row twice, the newest 200 kept), the banner's verdict and what it asks a fold for,
the markets panel's shaping, and the two predicates the store repaints on — so what decides
whether a desk repaints is checkable without a browser. THE RECORD RIDES THE BOOK POLL'S OWN
BEAT — the status block for the pin and the two behind counts, then the events after the cursor,
on the one timer the client has — and `spine` is read fresh on every beat, so a desk that records
nothing renders nothing and a service restarted with a home appears without a reload; the price of
that is one `/book/status` per beat on every desk, the whole desk-status composition, about 3 ms on
a box that records nothing. The strip is
a collapsible line at the foot of the app frame; the banner sits under the book's own header, so it
reaches the blotter and every other screen at once; the markets panel heads the Market Data screen,
where a reader comes for a close. A document opened from a file renders none of the three: the
record is the live book's and says nothing about a copy on somebody's disk.

**THE LISTS DECIDE AND THE COUNTS NEVER DO.** `events_behind` and `positions_behind` say how far
the record has moved past the file's PIN, and every write to the file re-pins it at the head — the
write that is the other half of a divergence included — so a banner gated on them is told `clean`
about a trade the desk deleted through its own verb, and un-shows a drift it has already shown as
soon as the next booking lands. The verdict is read off `/book/reconcile`'s three lists: `drifted`
where the file holds a deal nobody booked, where the two count a different number of clips, or
where the record holds more positions the file lacks than `positions_behind` can account for — a
trade the file has LOST, beside a lagging fill or alone; `behind` where the record holds rows
`positions_behind` accounts for; `clean` only where all
three lists are empty, so a fixings policy past the pin lights nothing. The fold itself costs the
head, so it is asked for where `positions_behind` is above zero or the FILE has moved since the
answer in hand — never on the beat, never on `events_behind`, and an answer once fetched stands
until another replaces it.

## Increment 5a — the tiers, the decisions, the close

**A FOURTH RESERVED NAME, and the document IS the workflow.** `tiers` joins `tolerance`, `firmness`
and `fixings` as a policy this package owns the shape of: an ORDERED list of tiers and the market
each designated process resolves by name, hashed into the store and declared through the ordinary
writer under `admin` like every other. The rule is one sentence — **the FIRST tier whose every
declared check passes is the one that applies** — and a key a tier omits is a check it does not
make, so a list ends at a catch-all declaring none and a ticket reaching the end of a list without
one falls in no tier at all, which is an answer rather than a default. Three checks and no fourth.
`max_notional` is an amount AND the currency it is an amount of, read against the notionals the
CALLER states, so no check here reads a market and a ticket whose notional is not stated in that
currency FAILS that tier by name — the safe direction, and the one a conversion inside a policy
check would lose. `max_tenor_years` is the ticket's own. `market` is the values vector the quote
pinned, which must be the one standing under the name the tier prices on. A tier names the `seat` an
AUTOMATIC approval signs under, or declares `four_eyes` and wants a human; both together is refused
at the declaration, since an automatic seat never books and the key would say nothing.

**Staleness is `firmness`'s, and a tier restating it is refused by name.** How old a board may be is
`pillar_seconds`, enforced on every booking before any tier is read, so a document or a tier carrying
it — or `firm`, the verdict it answers, or either of the two windows 5b retired — is
refused where it is declared rather than making one question two standards. `designations` is the
other section: process → market name, closed to the processes something actually resolves by name
(`settlement_export` is the one this increment binds), because a name nothing reads is a rule nobody
enforces. Neither a designation nor a tier's own `market` may be a `private/` name: a designated
process resolves the firm's board, a tier's market decides whether an AUTOMATIC seat signs, and
neither rests on a board one seat declared for itself.

**WHAT IS STORED IS THE COMPLETED DOCUMENT**, the practice `parse_firmness` set. The two defaults
this shape has — `four_eyes` false on a tier that names no seat, and an empty `designations` — are
written in before the bytes are hashed, so one workflow is ONE BLOB however the operator spelled it
and two desks declaring the same rules do not get two governance histories of one decision. A tier
that names a seat is completed with no `four_eyes` at all: the two keys together are refused, and an
automatic seat never books. A cap is a MAXIMUM on every axis — a ticket exactly at one passes and
the smallest step over it does not — and a notional is an AMOUNT of a currency and never a sign, so
a zero or a negative one is refused rather than clearing every cap there is.

**The evaluator is pure, and nothing calls it yet.** `derivus_spine/tiers.py` is `firmness.py`'s
shape — `assess` returns a verdict, `check` raises `TierRefused` carrying the same sentences — over
a parsed policy, a ticket and the market names standing, holding no log, clock, store or home, which
the gate asserts on the signatures AND on the module's own imports rather than on the docstring. The
verdict names the tier, the seat, every check read with the value measured and the bound declared,
and one sentence per failure of every tier tried, which is also the route the ticket took. A check
that could not be MADE is a failure and never a pass. `standing_approval` is the human half: **THE
LATEST VERDICT STANDS**, by LSN, because a verdict is never withdrawn, so a rejection filed after an
approval is what the record says last; under `four_eyes` the approver may not be the booker. Scope
is not re-checked there — the writer refused an unscoped approval at the append, so every verdict in
the fold was already a seat's.

**The decisions are verbs now, and so is the close.** `approve` and `reject` file the two verdicts
the vocabulary has carried since increment 1 with nothing filing them; both demand `approve` scope,
and an unscoped seat is refused with the denial landed, like every other verb. One seat signing one
plan twice is ONE fact — the semantic tuple carries no clock of the writer's own — while a second
seat signing the same plan is a second, because who signed is part of what was said. `declare_close`
blobs its values exactly as `declare_market` does, and a second close on one market SUPERSEDES the
first rather than correcting it, so a day restated is two facts and a fold taken as at the first
still answers what it answered. `quote_filed` gains the optional `ticket`: `plan_hash` is the BOOK's
plan, which is the right question for firmness and the wrong one for identity — two quotes struck
against an unmoved book pin the same one — while `ticket` is the plan the book WOULD have with this
quote's mirror spliced in, so an approval over it reaches this quote and no other and an amended
mirror is a new hash by construction. A body carrying none validates exactly as it did. The eighth
projector, `quotes`, reads those bodies: one row per quote id, carrying the SEAT that struck it off
the envelope, since no body holds one, a second filing under one id standing by the as-of key every
other keyed row here stands by, and the rows read in LSN order rather than their ids'. **It is the
one projector whose rows grow with the desk's own activity**: it opens a body per quote, about
0.16 ms each, so the fold is 7.7 ms on a two-thousand-event home holding three quotes and 327 ms on
one holding 1,978 — the envelope filter keeps it off every other event and nothing keeps it off its
own. `spine.quotes()` answers every quote the record holds and is a reading; the lookup a desk will
want is 5b's, when something asks about one ticket.

**A market is resolved BY NAME for the first time.** `spine.resolve_market(name, actor, process)`
composes the `markets` fold, the blob store and `read_values` — every piece of which existed, and
was composed in exactly one place — and answers `{name, values_hash, values}` as the objects
`patch_market` takes. A name nobody declared REFUSES rather than falling back on whatever market is
loaded, which is the whole point of binding a process to a market by name; with `process` named, it
must be the market the tiers policy designates for that process, and a home designating nothing
refuses too. **A NAME RESOLVES TO ITS LATEST DECLARATION BY LSN, and an official close is one way of
declaring one**: a close moves what the name answers and a later declaration moves it back, so a
reader cannot be handed yesterday's board on a market the desk has since closed. A
`private/<subject>/<name>` market resolves for the subject ITS NAME NAMES, and `declare_market`
refuses a private name whose subject is not the seat declaring it — self-declared means
self-declared, so the name and the fold cannot disagree about who owns a board. The ownership rule
and the designation rule are checked independently, neither standing behind the other. **The
surveillance and admin read of a private market is DEFERRED**: it wants a second entitlement class
and a `read` row per subject, which is the reclassification the dormant mechanism holds for desk
two, and a per-market ACL now would be the per-object ACL the design forbids by name. Until then the
owner rule refuses at the verb, without minting a fact for it.

**The mouths.** `DV_Spine declare <policy> <file.json> --actor` puts any reserved policy on the
record from a JSON file, parsed and canonicalised on the way into the store so one policy is one
blob however the operator spelled it, and `DV_Spine policy [name]` reports what is in force with the
blob and the LSN of the declaration that put it there, a name nobody declared reading as nulls. All
three come off ONE WALK — `policy.in_force` answers the position of the frame it chose — because
`policy_declared` is open-bodied and a declaration under a reserved name carrying no blob is a fact
this reading steps over rather than a position it could borrow. `Context` gains `approve`, `reject`
and `declare_close` beside `declare_market`, and the seam gains `tiers_policy`, `quotes` and
`verdicts` as folds.

**18 new gates in four files**, with the `quotes` golden and its rows joining the twelve in
`tests/test_spine_projections.py`. The shapes worth naming: the whole closure of the document met AT
THE DECLARATION, seventeen ways at once, including the two that are not merely shape — a restated
staleness window and a private market under a designation or a tier; three spellings of one workflow
hashing to one blob; the purity gate reading the evaluator's signatures and its import list, so an
evaluator that learned to open a home turns it red the day it does; the market resolved by name and
PROVED by patching a second context carrying another spot onto it; and the CLI round trip, whose
refused declaration leaves what was in force in force and whose open-bodied declaration lends its
position to nobody.

## Increment 5b — the acceptance, the tier step, and the decision verbs

**A MARKET'S IDENTITY IS ITS NUMBERS.** `values_hash` is taken of the values patch with the CLOCKS
projected out — `schema.without_clocks`, derived from the declarations rather than from a list of
names: a value-bound `Date` on a price factor (`Quote_Timestamp` on a surface) and the stamp on a
quote row. A cadence that re-reads a board it did not move leaves the hash bit-identical, because
rehashing on every tick brings no information: when a board was read is data on the row and on the
quote that cites it, and the record stamps every event with its own clock. `market_patch` is
untouched — it is what a tick APPLIES, and moving a stamp out of the values plane would make a
re-stamp a re-authoring — and `spine.values_of` takes the same projection, so the stored vector's
ADDRESS is still the hash the quote pinned and the verb asserts the two are one number.

**WE DO NOT CARE ABOUT A QUOTE UNTIL THE CLIENT ACCEPTS IT.** A desk quotes fifteen times a day and
cares about the one that comes back. `POST /book/structure` therefore runs in the CURIOSITY lane and
files nothing: the head does not move on a quote, asserted as absence on the head. What it writes is
the pending file, and under a home that file now carries everything the acceptance will file — the
book's two hashes, the values vector behind them as JSON that canonicalises back to its own hash, the
TICKET, the age of the board, `quoted_by` and the relayed client `request`. The fourteen quotes
nobody accepts die in `DV_HOME/tmp`.

**THE TICKET is the plan hash of the book AS THE ACCEPTANCE WOULD LEAVE IT** — this quote's MIRROR
spliced in and its pinned spot models merged, through the booking's own two seams (`splice_deal` and
`structures.pin_models`) and never a second spelling. It is what an approval signs, so it reaches
this quote and no other and an amended mirror is a new hash by construction, and it is the plan the
booking DOES leave: a fitted strip books a `Valuation Configuration` entry beside its deal, that
entry is plan, and a ticket taken without it would be a hash no booking ever reaches. Computed by
ONE function, at the quote and again at the acceptance, so the plan a decision is filed over and the
plan a booking checks cannot be two numbers. A plan does not read a spot, so the live one the
quote's own copy carries cannot move it; and a pin whose parameters the book no longer carries
refuses where the ticket is re-derived, before anything appends.

**`POST /book/quote` IS THE ACCEPTANCE**, and the booking where the policy admits it. In order, and
every refusal before anything appends: the desk's own `Quote Policy.firm_seconds`, which is a promise
to a client and comes first; the PLAN, an equality, refused where the book moved under the solve; the
PILLAR age of the board the quote was struck on, refused where a declared `pillar_seconds` says it
was already too old and refusing nothing where a home declared none; and the TICKET, re-derived from
the pending deal and this book, refused where the file no longer says what was quoted. Then the
MARKET, which is REPORTED as `{pinned, current, moved}` and NEVER refused: between a quote and the
client's word the board may move materially and we follow the spine as usual, the desk's own window
being the promise that bounds it. `firmness` is the one module that answers all four, and its
document is one window and no second — `values_seconds` and `plan_seconds` are refused by name at the
declaration, so a desk carrying the old shape is told where its question went. **THE LAST THREE ARE
READ INSIDE THE EDIT CLOSURE**, against the document the trade lands in: the closure reads under the
book lock and the read above it does not, so a check taken up there is a statement about a book
nobody booked into — and the desk window stays outside,
because it is about the quote rather than about the book.

**The board's age is the identity's own projection.** `board_age` reads `schema.quote_stamps`, which
is every clock `without_clocks` drops: the stamp on each quote row AND the value-bound `Date` on each
price factor. A book bootstrapped once and handed on carries its surfaces' `Quote_Timestamp` and no
quote rows at all, and it has an age for exactly the reason it has an identity.

**One write, and the record goes first.** Inside one `Book.transact` closure — the whole act under
the book lock, one pass and no redo, because an edit that APPENDS before it writes cannot be re-run
on a document it did not append against: a redo would meet the plan check its first pass passed and
leave the record holding a quote and a fill for a write that lost, the file without the trade, and
every retry answering "already booked" at a path that does not exist. The price is that a tick, a
bootstrap or a competing booking waits out the act — 73 ms on a young record, 199 ms on one two
thousand events in. **The line is AN EDIT THAT APPENDS BEFORE IT WRITES TAKES `transact`**, so
`/book/deals` takes it too WHERE A HOME IS CONFIGURED: its fill and its amendment are appended
inside the closure, and a redo whose second pass refuses what the first passed — a deal raced in
beside this one is newly said about — would leave the record holding a fill the file never took.
With no home nothing appends and the edit keeps `Book.mutate`'s optimistic passes, which is what
the lock costs: 2 ms of read and write against the 80 ms a booking's own verdict takes. What a lock
orders is the writers that ask for it: **the hub is the gatekeeper and the file is its projection**,
so a write that asks for neither — an editor saving `book.json` behind the service — is not a race
to be ordered but a divergence `/book/reconcile` names. In that closure: `quote_filed` under the
ACCEPTOR's seat, carrying the ticket and the pending values; then the tier step; then the `fill`
under the acceptor with the quote id as its execution reference; then the book file. The pending file
gains `accepted: {lsn, ticket}` in the same act, which is what lets a decision seek to ONE frame
instead of folding every quote the desk has ever struck, and `booked: {lsn, deal_path}` when the
fill lands. A retried acceptance coalesces every event onto the LSN it already has — the tuples
carry no `effective_time`, so this is asserted rather than built, and it is what a desk tier's
second act does. One that already BOOKED is told so by name: the booking was itself a move of the
book, so the plan check would send a salesperson to re-quote a trade the desk has already done, and
`booked` on the file answers `{written: false, booked}` before any of it. And AN EVENT APPENDED
BEFORE A LATER REFUSAL STANDS: the client took the price, which is a fact whatever the workflow then
says, so a held-back booking still answers `accepted` and leaves the file byte-identical.

**The tier step is enforcement by declaration.** No `tiers` policy in force is today's flow and the
fill lands unsigned. With one in force `spine.route_ticket` composes the document, the evaluator, the
market names standing and — for a tier wanting a human — the verdicts over this ticket, in ONE open
of the log. **Both folds ADVANCE rather than re-walk** (`spine.advancing`): this runs on every
acceptance, inside the write closure and so under the book lock, and the rows it
opens a body for are the very ones it mints — an approval per desk-tier booking and one per automatic
signature. Folding them from genesis costs **0.164 ms per decision filed**, the same per-body price
the `quotes` fold pays, so a desk two thousand decisions in would pay about 330 ms to book a trade.
Holding the `(lsn, state)` pair the fold already takes — one per projector, under the HISTORY they
were taken on (the genesis event hash, so a home re-minted in place drops them), dropped when it
changes — makes it **0.026 ms per decision**, 6.3× cheaper and about 57 ms at two thousand. Cheaper
but not flat, and the residual is not the walk: 0.011 of that is the envelope walk every fold pays
per EVENT, and the other 0.015 is `_from_seed`'s canonical copy of a state that grows with the
decisions — the same copy that makes the `activity` strip 219 ms where folding it costs 38. A seed
at a close is the remedy for both, and is the roadmap's row: a fresh process would start where the
day started, and the state copied would be the day's rather than the record's.

The ticket states its notional in its OWN currency, which needs no market data, and in every
other this book can VALUE it in, crossed at the book's own spots; a currency whose cross is not a
finite positive number — a spot block installed and not yet ticked carries its declared zero — is
simply not among them, so a cap in it fails ITS tier by name instead of taking down a closure that
has already filed the quote. A tier naming a SEAT signs under it between the acceptance and the fill,
three facts at consecutive LSNs; a seat the capabilities document does not scope for `approve` lands
a `capability_denied` in the writer's own voice, the acceptance stands, nothing books, and the answer
names the seat and the grant it lacks. A tier naming no seat answers `{written: false, accepted,
tier, waits_on}` — a NORMAL return, because the model's next move is to get it signed — and a ticket
no tier admits answers `refused` carrying every sentence of the route it took. **That last wears the
VALIDATION refusal's shape**, `{written: false, refused: [...]}`, and `accepted` is what tells them
apart: one touched nothing, the other is a price the client took that the desk's own policy will not
book. **No automatic rejection is filed**: a rejection is a seat's decision and the policy names no
seat for one.

**The decision verbs.** `POST /book/quote/approve` and `/reject` take the quote id and file over
`accepted.ticket` under the caller's own seat. A quote nobody accepted refuses by name — there is no
ticket to rule on before the client has taken the price — and `spine.quote_at(lsn, quote_id)` is
`Log.frame_at`'s seek by BYTE OFFSET followed by one body, refusing where that position holds another
type or another quote, so a pending file copied from another home is caught rather than believed and
the read costs the same whatever the desk has quoted. Approving twice is one fact, and the LATEST verdict
stands: a rejection filed after an approval is what the record says, and an approval after that moves
it back.

**The binding closes its own warning.** `book_deal` carries the `quantity`, `execution_reference` and
`actor` a recorded desk requires, `amend_deal` and `solve_structure` carry `actor`, `book_quote` is
documented as the acceptance, and `approve_quote`/`reject_quote` are the second seat's. The server
`INSTRUCTIONS` and the `quote_a_structure` prompt say the walk: quote, the client's word, accept, and
where the desk's policy wants a second seat, its approval, then accept again.

## What is not built yet

No DuckDB and no reading plane — increment 4 ships none, and every question a desk asks the record is
a fold. The market, close and settlement verbs are **5c**, and so is queue admission. No doorbell and
no generated MCP binding — **6 and 7**. No network anywhere
yet: tokens are verified, never fetched, and no write path is exposed beyond localhost. No class-key
rotation (rewrap adds recipients; rotation is a later logged event). The external anchor hook is the
checkpoint pair on `DV_Spine status`; wiring it to an anchor target is deployment data, out of scope
by the design's own sentence.

Two boundaries of increment 3's own, declared rather than discovered. `pin_result` reads its tolerance
policy and re-executes BEFORE the writer adjudicates the append, so an unscoped actor can make the hub
pay for one execution it will then refuse; the fix is not a second authorization check inside the verb
(one evaluator, one place) but queue admission, which the design puts under its own capability in
increment 5. And a STANDING run must post its job document — a `plan_id` names a parse,
`Context.save_json` is explicitly not a complete round trip, and a provenance chain whose first link is
a document nobody can recompile is worse than none — so it refuses by name.
