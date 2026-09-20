# The Spine

`derivus_spine/` is the append-only book of record being built around the engine — the center a desk
box is the edge of. The full seven-increment design lives in the owner's brief outside the tree; this
page documents what is BUILT, which is **increments 1, 2, 3 and 4**: the log, the blob store and the
chain (riding on them: identity, capability enforcement and key custody), on top of those the booking
verbs, the attestation lanes and the two-dimensional firmness check, and over all of it the
projections, the diary and the book file's pin. No network — a library, a CLI, five delegators on
`Context`, three read verbs on the service, and 253 gates.
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

103 in four files (`test_spine.py`, `test_spine_canon.py`, `test_spine_imports.py`,
`test_spine_store.py`; the glob `tests/test_spine*.py` is the wider ten-file set worth 237 of the 253
above, and `tests/test_diary.py` carries the rest),
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
and not a parameter; the Bloomberg tick is telemetry; and `/book/structure` is standing, where the
attestation IS the `quote_filed` it produces. Three consequences are gated rather than assumed, and the
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

**A quote pins TWO hashes and firmness is checked in two dimensions.** `quote_filed` carries the values
vector the quote was struck on and the book plan its marginal charge was solved against, beside the
solved coordinates, the edge, and — optionally — the relayed client request as an erasable field, which
needs no mechanism of its own because every body here is sealed. The two hashes are the BOOK's, taken
before the live spot lands on the quote's copy, because what an approval asks is whether the market and
the book this trade would LAND against have moved. `/book/quote` answers separately on each: VALUES (the
market moved, or its pin aged past the cadence that refreshes it) and PLAN (the book moved, or its pin
aged), each refusal naming its own dimension and its own remedy. They are disjoint by MEASUREMENT — a
vol tick moves `values_hash` and leaves `plan_hash` bit-identical, and the gate asserts that on a
fixture that ticks a vol with no booking in sight. The desk's own `Quote Policy.firm_seconds` is NOT
superseded: it is a promise to a client, these two are the record's statement about provenance, and an
approval passes all three. The windows are policy data (`values_seconds` defaulting to one tick of the
cadence, `plan_seconds` to the desk's ten minutes), declared as a hashed blob and resolved by fold.

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

## What is not built yet

No DuckDB and no reading plane — increment 4 ships none, and every question a desk asks the record is
a fold. No tier policy, no doorbell, no generated MCP binding — **5 through 7**. No network anywhere yet:
tokens are verified, never fetched, and no write path is exposed beyond localhost. No class-key
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
