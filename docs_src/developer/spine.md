# The Spine

`derivus_spine/` is the append-only book of record around the engine — the center a desk
box is the edge of. **1 through 7 — the design is built**: the log, the blob store
and the chain (riding on them: identity, capability enforcement and key custody), on top of those the
booking verbs, the attestation lanes and the firmness check, over all of it the
projections, the diary, the book file's pin and the desk's own readers of them, beside those the
tier policy, its evaluator and the verbs that file a decision and a close, through all of it the
quote lifecycle — a price recorded when the client accepts it, routed through the desk's own workflow
before it books — over the compute itself a queue that asks who is submitting before it runs
anything, around the whole of it a REPLICA: a read-only copy that pulls the hub's own frames,
verifies them where it stands and is told when there is something to pull, and over all of that an
ORACLE and the bank it reads — a mock bank founded through the CLI and played by its seats through
the binding over three closes, with an adversary and scripted faults beside it, held to fourteen
invariants. **8** adds the paper the book trades under -
its legal entities and agreements - with every position keyed where it sits, read there and priced
at its net, and seeds filed where a deployment says; **9** the money the settlements moved, what
every position cost, and the desk's P&L between the closes it marks; and the seats - seven verbs
granted at the nodes of a declared tree, the hub's own facts in the writer's voice, and a trade's
pending read rather than filed; and the collateral each agreement calls on a marked close, its
balance written into the plan. A library, a CLI, nine delegators on `Context`, eighteen read verbs
and ten write verbs on the service, and 412 gates.
Nothing here imports the engine, and exactly one module under `derivus/` imports `derivus_spine`:
`derivus/spine.py`.

## The package, and the one dependency

A sibling package on the `derivus_mcp` terms: in the wheel, never importing the engine. Its import
surface is **stdlib plus `cryptography`** (AES-GCM sealing, Ed25519 checkpoint signatures) and nothing
else, held by an AST gate over every module and a subprocess gate proving `import derivus_spine` pulls
no torch and no `derivus.*`. The extra is `pip install derivus[enterprise]`, orthogonal to `desk`.
`DV_Spine init | verify [--chain-only] | checkpoint | status | follow | oracle` is the console script; the home is
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

The vocabulary is CLOSED: seventeen trading fact types (`checkpoint` among them), the two custody
types, increment 3's three provenance types, the paper's two and the writer's own reserved one, with
validators naming the missing or surplus field. A consequence-shaped submission ("knocked_out") is
refused with the closure stated — knocks and expiries are projections, never events.

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

106 in four files (`test_spine.py`, `test_spine_canon.py`, `test_spine_imports.py`,
`test_spine_store.py`; the glob `tests/test_spine*.py` is the wider fifteen-file set worth 379 of
the 412 above, and `tests/test_diary.py` carries the rest),
all real stores in temp dirs, every fault injected by doctoring DATA on disk. The shapes worth naming:
three tampers on three copies, each caught by a different layer (body byte by the chain, envelope field
by the AAD, record_time by a keyless replica); a re-forged tail caught by the interior binding AND its
dual caught by the stale tag, each proven independently load-bearing in the posture where the other is
absent; the design's synthetic book (late booking, backdated amendment, republished fixing under one
(index, date, source) key, superseding close) driving all seventeen fact types through the writer
with as-of provably departing from as-at; and restore as file-copy, both replica shapes re-verified on
the far side.

## Increment 2 — identity, attribution, key custody

**Identity is bought, not built.** `identity.py` VERIFIES an OIDC ID token against a JWKS the deployment
hands in as data — nothing is fetched, verification stays local — under an RS256/ES256 allowlist that
refuses `none` and every HMAC by name before any key is selected (the alg-confusion attacks are gates,
not warnings), with `kid` selection across key types, the ES256 raw-`R||S` signature contract,
`exp`/`nbf` on an injectable clock, and OIDC Core's `azp` rule so a co-audienced client cannot replay
its own token as a spine credential. The subject reference is the token's `sub`, pseudonymous by the
the design's rule; display names live in `names.json`, a mutable side table OUTSIDE the log whose erasure is
gated to leave every chain byte identical.

**Capabilities are one document and one pure function.** The document — grants of (verb × node) over
`validate | book | approve | mark | document | settle | admin`, plus READ over entitlement classes —
is a hashed blob declared through the ordinary writer (`policy_declared` with the RESERVED policy name
`capabilities`), each declaration a complete replacement, resolved by fold-at-LSN so "could X do Y in
March" replays from the log like every other question. A node is a book or a path of named segments
under it, and a grant reaches every path below its node. Enforcement activates BY DECLARATION: a home
with no document runs as the single-user instrument it is — which is why every increment-1 home and
gate is bit-for-bit untouched — and once one is in force the writer refuses an unscoped append AND logs
the refusal as a `capability_denied` fact in the writer's own voice, because a decision is a fact. A
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
an attestation is the hub's own, filed in the writer's voice, a quote takes `book`, a promotion takes
`approve` — giving standing to a tuple this hub never witnessed is a second pair of eyes on somebody
else's claim. Two field kinds were added for
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

**The diary is the compile's own schedule, re-emitted.** `spine.diary(context)` runs the `Diary`
calculation, the COMPILE half of a base valuation — the market built, every deal constructed and its
schedules bound — stopped before the structure resolves; it reads what it just bound through
`utils.walk_schedules`, which is the walk `utils.bind_schedules` itself takes. One walk, so the leg
names a settlement reference is built from cannot drift from the binding.

**ONE SPELLING OF A PAYMENT.** A row is one payment per `(leg, pay day)`, never one per schedule
row, and its amount is `pricing.fixed_payments` — the line `pv_fixed_cashflows` discounts, factored
out so the two cannot be two numbers: the rate coupon and the fixed amount of every row sharing that
day, summed, and compounded where the leg's own terms compound. A bond repaying its principal with
its last coupon announces the sum, and two accrual sub-periods paying on one day announce one
payment. A schedule carrying RESETS determines nothing — its amount is a floating pricer's and the
diary does not spell a second one — so its rows read `amount: null, determined: false`: **a null
amount is never written as 0.0**, and `spine.export_settlements` — which takes the diary, one
market hash and the day the file settles, and reaches nothing else — refuses an undetermined row,
and a row naming no currency, BY NAME rather than instructing a wrong payment. `due_before` has no
default: a settlement file is struck FOR a day.

An `expiry` row carries `needs`, which is `election` where the deal type's terms vest an exercise in
an actor and null where a fixing determines the payoff. What a deal settles that its compile builds
no schedule for is DECLARED on its fields: a date - a field, or a table's date column - carries
`settles`, one `schema.Cash` per payment that day, naming the field its currency is in, the field
stating its amount where the terms fix it, and its sign, as its pricer books it through
`cash_settle`. A forward's settlement date declares its two legs, the one received and the one
paid, each determined; an FX swap's near and far dates, theirs; a non-deliverable or equity
forward's, an accumulator's and a TARF's settlement days and an autocall's coupon dates, one open
payment each; an option's delivery or settlement date, its payoff, falling back to its expiry. The
most specific block declaring any stands, so a binary's own settlement date stands in place of the
expiry its option base declares, and a declared day behind the base date has been paid, as a
schedule's paid row is dropped. `get_settlement_currencies()` is the reval-date accumulator — a
barrier registers its monitoring days in it — so it is not a payment ladder and is never read as
one. AN OPTION WAITS FOR TWO THINGS after its expiry: a `fixing` row on the underlying, because a
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
the roll, as settling August. A date does not move. The calculation names each row's deal by its
`Reference` and knows nothing of the record; `spine.diary` stamps the instrument and asks for the
key, so on a box without the extra a row carries `key: null` and everything else. A book rolled
PAST a coupon keeps every surviving row's key and loses only the row that was paid.

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
walks the job's deals, and for each one whose type declares an observation table — the type's own
`observes` (`schema.OBSERVES`), read by the compiler and the diary alike and never spelled twice —
writes the fixing in force at `lsn` into the cell that type declares it in, for a day ON OR BEFORE
the job's own base date: a print carries its date as text, so a forward-dated one is a legal fact
and writing it would price a barrier as observed on a day that has not happened. An FX table
observes a pair, and every `FxRate` is a rate against the book's base, so it names its non-base
legs - one for a base-relative pair, two for a cross - and its cell is the ratio of their prints,
the base's own rate being one, written once every leg is printed and no divisor prints zero. A
book holding an FX table therefore needs `FxRate.<ccy>` ordered in the fixings policy for each
non-base leg, as every observed index does. `/execute` and
`/prepare` both compile before they hash, so a plan named once and ticked as a delta is the plan
that runs; without a home, and where the record holds no position and no deal declares an
observation table, `compiled_job` returns its argument and the edge is what it always was. Which source is authoritative is POLICY:
`fixings_at` resolves across sources by the order the reserved `fixings` policy declares, and an
index A PLAN COMPILES AGAINST that the policy does not name refuses by name — a fixing whose
authority nobody vouched for is not a fixing a plan may use, and the refusal reaches a desk as a 422
in the record's own words. A READING refuses no print, its COMPILE HALF INCLUDED: `compiled_job` takes a
`strict` flag, and every read passes it false — the book is compiled as written plus whatever
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
regenerated from the positions fold rather than compared against it — waited on a netting set's own
node being in the record, its CSA terms being what no fact carried then; increment 8 files them as an
agreement's terms, and the file stays the materialisation, reconcile how it is checked, until the
tree is hydrated.

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
the official close standing per market - its latest by day - with the day it is for and the LSN of
that day's close it restated, the declared names,
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
one falls in no tier at all, which is an answer rather than a default. A scope and three checks.
`scope` is a node, read first: a tier covering one is read only for a ticket booking into it or under
it. `max_notional` is an amount AND the currency it is an amount of, read against the notionals the
CALLER states, so no check here reads a market and a ticket whose notional is not stated in that
currency FAILS that tier by name — the safe direction, and the one a conversion inside a policy
check would lose. `max_tenor_years` is the ticket's own. `market` is the values vector the quote
pinned, which must be the one standing under the name the tier prices on. A tier is AUTOMATIC — its
approval the hub's own act, filed in the writer's voice — unless it declares `four_eyes`, which wants
a seat other than the booker to sign; a tier naming a `seat` is refused by name at the declaration.

**Staleness is `firmness`'s, and a tier restating it is refused by name.** How old a board may be is
`pillar_seconds`, enforced on every booking before any tier is read, so a document or a tier carrying
it — or `firm`, the verdict it answers, or either of the two windows 5b retired — is
refused where it is declared rather than making one question two standards. `designations` is the
other section: process → market name, closed to the processes something actually resolves by name
(`settlement_export` is the one this increment binds), because a name nothing reads is a rule nobody
enforces. Neither a designation nor a tier's own `market` may be a `private/` name: a designated
process resolves the firm's board, a tier's market decides whether the hub signs AUTOMATICALLY,
and neither rests on a board one seat declared for itself.

**WHAT IS STORED IS THE COMPLETED DOCUMENT**, the practice `parse_firmness` set. The two defaults
this shape has — `four_eyes` false, and an empty `designations` — are written in before the bytes
are hashed, so one workflow is ONE BLOB however the operator spelled it and two desks declaring the
same rules do not get two governance histories of one decision. A cap is a MAXIMUM on every axis — a
ticket exactly at one passes and the smallest step over it does not — and a notional is an AMOUNT of
a currency and never a sign, so a zero or a negative one is refused rather than clearing every cap
there is.

**The evaluator is pure, and nothing calls it yet.** `derivus_spine/tiers.py` is `firmness.py`'s
shape — `assess` returns a verdict, `check` raises `TierRefused` carrying the same sentences — over
a parsed policy, a ticket and the market names standing, holding no log, clock, store or home, which
the gate asserts on the signatures AND on the module's own imports rather than on the docstring. The
verdict names the tier, every check read with the value measured and the bound declared, and one
sentence per failure of every tier tried, which is also the route the ticket took. A check that could
not be MADE is a failure and never a pass. `standing_approval` is the verdict half: **THE LATEST
VERDICT STANDS**, by LSN, because a verdict is never withdrawn, so a rejection filed after an approval
is what the record says last; under `four_eyes` the booker's own verdicts are not read. Scope is not
re-checked there — the writer refused an unscoped approval at the append, so every verdict in the
fold was already a seat's or the hub's own.

**The decisions are verbs now, and so is the close.** `approve` and `reject` file the two verdicts
the vocabulary has carried since increment 1 with nothing filing them; both demand `approve` over
the node the ticket books into - the job's own book, or the `portfolio` a verdict names - and an
unscoped seat is refused with the denial landed, like every other verb. One seat signing one
plan twice is ONE fact — the semantic tuple carries no clock of the writer's own — while a second
seat signing the same plan is a second, because who signed is part of what was said. `declare_close`
blobs its values exactly as `declare_market` does, and a second close on one market SUPERSEDES the
first rather than correcting it, so a day restated is two facts and a fold taken as at the first
still answers what it answered. `quote_filed` gains the optional `ticket`: `plan_hash` is the BOOK's
plan, which is the right question for firmness and the wrong one for identity — two quotes struck
against an unmoved book pin the same one — while `ticket` hashes the plan the book WOULD have with
this quote's mirror spliced in together with the quote id, so an approval over it reaches this quote
and no other and an amended mirror is a new hash by construction. A body carrying none validates
exactly as it did. The eighth
projector, `quotes`, reads those bodies: one row per quote id, carrying the SEAT that struck it off
the envelope, since no body holds one, a second filing under one id standing by the as-of key every
other keyed row here stands by, and the rows read in LSN order rather than their ids'. **It is the
one projector whose rows grow with the desk's own activity**: it opens a body per quote, about
0.16 ms each, so the fold is 7.7 ms on a two-thousand-event home holding three quotes and 327 ms on
one holding 1,978 — the envelope filter keeps it off every other event and nothing keeps it off its
own.

**A market is resolved BY NAME for the first time.** `spine.resolve_market(name, actor, process)`
composes the `markets` fold, the blob store and `read_values` — every piece of which existed, and
was composed in exactly one place — and answers `{name, values_hash, values}` as the objects
`patch_market` takes. A name nobody declared REFUSES rather than falling back on whatever market is
loaded, which is the whole point of binding a process to a market by name; with `process` named, it
must be the market the tiers policy designates for that process, and a home designating nothing
refuses too. **A NAME RESOLVES TO ITS LATEST DECLARATION BY AS-OF KEY, and an official close is
one way of declaring one**: a close moves what the name answers and a later declaration moves it
back, so a reader cannot be handed yesterday's board on a market the desk has since closed. The
close a market stands on is its latest BY DAY, so a past day restated never unseats a later day's. A
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
PLAN the acceptance leaves the book at, the age of the board, the `portfolio` it books into (stated
with the price, the book's own by default), `quoted_by` and the relayed client `request`. The
fourteen quotes nobody accepts die in `DV_HOME/tmp`.

**THE PLAN IS THE BOOK AS THE ACCEPTANCE WOULD LEAVE IT** (`pinned.plan`, `mirrored_plan`) — this
quote's MIRROR spliced in and its pinned spot models merged, through the booking's own two seams
(`splice_deal` and `structures.pin_models`) and never a second spelling: a fitted strip books a
`Valuation Configuration` entry beside its deal, that entry is plan, and a plan taken without it
would be one no booking ever reaches. The acceptance reaches it by booking and refuses where the two
are not one number, so the file still says what was quoted. A plan does not read a spot, so the live
one the quote's own copy carries cannot move it; and a pin whose parameters the book no longer
carries refuses before anything appends. The TICKET an approval signs is minted AT the acceptance:
`spine.ticket` over that plan and the fill as it is filed - the quote and the fill carry the one
ticket, and no other trade ever does.

**`POST /book/quote` IS THE ACCEPTANCE**, and the booking where the policy admits it. In order, and
every refusal before anything appends: the desk's own `Quote Policy.firm_seconds`, which is a promise
to a client and comes first; the PLAN, an equality, refused where the book moved under the solve; the
PILLAR age of the board the quote was struck on, refused where a declared `pillar_seconds` says it
was already too old and refusing nothing where a home declared none; and the PLAN the acceptance
leaves the book at, refused where it is not the one the quote pinned - a deploy that changes how a
document is read moves that hash too, so the pending quotes are drained before one, or re-quoted.
Then the MARKET, which is REPORTED as `{pinned, current, moved}` and NEVER refused: between a quote and the
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
to be ordered but a divergence `/book/reconcile` names. In that closure the mirror is spliced and
validated first, a refusal touching nothing; then ONE ACT OF THE WRITER (`spine.routed`):
`quote_filed` under the ACCEPTOR's seat, carrying the fill's ticket and the pending values; then the
tier step; then the `fill` under the acceptor with the quote id as its execution reference, at the
quote's portfolio, one posted with the acceptance refused by name; then the book file. The pending
file gains `accepted: {lsn, ticket}`, which is what lets a decision seek to ONE frame instead of
folding every quote the desk has ever struck, and `booked: {lsn, deal_path}` when the fill lands. An
acceptance retried is the same act, every event of it landing on the LSN it already has - after a
tier refused it, after the desk moved its tiers, after the files it wrote were lost - so one quote
files one `quote_filed` and books once. One that already BOOKED is told so by name: the booking was
itself a move of the book, so the plan check would send a salesperson to re-quote a trade the desk
has already done, and `booked` on the file answers `{written: false, booked}` before any of it. And
A QUOTE FILED BEFORE A TIER REFUSES STANDS: the client took the price, which is a fact whatever the
workflow then says, so a booking no tier admits still answers `accepted` and leaves the file
byte-identical.

**The tier step is enforcement by declaration.** No `tiers` policy in force is today's flow and the
fill lands unsigned. With one in force `spine.routed` composes the document, the evaluator, the
market names standing and — for a tier wanting a human — the verdicts over this ticket, on the
writer's own handle; a home whose decisions fold names no tiers policy walks nothing more. **Both
folds ADVANCE rather than re-walk** (`spine.advancing`): this runs on every booking and every
restrike, inside the write closure and so under the book lock, and the rows it
opens a body for are the very ones it mints — an approval per desk-tier booking and one per automatic
signature. Folding them from genesis costs **0.164 ms per decision filed**, the same per-body price
the `quotes` fold pays, so a desk two thousand decisions in would pay about 330 ms to book a trade.
Holding the `(lsn, state)` pair the fold already takes — one per projector, under the HISTORY they
were taken on (the genesis event hash, so a home re-minted in place drops them), dropped when it
changes — makes it **0.026 ms per decision**, 6.3× cheaper and about 57 ms at two thousand. Cheaper
but not flat, and the residual is not the walk: 0.011 of that is the envelope walk every fold pays
per EVENT, and the other 0.015 is `_from_seed`'s canonical copy of a state that grows with the
decisions — the same copy that makes the `activity` strip 219 ms where folding it costs 38. Holding
no pair, `advancing` starts from the newest seed at or behind its position
(`projections.latest_seed`), and `DV_Spine seed` mints the decisions and markets folds with every
other but the strip's, so a fresh process walks from where the day started; the copy stands, a seed
holding the record's state rather than the day's.

The ticket states its notional in its OWN currency, which needs no market data, and in every
other this book can VALUE it in, crossed at the book's own spots; a currency whose cross is not a
finite positive number — a spot block installed and not yet ticked carries its declared zero — is
simply not among them, so a cap in it fails ITS tier by name instead of taking down a closure that
has already filed the quote. The ticket's portfolio is the node the quote named, the book's own by
default. An AUTOMATIC tier is signed by the hub in the writer's own voice between the
acceptance and the fill, three facts at consecutive LSNs, the fill carrying the ticket - the
workflow an admin declared decides it, so no capabilities document can hold it back. Under a
four-eyes tier the trade BOOKS and reads PENDING, the answer carrying its `tier` and `waits_on`,
until another seat signs; and a ticket no tier admits answers `refused` carrying every sentence of
the route it took. **That last wears the VALIDATION refusal's shape**, `{written: false, refused:
[...]}`, and `accepted` is what tells them apart: one touched nothing, the other is a price the
client took that the desk's own policy will not book. **No automatic rejection is filed**: a
rejection is a seat's decision and the policy names no seat for one.

**The decision verbs.** `POST /book/approve` and `/reject` take a quote id and file over
`accepted.ticket` under the caller's own seat, where the booking sits. A quote nobody accepted
refuses by name — there is no ticket to rule on before the client has taken the price — and
`spine.quote_at(lsn, quote_id)` is
`Log.frame_at`'s seek by BYTE OFFSET followed by one body, refusing where that position holds another
type or another quote, so a pending file copied from another home is caught rather than believed and
the read costs the same whatever the desk has quoted. Approving twice is one fact, and the LATEST verdict
stands: a rejection filed after an approval is what the record says, and an approval after that moves
it back.

**The binding closes its own warning.** `book_deal` carries the `quantity`, `execution_reference` and
`actor` a recorded desk requires, `amend_deal` and `solve_structure` carry `actor`, `book_quote` is
documented as the acceptance, and `approve_quote`/`reject_quote` are the second seat's. The server
`INSTRUCTIONS` and the `quote_a_structure` prompt say the walk: quote, the client's word, accept,
and where the desk's policy wants a second seat, its approval of the trade booked pending.

## Increment 5c — the mark, the close, the settlement file, and the queue that asks first

**THE MARK IS THE BOOK'S OWN VALUES UNDER A NAME.** `POST /book/markets` hands
`Context.declare_market` the live book's projected vector, so what a name stands on is the bytes
`values_hash` already addresses and a mark and a quote pin one number. Officialness is a property of
the NAME and the record is what enforces it: a seat the capabilities document does not scope for
`mark` is a 422 in the writer's own words with the denial landed as a fact, and a
`private/<subject>/<name>` board whose subject is not the declaring seat is refused at the verb before
anything appends. Neither rule is restated in the endpoint, because a second place to get a rule right
is a second place to get it wrong. `spine.resolve_market` now answers the POSITION of the declaration
in force beside the name and the hash, so a file struck on a board says where that board was declared.
**THE OWNER RULE IS EVERY VERB'S THAT MOVES WHAT A NAME STANDS ON**, the CLOSE included: a name
resolves across the declarations of it and the closes on it alike, so a close inside another
subject's namespace would be a board its owner never declared and is the only reader of — which is
the second answer to who owns one board that the rule exists to prevent.

**THE CLOSE RUNS BEHIND THE CHECK.** `POST /book/close` declares what `GET /book/close/check`
answers, on the day it answers for: `market` defaults to `official` and `date` to the book's own
`Calculation.Base_Date`, parsed by the check's own reader, and a day the check calls illegal refuses
naming what it waits on with NOTHING APPENDED — a close over a payment nobody settled, a fixing nobody
printed or a payoff nobody elected is a clean bill nobody earned. Both defaults are CONVENTIONS, so
only an omitted key takes one: a date that names nothing is refused rather than defaulted, since a
string compare would call the empty one legal. The verdict is `close_verdict`, the rule as a pure
function over rows, so the read verb and the declaration answer ONE verdict about ONE document: the
close compiles the book it read and asks, where calling the read verb again would judge a book the
close is not struck on. That compile is the CALLER's job on the queue, like the export's. A second
close for one day SUPERSEDES that day's first, and the answer carries the `supersedes_lsn` of the
close of its day it stands over — read off the record's closes rather than off the close just filed,
because what a close stands over is a question about the record.

**THE SETTLEMENT FILE NAMES NO MARKET AND CANNOT.** `POST /book/settlements` takes the day it is
struck FOR — `due_before`, which has no default — and nothing else: which board it is struck on is the
market the `tiers` policy DESIGNATES for `settlement_export`, resolved by that name, so pointing the
export at another is unrepresentable rather than merely refused. A home designating nothing, and one
designating a name nothing stands under, refuse at SUBMISSION with the declaration that fixes it,
before a row is compiled. The rows are the DIARY's — the same job on the compute queue at a base
valuation's cost class that `GET /book/diary` caches, never a second compile path — and the answer is
the exporter's own, plus the market block and the count: an undetermined amount, and a row
naming no currency, refuse by name, because instructing a payment of zero is a wrong payment rather
than a missing one. Two keys say two things and are spelled apart: `values_hash` is the BOARD the
file was struck on, which is the exporter's own statement over the rows it exported, and `market` is
the name that board was resolved under with the position that name stands at, which is the record's.
**ADMISSION IS ASKED BEFORE THE DIARY CACHE IS READ**, here and on every reader of it: a warm cache
reaches no queue, and a settlement file a desk instructs payments from must not be a function of who
asked first. The miss therefore asks twice, which is one fold against the compile it guards.

**THE QUEUE IS THE HUB'S COMPUTE, AND IT ASKS FIRST.** One check, in `ComputeExecutor.submit` before
the job is enqueued, where every queued job passes — the diary, the tick, a what-if, a solve, an XVA
set and a standing run alike. What it asks for is **THE SCOPE THE WORK NEEDS**: a standing run is
numbers a fact is about to cite, a mark of the book its job values - the hub attests them in its own
voice - so it asks `mark` over that book, and the ordinary desk seat is turned away before the Monte
Carlo rather than after it, which is where increment 3's boundary left it. `/book/marks` hands the
queue the book its marks value, so the seat is asked `mark` over that book and never over the
marks job's own name. Every other lane mints nothing and asks `validate` over the book its document names or any
node under it, a seat working one node of a book being admitted to price the book.

Enforcement activates BY DECLARATION, as at the writer: with no capabilities document in force every
job is admitted and the box is the single-user instrument it was, and under one a request naming no
seat is refused by name — a job nobody signed for is one the record could not attribute, and it is
never taken for the deployment's own. **THE SEAT IS THE REQUEST'S**: `/execute`, `/book/price`,
`/book/solve`, `/book/model`, `/book/xva`, `/book/setup`, `/book/structure`, `/book/close` and
`/book/settlements` all take an `actor` and are admitted under it, so a stranger reaching this box is
a stranger to the record too. The POLL PATHS name `spine.hub()`, `DV_SPINE_ACTOR` — the diary read,
the tick and the securities verification are the deployment's own and that name is the whole of what
the metronome has, which is also why the one grant a ticking desk cannot withhold must not be the
grant that admits everybody.

**THE VERBS IMPLY NOTHING ABOUT EACH OTHER**, here as at the writer,
so a document declared before this increment must now say `validate` for every seat that prices,
quotes, solves or reads a diary: a `book` grant puts paper on the record and does not buy arithmetic.
A configured home that is not a home refuses the whole queue in the same sentence it always
refused a verb with — a record nobody can open is one no job on this box gets past.

A refusal is a fact: `SpineLog.refuse` is the denial verb IN PUBLIC, so the queue lands the same
`capability_denied` the authorization hook lands rather than learning to speak the writer's own
voice, and a repeated refusal coalesces onto the LSN it already has. The job never reaches the
executor and no result is stored — counted, never believed — and the caller gets a 422 in the
record's own words.
**This closes increment 3's own boundary**: `pin_result` re-executes before the writer adjudicates the
append, and it has no HTTP verb, so the queue was the whole of the path an unscoped actor had to this
box's arithmetic and the queue now asks first. The cost is a fold and NOTHING CACHES IT:
`capability.state_at` is 1.2 ms at 21 events and 20.5 ms at 1,994, with or without a document, while
the cheapest thing on this queue is a diary compile — the walk itself is the log's missing seek, which
is one row on the roadmap for every reader that pays it.

## Increment 6 — the seek, the replica, the doorbell

**A PAGE COSTS THE PAGE.** `SpineLog.frames(start_lsn=)` reached its first row by parsing every line
below it; it now SEEKS — one open, one `seek` onto the byte offset the log already held per LSN, and
a walk forward from there, re-globbing the segments exactly as before so a reader still sees the
writer's next frame. The offset is a LOWER BOUND and never a lookup, because a handle routinely
outlives another process's append: a page starting past this handle's head is answered from the
head's own offset and what came after is reached by the walk. Never N `frame_at` calls, which
re-open the file each time and measured slower than the walk they would replace. At 2,005 events a
ten-row page at the head goes **7.50 ms to 0.20 ms**, a two-hundred-row page 7.48 to 0.90, and a page
that answers nothing 7.40 to 0.16; the whole walk is unmoved, which is the point. Every one of the
nine callers gets faster and none changes shape. What the desk's beat still pays is the HANDLE: a
read opens a log and opening one scans every segment (11.3 ms at 2,005 events), so `/book/status`
goes 23.5 ms to 14.2 and an empty strip page 20.9 to 13.5 — the reading half closed, the open's own
half on the roadmap with its number.

**A REPLICA IS A READ-ONLY COPY THAT VERIFIES ITSELF.** `SpineLog.accept(frame)` is its whole write
path and asks four things and no fifth: the twelve fields, the next position, a link to this head,
and an event hash that recomputes over the bytes offered. It does not validate against the
vocabulary — a replica of a hub running a newer one must still chain, or the first type a hub learns
strands every copy of the record — does not authorize, scope being the hub's question answered where
the fact was made, and does not coalesce on the tag, a replica writing the order the hub gave it. It
DOES claim the home, so two followers on one replica meet `WriterBusy` rather than both writing LSN
n+1. What it writes is the hub's line BYTE FOR BYTE, which is gated as the two segment files being
equal rather than as the two frames carrying the same fields.

**THE STRIP CANNOT FEED A REPLICA**, which is why 6 has a read of its own. `GET /book/activity`
serves six of a frame's twelve fields plus a declared sentence, and every one of the six it drops is
something `accept` asks for. `GET /spine/frames?since=&limit=` serves the frame itself, `body` the
base64 ciphertext it is on the platter, opening nothing — so it answers on a crypto-shredded home
exactly as it answers on the hub — with `since` the last LSN delivered, as the strip's cursor is.
`GET /spine/blobs/{hash}` serves bytes by address, `store.get` re-hashing on the way out so the
serving side needs no trust, gated on the asking seat's READ row against the blob's class — firm for
everything while classification is dormant — a home declaring no document serving everyone as every
other enforcement here does, and a seat outside the rows or a read nobody signed refused by name.

**`DV_Spine follow <hub-url> --home <replica>`** is the pull: `while the page is not empty: pull the
frames after my head, accept each`. Catching up after an hour asleep is that loop with more pages in
it and never another path, and the whole of what it costs is DURABILITY: 0.16 ms a frame to pull over
a socket against 31 ms a frame to fsync one, which is the same byte-on-the-platter price the hub
paid to write it. A replica that is up to date is one empty pull, 2.3 ms. **`--once` is bounded by
THE HEAD THE FIRST PULL REPORTED** rather than by an empty page: a desk in session is a hub that
keeps writing, and a one-shot sync against one would otherwise have no reason to return — which is
what that field on the frames read is for. `--blobs`
pulls the bytes the chain cites, which is the ENTITLED posture by construction, a citation living in
the sealed body; a chain-only follower is told to materialize a key rather than quietly pulling
nothing. The one blob that cannot be discovered is the FIRST WRAP: which blob makes this seat
entitled is a question only an entitled reader can ask of the chain, so it arrives by address, which
is what a read by address is for.

**A FOLLOWER CHECKS THE RANGE IT LANDED, NOT THE HISTORY.** A copy that took bytes off a network and
did not check them is a copy of nothing — but re-deriving the whole chain on every beat is O(the
record) per EVENT, which is the cost 6a took out of the reader put back on the follower: on a
2,516-event home a ONE-FRAME beat spent **98.5 ms** inside a from-genesis pass, and somewhere past a
hundred thousand a follower stops keeping up with itself. What a page can have broken is its own
links and the one that joins it to the frame before, since what is behind it was re-derived when it
landed and nothing here rewrites a line, so `verify_chain` re-reads exactly that off the platter and
the same beat costs **13.9 ms**, of which 13 is the handle's own scan — the roadmap's row, not this
one. The whole chain is still checked where the walk is worth it — the first catch-up of a session,
`follow --verify` on every beat, `DV_Spine verify` on demand — and the checkpoint ladder and
referential closure stay whole-history assertions, both reaching behind any range.

**THE AUTHENTICITY BOUNDARY, said out loud**: a checkpoint signs the head BEFORE it, so a replica
proves authenticity up to its last pulled checkpoint and the frames past it are chained but unsigned.
A follower wanting signed history asks the hub to checkpoint, or waits.

**THE DOORBELL IS A NOTIFICATION AND NEVER A DELIVERY.** `GET /spine/doorbell` is an event stream
carrying `{lsn, head}` and nothing else, plus a comment on a fixed cadence so a proxy does not close
an idle one. The trigger is THE WRITER'S OWN APPEND: `SpineLog` announces where its head went at the
moment the bytes are durable, to listeners this process registered and nothing persists, so no poll
of the log runs inside the service and a watcher that refuses is logged rather than allowed to stop
an append. Because a beat carries a position and the pull asks from the position the replica stands
at, a beat dropped is covered by the next, a beat repeated is one empty pull, and a beat behind the
head is a pull that answers nothing — which is why three replicas fed one stream, one intact, one
losing every third beat, one hearing them twice and out of order, end at ONE head hash with all nine
projectors folding to byte-equal rows. The beat itself costs **0.164 ms** from the append landing to
a follower reading it over a socket on one box, and 0.014 ms where the two share a process — against
the two-second cadence it replaces. The desk's strip rides an `EventSource` on it and keeps the
two-second poll as the fallback and for the status block; the follower does the same, falling back
to its interval when the stream drops and reconnecting, with ONE catch-up either way.

**AN OPEN STREAM COSTS A TASK AND NOT A THREAD**, which is what lets a desk open as many as it has
tabs. The generator is ASYNC: it awaits the position and the heartbeat, and the writer's announce
reaches the loop from whatever thread wrote the frame. A sync generator handed to a streaming
response is driven through the thread pool instead, so every idle stream parks one of the forty
tokens EVERY other verb here shares and past forty tabs a booking, a price or a replica's pull waits
for somebody else's heartbeat. Measured on one box at 0, 20 and 50 open streams: **15.1, 15.7 and
15.0 ms** for the same frames read and **no thread taken at all**, where the sync shape put the
fiftieth reader behind a fifteen-second wait. And a client that goes away takes its listener with
it, dropped by the generator's own close rather than whenever a cyclic collection runs: fifty opened
and closed leave zero.

**The ninth projector.** `denials` reads `capability_denied` and answers `{lsn, subject, verb, book,
attempted_type}` — the only reading that says who was refused what, since the envelope says only that
a refusal happened and the strip renders one declared sentence for every one of them. Its reader is
the acceptance game's oracle, which holds a script's attempts against the refusals the record kept. A
refusal that never reaches the writer — a tier admitting no ticket, a stale board, a malformed
request — mints nothing and is not there.

**What 6 is NOT.** No peer writes, in any phase: `accept` takes only a frame the hub already chained,
so "submit to a replica" is unrepresentable rather than merely refused. No failover — the hub down
means booking stops, screens stay live off the local replica, and recovery is restarting the process.
No peer blob serving: a second entitlement-evaluating surface on a box that is not the writer, for a
phase with one deployment. And nothing of transport or authentication is built for it — the doorbell
is one `GET` on the service that already exists, on whatever port it already runs.

## Increment 7 — the oracle, and the bank it reads

**THE GENERATED BINDING WAS ALREADY THE DESIGN.** `GET /schema` publishes the desk's declarations
and `describe_structure` serves them, so a structure declared reaches a model with no edit and there
is nothing to generate; what 7 builds is the ACCEPTANCE TEST — a mock bank founded in a clean folder
and played through its days by SEATS through the binding, an adversary beside them, scripted faults
beside that, and an oracle that reads the record afterwards as a replica and answers fourteen
invariants. With the adversary off, the same days are the demo.

**THE ORACLE IS PURE OVER A HOME AND WHAT IT IS HANDED.** `derivus_spine/oracle.py` folds the
record the way every other reader does and re-runs the writer's own decisions through the writer's
own functions, so what it answers is a property of the bytes rather than of the process that made
them. What the record cannot hold arrives as DATA: the SCRIPT of what was asked - the refusals, the
lanes, the rows a settlement file instructed, the collateral calls a seat read - and two compiles
of the book an engine answered, the diary's rows and the P&L. Each invariant is one function
answering `{held, evidence}`, `held` being null for a question this home or what it was handed
could not put — a question nobody asked is never a question that held. The fourteen:

1. **copies agree** — two homes carry one head hash at the SHALLOWER of their two heads and every
   projector folds to byte-equal rows there, since a replica behind its hub has not pulled yet
   rather than disagreed;
2. **nothing outside its seat** — every frame judged at its envelope's book or at the firm
   re-adjudicated: `verb_for` naming what the type demanded and `evaluate` answering it at
   `scope_of` against the capability state BEFORE that frame, folded forward by `apply_event` — the
   writer's own step, so the same answer at the length of the record instead of its square; the
   writer's own voice is never gated and a home with no document in force evaluates yes, exactly as
   the append did. An approval in the writer's voice is held to the hub's own act: a tiers policy in
   force, and a fill or a restrike carrying its ticket after it - an untampered hub never strands
   one, every refusal its booking can meet coming first, and a crash between the two fsyncs is the
   one way an approval stands alone until the act is retried, which is right to name. The ENVELOPE
   HALF stands without a key — a type no verb declares is a write nobody could be scoped for, a type
   only the writer files under any other name is its voice forged, and the writer's name on anything
   its voice does not say is a seat wearing it;
3. **every refusal is a denial** — what the script says was refused AT THE WRITER is in the
   `denials` fold and nothing else is, matched on the four fields the writer files rather than on a
   position, since a seat refused twice coalesces. A refusal that never reached the writer mints
   nothing BY DESIGN, so it is stated as the boundary it is and its absence is what the set equality
   asserts;
4. **an amended plan is a new approval** — every ticket is one trade's: every event carrying one
   files the same trade - a retry the writer took twice, two seats filing one execution - and two
   different trades carrying one are named; and where the record carries a workflow at the position
   an amendment landed, the amendment carries a ticket of its own, so the position it restruck
   stands on no approval over the ticket it had. Whether one stands over the new one is a STATUS,
   derived and never failed. Where no tiers policy stands the desk declared no second pair of eyes,
   which is stated rather than failed;
5. **closes superseded, never edited** — THE SUPERSESSION HALF IS THE FOLD'S OWN and is not
   asserted, because it cannot be broken from a platter: `Markets.apply` COMPUTES `supersedes_lsn`
   off the close of the same day filed before the frame and stands a market on its latest close by
   day, ties by LSN - so a close restates its own day naming exactly the position it stood over,
   and a past day restated never unseats a later day's - and a reading that re-ran the fold to
   check the fold would be one spelling checking itself. What a platter CAN carry that the fold
   cannot is a close body the closed vocabulary would have refused, a copy of a newer hub or a hand
   on a file, and that is what this reads;
6. **every number's replay tuple** — the attestations are exactly the standing runs the script asked
   for, asked as a SET because content addressing dedupes numbers and not standing: an identical
   what-if after a standing run reaches the same four coordinates and must not read as a second
   attestation. A standing ask with no row is a number nothing can replay; a row no standing ask
   names is a lane that mints nothing having minted;
7. **the diary equals the filings** — every settlement filed against a derived key names a row the
   diary carries, and every confirmation a clip the record holds (`spine.fill_key`, its instrument
   and execution reference off the `positions` fold). The diary is a COMPILE of the book, not a fold
   of the record, so its rows are handed in;
8. **no delete** — the positions dense from genesis, every blob the chain CITES still addressable,
   the store holding no fewer at the end of the walk than at its start, and neither the store nor
   the vocabulary carrying a verb for forgetting. Referential closure is the deletion a record can
   actually suffer: the store has no verb for it, so what takes a blob is a file system, and the
   citing frame is still on the platter naming an address nothing answers for. A LINE that left is
   met earlier and harder, when the home is opened. A citation lives in a sealed body, so a keyless
   copy is left with the other three arms;
9. **duplicates coalesce** — every idempotency tag on one position, read off the envelope, so a copy
   holding no key answers it;
10. **nothing outside its scope** — the node half of the writer's decision: every frame judged at a
    node below a book - the portfolio a fill, a restrike, a quote or a verdict names, a node's parent
    - and every node admin's capabilities declaration, re-run through `declarable` as the writer ran
    it. A frame judged at a node is held to where the record admits it: its portfolio in its own
    book's tree; a restrike at the deepest node holding what it moved, off the `positions` fold at
    the frame before it; a fill, once any node of its book is declared, at the book or a declared
    node; and a verdict at a node covering every trade carrying its ticket, so an approver naming its
    own node signs nothing beside it;
11. **the P&L is additive, and this record's** — handed `{window, days, portfolios}`, each `GET
    /book/pnl`'s own answer: the days tile the window mark to mark and sum to it, per position and
    in total, and the portfolios read over the window sum to the book, in every figure but the
    new-deal split. THE SPINE ADMITS NO TOLERANCE OF ITS OWN: one set of floats summed in two orders
    agrees only within an epsilon, and the one used is the tolerance policy's for the `pnl` class
    (`policy.compare`) - none declared, any difference is a departure. It is ABSOLUTE, so a book
    whose figures reach 1e10 declares one past their rounding, and the one document serves the
    replay promotion too. Every answer is held to THIS record: each end a marks run it attested -
    day, cut, values, job and result - and each row's quantities the `costs` fold's at the two ends
    and its premiums the consideration of the fills between them. WHAT NO RECORD PROVES is a figure
    only the engine computes - a mark, a payment the diary determines, a spot - so answers whose
    every such figure was scaled alike still sum: that is the boundary of what data can hold, and
    the evidence says so;
12. **cash reconciles** — every movement's subject resolves where its kind says: a fee to an
    instrument a fill booked by then, collateral and margin to an agreement declared by then - which
    the verb asks when it files and a copy can carry past it. The diary is handed in as its rows -
    key -> `{amount, currency}`, the amount null where it is not determined, or the keys alone - and
    the payment standing against a determined row carries that amount in that currency within the
    `pnl` epsilon, an undetermined row taking any, while one row paid under two references is named;
    every row the script says a settlement file instructed stands `settled` - by a payment, or by a
    bare `settled` that moved no money - exactly as the diary reads it;
13. **every close marked** — each day closed on the market the workflow designates for P&L is
    marked wherever a book held anything at its close, the `positions` fold there: a standing run of
    that book's marks job (`marks:<book>@<cut>`, `verbs.marks_of` reading the cut after its LAST
    `@`, and a cut past the run that attested it believed of nobody) as of the day, on the values of
    a close declared for it at or before the run. A day the book held nothing owes none. A close
    restated after its day's marks is a READING - marks run forward, so the day reads as it was
    marked - named in the evidence and never failed;
14. **the call is the formula** — every collateral call the script read is
    `projections.CSA.call` over the exposure and the spots (`fx`) the service answered -
    compiles this TRUSTS, as data - with the `cash` fold at the read's `lsn` held as of its `date`
    and the terms the `agreements` fold stood on there, both folded ONCE forward through the reads:
    the service's `held` and every figure `CSA.WORKED` names are compared as the numbers they
    are. A call nobody could work out carries no numbers and is not worked - one that carries them
    is named - and a script whose every call was unknown puts no question.

**WHAT A COPY CAN SAY.** A crypto-shredded home answers 1, 8, 9 and the envelope half of 2 — the
chain, the positions, the tags and the types are the envelope's — and names the other ten as NOT
ASSESSED with the reason, never as a pass. A follower that pulled FRAMES AND NO BLOBS is the other
posture and is answered the same way: the capabilities fold reads a document out of the store and
fails closed on one that is gone, so re-running the writer against it would call every frame after
the declaration a forgery - 2, 4 and 10 therefore ask for the blob first and tell that copy to
follow with `--blobs`, as do 11 to 14, which read the tolerance policy, the workflow, the marks'
jobs and the agreements' terms. The capability walk is taken once for 2 and 10 alike.
`DV_Spine oracle --home <replica> [--script <json>] [--against <other home>] [--diary <json>]
[--pnl <json>]` prints the report and exits 1 on an invariant that did not hold.

**THE GAME IS `gates/spine_game/`, AND IT IS A GATE.** `play.py` founds a bank in a clean folder,
writes its book, serves it on an EPHEMERAL PORT and stands N followers up against it; then the seats
work through three closes, and the script of what was asked - with the diary's rows and the P&L the
engine answered - is written beside the run for the oracle to hold the record against.

**THE BANK IS FOUNDED WITH THE VERBS A DEPLOYMENT ALREADY HAS** (`roles.found`), nothing loading a
bank from a file: the founder runs `DV_Spine init`, enrolls every seat, `grant`s the firm - the back
office over `*`, each desk's head `admin` at its node - `rewrap`s the class key to the two seats
holding a key to every body, `declare`s the tiers policy (per desk: the hub signs a ticket under
the desk's size cap, a second seat signs above it; `pnl` and `settlement_export` designated), the
firmness window, the fixings' sources and the P&L's tolerance, declares the three desks as nodes
of the one book, and names every seat; each head then `grant`s the document again with its own
trader, salesperson and approver seated at its node - a scoped declaration the writer admits
because it moves nothing beyond that node. The seats: product control (`mark`, `validate` over
`*`) marks the board, prints the fixing, closes, marks and reads the P&L; legal (`document`)
declares each client and its one agreement, a collateralised netting set stating no collateral
rows; settlements, confirmations and collateral (`settle`, `validate`) move money and state; audit
(`validate`) reads and files nothing; a hub seat distinct from all of them is `DV_SPINE_ACTOR`;
and a stranger holds nothing.

**THREE CLOSES.** Yesterday's book carried in - a legacy trade booked through the hub, there being
no import verb - closed, marked and called on. The day's trading: sales quotes a collar for one
client and the FX trader accepts it over the desk's cap, so it books PENDING; a small forward the
hub signs itself; a booking into a node the desk never declared, refused; a deposit under the rates
desk's cap, an FRA and a par swap above it; a metal forward restruck while it waits. The approvers
work their worklists - the rates approver rejecting the swap, which stands - confirmations confirm
every clip its worklist names, settlements files a fee, product control prints the swap's first
reset, closes and marks, and collateral settles the calls its worklist names. The settlement day:
the book rolled and the rand moved, settlements strikes the file its worklist says is due and files
every row with the money it moved, product control closes and marks, reads each day, the window
over both with the second day's explain, and each desk, and collateral posts what the marks now
say. EVERY SEAT THAT HAS A WORKLIST ACTS ON IT; the script still decides what the traders book and
at what size, which ticket the rates approver rejects, and which P&L windows are read.

A ROLE IS A CALLABLE over the table, which is the whole of the interface — the scripted players are
functions, and a host driving `DV_MCP` against the same hub plays the same days by handing one of
its own in; the LLM-driven mode is that substitution and nothing else. ONE ACT REACHES PAST THE
BINDING and says so: a fixing is printed through the hub's writer, no endpoint filing one. A gate
holds the players to that: every binding call a published tool and the raw transport never, every
CLI verb a shipped subcommand, the writer only for a print.

**SIXTEEN OBJECTIVES, AND THE ANSWERS COME IN THREE SHAPES.** A DENIAL is the writer refusing an
append and filing the refusal as a fact: settlements booking a trade, a trader settling its own, an
approver signing a ticket booked at another desk's node - the verdict filed where the ticket books,
read off the record - the FX trader booking into the rates desk's node, the FX head declaring a
trader beyond its node and then itself `admin` over `*` - one denial between the two - and a
stranger's what-if, refused at the QUEUE before a solve is paid for. A REFUSAL turns an act away
before any append and mints nothing: a deleted trade booked again a hundred times its size under its
own execution reference - a trade is taken once, and the trade as it was is taken back - and the
writer's reserved name on a request. And some attempts are answered by the record CARRYING what
happened: a trader approving its own ticket appends the approval and the trade still reads pending,
because four eyes asks WHO signed - under any display name, the side table outside the log being
one the record never reads, so the seat renamed twice signs the same fact again; the approver's
approval of the day's collar, sent again, lands on the LSN it already has while the adversary's
fill waits on a ticket of its own; a losing trade restruck with no notional stated lands PENDING on
a ticket of its own, the fold behind it unchanged; two acceptances raced leave one fill at one LSN.
A blob taken off a copy's disk is named at the citation nothing answers for, a forged checkpoint
CHAINS on a copy and the verification refuses it, and a tampered line parts company with the chain
where it was altered - the hub untouched each time.

**SIX FAULTS, AND THE DAYS CARRY ON.** The HUB is a real process and is killed with the operating
system: served as `DV_Service` on the day's book while product control's marks stream into it,
killed between two appends and served again on the tail it left - the half line a kill mid-write
leaves put on its platter as data where the kill fell between lines - so the torn-tail rule is
observed on its own open rather than asserted: every acknowledged append stands, the chain
re-derives whole and the next act lands on it, nothing else writing while its process holds the
home. A replica is partitioned by not being
told and resumes by asking once. A print carrying a truth-time older than the one standing does not
win by arriving last, and the print it lost to keeps it on the row. A late fixing after the close is
answered by a SECOND close naming the position the first stood at, and the day marked again. And
one act said twice - a mark, a settlement under its reference - is one fact at one LSN, the money
moved once, because the semantic tuple carries no clock of the writer's own.

**Nothing is monkeypatched.** Real homes under the caller's own directory, a real service over a
real socket, real replicas pulling real frames, the hub's own process killed, and every other fault
injected as data on a disk. The hub is the single writer and every seat reaches it the way a model
would — an adversary with a second writer is not an objective, it is a divergence
`/book/reconcile` names. A play - the founding and three closes, 87 events - measured 37 to 774
seconds of wall time across this increment's runs on one box, fsync-bound and sharing the box with
whatever else runs there; a seat's read stays under 120 ms and the oracle under half a second over
three copies.

**THE BACK OFFICE HAS A VERB.** `status_transition` is what a settlement and a confirmation say,
demanding `settle`, and `verbs.transition` files it: `subject` is an ADDRESS — the derived cashflow
key a diary row carries, a clip's own key, or the instrument a trade books under — where the
vocabulary takes any name, because a state filed against something nobody can resolve is a state
nobody can read back. WHETHER THE BOOK ANNOUNCES A ROW under that key is a FOLD's question: the
record holds what it was told, so a settlement against a row since paid away lands rather than
refusing, and the oracle's seventh invariant is what reads the two against each other.
`Context.transition`, `POST /book/transition` and the `file_status` tool are the mouths, and
`close_check` stops waiting on a payment the moment one lands.

**A TRADE'S TERMS ARE A CITATION.** `book` and `amend` fsync the canonical instrument into the store
and name its address, so `BLOB_FIELDS` lists `fill.instrument` and `amendment`'s two: referential
closure resolves them and a follower pulling `--blobs` asks for them, which is the difference
between a copy of the record and a copy with every position in it and the terms behind none.

**ONE SCOPE FOR AN APPROVAL.** A ticket is one trade THIS book books, so a seat's verdict is filed
under the job's own book, or the node the ticket books into where the verdict names one; an
automatic tier's is the hub's own.

`tests/test_spine_oracle.py` makes each of the fourteen fail on a doctored home and names where;
`tests/test_spine_game.py` plays the bank twice - the blue days holding every invariant on three
copies, every red objective answered and readable, every fault converged, the keyless and blobless
copies naming what they cannot assess, the players held to the binding and the shipped verbs - every
killing mutation their docstrings name red.

## Increment 8 — the paper, and where a position sits

**A POSITION IS KEYED WHERE IT SITS.** A `fill` carries three more fields, each optional so a body
filed before them validates exactly as it did: the `agreement` and the `portfolio` it sits under,
and the `price` it was done at. Its `quantity` is the signed position CHANGE in units of the
instrument - 1 the instrument as written, -1 closing it, -0.5 unwinding half - and the position is
a fold, never written. `positions` is version 2 and keys by instrument × agreement × portfolio: one
instrument under two agreements is two positions, credit exposure being per agreement, and in two
portfolios it is two, a portfolio being where risk is owned. A fill filed before the key existed
sits under its netting set and its book, and an amendment carries every row forward under its own
key. An accepted quote books ONE UNIT of its mirror as written, whose terms carry the desk's side
and the notional struck.

**THE PAPER IS DECLARED.** A fifth part of the closed vocabulary, said by the seat that keeps the
legal documents rather than by the desk: `entity_declared` - an id, a name, and an optional parent
that groups it - and `agreement_declared` - an id, the entity, a kind as its declarer labels it,
and the terms: the netting set its positions compile into, cited by address so a replica pulling
blobs holds them. The act is a seventh capability verb, `document`; who holds it is the
deployment's grant, so the record names no department. The `entities` and `agreements` folds read
them, a restatement standing by the as-of key and an agreement naming the declaration it stood
over. `POST /book/entities` and `POST /book/agreements` are the service's mouths, the terms judged
by the engine before anything appends - not a netting set, carrying positions, stating a balance or
a holding (settlement state, never paper), or refused by their own declarations, each by name - with
a `GET` beside each, and the binding's `declare_legal_entity`, `declare_agreement` and
`describe_agreements`. A booking naming an `agreement` takes its counterparty from the agreement's
entity and must sit under the file's set of that name, the set being the agreement's
materialisation, which the first booking under a declared agreement brings; its `portfolio` is the
book's own name where none is stated.

**THE BOOK IS READ WHERE IT SITS.** A portfolio is a path whose top node is the book the fill is
filed under, where its permissions are granted, so a stated one outside it or with an empty segment
refuses by name at the booking - and once any node of the book's tree is declared, one that is
neither the book nor a declared node does too. `GET /book/positions` answers every position
standing at the head - a closed one stands no more, and one whose deal has expired at the book's
date stands until the day
it settles and rolls off on it, a trade settling at T+2 still held at T+1 and a structure held whole
expiring once every leg has - beside the nodes of the
book file carrying its instrument under its agreement's set, and the binding reads it as
`book_positions`. It is how the web UI groups the
book: the Portfolio screen and the blotter show the file's own nesting, the portfolio tree - a
folder per segment of the path - or the client tree - an entity under the parent declared for it,
holding its agreements and then the entities under it, every declared one standing whether or not
anything is booked there - and a position picked shows the deal the file holds for it. The deal's
`Tags` field and the deals block's `Tag_Titles` retire with it: they were where a desk wrote a
portfolio, a desk and a trader, and the fill carries all three.

**THE BOOK PRICES AT ITS NET.** A position is units of the instrument, so the compile that writes
the record's prints into the plan writes its positions too: every node of the file the record holds
a position in is written at its NET - the fills summed over every portfolio under its agreement's
set - through `spine.scaled`, the engine seeing amounts and never a quantity, which multiplies every
field the deal's type declares `sized` (a scalar, a table column, a leg's) and takes a negative net
as the mirror, every `side` the type declares flipped where it has one and the sizes signed where it
has none. An amortisation step is sized as a MAGNITUDE - the engine takes it off its principal's
magnitude whatever the principal's sign - so it scales and is never signed. The legs of an option on
its children - a swaption's underlying - are its terms, sized and never flipped; every other deal's
legs - a structure's, a cap's caplets - are themselves held. A net of one is the deal as written, a
net of nothing is no deal, and the same terms' second node under one set is the first one's clip,
ignored rather than priced twice - ignored and not removed, so every path a verb resolved against
the file still names its node. The desk's reads price the compiled book - the consolidated risk, the
XVA recalc, the what-if, a named calculation and a quote's risk-impact step - beside `/execute`,
`/prepare` and the diary, which always did; a quote's pins and its plan stay the file's, which is
what the acceptance holds them against. Every type a position can be held in declares its sizes
beside its trial - a structure and a mark-to-market cross-currency swap through their legs - and
held at half a unit each marks exactly half and held short exactly minus, to the bit; a block
stating none of the amounts its type declares refuses by name rather than pricing one unit. A
STRUCTURE BOOKED WHOLE IS ONE INSTRUMENT, its legs inside its terms: a leg booked into it refuses by
name, a leg amended amends the structure and carries its position onto the terms the edit leaves,
and a position the record holds in a leg of a structure it also holds refuses the read rather than
pricing the leg as the structure's.

**A SEED IS FILED WHERE THE DEPLOYMENT SAYS.** `DV_Spine seed --at <lsn|day> [--out <folder>]` mints
every fold but the strip's at an official close - the last one true on a day, where a day is named -
into a folder: one every seat reads after an end of day, or a seat's own for a day it wants to
stand at. A reader takes the newest seed at or behind its head from `DV_SPINE_SEEDS` (the home's
`seeds/` where unset), verified as ever and of its own projector's version only, and folds the
day's events on top; the seam's reads and a booking's `advancing` pair start there. When a seed is
minted and how long a folder keeps one are operating policy rather than machinery.

`tests/test_spine_paper.py` holds the record's half - the keyed fold over five clips and an
amendment, the paper's folds with a backdated restatement and a lost citation, the seed folder, the
verb - `tests/test_spine_engine.py` the service's, `web/scripts/positions_check.mjs` the two trees
and the grouped blotter, and `tests/test_position_scaling.py` a trial per family of types
(`tests/trial_<family>.py`), the census holding every type to its declaration and its trial, and
every declared amount to a trial that states it.

## Increment 9 — the money that moved, and what the book made

**A SETTLEMENT THAT MOVED MONEY SAYS HOW MUCH.** `status_transition` gains four optional fields,
so a body filed before them validates as it did: the signed `amount` from the bank's side -
received positive, paid or posted negative - the `asset` it is an amount of (a currency, or a
security held as collateral), its `kind`, and the settlement system's own `reference`. Money moves
under `settled` alone, on a value date the settlement states. The kind says what moved: a
`payment` settles the diary row its subject keys exactly as a bare `settled` does, so the close
check and the settlement file read it unchanged; a `fee` is money a trade cost that no row
announced, filed on its instrument; and `collateral` and `margin` move a balance held under an
agreement the record declares, since collateral arrives through the settlement interface like any
settlement and the balance per agreement is a fold of those rather than a statement. Only a payment
moves a state: the other three put nothing in the lifecycle or the blotter.

**THE REFERENCE IS THE MOVEMENT.** Retried under one reference it is one fact; two identical
movements under two references are two, where the bare tuple would have coalesced them; and an
amount filed again under its reference restates the one it corrects by the as-of key, naming the
filing it stood over. The twelfth projector, `cash`, keeps one standing movement per reference, and
a balance - the collateral under an agreement, what a row settled for, the fees on an instrument -
is a sum over its rows and stored nowhere. `POST /book/transition` takes the four fields and a
`value_date`, `GET /book/cash` answers the movements and their balances per kind, subject and asset
as of a `date` where one is named, and the binding's `file_status` and `book_cash` are the same
two. What a balance of collateral is asked for, and how it reaches a run, is
[collateral's](#collateral-the-call-on-a-close-and-the-balance-in-the-plan).
`tests/test_spine_verbs.py` holds the record's half and `tests/test_spine_engine.py` the
service's, six mutations red.

**A POSITION COSTS ITS AVERAGE.** The thirteenth projector, `costs`, keys every position as
`positions` does and carries its `basis` - the open quantity times its average price, signed with
the position - and what its reductions `realised`: a fill adding to a position adds its quantity
times its price, one reducing it realises the difference between the price and the average on the
part it closes, and the rest opens the other way at its own price. The price is the fill's
CONSIDERATION per unit as written, signed from the desk's side; one agreed in another currency
than the book reports in is filed with that `currency` and the `rate` the booking crossed it at,
the booking's own board, so nobody converts a premium by hand and the basis reads it in the book's
currency. An accepted quote files one too: nothing where its recipe solved a coordinate, the charge
riding inside the terms, and minus the client's premium where it solved nothing. A fill booked with
no price leaves the basis UNKNOWN while that lot is open, never a zero, and counts under
`unpriced`; a reduction closing at no price or against an average nobody priced is COUNTED under
`unpriced_reductions` rather than realised as nothing, so `realised` is what every reduction it
could price realised and a later priced trade is still known. An instrument hash is one contract,
so average cost over identical terms is specific identification per contract.

**A CLOSE IS MARKED, AND THE MARKS ARE ITS OWN NUMBERS.** A close names the calendar `date` it is
declared FOR. `pnl` joins `settlement_export` as a designated process, and `POST /book/marks`
values ONE UNIT of every instrument the book holds, or traded since its last marks, each as written
and filed under its own address, on the values of the close standing under that market - the close
declared for the book's own day, one for another day refused - as a standing run, attested like
any other. A position's value is then its quantity times its unit mark
and every grouping of the book is a sum; and the job the run attests is the book's market and
instruments as they stood, so a past close replays from the record rather than from a book file
that has moved since. When the marks are taken is the deployment's end of day. They close the
business day where they READ the book - the job names that position of the record, so a ticket
booked while the run waits on the worker is the next day's - and a fill belongs to the business day
of the first marks at or after it, an end-of-day cut, never the clock it was recorded by. The latest
marks of a day stand, so the last day marked again is restated, and marks run forward: a day behind
the last one marked is refused, so a close found wrong once a later day is marked stays as struck
and its error reverses in the next day's P&L.

**THE P&L IS THE MARKS' MOVE PLUS THE CASH, AND A MONTH IS THE SUM OF ITS DAYS.** `GET /book/pnl`
reads it between the marks of two days, or from the last marks to the book as it stands now - the
intraday flash, recorded nowhere, the day being lived rated as the book as it stands prices it -
per position and in total, in the reporting currency:
`value_end - value_start + premiums + payments + fees`. THE CLOSE IS THE END OF THE DAY: the
engine values a payment into the marks of the day it falls due, so a position's value is its
quantity times its unit mark less what the unit PAID that day (`paid_start`, `paid_end`) - every
payment the diary determines, and every one it does not that a settlement filed by the marks moved,
per unit held when it fell due; one nothing has settled by then is still the book's and stays in
its value until a filing moves it, and a window it leaves without the money is named. The premiums
are the fills' own on their trade date; a payment is the diary's amount where it determines one,
times what the position held at the END of the day it fell due - a trade's price that day carrying
what it pays, so a seller then is paid nothing and a buyer all of it - a settlement that moved
another amount being a BREAK listed beside the P&L and never an unknown, else the position's share
of what the settlements moved under the row's key - one settled with no amount having moved
nothing, and one over positions netting to nothing NAMED, a settlement naming no agreement to share
it by; a payment
and a fee are converted at the official close STANDING ON THEIR OWN DAY, after which the money is a
cash balance whose currency is its holder's and never the trade's. A SETTLEMENT OR A FEE COUNTS IN
THE WINDOW IT WAS FILED IN - the change in the `cash` fold between the two marks, a restatement
counting its correction - so a late filing lands in the day it was filed, one against a payment the
diary has since dropped found in the diaries of the marks before it, a business week of them, and a
day already struck never moves. A MOVEMENT IS READ AS THE RECORD STOOD AT THE END OF THE BUSINESS
DAY IT COUNTS IN, and a payment the diary determines as it stood at the end of the one it falls due
in: the close it is
rated at, so one restated afterwards moves nothing already counted, and whose it is - a payment by
what each position held when it fell due, a fee by what each traded on its own day, that being its
value date or the day it was filed where it is dated after that, else by what each held through
the day, long or short alike, else by what each traded on the record up to it, else by what each
held when it was filed - terms an amendment struck being traded by no fill - a fee nobody traded
or held NAMED. Every figure
but the new-deal split is therefore additive: two days sum to the window over both. `existing` is
what the positions held at the start moved and `trading` what the window's fills earned against
its end - null where a position traded to nothing has no mark at the end, which leaves its P&L
known. The realised half is the costs' plus the cash, at AVERAGE COST, which the answer names; a
position whose last day falls in the window CLOSES there at its settlement value - its open basis
released to realised, its payoff arriving as the payment it is - and after it stands only through
the money it still moves and realises nothing else, a payoff settling at T+2 counted in the window
it settles in, and so does one holding nothing, a portfolio unwound before a late fee still taking
its share. A STRUCTURE HELD WHOLE is paid through its legs, keyed as the book's diary keys each,
closes on its last leg's day and rolls off the positions once every leg has; an AMENDMENT restrikes
the position onto the terms it became from the START of the business day it was filed in, so what
falls due that day is paid on those terms, and a leg a restruck structure leaves as it was is paid
once, to whichever of the two held it when it fell due. Anything nobody can know - a fill with no
price, a reduction against a lot booked with none, a payment neither the diary nor a settlement
states, a settlement against a payment nothing here announces or that no position held when it fell
due, a missing mark, a day no close stands on - is named under `unknown`, a total over it is null,
and `complete` says there was one. A `portfolio`, an `agreement` or a `client` with every entity
grouped under it narrows it, and the binding's `mark_book` and `book_pnl` are the same two verbs.

**THE EXPLAIN SAYS WHY THE HELD POSITIONS MOVED, and never assembles the P&L out of its pieces.**
`explain` on `GET /book/pnl` takes what the positions a window started with made apart into three:
the CARRY of the start's book rolled to the end's day at its own quotes - the day moved, every
curve re-authored on it and, where the book declares a bootstrapper, the market re-bootstrapped as
`POST /book/date` rolls a book, a position whose last day fell in the window closing at nothing;
the MARKET, every risk factor's move between the two closes times the start's own sensitivity to
it, per quote where the factors are built from quotes and per factor for the rest; and the
RESIDUAL left over. It is three valuations -
the positions on the start's close with first-order sensitivities, the same positions on the end's
close for the levels, and the start rolled - run when asked, recorded nowhere and cached on what
they read, the carry taking out what a day paid as the values do. Reserves are not carried, so none
is explained. `GET /book/marks` answers the days the book was marked on, and the web UI's P&L screen
reads the two: a window between two marked days or to now, narrowed by portfolio, agreement or
client, the rows and totals with what nobody can know named, and the explain with its residual's
share - a window ending now offered again once the book moves rather than valued on every write.
`tests/test_spine_projections.py` and `tests/test_spine_verbs.py` hold the cost arithmetic and the
consideration, and `tests/test_pnl.py` holds the P&L on real homes: three marked closes - every
unit mark against the instrument valued alone, every row against its own identity, the portfolios
summing to the book, the two days to the window, a struck day read again unmoved after late
filings, the explain's residual equal to the carry's own move with the rand - and a test for each
shape a day meets: a coupon on a marked day, fixed and floating, settled on time, late and three
windows late, a strip bought on its coupon day, the day being lived and one after the last marks
before it; a structure held whole, one restruck, and one whose leg settles late; an amendment on
the morning its trade pays, beside fees on the new terms and dated forward and closes restated;
fees over long and short holdings, a day trade and a position traded to nothing; a payoff over a
flat net, money nobody held and a fee on nothing held; a forward paid the legs its settlement date
declares, held alone, as a structure's leg and closed out before it settles; a dead position closed
out and a lot nobody priced; and a ticket booked while the marks wait, the last day marked again
and a day behind it - on a book whose name carries the marks' own '@' - every killing mutation
their docstrings name red. `tests/test_diary.py` holds every payment a date declares against the
cash the run itself books, a forward's and an FX swap's legs among them, the days a table
declares settled, and the census of what every trial type observes. `web/scripts/pnl_check.mjs`
holds the screen's own arithmetic.

## Seats, nodes and the hub's own voice

**SEVEN VERBS, AND THE HUB IS NONE OF THE SEATS.** A document grants `validate`, `book`, `approve`,
`mark`, `document`, `settle` and `admin`. `settle` is what the back office does - a
`status_transition`: a payment settled, a trade confirmed, money moved under paper - so a
settlements seat books nothing and a trader settles nothing. What the hub itself says is no seat's:
a `run_completed` - numbers it ran - and a `snapshot_registered` are the WRITER's, filed in its own
voice under the reserved actor every `capability_denied` carries, through `SpineLog.own`, the one
path that stamps that actor and never gated. The public append refuses those types under any other
actor and the reserved actor on any type, the seam refuses the name whether a request or
`DV_SPINE_ACTOR` says it, and `init` refuses it before a byte is minted. An automatic tier's
approval is that voice too: the workflow an admin declared decides it, and the oracle holds one
only where a tiers policy stood and a fill or restrike carrying its ticket came after it. A standing
run is
admitted under the requester's `mark` over the book its job values and attested by the hub. Under a
capabilities document a request naming no seat is refused by name; `spine.hub()`, `DV_SPINE_ACTOR`,
is the deployment's own seat and the poll paths' alone. A document on the record is read in the
grammar it was declared under - a grant of a retired verb dropped, a tier naming a seat read as the
automatic one it was - and a new declaration of either is refused by name.

**A GRANT IS AT A NODE.** A node is a book or a path of named segments under it, and a grant at one
reaches every path below it and nothing beside it: `BANK/FX` covers `BANK/FX/Options` and neither
`BANK/FXO` nor `BANK` (`capability.under`), and a document naming a node with an empty segment is
refused where it is declared. An append is judged at `capability.scope_of` - the firm, which only
`*` reaches, for the firm's own facts whatever book the envelope names (a policy, a market or a
close, a fixing, a run or a snapshot, the paper, a checkpoint, retention and rehash, custody); the
`portfolio` a fill, an amendment, a quote or a verdict names; a declared node's parent; else the
envelope's book - one function the writer asks and the oracle asks again, the denial filing the
scope that was wanted. A `portfolio` outside the envelope's book is refused by name before anyone
is asked, a node sitting in its own book's tree. The queue asks `capability.holds_any`: a seat that
may `validate` one node of a book is admitted to price the book. An amendment's `portfolio` is the
deepest node holding every position in the terms it restrikes (`capability.holders`), read by
`spine.amend` off the positions fold advanced under the book lock and never stated by a caller;
the oracle re-derives it at the frame before and names one judged narrower than what it moved.

**THE TREE IS DECLARED.** `portfolio_declared {path}` puts a node on the record, judged at its
parent: `admin` at `BANK/FX` declares `BANK/FX/Options`, and a book is the firm's to declare. The
fourteenth projector, `portfolios`, folds them by path. Declaring is activation: a book with no
node declared books wherever its paths say, and once one stands a fill books into the book or a
node declared under it - read off the fold advanced on the write's own handle, and not read at all
for a booking at the book itself. `DV_Spine portfolio <path> --actor <seat>` declares one;
`verbs.declare_portfolio`, `spine.declare_portfolio` and `spine.portfolios` are the library's.

**A NODE ADMIN DECLARES ITS OWN SUBTREE.** The capabilities document is still one document, so a
seat holding `admin` at nodes replaces it where everything it moves sits under them:
`capability.declarable(old, new, subject)` answers yes for `admin` over `*`, and otherwise only
where the `read` rows stand as they were - a read row being a key to every body - and every grant
row outside the seat's nodes stands exactly as it was, a `*` row sitting under no node. The writer
parses the declaration on the arm the grants refuse and admits it on that answer, the denial naming
the rows moved outside; the oracle runs the same function over the same two documents. A node
admin's declaration leaves a break-glass recovery standing: the recovered seat stays admin until
it, or an admin over `*`, declares.

**THE TICKET IS THE TRADE.** A fill carries the `ticket` an approval of it signs -
`spine.ticket(plan_hash, trade)`, the content hash of the plan the book has once it lands and the
trade's own fields as its event files them - so an act the writer is handed again is the same fact,
coalescing onto the LSN it already has. AN AMENDMENT IS A TICKET TOO, over the plan the restrike
leaves and its own fields. AND A TRADE IS TAKEN ONCE: `/book/deals` answers a booking whose fill
already stands under its key - the instrument and the execution reference - with the fields it
states as `{written: false, booked: {lsn, deal_path}}` with nothing spliced or filed, the file
taking the trade back and nothing filed where it had lost it, a client that never saw its answer
sending the same execution again; one stating other fields under that reference is refused by name,
a second execution wanting its own; and a restrike sent again after it landed is answered the same
way (`spine._booked`, `spine._restruck`). A fill's clip keeps the LSN it was filed at, so the key
finds the fields it filed. The `positions` fold carries each clip under the ticket that books it -
its fill's, or the restrike's that restruck it - keeping the instrument and execution reference it
was filled under (`tickets`, version 3), so `spine.standing` reads every position off the positions
and decisions folds, both advanced and no fill reopened: each ticket `approved`, `rejected` or
`pending` off the verdict standing over it (`spine.status_of`), or `unticketed` where it carries
none or no tiers policy stood when it landed - the decisions fold keeping where the first stood
(version 2) - the position the first of `spine.STATUSES` among them and `pending` the quantity
awaiting a second seat. A ticket the hub did not sign is under four eyes, so its booker's own
verdicts are not read (`tiers.standing_verdict`): its approval clears nothing and withdraws no other
seat's rejection, and its rejection withdraws no other seat's approval. Under an automatic tier they
are read, so a booker's approval after another seat's rejection stands - that tier declared no
second pair of eyes. `spine.fill_key(instrument, execution_reference)` is a clip's own key, kept
through every restrike, which a `confirmed` transition is filed against. A tier is automatic unless
it declares `four_eyes`, may cover a `scope` read before its checks - `*` covering every ticket -
and a `seat` is refused by name.

**EVERY BOOKING IS ROUTED IN ONE ACT OF THE WRITER** (`spine.routed`), a fill and a restrike alike:
the body built, which asserts its shape; the tree read; and `SpineLog.admits` asking the vocabulary
and the seat's scope with nothing written - so EVERY REFUSAL A FILL OR A RESTRIKE CAN MEET COMES
BEFORE THE HUB SIGNS - then the ticket, then the route on the writer's own handle, then
an automatic tier's approval in the hub's voice, then the event. The route reads the notional the
trade STATES (`notional`, `notional_currency`, the whole trade's) crossed at the book's own spots as
a quote's is (`ticket_terms`), the tenor its type's declarations name (`schema.tenor_fields`: the
latest day it declares it settles, expires or matures on - a forward's settlement, a strip's last
row - none failing a tenor cap by name and nothing compiled under the lock) and the book's values.
Under four eyes the event LANDS carrying its ticket and reads pending, a trade that prices, settles
and exports all the same, the status being workflow and not economics; a restruck position reads
the restrike's ticket; and a ticket no tier admits writes nothing. A crash between the approval and
the event is the one way an approval can stand alone, and the oracle naming it is right until the
act is retried: the approval coalesces and the event lands after it.
`POST /book/approve` and `/book/reject` take a `ticket`, or a `quote_id` its acceptance's pending
file resolves, and file the verdict at the deepest node holding every position the ticket stands in
- read off the record and never off the caller, so an approver at one node does not sign another's
ticket by naming its own; a quote's ticket no tier admitted is ruled where the quote was filed.
`GET /book/positions` answers each position's `tickets`, `status` and `pending`, and a REJECTED
TRADE STANDS: it happened, and it is somebody's to close.

**THE SET IS THE AGREEMENT'S MATERIALISATION.** A booking naming a declared agreement whose netting
set the book file does not carry brings it - the declared terms spliced at the root, the booking's
parent where it names none - with the verdict's baseline taken before, so what the set itself wants
the book lacks, its counterparty's survival curve among it, is newly said and refuses by name. A
quote named for such an agreement splices the same set on its own copy, so its pins and its plan
are the book as the acceptance leaves it, and the acceptance splices it again: one materialisation
for the three, where a refusal would have made a direct booking the only way to open a new client.
`/book/reconcile` gains `terms_mismatch` - every set of the file materialising a declared agreement
whose paper is not the declared terms, with the fields that moved, its `Children`, its balance and
every holding's amount left out as positions and settlement state.

**READS BY NODE.** A seat that holds any grant at nodes (`spine.sight`, `capability.nodes`) reads
the rows under them through one `spine.visible` - a seat that may act at a node sees the rows it
acts on: the positions, the P&L with every total summed over those rows alone, the diary and the
reconcile rows by the positions their instruments sit in - their amounts the book's own compile at
the instrument's net across every node, presentation and never a per-node compile - and the money
and the strip by the book their envelope names, so a node seat reads neither. A grant over `*`, a
home with no document, and a read no seat signs on a box checking no token read the whole book - the
last is the deployment's own view, since a filter a self-declared name could lift protects nothing
it withholds. A READ RUNS UNDER NO REQUESTER: the diary it compiles is the hub's own cache, admitted
under `DV_SPINE_ACTOR` with the request's seat set aside (`spine.deployed`), so no read files a
denial. Risk, XVA, `/book` and `/results` stay hub-wide. THIS IS PRESENTATION while classification
is dormant: one class key opens every body, so a seat holding a `read` row reads every frame whole
off `/spine/frames`, and the filter stops being presentation the day a second class exists.

**THE WORKLIST.** `GET /book/worklist` is six lists of what waits on the seat asking, read off what
stands, each what one verb acts on where the seat holds it, each row `{kind, what, key, lsn,
since}` with `key` what the clearing fact is filed against: tickets pending a second seat, one row
each, for `approve`; payments due by the book's day nobody settled - the close check's own verdict,
a fee settling nothing - clips no `confirmed` status filed since their own ticket stands
against, at the key they were filled under (`spine.fill_key`) - a restrike making a re-confirmation
due, one stating the day it was matched - and the collateral calls the marks of the book's day
make, keyed by agreement, every movement filed counted and one nobody can work out listed with its
`unknown`, for `settle`; closes on the market designated for P&L on a DAY after the last one
marked - marks run forward, so no earlier day is owed - for `mark` over the book; and rejected
trades still standing in a position, for `book`. `counts` says how many each list
holds and each answers the newest `WORKLIST_ROWS` (the strip's 200). Nothing is filed for it, so a
row leaves the moment its fact lands; the binding's `worklist` asks it, and the web UI's banner
beside the reconcile banner asks it where the RECORD's head moved (`wantsWorklist`) - never on the
beat, and never on the file alone, a tick being telemetry.

**THE TREE HAS MOUTHS.** `GET` and `POST /book/portfolios` read and declare the tree - the binding's
`describe_portfolios` and `declare_portfolio` - and the web UI's portfolio tree is seeded from it, a
declared node a folder whether or not anything is booked under it.

**THE SEAT FROM A TOKEN.** Naming a key set - `DV_SPINE_JWKS`, a JWKS file, with `DV_SPINE_ISSUER`
and `DV_SPINE_AUDIENCE` - makes every request but the doorbell carry an ID token that set verifies,
locally and nothing fetched; none or a bad one is a 401 in the verifier's words. One ASYNC global
dependency sets the subject in the request's own context (`spine.REQUESTER`) - a sync one sets it in
a threadpool copy the endpoint never sees - and `spine.actor` answers it, a request naming another
seat refused and none ever taken for the hub; the metronome's thread and a read's own compile run
under `DV_SPINE_ACTOR`. The key set is read once per version of its file, and one that does not
read refuses `DV_Service` at startup and a request by name - never a 500. Unset, nothing is checked
and a request's seat is what it says. The web UI sends a stored sign-in token, else the seat its
settings control names as `actor` on the writes a seat signs and on every read; the binding and a
follower send `DV_SPINE_TOKEN`.

`tests/test_spine_capability.py` holds the voice, the node, scoped admin, the portfolio outside its
book, the recovery, the stored document and the `portfolio` verb; `test_spine_verbs.py` the settle
verb and a verdict judged where it books; `test_spine_tiers.py` the scope and the booker's verdict;
`test_spine_oracle.py` the node seat, the forged voice, the hub's approval, the restrike and the
firm's facts; `test_spine_engine.py` the queue's `mark`, the unnamed request, the reserved name, the
automatic tier, a home in the earlier grammar, the tree, the restrike and the status; and
`tests/test_pnl.py` the marks admitted over their book. `test_spine_engine.py` holds the service's
half too - a booking routed as a ticket, a restrike routed as one, a trade sent again answered where
it stands and refused under other fields, a booking, a restrike and an acceptance retried landing
where they stand, every refusal before the hub signs, a forward dated by its settlement, the
acceptance pending then signed, a verdict filed where its ticket books, the agreement's set brought
by its first booking and its first quote, reads by node, a read under no requester and the seat from
a token - `tests/test_pnl.py` the worklist by verb and by count, a structure's legs at its node and
a node's P&L, `test_spine_oracle.py` a ticket filed twice and a restrike nobody routed,
`test_schema_emission.py` the tenor census, `test_mcp.py` the walk through the binding and the
lookup by plain name, and `web/scripts/spine_check.mjs` and `positions_check.mjs` the banner's ask
and the seeded tree - every killing mutation their docstrings name red.

## Collateral — the call on a close, and the balance in the plan

**THE CALL IS A READING, AND ONE FORMULA WITH THE ENGINE'S.** `projections.CSA` is a credit support
annex's arithmetic at one date, pure and stdlib, so the service runs it and the oracle can. `of`
reads an agreement's declared terms as the engine reads them - every `CreditSupportList` at its
first value, the balance currency the agreement's where unstated, and each cash row of
`Collateral_Assets` at its `Haircut_Posted`, the one haircut the engine reads, whichever side holds
the asset; `required` is the engine's `At` term for term - the independent amount, plus the excess
over the received threshold above it, plus the excess below the posted threshold below it, the
three crossed from the agreement currency; and `call` is that less the collateral held after
haircuts, moving only where it clears the minimum transfer on its side STRICTLY, as
`scan_collateral_balance` transfers. Nothing is rounded, a netting set declaring no rounding, and
a positive independent amount is support the bank receives. COLLATERAL AND MARGIN ARE TWO BALANCES: `held` groups the `cash` fold's standing
movements once per read into balances per agreement, each kind per asset, as of a value date, and
the call reads the collateral alone.

**THE EXPOSURE IS THE ONE THE NETTING SET RECURSES ON.** `GET /book/collateral?date=` answers every
agreement whose terms collateralise on the close marked for the day - the book's own where none is
named, refused by name where the day has no marks - in the agreement's currency: `{agreement,
entity, currency, exposure, fx, held, margin, required, balance, call, direction,
minimum_transfer, unknown}`, `fx` the close's spots in that currency, so with the exposure a call
replays off the record, which the oracle's fourteenth does. The exposure is read off the P&L -
`spine.PnL.pnl` over the close alone, sharing
`valued_between` with the P&L's own read - as the set's recursion reads it under the marks job's
`Exclude_Paid_Today`: where the set holds the day's payments, the engine's default, the P&L's value
of each position with what the unit paid that day added back, its quantity times its unit mark;
where it excludes them, the P&L's value, the payment being cash. A position whose last day is the
close is held through it, as the set holds a deal until its cash moves; a pending trade prices and
settles like any other and counts; and a mark nobody has, or a spot the close lacks for an asset
held, is named under `unknown` with the call null over it, never a zero. It is crossed into the
agreement currency at the close's own spots, and so is every asset held. The transfer decision and
the amount agree with the engine's recursion on every path of the gates, to rounding at the
minimum-transfer edge.

**TWO READINGS OF THE BALANCE.** The read holds the balance AS OF THE DAY - the movements the record
holds at its head, `lsn`, value-dated on or before it - so `{date, marks, lsn}` replays it. The
worklist's `calls`, what the marks of the book's day call for or post, are the `settle` seat's and
count every movement FILED against the agreement whatever its value date, so a call settled
value-dated tomorrow leaves the list the moment it lands; a call nobody can work out is listed with
its `unknown`. `call` is what a settlement moves - received positive - and `direction` `call` where
the bank receives and `post` where it posts; the back office settles it through `POST
/book/transition` as `collateral` against the agreement, a movement like any other. A seat reads
the agreements a position it holds any grant at sits under - the agreement's whole exposure, a call
being per agreement - and a seat acting everywhere every one, one nothing is held under included.
The P&L is read only where a seat can see a call, so a seat holding `settle` nowhere pays nothing
for the worklist's. The binding's `collateral_calls` asks the read, and the banner counts the list.

**THE BALANCE IS PLAN WHERE A RUN READS IT.** `compiled_job` writes every collateralised netting set
the record holds collateral under its `Opening_Balance` where the job is compiled for a run that
reads one - `runs`, the calculation it runs as: the collateral held by the job's base date, after
haircuts, in the set's `Balance_Currency`, every other asset crossed at the job's own spots. The
engine MULTIPLIES it by that currency's spot, divides by what a unit of the set's collateral is
worth after haircuts at the base date, and opens its recursion there. A base valuation runs no recursion, so it
is compiled with no balance and a settlement moves neither its plan nor what it caches -
`/book/risk`'s etag among them. The margin is never written, nor a movement dated after the base
date, and a set the record moved nothing under is left as the file states it. A HELD ASSET THE
BOOK CANNOT CROSS REFUSES THE COMPILE by name in every lane that values the book - "a balance in an
asset the book carries no spot for is a market the book lacks; install the spot" - a plan,
`/book/xva`, `/book/price` as either calculation, `/book/risk` and a quote alike, nothing valued on
a zero, as a booking is refused for market data the book lacks; a diary, which values nothing,
compiles past it. The `cash` fold is advanced rather than re-walked, and folded only where such a
set exists. `/book/reconcile` leaves a balance out of the paper, so a written one never reads as
drift.

**WHAT IS NOT CARRIED.** Interest on a balance; an eligibility schedule beyond the haircuts a set
declares, `Haircut_Received` being read by neither side; a security held as collateral, which a
close's spots do not value and which refuses as any asset without a spot does; the holdings of a
set's `Collateral_Assets` - a declared row states no amount, and a sole cash row stating none is
one unit of it, the balance being held in units of its asset, so an agreement declaring one
eligible cash row prices collateralised, while a row among several states its weight or is refused
by name; and the hedge simulation's opening state, which reads the file.

`tests/test_collateral.py` holds the arithmetic - the dials against `utils.CreditSupportList`,
`At` in its three regimes, the strict minimum on both sides, `Haircut_Posted` on either side, the
two balances in one pass, nothing rounded, a mark nobody has named - and collateralised credit
Monte Carlos over jobs the record compiled: the engine opens on the balance written, on 4,096 paths
spanning both thresholds and both minimums the spine's call moves the balance where the engine's
scan does, to its required support, and on the day a forward under the set pays it does so under
either `Exclude_Paid_Today`. `tests/test_pnl.py` reads the call over three marked closes - settled,
margin beside it, a delivery dated ahead that the worklist drops at once, the day's payments under
both settings, gold nobody can value refusing every run that values the book, the risk's cache
kept, a day with no marks, a seat at one node, one acting everywhere and the worklist's `settle`
seat - every killing mutation their docstrings name red.

## What is not built yet

No DuckDB and no reading plane: every question a desk asks the record is a fold, the largest fold
state is 167 bytes per event, and the trigger is one projector's state passing a declared budget.
The network is READ-ONLY and localhost's: tokens are verified rather than fetched, the three reads a
replica uses serve and never take, and no write path is exposed beyond the box — under that posture,
with no key set named, `actor` is ATTRIBUTION rather than authentication and the honest control is
the bind address; with one named, the seat is the verified token's subject. No
class-key rotation (rewrap adds recipients; rotation is a later logged event). The external anchor
hook is the checkpoint pair on `DV_Spine status`; wiring it to an anchor target is deployment data,
out of scope by the design's own sentence.

Six boundaries stand, declared rather than discovered. A STANDING run must post its job document — a
`plan_id` names a parse, `Context.save_json` is explicitly not a complete round trip, and a provenance
chain whose first link is a document nobody can recompile is worse than none — so it refuses by name.
The surveillance and admin READ of a private market waits on a second entitlement class and a `read`
row per subject, the reclassification `vocabulary.classify` ships dormant for; until then the owner
rule refuses at the verb without minting a fact. No rejection is filed AUTOMATICALLY: a ticket no tier
admits answers every sentence of its route, and a verdict is a seat's decision the tiers policy names
no seat for. A material market move between a quote and the client's word is REPORTED rather than
refused; a desk wanting a refusal would declare per-field epsilons, which wants a values-vector diff
that says which numbers moved where the record compares two addresses. A blob is served by the hub
and by nobody else: a peer server is a second entitlement-evaluating surface on a box that is not
the writer, and this phase has one deployment. And OPENING a log still scans every segment to build
the head, the tag index and the offset per LSN — 11.3 ms at 2,005 events, which every reader pays
once and 6a's seek does not touch — so the reading half is closed and the open's own half waits on a
checkpointed index: the three maps written down at a close and re-read instead of rebuilt.
