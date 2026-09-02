# The Spine

`derivus_spine/` is the append-only book of record being built around the engine — the center a desk
box is the edge of. The full seven-increment design lives in the owner's brief outside the tree; this
page documents what is BUILT, which is **increments 1, 2 and 3**: the log, the blob store and the chain
(riding on them: identity, capability enforcement and key custody), and on top of those the booking
verbs, the attestation lanes and the two-dimensional firmness check. No projections, no network — a
library, a CLI, five delegators on `Context`, and 225 gates. Nothing here imports the engine, and
exactly one module under `derivus/` imports `derivus_spine`: `derivus/spine.py`.

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
```

The manifest is a PROJECTION — presence is the tree itself, and `verify` rebuilds what it needs by
walking it — so "the record never trusts what it can re-derive" holds from day one. DuckDB belongs to
the reading plane and arrives with increment 4. A blob whose bytes do not match its hash refuses on
read; a put colliding with different bytes at the same hash is a NAMED REFUSAL, never a dedup; and the
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
`test_spine_store.py`; the glob `tests/test_spine*.py` is the wider nine-file set worth the 225 above),
all real stores in temp dirs, every fault injected by doctoring DATA on disk. The shapes worth naming:
three tampers on three copies, each caught by a different layer (body byte by the chain, envelope field
by the AAD, record_time by a keyless replica); a re-forged tail caught by the interior binding AND its
dual caught by the stale tag, each proven independently load-bearing in the posture where the other is
absent; the brief's synthetic book (late booking, backdated amendment, republished fixing under one
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
brief's rule; display names live in `names.json`, a mutable side table OUTSIDE the log whose erasure is
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
crypto-shredded copy. **THREE residuals are declared**: the brief's two (forward-only revocation,
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

**The verbs live in the spine and the engine gets delegators.** The brief's build order says "booking
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

## What is not built yet

No projections, no diary, no DuckDB — **increment 4**, which is also where the book file becomes a
fold-and-hydrate projection pinned to its LSN, and where the plan compiler becomes a FOLD over fixings
supersession rather than a recompile of the document that was submitted. No tier policy, no doorbell, no
generated MCP binding — **5 through 7**. No network anywhere yet: tokens are verified, never fetched,
and no write path is exposed beyond localhost. No class-key rotation (rewrap adds recipients; rotation
is a later logged event). The external anchor hook is the checkpoint pair on `DV_Spine status`; wiring
it to an anchor target is deployment data, out of scope by the design's own sentence.

Two boundaries of increment 3's own, declared rather than discovered. `pin_result` reads its tolerance
policy and re-executes BEFORE the writer adjudicates the append, so an unscoped actor can make the hub
pay for one execution it will then refuse; the fix is not a second authorization check inside the verb
(one evaluator, one place) but queue admission, which the brief puts under its own capability in
increment 5. And a STANDING run must post its job document — a `plan_id` names a parse,
`Context.save_json` is explicitly not a complete round trip, and a provenance chain whose first link is
a document nobody can recompile is worse than none — so it refuses by name.
