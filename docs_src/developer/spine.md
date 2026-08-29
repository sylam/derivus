# The Spine

`derivus_spine/` is the append-only book of record being built around the engine — the center a
desk box is the edge of. The full workstream design (seven increments: log, identity, booking
verbs, projections, workflow tiers, distribution, bindings) lives in the owner's brief outside
the tree, per the no-specs-in-the-repo rule; this page documents what is BUILT, which is
**increments 1 and 2**: the log, the blob store, the chain — and riding them, identity,
capability enforcement and key custody. No verbs on `Context`, no projections, no network — a
library, a CLI, and 167 gates. Nothing under `derivus/` is touched, and nothing here imports it.

## The package, and the one dependency

A sibling package on the `derivus_mcp` terms: in the wheel, never importing the engine. Its
import surface is **stdlib plus `cryptography`** (AES-GCM sealing, Ed25519 checkpoint
signatures) and nothing else — held by an AST gate over every module and a subprocess gate
proving `import derivus_spine` pulls no torch and no `derivus.*`. The extra is
`pip install derivus[enterprise]`, orthogonal to `desk` (a desk edge under the spine is
`derivus[desk,enterprise]`). `DV_Spine init | verify [--chain-only] | checkpoint | status` is
the console script; the home is `--home`, else `DV_SPINE_HOME`, else `~/.derivus_spine` — the
spine is the CENTER's store and deliberately not `DV_HOME`, which is the edge's.

## The truth layer is files

```
DV_SPINE_HOME/
  log/segment-00000001.jsonl   append-only frames, fsync per append, roll at 64 MiB
  blobs/ab/cd/<sha256>         content-addressed, write-once: tmp + fsync + os.replace
  keys/                        blind.key, class_firm.key, the Ed25519 checkpoint pair
```

The manifest is a PROJECTION — presence is the tree itself, and `verify` rebuilds what it needs
by walking it — so "the record never trusts what it can re-derive" applies to every reading
surface from day one. DuckDB belongs to the reading plane and arrives with increment 4's
projections, never here. A blob under a hash that does not match its bytes refuses on read; a
put that collides with different bytes at the same hash is a NAMED REFUSAL, never a dedup; and
the store has **no verb for forgetting** — retention arrives in a later increment as a logged
event, and the absence of a delete method is gated as the increment-1 form of that law.

## Two hashes, and why the envelope carries a tag instead of one of them

Every event is a sealed body under a firm-visible envelope. `content_hash` is SHA-256 over the
RFC 8785 canonicalization of the semantic tuple (type, version, effective_time, actor, book,
body) — but it rides INSIDE the sealed body, because a plaintext hash in a public envelope is a
dictionary oracle for low-entropy bodies (an approval over a plan hash visible elsewhere could
be confirmed by hashing candidates). The envelope instead carries the `idempotency_tag`: an
HMAC of the same canonical bytes under a writer-held blind key, so no keyless confirmation
exists — gated with a toy key, and by the same tuple landing different tags in different homes.
`event_hash` is SHA-256 over (idempotency_tag, ciphertext_hash, prev_hash, record_time): the
chain, from a genesis `prev_hash` of sixty-four zeros. LSN is positional and outside both.

The canonicaliser is VENDORED (stdlib only) and held to the RFC's own vectors — the trap being
ECMAScript number serialization, where Python's `repr` has the right digits in the wrong
clothing (`1e+16` where ES writes `10000000000000000`, `1e-07` where ES writes `1e-7`); a
repr-passthrough implementation turns the gate file red by design. Keys sort by UTF-16 code
unit, ints past 2^53 and non-finite floats refuse by name.

Sealing is AES-256-GCM under the firm class key with a fresh 96-bit nonce, AAD over the nine
pre-LSN envelope fields — so no envelope field can move without the body refusing to open — and
the interior binding `{content_hash, payload}` closing the loop. Classification ships DORMANT:
one class, everyone holds it, and the day desk two arrives walls are a classification decision,
not a redesign. Sealing itself is not deferred with the classes: genesis-era bodies must be
crypto-shreddable, and the shred is gated — delete the class key and the chain still verifies
while every body is unreadable.

## The writer

One writer, and it is now enforced rather than asserted: the first `append` takes an exclusive
byte-range claim on `log/.writer.lock` (msvcrt/fcntl per platform) and a second writer refuses
with `WriterBusy` naming the lock — verify never takes it, so replicas read freely. The append
flow is ordered so nothing partial can land: validate against the closed vocabulary → canonical
bytes → tag → duplicate-tag check (on a hit the writer DECRYPTS the stored body and
byte-compares canonical plaintexts: equal is the safe retry, coalesced onto the existing LSN;
different is `CollisionRefusal` naming both) → every blob the body cites must already be in the
store (`vocabulary.BLOB_FIELDS` declares which fields name stored bytes, with its exclusions
justified in place) → stamp, seal, hash, one fsynced line. A retry with no caller-passed
`effective_time` coalesces because the tuple carries `null` there — the writer's own clock is
not part of the fact — and `as_of_key` resolves null to `record_time` at read time.

The vocabulary is CLOSED: sixteen fact types plus `checkpoint`, validators naming the missing
or surplus field, and a consequence-shaped submission ("knocked_out") refused with the closure
stated — knocks and expiries are projections, never events. The torn-tail rule is exact: a
final line with no terminating newline was never durable (the write is one call, newline last)
and is truncated on open; a newline-TERMINATED line that will not parse is a durable line that
was altered, and is `ChainBroken` — the distinction that stops the recovery path being usable
to roll a log back through its own checkpoints.

## Checkpoints, and verification as a replica

`checkpoint` events sign (lsn, head event_hash) with the deployment's Ed25519 key. The
verifying key is published at genesis as a firm-class policy blob, so a replica asserts
checkpoint AUTHENTICITY — not merely chain integrity — from checkpoint one, against the log's
own contents rather than any local file: the gates prove it by deleting the local
`checkpoint_verify.key` (verification still passes) and by deleting the published blob
(refusal naming it). Rotation is a fold: every `checkpoint_verifying_key` declaration is
recorded with its LSN and each checkpoint verifies under the key in force at ITS position, so
a logged rotation neither bricks history nor lets a later declaration retro-invalidate genesis.

`verify_home` runs in two modes. Entitled: chain, seals, interior bindings, blinded tags
(where the blind key is present), checkpoint signatures, referential closure over every cited
blob. Chain-only — the keyless replica posture, a home of `log/` and `blobs/` alone — verifies
the whole chain over ciphertext it cannot read and says honestly that checkpoint authenticity
and bodies were not assessed, rather than skipping silently. Frames refuse surplus fields in
both modes: there is no unauthenticated channel into the record.

## The gates

103, in four files (`tests/test_spine*.py`), all real stores in temp dirs, every fault injected
by doctoring DATA on disk — nothing monkeypatched. The shape worth naming: three tampers on
three copies, each caught by a different layer (body byte by the chain, envelope field by the
AAD, record_time by a keyless replica); a re-forged tail caught by the interior binding AND its
dual caught by the stale tag, each proven independently load-bearing in the posture where the
other is absent; the brief's synthetic book (late booking, backdated amendment, republished
fixing under one (index, date, source) key, superseding close) driving all sixteen fact types
through the writer with as-of provably departing from as-at; and restore as file-copy — the
two-directory replica and the three-directory disaster copy both re-verified on the far side.
Eleven mutants were run against the finished build and each now dies by a named test; the two
that survived the first pass (checkpoint-key provenance, the blinded-tag recomputation) are
exactly the assertions the review added.

## Increment 2 — identity, attribution, key custody

**Identity is bought, not built.** `identity.py` VERIFIES an OIDC ID token against a JWKS the
deployment hands in as data — nothing is fetched, verification stays local — under an
RS256/ES256 allowlist that refuses `none` and every HMAC by name before any key is selected
(the alg-confusion attacks are gates, not warnings), with `kid` selection across key types, the
ES256 raw-`R||S` signature contract, `exp`/`nbf` on an injectable clock, and OIDC Core's `azp`
rule for multi-audience tokens — a co-audienced client cannot replay its own token as a spine
credential. The subject reference is the token's `sub`, pseudonymous by the brief's rule;
display names live in `names.json`, a mutable side table OUTSIDE the log whose erasure is gated
to leave every chain byte identical.

**Capabilities are one document and one pure function.** The document — grants of (verb × book)
over `draft | validate | book | approve | mark | admin`, plus READ over entitlement classes —
is a hashed blob declared through the ordinary writer (`policy_declared` with the RESERVED
policy name `capabilities`), each declaration a complete replacement, resolved by fold-at-LSN
so "could X do Y in March" replays from the log like every other question. Enforcement
activates BY DECLARATION: a home with no document runs as the single-user instrument it is —
which is why every increment-1 home and gate is bit-for-bit untouched — and once one is in
force the writer refuses an unscoped append AND logs the refusal as a `capability_denied` fact
under the writer's reserved actor, because a decision is a fact. A document whose blob has been
doctored or lost folds to UNREADABLE rather than raising: every verb refuses by name, custody
hands out no key, and the break-glass walk leads OUT — the condition the recovery grant exists
for cannot brick the recovery itself. `break_glass_used` is gated on the genesis grant from
event one, the admin it restores is revocable by the next capabilities declaration (a recovery
is not a coup), and the classifier seam (`vocabulary.classify`) now derives the entitlement
class the envelope carries — "firm" for everything until desk two, one function instead of one
constant.

**Custody.** Per-seat X25519 keypairs at enrollment; the firm class key wrapped per
READ-entitled subject — ephemeral ECDH, HKDF, AES-GCM with the (class, subject) pair bound
into the AAD, so a wrap re-addressed to another seat refuses to open — plus an escrow wrap
under a declared escrow key. `rewrap` is idempotent and driven by the document in force;
`grant` reports the rewrap it now owes, so "rewrap on grant change" is a printed obligation
rather than operator memory; `materialize` turns a chain-only replica entitled off its wrap
without overwriting anything; escrow recovery is gated on a crypto-shredded copy. THREE
residuals are declared: the brief's two (forward-only revocation, traffic shape) and this
increment's own — a hub-minted seat key is a bootstrap the hub has seen, stated in
`custody.py`'s docstring, with seat-generated `--public-key` enrollment as the form that
eliminates it.

The CLI grows `enroll | grant | rewrap | name | whoami`. Of the 64 new gates, the shapes worth
naming: the strand-and-recover walk (a document stranding the last admin, every later
declaration refused, the genesis seat walking out, the recovered admin later revoked); the
stale-fold check (a revocation lands and the same open handle answers off the platter, never a
snapshot — the TOCTOU the review caught on the custody path); and the review's eleven mutants
plus the fix pass's thirteen, each dead by a named test.

## What is not built yet

No booking verbs on `Context`, no `result_pinned`, no attestation lanes — increment 3, which
waits on the engine's `Market Prices` partition row before quote pinning is wired. No
projections, no diary, no DuckDB — increment 4. No tier policy, no doorbell, no generated MCP
binding — 5 through 7. No network anywhere yet: tokens are verified, never fetched, and the
write path beyond localhost that increment 2 exists to gate arrives with increment 3's verbs —
already gated when it does. No class-key rotation (rewrap adds recipients; rotation is a later
logged event). The external anchor hook is the checkpoint pair on `DV_Spine status`; wiring it
to an anchor target is deployment data, out of scope by the design's own sentence.
