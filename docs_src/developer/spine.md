# The Spine

`derivus_spine/` is the append-only book of record the engine prices against. Every booking,
print, close, decision and attested run is a fact filed through one writer, chained, sealed and
never edited, and every view a desk reads is a fold of those facts. It is a package of its own
because a record must verify on a box that cannot price a trade.

| where | what |
| --- | --- |
| `log.py`, `store.py`, `seal.py`, `canon.py` | the single writer; the blob store; a home's keys; RFC 8785 |
| `vocabulary.py`, `capability.py`, `identity.py`, `custody.py` | what may be said, by whom; tokens; key wraps |
| `verbs.py`, `policy.py`, `firmness.py`, `tiers.py` | the acts; the reserved policies; two pure evaluators |
| `projections.py`, `verify.py`, `replica.py`, `oracle.py`, `cli.py` | folds; verification; the follower; invariants; `DV_Spine` |
| `derivus/spine.py` | the one seam with the engine: `compiled_job`, `routed`, `diary`, `PnL`, `Collateral` |
| `derivus/service.py`, `derivus_mcp/server.py` | the HTTP verbs and the same verbs as tools |

Tests named below are in `tests/test_spine*.py`, `test_diary.py`, `test_pnl.py`,
`test_collateral.py` and `test_position_scaling.py`, on real homes with faults injected as data; a
rule marked **Held by** goes red under that test if it is broken.

## How it fits {#how-it-fits}

Four mechanisms carry all of it. Whatever a desk, a controller or a credit officer asks is one of
them or a composition of them, so state a need in these four before adding a fifth.

- **Facts.** An event a seat files through a verb and the hub, the single writer, appends: a
  booking, a print, a declaration, an approval, a close. Each names who filed it, is chained to the
  one before and is never edited; a correction is another fact.
- **Blobs.** Whatever has content - terms, a market's values, a job, a result - is stored once under
  its hash and cited by it.
- **Folds.** Every view is a fold of the facts up to a point in the record: a position, a P&L, what
  stands - the latest declaration of a name there. Nothing derived is stored as truth, and the point
  is the citation, so nothing is priced against an older declaration by choice.
- **The engine.** A function of a job document and its market: the record compiles the document
  ([`compiled_job`](#the-plan)), the engine prices it, the result is filed by address.

A `fill` says who booked what, against whom and where, so one fold has two views: by portfolio it
is the book, where risk and P&L are owned; by agreement it is the credit view, the netting set an
exposure is priced on. The market a run reads is what stands, and `load_json` merges each load over
a context, so the hub's close runs the job a desk runs on its own box
([Architecture](architecture.md#market-data-context-job)). Seats, the portfolio tree, tiers and
close times are a bank's declarations, never code. What is unbuilt is under [Open](#open).

## The package and its dependency {#the-package}

`derivus_spine` imports the standard library and `cryptography` (the `enterprise` extra) and never
the engine. Only `derivus/spine.py` imports it, lazily, so `import derivus` never needs the extra.
`DV_SPINE_HOME` is the whole switch, read per call: unset, nothing records and lanes are inert, and
it never falls back to the CLI's default home, so a box where someone once ran `DV_Spine init` does
not start recording. **Held by**
`test_spine_imports.py::test_the_spine_imports_nothing_but_the_standard_library_and_cryptography`,
`::test_one_module_under_derivus_imports_the_spine_and_it_is_the_seam`,
`test_spine_engine.py::test_with_no_spine_home_the_edge_is_the_edge_it_always_was`.

`Context` has nine delegators (`book`, `amend`, `apply_lifecycle`, `transition`, `declare_market`,
`declare_close`, `approve`, `reject`, `pin_result`) that hand the spine engine-canonical bytes.
`spine.writing()` holds a process lock for one act, so threads queue rather than meet `WriterBusy`.
A `SpineRefusal` leaves as `SpineRefused`, a `ValueError`, so a 422 carries the library's sentence.

### The CLI {#the-cli}

`DV_Spine` works on `--home`, else `DV_SPINE_HOME`, else `~/.derivus_spine`; a refusal exits 1.

| verb | what it does |
| --- | --- |
| `init` | mint a home: keys, the genesis facts, the first checkpoint |
| `verify [--chain-only]` | `verify_home`, entitled or keyless |
| `checkpoint`, `status` | sign the head; print it, and whether bodies open |
| `follow <url> [--once] [--verify] [--interval] [--blobs] [--actor]` | pull a hub's frames into a replica |
| `oracle [--script] [--against] [--diary] [--pnl]` | the fourteen invariants; exit 1 where one failed |
| `seed [--at] [--out]` | mint the folds' seeds at an official close |
| `enroll`, `grant`, `rewrap`, `name`, `whoami` | seats, capabilities, key wraps, display names, token check |
| `declare`, `policy`, `portfolio` | declare a policy, report those in force, declare a node |

## The truth layer is files {#the-truth-layer}

```
DV_SPINE_HOME/
  .writer.lock                            the writer's claim
  log/segment-00000001.jsonl              frames, one fsync per append, rolled past 64 MiB
  blobs/ab/cd/<sha256>                    content-addressed, write-once (blobs/tmp/ is scratch)
  keys/                                   blind.key, class_firm.key, checkpoint_signing.key, checkpoint_verify.key
  keys/seats/<sha256(subject)[:16]>.key   a seat's private key where the hub minted it
  names.json                              display names, outside the log
  seeds/<projector>-<version>-<lsn>.json  a fold cached at a close: derivable, deletable
```

A replica is `log/` and `blobs/` alone. A blob put fsyncs a scratch file, `os.replace`s it onto
its address and fsyncs the directory on POSIX (Windows has no directory handle). A newly rolled
segment's directory entry is not fsynced, so a power loss just after a roll can lose its name. No
fact cites a blob not on the platter; a blob no longer hashing to its address refuses on read; a
colliding put refuses rather than dedups; and nothing deletes - a blob class shrinks only through a
logged `retention_declared`. **Held by** `test_spine_store.py`,
`test_spine.py::test_an_event_citing_an_absent_blob_does_not_append`,
`::test_a_cited_blob_that_went_quietly_is_caught_by_the_manifest`.

## Hashes and sealing {#hashes}

A frame is an envelope of exactly twelve fields over a sealed body. `content_hash` rides inside the
body, a plaintext hash in a public envelope being a dictionary oracle for low-entropy bodies; the
envelope carries a keyed `idempotency_tag` [@rfc2104] instead, and `event_hash` is over the
ciphertext so a keyless replica re-derives every link. "JCS" is RFC 8785 [@rfc8785] as `canon.py`
emits it; "engine form" is `json.dumps(obj, sort_keys=True, separators=(',', ':'),
cls=CustomJsonEncoder)` in UTF-8 (`derivus.content_hash`), which an RFC 8785 library does not
reproduce. SHA-256 is FIPS 180-4 [@fips180-4]; digests are lowercase hex.

| value | bytes | function |
| --- | --- | --- |
| `content_hash` (inside the body) | JCS of `{actor, body, book, effective_time, type, version}` | SHA-256 |
| `idempotency_tag` (envelope) | the same JCS bytes | HMAC-SHA256 under `keys/blind.key` |
| `body` (envelope, base64) | plaintext: JCS of `{content_hash, payload}`; AAD: JCS of the nine fields `actor, book, effective_time, entitlement_class, event_type, event_version, idempotency_tag, prev_hash, record_time` | AES-256-GCM under `keys/class_firm.key`, fresh random 12-byte nonce; stored as `nonce ‖ ciphertext ‖ 16-byte tag` |
| `ciphertext_hash` (recomputed, not stored) | the base64-decoded `body` | SHA-256 |
| `event_hash` | JCS of `{ciphertext_hash, idempotency_tag, prev_hash, record_time}`; genesis `prev_hash` is sixty-four zeros | SHA-256 |
| a `checkpoint`'s `signature` | JCS of `{event_hash, lsn}` | Ed25519 under `keys/checkpoint_signing.key` |
| a blob's address | the blob's bytes as stored | SHA-256 |
| `cashflow_key`, `ticket`, `fill_key`, a policy or capabilities blob, a seed's `state_hash` | JCS of the object | SHA-256 |
| an instrument's address (`fill.instrument`, `amendment`), an agreement's `terms`, `plan_hash`, `values_hash`, a run's job, values and result blobs | engine form | SHA-256 |

The forms compose because a blob is named by the SHA-256 of the bytes it is handed: `values_hash`
and the stored values vector's address are one number.

The vendored canonicaliser is held to the RFC's vectors; the trap is ECMAScript number spelling
(`repr` writes `1e+16` where ES writes `10000000000000000`). Keys sort by UTF-16 code unit. An
integer past 2^53, a non-finite float and a lone surrogate refuse, having no single JSON spelling;
a float at or past 2^53 becomes an integer literal, so a re-parse passes `log.parse_number` as
`parse_int`. **Held by** `test_spine_canon.py`.

Through the AAD no envelope field moves without the body refusing to open, and the interior
binding catches an insider who reseals a payload. Random 96-bit nonces under one key are bounded to
2^32 invocations by NIST SP 800-38D §8.3 [@nist-sp800-38d]; the class key is never rotated and
nothing counts its invocations. One entitlement class exists (`vocabulary.classify`), so a second
desk's walls are a classification decision. Deleting `class_firm.key` crypto-shreds: the chain
verifies, no body opens. **Held by**
`test_spine.py::test_a_doctored_envelope_field_breaks_the_seal_the_chain_alone_would_miss`,
`::test_a_re_forged_tail_frame_is_caught_by_the_interior_binding`,
`::test_destroying_the_class_key_leaves_the_chain_green_and_the_bodies_gone`.

## The writer {#the-writer}

The first `append` locks `.writer.lock` exclusively and re-reads the head, since two writers would
assign one LSN twice; reading never claims, so a replica is never locked out. `SpineLog.append`
then runs in this order, so no refusal lands any part of the fact:

1. the vocabulary, the actor, the book, a `portfolio` outside the envelope's book, the time format;
2. authorisation: a refused seat lands a `capability_denied`, then raises; a capabilities
   declaration is parsed here, while the operator can still fix it;
3. `record_time`, the canonical tuple, `content_hash`, the tag;
4. a tag already held: the stored body decrypted and byte-compared - equal coalesces onto its LSN,
   different raises `CollisionRefusal`;
5. every cited blob present (`BLOB_FIELDS`); seal, `event_hash`, pending blobs put, one fsynced line.

`effective_time` is the caller's or null, never the writer's clock, because the tuple contains it:
a retry is the same fact only if arrival is not part of it. A null reads as of `record_time`.
**Held by** `test_spine.py::test_a_second_writer_is_refused_rather_than_left_to_corrupt_the_home`,
`::test_a_retry_that_says_nothing_about_when_is_still_the_same_fact`,
`::test_a_duplicate_tag_over_different_bytes_is_refused_by_name`.

The vocabulary is closed, in five parts by who speaks: trading facts, custody, provenance
(`run_completed`, `result_pinned`, `quote_filed`), reference (the paper) and the writer's
`capability_denied`; bodies are closed at the field level except `policy_declared`. `knocked_out`
does not exist: a knock is a fold, and storing one would be a second truth about the barrier.
`SpineLog.own`, the one ungated path, stamps the reserved actor `writer` on denials, attested runs,
snapshots and an automatic tier's approval; no seat files the first three or takes that name.
**Held by** `test_spine.py::test_the_vocabulary_is_closed_and_its_bodies_are_shaped`,
`test_spine_capability.py::test_the_writers_own_voice_is_no_seat_and_files_only_its_own`.

The torn tail is the one repair, made on every open: the newline is a write's last byte, so an
unparseable unterminated last line of the last segment was interrupted and is truncated, while an
unparseable line anywhere else was altered and raises `ChainBroken`, so the repair cannot roll a
log back past its checkpoints. Opens scan without the claim, so a reader opening mid-append could
truncate that append's bytes; no gate covers that interleaving. **Held by**
`test_spine.py::test_a_torn_final_line_is_truncated_and_the_next_append_chains_onto_the_head`,
`::test_a_terminated_line_that_will_not_parse_is_a_broken_chain_not_a_torn_tail`.

## Checkpoints and verification {#verification}

Genesis writes the admin grant, the break-glass grant, the verifying key as a policy blob, and a
checkpoint. A `checkpoint` signs the head before it [@rfc8032] and verifies against the key in the
log, not `keys/`, so a copy holding no secret asserts authenticity; each verifies under the latest
`checkpoint_verifying_key` at or before it, so a rotation breaks no history. Only `init` and
`DV_Spine checkpoint` sign. **Held by**
`test_spine.py::test_the_verifying_key_comes_out_of_the_log_and_not_out_of_the_keys_directory`,
`::test_the_verifying_key_is_pinned_by_lsn_so_a_rotation_reads_both_sides`.

`verify_home` entitled checks the chain and every seal, binding, tag (where `blind.key` is present),
checkpoint and cited blob; chain-only checks the chain over ciphertext and reports checkpoints `not
assessed`, never a zero read as a count. Both refuse a thirteenth frame field, which no hash covers.

### Trust limits {#trust-limits}

| property | holds | does not hold |
| --- | --- | --- |
| Chain integrity | every `event_hash` recomputes and links, keyless | against anyone who can rewrite the whole home |
| Checkpoint authenticity | each verifies under the key in force at its LSN | a rotation is trusted on an intact chain; nothing asks that the previous key signed it |
| Tamper-evidence against the hub | against a genesis hash and key obtained out of band, or an agreeing replica | on the hub alone: its signing key is in its home, so write access re-signs a rewritten history |
| Signed history | up to a replica's last checkpoint | past it: chained, unsigned |
| Idempotency tags | recomputed where `blind.key` is present | elsewhere |
| Time | `record_time` is the hub's clock | no external timestamp or anchor; `DV_Spine status` prints the head to publish |
| Read revocation | a revoked seat gets no new wrap | forward-only: it keeps what its wrap opened; no class-key rotation exists |
| Who acted | with a key set, the verified token's subject | with none, the stated `actor` - attribution, not authentication; CORS admits any origin unless `--origin` narrows it |
| Confidentiality | bodies are sealed | envelopes go to any caller `/spine/frames` admits; it asks for no seat |

### What a replay tuple pins {#replay}

`(plan_hash, values_hash, engine_version, seed)`. `engine_version` is the hand-set
`derivus/_version.py`; no commit, torch or CUDA version or device is recorded; and
`Deterministic_Kernels` defaults to `No` (`Yes` calls `torch.use_deterministic_algorithms(True,
warn_only=True)`, which warns rather than refuses). Bit-equality on replay is to be expected only on
one build, stack and device, and for a GPU gradient only under `Yes`; elsewhere a tolerance policy
decides.

## Identity, capabilities and custody {#identity}

**Identity.** `identity.verify_id_token` checks an OIDC ID token [@openid] against a JWKS
[@rfc7519] handed in as data, refusing `none` and every HMAC before a key is selected, so the
alg-confusion forgery is unrepresentable, and holding `azp` so a co-audienced client cannot replay
its token. It answers `sub`; display names live in `names.json`, outside the chain. **Held by**
`test_spine_identity.py`.

Naming a key set (`DV_SPINE_JWKS`, with `DV_SPINE_ISSUER` and `DV_SPINE_AUDIENCE`) makes every request but the doorbell
carry a token, its subject set in the request's context by an async dependency (a sync one would
set a thread-pool copy). `spine.actor` answers that subject, refusing another name; else the stated
name; else, with no capabilities document, `DV_SPINE_ACTOR`; under one, an unnamed act refuses
rather than file a name nobody chose. **Held by**
`test_spine_engine.py::test_a_bearer_token_names_the_seat_and_nothing_else_does`.

**Capabilities.** A canonical blob under the reserved name `capabilities`, each declaration a
complete replacement, folded at an LSN so "could X do Y in March" replays: `{"grants": [{subject,
verb, book}], "read": [{subject, class}]}`. Seven verbs, none implying another:

| verb | files |
| --- | --- |
| `validate` | nothing: work the queue admits |
| `book` | `fill`, `amendment`, `election`, `quote_filed` |
| `approve` | `approval`, `rejection`, `result_pinned` |
| `mark` | `market_declared`, `official_close_declared`, `fixing_observed`, `determination` |
| `document` | `entity_declared`, `agreement_declared` |
| `settle` | `status_transition` |
| `admin` | `policy_declared`, `portfolio_declared`, `retention_declared`, `rehash_declared`, `checkpoint`, `seat_enrolled`, `key_wrapped` |

A type missing from the map demands `admin`, so a forgotten entry costs governance, not an open
write. A grant's `book` is a node reaching every path below it: `BANK/FX` covers `BANK/FX/Options`,
not `BANK/FXO` or `BANK`. `capability.scope_of`, shared by writer and oracle, judges the firm's own
facts (policies, markets, prints, runs, the paper, custody) at the firm, which only `*` reaches; a
fill, restrike, quote or verdict at its `portfolio`; a declared node at its parent; else the book.
**Held by**
`test_spine_capability.py::test_a_grant_at_a_node_reaches_every_path_under_it_and_nothing_beside_it`,
`test_spine_oracle.py::test_the_firms_own_facts_are_judged_at_the_firm_whatever_their_envelope_names`.

!!! warning "Enforcement activates by declaration"
    With no capabilities document in force every append and queued job is admitted; once one
    stands, every verb asks it. Break-glass is gated from genesis instead, because a recovery a
    missing declaration opened to anyone would be no recovery. **Held by**
    `test_spine_capability.py::test_a_fresh_home_enforces_nothing_and_writes_exactly_as_increment_one_did`,
    `::test_the_break_glass_handle_is_gated_from_event_one_and_not_by_declaration`.

A document whose blob was altered or lost folds to `UNREADABLE` instead of raising, since a raise
inside the writer's hook would take the recovery with it: every verb refuses and break-glass still
answers, restoring admin until the next declaration by it or an admin over `*`. A node admin
replaces the document only where every grant it moves sits under its nodes and the `read` rows, each
a key to every body, are unchanged (`capability.declarable`). **Held by**
`test_spine_capability.py::test_a_doctored_policy_blob_folds_to_unreadable_and_break_glass_walks_out_of_it`,
`::test_a_node_admin_declares_only_what_its_nodes_reach`.

**Custody.** The class key is wrapped to each `read` seat's X25519 [@rfc7748] key by ephemeral
ECDH, HKDF [@rfc5869] and AES-GCM with `{class, subject}` as AAD, so a misfiled wrap will not open.
`rewrap` is idempotent, `custody.materialize` entitles a replica without overwriting a key, and a
declared escrow key recovers the class key after a shred; `enroll --public-key` keeps a seat's
private key off the hub. **Held by** `test_spine_custody.py`.

## Lanes and attestation {#lanes}

A run is recorded if and only if a fact will cite it. `telemetry` (a tick) and `curiosity` (a
what-if, `/execute`'s default) mint nothing; `standing` mints `run_completed` in the writer's voice
with the replay tuple and the job, values and result. Content addressing dedupes numbers, not
standing: a standing run whose numbers exist still attests, one coalescing onto a run in flight is
promoted in `ComputeExecutor.submit`, and one whose attestation is refused fails rather than serve
an unbacked citation. It posts its document, a `plan_id` naming only a parse. **Held by**
`test_spine_engine.py::test_a_synthetic_tick_sequence_mints_nothing_and_the_absence_is_asserted`,
`::test_a_standing_run_that_coalesces_onto_one_in_flight_is_attested_when_it_lands`,
`::test_an_attestation_the_record_refuses_fails_the_run_it_was_for`.

`pin_result` promotes a tuple this hub did not witness through an injected `executor`, the record
being unable to import the engine: a cache hit on a matching `run_completed`, bit-equality, or
`policy.compare`. The fold never reads `result_pinned`, so a promotion is never evidence for the
next, and the spine admits no tolerance of its own: a home declaring none pins nothing. **Held by**
`test_spine_verbs.py::test_a_pin_never_bootstraps_itself_off_another_pin`,
`::test_a_home_that_declared_no_tolerance_policy_pins_nothing_at_all`.

## Folds, seeds and fixings {#folds}

A projector is `initial`, `apply`, `rows` and the types it `reads`; `fold` opens only those bodies,
and rows are sorted canonical JSON, so a golden replay is a byte comparison. The fourteen:
`positions` (net per instrument × agreement × portfolio, clips under each ticket), `costs`, `cash`,
`lifecycle` (prints under `(index, date, source)`, elections, statuses), `markets`, `attestations`,
`decisions`, `quotes`, `denials`, `blotter`, `entities`, `agreements`, `portfolios`, and `activity`,
the strip, which opens no body so a keyless replica shows every LSN.

A keyed row stands by the as-of key `(effective_time, lsn)`, so a backdated restatement does not win
by arriving last, and what it beat stays readable; a second administrator's print is a second row. A
knock is read off the fold (`projections.knocked`), never filed. **Held by**
`test_spine_projections.py::test_every_projector_replays_to_its_committed_golden`,
`::test_a_republished_print_supersedes_by_as_of_key_and_the_first_is_still_read`.

**Seeds.** A seed is a fold's `(lsn, state)` written at an official close, where the desk agrees
what the day was, carrying the close's event hash, projector, version and state hash; one from
another home or version, torn, or edited refuses by name. `DV_Spine seed` mints all but the strip,
whose state is its history; readers start at the newest seed in `DV_SPINE_SEEDS` (else `seeds/`). A
projector bumps its `version` when rows change, so an old seed refuses. **Held by**
`test_spine_projections.py::test_a_seeded_fold_is_the_from_genesis_fold_byte_for_byte`,
`::test_a_projector_one_version_on_folds_the_same_bodies_and_a_seed_is_verified_not_trusted`.

`spine.advancing` holds each projector's `(lsn, state)` and advances it, so a booking pays for the
events since the last, not the whole history; pairs are keyed on the genesis hash, so a re-minted
home is another record. **Held by**
`test_spine_engine.py::test_a_home_re_minted_where_the_last_one_stood_is_a_different_record`.

`fixings_at` resolves prints by the source order the `fixings` policy declares per index, the first
source holding a print winning. With no policy nothing resolves; under one, an index asked for and
not ordered refuses - a fixing nobody vouched for is not one a plan may use - and one not asked for
is left alone. **Held by** `test_spine_projections.py::test_a_fixing_with_no_declared_source_refuses_and_the_order_is_the_authority`.

## The plan as a fold {#the-plan}

`spine.compiled_job(document, lsn, strict, runs)` is the job a plan hashes: the document plus what
the record holds. `/execute`, `/prepare`, the diary, the marks and every valuing read compile first,
so a plan named once is the plan that runs and an auditor recompiles from what was printed.
**Held by** `test_spine_engine.py::test_the_plan_compiles_from_the_fold_rather_than_from_what_was_typed`.

- **Nets.** Each node the record holds a position in is written at its net over the agreement's
  set by `spine.scaled`: declared `sized` fields multiplied, a negative net taken as the mirror (a
  declared `side` flipped, else sizes signed). A swaption's underlying is its terms, never flipped.
  A zero net or a repeated node is ignored, not removed, so resolved paths stay valid. A block
  stating none of its amounts refuses rather than price one unit, and a structure held whole is one
  instrument, so a position in its leg too refuses. **Held by**
  `test_spine_engine.py::test_the_book_prices_every_position_at_its_net`, `test_position_scaling.py`.
- **Observations.** Where a type declares an observation table (`schema.OBSERVES`), the fixing in
  force fills its cell for rows dated on or before the base date - a forward-dated print is legal,
  and writing it would price a barrier observed on a day yet to come. An FX cell is the ratio of its
  non-base legs' prints. A plan is `strict` and refuses an unordered index; a read names it on the
  row. **Held by** `test_spine_engine.py::test_a_print_dated_after_the_base_date_is_left_standing`,
  `test_diary.py::test_a_cross_is_the_ratio_of_its_legs_prints_and_waits_for_both`.
- **Collateral balances**, for a run that reads one: [Collateral](#collateral).

## The diary, reconcile and the close {#the-diary}

`spine.diary` runs the `Diary` calculation, reading what it bound through `utils.walk_schedules`
(the binding's own walk) and each amount through `pricing.fixed_payments` (the pricer's own line),
so the two cannot disagree. An amount resets will fix reads `amount: null`, never 0.0, a
settlement file saying zero being a wrong payment. What a deal settles without a schedule is
declared on its fields (`settles=schema.Cash(...)`); monitoring days are never payments. A
cash-settled option waits for its expiry fixing and its payoff; an unreadable deal is an
`unreadable` row. A row's key, `cashflow_key(instrument, leg, kind, date)`, uses the date, since
positions renumber as paid rows drop. **Held by**
`test_diary.py::test_every_payment_the_diary_announces_is_the_run_s_own_cashflow`,
`::test_a_floating_amount_is_null_and_the_exporter_refuses_it_by_name`,
`::test_an_option_waits_for_its_fixing_and_its_settlement`,
`::test_a_derived_key_is_the_same_key_thirty_days_later`.

The book file keeps its content and gains `Spine: {lsn, head, hydrated_at}`, a sibling of `Calc`,
so the pin cannot move a plan. A booking appends first and writes the file second. `GET
/book/reconcile` folds at the head, where a lost write lives, and names by instrument address a
trade the file lost, a deal nobody booked, a differing clip count and paper that is not its declared
terms - a reading, never a refusal. **Held by**
`test_spine_engine.py::test_the_pin_cannot_move_the_plan_hash`,
`::test_a_fill_the_file_never_took_is_what_reconcile_is_for`.

A close is legal when nothing due by its day waits: a fixing nobody printed, a payment nobody
settled, an expiry nobody elected, or any `unreadable` row. `GET /book/close/check?date=` answers
that, and `POST /book/close` refuses an illegal day with nothing appended; a second close for a day
supersedes the first. `POST /book/settlements` takes only `due_before`, a file being struck for a
day, on the market the `tiers` policy designates, so pointing it elsewhere is unrepresentable.
**Held by** `test_diary.py::test_the_close_check_is_the_catch_up_rule`,
`::test_a_close_is_declared_only_on_a_day_the_check_calls_legal`,
`::test_the_settlement_file_is_struck_on_the_market_the_record_designates`.

## The record's readers on the desk {#readers}

`GET /book/activity` is the strip: with `?since=` the oldest rows after it, the cursor the last
delivered, so a capped walk loses nothing; it reads no book file. The web banner
(`web/src/spine.ts`) follows reconcile's lists, never the behind counts, because every write
re-pins the file at the head, a delete included. `GET /book/worklist` lists what waits on the
asking seat, each row keyed by what its clearing fact is filed against. **Held by**
`test_spine_engine.py::test_a_page_walks_the_record_forward_one_event_at_a_time`,
`::test_what_the_file_lost_is_drift_under_a_pin_that_has_seen_every_position`,
`test_pnl.py::test_the_worklist_names_what_waits_and_each_row_leaves_when_its_fact_lands`.

## Policies {#policies}

A policy is canonicalised, so one policy is one blob however spelled, and parsed at the
declaration, so one nobody could read never lands; the last declaration stands (`policy.in_force`).

| name | shape | absent means |
| --- | --- | --- |
| `capabilities` | `{grants, read}`, [above](#identity) | no enforcement |
| `tolerance` | `{"tolerances": {class: epsilon}}`, absolute, per top-level result class; a change of shape departs at any epsilon | nothing is pinned |
| `firmness` | `{"pillar_seconds": s}`, how old a quote's board may have been | nothing refused on staleness |
| `fixings` | `{"sources": {index: [source, ...]}}`, ordered | no index resolves |
| `tiers` | `{"tiers": [...], "designations": {process: market}}`, [below](#tiers) | fills land unsigned |

**Held by** `test_spine_verbs.py::test_a_policy_document_is_closed_at_the_field_level_and_refuses_where_it_is_declared`,
`::test_the_tolerance_comparison_is_structural_first_and_numeric_second`.

## Tiers and tickets {#tiers}

`spine.ticket(plan_hash, trade)` hashes the plan the book has once a trade lands with the trade's
fields as filed, so a retried fill or restrike coalesces and a changed plan needs a new approval.
The first tier whose `scope` covers the ticket and whose every declared check passes applies.
`max_notional` is `{amount, currency}` against notionals the caller states, so no check reads a
market and an unstated currency fails that tier; `max_tenor_years` reads the latest day the type
declares (`schema.tenor_fields`); `market` requires the quote's values to stand under that name. A
tier is automatic - the hub's approval in the writer's voice - unless it declares `four_eyes`;
staleness is `firmness`'s alone. **Held by** `test_spine_tiers.py`.

A booking is routed in one act (`spine.routed`): `SpineLog.admits` asks vocabulary and scope
writing nothing, then the ticket, an acceptance's `quote_filed`, the route, an automatic approval,
the event - so every refusal precedes the hub's signature, and only a crash between approval and
event leaves one alone, until retried. Under four eyes the trade lands pending (it prices and
settles; status is workflow); a ticket no tier admits writes nothing; nothing rejects automatically.
A currency the book cannot value is left off the ticket, so its cap fails rather than break the act.
**Held by**
`test_spine_engine.py::test_every_booking_is_a_ticket_the_tiers_route`,
`::test_every_refusal_a_booking_meets_comes_before_the_hub_signs`,
`::test_a_four_eyes_acceptance_books_pending_and_reads_the_verdict_that_stands`.

The latest verdict stands, none being withdrawn; under four eyes the booker's own are not read.
`POST /book/approve` and `/reject` file at the deepest node holding the ticket's positions, read off
the record, so an approver cannot sign another node's ticket. A rejected trade stands: it happened.
A trade sent again under its key is answered `{written: false, booked}`, other fields refused.
**Held by**
`test_spine_tiers.py::test_the_latest_verdict_stands_and_four_eyes_reads_who_filed_it`,
`test_spine_engine.py::test_a_verdict_is_filed_where_its_ticket_books`,
`::test_a_trade_sent_again_is_answered_where_it_stands`.

## Quotes and acceptance {#acceptance}

A quote records nothing; the record cares about the price a client takes. Under a home the pending
file ([Structures](structures.md#the-quote-lifecycle)) adds `pinned`: the book's two hashes and its
values, taken before the live spot lands; the board's age at `quoted_at`; and `plan`, the book as
acceptance leaves it. `values_hash` drops every clock (`schema.without_clocks`), so a board re-read
at the same prices is one market, and the age is the oldest of those clocks. **Held by**
`test_spine_engine.py::test_a_quote_records_nothing_and_files_what_the_acceptance_will`,
`::test_a_quote_pins_the_book_before_the_live_spot_lands_on_its_copy`.

`POST /book/quote` checks, in order and before anything appends: `firm_seconds`, the desk's promise;
the plan, refused where the book moved under the solve; the pillar age past `pillar_seconds`; and
the plan acceptance leaves, refused where it is not the one pinned (an edited file). The market is
reported as `{pinned, current, moved}`, never refused, the desk's window bounding it. Then one act:
`quote_filed` under the acceptor, the tier step, one unit of the mirror filled under the quote id,
the file. A quote filed before a tier refuses stands; the client took the price. **Held by**
`test_spine_engine.py::test_a_booking_between_the_quote_and_the_acceptance_refuses_with_nothing_appended`,
`::test_an_edited_pending_file_no_longer_says_what_was_quoted`,
`::test_a_moved_market_between_the_quote_and_the_acceptance_books_and_says_so`.

!!! warning "An edit that appends before it writes takes `Book.transact`"
    The acceptance, and `/book/deals` under a home, run under the book lock in one pass, because
    `Book.mutate` redoes a losing edit and a redone append would leave a fill for a write that lost.
    A write behind the service takes no lock and is a divergence reconcile names. **Held by**
    `test_spine_engine.py::test_a_write_racing_the_acceptance_leaves_no_half_act_in_either_order`,
    `::test_a_booking_raced_by_a_lockless_write_files_its_fill_exactly_once`.

## Markets, closes and marks {#markets}

Officialness belongs to a name, moved under `mark`; `private/<subject>/<name>` is one seat's board,
which no other subject may declare or close. `spine.resolve_market` answers a name's latest
declaration or close, refusing an undeclared name rather than fall back on the loaded market. A
market stands on its latest close by day, so a past day restated never unseats a later one.
**Held by** `test_spine_engine.py::test_a_market_resolves_by_name_and_a_private_one_resolves_for_its_owner`,
`test_spine_verbs.py::test_a_private_market_is_declared_by_the_seat_its_name_names`,
`test_pnl.py::test_a_past_day_restated_after_today_s_close_restates_that_day_alone`.

`POST /book/marks` values one unit of each instrument the book holds or traded since its last marks,
on the close standing under the market designated for `pnl`, as a standing run named
`marks:<book>@<cut>`. A day behind the last marked refuses - marks run forward, so a struck day never
moves - and the marks close the day where they read the book, so a ticket booked while the run waits
is the next day's. **Held by** `test_pnl.py::test_a_ticket_booked_while_the_marks_wait_is_the_next_day_s_and_marks_run_forward`.

## Queue admission {#admission}

`ComputeExecutor.submit` asks before anything runs: `mark` over its book for a standing run,
`validate` over the book or a node under it otherwise; a refusal files a denial and queues nothing.
Poll paths (a read's diary, the tick) run under `spine.hub()`, `DV_SPINE_ACTOR`, which a ticking
desk must grant `validate`. Admission precedes the diary cache, so a settlement file never depends
on who asked first. **Held by**
`test_spine_engine.py::test_under_a_document_the_queue_asks_before_the_executor_does`,
`::test_the_poll_paths_own_jobs_are_admitted_under_the_deployments_seat`.

## The replica and the doorbell {#replica}

`SpineLog.accept` is a replica's whole write path: the twelve fields, the next position, the link,
a recomputing event hash. It skips the vocabulary, so a replica of a newer hub still chains, and
authorisation, the hub's question. There are no peer writes and no failover. `GET /spine/frames`
serves ciphertext; `/spine/blobs/{digest}` serves a `read` seat. `DV_Spine follow` pulls until a
page is empty, `--once` only to the first reported head since a busy hub never yields one; it
verifies the whole chain once and each landed range after. **Held by**
`test_spine_doorbell.py::test_a_replica_chains_an_event_type_it_has_never_heard_of`,
`::test_a_one_shot_sync_returns_against_a_hub_that_never_stops_writing`.

`GET /spine/doorbell` is an event stream [@whatwg-sse] of `{lsn, head}` per durable append. A beat
is a notification, never a delivery: the pull asks from where the replica stands, so a dropped beat
is covered by the next. Its generator is async, so an open stream holds no shared worker thread. **Held by** `test_spine_doorbell.py::test_a_dropped_duplicated_or_reordered_doorbell_converges_every_replica`,
`::test_fifty_open_doorbells_park_no_worker_thread_and_leave_no_listener`.

## Where a position sits {#where-a-position-sits}

The `document` verb declares the paper: `entity_declared`, and `agreement_declared` whose terms are
the netting set its positions compile into, judged by the engine first. A fill's `quantity` is the
signed change in units; its optional `agreement`, `portfolio` and `price` say where and at what. One
instrument under two agreements is two positions, credit exposure being per agreement; in two
portfolios, two. The first booking or quote under a declared agreement splices its set from the
declared terms. **Held by** `test_spine_paper.py::test_a_position_is_keyed_where_it_sits`,
`test_spine_engine.py::test_the_first_booking_under_a_declared_agreement_brings_its_set`.

`portfolio_declared` is judged at its parent. Once any node of a book stands, a fill books into the
book or a declared node. `GET /book/positions` answers each standing position beside the file's
nodes, an expired one until it settles. A seat with grants at nodes reads only the rows under them
(`spine.visible`), totals summed over those - presentation while one class key opens every body.
**Held by** `test_spine_engine.py::test_a_declared_tree_is_where_a_book_books_and_a_restrike_is_judged`,
`::test_a_seat_reads_the_rows_under_the_nodes_it_acts_at`.

## Cash, cost and P&L {#pnl}

A `settled` transition may carry a signed `amount` (received positive), `asset`, `kind` and the
settlement system's `reference`: a `payment` settles its diary row, a `fee` sits on its instrument,
`collateral` and `margin` move a balance under a declared agreement. The reference is the movement,
so a correction under it restates the amount. `costs` keeps the open `basis` at average cost and
what reductions `realised`; a fill with no price leaves the basis unknown, never zero. **Held by**
`test_spine_verbs.py::test_a_settlement_that_moved_money_says_how_much_and_the_cash_fold_keeps_it`,
`test_spine_projections.py::test_a_position_costs_its_average_and_a_reduction_realises_against_it`.

`GET /book/pnl` reads, per position between two days' marks or to now,
`value_end - value_start + premiums + payments + fees`:

- the close is the end of the day, so a value is quantity times unit mark less what the unit paid
  that day, the engine valuing a payment into its due day's marks;
- a payment is the diary's amount times what the position held at the end of its due day, and a
  settlement of another amount is a break, never an unknown;
- a filing counts in the window it was filed in, as the record stood that day, converted at that
  day's close, so a struck day never moves and every figure but the new-deal split is additive;
- a position whose last day falls in the window closes there, its basis released;
- what nobody can know is named under `unknown`, and a total over it is null.

`explain` splits it into carry, market (factor moves times the start's sensitivities, per quote) and
residual. **Held by**
`test_pnl.py::test_the_book_makes_its_marks_move_plus_its_cash_and_a_month_is_its_days`,
`::test_a_payment_on_a_marked_day_is_that_day_s_cash_and_out_of_its_marks`,
`::test_money_nobody_can_place_is_named`.

## Collateral {#collateral}

`projections.CSA` reads a set's terms as the engine does - each `CreditSupportList` at its first
value, `Haircut_Posted` on either side - and `required` is the engine's `At`. `call` is that less
the collateral held after haircuts, moving only strictly past the minimum transfer on its side, as
`scan_collateral_balance` does; margin is a separate balance. Being pure, the oracle replays it.
`GET /book/collateral` answers each agreement on a marked close, its exposure the P&L's value of its
positions with the day's payment added back where the set holds it (`Exclude_Paid_Today` off); an
unknown mark makes the call null, never zero. **Held by**
`test_collateral.py::test_the_support_required_is_the_engine_s_own_at`,
`::test_a_call_clears_its_minimum_strictly_on_the_side_it_falls`,
`::test_a_mark_nobody_has_is_named_and_calls_nothing`.

`compiled_job` writes a collateralised set's `Opening_Balance` - collateral held by the base date,
after haircuts, in its `Balance_Currency` - for a run with a collateral recursion; a base valuation
has none, so a settlement moves neither its plan nor its cache. An asset without a spot refuses
every valuing compile rather than count as zero. **Held by**
`test_collateral.py::test_the_record_s_balance_opens_the_engine_s_recursion_and_the_call_is_its_transfer`.
Not carried: interest, `Haircut_Received`, securities as collateral, rating-dependent support,
and the hedge simulation's opening state.

## The oracle {#the-oracle}

`oracle.py` reads a home as a replica and re-runs the writer's decisions through its own functions,
so an answer is a property of the bytes; the script of what was asked, the diary's rows and the P&L
are handed in. Each invariant answers `{held, evidence}`, `held` null - never a pass - where the
question could not be put.

| invariant | holds where |
| --- | --- |
| 1 copies agree | two homes share head hash and rows at the shallower head |
| 2 nothing outside its seat | every frame's seat held its verb when it landed; the writer's voice said only its own |
| 3 every refusal is a denial | denials equal the refusals the script says reached the writer |
| 4 an amended plan is a new approval | each ticket is one trade's; a restrike has its own |
| 5 closes superseded, never edited | no close the vocabulary would refuse |
| 6 every number's replay tuple | attestations are exactly the standing runs asked for |
| 7 the diary equals the filings | each settlement names a diary row or clip |
| 8 no delete | positions dense, cited blobs present |
| 9 duplicates coalesce | each tag on one line |
| 10 nothing outside its scope | each node-judged frame held its verb there |
| 11 the P&L is additive | days sum to windows and portfolios to the book within the `pnl` tolerance |
| 12 cash reconciles | payments match the diary, a row is paid once |
| 13 every close marked | each `pnl` close has marks while the book held anything |
| 14 the call is the formula | each call read is `CSA.call` over the record |

A keyless copy answers 1, 8, 9 and half of 2 and names the rest not assessed. **Held by**
`test_spine_oracle.py`. `gates/spine_game/` plays a bank through the MCP binding over three closes
beside an adversary and scripted faults; its roles are callables, so a model host can play them.
**Held by** `test_spine_game.py`.

## Writing a new gate

- Doctor data, not code: the fault as bytes on disk or a fact through the writer.
- Assert absence on the head, not a filtered count; age by a declared zero window, not a sleep.
- A new fact type needs a validator, `EVENT_VERB`, `BLOB_FIELDS` and `SUMMARIES` entries; the
  closure gates stay red until all exist.

## Defaults {#defaults}

| setting | default | where |
| --- | --- | --- |
| Segment roll | 64 MiB | `log.SEGMENT_LIMIT` |
| Service bind, port; CORS | `127.0.0.1`, 8000; any origin | `DV_Service --bind`, `--port`, `--origin` |
| ID-token leeway; algorithms | 60 s; RS256, ES256 | `identity.LEEWAY_SECONDS`, `ALGORITHMS` |
| Replica page; read timeout | 500 frames; 30 s | `service.REPLICA_PAGE`, `replica.PAGE`, `replica.TIMEOUT` |
| Doorbell heartbeat; fallback interval | 15 s; 2 s | `service.DOORBELL_SECONDS`, `replica.INTERVAL` |
| Activity page; worklist rows | 200 | `book_activity(limit=)`, `service.WORKLIST_ROWS` |
| Quote firmness | 600 s | `Quote Policy.firm_seconds` |
| Pillar staleness, tolerance, capabilities | none declared | `policy.py`, `capability.py` |

## Open {#open}

Unbuilt parts of the design: a job's files are named by path, not declared in the record
([the engine](roadmap.md#the-engine)); the deal tree is the book file with nets written on, not
composed from the fold; and a print is not yet one fact feeding both a calibration's archive and a
deal's fixings ([designed, not built](roadmap.md#designed-not-built)). [The engine](roadmap.md#the-engine)
lists the record's other open rows - the log open's full scan, the rotation and replay limits, the
oracle's blind spots, reads outside admission, verbs under no seat, settlements naming no agreement,
unindexed fixings, legacy trades. `advancing` pairs are process-wide, so one home per service process
is the deployed shape ([tidy-ups](roadmap.md#tidy-ups)); one `settle` verb is a
[desk decision](roadmap.md#decisions-waiting-on-the-desk).
