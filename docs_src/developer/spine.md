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
- **Blobs.** Whatever has content - terms, a market's values, a file a job names, a job, a result -
  is stored once under its hash and cited by it.
- **Folds.** Every view is a fold of the facts up to a point in the record: a position, a P&L, what
  stands - the latest declaration of a name there. Nothing derived is stored as truth, and the point
  is the citation, so nothing is priced against an older declaration by choice.
- **The engine.** A function of a job document and its market: the record compiles the document
  ([`compiled_job`](#the-plan)), the engine prices it, the result is filed by address.

[The Spine in Motion](../animations/spine_in_motion.html) animates the whole of it, from the hub
minting its home to the folds producing the day's P&L: play it here or open it full-page.

<iframe src="../../animations/spine_in_motion.html" title="The Spine in Motion" loading="lazy"
        style="width:100%;height:860px;border:1px solid var(--md-default-fg-color--lightest);border-radius:6px"></iframe>

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
A `SpineRefusal` leaves as `SpineRefused`, a `ValueError`, so a 422 carries the library's sentence
and its kind (`SpineRefused.code`): `not_granted` for a `CapabilityDenied`, `no_actor` for an act
no seat signs, `refused` otherwise, which a refusal the service respells keeps.

### The CLI {#the-cli}

`DV_Spine` works on `--home`, else `DV_SPINE_HOME`, else `~/.derivus_spine`; a refusal exits 1.

| verb | what it does |
| --- | --- |
| `init` | mint a home: keys, the genesis facts, the first checkpoint |
| `verify [--chain-only]` | `verify_home`, entitled or keyless |
| `checkpoint`, `status` | sign the head; print it, and whether bodies open |
| `follow <url> [--once] [--verify] [--interval] [--blobs] [--actor]` | pull a hub's frames into a replica |
| `oracle [--script] [--against] [--diary] [--pnl]` | the fifteen invariants; exit 1 where one failed |
| `seed [--at] [--out]` | mint the folds' seeds at an official close |
| `enroll`, `grant`, `rewrap`, `name`, `whoami` | seats, capabilities, key wraps, display names, token check |
| `declare`, `policy`, `portfolio` | declare a policy, report those in force, declare a node |
| `file --name --kind --path` | declare a market data file or a calendar a job names, by its bytes |

## The truth layer is files {#the-truth-layer}

```
DV_SPINE_HOME/
  .writer.lock                            the writer's claim
  log/segment-00000001.jsonl              frames, one fsync per append, rolled past 64 MiB
  blobs/ab/cd/<sha256>                    content-addressed, write-once (blobs/tmp/ is scratch)
  keys/                                   blind.key, class_firm.key, checkpoint_signing.key, checkpoint_verify.key
  keys/seats/<sha256(subject)[:16]>.key   a seat's private key where the hub minted it
  names.json                              display names, outside the log
  index.json                              the head, offsets and tags at the last checkpoint: derivable, deletable
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
and the stored values vector's address are one number, and so are a declared file's blob and the
digest a run reads it under in `Stats['Files']`.

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
unparseable line anywhere else the open reads was altered and raises `ChainBroken`, so the repair
cannot roll a log back past its checkpoints. Opens scan without the claim, so a reader opening
mid-append could truncate that append's bytes; no gate covers that interleaving. **Held by**
`test_spine.py::test_a_torn_final_line_is_truncated_and_the_next_append_chains_onto_the_head`,
`::test_a_terminated_line_that_will_not_parse_is_a_broken_chain_not_a_torn_tail`.

Where a checkpoint lands, the writer puts the head, the byte offset of every LSN, each segment's
first LSN and the tag index in `index.json` beside `log/` (`INDEX`), and an open restores them and
reads only the lines past them, paying for the frames since the last checkpoint, whose cadence is
the deployment's. The index is derived and verified like a [seed](#folds): its state must hash to
the stamp on its first line and its head must be the frame this log holds at the offset it names,
so one from another history, one left ahead of a log restored behind it, or one edited costs the
open its full scan and is never read. A line altered behind it is met not by an open but by
`verify_home`, which re-derives every line off the platter. **Held by**
`test_spine.py::test_a_line_altered_behind_the_index_is_refused_by_verification_not_by_the_open`,
`::test_an_index_of_another_history_behind_its_log_or_edited_is_rebuilt_never_trusted`,
`::test_an_open_from_the_index_answers_what_streaming_every_line_answers`.

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

`(plan_hash, values_hash, engine_version, seed, quasi_seed)`. The two hashes cover the files a job
names by what they hold - a market data file's sections and its calendar in the plan, the file's
values in `values_hash` - so the tuple pins their content, never their path, and the digests of
the files a run read ride beside it rather than in it ([below](#lanes)); a replay reads a declared
file's bytes off the store ([Markets](#markets)).
`seed` is the `Random_Seed` a job states, null where it states none, a base valuation, a SIMM and
an SA-CCR declaring none; `quasi_seed` is the `Quasi_Seed` every calculation declares, its default where the
job states none (`spine.replay_seeds`). Neither is in `plan_hash`. **Held by**
`test_spine_engine.py::test_a_standing_run_attests_at_birth_and_the_other_lanes_mint_nothing`.
`engine_version` is the hand-set `derivus/_version.py`; no commit, torch or CUDA version or device
is recorded; and `Deterministic_Kernels` defaults to `No` (`Yes` calls
`torch.use_deterministic_algorithms(True, warn_only=True)`, which warns rather than refuses).
Bit-equality on replay is to be expected only on one build, stack and device, and for a GPU
gradient only under `Yes`; elsewhere a tolerance policy decides.

## Identity, capabilities and custody {#identity}

**Identity.** `identity.verify_id_token` checks an OIDC ID token [@openid] against a JWKS
[@rfc7519] handed in as data, refusing `none` and every HMAC before a key is selected, so the
alg-confusion forgery is unrepresentable, and holding `azp` so a co-audienced client cannot replay
its token. It answers `sub`; display names live in `names.json`, outside the chain. **Held by**
`test_spine_identity.py`.

Naming a key set (`DV_SPINE_JWKS`, with `DV_SPINE_ISSUER` and `DV_SPINE_AUDIENCE`) makes every request but the doorbell
carry a token, its subject set in the request's context by an async dependency (a sync one would
set a thread-pool copy). `spine.reader` answers that subject, refusing another name, else the stated
name, for a read and an act alike; `spine.actor` reads it, else, with no capabilities document, `DV_SPINE_ACTOR`; under one, an unnamed act refuses
rather than file a name nobody chose. **Held by**
`test_spine_engine.py::test_a_bearer_token_names_the_seat_and_nothing_else_does`,
`::test_a_read_naming_a_seat_its_token_does_not_prove_is_refused`.

**Capabilities.** A canonical blob under the reserved name `capabilities`, each declaration a
complete replacement, folded at an LSN so "could X do Y in March" replays: `{"grants": [{subject,
verb, book}], "read": [{subject, class}], "verbs": {act: verb}}`. Nine verbs, none implying
another, each demanded by these acts under the vocabulary's map:

| verb | files |
| --- | --- |
| `validate` | nothing: work the queue admits |
| `book` | `fill`, `amendment`, `election`, `quote_filed` |
| `approve` | `approval`, `rejection` |
| `mark` | `market_declared`, `official_close_declared`, `fixing_observed`, `determination`, `file_declared`, `result_pinned` |
| `document` | `entity_declared`, `agreement_declared` |
| `settle` | `status_transition` paying: `settled` moving no money, a `payment`, a `premium` or a `fee` |
| `confirm` | `status_transition` matching a trade: `confirmed` |
| `collateral` | `status_transition` moving `collateral` or `margin` under an agreement |
| `admin` | `policy_declared`, `portfolio_declared`, `retention_declared`, `rehash_declared`, `checkpoint`, `seat_enrolled`, `key_wrapped` |

A type missing from the map demands `admin`, so a forgotten entry costs governance, not an open
write. A transition is one type and three acts, read off its body (`capability.act_of`): a
`confirmed` status, money held under paper by its kind, `collateral` or `margin`, else a payment.
So a seat granted `settle` alone is refused a confirmation with the denial naming `confirm`, and
so for each pair - the separation of duties, which no grant implying another would keep. The
`lifecycle` fold keeps the two apart as well: a subject's `confirmed` stands under
`confirmations` and every other status a payment or a premium moves under `transitions`, so
neither seat's filing replaces the other's. **Held by**
`test_spine_verbs.py::test_confirm_and_settle_states_never_overwrite_each_other`.

**What the service asks where nothing is filed.** Four writes change what the book prices or
pays and append no event, so the service asks their verb itself (`spine.entitled`): a delete
through `POST /book/deals` asks `book` at the deepest node holding the position, anywhere under
the book where the record holds none; `POST /paths` and each beat of the scheduled tick ask
`mark`, the beat of the hub's own seat filing as `terminal_tick`, the act `POST /book/bloomberg`
names; `POST /book/settlements` asks `settle` over the book.
`POST /book/close` asks, over the firm and before it compiles anything, the verb its
declaration demands under the map in force, `mark` unless a document moves the act. The close and the
settlement file read the diary as the hub compiles it (`hub_diary`), so neither seat needs
`validate`. A tick the hub's seat may not mark submits nothing: the cadence warns once a beat,
naming the seat that can grant it, and beats on. **Held by**
`test_spine_engine.py::test_a_delete_asks_book_at_the_node_holding_the_position`,
`::test_a_path_save_asks_mark_over_the_book`,
`::test_a_close_asks_mark_over_the_firm_before_its_verdict_and_never_validate`,
`::test_a_settlement_file_asks_settle_over_the_book_and_never_validate`,
`::test_a_tick_the_hub_may_not_mark_is_refused_and_the_cadence_lives`.

**What a refused seat is told.** A refusal speaks to someone at a desk. A seat without the grant
an act asks reads one sentence, whoever turns it away (`log.DENIED`): the seat, the verb, the
node, what did not happen, who puts it right (`log.UNGRANTED`, a seat holding `admin` over the
node) and, once it lands, the LSN its denial stands at (`log.RECORDED`). **Held by**
`test_spine_engine.py::test_under_a_document_the_queue_asks_before_the_executor_does`,
`test_service.py::test_the_market_is_the_marking_seats_on_a_recorded_desk`.

No sentence the service, the record or a quote's notes can say names a route, a query string, an
install line or an MCP tool: it says what did not happen and who puts it right - a seat holding a
verb, or whoever runs the desk. One naming a command line, a flag, a variable or a folder of the
home, or printing a path on the service's own disk (`SERVER_PATHS`), is listed with its reason
(`OPERATOR_SECOND`): no seat at a desk meets it. The words are one list,
`tests/fixtures/operator_words.json`, every MCP tool but the one a desk word names among them,
which the web's gate reads too, and no string or markup text a screen can show says any of them.
**Held by**
`test_service.py::test_no_sentence_a_desk_reads_names_a_route_and_an_operators_part_is_listed`,
`test_mcp.py::test_every_tool_is_registered_and_carries_its_contract`,
`web/scripts/words_check.mjs`.

`verbs` is the map in force, the verb each act demands. A declaration is stored with the
vocabulary's verb for every act it does not move, governance staying `admin`, so the map in force
at an LSN is the one of the day it was declared, and a vocabulary moving an act moves a home at its
next declaration; a document stored without a map reads the vocabulary's. Its one read is `capability.verb_for(event_type, body, document)`: the writer's
hook asks it under the document in force at the head, the oracle's seat invariant under the one in
force at each frame, so an old frame is judged by the map of its day, and the worklist narrows
each list by the verb its clearing act demands (`spine.demands`). `GET /book/seat` answers that
map as `demands` beside the nodes the seat holds each verb at, and a screen holds a control by the
service's verdict on the act it performs (`may`, read by `heldBack` in `web/src/state.ts`) at its
node, computing none of its own: a node the answer does not name is offered, the act judged when
it is sent. An
acceptance files a quote and its fill, so its control is held where either is (`acceptHeld`).
`may` answers the verdict itself, act -> node -> null or the sentence the act is refused with
there before its denial's LSN (`spine.may`): the writer's (`SpineLog.denied`) for each act the
map has a seat's verb do, the queue's (`spine.admission`) for each run it admits (`RUNS`), at
the book, each node declared or holding a position, and each node above one; and each write of
the market by its name (`MARKET_WRITES`, which `marking` reads) at the node it is judged at,
the book served (`written`), the firm keyed `*` where none is. `sees` is null where the seat is
shown the whole book (`spine.reaches` there), else the nodes whose positions it is shown.
**Held by** `test_spine_engine.py::test_a_seat_reads_what_it_may_do_and_where`,
`test_service.py::test_the_market_is_the_marking_seats_on_a_recorded_desk`,
`::test_a_seat_reads_at_each_node_whether_it_may_act_in_the_records_own_words`, `web/scripts/desk_check.mjs`,
`test_spine_verbs.py::test_the_back_office_is_three_seats_and_none_of_them_does_anothers_work`,
`test_spine_capability.py::test_every_verb_bearing_type_refuses_an_unscoped_actor_and_records_the_refusal`,
`::test_a_capabilities_document_that_cannot_be_evaluated_does_not_land`,
`test_spine_oracle.py::test_a_confirmation_under_a_seat_that_only_settles_is_named`,
`::test_a_confirmation_filed_under_the_map_of_its_day_reads_inside_its_seat`,
`test_pnl.py::test_the_worklist_names_what_waits_and_each_row_leaves_when_its_fact_lands`.

A grant's `book` is a node reaching every path below it: `BANK/FX` covers `BANK/FX/Options`, not
`BANK/FXO` or `BANK`. `capability.scope_of`, shared by writer and oracle, judges the firm's own
facts (policies, markets, prints, runs, the paper, custody) at the firm, which only `*` reaches; a
fill, restrike, election, quote or verdict at its `portfolio`; a declared node at its parent; else
the book. **Held by**
`test_spine_capability.py::test_a_grant_at_a_node_reaches_every_path_under_it_and_nothing_beside_it`,
`test_spine_oracle.py::test_the_firms_own_facts_are_judged_at_the_firm_whatever_their_envelope_names`.

An act on an instrument - a restrike, an election, a delete - is judged where its positions sit:
the deepest node holding every position in it under the book (`capability.judged_at`), read off
the positions fold and never stated by a caller. `spine.amend` and `spine.apply_lifecycle` file
that node as the body's `portfolio`, and the oracle names a frame judged narrower than what it
moved. An election of an instrument the record holds no position in, and one whose body names no
node, is judged at its book. Each position `GET /book/positions` answers names that node
(`judged_at`, `spine.standing`), null where nothing of its terms is held; the web client holds a
trade's amend at it, and each position's unwind - a fill - at the position's own `portfolio`.
**Held by**
`test_spine_engine.py::test_an_election_is_judged_where_its_position_sits`,
`::test_a_declared_tree_is_where_a_book_books_and_a_restrike_is_judged`,
`test_spine_oracle.py::test_an_election_is_judged_at_the_node_its_body_names_and_else_at_its_book`,
`::test_a_restrike_judged_narrower_than_what_it_moved_is_named`.

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
a key to every body, and the map, the firm's, are unchanged (`capability.declarable`). **Held by**
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
with the replay tuple and the job, values and result, and `files`, the sha256 of each file its job
named as the load read it - the run's `Stats['Files']`, null where it named none - beside the tuple
and not in it, the two hashes already covering what the files hold; a pin files its claim's the
same way. Content addressing dedupes numbers, not standing: a standing run whose numbers exist
still attests, one coalescing onto a run in flight is promoted in `ComputeExecutor.submit`, and one
whose attestation is refused fails rather than serve an unbacked citation. It posts its document,
a `plan_id` naming only a parse. **Held by**
`test_spine_engine.py::test_a_synthetic_tick_sequence_mints_nothing_and_the_absence_is_asserted`,
`::test_a_standing_run_that_coalesces_onto_one_in_flight_is_attested_when_it_lands`,
`::test_an_attestation_the_record_refuses_fails_the_run_it_was_for`,
`::test_a_run_says_the_files_it_read_and_a_replay_reads_the_bytes_that_stood`.

`pin_result` promotes a tuple this hub did not witness through an injected `executor`, the record
being unable to import the engine: a cache hit on a matching `run_completed`, bit-equality, or
`policy.compare`. The fold never reads `result_pinned`, so a promotion is never evidence for the
next, and the spine admits no tolerance of its own: a home declaring none pins nothing. **Held by**
`test_spine_verbs.py::test_a_pin_never_bootstraps_itself_off_another_pin`,
`::test_a_home_that_declared_no_tolerance_policy_pins_nothing_at_all`.

## Folds, seeds and fixings {#folds}

A projector is `initial`, `apply`, `rows` and the types it `reads`; `fold` opens only those bodies,
and rows are sorted canonical JSON, so a golden replay is a byte comparison. The fifteen:
`positions` (net per instrument × agreement × portfolio, clips under each ticket), `costs`, `cash`,
`lifecycle` (prints under `(index, date, source)`, elections, statuses), `markets`, `files`,
`attestations`, `decisions`, `quotes`, `denials`, `blotter`, `entities`, `agreements`,
`portfolios`, and `activity`, the strip, which opens no body so a keyless replica shows every LSN.

A keyed row stands by the as-of key `(effective_time, lsn)`, so a backdated restatement does not win
by arriving last, and what it beat stays readable; a second administrator's print is a second row. A
knock is read off the fold (`projections.knocked`), never filed. **Held by**
`test_spine_projections.py::test_every_projector_replays_to_its_committed_golden`,
`::test_a_republished_print_supersedes_by_as_of_key_and_the_first_is_still_read`.

A keyed declaration - a market name, a close under its market and day, a file, an entity, an
agreement, a quote, a print, a subject's state, a movement under its reference, a policy document
(the capabilities document and the escrow key among them, read off the `decisions` fold) and a
seat's key - takes effect when it returns to earlier content. `verbs.restated` files it over what
its own reader has standing under the key: the content in force declared again restates that body,
which the writer coalesces, and anything else names the declaration it replaced (`supersedes`). A
file, an entity and an agreement are restated so on every declaration. The rest are filed by
`verbs.keyed` as offered, at the cost of one tag lookup, and restated only where the log already
holds that very fact (`SpineLog.held`) - the one case the writer would coalesce a return onto an
older LSN - so the fold that finds what stands runs only then. With no effective time, a late retry
of an earlier content arriving after a later one is such a return. An amendment is filed the same
way over the fact that last moved its terms (`verbs.moved`), so terms restruck, restruck back and
restruck again are three facts and the position follows each. **Held by**
`test_spine_verbs.py::test_terms_amended_back_and_amended_again_move_the_position_each_time`,
`test_spine_engine.py::test_a_trade_restruck_back_and_restruck_again_stands_where_the_file_has_it`,
`test_spine_verbs.py::test_a_keyed_declaration_returned_to_earlier_content_is_a_new_fact_in_force`,
`test_spine_capability.py::test_an_earlier_document_restored_by_grant_revokes_what_the_one_it_replaced_granted`,
`test_spine_custody.py::test_a_seat_key_or_an_escrow_key_restored_is_the_key_in_force_and_its_wrap_opens`,
`test_spine_projections.py::test_restatements_filed_naming_nothing_they_replaced_still_read`.

**Seeds.** A seed is a fold's `(lsn, state)` written at an official close, where the desk agrees
what the day was, carrying the close's event hash, projector, version and state hash; one from
another home or version, torn, or edited refuses by name. `DV_Spine seed` mints all but the strip,
whose state is its history; readers start at the newest seed in `DV_SPINE_SEEDS` (else `seeds/`). A
projector bumps its `version` when rows change, so an old seed refuses. **Held by**
`test_spine_projections.py::test_a_seeded_fold_is_the_from_genesis_fold_byte_for_byte`,
`::test_a_projector_one_version_on_folds_the_same_bodies_and_a_seed_is_verified_not_trusted`.

`spine.advancing` advances each projector's `(lsn, state)` pair, kept by the served book
(`service.Book.folds`), so a booking pays for the events since the last, not the whole history;
pairs are keyed on the genesis hash, so a re-minted home is another record, and a copied home is
served with its own. **Held by**
`test_spine_engine.py::test_a_home_re_minted_where_the_last_one_stood_is_a_different_record`,
`::test_a_home_copied_from_another_is_its_own_record`.

`fixings_at` resolves prints by the source order the `fixings` policy declares per index, the first
source holding a print winning. With no policy nothing resolves; under one, an index asked for and
not ordered refuses - a fixing nobody vouched for is not one a plan may use - and one not asked for
is left alone. **Held by** `test_spine_projections.py::test_a_fixing_with_no_declared_source_refuses_and_the_order_is_the_authority`.

## The plan as a fold {#the-plan}

`spine.compiled_job(document, lsn, strict, runs)` is the job a plan hashes: the document plus what
the record holds. `/execute`, `/prepare`, the diary, the marks and every valuing read compile first,
so a plan named once is the plan that runs and an auditor recompiles from what was printed.
**Held by** `test_spine_engine.py::test_the_plan_compiles_from_the_fold_rather_than_from_what_was_typed`.

Every write of a compile lands inside the deal tree, bar the book's credit run's calculation, which
it replaces rather than edits, so it copies `Calc.Deals` alone: the compiled job shares its market
with the document it was handed, which it leaves as read, and a caller that edits the job copies
the level it writes at, as `service.xva_document` and `service.consolidated_risk` do. **Held by**
`test_spine_engine.py::test_a_compile_copies_the_deal_tree_alone_and_loads_a_market_only_for_collateral_held`,
`test_service.py::test_a_set_s_xva_job_shares_the_book_s_market_and_writes_none_of_the_book`,
`::test_the_consolidated_risk_writes_nothing_on_the_document_it_values`.

- **Nets.** Each node the record holds a position in is written at its net over the agreement's
  set by `spine.scaled`: declared `sized` fields multiplied, a negative net taken as the mirror (a
  declared `side` flipped, else sizes signed). A swaption's underlying is its terms, never flipped.
  A zero net or a repeated node is ignored, not removed, so resolved paths stay valid. A block
  stating none of its amounts refuses rather than price one unit, and a structure held whole is one
  instrument, so a position in its leg too refuses. The risk answer shares each node's mark among
  the positions on it (`per_position`, `marked_positions`): `unit` the mark per unit of the terms
  as written, the P&L's own share of an amount among positions netting to it (`PnL.per_unit`),
  and `value` that times the position's quantity, both null at a net of nothing. **Held by**
  `test_spine_engine.py::test_the_book_prices_every_position_at_its_net`, `test_position_scaling.py`,
  `test_spine_engine.py::test_a_position_reads_its_own_share_of_its_deals_mark`.
- **Observations.** Where a type declares an observation table (`schema.OBSERVES`), the fixing in
  force fills its cell for rows dated on or before the base date - a forward-dated print is legal,
  and writing it would price a barrier observed on a day yet to come. An FX cell is the ratio of its
  non-base legs' prints. A plan is `strict` and refuses an unordered index; a read names it on the
  row. **Held by** `test_spine_engine.py::test_a_print_dated_after_the_base_date_is_left_standing`,
  `test_diary.py::test_a_cross_is_the_ratio_of_its_legs_prints_and_waits_for_both`.
- **Paper.** A netting set materialising a declared agreement, its `Reference` the agreement's id,
  whose paper (`spine.paper`) is not the terms standing at the compile's LSN is written those terms,
  its children, balance and holdings' amounts kept, so a correction prices from the event that
  recorded it and a job compiled before it prices the earlier paper. The file keeps its copy, which
  reconcile names. **Held by**
  `test_spine_engine.py::test_a_corrected_agreement_is_the_paper_every_run_prices_from_the_event_that_recorded_it`.
- **Collateral balances**, for a run that reads one: [Collateral](#collateral).
- **The calculation**, for the served book's own credit run (`official`): the `calculation`
  policy's block over the document's own, the one write outside the deal tree
  ([the official calculation](#official-calculation)).

## The diary, reconcile and the close {#the-diary}

`spine.diary` runs the `Diary` calculation, reading what it bound through `utils.walk_schedules`
(the binding's own walk) and each amount through `pricing.fixed_payments` (the pricer's own line),
so the two cannot disagree. An amount resets will fix reads `amount: null`, never 0.0, a
settlement file saying zero being a wrong payment. What a deal settles without a schedule is
declared on its fields (`settles=schema.Cash(...)`); monitoring days are never payments. A
cash-settled option waits for its expiry fixing and its payoff; a cashflow list held to an
`Investment_Horizon` pays and fixes to that day and, where it pays after it, is closed out there
on its `Settlement` leg at a null amount the market decides; an unreadable deal is an
`unreadable` row. A deal settling in its children's place (`Deal.settles_children`, a price swap)
announces its own settlement at the amount the market decides, and its children only the fixings
they need to its last day, no payment and nothing after. A row carries its deal's `reference`,
its `deal_path` and the `netting_set` it
sits in, and takes the instrument booked at that path (`service.booked_instruments`), references
not being unique in a book: two live deals under one reference are two instruments, each row keyed
under its own, and a deal its history resolved (`became`) is read at its node's path. A row's key,
`cashflow_key(instrument, leg, kind, date)`, uses the date, since positions renumber as paid rows
drop. The record's answer (`service.answered`) adds the `agreement` the row's set is where the
record declares it - what a settlement of the row names - so one instrument under two agreements
is two rows under one key, each settled by a payment naming its own agreement once one against
the key names any.
**Held by** `test_diary.py::test_every_payment_the_diary_announces_is_the_run_s_own_cashflow`,
`::test_a_floating_amount_is_null_and_the_exporter_refuses_it_by_name`,
`::test_an_option_waits_for_its_fixing_and_its_settlement`,
`::test_a_derived_key_is_the_same_key_thirty_days_later`,
`::test_two_live_deals_under_one_reference_are_two_instruments_each_settled_on_its_own_key`,
`::test_a_barrier_its_history_knocked_in_is_read_as_the_vanilla_it_became`,
`test_pnl.py::test_a_payment_is_shared_within_the_agreement_it_names`,
`test_bond_holding_json.py::test_the_diary_of_a_holding_pays_to_its_horizon_and_settles_there_and_nothing_after`,
`test_price_swap_json.py::test_the_diary_announces_one_settlement_on_the_maturity_and_of_the_children_their_fixings_to_it`.

**Premiums.** The book's diary also announces what its fills paid or received
(`service.announced`, `spine.premiums`): one `premium` row per fill of the book whose price moves
money, its consideration from the bank's side - minus the quantity times the price - in the fill's
own `currency`, else the book's reporting currency, due on the fill's `value_date`, keyed
`cashflow_key(instrument, execution_reference, 'premium', value_date)`. A priced fill stating no
value date is due on no day, its `reason` naming it, so no close waits on it; a fill with no price
or a price of nothing announces none. The worklist lists a premium due by the book's day among its
payments, the close waits on it as on a payment, the settlement file instructs it beside the day's
payments, and a `premium` settlement against its key clears it. The P&L never reads these rows -
its diary is the marks jobs' compile (`window_payments`) - so the consideration is counted once,
under `premiums` at the fill, and a premium settlement is no payment it reads. **Held by**
`test_diary.py::test_a_premium_is_due_on_its_value_date_until_a_premium_settles_it_into_cash`,
`::test_a_premium_with_no_value_date_is_due_on_no_day_and_one_with_no_price_is_none`,
`test_pnl.py::test_a_premium_announced_and_settled_moves_no_figure_of_the_p_and_l`,
`test_mcp.py::test_a_booking_s_value_date_reaches_the_record_and_the_diary_announces_its_premium`.

The book file keeps its content and gains `Spine: {lsn, head, hydrated_at}`, a sibling of `Calc`,
so the pin cannot move a plan. A booking appends first and writes the file second. `GET
/book/reconcile` folds at the head, where a lost write lives, the positions of the book the file
names (`spine.positions_of`, the reader the compile and the marks read too) - another book's fill
is none of its drift, nor of its `positions_behind` - and compares the two where the book
prices: a position is its instrument under its agreement, priced at the record's net on the first
node of those terms under that agreement's set, so how many nodes carry it is no disagreement and a
partial unwind filed against its node is none, nor a node of terms a restrike moved, which stand at
nothing as a position closed out does (`service.record_positions` keys what `spine.nets` keys).
It names a position no node carries, a node no
position carries - each by its set, the trade's reference (off the record's copy of the terms
where the file lost them) and the client's name - and paper that is not its declared terms; a
reading, never a refusal. **Held by**
`test_spine_engine.py::test_the_pin_cannot_move_the_plan_hash`,
`::test_a_fill_the_file_never_took_is_what_reconcile_is_for`,
`::test_another_book_on_the_record_is_none_of_this_books_drift`,
`::test_terms_a_restrike_moved_reconcile_as_the_book_prices_them`,
`::test_a_partial_unwind_leaves_the_file_and_the_record_agreeing`,
`test_service.py::test_a_booking_into_one_set_leaves_another_set_s_xva_run_where_it_was`.

A close is legal when nothing due by its day waits: a fixing nobody printed, a payment or a
premium nobody settled, an expiry nobody elected, or any `unreadable` row. A deal the compile left out is judged
as what its history made it, so a barrier its prints knocked out waits on its rebate and on nothing
after. `GET /book/close/check?date=` answers that, and `POST /book/close` refuses an illegal day
with nothing appended; a second close for a day supersedes the first. A close naming no market is
declared on the one the `tiers` policy designates for `pnl`, which the marks and the P&L read - a
desk has one official close - and on `official` where it designates none. `POST /book/settlements`
takes only `due_before`, a file being struck for a day, on the market the `tiers` policy
designates, so pointing it elsewhere is unrepresentable. It instructs every payment and premium due
by that day that nobody settled, totalled by currency, each row naming the `kind` its settlement is
filed under; a premium due on no day is never instructed.
**Held by** `test_diary.py::test_the_close_check_is_the_catch_up_rule`,
`::test_a_barrier_its_prints_knocked_out_owes_its_rebate_and_nothing_after`,
`::test_a_close_is_declared_only_on_a_day_the_check_calls_legal`,
`::test_the_settlement_file_is_struck_on_the_market_the_record_designates`,
`test_pnl.py::test_a_premium_announced_and_settled_moves_no_figure_of_the_p_and_l`,
`::test_a_close_naming_no_market_is_declared_on_the_market_the_p_and_l_reads`.

## The record's readers on the desk {#readers}

`GET /book/status` and `GET /book/risk` are polled, so each answer is kept on the served book
(`Book.polls`) under `service.poll_key`, read before anything is parsed or opened: the text's hash
and the book it names (`Book.named`), the sha256 of each file it names, the service's own files a
read names with their modification time and size - the XVA file for the status, keyed only once a
clock tick has passed since its last write - `spine.platter`, and today. `spine.platter` is the
home and the size of each log segment: a segment is appended to and never rewritten, so it moves
exactly when a frame lands and costs no open. A poll with nothing moved is a lookup
(`service.polled`), and an answer is kept only where its key still stands once it is worked out.
The risk asks admission before either of its two levels on every poll, and keeps its second level,
the hash of all the run reads, so a frame that moves none of it runs nothing. The status's
`spine.paper` is the LSN of the last entity or agreement declaration standing (`spine.papered`):
what a client holding the paper re-reads it on, the web client its entities and agreements
(`paperOwed` in `web/src/state.ts`), a booking moving its holdings alone. Its `close`, beside
`paper`, is the market the `tiers` policy designates for the P&L, null where none is, and whether
an official close `stands` declared on it (`service.spine_block`), and whether the book is
`marked` on it for its own base day (`service.book_marks_on`, read only where a close stands):
what a read of a marked close needs, as fields, so a client tells "no close stands" and "not
marked" without reading a refusal's words.
`GET /book/xva` reads its rows on every ask and keeps one part under the same key: `funding`, the
two curves a recalc asked
now would fund its FVA at (`service.funded`), read through `spine.official_calculation` as the run
reads them, a null for one the calculation does not state. The view does not go down with the
policy: where the record refuses that calculation the rows answer as ever and `funding` carries two
nulls beside `refused`, the sentence in the list a write's refusal answers. Its `limits` are the
[credit limits](#credit-limits) held against those same rows: what they read of the record
(`service.credit_limits`) is kept under the key, and each sum is of the rows the answer carries,
so a restated limit reads at the next poll and nothing runs. The web client says
what the next run funds at beside every FVA it shows - the XVA table, a client's rows, a credit
charge - off that answer alone (`fundedAt` in `web/src/client.ts`), and each credit limit on its
entity's row and page and under a credit charge: the limit, the usage of the sets run, the headroom
where every set has run, a flag where it is breached, and for a charge the usage without the trade
ahead of the usage with it (`limitRead`). **Held by**
`web/scripts/paper_check.mjs`, `web/scripts/client_check.mjs`,
`test_spine_engine.py::test_the_xva_view_names_the_curves_a_credit_run_would_fund_at_and_keeps_the_answer`,
`::test_the_xva_view_answers_its_rows_while_the_policy_in_force_cannot_be_read`,
`::test_the_limits_are_kept_under_the_poll_s_key_and_a_restated_one_is_read_at_once`,
`test_service.py::test_a_status_poll_with_nothing_moved_is_a_lookup_and_all_it_reads_is_in_its_key`,
`test_spine_engine.py::test_a_status_poll_is_answered_anew_once_the_record_moves_and_the_paper_has_its_own_stamp`,
`::test_the_status_says_which_market_the_p_and_l_reads_and_whether_a_close_stands_on_it`,
`::test_a_risk_poll_with_nothing_moved_is_a_lookup_and_a_frame_that_moves_no_number_runs_nothing`,
`::test_the_consolidated_risk_and_the_explain_ask_the_queue_s_question`.

`GET /book/activity` is the strip: with `?since=` the oldest rows after it, the cursor the last
delivered, so a capped walk loses nothing. The fold reads envelopes alone; the service opens the
bodies of the page it answers (`told`), so each row's `summary` names the trade and client, the
agreement or the entity it is about, `by` says who did it - the seat a denial refused, the tier
an automatic signature came from - and `deal_path`, `agreement` and `entity` are what it opens, a
restruck deal opening the terms the file holds now. The web banner (`web/src/spine.ts`) follows
reconcile's lists, never the behind counts, because every write re-pins the file at the head, a
delete included, and names each disagreement's trade, what is wrong and what to do.
`GET /book/worklist` lists what waits on the asking seat, `seat` null where none was named, each
row keyed by what its clearing fact is filed against, a call's naming the kinds of money it
settles as (`settles_as`, the record's `HELD`), and a ticket's row naming its trade, booker,
tier, booked price and the tier's sentence for why it waits. Its `pending` are the tickets the
seat may sign; those it is shown and may not are `elsewhere`, each with `may`, the record's
sentence why - the denial of its approval where the verdict is filed (`service.filed_at`), else
its tier's four eyes (`tiers.refuses`) - so a maker reads its own ticket waiting on a checker,
never as its to sign. **Held by**
`test_pnl.py::test_a_collateral_call_is_read_on_a_marked_close_and_settled_as_a_movement`,
`test_spine_engine.py::test_a_page_walks_the_record_forward_one_event_at_a_time`,
`::test_what_the_file_lost_is_drift_under_a_pin_that_has_seen_every_position`,
`::test_a_strip_row_names_the_trade_who_did_it_and_what_it_opens`,
`::test_the_worklist_names_each_ticket_and_the_seat_it_waits_on`,
`test_pnl.py::test_the_worklist_names_what_waits_and_each_row_leaves_when_its_fact_lands`.

The web client reads the worklist ONCE per head and seat into its store (`worklistAt` keys both),
and the banner, the Back Office and the approval queue count from that one answer. The Back
Office workspace (`web/src/views/BackOfficeView.tsx`) is the seat that signs, pays, confirms and
posts: the `pending` tickets with approve and refuse-with-a-reason, those `elsewhere` read only
beside the record's sentence for why this seat does not sign them, then the `payments`,
`unconfirmed` and `calls` lists - each the list of its own verb - every row beside the fact that
clears it, with the day's close check and the close behind its verdict. The Paper workspace
(`web/src/views/PaperView.tsx`) lists the entities and the agreements with their terms and
declares both, an agreement's terms the netting set rendered off the `NettingCollateralSet`
declaration `/schema` publishes, and opens on what a row of the strip names. Whether a set is
collateralised is the service's verdict, read as a run reads it, never the web's reading of the
flag: `collateralized` on each agreement `GET /book/agreements` answers, on each row of
`GET /book/xva` for a desk keeping no record, and the `collateralized_sets` a job's `/describe`
names for a run's tables. No screen shows a
set's own `Funding_Rate`, the term of its collateral adjustment: every view of a deal's fields
leaves it out (`onScreen` in `web/src/tokens.ts`), a set stating it keeps it through an edit, and
a client's readiness still asks the market for the curve it names. Each posts under the
acting seat and shows a refusal in the service's own words. **Held by** what they read,
`test_pnl.py::test_the_worklist_names_what_waits_and_each_row_leaves_when_its_fact_lands`,
`test_diary.py::test_a_close_is_declared_only_on_a_day_the_check_calls_legal` and
`test_spine_engine.py::test_the_paper_is_declared_and_a_booking_names_the_agreement_it_sits_under`,
`::test_the_agreements_read_says_whether_a_run_collateralises_each`,
`test_service.py::test_a_set_reads_collateralised_exactly_as_a_run_prices_it`,
`web/scripts/paper_check.mjs`, `web/scripts/client_check.mjs` and `web/scripts/trade_check.mjs`
for what no screen shows, and
the web's type check; the rest of the rendering is **Not gated**.

## Policies {#policies}

A policy is canonicalised, so one policy is one blob however spelled, and parsed at the
declaration, so one nobody could read never lands; the last declaration stands (`policy.in_force`).

| name | shape | absent means |
| --- | --- | --- |
| `capabilities` | `{grants, read, verbs}`, [above](#identity) | no enforcement |
| `tolerance` | `{"tolerances": {class: epsilon}}`, absolute, per top-level result class; a change of shape departs at any epsilon | nothing is pinned |
| `firmness` | `{"pillar_seconds": s}`, how old a quote's board may have been | nothing refused on staleness |
| `fixings` | `{"sources": {index: [source, ...]}}`, ordered | no index resolves |
| `tiers` | `{"tiers": [...], "designations": {process: market}}`, [below](#tiers) | fills land unsigned |
| `calculation` | `{"calculations": {type: block}}`, a block of the engine's own fields per calculation type, [below](#official-calculation) | a credit run reads its own document |
| `limits` | `{"limits": {entity: {"amount": a, "currency": c}}}`, a credit limit per legal entity, [below](#credit-limits) | no exposure is held against anything |

**Held by** `test_spine_verbs.py::test_a_policy_document_is_closed_at_the_field_level_and_refuses_where_it_is_declared`,
`::test_the_tolerance_comparison_is_structural_first_and_numeric_second`.

### The official calculation {#official-calculation}

What a recording desk's credit run runs at is policy, never a screen edit: the `calculation` policy
states, per calculation type, a block of that type's own fields in the engine's names. The record
checks the shape and knows no field (`policy.parse_calculation`), and refuses a type no official
run reads (`policy.OFFICIAL_RUNS`, `CreditMonteCarlo` alone): a block nothing reads is a rule nobody
enforces. The engine judges the fields. `spine.declare_calculations`, behind
`POST /book/calculations` and the `declare_official_calculation` tool, judges each block against its
type's declarations (`spine.judge_calculation`, the type token `Object` being the key and no field)
before anything appends, and reads no market: a curve the book does not carry fails the run that
reads it. `GET /book/calculations` answers the document in force with its blob and its position.
A document for an official run of 20 batches of 1,024 paths, both funding curves named, as
`DV_Spine declare calculation` takes it and as the route takes its one section beside `actor`:

```json
{"calculations": {"CreditMonteCarlo": {
  "Batch_Size": 1024, "Simulation_Batches": 20,
  "Funding_Valuation_Adjustment": {
    "Funding_Cost_Interest_Curve": "USD-FUNDING",
    "Funding_Benefit_Interest_Curve": "USD-OIS"}}}}
```

**Held by**
`test_spine_verbs.py::test_a_calculation_policy_is_a_block_per_official_run_and_the_record_checks_its_shape_alone`,
`test_spine_engine.py::test_the_official_calculation_is_judged_by_the_engine_before_the_record_takes_it`,
`test_mcp.py::test_the_official_calculation_reaches_a_model_as_the_record_declares_it`.

`spine.official_calculation(document, runs, lsn)` is the one reader: the block standing at `lsn`
laid over the document's own key by key, so a container the policy states replaces the document's
whole. [`compiled_job`](#the-plan) writes it where its caller says the compile is the served
book's own run (`official`, which `service.priced` passes and nothing else) and the job runs as a
credit run; it sits there because that function already holds what a job runs as and the position
it is compiled at. So the XVA recalc, the increment, a named credit calculation and a what-if
naming the type all start from it, a seat's own dials (`service.with_overrides`) going over it and
never official. Every other compile reads none: a job a seat posts whole to `/execute` or
`/prepare` states its own calculation, and a base valuation, the marks, a roll, a replay and a
diary are none of them the book's credit run, so a declaration moves none of their plans. A
recalc's own floor (`service.xva_document`) sits under both. A block declared past the judge, through
`DV_Spine declare`, is judged where it is read and refuses the run by name, in a sentence saying
that a seat holding `admin` declares it again and naming no route. A desk keeping no record has no
policy: its official calculation comes with its document. **Held by**
`test_spine_engine.py::test_a_credit_run_under_the_policy_is_the_run_of_the_same_block_written_in_the_book_file`,
`::test_the_policy_stands_over_the_book_file_which_stands_where_the_policy_says_nothing`,
`::test_a_field_an_adjustment_block_states_stands_over_the_recalc_s_floor`,
`::test_a_declaration_moves_no_plan_but_a_credit_run_s`,
`::test_a_restated_policy_is_a_new_plan_and_a_job_compiled_before_it_is_the_earlier_run`,
`::test_a_credit_job_a_seat_posts_whole_runs_at_its_own_calculation_under_any_policy`,
`::test_the_increment_and_the_credit_charge_start_from_the_policy`,
`::test_a_seat_s_own_credit_calculation_goes_over_the_policy`,
`::test_a_block_declared_past_the_engine_s_judge_refuses_the_credit_run_by_name`,
`test_service.py::test_a_desk_that_keeps_no_record_funds_at_what_its_document_states_and_declares_no_policy`.

### Credit limits {#credit-limits}

The convention. A counterparty credit limit is set per legal entity, and for a group on its
parent, on potential future exposure at a high percentile. Exposure nets inside a netting set and
adds across sets, the netting set being the unit of netting (the Basel Framework's counterparty
credit risk standard, CRE). Usage against a limit is monitored after each run and checked before a
trade: a trade that would take usage over the limit needs a credit officer's approval, and an
excess that market moves cause is reported and managed, not blocked (the interagency supervisory
guidance on counterparty credit risk management; the Basel Committee's guidelines for counterparty
credit risk management).

The `limits` policy declares the limits: per legal entity an amount above zero and the currency it
is an amount of. The record checks the shape and never whether it holds the entity
(`policy.parse_limits`). The declaring verb judges that: `spine.declare_limits`, behind
`POST /book/limits` and the `declare_credit_limits` tool, refuses by name a limit on an entity the
record does not hold before anything appends, a limit on one being a rule nothing enforces
(`spine.declared_entities`, the check an agreement's entity and a survival curve's counterparty
meet).
`GET /book/limits` answers the document in force with its blob and its position. The last
declaration stands whole. A document limiting a group on its parent and one member on its own, as
`DV_Spine declare limits` takes it and as the route takes its one section beside `actor`:

```json
{"limits": {
  "LEI-GROUP": {"amount": 25000000, "currency": "USD"},
  "LEI-A": {"amount": 10000000, "currency": "USD"}}}
```

**Held by**
`test_spine_verbs.py::test_a_limits_policy_is_a_credit_limit_per_legal_entity_and_the_record_checks_its_shape_alone`,
`test_spine_engine.py::test_a_limit_is_declared_on_a_legal_entity_the_record_holds_and_on_no_other`,
`test_mcp.py::test_the_credit_limits_reach_a_model_as_the_record_declares_them`.

`service.limit_usage` is the one owner of usage. An entity's usage is the sum, over the netting
sets the book carries against it and against every entity under it by the record's `parent` chain
(`service.credit_limits`, the walk `service.clients_under` makes), of each set's peak in the
highest-percentile `PFE_` column its standing XVA row carries (`service.draws_on`,
`service.pfe_column`). A set is its counterparty's, the `Credit_Support_Amounts.Counterparty` an
agreement's terms write as its entity. The rules:

- A peak below nothing counts as nothing: a set that owes the desk nothing takes none of another
  set's exposure away.
- A set with no run that is done is not guessed at. `sets` counts the sets summed and `unrun` the
  rest, and `headroom`, the limit less the usage, is stated only where `unrun` is nothing. Where
  every set is unrun nothing is measured: `usage` and `breached` are null too. An entity with no
  set at all uses 0.0.
- A limit in another currency than the rows' is not converted, the position a tier's size check
  takes of a notional: the entry states no usage and `not_comparable` says so naming both.
- `breached` is usage above the limit. Nothing is blocked and nothing runs.
- `clients` is the entities the usage covers, the limit's own among them, so a reader never walks
  the parent chain again.

`GET /book/xva` answers one entry per limited entity under `limits`, beside its rows
([the readers](#readers)). The credit charge, `POST /book/price` with `xva_increment`, answers the
same entry under `stats.Increment.limits` for each limit its set draws on, its counterparty's and
each above it (`service.limits_against`): this set's peak is the run's own in place of its row's,
so `usage`, `headroom` and `breached` are with the trade, and `usage_without` and
`breached_without` are the same sum and the same test at the peak of the run without it. What the other sets draw is read where the charge is asked and its
hash is part of what the result is named by, so the charge asked again after a limit is restated
or another row moves runs again. A set drawing on no limit, and every desk keeping no record, is
named and answered as it was. No tier reads a limit: a breach is reported and
[the approval it needs](#open) is not built. **Held by**
`test_spine_engine.py::test_usage_against_a_limit_is_the_peak_exposure_of_the_entity_s_sets_and_of_those_under_it`,
`::test_a_limit_on_an_entity_with_no_set_uses_nothing`,
`::test_the_limits_are_kept_under_the_poll_s_key_and_a_restated_one_is_read_at_once`,
`::test_a_credit_charge_holds_each_limit_its_set_draws_on_against_the_usage_with_the_trade`,
`test_service.py::test_a_desk_that_keeps_no_record_declares_no_credit_limit_and_claims_none`.

## Tiers and tickets {#tiers}

`spine.ticket(plan_hash, trade)` hashes the plan the book has once a trade lands with the trade's
fields as filed, so a retried fill or restrike coalesces and a changed plan needs a new approval.
The first tier whose `scope` covers the ticket and whose every declared check passes applies.
`max_notional` is `{amount, currency}` against the size the service reads off the deal as booked,
times the fill's quantity, and crosses at the book's spots (`service.ticket_terms`): what its type
declares for SA-CCR (`schema.Exposure`), an FX trade's larger side, a strip's amount at every
fixing still to come, a container's largest child; a restrike's is taken at every unit of the
positions it moves (`spine.amend`). No check reads a market, and a currency the ticket carries no
size in - a type declaring none, a cross the book cannot value - fails that tier;
`max_tenor_years` reads the latest day the type declares (`schema.tenor_fields`); `market`
requires the quote's values to stand under that name. A tier is automatic - the hub's approval in
the writer's voice - unless it declares `four_eyes`; staleness is `firmness`'s alone. **Held by**
`test_spine_tiers.py`,
`test_spine_engine.py::test_a_strip_is_sized_at_its_amount_at_every_fixing_still_to_come`,
`::test_a_structure_is_its_largest_leg_and_an_fx_trade_its_larger_side`,
`::test_a_fill_is_its_quantity_times_the_instrument_and_an_undeclared_type_no_size`,
`::test_a_restrike_is_sized_at_every_unit_of_the_positions_it_moves`.

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

The trade names the tier it fell in (`tier`, beside its ticket, which is taken over the trade as
built), so the record itself says which tickets the hub signs: the oracle holds an approval in the
writer's voice to an automatic tier its trade names, and a copy forging one over a trade that fell
in a four-eyes tier is named by the record alone. A copy rewriting the trade too has rewritten
history, which only a checkpoint or an agreeing copy catches ([trust limits](#trust-limits)); a
trade naming no tier was routed before trades named one. **Held by**
`test_spine_oracle.py::test_the_hubs_approval_of_a_ticket_its_trade_routed_to_four_eyes_is_forged`,
`test_spine_engine.py::test_an_automatic_tier_is_the_hubs_own_approval_between_the_quote_and_the_fill`,
`::test_a_retry_lands_every_event_on_the_lsn_it_already_has`.

The latest verdict stands, none being withdrawn; under four eyes the booker's own are not read
(`tiers.refuses`).
`POST /book/approve` and `/reject` file at the deepest node holding the ticket's positions, read off
the record, so an approver cannot sign another node's ticket. A rejected trade stands: it happened.
A trade sent again under its key is answered `{written: false, booked}`, other fields refused.
**Held by**
`test_spine_tiers.py::test_the_latest_verdict_stands_and_four_eyes_reads_who_filed_it`,
`test_spine_engine.py::test_a_verdict_is_filed_where_its_ticket_books`,
`::test_a_trade_sent_again_is_answered_where_it_stands`.

## Quotes and acceptance {#acceptance}

A quote records nothing; the record cares about the price a client takes. Under a home the pending
file ([Structures](structures.md#the-quote-lifecycle)) adds `pinned`: the book's two hashes, taken
before the live spot lands, and the board's age at `quoted_at`. The values vector behind
`values_hash` is kept once per market under that hash beside the pending files, and read through
`service.pinned_values`, which also reads a pending file filed with its vector on it. On every
desk it pins `terms` (`service.quoted_terms`): the mirror, the netting set and the model pin the acceptance
books. `values_hash` drops every clock (`schema.without_clocks`), so a board re-read
at the same prices is one market, and the age is the oldest of those clocks. **Held by**
`test_spine_engine.py::test_a_quote_records_nothing_and_files_what_the_acceptance_will`,
`::test_a_quote_pins_the_book_before_the_live_spot_lands_on_its_copy`,
`::test_a_pending_quote_keeps_its_vector_apart_and_one_filed_with_it_still_books`.

`POST /book/quote` checks, in order and before anything appends: `firm_seconds`, the desk's promise;
the `terms`, refused where the file no longer hashes to them (an edited file, or one filed before
the pin), whatever the book did; the plan, refused where the book moved under the solve and no
`Quote Policy` is declared; and the pillar age past `pillar_seconds`. The market is reported as
`{pinned, current, moved}`, never refused, the desk's window bounding it, and inside that window so
is a moved plan: the quote books as priced, `quote_filed` keeping the plan it was priced on and its
ticket the one it lands in. The price of that promise grows with the book and with how fast the
market moves. Then one act:
`quote_filed` under the acceptor, the tier step, one unit of the mirror filled under the quote id
at the premium, value dated on the pair's spot date for the trade day (`service.spot_day`, the
spot rule `structures.tenor_dates` dates the legs with), the file. A quote filed before a tier refuses stands; the client took the price. **Held by**
`test_spine_engine.py::test_a_booking_between_the_quote_and_the_acceptance_refuses_with_nothing_appended`,
`::test_a_moved_market_between_the_quote_and_the_acceptance_books_and_says_so`,
`::test_a_booking_elsewhere_inside_the_firm_window_is_read_as_a_market_move`,
`::test_an_edited_quote_refuses_whether_or_not_the_book_moved`,
`::test_an_accepted_premium_settles_on_the_pair_s_spot_date`.

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

**The files a job names.** A job names its market data file and its calendar by path and a job
standing alone runs off what they hold; the record declares their content. `file_declared`
(`mark`, judged at the firm) files a file's bytes under a name the desk gives it, its `kind`
`market` or `calendar`, its `blob` their sha256 - the digest `Context.digest` takes and a run
reports under `Stats['Files']`, one number. One name stands at a time, the latest by LSN, the body
naming the declaration whose content it replaced (`supersedes`), so a rollback to an older blob is
a new fact while the standing content declared again coalesces (the `files` fold).
`POST /book/files` declares the bytes at a path on the service's box and `GET /book/files` lists
what stands, each beside the file the live book names that holds its content now (`DV_Spine file`;
the MCP's `declare_file` and `book_files`). A replay hands its context the digests the attestation
carries (`spine.reading`, a maker of contexts holding those bytes, which `executor` builds on):
the context reads a declared file off the store and any other off its path, the replay's own
`Stats['Files']` saying whether it read the same. Every reader of a past marks job reads the same
way, `PnL.marked` carrying each marks run's `files`: the diary of that job
(`service.diary_rows`), the base currency a window crosses its cash at, the explain's valuations
of the start's job and the end's (`sensed`, `carried`, the maker threaded through
`quote_run` and `curve_edit`), and the call's `Exclude_Paid_Today`
(`Collateral.paid_today`), so a file rewritten since moves no struck day. **Held by**
`test_spine_verbs.py::test_a_file_is_declared_by_its_bytes_and_the_latest_under_a_name_stands`,
`test_spine_projections.py::test_the_file_standing_under_a_name_is_its_latest_declaration_by_lsn`,
`test_spine_engine.py::test_a_run_says_the_files_it_read_and_a_replay_reads_the_bytes_that_stood`,
`test_mcp.py::test_the_file_the_book_names_is_declared_and_listed_through_the_binding`,
`test_pnl.py::test_a_window_reads_the_files_its_marks_read_as_they_stood`,
`::test_the_explain_and_the_call_of_a_past_close_read_the_files_its_marks_read`.

## Queue admission {#admission}

`ComputeExecutor.submit` asks before anything runs: `mark` over its book for a standing run,
`validate` over the book or a node under it otherwise; a refusal files a denial and queues nothing.
A posted job is asked that question before it is loaded or compiled - `/execute` in its own lane,
`/validate`, `/describe` and `/prepare` as a what-if - so a seat refused costs the box nothing and
a job it could not load is its refusal, never the document's. The reads that value on the
request thread, `/book/risk` and a P&L read to now or with its
`explain` (`PNL_RUNS`, which a seat's `may` reads too), ask a what-if's question before
they value. A posted tick and a securities verification
queue under the seat that asked; the poll paths (a read's diary, the metronome's beat) run under
`spine.hub()`, `DV_SPINE_ACTOR`, which a ticking desk must grant `validate`. Admission precedes the
risk and diary caches, so no cached answer depends on who asked first. The web client reads the
service's verdict on every read and run the queue admits before it asks (`heldBack` of `run`), a
refused ask filing a denial: a credit charge, a calculation run, a quote, and a tick or a
verification after its write's own verdict, each control saying what holds it; every write of
the market by its route's own name (`market_tick`, `curve` and the rest); and the risk and a P&L
to now or explained (`pnlHeld`, the service's verdict on that run), which a held seat is not sent,
that line standing in the read's place with nothing another seat read (`readAs`, `bookRisk` in
`web/src/state.ts`). The
P&L page opens such a seat on the last two marked days, a window that asks no what-if (`opensOn`),
tells it where fewer are marked that such a window reads once two are (`noWindow`), and reads
again for another seat (`pnlRead`). Neither read is asked, and no such run is offered (`runHeld`),
before the grants are read for the seat on screen (`grantsOwed`), the book loading with where the record stands (`readBook`).
**Held by** `web/scripts/desk_check.mjs`, `web/scripts/pnl_check.mjs`,
`test_spine_engine.py::test_under_a_document_the_queue_asks_before_the_executor_does`,
`::test_the_poll_paths_own_jobs_are_admitted_under_the_deployments_seat`,
`::test_the_consolidated_risk_and_the_explain_ask_the_queue_s_question`,
`::test_the_tick_and_the_verification_queue_under_the_seat_that_asked`,
`::test_a_refusal_names_its_kind_beside_its_sentence`.

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
the netting set its positions compile into, judged by the engine first, the agreement's entity its
counterparty (`Credit_Support_Amounts.Counterparty`). A later declaration under one id names the
one whose content it replaced (`supersedes`), as a file's does, so a return to earlier content is a
new fact while the content in force declared again coalesces. An entity's name identifies one
entity: `verbs.declare_entity` refuses, by name, a declaration whose name stands under another id,
compared as written, and the fold reads two ids a log already holds under one name as it did. An
agreement's id belongs to one entity: `verbs.declare_agreement` refuses, by name, a declaration
under an id standing under another entity, and the fold reads an agreement a log already holds
moved to another as it did. A fill's `quantity` is the signed
change in units; its optional `agreement`, `portfolio` and `price` say where and at what, and
`value_date` the day that price's consideration settles - refused by name without a price. One
instrument under two agreements is two positions, credit exposure being per agreement; in two
portfolios, two. The first booking or quote under a declared agreement splices its set from the
declared terms, and every compile writes the terms standing at its LSN over it
([the plan](#the-plan)). **Held by** `test_spine_paper.py::test_a_position_is_keyed_where_it_sits`,
`::test_a_return_to_earlier_paper_is_a_new_fact_and_a_repeat_of_the_standing_paper_is_one`,
`::test_a_name_identifies_one_legal_entity`,
`::test_an_agreement_s_id_belongs_to_one_legal_entity`,
`test_spine_engine.py::test_the_first_booking_under_a_declared_agreement_brings_its_set`,
`::test_the_set_s_counterparty_is_the_agreement_s_entity`,
`::test_paper_restated_and_then_restored_stands_restored`,
`::test_a_name_another_entity_holds_is_refused_in_the_record_s_words`,
`::test_an_agreement_id_another_entity_holds_is_refused_in_the_record_s_words`.

`portfolio_declared` is judged at its parent. Once any node of a book stands, a fill books into the
book or a declared node. `GET /book/positions` answers each standing position beside the file's
nodes, an expired one until it settles and one netted to nothing at a quantity of nothing. Each
names the fill that booked it first (`booked`, which a restrike carries onto the new terms), the
one owner of who booked a trade: the Blotter's "Booked by" and its own-trades filter
(`bookedBy` in `web/src/positions.ts`), the trade's page and a position's first fill read it, and
never a ticket's seat, which a restrike signs again. A seat with grants at nodes reads only the
rows under them (`spine.visible`), totals summed over those - presentation while one class key
opens every body.
A client's pending quote is shown the same way, by the `portfolio` it books under, a quote being
confidential to those who need it (FX Global Code, Principle 19): `GET /book/quotes` lists, and
`GET /book/quote/{id}` and its sheet answer, the quotes under the seat's nodes and the quotes the
seat gave itself (`quotes_shown`), one the seat is not shown answering as an id the desk holds
nothing under. The seat a read is made as is its token's, another name refused, else the one it
names (`spine.reader`), which `spine.sight` and the worklist read too.
A trade's page tells a seat shown no position in it, no fill of it, no quote that landed at it or
no row of its history that the record holds none only where `GET /book/seat` answers that the
seat is shown the whole book (`sees` null), and else that the seat is shown none, with the nodes it
sees; the record's strip says the same to a seat shown no
row (`noneShown` in `web/src/state.ts`). The book marks a node two positions hold at their net, and
a screen showing one of them shows the risk answer's own share of that mark for the position
(`per_position`, read by `markShown` in `web/src/trade.ts`), off a risk read of the book as it
stands (`riskNow` in `web/src/state.ts`). An unwind's price is typed beside the position's unit
mark times the units unwound, with what
the desk makes or gives up against it at the price (`unwindAgainst`), or why no mark is read
(`noMark`).
**Held by** `test_spine_engine.py::test_a_declared_tree_is_where_a_book_books_and_a_restrike_is_judged`,
`::test_a_seat_reads_the_rows_under_the_nodes_it_acts_at`,
`::test_a_seat_is_shown_the_quotes_booking_under_the_nodes_it_acts_at`,
`::test_a_seat_is_shown_the_quotes_it_gave`,
`::test_a_token_s_seat_is_shown_the_quotes_it_gave`,
`::test_a_position_netted_to_nothing_reads_closed_where_it_sat`,
`web/scripts/desk_check.mjs`, `web/scripts/trade_check.mjs`, `web/scripts/positions_check.mjs`.

**The take-on.** A deal the book file holds that nobody booked - a book written before its record,
or landed by another system - prices as written, one unit, outside the positions, the marks and
the P&L. `POST /book/deals` `{action: 'fill', deal_path}` books it where it stands, as an `add`
books the node it splices: the record files the fill, the file takes only its pin. A fill of one
under the node's own reference is its take-on and moves no number, the compile writing a net of
one as the node is written; from then on it is a booked trade, and closing it files -1, which nets
it to nothing where the compile would otherwise write the mirror. `GET /book/reconcile` names what
waits under `in_file_not_in_record`, on a file nothing pinned too, a container nobody booked
anything in whole, so a structure is taken on as the one instrument it is. **Held by**
`test_pnl.py::test_a_book_written_before_its_record_is_taken_on_as_if_it_were_booked`.

## Cash, cost and P&L {#pnl}

A `settled` transition may carry a signed `amount` (received positive), `asset`, `kind` and the
settlement system's `reference`: a `payment` settles its diary row and a `premium` a fill's premium
row, the two kinds that move a row's state; a `fee` sits on its instrument; `collateral` and
`margin` move a balance under a declared agreement. The reference is the movement,
so a correction under it restates the amount. A payment or a fee may name the declared `agreement`
it was paid under, and `POST /book/transition` refuses a payment naming none by name where its
key is due under more than one netting set: the P&L shares a movement among the positions under
the agreement it names alone, a position no movement on its row reached reading that payment
unknown until its own lands. **Held by**
`test_pnl.py::test_a_payment_is_shared_within_the_agreement_it_names`. `costs` keeps the open
`basis` at average cost and what reductions `realised`; a fill with no price leaves the basis
unknown, never zero. **Held by**
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
- a settlement against a payment the window's diary has dropped is matched in the latest earlier
  marks whose diary still announces it, however late it settled;
- a position whose last day falls in the window closes there, its basis released;
- what nobody can know is named under `unknown`, and a total over it is null.

`explain` splits it into carry, market (factor moves times the start's sensitivities, per quote) and
residual. **Held by**
`test_pnl.py::test_the_book_makes_its_marks_move_plus_its_cash_and_a_month_is_its_days`,
`::test_a_payment_on_a_marked_day_is_that_day_s_cash_and_out_of_its_marks`,
`::test_money_nobody_can_place_is_named`,
`::test_a_settlement_eight_business_days_late_is_matched_to_the_row_it_settles`.

## Collateral {#collateral}

`projections.CSA` reads a set's terms as the engine does - each `CreditSupportList` at its first
value - and `required` is the engine's `At`. `call` is that less the collateral held, what the bank
holds counted at face and what it posted at one less its `Haircut_Posted`, as the engine's balance
counts each side (`collateral_bands`), moving only strictly past the minimum transfer on its side,
as `scan_collateral_balance` does; `Haircut_Received` is the closeout's credit on what is held
(`collateral_credit`), which no call reads. Margin is a separate balance. Being pure, the oracle
replays it.
`GET /book/collateral` answers each agreement on a marked close, its exposure the P&L's value of its
positions with the day's payment added back where the set holds it (`Exclude_Paid_Today` off); an
unknown mark makes the call null, never zero. **Held by**
`test_collateral.py::test_the_support_required_is_the_engine_s_own_at`,
`::test_a_call_clears_its_minimum_strictly_on_the_side_it_falls`,
`::test_what_the_bank_holds_counts_at_face_and_what_it_posted_one_less_its_haircut`,
`::test_each_side_is_cut_by_its_own_haircut_and_the_call_is_the_engine_s_transfer`,
`::test_a_mark_nobody_has_is_named_and_calls_nothing`.

`compiled_job` writes a collateralised set's `Opening_Balance` - collateral held by the base date,
counted as the call counts it, in its `Balance_Currency` - for a run with a collateral recursion; a
base valuation has none, so a settlement moves neither its plan nor its cache. An asset without a
spot refuses every valuing compile rather than count as zero. **Held by**
`test_collateral.py::test_the_record_s_balance_opens_the_engine_s_recursion_and_the_call_is_its_transfer`.
Not carried: interest, securities as collateral, rating-dependent support, and the hedge
simulation's opening state.

## The oracle {#the-oracle}

`oracle.py` reads a home as a replica and re-runs the writer's decisions through its own functions,
so an answer is a property of the bytes; the script of what was asked, the diary's rows and the P&L
are handed in. Each invariant answers `{held, evidence}`, `held` null - never a pass - where the
question could not be put.

| invariant | holds where |
| --- | --- |
| 1 copies agree | two homes share head hash and rows at the shallower head |
| 2 nothing outside its seat | every frame's seat held the verb its act demanded under the map in force when it landed; the writer's voice said only its own |
| 3 every refusal is a denial | denials equal the refusals the script says reached the writer |
| 4 an amended plan is a new approval | each ticket is one trade's; a restrike has its own |
| 5 closes superseded, never edited | no close the vocabulary would refuse |
| 6 every number's replay tuple | attestations are exactly the standing runs asked for |
| 7 the diary equals the filings | each settlement names a diary row or clip |
| 8 no delete | positions dense, cited blobs present |
| 9 duplicates coalesce | each tag on one line |
| 10 nothing outside its scope | each node-judged frame held its verb there |
| 11 the P&L is additive | days sum to windows and portfolios to the book within the `pnl` tolerance |
| 12 cash reconciles | payments match the diary, a row is paid once under each agreement |
| 13 every close marked | each `pnl` close has marks while the book held anything |
| 14 the call is the formula | each call read is `CSA.call` over the record |
| 15 files declared | each file an attested run read is a blob a `file_declared` at or before it cites; not asked where runs read files and none is declared |

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
| Pillar staleness, tolerance, capabilities, the official calculation, credit limits | none declared | `policy.py`, `capability.py` |

## Open {#open}

Unbuilt parts of the design: the deal tree is the book file with nets and paper written on, not
composed from the fold; a print is not yet one fact feeding both a calibration's archive and a
deal's fixings ([designed, not built](roadmap.md#designed-not-built)); and a trade that would
breach a [credit limit](#credit-limits) is reported, with no approval routed to a credit officer.
The rotation and replay limits are
[trust limits](#trust-limits) and the replay tuple's reach is on
[Model Limitations](../limitations.md#the-engine); which verbs take a seat and whether a rotation
is signed by the key it retires are
[desk decisions](../limitations.md#decisions-waiting-on-the-desk).
