# MCP Binding

`derivus_mcp/server.py` serves the derivus verbs as MCP tools, so a model can book, quote and value
in plain language. It is a client of `DV_Service` beside the web UI and the Excel add-in, and owns
no logic: each tool adapts one or two endpoints, so anything a tool needs that no endpoint answers
is a missing verb on the service, never code here. It sits outside the `derivus` package because
importing the engine pulls torch into a process that only speaks HTTP.

| where | what |
| --- | --- |
| `derivus_mcp/server.py` | the tools, prompts and resources on `MCP`, the host-facing `INSTRUCTIONS`, `main` (stdio) |
| `Service`, `configure` | the HTTP client; `session` is the transport seam a test drives in process |
| `_await_result` | the poll loop every waiting tool rides, reporting progress on each poll |
| `_summary`, `_booking` | a run trimmed to its replay tuple and table shapes; a booking answer to that booking |
| `derivus/service.py` | the endpoints, and `Metronome` behind `--tick` |

Tests named below are in `tests/test_mcp.py` unless another module is given; they drive the tool
functions against the in-process service (`configure(session=TestClient(service.app))`). The module
imports only `asyncio`, `os`, `time`, `requests`, `mcp` and `mcp_types`. **Held by**
`test_the_mcp_server_imports_neither_the_engine_nor_the_add_in`.

## Running it

```
pip install 'derivus[desk]'                     # service, binding and quote sheet; python 3.10+
DV_Service --book path/to/job.json
claude mcp add derivus -- DV_MCP
claude mcp add derivus --env RF_SERVICE_URL=http://host:8000 -- DV_MCP   # a service elsewhere
```

`RF_SERVICE_URL` (default `http://127.0.0.1:8000`) is the variable the Excel add-in reads, so one
setting configures every client; `DV_SPINE_TOKEN`, where set, is sent as the bearer on every call.
No `.mcp.json` is tracked, because it would pin one machine's paths into the repo.

## The instructions

A host shows the server's `instructions` to the model once per session, before any call, so
`INSTRUCTIONS` is the desk's orientation and the module docstring stays the maintainer's. They say
to start with `desk_status`, that a deal states its terms and leaves conventions to their declared
defaults, the wire forms (`{".Timestamp": "YYYY-MM-DD"}`, `{".Percent": 2.5}`), that an FX
`Strike_Price` is on the engine axis, that a refusal is an answer, that quoting is not booking, and
how the market and the record are worked. **Held by**
`test_the_instructions_a_host_shows_are_the_desks_orientation`.

## The tools

A tool's docstring is the contract the model reads, and every read carries the read-only hint so a
host runs discovery without asking to write. **Held by**
`test_every_tool_is_registered_and_carries_its_contract`, which holds the registry to exactly these
tools and the writers to exactly those without the hint.

| tool | endpoint: what |
| --- | --- |
| `desk_status` | `GET /book/status`: date, currency, curves and surfaces with their snap times, models, netting sets, last XVA, terminal |
| `list_instrument_types`, `describe_instrument_type` | `GET /schema`: the bookable types; one type's declared fields, a non-type answered with close matches |
| `describe_structure`, `describe_calculation_type`, `describe_factor_type`, `describe_configuration` | `GET /schema`: a structure, calculation, price factor, or the bootstrap dials |
| `job_skeleton` | `GET /schema/job`: a minimal job that loads |
| `read_book`, `read_deal` | `GET /book`: one row per deal with its `deal_path`; one deal verbatim |
| `book_deal`, `amend_deal`, `delete_deal` | `POST /book/deals`: add, merge fields at a path, remove |
| `validate_book`, `describe_book` | `POST /validate`, `/describe` over the live book |
| `describe_portfolios`, `declare_portfolio` | `GET`/`POST /book/portfolios`: the declared tree |
| `price_candidate`, `execute_book` | `POST /book/price`: the book with a candidate, or alone; nothing written |
| `describe_calculations`, `configure_calculation`, `run_calculation` | `GET`/`POST /calculations`, `POST /calculations/run`: the desk's named calculations |
| `solve_deal` | `POST /book/solve`: one field solved to a target, the deal ready to book |
| `solve_structure`, `book_quote` | `POST /book/structure`, `/book/quote`: a quote filed pending; its acceptance |
| `approve_ticket`, `reject_ticket`, `approve_quote`, `reject_quote` | `POST /book/approve`, `/book/reject`: a second seat's verdict |
| `worklist` | `GET /book/worklist`: what waits on the seat asking |
| `poll_result`, `fetch_table`, `deal_values` | `GET /results/{id}[/{table}]`: status and shapes; one capped page; `{reference: value}` |
| `update_market_quotes`, `patch_market_values` | `POST /book/market`: quote blocks ticked and bootstrapped; values patched |
| `configure_book` | `POST /book/configure`: one bootstrap dial, the market re-bootstrapped |
| `describe_curve`, `configure_curve` | `GET`/`POST /book/curve`: curves as definitions; a curve authored from its rows |
| `set_base_date` | `POST /book/date`: both dates moved, every curve re-rolled |
| `tick_market_from_bloomberg` | `POST /book/bloomberg`: surfaces, curve rows and spots off this terminal |
| `describe_securities`, `configure_securities`, `verify_securities` | `GET`/`POST /book/securities`, `POST /book/securities/verify`: the ticker seed, map and IPV join |
| `book_dependencies`, `setup_market` | `GET`/`POST /book/dependencies`, `POST /book/setup`: what a trade needs; that market built |
| `calibrate_spot_model` | `POST /book/model`: one pair's spot model fitted, never on the tick |
| `declare_survival_curve` | `POST /book/survival`: a counterparty's curve from the CVA desk's spread |
| `book_risk_summary` | `GET /book/risk`: the mark and its largest quote and factor rows |
| `xva_view`, `recalc_xva` | `GET`/`POST /book/xva`: the cached projection per netting set; the verb that moves it |
| `book_diary` | `GET /book/diary`: every payment, fixing and expiry |
| `close_check`, `declare_close` | `GET /book/close/check`, `POST /book/close`: whether a close is legal; the close |
| `book_markets`, `declare_market` | `GET`/`POST /book/markets`: the closes, names and snapshots; a name declared |
| `book_reconcile`, `book_activity` | `GET /book/reconcile`, `/book/activity`: file against record; the record's strip |
| `export_settlements` | `POST /book/settlements`: the settlement file for a day |
| `file_status`, `file_lifecycle` | `POST /book/transition`, `/book/lifecycle`: a settlement or match; a fixing, election or ruling |
| `book_cash`, `collateral_calls` | `GET /book/cash`, `/book/collateral`: money moved; what each CSA calls |
| `mark_book`, `book_pnl` | `POST /book/marks`, `GET /book/pnl`: the close's marks; the P&L between them |
| `book_positions` | `GET /book/positions`: positions under agreement and portfolio |
| `declare_legal_entity`, `declare_agreement`, `describe_agreements` | `/book/entities`, `/book/agreements`: the paper the book trades under |

A read that needs a record answers the service's 404 on a desk that keeps none, never an empty
answer a model would read as agreement. **Held by**
`test_the_record_reads_say_so_on_a_box_that_records_nothing`.

## Prompts and resources

A host offers prompts as commands and resources as documents, so both are contract. Three prompts,
each a numbered walk: `quote_a_structure`, `import_a_legacy_book` (the deals as one
`NettingCollateralSet` in one booking) and `morning_desk_check`. Four resources: `derivus://book`
(the job document), `derivus://schema/{store}` (one of the seven `/schema` stores),
`derivus://quote/{quote_id}` (the pending trade) and `derivus://quote/{quote_id}/sheet` (the
`.xlsx`, so the client's workbook travels as a file rather than a path on the service's disk); a
quote answers with those two URIs under `resources`. **Held by**
`test_the_host_is_offered_three_walks_and_four_documents`.

## The contracts

### Validate before write {#validate-before-write}

`book_deal`, `amend_deal` and `book_quote` write only if the edited copy says nothing new about the
deal, about a deal the book carried no message for, or about market data the book did not already
lack, because a deal the book cannot price loads and is then dropped by discovery. A refusal is a
normal return, `{written: false, refused: [...]}`, because the model's next move is to fix what it
names; a tool error means the call cannot proceed (the service down, a parent that takes no
children, a quote id with nothing filed). `price_candidate` and `solve_deal` run the same check
before they queue. **Held by**
`test_a_rejected_booking_is_an_answer_that_wrote_nothing`,
`test_what_a_model_actually_mistypes_comes_back_as_data`,
`test_the_store_names_what_must_be_stated_and_a_missing_term_refuses_by_name`,
`test_service.py::test_a_candidate_naming_market_data_the_book_lacks_is_refused`.

### A refusal names the thing and the remedy

A wrong argument refuses by name before anything prices, because the alternative is a solver's or
a tensor's exception naming neither the field nor the deal: a solve on an undeclared field, a
`bounds` that is not a pair or an unreachable target; a calculation override the calculation does
not declare, which would otherwise be ignored, or one that is not what it declares; a quote for a
netting set the book never opened; a credit Monte Carlo with no model or no live deal; a node whose
`Object` names no type, by walk position. **Held by**
`test_service.py::test_a_solve_refuses_a_field_and_a_bounds_the_deal_cannot_carry`,
`::test_a_calculation_override_is_judged_against_its_own_declarations`,
`::test_a_credit_monte_carlo_this_book_cannot_frame_refuses_before_it_draws`,
`::test_a_book_carrying_an_unnamed_deal_names_it_rather_than_dying`.

### A recorded desk books through these tools

Under `DV_SPINE_HOME` a booking also needs a signed `quantity` (the position change in units of the
deal), an `execution_reference` and an enclosing `NettingCollateralSet` naming a counterparty, and
every write files under the seat in `actor`. They are optional in the schema because one schema
serves a desk with a record and one without; the service's refusal names a missing one. A delete
records nothing, so it takes no seat. `describe_instrument_type` marks the fields a position scales
(`sized`), flips (`side`) and settles cash on (`settles`). **Held by**
`test_a_recorded_desk_books_and_accepts_through_the_binding`.

### Deals are addressed by position {#positional-addressing}

`deal_path` (`"0/2/1"`) is the identity because references are not unique in a book. Another host's
booking moves every position under a path, so `amend_deal` and `delete_deal` take the `reference`
read there and refuse when the path no longer holds it, rather than acting on whoever sits there
now. **Held by** `test_service.py::test_a_tick_that_keeps_losing_to_bookings_lands_on_its_last_pass`.

### Answers are summaries and pointers

The model's context is a budget. A run comes back as its replay tuple, stats and one line per table
(`"250 rows x 4 columns"`); a booking answer counts the book's other troubles and points at
`validate_book`; `fetch_table` pages at most `MAX_PAGE_ROWS` (200) and refuses a table wider than
`MAX_TABLE_COLUMNS` (60) unfetched; `deal_values` reads the shape first and refuses a cube or more
than 500 rows. **Held by** `test_a_run_comes_back_as_shapes_never_cells`,
`test_fetch_table_is_capped_and_a_cube_is_refused`, `test_deal_values_refuses_a_cube_without_fetching`,
`test_a_booking_answer_is_the_booking_not_the_book`.

Risk is whole-book and counterparty-blind, one base valuation cached on the book's content; XVA is
a projection per netting set in `DV_HOME/xva.json`, each row with its `as_of`, moved only by
`recalc_xva` because a credit Monte Carlo is too slow to ride a tick. The docstrings say so, so a
model neither reads an old CVA as live nor runs a simulation to answer a delta.

### Solves run in the service

`solve_deal` runs the root find server-side (brentq inside `bounds`, else a secant, exact in two
pricings for an affine field) and answers a `solved_deal` ready to book, so no pricing loop runs
through the conversation. A money `target` is crossed at the book's spots and recorded as
`Sales_Margin`. **Held by** `test_a_margin_target_is_money_and_the_deal_records_what_was_charged`,
`test_service.py::test_a_solve_brackets_a_nonlinear_strike`.

### Structures are declared

A structure is a class in `derivus/structures.py` served by `describe_structure`, so what a collar
is never depends on which model is driving. The model fills its parameters with strikes in market
terms, and the runner crosses the axis and selects the [variation](structures.md#variations) or
refuses ([Structures](structures.md)). Every FX `Strike_Price` and `Barrier_Price` declaration names
its axis for a model booking an option directly. **Held by**
`test_the_structure_store_is_the_quoting_menu`,
`test_the_fx_strike_axis_is_published_on_the_field_a_model_fills_in`.

### Quoting is not booking

`solve_structure` prices the legs at mid, charges the [two-way](structures.md#two-sided) and margin,
files the pending trade under `DV_HOME/tmp/<quote_id>` with its sheet where `xlsxwriter` is
installed, and records nothing: a desk quotes many times and the record holds what the client
takes. It prices on the live spot where this terminal answers, else the book's, and says which.
`book_quote` is the acceptance, on the client's word: it books the mirror (the quote is client
paper, the book the bank's side) under the quote's netting set through `book_deal`'s seam and keeps
the pending file as the audit trail. Past a declared `Quote Policy.firm_seconds` it refuses. Under a
record a moved market is reported, never refused, and a tier wanting a second seat books the trade
pending until `approve_quote`. **Held by**
`test_the_quoting_day_runs_from_a_structure_name_to_a_booked_collar`,
`test_a_recorded_desk_books_and_accepts_through_the_binding`,
`test_structures.py::test_a_quote_is_an_act_not_a_lookup`,
`test_service.py::test_a_quote_is_firm_only_for_the_window_the_book_declares`.

### A waiting tool speaks while it waits {#progress}

A desktop host cuts a call that stays quiet for about a minute and resets that clock on every
progress notification, and several runs take minutes. So each tool that sits on a run
(`price_candidate`, `execute_book`, `run_calculation`, `solve_deal`, `solve_structure`,
`recalc_xva`, `calibrate_spot_model`, `tick_market_from_bloomberg`, `verify_securities`,
`setup_market`) is `async`, threads its blocking calls, and waits in `_await_result`, which reports
the seconds waited, the `wait_seconds` and the run's own `progress` note on every poll: a quarter
second apart for two seconds, then a second. `ctx` never appears in a schema, since a host would try
to fill it. Past `wait_seconds` the answer is `{result_id, status, hint}` and the run carries on.
**Held by** `test_every_waiting_tool_is_async_and_takes_the_context` (its list omits
`run_calculation`), `test_no_tool_advertises_the_context_the_sdk_injects`,
`test_a_waiting_tool_notifies_the_host_the_whole_time_it_waits`.

### First use provisions the desk

On a fresh workstation `tick_market_from_bloomberg` is the setup: it creates `DV_HOME`, copies the
packaged seed in and verifies every candidate against this terminal before fetching, since a retired
series keeps quoting a plausible price and only its update date says so. Its answer is the
provisioning's own, not a run summary. **Held by**
`test_a_bloomberg_tick_returns_the_finished_payload`,
`test_service.py::test_the_bloomberg_verb_provisions_fetches_and_ticks_the_book`,
`::test_a_stale_quote_refuses_the_tick_by_name`,
`test_bloomberg_discover.py::test_first_use_provisions_once_and_the_second_run_probes_nothing`.

### A ticking service refreshes itself

`DV_Service --tick SECONDS` (`TICK_SECONDS`, 30, with no value) runs a `Metronome` submitting the
job `POST /book/bloomberg` submits, so a beat serialises with pricings and lands atomically. It
never stacks (a beat whose predecessor is in flight is skipped), never provisions (verifying a
workstation is a person's decision), and never dies: a failed beat is one warning and an untouched
book, and `TICK_FAILURES` (3) in a row stretch the interval `TICK_STRETCH` (5) times. `--tick` is a
usage error without `blpapi` or beside `--no-book`. **Held by**
`test_service.py::test_the_metronome_skips_the_beat_its_last_tick_is_still_in_flight`,
`::test_a_routine_tick_refuses_an_unprovisioned_home_and_leaves_the_book_alone`.

### A market is set up from what a trade needs

`book_dependencies` walks a candidate spliced as the what-if splices one, so the answer is that
trade's want-list, and names the seed entry that would supply each missing factor; `setup_market`
builds it in one call. It discovers only names the map has never heard of, checks every print for
freshness before fetching, and installs surface, spots and curves in one write the bootstrap
judges. A new currency lands as a pair or not at all, because its curve values its benchmarks in
the book's base through its `FxRate`; what can complete still lands, and `not_supplied` says why
the rest did not. A candidate `book_deal` would refuse is refused in its words, so a misspelt type
never reads "nothing is missing". **Held by** `test_a_market_is_set_up_from_what_a_trade_needs`,
`test_service.py::test_a_surface_is_held_with_the_leg_it_cannot_stand_on`,
`::test_what_the_write_can_complete_still_lands`, `::test_a_late_print_refuses_the_whole_set_up`.

### A curve is set up from its instruments

`configure_curve` takes the rows a desk quotes (tenor, security, number), and the tenor says what
each instrument is (`1Mx4M` a FRA, `6M1M` a forward-starting swap), so a model never authors the
instruments. The block is re-authored whole in one write, which is why this is not
`update_market_quotes`: a benchmark set is structure, and structure is never a tick. A terminal
print later than the book's date rolls both dates onto it, here and on the tick, and re-authors
every other curve there, so no two curves stand on different days; `set_base_date` moves the date
anywhere. **Held by** `test_a_curve_is_described_set_up_and_read_back`,
`test_the_date_is_set_by_name_and_the_curve_follows_it`,
`test_service.py::test_a_curve_set_up_off_a_later_print_rolls_the_book_and_every_other_curve`.

### Market data moves on the engine's terms

`update_market_quotes` may move only each row's `schema.MARKET_QUOTE_VALUES`, a changed pillar or
convention refusing as a re-authoring, and any bootstrap error refuses the whole write, so the book
never carries a market its bootstrap complained about. `patch_market_values` rides the
`bind='value'` seam, which refuses a structural key or a mistyped value, and refuses a
`Market Prices` block naming `update_market_quotes`, because a patched quote would leave the written
factors stale against it. **Held by** `test_service.py::test_a_quote_update_may_move_only_the_numbers`,
`::test_a_market_values_patch_reaches_the_file_and_a_structural_one_is_refused`,
`::test_a_bootstrap_that_complains_writes_nothing`.

## Open and non-goals

Progress over SSE, and a screen for the dependency walk and the set-up, are on the
[roadmap](roadmap.md#designed-not-built). The binding serves stdio only, which is what a desktop
host launches.
