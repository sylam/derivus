# Deployment and Operations

This page is for whoever installs and runs derivus: what each component is, how it starts, what it
reads from the environment, what it keeps on disk, who may write what, how it is secured, backed up
and upgraded, and what hardware it wants. [Getting Started](getting_started.md) is the same stack
from a user's side; the developer pages linked below hold the design behind each rule.

## Components

Everything installs from one package. The core is the engine alone; the extras add the service, the
model binding, the quote sheet and the record (the full list is on [Requirements](require.md#extras)).
Python 3.10 or later is required.

| install | what it brings |
| --- | --- |
| `pip install derivus` | the engine, called in process from Python |
| `pip install "derivus[desk]"` | the engine, `DV_Service`, `DV_MCP` and the `.xlsx` quote sheet; the web UI rides in the wheel |
| `pip install "derivus[desk,enterprise]"` | the above plus the book of record (`DV_Spine`), for a box that keeps one |

| component | what it is | how it runs |
| --- | --- | --- |
| engine | the `derivus` package: valuation, simulation, calibration, sensitivities | in process, `Context.load_json` then `run_job`; on CPU or GPU |
| `DV_Service` | the HTTP service over the engine: the live book, the job queue, the web UI | a long-running process; `python -m derivus.service` is the same program where the script is not on the PATH |
| web UI | a browser client of the service | served by `DV_Service` at `/ui/`; a release wheel carries its build |
| `DV_MCP` | the verbs as tools for a language model | a subprocess the model host starts over stdio; an HTTP client of the service, importing no engine |
| Excel add-in | worksheet functions over the service (`excel_integration/`) | xlwings in Excel, from a repository clone |
| `DV_Spine` | the command line over a spine home: mint, verify, checkpoint, follow, seats, policies | run by an operator or a scheduler; never listens |
| `DV_Bloomberg` | builds and re-verifies the Bloomberg security map | on a workstation with a Bloomberg terminal |
| `DV_Docs` | writes this documentation's tree and `mkdocs` configuration | from a repository checkout, with the `docs` extra |
| `DV_Batch`, `DV_Bootstrap` | batch runs over a folder of netting sets; market calibration jobs | command-line jobs spread over worker processes |

### `DV_Service`

One process serves every client. It runs one web worker with one compute thread: jobs queue, run one
at a time, and a base valuation waiting overtakes a simulation waiting, while a running job is never
interrupted. Results, parsed plans and the risk cache are held in memory and are gone when the
process stops; the book and the desk files below are on disk. A running service describes every
route at `/docs` and `/openapi.json`.

| flag | default | effect |
| --- | --- | --- |
| `-b`, `--bind` | `127.0.0.1` | the address to listen on |
| `-p`, `--port` | 8000 | the port |
| `-o`, `--origin` | any origin | a browser origin to allow; repeat for several |
| `-u`, `--ui` | the build the wheel ships | a directory holding a built web UI; refused at start if it holds no `index.html` |
| `-k`, `--book` | `book.json` in `DV_HOME` | the job file served as the live book; a missing file starts as a blank book |
| `--no-book` | off | serve the verbs with no book; every `/book` verb answers 404 |
| `-t`, `--tick [SECONDS]` | off; 30 when given no value | re-mark the book off this machine's Bloomberg session every interval |

`--tick` refuses to start without `blpapi` or beside `--no-book`. A tick runs the job
`POST /book/bloomberg` runs, so it queues behind pricings and writes atomically; it never
provisions a new workstation, skips a beat while the last is still running, and after three failures
in a row stretches its interval fivefold until one succeeds. A key set named in `DV_SPINE_JWKS` that
cannot be read also stops the service at start.

On Windows, a book write replaces the file atomically, which fails while another program holds it
locked. Excel does; keep the book closed there and read it through the UI.

### `DV_MCP`

The binding is an MCP server over stdio, started by the model host:

```
claude mcp add derivus -- DV_MCP
claude mcp add derivus --env RF_SERVICE_URL=http://hub.internal:8000 -- DV_MCP
```

Any host that launches a local stdio server configures the command `DV_MCP` the same way. It reads
`RF_SERVICE_URL` and `DV_SPINE_TOKEN` once, when it starts. The tool contract is on
[MCP Binding](developer/mcp.md).

### The Excel add-in

The add-in is to be replaced: a desk will reach Excel through worksheets the service writes on
demand and the MCP binding and the web UI hand over, as the quote sheet already is
([Roadmap](developer/roadmap.md#designed-not-built)). Until then: the add-in is not in the wheel. It runs from a clone of the repository, with xlwings installed and
the UDF module set to `excel_integration.xlwings_udfs`. Its module imports the engine and
`experiments/run_textbook_hedge.py`, so the Excel machine installs derivus as well. The pricing and
table functions go to the service at `RF_SERVICE_URL`; the solve and portfolio-sheet functions run
the engine inside Excel's Python. Its HTTP client sends no token, so it cannot reach a service that
checks tokens.

### `DV_Spine`

`DV_Spine` works on `--home`, else `DV_SPINE_HOME`, else `~/.derivus_spine`. Each verb prints its
answer as JSON on stdout; a refusal prints the library's sentence on stderr and exits 1.

| verb | what it does |
| --- | --- |
| `init` | mint a home: keys, the genesis grants, the first checkpoint |
| `verify [--chain-only]` | re-derive every hash, link, seal and checkpoint from the bytes on disk |
| `checkpoint`, `status` | sign the head; print the head and whether this home can open its bodies |
| `follow URL` | pull a hub's frames into this home (`--once`, `--blobs`, `--verify`, `--interval`, `--actor`) |
| `seed` | write each fold's state at an official close, so readers start there instead of at genesis |
| `oracle` | check the record's invariants; exit 1 where one fails |
| `enroll`, `grant`, `rewrap`, `name`, `whoami` | seats, the capabilities document, key wraps, display names, a token check |
| `declare`, `policy`, `portfolio` | declare a policy, report those in force, declare a node of the desk's tree |

The verbs' flags are on [The Spine](developer/spine.md#the-cli).

### Bloomberg

`blpapi` comes from Bloomberg's own index, not from PyPI:

```
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

A session is one of two kinds, decided once when it starts. With no `DV_BLOOMBERG_APP_NAME` it is a
Desktop API session on the terminal at `localhost:8194`, which can discover and verify securities.
With an application name it is a B-PIPE session to `DV_BLOOMBERG_HOST:DV_BLOOMBERG_PORT`, a price
feed that cannot discover, falling back to the local terminal if B-PIPE will not connect. The
security map is built on a terminal (`DV_Bloomberg discover`, then `DV_Bloomberg verify` to report
drift) and copied to a B-PIPE host's `DV_HOME`; `DV_Bloomberg discover` reads `seed.json` from
`DV_HOME` and refuses where there is none. The first `POST /book/bloomberg` on a workstation
provisions it: it creates `DV_HOME`, copies the packaged seed in and verifies every candidate
against the terminal. `derivus_bloomberg/README.md` covers the map and the seed.

## Environment variables

| variable | read by | default | what it does |
| --- | --- | --- | --- |
| `DV_HOME` | `DV_Service`, `DV_Bloomberg` | `~/.derivus` | the directory holding a desk's own files ([below](#what-lives-on-disk)) |
| `DV_SPINE_HOME` | `DV_Service`, `DV_Spine` | unset | the spine home to record to, minted by `DV_Spine init`; needs the `enterprise` extra. Unset, the service records nothing and behaves as a plain desk; it never falls back to `DV_Spine`'s default home |
| `DV_SPINE_ACTOR` | `DV_Service`, `DV_Spine init` | unset; `init` uses the account name | the deployment's own seat: the jobs no request signs (the diary read, the tick, the securities verification) run under it, and so does an unnamed request where no capabilities document is in force |
| `DV_SPINE_SEEDS` | `DV_Service` | the home's `seeds/` | the folder readers start their folds from |
| `DV_SPINE_JWKS` | `DV_Service` | unset | the path of a JSON Web Key Set file. Set, every request but `/spine/doorbell` must carry an ID token that set verifies; unset, no token is checked |
| `DV_SPINE_ISSUER` | `DV_Service` | unset | the issuer a token must name; with a key set named, an unset issuer or audience refuses every token |
| `DV_SPINE_AUDIENCE` | `DV_Service` | unset | the audience a token must carry |
| `DV_SPINE_TOKEN` | `DV_MCP`, `DV_Spine follow` | unset | the ID token sent as `Authorization: Bearer`; read once, when the process starts |
| `RF_SERVICE_URL` | `DV_MCP`, the Excel add-in, `ServiceClient` | `http://127.0.0.1:8000` | where the service is |
| `DV_BLOOMBERG_HOST` | Bloomberg sessions | `localhost` | the B-PIPE host |
| `DV_BLOOMBERG_PORT` | Bloomberg sessions | 8194 | the B-PIPE port |
| `DV_BLOOMBERG_APP_NAME` | Bloomberg sessions | unset | the registered B-PIPE application; unset means the Desktop API terminal |
| `DV_BOOTSTRAP_THREADS` | `DV_Bootstrap` | physical cores divided by workers, at least one | threads per calibration worker; also the default for `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` where those are unset |
| `CUDA_VISIBLE_DEVICES` | PyTorch | every device | which GPUs a process sees ([below](#hardware)); `DV_Batch` and `DV_Bootstrap` set it per worker |
| `DERIVUS_FVA_EXCLUSIONS` | `DV_Batch`, job `Legacy_FVA` | none; the job requires it | a CSV of `Netting,Reference` pairs to leave out of the run |
| `DERIVUS_STRESSED_MARKETDATA` | `DV_Batch`, job `SA_CVA` | unset | the folder holding the vega-stressed calibrations |
| `DERIVUS_PATH_TRANSFORM` | `DV_Batch` | `{}` | a JSON object mapping the file roots a job names to local ones |

On a CUDA device the engine also sets `CUBLAS_WORKSPACE_CONFIG` to `:4096:8` for the process, so
its reductions can reproduce.

## What lives on disk

### `DV_HOME`

```
DV_HOME/                      ~/.derivus unless DV_HOME names another
  book.json                   the live book, unless --book names another file
  calculations.json           the desk's named calculations
  xva.json                    the XVA projection: the last run of each netting set
  tmp/<quote_id>.json         a quote: what was quoted, the deal that books it, when
  tmp/<quote_id>.xlsx         its quote sheet, where the quote extra is installed
  seed.json                   the desk's Bloomberg ticker vocabulary
  seed.json.bak_<stamp>       the seed a vocabulary edit replaced
  security_map.json           the securities the terminal verified, with its evidence
```

| file | what it holds | if it is lost |
| --- | --- | --- |
| `book.json` | the book: deals, market, calculation. Without a spine home it is the book of record; with one, the record is the spine and this file is reconciled against it (`GET /book/reconcile`) | restore it: the deal tree lives here in either case |
| files the book names | a market-data file or calendar the book names by path | restore them with the book |
| `calculations.json`, `seed.json` | desk configuration | restore them; without `seed.json` the packaged seed is read |
| `security_map.json` | terminal evidence for each security | rebuilt by `DV_Bloomberg discover` on a terminal |
| `xva.json` | the last XVA run of each netting set | recomputed by `POST /book/xva`, at the cost of the runs |
| `tmp/` | the quotes given | unaccepted quotes are gone; an accepted one is also in the record where a spine home is configured |

### The spine home

```
DV_SPINE_HOME/
  .writer.lock                the writer's claim
  log/segment-00000001.jsonl  the facts, one fsynced line each, rolled past 64 MiB
  blobs/ab/cd/<sha256>        content cited by its hash, written once
  keys/                       blind.key, class_firm.key, checkpoint_signing.key, checkpoint_verify.key
  keys/seats/<id>.key         a seat's private key, where the hub minted it
  names.json                  display names, kept outside the log
  seeds/                      folds cached at a close
```

`log/` and `blobs/` are the record. `keys/` holds its secrets: `class_firm.key` opens every sealed
body, and deleting it destroys their content while the chain still verifies; `checkpoint_signing.key`
signs checkpoints; `blind.key` keys the duplicate tags. `names.json` is mutable and outside the chain.
`seeds/` is derivable and may be deleted: a reader finding no seed folds from genesis to the same
answer. The log and the blobs are append-only, a correction being another fact. The layout and
every hash are on [The Spine](developer/spine.md#the-truth-layer).

## Topology

### One box

`DV_Service` on a workstation, with `DV_HOME`, the clients on the same machine, and optionally the
Bloomberg terminal and a spine home. This is the Getting Started stack.

### A hub and its replicas

- **The hub** is one `DV_Service` process with `DV_SPINE_HOME` set, and the only writer of that home.
  One home belongs to one service process. Desks reach it over HTTP: browsers at `/ui/`, Excel and
  the model hosts through `RF_SERVICE_URL`. A model host runs `DV_MCP` on the desk's own machine.
- **Replicas** are homes on other machines that `DV_Spine follow HUB_URL --home DIR` keeps in step.
  A replica starts as an empty `log/` and `blobs/`. It follows the hub's doorbell, falling back to
  polling, and pulls frames whose bodies stay sealed, so it verifies the chain with no key
  (`DV_Spine verify --chain-only`). To open bodies or pull blobs (`--blobs`, as a seat admitted to
  read), the class key is put into the replica from the seat's wrap with
  `derivus_spine.custody.materialize`, a library call with no `DV_Spine` verb. Nothing is written to
  a replica but what it pulls, and there is no failover: promoting a copy is a restore.
- **Market data.** A Desktop API tick reads the terminal on the machine running `DV_Service`, so a
  hub that ticks from a terminal runs on that workstation; a hub on a server uses B-PIPE with a map
  built on a terminal.

### Who writes what

| what | writer |
| --- | --- |
| the book file | `DV_Service`, by atomic replace. Nothing stops another program writing it; under a spine home, `GET /book/reconcile` names what such a write leaves behind |
| `calculations.json`, `xva.json`, `tmp/` | `DV_Service` |
| `seed.json`, `security_map.json` | `DV_Service` (provisioning, `POST /book/securities`, `POST /book/securities/verify`); `DV_Bloomberg discover` writes the map |
| the hub's `log/` and `blobs/` | the hub's `DV_Service` and the `DV_Spine` verbs run against that home, one append at a time under `.writer.lock`; an act that finds another process writing is refused with `WriterBusy`, to be run again |
| a replica's `log/` and `blobs/` | `DV_Spine follow` alone |
| `keys/` | `DV_Spine init` (the home's keys) and `enroll` (a seat's key, where the hub mints it) |
| `names.json`, `seeds/` | `DV_Spine name`, `DV_Spine seed` |

## Security

**The network.** The service listens on `127.0.0.1:8000` unless `--bind` and `--port` say otherwise,
and it speaks plain HTTP: a deployment reachable from other machines puts a TLS-terminating proxy in
front. Browser origins are unrestricted until `--origin` names the allowed ones.

**Identity.** With no key set named, nothing authenticates a request: a request's `actor` is the
seat it states, which is attribution, not proof. Naming a key set makes the service an OIDC relying
party:

- `DV_SPINE_JWKS` names a JWKS file the deployment provides; the service fetches nothing. The file
  is re-read when it changes on disk, so a key rotation is a replaced file.
- Every request but the doorbell carries `Authorization: Bearer <ID token>`, verified locally:
  RS256 or ES256 only, issuer, audience, authorised party, expiry and not-before, with 60 seconds of
  clock leeway. The token's subject is the seat; a request stating another `actor` is refused, and a
  missing or bad token is a 401.
- Verification needs the `enterprise` extra. `DV_Spine whoami --token --jwks --issuer --audience`
  checks a token from the command line.
- `DV_MCP` and `DV_Spine follow` send `DV_SPINE_TOKEN`, read once, so a process whose token expires
  is restarted with a fresh one. The web UI sends the token held in the browser's local storage
  under the key "derivus.token"; derivus ships no sign-in page, so placing it there is the
  deployment's sign-in flow. The Excel add-in sends none.

**The capabilities document.** Who may do what is a document on the record, declared with
`DV_Spine grant --file`: grants of seven verbs (`validate`, `book`, `approve`, `mark`, `document`,
`settle`, `admin`) at nodes of the desk's tree, and the seats that may read sealed bodies. Until one
is declared, every act is admitted; once one stands, every verb asks it, and an unnamed request is
refused. The genesis seat keeps a break-glass handle from the first event. Under a document,
`DV_SPINE_ACTOR` needs `validate` for the jobs it runs. The verbs and their scope are on
[The Spine](developer/spine.md#identity).

**Key custody.** The hub's keys are files in its home, written readable by the owner alone on POSIX;
on Windows the directory's access control list governs them, and derivus sets none. The class key is
wrapped to each reading seat's public key (`DV_Spine rewrap`), and `DV_Spine enroll --public-key`
keeps a seat's private key off the hub; without it the hub mints the pair and the private half is a
file to hand over and delete. An escrow public key, declared with
`derivus_spine.custody.declare_escrow`, lets a custodian recover the class key.

**What verification proves.** With no key, `DV_Spine verify` proves that the chain links and every
hash recomputes; on a home holding the keys it also checks every seal and that each checkpoint
verifies under the key in force at it. It does not prove the hub did not rewrite its own history:
the hub's signing key is in its home. Publishing the head that `DV_Spine status` prints, or keeping a
replica, gives an outside reference to check against. History after the last checkpoint is chained
but unsigned, so a deployment runs `DV_Spine checkpoint` on a schedule of its choosing, and
`record_time` is the hub's clock with no external timestamp. The full table is
[Trust limits](developer/spine.md#trust-limits).

## Backup and restore

The record is files, so a backup is a copy and a restore is a copy back.

- **What to copy.** The spine home's `log/`, `blobs/`, `keys/` and `names.json`; `DV_HOME`'s
  `book.json`, `calculations.json`, `seed.json`, `security_map.json` and `tmp/`; and every file the
  book names by path. `seeds/`, `xva.json` and the service's in-memory state are rebuilt.
- **When.** Copy a home while no writer runs, with the service stopped, or copy a replica, which
  `DV_Spine follow` keeps byte-identical to the hub's log (with the blobs too where it follows with
  `--blobs`). A line cut off mid-write is the one damage a home repairs itself, truncating it when
  the log is next opened.
- **Keys apart.** `keys/` is the confidentiality of every body: keep its copy under the same custody
  as the hub, separately from the copies of `log/` and `blobs/`.
- **Verify after.** Run `DV_Spine verify --home DIR` on every copy taken and every restore made
  (`--chain-only` where the copy holds no keys), then start the service and read
  `GET /book/reconcile`, which names any trade the book file and the record disagree on.

## Upgrades

The engine's version is the string in `derivus/_version.py`, served by `GET /schema` as
`engine_version` and part of every run's replay tuple, so a run on a new version is a new result
and earlier results keep the version they ran under. A replay of an old tuple reproduces bit for bit
only on the same build, stack and device; across an upgrade, `pin_result` accepts a result the hub
did not witness only by bit-equality or within a declared `tolerance` policy
([replay](developer/spine.md#replay), [lanes](developer/spine.md#lanes)).

- A fold whose version moves refuses an older seed by name; mint new ones with `DV_Spine seed`, or
  let readers fold from genesis.
- A replica chains event types it has never seen, so the hub may be upgraded before its replicas.
- A wheel carries the web UI it was released with, so the UI upgrades with the package.

An upgrade is: stop the service, back up as above, install, `DV_Spine verify`, start, check
`engine_version` on `GET /schema`, and read `GET /book/reconcile`.

## Hardware

Derivus runs on CPU. An NVIDIA GPU, with drivers matching a CUDA build of PyTorch, is strongly
recommended for Monte Carlo work; `torch.cuda.is_available()` says whether the build sees one. A
CPU-only machine installs PyTorch's CPU wheel to avoid the much larger CUDA one
([Requirements](require.md#gpu-notes)).

How a run picks its device:

- `utils.calculation_device` takes the first visible CUDA device where the process sees any, and the
  CPU otherwise. Which devices are visible is `CUDA_VISIBLE_DEVICES`; hiding them all runs on the
  CPU. The device is a property of the machine, never of the job document.
- A base valuation runs in double precision and a credit Monte Carlo in single precision.
- `DV_Service` runs one job at a time on its device.
- In process, `Context.Credit_Monte_Carlo` with `runparallel` shards one simulation over every
  visible GPU, with the same answer for any number of workers. `DV_Batch` gives each worker one GPU;
  `DV_Bootstrap` assigns its workers to the GPUs in turn and divides the CPU cores between them.
- `Deterministic_Kernels: Yes` in a calculation asks PyTorch for deterministic kernels, which a
  bit-identical GPU gradient needs; it warns rather than refuses where a kernel has no deterministic
  form.
