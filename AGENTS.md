# Working on derivus

Read before changing anything. The developer pages are the onboarding; this file is the short list
of rules they elaborate.

## Read first

- `docs_src/developer/architecture.md` — what derivus is, and a table of every developer page with
  when to read it.
- `docs_src/developer/conventions.md` — the house rules, the change scope, what a landing runs.
- `docs_src/developer/roadmap.md` — what is known wrong, waiting on a decision, or designed and not
  built. Check it before proposing work.
- The user pages carry contracts too: `docs_src/json/index.md` (the job and market data files) and
  `docs_src/api_overview.md` (the public verbs).

Each developer page names the code it describes and the test that holds each rule (**Held by**); a
rule held by nothing says **Not gated**.

## Rules

- **JSON is the contract.** A job document is the whole program. No monkeypatching and no reaching
  into internals from a script or a test: a test is a JSON document through the public verbs with
  a defined outcome, and its docstring names the mutation that turns it red.
- **Do not touch** `Credit_Monte_Carlo.execute` (above all its CVA/FVA/CollVA/IM block) or
  `Base_Revaluation.execute`: everything downstream reconciles against them. The one wiring into
  that block is the boundary correction, worth exactly zero forward.
- **Registries, not functions.** A new type is a class or a row the engine already iterates; a
  class's `fields` are its schema; one class per concept, dispatched by name.
- **Look before you write.** Search `utils.py` and the package for an existing primitive; extend a
  `Deal` method with a parameter rather than writing beside it; delete a second spelling.
- **A refactor is proven, not assumed.** Price the documents it reaches before and after and
  compare bit for bit; a number that moves marks an untested part and gets a test, not a revert.
- **A fix lands whole.** It deletes its roadmap row and never adds rows for parts of it.
- **Numerically stable forms** (`expm1`, `log1p`, log-domain tails) and float64 precalculation.
- **Comments say why in at most three lines; docstrings say what the code does.** History, sessions
  and measurements go in the commit message, which is the ledger.
- **No data that is not ours to publish**: no client or book data, no references to untracked specs
  or scratch files. Generated files go under `artifacts/`, which is ignored.

## Verifying a change

```
python gates/reach.py <Symbol>             # what a change reaches: consumers, callers, documents
python gates/impacted.py --dirty --run     # the tests the diff reaches, line by line
python gates/doc_symbols.py                # every name the developer pages cite still exists
python derivus_docs.py && mkdocs build --strict    # the docs, when prose or a declaration moved
```

Run tests with `DV_HOME` on a temporary directory and `DV_SPINE_HOME` unset unless the test sets
its own, and with `CUDA_VISIBLE_DEVICES` naming the card a run may use. A behaviour change updates
the developer page that states the rule, and its **Held by** line.
