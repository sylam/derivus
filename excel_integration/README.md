# Excel + xlwings add-in

A free Excel front end for Derivus, and the first real client of `DV_Service`. The workbook talks
HTTP; it does not import the engine.

That is the whole design. There used to be a local worker and a pluggable queue in this folder —
`worker.py` and `queue_clients.py` — because something had to order the work and hold the results.
The service does both now: one compute lane, a cost-class priority queue, and a result store keyed
by the content hash of the replay tuple. Both modules are gone rather than deprecated.

## Install and run

```bash
pip install 'derivus[service]' xlwings requests
DV_Service --port 8000
```

Point the workbook at it (defaults to `http://127.0.0.1:8000`):

```bash
RF_SERVICE_URL=http://my-host:8000
```

Then in the xlwings add-in config, set the UDF module to `excel_integration.xlwings_udfs`.

The service has no authentication and open CORS: it is a **trusted-network** deployment. See
[the API overview](../docs_src/api_overview.md#the-same-verbs-over-http) for the endpoints
themselves, which are the same ones a browser SPA or a marimo notebook calls.

## Worksheet functions

| | |
|---|---|
| `RF_PRICE_JSON(job_json, patch_json)` | submit a job document; returns the `result_id` |
| `RF_PRICE_PATH(job_path, patch_json)` | the same, reading the document from a file |
| `RF_PROCESS_NEXT_REQUEST()` | poll the outstanding submission (button / macro) |
| `RF_GET_LAST_RESULT(field)` | a field of the last polled summary — `status`, `plan_hash`, … |
| `RF_GET_TABLE(table, offset, limit)` | one table of that result, spilling down and across |
| `RF_SOLVE_JSON` / `RF_SOLVE_PATH` | goal-seek on a deal field — in process, see below |

Every submission answers with a `result_id` and a status; there is no sync/async split. A base
valuation is `done` on the first press of the button, a simulation says `running` until it is not.
So the flow is always the same three steps:

```
=RF_PRICE_PATH("C:\jobs\Trade01.json", "")     ' submit, cell holds the result_id
RunPython("import excel_integration.xlwings_udfs as rf; rf.RF_PROCESS_NEXT_REQUEST()")
=RF_GET_TABLE("mtm")                            ' the numbers, once status is done
```

`patch_json` is a market **values** delta, exactly what `/execute` takes:

```json
{"FxRate.ZAR": {"Spot": 19.0}}
```

A finished result publishes the shape of each table, never the cells — `RF_GET_TABLE` fetches one,
and `offset` / `limit` page it. A group of tables (`cashflows`, `scenarios`) is named by its path,
so `=RF_GET_TABLE("cashflows/ZAR")`.

## Solve mode stays in process

`RF_SOLVE_JSON` / `RF_SOLVE_PATH` iterate a pricing run on one changing **deal** field (a strike, a
margin) through `scipy.optimize`, which is the STRUCTURING calculation the roadmap has yet to build.
A deal field is structural today, so every iterate is a fresh document and a fresh compile: pushing
the loop through HTTP would add a round trip per iterate and buy nothing. When the structuring calc
lands it becomes a `Calculation.Object` like any other, submitted through the same `/execute`.

The solve spec is unchanged:

```json
{
  "method": "brentq",
  "variables": [{"name": "strike", "path": "/Calc/Deals/Deals/Children/0/Instrument/field/Strike",
                 "initial": 100, "lower": 50, "upper": 200}],
  "targets":   [{"name": "net_mtm", "metric": "net_mtm", "target": 0.0, "weight": 1.0}]
}
```

- `variables[*].path` is a JSON Pointer into the job document; `lower` / `upper` bound the solver.
- `targets[*]` takes either `metric: "net_mtm"` (top portfolio MTM) or `path:` into the priced
  response (dot / bracket syntax).
- `brentq` is one variable and one target and needs a sign change across the bounds;
  `least_squares` handles several of either.

## Portfolio sheets

`RF_LOAD_PORTFOLIO`, `RF_SAVE_PORTFOLIO`, `RF_PRICE_PORTFOLIO` and `RF_SOLVE_PORTFOLIO` build a job
from the Portfolio / RiskFactors / Calculations sheets, and they are still in process. They go
through `portfolio_service.py`, which reads `derivus.fields.mapping` directly for the deal-type
menus and field defaults. Migrating that to `GET /schema` — which publishes exactly those
declarations — is the remaining end-state for this folder: after it, nothing here imports the
engine and the add-in installs without it.

## Where the client lives

`service_client.py` is the HTTP binding, and it imports neither `xlwings` nor `derivus` — a marimo
notebook or a plain script uses it exactly as the add-in does:

```python
from excel_integration.service_client import ServiceClient

client = ServiceClient()
plan = client.prepare(open('Trade01.json').read())
run = client.submit({'plan_id': plan['plan_id']}, {'FxRate.ZAR': {'Spot': 19.0}})
client.poll(run['result_id'])
client.fetch_table(run['result_id'], 'mtm', offset=0, limit=100)
```

`xlwings_udfs.py` is deliberately thin over it: reading a cell and writing a cell is all it does,
because it cannot be imported without Excel installed and so cannot be tested. Everything worth
gating is in `service_client.py`, against the real app — see `tests/test_service_client.py`.

Solace returns later as a second **transport** in front of the same verbs, per the roadmap; it is
not a second queue, and nothing in this folder waits for it.
