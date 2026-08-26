# MCP Binding

`mcp_integration/server.py` is the derivus verbs as MCP tools, for a model to book instruments in
plain language. It is the third client of the service — the web UI and the Excel add-in are the
other two — and it owns **no logic**: every tool is a thin adapter onto a `DV_Service` endpoint,
so anything a tool needs that an endpoint cannot answer is a missing verb on the service, never
code in this layer. It lives outside the `derivus` package on purpose: importing any of the
package pulls the whole engine (torch included, ~3s) into a process that only talks HTTP, and the
import gate in `tests/test_mcp.py` holds the module to `requests` + `mcp` and nothing else.

## Running it

```
pip install -r mcp_integration/requirements.txt          # mcp needs python 3.10+
DV_Service --book path/to/job.json --ui web/dist &       # the service does the work
claude mcp add derivus -- python mcp_integration/server.py
# a service somewhere else:
claude mcp add derivus --env RF_SERVICE_URL=http://host:8000 -- python mcp_integration/server.py
```

`RF_SERVICE_URL` is the same variable the Excel add-in reads — one setting configures every
client. There is deliberately no tracked `.mcp.json`: it would pin one machine's paths into the
repo.

## The tools

| | |
| --- | --- |
| `list_instrument_types` | every bookable type, the create-menu grouping, and `containers` |
| `describe_instrument_type` | one type's fields as declared — required, defaults, valid values |
| `describe_calculation_type` / `describe_factor_type` | the same for calculations and factors |
| `job_skeleton` | the envelope, as a job that loads |
| `read_book` / `read_deal` | the live book summarised per deal; one deal verbatim |
| `book_deal` / `delete_deal` | write verbs onto `POST /book/deals` |
| `price_candidate` / `execute_book` | `POST /book/price` — the what-if; waits, then hands back the id |
| `validate_book` / `describe_book` | the read verbs over the live document |
| `poll_result` / `fetch_table` / `deal_values` | results: status, one paged table, `{reference: value}` |

## The contracts that matter

**Validate-before-write.** `book_deal` never writes a deal something is said against — its own
authoring messages, or market data the book did not already lack. A refusal is a **normal return**
(`{written: false, refused: [...]}`), not a tool error, because the model's next move is to read
the messages and fix exactly what they name. Tool errors are reserved for *cannot proceed*: the
service down (named, with how to start it), an unknown type (with close matches), a parent that
takes no children.

**Deals are addressed positionally.** `deal_path` (`"0/2/1"`) is the identity everywhere —
the same one the web UI's tree uses — because references are not unique in a book.

**Par and margin are a solve, not a verb.** A linear payoff's value is affine in its amount, so
two `price_candidate` calls at trial amounts give the exact amount that lands the value on a
target — then `book_deal` the answer. The tool docstrings teach the model this pattern.

**Market data is not editable here**, by decision: the structural/value split
(`schema.partition_factor`) is engine-side, and a wrong structural edit silently changes the plan
— a wrong number, not a failure. The safe half already travels as the values `Patch` on execute;
when market-data editing lands it lands behind a `patch_market`-shaped verb.

## Testing

`tests/test_mcp.py` drives the tool functions directly against the in-process service
(`configure(session=TestClient(service.app))` — the same seam the Excel client uses), so the gates
run with no stdio and no sockets: the import discipline, the registry's contracts and read-only
hints, schema tools equal to the declarations, a booking that prices to the closed form, a
refusal that writes nothing, and the byte-identical book-then-delete round trip.
