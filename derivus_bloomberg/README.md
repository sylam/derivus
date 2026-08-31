# Derivus Bloomberg Market-Data Adapter

`derivus_bloomberg` reads a caller-configured Bloomberg FX volatility snapshot and writes the
existing Derivus `FXVolPrices` market-price block. It does not construct a smile or price a deal;
`FXVolSurfaceParameters` remains the owner of those operations.

It also reads a listed **equity option chain** and writes a Heston-Nandi quote block
(`equity_chain`, below). The two halves share one discipline and nothing else: FX calibrates off
the desk's own built surface because that surface *is* the market, while an equity's market is the
listed chain — so equities quote **premiums off actual listed contracts**, never implied vols off
somebody's fit.

## Requirements

Live requests require a Bloomberg-enabled workstation with Bloomberg's supported Python `blpapi`
SDK installed and the Desktop API service available. The adapter deliberately has no generic
Bloomberg requirements file because workstation installation is platform-specific. Importing
`derivus` or the adapter's normalization modules does not import `blpapi`.

The caller still owns the map. What the package ships is candidate vocabulary - a starting seed of
tickers to ASK about - AND the machinery that refuses to believe a word of it until your own
terminal answers for it: `DV_Bloomberg discover` probes every seeded candidate against your
workstation and writes a map in which every entry records the `NAME` Bloomberg answered, the
quote's last print, and when it was verified. `security_map.load` refuses an entry missing its
answered name or verification date (the last print may honestly be absent - not every field
carries one), so nothing unverified can reach a fetch - the "verify every security and field
against OVDV" rule enforced by the artifact rather than by discipline. The seed is a
questionnaire, never an answer key: shipping it moves the first hour of typing, not the trust
boundary, which was always the load-bearing rule.

## Discovery

```bash
DV_Bloomberg discover                       # reads DV_HOME/seed.json, writes DV_HOME/security_map.json
DV_Bloomberg verify                         # later: re-probe every entry, report drift, exit 1 on any
```

`DV_HOME` (default `~/.derivus`) is the user-data directory every `DV_*` tool shares — the live
book, the seed, the security map — outside any repo by construction. `--seed`, `--out` and
`--map` name other paths when you want them.

The starting seed SHIPS in the package as `derivus_bloomberg/seed.json`, its vocabulary spelled
on a live terminal (2026-08-27); first use copies it to `DV_HOME/seed.json`, which is yours to cut
down to the scope your desk actually quotes. The `fx_vol` pairs deliberately exclude `EURUSD`,
`GBPUSD`, `AUDUSD` and `NZDUSD`: those are quoted premium-UNADJUSTED and the adapter supports one
convention (see the note at the end), so mapping their vols would verify tickers whose surfaces
the adapter cannot yet honestly write - their spots are still mapped. Its top-level shape:

```
{
 "fx_vol":   {"pairs": [...], "expiries": {"1M": 0.0822, ...}, "pillars": [0.10, 0.25]},
 "fx_spot":  {"pairs": [...]},
 "rates":    {"USD": {"prefix": "USOSFR", "expect": "USD OIS", "weeks": true, "months": true,
                      "years": [...], "overnight": {"security": "SOFRRATE Index", ...}}, ...},
 "swaption": {"ZAR": {"prefix": "SASN", "expect": "ZAR SWPT NVOL",
                      "expiries": {"1Y": "01", ...}, "tenor_years": [1, 2, 3, 4, 5, 7, 10]}}
}
```

What that terminal session established, so the next one starts ahead:

- **The dead trap is real, and the update date is the only defence.** A retired benchmark keeps
  returning a plausible `PX_LAST` with no error: `SAONIA Index` read 8.855 nineteen years after
  its 2007 last print (the live rate is `ZARONIA Index`), `EONIA Index` still answers, and the
  GBP/JPY LIBOR fixings read like rates. Discovery classifies these `dead` off `LAST_UPDATE_DT`;
  `security_map.stale` makes the same check beside a tick, because `fetch_fx_vol` stamps every
  point with the retrieval clock and cannot see it.
- **ZAR has no OIS generic.** `SASWO*` does not exist; the strip is `SASW*`, a JIBAR-3M swap,
  annual 1Y-30Y only - the short end comes from the JIBAR fixings.
- **GBP wants `BPSWS`, not `BPSO`** - identical values where both exist, but only `BPSWS`
  carries the sub-1M points.
- **The legacy IBOR swap families (`EUSWE`, `BPSW`, `CDSW`, `JYSW`, `SFSW`) resolve and never
  price** - discovery classifies them `unpriced`. The live EUR projection strip is `EUSA*`
  (EURIBOR 6M, annual 1Y+).
- **Swaption vol prefixes are `{CCY}SN` (normal, basis points) and `{CCY}SV` (lognormal,
  percent)**; grids are per-entitlement and ragged - discovery records the cells YOUR terminal
  prices and rejects the rest by name.
- **The 5-delta FX pillar resolves on every pair and prices on none**; the 35-delta pillar is
  quoted to 5Y and thins beyond it per pair. Discovery finds your terminal's own extents.

## Usage

Once a map exists, a surface definition comes from it rather than from hand-authored tickers:

```python
from derivus import Context
from derivus_bloomberg import fetch_fx_vol, security_map
from derivus_bloomberg.fxvol import update_fx_vol_snapshot
from derivus_bloomberg.session import BloombergSession

mapped = security_map.load('map.json')
definition = security_map.fx_vol_definition(mapped, 'USDZAR',
                                            expiries=['1M', '3M', '6M', '1Y'], pillars=(0.25,))

context = Context().load_json('fx_option_job.json')
with BloombergSession() as bloomberg:
    late = security_map.stale(bloomberg, [q.security for q in definition.securities.values()])
    if late:
        raise SystemExit('stale quotes, not ticking: {}'.format(late))
    snapshot = fetch_fx_vol(bloomberg, definition)

update_fx_vol_snapshot(context.current_cfg, snapshot)
context.current_cfg.bootstrap()
```

Hand-building an `FXVolDefinition` still works and nothing requires a map - the map is how a
definition carries evidence:

```python
from derivus_bloomberg import FXQuoteSecurity, FXVolDefinition

definition = FXVolDefinition(
    pair='USDZAR',
    surface_name='USD.ZAR',
    currency='ZAR',
    expiries={'1M': 1.0 / 12.0},
    pillars=(0.25,),
    quote_scale=0.01,
    securities={
        ('1M', 'ATM', None): FXQuoteSecurity('<verified ATM security>', 'PX_LAST'),
        ('1M', 'RR', 0.25): FXQuoteSecurity('<verified 25D RR security>', 'PX_LAST'),
        ('1M', 'BF', 0.25): FXQuoteSecurity('<verified 25D BF security>', 'PX_LAST'),
    },
)
```

V1 supports only forward, premium-adjusted delta and the delta-neutral-straddle ATM convention.
Note the convention consequence: premium adjustment matches market practice where the premium is
paid in the base currency (`USDJPY`, `USDCHF`, `USDCAD`, `USDZAR`), while `EURUSD`, `GBPUSD`,
`AUDUSD` and `NZDUSD` are quoted premium-unadjusted - surfaces for those four are outside what
this adapter can honestly write today. Each point receives one UTC retrieval timestamp. Install
and refresh validate a complete block before assigning it; after that assignment, the caller owns
`Config.bootstrap()`. If bootstrap fails, discard or reload that configuration.

## The equity option chain

`equity_chain` turns an index's listed chain into one `HestonNandiComponentModelPrices` (or
`HestonNandiModelPrices`) block. It reaches the terminal twice — the underlying and its `OPT_CHAIN`
membership through `BloombergSession.bulk_reference_data_report`, then every member in batches
through the tolerant scalar reader — and spells no ticker of its own: a listed chain's membership
is the terminal's to state, so the trust boundary is the **screen** rather than a grammar.

```python
from derivus_bloomberg.equity_chain import EquityForward, EquityLadder, equity_hn_block, \
    fetch_equity_chain
from derivus_bloomberg.session import BloombergSession

with BloombergSession(timeout_ms=30000) as bloomberg:
    chain = fetch_equity_chain(bloomberg, 'SPX Index', datetime.date.today())

name, block = equity_hn_block(chain, EquityForward(
    underlying_factor='SPX', volatility_factor='SPX', discount_rate='USD',
    dividend_reference='SPX', rate=0.04, dividend_yield=0.015))
```

- **Premiums, not implied vols.** A listed price is a print; its implied vol is a convention.
  `Quote_Type` is `Premium`, `Quoted_Market_Value` is the terminal's own two-way mid, and
  `Quoted_Bid`/`Quoted_Ask`/`Timestamp` ride beside it.
- **Every answer is screened, in an order of distrust**, and every refusal lands on a ledger by
  name: `malformed`, `expired`, `unstated-exercise`, `american`, `unpriced`, `one-sided`,
  `crossed`, `off-market`, `wide`, `no-open-interest`, `undated`, `stale`. Half of any index chain
  is dead strikes, and this is where they stop. The **spot** is screened on the same two questions
  every contract is — a blank price and an absent or stale `LAST_UPDATE_DT` both refuse by name —
  because it is the number every strike, forward and weight in the block hangs off.
- **Indices only.** An American premium is not the European premium the fit prices against. A mixed
  board drops its American listings per contract and calibrates on what is left; a chain whose
  **census** says exercise style is what killed the ladder refuses by name with the remedy, rather
  than reporting the distinct-contract floor about a chain that is quoted at plenty of both.
- **The ladder reaches the product horizon** — 3M/6M/1Y/2Y/3Y ATM rungs plus 25-delta-equivalent
  wings at the first four — because equity autocalls run three to five years and a multi-year ATM
  term structure is what the component family's `L` curve is for. Every rung **snaps to a listed
  contract**; pillars and listed expiries are matched **one to one** (nearest claim wins, and a
  pillar left with nothing is dropped by name), so a board with a missing LEAP cannot put one
  contract into the block as two equations or write an `L` strip short of a knot. Two rungs that
  do land on one contract are emitted **once at their summed weight** — a repeated contract is a
  weight, not a second equation — and a ladder that collapses below eight distinct contracts
  refuses, naming the chain's own expiries.
- **The weight is `vega x sqrt(open interest) / (1 + spread/cap)`, normalised** — liquidity joins
  the vega weight, because a dead strike is not evidence.
- **The forward is declared.** `EquityForward` names the curve that funds the carry and the
  dividend reference, and carries the numbers the strikes were actually placed with; the chain's
  own parity-implied dividend yield is measured beside the declared one and reported in
  `Quote_Source` rather than averaged into it. Declare nothing and the chain's own carry is used —
  and then it is *screened*: a **median** over the strikes nearest the forward, so one fat-fingered
  print cannot move a pillar, inside a declared `parity_band`, so a whole bad neighbourhood refuses
  by name instead of placing the ladder.

The module imports the standard library and this package's own `errors` — no engine, no pandas, no
blpapi at import time — and `tests/test_equity_chain.py` holds it there, in a fresh interpreter as
well as in the source. That is also why this package re-exports the pandas-carrying FX names
lazily: importing a submodule imports its package first, so an eager `from .fxvol import ...` would
land pandas on a workstation that only wanted to read a listed chain.

Never commit Bloomberg values, credentials, or workstation-specific security maps - the map
`DV_Bloomberg discover` writes belongs outside any repo.
