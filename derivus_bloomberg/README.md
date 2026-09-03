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
  `Quoted_Bid`/`Quoted_Ask`/`Timestamp` ride beside it as columns the family **declares** — so a
  re-quoted chain on the same contracts is a value tick (`values_hash` moves, `plan_hash` stands)
  rather than a re-authoring. Under `Premium` the fit reads **no vol surface at all**: a chain is
  calibrated to its own prints, never to somebody's fit to them.
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
- **`quotes_per_expiry` takes the best prints instead of snapping a delta.** A listed chain is
  already quoted, so where the board is deep enough to choose, set it and every expiry a pillar
  claimed carries its ATM — the row the component bootstrap spends on that expiry's `L` pillar —
  plus `n-1` quotes **spanning** the smile: each side's out-of-the-money quotes are banded in
  standardised log-moneyness off that expiry's own ATM implied vol, one band per standard deviation
  and the last one open, and the best print per band by `sqrt(OI) / (1 + spread/cap)` is taken, near
  band first and sides alternating. An empty band falls back to the nearest one that is not, by
  name. **Liquidity chooses the quote; `Weight` is the normalised Black vega alone.** Unset — the
  default — none of this runs and the delta ladder above emits the same bytes it always did.
- **The forward is declared, and so is which curve does which job.** `EquityForward` names the
  curve that **grows** the forward (`funding_rate` — the equity's own repo curve, which
  `utils.calc_eq_forward` integrates), the curve the **premium discounts on** (`discount_rate`) and
  the dividend reference, and carries the numbers the strikes were actually placed with. Leave
  `funding_rate` blank and the two are one curve, which is an index with no borrow spread; the chain's
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

## The rates emitters — a curve strip and a swaption grid

`ir_curve` turns a verified swap strip into one `InterestRatePrices` block, and `swaption_vol`
turns a verified ATM swaption grid into one `HullWhite2FactorModelPrices` block. Neither spells a
ticker: both ask `discover` for its candidates and keep the ones this workstation's map believed,
so the grammar lives in exactly one place. Neither imports the engine — the blocks are emitted as
wire JSON (`{".Timestamp": ...}`, `{".DateOffset": "3M"}`, `{".Percent": 1.45}`), which is what
`Config.read_json` reads and `CustomJsonEncoder` writes.

**The conventions are seed-declared, not guessed.** A par swap rate means nothing without the
accrual it pays on, and nothing in `USOSFR10` says annual/annual ACT/360. So each seeded currency
carries a `conventions` block, the emitters READ it, and a currency without one refuses by name
listing every missing field. A wrong convention is then a **data fix in your seed**, never a code
fix. What ships:

```
"rates": {"USD": {..., "conventions": {
              "curve_day_count": "ACT_365", "spot_days": 2,
              "front": "overnight", "front_day_count": "ACT_360", "authoring": "OIS",
              "fixed_frequency": "1Y", "float_frequency": "1Y",
              "fixed_day_count": "ACT_360", "float_day_count": "ACT_360",
              "notional": 1000000.0, "quote_scale": 1.0}},
          "ZAR": {..., "conventions": {
              "curve_day_count": "ACT_365", "spot_days": 0,
              "front": "fixings/3M", "front_day_count": "ACT_365", "authoring": "Swap",
              "fixed_frequency": "3M", "float_frequency": "3M",
              "fixed_day_count": "ACT_365", "float_day_count": "ACT_365",
              "notional": 1000000.0, "quote_scale": 1.0}}},
"swaption": {"ZAR": {..., "conventions": {
              "fixed_frequency": "3M", "float_frequency": "3M",
              "fixed_day_count": "ACT_365", "float_day_count": "ACT_365",
              "distribution": "Normal", "quote_scale": 0.01, "weight": 1.0}}}
```

- **`front` is which verified entry seeds the short end**, and it is a declaration because the
  wrong answer is free: a SOFR OIS curve's front is the overnight print, a JIBAR-3M curve's front
  is the **3M JIBAR fixing**, and seeding a JIBAR curve with the ZARONIA print sitting beside it in
  the same map is a basis error nothing downstream reports. The other seeded fixings are ledgered
  `not-a-benchmark` rather than silently dropped. It is **validated against the seed's own
  vocabulary** — `overnight` or `fixings/<label>` — because unlike every other convention it is a
  *path*, and a path that names nothing does not fail, it aims somewhere else: `front: "strip/1Y"`
  would author the 1Y par swap as a one-day deposit and call it `overnight` in the Descriptor.
- **`authoring` is the shape.** `OIS` writes a `StructuredDeal` over an OIS-compounded floating leg
  and a fixed leg, **one float item per fixing window** — that is what makes the leg compound
  geometrically rather than average, see the compounding note in `quote_sensitivities.md`. `Swap`
  writes a vanilla `SwapInterestDeal` and the engine generates its legs. Coupons roll **backward**
  from maturity, unadjusted: the deals carry no calendars, so the engine rolls nothing either.
- **The OIS fixing windows partition the coupon**, which is what puts the two legs on one
  convention. Windows start on business days *inside* a coupon, but the coupon's own start is a
  boundary **whatever weekday it falls on** — a fixing accrues through a weekend at a coupon
  boundary exactly as it does inside one. Starting the leg at the first business day instead drops
  real accrual: on the USD 5Y effective 2026-09-02 the coupon starting Saturday 2028-09-02 accrued
  `1.00833333` against the fixed leg's `1.01388889`, two days of a one-year coupon lost on one side
  of a swap the solve holds at PV zero. Per-coupon accrual equality is gated.
- **`float_frequency` is refused where it would not be read.** V1 rolls both OIS legs off one
  schedule, so an `OIS` declaration whose `float_frequency` differs from its `fixed_frequency`
  refuses by name rather than emitting a block that ignores it. The `Swap` path reads both.
- **The quote never enters the deal.** Every rate-carrying field is authored at a neutral zero and
  the print rides in `Quoted_Market_Value`, because `QUOTE_WRITERS` is where the family puts a
  number. That is what makes a value-only re-tick pass `config.update_market_quote` as *updated*
  instead of refusing as a moved plan.
- **Size, stated up front.** An OIS block grows with the *sum* of its strip's tenors: the shipped
  USD strip is about 26,000 authored fixings and **~14 MB** of JSON (measured live). `CurveScreen.
  maximum_fixings` bounds it and refuses with the arithmetic rather than with a MemoryError.
- **The swaption grid quotes NORMAL vols, and since 2026-09-01 the engine prices them as such.**
  `SASN` is `ZAR SWPT NVOL` — basis points of Bachelier vol. `create_market_swaps` used to price
  every benchmark with `utils.black_european_option_price` whatever the surface declared, an order
  of magnitude away from the quote (9.7x–11.4x on the four-quote fixture, which is 1/F); it now
  reads the named `InterestYieldVol`'s `Distribution_Type` through `Factor3D.get_subtype` and
  strikes a Normal ladder's premium with Bachelier. **The declaration this depends on lives on the
  SURFACE, which this emitter does not author** — so a desk pointing `Swaption_Volatility` at a
  lognormally-declared factor still gets a lognormal fit of normal quotes, and `Quote_Source` says
  so in the block. A **zero** vol is still refused from the ladder by name, and the engine refuses
  it too now: it used to be a silent instruction to read the surface's ATM instead.
- **A re-quoted grid is a re-authoring.** `HullWhite2FactorModelPrices` quotes in
  `Instrument_Definitions` rather than `Points`, so `schema.partition_market_price` gives it an
  empty values half and no tick reaches it. `reauthor` drops the block and re-installs it, as
  `POST /book/hn` does. **A curve strip needs it too, for a weaker reason:** that family *does*
  have a values half and a same-day re-tick passes as *updated*, but `Effective_Date` and
  `Maturity_Date` are structure — so the next day's strip of the same benchmarks refuses, rightly,
  and reaches the book through `reauthor`. One function, exported as
  `derivus_bloomberg.reauthor`, reached by both emitters.
- **V1 scope:** a self-discounting single curve (blank `Discount_Rate`), the declared front point
  and the swap strip as seeded. No FRAs, no FX-forward outrights, no cross-currency, no projection
  curve; `ACT_365` and `ACT_360` accruals only. No `/book` verb for either emitter yet.

Never commit Bloomberg values, credentials, or workstation-specific security maps - the map
`DV_Bloomberg discover` writes belongs outside any repo.
