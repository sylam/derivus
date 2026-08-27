# Derivus Bloomberg FX Adapter

`derivus_bloomberg` reads a caller-configured Bloomberg FX volatility snapshot and writes the
existing Derivus `FXVolPrices` market-price block. It does not construct a smile or price a deal;
`FXVolSurfaceParameters` remains the owner of those operations.

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

Never commit Bloomberg values, credentials, or workstation-specific security maps - the map
`DV_Bloomberg discover` writes belongs outside any repo.
