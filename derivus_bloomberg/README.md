# Derivus Bloomberg FX Adapter

`derivus_bloomberg` reads a caller-configured Bloomberg FX volatility snapshot and writes the
existing Derivus `FXVolPrices` market-price block. It does not construct a smile or price a deal;
`FXVolSurfaceParameters` remains the owner of those operations.

## Requirements

Live requests require a Bloomberg-enabled workstation with Bloomberg's supported Python `blpapi`
SDK installed and the Desktop API service available. The adapter deliberately has no generic
Bloomberg requirements file because workstation installation is platform-specific. Importing
`derivus` or the adapter's normalization modules does not import `blpapi`.

The caller supplies the complete Bloomberg map. The package ships no ticker templates and no desk
surface - what it ships is the machinery to BUILD one and hold it honest: `DV_Bloomberg discover`
probes a seed vocabulary you own against your own terminal and writes a map in which every entry
records the `NAME` Bloomberg answered, the quote's last print, and when it was verified.
`security_map.load` refuses an entry missing its answered name or verification date (the last
print may honestly be absent - not every field carries one), so nothing unverified can reach a
fetch - the "verify every security and field against OVDV" rule enforced by the artifact rather
than by discipline.

## Discovery

```bash
DV_Bloomberg discover --seed seed.json --out C:\somewhere\outside\any\repo\map.json
DV_Bloomberg verify --map map.json          # later: re-probe every entry, report drift, exit 1 on any
```

A starting seed, with the vocabulary verified on a live terminal (2026-08-27) - copy it, then cut
it to the scope your desk actually quotes. The `fx_vol` pairs deliberately exclude `EURUSD`,
`GBPUSD`, `AUDUSD` and `NZDUSD`: those are quoted premium-UNADJUSTED and the adapter supports one
convention (see the note at the end), so mapping their vols would verify tickers whose surfaces
the adapter cannot yet honestly write - their spots are still mapped:

```json
{
 "fx_vol": {
  "pairs": ["USDJPY", "USDCHF", "USDCAD", "USDZAR", "EURZAR", "GBPZAR"],
  "expiries": {"1W": 0.0192, "2W": 0.0384, "1M": 0.0822, "2M": 0.1671, "3M": 0.2493,
               "6M": 0.4986, "9M": 0.7479, "1Y": 1.0, "2Y": 2.0},
  "pillars": [0.10, 0.25]
 },
 "fx_spot": {"pairs": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD",
                       "USDZAR", "EURGBP", "EURJPY", "EURCHF", "AUDJPY", "EURAUD", "GBPJPY",
                       "EURZAR", "GBPZAR"]},
 "rates": {
  "USD": {"prefix": "USOSFR", "expect": "USD OIS", "weeks": true, "months": true,
          "years": [1, 2, 3, 5, 7, 10, 15, 20, 30],
          "overnight": {"security": "SOFRRATE Index", "expect": "SOFR"}},
  "EUR": {"prefix": "EESWE", "expect": "EUR SWAP (ESTR)", "weeks": true, "months": true,
          "years": [1, 2, 3, 5, 7, 10, 15, 20, 30],
          "overnight": {"security": "ESTRON Index", "expect": "ESTR"}},
  "GBP": {"prefix": "BPSWS", "expect": "GBP SWAP (vs SONIA)", "weeks": true, "months": true,
          "years": [1, 2, 3, 5, 7, 10, 15, 20, 30],
          "overnight": {"security": "SONIO Index", "expect": "SONIA"}},
  "JPY": {"prefix": "JYSO", "expect": "JPY SWAP OIS", "weeks": true, "months": true,
          "years": [1, 2, 5, 10, 20, 30],
          "overnight": {"security": "MUTKCALM Index", "expect": "Bank of Japan"}},
  "CHF": {"prefix": "SFSNT", "expect": "CHF SARON", "weeks": true, "months": true,
          "years": [1, 2, 5, 10, 20, 30],
          "overnight": {"security": "SRFXON1 Index", "expect": "SARON"}},
  "CAD": {"prefix": "CDSO", "expect": "CAD SWAP OIS", "weeks": true, "months": true,
          "years": [1, 2, 5, 10, 20, 30],
          "overnight": {"security": "CAONREPO Index", "expect": "Canadian Overnight"}},
  "AUD": {"prefix": "ADSO", "expect": "AUD SWAP OIS", "weeks": true, "months": true,
          "years": [1, 2, 5, 10, 20, 30],
          "overnight": {"security": "RBACOR Index", "expect": "RBA"}},
  "ZAR": {"prefix": "SASW", "expect": "ZAR SWAP QTR", "years": [1, 2, 3, 5, 7, 10, 15, 20, 30],
          "overnight": {"security": "ZARONIA Index", "expect": "South African Overnight"},
          "fixings": {"1M": {"security": "JIBA1M Index", "expect": "Johannesburg"},
                      "3M": {"security": "JIBA3M Index", "expect": "Johannesburg"},
                      "6M": {"security": "JIBA6M Index", "expect": "Johannesburg"},
                      "12M": {"security": "JIBA12M Index", "expect": "Johannesburg"}}}
 },
 "swaption": {
  "ZAR": {"prefix": "SASN", "expect": "ZAR SWPT NVOL",
          "expiries": {"1M": "0A", "3M": "0C", "6M": "0F", "9M": "0I",
                       "1Y": "01", "2Y": "02", "5Y": "05", "7Y": "07", "10Y": "10"},
          "tenor_years": [1, 2, 3, 4, 5, 7, 10]}
 }
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
