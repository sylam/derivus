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
surface. Verify every security and field against OVDV or the desk's authoritative surface.

## Usage

```python
from derivus import Context
from derivus_bloomberg import FXQuoteSecurity, FXVolDefinition, fetch_fx_vol
from derivus_bloomberg.fxvol import update_fx_vol_snapshot
from derivus_bloomberg.session import BloombergSession

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

context = Context().load_json('fx_option_job.json')
with BloombergSession() as bloomberg:
    snapshot = fetch_fx_vol(bloomberg, definition)

update_fx_vol_snapshot(context.current_cfg, snapshot)
context.current_cfg.bootstrap()
```

V1 supports only forward, premium-adjusted delta and the delta-neutral-straddle ATM convention.
Each point receives one UTC retrieval timestamp. Install and refresh validate a complete block
before assigning it; after that assignment, the caller owns `Config.bootstrap()`. If bootstrap
fails, discard or reload that configuration.

Never commit Bloomberg values, credentials, or workstation-specific security maps.