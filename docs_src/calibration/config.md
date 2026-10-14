# Calibration Config

`calibration_config.json` declares the historical archive location and the calibration
method to use for each stochastic process class. It is loaded into the same `Config`
object as the MarketData JSON via a second `parse_json` call.

```json
{
  "CalibrationConfig": {
    "MarketDataArchiveFile": {
      "name": "./data/plat_archive.csv",
      "skiprows": 0,
      "index_column": 0,
      "sep": ","
    },
    "Calibrations": {
      "MarkovHMMSpotModel": {
        "ID": "",
        "Method": "MarkovHMMSpotCalibration",
        "N_States": 3,
        "Use_Student_T": "Yes",
        "Log_Price": "Yes"
      },
      "QuadraticCarryCurveModel": {
        "ID": "",
        "Method": "QuadraticCarryCurveCalibration"
      },
      "BasisLinkedSpotModel": {
        "ID": "",
        "Method": "BasisLinkedSpotCalibration",
        "Vol_Window": 21
      }
    }
  }
}
```

## `MarketDataArchiveFile`

Describes how to load the historical archive into a pandas DataFrame.

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Path to the archive file |
| `skiprows` | Yes | Rows to skip at the top of the file, usually 0 |
| `index_column` | Yes | Column (position or name) holding the dates |
| `sep` | No | Field separator, default `\t`; set `,` for CSV |

A missing required key raises `KeyError` at `parse_json`. The archive's index is normalised to
Excel-offset integers at load time; date strings in the index column (e.g. `2009/11/10`) are
parsed and converted. Column headers are `<Type>.<Name>` or `<Type>.<Name>,<sub-key>`; a numeric
sub-key is a tenor (see [Cross Factor](cross_factor.md)).

## `Calibrations`

A dict mapping **stochastic process class name** (the model in `Price Models.<Model>.<Factor>`)
to its calibration parameters. Each entry must contain:

- `Method` — the calibration class name (resolved via `globals()` in `stochasticprocess.py`)
- Any model-specific tuning knobs (clamps, EM iteration caps, RNG seeds, etc.)

A class reads only the tuning keys it declares in its `fields`; each estimator's section in this
chapter lists them with their defaults. Every other key is accepted and ignored without a message:
the shipped `tests/fixtures/calibration_config.json` carries `ID`, `Use_Pre_Computed_Statistics`,
`Data_Retrieval_Parameters`, `MLE_Parameters`, `Number_Of_PCA_Factors` and, under
`LogOUSpotModel`, `Kappa_Max`, `Sigma_Max` and `Vol_Shift`, none of which is read. A `Method` that
names no class in `stochasticprocess.py` (a misspelling or a retired estimator) fails at
`parse_json` with the unexplained `TypeError: 'NoneType' object is not callable`.

Calibration is keyed by **model class**, not by factor instance. All factors routed to the
same model class share the same calibration config. Per-factor configuration that depends
on data (e.g. number of states, Student-t flag) lives in this block; per-factor
configuration that depends on the factor (e.g. linked sibling factor) flows through the
[archive subkey convention](cross_factor.md), not through this config.

Calibration classes that don't need any tuning can have a minimal entry containing just
`ID` and `Method`. A class omitted from this block is **not** filtered out up front: the factor
still enters the map with `calibration=None` and is skipped downstream, named as
*"… is not calibrated and is skipped — CalibrationConfig names no Method for …"*; every other skip
is named in the calibration class's own words. A factor skipped for any reason writes no `Price
Models` entry, and a set where every factor is skipped is refused rather than reported empty. It
only reaches that far if the factor has archive columns at all; otherwise the unguarded
`self.archive_columns[…]` subscript in `calibrate_factors` raises `KeyError` first.
