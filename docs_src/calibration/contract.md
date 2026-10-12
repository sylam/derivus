# Calibration Class Contract

A calibration class is defined in `derivus/stochasticprocess.py` and carries:

| Member | Read by | Contract |
|---|---|---|
| `model_type` | `schema.emit_calibration` | The process class it calibrates, which is the key of its `Calibrations` entry |
| `fields` | `schema.emit_calibration` | A list of `schema.F` declaring every tuning key it reads; `[]` for none |
| `documentation_calibration` | the documentation build | Optional `(page, [markdown lines])`; the page is one of this chapter's |
| `__init__(self, model, param)` | `construct_calibration_config` | `model` is the process name, `param` its `Calibrations` entry |
| `self.num_factors` | `calibrate_factors` | The number of rows of the `correlation` it returns |
| `calibrate(...)` | `calibrate_factors` | Returns `utils.CalibrationInfo(param, correlation, delta)` |

```python
def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
    ...
    return utils.CalibrationInfo(param, correlation, delta)
```

The process it writes must also accept the written `param` at construction with `factor=None`,
and its `correlation_name` must list one sub-factor per row of `correlation`: the framework
constructs it to name the correlation rows, outside the per-factor guard, so a constructor that
refuses the parameters aborts the run ([Skips and refusals](general.md#skips-and-refusals)).

## Inputs

| Argument | Description |
|---|---|
| `data_frame` | The factor's archive columns, plus any related-factor columns the framework auto-pulled via the [subkey convention](cross_factor.md) |
| `vol_shift` | An additive volatility shift, passed through from `calibrate_factors(vol_shift=...)` (default 0.0). Only `LogOUSpotCalibration` reads it, adding it to $\hat\sigma$ before its clip |
| `num_business_days` | Days-per-year for annualisation. Default `252.0` |

The framework calls exactly that and passes nothing else —
`rate_value.calibration.calibrate(data_frame, vol_shift, num_business_days=252.0)`. Declaring
`**kwargs` is optional future-proofing that **no** shipped class takes, so a framework-passed
keyword would raise `TypeError` across all fourteen at once rather than being ignored. Two classes
add tuning positionals of their own with defaults, so the signature is not verbatim across the
family either.

The class receives `param` at construction time (the
[calibration_config](config.md) entry for its model class). Tuning knobs come from there, and the
class DECLARES them: a `fields` list of `schema.F` descriptors, which is what
`mapping['Calibration']['types'][<model>]` publishes and what a gate holds to the keys the class
actually reads, in both directions. A class that takes no tuning declares an empty list.

## Output: `CalibrationInfo`

A namedtuple with three fields. All three are load-bearing:

### `param`

A plain dict that becomes the JSON content of `Price Models.<ModelClass>.<FactorName>`.
Its shape must match what the simulator's process class reads at `precalculate` time —
the calibration writes, the simulator reads, no schema validation in between, so
consistency is on the author.

```python
param = {
    'A': float(a_hat),
    'Phi': float(phi_hat),
    'Sigma_By_State': sigma_by_state_list,
    'Calibration_DT_Years': 1.0 / num_business_days,
}
```

### `correlation`

A `(num_factors, num_delta_columns)` matrix that maps the columns of `delta` to the factor's
primitive Gaussian factors (declared via `correlation_name` on the process class). A
single-column estimator returns `[[1.0]]`; `QuadraticCarryCurveCalibration` and
`LogVar2FJCalibration` return the identity; the one-factor curve estimators return an equal-weight
row over their tenors and `PCAInterestRateCalibration` a per-component loadings matrix (see their
sections).

The framework uses this matrix during the consolidate-and-correlate step:

```
factor_correlations = a · ρ · aᵀ
```

where `a` is the block-diagonal stack of all factors' `correlation` matrices and `ρ` is
the empirical Pearson correlation of all `delta` columns concatenated.

### `delta`

A `pandas.DataFrame` of innovations, one column per column of the `correlation` matrix, indexed by the date each return STARTS on (`start_of_return`). The framework concatenates
all factors' `delta` columns and computes pairwise Pearson correlations across them; these become
the entries in the global `Correlations` block (subject to the absolute-value cutoff).

Pearson correlation is scale-free, so a column need not be standardised; it should be
approximately serially independent and homoskedastic under the calibrated model. What the
shipped classes emit:

- **Regime-switching HMM:** `(diff_t - μ_state(t)) / σ_state(t)` under the decoded regime.
- **GARCH:** $r_t/\sqrt{h_t}$, the return over its filtered conditional sd.
- **The shared OU regression:** the raw difference $\Delta x_t$, with no reversion term removed.
- **Two-equation and four-factor fits:** one standardised residual per primitive factor.

If `delta` columns are not approximately iid, cross-factor correlation estimates will be
contaminated by within-factor structure (autocorrelation, heteroskedasticity, regime
dependence). The framework's correlation cutoff filters out small spurious entries but
won't repair systematic bias.

## Registration

Calibration classes are resolved by string lookup at config load time:

```python
def construct_calibration_config(calibration_model, param):
    return globals().get(param['Method'])(calibration_model, param)
```

A `Method` naming no class is not refused by name: `globals().get` returns `None` and
`parse_json` fails with `TypeError: 'NoneType' object is not callable`.

To add a new calibration class:

1. Define the class in `derivus/stochasticprocess.py` — there is no "or wherever": the `globals()`
   dispatch above and `schema.emit_calibration(stochasticprocess)` both read that module and no
   other, so a class defined elsewhere is neither dispatchable nor published.
2. Give it a `model_type` naming the process it calibrates and a `fields` list declaring its
   tuning keys. The `Calibration` schema is emitted from those, keyed by `model_type`, with
   `Method` stamped from the class name.
3. Add an entry under `Calibrations` in `calibration_config.json` keyed by the model class, with
   `Method` set to the calibration class name. Tuning parameters go in the same entry.
4. Give the model class a `fields` list and a `factor_types` tuple naming the price factors it
   drives. `Process_factor_map` and the `Price Models` schema are both emitted from those.

No other registration is needed — the framework picks up the new method automatically.

## Independence requirement

Calibration classes operate on the data passed to `calibrate()`. They do **not** reach
into other factors' archives, calibrated parameters, or process instances. If a
calibration needs information from a sibling factor, the
[archive subkey convention](cross_factor.md) brings that data into `data_frame` and the
class stays self-contained.

Avoiding cross-class coupling keeps each calibration:

- **Reproducible** — same `data_frame` in, same params out, regardless of run order.
- **Testable** — fits run on isolated data slices without setting up the full Config.
- **Stable under refactor** — renaming or restructuring sibling factors doesn't ripple
  through unrelated calibration classes.
