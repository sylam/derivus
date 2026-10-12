# General JSON Datatypes

Every Derivus document is JSON (RFC 8259 [@rfc8259]): objects, arrays, strings, numbers, `true`/`false` and
`null`. On top of that, an object keyed by a name starting with a period is a **tag**: the loader
converts the value it holds into the engine object the tag names.

| Tag | Value | Becomes |
|---|---|---|
| `.Timestamp` | an ISO 8601 date [@iso8601], `"YYYY-MM-DD"`, or date-time | a date (`pandas.Timestamp`) |
| `.DateOffset` | a period string, e.g. `"3M"`, `"1Y6M"` (below) | a calendar period |
| `.DateOffset` with `.Offset` | two periods in one object | a grid item: a date and the step it repeats by |
| `.Grid` | an array of `.DateOffset` objects, with or without `.Offset` | a time grid (below) |
| `.Curve` | `{"meta": [...], "data": [[...], ...]}` | a curve: `data` rows of (tenor in years, value), sorted on load; a surface or space is a `.Curve` whose rows carry three or four numbers in the order its factor type reads them, `meta` saying how it is held (`[]` for a plain curve) |
| `.Percent` | a number | that number of percent: `{".Percent": 21.5}` is 0.215 |
| `.Basis` | a number | that number of basis points |
| `.DateList` | an array of `[date, value]` pairs | a dated list of values |
| `.DateEqualList` | an array of `[date, value, ...]` rows | a dated list of rows |
| `.CreditSupportList` | an array of `[rating, amount]` pairs | a credit-support table keyed by rating |
| `.Descriptor` | an array | the values, stored as given |
| `.ModelParams` | `{"modeldefaults": {...}, "modelfilters": {...}}` | a rule set (see [Model Configuration](model_configuration.md)) |
| `.Deal` | a deal block (see [Trade Data](trade_data.md)) | the deal its `Object` names |

## Periods

A period is one or more `<integer><unit>` terms, the unit one of `D` (days), `W` (weeks), `M`
(months) or `Y` (years), any case: `3M2D` is three months and two days. Periods are calendar
offsets, with no business-day adjustment. Time in years (curve tenors, `Expiry`) is a plain number.

## Time grids {#time-grids}

A grid is written either as `.Grid` or as text, the form a `Calculation`'s `Time_Grid` takes:

```
0d 2d 1w(1w) 3m(1m) 2y(3m)
```

Items are separated by single spaces. Each item `p` dates `base + p`; an item `p(s)` also repeats
every `s` from that date up to the next item's date, the last item up to the end the calculation
sets (see [Calculations](calculations.md#grids)). The grid above dates today, two days, then weekly
from one week to three months, monthly to two years, and quarterly after that. The same grid as
`.Grid`:

```json
{".Grid": [{".DateOffset": "0d"}, {".DateOffset": "2d"},
           {".DateOffset": "1w", ".Offset": "1w"}, {".DateOffset": "3m", ".Offset": "1m"},
           {".DateOffset": "2y", ".Offset": "3m"}]}
```

## Day counts {#day-counts}

The day-count menus (`Day_Count` and the like) offer six names. Two of them do not yet accrue the
convention they are named after:

| Name | Year fraction the code computes | Named after [@isda2006] |
|---|---|---|
| `ACT_365` | days / 365 | Actual/365 (Fixed) |
| `ACT_360` | days / 360 | Actual/360 |
| `ACT_365_ISDA` | days / 365, with no split at year ends | Actual/Actual (ISDA) |
| `ACT_ACT_ICMA` | days / 365, with no coupon-period denominator | Actual/Actual (ICMA) |
| `_30_360` | $(360(Y_2-Y_1) + 30(M_2-M_1) + D_2 - D_1)/360$, $D_1 = \min(d_1, 30)$, $D_2 = 30$ if $d_2 \ge 30$ and $d_1 \ge 30$, else $d_2$ | 30/360 (Bond Basis) |
| `_30E_360` | the same with $D_2 = \min(d_2, 30)$ | 30E/360 (Eurobond Basis) |

For the two 30/360 conventions, a schedule accrued several periods at once takes every period's
$d_1$ from the first period's start date.
