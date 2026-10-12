# Trade Data

The book is a tree of deals under `Deals.Children`. A container deal (a netting set, a structure)
holds its own `Children`. As an example, an **FXNonDeliverableForward** inside an uncollateralised
netting set:

```json
{
  "Deals": {
    "Children": [
      {
        "Instrument": {
          ".Deal": {
            "Object": "NettingCollateralSet",
            "Reference": "Test",
            "Collateralized": "False",
            "Netted": "True"
          }
        },
        "Children": [
          {
            "Instrument": {
              ".Deal": {
                "Object": "FXNonDeliverableForward",
                "Reference": "341",
                "MtM": "",
                "Sell_Currency": "USD",
                "Sell_Amount": 1000,
                "Settlement_Date": {
                  ".Timestamp": "2024-08-31"
                },
                "Settlement_Currency": "ZAR",
                "Buy_Amount": 14000,
                "Discount_Rate": "ZAR-SWAP",
                "Buy_Currency": "ZAR"
              }
            }
          }
        ]
      }
    ]
  }
}
```

A key starting with a period (`.Deal`, `.Timestamp`, ...) converts the value it holds into the
engine object of that name (see [General Types](general_types.md)). `Object` names the deal type,
and each type's page below lists the fields it declares:

| A field that is | When omitted |
|---|---|
| **Required** | `validate()` reports it, and the deal is not completed from the panel value the page shows |
| any other (a convention) | it reads as its declared default |
| not declared by the type | kept in the block and read by nothing; `validate()` does not report it, so a misspelt convention silently takes its default |

An `Object` naming no deal type loads as that name alone and is reported by `validate()`; a run
refuses it by name. `validate()` only reports: a message never stops a deal from pricing (see
[Checking a job](../api_overview.md#checking-a-job-before-running-it)).
