# Interest Rates

Two families: `InterestRate` constructs the zero curves from deposits, FRAs, swaps
and FX forward outrights, and `HullWhite2FactorModelParameters` then fits the two-factor Hull-White
model to at-the-money swaptions priced off those curves, which are always solved first. All six
families are summarised on [General](general.md).
