# Survival Rates

The historical estimator for the hazard-rate process of the Theory page
[Survival Rates](../theory/survival_rates.md). The archive holds one column per tenor,
`SurvivalProb.<Name>,<tenor>` with the tenor in years, each the quantity the `SurvivalProb` curve
holds: the negative log survival probability $\Lambda(\tau)=-\log S(0,\tau)$. The estimator runs the
shared OU regression of [Historical Calibration](general.md#the-shared-ou-regression).
