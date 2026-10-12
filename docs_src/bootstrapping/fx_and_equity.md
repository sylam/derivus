# Equity and FX

Four classes: `FXVolSurfaceParameters` constructs an FX volatility surface from delta quotes;
`GBMAssetPriceTSModelParameters` integrates a surface's ATM column into a lognormal term
structure; `OptionQuoteFamily` is the quote preparation the option-fitted families share; and
`LogVar2FJModelParameters` fits the [LogVar2FJ model](../theory/stochastic_volatility.md) to a
ladder of European options. All six families are summarised on [General](general.md); the
references below are cited by the class sections that follow.

## References

- Dimitri Reiswich, Uwe Wystup, "A Guide to FX Options Quoting Conventions", *The Journal of
  Derivatives* 18(2), 58-68, 2010, DOI 10.3905/jod.2010.18.2.058.
- Iain J. Clark, *Foreign Exchange Option Pricing: A Practitioner's Guide*, Wiley, 2011,
  ISBN 9780470683682, DOI 10.1002/9781119208679 (the registry dates the record 2012-01-02 and
  lists it as an edited book).
- Allan M. Malz, "Estimating the Probability Distribution of the Future Exchange Rate from Option
  Prices", *The Journal of Derivatives* 5(2), 18-36, 1997, DOI 10.3905/jod.1997.407988.
- Ole E. Barndorff-Nielsen, "Processes of normal inverse Gaussian type", *Finance and Stochastics*
  2(1), 41-68, 1998, DOI 10.1007/s007800050032 (the registry dates the record 1997-11).
- Kresimir Demeterfi, Emanuel Derman, Michael Kamal, Joseph Zou, "A Guide to Volatility and
  Variance Swaps", *The Journal of Derivatives* 6(4), 9-32, 1999, DOI 10.3905/jod.1999.319129.
- Gene H. Golub, John H. Welsch, "Calculation of Gauss quadrature rules", *Mathematics of
  Computation* 23(106), 221-230, 1969, DOI 10.1090/S0025-5718-69-99647-1.
- I. M. Sobol', "On the distribution of points in a cube and the approximate evaluation of
  integrals", *USSR Computational Mathematics and Mathematical Physics* 7(4), 86-112, 1967,
  DOI 10.1016/0041-5553(67)90144-9.
- Art B. Owen, "Scrambled net variance for integrals of smooth functions", *The Annals of
  Statistics* 25(4), 1997, DOI 10.1214/aos/1031594731.
- Hans U. Gerber, Elias S. W. Shiu, "Actuarial bridges to dynamic hedging and option pricing",
  *Insurance: Mathematics and Economics* 18(3), 183-218, 1996, DOI
  10.1016/0167-6687(96)85007-4, for the Esscher transform.
- Mary Ann Branch, Thomas F. Coleman, Yuying Li, "A Subspace, Interior, and Conjugate Gradient
  Method for Large-Scale Bound-Constrained Minimization Problems", *SIAM Journal on Scientific
  Computing* 21(1), 1-23, 1999, DOI 10.1137/S1064827595289108.
- R. Penrose, "A generalized inverse for matrices", *Mathematical Proceedings of the Cambridge
  Philosophical Society* 51(3), 406-413, 1955, DOI 10.1017/S0305004100030401.
