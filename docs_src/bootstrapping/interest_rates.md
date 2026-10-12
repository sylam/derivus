# Interest Rates

Two families: `InterestRateCurveParameters` constructs the zero curves from deposits, FRAs, swaps
and FX forward outrights, and `HullWhite2FactorModelParameters` then fits the two-factor Hull-White
model to at-the-money swaptions priced off those curves, which are always solved first. All six
families are summarised on [General](general.md); the references below are cited by the two class
sections that follow.

## References

- Ferdinando Ametrano, Marco Bianchetti, "Everything You Always Wanted to Know about Multiple
  Interest Rate Curve Bootstrapping but Were Afraid to Ask", *SSRN Electronic Journal*, 2013,
  DOI 10.2139/ssrn.2219548.
- Patrick S. Hagan, Graeme West, "Interpolation Methods for Curve Construction", *Applied
  Mathematical Finance* 13(2), 89-129, 2006, DOI 10.1080/13504860500396032.
- J. E. Dennis, Robert B. Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear
  Equations*, SIAM, 1996, DOI 10.1137/1.9781611971200.
- John C. Hull, Alan D. White, "Numerical Procedures for Implementing Term Structure Models II",
  *The Journal of Derivatives* 2(2), 37-48, 1994, DOI 10.3905/jod.1994.407908.
- Damiano Brigo, Fabio Mercurio, *Interest Rate Models - Theory and Practice*, 2nd ed., Springer
  Finance, Springer, 2006, DOI 10.1007/978-3-540-34604-3 (the registry record lists no authors).
- David F. Schrager, Antoon A. J. Pelsser, "Pricing Swaptions and Coupon Bond Options in Affine Term
  Structure Models", *Mathematical Finance* 16(4), 673-694, 2006, DOI
  10.1111/j.1467-9965.2006.00289.x.
- L. Bachelier, "Théorie de la spéculation", *Annales scientifiques de l'École normale supérieure*
  17, 21-86, 1900, DOI 10.24033/asens.476.
- David J. Wales, Jonathan P. K. Doye, "Global Optimization by Basin-Hopping and the Lowest Energy
  Structures of Lennard-Jones Clusters Containing up to 110 Atoms", *The Journal of Physical
  Chemistry A* 101(28), 5111-5116, 1997, DOI 10.1021/jp970984n.
- Mary Ann Branch, Thomas F. Coleman, Yuying Li, "A Subspace, Interior, and Conjugate Gradient
  Method for Large-Scale Bound-Constrained Minimization Problems", *SIAM Journal on Scientific
  Computing* 21(1), 1-23, 1999, DOI 10.1137/S1064827595289108.
- Bruce Christianson, "Reverse accumulation and implicit functions", *Optimization Methods and
  Software* 9(4), 307-322, 1998, DOI 10.1080/10556789808805697.
- Marc Henrard, "Adjoint algorithmic differentiation: calibration and implicit function theorem",
  *The Journal of Computational Finance* 17(4), 37-47, 2014, DOI 10.21314/jcf.2014.284.
