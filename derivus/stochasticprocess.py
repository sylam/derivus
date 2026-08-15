########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of Derivus.
#
# Derivus is free for noncommercial use under the terms of the PolyForm
# Noncommercial License 1.0.0. You should have received a copy of the license
# along with Derivus. If not, see
# <https://polyformproject.org/licenses/noncommercial/1.0.0>.
#
# Derivus is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
########################################################################

# import standard libraries
import copy
import itertools
import logging

# 3rd party libraries
import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.linalg import expm as matrix_expm, logm as matrix_logm
import torch
import torch.nn.functional as nnf

# Internal modules
from . import utils
from .schema import F, REQUIRED
from .instruments import get_fx_zero_rate_factor, get_equity_zero_rate_factor, get_dividend_rate_factor


def piecewise_linear(t, tenor, values, shared):
    if isinstance(values, np.ndarray):
        # no tensors needed - don't even bother caching
        dt = np.diff(t)
        interp = np.interp(t, tenor, values)
        return dt, interp[:-1], np.diff(interp) / dt
    else:
        key_code = ('piecewise_linear', values.data_ptr(), id(tenor), t.tobytes())

    if key_code not in shared.t_PreCalc:
        # linear interpolation of the vols at vol_tenor to the grid t
        dt = values.new(np.diff(t))
        interp = utils.interpolate_tensor(t, tenor, values)
        grad = (interp[1:] - interp[:-1]) / dt
        shared.t_PreCalc[key_code] = (dt, interp[:-1], grad)

    # interpolated vol
    return shared.t_PreCalc[key_code]


def integrate_piecewise_linear(fn_norm, shared, time_grid, tenor1, val1, tenor2=None, val2=None):
    def final_integration_points(only_np, int_points, interp_value):
        # return all but last point and make a tensor if necessary
        return int_points[:-1] if only_np else shared.t_PreCalc.setdefault(
            ('integration', tuple(int_points[:-1])), interp_value.new(int_points[:-1]))

    t = np.union1d(0.0, time_grid)
    np_only = isinstance(val1, np.ndarray)
    max_time = time_grid.max()
    fn, norm = fn_norm
    integration_points = np.union1d(t, tenor1[:tenor1.searchsorted(max_time)])

    if val2 is not None:
        integration_points = np.union1d(integration_points, tenor2[:tenor2.searchsorted(max_time)])
        _, interp_val1, m1 = piecewise_linear(integration_points, tenor1, val1, shared)
        dt, interp_val2, m2 = piecewise_linear(integration_points, tenor2, val2, shared)
        np_only = np_only and isinstance(val2, np.ndarray)
        t_integration_points = final_integration_points(np_only, integration_points, interp_val1)
        int_fn = fn(t_integration_points, interp_val1, interp_val2, dt, m1, m2)
    else:
        dt, interp_val, m = piecewise_linear(integration_points, tenor1, val1, shared)
        t_integration_points = final_integration_points(np_only, integration_points, interp_val)
        int_fn = fn(t_integration_points, interp_val, dt, m)

    if np_only:
        integral = np.pad(np.cumsum(int_fn) / norm, [1, 0], 'constant')
        return integral[integration_points.searchsorted(time_grid)]
    else:
        integral = nnf.pad(torch.cumsum(int_fn, dim=0) / norm, (1, 0))
        if integration_points.size == time_grid.size:
            return integral
        else:
            return integral[integration_points.searchsorted(time_grid)]


# Hull white analytic integrals for 1 and 2 factor models (assuming piecewise linear vols)

def hw_calc_H(a, exp):
    # sympy.simplify(sympy.integrate(sympy.exp(a * s) * (v + m * (s - t)), (s, t, t + dt)))
    # leave the division till later and simplify
    def H(t, v, dt, m):
        return (-a * v + m) * exp(a * t) + (a * m * dt + a * v - m) * exp(a * (dt + t))

    return H, a * a


def hw_calc_IJK(a, exp):
    # sympy.simplify(sympy.integrate(sympy.exp(a*s)*(vi+mi*(s-t))*(vj+mj*(s-t)), (s,t,t+dt)))
    # leave the division till later and simplify
    def IJK(t, vi, vj, dt, mi, mj):
        a2, dt2, mi_mj, mj_vi_p_mi_vj, vi_vj = a * a, dt * dt, mi * mj, mj * vi + mi * vj, vi * vj

        return ((a2 * (dt2 * mi_mj + dt * mj_vi_p_mi_vj + vi_vj) + 2 * mi_mj * (1 - a * dt) - a * mj_vi_p_mi_vj)
                * exp(a * dt) - a2 * vi_vj + a * mj_vi_p_mi_vj - 2 * mi_mj) * exp(a * t)

    return IJK, a ** 3


def hmm_forward_backward(log_pi, log_P, log_emit):
    """Log-space forward-backward for a discrete-state HMM. `log_emit` is (T, S) of
    per-step per-state emission log-densities; returns smoothed posteriors `gamma`
    (T, S), pairwise posteriors `xi` (T-1, S, S), and log-likelihood. Used by every
    Markov-style calibration class."""
    from scipy.special import logsumexp
    T, S = log_emit.shape
    log_alpha = np.full((T, S), -np.inf)
    log_alpha[0] = log_pi + log_emit[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + log_P, axis=0) + log_emit[t]
    log_lik = logsumexp(log_alpha[-1])
    log_beta = np.zeros((T, S))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_P + (log_emit[t + 1] + log_beta[t + 1])[None, :], axis=1)
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    log_xi = (log_alpha[:-1, :, None] + log_P[None, :, :]
              + log_emit[1:, None, :] + log_beta[1:, None, :])
    log_xi -= logsumexp(log_xi.reshape(T - 1, -1), axis=1)[:, None, None]
    xi = np.exp(log_xi)
    return gamma, xi, log_lik


def garch11_t_mle(x):
    """Zero-mean GARCH(1,1) with standardised Student-t innovations, MLE on the series `x`.
    Returns `(omega, alpha, beta, nu, h, se)` in the UNITS OF `x` — the caller owns any rescaling
    (`GARCHSpotCalibration` fits percent log returns and converts back). `h` is the filtered
    conditional-variance path off the fitted parameters, seeded at the sample variance, so `h[-1]`
    is the variance OF the last observation — one step stale as a "today's state" stamp, which is
    what `H0` carries.

    ONE caller now: `BasisLinkedSpotCalibration` used to fit its innovation here, on the residual of
    a separately estimated conditional mean, and that two-likelihood split is what
    `arx1_t_mle(..., garch=True)` absorbed. The recursion and the `h[0]` seeding convention there are
    this function's, deliberately, so the two estimators stamp seeds that mean the same thing.

    Uses `arch` if importable, else scipy L-BFGS-B on the identical log-likelihood, with
    asymptotic standard errors from a central-difference numerical Hessian (per-coordinate step,
    since ω~1e-2 and ν~7.5 differ by orders of magnitude)."""
    from scipy.special import gammaln
    var0 = float(np.var(x))

    def filtered(omega, alpha, beta):
        h = np.empty_like(x)
        h[0] = var0
        for i in range(1, len(x)):
            h[i] = omega + alpha * x[i - 1] ** 2 + beta * h[i - 1]
        return h

    def negll(th):
        omega, alpha, beta, nu = th
        if omega <= 0.0 or nu <= 2.0 or alpha < 0.0 or beta < 0.0:
            return 1.0e10
        h = filtered(omega, alpha, beta)
        ne = nu - 2.0
        ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(ne * np.pi)
              - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(x ** 2 / (h * ne)))
        return -ll.sum()

    try:
        from arch import arch_model
        res = arch_model(x, mean='Zero', vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
        omega, alpha, beta, nu = (float(res.params[k]) for k in ('omega', 'alpha[1]', 'beta[1]', 'nu'))
        se = {k: float(v) for k, v in zip(('omega', 'alpha', 'beta', 'nu'), res.std_err[
            ['omega', 'alpha[1]', 'beta[1]', 'nu']].values)}
        h = np.asarray(res.conditional_volatility, dtype=np.float64) ** 2
    except ImportError:
        from scipy.optimize import minimize
        opt = minimize(negll, np.array([0.01, 0.05, 0.90, 8.0]), method='L-BFGS-B',
                       bounds=[(1e-8, None), (0.0, 0.999), (0.0, 0.999), (2.05, 200.0)],
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        omega, alpha, beta, nu = opt.x
        step = 1e-4 * (np.abs(opt.x) + 1e-3)
        H = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                ei, ej = np.zeros(4), np.zeros(4)
                ei[i], ej[j] = step[i], step[j]
                H[i, j] = H[j, i] = (negll(opt.x + ei + ej) - negll(opt.x + ei - ej)
                                     - negll(opt.x - ei + ej) + negll(opt.x - ei - ej)
                                     ) / (4 * step[i] * step[j])
        se = dict(zip(('omega', 'alpha', 'beta', 'nu'), np.sqrt(np.abs(np.diag(np.linalg.inv(H))))))
        h = filtered(omega, alpha, beta)
    return float(omega), float(alpha), float(beta), float(nu), h, se


def arx1_t_mle(equations, nu_bounds=(3.0, 50.0), mean=None, garch=False):
    """Joint MLE of a SYSTEM of AR(1)/ARX(1) equations sharing one Student-t degrees of freedom:

        y_t = mu_t + phi(y_{t-1} - mu_t) + gamma*x_t + sigma_t*e_t,  e_t ~ standardised t_nu

    `equations` is `[(y, x), ...]`, `x` None for a plain AR(1) and otherwise aligned to the TARGET
    rows (`len(x) == len(y) - 1`), so a same-step regressor is `np.diff(other)`. Returns
    `([ARX1Fit, ...], nu, se)`.

    ONE nu, fitted jointly, because the process it calibrates draws ONE chi2 per step and shares it
    across its factors: the innovation vector is an elliptical multivariate t rather than t
    marginals under a Gaussian copula. Fitting the marginals separately and reconciling their two
    nu's afterwards would estimate a model nobody wrote. The same argument is why the two optional
    switches exist rather than a second estimator beside this one - a mean fitted under one loss and
    a variance under another is the same defect one level down:

    `mean` is the level the AR reverts to. None fits a scalar intercept per equation (the AR's own
    long-run mean). An ARRAY makes mu_t an OBSERVABLE the caller supplies, aligned to the target
    rows and applied to BOTH sides of the AR, and no intercept is fitted - which is the shape of a
    slow-mean model whose mu_t is a declared recursion on the realised path rather than a parameter
    (a zero array is then "no intercept"). It must be the mean the SIMULATOR will revert to at the
    same row, i.e. strictly lagged, or the fit is done against a filtration the model does not have.

    `garch` replaces the per-equation constant sigma with its own GARCH(1,1),
    sigma_t^2 = omega + alpha*e_{t-1}^2 + beta*sigma_{t-1}^2, seeded at the OLS residual variance -
    `garch11_t_mle`'s recursion and seeding convention, fitted HERE inside the same likelihood as
    the conditional mean instead of afterwards on the mean fit's residual.

    scipy L-BFGS-B on the exact log-likelihood from an OLS start, with asymptotic standard errors
    from a central-difference numerical Hessian (per-coordinate step, since phi ~ 1 and nu ~ 5
    differ by orders of magnitude) - the construction `garch11_t_mle` uses.

    Two callers: `QuadraticCarryCurveCalibration` (both switches off, bit-identically) and
    `BasisLinkedSpotCalibration`'s EXTENDED path, the sibling this estimator was written next to
    and which it now absorbs. That calibration's DEFAULT path is still OLS plus a moment-matched
    nu, and stays so by rule: it is what every shipped platinum world was calibrated with."""
    from scipy.optimize import minimize
    from scipy.special import gammaln

    prepared, th0, bounds, var0 = [], [], [], []
    n_scale = 3 if garch else 1
    for y, x in equations:
        y = np.asarray(y, dtype=np.float64)
        if mean is None:
            design = np.column_stack([np.ones(y.size - 1), y[:-1]] + ([] if x is None else [x]))
            target = y[1:]
        else:
            design = np.column_stack([y[:-1] - mean] + ([] if x is None else [x]))
            target = y[1:] - mean
        ols = np.linalg.lstsq(design, target, rcond=None)[0]
        resid0 = target - design @ ols
        prepared.append((y[1:], y[:-1], x))
        var0.append(resid0.var())
        th0.append(ols[1] if mean is None else ols[0])
        bounds.append((-0.9999, 0.9999))
        if mean is None:
            th0.append(ols[0] / (1.0 - ols[1]))
            bounds.append((None, None))
        if garch:
            th0 += [np.log(0.05 * var0[-1]), 0.05, 0.90]
            bounds += [(None, None), (0.0, 0.999), (0.0, 0.999)]
        else:
            th0.append(np.log(resid0.std()))
            bounds.append((None, None))
        if x is not None:
            th0.append(ols[-1])
            bounds.append((None, None))
    n_mean = int(mean is None)
    starts = np.cumsum([0] + [1 + n_mean + n_scale + (x is not None) for _, _, x in prepared])
    th0.append(min(max(6.0, nu_bounds[0]), nu_bounds[1]))
    bounds.append(nu_bounds)

    def residual(th, i):
        y1, y0, x = prepared[i]
        s = starts[i] + 1 + n_mean
        phi, mu = th[starts[i]], th[starts[i] + 1] if mean is None else mean
        e = y1 - (mu + phi * (y0 - mu))
        e = e if x is None else e - th[s + n_scale] * x
        if not garch:
            return e, np.exp(th[s])
        omega, alpha, beta = np.exp(th[s]), th[s + 1], th[s + 2]
        h = np.empty_like(e)
        h[0] = var0[i]
        for k in range(1, e.size):
            h[k] = omega + alpha * e[k - 1] ** 2 + beta * h[k - 1]
        return e, np.sqrt(h)

    def negll(th):
        nu, ne = th[-1], th[-1] - 2.0
        const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(ne * np.pi)
        out = 0.0
        for i in range(len(prepared)):
            e, sigma = residual(th, i)
            out -= (const - np.log(sigma)
                    - (nu + 1) / 2 * np.log1p(e * e / (sigma * sigma * ne))).sum()
        return out

    th = minimize(negll, np.array(th0), method='L-BFGS-B', bounds=bounds,
                  options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-10}).x
    n = th.size
    step = 1.0e-5 * (np.abs(th) + 1.0e-3)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = step[i], step[j]
            H[i, j] = H[j, i] = (negll(th + ei + ej) - negll(th + ei - ej)
                                 - negll(th - ei + ej) + negll(th - ei - ej)) / (4 * step[i] * step[j])
    se = np.sqrt(np.abs(np.diag(np.linalg.inv(H))))
    fits = []
    for i, (_, _, x) in enumerate(prepared):
        e, sigma = residual(th, i)
        s = starts[i] + 1 + n_mean
        fits.append(utils.ARX1Fit(
            float(th[starts[i]]), float(th[starts[i] + 1]) if mean is None else mean,
            sigma if garch else float(sigma),
            float(th[s + n_scale]) if x is not None else 0.0, e / sigma,
            (float(np.exp(th[s])), float(th[s + 1]), float(th[s + 2])) if garch else None))
    return fits, float(th[-1]), {
        'phi': [float(se[s]) for s in starts[:-1]], 'nu': float(se[-1]),
        'gamma': [float(se[s + 1 + n_mean + n_scale]) if x is not None else 0.0
                  for s, (_, _, x) in zip(starts[:-1], prepared)]}


# State-reveal tags for `reveal_state_at`: a CONTINUOUS segment is a first-order-differentiable
# risk factor (the diff-PCA pool); a SUFFICIENT segment is a minimal sufficient statistic (regime
# belief) revealed verbatim, bypassing PCA.
REVEAL_CONTINUOUS = 'continuous'
REVEAL_SUFFICIENT = 'sufficient'


class StochasticProcess(object):
    """Base class for all stochastic processes"""

    def __init__(self, factor, param):
        self.factor = factor
        self.param = param
        self.params_ok = True

    def copy(self):
        """Shallow copy. Use case: forking a process for nested simulation (inner MC)
        so the fork can be precalculated against a different shared state / time grid
        without clobbering the outer instance's precalc-derived attributes (`spot0`,
        `scenario_horizon`, `z_offset`, etc.). Construction-time references (`factor`,
        `param`, `implied`) are shared by reference, which is intentional — these are
        read-only after setup."""
        return copy.copy(self)

    def link_references(self, implied_tensor, implied_var, implied_factors):
        """link market variables across different risk factors"""
        pass

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass

    @classmethod
    def privileged_layout(cls, param):
        """Static {name: dim} schema of privileged factors this process emits, derivable from
        param alone for the policy's privileged-encoder sizing at construction time."""
        return {}

    def privileged_factors(self, simulated):
        """Privileged factors the asymmetric critic sees but the actor does not — dict of
        (T, B, dim) tensors keyed by name. `simulated` is this process's (T, B) path."""
        return {}

    def calibrated_annual_vol(self):
        """Annualized FRACTIONAL vol implied by this process's calibrated params, or None if
        the process does not characterize a single scalar spot vol. Consumed by the utility-
        scale fallback (hedge_bundle.Bundle._resolve_utility_scale) when Spot_Price_History is absent —
        the calibrated substitute for the realized-vol read off the history window. Default None
        (no vol characterization); spot models that own a calibrated vol override it."""
        return None

    def revealed_annual_vol(self, log_h):
        """Map this process's revealed log-conditional-variance surface `log_h` (T, B) to
        annualized fractional vol (T, B), or None if the process exposes no log-variance
        sufficient statistic. Consumed by the state-dependent bid/offer half-spread
        (`hedge_runtime.per_contract_kappa` Vol_Scale) — the per-step, per-path vol driver.
        Default None."""
        return None

    @property
    def bridge_variance_rate(self):
        """Annualized log-variance rate of THIS factor as a scalar, or None.

        A barrier is monitored continuously but a deal's time grid only observes the spot at its
        own dates, so whether a path crossed in between is a conditional probability rather than
        an observation - and a Brownian bridge needs the variance of the interval it spans.

        A RATE against elapsed time rather than a per-interval array, because a deal's grid and
        the scenario grid are different arrays (`deal_time_grid` indexes `mtm_time_grid`, while a
        process discretises `scen_time_grid`); an array would have to be indexed by one and is
        reached with the other. Elapsed time is common to both, so a rate cannot be misindexed.
        This is exact where the vol is constant; a process whose interval variance is not
        proportional to elapsed time should return None until it is given a form that can express
        that, rather than a rate that silently means something else.

        It must come from the SIMULATION model, not from a pricing implied vol: the latter is the
        vol for the option's remaining life, which is a different quantity that happens to have
        the same units. Processes returning None fall back to observing endpoints, which is what
        the engine did everywhere before.
        """
        return None

    def reveal_state_at(self, t, buffer):
        """Ordered market-state segments `[(block, reveal_kind), ...]` this factor exposes to
        the value function at scenario-time index `t` — the informative deep-state the DP/MPC
        solvers consume. `block` is factor-dims-leading with the batch axis TRAILING; the CALC
        owns the batch reshape (it knows outer `(*, B)` vs inner `(*, B, B2)` mode), so no batch
        rank leaks into this signature. Default: the whole factor state at `t` as ONE continuous
        segment — `buffer[self.factor_key][t]` covers a scalar spot, a compact carry curve, and a
        full tenor curve alike (a curve declares every tenor; the reducer decides what survives).
        `self.factor_key` is set for every stochastic factor by the calc before reveal is ever
        called (calculation.py, `value.factor_key = key`)."""
        return [(buffer[self.factor_key][t], REVEAL_CONTINUOUS)]

    def inner_fork_seed(self, factor_key, outer_buf, t):
        """Buffer entries seeding this process's inner-MC fork at outer time `t` from the
        outer path — the per-outer-path t=0 privileged sufficient statistic the fork reprices
        from (regime for the HMM, conditional variance for GARCH). Keyed by the buffer key the
        process's `generate` consumes, so the forker runs one uniform loop across process types
        instead of type-branching. Default: nothing to seed."""
        return {}

    # ---- Model-agnostic reseed protocol (the calc / solver speak only these verbs) ----------
    # A process owns every model-specific buffer key and recursion; the calc loops uniformly and
    # never mentions a regime, belief, or variance. Base implementations are inert no-ops.

    def outer_reseed(self):
        """Buffer entries seeding the NEXT outer run's t=0 state from THIS run's just-generated
        terminal state — the diff-ML `Randomize_Initial_State` burn-in. Returned (not written)
        so the calc can stash them across the reset that clears the buffer. Default: none."""
        return {}

    def reseed_from_path(self, simulated, shared_mem):
        """Re-derive and publish this process's path-dependent revealed state ALONG a supplied
        path `simulated` (the `Observed_Scenario` / stepper replay, where a driver path replaces
        the generated one). Publishes to `shared_mem.t_Scenario_Buffer` under this process's own
        keys. Default: nothing to replay."""
        pass

    def reseed_inner_state(self, factor_key, simulated, outer_buf, t, shared_mem, opts, with_grad):
        """Post-generate inner-fork coherence: publish any path-dependent revealed state the
        bootstrap's `reveal_state_at` consumes at t+1 (e.g. a filtered belief), keyed by this
        process. `opts` forwards hedging-problem switches opaquely (the calc never interprets
        them). Returns grad-leaf entries `{key: leaf}` to fold into the twin-loss `state_t_leaves`
        when `with_grad`. Default: nothing to publish, no leaves."""
        return {}

    def diff_state_leaves(self):
        """Ordered buffer-key suffixes of this process's DIFFERENTIABLE state coordinates — the
        market columns (before the trailing price) the twin loss supervises via a state leaf
        rather than the raw factor leaf. The solver reads this to project leaf grads into market
        columns without knowing any model concept. Default: none (price-only / masked state)."""
        return ()

    def forward_curve(self, tensor, time_grid_years, shared, mul_time=True):
        """`utils.calc_curve_forwards` lifted to a per-path BATCH of curves. `tensor` is
        the t=0 curve: `(n_tenors,)` calibrated, or `(n_tenors, B)` per-path (inner-MC
        fork or diff-ML t=0 burn-in). Calibrated → one call returning `(T, n_tenors)`.
        Per-path → loop the B columns and stack on a trailing batch axis → `(T, n_tenors, B)`.

        `utils.calc_curve_forwards` is shape-dispatched, so BOTH cases are one call — the
        per-path Python loop this used to run measured 0.92 ms per path (a 31-tenor curve at
        B=2048 cost 1.93 s, perfectly linear in B) and was the inner-MC fork's dominant cost."""
        return utils.calc_curve_forwards(
            self.factor, tensor, time_grid_years, shared, mul_time=mul_time)

    @staticmethod
    def align_rank(x, ndim):
        """Right-pad `x` with trailing singleton axes to rank `ndim` so a canonical
        lower-rank tensor (a calibrated `(T, n_tenors)` curve) broadcasts against a
        higher-rank one (the `(T, n_tenors, B)` per-path or `(T, n_tenors, B, B2)`
        inner stochastic component). Target rank is explicit; no-op when already there."""
        return x.reshape(*x.shape, *([1] * (ndim - x.ndim)))


class GBMAssetPriceModel(StochasticProcess):
    """The Geometric Brownian Motion Stochastic Process"""

    documentation = (
        'Asset Pricing', ['The spot price of an equity or FX rate can be modelled as Geometric Brownian Motion (GBM).',
                          'The model is specified as follows:',
                          '',
                          '$$ dS = \\mu S dt + \\sigma S dZ$$',
                          '',
                          'Its final form is:',
                          '',
                          '$$ S = exp \\Big( (\\mu-\\frac{1}{2}\\sigma^2)t + \\sigma dW(t)  \\Big ) $$',
                          '',
                          'Where:',
                          '',
                          '- $S$ is the spot price of the asset',
                          '- $dZ$ is the standard Brownian motion',
                          '- $\\mu$ is the constant drift of the asset',
                          '- $\\sigma$ is the constant volatility of the asset',
                          '- $dW(t)$ is a standard Wiener Process'])

    factor_types = ('EquityPrice', 'FxRate')
    fields = [
        F('Vol', 'Float', default=0),
        F('Drift', 'Float', default=0)
    ]

    def __init__(self, factor, param, implied_factor=None):
        super(GBMAssetPriceModel, self).__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        dt = np.diff(np.hstack(([0], time_grid.time_grid_years)))
        var = self.param['Vol'] * self.param['Vol'] * dt
        # store params in tensors
        self.drift = tensor.new((self.param['Drift'] * dt - 0.5 * var).reshape(-1, 1))
        self.vol = tensor.new((np.sqrt(var)).reshape(-1, 1))

        # store a reference to the current tensor
        self.spot = tensor

    @property
    def bridge_variance_rate(self):
        """Lognormal with a constant vol, so the bridge applies exactly between any two dates."""
        return float(self.param['Vol']) ** 2

    @property
    def correlation_name(self):
        return 'LognormalDiffusionProcess', [()]

    def generate(self, shared_mem):
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        if Z.ndim == 2:
            f1 = (self.drift + self.vol * Z).cumsum(axis=0)
            return self.spot * torch.exp(f1)
        # Inner MC (T, B, B2): per-step drift/vol arrays (T,1) -> (T,1,1); the per-outer-
        # path spot (B,)/(1,) -> (1,B,1)/(1,1,1) so each outer path's spot broadcasts
        # across its B2 inner fan-out.
        f1 = (self.drift.unsqueeze(-1) + self.vol.unsqueeze(-1) * Z).cumsum(axis=0)
        return self.align_rank(self.spot.unsqueeze(0), Z.ndim) * torch.exp(f1)


class GBMAssetPriceCalibration(object):
    """Lognormal drift and volatility from `utils.calc_statistics`. Takes no tuning."""
    model_type = 'GBMAssetPriceModel'
    fields = []

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0, vol_cuttoff=0.5, drift_cuttoff=0.1):
        stats, correlation, delta = utils.calc_statistics(
            data_frame, method='Log', num_business_days=num_business_days)
        mu = (stats['Drift'] + 0.5 * (stats['Volatility'] ** 2)).values[0]
        sigma = stats['Volatility'].values[0]

        return utils.CalibrationInfo(
            {'Vol': np.clip(sigma, 0.01, vol_cuttoff), 'Drift': np.clip(mu, -drift_cuttoff, drift_cuttoff)},
            [[1.0]], delta)


class GBMAssetPriceTSModelImplied(StochasticProcess):
    """The Geometric Brownian Motion Stochastic Process with implied drift and vol"""

    documentation = ('Asset Pricing', [
        'GBM with constant drift and vol may not be suited to model risk-neutral asset prices. A generalization that',
        'allows this would be to modify the volatility $\\sigma(t)$ and $\\mu(t)$ to be functions of time $t$.',
        'This can be specified as follows:',
        '',
        '$$ \\frac{dS(t)}{S(t)} = (r(t)-q(t)-v(t)\\sigma(t)\\rho) dt + \\sigma(t) dW(t)$$',
        '',
        'Note that no risk premium curve is captured. For Equity factors, its final form is:',
        '',
        '$$ S(t+\\delta) = F(t,t+\\delta)exp \\Big(\\rho(C(t+\\delta)-C(t)) -\\frac{1}{2}(V(t+\\delta)) - V(t))\
         + \\sqrt{V(t+\\delta) - V(t)}Z  \\Big) $$',
        '',
        'Where:',
        '',
        '- $\\sigma(t)$ is the volatility of the asset at time $t$',
        '- $v(t)$ is the *Quanto FX Volatility* of the asset at time $t$. $\\rho$ is then the *Quanto FX Correlation*',
        '- $V(t) = \\int_{0}^{t} \\sigma(s)^2 ds$',
        '- $C(t) = \\int_{0}^{t} v(s)\\sigma(s) ds$',
        '- $r$ is the interest rate in the asset currency',
        '- $q$ is the yield on the asset (If S is a foreign exchange rate, q is the foreign interest rate)',
        '- $F(t,t+\\delta)$ is the forward asset price at time t',
        '- $S$ is the spot price of the asset',
        '- $Z$ is a sample from the standard normal distribution',
        '- $\\delta$ is the increment in timestep between samples',
        '',
        'In the case that the $S(t)$ represents an FX rate, this can be further simplified to:',
        '',
        '$$S(t)=S(0)\\beta(t)exp\\Big(\\frac{1}{2}\\bar\\sigma(t)^2t+\\int_0^t\\sigma(s)dW(s)\\Big)$$',
        '',
        'Here $C(t)=\\bar\\sigma(t)^2t, \\beta(t)=exp\\Big(\\int_0^t(r(s)-q(s))ds\\Big), \\rho=-1$ and $v(t)=\\sigma('
        't)$'
    ])

    factor_types = ('EquityPrice', 'FxRate')
    fields = [
        F('Risk_Premium', 'Curve')
    ]

    def __init__(self, factor, param, implied_factor=None):
        super(GBMAssetPriceTSModelImplied, self).__init__(factor, param)
        self.implied = implied_factor
        # get the name of the underlying factor
        self.factor_type = self.factor.__class__.__name__
        # potentially handle quanto fx volatility
        self.quanto_fx_tenor = None

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        def calc_vol(t, v, dt, m):
            ''' sympy.simplify(sympy.integrate((v + m * (s - t)) ** 2 , (s, t, t + dt))) '''
            return dt * (dt ** 2 * m ** 2 / 3 + dt * m * v + v ** 2)

        def cal_quanto_fx_vol(t, vi, vj, dt, mi, mj):
            ''' sympy.simplify(sympy.integrate((vi + mi * (s - t)) * (vj + mj * (s - t)), (s, t, t + dt))) '''
            return dt * (2 * dt ** 2 * mi * mj + 3 * dt * mi * vj + 3 * dt * mj * vi + 6 * vi * vj) / 6

        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        # calc vols
        vol_tenor = self.implied.param['Vol'].array[:, 0]
        self.V = torch.unsqueeze(integrate_piecewise_linear(
            (calc_vol, 1.0), shared, time_grid.time_grid_years, vol_tenor, implied_tensor['Vol']), dim=1)
        # per-step incremental vol, anchored at V(0)=0 so the first step evolves from today (t=0)
        self.delta_vol = torch.sqrt(self.V - nnf.pad(self.V[:-1], (0, 0, 1, 0)))
        # we always evolve from today: prepend 0 to the sample times so the step sizes are the diffs
        # [t_1, t_2 - t_1, ...] and each step's drift is read from the curve as-of its start node
        # [0, t_1, ..., t_{N-1}] (when t_1 = 0 the first step is a no-op and we match spot today)
        self.delta_scen_t = np.diff(np.insert(time_grid.scen_time_grid, 0, 0)).reshape(-1, 1)
        # store a reference to the current tensor
        self.spot = tensor
        # the curve-read grid is the step starts: today (t=0) followed by all but the final sample
        today = time_grid.scenario_grid[:1].copy()
        today[:, utils.TIME_GRID_MTM] = 0.0
        self.scen_grid = np.vstack([today, time_grid.scenario_grid[:-1]])
        # check if we need to calculate the quanto fx vol
        if self.factor_type == 'EquityPrice' and self.quanto_fx_tenor is not None:
            # need to get the quantofx rate if necessary
            self.C = torch.unsqueeze(integrate_piecewise_linear(
                (cal_quanto_fx_vol, 1.0), shared, time_grid.time_grid_years,
                vol_tenor, implied_tensor['Vol'], self.quanto_fx_tenor, implied_tensor['Quanto_FX_Volatility']),
                dim=1)
            self.rho = implied_tensor['Quanto_FX_Correlation']
        else:
            self.C = 0.0
            self.rho = 0.0

    def link_references(self, implied_tensor, implied_var, implied_factors):
        """link market variables across different risk factors"""
        if self.factor_type == 'EquityPrice':
            fx_implied_index = utils.Factor('FxRate', self.factor.get_currency())
            if fx_implied_index in implied_var:
                FXImplied_vol_factor = utils.Factor(
                    'GBMAssetPriceTSModelParameters', self.factor.get_currency() + ('Vol',))
                # now set the Quanto_FX_Volatility to the same vol as the fx rate
                implied_tensor['Quanto_FX_Volatility'] = implied_var[fx_implied_index][FXImplied_vol_factor]
                if self.implied.param['Quanto_FX_Volatility'] is not None:
                    self.quanto_fx_tenor = self.implied.param['Quanto_FX_Volatility'].array[:, 0]
                else:
                    self.quanto_fx_tenor = implied_factors[
                        utils.Factor('GBMAssetPriceTSModelParameters', self.factor.get_currency())].get_tenor()

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        # this is valid for FX and Equity factors only
        if self.factor_type == 'EquityPrice':
            self.r_t = get_equity_zero_rate_factor(
                factor.name, static_ofs, stoch_ofs, all_tenors, all_factors)
            self.q_t = get_dividend_rate_factor(
                factor.name, static_ofs, stoch_ofs, all_tenors)
        elif self.factor_type == 'FxRate':
            self.r_t = get_fx_zero_rate_factor(
                self.factor.get_domestic_currency(None), static_ofs, stoch_ofs, all_tenors, all_factors)
            self.q_t = get_fx_zero_rate_factor(factor.name, static_ofs, stoch_ofs, all_tenors, all_factors)
        else:
            raise Exception('Unknown factor type {}'.format(self.factor_type))

    @property
    def correlation_name(self):
        return 'LognormalDiffusionProcess', [()]

    def generate(self, shared_mem):
        """
        Simulate the asset price path; drift comes from the foreign/domestic (or dividend/repo)
        zero curves.

        Dual-mode on Z.ndim — outer (T, B) or inner MC (T, B, B2). Under inner MC the rate curves
        are simulated to (scen, n_tenors, B, B2) and gathered with n_batch_dims=2, which collapses
        the curve stack (B,B2) -> B*B2, so the drift returns (T, B*B2) and is reshaped to
        (T, B, B2). Own per-step arrays (T,1) -> (T,1,1) and the per-outer-path spot (B,)/(1,) ->
        (1,B,1)/(1,1,1) so both broadcast across the B2 fan-out.
        """
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        if Z.ndim == 2:
            # Outer: drift from the foreign/domestic (or div/repo) zero curves; single batch.
            rt = utils.calc_time_grid_curve_rate(self.r_t, self.scen_grid, shared_mem)
            qt = utils.calc_time_grid_curve_rate(self.q_t, self.scen_grid, shared_mem)
            rt_rates = rt.gather_weighted_curve(shared_mem, self.delta_scen_t)
            qt_rates = qt.gather_weighted_curve(shared_mem, self.delta_scen_t)
            drift = torch.cumsum(torch.squeeze(rt_rates - qt_rates, dim=1), dim=0)
            f1 = torch.cumsum(self.delta_vol * Z, dim=0)
            return self.spot * torch.exp(drift - self.rho * self.C - 0.5 * self.V + f1)
        # Inner MC (T, B, B2): the n_batch_dims=2 gather returns drift (T, B*B2) — reshape it.
        B, B2 = Z.shape[1], Z.shape[2]
        rt = utils.calc_time_grid_curve_rate(self.r_t, self.scen_grid, shared_mem, n_batch_dims=2)
        qt = utils.calc_time_grid_curve_rate(self.q_t, self.scen_grid, shared_mem, n_batch_dims=2)
        rt_rates = rt.gather_weighted_curve(shared_mem, self.delta_scen_t)
        qt_rates = qt.gather_weighted_curve(shared_mem, self.delta_scen_t)
        drift = torch.cumsum(torch.squeeze(rt_rates - qt_rates, dim=1), dim=0).reshape(-1, B, B2)
        f1 = torch.cumsum(self.delta_vol.unsqueeze(-1) * Z, dim=0)
        C = self.C.unsqueeze(-1) if torch.is_tensor(self.C) else self.C
        return self.align_rank(self.spot.unsqueeze(0), Z.ndim) * torch.exp(
            drift - self.rho * C - 0.5 * self.V.unsqueeze(-1) + f1)


class GBMPriceIndexModel(StochasticProcess):
    """The Geometric Brownian Motion Stochastic Process for Price Indices - can contain adjustments for seasonality"""

    documentation = ('Inflation',
                     ['The model is specified as follows:',
                      '',
                      '$$ F(t) = exp \\Big( (\\mu-\\frac{\\sigma^2}{2})t + \\sigma W(t) \\Big)$$',
                      '',
                      'Where:',
                      '',
                      '- $\\mu$ is the drift of the price index',
                      '- $\\sigma$ is the volatility of the price index',
                      '- $W(t)$ is a standard Wiener Process under the real-world measure',
                      '',
                      'Note that the simulation of this model is identical to plain Geometric Brownian Motion with the',
                      'exception of modifying the scenario grid to coincide with allowable publication dates obtained',
                      'by the corresponding Price Index'])

    factor_types = ('PriceIndex',)
    fields = [
        F('Vol', 'Float', default=0),
        F('Drift', 'Float', default=0),
    ]

    def __init__(self, factor, param, implied_factor=None):
        super(GBMPriceIndexModel, self).__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # store randomnumber id's
        self.z_offset = process_ofs
        # calculate the correct scenario grid
        scenario_time_grid = np.array(
            [(x - self.factor.param['Last_Period_Start']).days
             for x in self.factor.get_last_publication_dates(ref_date, time_grid.scen_time_grid.tolist())],
            dtype=np.float64)
        # record the horizon
        self.scenario_horizon = scenario_time_grid.size

        dt = np.diff(np.hstack(([0], scenario_time_grid / utils.DAYS_IN_YEAR)))
        var = self.param['Vol'] * self.param['Vol'] * dt
        self.drift = tensor.new(self.param['Drift'] * dt - 0.5 * var).reshape(-1, 1)
        self.vol = tensor.new(np.sqrt(var)).reshape(-1, 1)

        # store a reference to the current tensor
        self.spot = tensor

    @property
    def correlation_name(self):
        return 'LognormalDiffusionProcess', [()]

    def generate(self, shared_mem):
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        if Z.ndim == 2:
            f1 = torch.cumsum(self.drift + self.vol * Z, dim=0)
            return self.spot * torch.exp(f1)
        # Inner MC (T, B, B2): per-step drift/vol (T,1) -> (T,1,1); per-outer-path spot
        # (B,)/(1,) -> (1,B,1)/(1,1,1) so each outer path's spot broadcasts across B2.
        f1 = torch.cumsum(self.drift.unsqueeze(-1) + self.vol.unsqueeze(-1) * Z, dim=0)
        return self.align_rank(self.spot.unsqueeze(0), Z.ndim) * torch.exp(f1)


class GBMPriceIndexCalibration(object):
    """Lognormal drift and volatility of a price index. Takes no tuning."""
    model_type = 'GBMPriceIndexModel'
    fields = []

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Log', num_business_days=num_business_days)
        mu = (stats['Drift'] + 0.5 * (stats['Volatility'] ** 2)).values[0]
        sigma = stats['Volatility'].values[0]

        return utils.CalibrationInfo({'Vol': sigma, 'Drift': mu}, [[1.0]], delta)


class HullWhite2FactorImpliedInterestRateModel(StochasticProcess):
    """Hull white 2 factor implied interest rate model for risk neutral simulation of yield curves"""

    documentation = (
        'Interest Rates',
        ['This is a generalization of the 1 factor Hull White model. There are 2 correlated risk ',
         'factors where the $i^{th}$ factor has a volatility curve $\\sigma_i(t)$, constant reversion',
         'speed $\\alpha_i$ and market price of risk $\\lambda_i$.',
         '',
         'Final form of the model is:',
         '',
         '$$ D(t,T) = \\frac{D(0,T)}{D(0,t)}exp\\Big(-\\frac{1}{2}\\sum_{i,j=1}^2\\rho_{ij}A_{ij}(t,'
         'T)-\\sum_{i=1}^2B_i(T-t)e^{-\\alpha_it}(Y_i(t) -\\tilde\\rho_i K_i(t) + \\lambda_i H_i('
         't))\\Big) $$',
         '',
         'Where:',
         '',
         '- $B_i(t) = \\frac{(1-e^{-\\alpha_i t})}{\\alpha_i}$, $Y_i(t)=\\int\\limits_0^t e^{\\alpha_i '
         's}\\sigma_i (s) dW_i(s)$, $W_1$ and $W_2$ are correlated Weiner Processes with correlation '
         '$\\rho$ ($\\rho_{ij}=\\rho$ if $i \\neq j$ else 1)',
         '- $A_{ij}(t,T)=B_i(T-t)B_j(T-t)e^{-(\\alpha_i+\\alpha_j)}J_{ij}(t)+\\frac{B_i(T-t)}{\\alpha_j}('
         'e^{-\\alpha_it}I_{ij}(t)-e^{-(\\alpha_i+\\alpha_j)t}J_{ij}(t))+\\frac{B_j(T-t)}{\\alpha_i}(e^{'
         '-\\alpha_jt}I_{ji}(t)-e^{-(\\alpha_i+\\alpha_j)t}J_{ji}(t))$',
         '- $H_i(t)=\\int\\limits_0^t e^{\\alpha_is}\\sigma_i(s)ds$',
         '- $I_{ij}(t)=\\int\\limits_0^t e^{\\alpha_is}\\sigma_i(s)\\sigma_j(s)ds$',
         '- $J_{ij}(t)=\\int\\limits_0^t e^{(\\alpha_i+\\alpha_j)s}\\sigma_i(s)\\sigma_j(s)ds$',
         '- $K_i(t)=\\int\\limits_0^t e^{\\alpha_is}\\sigma_i(s)v(s)ds$',
         '',
         'If the rate and base currencies match, $v(t)=0$ and $\\tilde\\rho_i=0$. Otherwise, $v(t)$ is',
         'the volatility of the rate currency (in base currency) and $\\tilde\\rho_i$ is the correlation',
         'between the FX rate and the $i^{th}$ factor. The increment $Y(t_{k+1})-Y(t_k)$ (where',
         '$0=t_0,t_1,t_2...$ corresponds to the simulation grid) is gaussian with zero mean and covariance',
         'Matrix $C_{ij}=\\rho_{ij}(J_{ij}(t_{k+1})-J_{ij}(t_k))$.',
         '',
         'The cholesky decomposition of $C$ is',
         '',
         '$$L=\\begin{pmatrix} \\sqrt C_{11} & 0 \\\\ \\frac{C_{12}}{\\sqrt C_{11}} & \\sqrt {C_{22}',
         '-\\frac{C_{12}^2}{C_{11}} } \\\\ \\end{pmatrix}$$',
         '',
         'The increment is simulated using $LZ$ where $Z$ is a 2D vector of independent normals at',
         'time step $k$.'])

    factor_types = ('InterestRate',)
    fields = [
        F('Lambda_1', 'Float', default=0),
        F('Lambda_2', 'Float', default=0)
    ]

    def __init__(self, factor, param, implied_factor):
        super(HullWhite2FactorImpliedInterestRateModel, self).__init__(factor, param)
        self.implied = implied_factor
        self.cache = {}
        self.factor_tenor = None
        self.grid_index = None
        self.BtT = None

    @staticmethod
    def num_factors():
        return 2

    def link_references(self, implied_tensor, implied_var, implied_factors):
        """link market variables across different risk factors"""
        fx_implied_index = utils.Factor('FxRate', self.factor.get_currency())
        if fx_implied_index in implied_var:
            FXImplied_vol_factor = utils.Factor(
                'GBMAssetPriceTSModelParameters', self.factor.get_currency() + ('Vol',))
            # now set the Quanto_FX_Volatility to the same vol as the fx rate
            implied_tensor['Quanto_FX_Volatility'] = implied_var[fx_implied_index][FXImplied_vol_factor]
        else:
            # handle the unlikely case were we need to simulate a curve but not the fx rate.
            quantofx = self.implied.get_quanto_fx()
            if quantofx is not None:
                implied_tensor['Quanto_FX_Volatility'] = implied_tensor['Sigma_1'].new_tensor(quantofx)

    def read_cache(self, ref_date, time_grid, tensor, shared, process_ofs):
        if not self.cache:
            self.z_offset = process_ofs
            self.scenario_horizon = time_grid.scen_time_grid.size
            time_grid_years = np.array([self.factor.get_day_count_accrual(
                ref_date, t) for t in time_grid.scen_time_grid])
            # get the factor's tenor points
            self.factor_tenor = tensor.new(self.factor.get_tenor().reshape(-1, 1))
            # flatten the tenor
            self.factor_tenor_full = tensor.new_tensor(self.factor.get_tenor(), dtype=torch.float64)
            # get the quanto vol
            self.quantofx = tensor.new_tensor(
                self.implied.param['Quanto_FX_Volatility'].array[:, 1], dtype=torch.float64)

            # cache the time-grid-derived tensors/variables
            self.cache['time_grid_years'] = time_grid_years
            self.cache['t'] = tensor.new(time_grid_years.reshape(-1, 1))

        # `fwd_curve` depends on `tensor` (the t=0 curve) — recompute every call (not
        # cached) so a per-path re-precalc (inner-MC fork / diff-ML t=0 burn-in) isn't
        # shadowed by the first call's calibrated curve. Batch-aware via the base seam.
        fwd_curve = self.forward_curve(tensor, self.cache['time_grid_years'], shared)
        return self.cache['time_grid_years'], fwd_curve, self.cache['t']

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # get the factor's tenor points
        time_grid_years, fwd_curve, t = self.read_cache(ref_date, time_grid, tensor, shared, process_ofs)

        # calculate known functions
        alpha = [implied_tensor['Alpha_1'][0].type(torch.float64),
                 implied_tensor['Alpha_2'][0].type(torch.float64)]
        lam = [self.param['Lambda_1'], self.param['Lambda_2']]
        corr = implied_tensor['Correlation'].type(torch.float64)
        vols = [implied_tensor['Sigma_1'].type(torch.float64),
                implied_tensor['Sigma_2'].type(torch.float64)]
        vols_tenor = [self.implied.param['Sigma_1'].array[:, 0],
                      self.implied.param['Sigma_2'].array[:, 0]]

        H = [integrate_piecewise_linear(
            hw_calc_H(alpha[i], torch.exp), shared, time_grid_years, vols_tenor[i], vols[i]) for i in range(2)]
        I = [[integrate_piecewise_linear(
            hw_calc_IJK(alpha[i], torch.exp), shared, time_grid_years, vols_tenor[i], vols[i], vols_tenor[j], vols[j])
            for j in range(2)] for i in range(2)]
        J = [[integrate_piecewise_linear(
            hw_calc_IJK(alpha[i] + alpha[j], torch.exp), shared,
            time_grid_years, vols_tenor[i], vols[i], vols_tenor[j], vols[j]) for j in range(2)] for i in range(2)]

        # Check if the curve is not the same as the base currency
        if self.implied.param['Quanto_FX_Volatility'] and self.implied.param['Quanto_FX_Volatility'].array.any():
            quantofx = self.implied.param['Quanto_FX_Volatility'].array
            if 'Quanto_FX_Correlation_1' in implied_tensor and 'Quanto_FX_Correlation_2' in implied_tensor:
                quantofxcorr = [implied_tensor['Quanto_FX_Correlation_1'][0].type(torch.float64),
                                implied_tensor['Quanto_FX_Correlation_2'][0].type(torch.float64)]
                t_quanto_vol = implied_tensor['Quanto_FX_Volatility'].type(torch.float64)
            else:
                quantofxcorr = self.implied.get_quanto_correlation(corr, vols)
                t_quanto_vol = self.quantofx

            K = [integrate_piecewise_linear(
                hw_calc_IJK(alpha[i], torch.exp), shared, time_grid_years,
                vols_tenor[i], vols[i], quantofx[:, 0], t_quanto_vol) for i in range(2)]
        else:
            quantofxcorr = [0.0, 0.0]
            K = [corr.new_zeros(time_grid_years.size) for i in range(2)]

        # now calculate the AtT
        AtT = 0.0
        BtT = [(1.0 - torch.exp(-alpha[i] * self.factor_tenor_full)) / alpha[i] for i in range(2)]
        CtT = []
        rho = [[1.0, corr], [corr, 1.0]]

        for i, j in itertools.product(range(2), range(2)):
            first_part = torch.exp(-(alpha[i] + alpha[j]) * t) * J[i][j].reshape(-1, 1)
            second_part = torch.exp(-alpha[i] * t) * I[i][j].reshape(-1, 1) - first_part
            third_part = torch.exp(-alpha[j] * t) * I[j][i].reshape(-1, 1) - first_part

            # get the covariance
            CtT.append(rho[i][j] * J[i][j])

            # all together now
            AtT += rho[i][j] * torch.matmul(
                torch.cat([first_part, second_part, third_part], dim=1),
                torch.stack([BtT[j] * BtT[i], BtT[i] / alpha[j], BtT[j] / alpha[i]]))

        t_CtT = torch.stack(CtT).T
        # get the change in variance
        delta_CtT = t_CtT[1:] - t_CtT[:-1]
        # check if the entire cholesky is +ve definite
        if (delta_CtT[:, 0] * delta_CtT[:, 3] > delta_CtT[:, 1] * delta_CtT[:, 2]).all():
            # get the correlation through time
            C = nnf.pad(torch.linalg.cholesky(delta_CtT.reshape(-1, 2, 2)), (0, 0, 0, 0, 1, 0))
            # all good
            self.params_ok = True
        else:
            # need to fix the cholesky
            cholesky = [tensor.new_zeros((2, 2), dtype=torch.float64)]
            for i, C in enumerate(delta_CtT):
                if (C[0] > 0.0) & (C[3] > 0.0) & (C[1] * C[2] < C[0] * C[3]):
                    cholesky.append(torch.linalg.cholesky(C.reshape(2, 2)))
                else:
                    cholesky.append(cholesky[-1])

            # the cholesky was broken
            self.params_ok = False
            C = torch.stack(cholesky)

        # intermediate results
        self.BtT = [Bi.type(shared.one.dtype).reshape(-1, 1) for Bi in BtT]
        self.YtT = [torch.exp(-alpha[i] * t).type(shared.one.dtype) for i in range(2)]
        self.KtT = [(quantofxcorr[i] * K[i].reshape(-1, 1)).type(shared.one.dtype) for i in range(2)]
        self.HtT = [(lam[i] * H[i].reshape(-1, 1)).type(shared.one.dtype) for i in range(2)]

        # needed for factor calcs later
        self.F1 = C[:, 0, 0].reshape(-1, 1).type(shared.one.dtype)
        self.F2 = C[:, 1].t().unsqueeze(axis=2).type(shared.one.dtype)

        # store the grid points used if necessary
        if len(time_grid.scenario_grid) != time_grid_years.size:
            self.grid_index = time_grid.scen_time_grid.searchsorted(time_grid.scenario_grid[:, utils.TIME_GRID_MTM])

        # Canonical (mode-agnostic): (T, n_tenors) calibrated / (T, n_tenors, B) per-path.
        # AtT is (T, n_tenors); align it to the curve rank so the sum broadcasts. generate
        # appends the broadcast axis vs the stochastic component in `sim_curve`.
        AtT = AtT.type(shared.one.dtype)
        self.drift = fwd_curve + 0.5 * self.align_rank(AtT, fwd_curve.ndim)

    def calc_factors(self, factor1, factor1and2):
        if factor1.ndim == 2:
            F1, F2, KtT, HtT, YtT = self.F1, self.F2, self.KtT, self.HtT, self.YtT
        else:
            # Inner MC (T,B,B2): per-step (T,1) constants -> (T,1,1) and F2 (2,T,1) ->
            # (2,T,1,1) so they broadcast against the (2,)(T,B,B2) random tensors.
            F1 = self.F1.unsqueeze(-1)
            F2 = self.F2.unsqueeze(-1)
            KtT = [k.unsqueeze(-1) for k in self.KtT]
            HtT = [h.unsqueeze(-1) for h in self.HtT]
            YtT = [y.unsqueeze(-1) for y in self.YtT]
        f1 = torch.unsqueeze(
            (torch.cumsum(factor1 * F1, dim=0) - KtT[0] + HtT[0]) * YtT[0], dim=1)
        f2 = torch.unsqueeze(
            (torch.cumsum(torch.sum(factor1and2 * F2, dim=0), dim=0) - KtT[1] + HtT[1]) * YtT[1],
            dim=1)
        return f1, f2

    @property
    def correlation_name(self):
        return 'HWImpliedInterestRate', [('F1',), ('F2',)]

    def generate(self, shared_mem):
        """
        Simulate the implied 2-factor Hull-White curve.

        Dual-mode on the random-number rank: outer (T,B) or inner MC (T,B,B2). Under inner MC the
        per-tenor coefficients (n_tenors,1) -> (n_tenors,1,1) so they broadcast against the
        (T,1,B,B2) factor tensors, and drift is rank-aligned inside sim_curve (a per-outer-path
        curve (T,n_tenors,B) broadcasts across the B2 fan-out). Stochastic-deflation (grid_index)
        mode returns a dict of curves and is incompatible with nested simulation — the fork stores
        a single tensor per factor — so it raises there.
        """

        def sim_curve(drift, Bt0, Bt1, f1, f2, factor_tenor):
            stoch_component = Bt0 * f1 + Bt1 * f2
            return (self.align_rank(drift, stoch_component.ndim) + stoch_component) / factor_tenor

        rng1 = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        factor1, factor2 = self.calc_factors(
            rng1, shared_mem.t_random_numbers[self.z_offset:self.z_offset + 2, :self.scenario_horizon])

        if rng1.ndim == 3:
            # Inner MC (T,B,B2): per-tenor coeffs unsqueeze to broadcast; deflation unsupported.
            if self.grid_index is not None:
                raise NotImplementedError(
                    "HullWhite2FactorImpliedInterestRateModel stochastic-deflation (grid_index) "
                    "mode returns a dict of curves and is not supported under nested (inner-MC) "
                    "simulation. Run with Inner_MC_Enabled='No' or disable deflation.")
            return sim_curve(self.drift, self.BtT[0].unsqueeze(-1), self.BtT[1].unsqueeze(-1),
                             factor1, factor2, self.factor_tenor.unsqueeze(-1))

        # check if we need deflators
        if self.grid_index is not None:
            # if we have a grid index, then we want to simulate a reduced curve (just the first 6 months) over a
            # finer time grid - this is useful for stochastic deflation - note that at least 4 tenor points need
            # to be included to correctly handle interpolation
            reduced_tenor_index = max(4, self.factor.tenors.searchsorted(0.5) + 1)

            full_grid_curve = sim_curve(
                self.drift[self.grid_index], self.BtT[0], self.BtT[1],
                factor1[self.grid_index], factor2[self.grid_index], self.factor_tenor)

            partial_grid_curve = sim_curve(
                self.drift[:, :reduced_tenor_index], self.BtT[0][:reduced_tenor_index],
                self.BtT[1][:reduced_tenor_index], factor1, factor2, self.factor_tenor[:reduced_tenor_index])

            return {shared_mem.scenario_keys['full']: full_grid_curve,
                    shared_mem.scenario_keys['reduced']: partial_grid_curve}
        else:
            return sim_curve(self.drift, self.BtT[0], self.BtT[1], factor1, factor2, self.factor_tenor)


class HullWhite1FactorInterestRateModel(StochasticProcess):
    """Hull White 1 factor model

    """

    documentation = ('Interest Rates', [
        'The instantaneous spot rate (or short rate) which governs the evolution of the yield curve is modeled as:',
        '',
        '$$ dr(t) = (\\theta (t)-\\alpha r(t) - v(t)\\sigma(t)\\rho)dt + \\sigma(t) dW^*(t)$$',
        '',
        'Where:',
        '',
        '- $\\sigma (t)$ is a deterministic volatility curve',
        '- $\\alpha$ is a constant mean reversion speed',
        '- $\\theta (t)$ is a deterministic curve derived from the vol, mean reversion and initial discount factors',
        '- $v(t)$ is the quanto FX volatility and $\\rho$ is the quanto FX correlation',
        '- $dW^*(t)$ is the risk neutral Wiener process related to the real-world Wiener Process $dW(t)$',
        'by $dW^*(t)=dW(t)+\\lambda dt$ where $\\lambda$ is the market price of risk (assumed to be constant)',
        '',
        'Final form of the model is:',
        '$$ D(t,T) = \\frac{D(0,T)}{D(0,t)}exp\\Big(-\\frac{1}{2}A(t,T)-B(T-t)e^{-\\alpha t}(Y(t) -\\rho K(t) + '
        '\\lambda H(t))\\Big)$$',
        '',
        'Where:',
        '',
        '- $B(t) = \\frac{(1-e^{-\\alpha t})}{\\alpha}, Y(t)=\\int\\limits_0^t e^{\\alpha s}\\sigma (s) dW$',
        '- $A(t,T)=\\frac{B(T-t)e^{-\\alpha t}}{\\alpha}(2I(t)-(e^{-\\alpha t}+e^{-\\alpha T})J(t))$',
        '- $H(t) = \\int\\limits_0^t e^{\\alpha s}\\sigma (s)ds$',
        '- $I(t) = \\int\\limits_0^t e^{\\alpha s}{\\sigma (s)}^2ds$',
        '- $J(t) = \\int\\limits_0^t e^{2\\alpha s}{\\sigma (s)}^2ds$',
        '- $K(t) = \\int\\limits_0^t e^{\\alpha s}v(s){\\sigma (s)}ds$',
        '',
        'The simulation of the random increment $Y(t_{k+1})-Y(t_k)$ (where $0=t_0,t_1,t_2,...$',
        'represents the simulation grid) is normal with zero mean and variance $J(t_{k+1})-J(t_k)$'])

    factor_types = ('InterestRate', 'InflationRate', 'DividendRate')
    fields = [
        F('Alpha', 'Float', default=0),
        F('Lambda', 'Float', default=0),
        F('Sigma', 'Curve'),
        F('Quanto_FX_Correlation', 'Float', default=0),
        F('Quanto_FX_Volatility', 'Curve')
    ]

    def __init__(self, factor, param, implied_factor=None):
        super(HullWhite1FactorInterestRateModel, self).__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        # ensures that tenors used are the same as the price factor
        factor_tenor = self.factor.get_tenor()
        alpha = self.param['Alpha']

        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        # store the forward curve
        time_grid_years = np.array([self.factor.get_day_count_accrual(
            ref_date, t) for t in time_grid.scen_time_grid])
        self.fwd_curve = self.forward_curve(tensor, time_grid_years, shared)

        # (N,2) knot pairs, the same layout Sigma is read in below and the zeros fallback
        # carries - the transposed read this replaces handed integrate_piecewise_linear the
        # PAIRS as (tenors, values), garbage for any real curve and invisible to every
        # quanto=0 fixture
        quantofx = self.param['Quanto_FX_Volatility'].array if self.param['Quanto_FX_Volatility'] else np.zeros(
            (1, 2))
        quantofxcorr = self.param.get('Quanto_FX_Correlation', 0.0)

        # grab the vols
        vols = self.param['Sigma'].array

        # calculate known functions
        H = integrate_piecewise_linear(
            hw_calc_H(alpha, np.exp), shared, time_grid_years, vols[:, 0], vols[:, 1])
        I = integrate_piecewise_linear(
            hw_calc_IJK(alpha, np.exp), shared, time_grid_years,
            vols[:, 0], vols[:, 1], vols[:, 0], vols[:, 1])
        J = integrate_piecewise_linear(
            hw_calc_IJK(2.0 * alpha, np.exp), shared, time_grid_years,
            vols[:, 0], vols[:, 1], vols[:, 0], vols[:, 1])
        K = integrate_piecewise_linear(
            hw_calc_IJK(alpha, np.exp), shared, time_grid_years,
            vols[:, 0], vols[:, 1], quantofx[:, 0], quantofx[:, 1])

        # Now precalculate the A and B matrices
        BtT = (1.0 - np.exp(-alpha * factor_tenor)) / alpha
        AtT = np.array([(BtT / alpha) * np.exp(-alpha * t) * (
                2.0 * It - (np.exp(-alpha * t) + np.exp(-alpha * (t + factor_tenor))) * Jt) for (It, Jt, t) in
                        zip(I, J, time_grid_years)])

        # The increments fed to the cumsum must be of the RAW levels K, H - hw_calc_H/hw_calc_IJK
        # already carry e^{+alpha*s} in their integrands, and the single e^{-alpha*t} decay is
        # applied once at assembly in generate (exp_minus_alpha_t), exactly as HW2F and the
        # hazard model spell it. Pre-decaying the levels inside the diff telescopes the cumsum
        # to e^{-alpha*t}K(t), and the terminal multiply then made these two legs e^{-2alpha*t}:
        # measured ratio-to-exact of exactly e^{-alpha*t} at every node (12 digits, 10y grid),
        # invisible to every fixture because Lambda and Quanto_FX_Correlation are 0 everywhere.
        self.delta_KtT = shared.one.new_tensor(
            np.hstack((0.0, quantofxcorr * np.diff(K)))).reshape(-1, 1)

        self.delta_HtT = shared.one.new_tensor(np.hstack(
            (0.0, self.param['Lambda'] * np.diff(H)))).reshape(-1, 1)

        delta_var = np.diff(J)
        self.exp_minus_alpha_t = shared.one.new(np.exp(-alpha * time_grid_years)).reshape(-1, 1)

        if delta_var.size:
            # needed for numerical stability
            delta_var[delta_var < 0.0] = delta_var[delta_var >= 0.0].min()
        self.delta_vol = shared.one.new_tensor(np.sqrt(np.hstack((0.0, delta_var))).reshape(-1, 1))

        self.AtT = shared.one.new_tensor(AtT)
        self.BtT = shared.one.new_tensor(BtT.reshape(-1, 1))
        # Canonical (mode-agnostic): (T, n_tenors) calibrated / (T, n_tenors, B) per-path.
        # AtT is (T, n_tenors); align it to the curve rank so the sum broadcasts.
        self.fwd_component = self.fwd_curve + 0.5 * self.align_rank(self.AtT, self.fwd_curve.ndim)
        self.factor_tenor = shared.one.new_tensor(factor_tenor.reshape(-1, 1))

    @property
    def correlation_name(self):
        return 'HWInterestRate', [('F1',)]

    def generate(self, shared_mem):
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        if Z.ndim == 2:
            f1 = (Z * self.delta_vol - self.delta_KtT + self.delta_HtT).cumsum(axis=0) * self.exp_minus_alpha_t
            stoch_component = self.BtT * torch.unsqueeze(f1, dim=1)
            fwd_component = self.align_rank(self.fwd_component, stoch_component.ndim)
            return (fwd_component + stoch_component) / self.factor_tenor
        # Inner MC (T,B,B2): per-step (T,1) arrays -> (T,1,1); per-tenor coefficients
        # BtT/factor_tenor (n_tenors,1) -> (n_tenors,1,1); fwd_component rank-aligned so a
        # per-outer-path curve (T,n_tenors,B) broadcasts across the B2 fan-out.
        f1 = (Z * self.delta_vol.unsqueeze(-1) - self.delta_KtT.unsqueeze(-1)
              + self.delta_HtT.unsqueeze(-1)).cumsum(axis=0) * self.exp_minus_alpha_t.unsqueeze(-1)
        stoch_component = self.BtT.unsqueeze(-1) * torch.unsqueeze(f1, dim=1)
        fwd_component = self.align_rank(self.fwd_component, stoch_component.ndim)
        return (fwd_component + stoch_component) / self.factor_tenor.unsqueeze(-1)


class HWInterestRateCalibration(object):
    """Mean reversion and reversion volatility averaged across the curve's tenors. Takes no
    tuning - the name is also why a calibration has to declare the process it calibrates."""
    model_type = 'HullWhite1FactorInterestRateModel'
    fields = []

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Diff',
                                                          num_business_days=num_business_days, max_alpha=4.0)
        alpha = stats['Mean Reversion Speed'].mean()
        sigma = stats['Reversion Volatility'].mean()
        correlation_coef = np.array([np.array([1.0 / np.sqrt(correlation.values.sum())] * tenor.size)])

        return utils.CalibrationInfo(
            {'Lambda': 0.0, 'Alpha': alpha, 'Sigma': utils.Curve([], [(0.0, sigma)]), 'Quanto_FX_Correlation': 0.0,
             'Quanto_FX_Volatility': 0.0}, correlation_coef, delta)


class HWHazardRateModel(StochasticProcess):
    """Hull White 1 factor hazard Rate model"""

    documentation = ('Survival Rates',
                     ['The Hull-White instantaneous hazard rate process is modeled as:',
                      '',
                      '$$ dh(t) = (\\theta (t)-\\alpha h(t))dt + \\sigma dW^*(t)$$',
                      '',
                      'All symbols defined as per Hull White 1 factor for interest rates.'
                      '',
                      'The final form of the model is',
                      '',
                      '$$ S(t,T) = \\frac{S(0,T)}{S(0,t)}exp\\Big(-\\frac{1}{2}A(t,T)-\\sigma B(T-t)(Y(t) + B(t)'
                      '\\lambda)\\Big)$$',
                      '',
                      'Where:',
                      '',
                      '- $B(t) = \\frac{1-e^{-\\alpha t}}{\\alpha}$, $Y(t) \\sim N(0, \\frac{1-e^{-2 \\alpha t}}{2'
                      '\\alpha})$',
                      '- $A(t,T)=\\sigma^2 B(T-t)\\Big(B(T-t)\\frac{B(2t)}{2}+B(t)^2\\Big)$'])

    factor_types = ('SurvivalProb',)
    fields = [
        F('Alpha', 'Float', default=0),
        F('Lambda', 'Float', default=0),
        F('Sigma', 'Float', default=0)
    ]

    def __init__(self, factor, param, implied_factor=None):
        super(HWHazardRateModel, self).__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        alpha = self.param['Alpha']
        factor_tenor = self.factor.get_tenor()

        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        time_grid_years = np.array([self.factor.get_day_count_accrual(
            ref_date, t) for t in time_grid.scen_time_grid])

        # store the forward curve    
        self.fwd_curve = self.forward_curve(tensor, time_grid_years, shared, mul_time=False)
        Bt = ((1.0 - np.exp(-alpha * time_grid_years)) / alpha).reshape(-1, 1)
        B2t = ((1.0 - np.exp(-2.0 * alpha * time_grid_years)) / alpha).reshape(-1, 1)
        sigma2 = self.param['Sigma'] ** 2
        BtT = ((1.0 - np.exp(-alpha * factor_tenor)) / alpha).reshape(1, -1)
        AtT = sigma2 * BtT * (0.5 * BtT * B2t + Bt ** 2)

        # OU variance: (1 - exp(-2αt)) / (2α) == 0.5 · B2t (reuse, don't recompute the exp)
        var = 0.5 * B2t
        delta_var = np.diff(np.insert(var, 0, 0, axis=0), axis=0)
        self.delta_vol = shared.one.new_tensor(np.sqrt(delta_var).reshape(-1, 1))

        # convert to tensors
        self.AtT = shared.one.new_tensor(AtT)
        self.BtT = shared.one.new_tensor(BtT)
        # Canonical (mode-agnostic): (T, n_tenors) calibrated / (T, n_tenors, B) per-path.
        # AtT is (T, n_tenors); align it to the curve rank so the sum broadcasts.
        self.fwd_component = self.fwd_curve + 0.5 * self.align_rank(self.AtT, self.fwd_curve.ndim)
        self.Bt = shared.one.new_tensor(Bt) if self.param['Lambda'] else 0.0

    @property
    def correlation_name(self):
        return 'HullWhiteProcess', [()]

    def generate(self, shared_mem):
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        if Z.ndim == 2:
            f1 = (Z * self.delta_vol).cumsum(dim=0)
            # add market price of risk (if non-zero):
            f1 = f1 + self.param['Lambda'] * self.Bt
            stoch_component = self.param['Sigma'] * torch.unsqueeze(self.BtT, dim=2) * torch.unsqueeze(f1, dim=1)
            return self.align_rank(self.fwd_component, stoch_component.ndim) + stoch_component
        # Inner MC (T,B,B2): per-step delta_vol/Bt (T,1) -> (T,1,1); per-tenor BtT
        # (1,n_tenors) -> (1,n_tenors,1,1); fwd_component rank-aligned across the B2 fan-out.
        f1 = (Z * self.delta_vol.unsqueeze(-1)).cumsum(dim=0)
        bt = self.Bt.unsqueeze(-1) if torch.is_tensor(self.Bt) else self.Bt
        f1 = f1 + self.param['Lambda'] * bt
        stoch_component = self.param['Sigma'] * torch.unsqueeze(self.BtT, dim=2).unsqueeze(-1) \
            * torch.unsqueeze(f1, dim=1)
        return self.align_rank(self.fwd_component, stoch_component.ndim) + stoch_component


class HWHazardRateCalibration(object):
    """Mean reversion and reversion volatility of a hazard rate curve. Takes no tuning."""
    model_type = 'HWHazardRateModel'
    fields = []

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(data_frame, method='Diff',
                                                          num_business_days=num_business_days, max_alpha=4.0)
        alpha = stats['Mean Reversion Speed'].mean()
        sigma = stats['Reversion Volatility'].values[0] / tenor[0]
        correlation_coef = np.array([np.array([1.0 / np.sqrt(correlation.values.sum())] * tenor.size)])

        return utils.CalibrationInfo({'Alpha': alpha, 'Sigma': sigma, 'Lambda': 0}, correlation_coef, delta)


class CSForwardPriceModel(StochasticProcess):
    """Clewlow-Strickland Model"""

    documentation = ('Energy Pricing',
                     ['For commodity/Energy deals, the Forward price is modeled directly. For each settlement date T,',
                      'the SDE for the forward price is:',
                      '',
                      '$$ dF(t,T) = \\mu F(t,T)dt + \\sigma e^{-\\alpha(T-t)}F(t,T)dW(t)$$',
                      '',
                      'Where:',
                      '',
                      '- $\\mu$ is the drift rate',
                      '- $\\sigma$ is the volatility',
                      '- $\\alpha$ is the mean reversion speed',
                      '- $W(t)$ is the standard Weiner Process',
                      '',
                      'Final form of the model is',
                      '',
                      '$$ F(t,T) = F(0,T)exp\\Big(\\mu t-\\frac{1}{2}\\sigma^2e^{-2\\alpha(T-t)}v(t)+\\sigma '
                      'e^{-\\alpha(T-t)}Y(t)\\Big)$$',
                      '',
                      'Where $Y$ is a standard Ornstein-Uhlenbeck Process with variance:',
                      '',
                      '$$v(t) = \\frac{1-e^{-2\\alpha t}}{2\\alpha}$$',
                      '',
                      'The spot rate is given by',
                      '',
                      '$$S(t)=F(t,t)=F(0,t)exp\\Big(\\mu t-\\frac{1}{2}\\sigma^2v(t)+\\sigma Y(t)\\Big)$$',
                      ''])

    factor_types = ('ForwardPrice',)
    fields = [
        F('Alpha', 'Float', default=0),
        F('Drift', 'Float', default=0),
        F('Sigma', 'Float', default=0)
    ]

    def __init__(self, factor, param, implied_factor=None):
        super(CSForwardPriceModel, self).__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        """
        Build the per-step vol/drift and the initial curve for the CS forward-price process.

        `tensor` is (n_tenors,) calibrated or (n_tenors, B) per-path (inner-MC fork or diff-ML t=0
        burn-in). `initial_curve` is stored in a generate-mode-agnostic canonical form — a leading
        time-broadcast axis plus n_tenors[, B]; the mode-specific trailing axes are appended in
        `generate`, which knows outer from inner via Z.ndim. precalc cannot make that call, since a
        per-path burn-in init and an inner-MC init are both (n_tenors, B). vol/drift are functions
        of the time grid and the params only, shaped (T, n_tenors, 1).
        """
        # canonical initial_curve: leading time-broadcast axis; generate appends the mode axis.
        if tensor.ndim == 1:
            self.initial_curve = tensor.reshape(1, -1)                      # (1, n_tenors)
        else:
            self.initial_curve = tensor.unsqueeze(0)                        # (1, n_tenors, B)
        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        #  rebase the dates
        excel_offset = (ref_date - utils.excel_offset).days
        excel_date_time_grid = time_grid.scen_time_grid + excel_offset
        tenors = (self.factor.get_tenor().reshape(1, -1) -
                  excel_date_time_grid.reshape(-1, 1)).clip(0.0, np.inf) / utils.DAYS_IN_YEAR
        tenor_rel = self.factor.get_tenor() - excel_offset
        delta = tenor_rel.reshape(1, -1).clip(
            time_grid.scen_time_grid[:-1].reshape(-1, 1),
            time_grid.scen_time_grid[1:].reshape(-1, 1)
        ) - time_grid.scen_time_grid[:-1].reshape(-1, 1)
        dt = np.insert(delta, 0, 0, axis=0) / utils.DAYS_IN_YEAR

        if implied_tensor is None:
            # need to scale the vol (as the variance is modelled using an OU Process)
            var_adj = (1.0 - np.exp(-2.0 * self.param['Alpha'] * dt.cumsum(axis=0))) / (2.0 * self.param['Alpha'])
            var = np.square(self.param['Sigma']) * np.exp(-2.0 * self.param['Alpha'] * tenors) * var_adj
            # get the vol
            vol = np.sqrt(np.diff(np.insert(var, 0, 0, axis=0), axis=0))
            self.vol = tensor.new(np.expand_dims(vol, axis=2))
            self.drift = tensor.new(np.expand_dims(self.param['Drift'] * dt.cumsum(axis=0) - 0.5 * var, axis=2))
        else:
            # need to scale the vol (as the variance is modelled using an OU Process)
            var_adj = (1.0 - torch.exp(-2.0 * implied_tensor['Alpha'] * tensor.new(dt.cumsum(axis=0)))) / (
                    2.0 * implied_tensor['Alpha'])
            var = torch.square(implied_tensor['Sigma']) * torch.exp(
                -2.0 * implied_tensor['Alpha'] * tensor.new(tenors)) * var_adj
            delta_var = torch.diff(nnf.pad(var, [0, 0, 1, 0]), dim=0)
            safe_delta = torch.where(delta_var > 0.0, delta_var, torch.ones_like(delta_var))
            vol = torch.where(delta_var > 0.0, torch.sqrt(safe_delta), torch.zeros_like(delta_var))
            self.vol = torch.unsqueeze(vol, dim=2)
            self.drift = torch.unsqueeze(-0.5 * var, dim=2)

    @property
    def correlation_name(self):
        return 'ClewlowStricklandProcess', [()]

    def generate(self, shared_mem):
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        if Z.ndim == 2:
            # Outer mode: Z is (T, B). Bit-exact preserves legacy behavior.
            z_portion = Z.unsqueeze(1) * self.vol                       # (T, n_tenors, B)
            path = torch.exp(self.drift + torch.cumsum(z_portion, dim=0))
        else:
            # Inner mode: Z is (T, B, B2). One path per outer × inner.
            vol4 = self.vol.unsqueeze(-1)                              # (T, n_tenors, 1, 1)
            drift4 = self.drift.unsqueeze(-1)                          # (T, n_tenors, 1, 1)
            z_portion = Z.unsqueeze(1) * vol4                          # (T, n_tenors, B, B2)
            path = torch.exp(drift4 + torch.cumsum(z_portion, dim=0))
        # Align the canonical initial_curve ((1, n_tenors) calibrated, (1, n_tenors, B)
        # per-path) to the path rank — broadcasts a calibrated curve across the batch,
        # an inner/burn-in curve element-wise.
        return self.align_rank(self.initial_curve, path.ndim) * path


class CSImpliedForwardPriceModel(CSForwardPriceModel):
    """The Clewlow Strickland Stochastic Process with implied vol and mean reversion"""

    documentation = ('Energy Pricing',
                     ['The risk-neutral (implied) variant of the Clewlow-Strickland forward price '
                      'model. The drift is set to zero, so the forward price is a martingale under '
                      'the pricing measure. For each settlement date T, the SDE for the forward '
                      'price is:',
                      '',
                      '$$ dF(t,T) = \\sigma e^{-\\alpha(T-t)}F(t,T)dW(t)$$',
                      '',
                      'Where:',
                      '',
                      '- $\\sigma$ is the volatility',
                      '- $\\alpha$ is the mean reversion speed',
                      '- $W(t)$ is the standard Weiner Process',
                      '',
                      'The volatility $\\sigma$ and mean reversion speed $\\alpha$ are read from the '
                      'implied vol surface.',
                      '',
                      'Final form of the model is',
                      '',
                      '$$ F(t,T) = F(0,T)exp\\Big(-\\frac{1}{2}\\sigma^2e^{-2\\alpha(T-t)}v(t)+\\sigma '
                      'e^{-\\alpha(T-t)}Y(t)\\Big)$$',
                      '',
                      'Where $Y$ is a standard Ornstein-Uhlenbeck Process with variance:',
                      '',
                      '$$v(t) = \\frac{1-e^{-2\\alpha t}}{2\\alpha}$$',
                      '',
                      'The spot rate is given by',
                      '',
                      '$$S(t)=F(t,t)=F(0,t)exp\\Big(-\\frac{1}{2}\\sigma^2v(t)+\\sigma Y(t)\\Big)$$',
                      ''])

    factor_types = ('ForwardPrice',)
    # the parameters are the implied factor's, not this block's - it carries none of its own
    fields = []

    def __init__(self, factor, param, implied_factor=None):
        super(CSImpliedForwardPriceModel, self).__init__(factor, param)
        # every parameter is read off the implied tensor in precalculate; a param-dict copy
        # here would be a second source no path reads (and one that could silently go stale)
        self.implied = implied_factor


class CSForwardPriceCalibration(object):
    """Clewlow-Strickland sigma, alpha and drift from log statistics. Takes no tuning."""
    model_type = 'CSForwardPriceModel'
    fields = []

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(
            data_frame, method='Log', num_business_days=num_business_days, max_alpha=5.0)
        alpha = stats['Mean Reversion Speed'].values[0]
        sigma = stats['Reversion Volatility'].values[0]
        mu = stats['Drift'].values[0] + 0.5 * (stats['Volatility'].values[0]) ** 2
        correlation_coef = np.array([np.array([1.0 / np.sqrt(correlation.values.sum())] * tenor.size)])

        return utils.CalibrationInfo({'Sigma': sigma, 'Alpha': alpha, 'Drift': mu}, correlation_coef, delta)


class PCAInterestRateModel(StochasticProcess):
    """The Principal Component Analysis model for interest rate curves Stochastic Process - defines the python
    interface and the low level cuda code"""

    documentation = (
        'Interest Rates',
        ['The parameters of the model are:',
         '- a volatility curve $\\sigma_\\tau$ for each tenor $\\tau$ of the zero curve $r_\\tau$',
         '- a mean reversion parameter $\\alpha$',
         '- eigenvalues $\\lambda_1,\\lambda_2,..,\\lambda_m$ and corresponding eigenvectors $Q_1(\\tau),Q_2(\\tau),'
         '...,Q_m(\\tau)$',
         '- optionally a historical yield curve $\\Theta(\\tau)$ for the long run mean of $r_\\tau$',
         '',
         'The stochastic process for the rate at each tenor on the interest rate curve is specified as:',
         '',
         '$$ dr_\\tau = r_\\tau ( u_\\tau  dt + \\sigma_\\tau dY )$$',
         '',
         '$$ dY_t = -\\alpha Ydt + dZ$$',
         '',
         'with $dY$  a standard Ornstein-Uhlenbeck process and $dZ$ a Brownian motion. It can be shown that:',
         '',
         '$$ Y(t) \\sim N(0, \\frac{1-e^{-2 \\alpha t}}{2 \\alpha})$$ ',
         '',
         'Currently, only the covarience matrix is used to define the eigenvectors with corresponding weight curves',
         '$w_k(\\tau)=Q_k(\\tau)\\frac{\\sqrt\\lambda_k}{\\sigma_\\tau}$ and normalized weight curve'
         '$$B_k(\\tau)=\\frac{w_k(\\tau)}{\\sqrt{\\sum_{l=1}^m w_l(\\tau)^2}}$$'
         '',
         'Final form of the model is',
         '',
         '$$ r_\\tau(t) = R_\\tau(t) exp \\Big( -\\frac{1}{2} \\sigma_\\tau^2 (\\frac{1-e^{-2 \\alpha t}}{2 \\alpha}) '
         '+ \\sigma_\\tau \\sum_{k=1}^{3} B_k(\\tau) Y_k(t) \\Big)$$',
         '',
         'Where:',
         '',
         '- $r_\\tau(t)$ is the zero rate with a tenor $\\tau$  at time $t$  ($t = 0$ denotes the current rate at '
         'tenor $\\tau$)',
         '- $\\alpha$ is the mean reversion level of zero rates',
         '- $Y_k(t)$ is the OU process associated with Principle component $k$',
         '',
         'To simulate the mean rate $R_\\tau(t)$ (note that $R_\\tau(0)=r_\\tau(0)$ ), there are 2 choices:',
         '',
         '**Drift To Forward** where the mean rate is the inital forward rate from $t$ to $t+\\tau$ so that',
         '',
         '$$\\frac{D(0,t+\\tau)}{D(0,t)}=e^{R_\\tau(t)\\tau}$$',
         '',
         '**Drift To Blend** is a weighted average function of the current rate and a mean reversion level',
         '$\\Theta_\\tau$',
         '',
         '$$R_\\tau(t)=[e^{-\\alpha t}r_\\tau (0) + (1-e^{-\\alpha t})\\Theta_\\tau]$$',
         ''
         ])

    factor_types = ('InterestRate', 'InflationRate', 'DividendRate')
    fields = [
        F('Reversion_Speed', 'Float', default=0),
        F('Historical_Yield', 'Curve'),
        F('Yield_Volatility', 'Curve'),
        F('Eigenvectors', 'Curve', default='[{"label":"1", "data":[[0.0,0.0]]},'
                                           '{"label":"2", "data":[[0.0,0.0]]},'
                                           '{"label":"3", "data":[[0.0,0.0]]}]'),
        F('Rate_Drift_Model', 'Text', default='Drift_To_Forward',
          values=['Drift_To_Forward', 'Drift_To_Blend']),
        F('Princ_Comp_Source', 'Text', default='Correlation',
          values=['Correlation', 'Covariance']),
        F('Distribution_Type', 'Text', default='Lognormal', values=['Lognormal', 'Normal'])
    ]

    def __init__(self, factor, param, implied_factor=None):
        super(PCAInterestRateModel, self).__init__(factor, param)
        # need to precalculate these for a specific set of tenors
        self.evecs = None
        self.vols = None

    def num_factors(self):
        return len(self.param['Eigenvectors'])

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        """
        Precompute the exact-OU step coefficients, Ito drift and forward curve for the PCA model.

        Steps are anchored at time_grid_years[0] so the per-step dt and `elapsed` (time since sim
        start) are correct under both outer mode (anchor = 0) and inner-MC kept-base mode
        (anchor > 0). Exact OU discretisation:
            Y_{k+1} = exp(-α Δt_k) Y_k + sqrt((1-exp(-2α Δt_k))/(2α)) Z_{k+1}

        `fwd_component` is stored in a canonical, generate-mode-agnostic form: (T, n_tenors)
        calibrated or (T, n_tenors, B) per-path. The mode-specific trailing axis is appended in
        `generate`, which knows outer from inner via the factor rank; precalc cannot make that
        call, since a per-path burn-in init and an inner-MC init are both (n_tenors, B).
        """
        # ensures that tenors used are the same as the price factor
        factor_tenor = self.factor.get_tenor()

        # store randomnumber id's
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        time_grid_years = np.array([self.factor.get_day_count_accrual(
            ref_date, t) for t in time_grid.scen_time_grid])

        # rescale and precalculate the eigenvectors
        evecs, evals = np.zeros((factor_tenor.size, self.num_factors())), []
        for index, eigen_data in enumerate(self.param['Eigenvectors']):
            evecs[:, index] = np.interp(factor_tenor, *eigen_data['Eigenvector'].array.T)
            evals.append(eigen_data['Eigenvalue'])

        # note that I don't need to divide by the volatility because I normalize
        # across tenors below . . .
        B = evecs.dot(np.diag(np.sqrt(evals)))
        B /= np.linalg.norm(B, axis=1).reshape(-1, 1)

        self.vols = np.interp(factor_tenor, *self.param['Yield_Volatility'].array.T)
        alpha = self.param['Reversion_Speed']

        # Anchored at time_grid_years[0]; exact OU discretisation.
        dt_steps   = np.diff(np.append([time_grid_years[0]], time_grid_years))  # [T]
        elapsed    = dt_steps.cumsum()                                        # [T] — time since sim start
        ou_decay   = np.exp(-alpha * dt_steps)                               # [T]
        ou_std     = np.sqrt((1.0 - np.exp(-2.0 * alpha * dt_steps)) / (2.0 * alpha))  # [T]
        self.ou_decay    = shared.one.new_tensor(ou_decay.reshape(-1, 1))    # [T, 1]
        self.ou_noise    = shared.one.new_tensor(ou_std.reshape(-1, 1))      # [T, 1]
        self.vols_tensor = shared.one.new_tensor(self.vols.reshape(-1, 1))   # [n_tenors, 1]

        # Ito drift: -½ σ_τ² Var(Y(t_k)) = -½ σ_τ² (1-exp(-2α t_k))/(2α)  (full value at each t_k)
        ou_var_cumul = (1.0 - np.exp(-2.0 * alpha * elapsed)) / (2.0 * alpha)  # [T]
        self.drift = shared.one.new_tensor(np.expand_dims(
            -0.5 * (self.vols * self.vols).reshape(1, -1) * ou_var_cumul.reshape(-1, 1), axis=2))

        # normalize the eigenvectors
        self.evecs = shared.one.new_tensor(B.T)                         # [n_factors, n_tenors]

        # Forward curve at each time-grid point. `tensor` is the current zero curve:
        # (n_tenors,) in outer mode, (n_tenors, B) in inner mode (per-outer-path init).
        if self.param['Rate_Drift_Model'] == 'Drift_To_Blend':
            hist_mean = scipy.interpolate.interp1d(*np.hstack(
                ([[0.0], [self.param['Historical_Yield'].array.T[-1][0]]],
                 self.param['Historical_Yield'].array.T)), kind='linear', bounds_error=False,
                    fill_value=self.param['Historical_Yield'].array.T[-1][-1])
            omega = shared.one.new_tensor(hist_mean(self.factor.tenors).reshape(1, -1))        # [1, n_tenors]
            decay = shared.one.new_tensor(np.exp(-alpha * elapsed).reshape(-1, 1))             # [T, 1]
            # R_τ(t) = exp(-α t) r_τ(0) + (1 - exp(-α t)) Θ_τ  (t is time since sim start)
            if tensor.ndim == 1:
                curve_t0 = tensor.reshape(1, -1)                                                # [1, n_tenors]
                fwd_curve = decay * curve_t0 + (1.0 - decay) * omega                            # [T, n_tenors]
            else:
                curve_t0 = tensor.unsqueeze(0)                                                  # [1, n_tenors, B]
                decay_b = decay.unsqueeze(-1)                                                   # [T, 1, 1]
                omega_b = omega.unsqueeze(-1)                                                   # [1, n_tenors, 1]
                fwd_curve = decay_b * curve_t0 + (1.0 - decay_b) * omega_b                      # [T, n_tenors, B]
        else:
            # Batch-aware forward curve (calibrated (n_tenors,) → [T, n_tenors];
            # per-path (n_tenors, B) → [T, n_tenors, B]) via the base-class seam, then
            # divide by the tenor with the divisor rank-aligned to the curve.
            fwd = self.forward_curve(tensor, elapsed, shared)
            factor_tenor_t = shared.one.new_tensor(factor_tenor.reshape(1, -1))                 # [1, n_tenors]
            fwd_curve = fwd / self.align_rank(factor_tenor_t, fwd.ndim)

        # Canonical mode-agnostic form; generate appends the mode-specific trailing axis.
        self.fwd_component = fwd_curve

    def calc_factors(self, factors):
        """
        Project the OU factor innovations onto the (normalised) PCA eigenvectors.

        `factors` is [n_factors, T, B] outer or [n_factors, T, B, B2] inner MC; the branch on ndim
        changes only the broadcasting shapes, the vectorisation is identical.

        Closed-form vectorisation of Y_{k+1} = d_k * Y_k + n_k * Z_k  (Y_0 = 0):
            Y_out[k] = D[k] * cumsum( n[i]/D[i] * Z[i] )[k],  D[k] = cumprod(d)[k]
        Two O(T) CUDA-native ops (cumprod, cumsum) replace the T-step Python loop. Numerically
        stable for typical PCA α (0.01–0.3); avoid α >> 1 on long grids.
        """
        evecs_mat = self.evecs                                         # [n_factors, n_tenors]
        D = torch.cumprod(self.ou_decay, dim=0)                         # [T, 1]

        if factors.ndim == 3:
            Y = D.unsqueeze(0) * torch.cumsum(
                (self.ou_noise / D).unsqueeze(0) * factors, dim=1)      # [n_factors, T, B]
            projected = torch.einsum('jt,jks->kts', evecs_mat, Y)       # [T, n_tenors, B]
            return self.drift + self.vols_tensor.unsqueeze(0) * projected
        else:
            D4 = D.view(1, -1, 1, 1)                                    # [1, T, 1, 1]
            noise4 = (self.ou_noise / D).view(1, -1, 1, 1)              # [1, T, 1, 1]
            Y = D4 * torch.cumsum(noise4 * factors, dim=1)              # [n_factors, T, B, B2]
            projected = torch.einsum('jt,jkbs->ktbs', evecs_mat, Y)     # [T, n_tenors, B, B2]
            return self.drift.unsqueeze(-1) + self.vols_tensor.view(1, -1, 1, 1) * projected

    @property
    def correlation_name(self):
        return 'InterestRateOUProcess', [('PC{}'.format(x),) for x in range(1, self.num_factors() + 1)]

    def generate(self, shared_mem):
        stoch = self.calc_factors(
            shared_mem.t_random_numbers[
            self.z_offset:self.z_offset + self.num_factors(), :self.scenario_horizon])

        # Align the canonical fwd_component ((T, n_tenors) calibrated, (T, n_tenors, B)
        # per-path) to stoch's rank ((T, n_tenors, B) outer, (T, n_tenors, B, B2) inner)
        # — broadcasts a calibrated curve across the batch, a per-path curve element-wise.
        return self.align_rank(self.fwd_component, stoch.ndim) * torch.exp(stoch)

    def reveal_state_at(self, t, buffer):
        # Excluded from the V̂ market state: the tradable futures already carry this curve's
        # hedging content; revealing every tenor bloats the basis.
        return []


class PCAInterestRateCalibration(object):
    """Principal components of the tenor covariance, with the three shape choices stamped through
    onto the emitted `Price Models` block - which is why all three are hard-keyed and required."""
    model_type = 'PCAInterestRateModel'
    fields = [
        F('Distribution_Type', 'Text', default=REQUIRED, values=['Lognormal', 'Normal'],
          description='Whether the simulated rate is lognormal or normal'),
        F('Matrix_Type', 'Text', default=REQUIRED, values=['Correlation', 'Covariance'],
          description='The matrix the principal components are taken of - written out as '
                      'Princ_Comp_Source'),
        F('Rate_Drift_Model', 'Text', default=REQUIRED,
          values=['Drift_To_Forward', 'Drift_To_Blend'],
          description='How the mean rate is simulated - see the PCAInterestRateModel theory')
    ]

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 3

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        min_rate = data_frame.min().min()
        force_positive = 0.0 #if min_rate > 0.0 else -5.0 * min_rate
        tenor = np.array([(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta = utils.calc_statistics(data_frame + force_positive, method='Log',
                                                          num_business_days=num_business_days, max_alpha=4.0)

        standard_deviation = stats['Reversion Volatility'].interpolate()
        covariance = np.dot(standard_deviation.values.reshape(-1, 1),
                            standard_deviation.values.reshape(1, -1)) * correlation
        aki, evecs, evals = utils.PCA(covariance, self.num_factors)
        meanReversionSpeed = stats['Mean Reversion Speed'].mean()
        volCurve = standard_deviation
        reversionLevel = stats['Long Run Mean'].interpolate().bfill().ffill()
        correlation_coef = aki.T

        return utils.CalibrationInfo(
            {
                'Reversion_Speed': meanReversionSpeed,
                'Historical_Yield': utils.Curve([], list(zip(tenor, reversionLevel))),
                'Yield_Volatility': utils.Curve([], list(zip(tenor, volCurve))),
                'Eigenvectors': [
                    {'Eigenvector': utils.Curve([], list(zip(tenor, evec))), 'Eigenvalue': eval}
                    for evec, eval in zip(evecs.real.T, evals.real)],
                'Rate_Drift_Model': self.param['Rate_Drift_Model'],
                'Princ_Comp_Source': self.param['Matrix_Type'],
                'Distribution_Type': self.param['Distribution_Type']
            },
            correlation_coef,
            delta
        )


class LogOUSpotModel(StochasticProcess):
    """Log-space Ornstein-Uhlenbeck spot-price model.

    Latent state X_t = log(S_t) follows:

        dX_t = Kappa * (Theta - X_t) dt + Sigma dW_t

    The exact discretisation over an interval dt is:

        X_{t+dt} = Theta + exp(-Kappa*dt) * (X_t - Theta) + eps
        eps ~ N(0, Sigma^2 * (1 - exp(-2*Kappa*dt)) / (2*Kappa))

    Simulated paths are returned as S_t = exp(X_t), which are always positive.
    One correlated Gaussian driver is consumed per step.
    """

    documentation = (
        'Asset Pricing',
        ['The log-spot $X_t = \\log S_t$ follows a mean-reverting Ornstein-Uhlenbeck process:',
         '',
         '$$ dX_t = \\kappa(\\theta - X_t)\\,dt + \\sigma\\,dW_t $$',
         '',
         'The exact discretisation over a step $\\delta$ is:',
         '',
         '$$ X_{t+\\delta} = \\theta + e^{-\\kappa\\delta}(X_t - \\theta)'
         ' + \\sigma\\sqrt{\\frac{1-e^{-2\\kappa\\delta}}{2\\kappa}}\\,Z,'
         '\\quad Z\\sim\\mathcal{N}(0,1) $$',
         '',
         'Simulated paths are returned as $S_t = \\exp(X_t) > 0$.',
         '',
         'Parameters:',
         '',
         '- **Spot**: Initial spot price $S_0 > 0$',
         '- **Kappa**: Mean-reversion speed $\\kappa > 0$',
         '- **Theta**: Long-run mean of $\\log S$',
         '- **Sigma**: Volatility $\\sigma \\geq 0$',
         '',
         'The stationary distribution of $\\log S$ is'
         ' $\\mathcal{N}\\!\\left(\\theta,\\,\\frac{\\sigma^2}{2\\kappa}\\right)$.'])

    factor_types = ('CommodityPrice',)
    fields = [
        F('Kappa', 'Float', default=0),
        F('Theta', 'Float', default=0),
        F('Sigma', 'Float', default=0)
    ]

    def __init__(self, factor, param, implied_factor=None):
        super(LogOUSpotModel, self).__init__(factor, param)
        self._validate_params()

    def _validate_params(self):
        p = self.param
        if p.get('Kappa', 0.0) <= 0.0:
            self.params_ok = False
            return
        if p.get('Sigma', 0.0) < 0.0:
            self.params_ok = False
            return

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        """
        Precompute the exact-OU step coefficients, the initial log-spot and the reversion target.

        θ is anchored at the current spot rather than at the calibrated long-run mean: production
        hedging should not bet that today's price reverts to a historical average — the agent
        hedges variance around the current regime, not directional drift toward a stale θ. Set
        `Anchor_Theta_At_Spot: false` for backtests that want the absolute calibrated θ.
        """
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        dt = np.diff(np.hstack(([0.0], time_grid.time_grid_years)))
        kappa = self.param['Kappa']
        sigma = self.param['Sigma']

        # exact OU step: mean-reversion factor and conditional std-dev
        e_kdt   = np.exp(-kappa * dt)
        var_step = sigma * sigma * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)

        self.e_kdt   = tensor.new(e_kdt.reshape(-1, 1))
        self.ou_vol  = tensor.new(np.sqrt(var_step).reshape(-1, 1))
        # Initial log-spot. AAD: keep on graph. `tensor` is the factor's `Spot`: (1,)
        # calibrated, or (B,) per-path (inner-MC fork / diff-ML t=0 burn-in). reshape(-1)
        # keeps the batch axis (a calibrated (1,) broadcasts like the old 0-d scalar).
        self.log_spot0 = torch.log(tensor).reshape(-1)
        # θ anchored at the current spot unless Anchor_Theta_At_Spot is disabled.
        if bool(self.param.get('Anchor_Theta_At_Spot', True)):
            self.theta = self.log_spot0                                                # 0-d tensor, on graph
        else:
            self.theta = tensor.new_tensor(float(self.param['Theta']))                 # constant, no grad

    @property
    def correlation_name(self):
        return 'OULogSpotProcess', [()]

    def generate(self, shared_mem):
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]  # [T, batch]
        if Z.ndim == 2:
            A = torch.cumprod(self.e_kdt, dim=0)                           # [T, 1]
            b = self.theta * (1.0 - self.e_kdt) + self.ou_vol * Z         # [T, batch]
            log_spot = A * (self.log_spot0 + torch.cumsum(b / A, dim=0))  # [T, batch]
            return torch.exp(log_spot)
        # Inner MC (T, B, B2): per-step arrays (T,1) -> (T,1,1); per-outer-path theta /
        # log_spot0 (B,)/(1,) -> (1,B,1)/(1,1,1) so each outer path broadcasts across B2.
        e_kdt = self.e_kdt.unsqueeze(-1)
        A = torch.cumprod(e_kdt, dim=0)
        theta = self.align_rank(self.theta.unsqueeze(0), Z.ndim)
        b = theta * (1.0 - e_kdt) + self.ou_vol.unsqueeze(-1) * Z
        log_spot0 = self.align_rank(self.log_spot0.unsqueeze(0), Z.ndim)
        log_spot = A * (log_spot0 + torch.cumsum(b / A, dim=0))
        return torch.exp(log_spot)

    @classmethod
    def privileged_layout(cls, param):
        return {'log_deviation': 1, 'kappa': 1, 'sigma': 1}

    def privileged_factors(self, simulated):
        spot = simulated.to(dtype=torch.float32)
        # Use self.theta (post-anchor) so the critic sees the actual θ used during
        # simulation. theta is (1,) calibrated / (B,) per-path — broadcasts against spot's
        # batch axis (each path subtracts its own θ); the trailing unsqueeze is the feature dim.
        theta = self.theta.to(dtype=torch.float32).reshape(-1)
        log_dev = (spot.clamp_min(1.0e-9).log() - theta).unsqueeze(-1)
        kappa_t = torch.full_like(log_dev, float(self.param['Kappa']))
        sigma_t = torch.full_like(log_dev, float(self.param['Sigma']))
        return {'log_deviation': log_dev, 'kappa': kappa_t, 'sigma': sigma_t}


class LogOUSpotCalibration(object):
    """Calibrate LogOUSpotModel parameters from historical spot price data.

    Uses ``utils.calc_statistics`` in log space to estimate:

    - **Kappa**  — mean-reversion speed (``Mean Reversion Speed``)
    - **Sigma**  — OU volatility in log space (``Reversion Volatility``)
    - **Theta**  — long-run mean of log S, recovered by inverting the
      lognormal expectation:

          Long Run Mean = exp(theta + sigma^2 / (4*kappa))
          =>  Theta = log(Long Run Mean) - sigma^2 / (4*kappa)

    The ``Spot`` parameter is the current market value and is not calibrated here.
    """
    model_type = 'LogOUSpotModel'
    fields = []

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0,
                  kappa_max=10.0, sigma_max=2.0):
        stats, correlation, delta = utils.calc_statistics(
            data_frame, method='Log', num_business_days=num_business_days)

        kappa = stats['Mean Reversion Speed'].values[0]
        sigma = stats['Reversion Volatility'].values[0]

        # calc_statistics 'Log' returns Long Run Mean = exp(theta + sigma^2/(4*kappa))
        # Invert to recover theta = long-run mean of log(S)
        lrm   = np.clip(stats['Long Run Mean'].values[0], 1e-8, np.inf)
        theta = np.log(lrm) - sigma ** 2 / (4.0 * kappa)

        return utils.CalibrationInfo(
            {'Kappa': np.clip(kappa, 1e-4, kappa_max),
             'Theta': theta,
             'Sigma': np.clip(sigma + vol_shift, 0.0, sigma_max)},
            [[1.0]], delta)


class MarkovHMMSpotModel(StochasticProcess):
    """N-state hidden-Markov spot-price model. Conditional on regime z_t, the per-step
    innovation is Gaussian (or Student-t if `Nu` is set on the state) with annualised
    `(Mu, Sigma)`. Long-memory autocorrelation comes from regime persistence; fat tails
    from regime mixture plus optional t-emissions. One framework Gaussian per step;
    regime transitions sampled from the independent quasi-RNG uniform stream.

    JSON config:
        States: list of N dicts {Mu, Sigma, [Nu]} per regime (annualised).
        Transition_Matrix: NxN row-stochastic at Calibration_DT_Years.
        Initial_State_Probs: length-N vector summing to 1.
        Calibration_DT_Years: step size of P (default 1/252).
        Log_Price: bool (default False) — emit log returns instead of raw price diffs."""

    documentation = (
        'Asset Pricing',
        ['An N-state hidden-Markov spot-price model with additive Gaussian emissions on the '
         'daily diff $\\Delta S_t = S_t - S_{t-1}$. Latent regime $z_t$ follows a Markov chain '
         'with transition matrix $P$. Conditional on $z_t$:',
         '',
         '$$ \\Delta S_t \\sim \\mathcal{N}(\\mu_{z_t}\\delta,\\, \\sigma_{z_t}^2\\delta) $$',
         '',
         'No mean reversion at the spot level; long-memory autocorrelation arises from regime '
         'persistence and fat tails from regime occupancy. Innovations are optionally Student-t '
         '(rescaled to unit variance) when a state carries a $\\nu$.',
         '',
         'Parameters:',
         '- **States**: List of $\\{\\mu, \\sigma\\}$ per regime (annualised).',
         '- **Nu** (optional, per state): Student-t degrees of freedom.',
         '- **Transition_Matrix**: NxN row-stochastic at the calibration step.',
         '- **Initial_State_Probs**: Initial regime distribution.',
         '- **Calibration_DT_Years**: Step size of $P$ (default 1/252).',
         '- **Log_Price**: bool (default False) — emit log returns instead of raw price diffs.'])

    factor_types = ('CommodityPrice',)
    fields = [
        F('States', 'Container', default=[],
          description='List of per-regime {Mu, Sigma} dicts - annualised drift and vol of the '
                      'additive spot increment (must have at least 2 regimes)'),
        F('Transition_Matrix', 'Table', default=[],
          description='NxN row-stochastic transition matrix at the calibration time step'),
        F('Initial_State_Probs', 'Table', default=[],
          description='Initial regime distribution (length-N vector summing to 1)'),
        F('Calibration_DT_Years', 'Float', default=1.0 / 252.0,
          description='Step size (in years) of the calibrated transition matrix; the model '
                      're-discretises P per simulation step via the CTMC generator')
    ]

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        """
        Precompute the per-state emission moments, the CTMC transition ladder and the initial spot.

        AAD: `spot0` is kept on the autograd graph so payoff sensitivities w.r.t. the initial spot
        flow through, and is stored unreshaped so inner-MC mode can pass a `(B,)` vector of
        per-outer-path initial spots; outer mode is the framework's usual `(1,)` scalar, broadcast
        at generate-time.
        """
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        # Anchor at time_grid_years[0] so per-step dt is correct under both outer mode
        # (scen_time_grid[0] = 0) and inner-MC kept-base mode (scen_time_grid[0] > 0).
        tg_years = time_grid.time_grid_years
        dt_arr = np.diff(np.hstack(([tg_years[0]], tg_years)))
        states = self.param['States']
        self.n_states = len(states)
        T = len(dt_arr)

        def _t(arr):
            return shared.one.new_tensor(arr)

        # Annualised (μ, σ) per state; per-step values applied in generate via dt scaling.
        self.mu_per_state = _t(np.array([float(s.get('Mu', 0.0)) for s in states], dtype=np.float64))
        self.sigma_per_state = _t(np.array([float(s['Sigma']) for s in states], dtype=np.float64))
        self.dt_per_step = _t(dt_arr)
        # Optional Student-t degrees of freedom per state. If any state has Nu the model
        # emits t-distributed innovations (rescaled to unit marginal variance so σ retains
        # its standard interpretation). Absent or all-None → Gaussian as before.
        nu_arr = [s.get('Nu') for s in states]
        if any(n is not None for n in nu_arr):
            self.nu_per_state = _t(np.array([float(n) if n is not None else 1.0e6 for n in nu_arr],
                                            dtype=np.float64))
        else:
            self.nu_per_state = None

        # Log-price mode: emissions are log returns; final price is exp(log_spot0 + cumsum).
        # When False, emissions are raw price diffs and price is spot0 + cumsum(dS).
        self.log_price = bool(self.param.get('Log_Price', False))

        # CTMC re-discretisation: the calibrated per-business-day transition matrix maps to the
        # scenario grid through the generator, P(dt) = expm(logm(P_calib)/dt_calib * dt).
        P_calib = np.array(self.param['Transition_Matrix'], dtype=np.float64)
        dt_calib = float(self.param['Calibration_DT_Years'])
        Q = np.real(matrix_logm(P_calib)) / dt_calib
        P_per_step = np.zeros((T, self.n_states, self.n_states))
        for t, dt in enumerate(dt_arr):
            P_per_step[t] = np.real(matrix_expm(Q * dt)) if dt > 1.0e-12 else np.eye(self.n_states)
        self.P_cum = _t(np.cumsum(P_per_step, axis=2))
        # Raw P kept for the forward belief filter (the cumulative form is sampling-only).
        self.P_per_step = _t(P_per_step)
        self.pi0_cum = _t(np.cumsum(self.param['Initial_State_Probs']))
        self.pi0_probs = _t(np.array(self.param['Initial_State_Probs'], dtype=np.float64))

        # Stored as-is (no reshape) so inner MC can pass per-outer-path spots.
        self.spot0 = tensor

    @property
    def correlation_name(self):
        return 'MarkovHMMSpotProcess', [()]

    def generate(self, shared_mem):
        """
        Sample the regime path and simulate the HMM emission path (log returns or price diffs).

        Outer mode honours a per-path t=0 regime override published by the diff-ML burn-in under
        `(factor_key, 'regime0_outer')`, mirroring the inner-mode `regime0_inner` pattern; absent
        it, the t=0 regime is drawn from the calibrated π_0.

        Outer mode also runs the forward HMM belief filter: the differential-ML build uses
        `P(regime_t | prices_{0..t})` as the regime coordinate of `market_t`, because the
        privileged true regime is unavailable to a decision rule at runtime. Belief is detached
        from autograd — it is consumed as a state coordinate, not a quantity we differentiate
        through, and the price-path autograd graph is preserved separately for the deal pricer.
        It is published to BOTH `privileged_factors()` (B-axis dim=1, so the concat works) AND
        `t_Scenario_Buffer` with a B-LAST shape (T, n_states, B) so the buffer's dim=-1 concat
        works, enabling `reveal_state_at` to route belief into the V̂ deep-state market block.
        """
        # Z is (T, B) in outer mode, (T, B, B2) in inner mode. Everything runs at the
        # calculation's global precision (shared.one): the precalc tensors were built with
        # shared.one.new_tensor, so they already carry the right dtype/device — no casts.
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        device = Z.device

        pi0_cum = self.pi0_cum
        P_cum = self.P_cum
        mu = self.mu_per_state
        sigma = self.sigma_per_state
        dt = self.dt_per_step
        nu = self.nu_per_state

        if Z.ndim == 2:
            # Outer mode: (T, B). Canonical Sobol orientation: dimension = the per-path
            # coordinates (T+1 uniforms — one initial draw plus one per transition), samples =
            # paths, i.e. draw(B) -> (B, T+1) transposed to (T+1, B). The TRANSPOSED form
            # (dim=B, samples=T+1) is a defect at large B: a B-dimensional Sobol sequence with
            # only ~T points has badly-distributed cross-dimension (= cross-PATH) projections,
            # correlating regime transitions across outer paths — measured as the B=512 policy
            # collapse (worse even in-sample).
            T, B = Z.shape
            u_regime = shared_mem.quasi_rng(T + 1, B)[1].transpose(0, 1).contiguous()
            # Per-path t=0 regime override from the diff-ML burn-in; else the π_0 draw.
            regime0_override = shared_mem.t_Scenario_Buffer.get(
                (self.factor_key, 'regime0_outer'))
            if regime0_override is not None:
                state = regime0_override.to(device=device, dtype=torch.long)
            else:
                state = torch.searchsorted(pi0_cum, u_regime[0]).clamp_max_(self.n_states - 1)
            regimes = torch.empty((T, B), dtype=torch.long, device=device)
            regimes[0] = state
            for t in range(1, T):
                cdf_rows = P_cum[t - 1].index_select(0, state)
                state = (cdf_rows < u_regime[t].unsqueeze(1)).sum(dim=1).clamp_max_(self.n_states - 1)
                regimes[t] = state

            if nu is not None:
                nu_t = nu[regimes]                                                   # (T, B)
                # Floor the chi-square draw — an underflow to 0 makes sqrt(nu/W) blow
                # up to inf (or 0*inf=NaN), corrupting ds before the log-path clamp.
                W = torch.distributions.Gamma(nu_t / 2.0, 0.5).sample().clamp_min(1.0e-6)
                t_innov = Z * torch.sqrt(nu_t / W)
                scale_to_unit_var = torch.sqrt((nu_t - 2.0).clamp_min(1.0e-3) / nu_t)
                innov = t_innov * scale_to_unit_var
            else:
                innov = Z

            mu_t = mu[regimes] * dt.view(T, 1)                                       # (T, B)
            std_t = sigma[regimes] * dt.view(T, 1).sqrt()
            ds = mu_t + std_t * innov

            s0 = self.spot0.expand(B)                                                # (1,) -> (B,)
            if self.log_price:
                log_path = s0.log().unsqueeze(0) + ds.cumsum(dim=0)                  # (T, B)
                # Floor the log-path before exp(): a fat-tailed Student-t innovation can
                # drive it below the float underflow threshold, where exp() returns 0.0
                # and breaks the strictly-positive price-level invariant downstream.
                spot_path = log_path.clamp_min(-10.0).exp()
            else:
                spot_path = s0.unsqueeze(0) + ds.cumsum(dim=0)                       # (T, B)
            # Forward HMM belief filter — outer-mode only; detached, published B-last.
            with torch.no_grad():
                belief_path = self._forward_belief(spot_path.detach(), device)
            self.last_regime_belief = belief_path
            shared_mem.t_Scenario_Buffer[(self.factor_key, 'regime_belief')] = \
                belief_path.permute(0, 2, 1).contiguous()                            # (T, n_states, B)
        else:
            # Inner mode: (T, B, B2). One regime path per outer × inner.
            T, B, B2 = Z.shape
            # Sobol dim = T+1 (inner timesteps, ≪ 21201 cap); samples = B*B2 paths (unbounded).
            # Each path is one (T+1)-dim Sobol point; transpose so timesteps lead.
            u_flat = shared_mem.quasi_rng(T + 1, B * B2)[1].transpose(0, 1).contiguous()  # (T+1, B*B2)
            u_regime = u_flat.reshape(T + 1, B, B2)

            regime0_override = shared_mem.t_Scenario_Buffer.get(
                (self.factor_key, 'regime0_inner'), None)
            regimes = torch.empty((T, B, B2), dtype=torch.long, device=device)
            if regime0_override is not None:
                # Per-outer-path initial regime: shape (B,), expanded across the B2 inner fan-out.
                state = regime0_override.to(device=device, dtype=torch.long)\
                    .view(B, 1).expand(B, B2).contiguous()
            else:
                state = torch.searchsorted(pi0_cum, u_regime[0]).clamp_max_(self.n_states - 1)
            regimes[0] = state
            n_states = self.n_states
            for t in range(1, T):
                cdf_rows = P_cum[t - 1].index_select(0, state.flatten())\
                    .reshape(B, B2, n_states)
                state = (cdf_rows < u_regime[t].unsqueeze(-1)).sum(dim=-1)\
                    .clamp_max_(n_states - 1)
                regimes[t] = state

            if nu is not None:
                nu_t = nu[regimes]                                                   # (T, B, B2)
                # Floor the chi-square draw — an underflow to 0 makes sqrt(nu/W) blow
                # up to inf (or 0*inf=NaN), corrupting ds before the log-path clamp.
                W = torch.distributions.Gamma(nu_t / 2.0, 0.5).sample().clamp_min(1.0e-6)
                t_innov = Z * torch.sqrt(nu_t / W)
                scale_to_unit_var = torch.sqrt((nu_t - 2.0).clamp_min(1.0e-3) / nu_t)
                innov = t_innov * scale_to_unit_var
            else:
                innov = Z

            mu_t = mu[regimes] * dt.view(T, 1, 1)                                    # (T, B, B2)
            std_t = sigma[regimes] * dt.view(T, 1, 1).sqrt()
            ds = mu_t + std_t * innov

            s0 = self.spot0                                                          # (B,)
            if self.log_price:
                log_path = s0.view(B, 1).log() + ds.cumsum(dim=0)                    # (T, B, B2)
                spot_path = log_path.clamp_min(-10.0).exp()                          # floor: see outer branch
            else:
                spot_path = s0.view(B, 1) + ds.cumsum(dim=0)                         # (T, B, B2)

        # Stashed for privileged_factors() called immediately after generate() in the sim loop;
        # also published for cross-process consumers under the (factor_key, kind) convention.
        self.last_regime_path = regimes
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'regimes')] = regimes
        return spot_path

    @classmethod
    def privileged_layout(cls, param):
        n = len(param.get('States') or [])
        return {'regime_onehot': n, 'regime_belief': n}

    def privileged_factors(self, simulated):
        regimes = self.last_regime_path
        belief = getattr(self, 'last_regime_belief', None)
        out = {
            'regime_onehot': torch.nn.functional.one_hot(
                regimes, num_classes=self.n_states).to(dtype=torch.float32),
        }
        if belief is not None:
            # Shape (T, B, n_states); accumulator concatenates along batch dim (last but one).
            out['regime_belief'] = belief.to(dtype=torch.float32)
        return out

    def reveal_state_at(self, t, buffer):
        """Regime-switching spot: belief-first / price-last (the calc concatenates the segments in
        this order). The regime stays LATENT — reveal the participant-inferable posterior
        `P(regime_t | prices_{0..t})` (SUFFICIENT statistic; step 2 keeps it out of diff-PCA) when
        the buffer carries a filtered belief whose rank matches the current mode (outer belief
        `(T,n,B)` vs regimes `(T,B)`; inner belief `(T,n,B,B2)` vs regimes `(T,B,B2)` ⇒
        `belief.dim() == regimes.dim() + 1`), else the degenerate true-regime one-hot fallback.
        The observable spot level — the deployable LME quote the liability marks to — is the last
        (CONTINUOUS) coordinate."""
        key = self.factor_key
        price = buffer[key][t].unsqueeze(0)                                   # (1, ...batch)
        regimes = buffer[(key, 'regimes')]
        belief = buffer.get((key, 'regime_belief'))
        if belief is not None and belief.dim() == regimes.dim() + 1:
            block = belief[t]                                                 # (n_states, ...batch)
        else:
            block = nnf.one_hot(regimes[t].long(), num_classes=self.n_states)\
                .to(dtype=buffer[key].dtype).movedim(-1, 0)                   # (n_states, ...batch)
        return [(block, REVEAL_SUFFICIENT), (price, REVEAL_CONTINUOUS)]

    def inner_fork_seed(self, factor_key, outer_buf, t):
        """Per-outer-path t=0 regime seed: the regime at the fork step, read by generate()'s
        `regime0_inner` hook so the inner fan-out continues the forked path's regime instead
        of redrawing from π_0."""
        return {(factor_key, 'regime0_inner'): outer_buf[(factor_key, 'regimes')][t]}

    def outer_reseed(self):
        """t=0 regime seed for the next outer run's burn-in: the terminal regime of this run."""
        return {(self.factor_key, 'regime0_outer'): self.last_regime_path[-1].detach()}

    def reseed_from_path(self, simulated, shared_mem):
        """Observed-path replay: refilter the HMM belief along the supplied price path and
        publish it (B-last, as generate does), so `reveal_state_at` returns the participant
        posterior on the replayed path instead of the true-regime one-hot fallback."""
        belief = self._forward_belief(simulated.detach(), simulated.device)
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'regime_belief')] = \
            belief.permute(0, 2, 1).contiguous()

    def reseed_inner_state(self, factor_key, simulated, outer_buf, t, shared_mem, opts, with_grad):
        """Inner-fork belief coherence: publish a one-step filtered belief (seeded from the outer
        entry posterior, updated on the inner price move) so the bootstrap's z_{t+1} regime block
        is the participant posterior, not the privileged true-regime one-hot. Off (→ one-hot)
        when `Inner_Belief_Filter` is not 'Yes' or no outer belief was published. With grad, the
        belief seed is a leaf so the twin loss supervises the belief column."""
        belief0 = outer_buf.get((factor_key, 'regime_belief'))
        if not (opts.get('Inner_Belief_Filter', 'Yes') == 'Yes' and belief0 is not None):
            return {}
        seed = belief0[t]
        leaves = {}
        if with_grad:
            seed = seed.detach().clone().requires_grad_(True)
            leaves[(factor_key, 'regime_belief')] = seed
        inner_belief = self.inner_forward_belief(simulated, seed)
        shared_mem.t_Scenario_Buffer[(factor_key, 'regime_belief')] = inner_belief
        if not getattr(self, '_logged_inner_belief', False):
            self._logged_inner_belief = True
            sums = inner_belief.sum(dim=1)
            logging.info(
                'inner belief filter ACTIVE for %s: shape=%s n_states=%d normalized=%s (replaces '
                'the privileged true-regime one-hot in the bootstrap z_{t+1})',
                utils.check_tuple_name(factor_key), tuple(inner_belief.shape), inner_belief.shape[1],
                bool(torch.allclose(sums, torch.ones_like(sums), atol=1.0e-4)))
        return leaves

    def diff_state_leaves(self):
        return ('regime_belief',)

    def calibrated_annual_vol(self):
        """Stationary regime-weighted annualized vol: σ = √(Σ_i π_i σ_i²), with π the stationary
        distribution of the calibration-DT transition matrix P (left eigenvector for eigenvalue 1).
        Log_Price=True → the state σ_i are already fractional log-vols and σ is returned as-is;
        raw-diff (Log_Price=False) → σ_i are absolute ($/√yr), divided by the initial spot to a
        fraction so the utility-scale formula c = volume·spot·σ·√τ recovers volume·σ_abs·√τ."""
        sig = np.array([float(s['Sigma']) for s in self.param['States']], dtype=np.float64)
        P = np.array(self.param['Transition_Matrix'], dtype=np.float64)
        evals, evecs = np.linalg.eig(P.T)
        pi = np.real(evecs[:, int(np.argmin(np.abs(evals - 1.0)))])
        pi = pi / pi.sum()
        sigma = float(np.sqrt(float((pi * sig * sig).sum())))
        if not bool(self.param.get('Log_Price', False)):
            spot = float(self.spot0.reshape(-1).median().item())
            sigma = sigma / spot if spot > 0.0 else 0.0
        return sigma

    def _forward_belief(self, spot_path, device):
        """Forward HMM belief filter — outer-mode only. Returns belief (T, B, n_states)
        where `belief[t, b, r] = P(regime_t = r | observed diffs through time t, path b)`,
        computed in log-space (logsumexp predict + logsumexp normalize) for numerical
        robustness under fat-tailed Student-t emissions. Per-step emission parameters and
        transition matrix match the simulator's exactly (same `mu_per_state`,
        `sigma_per_state`, `nu_per_state`, `dt_per_step`, `P_per_step`) — so on held-out
        sim data the filter is calibrated against the model that generated it.
        """
        T, B = spot_path.shape
        n_states = self.n_states

        pi0_probs = self.pi0_probs
        P_step = self.P_per_step                                           # (T, n, n)
        dt = self.dt_per_step                                              # (T,)

        # Observed per-step diffs: log returns if Log_Price, raw price diffs otherwise.
        # diffs[t-1] is the observation arriving at time t.
        if self.log_price:
            log_path = spot_path.clamp_min(1.0e-30).log()
            diffs = log_path[1:] - log_path[:-1]                           # (T-1, B)
        else:
            diffs = spot_path[1:] - spot_path[:-1]                         # (T-1, B)

        log_belief = torch.empty((T, B, n_states), dtype=pi0_probs.dtype, device=device)
        log_belief[0] = pi0_probs.clamp_min(1.0e-30).log().expand(B, n_states)

        for t in range(1, T):
            # Predict: log_b_pred[r'] = logsumexp_r (log_b[t-1, r] + log_P[t-1, r, r'])
            log_P = P_step[t - 1].clamp_min(1.0e-30).log()                 # (n, n)
            log_b_pred = torch.logsumexp(
                log_belief[t - 1].unsqueeze(-1) + log_P.unsqueeze(0), dim=-2)  # (B, n)
            if float(dt[t]) < 1.0e-12:
                # Degenerate step (e.g. forked grid where t=0's neighbour is zero); skip
                # the update — predict-only is a valid filter step under no observation.
                log_belief[t] = log_b_pred - torch.logsumexp(log_b_pred, dim=-1, keepdim=True)
                continue
            # Update: multiply by the per-state emission density of the observed diff.
            log_b_unnorm = log_b_pred + self._emission_log_likelihood(diffs[t - 1], t)  # (B, n)
            log_belief[t] = log_b_unnorm - torch.logsumexp(log_b_unnorm, dim=-1, keepdim=True)

        return log_belief.exp()

    def _emission_log_likelihood(self, diff, t_idx):
        """Per-state log emission density of an observed price diff arriving at step
        `t_idx`. `diff` is any shape `(...)`; returns `(..., n_states)`. Gaussian, or the
        unit-variance-rescaled scaled-Student-t when `nu_per_state` is set — matches the
        simulator's emission exactly. Shared by the outer filter (`_forward_belief`) and
        the inner-MC one-step filter (`inner_forward_belief`)."""
        dt_t = self.dt_per_step[t_idx]
        mean_r = self.mu_per_state * dt_t                                  # (n,)
        std_r = self.sigma_per_state * dt_t.sqrt()                         # (n,)
        z = (diff.unsqueeze(-1) - mean_r) / std_r                          # (..., n)
        nu = self.nu_per_state
        if nu is not None:
            # log f = lgamma((ν+1)/2) − lgamma(ν/2) − ½·log((ν−2)π) − log σ_r
            #         − ½·(ν+1)·log(1 + z²/(ν−2))
            nu_eff = (nu - 2.0).clamp_min(1.0e-3)
            return (torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0)
                    - 0.5 * (nu_eff.log() + float(np.log(np.pi))) - std_r.log()
                    ) - 0.5 * (nu + 1.0) * torch.log1p(z.pow(2) / nu_eff)
        # Gaussian: −½·log(2π σ²) − ½·z²
        return (-0.5 * (float(np.log(2.0 * np.pi)) + 2.0 * std_r.log())) - 0.5 * z.pow(2)

    def inner_forward_belief(self, inner_spot, belief0):
        """One-step (few-step) HMM filter over an INNER-MC price path, seeded with the
        outer entry posterior `belief0` — so the diff-ML bootstrap's `z_{t+1}` carries the
        participant's genuine belief, NOT a privileged true-regime one-hot. Differentiable
        in `inner_spot`: the belief column's pathwise slope falls out of the same AAD pass
        that yields the wealth/price differentials.

        `inner_spot`: `(T, B, B2)` inner price path (T=2 for the one-step bootstrap —
        index 0 = outer t, index 1 = outer t+1). `belief0`: `(n_states, B)` outer posterior
        at the fork step (B-last, as published to the buffer), broadcast across the B2
        fan-out. Returns belief `(T, n_states, B, B2)` — the buffer's B-last convention, so
        `reveal_state_at` returns `belief[τ] = (n_states, B, B2)`.

        Predict uses the transition over the FULL step `P_per_step[τ] = expm(Q·dt[τ])` (the
        real interval), not the dt=0 anchor `P_per_step[τ-1]=I`: the participant filters
        under the true regime dynamics regardless of the inner sampler's anchor."""
        T, B, B2 = inner_spot.shape
        n = self.n_states
        obs = inner_spot.clamp_min(1.0e-30).log() if self.log_price else inner_spot
        diffs = obs[1:] - obs[:-1]                                         # (T-1, B, B2)
        log_b = [belief0.movedim(0, -1).clamp_min(1.0e-30).log()           # (B, n)
                 .unsqueeze(1).expand(B, B2, n)]                           # seed → (B, B2, n)
        for tau in range(1, T):
            log_P = self.P_per_step[tau].clamp_min(1.0e-30).log()          # (n, n) over dt[τ]
            log_b_pred = torch.logsumexp(
                log_b[tau - 1].unsqueeze(-1) + log_P, dim=-2)              # (B, B2, n)
            log_unnorm = log_b_pred + self._emission_log_likelihood(diffs[tau - 1], tau)
            log_b.append(log_unnorm - torch.logsumexp(log_unnorm, dim=-1, keepdim=True))
        return torch.stack(log_b, dim=0).exp().movedim(-1, 1)              # (T, n, B, B2)


class MarkovHMMSpotCalibration(object):
    """Calibration of MarkovHMMSpotModel via in-house Baum-Welch on price diffs (or log
    returns when `Log_Price=True`). Per-state emission is Gaussian; M-step uses weighted
    mean and weighted variance. Optional Student-t refit picks a shared ν via
    method-of-moments on the unconditional mixture kurtosis — per-state μ, σ are kept
    and ν enters the simulator's t-rescaling so marginal variance per regime stays at σ².
    States are reordered ascending by σ post-fit. `delta` is the regime-standardised
    innovation series — approximately iid under the calibrated model."""
    model_type = 'MarkovHMMSpotModel'
    fields = [
        F('N_States', 'Integer', default=3, description='Number of regimes'),
        F('N_Iter', 'Integer', default=200, description='Baum-Welch iteration cap'),
        F('Seed', 'Integer', default=42, description='RNG seed for the EM initialisation'),
        F('Tol', 'Float', default=1e-6,
          description='Relative log-likelihood change that stops the EM loop'),
        F('Use_Student_T', 'Boolean', default=True,
          description='Refit a shared Student-t nu; false leaves the emissions Gaussian'),
        F('Nu_Min', 'Float', default=3.0, description='Floor on the fitted degrees of freedom'),
        F('Nu_Max', 'Float', default=50.0,
          description='Ceiling on the degrees of freedom, above which the t refit is dropped'),
        F('Log_Price', 'Boolean', default=True,
          description='Fit on log returns rather than raw price differences')
    ]

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    @staticmethod
    def _emission_logprob(diffs, means, sigmas):
        """Per-state log Normal(diffs | μ_s, σ_s²); returns (T, n_states)."""
        var = np.maximum(sigmas ** 2, 1.0e-12)
        return (-0.5 * np.log(2.0 * np.pi * var)
                - 0.5 * (diffs[:, None] - means) ** 2 / var)

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        """
        Fit the regime-switching HMM by EM, then derive the shared Student-t tail parameter.

        ν is a method-of-moments estimate off the unconditional kurtosis of the regime mixture,
        K = (3 + 6/(ν-4)) · Σπ_s σ_s⁴ / Var² - 3, inverted for ν as
            ν = 4 + 6 / [(K_emp + 3)·Var² / Σπ_s σ_s⁴ - 3]
        It uses the *model's* stationary variance rather than the sample variance, so the
        simulator round-trips on kurtosis — EM convergence may underfit the empirical Var.
        """
        from scipy import stats as scipy_stats

        n_states = int(self.param.get('N_States', 3))
        n_iter = int(self.param.get('N_Iter', 200))
        seed = int(self.param.get('Seed', 42))
        tol = float(self.param.get('Tol', 1.0e-6))
        use_t = bool(self.param.get('Use_Student_T', True))
        nu_min = float(self.param.get('Nu_Min', 3.0))
        nu_max = float(self.param.get('Nu_Max', 50.0))
        # Log_Price: fit on log returns. Scale-invariant, simulator exp()s to keep prices positive.
        log_price = bool(self.param.get('Log_Price', True))
        dt_calib = 1.0 / float(num_business_days)

        prices = data_frame.iloc[:, 0].astype(np.float64).dropna()
        if log_price:
            diffs = np.log(prices).diff().dropna()
        else:
            diffs = prices.diff().dropna()
        x = diffs.values

        # Init: spread σ across regimes for distinguishable EM start.
        rng = np.random.default_rng(seed)
        base_sigma = x.std()
        means = np.full(n_states, x.mean()) + rng.normal(0, base_sigma * 0.05, n_states)
        sigmas = base_sigma * np.linspace(0.5, 2.0, n_states)
        pi = np.full(n_states, 1.0 / n_states)
        P = np.full((n_states, n_states), 0.05) + 0.85 * np.eye(n_states)
        P /= P.sum(axis=1, keepdims=True)

        prev_lik = -np.inf
        for it in range(n_iter):
            log_emit = self._emission_logprob(x, means, sigmas)
            gamma, xi, log_lik = hmm_forward_backward(
                np.log(pi + 1e-12), np.log(P + 1e-12), log_emit)
            if it > 0 and abs(log_lik - prev_lik) < tol * abs(prev_lik):
                break
            prev_lik = log_lik
            pi = gamma[0]
            denom = gamma[:-1].sum(axis=0)
            P = xi.sum(axis=0) / np.maximum(denom[:, None], 1e-12)
            P /= P.sum(axis=1, keepdims=True)
            w_sum = gamma.sum(axis=0)
            means = (gamma * x[:, None]).sum(axis=0) / np.maximum(w_sum, 1e-12)
            sigmas = np.sqrt(np.maximum(
                (gamma * (x[:, None] - means) ** 2).sum(axis=0) / np.maximum(w_sum, 1e-12),
                1e-12))

        # Reorder states ascending by σ; remap P, π, posterior assignments accordingly.
        order = np.argsort(sigmas)
        means = means[order]
        sigmas = sigmas[order]
        P = P[np.ix_(order, order)]
        regimes = np.argmax(gamma, axis=1)
        remap = {old: new for new, old in enumerate(order)}
        regimes = np.array([remap[s] for s in regimes])
        occ = np.bincount(regimes, minlength=n_states) / len(regimes)
        nus = [None] * n_states

        # Method-of-moments shared ν off the mixture's stationary variance.
        if use_t:
            emp_kurt = float(scipy_stats.kurtosis(x, fisher=True))
            mu_total = float((occ * means).sum())
            mix_var = float((occ * sigmas**2).sum() + (occ * (means - mu_total)**2).sum())
            sum_pi_sigma4 = float(np.sum(occ * sigmas**4))
            denom = (emp_kurt + 3.0) * mix_var * mix_var / sum_pi_sigma4 - 3.0
            if denom > 1e-3:
                nu_global = 4.0 + 6.0 / denom
                nu_global = float(np.clip(nu_global, nu_min, nu_max))
                if nu_global < nu_max - 1e-6:
                    nus = [nu_global] * n_states

        # Annualised storage convention so model.precalculate's per-step
        # `μ·δ, σ²·δ` formula gives the calibration-step daily moments at δ=dt_calib.
        mu_year = means / dt_calib
        sigma_year = sigmas / np.sqrt(dt_calib)

        param = {
            'Log_Price': log_price,
            'States': [
                {'Mu': float(m), 'Sigma': float(s), **({'Nu': float(n)} if n is not None else {})}
                for m, s, n in zip(mu_year, sigma_year, nus)
            ],
            'Transition_Matrix': P.tolist(),
            'Initial_State_Probs': occ.tolist(),
            'Calibration_DT_Years': dt_calib,
        }

        # delta = regime-standardised innovation: (diff - μ_state) / σ_state under the
        # posterior regime path. Approximately iid N(0,1) so the framework's correlation
        # consolidation isn't contaminated by regime-induced heteroskedasticity.
        innov = (diffs.values - means[regimes]) / np.where(sigmas[regimes] > 0, sigmas[regimes], 1.0)
        delta = pd.DataFrame({data_frame.columns[0]: innov}, index=diffs.index)

        return utils.CalibrationInfo(param, [[1.0]], delta)


class GARCHSpotModel(StochasticProcess):
    """Zero-mean GARCH(1,1)-t spot-price model — martingale primary by construction:
    E[Δlog S | filtration] = μ·dt with `Mu` defaulting to 0. The conditional variance
    `h_t` is a deterministic recursion on realized returns (no belief filter), exactly
    observable, and revealed to the value function as `log h_t`. Drop-in for the HMM: same
    outer/inner generate contract, same buffer conventions, same privileged-state plumbing.

        ε_k ~ standardized Student-t(ν) (unit variance);  r_k = √h_k · ε_k
        Δlog S over step k = μ·dt_c + r_k;  h_{k+1} = ω + α·r_k² + β·h_k;  h_0 = H0

    No-lookahead: `h_t` is the variance of the step t→t+1, a function of returns strictly
    before t, so `log h_t` is known at decision time t.

    JSON config:
        Omega, H0: per-calibration-step variance of FRACTION log returns (ω>0, H0>0).
        Alpha, Beta: GARCH weights (α≥0, β≥0, α+β≤0.999).
        Nu: Student-t degrees of freedom (>2.05).
        Mu: annualised drift (default 0 — 16y of daily data cannot identify it).
        Log_Price: bool (default True; the model is defined in log-return units).
        Calibration_DT_Years: step size of the recursion (default 1/252).
        Convexity_Correction: Yes/No (default No). No = today's log-space-zero-mean (E[Δlog S]=
            Mu·dt), which leaves a spurious +½·Var Jensen drift in the PRICE (~½·annual-var, e.g.
            ~3%/yr at LR vol 0.2475). Yes subtracts ½·Var(r_t) from the per-step log-drift so the
            PRICE is the Mu-martingale: E[S_{t+1}/S_t]=exp(Mu·dt) (Gaussian-exact; the Student-t
            tail leaves a small positive residual). h recursion / revealed log_h are UNCHANGED."""

    documentation = (
        'Asset Pricing',
        ['A zero-mean GARCH(1,1) spot-price model with standardised Student-t innovations on '
         'the log return $r_t = \\Delta\\log S_t - \\mu\\delta$. The conditional variance follows',
         '',
         '$$ r_t = \\sqrt{h_t}\\,\\varepsilon_t,\\quad \\varepsilon_t\\sim t_\\nu\\ (\\text{unit var}),'
         '\\quad h_{t+1} = \\omega + \\alpha r_t^2 + \\beta h_t $$',
         '',
         'The variance $h_t$ is an exactly observable deterministic recursion on realised '
         'returns — no latent state, no belief filter — revealed to the value function as '
         '$\\log h_t$. Long-memory volatility comes from persistence $\\alpha+\\beta\\to 1$, fat '
         'tails from the Student-t emission.',
         '',
         'Parameters:',
         '- **Omega, H0**: per-step variance of fraction log returns.',
         '- **Alpha, Beta**: GARCH weights ($\\alpha+\\beta\\le 0.999$).',
         '- **Nu**: Student-t degrees of freedom.',
         '- **Mu**: annualised drift (default 0).',
         '- **Calibration_DT_Years**: step size of the recursion (default 1/252).',
         '- **Convexity_Correction**: Yes/No (default No). Yes makes the PRICE a Mu-martingale by '
         'subtracting $\\tfrac{1}{2}\\text{Var}(r_t)$ from the per-step log-drift; No leaves a '
         '$+\\tfrac{1}{2}\\text{var}$ Jensen drift.',
         '- **Log_Price**: bool (default True; the model is defined on log returns).'])

    factor_types = ('CommodityPrice',)
    fields = [
        F('Omega', 'Float', default=0.0, description='Variance intercept omega > 0'),
        F('Alpha', 'Float', default=0.0, description='Weight on the last squared return'),
        F('Beta', 'Float', default=0.0, description='Weight on the last conditional variance'),
        F('Nu', 'Float', default=0.0, description='Student-t degrees of freedom (> 2.05)'),
        F('Mu', 'Float', default=0.0, description='Annualised log drift'),
        F('H0', 'Float', default=0.0, description='Conditional variance at t=0'),
        F('Calibration_DT_Years', 'Float', default=1.0 / 252.0,
          description='Step size (in years) of the variance recursion'),
        F('Convexity_Correction', 'Text', default='No', values=['Yes', 'No'],
          description='Yes subtracts half the per-step variance from the log drift, making the '
                      'PRICE the Mu-martingale'),
        F('Carry_Drift', 'Text', default='No', values=['Yes', 'No'],
          description="Yes drifts the simulated log-spot each step at the FRONT of the "
                      "commodity's own carry curve (the factor's declared Forward_Rate, whose "
                      "process must publish (key, 'z0')) - cost-of-carry real-world dynamics, "
                      "under which every futures leg is near-driftless in the training world "
                      "instead of rolling down the curve with certainty")
    ]

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)
        assert (param['Omega'] > 0.0 and param['Alpha'] >= 0.0 and param['Beta'] >= 0.0
                and param['Alpha'] + param['Beta'] <= 0.999 and param['Nu'] > 2.05
                and param['H0'] > 0.0), f'GARCHSpotModel invalid params: {param}'
        self.carry_drift = str(param.get('Carry_Drift', 'No')) == 'Yes'
        # Convexity_Correction=Yes makes the PRICE the martingale (E[S_{t+1}/S_t]=exp(Mu·dt))
        # by subtracting ½·Var(r_t) from the per-step log-drift; No (default) is today's
        # log-space-zero-mean (E[Δlog S]=Mu·dt, so the price carries a +½·var Jensen drift).
        cc = param.get('Convexity_Correction', 'No')
        assert cc in ('Yes', 'No'), f'GARCHSpotModel Convexity_Correction must be Yes/No: {cc}'
        self.convexity = cc == 'Yes'

    @staticmethod
    def num_factors():
        return 1

    @property
    def correlation_name(self):
        return 'GARCHSpotProcess', [()]

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        """
        Precompute the fractional-trading-clock step schedule and the GARCH recursion parameters.

        The recursion is calibrated per business day (dt_c) while the sim grid runs in CALENDAR
        time (Time_Grid "0d 1d(1d)" ⇒ dt=1/365.25, NOT business-day adjusted), so f_t = dt_t/dt_c
        is the trading-time length of a grid step (≈0.69 on the production grid). `generate` scales
        the per-step variance by f_t, which makes the annualized vol and the mean-reversion RATE
        grid-invariant. A step spanning more than one calibration step walks its own sub-steps
        (utils.garch_correlated_substeps); one sub-step is the exact fractional step.

        AAD: `spot0` is kept on the autograd graph. In log mode h depends only on the generated
        innovations, never on spot0, so price-AAD w.r.t. spot0 is unaffected by the vol recursion.
        Outer mode passes a (1,) scalar; inner-MC mode a (B,) per-outer-path vector.
        """
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        # Anchor at time_grid_years[0] so per-step dt is correct under both outer mode
        # (scen_time_grid[0] = 0) and inner-MC kept-base mode (scen_time_grid[0] > 0).
        tg_years = time_grid.time_grid_years
        dt_arr = np.diff(np.hstack(([tg_years[0]], tg_years)))
        dt_c = float(self.param['Calibration_DT_Years'])

        def _t(arr):
            return shared.one.new_tensor(arr)

        self.omega = _t(float(self.param['Omega']))
        self.alpha = _t(float(self.param['Alpha']))
        self.beta = _t(float(self.param['Beta']))
        self.nu = _t(float(self.param['Nu']))
        self.h0_default = _t(float(self.param['H0']))
        self.drift = _t(float(self.param.get('Mu', 0.0)) * dt_arr)               # (T,) μ·dt per step
        self.dt_t = _t(dt_arr)                                                   # (T,) dt per step, for Carry_Drift
        # f_t = dt_t/dt_c, the trading-time length of a calendar grid step.
        self.f = _t(dt_arr / dt_c)                                               # (T,) trading-time step length
        self.sub_dt = utils.substep_schedule(dt_arr / dt_c)
        self.n_sub = np.array([len(s) for s in self.sub_dt])
        if np.any(self.n_sub >= 2):
            # Diagnostic (INFO, not WARN — precalculate reruns on every inner fork).
            logging.info('GARCHSpotModel coarse grid: n_sub up to %d — exact daily sub-stepping, '
                         'the correlated draw rides the √E[h]-weighted combination.',
                         int(self.n_sub.max()))
        self._log_lr_var = float(np.log(self.param['Omega'] / (1.0 - self.param['Alpha'] - self.param['Beta'])))

        # AAD: spot0 stays on the graph, unreshaped so inner MC can pass (B,).
        self.spot0 = tensor

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        """Resolve the carry-curve factor Carry_Drift reads - the factor's own declared
        Forward_Rate. Fail-loud here rather than silently pricing without the drift: a missing
        or unsimulated carry with the switch on is exactly the phantom (model-data drift
        mismatch) the switch exists to remove."""
        if not self.carry_drift:
            return
        name = self.factor.param.get('Forward_Rate')
        if not name:
            raise Exception(f'GARCHSpotModel {utils.check_tuple_name(factor)}: Carry_Drift=Yes '
                            f'but the factor declares no Forward_Rate to read the carry from')
        self.carry_key = utils.Factor('ForwardRate', utils.check_rate_name(name))
        if self.carry_key not in all_factors:
            raise Exception(f'GARCHSpotModel {utils.check_tuple_name(factor)}: Carry_Drift=Yes '
                            f'but {utils.check_tuple_name(self.carry_key)} is not in the factor '
                            f'universe')

    def _carry_drift(self, shared_mem, ndim):
        """The (T, ...) per-step log-drift with the carry term folded in, or the plain (T,) Mu
        drift when the switch is off. The carry process publishes `(key, 'z0')` in the fork-index
        convention (row t = what step t->t+1 consumes), (T, 1, ...batch); it simulates BEFORE
        this spot (Forward_Rate is a dependency of CommodityPrice, so topological order provides
        it) in the outer loop, the inner fork and the observed-path replay alike."""
        if not self.carry_drift:
            return self.drift
        z0 = shared_mem.t_Scenario_Buffer.get((self.carry_key, 'z0'))
        if z0 is None:
            raise Exception(
                f'GARCHSpotModel {utils.check_tuple_name(self.factor_key)}: Carry_Drift=Yes but '
                f'{utils.check_tuple_name(self.carry_key)} published no (key, \'z0\') series - '
                f'its process must simulate (and publish the front carry) before this spot')
        zt = z0.squeeze(1)
        # Per-factor scenario grids are PREFIXES of one master grid, each cut one row past its
        # own last dependent date - a composed spot can outlive its carry by a trailing row.
        # The missing tail is the front carry held flat at its last simulated value.
        pad = self.dt_t.shape[0] - zt.shape[0]
        if pad > 0:
            zt = torch.cat([zt, zt[-1:].expand(pad, *zt.shape[1:])])
        shape = (-1,) + (1,) * (ndim - 1)
        return self.drift.view(shape) + zt * self.dt_t.view(shape)

    def _simulate_returns(self, eps, z, h, drift=None):
        """Shared GARCH recursion (outer/inner) on the FRACTIONAL TRADING CLOCK. `eps` (T, ...)
        unit-variance standardised-t innovations, `z` (T, ...) the raw framework Gaussians they
        were scaled from (a coarse interval's sub-steps re-derive their own t-innovations from
        z[t]), `h` (...) the entry variance h_0 (per business day). Returns (ds, log_h), each
        (T, ...): ds[t] is Δlog S landing at grid point t (ds[0]=0 — the dt=0 anchor), log_h[t]
        is the revealed variance of the move t→t+1 (no-lookahead: a function of draws ≤ t only).

        Per step with trading-time length f_t = dt_t/dt_c:
            r_t = √(h_t·f_t)·ε_t,   h_{t+1} = h_t + f_t·(ω − (1−β)·h_t) + α·r_t².
        This is the standard recursion at f=1, has E-fixed-point ω/(1−α−β) for any f, and a
        per-step mean-reversion factor (1 − f(1−α−β)) so the decay RATE is grid-invariant in
        real time. A step spanning more than one calibration step walks the sub-steps that span
        it instead (utils.garch_correlated_substeps — z[t] rides the √E[h·dt]-weighted
        combination of the sub-step normals); log_h[t] is then the variance entering the NEXT
        interval, and that interval's own variance Σ h_j·dt_j is realized, not F_t-measurable.

        With Convexity_Correction, the deterministic log-drift also carries −½·Var(r_t) so the
        PRICE (not the log-price) is the Mu-martingale: E[exp(r_t − ½Var(r_t))] = 1 for Gaussian
        r_t. The correction touches ONLY ds (the log-price shift); the innovation r_t and the h
        recursion — hence revealed log_h — are untouched. (Student-t r has E[exp(r)]=∞, so −½Var
        slightly under-corrects, leaving a small positive residual price drift on the fat tail.)

        `drift` overrides the flat Mu·dt schedule with a per-(step, path) tensor — the
        Carry_Drift composite from `_carry_drift`; both index as drift[t]."""
        drift = self.drift if drift is None else drift
        log_h = torch.empty_like(eps)
        ds = torch.zeros_like(eps)
        log_h[0] = h.log()
        for t in range(1, eps.shape[0]):
            if self.n_sub[t] <= 1:
                h, var_step, r = self._advance_variance(h, t, lambda v: v.sqrt() * eps[t])
            else:
                h, var_step, r = utils.garch_correlated_substeps(
                    h, z[t], self.sub_dt[t], self.omega, self.alpha, self.beta, self.nu)
            ds[t] = drift[t] + (r - 0.5 * var_step if self.convexity else r)
            log_h[t] = h.log()
        return ds, log_h

    def _advance_variance(self, h, t, innovation):
        """One fractional-clock GARCH variance step (n_sub ≤ 1; a coarse interval routes to
        utils.garch_correlated_substeps in `_simulate_returns` instead), shared by the forward sim
        and the observed-path replay (`reseed_from_path`) so the forward≡replay invariant is
        STRUCTURAL, not maintained by copying. Computes Var(r_t) for the step, obtains the innovation
        via `innovation(var_step)` — the ONLY thing that differs between the two paths (√Var·ε in the
        forward sim; the realized convexity-undone log-return in replay) — and returns
        (h_{t+1}, var_step, r) off the exact fractional recursion h + f·(ω − (1−β)·h) + α·r²."""
        ft = self.f[t]
        var_step = h * ft                                                        # Var(r_t)
        r = innovation(var_step)
        return h + ft * (self.omega - (1.0 - self.beta) * h) + self.alpha * r * r, var_step, r

    def generate(self, shared_mem):
        """
        Simulate the GARCH spot path; Z is (T, B) outer, (T, B, B2) inner.

        One framework Gaussian per step; the standardised-t rescale draws its own Gamma (there is
        no regime sampling, hence no quasi_rng). ε is unit-variance and independent of h, so it
        precomputes fully vectorised — but only the fine steps read it (a coarse interval t-scales
        its own sub-step normals), so an all-coarse PFE grid skips the draw rather than allocating
        a dead (T,B) pair.
        """
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        nu = self.nu
        if (self.n_sub[1:] <= 1).any():                                          # t=0 is the anchor
            W = torch.distributions.Gamma(nu / 2.0, 0.5).sample(Z.shape).clamp_min(1.0e-6)
            eps = Z * torch.sqrt(nu / W) * torch.sqrt((nu - 2.0).clamp_min(1.0e-3) / nu)
        else:
            eps = Z

        drift = self._carry_drift(shared_mem, Z.ndim)
        if Z.ndim == 2:
            T, B = Z.shape
            # h0: the diff-ML t=0 randomization hook (mirrors regime0_outer); else H0 expanded.
            h0 = shared_mem.t_Scenario_Buffer.get((self.factor_key, 'h0_outer'))
            h = h0 if h0 is not None else torch.zeros_like(Z[0]) + self.h0_default
            ds, log_h = self._simulate_returns(eps, Z, h, drift)
            s0 = self.spot0.expand(B)                                            # (1,) -> (B,)
            log_path = s0.log().unsqueeze(0) + ds.cumsum(dim=0)                  # (T, B)
        else:
            T, B, B2 = Z.shape
            # h0: per-outer-path fork seed (mirrors regime0_inner), expanded across B2; else H0.
            h0 = shared_mem.t_Scenario_Buffer.get((self.factor_key, 'h0_inner'))
            h = h0.view(B, 1).expand(B, B2) if h0 is not None else torch.zeros_like(Z[0]) + self.h0_default
            ds, log_h = self._simulate_returns(eps, Z, h, drift)
            s0 = self.spot0                                                      # (B,)
            log_path = s0.view(B, 1).log() + ds.cumsum(dim=0)                    # (T, B, B2)
        # Floor the log-path before exp(): a fat-tailed Student-t innovation can drive it below
        # the float underflow threshold, where exp() returns 0.0 and breaks the price invariant.
        spot_path = log_path.clamp_min(-10.0).exp()

        # Revealed state, detached (consumed as a state coordinate, not differentiated through —
        # same rationale as the HMM belief detach). B-LAST shape so the buffer's dim=-1 concat
        # works: (T, 1, B) outer / (T, 1, B, B2) inner. Stash the (T, B) form for privileged_factors.
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'garch_log_h')] = log_h.detach().unsqueeze(1)
        self.last_log_h = log_h.detach()
        return spot_path

    @classmethod
    def privileged_layout(cls, param):
        return {'log_h': 1}

    def privileged_factors(self, simulated):
        # (T, B, 1) — matches the HMM's (T, B, n) accumulator convention. Called outer-mode only.
        return {'log_h': self.last_log_h.to(torch.float32).unsqueeze(-1)}

    def reveal_state_at(self, t, buffer):
        """GARCH spot: log_h-first / price-last (the calc concatenates the segments in this
        order — mirrors the HMM's belief-first/price-last packing). `log h_t` is the exactly
        observable SUFFICIENT statistic; the observable spot level is the last CONTINUOUS
        coordinate. Rank-check the buffered log-h against the current mode (outer (T,1,B) vs
        price (T,B); inner (T,1,B,B2) vs (T,B,B2) ⇒ log_h.dim() == price.dim()+1), else the
        defensive long-run-variance fallback when the buffer key is absent."""
        key = self.factor_key
        price = buffer[key][t].unsqueeze(0)                                      # (1, ...batch)
        log_h = buffer.get((key, 'garch_log_h'))
        if log_h is not None and log_h.dim() == buffer[key].dim() + 1:
            block = log_h[t]                                                     # (1, ...batch)
        else:
            block = torch.full_like(price, self._log_lr_var)
        return [(block, REVEAL_SUFFICIENT), (price, REVEAL_CONTINUOUS)]

    def inner_fork_seed(self, factor_key, outer_buf, t):
        """Per-outer-path t=0 conditional-variance seed: h0_inner = exp(outer log h_t), so the
        inner fan-out reprices from the forked path's vol state, not the calibrated H0. Without
        it the one-step bootstrap labels are wrong (repriced from the base-date variance)."""
        return {(factor_key, 'h0_inner'): outer_buf[(factor_key, 'garch_log_h')][t].reshape(-1).exp()}

    def outer_reseed(self):
        """t=0 conditional-variance seed for the next outer run's burn-in: terminal h of this run."""
        return {(self.factor_key, 'h0_outer'): self.last_log_h[-1].exp()}

    def reseed_from_path(self, simulated, shared_mem):
        """Observed-path replay: rerun the GARCH variance recursion (fractional clock) on the
        REALIZED returns of the supplied price path, publishing `garch_log_h` (so reveal returns
        the right log h along the replayed path) and `h0_outer` (the terminal h, for a continuing
        replay).
        Convexity coupling: the forward sim writes ds = drift − ½Var(r) + r, so the innovation is
        recovered as r = realized_logret − drift + ½Var(r) with the SAME correction, keeping the
        h recursion identical between forward sim and replay (Var(r)=h·f on the n_sub≤1 clock)."""
        # Replay is defined only on the daily grid: the intra-interval returns the variance
        # recursion replays are unobservable at coarser nodes.
        if np.any(self.n_sub >= 2):
            raise ValueError('reseed_from_path needs n_sub == 1 everywhere; grid is coarser than '
                             f'the trading day (n_sub up to {int(self.n_sub.max())})')
        obs = simulated.detach()
        logret = obs.clamp_min(1.0e-30).log()
        # The SAME composite drift as the forward sim, so the recovered innovation - hence the
        # replayed h - is identical between forward and replay with Carry_Drift on. The carry's
        # own reseed republishes the REALIZED z0 before this runs (topological order).
        drift = self._carry_drift(shared_mem, obs.dim())
        h = torch.zeros_like(obs[0]) + self.h0_default                       # (…B) = H0
        log_h = torch.empty_like(obs)
        log_h[0] = h.log()

        def realized(var_step, t):                                           # realized innovation r_t
            r = (logret[t] - logret[t - 1]) - drift[t]
            return r + 0.5 * var_step if self.convexity else r               # undo the −½Var(r) drift shift

        for t in range(1, obs.shape[0]):
            h, _, _ = self._advance_variance(h, t, lambda v: realized(v, t))
            log_h[t] = h.log()
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'garch_log_h')] = log_h.unsqueeze(1)
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'h0_outer')] = h

    def calibrated_annual_vol(self):
        """Long-run annualised (fractional) vol √(ω/(1−α−β) / dt_c) — the stationary GARCH vol.
        Log_Price is always True here (a log-return model), so it is returned as a fraction,
        consumed directly by the utility-scale fallback."""
        p = self.param
        return float(np.sqrt(p['Omega'] / (1.0 - p['Alpha'] - p['Beta']) / p['Calibration_DT_Years']))

    def revealed_annual_vol(self, log_h):
        """σ_t = √(exp(log h_t)/dt_c): the exactly-observed conditional vol annualized off the
        calibration business-day clock (h_t is the per-business-day variance)."""
        return (log_h.exp() / float(self.param['Calibration_DT_Years'])).sqrt()


class GARCHSpotCalibration(object):
    """Zero-mean GARCH(1,1)-t MLE of GARCHSpotModel on a business-daily close series.
    The fit runs in percent-return units (100·r) for conditioning, then converts ω, H0 back
    to fraction (×1e-4). Uses `arch` if importable, else scipy L-BFGS-B on the identical
    standardised-t log-likelihood. H0 is the filtered conditional variance at the final
    observation, so a calibrated world starts in TODAY's vol state. `Mu` is fixed at 0 —
    16y of daily data cannot identify the drift (s.e. ≈ σ/√years ≈ 5.6% unconditional) —
    and `Log_Price` is always True (the model is defined on log returns).
    `delta` is the standardised residual ε_t = r_t/√h_t — approximately iid under the
    calibrated model, so the framework's correlation consolidation isn't contaminated by
    the GARCH heteroskedasticity (the analog of the HMM's regime-standardised innovation)."""
    model_type = 'GARCHSpotModel'
    fields = [
        F('Outlier_Threshold', 'Float', default=0.25,
          description='|d log S| guard - returns above it are dropped'),
        F('Max_Persistence', 'Float', default=0.999,
          description='Cap on alpha+beta; beta is scaled down to hit it - the model\'s own '
                      'stationarity assertion'),
        F('Nu_Min', 'Float', default=2.05,
          description='Floor on the t degrees of freedom - the model\'s own assertion'),
        F('Convexity_Correction', 'Text', default='Yes', values=['Yes', 'No'],
          description='Stamped straight onto the emitted param block - a calibrated world is a '
                      'PRICE martingale with no harvestable Jensen drift, so this defaults on '
                      'while the MODEL\'s own default stays No for bit-identity')
    ]

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        outlier = float(self.param.get('Outlier_Threshold', 0.25))
        max_persistence = float(self.param.get('Max_Persistence', 0.999))
        nu_min = float(self.param.get('Nu_Min', 2.05))
        convexity = self.param.get('Convexity_Correction', 'Yes')

        dt_c = 1.0 / float(num_business_days)
        px = data_frame.iloc[:, 0].astype(np.float64).dropna()
        r = np.log(px).diff().dropna()
        r = r[r.abs() < outlier]                                                # outlier guard
        x = 100.0 * r.values                                                    # percent units

        omega, alpha, beta, nu, h, se = garch11_t_mle(x)
        h_last = float(h[-1])

        if alpha + beta > max_persistence:
            logging.warning('GARCH persistence %.5f > %.5f — scaling beta down.',
                            alpha + beta, max_persistence)
            beta = max_persistence - alpha
        nu = max(float(nu), nu_min)

        omega_f = float(omega) * 1.0e-4                                         # percent → fraction
        H0 = h_last * 1.0e-4
        persistence = alpha + beta
        lr_vol_ann = float(np.sqrt(omega_f / (1.0 - persistence) / dt_c))
        half_life = float(np.log(0.5) / np.log(persistence))
        logging.info(
            'GARCH(1,1)-t fit: omega=%.4e (se %.2e) alpha=%.4f (t=%.1f) beta=%.4f (t=%.1f) '
            'nu=%.3f (t=%.1f) | persistence=%.5f half-life=%.0fbd LR-ann-vol=%.4f H0=%.4e (ann %.4f)',
            omega_f, se['omega'] * 1e-4, alpha, alpha / se['alpha'], beta, beta / se['beta'],
            nu, nu / se['nu'], persistence, half_life, lr_vol_ann, H0, float(np.sqrt(H0 / dt_c)))

        param = {
            'Omega': omega_f, 'Alpha': float(alpha), 'Beta': float(beta), 'Nu': float(nu),
            'Mu': 0.0, 'H0': H0, 'Log_Price': True, 'Calibration_DT_Years': dt_c,
            'Convexity_Correction': convexity,
        }

        # delta = standardised residual ε_t = r_t/√h_t off the filtered variance path (both in
        # percent units, so the 100· scaling cancels) — approximately iid under the fit.
        delta = pd.DataFrame({data_frame.columns[0]: x / np.sqrt(h)}, index=r.index)

        return utils.CalibrationInfo(param, [[1.0]], delta)


class HestonNandiImpliedSpotModel(StochasticProcess):
    """Heston-Nandi GARCH(1,1) spot under the risk-neutral (LRNVR) measure — an IMPLIED process:
    the bootstrapped `HestonNandiModelParameters` factor that the semi-analytic pricer consumes
    ALSO drives the outer-scenario evolution of its own underlying, reading its parameters from
    `implied_tensor` so greeks flow to the single shared AAD leaf (static-leaf dedupe in
    Calculation._build_factor_state). The model itself is described in the `documentation` attr
    below; this docstring covers the code-facing seams only.

    * Variance recursion: `utils.hn_variance_step` (ONE source of truth, shared with the OSS
      pricers). (r-q) is gathered from the underlying's own rate/dividend curves exactly as
      GBMAssetPriceTSModelImplied (EquityPrice / FxRate branch on factor_type).
    * THE CLOCK: calibrated per trading day (dt_c = 1/`Steps_Per_Year`) vs a calendar scenario
      grid; f = dt/dt_c. n_sub == 1 (f ≤ 1.5): exact fractional step — return variance h·f,
      variance update BLENDED by f (h ← h + f·(hn_variance_step(h,z) − h)); at f=1 this is
      exactly hn_variance_step. Coarse PFE/CVA grids: the interval walks the sub-steps that
      SPAN it — whole trading days then the fractional remainder — through that same
      recursion (utils.hn_correlated_substeps), so a node marginal is the exact law of its
      own elapsed trading time and does not jump with grid spacing. What stays approximate
      is which linear functional carries the cross-factor correlation: the framework draw is
      the √E[h·dt]-weighted combination of the sub-step normals, an F_t-measurable choice
      that leaves every marginal exact but matches a correlated sibling only when it shares
      this variance profile.
    * Observable state: log h_t (predictable, F_t-measurable — no lookahead) in the `hn_log_h`
      buffer. The 7-verb protocol (privileged_layout / privileged_factors / reveal_state_at /
      revealed_annual_vol / inner_fork_seed / outer_reseed / reseed_from_path) mirrors
      GARCHSpotModel; generate handles outer (T,B) and inner (T,B,B2) with the fork seed on the
      MIDDLE axis. The framework Gaussian feeds z directly (no Student-t emission, unlike
      GARCH)."""


    documentation = (
        'Asset Pricing',
        ['A Heston-Nandi GARCH(1,1) risk-neutral spot model. Under the LRNVR measure the daily step '
         'is',
         '',
         '$$ \\Delta\\log S = (r-q) - \\tfrac12 h + \\sqrt{h}\\,z,\\quad '
         'h_{t+1} = \\omega + \\beta h + \\alpha (z - \\gamma^* \\sqrt{h})^2 $$',
         '',
         'with $z\\sim N(0,1)$ iid and $(r-q)$ the per-step cost of carry from the underlying\'s own '
         'interest-rate and dividend/repo curves. The $-\\tfrac12 h$ is the exact Gaussian convexity '
         'so the discounted spot is a price-martingale. The conditional variance $h_t$ is exactly '
         'observable (predictable) and revealed to the value function as $\\log h_t$.',
         '',
         'The parameters $(\\omega,\\alpha,\\beta,\\gamma^*,h_0)$ are the implied '
         '$HestonNandiModelParameters$ factor bootstrapped from the option surface — the same factor '
         'the semi-analytic pricer consumes. The persistence is $\\psi=\\beta+\\alpha\\gamma^{*2}$ and '
         'the stationary per-step variance $\\frac{\\omega+\\alpha}{1-\\psi}$. Simulation runs on the '
         'fractional trading clock $f=dt/dt_c$: exact at $f=1$, variance blended by $f$ for finer '
         'grids, and on coarser grids the step walks the sub-steps spanning it — whole trading '
         'days then the fractional remainder — so a node carries the law of its own elapsed '
         'trading time rather than a rounded number of days.'])

    factor_types = ('EquityPrice', 'FxRate')
    # the parameters live on the implied HestonNandiModelParameters factor, not in this block
    fields = []

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)
        self.implied = implied_factor
        # the underlying's class name selects the r/q curve gather in calc_references (EquityPrice /
        # FxRate), exactly as GBMAssetPriceTSModelImplied.
        self.factor_type = self.factor.__class__.__name__

    @staticmethod
    def num_factors():
        return 1

    @property
    def correlation_name(self):
        return 'HestonNandiSpotProcess', [()]

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        """
        Precompute the fractional trading clock, the HN recursion parameters and the drift plumbing.

        dt is anchored at time_grid_years[0] so the per-step spacing is correct under BOTH outer
        mode (scen_time_grid[0]=0) and inner-MC kept-base mode (>0), mirroring GARCHSpotModel.
        dt_c is the calibration (trading-day) step; the option bootstrapper works at
        Steps_Per_Year (default 252), so the same convention drives the sim.

        Parameters are read out of the HestonNandiModelParameters factor block by the canonical
        `utils.HN_PARAM_NAMES` (single source, mirroring CS's implied_tensor consumption): the
        implied_tensor branch when greeks are on (0-dim AAD leaves), else the calibrated scalars.
        The five scalars feed the explicit-arg utils.hn_* functions; H0 is the variance state
        (it seeds h), the rest are the recursion params.
        """
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size

        # Fractional trading clock, anchored at time_grid_years[0].
        tg_years = time_grid.time_grid_years
        dt_arr = np.diff(np.hstack(([tg_years[0]], tg_years)))
        dt_c = 1.0 / float(self.implied.param.get('Steps_Per_Year', 252.0))
        self.dt_c = dt_c
        self.f = shared.one.new_tensor(dt_arr / dt_c)                             # (T,) trading-time step length
        self.sub_dt = utils.substep_schedule(dt_arr / dt_c)
        self.n_sub = np.array([len(s) for s in self.sub_dt])
        if np.any(self.n_sub >= 2):
            # INFO (precalculate reruns on every inner fork).
            logging.info('HestonNandiImpliedSpotModel coarse grid: n_sub up to %d — exact daily '
                         'sub-stepping, the correlated draw rides the √E[h]-weighted combination.',
                         int(self.n_sub.max()))

        # Parameters ride the implied tensors when greeks are on (mirrors CS's numpy-vs-implied_tensor
        # branch); otherwise the calibrated scalars from the implied factor. Either way they are 0-dim
        # and broadcast against the (…B) variance state.
        p = self.implied.param
        # Read the factor block by the canonical HN_PARAM_NAMES (single source).
        if implied_tensor is not None:
            vals = [implied_tensor[k].reshape(()) for k in utils.HN_PARAM_NAMES]
        else:
            vals = [shared.one.new_tensor(float(p[k])) for k in utils.HN_PARAM_NAMES]
        self.omega, self.alpha, self.beta, self.gamma, self.h0_default = vals
        # detached scalar long-run log-variance for the reveal fallback (buffer key absent)
        om, al = float(p['Omega']), float(p['Alpha'])
        be, ga = float(p['Beta']), float(p['Gamma_Star'])
        self._log_lr_var = float(np.log((om + al) / (1.0 - (be + al * ga ** 2))))

        # Risk-neutral drift plumbing — identical to GBMAssetPriceTSModelImplied: the per-step carry
        # (r-q)·dt is read from the curve as-of each step's START node, anchored at today (t=0) so the
        # first step evolves from spot. delta_scen_t are the calendar step sizes.
        self.delta_scen_t = np.diff(np.insert(time_grid.scen_time_grid, 0, 0)).reshape(-1, 1)
        today = time_grid.scenario_grid[:1].copy()
        today[:, utils.TIME_GRID_MTM] = 0.0
        self.scen_grid = np.vstack([today, time_grid.scenario_grid[:-1]])

        # AAD: keep spot0 on the graph. In log space h depends only on the generated innovations,
        # never on spot0. Outer passes a (1,) scalar; inner-MC a (B,) per-outer-path vector.
        self.spot0 = tensor

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        # Risk-neutral drift links — the underlying's own curves supply r and q (mirrors
        # GBMAssetPriceTSModelImplied). EquityPrice: equity repo/zero + dividend; FxRate: domestic +
        # foreign (repo) zero.
        if self.factor_type == 'EquityPrice':
            self.r_t = get_equity_zero_rate_factor(
                factor.name, static_ofs, stoch_ofs, all_tenors, all_factors)
            self.q_t = get_dividend_rate_factor(factor.name, static_ofs, stoch_ofs, all_tenors)
        elif self.factor_type == 'FxRate':
            self.r_t = get_fx_zero_rate_factor(
                self.factor.get_domestic_currency(None), static_ofs, stoch_ofs, all_tenors, all_factors)
            self.q_t = get_fx_zero_rate_factor(factor.name, static_ofs, stoch_ofs, all_tenors, all_factors)
        else:
            raise Exception('HestonNandiImpliedSpotModel unsupported factor type {}'.format(self.factor_type))

    def _carry_per_step(self, shared_mem, shape):
        """Per-step cost of carry (r-q)·dt as a (T, …batch) tensor — the same r_t/q_t curve gather
        GBMAssetPriceTSModelImplied uses for its drift, but kept PER STEP (not cumulated): each HN
        daily step carries its own (r-q) and the full log-return is cumulated in generate. Outer
        (T,B); inner (T,B,B2) via n_batch_dims=2 (the curve stack collapses (B,B2)→B*B2, reshaped
        back). ds[0] stays the anchor (0) so carry[0] is unused. `shape` is the driver tensor's shape
        (Z in generate, the observed path in reseed_from_path)."""
        if len(shape) == 2:
            rt = utils.calc_time_grid_curve_rate(self.r_t, self.scen_grid, shared_mem)
            qt = utils.calc_time_grid_curve_rate(self.q_t, self.scen_grid, shared_mem)
            rt_rates = rt.gather_weighted_curve(shared_mem, self.delta_scen_t)
            qt_rates = qt.gather_weighted_curve(shared_mem, self.delta_scen_t)
            return torch.squeeze(rt_rates - qt_rates, dim=1)                      # (T, B)
        T, B, B2 = shape
        rt = utils.calc_time_grid_curve_rate(self.r_t, self.scen_grid, shared_mem, n_batch_dims=2)
        qt = utils.calc_time_grid_curve_rate(self.q_t, self.scen_grid, shared_mem, n_batch_dims=2)
        rt_rates = rt.gather_weighted_curve(shared_mem, self.delta_scen_t)
        qt_rates = qt.gather_weighted_curve(shared_mem, self.delta_scen_t)
        carry = torch.squeeze(rt_rates - qt_rates, dim=1)
        # A STOCHASTIC (simulated) curve fans the collapsed (B,B2)→B*B2 batch — reshape it back.
        # A STATIC curve carries no batch axis; keep it (T,1,1) so each per-step scalar broadcasts
        # across the (B,B2) fan-out.
        if carry.numel() == T * B * B2:
            return carry.reshape(T, B, B2)
        return carry.reshape(T, 1, 1)

    def _simulate_returns(self, z, h, carry):
        """Shared HN recursion (outer/inner) on the FRACTIONAL TRADING CLOCK. `z` (T, …) standard
        normals, `h` (…) the entry variance h_0. Returns (ds, log_h): ds[t] is Δlog S landing at t
        (ds[0]=0, the dt=0 anchor), log_h[t] the revealed variance of the move t→t+1 (F_t-measurable —
        a function of z[1..t] only, no lookahead).

        n_sub≤1: exact fractional step — var = h·f_t, and the variance update h ← h + f·(hn step − h)
        which is EXACTLY hn_variance_step at f=1. A step spanning more than one calibration step
        walks the sub-steps that span it (utils.hn_correlated_substeps — z[t] rides the
        √E[h·dt]-weighted combination of the sub-step normals)."""
        log_h = torch.empty_like(z)
        ds = torch.zeros_like(z)
        log_h[0] = h.log()
        for t in range(1, z.shape[0]):
            if self.n_sub[t] <= 1:
                h, var_step, r = self._advance_variance(h, t, lambda v: z[t])
            else:
                h, var_step, r = utils.hn_correlated_substeps(
                    h, z[t], self.sub_dt[t], self.omega, self.alpha, self.beta, self.gamma)
            ds[t] = carry[t] - 0.5 * var_step + r
            log_h[t] = h.log()
        return ds, log_h

    def _advance_variance(self, h, t, standard_normal):
        """One fractional-clock Heston–Nandi variance step (n_sub ≤ 1; a coarse interval routes to
        utils.hn_correlated_substeps in `_simulate_returns` instead), shared by the forward sim and
        the observed-path replay (`reseed_from_path`) so the forward≡replay invariant is STRUCTURAL,
        not maintained by copying. Computes Var(Δlog S) for the step, obtains the standard normal z
        via `standard_normal(var_step)` — the ONLY thing that differs between the paths (the drawn
        z[t] forward; the realized z=(Δlog S − carry + ½·Var)/√Var in replay) — and returns
        (h_{t+1}, var_step, r) with r = √Var·z."""
        ft = self.f[t]
        var_step = h * ft                                                        # Var(Δlog S) over the step
        sh = h.sqrt()
        z = standard_normal(var_step)
        r = var_step.sqrt() * z
        # fractional HN variance update: exact hn_variance_step at f=1, blended by f for f<1
        return h + ft * (utils.hn_variance_step(
            h, sh, z, self.omega, self.alpha, self.beta, self.gamma) - h), var_step, r

    def generate(self, shared_mem):
        # One framework Gaussian per step drives the HN z DIRECTLY (no Student-t rescale, no auxiliary
        # sampling stream). Z is (T, B) outer, (T, B, B2) inner.
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        carry = self._carry_per_step(shared_mem, Z.shape)

        if Z.ndim == 2:
            B = Z.shape[1]
            # h0: diff-ML t=0 randomization hook (mirrors GARCH regime0/h0_outer); else H0 expanded.
            h0 = shared_mem.t_Scenario_Buffer.get((self.factor_key, 'h0_outer'))
            h = h0 if h0 is not None else torch.zeros_like(Z[0]) + self.h0_default
            ds, log_h = self._simulate_returns(Z, h, carry)
            s0 = self.spot0.expand(B)                                            # (1,) -> (B,)
            log_path = s0.log().unsqueeze(0) + ds.cumsum(dim=0)                  # (T, B)
        else:
            B, B2 = Z.shape[1], Z.shape[2]
            # h0: per-outer-path fork seed on the MIDDLE (B) axis, expanded across B2; else H0.
            h0 = shared_mem.t_Scenario_Buffer.get((self.factor_key, 'h0_inner'))
            h = h0.view(B, 1).expand(B, B2) if h0 is not None else torch.zeros_like(Z[0]) + self.h0_default
            ds, log_h = self._simulate_returns(Z, h, carry)
            s0 = self.spot0                                                      # (B,)
            log_path = s0.view(B, 1).log() + ds.cumsum(dim=0)                    # (T, B, B2)
        # Floor before exp(): a large negative innovation can underflow exp() to 0.0 and break the
        # price invariant (same guard as GARCHSpotModel).
        spot_path = log_path.clamp_min(-10.0).exp()

        # Revealed log h, detached (a state coordinate, not differentiated through). B-LAST so the
        # buffer's dim=-1 concat works: (T,1,B) outer / (T,1,B,B2) inner.
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'hn_log_h')] = log_h.detach().unsqueeze(1)
        self.last_log_h = log_h.detach()
        return spot_path

    @classmethod
    def privileged_layout(cls, param):
        return {'log_h': 1}

    def privileged_factors(self, simulated):
        # (T, B, 1) — the HMM/GARCH accumulator convention. Called outer-mode only.
        return {'log_h': self.last_log_h.to(torch.float32).unsqueeze(-1)}

    def reveal_state_at(self, t, buffer):
        """log_h-first / price-last (matching GARCHSpotModel and the HMM belief-first/price-last
        packing). log h_t is the exactly-observable predictable variance; the spot level is the last
        CONTINUOUS coordinate. Rank-check the buffered log-h against the current mode, else the
        long-run-variance fallback when the buffer key is absent."""
        key = self.factor_key
        price = buffer[key][t].unsqueeze(0)                                      # (1, …batch)
        log_h = buffer.get((key, 'hn_log_h'))
        if log_h is not None and log_h.dim() == buffer[key].dim() + 1:
            block = log_h[t]                                                     # (1, …batch)
        else:
            block = torch.full_like(price, self._log_lr_var)
        return [(block, REVEAL_SUFFICIENT), (price, REVEAL_CONTINUOUS)]

    def inner_fork_seed(self, factor_key, outer_buf, t):
        """Per-outer-path t=0 conditional-variance seed: h0_inner = exp(outer log h_t), so the inner
        fan-out reprices from the forked path's vol state (not the calibrated H0)."""
        return {(factor_key, 'h0_inner'): outer_buf[(factor_key, 'hn_log_h')][t].reshape(-1).exp()}

    def outer_reseed(self):
        """t=0 conditional-variance seed for the next outer run's burn-in: terminal h of this run."""
        return {(self.factor_key, 'h0_outer'): self.last_log_h[-1].exp()}

    def reseed_from_path(self, simulated, shared_mem):
        """Observed-path replay: rerun the HN variance recursion (fractional clock) on the REALIZED
        returns of the supplied price path, publishing `hn_log_h` (so reveal returns the right log h
        along the replayed path) and `h0_outer` (terminal h). The innovation is recovered from the
        realized log-return with the SAME carry/convexity the forward sim used, so the h recursion
        stays lock-step: z = (Δlog S − carry + ½·Var)/√Var, Var = h·f."""
        # Replay is defined only on the daily grid: the intra-interval returns the variance
        # recursion replays are unobservable at coarser nodes.
        if np.any(self.n_sub >= 2):
            raise ValueError('reseed_from_path needs n_sub == 1 everywhere; grid is coarser than '
                             f'the trading day (n_sub up to {int(self.n_sub.max())})')
        obs = simulated.detach()
        logret = obs.clamp_min(1.0e-30).log()
        carry = self._carry_per_step(shared_mem, obs.shape).detach()
        h = torch.zeros_like(obs[0]) + self.h0_default                          # (…B) = H0
        log_h = torch.empty_like(obs)
        log_h[0] = h.log()

        def realized(var_step, t):                                             # z recovered from the realized return
            return ((logret[t] - logret[t - 1]) - carry[t] + 0.5 * var_step) / var_step.sqrt()

        for t in range(1, obs.shape[0]):
            h, _, _ = self._advance_variance(h, t, lambda v: realized(v, t))
            log_h[t] = h.log()
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'hn_log_h')] = log_h.unsqueeze(1)
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'h0_outer')] = h

    def calibrated_annual_vol(self):
        """Long-run annualised vol √( (ω+α)/(1−ψ) / dt_c ) — the stationary HN vol off the
        calibration (trading-day) clock."""
        p = self.implied.param
        om, al, be, ga = (float(p[k]) for k in ('Omega', 'Alpha', 'Beta', 'Gamma_Star'))
        lr_var = (om + al) / (1.0 - (be + al * ga ** 2))
        return float(np.sqrt(lr_var / self.dt_c))

    def revealed_annual_vol(self, log_h):
        """σ_t = √(exp(log h_t)/dt_c): the exactly-observed conditional vol annualized off the
        calibration trading-day clock (h_t is the per-trading-day variance)."""
        return (log_h.exp() / self.dt_c).sqrt()


class QuadraticCarryCurveModel(StochasticProcess):
    """Two-factor continuous carry curve on a `ForwardRate` factor. The quadratic log-futures curve

        F(t,T) = S(t)·exp(c(t)·τ + a(t)·τ²),    τ = (T − t) / DAYS_IN_YEAR

    is carried as the AVERAGE CARRY TO MATURITY, z(t,τ) = (c·τ + a·τ²)/τ = c + a·τ, because that is
    the quantity `utils.DerivedForwardCurve` already prices off: it gathers the carry curve at the
    query DATE with `multiply_by_time=False` and multiplies by τ, so a curve holding z reproduces
    the quadratic through ZERO new read code. z is AFFINE in τ and the gather interpolates linearly
    in the query date — affine in τ at a fixed row — so TWO knots reproduce the whole curve exactly
    between them, and the three listed futures that identified it come back to float precision.

    NOT beyond them. `utils.CurveTenor.get_index` CLIPS a query to [first knot, last knot], so the
    read outside the bracket is FLAT in z, i.e. the log-carry continues LINEARLY rather than
    quadratically. Measured on S=950, c=0.0163, a=−0.0011 with knots at τ = 0.5 and 1.0: exact to
    0 ULP for τ ∈ [0.5, 1.0], −2.5e−5 relative at τ = 0.05 and +8.3e−4 at τ = 1.5. The knots are
    DATES and the clip is in date space, so the rule the market data has to honour is: the first
    knot at or before the base date, the last at or after the longest fixing — then every (row
    date, query date) pair the book reaches is inside. They are the FACTOR's own knots and this
    process never chooses them; it publishes z at whatever τ each has aged to, negative τ included,
    which is a perfectly good value of an affine function and is what keeps the live reads exact.

    STATE. The polynomial coefficients are ill-conditioned (ρ(Δc,Δa) ≈ −0.96 on three nearby
    maturities: curvature and front slope trade off to hold the same observed futures), so the
    driven pair is the level/shape rotation at the declared `Reference_Tenors` τ_A < τ_B:

        L = (z(τ_A) + z(τ_B))/2      the carry LEVEL, mid-tenor average carry
        D =  z(τ_B) − z(τ_A)         the carry SHAPE, the knot spread

    invertibly, with τ̄ = (τ_A+τ_B)/2 and Δτ = τ_B−τ_A:

        z(τ) = L + D·(τ − τ̄)/Δτ,     a = D/Δτ,   c = L − D·τ̄/Δτ

    At the shipped τ_A, τ_B = 0.5, 1.0 that is L = c + 0.75a and D = 0.5a — the handover's
    carry-level and carry-shape factors, and the two archive columns' mean and difference.

    NO HIDDEN STATE, and that is the whole reason the fork verbs are inert here. The published
    curve IS the state: `precalculate` recovers (L,D) from the initial curve `tensor` by the same
    affine map, so the inner-MC fork (whose init is the outer path's curve at the fork row), the
    diff-ML burn-in (whose init is the terminal curve) and the observed-path replay all carry the
    state without a single private buffer key. A declared `L_0`/`D_0` pair would be a SECOND source
    for a number the factor already holds, and the market curve has to win — row 0 is published as
    `tensor` itself, so the t=0 forwards are the market's own.

    DYNAMICS per step, on the framework-correlated Z and one internal chi²:

        L_t = μ_L + φ_L(L_{t-1} − μ_L) + σ_L·ε^L_t
        D_t = μ_D + φ_D(D_{t-1} − μ_D) + Γ·(L_t − L_{t-1}) + σ_D·ε^D_t

    ΔL is the SAME step's level change — known once L_t is drawn, so this is a contemporaneous
    loading and not a lookahead; reading L_{t+1} − L_t would need the next step's draw. ONE
    Chi²(ν) is drawn per (step, path) and SHARED by both factors, so (ε^L, ε^D) is an elliptical
    bivariate t rather than two t marginals under a Gaussian copula — which is what the data shows,
    the largest ΔL days being the largest ΔD days.

    Γ AND THE DECLARED L/D CORRELATION ARE THE SAME COUPLING, twice: the one-step covariance is
    Cov(ΔL, ΔD) = Γ·σ_L² + ρ·σ_L·σ_D, one equation in two unknowns, and no fit can separate them.
    That is not a defect and it is not double counting — the calibration fits Γ on the conditional
    mean and hands the framework the RESIDUAL correlation, so the two always sum to the observed
    covariance and only their split moves. It does move: the sample's own Γ swings from −0.019 to
    −0.45 with the tail weight of the likelihood while ρ walks the other way, −0.22 to −0.06, which
    makes Γ the one number in this block a reader must not interpret on its own.

    THE CLOCK. φ and σ are calibrated per `Calibration_DT_Years` step while the sim grid runs in
    calendar time, so a step of length f = dt/dt_c takes φ_f = φ^f and σ_f = σ·√((1−φ_f²)/(1−φ²)):
    the exact stationary AR(1) aggregation, which makes the reversion RATE and the stationary
    variance grid-invariant and degrades to the random-walk σ·√f as φ → 1. Γ is not rescaled — it
    loads on whatever ΔL the step produced, exact at f = 1.

    A MODELLING CAVEAT WITH A NUMBER. φ_L is a NEAR-UNIT ROOT (0.9962 fitted, 2.3 s.e. from 1), so
    nothing may lean on carry reversion for value: over three months the conditional mean gives up
    only 1 − φ^63 ≈ 19% of a level deviation, and the level is statistically indistinguishable from
    a random walk. `tests/test_quadratic_carry_curve.py` holds that as a gate rather than a note.

    JSON config:
        Phi_L, Mu_L, Sigma_L: AR(1) coefficient, long-run mean and innovation STD of the level.
        Phi_D, Mu_D, Sigma_D: the same for the shape.
        Gamma: loading of the shape on the same step's level change.
        Nu: Student-t degrees of freedom, shared by both factors (> 2).
        Reference_Tenors: [τ_A, τ_B] in years — the tenors (L, D) are defined at.
        Calibration_DT_Years: step size (in years) of the calibrated recursions."""

    documentation = (
        'Energy Pricing',
        ['A two-factor continuous carry curve for a commodity futures curve. The quadratic '
         'log-futures curve $F(t,T) = S_t\\exp(c_t\\tau + a_t\\tau^2)$ is carried as the average '
         'carry to maturity',
         '',
         '$$ z(t,\\tau) = \\frac{c_t\\tau + a_t\\tau^2}{\\tau} = c_t + a_t\\tau, \\qquad '
         '\\log F(t,T) = \\log S_t + z(t,\\tau)\\,\\tau $$',
         '',
         'which is affine in $\\tau$, so two curve knots reproduce the whole quadratic exactly '
         'between them and the forward-curve reader needs no new code. The driven state is the '
         'well-conditioned level/shape rotation at two reference tenors $\\tau_A<\\tau_B$ '
         '(the raw polynomial coefficients have $\\rho(\\Delta c,\\Delta a)\\approx-0.96$):',
         '',
         '$$ L = \\tfrac{1}{2}(z(\\tau_A)+z(\\tau_B)), \\quad D = z(\\tau_B)-z(\\tau_A), \\quad '
         'z(\\tau) = L + D\\frac{\\tau-\\bar\\tau}{\\Delta\\tau} $$',
         '',
         'with dynamics',
         '',
         '$$ L_t = \\mu_L + \\phi_L(L_{t-1}-\\mu_L) + \\sigma_L\\varepsilon^L_t $$',
         '$$ D_t = \\mu_D + \\phi_D(D_{t-1}-\\mu_D) + \\Gamma(L_t-L_{t-1}) + '
         '\\sigma_D\\varepsilon^D_t $$',
         '',
         'where $(\\varepsilon^L,\\varepsilon^D)$ is a standardised bivariate Student-t: the '
         'framework supplies the correlated Gaussians and ONE $\\chi^2(\\nu)$ per step is shared '
         'by both factors, so the tails are jointly heavy rather than independently so.',
         '',
         '- **Phi_L, Mu_L, Sigma_L**: level AR(1) coefficient, mean and innovation std.',
         '- **Phi_D, Mu_D, Sigma_D**: the same for the shape.',
         '- **Gamma**: shape loading on the same step\'s level change.',
         '- **Nu**: Student-t degrees of freedom, shared by both factors.',
         '- **Reference_Tenors**: the two tenors (years) L and D are defined at.',
         '- **Calibration_DT_Years**: step size (in years) of the calibrated recursions.'])

    factor_types = ('ForwardRate',)
    fields = [
        F('Phi_L', 'Float', default=0.0, description='AR(1) coefficient of the carry level'),
        F('Mu_L', 'Float', default=0.0, description='Long-run mean of the carry level'),
        F('Sigma_L', 'Float', default=0.0,
          description='Innovation standard deviation of the carry level'),
        F('Phi_D', 'Float', default=0.0, description='AR(1) coefficient of the carry shape'),
        F('Mu_D', 'Float', default=0.0, description='Long-run mean of the carry shape'),
        F('Sigma_D', 'Float', default=0.0,
          description='Innovation standard deviation of the carry shape'),
        F('Gamma', 'Float', default=0.0,
          description='Loading of the carry shape on the same step\'s change in the level'),
        F('Nu', 'Float', default=5.0,
          description='Student-t degrees of freedom (> 2), shared by both factors'),
        F('Reference_Tenors', 'Container', default=[0.5, 1.0],
          description='The two tenors (in years) the level and shape are defined at - z(tau_A) '
                      'and z(tau_B) - stamped from the archive column sub-keys'),
        F('Calibration_DT_Years', 'Float', default=1.0 / 252.0,
          description='Step size (in years) of the calibrated AR(1) recursions')
    ]

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)
        # 0 <= phi < 1 rather than |phi| < 1: the fractional-step aggregation below is phi^f, which
        # has no real value for a negative coefficient, and neither carry factor is anti-persistent.
        assert (0.0 <= param['Phi_L'] < 1.0 and 0.0 <= param['Phi_D'] < 1.0 and param['Nu'] > 2.0
                and param['Sigma_L'] > 0.0 and param['Sigma_D'] > 0.0), \
            f'QuadraticCarryCurveModel invalid params: {param}'

    @staticmethod
    def num_factors():
        return 2

    @property
    def correlation_name(self):
        return 'QuadraticCarryCurveProcess', [('L',), ('D',)]

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        """
        Age the factor's dated knots onto the sim grid, rescale the recursions to it, and recover
        the initial (L, D) from the initial curve.

        `k[t]` is the knot's shape coordinate (τ − τ̄)/Δτ at row t, so a published row is
        `L + D·k[t]` — one expression covering the interpolation between the knots and the
        extrapolation of a knot whose date the simulation has passed (τ < 0, which is a perfectly
        good value of an affine function and is what keeps the read exact for live query dates).

        The per-step AR coefficients are anchored at `time_grid_years[0]`, so the step lengths are
        right under both outer mode (row 0 at the base date) and an inner-MC fork (row 0 at the
        fork date, its own `ref_date`).

        AAD: the recovery is a matmul by a CONSTANT 2x2, so ∂(published curve)/∂(market curve)
        flows without a solve; `tensor` is stored unreshaped so inner MC can pass `(2, B)`.
        """
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        knots = np.asarray(self.factor.get_tenor(), dtype=np.float64)
        if knots.size != 2:
            raise ValueError(
                f'QuadraticCarryCurveModel needs exactly 2 curve knots holding the average carry '
                f'z(tau) = c + a*tau (they identify the quadratic and must BRACKET the tenors the '
                f'book prices); the factor declares {knots.size}.')

        tau_a, tau_b = (float(x) for x in self.param['Reference_Tenors'])
        excel_offset = (ref_date - utils.excel_offset).days
        tau = (knots.reshape(1, -1) - (time_grid.scen_time_grid + excel_offset).reshape(-1, 1)
               ) / utils.DAYS_IN_YEAR                                        # (T, 2) ageing knots
        k = (tau - 0.5 * (tau_a + tau_b)) / (tau_b - tau_a)                  # (T, 2) shape coords
        self.k = shared.one.new_tensor(k)
        # z(0) = L + D*(0 - taubar)/dtau: the FRONT carry. Published per (step, path) as
        # (key, 'z0') for a spot whose Carry_Drift consumes it as its own log-drift rate.
        self.z0_coeff = -0.5 * (tau_a + tau_b) / (tau_b - tau_a)

        # Exact stationary AR(1) aggregation to a step of f calibration steps.
        tg_years = time_grid.time_grid_years
        f = np.diff(np.hstack(([tg_years[0]], tg_years))) / float(self.param['Calibration_DT_Years'])

        def _ar(phi, sigma):
            phi_f = phi ** f
            return (shared.one.new_tensor(phi_f),
                    shared.one.new_tensor(sigma * np.sqrt((1.0 - phi_f * phi_f) / (1.0 - phi * phi))))

        self.phi_L, self.sig_L = _ar(float(self.param['Phi_L']), float(self.param['Sigma_L']))
        self.phi_D, self.sig_D = _ar(float(self.param['Phi_D']), float(self.param['Sigma_D']))
        self.mu_L = float(self.param['Mu_L'])
        self.mu_D = float(self.param['Mu_D'])
        self.gamma = float(self.param['Gamma'])
        self.nu = float(self.param['Nu'])

        # (L, D) from the initial curve: z = M @ (L, D) with M = [[1, k], ...] at row 0.
        self.z0 = tensor
        self.state0 = shared.one.new_tensor(
            np.linalg.inv(np.column_stack([np.ones(2), k[0]]))) @ tensor

    def generate(self, shared_mem):
        """
        Simulate the two carry factors and publish the curve they imply; Z is (2, T, B) outer and
        (2, T, B, B2) inner.

        ONE loop covers both modes — every expression broadcasts, and only the initial state needs
        the mode, a `(2,)` calibrated pair against a `(2, B)` per-outer-path fork vector whose
        column must land on the MIDDLE axis and spread across the B2 fan-out.
        """
        Z = shared_mem.t_random_numbers[self.z_offset:self.z_offset + 2, :self.scenario_horizon]
        inner = Z.ndim == 4
        T, batch = Z.shape[1], Z.shape[2:]
        # ONE chi2 per (step, path), SHARED by the two factors, so the innovation pair is an
        # elliptical bivariate t; the √((ν-2)/W) rescale makes σ its realised std for any ν.
        W = torch.distributions.Chi2(
            shared_mem.one.new_tensor(self.nu)).sample(Z.shape[1:]).clamp_min(1.0e-6)
        eps = Z * torch.sqrt((self.nu - 2.0) / W)                            # (2, T, ...batch)
        k = self.k.reshape(T, 2, *([1] * len(batch)))
        out = torch.empty((T, 2, *batch), device=Z.device, dtype=Z.dtype)
        # `align_rank` rather than one `unsqueeze`: the initial curve is `(2,)` calibrated and
        # `(2, B)` per path, and the per-path one arrives in BOTH modes - an inner fork's row and
        # the diff-ML burn-in's terminal curve, which is a (2, B) pair in OUTER mode.
        out[0] = self.align_rank(self.z0, 1 + len(batch)).expand(2, *batch)
        state = self.state0.unsqueeze(-1) if inner else self.state0
        L, D = state[0], state[1]
        front = torch.empty((T, *batch), device=Z.device, dtype=Z.dtype)
        front[0] = torch.broadcast_to(L + self.z0_coeff * D, batch)
        for t in range(1, T):
            L_next = self.mu_L + self.phi_L[t] * (L - self.mu_L) + self.sig_L[t] * eps[0, t]
            D = (self.mu_D + self.phi_D[t] * (D - self.mu_D) + self.gamma * (L_next - L)
                 + self.sig_D[t] * eps[1, t])
            L = L_next
            out[t] = L + D * k[t]
            front[t] = L + self.z0_coeff * D
        # ATTACHED, not detached: a Carry_Drift spot consumes this as DYNAMICS, so the gradient
        # of its path with respect to the market carry curve flows through here by design -
        # unlike log_h/basis_mu, which are detached state COORDINATES. Fork-index convention:
        # front[t] is what the spot's step t->t+1 drifts at.
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'z0')] = front.unsqueeze(1)
        return out

    def reseed_from_path(self, simulated, shared_mem):
        """Observed-path replay: the curve IS the state (no hidden recursions), so the only
        republication owed is the derived front carry `(key, 'z0')` a Carry_Drift spot consumes -
        recomputed per row from the SUBSTITUTED curve by the same affine inversion precalculate
        uses, before the spot's own reseed runs (topological order provides that)."""
        z = simulated.detach()                                       # (T, 2, ...batch)
        k = self.k.reshape(self.k.shape[0], 2, *([1] * (z.dim() - 2)))
        D = (z[:, 1] - z[:, 0]) / (k[:, 1] - k[:, 0])
        L = z[:, 0] - D * k[:, 0]
        shared_mem.t_Scenario_Buffer[(self.factor_key, 'z0')] = (
            L + self.z0_coeff * D).unsqueeze(1)


class QuadraticCarryCurveCalibration(object):
    """Calibration of QuadraticCarryCurveModel from two archive columns of the average carry to
    maturity, `ForwardRate.<name>,<tau>` at the two reference tenors — whose sub-keys ARE the
    `Reference_Tenors` the fit stamps, so the state definition cannot drift from the data it was
    identified on.

    `L` and `D` are the two columns' mean and difference; the level is fitted as an AR(1)-t and the
    shape as an ARX(1)-t on the same step's ΔL, jointly and with ONE ν (`arx1_t_mle`), because the
    model shares one chi² draw between them. `delta` is the pair of standardised residuals, so the
    framework's correlation consolidation sees innovations that are approximately iid rather than
    the raw heteroskedastic changes; `correlation` is the 2x2 identity, each column mapping 1-1 to
    a primitive factor.

    WHAT THE TAIL WEIGHT DECIDES, this estimator's own output on data/plat_archive_sync.csv
    (3786 rows, 2010-2026). `Nu_Min` is a MODELLING choice and not a guard rail: it decides how
    much weight the likelihood puts on the tail, and the tail is where the level/shape coupling
    lives. Γ and the residual correlation ρ move in LOCKSTEP and in opposite directions —

        ν      φ_L      σ_L       φ_D      σ_D        Γ         ρ(δ_L, δ_D)
        2.05   0.9967   0.004672  0.9521   0.009989   +0.0063   −0.2367
        3.00   0.9962   0.001480  0.9468   0.003085   −0.0194   −0.2245
        6.00   0.9948   0.001336  0.9369   0.002678   −0.0710   −0.1996
        50.0   0.9897   0.001687  0.9077   0.003271   −0.2505   −0.1098
        200    0.9878   0.001857  0.8937   0.003568   −0.3559   −0.0553
        OLS    —        —         0.8841   —          −0.4469   (ΔR² 0.0545)

    — and that is the point, not a defect. Γ and ρ are COLLINEAR: they enter the one-step
    Cov(ΔL, ΔD) as Γ·σ_L² + ρ·σ_L·σ_D, one equation in two unknowns, so the fit slides along a line
    and only the SUM is identified. Both ends are stamped consistently — Γ off the conditional
    mean, ρ off the `delta` the framework consolidates — so the simulated coupling is the observed
    one wherever the fit lands. `tests/test_quadratic_carry_curve.py` generates the same market
    twice, once as pure Γ and once as pure ρ, and gets the same Γ back from both.

    The default `Nu_Min = 3.0` is the lowest ν whose innovation variance exists, matching
    `BasisLinkedSpotCalibration`. The unconstrained MLE runs to the floor (these series have excess
    kurtosis ~50), which is where the completed study's φ_L = 0.9967 and φ_D = 0.9521 come from —
    it reported the raw t SCALE, smaller than σ by √((ν−2)/ν), hence its 0.00073 against the
    0.004672 above."""
    model_type = 'QuadraticCarryCurveModel'
    fields = [
        F('Nu_Min', 'Float', default=3.0,
          description='Floor on the shared degrees of freedom - 3.0 is the lowest value whose '
                      'innovation variance exists, and it sets how much weight the fit puts on '
                      'the tail where the level/shape coupling lives'),
        F('Nu_Max', 'Float', default=50.0, description='Ceiling on the same fit'),
        F('Location_Window', 'Integer', default=0,
          description='When > 0, the LEVEL location (Mu_L, Phi_L) is refitted on this many '
                      'trailing rows while scale, tails and the shape equation keep the full '
                      'window - the carry level is the fast-moving component (a stale mean '
                      'forecasts worse than a random walk) while its scale and nu want the '
                      'longer sample. 0 = single-window legacy fit')
    ]

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 2

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        nu_bounds = (float(self.param.get('Nu_Min', 3.0)), float(self.param.get('Nu_Max', 50.0)))
        if len(data_frame.columns) != 2:
            raise ValueError(
                f'QuadraticCarryCurveCalibration needs exactly two average-carry columns '
                f'`ForwardRate.<name>,<tau>`; got {list(data_frame.columns)}.')
        cols = sorted(data_frame.columns, key=lambda c: float(c.split(',', 1)[1]))
        tenors = [float(c.split(',', 1)[1]) for c in cols]
        joint = data_frame[cols].astype(np.float64).dropna()
        z_a, z_b = joint[cols[0]].values, joint[cols[1]].values
        L, D = 0.5 * (z_a + z_b), z_b - z_a

        (fit_L, fit_D), nu, se = arx1_t_mle([(L, None), (D, np.diff(L))], nu_bounds=nu_bounds)
        phi_L, mu_L, sigma_L, res_L = fit_L.phi, fit_L.mu, fit_L.sigma, fit_L.resid
        phi_D, mu_D, sigma_D, gamma, res_D = fit_D.phi, fit_D.mu, fit_D.sigma, fit_D.gamma, fit_D.resid
        loc_window = int(self.param.get('Location_Window', 0))
        if 0 < loc_window < len(L):
            (fit_Lt,), nu_t, se_t = arx1_t_mle([(L[-loc_window:], None)], nu_bounds=nu_bounds)
            logging.info('carry level location from trailing %d rows: mu_L %.5f -> %.5f, '
                         'phi_L %.4f -> %.4f (scale/tails keep the full window)',
                         loc_window, mu_L, fit_Lt.mu, phi_L, fit_Lt.phi)
            phi_L, mu_L = fit_Lt.phi, fit_Lt.mu
        logging.info(
            'carry curve (L, D)-t fit: phi_L=%.4f (%.2f se from a unit root) mu_L=%.5f '
            'sigma_L=%.6f | phi_D=%.4f mu_D=%.6f sigma_D=%.6f gamma=%+.4f (t=%.1f) | nu=%.2f',
            phi_L, (1.0 - phi_L) / se['phi'][0], mu_L, sigma_L, phi_D, mu_D, sigma_D,
            gamma, gamma / se['gamma'][1], nu)

        param = {
            'Phi_L': phi_L, 'Mu_L': mu_L, 'Sigma_L': sigma_L,
            'Phi_D': phi_D, 'Mu_D': mu_D, 'Sigma_D': sigma_D,
            'Gamma': gamma, 'Nu': nu, 'Reference_Tenors': tenors,
            'Calibration_DT_Years': 1.0 / float(num_business_days),
        }
        archive_name = cols[0].split(',', 1)[0]
        delta = pd.DataFrame(
            {f'{archive_name},L': res_L, f'{archive_name},D': res_D}, index=joint.index[1:])
        return utils.CalibrationInfo(param, np.eye(2).tolist(), delta)


class ChainedBasisModel(StochasticProcess):
    """A basis on a CLOSED chain — that is what 'chained' means: the `Chained_Basis`
    declarations walk back to their start (the AM/PM session pair is the 2-cycle), and an open
    link riding another factor's finished path is the linked-parent family
    (`BasisLinkedSpotModel`), not this class. This factor draws as a BRIDGE between consecutive
    observations of its declared link's finished path:

        b(t) = P(t) + w·(P(t+1) − P(t)) + premium + σ·Z(t)

    The source is the declared `Chained_Basis` and nothing else — no naming convention — and it
    must have generated first (the read fails loud naming both when it has not). Because the
    loop's other members generate earlier in the positional order, the bridge is the acyclic
    spelling of the closed chain: it reproduces both measured links and the lag-1 news channel
    (corr(η_pm(t), η_am(t+1)) > 0) by construction, which same-row correlation alone cannot
    (measured: +0.29 in the data, ≈0 under R-only).

    MEMORYLESS given the source path — the data killed the reversal term (R² = 0.005) — so
    there is no sequential loop, no extra fork seed and no replay recursion: row 0 is the
    declared `Spot` (observed, like every factor's), a fork's row 0 is the forked value the
    calc already hands `precalculate`, and the burn-in's terminal carry rides the generic
    initial-state seam. Rows draw vectorised; the LAST bracketed row is decided by
    source-bracket availability (`t+1` beyond the source's grid → the forward half alone).

    The block declares the two LINK residuals and the source's DAILY step std — the three
    measured objects of the chain — and the bridge derives the exact identities

        w = 1/2 + (σ_ID² − σ_ON²) / (2·σ_D²),   σ² = σ_ID² − w²·σ_D²,   σ_half = σ_ID

    which reproduce all three at the source's own scale. The independence recomposition
    (w = σ_ID²/(σ_ID²+σ_ON²), σ_D² = σ_ID²+σ_ON²) is the ρ(ID,ON) = 0 special case and the
    data refutes it: the half-steps are ANTI-correlated (ρ ≈ −0.4 on the calibration window
    and on 2020+, the AM transient partially reverting overnight), so σ_ID²+σ_ON² overstates
    the daily step and an independence-derived bridge under-sizes both links ~10% at the true
    source scale. An infeasible triple (σ² ≤ 0, or w outside (0,1)) raises at precalculate.

    JSON config:
        Link_ID_Sigma, Link_ON_Sigma: the chain-link residual stds, $/oz
        Link_Daily_Sigma: the source's daily step std on the same window, $/oz
        Bridge_Premium: mean offset net of the bridge, $/oz"""

    documentation = (
        'Asset Pricing',
        ['A basis on a CLOSED chain (the `Chained_Basis` declarations walk back to their '
         'start — the AM/PM session pair is the 2-cycle), drawn as a bridge between '
         'consecutive observations of its declared link\'s finished path:',
         '',
         '$$ b_t = P_t + w\\,(P_{t+1} - P_t) + \\text{premium} + \\sigma Z_t $$',
         '',
         'with w and σ derived by the exact bridge identities from the two chain-link '
         'residuals and the source\'s daily step std — the half-steps are anti-correlated in '
         'the data, so the ρ = 0 recomposition is not used. The bridge is the acyclic '
         'spelling of the closed chain, reproducing the measured links and the lag-1 '
         'news channel by construction. An open link riding another factor\'s path is the '
         'linked-parent family (`BasisLinkedSpotModel`), not this class.',
         '',
         '- **Link_ID_Sigma, Link_ON_Sigma**: the chain-link residual stds',
         '- **Link_Daily_Sigma**: the source\'s daily step std on the same window',
         '- **Bridge_Premium**: mean offset net of the bridge'])

    factor_types = ('ObservedBasis',)
    fields = [
        F('Link_ID_Sigma', 'Float', default=0.0,
          description='Residual std of the intraday link (source(t) to this(t)), $/oz - the '
                      'bridge weight and residual derive from the two links'),
        F('Link_ON_Sigma', 'Float', default=0.0,
          description='Residual std of the overnight link (this(t) to source(t+1)), $/oz'),
        F('Link_Daily_Sigma', 'Float', default=0.0,
          description='Daily step residual std of the chain source on the same window, $/oz '
                      '- closes the bridge identities at the source\'s own scale'),
        F('Bridge_Premium', 'Float', default=0.0,
          description='Mean offset net of the bridge, $/oz')
    ]

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    @property
    def correlation_name(self):
        return 'ChainedBasisProcess', [()]

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        """The bridge source is the declared `Chained_Basis` — the WHOLE contract, no naming
        convention: a chain can run basis to basis (to basis…) and may or may not close a
        loop; this factor reads only its own declared link. The name resolves to whichever
        factor type carries it in the universe — a basis, or a composable primary for the Log
        bridge — loud when absent, unresolvable or ambiguous."""
        name = (self.factor.param or {}).get('Chained_Basis') if self.factor else None
        if not name:
            raise Exception(f'ChainedBasisModel {utils.check_tuple_name(factor)}: the factor '
                            f'block declares no Chained_Basis - the source is the declared '
                            f'link, never a naming convention')
        tup = utils.check_rate_name(name)
        types = [t for t in ('ObservedBasis',) + utils.BASIS_COMPOSABLE_TYPES
                 if utils.Factor(t, tup) in all_factors]
        if len(types) != 1:
            raise Exception(f'ChainedBasisModel {utils.check_tuple_name(factor)}: Chained_Basis '
                            f'{name!r} must resolve under exactly one factor type in the '
                            f'universe, found {types}')
        self.source_key = utils.Factor(types[0], tup)

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        p = self.param
        self.premium = float(p.get('Bridge_Premium', 0.0))
        sig_id, sig_on = float(p['Link_ID_Sigma']), float(p['Link_ON_Sigma'])
        sig_d = float(p['Link_Daily_Sigma'])
        assert sig_id > 0.0 and sig_on > 0.0 and sig_d > 0.0, \
            f'ChainedBasisModel needs all three link sigmas > 0: {p}'
        self.w = 0.5 + (sig_id * sig_id - sig_on * sig_on) / (2.0 * sig_d * sig_d)
        s2 = sig_id * sig_id - self.w * self.w * sig_d * sig_d
        assert 0.0 < self.w < 1.0 and s2 > 0.0, \
            f'ChainedBasisModel: infeasible link triple (w={self.w:.4f}, sigma^2={s2:.4f}): {p}'
        self.sigma = np.sqrt(s2)
        self.sigma_half = sig_id
        self.b0 = tensor

    def generate(self, shared_mem):
        """Vectorised bridge off the source's finished path; Z is (T, B) outer, (T, B, B2)
        inner and every expression broadcasts. `k` is the last bracketed row: source rows
        beyond this factor's grid keep the true bridge alive to the shared cutoff; past the
        source's own end the forward half draws alone."""
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        T, batch = Z.shape[0], Z.shape[1:]
        inner = Z.ndim == 3
        P = shared_mem.t_Scenario_Buffer.get(self.source_key)
        if P is None:
            raise Exception(
                f'ChainedBasisModel {utils.check_tuple_name(self.factor_key)}: chained source '
                f'{utils.check_tuple_name(self.source_key)} has not generated - the declared '
                f'link must be simulated before this factor')
        k = min(T, P.shape[0] - 1)                       # rows 1..k-1 bracket; k..T-1 are open
        out = torch.empty(Z.shape, device=Z.device, dtype=Z.dtype)
        out[0] = self.b0.unsqueeze(-1).expand(batch) if inner else self.b0.expand(batch)
        if k > 1:
            out[1:k] = (P[1:k] + self.w * (P[2:k + 1] - P[1:k])
                        + self.premium + self.sigma * Z[1:k])
        if k < T:
            out[k:] = P[k:T] + self.premium + self.sigma_half * Z[k:]
        return out


class FixingBridgeModel(StochasticProcess):
    """An intraday fixing bridged between consecutive observations of its parent price — the
    linked-parent family (an OPEN link, so no `Chained_Basis`: the parent's law never reads
    this factor back, which the data confirms — the overnight move is unpredictable from the
    intraday one, slope +0.006, R² 0.0000). The factor is the level basis (fixing − parent)
    whose composed name IS the fixing; the law, in the parent's log space:

        log B(t) = log P(t) + W·(log P(t+1) − log P(t)) + premium + σ_t·Z(t)
        σ_t² = W(1−W)·h_t·f_{t+1}       h_t from the parent's published garch_log_h

    The bracket placement is the point, not a nicety: it is what makes
    Var(P(t+1) | P(t), fixing(t)) = (1−W)·h — the measured 25% variance reduction the data
    carries (sd ratio 0.868, slope 1.006 on the intraday move) — exist in the simulated world
    at all. An independent draw of the same marginal (√(W·h)·ε, the Q-Q-equivalent law) scores
    ZERO on that conditional, and a solver cannot infer structure the world does not contain.

    MEMORYLESS given the parent path (the reversal term died in the data, R² = 0.005): no
    loop, no extra fork seed, no replay recursion — row 0 is the declared Spot, a fork's row 0
    is the value the calc hands `precalculate`, the burn-in rides the generic seam. The last
    bracketed row is decided by parent-bracket availability; past the parent's grid the
    forward intraday half draws alone (W·h·f_bd). Exact only where every step is one sub-step
    — a coarser grid is refused loud, the same bound GARCH replay enforces. The calendar
    grid's weekend convention is the parent's own accepted one and this law inherits it.

    JSON config:
        Bridge_Weight: intraday share of the day's variance (≈ 0.25)
        Bridge_Premium: mean log(fixing/parent) net of the bridge (≈ −6e-4)
        Calibration_DT_Years: the parent recursion's clock (default 1/252)"""

    documentation = (
        'Asset Pricing',
        ['An intraday fixing bridged between consecutive observations of its parent price '
         '(the linked-parent family — an open link, unlike the closed-chain '
         '`ChainedBasisModel`):',
         '',
         '$$ \\log B_t = \\log P_t + W(\\log P_{t+1} - \\log P_t) + \\text{premium} '
         '+ \\sigma_t Z_t, \\quad \\sigma_t^2 = W(1-W)h_t f_{t+1} $$',
         '',
         'published as the level basis whose composed name is the fixing. The bracket '
         'placement makes the measured conditional-variance reduction of the next parent '
         'observation given the fixing exist in the simulated world.',
         '',
         '- **Bridge_Weight**: intraday share of the day\'s variance',
         '- **Bridge_Premium**: mean log offset net of the bridge'])

    factor_types = ('ObservedBasis',)
    fields = [
        F('Bridge_Weight', 'Float', default=0.0,
          description='Intraday share of the day\'s variance - the bridge weight'),
        F('Bridge_Premium', 'Float', default=0.0,
          description='Mean log(fixing/parent) net of the bridge'),
        F('Calibration_DT_Years', 'Float', default=1.0 / 252.0,
          description='Step size (in years) of the parent\'s calibrated recursion')
    ]

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    @property
    def correlation_name(self):
        return 'FixingBridgeProcess', [()]

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        """The parent price: the name minus its last period — the linked-parent family's own
        documented convention (this is the open-link class; a declared link is the chained
        one's contract)."""
        parent = factor.name[:-1]
        types = [t for t in utils.BASIS_COMPOSABLE_TYPES if utils.Factor(t, parent) in all_factors]
        if len(types) != 1:
            raise Exception('FixingBridgeModel {0}: parent {1} must resolve under exactly one '
                            'composable spot type, found {2}'.format(
                                utils.check_tuple_name(factor), '.'.join(parent), types))
        self.linked_key = utils.Factor(types[0], parent)

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        p = self.param
        self.w = float(p['Bridge_Weight'])
        assert 0.0 < self.w < 1.0, f'FixingBridgeModel needs Bridge_Weight in (0,1): {p}'
        self.premium = float(p.get('Bridge_Premium', 0.0))
        tg_years = time_grid.time_grid_years
        dt_arr = np.diff(np.hstack(([tg_years[0]], tg_years)))
        dt_c = float(p.get('Calibration_DT_Years', 1.0 / 252.0))
        if any(len(s) > 1 for s in utils.substep_schedule(dt_arr / dt_c)):
            raise ValueError('FixingBridgeModel: the bridge needs a grid no coarser than the '
                             'trading day - h*f is not the interval variance across sub-steps '
                             '(the same bound GARCH replay enforces)')
        # one flat-tail pad row: a parent outliving this grid brackets the LAST row against a
        # step whose length lives on the parent's grid - held at the last own step (the carry
        # pad's convention)
        f = dt_arr / dt_c
        self.f = shared.one.new_tensor(np.hstack([f, f[-1:]]))
        self.f_bd = (1.0 / utils.DAYS_IN_YEAR) / dt_c
        self.b0 = tensor

    def generate(self, shared_mem):
        """Vectorised bridge off the parent's finished path; Z is (T, B) outer, (T, B, B2)
        inner. `k` is the last bracketed row (parent-bracket availability, not the factor's
        own horizon); the open rows draw the forward intraday half alone."""
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        T, batch = Z.shape[0], Z.shape[1:]
        inner = Z.ndim == 3
        P = shared_mem.t_Scenario_Buffer[self.linked_key]
        h = shared_mem.t_Scenario_Buffer[(self.linked_key, 'garch_log_h')].squeeze(1)[:T].exp()
        k = min(T, P.shape[0] - 1)
        log_p = P[:T].clamp_min(1.0e-30).log()
        out = torch.empty(Z.shape, device=Z.device, dtype=Z.dtype)
        out[0] = self.b0.unsqueeze(-1).expand(batch) if inner else self.b0.expand(batch)
        if k > 1:
            lp_next = P[2:k + 1].clamp_min(1.0e-30).log()
            sd = (self.w * (1.0 - self.w) * h[1:k]
                  * self.f[2:k + 1].view((-1,) + (1,) * len(batch))).sqrt()
            log_b = log_p[1:k] + self.w * (lp_next - log_p[1:k]) + self.premium + sd * Z[1:k]
            out[1:k] = log_b.clamp_min(-10.0).exp() - P[1:k]
        if k < T:
            sd = (self.w * h[k:] * self.f_bd).sqrt()
            log_b = log_p[k:] + self.premium + sd * Z[k:]
            out[k:] = log_b.clamp_min(-10.0).exp() - P[k:T]
        return out


class FixingBridgeCalibration(object):
    """Calibration of FixingBridgeModel: the pooled bracket OLS on 1-CALENDAR-day pairs (a
    weekend bracket has a different clock share; the data reads flat W on clean pairs) —
    slope = Bridge_Weight, intercept = Bridge_Premium; the variance is the model's h·f,
    nothing to stamp. The frame carries the own basis column and the parent price column
    through the existing ObservedBasis prefix pull."""
    model_type = 'FixingBridgeModel'
    fields = []

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        own_col = next(c for c in data_frame.columns if c.split('.', 1)[0] == 'ObservedBasis')
        parent = '.'.join(own_col.split(',', 1)[0].split('.')[1:-1])
        src_col = next((c for c in data_frame.columns if c.split(',', 1)[0] in
                        (f'CommodityPrice.{parent}', f'EquityPrice.{parent}',
                         f'FxRate.{parent}')), None)
        if src_col is None:
            raise ValueError(f'FixingBridgeCalibration: no parent price column for {parent!r} '
                             f'in the frame (have {list(data_frame.columns)})')
        panel = data_frame[[own_col, src_col]].astype(np.float64)
        idx = panel.index
        days = np.asarray((idx - utils.excel_offset).days
                          if isinstance(idx, pd.DatetimeIndex) else idx, dtype=np.int64)
        dd = np.diff(days)
        b, P = panel[own_col].values, panel[src_col].values
        m = (dd == 1) & np.isfinite(b[:-1]) & np.isfinite(P[:-1]) & np.isfinite(P[1:]) \
            & (P[:-1] + b[:-1] > 0)
        y = np.log((P[:-1] + b[:-1]) / P[:-1])[m]
        x = np.log(P[1:] / P[:-1])[m]
        X = np.column_stack([np.ones(y.size), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        prem, w = float(beta[0]), float(beta[1])
        resid = y - X @ beta
        param = {'Bridge_Weight': w, 'Bridge_Premium': prem,
                 'Calibration_DT_Years': 1.0 / float(num_business_days)}
        logging.info('fixing bridge fit: W=%.3f premium=%+.6f (n=%d, resid sd %.5f)',
                     w, prem, m.sum(), resid.std())
        delta = pd.DataFrame({own_col: resid / resid.std()}, index=idx[:-1][m])
        return utils.CalibrationInfo(param, [[1.0]], delta)


class ChainedBasisCalibration(object):
    """Calibration of ChainedBasisModel from the archive panel: own column + the chain source's
    column. The frame's source column is found structurally (the other ObservedBasis column the
    prefix pull delivered — for a session pair the chain source IS the positional parent, so
    the existing pull suffices; a chain whose link is not in the frame raises loud).

    Fits the two chain-link OLS residuals and the source's daily step std — the three measured
    objects from which the model derives the bridge — and the premium net of the derived
    weight. `delta` is the standardized BRIDGE residual (b − P − w·ΔP) on its own dates:
    uncorrelated with the source's step by construction, so an estimated correlation matrix
    cannot double-count the news channel the bridge already carries. `correlation` [[1.0]],
    `num_factors` 1."""
    model_type = 'ChainedBasisModel'
    fields = [
        F('Max_Session_Gap_Days', 'Integer', default=4,
          description='Largest calendar gap, in days, an overnight link pair may span')]

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        own_col = next(c for c in data_frame.columns if c.split('.', 1)[0] == 'ObservedBasis')
        src_col = next((c for c in data_frame.columns
                        if c != own_col and c.split('.', 1)[0] == 'ObservedBasis'), None)
        if src_col is None:
            raise ValueError(f'ChainedBasisCalibration: no chain-source basis column in the '
                             f'frame (have {list(data_frame.columns)})')

        panel = data_frame[[own_col, src_col]].astype(np.float64)
        idx = panel.index
        days = np.asarray((idx - utils.excel_offset).days
                          if isinstance(idx, pd.DatetimeIndex) else idx, dtype=np.int64)
        dd = np.diff(days)
        max_gap = int(self.param.get('Max_Session_Gap_Days', 4))
        b, P = panel[own_col].values, panel[src_col].values

        def ols(X, y):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return beta, y - X @ beta

        m_id = np.isfinite(b) & np.isfinite(P)
        _, r_id = ols(np.column_stack([np.ones(m_id.sum()), P[m_id]]), b[m_id])
        m_on = (dd <= max_gap) & np.isfinite(b[:-1]) & np.isfinite(P[1:])
        _, r_on = ols(np.column_stack([np.ones(m_on.sum()), b[:-1][m_on]]), P[1:][m_on])
        sig_id, sig_on = float(r_id.std()), float(r_on.std())
        m_br = (dd <= max_gap) & np.isfinite(b[:-1]) & np.isfinite(P[:-1]) & np.isfinite(P[1:])
        step = (P[1:] - P[:-1])[m_br]
        sig_d = float(step.std())
        w = 0.5 + (sig_id ** 2 - sig_on ** 2) / (2.0 * sig_d ** 2)
        bridge = (b[:-1] - P[:-1] - w * (P[1:] - P[:-1]))[m_br]
        prem = float(bridge.mean())
        param = {'Link_ID_Sigma': sig_id, 'Link_ON_Sigma': sig_on, 'Link_Daily_Sigma': sig_d,
                 'Bridge_Premium': prem}
        logging.info('chained bridge fit: ID $%.2f ON $%.2f daily $%.2f -> w=%.3f '
                     'premium=%+.3f (n_id=%d n_on=%d)', sig_id, sig_on, sig_d, w, prem,
                     m_id.sum(), m_on.sum())
        delta = pd.DataFrame({own_col: (bridge - prem) / bridge.std()}, index=idx[:-1][m_br])
        return utils.CalibrationInfo(param, [[1.0]], delta)


class BasisLinkedSpotModel(StochasticProcess):
    """Lagged-AR(1) basis driven by a sibling commodity-spot path and its HMM regime:

        b(t) = μ_t + a · ΔS(t) + φ · (b(t-1) − μ_t) + η(t)
        η(t) = σ_t · √((ν-2)/ν) · ε_t,    ε_t ~ t_ν

    ΔS is the linked spot's per-step diff and σ_t the innovation std — regime-keyed σ(s_t) off
    the linked spot's HMM state, or flat, or its own GARCH (below). Innovation is built from a
    framework-correlated Gaussian Z plus an internal Chi²(ν) draw; the √((ν-2)/ν) rescaling
    makes σ_t the realised std of η regardless of ν. The linked spot's path and regime path are
    read from `shared_mem.t_Scenario_Buffer`; the linked parent is this factor's own name minus
    its last period, and sim ordering is enforced by the name-prefix chain (parent -> basis).
    Initial b(0) is taken from the factor's `Spot` value.

    μ_t and a time-varying σ_t are the two OPTIONAL extensions, both OFF at their 0.0 defaults,
    both deterministic recursions on the realised path (the `GARCHSpotModel` observable-`h`
    idiom) — so neither consumes any randomness and the seeded draw order is the same on or off:

        μ_{t+1} = λ·μ_t + (1−λ)·b(t)                  `Slow_Mean_Lambda` λ, seeded by `Mu_0`
        σ_t² = ω_b + α_b·η(t−1)² + β_b·σ_{t−1}²       `G_Omega`/`G_Alpha`/`G_Beta`, seed `Sig2_0`

    λ = 0 is NOT the shipped model — it would make μ_t = b(t−1) — so the mean switch is
    STRUCTURAL rather than arithmetic: at the default the mean term is absent from the
    expression and the loop re-executes `b(t) = a·ΔS(t) + φ·b(t-1) + η(t)` in the shipped
    order, bitwise.

    Innovation precedence is `Sigma_By_State` > GARCH > flat `Sigma`. The regime branch is
    untouched and still wins, so GARCH fields declared beside `Sigma_By_State` are inert; against
    a flat `Sigma`, ω_b > 0 replaces it (and `Sigma` stays the value the model falls back to when
    the GARCH fields are deleted). `Sigma_By_State` and `Sigma` remain mutually exclusive.

    Both seeds follow the `H0` pattern — `Mu_0` is the mean the FIRST simulated step reverts to,
    `Sig2_0` the variance of the FIRST innovation, each stamped by the calibration from the end
    of the sample. Both recursions are per-path state and fork with the path: `(key,'basis_mu')`
    and `(key,'basis_sig2')` publish `state[t]` = what the step t→t+1 consumes, which is exactly
    what `inner_fork_seed` hands an inner run as ITS t=0.

    `Calibration_DT_Years` is declared metadata: the AR is per calibration STEP and nothing
    rescales it to the sim grid (the walk-forward driver converts Phi with it externally); the
    dead `Mu` field it used to sit beside is gone. `reseed_from_path` runs both recursions along an
    observed path — η_t recovered as b_t minus the conditional mean the same `_advance` computes —
    so an `Observed_Scenario` replay publishes the REPLAYED path's state, not the discarded one's.

    JSON config:
        A: concurrent ΔS loading
        Phi: AR(1) coefficient on b(t-1)
        Nu: Student-t degrees of freedom (shared across regimes)
        Sigma_By_State: list of σ_s indexed by linked-spot HMM state
        Sigma: flat innovation std (the alternative to Sigma_By_State)
        Slow_Mean_Lambda, Mu_0: the slow observable mean (0.0 = off)
        G_Omega, G_Alpha, G_Beta, Sig2_0: the own-GARCH innovation vol (0.0 = off)
        Calibration_DT_Years: float (default 1/252)"""

    documentation = (
        'Asset Pricing',
        ['A lagged-AR(1) basis driven by a sibling commodity-spot path and its HMM regime.',
         '',
         '$$ b(t) = a \\Delta S(t) + \\phi b(t-1) + \\eta(t),'
         '\\quad \\eta(t) = \\sigma(s_t)\\sqrt{(\\nu-2)/\\nu}\\,\\varepsilon_t,'
         '\\quad \\varepsilon_t \\sim t_\\nu $$',
         '',
         'Reads the linked spot path and its HMM regime path from the simulator shared '
         'buffer; the linked spot is simulated first (enforced via `dependant_fields`).',
         '',
         'Two optional extensions ride the same draws, each a deterministic recursion on the '
         'realised path and each off at its 0.0 default: a slow observable mean the AR reverts '
         'to, $\\mu_{t+1} = \\lambda\\mu_t + (1-\\lambda)b(t)$, and the basis\'s own GARCH(1,1) '
         'innovation variance $\\sigma_t^2 = \\omega_b + \\alpha_b\\eta_{t-1}^2 + '
         '\\beta_b\\sigma_{t-1}^2$. Innovation precedence is Sigma_By_State > GARCH > Sigma.',
         '',
         '- **A**: concurrent ΔS loading',
         '- **Phi**: AR(1) coefficient',
         '- **Nu**: Student-t degrees of freedom (shared across regimes)',
         '- **Sigma_By_State**: per-regime innovation std',
         '- **Slow_Mean_Lambda, Mu_0**: decay and seed of the slow observable mean',
         '- **G_Omega, G_Alpha, G_Beta, Sig2_0**: the own-GARCH innovation variance and its seed'])

    factor_types = ('ObservedBasis',)
    fields = [
        F('A', 'Float', default=0.0, description='Concurrent ΔS loading on the basis'),
        F('Phi', 'Float', default=0.0, description='AR(1) coefficient on the previous basis'),
        F('Nu', 'Float', default=5.0,
          description='Student-t degrees of freedom (basis innovation)'),
        F('Sigma_By_State', 'Container', default=[],
          description='Per-regime innovation std σ_s (indexed by linked-spot HMM state) - the '
                      'alternative to a flat Sigma'),
        F('Sigma', 'Float', default=0.0,
          description='Flat innovation std for a regime-free primary - the alternative to '
                      'Sigma_By_State'),
        F('Slow_Mean_Lambda', 'Float', default=0.0,
          description='Per-step decay of the slow observable mean the AR reverts to - 0 turns '
                      'the recursion off and the AR reverts to zero, as it always has'),
        F('Mu_0', 'Float', default=0.0,
          description='Mean the first simulated step reverts to - the slow-mean recursion\'s '
                      'seed, stamped from the end of the calibration sample'),
        F('G_Omega', 'Float', default=0.0,
          description='Variance intercept of the innovation\'s own GARCH(1,1) - > 0 turns it on '
                      'and it replaces the flat Sigma'),
        F('G_Alpha', 'Float', default=0.0,
          description='Weight on the last squared innovation'),
        F('G_Beta', 'Float', default=0.0,
          description='Weight on the last innovation variance'),
        F('Sig2_0', 'Float', default=0.0,
          description='Variance of the first simulated innovation - the GARCH recursion\'s seed, '
                      'stamped from the end of the calibration sample'),
        F('Calibration_DT_Years', 'Float', default=1.0 / 252.0,
          description='Step size (in years) of the calibrated AR(1)'),
        F('Reversion_Model', 'Text', default='Linear', values=['Linear', 'Band_Mixture'],
          description='Linear = the AR(1) deviation (the default, everything above). '
                      'Band_Mixture = dead-zone reversion around the slow level with 2-state '
                      'Markov mixture innovations - the Q-Q-ruled law for a physically '
                      'arbitraged spread'),
        F('Band_Kappa', 'Float', default=0.0,
          description='Half-width of the no-arbitrage dead zone around the slow level - '
                      'inside it the deviation feels no pull at all'),
        F('Band_Beta', 'Float', default=0.0,
          description='Per-step pull on the deviation beyond the band edge'),
        F('Mix_Q_Stress', 'Float', default=0.0,
          description='Stationary probability of the stress innovation state'),
        F('Mix_Sigma_Quiet', 'Float', default=0.0,
          description='Innovation std in the quiet state'),
        F('Mix_Sigma_Stress', 'Float', default=0.0,
          description='Innovation std in the stress state'),
        F('Mix_Stay_Stress', 'Float', default=0.0,
          description='P(stress tomorrow | stress today) - quiet persistence follows from '
                      'stationarity'),
        F('Mix_P0_Stress', 'Float', default=0.0,
          description='P(stress) at the first simulated step - the filtered posterior at the '
                      'end of the calibration sample')
    ]

    def __init__(self, factor, param, implied_factor=None):
        super().__init__(factor, param)

    @staticmethod
    def num_factors():
        return 1

    @property
    def correlation_name(self):
        return 'BasisLinkedSpotProcess', [()]

    def calc_references(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        """
        Resolve `linked_key`, the primary spot factor this basis rides.

        The linked parent is the name minus its last period (positional, like the InterestRate
        parent chain): ObservedBasis if that parent is itself a basis, else the one composable spot
        type it resolves under — loud if not exactly one. It is resolved here rather than in
        `generate` because the type needs all_factors; the graph stays acyclic (parent -> basis).
        """
        parent = factor.name[:-1]
        if len(parent) > 1:
            self.linked_key = utils.Factor('ObservedBasis', parent)
        else:
            types = [t for t in utils.BASIS_COMPOSABLE_TYPES if utils.Factor(t, parent) in all_factors]
            if len(types) != 1:
                raise Exception('ObservedBasis {0}: parent {1} must resolve under exactly one composable '
                                'spot type, found {2}'.format(utils.check_tuple_name(factor), '.'.join(parent), types))
            self.linked_key = utils.Factor(types[0], parent)

    def precalculate(self, ref_date, time_grid, tensor, shared, process_ofs, implied_tensor=None):
        """
        Precompute the OU basis parameters and the observed initial basis.

        Two innovation forms, chosen by whichever key the JSON block carries (exactly one):
            Sigma_By_State — regime-conditional σ_s, indexed by the primary's HMM regime path;
            Sigma          — flat single-vol OU, no regime read (for a regime-free primary, e.g.
                             the GARCH martingale primary).

        `self.garch` is the third form and the whole of the documented precedence: it can only be
        selected against a flat Sigma, so a regime-switching primary keeps today's behaviour even
        with the GARCH fields declared. `self.slow_mean` is likewise the mean term's structural
        switch — λ = 0 means "no mean term", not "λ = 0 in the recursion".

        AAD: `b0` is kept on the autograd graph so payoff sensitivities w.r.t. the observed initial
        basis flow through, and is stored unreshaped so inner-MC mode can pass a `(B,)` vector of
        per-outer-path initial bases; outer mode is `(1,)`.
        """
        self.z_offset = process_ofs
        self.scenario_horizon = time_grid.scen_time_grid.size
        self.A = float(self.param['A'])
        self.Phi = float(self.param['Phi'])
        self.Nu = float(self.param['Nu'])
        # Exactly one of Sigma_By_State / Sigma must be present.
        has_flat, has_regime = ('Sigma' in self.param), ('Sigma_By_State' in self.param)
        assert has_flat != has_regime, \
            f"BasisLinkedSpotModel needs exactly one of 'Sigma' / 'Sigma_By_State': {self.param}"
        self.sigma_by_state = (shared.one.new_tensor(np.array(self.param['Sigma_By_State'], dtype=np.float64))
                               if has_regime else None)
        self.sigma_flat = None if has_regime else shared.one.new_tensor(float(self.param['Sigma']))
        self.lam = float(self.param.get('Slow_Mean_Lambda', 0.0))
        self.g_omega = float(self.param.get('G_Omega', 0.0))
        self.g_alpha = float(self.param.get('G_Alpha', 0.0))
        self.g_beta = float(self.param.get('G_Beta', 0.0))
        self.slow_mean = self.lam != 0.0
        self.garch = self.sigma_by_state is None and self.g_omega > 0.0
        self.mu0 = shared.one.new_tensor(float(self.param.get('Mu_0', 0.0)))
        self.sig20 = shared.one.new_tensor(float(self.param.get('Sig2_0', 0.0)))
        # Band+mixture reversion (the Q-Q-ruled law): a dead zone of width Band_Kappa around the
        # slow level with pull Band_Beta outside it, and Gaussian innovations whose scale is a
        # 2-state Markov mixture (quiet/stress) - the mixture supplies the tails, so no t-scale
        # and no GARCH recursion. Requires the slow mean (the band is a deviation from it).
        self.band = str(self.param.get('Reversion_Model', 'Linear')) == 'Band_Mixture'
        if self.band:
            assert self.slow_mean, 'Band_Mixture reversion needs Slow_Mean_Lambda > 0'
            # fail loud, not silently inert: the mixture IS the innovation under the band, so a
            # declared regime vector would be ignored - the same either/or the Sigma pair gets
            assert self.sigma_by_state is None, \
                'Band_Mixture and Sigma_By_State are mutually exclusive (the mixture is the ' \
                'innovation law); declare a flat Sigma on a band basis'
            self.garch = False
            self.band_kappa = float(self.param['Band_Kappa'])
            self.band_beta = float(self.param['Band_Beta'])
            q, stay = float(self.param['Mix_Q_Stress']), float(self.param['Mix_Stay_Stress'])
            self.mix_sig2 = shared.one.new_tensor(
                [float(self.param['Mix_Sigma_Quiet']) ** 2, float(self.param['Mix_Sigma_Stress']) ** 2])
            self.mix_stay_s = stay
            self.mix_enter = q * (1.0 - stay) / (1.0 - q)          # quiet->stress, stationarity
            self.mix_q, self.p0_stress = q, float(self.param.get('Mix_P0_Stress', q))
        # AAD: b0 stays on the graph, unreshaped so inner MC can pass (B,).
        self.b0 = tensor

    def generate(self, shared_mem):
        """
        Simulate the OU basis on top of the linked primary spot path.

        The linked spot must have been generated first: the `dependant_fields` declaration on
        ObservedBasis makes CommodityPrice a dependency, so the simulator topo-orders it before
        us. The linked spot path is read in *price level* (dollars), not log-space — the HMM
        process exp()s its log-cumsum before publishing — and its path/regime shapes match this
        process's Z, since both processes ran in the same inner/outer mode.

        ONE loop covers outer (T, B) and inner (T, B, B2) — every expression in it broadcasts, so
        only `b_init` (a `(1,)` spot against a `(B,)` per-outer-path fork vector) needs the mode.
        The two extensions are the two `if`s inside it, and with both off the else arms are the
        shipped expression in the shipped order: nothing new is evaluated, not even at zero.

        Neither recursion draws: `W` is sampled once, in the shape and order it always was, and
        the GARCH arm reuses the SAME `Z`/`W` element it would have used flat — it only rescales
        it, so every seeded number in a world with the extensions off is untouched.
        """
        # Z is (T, B) outer / (T, B, B2) inner, correlated.
        Z = shared_mem.t_random_numbers[self.z_offset, :self.scenario_horizon]
        device = Z.device
        dtype = Z.dtype

        # Cross-process read; the linked spot is generated before us.
        linked_path = shared_mem.t_Scenario_Buffer[self.linked_key]
        # Device-side: a python `assert` on this would truth-test the tensor, i.e. one full
        # pipeline sync per generate() call (1125 per wf-gate solve). The invariant stays.
        torch._assert_async((linked_path > 0).all(), 'linked_path expected to be all positive')

        if self.sigma_by_state is not None:
            regimes = shared_mem.t_Scenario_Buffer.get((self.linked_key, 'regimes'))
            if regimes is None:
                raise KeyError(
                    f"{utils.check_tuple_name(self.factor_key)} uses regime-conditional "
                    f"Sigma_By_State but its primary {utils.check_tuple_name(self.linked_key)} "
                    f"publishes no regimes — give this basis a flat 'Sigma' (single-vol OU) or "
                    f"pair it with a regime-switching primary.")
            sigma_t = self.sigma_by_state[regimes]
        else:
            sigma_t = self.sigma_flat
        # Student-t innovation: η_t = sigma_t · ε_t · √((ν-2)/ν), ε_t ~ t_ν.
        # Identity: ε_t = Z · √(ν/W) where W ~ Chi²(ν). Combine the rescaling so the
        # marginal variance of η_t is sigma_t² regardless of ν.
        nu = self.Nu
        a = self.A

        inner = Z.ndim == 3
        T, batch = Z.shape[0], Z.shape[1:]
        # Floor the chi-square draw — same inf/NaN guard as the linked spot.
        W = torch.distributions.Chi2(shared_mem.one.new_tensor(nu)).sample(Z.shape).clamp_min(1.0e-6)
        scale = torch.sqrt((nu - 2.0) / W)
        eta = None if self.garch or self.band else sigma_t * Z * scale
        out = torch.empty(Z.shape, device=device, dtype=dtype)
        out[0] = self.b0.unsqueeze(-1).expand(batch) if inner else self.b0.expand(batch)
        # Per-path recursion state, seeded from the inner fork / burn-in when either published one.
        mu, sig2, mu_path, sig2_path = self._recursion_state(shared_mem, out, inner)

        s = regime_path = u_regime = None
        if self.band:
            # Regime uniforms off the independent quasi stream — the HMM's own convention — so
            # the mixture switch never consumes (or correlates with) the Gaussian innovation.
            if inner:
                u_regime = shared_mem.quasi_rng(T + 1, batch[0] * batch[1])[1].transpose(
                    0, 1).contiguous().reshape(T + 1, *batch)
            else:
                u_regime = shared_mem.quasi_rng(T + 1, batch[0])[1].transpose(0, 1).contiguous()
            seed = shared_mem.t_Scenario_Buffer.get(
                (self.factor_key, 'regime0' + ('_inner' if inner else '_outer')))
            if seed is None:
                s = (u_regime[0] < self.p0_stress).to(dtype)
            else:
                s = (seed.unsqueeze(-1).expand(batch) if inner else seed).to(dtype)
            regime_path = torch.empty(Z.shape, device=device, dtype=dtype)
            regime_path[0] = s

        def drawn(mean, s2):
            """Forward sim: η is DRAWN — regime-scaled Gaussian under the band mixture, rescaled
            by the live σ² under GARCH, else already scaled — and the level follows from it."""
            if self.band:
                eta_t = self.mix_sig2[s.long()].sqrt() * Z[t]
            else:
                eta_t = s2.sqrt() * Z[t] * scale[t] if self.garch else eta[t]
            return eta_t, mean + eta_t

        for t in range(1, T):
            if self.band:
                s = torch.where(s > 0.5, (u_regime[t] < self.mix_stay_s).to(dtype),
                                (u_regime[t] < self.mix_enter).to(dtype))
                regime_path[t] = s
            out[t], mu, sig2 = self._advance(
                mu, sig2, out[t - 1], a * (linked_path[t] - linked_path[t - 1]), drawn)
            if self.slow_mean:
                mu_path[t] = mu
            if self.garch:
                sig2_path[t] = sig2
        self._publish_recursions(shared_mem, mu_path, sig2_path, regime_path)
        return out

    def _advance(self, mu, sig2, prev, ds, innovation):
        """ONE step of the coupled (b, μ, σ²) recursion, shared by the forward sim and the
        observed-path replay so the forward ≡ replay invariant is STRUCTURAL rather than maintained
        by copying — `GARCHSpotModel._advance_variance`'s construction. The only thing that differs
        between the two is `innovation(mean, sig2) -> (η_t, b_t)`: the forward sim DRAWS η and
        derives the level, while replay is handed the level and derives η as its deviation from
        this same conditional mean. Returning both is what makes the replayed μ exact — a replay
        that rebuilt `b` as `mean + (b − mean)` would be one rounding off the observed level.

        Returns `(b_t, μ_{t+1}, σ²_{t+1})`, both states in the published fork-index convention: what
        the step t→t+1 consumes. Each extension's OFF arm returns its state untouched, so with both
        off the two expressions evaluated are the shipped ones, in the shipped order."""
        if self.band:
            d = prev - mu
            mean = prev + ds - self.band_beta * torch.sign(d) * (d.abs() - self.band_kappa).clamp_min(0.0)
        else:
            mean = mu + ds + self.Phi * (prev - mu) if self.slow_mean else ds + self.Phi * prev
        eta, b = innovation(mean, sig2)
        return (b,
                self.lam * mu + (1.0 - self.lam) * b if self.slow_mean else mu,
                self.g_omega + self.g_alpha * eta * eta + self.g_beta * sig2 if self.garch else sig2)

    def _recursion_state(self, shared_mem, out, inner):
        """`(μ_0, σ²_0, μ path, σ² path)` at t=0: the seeds (fork / burn-in / calibrated) and the
        per-path buffers each live recursion publishes — `None` for one that is off, which is what
        keeps an OFF world from allocating a `(T, B)` array nothing reads."""
        mu = self._recursion_seed(shared_mem, 'mu0', self.mu0, inner)
        sig2 = self._recursion_seed(shared_mem, 'sig20', self.sig20, inner)
        paths = [None, None]
        for i, (on, seed) in enumerate(((self.slow_mean, mu), (self.garch, sig2))):
            if on:
                paths[i] = torch.empty_like(out)
                paths[i][0] = seed
        return (mu, sig2, *paths)

    def _publish_recursions(self, shared_mem, mu_path, sig2_path, regime_path=None):
        """Publish whichever recursions ran, and stash them for `outer_reseed` — so a replayed run's
        terminal state is the REPLAYED path's, not the discarded simulated one's."""
        self.last_mu, self.last_sig2, self.last_regime = mu_path, sig2_path, regime_path
        for kind, path in (('basis_mu', mu_path), ('basis_sig2', sig2_path),
                           ('basis_regime', regime_path)):
            if path is not None:
                shared_mem.t_Scenario_Buffer[(self.factor_key, kind)] = path

    def _recursion_seed(self, shared_mem, kind, calibrated, inner):
        """t=0 state for one observable recursion: the inner fork's `<kind>_inner` (a `(B,)`
        per-outer-path vector, unsqueezed so it broadcasts across the B2 fan-out), or the diff-ML
        burn-in's `<kind>_outer`, else the calibrated scalar. Mirrors GARCH's `h0_inner`/
        `h0_outer` pair; a scalar broadcasts against either mode, so nothing is expanded."""
        seed = shared_mem.t_Scenario_Buffer.get(
            (self.factor_key, kind + ('_inner' if inner else '_outer')))
        return calibrated if seed is None else (seed.unsqueeze(-1) if inner else seed)

    def inner_fork_seed(self, factor_key, outer_buf, t):
        """Per-outer-path seeds for the two observable recursions, read at the fork row so the
        inner fan-out continues the forked path's mean / innovation-variance state instead of
        restarting from the calibrated `Mu_0`/`Sig2_0`. Both published arrays hold the state the
        step t→t+1 consumes, which IS the inner run's step 0→1 — so the fork is an index read,
        with no re-derivation to keep in step with `generate`. Detached: the inner tape
        differentiates back to its own `state_t` leaves, never through the outer path."""
        return {(factor_key, seed): outer_buf[(factor_key, kind)][t].detach()
                for kind, seed in (('basis_mu', 'mu0_inner'), ('basis_sig2', 'sig20_inner'),
                                   ('basis_regime', 'regime0_inner'))
                if (factor_key, kind) in outer_buf}

    def outer_reseed(self):
        """t=0 seeds for the next outer run's burn-in: this run's terminal mean / variance state,
        so a burn-in that carries the terminal basis LEVEL over carries the state that level was
        generated under with it. Nothing to seed when both recursions are off."""
        return {(self.factor_key, seed): path[-1].detach()
                for seed, path in (('mu0_outer', self.last_mu), ('sig20_outer', self.last_sig2),
                                   ('regime0_outer', self.last_regime))
                if path is not None}

    def reseed_from_path(self, simulated, shared_mem):
        """Observed-path replay: rerun both observable recursions ALONG the supplied basis path and
        republish `basis_mu` / `basis_sig2`, so a fork or a reveal at any row reads the REPLAYED
        path's state instead of the discarded simulated one's.

        Both are pure functions of the realised path. μ_t needs only b; σ_t² needs η_t, which IS
        b_t minus the model's own conditional mean at t — so the replay is `_advance` again with the
        innovation handed in rather than drawn, and no second copy of the arithmetic exists to drift
        from. ΔS is read from the buffer, which is the REPLAYED linked path when the world replays
        the parent too: the calc publishes each factor's path before the next process generates.

        Exactness is asymmetric, and MEASURED: replaying a path a seeded run produced reproduces
        that run's μ BITWISE — μ is a function of the observed level, which replay is handed rather
        than rebuilding — but its σ² only to 1e-15 relative in float64 and 6e-7 in float32, because
        σ² is a function of η and `fl(b_t − mean)` differs from the η that was drawn by the rounding
        error of the forward pass's own final addition. No spelling of this replay recovers η
        exactly from a rounded sum; what it can do is not compound the error, which is why
        `_advance` takes the level from the innovation supplier instead of re-deriving it."""
        if not (self.slow_mean or self.garch):
            return
        obs = simulated.detach()
        linked_path = shared_mem.t_Scenario_Buffer[self.linked_key]
        inner = obs.ndim == 3
        mu, sig2, mu_path, sig2_path = self._recursion_state(shared_mem, obs, inner)

        regime_path = None
        if self.band:
            # The regime is UNOBSERVED on a realised path: forward-filter P(stress | η_1..t) with
            # the mixture likelihoods and DRAW the published state off the quasi stream — the same
            # source generate uses, so the replayed regime is reproducible under the seed.
            T, batch = obs.shape[0], obs.shape[1:]
            if inner:
                u = shared_mem.quasi_rng(T + 1, batch[0] * batch[1])[1].transpose(
                    0, 1).contiguous().reshape(T + 1, *batch)
            else:
                u = shared_mem.quasi_rng(T + 1, batch[0])[1].transpose(0, 1).contiguous()
            seed = shared_mem.t_Scenario_Buffer.get(
                (self.factor_key, 'regime0' + ('_inner' if inner else '_outer')))
            p = obs.new_full(batch, self.p0_stress) if seed is None else \
                (seed.unsqueeze(-1).expand(batch) if inner else seed).to(obs.dtype)
            regime_path = torch.empty_like(obs)
            regime_path[0] = (u[0] < p).to(obs.dtype)
            sq2, ss2 = self.mix_sig2[0], self.mix_sig2[1]
            last_eta = [None]

        def realised(mean, s2):
            """Replay: the level is DATA and η is its deviation from the conditional mean."""
            eta_t = obs[t] - mean
            if self.band:
                last_eta[0] = eta_t
            return eta_t, obs[t]

        for t in range(1, obs.shape[0]):
            _, mu, sig2 = self._advance(
                mu, sig2, obs[t - 1], self.A * (linked_path[t] - linked_path[t - 1]), realised)
            if self.slow_mean:
                mu_path[t] = mu
            if self.garch:
                sig2_path[t] = sig2
            if self.band:
                eta2 = last_eta[0] * last_eta[0]
                prior = p * self.mix_stay_s + (1.0 - p) * self.mix_enter
                lq = torch.exp(-0.5 * eta2 / sq2) / sq2.sqrt()
                ls = torch.exp(-0.5 * eta2 / ss2) / ss2.sqrt()
                p = prior * ls / (prior * ls + (1.0 - prior) * lq)
                regime_path[t] = (u[t] < p).to(obs.dtype)
        self._publish_recursions(shared_mem, mu_path, sig2_path, regime_path)


class BasisLinkedSpotCalibration(object):
    """Calibration of BasisLinkedSpotModel. Self-contained: data_frame carries the basis
    column (`ObservedBasis.<primary>.<basis>`) plus the linked spot column, pulled by the
    archive-side name-prefix dependency. OLS on `b(t) = a·ΔS + φ·b(t-1) + η(t)`
    recovers (a, φ); ν from method-of-moments on the η excess kurt; per-regime σ from
    rolling-vol-tercile partitioning of η — terciles indexed in σ-ascending order to
    match the linked spot's HMM regime convention.

    That default path is FROZEN: it is what every calibrated platinum world in the repo carries,
    and `tests/test_basis_slow_mean_garch.py` holds it to the estimator above byte-for-byte, on the
    real archive and on a synthetic one. Declaring either switch below moves the whole system onto
    ONE likelihood instead.

    `Slow_Mean_Span > 0` regresses the DEVIATION from the slow observable mean, `b(t) − μ_t = a·ΔS
    + φ·(b(t-1) − μ_t) + η(t)`, with μ_t the span's EWMA of the basis through t−1 — strictly
    lagged, so the regressor is F_{t-1}-measurable exactly as it is in the simulator. It stamps
    `Slow_Mean_Lambda = 1 − 2/(span+1)` and `Mu_0`, the mean the NEXT observation would revert to.

    `GARCH_Innovation = 'Yes'` fits the innovation's own GARCH(1,1) and stamps
    `G_Omega`/`G_Alpha`/`G_Beta`/`Sig2_0`. It stamps a flat `Sigma` INSTEAD of `Sigma_By_State`,
    because the model's precedence puts the regime form first: stamping both would leave the GARCH
    inert. `Sigma` is then η's unconditional std — the model's behaviour if the four GARCH fields
    are deleted.

    ONE LIKELIHOOD, and what it cost to find out. Either switch routes the fit to `arx1_t_mle` with
    the EWMA handed in as `mean` and `garch` on, so (a, φ, ω_b, α_b, β_b, ν) are estimated together
    against the same Student-t. The path it replaces fitted the conditional mean by OLS — a Gaussian
    loss — and then a GARCH-t on that fit's residual, which is two halves under two likelihoods:
    the defect `QuadraticCarryCurveCalibration` names one factor over, and the reason a scratch
    study and this class reported different numbers from the same archive. Measured on
    `data/plat_archive_sync.csv` (3786 rows, 2010-2026, span 63):

        estimator                       a         φ       ν      ω_b     α_b     β_b
        OLS + GARCH-t on the residual   -0.0461   0.7593  5.01   0.1068  0.1183  0.8714
        joint, this class               -0.0227   0.6505  4.94   0.0793  0.1097  0.8830
        joint, with a held at 0         0         0.6433  5.06   0.0687  0.1016  0.8925

    THE IDENTIFIABILITY FINDING, and it is the carry curve's Γ-versus-ρ one factor over: `A` and the
    spot/basis entry of the framework's R matrix are the same coupling written twice. The one-step
    covariance is Cov(b_t − E_{t-1}[b_t], ΔS) = A·Var(ΔS) + Cov(η, ΔS), one equation in two
    unknowns, and the LOSS decides where on that line the fit lands: OLS is orthogonal to ΔS by
    construction, so it puts all −26.51 of the coupling in A and leaves the R channel exactly zero;
    the t likelihood downweights the tail days where the coupling is largest, halves A, and leaves
    −13.29 of −26.36 for the innovation. The totals differ by 0.6%. Neither is wrong and the split
    does not matter — what matters is that BOTH ends come from the same fit, because `delta` is what
    the framework consolidates into R while `A` is stamped in the block. Take one estimator's A
    beside another's residual and the simulated coupling is 150% of the observed one.

    `Nu` is the joint fit's, not a moment match: the ν ladder 3.0 (no GARCH) → 4.94 (GARCH) says the
    extreme tails were mostly unmodelled vol dynamics, the same reading `GARCHSpotCalibration` gets.
    `Sig2_0` is `garch11_t_mle`'s convention — the conditional variance OF the last observation,
    one step stale as a "today's state" stamp, shared with `GARCHSpotCalibration`'s `H0` so the two
    seeds mean the same thing (advancing one more step would read 17.25 against the stamped
    19.36; moving one without the other is what this file exists to prevent)."""
    model_type = 'BasisLinkedSpotModel'
    fields = [
        F('Nu_Min', 'Float', default=3.0,
          description='Floor on the degrees of freedom - the moment-matched solve at the defaults, '
                      'the joint likelihood once either switch below is declared'),
        F('Nu_Max', 'Float', default=50.0, description='Ceiling on the same solve'),
        F('Vol_Window', 'Integer', default=21,
          description='Rolling window, in business days, the regime terciles are cut on'),
        F('Max_Persistence', 'Float', default=0.999,
          description='Cap on G_Alpha + G_Beta of the basis GARCH - an escalating-variance '
                      'window can drive the MLE past a unit root, and an explosive basis '
                      'random-walks the simulation; beta is scaled down to the cap, matching '
                      'the spot calibration'),
        F('Slow_Mean_Span', 'Integer', default=0,
          description='EWMA span, in business days, of the slow observable mean the AR reverts '
                      'to - 0 fits the shipped zero-mean AR and stamps neither new field'),
        F('GARCH_Innovation', 'Text', default='No', values=['Yes', 'No'],
          description='Yes fits the innovation\'s own GARCH(1,1)-t and stamps a flat Sigma in '
                      'place of Sigma_By_State, which would otherwise take precedence over it'),
        F('Reversion_Model', 'Text', default='Linear', values=['Linear', 'Band_Mixture'],
          description='Band_Mixture fits dead-band reversion around the slow level with 2-state '
                      'Markov mixture innovations (alternating least squares + EM) in place of '
                      'the AR(1)/GARCH path - the Q-Q-ruled law for a physically arbitraged '
                      'spread. Needs Slow_Mean_Span > 0')
    ]

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.num_factors = 1

    def calibrate(self, data_frame, vol_shift, num_business_days=252.0):
        from scipy import stats as scipy_stats

        nu_min = float(self.param.get('Nu_Min', 3.0))
        nu_max = float(self.param.get('Nu_Max', 50.0))
        vol_window = int(self.param.get('Vol_Window', 21))
        span = int(self.param.get('Slow_Mean_Span', 0))
        fit_garch = self.param.get('GARCH_Innovation', 'No') == 'Yes'
        dt_calib = 1.0 / float(num_business_days)

        # Basis col is `ObservedBasis.<primary>.<basis>`; the linked parent is the name minus the
        # last period (CommodityPrice.<parent> at depth 2, else the parent ObservedBasis chain).
        basis_col = next(c for c in data_frame.columns if c.split('.', 1)[0] == 'ObservedBasis')
        parent = '.'.join(basis_col.split(',', 1)[0].split('.')[1:-1])
        linked_col = next((c for c in data_frame.columns
                           if c.split(',', 1)[0] in (f'CommodityPrice.{parent}', f'ObservedBasis.{parent}')), None)
        if linked_col is None:
            raise ValueError(
                f'BasisLinkedSpotCalibration: no linked-spot column for parent {parent!r} '
                f'in data_frame (have {list(data_frame.columns)}). The framework should '
                f'have auto-pulled it from the {basis_col!r} name prefix.')

        joint = data_frame[[basis_col, linked_col]].astype(np.float64).dropna()
        b = joint[basis_col].values
        lme_v = joint[linked_col].values
        dlme = np.diff(lme_v)
        # The slow mean is a level SHIFT of both sides: the deviation is regressed against a moving
        # rather than a zero mean. `ewm(adjust=False)` IS the simulator's recursion; `[:-1]` lags it
        # to the filtration the step it drives actually has, and `[-1]` is Mu_0.
        ewm = pd.Series(b).ewm(span=span, adjust=False).mean().values if span else None

        if str(self.param.get('Reversion_Model', 'Linear')) == 'Band_Mixture':
            assert span, 'Band_Mixture calibration needs Slow_Mean_Span > 0 (the band is a ' \
                         'deviation from the slow level)'
            return self._calibrate_band_mixture(basis_col, b, dlme, ewm, joint.index, dt_calib)

        if span or fit_garch:
            # ONE likelihood for the whole system: (a, φ, ω_b, α_b, β_b, ν) fitted together, with
            # μ_t entering as the OBSERVABLE the simulator will run rather than as a parameter.
            (fit,), nu_hat, se = arx1_t_mle(
                [(b, dlme)], nu_bounds=(nu_min, nu_max), garch=fit_garch,
                mean=ewm[:-1] if span else np.zeros(b.size - 1))
            a_hat, phi_hat, sigma_t = fit.gamma, fit.phi, fit.sigma
            eta = fit.resid * sigma_t                                     # raw innovation
            logging.info('basis ARX(1)-t joint fit: a=%+.5f (%.1f se) phi=%.4f nu=%.2f | '
                         'sigma=%.4f corr(delta, dS)=%+.4f', a_hat, a_hat / se['gamma'][0],
                         phi_hat, nu_hat, float(eta.std()), float(np.corrcoef(fit.resid, dlme)[0, 1]))
        else:
            y = b[1:]
            X = np.column_stack([dlme, b[:-1]])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a_hat, phi_hat = float(coef[0]), float(coef[1])
            eta = y - X @ coef
            eta_kurt = float(scipy_stats.kurtosis(eta, fisher=True))
            nu_hat = float(np.clip(4.0 + 6.0 / max(eta_kurt, 1.0e-3), nu_min, nu_max))

        if fit_garch:
            g_omega, g_alpha, g_beta = fit.garch
            max_persistence = float(self.param.get('Max_Persistence', 0.999))
            if g_alpha + g_beta > max_persistence:
                # An escalating-variance window (2026 squeeze on a 3y fit) drives the MLE past
                # a unit root, and an explosive basis GARCH random-walks the simulated basis to
                # +/- hundreds of $/oz. Same projection as GARCHSpotCalibration's.
                logging.warning('basis GARCH persistence %.5f > %.5f - scaling beta down.',
                                g_alpha + g_beta, max_persistence)
                g_beta = max_persistence - g_alpha
            vol = {'Sigma': float(eta.std()), 'G_Omega': g_omega, 'G_Alpha': g_alpha,
                   'G_Beta': g_beta, 'Sig2_0': float(sigma_t[-1] ** 2)}
            eta = fit.resid                                               # standardised for `delta`
            logging.info('basis GARCH(1,1)-t: omega=%.4f alpha=%.4f beta=%.4f nu=%.2f | '
                         'persistence=%.4f LR-sigma=%.4f Sig2_0=%.4f (flat Sigma %.4f)',
                         g_omega, g_alpha, g_beta, nu_hat, g_alpha + g_beta,
                         np.sqrt(g_omega / (1.0 - g_alpha - g_beta)), vol['Sig2_0'], vol['Sigma'])
        else:
            # Rolling-21d vol of ΔLME → tercile bins (low/mid/high). Index ascending in
            # vol matches the production HMM's σ-ascending state ordering, so per-regime σ
            # values are positionally consistent with the HMM regime path read at sim time.
            rolling_vol = pd.Series(dlme).rolling(vol_window, min_periods=vol_window).std()
            # Align rolling_vol to η (same length n-1)
            rolling_vol = rolling_vol.values
            valid = ~np.isnan(rolling_vol)
            if valid.sum() < 100:
                sigma_by_state = [float(eta.std())] * 3
            else:
                quantiles = np.quantile(rolling_vol[valid], [1.0 / 3, 2.0 / 3])
                tercile = np.zeros(len(eta), dtype=int)
                tercile[rolling_vol > quantiles[0]] = 1
                tercile[rolling_vol > quantiles[1]] = 2
                tercile[~valid] = 1                                                   # leading NaN → mid
                sigma_by_state = [float(eta[tercile == s].std()) if (tercile == s).sum() > 1
                                  else float(eta.std()) for s in range(3)]
            vol = {'Sigma_By_State': sigma_by_state}

        param = {
            'A': a_hat,
            'Phi': phi_hat,
            'Nu': nu_hat,
            **vol,
            'Calibration_DT_Years': dt_calib,
        }
        if span:
            param.update({'Slow_Mean_Lambda': 1.0 - 2.0 / (span + 1.0), 'Mu_0': float(ewm[-1])})
        delta = pd.DataFrame({basis_col: eta}, index=joint.index[1:])
        return utils.CalibrationInfo(param, [[1.0]], delta)

    def _calibrate_band_mixture(self, basis_col, b, dlme, ewm, index, dt_calib):
        """The Q-Q-ruled law: dead-band reversion around the lagged slow level plus a 2-state
        Gaussian scale mixture. Location (A, kappa, beta) by alternating least squares, the
        mixture by EM, stress persistence from the posterior state runs, and Mix_P0_Stress from
        the forward filter at the sample end. FIT on adjacent business days only (a hole spanning
        several days is one observation of a different law); the delta residuals cover every row
        so the correlation consolidation stays date-aligned with the other factors."""
        from scipy import optimize
        span = int(self.param.get('Slow_Mean_Span', 0))
        d = b[1:] - ewm[:-1]                                    # deviation from the LAGGED level
        dd, d0, ds = np.diff(d), d[:-1], dlme[1:]               # aligned to targets d[1:]
        gaps = np.asarray((pd.DatetimeIndex(index[2:]) - pd.DatetimeIndex(index[1:-1])).days)
        fit_m = gaps <= 4                                       # adjacent business days (weekends ok)

        a_hat, kappa, beta = 0.0, 5.0, 0.2
        x, y, s_ = d0[fit_m], dd[fit_m], ds[fit_m]
        for _ in range(3):
            pull = -beta * np.sign(x) * np.clip(np.abs(x) - kappa, 0.0, None)
            a_hat = float((s_ * (y - pull)).sum() / max((s_ * s_).sum(), 1.0e-12))
            res = optimize.minimize(
                lambda p: (((y - a_hat * s_) + p[1] * np.sign(x) * np.clip(np.abs(x) - p[0], 0.0, None)) ** 2).sum(),
                [kappa, beta], bounds=[(0.0, 25.0), (0.0, 1.0)])
            kappa, beta = float(res.x[0]), float(res.x[1])

        pull_all = -beta * np.sign(d0) * np.clip(np.abs(d0) - kappa, 0.0, None)
        eta_all = dd - a_hat * ds - pull_all
        eta = eta_all[fit_m]
        p_q, s1, s2 = 0.7, eta.std() * 0.4, eta.std() * 1.8
        for _ in range(300):
            f1 = p_q * np.exp(-0.5 * (eta / s1) ** 2) / s1
            f2 = (1.0 - p_q) * np.exp(-0.5 * (eta / s2) ** 2) / s2
            w = f1 / (f1 + f2)
            p_q = float(w.mean())
            s1 = float(np.sqrt((w * eta ** 2).sum() / w.sum()))
            s2 = float(np.sqrt(((1.0 - w) * eta ** 2).sum() / (1.0 - w).sum()))
        stress = w < 0.5
        stay = float((stress[1:] & stress[:-1]).sum() / max(stress[:-1].sum(), 1))
        q = 1.0 - p_q
        # forward filter over the full series for the t0 stress posterior
        enter = q * (1.0 - stay) / (1.0 - q)
        p = q
        for e_t in eta_all:
            prior = p * stay + (1.0 - p) * enter
            lq = np.exp(-0.5 * (e_t / s1) ** 2) / s1
            ls = np.exp(-0.5 * (e_t / s2) ** 2) / s2
            p = prior * ls / (prior * ls + (1.0 - prior) * lq)
        phi_eff = float((d0[fit_m] * d[1:][fit_m]).sum() / max((d0[fit_m] ** 2).sum(), 1.0e-12))
        logging.info(
            'basis band+mixture fit: kappa=$%.2f beta=%.3f A=%+.4f | quiet %.0f%% sd %.2f, '
            'stress %.0f%% sd %.2f, stay=%.2f p0=%.2f | phi_eff=%.3f (n_fit=%d of %d)',
            kappa, beta, a_hat, 100 * p_q, s1, 100 * (1 - p_q), s2, stay, p, phi_eff,
            fit_m.sum(), len(dd))

        sig_blend = np.sqrt(w * s1 ** 2 + (1.0 - w) * s2 ** 2)   # posterior-blended per-obs scale
        z_all = np.empty_like(eta_all)
        z_all[fit_m] = eta / sig_blend
        z_all[~fit_m] = eta_all[~fit_m] / float(np.sqrt(p_q * s1 ** 2 + (1 - p_q) * s2 ** 2))
        param = {
            'A': a_hat, 'Phi': phi_eff, 'Nu': 0.0,
            'Sigma': float(eta.std()),
            'Reversion_Model': 'Band_Mixture',
            'Band_Kappa': kappa, 'Band_Beta': beta,
            'Mix_Q_Stress': q, 'Mix_Sigma_Quiet': s1, 'Mix_Sigma_Stress': s2,
            'Mix_Stay_Stress': stay, 'Mix_P0_Stress': float(p),
            'Slow_Mean_Lambda': 1.0 - 2.0 / (span + 1.0), 'Mu_0': float(ewm[-1]),
            'Calibration_DT_Years': dt_calib,
        }
        delta = pd.DataFrame({basis_col: z_all}, index=index[2:])
        return utils.CalibrationInfo(param, [[1.0]], delta)


def construct_process(sp_type, factor, param, implied_factor=None):
    return globals().get(sp_type)(factor, param, implied_factor)


def construct_calibration_config(calibration_model, param):
    return globals().get(param['Method'])(calibration_model, param)
