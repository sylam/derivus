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

import logging
import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from . import utils

# useful constants
BARRIER_UP = -1.0
BARRIER_DOWN = 1.0
BARRIER_IN = -1.0
BARRIER_OUT = 1.0
OPTION_PUT = -1.0
OPTION_CALL = 1.0

BOUNDARY_MAX_AMPLIFICATION = 25.0
"""Largest ``||weights||_1`` a local-linear boundary solve may return before it is refused.

The weights sum to one by construction, so their L1 norm is EXACTLY the factor by which the fit can
amplify the jumps it is averaging - scale-free in the gap, in the jump and in the sample size, which
a threshold on the solve's determinant is not. Measured over 2685 solves in 82 runs: of the 2424
carrying any density, every one reads 1.00 to 8.06 except the single decision that broke the
Heston-Nandi barrier gradient, at 99.9. 25 sits 3.1x above the largest legitimate reading and 4.0x
below that one."""

KINK_ATOM_BANDWIDTH_FLOOR = 0.01
"""How far below a reporting row's OWN scale its Silverman bandwidth may fall before the kernel is
WRITTEN to zero rather than evaluated (``exposure_kink_term``). It decides nothing about atoms.
The test is on the row's COEFFICIENT OF VARIATION: ``eps <= 0.01 * mean|V|`` is exactly
``std/mean|V| <= 0.01/(1.06 * n**-0.2)`` - 0.0867 at 65536 paths, 0.0657 at 16384, 0.0377 at 1024.
A row carrying mass at the kink cannot land under it (straddling zero gives a CV near 1.25); what
does is a row whose whole mass sits at ONE level. A merely NARROW row is refused by the ladder."""

KINK_ATOM_LADDER = (2.0, 1.0, 0.5, 0.25)
"""The bandwidth ladder ``exposure_kink_term`` classifies a row on, as multiples of that row's own
Silverman width - a factor of EIGHT end to end. A row REFUSES when its ladder DIVERGES, and the
ladder is the only instrument that can say so: a kernel estimate of a density holds still as the
width narrows, while one of a point MASS reads ``p / (h * sqrt(2 pi))`` and climbs as ``1/h``."""

KINK_ATOM_LADDER_DIVERGENCE = 2.0
"""How far ``f_V(0)`` may CLIMB from the widest rung of ``KINK_ATOM_LADDER`` to the narrowest before
the row is refused as an ATOM (``exposure_kink_term``). The two families are 7.7x apart on this
reading: an atom at zero climbs 54.46 / 108.9 / 217.9 / 435.7 across the four rungs, a factor of
8.000, at every mass weight swept from 0.999 down to 0.0001, while the crossing rows of
``tests/test_cva_gamma_kink.py`` read 1.003 / 1.031 / 1.027 / 1.045. Two sits 1.9x above the density
and 4.0x below the mass, at any path count. ``exposure_kink_term`` logs the ladder per row at
DEBUG."""


# Heston-Nandi OSS spot models, opt-in per deal via the Valuation Configuration switch
# SpotModel='HestonNandi' or 'HestonNandiComponent'. The per-step advance is owned by utils;
# a pricer names its model once through the KIT below and walks it after that.


class PlainHestonNandiKit(object):
    """The plain Heston-Nandi model as an OSS walking kit: seed a state, sub-step it, advance it
    on a truncated draw, and price the analytic legs off the seed.

    The MATH stays in `utils` as free functions taking the recursion parameters as explicit
    trailing args; the kit owns the name->args unpack and the state. That state is OPAQUE to the
    pricer: reaching into it would spell out which model is held, which is the branch this exists
    to delete.
    """

    def __init__(self, scalars, knots, steps_per_year):
        *self.params, self.h0 = scalars          # (omega, alpha, beta, gamma_star), H0

    def seed(self):
        return (self.h0,)

    def h(self, state):
        """The predictable variance of the NEXT step - what an OSS truncation bound is built off."""
        return state[0]

    def substeps(self, Sj, state, b_step, n_steps, shared, num_sims, antithetic):
        Sj, h = utils.hn_unmonitored_substeps(
            Sj, state[0], b_step, n_steps, self.params, shared, num_sims, antithetic)
        return Sj, (h,)

    def advance(self, Sj, state, b_step, Z):
        Sj, h = utils.hn_daily_advance(Sj, state[0], b_step, Z, *self.params)
        return Sj, (h,)

    def scalar_state(self):
        """The SEED state reduced to scalars, for the analytic legs - which price the remaining
        horizon from the row's own entry state, never from a walked one."""
        return (self.h0.reshape(-1)[0],)

    def cdf_logret(self, x, n_steps, r_step):
        """Q(R_n <= x) off the seed state - the already-hit / parity digital leg."""
        om, al, be, ga = (v.reshape(-1)[0] for v in self.params)
        return utils.hn_cdf_logret(x, n_steps, self.scalar_state()[0], om, al, be, ga, r_step)

    def vanilla(self, S, K, n_steps, r_step, is_call):
        """The European closed form off the seed state, under THIS model rather than Black at the
        implied surface - the KI parity leg and the already-hit leg."""
        om, al, be, ga = (v.reshape(-1)[0] for v in self.params)
        return (utils.hn_call if is_call else utils.hn_put)(
            S, K, n_steps, self.scalar_state()[0], om, al, be, ga, r_step)


class ComponentHestonNandiKit(PlainHestonNandiKit):
    """The COMPONENT Heston-Nandi model as an OSS walking kit - the same three verbs over the pair
    (h, q) and a per-step intercept strip.

    THE DAY COUNTER IS PART OF THE STATE, because omega_t is a function of calendar position:
    omega_t = L_{t+1} - rho*L_t off the fitted L curve, on the calibration's own trading-day clock
    from the BASE DATE. So the state carries how many daily steps this row's walk has taken, and
    every verb slices the strip at it.

    THE ROW RE-SEEDS AT DAY ZERO: every OSS leg prices its MTM row's remaining horizon under the
    law seen FROM THE BASE DATE - h at H0, q at L(0) by the anchoring, omega from omega_0. A
    declared approximation of the same class as the plain model's H0 re-seed (a row at six months
    should enter at that row's own state); the closed-form gate prices exactly what the walk
    simulates because both start at the same place.
    """

    def __init__(self, scalars, knots, steps_per_year):
        *self.params, self.h0, self.l_values = scalars   # (alpha,beta,g1,rho,phi,g2), H0, L values
        self.knots = knots
        self.steps_per_year = float(steps_per_year)
        # rho SQUEEZED to 0-dim for the strip: scalar parameters arrive (-1, 1) to broadcast
        # against the [batch, sims] variance state, but the omega path is indexed by TRADING DAY
        # and has no path axis - a (1, 1) rho turns a (n,) strip into (1, n)
        self.rho = self.params[3].reshape(())
        self._omegas, self._built = None, 0

    def omegas(self, day, n_steps):
        """The `n_steps` intercepts starting at trading day `day`. The strip is built once and
        DOUBLED when a longer walk asks for it, so a row's walk costs one L interpolation rather
        than one per fixing.

        THE `is None` IS NOT REDUNDANT WITH THE LENGTH TEST: a daily fixing walks ZERO unmonitored
        sub-steps, so the FIRST call can be `omegas(0, 0)`, which the length test alone answers by
        slicing a strip that was never built."""
        if self._omegas is None or self._built < day + n_steps:
            self._built = max(day + n_steps, 2 * self._built, 64)
            l_path = utils.hn_component_l_path(
                self.knots, self.l_values, self._built, self.steps_per_year)
            self._omegas = utils.hn_component_omega_path(l_path, self.rho)
        return self._omegas[day:day + n_steps]

    def q0(self):
        """q_0 = L(0), THE ANCHORING - read off the curve rather than carried as a second field."""
        return utils.hn_component_l_path(self.knots, self.l_values, 0, self.steps_per_year)[0]

    def seed(self):
        return (self.h0, torch.zeros_like(self.h0) + self.q0(), 0)

    def substeps(self, Sj, state, b_step, n_steps, shared, num_sims, antithetic):
        h, q, day = state
        Sj, h, q = utils.hn_component_unmonitored_substeps(
            Sj, h, q, b_step, self.omegas(day, n_steps), self.params,
            shared, num_sims, antithetic)
        return Sj, (h, q, day + n_steps)

    def advance(self, Sj, state, b_step, Z):
        h, q, day = state
        Sj, h, q = utils.hn_component_daily_advance(
            Sj, h, q, b_step, Z, self.omegas(day, 1)[0], *self.params)
        return Sj, (h, q, day + 1)

    def scalar_state(self):
        return (self.h0.reshape(-1)[0], self.q0().reshape(-1)[0])

    def cdf_logret(self, x, n_steps, r_step):
        h0, q0 = self.scalar_state()
        return utils.hn_component_cdf_logret(
            x, list(self.omegas(0, n_steps)), h0, q0, *self.params, r_step)

    def vanilla(self, S, K, n_steps, r_step, is_call):
        h0, q0 = self.scalar_state()
        return (utils.hn_component_call if is_call else utils.hn_component_put)(
            S, K, list(self.omegas(0, n_steps)), h0, q0, *self.params, r_step)


#: The OSS kits, keyed by the `SpotModel` valuation option each deal declares. A pricer looks its
#: model up ONCE (`oss_model_kit`) and never names one again, so a third GARCH family is a class
#: here and a row in this dict rather than a fifth branch in four pricers.
OSS_SPOT_MODEL_KITS = {'HestonNandi': PlainHestonNandiKit,
                       'HestonNandiComponent': ComponentHestonNandiKit}


def oss_model_scalars(factor_dep, shared):
    """The declared spot model's parameter tensors, in the order its own kit unpacks them.

    Read by the canonical name tuple the MODEL owns (`utils.HN_PARAM_NAMES` /
    `HN_COMPONENT_PARAM_NAMES` plus its curve). Scalars come out (-1, 1) to broadcast against the
    [batch, sims] state; a CURVE parameter comes out flat, being indexed by knot and not by path.
    Empty tuple where the deal prices GBM, which is what `oss_model_kit` reads as None.
    """
    if 'HN_Params' not in factor_dep:
        return ()
    code = factor_dep['HN_Params'][0]
    block = {x.name[-1]: shared.t_Static_Buffer[x] for x in code[utils.FACTOR_INDEX_Offset]}
    if code[utils.FACTOR_INDEX_SubType] == 'HestonNandiComponent':
        return tuple([block[k].reshape(-1, 1) for k in utils.HN_COMPONENT_PARAM_NAMES] +
                     [block[utils.HN_COMPONENT_CURVE_NAME].reshape(-1)])
    return tuple(block[k].reshape(-1, 1) for k in utils.HN_PARAM_NAMES)


def oss_model_kit(factor_dep, scalars):
    """The declared spot model's kit, built from the tensors the pure inner function was handed.

    `scalars` are the model parameters as they cross the bound/theta split - `InnerMCRecompute`
    needs every tensor an explicit argument, so they arrive here rather than off `factor_dep`; the
    model NAME and the L curve's knots are compile-time facts and ride `factor_dep`. An empty
    `scalars` is a GBM deal and answers None.
    """
    if not scalars:
        return None
    model = factor_dep['HN_Params'][0][utils.FACTOR_INDEX_SubType]
    knots = factor_dep['HN_Params'][0][utils.FACTOR_INDEX_Tenor_Index].get(
        utils.HN_COMPONENT_CURVE_NAME)
    return OSS_SPOT_MODEL_KITS[model](scalars, knots, factor_dep['HN_Steps_Per_Year'])

def cash_settle(shared, currency, time_index, value):
    """Book `value` against `currency` at `time_index`, if that date is a settlement point."""
    if shared.t_Cashflows is not None and time_index in shared.t_Cashflows.get(currency, []):
        shared.t_Cashflows[currency][time_index] += value


def calc_moneyness(strike, spot, forward, deal_data, use_forward=False, invert_moneyness=False):
    '''
    Deals with different ways of calculating moneyness - either spot/strike, forward/strike
    or for svi/skew surfaces, return log(strike/forward) or the strike
    :param use_forward: use the forward to calculate moneyness (otherwise use the spot)
    :param invert_moneyness: if true, calculate moneyness as strike / forward (or spot)
    :param strike: strike value
    :param spot: spot price tensor
    :param forward: forward price tensor
    :param deal_data: Deal_data struct containing deal specific data
    :return: the moneyness ( or information relevant to calculate the moneyness)
    '''
    subtype = deal_data.Factor_dep['Volatility'][0][utils.FACTOR_INDEX_SubType]
    if subtype[0] in ['SVI', 'Skew']:
        # Sticky_Strike divides by the ATM_Ref, which can only be done later, so the raw strike
        # is returned here; Sticky_Moneyness is log of strike over forward
        return strike if subtype[1]=='Sticky_Strike' else torch.log(strike/forward)
    elif subtype[0] == 'Relative_Forward':
        return (strike - forward)/forward
    elif subtype[0] == 'Malz':
        forward_or_spot = forward if use_forward else spot
        return torch.log(strike / forward_or_spot if invert_moneyness else forward_or_spot / strike)
    else:
        # a regular 2d surface is assumed sticky_moneyness (TODO: other moneyness rules)
        forward_or_spot = forward if use_forward else spot
        return strike / forward_or_spot if invert_moneyness else forward_or_spot / strike


def forward_carry_rate(carry_rate, cum_t, dt):
    """The annualised carry over EACH fixing INTERVAL, from zero rates read at the fixing tenors.

    ``carry_rate[j]`` (a ``calc_eq_drift``/``calc_fx_drift`` gather with ``multiply_by_time=False``)
    is the AVERAGE rate over ``[t, T_j]``, so the interval ending at ``T_j`` carries a DIFFERENCE of
    cumulative integrals, ``(c_j*T_j - c_{j-1}*T_{j-1}) / dt_j`` - equal to ``c_j`` only on the
    first interval or a flat curve. The strip telescopes to ``total_log_forward``.

    Rank-polymorphic: ``[N_fix, batch]`` for one MTM row or ``[N_block, N_fix, batch]`` for a block,
    with ``cum_t``/``dt`` carrying no batch axis. A zero-length interval (an already-observed
    fixing, skipped by every caller) divides by one, keeping unread entries out of the backward
    pass. A flat curve is NOT bit-identical - differencing amplifies cumulative-time rounding by
    ``T_j/dt_j`` - so gates on this must not use ``torch.equal``.
    """
    cum = carry_rate * cum_t.unsqueeze(-1)
    step = dt[..., 1:].unsqueeze(-1)
    return torch.cat([carry_rate[..., :1, :], cum.diff(dim=-2) /
                      torch.where(step > 0, step, torch.ones_like(step))], dim=-2)


def total_log_forward(carry_rate, times):
    """``log F(t,T) / S(t)``: the interval carry strip integrated over a fixing strip.

    THE forward to expiry for every leg that needs one - the sum telescopes to ``c_N*T_N``. Takes
    ``forward_carry_rate``'s strip, never raw zero rates. Rank-polymorphic (``[N_fix, batch]`` or
    ``[N_block, N_fix, batch]``; ``times`` carries no batch axis), so a leg valuing one row and a
    leg valuing the block are the same expression - which is what keeps the two legs of
    ``pv_discrete_barrier_option`` on one forward.
    """
    return (carry_rate * times.unsqueeze(-1)).sum(dim=-2)


def forward_vol_strip(deal_data, strike, spot, carry_rate, cum_t, shared,
                      invert_moneyness=False, use_forwards=False):
    """The implied vol read at EVERY fixing's own tenor, at the moneyness the DEAL declares.

    A surface is quoted against an option's EXPIRY, so a simulation stepping a fixing strip reads it
    once per fixing. ``carry_rate`` is the ZERO carry (the raw gather - a forward wants the
    cumulative integral ``F_j = S * exp(c_j * T_j)``), and ``cum_t`` arrives as NUMPY because it is
    also the surface's hashable expiry key. ``use_forwards`` is the DEAL's declaration, so the strip
    reads the surface at the same moneyness the pricer marks its own European legs with; the
    per-fixing smile convention is an open modelling question (roadmap.md).

    Rank-polymorphic as ``forward_carry_rate``; the fixing axis returns at ``-2`` so the strip drops
    straight into ``forward_vol_rate``.
    """
    forward = spot.unsqueeze(-2) * torch.exp(carry_rate * carry_rate.new(cum_t).unsqueeze(-1))
    moneyness = calc_moneyness(strike, spot.unsqueeze(-2).expand_as(forward), forward, deal_data,
                               use_forward=use_forwards, invert_moneyness=invert_moneyness)
    return torch.stack([utils.calc_time_grid_vol_rate(
        deal_data.Factor_dep['Volatility'], moneyness[..., j, :],
        np.atleast_1d(cum_t[..., j]), shared).reshape(moneyness[..., j, :].shape)
        for j in range(cum_t.shape[-1])], dim=-2)


def forward_vol_rate(vols, cum_t, dt):
    """The annualised vol over EACH fixing INTERVAL, from implied vols read at the fixing tenors.

    An implied vol is CUMULATIVE (``sigma(T)^2 * T`` is total variance to ``T``), so the interval
    ending at ``T_j`` carries ``(sigma_j^2 T_j - sigma_{j-1}^2 T_{j-1}) / dt_j``, not
    ``sigma_j^2`` - ``forward_carry_rate``'s statement one factor over, with the same failure mode:
    the wrong allocation telescopes to the right TOTAL variance, so every European limit stays exact
    and only path-dependent monitoring is biased.

    ``clamp(min=eps)`` floors a DECLINING cumulative variance (an arbitrageable surface) rather than
    taking a negative square root. The floor does NOT telescope: the strip's total then exceeds the
    surface's own, so in-out parity against the simulated vanilla breaks while the analytic half
    stays exact. The fix for that is an arbitrage-free surface, not a different floor. ``j == 0``
    takes ``vols[0]`` directly, its window being the interval itself.

    Rank-polymorphic as ``forward_carry_rate``; a zero-length interval divides by one; a flat
    surface is NOT bit-identical, so gates on this must not use ``torch.equal``.
    """
    cum_var = vols * vols * cum_t.unsqueeze(-1)
    step = dt[..., 1:].unsqueeze(-1)
    fwd = torch.sqrt(cum_var.diff(dim=-2).clamp(min=torch.finfo(vols.dtype).eps) /
                     torch.where(step > 0, step, torch.ones_like(step)))
    return torch.cat([vols[..., :1, :], fwd], dim=-2)


def oss_uniforms(shared, n_fix, num_sims, sobol):
    """Antithetic uniforms for a one-step-survival loop: ``[n_fix, batch, 2 * num_sims]``.

    One Sobol/pseudo draw plus its ``1 - u`` mirror, which pairs the truncated final draws of an OSS
    step with the antithetic halves of ``hn_unmonitored_substeps``. NOT used by
    ``pv_MC_AutoCallSwap``'s no-averaging loop, which draws the same Sobol block but consumes it RAW
    - adopting this there would change that estimator.
    """
    if sobol:
        u = shared.quasi_rng(shared.simulation_batch, n_fix * num_sims)[1].T.reshape(
            n_fix, shared.simulation_batch, -1)
    else:
        u = torch.rand([n_fix, shared.simulation_batch, num_sims],
                       dtype=shared.one.dtype, device=shared.one.device)
    return torch.concat([u, 1.0 - u], dim=-1)


def oss_truncated_draw(u, z_bound, survive_below):
    """Survival probability and the survival-truncated standard normal draw - ONE spelling.

    Given the standardised bound of this step's barrier: ``survive_below=True`` means survival is
    ``{Z <= z_bound}`` and the draw maps into the lower tail, ``icdf(u * p)``; ``False`` means
    ``{Z >= z_bound}`` and the upper tail, ``icdf(Phi + u * p)``. Returns ``(p, Z)`` - what the
    knocked-out weight ``(1 - p) * L`` is worth belongs to the caller.

    Adopted by ``sim_spot_tarf`` and ``pv_MC_Accumulator``. THREE call sites still spell it inline
    and a defect fixed here must be checked against all three: ``sim_spot_oss`` twice (its down-side
    base ``(1-p) + u*p`` differs from ``Phi + u*p`` in the last bit below a half, so absorbing it
    would re-baseline the barrier's pinned fixtures) and ``pv_MC_AutoCallSwap`` up-side only (whose
    ``p`` is formed a screen away from where the draw is taken).
    ``tests/test_branch_and_weight.py`` gates each inline spelling against this one.
    """
    eps = torch.finfo(u.dtype).eps
    Phi = utils.norm_cdf(z_bound)
    if survive_below:
        p = Phi
        Z = utils.norm_icdf(torch.clamp(u * p, eps, 1.0 - eps))
    else:
        p = 1.0 - Phi
        Z = utils.norm_icdf(torch.clamp(Phi + u * p, eps, 1.0 - eps))
    return p, Z


def branch_and_weight(shared, deal_data):
    """Is the SMOOTH estimator on for this deal - and under a non-GBM spot model, the refusal.

    ``Branch_And_Weight`` (declared on ``Base_Revaluation`` alone, default 'No') SWAPS the value
    estimator rather than adding to it: at each fixing the fired branch is integrated analytically
    against the conditioning step's own law and the continuing branch draws from the truncated one,
    so the same expectation comes back with lower variance and no indicator left on the tape. It is
    a switch and not an addition because one decision must have ONE estimator - on the smooth path
    a deal registers no ``BoundarySet``, or the boundary flux is counted twice. Off, and absent, is
    the crisp OSS path bit for bit.

    THE CONDITIONING STEP IS THE FIXING INTERVAL'S OWN LOGNORMAL LAW, which is what makes ``p`` a
    ``Phi`` and the continuing draw a ``Phi^-1`` (``oss_truncated_draw``). Under GBM the fixing
    interval IS the simulated step, so the strips a pricer already walks (``forward_carry_rate`` /
    ``forward_vol_rate``) are the ``m`` and ``s`` the construction wants.

    Under Heston-Nandi that law is not in hand and this REFUSES by name: the HN walk is daily, so
    the only Gaussian conditional available is the last daily sub-step, whatever the fixing spacing.
    The fixing-interval conditional needs the k-step law ``exp(A_k + B_k h + C_k q)`` - the stride's
    cached ``Phi`` and its survival-truncated inverse - and HN branch-and-weight lands with it.

    Read ONCE per deal at the top of a pricer, so the refusal lands before a draw is taken.
    """
    if not shared.branch_and_weight:
        return False
    factor_dep = deal_data.Factor_dep
    if 'HN_Params' in factor_dep:
        model = factor_dep['HN_Params'][0][utils.FACTOR_INDEX_SubType]
        raise ValueError(
            "Branch_And_Weight: 'Yes' is refused on {} under SpotModel={!r} - the smooth estimator "
            "conditions on the FIXING interval's own lognormal law, and a {} walk is DAILY, so the "
            'only Gaussian conditional in hand is the last daily sub-step (the s^-3 regime, at '
            'monthly fixings too, because the walk is daily whatever the fixings are). The '
            'fixing-interval law and its survival-truncated inversion are THE STRIDE\'s cached Phi '
            'and inverse-CDF verbatim - utils.hn_cdf_logret is the existing half - and HN '
            'branch-and-weight lands the day the stride does (roadmap.md, "Branch and weight for '
            'TARFs and autocalls"). Applying a Gaussian p here would be a wrong number wearing the '
            "right estimator's name. What works today: price this deal under GBM (drop SpotModel), "
            "or run it with Branch_And_Weight: 'No', which is the default and is the crisp OSS "
            'estimator this deal already prices under, unchanged.'.format(
                deal_data.Instrument.field.get('Reference'), model, model))
    return True


def lognormal_partial_moment(spot, drift, vol, z_bound, fired_above, power=1.0):
    """``E[S_k**power * 1{fired}]`` under ONE fixing interval's lognormal law - the analytic half of
    a fired branch.

    ``S_k = spot * exp(drift + vol * Z)`` with ``Z`` standard normal over the interval (``drift``
    and ``vol`` are the interval's own ``m`` and ``s``), and ``z_bound`` is the trigger standardised
    the way ``oss_truncated_draw`` standardises it. ``fired_above`` says which tail fires: True for
    a trigger crossed from below. ``E[S^a 1{Z > z}] = spot^a exp(a*m + a^2 s^2/2) Phi(a*s - z)``,
    and the down side reflects, so one expression carries every moment a fired payoff needs:

    - ``power=0`` IS the fired PROBABILITY. A caller already holding the survival ``p`` from
      ``oss_truncated_draw`` should use ``1 - p`` instead: those two are exact complements, and the
      survival ledger's telescoping identity is exact only because they are (``SurvivalLedger``).
    - ``power=1`` is the forward-weighted mass ``E[S 1{fired}]`` every gain reads.
    - ``power=-1`` is the same for an INVERTED accrual, whose payoff is written in ``1/S``
      (``pv_MC_Tarf``'s ``InvertedTarget``).

    THE FIRED BRANCH IS AN EXPECTATION, NEVER A PROBABILITY TIMES A SAMPLE. The jump and the
    decision share the fixing, so ``p * (realised payoff)`` is biased and does not vanish with path
    count; every closed form here is ``E[payoff * 1{fired}]``, conditional on the state one step
    back.
    """
    sign = 1.0 if fired_above else -1.0
    tail = utils.norm_cdf(sign * (power * vol - z_bound))
    if not power:
        # the fired probability: both scale factors are identities, skipped rather than evaluated
        # because this runs once per fixing per path
        return tail
    return (spot ** power) * torch.exp(power * (drift + 0.5 * power * vol * vol)) * tail


def lognormal_fired_gain(spot, drift, vol, z_bound, strike, fired_above, power=1.0):
    """``E[(S_k**power - strike**power) * 1{fired}]`` - a gain paid on one tail of the interval.

    Black's own difference of two partial moments, in the conditioning step's law rather than the
    expiry's. ``power=-1`` prices the inverted accrual, where the gain is written in reciprocals and
    ``strike**power`` is ``1/K``; the caller supplies the payoff's sign, because which side is a
    gain is the deal's convention and not this expression's.

    FOR A LEG WHOSE PAYOFF IS A GAIN OVER A TAIL - ``pv_MC_Tarf``'s knock-IN and
    ``pv_MC_AutoCallSwap``'s put leg. A TARF's own fired branch is NOT built from it: the fixing
    that fills the target pays the remaining target ``R``, measurable one fixing back, so its
    conditional expectation is itself and the branch is ``(1 - p) * R``.
    """
    return (lognormal_partial_moment(spot, drift, vol, z_bound, fired_above, power) -
            (strike ** power) * lognormal_partial_moment(
                spot, drift, vol, z_bound, fired_above, 0.0))


class SurvivalLedger(object):
    """The alive weight down a fixing strip, and what mass each fixing FIRED - one object, so the
    telescoping identity is checkable rather than implicit.

    Every branch-and-weight product walks the same two lines: the fixing fires with weight
    ``alive * (1 - p)`` and pays its analytic tail expectation, and the alive weight advances to
    ``alive * p`` for the truncated draw to carry on from.

    THE IDENTITY: ``sum_j alive_{j-1} * (1 - p_j) + alive_final == alive_0``, per path, exactly to
    float roundoff. Stated against the STARTING weight rather than against one, because a block that
    opens on a partly-resolved deal starts at whatever the observed prefix left. It holds through
    observed fixings too, where ``p`` is an exact 0/1 indicator rather than a ``Phi``. It is exact
    only because the fired mass is spelled ``1 - p`` off the SAME ``p`` the draw was truncated with;
    an independent ``Phi(-z)`` beside a survival ``Phi(z)`` leaves a residual that says nothing
    about the ledger.

    ``check`` logs the residual at DEBUG per block rather than asserting in a hot path;
    ``conservation`` is the number itself, for a gate to hold to roundoff.
    """

    def __init__(self, alive):
        self.initial = alive
        self.alive = alive
        # None until the first fixing, so a strip with no fixings conserves trivially rather than
        # needing a zero of the right shape up front
        self.fired = None

    def fire(self, p_survive):
        """This fixing's FIRED weight; the alive weight advances behind it. ``p_survive`` is the
        survival probability the truncated draw was taken with (``oss_truncated_draw``), so the
        fired mass is its exact complement."""
        weight = self.alive * (1.0 - p_survive)
        self.fired = weight if self.fired is None else self.fired + weight
        self.alive = self.alive * p_survive
        return weight

    def conservation(self):
        """The telescoping residual, per path: zero to roundoff or the ledger has lost mass."""
        return (self.alive - self.initial) if self.fired is None else (
            self.fired + self.alive - self.initial)

    def check(self, label):
        """Log the identity at DEBUG - one line per block, naming what walked it."""
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug('LEDGER %s conservation=%.3g alive=%.6g', label,
                          float(self.conservation().abs().max()), float(self.alive.mean()))


def boundary_weights(gap, bandwidth):
    """Density at the boundary and local-linear regression weights, estimated SEPARATELY.

    The term to recover is ``f_G(0) * E[jump * dG/dtheta | G = 0]``. Folding both halves into one
    kernel is a local-CONSTANT estimator whose O(bandwidth) bias never settles; local-linear weights
    cancel the first-order term, so the estimate holds still over a range of bandwidths.

    Returns ``(density_at_zero, weights)`` with the weights summing to one, so the caller multiplies
    rather than averages - and that sum makes ``||weights||_1`` exactly the solve's amplification.
    Past ``BOUNDARY_MAX_AMPLIFICATION`` the solve is refused on what it PRODUCED and lands on the
    empty-kernel branch (weights zero, correction exactly zero).

    ONE sample has no spread for a kernel width - ``std()`` would be NaN and reach ``backward()``
    silently - so zero width is returned and the correction is exactly zero, which is what one
    scenario means. Base valuation reaches this path.
    """
    g = gap.detach()
    spread = g.std() if g.numel() > 1 else torch.zeros_like(g.reshape(-1)[:1].squeeze())
    width = bandwidth * spread.clamp_min(torch.finfo(g.dtype).eps)
    k = torch.exp(-0.5 * (g / width) ** 2)
    s0, s1, s2 = k.sum(), (k * g).sum(), (k * g * g).sum()
    # nothing at all near the boundary makes this exactly 0
    denominator = s2 * s0 - s1 * s1
    solvable = denominator != 0
    weights = torch.where(solvable, k * (s2 - g * s1) / torch.where(
        solvable, denominator, torch.ones_like(denominator)), torch.zeros_like(k))
    weights = torch.where(weights.abs().sum() <= BOUNDARY_MAX_AMPLIFICATION,
                          weights, torch.zeros_like(weights))
    density = s0 / (g.numel() * math.sqrt(2.0 * math.pi) * width)
    return density, weights


def stochastic_boundary_correction(gap, objective_jump, bandwidth):
    """A term worth EXACTLY ZERO in the forward pass that carries the missing boundary derivative
    into the backward one.

    Ordinary AAD differentiates an expectation containing 1{gap > 0} with the decision frozen,
    dropping ``f_G(0) * E[jump * dG/dtheta | G = 0]``. ``gap - gap.detach()`` is numerically zero
    with derivative one, so adding this to the scalar handed to backward() reports an unchanged
    number and still propagates the missing term through ``gap`` to every factor at once.

    ``objective_jump`` is a COEFFICIENT and stays detached: its own pathwise derivatives are already
    in the ordinary AAD value, and differentiating the counterfactual would count them twice.
    Summed, not averaged: the local-linear weights already carry the 1/N through the density.
    """
    density, weights = boundary_weights(gap, bandwidth)
    return ((gap - gap.detach()) * (density * weights * objective_jump).detach()).sum()


def boundary_correction(shared, objective, reported_mtm, bandwidth):
    """Total boundary correction for every recorded decision - a margin call's transfer, a barrier
    crossing, an autocall trigger, any observed event whose value jump is real.

    How each counterfactual is produced belongs to the set that recorded it. This half is `score`:
    the reported PORTFOLIO plus a change to it. Never a set's own level, because the objective is
    applied to `resolve_structure`'s root sum over every netting set, and a collateralised set's
    post-collateral net sits at the relu kink by construction. `gap > 0` means the trigger fired,
    matching a jump of J(fired) - J(did not).
    """
    def score(delta):
        return objective(reported_mtm + delta)

    corrections = []
    for bset in shared.boundary_sets:
        for k, (gap, jump) in enumerate(bset.objective_jumps(score)):
            corrections.append(stochastic_boundary_correction(gap, jump, bandwidth))
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                # the forward value is zero by construction; the COEFFICIENT it multiplies into
                # the gap's graph is the reading
                density, weights = boundary_weights(gap, bandwidth)
                logging.debug('BOUNDARY %s decision=%d coeff=%.6g',
                              type(bset).__name__, k,
                              float((density * weights * jump).sum()))
    return torch.stack(corrections).sum() if corrections else None


def _kink_density_at_zero(Vbar, width, n_paths_axis):
    """``f_V(0)`` per reporting row, by the same normalised Gaussian kernel ``exposure_kink_term``
    carries into the tape but at an ARBITRARY width - which is what turns one density reading into
    the ladder that can say whether it is a density at all."""
    return torch.exp(-0.5 * (Vbar / width) ** 2).mean(
        dim=n_paths_axis, keepdim=True) / (width * math.sqrt(2.0 * math.pi))


def kink_kernel(Gbar, axis, label):
    """The kernel machinery every ``max(x, 0)`` on a simulated quantity needs: Silverman per row,
    the atom LADDER, and the normalised Gaussian kernel evaluated AT the kink.

    ``Gbar`` is the relu's argument, already DETACHED by the caller - that discipline stays the
    caller's, so this cannot silently put ``K'`` on a tape. ``axis`` is the sample axis the
    bandwidth and the density are taken over: reporting PATHS for ``exposure_kink_term``, the inner
    sims of one fixing for ``accrual_kink_term``. Everything else broadcasts, so one bandwidth per
    row (or per fixing, per scenario) reaches its own samples.

    Returns ``(kernel, atom, rungs)``. The REFUSAL is the CALLER's: an atom means the same thing at
    both, but the remedies at a reporting row are not the remedies at a fixing, so the caller reads
    ``atom`` and says its own sentence carrying ``rungs`` as the reading.

    ``KINK_ATOM_BANDWIDTH_FLOOR`` DECIDES NOTHING ABOUT ATOMS: it says where the bandwidth has
    collapsed under the row's own scale, and there the kernel is WRITTEN to zero rather than
    evaluated, because a zero bandwidth is 0/0 forward and NaN backward. The ladder still runs under
    it, so a row pinned at the kink on a handful of samples is refused rather than written off.
    """
    n = Gbar.shape[axis]
    # one sample -> zero width; std() would be NaN and a single path has no density to estimate
    spread = Gbar.std(dim=axis, keepdim=True) if n > 1 else torch.zeros_like(
        Gbar.narrow(axis, 0, 1))
    eps = 1.06 * spread * n ** -0.2
    # the row's OWN scale, so the floor is scale-free in the argument's units
    floor = KINK_ATOM_BANDWIDTH_FLOOR * Gbar.abs().mean(dim=axis, keepdim=True)
    collapsed = eps <= floor

    # THE LADDER, which is what says atom - and it runs under the floor too. A rung on a row with
    # no bandwidth at all is substituted, not divided, so no branch reaches a comparison as NaN
    has_width = eps > 0
    unit = torch.ones_like(eps)
    with torch.no_grad():
        rungs = [_kink_density_at_zero(Gbar, torch.where(has_width, c * eps, unit), axis)
                 for c in KINK_ATOM_LADDER]
    # a product rather than a ratio: the widest rung is exactly zero wherever the kernel underflows
    # at every width, and that row is far from the kink rather than divergent
    atom = has_width & (rungs[0] > 0) & (rungs[-1] >= KINK_ATOM_LADDER_DIVERGENCE * rungs[0])

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        # the term's forward value is an exact zero on every row; the LADDER is the reading, and
        # it is how a row that passed is told from a row that had nothing to say
        for r, rung in enumerate(torch.stack([x.reshape(-1) for x in rungs], dim=-1).tolist()):
            logging.debug('KINK %s row=%d f(0)=%.6g ladder=%s climb=%.3f collapsed=%d', label, r,
                          rung[1], '/'.join('{:.6g}'.format(x) for x in rung),
                          rung[-1] / rung[0] if rung[0] else float('nan'),
                          int(collapsed.reshape(-1)[r]))

    # a collapsed row admits nothing: written, not evaluated, because eps is an exact zero there
    # and 0/0 would reach backward() as NaN through the unselected branch
    width = torch.where(collapsed, unit, eps)
    kernel = torch.where(collapsed, torch.zeros_like(eps), torch.exp(
        -0.5 * (Gbar / width) ** 2) / (width * math.sqrt(2.0 * math.pi)))
    return kernel, atom, rungs


def _kink_atom_reading(atom, rungs):
    """The rows an atom was diagnosed on, the worst of them, and its ladder rendered - the reading
    both kink refusals quote."""
    flat = atom.reshape(-1)
    rows = torch.nonzero(flat).reshape(-1).tolist()
    readings = torch.stack([rung.reshape(-1)[flat] for rung in rungs], dim=-1)
    worst = int(torch.argmax(readings[:, -1] / readings[:, 0]))
    return (rows, rows[worst],
            ' / '.join('{:.4g}'.format(float(x)) for x in readings[worst]),
            ' / '.join('{:g}'.format(c) for c in KINK_ATOM_LADDER),
            float(readings[worst, -1] / readings[worst, 0]),
            KINK_ATOM_LADDER[0] / KINK_ATOM_LADDER[-1])


def accrual_kink_term(gain, fixing, n_paths_axis=1):
    """The per-fixing accrual's missing SECOND derivative, on the CONTINUING branch of a
    branch-and-weight simulation - ``exposure_kink_term``'s statement one product in.

    An accrual pays ``max(cp * (S_j - K), 0)`` at every fixing, and second-order AAD drops
    ``delta(g) * g_theta g_theta^T`` there. With ``u = gain - gain.detach()`` this returns
    ``0.5 * K_eps(gain.detach()) * u**2`` per (scenario, path), for the caller to weight by that
    path's SURVIVAL weight and the payment's own discount - those weights are the deal's, not the
    kink's. ``gain`` is the relu's ARGUMENT, ``cp * (S_j - K)``, not the payoff.

    Value is an exact zero because ``u`` is an exact IEEE zero; first order accumulates ``+0.0`` bit
    for bit; the Hessian contributes the kernel estimate of
    ``f_g(0) * E[g_theta g_theta^T | g = 0]``. Never BUILT when second order is not asked for, so
    the cost is zero otherwise.

    ONLY the continuing branch: a fixing the row has already OBSERVED is a point mass by
    construction - one spot on every inner path - so callers keep observed fixings out rather than
    relying on the bandwidth collapsing. A fixing whose conditional law carries a genuine ATOM at
    the strike REFUSES by name, on the same ladder and divergence threshold as the exposure term.
    """
    Gbar = gain.detach()
    kernel, atom, rungs = kink_kernel(Gbar, n_paths_axis, 'accrual')
    if bool(atom.any()):
        rows, worst, reading, ladder, climb, span = _kink_atom_reading(atom, rungs)
        raise utils.SecondOrderRefused(
            'accrual_kink_term: fixing {} carries an ATOM of accrual at the strike on scenario(s) '
            '{} - the density f_g(0) this term estimates does not settle as its bandwidth narrows, '
            'it CLIMBS. Scenario {} reads {} at {} x its own Silverman width: a factor of {:.2f} '
            'across a ladder of {:.0f}, which is the 1/bandwidth signature of a point mass rather '
            "than the plateau of a density. What works today: ask for first order alone (Greeks: "
            "'First'), which this term does not touch and which the branch-and-weight path reports "
            'unchanged; or run the deal with Branch_And_Weight off, whose crisp estimator answers '
            'zero there rather than refusing - a smaller number, not a smaller error; or move the '
            'fixing off the strike it is pinned against.'.format(
                fixing, rows, worst, reading, ladder, climb, span))
    u = gain - Gbar
    return 0.5 * kernel * u * u


def exposure_kink_term(V, n_paths_axis=1):
    """A term worth EXACTLY ZERO in the forward pass and BIT-IDENTICALLY zero at first order that
    carries the exposure relu's missing SECOND derivative into the double backward.

    What second-order AAD drops at ``relu(V)`` is ``delta(V) * V_theta V_theta^T`` - a density AT
    the boundary times an outer product, the same object the first-order boundary fix estimates.
    With ``u = V - V.detach()`` this returns ``0.5 * K_eps(V.detach()) * u**2`` per (row, path),
    added beside ``relu(V)`` under the objective's own weights. Value is an exact zero because ``u``
    is an exact IEEE zero; the gradient ``K*u*V_theta`` accumulates ``+0.0`` bit for bit; the
    Hessian ``K_eps(Vbar) * V_theta V_theta^T`` has weighted path mean
    ``f_V(0) * E[V_theta V_theta^T | V = 0]``, the dropped term.

    ``K``'s argument is DETACHED, which confines the construction to the density's VALUE and keeps
    ``K'`` off the tape. ``K`` is the NORMALISED Gaussian kernel ``phi(x/eps)/eps``, because the
    caller's reduction is a mean over paths and it is that mean that has to be the density estimate.
    The bandwidth, the ladder and the kernel are ``kink_kernel``'s; what stays here is the REFUSAL.

    AN ATOM IS DIAGNOSED ON THE BANDWIDTH LADDER, the only instrument that separates a point mass at
    the kink from a narrow density there: both have the same spread, near-zero mass and Silverman
    width, and only ``f_V(0)``'s behaviour under a varying width tells them apart. A row whose
    narrowest rung stands ``KINK_ATOM_LADDER_DIVERGENCE`` times its widest is REFUSED by name with
    its row index and readings - the gamma there is genuinely singular rather than merely noisy.

    Where the bandwidth collapses under ``KINK_ATOM_BANDWIDTH_FLOOR`` of the row's own scale the
    kernel is WRITTEN to zero rather than evaluated (0/0 forward, NaN backward). Both answers there
    are zero anyway: such a row's whole mass sits at one level, so ``V_theta`` is zero at the kink
    and the kernel underflows away from it.

    THE COLLATERALISED CASE this refusal was written for - a net matched inside its threshold,
    pinned at the kink with real ``V_theta`` - cannot reach here: a ``NettingCollateralSet`` with
    ``Collateralized: 'True'`` registers an ``MTABoundarySet``, so ``Hessian: 'Yes'`` is refused one
    step earlier in ``Credit_Monte_Carlo.execute``. Estimating through a margin period, which is
    what would smear such an atom into a density, is future work.

    ONE sample has no spread for a kernel width, as in ``boundary_weights``: zero width is returned
    and the row is classified on that.

    DECLARED LIMITATION: the O(eps^2) bandwidth bias shows on volga, so this is a gamma-and-vanna
    term rather than a volga one. Reference readings against Black 2.6521 / 0.1989: gamma 2.6415 and
    vanna 0.1992 at 400k paths, and +/- 0.13 per xVA row at 1024 paths, +/- 0.07 at 4096.
    """
    Vbar = V.detach()
    kernel, atom, rungs = kink_kernel(Vbar, n_paths_axis, 'exposure')

    if bool(atom.any()):
        rows, worst_row, reading, ladder, climb, span = _kink_atom_reading(atom, rungs)
        raise utils.SecondOrderRefused(
            'exposure_kink_term: reporting row(s) {} carry an ATOM of exposure at the kink - the '
            'density f_V(0) this term estimates does not settle as its bandwidth narrows, it '
            'CLIMBS. Row {} reads {} at {} x its own Silverman width: a factor of {:.2f} across a '
            'ladder of {:.0f}, which is the 1/bandwidth signature of a point mass rather than the '
            'plateau of a density. The gamma there is genuinely singular, not merely noisy, and a '
            'number would be a lie about a divergence. What works today: ask for the first-order '
            "block alone (Hessian: 'No'), which this term does not touch and which still reports "
            'grad_cva; or clip the reporting grid off the named row(s) if they are not rows you '
            'need; or drop the position that pins the net. A MARGIN PERIOD or a transfer THRESHOLD '
            'is what would smear an atom into a density, but that is not a book you can price here '
            'yet - a collateralised set registers an MTA boundary correction and is refused one '
            'step earlier - and estimating through one is the conditional-p mixture on the roadmap '
            '(Second-order flux at a JUMP), pinned to the stride.'.format(
                rows, worst_row, reading, ladder, climb, span))

    u = V - Vbar
    return 0.5 * kernel * u * u


class InnerMCRecompute(torch.autograd.Function):
    """A pricer's inner Monte Carlo as one node: simulated untaped, RE-simulated to differentiate.

    The autograd tape of every inner pricing is what does not fit in memory; the simulation itself
    is cheap. The forward runs under ``no_grad`` and keeps nothing but its inputs, and the backward
    re-runs the SAME callable under ``enable_grad``, contracting the cotangent through one block's
    graph at a time. Subsystem documentation: calc_lifecycle.md#recompute-inner-mc.

    THE COUNTER IS THE STORAGE. What is saved is where each random stream stood
    (``utils.rng_position``), never what it produced: Sobol draws are memoized, so rewinding the
    counter replays the same tensors; the plain generator's state is saved and restored, and the
    live position is put back after the replay.

    THE ADOPTER'S SHAPE: ``simulate(*bound)(*theta)`` - bound arguments are the block's shape, theta
    is every graph-carrying tensor (anything read from a closure differentiates as a constant). It
    returns a TUPLE whose element 0 is the marks and whose remaining elements are by-products the
    caller performs ONCE - settled cashflows, decision gaps, counterfactual branches - each a
    TOP-LEVEL element (a tensor nested inside a list is not an output and its gradient is silently
    zero). The callable must be PURE in everything but its return: a side effect fires a second time
    in the backward, and a cashflow booked inside is booked under ``no_grad``, severing the channel
    a collateralised exposure reads through ``C_ts_te``.

    A DECISION'S GAP IS AN OUTPUT when the simulation decided it - the boundary correction's
    coefficient then arrives as that output's cotangent. A decision taken on OUTER state keeps its
    own graph and registers outside the node.

    A registration held on ``shared`` makes a reference cycle through this node
    (shared -> boundary_sets -> gap -> node -> simulate -> shared); ``reset()`` clearing
    ``boundary_sets`` per batch is what frees it.

    SECOND DERIVATIVES ARE REFUSED. The replay is rooted at DETACHED copies of the saved inputs -
    what stops the inner ``autograd.grad`` double-counting the first derivative - so a second
    derivative taken through the node comes back partly ZERO. Ask for the second-order block with
    the switch off.
    """

    @staticmethod
    def forward(ctx, simulate, position, shared, *theta):
        # undefined cotangents stay undefined: most outputs are coefficients nothing
        # differentiates, and materializing zeros allocates in backward what this node avoids
        ctx.set_materialize_grads(False)
        with torch.no_grad():
            outputs = simulate(*theta)
        ctx.simulate, ctx.position, ctx.shared = simulate, position, shared
        ctx.save_for_backward(*theta)
        return outputs

    @staticmethod
    def backward(ctx, *cotangents):
        # ASKED HERE, not inside the `enable_grad` block below: the engine runs a backward node
        # with grad mode set to `create_graph`, and in there the answer is always True
        if torch.is_grad_enabled():
            raise Exception(
                'Recompute_Inner_MC: create_graph is not supported - the recompute is rooted at '
                'DETACHED copies of its inputs, so a second derivative taken through it is severed '
                'from the graph the outer pass holds and comes back partly zero. Ask for the '
                'second-order block with the switch off.')
        wanted = [i for i, t in enumerate(ctx.saved_tensors) if t.requires_grad]
        if not wanted:
            return (None, None, None) + (None,) * len(ctx.saved_tensors)
        theta = [t.detach() for t in ctx.saved_tensors]
        for i in wanted:
            theta[i].requires_grad_(True)
        with torch.enable_grad():
            live = utils.rng_position(ctx.shared, ctx.position)
            try:
                outputs = ctx.simulate(*theta)
            finally:
                utils.rng_position(ctx.shared, live)
            paired = [(out, cotangent) for out, cotangent in zip(outputs, cotangents)
                      if cotangent is not None and out.requires_grad]
            grads = torch.autograd.grad(
                [out for out, _ in paired], [theta[i] for i in wanted],
                [cotangent for _, cotangent in paired], allow_unused=True) if paired else ()
        theta_grads = [None] * len(theta)
        for i, grad in zip(wanted, grads):
            theta_grads[i] = grad
        return (None, None, None) + tuple(theta_grads)

    @classmethod
    def run(cls, shared, simulate, *theta):
        """The node, or not - the ONE line an adopting pricer writes.

        `Recompute_Inner_MC` is a property of the valuation engine and the machine it is on, so it
        governs every adopter at once and no pricer carries a flag of its own. Off, `simulate` is
        called and taped directly; on, it goes through the node with the RNG position taken at the
        call site, which is where the streams stand where the replay must find them.
        """
        return (cls.apply(simulate, utils.rng_position(shared), shared, *theta)
                if shared.recompute_inner_mc else simulate(*theta))


def cva_per_scenario(pv_exposure, prob, recovery):
    """CVA as a PER-SCENARIO vector, whose mean is the reported CVA.

    THE REDUCTION ORDER IS LOAD-BEARING: the reported number is a MEAN over paths of a SUM over
    time, so any boundary correction assembled against it must also be a mean over paths or it is
    silently scaled by the path count - silently, because such a correction has zero forward value
    and only the gradients would be wrong.
    """
    return ((1.0 - recovery) * 0.5 * (pv_exposure[1:] + pv_exposure[:-1]) * prob).sum(axis=0)


def fva_per_scenario(pv, cost_spread, benefit_spread):
    """FVA as a PER-SCENARIO vector, whose mean is the reported FCA_t - FBA_t.

    `pv` is SIGNED - the funding cost rides its positive part and the benefit its negative one -
    where `cva_per_scenario`'s exposure arrives already relu'd. Same load-bearing reduction order:
    a per-path vector of a sum over time.
    """
    plus, minus = torch.relu(pv), torch.relu(-pv)
    return (torch.sum(cost_spread * (plus[1:] + plus[:-1]) / 2, dim=0)
            - torch.sum(benefit_spread * (minus[1:] + minus[:-1]) / 2, dim=0))


class SensitivitiesEstimator(object):
    """AAD sensitivities of one value against a list of model parameters, first and second order."""

    def __init__(self, value, params, create_graph=False):
        """Run the backward pass over `value` and keep the gradient of every parameter that has
        one. `create_graph` retains the graph the second-order block differentiates through."""
        value.backward(retain_graph=True, create_graph=create_graph)
        self.params = {key: tensor for key, tensor in params if tensor.grad is not None}
        self.grad = {key: tensor.grad for key, tensor in self.params.items()}
        self.flat_grad = self.flatten(list(self.grad.values()))
        self.device = self.flat_grad.device
        self.dtype = self.flat_grad.dtype
        self.list_param = list(self.params.values())
        self.P = len(self.flat_grad)

    def report_grad(self):
        """Per-factor gradients as numpy arrays, COPIED off the live ``.grad`` buffers.

        ``np.array`` must copy: ``.numpy()`` on a CPU tensor returns a VIEW of the ``.grad`` buffer
        torch keeps accumulating into, so a later ``backward()`` through the same leaves would
        rewrite a report already handed out.
        """
        return {utils.check_scope_name(factor): np.array(tensor.cpu().detach())
                for factor, tensor in self.grad.items()}

    def report_hessian(self, allow_unused=False):
        """The full (P, P) Hessian as a numpy array, in the same flat factor order as
        ``report_grad`` concatenates - which is what lets the report label both of its axes off
        the first-order index.

        Assembled UPPER-TRIANGLE-ONLY and mirrored: ``get_H_op`` differentiates factor i's gradient
        against factors i onwards, so the blocks below the diagonal are never computed, and the
        strict lower triangle is wiped before the mirror so the diagonal block's own sub-diagonal is
        not counted twice. Symmetry is therefore exact by construction.

        ``allow_unused`` is what a portfolio needs and a single objective does not: a factor the
        value depends on LINEARLY has no second-order path at all, and autograd returns None rather
        than zeros for it.
        """
        h_op = self.get_H_op(allow_unused)
        hessian = np.zeros((self.P, self.P))

        j = 0
        for row in h_op:
            v, _ = row.shape
            hessian[j:j + v, j:] = row
            j += v

        hessian[np.tril_indices(self.P, k=-1)] = 0.0
        return hessian + np.triu(hessian, k=1).T

    def get_Hv_op(self, v):
        """The Hessian-vector product H @ v, flattened."""
        hv = self.flatten(torch.autograd.grad(
            self.flat_grad, self.list_param, grad_outputs=v, only_inputs=True, retain_graph=True))

        return hv

    @staticmethod
    def flatten(params):
        """Concatenate a list of tensors into one 1D tensor."""
        return torch.cat([_params.reshape(-1) for _params in params], dim=0)

    def get_H_op(self, allow_unused):
        """The Hessian's upper-triangular blocks, one per factor, as numpy arrays: factor i's
        gradient differentiated against factors i onwards. `allow_unused` substitutes zeros where
        autograd returns None for a factor the value depends on linearly."""

        glist = self.grad.values()
        plist = self.list_param
        e = {l: torch.eye(l, dtype=self.dtype, device=self.device) for l in set([len(g) for g in glist])}
        if allow_unused:
            z = {l: torch.zeros(l, dtype=self.dtype, device=self.device) for l in set([len(p) for p in plist])}

        hessian = []

        for i, g in enumerate(glist):
            g_size = len(g)
            row = []
            for block in [torch.autograd.grad(
                    g, plist[i:], grad_outputs=x, only_inputs=True,
                    allow_unused=allow_unused, retain_graph=True) for x in e[g_size]]:

                if allow_unused:
                    row.append(torch.cat([b if b is not None else z[len(p)] for b, p in zip(block, plist[i:])]))
                else:
                    row.append(torch.cat(block))

            hessian.append(torch.stack(row).cpu().detach().numpy())

        return hessian


def greeks(shared, deal_data, mtm):
    """Report `Greeks_First`, and `Greeks_Second` when `Greeks: 'All'` asked for it - in that
    order, because the second-order report labels its axes off the first-order index
    (`Calculation.gradients_as_df`)."""
    greeks_calc = SensitivitiesEstimator(mtm, shared.calc_greeks, create_graph=shared.gamma)
    deal_data.Calc_res['Greeks_First'] = greeks_calc.report_grad()
    # use this only when all the vols and curves are sparsely represented (check greeks_calc.P)
    if shared.gamma:
        deal_data.Calc_res['Greeks_Second'] = greeks_calc.report_hessian(allow_unused=True)
        # `create_graph` leaves every leaf holding a `.grad` whose graph reaches back to it, a cycle
        # refcounting cannot break; both reports are numpy copies by now. PARTIAL: leaves upstream
        # of what the estimator was offered keep their own cycle until gc gets there
        for tensor in greeks_calc.params.values():
            tensor.grad = None


def interp_to_mtm_grid(mtm, time_grid, deal_data, interpolate_grid=True):
    """Deal grid -> MTM grid: the purely LINEAR half of `interpolate`, with no result stashing.

    A boundary counterfactual has to put its branch values on exactly the grid the reported value
    landed on, so both go through this rather than through two copies of the gather and the pad.
    """
    if interpolate_grid and deal_data.Time_dep.interp.size > deal_data.Time_dep.deal_time_grid.size:
        mtm = utils.gather_interp_matrix(mtm, deal_data.Time_dep)

    if mtm.shape != (1, 1) and mtm.shape[0] < time_grid.mtm_time_grid.size:
        return F.pad(mtm, [0, 0, 0, time_grid.mtm_time_grid.size - deal_data.Time_dep.interp.size])
    else:
        return mtm


def deal_to_mtm_grid(time_grid, deal_data, fx_rep):
    """The map from a pricer's own output to the MTM-grid rows the deal reports on, detached.

    Everything ``Deal.generate``/``calculate`` do to a theo price and nothing else: scale into the
    reporting currency, then interpolate - fx BEFORE interpolation, since a boundary branch is an
    alternative value for the same deal.

    Two things this closure must NOT hold. ``fx_rep`` is detached AT CAPTURE, or a boundary set
    outliving the pricing call pins the deal's whole autograd graph until the batch's backward pass.
    And it does not take ``shared``, which would be a reference cycle
    (shared -> boundary_sets -> closure -> shared) refcounting cannot break.
    """
    scale = fx_rep.detach()
    return lambda profile: interp_to_mtm_grid(profile * scale, time_grid, deal_data).detach()


def interpolate(mtm, shared, time_grid, deal_data, interpolate_grid=True):
    """Deal grid -> MTM grid, stashing the interpolated value on the deal's result block."""
    if interpolate_grid and deal_data.Time_dep.interp.size > deal_data.Time_dep.deal_time_grid.size:
        mtm = utils.gather_interp_matrix(mtm, deal_data.Time_dep)

    if deal_data.Calc_res is not None:
        shared.save_results(deal_data.Calc_res, {'Value': mtm})
        if shared.keep_tensor:
            deal_data.Calc_res['tensor'] = mtm

    # the gather is already done - only the pad is left
    return interp_to_mtm_grid(mtm, time_grid, deal_data, interpolate_grid=False)


def getbarrierpayoff(direction, eta, phi, strike, H):
    '''
    Build the single-barrier payoff function for one (direction, eta, phi, strike, H) combination,
    by Merton, Reiner and Rubinstein. The branches below select sums of these six terms, which is
    what their labels name:

    A = phi * ( spot * sympy.exp ( (b-r)*expiry ) * normcdf ( phi * x1 ) -
            strike * sympy.exp( -r*expiry ) * normcdf ( phi * ( x1 - vol ) ) )
    B = phi * ( spot * sympy.exp ( (b-r)*expiry ) * normcdf ( phi * x2 ) -
            strike * sympy.exp( -r*expiry ) * normcdf ( phi * ( x2 - vol ) ) )
    C = phi * ( spot * sympy.exp ( (b-r)*expiry + log_bar*2*(mu+1) ) * normcdf (eta*y1) -
            strike * sympy.exp ( -r*expiry + log_bar*2*mu ) * normcdf ( eta * ( y1 - vol ) ) )
    D = phi * ( spot * sympy.exp ( (b-r)*expiry + log_bar*2*(mu+1) ) * normcdf (eta*y2) -
            strike * sympy.exp ( -r*expiry + log_bar*2*mu ) * normcdf ( eta * ( y2 - vol ) ) )
            
    E = cash_rebate * sympy.exp ( -r*expiry ) * ( normcdf ( eta * ( x2 - vol ) ) -
        sympy.exp(log_bar*2*mu) * normcdf ( eta * ( y2 - vol ) ) )
    F = cash_rebate * ( sympy.exp ( log_bar*(mu+lam) ) * normcdf (eta*z) +
        sympy.exp (log_bar*(mu-lam)) * normcdf ( eta * ( z - 2*lam*vol) ) )
    '''

    def barrier_option(sigma, expiry, cash_rebate, b, r, spot, barrier):

        sigma2 = sigma * sigma
        vol = sigma * torch.sqrt(expiry)
        mu = (b - 0.5 * sigma2) / sigma2
        log_bar = torch.log(barrier / spot)
        x1 = torch.log(spot / strike) / vol + (1.0 + mu) * vol
        x2 = torch.log(spot / barrier) / vol + (1.0 + mu) * vol

        y1 = torch.log((barrier * barrier) / (spot * strike)) / vol + (1.0 + mu) * vol
        y2 = log_bar / vol + (1.0 + mu) * vol
        lam = torch.sqrt(mu * mu + 2.0 * r / sigma2)
        z = log_bar / vol + lam * vol
        eta_scale = 0.7071067811865476 * eta
        phi_scale = 0.7071067811865476 * phi
        expiry_r = expiry * r

        if direction == BARRIER_IN:
            if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > H) or
                    (phi == OPTION_PUT and eta == BARRIER_DOWN and strike <= H)):
                # A+E
                return (cash_rebate * (
                        (0.5 * torch.erfc(eta_scale * (-vol + y2)) - 1.0) * torch.exp(2 * log_bar * mu) +
                        0.5 * torch.erfc(eta_scale * (vol - x2))) - phi * (
                                spot * (0.5 * torch.erfc(phi_scale * x1) - 1.0) * torch.exp(b * expiry) +
                                0.5 * strike * torch.erfc(phi_scale * (vol - x1)))) * torch.exp(-expiry_r)
            elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= H) or
                  (phi == OPTION_PUT and eta == BARRIER_DOWN and strike > H)):
                # B-C+D+E
                return (cash_rebate * (
                        (0.5 * torch.erfc(eta_scale * (-vol + y2)) - 1.0) * torch.exp(2 * log_bar * mu) +
                        0.5 * torch.erfc(eta_scale * (vol - x2))) - phi * (
                                spot * (0.5 * torch.erfc(phi_scale * x2) - 1.0) * torch.exp(b * expiry) +
                                0.5 * strike * torch.erfc(phi_scale * (vol - x2))) + phi * (
                                spot * (0.5 * torch.erfc(eta_scale * y1) - 1.0) * torch.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1)) - spot * (
                                        0.5 * torch.erfc(eta_scale * y2) - 1.0) *
                                torch.exp(expiry * (b - r) + 2 * log_bar * (mu + 1)) + 0.5 * strike * torch.exp(
                            -expiry_r + 2 * log_bar * mu) * torch.erfc(eta_scale * (vol - y1)) -
                                0.5 * strike * torch.exp(-expiry_r + 2 * log_bar * mu) * torch.erfc(
                            eta_scale * (vol - y2))) * torch.exp(expiry_r)) * torch.exp(-expiry_r)
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > H) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike <= H)):
                # A-B+D+E
                return (cash_rebate * ((0.5 * torch.erfc(eta_scale * (-vol + y2)) - 1.0) * torch.exp(
                    2 * log_bar * mu) + 0.5 * torch.erfc(eta_scale * (vol - x2))) - phi * (
                                spot * (0.5 * torch.erfc(eta_scale * y2) - 1.0) * torch.exp(expiry * (
                                b - r) + 2 * log_bar * (mu + 1)) + 0.5 * strike * torch.exp(
                            -expiry_r + 2 * log_bar * mu) * torch.erfc(
                            eta_scale * (vol - y2))) * torch.exp(expiry_r) - phi * (spot * (0.5 * torch.erfc(
                    phi_scale * x1) - 1.0) * torch.exp(b * expiry) + 0.5 * strike * torch.erfc(
                    phi_scale * (vol - x1))) + phi * (spot * (0.5 * torch.erfc(
                    phi_scale * x2) - 1.0) * torch.exp(b * expiry) + 0.5 * strike * torch.erfc(
                    phi_scale * (vol - x2)))) * torch.exp(-expiry_r)
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike > H)):
                # C+ E
                return (cash_rebate * ((0.5 * torch.erfc(eta_scale * (-vol + y2)) - 1.0) * torch.exp(
                    2 * log_bar * mu) + 0.5 * torch.erfc(eta_scale * (vol - x2))) - phi * (spot * (
                        0.5 * torch.erfc(eta_scale * y1) - 1.0) * torch.exp(expiry * (b - r) + 2 * log_bar * (
                        mu + 1)) + 0.5 * strike * torch.exp(-expiry_r + 2 * log_bar * mu) * torch.erfc(eta_scale * (
                        vol - y1))) * torch.exp(expiry_r)) * torch.exp(-expiry_r)
        else:
            if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > H) or
                    (phi == OPTION_PUT and eta == BARRIER_DOWN and strike <= H)):
                # F
                return -cash_rebate * ((0.5 * torch.erfc(eta_scale * z) - 1.0) * torch.exp(2 * lam * log_bar) -
                                       0.5 * torch.erfc(eta_scale * (2 * lam * vol - z))) * torch.exp(
                    -log_bar * (lam - mu))
            elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= H) or
                  (phi == OPTION_PUT and eta == BARRIER_DOWN and strike > H)):
                # A - B + C - D + F
                return (-cash_rebate * ((0.5 * torch.erfc(eta_scale * z) - 1.0) * torch.exp(2 * lam * log_bar)
                                        - 0.5 * torch.erfc(eta_scale * (2 * lam * vol - z))) * torch.exp(expiry_r)
                        + phi * (-spot * (0.5 * torch.erfc(eta_scale * y1) - 1.0) * torch.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1)) + spot * (
                                         0.5 * torch.erfc(eta_scale * y2) - 1.0) *
                                 torch.exp(expiry * (b - r) + 2 * log_bar * (mu + 1)) - 0.5 * strike * torch.exp(
                                    -expiry_r + 2 * log_bar * mu) *
                                 torch.erfc(eta_scale * (vol - y1)) + 0.5 * strike * torch.exp(
                                    -expiry_r + 2 * log_bar * mu) *
                                 torch.erfc(eta_scale * (vol - y2))) * torch.exp(expiry_r + log_bar * (lam - mu)) +
                        phi * (-spot * (0.5 * torch.erfc(phi_scale * x1) - 1.0) * torch.exp(b * expiry) + spot * (
                                0.5 * torch.erfc(phi_scale * x2) - 1.0) * torch.exp(b * expiry) - 0.5 * strike *
                               torch.erfc(phi_scale * (vol - x1)) + 0.5 * strike * torch.erfc(phi_scale * (vol - x2)))
                        * torch.exp(log_bar * (lam - mu))) * torch.exp(-expiry_r - log_bar * (lam - mu))
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > H) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike <= H)):
                # B - D + F
                return (-cash_rebate * ((0.5 * torch.erfc(eta_scale * z) - 1.0) * torch.exp(2 * lam * log_bar)
                                        - 0.5 * torch.erfc(eta_scale * (2 * lam * vol - z))) * torch.exp(expiry_r) +
                        phi * (spot * (0.5 * torch.erfc(eta_scale * y2) - 1.0) * torch.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1))
                               + 0.5 * strike * torch.exp(-expiry_r + 2 * log_bar * mu) * torch.erfc(
                                    eta_scale * (vol - y2))) * torch.exp(
                            expiry_r + log_bar * (lam - mu)) - phi * (
                                spot * (0.5 * torch.erfc(phi_scale * x2) - 1.0) * torch.exp(
                            b * expiry) + 0.5 * strike * torch.erfc(phi_scale * (vol - x2))) * torch.exp(
                            log_bar * (lam - mu))) * torch.exp(
                    -expiry_r - log_bar * (lam - mu))
            elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or
                  (phi == OPTION_CALL and eta == BARRIER_DOWN and strike > H)):
                # A-C+F
                return (-cash_rebate * ((0.5 * torch.erfc(eta_scale * z) - 1.0) * torch.exp(2 * lam * log_bar)
                                        - 0.5 * torch.erfc(eta_scale * (2 * lam * vol - z))) * torch.exp(expiry_r) +
                        phi * (spot * (0.5 * torch.erfc(eta_scale * y1) - 1.0) * torch.exp(
                            expiry * (b - r) + 2 * log_bar * (mu + 1))
                               + 0.5 * strike * torch.exp(-expiry_r + 2 * log_bar * mu) * torch.erfc(
                                    eta_scale * (vol - y1))) * torch.exp(
                            expiry_r + log_bar * (lam - mu)) - phi * (
                                spot * (0.5 * torch.erfc(phi_scale * x1) - 1.0) * torch.exp(
                            b * expiry) + 0.5 * strike * torch.erfc(phi_scale * (vol - x1))) * torch.exp(
                            log_bar * (lam - mu))) * torch.exp(
                    -expiry_r - log_bar * (lam - mu))

    return barrier_option


def getpartialbarrierpayoff(isKnockIn, eta, phi, spot, strike, barrier, startBarrier, limit, expiry, r, b, sigma):
    '''Value a partial-time barrier option: the barrier is monitored only over the window that
    starts (`startBarrier`) or ends at `limit`. Knock-ins are priced by parity off the vanilla.'''

    def BarrierPutCallTransformation(spot, strike, barrier, r, b, upDown):
        return strike, spot, spot * strike / barrier, r - b, -b, -upDown

    def PartialBarrierCalc(forward, strike, log1, log2, rho1, rho2, p1, p2, p3, p4, p5, p6, p7, p8):
        return (forward * (utils.ApproxBivN(p1, p2, rho1) - torch.exp(log1) * utils.ApproxBivN(p3, p4, rho2)) -
                strike * (utils.ApproxBivN(p5, p6, rho1) - torch.exp(log2) * utils.ApproxBivN(p7, p8, rho2)))

    def partial_barrier_option(spot, strike, barrier, r, b, eta):
        rho = torch.sqrt(limit / expiry)
        vol = sigma * torch.sqrt(expiry)
        vollimit = sigma * torch.sqrt(limit)
        halfvv = 0.5 * sigma * sigma
        logSpotOverStrike = torch.log(spot / strike)
        logSpotOverBarrier = torch.log(spot / barrier)
        d1 = (logSpotOverStrike + (b + halfvv) * expiry) / vol
        d2 = d1 - vol
        f1 = (logSpotOverStrike - 2.0 * logSpotOverBarrier + (b + halfvv) * expiry) / vol
        f2 = f1 - vol
        e1 = (logSpotOverBarrier + (b + halfvv) * limit) / vollimit
        e2 = e1 - vollimit
        e3 = e1 - 2.0 * logSpotOverBarrier / vollimit
        e4 = e3 - vollimit
        mu2 = b / halfvv - 1.0
        forward = spot * torch.exp(b * expiry)
        log1 = -logSpotOverBarrier * (mu2 + 2.0)
        log2 = -logSpotOverBarrier * mu2

        if startBarrier:
            etaRho = eta * rho
            pv = PartialBarrierCalc(
                forward, strike, log1, log2, etaRho, etaRho, d1, e1 * eta,
                f1, e3 * eta, d2, e2 * eta, f2, e4 * eta)
        else:
            g1 = (logSpotOverBarrier + (b + halfvv) * expiry) / vol
            g2 = g1 - vol
            g3 = g1 - 2.0 * logSpotOverBarrier / vol
            g4 = g3 - vol

            if eta == 0:  # type B1

                pv = PartialBarrierCalc(
                    forward, strike, log1, log2, rho, -rho, d1, e1, f1, -e3, d2, e2, f2, -e4)
                temp = PartialBarrierCalc(
                    forward, strike, log1, log2, rho, -rho, g1, e1, g3, -e3, g2, e2, g4, -e4)
                temp += PartialBarrierCalc(
                    forward, strike, log1, log2, rho, -rho, -g1, -e1, -g3, e3, -g2, -e2, -g4, e4)
                temp -= PartialBarrierCalc(
                    forward, strike, log1, log2, rho, -rho, -d1, -e1, -f1, e3, -d2, -e2, -f2, e4)

                # `pv` is the X >= H expression and `temp` the X < H one
                pv = torch.where(torch.as_tensor(strike < barrier, device=forward.device),
                                 temp, pv)
            else:
                if eta == 1:

                    pv = PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, g1, e1, g3, -e3, g2, e2, g4, -e4)
                    temp = PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, d1, e1, f1, -e3, d2, e2, f2, -e4)

                    pv = torch.where(torch.as_tensor(strike < barrier, device=forward.device),
                                     pv, temp)

                else:  # eta == -1

                    pv = PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, -g1, -e1, -g3, e3, -g2, -e2, -g4, e4)
                    pv -= PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, -d1, -e1, e3, -f1, -d2, -e2, e4, -f2)
                    pv *= (strike < barrier)

        return pv * torch.exp(-r * expiry)

    if isKnockIn:
        bs = utils.black_european_option(
            spot * torch.exp(b * expiry), strike, sigma * torch.sqrt(expiry),
            1.0, 1.0, phi, None) * torch.exp(-r * expiry)

    if phi == -1:
        spot, strike, barrier, r, b, eta = BarrierPutCallTransformation(spot, strike, barrier, r, b, eta)

    pv = partial_barrier_option(spot, strike, barrier, r, b, eta)

    if isKnockIn:
        return bs - pv
    else:
        return pv


def compo_vol(vols, fx_vols, rho):
    """Vol of the PRODUCT S*X from its two lognormal legs. Exact for the joint law; the simulating
    pricers apply it per INTERVAL with the fx leg read at expiry, which is the same deal-level
    fidelity as the quanto carry adjustment beside it."""
    return torch.sqrt(vols * vols + 2.0 * rho * vols * fx_vols + fx_vols * fx_vols)


def calc_vol_adjustment(factor_dep, deal_time, expiry, vols, shared, fixings=None):
    """The quanto/compo adjustment, factored for BOTH consumer shapes.

    A forward-based pricer (`pv_european_option`) rebuilds its terminal law as
    `s_adj * forward * exp(b_adj * T)` - quanto bends the carry, compo swaps the underlying for S*X
    so its fx half arrives as the outright fx forward. A SIMULATING pricer takes the same
    information factored the way a path steps: `spot_scale` (the fx cross, 1 for quanto),
    `carry_adj` (per-fixing, built only when `fixings` is passed), and `fx_vol`/`rho` to compose its
    interval vol strip (None for quanto, whose measure change touches no vol).
    """
    # None means get the ATM vol for this expiry (can change depending on the vol surface type)
    fx_vols = utils.calc_time_grid_vol_rate(factor_dep['FXVol'], None, expiry, shared)

    if 'QuantoImpliedCorrelation' in factor_dep:
        # quanto fx deal
        rho = utils.implied_correlation(
            factor_dep['QuantoImpliedCorrelation'], factor_dep['Correlation_Sign'])
        atm_vol = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], None, expiry, shared)
        b_adj = -atm_vol * fx_vols * rho
        return {'vol': vols, 'b_adj': b_adj, 's_adj': 1.0, 'spot_scale': 1.0,
                'carry_adj': b_adj.unsqueeze(1) if fixings is not None else None,
                'fx_vol': None, 'rho': rho}
    else:
        rho = utils.implied_correlation(
            factor_dep['CompoImpliedCorrelation'], factor_dep['Correlation_Sign'])
        forwardfx = utils.calc_fx_forward(
            factor_dep['Local'], factor_dep['Other'],
            int(factor_dep['Expiry']), deal_time, shared)
        return {'vol': compo_vol(vols, fx_vols, rho),
                'b_adj': torch.zeros_like(fx_vols), 's_adj': forwardfx,
                'spot_scale': utils.calc_fx_cross(
                    factor_dep['Local'][0], factor_dep['Other'][0], deal_time, shared),
                'carry_adj': utils.calc_fx_drift(
                    factor_dep['Local'], factor_dep['Other'], fixings, deal_time, shared,
                    multiply_by_time=False) if fixings is not None else None,
                'fx_vol': fx_vols, 'rho': rho}


def smooth_max(x, k, eps=0.01):
    return torch.where(x < k - eps, torch.zeros_like(x), torch.where(
        x > k + eps, x - k, (-k * x + 0.5 * x ** 2 + eps * x + 0.5 * (k - eps) ** 2) / (2 * eps)))


def smooth_relu(x, eps=0.005):
    return torch.where(x < - eps, torch.zeros_like(x), torch.where(
        x > eps, x, (0.5 * x ** 2 + eps * x + 0.5 * eps ** 2) / (2 * eps)))


def smooth_heaviside_up(x, k, eps=0.01):
    return torch.where(x < k - eps, torch.zeros_like(x),
                       torch.where(x > k + eps, torch.ones_like(x), 0.5 + (x - k) / (2.0 * eps)))


def smooth_heaviside_down(x, k, eps=0.01):
    return torch.where(x < k - eps, torch.ones_like(x),
                       torch.where(x > k + eps, torch.zeros_like(x), 0.5 + (k - x) / (2.0 * eps)))


def pv_discrete_barrier_option(shared, time_grid, deal_data, spot, b, tau, fx_rep,
                               invert_moneyness=False, use_forwards=False, isdigital=False):
    """One-step-survival (OSS) Monte Carlo for a discretely monitored barrier option.

    At each observation date the analytic survival probability contributes the knock-out term
    exactly and only surviving paths draw a truncated step - differentiable end to end (no
    smoothing) and free of the variance paths near the barrier would contribute. ``BARRIER_IN`` is
    priced by in-out parity, so its vanilla leg is valued from the block spot forwarded by
    ``total_log_forward`` in the DECLARED model - Black, or the Heston-Nandi closed form.

    Reads. The carry arrives as the interval strip (``forward_carry_rate``) and the vol as TWO
    quantities: the per-interval strip (``forward_vol_rate(forward_vol_strip(...))``) the simulation
    steps on, and ``sd_to_expiry = sigma(K, tau) * sqrt(tau)`` for the European legs - two
    legitimate reads of one surface, deliberately not unified. The strip reads the moneyness the
    deal declares (``use_forwards``); the per-fixing smile convention is an open modelling question
    (roadmap.md). Under Heston-Nandi neither vol quantity is built, so a leg that reaches for an
    implied surface fails on the shape rather than quietly pricing a second model. Quanto bends the
    per-fixing carry; compo prices the product S*X (``calc_vol_adjustment``).

    Conventions this loop owns:

    - ``Cash_Rebate`` is an absolute cash amount and enters PER-UNIT, divided by the UNSIGNED size:
      ``nominal`` already carries ``Buy_Sell``, so dividing by it would book a sold knock-out's
      rebate as a receipt.
    - The per-row hit mask carries STRICTLY EARLIER observations only. The block's own date is
      priced by ``sim_spot_oss`` as its zero-length first step, so folding it in would count the
      same crossing twice.
    - The rebate settles on the date it falls due, excluding the TERMINAL row - the single settle
      after the loop already pays that one in full inside ``mtm_list[-1][-1]``.
    - ONE model and ONE forward price every leg: the already-hit KI vanilla runs the parity leg's
      own algebra on the parity leg's own discretisation (and raises the same batch-constant-carry
      refusal), and both legs read ``total_log_forward``. The variance floor binds only on the
      zero-length first interval.

    BOUNDARY AAD (``shared.boundary_aad``): the gap is each crossing's margin, graph retained,
    signed so ``gap > 0`` means CROSSED - a down barrier is crossed from above, so its gap is
    ``log(barrier/spot)``. The whole latch is built OUTSIDE the recompute node, the decision being
    an outer scenario spot. Blocks where every scenario has resolved skip the OSS and carry no
    counterfactual - running one would draw random numbers and move the reported value.
    ``sim_spot_oss`` is pure, so the recompute port only hoists the Heston-Nandi scalars into theta.

    BRANCH AND WEIGHT (``Branch_And_Weight: 'Yes'``, base valuation only) SWAPS the estimator for
    that registration rather than joining it. Off, and absent, is every line above bit for bit.

    THIS SAMPLER IS ALREADY THE SMOOTH ESTIMATOR: every monitored step is an analytic ``p`` with a
    survival-truncated continuation, the knock-out's rebate is paid analytically at the fixing it
    falls due (``(1 - p) * L * rebate * D_j``), the knock-IN's is the same survival weight carried to
    expiry on the parity leg, and a digital's terminal step is INTEGRATED. No indicator is formed on
    the simulated tape, so the value under the switch is bit-identical. What changes is what a run
    may then do with it:

    - THE OUTER REGISTRATION IS SKIPPED, so ``Greeks: 'All'`` flows where it used to refuse. Base
      valuation - the one calculation declaring the switch - has ONE scenario, so
      ``boundary_weights`` finds no spread for a kernel width and the correction is exactly zero.
    - THE TERMINAL EUROPEAN'S KINK gets its second-order term (``accrual_kink_term``), that relu
      being the only place left where pathwise AAD would answer a gamma of exactly zero. It is added
      to ``surv_payoff``, BEFORE the in-out parity subtraction, so both legs carry one corrected
      quantity and parity holds at second order rather than only at value.

    IN-OUT PARITY IS THIS PRICER'S CONSERVATION STATEMENT and is exact on the smooth path:
    ``KO + KI == vanilla + rebate`` in a zero-rate world, at VALUE, DELTA and GAMMA. The rebate half
    is the ``SurvivalLedger`` identity read off two prices: the knock-out pays
    ``sum_j fired_j * D_j`` of it and the knock-in ``alive_T * D_T``, so their sum is the rebate
    exactly when the telescoping holds.

    DECLARED LIMITATION at second order: a LIVE monthly down-and-out reports gamma within ~0.6% of a
    CRN ladder of its own corrected delta at 32768 paths, at ~2% ladder flatness; the never-knocking
    control lands within ~1.4% of Black.
    """

    def sim_spot_oss(offset, sobol, num_sims,
                     spot_prices, vols, sd_to_expiry, times, carry, discount_rates, *hn_scalars):
        """Run the OSS inner Monte Carlo and return the per-block mean PV as the one-element tuple
        ``(mtm,)`` - the tuple is ``InnerMCRecompute``'s contract.

        PURE bound/theta split: the leading arguments are the block's shape, bound per block with
        ``partial``; the trailing ones are every graph-carrying tensor the node can return a gradient
        for. It takes BOTH vol quantities - ``vols`` the per-interval strip ``[N_block, N_fix, batch]``
        the simulation steps on, ``sd_to_expiry`` the European standard deviation ``[N_block, batch]``
        the parity vanilla is valued at.

        Under Heston-Nandi the KI parity vanilla is the HN closed form over ``n_total`` daily steps
        with a scalar per-step carry, valid only while the carry is batch-constant - hence the loud
        refusal under stochastic rates.

        A digital's TERMINAL step is INTEGRATED, not sampled: an indicator on a drawn spot has zero
        derivative almost everywhere, so sampling it loses the density term that is most of a
        digital's delta and vega. The HN branch keeps the indicator - its per-path conditional
        variance has no scalar closed form to integrate against.
        """
        eps = torch.finfo(shared.one.dtype).eps
        # the declared model as a walking kit - the pricer names it once, here
        kit = oss_model_kit(factor_dep, hn_scalars)
        hn = kit is not None

        # `carry` is the INTERVAL carry rate (forward_carry_rate), so this product is the interval
        # integral. GBM folds it with the vol; HN keeps it raw, its recursion supplying variance
        dt = times.unsqueeze(axis=2)                        # [N_block, N_fix, 1]
        carry_int = carry * dt                              # [N_block, N_fix, batch]
        if not hn:
            # `vols` is the INTERVAL vol strip (forward_vol_rate), so this is interval variance
            var = vols * vols * dt                          # [N_block, N_fix, batch]
            # the floor is the ZERO-LENGTH interval's survival width and only that; both consumers
            # read the SAME variance, so every interval is a martingale under its own drawn law
            var = torch.where(dt > 0, var, var.clamp(min=1e-4))
            drift = carry_int - 0.5 * var                   # [N_block, N_fix, batch]
            vol = torch.sqrt(var)                           # [N_block, N_fix, batch]

        isBarrierDate_block = BarrierDates[offset:]

        mcmc = []
        for blk, (s, D) in enumerate(zip(spot_prices, discount_rates)):
            # s: [batch], D: [N_fix, batch]
            N_fix = D.shape[0]
            if not hn:
                r, sigma = drift[blk], vol[blk]  # [N_fix, batch] each

            # antithetic variates: [N_fix, batch, 2*num_sims] (shared OSS spelling, bit-identical)
            u = oss_uniforms(shared, N_fix, num_sims, sobol)

            D_T = D[-1].reshape(-1, 1)  # terminal discount: [batch, 1]

            if hn:
                # per-interval daily sub-step counts and per-step carry (r-q). n_sub floors at 1.
                nj = [max(int(round(float(t) * hn_spy)), 1) for t in times[blk]]
                b_steps = [(carry_int[blk][j] / nj[j]).reshape(-1, 1) for j in range(N_fix)]
                st = kit.seed()

            if direction == BARRIER_IN:
                if not hn:
                    # the parity vanilla: KI = Vanilla - KO_pure + rebate * E[L_T]. ONE VOL prices
                    # every European leg - the same `sd_to_expiry` the already-hit leg marks with,
                    # not the strip's own total
                    fwd_to_T = s * torch.exp(total_log_forward(carry[blk], times[blk]))
                    vol_to_T = sd_to_expiry[blk]
                    if isdigital:
                        vanilla_pv = utils.black_european_option(
                            fwd_to_T, strike, vol_to_T, 1.0, 1.0, phi, shared, cash_payoff=1.0) * D[-1]
                    else:
                        vanilla_pv = utils.black_european_option(
                            fwd_to_T, strike, vol_to_T, 1.0, 1.0, phi, shared) * D[-1]
                else:
                    # the HN closed form; a scalar r_step needs batch-constant carry, so the guard
                    # is loud rather than silent
                    n_total = int(sum(nj))
                    carry_total = total_log_forward(carry[blk], times[blk]).reshape(-1)
                    if float(carry_total.detach().max() - carry_total.detach().min()) > 1.0e-9:
                        raise ValueError(
                            'HN KI closed-form leg needs batch-constant carry; carry varies across '
                            'scenarios by {:.2e} (stochastic rates?) - extend hn_call to batched '
                            'carry or price this deal under GBM'.format(
                                float(carry_total.max() - carry_total.min())))
                    r_step = carry_total[0] / n_total  # scalar b*T/n
                    if isdigital:
                        q_below = kit.cdf_logret(torch.log(strike / s), n_total, r_step)
                        vanilla_pv = ((1.0 - q_below) if phi == OPTION_CALL else q_below) * D[-1]
                    else:
                        fwd_growth = torch.exp(r_step * n_total)
                        vanilla = kit.vanilla(s, strike, n_total, r_step, phi == OPTION_CALL)
                        vanilla_pv = vanilla * fwd_growth * D[-1]
                vanilla_pv = vanilla_pv.reshape(-1, 1)  # [batch, 1]

            P = shared.one.new_zeros(shared.simulation_batch, 2 * num_sims)
            L = shared.one.new_ones(shared.simulation_batch, 2 * num_sims)
            surv_payoff = None
            # initialise Sj by broadcasting spot into [batch, 2*num_sims]
            Sj = s.reshape(-1, 1) + P

            for j in range(N_fix):
                if hn:
                    b_step = b_steps[j]
                    if isBarrierDate_block[j] > 0:
                        # nj-1 unmonitored daily steps + 1 monitored: the OSS truncation is at the
                        # CONSTANT barrier, F-measurable over the interval, so the scheme is exact
                        Sj, st = kit.substeps(
                            Sj, st, b_step, nj[j] - 1, shared, num_sims, antithetic=True)
                        h = kit.h(st)
                        z_max = (torch.log(barrier / Sj) - (b_step - 0.5 * h)) / torch.sqrt(h)
                        if eta == BARRIER_UP:
                            p = utils.norm_cdf(z_max)
                            Z = utils.norm_icdf(torch.clamp(u[j] * p, eps, 1.0 - eps))
                        else:
                            p = 1.0 - utils.norm_cdf(z_max)
                            Z = utils.norm_icdf(torch.clamp((1.0 - p) + u[j] * p, eps, 1.0 - eps))
                        if direction == BARRIER_OUT:
                            P = P + (1.0 - p) * L * rebate_per_unit * D[j].reshape(-1, 1)
                        L = p * L
                        Sj, st = kit.advance(Sj, st, b_step, Z)
                    else:
                        # non-barrier observation date (incl. expiry): full nj unconditional steps
                        Sj, st = kit.substeps(
                            Sj, st, b_step, nj[j], shared, num_sims, antithetic=True)
                    continue

                r_j = r[j].reshape(-1, 1)      # [batch, 1]
                sig_j = sigma[j].reshape(-1, 1) # [batch, 1]

                if isdigital and j == N_fix - 1:
                    # integrate the terminal step: a sampled indicator puts no density term on the
                    # tape, which is most of a digital's delta and vega
                    z_pay = (torch.log(strike / Sj) - r_j) / sig_j
                    lo, hi = (z_pay, None) if phi == OPTION_CALL else (None, z_pay)
                    if isBarrierDate_block[j] > 0:
                        z_max = (torch.log(barrier / Sj) - r_j) / sig_j
                        if eta == BARRIER_UP:
                            p = utils.norm_cdf(z_max)
                            hi = z_max if hi is None else torch.minimum(hi, z_max)
                        else:
                            p = 1.0 - utils.norm_cdf(z_max)
                            lo = z_max if lo is None else torch.maximum(lo, z_max)
                        if direction == BARRIER_OUT:
                            P = P + (1.0 - p) * L * rebate_per_unit * D[j].reshape(-1, 1)
                    # joint P(pays AND survives this fixing); survival is inside it, so it weights
                    # the L carried IN, while L itself still advances for the parity rebate leg
                    joint = ((utils.norm_cdf(hi) if hi is not None else 1.0) -
                             (utils.norm_cdf(lo) if lo is not None else 0.0)).clamp(min=0.0)
                    surv_payoff = L * joint
                    if isBarrierDate_block[j] > 0:
                        L = p * L
                    continue

                if isBarrierDate_block[j] > 0:
                    # z_max: the standard-normal threshold at which spot = H, under
                    # log(S_j/S_{j-1}) = r_j + sig_j * Z (r_j already carries -0.5*var)
                    z_max = (torch.log(barrier / Sj) - r_j) / sig_j  # [batch, 2*num_sims]

                    if eta == BARRIER_UP:
                        # survive = spot ends below H => Z < z_max
                        p = utils.norm_cdf(z_max)
                        Z = utils.norm_icdf(torch.clamp(u[j] * p, eps, 1.0 - eps))
                    else:
                        # survive = spot ends above H => Z > z_max
                        p = 1.0 - utils.norm_cdf(z_max)
                        Z = utils.norm_icdf(torch.clamp((1.0 - p) + u[j] * p, eps, 1.0 - eps))

                    if direction == BARRIER_OUT:
                        # paths hitting the barrier pay the cash rebate at this fixing
                        P = P + (1.0 - p) * L * rebate_per_unit * D[j].reshape(-1, 1)

                    L = p * L
                else:
                    # non-barrier observation date: unrestricted GBM step
                    Z = utils.norm_icdf(torch.clamp(u[j], eps, 1.0 - eps))

                Sj = Sj * torch.exp(r_j + sig_j * Z)

            # terminal payoff on paths that survived all barrier dates; surv_payoff is already
            # L-weighted when the last step was integrated rather than sampled (GBM digitals)
            if surv_payoff is None:
                gain = phi * (Sj - strike)
                payoff = (gain > 0).to(D_T.dtype) if isdigital else torch.relu(gain)
                if second_order and not isdigital:
                    # the ONE kink this loop still samples: every barrier decision above is already
                    # a `Phi`, and a digital's terminal step is integrated rather than sampled
                    payoff = payoff + accrual_kink_term(gain, N_fix - 1)
                surv_payoff = L * payoff

            if direction == BARRIER_OUT:
                # survivors receive the vanilla payoff at expiry
                P = P + surv_payoff * D_T
            else:
                # in-out parity: KI = Vanilla - KO_pure + rebate * E[survival]. Individual paths
                # can go negative - P is a control-variate estimator, not a per-path price - but
                # the option value is always >= 0, so the mean is clamped
                P = vanilla_pv - D_T * (surv_payoff - L * rebate_per_unit)

            mcmc.append(P.mean(dim=1).clamp(min=0.0))

        return (torch.stack(mcmc),)

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    samples = factor_dep['Observation_Dates']
    start_idx = samples.get_start_index(deal_time)
    dual_samples = samples.dual()
    start_index, counts = np.unique(start_idx, return_counts=True)

    BarrierDates = factor_dep['Barrier_Dates']
    phi = OPTION_CALL if deal_data.Instrument.field['Option_Type'] == 'Call' else OPTION_PUT
    eta = BARRIER_DOWN if 'Down' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_UP
    direction = BARRIER_OUT if 'Out' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_IN
    barrier = deal_data.Instrument.field['Barrier_Price']
    strike = deal_data.Instrument.field['Strike_Price']
    cash_rebate = deal_data.Instrument.field.get('Cash_Rebate', 0.0)

    size = deal_data.Instrument.field['Cash_Payoff'] if isdigital else deal_data.Instrument.field['Units']
    nominal = factor_dep['Buy_Sell'] * size

    # per-unit because sim_spot_oss scales by nominal; the UNSIGNED size, or Buy_Sell (already
    # inside nominal) cancels and a sold knock-out books the rebate it pays as a receipt
    rebate_per_unit = cash_rebate / size

    # the declared model's parameter tensors, by ITS OWN canonical name tuple; () is GBM
    hn_scalars = oss_model_scalars(factor_dep, shared)
    hn = bool(hn_scalars)
    if hn:
        hn_spy = factor_dep['HN_Steps_Per_Year']


    sobol = False
    if shared.simulation_batch > 16:
        sobol = True
        shared.reset_qrg()

    expiry = daycount_fn(tau)
    expiry_years_key = ('Expiry_Years', tuple(expiry))
    if expiry_years_key not in factor_dep:
        factor_dep[expiry_years_key] = spot.new(expiry.reshape(-1, 1))

    expiry_years = factor_dep[expiry_years_key]
    forward = spot * torch.exp(b * expiry_years)
    moneyness = calc_moneyness(shared.one * strike, spot, forward, deal_data, use_forwards, invert_moneyness)

    # per-scenario flag: has the barrier been crossed at a past discrete observation date? Once
    # set, KO scenarios are worth 0 and KI scenarios are worth the vanilla European
    barrier_hit = shared.one.new_zeros(shared.simulation_batch, dtype=torch.bool)
    row_ofs = 0
    # read once before a draw is taken so a non-GBM refusal lands first; it SUPERSEDES the
    # registration below rather than joining it
    smooth = branch_and_weight(shared, deal_data)
    second_order = smooth and getattr(shared, 'gamma', False)
    boundary_aad = getattr(shared, 'boundary_aad', False) and not smooth
    b_gaps, b_crossed, b_obs_before, b_alive, b_dead = [], [], [], [], []

    for index, (discount_block, spot_block, moneyness_block, rem_exp) in enumerate(
            utils.split_counts([discount, spot, moneyness, expiry_years], counts, shared)):

        t_block = discount_block.time_grid
        sample_index_t = start_index[index]

        # per-row hit mask [N_mtm, batch]: hits at STRICTLY EARLIER observations only - this block's
        # own date is priced by sim_spot_oss as its zero-length first step
        row_barrier_hit = barrier_hit.unsqueeze(0).expand(len(t_block), -1)
        row_ofs += len(t_block)
        if boundary_aad:
            # how many observations have already resolved for every row of this block
            b_obs_before.extend([len(b_gaps)] * len(t_block))
        if BarrierDates[sample_index_t] > 0:
            crossed = ((spot_block[-1] > barrier) if eta == BARRIER_UP else
                       (spot_block[-1] < barrier))
            if boundary_aad:
                # margin of the decision, graph retained; signed so gap > 0 means CROSSED
                b_gaps.append(torch.log(spot_block[-1] / barrier) * (
                    -1.0 if eta == BARRIER_DOWN else 1.0))
                b_crossed.append(crossed.detach())
            if cash_rebate and direction == BARRIER_OUT:
                # settle the rebate on the date it falls due, as pv_barrier_option does; the
                # TERMINAL row is excluded - the settle after the loop already pays it in full
                if row_ofs < len(deal_data.Time_dep.deal_time_grid):
                    newly_hit = (crossed & ~barrier_hit).to(spot_block.dtype)
                    cash_settle(shared, factor_dep['SettleCurrency'],
                                deal_data.Time_dep.deal_time_grid[row_ofs - 1],
                                nominal * rebate_per_unit * newly_hit)
            barrier_hit = barrier_hit | crossed   # carry forward into the next block

        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM]
        fixings = (dual_samples.np[np.newaxis, sample_index_t:, utils.RESET_INDEX_End_Day] -
                   t_block[:, utils.TIME_GRID_MTM, np.newaxis])

        drifts = utils.calc_eq_drift(
            deal_data.Factor_dep['Equity_Zero'], deal_data.Factor_dep['Dividend_Yield'],
            fixings, t_block, shared, multiply_by_time=False)

        fixing_block = daycount_fn(fixings)
        expiry = daycount_fn(tenor_block)
        # the EUROPEAN read: the surface's own vol at (K, remaining expiry). Under a non-GBM spot
        # model the implied surface has no consumer, so neither vol quantity is built there
        expiry_vols = utils.calc_time_grid_vol_rate(
            factor_dep['Volatility'], moneyness_block, expiry,
            shared) if not hn else spot_block.new_empty(0)

        adj = None
        if factor_dep.get('Check_Payoff_Type', False):
            # a DEAL-LEVEL adjustment, so its vol input stays the expiry read. Barrier and strike
            # are payoff-currency quantities, so they compare against the scaled spot as authored
            adj = calc_vol_adjustment(factor_dep, deal_time, expiry, expiry_vols, shared, fixings)
            expiry_vols = adj['vol']
            drifts = drifts + adj['carry_adj']
            spot_block = spot_block * adj['spot_scale']

        sample_ts = drifts.new(
            np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)]))
        cum_t = drifts.new(fixing_block)
        # the carry over each INTERVAL, which is not the zero carry to each fixing
        # (forward_carry_rate); `fixing_block` is the cumulative strip the difference needs
        fwd_drifts = forward_carry_rate(drifts, cum_t, sample_ts)
        # the SIMULATION read: the surface at every fixing's own TENOR and the deal's own declared
        # moneyness, differenced into forward variance (forward_vol_rate)
        interval_vols = forward_vol_rate(forward_vol_strip(
            deal_data, shared.one * strike, spot_block, drifts, fixing_block, shared,
            invert_moneyness, use_forwards), cum_t, sample_ts) if not hn else spot_block.new_empty(0)
        if adj is not None and adj['fx_vol'] is not None:
            # compo: the simulation steps S*X, so each interval's vol is the PRODUCT's - the
            # asset's own interval strip composed with the fx expiry read
            interval_vols = compo_vol(interval_vols, adj['fx_vol'].unsqueeze(1), adj['rho'])
        # sigma(K, tau) * sqrt(tau): the ONE European standard deviation, marked by the already-hit
        # leg below and by the in-out-parity vanilla inside `sim_spot_oss`
        sd_to_expiry = expiry_vols * torch.sqrt(rem_exp.clamp(min=1e-4)) if not hn else expiry_vols

        # discount rates per fixing: [N_block, N_fix, batch]
        discount_rates = utils.calc_discount_rate(discount_block, fixings, shared)

        all_hit = row_barrier_hit.all()
        some_hit = row_barrier_hit.any()

        # vanilla European for already-hit scenarios (KI) or zeros (KO); also built when a
        # counterfactual will want it - a closed form cannot perturb the OSS stream
        if some_hit or boundary_aad:
            if direction == BARRIER_IN:
                # ONE forward, shared with the in-out-parity leg inside sim_spot_oss - the two
                # value the same European on the same state
                log_fwd = total_log_forward(fwd_drifts, sample_ts)              # [N_block, batch]
                if hn:
                    # an already-knocked-in KI IS a vanilla, priced under the DECLARED model on the
                    # parity leg's own discretisation - same per-row n_total, same scalar carry
                    carry_det = log_fwd.detach()  # a guard reads a magnitude, not the tape
                    carry_spread = float((carry_det.amax(dim=1) - carry_det.amin(dim=1)).max())
                    if carry_spread > 1.0e-9:
                        raise ValueError(
                            'HN already-hit KI leg needs batch-constant carry; carry varies across '
                            'scenarios by {:.2e} (stochastic rates?) - extend hn_call to batched '
                            'carry or price this barrier under GBM'.format(carry_spread))
                    kit = oss_model_kit(factor_dep, hn_scalars)
                    rows = []
                    for row, spot_row in enumerate(spot_block):
                        # n_steps drives the variance recursion, so it is a scalar per MTM row
                        n_total = sum(max(int(round(float(t) * hn_spy)), 1) for t in sample_ts[row])
                        r_step = log_fwd[row][0] / n_total
                        if isdigital:
                            q_below = kit.cdf_logret(
                                torch.log(strike / spot_row), n_total, r_step)
                            rows.append((1.0 - q_below) if phi == OPTION_CALL else q_below)
                        else:
                            rows.append(kit.vanilla(
                                spot_row, strike, n_total, r_step, phi == OPTION_CALL
                            ) * torch.exp(r_step * n_total))
                    vanilla = torch.stack(rows)
                else:
                    fwd_to_expiry = spot_block * torch.exp(log_fwd)                     # [N_block, batch]
                    vol_to_T = sd_to_expiry                                             # [N_block, batch]
                    if isdigital:
                        vanilla = utils.black_european_option(
                            fwd_to_expiry, strike, vol_to_T, 1.0, 1.0, phi, shared, cash_payoff=1.0)
                    else:
                        vanilla = utils.black_european_option(
                            fwd_to_expiry, strike, vol_to_T, 1.0, 1.0, phi, shared)
                terminal_df = torch.squeeze(utils.calc_discount_rate(
                    discount_block, tenor_block.reshape(-1, 1), shared), dim=1)     # [N_block, batch]
                hit_value = nominal * vanilla * terminal_df
            else:
                hit_value = shared.one.new_zeros(len(t_block), shared.simulation_batch)

        if all_hit:
            # every row of every scenario has already resolved - skip OSS entirely
            theo_cashflow = hit_value
        else:
            simulate = partial(sim_spot_oss, sample_index_t, sobol, shared.MCMC_sims)
            theta = (spot_block, interval_vols, sd_to_expiry, sample_ts, fwd_drifts,
                     discount_rates) + hn_scalars
            # the SAME callable either way: under the node it is called twice
            oss_result, = InnerMCRecompute.run(shared, simulate, *theta)
            oss_result = nominal * oss_result
            # override per-row per-scenario where the barrier had already been crossed
            theo_cashflow = torch.where(
                row_barrier_hit, hit_value, oss_result) if some_hit else oss_result

        if boundary_aad:
            b_dead.append(hit_value.detach())
            # all_hit blocks skipped the OSS, so they carry no counterfactual
            b_alive.append((hit_value if all_hit else oss_result).detach())

        mtm_list.append(theo_cashflow)

    if boundary_aad and b_gaps and time_grid.report_index is not None:
        # branches stay on THIS pricer's grid and in its own currency; report_index is None on a
        # grid nobody reports off, where there is no reporting row to register against
        shared.boundary_sets.append(utils.LatchedBoundarySet(
            gaps=b_gaps, fired=b_crossed, obs_before=np.array(b_obs_before),
            untriggered=torch.cat(b_alive, dim=0), triggered=torch.cat(b_dead, dim=0),
            to_mtm=deal_to_mtm_grid(time_grid, deal_data, fx_rep),
            report_index=time_grid.report_index))

    # barrier options settle once at expiry
    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], mtm_list[-1][-1])
    return torch.cat(mtm_list, dim=0)


def pv_barrier_option(shared, time_grid, deal_data, nominal, spot, b,
                      tau, payoff_currency, invert_moneyness=False, use_forwards=False):
    """Single-barrier option valued by closed form at each scenario date, weighted by a
    Brownian-bridge touch probability.

    A barrier is monitored CONTINUOUSLY but the scenario grid only observes its own dates, so asking
    "is the spot beyond the barrier now" misses every path that crossed and came back - on a
    quarterly grid that overstates survival by 0.18. ``touched`` is the bridge probability and it
    WEIGHTS both branches, which is what makes the deal value an expectation rather than one draw,
    and differentiable where an indicator was not.

    Discrete monitoring uses Broadie-Glasserman-Kou: the continuous formula against a barrier
    shifted AWAY from the live region by the monitoring interval's vol. The shift direction is the
    barrier TYPE, not where the spot happens to sit.
    """
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    factor_dep = deal_data.Factor_dep
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # work out what we're pricing
    phi = OPTION_CALL if deal_data.Instrument.field['Option_Type'] == 'Call' else OPTION_PUT
    eta = BARRIER_DOWN if 'Down' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_UP
    direction = BARRIER_OUT if 'Out' in deal_data.Instrument.field['Barrier_Type'] else BARRIER_IN
    buy_or_sell = 1.0 if deal_data.Instrument.field['Buy_Sell'] == 'Buy' else -1.0
    barrier = deal_data.Instrument.field['Barrier_Price']
    strike = deal_data.Instrument.field['Strike_Price']
    cash_rebate = deal_data.Instrument.field['Cash_Rebate']

    # get the zero curve
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    expiry = daycount_fn(tau)
    # cache the expiry tenors
    expiry_years_key = ('Expiry_Years', tuple(expiry))
    if expiry_years_key not in factor_dep:
        factor_dep[expiry_years_key] = spot.new(expiry.reshape(-1, 1))

    expiry_years = factor_dep[expiry_years_key]
    forward = spot * torch.exp(b * expiry_years)
    moneyness = calc_moneyness(shared.one * strike, spot, forward, deal_data, use_forwards, invert_moneyness)
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    if factor_dep.get('Check_Payoff_Type', False):
        # need quanto/compo adjustments
        adj = calc_vol_adjustment(factor_dep, deal_time, expiry, sigma, shared)
        sigma = adj['vol']
        b = b + adj['b_adj']

    barrierOption = getbarrierpayoff(direction, eta, phi, strike, barrier)
    r = torch.squeeze(discounts.gather_weighted_curve(shared, tau.reshape(-1, 1), multiply_by_time=False), dim=1)

    mtm_list = []
    prev_touched = 0.0
    interval_variance = utils.bridge_interval_variance(shared, factor_dep, deal_time)
    prev_spot = None

    for index, (raw_sig, exp, b_t, r_t, s_t, f_t, cash_index) in enumerate(zip(
            sigma, expiry_years, b, r, spot, forward, deal_data.Time_dep.deal_time_grid)):

        # barrier options are very sensitive to vols - so we clamp them to 5%
        sig = raw_sig.clamp(min=0.05)

        # Broadie-Glasserman-Kou shift, AWAY from the live region by the monitoring interval's vol
        barrier_t = barrier * torch.exp(
            (1.0 if eta == BARRIER_UP else -1.0) * sig * factor_dep['Barrier_Monitoring']
        ) if factor_dep['Barrier_Monitoring'] else barrier

        # the bridge tests the SAME barrier the closed form prices against, or the profile is not
        # a martingale
        touched = utils.barrier_touched(prev_touched, prev_spot, s_t, barrier_t,
                                        interval_variance[index], eta == BARRIER_UP)
        prev_spot = s_t

        if (1 - touched).any() and expiry[index] > 0:

            payoff = buy_or_sell * nominal * barrierOption(
                sig, exp, cash_rebate / nominal, b_t, r_t, s_t, barrier_t)
        elif direction == BARRIER_OUT and expiry[index] == 0.0:
            # the closed form divides by sqrt(expiry) and cannot be asked for the value AT expiry,
            # where a surviving knock-out is worth exactly its intrinsic
            payoff = buy_or_sell * nominal * torch.relu(phi * (f_t - strike))
        else:
            payoff = 0.0

        barrier_part = (1.0 - touched) * payoff

        if direction == BARRIER_IN:
            payoff_european = buy_or_sell * (
                utils.black_european_option(f_t, strike, sig, expiry[index], 1.0, phi, shared) * torch.exp(
                    -r_t * exp) if expiry[index] else torch.relu(phi * (f_t - strike))
            )
            european_part = touched * (nominal * payoff_european)
            rebate_part = buy_or_sell * cash_rebate * (1 - touched) if expiry[index] == 0.0 else 0.0
            mtm_list.append(rebate_part + european_part + barrier_part)
        else:
            rebate_part = buy_or_sell * cash_rebate * (touched - prev_touched)
            mtm_list.append(rebate_part + barrier_part)
            if cash_rebate and expiry[index] > 0.0:
                cash_settle(shared, payoff_currency, cash_index, rebate_part)

        prev_touched = touched

    cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], mtm_list[-1])
    return torch.stack(mtm_list)


def pv_one_touch_option(shared, time_grid, deal_data, nominal, spot, b,
                        tau, payoff_currency, invert_moneyness=False, use_forwards=False):
    """One-touch (or no-touch) digital valued by closed form at each scenario date, weighted by the
    same Brownian-bridge touch probability ``pv_barrier_option`` uses.

    Under ``Payment_Timing='Expiry'`` the nominal is paid at expiry if the barrier was EVER touched,
    so between the touch and expiry the path holds a CERTAIN claim on the nominal, worth its
    discounted value rather than nothing.
    """
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # work out what we're pricing

    eta = BARRIER_DOWN if 'Down' in deal_data.Instrument.field['Barrier_Type_One'] else BARRIER_UP
    buy_or_sell = 1.0 if deal_data.Instrument.field['Buy_Sell'] == 'Buy' else -1.0
    barrier = deal_data.Instrument.field['Barrier_Price']

    # get the zero curve
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    expiry = daycount_fn(tau)
    # cache the expiry tenors
    expiry_years_key = ('Expiry_Years', tuple(expiry))
    if expiry_years_key not in factor_dep:
        factor_dep[expiry_years_key] = spot.new(expiry.reshape(-1, 1))

    expiry_years = factor_dep[expiry_years_key]
    forward = spot * torch.exp(b * expiry_years)
    moneyness = calc_moneyness(shared.one * barrier, spot, forward, deal_data, use_forwards, invert_moneyness)
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    if factor_dep.get('Check_Payoff_Type', False):
        # need quanto/compo adjustments
        adj = calc_vol_adjustment(factor_dep, deal_time, expiry, sigma, shared)
        sigma = adj['vol']
        b = b + adj['b_adj']

    r = torch.squeeze(discounts.gather_weighted_curve(
        shared, tau.reshape(-1, 1), multiply_by_time=False), dim=1)

    mtm_list = []
    prev_touched = 0.0
    eta_scale = 0.7071067811865476 * eta
    # the same continuous-vs-observed mismatch as pv_barrier_option, but this payoff pays ON touch,
    # so missing a crossing UNDERSTATES it rather than overstating survival
    interval_variance = utils.bridge_interval_variance(shared, factor_dep, deal_time)
    prev_spot = None

    for index, (raw_sig, exp, b_t, r_t, s_t, cash_index) in enumerate(zip(
            sigma, expiry_years, b, r, spot, deal_data.Time_dep.deal_time_grid)):

        # the same monitoring-adjusted barrier the closed form below prices against
        barrier_t = barrier * torch.exp(
            (1.0 if eta == BARRIER_UP else -1.0) * raw_sig * factor_dep['Barrier_Monitoring']
        ) if factor_dep['Barrier_Monitoring'] else barrier

        touched = utils.barrier_touched(prev_touched, prev_spot, s_t, barrier_t,
                                        interval_variance[index], eta == BARRIER_UP)
        prev_spot = s_t

        if (1 - touched).any() and expiry[index] > 0:

            root_tau = torch.sqrt(exp)
            mu = b_t / raw_sig - 0.5 * raw_sig
            log_vol = torch.log(barrier_t / s_t) / raw_sig
            barrovert = log_vol / root_tau

            if deal_data.Instrument.field['Payment_Timing'] == 'Expiry':
                muroot = mu * root_tau
                d1 = muroot - barrovert
                d2 = -muroot - barrovert
                payoff = torch.exp(-r_t * exp) * 0.5 * (
                        torch.erfc(eta_scale * d1) + torch.exp(2.0 * mu * log_vol) * torch.erfc(eta_scale * d2))

            elif deal_data.Instrument.field['Payment_Timing'] == 'Touch':
                lamb = torch.sqrt(smooth_relu(mu * mu + 2.0 * r_t))
                lambroot = lamb * root_tau
                d1 = lambroot - barrovert
                d2 = -lambroot - barrovert
                payoff = 0.5 * (torch.exp((mu - lamb) * log_vol) * torch.erfc(eta_scale * d1) +
                                torch.exp((mu + lamb) * log_vol) * torch.erfc(eta_scale * d2))
        else:
            payoff = 0.0

        # value still contingent on a touch that has not happened yet
        one_touch_part = (1.0 - touched) * buy_or_sell * nominal * payoff

        if deal_data.Instrument.field['Payment_Timing'] == 'Touch':
            # paid AT the touch, so it settles here and leaves the deal - already-touched paths
            # carry nothing forward, which is why only the increment appears
            rebate_part = buy_or_sell * nominal * (touched - prev_touched)
            if rebate_part.any():
                cash_settle(shared, payoff_currency, cash_index, rebate_part)
        elif expiry[index] == 0.0:
            rebate_part = buy_or_sell * nominal * touched
            if rebate_part.any():
                cash_settle(shared, payoff_currency, cash_index, rebate_part)
        else:
            # touched but paid at expiry: a CERTAIN claim on the nominal, so carry its discounted
            # value rather than zero
            rebate_part = touched * buy_or_sell * nominal * torch.exp(-r_t * exp)

        mtm_list.append(rebate_part + one_touch_part)
        prev_touched = touched

    return torch.stack(mtm_list)


def pv_partial_barrier_option(shared, time_grid, deal_data, nominal,
                              spot, b, tau, tau1, payoff_currency, invert_moneyness=False):
    """Partial-time barrier option over the deal's scenario grid: the closed form on the surviving
    branch, the vanilla (knock-in) or the rebate (knock-out) on the touched one, weighted by a
    Brownian-bridge touch probability accumulated only INSIDE the barrier window."""
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    factor_dep = deal_data.Factor_dep
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # work out what we're pricing
    barrierType = deal_data.Instrument.field['Barrier_Type']
    isKnockIn = barrierType in ['Up_And_In', 'Down_And_In']

    eta = 0.0
    if barrierType in ['Down_And_Out', 'Down_And_In']:
        eta = BARRIER_DOWN
    elif barrierType in ['Up_And_Out', 'Up_And_In']:
        eta = BARRIER_UP

    phi = OPTION_CALL if deal_data.Instrument.field['Option_Type'] == 'Call' else OPTION_PUT
    buy_or_sell = 1.0 if deal_data.Instrument.field['Buy_Sell'] == 'Buy' else -1.0
    direction = BARRIER_OUT if 'Out' in barrierType else BARRIER_IN
    barrier = deal_data.Instrument.field['Barrier_Price']
    strike = deal_data.Instrument.field['Strike_Price']
    cash_rebate = deal_data.Instrument.field.get('Cash_Rebate', 0.0)
    # get the zero curve
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time[:-1], shared)

    expiry = daycount_fn(tau)
    limit = daycount_fn(tau1)
    need_spot_at_expiry = deal_time.shape[0] - expiry.size
    spot_prior, spot_at = torch.split(spot, (expiry.size, need_spot_at_expiry))
    moneyness = calc_moneyness(shared.one * strike, spot_prior, spot_prior, deal_data, False, invert_moneyness)
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    if factor_dep['Barrier_Monitoring']:
        # Broadie-Glasserman-Kou shift, AWAY from the live region by the barrier TYPE
        adj_barrier = barrier * torch.exp(
            (1.0 if eta == BARRIER_UP else -1.0) * sigma * factor_dep['Barrier_Monitoring'])
    else:
        adj_barrier = barrier

    r = torch.squeeze(discounts.gather_weighted_curve(
        shared, tau.reshape(-1, 1), multiply_by_time=False), dim=1)

    # cache the expiry tenors
    expiry_years_key = ('Expiry_Years', tuple(expiry), tuple(limit))
    if expiry_years_key not in factor_dep:
        factor_dep[expiry_years_key] = (spot.new(expiry.reshape(-1, 1)), spot.new(limit.reshape(-1, 1)))

    expiry_years, limit_years = factor_dep[expiry_years_key]

    at_start = deal_data.Instrument.field['Barrier_At_Start'] == 'Yes'
    # the clamp keeps the window positive (the closed form is NaN at zero) and is also the
    # VALUATION there, both cases being limits of the same formula to O(clamp): a vanishing start
    # window IS the vanilla on its surviving branch, a vanishing head IS the full barrier
    barrier_payoff = buy_or_sell * nominal * getpartialbarrierpayoff(
        isKnockIn, eta, phi, spot_prior, strike, adj_barrier, at_start,
        limit_years.clamp(min=1e-6), expiry_years, r, b, sigma)

    if need_spot_at_expiry:
        # the SAME running touch probability as `pv_barrier_option`, accumulated ONLY inside the
        # barrier window. The interval straddling the limit date is attributed to the side its far
        # end sits on, and the bridge tests the SAME shifted barrier the closed form prices against
        interval_variance = utils.bridge_interval_variance(shared, factor_dep, deal_time)
        interval_days = deal_time[:, utils.TIME_GRID_MTM]
        in_window = np.concatenate([
            [at_start or factor_dep['Limit_Date'] <= interval_days[0]],
            (interval_days[1:] <= factor_dep['Limit_Date']) if at_start else
            (interval_days[:-1] >= factor_dep['Limit_Date'])])
        rows, prev_touched, prev_spot = [], 0.0, None
        last_bar = adj_barrier.shape[0] - 1 if torch.is_tensor(adj_barrier) else None
        for index in range(deal_time.shape[0]):
            if in_window[index]:
                bar_t = adj_barrier[min(index, last_bar)] if last_bar is not None else adj_barrier
                prev_touched = utils.barrier_touched(
                    prev_touched, prev_spot, spot[index], bar_t, interval_variance[index],
                    eta == BARRIER_UP)
            elif index == 0:
                prev_touched = torch.zeros_like(spot[0])
            prev_spot = spot[index]
            rows.append(prev_touched)
        barrier_touched = torch.stack(rows)
        first_touch = barrier_touched[1:] - barrier_touched[:-1]
        payoff_at = buy_or_sell * torch.relu(phi * (spot_at - strike))

        if direction == BARRIER_IN:
            forward = spot_prior * torch.exp(b * expiry_years)
            payoff_prior = utils.black_european_option(
                forward, strike, sigma, expiry, buy_or_sell, phi, shared) * torch.exp(-r * expiry_years)
            european_part = barrier_touched * (nominal * torch.cat([payoff_prior, payoff_at], dim=0))
            barrier_part = (1.0 - barrier_touched) * F.pad(
                barrier_payoff, [0, 0, 0, 1], value=buy_or_sell * cash_rebate)
            combined = european_part + barrier_part
            # the knock-in can only settle at the end
            cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], combined[-1])
        else:
            barrier_part = (1.0 - barrier_touched) * torch.cat([barrier_payoff, nominal * payoff_at], dim=0)
            rebate_part = buy_or_sell * cash_rebate * first_touch
            combined = F.pad(buy_or_sell * cash_rebate * first_touch, [0, 0, 1, 0]) + barrier_part
            cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], barrier_part[-1])
            if cash_rebate:
                for cash_index, cash in zip(deal_data.Time_dep.deal_time_grid[1:], rebate_part):
                    cash_settle(shared, payoff_currency, cash_index, cash)
    else:
        combined = barrier_payoff

    return combined


def pv_american_option(shared, time_grid, deal_data, nominal, moneyness, spot, forward):
    """American option by the Bjerksund-Stensland trigger approximation, floored at Black.

    Rows where the carry is at or above the rate hold no early exercise premium and take Black
    directly. A scenario whose spot reaches the exercise trigger settles its intrinsic on that date
    and is worth nothing after.
    """
    def phi(gamma, H, I):
        kappa = (2.0 * safe_b) / sigma2 + 2.0 * gamma - 1.0
        d = (torch.log(H / safe_S) - (safe_b + (gamma - 0.5) * sigma2) * tau) / vol
        lamb = -safe_r + gamma * safe_b + 0.5 * gamma * (gamma - 1.0) * sigma2
        log_IS = torch.log(I / safe_S)
        safe_exp = (kappa * log_IS).clamp(max=25.0)
        ret = utils.norm_cdf(d) - torch.exp(safe_exp) * utils.norm_cdf(d - 2.0 * log_IS / vol)
        return torch.exp(lamb * tau) * ret

    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    tenor_in_days = factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]
    expiry = discount.code[0][utils.FACTOR_INDEX_Daycount](tenor_in_days)
    sigma = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    # clamped away from zero: the formulae below divide by tau
    tau = spot.new(expiry.reshape(-1, 1)).clamp(min=1e-5)
    b = torch.log(forward / spot) / tau
    r = torch.squeeze(
        discount.gather_weighted_curve(shared, tenor_in_days.reshape(-1, 1), multiply_by_time=False), dim=1)
    vol = sigma * torch.sqrt(tau)
    # a put is the call on the transposed (spot, strike) with the carry reflected
    if factor_dep['Option_Type'] > 0:
        S, K = spot, factor_dep['Strike_Price'] * shared.one
    else:
        S, K = shared.one * factor_dep['Strike_Price'], spot
        r, b = r - b, -b

    sigma2 = sigma * sigma
    american = b < r - 1e-6
    safe_b = american * b
    b_over_sigma2 = safe_b / sigma2
    # pad all non american exercise points (0.375 is arbitrary)
    safe_r = american * r + ~american * 0.375 * sigma2
    # clamped to keep the root and the gradients defined on the padded points
    safe_sqrt = ((b_over_sigma2 - 0.5) ** 2 + 2.0 * safe_r / sigma2).clamp(min=1e-6)
    beta = (0.5 - b_over_sigma2) + torch.sqrt(safe_sqrt)
    r_b = safe_r - safe_b
    # the exercise trigger, at zero maturity and at infinity
    B_0 = K * torch.maximum(safe_r / r_b, torch.ones_like(safe_r))
    B_inf = K * beta / (beta - 1)

    if (K > 0.0).all():
        h_tau = -(b * tau + 2 * vol) * (B_0 / (B_inf - B_0))
        I = B_0 + (B_inf - B_0) * (1 - torch.exp(h_tau))
        safe_S = torch.min(S - 1e-6, I)
        C_BS = (I - K) * torch.exp(torch.log(safe_S / I) * beta) * (1.0 - phi(beta, I, I))
        x = phi(1.0, I, I)
        y = phi(1.0, K, I)
        C_BS += safe_S * (x - y)
        x = phi(0.0, K, I)
        y = phi(0.0, I, I)
        C_BS += K * (x - y)
    else:
        I = B_0
        C_BS = B_0

    Black = utils.black_european_option(
        S * torch.exp(b * tau), K, vol, 1.0, 1.0, 1.0, shared) * torch.exp(-r * tau)

    C_BS = torch.maximum(Black, C_BS)

    theo_price = (b >= r) * Black + (b < r) * ((S < I) * C_BS + (S >= I) * (S - K))
    early_exercise = (b < r) * (S >= I)

    knocked_out = early_exercise.cumsum(axis=0) > 0
    first_knockout = torch.concat([knocked_out[0].reshape(1, -1), knocked_out[1:] ^ knocked_out[:-1]])
    # the deal is in force today whatever today's spot does
    knocked_out[0] = False
    value = factor_dep['Buy_Sell'] * nominal * theo_price * ~knocked_out

    exercise_val = factor_dep['Buy_Sell'] * nominal * (S - K) * first_knockout
    for t, cashflows in zip(deal_data.Time_dep.deal_time_grid, exercise_val):
        if cashflows.any():
            cash_settle(shared, factor_dep['SettleCurrency'], t, cashflows)

    return value


def pv_european_option(shared, time_grid, deal_data, nominal, moneyness, forward, binary=False,
                       digital_spread=None):
    """European option (or, under `binary`, its cash-or-nothing digital) at each scenario date.

    `digital_spread` prices the digital as a tight call/put spread instead of a closed-form binary,
    which is what puts the smile's dVol/dK term into the price.
    """
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    tenor_in_days = factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]
    settle_in_days = factor_dep['Settlement'] - deal_time[:, utils.TIME_GRID_MTM]
    expiry = discount.code[0][utils.FACTOR_INDEX_Daycount](tenor_in_days)
    vols = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    adj = None
    if factor_dep.get('Check_Payoff_Type', False):
        adj = calc_vol_adjustment(factor_dep, deal_time, expiry, vols, shared)
        vols = adj['vol']
        forward = adj['s_adj'] * forward * torch.exp(adj['b_adj'] * shared.one.new(expiry.reshape(-1, 1)))

    if binary:
        if digital_spread is not None:
            # each leg re-queries the surface at its OWN strike. The quanto drift bend (ATM-vol
            # b_adj) and the compo forward are strike-free and already in `forward`; a compo leg
            # composes its own strike's vol
            eps, m_lo, m_hi = digital_spread
            strike = factor_dep['Strike_Price']
            legs = []
            for shift, m in ((-1.0, m_lo), (1.0, m_hi)):
                leg_vols = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], m, expiry, shared)
                if adj is not None and adj['fx_vol'] is not None:
                    leg_vols = compo_vol(leg_vols, adj['fx_vol'], adj['rho'])
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug('DIGITAL_SPREAD eps=%.6g strike=%.6g leg=%s vol=%.6g',
                                  eps, strike, '-+'[shift > 0], float(leg_vols[0].mean()))
                legs.append(utils.black_european_option(
                    forward, strike * (1.0 + shift * eps), leg_vols, expiry, 1.0,
                    factor_dep['Option_Type'], shared))
            value = nominal * factor_dep['Buy_Sell'] * factor_dep['Option_Type'] * (
                legs[0] - legs[1]) / (2.0 * eps * strike)
        else:
            value = utils.black_european_option(
                forward, factor_dep['Strike_Price'], vols, expiry,
                factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared, cash_payoff=nominal)
    else:
        theo_price = utils.black_european_option(
            forward, factor_dep['Strike_Price'], vols, expiry,
            factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared)
        value = nominal * theo_price

    discount_rates = torch.squeeze(
        utils.calc_discount_rate(discount, settle_in_days.reshape(-1, 1), shared), dim=1)

    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], value[-1])

    return value * discount_rates


def pv_MC_Accumulator(shared, time_grid, deal_data, spot, fx_rep):
    """Fixing-observed FX accumulator with a knock-out barrier, by one-step-survival Monte Carlo.

    At each surviving fixing the signed payoff ``N1 * relu(cp*(S_j - K)) - N2 * relu(-cp*(S_j - K))``
    pays at that fixing's own settlement date; the fixing that breaches the barrier pays nothing and
    cancels every later fixing. Survival at a FUTURE fixing is the analytic OSS truncation at the
    constant barrier (``oss_truncated_draw`` - F-measurable over the interval, so exact); a fixing
    the row has already OBSERVED is resolved by its exact indicator off the outer sample - no
    simulated zero-length step, so no variance floor.

    A fixing dated ON the base date resolves off the SIMULATED spot, not the recorded print: a
    sensitivity bump can only reach the fixing if the fixing follows the factor. The two designs
    agree in value and differ by the whole fixing's contribution in delta (``make_fixing_data``).

    The carry and vol arrive as interval strips built at the call site, ``use_forwards=True`` (every
    FX pricer reads the surface at forward moneyness, so the never-knocking limit prices at the
    quote the FX vanillas mark with). Under Heston-Nandi the strips are not built; the recursion
    supplies the variance and ``sim_spot_tarf``'s (F1)-(F4) apply verbatim. RECOMPUTE: pure
    bound/theta split; settles are RETURNED, and grouped settlements (several fixings paying one
    value date) accumulate through ``cash_settle``.

    BOUNDARY AAD: the knock-out is decided on OUTER scenario state, so the whole latch is built
    outside the node - gaps signed so ``gap > 0`` means CROSSED, ``untriggered`` the alive branch on
    a weight the observed indicators never zero, ``triggered`` ZEROS. A deal declared
    ``Barrier_Hit`` registers nothing: its death is data, not simulated state. The alive prefix
    folds ``Barrier_Hit`` with every settled fixing's own breach test, so the deal prices from the
    data even without the flag.

    DECLARED LIMITATION: the zero dead branch omits the pending settlements a knocked deal still
    owes - exact wherever the settlement lag is shorter than the fixing spacing, NOT one-signed on a
    leveraged deal, and confined to cells the latch calls dead (magnitudes in roadmap.md). An exact
    branch would be a per-decision head profile, a fourth BoundarySet shape.

    BRANCH AND WEIGHT (``Branch_And_Weight: 'Yes'``, base valuation only) SWAPS the estimator for
    that registration rather than joining it - one decision, one estimator. Off, and absent, is
    every line above bit for bit.

    THIS LOOP IS ALREADY THE SMOOTH ESTIMATOR: survival at a future fixing is
    ``oss_truncated_draw``'s analytic ``p``, the continuation is its truncated draw, and the fired
    branch pays exactly zero, so there is no ``p x realised`` shortcut in it and no indicator on the
    tape between the observed fixings. The VALUE under the switch is bit-identical; what changes is:

    - THE REGISTRATION IS SKIPPED, so ``Greeks: 'All'`` flows where it used to refuse.
    - THE PER-FIXING ACCRUAL KINK is built at second order (``accrual_kink_term``), because both
      legs are ``relu``s of one argument and pathwise AAD reports their gamma as an exact zero.

    THE PENDING HEAD IS ARITHMETIC HERE, not a branch to build: each fixing is already discounted to
    its OWN settlement date and paid with weight ``L_{j-1} * p_j``, so a path knocking at fixing
    ``k`` has already been paid for every fixing before it, whatever their settlement lag.

    WHAT THE SWITCH DOES NOT SMOOTH: the observed fixings' own breach tests and the ``Barrier_Hit``
    flag, which are data rather than simulated state, so their indicators stay exact. Under base
    valuation they are historic prints carrying no graph, so the skipped registration contributes
    exactly zero.
    """

    def survives(s):
        return (s < barrier) if barrier_up else (s > barrier)

    def sim_spot_accumulator(settlement, fixing_offset, sobol, num_sims,
                             spot_prices, times, carry, prev_alive, discount_rates, vols_all,
                             past_fixings, *hn_scalars):
        """OSS inner Monte Carlo over one block of MTM rows; mean PV and alive-branch PV per row.

        PURE bound/theta split for `InnerMCRecompute`. Returns `(mtm, alive_pv, settled,
        settle_rows)`: the block's rows, the never-knocked branch the latch needs (empty when
        sensitivities are off), and the first-fixing cashflow of every row that settles today with
        the block-local rows those land on. `prev_alive` is theta, carrying the (a.e. zero) graph
        of the observed samples that built it.
        """
        # the declared model as a walking kit - the pricer names it once, here
        kit = oss_model_kit(factor_dep, hn_scalars)
        hn = kit is not None
        mcmc, alive, settled, settle_rows = [], [], [], []
        for i, (D, s, carry_rate, delta_t, tau) in enumerate(zip(
                discount_rates, spot_prices, carry, times, settlement)):
            reduced_samples = len(delta_t)
            if reduced_samples == 0:
                mcmc.append(shared.one.new_zeros(shared.simulation_batch))
                if boundary_aad:
                    alive.append(shared.one.new_zeros(shared.simulation_batch))
                continue
            if not hn:
                vols = vols_all[i]
            u = oss_uniforms(shared, reduced_samples, num_sims, sobol)
            Sj = torch.unsqueeze(s, 1)
            st = kit.seed() if hn else None
            P = shared.one.new_zeros((shared.simulation_batch, 2 * num_sims))
            L = prev_alive.reshape(-1, 1) * shared.one.new_ones((shared.simulation_batch, 2 * num_sims))
            if boundary_aad:
                # the alive branch: observed indicators (the latch's decisions) never zero it;
                # the smooth future survivals still apply - the left limit, not a re-simulation
                P_alive, L_alive = P, torch.ones_like(L)
            # started at `prev_alive`, not at one: the observed prefix has already killed paths
            ledger = SurvivalLedger(L) if smooth else None
            for j in range(reduced_samples):
                dt = delta_t[j]
                Dj = D[j].reshape(-1, 1)
                if dt > 0:
                    fwd_carry = carry_rate[j].reshape(-1, 1)
                    if hn:
                        # daily HN sub-steps; only the LAST is monitored, so the OSS truncation at
                        # the constant barrier is exact
                        n_sub = max(int(round(float(dt) * hn_spy)), 1)
                        b_step = fwd_carry * dt / n_sub
                        Sj, st = kit.substeps(
                            Sj, st, b_step, n_sub - 1, shared, num_sims, antithetic=True)
                        h = kit.h(st)
                        z_bound = (torch.log(barrier / Sj) - (b_step - 0.5 * h)) / torch.sqrt(h)
                        p, Z = oss_truncated_draw(u[j], z_bound, barrier_up)
                        Sj, st = kit.advance(Sj, st, b_step, Z)
                    else:
                        fwd_vol = vols[j].reshape(-1, 1)
                        vol_dt = fwd_vol * torch.sqrt(dt)
                        fwd_drift = (fwd_carry - 0.5 * fwd_vol * fwd_vol) * dt
                        z_bound = (torch.log(barrier / Sj) - fwd_drift) / vol_dt
                        p, Z = oss_truncated_draw(u[j], z_bound, barrier_up)
                        Sj = Sj * torch.exp(fwd_drift + vol_dt * Z)
                    payoff_spot = Sj
                    p_alive = p
                else:
                    # a fixing this row has already OBSERVED: resolved by its exact indicator off
                    # the outer sample - no simulated zero-length step, no variance floor
                    payoff_spot = past_fixings[fixing_offset + j].reshape(-1, 1).expand_as(P)
                    p = survives(payoff_spot).to(P.dtype)
                    p_alive = 1.0
                intrinsic = (payoff_spot - strike) * callOrPut
                cashflow = F.relu(intrinsic) * notional_itm - F.relu(-intrinsic) * notional_otm
                if second_order and dt > 0:
                    # ONE kernel serves both legs - `relu(-x)` kinks where `relu(x)` does and the
                    # term is even - so the net carries (N1 - N2) of it. `dt > 0` keeps OBSERVED
                    # fixings out: their spot is one point per path, a mass rather than a density
                    kink = accrual_kink_term(intrinsic, j)
                    cashflow = cashflow + kink * (notional_itm - notional_otm)
                cf_step = L * p * cashflow
                P = P + Dj * cf_step
                if ledger is not None:
                    ledger.fire(p)
                    L = ledger.alive
                else:
                    L = L * p
                if boundary_aad:
                    P_alive = P_alive + Dj * L_alive * p_alive * cashflow
                    L_alive = L_alive * p_alive
                # grouped settlements: EVERY fixing whose value date is today returns its own
                # (row, amount) pair - RETURNED, because a recompute would settle twice
                if tau[j] == 0:
                    settled.append(cf_step.mean(axis=1))
                    settle_rows.append(i)
            if ledger is not None:
                ledger.check('ACCUMULATOR row {}'.format(i))
            mcmc.append(P.mean(axis=1))
            if boundary_aad:
                alive.append(P_alive.mean(axis=1))
        return (torch.stack(mcmc),
                torch.stack(alive) if alive else prev_alive.new_empty(0),
                torch.stack(settled) if settled else prev_alive.new_empty(0),
                settle_rows)

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    samples = factor_dep['Fixings']
    start_idx = samples.get_start_index(deal_time)
    start_index, counts = np.unique(start_idx, return_counts=True)

    fx_samples = factor_dep['Price_Fixings']
    known_resets = fx_samples.known_resets(shared.simulation_batch)
    sim_samples = fx_samples.schedule[
        (fx_samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (fx_samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
    next_samples = utils.calc_fx_cross(
        factor_dep['Underlying_Currency'][0], factor_dep['Currency'][0],
        sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
    all_samples = torch.cat(
        [torch.cat(known_resets, dim=0), next_samples], dim=0) if known_resets else next_samples

    settle_idx = np.searchsorted(factor_dep['Settlement'], deal_time[:, utils.TIME_GRID_MTM]).astype(np.int64)
    fixing_indices = counts.cumsum() - 1
    settle_index = settle_idx[fixing_indices]

    strike = factor_dep['Strike_Price']
    barrier = factor_dep['Barrier_Price']
    barrier_up = factor_dep['Barrier_Up']
    callOrPut = factor_dep['Option_Type']
    buy_sell = factor_dep['Buy_Sell']
    notional_itm = factor_dep['Notional1'] * shared.one
    notional_otm = factor_dep['Notional2'] * shared.one

    # what the pricer DECIDED, which a value cannot show: how many fixings it resolved off the
    # outer path, whether the deal came in already dead, and how the book split into blocks
    logging.debug('ACCUMULATOR %s fixings=%d resolved=%d barrier=%.6g up=%d dead=%d blocks=%d',
                  deal_data.Instrument.field.get('Reference'), len(fx_samples.schedule),
                  len(known_resets) + len(sim_samples), barrier, int(barrier_up),
                  int(bool(factor_dep['Barrier_Hit'])), len(counts))

    # read once before a draw is taken so a non-GBM refusal lands first; it SUPERSEDES the
    # registration below rather than joining it
    smooth = branch_and_weight(shared, deal_data)
    # curvature is the switch: the per-fixing kink term is never BUILT unless a second derivative
    # was asked for, so it costs exactly nothing on a first-order run
    second_order = smooth and getattr(shared, 'gamma', False)

    # a declared-dead deal has no flux to record, and folding that in here keeps the simulation
    # from building an alive branch nobody reads
    boundary_aad = (getattr(shared, 'boundary_aad', False) and
                    not factor_dep['Barrier_Hit'] and not smooth)
    b_gaps, b_crossed, b_obs_before, alive_blocks, b_pending = [], [], [], [], []

    # the declared model's parameter tensors, by ITS OWN canonical name tuple; () is GBM
    hn_scalars = oss_model_scalars(factor_dep, shared)
    hn = bool(hn_scalars)
    if hn:
        hn_spy = factor_dep['HN_Steps_Per_Year']

    # prefix knock-out state: the declared flag OR any observed sample's own breach. `Barrier_Hit`
    # and the settled fixings' tests are folded in calc_dependencies; the unsettled observed and
    # simulated fixings fold here, per sample
    state = shared.one.new_ones(shared.simulation_batch) * (0.0 if factor_dep['Barrier_Hit'] else 1.0)
    alive_seq = [state]
    for k in range(len(all_samples)):
        state = state * survives(all_samples[k]).to(shared.one.dtype)
        alive_seq.append(state)
    fixing_days = fx_samples.schedule[:, utils.RESET_INDEX_Reset_Day]

    sobol = False
    if shared.simulation_batch > 16:
        sobol = True
        shared.reset_qrg()

    for b_idx, (discount_block, spot_block) in enumerate(
            utils.split_counts([discount, spot], counts, shared)):
        settle_index_local = int(settle_index[b_idx])
        t_block = discount_block.time_grid
        fixings = (fx_samples[np.newaxis, settle_index_local:, utils.RESET_INDEX_End_Day] -
                   t_block[:, utils.TIME_GRID_MTM, np.newaxis]).clip(min=0)
        drifts = utils.calc_fx_drift(
            factor_dep['Underlying_Currency'], factor_dep['Currency'],
            fixings, t_block, shared, multiply_by_time=False)
        fixing_block = daycount_fn(fixings)
        settlement = (factor_dep['Settlement'][np.newaxis, settle_index_local:] -
                      t_block[:, utils.TIME_GRID_MTM, np.newaxis])
        sample_ts = drifts.new(
            np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)]))
        discount_rates = utils.calc_discount_rate(discount_block, settlement, shared)
        cum_t = drifts.new(fixing_block)
        # the interval carry and vol strips, both theta rather than loop state, built where every
        # OSS adopter builds theirs; both take the ZERO carry `drifts` (a forward is cumulative)
        fwd_drifts = forward_carry_rate(drifts, cum_t, sample_ts)
        vols = forward_vol_rate(forward_vol_strip(
            deal_data, strike, spot_block, drifts, fixing_block, shared,
            factor_dep['Invert_Moneyness'], use_forwards=True), cum_t,
            sample_ts) if not hn else spot_block.new_empty(0)

        simulate = partial(sim_spot_accumulator, settlement, settle_index_local, sobol, shared.MCMC_sims)
        theta = (spot_block, sample_ts, fwd_drifts, alive_seq[settle_index_local],
                 discount_rates, vols, all_samples) + hn_scalars
        # the SAME callable either way: under the node it is called twice
        block_mtm, block_alive, block_settled, settle_rows = InnerMCRecompute.run(
            shared, simulate, *theta)
        theo_price = buy_sell * block_mtm

        # the by-products, performed once off the forward's result
        for row, value in zip(settle_rows, block_settled):
            cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
                time_grid.mtm_time_grid, t_block[row, utils.TIME_GRID_MTM]), buy_sell * value)

        if boundary_aad:
            # `obs_before` is stamped PER ROW off the block's LAST row: blocks are cut on the UNION
            # of fixing and settlement dates, so a fixing date is routinely a block's last row and a
            # count taken at its first is short by exactly that decision. A row's own fixing COUNTS
            rows_obs = np.searchsorted(fixing_days, t_block[:, utils.TIME_GRID_MTM], side='right')
            while len(b_gaps) < int(rows_obs[-1]):
                s_k = all_samples[len(b_gaps)]
                b_gaps.append(torch.log(s_k / barrier) * (1.0 if barrier_up else -1.0))
                b_crossed.append(~survives(s_k).detach())
            b_obs_before.extend(rows_obs.tolist())
            alive_blocks.append(block_alive)
            # the pending head, per row: fixings this row has OBSERVED whose settlements have not
            # landed. Blocks break on every schedule date, so the block's unsettled window
            # [settle_index_local, rows_obs) is every row's window
            for r_local in range(len(t_block)):
                j_hi = int(rows_obs[r_local])
                if j_hi > settle_index_local:
                    intrinsic = (torch.stack([all_samples[j] for j in range(
                        settle_index_local, j_hi)]) - strike) * callOrPut
                    pay = (F.relu(intrinsic) * factor_dep['Notional1'] -
                           F.relu(-intrinsic) * factor_dep['Notional2'])
                    d_settle = discount_rates[r_local][:j_hi - settle_index_local].reshape(
                        j_hi - settle_index_local, -1)
                    b_pending.append((settle_index_local, (buy_sell * d_settle * pay).detach()))
                else:
                    b_pending.append(None)
        mtm_list.append(theo_price)

    mtm = torch.cat(mtm_list, dim=0)

    # report_index is None on a grid nobody reports off - an HMC tradable, a calibration's
    # benchmark grid - and there is no reporting row for a counterfactual to land on
    if boundary_aad and b_gaps and time_grid.report_index is not None:
        untriggered = buy_sell * torch.cat(alive_blocks, dim=0).detach()
        latch = utils.LatchedBoundarySet(
            gaps=b_gaps, fired=b_crossed, obs_before=np.array(b_obs_before),
            untriggered=untriggered, triggered=torch.zeros_like(untriggered),
            pending=b_pending,
            to_mtm=deal_to_mtm_grid(time_grid, deal_data, fx_rep),
            report_index=time_grid.report_index)
        shared.boundary_sets.append(latch)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            # the registration reconstructed at the BOOKED flags against the engine's own rows -
            # the number that says the counterfactual profiles describe the reported deal
            with torch.no_grad():
                prefix = [torch.zeros_like(b_crossed[0])]
                for flag in b_crossed:
                    prefix.append(prefix[-1] | flag)
                recon = latch.select(prefix)
                head = max((float(c.abs().max()) for _, c in filter(None, b_pending)),
                           default=0.0)
                logging.debug('ACCUMULATOR_LATCH recon_max=%.3g head_max=%.3g scale=%.3g',
                              float((recon - mtm.detach()).abs().max()),
                              head, float(mtm.detach().abs().max()))

    return mtm


def pv_MC_ExtendableForward(shared, time_grid, deal_data, spot, fx_rep):
    """The bank-exercisable Strip/Rolling FX extendable forward.

    K1 applies to all guaranteed fixings through Extension_Date inclusive; K2 to every fixing an
    extension creates. On a decision date the current fixing is already in force: its cashflow is
    valued first, then the bank extends or the deal terminates permanently. The reporting side and
    the deciding side are the same book, so `forward_sign` orients both the payoff and the
    exercise rule - a sold deal is optimised exactly as a bought one, from its own side.

    Future decisions are One-Step Survival: a backward one-dimensional Gauss-Hermite quadrature
    builds the zero-continuation boundary H_j per rolling decision (for the strip the single
    boundary is closed-form, K2*B/A), and the forward pass draws only the continuing branch,
    weighting it by the analytic continuation probability. Truncated draws happen ONLY at
    decisions; between them cashflows are propagated as exact lognormal expectations. The boundary
    is detached: by value matching C_j(H_j) = 0, the first-order moving-boundary term vanishes.

    SCOPE: GBM off the implied surface, the smile frozen at K2 moneyness for transition variances.
    Heston-Nandi would make the continuation state two-dimensional and is deliberately not
    approximated. Reconstructed outer-grid decisions register a `LatchedBoundarySet`: the alive
    branch is the facts-only world, the dead branch the survived-weighted pending head, both
    derived from the same `value = fixed + state * live` row arithmetic the forward reports.
    """
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    k1 = factor_dep['Strike_Price']
    k2 = factor_dep['Extension_Strike']
    notional = factor_dep['Notional']
    forward_sign = factor_dep['Buy_Sell'] * factor_dep['Option_Type']
    style = factor_dep['Extension_Style']
    extension_index = int(factor_dep['Extension_Index'])
    decision_indices = [int(x) for x in factor_dep['Decision_Indices']]
    decision_set = set(decision_indices)
    decision_codes = factor_dep['Decision_Codes']
    fixing_days = factor_dep['Fixing_Days']
    settlement_days = factor_dep['Settlement']
    n_fix = len(fixing_days)
    eps = torch.finfo(shared.one.dtype).eps

    # observed fixings, one-to-one with the schedule: a historical print, or the scenario spot at
    # a fixing row the outer grid has reached; None while the fixing is still in the future
    fx_samples = factor_dep['Price_Fixings']
    max_mtm = deal_time[:, utils.TIME_GRID_MTM].max()
    sim_mask = ((fx_samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                (fx_samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= max_mtm))
    sim_schedule = fx_samples.schedule[sim_mask]
    next_samples = utils.calc_fx_cross(
        factor_dep['Underlying_Currency'][0], factor_dep['Currency'][0],
        sim_schedule[:, :utils.RESET_INDEX_Scenario + 1], shared) if len(sim_schedule) else []

    observed_fixings = []
    sim_ptr = 0
    for row in fx_samples.schedule:
        observed = None
        if row[utils.RESET_INDEX_Value]:
            observed = shared.one.new_full((shared.simulation_batch,),
                                           float(row[utils.RESET_INDEX_Value]))
        elif (row[utils.RESET_INDEX_Scenario] > -1 and
              row[utils.RESET_INDEX_Reset_Day] <= max_mtm):
            observed = next_samples[sim_ptr]
            sim_ptr += 1
        observed_fixings.append(observed)

    mtm_days = deal_time[:, utils.TIME_GRID_MTM].astype(np.int64)
    row_by_fixing = {}
    for i, d in enumerate(fixing_days):
        rows = np.flatnonzero(mtm_days == d)
        if rows.size:
            row_by_fixing[i] = int(rows[0])

    grid_size = max(int(factor_dep['Boundary_Grid_Size']), 33)
    if grid_size % 2 == 0:
        grid_size += 1
    quadrature = max(int(factor_dep['Boundary_Quadrature']), 8)
    boundary_chunk = max(int(factor_dep['Boundary_Batch_Chunk']), 1)
    std_width = factor_dep['Boundary_Std_Width']
    min_log_width = factor_dep['Boundary_Min_Log_Width']
    gh_x_np, gh_w_np = np.polynomial.hermite.hermgauss(quadrature)
    gh_z = torch.tensor(np.sqrt(2.0) * gh_x_np, dtype=shared.one.dtype, device=shared.one.device)
    gh_w = torch.tensor(gh_w_np / np.sqrt(np.pi), dtype=shared.one.dtype, device=shared.one.device)
    grid_u = torch.linspace(-1.0, 1.0, grid_size, dtype=shared.one.dtype, device=shared.one.device)

    logging.debug('EXTENDABLE %s style=%s fixings=%d decisions=%d known=%d extension=%d',
                  deal_data.Instrument.field.get('Reference'), style, n_fix,
                  len(decision_indices), int((decision_codes >= 0).sum()), extension_index)

    def market_inputs(row_index, first_index):
        """Frozen-curve/frozen-surface GBM inputs from one MTM row to schedule[first_index:]."""
        if first_index >= n_fix:
            return None
        t_row = deal_time[row_index:row_index + 1]
        mtm_day = int(t_row[0, utils.TIME_GRID_MTM])
        ids = np.arange(first_index, n_fix, dtype=np.int64)
        fix_days = np.maximum(fixing_days[ids] - mtm_day, 0).reshape(1, -1)
        set_days = (settlement_days[ids] - mtm_day).reshape(1, -1)
        discount_row = utils.calc_time_grid_curve_rate(factor_dep['Discount'], t_row, shared)
        carry = utils.calc_fx_drift(
            factor_dep['Underlying_Currency'], factor_dep['Currency'],
            fix_days, t_row, shared, multiply_by_time=False)[0]
        full_np = daycount_fn(fix_days)[0]
        full = carry.new(full_np).reshape(-1, 1)
        log_fwd = carry * full
        s = spot[row_index]
        forward = s.unsqueeze(0) * torch.exp(log_fwd)
        moneyness = k2 / forward if factor_dep['Invert_Moneyness'] else forward / k2
        vols = []
        for mon, tau in zip(moneyness, full_np):
            if float(tau) <= 0.0:
                vols.append(torch.zeros_like(s))
            else:
                vols.append(utils.calc_time_grid_vol_rate(
                    factor_dep['Volatility'], mon.reshape(1, -1), np.array([tau]),
                    shared).reshape(-1))
        vols = torch.stack(vols)
        total_var = (vols * vols * full).clamp(min=0.0)
        if len(total_var) > 1 and float((total_var[1:] - total_var[:-1]).min()) < -1e-10:
            # every transition variance is a difference of these rows, so a decreasing total
            # variance would clamp into a degenerate transition whose survival saturates
            raise ValueError('total variance decreases across extendable-forward fixings')
        return {
            'first': first_index,
            'spot': s,
            'log_fwd': log_fwd,
            'total_var': total_var,
            'd_fix': utils.calc_discount_rate(discount_row, fix_days, shared)[0],
            'd_settle': utils.calc_discount_rate(discount_row, set_days, shared)[0],
        }

    def interp_batch(x_grid, y_grid, x):
        """Batched linear interpolation with linear extrapolation. x_grid/y_grid: [B,G]."""
        B, G = x_grid.shape
        x_flat = x.reshape(B, -1)
        idx = torch.searchsorted(x_grid.contiguous(), x_flat.contiguous(), right=True).clamp(1, G - 1)
        x0 = torch.gather(x_grid, 1, idx - 1)
        x1 = torch.gather(x_grid, 1, idx)
        y0 = torch.gather(y_grid, 1, idx - 1)
        y1 = torch.gather(y_grid, 1, idx)
        w = (x_flat - x0) / (x1 - x0).clamp(min=eps)
        return (y0 + w * (y1 - y0)).reshape_as(x)

    def root_from_continuation(x_grid, continuation):
        """The spot at which the continuation value crosses zero, per batch row."""
        # forward_sign makes the continuation increasing for both deal directions; cummax only
        # removes quadrature ripples for LOCATING the zero - the recursion keeps the raw values
        monotone = torch.cummax(continuation * forward_sign, dim=1).values
        count = (monotone <= 0.0).sum(dim=1)
        gather = count.clamp(1, x_grid.shape[1] - 1).reshape(-1, 1)
        x0 = torch.gather(x_grid, 1, gather - 1).squeeze(1)
        x1 = torch.gather(x_grid, 1, gather).squeeze(1)
        y0 = torch.gather(monotone, 1, gather - 1).squeeze(1)
        y1 = torch.gather(monotone, 1, gather).squeeze(1)
        root = x0 - y0 * (x1 - x0) / (y1 - y0).clamp(min=eps)
        root = torch.where(count == 0, x_grid[:, 0] * 0.5, root)
        root = torch.where(count >= x_grid.shape[1], x_grid[:, -1] * 2.0, root)
        return root

    boundary_cache = {}

    @torch.no_grad()
    def extension_boundaries(row_index, first_decision):
        """{global decision index: spot boundary} under the market state at row_index."""
        if first_decision is None or first_decision > n_fix - 2:
            return {}
        cache_key = (int(row_index), int(first_decision))
        if cache_key in boundary_cache:
            return boundary_cache[cache_key]

        data = market_inputs(row_index, first_decision)
        if data is None:
            return {}
        log_fwd, total_var = data['log_fwd'], data['total_var']
        d_fix, d_settle = data['d_fix'], data['d_settle']
        first = data['first']

        if style == 'strip':
            # one exercise date and a LINEAR continuation (a strip of K2 forwards), so the
            # zero-value boundary is closed-form
            m = extension_index - first
            future = slice(m + 1, None)
            rel_pay = d_settle[future] / d_fix[m]
            growth = torch.exp(log_fwd[future] - log_fwd[m])
            A = torch.sum(rel_pay * growth, dim=0)
            B = torch.sum(rel_pay, dim=0)
            result = {extension_index: (k2 * B / A.clamp(min=eps)).detach()}
            boundary_cache[cache_key] = result
            return result

        # rolling: C_j(s) = PV at f_j of the next K2 fixing plus the option to continue again, by
        # backward Gauss-Hermite recursion on a per-scenario spot grid. Chunked because the
        # [batch, grid, quadrature] temporary is otherwise too large for XVA batches
        B_total = log_fwd.shape[1]
        pieces = {j: [] for j in range(first_decision, n_fix - 1)}
        final_local = n_fix - 1 - first

        for lo in range(0, B_total, boundary_chunk):
            hi = min(lo + boundary_chunk, B_total)
            lf, tv = log_fwd[:, lo:hi], total_var[:, lo:hi]
            df, ds = d_fix[:, lo:hi], d_settle[:, lo:hi]

            rem_var = (tv[final_local] - tv[0]).clamp(min=eps).reshape(-1)
            half_width = torch.maximum(std_width * torch.sqrt(rem_var),
                                       rem_var.new_full(rem_var.shape, min_log_width))
            x_grid = torch.exp(half_width.reshape(-1, 1) * grid_u.reshape(1, -1))
            s_grid = k2 * x_grid
            continuation_values = {}

            for j in range(n_fix - 2, first_decision - 1, -1):
                a = j - first
                k = a + 1
                log_inc = (lf[k] - lf[a]).reshape(-1)
                var_inc = (tv[k] - tv[a]).clamp(min=eps).reshape(-1)
                d_pay = (ds[k] / df[a]).reshape(-1)
                d_cont = (df[k] / df[a]).reshape(-1)
                s_next = s_grid.unsqueeze(-1) * torch.exp(
                    (log_inc - 0.5 * var_inc).reshape(-1, 1, 1) +
                    torch.sqrt(var_inc).reshape(-1, 1, 1) * gh_z.reshape(1, 1, -1))
                value = d_pay.reshape(-1, 1, 1) * forward_sign * notional * (s_next - k2)
                if j + 1 <= n_fix - 2:
                    c_next = interp_batch(x_grid, continuation_values[j + 1], s_next / k2)
                    value = value + d_cont.reshape(-1, 1, 1) * F.relu(c_next)
                c_j = torch.sum(value * gh_w.reshape(1, 1, -1), dim=-1)
                continuation_values[j] = c_j
                pieces[j].append((k2 * root_from_continuation(x_grid, c_j)).detach())

        result = {j: torch.cat(parts, dim=0) for j, parts in pieces.items()}
        boundary_cache[cache_key] = result
        return result

    # the outer-path lifecycle: a supplied historical Yes/No is authoritative, otherwise the
    # decision is reconstructed from that scenario's own spot and boundary. `fact` twins `alive`
    # counting only the CODE-supplied decisions, which is the latch's alive branch
    boundary_aad = getattr(shared, 'boundary_aad', False)
    alive = shared.one.new_ones(shared.simulation_batch)
    fact = alive
    alive_enter, alive_after, fact_enter, fact_after, b_latch = [], [], [], [], []
    for j in range(n_fix):
        alive_enter.append(alive)
        fact_enter.append(fact)
        # a decision the outer grid never reaches is never read - the OSS pass owns every future
        # decision - and a state already dead on every path has no boundary to build
        if j in decision_set and fixing_days[j] <= max_mtm and bool((alive != 0).any()):
            code = int(decision_codes[j])
            if code >= 0:
                extend = alive.new_full(alive.shape, bool(code), dtype=torch.bool)
                fact = fact * extend.to(fact.dtype)
            else:
                row_index = row_by_fixing.get(j)
                if row_index is None:
                    raise ValueError('Extension fixing {} is missing from the MTM grid'.format(j))
                H = extension_boundaries(row_index, j).get(j)
                if H is None:
                    raise ValueError('Unable to construct extension boundary at fixing {}'.format(j))
                s_fix = observed_fixings[j]
                if s_fix is None:
                    s_fix = spot[row_index]
                extend = (s_fix > H) if forward_sign > 0 else (s_fix < H)
                if boundary_aad:
                    # gap > 0 means the latch FIRED - the deal terminated; the graph rides the
                    # scenario's own spot and the boundary is detached
                    gap = torch.log(H / s_fix) if forward_sign > 0 else torch.log(s_fix / H)
                    b_latch.append((j, gap, (~extend).detach()))
            alive = alive * extend.to(alive.dtype)
        alive_after.append(alive)
        fact_after.append(fact)

    sobol = False
    if shared.simulation_batch > 16:
        sobol = True
        shared.reset_qrg()

    def oss_extend(prev_spot, log_inc, var_inc, barrier, u):
        """Continuation probability and the extended-branch draw at one decision."""
        var_inc = var_inc.reshape(-1).clamp(min=eps)
        sd = torch.sqrt(var_inc).reshape(-1, 1)
        drift = (log_inc.reshape(-1) - 0.5 * var_inc).reshape(-1, 1)
        z_bound = (torch.log(barrier.reshape(-1, 1) / prev_spot) - drift) / sd
        p, Z = oss_truncated_draw(u, z_bound, survive_below=forward_sign < 0)
        return p, prev_spot * torch.exp(drift + sd * Z)

    def state_at(after, mtm_day):
        past = [j for j in decision_indices if fixing_days[j] <= mtm_day]
        return after[past[-1]] if past else shared.one.new_ones(shared.simulation_batch)

    # registered decisions in day order; a fixing's pending bucket is the count of them strictly
    # before its own day, and a row's `obs_before` the count at or before the row's
    reg_days = np.array([fixing_days[j] for j, _, _ in b_latch])
    mtm_values, b_alive_rows, b_trig_rows, b_pending = [], [], [], []
    for row_index, mtm_day in enumerate(mtm_days):
        start_idx = int(np.searchsorted(settlement_days, mtm_day, side='left'))
        if start_idx >= n_fix:
            mtm_values.append(shared.one.new_zeros(shared.simulation_batch))
            b_alive_rows.append(mtm_values[-1])
            b_trig_rows.append(mtm_values[-1])
            b_pending.append(None)
            continue

        data = market_inputs(row_index, start_idx)
        first = data['first']
        log_fwd, total_var = data['log_fwd'], data['total_var']
        d_fix, d_settle = data['d_fix'], data['d_settle']
        s0 = data['spot']
        pos = lambda j: j - first
        state_now = state_at(alive_after, mtm_day)
        fact_now = state_at(fact_after, mtm_day)
        # `live` is the STATE-FREE live part: value = fixed + state * live, and the latch's alive
        # branch is the same arithmetic at the facts-only state
        fixed = shared.one.new_zeros(shared.simulation_batch)
        fixed_fact = shared.one.new_zeros(shared.simulation_batch)
        trig_row = shared.one.new_zeros(shared.simulation_batch)
        buckets = {}
        live = shared.one.new_zeros(shared.simulation_batch)
        settlement_cf = shared.one.new_zeros(shared.simulation_batch)

        # fixed-but-unsettled cashflows survive a later decision not to extend: their alive state
        # is the state ENTERING their fixing, never the current post-decision state
        for j in range(start_idx, n_fix):
            if fixing_days[j] > mtm_day:
                break
            observed = observed_fixings[j]
            if observed is None:
                if fixing_days[j] == mtm_day:
                    observed = s0
                elif not bool((alive_enter[j] != 0).any()):
                    # terminated before this fixing on every path: a dead deal's unprinted dates
                    # are not missing data
                    continue
                else:
                    raise ValueError('Missing observed extendable-forward fixing {}'.format(j))
            strike_j = k1 if j <= extension_index else k2
            cf = alive_enter[j] * forward_sign * notional * (observed - strike_j)
            fixed = fixed + d_settle[pos(j)] * cf
            if settlement_days[j] == mtm_day:
                settlement_cf = settlement_cf + cf
            if boundary_aad:
                # the pending head, bucketed by how many REGISTERED decisions precede the fixing:
                # bucket 0 is certain given the facts and lives on the triggered profile, later
                # buckets are the survived-weighted slices of `pending`
                c = (d_settle[pos(j)] * fact_enter[j] * forward_sign * notional *
                     (observed - strike_j)).detach()
                fixed_fact = fixed_fact + c
                m = int(np.searchsorted(reg_days, fixing_days[j], side='left'))
                if m == 0:
                    trig_row = trig_row + c
                else:
                    buckets[m - 1] = buckets.get(m - 1, 0.0) + c

        if (settlement_cf != 0).any():
            cash_settle(shared, factor_dep['SettleCurrency'],
                        deal_data.Time_dep.deal_time_grid[row_index], settlement_cf)

        # future K1 fixings through the extension date are guaranteed and linear. The extension
        # fixing itself belongs here; the decision only controls LATER K2 fixings
        for j in range(max(start_idx, 0), extension_index + 1):
            if fixing_days[j] <= mtm_day:
                continue
            p = pos(j)
            expected_s = s0 * torch.exp(log_fwd[p])
            live = live + d_settle[p] * forward_sign * notional * (expected_s - k1)

        if style == 'strip':
            if mtm_day < fixing_days[extension_index]:
                H = extension_boundaries(row_index, extension_index)[extension_index]
                m = pos(extension_index)
                # market value at f_m of the whole K2 tail: C_m(S) = sign * N * (A * S - K2 * B)
                tail = [pos(j) for j in range(extension_index + 1, n_fix)]
                rel_pay = d_settle[tail] / d_fix[m]
                growth = torch.exp(log_fwd[tail] - log_fwd[m])
                A = torch.sum(rel_pay * growth, dim=0)
                B = torch.sum(rel_pay, dim=0)
                u = oss_uniforms(shared, 1, shared.MCMC_sims, sobol)[0]
                prev = s0.reshape(-1, 1).expand(-1, u.shape[-1])
                p_ext, s_ext = oss_extend(prev, log_fwd[m], total_var[m], H, u)
                c_ext = forward_sign * notional * (A.reshape(-1, 1) * s_ext - k2 * B.reshape(-1, 1))
                live = live + (d_fix[m].reshape(-1, 1) * p_ext * c_ext).mean(dim=1)
            else:
                # resolved: an exercised strip's remaining K2 fixings are ordinary forwards
                for j in range(max(start_idx, extension_index + 1), n_fix):
                    if fixing_days[j] <= mtm_day:
                        continue
                    p = pos(j)
                    expected_s = s0 * torch.exp(log_fwd[p])
                    live = live + d_settle[p] * forward_sign * notional * (expected_s - k2)

        else:  # rolling
            # a decision on the current fixing is resolved AT that fixing row; the next live
            # fixing, if any, was already created by it
            if mtm_day < fixing_days[extension_index]:
                first_decision = extension_index
            else:
                future_decisions = [j for j in decision_indices if fixing_days[j] > mtm_day]
                first_decision = future_decisions[0] if future_decisions else None

            boundaries = extension_boundaries(row_index, first_decision) \
                if first_decision is not None else {}
            n_future = len([j for j in decision_indices if fixing_days[j] > mtm_day])
            u_all = oss_uniforms(shared, n_future, shared.MCMC_sims, sobol) if n_future else None
            u_ptr = 0
            sims = 2 * shared.MCMC_sims
            L = shared.one.new_ones((shared.simulation_batch, sims))
            prev_spot = s0.reshape(-1, 1).expand(-1, sims)
            prev_index = None

            # before Extension_Date the first decision sits on a guaranteed K1 fixing: its K1
            # cashflow was valued above, and only the continuation branch starts the K2 chain
            if mtm_day < fixing_days[extension_index]:
                j = extension_index
                pj = pos(j)
                p_ext, prev_spot = oss_extend(prev_spot, log_fwd[pj], total_var[pj],
                                              boundaries[j], u_all[u_ptr])
                u_ptr += 1
                L = L * p_ext
                prev_index = j
                next_fixing = j + 1
            else:
                after = [j for j in range(extension_index + 1, n_fix) if fixing_days[j] > mtm_day]
                next_fixing = after[0] if after else n_fix

            P = shared.one.new_zeros((shared.simulation_batch, sims))
            for j in range(next_fixing, n_fix):
                pj = pos(j)
                if prev_index is None:
                    log_inc, var_inc = log_fwd[pj], total_var[pj]
                else:
                    pp = pos(prev_index)
                    log_inc = log_fwd[pj] - log_fwd[pp]
                    var_inc = (total_var[pj] - total_var[pp]).clamp(min=eps)

                expected_s = prev_spot * torch.exp(log_inc.reshape(-1, 1))
                P = P + d_settle[pj].reshape(-1, 1) * L * forward_sign * notional * (expected_s - k2)

                if j in decision_set:
                    p_ext, prev_spot = oss_extend(prev_spot, log_inc, var_inc,
                                                  boundaries[j], u_all[u_ptr])
                    u_ptr += 1
                    L = L * p_ext
                    # prev_index tracks where prev_spot LIVES - the last drawn decision - so a
                    # non-decision fixing must not advance it
                    prev_index = j

            live = live + P.mean(dim=1)

        value = fixed + state_now * live
        mtm_values.append(value)
        if boundary_aad:
            b_alive_rows.append((fixed_fact + fact_now * live).detach())
            b_trig_rows.append(trig_row)
            if buckets:
                j0 = min(buckets)
                width = max(buckets) - j0 + 1
                b_pending.append((j0, torch.stack(
                    [buckets.get(j0 + t, torch.zeros_like(trig_row)) for t in range(width)])))
            else:
                b_pending.append(None)

    mtm = torch.stack(mtm_values)

    # the reconstructed decisions' whole reach: the alive branch is the facts-only world, the dead
    # branch the survived-weighted pending head on `triggered`
    if boundary_aad and b_latch and time_grid.report_index is not None:
        latch = utils.LatchedBoundarySet(
            gaps=[g for _, g, _ in b_latch], fired=[f for _, _, f in b_latch],
            obs_before=np.searchsorted(reg_days, mtm_days, side='right'),
            untriggered=torch.stack(b_alive_rows),
            triggered=torch.stack(b_trig_rows).detach(),
            pending=b_pending,
            to_mtm=deal_to_mtm_grid(time_grid, deal_data, fx_rep),
            report_index=time_grid.report_index)
        shared.boundary_sets.append(latch)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                prefix = [torch.zeros_like(latch.fired[0])]
                for flag in latch.fired:
                    prefix.append(prefix[-1] | flag)
                recon = latch.select(prefix)
                logging.debug('EXTENDABLE_LATCH decisions=%d recon_max=%.3g scale=%.3g',
                              len(b_latch), float((recon - mtm.detach()).abs().max()),
                              float(mtm.detach().abs().max()))

    return mtm


def pv_MC_Tarf(shared, time_grid, deal_data, spot, fx_rep):
    """One-step-survival Monte Carlo for a TARF.

    Analytic KO-in-step (``(1 - p_survive) * remaining_target`` at each fixing), truncated draws for
    the surviving branch, and only the ITM leg accrues toward the target - the OTM leg affects PV
    only, plus the optional knock-in barrier.

    Heston-Nandi limitations - accepted design semantics, stated so they are not re-derived:
    (F1) each fixing spans ``n_sub = max(round(dt * Steps_Per_Year), 1)`` whole HN daily steps, so
         per-fixing variance is quantised to trading days with a floor of one;
    (F2) ``Steps_Per_Year`` must match the daily clock the HN factor was calibrated on - a mismatch
         silently rescales the variance horizon;
    (F3) the ``n_sub - 1`` unmonitored sub-step normals come from the global torch generator, so
         finite-difference greeks w.r.t. non-HN factors pick up RNG noise (AAD HN greeks do not);
    (F4) ``h`` re-seeds to ``H0`` at every MTM row - there is no outer-grid variance term structure.
    The HN scalars ride the AAD graph out of ``t_Static_Buffer``; when absent the GBM path is
    byte-identical. RECOMPUTE: the vol strip is built at the call site and the simulation RETURNS
    its settled cashflows and registrations - the node's inputs are its whole theta surface.

    BOUNDARY AAD registers two decisions taken on simulated state: the target FILLING (the LATCHED
    shape, since a later block's accrual is only ever larger) and the OTM leg KNOCKING IN (per inner
    path). The redemption gap is built from the UNCLAMPED accrual ``raw``, because
    ``accumulation == min(raw, target)`` and the clamp kills the derivative at the decision, so a
    gap built from the clamped series would carry no graph on the very paths that decided. It stays
    in target units, ``boundary_weights`` scaling its kernel by ``gap.std()``. A graphless historic
    decision is registered anyway: it contributes exactly zero, and skipping it would corrupt the
    latch reconstruction.

    BRANCH AND WEIGHT (``Branch_And_Weight: 'Yes'``, base valuation only) SWAPS the estimator for
    both of those registrations rather than joining them - one decision, one estimator, or the flux
    is counted twice. Off, and absent, is every line above bit for bit.

    Most of the construction is already here: the KO-in-step term IS the fired branch integrated
    against the fixing interval's own lognormal law, the moving trigger ``B_pnl = K + R/N`` IS
    ``K + R_{k-1}``, and the continuation IS the truncated draw. THE FIRED PAYMENT IS THE SAME ONE
    ON BOTH PATHS - the fixing that fills the target pays the remaining target ``R``, measurable one
    fixing back, so its conditional expectation is itself. What the switch adds:

    - THE KNOCK-IN INTEGRATED RATHER THAN SAMPLED. ``relu(-eff_intr) * 1{hit}`` is supported on one
      tail of the conditioning step's law, wholly inside the surviving set, so it closes in that
      law's own partial moments (``lognormal_fired_gain``) and the indicator - the whole reason for
      the ``InnerBoundarySet`` - leaves the tape. Rao-Blackwellisation, not smoothing.
    - THE PER-FIXING ACCRUAL KINK at second order (``accrual_kink_term``), added to ``cf_itm`` so it
      reaches BOTH of its consumers - the payment, and the running target ``R``, which is what
      carries an accrual across later decisions. Never built when curvature is not asked for.
    - THE TARGET PIN BECOMES EXACT: the pinned payment is a conditional expectation the loop
      evaluates in closed form, so there is nothing left to estimate or register. Crisp, that pin's
      flux resolves no better than ~10%.

    WHAT THE SWITCH DOES NOT SMOOTH: a decision taken on an OBSERVED sample. The redemption latch's
    gaps are the accrual of fixings the row has already seen, which is data rather than simulated
    state, so its indicator stays exact. Under base valuation those samples are historic prints
    carrying no graph, so the skipped registration contributes exactly zero.

    DECLARED LIMITATION: second-order variance in the conditioning step scales like ``s_k**-3``,
    which is fine at monthly and quarterly fixing spacing and rough at daily. A daily accumulator's
    economically right gamma is the desk's own call-spread width, which is a fixed smoothing
    converging at the ordinary rate; the measured spread is in ``tests/test_branch_and_weight.py``.
    """

    def accrued(s, k, C, inverted):
        """One fixing's accrual toward the target, standard or inverted."""
        if inverted:
            iv = (1.0 / s - 1.0 / k) * C * (-1.0)
        else:
            iv = (s - k) * C
        return F.relu(iv).reshape(-1, 1)

    def calc_accum_value(targetValue, accumulated, s, k, C, inverted):
        """The running accrual after one more fixing, clamped at the target."""
        return (accumulated + accrued(s, k, C, inverted)).clamp(max=targetValue)

    def sim_spot_tarf(settlement, sobol, num_sims,
                      spot_prices, times, carry, prev_accum, discount_rates, vols_all,
                      past_fixings, *hn_scalars):
        """Inner one-step-survival Monte Carlo over one block of MTM rows; the mean PV per row.

        PURE bound/theta split for ``InnerMCRecompute``; by-products are RETURNED and performed once
        by the caller. Returns ``(mtm, untriggered, settled, settle_rows, knock_rows) + gaps +
        jumps``: the block's rows, the redemption latch's counterfactual rows, the cashflows
        settling today, the block-local rows they and the knock-in decisions land on, then one gap
        and one jump per (row, fixing) - the gaps being the outputs whose cotangent carries the
        correction.

        Heston-Nandi: ``h`` re-seeds to ``H0`` per MTM row and recurses through every daily
        sub-step. Only the LAST step of a fixing is truncated - the TARF accrues and knocks AT it -
        and the PnL barrier depends only on the remaining target, constant over the interval and
        F-measurable at the truncation, so the scheme is exact.

        Two decisions fork here under ``boundary_aad``:

        - REDEMPTION: the accrual clamps AT ``targetValue``, so the equality fires on a
          positive-measure set. The counterfactual is the same loop on a weight that was never
          zeroed - one extra accumulator, nothing re-simulated, no draw consumed.
        - KNOCK-IN: decided on ``Sj``, one decision per inner path, with jump
          ``J(knocked in) - J(did not) = -Dj * L * p * relu(-intr) * N_otm`` and the gap in log
          space so a bandwidth means the same as at a barrier. At an already-observed fixing
          (``dt <= 0``) the decision is per SCENARIO and the gap is expanded along the inner axis -
          exact, because the pooled kernel's ``1/n`` weights cancel the ``n`` copies of each jump.
        """

        K = strike
        # the declared model as a walking kit - the pricer names it once, here
        kit = oss_model_kit(factor_dep, hn_scalars)
        hn = kit is not None
        # per-block results, and the by-products the caller performs once
        mcmc, alive, settled, gaps, jumps = [], [], [], [], []
        settle_rows, knock_rows = [], []
        for i, (D, s, carry_rate, delta_t, tau) in enumerate(zip(
                discount_rates, spot_prices, carry, times, settlement)):

            # the number of fixings still to simulate in this block
            reduced_samples = len(delta_t)
            if not hn:
                vols = vols_all[i]
            if reduced_samples:
                u = oss_uniforms(shared, reduced_samples, num_sims, sobol)

            Sj = torch.unsqueeze(s, 1)  # [batch, 1]
            # the HN variance state, re-seeded per MTM row
            st = kit.seed() if hn else None
            P = shared.one.new_zeros((shared.simulation_batch, 2*num_sims))
            L = shared.one.new_ones((shared.simulation_batch, 2*num_sims))
            remaining_target = targetValue-prev_accum
            # paths that have already redeemed
            q = remaining_target == 0.0
            # L_alive runs the same loop on a weight that was never zeroed - the left limit, not a
            # re-simulation
            L_alive = L if not boundary_aad else torch.ones_like(L)
            P_alive = P
            L = torch.where(q, torch.zeros_like(L), L)
            # started at the weight the observed prefix left, not at one - a redeemed path is dead
            # before the strip begins
            ledger = SurvivalLedger(L) if smooth else None
            # the remaining target in currency units, per path
            R = (remaining_target * factor_dep['Notional1']).expand_as(P)
            N_itm = notional1
            N_otm = notional2
            for j in range(reduced_samples):
                # `carry` and `vols` both arrive as INTERVAL strips (forward_carry_rate,
                # forward_vol_rate)
                dt = delta_t[j]
                Dj = D[j].reshape(-1, 1)
                use_past_fixing = False

                if dt > 0:
                    fwd_carry = carry_rate[j].reshape(-1, 1)  # [batch,1]
                    if not hn:
                        fwd_vol = vols[j].reshape(-1, 1)      # [batch,1]
                        vol_dt = fwd_vol * torch.sqrt(dt)
                else:
                    use_past_fixing = True

                if not use_past_fixing:
                    # ONE-STEP SURVIVAL against the PnL barrier: the step knocks out if
                    # (S_i - K)*cp >= R/N_itm and ITM, so the spot-level cap is
                    # B_pnl = K + (R/N_itm)*cp; an inverted target has the reciprocal form
                    N_i = N_itm
                    if not invertedTarget:
                        B_pnl = K + (R / N_i) * callOrPut # [batch, sims]
                    else:
                        # (1/S_i - 1/K)*(-cp) * N_i >= R, i.e. rhs = 1/K + (R/N)*(-cp) either way:
                        # a call needs S >= 1/(1/K - R/N), a put needs S <= 1/(1/K + R/N)
                        rhs = (1.0 / K) + (R / N_i) * (-callOrPut)
                        B_pnl = 1.0 / rhs

                    if hn:
                        # HN calibrates per day, so this fixing spans n_sub daily steps and only
                        # the LAST is truncated, which is what makes the scheme exact
                        n_sub = max(int(round(float(dt) * hn_spy)), 1)
                        b_step = fwd_carry * dt / n_sub  # per-step r-q; the total is fwd_carry*dt
                        # the first n_sub-1 unmonitored daily steps, antithetic to align with the
                        # u <-> 1-u halves of the truncated final draw below
                        Sj, st = kit.substeps(
                            Sj, st, b_step, n_sub - 1, shared, num_sims, antithetic=True)
                        h = kit.h(st)
                        fwd_drift = b_step - 0.5 * h  # the FINAL (monitored) daily step's drift
                        z_max = (torch.log(B_pnl / Sj) - fwd_drift) / torch.sqrt(h)
                    else:
                        # the lognormal cap standardised, off the current Sj as S_{i-1}
                        fwd_drift = (fwd_carry - 0.5 * fwd_vol * fwd_vol) * dt
                        z_max = (torch.log(B_pnl/Sj) - fwd_drift) / vol_dt
                    # the survival side follows the PnL cap's direction
                    p, Z = oss_truncated_draw(u[j], z_max, callOrPut > 0)

                    # the analytic KO-in-step: the knocked-out weight pays the REMAINING TARGET,
                    # discounted at the j-th point. R is measurable one fixing back, so
                    # `(1 - p) * R` IS its conditional expectation
                    P = P + (1.0 - p) * L * R * Dj
                    if boundary_aad:
                        P_alive = P_alive + (1.0 - p) * L_alive * R * Dj
                    # THE KNOCK-IN, INTEGRATED: `relu(-eff_intr) * 1{hit}` is supported on one tail
                    # of THIS interval's law, `otm_bound` carrying the strike's side and the
                    # barrier's, and lies wholly inside the surviving set for every R >= 0
                    otm_analytic = None
                    if smooth and otm_bound is not None:
                        z_otm = (torch.log(otm_bound / Sj) - fwd_drift) / vol_dt
                        otm_analytic = Dj * L * N_otm * (-gain_sign) * lognormal_fired_gain(
                            Sj, fwd_drift, vol_dt, z_otm, K, callOrPut < 0, gain_power)
                    if hn:
                        # HN increment + h-recursion on the truncated final draw; leverage-
                        # asymmetric because Z is survival-truncated (see hn_daily_advance)
                        Sj, st = kit.advance(Sj, st, b_step, Z)
                    else:
                        Sj = Sj * (torch.exp(fwd_drift + vol_dt * Z) if dt > 0 else 1.0)
                else:
                    Sj = past_fixings[-min(reduced_samples, num_samples)].reshape(-1, 1)
                    p = 1.0
                    # an OBSERVED fixing has no conditioning step to integrate against: its spot is
                    # data, so its knock-in is an exact indicator and its accrual carries no density
                    otm_analytic = None
                # the effective intrinsic in the target's own measure, clamped at the per-path
                # remaining target
                if not invertedTarget:
                    eff_intr = (Sj - K) * callOrPut
                else:
                    eff_intr = (1.0 / Sj - 1.0 / K) * (-callOrPut)
                intr = eff_intr.clamp(max=remaining_target)
                itm_mask = (intr > 0.0)
                # ONE kernel serves both legs: `relu(-x)` kinks exactly where `relu(x)` does and the
                # term is even, so the net cashflow carries `(N_itm - N_otm)` of it - which is also
                # why an unleveraged deal has no kink at all
                kink = accrual_kink_term(intr, j) if (
                    second_order and not use_past_fixing) else None
                if otm_analytic is not None:
                    # integrated above; nothing of this leg is sampled, so no indicator is formed
                    barrier_hit = torch.zeros_like(Sj)
                elif barrier > 0.0:
                    barrier_intr = (barrier - Sj) * callOrPut
                    barrier_hit = (barrier_intr >= 0.0).to(Sj.dtype)
                else:
                    barrier_hit = torch.ones_like(Sj)
                # the signed per-fixing cashflow
                cf_itm = F.relu(intr) * N_itm
                cf_otm = F.relu(-intr) * N_otm * barrier_hit
                if kink is not None:
                    # worth an exact zero forward and bit-zero at first order, so it rides the
                    # deal's own weights rather than a detached copy of them
                    cf_itm = cf_itm + kink * N_itm
                    if otm_analytic is None:
                        cf_otm = cf_otm + kink * N_otm * barrier_hit
                cf_step = L * p * (cf_itm - cf_otm)
                P = P + Dj * cf_step
                if otm_analytic is not None:
                    P = P - otm_analytic
                if boundary_aad:
                    P_alive = P_alive + Dj * L_alive * p * (cf_itm - cf_otm)
                    if barrier > 0.0:
                        # one decision per inner path; gap > 0 means KNOCKED IN, in log space, and
                        # `expand_as` also covers the already-observed fixing
                        jump = (-buy_sell * Dj * L * p * F.relu(-intr) * N_otm).detach()
                        gaps.append((callOrPut * torch.log(barrier / Sj)).expand_as(jump))
                        jumps.append(jump)
                        knock_rows.append(i)
                # the remaining target, decremented by this fixing's accrual on survivors
                accr = cf_itm  # correct for both standard and inverted targets
                if smooth:
                    # the mask is inert at value and first order - `accr` is an exact zero off it -
                    # and it would SEVER the kink term's curvature from the moving trigger that
                    # curvature has to reach
                    R = R - accr
                else:
                    R = torch.where(itm_mask, R - accr, R)  # no overshoot: R stays >= 0
                if smooth:
                    ledger.fire(p)
                    L = ledger.alive
                else:
                    L = p * L
                if boundary_aad:
                    L_alive = p * L_alive
                # a settlement at tau == 0 is RETURNED, because a recompute would settle twice
                if j == 0 and tau[0] == 0:
                    settled.append(cf_step.mean(axis=1))
                    settle_rows.append(i)
            if ledger is not None:
                ledger.check('TARF row {}'.format(i))
            mcmc.append(P.mean(axis=1))
            if boundary_aad:
                alive.append(P_alive.mean(axis=1))

        # a fresh empty each time, never one tensor returned twice - an autograd.Function output
        # list is positional and the same object in two slots is one output with two names
        return (torch.stack(mcmc),
                torch.stack(alive) if alive else prev_accum.new_empty(0),
                torch.stack(settled) if settled else prev_accum.new_empty(0),
                settle_rows, knock_rows) + tuple(gaps) + tuple(jumps)

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # `Fixings` includes settlement dates as well as fixing dates
    samples = factor_dep['Fixings']
    start_idx = samples.get_start_index(deal_time)
    start_index, counts = np.unique(start_idx, return_counts=True)

    fx_samples = factor_dep['Price_Fixings']
    known_resets = fx_samples.known_resets(shared.simulation_batch)
    sim_samples = fx_samples.schedule[
        (fx_samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (fx_samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
    next_samples = utils.calc_fx_cross(
        factor_dep['Underlying_Currency'][0], factor_dep['Currency'][0],
        sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
    all_samples = torch.cat(
        [torch.cat(known_resets, dim=0), next_samples], dim=0) if known_resets else next_samples
    num_samples = len(all_samples)

    settle_idx = np.searchsorted(factor_dep['Settlement'], deal_time[:, utils.TIME_GRID_MTM]).astype(np.int64)
    fixing_indices = counts.cumsum() - 1
    settle_index = settle_idx[fixing_indices]

    targetValue = deal_data.Instrument.field['TargetLevel']
    barrier = deal_data.Instrument.field.get('Barrier', 0.0)
    notional1 = factor_dep['Notional1'] * shared.one
    notional2 = factor_dep['Notional2'] * shared.one
    strike = factor_dep['Strike_Price']
    invertedTarget = deal_data.Instrument.field['InvertedTarget']
    callOrPut = factor_dep['Option_Type']
    buy_sell = factor_dep['Buy_Sell']

    # `resolved` counts the fixings taken off the realized path and is NOT `num_samples`:
    # `calc_fx_cross` on an empty schedule returns a broadcast row, so `len(all_samples)` reads 1
    # where nothing is resolved at all
    logging.debug('TARF %s fixings=%d resolved=%d target=%.6g barrier=%.6g blocks=%d',
                  deal_data.Instrument.field.get('Reference'), len(fx_samples.schedule),
                  len(known_resets) + len(sim_samples), targetValue, barrier, len(counts))

    # read once before a draw is taken so a non-GBM refusal lands first. It SUPERSEDES the two
    # registrations below rather than joining them - one decision, one estimator
    smooth = branch_and_weight(shared, deal_data)
    # `eff_intr = gain_sign * (S**gain_power - K**gain_power)` on BOTH targets, which is what lets
    # one closed form serve the inverted accrual (`lognormal_fired_gain`'s own `power`)
    gain_power = -1.0 if invertedTarget else 1.0
    gain_sign = -callOrPut if invertedTarget else callOrPut
    # the OTM leg's support once the knock-in is folded in: `relu(-eff_intr)` needs the OTM side of
    # the strike and `barrier_hit` needs the barrier's, so ONE bound carries both. In spot, not in
    # the accrual's units - the knock-in is declared on spot however the target is written
    otm_bound = (min(barrier, strike) if callOrPut > 0 else max(barrier, strike)) if barrier > 0.0 \
        else None
    # curvature is the switch: the per-fixing kink term is never BUILT when nobody asked for a
    # second derivative, so it costs exactly nothing on a first-order run
    second_order = smooth and getattr(shared, 'gamma', False)

    # gated: the target filling and the OTM knock-in both jump on simulated state
    boundary_aad = getattr(shared, 'boundary_aad', False) and not smooth
    b_gaps, b_fired, b_obs_before, b_inner, alive, row_ofs = [], [], [], [], [], 0

    # the declared model's parameter tensors, by ITS OWN canonical name tuple; () is GBM
    hn_scalars = oss_model_scalars(factor_dep, shared)
    hn = bool(hn_scalars)
    if hn:
        hn_spy = factor_dep['HN_Steps_Per_Year']


    # the accumulation to date, over the declared past values and then the simulated fixings
    acc = shared.one * 0.0
    for sample_val in fx_samples.schedule[:, utils.RESET_INDEX_Value]:
        if sample_val:
            acc = calc_accum_value(targetValue, acc, sample_val * shared.one, strike, callOrPut, invertedTarget)

    accumulation = [acc]
    for sample_val in next_samples:
        accumulation.append(calc_accum_value(
            targetValue, accumulation[-1], sample_val, strike, callOrPut, invertedTarget))

    # the UNCLAMPED accrual: the clamp kills the derivative AT the decision, so a gap needs `raw`
    if boundary_aad:
        raw = [acc]
        for sample_val in next_samples:
            raw.append(raw[-1] + accrued(sample_val, strike, callOrPut, invertedTarget))

    sobol = False
    # a quasi random generator for large batches; the counter is reset so that subsequent runs
    # reuse the same quasi random numbers
    if shared.simulation_batch > 16:
        sobol = True
        shared.reset_qrg()

    for b_idx, (discount_block, spot_block) in enumerate(
            utils.split_counts([discount, spot], counts, shared)):
        settle_index_local = settle_index[b_idx]
        t_block = discount_block.time_grid
        fixings = (fx_samples[np.newaxis, settle_index_local:, utils.RESET_INDEX_End_Day] -
                   t_block[:, utils.TIME_GRID_MTM, np.newaxis]).clip(min=0)
        drifts = utils.calc_fx_drift(
            deal_data.Factor_dep['Underlying_Currency'], deal_data.Factor_dep['Currency'],
            fixings, t_block, shared, multiply_by_time=False)
        fixing_block = daycount_fn(fixings)
        settlement = (factor_dep['Settlement'][np.newaxis, settle_index_local:] -
                      t_block[:, utils.TIME_GRID_MTM, np.newaxis])
        sample_ts = drifts.new(
            np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)]))
        discount_rates = utils.calc_discount_rate(discount_block, settlement, shared)
        cum_t = drifts.new(fixing_block)
        # the interval carry and vol strips, both theta rather than loop state. Both take the ZERO
        # carry `drifts`, a forward to each tenor being the CUMULATIVE integral. This pricer reads
        # the surface NOWHERE else, so the forward moneyness IS what it declares
        fwd_drifts = forward_carry_rate(drifts, cum_t, sample_ts)
        vols = forward_vol_rate(forward_vol_strip(
            deal_data, strike, spot_block, drifts, fixing_block, shared,
            factor_dep['Invert_Moneyness'], use_forwards=True), cum_t,
            sample_ts) if not hn else spot_block.new_empty(0)

        simulate = partial(sim_spot_tarf, settlement, sobol, shared.MCMC_sims)
        theta = (spot_block, sample_ts, fwd_drifts, accumulation[settle_index_local],
                 discount_rates, vols, all_samples) + hn_scalars
        # the SAME callable either way: under the node it is called twice
        outputs = InnerMCRecompute.run(shared, simulate, *theta)
        block_mtm, block_alive, block_settled, settle_rows, knock_rows = outputs[:5]
        theo_price = buy_sell * block_mtm

        # the by-products, performed once off the forward's result. The settled cash carries the
        # deal's direction exactly as the mtm does, or a sold TARF books its fixing as a receipt
        for row, value in zip(settle_rows, block_settled):
            cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
                time_grid.mtm_time_grid, t_block[row, utils.TIME_GRID_MTM]), buy_sell * value)

        if boundary_aad:
            alive.append(block_alive)
            # one knock-in decision per (row, fixing), each with the row it was taken on;
            # `fixed` names the tuple's fixed prefix - adding an output means moving it, loudly
            fixed, n_inner = 5, len(knock_rows)
            b_inner.extend([[row_ofs + row, outputs[fixed + k], outputs[fixed + n_inner + k]]
                            for k, row in enumerate(knock_rows)])
            # one latched redemption decision per block; `expand` because block 0's accrual is the
            # HISTORIC one, a single number rather than a per-scenario tensor
            b_gaps.append((raw[settle_index_local] - targetValue).squeeze(dim=1).expand(
                shared.simulation_batch))
            b_fired.append((targetValue - accumulation[settle_index_local] == 0.0).squeeze(
                dim=1).expand(shared.simulation_batch).detach())
            b_obs_before.extend([len(b_gaps)] * theo_price.shape[0])
        row_ofs += theo_price.shape[0]
        mtm_list.append(theo_price)

    mtm = torch.cat(mtm_list, dim=0)

    # report_index is None on a grid nobody reports off, where there is no reporting row for a
    # counterfactual to land on
    if boundary_aad and time_grid.report_index is not None:
        to_mtm = deal_to_mtm_grid(time_grid, deal_data, fx_rep)
        if b_gaps:
            # redemption: `triggered` is worth zero for every row the decision reaches, and
            # `untriggered` is the SAME loop run on a weight that was never zeroed
            untriggered = buy_sell * torch.cat(alive, dim=0).detach()
            shared.boundary_sets.append(utils.LatchedBoundarySet(
                gaps=b_gaps, fired=b_fired, obs_before=np.array(b_obs_before),
                untriggered=untriggered, triggered=torch.zeros_like(untriggered),
                to_mtm=to_mtm, report_index=time_grid.report_index))
        if b_inner:
            rows, gaps, jumps = zip(*b_inner)
            shared.boundary_sets.append(utils.InnerBoundarySet(
                gaps=list(gaps), rows=list(rows), jumps=list(jumps), reported=mtm.detach(),
                to_mtm=to_mtm, report_index=time_grid.report_index))

    return mtm


def pv_MC_AutoCallSwap(shared, time_grid, deal_data, spot, moneyness, fx_rep):
    """Autocallable swap by inner Monte Carlo, one-step-survival when there is no averaging.

    RECOMPUTE: ``sim_spot`` is a pure bound/theta split called through ``InnerMCRecompute`` - the
    floating leg, past fixings and Heston-Nandi scalars enter as theta, and settled cashflows and
    trigger registrations are RETURNED, not performed. THE VOL IS AN INTERVAL STRIP in both branches
    (``forward_vol_rate(forward_vol_strip(...))``), read at the moneyness the deal declares; the
    per-fixing smile convention is open in roadmap.md. Quanto bends the carry at the expiry read;
    compo prices the product S*X (``calc_vol_adjustment``).

    THE KNOCK-OUT LATCH IS CARRIED. ``terminationDate`` is stamped inside ``sim_spot`` at each
    observed fixing, returned as a by-product and handed to the next block's theta - an autocalled
    path settles its coupon once and marks zero from then on. The averaging branch cannot stamp it,
    its termination being a smoothed per-inner-path weight with no crisp per-scenario decision, and
    returns it unchanged.

    BOUNDARY AAD: the trigger's gap is decided on ``Sj`` INSIDE the simulation, so under the node it
    is an OUTPUT whose cotangent carries the correction. Each STAMPED decision (``tau == 0``)
    registers ONE ``LatchedBoundarySet`` entry carrying its whole reach: the carried latch killing
    every later row, the own-row override forking the decision row itself, and the ledger triple for
    the payment the decision controls, which a collateralised exposure reads through ``C_ts_te``.
    One decision is ONE counterfactual: an objective with a kink scores two partial counterfactuals
    differently from their sum. A re-observation of an old decision registers nothing.

    BRANCH AND WEIGHT REACHES THE NO-AVERAGING ARM (``Branch_And_Weight: 'Yes'``, base valuation
    only), and it SUPERSEDES the registration above rather than joining it. Most of the construction
    is already here: a constant coupon makes ``(1 - p) * L * coup * D_j`` a conditional expectation
    rather than a probability times a sample, the funding leg pays on the survival weight, and the
    continuation is the truncated draw. What the switch adds is the deal's SECOND per-fixing
    decision:

    THE PUT LEG IS INTEGRATED RATHER THAN SAMPLED. ``breach * (rebate - (1 - S/strike))`` is
    ``(S - strike * (1 - rebate)) * 1{S <= B} / strike`` - a gain over one tail of the interval's
    own law - so it closes in that law's partial moments (``lognormal_fired_gain``) and the
    indicator leaves the tape. Crisp it is a bare jump of ``rebate - (1 - B/strike)``, an exact zero
    only where the barrier sits ON the strike with no rebate: 18-22% off its own CRN delta ladder
    at a 70% barrier, against 0.00% on the strike.

    THE ORDER IN THE LOOP DECIDES THE FORM, and it is the thing to read before touching this. The
    coupon block computes the SURVIVAL ``p``, pays ``(1 - p) * L``, advances ``L`` to ``p * L``, and
    only then advances ``Sj`` by a survival-truncated draw; the barrier block runs after all four.
    So the sample it replaces is drawn from the law truncated to the surviving ``{S <= K}`` and
    already carries that fixing's ``p`` in ``L``, which makes the analytic term a CONDITIONAL
    expectation: the raw partial moments over the breach region, divided by ``p``. That division is
    spelled as the weight from before ``p`` entered ``L`` - exactly ``L / p``, without the ``0/0``
    on a fixing every path fires - and the region is intersected with the surviving set,
    ``min(B, K)``, so what is integrated lies inside the truncated support.

    WHAT STAYS AN INDICATOR, there being no conditioning step to integrate it against: a barrier
    date this iteration's coupon block did not advance ``Sj`` over. Two of the three ways that
    happens are EXACT - an OBSERVED fixing (``dt <= 0``, where ``Sj`` is the scenario's own spot and
    the breach is DATA) and a block opening on a fixing that is not aligned (``Sj`` is that coupon's
    own observed price fixing). DECLARED DEFECT, the third: on a barrier date whose coupon row is
    ``<= 0`` the indicator is exact on a STALE spot - the previous fixing's - and ``coupon_index``,
    un-advanced, mis-tenors the next coupon with it. It belongs to this arm rather than to the
    switch, which leaves the branch byte for byte; roadmap.md carries it with its remedy, and
    ``test_branch_and_weight.py::test_a_zero_coupon_barrier_date_reads_a_stale_spot`` pins it.

    THE AVERAGING ARM REFUSES BY NAME rather than no-opping: its termination is a smoothed
    per-inner-path weight with no crisp per-scenario decision to replace, and its own ``breached``
    is a hard indicator on the AVERAGE.
    """
    def sim_autocall(S, isBarrierDate, isFixingDate, isFloatDate, floating, threshold, coupon, terminationDate):
        """The AVERAGING arm's inner path walk: coupons trigger off the running average of spots
        and termination is a smoothed heaviside, so there is no crisp per-scenario decision."""
        avg = 0.0
        averageCounter = 0.0
        fx = isQuanto * (fixFXRate - 1.0) + 1.0

        resetTime = 0
        breachEvent = barrierIsHit - 0.01
        floatingTime = 0
        sumOfSpots = 0.0

        tMax = len(S)
        mtm_list = S.new_zeros((len(isFixingDate),) + S.shape[1:])

        for t in range(tMax):
            inforce = 1.0 * (terminationDate < 0.0)

            if isBarrierDate[t] > 0.0:
                breachEvent = inforce * (S[t] <= putBarrier)

            if isFixingDate[t] > 0.0:
                sumOfSpots = sumOfSpots + S[t]
                averageCounter = averageCounter + 1.0

            if isFloatDate[t] > 0.0:
                lastKnownFloatingRate = -floating[floatingTime]
                mtm_list[resetTime] = inforce * fx * lastKnownFloatingRate
                floatingTime = floatingTime + 1

                if coupon[t] <= 0.0:
                    resetTime = resetTime + 1

            if coupon[t] > 0.0:
                avg = sumOfSpots / averageCounter

                sumOfSpots = 0.0
                averageCounter = 0.0
                termination = inforce * smooth_heaviside_up(avg, threshold[t] * strike)
                mtm_list[resetTime] += termination * fx * (rebate + coupon[t])
                terminationDate = (1 - termination) * terminationDate + termination * resetTime
                breachEvent = (1 - termination) * breachEvent

                resetTime = resetTime + 1

        alive = 1.0 * (terminationDate < 0.0)
        breached = 1.0 * (breachEvent > 0.0)
        mtm_list[resetTime - 1] += alive * fx * (rebate + breached * (rebate - (strike - avg) / strike))

        return mtm_list.mean(axis=2)

    def sim_spot(offset, times, t_tenor, last_fixing, sobol, num_sims,
                 spot_prices, vols, carry, terminationDate, discount_rates, floating_leg,
                 past_fixings, *hn_scalars):
        """Inner one-step-survival Monte Carlo over one block of MTM rows; the mean PV per row.

        PURE bound/theta split for ``InnerMCRecompute``: leading arguments are the block's shape,
        bound per block; trailing ones are every graph-carrying tensor. ``times`` is bound because
        it is built from numpy and stays numpy on a block whose fixings have all been observed.

        Returns ``(mtm, settled, settle_rows, event_rows, terminationDate, alive) + gaps + fired +
        survived + cash_on`` - the marks, the by-products the caller performs once, then per
        decision: a gap (carrying the correction's cotangent), two detached branch coefficients, and
        the payment the trigger controls had it fired.

        ``terminationDate`` arrives as the latch a PRIOR block left (-1 alive, else the fixing index
        that killed the path), is stamped where a fixing is observed (``tau == 0``, off the
        scenario's own spot) and returned for the next block. The accumulators run ALIVE with the
        latch masking the exits: ``P`` is homogeneous in its initial weight, so the masked exit
        equals the killed weight bit for bit, and the same pass yields the ``alive`` rows the
        latched registration needs, which a killed weight could never produce.

        Under ``boundary_aad`` a row whose autocall is OBSERVED (``dt == 0``: the fixing IS the row,
        ``Sj`` the scenario's own spot) forks a counterfactual. The gap is signed so ``gap > 0``
        means FIRED (``p == 0``). Firing zeroes the weight, so the fired branch is exactly that
        coupon; the surviving branch (``P_cf``/``L_cf``) is the same accumulation on a weight the
        trigger never touched, created at most once per row.
        """

        timesteps, num_samples = times.shape
        # the declared model as a walking kit - the pricer names it once, here
        kit = oss_model_kit(factor_dep, hn_scalars)
        hn = kit is not None
        # the by-products the caller performs once
        settled, settle_rows, event_rows, gaps, fired, survived = [], [], [], [], [], []
        alive, cash_on = [], []

        isBarrierDate = BarrierDates[offset:]
        isFloatingDate = Floating[offset:]
        threshold = Threshold[offset:]
        coupon = Coupon[offset:]

        # exactly one fixing per coupon date means there is no averaging
        if factor_dep['no_averaging']:
            eps = torch.finfo(shared.one.dtype).eps
            fx = isQuanto * (fixFXRate - 1.0) + 1.0
            # block-entry latch state: a scenario that fires at THIS block's fixing settles its
            # coupon here and dies from the next block on
            dead = (terminationDate != -1).squeeze(1)
            mcmc = []
            for i, (tau, df, s, v, carry_rate, delta_t, floating) in enumerate(zip(
                    t_tenor, discount_rates, spot_prices, vols, carry, times, floating_leg)):

                # zero when only the floating leg is left
                reduced_samples = len(delta_t)
                if reduced_samples:
                    if sobol:
                        u = shared.quasi_rng(shared.simulation_batch, reduced_samples * num_sims)[1].T.reshape(
                            reduced_samples, shared.simulation_batch, -1)
                    else:
                        u = torch.rand([reduced_samples, shared.simulation_batch, num_sims],
                                       dtype=shared.one.dtype, device=shared.one.device)
                if last_fixing is None:
                    Sj = torch.unsqueeze(s, 1)
                    fixing_aligned = True
                else:
                    Sj = torch.unsqueeze(past_fixings[last_fixing], 1)
                    fixing_aligned = False
                if hn:
                    st = kit.seed()  # re-seed the variance state at the start of this MTM row

                P = torch.zeros((shared.simulation_batch, num_sims), dtype=shared.one.dtype, device=shared.one.device)
                L = torch.ones((shared.simulation_batch, num_sims), dtype=shared.one.dtype, device=shared.one.device)
                # the counterfactual for this row's OBSERVED autocall; it does not exist until the
                # trigger is met, at most once per row
                P_cf = L_cf = None
                # the accumulators run alive, so the ledger opens at one
                ledger = SurvivalLedger(L) if smooth else None
                # the vol is per INTERVAL, so it is picked off the strip at the coupon's own index
                # rather than hoisted out of the loop
                D = df.unsqueeze(2)

                coupon_index = 0

                for j, (coup, thresh, FloatingDate, barrier) in enumerate(
                        zip(coupon, threshold, isFloatingDate, isBarrierDate)):
                    # the conditioning step THIS iteration's coupon block advanced `Sj` over, or
                    # None - the put leg below integrates against it, and only a fresh one is one
                    interval = None

                    if FloatingDate > 0:
                        P = P + L * fx * -FloatingDate * D[j]
                        if P_cf is not None:
                            P_cf = P_cf + L_cf * fx * -FloatingDate * D[j]

                    if coup > 0:
                        K = thresh * strike
                        dt = delta_t[coupon_index] if fixing_aligned else 0.0
                        if dt > 0:
                            # `carry` arrives as the INTERVAL carry strip (forward_carry_rate)
                            forward_carry = carry_rate[coupon_index].reshape(-1, 1)
                            if hn:
                                # HN daily sub-stepping to the coupon date. The autocall knocks out
                                # only AT the coupon observation, so the OSS truncation - survival
                                # is spot BELOW K - applies only on the final daily step
                                n_sub = max(int(round(float(dt) * hn_spy)), 1)
                                b_step = forward_carry * dt / n_sub
                                Sj, st = kit.substeps(
                                    Sj, st, b_step, n_sub - 1, shared, num_sims, antithetic=False)
                                h = kit.h(st)
                                p = utils.norm_cdf(
                                    (torch.log(K / Sj) - (b_step - 0.5 * h)) / torch.sqrt(h))
                            else:
                                vol = v[coupon_index].reshape(-1, 1)  # [batch,1]
                                vol_dt = vol * torch.sqrt(dt)
                                p = utils.norm_cdf(
                                    (torch.log(K / Sj) - (forward_carry - 0.5 * vol * vol) * dt) / vol_dt)
                                if smooth and putBarrier > 0.0:
                                    # held for the barrier block a screen below: `Sj` before the
                                    # advance, this interval's `m` and `s`, the weight from BEFORE
                                    # `p` enters `L`, and the breach region inside the surviving set
                                    interval = (Sj, (forward_carry - 0.5 * vol * vol) * dt,
                                                vol_dt, L, min(putBarrier, K))
                        else:
                            p = torch.where(K > Sj, 1.0, 0.0)
                            if tau == 0.0:
                                # stamp the latch: the fixing IS this row, the decision is the
                                # scenario's own spot, and the marker is the fixing that killed it
                                terminationDate = torch.where(
                                    (terminationDate == -1) & (p == 0), float(offset), terminationDate)

                        if P_cf is not None:
                            # every LATER coupon is the same smooth p in both worlds - the weight
                            # never feeds back into Sj, so the fork is one extra accumulator rather
                            # than a second simulation
                            P_cf = P_cf + fx * (1 - p) * L_cf * coup * D[j]
                            L_cf = p * L_cf
                        elif boundary_aad and dt <= 0:
                            # the autocall is OBSERVED here (dt == 0), so Sj is the scenario's own
                            # spot and gap > 0 means the trigger FIRED. The branches are DEAD-AWARE:
                            # a scenario an earlier fixing latched has nothing left to jump
                            event_rows.append(i)
                            gaps.append(torch.log(Sj / K).squeeze(dim=1))
                            fired.append(torch.where(
                                dead, 0.0, (P + fx * L * coup * D[j]).mean(axis=1)).detach())
                            # the payment the trigger makes IF it fires, UNMASKED: which worlds
                            # reach it is the latch's question, and a scenario dead as booked is
                            # alive in the world where an earlier trigger is forced off
                            cash_on.append((fx * coup * D[j]).squeeze(1).detach())
                            P_cf, L_cf = P, L

                        # the payment this coupon makes: the knocked-out weight times the coupon,
                        # discounted from the row to the coupon's own date. On an OBSERVED coupon
                        # (dt <= 0) that discount is 1 and this IS the cash that changes hands.
                        coupon_cash = fx * (1 - p) * L * coup * D[j]
                        P = P + coupon_cash
                        if ledger is not None:
                            # `L * p` and `p * L` are the same IEEE product, so the ledger's
                            # advance is the crisp line's; what it adds is the fired mass beside
                            # it, spelled off the SAME `p`, which is what makes the identity exact
                            ledger.fire(p)
                            L = ledger.alive
                        else:
                            L = p * L
                        if tau == 0.0 and dt <= 0:
                            # SETTLE HERE, not at the bottom of the loop: `tau` is row-level, so a
                            # test down there holds for every coupon and would book the running `P`
                            # - the accumulated VALUE, not the payment - once per coupon
                            row_cash = torch.where(dead, 0.0, coupon_cash.mean(axis=1))
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug(
                                    'AUTOCALL_SETTLE row=%d j=%d coup=%.6g cash=%.6g P=%.6g',
                                    i, j, float(coup), float(row_cash.mean()), float(P.mean()))
                            settled.append(row_cash)
                            settle_rows.append(i)

                        if fixing_aligned:
                            # prevent underflow or overflow
                            safe_pu = torch.clamp(p * u[coupon_index], min=eps, max=1.0-eps)

                            if hn:
                                # the survival-truncated final draw and its h-recursion; at
                                # dt <= 0 (the terminal fixing) there is no interval to advance
                                if dt > 0:
                                    Sj, st = kit.advance(
                                        Sj, st, b_step, utils.norm_icdf(safe_pu))
                            else:
                                Sj = Sj * (torch.exp(
                                    (forward_carry - 0.5 * vol * vol) * dt + vol_dt * utils.norm_icdf(
                                        safe_pu)) if dt > 0 else 1.0)
                            coupon_index += 1
                        else:
                            fixing_aligned = True

                    if barrier > 0 and interval is not None:
                        # THE PUT LEG, INTEGRATED: `b_eff = min(B, K)` puts the breach region inside
                        # the SURVIVING set, and `L_prev` - the weight from before `p` entered `L` -
                        # IS `L / p` without the 0/0 where every path fired
                        Sj_prev, m, s, L_prev, b_eff = interval
                        z_b = (torch.log(b_eff / Sj_prev) - m) / s
                        P = P + L_prev * D[j] * fx * lognormal_fired_gain(
                            Sj_prev, m, s, z_b, strike * (1.0 - rebate), False) / strike
                    elif barrier > 0:
                        # no conditioning step this iteration: EXACT on the spot the deal names for
                        # an OBSERVED fixing and an unaligned block opening, but on a STALE spot
                        # where the coupon row is <= 0 (roadmap.md's Known-defects)
                        breach = torch.where(Sj <= putBarrier, 1.0, 0.0)
                        P = P + L * D[j] * fx * breach * (rebate - (1.0 - Sj / strike))
                        if P_cf is not None:
                            P_cf = P_cf + L_cf * D[j] * fx * breach * (rebate - (1.0 - Sj / strike))


                if ledger is not None:
                    ledger.check('AUTOCALL row {}'.format(i))
                row_value = P.mean(axis=1)
                alive.append(row_value)
                mcmc.append(torch.where(dead, torch.zeros_like(row_value), row_value))
                if P_cf is not None:
                    survived.append(torch.where(dead, 0.0, P_cf.mean(axis=1)).detach())
        else:
            isFixingDate = FixingDates[offset:]
            dt = times.unsqueeze(axis=2)
            # `vols` is the INTERVAL strip (forward_vol_rate), so this product is the interval
            # variance. The floor is the ZERO-LENGTH interval's width and only that, and both
            # consumers read the SAME variance
            var = vols * vols * dt
            var = torch.where(dt > 0, var, var.clamp(min=1e-4))
            drift = carry * dt - 0.5 * var
            vol = torch.sqrt(var)

            mcmc = []
            for i, (tau, D, s, r, sigma, floating) in enumerate(zip(
                    t_tenor, discount_rates, spot_prices, drift, vol, floating_leg)):
                if sobol:
                    z = shared.quasi_rng(shared.simulation_batch, num_samples * num_sims)[0].T.reshape(
                        num_samples, shared.simulation_batch, -1)
                else:
                    z = torch.randn([num_samples, shared.simulation_batch, num_sims],
                                    dtype=shared.one.dtype, device=shared.one.device)
                f1 = (r.unsqueeze(axis=2) + sigma.unsqueeze(axis=2) * z).cumsum(axis=0)
                mcmc_spot = s.reshape(1, -1, 1) * torch.exp(f1)
                val = sim_autocall(
                    mcmc_spot, isBarrierDate, isFixingDate, isFloatingDate, floating, threshold, coupon,
                    terminationDate)
                pv = (val * D).sum(axis=0)
                mcmc.append(pv)

                # RETURNED, because a replay would settle these twice
                if tau == 0.0:
                    settled.append(pv)
                    settle_rows.append(i)
            # the averaging branch carries NO latch out: its termination is a smoothed
            # per-inner-path weight, so there is no crisp per-scenario decision to stamp. The
            # incoming latch is respected (sim_autocall's `inforce`) and returned unchanged

        return (torch.stack(mcmc),
                torch.stack(settled) if settled else spot_prices.new_empty(0),
                settle_rows, event_rows, terminationDate,
                torch.stack(alive) if alive else spot_prices.new_empty(0),
                ) + tuple(gaps) + tuple(fired) + tuple(survived) + tuple(cash_on)

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    samples = factor_dep['Fixings']
    start_idx = samples.get_start_index(deal_time)
    dual_samples = samples.dual()
    start_index, counts = np.unique(start_idx, return_counts=True)

    if factor_dep['no_averaging']:
        # a reset can be prior to its coupon date, so the two schedules are tracked separately
        equity_samples = factor_dep['Price_Fixing']
        coupon_samples = factor_dep['Coupon_Fixing']
        eq_start_idx = equity_samples.get_start_index(deal_time)
        cp_start_idx = coupon_samples.get_start_index(deal_time)
        known_resets = equity_samples.known_resets(shared.simulation_batch)
        sim_samples = equity_samples.schedule[
            (equity_samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
            (equity_samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
        past_samples = utils.calc_time_grid_spot_rate(
            deal_data.Factor_dep['Equity'], sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
        all_eq_samples = torch.cat(
            [torch.cat(known_resets, dim=0), past_samples], dim=0) if known_resets else past_samples
        fixing_indices = counts.cumsum() - 1
        eq_start_index = eq_start_idx[fixing_indices]
        cp_start_index = cp_start_idx[fixing_indices]
        coupon_equity_index = equity_samples.schedule[:, utils.RESET_INDEX_Reset_Day].searchsorted(
            coupon_samples.schedule[:, utils.RESET_INDEX_Reset_Day], 'right') - 1
        coupon_equity_index = np.append(coupon_equity_index, coupon_equity_index[-1] + 1)

    if 'Forward' in factor_dep:
        resets = factor_dep['Cashflows'].Resets
        forward = utils.calc_time_grid_curve_rate(factor_dep['Forward'], deal_time, shared)
        old_resets = resets.get_simulated_resets(
            deal_time[:, utils.TIME_GRID_MTM].max(), factor_dep['Forward'], shared)

        cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
        cash_start_index = dict(zip(start_idx, cash_start_idx))
    else:
        # unused: there is no floating leg
        forward = discount

    BarrierDates = factor_dep['Barrier_Dates']
    if not factor_dep['no_averaging']:
        FixingDates = [1 if x != -1 else -1 for x in factor_dep['Price_Fixing']]
    Threshold = factor_dep['Autocall_Thresholds']
    Floating = factor_dep['Autocall_Floating']
    Coupon = factor_dep['Autocall_Coupons']
    putBarrier = factor_dep['Barrier']
    isQuanto = 1.0 * (deal_data.Instrument.field.get('Payoff_Type') == 'Quanto')
    fixFXRate = deal_data.Instrument.field.get('FixFXRate', 1.0)
    rebate = deal_data.Instrument.field.get('Rebate', 0.0)
    barrierIsHit = 1.0 * (deal_data.Instrument.field.get('BarrierIsHit') is not None)
    strike = factor_dep['Strike_Price']
    nominal = factor_dep['Buy_Sell'] * deal_data.Instrument.field['Units']
    terminationDate = -shared.one.new_ones(shared.simulation_batch, 1)

    # `averaging` changes the PRODUCT, not just the estimator: the OSS path integrates each
    # coupon's survival analytically, the averaging path simulates a mean of spots
    logging.debug('AUTOCALL %s coupons=%d thresholds=%d averaging=%d barrier=%.6g blocks=%d',
                  deal_data.Instrument.field.get('Reference'), int((np.asarray(Coupon) > 0).sum()),
                  len(Threshold), int(not factor_dep['no_averaging']), putBarrier, len(counts))

    # read once before a draw is taken so a non-GBM refusal lands first. It SUPERSEDES the
    # registration below rather than joining it - one decision, one estimator
    smooth = branch_and_weight(shared, deal_data)
    if smooth and not factor_dep['no_averaging']:
        raise ValueError(
            "Branch_And_Weight: 'Yes' is refused on {} because it averages - more than one price "
            'fixing per coupon, or a barrier date off the coupon dates, so the deal prices on the '
            'AVERAGING arm. That arm has no crisp per-scenario decision for the switch to replace: '
            'its termination is a smoothed per-inner-path weight (pricing.smooth_heaviside_up) and '
            'its breach is a hard indicator on the AVERAGE, whose conditioning law is the '
            'distribution of a MEAN of spots and not one fixing interval\'s lognormal - which the '
            'construction does not have (roadmap.md, "Branch and weight for TARFs and autocalls"). '
            'Pricing it here would put the smooth estimator\'s name on an estimator that is not it. '
            'What works today: book this deal with ONE price fixing per coupon date and every '
            "barrier date ON a coupon date, which is the no-averaging arm the switch does price; or "
            "run it with Branch_And_Weight: 'No', the default, which is the crisp estimator this "
            'deal already prices under, unchanged.'.format(
                deal_data.Instrument.field.get('Reference')))

    # the declared model's parameter tensors, by ITS OWN canonical name tuple; () is GBM. HN is
    # only wired into the no_averaging arm
    hn_scalars = oss_model_scalars(factor_dep, shared)
    hn = bool(hn_scalars)
    if hn:
        hn_spy = factor_dep['HN_Steps_Per_Year']


    sobol = False
    # a quasi random generator for large batches; the counter is reset so that subsequent runs
    # reuse the same quasi random numbers
    if shared.simulation_batch > 16:
        sobol = True
        shared.reset_qrg()

    # an autocall taken on the reporting row's own spot is a real redemption, so the value is not
    # what is wrong; what ordinary AAD drops is the flux of scenarios across the threshold
    boundary_aad = getattr(shared, 'boundary_aad', False) and not smooth
    row_ofs = 0
    b_latch, b_obs, b_alive, b_cash = [], [], [], []

    for index, (forward_block, discount_block, spot_block, moneyness_block) in enumerate(
            utils.split_counts([forward, discount, spot, moneyness], counts, shared)):

        t_block = discount_block.time_grid
        sample_index_t = start_index[index]

        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM]
        all_fixings = (dual_samples.np[np.newaxis, sample_index_t:, utils.RESET_INDEX_End_Day] -
                       t_block[:, utils.TIME_GRID_MTM, np.newaxis])
        fixings = (factor_dep['Price_Fixing'].schedule[np.newaxis, eq_start_index[index]:, utils.RESET_INDEX_End_Day] -
                   t_block[:, utils.TIME_GRID_MTM, np.newaxis]) if factor_dep['no_averaging'] else all_fixings

        drifts = utils.calc_eq_drift(
            deal_data.Factor_dep['Equity_Zero'], deal_data.Factor_dep['Dividend_Yield'],
            fixings, t_block, shared, multiply_by_time=False)
        fixing_block = daycount_fn(fixings)

        # the EXPIRY read - a deal-level quantity, which is all the quanto/compo adjustment wants.
        # It is NOT what the simulation steps on: that is the interval strip below
        expiry = daycount_fn(tenor_block)
        expiry_vols = utils.calc_time_grid_vol_rate(
            factor_dep['Volatility'], moneyness_block, expiry, shared)

        adj = None
        if factor_dep.get('Check_Payoff_Type', False):
            # strike and thresholds are payoff-currency quantities, so they compare against the
            # scaled spot as authored
            adj = calc_vol_adjustment(factor_dep, deal_time, expiry, expiry_vols, shared, fixings)
            expiry_vols = adj['vol']
            drifts = drifts + adj['carry_adj']
            spot_block = spot_block * adj['spot_scale']

        sample_ts = drifts.new(
            np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)])
        ) if fixing_block.any() else fixing_block

        if 'Forward' in factor_dep:
            cashflows = factor_dep['Cashflows'].dual(cash_start_index[sample_index_t])
            reset_offset = factor_dep['Cashflows'].offsets[cash_start_index[sample_index_t]][1]
            reset_ofs, reset_count = np.unique(
                np.searchsorted(
                    resets.schedule[reset_offset:, utils.RESET_INDEX_Reset_Day],
                    t_block[:, utils.TIME_GRID_MTM]),
                return_counts=True)

            floating_legs = []
            forward_blocks = forward_block.split_counts(
                reset_count, shared) if len(reset_count) > 1 else [forward_block]

            for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
                time_block = t_block[:, utils.TIME_GRID_MTM]
                time_slice = time_block[:size].reshape(-1, 1)
                reset_block = resets.dual(reset_offset)
                future_starts = reset_block.np[offset:, utils.RESET_INDEX_Start_Day] - time_slice
                future_ends = reset_block.np[offset:, utils.RESET_INDEX_End_Day] - time_slice
                future_weights = (reset_block.tn[offset:, utils.RESET_INDEX_Weight] /
                                  reset_block.tn[offset:, utils.RESET_INDEX_Accrual]).reshape(1, -1, 1)
                future_resets = torch.expm1(forward_rates.gather_weighted_curve(
                    shared, future_ends, future_starts)) * future_weights

                past_resets = torch.tile(
                    torch.unsqueeze(old_resets[reset_offset:reset_offset + offset], dim=0), [size, 1, 1])
                all_resets = torch.concat([past_resets, future_resets], dim=1) if past_resets.any() else future_resets
                all_int, _ = pricer_float_cashflows(all_resets, cashflows.tn, shared)
                # the trailing axis is the inner-simulation one
                floating_legs.append(torch.unsqueeze(all_int, dim=-1))

            floating_leg = torch.concat(floating_legs) if len(floating_legs) > 1 else floating_legs[0]
        else:
            # unused: there is no floating leg
            floating_leg = spot_block

        discount_rates = utils.calc_discount_rate(discount_block, all_fixings, shared)
        if boundary_aad and factor_dep['no_averaging']:
            # decisions latched STRICTLY BEFORE this block's rows, at block granularity - the
            # block's own fixing is priced by its own simulation (the barrier's spelling)
            b_obs.extend([len(b_latch)] * len(t_block))
        # skip the simulation once EVERY scenario has autocalled - nothing is left to price
        if (terminationDate == -1).any():
            if factor_dep['no_averaging']:
                fixing_index = coupon_equity_index[cp_start_index[index]]
                last_fixing = None if fixing_index == eq_start_index[index] else fixing_index
            else:
                last_fixing = None
            # the interval carry and vol strips, which is also what puts the AVERAGING branch's
            # `carry * dt` and `vols * vols * dt` on interval integrals. Both take the ZERO carry
            # `drifts`; the strip takes the spot moneyness this pricer marks its Europeans with
            cum_t = drifts.new(fixing_block) if fixing_block.any() else fixing_block
            fwd_drifts = forward_carry_rate(
                drifts, cum_t, sample_ts) if fixing_block.any() else drifts
            interval_vols = forward_vol_rate(forward_vol_strip(
                deal_data, strike * shared.one, spot_block, drifts, fixing_block, shared),
                cum_t, sample_ts) if fixing_block.any() else expiry_vols.unsqueeze(-2)
            if adj is not None and adj['fx_vol'] is not None and fixing_block.any():
                # compo: the simulation steps S*X, so each interval's vol is the PRODUCT's; the
                # expiry-read else-branch above is already compo via adj['vol']
                interval_vols = compo_vol(interval_vols, adj['fx_vol'].unsqueeze(1), adj['rho'])
            simulate = partial(sim_spot, sample_index_t, sample_ts,
                               all_fixings[:, 0], last_fixing, sobol, shared.MCMC_sims)
            theta = (spot_block, interval_vols, fwd_drifts, terminationDate, discount_rates,
                     floating_leg,
                     all_eq_samples if factor_dep['no_averaging'] else spot_block.new_empty(0)
                     ) + hn_scalars
            # the SAME callable either way: under the node it is called twice
            outputs = InnerMCRecompute.run(shared, simulate, *theta)
            # `terminationDate` comes back stamped by this block's observed fixing and is handed to
            # the NEXT block's theta - an autocalled path pays once and is worth nothing after
            theo_cashflow, block_settled, settle_rows, event_rows, terminationDate, block_alive = \
                outputs[:6]
            # the by-products, performed once off the forward's result. `nominal` is
            # `Buy_Sell * Units` and scales the MARK three lines down, so the settled cash carries
            # the same scaling
            for row, value in zip(settle_rows, block_settled):
                cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
                    time_grid.mtm_time_grid, t_block[row, utils.TIME_GRID_MTM]), nominal * value)
            fixed, n_events = 6, len(event_rows)
            if boundary_aad and factor_dep['no_averaging']:
                b_alive.append(block_alive)
                settle_map = dict(zip(settle_rows, block_settled))
                # per STAMPED decision (tau == 0): gap, flag (`gap >= 0` IS the trigger), the
                # own-row fork, and the payment as a per-event `(mtm_row, amount, booked)` fact in
                # reporting currency. A pending window's re-observation is not a new decision
                for k, row in enumerate(event_rows):
                    if all_fixings[row, 0] != 0.0:
                        continue
                    gap = outputs[fixed + k]
                    b_latch.append([gap, (gap >= 0).detach(), row_ofs + row,
                                    outputs[fixed + n_events + k],
                                    outputs[fixed + 2 * n_events + k]])
                    fxr = (fx_rep[row_ofs + row] if fx_rep.dim() > 1 and
                           fx_rep.shape[0] > 1 else fx_rep.reshape(-1)[0])
                    b_cash.append((int(np.searchsorted(
                        time_grid.mtm_time_grid, t_block[row, utils.TIME_GRID_MTM])),
                        (nominal * fxr * outputs[fixed + 3 * n_events + k]).detach(),
                        (nominal * fxr * settle_map[row]).detach()))
        else:
            theo_cashflow = drifts.new_zeros((len(t_block), shared.simulation_batch))
            if boundary_aad and factor_dep['no_averaging']:
                # every scenario resolved: both branches are the same zero, so the rows carry no
                # counterfactual - the barrier's all_hit discipline
                b_alive.append(theo_cashflow)
        theo_price = nominal * theo_cashflow
        row_ofs += theo_price.shape[0]
        mtm_list.append(theo_price)

    mtm = torch.cat(mtm_list, dim=0)

    if b_latch and time_grid.report_index is not None:
        # one counterfactual per decision, its whole reach: latch + own-row fork + every payment
        # row the flip touches. Branches stay on the pricer's grid; report_index is None on a grid
        # nobody reports off
        gaps, flags, own_rows, on_v, off_v = zip(*b_latch)
        untriggered = (nominal * torch.cat(b_alive, dim=0)).detach()
        shared.boundary_sets.append(utils.LatchedBoundarySet(
            gaps=list(gaps), fired=list(flags), obs_before=np.array(b_obs),
            untriggered=untriggered, triggered=torch.zeros_like(untriggered),
            own_row=[(r, nominal * a, nominal * b)
                     for r, a, b in zip(own_rows, on_v, off_v)],
            cash_events=b_cash,
            to_mtm=deal_to_mtm_grid(time_grid, deal_data, fx_rep),
            report_index=time_grid.report_index))

    return mtm


def pv_discrete_asian_option(shared, time_grid, deal_data, nominal, spot, forward,
                             past_factor_list, invert_moneyness=False, use_forwards=False):
    """Discretely sampled Asian option by two-moment matching: the average of the remaining samples
    is fitted to a lognormal and priced by Black against a strike net of the realised average.

    Past the averaging period the value is the intrinsic on the realised average, priced at a token
    maturity so the AAD graph stays defined.
    """
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    expiry = daycount_fn(factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
    # clamped away from zero: the carry below divides by it
    safe_expiry = spot.new(expiry.reshape(-1, 1)).clamp(min=1e-5)
    b = torch.log(forward / spot) / safe_expiry
    eps = torch.finfo(shared.one.dtype).eps
    samples = factor_dep['Samples']
    known_resets = samples.known_resets(shared.simulation_batch)
    start_idx = samples.get_start_index(deal_time)
    sim_samples = samples.schedule[
        (samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]

    # an unsimulated spot is held flat across the samples
    if spot.shape != forward.shape:
        past_samples = spot.expand(sim_samples.shape[0], shared.simulation_batch)
        spot = spot.expand(*forward.shape)
    else:
        past_sample_factor = [utils.calc_time_grid_spot_rate(
            past_factor, sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
            for past_factor in past_factor_list]
        past_samples = past_sample_factor[0] if len(
            past_sample_factor) == 1 else past_sample_factor[0] / past_sample_factor[1]

    all_samples = torch.cat(
        [torch.cat(known_resets, dim=0), past_samples], dim=0) if known_resets else past_samples
    dual_samples = samples.dual()

    start_index, counts = np.unique(start_idx, return_counts=True)

    for index, (discount_block, spot_block, forward_block, carry_block) in enumerate(
            utils.split_counts([discount, spot, forward, b], counts, shared)):
        t_block = discount_block.time_grid
        sample_index_t = start_index[index]
        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM]

        sample_tau = daycount_fn(
            dual_samples.np[sample_index_t:, utils.RESET_INDEX_End_Day].reshape(1, -1) -
            t_block[:, utils.TIME_GRID_MTM, np.newaxis])
        sample_ts = carry_block.new(sample_tau)

        weight_t = dual_samples.tn[sample_index_t:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
        normalize = dual_samples.tn[sample_index_t:, utils.RESET_INDEX_Weight].sum()
        average = torch.sum(
            all_samples[:sample_index_t] * dual_samples.tn[:sample_index_t, utils.RESET_INDEX_Weight].reshape(-1, 1),
            dim=0)

        if sample_tau.size:
            # strike_bar goes negative when the realised average exceeds the strike, which Black
            # cannot take
            strike_bar = torch.clamp(
                factor_dep['Strike'] - average.reshape(1, -1).expand(counts[index], -1), min=1e-5)
            sample_fwd = spot_block.unsqueeze(1) * torch.exp(carry_block.unsqueeze(1) * sample_ts.unsqueeze(2))
            moneyness = calc_moneyness(
                (strike_bar / normalize.clamp(min=eps)).unsqueeze(1), spot_block.unsqueeze(1), sample_fwd,
                deal_data, use_forwards, invert_moneyness)
            # the vol at each sample's own tenor (TODO: generalise if vols become time-dependent)
            vols = torch.stack([utils.calc_time_grid_vol_rate(
                factor_dep['Volatility'], mon, s_tau, shared) for mon, s_tau in zip(moneyness, sample_tau)])
            if factor_dep.get('Check_Payoff_Type', False):
                adj = [calc_vol_adjustment(
                    factor_dep, deal_time, s_tau, vol, shared) for s_tau, vol in zip(sample_tau, vols)]
                vols = torch.stack([x['vol'] for x in adj])
                carry_block = torch.stack([cb + x['b_adj'] for cb, x in zip(carry_block, adj)])
            else:
                carry_block = torch.unsqueeze(carry_block, dim=1)

            sample_ft = weight_t * torch.exp(carry_block * torch.unsqueeze(sample_ts, dim=2))

            M1 = torch.sum(sample_ft, dim=1)
            product_t = sample_ft * torch.exp(torch.unsqueeze(sample_ts, dim=2) * (vols * vols))
            sum_t = F.pad(torch.cumsum(product_t[:, :-1], dim=1), [0, 0, 1, 0, 0, 0])
            M2 = torch.sum(sample_ft * (product_t + 2.0 * sum_t), dim=1)

            # the clamps keep the logs and the root defined, so the gradients are not NaN
            MM = torch.log(M2.clamp(min=eps)) - 2.0 * torch.log(M1.clamp(min=eps))
            var_t = torch.where((M1 > eps) & (M2 > eps), MM, 0.0)
            vol_t = torch.where(var_t > 0.0, torch.sqrt(var_t.clamp(min=eps)), 0.0)
            tau = 1.0
            forward_price = M1 * spot_block
        else:
            # past the averaging period, so the intrinsic is known
            strike_bar = factor_dep['Strike']
            # tau cannot be exactly zero without breaking the AAD
            tau = 1e-5
            vol_t = torch.zeros_like(average)
            forward_price = average

        if factor_dep['Digital']:
            theo_price = utils.black_european_option(
            forward_price, strike_bar, vol_t, tau,
            factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared, cash_payoff=nominal)
            value = theo_price
        else:
            theo_price = utils.black_european_option(
                forward_price, strike_bar, vol_t, tau,
                factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared)
            value = nominal * theo_price

        discount_rates = torch.squeeze(
            utils.calc_discount_rate(discount_block, tenor_block.reshape(-1, 1), shared),
            dim=1)

        mtm_list.append(value * discount_rates)

    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], value[-1])
    mtm = torch.cat(mtm_list, dim=0)

    return mtm


def pv_discrete_double_asian_option(shared, time_grid, deal_data, nominal, spot, forward,
                                    past_factor_list, invert_moneyness=False, use_forwards=False):
    """Spread option on two discretely sampled averages, by two-moment matching on each leg and
    Bjerksund-Stensland on the pair. Past both averaging periods it is the intrinsic on the
    realised spread."""
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    expiry = daycount_fn(factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
    # clamped away from zero: the carry below divides by it
    safe_expiry = spot.new(expiry.reshape(-1, 1)).clamp(min=1e-5)
    b = torch.log(forward / spot) / safe_expiry
    # the alpha multipliers, usually just 1
    alphas = [factor_dep['Alpha_1'], factor_dep['Alpha_2']]
    # the resets of both samples merged, so a block boundary serves either schedule
    sample_reset_days = np.union1d(
        *[factor_dep[i].schedule[:, utils.RESET_INDEX_Reset_Day] for i in ['Samples_1', 'Samples_2']])
    start_idx = np.searchsorted(sample_reset_days, deal_time[:, utils.TIME_GRID_MTM], side='right').astype(np.int64)
    start_index, counts = np.unique(start_idx, return_counts=True)

    all_samples = []
    dual_samples = []
    start_samples = []

    for i in ['Samples_1', 'Samples_2']:
        samples = factor_dep[i]
        sample_idx = samples.get_start_index(deal_time, offset=1)
        known_resets = samples.known_resets(shared.simulation_batch)
        sim_samples = samples.schedule[(samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                                       (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                        deal_time[:, utils.TIME_GRID_MTM].max())]

        # an unsimulated spot is held flat across the samples
        if spot.shape != forward.shape:
            past_samples = spot.expand(sim_samples.shape[0], shared.simulation_batch)
            spot = spot.expand(*forward.shape)
        else:
            past_sample_factor = [utils.calc_time_grid_spot_rate(
                past_factor, sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
                for past_factor in past_factor_list]
            past_samples = past_sample_factor[0] if len(
                past_sample_factor) == 1 else past_sample_factor[0] / past_sample_factor[1]

        full_sample = torch.cat(
            [torch.cat(known_resets, dim=0), past_samples], dim=0) if known_resets else past_samples
        dual_sample = samples.dual()
        dual_samples.append(dual_sample)
        # the samples carry their weights already applied
        all_samples.append(full_sample * dual_sample.tn[:, utils.RESET_INDEX_Weight].reshape(-1, 1))
        # this sample's index relative to the merged resets above
        start_samples.append({x: y for x, y in zip(start_idx, sample_idx)})

    for index, (discount_block, spot_block, forward_block, carry_block) in enumerate(
            utils.split_counts([discount, spot, forward, b], counts, shared)):
        t_block = discount_block.time_grid
        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM]

        # moment matching only applies before expiry
        if tenor_block.any():
            # at-the-money vols
            moneyness_block = (forward_block if use_forwards else spot_block) / spot_block
            moneyness = 1.0 / moneyness_block if invert_moneyness else moneyness_block
            vols = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, daycount_fn(tenor_block), shared)

            mu = []
            sigma = []
            lambdas = []
            sample_fts = []
            sample_tss = []

            for alpha, start_idx, dual_sample, all_sample in zip(alphas, start_samples, dual_samples, all_samples):
                sample_index_t = start_idx[index]
                sample_ts = carry_block.new(
                    daycount_fn(dual_sample.np[sample_index_t:, utils.RESET_INDEX_End_Day].reshape(1, -1) -
                                t_block[:, utils.TIME_GRID_MTM, np.newaxis]))

                weight_t = dual_sample.tn[sample_index_t:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
                sample_ft = weight_t * torch.exp(
                    torch.unsqueeze(carry_block, dim=1) * torch.unsqueeze(sample_ts, dim=2))
                M1 = torch.sum(sample_ft, dim=1)
                # the realized average so far
                average = torch.sum(all_sample[:sample_index_t], dim=0)

                product_t = sample_ft * torch.exp(
                    torch.unsqueeze(sample_ts, dim=2) * torch.unsqueeze(vols * vols, dim=1))
                sum_t = F.pad(torch.cumsum(product_t[:, :-1], dim=1), [0, 0, 1, 0, 0, 0])
                M2 = torch.sum(sample_ft * (product_t + 2.0 * sum_t), dim=1)

                # the clamp keeps the root defined, so the gradients are not NaN
                MM = torch.log(M2) - 2.0 * torch.log(M1)
                MM_ok = MM.clamp(min=1e-6)
                vol_t = torch.sqrt(MM_ok)

                mu.append(M1)
                sigma.append(vol_t)
                sample_fts.append(sample_ft)
                sample_tss.append(sample_ts)
                lambdas.append(alpha * average)

            min_ts = torch.minimum(sample_tss[0].unsqueeze(2), sample_tss[1].unsqueeze(1))  # [T, N, N]
            sum_rho = torch.sum(
                sample_fts[0].unsqueeze(2) * sample_fts[1].unsqueeze(1) *
                torch.exp(min_ts * vols.pow(2).unsqueeze(1).unsqueeze(1)), dim=1)

            M_rho = torch.sum(sum_rho, dim=1)
            MM_rho = (torch.log(M_rho) - torch.log(mu[0]) - torch.log(mu[1])) / (sigma[0] * sigma[1])
            K_bar = factor_dep['Alpha_0'] * factor_dep['Strike'] - lambdas[0] + lambdas[1]

            theo_price = factor_dep['Buy_Sell'] * utils.Bjerksund_Stensland(
                factor_dep['Option_Type'], -factor_dep['Option_Type'],
                -factor_dep['Option_Type'] * K_bar, alphas[0] * spot_block * mu[0], alphas[1] * spot_block * mu[1],
                K_bar, sigma[0], sigma[1], MM_rho, factor_dep['Option_Type'])
        else:
            lambdas = [alpha * all_sample.sum(axis=0) for alpha, all_sample in zip(alphas, all_samples)]
            K_bar = factor_dep['Alpha_0'] * factor_dep['Strike'] - lambdas[0] + lambdas[1]
            theo_price = factor_dep['Buy_Sell'] * smooth_relu(-factor_dep['Option_Type'] * K_bar)

        discount_rates = torch.squeeze(
            utils.calc_discount_rate(discount_block, tenor_block.reshape(-1, 1), shared),
            dim=1)

        cash = nominal * theo_price
        mtm_list.append(cash * discount_rates)

    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])
    mtm = torch.cat(mtm_list, dim=0)

    return mtm


def pv_energy_option(shared, time_grid, deal_data, nominal):
    """Average-price energy option: the remaining forward-price samples, converted at their own fx
    forwards, are two-moment matched to a lognormal and priced by Black against a strike net of the
    realised average. Past the sampling period it is the intrinsic on that average."""
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    samples = factor_dep['Cashflow'].Resets
    known_samples = samples.known_resets(shared.simulation_batch)
    start_idx = samples.get_start_index(deal_time)
    sim_samples = samples.schedule[
        (samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
    fx_spot = utils.calc_fx_cross(
        factor_dep['ForwardFX'][0], factor_dep['CashFX'][0], sim_samples, shared)
    fx_rep = utils.calc_fx_cross(
        factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)
    all_samples = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], sim_samples, shared)

    sample_values = all_samples.gather_weighted_curve(
        shared, sim_samples[:, utils.RESET_INDEX_End_Day, np.newaxis] if sim_samples.size else np.zeros((1, 1)),
        multiply_by_time=False) * torch.unsqueeze(fx_spot, dim=1)

    past_samples = torch.squeeze(
        torch.cat([torch.stack(known_samples), sample_values], dim=0)
        if known_samples else sample_values, dim=1)

    forwards = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    dual_samples = samples.dual()
    start_index, start_counts = np.unique(start_idx, return_counts=True)

    for index, (forward_block, discount_block) in enumerate(utils.split_counts(
            [forwards, discounts], start_counts, shared)):

        t_block = discount_block.time_grid
        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM].reshape(-1, 1)

        sample_t = dual_samples[start_index[index]:]
        average = torch.sum(
            past_samples[:start_index[index]] *
            dual_samples.tn[:start_index[index], utils.RESET_INDEX_Weight].reshape(-1, 1),
            dim=0)

        if sample_t.np.any():
            sample_ts = np.tile(
                sample_t.np[np.newaxis, :, utils.RESET_INDEX_End_Day],
                [t_block.shape[0], 1])
            weight_t = sample_t.tn[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)

            future_resets = forward_block.gather_weighted_curve(
                shared, sample_ts, multiply_by_time=False)

            forwardfx = utils.calc_fx_forward(
                factor_dep['ForwardFX'], factor_dep['CashFX'],
                sample_t.np[:, utils.RESET_INDEX_Start_Day], t_block, shared)

            sample_ft = weight_t * future_resets * forwardfx

            # the tenor each sample's vol is read at
            sample_block = daycount_fn(
                sample_t.np[:, utils.RESET_INDEX_Start_Day].reshape(1, -1)
                - t_block[:, utils.TIME_GRID_MTM, np.newaxis])

            M1 = torch.sum(sample_ft, dim=1)
            strike_bar = factor_dep['Strike'] - average

            moneyness = calc_moneyness(
                strike_bar.unsqueeze(0), None, future_resets * forwardfx,
                deal_data, use_forward=True)

            # the vol at each sample's own tenor (TODO: generalise if vols become time-dependent)
            vols = torch.stack([utils.calc_time_grid_vol_rate(
                factor_dep['Volatility'], mon, s_tau, shared) for mon, s_tau in zip(moneyness, sample_block)])

            vol2 = vols * vols

            # a compo deal prices the product, so each sample's variance is the PRODUCT's
            if 'FXCompoVol' in factor_dep:
                fx_vols = torch.stack(
                    [utils.calc_time_grid_vol_rate(factor_dep['FXCompoVol'], None, sb, shared)
                     for sb in sample_block])
                vol2 += fx_vols * fx_vols + 2.0 * fx_vols * vols * utils.implied_correlation(
                    factor_dep['ImpliedCorrelation'])

            product_t = sample_ft * torch.exp(
                sample_ft.new(np.expand_dims(sample_block, axis=2)) * vol2)

            # an exclusive cumsum on axis=1
            sum_t = F.pad(torch.cumsum(product_t[:, :-1], dim=1), [0, 0, 1, 0, 0, 0])
            M2 = torch.sum(sample_ft * (product_t + 2.0 * sum_t), dim=1)
            MM = torch.log(M2) - 2.0 * torch.log(M1)
            # the clamp keeps the root defined, so the gradients are not NaN
            MM_ok = MM.clamp(min=1e-5)
            vol_t = torch.sqrt(MM_ok)
            theo_price = utils.black_european_option(
                M1, strike_bar, vol_t, 1.0,
                factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared)
        else:
            forward_p = average.reshape(1, -1)
            theo_price = factor_dep['Buy_Sell'] * smooth_relu(
                factor_dep['Option_Type'] * (forward_p - factor_dep['Strike']))

        discount_rates = torch.squeeze(
            utils.calc_discount_rate(discount_block, tenor_block, shared), dim=1)

        cash = nominal * theo_price
        mtm_list.append(cash * discount_rates)

    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])
    mtm = fx_rep * torch.cat(mtm_list, dim=0)

    return mtm


def pricer_float_cashflows(all_resets, t_cash, shared):
    """Plain floating interest per cashflow, `(rate + margin) * accrual`, and the margin term."""
    margin = (t_cash[:, utils.CASHFLOW_INDEX_FloatMargin] * t_cash[:, utils.CASHFLOW_INDEX_Year_Frac])
    all_int = all_resets * t_cash[:, utils.CASHFLOW_INDEX_Year_Frac].reshape(1, -1, 1) + margin.reshape(1, -1, 1)

    return all_int, margin


def _cap_floor_payoff(all_resets, strike, factor_dep, expiries, tenor, call_or_put, shared,
                      vol_expiries):
    """One caplet/floorlet payoff per (rate, strike, expiry) row - the law both aggregation
    shapes share: the period pricer applies it to one compounded rate per cashflow, the daily
    pricer to every reset in the period."""
    expiry = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](expiries)
    vol_expiry = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](vol_expiries)
    digital_payoff = factor_dep['Digital_Payoff_Rate'] if 'Digital_Payoff_Rate' in factor_dep else 0.0
    dist, shf = factor_dep['VolSurface'][0][utils.FACTOR_INDEX_SubType]
    pricing_fn = utils.black_european_option if dist == 'Lognormal' else utils.bachelier_european_option

    if digital_payoff and factor_dep.get('Digital_Spread', 0.0) > 0:
        # call/put spread replication, with vols re-queried at the two spread strikes so the smile
        # is picked up. The half-width is ABSOLUTE in rate - a relative width dies at a zero strike
        # and flips its legs at a negative one - and `call_or_put` orients the difference
        eps = factor_dep['Digital_Spread']
        strike_lo = strike - eps
        strike_hi = strike + eps
        mn_lo = -100.0 * (all_resets - strike_lo)
        mn_hi = -100.0 * (all_resets - strike_hi)
        vols_lo = utils.calc_tenor_cap_time_grid_vol_rate(
            factor_dep['VolSurface'], mn_lo, vol_expiry, tenor, shared)
        vols_hi = utils.calc_tenor_cap_time_grid_vol_rate(
            factor_dep['VolSurface'], mn_hi, vol_expiry, tenor, shared)
        return digital_payoff * call_or_put * (
            pricing_fn(all_resets, strike_lo, vols_lo, expiry, 1.0, call_or_put, shared, shift=shf.amount) -
            pricing_fn(all_resets, strike_hi, vols_hi, expiry, 1.0, call_or_put, shared, shift=shf.amount)
        ) / (2.0 * eps)

    mn_option = -100.0 * (all_resets - strike)
    vols = utils.calc_tenor_cap_time_grid_vol_rate(
        factor_dep['VolSurface'], mn_option, vol_expiry, tenor, shared)
    return pricing_fn(
        all_resets, strike, vols, expiry, 1.0, call_or_put, shared,
        cash_payoff=digital_payoff, shift=shf.amount)


def _pricer_cap_floor(all_resets, t_cash, factor_dep, expiries, tenor, call_or_put, shared, vol_expiries):
    """One optionlet per cashflow on that cashflow's compounded rate, times its year fraction."""
    strike = t_cash[:, utils.CASHFLOW_INDEX_Strike].reshape(1, -1, 1)
    payoff = _cap_floor_payoff(
        all_resets, strike, factor_dep, expiries, tenor, call_or_put, shared, vol_expiries)
    all_int = t_cash[:, utils.CASHFLOW_INDEX_Year_Frac].reshape(1, -1, 1) * payoff
    margin = shared.one.new_zeros(len(t_cash))

    return all_int, margin


def _pricer_cap_floor_daily(all_resets, strike, accrual, expiries, tenor, factor_dep,
                            call_or_put, lengths, shared):
    """Every reset of an aggregated period as its own optionlet, summed back to the period -
    the cap-then-compound convention (`Averaging_Method: Pre_Aggregation`), where the period
    pricer's compound-then-cap is the other order and Jensen separates the two."""
    payoff = _cap_floor_payoff(
        all_resets, strike, factor_dep, expiries, tenor, call_or_put, shared, expiries)
    daily_int = accrual.view(1, -1, 1) * payoff
    all_int = torch.segment_reduce(daily_int, reduce="sum", lengths=lengths, axis=1)
    margin = shared.one.new_zeros(lengths.shape[1])

    return all_int, margin


def pricer_cap(all_resets, t_cash, factor_dep, expiries, tenor, shared, vol_expiries):
    """Caplet interest per cashflow."""
    return _pricer_cap_floor(all_resets, t_cash, factor_dep, expiries, tenor, 1.0, shared, vol_expiries)


def pricer_floor(all_resets, t_cash, factor_dep, expiries, tenor, shared, vol_expiries):
    """Floorlet interest per cashflow."""
    return _pricer_cap_floor(all_resets, t_cash, factor_dep, expiries, tenor, -1.0, shared, vol_expiries)


def pv_float_cashflow_list(shared: utils.Calculation_State, time_grid: utils.TimeGrid, deal_data: utils.DealDataType,
                           cashflow_pricer, mtm_currency=None, settle_cash=True):
    """The floating leg. Its one invariant is CANONICAL FORM: by the time the fold below runs,
    every cashflow row carries exactly ONE effective reset, so a row is a rate and an accrual.

    Three reductions restore that form, and each lives somewhere different:

    - **Averaging** happens at COMPILE. `make_float_cashflows` writes `Weight = 1/n` on each of a
      row's `n` resets and `get_simulated_resets` applies `Weight / Accrual`, so several fixings
      arrive as one already-averaged rate. The weight is baked in, so it must never reach a path
      that compounds - a `1/n` weighted reset compounded geometrically compounds at `1/n` of its
      rate.
    - **OIS** happens HERE, and the switch is a SHAPE difference:
      `all_resets.shape[1] != reset_cashflows.np.shape[0]` means a row owns several resets, which is
      what `compress_no_compounding(groupsize=-1)` produces. Those are compounded geometrically by
      `segment_reduce` back to one rate per row.
    - **Method compounding** is the ordered ragged fold at the end, on a different axis: several
      cashflow ROWS sharing one payment date (`cash_counts > 1`), each already canonical,
      accumulated in order with the margin placed by convention.

    The three margin conventions differ only in where the spread sits. Writing `int` for
    `(rate + margin) * accrual`, `mrg` for `margin * accrual` and `N` for the nominal:

    | method | fold |
    | --- | --- |
    | `Include_Margin` | `total + int * (total + N)` - everything compounds |
    | `Flat` | `total + int * N + total * (int - mrg)` |
    | `Exclude_Margin` | `comp = comp + (int - mrg) * (comp + N)`; `simple += mrg * N`; pays `comp + simple` |

    Under `Flat`, cashflow i pays `P.I_i.(1+J_{i+1})...(1+J_n)`: the FULL interest enters the
    compounding pot and the pot grows at rate-only `J = I - m.alpha`. Under `Exclude_Margin` it pays
    `P.m_i.alpha_i + P.J_i.(1+J_{i+1})...(1+J_n)`: each period's margin is SIMPLE on the nominal and
    only the rate part compounds. The two differ by `sum_i m_i.alpha_i.N.(prod_{j>i}(1+J_j) - 1)` at
    any positive spread, so Exclude cannot be a single-accumulator fold - a margin lump inside
    `total` would earn `(1+J)` in every later step, which is exactly `Flat`.
    """
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    resets = factor_dep['Cashflows'].Resets

    if mtm_currency:
        sim_fx_forward = utils.calc_fx_forward(
            mtm_currency, factor_dep['Currency'], factor_dep['Cashflows'].FXResets[:, utils.RESET_INDEX_Reset_Day],
            factor_dep['Cashflows'].FXResets[:, :utils.RESET_INDEX_Scenario + 1], shared, only_diag=True)

        known_fx = factor_dep['Cashflows'].known_resets(
            shared.simulation_batch, utils.CASHFLOW_INDEX_FXResetValue, utils.CASHFLOW_INDEX_FXResetDate)

        # the fx FORWARDS at each reset, which differ slightly from the spot fx rate
        old_fx_rates = (torch.cat([torch.stack(known_fx), sim_fx_forward], dim=0)
                        if known_fx else sim_fx_forward).squeeze(dim=1)

    forwards = utils.calc_time_grid_curve_rate(factor_dep['Forward'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    old_resets = resets.get_simulated_resets(
        deal_time[:, utils.TIME_GRID_MTM].max(), factor_dep['Forward'], shared)

    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
    start_index, start_counts = np.unique(cash_start_idx, return_counts=True)

    for index, (forward_block, discount_block) in enumerate(utils.split_counts(
            [forwards, discounts], start_counts, shared)):

        # a dual (numpy and tensor) representation of the same cashflows
        cashflows = factor_dep['Cashflows'].dual(start_index[index])

        cash_pmts, cash_index, cash_counts = np.unique(
            cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

        reset_offset = factor_dep['Cashflows'].offsets[start_index[index]][1]
        pmts_offset = cash_index + (cash_counts - 1)

        time_ofs = 0
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        reset_block = resets.dual(reset_offset)
        reset_ofs, reset_count = np.unique(resets.split_block_resets(
            reset_offset, time_block), return_counts=True)

        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        # the forward block splits again where rows within it own different reset counts
        forward_blocks = forward_block.split_counts(
            reset_count, shared) if len(reset_count) > 1 else [forward_block]

        payments = []

        for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
            time_slice = time_block[time_ofs:time_ofs + size].reshape(-1, 1)
            future_starts = reset_block.np[offset:, utils.RESET_INDEX_Start_Day] - time_slice
            future_ends = reset_block.np[offset:, utils.RESET_INDEX_End_Day] - time_slice
            future_weights = (reset_block.tn[offset:, utils.RESET_INDEX_Weight]
                              / reset_block.tn[offset:, utils.RESET_INDEX_Accrual]).reshape(1, -1, 1)
            future_resets = torch.expm1(forward_rates.gather_weighted_curve(
                shared, future_ends, future_starts)) * future_weights

            all_resets = torch.cat(
                [old_resets[reset_offset:reset_offset + offset].expand(size, -1, -1), future_resets], dim=1)

            # cashflows that pay no interest (bullets) own no resets and are held out of the fold
            if cashflows.np[:, utils.CASHFLOW_INDEX_NumResets].all():
                reset_cashflows = cashflows
            else:
                reset_cash_index = np.where(cashflows.np[:, utils.CASHFLOW_INDEX_NumResets])[0]
                reset_cashflows = cashflows[reset_cash_index]
                cash_counts *= (cashflows.np[:, utils.CASHFLOW_INDEX_NumResets] >= 1).astype(np.int32)
                cash_index = reset_cash_index.searchsorted(cash_index)

            if mtm_currency:
                fx_past_index = start_index[index]
                if cashflows.np[:, utils.CASHFLOW_INDEX_FXResetDate].min() < time_slice.max():
                    fx_end_index = fx_past_index+1
                else:
                    # forward starting: there are no past fx resets
                    fx_end_index = fx_past_index

                past_fx_resets = old_fx_rates[fx_past_index: fx_end_index].expand(size, -1, -1)

                future_fx_resets = utils.calc_fx_forward(
                    mtm_currency, factor_dep['Currency'],
                    cashflows.np[past_fx_resets.shape[1]:, utils.CASHFLOW_INDEX_FXResetDate],
                    discount_block.time_grid[time_ofs:time_ofs + size], shared)

                all_fx_resets = torch.cat([past_fx_resets, future_fx_resets], dim=1)

                # the nominal in the resetting currency, and the next period's
                Pi = all_fx_resets * cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1)
                Pi_1 = F.pad(Pi[:, 1:, :], [0, 0, 0, 1, 0, 0])

            time_ofs += size

            needs_aggregation = all_resets.shape[1] != reset_cashflows.np.shape[0]
            if needs_aggregation:
                # OIS: a row owning several resets is the SHAPE the compile side produces on
                # purpose - see docs_src/developer/quote_sensitivities.md#curve-contracts
                reset_per_cashflows = factor_dep['Cashflows'].offsets[start_index[index]:, 0]
                accrual = reset_block.tn[:, utils.RESET_INDEX_Accrual]  # aligns with all_resets[1]
                reset_split = torch.as_tensor(
                    reset_per_cashflows[reset_per_cashflows > 0], device=shared.one.device, dtype=torch.long)
                lengths = reset_split.unsqueeze(0).expand(all_resets.shape[0], -1)
                if factor_dep['AveragingMethod'] != 'Pre_Aggregation':
                    # compound the resets into one rate per cashflow; Pre_Aggregation keeps them
                    # separate for the daily pricer below, which caps each BEFORE aggregating.
                    # Dropping log1p/expm1 would make this an average rate rather than a compound
                    log_rt = torch.log1p(all_resets * accrual.view(1, -1, 1))
                    sum_log = torch.segment_reduce(log_rt, reduce="sum", lengths=lengths, axis=1)
                    sum_acc = torch.segment_reduce(accrual, reduce="sum", lengths=reset_split, axis=0)
                    all_resets = torch.expm1(sum_log)/sum_acc.view(1, -1, 1)

            if cashflow_pricer in [pricer_cap, pricer_floor]:
                # the vol surface's tenor axis: averaged because the cashflows are meant to share
                # one year fraction, which in practice they do not exactly
                tenor = cashflows.np[:, utils.CASHFLOW_INDEX_Year_Frac].mean()
                expiries = cashflows.np[:, utils.CASHFLOW_INDEX_Start_Day] - time_slice
                if factor_dep['AveragingMethod'] == 'Pre_Aggregation' and needs_aggregation:
                    # each reset in the period is its own optionlet - its own dates, accrual and a
                    # strike broadcast from its parent cashflow - rather than one option on the
                    # compounded period
                    daily_ends = reset_block.np[:, utils.RESET_INDEX_End_Day] - time_slice
                    strike_daily = torch.repeat_interleave(
                        reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Strike], reset_split).reshape(1, -1, 1)
                    call_or_put = 1.0 if cashflow_pricer is pricer_cap else -1.0
                    all_int, all_margin = _pricer_cap_floor_daily(
                        all_resets, strike_daily, accrual, daily_ends, tenor,
                        factor_dep, call_or_put, lengths, shared)
                else:
                    if factor_dep['AveragingMethod'] == 'Post_Aggregation':
                        # an option on the average: the vol is read at the period END and the Black
                        # time carries the averaging decay, variance accruing on the not-yet-fixed
                        # remainder, integrated exactly inside the period
                        t_k = cashflows.np[:, utils.CASHFLOW_INDEX_Start_Day] - time_slice
                        t_kp1 = cashflows.np[:, utils.CASHFLOW_INDEX_End_Day] - time_slice
                        expiries = t_kp1
                        eff_expiries = np.where(
                            (t_k <= 0) & (t_kp1 > 0),
                            t_kp1 ** 3 / (3 * (t_kp1 - t_k) ** 2),
                            t_kp1 - (2 / 3) * (t_kp1 - t_k))
                    else:
                        eff_expiries = expiries
                    all_int, all_margin = cashflow_pricer(
                        all_resets, reset_cashflows.tn, factor_dep, eff_expiries, tenor, shared, expiries)
            else:
                all_int, all_margin = cashflow_pricer(all_resets, reset_cashflows.tn, shared)

            # the common case: no method compounding, so rows sharing a payment date simply sum
            if mtm_currency is None and factor_dep['CompoundingMethod'] in ['None', 'OIS']:
                interest = all_int * reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1)
                if (cash_counts == 1).all():
                    total = interest
                else:
                    split_interest = torch.split(interest, tuple(cash_counts), dim=1)
                    total = torch.stack([i.sum(dim=1) for i in split_interest], dim=1)
            else:
                # a ragged fold pads by one, so the default offset lands on an inert row
                if cash_counts.min() != cash_counts.max():
                    interest = F.pad(all_int, [0, 0, 0, 1, 0, 0])
                    nominal = F.pad(reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal], [0, 1])
                    margin = F.pad(all_margin, [0, 1])
                else:
                    interest = all_int
                    margin = all_margin
                    nominal = reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal]

                default_offst = np.ones(cash_index.size, dtype=np.int32) * (interest.shape[1] - 1)
                total = 0.0
                # Exclude_Margin cannot be a single-accumulator fold: period i's margin is paid
                # SIMPLE, and a margin lump inside `total` would earn (1+J) in every later step -
                # which is exactly Flat. The simple pot stays outside and joins at the end.
                simple_margin = 0.0

                for i in range(cash_counts.max()):
                    offst = default_offst.copy()
                    offst[cash_counts > i] = i + cash_index[cash_counts > i]
                    int_i = interest[:, offst]

                    if mtm_currency:
                        total = total + int_i * Pi + (Pi - Pi_1)
                    elif factor_dep['CompoundingMethod'] == 'Include_Margin':
                        total = total + int_i * (total + nominal[offst].reshape(1, -1, 1))
                    elif factor_dep['CompoundingMethod'] == 'Flat':
                        # cashflow i pays P.I_i.(1+J_{i+1})...(1+J_n): the FULL interest enters the
                        # pot and the pot compounds at rate-only J = I - m.alpha
                        total = total + (int_i * nominal[offst].reshape(1, -1, 1)) + total * (
                                int_i - margin[offst].reshape(1, -1, 1))
                    elif factor_dep['CompoundingMethod'] == 'Exclude_Margin':
                        # cashflow i pays P.m_i.alpha_i + P.J_i.(1+J_{i+1})...(1+J_n): only the rate
                        # part compounds, each period's margin being simple on the nominal
                        margin_i = margin[offst].reshape(1, -1, 1)
                        nominal_i = nominal[offst].reshape(1, -1, 1)
                        total = total + (int_i - margin_i) * (total + nominal_i)
                        simple_margin = simple_margin + margin_i * nominal_i
                    else:
                        raise Exception(
                            'Floating cashflow list method {} not implemented'.format(
                                factor_dep['CompoundingMethod']))

                total = total + simple_margin

            payments.append(total + cashflows.tn[
                pmts_offset, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1))

        all_payments = torch.cat(payments, dim=0) if len(payments) > 1 else payments[0]

        if settle_cash:
            cash_settle(shared, factor_dep['SettleCurrency'],
                        np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), all_payments[-1][0])
        mtm_list.append(torch.sum(all_payments * discount_rates, dim=1))

    return torch.cat(mtm_list, dim=0)


def pv_fixed_cashflows(shared, time_grid, deal_data, ignore_fixed_rate=False, settle_cash=True):
    """The fixed leg. `ignore_fixed_rate` prices it at a unit rate, which is what a par-rate solve
    divides by; `Settlement_Amount` makes it a forward, carried to the settlement date on repo and
    discounted back on the settlement curve."""
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    schedule = factor_dep['Cashflows']
    cash_start_idx = schedule.get_cashflow_start_index(deal_time)
    settlement_amt = factor_dep.get('Settlement_Amount', 0.0)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    # a forward-settled deal carries its own repo curve
    repo_code = factor_dep.get('Repo_Rate', factor_dep['Discount'])
    repo = utils.calc_time_grid_curve_rate(repo_code, deal_time, shared)
    settlement_code = factor_dep.get('Settlement_Rate', factor_dep['Discount'])
    settlement = utils.calc_time_grid_curve_rate(settlement_code, deal_time, shared)

    start_index, counts = np.unique(cash_start_idx, return_counts=True)

    for index, [discount_block, repo_block, settle_block] in enumerate(utils.split_counts(
            [discounts, repo, settlement], counts, shared)):
        cashflows = schedule.dual(start_index[index])

        cash_pmts, cash_index, cash_counts = np.unique(
            cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)

        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        if 'SettleFX' in factor_dep:
            discountfx = utils.calc_fx_forward(
                factor_dep['SettleFX'], factor_dep['Currency'],
                cash_pmts, discount_block.time_grid, shared)
        else:
            discountfx = shared.one

        if settlement_amt:
            settlement_days = (factor_dep['Settlement_Date'] - time_block).clip(min=0).reshape(-1, 1)
            repo_discount = utils.calc_discount_rate(repo_block, settlement_days, shared)
            settlement_discount = torch.squeeze(
                utils.calc_discount_rate(settle_block, settlement_days, shared), dim=1)
            # a forward carries the post-settlement bond value to the forward date on repo, then
            # discounts that settlement value on the bond curve
            discount_rates = discount_rates / repo_discount
        else:
            settlement_discount = shared.one

        all_int = (1.0 if ignore_fixed_rate else cashflows.tn[:, utils.CASHFLOW_INDEX_FixedRate]
                   ) * cashflows.tn[:, utils.CASHFLOW_INDEX_Year_Frac]

        # cached on the schedule's tensor half, so it lives exactly as long as that half - not in
        # `t_Buffer`, which clears per batch. Keyed by the slice it sums, so a deal priced on two
        # grids cannot read the other's
        payment_key = ('Payments', start_index[index], ignore_fixed_rate)

        if schedule.derived.get(payment_key) is None:
            payments = 0.0

            if cash_counts.min() != cash_counts.max():
                interest = F.pad(all_int, [0, 1])
                nominal = F.pad(cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal], [0, 1])
                fixed_amt = F.pad(cashflows.tn[:, utils.CASHFLOW_INDEX_FixedAmt], [0, 1])
            else:
                interest = all_int
                nominal = cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal]
                fixed_amt = cashflows.tn[:, utils.CASHFLOW_INDEX_FixedAmt]

            default_offst = np.ones(cash_index.size, dtype=np.int32) * (len(interest) - 1)

            for i in range(cash_counts.max()):
                offst = default_offst.copy()
                offst[cash_counts > i] = i + cash_index[cash_counts > i]
                int_i = interest[offst]

                if factor_dep.get('Compounding', False):
                    payments += (payments + nominal[offst]) * int_i + fixed_amt[offst]
                else:
                    payments += int_i * nominal[offst] + fixed_amt[offst]

            schedule.derived[payment_key] = payments

        mtm_list.append(
            (torch.sum(
                discountfx * discount_rates * schedule.derived[payment_key].reshape(1, -1, 1), dim=1) -
             settlement_amt) * settlement_discount)

        if settle_cash:
            if factor_dep.get('Settlement_Date') is not None:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, factor_dep['Settlement_Date']),
                            mtm_list[-1][-1])
            else:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]),
                            schedule.derived[payment_key][0])

    return torch.cat(mtm_list, dim=0)


def pv_index_cashflows(shared, time_grid, deal_data, settle_cash=True):
    """The inflation-linked leg: each cashflow's fixed interest grown by the ratio of its final to
    its base index reference, discounted at the pay date. A `Settlement_Date` makes it a forward,
    carried on repo and discounted back on the settlement curve."""
    def calc_index(schedule, sim_schedule):
        """The index level at each reset, projected off the last published print."""
        weight = schedule.tn[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
        dates = schedule.np[np.newaxis, :, utils.RESET_INDEX_Reset_Day] - \
                last_pub_block[:, np.newaxis, utils.RESET_INDEX_Reset_Day]
        index_t = torch.unsqueeze(last_index_block, dim=1) / utils.calc_discount_rate(
            forecast_block, dates, shared)

        # rows straddling the last publication mix published prints with projections
        if dates[dates < 0].any():
            future_indices = (dates >= 0).all(axis=1).argmin()
            future_index_t, past_index_t = torch.split(
                index_t, (future_indices, dates.shape[0] - future_indices))

            mixed_indices_t = []
            for mixed_dates, mixed_indices in zip(dates[future_indices:], past_index_t):
                future_resets = mixed_dates.size - (mixed_dates[::-1] > 0).argmin()
                past_resets_t, future_resets_t = torch.split(
                    mixed_indices, (future_resets, mixed_dates.size - future_resets))
                mixed_indices_t.append(
                    torch.cat([sim_schedule[:future_resets], future_resets_t], dim=0))

            values = weight * torch.cat([future_index_t, torch.stack(mixed_indices_t)], dim=0)
        else:
            values = weight * index_t

        if resets_per_cf > 1:
            return torch.sum(values.reshape(
                last_pub_block.shape[0], -1, resets_per_cf, shared.simulation_batch), dim=2)
        else:
            return values

    def get_index_val(cash_index_vals, schedule, sim_schedule, resets_per_cf, offset):
        """A cashflow's index reference: its declared value if it has one, else the projection."""
        if (cash_index_vals.np < 0).any():
            num_known = cash_index_vals.np[cash_index_vals.np > 0].size
            reset_offset = resets_per_cf * (offset + num_known)
            if num_known:
                known_indices = cash_index_vals.tn[cash_index_vals.np < 0].reshape(
                    1, -1, 1).expand(last_pub_block.shape[0], -1, shared.simulation_batch)
                return torch.cat([known_indices, calc_index(
                    schedule[reset_offset:], sim_schedule[reset_offset:])], dim=1)
            else:
                return calc_index(schedule[reset_offset:], sim_schedule[reset_offset:])
        else:
            return cash_index_vals.tn.reshape(1, -1, 1)

    def filter_resets(resets, index):
        """Every reset the outer grid has reached: the declared prints, then the simulated ones."""
        known_resets = resets.known_resets(shared.simulation_batch)
        sim_resets = resets.schedule[(resets.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
                                     (resets.schedule[:, utils.RESET_INDEX_Reset_Day] <=
                                      deal_time[:, utils.TIME_GRID_MTM].max())]
        old_resets = utils.calc_time_grid_spot_rate(index, sim_resets[:, :utils.RESET_INDEX_Scenario + 1], shared)
        return torch.cat([torch.cat(known_resets, dim=0), old_resets], dim=0) if known_resets else old_resets

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    resets_per_cf = factor_dep['Resets_Per_Cashflow']

    last_published = factor_dep['Cashflows'].Resets.schedule[deal_data.Time_dep.deal_time_grid]
    last_published_index = utils.calc_time_grid_spot_rate(factor_dep['PriceIndex'], deal_time, shared)
    index_forecast = utils.calc_time_grid_curve_rate(factor_dep['ForecastIndex'], deal_time, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    repo = utils.calc_time_grid_curve_rate(factor_dep['Repo_Rate'], deal_time, shared)
    settlement = utils.calc_time_grid_curve_rate(factor_dep['Settlement_Rate'], deal_time, shared)

    base_resets = factor_dep['Base_Resets']
    final_resets = factor_dep['Final_Resets']

    all_base_resets = filter_resets(base_resets, factor_dep['PriceIndex'])
    all_final_resets = filter_resets(final_resets, factor_dep['PriceIndex'])

    start_index, start_counts = np.unique(cash_start_idx, return_counts=True)
    last_pub_blocks = np.split(last_published, start_counts.cumsum())

    for index, (forecast_block, discount_block, repo_block, settle_block, last_index_block) in enumerate(
            utils.split_counts([index_forecast, discounts, repo, settlement, last_published_index], start_counts, shared)):

        last_pub_block = last_pub_blocks[index]
        # a dual (numpy and tensor) representation of the same cashflows
        cashflows = factor_dep['Cashflows'].dual(start_index[index])

        cash_pmts, cash_index, cash_counts = np.unique(
            cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)

        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        all_base_index_vals = get_index_val(
            cashflows[:, utils.CASHFLOW_INDEX_BaseReference], base_resets.dual(),
            all_base_resets, resets_per_cf, start_index[index])
        all_final_index_vals = get_index_val(
            cashflows[:, utils.CASHFLOW_INDEX_FinalReference], final_resets.dual(),
            all_final_resets, resets_per_cf, start_index[index])

        interest = (cashflows.tn[:, utils.CASHFLOW_INDEX_FixedRate] *
                    cashflows.tn[:, utils.CASHFLOW_INDEX_Year_Frac]).reshape(1, -1, 1)
        growth = cashflows.tn[:, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1) * (
                all_final_index_vals / all_base_index_vals)
        payment_all = cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1) * growth * interest

        # rows sharing a payment date sum
        if (cash_counts == 1).all():
            payments = payment_all
        else:
            payments = torch.stack([payment.sum(dim=1) for payment in torch.split(
                payment_all, tuple(cash_counts), dim=1)], dim=1)

        asset_pv = torch.sum(payments * discount_rates, dim=1)

        if factor_dep['Settlement_Date'] is not None:
            settlement_days = (factor_dep['Settlement_Date'] - time_block).clip(min=0).reshape(-1, 1)
            repo_discount = torch.squeeze(utils.calc_discount_rate(repo_block, settlement_days, shared), dim=1)
            settlement_discount = torch.squeeze(
                utils.calc_discount_rate(settle_block, settlement_days, shared), dim=1)
            # a forward carries the post-settlement indexed bond value to the forward date on repo,
            # then discounts that settlement value on the bond curve
            mtm = asset_pv * settlement_discount / repo_discount
        else:
            mtm = asset_pv

        mtm_list.append(mtm)

        if settle_cash:
            if factor_dep['Settlement_Date'] is not None:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, factor_dep['Settlement_Date']),
                            mtm_list[-1][-1])
            else:
                cash_settle(shared, factor_dep['SettleCurrency'],
                            np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]),
                            payments[-1][0])

    return torch.cat(mtm_list, dim=0)


def pv_energy_cashflows(shared, time_grid, deal_data):
    """The energy leg: each cashflow's weighted average of forward prices, converted at the fx
    forwards of its own reset dates, scaled by the nominal and discounted at the pay date."""
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    resets = factor_dep['Cashflows'].Resets
    known_resets = resets.known_resets(shared.simulation_batch)
    sim_resets = resets.schedule[
        (resets.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (resets.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
    all_fx_spot = utils.calc_fx_cross(
        factor_dep['ForwardFX'][0], factor_dep['CashFX'][0], sim_resets, shared)

    if factor_dep['ForwardPrice'] is None:
        # the forward curve reconstructed from the simulated components,
        # spot x exp((carry + repo) tau); past resets sample F(reset_day, fixing_day) off the same
        # curve, where F(t,t) is the spot
        forwards = utils.DerivedForwardCurve(
            utils.calc_time_grid_spot_rate(factor_dep['Commodity'], deal_time, shared),
            utils.calc_time_grid_curve_rate(factor_dep['Carry_Rate'], deal_time, shared),
            utils.calc_time_grid_curve_rate(factor_dep['Repo_Rate'], deal_time, shared),
            factor_dep['ForwardStart'][deal_data.Time_dep.deal_time_grid], deal_time)
        all_resets = utils.DerivedForwardCurve(
            utils.calc_time_grid_spot_rate(factor_dep['Commodity'], sim_resets, shared),
            utils.calc_time_grid_curve_rate(factor_dep['Carry_Rate'], sim_resets, shared),
            utils.calc_time_grid_curve_rate(factor_dep['Repo_Rate'], sim_resets, shared),
            sim_resets[:, utils.RESET_INDEX_Start_Day], sim_resets)
    else:
        forwards = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], deal_time, shared)
        all_resets = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], sim_resets, shared)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

    reset_samples = all_resets.gather_weighted_curve(
        shared, sim_resets[:, utils.RESET_INDEX_End_Day].reshape(-1, 1), multiply_by_time=False)
    reset_values = torch.unsqueeze(
        torch.squeeze(reset_samples, dim=1) * all_fx_spot, dim=1) \
        if sim_resets.any() else shared.fillvalue
    old_resets = torch.squeeze(
        torch.cat([torch.stack(known_resets), reset_values], dim=0) if known_resets else reset_values, dim=1)

    start_index, start_counts = np.unique(cash_start_idx, return_counts=True)

    for index, (forward_block, discount_block) in enumerate(utils.split_counts(
            [forwards, discounts], start_counts, shared)):

        cashflows = factor_dep['Cashflows'].dual(start_index[index])

        cash_pmts, cash_index, cash_counts = np.unique(
            cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

        reset_offset = int(cashflows.np[0, utils.CASHFLOW_INDEX_ResetOffset])

        time_ofs = 0
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        reset_block = resets.dual(reset_offset)
        reset_ofs, reset_count = np.unique(
            resets.split_block_resets(reset_offset, time_block), return_counts=True)

        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)
        # the fx adjustment where the payoff currency differs from the deal currency
        discountfx = utils.calc_fx_forward(
            factor_dep['CashFX'], factor_dep['Currency'],
            cash_pmts, discount_block.time_grid, shared)

        # the forward block splits again where rows within it own different reset counts
        forward_blocks = forward_block.split_counts(
            reset_count, shared) if len(reset_count) > 1 else [forward_block]

        payments = []

        for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
            past_resets = torch.unsqueeze(
                old_resets[reset_offset:reset_offset + offset], dim=0).expand(size, -1, -1)

            future_ends = np.tile(reset_block.np[offset:, utils.RESET_INDEX_End_Day], [size, 1])

            if future_ends.any():
                future_resets = forward_rates.gather_weighted_curve(
                    shared, future_ends, multiply_by_time=False)

                forwardfx = utils.calc_fx_forward(
                    factor_dep['ForwardFX'], factor_dep['CashFX'],
                    reset_block.np[offset:, utils.RESET_INDEX_Reset_Day],
                    discounts.time_grid[time_ofs:time_ofs + size], shared)

                all_resets = torch.cat([past_resets, future_resets * forwardfx], dim=1)
            else:
                all_resets = past_resets

            time_ofs += size

            all_payoffs = all_resets * reset_block.tn[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
            split_payoffs = tuple(cashflows.np[:, utils.CASHFLOW_INDEX_NumResets].astype(np.int32))
            payoff = torch.stack([torch.sum(x, dim=1) for x in torch.split(
                all_payoffs, split_payoffs, dim=1)], dim=1)

            payment = cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal] * (
                    cashflows.tn[:, utils.CASHFLOW_INDEX_Start_Mult] * payoff +
                    cashflows.tn[:, utils.CASHFLOW_INDEX_FloatMargin])

            payments.append(payment)

        all_payments = torch.cat(payments, dim=0)

        cash_settle(shared, factor_dep['SettleCurrency'],
                    np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), all_payments[-1][0])
        mtm_list.append(torch.sum(discountfx * all_payments * discount_rates, dim=1))

    return torch.cat(mtm_list, dim=0)


@utils.log_exception
def pv_average_price_swap(shared, time_grid, deal_data):
    """A commodity average-price swap: ``(A - K) * Buy_Sell * Units`` paid at ``Settlement_Date``,
    where ``A = sum_j w_j S(T_j)`` samples a possibly COMPOSED spot (primary + ``ObservedBasis``).

    Closed form conditional on the path - no inner MC, because nothing here decides on simulated
    state. Per reporting row the average splits at that row, and the two halves read from genuinely
    different places:

    REALISED, ``T_j <= t``: the composed spot gathered at the FIXING's own scenario row (a reset
    row's first three columns are a time-grid triple, so ``calc_time_grid_spot_rate`` reads the path
    where the fixing happened); a fixing before the base date is its declared ``Price``. Reading the
    current row's spot instead would be stale by the whole drift between the two dates.

    PROJECTED, ``T_j > t``: ``F_primary(t, T_j) + E_t[b(T_j)]``. The forward is the cost-of-carry
    read on the PRIMARY spot alone - the basis is not a traded carry and does not belong under the
    exponential; it projects under ITS OWN simulated law. With the slow mean on, that law is the
    coupled pair ``b_t = mu_{t-1} + phi (b_{t-1} - mu_{t-1})``,
    ``mu_t = lam mu_{t-1} + (1-lam) b_t``, whose mean matrix is row-stochastic (eigenvalues 1 and
    ``lam*phi``), so ``E_t[b(T_j)] = mu_t + (pi_b + (1-pi_b) (lam phi)**n_j) (b_t - mu_t)`` with
    ``pi_b = (1-lam) phi / ((1-lam) phi + 1 - phi)``.

    A deviation does NOT die: the pair has a unit root and keeps ``pi_b`` of it forever, so a naive
    ``phi**n`` decay misprices every unrealised fixing by ``pi_b (b_t - mu_t)``. ``Phi``/``lam`` are
    the simulated model's declared coefficients and ``mu_t`` its published slow mean, read per path
    off ``(key, 'basis_mu')`` in the fork-index convention (``state[t]`` is what the step t->t+1
    consumes). With the slow-mean extension off there is no published series and no unit root, and
    the plain ``phi**n_j`` is exact.

    ``n_j`` counts SIMULATION STEPS between the two rows, fractionally, because a simulation step is
    what the basis model applies ``Phi`` to; decaying in calendar days or calibration steps projects
    a different model from the one being simulated, and the two coincide only on a calendar-daily
    grid.

    Both halves carry the same normalised weights, so ``K`` comes off the average once. The
    settlement books once, at the deal grid's last row.
    """
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    samples = factor_dep['Samples']
    dual = samples.dual()

    # realised fixings: the declared past values, then the composed spot at each fixing's OWN row
    known = samples.known_resets(shared.simulation_batch)
    sim_samples = samples.schedule[
        (samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
    simulated = utils.calc_time_grid_spot_rate(
        factor_dep['Commodity'], sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
    realised = torch.cat([torch.cat(known, dim=0), simulated], dim=0) if known else simulated

    # the split is a MASK over (row, fixing) rather than a block loop - the payoff has no branch
    # for a block boundary to serve, and every projected term is well defined at a realised fixing
    t_mtm = deal_time[:, utils.TIME_GRID_MTM].reshape(-1, 1)
    # make_sampling_data writes reset == start == end, so one column is both "has it fixed" and
    # "what date is sampled"
    fixing_day = dual.np[:, utils.RESET_INDEX_End_Day].reshape(1, -1)
    past = shared.one.new_tensor(1.0 * (fixing_day < t_mtm)).unsqueeze(-1)
    weight = dual.tn[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
    tau = np.clip(fixing_day - t_mtm, 0.0, None)

    spot = utils.calc_time_grid_spot_rate(factor_dep['Spot'], deal_time, shared)
    carry = utils.calc_time_grid_curve_rate(factor_dep['Carry'], deal_time, shared)
    repo = utils.calc_time_grid_curve_rate(factor_dep['Repo'], deal_time, shared)
    projected = spot.unsqueeze(1) * torch.exp(
        carry.gather_weighted_curve(
            shared, np.tile(factor_dep['base_index'] + fixing_day, (t_mtm.size, 1)),
            multiply_by_time=False) * spot.new_tensor(tau / utils.DAYS_IN_YEAR).unsqueeze(-1) +
        repo.gather_weighted_curve(shared, tau))

    if factor_dep['Basis']:
        # steps, not days: the fractional scenario position of the fixing less that of the row
        steps = np.clip(
            (dual.np[:, utils.RESET_INDEX_Scenario] +
             dual.np[:, utils.RESET_INDEX_Time_Grid]).reshape(1, -1) -
            (deal_time[:, utils.TIME_GRID_ScenarioPriorIndex] +
             deal_time[:, utils.TIME_GRID_PriorScenarioDelta]).reshape(-1, 1), 0.0, None)
        basis = utils.calc_time_grid_spot_rate(factor_dep['Basis'], deal_time, shared).unsqueeze(1)
        phi = factor_dep['Basis_Phi']
        if factor_dep['Basis_Mu']:
            mean = utils.calc_time_grid_spot_rate(
                factor_dep['Basis_Mu'], deal_time, shared).unsqueeze(1)
            # the simulated (b, mu) pair is row-stochastic (eigenvalues 1 and lam*phi), so a
            # deviation keeps the unit root's share pi_b of itself forever rather than dying
            lam = factor_dep['Basis_Lambda']
            pi_b = (1.0 - lam) * phi / ((1.0 - lam) * phi + 1.0 - phi)
            decay = pi_b + (1.0 - pi_b) * (lam * phi) ** steps
        else:
            mean, decay = 0.0, phi ** steps
        projected = projected + mean + spot.new_tensor(decay).unsqueeze(-1) * (basis - mean)

    n = realised.shape[0]
    average = (torch.sum(projected * weight * (1.0 - past), dim=1) +
               torch.sum(realised.unsqueeze(0) * weight[:, :n] * past[:, :n], dim=1))

    nominal = factor_dep['Buy_Sell'] * deal_data.Instrument.field['Units']
    cash = nominal * (average - factor_dep['Strike'])
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    discount_rates = torch.squeeze(utils.calc_discount_rate(
        discount, (factor_dep['Settlement'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1),
        shared), dim=1)

    cash_settle(shared, factor_dep['SettleCurrency'],
                deal_data.Time_dep.deal_time_grid[-1], cash[-1])

    return cash * discount_rates * utils.calc_fx_cross(
        factor_dep['Currency'], shared.Report_Currency, deal_time, shared)


def pv_credit_cashflows(shared, time_grid, deal_data, return_par_spread=False):
    """A single-name CDS: the premium leg survival-weighted (with an accrued-on-default adjustment
    where `Accrue_Fee` is set) against the protection leg on each period's marginal default
    probability. `return_par_spread` answers the breakeven spread instead of the value."""
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    surv = utils.calc_time_grid_curve_rate(factor_dep['Name'], deal_time, shared)
    start_index, counts = np.unique(cash_start_idx, return_counts=True)

    for index, (discount_block, surv_block) in enumerate(
            utils.split_counts([discounts, surv], counts, shared)):
        # a dual (numpy and tensor) representation of the same cashflows
        cashflows = factor_dep['Cashflows'].dual(start_index[index])
        cash_pmts, cash_index = np.unique(cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True)
        cash_sts = np.unique(cashflows.np[:, utils.CASHFLOW_INDEX_Start_Day])

        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        start_pmts = (cash_sts.reshape(1, -1) - time_block.reshape(-1, 1))

        Dt_T = utils.calc_discount_rate(discount_block, future_pmts, shared)
        Dt_Tm1 = utils.calc_discount_rate(discount_block, start_pmts.clip(min=0), shared)

        survival_T = utils.calc_discount_rate(surv_block, future_pmts, shared, multiply_by_time=False)
        survival_t = utils.calc_discount_rate(surv_block, start_pmts.clip(min=0), shared, multiply_by_time=False)

        interest = (cashflows.tn[cash_index, utils.CASHFLOW_INDEX_FixedRate] *
                    cashflows.tn[cash_index, utils.CASHFLOW_INDEX_Year_Frac])

        marginal_PD = survival_t - survival_T
        if factor_dep['Accrue_Fee']:
            past_accrual = cashflows.tn.new(
                daycount_fn(-start_pmts.clip(max=0))) / cashflows.tn[cash_index, utils.CASHFLOW_INDEX_Year_Frac]
            adjustment = (past_accrual + 0.5 * (1-past_accrual)).unsqueeze(dim=2)
        else:
            adjustment = 0.0

        # negative by convention: the protection buyer PAYS the premium
        premium = -(interest[cash_index] * cashflows.tn[cash_index, utils.CASHFLOW_INDEX_Nominal]).reshape(1, -1, 1)
        pv_premium = premium * (survival_T + adjustment * marginal_PD) * Dt_T
        pv_credit = (1.0 - factor_dep['Recovery_Rate'].recovery_rate()) * cashflows.tn[
            cash_index, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1) * 0.5 * (
                            Dt_T + Dt_Tm1) * marginal_PD

        if return_par_spread:
            value = pv_credit.sum(1)/-pv_premium.sum(1)
        else:
            cash_settle(shared, factor_dep['SettleCurrency'],
                        np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), premium[0, 0, 0])
            value = torch.sum(pv_credit + pv_premium, dim=1)

        mtm_list.append(value)

    return torch.cat(mtm_list, dim=0)


def expected_rate_gaussian_copula(
    shared,
    p_default,   # (T, Pm, S, N) marginal cumulative default probs by horizon
    c: float,
    n: int,
    k0: int,
    rho: float,
    z: torch.Tensor,   # (G,)
    w: torch.Tensor,   # (G,)
):
    """The STEP-DOWN expected rate E[c (1 - (k0+k)/n) 1_{k0+k<n}] at each horizon, under a
    one-factor Gaussian copula on the marginal cumulative default probabilities `p_default`.

    `k` is the number of the N names defaulting by the horizon and `k0` (`Defaults_So_Far`) the
    count already lost before the valuation date, so the rate steps down from c in 1/n increments
    and is zero once k0+k reaches `n` (`Max_Defaults`). n = 1 with k0 = 0 collapses to c P(no
    default), the only case in which this equals c P(k < n).

    Conditional on the common factor z each name defaults independently with probability
    q_j = Phi((Phi^-1(p_j) - sqrt(rho) z)/sqrt(1-rho)), so P(k = m | z) = P0 e_m(r) with
    P0 = prod(1-q_j) and e_m the m-th elementary symmetric polynomial of the odds
    r_j = q_j/(1-q_j). Only e_0..e_{n-k0-1} can carry a non-zero rate, so the recurrence stops
    there: it is O(N (n-k0)) and, unlike the closed form e_2 = (S_1^2 - S_2)/2, free of
    cancellation in float32.

    The z integral is Gauss-Hermite on the caller's (`z`, `w`) nodes, EXACT in z at rho = 0 and
    progressively harder as rho approaches 1 (`Quadrature_Points`).

    Returns (T, Pm, S): E[rate(horizon)] at each hazard sample point.
    """
    eps = torch.finfo(shared.one.dtype).eps

    Tsteps, Pm, Sscen, Nnames = p_default.shape

    # clamped off 0 and 1, where the inverse CDF diverges
    p = p_default.clamp(eps, 1.0 - eps)

    a = utils.norm_icdf(p)                                    # (T,Pm,S,N)

    sr = np.sqrt(rho)
    s1 = np.sqrt(1.0 - rho)

    # broadcast over the GH nodes: (G, T, Pm, S, N)
    z_ = z.view(-1, 1, 1, 1, 1)
    a_ = a.unsqueeze(0)

    q = utils.norm_cdf((a_ - sr * z_) / s1).clamp(eps, 1.0 - eps)  # (G,T,Pm,S,N)

    # P0 = prod(1-q), in logs: (G,T,Pm,S)
    logP0 = torch.log1p(-q).sum(dim=-1)
    P0 = torch.exp(logP0)

    r = q / (1.0 - q)     # (G,T,Pm,S,N)

    # elementary symmetric polynomials e_0..e_Kmax of the odds, by stable recurrence over names
    Kmax = max(n - k0 - 1, 0)
    e = shared.one.new_zeros((q.shape[0], Tsteps, Pm, Sscen, Kmax + 1))
    e[..., 0] = 1.0
    for i in range(Nnames):
        ri = r[..., i]  # (G,T,Pm,S)
        for k in range(min(i + 1, Kmax), 0, -1):
            e[..., k] = e[..., k] + ri * e[..., k - 1]

    ks = torch.arange(0, Kmax + 1, device=shared.one.device, dtype=shared.one.dtype)
    rate_k = (c * (1.0 - (k0 + ks) / float(n))).clamp_min(0.0)
    rate_z = (P0.unsqueeze(-1) * e * rate_k.view(1, 1, 1, 1, -1)).sum(dim=-1)  # (G,T,Pm,S)

    # integrate over z with the GH weights
    w_ = w.view(-1, 1, 1, 1)  # (G,1,1,1)
    E_rate = (w_ * rate_z).sum(dim=0)  # (T,Pm,S)

    return E_rate


def pv_credit_step_down_cashflows(shared, time_grid, deal_data):
    """The premium leg of an n-th-to-default basket: per payment period the expected step-down
    rate is integrated over the accrual window by trapezoid on hazard samples 30 days apart,
    weighted by that period's nominal and discounted at the PAY date.

    TWO DAY COUNTS, doing two different jobs. The ACCRUAL measure dt is the deal's own
    `Accrual_Day_Count` (`factor_dep['Accrual_Daycount']`), which is what the coupon is quoted
    against; DISCOUNTING and the hazard curve lookups keep each curve's own day count, which is a
    property of the curve and not of the contract. They coincide only when the deal is written on
    its discount curve's convention.

    The accrual window opens at the first unpaid period's own `Start_Day`, so a forward-starting
    deal accrues from its effective date and a seasoned one from the start of the period it is in.
    `get_cashflows` builds every period as (start_i, pay_i) = (reset_i, reset_{i+1}), so periods
    are contiguous and that one start plus the pay days ARE all the boundaries. Samples before the
    valuation date give a negative hazard time, clipped to zero survival weight: the already-accrued
    part of the running coupon is earned at the full rate.
    """
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    accrual_daycount_fn = factor_dep['Accrual_Daycount']
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
    rho = factor_dep['Correlation']
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    # the index proxy the per-name hazards are scaled against
    surv = utils.calc_time_grid_curve_rate(factor_dep['Name'], deal_time, shared)
    surv_base = utils.calc_time_grid_curve_rate(factor_dep['Name'], np.zeros((1, 3)), shared)
    names = [utils.calc_time_grid_curve_rate(name, deal_time, shared) for name in factor_dep['Names']]
    # Gauss-Hermite nodes for integral of exp(-x^2) f(x) dx, transformed to the standard normal
    x, w = np.polynomial.hermite.hermgauss(factor_dep['Quadrature_Points'])
    z = np.sqrt(2.0) * x
    ww = w / np.sqrt(np.pi)
    z_t = shared.one.new(z)
    w_t = shared.one.new(ww)
    start_index, counts = np.unique(cash_start_idx, return_counts=True)

    for index, (discount_block, surv_block) in enumerate(
            utils.split_counts([discounts, surv], counts, shared)):
        # a dual (numpy and tensor) representation of the same cashflows
        cashflows = factor_dep['Cashflows'].dual(start_index[index])
        cash_pmts, cash_index = np.unique(cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True)
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        # the default-sampling window, which opens at the first unpaid period's start
        samples_points = np.r_[cashflows.np[0, utils.CASHFLOW_INDEX_Start_Day], cash_pmts].astype(np.int64)
        # hazard_t can be negative, which is the accrued part of the running coupon
        hazard_t = (np.unique(
            np.r_[samples_points, np.concatenate(
                [np.arange(a + 30, b, 30) for a, b in zip(
                    samples_points[:-1], samples_points[1:])])]).reshape(1, -1) -
                          time_block.reshape(-1, 1))
        hazard_samples = hazard_t.clip(min=0)
        Dt_T = utils.calc_discount_rate(discount_block, future_pmts, shared)

        index_cum_hazard_T = surv_block.gather_weighted_curve(shared, hazard_samples, multiply_by_time=False)
        base_index = surv_base.gather_weighted_curve(shared, hazard_samples[0].reshape(1,-1),  multiply_by_time=False)
        g = index_cum_hazard_T/base_index
        fwd_hazard_names = torch.stack(
            [g * x.gather_weighted_curve(shared, hazard_samples, multiply_by_time=False) for x in names], dim=3)

        S_nodes = torch.exp(-fwd_hazard_names)  # (T, M+1, S, N)
        Cum_PD = (1.0 - S_nodes)
        # in the deal's own ACCRUAL day count, and left signed so accrued coupons carry
        dt = shared.one.new(np.diff(accrual_daycount_fn(hazard_t), axis=1)).unsqueeze(2)
        E_rate = expected_rate_gaussian_copula(
            shared=shared,
            p_default=Cum_PD,
            c=factor_dep['Coupon'],
            n=factor_dep['Max_Defaults'],
            k0=factor_dep['Defaults_So_Far'],
            rho=rho,
            z=z_t,
            w=w_t
        )  # (Sc,P)

        # integrate E[rate] over each accrual window by the trapezoid rule
        trap = 0.5 * (E_rate[:, :-1, :] + E_rate[:, 1:, :])
        trapz = trap * dt # (T,Pm-1,S)
        lengths = torch.as_tensor(
            [np.r_[0,hs.searchsorted(x, side='right')-1] for hs,x in zip(hazard_samples, future_pmts)],
            device=shared.one.device)
        expected_coupons = torch.segment_reduce(trapz, reduce="sum", lengths=lengths.diff(), axis=1)
        # per-period nominal carries Principal, Buy_Sell and any Amortisation; the minus is
        # pv_credit_cashflows' premium convention - the buyer PAYS the coupon
        premium = -expected_coupons * cashflows.tn[cash_index, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1)

        cash_settle(shared, factor_dep['SettleCurrency'],
                    np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), premium[-1, 0])

        mtm_list.append(torch.sum(premium * Dt_T, dim=1))

    return torch.cat(mtm_list, dim=0)


def pv_equity_cashflows(shared, time_grid, deal_data):
    """The equity leg: each period pays its start/end multiples of the equity level plus the
    realized dividends between them. A period is valued off its observed samples once both ends
    have fixed, off forwards while both are ahead, and off the mix in between."""
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    forward_settle_days = factor_dep['Bus_Ofs'][deal_data.Time_dep.deal_time_grid]
    eq_spot = utils.calc_time_grid_spot_rate(factor_dep['Equity'], deal_time, shared)
    cash = factor_dep['Flows']

    # the three indices a block is grouped on: start fixed, end fixed, and paid
    cash_start_idx = np.searchsorted(
        cash.schedule[:, utils.CASHFLOW_INDEX_Start_Day], deal_time[:, utils.TIME_GRID_MTM], side='right')
    cash_end_idx = np.searchsorted(
        cash.schedule[:, utils.CASHFLOW_INDEX_End_Day], deal_time[:, utils.TIME_GRID_MTM], side='right')
    cash_pay_idx = cash.get_cashflow_start_index(deal_time)

    all_samples = []

    for samples in cash.Resets.split_groups(2):
        known_sample = samples.known_resets(shared.simulation_batch, include_today=True)
        sim_samples = samples.schedule[
            (samples.schedule[:, utils.RESET_INDEX_Value] == 0.0) &
            (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]

        past_samples = utils.calc_time_grid_spot_rate(
            factor_dep['Equity'], sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)

        # an unsimulated spot is held flat across the scenarios
        if past_samples.shape[1] != shared.simulation_batch:
            past_samples = past_samples.expand(sim_samples.shape[0], shared.simulation_batch)

        all_samples.append(torch.cat(
            [torch.cat(known_sample, dim=0), past_samples], dim=0) if known_sample else past_samples)

    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    repo_discounts = utils.calc_time_grid_curve_rate(factor_dep['Equity_Zero'], deal_time, shared)
    eq_div_curve = utils.calc_time_grid_curve_rate(factor_dep['Dividend_Yield'], deal_time, shared)

    cashflows = cash.dual()

    all_index, all_counts = np.unique(list(
        zip(cash_start_idx, cash_end_idx, cash_pay_idx)), axis=0, return_counts=True)
    time_block_index = 0

    for index, (discount_block, repo_block, divi_block, eq_block) in enumerate(
            utils.split_counts([discounts, repo_discounts, eq_div_curve, eq_spot], all_counts, shared)):

        start_idx, end_idx, pay_idx = all_index[index]

        cashflow_start = cashflows.np[start_idx:, utils.CASHFLOW_INDEX_Start_Day].reshape(1, -1)
        cashflow_pay = cashflows.np[pay_idx:, utils.CASHFLOW_INDEX_Pay_Day].reshape(1, -1)

        payoffs = []
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cashflow_pay - time_block.reshape(-1, 1)
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        # both ends fixed: the period is valued off its own observed samples
        if pay_idx < end_idx:
            St0 = torch.unsqueeze(all_samples[0][pay_idx:end_idx], 0)
            St1 = torch.unsqueeze(all_samples[1][pay_idx:end_idx], 0)

            Ht0_t1 = utils.calc_realized_dividends(
                St0, factor_dep['Equity_Zero'], factor_dep['Dividend_Yield'],
                utils.calc_dividend_samples(
                    cashflows.np[pay_idx:end_idx, utils.CASHFLOW_INDEX_Start_Day],
                    cashflows.np[end_idx - 1:end_idx, utils.CASHFLOW_INDEX_End_Day], time_grid), shared)

            units = cashflows.tn[pay_idx:end_idx, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1)
            end_mult = cashflows.tn[pay_idx:end_idx, utils.CASHFLOW_INDEX_End_Mult].reshape(1, -1, 1)
            div_mult = cashflows.tn[pay_idx:end_idx, utils.CASHFLOW_INDEX_Dividend_Mult].reshape(1, -1, 1)
            start_mult = cashflows.tn[pay_idx:end_idx, utils.CASHFLOW_INDEX_Start_Mult].reshape(1, -1, 1)
            payoff = (end_mult * St1 - start_mult * St0 + div_mult * Ht0_t1)

            payment = payoff * units

            if factor_dep['PrincipleNotShares']:
                payment /= St0

            payoffs.append(payment.expand(time_block.size, -1, -1) if time_block.size > 1 else payment)

            cash_settle(shared, factor_dep['SettleCurrency'],
                        np.searchsorted(time_grid.mtm_time_grid, cashflow_pay[0][0]),
                        torch.sum(payment, dim=1)[0])

        # the start fixed and the end still ahead: the observed sample against a forward
        if end_idx < start_idx:
            cf_end = cashflows.np[end_idx, utils.CASHFLOW_INDEX_End_Adj] - time_block.reshape(-1, 1)
            cf_settle = forward_settle_days[time_block_index:time_block_index+all_counts[index]].reshape(-1,1)
            repo_carry = repo_block.gather_weighted_curve(shared, cf_end, cf_settle)
            divi_carry = divi_block.gather_weighted_curve(shared, cf_end, cf_settle)
            forward_end = eq_block.unsqueeze(1) * torch.exp(repo_carry - divi_carry)

            St0 = torch.unsqueeze(all_samples[0][end_idx:start_idx], dim=0)
            Ht0_t = utils.calc_realized_dividends(
                St0, factor_dep['Equity_Zero'], factor_dep['Dividend_Yield'],
                utils.calc_dividend_samples(
                    cashflows.np[end_idx:start_idx, utils.CASHFLOW_INDEX_Start_Day], time_block, time_grid), shared)

            units = cashflows.tn[end_idx:start_idx, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1)
            end_mult = cashflows.tn[end_idx:start_idx, utils.CASHFLOW_INDEX_End_Mult].reshape(1, -1, 1)
            div_mult = cashflows.tn[end_idx:start_idx, utils.CASHFLOW_INDEX_Dividend_Mult].reshape(1, -1, 1)
            start_mult = cashflows.tn[end_idx:start_idx, utils.CASHFLOW_INDEX_Start_Mult].reshape(1, -1, 1)

            payoff = (end_mult - div_mult) * forward_end + div_mult * (
                    torch.unsqueeze(eq_block, dim=1) + Ht0_t) * torch.exp(repo_carry) - start_mult * St0

            if factor_dep['PrincipleNotShares']:
                payoff /= St0

            payoffs.append(payoff * units)

        # both ends still ahead: forwards at each
        if cashflow_start.any():
            cf_start = cashflows.np[start_idx:, utils.CASHFLOW_INDEX_Start_Adj] - time_block.reshape(-1, 1)
            cf_end  = cashflows.np[start_idx:, utils.CASHFLOW_INDEX_End_Adj] - time_block.reshape(-1, 1)
            cf_settle = forward_settle_days[time_block_index:time_block_index+all_counts[index]].reshape(-1,1)
            repo_start = repo_block.gather_weighted_curve(shared, cf_start, cf_settle)
            repo_end = repo_block.gather_weighted_curve(shared, cf_end, cf_settle)
            divi_start = divi_block.gather_weighted_curve(shared, cf_start, cf_settle)
            divi_end = divi_block.gather_weighted_curve(shared, cf_end, cf_settle)

            forward_start = eq_block.unsqueeze(1) * torch.exp(repo_start - divi_start)
            forward_end = eq_block.unsqueeze(1) * torch.exp(repo_end - divi_end)

            if factor_dep['PrincipleNotShares']:
                factor1 = forward_end / forward_start
                factor2 = 1.0
            else:
                factor1 = forward_end
                factor2 = forward_start

            units = cashflows.tn[start_idx:, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1)
            end_mult = cashflows.tn[start_idx:, utils.CASHFLOW_INDEX_End_Mult].reshape(1, -1, 1)
            div_mult = cashflows.tn[start_idx:, utils.CASHFLOW_INDEX_Dividend_Mult].reshape(1, -1, 1)
            start_mult = cashflows.tn[start_idx:, utils.CASHFLOW_INDEX_Start_Mult].reshape(1, -1, 1)

            payoff = (end_mult - div_mult) * factor1 + (
                    div_mult * torch.exp(repo_end - repo_start) - start_mult) * factor2

            payoffs.append(payoff * units)

        time_block_index += all_counts[index]
        payments = torch.cat(payoffs, dim=1) if len(payoffs) > 1 else payoffs[0]
        mtm_list.append(torch.sum(payments * discount_rates, dim=1))

    return torch.cat(mtm_list, dim=0)


@utils.log_exception
def pv_fixed_leg(shared, time_grid, deal_data):
    """The fixed leg in the reporting currency."""
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
    mtm = pv_fixed_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_energy_leg(shared, time_grid, deal_data):
    """The energy leg in the reporting currency."""
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
    mtm = pv_energy_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_float_leg(shared, time_grid, deal_data):
    """The floating leg in the reporting currency, under the cashflow pricer the deal declares."""
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
    model = deal_data.Factor_dep.get('Model', pricer_float_cashflows)
    mtm = pv_float_cashflow_list(shared, time_grid, deal_data, model) * FX_rep

    return mtm


@utils.log_exception
def pv_index_leg(shared, time_grid, deal_data):
    """The inflation-linked leg in the reporting currency."""
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
    mtm = pv_index_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_cds_leg(shared, time_grid, deal_data):
    """The single-name CDS in the reporting currency."""
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
    mtm = pv_credit_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_credit_step_down_leg(shared, time_grid, deal_data):
    """The basket premium leg in the reporting currency."""
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
    mtm = pv_credit_step_down_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_equity_leg(shared, time_grid, deal_data):
    """The equity leg in the reporting currency."""
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
    mtm = pv_equity_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm
