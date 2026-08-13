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

The weights sum to one by construction, so their L1 norm is EXACTLY the factor by which the fit
can amplify the jumps it is averaging: 1.0 when no weight is negative, unbounded as the kernel's
first two moments collapse onto each other. It is scale-free in the gap, in the jump and in the
sample size, which a threshold on the solve's determinant is not - by Cauchy-Schwarz that
determinant over ``s0 * s2`` lies in [0, 1] whatever the units, and the pathology and a legitimate
solve straddle any level fitted to one of them.

Measured over 2685 solves in 82 runs - every boundary registration in the repo (discrete barrier,
collateralised latch, FVA, multi-batch, swaption exercise, autocall, TARF, MTA), 512 to 4096
paths, five seeds, bandwidths 0.005 to 0.2. Of the 2424 that carry any density at all, every one
reads 1.00 to 8.06 but the single decision that broke the Heston-Nandi barrier gradient, which
reads 99.9: a kernel holding exactly two points, 1.20 widths out and 0.021 apart, weighted +50.4
and -49.5, contributing 112% of a coefficient it had no business dominating. 25 sits 3.1x above
the largest legitimate reading and 4.0x below that one. The other six solves above 8 are collateral
transfer decisions whose kernel mass is 1e-9 or less, which the estimator was already returning
essentially nothing for."""


# ======================================================================================
# Heston-Nandi GARCH(1,1) OSS pricers (TARF, discrete barrier, autocall). Opt-in per deal via
# the Valuation Configuration switch SpotModel='HestonNandi' (see the deal calc_dependencies
# branches and pv_MC_Tarf for the OSS scheme + known limitations F1-F4). The per-step advance
# (utils.hn_daily_advance / utils.hn_unmonitored_substeps) is owned by utils; both the
# unmonitored sub-steps and the survival-truncated final step of every
# fixing/observation interval, in ALL THREE pricers, route through it.
# ======================================================================================

def cash_settle(shared, currency, time_index, value):
    # need to check if the time_index
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
        # need to divide the strike by the ATM_Ref (can only be done later)
        # so we just return the strike here (we know the moneyness rule and will do the rest later)
        # otherwise, (Sticky_Moneyness) - we return log of strike over forward
        return strike if subtype[1]=='Sticky_Strike' else torch.log(strike/forward)
    elif subtype[0] == 'Relative_Forward':
        return (strike - forward)/forward
    elif subtype[0] == 'Malz':
        forward_or_spot = forward if use_forward else spot
        return torch.log(strike / forward_or_spot if invert_moneyness else forward_or_spot / strike)
    else:
        # regular 2d vol surface assumed to be sticky_moneyness - need to handle other moneyness rules (TODO!)
        forward_or_spot = forward if use_forward else spot
        return strike / forward_or_spot if invert_moneyness else forward_or_spot / strike


def forward_carry_rate(carry_rate, cum_t, dt):
    """The annualised carry over EACH fixing INTERVAL, from zero rates read at the fixing tenors.

    ``carry_rate[j]`` is what ``calc_eq_drift``/``calc_fx_drift`` return with
    ``multiply_by_time=False``: the curve gathered AT tenor ``T_j``, which is an AVERAGE over
    ``[t, T_j]`` measured from the valuation row. The carry over the interval ENDING at ``T_j`` is
    therefore a DIFFERENCE OF CUMULATIVE INTEGRALS, ``(c_j*T_j - c_{j-1}*T_{j-1}) / dt_j``, and not
    ``c_j`` - the two agree only for the FIRST interval, where the cumulative window IS the
    interval, and on a FLAT curve. Both degeneracies hold in every barrier fixture in this repo,
    which is why ``pv_discrete_barrier_option`` multiplied one window's average by another window's
    length for as long as it existed while ``pv_MC_Tarf`` and ``pv_MC_AutoCallSwap`` differenced.
    Measured on a 2%->5% curve with quarterly fixings, the strip's total ran -1.12% on the forward
    factor; on this repo's own sloped fixture the forward to expiry ran -4.28%.

    All FOUR adopters now read the strip from here - ``sim_spot_oss``, ``pv_MC_Tarf``, and both
    branches of ``pv_MC_AutoCallSwap``, the averaging one having its own ``carry * dt`` that the
    defect report did not name - so the drift of their simulations and the forward their
    closed-form legs are valued at cannot be spelled differently again. Summing the strip
    telescopes to ``c_N*T_N`` - see ``total_log_forward``, which is that sum.

    Rank-polymorphic on the same rule as ``total_log_forward``: ``[N_fix, batch]`` for one MTM row
    and ``[N_block, N_fix, batch]`` for a whole block, with ``cum_t`` and ``dt`` carrying no batch
    axis. A ZERO-LENGTH interval is a fixing the row has already observed and every caller skips
    it, so it divides by one rather than by zero - which keeps the unread entries out of the
    backward pass as well as the forward one.

    A FLAT curve does NOT come back bit-identical, and no gate on this may be written with
    ``torch.equal``: differencing amplifies the rounding of a cumulative time by ``T_j/dt_j``,
    measured at 6 eps on the engine's monthly ACT/365 strip against that bound of 12.
    """
    cum = carry_rate * cum_t.unsqueeze(-1)
    step = dt[..., 1:].unsqueeze(-1)
    return torch.cat([carry_rate[..., :1, :], cum.diff(dim=-2) /
                      torch.where(step > 0, step, torch.ones_like(step))], dim=-2)


def total_log_forward(carry_rate, times):
    """``log F(t,T) / S(t)``: the carry integrated over a fixing strip.

    THE forward to expiry, for every leg that needs one. ``carry_rate`` is the annualised carry per
    fixing INTERVAL - what ``forward_carry_rate`` builds, NOT the raw zero rates the curve is
    gathered at - and ``times`` the year-fraction of each interval, so the integral is their
    product summed over the fixing axis, the axis ``times`` names by carrying no batch dimension.
    That sum telescopes to ``c_N*T_N``, one gather at expiry. Rank-polymorphic on purpose:
    ``[N_fix, batch]`` for one MTM row and ``[N_block, N_fix, batch]`` for a whole block are the
    same expression, so the leg that values one row and the leg that values the block cannot spell
    it differently.

    That is the point of the function existing at all. Two legs of ``pv_discrete_barrier_option``
    value the SAME European on the SAME state - a knocked-in barrier - and the already-hit one
    carried a second spelling that summed annualised RATES with no ``dt`` and added a half-variance
    whose cancelling subtraction lives on the other branch. It read 106.7% high in log-forward on
    this repo's own fixture and no gate saw it: base valuation never evaluates the leg, the one
    exposure-grid barrier is the model-free zeros branch, and ``r = q = 0`` everywhere zeroes the
    missing ``dt``. A shared expression makes the divergence unrepresentable rather than merely
    detected, which is why this is a function and not a comment.
    """
    return (carry_rate * times.unsqueeze(-1)).sum(dim=-2)


def forward_vol_strip(deal_data, strike, spot, carry_rate, cum_t, shared,
                      invert_moneyness=False, use_forwards=False):
    """The implied vol read at EVERY fixing's own tenor, at the moneyness the DEAL declares.

    An implied surface is quoted against an option's EXPIRY, so a simulation that steps a fixing
    strip has to read it once per fixing - at ``T_j``, and at the moneyness an option expiring
    ``T_j`` would be quoted at. ``carry_rate`` is therefore the ZERO carry
    (the raw ``calc_eq_drift``/``calc_fx_drift`` gather at each tenor, NOT the interval strip
    ``forward_carry_rate`` builds), because a forward wants the cumulative integral: ``F_j =
    S * exp(c_j * T_j)``. ``cum_t`` is that cumulative tenor and arrives as NUMPY - it is the
    surface's expiry KEY as well as the exponent's year fraction, and the key has to be hashable.

    Rank-polymorphic on the same rule as ``forward_carry_rate``: ``[N_fix, batch]`` for one MTM row
    and ``[N_block, N_fix, batch]`` for a whole block, ``cum_t`` carrying no batch axis, and the
    fixing axis returned at ``-2`` either way so the strip drops straight into ``forward_vol_rate``.

    ``use_forwards`` IS THE DEAL'S, not this function's. It was hard-coded ``True`` here, mirroring
    ``pv_MC_Tarf`` - the sibling that already read the surface per fixing and has no European limit
    to violate. Its two other adopters DO have one, and they declare ``use_forwards = False``, so a
    forward read made the simulation step a different law from the quote the same pricer marks its
    own European legs with. Measured on a smiley surface (``vol = 0.2479 + 0.35*(m-1)^2``, ten
    seeds, 65536 inner paths): a never-knocking ``Down_And_Out`` read 1175.00 against Black at the
    declared quote 1163.96, ``+0.948%`` and 8.3 standard errors, and in-out parity read ``-11.03``
    where it must read zero. INTERNAL CONSISTENCY WINS: the strip reads what the deal declares, and
    per-fixing SMILE - a real modelling question, since a desk quoting sticky-forward moneyness
    wants the other convention - is open in ``roadmap.md`` with those numbers on it. The TERM
    STRUCTURE half is untouched by the choice: alternating only this flag on the smile-free sloped
    surface leaves the prices agreeing to 2.274e-13.

    The moneyness goes through ``calc_moneyness``, the one place that knows what a surface's
    subtype wants (SVI/Skew take a log, Malz its own ratio) - ``pv_MC_Tarf`` had its own
    ``forward / strike`` inline, which is that function's ``else`` branch and silently the wrong
    query on any surface that is not a plain 2d grid. The SPOT is broadcast onto the fixing axis
    first: a declared spot read is one moneyness for the whole strip, and without the expand
    ``calc_moneyness`` returns a tensor one rank short and the per-fixing loop indexes off the end.
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

    An implied vol is CUMULATIVE by definition - ``sigma(T)^2 * T`` is the total variance to ``T``
    - so the variance of the interval ending at ``T_j`` is a DIFFERENCE of cumulative variances,
    ``(sigma_j^2 T_j - sigma_j-1^2 T_j-1) / dt_j``, and not ``sigma_j^2``. The two agree only for
    the FIRST interval, where the cumulative window IS the interval, and on a FLAT surface. This is
    ``forward_carry_rate``'s statement one factor over, and it has the same failure mode: the
    wrong allocation TELESCOPES to the right total variance, so the terminal distribution, every
    European limit, in-out parity and every CRN gradient gate stay exactly right and only the
    path-dependent MONITORING is biased.

    Measured on two surfaces carrying the same 1y implied vol (flat 0.2479 against a term
    structure running 0.10 to 0.32 at 2y): the true interval strip runs 0.111 -> 0.336 against the
    single 0.2479 that ``pv_discrete_barrier_option`` and ``pv_MC_AutoCallSwap`` applied to every
    interval, and the two surfaces priced ``Down_And_Out``/``Down_And_In``/``Up_And_Out`` BITWISE
    IDENTICALLY. Against a fine-step oracle under the surface's own instantaneous vol the sloped
    world read -1.46%, +11.53% and -11.07%.

    ``clamp(min=eps)`` handles a DECLINING cumulative variance - an arbitrageable surface, or the
    same interpolation kink the GBM term-structure quote work documents - by flooring the interval
    at one eps rather than taking the square root of a negative. IT DOES NOT TELESCOPE, and that is
    a disclosure and not a bug: a floored interval contributes ~0 where the surface says it should
    contribute a NEGATIVE variance, so the strip's total EXCEEDS the surface's own total and the
    simulation no longer reproduces the European quote the same pricer marks with. Measured on a
    monthly barrier fixture whose surface runs 0.30 at 6m into 0.20 at 9m - three intervals of
    declining cumulative variance - in-out parity against the simulated vanilla reads -101.08 +/-
    1.37 on ten seeds, -7.99% of that vanilla, where an arbitrage-free surface reads -0.19. The
    analytic half is unaffected (KO + KI is still Black at the declared quote to 2.3e-12), which is
    exactly why this shows up as a parity break rather than as a wrong-looking price. It is inherent
    to flooring:
    ``pv_MC_Tarf`` has carried the same clamp since it was written. The fix is an arbitrage-free
    surface, not a different floor. ``j == 0`` takes ``vols[0]`` directly, that window being the
    interval itself. Both are ``pv_MC_Tarf``'s semantics, term for term, because they were its
    expression before they were this one.

    Rank-polymorphic exactly as ``forward_carry_rate``, and a ZERO-LENGTH interval divides by one
    rather than by zero for the same reason: it is a fixing the row has already observed and every
    caller skips it. A FLAT surface does NOT come back bit-identical and no gate on this may use
    ``torch.equal`` - differencing amplifies the rounding of a cumulative time by ``T_j/dt_j``.
    """
    cum_var = vols * vols * cum_t.unsqueeze(-1)
    step = dt[..., 1:].unsqueeze(-1)
    fwd = torch.sqrt(cum_var.diff(dim=-2).clamp(min=torch.finfo(vols.dtype).eps) /
                     torch.where(step > 0, step, torch.ones_like(step)))
    return torch.cat([vols[..., :1, :], fwd], dim=-2)


def boundary_weights(gap, bandwidth):
    """Density at the boundary and local-linear regression weights, estimated SEPARATELY.

    The term to recover is ``f_G(0) * E[jump * dG/dtheta | G = 0]``. Folding both halves into one
    kernel - weighting each path by a smoothed Dirac and averaging - is a local-CONSTANT estimator,
    whose bias is O(bandwidth) with an f'/f term rather than O(bandwidth^2). Measured on this book
    that bias never settles: the correction tracked the bandwidth from -70k to -235k with no
    plateau, and stayed put across a 16x change in path count, so it was estimator bias and not
    Monte Carlo noise.

    Local-linear weights cancel that first-order term, so the estimate should hold still over a
    range of bandwidths - which is the only acceptance criterion worth having here, since no
    single bandwidth can be argued for on its own.

    Returns ``(density_at_zero, weights)`` with the weights summing to one, so the caller
    multiplies rather than averages.

    THAT SUM IS ALSO THE ACCEPTANCE TEST. Cancelling the first-order term is what makes the
    weights signed, and a kernel that admits two neighbouring points a long way out cancels it by
    subtracting two enormous numbers - measured, +50.4 and -49.5 on two points 0.02 widths apart,
    whose differing jumps then did not cancel and supplied 112% of that gradient. Since the
    weights sum to one, ``||weights||_1`` is exactly that amplification, so the solve is refused
    on what it PRODUCED rather than on its determinant: the determinant over ``s0 * s2`` is
    Cauchy-Schwarz-bounded into [0, 1] and the pathology (8.6e-05) sits beside a legitimate solve
    (6.2e-03) close enough that no level separates them. See ``BOUNDARY_MAX_AMPLIFICATION``.
    Refusing costs one reduction and lands on the branch a degenerate gap already takes - weights
    zero, correction exactly zero, which is also what an empty kernel returns.

    ONE sample has no spread for a kernel to be scaled by, and ``torch.std`` returns NaN there
    rather than raising - which propagates into the scalar handed to ``backward()`` and is invisible
    while the degenerate gap happens to carry no graph (``0 * NaN`` is NaN forward but reaches
    nothing). Zero width is the right degenerate answer: the kernel admits nothing, the local-linear
    solve is refused, and the correction is exactly zero - which is what one scenario means here.
    That path is reached by base valuation, whose whole simulation is the pricer's own inner Monte
    Carlo.
    """
    g = gap.detach()
    # one sample -> zero width; std() would be NaN and the correction is exactly zero anyway
    spread = g.std() if g.numel() > 1 else torch.zeros_like(g.reshape(-1)[:1].squeeze())
    width = bandwidth * spread.clamp_min(torch.finfo(g.dtype).eps)
    k = torch.exp(-0.5 * (g / width) ** 2)
    s0, s1, s2 = k.sum(), (k * g).sum(), (k * g * g).sum()
    # nothing at all near the boundary makes this exactly 0; float64 cannot land it any nearer
    denominator = s2 * s0 - s1 * s1
    solvable = denominator != 0
    weights = torch.where(solvable, k * (s2 - g * s1) / torch.where(
        solvable, denominator, torch.ones_like(denominator)), torch.zeros_like(k))
    # the solve is refused on what it PRODUCED, which is the only thing that separates the two
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
    number and still propagates the missing term through ``gap`` to every factor at once - no
    bump, no second valuation, cost independent of how many factors there are.

    ``objective_jump`` is a COEFFICIENT and stays detached: its own pathwise derivatives are
    already in the ordinary AAD value, and differentiating the counterfactual would count them
    twice.

    Summed, not averaged: the local-linear weights already carry the 1/N through the density.
    """
    density, weights = boundary_weights(gap, bandwidth)
    return ((gap - gap.detach()) * (density * weights * objective_jump).detach()).sum()


def boundary_correction(shared, objective, reported_mtm, bandwidth):
    """Total boundary correction for every recorded decision - a margin call's transfer, a barrier
    crossing, an autocall trigger, any observed event whose value jump is real.

    They are one defect: a decision taken on simulated state whose derivative the frozen-decision
    graph drops. They differ only in how the counterfactual is produced - a replayed balance scan,
    branches the pricer already evaluated - and that half belongs to the set that recorded it,
    along with the netting arithmetic that carries its decision out to the portfolio.

    This half is `score`: the reported PORTFOLIO plus a change to it. Never a set's own level,
    because the objective is applied to `resolve_structure`'s root sum over every netting set, and
    a collateralised set's post-collateral net sits at the relu kink by construction - which is
    where scoring it in isolation goes furthest wrong. `gap > 0` means the trigger fired, matching
    a jump of J(fired) - J(did not).
    """
    def score(delta):
        return objective(reported_mtm + delta)

    corrections = [stochastic_boundary_correction(gap, jump, bandwidth)
                   for bset in shared.boundary_sets
                   for gap, jump in bset.objective_jumps(score)]
    return torch.stack(corrections).sum() if corrections else None


class InnerMCRecompute(torch.autograd.Function):
    """A pricer's inner Monte Carlo as one node: simulated untaped, RE-simulated to differentiate.

    An MC-priced deal builds a graph per pricing that the terminal backward holds until it runs -
    every fixing of every reporting row of every deal, all resident at once. The simulation itself
    is cheap and the tape is what does not fit, so this trades the tape for a second forward: the
    node's own forward runs under `no_grad` and leaves NOTHING behind but its inputs, and its
    backward re-runs the same simulation under `enable_grad` and contracts the cotangent through
    that one graph, which dies as soon as it has. Peak is one inner graph rather than all of them.

    THE COUNTER IS THE STORAGE. A recompute is only a recompute if the second pass draws the same
    numbers, and the numbers are exactly what is too big to keep - so what is saved is where each
    stream STOOD (`utils.rng_position`), not what it produced. Sobol draws are memoized
    per (dimension, sample_size, batch), so rewinding the counter makes the replay return the very
    same tensor; the regular generator has no memo, so its own state is saved and restored. The
    live position is put back afterwards as STATED contract rather than observed necessity: every
    current pricer reseeks absolutely per batch, so deleting the restore is unobservable today
    (measured by mutation) - it protects the caller that does not reseek, which nothing yet is.

    ONE FUNCTION, CALLED TWICE. `simulate` is the same object in both passes - not a differentiable
    copy of a fast forward - because two spellings of one simulation agree until the day one of
    them is edited, and the failure is a wrong gradient beside a right price, which no price gate
    sees. For the same reason it must be PURE in everything but its return: a side effect (a
    cashflow accrued, a decision registered) would fire a second time in the backward, so a pricer
    RETURNS those instead and the caller performs them once, off the forward's result.

    ITS INPUTS ARE ITS WHOLE THETA SURFACE. Autograd can only return a gradient for a tensor that
    was passed to `apply`, so anything `simulate` reads out of a closure is differentiated as a
    constant - silently. A pricer therefore hoists every graph-carrying quantity into `theta`,
    including ones it would otherwise compute inside the loop (a vol strip off the surface, say).

    A DECISION'S GAP IS AN OUTPUT WHEN THE SIMULATION IS WHAT DECIDED IT, which is what makes it
    survive the untaped forward. Boundary corrections need `gap` to carry a graph
    (`stochastic_boundary_correction` is `gap - gap.detach()` times a detached coefficient) and the
    forward pass here has none to give: the gap's VALUE is computed under `no_grad` for the
    registration to report, and the node is what connects it. The coefficient is assembled at the
    objective, exactly as before, and arrives as that output's cotangent - so the graph-carrying
    half of the correction is built during the backward, inside the recompute, and the split costs
    the estimator nothing. The contract is unchanged: what reaches `backward()` differs, what is
    reported does not.

    The converse is the rule, and it is about WHERE THE GRAPH LIVES rather than about symmetry: a
    decision taken on OUTER state - a scenario spot at a barrier observation, an accrual series the
    block loop built - keeps its own graph whatever this node does to the inner pass, so its gap
    stays where it was. `pv_discrete_barrier_option` registers its whole latch outside; `pv_MC_Tarf`
    registers the redemption latch outside and the knock-in, decided on an inner draw, as outputs.

    THE ADOPTER'S SHAPE (`run` below is the one line; the rest is the callable's own contract):

    - `simulate(*bound)(*theta)` - the leading arguments are the block's SHAPE, bound per block with
      `partial`, and are numpy, ints and flags; the trailing ones are every tensor it reads that can
      carry a graph, which is exactly what the node can return a gradient for.
    - It returns a TUPLE whose element 0 is the block's marks and whose remaining elements are the
      by-products the caller performs once - settled cashflows, decision gaps, counterfactual
      branches - each accompanied by the plain-Python row index that places it. Non-tensors pass
      straight through and take a `None` cotangent, which is why the pairing below tests the
      cotangent first.
    - Every graph-carrying by-product is a TOP-LEVEL element of that tuple. A tensor nested inside a
      returned list is NOT an output: measured, it comes back with `requires_grad False` and no
      `grad_fn`, so its half of a correction would be silently zero.
    - A settled cashflow is one of those outputs, not a side effect. A replay would settle it twice,
      and WHICH harvest sees that depends on when it reads the buffer: the per-netting-set
      `save_cashflows` snapshot is taken in the forward pass and survives the defect, while
      `Credit_Monte_Carlo` builds its REPORTED frame from `t_Cashflows` after the batch's backward
      (calculation.py:1727) and does not - measured on the TARF fixture, a replay that books moves
      that frame by 6.0e+06 while the cva, the profile and the whole CVA gradient stay bit-identical
      (`tests/test_recompute_inner_mc.py`, and it is only reachable with sensitivities on, since
      with them off no backward runs at all). What ALSO breaks is the graph - booked inside, the
      cashflow is booked under `no_grad`, and a collateralised exposure reading `t_Cashflows`
      through `C_ts_te` loses that whole channel (measured, 8.7% of the autocall's CVA gradient).

    A registration held on `shared` makes shared -> boundary_sets -> gap -> this node -> `simulate`
    -> shared, a cycle refcounting cannot break; `reset()` clearing `boundary_sets` per batch drops
    one edge and frees it deterministically. With sensitivities off nothing outside the graph holds
    the node at all.

    SECOND DERIVATIVES ARE REFUSED, on the `LeastSquaresSolve` precedent and for a sharper reason.
    The replay is rooted at DETACHED copies of the saved inputs - that is what stops `autograd.grad`
    walking back into the outer graph and double-counting the first derivative - and a second
    derivative taken through a detached leaf is severed from the graph the outer pass holds. It does
    not raise on its own: it comes back with the entries that needed that path set to ZERO, which is
    a Hessian that looks like a Hessian. Measured on the TARF fixture, `Greeks: 'All'`: three
    leading entries of -1.74e6 / -4.03e5 / -1.67e4 taped, all three zero recomputed, and a fourth
    that merely disagreed. Keeping the inputs attached instead was tried and breaks the FIRST
    derivative outright, so the honest node is a first-order one that says so.
    """

    @staticmethod
    def forward(ctx, simulate, position, shared, *theta):
        # undefined cotangents stay undefined: most outputs (detached jumps, counterfactual
        # branches) are coefficients nothing differentiates, and materializing zeros for them
        # would allocate in backward exactly what this node exists to avoid
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
        """The node, or not - the ONE line an adopting pricer writes, and the whole switch.

        `Recompute_Inner_MC` is a property of the valuation engine and the machine it is on, so it
        governs every adopter at once and no pricer carries a flag of its own. Off, `simulate` is
        called and taped exactly as it always was; on, it goes through the node with the RNG
        position taken at the call site - which is where the streams still stand where the replay
        must find them.

        `cls.apply` so a subclass runs itself. The mutation gates rebind the module global, which
        either spelling follows (measured) - the spelling states the intent, it is not what the
        gates enforce.
        """
        return (cls.apply(simulate, utils.rng_position(shared), shared, *theta)
                if shared.recompute_inner_mc else simulate(*theta))


def cva_per_scenario(pv_exposure, prob, recovery):
    """CVA as a PER-SCENARIO vector, whose mean is the reported CVA.

    Pulled out of Credit_Monte_Carlo.execute so a counterfactual netting-set MTM can be scored on
    the same objective without re-deriving it. The reduction order is load-bearing: the reported
    number is a MEAN over paths of a SUM over time, so any boundary correction assembled against
    it must also be a mean over paths or it is silently scaled by the path count - silently,
    because such a correction has zero forward value and only the gradients would be wrong.

    The `<xva>_per_scenario` family grows one member per adjustment that wants a boundary
    counterfactual; at the third member it becomes per-adjustment objects living here.
    """
    return ((1.0 - recovery) * 0.5 * (pv_exposure[1:] + pv_exposure[:-1]) * prob).sum(axis=0)


def fva_per_scenario(pv, cost_spread, benefit_spread):
    """FVA as a PER-SCENARIO vector, whose mean is the reported FCA_t - FBA_t.

    `pv` is SIGNED - the funding cost rides its positive part and the benefit its negative one -
    where `cva_per_scenario`'s exposure arrives already relu'd; the two must not share a
    parameter name with opposite preconditions. Same estimator, same load-bearing reduction
    order: a per-path vector of a sum over time, so a boundary correction assembled against it
    is a mean over paths rather than silently scaled by the path count.
    """
    plus, minus = torch.relu(pv), torch.relu(-pv)
    return (torch.sum(cost_spread * (plus[1:] + plus[:-1]) / 2, dim=0)
            - torch.sum(benefit_spread * (minus[1:] + minus[:-1]) / 2, dim=0))


class SensitivitiesEstimator(object):
    """ Implements the AAD sensitivities (both first and second derivatives)"""

    def __init__(self, value, params, create_graph=False):
        """
        Args:
            value: function output (tensor)
            params: List of model parameters (list of tensor(s))
        """
        # run the backward pass
        value.backward(retain_graph=True, create_graph=create_graph)
        # store associated gradients
        self.params = {key: tensor for key, tensor in params if tensor.grad is not None}
        self.grad = {key: tensor.grad for key, tensor in self.params.items()}
        self.flat_grad = self.flatten(list(self.grad.values()))
        self.device = self.flat_grad.device
        self.dtype = self.flat_grad.dtype
        self.list_param = list(self.params.values())
        self.P = len(self.flat_grad)

    def report_grad(self):
        """Per-factor gradients as numpy arrays, COPIED off the live ``.grad`` buffers.

        ``np.array`` copies; ``.numpy()`` on a CPU tensor would return a VIEW of the ``.grad``
        buffer torch keeps accumulating into, so a second ``backward()`` through the same leaves
        silently rewrites a report already handed out. On CUDA ``.cpu()`` copies and hides that; on
        the default device it does not, and the two disagree.
        """
        return {utils.check_scope_name(factor): np.array(tensor.cpu().detach())
                for factor, tensor in self.grad.items()}

    def report_hessian(self, allow_unused=False):
        """The full (P, P) Hessian as a numpy array, in the same flat factor order as
        ``report_grad`` concatenates - which is what lets the report label both of its axes off
        the first-order index.

        Assembled UPPER-TRIANGLE-ONLY and mirrored: ``get_H_op`` differentiates factor i's
        gradient against factors i onwards, so the blocks below the diagonal are never computed,
        and the strict lower triangle is wiped before the mirror so the diagonal block's own
        sub-diagonal is not counted twice. Symmetry is therefore exact by construction rather
        than approximately - a gate reading ``H - H.T`` measures the assembly, not the AAD.

        ``allow_unused`` is what a portfolio needs and a single objective does not: a factor the
        value depends on LINEARLY has no second-order path at all, and autograd returns None
        rather than zeros for it.
        """
        h_op = self.get_H_op(allow_unused)
        hessian = np.zeros((self.P, self.P))

        # store it in a matrix
        j = 0
        for row in h_op:
            v, _ = row.shape
            hessian[j:j + v, j:] = row
            j += v

        # zero out the lower indices
        hessian[np.tril_indices(self.P, k=-1)] = 0.0
        return hessian + np.triu(hessian, k=1).T

    def get_Hv_op(self, v):
        """
        Implements a Hessian vector product estimator Hv op defined as the
        matrix multiplication of the Hessian matrix H with the vector v.

        Args:
            v: Vector to multiply with Hessian (tensor)

        Returns:
            Hv_op: Hessian vector product op (tensor)
        """
        hv = self.flatten(torch.autograd.grad(
            self.flat_grad, self.list_param, grad_outputs=v, only_inputs=True, retain_graph=True))

        return hv

    @staticmethod
    def flatten(params):
        """
        Flattens the list of tensor(s) into a 1D tensor
        
        Args:
            params: List of model parameters (List of tensor(s))
        
        Returns:
            A flattened 1D tensor
        """
        return torch.cat([_params.reshape(-1) for _params in params], dim=0)

    def get_H_op(self, allow_unused):
        """ 
        Implements a Hessian estimator op by forming p Hessian vector
        products using HessianEstimator.get_Hv_op(v) for all v's in R^P
        
        Args:
            None
        
        Returns:
            H_op: Hessian matrix op (tensor)
            :param allow_unused:
        """

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
    """`Greeks_First` always, `Greeks_Second` when `Greeks: 'All'` asked for it - the two blocks
    the base valuation reports, in that order because the second-order report labels its axes off
    the first-order index (`Calculation.gradients_as_df`)."""
    greeks_calc = SensitivitiesEstimator(mtm, shared.calc_greeks, create_graph=shared.gamma)
    deal_data.Calc_res['Greeks_First'] = greeks_calc.report_grad()
    # use this only when all the vols and curves are sparsely represented (check greeks_calc.P)
    if shared.gamma:
        deal_data.Calc_res['Greeks_Second'] = greeks_calc.report_hessian(allow_unused=True)
        # `create_graph` leaves every leaf holding a `.grad` whose own graph reaches back to it -
        # a cycle refcounting cannot break, so the whole second-order tape survives the run and
        # waits for a gc pass to free it (on CUDA, holding device memory). Both reports are
        # numpy copies by now, so dropping the buffers costs nothing and is what torch's own
        # create_graph warning prescribes.
        # PARTIAL, and honestly so: this reaches the tensors the estimator was OFFERED. A
        # bootstrapped factor is offered `leaf + (theta - theta.detach())` (`factor_leaf`), a
        # NON-leaf with `retain_grad`, and the true leaf inside it - and every leaf upstream
        # through theta, the quotes included - keeps its own `.grad` and its own cycle until gc
        # gets there. Chasing them would mean walking `grad_fn` from a reported tensor, which is
        # more machinery than the residue is worth: it costs memory only, gc recovers it, and no
        # reported number depends on it either way.
        for tensor in greeks_calc.params.values():
            tensor.grad = None


def interp_to_mtm_grid(mtm, time_grid, deal_data, interpolate_grid=True):
    """Deal grid -> MTM grid: the purely LINEAR half of `interpolate`, with no result stashing.

    A boundary counterfactual has to put its branch values on exactly the grid the reported value
    landed on. Reproducing the gather and the pad at the call site is how they drift apart, and a
    delta misaligned by a row is invisible in a forward pass worth exactly zero - it would only
    show up as a wrong gradient. So both go through this rather than through two copies of it.
    """
    if interpolate_grid and deal_data.Time_dep.interp.size > deal_data.Time_dep.deal_time_grid.size:
        # interpolate it
        mtm = utils.gather_interp_matrix(mtm, deal_data.Time_dep)

    if mtm.shape != (1, 1) and mtm.shape[0] < time_grid.mtm_time_grid.size:
        # pad it with zeros and return
        return F.pad(mtm, [0, 0, 0, time_grid.mtm_time_grid.size - deal_data.Time_dep.interp.size])
    else:
        return mtm


def deal_to_mtm_grid(time_grid, deal_data, fx_rep):
    """The map from a pricer's own output to the MTM-grid rows the deal reports it on, detached.

    Everything `Deal.generate` and `Deal.calculate` do to a theo price and nothing else: scale
    into the reporting currency, then interpolate. A boundary branch is an alternative value for
    the SAME deal, so it has to travel the same two steps - and both were being skipped.

    `fx_rep` is `shared.one` only when the payoff and reporting currencies match; otherwise it is a
    simulated (T, B) cross, and a branch registered without it is a delta in the wrong currency
    whose own flux never reaches the tape. It multiplies on the pricer grid, BEFORE the
    interpolation, because that is the order the reported mtm was built in.

    Two things this closure must NOT hold, both found by measurement rather than by reading.

    `fx_rep` is detached AT CAPTURE, not merely on the way out: a boundary set outlives the pricing
    call, so a closure over a live cross pins that deal's whole autograd graph until the batch's
    backward pass. The rule that jumps stay detached is a memory contract as well as a correctness
    one - the branch values are coefficients.

    And it does not take `shared`. It has no use for it (`interp_to_mtm_grid` never did either),
    and holding it makes a REFERENCE CYCLE: shared -> boundary_sets -> this closure -> shared, which
    refcounting cannot break, so the calculation state and everything reachable from it survives
    the run and waits on the cyclic collector. Measured together: 19.6 GB still resident after one
    collateralised barrier run where the same run held 32 MiB before, and the next run OOMed.
    """
    scale = fx_rep.detach()
    return lambda profile: interp_to_mtm_grid(profile * scale, time_grid, deal_data).detach()


def interpolate(mtm, shared, time_grid, deal_data, interpolate_grid=True):
    if interpolate_grid and deal_data.Time_dep.interp.size > deal_data.Time_dep.deal_time_grid.size:
        # interpolate it
        mtm = utils.gather_interp_matrix(mtm, deal_data.Time_dep)

    # check if we want to store the mtm value for this instrument
    if deal_data.Calc_res is not None:
        shared.save_results(deal_data.Calc_res, {'Value': mtm})
        # add this as a tensor if we need to
        if shared.keep_tensor:
            deal_data.Calc_res['tensor'] = mtm

    # the gather is already done - only the pad is left
    return interp_to_mtm_grid(mtm, time_grid, deal_data, interpolate_grid=False)


def getbarrierpayoff(direction, eta, phi, strike, H):
    '''
    Function to generate the barrier payoff function using these formulae:
    (import sympy with the necessary symbols to see how to derive the code below)
    
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

    This is for single Barrier options and based on Merton, Reiner and Rubinstein.
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
    '''
    Function to generate the partial barrier payoff function
    '''

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
                    forward, strike, log1, log2, rho, -rho, -g1, -e1, -g3, -e3, -g2, -e2, -g4, -e4)
                temp -= PartialBarrierCalc(
                    forward, strike, log1, log2, rho, -rho, -d1, -e1, -f1, -e3, -d2, -e2, -f2, -e4)

                pv = torch.where(strike < barrier, pv, temp)
            else:
                if eta == 1:

                    pv = PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, g1, e1, g3, -e3, g2, e2, g4, -e4)
                    temp = PartialBarrierCalc(
                        forward, strike, log1, log2, rho, -rho, d1, e1, f1, -e3, d2, e2, f2, -e4)

                    pv = torch.where(strike < barrier, pv, temp)

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


def calc_vol_adjustment(factor_dep, deal_time, expiry, vols, shared):
    # None means get the ATM vol for this expiry (can change depending on the vol surface type)
    fx_vols = utils.calc_time_grid_vol_rate(factor_dep['FXVol'], None, expiry, shared)

    # b_adj adjusts the carry on the forward, s_adj is to scale the forward directly (as a factor)
    if 'QuantoImpliedCorrelation' in factor_dep:
        # quanto fx deal
        rho = utils.implied_correlation(
            factor_dep['QuantoImpliedCorrelation'], factor_dep['Correlation_Sign'])
        atm_vol = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], None, expiry, shared)
        return {'vol': vols, 'b_adj': -atm_vol * fx_vols * rho, 's_adj': 1.0}
    else:
        rho = utils.implied_correlation(
            factor_dep['CompoImpliedCorrelation'], factor_dep['Correlation_Sign'])
        tenor_in_days = factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]
        forwardfx = utils.calc_fx_forward(
            factor_dep['Local'], factor_dep['Other'],
            tenor_in_days, deal_time, shared)
        compo_vols = torch.sqrt(
            fx_vols * fx_vols + 2.0 * fx_vols * vols * rho + vols * vols)
        return {'vol': compo_vols, 'b_adj': 0.0, 's_adj':forwardfx}


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
    """
    One-step survival (OSS) variant of pv_discrete_barrier_option.

    At each barrier observation date j, uses the analytic survival probability
    P(S_j on correct side of H | S_{j-1}) to:
      - Contribute an analytic KO-in-step term directly to the PV
      - Draw truncated GBM steps for the surviving paths only
      - Eliminate variance from paths near the barrier

    Advantages over the naive path simulation:
      - No smooth_heaviside approximation needed — the analytic probability is
        exact and differentiable via norm_cdf/norm_icdf
      - Lower variance: barrier-touching paths contribute analytically
      - For BARRIER_IN: in-out parity's vanilla is valued from the block spot forwarded
        by ``total_log_forward``, in the DECLARED model - Black or the HN closed form

    ``Cash_Rebate`` is an ABSOLUTE cash amount, which is how ``pv_barrier_option`` reads it - that
    routine hands the closed form ``cash_rebate/nominal`` and multiplies the result back by nominal.
    Everything ``sim_spot_oss`` returns is scaled by nominal too, so the rebate goes in per-unit,
    divided by the UNSIGNED size rather than by nominal: nominal already carries Buy_Sell (which is
    NOT true in ``pv_barrier_option``, where buy_or_sell is a separate factor), so dividing by it
    would cancel the direction and bring every rebate leg back as +cash_rebate whichever way the deal
    is done - a sold knock-out would book the rebate it must PAY as a receipt.

    Heston-Nandi is opt-in via ``SpotModel='HestonNandi'``: the five GARCH scalars ride the AAD graph
    out of ``t_Static_Buffer`` (identical resolution to the TARF), and when absent ``sim_spot_oss``
    takes the byte-identical GBM path.

    ONE MODEL PRICES EVERY LEG. Both vanillas switch: the KI in-out-parity vanilla inside
    ``sim_spot_oss`` AND the already-hit ``hit_value`` this loop overrides it with. They are the SAME
    state - a knocked-in barrier is a European - so a Black vanilla beside an HN one is a model mix
    inside one payoff, and at 1y ATM on the repo's fixture the two disagree by 15.8%. Worse, the
    ``torch.where(row_barrier_hit, hit_value, oss_result)`` selection puts both models in one tensor,
    element by element. The already-hit leg therefore runs the parity leg's own algebra on the parity
    leg's own discretisation: the same per-row ``n_total`` summed from the same ``sample_ts`` daily
    sub-step counts, and the same scalar per-step carry - which is why it raises the same
    batch-constant-carry refusal, in its own name. ``n_steps`` drives the variance recursion so it is
    scalar per MTM row and the leg is a Python loop over the block's rows; ``rem_exp`` is
    batch-constant (it is the deal clock, shape ``[N_block, 1]``), so nothing is averaged away.

    AND ONE FORWARD PRICES EVERY LEG, for the same reason and by the same means: both vanillas call
    ``total_log_forward``, so a second spelling is unrepresentable rather than merely wrong. There
    was one - the already-hit leg summed annualised RATES with no ``dt`` and added a half-variance
    whose cancelling subtraction lives on the parity branch - and it read 106.7% high in log-forward
    while every gate in the suite stayed green. The parity leg's own spelling was
    ``(r + 0.5 * sigma^2)``, which reconstructs the carry by undoing the ``-0.5 * var`` folded into
    ``drift``; that round trip leaked the ``1e-4`` VARIANCE FLOOR into the DRIFT, adding exactly
    ``5e-5`` to the log-forward on every row whose first fixing interval has zero length - which is
    every row that IS an observation date. Measured: 13 of 37 rows on the repo's monthly fixture,
    worth -0.025% of EPE.

    THAT FLOOR IS NOW THE ZERO-LENGTH INTERVAL'S AND NOTHING ELSE'S. It was unconditional, which was
    harmless only while the defect below handed every interval ``sigma(T)^2`` - the largest vol on
    an upward surface. A correct forward-variance strip hands it genuinely small numbers and it
    binds wherever ``sigma_fwd < 0.01/sqrt(dt)``, 19.1% annualised at daily monitoring: 114 of 365
    daily intervals on an upward 0.12 -> 0.24 surface, against 0 of 53 weekly and 0 of 13 monthly,
    which is every barrier fixture in this repo. Measured there against a floor-free oracle, eight
    seeds: the SHIPPED spelling read +1.584% on a ``Down_And_Out`` and -6.870% on an ``Up_And_Out``,
    the floor alone +0.049% and -5.576%, and this one +0.183% (1.7 se) and +0.403% (0.9 se). The
    weekly and monthly readings are bitwise identical under all three, and the repo's own monthly
    exposure grid is bitwise identical - value, CVA and all 13 CVA-gradient entries - whether the
    floor is conditioned or not, which is why the density gate had to be written rather than
    inherited. ``drift`` and ``vol`` now read ONE ``var``, so a floored
    interval is a martingale under the law it is actually drawn from; the old spelling took the
    drift from the unclamped variance and moves that exposure grid +0.0215% and its CVA +0.0240%
    with no gate anywhere able to see it. WHAT THE ``dt == 0`` CLAMP BUYS is the GRADIENT, measured
    by deleting it: every value stays finite (the profile moves -0.046%) and 11 of the 13
    CVA-gradient entries go NaN, ``sqrt`` having an infinite derivative at zero and the variance
    carrying the surface's graph. The remaining inconsistency is that a zero-length interval is
    still SIMULATED with 1% of lognormal vol rather than resolved by an exact indicator, which
    would consume a Sobol draw and move every barrier constant - open in the roadmap.

    AND THAT FORWARD IS BUILT FROM INTERVAL CARRY, which is the seam's other half and was the
    defect one layer down: the strip handed in is ``forward_carry_rate``'s, not the zero rates the
    curve is gathered at, so ``carry * dt`` is an interval integral rather than one window's
    average times another window's length. It drives ``drift``, so on a sloped curve the OSS
    simulation's own ``E[S_T]`` was not ``F(t,T)`` either - measured -4.28% on the repo's sloped
    fixture, exactly 0 on every flat one, which is why no fixture here had ever seen it.

    AND THE VOL IS TWO QUANTITIES, both legitimate, deliberately not unified into one read. The
    SIMULATION wants the per-INTERVAL strip: the surface read at every fixing's own TENOR
    (``forward_vol_strip``), differenced into forward variance (``forward_vol_rate``), because an
    implied vol is cumulative and the variance of ``[T_j-1, T_j]`` is a difference and not
    ``sigma(T_j)^2 * dt_j``. The EUROPEAN legs - the already-hit KI mark and the in-out-parity
    vanilla - want ``sigma(K, tau) * sqrt(tau)``, the surface's own quote for the option being
    valued, and they share it as ``sd_to_expiry`` for the same reason both read one forward.

    TWO READS OF ONE SURFACE MUST AGREE ON THE MONEYNESS, and this pricer's two did not. The strip
    read every fixing at its own FORWARD moneyness, hard-coded, mirroring ``pv_MC_Tarf`` - which has
    no European limit to violate. This one does, and it declares ``use_forwards`` (False on every
    fixture here), so on a smiley surface the simulation stepped a law the pricer's own European
    legs disagreed with: a never-knocking ``Down_And_Out`` read +0.948% and 8.3 standard errors from
    Black at the declared quote, and in-out parity -11.03 instead of -0.19. The strip now takes the
    deal's flag. What that costs is the per-fixing SMILE, which is a genuine modelling question and
    is open in the roadmap with those numbers; the TERM STRUCTURE half is untouched by the choice
    (1.1e-15 relative, alternating only the flag on a smile-free sloped surface).

    THE DEFECT THIS REPLACED read ONE implied vol per MTM row - at the strike's moneyness and the
    EXPIRY tenor - and applied it to every monitoring interval. The wrong allocation telescopes to
    the right total variance, so the terminal distribution, every European limit, in-out parity and
    every CRN gradient gate were EXACTLY right and only the barrier MONITORING was biased. Two
    surfaces carrying the same 1y implied vol (flat 0.2479 against 0.10 rising to 0.32 at 2y)
    priced ``Down_And_Out``/``Down_And_In``/``Up_And_Out`` BITWISE IDENTICALLY, while the true
    interval strip ran 0.111 -> 0.336. Against a fine-step oracle the sloped world read -1.46%,
    +11.53% and -11.07%.

    AND THE VOL STRIP IS NOT BUILT UNDER HN, as in ``pv_MC_Tarf``. Nothing correct reads the implied
    surface under a non-GBM spot model - the OSS steps the recursion, the already-hit leg is the same
    closed form, and quanto/compo (the third reader, ``Check_Payoff_Type``) is refused at compile
    time - so ``vols`` is an empty tensor there. That is what stops this recurring: the next leg that
    reaches for the surface under HN dies on the shape rather than pricing a second model quietly.

    The per-row hit mask is the state carried IN: every scenario hit at a STRICTLY EARLIER
    observation. The current observation is NOT folded in, because this block's last row IS its
    observation date and ``sim_spot_oss`` prices that date as its own first (zero-length) step, so
    adding it would count the same crossing twice. The barrier is observed only on its own dates: an
    earlier form cumsum-tested EVERY mtm row of the block, so on the repo's own fixture it monitored
    37 rows against 12 barrier dates and knocked scenarios out on dates the deal never observes - and
    it monitored expiry too, which BarrierDates flags -1. It also missed the FIRST barrier date
    entirely, because ``range(prev_sample_idx, sample_index_t)`` is empty for block 0 and lags one
    block thereafter, testing block i's rows because block i-1's observation had fired.

    The rebate falls due AT the knock-out. ``sim_spot_oss`` already puts it in the mtm of that row -
    its zero-length first step gives p=0 for a crossed path, leaving ``(1-p)*L*rebate`` - but nothing
    settled it, and from the next row on ``hit_value`` is zero, so the cash would be priced and then
    paid nowhere; the increment is settled per date, the same way ``pv_barrier_option`` does. The
    TERMINAL row is excluded, since the single settle after the loop already pays it in full as part
    of ``mtm_list[-1][-1]`` - the rebate is in that mtm because ``sim_spot_oss`` accrued it.
    ``pv_barrier_option`` guards the same double count with ``expiry[index] > 0.0``; a deal whose
    last barrier date IS expiry hits it, and instruments.py unions Expiry_Date into the observation
    dates, so that is common.

    Under ``boundary_aad`` the recorded gap is the margin by which each decision was made, graph
    retained - it carries every factor that moved the spot to the barrier - and is signed so
    ``gap > 0`` means CROSSED, matching the jump ``J(crossed) - J(did not)``: a down barrier is
    crossed from ABOVE, so its gap is ``log(barrier/spot)``. The already-hit vanilla is a closed form
    drawing no random numbers, so building it for the counterfactual cannot shift the OSS stream and
    cannot move the reported value. Blocks where EVERY scenario has resolved skip the OSS and so
    carry no counterfactual - running it there would draw random numbers and move the reported value
    - which is also where the kernel weight is negligible, every scenario being far past the barrier
    rather than near it.

    RECOMPUTE (``shared.recompute_inner_mc``, the ``Recompute_Inner_MC`` calculation field). Off,
    ``sim_spot_oss`` is called once and taped. On, it goes through ``InnerMCRecompute`` - untaped to
    price, re-run under ``enable_grad`` to differentiate. This is the cheapest of the three ports
    because ``sim_spot_oss`` was ALREADY pure: it settles no cash (the barrier settles once, after
    the block loop, and its rebate settles off ``crossed``, which the OSS does not produce) and it
    registers nothing. Only the Heston-Nandi scalars had to move out of the closure and into theta.

    ITS GAPS STAY OUTSIDE THE NODE, and not by analogy with the TARF's redemption latch: the
    decision is ``spot_block[-1]`` against the barrier, an OUTER scenario spot at an observation
    date, which the inner simulation neither produces nor is asked about. Its graph is the scenario
    generation's, which this node does not untape, so the registration needs nothing from the
    replay - the whole latch (gaps, both branches, the crossed flags) is built from block tensors
    the loop already has. Off is bit-identical, price and gradient.
    """

    def sim_spot_oss(offset, sobol, num_sims,
                     spot_prices, vols, sd_to_expiry, times, carry, discount_rates, *hn_scalars):
        """Run the OSS inner Monte Carlo and return the per-block mean PV.

        PURE, and split into a bound half and a theta half, because `InnerMCRecompute` calls it
        TWICE - once untaped to price and once taped to differentiate - whenever
        `Recompute_Inner_MC` is on. The leading arguments are the block's shape and are bound per
        block; the trailing ones are every tensor it reads that can carry a graph, which is what the
        node can return a gradient for. It has no by-products, so it returns the one-element tuple
        `(mtm,)` - the tuple is the node's contract, not a shape it happens to have here.

        IT TAKES BOTH VOL QUANTITIES, because there are two and they are not the same read (see the
        pricer docstring): ``vols`` is the per-INTERVAL strip the simulation steps on, shaped
        ``[N_block, N_fix, batch]`` like the carry beside it, and ``sd_to_expiry`` is the EUROPEAN
        standard deviation ``sigma(K, tau) * sqrt(tau)`` the parity vanilla is valued at, shaped
        ``[N_block, batch]`` and shared with the already-hit leg outside.

        BARRIER_IN is priced by in-out parity, so the leg needs a vanilla. Under HN the SMILE BITES:
        that vanilla is the HN CLOSED FORM, not a normal at aggregate variance. It runs ``n_total``
        daily steps to expiry from ``h1 = H0``, with a per-step carry reproducing the forward
        ``S*exp(b*T)`` (undiscounted forward-measure value = ``hn_call * exp(b*T)``), then discounts
        at the real curve ``D[-1]`` - forward and discount separated as in ``black_european_option``.
        Reducing the carry to a scalar ``r_step`` is valid ONLY when carry is batch-constant
        (deterministic rates/dividends): under stochastic-rate CVA the per-scenario carries diverge
        and a scalar leg would misprice the KI vanilla by O(10%) of its value, silently - hence the
        loud guard. The fix is a batched-carry ``hn_call``.

        A digital's TERMINAL step is INTEGRATED, not sampled. An indicator on the drawn spot has
        zero derivative almost everywhere, so the density term that is most of a digital's delta and
        vega never reaches the tape: measured 10.5% low on a knock-out digital's delta and 14.7%
        high on its vega, and once the barrier is out of reach the reported delta and vega are
        EXACTLY zero, with the equity and vol factors absent from the greeks report rather than
        showing zero. The survival legs are already integrated this way - this is the same idiom one
        step later, and it also drops the last step's sampling noise. The HN branch always samples,
        its per-path conditional variance having no scalar closed form to integrate against, so an
        HN digital keeps the indicator.
        """
        eps = torch.finfo(shared.one.dtype).eps
        hn = bool(hn_scalars)
        if hn:
            *hn_params, H0 = hn_scalars

        # `carry` is the INTERVAL carry rate (forward_carry_rate), so this product is the interval
        # integral. GBM folds it with the vol; HN keeps it raw (its recursion supplies variance).
        dt = times.unsqueeze(axis=2)                        # [N_block, N_fix, 1]
        carry_int = carry * dt                              # [N_block, N_fix, batch]
        if not hn:
            # `vols` is the INTERVAL vol strip (forward_vol_rate), so this product is the interval
            # variance - one implied vol applied to every interval is the defect one factor over
            var = vols * vols * dt                          # [N_block, N_fix, batch]
            # the floor is the ZERO-LENGTH interval's survival width and only that (see docstring);
            # a real interval simulates at its own dispersion, and both consumers read the SAME
            # variance so every interval is a martingale under the law it is actually drawn from
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

            if sobol:
                u = shared.quasi_rng(shared.simulation_batch, N_fix * num_sims)[1].T.reshape(
                    N_fix, shared.simulation_batch, -1)
            else:
                u = torch.rand([N_fix, shared.simulation_batch, num_sims],
                               dtype=shared.one.dtype, device=shared.one.device)
            # antithetic variates: [N_fix, batch, 2*num_sims]
            u = torch.concat([u, 1.0 - u], dim=-1)

            D_T = D[-1].reshape(-1, 1)  # terminal discount: [batch, 1]

            if hn:
                # per-interval daily sub-step counts and per-step carry (r-q). n_sub floors at 1.
                nj = [max(int(round(float(t) * hn_spy)), 1) for t in times[blk]]
                b_steps = [(carry_int[blk][j] / nj[j]).reshape(-1, 1) for j in range(N_fix)]
                h = H0

            if direction == BARRIER_IN:
                if not hn:
                    # Precompute analytic vanilla for parity: KI = Vanilla - KO_pure + rebate * E[L_T]
                    # ONE VOL prices every European leg, as ONE FORWARD does: this is the same
                    # `sd_to_expiry` the already-hit leg marks with, not the strip's own total.
                    fwd_to_T = s * torch.exp(total_log_forward(carry[blk], times[blk]))
                    vol_to_T = sd_to_expiry[blk]
                    if isdigital:
                        vanilla_pv = utils.black_european_option(
                            fwd_to_T, strike, vol_to_T, 1.0, 1.0, phi, shared, cash_payoff=1.0) * D[-1]
                    else:
                        vanilla_pv = utils.black_european_option(
                            fwd_to_T, strike, vol_to_T, 1.0, 1.0, phi, shared) * D[-1]
                else:
                    # HN parity vanilla = HN closed form; scalar r_step needs batch-constant
                    # carry, so the guard is loud rather than silent (see docstring)
                    n_total = int(sum(nj))
                    carry_total = total_log_forward(carry[blk], times[blk]).reshape(-1)
                    if float(carry_total.detach().max() - carry_total.detach().min()) > 1.0e-9:
                        raise ValueError(
                            'HN KI closed-form leg needs batch-constant carry; carry varies across '
                            'scenarios by {:.2e} (stochastic rates?) - extend hn_call to batched '
                            'carry or price this deal under GBM'.format(
                                float(carry_total.max() - carry_total.min())))
                    r_step = carry_total[0] / n_total  # scalar b*T/n
                    om, al, be, ga = (v.reshape(-1)[0] for v in hn_params)  # scalar recursion params
                    H0_s = H0.reshape(-1)[0]
                    if isdigital:
                        q_below = utils.hn_cdf_logret(
                            torch.log(strike / s), n_total, H0_s, om, al, be, ga, r_step)
                        vanilla_pv = ((1.0 - q_below) if phi == OPTION_CALL else q_below) * D[-1]
                    else:
                        fwd_growth = torch.exp(r_step * n_total)
                        vanilla = (utils.hn_call if phi == OPTION_CALL else utils.hn_put)(
                            s, strike, n_total, H0_s, om, al, be, ga, r_step)
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
                        # nj-1 unmonitored daily steps + 1 monitored (OSS truncation at the
                        # CONSTANT barrier, F-measurable over the interval - exact, product unchanged)
                        Sj, h = utils.hn_unmonitored_substeps(
                            Sj, h, b_step, nj[j] - 1, hn_params, shared, num_sims, antithetic=True)
                        sh = torch.sqrt(h)
                        z_max = (torch.log(barrier / Sj) - (b_step - 0.5 * h)) / sh
                        if eta == BARRIER_UP:
                            p = utils.norm_cdf(z_max)
                            Z = utils.norm_icdf(torch.clamp(u[j] * p, eps, 1.0 - eps))
                        else:
                            p = 1.0 - utils.norm_cdf(z_max)
                            Z = utils.norm_icdf(torch.clamp((1.0 - p) + u[j] * p, eps, 1.0 - eps))
                        if direction == BARRIER_OUT:
                            P = P + (1.0 - p) * L * rebate_per_unit * D[j].reshape(-1, 1)
                        L = p * L
                        Sj, h = utils.hn_daily_advance(Sj, h, b_step, Z, *hn_params)
                    else:
                        # non-barrier observation date (incl. expiry): full nj unconditional steps
                        Sj, h = utils.hn_unmonitored_substeps(
                            Sj, h, b_step, nj[j], hn_params, shared, num_sims, antithetic=True)
                    continue

                r_j = r[j].reshape(-1, 1)      # [batch, 1]
                sig_j = sigma[j].reshape(-1, 1) # [batch, 1]

                if isdigital and j == N_fix - 1:
                    # integrate the terminal step: a sampled indicator puts no density term on the
                    # tape, and the delta/vega it costs are measured in the docstring
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
                    # z_max: the standard-normal threshold at which spot = H
                    # GBM: log(S_j/S_{j-1}) = r_j + sig_j * Z  (r_j already includes -0.5*var term)
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
                payoff = ((phi * (Sj - strike) > 0).to(D_T.dtype) if isdigital
                          else torch.relu(phi * (Sj - strike)))
                surv_payoff = L * payoff

            if direction == BARRIER_OUT:
                # survivors receive the vanilla payoff at expiry
                P = P + surv_payoff * D_T
            else:
                # BARRIER_IN via in-out parity: KI = Vanilla - KO_pure + rebate * E[survival]
                # Individual paths can go negative (P is a control-variate estimator, not a
                # per-path price), but the option value is always ≥ 0, so clamp the mean.
                P = vanilla_pv - D_T * (surv_payoff - L * rebate_per_unit)

            mcmc.append(P.mean(dim=1).clamp(min=0.0))

        return (torch.stack(mcmc),)

    # --- outer block loop (identical setup to pv_discrete_barrier_option) ---
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

    # per-unit because sim_spot_oss scales by nominal; divide by the UNSIGNED size or Buy_Sell,
    # already inside nominal, cancels and a sold knock-out books the rebate it pays as a receipt
    rebate_per_unit = cash_rebate / size

    # opt-in Heston-Nandi spot model (SpotModel='HestonNandi'); absent => byte-identical GBM path
    hn = 'HN_Params' in factor_dep
    if hn:
        hn_p = {x.name[-1]: shared.t_Static_Buffer[x].reshape(-1, 1)
                for x in factor_dep['HN_Params'][0][utils.FACTOR_INDEX_Offset]}
        *hn_params, H0 = (hn_p[k] for k in utils.HN_PARAM_NAMES)  # the four recursion params + H0 (seeds h)
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

    # Per-scenario flag: has the barrier been crossed at a past discrete observation date?
    # Once set, KO scenarios are worth 0 and KI scenarios are worth the vanilla European.
    barrier_hit = shared.one.new_zeros(shared.simulation_batch, dtype=torch.bool)
    row_ofs = 0
    # A crossing is OBSERVED, so its value jump is real and must not be smoothed; what ordinary AAD
    # drops is the flux of scenarios across the barrier. Recording the decision costs nothing when
    # sensitivities are not wanted, so it is gated rather than always on.
    boundary_aad = getattr(shared, 'boundary_aad', False)
    b_gaps, b_crossed, b_obs_before, b_alive, b_dead = [], [], [], [], []

    for index, (discount_block, spot_block, moneyness_block, rem_exp) in enumerate(
            utils.split_counts([discount, spot, moneyness, expiry_years], counts, shared)):

        t_block = discount_block.time_grid
        sample_index_t = start_index[index]

        # per-row hit mask [N_mtm, batch]: hits at STRICTLY EARLIER observations only - this block's
        # own date is priced by sim_spot_oss as its zero-length first step (see docstring)
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
        # the EUROPEAN read: the surface's own vol at (K, remaining expiry). The implied surface
        # has NO consumer under a non-GBM spot model - the OSS steps the recursion and the
        # already-hit leg is the same closed form - so neither vol quantity is built there, and a
        # leg that reaches for one fails on the shape instead of quietly pricing another model
        expiry_vols = utils.calc_time_grid_vol_rate(
            factor_dep['Volatility'], moneyness_block, expiry,
            shared) if not hn else spot_block.new_empty(0)

        if factor_dep.get('Check_Payoff_Type', False):
            # a DEAL-LEVEL drift adjustment, so its input stays the expiry read; the strip below
            # then rides the adjusted carry. Compo is unreachable here - `b_adj` is a python float
            # on that branch and `torch.unsqueeze` refuses it - so `adj['vol']` is `vols` identity.
            adj = calc_vol_adjustment(factor_dep, deal_time, expiry, expiry_vols, shared)
            expiry_vols = adj['vol']
            drifts = drifts + torch.unsqueeze(adj['b_adj'], 1)

        sample_ts = drifts.new(
            np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)]))
        cum_t = drifts.new(fixing_block)
        # the carry over each INTERVAL, which is not the zero carry to each fixing (see
        # forward_carry_rate); `fixing_block` is the cumulative strip the difference needs
        fwd_drifts = forward_carry_rate(drifts, cum_t, sample_ts)
        # the SIMULATION read: the surface at every fixing's own TENOR and the deal's own declared
        # moneyness, differenced into forward variance. An implied vol is cumulative, so an
        # interval's variance is a DIFFERENCE and not sigma(T)^2*dt - which telescopes to the same
        # total and is therefore invisible to every European limit (see forward_vol_rate).
        interval_vols = forward_vol_rate(forward_vol_strip(
            deal_data, shared.one * strike, spot_block, drifts, fixing_block, shared,
            invert_moneyness, use_forwards), cum_t, sample_ts) if not hn else spot_block.new_empty(0)
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
                # value the same European on the same state, so they read one expression
                log_fwd = total_log_forward(fwd_drifts, sample_ts)              # [N_block, batch]
                if hn:
                    # an already-knocked-in KI IS a vanilla, and it is priced under the DECLARED
                    # model on the parity leg's own discretisation - same per-row n_total, same
                    # scalar carry - so one law prices both legs of this pricer (see sim_spot_oss)
                    carry_det = log_fwd.detach()  # a guard reads a magnitude, not the tape
                    carry_spread = float((carry_det.amax(dim=1) - carry_det.amin(dim=1)).max())
                    if carry_spread > 1.0e-9:
                        raise ValueError(
                            'HN already-hit KI leg needs batch-constant carry; carry varies across '
                            'scenarios by {:.2e} (stochastic rates?) - extend hn_call to batched '
                            'carry or price this barrier under GBM'.format(carry_spread))
                    om, al, be, ga = (v.reshape(-1)[0] for v in hn_params)
                    H0_s = H0.reshape(-1)[0]
                    rows = []
                    for row, spot_row in enumerate(spot_block):
                        # n_steps drives the variance recursion, so it is a scalar per MTM row
                        n_total = sum(max(int(round(float(t) * hn_spy)), 1) for t in sample_ts[row])
                        r_step = log_fwd[row][0] / n_total
                        if isdigital:
                            q_below = utils.hn_cdf_logret(
                                torch.log(strike / spot_row), n_total, H0_s, om, al, be, ga, r_step)
                            rows.append((1.0 - q_below) if phi == OPTION_CALL else q_below)
                        else:
                            rows.append((utils.hn_call if phi == OPTION_CALL else utils.hn_put)(
                                spot_row, strike, n_total, H0_s, om, al, be, ga, r_step
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
            # Every row of every scenario has already resolved — skip OSS entirely
            theo_cashflow = hit_value
        else:
            simulate = partial(sim_spot_oss, sample_index_t, sobol, shared.MCMC_sims)
            theta = (spot_block, interval_vols, sd_to_expiry, sample_ts, fwd_drifts,
                     discount_rates) + (tuple(hn_params) + (H0,) if hn else ())
            # the SAME callable either way: under the node it is called twice (see InnerMCRecompute)
            oss_result, = InnerMCRecompute.run(shared, simulate, *theta)
            oss_result = nominal * oss_result
            # some_hit: override per-row per-scenario where the barrier had already been crossed
            theo_cashflow = torch.where(
                row_barrier_hit, hit_value, oss_result) if some_hit else oss_result

        if boundary_aad:
            b_dead.append(hit_value.detach())
            # all_hit blocks skipped the OSS, so they carry no counterfactual
            b_alive.append((hit_value if all_hit else oss_result).detach())

        mtm_list.append(theo_cashflow)

    if boundary_aad and b_gaps and time_grid.report_index is not None:
        # Branches stay on THIS pricer's grid and in its own currency; `to_mtm` is the deal's own
        # map onto the MTM grid, which the collateral chain consumes. report_index rides along so
        # the additive route can go on to the reporting grid at the point of use - and is None on
        # a grid nobody reports off (an HMC tradable, a calibration's benchmark grid), which is
        # what makes the registration not worth making.
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
    """
    Single-barrier option valued by closed form at each scenario date, weighted by a Brownian-bridge
    touch probability.

    A barrier is monitored CONTINUOUSLY but the scenario grid only observes its own dates, so asking
    "is the spot beyond the barrier now" misses every path that crossed and came back - measured on a
    quarterly grid, that overstates survival by 0.18, i.e. 61% too many paths treated as still alive.
    ``touched`` is that probability and it WEIGHTS both branches, so it carries through with no
    further change - which is what makes the deal value the correct expectation instead of one draw,
    and what makes it differentiable where an indicator was not.

    Discrete monitoring uses Broadie-Glasserman-Kou: a discretely monitored barrier is priced by the
    continuous formula against a barrier shifted AWAY from the live region by the monitoring
    interval's vol. The shift direction is the barrier TYPE, not where the spot happens to sit - the
    older ``2 * (barrier > s_t) - 1`` evaluates to exactly this on the surviving support, and the
    surviving support is all that is ever weighted, so dropping the indicator is inert.
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
    # `touched` is a bridge PROBABILITY weighting both branches, not an indicator (see docstring)
    interval_variance = utils.bridge_interval_variance(shared, factor_dep, deal_time)
    prev_spot = None

    for index, (raw_sig, exp, b_t, r_t, s_t, f_t, cash_index) in enumerate(zip(
            sigma, expiry_years, b, r, spot, forward, deal_data.Time_dep.deal_time_grid)):

        # barrier options are very sensitive to vols - so we clamp them to 5%
        sig = raw_sig.clamp(min=0.05)

        # Broadie-Glasserman-Kou shift, AWAY from the live region by the monitoring interval's vol;
        # the direction is the barrier TYPE, not the spot's side (see docstring)
        barrier_t = barrier * torch.exp(
            (1.0 if eta == BARRIER_UP else -1.0) * sig * factor_dep['Barrier_Monitoring']
        ) if factor_dep['Barrier_Monitoring'] else barrier

        # The bridge has to test the SAME barrier the closed form prices against. Handed the raw
        # one it monitored continuously a barrier the product observes monthly, and the two
        # disagreed: the profile decayed 11.6% where it must be a martingale.
        touched = utils.barrier_touched(prev_touched, prev_spot, s_t, barrier_t,
                                        interval_variance[index], eta == BARRIER_UP)
        prev_spot = s_t

        if (1 - touched).any() and expiry[index] > 0:

            payoff = buy_or_sell * nominal * barrierOption(
                sig, exp, cash_rebate / nominal, b_t, r_t, s_t, barrier_t)
        elif direction == BARRIER_OUT and expiry[index] == 0.0:
            # the closed form divides by sqrt(expiry), so it cannot be asked for the value AT
            # expiry - where a surviving knock-out is worth exactly its intrinsic. The knock-IN
            # branch already carries its own expiry case and reaches here worth only its rebate.
            payoff = buy_or_sell * nominal * torch.relu(phi * (f_t - strike))
        else:
            payoff = 0.0

        barrier_part = (1.0 - touched) * payoff

        if direction == BARRIER_IN:
            # barrier ? and in
            payoff_european = buy_or_sell * (
                utils.black_european_option(f_t, strike, sig, expiry[index], 1.0, phi, shared) * torch.exp(
                    -r_t * exp) if expiry[index] else torch.relu(phi * (f_t - strike))
            )
            european_part = touched * (nominal * payoff_european)
            rebate_part = buy_or_sell * cash_rebate * (1 - touched) if expiry[index] == 0.0 else 0.0
            mtm_list.append(rebate_part + european_part + barrier_part)
        else:
            # barrier ? and out
            rebate_part = buy_or_sell * cash_rebate * (touched - prev_touched)
            mtm_list.append(rebate_part + barrier_part)
            # settle cashflows (The potential rebate)
            if cash_rebate and expiry[index] > 0.0:
                cash_settle(shared, payoff_currency, cash_index, rebate_part)

        prev_touched = touched

    # settle cashflows (The one at expiry)
    cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], mtm_list[-1])
    return torch.stack(mtm_list)


def pv_one_touch_option(shared, time_grid, deal_data, nominal, spot, b,
                        tau, payoff_currency, invert_moneyness=False, use_forwards=False):
    """
    One-touch (or no-touch) digital valued by closed form at each scenario date, weighted by the same
    Brownian-bridge touch probability ``pv_barrier_option`` uses.

    Under ``Payment_Timing='Expiry'`` the nominal is paid at expiry if the barrier was EVER touched,
    so between the touch and expiry the path holds a CERTAIN claim on the nominal - worth its
    discounted value, not nothing. Carrying zero there reported a touched one-touch as worthless for
    the rest of its life and then jumped it to the nominal on the last date.
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
    # Same continuous-vs-observed mismatch as pv_barrier_option, but this payoff pays ON touch,
    # so missing a crossing UNDERSTATES it rather than overstating survival.
    interval_variance = utils.bridge_interval_variance(shared, factor_dep, deal_time)
    prev_spot = None

    for index, (raw_sig, exp, b_t, r_t, s_t, cash_index) in enumerate(zip(
            sigma, expiry_years, b, r, spot, deal_data.Time_dep.deal_time_grid)):

        # the same monitoring-adjusted barrier the closed form below prices against - see
        # pv_barrier_option for why the shift direction is the barrier type and not the spot's side
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

        # settle cashflows (The potential rebate)
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
            # value rather than zero (see docstring)
            rebate_part = touched * buy_or_sell * nominal * torch.exp(-r_t * exp)

        mtm_list.append(rebate_part + one_touch_part)
        prev_touched = touched

    return torch.stack(mtm_list)


def pv_partial_barrier_option(shared, time_grid, deal_data, nominal,
                              spot, b, tau, tau1, payoff_currency, invert_moneyness=False):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    factor_dep = deal_data.Factor_dep
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # work out what we're pricing
    barrierType = deal_data.Instrument.field['Barrier_Type']
    isKnockIn = barrierType in ['Up_And_In', 'Down_And_In', 'In']

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
        adj_barrier = barrier * torch.exp(
            (2.0 * (barrier > spot[0][0]).type(shared.one.dtype) - 1.0) * sigma * factor_dep['Barrier_Monitoring'])
    else:
        adj_barrier = barrier

    r = torch.squeeze(discounts.gather_weighted_curve(
        shared, tau.reshape(-1, 1), multiply_by_time=False), dim=1)

    # cache the expiry tenors
    expiry_years_key = ('Expiry_Years', tuple(expiry), tuple(limit))
    if expiry_years_key not in factor_dep:
        factor_dep[expiry_years_key] = (spot.new(expiry.reshape(-1, 1)), spot.new(limit.reshape(-1, 1)))

    expiry_years, limit_years = factor_dep[expiry_years_key]

    barrier_payoff = buy_or_sell * nominal * getpartialbarrierpayoff(
        isKnockIn, eta, phi, spot_prior, strike, adj_barrier,
        deal_data.Instrument.field['Barrier_At_Start'] == 'Yes',
        limit_years, expiry_years, r, b, sigma)

    if need_spot_at_expiry:
        # work out barrier
        if eta == BARRIER_UP:
            touched = (spot[:-1] < barrier) & (spot[1:] > barrier)
        else:
            touched = (spot[:-1] > barrier) & (spot[1:] < barrier)

        # barrier payoff
        barrier_touched = F.pad((torch.cumsum(touched, dim=0) > 0).type(shared.one.dtype), [0, 0, 1, 0])
        first_touch = barrier_touched[1:] - barrier_touched[:-1]
        # final payoff
        payoff_at = buy_or_sell * torch.relu(phi * (spot_at - strike))

        if direction == BARRIER_IN:
            forward = spot_prior * torch.exp(b * expiry_years)
            payoff_prior = utils.black_european_option(
                forward, strike, sigma, expiry, buy_or_sell, phi, shared) * torch.exp(-r * expiry_years)
            european_part = barrier_touched * (nominal * torch.cat([payoff_prior, payoff_at], dim=0))
            barrier_part = (1.0 - barrier_touched) * F.pad(
                barrier_payoff, [0, 0, 0, 1], value=buy_or_sell * cash_rebate)
            combined = european_part + barrier_part
            # settle cashflows (can only happen at the end)
            cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], combined[-1])
        else:
            # barrier out
            barrier_part = (1.0 - barrier_touched) * torch.cat([barrier_payoff, nominal * payoff_at], dim=0)
            rebate_part = buy_or_sell * cash_rebate * first_touch
            combined = F.pad(buy_or_sell * cash_rebate * first_touch, [0, 0, 1, 0]) + barrier_part
            # settle cashflows (The one at expiry)
            cash_settle(shared, payoff_currency, deal_data.Time_dep.deal_time_grid[-1], barrier_part[-1])
            # settle cashflows (The potential rebate knockout)
            if cash_rebate:
                for cash_index, cash in zip(deal_data.Time_dep.deal_time_grid[1:], rebate_part):
                    cash_settle(shared, payoff_currency, cash_index, cash)
    else:
        combined = barrier_payoff

    return combined


def pv_american_option(shared, time_grid, deal_data, nominal, moneyness, spot, forward):
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

    # make sure there are no zeros
    tau = spot.new(expiry.reshape(-1, 1)).clamp(min=1e-5)
    # cost of carry
    b = torch.log(forward / spot) / tau
    # interest rates
    r = torch.squeeze(
        discount.gather_weighted_curve(shared, tenor_in_days.reshape(-1, 1), multiply_by_time=False), dim=1)
    # actual volatility
    vol = sigma * torch.sqrt(tau)
    # adjust if this is a put option
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
    # make sure we avoid nans
    safe_sqrt = ((b_over_sigma2 - 0.5) ** 2 + 2.0 * safe_r / sigma2).clamp(min=1e-6)
    beta = (0.5 - b_over_sigma2) + torch.sqrt(safe_sqrt)
    r_b = safe_r - safe_b
    # calculate the barrier
    B_0 = K * torch.maximum(safe_r / r_b, torch.ones_like(safe_r))
    B_inf = K * beta / (beta - 1)

    # check if the strike is > 0
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

    # check the first knockout
    knocked_out = early_exercise.cumsum(axis=0) > 0
    first_knockout = torch.concat([knocked_out[0].reshape(1, -1), knocked_out[1:] ^ knocked_out[:-1]])
    # make sure we are still in force today
    knocked_out[0] = False
    value = factor_dep['Buy_Sell'] * nominal * theo_price * ~knocked_out

    # handle cashflows
    exercise_val = factor_dep['Buy_Sell'] * nominal * (S - K) * first_knockout
    for t, cashflows in zip(deal_data.Time_dep.deal_time_grid, exercise_val):
        if cashflows.any():
            cash_settle(shared, factor_dep['SettleCurrency'], t, cashflows)

    return value


def pv_european_option(shared, time_grid, deal_data, nominal, moneyness, forward, binary=False):
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    tenor_in_days = factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]
    settle_in_days = factor_dep['Settlement'] - deal_time[:, utils.TIME_GRID_MTM]
    expiry = discount.code[0][utils.FACTOR_INDEX_Daycount](tenor_in_days)
    vols = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, expiry, shared)

    # check if this is a compo or quanto deal
    if factor_dep.get('Check_Payoff_Type', False):
        # need quanto/compo adjustments
        adj = calc_vol_adjustment(factor_dep, deal_time, expiry, vols, shared)
        vols = adj['vol']
        forward = adj['s_adj'] * forward * torch.exp(adj['b_adj'] * shared.one.new(expiry.reshape(-1, 1)))

    if binary:
        theo_price = utils.black_european_option(
            forward, factor_dep['Strike_Price'], vols, expiry,
            factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared, cash_payoff=nominal)
        value = theo_price
    else:
        theo_price = utils.black_european_option(
            forward, factor_dep['Strike_Price'], vols, expiry,
            factor_dep['Buy_Sell'], factor_dep['Option_Type'], shared)
        value = nominal * theo_price

    discount_rates = torch.squeeze(
        utils.calc_discount_rate(discount, settle_in_days.reshape(-1, 1), shared), dim=1)

    # handle cashflows (if necessary)
    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], value[-1])

    return value * discount_rates


def pv_MC_Tarf(shared, time_grid, deal_data, spot, fx_rep):
    """
    One-step survival Monte Carlo for TARF (autograd-friendly).
    - Analytic KO-in-step: adds (1 - p_survive) * remaining_target at each fixing
    - Survival branch: GBM step with Z ~ N(0,1) truncated at z_max from the PnL barrier
    - Accrual: only ITM leg contributes to target; OTM affects PV only (and optional knock-in barrier)

    KNOWN LIMITATIONS of the opt-in Heston-Nandi spot model (SpotModel='HestonNandi'); accepted
    design semantics, stated so they are not re-discovered:
    (F1) each fixing spans n_sub = max(round(dt * Steps_Per_Year), 1) whole HN daily steps, so the
         per-fixing variance is quantised to an integer number of trading days with a floor of one -
         a sub-3-calendar-day fixing still carries one full day of HN variance.
    (F2) Steps_Per_Year (default 252) MUST match the daily clock the HN factor was calibrated on; a
         mismatch silently rescales the variance horizon (n_sub and hence Sum h) with no error.
    (F3) the n_sub-1 unconditional sub-step normals are drawn from the global torch generator, not
         the reset Sobol stream - deterministic per run seed, but finite-difference greeks w.r.t.
         NON-HN factors pick up this RNG noise (HN-parameter greeks are AAD and unaffected).
    (F4) h re-seeds to H0 at the start of every MTM row (static implied factor); there is no
         outer-grid variance term structure - each valuation date restarts the recursion at H0.

    The HN scalars (Omega, Alpha, Beta, Gamma_Star, H0) ride the AAD graph out of
    ``t_Static_Buffer``, unpacked exactly like the SVI wing params (utils.py:2052); when they are
    absent ``sim_spot_tarf`` takes the byte-identical GBM path. Asset-class agnostic - it works for
    the FX cross here and is unchanged for an equity or commodity underlying.

    RECOMPUTE (``shared.recompute_inner_mc``, the ``Recompute_Inner_MC`` calculation field). Off,
    ``sim_spot_tarf`` is called once and taped. On, it is called through ``InnerMCRecompute`` -
    untaped to price, re-run under ``enable_grad`` to differentiate - and the peak becomes one
    block's graph rather than every block's. That is why the vol strip is built at the call site
    (``forward_vol_strip`` into ``forward_vol_rate``) and why the simulation returns its settled
    cashflows and boundary registrations instead of performing them: the node's inputs must be its
    whole theta surface, and a side effect inside it would fire twice. Off is bit-identical, price
    and gradient.

    BOUNDARY AAD (``shared.boundary_aad``). Two decisions are taken on simulated state and jump: the
    target filling (the deal has redeemed and is worth nothing thereafter) and the OTM leg knocking
    in. Both jumps are real product economics, so neither may be smoothed; what ordinary AAD drops is
    the flux of paths across them. Recording costs nothing when sensitivities are not wanted, hence
    the gate.

    The redemption gap is built from the UNCLAMPED accrual ``raw``, which is the same series with the
    min taken off - every step's clamp is at ``targetValue`` and the increments are non-negative, so
    ``accumulation[k] == min(raw[k], target)`` exactly. The clamp is also what kills the derivative
    AT the decision (past it the reported accrual is a constant), so a gap built from
    ``accumulation`` would carry no graph on the very paths that made the decision, while
    ``raw[k] - target`` is the same test, signed the same way, and carries every factor that moved
    the fixings there. It is in target units and deliberately NOT rescaled to be dimensionless the
    way a barrier gap is: ``boundary_weights`` sets its kernel width to ``bandwidth * gap.std()``, so
    the whole estimator is invariant to the gap's scale - measured, dividing through by the target
    left every digit of the correction unchanged.

    There is one redemption decision per block: ``q`` is read off that block's accrual, so it is the
    same test for every row the block covers, and a later block's accrual can only be larger - the
    flags latch, which is what makes this the LATCHED shape. The gaps are ``expand``-ed because block
    0's accrual is the HISTORIC one, a single number rather than a per-scenario tensor. That decision
    is registered like any other rather than skipped: it carries no graph, so the local-linear fit
    degenerates and it contributes exactly zero, whereas skipping it would leave the latch
    reconstruction wrong on a deal whose target had already filled before the base date.
    """

    # --- Accumulated target from past observed fixings --------------------------
    def accrued(s, k, C, inverted):
        if inverted:
            iv = (1.0 / s - 1.0 / k) * C * (-1.0)
        else:
            iv = (s - k) * C
        return F.relu(iv).reshape(-1, 1)

    def calc_accum_value(targetValue, accumulated, s, k, C, inverted):
        return (accumulated + accrued(s, k, C, inverted)).clamp(max=targetValue)

    def bs_call_put_fwd(F, K, sdt, D):
        """
        Put via parity in forward measure.
        """
        d1 = (torch.log(F / K) + 0.5 * sdt * sdt) / sdt
        d2 = d1 - sdt
        call =  D * (F * utils.norm_cdf(d1) - K * utils.norm_cdf(d2))
        return call, call - D * (F - K)

    def sim_spot_tarf(settlement, sobol, num_sims,
                      spot_prices, times, carry, prev_accum, discount_rates, vols_all,
                      past_fixings, *hn_scalars):
        """
        Inner one-step-survival Monte Carlo over one block of MTM rows; returns the mean PV per row.

        PURE, and split into a bound half and a theta half, because `InnerMCRecompute` calls it
        TWICE - once untaped to price and once taped to differentiate - whenever
        `Recompute_Inner_MC` is on. The leading arguments are the block's shape and are bound per
        block; the trailing ones are every tensor it reads that can carry a graph, which is what
        the node can return a gradient for. Its by-products are RETURNED rather than performed:
        a settled cashflow and a boundary registration must happen once, off the forward's result,
        and the caller is the only place that knows which call it is looking at.

        Returns `(mtm, untriggered, settled, settle_rows, knock_rows) + gaps + jumps`: the block's
        rows, the counterfactual rows the redemption latch needs, the first-fixing cashflow of every
        row that settles today, the block-local rows those cashflows and those knock-in decisions
        land on, and then one knock-in gap and one knock-in jump per (row, fixing). The gaps are the
        outputs whose cotangent carries the boundary correction back into the simulation.

        Heston-Nandi: ``h`` is re-seeded to ``H0`` at the start of each forward simulation (each MTM
        row) and then recursed continuously - including through the truncated final draw - across
        every daily sub-step. At the base date ``h1 = H0`` is exactly the calibrated initial
        variance; at a future MTM row it is the base-date seed, per (F4) above. Each fixing spans
        ``n_sub`` whole daily steps of which the first ``n_sub - 1`` are UNCONDITIONAL and
        UNMONITORED - the TARF only accrues and knocks AT the fixing - so the OSS truncation applies
        ONLY on the last step, where the daily log-return is conditionally Gaussian given ``h``. The
        barrier ``B_pnl`` depends only on the remaining target ``R``, constant over the interval and
        F-measurable at the truncation, so the scheme is EXACT (the product is unchanged). A
        mean-matched single-step normal bridge misprices the KO probability, with the error growing
        in ``n``; naive daily-monitored OSS prices a DIFFERENT product.

        Under ``boundary_aad`` two decisions are forked here.

        Redemption: a path whose accrual has reached the target has redeemed, so the jump to zero is
        real product economics - but ``q`` is an exact float equality firing on a POSITIVE-MEASURE
        set (``calc_accum_value`` clamps AT ``targetValue``), which makes it a knock-out decision
        like any other, whose flux ordinary AAD drops. The counterfactual is the SAME loop on a
        weight that was never zeroed: the weight feeds nothing back into the spot, the truncation or
        the remaining target, so one extra accumulator answers "had it not quite filled" exactly,
        with no re-simulation and no random number drawn. It IS the left limit - everything the
        remaining target enters is continuous at 0.

        Knock-in: decided by ``Sj``, a draw of THIS pricer's inner simulation, so it is one decision
        per inner path and the jump it carries is the OTM leg switching on for that path alone,
        ``J(knocked in) - J(did not) = -Dj * L * p * relu(-intr) * N_otm``, everything else being
        shared. The gap is signed so ``gap > 0`` means KNOCKED IN, and is taken in log space so a
        bandwidth means the same thing here as at a barrier observation. At a fixing this row has
        ALREADY OBSERVED (``dt <= 0``), ``Sj`` is a past reset rather than an inner draw, so the
        decision is one per SCENARIO and the gap is constant along the inner axis - which
        ``expand_as`` says explicitly. That is not an approximation: the pooled kernel over ``B * n``
        samples of which only ``B`` are distinct reproduces the whole-scenario estimator exactly,
        because the ``1/n`` in the weights cancels the ``n`` copies of each jump. Under CMC every
        fixing date IS a reporting row (a deal's own dates are folded into the grid), so skipping
        that case leaves one decision per fixing unregistered - measured at 5.50% of the CVA delta
        on the repo's own fixture.
        """

        # Styles & shapes follow your autocall implementation
        eps = torch.finfo(shared.one.dtype).eps
        # easier to type
        K = strike
        hn = bool(hn_scalars)
        if hn:
            *hn_params, H0 = hn_scalars
        # Per-block results, and the by-products the caller performs once (see docstring)
        mcmc, alive, settled, gaps, jumps = [], [], [], [], []
        settle_rows, knock_rows = [], []
        # Loop over block rows (same zipped signature you use)
        for i, (D, s, carry_rate, delta_t, tau) in enumerate(zip(
                discount_rates, spot_prices, carry, times, settlement)):

            # reduced_samples = number of GBM “substeps” in this coupon/fixing leg
            reduced_samples = len(delta_t)
            if not hn:
                vols = vols_all[i]
            # RNG (uniforms for truncated draws)
            if reduced_samples:
                if sobol:
                    u = shared.quasi_rng(shared.simulation_batch, reduced_samples * num_sims)[1].T.reshape(
                        reduced_samples, shared.simulation_batch, -1)
                else:
                    u = torch.rand([reduced_samples, shared.simulation_batch, num_sims],
                                   dtype=shared.one.dtype, device=shared.one.device)

                # now we should have antithetic uniform samples
                u = torch.concat([u,1.0-u], dim=-1)

            Sj = torch.unsqueeze(s, 1)  # [batch, 1] as you do
            # HN predictable variance of the first daily step; re-seeded per MTM row (see docstring)
            h = H0 if hn else None
            # Running PV and survival weight
            P = shared.one.new_zeros((shared.simulation_batch, 2*num_sims))
            L = shared.one.new_ones((shared.simulation_batch, 2*num_sims))
            # Remaining target (R) per path at this block start
            remaining_target = targetValue-prev_accum
            # which simulations are still alive?
            q = remaining_target == 0.0
            # redemption is a knock-out decision like any other; L_alive runs the same loop on a
            # weight that was never zeroed - the left limit, not a re-simulation (see docstring)
            L_alive = L if not boundary_aad else torch.ones_like(L)
            P_alive = P
            L = torch.where(q, torch.zeros_like(L), L)
            # Update the remaining targets
            R = (remaining_target * factor_dep['Notional1']).expand_as(P)
            # Per-fixing notional tensors (broadcastable to [batch, sims])
            N_itm = notional1
            N_otm = notional2
            # Iterate over fixings within this block using your “coupon_index”-like stepper
            for j in range(reduced_samples):
                # `carry` and `vols` both arrive as INTERVAL strips (forward_carry_rate,
                # forward_vol_rate) - the differencing they need is the same statement twice
                dt = delta_t[j]
                Dj = D[j].reshape(-1, 1)
                use_past_fixing = False

                if dt > 0:
                    fwd_carry = carry_rate[j].reshape(-1, 1)  # [batch,1]
                    if not hn:
                        fwd_vol = vols[j].reshape(-1, 1)      # [batch,1]
                        vol_dt = fwd_vol * torch.sqrt(dt)
                else:
                    #use past fixing
                    use_past_fixing = True

                if not use_past_fixing:
                    # ---- ONE-STEP SURVIVAL: compute p_survive via PnL barrier -----------------
                    # Standard accrual: KO within this step if (S_i - K)*cp >= R/N_itm and ITM
                    # => spot-level cap B_pnl = K + (R/N_itm) for cp=+1; inverted has reciprocal form
                    N_i = N_itm
                    if not invertedTarget:
                        B_pnl = K + (R / N_i) * callOrPut # [batch, sims]
                    else:
                        # (1/S_i - 1/K)*(-cp) * N_i >= R
                        # call (cp=+1): 1/K - 1/S >= R/N  =>  S >= 1/(1/K - R/N)  =>  rhs = 1/K - R/N = 1/K + (R/N)*(-cp)
                        # put  (cp=-1): 1/S - 1/K >= R/N  =>  S <= 1/(1/K + R/N)  =>  rhs = 1/K + R/N = 1/K + (R/N)*(-cp)
                        rhs = (1.0 / K) + (R / N_i) * (-callOrPut)
                        B_pnl = 1.0 / rhs

                    if hn:
                        # HN calibrates per day, so this fixing spans n_sub daily steps and only the
                        # LAST is truncated - which is what makes the scheme exact (see docstring)
                        n_sub = max(int(round(float(dt) * hn_spy)), 1)
                        b_step = fwd_carry * dt / n_sub  # per-step cost-of-carry r-q; total = fwd_carry*dt
                        # the first n_sub-1 unmonitored daily steps (shared advance; antithetic to
                        # align with the u<->1-u halves of the truncated final draw below)
                        Sj, h = utils.hn_unmonitored_substeps(
                            Sj, h, b_step, n_sub - 1, hn_params, shared, num_sims, antithetic=True)
                        sh = torch.sqrt(h)
                        fwd_drift = b_step - 0.5 * h  # per-step drift of the FINAL (monitored) daily step
                        z_max = (torch.log(B_pnl / Sj) - fwd_drift) / sh
                    else:
                        # Lognormal cap -> z_max
                        # NOTE: use current Sj as “S_{i-1}”
                        fwd_drift = (fwd_carry - 0.5 * fwd_vol * fwd_vol) * dt
                        z_max = (torch.log(B_pnl/Sj) - fwd_drift) / vol_dt
                    PhiB = utils.norm_cdf(z_max)  # = P(Z <= z_max)
                    
                    if (callOrPut > 0):
                        p = PhiB  # survival = Z <= z_max
                        Z = utils.norm_icdf(torch.clamp(u[j] * p, eps, 1.0 - eps))
                    else:
                        p = 1.0 - PhiB  # survival = Z >= z_max
                        Z = utils.norm_icdf(torch.clamp(PhiB + u[j] * p, eps, 1.0 - eps))
                        
                    # ---- Analytic KO-in-step contribution -------------------------------------
                    # KO pays the *remaining target* this step, discounted at the j-th discount point
                    P = P + (1.0 - p) * L * R * Dj
                    if boundary_aad:
                        P_alive = P_alive + (1.0 - p) * L_alive * R * Dj
                    if hn:
                        # HN increment + h-recursion on the truncated final draw (shared advance;
                        # leverage-asymmetric because Z is survival-truncated - see hn_daily_advance)
                        Sj, h = utils.hn_daily_advance(Sj, h, b_step, Z, *hn_params)
                    else:
                        # GBM increment
                        Sj = Sj * (torch.exp(fwd_drift + vol_dt * Z) if dt > 0 else 1.0)
                else:
                    Sj = past_fixings[-min(reduced_samples, num_samples)].reshape(-1, 1)
                    p = 1.0
                # ---- Economic PV for this fixing -----------------------------------------
                # compute effective intrinsic in the correct measure
                if not invertedTarget:
                    eff_intr = (Sj - K) * callOrPut
                else:
                    eff_intr = (1.0 / Sj - 1.0 / K) * (-callOrPut)
                # clamp at the per-path remaining target (R is in currency units; divide back to target units)
                intr = eff_intr.clamp(max=remaining_target)
                itm_mask = (intr > 0.0)  # ITM
                # Optional knock-in on OTM leg
                if barrier > 0.0:
                    barrier_intr = (barrier - Sj) * callOrPut
                    barrier_hit = (barrier_intr >= 0.0).to(Sj.dtype)
                else:
                    barrier_hit = torch.ones_like(Sj)
                # economic per-fixing cashflow (signed)
                cf_itm = F.relu(intr) * N_itm  # ≥ 0
                cf_otm = F.relu(-intr) * N_otm * barrier_hit  # ≤ 0
                cf_step = L * p * (cf_itm - cf_otm)  # signed
                # add discounted PV to P
                P = P + Dj * cf_step
                if boundary_aad:
                    P_alive = P_alive + Dj * L_alive * p * (cf_itm - cf_otm)
                    if barrier > 0.0:
                        # one decision per inner path; gap > 0 means KNOCKED IN, in log space, and
                        # `expand_as` also covers the already-observed fixing (see docstring)
                        jump = (-buy_sell * Dj * L * p * F.relu(-intr) * N_otm).detach()
                        gaps.append((callOrPut * torch.log(barrier / Sj)).expand_as(jump))
                        jumps.append(jump)
                        knock_rows.append(i)
                # ---- Update remaining target R on survivors ------------------------------
                accr = cf_itm  # = F.relu(intr) * N_itm, correct for both standard and inverted
                R = torch.where(itm_mask, R - accr, R)  # survival construction ensures no overshoot
                # ---- Update survival weight ----------------------------------------------
                L = p * L
                if boundary_aad:
                    L_alive = p * L_alive
                # (Optional) settlement at tau==0 - RETURNED, because a recompute would settle twice
                if j == 0 and tau[0] == 0:
                    settled.append(cf_step.mean(axis=1))
                    settle_rows.append(i)
            # End-of-block: push mean PV over sims
            mcmc.append(P.mean(axis=1))
            if boundary_aad:
                alive.append(P_alive.mean(axis=1))

        # a fresh empty each time, never one tensor returned twice - an autograd.Function output
        # list is positional and the same object in two slots is one output with two names
        return (torch.stack(mcmc),
                torch.stack(alive) if alive else prev_accum.new_empty(0),
                torch.stack(settled) if settled else prev_accum.new_empty(0),
                settle_rows, knock_rows) + tuple(gaps) + tuple(jumps)

    # --- Main block loop --------------------------------
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # now precalc all past resets - Fixings includes settlement and fixings dates
    samples = factor_dep['Fixings']
    start_idx = samples.get_start_index(deal_time)
    start_index, counts = np.unique(start_idx, return_counts=True)

    # make sure we can access the numpy and tensor components
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
    # need to get the index of the fixing and the equity
    fixing_indices = counts.cumsum() - 1
    settle_index = settle_idx[fixing_indices]

    # params for the tarf
    targetValue = deal_data.Instrument.field['TargetLevel']
    barrier = deal_data.Instrument.field.get('Barrier', 0.0)
    notional1 = factor_dep['Notional1'] * shared.one
    notional2 = factor_dep['Notional2'] * shared.one
    strike = factor_dep['Strike_Price']
    invertedTarget = deal_data.Instrument.field['InvertedTarget']
    callOrPut = factor_dep['Option_Type']
    buy_sell = factor_dep['Buy_Sell']

    # gated: the target filling and the OTM knock-in both jump on simulated state (see docstring)
    boundary_aad = getattr(shared, 'boundary_aad', False)
    b_gaps, b_fired, b_obs_before, b_inner, alive, row_ofs = [], [], [], [], [], 0

    # opt-in Heston-Nandi spot model; absent => byte-identical GBM path (see docstring)
    hn = 'HN_Params' in factor_dep
    if hn:
        hn_p = {x.name[-1]: shared.t_Static_Buffer[x].reshape(-1, 1)
                for x in factor_dep['HN_Params'][0][utils.FACTOR_INDEX_Offset]}
        *hn_params, H0 = (hn_p[k] for k in utils.HN_PARAM_NAMES)  # the four recursion params + H0 (seeds h)
        hn_spy = factor_dep['HN_Steps_Per_Year']


    # calculate the correct accumulation to date
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
    # use a quasi random generator if we are simulating a large batch
    if shared.simulation_batch > 16:
        # reset the sobol counter (so that subsequent runs reuse the same quasi random numbers)
        sobol = True
        shared.reset_qrg()

    # split vectors into blocks based on start_index/counts
    for b_idx, (discount_block, spot_block) in enumerate(
            utils.split_counts([discount, spot], counts, shared)):
        # get the starting index for fixings/settlement
        settle_index_local = settle_index[b_idx]
        # calculate the fixing dates
        t_block = discount_block.time_grid
        fixings = (fx_samples[np.newaxis, settle_index_local:, utils.RESET_INDEX_End_Day] -
                   t_block[:, utils.TIME_GRID_MTM, np.newaxis]).clip(min=0)
        drifts = utils.calc_fx_drift(
            deal_data.Factor_dep['Underlying_Currency'], deal_data.Factor_dep['Currency'],
            fixings, t_block, shared, multiply_by_time=False)
        fixing_block = daycount_fn(fixings)
        # Discounting and settlement
        settlement = (factor_dep['Settlement'][np.newaxis, settle_index_local:] -
                      t_block[:, utils.TIME_GRID_MTM, np.newaxis])
        sample_ts = drifts.new(
            np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)]))
        discount_rates = utils.calc_discount_rate(discount_block, settlement, shared)
        cum_t = drifts.new(fixing_block)
        # the interval carry and vol strips, both theta rather than loop state, and both built
        # where their siblings build theirs (forward_carry_rate, forward_vol_rate). Both take the
        # ZERO carry `drifts`: a forward to each tenor is the CUMULATIVE integral. This pricer
        # reads the surface NOWHERE else, so the forward moneyness IS what it declares.
        fwd_drifts = forward_carry_rate(drifts, cum_t, sample_ts)
        vols = forward_vol_rate(forward_vol_strip(
            deal_data, strike, spot_block, drifts, fixing_block, shared,
            factor_dep['Invert_Moneyness'], use_forwards=True), cum_t,
            sample_ts) if not hn else spot_block.new_empty(0)

        simulate = partial(sim_spot_tarf, settlement, sobol, shared.MCMC_sims)
        theta = (spot_block, sample_ts, fwd_drifts, accumulation[settle_index_local],
                 discount_rates, vols, all_samples) + (tuple(hn_params) + (H0,) if hn else ())
        # the SAME callable either way: under the node it is called twice (see InnerMCRecompute)
        outputs = InnerMCRecompute.run(shared, simulate, *theta)
        block_mtm, block_alive, block_settled, settle_rows, knock_rows = outputs[:5]
        theo_price = buy_sell * block_mtm

        # the by-products, performed once off the forward's result (see sim_spot_tarf)
        for row, value in zip(settle_rows, block_settled):
            cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
                time_grid.mtm_time_grid, t_block[row, utils.TIME_GRID_MTM]), value)

        if boundary_aad:
            alive.append(block_alive)
            # one knock-in decision per (row, fixing), each with the row it was taken on;
            # `fixed` names the tuple's fixed prefix - adding an output means moving it, loudly
            fixed, n_inner = 5, len(knock_rows)
            b_inner.extend([[row_ofs + row, outputs[fixed + k], outputs[fixed + n_inner + k]]
                            for k, row in enumerate(knock_rows)])
            # one latched redemption decision per block; `expand` because block 0's accrual is the
            # HISTORIC one, a single number rather than a per-scenario tensor (see docstring)
            b_gaps.append((raw[settle_index_local] - targetValue).squeeze(dim=1).expand(
                shared.simulation_batch))
            b_fired.append((targetValue - accumulation[settle_index_local] == 0.0).squeeze(
                dim=1).expand(shared.simulation_batch).detach())
            b_obs_before.extend([len(b_gaps)] * theo_price.shape[0])
        row_ofs += theo_price.shape[0]
        mtm_list.append(theo_price)

    # Reporting currency MTM
    mtm = torch.cat(mtm_list, dim=0)

    # report_index is None on a grid nobody reports off - an HMC tradable, a calibration's
    # benchmark grid - and there is no reporting row for a counterfactual to land on
    if boundary_aad and time_grid.report_index is not None:
        to_mtm = deal_to_mtm_grid(time_grid, deal_data, fx_rep)
        if b_gaps:
            # Redemption: `triggered` is worth zero for every row the decision reaches, and
            # `untriggered` is the SAME loop run on a weight that was never zeroed.
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

    RECOMPUTE (``shared.recompute_inner_mc``, the ``Recompute_Inner_MC`` calculation field). Off,
    ``sim_spot`` is called once and taped. On, it goes through ``InnerMCRecompute`` - untaped to
    price, re-run under ``enable_grad`` to differentiate - and the peak becomes one block's graph
    rather than every block's. That is why the floating leg, the past equity fixings and the
    Heston-Nandi scalars are passed IN rather than read from the enclosing scope, and why the
    simulation returns its settled cashflows and its trigger registrations instead of performing
    them: the node's inputs must be its whole theta surface, and a side effect inside it would fire
    twice. ON reproduces the pre-refactor gradients bit for bit; the refactored OFF path is 1 ulp
    from them under collateral - the stacked settle reorders that chain's backward accumulation -
    which the ulp gate pins by name.

    THE VOL IS AN INTERVAL STRIP, in BOTH branches, and it used to be one implied vol read at the
    expiry moneyness and the expiry tenor - the pricer's own comment said so and asked for the fix.
    An implied vol is cumulative, so an interval's variance is a DIFFERENCE of cumulative variances
    (``forward_vol_rate``) off a surface read at each fixing's own forward moneyness
    (``forward_vol_strip``); ``pv_MC_Tarf`` differenced and this pricer did not, in the same file.
    Because the wrong allocation telescopes to the right TOTAL variance and every fixture here is
    flat, the correction leaves this repo's autocall value and exposure profile BITWISE UNCHANGED
    and moves six of thirteen CVA-gradient entries - the vol surface's own nodes, by up to 157%,
    with their SUM bit-invariant. The quanto/compo adjustment keeps the EXPIRY read: it is a
    deal-level drift adjustment, not a step of the simulation.

    WHAT IT IS WORTH, since no fixture here can show it. On two surfaces carrying the same 1y
    implied vol - flat 0.2479 against a term structure running 0.10 to 0.32 - a quarterly autocall
    struck 2% out of the money priced 0.034026629996 on BOTH, bitwise, and the sloped one is worth
    0.037094 by brute-force Monte Carlo: -8.27%, 207 standard errors. The strip reads 0.037064.
    Twice the barrier's percentage on the same surfaces, because a digital trigger observed
    quarterly is all monitoring and has no terminal payoff to telescope back into.

    TWO MONEYNESS CONVENTIONS LIVE HERE, and this pricer is where the difference is exactly
    measurable. ``forward_vol_strip`` read each fixing at its own FORWARD moneyness, mirroring
    ``pv_MC_Tarf``; the deal declares ``use_forwards = False``, so the expiry read above - and every
    value this pricer returned before the port - is the SPOT read. A ONE-COUPON autocall is a
    closed-form digital, which makes that gap noise-free: on a surface flat in tenor and smiley in
    moneyness the forward read is 0.024428368300 against the declared 0.024490310460, -0.2529%,
    both exact. The strip now takes the DECLARED read, so this pricer reproduces its own European
    quote to 0 ULP; whether a simulation should instead read the quote its own forward implies is a
    modelling decision and it is open in the roadmap. It is NOT a consequence of the interval strip,
    which prices identically to 6.9e-18 under either convention on a smile-free surface.

    BOUNDARY AAD (``shared.boundary_aad``). The coupon trigger observed on its own fixing date is a
    real redemption, so the value is not what is wrong; what ordinary AAD drops is the flux of
    scenarios across the threshold. Its gap is decided on ``Sj`` INSIDE the simulation - the row's
    own spot, or a past fixing, selected by loop state - so under the node it is an OUTPUT and the
    correction's coefficient arrives as that output's cotangent. Building it outside instead would
    mean a second spelling of the trigger condition, which is the failure the node exists to avoid.
    """
    def sim_autocall(S, isBarrierDate, isFixingDate, isFloatDate, floating, threshold, coupon, terminationDate):
        avg = 0.0
        averageCounter = 0.0
        fx = isQuanto * (fixFXRate - 1.0) + 1.0

        # MAIN LOOP
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
        """
        Inner one-step-survival Monte Carlo over one block of MTM rows; returns the mean PV per row.

        PURE, and split into a bound half and a theta half, because `InnerMCRecompute` calls it
        TWICE - once untaped to price and once taped to differentiate - whenever
        `Recompute_Inner_MC` is on. The leading arguments are the block's shape and are bound per
        block; the trailing ones are every tensor it reads that can carry a graph, which is what the
        node can return a gradient for. `times` is bound rather than theta because it is built from
        numpy through `Tensor.new` and carries no graph at all - and because it stays numpy on a
        block whose fixings have all been observed.

        Returns `(mtm, settled, settle_rows, event_rows) + gaps + fired + survived`: the block's
        rows, the cashflows the caller settles once, the block-local rows those cashflows and those
        trigger decisions land on, and then one gap, one fired branch and one surviving branch per
        decision. The gaps are the outputs whose cotangent carries the boundary correction back into
        the simulation; the branches are coefficients and go in detached.

        Under ``boundary_aad`` a row whose autocall is OBSERVED forks a counterfactual. ``dt == 0``
        means the fixing date IS this reporting row, so ``Sj`` is the scenario's own spot rather than
        an inner draw and no smoothing can be justified - the note really has redeemed; what the tape
        drops is the flux of scenarios across the threshold. The gap is signed so ``gap > 0`` means
        the trigger FIRED (spot at or above ``K``, hence ``p == 0``), matching
        ``jump = J(fired) - J(did not)``. Firing zeroes the weight, so every later term of the row
        vanishes and the fired branch is exactly that coupon, while the surviving branch forks there
        with the weight intact: ``P_cf``/``L_cf`` are the same accumulation on a survival weight the
        trigger never touched. They do not exist until the trigger is met, which happens at most once
        per row - a fixing lands on the last row of its block, and its first coupon is the only one
        that can be time-aligned.
        """

        timesteps, num_samples = times.shape
        hn = bool(hn_scalars)
        if hn:
            *hn_params, H0 = hn_scalars
        # the by-products the caller performs once (see docstring)
        settled, settle_rows, event_rows, gaps, fired, survived = [], [], [], [], [], []

        isBarrierDate = BarrierDates[offset:]
        isFloatingDate = Floating[offset:]
        threshold = Threshold[offset:]
        coupon = Coupon[offset:]

        # if there is exactly 1 fixing per coupon date, then there is no averaging
        if factor_dep['no_averaging']:
            # needed for numerical stability
            eps = torch.finfo(shared.one.dtype).eps
            fx = isQuanto * (fixFXRate - 1.0) + 1.0
            mcmc = []
            for i, (tau, df, s, v, carry_rate, delta_t, floating) in enumerate(zip(
                    t_tenor, discount_rates, spot_prices, vols, carry, times, floating_leg)):

                reduced_samples = len(delta_t)
                # reduced samples can be zero if there's just the floating leg left
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
                    h = H0  # re-seed the HN variance at the start of this MTM row

                P = torch.zeros((shared.simulation_batch, num_sims), dtype=shared.one.dtype, device=shared.one.device)
                L = torch.ones((shared.simulation_batch, num_sims), dtype=shared.one.dtype, device=shared.one.device)
                # counterfactual for this row's OBSERVED autocall; it does not exist until the
                # trigger is met, at most once per row (see docstring)
                P_cf = L_cf = None
                # which simulations are still alive?
                q = terminationDate != -1
                # update the Live matrix
                L = torch.where(q, torch.zeros_like(L), L)
                # set the correct shape for discounting; the vol is per INTERVAL, so it is picked
                # off the strip at the coupon's own index rather than hoisted out of the loop
                D = df.unsqueeze(2)

                coupon_index = 0

                for j, (coup, thresh, FloatingDate, barrier) in enumerate(
                        zip(coupon, threshold, isFloatingDate, isBarrierDate)):

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
                                # HN daily sub-stepping to the coupon date (autocall knocks out only
                                # AT the coupon observation, so the OSS truncation - survival = spot
                                # BELOW the autocall threshold K - applies only on the final daily step)
                                n_sub = max(int(round(float(dt) * hn_spy)), 1)
                                b_step = forward_carry * dt / n_sub
                                Sj, h = utils.hn_unmonitored_substeps(
                                    Sj, h, b_step, n_sub - 1, hn_params, shared, num_sims, antithetic=False)
                                sh = torch.sqrt(h)
                                p = utils.norm_cdf((torch.log(K / Sj) - (b_step - 0.5 * h)) / sh)
                            else:
                                vol = v[coupon_index].reshape(-1, 1)  # [batch,1]
                                vol_dt = vol * torch.sqrt(dt)
                                p = utils.norm_cdf(
                                    (torch.log(K / Sj) - (forward_carry - 0.5 * vol * vol) * dt) / vol_dt)
                        else:
                            p = torch.where(K > Sj, 1.0, 0.0)
                            if tau == 0.0:
                                terminationDate = torch.where((terminationDate == -1) & (p == 0), reduced_samples, terminationDate)

                        if P_cf is not None:
                            # every LATER coupon is the same smooth p in both worlds - the weight
                            # never feeds back into Sj, so the fork is one extra accumulator rather
                            # than a second simulation
                            P_cf = P_cf + fx * (1 - p) * L_cf * coup * D[j]
                            L_cf = p * L_cf
                        elif boundary_aad and dt <= 0:
                            # the autocall is OBSERVED here (dt == 0), so Sj is the scenario's own
                            # spot and gap > 0 means the trigger FIRED (see docstring)
                            event_rows.append(i)
                            gaps.append(torch.log(Sj / K).squeeze(dim=1))
                            fired.append((P + fx * L * coup * D[j]).mean(axis=1).detach())
                            P_cf, L_cf = P, L

                        P = P + fx * (1 - p) * L * coup * D[j]
                        L = p * L

                        if fixing_aligned:
                            # prevent underflow or overflow
                            safe_pu = torch.clamp(p * u[coupon_index], min=eps, max=1.0-eps)

                            if hn:
                                # survival-truncated final draw + h-recursion (shared advance)
                                if dt > 0:
                                    Sj, h = utils.hn_daily_advance(
                                        Sj, h, b_step, utils.norm_icdf(safe_pu), *hn_params)
                                # dt<=0 (terminal fixing): no interval, Sj/h unchanged (mirrors Sj*1.0)
                            else:
                                Sj = Sj * (torch.exp(
                                    (forward_carry - 0.5 * vol * vol) * dt + vol_dt * utils.norm_icdf(
                                        safe_pu)) if dt > 0 else 1.0)
                            coupon_index += 1
                        else:
                            fixing_aligned = True

                    if barrier > 0:
                        breach = torch.where(Sj <= putBarrier, 1.0, 0.0)
                        P = P + L * D[j] * fx * breach * (rebate - (1.0 - Sj / strike))
                        if P_cf is not None:
                            P_cf = P_cf + L_cf * D[j] * fx * breach * (rebate - (1.0 - Sj / strike))

                    # RETURNED, because a replay of this simulation would settle it twice
                    if tau == 0.0:
                        settled.append(P.mean(axis=1))
                        settle_rows.append(i)

                mcmc.append(P.mean(axis=1))
                if P_cf is not None:
                    survived.append(P_cf.mean(axis=1).detach())
        else:
            isFixingDate = FixingDates[offset:]
            dt = times.unsqueeze(axis=2)
            # `vols` is the INTERVAL strip (forward_vol_rate), so this product is the interval
            # variance rather than one implied vol applied to every interval. The floor is the
            # ZERO-LENGTH interval's width and only that - `sim_spot_oss` carries the same
            # expression and the same statement - and both consumers read the SAME variance
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

                # settle potential cashflows - RETURNED, a replay would settle them twice
                if tau == 0.0:
                    settled.append(pv)
                    settle_rows.append(i)
            # if the last cashflow wasn't 0 then the autocall hasn't knocked out yet
            terminationDate = -1.0 * (terminationDate == -1) * (pv != 0.0).reshape(-1, 1)

        return (torch.stack(mcmc),
                torch.stack(settled) if settled else spot_prices.new_empty(0),
                settle_rows, event_rows) + tuple(gaps) + tuple(fired) + tuple(survived)

    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    # get the discount curve and daycount
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # set up the calculation grid
    samples = factor_dep['Fixings']
    start_idx = samples.get_start_index(deal_time)
    dual_samples = samples.dual()
    start_index, counts = np.unique(start_idx, return_counts=True)

    if factor_dep['no_averaging']:
        # resets could be prior to the coupon date - need to store this
        equity_samples = factor_dep['Price_Fixing']
        coupon_samples = factor_dep['Coupon_Fixing']
        eq_start_idx = equity_samples.get_start_index(deal_time)
        cp_start_idx = coupon_samples.get_start_index(deal_time)
        # grab the known fixings and join with any potential simulated fixings
        known_resets = equity_samples.known_resets(shared.simulation_batch)
        sim_samples = equity_samples.schedule[
            (equity_samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
            (equity_samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
        past_samples = utils.calc_time_grid_spot_rate(
            deal_data.Factor_dep['Equity'], sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)
        all_eq_samples = torch.cat(
            [torch.cat(known_resets, dim=0), past_samples], dim=0) if known_resets else past_samples
        # need to get the index of the fixing and the equity
        fixing_indices = counts.cumsum() - 1
        eq_start_index = eq_start_idx[fixing_indices]
        cp_start_index = cp_start_idx[fixing_indices]
        coupon_equity_index = equity_samples.schedule[:, utils.RESET_INDEX_Reset_Day].searchsorted(
            coupon_samples.schedule[:, utils.RESET_INDEX_Reset_Day], 'right') - 1
        coupon_equity_index = np.append(coupon_equity_index, coupon_equity_index[-1] + 1)

    # make sure we can price the floating leg (if present)
    if 'Forward' in factor_dep:
        resets = factor_dep['Cashflows'].Resets
        forward = utils.calc_time_grid_curve_rate(factor_dep['Forward'], deal_time, shared)
        old_resets = resets.get_simulated_resets(
            deal_time[:, utils.TIME_GRID_MTM].max(), factor_dep['Forward'], shared)

        cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
        cash_start_index = dict(zip(start_idx, cash_start_idx))
    else:
        # not used - set it to discount
        forward = discount

    # params for the autocallable
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

    # Heston-Nandi spot model (present iff HestonNandiModelParameters.<equity> was resolved): the
    # five GARCH scalars ride the AAD graph out of t_Static_Buffer, unpacked exactly like the TARF.
    # When absent, sim_spot takes the byte-identical GBM path. HN is only wired into no_averaging.
    hn = 'HN_Params' in factor_dep
    if hn:
        hn_p = {x.name[-1]: shared.t_Static_Buffer[x].reshape(-1, 1)
                for x in factor_dep['HN_Params'][0][utils.FACTOR_INDEX_Offset]}
        *hn_params, H0 = (hn_p[k] for k in utils.HN_PARAM_NAMES)  # the four recursion params + H0 (seeds h)
        hn_spy = factor_dep['HN_Steps_Per_Year']


    sobol = False
    # use a quasi random generator if we are simulating a large batch
    if shared.simulation_batch > 16:
        # reset the sobol counter (so that subsequent runs reuse the same quasi random numbers)
        sobol = True
        shared.reset_qrg()

    # An autocall taken on the reporting row's own spot is a real redemption, so the value is not
    # what is wrong; what ordinary AAD drops is the flux of scenarios across the threshold.
    # Recording it costs nothing when sensitivities are not wanted, so it is gated rather than on.
    boundary_aad = getattr(shared, 'boundary_aad', False)
    b_events, row_ofs = [], 0

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
        # It is NOT what the simulation steps on: that is the interval strip below.
        expiry = daycount_fn(tenor_block)
        expiry_vols = utils.calc_time_grid_vol_rate(
            factor_dep['Volatility'], moneyness_block, expiry, shared)

        if factor_dep.get('Check_Payoff_Type', False):
            # need quanto/compo adjustments
            adj = calc_vol_adjustment(factor_dep, deal_time, expiry, expiry_vols, shared)
            expiry_vols = adj['vol']
            drifts = drifts + torch.unsqueeze(adj['b_adj'], 1)

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

                # now deal with past resets
                past_resets = torch.tile(
                    torch.unsqueeze(old_resets[reset_offset:reset_offset + offset], dim=0), [size, 1, 1])
                all_resets = torch.concat([past_resets, future_resets], dim=1) if past_resets.any() else future_resets
                # we now have all the expected floating payments for each scenario
                all_int, _ = pricer_float_cashflows(all_resets, cashflows.tn, shared)
                # need to reshape this for the mcmc pricing
                floating_legs.append(torch.unsqueeze(all_int, dim=-1))

            # Sometimes we need to combine the floating legs
            floating_leg = torch.concat(floating_legs) if len(floating_legs) > 1 else floating_legs[0]
        else:
            # not used - set it to the spot_block
            floating_leg = spot_block

        # calc the discount rates
        discount_rates = utils.calc_discount_rate(discount_block, all_fixings, shared)
        # sometimes the autocall kicks out early for this batch - skip redundant calcs
        if terminationDate.any():
            if factor_dep['no_averaging']:
                fixing_index = coupon_equity_index[cp_start_index[index]]
                last_fixing = None if fixing_index == eq_start_index[index] else fixing_index
            else:
                last_fixing = None
            # the interval carry and vol strips, built where their siblings build theirs - which is
            # also what puts the AVERAGING branch's `carry * dt` and `vols * vols * dt` on interval
            # integrals. Both take the ZERO carry `drifts`; a forward to a tenor is cumulative. The
            # strip takes the default (spot) moneyness because that is what `moneyness` above - the
            # quote this pricer marks its own Europeans with - was built with.
            cum_t = drifts.new(fixing_block) if fixing_block.any() else fixing_block
            fwd_drifts = forward_carry_rate(
                drifts, cum_t, sample_ts) if fixing_block.any() else drifts
            interval_vols = forward_vol_rate(forward_vol_strip(
                deal_data, strike * shared.one, spot_block, drifts, fixing_block, shared),
                cum_t, sample_ts) if fixing_block.any() else expiry_vols.unsqueeze(-2)
            simulate = partial(sim_spot, sample_index_t, sample_ts,
                               all_fixings[:, 0], last_fixing, sobol, shared.MCMC_sims)
            theta = (spot_block, interval_vols, fwd_drifts, terminationDate, discount_rates,
                     floating_leg,
                     all_eq_samples if factor_dep['no_averaging'] else spot_block.new_empty(0)
                     ) + (tuple(hn_params) + (H0,) if hn else ())
            # the SAME callable either way: under the node it is called twice (see InnerMCRecompute)
            outputs = InnerMCRecompute.run(shared, simulate, *theta)
            theo_cashflow, block_settled, settle_rows, event_rows = outputs[:4]

            # the by-products, performed once off the forward's result (see sim_spot)
            for row, value in zip(settle_rows, block_settled):
                cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
                    time_grid.mtm_time_grid, t_block[row, utils.TIME_GRID_MTM]), value)
            # one trigger decision per row that OBSERVED its coupon, with the row it landed on
            fixed, n_events = 4, len(event_rows)
            b_events.extend([[row_ofs + row, outputs[fixed + k], outputs[fixed + n_events + k],
                              outputs[fixed + 2 * n_events + k]] for k, row in enumerate(event_rows)])
        else:
            theo_cashflow = torch.zeros_like(drifts)
        theo_price = nominal * theo_cashflow
        row_ofs += theo_price.shape[0]
        # theo_price = (theo_cashflow * discount_rates).sum(axis=1)
        # # settle potential cashflows
        # cash_settle(shared, factor_dep['SettleCurrency'], np.searchsorted(
        #     time_grid.mtm_time_grid, t_block[-1][utils.TIME_GRID_MTM]), theo_cashflow[-1][0])
        # # if the last cashflow was 0 then the autocall hasn't knocked out yet
        # terminationDate = -1.0 * (terminationDate == -1) * (theo_cashflow[-1][0] == 0.0).reshape(-1, 1)
        mtm_list.append(theo_price)

    # mtm in reporting currency
    mtm = torch.cat(mtm_list, dim=0)

    if b_events and time_grid.report_index is not None:
        # Branches stay on THIS pricer's grid and in its own currency; `to_mtm` is the deal's own
        # map onto the MTM grid, which the collateral chain consumes. report_index rides along so
        # the additive route can go on to the reporting grid at the point of use - and is None on
        # a grid nobody reports off (an HMC tradable, a calibration's benchmark grid).
        rows, gaps, fired, survived = zip(*b_events)
        shared.boundary_sets.append(utils.RowBoundarySet(
            gaps=list(gaps), rows=list(rows),
            fired=[nominal * x for x in fired], survived=[nominal * x for x in survived],
            reported=mtm.detach(), to_mtm=deal_to_mtm_grid(time_grid, deal_data, fx_rep),
            report_index=time_grid.report_index))

    return mtm


def pv_discrete_asian_option(shared, time_grid, deal_data, nominal, spot, forward,
                             past_factor_list, invert_moneyness=False, use_forwards=False):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    expiry = daycount_fn(factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
    # make sure there are no zeros
    safe_expiry = spot.new(expiry.reshape(-1, 1)).clamp(min=1e-5)
    # cost of carry
    b = torch.log(forward / spot) / safe_expiry
    # now precalc all past resets
    eps = torch.finfo(shared.one.dtype).eps
    samples = factor_dep['Samples']
    known_resets = samples.known_resets(shared.simulation_batch)
    start_idx = samples.get_start_index(deal_time)
    sim_samples = samples.schedule[
        (samples.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]

    # check if the spot was simulated - if not, hold it flat
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
    # make sure we can access the numpy and tensor components
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

        # check if we still need to account for future averages.
        if sample_tau.size:
            # strike_bar can be negative if the average is higher than the original strike price - need to clamp this.
            strike_bar = torch.clamp(
                factor_dep['Strike'] - average.reshape(1, -1).expand(counts[index], -1), min=1e-5)
            sample_fwd = spot_block.unsqueeze(1) * torch.exp(carry_block.unsqueeze(1) * sample_ts.unsqueeze(2))
            moneyness = calc_moneyness(
                (strike_bar / normalize.clamp(min=eps)).unsqueeze(1), spot_block.unsqueeze(1), sample_fwd,
                deal_data, use_forwards, invert_moneyness)
            # get the vol per sample tau - might need to generalize this if we start modelling vols through time
            vols = torch.stack([utils.calc_time_grid_vol_rate(
                factor_dep['Volatility'], mon, s_tau, shared) for mon, s_tau in zip(moneyness, sample_tau)])
            if factor_dep.get('Check_Payoff_Type', False):
                # need quanto/compo adjustments
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

            # trick to avoid nans in the gradients
            MM = torch.log(M2.clamp(min=eps)) - 2.0 * torch.log(M1.clamp(min=eps))
            var_t = torch.where((M1 > eps) & (M2 > eps), MM, 0.0)
            vol_t = torch.where(var_t > 0.0, torch.sqrt(var_t.clamp(min=eps)), 0.0)
            tau = 1.0
            forward_price = M1 * spot_block
        else:
            # We're past the averaging period - we know the intrinsic value
            strike_bar = factor_dep['Strike']
            # can't set tau exactly to 0 - it will break the AAD
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

    # potential cashflows
    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], value[-1])
    # mtm in reporting currency
    mtm = torch.cat(mtm_list, dim=0)

    return mtm


def pv_discrete_double_asian_option(shared, time_grid, deal_data, nominal, spot, forward,
                                    past_factor_list, invert_moneyness=False, use_forwards=False):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    expiry = daycount_fn(factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
    # make sure there are no zeros
    safe_expiry = spot.new(expiry.reshape(-1, 1)).clamp(min=1e-5)
    # cost of carry
    b = torch.log(forward / spot) / safe_expiry
    # load the alpha multipliers (usually just 1)
    alphas = [factor_dep['Alpha_1'], factor_dep['Alpha_2']]
    # merge the resets for both samples before calculating the start_idx
    sample_reset_days = np.union1d(
        *[factor_dep[i].schedule[:, utils.RESET_INDEX_Reset_Day] for i in ['Samples_1', 'Samples_2']])
    start_idx = np.searchsorted(sample_reset_days, deal_time[:, utils.TIME_GRID_MTM], side='right').astype(np.int64)
    # set the unique index
    start_index, counts = np.unique(start_idx, return_counts=True)
    # now precalc all past resets

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

        # check if the spot was simulated - if not, hold it flat
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
        # make sure we can access the numpy and tensor components
        dual_sample = samples.dual()
        dual_samples.append(dual_sample)
        # store the sample with the weights applied
        all_samples.append(full_sample * dual_sample.tn[:, utils.RESET_INDEX_Weight].reshape(-1, 1))
        # record the index of this sample relative the merged resets calculated earlier
        start_samples.append({x: y for x, y in zip(start_idx, sample_idx)})

    for index, (discount_block, spot_block, forward_block, carry_block) in enumerate(
            utils.split_counts([discount, spot, forward, b], counts, shared)):
        t_block = discount_block.time_grid
        tenor_block = factor_dep['Expiry'] - t_block[:, utils.TIME_GRID_MTM]

        # only do moment matching for tenors prior to expiry
        if tenor_block.any():
            # use at the money vols
            moneyness_block = (forward_block if use_forwards else spot_block) / spot_block
            moneyness = 1.0 / moneyness_block if invert_moneyness else moneyness_block
            vols = utils.calc_time_grid_vol_rate(factor_dep['Volatility'], moneyness, daycount_fn(tenor_block), shared)

            mu = []
            sigma = []
            lambdas = []
            sample_fts = []
            sample_tss = []

            for alpha, start_idx, dual_sample, all_sample in zip(alphas, start_samples, dual_samples, all_samples):
                # get the sample at time t
                sample_index_t = start_idx[index]
                sample_ts = carry_block.new(
                    daycount_fn(dual_sample.np[sample_index_t:, utils.RESET_INDEX_End_Day].reshape(1, -1) -
                                t_block[:, utils.TIME_GRID_MTM, np.newaxis]))

                weight_t = dual_sample.tn[sample_index_t:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
                sample_ft = weight_t * torch.exp(
                    torch.unsqueeze(carry_block, dim=1) * torch.unsqueeze(sample_ts, dim=2))
                M1 = torch.sum(sample_ft, dim=1)
                # realized average so far
                average = torch.sum(all_sample[:sample_index_t], dim=0)

                product_t = sample_ft * torch.exp(
                    torch.unsqueeze(sample_ts, dim=2) * torch.unsqueeze(vols * vols, dim=1))
                sum_t = F.pad(torch.cumsum(product_t[:, :-1], dim=1), [0, 0, 1, 0, 0, 0])
                M2 = torch.sum(sample_ft * (product_t + 2.0 * sum_t), dim=1)

                # trick to avoid nans in the gradients
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

    # potential cashflows
    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])
    # mtm in reporting currency
    mtm = torch.cat(mtm_list, dim=0)

    return mtm


def pv_energy_option(shared, time_grid, deal_data, nominal):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]

    # first precalc all past resets
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
    # need the tensor and numpy data
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

            # needed for vol lookup
            sample_block = daycount_fn(
                sample_t.np[:, utils.RESET_INDEX_Start_Day].reshape(1, -1)
                - t_block[:, utils.TIME_GRID_MTM, np.newaxis])

            M1 = torch.sum(sample_ft, dim=1)
            strike_bar = factor_dep['Strike'] - average

            moneyness = calc_moneyness(
                strike_bar.unsqueeze(0), None, future_resets * forwardfx,
                deal_data, use_forward=True)

            # get the vol per sample tau - might need to generalize this if we start modelling vols through time
            vols = torch.stack([utils.calc_time_grid_vol_rate(
                factor_dep['Volatility'], mon, s_tau, shared) for mon, s_tau in zip(moneyness, sample_block)])

            vol2 = vols * vols

            # Note - need to allow for compo deals
            if 'FXCompoVol' in factor_dep:
                fx_vols = torch.stack(
                    [utils.calc_time_grid_vol_rate(factor_dep['FXCompoVol'], None, sb, shared)
                     for sb in sample_block])
                vol2 += fx_vols * fx_vols + 2.0 * fx_vols * vols * utils.implied_correlation(
                    factor_dep['ImpliedCorrelation'])

            product_t = sample_ft * torch.exp(
                sample_ft.new(np.expand_dims(sample_block, axis=2)) * vol2)

            # do an exclusive cumsum on axis=1
            sum_t = F.pad(torch.cumsum(product_t[:, :-1], dim=1), [0, 0, 1, 0, 0, 0])
            M2 = torch.sum(sample_ft * (product_t + 2.0 * sum_t), dim=1)
            MM = torch.log(M2) - 2.0 * torch.log(M1)
            # trick to allow the gradients to be defined
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

    # potential cashflows
    cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])
    # mtm in reporting currency
    mtm = fx_rep * torch.cat(mtm_list, dim=0)

    return mtm


def pricer_float_cashflows(all_resets, t_cash, shared):
    margin = (t_cash[:, utils.CASHFLOW_INDEX_FloatMargin] * t_cash[:, utils.CASHFLOW_INDEX_Year_Frac])
    all_int = all_resets * t_cash[:, utils.CASHFLOW_INDEX_Year_Frac].reshape(1, -1, 1) + margin.reshape(1, -1, 1)

    return all_int, margin


def _pricer_cap_floor(all_resets, t_cash, factor_dep, expiries, tenor, call_or_put, shared, vol_expiries):
    strike = t_cash[:, utils.CASHFLOW_INDEX_Strike].reshape(1, -1, 1)
    expiry = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](expiries)
    vol_expiry = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](vol_expiries)
    digital_payoff = factor_dep['Digital_Payoff_Rate'] if 'Digital_Payoff_Rate' in factor_dep else 0.0
    dist, shf = factor_dep['VolSurface'][0][utils.FACTOR_INDEX_SubType]
    pricing_fn = utils.black_european_option if dist=='Lognormal' else utils.bachelier_european_option

    if digital_payoff and factor_dep.get('Digital_Spread', 0.0) > 0:
        # call/put spread replication: vols are re-queried at the two spread strikes so the
        # smile is automatically picked up — no separate dvol/dK correction needed.
        eps = factor_dep['Digital_Spread']
        strike_lo = strike * (1.0 - eps)
        strike_hi = strike * (1.0 + eps)
        mn_lo = -100.0 * (all_resets - strike_lo)
        mn_hi = -100.0 * (all_resets - strike_hi)
        vols_lo = utils.calc_tenor_cap_time_grid_vol_rate(
            factor_dep['VolSurface'], mn_lo, vol_expiry, tenor, shared)
        vols_hi = utils.calc_tenor_cap_time_grid_vol_rate(
            factor_dep['VolSurface'], mn_hi, vol_expiry, tenor, shared)
        payoff = digital_payoff * (
            pricing_fn(all_resets, strike_lo, vols_lo, expiry, 1.0, call_or_put, shared, shift=shf.amount) -
            pricing_fn(all_resets, strike_hi, vols_hi, expiry, 1.0, call_or_put, shared, shift=shf.amount)
        ) / (2.0 * eps * strike)
    else:
        mn_option = -100.0 * (all_resets - strike)
        vols = utils.calc_tenor_cap_time_grid_vol_rate(
            factor_dep['VolSurface'], mn_option, vol_expiry, tenor, shared)
        payoff = pricing_fn(
            all_resets, strike, vols, expiry, 1.0, call_or_put, shared, 
            cash_payoff=digital_payoff, shift=shf.amount)

    all_int = t_cash[:, utils.CASHFLOW_INDEX_Year_Frac].reshape(1, -1, 1) * payoff
    margin = shared.one.new_zeros(len(t_cash))

    return all_int, margin


def pricer_cap(all_resets, t_cash, factor_dep, expiries, tenor, shared, vol_expiries):
    return _pricer_cap_floor(all_resets, t_cash, factor_dep, expiries, tenor, 1.0, shared, vol_expiries)


def pricer_floor(all_resets, t_cash, factor_dep, expiries, tenor, shared, vol_expiries):
    return _pricer_cap_floor(all_resets, t_cash, factor_dep, expiries, tenor, -1.0, shared, vol_expiries)


def pv_float_cashflow_list(shared: utils.Calculation_State, time_grid: utils.TimeGrid, deal_data: utils.DealDataType,
                           cashflow_pricer, mtm_currency=None, settle_cash=True):
    """The floating leg. Its one invariant is CANONICAL FORM: by the time the fold below runs,
    every cashflow row carries exactly ONE effective reset, so a row is a rate and an accrual.

    Three reductions restore that form, and each lives somewhere different:

    - **Averaging** happens at COMPILE. `make_float_cashflows` writes `Weight = 1/n` on each of a
      row's `n` resets and `get_simulated_resets` applies `Weight / Accrual`, so several fixings
      arrive as one already-averaged rate. The weight is baked in, which is why it must never reach
      a path that compounds - a `1/n` weighted reset compounded geometrically compounds at `1/n` of
      its rate.
    - **OIS** happens HERE, and the switch is a SHAPE difference:
      `all_resets.shape[1] != reset_cashflows.np.shape[0]` means a row owns several resets, which
      `compress_no_compounding(groupsize=-1)` is what produces. Those are compounded geometrically
      by `segment_reduce` back to one rate per row.
    - **Method compounding** is the ordered ragged fold at the end, and it is a different axis: it
      is several cashflow ROWS sharing one payment date (`cash_counts > 1`), each already canonical,
      accumulated in order with the margin placed by convention.

    The three margin conventions differ only in where the spread sits. Writing `int` for
    `(rate + margin) * accrual`, `mrg` for `margin * accrual` and `N` for the nominal:

    | method | fold |
    | --- | --- |
    | `Include_Margin` | `total + int * (total + N)` - everything compounds |
    | `Flat` | `total + int * N + total * (int - mrg)` |
    | `Exclude_Margin` | `comp = comp + (int - mrg) * (comp + N)`; `simple += mrg * N`; pays `comp + simple` |

    Per the convention spec: under `Flat`, cashflow i pays `P.I_i.(1+J_{i+1})...(1+J_n)` - the FULL
    interest enters the compounding pot and the pot grows at rate-only `J = I - m.alpha`. Under
    `Exclude_Margin` it pays `P.m_i.alpha_i + P.J_i.(1+J_{i+1})...(1+J_n)` - each period's margin
    is SIMPLE on the nominal and only the rate part compounds. The two therefore differ by
    `sum_i m_i.alpha_i.N.(prod_{j>i}(1+J_j) - 1)` at any positive spread, and Exclude cannot be a
    single-accumulator fold: a margin lump inside `total` would earn `(1+J)` in every later step,
    which is exactly `Flat` - the trap a one-line restatement of the convention falls into.
    """
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    # first precalc all past resets
    resets = factor_dep['Cashflows'].Resets

    if mtm_currency:
        # precalc the FX forwards
        sim_fx_forward = utils.calc_fx_forward(
            mtm_currency, factor_dep['Currency'], factor_dep['Cashflows'].FXResets[:, utils.RESET_INDEX_Reset_Day],
            factor_dep['Cashflows'].FXResets[:, :utils.RESET_INDEX_Scenario + 1], shared, only_diag=True)

        known_fx = factor_dep['Cashflows'].known_resets(
            shared.simulation_batch, utils.CASHFLOW_INDEX_FXResetValue, utils.CASHFLOW_INDEX_FXResetDate)

        # fetch fx rates - note that there is a slight difference between this and the spot fx rate
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

        # cashflows is a dual representation
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

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        # do we need to split the forward block further?
        forward_blocks = forward_block.split_counts(
            reset_count, shared) if len(reset_count) > 1 else [forward_block]

        # empty list for payments
        payments = []

        for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
            time_slice = time_block[time_ofs:time_ofs + size].reshape(-1, 1)
            future_starts = reset_block.np[offset:, utils.RESET_INDEX_Start_Day] - time_slice
            future_ends = reset_block.np[offset:, utils.RESET_INDEX_End_Day] - time_slice
            future_weights = (reset_block.tn[offset:, utils.RESET_INDEX_Weight]
                              / reset_block.tn[offset:, utils.RESET_INDEX_Accrual]).reshape(1, -1, 1)
            future_resets = torch.expm1(forward_rates.gather_weighted_curve(
                shared, future_ends, future_starts)) * future_weights

            # now deal with past resets
            all_resets = torch.cat(
                [old_resets[reset_offset:reset_offset + offset].expand(size, -1, -1), future_resets], dim=1)

            # handle cashflows that don't pay interest (e.g. bullets)
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
                    # get the past fx rates - only works if not forward starting
                    fx_end_index = fx_past_index+1
                else:
                    # forward starting
                    fx_end_index = fx_past_index

                past_fx_resets = old_fx_rates[fx_past_index: fx_end_index].expand(size, -1, -1)

                # now deal with fx rates
                future_fx_resets = utils.calc_fx_forward(
                    mtm_currency, factor_dep['Currency'],
                    cashflows.np[past_fx_resets.shape[1]:, utils.CASHFLOW_INDEX_FXResetDate],
                    discount_block.time_grid[time_ofs:time_ofs + size], shared)

                all_fx_resets = torch.cat([past_fx_resets, future_fx_resets], dim=1)

                # calculate the Nominal in the correct currency
                Pi = all_fx_resets * cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1)
                Pi_1 = F.pad(Pi[:, 1:, :], [0, 0, 0, 1, 0, 0])

            time_ofs += size

            if all_resets.shape[1] != reset_cashflows.np.shape[0]:
                # OIS: a row owning several resets is the SHAPE the compile side produces on
                # purpose - see docs_src/developer/quote_sensitivities.md#curve-contracts
                # check if the resets need to be averaged (compounded) before being applied (i.e. OIS)
                reset_per_cashflows = factor_dep['Cashflows'].offsets[start_index[index]:, 0]
                accrual = reset_block.tn[:, utils.RESET_INDEX_Accrual]  # should align with all_resets dim=1
                # note that if we want to do average rate (not compounding) - we just need to drop log1p and expm1
                log_rt = torch.log1p(all_resets * accrual.view(1, -1, 1))
                reset_split = torch.as_tensor(
                    reset_per_cashflows[reset_per_cashflows > 0], device=shared.one.device, dtype=torch.long)
                lengths = reset_split.unsqueeze(0).expand(log_rt.shape[0], -1)
                sum_log = torch.segment_reduce(log_rt, reduce="sum", lengths=lengths, axis=1)
                sum_acc = torch.segment_reduce(accrual, reduce="sum", lengths=reset_split, axis=0)
                all_resets = torch.expm1(sum_log)/sum_acc.view(1, -1, 1)

            # check if we need extra information to price caps or floors
            if cashflow_pricer in [pricer_cap, pricer_floor]:
                expiries = cashflows.np[:, utils.CASHFLOW_INDEX_Start_Day] - time_slice
                if factor_dep['AveragingMethod']=='Post_Aggregation':
                    # adjust the expiry date to adjust the vol
                    t_k = cashflows.np[:, utils.CASHFLOW_INDEX_Start_Day] - time_slice
                    t_kp1 = cashflows.np[:, utils.CASHFLOW_INDEX_End_Day] - time_slice
                    eff_expiries = t_kp1 - (2/3)*(t_kp1 - t_k)
                else:
                    eff_expiries = expiries
                # note that the tenor (Year Frac) is averaged
                # all the cashflows are supposed to have the same year frac
                # (but practically not - should be ok to do this)
                tenor = cashflows.np[:, utils.CASHFLOW_INDEX_Year_Frac].mean()
                all_int, all_margin = cashflow_pricer(
                    all_resets, reset_cashflows.tn, factor_dep, eff_expiries, tenor, shared, expiries)
            else:
                all_int, all_margin = cashflow_pricer(all_resets, reset_cashflows.tn, shared)

            # handle the common case of no compounding or OIS compounding
            if mtm_currency is None and factor_dep['CompoundingMethod'] in ['None', 'OIS']:
                interest = all_int * reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1)
                if (cash_counts == 1).all():
                    total = interest
                else:
                    split_interest = torch.split(interest, tuple(cash_counts), dim=1)
                    total = torch.stack([i.sum(dim=1) for i in split_interest], dim=1)
            else:
                # check if there are a different number of resets per cashflow
                if cash_counts.min() != cash_counts.max():
                    interest = F.pad(all_int, [0, 0, 0, 1, 0, 0])
                    # nom = torch.ones_like(reset_cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal])
                    # nominal = F.pad(nom, [0, 1])
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
                        # spec: cashflow i pays P.I_i.(1+J_{i+1})...(1+J_n) - FULL interest enters
                        # the pot, the pot compounds at rate-only J = I - m.alpha
                        total = total + (int_i * nominal[offst].reshape(1, -1, 1)) + total * (
                                int_i - margin[offst].reshape(1, -1, 1))
                    elif factor_dep['CompoundingMethod'] == 'Exclude_Margin':
                        # spec: cashflow i pays P.m_i.alpha_i + P.J_i.(1+J_{i+1})...(1+J_n) - only
                        # the rate part compounds; the margin of each period is simple on nominal
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

        # now finish the payments
        all_payments = torch.cat(payments, dim=0) if len(payments) > 1 else payments[0]

        # settle any cashflows
        if settle_cash:
            cash_settle(shared, factor_dep['SettleCurrency'],
                        np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), all_payments[-1][0])
        # add it to the list
        mtm_list.append(torch.sum(all_payments * discount_rates, dim=1))

    return torch.cat(mtm_list, dim=0)


def pv_fixed_cashflows(shared, time_grid, deal_data, ignore_fixed_rate=False, settle_cash=True):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

    schedule = factor_dep['Cashflows']
    cash_start_idx = schedule.get_cashflow_start_index(deal_time)
    settlement_amt = factor_dep.get('Settlement_Amount', 0.0)
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    # there could be a repo curve if this is settled in future
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
        # payment times
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        if 'SettleFX' in factor_dep:
            discountfx = utils.calc_fx_forward(
                factor_dep['SettleFX'], factor_dep['Currency'],
                cash_pmts, discount_block.time_grid, shared)
        else:
            discountfx = shared.one

        # is this a forward?
        if settlement_amt:
            settlement_days = (factor_dep['Settlement_Date'] - time_block).clip(min=0).reshape(-1, 1)
            repo_discount = utils.calc_discount_rate(repo_block, settlement_days, shared)
            settlement_discount = torch.squeeze(
                utils.calc_discount_rate(settle_block, settlement_days, shared), dim=1)
            # For forward-style deals, carry the post-settlement indexed bond value to the
            # forward date on repo, then discount that settlement value on the bond curve.
            discount_rates = discount_rates / repo_discount
        else:
            settlement_discount = shared.one

        # empty list for payments
        all_int = (1.0 if ignore_fixed_rate else cashflows.tn[:, utils.CASHFLOW_INDEX_FixedRate]
                   ) * cashflows.tn[:, utils.CASHFLOW_INDEX_Year_Frac]

        # Built from the schedule's tensor half and nothing else, so it lives exactly as long as
        # that half - not in `t_Buffer`, which clears per batch. Keyed by the slice it sums, so a
        # deal priced on two grids (the inner-MC fork windows one) cannot read the other's.
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

        # add to the mtm
        mtm_list.append(
            (torch.sum(
                discountfx * discount_rates * schedule.derived[payment_key].reshape(1, -1, 1), dim=1) -
             settlement_amt) * settlement_discount)

        # settle any cashflows
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
    def calc_index(schedule, sim_schedule):
        weight = schedule.tn[:, utils.RESET_INDEX_Weight].reshape(1, -1, 1)
        dates = schedule.np[np.newaxis, :, utils.RESET_INDEX_Reset_Day] - \
                last_pub_block[:, np.newaxis, utils.RESET_INDEX_Reset_Day]
        index_t = torch.unsqueeze(last_index_block, dim=1) / utils.calc_discount_rate(
            forecast_block, dates, shared)

        # split if necessary
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
        # cashflows is a dual representation (numpy and tensor) of the same cashflows
        cashflows = factor_dep['Cashflows'].dual(start_index[index])

        cash_pmts, cash_index, cash_counts = np.unique(
            cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

        # payment times            
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)

        all_base_index_vals = get_index_val(
            cashflows[:, utils.CASHFLOW_INDEX_BaseReference], base_resets.dual(),
            all_base_resets, resets_per_cf, start_index[index])
        all_final_index_vals = get_index_val(
            cashflows[:, utils.CASHFLOW_INDEX_FinalReference], final_resets.dual(),
            all_final_resets, resets_per_cf, start_index[index])

        # empty list for payments
        interest = (cashflows.tn[:, utils.CASHFLOW_INDEX_FixedRate] *
                    cashflows.tn[:, utils.CASHFLOW_INDEX_Year_Frac]).reshape(1, -1, 1)
        growth = cashflows.tn[:, utils.CASHFLOW_INDEX_FixedAmt].reshape(1, -1, 1) * (
                all_final_index_vals / all_base_index_vals)
        payment_all = cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1) * growth * interest

        # reduce if any counts are duplicated
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
            # For forward-style deals, carry the post-settlement indexed bond value to the
            # forward date on repo, then discount that settlement value on the bond curve.
            mtm = asset_pv * settlement_discount / repo_discount
        else:
            mtm = asset_pv

        # add it to the list
        mtm_list.append(mtm)

        # settle any cashflows
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
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)

    # first precalc all past resets
    resets = factor_dep['Cashflows'].Resets
    known_resets = resets.known_resets(shared.simulation_batch)
    sim_resets = resets.schedule[
        (resets.schedule[:, utils.RESET_INDEX_Scenario] > -1) &
        (resets.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]
    all_fx_spot = utils.calc_fx_cross(
        factor_dep['ForwardFX'][0], factor_dep['CashFX'][0], sim_resets, shared)

    if factor_dep['ForwardPrice'] is None:
        # reconstruct the forward curve from the simulated components - spot x exp((carry + repo) tau);
        # past resets sample F(reset_day, fixing_day) off the same curve (F(t,t) = spot)
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

        # discount rates
        discount_rates = utils.calc_discount_rate(discount_block, future_pmts, shared)
        # fx adjustment when payoff currency differs from deal currency
        discountfx = utils.calc_fx_forward(
            factor_dep['CashFX'], factor_dep['Currency'],
            cash_pmts, discount_block.time_grid, shared)

        # we need to split the forward block further
        forward_blocks = forward_block.split_counts(
            reset_count, shared) if len(reset_count) > 1 else [forward_block]

        # empty list for payments
        payments = []

        for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):
            # past resets
            past_resets = torch.unsqueeze(
                old_resets[reset_offset:reset_offset + offset], dim=0).expand(size, -1, -1)

            # future resets
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

            # now we can price the cashflows
            payment = cashflows.tn[:, utils.CASHFLOW_INDEX_Nominal] * (
                    cashflows.tn[:, utils.CASHFLOW_INDEX_Start_Mult] * payoff +
                    cashflows.tn[:, utils.CASHFLOW_INDEX_FloatMargin])

            payments.append(payment)

        # now finish the payments
        all_payments = torch.cat(payments, dim=0)

        # settle any cashflows
        cash_settle(shared, factor_dep['SettleCurrency'],
                    np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), all_payments[-1][0])
        # add it to the list
        mtm_list.append(torch.sum(discountfx * all_payments * discount_rates, dim=1))

    return torch.cat(mtm_list, dim=0)


def pv_credit_cashflows(shared, time_grid, deal_data, return_par_spread=False):
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
        # get the duel cashflow at the correct index
        cashflows = factor_dep['Cashflows'].dual(start_index[index])
        cash_pmts, cash_index = np.unique(cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True)
        cash_sts = np.unique(cashflows.np[:, utils.CASHFLOW_INDEX_Start_Day])

        # payment times            
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

        # note the minus sign here
        premium = -(interest[cash_index] * cashflows.tn[cash_index, utils.CASHFLOW_INDEX_Nominal]).reshape(1, -1, 1)
        pv_premium = premium * (survival_T + adjustment * marginal_PD) * Dt_T
        pv_credit = (1.0 - factor_dep['Recovery_Rate'].recovery_rate()) * cashflows.tn[
            cash_index, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1) * 0.5 * (
                            Dt_T + Dt_Tm1) * marginal_PD

        if return_par_spread:
            value = pv_credit.sum(1)/-pv_premium.sum(1)
        else:
            # settle any cashflows
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
    q_j = Φ((Φ^{-1}(p_j) - √ρ z)/√(1-ρ)), so P(k = m | z) = P0 e_m(r) with P0 = ∏(1-q_j) and e_m
    the m-th elementary symmetric polynomial of the odds r_j = q_j/(1-q_j). Only e_0..e_{n-k0-1}
    can carry a non-zero rate, so the recurrence stops there - it is O(N (n-k0)) and, unlike the
    closed forms e_2 = (S_1^2 - S_2)/2, free of cancellation in float32.

    The z integral is Gauss-Hermite on the caller's (`z`, `w`) nodes, which is EXACT in z at
    ρ = 0 and progressively harder as ρ → 1 - see `Quadrature_Points`.

    Returns:
      E_rate: (T, Pm, S) = E[ rate(horizon) ] at each hazard sample point.
    """
    eps = torch.finfo(shared.one.dtype).eps

    Tsteps, Pm, Sscen, Nnames = p_default.shape

    # Clamp for stability
    p = p_default.clamp(eps, 1.0 - eps)

    # a = Φ^{-1}(p): (T,Pm,S,N)
    a = utils.norm_icdf(p)

    sr = np.sqrt(rho)
    s1 = np.sqrt(1.0 - rho)

    # Broadcast over GH nodes: (G, T, Pm, S, N)
    z_ = z.view(-1, 1, 1, 1, 1)
    a_ = a.unsqueeze(0)

    q = utils.norm_cdf((a_ - sr * z_) / s1).clamp(eps, 1.0 - eps)  # (G,T,Pm,S,N)

    # P0 = ∏(1-q): (G,T,Pm,S)
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

    # Integrate over z with GH weights => (T,Pm,S)
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
    are contiguous by construction and that one start plus the pay days ARE all the boundaries.
    Samples before the valuation date give a negative hazard time, clipped to zero survival
    weight: the already-accrued part of the running coupon is earned at the full rate.
    """
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    accrual_daycount_fn = factor_dep['Accrual_Daycount']
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    cash_start_idx = factor_dep['Cashflows'].get_cashflow_start_index(deal_time)
    rho = factor_dep['Correlation']
    discounts = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
    # this is the index proxy
    surv = utils.calc_time_grid_curve_rate(factor_dep['Name'], deal_time, shared)
    surv_base = utils.calc_time_grid_curve_rate(factor_dep['Name'], np.zeros((1, 3)), shared)
    names = [utils.calc_time_grid_curve_rate(name, deal_time, shared) for name in factor_dep['Names']]
    # calculate the gassian weights
    x, w = np.polynomial.hermite.hermgauss(factor_dep['Quadrature_Points'])  # for ∫ e^{-x^2} f(x) dx
    z = np.sqrt(2.0) * x  # transform to standard normal
    ww = w / np.sqrt(np.pi)  # weights for φ(z)
    z_t = shared.one.new(z)
    w_t = shared.one.new(ww)
    # cashflow start dates
    start_index, counts = np.unique(cash_start_idx, return_counts=True)

    for index, (discount_block, surv_block) in enumerate(
            utils.split_counts([discounts, surv], counts, shared)):
        # get the duel cashflow at the correct index
        cashflows = factor_dep['Cashflows'].dual(start_index[index])
        cash_pmts, cash_index = np.unique(cashflows.np[:, utils.CASHFLOW_INDEX_Pay_Day], return_index=True)
        # payment times
        time_block = discount_block.time_grid[:, utils.TIME_GRID_MTM]
        future_pmts = cash_pmts.reshape(1, -1) - time_block.reshape(-1, 1)
        # samples for estimating default - the window opens at the first unpaid period's start
        samples_points = np.r_[cashflows.np[0, utils.CASHFLOW_INDEX_Start_Day], cash_pmts].astype(np.int64)
        # note hazard_t can contain negative times - represents accrued payments
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
        # allow dt to be negative (to account for accrued coupons)
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

        # use trapazoid rule to calc E[rate] over the horizon
        trap = 0.5 * (E_rate[:, :-1, :] + E_rate[:, 1:, :])
        trapz = trap * dt # (T,Pm-1,S)
        lengths = torch.as_tensor(
            [np.r_[0,hs.searchsorted(x, side='right')-1] for hs,x in zip(hazard_samples, future_pmts)],
            device=shared.one.device)
        expected_coupons = torch.segment_reduce(trapz, reduce="sum", lengths=lengths.diff(), axis=1)
        # per-period nominal carries Principal, Buy_Sell and any Amortisation; the minus is
        # pv_credit_cashflows' premium convention - the buyer PAYS the coupon
        premium = -expected_coupons * cashflows.tn[cash_index, utils.CASHFLOW_INDEX_Nominal].reshape(1, -1, 1)

        # settle any cashflows
        cash_settle(shared, factor_dep['SettleCurrency'],
                    np.searchsorted(time_grid.mtm_time_grid, cash_pmts[0]), premium[-1, 0])

        mtm_list.append(torch.sum(premium * Dt_T, dim=1))

    return torch.cat(mtm_list, dim=0)


def pv_equity_cashflows(shared, time_grid, deal_data):
    mtm_list = []
    factor_dep = deal_data.Factor_dep
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    forward_settle_days = factor_dep['Bus_Ofs'][deal_data.Time_dep.deal_time_grid]
    eq_spot = utils.calc_time_grid_spot_rate(factor_dep['Equity'], deal_time, shared)
    cash = factor_dep['Flows']

    # needed for grouping 
    cash_start_idx = np.searchsorted(
        cash.schedule[:, utils.CASHFLOW_INDEX_Start_Day], deal_time[:, utils.TIME_GRID_MTM], side='right')
    cash_end_idx = np.searchsorted(
        cash.schedule[:, utils.CASHFLOW_INDEX_End_Day], deal_time[:, utils.TIME_GRID_MTM], side='right')
    cash_pay_idx = cash.get_cashflow_start_index(deal_time)

    # first precalc all past resets
    all_samples = []

    for samples in cash.Resets.split_groups(2):
        known_sample = samples.known_resets(shared.simulation_batch, include_today=True)
        sim_samples = samples.schedule[
            (samples.schedule[:, utils.RESET_INDEX_Value] == 0.0) &
            (samples.schedule[:, utils.RESET_INDEX_Reset_Day] <= deal_time[:, utils.TIME_GRID_MTM].max())]

        past_samples = utils.calc_time_grid_spot_rate(
            factor_dep['Equity'], sim_samples[:, :utils.RESET_INDEX_Scenario + 1], shared)

        # fetch all fixed resets
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

        # need equity forwards for start and end cashflows
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

            # settle cashflow if necessary
            cash_settle(shared, factor_dep['SettleCurrency'],
                        np.searchsorted(time_grid.mtm_time_grid, cashflow_pay[0][0]),
                        torch.sum(payment, dim=1)[0])

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

        # update the time block
        time_block_index += all_counts[index]
        # now finish the payments
        payments = torch.cat(payoffs, dim=1) if len(payoffs) > 1 else payoffs[0]
        mtm_list.append(torch.sum(payments * discount_rates, dim=1))

    return torch.cat(mtm_list, dim=0)


@utils.log_exception
def pv_fixed_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
    mtm = pv_fixed_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_energy_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
    mtm = pv_energy_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_float_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
    model = deal_data.Factor_dep.get('Model', pricer_float_cashflows)
    mtm = pv_float_cashflow_list(shared, time_grid, deal_data, model) * FX_rep

    return mtm


@utils.log_exception
def pv_index_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
    mtm = pv_index_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_cds_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
    mtm = pv_credit_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_credit_step_down_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
    mtm = pv_credit_step_down_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm


@utils.log_exception
def pv_equity_leg(shared, time_grid, deal_data):
    deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
    FX_rep = utils.calc_fx_cross(
        deal_data.Factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
    mtm = pv_equity_cashflows(shared, time_grid, deal_data) * FX_rep

    return mtm
