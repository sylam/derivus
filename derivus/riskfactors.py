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
import logging
import itertools
import numpy as np
import pandas as pd

from . import utils
from .schema import F
from scipy.interpolate import RectBivariateSpline
from scipy.special import ndtr

#: The methods `Factor1D.check_interpolation` implements - what a curve factor routed through
#: `Price Factor Interpolation` may be set to. Listed by the classes that opt in.
INTERPOLATION_METHODS = ('HermiteRT', 'Hermite', 'LinearRT', 'Linear')

# map the names of various factor interpolations to something simpler
factor_interp_map = {
    'CubicSplineCurveInterpolation': 'Hermite',
    'HermiteInterpolationCurveGetValue': 'Hermite',
    'LinearInterFlatExtrapCurveGetValue': 'Linear',
    'LogLinearCurveInterpolation': 'LinearRT',
    'CubicSplineOnXTimesYCurveInterpolation': 'HermiteRT',
    'HermiteRTInterpolationCurveGetValue': 'HermiteRT',
    'NaturalCubicSplineCurveInterLinearExtrap': 'Hermite',
    'LogNaturalCubicSplineCurveInterFlatExtrap': 'HermiteRT',
    'HermiteRT': 'HermiteRT',
    'Hermite': 'Hermite',
    'LinearRT': 'LinearRT',
    'Linear': 'Linear',
    'LinearExtrapolate': 'LinearExtrapolate'
}


class Factor0D(object):
    """Represents an instantaneous Rate (0D) risk factor"""

    def __init__(self, param):
        self.param = param
        self.delta = 0.0

    def bump(self, amount, relative=True):
        self.delta = self.param['Spot'] * amount if relative else amount

    def get_subtype(self):
        return None

    def get_delta(self):
        return self.delta

    def get_tenor_indices(self):
        '''
        Returns this factor's indices - always 0.0 for a spot rate. Whatever this returns is
        what the tenors parameter of current_value accepts.
        '''

        return np.array([[0.0]])

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param['Spot'] + self.delta])


class Factor1D(object):
    """Represents a risk factor with a term structure (1D)"""

    def __init__(self, param):
        """Sets up the tenor grid and the interpolation scheme.

        One scheme over the whole curve, unless `Near_Interpolation`, `Near_Date` and a base_date
        are all present: then two are stacked, split on the near index (the day count accrual to
        `Near_Date` clipped to the tenor range, right-inserted less one). Each is fitted on its own
        slice - the near leg on `[:near_idx_end+1]`, whose +1 keeps the near tenor in the near leg,
        and the far leg on `[near_idx_end:]`.
        """
        self.param = param
        self.tenors = self.get_tenor()
        self.base_date = self.param.get('base_date')
        self.delta = np.zeros_like(self.tenors)

        # not used unless we interpolate different parts of the curve differently
        near_interp = self.param.get('Near_Interpolation')  # e.g. 'LinearRT'
        if near_interp and self.param.get('Near_Date') and self.base_date is not None:
            near_interp = factor_interp_map.get(near_interp, 'Linear')
            near_date = self.param['Near_Date']  # implement parse
            near_tenor = np.clip(
                utils.get_day_count_accrual(
                self.base_date, (near_date - self.base_date).days, utils.get_day_count(self.param['Day_Count'])),
                self.tenors.min(), self.tenors.max()
            )
            near_idx_end = self.tenors.searchsorted(near_tenor, side='right') - 1
            # stack 2 interpolation types split on the near index (note we add 1 to the near tenor)
            self.interpolation = [(0, near_idx_end, self.check_interpolation(
                near_interp, self.tenors[:near_idx_end+1],
                self.param['Curve'].array[:near_idx_end+1, 1])),
                (near_idx_end, self.tenors.size-1, self.check_interpolation(
                    self.param.get('Interpolation'), self.tenors[near_idx_end:],
                    self.param['Curve'].array[near_idx_end:, 1]))
            ]
        else:
            # regular interpolation type
            self.interpolation = [self.check_interpolation(
                self.param.get('Interpolation'), self.tenors,
                self.param.get('Curve', utils.Curve([], [(0.0, 0.0)])).array[:, 1])]

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        if self.param['Curve'] is None or not isinstance(self.param['Curve'], utils.Curve):
            self.param['Curve'] = utils.Curve([], [(0.0, 0.0)])

        # make sure there are no duplicate tenors    
        tenors = np.unique(self.param['Curve'].array[:, 0])
        rates = np.interp(tenors, *self.param['Curve'].array.T)
        self.param['Curve'].array = np.vstack((tenors, rates)).T
        return self.param['Curve'].array[:, 0]

    def get_tenor_indices(self):
        '''
        Returns this factor's indices - for a curve, the tenor points themselves. Whatever this
        returns is what the tenors parameter of current_value accepts.
        '''
        return self.tenors.reshape(-1, 1)

    def bump(self, amount, relative=True):
        self.delta = self.param['Curve'].array[:, 1] * amount if relative else np.ones_like(self.tenors) * amount

    def get_delta(self):
        return self.delta.mean()

    def get_subtype(self):
        return self.param.get('Sub_Type')

    def check_interpolation(self, interpolation, tenors, rates):
        if interpolation == 'HermiteRT':
            g, c = utils.hermite_interpolation(tenors, rates * tenors)
            return 'HermiteRT', g, c
        elif interpolation == 'Hermite':
            g, c = utils.hermite_interpolation(tenors, rates)
            return 'Hermite', g, c
        elif interpolation == 'LinearRT':
            return ('LinearRT',)
        elif interpolation == 'LinearExtrapolate':
            return ('LinearExtrapolate',)
        else:
            return ('Linear',)

    @staticmethod
    def interpolate(tenors, tenor_segment, values, interp):
        if interp[0] in ['HermiteRT', 'Hermite']:
            index = np.searchsorted(tenor_segment, tenors, side='right') - 1
            index_next = np.clip(index + 1, 0, tenor_segment.size - 1)
            dt = np.clip(tenor_segment[index_next] - tenor_segment[index], 1 / 365.0, np.inf)
            m = np.clip((tenors - tenor_segment[index]) / dt, 0.0, 1.0)
            g, c = interp[1:]
            rate, denom = (values * tenor_segment, tenors) if interp[0] == 'HermiteRT' else (
                values, 1.0)
            val = rate[index] * (1.0 - m) + m * rate[index_next] + m * (
                    1.0 - m) * g[index] + m * m * (1.0 - m) * c[index]
            return val / denom
        elif interp[0] == 'LinearRT':
            return np.interp(tenors, tenor_segment, tenor_segment * values)/tenors
        elif interp[0] == 'LinearExtrapolate' and tenor_segment.size > 1:
            val = np.interp(tenors, tenor_segment, values)
            s0 = (values[1] - values[0]) / (tenor_segment[1] - tenor_segment[0])
            s1 = (values[-1] - values[-2]) / (tenor_segment[-1] - tenor_segment[-2])
            val = np.where(tenors < tenor_segment[0], values[0] + s0 * (tenors - tenor_segment[0]), val)
            return np.where(tenors > tenor_segment[-1], values[-1] + s1 * (tenors - tenor_segment[-1]), val)
        else:
            return np.interp(tenors, tenor_segment, values)

    def current_value(self, tenor_index=None, offset=0.0):
        """Returns the value of the rate at each tenor point (if set) else returns what's
        stored in the Curve parameter"""
        bumped_val = self.param['Curve'].array[:, 1] + self.delta
        # get the tenors - make sure it's in range
        tenors = ((np.array(tenor_index) if tenor_index is not None else self.tenors) + offset).clip(
            self.tenors.min(), self.tenors.max())
        values = []
        nseg = len(self.interpolation)
        if nseg>1:
            for k, interp in enumerate(self.interpolation):
                start_index, end_index, interp_params = interp
                end_mask = (tenors <= self.tenors[end_index]) if k==nseg-1 else (tenors < self.tenors[end_index])
                subsegment = tenors[(tenors >= self.tenors[start_index]) & end_mask]
                sub_tenor = self.tenors[start_index:end_index+1]
                sub_values = bumped_val[start_index:end_index+1]
                values.append(Factor1D.interpolate(subsegment, sub_tenor, sub_values, interp_params))

            return np.concatenate(values)
        else:
            return Factor1D.interpolate(tenors, self.tenors, bumped_val, self.interpolation[0])


class Factor2D(object):
    """Represents a risk factor that's a surface (2D) - Currently this is only vol surfaces"""
    svi_params = ['ATM_Ref', 'a', 'b', 'rho', 'm', 'sigma']
    skew_params = ['ATM_Vol', 'ATM_Ref', 's', 'L', 'R', 'C', 'D', 'lam', 'rho']
    #: The log-moneyness nodes every Malz x-grid refines FROM.
    malz_seed_grid = np.array([-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5], dtype=float)
    #: The vol error the refinement drives each midpoint below. `FXVolSurfaceParameters` lets a
    #: block override it and falls back to THIS, so the two cannot be different numbers.
    malz_tol = 1e-4

    def __init__(self, param):
        # default empty surfaces to 1%
        if 'Surface' not in param or not param['Surface'].array.any():
            param['Surface'] = utils.Curve([], [[0, 0, 0.01]])

        self.param = param
        self.flat = None
        self.index_map = {}
        self.update()

    @staticmethod
    def get_day_count():
        """hardcode the daycount for dividend rates to act/365"""
        return utils.DAYCOUNT_ACT365

    def get_subtype(self):
        return (self.param.get('Surface_Type', 'Explicit'),
                self.param.get('Moneyness_Rule', 'Sticky_Moneyness'))

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attributes"""
        params = self.svi_params if self.get_subtype()[0] == 'SVI' else self.skew_params
        return np.unique([self.param[x].array[:, 0] for x in params])

    def update(self):
        self.expiry = self.get_expiry()
        if self.solves_delta_surface():
            # a block carrying deltas is UNSOLVED: solve here, on a grid refined against these
            # numbers. One bootstrapped by `FXVolSurfaceParameters` arrives solved on its pinned
            # grid and falls straight through
            skews = self.malz_skews(self.param['Delta_Surface'].array, self.expiry)
            self.param['Surface'] = utils.Curve([], self.malz_surface(skews, self.malz_grid(skews)))

        self.moneyness = self.get_moneyness()
        self.vols = self.get_vols()
        self.tenor_ofs = np.array([len(x) for x in self.index_map.values()])

    def get_moneyness(self):
        """Gets the moneyness points stored in the Surface attribute"""
        return np.unique(self.param['Surface'].array[:, 0])

    def get_expiry(self):
        """Gets the expiry points stored in the Surface attribute"""
        # an UNSOLVED Malz block carries its expiries on the delta surface; a solved one on the
        # log-moneyness Surface, like every other subtype
        return np.unique(self.param['Delta_Surface' if self.solves_delta_surface()
                                    else 'Surface'].array[:, 1])

    def solves_delta_surface(self):
        """Whether this block still has to run the Malz solver - a Malz surface CARRYING deltas.

        `Surface_Type` says two things and only one is about the solver: it names the moneyness
        convention the engine reads at (log(F/K), term-interpolated in total variance), and it says
        a delta smile is the form the block was authored in. A block bootstrapped by
        `FXVolSurfaceParameters` is the first without the second - it carries the solved
        log-moneyness `Surface` on a pinned x-grid, and re-solving would move it on every tick.
        """
        deltas = self.param.get('Delta_Surface')
        return self.get_subtype()[0] == 'Malz' and deltas is not None and deltas.array.any()

    @classmethod
    def malz_skews(cls, delta_surface, expiries):
        """`{T: skew}` - one prepared wing pair per expiry of a (delta, expiry, vol) surface.

        Vols are clipped to 1e-4 so a zero or negative quote cannot divide by zero downstream.
        """
        return {T: cls.malz_skew(delta_surface[delta_surface[:, 1] == T][:, 0],
                                 delta_surface[delta_surface[:, 1] == T][:, 2].clip(min=1e-4), T)
                for T in expiries}

    @staticmethod
    def malz_skew(delta, vol, T):
        """One expiry's delta smile, split into the two wings `malz_sigma` interpolates over.

        THE DELTA CONVENTION: a premium-adjusted FORWARD delta ((K/F)N(d2) for a call), with an ATM
        quote at that convention's delta-neutral straddle, K = F exp(-sigma^2 T/2), hence
        |delta| = 0.5 exp(-sigma^2 T/2). So +-0.5 is a LABEL rather than a delta and is replaced
        here by the delta it stands for. A wing missing its ATM node gets it mirrored.

        A smile carrying both labels at different vols quotes two ATM numbers and the +0.5 one
        wins, setting `delta_atm` and hence where every other node sits; the -0.5 vol is still read
        as that wing's node.
        """
        d, v = np.asarray(delta, dtype=float), np.asarray(vol, dtype=float)
        order = np.argsort(d)
        d, v = d[order], v[order]

        atm = np.isclose(np.abs(d), 0.5)
        if not atm.any():
            raise ValueError('Malz skew missing ATM label (+-0.5).')
        sigma_atm = float(v[atm][-1])  # ascending d, so [-1] is the PREFERRED +0.5 label
        delta_atm = 0.5 * np.exp(-0.5 * sigma_atm * sigma_atm * T)
        d = np.where(atm, np.sign(d) * delta_atm, d)

        # both wings need the ATM node - a smile quoted on one side only is mirrored onto the other
        for side in (-1.0, 1.0):
            if not np.any(np.isclose(d, side * delta_atm)):
                d, v = np.append(d, side * delta_atm), np.append(v, sigma_atm)
        order = np.argsort(d)
        d, v = d[order], v[order]

        return {'d_put': d[d <= 0.0], 'v_put': v[d <= 0.0],
                'd_call': d[d >= 0.0], 'v_call': v[d >= 0.0],
                'sigma_atm': sigma_atm, 'delta_atm': delta_atm}

    @staticmethod
    def malz_delta(skew, T, x, iterations=64):
        """The delta each log-moneyness x = log(F/K) resolves to, as `(delta*, is_call, bracketed)`.

        The vol at x is the fixed point of sigma = skew(delta(sigma, x)): the wing is piecewise
        linear in delta and the delta of the strike x names depends on the vol looked up.
        x <= sigma_atm^2 T / 2 is the call wing, and each side is bracketed by its own extreme
        deltas - where the fixed point falls outside, the wing is CLAMPED to the nearer endpoint,
        which flat-extrapolates the smile beyond its widest quoted delta. Vectorised bisection.

        NOT DIFFERENTIABLE: a bisection's iterates are dyadic combinations of the bracket
        ENDPOINTS, so a tape through this loop reports the bracket's derivative rather than the
        root's. `FXVolSurfaceParameters.carried_sigma` is the derivative twin.
        """
        x = np.asarray(x, dtype=float)
        sqrt_t = np.sqrt(T)
        is_call = x <= 0.5 * skew['sigma_atm'] * skew['sigma_atm'] * T
        k_over_f = np.exp(-x)

        def residual(delta):
            sigma = np.where(is_call, np.interp(delta, skew['d_call'], skew['v_call']),
                             np.interp(delta, skew['d_put'], skew['v_put']))
            d2 = (x - 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
            return k_over_f * np.where(is_call, ndtr(d2), -ndtr(-d2)) - delta

        lo = np.where(is_call, max(0.0, skew['d_call'].min()), skew['d_put'].min())
        hi = np.where(is_call, skew['d_call'].max(), min(0.0, skew['d_put'].max()))
        f_lo, f_hi = residual(lo), residual(hi)
        clamped = np.where(np.abs(f_lo) < np.abs(f_hi), lo, hi)

        left, f_left, right = lo.copy(), f_lo.copy(), hi.copy()
        for _ in range(iterations):
            middle = 0.5 * (left + right)
            f_middle = residual(middle)
            below = f_left * f_middle <= 0.0
            right = np.where(below, middle, right)
            left = np.where(below, left, middle)
            f_left = np.where(below, f_left, f_middle)

        bracketed = f_lo * f_hi <= 0.0
        return np.where(bracketed, 0.5 * (left + right), clamped), is_call, bracketed

    @classmethod
    def malz_sigma(cls, skew, T, x, iterations=64):
        """The Malz vol at EVERY log-moneyness x = log(F/K) of a grid - the wing read at `delta*`.
        `malz_delta` is the solve; this is the piecewise linear lookup on the side it names."""
        delta_star, is_call, _ = cls.malz_delta(skew, T, x, iterations)
        return np.where(is_call, np.interp(delta_star, skew['d_call'], skew['v_call']),
                        np.interp(delta_star, skew['d_put'], skew['v_put']))

    @classmethod
    def malz_error(cls, skew, T, nodes):
        """The vol error interpolating between `nodes` makes at each interval's midpoint.

        Measured in TOTAL VARIANCE, which is what the pricing path interpolates a Malz surface in.
        It is the criterion `malz_grid` refines against, and the measure of how well a PINNED grid
        still resolves a smile that has ticked since the grid was built.
        """
        middles = 0.5 * (nodes[:-1] + nodes[1:])
        variance = cls.malz_sigma(skew, T, nodes) ** 2 * T
        return np.abs(np.sqrt(np.interp(middles, nodes, variance) / T) -
                      cls.malz_sigma(skew, T, middles))

    @classmethod
    def malz_grid(cls, skews, tol=None):
        """`{T: x nodes}` - the log-moneyness grid the smile is resolved to `tol` vol on.

        A midpoint the current nodes cannot interpolate to `tol` becomes a node. Refinement is
        COMPILE-TIME work and its grid is part of the plan. Each expiry refines its own grid - a
        one-week smile needs nodes a ten-year one does not - which the flat surface carries as
        ragged rows.
        """
        grid = {}
        for T, skew in skews.items():
            nodes = cls.malz_seed_grid.copy()
            while True:
                missed = cls.malz_error(skew, T, nodes) > (cls.malz_tol if tol is None else tol)
                if not missed.any():
                    break
                nodes = np.sort(np.append(nodes, 0.5 * (nodes[:-1] + nodes[1:])[missed]))
            grid[T] = nodes
        return grid

    @classmethod
    def malz_surface(cls, skews, grid):
        """`[[x, T, vol], ...]` - the smile evaluated on a GIVEN log-moneyness grid.

        Separate from `malz_grid` because the two run at different times: the grid is refined once
        when the plan is compiled, this runs again on every tick that moves the quotes.
        """
        return [[x, T, vol] for T, nodes in grid.items()
                for x, vol in zip(nodes, cls.malz_sigma(skews[T], T, nodes))]

    def get_vols(self):
        """Uses flat extrapolation along moneyness and then linear interpolation along expiry"""
        surface = self.param['Surface'].array
        # sorted by moneyness within expiry (lexsort takes its primary key last)
        self.sorted_vol = surface[np.lexsort((surface[:, 0], surface[:, 1]))]
        self.index_map.clear()
        for element in self.sorted_vol:
            self.index_map.setdefault(element[1], []).append(element[0])
        self.flat = self.sorted_vol[:, 2]
        return np.array(
            [np.interp(self.moneyness, surface[surface[:, 1] == x][:, 0], surface[surface[:, 1] == x][:, 2])
             for x in self.expiry])

    def get_all_tenors(self):
        return np.hstack((self.moneyness, self.expiry))

    def get_tenor_indices(self):
        '''
        Returns the sorted (moneyness, expiry) surface indices for an Explicit vol surface, else
        the SVI or Skew parameter tenors. Symmetric with current_value.
        '''
        if self.get_subtype()[0] == 'SVI':
            tau = self.get_tenor().reshape(-1, 1)
            return {x: tau for x in self.svi_params}
        elif self.get_subtype()[0] == 'Skew':
            tau = self.get_tenor().reshape(-1, 1)
            return {x: tau for x in self.skew_params}
        else:
            return self.sorted_vol[:, :2]

    def current_value(self, tenors=None, offset=0.0):
        """Returns the value of the vol surface"""
        if self.get_subtype()[0] == 'SVI':
            # "tenors" here is actually sampled moneyness for SVI surfaces - we already have tenors
            if tenors is not None:
                surf = []
                # loop over params and calculate the vols
                for t, atm_ref, a, b, rho, m, sigma in np.array(
                        [self.get_tenor()] + [self.param[x].array[:, 1] for x in self.svi_params]).T:
                    x = (tenors - m)
                    variance = a + b * (rho * x + np.sqrt(x ** 2 + sigma ** 2))
                    surf.extend([[m, t, v] for m, v in zip(tenors, np.sqrt(variance))])
                return utils.Curve([2, 'default'], surf)

            return {x: self.param[x].array[:, 1] for x in self.svi_params}

        elif self.get_subtype()[0] == 'Skew':
            if tenors is not None:
                surf = []
                # similar to SVI - this is a parameterized vol surface
                for t, atm_vol, atm_ref, s, L, R, C, D, lam, rho in np.array(
                        [self.get_tenor()] + [self.param[x].array[:, 1] for x in self.skew_params]).T:

                    # Left wing
                    s2LC = s + 2.0 * L * C
                    gamma = s2LC / (-2.0 * C * lam)
                    beta = s2LC * (1.0 + 1.0 / lam)
                    alpha = atm_vol + C * ((s - beta) + C * (L - gamma))

                    # Right wing
                    s2RD = s + 2.0 * R * D
                    gamma_r = s2RD / (-2.0 * D * rho)
                    beta_r = s2RD * (1.0 + 1.0 / rho)
                    alpha_r = atm_vol + D * ((s - beta_r) + D * (R - gamma_r))

                    regions = [
                        lambda x: np.ones_like(x) * (alpha + C * (beta * (1.0 + lam) + gamma * (1.0 + lam) ** 2 * C)) if lam else atm_vol + C * (s + L * C),
                        lambda x: alpha + x * (beta + gamma * x) if lam else atm_vol + C * (s + L * C),
                        lambda x: atm_vol + x * (s + L * x),
                        lambda x: atm_vol + x * (s + R * x),
                        lambda x: alpha_r + x * (beta_r + gamma_r * x) if rho else atm_vol + D * (s + R * D),
                        lambda x: np.ones_like(x) * (alpha_r + D * (
                                beta_r * (1.0 + rho) + gamma_r * (1.0 + rho) ** 2 * D)) if rho else atm_vol + D * (s + R * D)
                    ]

                    conditions = [
                        (tenors <= (1 + lam) * C),
                        ((1 + lam) * C < tenors) & (tenors <= C),
                        (C < tenors) & (tenors <= 0.0),
                        (0 < tenors) & (tenors <= D),
                        (D < tenors) & (tenors <= (1 + rho) * D),
                        ((1 + rho) * D < tenors)
                    ]

                    vol = np.piecewise(tenors, conditions, regions)
                    surf.extend([[m, t, v] for m, v in zip(tenors, vol)])

                return utils.Curve([2, 'default'], surf)
            return {x: self.param[x].array[:, 1] for x in self.skew_params}

        else:
            if tenors is not None and self.expiry.size > 1 and self.moneyness.size > 1:
                interpolator = RectBivariateSpline(self.expiry, self.moneyness, self.vols, kx=1, ky=1)
                return np.array([interpolator(y + offset, x)[0][0] for x, y in tenors])

            return self.flat


class Factor3D(object):
    """Represents a risk factor that's a space (3D) - Things like swaption volatility spaces"""
    MONEYNESS_INDEX = 0
    EXPIRY_INDEX = 1
    TENOR_INDEX = 2

    def __init__(self, param):
        self.param = param
        self.update()

    def update(self):
        self.moneyness = self.get_moneyness()
        self.expiry = self.get_expiry()
        self.tenor = self.get_tenor()
        self.vols = self.get_vols()
        self.tenor_ofs = np.array([0, self.moneyness.size, self.moneyness.size + self.expiry.size])

    def get_subtype(self):
        return (self.param.get('Distribution_Type', 'Lognormal'), self.param.get('Shift', utils.Percent(0)))

    def get_moneyness(self):
        """Gets the moneyness points stored in the Surface attribute"""
        return np.unique(self.param['Surface'].array[:, self.MONEYNESS_INDEX])

    def get_expiry(self):
        """Gets the expiry points stored in the Surface attribute"""
        return np.unique(self.param['Surface'].array[:, self.EXPIRY_INDEX])

    def get_tenor(self):
        """Gets the tenor points stored in the Surface attribute"""
        return np.unique(self.param['Surface'].array[:, self.TENOR_INDEX])

    def get_vols(self):
        """Uses flat extrapolation along moneyness and then linear interpolation along expiry"""
        vols = []
        for tenor in self.tenor:
            surface = self.param['Surface'].array[self.param['Surface'].array[:, self.TENOR_INDEX] == tenor]
            for x in self.expiry:
                exp_surface = surface[surface[:, self.EXPIRY_INDEX] == x]
                if not exp_surface.any():
                    expiries = np.unique(surface[:, self.EXPIRY_INDEX])
                    nearest_expiry = expiries[expiries.searchsorted(x).clip(0, expiries.size - 1)]
                    exp_surface = surface[surface[:, self.EXPIRY_INDEX] == nearest_expiry]
                sigma = exp_surface[:, 3]
                mns = exp_surface[:, self.MONEYNESS_INDEX]
                vol = np.interp(self.moneyness, mns, sigma)
                vols.append(vol)

        return np.array(vols)

    def get_all_tenors(self):
        return np.hstack((self.moneyness, self.expiry, self.tenor))

    def get_tenor_indices(self):
        return np.array(list(itertools.product(self.tenor, self.expiry, self.moneyness)))

    def current_value(self, tenors=None, offset=0.0):
        """
        Returns the value of the vol space. Symmetric with get_tenor_indices:
        current_value(get_tenor_indices()) is the list of corresponding vols.
        """
        if tenors is not None and self.expiry.size > 1 and self.moneyness.size > 1:
            interpolator = [RectBivariateSpline(
                self.expiry, self.moneyness, vol.reshape(self.expiry.size, -1), kx=1, ky=1)
                for vol in self.vols.reshape(self.tenor.size, -1)]
            interp_vols = []
            for tenor in tenors:
                index = np.clip(self.tenor.searchsorted(
                    tenor[self.TENOR_INDEX], side='right') - 1, 0, self.tenor.size - 1)
                index_p1 = np.clip(index + 1, 0, self.tenor.size - 1)
                val = np.interp(
                    tenor[self.TENOR_INDEX], self.tenor[[index, index_p1]],
                    np.dstack([interpolator[index](tenor[self.EXPIRY_INDEX], tenor[self.MONEYNESS_INDEX]),
                               interpolator[index_p1](tenor[self.EXPIRY_INDEX],
                                                      tenor[self.MONEYNESS_INDEX])]).flatten())
                interp_vols.append(val)
            return np.array(interp_vols)

        return self.vols.ravel()


class FxRate(Factor0D):
    """
    Represents the price of a currency relative to the base currency (snapped at end of day).
    """
    fields = [
        F('Domestic_Currency', 'Text', default=''),
        F('Interest_Rate', 'Text', default='', obj='Tuple',
          description='Associated interest rate curve name'),
        F('Spot', 'Float', default=0, bind='value', description='Spot rate in base currency')
    ]

    def __init__(self, param):
        super(FxRate, self).__init__(param)

    def get_repo_curve_name(self, default):
        return utils.check_rate_name(self.param['Interest_Rate'] if self.param['Interest_Rate'] else default)

    def get_domestic_currency(self, default):
        return utils.check_rate_name(self.param['Domestic_Currency'] if self.param['Domestic_Currency'] else default)


class FuturesPrice(Factor0D):
    """
    Spot price of a single listed futures contract. Used when a deal references the price of one
    specific futures by name rather than a forward curve indexed by maturity.
    """
    fields = [
        F('Price', 'Float', default=0, bind='value',
          description='The current settlement price of the futures contract in its quote currency')
    ]

    def __init__(self, param):
        super(FuturesPrice, self).__init__(param)

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param['Price']])


class PriceIndex(Factor0D):
    """
    Used to represent things like CPI/Stock Indices etc.
    """
    fields = [
        F('Index', 'Curve', description='Series of (date, value) pairs, the date an excel integer'),
        F('Next_Publication_Date', 'Date', default=''),
        F('Last_Period_Start', 'Date', default=''),
        F('Publication_Period', 'Text', default='Monthly', values=['Monthly', 'Quarterly']),
        F('Currency', 'Text', default='')
    ]

    def __init__(self, param):
        super(PriceIndex, self).__init__(param)
        # the start date for excel's date offset
        self.start_date = utils.excel_offset
        # the offset to the latest index value
        self.last_period = self.param['Last_Period_Start'] - self.start_date
        self.latest_index = np.where(self.param['Index'].array[:, 0] >= self.last_period.days)[0]

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param['Index'].array[self.latest_index[0]][1]] if self.latest_index.any()
                        else [self.param['Index'].array[-1][1]])

    def get_reference_value(self, ref_date):
        query = float((ref_date - self.start_date).days)
        return np.interp(query, *self.param['Index'].array.T)

    def get_last_publication_dates(self, base_date, time_grid):
        roll_period = pd.DateOffset(months=3) \
            if self.param['Publication_Period'] == 'Quarterly' else pd.DateOffset(months=1)
        last_date = base_date + pd.DateOffset(days=time_grid[-1])
        publication = pd.date_range(self.param['Last_Period_Start'], last_date, freq=roll_period)
        next_publication = pd.date_range(self.param['Next_Publication_Date'], last_date + roll_period, freq=roll_period)
        eval_dates = [(base_date + pd.DateOffset(days=t)).to_datetime64() for t in time_grid]
        return publication[np.searchsorted(next_publication.tolist(), eval_dates, side='right')]


class EquityPrice(Factor0D):
    """
    This is just the equity price on a particular end of day
    """
    fields = [
        F('Issuer', 'Text', default=''),
        F('Respect_Default', 'Text', default='Yes', values=['Yes', 'No']),
        F('Jump_Level', 'Float', default=0.0, obj='Percent'),
        F('Currency', 'Text', default=''),
        F('Interest_Rate', 'Text', default='', obj='Tuple', description='The equity repo curve'),
        F('Spot', 'Float', default=0, bind='value',
          description='Spot price in the specified Currency')
    ]

    def __init__(self, param):
        super(EquityPrice, self).__init__(param)

    def get_repo_curve_name(self):
        return utils.check_rate_name(
            self.param['Interest_Rate'] if self.param.get('Interest_Rate') else self.param['Currency'])

    def get_currency(self):
        return utils.check_rate_name(self.param['Currency'])


class CommodityPrice(EquityPrice):
    """
    This is just the commodity description. We don't use the price defined here
    we instead use the forwardPrice linked to the reference price. We just need this object
    to read the carry curve and the currency
    """
    fields = [
        F('Spot', 'Float', default=0, bind='value',
          description='Spot price in the specified Currency'),
        F('Currency', 'Text', default=''),
        F('Interest_Rate', 'Text', default='', obj='Tuple',
          description='The repo/lease (carry funding) curve'),
        F('Forward_Rate', 'Text', default='', obj='Tuple',
          description='The carry curve; falls back to Currency when blank')
    ]

    def __init__(self, param):
        super(CommodityPrice, self).__init__(param)

    def get_carry_curve_name(self):
        return utils.check_rate_name(
            self.param['Forward_Rate'] if self.param.get('Forward_Rate') else self.param['Currency'])


class ObservedBasis(Factor0D):
    """
    Represents the implied basis between two related observables (e.g. LME spot
    fix vs CME-derived synthetic spot for platinum), expressed as the price-level
    offset b(t) such that one observable = the other + b(t). The observed parent is
    the factor's own name minus its last period (e.g. ObservedBasis.PLATINUM_CME.LME_CME
    is observed against CommodityPrice.PLATINUM_CME). The state is exposed as the spot value.
    """
    fields = [F('Spot', 'Float', default=0, bind='value',
                description='Initial basis level $b_0$'),
              F('Chained_Basis', 'Text', default='',
                description='Name of the factor this one is chained to - another basis (a chain '
                            'may run link to link and may or may not close a loop) or the '
                            'primary itself. The declaration is the whole contract: whenever '
                            'this factor enters a calculation\'s universe its link follows, '
                            'and a chained process reads exactly the link it declares'),
              F('Chained_Lag', 'Integer', default=0,
                description='Rows back at which this factor\'s law references its declared '
                            'link. 0 (same row) makes the link a generation dependency - the '
                            'link simulates first; 1 marks the chain\'s day boundary (this '
                            'factor steps off the link\'s PREVIOUS row) and orders nothing. '
                            'A closed chain must lag somewhere, or it is a same-instant loop')]

    def __init__(self, param):
        super(ObservedBasis, self).__init__(param)


class ForwardPriceSample(Factor0D):
    """
    This is just the sampling method for Forward Prices
    """
    fields = [
        F('Offset', 'Integer', default=0, description='Calendar day offset'),
        F('Holiday_Calendar', 'Text', default='',
          description='Name of the calendar to use in the calendar xml file'),
        F('Sampling_Convention', 'Text', default='ForwardPriceSampleDaily',
          values=['ForwardPriceSampleDaily', 'ForwardPriceSampleBullet'])
    ]

    def __init__(self, param):
        super(ForwardPriceSample, self).__init__(param)

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param['Offset']])

    def get_holiday_calendar(self):
        return self.param.get('Holiday_Calendar')

    def get_sampling_convention(self):
        return self.param.get('Sampling_Convention')


class ReferenceVol(object):
    """Pairs a forward-price vol space with the reference-price lookup it is quoted against."""
    fields = [
        F('ForwardPriceVol', 'Text', default='', obj='Tuple',
          description='Name of the ForwardPriceVol price factor to use'),
        F('ReferencePrice', 'Text', default='', obj='Tuple',
          description='Name of the ReferencePrice price factor that defines the reference lookup')
    ]

    def __init__(self, param):
        self.param = param

    def get_forwardprice_vol(self):
        return utils.check_rate_name(self.param['ForwardPriceVol'])

    def get_forwardprice(self):
        return utils.check_rate_name(self.param['ReferencePrice'])


class InterestRateJacobian(object):
    def __init__(self, param):
        self.param = param
        self.tenors = None
        self.instruments = sorted(self.param.keys())
        self.jacobian = None
        self.inverse_jacobian = None

    def update(self, ir_curve):
        self.tenors = ir_curve.tenors
        j = []
        for bench in self.instruments:
            zero = np.zeros_like(self.tenors)
            sense = self.param[bench].array
            zero[np.searchsorted(self.tenors, sense[:, 0])] = sense[:, 1]
            j.append(zero)
        self.jacobian = np.array(j)
        self.inverse_jacobian = np.linalg.pinv(self.jacobian)

    def benchmark_tenors(self):
        return np.sort([x.array[-1, 0] for x in self.param.values()])

    def current_value(self):
        """Returns the value of the vol surface"""

        def approx_benchmark(ir_gradients):
            sensitivities = ir_gradients.reshape(1, -1).dot(self.inverse_jacobian)
            bench = dict(zip(self.instruments, sensitivities[0]))
            errors = sensitivities.dot(self.jacobian)[0] - ir_gradients
            return bench, errors

        return approx_benchmark


class Correlation(Factor0D):
    """The market implied correlation between the two rates its name pairs, e.g.
    `Correlation.FxRate.USD.ZAR/ReferencePrice.BRENT_OIL-IPE.USD`."""
    fields = [F('Value', 'Float', default=0, bind='value',
                description='Market implied correlation between the two rates the name pairs')]

    def __init__(self, param):
        super(Correlation, self).__init__(param)

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param.get('Value', 0.0) + self.delta])


class DividendRate(Factor1D):
    """
    Represents the Dividend Yield risk factor
    """
    fields = [
        F('Currency', 'Text', default=''),
        F('Curve', 'Curve', bind='value', description='Continuous dividend yield')
    ]

    def __init__(self, param):
        super(DividendRate, self).__init__(param)
        tenor_delta = (1.0 / np.array(self.tenors[:-1]).clip(1e-5, np.inf)) - \
                      (1.0 / np.array(self.tenors[1:]).clip(1e-5, np.inf))
        self.tenor_delta = np.hstack((tenor_delta, [1.0]))
        self.min_tenor = max(1e-5, self.tenors.min())
        self.max_tenor = max(1e-5, self.tenors.max())

    def get_currency(self):
        return utils.check_rate_name(self.param['Currency'])

    @staticmethod
    def get_day_count():
        """hardcode the daycount for dividend rates to act/365"""
        return utils.DAYCOUNT_ACT365

    def current_value(self, tenor_index=None, offset=0):
        """Returns the value of the rate at each tenor point (if set) else returns what's
        stored in the Curve parameter"""
        bumped_val = self.param['Curve'].array[:, 1] + self.delta
        # get the tenors
        ten = (np.array(tenor_index) if tenor_index is not None else self.tenors) + offset
        max_tenor = max(self.tenors.size - 1, 0)
        index = np.clip(
            np.searchsorted(self.tenors, ten, side='right') - 1,
            0, max_tenor)
        index_next = np.clip(index + 1, 0, max_tenor)

        alpha = (1.0 / self.tenors[index].clip(1e-5, np.inf) -
                 1.0 / ten.clip(self.min_tenor, self.max_tenor)) / self.tenor_delta[index]

        return bumped_val[index] * (1.0 - alpha) + alpha * bumped_val[index_next]


class SurvivalProb(Factor1D):
    """
    Represents the Probability of Survival risk factor
    """
    fields = [
        F('Recovery_Rate', 'Float', default=0.4, bounds=(0.0, 1.0), bind='value',
          description='The assumed recovery amount. Enter 0.4 for 40%'),
        F('Minimum_Recovery_Rate', 'Text', default='<undefined>'),
        F('Issuer', 'Text', default=''),
        F('Curve', 'Curve', bind='value', description='Negative log survival probability')
    ]

    def __init__(self, param):
        super(SurvivalProb, self).__init__(param)

    def get_day_count(self):
        """hardcode the daycount for Survival Probability rates to act/365"""
        return utils.DAYCOUNT_ACT365

    def check_interpolation(self, interpolation, tenors, rates):
        if interpolation == 'Linear':
            return ('Linear',)
        else:
            return ('LinearExtrapolate',)

    def get_day_count_accrual(self, ref_date, time_in_days):
        return utils.get_day_count_accrual(ref_date, time_in_days, self.get_day_count())

    def recovery_rate(self):
        return self.param.get('Recovery_Rate')

    def survival(self, tenors, scale = 1.0):
        H = self.current_value(tenors, offset=0.0, scale=scale)
        return np.exp(-H)

    def current_value(self, tenor_index=None, offset=0.0, scale = 1.0):
        """Returns the value of the rate at each tenor point (if set) else returns what's
        stored in the Curve parameter"""
        bumped_val = self.param['Curve'].array[:, 1] + self.delta

        if self.interpolation[0] == 'Linear':
            tenors = ((np.array(tenor_index) if tenor_index is not None else self.tenors) + offset).clip(
                self.tenors.min(), self.tenors.max())
            return scale * np.interp(tenors, self.tenors, bumped_val)
        else:
            # get the tenors - make sure we clip the min range (we can extrapolate linearly)
            tenors = ((np.array(tenor_index) if tenor_index is not None else self.tenors) + offset).clip(
                min=self.tenors.min())
            max_tenor = tenors.max(initial=0)

            if max_tenor > self.tenors.max():
                point_at_inf = max_tenor * bumped_val[-1] / self.tenors[-1]
                # return a linearly extrapolated surface
                return scale * np.interp(tenors, np.append(self.tenors, max_tenor), np.append(bumped_val, point_at_inf))
            else:
                return scale * np.interp(tenors, self.tenors, bumped_val)


class InterestRate(Factor1D):
    """
    Represents an Interest Rate risk factor - basically a time indexed array of rates
    Remember that the tenors are normally expressed as year fractions - not days.
    """
    interpolation_methods = INTERPOLATION_METHODS
    fields = [
        F('Sub_Type', 'Text', default='',
          description='Set to BasisSpread if this curve is a spread over its parent'),
        F('Day_Count', 'Text', default='ACT_365', description='Daycount convention for this curve',
          values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']),
        F('Accrual_Calendar', 'Text', default=''),
        F('Currency', 'Text', default='', description='The associated currency for this curve'),
        F('Curve', 'Curve', bind='value', description='Continuously compounded interest rate'),
        F('Near_Interpolation', 'Text', default='',
          description='Interpolation to use up to Near Date, when the near end is quoted differently'),
        F('Near_Date', 'Date', default='')
    ]

    def __init__(self, param):
        super(InterestRate, self).__init__(param)

    def get_currency(self):
        return utils.check_rate_name(self.param['Currency'])

    def get_day_count(self):
        return utils.get_day_count(self.param['Day_Count'])

    def get_subtype(self):
        return 'InterestRate' + (self.param['Sub_Type'] if self.param['Sub_Type'] else '')

    def get_day_count_accrual(self, ref_date, time_in_days):
        return utils.get_day_count_accrual(ref_date, time_in_days, self.get_day_count())


class InflationRate(Factor1D):
    """
    Represents an Interest Rate (1D) risk factor - basically a time indexed array of rates
    Remember that the tenors are normally expressed as year fractions - not days.
    """
    interpolation_methods = INTERPOLATION_METHODS
    fields = [
        F('Price_Index', 'Text', default='', obj='Tuple',
          description='Name of the associated PriceIndex factor'),
        F('Reference_Name', 'Text', default='IndexReferenceInterpolated3M',
          description='Price index reference rule',
          values=['IndexReferenceInterpolated1M', 'IndexReferenceInterpolated2M',
                  'IndexReferenceInterpolated3M', 'IndexReferenceInterpolated4M']),
        F('Day_Count', 'Text', default='ACT_365', description='Daycount convention for this curve',
          values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']),
        F('Accrual_Calendar', 'Text', default=''),
        F('Currency', 'Text', default='', description='The associated currency for this curve'),
        F('Curve', 'Curve', bind='value', description='Continuously compounded inflation growth rate')
    ]

    def __init__(self, param):
        super(InflationRate, self).__init__(param)

    def get_reference_name(self):
        return self.param['Reference_Name']

    def get_day_count(self):
        return utils.get_day_count(self.param['Day_Count'])

    def get_day_count_accrual(self, ref_date, time_in_days):
        return utils.get_day_count_accrual(ref_date, time_in_days, self.get_day_count())


class ForwardPrice(Factor1D):
    """
    Used to represent things like Futures prices on OIL/GOLD/Platinum etc.
    """
    fields = [
        F('Currency', 'Text', default='', description='The associated currency for this curve'),
        F('Curve', 'Curve', bind='value',
          description='(excel date, price) pairs giving the forward price at each contract expiry'),
        F('Fixings', 'Text', default='')
    ]

    def __init__(self, param):
        super(ForwardPrice, self).__init__(param)

    def get_currency(self):
        return utils.check_rate_name(self.param['Currency'])

    def get_day_count(self):
        return utils.DAYCOUNT_None


class ForwardRate(ForwardPrice):
    """A rate (e.g. cost-of-carry, convenience yield) sampled at absolute contract
    expiry dates — same tenor convention as `ForwardPrice` (Excel date offsets), but
    semantically a rate rather than a price. Distinct from `InterestRate` (which is
    quoted at relative year-tenors); used for forward-curve dynamics where each curve
    knot is tied to a specific dated contract.

    Routed through `Price Factor Interpolation`: a carry curve whose rate is linear in
    tenor (z = c + a*tau) is identified exactly by two knots, and `LinearExtrapolate`
    extends that line outside them — the flat-clipping `Linear` default loses the
    curvature term wherever a contract trades in front of the first knot."""
    interpolation_methods = ('Linear', 'LinearExtrapolate')
    fields = [
        F('Currency', 'Text', default='', description='The associated currency for this curve'),
        F('Curve', 'Curve', bind='value',
          description='(excel date, rate) pairs giving the rate at each contract expiry')
    ]


class ReferencePrice(Factor1D):
    """
    Used to represent how lookups on the Forward/Futures curve are performed.
    """
    fields = [
        F('Fixing_Curve', 'Curve',
          description='(date, reference date) pairs giving the delivery date for a particular '
                      'date, both in excel format'),
        F('ForwardPrice', 'Text', default='', obj='Tuple',
          description='Name of the associated ForwardPrice factor')
    ]

    def __init__(self, param):
        super(ReferencePrice, self).__init__(param)
        # the start date for excel's date offset
        self.start_date = utils.excel_offset

    def get_forwardprice(self):
        return utils.check_rate_name(self.param['ForwardPrice'])

    def get_fixings(self, resets_in_excel):
        return np.interp(
            resets_in_excel, self.param['Fixing_Curve'].array[:, 0], self.param['Fixing_Curve'].array[:, 1])

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        return self.param['Fixing_Curve'].array[:, 0]

    def current_value(self, tenors=None, offset=0.0):
        """Returns the value of the rate at each tenor point (if set) else returns what's
        stored in the Curve parameter"""
        return np.interp(tenors, *self.param['Fixing_Curve'].array.T) if tenors is not None else \
            self.param['Fixing_Curve'].array[:, 1]


class CSForwardPriceModelParameters(Factor0D):
    """
    Represents the Bootstrapped CS implied parameters for a risk neutral process
    """
    fields = [
        F('Sigma', 'Float', default=0, bind='value',
          description='Bootstrapped forward-price volatility'),
        F('Alpha', 'Float', default=0, bind='value',
          description='Bootstrapped mean reversion speed')
    ]

    def __init__(self, param):
        super(CSForwardPriceModelParameters, self).__init__(param)

    def get_tenor_indices(self):
        zero = np.array([[0.0]])
        return {'Alpha': zero,
                'Sigma': zero}

    def current_value(self, tenors=None, offset=0.0):
        """Returns the parameters of the CS factor model as a dictionary"""
        return {'Alpha': np.array([self.param['Alpha']]), 'Sigma': np.array([self.param['Sigma']])}


class HestonNandiModelParameters(Factor0D):
    """
    Represents the Bootstrapped Heston-Nandi GARCH(1,1) implied parameters for a risk neutral process.
    Asset class agnostic - the underlying may be any spot (0D) factor (FX, equity, commodity, futures);
    the factor name is just the underlying's name.

    The persistence is $\\psi=\\beta+\\alpha\\gamma^{*2}$ (must be less than 1) and the stationary
    per-step variance is $\\frac{\\omega+\\alpha}{1-\\psi}$.
    """
    fields = [
        F('Omega', 'Float', default=0, bind='value',
          description='Constant $\\omega$ of the per-step variance recursion'),
        F('Alpha', 'Float', default=0, bind='value', description='ARCH coefficient $\\alpha$'),
        F('Beta', 'Float', default=0, bind='value', description='GARCH coefficient $\\beta$'),
        F('Gamma_Star', 'Float', default=0, bind='value',
          description='Risk neutral leverage $\\gamma^*=\\gamma+\\lambda+\\frac{1}{2}$'),
        F('H0', 'Float', default=0, bind='value',
          description='The predictable variance $h_1$ of the first step from the base date')
    ]
    # one source of truth for the key set - get_tenor_indices and current_value must agree; the
    # canonical name tuple lives in utils (the explicit-arg hn_* pricers/simulator consume the same set)
    parameters = utils.HN_PARAM_NAMES

    def __init__(self, param):
        super(HestonNandiModelParameters, self).__init__(param)

    def get_tenor_indices(self):
        zero = np.array([[0.0]])
        return {x: zero for x in self.parameters}

    def current_value(self, tenors=None, offset=0.0):
        """Returns the parameters of the Heston-Nandi factor model as a dictionary"""
        return {x: np.array([self.param[x]]) for x in self.parameters}


class HestonNandiComponentModelParameters(Factor0D):
    """The bootstrapped COMPONENT Heston-Nandi (Christoffersen-Jacobs-Ornthanalai-Wang) parameters.

    The variance splits into a long-run component $q_t$ and a short-run deviation that is a pure
    AR(1) at $\\beta$; the recursions and the L-curve construction are in
    [Market Prices](market_prices.md#hestonnandi-component).

    THERE IS NO OMEGA FIELD: $\\omega_t=L_{t+1}-\\rho L_t$ is a function of the **L_Curve**, whose
    anchoring ($q_0=L(0)$) makes $E_0[q_t]=L_t$ exactly, so L is the model's expected long-run
    variance path. L is piecewise-linear between its knots and flat outside them. AND NO Q0 FIELD:
    $q_0$ is $L(0)$, off the curve's first knot - written at tenor 0 with value **H0**, the two
    states held equal at the base date since no option is quoted there.

    The knots are STRUCTURAL (the calibration ladder's pillars); the curve's VALUES are
    `bind='value'` leaves, so a greek flows to each fitted pillar as to the seven scalars.
    """
    fields = [
        F('Alpha', 'Float', default=0, bind='value',
          description='Short-run ARCH coefficient $\\alpha$'),
        F('Beta', 'Float', default=0, bind='value',
          description='Short-run persistence $\\beta$ - the AR(1) coefficient of $h_t-q_t$'),
        F('Gamma_1', 'Float', default=0, bind='value',
          description='Short-run leverage $\\gamma_1$ (its SIGN is the direction of the smile)'),
        F('Rho', 'Float', default=0, bind='value',
          description='Long-run persistence $\\rho$ of the component $q_t$'),
        F('Phi', 'Float', default=0, bind='value',
          description='Long-run ARCH coefficient $\\phi$'),
        F('Gamma_2', 'Float', default=0, bind='value',
          description='Long-run leverage $\\gamma_2$'),
        F('H0', 'Float', default=0, bind='value',
          description='The predictable variance $h_0$ of the first step from the base date'),
        F('L_Curve', 'Curve', bind='value',
          description='The expected long-run per-step variance path $L_t$, in years - '
                      '$\\omega_t=L_{t+1}-\\rho L_t$ and $q_0=L(0)$')
    ]
    # one source of truth for the scalar key set (utils owns the canonical tuple - the explicit-arg
    # hn_component_* pricers/simulator consume the same set) plus the one curve parameter
    parameters = utils.HN_COMPONENT_PARAM_NAMES
    curve = utils.HN_COMPONENT_CURVE_NAME

    def __init__(self, param):
        super(HestonNandiComponentModelParameters, self).__init__(param)

    def get_tenor(self):
        """The L curve's knots, in years - the ONE term structure this factor carries."""
        return self.param[self.curve].array[:, 0]

    def get_tenor_indices(self):
        zero = np.array([[0.0]])
        return dict({x: zero for x in self.parameters},
                    **{self.curve: self.get_tenor().reshape(-1, 1)})

    def curve_tenors(self):
        """`{parameter: knots}` for every CURVE parameter - what a consumer needs to read the
        values back as a function of time, and structural, so it is resolved once at dependency
        time rather than carried on the tensor side."""
        return {self.curve: self.get_tenor()}

    def current_value(self, tenors=None, offset=0.0):
        """Returns the parameters of the component factor model as a dictionary. The curve answers
        its VALUES (one per knot), which is what `bind='value'` publishes as leaves."""
        return dict({x: np.array([self.param[x]]) for x in self.parameters},
                    **{self.curve: self.param[self.curve].array[:, 1]})


class LogVar2FJModelParameters(Factor0D):
    """The LogVar2FJ parameters - two mean-reverting log-variance factors and a co-jump.

    $h_t=\\exp(\\text{cap}(\\ell_t+s_t))$ is the annualised DIFFUSIVE variance; the slow factor
    $\\ell$ reverts to **L_Curve** at $\\kappa_\\ell$ and the fast $s$ to zero at $\\kappa_s$, and a
    compound-Poisson event moves the return by a Gaussian $N(\\mu_J,\\sigma_J^2)$ and lifts $s$ by
    $\\nu$. Given the shocks and the counts a block return is exactly Gaussian, which is the whole
    of the pricing (logvar2fj_spec.md).

    **Rho_S** and **Mu_J** are the forward-skew levers and are piecewise CONSTANT on calendar-time
    buckets: their knots ARE the buckets' start times in years, the two curves carry the same ones,
    and one knot at 0 is the constant-parameter model. A spot smile never sees a later bucket and a
    forward smile prices on nothing else (spec 2.3.1), which is why the lever is calendar time and
    not the vol state. The constructor asserts the idiosyncratic share
    $c(t)=1-\\rho_s(t)^2-\\rho_\\ell^2\\ge$ **LV_C_MIN** in every bucket, refusing with the bucket's
    time and the three numbers.

    **Lambda**, **Cap_A** and **Cap_Beta** are STRUCTURAL, not leaves: the counts' law is not on
    the tape, and the cap is a guard that a calibrated model never reaches, so a derivative
    reported at either would be wrong. Every curve's knots are structural and its VALUES are
    `bind='value'` leaves, as the component Heston-Nandi L curve's are.
    """
    fields = [
        F('Kappa_L', 'Float', default=0, bind='value',
          description='Slow reversion speed $\\kappa_\\ell$, per year'),
        F('Sigma_L', 'Float', default=0, bind='value',
          description='Slow vol-of-log-variance $\\sigma_\\ell$'),
        F('Rho_L', 'Float', default=0, bind='value', description='Slow leverage $\\rho_\\ell$'),
        F('Kappa_S', 'Float', default=0, bind='value',
          description='Fast reversion speed $\\kappa_s$, per year'),
        F('Sigma_S', 'Float', default=0, bind='value',
          description='Fast vol-of-log-variance $\\sigma_s$'),
        F('Sigma_J', 'Float', default=0, bind='value', description='Jump dispersion $\\sigma_J$'),
        F('Nu', 'Float', default=0, bind='value',
          description='Fast log-variance co-jump $\\nu$ per event'),
        F('Lambda', 'Float', default=0,
          description='Jump intensity $\\lambda$ per year - STRUCTURAL, bumped by re-authoring'),
        F('Cap_A', 'Float', default=4.605170185988092,
          description='Log-variance cap level $a$ - STRUCTURAL, default $\\log 100$ (1000% vol)'),
        F('Cap_Beta', 'Float', default=0.25,
          description='Log-variance cap width $\\beta$ - STRUCTURAL'),
        F('L_Curve', 'Curve', bind='value',
          description='Log annualised DIFFUSIVE variance $L$ at knots in years, piecewise linear '
                      'between them and flat outside'),
        F('Rho_S', 'Curve', bind='value',
          description='Fast leverage $\\rho_s(t)$, piecewise constant on buckets starting at its '
                      'knots (years)'),
        F('Mu_J', 'Curve', bind='value',
          description='Mean log-return jump $\\mu_J(t)$, piecewise constant on the same buckets')
    ]
    #: one source of truth for each name set - utils owns the canonical tuples, which the free
    #: functions and the kit consume by the same names
    parameters = utils.LV_PARAM_NAMES
    structural = utils.LV_STRUCTURAL_NAMES
    curves = utils.LV_CURVE_NAMES

    def __init__(self, param):
        super(LogVar2FJModelParameters, self).__init__(param)
        flat = [c for c in self.curves if not isinstance(self.param[c], utils.Curve)]
        if flat:
            raise ValueError(
                'LogVar2FJModelParameters: %s must be authored as CURVES - knots in years, and '
                'for the two levers those knots ARE the calendar buckets. A bare number is the '
                'phase-1 spelling; [[0.0, x]] is the one-bucket model that reproduces it'
                % ', '.join(flat))
        knots = self.curve_tenors()
        if not np.array_equal(knots['Rho_S'], knots['Mu_J']):
            raise ValueError(
                'LogVar2FJModelParameters: Rho_S and Mu_J are piecewise constant on the SAME '
                'calendar buckets, so their knots must be equal - %s against %s'
                % (knots['Rho_S'].tolist(), knots['Mu_J'].tolist()))
        rho_s, rho_l = self.param['Rho_S'].array[:, 1], float(self.param['Rho_L'])
        c = 1.0 - rho_s * rho_s - rho_l * rho_l
        for i in np.flatnonzero(c < utils.LV_C_MIN):
            raise ValueError(
                'LogVar2FJModelParameters: the bucket at %gy declares Rho_S %g against Rho_L %g, '
                'so c = 1 - Rho_S^2 - Rho_L^2 = %g, below LV_C_MIN %g. The truncation conditions '
                "on c of the interval's variance, and a surface wanting less wants a one-shock "
                'model (spec 2.2.2) - fit against the bound, do not declare past it'
                % (knots['Rho_S'][i], rho_s[i], rho_l, c[i], utils.LV_C_MIN))

    def get_tenor_indices(self):
        zero = np.array([[0.0]])
        return dict({x: zero for x in self.parameters},
                    **{c: self.param[c].array[:, :1] for c in self.curves})

    def curve_tenors(self):
        """Every structural fact a consumer needs off this factor: each curve's knots - the L
        pillars, and for the two levers the buckets - and the three parameters that are not
        leaves. Resolved once at dependency time, so nothing rides the tensor side that carries
        no derivative."""
        return dict({c: self.param[c].array[:, 0] for c in self.curves},
                    **{x: self.param[x] for x in self.structural})

    def current_value(self, tenors=None, offset=0.0):
        """The LEAVES only - the seven scalars and each curve's values, which is what
        `bind='value'` publishes and what a greek flows to."""
        return dict({x: np.array([self.param[x]]) for x in self.parameters},
                    **{c: self.param[c].array[:, 1] for c in self.curves})


class GBMAssetPriceTSModelParameters(Factor1D):
    """
    Represents the Bootstrapped TS implied parameters for a risk neutral process
    """
    fields = [
        # NOT bind='value': get_quanto_fx() returns None when this array is all zeros, so the
        # NUMBERS decide whether the implied tensor is published at all
        F('Quanto_FX_Volatility', 'Curve',
          description='Vol of the payoff-currency FX rate, read only when quanto'),
        F('Vol', 'Curve', bind='value',
          description='Term structure of the bootstrapped asset volatility'),
        # NOT bind='value' either: get_tenor_indices branches on its TRUTHINESS, so the leaf set
        # follows the number
        F('Quanto_FX_Correlation', 'Float', default=0,
          description='Correlation between the asset and the payoff-currency FX rate')
    ]

    def __init__(self, param):
        super(GBMAssetPriceTSModelParameters, self).__init__(param)

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        return self.param['Vol'].array[:, 0]

    def get_tenor_indices(self):
        if self.param.get('Quanto_FX_Correlation', 0.0):
            return {'Vol': self.param['Vol'].array[:, 0].reshape(-1, 1),
                    'Quanto_FX_Correlation': np.array([[0.0]])}
        else:
            return {'Vol': self.param['Vol'].array[:, 0].reshape(-1, 1)}

    def current_value(self, tenors=None, offset=0.0):
        """Returns the parameters of the GBM factor model as a dictionary"""
        if self.param.get('Quanto_FX_Correlation', 0.0):
            return {'Vol': self.param['Vol'].array[:, 1],
                    'Quanto_FX_Correlation': np.array([self.param['Quanto_FX_Correlation']])}
        else:
            return {'Vol': self.param['Vol'].array[:, 1]}


class HullWhite2FactorModelParameters(Factor1D):
    """
    Represents the Bootstrapped implied parameters for a hull-white 2-factor model
    """
    fields = [
        # NOT bind='value': get_quanto_fx()'s all-zero test makes this a code path, not a value
        F('Quanto_FX_Volatility', 'Curve',
          description='Vol of the payoff-currency FX rate, read only when quanto'),
        F('Alpha_1', 'Float', default=0, bind='value',
          description='Mean reversion speed of the first factor'),
        F('Sigma_1', 'Curve', bind='value',
          description='Volatility term structure of the first factor'),
        F('Quanto_FX_Correlation_1', 'Float', default=0, bind='value',
          description='Correlation between the first factor and the payoff-currency FX rate'),
        F('Alpha_2', 'Float', default=0, bind='value',
          description='Mean reversion speed of the second factor'),
        F('Sigma_2', 'Curve', bind='value',
          description='Volatility term structure of the second factor'),
        F('Quanto_FX_Correlation_2', 'Float', default=0, bind='value',
          description='Correlation between the second factor and the payoff-currency FX rate'),
        F('Correlation', 'Float', default=0, bind='value',
          description='Correlation between the two factors')
    ]

    def __init__(self, param):
        super(HullWhite2FactorModelParameters, self).__init__(param)

    def get_instantaneous_correlation(self):
        return self.param.get('short_rate_fx_correlation')

    def get_quanto_correlation(self, corr, vols):
        C = self.get_instantaneous_correlation()
        if C is not None:
            # we calculate the average value of the vol curves
            s1 = sum([x * y for x, y in zip(
                np.diff(self.param['Sigma_1'].array[:, 0]),
                (vols[0][:-1] + vols[0][1:]) / 2)]) / self.param['Sigma_1'].array[-1, 0] if len(
                self.param['Sigma_1'].array) > 1 else vols[0][0]
            s2 = sum([x * y for x, y in zip(
                np.diff(self.param['Sigma_2'].array[:, 0]),
                (vols[1][:-1] + vols[1][1:]) / 2)]) / self.param['Sigma_2'].array[-1, 0] if len(
                self.param['Sigma_2'].array) > 1 else vols[1][0]
            p = corr[0]
            scale = C / (s1 ** 2 + s2 ** 2 + 2.0 * p * s1 * s2) ** .5
            return [scale * (s1 + p * s2), scale * (p * s1 + s2)]
        else:
            return self.param.get('Quanto_FX_Correlation_1'), self.param.get('Quanto_FX_Correlation_2')

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        if self.param['Quanto_FX_Volatility'] is None:
            self.param['Quanto_FX_Volatility'] = utils.Curve([], [(0.0, 0.0)])
        return self.param['Quanto_FX_Volatility'].array[:, 0]

    def get_vol_tenors(self):
        return [self.param['Sigma_1'].array[:, 0], self.param['Sigma_2'].array[:, 0]]

    def get_tenor_indices(self):
        zero = np.array([[0.0]])
        sig1, sig2 = self.get_vol_tenors()
        return {'Alpha_1': zero,
                'Alpha_2': zero,
                'Correlation': zero,
                'Sigma_1': sig1.reshape(-1, 1),
                'Sigma_2': sig2.reshape(-1, 1),
                'Quanto_FX_Correlation_1': zero,
                'Quanto_FX_Correlation_2': zero}

    def get_quanto_fx(self):
        if self.param.get('Quanto_FX_Volatility') is not None and self.param[
            'Quanto_FX_Volatility'].array.any():
            return self.param['Quanto_FX_Volatility'].array[:, 1]
        else:
            return None

    def current_value(self, tenors=None, offset=0.0, include_quanto=False):
        """Returns the parameters of the HW2 factor model as a dictionary"""

        params = dict([('Alpha_1', np.array([self.param['Alpha_1']])),
                       ('Alpha_2', np.array([self.param['Alpha_2']])),
                       ('Correlation', np.array([self.param['Correlation']])),
                       ('Sigma_1', self.param['Sigma_1'].array[:, 1]),
                       ('Sigma_2', self.param['Sigma_2'].array[:, 1])])

        if self.get_instantaneous_correlation() is None and (
                'Quanto_FX_Correlation_1' in self.param and 'Quanto_FX_Correlation_2' in self.param):
            # needs to be looked up if there's no instantaneous correlation - otherwise it's calculated
            params['Quanto_FX_Correlation_1'] = np.array([self.param['Quanto_FX_Correlation_1']])
            params['Quanto_FX_Correlation_2'] = np.array([self.param['Quanto_FX_Correlation_2']])

            if include_quanto:
                params['Quanto_FX_Volatility'] = self.param['Quanto_FX_Volatility'].array[:, 1]

        return params


class VolatilityGrid(Factor2D):
    """A (moneyness, expiry) vol surface - the ONE implementation, shared by every asset class.

    All the behaviour that varies is the SUBTYPE, `(Surface_Type, Moneyness_Rule)`, so FX, equity
    and commodity surfaces share this body and every one of them can author SVI, Skew and Malz.

    The asset-class TAG is carried by the three aliases below rather than by this class. A
    sensitivity is reported under the risk class of the factor it is taken with respect to, CRIF
    names those surfaces `Risk_FXVol`/`Risk_EquityVol`/`Risk_CommodityVol`, and a factor-keyed
    gradient carries nothing but `Factor(type, name)` - so `utils.FactorRiskClass` must be a pure
    function of the type, which one untagged name would make undecidable."""
    fields = [
        F('Surface_Type', 'Text', default='Explicit',
          values=['Explicit', 'SVI', 'Skew', 'Malz', 'Relative_Forward']),
        F('Surface', 'Surface', bind='value',
          description='(moneyness, expiry, volatility) triples, flat extrapolated and linearly '
                      'interpolated. Read when Surface Type is Explicit'),
        F('Moneyness_Rule', 'Text', default='Sticky_Moneyness',
          values=['Sticky_Strike', 'Sticky_Moneyness', 'Sticky_Delta']),
        # NOT bind='value': update() runs the Malz solver on these and rebuilds Surface on an
        # x-grid refined against the numbers, so the moneyness grid follows them
        F('Delta_Surface', 'Surface',
          description='(delta, expiry, volatility) triples, read when Surface Type is Malz'),
        F('ATM_Ref', 'Curve', description='Skew and SVI parameter'),
        F('ATM_Vol', 'Curve', description='Skew and SVI parameter'),
        F('a', 'Curve', description='SVI parameter'),
        F('b', 'Curve', description='SVI parameter'),
        F('s', 'Curve', description='Skew parameter'),
        F('L', 'Curve', description='Skew parameter'),
        F('R', 'Curve', description='Skew parameter'),
        F('C', 'Curve', description='Skew parameter'),
        F('D', 'Curve', description='Skew parameter'),
        F('lam', 'Curve', description='Skew parameter'),
        F('rho', 'Curve', description='Skew and SVI parameter'),
        F('m', 'Curve', description='SVI parameter'),
        F('sigma', 'Curve', description='SVI parameter'),
        F('Currency', 'Text', default=''),
        # bind='value' because it TRAVELS WITH THE VOLS: a tick delivers new numbers and the time
        # it saw them, and a stamp that invalidated the plan would recompile on every tick
        F('Quote_Timestamp', 'Date', default='', bind='value',
          description='When the quotes this surface was bootstrapped from were observed - the '
                      'latest contributing one. Reported for staleness, never read by pricing'),
        # STRUCTURAL, unlike the stamp beside it: it says what the moneyness grid IS, so a
        # re-bootstrap asking for another one is asking for another grid
        F('Grid_Tolerance', 'Float', default=0.0,
          description='The vol error the moneyness grid of a bootstrapped surface was refined to. '
                      '0 on a surface no bootstrapper built. Never read by pricing - it is what a '
                      're-bootstrap checks before reusing these nodes')
    ]


# The asset-class tags: the type a deal's vol surface is authored and keyed under, and what
# `utils.FactorRiskClass` partitions on. `fields` is re-declared as the SAME list object because
# `emit_factor` is own-attr only - an alias that merely inherited could not be authored at all.

class FXVol(VolatilityGrid):
    """The vol surface of an FX pair - CRIF `Risk_FXVol`."""
    fields = VolatilityGrid.fields


class EquityPriceVol(VolatilityGrid):
    """The vol surface of an equity or equity index - CRIF `Risk_EquityVol`."""
    fields = VolatilityGrid.fields


class CommodityPriceVol(VolatilityGrid):
    """The vol surface of a commodity or energy reference price - CRIF `Risk_CommodityVol`."""
    fields = VolatilityGrid.fields


class InterestYieldVol(Factor3D):
    """A swaption volatility space, read at (moneyness, expiry, underlying tenor).

    `Property_Aliases` - a list of key/value pairs carrying `BlackScholesDisplacedShiftValue` - is
    read with `.get` but has no descriptor, so no schema-authored block can carry it. It is the
    LEGACY spelling of the displacement, and `displacement` below states what outranks it.
    """
    fields = [
        F('Surface', 'Space', bind='value',
          description='(moneyness, expiry, tenor, volatility) quads, flat extrapolated and '
                      'linearly interpolated'),
        F('Shift', 'Float', default=0, obj='Percent',
          description='Displacement of the shifted lognormal quote. Read by the calibration as '
                      'well as by the deal path since 2026-09-01: a non-zero value outranks the '
                      'undeclared Property_Aliases legacy, and one authored beside a Normal '
                      'distribution refuses by name rather than being ignored'),
        F('Distribution_Type', 'Text', default='Lognormal', values=['Lognormal', 'Normal'],
          description='The convention these vols are quoted in. It reaches the HW2F calibration '
                      'through the benchmark PREMIUM - Lognormal strikes it with Black, Normal '
                      'with Bachelier off an absolute normal vol - and the deal path through '
                      'get_subtype')
    ]

    def __init__(self, param):
        super(InterestYieldVol, self).__init__(param)
        self.delta = 0.0
        self.atm_surface = None
        self.premiums = None

    def set_premiums(self, df, currency):
        if df is not None:
            self.premiums = df[df['Currency'] == currency[0]]

    def get_premium(self, expiry, tenor):
        prem = self.premiums[(self.premiums['UnderlyingTenor'] == tenor) &
                             (self.premiums['Expiry'] == expiry)]['Payer']
        return prem.values[0] / 10000.0

    def get_strike_from_premiums(self, expiry, tenor):
        strike = self.premiums[(self.premiums['UnderlyingTenor'] == tenor) &
                               (self.premiums['Expiry'] == expiry)]['StrikeValue']
        return strike.values[0] / 100.0

    @property
    def BlackScholesDisplacedShiftValue(self):
        """The LEGACY displacement, in percentage points - `Property_Aliases`, then a premiums file.

        Undeclared: no schema-authored block can carry it. `displacement` below is what reads it,
        behind the declared `Shift`.
        """
        shift_value = 0.0
        Property_Aliases = self.param.get('Property_Aliases')
        if Property_Aliases is not None:
            for property_alias in Property_Aliases:
                if 'BlackScholesDisplacedShiftValue' in property_alias:
                    return property_alias['BlackScholesDisplacedShiftValue']
        elif self.premiums is not None:
            return self.premiums['Shift'].apply(lambda x: float(x.replace('%', ''))).unique()[0]
        return shift_value

    @property
    def displacement(self):
        """This surface's shifted-lognormal displacement, in the STRIKE's own units.

        THE PRECEDENCE: the declared `Shift` wins; `Property_Aliases` - which has no descriptor, so
        no schema-authored block can carry it - is the legacy behind it; a premiums file's `Shift`
        column is the fallback under that.

        A `Shift` of ZERO is not an instruction: it is the field's own declared default, so an
        authored zero and an unauthored one are the same document.

        The units are the STRIKE's and not the legacy alias's percentage points - a `Percent(2.0)`
        already holds `2.0/100.0` in `amount`, so both routes divide once and agree to the bit.

        A displacement under `Distribution_Type: 'Normal'` REFUSES: a shift displaces a lognormal
        quote's strike, and a normal vol has nothing to displace.
        """
        distribution, shift = self.get_subtype()
        declared, legacy = float(shift), self.BlackScholesDisplacedShiftValue / 100.0
        if distribution == 'Normal' and (declared or legacy):
            raise Exception(
                "InterestYieldVol: a displacement of {:g}% is authored by {} beside "
                "Distribution_Type 'Normal', and a normal vol has no strike to displace - a "
                'shifted-lognormal displacement is a lognormal concept. Either drop that '
                "displacement or declare Distribution_Type 'Lognormal'".format(
                    100.0 * (declared or legacy),
                    'Shift' if declared else "Property_Aliases' BlackScholesDisplacedShiftValue"))
        return declared or legacy

    @property
    def ATM(self):
        if self.atm_surface is None:
            mn_ix = np.searchsorted(self.moneyness, 0.0)
            atm_vol = np.array([np.interp(1.0, self.moneyness[mn_ix - 1:mn_ix + 1], y)
                                for y in self.get_vols()[:, mn_ix - 1:mn_ix + 1]])
            self.atm_surface = RectBivariateSpline(
                self.tenor, self.expiry, atm_vol.reshape(self.tenor.size, self.expiry.size))
        return self.atm_surface


class InterestRateVol(Factor3D):
    """A caplet volatility space, read at (moneyness, expiry, underlying tenor)."""
    fields = [
        F('Surface', 'Space', bind='value',
          description='(moneyness, expiry, tenor, volatility) quads, flat extrapolated and '
                      'linearly interpolated')
    ]

    def __init__(self, param):
        super(InterestRateVol, self).__init__(param)


class ForwardPriceVol(Factor3D):
    """A forward-price volatility space. Its column order is the REVERSE of `Factor3D`'s -
    delivery first, moneyness last - which the three index constants below are what state."""
    TENOR_INDEX = 0
    EXPIRY_INDEX = 1
    MONEYNESS_INDEX = 2

    fields = [
        F('Surface', 'Space', bind='value',
          description='(delivery, expiry, moneyness, volatility) quads, flat extrapolated and '
                      'linearly interpolated')
    ]

    def __init__(self, param):
        self.flat = None
        self.index_map = {}
        super(ForwardPriceVol, self).__init__(param)

    def get_vols(self):
        """Uses flat extrapolation along moneyness and then linear interpolation along expiry"""
        vols = []
        surface = self.param['Surface'].array
        # sorted by moneyness within expiry within delivery (lexsort takes its primary key last)
        self.sorted_vol = surface[np.lexsort(
            (surface[:, self.MONEYNESS_INDEX], surface[:, self.EXPIRY_INDEX], surface[:, self.TENOR_INDEX]))]
        self.index_map.clear()

        # store the offsets of all the tenor indices
        self.index_map[self.TENOR_INDEX] = np.append(
            self.sorted_vol[:, self.TENOR_INDEX].searchsorted(self.get_tenor()), len(surface))

        for start, end in zip(
                self.index_map[self.TENOR_INDEX][:-1], self.index_map[self.TENOR_INDEX][1:]):
            expiry_valid = np.unique(self.sorted_vol[start:end, self.EXPIRY_INDEX]) if end > start else np.array([
                self.sorted_vol[start, self.EXPIRY_INDEX]])
            self.index_map.setdefault(self.EXPIRY_INDEX, []).append((expiry_valid, start, end))

        for expiry in self.index_map[self.EXPIRY_INDEX]:
            idx_start, idx_end = expiry[1:]
            starts = np.append(
                idx_start + self.sorted_vol[idx_start: idx_end, self.EXPIRY_INDEX].searchsorted(expiry[0]), idx_end)
            for start, end in zip(starts[:-1], starts[1:]):
                moneyness_valid = self.sorted_vol[start: end, self.MONEYNESS_INDEX] if end > start else np.array([
                    self.sorted_vol[start, self.MONEYNESS_INDEX]])
                self.index_map.setdefault(self.MONEYNESS_INDEX, []).append((moneyness_valid, start, end))

        self.flat = self.sorted_vol[:, 3]

        for moneyness in self.moneyness:
            surface = self.param['Surface'].array[self.param['Surface'].array[:, self.MONEYNESS_INDEX] == moneyness]
            for x in self.expiry:
                sigma = surface[surface[:, self.EXPIRY_INDEX] == x, 3]
                tenor = surface[surface[:, self.EXPIRY_INDEX] == x, self.TENOR_INDEX]
                vols.append(np.interp(self.tenor, tenor, sigma) if sigma.any() else np.zeros_like(self.tenor))

        return np.array(vols)

    def current_value(self, tenors=None, offset=0.0):
        """
        Returns the value of the vol space. Symmetric with get_tenor_indices:
        current_value(get_tenor_indices()) is the list of corresponding vols.
        """
        if tenors is not None:
            interp_vols = []
            if self.expiry.size > 1 and self.moneyness.size > 1:
                interpolator = [
                    RectBivariateSpline(self.expiry, self.tenor, vol.reshape(self.expiry.size, -1), kx=1, ky=1)
                    for vol in self.vols.reshape(self.moneyness.size, -1)]

                for tenor in tenors:
                    index = np.clip(self.moneyness.searchsorted(
                        tenor[self.MONEYNESS_INDEX], side='right') - 1, 0, self.moneyness.size - 1)
                    index_p1 = np.clip(index + 1, 0, self.moneyness.size - 1)
                    val = np.interp(
                        tenor[self.MONEYNESS_INDEX], self.moneyness[[index, index_p1]],
                        np.dstack([interpolator[index](tenor[self.EXPIRY_INDEX], tenor[self.TENOR_INDEX]),
                                   interpolator[index_p1](tenor[self.EXPIRY_INDEX],
                                                          tenor[self.TENOR_INDEX])]).flatten())
                    interp_vols.append(val)
                return np.array(interp_vols)
            elif self.expiry.size > 1:
                for tenor in tenors:
                    val = np.interp(
                        tenor[self.EXPIRY_INDEX], self.sorted_vol[:, self.EXPIRY_INDEX], self.sorted_vol[:, -1])
                    interp_vols.append(val)
                return np.array(interp_vols)

        return self.flat

    def get_tenor_indices(self):
        return self.sorted_vol[:, :3]


@utils.log_exception
def construct_factor(factor, price_factors, factor_interp, base_date=None):
    # now lookup the params of the factor
    pf = price_factors[utils.resolve_factor_key(factor, price_factors)]
    # change the logging name in case there are any errors
    logging.root.name = '.'.join(factor.name)
    # check the interpolation on interest Rates - can add more methods/price factors as desired
    pf_local = dict(pf)  # shallow copy
    if factor.type in ['InterestRate', 'InflationRate', 'ForwardRate']:
        interp_method = factor_interp.search(factor, pf, True)
        pf_local.update(
            {'Interpolation' : factor_interp_map.get(interp_method, 'Linear'),
             'base_date': base_date})
    return globals().get(factor.type)(pf_local)


if __name__ == '__main__':
    pass
