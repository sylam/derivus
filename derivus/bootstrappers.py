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
import time
import logging
import threading
from collections import namedtuple, OrderedDict
from functools import partial

# third party stuff
import numpy as np
import pandas as pd
import torch

# Internal modules
from . import utils, pricing, instruments, riskfactors, stochasticprocess
from .schema import F, OPTION_QUOTE, REQUIRED, Row
from ._version import __version__

import scipy.optimize


def resolve_factor(name, price_factors, candidates):
    """The factor `name` refers to, typed by the first candidate the price factors hold a block for.

    A market-price block names its inputs by NAME - the type is the market data's business, not the
    quote author's - so a bootstrapper declares the candidate types and this picks between them.
    That is also how the asset-class vol tags and the untagged name they replace are both readable:
    `utils.TwoDimensionalFactors` IS the candidate list for a vol surface."""
    rate = utils.check_rate_name(name)
    return utils.Factor(next(x for x in candidates if utils.check_tuple_name(
        utils.Factor(x, rate)) in price_factors), rate)


class market_swap_class(namedtuple('market_swap', 'deal_data price weight quote premium',
                                   defaults=(None, None))):
    """One benchmark swaption of a risk-neutral IR calibration: the compiled par swap, the market
    premium the model has to reproduce, and the weight it carries in the objective.

    `quote` and `premium` are the QUOTE SIDE and are ABSENT by default - the float64 leaf the market
    number arrived on, and the MAP from that leaf to this swaption's premium. See
    `create_market_swaps` for what a quote is here and what the map is.

    `premium` is a callable and not a tensor, so the twin is rebuilt inside every evaluation rather
    than compiled once with the benchmark set. That is not a style choice: `make_basin_hopping_loss`
    calls `total_loss.backward()` with no `retain_graph`, which frees the whole graph the loss was
    built on - a compile-time subgraph hanging off the residual would be freed with the first
    evaluation and every one after it would raise. Rebuilding costs one scalar Black per benchmark
    per evaluation, against a Monte Carlo over the whole path set.
    """

    def error(self, model, resid):
        """This swaption's weighted relative pricing error against its `model` price.

        The quote rides in as a SPLICE, `base + (carried - detach(carried))` - the boundary
        correction's shape, and here for its reason: worth EXACTLY zero in the forward pass, with
        derivative one. `base` is the expression the solve always minimised, evaluated off the numpy
        market premium in the calculation's own precision, so enabling the quote side cannot move a
        mark by construction rather than by a claim anyone has to re-check. `carried` is that same
        expression off the float64 twin, and only its derivative survives the subtraction.

        `model` is DETACHED in the carried half and only there. The splice is worth zero in the
        forward pass but its derivative is not selective: left attached, `carried` reaches the model
        parameters as well as the quote and the calibration Jacobian comes out DOUBLED, which no
        price gate can see - the residual is bit-identical and the optimizer just walks a different
        path. The quote derivative of the error does not involve the model's own sensitivity, so
        detaching is what the chain rule says, not a workaround.

        Splicing here rather than at the price is deliberate: `price` is a numpy scalar, and torch
        divides a tensor by a scalar at the SCALAR's precision. Replacing it with a float64 tensor
        rounds twice where the engine rounds once, which moved the residual by an ulp - measured,
        not feared.
        """
        base = self.weight * resid(100.0 * (self.price / model - 1.0))
        if self.premium is None:
            return base
        carried = self.weight * resid(100.0 * (self.premium(self.quote) / model.detach() - 1.0))
        return base + (carried - carried.detach()).to(base.dtype)


date_desc = {'years': 'Y', 'months': 'M', 'days': 'D'}
# date formatter
date_fmt = lambda x: ''.join(['{0}{1}'.format(v, date_desc[k]) for k, v in x.kwds.items()])


class RiskNeutralInterestRate_State(utils.Calculation_State):
    def __init__(self, scenario_keys, batch_size, device, dtype, nomodel='Constant'):
        super(RiskNeutralInterestRate_State, self).__init__(
            None, torch.ones([1, 1], dtype=dtype, device=device), 2048, None, nomodel, batch_size, False)
        # these are tensors
        self.t_PreCalc = {}
        self.scenario_keys = scenario_keys
        self.t_random_batch = None
        self.batch_index = 0
        self.t_Scenario_Buffer = {}

    @property
    def t_random_numbers(self):
        return self.t_random_batch[self.batch_index]

    def reset(self, num_batches, numfactors, time_grid):
        # clear the buffers
        self.t_Buffer.clear()
        self.t_PreCalc.clear()

        if self.t_random_batch is None:
            # the sobol engine in torch > 1.8 goes up to dimension 21201 - so this should be fine
            self.sobol = torch.quasirandom.SobolEngine(
                dimension=time_grid.time_grid_years.size * numfactors, scramble=True, seed=1234)
            # skip this many samples
            self.sobol.fast_forward(2048)
            # make sure we don't include 1 or 0
            sample_sobol = self.sobol.draw(self.simulation_batch * num_batches).reshape(
                num_batches, self.simulation_batch, -1)
            sample = torch.erfinv(2 * (0.5 + (1 - torch.finfo(sample_sobol.dtype).eps) * (
                    sample_sobol - 0.5)) - 1).reshape(
                num_batches, self.simulation_batch, -1) * 1.4142135623730951
            self.t_random_batch = sample.transpose(1, 2).reshape(
                num_batches, numfactors, -1, self.simulation_batch).to(self.one.device)


def create_float_cashflows(base_date, cashflow_obj, frequency):
    cashflows = []
    for cashflow, reset in zip(cashflow_obj.schedule, cashflow_obj.Resets.schedule):
        cashflows.append({
            'Payment_Date': base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_Pay_Day]),
            'Notional': 1.0,
            'Accrual_Start_Date': base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_Start_Day]),
            'Accrual_End_Date': base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_End_Day]),
            'Accrual_Year_Fraction': cashflow[utils.CASHFLOW_INDEX_Year_Frac],
            'Fixed_Amount': cashflow[utils.CASHFLOW_INDEX_FixedAmt],
            'Resets': [[base_date + pd.offsets.Day(reset[utils.RESET_INDEX_Reset_Day]),
                        base_date + pd.offsets.Day(reset[utils.RESET_INDEX_Start_Day]),
                        base_date + pd.offsets.Day(reset[utils.RESET_INDEX_End_Day]),
                        reset[utils.RESET_INDEX_Accrual],
                        frequency, 'ACT_365', '0D', 0.0, 'No', utils.Percent(0.0)]],
            'Margin': utils.Basis(0.0)
        })
    return cashflows


def black_premium(pvbp, strike, expiry, delta, quote):
    """One ATM swaption's premium as a differentiable function of its VOL quote.

    `utils.black_european_option` is the engine's own tensor Black - what the cap/floor and swaption
    pricers value an option with - so this is the twin of the numpy `black_european_option_price`
    that `create_market_swaps` prices the market premium with, not a second opinion of it. At the
    money, which is the only place this is called, the two came out bit-identical at every point
    measured and a gate holds them to 1e-12.
    """
    return pvbp * utils.black_european_option(
        quote.new_tensor(strike), quote.new_tensor(strike), quote + delta, expiry, 1.0, 1.0, None)


def create_market_swaps(base_date, time_grid, curve_index, vol_surface, curve_factor,
                        instrument_definitions, rate=None, unit=None):
    """The benchmark swaptions of one risk-neutral IR calibration: a compiled par swap, the market
    premium the model has to reproduce, and the objective weight.

    THE QUOTE SIDE. `unit` is the residual's unit tensor when the block asks for `Quote_Sensitivity`
    and `None` otherwise, so absent by default nothing outside such a solve knows this exists. The
    market premium here is built by numpy (`utils.black_european_option_price` is scipy end to end),
    so it crosses into the residual as a scalar and the quote behind it is severed by construction.
    What this hands over to close that is a PAIR per swaption: the quote as a float64 leaf, and the
    map from that leaf back to this swaption's premium. `market_swap_class.error` is where the two
    are spliced onto the residual.

    What a quote IS depends on what the block quotes. A vol-quoted swaption - `Market_Volatility` on
    the row, or the surface's ATM read when that column is zero - carries the VOL, and the map is
    `black_premium`, the differentiable preamble that turns a vol into a premium. A premium-quoted
    one carries the PREMIUM itself and the map is the identity. The vol surface's own interpolation
    is numpy (`RectBivariateSpline`), so the leaf is the ATM vol AT this swaption's expiry and
    tenor rather than a node of the surface.
    """
    # a premium bumped by `Volatility_Delta` reaches the residual through a brentq implied-vol
    # solve, and a numerical root find carries no derivative - so the quote side declines it
    if unit is not None and vol_surface.premiums is not None and vol_surface.delta:
        raise Exception('Quote_Sensitivity: a premium re-struck at Volatility_Delta reaches the '
                        'residual through a brentq implied-vol solve, which carries no derivative')
    # store these benchmark swap definitions if necessary
    benchmarks = []
    # store the benchmark instruments
    all_deals = {}
    # cater for shifted lognormal vols
    shift_parameter = vol_surface.BlackScholesDisplacedShiftValue / 100.0
    for instrument in instrument_definitions:
        # set up the instrument
        effective = base_date + instrument['Start']
        maturity = effective + instrument['Tenor']
        exp_days = (effective - base_date).days
        tenor = (maturity - effective).days / utils.DAYS_IN_YEAR
        expiry = exp_days / utils.DAYS_IN_YEAR
        time_index = np.searchsorted(time_grid.mtm_time_grid, [exp_days], side='right') - 1
        swaption_name = 'Swaption_{}_{}'.format(
            date_fmt(instrument['Start']), date_fmt(instrument['Tenor']))

        float_pay_dates = instruments.generate_dates_backward(
            maturity, effective, instrument['Floating_Frequency'])

        float_cash = utils.generate_float_cashflows(
            base_date, time_grid, float_pay_dates, 1.0, None, None,
            instrument['Floating_Frequency'], pd.DateOffset(month=0),
            utils.get_day_count(instrument['Floating_Day_Count']), 0.0)

        K, pvbp = float_cash.get_par_swap_rate(base_date, curve_factor)

        if instrument['Fixed_Frequency'] != instrument['Floating_Frequency']:
            fixed_pay_dates = instruments.generate_dates_backward(
                maturity, effective, instrument['Fixed_Frequency'])
            fixed_cash = utils.generate_fixed_cashflows(
                base_date, fixed_pay_dates, 1.0, None, utils.get_day_count(instrument['Fixed_Day_Count']), 0.0)
            pv_float = K * pvbp
            pvbp = fixed_cash.get_par_swap_rate(base_date, curve_factor)
            K = pv_float / pvbp
            fixed_cash.set_fixed_amount(K)
            fixed_indices = float_cash[:, utils.CASHFLOW_INDEX_Pay_Day].searchsorted(
                fixed_cash[:, utils.CASHFLOW_INDEX_Pay_Day])

            if not (float_cash[fixed_indices, utils.CASHFLOW_INDEX_Pay_Day] ==
                    fixed_cash[:, utils.CASHFLOW_INDEX_Pay_Day]).all():
                logging.error('Float leg and Fixed legs do not coincide')
                raise Exception('Float leg and Fixed legs do not coincide')

            # set the float leg fixed amount
            float_cash.schedule[fixed_indices, utils.CASHFLOW_INDEX_FixedAmt] = \
                -fixed_cash[:, utils.CASHFLOW_INDEX_FixedAmt]
        else:
            float_cash.set_fixed_amount(-K)

        # get the atm vol
        if instrument['Market_Volatility'].amount:
            vol = instrument['Market_Volatility'].amount
        else:
            vol = vol_surface.ATM(tenor, expiry)[0][0]

        deal_data = utils.DealDataType(
            Instrument=None, Factor_dep={'Cashflows': float_cash, 'Forward': curve_index,
                                         'Discount': curve_index, 'CompoundingMethod': 'None'},
            Time_dep=utils.DealTimeDependencies(time_grid.mtm_time_grid, time_index), Calc_res=None)

        shifted_strike = K + shift_parameter
        # first check if we have the actual premium (not implied)
        if vol_surface.premiums is not None:
            swaption_price = vol_surface.get_premium(date_fmt(instrument['Start']), date_fmt(instrument['Tenor']))
            if vol_surface.delta:
                try:
                    implied_vol = scipy.optimize.brentq(lambda v: pvbp * utils.black_european_option_price(
                        shifted_strike, shifted_strike, 0.0, v, expiry, 1.0, 1.0) - swaption_price, 0.01, vol + .5)
                except:
                    modified_k = vol_surface.get_strike_from_premiums(date_fmt(instrument['Start']),
                                                                      date_fmt(instrument['Tenor']))
                    logging.warning(
                        'Implied vol calc during delta bump failed - calculated strike is {} - using strike from premium file {}'.format(
                            K, modified_k))
                    shifted_strike = modified_k + shift_parameter
                    implied_vol = scipy.optimize.brentq(lambda v: pvbp * utils.black_european_option_price(
                        shifted_strike, shifted_strike, 0.0, v, expiry, 1.0, 1.0) - swaption_price, 0.01, vol + .5)

                swaption_price = pvbp * utils.black_european_option_price(
                    shifted_strike, shifted_strike, 0.0, implied_vol + vol_surface.delta, expiry, 1.0, 1.0)
        else:
            swaption_price = pvbp * utils.black_european_option_price(
                shifted_strike, shifted_strike, 0.0, vol + vol_surface.delta, expiry, 1.0, 1.0)

        # the quote side - a float64 leaf and the map back to this swaption's premium, see docstring
        quote, premium = None, None
        if unit is not None:
            premium_quoted = vol_surface.premiums is not None
            quote = unit.new_tensor(
                swaption_price if premium_quoted else vol, dtype=torch.float64).requires_grad_(True)
            premium = (lambda q: q) if premium_quoted else partial(
                black_premium, pvbp, shifted_strike, expiry, vol_surface.delta)

        # store this
        all_deals[swaption_name] = market_swap_class(
            deal_data=deal_data, price=swaption_price, weight=instrument['Weight'],
            quote=quote, premium=premium)

        # store the benchmark
        if rate is not None:
            benchmarks.append(
                instruments.construct_instrument(
                    {'Object': 'CFFloatingInterestListDeal',
                     'Reference': swaption_name,
                     'Currency': curve_factor.param['Currency'],
                     'Discount_Rate': '.'.join(rate),
                     'Forecast_Rate': '.'.join(rate),
                     'Buy_Sell': 'Buy',
                     'Cashflows': {'Items': create_float_cashflows(
                         base_date, float_cash, instrument['Floating_Frequency'])}},
                    {})
            )

    return all_deals, benchmarks


class CSForwardPriceModelParameters(object):
    documentation = (
        'Energy',
        ['For Risk Neutral simulation, the Clewlow Strickland Model is calibrated to a set of European Energy',
         'futures options $J$.',
         'an integrated curve $\\bar{\\sigma}(t)$ needs to be specified and is',
         'interpreted as the average volatility at time $t$. This is typically obtained from the corresponding',
         'ATM volatility. This is then used to construct a new variance curve $V(t)$ which is defined as',
         '$V(0)=0, V(t_i)=\\bar{\\sigma}(t_i)^2 t_i$ and $V(t)=\\bar{\\sigma}(t_n)^2 t$ for $t>t_n$ where',
         '$t_1,...,t_n$ are discrete points on the ATM volatility curve.',
         '',
         'Points on the curve that imply a decrease in variance (i.e. $V(t_i)<V(t_{i-1})$) are adjusted to',
         '$V(t_i)=\\bar\\sigma(t_i)^2t_i=V(t_{i-1})$. This curve is then used to construct *instantaneous* curves',
         'that are then input to the corresponding stochastic process.',
         '',
         'The relationship between integrated $F(t)=\\int_0^t f_1(s)f_2(s)ds$ and instantaneous curves $f_1, f_2$',
         'where the instantaneous curves are defined on discrete points $P={t_0,t_1,..,t_n}$ with $t_0=0$ is defined',
         'on $P$ by Simpson\'s rule:',
         '',
         '$$F(t_i)=F(t_{i-1})+\\frac{t_i-t_{i-1}}{6}\\Big(f(t_i)+4f(\\frac{t_i+t_{i-1}}{2})+f(t_i)\\Big)$$',
         '',
         'and $f(t)=f_1(t)f_2(t)$. Integrated curves are flat extrapolated and linearly interpolated.'
         ]
    )

    market_factor_type = 'CSForwardPriceModelPrices'
    fields = [
        F('Energy', 'Text', default=REQUIRED, description='The ForwardPrice factor to calibrate'),
        F('Forward_Volatility', 'Text', default=REQUIRED,
          description='The ForwardPriceVol surface the quoted vols are read off'),
        F('Discount_Rate', 'Text', default=REQUIRED,
          description='The InterestRate curve the premiums discount on'),
        F('Quote_Type', 'Text', default='Implied_Volatility', values=['Implied_Volatility'],
          description='How Quoted_Market_Value reads - this family takes vols only'),
        F('Energy_Futures_Options', 'Table', default='null',
          row=Row(OPTION_QUOTE[:1] + [F('Settlement_Date', 'Date',
                                        description='Futures settlement, which sets the '
                                                    'Clewlow-Strickland decay term')] +
                  OPTION_QUOTE[1:]),
          description='The option quotes sigma and alpha are fitted to')
    ]

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Checks for Declining variance in the ATM vols of the relevant price factor and corrects accordingly.
        '''

        def B(a, t):
            return (1.0 - np.exp(-a * t)) / a if a != 0 else t

        def V(sigma, alpha, T, S):
            return sigma * sigma * np.exp(-2.0 * alpha * S) * B(-2.0 * alpha, T)

        def calc_error(x, options):
            sigma, alpha = x
            error = 0.0

            for option in options:
                discount = np.exp(-option['r'] * option['T'])
                error += option['Weight'] * (option['Premium'] - utils.black_european_option_price(
                    option['Forward'], option['Strike'], 0.0, np.sqrt(V(sigma, alpha, option['T'], option['S'])),
                    1.0, option['Units'], 1.0 if option['Option_Type'] == 'Call' else -1.0) * discount) ** 2
            return error

        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                # get the vol surface
                if 'ForwardPriceVol.' + implied_params['instrument']['Forward_Volatility'] in price_factors:
                    vol_factor = utils.Factor('ForwardPriceVol', utils.check_rate_name(
                        implied_params['instrument']['Forward_Volatility']))
                if 'ForwardPrice.' + implied_params['instrument']['Energy'] in price_factors:
                    energy_factor = utils.Factor('ForwardPrice', utils.check_rate_name(
                        implied_params['instrument']['Energy']))
                if 'InterestRate.' + implied_params['instrument']['Discount_Rate'] in price_factors:
                    discount_factor = utils.Factor('InterestRate', utils.check_rate_name(
                        implied_params['instrument']['Discount_Rate']))

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                    vol_surface.delta = sys_params.get('Volatility_Delta', 0.0)
                    forward = riskfactors.construct_factor(energy_factor, price_factors, factor_interp)
                    discount = riskfactors.construct_factor(discount_factor, price_factors, factor_interp)
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                # need to loop over this and create some market prices.
                quote_type = implied_params['instrument']['Quote_Type']
                for option in implied_params['instrument']['Energy_Futures_Options']:
                    t = discount.get_day_count_accrual(
                        sys_params['Base_Date'], (option['Expiry_Date'] - sys_params['Base_Date']).days)
                    d = discount.get_day_count_accrual(
                        sys_params['Base_Date'], (option['Settlement_Date'] - sys_params['Base_Date']).days)
                    expiry_excel = (option['Expiry_Date'] - utils.excel_offset).days
                    settlement_excel = (option['Settlement_Date'] - utils.excel_offset).days
                    forward_at_exp = forward.current_value(expiry_excel)
                    forward_at_settle = forward.current_value(settlement_excel)
                    r = discount.current_value(t)
                    if quote_type == 'Implied_Volatility':
                        sigma = vol_surface.current_value([[t, d, 1.0]])[0] if not option['Quoted_Market_Value'] else \
                            option['Quoted_Market_Value']
                        sigma += vol_surface.delta
                    else:
                        logging.error('quote_type {} not supported yet'.format(quote_type))
                        continue

                    option['Strike'] = forward_at_exp if not option['Strike'] else option['Strike']
                    option['Forward'] = forward_at_settle
                    option['r'] = r
                    option['S'] = d
                    option['T'] = t
                    option['sigma'] = sigma
                    option['Premium'] = utils.black_european_option_price(
                        option['Forward'], option['Strike'], r, sigma, t,
                        option['Units'], 1.0 if option['Option_Type'] == 'Call' else -1.0)

                result = scipy.optimize.minimize(
                    calc_error, (0.5, 0.1),
                    args=(implied_params['instrument']['Energy_Futures_Options'],),
                    bounds=[(0.001, 2.5), (-1, 2.0)])

                # log the results
                for option in implied_params['instrument']['Energy_Futures_Options']:
                    vol = np.sqrt(V(result.x[0], result.x[1], option['T'], option['S']) / option['T'])
                    discount = np.exp(-option['r'] * option['T'])
                    fitted_premium = utils.black_european_option_price(
                        option['Forward'], option['Strike'], 0.0,
                        np.sqrt(V(result.x[0], result.x[1], option['T'], option['S'])),
                        1.0, option['Units'], 1.0 if option['Option_Type'] == 'Call' else -1.0) * discount
                    err = (fitted_premium - option['Premium']) ** 2
                    logging.info(
                        'Commodity {} strike {}, expiry {}, vol {}, c_vol {}, premium {}, c_premium {}, err {}'.format(
                            implied_params['instrument']['Energy'], option['Strike'], option['Expiry_Date'],
                            option['sigma'], vol, option['Premium'], fitted_premium, err))

                price_param = utils.Factor(self.__class__.__name__, market_factor.name)

                price_factors[utils.check_tuple_name(price_param)] = {
                    'Property_Aliases': None,
                    'Sigma': result.x[0],
                    'Alpha': result.x[1]}


class HestonNandiModelParameters(object):
    documentation = (
        'Fx And Equity',
        ['For Risk Neutral simulation, the Heston-Nandi GARCH(1,1) model is calibrated to a set of European',
         'options $J$ on a spot underlying. The model is ASSET CLASS AGNOSTIC - the *Underlying* may be any',
         'spot (0D) price factor (**FxRate**, **EquityPrice**, **CommodityPrice**, **FuturesPrice**) and the',
         '*Volatility* any (moneyness, expiry) vol surface (**FXVol**, **EquityPriceVol**,',
         '**CommodityPriceVol**); the type of each is looked up from the price factors, or named explicitly',
         'with *Underlying_Type* / *Volatility_Type*. Under the locally risk neutral valuation relationship',
         '(LRNVR) $\\lambda^*=-\\frac{1}{2}$, so the model is parameterised directly in $\\gamma^*$:',
         '',
         '$$\\log\\frac{S_{t+1}}{S_t}=(r-q)-\\frac{h_{t+1}}{2}+\\sqrt{h_{t+1}}z_{t+1}$$',
         '',
         '$$h_{t+1}=\\omega+\\beta h_t+\\alpha\\Big(z_t-\\gamma^*\\sqrt{h_t}\\Big)^2$$',
         '',
         'with $z\\sim N(0,1)$ i.i.d. and $h_{t+1}$ predictable (known at $t$), hence the fitted initial',
         'variance is $h_1$ - the variance of the *first* step - and is stored as **H0**. Option values come',
         'from the recursive characteristic function of Heston and Nandi (2000) inverted by Gauss-Legendre',
         'quadrature (see the Heston-Nandi section of `derivus.utils`). The optional *Yield* (a dividend, repo, convenience or carry',
         'curve) enters as $q$ - the drift is $r-q$ and the value carries the extra $e^{-qt}$ factor - so',
         'equity, FX and commodity underlyings are all handled by the same objective.',
         '',
         'Writing the persistence as $\\psi=\\beta+\\alpha\\gamma^{*2}$ and the stationary per-step variance as',
         '$m=\\frac{\\omega+\\alpha}{1-\\psi}$, the objective',
         '',
         '$$\\sum_{j\\in J}w_j\\Big(V_j-V_j(\\omega,\\alpha,\\beta,\\gamma^*,h_1)\\Big)^2$$',
         '',
         'is minimized with L-BFGS-B over $\\Big(\\log\\omega,\\psi,l,\\frac{\\gamma^*}{1000},\\log h_1\\Big)$',
         'where $\\alpha=\\frac{l\\psi}{\\gamma^{*2}}$ and $\\beta=\\psi(1-l)$ for a leverage share',
         '$l\\in[0,1]$. Stationarity is therefore a *box constraint on a fitted parameter*',
         '($\\psi\\le1-10^{-6}$) and holds at every point the optimizer visits - there is no penalty term and',
         'no infeasible iterate. Gradients are exact (torch autograd through the inversion).',
         '',
         'Target premia are the Black prices at the corresponding vol surface point (as per the Clewlow',
         'Strickland bootstrapper) unless *Quote_Type* is **Premium**, in which case the quoted values are',
         'used directly. A previously bootstrapped price factor (if present) is used to warm start the fit.',
         '',
         'MONEYNESS CONVENTION. Unlike the other bootstrappers this one queries the surface AWAY FROM',
         'THE MONEY, where the five moneyness conventions in this framework no longer coincide, so the',
         'lookup point is produced by `pricing.calc_moneyness` - the same dispatch every option deal',
         'uses - off the surface *SubType*, with *Use_Forward* and *Invert_Moneyness* (Yes/No, both',
         'defaulting to **No**, i.e. $\\frac{S}{K}$, as they do in the pricing path). Supported',
         '*Surface_Types* are **Explicit**, **Relative_Forward** and **Malz** - the ones whose vol at a',
         'strike is a table lookup. **SVI** and **Skew** surfaces are parametric (the vol needs the',
         'ATM_Ref/wing machinery of the pricing path) and are REFUSED with an error rather than',
         'mis-looked-up: quote those premiums directly with *Quote_Type* **Premium**.'
         ]
    )

    # The Fourier inversion needs double precision - the framework default (float32) destroys the
    # cancellation in P1/P2 - so the dtype this is constructed with is deliberately ignored.
    prec = torch.float64
    # x = (log Omega, psi, leverage share, Gamma_Star/1000, log H0) - see reparam
    bounds = [(np.log(1e-12), np.log(1e-3)), (0.0, 1.0 - 1e-6), (0.0, 1.0),
              (1e-3, 5.0), (np.log(1e-10), np.log(1e-2))]
    # candidate price factor types for each instrument input - the underlying is any spot (0D)
    # factor and the volatility any (moneyness, expiry) surface, so one instrument definition
    # serves FX, equity and commodity underlyings
    factor_types = {'Underlying': ['FxRate', 'EquityPrice', 'CommodityPrice', 'FuturesPrice'],
                    'Volatility': utils.TwoDimensionalFactors,
                    'Discount_Rate': ['InterestRate'],
                    'Yield': ['DividendRate', 'InterestRate']}
    # Surface_Types whose vol at a strike is a TABLE LOOKUP, hence usable here. SVI/Skew are
    # parametric - their vol needs the ATM_Ref/wing machinery of the pricing path (Factor2D
    # returns the parameters, not a vol), so a synthesised premium would be silently wrong.
    tabular_surfaces = ('Explicit', 'Relative_Forward', 'Malz')

    market_factor_type = 'HestonNandiModelPrices'
    # the four factor references, each with the optional `_Type` its `resolve` reads, whose valid
    # values ARE the candidate list - one source, so a new candidate cannot miss the schema.
    # `Yield` is the one reference the fit runs without; the rest are hard-read.
    fields = [F(field, 'Text', default='' if field == 'Yield' else REQUIRED,
                description='The {} factor - one of {}'.format(
                    field.replace('_', ' ').lower(), ', '.join(types)))
              for field, types in factor_types.items()] + [
        F(field + '_Type', 'Text', default='', values=[''] + list(types),
          description='Names the factor type explicitly, where the name exists under more than one')
        for field, types in factor_types.items()] + [
        F('Quote_Type', 'Text', default='Implied_Volatility',
          values=['Implied_Volatility', 'Premium'],
          description='Whether Quoted_Market_Value is a vol to price at or a premium to fit'),
        F('Use_Forward', 'Text', default='No', values=['Yes', 'No'],
          description='Moneyness against the forward rather than the spot'),
        F('Invert_Moneyness', 'Text', default='No', values=['Yes', 'No'],
          description='Moneyness as K/S rather than S/K'),
        F('Steps_Per_Year', 'Float', default=252.0,
          description='GARCH steps an expiry is spread over'),
        F('Quadrature_Panels', 'Integer', default=64,
          description='Gauss-Legendre panels the characteristic function is inverted on'),
        F('European_Options', 'Table', default='null', row=Row(OPTION_QUOTE),
          description='The option quotes the five parameters are fitted to')]

    def __init__(self, param, device, dtype):
        self.device = device
        self.param = param

    @classmethod
    def resolve(cls, instrument, field, price_factors):
        """The factor named by instrument[field], typed by the first candidate that exists in the
        price factors, or by an explicit instrument[field + '_Type']. None if the field is unset."""
        if not instrument.get(field):
            return None
        return resolve_factor(instrument[field], price_factors, [instrument[field + '_Type']]
                              if instrument.get(field + '_Type') else cls.factor_types[field])

    @staticmethod
    def reparam(x):
        """Maps the fitted vector x to (Omega, Alpha, Beta, Gamma_Star, H0).

        STATIONARITY IS ENFORCED BY CONSTRUCTION, not by a penalty: the optimizer fits the
        persistence psi = Beta + Alpha*Gamma_Star^2 itself (a plain box bound psi <= 1-1e-6) and
        splits it between the two channels with a leverage share l in [0, 1]. Omega and H0 are
        fitted in logs so they stay positive and so their scale (~1e-6) doesn't wreck the line
        search against Gamma_Star (~1e3, hence the /1000).
        """
        psi, lev, gamma = x[1], x[2], x[3] * 1000.0
        return torch.exp(x[0]), lev * psi / gamma ** 2, psi * (1.0 - lev), gamma, torch.exp(x[4])

    @staticmethod
    def unreparam(omega, alpha, beta, gamma, h0):
        """Inverse of reparam (used to warm start off an existing price factor)."""
        psi = beta + alpha * gamma ** 2
        return np.array([np.log(omega), psi, alpha * gamma ** 2 / psi, gamma / 1000.0, np.log(h0)])

    @classmethod
    def moneyness(cls, strike, spot, forward, vol_surface, use_forward, invert_moneyness):
        """The moneyness coordinate to look the vol surface up at.

        There are FIVE conventions in this framework and they are dispatched off the surface's
        SubType, so this DELEGATES to pricing.calc_moneyness - the same function every option deal
        uses - rather than reimplementing the dispatch. calc_moneyness only reads the SubType out
        of deal_data, so a minimal Deal_data carrying this surface's SubType is all it needs.
        """
        deal_data = utils.DealDataType(
            Instrument=None, Time_dep=None, Calc_res=None,
            Factor_dep={'Volatility': [(None, None, vol_surface.get_subtype())]})
        return float(pricing.calc_moneyness(
            *[torch.tensor(float(x), dtype=cls.prec) for x in (strike, spot, forward)],
            deal_data, use_forward, invert_moneyness))

    @staticmethod
    def price(spot, strike, is_call, units, omega, alpha, beta, gamma, r, n, h0, panels, yield_discount=1.0):
        """Heston-Nandi European option value - puts by put-call parity off the call.

        ``r`` is the per step COST OF CARRY r-q (so the simulated spot has the right forward) and
        ``yield_discount`` = exp(-q*t) converts the resulting value back to a discounting at r:
        the internal price is exp(-(r-q)t)[F P1 - K P2], the value is exp(-rt)[F P1 - K P2]. Parity
        survives the rescale, so puts are still call - S + K exp(-(r-q)n) times the same factor."""
        call = utils.hn_call(spot, strike, n, h0, omega, alpha, beta, gamma, r, panels=panels)
        return units * yield_discount * (call - (1.0 - is_call) * (spot - strike * torch.exp(-r * n)))

    def calc_error(self, x, groups, spot, panels, scale):
        """Weighted squared premium error and its exact gradient (autograd).

        ``scale`` is the mean squared quoted premium: L-BFGS-B's gradient tolerance is ABSOLUTE, so
        without it the fit would stop early on a low priced underlying (an fx rate) and late on a
        high priced one. Dividing by a constant leaves the relative Weights untouched."""
        x_t = torch.tensor(x, device=self.device, dtype=self.prec, requires_grad=True)
        omega, alpha, beta, gamma, h0 = self.reparam(x_t)
        error = 0.0
        for n, b, q, strike, is_call, units, weight, premium in groups:
            fitted = self.price(spot, strike, is_call, units,
                                omega, alpha, beta, gamma, b, n, h0, panels, q)
            error = error + (weight * (premium - fitted) ** 2).sum() / scale
        error.backward()
        return float(error.detach()), x_t.grad.cpu().numpy()

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Calibrates the risk neutral Heston-Nandi GARCH(1,1) parameters to a set of European options on any
        spot underlying and writes them out as a HestonNandiModelParameters price factor.
        '''

        def tensor(x):
            return torch.tensor(x, device=self.device, dtype=self.prec)

        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                instrument = implied_params['instrument']

                # resolve the underlying spot, its vol surface, the discount curve and any yield
                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(
                        self.resolve(instrument, 'Volatility', price_factors), price_factors, factor_interp)
                    vol_surface.delta = sys_params.get('Volatility_Delta', 0.0)
                    underlying = riskfactors.construct_factor(
                        self.resolve(instrument, 'Underlying', price_factors), price_factors, factor_interp)
                    discount = riskfactors.construct_factor(
                        self.resolve(instrument, 'Discount_Rate', price_factors), price_factors, factor_interp)
                    yield_factor = self.resolve(instrument, 'Yield', price_factors)
                    carry = riskfactors.construct_factor(
                        yield_factor, price_factors, factor_interp) if yield_factor else None
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                spot = float(underlying.current_value()[0])
                quote_type = instrument['Quote_Type']
                steps_per_year = instrument.get('Steps_Per_Year', 252.0)
                panels = instrument.get('Quadrature_Panels', 64)
                use_forward = instrument.get('Use_Forward') == 'Yes'
                invert_moneyness = instrument.get('Invert_Moneyness') == 'Yes'

                # a mis-looked-up vol would produce a wrong-but-converged calibration - the worst
                # outcome - so refuse the surface rather than guess at its convention
                subtype = vol_surface.get_subtype()
                if quote_type == 'Implied_Volatility' and subtype[0] not in self.tabular_surfaces:
                    logging.error(
                        'Cannot bootstrap {0} - volatility {1} has Surface_Type {2} (Moneyness_Rule {3}); '
                        'only {4} surfaces can be queried at a strike. Quote premiums directly '
                        '(Quote_Type Premium) instead'.format(
                            market_price, instrument['Volatility'], subtype[0], subtype[1],
                            '/'.join(self.tabular_surfaces)))
                    continue

                # need to loop over this and create some market prices - group by expiry so that all
                # the strikes of one expiry share a single characteristic function recursion
                expiries = {}
                for option in instrument['European_Options']:
                    t = discount.get_day_count_accrual(
                        sys_params['Base_Date'], (option['Expiry_Date'] - sys_params['Base_Date']).days)
                    r = float(discount.current_value(t))
                    q = float(carry.current_value(t)) if carry is not None else 0.0
                    forward = spot * np.exp((r - q) * t)
                    sign = 1.0 if option['Option_Type'] == 'Call' else -1.0
                    option['Strike'] = forward if not option['Strike'] else option['Strike']
                    option['r'] = r
                    option['q'] = q
                    option['T'] = t
                    # the number of GARCH steps to expiry - the carry is spread over them so that
                    # exp(-b_step*n) is exactly exp(-(r-q)*t)
                    option['n'] = max(int(round(t * steps_per_year)), 1)
                    if quote_type == 'Implied_Volatility':
                        moneyness = self.moneyness(
                            option['Strike'], spot, forward, vol_surface, use_forward, invert_moneyness)
                        sigma = vol_surface.current_value([[moneyness, t]])[0] if not option[
                            'Quoted_Market_Value'] else option['Quoted_Market_Value']
                        sigma += vol_surface.delta
                        option['Moneyness'] = moneyness
                        option['Premium'] = utils.black_european_option_price(
                            forward, option['Strike'], r, sigma, t, option['Units'], sign)
                    elif quote_type == 'Premium':
                        option['Premium'] = option['Units'] * option['Quoted_Market_Value']
                        # back out the Black vol of the quote (seeds the fit and the diagnostics)
                        call = option['Quoted_Market_Value'] + (0.0 if sign > 0 else
                                                                forward - option['Strike']) * np.exp(-r * t)
                        sigma = np.sqrt(utils.bs_implied_total_var(
                            call, spot * np.exp(-q * t), option['Strike'], r * t, 1) / t)
                    else:
                        logging.error('quote_type {} not supported yet'.format(quote_type))
                        continue
                    option['sigma'] = sigma
                    expiries.setdefault(option['n'], []).append(option)

                groups = [(n, tensor((opts[0]['r'] - opts[0]['q']) * opts[0]['T'] / n),
                            tensor(np.exp(-opts[0]['q'] * opts[0]['T'])),
                            tensor([x['Strike'] for x in opts]),
                            tensor([1.0 if x['Option_Type'] == 'Call' else 0.0 for x in opts]),
                            tensor([x['Units'] for x in opts]),
                            tensor([x['Weight'] for x in opts]),
                            tensor([x['Premium'] for x in opts])) for n, opts in expiries.items()]

                price_param = utils.Factor(self.__class__.__name__, market_factor.name)
                param_name = utils.check_tuple_name(price_param)
                if param_name in price_factors:
                    # warm start off the previous fit
                    old = price_factors[param_name]
                    x0 = np.clip(self.unreparam(*(old[k] for k in utils.HN_PARAM_NAMES)),
                                 *np.array(self.bounds).T)
                else:
                    var = np.mean([x['sigma'] for opts in expiries.values()
                                   for x in opts]) ** 2 / steps_per_year
                    x0 = np.array([np.log(0.1 * var), 0.9, 0.5, 0.1, np.log(var)])

                scale = np.mean([x['Premium'] ** 2 for opts in expiries.values() for x in opts])
                result = scipy.optimize.minimize(
                    self.calc_error, x0, args=(groups, spot, panels, scale), jac=True,
                    method='L-BFGS-B', bounds=self.bounds,
                    # the default ftol/gtol are calibrated for an O(1e2) objective - the normalised
                    # one starts at O(1) and a good fit is O(1e-12), so let it run to that
                    options={'ftol': 1e-15, 'gtol': 1e-12})

                omega, alpha, beta, gamma, h0 = [
                    float(x) for x in self.reparam(tensor(result.x))]

                # log the results
                with torch.no_grad():
                    for n, b, q, strike, is_call, units, weight, premium in groups:
                        pt = [tensor(x) for x in (omega, alpha, beta, gamma)]     # the four recursion params
                        fitted = self.price(spot, strike, is_call, units, *pt, b, n, tensor(h0), panels, q)
                        for option, fitted_premium in zip(expiries[n], fitted.cpu().numpy()):
                            vol = utils.hn_implied_vol(
                                spot, option['Strike'], n, tensor(h0), *pt, b, steps_per_year, panels=panels)
                            logging.info(
                                'Underlying {} strike {}, expiry {}, steps {}, vol {}, c_vol {}, premium {}, '
                                'c_premium {}, err {}'.format(
                                    instrument['Underlying'], option['Strike'], option['Expiry_Date'], n,
                                    option['sigma'], vol, option['Premium'], fitted_premium,
                                    (fitted_premium - option['Premium']) ** 2))

                logging.info(
                    'Underlying {} Heston-Nandi Omega {}, Alpha {}, Beta {}, Gamma_Star {}, H0 {}, '
                    'persistence {}, long run vol {}, sse {} ({})'.format(
                        instrument['Underlying'], omega, alpha, beta, gamma, h0,
                        utils.hn_persistence(alpha, beta, gamma),
                        utils.hn_ann_vol(omega, alpha, beta, gamma, steps_per_year),
                        result.fun, result.message))

                # emit in the canonical HN_PARAM_NAMES order (single source), paired with reparam's
                # documented (Omega, Alpha, Beta, Gamma_Star, H0) output tuple
                price_factors[param_name] = {
                    'Property_Aliases': None,
                    **dict(zip(utils.HN_PARAM_NAMES, (omega, alpha, beta, gamma, h0)))}


class GBMAssetPriceTSModelParameters(object):
    documentation = (
        'Fx And Equity',
        ['For Risk Neutral simulation, an integrated curve $\\bar{\\sigma}(t)$ needs to be specified and is',
         'interpreted as the average volatility at time $t$. This is typically obtained from the corresponding',
         'ATM volatility. This is then used to construct a new variance curve $V(t)$ which is defined as',
         '$V(0)=0, V(t_i)=\\bar{\\sigma}(t_i)^2 t_i$ and $V(t)=\\bar{\\sigma}(t_n)^2 t$ for $t>t_n$ where',
         '$t_1,...,t_n$ are discrete points on the ATM volatility curve.',
         '',
         'Points on the curve that imply a DECREASE in forward variance are adjusted up to the least',
         'variance that step can reach, which is the one a zero instantaneous vol over it leaves:',
         '$V(t_i)=V(t_{i-1})+\\frac{t_i-t_{i-1}}{3}\\sigma(t_{i-1})^2$. This curve is then used to construct',
         '*instantaneous* curves that are then input to the corresponding stochastic process.',
         '',
         'The relationship between integrated $F(t)=\\int_0^t f_1(s)f_2(s)ds$ and instantaneous curves $f_1, f_2$',
         'where the instantaneous curves are defined on discrete points $P={t_0,t_1,..,t_n}$ with $t_0=0$ is defined',
         'on $P$ by Simpson\'s rule:',
         '',
         '$$F(t_i)=F(t_{i-1})+\\frac{t_i-t_{i-1}}{6}\\Big(f(t_i)+4f(\\frac{t_i+t_{i-1}}{2})+f(t_i)\\Big)$$',
         '',
         'and $f(t)=f_1(t)f_2(t)$. Integrated curves are flat extrapolated and linearly interpolated.'
         ]
    )

    market_factor_type = 'GBMAssetPriceTSModelPrices'
    factor_types = {'Asset_Price_Volatility': utils.TwoDimensionalFactors}
    #: The precision the TAPE runs in - the value path is numpy and has no dtype to pick. It is
    #: float64 on the CPU whatever the job asked for: `construct_bootstrapper`'s dtype does not
    #: reach it, and a handful of scalar operations has nothing to gain from a device whose
    #: rounding is its own.
    dtype = torch.float64
    fields = [
        F('Asset_Price_Volatility', 'Text', default=REQUIRED,
          description='The vol surface whose ATM column becomes the integrated vol curve'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the integrated vol curve connected to the ATM vols it was built from, '
                      'so a calculation\'s backward pass reports dV/dq beside dV/dtheta. The '
                      'written curve is identical either way')
    ]

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param
        #: What a block asking for `Quote_Sensitivity` leaves behind: the integrated vol curve STILL
        #: CONNECTED to its ATM quotes, under the key `_build_factor_state` mints its `Vol` leaf
        #: with, and the quote leaf per block. `Config.bootstrap` harvests both - they are tensors,
        #: so they cannot live in `Price Factors`, which is data and gets written back out as JSON.
        self.calibrated = {}
        self.quote_leaves = {}

    @staticmethod
    def atm_column(vol_factor, vol_surface, market_prices, price_factors):
        """The ATM vol per surface expiry, and where each number came from.

        TWO SOURCES, and which one a config gets is a property of the surface's PROVENANCE rather
        than a switch. Where this same market data carries an `FXVolPrices` block for the surface
        being integrated AND that surface is the one the block WROTE, its ATM rows ARE its ATM vols
        - `Factor2D.malz_skew` puts the +-0.5 label's vol at the delta-neutral straddle strike, so
        the identity is the surface's own construction - and the quotes are taken straight off it:
        exact, and the coordinate a desk explains a vol P&L in. Reading them back off the refined
        log-moneyness grid instead would recover the same numbers to the grid's own tolerance and no
        better, and it would put the Malz delta solve on the tape to say so.

        PROVENANCE IS EVIDENCE, NOT A NAME. A hand-authored surface can sit under a name a quote
        block also uses, and then the two are simply different market data: the pricers read the
        authored surface and preferring the quotes would integrate a curve nothing else agrees with.
        What is checked is the fingerprint `FXVolSurfaceParameters` leaves on what it writes and
        `pinned_grid` reads back - the `Malz` subtype beside the `Grid_Tolerance` the grid was built
        at - so the preference follows the surface rather than the string.

        Anything else is authored data, and the ATM column is what `np.interp` reads off it at
        moneyness 1 - unchanged, and the entries it returns ARE the quotes. Where the surface
        carries a node AT moneyness 1 that read is the node itself, so dV/d(ATM column) is
        dV/d(surface node) there.

        KNOWN, AND NOT FIXED HERE: moneyness 1 is the ATM coordinate of a RATIO surface (F/K), and
        a `Malz` surface's axis is log(F/K), whose ATM is at 0. A hand-authored Malz surface
        therefore reads its last log-moneyness node - a wing - and has done since before quotes had
        derivatives. The preferred path above is the one Malz surfaces reach in practice, so this
        increment names the defect rather than moving a shipped forward for it.
        """
        family = FXVolSurfaceParameters
        quoted = market_prices.get(utils.check_tuple_name(utils.Factor(
            family.market_factor_type, vol_factor.name)))
        written = price_factors.get(utils.check_tuple_name(vol_factor), {})
        if quoted is not None and written.get('Surface_Type') == family.surface_type and \
                'Grid_Tolerance' in written:
            atm = family.atm_quotes(family.used(quoted['instrument']))
            if set(atm) != set(vol_surface.expiry):
                raise ValueError(
                    '{} is quoted at expiries {} and carries a surface over {} - the quotes moved '
                    'since it was built, so re-bootstrap the surface before integrating it'.format(
                        utils.check_tuple_name(vol_factor),
                        ', '.join('{:g}'.format(T) for T in sorted(atm)),
                        ', '.join('{:g}'.format(T) for T in sorted(vol_surface.expiry))))
            return [atm[expiry] for expiry in vol_surface.expiry], 'its own ATM quotes'

        mn_ix = np.searchsorted(vol_surface.moneyness, 1.0)
        return [np.interp(1, vol_surface.moneyness[mn_ix - 1:mn_ix + 1], y) for y in
                vol_surface.get_vols()[:, mn_ix - 1:mn_ix + 1]], 'the surface at moneyness 1'

    @staticmethod
    def integrated_vol(atm_vol, expiry):
        """The ATM column as the integrated vol curve the process reads - THE VALUE PATH.

        `V(t_i) = sigma_bar(t_i)^2 t_i` is the total variance the column implies, and the walk is
        Simpson's rule inverted for the instantaneous vol over each step,

            V(t_i) - V(t_{i-1}) = (dt/3)(sigma_{i-1}^2 + sigma_{i-1} sigma_i + sigma_i^2)

        a quadratic in `sigma_i` whose positive root is taken. Returns the curve and the expiries
        the repair below fired at.

        This is the numpy walk this family has always shipped, arithmetic untouched, and it is the
        only thing a mark is ever built from - see `carried_vol` for the derivative twin and for why
        the two are not one function.

        THE MAP IS PIECEWISE, and the switch is a KINK. A column implying a DECLINING forward
        variance has no root, so `V(t_i)` is floored at the least variance the step can reach - the
        one `sigma_i = 0` leaves - and the written vol is that floor rather than the quote. On the
        smooth side the written vol is the quote and `d/dq` is 1; on the floored side the floor does
        not involve that quote at all and `d/dq` is 0, in that column AND in every later one.

        ONLY `sigma_bar` IS WRITTEN. `sigma` is the walk's own state - it sizes the next step's
        floor and is never published - so where no repair fires the curve is `sqrt(q^2 t / t)`, which
        is `q` up to the rounding of a square and its root and is exactly `q` on most columns. The
        arithmetic only bites where variance declines, which is where the gates put it.
        """
        if expiry.size == 1:
            return list(atm_vol), []

        dt = np.diff(np.append(0, expiry))
        var = expiry * np.array(atm_vol) ** 2
        sig, vol, var_tm1, floored = atm_vol[:1], atm_vol[:1], var[0], []

        for var_t, delta_t, t_i in zip(var[1:], dt[1:] / 3.0, expiry[1:]):
            M = var_tm1 + delta_t * (sig[-1] ** 2)
            if var_t < M:
                floored.append(t_i)
                var_t = M

            a, b, c = delta_t, sig[-1] * delta_t, M - var_t
            sig.append((-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a))
            vol.append(np.sqrt(var_t / t_i))
            var_tm1 = var_t

        return vol, floored

    @staticmethod
    def carried_vol(atm_vol, expiry):
        """The same walk on a tape - A DERIVATIVE CARRIER, and never a value.

        It rides in as the splice `integrated_vol + (carried - carried.detach())`, the shape the
        rest of the quote side uses: worth exactly zero in the forward pass, derivative one. So the
        curve a job ships is the numpy walk's, bit for bit, whatever this returns.

        THAT SEPARATION IS THE POINT, and it is measured rather than assumed. Every operation here
        is correctly rounded under IEEE-754, which makes it tempting to have one walk; `torch.sqrt`
        is still one ulp below `np.sqrt` on better than one float64 in a hundred on this box, and a
        torch walk re-associates the expression tree besides. Letting it write the curve moved the
        shipped vols on 24% of 4000 random ATM columns. An ulp of a shipped vol is not a rounding
        question, it is a different number in a report.

        THE DISCRIMINANT IS GUARDED, and only here. `sqrt` has an INFINITE derivative at zero, and
        the floored branch walks straight into it: one repair leaves `sigma` exactly zero, so a
        second reaches `b = 0` beside a `c` that cancels to zero and the discriminant IS zero. The
        forward value is fine - the root is zero - but the backward pass multiplies that infinity by
        the zero `d(b^2)/db` and reports NaN, which then eats the whole Jacobian rather than one
        entry. The root there is zero, so it is written as zero and the `sqrt` never sees the point.
        """
        third = np.diff(np.append(0.0, expiry)) / 3.0
        variance = atm_vol * atm_vol * atm_vol.new_tensor(expiry)
        sigma, curve, previous = atm_vol[0], [atm_vol[0]], variance[0]

        for i in range(1, expiry.size):
            floor = previous + third[i] * sigma * sigma
            previous = floor if variance[i] < floor else variance[i]
            b = sigma * third[i]
            disc = b * b - 4.0 * third[i] * (floor - previous)
            real = disc > 0
            root = torch.where(real, torch.sqrt(torch.where(real, disc, torch.ones_like(disc))),
                               torch.zeros_like(disc))
            sigma = (-b + root) / (2.0 * third[i])
            curve.append(torch.sqrt(previous / expiry[i]))

        return torch.stack(curve)

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Turns the ATM column of the named vol surface into the integrated vol curve the risk neutral
        process reads, repairing any declining variance on the way - see `integrated_vol`.

        A block asking for `Quote_Sensitivity` leaves that curve behind still connected to the ATM
        vols it was built from, so `Calculation.factor_leaf` can offer the connected tensor where it
        would otherwise mint a `Vol` leaf out of numpy. The map is explicit, so there is no solve
        here and no implicit function theorem: the curve IS a differentiable function of the quotes
        and autograd walks it.

        The tape is a SPLICE over the shipped walk and not a replacement for it - see `carried_vol`
        - so what a block asking for quote sensitivities changes is what `backward()` can reach and
        nothing else. Every number written below comes out of `integrated_vol` either way.
        '''
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                # get the vol surface
                vol_factor = resolve_factor(implied_params['instrument']['Asset_Price_Volatility'],
                                            price_factors, self.factor_types['Asset_Price_Volatility'])
                implied_param = vol_factor.name
                # this asks whether the thing being MODELLED is an fx rate - the surface's own tag
                # says which asset class it belongs to, but the model is named after the underlying
                is_fx = utils.check_tuple_name(utils.Factor('FxRate', rate[1:])) in price_factors

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                connect = implied_params['instrument'].get('Quote_Sensitivity', 'No') == 'Yes'
                atm_vol, source = self.atm_column(
                    vol_factor, vol_surface, market_prices, price_factors)
                curve, floored = self.integrated_vol(atm_vol, vol_surface.expiry)

                # store the output
                price_param = utils.Factor(self.__class__.__name__, market_factor.name)
                model_param = utils.Factor('GBMAssetPriceTSModelImplied', market_factor.name)
                vol = utils.Curve(['Integrated'], list(zip(vol_surface.expiry, curve)))

                if is_fx:
                    quanto_fx_corr = 0.0
                else:
                    quanto_fx_corr = price_factors.get(
                        'Correlation.EquityPrice.{}.{}/FxRate.{}.{}'.format(
                            rate[-1], implied_param[-1], *sorted([sys_params['Base_Currency'], implied_param[-1]])),
                        {'Value': 0.0})['Value']

                price_factors[utils.check_tuple_name(price_param)] = {
                    'Property_Aliases': None,
                    'Vol': vol,
                    'Quanto_FX_Volatility': None,
                    'Quanto_FX_Correlation': quanto_fx_corr}
                price_models[utils.check_tuple_name(model_param)] = {'Risk_Premium': None}

                if connect:
                    quotes = torch.tensor(atm_vol, dtype=self.dtype, requires_grad=True)
                    carried = self.carried_vol(quotes, vol_surface.expiry)
                    self.calibrated[utils.Factor(
                        price_param.type, price_param.name + ('Vol',))] = torch.tensor(
                        curve, dtype=self.dtype) + (carried - carried.detach())
                    self.quote_leaves[market_price] = (
                        ['ATM {:g}'.format(expiry) for expiry in vol_surface.expiry], quotes)

                logging.info('{} built from {} ATM vols off {}'.format(
                    utils.check_tuple_name(price_param), len(atm_vol), source))
                if floored:
                    logging.warning('Fixed declining variance for {} at {}'.format(
                        market_price, ', '.join('{:g}'.format(expiry) for expiry in floored)))


class SwaptionCalibration(object):
    """One risk-neutral swaption calibration as an operand: the residual, and the solve over it.

    The residual is what `calc_loss_on_ir_curve` builds - one weighted relative pricing error per
    `Instrument_Definitions` row, priced by brute-force Monte Carlo under a FROZEN Sobol sample -
    and the solve is the optimizer chain `calc_loss` hands over. Holding both beside the parameter
    dict they share is what lets `LeastSquaresSolve` run the ordinary solve in its forward pass and
    then differentiate the same residual in its backward.

    The parameter vector is FLAT here and a dict everywhere else: scipy takes a vector, the process
    takes `{name: tensor}`, and the two scipy adapters own that boundary with `tn_var.data =
    torch.from_numpy(x)`. `__call__` is the third crossing and the only differentiable one - it
    builds VIEWS of a flat tensor rather than writing `.data`, because an implicit function theorem
    needs an edge from the residual back to the parameters and `.data` is exactly what severs one.
    """

    def __init__(self, name, loss_fn, implied_var, optimizers, process, market_swaps):
        self.name = name
        self.loss_fn = loss_fn
        self.implied_var = implied_var
        self.optimizers = optimizers
        self.process = process
        self.market_swaps = market_swaps
        self.keys = list(implied_var)
        self.sizes = [implied_var[key].numel() for key in self.keys]

    @property
    def quotes(self):
        """The quote leaf per benchmark, or `()` when the block did not ask for `Quote_Sensitivity`
        - which is what makes the wrapper a pass-through with no edge recorded."""
        return tuple(swap.quote for swap in self.market_swaps.values() if swap.quote is not None)

    @property
    def descriptors(self):
        """The benchmark names of `quotes`, in its order - what `quote_leaves` pairs them with."""
        return [name for name, swap in self.market_swaps.items() if swap.quote is not None]

    def split(self, theta):
        """`{name: tensor}` in the closure's own parameter order, SHARING theta's graph.

        The one place the flat vector is taken apart: `__call__` builds the residual's parameter
        views with it and the attachment publishes theta* per named parameter with it, so a factor
        leaf cannot be handed the wrong slice of the vector the Jacobian was read off.
        """
        return dict(zip(self.keys, theta.split(self.sizes)))

    def unflatten(self, theta):
        """`{name: numpy}` in the closure's own parameter order - the shape `save_params` takes."""
        return {name: value.detach().cpu().numpy() for name, value in self.split(theta).items()}

    def __call__(self, x):
        """The residual vector at flat parameters `x`, differentiable in `x` AND in the quotes.

        A FRESH dict of views is handed to the closure rather than the standing `implied_var`,
        because those are leaves whose `.data` the scipy adapters overwrite: `split` keeps the
        graph the Jacobian is read off. `x` carries the closure's own precision, so the residual is
        the same number the solve stopped on - the float64 promotion belongs to the linear algebra
        downstream, not to the pricing.
        """
        return torch.stack(list(self.loss_fn(self.split(x))[1].values()))

    def solve(self):
        """theta* as a flat tensor: the optimizer chain, run exactly as a bootstrap runs it.

        Basin hopping then least squares, `x0` chained from one to the next, and a candidate is
        ACCEPTED only if it beats the running best AND the process it implies is well posed - so
        the answer can be the seed, which is what `LeastSquaresSolve` checks stationarity for.
        """
        calibrated_swaptions, errors = self.loss_fn(self.implied_var)
        batch_loss = torch.stack(list(errors.values())).sum().cpu().detach().numpy()
        vars = {k: v.cpu().detach().numpy() for k, v in self.implied_var.items()}
        # initialize the soln with the current values
        soln = (batch_loss, vars)
        logging.info('{} - Batch loss {}'.format(self.name, batch_loss))
        for k, v in sorted(vars.items()):
            logging.info('{} - {}'.format(k, v))

        for k, v in sorted(calibrated_swaptions.items()):
            value = v.cpu().detach().numpy()
            price = self.market_swaps[k].price
            logging.debug('{},market_value,{:f},sim_model_value,{:f},error,{:.0f}%'.format(
                k, price, value, 100.0 * (price - value) / price))

        # minimize
        result = None
        num_optimizers = len(self.optimizers)
        for op_loop in range(num_optimizers):
            optim = self.optimizers[op_loop % num_optimizers]
            x0 = result['x'] if result is not None else optim[1]
            if optim[0] == 'basin':
                result = scipy.optimize.basinhopping(
                    optim[2], x0=x0, take_step=optim[3], accept_test=optim[4], T=5.0, niter=50,
                    minimizer_kwargs={"method": "L-BFGS-B", "jac": True, "bounds": optim[5]},
                    rng=optim[6])
                batch_loss = float(optim[2](result['x'])[0])
            elif optim[0] == 'leastsq':
                result = scipy.optimize.least_squares(
                    optim[2], x0=x0, jac=optim[3], bounds=optim[4])
                batch_loss = optim[2](result['x']).sum()

            if batch_loss < soln[0] and self.process.params_ok:
                sim_swaptions, errors = self.loss_fn(self.implied_var)
                vars = {k: v.cpu().detach().numpy() for k, v in self.implied_var.items()}
                soln = (batch_loss, vars)
                logging.info('{} - run {} - Batch loss {}'.format(self.name, op_loop, batch_loss))
                for k, v in sorted(vars.items()):
                    logging.info('{} - {}'.format(k, v))
                for k, v in sim_swaptions.items():
                    value = v.cpu().detach().numpy()
                    price = self.market_swaps[k].price
                    logging.info('{},market_value,{:f},sim_model_value,{:f},error,{:.0f}%'.format(
                        k, price, value, 100.0 * (price - value) / price))

        theta = np.concatenate([soln[1][key] for key in self.keys])
        return torch.tensor(theta, dtype=self.implied_var[self.keys[0]].dtype,
                            device=self.implied_var[self.keys[0]].device)


class LeastSquaresSolve(torch.autograd.Function):
    """The swaption calibration as one differentiable node: quotes in, calibrated parameters out.

    FORWARD IS THE ORDINARY SOLVE. It calls `SwaptionCalibration.solve` and nothing else - the same
    optimizer chain, the same acceptance rule, the same frozen Sobol sample - so enabling quote
    gradients cannot move theta* by construction rather than by a claim anyone has to re-check.
    Autograd runs `forward` with grad mode off and both optimizers need it on (basin hopping calls
    `backward()` on its loss, least squares reads its Jacobian off `autograd.grad`), so it is
    re-enabled here and the graph each evaluation builds is discarded with the evaluation.

    BACKWARD IS THE IMPLICIT FUNCTION THEOREM AT THE STATIONARITY FIXED POINT. This is a
    least-squares minimum, not a root: `r(theta*, q)` is not zero and never will be, so what is
    held fixed is the GRADIENT of half the sum of squares, `g = J^T r = 0`. Differentiating that
    and dropping the term in `d(J^T)/dtheta . r` - the Gauss-Newton approximation, exact in the
    limit of a zero residual - gives

        (J^T J) dtheta/dq = -J^T dr/dq

    so a cotangent `v = dL/dtheta*` contracts as `w = (J^T J)^+ v` then `dL/dq = -(dr/dq)^T (J w)`.
    Both derivatives come from autograd on ONE fresh evaluation of the residual at `(theta*, q)`,
    functionally through `autograd.grad` rather than off `.grad`: the scipy adapters clear `.grad`
    per evaluation and the quote leaves accumulate across them, so a harvested `.grad` is the sum
    over an optimizer's whole path rather than the derivative at its answer.

    J^T J IS RANK DEFICIENT and that is a property of the problem, not a defect: J has one row per
    benchmark and 23 columns, so on any block quoting fewer than 23 swaptions the null space is
    what the quote set does not identify. The solve is therefore a PSEUDO-inverse at a declared
    relative cutoff, and `dtheta/dq` in a null direction is the MINIMUM-NORM representative - one
    member of a family the data cannot choose between. No ridge is added: a Tikhonov term would
    return a unique-looking number that is a derivative of a different problem.

    STATIONARITY IS CHECKED, NOT ASSUMED. `solve` accepts whatever the chain returned, which may be
    the seed if nothing beat it, and the whole contraction above is worthless off the fixed point -
    so `||J^T r||` above the declared tolerance RAISES, naming the norm, rather than returning a
    quietly-wrong Jacobian.

    Every `grad` here retains the graph, for the reason `CalibrationSolve` documents: the residual's
    subgraph is shared with the evaluation the Jacobian was read off.
    """

    @staticmethod
    def forward(ctx, calibration, rcond, stationarity, *quotes):
        with torch.enable_grad():
            theta = calibration.solve()
        ctx.calibration, ctx.theta = calibration, theta
        ctx.rcond, ctx.stationarity = rcond, stationarity
        return theta

    @staticmethod
    def backward(ctx, cotangent):
        # the engine runs a backward node with grad mode set to `create_graph`, so this is the
        # second differentiation asking to be recorded - and Gauss-Newton has no second derivative
        if torch.is_grad_enabled():
            raise Exception('Swaption calibration: create_graph is not supported - the backward is '
                            'a Gauss-Newton contraction and carries no second derivative')
        calibration = ctx.calibration
        with torch.enable_grad():
            x = ctx.theta.detach().requires_grad_(True)
            residual = calibration(x)
            jacobian = torch.stack([torch.autograd.grad(residual[i], x, retain_graph=True)[0]
                                    for i in range(residual.numel())]).double()
            gradient = jacobian.t() @ residual.detach().double()
            if float(gradient.norm()) > ctx.stationarity:
                raise Exception(
                    'Swaption calibration: theta* is not stationary - ||J^T r|| is {:.6g} against a '
                    'Stationarity_Tol of {:.6g}, so the implicit function theorem does not hold '
                    'there'.format(float(gradient.norm()), ctx.stationarity))
            w = torch.linalg.pinv(jacobian.t() @ jacobian, hermitian=True,
                                  rtol=ctx.rcond) @ cotangent.double()
            grads = torch.autograd.grad(residual, calibration.quotes, retain_graph=True,
                                        grad_outputs=-(jacobian @ w).to(residual.dtype))
        return (None, None, None) + grads


class RiskNeutralInterestRateModel(object):
    def __init__(self, param, device, dtype):
        self.param = param
        self.num_batches = 1
        self.batch_size = 8192
        self.device = device
        self.prec = dtype
        #: What a block asking for `Quote_Sensitivity` leaves behind: theta* STILL CONNECTED to its
        #: quotes, one entry per named model parameter, and the quote leaf per block.
        #: `Config.bootstrap` harvests both - they are tensors, so they cannot live in
        #: `Price Factors`, which is data and gets written back out as JSON.
        self.calibrated = {}
        self.quote_leaves = {}

    def calc_loss_on_ir_curve(self, implied_params, base_date, time_grid, process,
                              implied_obj, ir_factor, vol_surface, resid=lambda x: x * x, jac=False):
        """The swaption calibration's residual closure: implied parameters in, one weighted relative
        pricing error per benchmark out, priced by brute-force Monte Carlo through the engine's own
        `pv_float_cashflow_list`.

        COMMON RANDOM NUMBERS ARE FROZEN PER SOLVE. The Sobol engine is built once, on the state
        this call creates - `reset` re-seeds nothing once `t_random_batch` exists - so every
        evaluation of the closure sees the same paths and the optimizer is differencing the
        parameters rather than the sample. What `reset` DOES clear is `t_Buffer` and `t_PreCalc`,
        which is the memo trap: those tables are keyed by factor and time, not by the tensor's
        identity, so a state carried across two parameter sets would answer the second call with
        the first call's curves.

        THE QUOTE SIDE severs at the market price and nowhere else. `swap.price` is a numpy scalar,
        built out of scipy by `create_market_swaps`, so the swaption vol behind it reaches the
        residual as a constant; `market_swap_class.error` is where the splice that closes it goes,
        and it is absent unless the block asked for `Quote_Sensitivity`. Three severances stay open
        deliberately, because their upstream is not a quote of THIS calibration: `get_par_swap_rate`
        prices the strike and the pvbp in numpy off the zero curve, `set_fixed_amount` writes that
        strike into the schedule's numpy half, and the ATM read interpolates the vol SURFACE with
        `RectBivariateSpline`. The first two are the calibrated curve, which is increment 1's quote;
        the third is the surface-node-to-ATM map, which is a quote of the surface rather than of the
        swaption.
        """

        def loss(implied_var):
            # first, reset the shared_mem
            shared_mem.reset(self.num_batches, numfactors, time_grid)
            # now set up the calc
            process.precalculate(base_date, time_grid, stoch_var, shared_mem, 0, implied_tensor=implied_var)
            tensor_swaptions = {}
            # needed to interpolate the zero curve
            delta_scen_t = np.diff(time_grid.scen_time_grid).reshape(-1, 1)

            for batch_index in range(self.num_batches):
                # load up the batch
                shared_mem.batch_index = batch_index
                # simulate the price factor - only need the full curve at the mtm time points
                shared_mem.t_Scenario_Buffer = process.generate(shared_mem)
                # get the discount factors
                Dfs = utils.calc_time_grid_curve_rate(
                    curve_index_reduced, time_grid.calc_time_grid(time_grid.scen_time_grid[:-1]),
                    shared_mem)
                # get the index in the deflation factor just prior to the given grid
                deflation = Dfs.reduce_deflate(delta_scen_t, time_grid.mtm_time_grid, shared_mem)
                # go over the instrument definitions and build the calibration
                for swaption_name, market_data in market_swaps.items():
                    expiry = market_data.deal_data.Time_dep.mtm_time_grid[
                        market_data.deal_data.Time_dep.deal_time_grid[0]]
                    DtT = deflation[expiry]
                    par_swap = pricing.pv_float_cashflow_list(
                        shared_mem, time_grid, market_data.deal_data,
                        pricing.pricer_float_cashflows, settle_cash=False)
                    sum_swaption = torch.sum(torch.relu(DtT * par_swap))
                    if swaption_name in tensor_swaptions:
                        tensor_swaptions[swaption_name] += sum_swaption
                    else:
                        tensor_swaptions[swaption_name] = sum_swaption

            calibrated_swaptions = {k: v / (self.batch_size * self.num_batches) for k, v in tensor_swaptions.items()}
            errors = {k: swap.error(calibrated_swaptions[k], resid)
                      for k, swap in market_swaps.items()}
            return calibrated_swaptions, errors

        # set up the stochastic factors
        stochastic_factors = {ir_factor: process}
        # calculate a reverse lookup for the tenors and store the daycount code
        all_tenors = utils.update_tenors(base_date, stochastic_factors)
        # calculate the curve indices
        index_keys = {'full': utils.Factor(ir_factor.type, ir_factor.name + ('full',)),
                      'reduced': utils.Factor(ir_factor.type, ir_factor.name + ('reduced',))}
        # calculate the tenor curve index
        c_index = instruments.calc_factor_index(ir_factor, {}, stochastic_factors, all_tenors)
        # now edit the curve indices with the correct names - one reduced, one full
        curve_index = [(c_index[utils.FACTOR_INDEX_Stoch], index_keys['full']) + c_index[2:]]
        curve_index_reduced = [(c_index[utils.FACTOR_INDEX_Stoch], index_keys['reduced']) + c_index[2:]]
        # set up a common context - we leave out the random numbers and pass it in explicitly below
        shared_mem = RiskNeutralInterestRate_State(index_keys, self.batch_size, self.device, self.prec)
        # calc the market swap rates and instrument_definitions - the unit tensor is what switches
        # the quote side on, and puts its leaves on the calculation's own device
        market_swaps, benchmarks = create_market_swaps(
            base_date, time_grid, curve_index, vol_surface, process.factor,
            implied_params['instrument']['Instrument_Definitions'], ir_factor.name,
            shared_mem.one if implied_params['instrument'].get(
                'Quote_Sensitivity', 'No') == 'Yes' else None)
        # number of random factors to use
        numfactors = process.num_factors()
        # the calibration swaps are compiled here rather than by a DealStructure, so they bind here
        for market_data in market_swaps.values():
            utils.bind_schedules(market_data.deal_data.Factor_dep, shared_mem.one)
        # set up the variables
        implied_var = {}
        stoch_var = torch.tensor(
            process.factor.current_value(), device=self.device, dtype=self.prec, requires_grad=jac)

        for param_name, param_value in implied_obj.current_value(include_quanto=jac).items():
            implied_var[param_name] = torch.tensor(
                param_value, dtype=self.prec, device=self.device, requires_grad=True)

        if jac:
            return stoch_var, implied_var, loss
        else:
            return implied_var, loss, market_swaps, benchmarks

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        base_date = sys_params['Base_Date']
        base_currency = sys_params['Base_Currency']
        master_curve_list = sys_params.get('Master_Curves')

        if sys_params.get('Swaption_Premiums') is not None:
            swaption_premiums = pd.read_csv(sys_params['Swaption_Premiums'], index_col=0)
            ATM_Premiums = swaption_premiums[swaption_premiums['Strike'] == 'ATM']
        else:
            ATM_Premiums = None

        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type == self.market_factor_type:
                # fetch the factors
                ir_factor = utils.Factor('InterestRate', rate[1:])
                vol_factor = utils.Factor('InterestYieldVol', utils.check_rate_name(
                    implied_params['instrument']['Swaption_Volatility']))

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    swaptionvol = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                    swaptionvol.delta = sys_params.get('Volatility_Delta', 0.0)
                    ir_curve = riskfactors.construct_factor(ir_factor, price_factors, factor_interp)
                    swaptionvol.set_premiums(ATM_Premiums, ir_curve.get_currency())
                except KeyError as k:
                    logging.warning('Missing price factor {} - Unable to bootstrap {}'.format(k.args, market_price))
                    continue
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                if master_curve_list and master_curve_list.get(ir_curve.get_currency()[0]) != rate[1]:
                    logging.warning('curve is not Risk Free {} - skipping and will reassign later'.format(market_price))
                    continue

                # set of dates for the calibration
                mtm_dates = set(
                    [base_date + x['Start'] for x in implied_params['instrument']['Instrument_Definitions']])

                # grab the implied process
                implied_obj, process, vol_tenors = self.implied_process(
                    base_currency, price_factors, price_models, ir_curve, rate)

                # set up the time grid
                time_grid = utils.TimeGrid(mtm_dates, mtm_dates, mtm_dates)
                # add a delta of 10 days to the time_grid_years (without changing the scenario grid
                # this is needed for stochastically deflating the exposure later on
                time_grid.set_base_date(base_date, delta=(10, vol_tenors * utils.DAYS_IN_YEAR))

                # calculate the error
                loss_fn, optimizers, implied_var, market_swaptions, benchmarks = self.calc_loss(
                    implied_params, base_date, time_grid, process, implied_obj, ir_factor, swaptionvol)

                if debug is not None:
                    debug.deals['Deals']['Children'] = [{'instrument': x} for x in benchmarks]
                    try:
                        debug.write_trade_file(market_factor.name[0] + '.aap')
                    except Exception:
                        logging.error('Could not write output file {}'.format(market_factor.name[0] + '.aap'))

                # check the time
                time_now = time.monotonic()
                calibration = SwaptionCalibration(
                    market_factor.name[0], loss_fn, implied_var, optimizers, process, market_swaptions)
                # the chain goes through the implicit-function wrapper either way: with no quotes
                # on the tape no edge is recorded and the wrapper is a pass-through, which is what
                # makes "gradients cannot move theta*" structural rather than a claim
                theta = LeastSquaresSolve.apply(
                    calibration,
                    float(implied_params['instrument'].get('Jacobian_Rcond', 1e-8)),
                    float(implied_params['instrument'].get('Stationarity_Tol', 1e-3)),
                    *calibration.quotes)

                # save this - `unflatten` detaches, so `Price Factors` gets plain numpy
                self.save_params(calibration.unflatten(theta), price_factors, implied_obj, rate)

                # the connected half, for the calculation that consumes these parameters: one entry
                # per named parameter under the key `_build_factor_state` mints its leaf with
                if calibration.quotes:
                    params_factor = utils.Factor(self.__class__.__name__, rate[1:])
                    self.calibrated.update({
                        utils.Factor(params_factor.type, params_factor.name + (name,)): value
                        for name, value in calibration.split(theta).items()})
                    # the optimizer chain called backward() on every evaluation it made, so `.grad`
                    # standing here is the sum over its whole path - six orders out, with a NaN in
                    # it. A calculation reports what IT accumulates, so the leaf is handed over clean
                    for quote in calibration.quotes:
                        quote.grad = None
                    self.quote_leaves[market_price] = (calibration.descriptors, calibration.quotes)

                # record the time
                logging.info('This took {} seconds.'.format(time.monotonic() - time_now))


class HullWhite2FactorModelParameters(RiskNeutralInterestRateModel):
    documentation = (
        'Interest Rates',
        ['A set of parameters $\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho$ are estimated from ATM',
         'swaption volatilities. Swaption volatilities are preferred to caplets to better estimate $\\rho$.'
         'Although assuming that $\\sigma_1, \\sigma_2$ are constant makes the calibration of this model',
         'considerably easier, in general, $\\sigma_1, \\sigma_2$ should be allowed a piecewise linear term',
         'structure dependent on the underlying swaptions.',
         '',
         'For a set of $J$ ATM swaptions, we need to minimize:',
         '',
         '$$E=\\sum_{j\\in J} \\omega_j (V_j(\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho)-V_j)^2$$',
         '',
         'Where $V_j(\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho)$ is the price of the $j^{th}$ swaption',
         'under the model, $V_j$ is the market value of the $j^{th}$ swaption and $ \\omega_j$ is the corresponding',
         'weight. The market value is calculated using the standard pricing functions',
         '',
         'To find a good minimum of the model value, basin hopping as implemented [here](https://docs.scipy.org/doc\
/scipy/reference/generated/scipy.optimize.basinhopping.html) as well as',
         'least squares [optimization](https://docs.scipy.org/doc/scipy/reference/generated/\
scipy.optimize.leastsq.html) are used.',
         '',
         'The error $E$ is algorithmically differentiated and then solved via brute-force monte carlo',
         'using tensorflow and scipy.',
         '',
         'If the currency of the interest rate is not the same as the base currency, then a quanto correction needs',
         'to be made. Assume $C$ is the value of the interest rate/FX correlation price factor (can be estimated from',
         'historical data), then the FX rate follows:',
         '',
         '$$d(log X)(t)=(r_0(t)-r(t)-\\frac{1}{2}v(t)^2)dt+v(t)dW(t)$$',
         '',
         'with $r(t)$ the short rate and $r_0(t)$ the short rate in base currency. The short rate with a quanto',
         'correction is:',
         '',
         '$$dr(t)=r_T(0,t)dt+\\sum_{i=1}^2 (\\theta_i(t)-\\alpha_i x_i(t)- \\bar\\rho_i\\sigma_i v(t))dt+\\sigma_i dW_i(t)$$',
         '',
         'where $W_1(t),W_2(t)$ and $W(t)$ are standard Wiener processes under the rate currency\'s risk neutral measure',
         'and $r_T(t,T)$ is the partial derivative of the instantaneous forward rate r(t,T) with respect to the maturity ',
         'date $T$.'
         '',
         'Define:',
         '',
         '$$F(u,v)=\\frac{\\sigma_1u+\\sigma_2v}{\\sqrt{\\sigma_1^2+\\sigma_2^2+2\\rho\\sigma_1\\sigma_2}}$$',
         '',
         'Then $\\bar\\rho_1, \\bar\\rho_2$ are assigned:',
         '',
         '$$\\bar\\rho_1=F(1,\\rho)C$$',
         '',
         '$$\\bar\\rho_2=F(\\rho,1)C$$',
         ]
    )

    market_factor_type = 'HullWhite2FactorModelPrices'
    fields = [
        F('Swaption_Volatility', 'Text', default=REQUIRED,
          description='The InterestYieldVol surface the benchmark swaptions are priced off'),
        F('Instrument_Definitions', 'Table', default='null', row=Row([
            F('Start', 'Period', description='Forward start, from the base date'),
            F('Tenor', 'Period', description='Swap tenor, from the start'),
            F('Floating_Frequency', 'Period'), F('Fixed_Frequency', 'Period'),
            F('Floating_Day_Count', 'Text',
              values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']),
            F('Fixed_Day_Count', 'Text',
              values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']),
            F('Market_Volatility', 'Percent',
              description='The quoted ATM vol; 0 reads the swaption surface'),
            F('Weight', 'Float', description='Relative weight in the objective')]),
          description='The forward starting swaps the swaptions are struck on'),
        F('Random_Seed', 'Integer', default=5120,
          description='Seeds the basin-hopping random search - the step taker and the Metropolis '
                      'accept test both draw from it. Without it the search draws from the process '
                      'global and the calibration is a function of whatever ran before it: on the '
                      'gate fixture theta* moves 0.93 absolute between ambient seeds. The Monte '
                      'Carlo paths are a separately frozen Sobol sample and do not move with it'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep each benchmark swaption connected to the quote it was priced off - the '
                      'row\'s Market_Volatility, the surface\'s ATM read, or the premium - so the '
                      'residual differentiates in the quote as well as in the model parameters. '
                      'The splice is worth exactly zero in the forward pass, so the calibrated '
                      'parameters are identical either way'),
        F('Jacobian_Rcond', 'Float', default=1e-8,
          description='Relative cutoff on the eigenvalues of the Gauss-Newton matrix J\'J when the '
                      'backward pass inverts it. J has one row per benchmark and 23 columns, so '
                      'that matrix is rank deficient on every block quoting fewer swaptions than '
                      'that and the inverse is a pseudo-inverse: below the cutoff a direction is '
                      'one the quotes do not identify and its dtheta/dq is the minimum-norm '
                      'representative. Only used when Quote_Sensitivity is Yes'),
        F('Stationarity_Tol', 'Float', default=1e-3,
          description='How far off stationarity theta* may be before the quote Jacobian is refused, '
                      'as the 2-norm of J\'r. The optimizer chain accepts whatever it returned - '
                      'possibly the seed, if nothing beat it - and the implicit function theorem '
                      'holds only where that gradient vanishes, so above this the backward raises '
                      'and names the norm rather than reporting a quietly wrong number'),
        F('Generate_Instruments', 'Text', default='No', values=['Yes', 'No'],
          description='Unbuilt: generate the definitions from Generation_Parameters instead'),
        F('Generation_Parameters', 'Container', default={
            'First_Start': '1Y', 'Last_Start': '9Y', 'First_Tenor': '1Y', 'Last_Tenor': '9Y',
            'First_Maturity': '10Y', 'Last_Maturity': '10Y', 'Fixed_Frequency': '6M',
            'Floating_Frequency': '6M', 'Day_Count': 'ACT_365', 'Index_Offset': 0},
          sub_fields=[
            F('First_Start', 'Period', default='1Y'), F('Last_Start', 'Period', default='9Y'),
            F('First_Tenor', 'Period', default='1Y'), F('Last_Tenor', 'Period', default='9Y'),
            F('First_Maturity', 'Period', default='10Y'),
            F('Last_Maturity', 'Period', default='10Y'),
            F('Fixed_Frequency', 'Period', default='6M'),
            F('Floating_Frequency', 'Period', default='6M'),
            F('Day_Count', 'Text', default='ACT_365',
              values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']),
            F('Index_Offset', 'Integer', default=0)],
          description='Unbuilt: the grid Generate_Instruments would sweep')
    ]

    def __init__(self, param, device, dtype):
        super(HullWhite2FactorModelParameters, self).__init__(param, device, dtype)
        self.sigma_bounds = (1e-5, 0.09)
        self.alpha_bounds = (-0.5, 2.4)
        self.corr_bounds = (-.95, 0.95)

    def calc_loss(self, implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface):

        def split_param(x):
            corr = x[2:3]
            alpha = x[:2]
            sigmas = x[3:]
            return sigmas, alpha, corr

        def make_basin_callbacks(step, sigma_min_max, alpha_min_max, corr_min_max, rng):
            """The two callbacks basin hopping needs, drawing from `rng` and never from the process
            global. A step taken off `np.random` makes the whole calibration a function of whatever
            ran before it in the same interpreter - measured on the gate fixture, theta* moves 0.93
            absolute between ambient seeds - and nothing raises, so the block declares the seed and
            the same generator serves the Metropolis test in `SwaptionCalibration.solve`."""

            def bounds_check(**kwargs):
                x = kwargs["x_new"]
                sigmas, alpha, corr = split_param(x)
                sigma_ok = (sigmas > sigma_min_max[0]).all() and (sigmas < sigma_min_max[1]).all()
                alpha_ok = (alpha > alpha_min_max[0]).all() and (alpha < alpha_min_max[1]).all()
                corre_ok = (corr > corr_min_max[0]).all() and (corr < corr_min_max[1]).all()
                return sigma_ok and alpha_ok and corre_ok and process.params_ok

            def basin_step(x):
                sigmas, alpha, corr = split_param(x)
                # update vars
                sigmas = (sigmas * np.exp(rng.uniform(-step, step, sigmas.size))).clip(*sigma_min_max)
                alpha = (alpha * np.exp(rng.uniform(-step, step, alpha.size))).clip(*alpha_min_max)
                corr = (corr + rng.uniform(-step, step, corr.size)).clip(*corr_min_max)

                return np.concatenate((alpha, corr, sigmas))

            return bounds_check, basin_step

        def make_basin_hopping_loss(loss_fn, implied_vars, device, with_grad=False):
            # makes it possible to call the scipy basinhopper
            def basin_hopper(x):
                for tn_var, np_var in zip(implied_vars.values(), np.split(x, split_param)):
                    tn_var.grad = None
                    tn_var.data = torch.from_numpy(np_var).to(device)

                try:
                    _, error = loss_fn(implied_vars)
                except Exception as e:
                    print("Warning x ({}) - {}".format(x, e.args))
                    return 100.0 * sum(len_vars), [100.0 * sum(len_vars)] * sum(len_vars)
                else:
                    total_loss = torch.sum(torch.stack(list(error.values())))
                    if with_grad:
                        total_loss.backward()
                        grad = torch.cat([x.grad for x in implied_vars.values()]).cpu().detach().numpy()
                        return total_loss.cpu().detach().numpy(), grad
                    else:
                        return total_loss.cpu().detach().numpy()

            len_vars = [len(x) for x in implied_vars.values()]
            split_param = np.cumsum(len_vars[:-1])
            return basin_hopper

        def make_least_squares_loss(loss_fn, implied_vars, device):
            # makes it possible to call the scipy least squares algo
            def calc_loss(x):
                for tn_var, np_var in zip(implied_vars.values(), np.split(x, split_param)):
                    tn_var.grad = None
                    tn_var.data = torch.from_numpy(np_var).to(device)
                _, error = loss_fn(implied_vars)
                return torch.stack(list(error.values()))

            def jacobian(x):
                loss = calc_loss(x)
                # full jacobian - takes a second or so
                jac = torch.stack([torch.cat(torch.autograd.grad(
                    loss, list(implied_vars.values()), x, retain_graph=True))
                    for x in torch.eye(len(loss), device=device)])
                return jac.cpu().numpy()

            def least_squares(x):
                return calc_loss(x).cpu().detach().numpy()

            len_vars = [len(x) for x in implied_vars.values()]
            split_param = np.cumsum(len_vars[:-1])
            return least_squares, jacobian

        # get the swaption error and market values
        implied_var_dict, loss_fn, market_swaptions, benchmarks = self.calc_loss_on_ir_curve(
            implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface)

        bounds = []
        for k, v in implied_var_dict.items():
            if k.startswith('Alpha'):
                bounds.append([self.alpha_bounds])
            elif k == 'Correlation':
                bounds.append([self.corr_bounds])
            else:
                bounds.append([self.sigma_bounds] * len(v))

        var_to_bounds = np.vstack(bounds)
        # ONE generator for the whole random search - the step taker draws from it here and basin
        # hopping's own Metropolis test draws from it in `solve`, which is the single stream the
        # process global used to be
        rng = np.random.RandomState(int(implied_params['instrument'].get('Random_Seed', 5120)))
        bounds_ok, make_step = make_basin_callbacks(
            0.125, self.sigma_bounds, self.alpha_bounds, self.corr_bounds, rng)

        basin_hopper_fn_grad = make_basin_hopping_loss(loss_fn, implied_var_dict, self.device, True)
        x0 = torch.cat(list(implied_var_dict.values())).cpu().detach().numpy()
        lsq_fn, jacobian = make_least_squares_loss(loss_fn, implied_var_dict, self.device)

        optimizers = [('basin', x0, basin_hopper_fn_grad, make_step, bounds_ok, var_to_bounds, rng),
                      ('leastsq', x0, lsq_fn, jacobian, list(zip(*var_to_bounds)))]

        return loss_fn, optimizers, implied_var_dict, market_swaptions, benchmarks

    def implied_process(self, base_currency, price_factors, price_models, ir_curve, rate):
        vol_tenors = np.array([0, 1, 3, 6, 12, 24, 48, 72, 96, 120]) / 12.0
        # construct an initial guess - need to read from params
        param_name = utils.check_tuple_name(
            utils.Factor(type=self.__class__.__name__, name=rate[1:]))

        # check if we need a quanto fx vol
        fx_factor = utils.Factor('GBMAssetPriceTSModelParameters', ir_curve.get_currency())
        ir_factor = utils.Factor('InterestRate', ir_curve.get_currency())
        fx_factor_name = utils.check_tuple_name(fx_factor)
        ir_factor_name = utils.check_tuple_name(ir_factor)

        if fx_factor_name in price_factors:
            quanto_fx = price_factors[fx_factor_name]['Vol']
            curr_pair = sorted((base_currency,) + ir_curve.get_currency())
            correlation_name = 'Correlation.FxRate.{}/{}'.format('.'.join(curr_pair), ir_factor_name)
            # check if the quote is against the base currency
            sign = 1.0
            if curr_pair[0] == base_currency:
                sign = -1.0
                logging.info('Reversing Correlation as {} is quoted against the base currency'.format(correlation_name))
            # the correlation between fx and ir - needed to establish Quanto Correlation 1 and 2
            C = sign * price_factors.get(correlation_name, {'Value': 0.0})['Value']
        else:
            C = None
            quanto_fx = None

        if param_name in price_factors:
            param = price_factors[param_name]
            implied_obj = riskfactors.HullWhite2FactorModelParameters(
                {'Quanto_FX_Volatility': quanto_fx,
                 'short_rate_fx_correlation': C,
                 'Alpha_1': np.clip(param['Alpha_1'], *self.alpha_bounds),
                 'Alpha_2': np.clip(param['Alpha_2'], *self.alpha_bounds),
                 'Correlation': np.clip(param['Correlation'], *self.corr_bounds),
                 'Sigma_1': utils.Curve([], list(zip(
                     vol_tenors, np.interp(vol_tenors, *param['Sigma_1'].array.T).clip(*self.sigma_bounds)))),
                 'Sigma_2': utils.Curve([], list(zip(
                     vol_tenors, np.interp(vol_tenors, *param['Sigma_2'].array.T).clip(*self.sigma_bounds))))})
        else:
            implied_obj = riskfactors.HullWhite2FactorModelParameters(
                {'Quanto_FX_Volatility': quanto_fx,
                 'short_rate_fx_correlation': C,
                 'Alpha_1': 0.1, 'Alpha_2': 0.1, 'Correlation': 0.01,
                 'Sigma_1': utils.Curve([], list(zip(vol_tenors, [0.01] * vol_tenors.size))),
                 'Sigma_2': utils.Curve([], list(zip(vol_tenors, [0.01] * vol_tenors.size)))})

        # need to create a process and params as variables to pass to tf
        process = stochasticprocess.HullWhite2FactorImpliedInterestRateModel(
            ir_curve, {'Lambda_1': 0.0, 'Lambda_2': 0.0}, implied_obj)

        return implied_obj, process, vol_tenors

    def save_params(self, vars, price_factors, implied_obj, rate):
        param_name = utils.check_tuple_name(
            utils.Factor(type=self.__class__.__name__, name=rate[1:]))
        # grab the sigma tenors
        sig1_tenor, sig2_tenor = implied_obj.get_vol_tenors()
        # store the basic paramters
        param = {'Property_Aliases': None,
                 'Quanto_FX_Volatility': None,
                 'Alpha_1': float(vars['Alpha_1'][0]),
                 'Sigma_1': utils.Curve([], list(zip(sig1_tenor, vars['Sigma_1']))),
                 'Alpha_2': float(vars['Alpha_2'][0]),
                 'Sigma_2': utils.Curve([], list(zip(sig2_tenor, vars['Sigma_2']))),
                 'Correlation': float(vars['Correlation'][0])}

        # grab the quanto fx correlations
        quanto_fx1, quanto_fx2 = implied_obj.get_quanto_correlation(
            vars['Correlation'], [vars['Sigma_1'], vars['Sigma_2']])

        if quanto_fx1 is not None and quanto_fx2 is not None:
            param.update({
                'Quanto_FX_Volatility': implied_obj.param['Quanto_FX_Volatility'],
                'Quanto_FX_Correlation_1': quanto_fx1,
                'Quanto_FX_Correlation_2': quanto_fx2})

        price_factors[param_name] = param
        # return the final implied object
        return riskfactors.HullWhite2FactorModelParameters(param)


def leaf_deals(node):
    """The deals a deal-tree node prices - itself, or its children if it is a container."""
    if node.get('Children'):
        return [leaf for child in node['Children'] for leaf in leaf_deals(child)]
    return [node['Instrument']]


class Benchmark_State(utils.Calculation_State):
    """The pricing state a t0 benchmark valuation needs: one date, one path, float64.

    `t_Static_Buffer` is the point of it. That dict is where every pricer reads a static curve
    from, so putting a `requires_grad` tensor in it is what puts the curve's nodes on the tape -
    there is no other seam that reaches the pricers without a numpy round trip in between. It is
    built fresh per evaluation because `t_Buffer` is a memo table keyed by `(stoch, Factor)` and
    not by the tensor's identity, so a reused state answers the second call with the first call's
    curves.

    Boundary registration is off: a deposit, an FRA and a swap leg take no decision on simulated
    state, so there is nothing for the correction to carry.
    """

    def __init__(self, static_buffer, one, report_currency):
        super(Benchmark_State, self).__init__(
            static_buffer, one, 1, report_currency, 'Constant', 1, False)
        self.boundary_aad = False
        self.boundary_sets = []


class BenchmarkInstruments(object):
    """The benchmark instruments of one curve solve, compiled once and priced at t0 off curve node
    TENSORS - so `torch.autograd.grad(pv, theta)` is the calibration Jacobian's row.

    A benchmark is a deal-tree NODE, `{'Instrument': deal, 'Children': [...]}`, exactly as
    `Trade Data` authors it: a deposit or an FRA is one deal, a par swap is one `SwapInterestDeal`,
    and an OIS swap is a container over an OIS-compounded floating leg and a fixed leg. Its PV is
    the sum of its leaves' PVs, each already converted to the reporting currency by its own
    `pv_*_leg`; there is no netting or collateral rule to apply on top, which is what lets this
    stay out of `DealStructure`.

    **The graph audit.** The factor-construction path severs autograd in four places, and every one
    of them is on the way IN to `t_Static_Buffer` rather than on the way out:

    - `Calculation._build_factor_state` and `Base_Revaluation.update_factors` mint every leaf as
      `torch.tensor(factor.current_value(), requires_grad=...)`. That is a fresh leaf built from a
      numpy array, so anything upstream of it is severed by construction. This class writes theta
      straight into the buffer instead and never calls `current_value` for a curve it is solving.
    - `riskfactors.Factor1D.current_value` is numpy end to end, and `Factor1D.get_tenor` REWRITES
      `param['Curve'].array` (dedupe plus `np.interp`) as a side effect of construction - so the
      node order theta is indexed by is the rewritten one, read back off the constructed factor.
    - `Factor1D.check_interpolation` precomputes the Hermite `(g, c)` pair from the numpy rate
      column. Those coefficients are constants in theta, and the pricing path does not use them:
      `utils.Interpolation.build` re-derives the pair from the buffer TENSOR, and `all_tenors`
      carries only the interpolation KIND and the tenor grid. A Hermite curve differentiates.
    - `utils.TensorSchedule.bind` mints the cashflow schedule's tensor half with `new_tensor`,
      which is where the QUOTE - a fixed rate, a margin - stops being differentiable. Severed until
      `TensorSchedule.carry` gave the tensor half an overlay; `_carry_quotes` builds it, and only
      then is the residual differentiable in `q` as well as in theta.

    One trap that is not a severance: `utils.CurveTenor` caches its tenor grid as a tensor built
    from the first tensor that queries it. `all_tenors` is rebuilt per instance here, so a float64
    solve cannot inherit a float32 grid from whatever ran before it.

    `quotes` and `bumped_nodes` are the quote side, and they come as a pair: the quotes the set was
    authored at, in percent, and the SAME set authored one percent higher. The second is what says
    which schedule columns the quote writes - see `_carry_quotes`.
    """

    #: The solve is float64 whatever the simulation runs in - a bootstrap that converges to 1e-10
    #: cannot be done in float32, and the Jacobian it hands the implicit-function theorem is only
    #: as good as the residual it came from.
    dtype = torch.float64

    def __init__(self, nodes, price_factors, factor_interp, base_date, currency, calendars,
                 solve_for, device, quotes=None, bumped_nodes=None):
        # `config` imports `construct_bootstrapper` from this module, so the module-level edge runs
        # one way only and discovery is reached from inside the call
        from .config import Config

        cfg = Config(base_currency=currency)
        cfg.params['Price Factors'] = price_factors
        cfg.params['Price Factor Interpolation'] = factor_interp
        cfg.params['System Parameters']['Base_Date'] = base_date
        cfg.holidays = calendars
        cfg.set_calculation_children(nodes)
        # the engine's own discovery, so the benchmark set pulls exactly the factors a valuation
        # would. Single currency by construction: reporting IS the curve's currency, which makes
        # every `calc_fx_cross` the identity
        dependent_factors, _, _, _ = cfg.discover_factors(
            {'Currency': currency}, base_date, '0d', False)

        self.factors = {factor: riskfactors.construct_factor(
            factor, price_factors, factor_interp, base_date=base_date) for factor in dependent_factors}
        self.solve_for = tuple(solve_for)
        # the knot grid theta is indexed by, read off the factor AFTER `get_tenor` has rewritten it
        self.tenors = {factor: self.factors[factor].tenors for factor in self.solve_for}
        self.all_tenors = utils.update_tenors(base_date, self.factors)
        self.time_grid = utils.TimeGrid({base_date}, {base_date}, {base_date})
        self.time_grid.set_base_date(base_date)
        self.time_grid.set_report_dates(base_date, {base_date})
        self.one = torch.ones([1, 1], dtype=self.dtype, device=device)
        self.report_currency = instruments.get_fxrate_factor(
            utils.check_rate_name(currency), self.factors, {})
        # every factor the solve is NOT solving for is a constant of it
        self.constants = {factor: torch.tensor(
            obj.current_value(), dtype=self.dtype, device=device)
            for factor, obj in self.factors.items()
            if factor.type not in utils.DimensionLessFactors and factor not in self.solve_for}

        self.benchmarks = [[self._compile(leaf, base_date, calendars) for leaf in leaf_deals(node)]
                           for node in nodes]
        self.quotes = None if quotes is None else torch.tensor(
            quotes, dtype=self.dtype, device=device, requires_grad=True)
        if self.quotes is not None:
            self._carry_quotes(bumped_nodes, base_date, calendars)
        # the benchmark set is compiled OUTSIDE a calculation, so it binds its own schedules - and
        # it binds them last, because the quote overlay is spliced into the copy `bind` makes
        for legs in self.benchmarks:
            for leg in legs:
                utils.bind_schedules(leg.Factor_dep, self.one)

    def _compile(self, deal, base_date, calendars):
        """One leaf deal's compiled form - the same `Factor_dep` / `Time_dep` pair a valuation
        builds, on a grid holding the base date alone."""
        return utils.DealDataType(
            Instrument=deal,
            Factor_dep=deal.calc_dependencies(
                base_date, self.factors, {}, self.factors, self.all_tenors, self.time_grid, calendars),
            Time_dep=self.time_grid.calc_deal_grid({base_date}),
            Calc_res=None)

    def _carry_quotes(self, bumped_nodes, base_date, calendars):
        """Put the quote leaf on every schedule column the quote WRITES.

        Which columns those are is MEASURED, not declared: the same benchmark set authored one
        percent higher is compiled, and the columns that moved are the value columns with the
        difference as their slope. The authoring map is affine in the quote - a percent scaled into
        a fixed-rate column, a margin - so one bumped compile IS the derivative rather than a
        difference quotient of one. That keeps `QUOTE_WRITERS` the only place a quotable instrument
        is declared: where its number lands is read off it, never restated beside it.

        The splice is `base + (q - q.detach()) * slope` - the boundary correction's shape, worth
        exactly zero in the forward pass with derivative one. So the tensor half is bit-identical
        to the copy `new_tensor` would have made and enabling quote gradients cannot move the
        solve. It is a derivative carrier and NOT a reparameterisation: the value does not follow a
        moved quote, because the pricers memoize payment tensors off the schedule the same way
        `t_Buffer` memoizes curves, so a different quote needs a fresh closure.

        Resets carry no overlay. No increment-1 quote reaches one - a deposit's pinned rate, an
        FRA's margin and a fixed leg's rate all land in the cashflow schedule - and a reset value
        also leaves through `known_resets`, which reads numpy. Measured rather than assumed: a
        moved reset column raises here.
        """
        for index, (legs, node) in enumerate(zip(self.benchmarks, bumped_nodes)):
            delta = self.quotes[index] - self.quotes[index].detach()
            # the bumped set never went through discovery, and discovery is what resets a deal
            bumped_legs = leaf_deals(node)
            for leaf in bumped_legs:
                leaf.reset(calendars)
            for leg, plus in zip(legs, [self._compile(leaf, base_date, calendars)
                                        for leaf in bumped_legs]):
                for name, schedule in leg.Factor_dep.items():
                    if not isinstance(schedule, utils.TensorCashFlows):
                        continue
                    bumped = plus.Factor_dep[name]
                    if schedule.Resets is not None and (
                            bumped.Resets.schedule != schedule.Resets.schedule).any():
                        raise Exception(
                            'Curve bootstrap: {} writes its quote into a RESET column, which the '
                            'schedule overlay does not reach'.format(name))
                    moved = bumped.schedule - schedule.schedule
                    columns = np.flatnonzero(np.abs(moved).max(axis=0))
                    if columns.size:
                        schedule.carry({int(column): self._column(schedule.schedule[:, column]) +
                                        delta * self._column(moved[:, column]) for column in columns})

    def _column(self, values):
        return torch.tensor(values, dtype=self.dtype, device=self.one.device)

    def reads(self, theta):
        """The factors OUTSIDE `solve_for` this residual ACTUALLY reads, measured rather than
        declared - the coupling detector a multi-curve set is grouped by.

        `_carry_quotes`'s idiom applied to the other side of the residual: instead of asking a block
        what it says it discounts on, make every constant a leaf, evaluate the residual once and see
        which ones a backward pass reaches. One residual and one backward, so it costs a fraction of
        a Newton iteration - and it catches the coupling a `Discount_Rate` field cannot state,
        because what a benchmark PROJECTS off is authored inside its own deal block. A block whose
        `Discount_Rate` is blank but whose swaps forecast off another curve reads that curve, and
        the declaration says self-discounting.

        A residual with no graph at all reads nothing, which is the self-discounting single-curve
        case: its only curve is the one being solved for and every constant is unreachable.
        """
        constants = list(self.constants)
        with torch.enable_grad():
            for factor in constants:
                self.constants[factor].requires_grad_(True)
            residual = self(theta)
            gradients = torch.autograd.grad(
                residual.sum(), [self.constants[factor] for factor in constants],
                allow_unused=True) if residual.requires_grad else [None] * len(constants)
            for factor in constants:
                self.constants[factor].requires_grad_(False)
        return {factor for factor, gradient in zip(constants, gradients)
                if gradient is not None and gradient.abs().max() > 0}

    def __call__(self, theta):
        """The benchmark PV vector at curve nodes `theta`, a `{Factor: tensor}` over `solve_for`."""
        shared = Benchmark_State({**self.constants, **theta}, self.one, self.report_currency)
        # one date and one path, so a leg's PV is a scalar - `reshape` says so and fails loud
        return torch.stack([
            sum(leg.Instrument.generate(shared, self.time_grid, leg).reshape(()) for leg in legs)
            for legs in self.benchmarks])


def damped_newton(residual, theta, n_iter, tol, halvings):
    """Solve `residual(theta) = 0` for a `{Factor: tensor}` of curve nodes, in float64.

    The curves are flattened into ONE system, so a projection curve solved against a discount curve
    in the same call is a single Jacobian rather than two coupled ones. That Jacobian comes from
    autograd on the residual - one backward pass per benchmark gives the whole row - which is the
    same derivative the implicit function theorem needs on the other side, so the residual is
    written once and differentiated twice.

    Damping is a backtracking line search on the max-norm of the residual: full step first, halved
    until it decreases. Newton takes the full step everywhere near the root, so the damping is
    insurance against a bad first iterate rather than something the converged path exercises.

    The three knobs are the caller's, not this function's: they are declared fields of the block
    being solved, so a job tightens or loosens the solve without a code edit.
    """
    keys = list(theta)
    sizes = [theta[key].numel() for key in keys]

    def unflatten(flat):
        return dict(zip(keys, flat.split(sizes)))

    x = torch.cat([theta[key].detach() for key in keys])
    for iteration in range(n_iter):
        x = x.detach().requires_grad_(True)
        f = residual(unflatten(x))
        jacobian = torch.stack([torch.autograd.grad(f[i], x, retain_graph=True)[0]
                                for i in range(f.numel())])
        step = torch.linalg.solve(jacobian, f.detach())

        # convergence is tested on the step BEFORE the line search, not after it: a step this small
        # is inside the linear solve's own rounding, and asking a residual already at noise level
        # to decrease again is a test nothing passes
        if step.abs().max() <= tol:
            return unflatten((x - step).detach())

        norm = f.detach().abs().max()
        damping = 1.0
        for _ in range(halvings + 1):
            trial = x.detach() - damping * step
            if residual(unflatten(trial)).abs().max() < norm:
                break
            damping *= 0.5
        else:
            raise Exception('Curve bootstrap: no damped Newton step reduces the residual '
                            '(iteration {}, residual {:.6g})'.format(iteration, float(norm)))
        x = trial

    raise Exception('Curve bootstrap: {} Newton iterations without converging'.format(n_iter))


def split_theta(benchmarks, theta):
    """The flat solved vector back as the `{Factor: nodes}` the residual takes, in `solve_for`
    order - which is the order `CalibrationSolve` concatenated it in."""
    sizes = [benchmarks.tenors[factor].size for factor in benchmarks.solve_for]
    return dict(zip(benchmarks.solve_for, theta.split(sizes)))


def residual_jacobians(benchmarks, theta):
    """The residual at `theta` and both its Jacobians: `dF/dtheta` (n x n) and `dF/dq` (n x m).

    ONE backward pass per benchmark gives both, off a forward pass the residual itself comes out of.
    The quote side is another output of the pass the theta side already needs, so materialising the
    whole `dF/dq` costs nothing over contracting one cotangent through it - which is why
    `CalibrationSolve.backward`, the artifact's calibration Jacobian and its drift metric read the
    same function rather than each writing the derivative out again.

    Every `grad` retains the graph: the residual's subgraph is shared with the forward pass -
    `pv_fixed_cashflows` memoizes its payment tensor in `Factor_dep` - so freeing it would take the
    forward pass's graph with it. A quote that writes into no schedule column raises here rather
    than reporting a zero row, which is the same failure `_carry_quotes` refuses to guess at.
    """
    x = torch.cat([theta[factor] for factor in benchmarks.solve_for]).detach().requires_grad_(True)
    residual = benchmarks(split_theta(benchmarks, x))
    rows = [torch.autograd.grad(residual[i], [x, benchmarks.quotes], retain_graph=True)
            for i in range(residual.numel())]
    return (residual, torch.stack([row[0] for row in rows]),
            torch.stack([row[1] for row in rows]))


def calibration_jacobian(benchmarks, theta):
    """`dtheta/dq` at the fixed point.

    The implicit function theorem in its MATRIX form - `dtheta/dq = -(dF/dtheta)^-1 (dF/dq)` - which
    is `CalibrationSolve.backward`'s own arithmetic with every cotangent solved at once instead of
    one. There is NO SECOND SOLVE: the fixed point is where the forward pass left it and only the
    residual is differentiated, so this costs one Newton iteration's worth of work whatever m is.

    `dF/dtheta` has to be invertible, which is a ROOT FIND's property and not this family's alone -
    a least-squares fixed point would contract a pseudo-inverse here instead. `J` itself is n x m
    and the artifact does not assume the two are equal, even though the knot rule makes them so.

    Over a COUPLED SET this is the whole block matrix: `solve_for` holds every curve of the set, so
    `dF/dtheta` carries the cross terms the residual reads and `dtheta_2/dq_1` falls out of the one
    inverse. That is the difference between an operator that carries a multi-curve tick and one that
    silently drops a first-order term - see `coupled_sets`.
    """
    with torch.enable_grad():
        _, d_theta, d_quote = residual_jacobians(benchmarks, theta)
    return -torch.linalg.solve(d_theta, d_quote)


class CalibrationSolve(torch.autograd.Function):
    """The bootstrap as one differentiable node: quotes in, calibrated nodes out.

    FORWARD IS THE ORDINARY SOLVE. It calls `damped_newton` and nothing else - same iterations,
    same tolerances, same float64 - so enabling quote gradients cannot move a mark by construction
    rather than by a claim anyone has to check. Autograd runs `forward` with grad mode off, which
    the solve needs on for its own Jacobian, so it is re-enabled here and the graph the iteration
    builds is discarded with the iteration.

    BACKWARD IS THE IMPLICIT FUNCTION THEOREM, never an unrolled solver. At the fixed point
    `F(theta*, q) = 0`, so `dtheta/dq = -(dF/dtheta)^-1 (dF/dq)` and a cotangent `v = dL/dtheta*`
    contracts to

        w = (dF/dtheta)^-T v      then      dL/dq = -(dF/dq)^T w

    Both come from `residual_jacobians`, autograd on the residual closure itself evaluated once at
    `(theta*, q)` - one backward pass per benchmark, giving a row of each. So the residual is
    WRITTEN ONCE AND DIFFERENTIATED TWICE, and the quote derivative cannot drift from the one the
    solve converged on; nor can it drift from the `dtheta/dq` a calibration artifact publishes,
    which is the same two pieces with every cotangent solved at once.

    The Jacobian is recomputed at theta* rather than reused from the last Newton step, which was
    taken at the iterate BEFORE it. Cost is one iteration's worth on a system whose dimension is
    the benchmark count.
    """

    @staticmethod
    def forward(ctx, benchmarks, seed, n_iter, tol, halvings, quotes):
        with torch.enable_grad():
            theta = damped_newton(benchmarks, seed, n_iter, tol, halvings)
        ctx.benchmarks, ctx.theta = benchmarks, theta
        return torch.cat([theta[factor] for factor in benchmarks.solve_for])

    @staticmethod
    def backward(ctx, cotangent):
        with torch.enable_grad():
            _, d_theta, d_quote = residual_jacobians(ctx.benchmarks, ctx.theta)
            w = torch.linalg.solve(d_theta.t(), cotangent)
        return None, None, None, None, None, -(d_quote.t() @ w)


class CalibrationArtifact(object):
    """ONE CALIBRATION OF ONE COUPLED SET, FROZEN AS AN OPERATOR - `(theta*, J, q0, timestamp)` and
    the compiled benchmark set the first two were read off.

    THE CONTRACT. `theta*` is the solved node vector in `solve_for` order, `J` is `dtheta/dq` at
    that fixed point from `calibration_jacobian` (exact by the implicit function theorem - the
    curve family's solve is a unique root and the knot rule makes `J` square, though nothing here
    assumes it is), and `q0` is the quote vector it was fitted at in percent. Between two fits, a
    small tick propagates LINEARLY: `theta ~ theta* + J (q_now - q0)`, one matvec.

    IT COVERS THE SET, NOT THE BLOCK. `members` are the `Market Prices` blocks that solve as ONE
    system, in the order their quotes and nodes are concatenated in, and `J` is the whole block
    matrix - so `dtheta_2/dq_1` is a column of it rather than a term nobody carried. A partial ride
    is unrepresentable: there is one theta, one q0 and one drift number for the set, and
    `coupled_sets` refuses to publish an operator over part of one.

    `timestamp` is when the fit happened, and it is REPORTED rather than read: every ride, every
    refusal and every refit names the artifact it rode and when that artifact was fitted, because
    "how stale" is the question the drift number is an answer to. It reaches no number and no hash,
    so a wall clock cannot make two runs disagree.

    IT IS PLAN-SIDE AND CONTENT-ADDRESSED. `key` is the SLOT - every member block's declarations,
    the base date, the interpolation scheme and the engine version, with the quote NUMBERS shadowed
    out (`plan_key`), so every tick of one strip lands on the same slot and a re-authored strip
    addresses a different one. `artifact_id` is the artifact's OWN identity - the slot plus the
    quotes it was fitted at - so it MOVES with every refit, is REPORTED in the results of every run
    that rides it, and is a replay coordinate rather than a timestamp anyone has to trust.

    NOTHING HERE MUTATES. There is no `theta_current`: `ride` is a pure function of this artifact
    and the quotes it is handed, evaluated per EXECUTE and stored nowhere, so two EXECUTEs off one
    `(artifact, q_now)` are bit-identical by construction. The artifact is replaced, never edited -
    a refit publishes a new one into the same slot under a new `artifact_id`.

    It holds TENSORS and a compiled deal tree, so it cannot live in `Price Factors` and it cannot
    be serialised: it lives in `ARTIFACTS`, in process, beside the plan cache. A cold start has no
    artifact and the first tick REFUSES rather than pricing something else - which is the honest
    statelessness, since a re-bootstrap rederives the artifact from the job document and a plan
    that is not in the cache is a 404 rather than a different number.
    """

    def __init__(self, key, members, theta, jacobian, quotes, benchmarks, drift=None):
        # `config` imports from this module, so the package edge runs one way only and the hash is
        # reached from inside the call
        from . import content_hash

        self.key = key
        self.members = tuple(members)
        self.theta = theta
        self.jacobian = jacobian
        self.quotes = quotes
        self.benchmarks = benchmarks
        self.drift = drift
        self.timestamp = pd.Timestamp.utcnow()
        self.artifact_id = content_hash({'key': key, 'quotes': quotes.tolist()})

    @property
    def factors(self):
        """The curves this operator carries, in the order `theta` concatenates them."""
        return self.benchmarks.solve_for

    @property
    def jacobian_norm(self):
        """`||J||inf`, the induced max-row-sum norm - the CONVERSION between the quote-space units
        `Drift_Tolerance` is declared in and the curve units a desk reads a staleness in, since
        `||theta_ridden - theta_refit||inf <= ||J||inf ||r||inf` to first order."""
        return float(self.jacobian.abs().sum(dim=1).max())

    def ride(self, quotes):
        """`theta* + J (q_now - q0)` - the operator. Pure, and a matvec."""
        return self.theta + self.jacobian @ (quotes - self.quotes)

    def nodes(self, theta, factor):
        """One member curve's slice of a set-wide theta, as the numpy column a price factor is."""
        return split_theta(self.benchmarks, theta)[factor].detach().cpu().numpy()

    def mispricing(self, theta, quotes):
        """Every benchmark's residual at `(theta, quotes)`, IN QUOTE SPACE: the move in that
        benchmark's own quote, in percent, that would close it. EXACT, at any theta and any quote.

        The set was compiled at `q0`, so pricing it at a moved quote would need a re-authoring and a
        re-compile - a refit's cost, and a drift gate that costs a refit is pointless. It does not
        need one: a benchmark's PV is AFFINE in its own quote at fixed theta (measured in
        `_carry_quotes`, second difference exactly zero), so

            F(theta, q) = F(theta, q0) + (dF/dq)(q - q0)

        holds with no remainder - PROVIDED `dF/dq` is taken at the theta being scored. It is:
        `residual_jacobians` re-differentiates the residual HERE, at the ridden theta, one backward
        pass per benchmark off the forward pass the residual itself comes out of. That is 71.7ms
        against a 594ms refit on the ZAR strip - 97% of what a ride costs, and what makes this a
        measurement rather than an estimate.

        Reusing the `dF/dq` stored at `theta*` would be cheaper by one backward and WRONG in the
        direction that matters: its miss is `(d2F/dtheta dq) dtheta dq`, the same order as the
        residual it estimates, and it reads LOW on tick shapes that excite the Jacobian's small
        singular values - 0.886 of the truth at worst over a scan of sign patterns, which admits a
        ride the tolerance was written to refuse.

        Dividing each row by its own quote sensitivity is what makes `Drift_Tolerance` a number a
        desk can set: a PV depends on the benchmark's notional and a quote does not. It is the same
        normaliser the self-delta identity uses, and it is a row max rather than a diagonal so that
        a family whose benchmarks are not one-quote-each stays expressible.
        """
        with torch.enable_grad():
            residual, _, d_quote = residual_jacobians(
                self.benchmarks, split_theta(self.benchmarks, theta))
        d_quote = d_quote.detach()
        return ((residual.detach() + d_quote @ (quotes - self.quotes)) /
                d_quote.abs().amax(dim=1))


class ArtifactStore(object):
    """Calibration artifacts under their plan keys - the PlanCache's discipline for the other half
    of a prepared job.

    Bounded and least-recently-used for the reason the plan cache is: an artifact is a refit, and
    never the record of anything - the replay tuple is. Locked because a slot is written by
    whatever thread ran the bootstrap and read by whatever thread runs the EXECUTE.

    Content-addressed, so an entry is IMMUTABLE under its key: a refit REPLACES the artifact in a
    slot rather than editing one. A moved quote NUMBER keeps the slot - that is the whole point of
    `plan_key`, and what makes a ride possible - while a re-authored quote SET addresses a
    different one and finds it empty.

    `covering` is the lookup a valuation needs, because a valuation holds a FACTOR and not a set. It
    returns CANDIDATES - every artifact holding that curve, most-recently-used first - and never
    picks one: the caller recomputes each candidate's slot off the market data standing now and
    takes the one that still addresses it, so content addressing decides and the scan only narrows.
    Two artifacts can cover one curve at once (a Hermite job and a linear one), and a lookup that
    returned the first would hide the second behind it.

    Scanning a bounded store rather than keeping a factor index is deliberate - an index is a thing
    that can disagree with the store it indexes, and the store is 32 entries.
    """

    def __init__(self, size=32):
        self.size = size
        self.artifacts = OrderedDict()
        self.lock = threading.Lock()

    def put(self, artifact):
        with self.lock:
            self.artifacts[artifact.key] = artifact
            self.artifacts.move_to_end(artifact.key)
            if len(self.artifacts) > self.size:
                self.artifacts.popitem(last=False)
            return artifact.key

    def get(self, key):
        with self.lock:
            if key not in self.artifacts:
                return None
            self.artifacts.move_to_end(key)
            return self.artifacts[key]

    def covering(self, factor):
        with self.lock:
            return [self.artifacts[key] for key in reversed(self.artifacts)
                    if factor in self.artifacts[key].factors]


#: Where a published calibration artifact lives - in process, beside the service's plan cache and
#: for the same reasons. It holds tensors and a compiled benchmark set, so `Price Factors` (which
#: is data, and gets written back out as JSON) is not an option and neither is a file.
ARTIFACTS = ArtifactStore()


def quote_nodes(points, discount_rate, shift=0.0):
    """The used quotes as deal-tree nodes, each authored at its own quote plus `shift` percent.

    Deep-copied because authoring WRITES the quote and the discount curve into the block, and the
    market data it came out of is data.
    """
    nodes = []
    for point in points:
        authored = dict(copy.deepcopy(point['Deal']), Object=point['DealType'])
        author_quote(authored, point['Quoted_Market_Value'] + shift, discount_rate)
        nodes.append(quote_node(authored, {}))
    return nodes


def _pin_deposit_schedule(deal, quote):
    """A deposit has no rate field of its own. Pinning every accrual start is what makes it price
    as a fixed leg, which is also what keeps it off the forecast curve the solve is building -
    `DepositDeal.reset` drops that dependency when the schedule covers every start."""
    starts = instruments.generate_dates_backward(
        deal['Maturity_Date'], deal['Effective_Date'], deal['Payment_Frequency'])[:-1]
    deal['Interest_Rate_Schedule'] = utils.DateList({date: quote for date in starts})


def _fixed_cashflow_rate(deal, quote):
    """The fixed leg of a two-leg benchmark carries the quote on every row of its schedule."""
    for item in deal['Cashflows']['Items']:
        item['Rate'] = utils.Percent(quote)


#: Where a quote's number goes, per instrument type, keyed by the `Object` string. A quote NAMES an
#: instrument type and carries a block of it, so this is the ONE thing the family knows about a type
#: beyond that type's own declarations - and it is a registry rather than a branch because a new
#: quotable instrument is then a row. A container carries no rate itself; its fixed leg does.
QUOTE_WRITERS = {
    'DepositDeal': _pin_deposit_schedule,
    'FRADeal': lambda deal, quote: deal.update({'FRA_Rate': quote}),
    'SwapInterestDeal': lambda deal, quote: deal.update({'Swap_Rate': quote}),
    'CFFixedInterestListDeal': _fixed_cashflow_rate,
}


def author_quote(deal, quote, discount_rate):
    """Author an instrument block AT its quote, discounting on `discount_rate`.

    The split is the one the family is for: what an instrument PROJECTS off is its own business and
    it names that curve itself, while what the quote set DISCOUNTS on is a property of the curve
    set and is stated once on the block. Recurses into `Children`, so a two-leg benchmark gets the
    quote on the leg that holds a rate and the discount curve on both.
    """
    for child in deal.get('Children', ()):
        author_quote(child, quote, discount_rate)
    deal['Discount_Rate'] = discount_rate
    writer = QUOTE_WRITERS.get(deal['Object'])
    if writer:
        writer(deal, quote)


def quote_node(deal, valuation_options):
    """A deal-tree node from an authored instrument block - the shape `set_calculation_children`
    takes. `Config.parse_json` builds it from `.Deal` markers and a `Children` list; a quote carries
    the same block inline, so it is built here instead."""
    node = {'Instrument': instruments.construct_instrument(
        {key: value for key, value in deal.items() if key != 'Children'}, valuation_options)}
    if deal.get('Children'):
        node['Children'] = [quote_node(child, valuation_options) for child in deal['Children']]
    return node


def quote_knots(nodes, base_date, day_count, calendars):
    """The curve's knot grid: ONE knot per benchmark, at that benchmark's last cashflow date.

    That is the only placement that makes the system square, and squareness is the whole shape of a
    bootstrap: a knot with no instrument maturing at it is unidentified, and two instruments
    maturing between the same pair of knots leave the curve under-determined between them. Below
    the shortest knot the curve is flat, which `CurveTenor` gives by clipping, so the front stub
    costs no extra unknown. The output grid IS this grid - there is no second grid to write the
    result onto, because interpolating a solved curve onto one would stop it repricing its quotes.

    Returned in NODE order and in the curve's own day count, so a caller can pair each knot with
    the quote that identifies it; the curve itself is sorted.
    """
    code = utils.get_day_count(day_count)
    maturities = []
    for node in nodes:
        leaves = leaf_deals(node)
        for leaf in leaves:
            leaf.reset(calendars)
        maturities.append(max(max(leaf.get_reval_dates()) for leaf in leaves))
    return np.array([utils.get_day_count_accrual(
        base_date, (maturity - base_date).days, code) for maturity in maturities])


class InterestRateCurveParameters(object):
    """A zero curve solved from deposit, FRA and swap quotes, priced by the engine's own pricers.

    A quote is an instrument, a `Quote_Type` and a number - see the developer note on
    [Market Prices](../developer/market_prices.md). Each `Points` entry names an instrument type in
    `DealType` and carries a block of it in `Deal`, so the `Instrument` store's declarations ARE
    this family's quote schema and nothing about a swap is described twice. The family authors that
    block at its `Quoted_Market_Value`, and the residual is what the instrument is then worth at
    t0: a fair benchmark prices to zero, so the solve is a root find on the PV vector.

    Two blocks make a multi-curve set - an OIS discount curve solved from OIS quotes, then a
    projection curve solved from FRAs and par swaps discounting on it - and the second must be
    solved after the first, which is what `Discount_Rate` orders. A block whose `Discount_Rate` is
    blank discounts on the curve it is building, which is the degenerate single-curve
    configuration and the harder solve, since the unknown appears on both sides.

    Unlike the other four families this writes an `InterestRate` price factor rather than a
    `<ClassName>` parameter block, which is what `price_factor_type` declares.
    """
    market_factor_type = 'InterestRatePrices'
    #: The `Price Factors` type this family writes. The other four write a block named for their own
    #: class, so the emitter can recover it; a bootstrapped curve is an ordinary `InterestRate` and
    #: no rule recovers that from `InterestRateCurveParameters`.
    price_factor_type = 'InterestRate'
    #: The instrument types a quote in this family may be. Each is a declared `Instrument` type,
    #: so the quote's schema IS that type's declarations - reused by reference, never restated.
    #: `StructuredDeal` is how a two-leg benchmark is authored - an OIS swap is its compounded
    #: floating leg and its fixed leg, composed by `Children`.
    quote_instruments = ('DepositDeal', 'FRADeal', 'SwapInterestDeal', 'StructuredDeal')
    #: Block fields a calibration artifact is NOT a function of - the LIFECYCLE switches, read when
    #: one is published or ridden rather than when it is fitted. `plan_key` shadows them out for one
    #: reason: a knob that governs the ride must not also hide the artifact it governs, or loosening
    #: `Drift_Tolerance` would silently mean "refit" instead of "allow more". `Quote_Sensitivity`
    #: joins them because it provably moves neither theta* nor J - that is what its bit-identity
    #: gate says. Everything else on the block is an input to the solve.
    lifecycle_fields = ('Quote_Sensitivity', 'Quote_Propagation', 'Drift_Tolerance')
    fields = [
        F('Currency', 'Text', default=REQUIRED, description='The currency of the curve to build'),
        F('Day_Count', 'Text', default='ACT_365',
          values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA'],
          description='Daycount the solved curve\'s tenors are expressed in'),
        F('Discount_Rate', 'Text', default='',
          description='The curve the quotes discount on; blank builds a self-discounting curve'),
        F('N_Iter', 'Integer', default=50,
          description='Newton iteration cap. Newton is quadratic near the root and a par-rate seed '
                      'is already within a few basis points, so a well-posed strip converges in '
                      'single digits; reaching the cap raises rather than returning a half-solved '
                      'curve'),
        F('Tol', 'Float', default=1e-14,
          description='Convergence tolerance on the Newton STEP, in rate space. A zero rate is '
                      'O(1e-2), so 1e-14 is about 1e-12 relative - inside the 1e-10 a round trip '
                      'asks for, and where the linear solve\'s own rounding stops the iteration '
                      'improving'),
        F('Damping_Halvings', 'Integer', default=6,
          description='How many times the line search may halve a Newton step before giving up. '
                      'Below that the step LENGTH is not what is wrong, so the solve says so '
                      'rather than creeping towards a root it will not reach'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the solved curve connected to its quotes, so a calculation\'s backward '
                      'pass reports dV/dq beside dV/dtheta. Costs one extra compile of the '
                      'benchmark set and holds the residual graph for the life of the config; the '
                      'solved numbers are identical either way'),
        F('Quote_Propagation', 'Text', default='No', values=['No', 'Linear'],
          description='How a quote that moves between bootstraps reaches the curve. No re-solves, '
                      'which is what a job does today. Linear publishes a calibration artifact '
                      '(theta*, dtheta/dq, q0) at each bootstrap and RIDES it at every calculation '
                      'after - theta* + dtheta/dq (q_now - q0), a matvec instead of a solve - '
                      'refusing when the ridden curve no longer reprices the benchmarks inside '
                      'Drift_Tolerance, and refusing when no artifact answers to the plan. It is a '
                      'property of a COUPLED SET rather than of a block: blocks whose residuals '
                      'read each other\'s curves are solved as one system and ridden as one '
                      'operator, so every block of such a set must declare it or none may. Costs '
                      'one extra compile and one backward pass per block to measure the set, plus '
                      'the compile Quote_Sensitivity costs'),
        F('Drift_Tolerance', 'Float', default=1e-3,
          description='How far out of par a ridden curve may leave this block\'s own benchmarks '
                      'before Quote_Propagation refuses and asks for a refit, measured in PERCENT '
                      'OF QUOTE - so 1e-3 is a tenth of a basis point of mispricing. The ride is '
                      'second-order accurate, so this is a bound on the SQUARE of the tick: on the '
                      'round-trip worlds it admits about 11bp and refuses a 25bp move. Only read '
                      'when Quote_Propagation is Linear'),
        F('Points', 'Container', default={
            'Use': 'Yes', 'Deal': {}, 'Descriptor': '', 'DealType': 'DepositDeal',
            'Quote_Type': 'Par_Rate', 'Quoted_Market_Value': 0.0},
          sub_fields=[
            F('Use', 'Text', default='Yes', values=['Yes', 'No'],
              description='Whether this quote enters the solve'),
            F('Deal', 'Container', default={},
              description='The instrument itself, authored as a deal of type DealType'),
            F('Descriptor', 'Text', default='', description='Free text naming the quote'),
            F('DealType', 'Text', default='DepositDeal', values=list(quote_instruments),
              description='The instrument type the quote is a price for'),
            F('Quote_Type', 'Text', default='Par_Rate', values=['Par_Rate'],
              description='What Quoted_Market_Value is; the solve holds the instrument at par'),
            F('Quoted_Market_Value', 'Float',
              description='The quote, in percent, the instrument is authored at')],
          description='One market quote: an instrument, what kind of number is quoted, and the '
                      'number')
    ]

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param
        #: What a block asking for `Quote_Sensitivity` leaves behind: the solved nodes STILL
        #: CONNECTED to their quotes, per curve, and the quote leaf per block. `Config.bootstrap`
        #: harvests both - they are tensors, so they cannot live in `Price Factors`, which is data.
        self.calibrated = {}
        self.quote_leaves = {}

    def in_dependency_order(self, market_prices):
        """This family's blocks, one that discounts on a curve another block BUILDS coming after it.

        Dict order is the JSON author's, and a projection curve solved before its discount curve is
        solved against a curve that does not exist yet - which fails on the lookup rather than
        quietly, but fails on the author's ordering rather than on anything they got wrong.
        """
        blocks = {}
        for name, implied_params in market_prices.items():
            rate = utils.check_rate_name(name)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type == self.market_factor_type:
                blocks[name] = implied_params
        # keyed by the curve NAME a `Discount_Rate` field carries, which is the block's name without
        # its `Market Prices` type
        builds = {'.'.join(utils.check_rate_name(name)[1:]): name for name in blocks}
        graph = {name: [builds[implied_params['instrument']['Discount_Rate']]]
                 if implied_params['instrument']['Discount_Rate'] in builds else []
                 for name, implied_params in blocks.items()}
        return [(name, blocks[name]) for name in utils.topological_sort(graph)]

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices,
                  calendars, debug=None):
        """Solve every block for the zero curve that reprices its used quotes to par, one COUPLED
        SET at a time.

        A set is the group of blocks whose residuals read each other's curves, and it is MEASURED
        rather than declared - see `coupled_sets`. Forming one costs a compile and a backward pass
        per block, and it buys exactly one thing: an operator whose Jacobian carries the coupling.
        So it is only formed where an operator was asked for; with `Quote_Propagation` nowhere in
        the section, every block is its own group and this is the dependency-ordered loop it has
        always been, bit for bit.
        """
        base_date = sys_params['Base_Date']
        blocks = self.in_dependency_order(market_prices)
        groups = self.coupled_sets(blocks, price_factors, factor_interp, base_date, calendars) \
            if any(entry['instrument'].get('Quote_Propagation', 'No') == 'Linear'
                   for _, entry in blocks) else [[block] for block in blocks]

        for group in groups:
            self.solve_set(group, price_factors, factor_interp, base_date, calendars)

    def seed(self, market_price, block, price_factors, base_date, calendars):
        """Write this block's par-rate seed curve into `Price Factors`, and give back what the solve
        reads off it: `(curve factor, used quotes, deal nodes, discount rate)`.

        Seeding comes first because the benchmark closure constructs the curve factor OUT of
        `Price Factors` - and a par rate is within a few basis points of the zero rate at the same
        maturity, so it is also the seed the solve starts from. `Curve` sorts the pairs, so each
        knot keeps the quote that identifies it.
        """
        curve = utils.Factor('InterestRate', utils.check_rate_name(market_price)[1:])
        discount_rate = block['Discount_Rate'] or '.'.join(curve.name)
        points = self.used_quotes(block, market_price)
        nodes = quote_nodes(points, discount_rate)
        price_factors[utils.check_tuple_name(curve)] = {
            'Property_Aliases': None, 'Sub_Type': None, 'Currency': block['Currency'],
            'Day_Count': block['Day_Count'], 'Curve': utils.Curve([], list(zip(
                quote_knots(nodes, base_date, block['Day_Count'], calendars),
                [point['Quoted_Market_Value'] / 100.0 for point in points])))}
        return curve, points, nodes, discount_rate

    def coupled_sets(self, blocks, price_factors, factor_interp, base_date, calendars):
        """This family's blocks grouped into the SETS that have to solve as one system - MEASURED.

        Two blocks are coupled when one's residual READS the curve the other builds, and that is not
        the question `Discount_Rate` answers. What a benchmark projects off is authored inside its
        own deal block, so a strip declaring a blank `Discount_Rate` - self-discounting, by the
        declaration - can still forecast off a neighbour's curve. Measured on such a world, a 10bp
        tick moved the "independent" curve by 568 basis points while every declaration said it stood
        alone. `BenchmarkInstruments.reads` answers by differentiation instead: one compile and one
        backward pass per block, and it cannot be fooled by what a field says.

        The groups are the connected components of that relation, in dependency order, and a group
        is solved and ridden WHOLE. That is what puts `dtheta_2/dq_1` inside `J` rather than leaving
        it to the order the blocks were solved in - an ordering carries a coupling through a
        bootstrap and carries nothing at all through a ride.

        Every block is seeded before anything is measured, because a block that forecasts off a
        curve nobody has built yet cannot be compiled - and an undeclared dependency is exactly the
        case `in_dependency_order` cannot order.
        """
        seeded, builds, ordered = {}, {}, dict(blocks)
        for market_price, entry in blocks:
            curve, points, nodes, _ = self.seed(
                market_price, entry['instrument'], price_factors, base_date, calendars)
            seeded[market_price] = (curve, nodes)
            builds[curve] = market_price

        components = {market_price: {market_price} for market_price in ordered}
        for market_price, entry in blocks:
            curve, nodes = seeded[market_price]
            benchmarks = BenchmarkInstruments(
                nodes, price_factors, factor_interp, base_date,
                entry['instrument']['Currency'], calendars, [curve], self.device)
            reads = benchmarks.reads({curve: torch.tensor(
                benchmarks.factors[curve].current_value(),
                dtype=BenchmarkInstruments.dtype, device=self.device)})
            for factor in reads & set(builds):
                merged = components[market_price] | components[builds[factor]]
                for name in merged:
                    components[name] = merged

        groups, seen = [], set()
        for market_price in ordered:
            if market_price not in seen:
                seen |= components[market_price]
                groups.append([(name, ordered[name]) for name in ordered
                               if name in components[market_price]])
        return groups

    def solve_set(self, group, price_factors, factor_interp, base_date, calendars):
        """Solve one coupled set: ONE Newton system over every curve in it, one Jacobian, one
        artifact.

        A set of one is the single-curve solve this family has always done. A set of more is the
        multi-curve one, and flattening it is `damped_newton`'s own shape rather than a new solver -
        `solve_for` is a list, the residual takes a `{Factor: nodes}` over it, and the block
        Jacobian that falls out is what `calibration_jacobian` inverts in one go.

        The seed theta is read off the CONSTRUCTED factor rather than off the authored block, so it
        is aligned with the tenor grid the pricers gather against, whatever `get_tenor` made of the
        block. The solve goes through the implicit-function wrapper either way: with no quotes on
        the tape its forward IS `damped_newton` and no edge is recorded, which is what makes
        "gradients cannot move a mark" structural rather than a claim.

        Solver knobs are declared PER BLOCK, and a set takes the STRICTEST of them: a system is only
        as converged as its tightest member asked to be.
        """
        members = [(market_price, entry['instrument']) for market_price, entry in group]
        propagate = [block.get('Quote_Propagation', 'No') == 'Linear' for _, block in members]
        connect = [block.get('Quote_Sensitivity', 'No') == 'Yes' for _, block in members]
        if any(propagate) and not all(propagate):
            raise Exception(
                'Quote_Propagation is a property of a COUPLED SET, and {} solve as one system - '
                'measured, not declared. {} declares it while {} does not, and a partial ride is '
                'the one configuration this operator cannot express: the declining block would be '
                'priced off the curve the last bootstrap wrote while its partner rode the tick, '
                'and the drift metric would report the ridden half as perfectly fresh. Measured on '
                'the USD world at a 10bp OIS tick, that reads a PV of 9829.62 where the refit says '
                '9621.25, against a true move of -23.36 - wrong sign, 8.9x the size, drift 4.5e-4. '
                'Declare Quote_Propagation on every block of the set, or on none.'.format(
                    ' + '.join(name for name, _ in members),
                    ' + '.join(name for (name, _), asks in zip(members, propagate) if asks),
                    ' + '.join(name for (name, _), asks in zip(members, propagate) if not asks)))
        currencies = {block['Currency'] for _, block in members}
        if len(currencies) > 1:
            raise Exception(
                'Quote_Propagation: {} are coupled but priced in {} - a benchmark set has one '
                'reporting currency, so this set cannot be compiled as one system. Leave '
                'Quote_Propagation at No on a cross-currency curve set.'.format(
                    ' + '.join(name for name, _ in members), ' and '.join(sorted(currencies))))

        seeded = [self.seed(market_price, block, price_factors, base_date, calendars)
                  for market_price, block in members]
        # both switches want the quote side of the residual - one to report through it, one to
        # publish it as an operator - and it is one extra compile either way
        carry = any(connect) or any(propagate)
        points = [point for _, block_points, _, _ in seeded for point in block_points]
        curves = [curve for curve, _, _, _ in seeded]

        time_now = time.monotonic()
        benchmarks = BenchmarkInstruments(
            [node for _, _, nodes, _ in seeded for node in nodes], price_factors, factor_interp,
            base_date, members[0][1]['Currency'], calendars, curves, self.device,
            quotes=[point['Quoted_Market_Value'] for point in points] if carry else None,
            bumped_nodes=[node for _, block_points, _, discount_rate in seeded
                          for node in quote_nodes(block_points, discount_rate, 1.0)]
            if carry else None)
        # seed theta off the CONSTRUCTED factor - see the docstring on grid alignment
        theta = CalibrationSolve.apply(
            benchmarks,
            {curve: torch.tensor(benchmarks.factors[curve].current_value(),
                                 dtype=BenchmarkInstruments.dtype, device=self.device)
             for curve in curves},
            max(int(block.get('N_Iter', 50)) for _, block in members),
            min(float(block.get('Tol', 1e-14)) for _, block in members),
            max(int(block.get('Damping_Halvings', 6)) for _, block in members),
            benchmarks.quotes)

        solved = split_theta(benchmarks, theta)
        # a set-wide quote leaf reports dV/dq across the whole system, so its descriptors have to
        # name the block each quote came off; a set of one is the block's own list unchanged
        descriptors = [point['Descriptor'] if len(members) == 1 else
                       '{}: {}'.format(market_price, point['Descriptor'])
                       for (market_price, _), (_, block_points, _, _) in zip(members, seeded)
                       for point in block_points]
        for curve, (market_price, _), wants in zip(curves, members, connect):
            price_factors[utils.check_tuple_name(curve)]['Curve'] = utils.Curve(
                [], list(zip(benchmarks.tenors[curve], solved[curve].detach().cpu().numpy())))
            if wants:
                self.calibrated[curve] = solved[curve]
                self.quote_leaves[market_price] = (descriptors, benchmarks.quotes)
        if all(propagate):
            self.publish(members, factor_interp, base_date, benchmarks, theta.detach())

        residuals = benchmarks(split_theta(benchmarks, theta.detach())).detach()
        logging.info('{} bootstrapped from {} quotes in {:.2f} seconds, residual {:.3g}'.format(
            ' + '.join(utils.check_tuple_name(curve) for curve in curves), len(points),
            time.monotonic() - time_now, float(residuals.abs().max())))
        for point, residual in zip(points, residuals):
                logging.info('  {} at {:.4f} reprices to {:.3g}'.format(
                    point['Descriptor'], point['Quoted_Market_Value'], float(residual)))

    @classmethod
    def takes(cls, point, market_price):
        """Whether this family prices the quote, the way Clewlow-Strickland says so of a vol type.

        `Par_Rate` is the only convention built: every increment-1 benchmark is held at PV zero.
        A futures price and a money-market rate on a different basis are conventions this would
        have to author differently, and they are declared when they are read.
        """
        if point['Quote_Type'] == 'Par_Rate':
            return True
        logging.error('{} quote {} - Quote_Type {} not supported yet'.format(
            market_price, point['Descriptor'], point['Quote_Type']))
        return False

    @classmethod
    def used_quotes(cls, block, market_price):
        """The quotes that enter the solve, in the order theta, `J` and `q0` are all indexed by.

        A classmethod because the RIDE needs the same list off a block nobody is bootstrapping, and
        a second filter written beside this one is how a ridden theta ends up indexed by a
        different quote vector than the artifact was fitted with.
        """
        return [point for point in block['Points']
                if point['Use'] == 'Yes' and cls.takes(point, market_price)]

    @classmethod
    def plan_key(cls, members, factor_interp, base_date):
        """The SLOT an artifact lives in: every member block of the coupled set, the base date, the
        interpolation scheme and the engine version - with the quote NUMBERS and the
        `lifecycle_fields` shadowed out.

        Plan-side coordinates, and the same split `Config.plan_hash` takes over a price factor - a
        value is shadowed to `None` rather than dropped, so the key SET stays structural and adding
        a quote is a different plan. Every tick of one strip therefore lands on the SAME slot,
        which is what makes a ride possible at all, while a re-authored instrument, a flipped
        `Use`, a different `Day_Count`, a different solver knob or a new engine build lands on a
        different one and finds nothing to ride.

        The key names the SET rather than the block, so re-authoring a discount strip moves the slot
        of every curve solved against it - which is the point: a projection curve riding a `J` fitted
        against quotes that no longer exist is exactly the artifact that must not be findable.

        `base_date` and `Price Factor Interpolation` are in it because the SOLVE reads them and the
        block does not carry them. Both were measured riding each other's theta* out of one slot
        before they were named here: two jobs 45 days apart shared a slot, and a Linear job rode a
        Hermite solve 0.53bp away from its own.
        """
        # `config` imports from this module, so the package edge runs one way only and the hash is
        # reached from inside the call
        from . import content_hash

        return content_hash({
            'engine_version': __version__, 'base_date': base_date, 'interpolation': factor_interp,
            'set': [{'market_price': market_price,
                     'block': dict(block, Points=[dict(point, Quoted_Market_Value=None)
                                                  for point in block['Points']],
                                   **{field: None for field in cls.lifecycle_fields})}
                    for market_price, block in members]})

    @classmethod
    def slot(cls, names, market_prices, factor_interp, base_date):
        """The key those member blocks address in `market_prices` NOW, or `None` if one is gone.

        What turns `ArtifactStore.find`'s scan back into content addressing: an artifact answers for
        a curve only if the plan it was fitted against is still the plan standing.
        """
        members = [(name, market_prices.get(name, {}).get('instrument')) for name in names]
        if any(block is None for _, block in members):
            return None
        return cls.plan_key(members, factor_interp, base_date)

    @classmethod
    def publish(cls, members, factor_interp, base_date, benchmarks, theta):
        """Freeze this solve as an artifact, and MEASURE what the last one would have been worth.

        The drift is the whole point of refitting on a schedule rather than on every tick: with the
        previous artifact still in the slot, `theta_refit - theta_ridden` says how far the linear
        operator had drifted by the time it was replaced, and the ridden theta's benchmark residual
        says the same thing in the space the tolerance is declared in. Both are logged against the
        set's own solver `Tol`, and both are published ON the new artifact - so the record of how
        stale the last calibration got travels with the calibration that replaced it.

        The refreshed artifact takes the old one's SLOT and carries a new `artifact_id`, because
        the id is the slot plus the quotes it was fitted at.
        """
        key = cls.plan_key(members, factor_interp, base_date)
        artifact = CalibrationArtifact(
            key, [market_price for market_price, _ in members], theta,
            calibration_jacobian(benchmarks, split_theta(benchmarks, theta)),
            benchmarks.quotes.detach(), benchmarks)
        name = ' + '.join(market_price for market_price, _ in members)

        previous = ARTIFACTS.get(key)
        if previous is not None:
            ridden = previous.ride(artifact.quotes)
            artifact.drift = {
                'tick': float((artifact.quotes - previous.quotes).abs().max()),
                'theta': float((theta - ridden).abs().max()),
                'quote': float(artifact.mispricing(ridden, artifact.quotes).abs().max()),
                'rode': previous.artifact_id, 'fitted': previous.timestamp}
            logging.info(
                '{} refit: artifact {} (fitted {}) rode a {:.4g}% tick to a drift of {:.3g} in '
                'theta and {:.3g}% in quote space (solver Tol {:.3g}), replaced by {}'.format(
                    name, previous.artifact_id[:12], previous.timestamp, artifact.drift['tick'],
                    artifact.drift['theta'], artifact.drift['quote'],
                    min(float(block.get('Tol', 1e-14)) for _, block in members),
                    artifact.artifact_id[:12]))
        else:
            logging.info('{} refit: artifact {} published, nothing in the slot to score'.format(
                name, artifact.artifact_id[:12]))
        ARTIFACTS.put(artifact)

    @classmethod
    def propagate(cls, factor, market_prices, factor_interp, base_date):
        """The curve `factor` RIDDEN to the quotes standing in `market_prices` now, or `None` where
        no block asks for one - the operator, evaluated per EXECUTE and storing nothing.

        Two ways to get `None`, and each is today's path unchanged: a factor this family does not
        write, and a block that did not ask for `Quote_Propagation`.

        A BLOCK THAT DID ASK AND FINDS NO ARTIFACT REFUSES. That is the house rule for a plan the
        cache cannot answer - a miss is a 404, never a different number - and it is what closes the
        replay hole: a silent fall back to `theta*` reprices the book (13.4% on the eviction probe)
        while `plan_hash`, `values_hash`, the engine version and the seed all stay identical, so
        nothing in the replay tuple could tell the two runs apart. A refusal is not a number, so it
        cannot be mistaken for one. A cold process therefore rides nothing and says so out loud: an
        artifact holds tensors and a compiled deal tree, cannot be serialised, and a re-bootstrap
        rederives it from the job document.

        A ride that would leave the benchmarks further out of par than `Drift_Tolerance` refuses for
        the same reason - the alternative is a plausible wrong curve. The tolerance is the SET's
        strictest, so a coupled set rides or refuses whole; and the artifact it is scored against
        is the one whose plan is still standing, which `slot` rechecks rather than trusts.
        """
        if factor.type != cls.price_factor_type:
            return None
        market_price = utils.check_tuple_name(utils.Factor(cls.market_factor_type, factor.name))
        block = market_prices.get(market_price, {}).get('instrument')
        if block is None or block.get('Quote_Propagation', 'No') != 'Linear':
            return None

        covering = ARTIFACTS.covering(factor)
        artifact = next((found for found in covering if found.key == cls.slot(
            found.members, market_prices, factor_interp, base_date)), None)
        if artifact is None:
            raise utils.CalibrationStale(
                '{}: Quote_Propagation is Linear and no calibration artifact answers to this plan '
                '- {}. Bootstrap the job and the same EXECUTE runs off the artifact that publishes; '
                'an artifact holds tensors and a compiled benchmark set, so it cannot be serialised '
                'and a fresh process has none. A plan the store cannot answer is a MISS, and a miss '
                'is not permission to price off the curve the last bootstrap wrote.'.format(
                    market_price, 'the store holds none for this curve' if not covering else
                    '{} cover it, each fitted against a different plan ({})'.format(
                        len(covering), '; '.join(' + '.join(found.members) for found in covering))))
        # a ride is a USE: the slot a tick stream keeps riding must not age out under one that is
        # merely published beside it
        ARTIFACTS.get(artifact.key)

        tolerance = min(float(market_prices[name]['instrument'].get('Drift_Tolerance', 1e-3))
                        for name in artifact.members)
        quotes = torch.tensor(
            [point['Quoted_Market_Value'] for name in artifact.members
             for point in cls.used_quotes(market_prices[name]['instrument'], name)],
            dtype=artifact.theta.dtype, device=artifact.theta.device)
        theta = artifact.ride(quotes)
        drift = float(artifact.mispricing(theta, quotes).abs().max())
        tick = float((quotes - artifact.quotes).abs().max())
        # the tolerance is declared in percent of quote; ||J||inf converts it into the curve units
        # the staleness is actually felt in, which is what a desk sets the number against
        curve_units = tolerance * artifact.jacobian_norm * 1e4
        if drift > tolerance:
            raise utils.CalibrationStale(
                '{}: Quote_Propagation refused - riding artifact {} (fitted {}) over a {:.4g}% '
                'tick leaves its benchmarks {:.3g}% of quote out of par, past the declared '
                'Drift_Tolerance {:.3g} (at most {:.3g}bp of zero rate on this set, ||J||inf '
                '{:.4g}). The linear operator is only second-order accurate and this move is too '
                'big for it, so re-bootstrap and the same job runs off the refit.'.format(
                    market_price, artifact.artifact_id[:12], artifact.timestamp, tick, drift,
                    tolerance, curve_units, artifact.jacobian_norm))
        logging.info(
            '{}: rode artifact {} (fitted {}) over a {:.4g}% tick, benchmarks {:.3g}% of quote out '
            'of par against a tolerance of {:.3g} ({:.3g}bp of zero rate, ||J||inf {:.4g})'.format(
                market_price, artifact.artifact_id[:12], artifact.timestamp, tick, drift,
                tolerance, curve_units, artifact.jacobian_norm))
        return artifact.nodes(theta, factor), artifact.artifact_id


class FXVolSurfaceParameters(object):
    """An `FXVol` surface bootstrapped from the ATM / risk-reversal / butterfly quotes it ticks in as.

    An FX smile is quoted in DELTA - one ATM vol per expiry and, per delta pillar, the risk
    reversal and butterfly that say how the two wings sit around it - so the quote set is three
    numbers per (expiry, pillar) and the surface the engine prices off is a log-moneyness one. The
    algebra between them is the strangle pair, `vol(call) = ATM + BF + RR/2` and
    `vol(put) = ATM + BF - RR/2`, followed by the delta-to-log-moneyness solve `Factor2D` has
    always carried for a `Malz` surface. Neither half is new here; what is new is WHERE they run.

    **The x-grid is pinned.** The solve does not evaluate a smile at prescribed strikes - it
    refines a log-moneyness grid until interpolating between its nodes resolves the smile, so the
    grid is a function of the quotes. Run at factor-construction time that makes every vol tick
    potentially STRUCTURAL: a moved node is a moved tenor grid, a new plan, and a recompile per
    tick. So the refinement runs HERE, once, and the grid it produced is part of the written
    factor. A re-bootstrap that finds a surface already written for the same expiries at the same
    tolerance reuses that grid and moves only the vols on it - which is what makes a tick a
    `bind='value'` patch. `Grid_Tolerance` counts because it SIZES the grid, so it is written
    beside the grid it built and it is structural: asking for a different one is not asking for
    this grid, and the pin breaks. The cost of the pin is measured rather than hidden - the log
    says what the pinned grid resolves the CURRENT quotes to, beside the tolerance it was built at.

    **The conventions are declared because the solve implements exactly one of each.** The delta a
    pillar names is a PREMIUM-ADJUSTED FORWARD delta ((K/F)N(d2) for a call), and the ATM quote is
    that convention's DELTA-NEUTRAL STRADDLE, K = F exp(-sigma^2 T / 2). Those are the two
    conventions `Factor2D.malz_skew` implements, so they are the only values these fields offer -
    a spot delta or an ATMF quote would need different algebra, and a value the engine cannot
    honour is the same defect as a field nothing reads.

    A quote `Timestamp` survives a save at the resolution it was authored: a plain date stays a
    date and an intraday stamp keeps its time, so a stamped surface round-trips to the moment its
    quotes were seen.

    Like `InterestRatePrices` this writes a typed price factor rather than a `<ClassName>`
    parameter block, which is what `price_factor_type` declares.
    """
    market_factor_type = 'FXVolPrices'
    #: The `Price Factors` type this family writes - a `Malz` `FXVol`, the same block a delta
    #: surface authored by hand builds, minus the delta surface: it arrives SOLVED, so
    #: `Factor2D.solves_delta_surface` is false and the pinned grid survives construction.
    price_factor_type = 'FXVol'
    #: What the written block says beyond its surface. `Surface_Type` names the moneyness
    #: convention the engine reads it at (log(F/K), interpolated in total variance);
    #: `Moneyness_Rule` is the factor's own declared default and no Malz code path reads it.
    surface_type, moneyness_rule = 'Malz', 'Sticky_Moneyness'
    #: The tolerances a grid can actually be BUILT at, declared on the field below and read back by
    #: `bootstrap` - the one `bounds=` in the schema the engine enforces rather than publishes,
    #: because outside it there is no grid to write. Refinement halves an interval until the
    #: midpoint's vol error falls under the tolerance, so at 0.0 no midpoint ever qualifies (7.6M
    #: nodes on one expiry after 21 passes, still doubling); 1e-8 is 4599 nodes for a four-expiry
    #: smile, which is a large plan but a plan. At the top the seed grid already passes and
    #: refining is a no-op, so 1 is where the knob stops meaning anything rather than where it
    #: breaks.
    grid_tolerance_bounds = (1e-8, 1.0)
    #: The precision the TAPE runs in - the value path is numpy and has no dtype to pick. The twin
    #: divides by the residual's own slope at the root, so it is float64 on the CPU whatever the job
    #: asked for: `construct_bootstrapper`'s dtype does not reach it.
    dtype = torch.float64
    fields = [
        F('Currency', 'Text', default='',
          description='The currency stamped on the surface this builds'),
        F('Delta_Type', 'Text', default='Forward', values=['Forward'],
          description='The delta a Pillar names. The solve inverts a FORWARD delta - there is no '
                      'spot-delta discounting in it - so that is the one convention offered'),
        F('Premium_Adjusted', 'Text', default='Yes', values=['Yes'],
          description='Whether the pillar delta is premium adjusted. The solve inverts '
                      '(K/F)N(d2), which is the premium-adjusted (percentage-foreign) delta'),
        F('ATM_Convention', 'Text', default='Delta_Neutral_Straddle',
          values=['Delta_Neutral_Straddle'],
          description='What an ATM quote is the vol of. The solve places it at the strike whose '
                      'premium-adjusted straddle is delta neutral, K = F exp(-sigma^2 T/2); an '
                      'ATMF quote would sit at a different strike and is not built'),
        F('Grid_Tolerance', 'Float', default=1e-4, bounds=grid_tolerance_bounds,
          description='The vol error the log-moneyness grid is refined to when it is BUILT. Not '
                      'reached again per tick: the grid is pinned, so this sizes the plan rather '
                      'than the quote fit, and the log reports what the pinned grid still '
                      'resolves the quotes to. Changing it is STRUCTURAL - it breaks the pin and '
                      'refines a new grid. Bounded because refinement does not terminate below '
                      'the floor'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the log-moneyness surface connected to the ATM / RR / BF quotes it was '
                      'built from, so a calculation\'s backward pass reports dV/dq beside '
                      'dV/dtheta. The written surface is identical either way'),
        F('Points', 'Table', default='null', row=Row([
            F('Use', 'Text', default='Yes', values=['Yes', 'No'],
              description='Whether this quote enters the surface'),
            F('Expiry', 'Float',
              description='Expiry in YEARS - the surface\'s own expiry axis, so no day count '
                          'stands between the quote and the coordinate it lands on'),
            F('Pillar', 'Float',
              description='The delta the wings are quoted at, as a magnitude (0.25 is the 25 '
                          'delta pair). Not read on an ATM row, which is quoted at no pillar'),
            F('Quote_Type', 'Text', default='ATM', values=['ATM', 'RR', 'BF'],
              description='The ATM vol, the risk reversal (call less put) or the butterfly (the '
                          'wing pair\'s average over ATM)'),
            F('Quoted_Market_Value', 'Float',
              description='The quote, in the surface\'s own units - 0.12 for 12 vols, and a risk '
                          'reversal of -0.35 vols is -0.0035'),
            F('Timestamp', 'Date', default='',
              description='When this quote was observed. Stored and reported - the surface '
                          'carries the latest of them - and never read by pricing')]),
          description='One quote: an expiry, a delta pillar, what kind of number is quoted, the '
                      'number, and when it was seen')
    ]

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param
        #: What a block asking for `Quote_Sensitivity` leaves behind: the log-moneyness surface
        #: STILL CONNECTED to the quotes it was built from, under the key `_build_factor_state`
        #: mints the `FXVol` leaf with, and the quote leaf per block. `Config.bootstrap` harvests
        #: both - they are tensors, so they cannot live in `Price Factors`, which is data and gets
        #: written back out as JSON.
        self.calibrated = {}
        self.quote_leaves = {}

    @staticmethod
    def used(block):
        """The block's quotes that enter the surface - `Use` holds one out without deleting it."""
        return [point for point in block['Points'] if point['Use'] == 'Yes']

    @staticmethod
    def descriptor(point):
        """What a quote is CALLED where `dV/dq` is reported: its type, its pillar, its expiry."""
        return ('ATM {:g}'.format(point['Expiry']) if point['Quote_Type'] == 'ATM' else
                '{} {:g} {:g}'.format(point['Quote_Type'], point['Pillar'], point['Expiry']))

    @staticmethod
    def atm_quotes(quotes):
        """`{expiry: ATM vol}` - the ATM row per expiry, the number that expiry's wings sit around.

        The surface's ATM vol at an expiry IS this number: `Factor2D.malz_skew` places it at the
        delta-neutral straddle strike and every other node is quoted relative to it. So it is what
        `smile` builds the wings off, and it is what `GBMAssetPriceTSModelParameters` takes as the
        ATM column when a surface this family built is the one being integrated.
        """
        return {point['Expiry']: point['Quoted_Market_Value']
                for point in quotes if point['Quote_Type'] == 'ATM'}

    @classmethod
    def smile(cls, quotes):
        """The quotes as a `(delta, expiry, vol)` surface - the strangle pair, per expiry pillar.

        `vol(call) = ATM + BF + RR/2` and `vol(put) = ATM + BF - RR/2`, with the ATM vol itself
        carried at the +-0.5 LABEL the delta solve reads it off (0.5 is not a delta there, and the
        solve replaces the label with the delta-neutral straddle's own delta). A pillar quoted
        with only one of the two is read with the other at zero, which is how a symmetric smile is
        authored; an expiry with wings but no ATM quote raises `KeyError` on that expiry, because
        the wings are quoted AROUND a number that is not there.

        A `Pillar` of 0.5 is refused. It is the one delta that is not a delta here - it is the ATM
        label - so a wing quoted at it would land a second, different vol on the ATM row's own
        coordinate and the surface would silently carry whichever survived the sort. A 50 delta
        pair is quoted as the ATM row by convention, which is where it has to be authored.
        """
        atm = cls.atm_quotes(quotes)
        wings = {(point['Expiry'], point['Pillar'], point['Quote_Type']):
                 point['Quoted_Market_Value'] for point in quotes if point['Quote_Type'] != 'ATM'}
        pillars = sorted({key[:2] for key in wings})

        surface = [[0.5, expiry, vol] for expiry, vol in atm.items()]
        for expiry, pillar in pillars:
            if np.isclose(pillar, 0.5):
                raise ValueError(
                    'the {} quote at expiry {:g} is on Pillar {:g}, which collides with the ATM '
                    'label - a 50 delta pair is quoted as the ATM row'.format(
                        '/'.join(sorted(k[2] for k in wings if k[:2] == (expiry, pillar))),
                        expiry, pillar))
            rr, bf = wings.get((expiry, pillar, 'RR'), 0.0), wings.get((expiry, pillar, 'BF'), 0.0)
            surface.append([pillar, expiry, atm[expiry] + bf + 0.5 * rr])
            surface.append([-pillar, expiry, atm[expiry] + bf - 0.5 * rr])
        return np.array(sorted(surface))

    @staticmethod
    def carried_smile(quotes, values):
        """`smile`'s vol column on a tape - the strangle algebra, mirrored, and nothing else.

        Row for row and in `smile`'s own order, so the frozen structure the solve reads off the
        value path addresses this vector. The algebra is `+`, `*` and a sort, all of which torch
        and numpy agree on to the last bit in float64 - there is no `sqrt` here to disagree over -
        so the mirror is bit-identical rather than close, and gated as such.
        """
        atm = {point['Expiry']: value for point, value in zip(quotes, values)
               if point['Quote_Type'] == 'ATM'}
        wings = {(point['Expiry'], point['Pillar'], point['Quote_Type']): value
                 for point, value in zip(quotes, values) if point['Quote_Type'] != 'ATM'}

        zero = values.new_zeros(())
        rows = [(0.5, expiry, vol) for expiry, vol in atm.items()]
        for expiry, pillar in sorted({key[:2] for key in wings}):
            rr = wings.get((expiry, pillar, 'RR'), zero)
            bf = wings.get((expiry, pillar, 'BF'), zero)
            rows.append((pillar, expiry, atm[expiry] + bf + 0.5 * rr))
            rows.append((-pillar, expiry, atm[expiry] + bf - 0.5 * rr))
        return torch.stack([vol for _, _, vol in sorted(rows, key=lambda row: row[:2])])

    @classmethod
    def carried_skews(cls, delta_surface, expiries, vols):
        """`Factor2D.malz_skews` on a tape - `vols` is that surface's vol column, still connected."""
        return {T: cls.carried_skew(delta_surface[delta_surface[:, 1] == T][:, 0],
                                    vols[delta_surface[:, 1] == T].clamp(min=1e-4), T)
                for T in expiries}

    @staticmethod
    def carried_skew(delta, vols, T):
        """`Factor2D.malz_skew` on a tape - the same wing pair, node for node, still connected.

        WHAT IS TAPED AND WHAT IS FROZEN. The wing vols are, and so is `delta_atm`: the ATM quote
        says where the delta-neutral straddle sits, so it MOVES the two ATM nodes of the delta grid
        the wings are indexed by, and a twin that treated those deltas as constants would silently
        drop that channel. What is read off the numbers rather than differentiated is the LAYOUT -
        the ordering, which node carries the +-0.5 label, and which side had its ATM node mirrored
        in - because a permutation has no derivative, and because the layout is exactly what the
        value path's frozen indices address.
        """
        d = np.asarray(delta, dtype=float)
        order = np.argsort(d)
        d, v = d[order], list(vols[order])

        atm = np.isclose(np.abs(d), 0.5)
        sigma_atm = v[np.flatnonzero(atm)[-1]]  # ascending d, so the PREFERRED +0.5 label
        delta_atm = 0.5 * torch.exp(-0.5 * sigma_atm * sigma_atm * T)
        label = float(delta_atm.detach())
        nodes = [np.sign(di) * delta_atm if a else sigma_atm.new_tensor(di)
                 for di, a in zip(d, atm)]
        d = np.where(atm, np.sign(d) * label, d)

        # both wings need the ATM node - a smile quoted on one side only is mirrored onto the other
        for side in (-1.0, 1.0):
            if not np.any(np.isclose(d, side * label)):
                d, nodes, v = np.append(d, side * label), nodes + [side * delta_atm], v + [sigma_atm]

        order = np.argsort(d)
        d = d[order]
        deltas = torch.stack([nodes[i] for i in order])
        vols = torch.stack([v[i] for i in order])
        return {'d_put': deltas[d <= 0.0], 'v_put': vols[d <= 0.0],
                'd_call': deltas[d >= 0.0], 'v_call': vols[d >= 0.0],
                'sigma_atm': sigma_atm, 'delta_atm': delta_atm}

    @classmethod
    def carried_sigma(cls, skew, carried, T, x):
        """`Factor2D.malz_sigma` on a tape - AND THE BISECTION IS NOT ON IT.

        The 64 halvings are a fixed-length loop of correctly-rounded arithmetic and every operation
        in them tapes, which is exactly the trap. A bisection's iterates are DYADIC combinations of
        the two bracket endpoints - `left` and `right` are only ever `lo`, `hi` or a midpoint of
        two such - so what a tape through the loop differentiates is where the BRACKET is, not
        where the root is. On the call wing `lo` is a quoted pillar delta and `hi` is `delta_atm`,
        so that derivative carries the ATM quote and no risk reversal or butterfly at all, while
        the true root moves with the wing vols the residual is built from. It is the workstream's
        own failure mode: a number that is right and a gradient that is not.

        So the tape starts at the CONVERGED root. `delta*` is a constant here and the differentiable
        one is one Newton step off it, `delta - R(delta, q) / (dR/ddelta)`, which is the implicit
        function theorem written as an expression: worth the solve's own residual forward (nothing,
        and it reaches no mark either way) and exactly `-R_q / R_delta` backward. WHAT MAKES IT THE
        THEOREM IS THAT `delta*` IS THE ROOT, not that the slope is detached - the detach is
        conceptual hygiene, and measured to be nothing else (see below).

        The CLAMPED nodes take the other branch, and it is not a repair. Where the fixed point
        falls outside the wing's bracket there is no root to differentiate: the vol IS the endpoint
        knot's, which is what flat-extrapolating a smile beyond its widest quoted delta means, so
        the taped delta is that knot and the derivative is the knot vol's own - `1` in that wing's
        ATM quote, `1` in its butterfly, `+-1/2` in its risk reversal, and zero in everything else.
        The two branches meet where the root arrives AT the endpoint, so the switch is a kink and
        what autograd reports is the one-sided derivative of the branch the node is in.

        THE WING SPAN IS GUARDED, and an ordinary config reaches it: an ATM-only smile has
        `malz_skew` mirror its ONE node onto both sides, so each wing is a single knot whose span
        is exactly zero. Dividing before selecting puts a NaN on every entry of the Jacobian while
        the value path writes a perfectly good flat surface.
        """
        delta_star, is_call, bracketed = riskfactors.Factor2D.malz_delta(skew, T, x)
        sigma = carried['sigma_atm'].new_zeros(np.shape(x))

        for side, wing in ((1.0, 'call'), (-1.0, 'put')):
            on_wing = is_call if side > 0 else ~is_call
            if not on_wing.any():
                continue
            knots, values, grid = carried['d_' + wing], carried['v_' + wing], skew['d_' + wing]
            xs, root, live = x[on_wing], delta_star[on_wing], bracketed[on_wing]
            # the SEGMENT the root sits in and, for a clamped node, the endpoint knot it sits ON -
            # both frozen, because an interval index is not a differentiable quantity
            seg = np.clip(np.searchsorted(grid, root, side='right') - 1,
                          0, max(grid.size - 2, 0))
            top = np.minimum(seg + 1, grid.size - 1)
            near = np.abs(root[:, None] - grid[None, :]).argmin(1)
            span = knots[top] - knots[seg]
            wide = values.new_tensor(grid[top] != grid[seg], dtype=torch.bool)
            k_over_f, log_mny = values.new_tensor(np.exp(-xs)), values.new_tensor(xs)

            def wing_vol(delta):
                # the double where over a ONE-KNOT wing's zero span - see the docstring
                low = values[seg]
                rise = torch.where(wide, (values[top] - low) / torch.where(
                    wide, span, torch.ones_like(span)), torch.zeros_like(span))
                return low + (delta - knots[seg]) * rise

            def residual(delta):
                vol = wing_vol(delta)
                d2 = (log_mny - 0.5 * vol * vol * T) / (vol * np.sqrt(T))
                return k_over_f * side * utils.norm_cdf(side * d2) - delta

            # THREE HONEST NEGATIVES on the next two lines, all measured and all no-ops. `base` is
            # minted from numpy and carries no graph, so its `.detach()` is dead; asking the slope
            # for `create_graph` changes no reported number either, because `R(delta*)` is ~1e-17
            # and multiplies the extra term away - the theorem holds off the ROOT, not off the
            # detach. And nothing CANCELS an unbracketed node's slope to zero the way a one-knot
            # wing cancels its span, so the `on_tape` guard below is idiom rather than a measured
            # hazard: a 375-point sweep drives |dR/ddelta| no lower than 0.948.
            base = values.new_tensor(root)
            probe = base.detach().requires_grad_(True)
            d_delta = torch.autograd.grad(residual(probe).sum(), probe)[0]
            on_tape = values.new_tensor(live, dtype=torch.bool)
            step = residual(base) / torch.where(on_tape, d_delta, torch.ones_like(d_delta))
            sigma[on_wing] = wing_vol(torch.where(on_tape, base - step, knots[near]))

        return sigma

    @classmethod
    def carried_surface(cls, skews, carried, grid):
        """`Factor2D.malz_surface`'s vol column on a tape, row for row and in its order."""
        return torch.cat([cls.carried_sigma(skews[T], carried[T], T, nodes)
                          for T, nodes in grid.items()])

    @classmethod
    def pinned_grid(cls, written, expiries, tolerance):
        """The log-moneyness grid a previously written surface already carries, or None.

        `written` is the PRICE FACTOR block this family wrote last, not a quote block. Four ways to
        get None, and each is a grid that is not the one the quotes are asking for: nothing to pin
        to; a surface on a different SUBTYPE, whose moneyness axis is S/K rather than log(F/K) and
        whose nodes therefore mean something else entirely; a surface over a different set of
        EXPIRIES, which a rebuild answers and stretching a grid does not; and a surface refined to
        a different TOLERANCE, which is the knob that sizes the grid - honouring it on the values
        while ignoring it on the nodes would make it a field nothing reads.
        """
        surface = written.get('Surface') if written else None
        if surface is None or not surface.array.any():
            return None
        if written.get('Surface_Type') != cls.surface_type:
            return None
        if not np.array_equal(np.unique(surface.array[:, 1]), expiries):
            return None
        if written.get('Grid_Tolerance') != tolerance:
            return None
        return {T: surface.array[surface.array[:, 1] == T][:, 0] for T in expiries}

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices,
                  calendars, debug=None):
        """Turn each block's quotes into the log-moneyness `FXVol` surface the pricers read.

        The x-grid is taken from the factor this wrote last if it is still describing the same
        expiries at the same tolerance - see the class docstring on pinning - and refined from
        scratch otherwise.

        A block asking for `Quote_Sensitivity` leaves that surface behind still connected to the
        ATM / RR / BF quotes it was built from, so `Calculation.factor_leaf` can offer the connected
        tensor where it would otherwise mint an `FXVol` leaf out of numpy. The tape is a SPLICE over
        the shipped conversion and not a replacement for it - see `carried_sigma` - so what the
        switch changes is what `backward()` can reach and nothing else. Every number written below
        comes out of `Factor2D.malz_surface` either way.

        THE GRID IS NOT DIFFERENTIATED. It is refined against the quotes when it is BUILT and
        pinned from then on, which is what makes a tick a values patch; the twin moves the VOLS on
        frozen nodes. A rebuild is a new plan, and a derivative across two plans is not a
        derivative.
        """
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                block = implied_params['instrument']
                vol_name = utils.check_tuple_name(
                    utils.Factor(self.price_factor_type, market_factor.name))

                tolerance = float(block.get('Grid_Tolerance', riskfactors.Factor2D.malz_tol))
                # the one bounds= the engine reads: outside it there is no grid to refine to
                if not self.grid_tolerance_bounds[0] <= tolerance <= self.grid_tolerance_bounds[1]:
                    raise ValueError(
                        '{}: Grid_Tolerance {:g} is outside [{:g}, {:g}] - the refinement does not '
                        'terminate there'.format(market_price, tolerance,
                                                 *self.grid_tolerance_bounds))

                quotes = self.used(block)
                delta_surface = self.smile(quotes)
                expiries = np.unique(delta_surface[:, 1])
                skews = riskfactors.Factor2D.malz_skews(delta_surface, expiries)

                grid = self.pinned_grid(price_factors.get(vol_name), expiries, tolerance)
                pinned = grid is not None
                if not pinned:
                    grid = riskfactors.Factor2D.malz_grid(skews, tolerance)

                surface = riskfactors.Factor2D.malz_surface(skews, grid)
                stamps = [point['Timestamp'] for point in quotes if point['Timestamp']]
                price_factors[vol_name] = {
                    'Property_Aliases': None, 'Surface_Type': self.surface_type,
                    'Moneyness_Rule': self.moneyness_rule,
                    'Currency': block.get('Currency', ''),
                    'Grid_Tolerance': tolerance,
                    'Quote_Timestamp': max(stamps) if stamps else '',
                    'Surface': utils.Curve([], surface)}

                if block.get('Quote_Sensitivity', 'No') == 'Yes':
                    leaves = torch.tensor([point['Quoted_Market_Value'] for point in quotes],
                                          dtype=self.dtype, requires_grad=True)
                    carried = self.carried_surface(skews, self.carried_skews(
                        delta_surface, expiries, self.carried_smile(quotes, leaves)), grid)
                    # `Factor2D` sorts what it is handed by (expiry, moneyness) and mints a leaf
                    # out of THAT column, so the twin is put in the same order rather than assumed
                    # to be in it
                    rows = np.array(surface)
                    order = np.lexsort((rows[:, 0], rows[:, 1]))
                    self.calibrated[utils.Factor(self.price_factor_type, market_factor.name)] = \
                        torch.tensor(rows[order, 2], dtype=self.dtype) + (
                            carried[order] - carried[order].detach())
                    self.quote_leaves[market_price] = (
                        [self.descriptor(point) for point in quotes], leaves)

                logging.info('{} built from {} quotes on a {} grid of {} nodes as at {}'.format(
                    vol_name, len(quotes), 'pinned' if pinned else 'refined',
                    sum(len(nodes) for nodes in grid.values()),
                    price_factors[vol_name]['Quote_Timestamp'] or 'no stated time'))
                for T, nodes in grid.items():
                    logging.info('  expiry {:.4f}: {} nodes resolving the smile to {:.3g} vol, '
                                 'built at {:.3g}'.format(
                                     T, len(nodes),
                                     float(riskfactors.Factor2D.malz_error(
                                         skews[T], T, nodes).max()), tolerance))


def construct_bootstrapper(btype, param, dtype=torch.float32):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return globals().get(btype)(param, device, dtype)
