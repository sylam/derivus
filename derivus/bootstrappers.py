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
from collections import namedtuple
from functools import partial

# third party stuff
import numpy as np
import pandas as pd
import torch

# Internal modules
from . import utils, pricing, instruments, riskfactors, stochasticprocess
from .schema import F, OPTION_QUOTE, REQUIRED, Row

import scipy.optimize


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
         '*Volatility* any (moneyness, expiry) vol surface (**VolatilityGrid**, whatever the asset',
         'class); the type of each is looked up from the price factors, or named explicitly',
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
                    'Volatility': ['VolatilityGrid'],
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
        rate = utils.check_rate_name(instrument[field])
        types = [instrument[field + '_Type']] if instrument.get(field + '_Type') else cls.factor_types[field]
        return utils.Factor(next(x for x in types if utils.check_tuple_name(
            utils.Factor(x, rate)) in price_factors), rate)

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

    market_factor_type = 'GBMAssetPriceTSModelPrices'
    fields = [
        F('Asset_Price_Volatility', 'Text', default=REQUIRED,
          description='The VolatilityGrid whose ATM column becomes the integrated vol curve')
    ]

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Checks for Declining variance in the ATM vols of the relevant price factor and corrects accordingly.
        '''
        eq_vols = {}
        fx_vols = {}
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                # get the vol surface
                implied_param = utils.check_rate_name(implied_params['instrument']['Asset_Price_Volatility'])
                vol_factor = utils.Factor('VolatilityGrid', implied_param)
                # asset class is a property of the UNDERLYING, not of its vol surface: one
                # VolatilityGrid serves every asset class, so this asks whether the thing being
                # modelled is an fx rate rather than which type its surface was declared under
                is_fx = utils.check_tuple_name(utils.Factor('FxRate', rate[1:])) in price_factors

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                mn_ix = np.searchsorted(vol_surface.moneyness, 1.0)
                atm_vol = [np.interp(1, vol_surface.moneyness[mn_ix - 1:mn_ix + 1], y) for y in
                           vol_surface.get_vols()[:, mn_ix - 1:mn_ix + 1]]

                # store the output
                price_param = utils.Factor(self.__class__.__name__, market_factor.name)
                model_param = utils.Factor('GBMAssetPriceTSModelImplied', market_factor.name)

                if vol_surface.expiry.size > 1:
                    dt = np.diff(np.append(0, vol_surface.expiry))
                    var = vol_surface.expiry * np.array(atm_vol) ** 2
                    sig = atm_vol[:1]
                    vol = atm_vol[:1]
                    var_tm1 = var[0]
                    fixed_variance = False

                    for var_t, delta_t, t_i in zip(var[1:], dt[1:] / 3.0, vol_surface.expiry[1:]):
                        M = var_tm1 + delta_t * (sig[-1] ** 2)
                        if var_t < M:
                            fixed_variance = True
                            var_t = M

                        a = delta_t
                        b = sig[-1] * delta_t
                        c = M - var_t

                        sig.append((-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a))
                        vol.append(np.sqrt(var_t / t_i))
                        var_tm1 = var_t

                    if fixed_variance:
                        logging.warning('Fixed declining variance for {0}'.format(market_price))
                else:
                    vol = atm_vol

                if is_fx:
                    fx_vols[rate[-1]] = [utils.Curve(['Integrated'], list(zip(vol_surface.expiry, vol))), implied_param]
                    price_factors[utils.check_tuple_name(price_param)] = {
                        'Property_Aliases': None,
                        'Vol': fx_vols[rate[-1]][0],
                        'Quanto_FX_Volatility': None,
                        'Quanto_FX_Correlation': 0.0}
                    price_models[utils.check_tuple_name(model_param)] = {'Risk_Premium': None}
                else:
                    quanto_fx_corr = price_factors.get(
                        'Correlation.EquityPrice.{}.{}/FxRate.{}.{}'.format(
                            rate[-1], implied_param[-1], *sorted([sys_params['Base_Currency'], implied_param[-1]])),
                        {'Value': 0.0})['Value']
                    price_factors[utils.check_tuple_name(price_param)] = {
                        'Property_Aliases': None,
                        'Vol': utils.Curve(['Integrated'], list(zip(vol_surface.expiry, vol))),
                        'Quanto_FX_Volatility': None,
                        'Quanto_FX_Correlation': quanto_fx_corr}
                    price_models[utils.check_tuple_name(model_param)] = {'Risk_Premium': None}
                    # store this for later quanto correction
                    eq_vols[rate[-1]] = [utils.check_tuple_name(price_param), implied_param[-1]]


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

    def unflatten(self, theta):
        """`{name: numpy}` in the closure's own parameter order - the shape `save_params` takes."""
        return dict(zip(self.keys, np.split(
            theta.detach().cpu().numpy(), np.cumsum(self.sizes[:-1]))))

    def __call__(self, x):
        """The residual vector at flat parameters `x`, differentiable in `x` AND in the quotes.

        A FRESH dict of views is handed to the closure rather than the standing `implied_var`,
        because those are leaves whose `.data` the scipy adapters overwrite: `x.split()` keeps the
        graph the Jacobian is read off. `x` carries the closure's own precision, so the residual is
        the same number the solve stopped on - the float64 promotion belongs to the linear algebra
        downstream, not to the pricing.
        """
        return torch.stack(list(self.loss_fn(dict(zip(self.keys, x.split(self.sizes))))[1].values()))

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

                # save this
                final_implied_obj = self.save_params(
                    calibration.unflatten(theta), price_factors, implied_obj, rate)

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

    Both come from autograd on the residual closure itself, evaluated once at `(theta*, q)`: the
    n x n matrix by one backward pass per benchmark, the q side as a single vector-Jacobian product
    with `-w` as its cotangent. So the residual is WRITTEN ONCE AND DIFFERENTIATED TWICE, and the
    quote derivative cannot drift from the one the solve converged on.

    The Jacobian is recomputed at theta* rather than reused from the last Newton step, which was
    taken at the iterate BEFORE it. Cost is one iteration's worth on a system whose dimension is
    the benchmark count.

    Every `grad` here retains the graph. The residual's own subgraph is shared with the forward
    pass - `pv_fixed_cashflows` memoizes its payment tensor in `Factor_dep` - so freeing it would
    take the forward pass's graph with it.
    """

    @staticmethod
    def forward(ctx, benchmarks, seed, n_iter, tol, halvings, quotes):
        with torch.enable_grad():
            theta = damped_newton(benchmarks, seed, n_iter, tol, halvings)
        ctx.benchmarks, ctx.theta = benchmarks, theta
        return torch.cat([theta[factor] for factor in benchmarks.solve_for])

    @staticmethod
    def backward(ctx, cotangent):
        benchmarks, theta = ctx.benchmarks, ctx.theta
        keys = list(benchmarks.solve_for)
        sizes = [theta[factor].numel() for factor in keys]
        with torch.enable_grad():
            x = torch.cat([theta[factor] for factor in keys]).requires_grad_(True)
            residual = benchmarks(dict(zip(keys, x.split(sizes))))
            jacobian = torch.stack([torch.autograd.grad(residual[i], x, retain_graph=True)[0]
                                    for i in range(residual.numel())])
            w = torch.linalg.solve(jacobian.t(), cotangent)
            quote_grad, = torch.autograd.grad(
                residual, benchmarks.quotes, grad_outputs=-w.detach(), retain_graph=True)
        return None, None, None, None, None, quote_grad


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
        """Solve each block for the zero curve that reprices every used quote to par.

        The seed theta is read off the CONSTRUCTED factor rather than off the authored block, so it
        is aligned with the tenor grid the pricers gather against, whatever `get_tenor` made of the
        block. The solve goes through the implicit-function wrapper either way: with no quotes on
        the tape its forward IS `damped_newton` and no edge is recorded, which is what makes
        "gradients cannot move a mark" structural rather than a claim.
        """
        base_date = sys_params['Base_Date']

        for market_price, implied_params in self.in_dependency_order(market_prices):
            block = implied_params['instrument']
            curve = utils.Factor('InterestRate', utils.check_rate_name(market_price)[1:])
            curve_name = utils.check_tuple_name(curve)
            discount_rate = block['Discount_Rate'] or '.'.join(curve.name)

            quotes = [point for point in block['Points'] if point['Use'] == 'Yes'
                      and self.takes(point, market_price)]
            nodes = quote_nodes(quotes, discount_rate)
            connect = block.get('Quote_Sensitivity', 'No') == 'Yes'

            # seed the block first: the closure constructs the curve factor out of `Price Factors`,
            # and a par rate is within a few basis points of the zero rate at the same maturity.
            # `Curve` sorts the pairs, so each knot keeps the quote that identifies it
            price_factors[curve_name] = {
                'Property_Aliases': None, 'Sub_Type': None, 'Currency': block['Currency'],
                'Day_Count': block['Day_Count'], 'Curve': utils.Curve([], list(zip(
                    quote_knots(nodes, base_date, block['Day_Count'], calendars),
                    [point['Quoted_Market_Value'] / 100.0 for point in quotes])))}

            time_now = time.monotonic()
            benchmarks = BenchmarkInstruments(
                nodes, price_factors, factor_interp, base_date, block['Currency'], calendars,
                [curve], self.device,
                quotes=[point['Quoted_Market_Value'] for point in quotes] if connect else None,
                bumped_nodes=quote_nodes(quotes, discount_rate, 1.0) if connect else None)
            # seed theta off the CONSTRUCTED factor - see the docstring on grid alignment
            theta = CalibrationSolve.apply(
                benchmarks,
                {curve: torch.tensor(benchmarks.factors[curve].current_value(),
                                     dtype=BenchmarkInstruments.dtype, device=self.device)},
                int(block.get('N_Iter', 50)), float(block.get('Tol', 1e-14)),
                int(block.get('Damping_Halvings', 6)), benchmarks.quotes)

            price_factors[curve_name]['Curve'] = utils.Curve(
                [], list(zip(benchmarks.tenors[curve], theta.detach().cpu().numpy())))
            if connect:
                self.calibrated[curve] = theta
                self.quote_leaves[market_price] = ([point['Descriptor'] for point in quotes],
                                                   benchmarks.quotes)

            residuals = benchmarks({curve: theta.detach()}).detach()
            logging.info('{} bootstrapped from {} quotes in {:.2f} seconds, residual {:.3g}'.format(
                curve_name, len(quotes), time.monotonic() - time_now,
                float(residuals.abs().max())))
            for point, residual in zip(quotes, residuals):
                logging.info('  {} at {:.4f} reprices to {:.3g}'.format(
                    point['Descriptor'], point['Quoted_Market_Value'], float(residual)))

    def takes(self, point, market_price):
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


def construct_bootstrapper(btype, param, dtype=torch.float32):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return globals().get(btype)(param, device, dtype)
