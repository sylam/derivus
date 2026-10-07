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
from itertools import groupby

# third party stuff
import numpy as np
import pandas as pd
import torch

# Internal modules
from . import utils, pricing, instruments, riskfactors, stochasticprocess, schema
from .schema import (DAY_COUNTS, F, OPTION_QUOTE, PRICES_KEY, QUOTE_TWO_WAY, REQUIRED, Row,
                     completed, declared_defaults, leaf_deals, partition_market_price, quote_table)
from ._version import __version__

import scipy.optimize
import scipy.stats


def resolve_factor(name, price_factors, candidates):
    """The factor `name` refers to, typed by the first candidate the price factors hold a block for.

    A block names its inputs by name only, so the bootstrapper declares the candidate types
    (`utils.TwoDimensionalFactors` is the candidate list for a vol surface)."""
    rate = utils.check_rate_name(name)
    return utils.Factor(next(x for x in candidates if utils.check_tuple_name(
        utils.Factor(x, rate)) in price_factors), rate)


def reference_fields(factor_types, required_by_quote_type, notes):
    """One `F` per named factor reference, then its optional `_Type` sibling.

    The `_Type` values are the candidate list, so a new candidate cannot miss the schema. REQUIRED
    only where every quote type requires it; a reference required under one and inert under another
    states that half through `notes` and is enforced where it is read.

    A module-level function because a class-scope name is not visible inside a class-body
    comprehension, so neither the quote-type map nor `notes` could be read there.
    """
    always = {field for field in factor_types
              if all(field in required for required in required_by_quote_type.values())}
    return [F(field, 'Text', default=REQUIRED if field in always else '',
              description='The {} factor - one of {}{}'.format(
                  field.replace('_', ' ').lower(), ', '.join(types), notes.get(field, '')))
            for field, types in factor_types.items()] + [
               F(field + '_Type', 'Text', default='', values=[''] + list(types),
                 description='Names the factor type explicitly, where the name exists under more than one')
               for field, types in factor_types.items()]


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

    def clear(self):
        """Empties the memo buffers, which every evaluation must do before it starts.

        `t_Buffer` is keyed by factor and time and `t_PreCalc` by factor and integrand, neither by
        the tensor's identity, so a state carried across two parameter sets would answer the second
        call with the first's curves. Only the Monte Carlo objective goes on to need a sample, which
        is why `reset` is a second call.
        """
        self.t_Buffer.clear()
        self.t_PreCalc.clear()

    def reset(self, num_batches, numfactors, time_grid):
        # clear the buffers
        self.clear()

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


class Family:
    """What every price family is built from: the block it was configured with, completed by
    its own declared defaults (a quote's own instrument is unioned onto it and wins on conflict),
    and the device and precision of the job. A family that pins its own precision - the
    LogVar2FJ fit is double whatever the job is - declares `prec` on the class and keeps it.
    """
    prec = None

    def __init__(self, param, device, dtype):
        self.param = declared_defaults(type(self), param)
        self.device = device
        if self.prec is None:
            self.prec = dtype


class Construction(Family):
    """A construction turns quotes into a market factor with no model behind it - deposits, FRAs
    and swaps into a curve, delta quotes into a volatility surface, a volatility strip into a
    lognormal term structure. Deterministic - closed form or a chain of one-dimensional roots, no
    fitted parameters - and cheap enough to run on every tick.

    Every family declares `market_factor_type` (the `Market Prices` type it selects work by),
    `price_factor_type` (the `Price Factors` block it writes, which keys its `Bootstrapper
    Configuration` entry) and `reads` (what orders a run against what other families write).
    """


class ImpliedCalibration(Family):
    """An implied calibration turns factors plus benchmark instruments into model parameters by a
    solve - Hull-White off the curve and the swaption volatilities, LogVar2FJ off the surface's
    ladder, Clewlow-Strickland off energy futures options. Seconds rather than milliseconds, and
    only worth running when what it reads has moved.

    Declares the same three as a construction.
    """


class CSForwardPriceModelParameters(ImpliedCalibration):
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
    #: The `Price Factors` type this family writes, which is what a `Bootstrapper
    #: Configuration` entry names it by - here, its own class name.
    price_factor_type = 'CSForwardPriceModelParameters'
    #: The `Price Factors` types this family READS, which is what orders a run against
    #: what other families write: `Energy`, `Forward_Volatility` and `Discount_Rate`.
    reads = ('ForwardPrice', 'ForwardPriceVol', 'InterestRate')
    fields = [
        F('Energy', 'Text', default=REQUIRED, description='The ForwardPrice factor to calibrate'),
        F('Forward_Volatility', 'Text', default=REQUIRED,
          description='The ForwardPriceVol surface the quoted vols are read off'),
        F('Discount_Rate', 'Text', default=REQUIRED,
          description='The InterestRate curve the premiums discount on'),
        F('Quote_Type', 'Text', default='Implied_Volatility', values=['Implied_Volatility'],
          description='How Quoted_Market_Value reads - this family takes vols only'),
        F('Sigma_Bounds', 'Text', default='0.001,2.5',
          description='The box the instantaneous vol is fitted in, lower,upper'),
        F('Alpha_Bounds', 'Text', default='-1.0,2.0',
          description='The box the Clewlow-Strickland decay is fitted in, lower,upper; a negative '
                      'alpha grows the vol out to settlement'),
        F('Seed', 'Text', default='0.5,0.1',
          description='Where the local minimisation starts on a cold block, '
                      'sigma,alpha, each strictly inside its box; a block whose '
                      'parameter factor exists starts off that'),
        F('Energy_Futures_Options', 'Table', default='null',
          row=Row(OPTION_QUOTE[:1] + [F('Settlement_Date', 'Date',
                                        description='Futures settlement, which sets the '
                                                    'Clewlow-Strickland decay term')] +
                  OPTION_QUOTE[1:]),
          description='The option quotes sigma and alpha are fitted to')
    ]

    @staticmethod
    def typed(block):
        """`(seed, sigma box, alpha box)` off one block - THE ONE READ, performed at construction
        so a malformed field refuses before a quote is read and again per block, where a quote's
        own instrument may have overridden it."""
        return (utils.LogVar2FJ.parse_floats(block['Seed'], 'Seed', 2),
                utils.LogVar2FJ.parse_bounds(block['Sigma_Bounds'], 'Sigma_Bounds'),
                utils.LogVar2FJ.parse_bounds(block['Alpha_Bounds'], 'Alpha_Bounds'))

    def __init__(self, param, device, dtype):
        super().__init__(param, device, dtype)
        self.typed(self.param)

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars):
        '''
        Checks for Declining variance in the ATM vols of the relevant price factor and corrects accordingly.
        '''

        def B(a, t):
            return -np.expm1(-a * t) / a if a != 0 else t

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
                # the quote's own instrument wins on conflict
                implied_params = dict(implied_params, instrument=dict(
                    self.param, **implied_params['instrument']))
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

                block = implied_params['instrument']
                seed, sigma_box, alpha_box = self.typed(block)
                price_param = utils.check_tuple_name(
                    utils.Factor(self.__class__.__name__, market_factor.name))
                # this family's own written factor IS the warm start: present, the minimisation
                # starts at it (clipped to the declared boxes) rather than at Seed
                previous = price_factors.get(price_param)
                if previous is not None:
                    seed = [np.clip(previous['Sigma'], *sigma_box),
                            np.clip(previous['Alpha'], *alpha_box)]
                    logging.info('{} - warm start off {}'.format(market_price, price_param))
                result = scipy.optimize.minimize(
                    calc_error, seed, args=(block['Energy_Futures_Options'],),
                    bounds=[sigma_box, alpha_box])

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

                price_factors[price_param] = {
                    'Property_Aliases': None,
                    'Sigma': result.x[0],
                    'Alpha': result.x[1]}


#: The six ladder dials of one family's `Bootstrapper Configuration` entry, in the units
#: `fx_surface_block` works in: `atm`, `wings` and `tolerance` in YEARS, `pillars` as delta
#: magnitudes, `days` the year the emitted dates are counted in, `minimum` a contract count.
FxLadder = namedtuple('ladder', 'atm wings pillars days tolerance minimum')


class OptionQuoteFamily(ImpliedCalibration):
    documentation = (
        'Fx And Equity',
        ['The quote preparation every European option family in this module shares - what a block',
         'names, what its quote type reads, the forward its premia are priced off and where the',
         'vol surface is looked up. The MODEL is the subclass\'s; this is the ladder.',
         '',
         'A family here is ASSET CLASS AGNOSTIC - the *Underlying* may be any spot (0D) price',
         'factor (**FxRate**, **EquityPrice**, **CommodityPrice**, **FuturesPrice**) and the',
         '*Volatility* any (moneyness, expiry) vol surface (**FXVol**, **EquityPriceVol**,',
         '**CommodityPriceVol**); the type of each is looked up from the price factors, or named',
         'explicitly with *Underlying_Type* / *Volatility_Type*. The optional *Yield* (a dividend,',
         'repo, convenience or carry curve) enters as $q$ - the drift is $r-q$ and the value',
         'carries the extra $e^{-qt}$ factor - so equity, FX and commodity underlyings are all',
         'handled by the same objective.',
         '',
         'THE FORWARD IS THE PRICER\'S. $r$ is the *Discount_Rate* curve and is what the premium',
         'discounts on; the forward GROWS at the optional *Funding_Rate* curve instead where one is',
         'named, which is the curve `utils.calc_eq_forward` integrates - an equity\'s own repo curve',
         '(**EquityPrice.Interest_Rate**), not the curve its deals discount on. Left blank the two are',
         'one curve, which is the one-curve world and what an FX pair always was; named, an index',
         'carrying a repo/borrow spread calibrates at the forward it is priced at rather than one a',
         'spread away from it.',
         '',
         'Target premia are the Black prices at the corresponding vol surface point (as per the Clewlow',
         'Strickland bootstrapper) unless *Quote_Type* is **Premium**, in which case the quoted values are',
         'used directly. A previously bootstrapped price factor (if present) is used to warm start the fit.',
         '',
         'WHICH REFERENCES ARE REQUIRED IS THE QUOTE TYPE\'S. **Implied_Volatility** reads *Underlying*,',
         '*Volatility* and *Discount_Rate*; **Premium** reads *Underlying* and *Discount_Rate* and NO',
         'surface at all - a listed chain is calibrated to its own prints rather than to somebody\'s fit',
         'to them. *Yield* and *Funding_Rate* are optional under both. A reference a block does not name',
         'and its quote type reads REFUSES by name; it does not skip.',
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

    # Every premium here is read in double: the framework default (float32) destroys the
    # cancellation, so the dtype this is constructed with is deliberately ignored.
    prec = torch.float64
    # candidate types per input: any spot (0D) factor, any (moneyness, expiry) surface - so one
    # instrument definition serves FX, equity and commodity underlyings
    factor_types = {'Underlying': ['FxRate', 'EquityPrice', 'CommodityPrice', 'FuturesPrice'],
                    'Priced_In': ['FxRate'],
                    'Volatility': utils.TwoDimensionalFactors,
                    'Discount_Rate': ['InterestRate'],
                    'Yield': ['DividendRate', 'InterestRate'],
                    'Funding_Rate': ['InterestRate']}

    #: What this family READS, flattened out of `factor_types` - `reads` is what orders a run and
    #: `factor_types` is the reference resolution, so neither is overloaded with the other's job.
    reads = tuple(sorted({kind for kinds in factor_types.values() for kind in kinds}))

    #: What each quote type requires. `Implied_Volatility` prices its target premium off the
    #: surface; `Premium` is handed the number, so a `Volatility` name on such a block is inert and
    #: a chain-sourced ladder fits with no surface in the book at all. Declared here rather than as
    #: a `default=REQUIRED`, which one static default cannot express.
    quote_type_references = {'Implied_Volatility': ('Underlying', 'Volatility', 'Discount_Rate'),
                             'Premium': ('Underlying', 'Discount_Rate')}

    #: The carry references, read whenever named and required by no quote type: no `Yield` is no
    #: carry, no `Funding_Rate` is a forward funded by `Discount_Rate`. A fit reads these plus what
    #: its quote type requires; a reference in neither list is not looked at.
    optional_references = ('Priced_In', 'Yield', 'Funding_Rate')

    #: What an optional reference's absence means, appended to its declared description.
    reference_notes = {
        'Priced_In': '. The currency the Underlying is priced in where that is not the book\'s '
                     'base: the spot is read as the RATIO of the two, so a CROSS is fitted on the '
                     'pair\'s own axis rather than on either leg\'s base-priced one. Blank is the '
                     'base, whose own rate is identically one. Never a two-token FxRate name, '
                     'which discovery reads as a spot plus an ObservedBasis tail',
        'Volatility': '. REQUIRED under Quote_Type Implied_Volatility, which prices its target '
                      'premium off it; INERT under Premium, where the quote IS the premium and no '
                      'surface is read at all',
        'Yield': '. Blank is no carry, q = 0',
        'Funding_Rate': '. The curve the FORWARD grows at - an equity\'s own repo curve '
                        '(EquityPrice.Interest_Rate), which is what utils.calc_eq_forward '
                        'integrates, rather than the curve the premium discounts on. Blank funds '
                        'the forward off Discount_Rate, which is the one-curve world and what an '
                        'FX pair always was'}
    # Surface_Types whose vol at a strike is a table lookup, hence usable here. SVI/Skew are
    # parametric - Factor2D returns the parameters, not a vol - so a synthesised premium would be
    # silently wrong.
    tabular_surfaces = ('Explicit', 'Relative_Forward', 'Malz')

    # the five factor references, each with the optional `_Type` `resolve` reads; what is REQUIRED
    # is derived from `quote_type_references` - see `reference_fields`
    fields = reference_fields(factor_types, quote_type_references, reference_notes) + [
        F('Quote_Type', 'Text', default='Implied_Volatility',
          values=['Implied_Volatility', 'Premium'],
          description='Whether Quoted_Market_Value is a vol to price at or a premium to fit'),
        F('Use_Forward', 'Text', default='No', values=['Yes', 'No'],
          description='Moneyness against the forward rather than the spot'),
        F('Invert_Moneyness', 'Text', default='No', values=['Yes', 'No'],
          description='Moneyness as K/S rather than S/K'),
        F('Steps_Per_Year', 'Float', default=252.0,
          description='GARCH steps an expiry is spread over'),
        F('Quote_Timestamp', 'Date', default='',
          description='When the quotes were seen, the vol surface\'s own as-of where the block was '
                      'authored off one; stored and reported, never read by the fit'),
        F('Quote_Source', 'Text', default='',
          description='How this block was authored, in one line: what the vols were read off and '
                      'any nearest quoted expiry used in place of one the ladder asks for'),
        F('ATM_Expiries', 'Text', default='1,2,3,6,9,12',
          description='The ATM term structure quoted, in months, comma-separated and increasing; '
                      'the longest rung also caps how far out a quote is read'),
        F('Wing_Expiries', 'Text', default='3,6',
          description='The expiries each delta pillar is quoted at on both wings, in months, which '
                      'identify the skew and the wings\' width'),
        F('Wing_Pillars', 'Text', default='0.25',
          description='The delta magnitudes each wing expiry is quoted at, each in (0, '
                      '0.5], 0.5 being the straddle'),
        F('Days_Per_Year', 'Float', default=365.0,
          description='Days in a year when a surface expiry in years is emitted as an Expiry_Date, '
                      'rounded to the nearest whole day'),
        F('Expiry_Tolerance', 'Float', default=7.0,
          description='How far past the ladder\'s longest rung a surface pillar may still be '
                      'snapped to, in days of Days_Per_Year'),
        F('Minimum_Contracts', 'Integer', default=6,
          description='Distinct (expiry, strike) contracts the ladder must survive snapping with'),
        F('European_Options', 'Table', default='null', row=Row(OPTION_QUOTE + QUOTE_TWO_WAY),
          description='The option quotes the five parameters are fitted to, each with the two-way '
                      'it was dealt on and the print\'s own clock where the source printed them')]

    def __init__(self, param, device, dtype):
        super(OptionQuoteFamily, self).__init__(param, device, dtype)
        #: This family's own ladder, typed once - so a malformed dial refuses at construction,
        #: before a quote is read, as every other hyperparameter does.
        self.ladder = self.fx_ladder(self.param)

    @classmethod
    def fx_ladder(cls, section=None):
        """The ladder off a `Bootstrapper Configuration` entry completed by this family's own
        declarations; `None` - or the legacy CSV string, which declares no dial - is those
        declarations alone.

        Months and days become YEARS here, which is the surface's own clock and the only one past
        this point. Refuses BY NAME on a ladder out of order, a delta pillar outside (0, 0.5] and
        a floor below one contract.
        """
        read = declared_defaults(cls, section if isinstance(section, dict) else {})
        days = float(read['Days_Per_Year'])
        months = lambda name: tuple(
            x / 12.0 for x in utils.LogVar2FJ.parse_floats(read[name], name))
        ladder = FxLadder(
            atm=months('ATM_Expiries'), wings=months('Wing_Expiries'),
            pillars=utils.LogVar2FJ.parse_floats(read['Wing_Pillars'], 'Wing_Pillars'),
            days=days, tolerance=float(read['Expiry_Tolerance']) / days,
            minimum=int(read['Minimum_Contracts']))
        for name, rungs in (('ATM_Expiries', ladder.atm), ('Wing_Expiries', ladder.wings)):
            if not all(before < rung for before, rung in zip((0.0,) + rungs, rungs)):
                raise ValueError(
                    '{}: the rungs are positive months in increasing order, read {!r}. An '
                    'unordered ladder quotes one expiry twice and calls it a term '
                    'structure'.format(name, read[name]))
        if not all(0.0 < pillar <= 0.5 for pillar in ladder.pillars):
            raise ValueError(
                'Wing_Pillars: a delta pillar is a magnitude in (0, 0.5], read {!r}. 0.5 is the '
                'straddle and past it the call pillar names the put wing'.format(
                    read['Wing_Pillars']))
        if ladder.minimum < 1:
            raise ValueError(
                'Minimum_Contracts: the floor on distinct contracts reads {}. A ladder that '
                'survives snapping with nothing identifies nothing'.format(ladder.minimum))
        return ladder

    @classmethod
    def resolve(cls, instrument, field, price_factors):
        """The factor named by instrument[field], typed by the first candidate that exists in the
        price factors, or by an explicit instrument[field + '_Type']. None if the field is unset."""
        if not instrument.get(field):
            return None
        return resolve_factor(instrument[field], price_factors, [instrument[field + '_Type']]
        if instrument.get(field + '_Type') else cls.factor_types[field])

    @classmethod
    def resolve_references(cls, market_price, instrument, price_factors, factor_interp):
        """`{field: constructed factor}` for every reference this block names and its quote type
        reads - and a named refusal for every one it needs and cannot get.

        A missing reference refuses; it does not skip. What is required is the quote type's
        (`quote_type_references`), so a `Volatility` name on a `Premium` block is not resolved at
        all; `optional_references` are read whenever named. Everything is resolved before an option
        is looked at, so a book carrying two ladders fails on the one that is wrong.
        """
        quote_type = instrument.get('Quote_Type')
        if quote_type not in cls.quote_type_references:
            raise ValueError(
                '{}: Quote_Type {!r} is not one this family fits - it takes {}. Implied_Volatility '
                'prices each quote at the Volatility surface and fits that premium; Premium fits '
                'the number in Quoted_Market_Value directly'.format(
                    market_price, quote_type, ' or '.join(sorted(cls.quote_type_references))))
        required = cls.quote_type_references[quote_type]
        resolved = {}
        for field in tuple(required) + cls.optional_references:
            if not instrument.get(field):
                if field in required:
                    raise ValueError(
                        '{}: Quote_Type {} requires {}, and {} is blank. Name the factor the fit '
                        'should read; a reference the block does not name is a calibration that '
                        'writes no price factor{}'.format(
                            market_price, quote_type, '/'.join(required), field,
                            '. Quote the premiums directly (Quote_Type Premium), which requires '
                            'only {}, if the book carries no surface to price them off'.format(
                                '/'.join(cls.quote_type_references['Premium']))
                            if field == 'Volatility' else ''))
                continue
            try:
                factor = cls.resolve(instrument, field, price_factors)
            except StopIteration:
                raise ValueError(
                    '{}: {} names {!r} and the book\'s Price Factors carry no {} block for it. Add '
                    'the factor, or point the field at one the book carries'.format(
                        market_price, field, instrument[field],
                        '/'.join('{}.{}'.format(candidate, instrument[field])
                                 for candidate in ([instrument[field + '_Type']]
                                                   if instrument.get(field + '_Type')
                                                   else cls.factor_types[field]))))
            try:
                resolved[field] = riskfactors.construct_factor(
                    factor, price_factors, factor_interp)
            except Exception as failure:
                raise ValueError(
                    '{}: {} names {!r}, which resolved to {} and would not construct: {}'.format(
                        market_price, field, instrument[field],
                        utils.check_tuple_name(factor), failure))
        return resolved

    def tensor(self, x):
        """A scalar leaf on the fit's device and precision; a None - no cap - stays None."""
        return None if x is None else torch.tensor(float(x), device=self.device, dtype=self.prec)

    def vector(self, xs):
        return torch.tensor([float(x) for x in xs], device=self.device, dtype=self.prec)

    @classmethod
    def moneyness(cls, strike, spot, forward, vol_surface, use_forward, invert_moneyness):
        """The moneyness coordinate to look the vol surface up at.

        Five conventions, dispatched off the surface's SubType, so this delegates to
        `pricing.calc_moneyness` - the same function every option deal uses. That reads only the
        SubType out of deal_data, so a minimal `DealDataType` carrying it is all it needs.
        """
        deal_data = utils.DealDataType(
            Instrument=None, Time_dep=None, Calc_res=None,
            Factor_dep={'Volatility': [(None, None, vol_surface.get_subtype())]})
        return float(pricing.calc_moneyness(
            *[torch.tensor(float(x), dtype=cls.prec) for x in (strike, spot, forward)],
            deal_data, use_forward, invert_moneyness))

    @staticmethod
    def fx_surface_expiry(surface, expiry, cap, tolerance):
        """The surface's own expiry nearest `expiry` at or under `cap`, and whether it had to
        substitute. `(None, True)` where the surface carries no admissible pillar at all.

        The quote moves to the nearest pillar the surface was BUILT from and the block records it in
        `Quote_Source`; interpolating between two would put a number nobody quoted into the
        objective. `cap` is the ladder's longest rung widened by `tolerance`, and a rung with
        nothing admissible under it is dropped and recorded.
        """
        admissible = surface.expiry[surface.expiry <= cap + tolerance]
        if not admissible.size:
            return None, True
        nearest = float(admissible[np.argmin(np.abs(admissible - expiry))])
        return nearest, not np.isclose(nearest, expiry)

    @staticmethod
    def fx_atm_coordinate(vol_at, T, iterations=64):
        """`(x, vol)` of the delta-neutral straddle on a Malz surface at expiry `T`.

        The ATM convention `FXVolSurfaceParameters` writes and `Factor2D.malz_skew` places the +-0.5
        label's vol at: `K = F exp(-sigma^2 T/2)`, so `x = sigma^2 T / 2` with `sigma` the surface's
        own vol at that x. Reading at `x = 0` would be the ATMF vol, a different number on a skewed
        smile. Iterated as a fixed point - a contraction, slope of order 1e-2 here.
        """
        vol = vol_at(0.0)
        for _ in range(iterations):
            moved = vol_at(0.5 * vol * vol * T)
            if abs(moved - vol) < 1e-14:
                vol = moved
                break
            vol = moved
        return 0.5 * vol * vol * T, vol

    @staticmethod
    def fx_pillar_delta(vol_at, T, x, side):
        """The premium-adjusted forward delta magnitude at log-moneyness `x = log(F/K)`, on the call
        wing (`side` +1) or the put wing (-1) - `(K/F)N(d2)` and `(K/F)N(-d2)`.

        The one delta convention the Malz solve inverts, so inverting this finds the strike that
        solve placed the pillar's vol at. `d2` is built off the surface's own vol at `x`, which
        makes it a smile delta rather than a flat-vol one.
        """
        vol = vol_at(x)
        d2 = (x - 0.5 * vol * vol * T) / (vol * np.sqrt(T))
        return float(np.exp(-x) * scipy.stats.norm.cdf(side * d2))

    @classmethod
    def fx_pillar_coordinate(cls, vol_at, T, pillar, side, x_atm, iterations=100):
        """The log-moneyness whose premium-adjusted forward delta is `pillar` on one wing.

        Bisection between the delta-neutral straddle and a strike three log-units out: the delta is
        monotone in `x` along each wing. An unbracketed pillar refuses by name rather than clamping,
        a clamped strike being one that enters the objective as if it were the wing.
        """
        far = 3.0
        low, high = (-far, x_atm) if side > 0 else (x_atm, far)
        error = lambda x: cls.fx_pillar_delta(vol_at, T, x, side) - pillar
        if error(low) * error(high) > 0.0:
            raise ValueError(
                'the {:g} delta {} at expiry {:.4f} is not reachable on this surface - its delta '
                'runs from {:.4f} to {:.4f} over log-moneyness [{:g}, {:g}]. Quote the wing the '
                'surface carries, or widen it'.format(
                    pillar, 'call' if side > 0 else 'put', T,
                    cls.fx_pillar_delta(vol_at, T, low, side),
                    cls.fx_pillar_delta(vol_at, T, high, side), low, high))
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            if error(low) * error(middle) <= 0.0:
                high = middle
            else:
                low = middle
        return 0.5 * (low + high)

    @staticmethod
    def fx_black_vega(forward, strike, rate, vol, T):
        """Black vega of one unit of the option - `exp(-rT) F n(d1) sqrt(T)`.

        The objective weight before normalisation: vega makes it scale-free across a term structure,
        where an unweighted least squares would fit the back end alone. Puts and calls share it.
        """
        stddev = vol * np.sqrt(T)
        d1 = (np.log(forward / strike) + 0.5 * stddev * stddev) / stddev
        return float(np.exp(-rate * T) * forward * scipy.stats.norm.pdf(d1) * np.sqrt(T))

    @classmethod
    def fx_surface_block(cls, pair, price_factors, sys_params, factor_interp,
                         leverage_prior=None, section=None):
        """`(Market Prices name, block)` - this family's quote block, authored off a pair's built
        `FXVol` surface.

        THE LADDER IS THE BOOK'S. `section` is this family's `Bootstrapper Configuration` entry and
        every rung, pillar and floor below is read off it through the family's own declarations;
        `None` is those declarations alone, which is what a caller holding no book gets.

        Vega-weighted implied vols read off the surface - ATM at `ATM_Expiries`, plus each of
        `Wing_Pillars` on both wings at `Wing_Expiries` - normalised by Black vega off the same
        surface. An expiry the surface does not carry moves to the nearest quoted one at or under
        the longest ATM rung, or is dropped where it carries none; `Quote_Source` records either.

        Ten rungs are not ten quotes: a substituted rung lands on a contract another rung already
        named, and a repeat is a weight rather than an observation. So DISTINCT `(expiry, strike)`
        contracts are counted after snapping and a ladder below `Minimum_Contracts` refuses.

        The vols are the surface's, UNSHIFTED - `Volatility_Delta` is a shift the fit applies to
        every quoted vol it prices a premium off, and applying it here too would bump twice.

        TWO CLOCKS, deliberately. `T` is the surface's own expiry axis and is what the surface is
        read at; `t` is what the emitted `Expiry_Date` resolves to through the discount curve's day
        count and is what the FORWARD hangs off. They agree only under ACT_365 - reading the surface
        at `t` under ACT_360 puts the 1Y rung past the last expiry the surface carries.

        The strikes are the surface's own coordinates: the ATM one is the delta-neutral straddle
        `K = F exp(-sigma^2 T/2)`, each wing the strike whose premium-adjusted forward delta is one
        of `Wing_Pillars`, found by inverting the delta the Malz solve inverted off the same vols.

        No `Funding_Rate` is declared, and an FX pair needs none: `Discount_Rate` and `Yield` are
        exactly the pair `utils.calc_fx_forward` builds the priced forward from, so the calibrated
        forward already grows at the curve the pricer grows it on.

        THE DESK'S LEVERAGE PRIOR arrives ALREADY ON THE AXIS THIS FITS, where one is handed in and
        this family declares the field: what a seed states its number about is the seed's own
        business, and the caller is what knows both. This writes it as given and `Quote_Source`
        records which token the axis prices in what, so a number on the wrong axis is readable.

        ORIENTATION. An `FXVol.A.B` x-axis is `log(F/K)` for `A` priced in `B`, while the `FxRate`
        fitted is priced in the DOMESTIC currency - so the underlying is whichever token is not
        domestic, and the block declares `Use_Forward` Yes with `Invert_Moneyness` as the deal sets
        it. Inverting flips the sign of Gamma_Star's skew, so what is written describes the rate the
        pricer simulates, orientation included.

        THE AXIS IS THE PAIR'S AND THE SURFACE IS THE BOOK'S. Which rate is fitted comes from
        `utils.spot_model_pair`, the rule a deal's own lookup takes - the non-base token priced in
        the base, or for a CROSS the alphabetically later token priced in the earlier - so one pair
        has one law however a desk spells its surface. The surface itself is found in whichever
        order the book stores it and `invert` absorbs the difference; a cross declares the quote
        leg as `Priced_In`, reads its spot as the ratio of the two base-priced rates and is filed
        under `<later>.<earlier>`, while a pair with a base leg declares no `Priced_In`, keeps its
        one-token name and reads a denominator of exactly one.

        Refuses by name, with the remedy, on: a pair the book carries neither spelling of or both,
        a surface type no strike can be looked up on, a ladder below `Minimum_Contracts`, and a
        missing spot or discount curve.
        """
        ladder = cls.fx_ladder(section)
        asked = utils.check_rate_name(pair)
        # a surface is a property of the PAIR and a book stores it one way round; the fitted axis
        # is not the stored order's (`utils.spot_model_pair`) and `invert` absorbs the difference,
        # exactly as it already does for a base leg quoted the other way up
        orders = [utils.check_tuple_name(utils.Factor('FXVol', order))
                  for order in dict.fromkeys((asked, tuple(reversed(asked))))]
        # the spelling ASKED FOR first, so a book carrying both is fitted off the one named and no
        # refusal is added where there was none
        vol_name = next((order for order in orders if order in price_factors), None)
        if vol_name is None:
            raise ValueError(
                'no {} in the book\'s Price Factors - there is no built surface to read {} off. '
                'Tick the pair\'s FXVolPrices block first (/book/market or /book/bloomberg), '
                'which bootstraps it'.format(' or '.join(orders), pair))
        name = utils.check_rate_name(vol_name)[1:]
        if len(name) != 2:
            raise ValueError(
                '{} names {} currenc{} - a pair\'s law is one rate priced in another, so the '
                'surface this reads is quoted on exactly two. Name the pair itself'.format(
                    vol_name, len(name), 'y' if len(name) == 1 else 'ies'))

        surface = riskfactors.construct_factor(
            utils.Factor('FXVol', name), price_factors, factor_interp)
        subtype = surface.get_subtype()
        if subtype[0] not in cls.tabular_surfaces:
            raise ValueError(
                '{} has Surface_Type {} - only {} surfaces carry a vol AT A STRIKE, which is what '
                'a quote is. Author the quotes as premiums (Quote_Type Premium) instead'.format(
                    vol_name, subtype[0], '/'.join(cls.tabular_surfaces)))

        # `Factor2D.current_value` only interpolates a grid with two of each coordinate; handed a
        # degenerate one it answers the whole flat vol vector
        if surface.expiry.size < 2 or surface.moneyness.size < 2:
            raise ValueError(
                '{} carries {} expiries x {} moneyness nodes - a surface has to be a grid before a '
                'vol can be read off it at a strike. Quote the pair at more than one '
                'expiry'.format(vol_name, surface.expiry.size, surface.moneyness.size))

        base_date = sys_params['Base_Date']
        base = sys_params.get('Base_Currency', 'USD')
        # THE FITTED AXIS IS THE PAIR'S, never the stored order's: the deal's own rule, so a fit
        # and a lookup cannot pick different laws off a spelling
        underlying, priced_in = utils.spot_model_pair(name[0], name[1], base)
        domestic = priced_in or base
        priced_in = priced_in or ''
        invert = name[0] == domestic

        rates = {}
        for currency in dict.fromkeys((underlying, domestic)):
            spot_name = utils.check_tuple_name(utils.Factor('FxRate', (currency,)))
            if spot_name not in price_factors:
                raise ValueError(
                    'no {} in the book\'s Price Factors - a smile is quoted around a spot, and '
                    'the parameters this writes describe {} priced in {}, which is the ratio of '
                    'those two rates. Add the FxRate block for {}'.format(
                        spot_name, underlying, domestic, currency))
            rates[currency] = riskfactors.construct_factor(
                utils.Factor('FxRate', (currency,)), price_factors, factor_interp)
        # ONE read per leg, and it is the PRICER'S: `calc_fx_forward` grows each rate on the curve
        # its own `FxRate` names, so the fit's forward is the forward the deal is priced on
        curve_of = lambda currency: price_factors[utils.check_tuple_name(
            utils.Factor('FxRate', (currency,)))].get('Interest_Rate') or currency
        carry_name, discount_name = curve_of(underlying), curve_of(domestic)
        for currency, curve in ((underlying, carry_name), (domestic, discount_name)):
            if utils.check_tuple_name(
                    utils.Factor('InterestRate', utils.check_rate_name(curve))) not in price_factors:
                raise ValueError(
                    'no InterestRate.{0} in the book\'s Price Factors - the strikes hang off the '
                    'forward, and the forward is this pair\'s two curves. Add the {0} curve, or '
                    'point FxRate.{1}\'s Interest_Rate at a curve the book '
                    'carries'.format(curve, currency))

        spot = cls.spot_priced_in(
            float(rates[underlying].current_value()[0]),
            rates[domestic] if priced_in else None)
        discount = riskfactors.construct_factor(
            utils.Factor('InterestRate', utils.check_rate_name(discount_name)),
            price_factors, factor_interp)
        carry = riskfactors.construct_factor(
            utils.Factor('InterestRate', utils.check_rate_name(carry_name)),
            price_factors, factor_interp)
        cap = max(ladder.atm)

        def pillar(expiry):
            """One admissible expiry's `(T, moved, days, t, F, r, vol_at)`, or `None` where the
            surface carries no pillar the ladder may snap to. `T` is the surface's coordinate and
            `t` the emitted date's accrual - see the two-clock note in `fx_surface_block`."""
            T, moved = cls.fx_surface_expiry(surface, expiry, cap, ladder.tolerance)
            if T is None:
                return None
            days = int(round(T * ladder.days))
            t = discount.get_day_count_accrual(base_date, days)
            rate = float(discount.current_value(t))
            forward = spot * np.exp((rate - float(carry.current_value(t))) * t)
            # the surface unshifted - the fit applies `Volatility_Delta` to every vol it reads
            return T, moved, days, t, forward, rate, (
                lambda x: float(surface.current_value([[x, T]])[0]))

        quotes, substituted = [], []

        def quote(days, forward, rate, t, x, vol):
            """One `OPTION_QUOTE` row: the strike this coordinate names in the underlying's own
            units (inverting `calc_moneyness`, as `Invert_Moneyness` declares), the surface's vol
            there, and the Black vega that becomes its weight."""
            strike = forward * np.exp(x if invert else -x)
            quotes.append({
                'Expiry_Date': base_date + pd.DateOffset(days=days), 'Strike': strike,
                # the OTM leg, which is the one a desk deals; the fit is blind to the choice, puts
                # being priced by parity off the call
                'Option_Type': 'Call' if strike >= forward else 'Put', 'Units': 1.0,
                'Weight': cls.fx_black_vega(forward, strike, rate, vol, t),
                'Quoted_Market_Value': vol})

        for expiry in ladder.atm:
            found = pillar(expiry)
            if found is None:
                substituted.append('ATM {:g} DROPPED - no pillar at or under {:g}'.format(
                    expiry, cap))
                continue
            T, moved, days, t, forward, rate, vol_at = found
            x, vol = cls.fx_atm_coordinate(vol_at, T)
            quote(days, forward, rate, t, x, vol)
            if moved:
                substituted.append('ATM {:g} -> {:g}'.format(expiry, T))

        wings = '/'.join('{:g}'.format(x) for x in ladder.pillars)
        for expiry in ladder.wings:
            found = pillar(expiry)
            if found is None:
                substituted.append('{}d {:g} DROPPED - no pillar at or under {:g}'.format(
                    wings, expiry, cap))
                continue
            T, moved, days, t, forward, rate, vol_at = found
            x_atm, _ = cls.fx_atm_coordinate(vol_at, T)
            for delta in ladder.pillars:
                for side in (1.0, -1.0):
                    x = cls.fx_pillar_coordinate(vol_at, T, delta, side, x_atm)
                    quote(days, forward, rate, t, x, vol_at(x))
            if moved:
                substituted.append('{}d {:g} -> {:g}'.format(wings, expiry, T))

        # a repeated contract is a weight rather than an observation, so what is counted is the
        # number of DISTINCT (expiry, strike) contracts
        contracts = {(point['Expiry_Date'], point['Strike']) for point in quotes}
        if len(contracts) < ladder.minimum:
            raise ValueError(
                '{} carries pillars {} - the ladder (ATM {}, {}d wings {}) collapses onto {} '
                'distinct contract{} on it, and {} do not identify {}, and a collapsed ladder has '
                'no term structure in it. Quote the pair at more expiries (at least {} distinct '
                'contracts, so at least three pillars at or under {:g}), or author the '
                '{} block by hand. What each rung did: {}'.format(
                    vol_name, '/'.join('{:g}'.format(x) for x in surface.expiry),
                    '/'.join('{:g}'.format(x) for x in ladder.atm), wings,
                    '/'.join('{:g}'.format(x) for x in ladder.wings), len(contracts),
                    '' if len(contracts) == 1 else 's', len(contracts),
                    cls.identification_note, ladder.minimum, cap,
                    cls.market_factor_type,
                    ', '.join(substituted) or 'every rung landed on a pillar it was asked for'))

        # normalised over the rungs as emitted - the weights are relative in the objective
        total = sum(point['Weight'] for point in quotes)
        if not total > 0.0:
            raise ValueError('{} priced every quote at zero vega, so there is no weight to '
                             'normalise and nothing the fit would be sensitive to. Quote the '
                             'surface at a positive vol'.format(vol_name))
        for point in quotes:
            point['Weight'] /= total

        source = '{} ATM {} + {}d wings {}, off {} as at {}'.format(
            len(quotes), '/'.join('{:g}'.format(x) for x in ladder.atm),
            wings, '/'.join('{:g}'.format(x) for x in ladder.wings), vol_name,
            price_factors[vol_name].get('Quote_Timestamp') or 'no stated time')
        if substituted:
            source += ('; rungs the surface does not carry, moved to the nearest quoted at or '
                       'under {:g} or dropped where it carries none: {}'.format(
                cap, ', '.join(substituted)))

        declared = {field.name: field.default for field in cls.fields}
        # written exactly where the family DECLARES it, as the step clock is: `fx_surface_block` is
        # inherited and a family with no leverage in it has no prior to state
        desk = {} if leverage_prior is None or 'Leverage_Prior' not in declared else {
            'Leverage_Prior': float(leverage_prior)}
        if desk:
            source += ('; Leverage_Prior {:+g} on FxRate.{}\'s own axis priced in {}, off the '
                       'desk seed'.format(float(leverage_prior), underlying, domestic))
        return utils.check_tuple_name(utils.Factor(
            cls.market_factor_type,
            (underlying,) + ((priced_in,) if priced_in else ()))), {
                   'instrument': {
                       **desk,
                       'Underlying': underlying, 'Underlying_Type': 'FxRate', 'Priced_In': priced_in,
                       'Volatility': '.'.join(name), 'Volatility_Type': 'FXVol',
                       'Discount_Rate': discount_name, 'Discount_Rate_Type': 'InterestRate',
                       'Yield': carry_name, 'Yield_Type': 'InterestRate',
                       'Quote_Type': 'Implied_Volatility',
                       # the surface's x-axis is log(F/K) on the pair, so the lookup is against the
                       # forward and inverts where the pair's own deals invert it
                       'Use_Forward': 'Yes', 'Invert_Moneyness': 'Yes' if invert else 'No',
                       # the step clock is what the fitted parameters mean - a deal's `Steps_Per_Year`
                       # must be this number - so it is stated, at the field's own declared default
                       'Steps_Per_Year': declared['Steps_Per_Year'],
                       'Quote_Timestamp': price_factors[vol_name].get('Quote_Timestamp') or '',
                       'Quote_Source': source,
                       'European_Options': quotes}}

    @staticmethod
    def effective_yield(discount_rate, funding, carry, t):
        """The `q` the objective runs on at accrual `t`: the dividend (or foreign) carry, plus the
        basis between the curve the premium discounts on and the curve the forward grows at.

        The whole arithmetic hangs off `r` and `q`: the forward is `spot exp((r-q)t)`, the per-step
        carry `(r-q)t/n`, and the value carries `exp(-qt)` so it discounts at `r`. Folding the
        funding basis `r - f` into `q` grows the forward at `f - carry` - what
        `utils.calc_eq_forward` integrates - while the premium still discounts at `r`.

        With no `Funding_Rate` the basis term is not evaluated, so `q` is the plain carry. Every leg
        is read at the accrual the `Discount_Rate` curve's own day count gives.
        """
        q = 0.0 if carry is None else float(carry.current_value(t))
        return q if funding is None else q + (discount_rate - float(funding.current_value(t)))

    @classmethod
    def spot_priced_in(cls, spot, priced_in):
        """`spot` re-priced in the `Priced_In` FACTOR, which is the ratio of the two base-priced
        rates.

        An `FxRate` is its currency in the book's BASE, so `B` priced in `A` is `FxRate.B` over
        `FxRate.A`. `None` is the base itself, whose own rate is identically one, so the read is
        `spot / 1.0` and a pair with a base leg is bit-identical.
        """
        return spot / (1.0 if priced_in is None else float(priced_in.current_value()[0]))

    @classmethod
    def resolve_block(cls, market_price, instrument, price_factors, factor_interp, sys_params):
        """`({field: constructed factor}, spot)` - everything a block names, resolved before an
        option is looked at, so a book carrying two ladders fails on the one that is wrong.

        The spot is the `Underlying`'s rate priced in the block's `Priced_In` - one read, a
        denominator of exactly one where the block names none.

        THE BLOCK'S NAME AND ITS `Priced_In` ARE ONE FACT and must agree, because the factor is
        named off the NAME and the axis is fitted off the FIELD: a cross-keyed block declaring none
        would file the base-priced law under the key a cross deal reads. A `Priced_In` of more than
        one token refuses with them - a composed rate is a spot plus a basis tail, never a cross.

        A missing reference refuses by name (`resolve_references`), and so does a surface whose vol
        is not a table lookup at a strike: a mis-looked-up vol converges to the wrong answer.
        """
        priced_in = instrument.get('Priced_In') or ''
        declared = utils.check_rate_name(priced_in) if priced_in else ()
        tail = utils.check_rate_name(market_price)[2:]
        # an equity or commodity name tail is a basis chain, not a currency the underlying is
        # priced in, so only an FxRate underlying is held to the cross rule; a blank one, or one
        # the book cannot resolve, is refused by name below, so it is held to it too
        try:
            underlying = cls.resolve(instrument, 'Underlying', price_factors)
            fx = getattr(underlying, 'type', 'FxRate') == 'FxRate'
        except StopIteration:
            fx = True
        if fx and (len(declared) > 1 or declared != tail):
            raise ValueError(
                '{0}: the block is filed under a name saying its underlying is priced in {1} and '
                'declares Priced_In {2!r}. The name and the field are one fact - the factor is '
                'named off the name and the axis is fitted off the field - and a currency is ONE '
                'token. File a cross as {3}.<underlying>.<priced in> declaring that same token, '
                'or drop both for a rate priced in the base'.format(
                    market_price, '.'.join(tail) or 'the base', priced_in,
                    cls.market_factor_type))
        factors = cls.resolve_references(market_price, instrument, price_factors, factor_interp)
        surface = factors.get('Volatility')
        if surface is not None:
            surface.delta = sys_params.get('Volatility_Delta', 0.0)
        if instrument['Quote_Type'] == 'Implied_Volatility':
            subtype = surface.get_subtype()
            if subtype[0] not in cls.tabular_surfaces:
                raise ValueError(
                    '{0}: volatility {1} has Surface_Type {2} (Moneyness_Rule {3}); only {4} '
                    'surfaces carry a vol AT A STRIKE, which is what Quote_Type '
                    'Implied_Volatility prices its target premium at. Quote the premiums directly '
                    '(Quote_Type Premium) instead'.format(
                        market_price, instrument['Volatility'], subtype[0], subtype[1],
                        '/'.join(cls.tabular_surfaces)))
        return factors, cls.spot_priced_in(
            float(factors['Underlying'].current_value()[0]), factors.get('Priced_In'))

    def prepare_quotes(self, sys_params, instrument, factors, spot, market_price=None):
        """Each `European_Options` row as `(option, t, r, q, forward, sign, strike, sigma,
        premium)` - the numbers every family in this hierarchy builds its objective from.

        `t` is the `Discount_Rate` curve's own day count accrual, the forward grows at
        `effective_yield`'s `r - q`, and a blank `Strike` is the forward. Under
        `Implied_Volatility` the premium is Black at the surface's vol AT THE STRIKE; under
        `Premium` the quote IS the premium and `sigma` is inverted back out of it, which is what
        seeds a fit and what its diagnostics are read in.
        """
        discount, surface = factors['Discount_Rate'], factors.get('Volatility')
        carry, funding = factors.get('Yield'), factors.get('Funding_Rate')
        quote_type = instrument['Quote_Type']
        use_forward = instrument.get('Use_Forward') == 'Yes'
        invert_moneyness = instrument.get('Invert_Moneyness') == 'Yes'
        for option in quote_table(instrument, market_price, 'European_Options'):
            t = discount.get_day_count_accrual(
                sys_params['Base_Date'], (option['Expiry_Date'] - sys_params['Base_Date']).days)
            r = float(discount.current_value(t))
            q = self.effective_yield(r, funding, carry, t)
            forward = spot * np.exp((r - q) * t)
            sign = 1.0 if option['Option_Type'] == 'Call' else -1.0
            strike = forward if not option['Strike'] else option['Strike']
            if quote_type == 'Implied_Volatility':
                moneyness = self.moneyness(
                    strike, spot, forward, surface, use_forward, invert_moneyness)
                sigma = surface.current_value([[moneyness, t]])[0] if not option[
                    'Quoted_Market_Value'] else option['Quoted_Market_Value']
                sigma += surface.delta
                premium = utils.black_european_option_price(
                    forward, strike, r, sigma, t, option['Units'], sign)
            else:
                premium = option['Units'] * option['Quoted_Market_Value']
                # back out the Black vol of the quote (seeds the fit and the diagnostics)
                call = option['Quoted_Market_Value'] + (0.0 if sign > 0 else
                                                        forward - strike) * np.exp(-r * t)
                sigma = np.sqrt(utils.bs_implied_total_var(
                    call, spot * np.exp(-q * t), strike, r * t, 1) / t)
            yield option, t, r, q, forward, sign, strike, sigma, premium

    @staticmethod
    def quote_trailer(instrument):
        """Where the quotes came from, in the record beside the parameters they produced."""
        if instrument.get('Quote_Source') or instrument.get('Quote_Timestamp'):
            logging.info('  quotes: {} (as at {})'.format(
                instrument.get('Quote_Source') or 'authored by hand',
                instrument.get('Quote_Timestamp') or 'no stated time'))


#: One quote inside the fit: its block index `j`, the contract (`ratio` being the strike over spot,
#: built once because the price reads it every evaluation), and the two market numbers the
#: vol-space residual is built from - the target premium and the Black vega at the quoted vol.
LVQuote = namedtuple(
    'quote',
    'row j spot strike ratio is_call units T rate carry forward premium vega weight '
    'sigma quoted')

#: What a vanilla is priced off, whichever estimator made it: the conditional drift and variance
#: per row per block, the row weights per block (`None` where the rows are equally weighted paths)
#: and the mixer's own clock, which the residual's shape is read off.
LVPriced = namedtuple('priced', 'M var weight mixer')

#: One forward-start target: a window `[j1, j2]` of the same walk, a strike as a fraction of
#: S_T1, and the vol it is aimed at - the objective is in vol points, so this block carries
#: no premium, and a REPORTED row (the reserve line's) carries no target vol either.
LVForward = namedtuple('forward', 'j1 j2 T1 tenor strike ratio carry weight target')


class LVPriors:
    """The prior rows one LogVar2FJ calibration carries, read off its completed block, a history and
    the class tables: each soft row's target, scale and report line, the cold seed and the slow
    pin. What only the fit knows is passed in."""

    #: The lever the leverage prior's second row is on: the product of two bucket curves, which is
    #: what a smile carries.
    LEVERAGE = 'Rho_S*Sigma_S'

    #: The lever the residual's skew prior is on: the share the smile sees, a row on `Beta` alone
    #: being obeyed for free by inflating `Alpha`.
    SHARE = 'Beta/Alpha'

    #: The prior rows that are one row per fitted bucket rather than one number.
    BUCKET_ROWS = ('Rho_S', 'Alpha', LEVERAGE, SHARE)

    def __init__(self, market_price, read, history, asset_class, typed, box, priors, weights,
                 wings):
        self.market_price, self.read, self.history = market_price, read, history
        self.asset_class, self.box, self.priors, self.wings = asset_class, box, priors, wings
        self.slow_priors, self.leverage_priors = typed['slow'], typed['leverage']
        self.product_priors, self.sigma_reference = typed['product'], typed['reference']
        #: the residual's class prior per asset class with its spread: `Alpha`'s on `log alpha`,
        #: the skew's on the share
        self.class_priors = {'Alpha': (typed['alpha'], float(read['Alpha_Prior_Sd'])),
                             self.SHARE: typed['share']}
        self.leverage_weight = float(read['Leverage_Prior_Weight'])
        self.slow_horizon = float(read['Slow_Horizon'])
        self.residual_horizon = float(read['Residual_Horizon'])
        self.contamination = float(read['Contamination_Ratio'])
        #: what one quote missing by one vol point costs the residual: the price of one prior SE
        self.quote_point = 0.01 * float(np.sqrt(np.mean([weight ** 2 for weight in weights])))
        #: the leverage prior in force: its two targets, their standard errors and their source
        self.rho, self.rho_sd, self.product, self.product_sd, self.source = (
            self.leverage_prior() if priors else (
                0.0, None, 0.0, None,
                'none - Model_Priors is Off, the vanilla-only objective and its box'))

    def declared(self, name):
        """One optional numeric block field as a float, or `None` where it is blank."""
        text = str(self.read[name]).strip()
        return float(text) if text else None

    def leverage_prior(self):
        """`(rho_s, SE, product, SE, source)` on the axis the block fits, each read in one order:
        the block's declaration, else a history in `Price Models`, else the class default. A blank
        standard error takes the nominal weight."""
        rho, rho_sd = self.declared('Leverage_Prior'), self.declared('Leverage_Prior_SE')
        product, product_sd = (self.declared('Leverage_Product_Prior'),
                               self.declared('Leverage_Product_Prior_SE'))
        for name, error in (('Leverage_Prior_SE', rho_sd),
                            ('Leverage_Product_Prior_SE', product_sd)):
            if error is not None and not error > 0.0:
                raise ValueError(
                    '{}: {} reads {:g}. It is the standard error the prior row is weighted by - '
                    'one quote-vol-point per SE - so a non-positive one divides the row by '
                    'nothing. Write the error the estimate carries, or leave it blank for the '
                    'nominal weight'.format(self.market_price, name, error))
        source = ['the declared Leverage_Prior', 'the declared Leverage_Product_Prior']
        if product is None and rho is not None:
            product, source[1] = rho * self.sigma_reference, (
                'the declared Leverage_Prior at Sigma_S_Reference {:g}'.format(
                    self.sigma_reference))
        if self.history is not None and (rho is None or product is None):
            hist, sigma = (float(self.history[name]) for name in ('Rho_S', 'Sigma_S'))
            if rho is None:
                rho, rho_sd, source[0] = hist, float(self.history['Rho_S_SE']), (
                    "the history's own Rho_S")
            if product is None:
                product, source[1] = hist * sigma, (
                    "the history's Rho_S x Sigma_S {:.4f}, by the delta method off their own "
                    'standard errors'.format(sigma))
                product_sd = float(np.hypot(sigma * float(self.history['Rho_S_SE']),
                                            hist * float(self.history['Sigma_S_SE'])))
        default = 'the {} class default'.format(self.asset_class)
        if rho is None:
            rho, source[0] = self.leverage_priors[self.asset_class][0], default
        if product is None:
            product, source[1] = self.product_priors[self.asset_class][0], default
        return rho, rho_sd, product, product_sd, '; '.join(
            '{} {:+.4f}{} from {}'.format(
                name, value, '' if error is None else ' +- {:.4f}'.format(error), why)
            for name, value, error, why in (
                ('rho_s', rho, rho_sd, source[0]),
                ('the product rho_s*sigma_s', product, product_sd, source[1])))

    def alpha_prior(self):
        """The history's `alpha^P`, or `None`: it seeds the fit and is reported beside `alpha^Q`."""
        return None if self.history is None else float(self.history['Alpha'])

    def contaminated(self):
        """Why the history's `alpha^P` may not be believed - its clock share `C_Eff` past
        `Contamination_Ratio` times its `c`, or no `C_Eff` to read - or `''`."""
        missing = [x for x in ('C_Eff', 'C') if x not in self.history]
        if missing:
            return 'the block carries no {} to read its clock share by'.format('/'.join(missing))
        c_eff, c = float(self.history['C_Eff']), float(self.history['C'])
        return '' if c_eff <= self.contamination * c else (
            "its clock share C_Eff {:.4f} is {:.1f}x the model's c {:.4f}, past the {:g}x that "
            'reads as leverage rather than residual'.format(c_eff, c_eff / c, c,
                                                            self.contamination))

    def alpha_seed(self):
        """The cold `Alpha`: the history's `alpha^P` where its clock share is clean, else the class
        prior."""
        alpha = self.alpha_prior()
        return (self.class_priors['Alpha'][0][self.asset_class][0]
                if alpha is None or self.contaminated() else alpha)

    def cold(self):
        """The levers the priors seed a cold fit at, per bucket: `Rho_S` signed by the leverage
        prior, `Alpha` at `alpha_seed` and `Beta` at the class skew share times it."""
        alpha = self.alpha_seed()
        return {'Rho_S': float(np.copysign(0.75, self.rho or -1.0)),
                'Beta': self.class_priors[self.SHARE][0][self.asset_class][0] * alpha,
                'Alpha': alpha}

    def prior_box(self, name, rho_s):
        """The box a prior's target is clipped into, in the model's own numbers, given the fit's
        `Rho_S` box `rho_s`: the product's edge is that times the top of `Sigma_S_Bounds`, and
        `Alpha`'s is the unconstrained box mapped through the fit's transform."""
        if name == self.SHARE:
            edge = np.sqrt(1.0 - utils.LogVar2FJ.COND_MIN)
            return -edge, edge
        if name == self.LEVERAGE:
            edge = rho_s[1] * self.box['Sigma_S'][1]
            return -edge, edge
        if name == 'Rho_S':
            return rho_s
        if name != 'Alpha':
            return self.box[name]
        return tuple(0.5 + utils.LogVar2FJ.AB_EPS + np.logaddexp(0.0, np.array(self.box['Alpha'])))

    def history_estimate(self, name):
        """A history's own `(estimate, SE)` for a residual row in that row's units - `log alpha`, or
        the share `beta/alpha` with its delta-method error - or `(None, None)` without a history."""
        if self.history is None:
            return None, None
        alpha, alpha_sd = (float(self.history[key]) for key in ('Alpha', 'Alpha_SE'))
        if name == 'Alpha':
            return alpha, alpha_sd / abs(alpha)
        share = float(self.history['Beta']) / alpha
        return share, float(np.hypot(float(self.history['Beta_SE']),
                                     share * alpha_sd)) / abs(alpha)

    def residual_hierarchy(self):
        """`Alpha` and the skew share as rows `(lever, target, SE, in logs)` - the history's
        estimate where its SE is at or under the class spread and its `alpha^P` is clean, else the
        class default - with the report line naming the tier in force."""
        wings = any(T <= self.residual_horizon for T in self.wings)
        edge = 'its shortest wing expiry is {} against the {:g}y Residual_Horizon'.format(
            '{:g}y'.format(min(self.wings)) if self.wings else 'none at all',
            self.residual_horizon)
        rows, told = [], []
        for name, logs in (('Alpha', True), (self.SHARE, False)):
            classes, spread = self.class_priors[name]
            value, error = self.history_estimate(name)
            why = 'no history carries an estimate' if value is None else self.contaminated()
            if why and value is not None:
                why += ", so its {}^P {:.4f} is REPORTED AND NOT USED".format(name.lower(), value)
            elif not why and error > spread:
                why = 'its {}^P {:.4f} carries an SE of {:.4g} in the prior\'s own units, past ' \
                      'the {:g} class spread'.format(name.lower(), value, error, spread)
            rows.append((name, classes[self.asset_class][0], spread, logs) if why
                        else (name, value, error, logs))
            told.append('{}: {}'.format(name.lower(), (
                'CLASS PRIOR {:+.4f} at spread {:g}, history uninformative - {}'.format(
                    rows[-1][1], spread, why) if why else
                "the history's {}^P {:+.4f} with its SE {:.4g}".format(
                    name.lower(), value, error))))
        return rows, ['{} ({}) - {}'.format(
            'the short-dated wings reach the residual and outvote its soft rows where they mean '
            'it' if wings else 'the residual is not identified by this ladder at all, so its rows '
                               'are the whole of what states it', edge, '; '.join(told))]

    def soft_priors(self, rho_s, identified):
        """Every prior as a soft row `(lever, target, scale, in logs)`, one SE costing one quote vol
        point, its target clipped into `prior_box`, with the report lines and the residual targets a
        cold fit is seeded at; `identified` says whether the ladder fits the slow pair."""
        rows = [('Rho_S', self.rho, self.rho_sd, False, True),
                (self.LEVERAGE, self.product, self.product_sd, False, True)]
        if self.history is not None:
            rows += [(name, float(self.history[name]), float(self.history[name + '_SE']), False,
                      identified) for name in utils.LogVar2FJ.SLOW_HISTORY[1][:2]]
        hierarchy, lines = self.residual_hierarchy()
        rows += [row + (True,) for row in hierarchy]
        soft, seeds = [], []
        for name, target, error, logs, fitted in rows:
            low, high = self.prior_box(name, rho_s)
            inside = float(np.clip(target, low, high))
            if inside != target or not fitted:
                lines.append('the {} prior {:+.4f}{} {}'.format(
                    name, target, '' if error is None else ' +- {:.4f}'.format(error),
                    ('is OUTSIDE the ({:g}, {:g}) box this fit moves it in, so its row is CLIPPED '
                     'to {:+.4f}'.format(low, high, inside) if inside != target else
                     'is inside its box') + ('' if fitted else
                                             ' - and carries NO ROW, this ladder PINNING the '
                                             'slow pair rather than fitting it')))
            if fitted:
                soft.append((
                    name, inside, self.quote_point / error if error is not None else
                    self.leverage_weight / (self.sigma_reference if name == self.LEVERAGE else 1.0),
                    logs))
                if name in ('Alpha', self.SHARE):
                    seeds.append((name, inside))
        if not self.priors:
            lines.append('the prior rows are NOT IN FORCE, Model_Priors is Off')
        return soft, lines, seeds

    def fast_sign(self, rho_s):
        """Which way the fast leverage leans: the fitted `rho_s`, else the leverage prior's sign
        where the box left it at zero."""
        return rho_s or self.rho or -1.0

    def slow_prior(self, rho_s):
        """`(rho_l, sigma_l, source)` for the stage-4 pin: `Slow_Factor_Prior` where the block
        declares one, else the class default's magnitude signed by `fast_sign(rho_s)`."""
        rows = [float(x) for x in
                str(self.read['Slow_Factor_Prior']).split(',') if x.strip()]
        floor = self.box['Sigma_L'][0]
        if len(rows) == 2 and rows[1] >= floor:
            return rows[0], rows[1], 'the declared Slow_Factor_Prior'
        if rows:
            raise ValueError(
                "{}: Slow_Factor_Prior reads {!r}. It is the PAIR rho_l,sigma_l held where the "
                "ladder carries no wing at {:g}y or longer, with sigma_l AT OR ABOVE the {:g} "
                "floor stage 4 keeps beneath any prior - a two-year surface cannot claim "
                "five-year vol is certain, and a zero slow factor collapses a CVA profile's vol "
                "distribution onto the fast factor's spread, which reverts within months. Write "
                "both at or above the floor, or leave the field blank for the {} default".format(
                    self.market_price, self.read['Slow_Factor_Prior'], self.slow_horizon,
                    floor, self.asset_class))
        rho_l, sigma_l = self.slow_priors[self.asset_class]
        return float(np.copysign(rho_l, self.fast_sign(rho_s))), sigma_l, (
            'the {} class default, signed by the Rho_S {:+.4f} in force at the pin'.format(
                self.asset_class, rho_s))

    def pin_slow(self, rho_s):
        """`(rho_l, sigma_l, line)` for a ladder that does not identify the slow pair: the prior it
        is pinned at, given the fitted `rho_s`, and the line the report earns for it."""
        rho_l, sigma_l, source = self.slow_prior(rho_s)
        return rho_l, sigma_l, (
            'pinned: not identified by this ladder - Rho_L {:+.4f} and Sigma_L {:.4f} are held at '
            '{}, the longest wing expiry being {} against the {:g}y stage 4 asks '
            'for'.format(rho_l, sigma_l, source,
                         '{:g}y'.format(max(self.wings)) if self.wings else 'none at all',
                         self.slow_horizon))


class LVFit(utils.Residual):
    """One LogVar2FJ calibration: the prepared quotes, the walk they are priced on, the fitted state
    and every verb that moves it. A fitted vector is spliced into the state by `build`;
    `LogVar2FJModelParameters` prepares the quotes and runs the stages."""

    #: What the walk is handed per step: the two levers the state and the clock read.
    step_names = pricing.LogVar2FJKit.step_names

    #: The levers an expiry's wing quotes free, in order, in `Bootstrap` mode; the rest are TIED to
    #: the bucket before.
    FREE_ORDER = ('Beta', 'Sigma_S', 'Rho_S', 'Alpha')

    #: The residual's two levers, fitted in the unconstrained coordinates `utils.LogVar2FJ.ab` maps
    #: inside `|beta| < alpha`, `|beta + 1| < alpha`; the transform lives here and nowhere else.
    RAW = ('Alpha', 'Beta')

    #: The multiple of one quote row past which a prior row's coordinate counts as not identified.
    PRIOR_RATIO = 100.0

    def __init__(self, family, market_price, instrument, factors, previous):
        self.family, self.market_price = family, market_price
        self.prec, self.device = family.prec, family.device
        self.factors = factors
        utils.LogVar2FJ.retired('{} block'.format(market_price), instrument)
        #: the block completed by its own declarations, so every read is an index
        self.instrument = read = declared_defaults(type(family), instrument)
        self.mode = read['Fit_Mode']
        self.delta = 1.0 / float(read['Steps_Per_Year'])
        self.c_min, self.rcond = float(read['C_Min']), float(read['Jacobian_Rcond'])
        self.stationarity = float(read['Stationarity_Tol'])
        self.tolerance, self.max_iter = float(read['Tolerance']), int(read['Max_Iterations'])
        self.step_tolerance = float(read['Step_Tolerance'])
        self.pillar_tol, self.smoothness = float(read['Pillar_Tolerance']), float(
            read['Bucket_Smoothness'])
        self.is_vol = read['Quote_Type'] == 'Implied_Volatility'
        self.sampling = read['Sampling']
        #: the factor type `Underlying` resolves to, which every class table is keyed by
        self.asset_class = type(factors['Underlying']).__name__
        #: the block's typed hyperparameters, the class tables among them `LVPriors` reads
        self.typed = typed = utils.LogVar2FJ.typed(
            market_price, read, family.factor_types['Underlying'], self.asset_class)
        #: the vanilla estimator, and the quadrature's node counts where it is one
        self.quadrature = read['Vanilla_Pricer'] == 'Quadrature'
        self.nodes = typed['nodes']
        self.sampled = None
        if self.quadrature and utils.LogVar2FJ.declared(read.get('Cap_A')) is not None:
            raise ValueError(
                '{}: Vanilla_Pricer Quadrature prices the instantaneous variance as the EXPONENTIAL '
                'of a Gaussian, which is what makes the clock\'s moments closed form, and this '
                'block declares Cap_A {:g} - a corner the walk would take and the moments cannot. '
                'Omit Cap_A and the fit runs unbounded, writing the corner onto the factor as it '
                'does now, or price the vanillas on the Walk'.format(
                    market_price, utils.LogVar2FJ.declared(read['Cap_A'])))
        self.source = read['Forward_Smile_Source']
        # a fit that walks nothing - quadrature vanillas and no forward block - runs on the HOST,
        # where tensors this small dispatch faster than a card launches; the walk wants the card
        if self.quadrature and self.source == 'None':
            self.device = torch.device('cpu')
        if self.source == 'Prior':
            raise ValueError(
                '{}: Forward_Smile_Source Prior is WITHDRAWN. The block exists with a market or a '
                'reference source - Quotes or Reference, both off Forward_Smiles - or not at all. '
                'A desk\'s own VIEW of the forward smile is carried by the reserve line the block '
                'being off reports (Stickiness_Band here, Skew_Gradient on the factor, '
                'Skew_Reserve at the deal), because a view fitted as a target is paid for in '
                'vanilla fit and the cap that bounded that payment covered one stage while the '
                'damage occurred in another'.format(market_price))
        #: Off is the vanilla-only objective and box: no prior row and no soft shape term
        self.priors = read['Model_Priors'] == 'On'
        self.skew_band = float(read['Stickiness_Band'])
        self.shape_floor = float(read['Residual_Shape_Floor'])
        #: the fitted box per coordinate, `Alpha`/`Beta` in the unconstrained coordinates `RAW`
        #: names; `Rho_S`'s own is derived at every stage
        self.box = typed['box']
        self.stage_horizons = typed['horizons']
        self.slow_horizon = float(read['Slow_Horizon'])
        self.l_iterations = int(read['Xi_Solve_Iterations'])
        self.l_damping = float(read['Xi_Solve_Damping'])
        self.cap_headroom_max = float(read['Cap_Headroom_Max'])
        self.log_vol_sd_band = typed['band']
        self.atm_miss_max = float(read['Atm_Miss_Max'])
        self.prior_strikes = utils.LogVar2FJ.parse_floats(read['Prior_Strikes'],
                                                          'Prior_Strikes')
        self.psi_strikes = typed['strikes']
        self.c_margin = float(read['C_Margin'])
        self.soft_penalty = float(read['Soft_Penalty'])
        self.shape_penalty = float(read['Shape_Penalty'])
        self.spot_rung_tol = float(read['Spot_Rung_Tolerance'])
        self.previous, self.tables = previous, []
        #: what the fit spent: evaluations, Jacobians and pillar passes
        self.calls = {'n': 0, 'j': 0, 'l': 0}
        self.targets, self.guarded, self.ties = [], [], {}
        #: the target difference pair per forward tenor and the spot rung each tenor's smile is
        #: read at
        self.tilt, self.spot_rung, self.reported = {}, {}, []
        self.pinned_slow, self.identified_days, self.history = '', [], None
        #: every soft prior row in force as `(lever, target, scale, in logs)`, and their lines
        self.prior_rows, self.prior_lines = [], []
        self.notes, self.final, self.leaf, self.theta = [], {}, None, None
        #: the last stage's box, the gradient `J^T r` it stopped on and the stage that hit its cap
        self.edges, self.slope, self.capped = None, None, None

    def table(self, name):
        """One optional Table as rows, the completed block's declared blank `'null'` being none."""
        rows = self.instrument[name]
        return rows if isinstance(rows, list) else []

    def tensor(self, x):
        """A scalar leaf on the fit's own device and precision; a None stays None."""
        return None if x is None else torch.tensor(float(x), device=self.device, dtype=self.prec)

    def vector(self, xs):
        """Numbers as one tensor on the fit's own device and precision."""
        return torch.tensor([float(x) for x in xs], device=self.device, dtype=self.prec)

    def draw(self, paths, steps, blocks, seed):
        """The walk's two normals per step and the mixer's uniform per block, fixed for the whole
        fit and antithetic; `Sampling` picks the stream, generated on the host so a seed names one
        draw whatever device the fit runs on."""
        half = max(int(paths) // 2, 1)
        if self.sampling == 'Sobol':
            width, cap = 2 * int(steps) + int(blocks), torch.quasirandom.SobolEngine.MAXDIM
            if width > cap:
                raise ValueError(
                    '{}: Sampling Sobol wants {} dimensions - two per internal step plus one per '
                    'block - against the {} one scrambled engine carries. Declare Sampling Pseudo, '
                    'or coarsen Steps_Per_Year.'.format(
                        self.market_price, width, cap))
            engine = torch.quasirandom.SobolEngine(width, scramble=True, seed=int(seed))
            engine.fast_forward(utils.Calculation_State.QUASI_ANCHOR)
            draws = engine.draw(half, dtype=self.prec).clamp(1e-6, 1.0 - 1e-6)
            z_l, z_s = (utils.norm_icdf(draws[:, :steps]),
                        utils.norm_icdf(draws[:, steps:2 * steps]))
            u = draws[:, 2 * steps:]
        else:
            kw = {'generator': torch.Generator().manual_seed(int(seed)), 'dtype': self.prec}
            z_l, z_s = torch.randn(half, steps, **kw), torch.randn(half, steps, **kw)
            u = torch.rand(half, blocks, **kw)
        return tuple(torch.cat([x, y]).to(self.device)
                     for x, y in ((z_l, -z_l), (z_s, -z_s), (u, 1.0 - u)))

    def lstar(self, scalars, levers, levels):
        """The OU mean level at the walk's grid times, `log xi(t) - Var(l+s)(t)/2`, off the segment
        strip in log xi."""
        params = dict(scalars, Sigma_S=levers['Sigma_S'][self.step_bucket[:-1]])
        return self.l_at(levels) - 0.5 * utils.LogVar2FJ.state_variance(params, self.deltas)

    def walk(self, scalars, levers, curve, n):
        """`(M, Sigma^2)` cumulated to every grid point off one internal-step walk, so a maturity is
        a prefix and a forward window the difference of two; the strip's mixers are one draw."""
        params = dict(scalars, **{name: levers[name][self.step_bucket[:n]]
                                  for name in utils.LogVar2FJ.BUCKET_NAMES})
        eta_l, eta_s, uG = self.draws
        s = eta_l.new_zeros(eta_l.shape[0])
        l, M, clocks, starts, a = s + curve[0], [], [], [], 0
        for b in [int(x) for x in self.upto if x <= n]:
            m, clock, l, s = utils.LogVar2FJ.walk(
                dict(params, **{x: params[x][a:b] for x in self.step_names}),
                curve[a:b + 1], self.deltas[a:b], eta_l[:, a:b], eta_s[:, a:b], (l, s), False)
            M.append(m)
            clocks.append(clock)
            starts.append(a)
            a = b
        alpha = torch.stack([params['Alpha'][i] for i in starts])
        beta = torch.stack([params['Beta'][i] for i in starts])
        delta, mu, gamma = utils.LogVar2FJ.nig_budget(torch.stack(clocks, -1), alpha, beta)
        G = utils.LogVar2FJ.ig_quantile(uG[:, :len(starts)], delta / gamma, delta * delta)
        clock = G.cumsum(-1)
        return LVPriced((torch.stack(M, -1) + mu + beta * G).cumsum(-1), clock, None, clock)

    def kernel(self, scalars, levers):
        """The quadrature's parameter-only matrices for one sweep, or None where the walk prices the
        vanillas; every pillar pass of that sweep reads them."""
        return utils.LogVar2FJ.quad_kernel(
            dict(scalars, **{name: levers[name][self.step_bucket[:self.n]]
                             for name in utils.LogVar2FJ.BUCKET_NAMES}),
            self.deltas, self.times) if self.quadrature else None

    def priced(self, scalars, levers, curve, n, kernel=None):
        """What a vanilla to step `n` is priced off: `walk`'s cumulated sums, or the quadrature's
        nodes where the block declares them - one law, two estimators."""
        if not self.quadrature:
            return self.walk(scalars, levers, curve, n)
        return LVPriced(*utils.LogVar2FJ.quadrature(
            kernel if kernel is not None else self.kernel(scalars, levers),
            dict(scalars, **{name: levers[name][self.step_bucket[:n]]
                             for name in utils.LogVar2FJ.BUCKET_NAMES}),
            curve, [int(x) for x in self.upto if x <= n], self.nodes))

    def walked(self, priced):
        """The per-path blocks the forward-start rows read: the walk's own sums, whatever priced the
        vanillas."""
        return priced if self.sampled is None else self.sampled

    @staticmethod
    def mix(gain, weight):
        """One node set's price: the paths' own mean, or the quadrature's weights."""
        return gain.mean() if weight is None else (weight * gain).sum()

    @staticmethod
    def conditional_black(M, var, carry, strike, weight=None):
        """`(S_T/S - k)^+` per path or per node, `pricing.lognormal_fired_gain` at the block's own
        Gaussian law, priced off the estimator's own forward as a martingale control variate."""
        sigma = utils.sqrt_or_zero(var)
        total = M + 0.5 * var
        # a node the mixer's density leaves at an exact zero has no mass and no derivative either;
        # its log is floored rather than taken, `inf * 0` being a NaN on every leaf behind it
        drift = M + carry - torch.logsumexp(
            total if weight is None else total + torch.log(
                weight.clamp(min=torch.finfo(weight.dtype).tiny)), 0) + (
                    np.log(M.shape[0]) if weight is None else 0.0)
        return pricing.lognormal_fired_gain(
            1.0, drift, sigma, (torch.log(strike) - drift) / sigma, strike, True)

    def implied(self, price, forward, strike, T):
        """A conditional-Black price as a Black vol: `utils.implied_vol` off the tape, then one
        Newton splice at its own vega, so `dsigma/dtheta` is `dP/dtheta` over that vega."""
        sigma = utils.implied_vol(price.detach(), forward, strike, 0.0, 1, T, 1.0, 0.0)
        sd = sigma * np.sqrt(T)
        d1 = (np.log(forward / strike) + 0.5 * sd * sd) / sd
        vega = max(forward * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi), 1.0e-12)
        return sigma + (price - price.detach()) / (vega * np.sqrt(T))

    def smile(self, cum, j, carry, T, strikes):
        """The model's own smile at block `j`, as vols at `strikes` given as fractions of that
        block's forward."""
        cum, forward = self.walked(cum), np.exp(carry)
        return [self.implied(self.conditional_black(
            cum[0][..., j], cum[1][..., j], carry, self.tensor(k * forward)).mean(),
                             forward, k * forward, T) for k in strikes]

    def shape(self, vols):
        """`(atm, slope, butterfly)` of a `psi_strikes` smile in decimal vol: the 90-110 difference
        over its log-strike gap, and the 90/110 average less the ATM."""
        low, atm, high = vols
        return atm, (low - high) / np.log(self.psi_strikes[2] / self.psi_strikes[0]), \
               0.5 * (low + high) - atm

    def spot_shape(self, cum, tenor):
        """The model's own spot `(atm, slope, butterfly)` at the quoted maturity nearest `tenor`."""
        j, carry, T = self.spot_rung[tenor]
        return self.shape(self.smile(cum, j, carry, T, self.psi_strikes))

    def market_shape(self, quotes):
        """The market's own `(atm, slope, butterfly)` at one expiry, its quoted vols read in
        log-moneyness at `psi_strikes`, warning where the rung does not reach them."""
        rows = sorted((np.log(quote.strike / quote.forward), quote.sigma) for quote in quotes)
        wants = [np.log(k) for k in self.psi_strikes]
        if min(wants) < rows[0][0] or max(wants) > rows[-1][0]:
            logging.warning(
                '{}: the {:g}y rung is quoted over {:.0%}-{:.0%} of its own forward and the objective '
                'reads the spot slope and butterfly at {}, so an end outside that is the NEAREST '
                'quote rather than the strike asked for, and the target DIFFERENCE this tenor '
                'is measured against understates both. Quote the wings at that expiry'.format(
                    self.market_price, quotes[0].T, np.exp(rows[0][0]), np.exp(rows[-1][0]),
                    '/'.join('{:g}'.format(k) for k in self.psi_strikes)))
        return self.shape(np.interp(wants, *zip(*rows)))

    def target_vol(self, target, shape):
        """The vol one forward row is aimed at: the model's own spot smile at that tenor, tilted by
        the target differences in slope and butterfly, the ATM level left to the ladder."""
        d_skew, d_bfly = self.tilt[(target.T1, target.tenor)]
        atm, slope, bfly = shape[0], shape[1] + d_skew, shape[2] + d_bfly
        a, b = np.log(self.psi_strikes[0]), np.log(self.psi_strikes[2])
        q = -(bfly + 0.5 * slope * (a + b)) / (a * b)
        u = np.log(target.strike)
        return atm + (-slope - q * (a + b)) * u + q * u * u

    def value(self, cum, quote):
        """One quote's model premium: the conditional Black over the paths, the discount and the
        yield rescale, and a put by parity off the analytic forward."""
        weight = None if cum.weight is None else cum.weight[..., quote.j]
        gain = quote.spot * self.mix(self.conditional_black(
            cum[0][..., quote.j], cum[1][..., quote.j], quote.carry, quote.ratio, weight), weight)
        return quote.units * np.exp(-quote.rate * quote.T) * (
            gain if quote.is_call else gain - (quote.forward - quote.strike))

    def forward_vol(self, cum, target):
        """One forward-start row's model vol, `forward_value` inverted by `implied`."""
        return self.implied(self.forward_value(self.walked(cum), target), np.exp(target.carry),
                            target.strike, target.tenor)

    def forward_value(self, cum, target):
        """One forward-start target's model premium in units of `S_T1`: the conditional Black over
        `[T1, T2]` alone, averaged under the share measure where the source is traded `Quotes`."""
        gain = self.conditional_black(
            cum[0][..., target.j2] - cum[0][..., target.j1],
            cum[1][..., target.j2] - cum[1][..., target.j1], target.carry, target.ratio)
        if self.source != 'Quotes':
            return gain.mean()
        share = torch.softmax(cum[0][..., target.j1] + 0.5 * cum[1][..., target.j1], 0)
        return (share * gain).sum()

    def market(self, quotes):
        """Every quote's market premium with its quoted number on the tape as a splice worth zero
        forward, so `dr/dq` rides a premium that cannot move a fitted digit."""
        premia = self.vector([quote.premium for quote in quotes])
        if self.leaf is None:
            return premia
        rows = [self.at_quote(quote, self.leaf[quote.row]) for quote in quotes]
        spliced = torch.stack(rows)
        return premia + (spliced - spliced.detach())

    def at_quote(self, quote, quoted):
        """One quote's premium as a function of the number quoted: discounted Black at a vol, the
        units times it at a premium."""
        if not self.is_vol:
            return quote.units * quoted
        return quote.units * np.exp(-quote.rate * quote.T) * utils.black_european_option(
            quoted.new_tensor(quote.forward), float(quote.strike), quoted, float(quote.T), 1.0,
            1.0 if quote.is_call else -1.0, None)

    def quote_vol(self, cum, quote):
        """The model-minus-market difference in Black vol at one quote, both premia inverted off the
        tape from the same forward, as a decimal."""
        both = [utils.implied_vol(
            premium, quote.spot * np.exp(quote.carry - quote.rate * quote.T), quote.strike,
                     quote.rate * quote.T, 1, quote.T, quote.units,
            0.0 if quote.is_call else (quote.forward - quote.strike) * np.exp(
                -quote.rate * quote.T))
            for premium in (float(self.value(cum, quote).detach()), quote.premium)]
        return both[0] - both[1]

    def cap_headroom(self, scalars, levers, curve):
        """The mass of path-days at or above the corner the factor will carry, read once at the
        written parameters; zero where the walk is unbounded."""
        eta_l, eta_s, _ = self.draws
        a = self.cap_level()
        if a is None:
            return 0.0
        sigma_s = levers['Sigma_S'][self.step_bucket[:-1]]
        w_s = utils.LogVar2FJ.ou_step_weights(scalars['Kappa_S'], sigma_s, self.deltas)[1]
        w_l = utils.LogVar2FJ.ou_step_weights(scalars['Kappa_L'], scalars['Sigma_L'], self.deltas)[1]
        zero = eta_l.new_zeros(eta_l.shape[0])
        state = (utils.LogVar2FJ.ou_path(scalars['Kappa_S'], w_s, eta_s, zero, self.deltas)
                 + utils.LogVar2FJ.ou_path(scalars['Kappa_L'], w_l, eta_l, zero, self.deltas))
        near = ((curve + state)[:, :-1] >= a).sum()
        return float(near) / (eta_l.shape[0] * self.deltas.shape[0])

    def l_knots(self, levels):
        """The xi curve as written, in log: one level per segment on the knots starting it and flat
        beyond, an `Event_Days` date carrying its own one-day knot pair."""
        pillars = torch.stack(list(levels))
        knots = self.knots[:pillars.numel()]
        if not self.event_times.size:
            return knots, pillars
        extra = utils.TermStructure(knots, pillars).at(self.vector(self.event_times)) + self.event_log
        merged = np.concatenate([knots, self.event_times])
        order = np.argsort(merged, kind='stable')
        return merged[order], torch.cat([pillars, extra])[order.tolist()]

    def l_at(self, levels):
        """`log xi` at the walk's grid times."""
        return utils.TermStructure(*self.l_knots(levels)).at(self.times)

    def solve_l(self, scalars, levers, kernel=None):
        """The inner triangular bootstrap, run at every outer iterate: each xi segment a damped
        Newton against its own ATM premium, warm started off the last sweep, the level returned as
        one Newton step at the root so `dxi/dtheta` and `dxi/dq` ride the tape."""
        targets = self.market(self.atm)
        kernel = self.kernel(scalars, levers) if kernel is None else kernel
        levels, misses = [], []
        for k, quote in enumerate(self.atm):

            def repriced(at):
                self.calls['l'] += 1
                return self.value(self.priced(scalars, levers,
                                              self.lstar(scalars, levers, levels + [at]),
                                              int(self.upto[quote.j]), kernel), quote) - targets[k]

            at = self.warm[k]
            for _ in range(self.l_iterations):
                pillar = at.detach().requires_grad_(True)
                shift = repriced(pillar)
                slope = torch.autograd.grad(shift, pillar, retain_graph=True)[0].detach()
                miss = float(shift.detach()) / quote.premium
                if abs(miss) < self.pillar_tol:
                    break
                at = pillar.detach() - (shift.detach() / slope).clamp(
                    -self.l_damping, self.l_damping)
            levels.append(pillar.detach() - shift / slope)
            self.warm[k] = pillar.detach()
            misses.append(miss)
        self.atm_misses = misses
        return levels

    def build(self, x, coords):
        """The full parameter set as tensors: the fitted coordinates off `x`, the rest off the
        state, `Alpha`/`Beta` mapped out of their unconstrained coordinates, then the `Bootstrap`
        ties."""
        scalars = {name: self.tensor(value) for name, value in self.state.items()
                   if name not in utils.LogVar2FJ.BUCKET_NAMES}
        levers = {name: [self.tensor(v) for v in self.state[name]]
                  for name in utils.LogVar2FJ.BUCKET_NAMES}
        raw = {}
        for i, (name, bucket) in enumerate(coords):
            if bucket is None:
                scalars[name] = x[i]
            elif name in self.RAW:
                raw[(name, bucket)] = x[i]
            else:
                levers[name][bucket] = x[i]
        for bucket in sorted({b for _, b in raw}):
            alpha, beta = utils.LogVar2FJ.ab(*[raw.get((name, bucket), self.tensor(
                self.raw_of(name, bucket))) for name in self.RAW])
            levers['Alpha'][bucket], levers['Beta'][bucket] = alpha, beta
        for (name, bucket), _ in sorted(self.ties.items(), key=lambda item: item[0][1]):
            levers[name][bucket] = levers[name][bucket - 1]
        return scalars, {name: torch.stack(v) for name, v in levers.items()}

    def raw_of(self, name, bucket):
        """One bucket's `Alpha` or `Beta` in the coordinate the fit moves it in."""
        return utils.LogVar2FJ.ab_inv(self.state['Alpha'][bucket],
                                      self.state['Beta'][bucket])[self.RAW.index(name)]

    def evaluate(self, x=None, coords=()):
        """One outer iterate: the parameters, the xi strip re-bootstrapped at them and the price
        state; the strip is banked, and so is the walk where quadrature fits forward rows."""
        scalars, levers = self.build(x, coords)
        kernel = self.kernel(scalars, levers)
        self.levels = self.solve_l(scalars, levers, kernel)
        curve = self.lstar(scalars, levers, self.levels)
        self.sampled = (self.walk(scalars, levers, curve, self.n)
                        if self.quadrature and (self.targets or self.reported) else None)
        return scalars, levers, self.priced(scalars, levers, curve, self.n, kernel)

    def rows(self, cum, judged, forwards):
        """Every row's residual, weighted: a vanilla's premium miss over its own market vega, and a
        forward-start's vol miss."""
        market, shapes = self.market(judged), self.spot_shapes(cum, forwards)
        return ([quote.weight * (self.value(cum, quote) - market[i]) / quote.vega
                 for i, quote in enumerate(judged)] +
                [target.weight * (self.forward_vol(cum, target)
                                  - self.target_vol(target, shapes[target.tenor]))
                 for target in forwards])

    def residual_shape(self, levers, cum, j=None, bucket=0):
        """The residual's shape `alpha*delta_A` to block `j`, the shortest calibrated expiry by
        default: 1 strongly non-Gaussian, 15 nearly Gaussian."""
        alpha, beta = levers['Alpha'][bucket], levers['Beta'][bucket]
        j = self.atm[0].j if j is None else j
        return cum.mixer[..., j].mean() * alpha * torch.sqrt(alpha * alpha - beta * beta)

    def residual(self, x, coords, judged, forwards=(), smooth=None):
        """The stage's residual vector: its own rows first, then the prior rows, the shape floor,
        the two share penalties and, to bucket `smooth`, the levers' smoothness."""
        scalars, levers, cum = self.evaluate(x, coords)
        terms = self.rows(cum, judged, forwards)
        # each prior is a SOFT ROW scaled so one standard error of miss costs what one quote
        # missing by one vol point costs, in LOGS where the spread is log-normal - never a pin
        for name, target, scale, logs in (self.prior_rows if self.priors else ()):
            value = (levers['Rho_S'] * levers['Sigma_S'] if name == LVPriors.LEVERAGE else
                     levers['Beta'] / levers['Alpha'] if name == LVPriors.SHARE else
                     levers[name] if name in utils.LogVar2FJ.BUCKET_NAMES else scalars[name])
            terms.append(scale * ((torch.log(value) - np.log(target)) if logs
                                  else (value - target)).reshape(-1))
        if self.priors and self.shape_floor > 0.0:
            terms.append(self.shape_penalty * torch.relu(
                1.0 - self.residual_shape(levers, cum) / self.shape_floor).reshape(1))
        terms.append(self.soft_penalty * torch.relu(
            self.c_min + self.c_margin
            - (1.0 - levers['Rho_S'] ** 2 - scalars['Rho_L'] ** 2)))
        terms.append(self.soft_penalty * torch.relu(
            utils.LogVar2FJ.COND_MIN + self.c_margin
            - (1.0 - (levers['Beta'] / levers['Alpha']) ** 2)))
        if smooth:
            terms += [self.smoothness * levers[name][:smooth + 1].diff()
                      for name in utils.LogVar2FJ.BUCKET_NAMES]
        return torch.cat([term.reshape(-1) for term in terms])

    def jacobian(self, x, coords, judged, forwards, **kw):
        """`dr/dx` at a fitted leaf, by one vmapped backward over the residual rows."""
        self.calls['j'] += 1
        leaf = torch.tensor(np.asarray(x, dtype=float), device=self.device,
                            dtype=self.prec, requires_grad=True)
        terms = self.residual(leaf, coords, judged, forwards, **kw)
        return utils.vmapped_jacobian(terms, leaf).cpu().numpy()

    def label(self, coord):
        """One coordinate's name, a bucket lever's carrying its bucket's start tenor."""
        return coord[0] if coord[1] is None else '{}[{:g}y]'.format(
            coord[0], self.buckets[coord[1]])

    def value_of(self, coord):
        """One coordinate's value in the space the fit moves it in."""
        if coord[0] in self.RAW:
            return self.raw_of(*coord)
        return self.state[coord[0]] if coord[1] is None else self.state[coord[0]][coord[1]]

    def bounds_of(self, coord, coords):
        """One coordinate's box; `Rho_S`'s is `C_Min`'s off the state's `Rho_L`, the widest where
        the stage moves `Rho_L` too, symmetric with a prior and one-sided under `Model_Priors`
        Off."""
        if coord[0] != 'Rho_S':
            return self.box[coord[0]]
        rho_l = 0.0 if ('Rho_L', None) in coords else self.state['Rho_L']
        edge = np.sqrt(max(1.0 - rho_l * rho_l - self.c_min, 0.0))
        return (-edge, edge) if self.priors else (-edge, 0.0)

    def stage(self, tag, coords, judged, forwards=(), **kw):
        """One stage: `least_squares` over `coords` against its own rows, the fitted values written
        back, its identification table and box kept, the strip settled as the next warm start."""
        edges = [self.bounds_of(coord, coords) for coord in coords]
        x0 = np.clip([self.value_of(coord) for coord in coords], *map(np.array, zip(*edges)))

        def residual(x):
            self.calls['n'] += 1
            return self.residual(
                self.vector(x), coords, judged, forwards, **kw).detach().cpu().numpy()

        started = time.time()
        result = scipy.optimize.least_squares(
            residual, x0, bounds=tuple(zip(*edges)), method='trf', x_scale='jac',
            jac=lambda x, *rest: self.jacobian(x, coords, judged, forwards, **kw),
            ftol=self.tolerance, xtol=self.step_tolerance, max_nfev=self.max_iter)
        # written back through `build`, so the residual pair leaves its own coordinates and the
        # ties of `Bootstrap` mode land in the state the report prints
        scalars, levers = self.build(self.vector(result.x), coords)
        for coord in coords:
            if coord[1] is None:
                self.state[coord[0]] = float(scalars[coord[0]])
        for name in utils.LogVar2FJ.BUCKET_NAMES:
            self.state[name] = [float(v) for v in levers[name]]
        # the DATA rows and, right behind them in `residual`'s own order, the prior rows
        rows = len(judged) + len(forwards)
        priors = sum(self.buckets.size if name in LVPriors.BUCKET_ROWS else 1
                     for name, *_ in self.prior_rows) if self.priors else 0
        self.tables.append((tag, [self.label(coord) for coord in coords], result.jac[:rows],
                            result.jac[rows:rows + priors]))
        self.edges, self.slope = np.array(edges, dtype=float).T, result.grad
        self.capped = tag if result.nfev >= self.max_iter else None
        logging.info('  stage {}: {} rows, {} evaluations, residual {:.4e}{}, {:.1f}s'.format(
            tag, result.fun.size, result.nfev, float(np.sqrt((result.fun ** 2).sum())),
            '' if result.nfev < self.max_iter else ' CAPPED at Max_Iterations',
            time.time() - started))
        self.settle(coords)
        self.fitted, self.judged, self.final = coords, judged, kw

    def settle(self, coords=()):
        """The xi strip at what the stage landed on, kept as the next stage's warm start."""
        scalars, levers = self.build(None, ())
        self.levels = [x.detach() for x in self.solve_l(scalars, levers)]
        self.warm = list(self.levels)

    def solve(self):
        """The staged fit; returns theta* over the last stage's coordinates, the flat vector
        `utils.LeastSquaresSolve` hangs the quote derivative on."""
        started = time.time()
        if self.quadrature:
            logging.info('  {}: the vanilla rows price by QUADRATURE, {} clock nodes x {} mixer '
                         'nodes and no draws'.format(self.market_price, *self.nodes))
        self.soft_priors()
        self.settle()
        (self.bootstrap_stages if self.mode == 'Bootstrap' else self.global_stages)()
        self.elapsed = time.time() - started
        self.theta = self.vector([self.value_of(coord) for coord in self.fitted])
        return self.theta

    def __call__(self, x):
        """The last stage's residual at fitted coordinates `x`, the rows the solver stepped on."""
        return self.residual(x, self.fitted, self.judged, self.targets, **self.final)

    @property
    def labels(self):
        """One name per fitted coordinate, so a coordinate the box holds is named."""
        return [self.label(coord) for coord in self.fitted]

    @property
    def descriptors(self):
        """One name per quote, in `leaf`'s own order."""
        return ['{:g}y {:g}'.format(quote.T, quote.strike) for quote in self.quotes]

    def cap_level(self):
        """The corner the written factor carries: `L(0)` plus twelve stationary log-vol sds at the
        widest bucket where none is declared, a declared level raised to that, None where null."""
        level = self.levels[0]
        rule = float(level.detach() if torch.is_tensor(level) else level) + 12.0 * max(self.spreads())
        if 'Cap_A' not in self.instrument:
            return rule
        a = utils.LogVar2FJ.declared(self.instrument['Cap_A'])
        return a if a is None else max(a, rule)

    def spreads(self):
        """The stationary log-vol sd per bucket, half the log-variance one."""
        return [0.5 * np.sqrt(sigma_s ** 2 / (2.0 * self.state['Kappa_S'])
                              + self.state['Sigma_L'] ** 2 / (2.0 * self.state['Kappa_L']))
                for sigma_s in self.state['Sigma_S']]

    @property
    def polish_only(self):
        """Is this fit a warm start off a previous factor, run as the joint polish alone?
        `Bootstrap` has no joint stage and stays cold."""
        return self.previous is not None and self.mode != 'Bootstrap'

    def global_stages(self):
        """The `Global` order: the residual off the short end, the fast pair with the forward rows,
        the slow pair or its pin, the forward block's later buckets, then a joint polish."""
        if not self.polish_only:
            short = [q for q in self.quotes if q.T <= self.stage_horizons[0]]
            middle = [q for q in self.quotes if q.T <= self.stage_horizons[1]]
            self.stage('2 (alpha, beta)', [('Alpha', 0), ('Beta', 0)],
                       short or middle or self.quotes)
            # the forward rows enter at stage 3: the split between the residual's skew and
            # `rho_s sigma_s` is what a forward smile sees and a spot smile does not
            self.stage('3 (rho_s, sigma_s)', [('Rho_S', 0), ('Sigma_S', 0)],
                       middle or self.quotes, self.targets)
        if not self.identified_slow():
            self.pin_slow()
        elif not self.polish_only:
            self.stage('4 (rho_l, sigma_l)', [('Rho_L', None), ('Sigma_L', None)],
                       [q for q in self.quotes if q.T > self.stage_horizons[1]])

        last = self.buckets.size - 1
        if self.targets and last and not self.polish_only:
            self.guarded = self.guard_rungs()
            later = list(range(1, self.buckets.size))
            for tag, name in (('5a beta(t)', 'Beta'), ('5b rho_s(t)', 'Rho_S')):
                self.stage(tag, [(name, b) for b in later], self.guarded, self.targets,
                           smooth=last)

        polish = [(name, None) for name in
                  (('Sigma_L', 'Rho_L') if self.identified_slow() else ())]
        polish += [(name, b) for name in utils.LogVar2FJ.BUCKET_NAMES
                   for b in range(self.buckets.size)]
        self.stage('6 joint polish', polish, self.quotes, self.targets, smooth=last)

    def guard_rungs(self):
        """The quotes stage 5 fits against: the rung nearest each forward target's `T1` and
        `T1 + Delta`, within `Spot_Rung_Tolerance` of Delta, refusing where there is none."""
        rungs = sorted({quote.T for quote in self.quotes})
        wanted, maturities = set(), set()
        for target in self.targets:
            for T in (target.T1, target.T1 + target.tenor):
                maturities.add(T)
                near = min(rungs, key=lambda x: abs(x - T))
                if abs(near - T) <= self.spot_rung_tol * target.tenor:
                    wanted.add(near)
        if not wanted:
            raise ValueError(
                '{}: stage 5 fits the later buckets to the forward rows and REPORTS what that '
                'cost the vanillas at those rows\' own maturities, and this ladder quotes {} - no '
                'expiry within {:.0%} of Delta of any of {} - so it would read no quotes at all. '
                'Quote a rung at those maturities, drop the later Param_Buckets, or set '
                'Forward_Smile_Source to None, where one bucket is the model'.format(
                    self.market_price, '/'.join('{:g}y'.format(T) for T in rungs),
                    self.spot_rung_tol,
                    '/'.join('{:g}y'.format(T) for T in sorted(maturities))))
        return [quote for quote in self.quotes if quote.T in wanted]

    def identified_slow(self):
        """Does this ladder carry a wing at `Slow_Horizon` or longer, which is what identifies the
        slow pair?"""
        return any(T >= self.slow_horizon for T in self.wings)

    def soft_priors(self):
        """Puts the prior rows `LVPriors` answers in force, a cold fit's residual pair seeded at its
        rows' targets."""
        self.prior_rows, self.prior_lines, seeds = self.prior.soft_priors(
            self.bounds_of(('Rho_S', 0), ()), self.identified_slow())
        if self.priors and not self.polish_only:
            for name, inside in seeds:
                if name == 'Alpha':
                    self.state['Alpha'] = [inside] * len(self.state['Alpha'])
                else:
                    self.state['Beta'] = [inside * alpha for alpha in self.state['Alpha']]

    def pin_slow(self):
        """Stage 4 where the ladder does not identify the slow pair: the prior's pin, into the
        state."""
        self.state['Rho_L'], self.state['Sigma_L'], self.pinned_slow = self.prior.pin_slow(
            self.state['Rho_S'][0])

    def bootstrap_stages(self):
        """`Bootstrap` mode: the slow pair or its pin, then bucket `k` fitted to expiry `k`'s wings
        given the buckets before it, freeing as many of `FREE_ORDER` as it has wing quotes."""
        long = [q for q in self.quotes if q.T > self.stage_horizons[1]]
        if self.identified_slow():
            self.stage('4 (rho_l, sigma_l)', [('Rho_L', None), ('Sigma_L', None)], long)
        else:
            self.pin_slow()
        for k, expiry in enumerate(self.wings):
            rung = self.wings[expiry]
            free = self.FREE_ORDER[:min(len(rung), len(self.FREE_ORDER))]
            self.ties.update({(name, k): 'carry'
                              for name in self.FREE_ORDER[len(free):] if k})
            self.free[k] = free
            self.stage('bootstrap bucket {:g}y ({} wing quote{}, free {})'.format(
                expiry, len(rung), '' if len(rung) == 1 else 's', '/'.join(free)),
                [(name, k) for name in free], rung, smooth=k)

    def own_segment(self, t):
        """Is the ATM segment holding `t` one internal step, the event day its own pillar?"""
        k = int(np.searchsorted(self.knots, t + utils.BUCKET_TOL)) - 1
        return 0 <= k < self.knots.size - 1 and round(
            (self.knots[k + 1] - self.knots[k]) / self.delta) <= 1

    def finish(self, theta):
        """The fit at theta*: the polish's identification table again without the forward rows where
        a source is set, the reserve line's rows, the walk at theta and every quote's vol miss."""
        if self.targets and self.tables:
            self.tables.append((self.tables[-1][0] + ', vanillas only', self.labels,
                                self.jacobian([self.value_of(x) for x in self.fitted], self.fitted,
                                              self.quotes, (), **self.final)[:len(self.quotes)],
                                []))
        self.skew_rows, self.skew_declared = self.skew_gradient(), False
        self.scalars, self.levers, self.cum = self.evaluate(theta, self.fitted)
        self.misses = [100.0 * self.quote_vol(self.cum, quote) for quote in self.quotes]

    def verify(self):
        """Refuses what a calibrated surface may not carry - an ATM pillar out of reach, the two
        declared `Refuse | Floor` guards, cap headroom - before the report, by name."""
        stuck = [i for i, x in enumerate(self.atm_misses) if not abs(x) <= self.atm_miss_max]
        if stuck:
            gone = [i for i in stuck if not np.isfinite(self.atm_misses[i])]
            raise ValueError(
                '{}: no xi level reprices the {} ATM pillar{} - {}. The triangular bootstrap is '
                'monotone in a pillar\'s own level, so a pillar that walks its Newton steps out '
                'and still misses is one the OTHER parameters have put out of reach. Quote a '
                'surface this model can reach{}'.format(
                    self.market_price, len(stuck), '' if len(stuck) == 1 else 's',
                    ', '.join('{:g}y {:+.2%}'.format(self.knots[i + 1], self.atm_misses[i])
                              for i in stuck),
                    '' if not gone else
                    '. {} of them priced to NaN, which is an arithmetic failure and not a '
                    'miss: the walk overflowed before any level could be wrong'.format(
                        len(gone))))

        # the BOX is the event, not the soft margin above it: the fit is bounded at exactly C_Min,
        # so a bucket landing there is one the box stopped rather than an interior optimum
        c = 1.0 - np.array(self.state['Rho_S']) ** 2 - self.state['Rho_L'] ** 2
        edge = np.flatnonzero(c <= self.c_min + 1e-9)
        if edge.size:
            convexity = self.band(1.05, 1.25)
            message = (
                '{}: the idiosyncratic share c = 1 - Rho_S^2 - Rho_L^2 is ON its floor C_Min={:g} '
                'in {} of {} bucket{} ({}), so the box - not the data - is what stopped Rho_S, and '
                'this surface wants MORE leverage than one shock plus a co-jump can carry: it '
                'wants a ONE-SHOCK model. The 110-120% convexity residual that goes '
                'with the floor is {}'.format(
                    self.market_price, self.c_min, edge.size, c.size, '' if c.size == 1 else 's',
                    ', '.join('{:g}y c {:.3f} at Rho_S {:+.4f}'.format(
                        self.buckets[i], c[i], self.state['Rho_S'][i]) for i in edge),
                    'nothing quoted there' if convexity is None else
                    '{:+.3f} vol points RMS, worst {:+.3f}'.format(*convexity)))
            self.guard('Idiosyncratic_Share', message,
                       'Floor to TAKE C_Min and have the fit say so, lower C_Min (a book may run '
                       'it near 0.06, at the cost of the second-order noise the share buys), or '
                       'fit a surface whose skew this structure can reach')

        # nothing is scaled and nothing re-solved: theta* is where the fit left it, and the guard
        # is a READING of it - which is what keeps the quote contraction taken at the same point
        spreads, (lo, hi) = self.spreads(), self.log_vol_sd_band
        stuck = [i for i, sd in enumerate(spreads) if not lo <= sd <= hi]
        if stuck:
            message = (
                '{}: the stationary log-vol sd 0.5*sqrt(Sigma_S^2/2Kappa_S + Sigma_L^2/2Kappa_L) '
                'sits outside the {:g}-{:g} band declared for {} in {} of {} bucket{} ({}), so this '
                'surface wants vol dynamics its market is not declared to carry'.format(
                    self.market_price, lo, hi, self.asset_class, len(stuck), len(spreads),
                    '' if len(spreads) == 1 else 's',
                    ', '.join('{:g}y sd {:.3f} at Sigma_S {:.4f}, Sigma_L {:.4f}'.format(
                        self.buckets[i], spreads[i], self.state['Sigma_S'][i],
                        self.state['Sigma_L']) for i in stuck)))
            self.guard('Stationary_Spread', message,
                       'Floor to TAKE the fit and have it say so, widen Kappa_S or Kappa_L (the '
                       'sd is theirs as much as the vol-of-vol pair), or quote a surface whose '
                       'vol-of-vol sits in the band')

        scalars, levers = self.build(None, ())
        self.headroom = self.cap_headroom(scalars, levers, self.l_at(self.levels))
        if self.headroom > self.cap_headroom_max:
            raise ValueError(
                '{}: {:.3e} of path-days sit at or above the corner Cap_A={:.4g}, above '
                'the {:g} a calibrated surface may carry. The cap exists to make E[S^p] finite and '
                'to stop an exp overflowing, NOT to shape a smile, so a fit that reaches it is a '
                'failure rather than a warning. Raise Cap_A, or fit a surface whose '
                'vol-of-vol this model can carry'.format(
                    self.market_price, self.headroom, self.cap_level(),
                    self.cap_headroom_max))

    def guard(self, field, message, remedy):
        """One `Refuse | Floor` guard: Refuse raises the message with the remedy, Floor takes what
        the fit reached and notes it."""
        if self.instrument[field] == 'Refuse':
            raise ValueError('{}. Set {} to {}'.format(message, field, remedy))
        self.notes.append('FLOORED - ' + message.split(': ', 1)[1])

    def prior_ratios(self, labels, jacobian, priors):
        """Each prior row's column norm over one quote row's per coordinate, or `None` where the
        quotes carry under `Jacobian_Rcond` of it and are silent on the coordinate."""
        matrix = np.atleast_2d(np.asarray(jacobian, dtype=float))
        data = np.linalg.norm(matrix, axis=0)
        rows = (np.linalg.norm(np.atleast_2d(priors), axis=0) if len(priors)
                else np.zeros(len(labels)))
        quote = data / max(np.sqrt(matrix.shape[0]), 1.0)
        return {name: None if silent else row / q for name, row, q, silent in
                zip(labels, rows, quote, data < self.rcond * rows)}

    def unidentified(self):
        """`{coordinate: ratio or None}` for every prior row past `PRIOR_RATIO` quote rows or on a
        coordinate the quotes are silent on, read at the last stage that fitted it."""
        seen = {}
        for table in self.tables:
            seen.update(self.prior_ratios(*table[1:]))
        return {name: ratio for name, ratio in seen.items()
                if ratio is None or ratio > self.PRIOR_RATIO}

    def on_guard(self):
        """Every guard theta* sits on, as one sentence, or `''`: a box edge, the conditioning or
        `C_Min` margin, or a prior row on a coordinate the quotes do not identify."""
        held, bound = [], np.sqrt(1.0 - utils.LogVar2FJ.COND_MIN)
        for name in ('Sigma_S', 'Alpha', 'Sigma_L'):
            values = [self.state[name]] if name == 'Sigma_L' else self.state[name]
            low, high = self.box[name]
            for i, edges in enumerate(zip(*utils.on_box(values, low, high))):
                if any(edges):
                    held.append('{} on its {:g} box'.format(
                        self.label((name, None if name == 'Sigma_L' else i)),
                        low if edges[0] else high))
        skew = np.abs(np.array(self.state['Beta'])) / np.array(self.state['Alpha'])
        for i in np.flatnonzero(skew >= bound - self.c_margin):
            held.append('|beta|/alpha {:.3f} at {:g}y within {:g} of {:.4f}'.format(
                skew[i], self.buckets[i], self.c_margin, bound))
        c = 1.0 - np.array(self.state['Rho_S']) ** 2 - self.state['Rho_L'] ** 2
        for i in np.flatnonzero(c <= self.c_min + self.c_margin):
            held.append('c {:.3f} at {:g}y within {:g} of its C_Min floor {:g}'.format(
                c[i], self.buckets[i], self.c_margin, self.c_min))
        held += ['prior on an unidentified coordinate: {} {}'.format(
            name, 'quotes silent' if ratio is None else '{:.0f}x'.format(ratio))
            for name, ratio in sorted(self.unidentified().items())]
        return 'ON GUARD: {}'.format('; '.join(held)) if held else ''

    def written(self):
        """The `LogVar2FJModelParameters` price factor: the scalars, the structural fields and the
        five curves, the xi strip being the one `finish` banked."""
        knots, values = self.l_knots(self.levels)
        values = values.detach().exp()
        return {'Property_Aliases': None,
                **{name: float(self.state[name]) for name in utils.LogVar2FJ.PARAM_NAMES},
                'Cap_A': self.cap_level(),
                'Steps_Per_Year': float(self.instrument['Steps_Per_Year']),
                'C_Min': self.c_min, 'Residual_Law': self.law,
                'On_Guard': self.on_guard(),
                'Skew_Gradient': self.skew_line(),
                'Stickiness_Band': self.skew_band,
                'Xi_Curve': utils.Curve([], [[float(t), float(v)]
                                             for t, v in zip(knots, values)]),
                **{name: utils.Curve([], [[float(t), float(v)] for t, v
                                          in zip(self.buckets, self.state[name])])
                   for name in utils.LogVar2FJ.BUCKET_NAMES}}

    def connect(self):
        """Every fitted parameter still connected to the quotes, keyed as `_build_factor_state`
        mints its leaf - what `Calculation.factor_leaf` is offered under `Quote_Sensitivity`."""
        _, values = self.l_knots(self.levels)
        return dict({name: self.scalars[name].reshape(1) for name in utils.LogVar2FJ.PARAM_NAMES},
                    Xi_Curve=torch.exp(values),
                    **{name: self.levers[name] for name in utils.LogVar2FJ.BUCKET_NAMES})

    def forward_smile(self, cum):
        """`{(T1, Delta): {strike: (model vol, target vol)}}` for every forward-start target."""
        smiles, shapes = {}, self.spot_shapes(cum)
        for target in self.targets:
            smiles.setdefault((target.T1, target.tenor), {})[target.strike] = (
                float(self.forward_vol(cum, target).detach()),
                float(self.target_vol(target, shapes[target.tenor]).detach()))
        return smiles

    def spot_shapes(self, cum, forwards=None):
        """The model's own spot `(atm, slope, butterfly)` at each forward row's own `Delta`."""
        rows = self.targets if forwards is None else forwards
        return {target.tenor: self.spot_shape(cum, target.tenor) for target in rows}

    def skew_gradient(self):
        """The reserve line's model half per forward tenor: `d(Delta_skew)/dBeta` and
        `d(Delta_skew)/dRho_S` in the last bucket, in vol points per unit of the model's own
        number, at theta* off the forward rows' graph; the banked strip is put back."""
        if not (self.targets or self.reported):
            return {}
        bucket = self.buckets.size - 1
        coords = [('Beta', bucket), ('Rho_S', bucket)]
        leaf = torch.tensor([self.value_of(coord) for coord in coords], dtype=self.prec,
                            device=self.device, requires_grad=True)
        banked, warm, rows = self.levels, self.warm, {}
        levers, cum = self.evaluate(leaf, coords)[1:]
        scale = [float(torch.autograd.grad(levers[name][bucket], leaf, retain_graph=True)[0][i])
                 for i, (name, _) in enumerate(coords)]
        span = np.log(self.psi_strikes[2] / self.psi_strikes[0])
        tenors = {}
        for target in self.targets or self.reported:
            tenors.setdefault((target.T1, target.tenor), {})[
                round(target.strike, 10)] = target
        for (T1, tenor), strikes in tenors.items():
            if not {round(k, 10) for k in self.psi_strikes} <= set(strikes):
                continue
            low, high = (self.forward_vol(cum, strikes[round(k, 10)])
                         for k in (self.psi_strikes[0], self.psi_strikes[2]))
            spot = self.smile(cum, *self.spot_rung[tenor], self.psi_strikes)
            skew = ((low - high) - (spot[0] - spot[2])) / span
            rows[(T1, tenor)] = [100.0 * float(g) / s for g, s in zip(
                torch.autograd.grad(skew, leaf, retain_graph=True)[0], scale)]
        self.levels, self.warm = banked, warm
        return rows

    def skew_line(self):
        """`Skew_Gradient`'s text: the earliest window `skew_rows` holds, blank where none was
        read."""
        nearest = min(self.skew_rows, default=None)
        return '' if nearest is None else '{:.12g},{:.12g}'.format(*self.skew_rows[nearest])

    def band(self, lo, hi):
        """`(RMS, worst)` vol-point miss over the moneyness band `[lo, hi]`, or `None` where nothing
        is quoted in it."""
        rows = [x for x, quote in zip(self.misses, self.quotes)
                if lo <= quote.strike / quote.forward <= hi]
        return (np.sqrt(np.mean([x * x for x in rows])), max(rows, key=abs)) if rows else None


class LVReport:
    """What one finished LogVar2FJ fit measured, logged beside what it wrote."""

    def __init__(self, fit):
        self.fit = fit

    def shape_readings(self, tenors=(1.0 / 12.0, 1.0)):
        """`[(T, alpha*delta_A)]` at the calibrated expiries nearest `tenors`, each read in the
        bucket it falls in."""
        fit, rows, seen = self.fit, [], set()
        for tenor in tenors:
            quote = min(fit.atm, key=lambda q: abs(q.T - tenor))
            bucket = int(fit.grid.index_at(quote.T))
            if quote.T not in seen:
                seen.add(quote.T)
                rows.append((quote.T, float(fit.residual_shape(
                    fit.levers, fit.cum, quote.j, bucket).detach())))
        return rows

    def identification(self, tag, labels, jacobian, priors):
        """One stage's table: the singular values of its data rows' column-scaled Jacobian beside
        the unscaled norms, each prior row in quote rows, and every direction under
        `Jacobian_Rcond` named by its parameter loadings."""
        matrix = np.atleast_2d(np.asarray(jacobian, dtype=float))
        norms = np.linalg.norm(matrix, axis=0)
        values, vectors = np.linalg.svd(matrix / np.where(norms > 0.0, norms, 1.0))[1:]
        values = np.concatenate([values, np.zeros(len(labels) - values.size)])
        logging.info('  identification, {}: singular values {}; column norms {}'.format(
            tag, '  '.join('{:.3e}'.format(x) for x in values),
            ', '.join('{} {:.2e}'.format(name, x) for name, x in zip(labels, norms))))
        if len(priors):
            logging.info('    the PRIOR rows there, each as a multiple of ONE quote row at the '
                         "data's own RMS: {}".format(', '.join(
                '{} {}'.format(name, 'silent' if ratio is None else
                '{:.3g}x'.format(ratio))
                for name, ratio in
                self.fit.prior_ratios(labels, jacobian, priors).items())))
        for i in np.flatnonzero(values <= self.fit.rcond * max(values[0], 1e-300)):
            logging.info('    FLAT at {:.3e} ({:.1e} of the largest): {}'.format(
                values[i], values[i] / values[0], ', '.join(
                    '{} {:+.3f}'.format(labels[k], vectors[i][k])
                    for k in np.argsort(-np.abs(vectors[i])))))

    def log(self):
        """The fit's parameters, its curve against the market's ATM^2, the per-rung and banded
        misses, the residual's shape, the priors and guards, the reserve line, the identification
        tables and the cost, at INFO."""
        fit = self.fit
        quotes, buckets, misses, prior = fit.quotes, fit.buckets, fit.misses, fit.prior
        logging.info('{} LogVar2FJ ({} mode): {}'.format(
            fit.market_price, fit.mode, ', '.join(
                '{} {}'.format(name, utils.LogVar2FJ.text(fit.state[name], '.6g'))
                for name in utils.LogVar2FJ.PARAM_NAMES + utils.LogVar2FJ.STRUCTURAL_NAMES)))
        logging.info('  THE CURVE per segment: xi IS the model\'s expected forward variance E[h], '
                     'so it stands beside the market\'s own ATM^2 forward variance, the gap being '
                     'Black\'s convexity:')
        for lo, hi, level, x in zip(fit.knots[:-1], fit.knots[1:], fit.levels, fit.xi):
            fitted = np.exp(float(level.detach()))
            logging.info('    {:5.3f}-{:5.3f}y  market ATM^2 {:.2%} vol, fitted xi {:.2%} vol '
                         '({:+.2f} vol points)'.format(
                lo, hi, np.sqrt(max(x, 0.0)), np.sqrt(fitted),
                100.0 * (np.sqrt(fitted) - np.sqrt(max(x, 0.0)))))
        for name in utils.LogVar2FJ.BUCKET_NAMES:
            logging.info('  {}: {}'.format(name, ', '.join(
                '{:g}y {:+.4f}{}'.format(t, v, '' if not fit.free else ' ({})'.format(
                    'free' if name in fit.free.get(b, ()) else
                    'tied' if (name, b) in fit.ties else 'held'))
                for b, (t, v) in enumerate(zip(buckets, fit.state[name])))))
        for T in sorted({quote.T for quote in quotes}):
            # keyed by POSITION: two quotes at one strike and expiry compare equal as tuples
            rung = [i for i, quote in enumerate(quotes) if quote.T == T]
            worst = max(rung, key=lambda i: abs(misses[i]))
            logging.info('    {:5.3f}y  RMSE {:5.3f} vol points over {} quotes, worst {:+.3f} at '
                         '{:.0%} of forward'.format(
                T, np.sqrt(np.mean([misses[i] ** 2 for i in rung])), len(rung),
                misses[worst], quotes[worst].strike / quotes[worst].forward))
        weights = np.array([quote.weight for quote in quotes]) ** 2
        logging.info('  RMSE {:.3f} vol points unweighted over {} quotes, {:.3f} vega-weighted '
                     '(the objective\'s own); the bootstrap\'s ATM misses {}'.format(
            np.sqrt(np.mean([x ** 2 for x in misses])), len(quotes),
            np.sqrt(np.dot(weights, np.square(misses)) / weights.sum()),
            ', '.join('{:+.1e}'.format(x) for x in fit.atm_misses)))
        logging.debug('  every quote, model minus market in vol points: {}'.format(', '.join(
            '{} {:+.6f}'.format(name, miss)
            for name, miss in zip(fit.descriptors, misses))))
        for tag, lo, hi in (('wing 70-80%', 0.65, 0.85), ('convexity 110-120%', 1.05, 1.25)):
            found = fit.band(lo, hi)
            logging.info('  {} residual: {}'.format(
                tag, 'nothing quoted there' if found is None else
                '{:+.3f} vol points RMS, worst {:+.3f}'.format(*found)))

        c = 1.0 - np.array(fit.state['Rho_S']) ** 2 - fit.state['Rho_L'] ** 2
        cond = 1.0 - (np.array(fit.state['Beta']) / np.array(fit.state['Alpha'])) ** 2
        logging.info(
            '  residual (alpha, beta) {}; leverage products rho_s*sigma_s {}, rho_l*sigma_l '
            '{:+.3f}; c {}, conditioning share gamma^2/alpha^2 {}, c_eff {}'.format(
                '/'.join('({:.3f}, {:+.3f})'.format(a, b) for a, b
                         in zip(fit.state['Alpha'], fit.state['Beta'])),
                '/'.join('{:+.3f}'.format(x * y) for x, y
                         in zip(fit.state['Rho_S'], fit.state['Sigma_S'])),
                fit.state['Rho_L'] * fit.state['Sigma_L'],
                '/'.join('{:.3f}'.format(x) for x in c),
                '/'.join('{:.3f}'.format(x) for x in cond),
                '/'.join('{:.3f}'.format(x * y) for x, y in zip(c, cond))))
        # `c_eff` is the share of a return's variance that stays in the Gaussian a stride reads, so
        # the OSS advantage scales with its 3/2 power against the reference 0.22 it is sized on
        logging.info(
            '  efficiency factor (sqrt(c_eff)/0.22)^3 {}; residual shape alpha*delta_A {}, against '
            'the {:g} floor (1 strongly non-Gaussian, 15 nearly Gaussian){}'.format(
                '/'.join('{:.2f}'.format((np.sqrt(x * y) / 0.22) ** 3)
                         for x, y in zip(c, cond)),
                ', '.join('{:.4g}y {:.4g}'.format(T, x) for T, x in self.shape_readings()),
                fit.shape_floor, '' if fit.priors and fit.shape_floor > 0.0 else
                ' - NOT IN FORCE'))
        weights = [row[2] for row in fit.prior_rows[:2]] or [
            prior.leverage_weight, prior.leverage_weight / prior.sigma_reference]
        logging.info(
            '  leverage prior, TWO rows - {} - at weights {:g} on rho_s and {:g} on the product, '
            'and one standard error of ANY prior row costs {:.3g}, which is what one quote missing '
            'by one vol point costs on this ladder{}. The rho_s row also signs the seed and stage '
            '4'.format(prior.source, weights[0], weights[1], prior.quote_point,
                       '' if fit.priors else '; NOT IN FORCE, Model_Priors is Off'))
        guard = fit.on_guard()
        if guard:
            logging.warning('  {}: {} - the box or the floor is holding theta*, not the data; the '
                            'factor carries the flag'.format(fit.market_price, guard))
        alpha = prior.alpha_prior()
        if alpha is not None:
            ratio = alpha / (fit.state['Alpha'][0] or float('nan'))
            logging.info(
                '  alpha^P {:.3f} against alpha^Q {:.3f}, ratio {:.2f}{} - Esscher invariance is '
                'an assumption about the risk premium, not a theorem about the market'.format(
                    alpha, fit.state['Alpha'][0], ratio,
                    '' if 0.5 <= ratio <= 2.0 else ' - PAST 2x, the sanity check'))
        for line in ([] if fit.slope is None else utils.active_bounds(
                fit.labels, np.array([fit.value_of(x) for x in fit.fitted]),
                fit.edges[0], fit.edges[1], fit.slope)):
            logging.info('  {}'.format(line))
        logging.info('  stationary log-vol sd {}; corner {} with {:.2e} of path-days at or above '
                     'it'.format('/'.join('{:.3f}'.format(x) for x in fit.spreads()),
                                 utils.LogVar2FJ.text(fit.cap_level(), '.4g'), fit.headroom))
        for note in fit.notes:
            logging.warning('  {}: {}'.format(fit.market_price, note))
        for line in ([fit.pinned_slow] if fit.pinned_slow else []) + \
                    fit.prior_lines + fit.identified_days:
            logging.info('  {}'.format(line))
        # THE RESERVE LINE wherever the forward smile was not QUOTED: the calibrator has no deal,
        # so it states the map from the two levers to Delta_skew and writes the declared tenor's
        # pair on the factor, `utils.LogVar2FJ.skew_reserve` composing the rest at the deal
        written = min(fit.skew_rows, default=None)
        for (T1, tenor), (d_beta, d_rho) in (
                {} if fit.source in ('Quotes', 'Reference') else fit.skew_rows).items():
            logging.info(
                '  reserve line at {:g}y into {:g}y, {}{}: band {:g} vol points, '
                'd(Delta_skew)/dBeta {:+.4g} and d(Delta_skew)/dRho_S {:+.4g} vol points per '
                'unit at the {:g}y bucket - a deal\'s |dPV/dDelta_skew| x band is '
                'utils.LogVar2FJ.skew_reserve of these and its own (dPV/dBeta, dPV/dRho_S)'
                .format(T1, tenor, 'the DECLARED window read post-fit on its own grid'
                if fit.skew_declared else 'the LADDER\'s own block ends, no '
                                          'declared window being reachable',
                        ' - WRITTEN on the factor as Skew_Gradient, the tenor every deal\'s '
                        'reserve is quoted at' if (T1, tenor) == written else '',
                        fit.skew_band, d_beta, d_rho, buckets[-1]))
        for table in fit.tables:
            self.identification(*table)
        logging.info('  {} evaluations, {} Jacobians and {} pillar passes in {:.1f}s'.format(
            fit.calls['n'], fit.calls['j'], fit.calls['l'], fit.elapsed))
        if fit.polish_only:
            logging.info('  WARM START off the factor Price Factors already carries: the staged '
                         'search is the basin, and a fitted factor names it, so this fit is the '
                         'joint POLISH from that state alone - {} evaluations'.format(
                fit.calls['n']))
        fit.family.quote_trailer(fit.instrument)


class LogVar2FJModelParameters(OptionQuoteFamily):
    documentation = (
        'Fx And Equity',
        ['The LogVar2FJ model fitted to European options and, where the',
         'desk has them, to FORWARD-START smiles. Two mean-reverting log-variance factors carry a',
         'normal-inverse-Gaussian residual on the variance clock:',
         '',
         '$$h_t=\\exp\\big(\\mathrm{cap}(\\ell_t+s_t)\\big),\\qquad',
         '\\ell\\to L^*(t)\\ \\text{at}\\ \\kappa_\\ell,\\qquad s\\to0\\ \\text{at}\\ \\kappa_s$$',
         '',
         'and the part of a return leverage does not explain is $NIG(\\alpha(t),\\beta(t))$ on the',
         'clock $cV$. GIVEN the two shocks and the mixer a block return is EXACTLY Gaussian, so a',
         'vanilla is that block\'s conditional Black averaged over paths -',
         '`pricing.lognormal_fired_gain` per path, no new closed form - and the whole objective',
         'stays on one AAD tape. Each block is priced off its own SAMPLE forward, a martingale',
         'control variate that costs one `logsumexp` and takes the fixed draws\' level bias out of',
         'every strike at once.',
         '',
         'THE OBJECTIVE IS IN VOL SPACE TO FIRST ORDER. Each quote contributes',
         '$(V_{model}-V_{market})/\\mathcal{V}_{market}$ with $\\mathcal{V}$ the Black vega at the',
         'quoted vol, so autograd carries the residual with no root find on the tape. The draws are',
         'FIXED and antithetic, which is what makes the objective deterministic and its gradient',
         'the derivative of the number `least_squares` is minimising. The report prints the TRUE',
         'inversion of both premia, unweighted and vega-weighted: three functionals, named, because',
         'a fit is quoted in whichever the reader had in mind.',
         '',
         'ONE WALK PER EVALUATION, ON THE QUOTES OWN CLOCK. The internal grid steps one trading',
         'day on the *Steps_Per_Year* clock - the day is the model - and a block ends at',
         'every quoted maturity and at every forward target $T_1$ and $T_2$ - its last step a STUB',
         'landing it exactly on $T$, so the variance the fit reads at a maturity is the variance a',
         'pricer reads at that tenor of the curve written out. The block sums are cumulated, so a',
         'maturity is a prefix and a forward window the difference of two prefixes.',
         '',
         'THE FIT IS TWO NESTED SOLVES, and the inner one runs at EVERY outer iterate.',
         '',
         '1. THE INNER TRIANGULAR BOOTSTRAP. $\\xi$ is the EXPECTED FORWARD VARIANCE $E[h]$,',
         'PIECEWISE CONSTANT on the segments between ATM expiries - flat forward variance, the',
         'shape var-swap strips are quoted in - and the OU level is DERIVED from it per candidate,',
         '$L^*=\\log\\xi-\\mathrm{Var}(\\ell+s)/2$, so the ATM level is invariant to the',
         'vol-of-vol by construction and the solve is near-identity. The levels are solved',
         'SEQUENTIALLY, one per segment against that segment own ATM expiry premium: an option to',
         '$T$ never reads $\\xi$ beyond $T$, so the system is exactly triangular, and the premium',
         'is monotone in the level, so a damped Newton off the previous sweep converges in a step',
         'or two and is iterated to *Pillar_Tolerance*.',
         '2. THE OUTER FIT concentrates $\\xi$ out - the ATM ladder is a CONSTRAINT, not a term, so',
         'no smile improvement may pay for an ATM miss - plus the soft constraints of 5.2 and, in',
         '`Global` mode with a forward source, the forward-smile block of 5.3.',
         '',
         'The gradient is what makes that affordable. The level each pillar RETURNS is one NEWTON',
         'STEP at its own root, $x_k=x_k^*-F_k/\\mathrm{detach}(\\partial F_k/\\partial x_k)$, taken',
         'off the same graph the last iteration built - so the implicit function theorem across the',
         'triangle is an EXPRESSION autograd differentiates rather than a rule, and the outer',
         'solver is `least_squares` with an exact vmapped Jacobian rather than a simplex.',
         '',
         'BOOTSTRAP MODE IS THAT TRIANGULAR DISCIPLINE APPLIED TO THE SMILE (5.4.1): the ladder',
         'WING EXPIRIES are the calendar buckets, bucket $k$ is fitted to expiry $k$ wing quotes',
         'given buckets $<k$, and how many parameters it frees is how many wing quotes that expiry',
         'carries, in the order $\\beta,\\sigma_s,\\rho_s,\\alpha$; the rest are TIED, carried',
         'from the previous bucket. The slow pair stays global.',
         '',
         'RISK IN QUOTE SPACE. *Quote_Sensitivity* **Yes** keeps the written parameters connected',
         'to the numbers quoted, so one backward pass reports $dV/dq$ beside $dV/d\\theta$. The',
         'outer fit is a least-squares minimum, so its half is the Gauss-Newton contraction at the',
         'stationarity point $(J^TJ)\\,d\\theta/dq=-J^T dr/dq$ - `utils.LeastSquaresSolve`, the node the',
         'swaption family also solves through - taken over the coordinates the KKT active set',
         'leaves FREE, since one the box holds is held by the box and its own derivative is zero;',
         'the $\\xi$ strip half is the same Newton splice the inner solve already carries, so',
         '`dxi/dq` needs no rule of its own. The market premium each row measures against carries',
         'its quote as a splice worth zero forward, so turning this on cannot move a fitted digit.',
         'REFUSED in `Bootstrap` mode: bucket $k$ is fitted given buckets $<k$ and $\\theta^*$ is',
         'then a stationary point of no single objective, so the contraction would report the last',
         'bucket derivative as the whole.',
         '',
         'THE STAGES, warm-started, each `least_squares` over its own subset:',
         '',
         '0. $\\kappa_\\ell$, $\\kappa_s$ and the cap are STRUCTURAL and never enter a fitted',
         'vector; $(\\alpha,\\beta)$ seed at the index-sized $(44,-22)$ - $\\alpha$ at a history own',
         '$\\alpha^P$ where the job carries one - and $(\\rho_s,\\sigma_s)$ at $(-0.75,2.4)$, $\\rho_s$',
         'signed by the leverage prior. The residual drift is FORCED by no-arbitrage rather than fitted, so',
         'there is no intensity and no share to size.',
         '1. The curve $\\xi$ from the ATM$^2$ forward-variance increments, then the',
         'triangular bootstrap - at every iterate thereafter.',
         '2. $(\\alpha,\\beta)$ on the 1-3 month rows, in the UNCONSTRAINED coordinates',
         '$\\alpha=\\tfrac12+\\varepsilon+\\mathrm{softplus}(a)$,',
         '$\\beta=-\\tfrac12+(\\alpha-\\tfrac12-\\varepsilon)\\tanh(b)$, which land inside',
         '$|\\beta|<\\alpha$, $|\\beta+1|<\\alpha$ for every iterate - the transform lives in the',
         'calibrator and the model applies none. $\\alpha$ and the residual SKEW SHARE',
         '$\\beta/\\alpha$ - the coordinate the price depends on, a row on $\\beta$ alone being',
         'obeyed for free by running $\\alpha$ to its ceiling - each take a PRIOR ROW in a',
         'HIERARCHY of three: the HISTORY own estimate where its standard error is at or under the',
         'class prior spread, which is what INFORMATIVE means here; else the CLASS DEFAULT',
         '(*Alpha_Prior_Defaults*, *Residual_Skew_Share_Defaults*) at that spread, reported by name',
         'as *class prior, history uninformative* with the standard error that failed the test.',
         'Both rows are ALWAYS PRESENT and *Residual_Horizon* switches neither off - it names',
         'whether this ladder prices the 1-3m tails at all, the wings outvoting a soft row where',
         'they mean it. A',
         'contaminated $\\alpha^P$ - clock share past *Contamination_Ratio* times the model own $c$',
         '- is uninformative whatever its error, the share with it. Every row is SOFT and',
         'never a pin, so the wings still move the pair and it stays in $\\theta^*$, in the',
         'Jacobian and in the quote contraction. 3. $(\\rho_s,\\sigma_s)$ on the 1-12',
         'month rows WITH the forward rows, which are what identifies the split between the',
         'residual skew and $\\rho_s\\sigma_s$. 4. $(\\rho_\\ell,\\sigma_\\ell)$ beyond one year, and ONLY where',
         'the ladder carries wing quotes at 18 months or longer; otherwise both are PINNED with',
         'the report line *pinned: not identified by this ladder*, as the $\\kappa$ pair is. WHAT',
         'THEY ARE PINNED AT is read in one order: *Slow_Factor_Prior* where the block declares',
         'one, else the ASSET CLASS default off the factor type *Underlying* resolves',
         'to - FX $(0.2,0.5)$, an index $(0.4,1.0)$ - whose SIGN is that of the $\\rho_s$ in force',
         'at the pin, so the slow skew is never set against the fast one stage 3 has just fitted.',
         'A HISTORY IS NOT PINNED AT: it has no box beneath it, so an estimate outside',
         '*Rho_L_Bounds* went straight through, crushed the derived $\\rho_s$ box under the fitted',
         'value and took a later stage non-finite - a history slow pair is a soft ROW like every',
         'other prior, and only where the ladder FITS that pair.',
         'A pin is applied to every name without an 18-month wing, so a floor-by-default would',
         'understate a whole book long-horizon vol in ONE direction; the floor is the BOX beneath',
         'both instead - $\\sigma_\\ell\\ge0.3$, there for exposures, a zero slow factor',
         'collapsing a CVA profile vol distribution onto the fast factor spread, which reverts',
         'within months. A DECLARED prior under it refuses by name; a fit sitting on it with long',
         'expiries present is a RESULT, reported and never refused.',
         '5. THE FORWARD BLOCK, which exists with a MARKET OR REFERENCE SOURCE or not at all',
         '- *Quotes* and *Reference* are sources and are fitted, and there is no third',
         'form - a prior on the forward smile is a desk VIEW, and a view fitted as a target is',
         'paid for in vanilla fit. What carries a view is the RESERVE LINE below. Where the block',
         'runs: the later buckets of $\\beta(t)$, then of $\\rho_s(t)$.',
         'THE TARGETS ARE',
         'DIFFERENCES, in vol points and BOTH of them: $\\Delta_{skew}$ is the forward 90-110',
         'slope less the spot one at maturity $\\Delta$, $\\Delta_{bfly}$ the same for the 90/110',
         'butterfly, both MEASURED off the source own forward smile less the MARKET own',
         'spot one. So the residual is ONE term whatever the source - the model difference less',
         'the target - taken on the model OWN spot smile per evaluation, in VOL POINTS. A RATIO',
         'would divide by a spot quantity within a few tenths of zero (the butterfly at 3-6',
         'months on an index, the skew on any symmetric FX smile) and ask the fit to match noise.',
         'Neither targets the forward ATM LEVEL, which the ATM ladder already pins.',
         'The spot smile at $T$ never sees a bucket',
         'later than $T$ and a forward smile prices on nothing else, which is why the lever is',
         'CALENDAR TIME and not the vol state (2.3.1, measured three ways).',
         '6. A joint polish over everything but the $\\kappa$ pair and the cap.',
         '',
         'BOUNDS LIVE HERE AND NOWHERE ELSE. The engine puts no transform on $\\rho_s$ - one cost a',
         'tenth of the spot skew before it was found - so the box is the calibrator own:',
         '$\\rho_s\\in[-\\rho_{max},\\rho_{max}]$, $\\rho_{max}=\\sqrt{1-\\rho_\\ell^2-c_{min}}$, SYMMETRIC',
         'because the sign is the leverage prior to state (one-sided under *Model_Priors* Off),',
         're-derived as $\\rho_\\ell$ moves, with a',
         'soft penalty sized to bite in the last percent of it. A bucket landing ON that box is the',
         'box stopping $\\rho_s$ rather than the data, which is a surface asking for a ONE-SHOCK',
         'model. Neither guard Floor moves a fitted number: both TAKE what the fit reached and say',
         'so, which is what leaves the quote contraction at the point it was taken.',
         '',
         'THE PRIORS ARE ROUTINE, AND THEY ALL COST THE SAME. A leverage prior the fit is',
         'never without, TWO rows - $\\rho_s$ and the product $\\rho_s\\sigma_s$, each read from',
         'the block (*Leverage_Prior*, *Leverage_Product_Prior*, with their own standard errors',
         'where the desk has them), else a history, else the class default, on the ENGINE axis, so',
         '$\\sigma_s$ sits at the RATIO of the two rather than on its box; a floor on the residual',
         'shape $\\alpha\\delta_A$ at the',
         'shortest expiry, *Residual_Shape_Floor*, nothing bounding $\\alpha$ from below and a',
         'symmetric smile buying convexity at the map own floor otherwise; the residual scale and',
         'its skew share; and a history slow pair where the ladder',
         'fits it. EVERY one of those rows is scaled so that ONE STANDARD ERROR of miss costs what',
         'ONE QUOTE MISSING BY ONE VOL POINT costs on the ladder at hand - the quote rows being',
         'decimal-vol misses times weights normalised to $\\sum w^2=1$, that price is a vol point',
         'at their root-mean-square, 0.0025 on a sixteen-rung ladder - so any quote that speaks',
         'outvotes any of them, and every history number is CLIPPED into the box the fit moves',
         'that lever in, named where it bites. A prior with no standard error to its name (a',
         'declaration, a class leverage default) takes *Leverage_Prior_Weight* 0.02, which is the',
         'same statement with 0.1 for the error nobody declared. *Model_Priors* Off is the',
         'vanilla-only objective and box, there so a document declaring none of them refits to the',
         'bit.',
         '',
         'THE RESERVE LINE IS OWED WHEREVER THE FORWARD SMILE WAS NOT QUOTED, which is every',
         'ladder today, so it is written on the factor: $\\partial\\Delta_{skew}/\\partial\\beta$ and',
         '$\\partial\\Delta_{skew}/\\partial\\rho_s$ in the last bucket at the DECLARED forward tenor,',
         'beside *Stickiness_Band*. With the block OFF those rows are REPORTED rather than',
         'targeted - the same *Forward_Tenors* windows moved onto the grid own BLOCK ENDS, since a',
         'window end that is not one splits a block into two mixers and moves $\\theta^*$. What',
         'the FACTOR carries is read once more at the DECLARED window, on a grid built to carry it',
         'exactly and off the $\\theta^*$ just written. A deal reporting *Greeks* **First** composes',
         '$|\\partial PV/\\partial\\Delta_{skew}|\\times$ band from them and its own two',
         'derivatives (`utils.LogVar2FJ.skew_reserve`) and reports it as **Skew_Reserve** - PER',
         'DEAL and for the portfolio, the set number being those deals contracted rather than',
         'summed.',
         '',
         'WHAT THE FIT REPORTS. The vol-point miss per maturity and over the surface, unweighted',
         'and vega-weighted; the wing and convexity residuals; per bucket the parameters and',
         'which were free; the leverage products, $c$, the conditioning share',
         '$\\gamma^2/\\alpha^2$ and their product $c_{eff}$ with the efficiency factor',
         '$(\\sqrt{c_{eff}}/0.22)^3$; the residual shape $\\alpha\\delta_A$ at 1m and 1y against the',
         'floor; the leverage prior in force with its source; every guard $\\theta^*$ is sitting',
         'ON, written on the factor as **On_Guard** and warned here; $\\alpha^P$ against $\\alpha^Q$, flagged',
         'past 2x, where a history carries one; the stationary log-vol sd; the cap',
         'headroom; THE CURVE per segment - the market own ATM$^2$ forward variance beside the',
         'fitted $\\xi$, which IS the model $E[h]$, so the gap is Black convexity and there is no',
         'log level in either; the reserve line per forward tenor where the forward smile was not',
         'quoted; and THE IDENTIFICATION TABLE, the SVD of the DATA rows of the Jacobian at each',
         'fitted point, its columns scaled by $1/\\|J_{:,j}\\|$ and the unscaled norms beside it - scaling reads',
         'conditioning, and only the norm says that a well-conditioned direction moves nothing. The',
         'penalty rows are left out because their singular value is an algebraic constant of',
         '*C_Min*. With a forward source the polish table is taken TWICE, with the forward rows and',
         'without, which is what says whether the later bucket is pinned by them or by nothing.'
         ]
    )

    #: The three structural numbers the PRICE FACTOR declares, read here rather than re-spelt so
    #: the calibrator's bound and the factor's own load assertion cannot disagree.
    FACTOR_DEFAULTS = {field.name: field.default
                       for field in riskfactors.LogVar2FJModelParameters.fields}

    market_factor_type = 'LogVar2FJModelPrices'
    #: The `Price Factors` type this family writes, which is what a `Bootstrapper
    #: Configuration` entry names it by - here, its own class name.
    price_factor_type = 'LogVar2FJModelParameters'

    identification_note = ('the forward-smile targets: the later buckets of Rho_S and Beta are '
                           'identified by them and by nothing else')

    #: The shared block plus this family's own. Three ladder dials are REDECLARED rather than
    #: added: the emitted store keys by name, so a name declared twice loses a descriptor.
    #: `European_Options` stays LAST, as it is there.
    fields = [field for field in OptionQuoteFamily.fields[:-1] if field.key not in (
        'Wing_Expiries', 'Wing_Pillars', 'Minimum_Contracts')] + [
                 F('Wing_Expiries', 'Text', default='1,3,6,12',
                   description='The expiries each delta pillar is quoted at on both wings, in months; under '
                               'Fit_Mode Bootstrap, the calendar buckets Rho_S and Beta are fitted per'),
                 F('Wing_Pillars', 'Text', default='0.25,0.10',
                   description='The delta magnitudes each wing expiry is quoted at, each in (0, '
                               '0.5], 0.5 being the straddle'),
                 F('Minimum_Contracts', 'Integer', default=8,
                   description='Distinct (expiry, strike) contracts the ladder must survive snapping with'),
                 F('Fit_Mode', 'Text', default='Global', values=['Global', 'Bootstrap'],
                   description='Global fits one bucket of shape parameters to every wing quote jointly; '
                               'Bootstrap makes the wing expiries the calendar buckets and fits each '
                               'given the ones before it'),
                 F('Vanilla_Pricer', 'Text', default='Quadrature', values=['Walk', 'Quadrature'],
                   description='How a vanilla quote is priced inside the fit: Walk, the conditional Black '
                               'on the fixed draws path by path; Quadrature, the same law\'s surrogate on a '
                               'deterministic Gauss-Hermite grid, on one residual bucket only'),
                 F('Quadrature_Nodes', 'Text', default='24,16',
                   description='Gauss-Hermite nodes Vanilla_Pricer Quadrature spends on the clock and on '
                               'the mixer, comma separated'),
                 F('Paths', 'Integer', default=8192,
                   description='Paths the fixed antithetic draws carry, which sets the noise floor '
                               'under every fitted number'),
                 F('Random_Seed', 'Integer', default=1,
                   description='Seeds the fixed draws, the pseudo-random generator or the Sobol scramble'),
                 F('Sampling', 'Text', default='Pseudo', values=['Sobol', 'Pseudo'],
                   description='The stream the fixed draws come off: Sobol, a scrambled sequence '
                               'over two dimensions per internal step and one per block; Pseudo, '
                               'the pseudo-random generator'),
                 F('Kappa_L', 'Float', default=0.5,
                   description='Structural slow log-variance reversion speed, per year; never fitted'),
                 F('Kappa_S', 'Float', default=6.0,
                   description='Structural fast log-variance reversion speed, per year; never fitted'),
                 F('Cap_A', 'Float', default=FACTOR_DEFAULTS['Cap_A'],
                   description='Structural log-variance corner min(l+s, Cap_A), never below L(0) '
                               '+ 6*s_inf; omitted, the fit runs unbounded and writes that '
                               'level, and null writes none'),
                 F('C_Min', 'Float', default=FACTOR_DEFAULTS['C_Min'],
                   description='Floor on the idiosyncratic share c = 1 - Rho_S^2 - Rho_L^2, written onto '
                               'the factor and bounding Rho_S in the fit'),
                 F('Idiosyncratic_Share', 'Text', default='Refuse', values=['Refuse', 'Floor'],
                   description='What a bucket landing on the C_Min box does: Refuse names the bucket and '
                               'the c it wanted; Floor takes C_Min and says so'),
                 F('Stationary_Spread', 'Text', default='Refuse', values=['Refuse', 'Floor'],
                   description='What a stationary log-vol sd outside Log_Vol_Sd_Band does: Refuse names the '
                               'bucket, its sd and the pair; Floor takes the fit as it stands and says so'),
                 F('Residual_Law', 'Text', default='NIG', values=['NIG', 'Gaussian'],
                   description='The law of the part of a return leverage does not explain: NIG is the '
                               'model, Gaussian drops the mixer as a limit and test mode'),
                 F('Wing_Side', 'Text', default='Put', values=['Put', 'Both'],
                   description='Which wing Wing_Weight lifts: Put, which is what an index desk hedges, or '
                               'Both, which is how an FX smile is dealt'),
                 F('Wing_Weight', 'Float', default=1.0,
                   description='Multiplier on the vega weight of the wing quotes '
                               'Wing_Side names; 1.0 is off'),
                 F('Expiry_Weights', 'Text', default='',
                   description='Optional tenor:weight list (3m:2,1y:0.5) weighting the fit along the term '
                               'structure, each rung matched to the quoted maturity nearest the tenor'),
                 F('Param_Buckets', 'Table', default='null', row=Row([
                     F('Tenor', 'Float', description='Years from the base date the bucket STARTS at')]),
                   description='Calendar-time buckets the four levers are piecewise constant on; empty is '
                               'one bucket, and Fit_Mode Bootstrap uses the wing expiries instead'),
                 F('Event_Days', 'Text', default='',
                   description='Optional comma-separated dates each carrying a one-day L '
                               'segment, so the event day\'s variance is one number the '
                               'straddling ATM pillar re-solves around'),
                 F('Event_Variance_Prior', 'Float', default=1.0,
                   description='Multiplier on an event day\'s own diffusive variance over the segment '
                               'enclosing it; 1.0 is off'),
                 F('Leverage_Prior', 'Float', default='',
                   description='The Rho_S the soft leverage prior pulls towards, on the engine\'s axis; '
                               'blank reads a LogVar2FJ history\'s Rho_S, else the asset-class default'),
                 F('Leverage_Prior_SE', 'Float', default='',
                   description='The standard error of the declared Leverage_Prior, which weights its row; '
                               'blank takes the nominal Leverage_Prior_Weight'),
                 F('Leverage_Product_Prior', 'Float', default='',
                   description='The prior on the product rho_s*sigma_s; blank is a declared Leverage_Prior '
                               'times Sigma_S_Reference, else a history, else the class default'),
                 F('Leverage_Product_Prior_SE', 'Float', default='',
                   description='The standard error of the declared Leverage_Product_Prior; blank takes the '
                               'nominal Leverage_Prior_Weight over Sigma_S_Reference'),
                 F('Residual_Shape_Floor', 'Float', default=0.30,
                   description='Soft floor on the residual\'s shape alpha*delta_A at the '
                               'shortest calibrated expiry, 1 strongly non-Gaussian and 15 '
                               'nearly Gaussian; 0 is off'),
                 F('Stickiness_Band', 'Float', default=0.5,
                   description='The band, in vol points, a deal\'s forward-skew reserve is taken over '
                               'where the forward smile was not quoted'),
                 F('Model_Priors', 'Text', default='On', values=['On', 'Off'],
                   description='Whether the routine soft terms, the leverage prior and the '
                               'residual shape floor, are in force; Off is the vanilla-only '
                               'objective with its one-sided Rho_S box'),
                 F('Forward_Smile_Source', 'Text', default='None',
                   values=['None', 'Quotes', 'Reference'],
                   description='Where the forward-skew target comes from: Quotes, traded '
                               'forward-starts read off Forward_Smiles; Reference, a reference '
                               'model\'s forward smiles off the same table; None, no target and a '
                               'reserve line reported; Global mode only'),
                 F('Forward_Smiles', 'Table', default='null', row=Row([
                     F('T1', 'Float', description='Forward start, in years'),
                     F('Delta', 'Float', description='Tenor of the forward-start option, in years'),
                     F('Strike', 'Float', description='Strike as a fraction of S_T1'),
                     F('Target_Vol', 'Float', description='The forward-start implied vol to hit'),
                     F('Weight', 'Float', default=1.0,
                       description='Relative weight within the forward block')]),
                   description='The rows Forward_Smile_Source Quotes or Reference reads its targets from'),
                 F('Forward_Tenors', 'Text', default='6m:6m,1y:1y,1y:3m',
                   description='The (T1:Delta) windows the RESERVE LINE is reported on where the forward '
                               'block is off, each moved onto the two grid block ends nearest its own'),
                 F('Slow_Factor_Prior', 'Text', default='',
                   description='The rho_l,sigma_l pair held where no wing quote reaches '
                               'Slow_Horizon; blank takes a history, else the asset-class default, '
                               'signed by the Rho_S in force'),
                 F('Forward_Weight', 'Float', default=1.0 / 3.0,
                   description='The forward block\'s share of the total objective weight, the vanillas '
                               'carrying the rest'),
                 F('Bucket_Smoothness', 'Float', default=0.02,
                   description='Weight on the difference between adjacent buckets of any lever, over '
                               'the buckets fitted so far'),
                 F('Stationarity_Tol', 'Float', default=1e-4,
                   description='How far off stationarity a stage may stop, as ||J^T r|| over the free '
                               'coordinates on the weighted vol-space residual, before '
                               'Quote_Sensitivity is refused'),
                 F('Jacobian_Rcond', 'Float', default=1e-3,
                   description='Relative singular-value cutoff on the column-scaled Jacobian below which a '
                               'direction is flat, and at which the quote contraction pseudo-inverts it'),
                 F('Max_Iterations', 'Integer', default=150,
                   description='Evaluations least_squares may spend per stage (scipy\'s max_nfev)'),
                 F('Tolerance', 'Float', default=1e-8,
                   description='Convergence tolerance (scipy\'s ftol) on each stage\'s weighted vol-space '
                               'residual'),
                 F('Step_Tolerance', 'Float', default=1e-12,
                   description='Convergence tolerance (scipy\'s xtol) on each stage\'s step in the fitted '
                               'coordinates, relative to their size'),
                 F('Pillar_Tolerance', 'Float', default=1e-10,
                   description='The relative ATM miss each L segment\'s own Newton solve stops '
                               'at, at every outer iterate'),
                 F('Sigma_L_Bounds', 'Text', default='0.3,2.0',
                   description='Fitted box on Sigma_L, lower,upper'),
                 F('Rho_L_Bounds', 'Text', default='-0.6,0.0', description='Fitted box on Rho_L, lower,upper'),
                 F('Sigma_S_Bounds', 'Text', default='0.5,5.0',
                   description='Fitted box on Sigma_S, lower,upper'),
                 F('Alpha_Bounds', 'Text', default='-2.0,500.0',
                   description='Fitted box on Alpha in the UNCONSTRAINED coordinate LVFit.RAW names, '
                               'lower,upper'),
                 F('Beta_Bounds', 'Text', default='-3.0,3.0',
                   description='Fitted box on Beta in the UNCONSTRAINED coordinate LVFit.RAW names, '
                               'lower,upper'),
                 F('Stage_Horizons', 'Text', default='0.25,1.0',
                   description='Where the stages cut the ladder, in years: the wing\'s own '
                               'horizon, then the sub-year smile'),
                 F('Slow_Horizon', 'Float', default=1.5,
                   description='The shortest wing expiry, in years, that identifies the slow pair; with no '
                               'wing at or beyond it Rho_L/Sigma_L are pinned'),
                 F('Xi_Solve_Iterations', 'Integer', default=12,
                   description='Chord steps one xi pillar\'s Newton solve gets per round'),
                 F('Xi_Solve_Damping', 'Float', default=0.5,
                   description='The largest move in log-variance one xi pillar chord step may take'),
                 F('Cap_Headroom_Max', 'Float', default=1e-5,
                   description='The mass of path-days at or above the corner the factor will carry that '
                               'REFUSES the fit; nothing where the walk is unbounded'),
                 F('Log_Vol_Sd_Band', 'Text',
                   default='FxRate:0.2,0.9; EquityPrice:0.4,0.9; CommodityPrice:0.4,0.9; FuturesPrice:0.4,0.9',
                   description='The band the fitted stationary log-vol sd may sit in, as class:lower,upper '
                               'entries separated by semicolons or one lower,upper for every class'),
                 F('Atm_Miss_Max', 'Float', default=1e-4,
                   description='The relative ATM miss a pillar may still carry once the fit is '
                               'done, after Pillar_Tolerance'),
                 F('Prior_Strikes', 'Text', default='0.90,0.95,1.00,1.05,1.10',
                   description='The strikes the reserve line\'s reported forward windows are read at, as '
                               'fractions of S_T1, comma-separated'),
                 F('Psi_Strikes', 'Text', default='0.90,1.00,1.10',
                   description='The three strikes, as fractions of the forward, the spot and forward '
                               'smiles\' slope and butterfly are read at'),
                 F('C_Margin', 'Float', default=0.05,
                   description='How far inside C_Min and the conditioning-share floor the soft penalty '
                               'starts biting, as a margin on c'),
                 F('Soft_Penalty', 'Float', default=0.25,
                   description='What a full C_Margin violation costs in residual'),
                 F('Leverage_Prior_Weight', 'Float', default=0.02,
                   description='Weight on the leverage prior\'s Rho_S row where the prior carries no '
                               'standard error, and this over Sigma_S_Reference on its product row'),
                 F('Sigma_S_Reference', 'Float', default=2.4,
                   description='The vol-of-vol a declared Leverage_Prior is multiplied by to reach its '
                               'product prior, and Leverage_Prior_Weight divided by to weight that row'),
                 F('Shape_Penalty', 'Float', default=0.05,
                   description='What a full violation of Residual_Shape_Floor costs in residual, on the '
                               'relative row relu(1 - shape/floor)'),
                 F('Residual_Horizon', 'Float', default=0.25,
                   description='The shortest expiry, in years, that identifies the residual '
                               'pair, named in the report to say whether the ladder prices those '
                               'tails; it switches no row off'),
                 F('Alpha_Prior_Defaults', 'Text',
                   default='FxRate:44; EquityPrice:44; CommodityPrice:44; FuturesPrice:44',
                   description='The Alpha prior per asset class, as class:value entries separated '
                               'by semicolons, the fallback where neither the ladder nor a '
                               'history identifies Alpha'),
                 F('Alpha_Prior_Sd', 'Float', default=0.5,
                   description='The log-normal spread on Alpha_Prior_Defaults, and the largest log-unit '
                               'standard error at which a history\'s Alpha estimate takes the row'),
                 F('Residual_Skew_Share_Defaults', 'Text',
                   default='FxRate:-0.5; EquityPrice:-0.5; CommodityPrice:-0.5; FuturesPrice:-0.5',
                   description='The prior on the residual\'s skew share Beta/Alpha per asset class, as '
                               'class:value entries separated by semicolons'),
                 F('Residual_Skew_Share_Sd', 'Float', default=0.2,
                   description='The spread on Residual_Skew_Share_Defaults, and the largest delta-method '
                               'error at which a history\'s skew-share estimate takes the row'),
                 F('Contamination_Ratio', 'Float', default=2.0,
                   description='How many times the model\'s own c a history\'s clock share C_Eff may read '
                               'before its alpha^P is uninformative'),
                 F('Spot_Rung_Tolerance', 'Float', default=0.25,
                   description='How far the rung a forward tenor\'s spot smile is read at may sit from that '
                               'tenor, as a fraction of it, before the row is dropped'),
                 F('Slow_Factor_Prior_Defaults', 'Text',
                   default='FxRate:0.2,0.5; EquityPrice:0.4,1.0; CommodityPrice:0.4,1.0; '
                           'FuturesPrice:0.4,1.0',
                   description='The rho_l,sigma_l magnitudes per asset class, as class:pair entries '
                               'separated by semicolons, signed by the fitted fast leverage'),
                 F('Leverage_Prior_Defaults', 'Text',
                   default='FxRate:0.0; EquityPrice:-0.7; CommodityPrice:-0.7; FuturesPrice:-0.7',
                   description='The Rho_S prior per asset class on the engine\'s axis, as class:value '
                               'entries separated by semicolons'),
                 F('Leverage_Product_Defaults', 'Text',
                   default='FxRate:0.0; EquityPrice:-1.9; CommodityPrice:-1.9; FuturesPrice:-1.9',
                   description='The prior on the leverage product Rho_S*Sigma_S per asset class, as '
                               'class:value entries separated by semicolons'),
                 F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
                   description='Keep the written parameters connected to the numbers quoted, so a '
                               'calculation\'s backward pass reports dV/dq beside dV/dtheta')
             ] + OptionQuoteFamily.fields[-1:]

    def __init__(self, param, device, dtype):
        # the constructed dtype is ignored (see `prec`); the DEVICE is the job's, the walk being
        # bandwidth-bound since `utils.LogVar2FJ.ou_path` and 25x cheaper on a card
        super(LogVar2FJModelParameters, self).__init__(param, device, dtype)
        # refuse a malformed hyperparameter BEFORE a single quote is read, not deep inside the
        # first fit that happens to touch it - a quote overriding one is checked again there
        where = 'Bootstrapper Configuration ' + type(self).__name__
        utils.LogVar2FJ.retired(where, self.param)
        utils.LogVar2FJ.typed(where, self.param, self.factor_types['Underlying'])
        #: What `Quote_Sensitivity` leaves behind: every fitted parameter still connected to its
        #: quotes, keyed as `_build_factor_state` mints its leaf, plus the quote leaf per block.
        #: `Config.bootstrap` harvests both - tensors cannot live in `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices,
                  calendars):
        """Calibrates the LogVar2FJ parameters and writes a `LogVar2FJModelParameters` price
        factor.

        The quote preparation is the plain family's, quote for quote. The quotes then SPLIT into an
        ATM ladder the inner bootstrap consumes one pillar at a time - the ATM quote at an expiry
        being the one nearest its own forward, so a hand-authored block reads as an emitted one
        does - and the whole set, which every fitted stage is judged on. A previously written
        factor warm starts every scalar, every lever and the curve. `LVFit` owns the state and the
        verbs; what is here is the quotes, the grid they are priced on, and the order the stages
        run in.
        """
        for market_price, implied_params in market_prices.items():
            fit = self.fit_for(market_price, implied_params, sys_params, price_models,
                               price_factors, factor_interp)
            if fit is None:
                continue
            param_name = self.factor_name(market_price)

            connect = fit.instrument['Quote_Sensitivity'] == 'Yes'
            if connect and fit.mode == 'Bootstrap':
                raise ValueError(
                    '{}: Quote_Sensitivity is not available under Fit_Mode Bootstrap. Bucket k is '
                    'fitted to expiry k\'s wings GIVEN the buckets before it, so theta* is a '
                    'stationary point of no single objective and the Gauss-Newton contraction '
                    'would report the LAST bucket\'s quote derivative as the whole surface\'s. '
                    'Fit Global, which carries it, or read the risk on the parameters'.format(
                        market_price))
            if connect:
                # the FIT's own device and not the family's: a quadrature fit with no forward
                # block runs on the host, and `market` splices this leaf onto premia built there
                fit.leaf = torch.tensor(
                    [quote.quoted for quote in fit.quotes], device=fit.device, dtype=self.prec,
                    requires_grad=True)

            theta = utils.LeastSquaresSolve.apply(fit, fit.rcond, fit.stationarity, fit.leaf)
            if connect and fit.capped:
                raise ValueError(
                    '{}: stage {} stopped CAPPED at Max_Iterations={}, so theta* is where the '
                    'evaluation budget ran out and not where the objective is stationary - the '
                    'Gauss-Newton contraction has no fixed point to be taken at. Raise '
                    'Max_Iterations, loosen Tolerance, or read the risk on the parameters'.format(
                        market_price, fit.capped, fit.max_iter))
            fit.finish(theta)
            fit.verify()
            # THE DECLARED WINDOW, off the factor this fit just wrote: its own grid ends where the
            # ladder does, so the rows `finish` read are the nearest block ends and not the tenor
            price_factors[param_name] = fit.written()
            declared = self.declared_skew_rows(
                market_price, implied_params, sys_params, price_models, price_factors,
                factor_interp)
            fit.skew_rows, fit.skew_declared = declared or fit.skew_rows, bool(declared)
            price_factors[param_name]['Skew_Gradient'] = fit.skew_line()
            LVReport(fit).log()

            if connect:
                factor = utils.Factor(self.price_factor_type,
                                      utils.check_rate_name(market_price)[1:])
                self.calibrated.update({
                    utils.Factor(factor.type, factor.name + (name,)): value
                    for name, value in fit.connect().items()})
                self.quote_leaves[market_price] = (fit.descriptors, fit.leaf)

    def factor_name(self, market_price):
        """The `Price Factors` key one block's fit reads its warm start off and writes back to."""
        return utils.check_tuple_name(
            utils.Factor(self.price_factor_type, utils.check_rate_name(market_price)[1:]))

    def fit_for(self, market_price, implied_params, sys_params, price_models, price_factors,
                factor_interp, declared=None):
        """One block's prepared `LVFit`, warm started off the factor already written for it - the
        quotes, the grid and the seed, with no stage run. None where the block belongs to another
        family or carries nothing to fit.

        The quote's own instrument wins over the `Bootstrapper Configuration` block, and
        `declared` over both: a reading of a written factor says what it wants priced without
        editing the document.
        """
        if utils.check_rate_name(market_price)[0] != self.market_factor_type:
            return None
        instrument = {**self.param, **implied_params['instrument'], **(declared or {})}
        factors, spot = self.resolve_block(
            market_price, instrument, price_factors, factor_interp, sys_params)
        fit = LVFit(self, market_price, instrument, factors,
                    price_factors.get(self.factor_name(market_price)))
        fit.history = self.slow_history(market_price, instrument, price_models)
        return fit if self.prepare(fit, sys_params, factors, spot) else None

    def declared_skew_rows(self, market_price, implied_params, sys_params, price_models,
                           price_factors, factor_interp):
        """The reserve line's model half at the windows `Forward_Tenors` DECLARES, read off the
        factor just written on a grid built to carry them exactly - one forward pass, no stage run,
        the same route `forward_smiles` takes. Empty where this ladder reaches no declared window,
        and the fit's own block-end rows stand."""
        instrument = dict(self.param, **implied_params['instrument'])
        strikes = utils.LogVar2FJ.parse_floats(instrument['Psi_Strikes'], 'Psi_Strikes', 3)
        fit = self.fit_for(
            market_price, implied_params, sys_params, price_models, price_factors, factor_interp,
            {'Forward_Smile_Source': 'Reference', 'Forward_Smiles': [
                {'T1': T1, 'Delta': delta, 'Strike': k, 'Target_Vol': 0.0}
                for T1, delta in ([utils.LogVar2FJ.tenor(x) for x in pair.split(':')]
                                  for pair in str(instrument['Forward_Tenors']).split(','))
                for k in strikes]})
        return {} if fit is None or not fit.targets else fit.skew_gradient()

    def forward_smiles(self, sys_params, price_models, price_factors, factor_interp, market_prices,
                       rows=None):
        """`{market price: {(T1, Delta): {strike: vol}}}` - the forward-start implied vols the
        factor `Price Factors` already carries gives, priced by the fit's own forward-start
        machinery at that factor's parameters and no stage run.

        `rows` are `Forward_Smiles` rows; without them the windows `Forward_Tenors` names are read
        at `Psi_Strikes`, which is the smile a written factor is asked for. A window this ladder
        cannot reach is dropped by name, exactly as a fitted one is. The block's own
        `Forward_Smile_Source` picks the INSTRUMENT - a traded forward-start under `Quotes`, the
        ratio expectation under `Reference`, which is what a block declaring neither is read as.
        """
        smiles = {}
        for market_price, implied_params in market_prices.items():
            instrument = dict(self.param, **implied_params['instrument'])
            windows = [[utils.LogVar2FJ.tenor(x) for x in pair.split(':')]
                       for pair in str(instrument['Forward_Tenors']).split(',')]
            table = rows or [
                {'T1': T1, 'Delta': delta, 'Strike': k, 'Target_Vol': 0.0}
                for T1, delta in windows
                for k in utils.LogVar2FJ.parse_floats(instrument['Psi_Strikes'], 'Psi_Strikes', 3)]
            source = instrument['Forward_Smile_Source']
            fit = self.fit_for(
                market_price, implied_params, sys_params, price_models, price_factors,
                factor_interp, {'Forward_Smiles': table, 'Forward_Smile_Source':
                    'Reference' if source == 'None' else source})
            if fit is not None:
                smiles[market_price] = {
                    window: {strike: model for strike, (model, _) in smile.items()}
                    for window, smile in fit.forward_smile(fit.evaluate()[2]).items()}
        return smiles

    def prepare(self, fit, sys_params, factors, spot):
        """The quotes, the buckets and the grid the whole fit is priced on - everything `LVFit`
        needs before a stage runs. False where the block carries nothing to fit.

        THE GRID IS THE QUOTES' OWN: a block ends at every quoted maturity and at every forward
        window's two ends, its steps one trading day except the last, a STUB landing the
        block exactly on T. So the variance the fit reads at a maturity is the variance a pricer
        reads at that tenor of the curve written out, to the digit.
        """
        quotes = [LVQuote(
            row=None, j=None, spot=spot, strike=strike, ratio=fit.tensor(strike / spot),
            is_call=sign > 0, units=option['Units'], T=t, rate=r, carry=(r - q) * t,
            forward=forward, premium=premium, weight=float(option['Weight']), sigma=sigma,
            quoted=sigma if fit.is_vol else option['Quoted_Market_Value'],
            vega=option['Units'] * self.fx_black_vega(forward, strike, r, sigma, t))
            for option, t, r, q, forward, sign, strike, sigma, premium
            in self.prepare_quotes(sys_params, fit.instrument, factors, spot, fit.market_price)]
        dead = [quote for quote in quotes if not quote.vega > 0.0]
        if dead:
            logging.warning(
                '{}: {} of {} quotes carry a Black vega of zero at their own quoted vol and are '
                'DROPPED - the residual is a premium miss over that vega, so a contract nothing '
                'prices has no vol-space reading. Strikes: {}'.format(
                    fit.market_price, len(dead), len(quotes),
                    ', '.join('{:.4g} at {:.3f}y'.format(x.strike, x.T) for x in dead)))
            quotes = [quote for quote in quotes if quote.vega > 0.0]
        if not quotes:
            logging.error('{} carries no quotes - nothing to bootstrap'.format(fit.market_price))
            return False

        # the split, by POSITION - two quotes at one strike and expiry compare equal as tuples:
        # one ATM per distinct expiry (nearest its own forward), the rest that expiry's wings
        by_expiry = {}
        for i, quote in enumerate(quotes):
            by_expiry.setdefault(quote.T, []).append(i)
        atm = [min(by_expiry[T], key=lambda i: abs(quotes[i].strike / quotes[i].forward - 1.0))
               for T in sorted(by_expiry)]
        wings = {T: [i for i in by_expiry[T] if i not in set(atm)]
                 for T in sorted(by_expiry) if len(by_expiry[T]) > 1}
        fit.buckets = self.param_buckets(fit, wings)
        if fit.quadrature and fit.buckets.size > 1:
            raise ValueError(
                '{}: Vanilla_Pricer Quadrature prices the residual on ONE inverse-Gaussian mixer '
                'over the whole clock to a maturity, and this block carries {} residual buckets '
                '({}) - a sum of mixers at different Alpha and Beta is not inverse Gaussian and '
                'has no quantile to integrate against. Declare one Param_Buckets row, or price '
                'the vanillas on the Walk'.format(
                    fit.market_price, fit.buckets.size,
                    ', '.join('{:g}y'.format(t) for t in fit.buckets)))
        fit.free = {}

        quotes = self.emphasis(fit, quotes, set(atm))
        rungs = {T: [quotes[i] for i in rows] for T, rows in by_expiry.items()}
        targets = self.reachable(fit, self.forward_rows(fit, factors), rungs)
        # EVERY BUCKET KNOT IS A BLOCK END, so a block's residual sits in one bucket and takes
        # exactly one mixer - the no-knot-inside-a-clock rule, written into the grid
        ends = sorted(set(by_expiry) | {float(t) for t in fit.buckets if t > 0.0}
                      | {t for target in targets
                         for t in (target.T1, target.T1 + target.tenor)})
        at = {T: j for j, T in enumerate(ends)}
        # the reserve line is owed wherever the forward smile was NOT fitted, so a block that is
        # off still reports its rows - on the ends the grid already has, which is what leaves the
        # vanilla-only fit the vanilla-only fit
        fit.reported = [] if targets else self.reported_rows(fit, factors, at)
        # every forward tenor's own spot rung: the quoted maturity nearest it, which is where the
        # model's spot smile is read for the two ratios and for the target DIFFERENCES
        near = {target.tenor: min(rungs, key=lambda T: abs(T - target.tenor))
                for target in targets + fit.reported}
        fit.spot_rung = {d: (at[T], rungs[T][0].carry, T) for d, T in near.items()}
        fit.tilt = self.tilts(fit, targets, rungs)
        spans = np.diff([0.0] + ends)
        counts = np.maximum(np.round(spans / fit.delta), 1.0).astype(int)
        fit.upto = np.cumsum(counts)
        fit.n = int(fit.upto[-1])
        share = float(fit.instrument['Forward_Weight']) if targets else 0.0
        total = sum(quote.weight for quote in quotes)
        fit.quotes = [quote._replace(row=i, j=at[quote.T],
                                     weight=np.sqrt(quote.weight * (1.0 - share) / total))
                      for i, quote in enumerate(quotes)]
        fit.atm = [fit.quotes[i] for i in atm]
        fit.wings = {T: [fit.quotes[i] for i in rows] for T, rows in wings.items()}
        if targets:
            total = sum(target.weight for target in targets)
            targets = [target._replace(
                j1=at[target.T1], j2=at[target.T1 + target.tenor],
                weight=np.sqrt(target.weight * share / total)) for target in targets]
        fit.targets = targets

        fit.deltas = fit.vector(np.concatenate(
            [np.append(np.full(n - 1, fit.delta), span - (n - 1) * fit.delta)
             for n, span in zip(counts, spans)]))
        fit.times = torch.cat([fit.deltas.new_zeros(1), fit.deltas.cumsum(0)])
        #: the bucket grid and each grid time's segment, searched ONCE: every lever the walk
        #: reads is a leaf indexed with it
        fit.grid = utils.TermStructure(fit.buckets, device=fit.device)
        fit.step_bucket = fit.grid.index(fit.times)
        fit.draws = fit.draw(int(fit.instrument['Paths']), fit.n, len(ends),
                             int(fit.instrument['Random_Seed']))
        fit.knots = np.array([0.0] + [quote.T for quote in fit.atm])
        variance = np.array([quote.sigma ** 2 * quote.T for quote in fit.atm])
        fit.xi = np.diff(np.concatenate([[0.0], variance])) / np.diff(fit.knots)
        self.event_knots(fit, sys_params)
        self.seed(fit)
        return True

    def emphasis(self, fit, quotes, atm):
        """The desk's own emphasis on the vega weights (5.4.4): `Wing_Weight` on the wings
        `Wing_Side` names, and `Expiry_Weights`' tenor:weight list along the term structure, each
        rung matched to the quoted maturity nearest the tenor.

        Both are RELATIVE - the normalisation that follows divides them out - so the defaults
        (1.0 and blank) leave every weight the vega it was.
        """
        wing, side = float(fit.instrument['Wing_Weight']), fit.instrument['Wing_Side']
        expiries = sorted({quote.T for quote in quotes})
        by_expiry = {}
        for pair in str(fit.instrument['Expiry_Weights']).split(','):
            if pair.strip():
                tenor, value = pair.split(':')
                by_expiry[min(expiries, key=lambda T: abs(T - utils.LogVar2FJ.tenor(tenor)))] = float(value)
        if wing == 1.0 and not by_expiry:
            return quotes
        logging.info('  {}: the fit is weighted {} on the {} wing{} and {} along the term '
                     'structure'.format(
            fit.market_price, wing, side.lower(), '' if side == 'Put' else 's',
            ', '.join('{:g}y {:g}'.format(T, x) for T, x in sorted(by_expiry.items()))
            or 'evenly'))
        lifted = lambda i, quote: i not in atm and (
                side == 'Both' or quote.strike < quote.forward)
        return [quote._replace(weight=quote.weight * by_expiry.get(quote.T, 1.0)
                                      * (wing if lifted(i, quote) else 1.0))
                for i, quote in enumerate(quotes)]

    def param_buckets(self, fit, wings):
        """The calendar buckets the four levers are piecewise constant on: `Param_Buckets` in
        `Global` mode, and in `Bootstrap` mode the ladder's own WING EXPIRIES, a bucket starting
        where the previous expiry ended so an option to `E_k` reads buckets 0..k (5.4.1).

        A ladder whose wings all land on one expiry has no smile term structure in it and is
        REFUSED in that mode rather than fitted as one bucket under another name.
        """
        if fit.mode != 'Bootstrap':
            return np.array([0.0] + sorted(
                float(row['Tenor']) for row in fit.table('Param_Buckets')
                if float(row['Tenor']) > 0.0))
        if len(wings) < 2:
            raise ValueError(
                '{}: Fit_Mode Bootstrap makes the ladder\'s WING EXPIRIES the calendar buckets, '
                'and this ladder carries wing quotes at {} - a ladder whose wings collapse onto '
                'one expiry has no smile term structure to bootstrap through. Quote wings at more '
                'expiries (the emitter asks for {} at {} delta), or fit Global, where one bucket '
                'is the model'.format(
                    fit.market_price,
                    ', '.join('{:g}y'.format(T) for T in wings) or 'no expiry at all',
                    '/'.join('{:g}y'.format(T) for T in self.ladder.wings),
                    '/'.join('{:g}'.format(p) for p in self.ladder.pillars)))
        if fit.table('Param_Buckets'):
            logging.warning('{}: Param_Buckets is ignored under Fit_Mode Bootstrap - the buckets '
                            'are the ladder\'s wing expiries {}'.format(
                fit.market_price,
                ', '.join('{:g}y'.format(T) for T in wings)))
        return np.array([0.0] + list(wings)[:-1])

    def reachable(self, fit, targets, rungs):
        """The forward rows this ladder can be judged on, the rest DROPPED by name.

        `Delta_skew` is the forward slope less the SPOT slope at maturity `Delta`, read at the
        quoted rung nearest it - so a tenor whose nearest rung is a quarter of `Delta` away is
        differenced against a smile at the wrong maturity, and the default block reaching a year
        would put a 1y forward smile beside a three-week one on a three-week ladder. It also
        stops the grid walking to two years for a ladder that stops in three weeks.
        """
        far = {target.tenor for target in targets
               if abs(min(rungs, key=lambda T: abs(T - target.tenor)) - target.tenor)
               > fit.spot_rung_tol * target.tenor}
        if not far:
            return targets
        logging.warning(
            '{}: the forward tenor{} {} {} DROPPED - this ladder quotes {} and the objective reads the '
            'spot slope at the rung nearest Delta, which here is more than {:.0%} of Delta away, '
            'so the difference would be taken against a smile at the wrong maturity. Quote the '
            'expiry, or name Forward_Tenors this ladder reaches'.format(
                fit.market_price, '' if len(far) == 1 else 's',
                '/'.join('{:g}y'.format(d) for d in sorted(far)),
                'is' if len(far) == 1 else 'are',
                '/'.join('{:g}y'.format(T) for T in sorted(rungs)), fit.spot_rung_tol))
        kept = [target for target in targets if target.tenor not in far]
        if not kept:
            fit.source = 'None'
        return kept

    def forward_rows(self, fit, factors):
        """The forward-start rows, in the instrument `Forward_Smile_Source` names.

        `Quotes` and `Reference` read `Forward_Smiles`, both SOURCES and both fitted; there is no
        third form. A table with no source, or a source in `Bootstrap` mode, refuses by name.
        """
        rows, source = fit.table('Forward_Smiles'), fit.source
        if source != 'None' and fit.mode == 'Bootstrap':
            raise ValueError(
                '{}: Forward_Smile_Source {} is Global mode only. In Bootstrap mode '
                'forward smiles are CONSEQUENCES of the bucket term structure and psi is reported '
                'rather than targeted - the report prints it. Set Forward_Smile_Source to None, or '
                'fit Global'.format(fit.market_price, source))
        if rows and source == 'None':
            raise ValueError(
                '{}: Forward_Smiles carries {} rows and Forward_Smile_Source is None, so nothing '
                'would be measured against them. Say what they are: Quotes (traded forward-starts, '
                'priced under the share measure), or Reference (a reference model\'s forward '
                'smiles, priced as the ratio expectation)'.format(fit.market_price, len(rows)))
        if not rows and source in ('Quotes', 'Reference'):
            raise ValueError(
                '{}: Forward_Smile_Source {} reads its targets from Forward_Smiles and the table '
                'is empty, so stage 5 would be skipped and a vanilla-only fit written under a '
                'block that asked for a forward target. Author the rows, or name None and take '
                'the reserve line the block being off reports'.format(fit.market_price, source))
        return [] if source == 'None' else self.forward_targets(fit, factors, rows)

    def reported_rows(self, fit, factors, at):
        """The forward windows a fit with NO block still REPORTS its reserve line on:
        each `Forward_Tenors` pair moved onto the two grid BLOCK ENDS nearest its own, the second
        strictly beyond the first.

        Moved rather than declared because a window end that is not a block end is not on the
        walk's grid at all, and putting one there splits a block into two mixers and moves theta*
        - so the reserve would be quoted off a different fit from the one written. These rows are
        therefore the window the LADDER has, which on a chain quoting 0.5y and 2.7y is not the
        6m-into-6m a desk asked for; what the FACTOR carries is `declared_skew_rows`' post-fit
        reading at the declared window, on a grid built for it once theta* is fixed.
        """
        windows, rows = set(), []
        for pair in str(fit.instrument['Forward_Tenors']).split(','):
            wanted = [utils.LogVar2FJ.tenor(x) for x in pair.split(':')]
            t1 = min(at, key=lambda T: abs(T - wanted[0]))
            beyond = [T for T in at if T > t1]
            if beyond:
                windows.add((t1, min(beyond, key=lambda T: abs(T - t1 - wanted[1]))))
        for t1, t2 in sorted(windows):
            rows += [{'T1': t1, 'Delta': t2 - t1, 'Strike': k, 'Target_Vol': 0.0}
                     for k in fit.prior_strikes]
        # by ROUNDED end rather than by sum: `T1 + tenor` is `t1 + (t2 - t1)` and need not be `t2`
        index = {round(T, 10): j for T, j in at.items()}
        return [target._replace(j1=index[round(target.T1, 10)],
                                j2=index[round(target.T1 + target.tenor, 10)])
                for target in self.forward_targets(fit, factors, rows)]

    def forward_targets(self, fit, factors, rows):
        """`LVForward` per row, each carrying its own window's carry off the two maturities the
        discount and yield curves read."""
        discount, carry = factors['Discount_Rate'], factors.get('Yield')
        funding = factors.get('Funding_Rate')
        targets = []
        for row in rows:
            t1, t2 = float(row['T1']), float(row['T1']) + float(row['Delta'])
            r1, r2 = float(discount.current_value(t1)), float(discount.current_value(t2))
            # the window's own carry, differenced off the two maturities the curves read
            window = ((r2 - self.effective_yield(r2, funding, carry, t2)) * t2
                      - (r1 - self.effective_yield(r1, funding, carry, t1)) * t1)
            targets.append(LVForward(
                j1=None, j2=None, T1=t1, tenor=t2 - t1, strike=float(row['Strike']),
                ratio=fit.tensor(float(row['Strike'])), carry=window,
                weight=float(row.get('Weight', 1.0)), target=float(row['Target_Vol'])))
        return targets

    def tilts(self, fit, targets, rungs):
        """`{(T1, Delta): (Delta_skew, Delta_bfly)}` in DECIMAL vol - the differences the objective
        targets, ONE pair per forward tenor whatever the source, so the residual is one term: the
        model's difference less the target's.

        Both sources MEASURE it - the source's own forward slope and butterfly at `psi_strikes`
        less the MARKET's own spot ones at the rung nearest Delta, read off the quoted vols in
        log-moneyness. Neither targets the forward ATM LEVEL: the ATM ladder pins it and the L
        strip reprices it exactly.
        """
        smiles = {}
        for target in targets:
            smiles.setdefault((target.T1, target.tenor), {})[
                round(target.strike, 10)] = target.target
        tilt = {}
        for (T1, tenor), smile in smiles.items():
            missing = [k for k in fit.psi_strikes if round(k, 10) not in smile]
            if missing:
                raise ValueError(
                    '{}: Forward_Smiles quotes {:g}y into {:g}y at strikes {} and the objective targets '
                    'the DIFFERENCES - the forward 90-110 slope and 90/110 butterfly less the spot '
                    'ones at that maturity - so the tenor needs {}. Quote {}'.format(
                        fit.market_price, T1, tenor,
                        '/'.join('{:g}'.format(k) for k in sorted(smile)),
                        '/'.join('{:g}'.format(k) for k in fit.psi_strikes),
                        ', '.join('{:g}'.format(k) for k in missing)))
            forward = fit.shape([smile[round(k, 10)] for k in fit.psi_strikes])
            spot = fit.market_shape(rungs[fit.spot_rung[tenor][2]])
            tilt[(T1, tenor)] = (forward[1] - spot[1], forward[2] - spot[2])
        return tilt

    def slow_history(self, market_price, instrument, price_models):
        """The historical estimate where the job's `Price Models` carries a `LogVar2FJCalibration`
        block for this underlying, read by the shape `utils.LogVar2FJ.SLOW_HISTORY` declares
        for both lanes - the slow pair, `Alpha`, `Beta` and `Rho_S`, each with its own standard
        error, because every one of them enters as a SOFT ROW weighted by that error and none of
        them is pinned. The estimator is the calibration's; this is its reader.

        A history is the FITTED AXIS's, so a block declaring `Priced_In` reads its own pair key and
        never the underlying priced in the base, which is a different rate."""
        model, keys = utils.LogVar2FJ.SLOW_HISTORY
        axis = utils.check_rate_name(instrument['Underlying']) + (
            (instrument['Priced_In'],) if instrument.get('Priced_In') else ())
        block = price_models.get(utils.check_tuple_name(utils.Factor(model, axis)))
        shape = [name for key in keys for name in (key, key + '_SE')]
        missing = [] if block is None else [
            name for name in shape if name not in block
                                      or (name.endswith('_SE') and not float(block[name]) > 0.0)]
        if missing:
            raise ValueError(
                '{}: Price Models carries {}.{}, which every soft prior row is read off, and it '
                'is missing or reads a non-positive {}. A historical prior is every estimate AND '
                'its POSITIVE standard error ({}), the error being what the row is weighted by - '
                're-run the calibration, or drop the block and the declared and class-default '
                'priors stand'.format(
                    market_price, model, '.'.join(axis), '/'.join(missing), '/'.join(shape)))
        return block

    def event_knots(self, fit, sys_params):
        """`Event_Days` as a one-INTERNAL-STEP SEGMENT each - a day at the default delta (spec
        5.4.3) - carrying `Event_Variance_Prior` times the enclosing segment's xi, the knot
        after it restoring that level. A day whose ATM segment is ALREADY one step is IDENTIFIED
        by its own pillar, so the prior is ignored there and the report says so. The variance on
        the event day is then one number and the ATM pillar straddling it gives that variance back
        over the days around it; a date with no straddling expiry carries the prior alone."""
        days = [x.strip() for x in str(fit.instrument['Event_Days']).split(',') if x.strip()]
        prior = float(fit.instrument['Event_Variance_Prior'])
        base, discount = sys_params['Base_Date'], fit.factors['Discount_Rate']
        marks = {}
        for day in days:
            t = discount.get_day_count_accrual(base, (pd.Timestamp(day) - base).days)
            if fit.own_segment(t):
                fit.identified_days.append(
                    'the event day {} is its OWN L segment - expiries on the business days either '
                    'side - so the ordinary pillar bootstrap sets its variance and '
                    'Event_Variance_Prior is IGNORED'.format(day))
                continue
            marks.setdefault(t + fit.delta, 0.0)  # the restore, unless a day is
            marks[t] = np.log(prior)  # already bumped there
        keep = sorted(t for t in marks
                      if t > 0.0 and np.min(np.abs(fit.knots - t)) > utils.BUCKET_TOL)
        fit.event_times = np.array(keep)
        fit.event_log = fit.vector([marks[t] for t in keep])
        priced = len(days) - len(fit.identified_days)
        if priced:
            logging.info('  {} event day{} carrying {:g}x the diffusive variance of the day '
                         'around them, as {} extra L knots'.format(
                priced, '' if priced == 1 else 's', prior, fit.event_times.size))

    def seed(self, fit):
        """STAGE 0, the structural half: the index-sized residual and leverage, and the
        xi strip at the market's own ATM^2 forward-variance increments - a near-identity seed, xi
        being the expected forward variance itself. The kappas and the cap width are read and never
        moved; a previous factor then replaces every seed but those.
        """
        read = fit.instrument
        n = fit.buckets.size
        fit.law = read['Residual_Law']
        fit.prior = LVPriors(fit.market_price, read, fit.history, fit.asset_class, fit.typed,
                             fit.box, fit.priors, [quote.weight for quote in fit.quotes],
                             tuple(fit.wings))
        # the SEED's sign is the prior's - a zero prior seeds the index number and the symmetric
        # box decides - and the residual seeds where the priors put it
        cold = fit.prior.cold()
        fit.state = {'Kappa_L': float(read['Kappa_L']), 'Kappa_S': float(read['Kappa_S']),
                     'Sigma_L': 1.0, 'Rho_L': -0.4,
                     'Cap_A': utils.LogVar2FJ.declared(read.get('Cap_A')),
                     'Rho_S': [cold['Rho_S']] * n, 'Beta': [cold['Beta']] * n,
                     'Sigma_S': [2.4] * n, 'Alpha': [cold['Alpha']] * n}

        previous, levels = fit.previous, None
        if previous:
            # before the warm start indexes it: a retired-era factor is missing every name this
            # model declares, and a KeyError says neither which key nor what replaced it
            utils.LogVar2FJ.retired('{}, the factor {} warm starts from'.format(
                self.factor_name(fit.market_price), fit.market_price), previous)
            law = str(previous.get('Residual_Law', 'NIG'))
            if law != fit.law:
                raise ValueError(
                    '{}: the factor this fit warm starts from declares Residual_Law {}, and the '
                    'block declares {}. Gaussian is a limit/test mode with no mixer, so the two '
                    'are different models and neither seeds the other. Declare the same law, or '
                    'drop the previous factor'.format(fit.market_price, law, fit.law))
            fit.state.update({name: float(previous[name]) for name in utils.LogVar2FJ.PARAM_NAMES
                              if not name.startswith('Kappa')})
            fit.state.update({name: utils.TermStructure(
                previous[name].array[:, 0], fit.vector(previous[name].array[:, 1])).at(
                fit.vector(fit.buckets)).tolist() for name in utils.LogVar2FJ.BUCKET_NAMES})
            levels = list(torch.log(utils.TermStructure(
                previous['Xi_Curve'].array[:, 0],
                fit.vector(previous['Xi_Curve'].array[:, 1])).at(
                fit.vector(fit.knots[:-1]))))
        if levels is None:
            levels = [fit.tensor(np.log(x)) for x in fit.xi]
        fit.levels, fit.warm = list(levels), list(levels)


class GBMAssetPriceTSModelParameters(Construction):
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
    #: The `Price Factors` type this family writes, which is what a `Bootstrapper
    #: Configuration` entry names it by - here, its own class name.
    price_factor_type = 'GBMAssetPriceTSModelParameters'
    factor_types = {'Asset_Price_Volatility': utils.TwoDimensionalFactors}
    #: What this family READS: the surface `Asset_Price_Volatility` names, whose ATM column is the
    #: integrated vol curve - and which `FXVolPrices` may have written in the same run.
    reads = tuple(utils.TwoDimensionalFactors)
    #: The precision the TAPE runs in - the value path is numpy and has no dtype to pick. Float64 on
    #: the CPU whatever the job asked for; `construct_bootstrapper`'s dtype does not reach it.
    dtype = torch.float64
    fields = [
        F('Asset_Price_Volatility', 'Text', default=REQUIRED,
          description='The vol surface whose ATM column becomes the integrated vol curve'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the integrated vol curve connected to the ATM vols it was built from, '
                      'so a calculation\'s backward pass reports dV/dq beside dV/dtheta')
    ]

    def __init__(self, param, device, dtype):
        super().__init__(param, device, dtype)
        #: What `Quote_Sensitivity` leaves behind: the integrated vol curve still connected to its
        #: ATM quotes, keyed as `_build_factor_state` mints its `Vol` leaf, plus the quote leaf per
        #: block. `Config.bootstrap` harvests both - tensors cannot live in `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}

    @staticmethod
    def atm_column(vol_factor, vol_surface, market_prices, price_factors):
        """The ATM vol per surface expiry, and where each number came from.

        Two sources, chosen by the surface's PROVENANCE. Where this market data carries an
        `FXVolPrices` block for the surface being integrated and that surface is the one the block
        wrote, its ATM rows ARE its ATM vols (`Factor2D.malz_skew` puts the +-0.5 label's vol at the
        delta-neutral straddle strike), so they are taken straight off it.

        Provenance is evidence, not a name: a hand-authored surface can sit under a name a quote
        block also uses. What is checked is the fingerprint `FXVolSurfaceParameters` leaves and
        `pinned_grid` reads back - the `Malz` subtype beside its `Grid_Tolerance`.

        Anything else is authored data, and the ATM column is `np.interp` at moneyness 1. Where the
        surface carries a node there, that read is the node itself.

        KNOWN LIMITATION: moneyness 1 is the ATM coordinate of a RATIO surface, while a `Malz`
        axis is log(F/K), whose ATM is at 0 - so a hand-authored Malz surface reads a wing.
        """
        family = FXVolSurfaceParameters
        quoted_name = utils.check_tuple_name(utils.Factor(
            family.market_factor_type, vol_factor.name))
        quoted = market_prices.get(quoted_name)
        written = price_factors.get(utils.check_tuple_name(vol_factor), {})
        if quoted is not None and written.get('Surface_Type') == family.surface_type and \
                'Grid_Tolerance' in written:
            atm = family.atm_quotes(family.used(quoted['instrument'], quoted_name))
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
        """The ATM column as the integrated vol curve the process reads - the value path.

        `V(t_i) = sigma_bar(t_i)^2 t_i` is the total variance the column implies, and the walk is
        Simpson's rule inverted for the instantaneous vol over each step,

            V(t_i) - V(t_{i-1}) = (dt/3)(sigma_{i-1}^2 + sigma_{i-1} sigma_i + sigma_i^2)

        a quadratic in `sigma_i` whose positive root is taken. Returns the curve and the expiries
        the repair fired at. The numpy walk is the only thing a mark is built from; `carried_vol`
        is the derivative twin.

        The map is piecewise and the switch is a KINK. A column implying a declining forward
        variance has no root, so `V(t_i)` is floored at what `sigma_i = 0` leaves and the written
        vol is that floor rather than the quote - `d/dq` is 1 on the smooth side and 0 on the
        floored one, in that column and every later one.

        Only `sigma_bar` is written; `sigma` is the walk's own state, sizing the next step's floor.
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
        """The same walk on a tape - a derivative carrier, never a value.

        It rides in as the splice `integrated_vol + (carried - carried.detach())`: exactly zero in
        the forward pass, derivative one, so the shipped curve is the numpy walk's bit for bit.

        The two walks stay separate because they do not agree to the bit: `torch.sqrt` is one ulp
        below `np.sqrt` on better than one float64 in a hundred here and re-associates the
        expression tree, which moved the shipped vols on 24% of 4000 random ATM columns.

        The discriminant is guarded, and only here. One repair leaves `sigma` exactly zero, so a
        second reaches a discriminant of exactly zero, where `sqrt` has an infinite derivative: the
        backward pass multiplies it by the zero `d(b^2)/db` and NaNs the whole Jacobian. The root
        there is zero, so it is written as zero and `sqrt` never sees the point.
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

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars):
        '''
        Turns the ATM column of the named vol surface into the integrated vol curve the risk neutral
        process reads, repairing any declining variance on the way - see `integrated_vol`.

        `Quote_Sensitivity` leaves that curve behind still connected to its ATM vols, so
        `Calculation.factor_leaf` can offer the connected tensor rather than minting a `Vol` leaf out
        of numpy. The map is explicit - no solve, no implicit function theorem - and the tape is a
        SPLICE over the shipped walk, so every number written comes out of `integrated_vol`.
        '''
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                # the quote's own instrument wins on conflict
                implied_params = dict(implied_params, instrument=dict(
                    self.param, **implied_params['instrument']))
                # get the vol surface
                vol_factor = resolve_factor(implied_params['instrument']['Asset_Price_Volatility'],
                                            price_factors, self.factor_types['Asset_Price_Volatility'])
                implied_param = vol_factor.name
                # whether the thing being MODELLED is an fx rate - the model is named after the
                # underlying, while the surface's tag names its own asset class
                is_fx = utils.check_tuple_name(utils.Factor('FxRate', rate[1:])) in price_factors

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                connect = implied_params['instrument']['Quote_Sensitivity'] == 'Yes'
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


class swaption_objective_class(namedtuple('swaption_objective', 'loss reduce reprice')):
    """What a risk-neutral swaption calibration minimises, as one record: the residual closure, the
    scalar the optimizer chain compares two candidates with, and the estimator that audits it.

    `loss(implied_var)` is `(model value per benchmark, residual per benchmark)`. The model value is
    a PREMIUM under either objective, so `SwaptionCalibration.solve`'s log is one thing; the residual
    is the objective's own and the two are not the same shape.

    `reduce(residuals)` exists because the two stages do not minimise the same function on the Monte
    Carlo path: `market_swap_class.error` returns a residual that is already a square, so basin
    hopping's `sum(r)` is the sum of squared pricing errors while `least_squares` sees a quartic. On
    the analytic path the residual is plain and `sum(r^2)` is both stages' objective. It takes a
    numpy array or a torch tensor.

    `reprice` is the Monte Carlo closure on a block that solved ANALYTICALLY, `None` otherwise - see
    `SwaptionCalibration.honesty_reprice`.

    Both objectives carry a quote side, the same leaf spliced onto two different residuals: the
    Monte Carlo one is already squared, so the dropped Gauss-Newton terms are each half what they
    correct and cancel ([Quote Sensitivities](quote_sensitivities.md#the-dropped-term)); the
    analytic one is separable, so its cross term is structurally zero.
    """


class SwaptionCalibration(utils.Residual):
    """One risk-neutral swaption calibration as an operand: the residual, and the solve over it.

    The residual is what `calc_loss_on_ir_curve` builds - one weighted error per
    `Instrument_Definitions` row, per the block's `Objective` - and the solve is the optimizer chain
    `calc_loss` hands over. Holding both beside the parameter dict they share lets
    `utils.LeastSquaresSolve` run the ordinary solve forward and differentiate the same residual backward;
    the frame - labels, box, the flat vector's split - is `utils.Residual`'s.

    The parameter vector is FLAT here and a dict everywhere else: scipy takes a vector, the process
    takes `{name: tensor}`, and the two scipy adapters own that boundary with `tn_var.data = ...`.
    `__call__` is the third crossing and the only differentiable one - it builds VIEWS of a flat
    tensor rather than writing `.data`, which would sever the edge the theorem needs.
    """

    def __init__(self, name, objective, implied_var, optimizers, process, market_swaps):
        self.name = name
        self.objective = objective
        self.implied_var = implied_var
        self.optimizers = optimizers
        self.process = process
        self.market_swaps = market_swaps
        self.keys = list(implied_var)
        self.sizes = [implied_var[key].numel() for key in self.keys]
        #: the fitted box, which `interior` reads its KKT active set off. A chain with no
        #: least-squares stage declares none, and an unbounded box holds nothing.
        box = next((optim[4] for optim in optimizers or () if optim[0] == 'leastsq'), None)
        self.edges = (np.array(box, dtype=float) if box is not None
                      else np.tile([[-np.inf], [np.inf]], sum(self.sizes)))

    @property
    def quotes(self):
        """The quote leaf per benchmark, or `()` where the block asked for no `Quote_Sensitivity` -
        which is what makes the wrapper a pass-through with no edge recorded."""
        return tuple(swap.quote for swap in self.market_swaps.values() if swap.quote is not None)

    @property
    def descriptors(self):
        """The benchmark names of `quotes`, in its order - what `quote_leaves` pairs them with."""
        return [name for name, swap in self.market_swaps.items() if swap.quote is not None]

    def __call__(self, x):
        """The residual vector at flat parameters `x`, differentiable in `x` and in the quotes.

        A fresh dict of views rather than the standing `implied_var`, whose `.data` the scipy
        adapters overwrite. `x` carries the closure's own precision, so the residual is the number
        the solve stopped on - the float64 promotion belongs to the linear algebra downstream.
        """
        return torch.stack(list(self.objective.loss(self.split(x))[1].values()))

    def honesty_reprice(self, theta):
        """What the engine's own estimator makes of an analytically-solved theta*, or `None`.

        The analytic objective fits Schrager-Pelsser normal vols, which freeze the annuity's
        weights, so such a block has never been asked what the Monte Carlo the rest of the library
        prices with makes of the answer. One pass at theta*, at the block's own path count, reports
        the worst benchmark's relative PREMIUM residual by name - the premium gap being what a mark
        moves by, and what carries the simulation's own numeraire error. No tolerance, no move.
        """
        if self.objective.reprice is None:
            return None
        # `.data` on a CLONE, not on the view: a leaf aliasing theta's storage would move with
        # anything that later wrote through theta
        for name, value in self.split(theta).items():
            self.implied_var[name].data = value.detach().clone()
        prices, _ = self.objective.reprice(self.implied_var)
        errors = {name: float(value.detach()) / self.market_swaps[name].price - 1.0
                  for name, value in prices.items()}
        worst = max(errors, key=lambda name: abs(errors[name]))
        return worst, errors[worst]

    def solve(self):
        """theta* as a flat tensor: the optimizer chain, run exactly as a bootstrap runs it.

        Basin hopping then least squares, `x0` chained from one to the next, and a candidate is
        accepted only if it beats the running best and the process it implies is well posed - so the
        answer can be the seed, which is what `utils.LeastSquaresSolve` checks stationarity for.

        The acceptance test compares one scalar across the seed and both stages, so that scalar is
        `objective.reduce` rather than a `sum` spelled three times. Which coordinates the box holds
        is `interior`'s reading at theta* and not a stage's report.
        """
        calibrated_swaptions, errors = self.objective.loss(self.implied_var)
        batch_loss = self.objective.reduce(
            torch.stack(list(errors.values()))).cpu().detach().numpy()
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
                    optim[2], x0=x0, take_step=optim[3], accept_test=optim[4],
                    T=optim[7], niter=optim[8],
                    minimizer_kwargs={"method": "L-BFGS-B", "jac": True, "bounds": optim[5]},
                    rng=optim[6])
                batch_loss = float(optim[2](result['x'])[0])
            elif optim[0] == 'leastsq':
                result = scipy.optimize.least_squares(
                    optim[2], x0=x0, jac=optim[3], bounds=optim[4])
                batch_loss = self.objective.reduce(optim[2](result['x']))

            if batch_loss < soln[0] and self.process.params_ok:
                sim_swaptions, errors = self.objective.loss(self.implied_var)
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

        theta = torch.tensor(np.concatenate([soln[1][key] for key in self.keys]),
                             dtype=self.implied_var[self.keys[0]].dtype,
                             device=self.implied_var[self.keys[0]].device)
        # ONE evaluation, to say which bounds bind - a calibration statement, not a sensitivity
        x = theta.detach().requires_grad_(True)
        residual = self(x)
        jacobian = torch.autograd.grad(
            residual, x, torch.eye(residual.numel(), dtype=x.dtype, device=x.device),
            is_grads_batched=True)[0].double()
        for line in utils.active_bounds(self.labels, theta.detach().cpu().numpy(), self.edges[0],
                                        self.edges[1],
                                        (jacobian.t() @ residual.detach().double()).cpu().numpy()):
            logging.info('{} - {}'.format(self.name, line))
        return theta


class RiskNeutralInterestRateModel(ImpliedCalibration):
    def __init__(self, param, device, dtype):
        super().__init__(param, device, dtype)
        #: The Monte Carlo sample shape of the last block built - a REPORT. Nothing prices off
        #: these: the residual closure captures its own shape as locals, because one bootstrapper
        #: runs every curve and a closure reaching through `self` would take the next block's count.
        self.batch_size = None
        self.num_batches = None
        #: What `Quote_Sensitivity` leaves behind: theta* still connected to its quotes, one entry
        #: per named model parameter, plus the quote leaf per block. `Config.bootstrap` harvests
        #: both - tensors cannot live in `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}

    def param_name(self, rate):
        """This family's own written factor for that curve - the name `implied_process` seeds off,
        `save_params` writes, and whose presence IS the warm start."""
        return utils.check_tuple_name(utils.Factor(type=self.__class__.__name__, name=rate[1:]))

    def calc_loss_on_ir_curve(self, implied_params, base_date, time_grid, process,
                              implied_obj, ir_factor, vol_surface, resid=lambda x: x * x, jac=False):
        """The swaption calibration's residual closure: implied parameters in, one weighted error
        per benchmark out.

        TWO OBJECTIVES, ONE DECLARED SWITCH, `Analytic` the default. It prices each benchmark with
        `stochasticprocess.HullWhite2FactorImpliedInterestRateModel.schrager_pelsser_swaption` and
        differences NORMAL VOLS, plain (`market_swap_class.normal_vol_error`). `Monte_Carlo` prices
        each through the engine's own `pv_float_cashflow_list` and differences the weighted relative
        pricing error, ALREADY SQUARED.

        Why the closed form is the default: it sits inside one Monte Carlo evaluation's own noise at
        22 of 25 benchmarks (the simulation's numeraire bias -0.35% to -1.61% exceeds the freezing
        bias -0.13 to +2.17bp), its residual is quadratic rather than quartic so `||J'r||` at theta*
        is 8.63e-7 against 3.16e2, it is deterministic in the sample, and it is 4.6x faster on the
        four-quote block. `Monte_Carlo` keeps being the engine's OWN estimator.

        The Monte Carlo closure is built either way: on an analytic block it is the auditor, and
        `SwaptionCalibration.honesty_reprice` runs it once at theta*.

        Both objectives price under the DOMESTIC measure, arranged upstream: `process` arrives built
        on a quanto-suppressed implied object, so `precalculate` assembles `K = 0`. See
        `implied_process` for the Girsanov argument.

        COMMON RANDOM NUMBERS ARE FROZEN PER SOLVE - the Sobol engine is built once and `reset`
        re-seeds nothing once `t_random_batch` exists, so the optimizer differences the parameters
        rather than the sample. `clear` is the memo half alone, all the analytic path needs. The
        sample shape is frozen as LOCALS: this residual outlives its block, `utils.LeastSquaresSolve`
        holding it for a backward that runs after the loop.

        The batch loop clears `t_Buffer` and not `t_PreCalc`. `calc_time_grid_curve_rate` keys on
        the curve code and time grid rather than the batch, so clearing only outside the loop makes
        every later batch re-read batch zero's curve. `t_PreCalc` holds integrals in theta rather
        than in the sample, so clearing it per batch would re-integrate the same numbers.

        THE QUOTE SIDE severs at the market price and nowhere else, on either objective: `swap.price`
        is a numpy scalar, and the splice that closes it sits on `market_swap_class.error` or on
        `market_swap_class.market_normal_vol`. One quote leaf per benchmark serves both. Two
        severances stay open deliberately, their upstream being the calibrated curve rather than a
        quote of THIS calibration: `get_par_swap_rate` and `set_fixed_amount`.
        """
        # completed by the family's declarations, because this closure is also built directly by
        # a gate off a hand-authored block; `bootstrap` hands it the section union already
        block = declared_defaults(type(self), implied_params['instrument'])
        objective = block['Objective']
        quote_sensitivity = block['Quote_Sensitivity']
        if objective not in ('Monte_Carlo', 'Analytic'):
            raise Exception(
                "Swaption calibration: Objective '{}' is not one this family prices - it is "
                "'Analytic' (the default: the Schrager-Pelsser normal vols) or 'Monte_Carlo' "
                "(every benchmark through the engine's own Monte Carlo). Correct the block's "
                "Objective to one of those two".format(objective))
        # the analytic residual is small tensors and a backward per evaluation, so it solves on
        # the host; the Monte Carlo one keeps the job's device, the faster at its declared sample
        device = torch.device('cpu') if objective == 'Analytic' else self.device
        # the closures below capture THESE locals, not the attributes they are mirrored onto
        batch_size = int(block['Simulations'])
        num_batches = int(block['Batches'])
        self.batch_size, self.num_batches = batch_size, num_batches

        def loss(implied_var):
            # first, reset the shared_mem
            shared_mem.reset(num_batches, numfactors, time_grid)
            # now set up the calc
            process.precalculate(base_date, time_grid, stoch_var, shared_mem, 0, implied_tensor=implied_var)
            tensor_swaptions = {}
            # needed to interpolate the zero curve
            delta_scen_t = np.diff(time_grid.scen_time_grid).reshape(-1, 1)

            for batch_index in range(num_batches):
                # the curve memo is keyed by curve and time and not by batch; `t_PreCalc` stays
                shared_mem.t_Buffer.clear()
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

            calibrated_swaptions = {k: v / (batch_size * num_batches) for k, v in tensor_swaptions.items()}
            errors = {k: swap.error(calibrated_swaptions[k], resid)
                      for k, swap in market_swaps.items()}
            return calibrated_swaptions, errors

        def analytic_loss(implied_var):
            """The same benchmarks, priced by Schrager-Pelsser, differenced as normal vols.

            `covariance` builds J and is the only thing this runs - no sample, no simulated curve,
            none of the simulation `precalculate` builds on it - so the two objectives share their
            front half: the same reversion-speed floors, series branches, `params_ok` and
            `Correlation` leaves.

            `clear` and not `reset`: the memo tables go per evaluation, the Sobol draw is not paid.
            """
            shared_mem.clear()
            process.covariance(base_date, time_grid, stoch_var, shared_mem, 0, implied_var)
            swaptions = dict(zip(market_swaps, stochasticprocess.hw2f_rows(
                process.schrager_pelsser_swaptions(schedules))))
            return ({name: swaption.premium for name, swaption in swaptions.items()},
                    {name: market_swaps[name].normal_vol_error(swaption)
                     for name, swaption in swaptions.items()})

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
        shared_mem = RiskNeutralInterestRate_State(index_keys, batch_size, device, self.prec)
        # the unit tensor switches the quote side on and puts its leaves on the right device
        market_swaps = utils.create_market_swaps(
            base_date, time_grid, curve_index, vol_surface, process.factor,
            block['Instrument_Definitions'],
            shared_mem.one if quote_sensitivity == 'Yes' else None,
            declared=block['Distribution_Type'])
        # number of random factors to use
        numfactors = process.num_factors()
        # compiled here rather than by a DealStructure, so they bind here
        for market_data in market_swaps.values():
            utils.bind_schedules(market_data.deal_data.Factor_dep, shared_mem.one)
        # one list for the life of the closure: the analytic swaptions build its legs once
        schedules = [market_data.schedule for market_data in market_swaps.values()]
        # set up the variables
        implied_var = {}
        stoch_var = torch.tensor(
            process.factor.current_value(), device=device, dtype=self.prec, requires_grad=jac)

        for param_name, param_value in implied_obj.current_value(include_quanto=jac).items():
            implied_var[param_name] = torch.tensor(
                param_value, dtype=self.prec, device=device, requires_grad=True)

        # `reduce` squares on the analytic path because the residual does not; `reprice` is the
        # Monte Carlo standing by as auditor. The switch is read once, here.
        chosen = swaption_objective_class(
            loss=analytic_loss, reduce=lambda r: (r * r).sum(), reprice=loss
        ) if objective == 'Analytic' else swaption_objective_class(
            loss=loss, reduce=lambda r: r.sum(), reprice=None)

        if jac:
            return stoch_var, implied_var, chosen.loss
        else:
            return implied_var, chosen, market_swaps

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars):
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
                # the quote's own instrument wins on conflict; every downstream read of
                # implied_params['instrument'] (including calc_loss) sees the union
                implied_params = dict(implied_params, instrument=dict(
                    self.param, **implied_params['instrument']))
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

                # this family's own written factor for this curve IS the warm start - there either
                # are parameters in the price factors or there are not
                warm = self.param_name(rate) in price_factors
                if warm:
                    logging.info(
                        '{} - warm start off {}: the basin search is skipped and the least squares '
                        'runs from that factor'.format(market_factor.name[0], self.param_name(rate)))

                # grab the implied process
                implied_obj, process, vol_tenors = self.implied_process(
                    base_currency, price_factors, price_models, ir_curve, rate,
                    vol_tenors=self.sigma_knots(implied_params['instrument']['Sigma_Knots']))

                # set up the time grid
                time_grid = utils.TimeGrid(mtm_dates, mtm_dates, mtm_dates)
                # add a delta of 10 days to the time_grid_years (without changing the scenario grid
                # this is needed for stochastically deflating the exposure later on
                time_grid.set_base_date(base_date, delta=(10, vol_tenors * utils.DayCount.DAYS_IN_YEAR))

                # calculate the error
                objective, optimizers, implied_var, market_swaptions = self.calc_loss(
                    implied_params, base_date, time_grid, process, implied_obj, ir_factor,
                    swaptionvol, warm)

                # check the time
                time_now = time.monotonic()
                calibration = SwaptionCalibration(
                    market_factor.name[0], objective, implied_var, optimizers, process,
                    market_swaptions)
                # through the implicit-function wrapper either way: with no quotes on the tape no
                # edge is recorded and the wrapper is a pass-through
                theta = utils.LeastSquaresSolve.apply(
                    calibration,
                    float(implied_params['instrument']['Jacobian_Rcond']),
                    float(implied_params['instrument']['Stationarity_Tol']),
                    *calibration.quotes)

                # reported by name rather than checked against a tolerance - see `honesty_reprice`
                reprice = calibration.honesty_reprice(theta)
                if reprice is not None:
                    logging.info(
                        '{} - Analytic objective - at theta* the engine\'s own Monte Carlo prices '
                        'its worst benchmark, {}, {:+.2f}% away from market'.format(
                            market_factor.name[0], reprice[0], 100.0 * reprice[1]))

                # save this - `unflatten` detaches, so `Price Factors` gets plain numpy
                self.save_params(calibration.unflatten(theta), price_factors, implied_obj, rate)

                # the connected half: one entry per named parameter, under the key
                # `_build_factor_state` mints its leaf with
                if calibration.quotes:
                    params_factor = utils.Factor(self.__class__.__name__, rate[1:])
                    self.calibrated.update({
                        utils.Factor(params_factor.type, params_factor.name + (name,)): value
                        for name, value in calibration.split(theta).items()})
                    # the chain called backward() per evaluation, so a `.grad` standing here is the
                    # sum over its whole path - the leaf is handed over clean
                    for quote in calibration.quotes:
                        quote.grad = None
                    self.quote_leaves[market_price] = (calibration.descriptors, calibration.quotes)

                # record the time
                logging.info('This took {} seconds.'.format(time.monotonic() - time_now))


class HullWhite2FactorModelParameters(RiskNeutralInterestRateModel):
    documentation = (
        'Interest Rates',
        ['The parameters $\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho$ are fitted to ATM swaption',
         'volatilities - swaptions rather than caplets, because a swaption sees both factors and so',
         'identifies $\\rho$. $\\sigma_1, \\sigma_2$ are piecewise constant on the `Sigma_Knots` term',
         'structure (ten knots from 0 to 10Y unless declared).',
         '',
         '`Objective` names what the solve minimises over the `Instrument_Definitions` rows.',
         '`Analytic`, the default, prices every benchmark with the Schrager-Pelsser closed form - the',
         'swap rate\'s diffusion frozen at the forward curve\'s loadings, so an ATM payer swaption is a',
         'normal (Bachelier) price in closed form - and differences normal vols:',
         '',
         '$$E=\\sum_{j\\in J} \\omega_j \\big(\\sigma^N_j(\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho)-\\sigma^N_j\\big)^2$$',
         '',
         'with $\\sigma^N_j$ the $j^{th}$ benchmark\'s market normal vol and $\\omega_j$ its weight.',
         '`Monte_Carlo` prices every benchmark through the engine\'s own paths and differences the',
         'squared relative premium error; it is the estimator the closed form was measured against and',
         'remains available as the oracle. Either residual carries its algorithmic Jacobian and is',
         'solved by `least_squares` on it; an analytic solve ends by repricing $\\theta^*$ through the',
         'Monte Carlo estimator and logging the worst benchmark\'s premium residual by name.',
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
         '',
         'That correction belongs to the SIMULATION and not to this fit. The market premium being',
         'repriced is $E^{Q_{dom}}[D_{dom}\\cdot\\text{payoff}]$ - struck, deflated and quoted in the',
         'rate currency - so the calibration prices it on domestic-measure paths whatever the base',
         'currency of the job, and $\\bar\\rho_1,\\bar\\rho_2$ are held at zero throughout the solve.',
         'Girsanov moves drifts and leaves quadratic variation alone, so the fitted',
         '$\\sigma_1,\\sigma_2,\\alpha_1,\\alpha_2,\\rho$ are the same numbers under either measure:',
         'they are calibrated domestically and $\\bar\\rho_1,\\bar\\rho_2$ are assembled from them',
         'above and written to the price factor, where a scenario run reads them.',
         ]
    )

    market_factor_type = 'HullWhite2FactorModelPrices'
    #: The `Price Factors` type this family writes, which is what a `Bootstrapper
    #: Configuration` entry names it by - here, its own class name.
    price_factor_type = 'HullWhite2FactorModelParameters'
    #: What this family READS: the curve its swaptions price off, the surface
    #: `Swaption_Volatility` names, and the quanto FX vol `implied_process` takes off the
    #: GBM family's own written block where the rate currency is not the base.
    reads = ('InterestRate', 'InterestYieldVol', 'GBMAssetPriceTSModelParameters')
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
              description='The quoted ATM vol, in the convention the surface declares: a lognormal '
                          'Black vol, or an absolute normal one where Distribution_Type is Normal'),
            F('Weight', 'Float', description='Relative weight in the objective')]),
          description='The forward starting swaps the swaptions are struck on'),
        F('Sigma_Knots', 'Table', default='null', row=Row([F('Tenor', 'Period')]),
          description='The knots of both sigma term structures, as periods from the base date; absent, '
                      'the ten at 0, 1M, 3M, 6M, 1Y, 2Y, 4Y, 6Y, 8Y and 10Y'),
        F('Objective', 'Text', default='Analytic', values=['Monte_Carlo', 'Analytic'],
          description='What the solve minimises: Analytic, normal-vol differences off the '
                      'Schrager-Pelsser closed form; Monte_Carlo, the squared relative premium '
                      'error priced on the engine\'s own paths'),
        F('Simulations', 'Integer', default=8192,
          description='Paths per batch the Monte Carlo objective prices its benchmarks on, from a '
                      'Sobol sample frozen for the whole solve'),
        F('Batches', 'Integer', default=1,
          description='How many batches of Simulations Sobol paths the sample holds, walked a '
                      'batch at a time so memory stays at one batch'),
        F('Sigma_Bounds', 'Text', default='1e-5,0.09',
          description='The box on every sigma knot of both term structures, lower,upper'),
        F('Alpha_Bounds', 'Text', default='-0.5,2.4',
          description='The box on both reversion speeds, lower,upper'),
        F('Alpha_Seed', 'Text', default='0.5,0.05',
          description='Where the two reversion speeds start on a cold block, alpha_1,alpha_2; a '
                      'block whose parameter factor exists starts off that'),
        F('Correlation_Bounds', 'Text', default='-0.95,0.95',
          description='The box on the correlation between the two factors, lower,upper - inside '
                      '+-1, where the two-factor covariance stays positive definite'),
        F('Basin_Step', 'Float', default=0.125,
          description='The basin-hopping step s: sigma and alpha move by the ratio exp(U(-s, s)) '
                      'and the correlation by U(-s, s), each clipped back into its box'),
        F('Basin_Temperature', 'Float', default=5.0,
          description='The Metropolis temperature the random search accepts an uphill candidate '
                      'at, in the units of the objective it is minimising'),
        F('Basin_Hops', 'Integer', default=50,
          description='How many basin hops, each one L-BFGS-B minimisation, the random search '
                      'takes before the least-squares stage is handed its x0'),
        F('Random_Seed', 'Integer', default=5120,
          description='Seeds the basin-hopping random search, its step taker and Metropolis accept '
                      'test alike; the Monte Carlo paths are frozen separately'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep each benchmark swaption connected to the quote it was '
                      'priced off, so the residual differentiates in the quote as well '
                      'as in the model parameters'),
        F('Jacobian_Rcond', 'Float', default=1e-5,
          description='Relative cutoff on the singular values of the column-scaled Jacobian the '
                      'backward pass pseudo-inverts; only used with Quote_Sensitivity Yes'),
        F('Stationarity_Tol', 'Float', default=1e-3,
          description='How far off stationarity theta* may be before the quote Jacobian is '
                      'refused, as the 2-norm of J\'r'),
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
              F('Day_Count', 'Text', default='ACT_365', values=list(DAY_COUNTS)),
              F('Index_Offset', 'Integer', default=0)],
          description='Unbuilt: the grid Generate_Instruments would sweep'),
        F('Quote_Timestamp', 'Date', default='',
          description='When the ladder was seen, the terminal\'s own as-of where the block was '
                      'authored off a screen; stored and reported, never read by the fit'),
        F('Quote_Source', 'Text', default='',
          description='How this block was authored, in one line: what the vols were read off and '
                      'the convention they are quoted in'),
        F('Distribution_Type', 'Text', default='', values=['', 'Lognormal', 'Normal'],
          description='The convention Market_Volatility is quoted in, checked against the named '
                      'surface\'s; blank leaves it unchecked and prices in the surface\'s own')
    ]

    def __init__(self, param, device, dtype):
        super(HullWhite2FactorModelParameters, self).__init__(param, device, dtype)
        #: the fitted box per coordinate, read once off the section: the basin step clips into it,
        #: the least-squares stage is bounded by it and `SwaptionCalibration.interior` reads its
        #: KKT active set off it
        self.sigma_bounds = utils.LogVar2FJ.parse_bounds(self.param['Sigma_Bounds'], 'Sigma_Bounds')
        self.alpha_bounds = utils.LogVar2FJ.parse_bounds(self.param['Alpha_Bounds'], 'Alpha_Bounds')
        self.alpha_seed = utils.LogVar2FJ.parse_floats(self.param['Alpha_Seed'], 'Alpha_Seed', 2)
        self.corr_bounds = utils.LogVar2FJ.parse_bounds(self.param['Correlation_Bounds'], 'Correlation_Bounds')

    def calc_loss(self, implied_params, base_date, time_grid, process, implied_obj, ir_factor,
                  vol_surface, warm=False):
        """The residual and the optimizer chain over it.

        `warm` is the ruling: a block whose parameter factor already exists is a warm start -
        `implied_process` seeded `x0` off it, so the random search that finds a basin is skipped and
        the least squares polishes from that seed alone. Cold, both stages run.
        """

        def split_param(x):
            corr = x[2:3]
            alpha = x[:2]
            sigmas = x[3:]
            return sigmas, alpha, corr

        def make_basin_callbacks(step, sigma_min_max, alpha_min_max, corr_min_max, rng):
            """The two callbacks basin hopping needs, drawing from `rng` and never from the process
            global - off `np.random` the calibration is a function of whatever ran before it in the
            same interpreter (theta* moves 0.93 absolute between ambient seeds on the gate fixture).
            The same generator serves the Metropolis test in `SwaptionCalibration.solve`."""

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

        def make_basin_hopping_loss(objective, implied_vars, device, with_grad=False):
            """The scipy basinhopper's scalar-and-gradient adapter over the residual closure.

            The scalar is `objective.reduce` and not a `sum` written here, because `solve`'s
            acceptance test compares it against the least-squares stage's own number.
            """
            loss_fn = objective.loss

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
                    total_loss = objective.reduce(torch.stack(list(error.values())))
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
            last = {}

            def calc_loss(x):
                for tn_var, np_var in zip(implied_vars.values(), np.split(x, split_param)):
                    tn_var.grad = None
                    tn_var.data = torch.from_numpy(np_var).to(device)
                _, error = loss_fn(implied_vars)
                last.update(x=x.copy(), residual=torch.stack(list(error.values())))
                return last['residual']

            def jacobian(x):
                # scipy asks for J where it last evaluated the residual, whose graph is kept for it
                loss = last['residual'] if np.array_equal(last.get('x'), x) else calc_loss(x)
                last.clear()
                return utils.vmapped_jacobian(loss, list(implied_vars.values())).cpu().numpy()

            def least_squares(x):
                return calc_loss(x).cpu().detach().numpy()

            len_vars = [len(x) for x in implied_vars.values()]
            split_param = np.cumsum(len_vars[:-1])
            return least_squares, jacobian

        # get the swaption error and market values
        implied_var_dict, objective, market_swaptions = self.calc_loss_on_ir_curve(
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
        # one generator for the whole random search - the step taker here, the Metropolis test in
        # `solve` - so the search is a function of `Random_Seed` alone
        block = declared_defaults(type(self), implied_params['instrument'])
        rng = np.random.default_rng(int(block['Random_Seed']))
        bounds_ok, make_step = make_basin_callbacks(
            float(block['Basin_Step']), self.sigma_bounds, self.alpha_bounds, self.corr_bounds,
            rng)

        # both adapters are the objective's, whichever the block declared - one `.data` boundary
        device = next(iter(implied_var_dict.values())).device
        basin_hopper_fn_grad = make_basin_hopping_loss(objective, implied_var_dict, device, True)
        x0 = torch.cat(list(implied_var_dict.values())).cpu().detach().numpy()
        lsq_fn, jacobian = make_least_squares_loss(objective.loss, implied_var_dict, device)

        lower, upper = var_to_bounds.T
        if not np.isfinite(x0).all():
            raise ValueError('HullWhite2FactorModelParameters initial parameters are not finite')
        x0 = np.clip(x0, lower, upper)
        x0 = np.where(x0 <= lower, np.nextafter(lower, upper), x0)
        x0 = np.where(x0 >= upper, np.nextafter(upper, lower), x0)

        basin = ('basin', x0, basin_hopper_fn_grad, make_step, bounds_ok, var_to_bounds,
                 rng, float(block['Basin_Temperature']), int(block['Basin_Hops']))
        leastsq = ('leastsq', x0, lsq_fn, jacobian, list(zip(*var_to_bounds)))

        return objective, [leastsq] if warm else [basin, leastsq], implied_var_dict, market_swaptions

    @staticmethod
    def sigma_knots(rows):
        """The declared `Sigma_Knots` as years (months / 12, the default grid's own arithmetic), or
        None for the default. A knot with a day part is refused: the grid is monthly.

        A LIST or nothing, which is `schema.quote_rows`' own reading: a block completed from its
        declarations carries the string `'null'` where a document carries rows.
        """
        if not isinstance(rows, list) or not rows:
            return None
        months = []
        for row in rows:
            offset = row[0] if isinstance(row, (list, tuple)) else row
            kwds = getattr(offset, 'kwds', {})
            if kwds.get('days') or kwds.get('weeks'):
                raise ValueError('Sigma_Knots takes whole months: {} has a day part'.format(offset))
            months.append(12 * kwds.get('years', 0) + kwds.get('months', 0))
        return np.array(sorted(set(months)), dtype=float) / 12.0

    def implied_process(self, base_currency, price_factors, price_models, ir_curve, rate,
                        vol_tenors=None):
        """The seed parameters and the process the objective prices through - two implied objects on
        a quanto'd curve, carrying the same numbers.

        CALIBRATE DOMESTICALLY, SIMULATE GLOBALLY. The market premium being repriced is
        $E^{Q_{dom}}[D_{dom}\\cdot(\\text{payoff})]$, struck and quoted in the RATE currency, so it
        is priced on domestic-measure paths whatever the job's base currency. Girsanov moves drifts
        and leaves quadratic variation alone, so $\\sigma_1,\\sigma_2,\\alpha_1,\\alpha_2,\\rho$ are
        the same numbers under either measure.

        So the two FX inputs `precalculate` builds $K$ out of are SUPPRESSED on the implied object
        the process is built on, taking that assembly down its base-currency branch to the bit, and
        left standing on the object returned, which is what `save_params` emits
        `Quanto_FX_Volatility` and `Quanto_FX_Correlation_1/2` off. All three consumers of the
        objective go through this one process. The simulator is untouched - `precalculate` still
        installs $K$ in a scenario run, being handed the emitted factor rather than this twin.

        The seed is asymmetric by ruling; `Alpha_Seed` says why. A block whose parameter factor
        already exists warm-starts off it instead, clipped to the declared bounds, and `calc_loss`
        then runs the least squares alone from that seed.
        """
        if vol_tenors is None:
            vol_tenors = np.array([0, 1, 3, 6, 12, 24, 48, 72, 96, 120]) / 12.0
        # construct an initial guess - need to read from params
        param_name = self.param_name(rate)

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
                 'Alpha_1': self.alpha_seed[0], 'Alpha_2': self.alpha_seed[1], 'Correlation': 0.01,
                 'Sigma_1': utils.Curve([], list(zip(vol_tenors, [0.01] * vol_tenors.size))),
                 'Sigma_2': utils.Curve([], list(zip(vol_tenors, [0.01] * vol_tenors.size)))})

        # the domestic twin: every invariant, minus the two FX inputs, so `precalculate` assembles
        # K = 0. The None survives `read_cache` because Factor1D.get_tenor normalizes it to a Curve
        domestic_obj = riskfactors.HullWhite2FactorModelParameters(
            dict(implied_obj.param, Quanto_FX_Volatility=None, short_rate_fx_correlation=None))

        # need to create a process and params as variables to pass to tf
        process = stochasticprocess.HullWhite2FactorImpliedInterestRateModel(
            ir_curve, {'Lambda_1': 0.0, 'Lambda_2': 0.0}, domestic_obj)

        return implied_obj, process, vol_tenors

    def save_params(self, vars, price_factors, implied_obj, rate):
        param_name = self.param_name(rate)
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


class Benchmark_State(utils.Calculation_State):
    """The pricing state a t0 benchmark valuation needs: one date, one path, float64.

    `t_Static_Buffer` is the point of it - every pricer reads a static curve from that dict, so a
    `requires_grad` tensor placed there is what puts the curve's nodes on the tape. It is built
    fresh per evaluation because `t_Buffer` is memoized by `(stoch, Factor)`, not by identity.

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

    A benchmark is a deal-tree NODE, `{'Instrument': deal, 'Children': [...]}`, as `Trade Data`
    authors it: a deposit or FRA is one deal, a par swap one `SwapInterestDeal`, an OIS swap a
    container over a compounded floating leg and a fixed one. Its PV is the sum of its leaves' PVs,
    each already in the reporting currency; no netting or collateral rule on top, which is what lets
    this stay out of `DealStructure`.

    **The graph audit.** The factor-construction path severs autograd in four places, every one on
    the way IN to `t_Static_Buffer`:

    - `Calculation._build_factor_state` and `Base_Revaluation.update_factors` mint every leaf off a
      numpy array. This class writes theta straight into the buffer and never calls `current_value`
      for a curve it is solving.
    - `riskfactors.Factor1D.current_value` is numpy end to end, and `Factor1D.get_tenor` REWRITES
      `param['Curve'].array` as a side effect of construction - so the node order theta is indexed
      by is the rewritten one, read back off the constructed factor.
    - `Factor1D.check_interpolation` precomputes the Hermite `(g, c)` pair from the numpy rate
      column. The pricing path does not use it: `utils.Interpolation.build` re-derives the pair from
      the buffer TENSOR, so a Hermite curve differentiates.
    - `utils.TensorSchedule.bind` mints the schedule's tensor half with `new_tensor`, which is where
      the QUOTE stops being differentiable. `_carry_quotes` builds the overlay that closes it.

    One trap that is not a severance: `utils.CurveTenor` caches its tenor grid as a tensor built
    from the first tensor that queries it. `all_tenors` is rebuilt per instance here, so a float64
    solve cannot inherit a float32 grid.

    `quotes` and `bumped_nodes` are the quote side: the quotes the set was authored at, in percent,
    and the same set authored one percent higher - the second says which schedule columns the quote
    writes (see `_carry_quotes`).
    """

    #: The solve is float64 whatever the simulation runs in: a bootstrap converging to 1e-10 cannot
    #: be done in float32, and the Jacobian is only as good as the residual it came from.
    dtype = torch.float64

    def __init__(self, nodes, price_factors, factor_interp, base_date, currency, calendars,
                 solve_for, device, quotes=None, bumped_nodes=None):
        # `config` imports from this module, so the package edge runs one way only
        from .config import Config

        cfg = Config(base_currency=currency)
        cfg.params['Price Factors'] = price_factors
        cfg.params['Price Factor Interpolation'] = factor_interp
        cfg.params['System Parameters']['Base_Date'] = base_date
        cfg.holidays = calendars
        cfg.set_calculation_children(nodes)
        # the engine's own discovery, so the set pulls exactly the factors a valuation would.
        # Single currency by construction, so every `calc_fx_cross` is the identity
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
        # compiled outside a calculation, so it binds its own schedules - and binds them LAST,
        # because the quote overlay is spliced into the copy `bind` makes
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

        Which columns those are is MEASURED: the same set authored one percent higher is compiled,
        and the columns that moved are the value columns with the difference as their slope. The
        authoring map is affine in the quote, so one bumped compile IS the derivative - which keeps
        each type's own `quoted` the only place a quotable instrument is declared.

        The splice is `base + (q - q.detach()) * slope`, exactly zero forward with derivative one,
        so enabling quote gradients cannot move the solve. It is a derivative carrier and NOT a
        reparameterisation - the pricers memoize payment tensors off the schedule, so a different
        quote needs a fresh closure.

        Resets carry no overlay - a reset value also leaves through `known_resets`, which reads
        numpy - and a moved reset column raises here.

        A quote that moves NO column raises too: an `FXForwardDeal` writes its outright into
        `Buy_Amount`, which `generate` reads as a float off the deal, so nothing of its compiled
        form moves and `dF/dq` would be a silent zero row. Being measured, the refusal stops firing
        on its own the day such a type grows a schedule.
        """
        for index, (legs, node) in enumerate(zip(self.benchmarks, bumped_nodes)):
            delta = self.quotes[index] - self.quotes[index].detach()
            # the bumped set never went through discovery, and discovery is what resets a deal
            bumped_legs = leaf_deals(node)
            for leaf in bumped_legs:
                leaf.reset(calendars)
            carried = 0
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
                        carried += columns.size
                        schedule.carry({int(column): self._column(schedule.schedule[:, column]) +
                                                     delta * self._column(moved[:, column]) for column in columns})
            if not carried:
                raise Exception(
                    'Quote_Sensitivity: benchmark {} ({}) writes its quote into no cashflow '
                    'schedule column, so the increment-1 overlay reaches nothing and dV/dq for it '
                    'would be reported as a silent zero rather than refused. An FXForwardDeal '
                    'quote lands in Buy_Amount, which the pricer reads as a float off the deal and '
                    'not off a schedule - the only seam the overlay carries. Leave '
                    'Quote_Sensitivity at No and Quote_Propagation at No on a block carrying such '
                    'a quote; the solved curve is identical either way.'.format(
                        ' + '.join(str(leg.Instrument.field.get('Reference', '?')) for leg in legs),
                        ' + '.join(type(leg.Instrument).__name__ for leg in legs)))

    def _column(self, values):
        return torch.tensor(values, dtype=self.dtype, device=self.one.device)

    def reads(self, theta):
        """The factors outside `solve_for` this residual actually reads, measured rather than
        declared - the coupling detector a multi-curve set is grouped by.

        Every constant is made a leaf and the residual differentiated once; what a backward pass
        reaches is what it reads. One residual and one backward, a fraction of a Newton iteration,
        and it catches the coupling a `Discount_Rate` field cannot state - what a benchmark PROJECTS
        off is authored inside its own deal block.

        A residual with no graph reads nothing, which is the self-discounting single-curve case.
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


def quote_nodes(points, discount_rate, shift=0.0):
    """The used quotes as deal-tree nodes, each authored at its own quote plus `shift` percent.

    Deep-copied because authoring WRITES the quote and the discount curve into the block.
    """
    nodes = []
    for point in points:
        authored = completed(dict(copy.deepcopy(point['Deal']), Object=point['DealType']))
        author_quote(authored, point['Quoted_Market_Value'] + shift, discount_rate)
        nodes.append(instruments.deal_node(authored, {}))
    return nodes


def author_quote(block, quote, discount_rate):
    """Author an instrument block AT its quote, discounting on `discount_rate`, legs included.

    What an instrument PROJECTS off it names itself; what the quote set DISCOUNTS on is a property
    of the curve set, stated once on the block. A type's `quote` runs before the deal is
    constructed and reads the block's own conventions - a deposit's wants the payment frequency -
    so the block is `completed` first. Where the number lands, and in what unit, is the type's own
    declaration (`Deal.quoted`, `Deal.quote`): a container declares none and its fixed leg does.
    """
    for _, leg in schema.walk([block]):
        leg['Discount_Rate'] = discount_rate
        declared = getattr(instruments, leg['Object'], None)
        if declared is not None and declared.quoted:
            declared.quote(leg, quote)


def par_quotes(points, discount_rate, currency, price_factors, factor_interp, base_date,
               calendars):
    """The quote at which each benchmark of `points` is worth exactly zero on `price_factors`.

    PV is affine in the quote, so the root is `PV(0) / (PV(0) - PV(1))` and not a search - in the
    unit each type reads its quote in, percent for a rate and the outright for an FX forward."""
    level = [dict(point, Quoted_Market_Value=0.0) for point in points]
    priced = [BenchmarkInstruments(
        quote_nodes(level, discount_rate, shift), price_factors, factor_interp, base_date, currency,
        calendars, [], torch.device('cpu'))({}).detach().cpu().numpy() for shift in (0.0, 1.0)]
    return priced[0] / (priced[0] - priced[1])


def knot_prices(curve, price_factors, factor_interp, base_date, calendars, conventions=(),
                fras_to=None):
    """`(Market Prices name, block)` quoting `curve` on its OWN knots, each benchmark at the quote
    it is worth exactly zero at on the curve as it stands - the bootstrap run backwards, which is
    how a book taken on with curves and no quotes gets market prices.

    One benchmark per knot, maturing on it under the knot rule (`quote_knots`), so solving the block
    returns the curve: a knot within one `Float_Frequency` of the base date is a deposit from it -
    a FRA from it where the block discounts on another curve, a deposit pinning its rate and so
    reading that curve alone - one whose float period starts before `fras_to` a FRA over that
    period, any other a swap from the base date. `conventions` states block fields over the
    family's declarations; the near split is the curve's own. A knot no whole day lands on refuses.
    """

    def period(value):
        return utils.parse_period(value) if isinstance(value, str) else value

    rate = utils.check_rate_name(curve)
    name = '.'.join(rate)
    factor = price_factors[utils.check_tuple_name(utils.Factor('InterestRate', rate))]
    stated = declared_defaults(InterestRateCurveParameters, dict(
        {'Currency': factor['Currency'], 'Day_Count': factor['Day_Count']}, **dict(conventions)))
    block = dict({key: stated[key] for key in (
        'Currency', 'Day_Count', 'Discount_Rate', 'Fixed_Day_Count', 'Float_Day_Count',
        'Front_Day_Count', 'Compounding')}, **{key: period(stated[key]) for key in (
        'Fixed_Frequency', 'Float_Frequency')})
    if factor.get('Near_Interpolation') and factor.get('Near_Date'):
        block['Near_Interpolation'] = riskfactors.factor_interp_map.get(
            factor['Near_Interpolation'], riskfactors.INTERPOLATION_DEFAULT)
        block['Near_Tenor'] = pd.DateOffset(days=(factor['Near_Date'] - base_date).days)

    code = utils.DayCount.code(block['Day_Count'])
    landed, stray = [], []
    for knot in factor['Curve'].array[:, 0]:
        day = min(range(int(knot * 360) - 1, int(knot * 366) + 2),
                  key=lambda day: abs(utils.DayCount.accrual(base_date, day, code) - knot))
        if day > 0 and abs(utils.DayCount.accrual(base_date, day, code) - knot) < 1e-9:
            landed.append(base_date + pd.Timedelta(days=day))
        else:
            stray.append('{:.12g}'.format(knot))
    if stray:
        raise ValueError('{}: no benchmark matures on the knot(s) {} - no whole day from the base '
                         'date {:%Y-%m-%d} accrues to them in {}'.format(
            name, ', '.join(stray), base_date, block['Day_Count']))

    fixed, floating = block['Fixed_Frequency'], block['Float_Frequency']
    fras_to, zero = period(fras_to), pd.DateOffset(months=0)
    foreign = block['Discount_Rate'] not in ('', name)
    points = []
    for end in landed:
        start, span = end - floating, pd.DateOffset(days=(end - base_date).days)
        terms = {'Reference': '{}_{:%Y%m%d}'.format(name, end), 'Currency': block['Currency'],
                 'Interest_Rate': name, 'Maturity_Date': end}
        front = end <= base_date + floating or start <= base_date
        if front and not foreign:
            kind, deal = 'DepositDeal', dict(
                terms, Effective_Date=base_date, Payment_Frequency=span, Interest_Frequency=span,
                Accrual_Day_Count=block['Front_Day_Count'], Amount=1e6,
                Interest_Rate_Schedule=utils.DateList({}))
        elif front or (fras_to is not None and start < base_date + fras_to):
            start = base_date if front else start
            kind, deal = 'FRADeal', dict(
                terms, Effective_Date=start, Reset_Date=start, Day_Count=block['Float_Day_Count'],
                Principal=1e6, FRA_Rate=0.0, Borrower_Lender='Borrower')
        else:
            kind, deal = 'SwapInterestDeal', dict(
                terms, Effective_Date=base_date, Pay_Rate_Type='Fixed', Pay_Frequency=fixed,
                Pay_Interest_Frequency=fixed, Pay_Day_Count=block['Fixed_Day_Count'],
                Receive_Frequency=floating, Receive_Interest_Frequency=zero,
                Receive_Day_Count=block['Float_Day_Count'], Index_Tenor=zero, Index_Frequency=zero,
                Index_Day_Count=block['Float_Day_Count'], Compounding_Method=block['Compounding'],
                Principal=1e6, Swap_Rate=0.0)
        points.append({'Use': 'Yes', 'Descriptor': '{} {} {:%Y-%m-%d}'.format(name, kind, end),
                       'DealType': kind, 'Quote_Type': 'Par_Rate', 'Quoted_Market_Value': 0.0,
                       'Deal': deal})

    for point, quote in zip(points, par_quotes(
            points, block['Discount_Rate'] or name, block['Currency'], price_factors, factor_interp,
            base_date, calendars)):
        point['Quoted_Market_Value'] = float(quote)
    return '{}.{}'.format(InterestRateCurveParameters.market_factor_type, name), {
        'instrument': dict(block, Points=points)}


class InterestRateCurveParameters(Construction):
    """A zero curve solved from deposit, FRA, swap and FX forward quotes, priced by the engine's
    own pricers.

    A quote is an instrument, a `Quote_Type` and a number - see the developer note on
    [Market Prices](../developer/market_prices.md). Each `Points` entry names an instrument type in
    `DealType` and carries a block of it in `Deal`, so the `Instrument` store's declarations ARE
    this family's quote schema. The family authors that block at its `Quoted_Market_Value`, and a
    fair benchmark prices to zero, so the solve is a root find on the t0 PV vector.

    THE BLOCK IS THE CURVE'S DEFINITION: beside the quotes it declares the conventions they were
    authored under - the calendar, the settlement lag, both legs' frequency and day count, whether
    the swap rows compound overnight - and the interpolation the solved curve carries, including a
    `Near_Interpolation` up to `Near_Tenor` where the near end is quoted in another instrument. Each
    row carries the `Tenor` it was authored from, so a strip re-rolls on a new date from the block.

    Two blocks make a multi-curve set - an OIS discount curve, then a projection curve discounting
    on it - and `Discount_Rate` is what orders them. A blank `Discount_Rate` discounts on the curve
    being built, the single-curve configuration and the harder solve.

    Unlike the other families this writes an `InterestRate` price factor rather than a
    `<ClassName>` parameter block, which is what `price_factor_type` declares.
    """
    market_factor_type = 'InterestRatePrices'
    #: The `Price Factors` type this family writes. The others write a block named for their own
    #: class, so the emitter recovers it; no rule recovers `InterestRate` from this class name.
    price_factor_type = 'InterestRate'
    #: What this family READS, off the benchmark deals its own discovery pulls: the curves a block
    #: forecasts and discounts on, and the reporting `FxRate` a cross-currency benchmark crosses at.
    #: `InterestRate` is what it also WRITES, which orders its blocks among themselves and is what
    #: `in_dependency_order` does; it carries no edge to another family.
    reads = ('InterestRate', 'FxRate')
    #: The instrument types a quote may be, each a declared `Instrument` type - so the quote's
    #: schema IS that type's declarations. `StructuredDeal` is how a two-leg benchmark is authored;
    #: `FXForwardDeal` crosses currencies, its quote being a forward OUTRIGHT held at par.
    quote_instruments = ('DepositDeal', 'FRADeal', 'SwapInterestDeal', 'StructuredDeal',
                         'FXForwardDeal')
    #: Block fields an artifact is NOT a function of - the lifecycle switches, read when one is
    #: published or ridden rather than when it is fitted. `plan_key` shadows them out so a knob
    #: governing the ride cannot also hide the artifact it governs. `Quote_Sensitivity` joins them
    #: because it provably moves neither theta* nor J.
    lifecycle_fields = ('Quote_Sensitivity', 'Quote_Propagation', 'Drift_Tolerance')
    fields = [
        F('Currency', 'Text', default=REQUIRED, description='The currency of the curve to build'),
        F('Day_Count', 'Text', default='ACT_365', values=list(DAY_COUNTS),
          description='Daycount the solved curve\'s tenors are expressed in'),
        F('Discount_Rate', 'Text', default='',
          description='The curve the quotes discount on; blank builds a self-discounting curve'),
        F('Calendar', 'Text', default='',
          description='The holiday calendar the benchmark dates were rolled against, named in the '
                      'job\'s calendar file; blank is Monday to Friday'),
        F('Spot_Days', 'Integer', default=0,
          description='Settlement lag in business days - where a spot-starting benchmark begins'),
        F('Fixed_Frequency', 'Period', default='3M',
          description='Coupon frequency of a swap benchmark\'s fixed leg'),
        F('Float_Frequency', 'Period', default='3M',
          description='Coupon frequency of a swap benchmark\'s floating leg'),
        F('Fixed_Day_Count', 'Text', default='ACT_365', values=list(DAY_COUNTS),
          description='Daycount a swap benchmark\'s fixed leg accrues on'),
        F('Float_Day_Count', 'Text', default='ACT_365', values=list(DAY_COUNTS),
          description='Daycount a swap benchmark\'s floating leg and its index accrue on'),
        F('Front_Day_Count', 'Text', default='ACT_365', values=list(DAY_COUNTS),
          description='Daycount the front deposit accrues on'),
        F('Compounding', 'Text', default='None', values=['None', 'OIS'],
          description='What the swap rows are: OIS marks an overnight-compounded benchmark'),
        F('Near_Interpolation', 'Text', default='',
          values=[''] + list(riskfactors.INTERPOLATION_METHODS),
          description='Interpolation the solved curve carries up to Near_Tenor, where the near end '
                      'is quoted in a different instrument; blank leaves one scheme over all of it'),
        F('Near_Tenor', 'Period', default='',
          description='Where the near interpolation stops, as a period from the base date'),
        F('N_Iter', 'Integer', default=50,
          description='Newton iteration cap'),
        F('Tol', 'Float', default=1e-13,
          description='Convergence tolerance on the Newton step, in rate space'),
        F('Damping_Halvings', 'Integer', default=6,
          description='How many times the line search may halve a Newton step before giving up'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the solved curve connected to its quotes, so a calculation\'s backward '
                      'pass reports dV/dq beside dV/dtheta'),
        F('Quote_Propagation', 'Text', default='No', values=['No', 'Linear'],
          description='How a quote that moves between bootstraps reaches the curve: No '
                      're-solves; Linear rides the calibration artifact published at the '
                      'bootstrap, theta* + dtheta/dq (q_now - q0)'),
        F('Drift_Tolerance', 'Float', default=1e-3,
          description='How far out of par, in percent of quote, a Linear ride may leave this '
                      'block\'s own benchmarks before a refit is asked for'),
        F('Points', 'Container', default={
            'Use': 'Yes', 'Deal': {}, 'Descriptor': '', 'DealType': 'DepositDeal',
            'Quote_Type': 'Par_Rate', 'Quoted_Market_Value': 0.0},
          sub_fields=[
              F('Use', 'Text', default='Yes', values=['Yes', 'No'],
                description='Whether this quote enters the solve'),
              F('Deal', 'Container', default={},
                description='The instrument itself, authored as a deal of type DealType'),
              F('Descriptor', 'Text', default='', description='Free text naming the quote'),
              F('Tenor', 'Text', default='',
                description='The label this benchmark was authored from, in the block\'s conventions '
                            '(ON, 3M, 1Mx4M for a FRA, 6M1M for a swap starting in six months)'),
              F('Security', 'Text', default='',
                description='The security the row is quoted off, where it came from a market data '
                            'source; stored and reported, never read by the solve'),
              F('DealType', 'Text', default='DepositDeal', values=list(quote_instruments),
                description='The instrument type the quote is a price for'),
              F('Quote_Type', 'Text', default='Par_Rate', values=['Par_Rate'],
                description='What Quoted_Market_Value is; the solve holds the instrument at par'),
              F('Quoted_Market_Value', 'Float',
                description='The quote, in the unit its DealType reads: percent for a rate '
                            'benchmark, the outright in Buy_Currency per unit of '
                            'Sell_Currency for an FXForwardDeal'),
              F('Quoted_Bid', 'Float',
                description='The bid side of this quote, in the same unit as the mid; optional '
                            'and never read by the solve'),
              F('Quoted_Ask', 'Float',
                description='The offer side, the pair of Quoted_Bid; optional '
                            'and never read by the solve'),
              F('Timestamp', 'Date', default='',
                description='When this quote was observed; stored and reported, never read '
                            'by the solve')],
          description='One market quote: an instrument, what kind of number is quoted, the number, '
                      'its two-way sides where the source printed them, and when it was seen')
    ]

    def __init__(self, param, device, dtype):
        super().__init__(param, device, dtype)
        # the solve is dispatch-bound - one small backward per quote per iteration - so it runs on
        # the host: three desk curves in 5 s there against 17 s on the card
        self.device = torch.device('cpu')
        #: What `Quote_Sensitivity` leaves behind: the solved nodes still connected to their quotes,
        #: per curve, plus the quote leaf per block; and what `Quote_Propagation` publishes, one
        #: artifact per coupled set. `Config.bootstrap` harvests all three - tensors cannot live in
        #: `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}
        self.published = []

    @staticmethod
    def quote_knots(nodes, base_date, day_count, calendars):
        """The curve's knot grid: one knot per benchmark, at that benchmark's last cashflow date.

        The only placement that makes the system square - a knot with no instrument maturing at it
        is unidentified, and two instruments between one pair of knots leave the curve
        under-determined. Below the shortest knot the curve is flat by `CurveTenor`'s clipping, so
        the front stub costs no unknown. The output grid IS this grid: interpolating onto a second
        would stop the curve repricing its quotes.

        Returned in NODE order and in the curve's own day count, so a caller can pair each knot
        with the quote that identifies it; the curve itself is sorted.
        """
        code = utils.DayCount.code(day_count)
        maturities = []
        for node in nodes:
            leaves = leaf_deals(node)
            for leaf in leaves:
                leaf.reset(calendars)
            maturities.append(max(max(leaf.get_reval_dates()) for leaf in leaves))
        return np.array([utils.DayCount.accrual(
            base_date, (maturity - base_date).days, code) for maturity in maturities])

    @staticmethod
    def benchmark_curves(block, market_price=None):
        """Every `InterestRate` curve this block's used benchmark deals NAME, read off each deal
        type's own `factor_fields` and recursing into `Children`.

        `Discount_Rate` orders the ordinary multi-curve case but cannot order a CROSS-CURRENCY
        benchmark: an `FXForwardDeal` names the other leg's curve in `Sell_Discount_Rate`, inside
        the deal, so a block with a blank `Discount_Rate` can still read a curve nobody has built.

        Read off the deal CLASS, because this runs before anything is seeded. Being a declaration
        read it is strictly weaker than `BenchmarkInstruments.reads`, which measures the same
        coupling but needs every curve to exist first.
        """
        return {'.'.join(utils.check_rate_name(deal[field]))
                for point in quote_table(block, market_price) if point.get('Use', 'Yes') == 'Yes'
                for path, deal in schema.walk([point['Deal']])
                for field, candidates in getattr(getattr(
                instruments, point['DealType'] if path == '0' else deal.get('Object', ''),
                None), 'factor_fields', {}).items()
                if 'InterestRate' in candidates and deal.get(field)}

    def in_dependency_order(self, market_prices):
        """This family's blocks, one that READS a curve another block BUILDS coming after it.

        A block reads a curve two ways and both order it: `Discount_Rate`, and what its benchmark
        deals NAME (`benchmark_curves`). A block naming its own curve is the self-discounting
        configuration and orders nothing. A cycle is refused by name here rather than as the bare
        `RuntimeError` the sort would raise.
        """
        blocks = {}
        for name, implied_params in market_prices.items():
            rate = utils.check_rate_name(name)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type == self.market_factor_type:
                # the quote's own instrument wins on conflict; everything downstream of here -
                # the dependency graph, the coupled sets, the solve and `publish` - sees the union
                blocks[name] = dict(implied_params, instrument=dict(
                    self.param, **implied_params['instrument']))
        # keyed by the curve name a `Discount_Rate` carries - the block's name without its type
        builds = {'.'.join(utils.check_rate_name(name)[1:]): name for name in blocks}
        graph = {}
        for name, implied_params in blocks.items():
            block = implied_params['instrument']
            reads = {block['Discount_Rate']} | self.benchmark_curves(block, name)
            graph[name] = sorted({builds[curve] for curve in reads
                                  if curve in builds and builds[curve] != name})
        # `topological_sort` deletes what it resolves, so what is left is exactly the cycle
        unresolved = dict(graph)
        try:
            order = utils.topological_sort(unresolved)
        except RuntimeError:
            raise Exception(
                'Curve bootstrap: {} cannot be put in a solve order - each reads a curve another '
                'builds, so whichever is solved first is solved against a curve that does not '
                'exist yet ({}). A mutually-referencing set has to be solved as ONE system; this '
                'family solves a block at a time.'.format(
                    ' + '.join(sorted(unresolved)),
                    '; '.join('{} reads {}'.format(name, ' + '.join(edges))
                              for name, edges in sorted(unresolved.items()))))
        return [(name, blocks[name]) for name in order]

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices,
                  calendars):
        """Solve every block for the zero curve that reprices its used quotes to par, one COUPLED
        SET at a time.

        A set is the group of blocks whose residuals read each other's curves, measured rather than
        declared (`coupled_sets`). Forming one costs a compile and a backward pass per block and
        buys a Jacobian that carries the coupling, so it is formed only where something reads that
        Jacobian - a ride, or a quote derivative, whose half across the coupling a block solved
        alone drops: an OIS quote then moves nothing priced off the curve discounting on it. With
        neither asked for this is a dependency-ordered loop. A benchmark set has one reporting
        currency, so a quote derivative's set stops at a currency - the other side a constant, as
        it always was - where a ride across one refuses (`solve_set`).
        """
        base_date = sys_params['Base_Date']
        blocks = self.in_dependency_order(market_prices)
        ride = any(entry['instrument']['Quote_Propagation'] == 'Linear' for _, entry in blocks)
        groups = self.coupled_sets(blocks, price_factors, factor_interp, base_date, calendars) \
            if ride or any(entry['instrument']['Quote_Sensitivity'] == 'Yes'
                           for _, entry in blocks) else [[block] for block in blocks]
        if not ride:
            # a run of one currency at a time, so the dependency order the set came in stands
            groups = [list(run) for group in groups for _, run in groupby(
                group, key=lambda member: member[1]['instrument']['Currency'])]

        for group in groups:
            self.solve_set(group, price_factors, factor_interp, base_date, calendars)

    def seed(self, market_price, block, price_factors, base_date, calendars):
        """Write this block's par-rate seed curve into `Price Factors`, and give back what the solve
        reads off it: `(curve factor, used quotes, deal nodes, discount rate)`.

        Seeding comes first because the benchmark closure constructs the curve factor OUT of
        `Price Factors`; a par rate is within a few basis points of the zero rate at the same
        maturity, so it is also the seed. `Curve` sorts the pairs, so each knot keeps its quote.

        The `/100` is the SEED's and not the quote's - `author_quote` scales nothing. So an
        amount-valued quote seeds nonsense and converges anyway: an 18.32 outright seeds an 18.32%
        zero rate against a true 8.99% and damped Newton walks it to zero residual. Branching on the
        deal type here would put knowledge of a type somewhere other than its own `quote`.
        """
        curve = utils.Factor('InterestRate', utils.check_rate_name(market_price)[1:])
        discount_rate = block['Discount_Rate'] or '.'.join(curve.name)
        points = self.used_quotes(block, market_price)
        nodes = quote_nodes(points, discount_rate)
        knots = self.quote_knots(nodes, base_date, block['Day_Count'], calendars)
        # a knot at tenor zero identifies nothing - the curve is flat below its shortest by
        # clipping - and the Newton solve meets it as a singular Jacobian rather than a bad curve
        dead = ['{} (last cashflow {:%Y-%m-%d})'.format(
            point['Descriptor'], max(max(leaf.get_reval_dates()) for leaf in leaf_deals(node)))
            for point, node, knot in zip(points, nodes, knots) if knot <= 0.0]
        if dead:
            raise ValueError(
                '{}: {} matured on or before the base date {:%Y-%m-%d}, so the knot each '
                'identifies lands at tenor zero, which no curve carries. Hold the quote out with '
                'Use No, or move the base date'.format(market_price, '; '.join(dead), base_date))
        written = {
            'Property_Aliases': None, 'Sub_Type': None, 'Currency': block['Currency'],
            'Day_Count': block['Day_Count'], 'Curve': utils.Curve([], list(zip(
                knots, [point['Quoted_Market_Value'] / 100.0 for point in points])))}
        if block['Near_Interpolation']:
            if not block['Near_Tenor']:
                raise ValueError('{}: Near_Interpolation {} names no Near_Tenor to stop at'.format(
                    market_price, block['Near_Interpolation']))
            # the near end is quoted in another instrument, so it carries another scheme: the
            # factor stacks the two on the knot the tenor lands in and the solve reads both
            written.update({'Near_Interpolation': block['Near_Interpolation'],
                            'Near_Date': base_date + block['Near_Tenor']})
        price_factors[utils.check_tuple_name(curve)] = written
        return curve, points, nodes, discount_rate

    def coupled_sets(self, blocks, price_factors, factor_interp, base_date, calendars):
        """This family's blocks grouped into the SETS that have to solve as one system - measured.

        Two blocks are coupled when one's residual READS the curve the other builds, which is not
        the question `Discount_Rate` answers: what a benchmark projects off is authored inside its
        own deal block, so a strip declaring a blank `Discount_Rate` can still forecast off a
        neighbour's curve - on such a world a 10bp tick moved the "independent" curve 568bp.
        `BenchmarkInstruments.reads` answers by differentiation instead.

        The groups are the connected components of that relation, in dependency order, and a group
        is solved and ridden WHOLE - which is what puts `dtheta_2/dq_1` inside `J`, an ordering
        carrying a coupling through a bootstrap but nothing through a ride.

        Every block is seeded before anything is measured, a block forecasting off an unbuilt curve
        being uncompilable.
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
        """Solve one coupled set: one Newton system over every curve in it, one Jacobian, one
        artifact.

        Flattening a multi-curve set is `utils.damped_newton`'s own shape rather than a new solver -
        `solve_for` is a list, the residual takes a `{Factor: nodes}` over it, and the block
        Jacobian that falls out is what `utils.calibration_jacobian` inverts in one go.

        The seed theta is read off the CONSTRUCTED factor, so it is aligned with the tenor grid the
        pricers gather against whatever `get_tenor` made of the block. The solve goes through the
        implicit-function wrapper either way; with no quotes on the tape no edge is recorded.

        Solver knobs are declared per block and a set takes the STRICTEST of them.
        """
        members = [(market_price, entry['instrument']) for market_price, entry in group]
        propagate = [block['Quote_Propagation'] == 'Linear' for _, block in members]
        connect = [block['Quote_Sensitivity'] == 'Yes' for _, block in members]
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
        # both switches want the quote side of the residual, one extra compile either way
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
        # seed theta off the constructed factor - see the docstring on grid alignment
        theta = utils.CalibrationSolve.apply(
            benchmarks,
            {curve: torch.tensor(benchmarks.factors[curve].current_value(),
                                 dtype=BenchmarkInstruments.dtype, device=self.device)
             for curve in curves},
            max(int(block['N_Iter']) for _, block in members),
            min(float(block['Tol']) for _, block in members),
            max(int(block['Damping_Halvings']) for _, block in members),
            benchmarks.quotes)

        solved = utils.split_theta(benchmarks, theta)
        # a set-wide quote leaf reports dV/dq across the system, so its descriptors name the block
        # each quote came off; a set of one is the block's own list unchanged
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
            self.published.append(
                self.publish(members, factor_interp, base_date, benchmarks, theta.detach()))

        residuals = benchmarks(utils.split_theta(benchmarks, theta.detach())).detach()
        logging.info('{} bootstrapped from {} quotes in {:.2f} seconds, residual {:.3g}'.format(
            ' + '.join(utils.check_tuple_name(curve) for curve in curves), len(points),
            time.monotonic() - time_now, float(residuals.abs().max())))
        for point, residual in zip(points, residuals):
            logging.info('  {} at {:.4f} reprices to {:.3g}'.format(
                point['Descriptor'], point['Quoted_Market_Value'], float(residual)))

    @classmethod
    def takes(cls, point, market_price):
        """Whether this family prices the quote. `Par_Rate` is the only convention built - every
        benchmark is held at PV zero. A futures price and a money-market rate on a different basis
        would have to be authored differently.
        """
        if point['Quote_Type'] == 'Par_Rate':
            return True
        logging.error('{} quote {} - Quote_Type {} not supported yet'.format(
            market_price, point['Descriptor'], point['Quote_Type']))
        return False

    @classmethod
    def used_quotes(cls, block, market_price):
        """The quotes that enter the solve, in the order theta, `J` and `q0` are all indexed by.

        A classmethod because the RIDE needs the same list off a block nobody is bootstrapping; a
        second filter beside this one is how a ridden theta ends up indexed differently from the
        artifact it rode.
        """
        return [point for point in quote_table(block, market_price)
                if point['Use'] == 'Yes' and cls.takes(point, market_price)]

    @classmethod
    def plan_key(cls, members, factor_interp, base_date):
        """The SLOT an artifact lives in: every member block of the coupled set, the base date, the
        interpolation scheme and the engine version - with the `lifecycle_fields` shadowed out and
        the quote VALUES projected away.

        Literally `schema.partition_market_price`'s structural half, the split `Config.plan_hash`
        takes over the same section, so the two cannot drift. Every tick of one strip lands on the
        same slot, which is what makes a ride possible, while a re-authored instrument, a flipped
        `Use`, a different `Day_Count`, a different solver knob or a new engine build lands
        elsewhere. A row that gains a `Quoted_Bid` keeps its slot: the solve reads neither side.

        The key names the SET rather than the block, so re-authoring a discount strip moves the slot
        of every curve solved against it.

        `base_date` and `Price Factor Interpolation` are in it because the SOLVE reads them and the
        block does not carry them. Without them two jobs 45 days apart share a slot, and a Linear
        job rides a Hermite solve 0.53bp away from its own.

        The block is COMPLETED by the family's declarations first, so an omitted knob and one
        written at its own default share a slot. A knob declared in `Bootstrapper Configuration`
        instead of on the block does not reach here - `propagate` runs off a document with no
        bootstrapper in it - so a riding set declares its knobs on its blocks or the ride refuses
        by name at the next EXECUTE.
        """
        # `config` imports from this module, so the package edge runs one way only
        from . import content_hash

        return content_hash({
            'engine_version': __version__, 'base_date': base_date, 'interpolation': factor_interp,
            'set': [{'market_price': market_price,
                     'block': dict(partition_market_price(
                         {'instrument': declared_defaults(cls, block)})[0]['instrument'],
                                   **{field: None for field in cls.lifecycle_fields})}
                    for market_price, block in members]})

    @classmethod
    def slot(cls, names, market_prices, factor_interp, base_date):
        """The key those member blocks address in `market_prices` NOW, or `None` if one is gone.

        What turns `utils.ArtifactStore.find`'s scan back into content addressing: an artifact answers for
        a curve only if the plan it was fitted against is still the plan standing.
        """
        members = [(name, market_prices.get(name, {}).get('instrument')) for name in names]
        if any(block is None for _, block in members):
            return None
        return cls.plan_key(members, factor_interp, base_date)

    @classmethod
    def publish(cls, members, factor_interp, base_date, benchmarks, theta):
        """Freeze this solve as an artifact for the config's store, which scores it against the
        one it takes the slot of (`utils.CalibrationArtifact.replacing`) under a new `artifact_id`."""
        return utils.CalibrationArtifact(
            cls.plan_key(members, factor_interp, base_date),
            [market_price for market_price, _ in members], theta,
            utils.calibration_jacobian(benchmarks, utils.split_theta(benchmarks, theta)),
            benchmarks.quotes.detach(), benchmarks, min(float(block['Tol']) for _, block in members))

    @classmethod
    def propagate(cls, artifacts, factor, market_prices, factor_interp, base_date):
        """The curve `factor` RIDDEN to the quotes standing in `market_prices` now off the config's
        `artifacts`, or `None` where no block asks for one - the operator, evaluated per EXECUTE and
        storing nothing.

        Two ways to get `None`: a factor this family does not write, and a block that did not ask
        for `Quote_Propagation`.

        A block that DID ask and finds no artifact REFUSES - a miss is a 404 rather than a different
        number. That closes the replay hole: falling back to `theta*` reprices the book (13.4% on
        the eviction probe) while `plan_hash`, `values_hash`, the engine version and the seed all
        stay identical. A fresh context rides nothing and says so, an artifact being unserialisable.

        A ride leaving the benchmarks further out of par than `Drift_Tolerance` refuses too. The
        tolerance is the SET's strictest, so a coupled set rides or refuses whole, and `slot`
        rechecks that the artifact's plan is the one still standing.
        """
        if factor.type != cls.price_factor_type:
            return None
        market_price = utils.check_tuple_name(utils.Factor(cls.market_factor_type, factor.name))
        block = market_prices.get(market_price, {}).get('instrument')
        if block is None or declared_defaults(cls, block)['Quote_Propagation'] != 'Linear':
            return None

        covering = artifacts.covering(factor)
        artifact = next((found for found in covering if found.key == cls.slot(
            found.members, market_prices, factor_interp, base_date)), None)
        if artifact is None:
            raise utils.CalibrationStale(
                '{}: Quote_Propagation is Linear and no calibration artifact answers to this plan '
                '- {}. Bootstrap the job and the same EXECUTE runs off the artifact that publishes; '
                'an artifact holds tensors and a compiled benchmark set, so it cannot be serialised '
                'and a fresh context has none. A plan the store cannot answer is a MISS, and a miss '
                'is not permission to price off the curve the last bootstrap wrote.'.format(
                    market_price, 'the store holds none for this curve' if not covering else
                    '{} cover it, each fitted against a different plan ({})'.format(
                        len(covering), '; '.join(' + '.join(found.members) for found in covering))))
        # a ride is a USE: a ridden slot must not age out under one merely published beside it
        artifacts.get(artifact.key)

        tolerance = min(
            float(declared_defaults(cls, market_prices[name]['instrument'])['Drift_Tolerance'])
            for name in artifact.members)
        quotes = torch.tensor(
            [point['Quoted_Market_Value'] for name in artifact.members
             for point in cls.used_quotes(market_prices[name]['instrument'], name)],
            dtype=artifact.theta.dtype, device=artifact.theta.device)
        theta = artifact.ride(quotes)
        drift = float(artifact.mispricing(theta, quotes).abs().max())
        tick = float((quotes - artifact.quotes).abs().max())
        # the tolerance is in percent of quote; ||J||inf converts it to the curve units it is felt in
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


class FXVolSurfaceParameters(Construction):
    """An `FXVol` surface bootstrapped from the ATM / risk-reversal / butterfly quotes it ticks in as.

    An FX smile is quoted in DELTA - one ATM vol per expiry and, per delta pillar, the risk reversal
    and butterfly that say how the two wings sit around it - while the surface the engine prices off
    is a log-moneyness one. The algebra between them is the strangle pair,
    `vol(call) = ATM + BF + RR/2` and `vol(put) = ATM + BF - RR/2`, followed by the
    delta-to-log-moneyness solve `Factor2D` carries for a `Malz` surface. What this family fixes is
    WHERE they run.

    **The x-grid is pinned.** The solve refines a log-moneyness grid until interpolating between its
    nodes resolves the smile, so the grid is a function of the quotes. Run at factor-construction
    time that would make every vol tick STRUCTURAL - a moved node is a moved tenor grid, a new plan
    and a recompile. So the refinement runs here, once, and the grid is part of the written factor;
    a re-bootstrap finding a surface already written for the same expiries at the same tolerance
    reuses it and moves only the vols, which is what makes a tick a `bind='value'` patch.
    `Grid_Tolerance` SIZES the grid, so it is structural and asking for a different one breaks the
    pin. The log says what the pinned grid resolves the CURRENT quotes to.

    **The conventions are declared because the solve implements exactly one of each.** The delta a
    pillar names is a premium-adjusted FORWARD delta ((K/F)N(d2) for a call), and the ATM quote is
    that convention's delta-neutral straddle, K = F exp(-sigma^2 T / 2).

    A quote `Timestamp` survives a save at the resolution it was authored.

    **A point may carry a two-way, and nothing here reads it.** `Quoted_Bid`/`Quoted_Ask` ride the
    row beside the mid for `derivus.structures` to charge a spread on. Every line below addresses
    `Quoted_Market_Value` by name, so what this writes is the mid surface either way.

    Like `InterestRatePrices` this writes a typed price factor rather than a `<ClassName>`
    parameter block, which is what `price_factor_type` declares.
    """
    market_factor_type = 'FXVolPrices'
    #: The `Price Factors` type this family writes - a `Malz` `FXVol`, minus the delta surface: it
    #: arrives SOLVED, so `Factor2D.solves_delta_surface` is false and the pinned grid survives.
    price_factor_type = 'FXVol'
    #: What this family READS out of `Price Factors`: nothing. The quotes are the block's own and
    #: the only factor it looks at is the `FXVol` it wrote last time, for the pinned grid.
    reads = ()
    #: `Surface_Type` names the moneyness convention the engine reads the block at (log(F/K),
    #: interpolated in total variance); `Moneyness_Rule` is the factor's own declared default and no
    #: Malz code path reads it.
    surface_type, moneyness_rule = 'Malz', 'Sticky_Moneyness'
    #: `Grid_Tolerance`'s own DOMAIN and not a dial, which is why it is declared here and not as a
    #: field: refinement halves an interval until the midpoint's vol error falls under the
    #: tolerance, so at 0.0 no midpoint qualifies (7.6M nodes on one expiry after 21 passes, still
    #: doubling) while 1e-8 is 4599 nodes for a four-expiry smile, and at 1 the seed grid already
    #: passes. `bootstrap` enforces it and the field declares it as its `bounds`, one spelling.
    grid_tolerance_bounds = (1e-8, 1.0)
    #: The precision the TAPE runs in - the value path is numpy and has no dtype to pick. Float64 on
    #: the CPU whatever the job asked for, the twin dividing by the residual's slope at the root.
    dtype = torch.float64
    fields = [
        F('Currency', 'Text', default='',
          description='The currency stamped on the surface this builds'),
        F('Delta_Type', 'Text', default='Forward', values=['Forward'],
          description='The delta a Pillar names; the solve inverts a forward delta, '
                      'the one convention offered'),
        F('Premium_Adjusted', 'Text', default='Yes', values=['Yes'],
          description='Whether the pillar delta is premium adjusted, the solve inverting the '
                      'premium-adjusted (percentage-foreign) delta (K/F)N(d2)'),
        F('ATM_Convention', 'Text', default='Delta_Neutral_Straddle',
          values=['Delta_Neutral_Straddle'],
          description='What an ATM quote is the vol of: the delta-neutral premium-adjusted '
                      'straddle, struck at K = F exp(-sigma^2 T/2)'),
        F('Grid_Tolerance', 'Float', default=1e-4, bounds=grid_tolerance_bounds,
          description='The vol error the log-moneyness grid is refined to when it is built, a '
                      'structural setting that sizes the pinned grid rather than each quote fit'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the log-moneyness surface connected to the ATM / RR / BF '
                      'quotes it was built from, so a calculation\'s backward pass '
                      'reports dV/dq beside dV/dtheta'),
        F('Points', 'Table', default='null', row=Row([
            F('Use', 'Text', default='Yes', values=['Yes', 'No'],
              description='Whether this quote enters the surface'),
            F('Expiry', 'Float',
              description='Expiry in years, the surface\'s own expiry axis'),
            F('Pillar', 'Float',
              description='The delta the wings are quoted at, as a magnitude (0.25 is the 25 delta '
                          'pair), unread on an ATM row'),
            F('Quote_Type', 'Text', default='ATM', values=['ATM', 'RR', 'BF'],
              description='The ATM vol, the risk reversal (call less put) or the butterfly (the '
                          'wing pair\'s average over ATM)'),
            F('Quoted_Market_Value', 'Float',
              description='The quote, in the surface\'s own units: 0.12 for 12 vols, and a risk '
                          'reversal of -0.35 vols is -0.0035'),
            F('Quoted_Bid', 'Float',
              description='The bid side of this quote, in the surface\'s own units; optional and '
                          'never read by the surface or its marks'),
            F('Quoted_Ask', 'Float',
              description='The offer side, the pair of Quoted_Bid, which a structure reads to '
                          'quote a client two-sided and without which it quotes at mid'),
            F('Timestamp', 'Date', default='',
              description='When this quote was observed; stored and reported, the surface carrying '
                          'the latest, and never read by pricing')]),
          description='One quote: an expiry, a delta pillar, what kind of number is quoted, the '
                      'number, and when it was seen')
    ]

    def __init__(self, param, device, dtype):
        super().__init__(param, device, dtype)
        #: What `Quote_Sensitivity` leaves behind: the log-moneyness surface still connected to its
        #: quotes, keyed as `_build_factor_state` mints the `FXVol` leaf, plus the quote leaf per
        #: block. `Config.bootstrap` harvests both - tensors cannot live in `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}

    @staticmethod
    def used(block, market_price=None):
        """The block's quotes that enter the surface - `Use` holds one out without deleting it."""
        return [point for point in quote_table(block, market_price) if point['Use'] == 'Yes']

    @staticmethod
    def descriptor(point):
        """What a quote is CALLED where `dV/dq` is reported: its type, its pillar, its expiry."""
        return ('ATM {:g}'.format(point['Expiry']) if point['Quote_Type'] == 'ATM' else
                '{} {:g} {:g}'.format(point['Quote_Type'], point['Pillar'], point['Expiry']))

    @staticmethod
    def atm_quotes(quotes):
        """`{expiry: ATM vol}` - the ATM row per expiry, the number that expiry's wings sit around.

        The surface's ATM vol at an expiry IS this number: `Factor2D.malz_skew` places it at the
        delta-neutral straddle strike. So it is what `smile` builds the wings off, and what
        `GBMAssetPriceTSModelParameters` takes as the ATM column of a surface this family built.
        """
        return {point['Expiry']: point['Quoted_Market_Value']
                for point in quotes if point['Quote_Type'] == 'ATM'}

    @classmethod
    def smile(cls, quotes):
        """The quotes as a `(delta, expiry, vol)` surface - the strangle pair, per expiry pillar.

        `vol(call) = ATM + BF + RR/2` and `vol(put) = ATM + BF - RR/2`, with the ATM vol carried at
        the +-0.5 LABEL the delta solve reads it off (0.5 is not a delta there - the solve replaces
        the label with the delta-neutral straddle's own delta). A pillar quoted with only one of the
        two is read with the other at zero; an expiry with wings but no ATM quote raises `KeyError`.

        A `Pillar` of 0.5 is refused: a wing quoted at the ATM label would land a second vol on the
        ATM row's coordinate. A 50 delta pair is quoted as the ATM row.
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

        Row for row and in `smile`'s own order, so the frozen structure the value path leaves
        addresses this vector. The algebra is `+`, `*` and a sort, on which torch and numpy agree to
        the last bit in float64, so the mirror is bit-identical and gated as such.
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

        The wing vols are taped and so is `delta_atm`, the ATM quote saying where the delta-neutral
        straddle sits and so MOVING the two ATM nodes of the delta grid. What is read off the
        numbers rather than differentiated is the LAYOUT - the ordering, which node carries the
        +-0.5 label, which side had its ATM node mirrored in - a permutation having no derivative.
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
        """`Factor2D.malz_sigma` on a tape - and the bisection is NOT on it.

        A bisection's iterates are dyadic combinations of the bracket endpoints, so a tape through
        the 64 halvings differentiates where the BRACKET is rather than where the root is: on the
        call wing that derivative carries the ATM quote and no risk reversal or butterfly at all,
        while the true root moves with the wing vols.

        So the tape starts at the CONVERGED root. `delta*` is a constant here and the differentiable
        one is one Newton step off it, `delta - R(delta, q) / (dR/ddelta)` - the implicit function
        theorem as an expression, worth the solve's own residual forward and exactly `-R_q/R_delta`
        backward. What makes it the theorem is that `delta*` is the ROOT, not that the slope is
        detached.

        The CLAMPED nodes take the other branch, and it is not a repair: outside the wing's bracket
        there is no root to differentiate, the vol IS the endpoint knot's, and the derivative is
        that knot vol's own. The two branches meet where the root arrives at the endpoint, so the
        switch is a kink and autograd reports the branch's one-sided derivative.

        The wing span is guarded, and an ordinary config reaches it: an ATM-only smile has
        `malz_skew` mirror its one node onto both sides, so each wing is a single knot of zero span.
        Dividing before selecting would NaN the whole Jacobian while the value path writes a
        perfectly good flat surface.
        """
        delta_star, is_call, bracketed = riskfactors.Factor2D.malz_delta(skew, T, x)
        sigma = carried['sigma_atm'].new_zeros(np.shape(x))

        for side, wing in ((1.0, 'call'), (-1.0, 'put')):
            on_wing = is_call if side > 0 else ~is_call
            if not on_wing.any():
                continue
            knots, values, grid = carried['d_' + wing], carried['v_' + wing], skew['d_' + wing]
            xs, root, live = x[on_wing], delta_star[on_wing], bracketed[on_wing]
            # the segment the root sits in and, for a clamped node, the endpoint knot it sits on -
            # both frozen, an interval index not being a differentiable quantity
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

            # `base` carries no graph, so its `.detach()` and the slope's missing `create_graph`
            # are no-ops - the theorem holds off the ROOT. |dR/ddelta| stays above 0.948 in a
            # 375-point sweep, so the `on_tape` guard is idiom
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
        get None, each a grid the quotes are not asking for: nothing to pin to; a different SUBTYPE,
        whose moneyness axis is S/K rather than log(F/K); a different set of EXPIRIES, which a
        rebuild answers and stretching does not; and a different TOLERANCE.
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
                  calendars):
        """Turn each block's quotes into the log-moneyness `FXVol` surface the pricers read.

        The x-grid is taken from the factor this wrote last if it still describes the same expiries
        at the same tolerance - see the class docstring on pinning - and refined otherwise.

        `Quote_Sensitivity` leaves that surface behind still connected to its ATM / RR / BF quotes,
        so `Calculation.factor_leaf` can offer the connected tensor rather than minting an `FXVol`
        leaf out of numpy. The tape is a SPLICE over the shipped conversion (`carried_sigma`), so
        every number written comes out of `Factor2D.malz_surface`.

        The grid is NOT differentiated. It is refined against the quotes when built and pinned from
        then on, which is what makes a tick a values patch; the twin moves the vols on frozen nodes.
        """
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                # the quote's own instrument wins on conflict
                block = dict(self.param, **implied_params['instrument'])
                vol_name = utils.check_tuple_name(
                    utils.Factor(self.price_factor_type, market_factor.name))

                tolerance = float(block['Grid_Tolerance'])
                # the one bounds= the engine reads: outside it there is no grid to refine to
                if not self.grid_tolerance_bounds[0] <= tolerance <= self.grid_tolerance_bounds[1]:
                    raise ValueError(
                        '{}: Grid_Tolerance {:g} is outside [{:g}, {:g}] - the refinement does not '
                        'terminate there'.format(market_price, tolerance,
                                                 *self.grid_tolerance_bounds))

                quotes = self.used(block, market_price)
                delta_surface = self.smile(quotes)
                expiries = np.unique(delta_surface[:, 1])
                skews = riskfactors.Factor2D.malz_skews(delta_surface, expiries)

                grid = self.pinned_grid(price_factors.get(vol_name), expiries, tolerance)
                pinned = grid is not None
                if not pinned:
                    grid = riskfactors.Factor2D.malz_grid(skews, tolerance)

                surface = riskfactors.Factor2D.malz_surface(skews, grid)
                stamps = [point['Timestamp'] for point in quotes if point.get('Timestamp')]
                price_factors[vol_name] = {
                    'Property_Aliases': None, 'Surface_Type': self.surface_type,
                    'Moneyness_Rule': self.moneyness_rule,
                    'Currency': block['Currency'],
                    'Grid_Tolerance': tolerance,
                    'Quote_Timestamp': max(stamps) if stamps else '',
                    'Surface': utils.Curve([], surface)}

                if block['Quote_Sensitivity'] == 'Yes':
                    leaves = torch.tensor([point['Quoted_Market_Value'] for point in quotes],
                                          dtype=self.dtype, requires_grad=True)
                    carried = self.carried_surface(skews, self.carried_skews(
                        delta_surface, expiries, self.carried_smile(quotes, leaves)), grid)
                    # `Factor2D` sorts by (expiry, moneyness) and mints a leaf out of THAT column,
                    # so the twin is put in the same order rather than assumed to be in it
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


def family_class(btype):
    """The price family a `Bootstrapper Configuration` entry names: the `Price Factors` TYPE it
    writes, or its class name, which stays an alias for every book written before that. An unknown
    name refuses by name, listing both spellings."""
    cls = WRITERS.get(btype)
    if cls is None:
        raise ValueError(
            'Bootstrapper Configuration names {}, which is no price family; the families are {} '
            '(or, as older books spell them, {})'.format(
                btype, ', '.join(sorted(x.price_factor_type for x in FAMILIES)),
                ', '.join(sorted(x.__name__ for x in FAMILIES))))
    return cls


def bootstrap_order(section):
    """The `Bootstrapper Configuration` entries in the order they must RUN: a topological sort over
    what each family writes (`price_factor_type`) against what it reads (`reads`), so a curve is
    solved before the fit that prices on it whatever order the file gives, and both the in-process
    and the multiprocessing path take the same order.

    A read no CONFIGURED family writes carries no edge - that factor is already in `Price Factors`,
    which is the ordinary case - and a family reading what it writes orders only its own blocks,
    which is its own job. Independent entries keep the file's order, `topological_sort` walking the
    mapping as it was built. A cycle refuses by name.
    """
    writes = {}
    for name in section:
        writes.setdefault(family_class(name).price_factor_type, []).append(name)
    graph = {name: sorted({writer for kind in family_class(name).reads
                           for writer in writes.get(kind, ()) if writer != name})
             for name in section}
    unresolved = dict(graph)
    try:
        return utils.topological_sort(unresolved)
    except RuntimeError:
        raise ValueError(
            'Bootstrapper Configuration: {} cannot be put in a run order - each reads a factor '
            'another writes, so whichever runs first prices off one that does not exist yet ({}). '
            'Bootstrap them in separate runs'.format(
                ' + '.join(sorted(unresolved)),
                '; '.join('{} reads what {} writes'.format(name, ' + '.join(edges))
                          for name, edges in sorted(unresolved.items()))))


def block_readers(market_prices):
    """`{block: [every block reading what it writes]}` over one `Market Prices` section.

    A block is read two ways and both are the families' own declarations. `reads` names the price
    factor TYPES a family prices on, which is the whole of it for a surface or a spot model; the
    curve family names the very curve, in `Discount_Rate` and inside its benchmark deals
    (`benchmark_curves`), so a curve is read by the curves discounting on it and no others. A block
    of a type no family reads is nobody's dependency.
    """
    writer = {cls.market_factor_type: cls for cls in FAMILIES}
    blocks = {name: writer[utils.check_rate_name(name)[0]] for name in market_prices
              if utils.check_rate_name(name)[0] in writer}
    curves = InterestRateCurveParameters
    named = {name: {block.get('Discount_Rate')} | curves.benchmark_curves(block, name)
             for name, cls in blocks.items() if cls is curves
             for block in [market_prices[name].get('instrument', {})]}
    readers = {}
    for reader, cls in blocks.items():
        for written, wrote in blocks.items():
            if written != reader and wrote.price_factor_type in cls.reads and (
                    cls is not curves or wrote is not curves
                    or '.'.join(utils.check_rate_name(written)[1:]) in named[reader]):
                readers.setdefault(written, []).append(reader)
    return readers


def bootstrap_dependents(market_prices, moved):
    """The `Market Prices` blocks a run must COVER once the blocks named by `moved` carry new
    numbers: those blocks, plus every block that reads what one of them writes, closed over."""
    return utils.closed_over(block_readers(market_prices), moved)


def bootstrap_precedents(market_prices, wanted):
    """The blocks a fit of `wanted` stands on: those blocks, plus every block one of them reads,
    closed over - the walk `bootstrap_dependents` takes, backwards. A curve discounting on another
    is one system with it, so its quotes move whatever the first is read by."""
    reads = {}
    for written, readers in block_readers(market_prices).items():
        for reader in readers:
            reads.setdefault(reader, []).append(written)
    return utils.closed_over(reads, wanted)


def bootstrap_writers(market_prices, factors):
    """The blocks writing one of the price factors named in `factors` - a block writes its family's
    `price_factor_type` under its own name."""
    writes = {cls.market_factor_type: cls.price_factor_type for cls in FAMILIES}
    return {name for name in market_prices
            for rate in [utils.check_rate_name(name)] if rate[0] in writes
            and utils.check_tuple_name(utils.Factor(writes[rate[0]], rate[1:])) in factors}


def market_prices_for(btype, market_prices, declared=None):
    """The `Market Prices` blocks the family named by one `Bootstrapper Configuration` entry reads.

    THE CONFIGURATION DRIVES THE LOOP where the engine is importable: `Config.bootstrap` selects
    here and hands a family its own blocks. `declared` is the entry's own `Prices` STEM where it
    carries one, VERIFIED against the family's own `market_factor_type` so a section routing a
    family at another family's type refuses by name rather than fitting nothing. Each family
    still filters by type in its own `bootstrap`, because `derivus_bootstrap` hands one task the
    whole section where it must.
    """
    wanted = family_class(btype).market_factor_type
    if declared and declared + 'Prices' != wanted:
        raise ValueError(
            'Bootstrapper Configuration.{0}: {1} {2!r} routes it at {3}, which {0} does not read - '
            'it reads {4}. Write {1} {5!r}, or configure the family that reads {3}'.format(
                btype, PRICES_KEY, declared, declared + 'Prices', wanted,
                wanted[:-len('Prices')]))
    return {name: block for name, block in market_prices.items()
            if utils.check_rate_name(name)[0] == wanted}


def construct_bootstrapper(btype, param, dtype=torch.float32):
    """One family built off its `Bootstrapper Configuration` entry: the entry's hyperparameters
    without the routing key, and `{}` for the legacy CSV string, whose positional tail declares
    none - every field it does not carry is the declaration's own default."""
    device = utils.calculation_device()
    param = ({key: value for key, value in param.items() if key != PRICES_KEY}
             if isinstance(param, dict) else {})
    return family_class(btype)(param, device, dtype)


#: THE SIX PRICE FAMILIES: the registry below and every refusal naming them read this one tuple.
FAMILIES = (CSForwardPriceModelParameters, LogVar2FJModelParameters,
            GBMAssetPriceTSModelParameters, HullWhite2FactorModelParameters,
            InterestRateCurveParameters, FXVolSurfaceParameters)

#: THE REGISTRY: what a `Bootstrapper Configuration` entry may name a family by -> the family, in
#: both spellings - the `Price Factors` type it writes, so a section reads as the factors it
#: produces, and the class name every book written before that spells it by.
WRITERS = dict([(x.price_factor_type, x) for x in FAMILIES]
               + [(x.__name__, x) for x in FAMILIES])
