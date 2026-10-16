"""The HW2F composition harness: calibrate domestic, simulate global, and read the seam.

Half A fits a swaption ladder under its own currency's risk-neutral measure, so theta* is invariant
to the FX/IR correlation. Half B simulates the fitted curve in a base-currency job, quanto drift
included, and reads identity 1, the base-deflated, FX-converted exposure at each expiry against
spot times the domestic Schrager-Pelsser price, beside the payoff-free numeraire identity, one unit
of the rate currency at T against `X_0 P(0,T)`. This module runs the pipeline on a snapshot
directory and prints its tables; `tests/test_hw2f_composition.py` runs the same functions on an
authored world.

    PYTHONIOENCODING=utf-8 python gates/hw2f_composition.py --snapshot DIR [--theta theta.json]

The fit is deterministic, so the report prints theta* on one `THETA_STAR {...}` line and `--theta`
reads it back, a file or the JSON inline, to run everything downstream of the solve.
"""
import argparse
import copy
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

import derivus
from derivus import riskfactors, utils
from derivus.bootstrappers import (BenchmarkInstruments, HullWhite2FactorModelParameters,
                                   quote_nodes)
from derivus.config import Config, CustomJsonEncoder
from derivus.instruments import construct_instrument
from derivus.utils import generate_dates_backward

DTYPE = torch.float64
#: The three benchmarks the composition prices, as `(start, tenor)` period strings. Every one is a
#: cell of the live ladder; they span the grid's expiry axis rather than clustering on it.
COMPOSITION_CELLS = (('1Y', '2Y'), ('2Y', '5Y'), ('5Y', '5Y'))


# ---------------------------------------------------------------------------------------------
# THE SNAPSHOT
# ---------------------------------------------------------------------------------------------

def read_snapshot(directory):
    """The emitted blocks of one snapshot, decoded through the engine's OWN JSON codec.

    `Config.read_json` is what turns `.Percent`, `.DateOffset`, `.Timestamp` and `.DateList` back
    into the objects a `Market Prices` block is made of, so a snapshot file and a hand-authored
    block are the same shape by construction rather than by a second decoder written here.
    """
    config = Config()
    out = {'dir': directory, 'blocks': {}}
    for filename in ('usd_curve.json', 'zar_curve.json', 'zar_swaption.json', 'usdzar_fxvol.json'):
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            out['blocks'].update(config.read_json(path))
    spot = os.path.join(directory, 'usdzar_spot.json')
    if os.path.exists(spot):
        out['spot'] = config.read_json(spot)
    manifest = os.path.join(directory, 'manifest.json')
    if os.path.exists(manifest):
        out['manifest'] = config.read_json(manifest)
    return out


# ---------------------------------------------------------------------------------------------
# THE WORLD - one builder, two callers (the live snapshot and the canned twin)
# ---------------------------------------------------------------------------------------------

def new_config(base_date, base_currency='USD'):
    """An empty job config at `base_date`, based in `base_currency`, with no market prices and no
    bootstrap configuration."""
    config = Config(base_currency=base_currency)
    config.params['System Parameters']['Base_Date'] = pd.Timestamp(base_date)
    config.params['System Parameters']['Base_Currency'] = base_currency
    config.params['Market Prices'] = {}
    config.params['Bootstrapper Configuration'] = {}
    return config


def seed_fx(config, currency, discount_curve, spot, domestic=None):
    """One `FxRate` price factor, its `spot` the value of one unit of `currency` in the base
    currency (under a USD base, the reciprocal of a USDZAR screen print). `domestic` is the currency
    whose rate the FX drift reads first, required where the rate is simulated; the base currency's
    own factor keeps None."""
    config.params['Price Factors']['FxRate.{}'.format(currency)] = {
        'Domestic_Currency': domestic, 'Interest_Rate': discount_curve, 'Priority': 1,
        'Spot': float(spot)}


def install_blocks(config, blocks):
    """Install `Market Prices` blocks by name. Each is `{'instrument': {...}}`, the shape every
    bootstrapper reads and the shape the emitters write."""
    for name, block in blocks.items():
        if not isinstance(block, dict) or 'instrument' not in block:
            raise ValueError('{}: a Market Prices block is {{"instrument": {{...}}}}'.format(name))
        config.params['Market Prices'][name] = copy.deepcopy(block)


def yield_vol_surface(config, name, currency, distribution='Normal'):
    """The `InterestYieldVol` surface a swaption block names, undisplaced: the block's rows carry the
    vols and the surface only their convention, so its quads are placeholders."""
    quads = [[m, e, t, 0.01] for t in (1.0, 5.0, 10.0) for e in (0.25, 1.0, 5.0, 10.0)
             for m in (-0.01, 0.0, 0.01)]
    config.params['Price Factors']['InterestYieldVol.{}'.format(name)] = {
        'Property_Aliases': None, 'Currency': currency, 'Distribution_Type': distribution,
        'Shift': utils.Percent(0), 'Surface': utils.Curve([], quads)}


def fx_gbm_block(surface_name):
    """The `GBMAssetPriceTSModelPrices` block that writes the FX rate's vol curve off the ATM column
    of `surface_name`, the composition's one authored FX input."""
    return {'instrument': {'Asset_Price_Volatility': surface_name, 'Quote_Sensitivity': 'No'}}


def set_fx_ir_correlation(config, base_currency, currency, ir_curve, value):
    """The `Correlation.FxRate.<sorted pair>/InterestRate.<curve>` price factor at the desk's
    `value`, the mark the bootstrap derives the quanto correlations from, or removed when `value` is
    None; returns its name."""
    pair = '.'.join(sorted((base_currency, currency)))
    name = 'Correlation.FxRate.{}/InterestRate.{}'.format(pair, ir_curve)
    if value is None:
        config.params['Price Factors'].pop(name, None)
    else:
        config.params['Price Factors'][name] = {'Value': float(value)}
    return name


def run_bootstrappers(config, families):
    """Run `families` in the order given, one `Config.bootstrap` call each: the verb runs its
    section sorted, which would put `InterestRate` behind the swaption fit that reads its curve."""
    for family in families:
        config.params['Bootstrapper Configuration'] = {family: {}}
        config.bootstrap()
    config.params['Bootstrapper Configuration'] = {}


class LogCapture(logging.Handler):
    """Every record the bootstrap emits, kept as text.

    The honesty reprice is REPORTED BY NAME AND NOWHERE ELSE - `RiskNeutralInterestRateModel`
    logs what the engine's own Monte Carlo makes of an analytically-solved theta* and returns
    nothing - so reading it off the log is reading the engine's own number rather than a second
    computation of it.
    """

    def __init__(self, level=logging.INFO):
        super().__init__(level=level)
        self.lines = []

    def emit(self, record):
        try:
            self.lines.append(record.getMessage())
        except Exception:
            pass

    def matching(self, needle):
        return [line for line in self.lines if needle in line]


# ---------------------------------------------------------------------------------------------
# HALF A - THE DOMESTIC FIT
# ---------------------------------------------------------------------------------------------

def solve_curve(config, market_price):
    """Bootstrap the `InterestRate` family and reprice `market_price`'s benchmarks at the solved
    nodes through `BenchmarkInstruments`, each residual as a PV and as the par-rate move closing it,
    read off the quotes and the quotes plus one percent since a benchmark's PV is affine in its
    quote. The family is solved whole on every call, which is idempotent."""
    block = config.params['Market Prices'][market_price]['instrument']
    run_bootstrappers(config, ['InterestRate'])

    curve = utils.Factor('InterestRate', utils.check_rate_name(market_price)[1:])
    discount_rate = block['Discount_Rate'] or '.'.join(curve.name)
    points = [p for p in block['Points'] if p.get('Use', 'Yes') == 'Yes']
    base_date = config.params['System Parameters']['Base_Date']

    def priced(shift):
        benchmarks = BenchmarkInstruments(
            quote_nodes(points, discount_rate, shift), config.params['Price Factors'],
            config.params['Price Factor Interpolation'], base_date, block['Currency'],
            config.holidays, [curve], torch.device('cpu'))
        theta = {curve: torch.tensor(benchmarks.factors[curve].current_value(),
                                     dtype=BenchmarkInstruments.dtype)}
        return benchmarks(theta).detach().cpu().numpy().astype(np.float64)

    at_quote, at_quote_plus = priced(0.0), priced(1.0)
    slope = at_quote_plus - at_quote
    par_bp = np.where(slope != 0.0, -100.0 * at_quote / np.where(slope != 0.0, slope, 1.0), np.nan)
    nodes = config.params['Price Factors'][utils.check_tuple_name(curve)]['Curve'].array
    return {'market_price': market_price, 'curve': utils.check_tuple_name(curve),
            'descriptors': [p.get('Descriptor', '') for p in points],
            'pv': at_quote, 'par_bp': par_bp, 'nodes': nodes,
            'max_abs_pv': float(np.abs(at_quote).max()),
            'max_abs_bp': float(np.nanmax(np.abs(par_bp)))}


def fit_swaptions(config, market_price):
    """Run the HW2F swaption calibration on the block as authored, its `Objective` the declared
    default, and return theta* as `save_params` wrote it, the wall clock and the bootstrap's log
    lines, the honesty reprice among them."""
    capture = LogCapture()
    previous = logging.root.level
    logging.root.setLevel(min(previous, logging.INFO))
    logging.root.addHandler(capture)
    started = time.monotonic()
    try:
        run_bootstrappers(config, ['HullWhite2FactorModelParameters'])
    finally:
        logging.root.removeHandler(capture)
        logging.root.setLevel(previous)
    seconds = time.monotonic() - started

    rate = utils.check_rate_name(market_price)
    name = utils.check_tuple_name(
        utils.Factor('HullWhite2FactorModelParameters', rate[1:]))
    if name not in config.params['Price Factors']:
        raise RuntimeError('{} wrote no {} - the fit did not run'.format(market_price, name))
    reprice = capture.matching('away from market')
    return {'factor': name, 'param': config.params['Price Factors'][name], 'seconds': seconds,
            'honesty': reprice[-1] if reprice else None, 'log': capture.lines}


def theta_vector(param):
    """theta* as one flat float64 array: Alpha_1, Sigma_1, Alpha_2, Sigma_2, Correlation."""
    return np.concatenate([
        np.atleast_1d(np.float64(param['Alpha_1'])),
        param['Sigma_1'].array[:, 1].astype(np.float64),
        np.atleast_1d(np.float64(param['Alpha_2'])),
        param['Sigma_2'].array[:, 1].astype(np.float64),
        np.atleast_1d(np.float64(param['Correlation']))])


#: The one-line prefix the report writes theta* under and `--theta` reads it back from. A token
#: rather than prose so a verifier can lift theta* out of a log with a single grep.
THETA_LINE = 'THETA_STAR '
#: Every field `save_params` writes, in its order: the simulator reads the quanto fields too, which
#: are emitted, not fitted.
THETA_FIELDS = ('Property_Aliases', 'Quanto_FX_Volatility', 'Alpha_1', 'Sigma_1', 'Alpha_2',
                'Sigma_2', 'Correlation', 'Quanto_FX_Correlation_1', 'Quanto_FX_Correlation_2')


def theta_json(param):
    """theta* as one line of `CustomJsonEncoder` JSON, what the report prints and `--theta` reads.
    Scalars go out as floats: the encoder writes a numpy float as `.Unknown`, logging, not raising."""
    emitted = {}
    for key in THETA_FIELDS:
        if key not in param:
            continue
        value = param[key]
        emitted[key] = float(value) if isinstance(
            value, (int, float, np.floating, np.integer)) else value
    return json.dumps(emitted, cls=CustomJsonEncoder)


def read_theta(text):
    """`--theta`'s argument, a file or the JSON inline, as a parameter block, refused unless it
    carries the fitted fields and both quanto correlations."""
    config = Config()
    block = config.read_json(text) if os.path.exists(text) else config.read_json(
        (text, 'inline theta'))
    missing = [key for key in ('Alpha_1', 'Sigma_1', 'Alpha_2', 'Sigma_2', 'Correlation')
               if key not in block]
    if missing:
        raise ValueError(
            '--theta: not a HullWhite2FactorModelParameters block - missing {}'.format(
                ', '.join(missing)))
    if not block.get('Quanto_FX_Correlation_1') or not block.get('Quanto_FX_Correlation_2'):
        raise ValueError(
            '--theta: the block carries no quanto correlations - half B off this theta* would be '
            'a composition with no measure change in it')
    return block


def install_theta(config, market_price, theta):
    """`theta` written onto `Price Factors` in place of a solve, returned in `fit_swaptions`' shape
    with no wall clock or honesty reprice and flagged `injected`."""
    name = utils.check_tuple_name(utils.Factor(
        'HullWhite2FactorModelParameters', utils.check_rate_name(market_price)[1:]))
    config.params['Price Factors'][name] = copy.deepcopy(theta)
    return {'factor': name, 'param': config.params['Price Factors'][name], 'seconds': 0.0,
            'honesty': None, 'log': [], 'injected': True}


def fit_closure(config, market_price, device=torch.device('cpu'), dtype=DTYPE):
    """The calibration's residual closure at the theta* standing on the price factor, built as
    `RiskNeutralInterestRateModel.bootstrap` builds it (`implied_process`, which seeds off that
    factor, then `calc_loss_on_ir_curve`) and evaluated once."""
    boot = HullWhite2FactorModelParameters({}, device, dtype)
    block = config.params['Market Prices'][market_price]['instrument']
    factors, interp = (config.params['Price Factors'],
                       config.params['Price Factor Interpolation'])
    base_date = config.params['System Parameters']['Base_Date']
    base_currency = config.params['System Parameters']['Base_Currency']
    rate = utils.check_rate_name(market_price)
    ir_factor = utils.Factor('InterestRate', rate[1:])
    ir_curve = riskfactors.construct_factor(ir_factor, factors, interp)
    surface = riskfactors.construct_factor(
        utils.Factor('InterestYieldVol', utils.check_rate_name(block['Swaption_Volatility'])),
        factors, interp)
    surface.delta = 0.0
    surface.set_premiums(None, ir_curve.get_currency())
    implied_obj, process, vol_tenors = boot.implied_process(
        base_currency, factors, {}, ir_curve, rate)
    mtm = set([base_date + x['Start'] for x in block['Instrument_Definitions']])
    time_grid = utils.TimeGrid(mtm, mtm, mtm)
    time_grid.set_base_date(base_date, delta=(10, vol_tenors * utils.DayCount.DAYS_IN_YEAR))
    implied_var, objective, swaps = boot.calc_loss_on_ir_curve(
        {'instrument': block}, base_date, time_grid, process, implied_obj, ir_factor, surface)
    # `schrager_pelsser_swaption` reads what `covariance` builds, and building the closure
    # does not run it
    objective.loss(implied_var)
    return dict(model=boot, process=process, implied_var=implied_var, objective=objective,
                swaps=swaps, curve=ir_curve, time_grid=time_grid, block=block,
                surface=surface, implied_obj=implied_obj, base_date=base_date)


def fit_readings(closure):
    """Per benchmark, the model and market normal vols at theta* and the objective's own residual,
    in basis points; with the residuals' rms and the worst benchmark."""
    prices, errors = closure['objective'].loss(closure['implied_var'])
    rows = {}
    for name, swap in closure['swaps'].items():
        model = closure['process'].schrager_pelsser_swaption(
            swap.schedule.expiry, swap.schedule.pay_times, swap.schedule.accruals)
        market_nvol = float(swap.market_normal_vol(model.annuity).detach())
        rows[name] = {
            'model_premium': float(model.premium.detach()), 'market_premium': float(swap.price),
            'model_nvol_bp': 1e4 * float(model.normal_vol.detach()),
            'market_nvol_bp': 1e4 * market_nvol,
            'residual_bp': 1e4 * float(errors[name].detach()),
            'annuity': float(model.annuity.detach()),
            'swap_rate': float(model.swap_rate.detach()),
            'expiry': float(swap.schedule.expiry), 'weight': float(swap.weight)}
    residuals = np.array([row['residual_bp'] for row in rows.values()])
    worst = max(rows, key=lambda n: abs(rows[n]['residual_bp']))
    return {'rows': rows, 'rms_bp': float(np.sqrt((residuals ** 2).mean())),
            'worst': worst, 'worst_bp': rows[worst]['residual_bp'],
            'n': len(rows)}


# ---------------------------------------------------------------------------------------------
# THE QUANTO CORRELATIONS
# ---------------------------------------------------------------------------------------------

def reemit_quanto(param, base_currency, currency, rho_quote):
    """`param` with `Quanto_FX_Correlation_1/2` derived again by `get_quanto_correlation` from the
    desk's mark `rho_quote`, sign-reversed as the bootstrap reverses it, or zero when `rho_quote` is
    None: theta* does not move with the mark, and the drift does."""
    if rho_quote is None:
        return dict(param, Quanto_FX_Correlation_1=0.0, Quanto_FX_Correlation_2=0.0)
    sign = -1.0 if sorted((base_currency, currency))[0] == base_currency else 1.0
    obj = riskfactors.HullWhite2FactorModelParameters(
        dict(param, short_rate_fx_correlation=sign * float(rho_quote)))
    first, second = obj.get_quanto_correlation(
        np.atleast_1d(float(param['Correlation'])),
        [param['Sigma_1'].array[:, 1], param['Sigma_2'].array[:, 1]])
    return dict(param, Quanto_FX_Correlation_1=float(first),
                Quanto_FX_Correlation_2=float(second))


def quanto_correlations(param):
    """The `Correlations` rows `(FX x F1, FX x F2, F1 x F2)` under which the paths carry the
    covariance `param`'s quanto correlations assume, in the innovation basis."""
    rho = float(param['Correlation'])
    bar1 = float(param.get('Quanto_FX_Correlation_1') or 0.0)
    bar2 = float(param.get('Quanto_FX_Correlation_2') or 0.0)
    span = np.sqrt(max(1.0 - rho * rho, 1e-12))
    return bar1, (bar2 - rho * bar1) / span, 0.0


# ---------------------------------------------------------------------------------------------
# HALF B - THE COMPOSITION
# ---------------------------------------------------------------------------------------------

def par_forward_swap_leg(curve, base_date, start, tenor, frequency, day_count):
    """The fixed leg of the swap starting `start` after `base_date` for `tenor`, and its par rate
    `(P(0,T0) - P(0,Tn))/A(0)` off the solved t=0 curve, the analytic swaption's at-the-money rate.
    Times are year fractions on the curve's own day count, the clock `J` is integrated on."""
    effective = base_date + start
    dates = generate_dates_backward(effective + tenor, effective, frequency)
    cash = utils.TensorCashFlows.generate_fixed(
        base_date, dates, 1.0, None, utils.DayCount.code(day_count), 0.0)
    pay_days = cash.schedule[:, utils.CASHFLOW_INDEX_Pay_Day]
    tau = cash.schedule[:, utils.CASHFLOW_INDEX_Year_Frac]
    exp_days = (effective - base_date).days
    T0 = float(curve.get_day_count_accrual(base_date, exp_days))
    pay_times = curve.get_day_count_accrual(base_date, pay_days)
    P = np.exp(-curve.current_value(pay_times) * pay_times)
    P0 = float(np.exp(-curve.current_value(np.array([T0]))[0] * T0))
    strike = float((P0 - P[-1]) / (tau * P).sum())
    return {'effective': effective, 'maturity': effective + tenor, 'T0': T0,
            'pay_times': pay_times, 'accruals': tau, 'strike': strike,
            'annuity': float((tau * P).sum()), 'exp_days': exp_days}


def forward_par_swap(reference, currency, curve, effective, maturity, strike,
                     frequency, day_count):
    """A forward-starting payer swap at `strike`, one reset a coupon, whose positive part at its
    effective date is the payer swaption's exercise value."""
    return {
        'Object': 'SwapInterestDeal', 'Reference': reference, 'Currency': currency,
        'Discount_Rate': curve, 'Interest_Rate': curve,
        'Effective_Date': effective, 'Maturity_Date': maturity,
        'Pay_Rate_Type': 'Fixed', 'Pay_Frequency': frequency, 'Pay_Day_Count': day_count,
        'Pay_Interest_Frequency': frequency, 'Pay_Timing': 'End', 'Pay_Payment_Offset': 0,
        'Pay_Accrual_Calendars': None, 'Pay_Payment_Calendars': None,
        'Pay_First_Coupon_Date': None, 'Pay_Penultimate_Coupon_Date': None,
        'Receive_Frequency': frequency, 'Receive_Day_Count': day_count,
        'Receive_Interest_Frequency': pd.DateOffset(months=0), 'Receive_Timing': 'End',
        'Receive_Payment_Offset': 0, 'Receive_Accrual_Calendars': None,
        'Receive_Payment_Calendars': None, 'Receive_First_Coupon_Date': None,
        'Receive_Penultimate_Coupon_Date': None,
        'Index_Tenor': pd.DateOffset(months=0), 'Index_Day_Count': day_count,
        'Index_Frequency': pd.DateOffset(months=0), 'Index_Offset': 0,
        'Index_Calendars': None, 'Index_Publication_Calendars': None,
        'Reset_Type': 'Standard', 'Rate_Multiplier': 1.0, 'Rate_Constant': utils.Percent(0.0),
        'Floating_Margin': 0.0, 'Fixed_Compounding': 'No', 'Compounding_Method': 'None',
        'Known_Rates': None, 'Amortisation': None, 'Swap_Rate': 100.0 * strike, 'Principal': 1.0,
        'Interest_Rate_Volatility': '', 'Discount_Rate_Volatility': ''}


def unit_cashflow(reference, currency, curve, payment_date):
    """One unit of `currency` paid at `payment_date`, discounted on `curve`: in the base currency
    its row-zero mark is the deflator `D_base(0,T)`, which `CreditMonteCarlo` does not report, and in
    the rate currency its row-T mark is `X_T`."""
    return {
        'Object': 'CFFixedInterestListDeal', 'Reference': reference, 'Currency': currency,
        'Discount_Rate': curve, 'Buy_Sell': 'Buy', 'Description': '',
        'Settlement_Date': None, 'Settlement_Amount': 0.0, 'Settlement_Style': 'Physical',
        'Settlement_Amount_Is_Clean': 'Yes', 'Is_Defaultable': 'No', 'Repo_Rate': '',
        'Recovery_Rate': '', 'Survival_Probability': '', 'Investment_Horizon': None,
        'Issuer': '', 'Settlement_Rate': '', 'Calendars': None, 'Rate_Currency': '',
        'Cashflows': {'Compounding': 'No', 'Items': [{
            'Payment_Date': payment_date, 'Notional': 0.0, 'Rate': utils.Percent(0.0),
            'Accrual_Start_Date': payment_date, 'Accrual_End_Date': payment_date,
            'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': 0.0,
            'Fixed_Amount': 1.0, 'Discounted': 'No',
            'FX_Reset_Date': None, 'Known_FX_Rate': 0.0}]}}


def netting_set(reference, children):
    """An uncollateralised netted set over `children`, whose profile `CreditMonteCarlo` reports in
    the set's `Calc_res` where `Results['mtm']` holds only the sum of the sets."""
    return {'Instrument': construct_instrument(
        {'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
         'Collateralized': 'False'}, {}),
        'Children': [{'Instrument': construct_instrument(child, {})} for child in children]}


def composition_config(fitted, ir_curve, fx_currency, fx_spot, benchmarks, base_currency='USD',
                       base_curve='USD', rho_quote=0.4, suppress_quanto_drift=False,
                       fx_axis_sign=1.0):
    """The base-currency world the composition runs in, off the fitted config's price factors: the
    base curve static, the FX rate a `GBMAssetPriceTSModelImplied`, the rate curve a
    `HullWhite2FactorImpliedInterestRateModel` (named behind a `Currency` filter) with its quanto
    correlations re-emitted at `rho_quote` and the `Correlations` rows `quanto_correlations` derives,
    and per benchmark three netting sets: the swap, the deflator and the FX level.

    `suppress_quanto_drift` zeroes the quanto correlations with the rows kept, the measure change
    missing; `fx_axis_sign` of -1 authors the FX spot on the screen's axis."""
    base_date = fitted.params['System Parameters']['Base_Date']
    config = new_config(base_date, base_currency)
    config.params['Price Factors'] = copy.deepcopy(fitted.params['Price Factors'])
    config.params['Price Models'] = copy.deepcopy(fitted.params['Price Models'])
    config.params['Price Factor Interpolation'] = fitted.params['Price Factor Interpolation']

    seed_fx(config, base_currency, base_curve, 1.0)
    seed_fx(config, fx_currency, ir_curve,
            fx_spot if fx_axis_sign > 0 else 1.0 / fx_spot, domestic=base_currency)
    set_fx_ir_correlation(config, base_currency, fx_currency, ir_curve, rho_quote)

    hw_name = utils.check_tuple_name(
        utils.Factor('HullWhite2FactorModelParameters', utils.check_rate_name(ir_curve)))
    hw = reemit_quanto(config.params['Price Factors'][hw_name], base_currency, fx_currency,
                       rho_quote)
    config.params['Price Factors'][hw_name] = hw
    rho_1, rho_2, rho_12 = quanto_correlations(hw)
    if suppress_quanto_drift:
        hw = dict(hw, Quanto_FX_Correlation_1=0.0, Quanto_FX_Correlation_2=0.0)
        config.params['Price Factors'][hw_name] = hw

    config.params['Model Configuration'].append(
        'FxRate', (), 'GBMAssetPriceTSModelImplied')
    config.params['Model Configuration'].append(
        'InterestRate', ('Currency', fx_currency),
        'HullWhite2FactorImpliedInterestRateModel')
    # the process reads its market prices of risk off this block, which nothing defaults
    config.params['Price Models'][
        'HullWhite2FactorImpliedInterestRateModel.{}'.format(ir_curve)] = {
        'Lambda_1': 0.0, 'Lambda_2': 0.0}

    fx_process = 'LognormalDiffusionProcess.{}'.format(fx_currency)
    hw_process = 'HWImpliedInterestRate.{}'.format(ir_curve)
    config.params['Correlations'] = {
        (fx_process, hw_process + '.F1'): rho_1,
        (fx_process, hw_process + '.F2'): rho_2,
        (hw_process + '.F1', hw_process + '.F2'): rho_12}

    children = []
    for name, leg in benchmarks.items():
        children.append(netting_set('NCS_' + name, [forward_par_swap(
            'SWAP_' + name, fx_currency, ir_curve, leg['effective'], leg['maturity'],
            leg['strike'], leg['frequency'], leg['day_count'])]))
        # the deflator: a unit of the BASE currency at the expiry, whose row-zero mark is
        # D_base(0,T) - see `unit_cashflow`
        children.append(netting_set('DF_' + name, [unit_cashflow(
            'DF_' + name, base_currency, base_curve, leg['effective'])]))
        # the FX level: a unit of the RATE currency at the expiry, whose row-T mark is X_T in
        # the reporting currency - see `numeraire_readings`
        children.append(netting_set('FXP_' + name, [unit_cashflow(
            'FXP_' + name, fx_currency, ir_curve, leg['effective'])]))

    config.deals = {'Attributes': {'Reference': 'hw2f_composition'},
                    'Deals': {'Children': children},
                    'Calculation': {'Object': 'CreditMonteCarlo', 'Base_Date': base_date,
                                    'Currency': base_currency}}
    return config, {'rho_F1': rho_1, 'rho_F2': rho_2, 'rho_F1F2': rho_12,
                    'quanto_1': float(hw.get('Quanto_FX_Correlation_1') or 0.0),
                    'quanto_2': float(hw.get('Quanto_FX_Correlation_2') or 0.0),
                    'fx_spot': config.params['Price Factors'][
                        'FxRate.{}'.format(fx_currency)]['Spot']}


def run_composition(config, grid, batch_size, batches, seed, prec=DTYPE):
    """One `CreditMonteCarlo` run: each netting set's `Calc_res['Value']`, its batches joined along
    the path axis, with the report dates and the stats. `Dynamic_Scenario_Dates` keeps its default,
    which puts each benchmark's expiry on the grid."""
    overrides = {
        'Run_Date': pd.Timestamp(config.params['System Parameters']['Base_Date']).strftime(
            '%Y-%m-%d'),
        'Time_grid': grid, 'Batch_Size': batch_size, 'Simulation_Batches': batches,
        'Random_Seed': seed, 'Currency': config.params['System Parameters']['Base_Currency'],
        'Tenor_Offset': 0.0, 'MCMC_Simulations': 1,
        'Deflation_Interest_Rate': config.params['System Parameters']['Base_Currency'],
        'Generate_Cashflows': 'No'}
    started = time.monotonic()
    calc, out = derivus.run_cmc(config, prec=prec, overrides=overrides)
    seconds = time.monotonic() - started

    dates = np.array(sorted(calc.time_grid.mtm_dates))[calc.time_grid.report_index]
    profiles = {}
    for structure in out['Netting'].sub_structures:
        reference = structure.obj.Instrument.field.get('Reference')
        values = structure.obj.Calc_res.get('Value')
        if not values:
            raise RuntimeError('netting set {} reported no Value block'.format(reference))
        profiles[reference] = np.concatenate(values, axis=-1).astype(np.float64)
    return {'profiles': profiles, 'dates': pd.DatetimeIndex(dates), 'seconds': seconds,
            'mtm': out['Results']['mtm'], 'stats': out['Stats'], 'grid': grid,
            'paths': batch_size * batches}


def check_sets_sum_to_the_report(run):
    """The largest absolute difference between the per-set profiles' sum and `Results['mtm']`; a
    scenario-free set, one column a batch, is broadened for the comparison only."""
    rows, cols = run['mtm'].shape
    total = np.zeros((rows, cols))
    for profile in run['profiles'].values():
        wide = profile if profile.shape[1] == cols else np.repeat(
            profile, cols // profile.shape[1], axis=1)
        total[:profile.shape[0]] += wide
    return float(np.abs(total - run['mtm'].values).max())


def identity_readings(run, benchmarks, closure, fx_spot_base_per_unit, numeraire=None):
    """Identity 1 and identity 2 per benchmark, off one run.

    Identity 1 is `D_base(0,T)` times the mean positive expiry row of the swap's set, in the base
    currency, against `X_0` times the Schrager-Pelsser price at theta*: `miss_rel`, and `sigma` in
    per-path standard errors. Given the `numeraire` readings, `residual_rel` divides the payoff-free
    half out, the same reading under the rate currency's T-forward measure. Identity 2 is the
    domestic price against the observed premium."""
    rows = {}
    for name, leg in benchmarks.items():
        swap = run['profiles']['NCS_' + name]
        deflator_profile = run['profiles']['DF_' + name]
        where = np.flatnonzero(run['dates'] == pd.Timestamp(leg['effective']))
        if not where.size:
            raise RuntimeError(
                '{}: the expiry {} is not a reporting row - the grid did not carry it '
                'exactly'.format(name, leg['effective']))
        row = int(where[0])
        if row >= swap.shape[0]:
            raise RuntimeError('{}: expiry row {} beyond the set profile ({} rows)'.format(
                name, row, swap.shape[0]))
        exposure = np.maximum(swap[row], 0.0)
        deflator = float(deflator_profile[0].mean())
        deflator_sd = float(deflator_profile[0].std())
        n = exposure.size
        epe = float(exposure.mean())
        per_path_se = float(exposure.std(ddof=1) / np.sqrt(n))

        sp = closure['process'].schrager_pelsser_swaption(
            leg['T0'], leg['pay_times'], leg['accruals'])
        market = closure['swaps'][leg['benchmark']]
        domestic = float(sp.premium.detach())
        composed = deflator * epe
        target = fx_spot_base_per_unit * domestic
        rows[name] = {
            'row': row, 'date': run['dates'][row], 'paths': n,
            'deflator': deflator, 'deflator_sd': deflator_sd,
            'epe': epe, 'epe_se': per_path_se,
            'deflated_epe': composed, 'deflated_epe_se': deflator * per_path_se,
            'spot_x_sp': target, 'miss': composed - target,
            'miss_rel': (composed - target) / target if target else np.nan,
            'sigma': (composed - target) / (deflator * per_path_se) if per_path_se else np.nan,
            'sp_premium': domestic, 'sp_nvol_bp': 1e4 * float(sp.normal_vol.detach()),
            'market_premium': float(market.price),
            'market_nvol_bp': 1e4 * float(market.market_normal_vol(sp.annuity).detach()),
            'model_vs_market': domestic - float(market.price),
            'model_vs_market_rel': domestic / float(market.price) - 1.0}
        if numeraire is not None:
            # dividing E[D X_T H_T] by the payoff-free E[D X_T] is E^T[H_T], the expectation under
            # the rate currency's T-forward measure, whose target is SP / P_rate(0,T)
            scale = 1.0 + numeraire[name]['numeraire_rel']
            rows[name]['residual_rel'] = (composed / scale) / target - 1.0 if target else np.nan
            rows[name]['numeraire_rel'] = numeraire[name]['numeraire_rel']
    return rows


def numeraire_readings(run, benchmarks, closure, fx_spot_base_per_unit):
    """The payoff-free half of identity 1 per benchmark: one unit of the rate currency at T, worth
    `X_0 P_rate(0,T)`, against `D_base(0,T) E[X_T]`, as `numeraire_rel` and `se_rel`. The FX drift
    accumulates each curve's rate over a step, a discrete rolling bond for the simulated curve and
    the frozen t=0 short rate for a static one, so the reading is a function of the scenario step."""
    rows = {}
    curve, base_date = closure['curve'], closure['base_date']
    for name, leg in benchmarks.items():
        row = int(np.flatnonzero(run['dates'] == pd.Timestamp(leg['effective']))[0])
        deflator = float(run['profiles']['DF_' + name][0].mean())
        fx_level = run['profiles']['FXP_' + name][row]
        T = float(curve.get_day_count_accrual(base_date, leg['exp_days']))
        rate_df = float(np.exp(-curve.current_value(np.array([T]))[0] * T))
        got = deflator * float(fx_level.mean())
        want = fx_spot_base_per_unit * rate_df
        se = deflator * float(fx_level.std(ddof=1)) / np.sqrt(fx_level.size)
        rows[name] = {'row': row, 'deflator': deflator, 'fx_forward': float(fx_level.mean()),
                      'composed': got, 'target': want, 'numeraire_rel': got / want - 1.0,
                      'se_rel': se / want, 'rate_df': rate_df, 'T': T}
    return rows


# ---------------------------------------------------------------------------------------------
# ONE PIPELINE, BOTH CALLERS
# ---------------------------------------------------------------------------------------------

def build_and_fit(base_date, blocks, ir_market_prices, swaption_market_price, vol_surface_name,
                  swaption_currency, fx_currency, ir_curve, fx_spot_base_per_unit,
                  base_currency='USD', base_curve='USD', rho_quote=0.4,
                  fxvol_market_price=None, theta=None):
    """Half A in dependency order: the curves solved, the FX vol curve written, the ladder fitted
    or `theta` installed in its place, and the closure and its readings at theta*."""
    config = new_config(base_date, base_currency)
    seed_fx(config, base_currency, base_curve, 1.0)
    seed_fx(config, fx_currency, ir_curve, fx_spot_base_per_unit)
    install_blocks(config, blocks)

    curves = {name: solve_curve(config, name) for name in ir_market_prices}

    if fxvol_market_price is not None:
        run_bootstrappers(config, ['FXVol'])
        surface = '.'.join(utils.check_rate_name(fxvol_market_price)[1:])
        config.params['Market Prices'][
            'GBMAssetPriceTSModelPrices.{}'.format(fx_currency)] = fx_gbm_block(surface)
        run_bootstrappers(config, ['GBMAssetPriceTSModelParameters'])

    yield_vol_surface(config, vol_surface_name, swaption_currency, 'Normal')
    set_fx_ir_correlation(config, base_currency, fx_currency, ir_curve, rho_quote)
    fit = (fit_swaptions(config, swaption_market_price) if theta is None
           else install_theta(config, swaption_market_price, theta))
    closure = fit_closure(config, swaption_market_price)
    readings = fit_readings(closure)
    return {'config': config, 'curves': curves, 'fit': fit, 'closure': closure,
            'readings': readings}


def refit_at(world, rho_quote, drop_fx_factor=False):
    """The same ladder fitted again on a world differing ONLY in the FX inputs - the fit's own
    invariance, measured rather than argued.

    `rho_quote` of None removes the `Correlation` price factor outright; `drop_fx_factor` removes
    the FX GBM parameters as well, which takes `implied_process` down the branch where there is no
    quanto to suppress at all. theta* must be BIT-IDENTICAL across all three: the seam suppresses
    both FX inputs on the object the objective's process is built on, so the correlation reaches
    nothing the residual reads and the two runs are the same arithmetic on the same sample.
    """
    config = new_config(world['config'].params['System Parameters']['Base_Date'],
                        world['config'].params['System Parameters']['Base_Currency'])
    config.params['Price Factors'] = copy.deepcopy(world['config'].params['Price Factors'])
    config.params['Price Models'] = copy.deepcopy(world['config'].params['Price Models'])
    config.params['Price Factor Interpolation'] = world['config'].params[
        'Price Factor Interpolation']
    config.params['Market Prices'] = copy.deepcopy(world['config'].params['Market Prices'])
    meta = world['meta']
    if drop_fx_factor:
        config.params['Price Factors'].pop(utils.check_tuple_name(utils.Factor(
            'GBMAssetPriceTSModelParameters', utils.check_rate_name(meta['fx_currency']))), None)
    set_fx_ir_correlation(
        config, meta['base_currency'], meta['fx_currency'], meta['ir_curve'], rho_quote)
    # the fit re-seeds off the standing parameters, so the previous solve must not be the seed:
    # both runs have to start where the first one did or "identical" is a statement about the seed
    config.params['Price Factors'].pop(world['fit']['factor'], None)
    return fit_swaptions(config, meta['swaption_market_price'])


def residual_vector(world, rho_quote, drop_fx_factor=False):
    """The objective's OWN residual vector at the STANDING theta*, with the FX inputs set here.

    THE SAME INVARIANCE `refit_at` MEASURES, READ WITHOUT SOLVING FOR IT. `refit_at` answers "does
    the ARGMIN move", which costs a full solve a side; this answers the stronger question "does the
    OBJECTIVE move", at one residual evaluation a side. An argmin that did not move because two
    different objectives happen to share a minimiser would pass the first and fail this; and a
    residual vector that is bit-identical at a common theta makes the argmin identity a corollary
    rather than a coincidence, since two runs then minimise the same function on the same sample.

    IT IS EXACT BY CONSTRUCTION, not by tolerance. `implied_process` builds the objective's process
    on a domestic twin carrying `Quanto_FX_Volatility=None` and `short_rate_fx_correlation=None`,
    so `precalculate` takes the branch that assembles `K` as zeros and the FX/IR correlation reaches
    nothing the residual reads - FOR ALL theta, not just at the solved one. So the right reading is
    `max |delta| == 0.0` and the right assertion is equality.

    THE FITTED FACTOR IS KEPT, which is the opposite of what `refit_at` does and for the opposite
    reason: this comparison needs a COMMON theta, and theta* standing on `Price Factors` is what
    `implied_process` reads its point from.
    """
    meta = world['meta']
    config = new_config(world['config'].params['System Parameters']['Base_Date'],
                        world['config'].params['System Parameters']['Base_Currency'])
    config.params['Price Factors'] = copy.deepcopy(world['config'].params['Price Factors'])
    config.params['Price Models'] = copy.deepcopy(world['config'].params['Price Models'])
    config.params['Price Factor Interpolation'] = world['config'].params[
        'Price Factor Interpolation']
    config.params['Market Prices'] = copy.deepcopy(world['config'].params['Market Prices'])
    if drop_fx_factor:
        config.params['Price Factors'].pop(utils.check_tuple_name(utils.Factor(
            'GBMAssetPriceTSModelParameters', utils.check_rate_name(meta['fx_currency']))), None)
    set_fx_ir_correlation(
        config, meta['base_currency'], meta['fx_currency'], meta['ir_curve'], rho_quote)
    closure = fit_closure(config, meta['swaption_market_price'])
    _, errors = closure['objective'].loss(closure['implied_var'])
    return np.array([float(errors[name].detach()) for name in sorted(errors)], dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------------------------

def print_curve_table(curves):
    print('\nHALF A - CURVE SOLVES (reprice at the solved nodes, through BenchmarkInstruments)')
    print('  {:<28} {:>6} {:>16} {:>14}'.format('block', 'quotes', 'max |PV| (ccy)', 'max |bp|'))
    for name, reading in curves.items():
        print('  {:<28} {:>6} {:>16.3e} {:>14.3e}'.format(
            name, len(reading['pv']), reading['max_abs_pv'], reading['max_abs_bp']))


def print_fit_table(fit, readings, cells=None):
    param = fit['param']
    print('\nHALF A - THE FIT (Objective ABSENT: the declared Analytic default)')
    print('  wall clock              {}'.format(
        'theta* INJECTED (--theta) - no solve ran' if fit.get('injected')
        else '{:.1f} s'.format(fit['seconds'])))
    print('  Alpha_1 / Alpha_2       {:+.10f} / {:+.10f}'.format(
        float(param['Alpha_1']), float(param['Alpha_2'])))
    print('  Correlation             {:+.10f}'.format(float(param['Correlation'])))
    print('  Sigma_1 (bp)            {}'.format(' '.join(
        '{:.1f}'.format(1e4 * v) for v in param['Sigma_1'].array[:, 1])))
    print('  Sigma_2 (bp)            {}'.format(' '.join(
        '{:.1f}'.format(1e4 * v) for v in param['Sigma_2'].array[:, 1])))
    print('  quanto correlations     {} / {}'.format(
        param.get('Quanto_FX_Correlation_1'), param.get('Quanto_FX_Correlation_2')))
    print('  fit rms (normal vol)    {:.3f} bp over {} benchmarks'.format(
        readings['rms_bp'], readings['n']))
    print('  worst benchmark         {} at {:+.3f} bp'.format(
        readings['worst'], readings['worst_bp']))
    print('  honesty reprice         {}'.format(
        fit['honesty'] or ('NOT AVAILABLE - the auditor runs inside the solve'
                           if fit.get('injected') else 'NOT LOGGED')))
    # flush left under its token, so `sed -n 's/^THETA_STAR //p'` lifts it out for `--theta`
    print('{}{}'.format(THETA_LINE, theta_json(param)))
    if cells:
        print('  {:<22} {:>12} {:>12} {:>10}'.format(
            'benchmark', 'model (bp)', 'market (bp)', 'resid (bp)'))
        for name in cells:
            row = readings['rows'][name]
            print('  {:<22} {:>12.2f} {:>12.2f} {:>10.3f}'.format(
                name, row['model_nvol_bp'], row['market_nvol_bp'], row['residual_bp']))


def print_identity_table(label, rows, wiring, run, numeraire=None):
    print('\nHALF B - {}   [{} paths, grid {}, {:.1f} s]'.format(
        label, run['paths'], run['grid'], run['seconds']))
    print('  correlations installed  F1 {:+.10f}   F2 {:+.10f}   F1xF2 {:+.1f}'.format(
        wiring['rho_F1'], wiring['rho_F2'], wiring['rho_F1F2']))
    print('  quanto on the factor    {:+.10f} / {:+.10f}'.format(
        wiring['quanto_1'], wiring['quanto_2']))
    print('  {:<8} {:>12} {:>15} {:>15} {:>12} {:>9} {:>8} {:>10} {:>10}'.format(
        'cell', 'D_base(0,T)', 'deflated EPE', 'X0 x SP', 'miss', 'rel', 'sigma',
        'numeraire', 'residual'))
    for name, row in rows.items():
        print('  {:<8} {:>12.8f} {:>15.10f} {:>15.10f} {:>12.3e} {:>8.3f}% {:>8.2f} '
              '{:>9.3f}% {:>9.3f}%'.format(
                  name, row['deflator'], row['deflated_epe'], row['spot_x_sp'], row['miss'],
                  100.0 * row['miss_rel'], row['sigma'],
                  100.0 * row.get('numeraire_rel', np.nan),
                  100.0 * row.get('residual_rel', np.nan)))
    print('  {:<8} {:>15} {:>15} {:>13} {:>15} {:>10}'.format(
        '', 'se(deflated)', 'se rel', 'model-market', 'model/market', 'SP nvol'))
    for name, row in rows.items():
        print('  {:<8} {:>15.3e} {:>14.3f}% {:>13.3e} {:>14.3f}% {:>9.2f}bp'.format(
            name, row['deflated_epe_se'],
            100.0 * row['deflated_epe_se'] / row['spot_x_sp'], row['model_vs_market'],
            100.0 * row['model_vs_market_rel'], row['sp_nvol_bp']))


def print_numeraire_table(rows):
    print('  {:<8} {:>15} {:>15} {:>10} {:>9}'.format(
        'cell', 'D_base E[X_T]', 'X0 D_rate(0,T)', 'rel', 'se rel'))
    for name, row in rows.items():
        print('  {:<8} {:>15.10f} {:>15.10f} {:>9.3f}% {:>8.3f}%'.format(
            name, row['composed'], row['target'], 100.0 * row['numeraire_rel'],
            100.0 * row['se_rel']))


def print_crn_table(left_label, left, right_label, right):
    print('\nHALF B - RHO-INVARIANCE OF THE COMPOSITION ({} against {}, one Random_Seed)'.format(
        left_label, right_label))
    print('  {:<8} {:>15} {:>15} {:>12} {:>9} {:>9}'.format(
        'cell', left_label, right_label, 'difference', 'rel', 'sigma'))
    for name in left:
        a, b = left[name]['deflated_epe'], right[name]['deflated_epe']
        se = np.hypot(left[name]['deflated_epe_se'], right[name]['deflated_epe_se'])
        print('  {:<8} {:>15.10f} {:>15.10f} {:>12.3e} {:>8.3f}% {:>9.2f}'.format(
            name, a, b, a - b, 100.0 * (a / b - 1.0) if b else np.nan,
            (a - b) / se if se else np.nan))


# ---------------------------------------------------------------------------------------------
# THE BENCHMARK LEGS
# ---------------------------------------------------------------------------------------------

def benchmark_legs(closure, cells, frequency, day_count):
    """`{label: leg}` for `cells`, each strike rebuilt off the solved curve and refused unless
    within 1e-10 of the coupon `create_market_swaps` wrote, so the swap is the swaption's
    underlying."""
    out = {}
    for start, tenor in cells:
        label = '{}x{}'.format(start, tenor)
        name = 'Swaption_{}_{}'.format(start, tenor)
        if name not in closure['swaps']:
            raise RuntimeError('{} is not a benchmark of this ladder'.format(name))
        swap = closure['swaps'][name]
        leg = par_forward_swap_leg(
            closure['curve'], closure['base_date'], pd.DateOffset(**_period(start)),
            pd.DateOffset(**_period(tenor)), frequency, day_count)
        schedule = swap.deal_data.Factor_dep['Cashflows'].schedule
        nonzero = np.flatnonzero(schedule[:, utils.CASHFLOW_INDEX_FixedAmt] != 0.0)
        written = schedule[nonzero, utils.CASHFLOW_INDEX_FixedAmt]
        drift = float(np.abs(-leg['strike'] * leg['accruals'] / written - 1.0).max())
        if drift > 1e-10:
            raise RuntimeError(
                '{}: the strike rebuilt off the curve is {:.3e} away from the coupon '
                'create_market_swaps wrote - the swap is not the swaption\'s '
                'underlying'.format(label, drift))
        leg.update({'benchmark': name, 'frequency': frequency, 'day_count': day_count,
                    'strike_check': drift})
        out[label] = leg
    return out


def _period(text):
    """`'1Y'` / `'18M'` -> the `DateOffset` keywords. One spelling, and it is the config's own."""
    return Config().parse_period(text).kwds


def composition_reading(world, benchmarks, fx_spot, grid, batch, batches, seed,
                        rho_quote, ir_curve='ZAR', fx_currency='ZAR', **mutation):
    """One composition run and everything read off it: the numeraire identity, identity 1,
    identity 2, and the check that the per-set profiles ARE the reported table."""
    config, wiring = composition_config(
        world['config'], ir_curve, fx_currency, fx_spot, benchmarks, rho_quote=rho_quote,
        **mutation)
    run = run_composition(config, grid, batch, batches, seed)
    run['sum_check'] = check_sets_sum_to_the_report(run)
    numeraire = numeraire_readings(run, benchmarks, world['closure'], fx_spot)
    rows = identity_readings(run, benchmarks, world['closure'], fx_spot, numeraire)
    return {'run': run, 'wiring': wiring, 'numeraire': numeraire, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--grids', default='0d 3m(3m),0d 1m(1m),0d 1w(1w)',
                        help='comma-separated scenario grids to sweep')
    parser.add_argument('--batch', type=int, default=16384)
    parser.add_argument('--batches', type=int, default=16)
    parser.add_argument('--seed', type=int, default=5120)
    parser.add_argument('--rho', type=float, default=0.4)
    parser.add_argument('--theta', default=None,
                        help='a theta* block, a file or the JSON inline, injected in place of the '
                             'fit: the line a report prints under THETA_STAR')
    args = parser.parse_args()
    theta = read_theta(args.theta) if args.theta else None

    logging.root.setLevel(logging.INFO)
    logging.basicConfig(format='%(levelname)s %(name)s %(message)s', stream=sys.stdout)
    print('torch {}  device {}'.format(
        torch.__version__,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))

    snapshot = read_snapshot(args.snapshot)
    blocks = snapshot['blocks']
    spot_screen = float(snapshot['spot']['value'])
    base_date = pd.Timestamp(snapshot['manifest']['as_of'])
    fx_spot = 1.0 / spot_screen
    print('\nSNAPSHOT {}'.format(args.snapshot))
    print('  as of {}   USDZAR {} (screen, ZAR per USD)   FxRate.ZAR {:.10f} (USD per ZAR)'.format(
        base_date.date(), spot_screen, fx_spot))
    for name, block in sorted(blocks.items()):
        rows = block['instrument'].get('Points') or block['instrument'].get(
            'Instrument_Definitions') or []
        print('  {:<38} {} rows'.format(name, len(rows)))

    started = time.monotonic()
    world = build_and_fit(
        base_date, blocks,
        ir_market_prices=['InterestRatePrices.USD', 'InterestRatePrices.ZAR'],
        swaption_market_price='HullWhite2FactorModelPrices.ZAR',
        vol_surface_name=blocks['HullWhite2FactorModelPrices.ZAR']['instrument'][
            'Swaption_Volatility'],
        swaption_currency='ZAR', fx_currency='ZAR', ir_curve='ZAR',
        fx_spot_base_per_unit=fx_spot, base_currency='USD', base_curve='USD',
        rho_quote=args.rho, fxvol_market_price='FXVolPrices.USD.ZAR', theta=theta)
    world['meta'] = {'base_currency': 'USD', 'fx_currency': 'ZAR', 'ir_curve': 'ZAR',
                     'swaption_market_price': 'HullWhite2FactorModelPrices.ZAR'}

    print_curve_table(world['curves'])
    cells = ['Swaption_{}_{}'.format(s, t) for s, t in COMPOSITION_CELLS]
    print_fit_table(world['fit'], world['readings'], cells)

    mutations = (('rho = 0.0', dict(rho_quote=0.0)),
                 ('rho absent', dict(rho_quote=None)),
                 ('no FX factor', dict(rho_quote=None, drop_fx_factor=True)))

    # THE RESIDUAL ROUTE, and it is the one that reads: one evaluation of the objective a side, no
    # solve anywhere in it. The vector is what the correlation would have to move to move the
    # argmin, and `residual_vector` says why it cannot.
    print('\nHALF A - RHO-INVARIANCE OF THE FIT, THE RESIDUAL ROUTE (no solve)')
    residual = residual_vector(world, args.rho)
    print('  {:<16} ||r|| {:.12e} over {} benchmarks'.format(
        'at rho = {:g}'.format(args.rho), float(np.sqrt((residual ** 2).sum())), residual.size))
    for label, kwargs in mutations:
        other = residual_vector(world, **kwargs)
        print('  {:<16} residual {:<18} max |delta| {:.3e}'.format(
            label, 'BIT-IDENTICAL' if np.array_equal(residual, other) else 'MOVED',
            float(np.abs(residual - other).max())))

    # THE THETA ROUTE, which is the same statement about the argmin and costs a full solve a side.
    # Under `--theta` it is skipped by name: re-solving three times is exactly the hour injection
    # exists to not re-pay, and the invariance above is the stronger reading anyway.
    print('\nHALF A - RHO-INVARIANCE OF THE FIT, THE THETA ROUTE (a full solve a side)')
    if theta is not None:
        print('  SKIPPED - theta* was injected; three re-solves is the cost --theta declines')
    else:
        reference = theta_vector(world['fit']['param'])
        for label, kwargs in mutations:
            other = refit_at(world, **kwargs)
            vector = theta_vector(other['param'])
            identical = np.array_equal(reference, vector)
            print('  {:<16} theta* {:<18} max |delta| {:.3e}   ({:.1f} s)'.format(
                label, 'BIT-IDENTICAL' if identical else 'MOVED',
                float(np.abs(reference - vector).max()), other['seconds']))

    frequency = blocks['HullWhite2FactorModelPrices.ZAR']['instrument'][
        'Instrument_Definitions'][0]['Fixed_Frequency']
    day_count = blocks['HullWhite2FactorModelPrices.ZAR']['instrument'][
        'Instrument_Definitions'][0]['Fixed_Day_Count']
    benchmarks = benchmark_legs(world['closure'], COMPOSITION_CELLS, frequency, day_count)
    print('\nHALF B - THE TRADE (par forward swaps off the SOLVED curve)')
    print('  {:<8} {:<12} {:<12} {:>9} {:>11} {:>11} {:>11}'.format(
        'cell', 'effective', 'maturity', 'T0', 'strike', 'annuity', 'strike chk'))
    for name, leg in benchmarks.items():
        print('  {:<8} {:<12} {:<12} {:>9.6f} {:>10.6f}% {:>11.6f} {:>11.2e}'.format(
            name, str(leg['effective'].date()), str(leg['maturity'].date()), leg['T0'],
            100.0 * leg['strike'], leg['annuity'], leg['strike_check']))

    for grid in [g.strip() for g in args.grids.split(',')]:
        readings = {}
        for label, rho, mutation in (
                ('rho = {:g}'.format(args.rho), args.rho, {}),
                ('rho = 0', 0.0, {}),
                ('rho = {:g}, K SUPPRESSED'.format(args.rho), args.rho,
                 {'suppress_quanto_drift': True}),
                ('rho = {:g}, FX AXIS FLIPPED'.format(args.rho), args.rho,
                 {'fx_axis_sign': -1.0})):
            reading = composition_reading(
                world, benchmarks, fx_spot, grid, args.batch, args.batches, args.seed, rho,
                **mutation)
            print_identity_table(label, reading['rows'], reading['wiring'], reading['run'],
                                 reading['numeraire'])
            print('  per-set profiles against Results[\'mtm\']: max |diff| {:.3e}'.format(
                reading['run']['sum_check']))
            print('  THE PAYOFF-FREE HALF (one unit of ZAR at T, a tradable):')
            print_numeraire_table(reading['numeraire'])
            readings[label] = reading
        print_crn_table('rho = {:g}'.format(args.rho),
                        readings['rho = {:g}'.format(args.rho)]['rows'],
                        'rho = 0', readings['rho = 0']['rows'])
        for mutant in ('rho = {:g}, K SUPPRESSED'.format(args.rho),
                       'rho = {:g}, FX AXIS FLIPPED'.format(args.rho)):
            print('\n  KILL MAGNITUDE - {}'.format(mutant))
            base = readings['rho = {:g}'.format(args.rho)]['rows']
            for name in base:
                clean, dirty = base[name]['miss_rel'], readings[mutant]['rows'][name]['miss_rel']
                se = base[name]['deflated_epe_se'] / base[name]['spot_x_sp']
                print('    {:<8} identity {:+8.3f}% -> {:+12.3f}%   moved {:+11.3f}% '
                      '({:.1f} sigma)'.format(name, 100.0 * clean, 100.0 * dirty,
                                              100.0 * (dirty - clean), abs(dirty - clean) / se))

    print('\nTOTAL WALL {:.1f} s'.format(time.monotonic() - started))


if __name__ == '__main__':
    main()
