"""The HW2F composition harness: calibrate domestic, simulate global, and read the seam.

The measure architecture that landed on 2026-08-31 says two things at once, and only a
composition can hold both up at the same time:

  A  A ZAR swaption ladder is fitted under the ZAR risk-neutral measure whatever the job's base
     currency, because that is the measure the premium is quoted in. The FIT is therefore
     INVARIANT to the FX/IR correlation.
  B  A ZAR curve simulated inside a USD-based job carries the quanto drift `K`, assembled from
     the very parameters the domestic fit produced. The COMPOSITION is therefore NOT invariant to
     that correlation - and yet the USD-deflated, FX-converted price it produces must land on the
     domestic price the fit itself reports.

Both halves are one Girsanov statement: drifts move, quadratic variation does not. This module
measures each half on live prints and reports the numbers under that sentence.

THIS IS AN ACCEPTANCE INSTRUMENT, on the `gates/recompute_inner_mc.py` precedent: it takes a
snapshot directory and PRINTS reading tables rather than asserting. The snapshot never enters the
repository and neither do the prints - what the run establishes belongs in the docs. The module
itself is tracked (unlike most of `gates/`) for one reason: `tests/test_hw2f_composition.py` is
its CANNED twin and imports the functions below, so the pipeline has one spelling and the gate is
what says the spelling is right.

    PYTHONIOENCODING=utf-8 python gates/hw2f_composition.py --snapshot DIR

THE FIT IS 62-71 MINUTES AND IT IS DETERMINISTIC, so `--theta` exists and is SOUND. Half A's own
gate is that the same snapshot at the same seed re-solves to a BIT-IDENTICAL theta* - measured
across independent processes and across an engine bump - which is exactly the statement that the
solve carries no information a second solve would add. The instrument persists nothing, so every
independent verification otherwise re-pays that hour to arrive at a number already published; the
report therefore PRINTS theta* on one machine-readable line (`THETA_STAR {...}`, the engine's own
JSON codec) and `--theta` reads that line back - as a file or inline - and runs everything
DOWNSTREAM of it: the closure, the fit readings, the residual-route rho-invariance, and the whole
of half B. What injection cannot reproduce is the solve's own by-products - the wall clock and the
engine's honesty reprice - and the fit table says INJECTED rather than reporting them as zero.

    PYTHONIOENCODING=utf-8 python gates/hw2f_composition.py --snapshot DIR --theta theta.json

THE PUBLISHED TABLES WERE READ AT 131,072 PATHS, and that is NOT this file's default (16384 x 16 =
262,144). Nothing else records the invocation, so it is recorded here: `--batch 16384 --batches 8`
is the sample the reported half-B tables came off, and a bare `--snapshot DIR` doubles it (and the
wall clock with it) rather than reproducing them.

WHAT THE ENGINE DOES NOT REPORT, and how this reads it anyway. `Credit_Monte_Carlo` reports the
exposure profile UNDEFLATED (`Results['mtm']`, reporting currency, report grid) and applies
`Deflation_Interest_Rate` only inside the CVA/FVA/CollVA reductions, which are scalars - there is
no deflator series and no deflated profile among its output keys, so a deflated expiry-row EPE
cannot be read off the reported tables. Rather than patch the engine to publish one, the deflator
is obtained from the engine's OWN table: an extra netting set per benchmark holding a single
base-currency cashflow of 1.0 at that benchmark's expiry, whose row-zero mark IS `D_base(0,T)`.
The remedy the finding names is to publish `Dt_T` beside `mtm`; nothing here does that.

WHAT THE RUN FOUND, so a reader knows what these tables are for. Identity 1 does not close on its
own, and the miss is NOT the measure change: it factors into a payoff-free NUMERAIRE identity -
one unit of the rate currency at T is a tradable worth `X_0 P(0,T)` - times a residual, and the
first factor is the discretely-rolled money market the FX drift accumulates. So the reading is a
function of the scenario STEP, and `--grids` sweeps that axis rather than picking a rung of it.
Two mechanisms live in there and they behave oppositely (see `numeraire_readings`): the SIMULATED
leg's gap shrinks like sqrt(dt), while the STATIC leg's GROWS as the grid refines, because a
frozen curve does not roll and a finer step reads a shorter rate off it.

THE NUMERAIRE IS *A* TERM IN THE MISS, SIZED PER CELL - not the whole of it. On the live weekly
sweep the payoff-free factor is 94% of identity 1's miss at 1Yx2Y (-0.455 of -0.481), 30% at
2Yx5Y (-1.531 of -5.090) and 51% at 5Yx5Y (-4.735 of -9.354), so a sentence that reads "the miss
is a numeraire term rather than a measure one" is true of the short cell and false of the middle
one. What separates a WRONG DRIFT from a COARSE NUMERAIRE is neither of those readings but the
ZAR T-forward martingale, which needs no analytic price and no deflator at all: a par-struck
forward swap has `E^T[V(T)] = 0` exactly, and that expectation is `mean(swap row)/mean(FX row)`.
It closes to +0.27 sigma at 2Yx5Y where identity 1's own residual is -3.6% (~10 sigma), so that
residual is provably NOT a drift error - it is distributional and step-dependent; it is 3.1 sigma
short at 5Yx5Y at the finest grid and halves with the step, which is a converging discretisation
rather than a structural miss; and it moves 15-35 sigma when `K` is suppressed, which is what says
the drift is real, correctly signed and load-bearing. The composition's own rho-invariance and the
two mutations are clean throughout, which is what says the seam itself is right.
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
from derivus.instruments import construct_instrument, generate_dates_backward

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
    """An empty job config at `base_date`, based in `base_currency`.

    Nothing is bootstrapped and no factor is authored: `seed_fx` and `install_blocks` put the
    inputs in, and the bootstrap verb turns them into price factors.
    """
    config = Config(base_currency=base_currency)
    config.params['System Parameters']['Base_Date'] = pd.Timestamp(base_date)
    config.params['System Parameters']['Base_Currency'] = base_currency
    config.params['Market Prices'] = {}
    config.params['Bootstrapper Configuration'] = {}
    return config


def seed_fx(config, currency, discount_curve, spot, domestic=None):
    """One `FxRate` price factor.

    THE AXIS IS BASE-PER-UNIT, which is what `FxRate.Spot`'s own descriptor says ("Spot rate in
    base currency"): the number stored is the value of ONE unit of `currency` measured in the job's
    base currency. A USDZAR screen print of 16.1227 is ZAR per USD, so under a USD base the ZAR
    factor carries its RECIPROCAL. Getting this the wrong way round is a 260x error that still
    prices, which is why `fx_axis_sign` is a mutation the harness runs rather than a comment.

    `domestic` IS REQUIRED WHERE THE RATE IS SIMULATED and harmless where it is not.
    `GBMAssetPriceTSModelImplied.calc_references` builds the drift as
    `r(domestic) - r(this currency)` and reads the first through `get_domestic_currency(None)`,
    which does not survive a `None` on both sides: an FX factor left with `Domestic_Currency: None`
    and given a process raises `AttributeError: 'NoneType' object has no attribute 'split'` out of
    `check_rate_name`. The base currency's own factor keeps `None`, since `find_models` excludes it
    from the stochastic set by name.
    """
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
    """Author the `InterestYieldVol` surface a swaption block names.

    THE SURFACE SUPPLIES THE CONVENTION AND THE ROWS SUPPLY THE VOLS. `create_market_swaps` reads
    `Market_Volatility` off each `Instrument_Definitions` row and reads the surface for exactly
    three things: its declared `Distribution_Type` (which picks the `PREMIUM_CONVENTIONS` pair -
    Bachelier under `'Normal'`), its displacement, and whether a premiums file was attached. So the
    quads authored here are never priced against; they exist because the factor has to construct.
    A DISPLACEMENT IS NOT AUTHORED: `Shift` is left at the field's own zero, and a non-zero one
    beside `'Normal'` refuses by name in `InterestYieldVol.displacement`.
    """
    quads = [[m, e, t, 0.01] for t in (1.0, 5.0, 10.0) for e in (0.25, 1.0, 5.0, 10.0)
             for m in (-0.01, 0.0, 0.01)]
    config.params['Price Factors']['InterestYieldVol.{}'.format(name)] = {
        'Property_Aliases': None, 'Currency': currency, 'Distribution_Type': distribution,
        'Shift': utils.Percent(0), 'Surface': utils.Curve([], quads)}


def fx_gbm_block(surface_name):
    """The `GBMAssetPriceTSModelPrices` block that turns an FX vol SURFACE into the integrated vol
    curve a `GBMAssetPriceTSModelImplied` process reads.

    The snapshot emits the surface (`FXVolPrices`) and stops there, because the ATM column is a
    derived quantity and the emitter's job is quotes. This block is the one authored input of the
    composition's FX leg, and it names nothing but the surface: `GBMAssetPriceTSModelParameters`
    reads the ATM column off it and writes the curve, and because the underlying is an FX rate the
    equity-side quanto correlation it would otherwise look up is held at zero.
    """
    return {'instrument': {'Asset_Price_Volatility': surface_name, 'Quote_Sensitivity': 'No'}}


def set_fx_ir_correlation(config, base_currency, currency, ir_curve, value):
    """The FX/IR correlation the CALIBRATION reads, or remove it when `value` is None.

    `RiskNeutralInterestRateModel.implied_process` reads a `Correlation` PRICE FACTOR named
    `Correlation.FxRate.<sorted pair>/InterestRate.<curve>` - not the `Correlations` SECTION, which
    is the simulator's. It then FLIPS THE SIGN when the sorted pair starts with the base currency,
    because the quote is on the pair as a desk names it (ZAR per USD) while the simulated `FxRate`
    is base-per-unit (USD per ZAR). So the number authored here is the desk's, and `C` inside the
    calibration is its negative.
    """
    pair = '.'.join(sorted((base_currency, currency)))
    name = 'Correlation.FxRate.{}/InterestRate.{}'.format(pair, ir_curve)
    if value is None:
        config.params['Price Factors'].pop(name, None)
    else:
        config.params['Price Factors'][name] = {'Value': float(value)}
    return name


def run_bootstrappers(config, families):
    """Run exactly `families`, in the order given, through the Context bootstrap verb.

    ORDER IS NOT ALPHABETICAL AND HAS TO BE FORCED. `Config.bootstrap` iterates
    `sorted(self.params['Bootstrapper Configuration'])`, which puts `InterestRateCurveParameters`
    LAST - behind the swaption fit that reads the curve it builds. So the section is set to one
    family at a time and the verb is called once per stage: the curves, then the FX surface and its
    GBM parameters, then the swaption fit. Every stage is the library's own bootstrap; nothing here
    reaches past it.
    """
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
    """Bootstrap one `InterestRatePrices` block and report what the solve left on the table.

    The residual is the benchmark set RE-PRICED at the solved nodes, through
    `BenchmarkInstruments` - the same compiled deals the solve differentiated. A PV is reported
    twice: in the block's own currency units, and as the PAR-RATE move that would close it. The
    second reading needs no root find, because a benchmark's PV is AFFINE in its quote: pricing the
    set once at the quotes and once at the quotes plus one percent locates the par rate exactly,
    which is the same identity `InterestRateCurveParameters.seed` leans on.

    THE BOOTSTRAP VERB RUNS THE WHOLE FAMILY, not one block, so calling this once per block
    re-solves every block each time. That is deliberate rather than tolerated: selecting a single
    block would mean reaching past `Config.bootstrap` into the family's own dependency ordering,
    and the family is what decides which curves are coupled. The cost is one extra solve of each
    strip (12.5 s USD, 7.8 s ZAR live), and the solve is idempotent.
    """
    block = config.params['Market Prices'][market_price]['instrument']
    run_bootstrappers(config, ['InterestRateCurveParameters'])

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
    """Run the HW2F swaption calibration and return theta*, the wall clock and the engine's log.

    `Objective` IS DELIBERATELY ABSENT from the block: the declared default is `'Analytic'` since
    the 2026-08-31 flip, and a harness that spelled it would stop reading the default the day it
    moved. What comes back is what `save_params` wrote - plain numpy on the price factor - plus the
    lines the bootstrap logged, which is where the honesty reprice lives.
    """
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
    """theta* as one flat float64 array, in the closure's own parameter order.

    `SwaptionCalibration.keys` is `implied_var`'s insertion order, which is
    `HullWhite2FactorModelParameters.current_value()`'s - Alpha_1, Sigma_1, Alpha_2, Sigma_2,
    Correlation. The bit-identity comparison is made on THIS vector rather than on a dict of
    arrays, so two solves that differ in one sigma knot differ in one float.
    """
    return np.concatenate([
        np.atleast_1d(np.float64(param['Alpha_1'])),
        param['Sigma_1'].array[:, 1].astype(np.float64),
        np.atleast_1d(np.float64(param['Alpha_2'])),
        param['Sigma_2'].array[:, 1].astype(np.float64),
        np.atleast_1d(np.float64(param['Correlation']))])


#: The one-line prefix the report writes theta* under and `--theta` reads it back from. A token
#: rather than prose so a verifier can lift theta* out of a log with a single grep.
THETA_LINE = 'THETA_STAR '
#: Every field `save_params` writes onto the parameter block, in its own order. The whole block is
#: emitted rather than `theta_vector`'s flat form, because the SIMULATOR reads more than the solver
#: solved for: `Quanto_FX_Volatility` and the two quanto correlations are emission, not fit, and a
#: composition run off a theta* missing them is a world with no drift in it.
THETA_FIELDS = ('Property_Aliases', 'Quanto_FX_Volatility', 'Alpha_1', 'Sigma_1', 'Alpha_2',
                'Sigma_2', 'Correlation', 'Quanto_FX_Correlation_1', 'Quanto_FX_Correlation_2')


def theta_json(param):
    """theta* as ONE LINE of the engine's own JSON - what the report prints and `--theta` reads.

    `CustomJsonEncoder` IS THE SPELLING, so a `Sigma_1` knot leaves as `.Curve` and comes back
    through `Config.read_json` as the `utils.Curve` the price factor held - one codec, the same one
    `read_snapshot` decodes a Market Prices block with, rather than a second serialiser written
    here that would have its own idea of a curve.

    THE SCALARS ARE COERCED and that is not tidying: `get_quanto_correlation` returns numpy floats,
    and a `np.float64` is not JSON-serialisable - it falls through to the encoder's `default`, which
    LOGS an error and emits `{'.Unknown': ...}` rather than raising. A theta* line carrying two
    `.Unknown` quanto correlations would read back as a dict, price without complaint, and be a
    world with no quanto drift in it.
    """
    emitted = {}
    for key in THETA_FIELDS:
        if key not in param:
            continue
        value = param[key]
        emitted[key] = float(value) if isinstance(
            value, (int, float, np.floating, np.integer)) else value
    return json.dumps(emitted, cls=CustomJsonEncoder)


def read_theta(text):
    """`--theta`'s argument as a parameter block: a path to a file, or the JSON itself inline.

    Both routes go through `Config.read_json`, which takes a filename or a `(text, name)` tuple -
    the engine's own two entry points to one decoder. What comes back is checked for the fields the
    solve produces, because the failure that matters is a well-formed JSON object that is not a
    theta*: it would install silently and every reading downstream would be of some other world.
    """
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
    """A theta* block WRITTEN onto `Price Factors` in place of a solve.

    Returns the shape `fit_swaptions` returns, so nothing downstream can tell an injected theta*
    from a solved one - which is the point, since the two ARE the same numbers. What injection
    cannot manufacture is the solve's by-products: `seconds` is zero and there is no honesty
    reprice, so the block is flagged `injected` and the fit table says so rather than reporting a
    zero wall clock and a missing auditor as if the fit had run and been silent.
    """
    name = utils.check_tuple_name(utils.Factor(
        'HullWhite2FactorModelParameters', utils.check_rate_name(market_price)[1:]))
    config.params['Price Factors'][name] = copy.deepcopy(theta)
    return {'factor': name, 'param': config.params['Price Factors'][name], 'seconds': 0.0,
            'honesty': None, 'log': [], 'injected': True}


def fit_closure(config, market_price, device=torch.device('cpu'), dtype=DTYPE):
    """The calibration's own residual closure standing at the theta* on the price factor.

    Built exactly as `RiskNeutralInterestRateModel.bootstrap` builds it - `implied_process` then
    `calc_loss_on_ir_curve` - so the residual read here is the residual the solve stopped on.
    THE SEED IS theta*: `implied_process` reads its initial guess off `Price Factors`, which the
    fit has just written, and both the interpolation onto the vol tenors and the clip to the
    solver's own bounds are identities there. Nothing is patched and nothing is set by hand.
    """
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
    time_grid.set_base_date(base_date, delta=(10, vol_tenors * utils.DAYS_IN_YEAR))
    implied_var, objective, swaps, _ = boot.calc_loss_on_ir_curve(
        {'instrument': block}, base_date, time_grid, process, implied_obj, ir_factor, surface)
    # THE CLOSURE IS RETURNED READY, and that is not a convenience. `schrager_pelsser_swaption`
    # reads J, the reversion speeds and the correlation off `precalculate` and REFUSES BY NAME
    # when none has run; `calc_loss_on_ir_curve` builds the closure without running one. Every
    # consumer here goes on to price the analytic swaption, so the one evaluation happens once,
    # at theta*, rather than being a precondition each caller has to remember.
    objective.loss(implied_var)
    return dict(model=boot, process=process, implied_var=implied_var, objective=objective,
                swaps=swaps, curve=ir_curve, time_grid=time_grid, block=block,
                surface=surface, implied_obj=implied_obj, base_date=base_date)


def fit_readings(closure):
    """Per-benchmark model and market NORMAL vols at theta*, and the fit's own summary.

    The residual is the objective's own - `market_swap_class.normal_vol_error`, a weighted
    difference of ABSOLUTE normal vols - so the rms reported here is the rms of exactly what the
    solve minimised, in the units the ladder was quoted in. Reported in basis points, which is what
    a desk compares.
    """
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
# THE QUANTO SEAM - two halves, authored in two places, and nothing checks they agree
# ---------------------------------------------------------------------------------------------

def reemit_quanto(param, base_currency, currency, rho_quote):
    """The fitted parameter block with `Quanto_FX_Correlation_1/2` RE-EMITTED at `rho_quote`.

    THE FIT IS RHO-INVARIANT AND THE EMISSION IS NOT, and this is the seam where that shows.
    `save_params` writes the two quanto correlations off `get_quanto_correlation`, which is a
    function of the SOLVED sigmas and correlation AND of the FX/IR correlation standing in
    `Price Factors` at the moment of the solve. A composition run at a different rho therefore
    needs them re-derived rather than carried over - carrying them over is a world whose drift
    belongs to one correlation and whose covariance to another, and it is invisible: theta* is
    identical either way, so nothing about the parameter block looks stale.

    The re-derivation is the ENGINE'S OWN function, called on an object carrying the same
    `short_rate_fx_correlation` the calibration would have read - including the sign flip that
    `implied_process` applies when the sorted pair starts with the base currency.
    """
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
    """The `Correlations`-section entries that make the SIMULATOR's covariance agree with the drift
    the fitted factor already carries: `(rho_F1, rho_F2, rho_F1F2)`.

    THE TWO HALVES OF A QUANTO ARE AUTHORED IN TWO DIFFERENT PLACES AND IN TWO DIFFERENT BASES.
    `save_params` writes `Quanto_FX_Correlation_1/2` onto the price factor, and `precalculate`
    multiplies them into `K` - that is the DRIFT half, and it is stated in the model's own basis:
    rho-bar_i = corr(dW, dW_i), with `dW_1, dW_2` the pair correlated at rho. The `Correlations`
    SECTION is the covariance half, and its rows are `HWImpliedInterestRate.<curve>.F1` and `.F2`,
    which are the INDEPENDENT normals the process's own cholesky of `delta_CtT` consumes -
    `calc_factors` builds `Y_1` from row F1 alone and `Y_2` from `C10*F1 + C11*F2`. So an author
    who copies rho-bar_2 into the section has installed a drift and a covariance that disagree, and
    nothing in the library says so.

    THE CHANGE OF BASIS, once. Writing `a = corr(Z_fx, Z_1)`, `b = corr(Z_fx, Z_2)`:

        corr(dW, dW_1) = a
        corr(dW, dW_2) = rho * a + sqrt(1 - rho^2) * b

    since over a step `corr(Y_1, Y_2) -> rho` exactly (`J_12/sqrt(J_11 J_22) -> 1` as the step
    shrinks with locally constant integrands). Setting those to rho-bar_1, rho-bar_2 gives

        a = rho-bar_1,    b = (rho-bar_2 - rho * rho-bar_1) / sqrt(1 - rho^2)

    and the pair is CONSISTENT BY CONSTRUCTION with the single-index form `get_quanto_correlation`
    emits: substituting rho-bar_i = C(s_i + rho s_j)/D gives `a^2 + b^2 = C^2` exactly, so the FX
    Brownian keeps `sqrt(1 - C^2)` of its own independent part and `C` is recovered as the
    instantaneous FX/short-rate correlation it was authored as.

    F1 AGAINST F2 IS ZERO. The process applies rho itself, inside `delta_CtT`; correlating the two
    raw rows as well would apply it twice.
    """
    rho = float(param['Correlation'])
    bar1 = float(param.get('Quanto_FX_Correlation_1') or 0.0)
    bar2 = float(param.get('Quanto_FX_Correlation_2') or 0.0)
    span = np.sqrt(max(1.0 - rho * rho, 1e-12))
    return bar1, (bar2 - rho * bar1) / span, 0.0


# ---------------------------------------------------------------------------------------------
# HALF B - THE COMPOSITION
# ---------------------------------------------------------------------------------------------

def par_forward_swap_leg(curve, base_date, start, tenor, frequency, day_count):
    """The forward-starting swap's own fixed leg and its par rate, off the SOLVED t=0 curve.

    THE CLOCK IS THE CURVE'S OWN DAY COUNT, not `utils.DAYS_IN_YEAR`: `read_cache` builds the grid
    `J` is integrated on with `factor.get_day_count_accrual`, and `schrager_pelsser_swaption` reads
    its expiry against that grid. The two differ by 7e-4 years at a 1Y expiry, which is enough to
    miss the node the benchmark put there.

    The par rate is the analytic swaption's OWN at-the-money rate, `(P(0,T0) - P(0,Tn))/A(0)`, so
    the swap this strikes is the swaption's underlying and not a near neighbour of it.
    """
    effective = base_date + start
    dates = generate_dates_backward(effective + tenor, effective, frequency)
    cash = utils.generate_fixed_cashflows(
        base_date, dates, 1.0, None, utils.get_day_count(day_count), 0.0)
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
    """A forward-starting payer swap: pay fixed at `strike`, receive the single-reset float leg.

    Its value at the expiry row is `A(T0)(S(T0) - K)`, so `relu` of that row IS the payer
    swaption's exercise value - which is the whole reason the exposure profile of this deal can be
    read against a swaption price at all. `Index_Tenor` of zero months is what gives each coupon
    ONE reset spanning its own accrual period, the vanilla shape the ladder's conventions describe.
    `Swap_Rate` is quoted in PERCENT, which is what `SwapInterestDeal` divides by 100.
    """
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
    """One unit of `currency` paid at `payment_date`, discounting on `curve`.

    ITS ROW-ZERO MARK IS THE DEFLATOR. `Credit_Monte_Carlo` publishes no deflator series and no
    deflated profile (see the module docstring), so `D(0,T)` is read off the engine's own reported
    table by putting a deal in the book whose t0 value is exactly that number. Notional is zero and
    the whole payment rides `Fixed_Amount`, so no rate, accrual or day count reaches the answer.
    It is not a substitute for the engine's own quantity but a second reading of it: the
    non-stochastic branch at `derivus/calculation.py:1514-1516` computes `Dt_T` as the deterministic
    t=0 base-curve discount factors, and this mark reproduces that number exactly (cross-path sd
    exactly 0.0, the base curve being static).

    THE DEFLATOR AND THE NUMERAIRE THE FX DRIFT ROLLS ARE TWO DIFFERENT OBJECTS, and confusing them
    is what identity 1's whole residual is about. This reading is `D(0,T) = exp(-r(0,T) T)`, off the
    t=0 curve. What the simulation actually divides by is what
    `GBMAssetPriceTSModelImplied.generate` accumulates in the drift (`stochasticprocess.py:800-803`,
    `cumsum(r_base dt - r_rate dt)` off each curve's SCENARIO view), which for a STATIC base curve
    is `T x r(0, dt)` - a frozen short rate rolled forward, not `-log D(0,T)`. Identity 1 as posed
    can only close where those two agree: the canned twin's FLAT USD curve makes them equal by
    construction, which is why that choice is load-bearing rather than cosmetic, and the live long
    cells are exactly where they do not agree.
    """
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
    """An uncollateralised, netted set over `children` - the container whose own `Calc_res` block
    carries a per-benchmark profile.

    A BARE DEAL REPORTS ONE ROW. `Credit_Monte_Carlo` sums every netting set into the reported
    `mtm` table, so a book of three benchmarks in ONE set would report their sum and nothing else;
    a set per benchmark is what makes each profile separately readable, off `Netting`, which the
    calculation returns beside `Results`.
    """
    return {'Instrument': construct_instrument(
        {'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
         'Collateralized': 'False'}, {}),
        'Children': [{'Instrument': construct_instrument(child, {})} for child in children]}


def composition_config(fitted, ir_curve, fx_currency, fx_spot, benchmarks, base_currency='USD',
                       base_curve='USD', rho_quote=0.4, suppress_quanto_drift=False,
                       fx_axis_sign=1.0):
    """The USD-based world the composition runs in, built off the FITTED config's price factors.

    THE WIRING, factor by factor. The USD curve is STATIC - it is the numeraire and nothing asks it
    to move - so `Model Configuration` names `HullWhite2FactorImpliedInterestRateModel` behind a
    `Currency` FILTER rather than as a default: a default would resolve the USD curve too and
    `find_models` would log an error about the parameters it cannot find. The ZAR FX rate is a
    `GBMAssetPriceTSModelImplied` off the integrated vol curve the FX-vol bootstrap wrote; the base
    currency's own `FxRate` is excluded by `find_models` itself, which is why no USD FX process
    appears. The ZAR curve simulates off `HullWhite2FactorModelParameters.ZAR` exactly as emitted,
    quanto drift included.

    `suppress_quanto_drift` is THE MUTATION, and it is a document rather than a patch: the emitted
    `Quanto_FX_Correlation_1/2` are authored to zero while the `Correlations` section keeps the
    correlated Brownians. That is a world where the FX and the rates move together and the measure
    change is missing - precisely the failure a wrong `K` would be - and it is a world a desk could
    author by hand.

    `fx_axis_sign` of -1 is THE OTHER MUTATION: the FX factor authored on the screen's own axis
    (ZAR per USD) instead of the engine's (USD per ZAR).
    """
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
    # THE MARKET PRICES OF RISK ARE A `Price Models` ENTRY AND THE ENGINE DOES NOT DEFAULT THEM.
    # `Config.find_models` classifies an implied model off the implied FACTOR and needs no
    # `Price Models` block - `calc_lifecycle.md` states that as an invariant - but
    # `HullWhite2FactorImpliedInterestRateModel.precalculate` then reads `self.param['Lambda_1']`
    # unguarded off exactly that missing block. Omitting this line raises
    # `TypeError: 'NoneType' object is not subscriptable` from inside a precalculate, which names
    # neither the field nor the factor. Reported as a finding; authored here.
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

    config.deals = {'Attributes': {'Reference': 'hw2f_composition', 'Tag_Titles': ''},
                    'Deals': {'Children': children},
                    'Calculation': {'Object': 'CreditMonteCarlo', 'Base_Date': base_date,
                                    'Currency': base_currency}}
    return config, {'rho_F1': rho_1, 'rho_F2': rho_2, 'rho_F1F2': rho_12,
                    'quanto_1': float(hw.get('Quanto_FX_Correlation_1') or 0.0),
                    'quanto_2': float(hw.get('Quanto_FX_Correlation_2') or 0.0),
                    'fx_spot': config.params['Price Factors'][
                        'FxRate.{}'.format(fx_currency)]['Spot']}


def run_composition(config, grid, batch_size, batches, seed, prec=DTYPE):
    """One `CreditMonteCarlo` run; the per-netting-set profiles, the report dates and the stats.

    THE PER-SET PROFILES ARE THE ENGINE'S OWN. `Credit_Monte_Carlo` returns `Netting` - the root
    `DealStructure` - beside `Results`, and each sub-structure carries its own `Calc_res['Value']`:
    one array per batch, written by `pricing.interpolate` on the way out of the netting set's
    `post_process`. Nothing is re-priced or re-derived here; the arrays are concatenated along the
    path axis and the reported root `mtm` is checked against their sum, which is what says the
    reading is the same number the calculation reported.

    `Dynamic_Scenario_Dates` is LEFT AT THE ENGINE'S OWN DEFAULT ('Yes'), which unions the deals'
    own reset dates into both grids - that is what puts each benchmark's expiry on the grid
    EXACTLY rather than near it.
    """
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
    """The per-set profiles against `Results['mtm']`: the reading is the reported table or it is
    nothing. Returns the max absolute difference, which is 0.0 when they are the same numbers.

    A set with no scenario dependence (the unit cashflows) prices at width 1 per batch and comes
    back `(rows, batches)` rather than `(rows, paths)`; it is broadened here for the comparison
    only, never for a reading."""
    rows, cols = run['mtm'].shape
    total = np.zeros((rows, cols))
    for profile in run['profiles'].values():
        wide = profile if profile.shape[1] == cols else np.repeat(
            profile, cols // profile.shape[1], axis=1)
        total[:profile.shape[0]] += wide
    return float(np.abs(total - run['mtm'].values).max())


def identity_readings(run, benchmarks, closure, fx_spot_base_per_unit, numeraire=None):
    """Identity 1 and identity 2, per benchmark, off one run.

    IDENTITY 1 - THE COMPOSITION AGAINST THE MODEL'S OWN DOMESTIC PRICE. Under the base measure a
    ZAR payoff `H_T` is worth `E[D_usd(0,T) X_T H_T]`, and under the ZAR measure the same payoff is
    `E[D_zar(0,T) H_T]`; the change of measure says the first is `X_0` times the second. The left
    side is read here as `D_usd(0,T)` (the unit-cashflow set's row-zero mark) times the mean over
    paths of the swap set's positive expiry row - which is already in USD, because the netting set
    marks in the reporting currency and the reporting currency is the base one. The right side is
    `X_0` times `schrager_pelsser_swaption` at theta*. This is MODEL-INTERNAL: both sides are the
    same fitted parameters, so nothing about the market can move it, and it is the test of `K` - a
    wrong drift moves it linearly in the correlation.

    IT DOES NOT CLOSE ON ITS OWN, AND WHAT IS LEFT OVER IS NOT THE MEASURE. Identity 1 factors as
    the payoff-free numeraire identity times a residual, and the first factor is a discretisation:
    the FX drift accumulates a discretely-rolled money market in both currencies (see
    `numeraire_readings`), so the whole reading is a function of the scenario STEP. `miss_rel` is
    what a job would report.

    `residual_rel` IS THE ZAR T-FORWARD IDENTITY, and that is a stronger statement than "the
    numeraire divided out". Dividing the composed price by the payoff-free reading is dividing
    `E[D_usd(0,T) X_T H_T]` by `E[D_usd(0,T) X_T]`, which IS the expectation of `H_T` under the ZAR
    T-forward measure - the ratio the numeraire pair defines, with every deflator and every rolled
    bond cancelling between the two. Its target, `X_0 SP / (X_0 P_zar(0,T))`, is `SP / P_zar(0,T)`:
    the analytic premium in T-forward units. Computed that way directly - `E[D X_T V+]/E[D X_T]`
    against `SP/P_zar(0,T)`, no numeraire factorisation anywhere in it - it reproduces the number
    below at rel +0.000e+00 on both canned cells. So this is not a miss with a fudge applied; it is
    a second identity, stated under the measure the pair of readings actually defines.

    THE STANDARD ERROR is the per-path one, `std(relu(V))/sqrt(N)`: the outer draws are pseudo-
    random `torch.randn`, i.i.d. across paths and across batches, so a batch-mean error bar would
    be the same quantity with fewer degrees of freedom.

    IDENTITY 2 - MODEL AGAINST MARKET. The same domestic price against the observed premium, which
    is `create_market_swaps`' own Bachelier at the solved curve's annuity and forward. It carries
    the fit rms and is an HONESTY STATEMENT rather than a pass: a model that reproduced every quote
    exactly would be a model with one parameter per quote.
    """
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
            # the SAME reading under the ZAR T-forward measure: dividing by the payoff-free half
            # is dividing E[D X_T H_T] by E[D X_T], and the ratio IS E^T[H_T] - see the docstring.
            scale = 1.0 + numeraire[name]['numeraire_rel']
            rows[name]['residual_rel'] = (composed / scale) / target - 1.0 if target else np.nan
            rows[name]['numeraire_rel'] = numeraire[name]['numeraire_rel']
    return rows


def numeraire_readings(run, benchmarks, closure, fx_spot_base_per_unit):
    """THE PAYOFF-FREE HALF OF IDENTITY 1, benchmark by benchmark.

    One unit of the rate currency paid at T is a TRADABLE, and its value today is
    `X_0 P_zar(0,T)`. Under the base measure that is `E[D_usd(0,T) X_T]`, so the composition has an
    identity of its own with no option in it, and it decides where a miss lives: whatever this
    reading is off by, identity 1 is off by at least as much, and only the REMAINDER is the
    measure change or the exposure.

    What it measures, in the engine's terms, is the discretely-rolled money market both currencies'
    curves imply. `GBMAssetPriceTSModelImplied.generate` builds the FX drift as
    `cumsum(r_base(t_k, dt) dt - r_rate(t_k, dt) dt)` off each curve's own scenario view, so:

    - the RATE currency is simulated and rolls, and what accumulates is `sum -log P(t_k, t_k+1)` -
      the discrete rolling bond, whose expectation is not the discount factor and whose gap is the
      1Y-first-node lesson's number on this world;
    - the BASE currency, being STATIC, does not roll at all. Its scenario view is the frozen t=0
      curve at every step, so the drift accumulates `T * r_base(0, dt)` rather than
      `-log D_base(0,T)`. That error is the whole slope of the base curve between the step tenor
      and T, and it does NOT shrink with the scenario grid - it GROWS, because a finer step reads a
      shorter (and here lower) rate.

    `numeraire_rel` is the pair of them together, which is what identity 1 actually carries.
    """
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
    """Half A end to end: curves solved, the FX surface built, the ladder fitted.

    Returns the config carrying every solved factor plus the readings each stage produced. The
    ORDER is forced (see `run_bootstrappers`) and is the dependency order: a swaption block reads
    the curve it is quoted against, and the quanto emission reads the FX vol curve.

    `theta` INJECTS a known theta* in place of the solve and changes nothing else - the curves are
    still solved (12.5 s and 7.8 s live, and the composition's strikes are read off them), the FX
    surface is still built, and the closure and its readings are still built at the parameters
    standing on the factor. It is sound because the fit is deterministic: the same snapshot at the
    same seed re-solves bit-identically, which is half A's own gate.
    """
    config = new_config(base_date, base_currency)
    seed_fx(config, base_currency, base_curve, 1.0)
    seed_fx(config, fx_currency, ir_curve, fx_spot_base_per_unit)
    install_blocks(config, blocks)

    curves = {name: solve_curve(config, name) for name in ir_market_prices}

    if fxvol_market_price is not None:
        run_bootstrappers(config, ['FXVolSurfaceParameters'])
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
    # theta*, machine-readable, on one line: this is what `--theta` reads back, and printing it is
    # what makes the next verification of these tables cost the composition rather than the fit.
    # FLUSH LEFT and under a token, so the extraction is one line and needs no parser:
    #     python gates/hw2f_composition.py --snapshot DIR | sed -n 's/^THETA_STAR //p' > theta.json
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
    """`{label: leg}` for the composition cells, off the SOLVED curve and the FITTED closure.

    The strike is rebuilt here rather than taken off the calibration's schedule, and then CHECKED
    against the coupon `create_market_swaps` wrote into the benchmark's own float leg - a swap
    struck at a rate the analytic price is not at-the-money for would fail the identity for a
    reason that has nothing to do with the measure.
    """
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
                        help='comma-separated scenario grids to sweep - the miss is a function '
                             'of this and the sweep is the finding')
    parser.add_argument('--batch', type=int, default=16384)
    parser.add_argument('--batches', type=int, default=16,
                        help='the published half-B tables were read at 8 (131,072 paths), not at '
                             'this default (262,144) - see the module docstring')
    parser.add_argument('--seed', type=int, default=5120)
    parser.add_argument('--rho', type=float, default=0.4)
    parser.add_argument('--theta', default=None,
                        help='a theta* block - a FILE or the JSON inline - injected in place of '
                             'the 62-71 minute fit. The report prints the line this reads back, '
                             'under the THETA_STAR token; the fit is deterministic, which is what '
                             'makes injecting it sound')
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
