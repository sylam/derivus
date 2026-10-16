"""A static curve under `NoModel: RiskNeutral` is the base-date curve rolled forward: a read from a
row is that curve's forward from the row, so on the base date it reads what it reads under
`Constant`, and on a flat world it reads so on every row, the roll of a flat curve being the flat
curve. The readers gated are the ones that spelled the read for themselves: the floating resets a
swap, a FRA and an autocall swap forecast and the forwards an equity leg grows between its
settlement and its period ends, read between two tenors through `gather_weighted_curve`'s start
points; and the carries the one-step-survival pricers (the TARF, the discrete barrier, the
autocall), the closed-form barrier, the one-touch, the partial-time barrier, the extendable forward
and the compo geometry read as a rate, now `pricing.zero_carry_rate` through `deal_carry_rate`.
Every curve is static on a flat world, the one modelled factor a spot at no vol; the trial deals
sit beside the trial equity, the one deal of the pair that reaches it.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import derivus
from derivus.config import CustomJsonEncoder
import test_compo_analytic_pricers as compo
import test_position_scaling as trials

NETTING_SET = {'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
               'Collateralized': 'False'}
NO_VOL = {'Vol': 0.0, 'Drift': 0.0}


def credit_monte_carlo(job, nomodel, models):
    """`job`'s deals under one netting set on a quarterly credit Monte Carlo, every curve static,
    each spot of `models` (`{factor type: name}`) a GBM at no vol, read under `nomodel`: the
    exposure profile, rows by scenarios."""
    job = json.loads(json.dumps(job, cls=CustomJsonEncoder))
    calc = job['Calc']
    calc['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': calc['Calculation']['Base_Date'],
        'Currency': calc['Calculation']['Currency'], 'Time_Grid': '0d 3m(3m)', 'Batch_Size': 16,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'NoModel': nomodel,
        'Deflation_Interest_Rate': calc['Calculation']['Currency']}
    calc['Deals']['Deals']['Children'] = [
        {'Instrument': {'.Deal': NETTING_SET}, 'Children': calc['Deals']['Deals']['Children']}]
    market = calc['MergeMarketData']['ExplicitMarketData']
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {kind: 'GBMAssetPriceModel' for kind in models}, 'modelfilters': {}}}
    market['Price Models'] = {'GBMAssetPriceModel.' + name: NO_VOL for name in models.values()}
    context = derivus.Context()
    context.load_json((json.dumps(job), 'static'))
    return context.run_job()[1]['Results']['mtm'].values.astype(np.float64)


def trial(family, reference):
    """The trial deal beside the trial equity, on the declared-defaults world and its family's."""
    deals = {deal['Reference']: deal for deal in trials.FAMILIES[family].DEALS}
    equity = next(deal for deal in trials.FAMILIES['equity'].DEALS if deal['Reference'] == 'EQS')
    return trials.document(SimpleNamespace(
        DEALS=[equity, deals[reference]], FACTORS=trials.FAMILIES[family].FACTORS,
        CONFIGURATION=trials.FAMILIES[family].CONFIGURATION)), {'EquityPrice': 'EQ'}


def composite(deal):
    """A compo deal alone on the compo world, its equity and its fx both held at no vol."""
    return compo._job([deal], compo._factors()), {'EquityPrice': 'EQ', 'FxRate': 'EUR'}


READERS = {
    'swap': lambda: trial('core', 'SWAP'),
    'fra': lambda: trial('core', 'FRA'),
    'equity leg': lambda: trial('equity_swaps', 'EQL_PRINCIPAL'),
    'autocall swap': lambda: trial('equity_swaps', 'ACS_V2'),
    'tarf': lambda: trial('core', 'TARF'),
    'discrete barrier': lambda: trial('equity', 'EQKO'),
    'barrier': lambda: trial('core', 'BARR'),
    'one touch': lambda: trial('fx', 'FXOT'),
    'partial barrier': lambda: trial('fx', 'FXPKO'),
    'extendable forward': lambda: trial('fx', 'FXEXT'),
    'commodity forward': lambda: trial('commodity', 'METAL_FWD'),
    'energy option': lambda: trial('commodity', 'FUEL_OPT'),
    'compo one touch': lambda: composite(compo._one_touch('CT', 'Down', compo.DOWN)),
    'compo discrete barrier': lambda: composite(dict(
        compo._barrier('CB', 'Down_And_Out', compo.DOWN),
        Barrier_Dates=[[day, ''] for day in compo.SAMPLES])),
}


@pytest.mark.parametrize('reader', list(READERS))
def test_a_static_curve_reads_the_same_on_the_base_date_under_either_no_model(reader):
    """Each reader's profile under `RiskNeutral` is its profile under `Constant` on every row and
    scenario, within float32 rounding of the first row's scale (1e-7 measured, 1e-5 allowed).

    Killing mutations, the first row's mean with the trial equity's 10,000 in it, `Constant`
    against the mutant. The risk-neutral branch of `gather_weighted_curve` reading its end points
    for the start points, which forecasts every floating reset at 0% and grows no forward between
    two tenors: the swap 10,878.41 against -116,525.27, the FRA 10,409.80 against -769.53, the
    equity leg 80,138.63 against 40,721.77, the autocall swap -2,804.51 against 20,582.74.
    `zero_carry_rate` reading the curve unscaled, which a static curve under `RiskNeutral` answers
    with a difference of two zero rates, zero on this flat world: the TARF 9,927.68 against
    9,866.39, the discrete barrier 10,884.44 against 10,754.61, the barrier 10,078.93 against
    10,066.09, the one-touch 16,014.74 against 15,614.99, the partial-time barrier 10,093.70
    against 10,083.24, the autocall swap against -6,860.70, the compo one-touch 668.42 against
    677.98. The closed-form rate `r` read unscaled: the barrier against 10,082.15, the one-touch
    against 16,260.20, the partial-time barrier against 10,096.86, the compo one-touch against
    681.92. The compo discrete barrier is red under each, and under the compo geometry read on the
    deal's rows rather than its block's, which skips it by name on any grid of more than one block.
    A curve keyed by date rolled like a rate curve, a day added to a date and two prices
    differenced: the commodity forward 60,250.63 against 10,000.00, the energy option 15,532.63
    against 10,000.00.
    """
    job, models = READERS[reader]()
    constant = credit_monte_carlo(job, 'Constant', models)
    risk_neutral = credit_monte_carlo(job, 'RiskNeutral', models)
    assert constant.shape == risk_neutral.shape
    scale = np.abs(constant[0]).max()
    gap = np.abs(risk_neutral - constant).max()
    assert gap <= 1e-5 * scale, (reader, constant[0].mean(), risk_neutral[0].mean(), gap / scale)
