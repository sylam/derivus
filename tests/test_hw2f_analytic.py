"""Hull-White near zero reversion speed, and the HW2F calibration through its job document.

THE SERIES BRANCH. `hw_calc_H` divides by `a*a`, `hw_calc_IJK` by `a**3`, `hw_calc_B`'s derivatives
by powers of `a`, and both `AtT` assemblies divide by a reversion speed again. Every one but the
last is a REMOVABLE singularity, taken by its power series below `HW_SERIES_REACH` in `a dt`; the
`AtT` quotient is held off zero by `HW_ALPHA_FLOOR`. Zero is reachable without solving: `Alpha`,
`Alpha_1` and `Alpha_2` declare `default=0`. What a document sees of it is the short rate's mean, a
closed form at every speed - Ho-Lee and G2++ at zero - read off a credit Monte Carlo's tenor-0
row. The digit-level readings the branch was placed on (the closed form 2.7e+10 out at |a| = 1e-8,
1e-11 lost at a = 0.05, the quotient's 3.8e-8 at the floor) are the engine docstrings'.

THE CALIBRATION is `fixtures/hw2f_four_quote_job.json` through `derivus.Context`: four ATM
swaptions on a humped ZAR curve, the analytic objective the family declares by default, and the
quote side on. The Schrager-Pelsser checker against the brute-force Monte Carlo, the stationarity
readings of the two objectives and the refuted re-solve oracle are records:
`docs_src/developer/roadmap.md` and `quote_sensitivities.md` carry their numbers.
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
import test_declared_defaults as book
import trial_fx
from derivus import utils

ID_BLOCK = 'HullWhite2FactorModelPrices.ZAR-JIBAR-3M'


#: The two other models dividing a simulated curve by its tenor, on EUR at 1% and 5% reversion.
ZERO_KNOT_MODELS = {
    'HullWhite2FactorImpliedInterestRateModel': ({'Lambda_1': 0.0, 'Lambda_2': 0.0}, {
        'HullWhite2FactorModelParameters.EUR': {
            'Alpha_1': 0.05, 'Alpha_2': 0.5, 'Correlation': -0.5,
            'Sigma_1': utils.Curve([], [[0.0, 0.01], [10.0, 0.01]]),
            'Sigma_2': utils.Curve([], [[0.0, 0.005], [10.0, 0.005]]),
            'Quanto_FX_Correlation_1': 0.0, 'Quanto_FX_Correlation_2': 0.0,
            'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]])}}),
    'PCAInterestRateModel': ({
        'Reversion_Speed': 0.05, 'Historical_Yield': utils.Curve([], [[0.0, 0.02], [10.0, 0.02]]),
        'Yield_Volatility': utils.Curve([], [[0.0, 0.2], [10.0, 0.2]]),
        'Eigenvectors': [{'Eigenvalue': 1.0, 'Eigenvector': utils.Curve([], [[0.0, 1.0], [10.0, 1.0]])}],
        'Rate_Drift_Model': 'Drift_To_Forward', 'Princ_Comp_Source': 'Correlation',
        'Distribution_Type': 'Lognormal'}, {})}


#: A sloped EUR curve stating a 0 knot and one under a day on its line, linear in the zero rate as
#: the factor reads it by default.
SLOPED = [[0.0, 0.01], [0.001, 0.01004], [0.25, 0.02], [1.0, 0.025], [5.0, 0.03]]


@pytest.mark.parametrize('model', ['HullWhite1FactorInterestRateModel'] + list(ZERO_KNOT_MODELS))
def test_a_tenor_0_knot_reads_the_stated_curve_and_every_deal_on_it_prices(model):
    """`InterestRate.EUR` states a 0 knot on a sloped curve, and both Hull-Whites and the PCA model
    divide the simulated curve by its tenor - 0/0 there, NaN on every row of every deal reading
    EUR. The knot's forward is read at its limit, and its model terms across a day. The fx trial
    family and a EUR cashflow 30 days out, on a static EURUSD under each model on EUR, antithetic,
    in float64:

    - row 0 of the simulated curve is the stated curve at every knot on every path, the 0 knot to
      its ulp and the rest bit for bit - the 0.001y knot spanning its own tenor; the cashflow's
      row 0 is its base valuation bit for bit, and the six closed-form deals' to 1e-13;
    - under the Hull-White at 1% and 5% reversion each row's tenor-0 mean is the short rate's,
      f(0, t) + s^2/(2a^2)(1 - e^{-at})^2 with f(0, t) = r(t) + t r'(t) off the stated knots, to
      3e-7 - the model terms read across their day, 2.4e-7 at two years.

    Killing mutations: the floor 0, every row NaN; the forward read across the floor alone, the
    0 knot a day's slope off the stated 1% at 1.0110%; the floor a year; the floor and the limit
    on every knot under a day, the 0.001y knot at the 0 knot's 1% against its stated 1.004%.
    """
    block, factors = ZERO_KNOT_MODELS.get(model, (None, {}))
    factors = dict(factors, **{'InterestRate.EUR': dict(book.FACTORS['InterestRate.EUR'],
                                                       Curve=utils.Curve([], SLOPED))})
    cashflow = {'Object': 'FixedCashflowDeal', 'Reference': 'CF30D', 'Currency': 'EUR',
                'Discount_Rate': 'EUR', 'Amount': 1e6,
                'Payment_Date': book.WORLD_BASE + pd.DateOffset(days=30)}
    deals = trial_fx.DEALS + [cashflow]
    calc, out = book.simulated(
        deals, () if block else ('EUR',), factors=factors, prec=torch.float64,
        models={'InterestRate.EUR': (model, block)} if block else None, Antithetic='Yes',
        Calc_Scenarios='All', Generate_Cashflows='No')
    curve = out['Results']['scenarios']['InterestRate.EUR']
    assert np.isfinite(curve.values).all()
    for tenor, rate in SLOPED:
        row0 = curve.xs(tenor, level=0).iloc[:, 0]
        assert (row0 == rate).all() if tenor else (abs(row0 - rate) <= np.spacing(rate)).all(), tenor
    rows = {deal.Instrument.field['Reference']: deal.Calc_res['Value'][0]
            for deal in calc.netting_sets.deals()}
    assert len(rows) == len(deals) and all(np.isfinite(v).all() for v in rows.values())
    marks, _ = book.marks(deals, factors)
    assert rows['CF30D'][0].mean() == float.fromhex(marks['CF30D'])
    for ref in ('FXASN', 'FXDASN', 'FXOT', 'FXNT', 'FXPKO', 'FXKOR'):
        assert rows[ref][0].mean() == pytest.approx(float.fromhex(marks[ref]), rel=1e-13), ref
    if block:
        return
    a, s, knots = 0.05, 0.01, np.array(SLOPED)
    for date, mean in curve.xs(0.0, level=0).mean(axis=0).items():
        t = (date - book.WORLD_BASE).days / 365.0
        segment = min(np.searchsorted(knots[:, 0], t, side='right') - 1, len(knots) - 2)
        slope = np.diff(knots[segment:segment + 2, 1])[0] / np.diff(knots[segment:segment + 2, 0])[0]
        short = np.interp(t, knots[:, 0], knots[:, 1]) + t * slope
        assert abs(mean - short - s * s / (2 * a * a) * np.expm1(-a * t) ** 2) <= 3e-7, date


#: The short rate's convexity at zero and near zero speed on each model, against the 0.05 / 0.5
#: speeds the world above runs - the rate itself is f(0, t) off the stated knots.
SPEEDS = [('HullWhite1FactorInterestRateModel', 0.0), ('HullWhite1FactorInterestRateModel', 1e-6),
          ('HullWhite2FactorImpliedInterestRateModel', 0.0),
          ('HullWhite2FactorImpliedInterestRateModel', 1e-6),
          ('HullWhite2FactorImpliedInterestRateModel', 0.5)]


@pytest.mark.parametrize('model,speed', SPEEDS)
def test_a_reversion_speed_through_zero_is_the_closed_form_short_rate(model, speed):
    """A EUR cashflow under each Hull-White on the sloped curve, antithetic in float64, with one
    reversion speed at zero - the field's own default - at 1e-6 and, for the two-factor model, at
    0.5. Each row's tenor-0 mean is the short rate's f(0, t) plus its convexity, B(t) = (1 -
    e^{-at})/a and t at a = 0:

    - one factor, sigma 0.01: s^2/2 B(t)^2, Ho-Lee at zero;
    - two factors, 0.01 at 5% and 0.005 at `speed`, correlated -0.5:
      s1^2/2 B1^2 + s2^2/2 B2^2 + rho s1 s2 B1 B2,

    to 1e-7; measured 1.1e-8 one factor and 8.4e-9 two, at every speed.

    Killing mutations: the series branch taken nowhere - the closed forms divide by zero, NaN at a
    speed of 0; the floor dropped - `AtT` is 0 x inf at 0; a series coefficient off by one power,
    every row off its closed form at every speed.
    """
    flat = utils.Curve([], [[0.0, 0.0], [10.0, 0.0]])
    factors = {'InterestRate.EUR': dict(book.FACTORS['InterestRate.EUR'], Curve=utils.Curve([], SLOPED))}
    if model == 'HullWhite1FactorInterestRateModel':
        block = {'Alpha': speed, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
                 'Quanto_FX_Volatility': flat, 'Sigma': utils.Curve([], [[0.0, 0.01], [10.0, 0.01]])}
        loadings = ((0.01, speed),)
    else:
        block = {'Lambda_1': 0.0, 'Lambda_2': 0.0}
        factors['HullWhite2FactorModelParameters.EUR'] = {
            'Alpha_1': 0.05, 'Alpha_2': speed, 'Correlation': -0.5,
            'Sigma_1': utils.Curve([], [[0.0, 0.01], [10.0, 0.01]]),
            'Sigma_2': utils.Curve([], [[0.0, 0.005], [10.0, 0.005]]),
            'Quanto_FX_Correlation_1': 0.0, 'Quanto_FX_Correlation_2': 0.0, 'Quanto_FX_Volatility': flat}
        loadings = ((0.01, 0.05), (0.005, speed))
    cashflow = {'Object': 'FixedCashflowDeal', 'Reference': 'CF30D', 'Currency': 'EUR',
                'Discount_Rate': 'EUR', 'Amount': 1e6,
                'Payment_Date': book.WORLD_BASE + pd.DateOffset(days=30)}
    _, out = book.simulated([cashflow], (), factors=factors, prec=torch.float64,
                            models={'InterestRate.EUR': (model, block)}, Antithetic='Yes',
                            Calc_Scenarios='All', Generate_Cashflows='No')
    curve = out['Results']['scenarios']['InterestRate.EUR']
    assert np.isfinite(curve.values).all()
    knots = np.array(SLOPED)
    for date, mean in curve.xs(0.0, level=0).mean(axis=0).items():
        t = (date - book.WORLD_BASE).days / 365.0
        segment = min(np.searchsorted(knots[:, 0], t, side='right') - 1, len(knots) - 2)
        slope = np.diff(knots[segment:segment + 2, 1])[0] / np.diff(knots[segment:segment + 2, 0])[0]
        short = np.interp(t, knots[:, 0], knots[:, 1]) + t * slope
        B = [s * (t if a == 0.0 else -np.expm1(-a * t) / a) for s, a in loadings]
        convexity = 0.5 * sum(b * b for b in B) + (-0.5 * B[0] * B[1] if len(B) == 2 else 0.0)
        assert abs(mean - short - convexity) <= 1e-7, (date, mean - short - convexity)


# ---------------------------------------------------------------------- the four-quote document

#: theta* of the four-quote block under the analytic objective, CPU float64, `Random_Seed` 5120:
#: four quotes against 23 parameters, so the fit INTERPOLATES, `||J'r||` 7.36e-8 against `||r||`
#: 6.67e-7.
AN_FOUR_THETA = {
    'Alpha_1': [0.32366193283076616],
    'Alpha_2': [0.03159569266668131],
    'Correlation': [-0.1565027209516267],
    'Sigma_1': [0.010723201039816721, 0.0047171113138259465, 0.01105966344498534,
                0.0029869336448657097, 0.006147380519528253, 0.014499710721411836,
                0.008805534966408953, 0.013599648494531245, 0.02206557694852036,
                0.013758711752955581],
    'Sigma_2': [0.0112443341051478, 0.006337283787322206, 0.019914566165825723,
                0.007078448665925047, 0.032773632020118466, 0.01580518503862898,
                0.024763249005316706, 0.025305113582427433, 0.025989907947955082,
                0.02032383961283696]}

#: `d(value)/dtheta` at `AN_FOUR_THETA`, the value being the four benchmarks priced by the engine's
#: own MONTE CARLO and summed - the cotangent the triangle reads its quote deltas in, recorded so a
#: document can be contracted with it without rebuilding the world.
AN_FOUR_COTANGENT = {
    'Alpha_1': [-0.002394625444929287],
    'Alpha_2': [-0.644846903737083],
    'Correlation': [0.019818474773087114],
    'Sigma_1': [0.002606564687650764, 0.004136865590535874, 0.014910968583565661,
               0.0054960754981406065, -0.0036576707830940085, 0.08797425834981634,
               0.009626188830825728, -0.011877314836208564, 0.0017974298600863118,
               0.011880089429211215],
    'Sigma_2': [0.02990214450466197, 0.1032865249445531, 0.24178788782579, 0.4432917415375396,
               1.4481529785218696, 1.1972815977810245, 0.5416931130766405, 0.550630304732666,
               0.576805097308654, 0.268203312428934]}

#: what that cotangent reads on the four quotes, in `descriptors` order
AN_FOUR_DELTAS = (0.04136113, 0.15476361, 0.11775876, 0.30144759)


def test_the_four_quote_job_document_pins_theta_and_its_quote_deltas(caplog):
    """THE JSON PIN of the four-quote block: `fixtures/hw2f_four_quote_job.json` through
    `derivus.Context` with its `Objective` taken out, so the family's declared default solves it,
    and the reading taken where a desk takes it - on the published `calibrated` tensors and the
    published quote leaf, nothing rebuilt here. RELATIVE and not to the bit: the document picks
    the machine's own device, and the recorded theta* is a CPU float64 solve's, the two agreeing to
    8.9e-7 rather than digit for digit - on a 19-dimensional null space the two chains stop at
    neighbouring points of one manifold.

    The contraction is minimum-norm in the metric the solver steps in (the column-scaled Jacobian):
    the fourth benchmark's delta is 0.3014 where the unscaled spelling read 0.2667, theta* unchanged.
    The published leaf carries no `.grad`: the chain backpropagated through it at every evaluation,
    0.3% to 2.3% of the one-pass answer standing there had `bootstrap` not cleared it. An analytic
    solve fits frozen-annuity vols, so `bootstrap` reports what the engine's own Monte Carlo makes of
    theta* once, naming the worst benchmark: 10Y x 10Y, -3.66% at the 2048 paths the block declares,
    mostly the simulation's own numeraire bias.

    Killing mutations: the default `Objective` flipped to `Monte_Carlo`; the pseudo-inverse taken on
    the unscaled Jacobian - the fourth delta reads 0.2667; the leaves published with `.grad`
    standing; the honesty reprice not run.
    """
    config = four_quote(lambda m: m['Market Prices'][ID_BLOCK]['instrument'].pop('Objective'))
    with caplog.at_level(logging.INFO, logger=''):
        config.bootstrap()
    lines = [r.getMessage() for r in caplog.records if 'Analytic objective' in r.getMessage()]
    assert len(lines) == 1 and 'Swaption_10Y_10Y' in lines[0], lines
    percent = float(lines[0].rsplit(',', 1)[1].split('%')[0])
    assert 0.05 < abs(percent) < 25.0, lines[0]
    solved = {utils.check_tuple_name(key).rsplit('.', 1)[-1]: value
              for key, value in config.calibrated_factors.items()}
    for name, recorded in AN_FOUR_THETA.items():
        landed = solved[name].detach().cpu().reshape(-1).numpy()
        assert np.abs(landed / np.array(recorded) - 1.0).max() < 1e-6, (
            '{}: the document solved to {} against the recorded {}'.format(
                name, list(landed), recorded))

    descriptors, quotes = config.quote_leaves[ID_BLOCK]
    assert all(leaf.grad is None for leaf in quotes), 'a leaf was published with .grad standing'
    value = sum((torch.as_tensor(part, dtype=solved[name].dtype, device=solved[name].device)
                 * solved[name].reshape(-1)).sum()
                for name, part in AN_FOUR_COTANGENT.items())
    deltas = np.array([float(g) for g in torch.autograd.grad(value, quotes)])
    assert np.abs(deltas / np.array(AN_FOUR_DELTAS) - 1.0).max() < 1e-6, (
        'the four quote deltas read {} against the recorded {} on {}'.format(
            list(deltas), AN_FOUR_DELTAS, list(descriptors)))


#: theta* of the four-quote block under `Objective: Monte_Carlo`, polished from `AN_FOUR_THETA`
#: standing as the block's written factor - the draw, the device and the Sobol sample the block's
#: own `Random_Seed` and `Simulations` freeze. Banked on an RTX 3090.
MC_WARM_THETA = {
    'Alpha_1': ['0x1.524b5405891dep-2'],
    'Alpha_2': ['0x1.cc8c6e8261099p-6'],
    'Correlation': ['-0x1.206c9f7e355cep-3'],
    'Sigma_1': ['0x1.854b979f40b8fp-7', '0x1.7c80ce8278ca0p-8', '0x1.82dd0081c2927p-7',
                '0x1.0bd9440e09e22p-8', '0x1.6bb93853144f2p-8', '0x1.173cd58a3a8c1p-6',
                '0x1.0d9e9ada351bdp-7', '0x1.d334f31f05c0fp-7', '0x1.61f1acf8d001ep-6',
                '0x1.beb7b38eb85e3p-7'],
    'Sigma_2': ['0x1.8cbb3d3121122p-7', '0x1.e20c98fd55481p-8', '0x1.4f34e4aab8fdep-6',
                '0x1.0c57876643a57p-7', '0x1.06a27d369a8b7p-5', '0x1.02afcd9d72775p-6',
                '0x1.98856a9c0c715p-6', '0x1.a42f30de2e030p-6', '0x1.af67d94642186p-6',
                '0x1.51ce06062f507p-6']}
#: the knots `HullWhite2FactorModelParameters` declares for both sigma curves, in years
SIGMA_KNOTS = [0.0, 1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_the_monte_carlo_objective_polishes_the_written_factor_to_its_bank(caplog):
    """The block's own estimator, the one the analytic solve is audited against: the four-quote
    document under `Objective: Monte_Carlo`, its `HullWhite2FactorModelParameters` factor already
    standing at the analytic theta*, so the fit is a WARM START - the least squares alone from
    that factor, no basin search - and lands on `MC_WARM_THETA` bit for bit. The Monte Carlo
    residual is conditional on its draw, so the block's `Random_Seed` and `Simulations` are part of
    the contract and the bank is the card's (a CPU-only box re-banks rather than reading a defect).
    3 s, where the cold chain on the same block paid 73.

    Killing mutations: the relative premium error inverted (model over market); the premium
    convention swapped, the Lognormal surface priced as Normal.
    """
    def warm(market):
        market['Market Prices'][ID_BLOCK]['instrument']['Objective'] = 'Monte_Carlo'
        factor = {name: AN_FOUR_THETA[name][0] for name in ('Alpha_1', 'Alpha_2', 'Correlation')}
        factor.update({name: {'.Curve': {'meta': [], 'data': [
            [t, v] for t, v in zip(SIGMA_KNOTS, AN_FOUR_THETA[name])]}} for name in ('Sigma_1', 'Sigma_2')})
        market['Price Factors']['HullWhite2FactorModelParameters.ZAR-JIBAR-3M'] = factor

    config = four_quote(warm)
    with caplog.at_level(logging.INFO, logger=''):
        config.bootstrap()
    assert any('warm start off HullWhite2FactorModelParameters' in r.getMessage()
               for r in caplog.records), 'the written factor did not warm start the fit'
    written = config.params['Price Factors']['HullWhite2FactorModelParameters.ZAR-JIBAR-3M']
    for name, banked in MC_WARM_THETA.items():
        value = written[name]
        read = value.array[:, 1] if hasattr(value, 'array') else np.atleast_1d(value)
        assert [float(x).hex() for x in read] == banked, (name, list(read))


def four_quote(edit=None):
    """`fixtures/hw2f_four_quote_job.json` as a `Config`, `edit(document)` applied to the wire form
    first."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures',
                           'hw2f_four_quote_job.json'), encoding='utf-8') as handle:
        document = json.load(handle)
    if edit:
        edit(document['Calc']['MergeMarketData']['ExplicitMarketData'])
    return derivus.Context().load_json((json.dumps(document), 'hw2f_four_quote_job')).current_cfg


def rows(market):
    return market['Market Prices'][ID_BLOCK]['instrument']['Instrument_Definitions']


REFUSED = {
    'objective': (lambda m: m['Market Prices'][ID_BLOCK]['instrument'].update(Objective='analytic'),
                  r"Objective 'analytic' is not one this family prices - it is 'Analytic'"),
    'block convention': (
        lambda m: m['Market Prices'][ID_BLOCK]['instrument'].update(Distribution_Type='Normal'),
        r"HullWhite2FactorModelPrices declares Distribution_Type 'Normal' and the InterestYieldVol "
        r"it names declares 'Lognormal'"),
    'zero vol': (lambda m: rows(m)[1].update(Market_Volatility={'.Percent': 0.0}),
                 'Swaption_2Y_5Y: Market_Volatility is quoted ZERO'),
    'absent vol': (lambda m: rows(m)[2].pop('Market_Volatility'),
                   'Swaption_3Y_3Y: the benchmark carries no Market_Volatility'),
    'surface convention': (
        lambda m: m['Price Factors']['InterestYieldVol.ZAR_SWAPTION'].update(
            Distribution_Type='Bachelier'),
        "'Bachelier', which is not a convention this calibration prices")}


@pytest.mark.parametrize('case', sorted(REFUSED))
def test_a_block_the_family_cannot_price_refuses_by_name(case):
    """The four-quote document, one field wrong, refuses before a benchmark is fitted, naming what
    it read: an `Objective` that is neither spelling, a block declaring a convention its surface
    does not, a zero or absent `Market_Volatility` - which used to read the surface's own ATM, a
    quote nobody gave - and a surface convention no premium is priced in.

    Killing mutations: an unknown objective falling through to the analytic branch; the block's
    declaration left unread; a zero vol read off the surface's ATM again.
    """
    edit, message = REFUSED[case]
    with pytest.raises(Exception, match=message):
        four_quote(edit).bootstrap()
