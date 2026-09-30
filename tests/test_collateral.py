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

"""The collateral call - a CSA's arithmetic at one date, and the balance the record holds written
into the plan - held to the engine's own recursion.

`derivus_spine.collateral` is pure over plain data, so most of this file hands it numbers, exact
in binary so every assertion is an equality. The engine gates run a collateralised credit Monte
Carlo over a job the record compiled: the balance the record holds is the one the engine starts
from, and on every path the spine's call is the engine's own transfer - its required support where
the engine moved the balance, the balance where it did not - on the day a deal under the set pays
too, under either reading of that day's payment.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import derivus
from derivus import collateral as reading, spine, utils
from derivus.instruments import NettingCollateralSet
from derivus.schema import declared_fields
from derivus_spine import collateral, init_home

ACTOR = 'subject-collateral'
BASE, NEXT = '2024-06-28', '2024-06-29'
#: Units of the dollar - the book's base and reporting currency - per unit of each currency.
FX = {'USD': 1.0, 'ZAR': 0.0625, 'EUR': 1.25}
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures',
                        'autocall_job.json')


def listed(*rows):
    """A `CreditSupportList` in its wire form, `[rating, amount]` rows."""
    return {'.CreditSupportList': [list(row) for row in rows]}


#: A CSA in rand, which the dollar values at a sixteenth: the independent amount 1,000 USD, the
#: thresholds 125,000 and -100,000 USD, the minimum transfers 20,000 received and 30,000 posted.
TERMS = {'Object': 'NettingCollateralSet', 'Reference': 'CSA-1', 'Netted': 'True',
         'Collateralized': 'True', 'Agreement_Currency': 'ZAR', 'Balance_Currency': 'EUR',
         'Liquidation_Period': 0.0, 'Settlement_Period': 0.0,
         'Credit_Support_Amounts': {
             'Counterparty': 'CPTY', 'Independent_Amount': listed((1, 16_000.0)),
             'Received_Threshold': listed((1, 2_000_000.0), (2, 1.0)),
             'Posted_Threshold': listed((1, -1_600_000.0)),
             'Minimum_Received': listed((1, 320_000.0)),
             'Minimum_Posted': listed((1, 480_000.0))}}
#: The same CSA declaring dollars at half off posted and a quarter off received - the engine reading
#: the first alone, whichever side holds them.
HAIRCUTS = dict(TERMS, Collateral_Assets={'Cash_Collateral': [
    {'Currency': 'USD', 'Haircut_Posted': {'.Percent': 50.0},
     'Haircut_Received': {'.Percent': 25.0}}]})


def test_the_dials_are_read_as_the_engine_reads_them():
    """THE CSA IS THE ENGINE'S. Every `CreditSupportList` is read at its FIRST value exactly as
    `utils.CreditSupportList.value` reads it - a dict over the rows, so a rating listed twice keeps
    its first place and its last amount - `Haircut_Posted` as the fraction the engine's `Percent`
    decodes to, and the balance currency the agreement's where unstated, as the engine defaults
    it. Terms that collateralise nothing have no CSA, and a dial the terms state none of reads
    nothing - but the independent amount, which the engine reads as zero.

    Killing mutation: the first row's rating read as its amount, which puts every threshold at 1.
    """
    dials = collateral.csa(HAIRCUTS)
    stated = TERMS['Credit_Support_Amounts']
    for dial in collateral.DIALS:
        assert dials[dial] == utils.CreditSupportList(stated[dial]['.CreditSupportList']).value()
    assert dials['Received_Threshold'] == 2_000_000.0, 'the first rating, not the last'
    twice = listed((1, 5.0), (2, 6.0), (1, 7.0))
    assert collateral.csa(dict(TERMS, Credit_Support_Amounts=dict(stated, Minimum_Posted=twice)))[
        'Minimum_Posted'] == utils.CreditSupportList(twice['.CreditSupportList']).value() == 7.0
    assert dials['haircuts'] == {'USD': float(utils.Percent(50.0))}
    assert (dials['Agreement_Currency'], dials['Balance_Currency']) == ('ZAR', 'EUR')
    assert collateral.csa(dict(TERMS, Balance_Currency=''))['Balance_Currency'] == 'ZAR'
    assert collateral.csa(dict(TERMS, Collateralized='False')) is None
    assert collateral.csa({'Object': 'NettingCollateralSet'}) is None
    bare = collateral.csa(dict(TERMS, Credit_Support_Amounts={
        'Received_Threshold': listed((1, 5.0))}))
    assert (bare['Independent_Amount'], bare['Received_Threshold'], bare['Posted_Threshold'],
            bare['haircuts']) == (0.0, 5.0, None, {})


def test_the_support_required_is_the_engine_s_own_at():
    """THE SUPPORT REQUIRED IS THE ENGINE'S `At`, term for term: the independent amount, plus the
    excess over the received threshold above it, plus the excess below the posted threshold below
    it - the three crossed from the agreement currency - and at either threshold itself nothing
    beyond the independent amount.

    Killing mutation: the thresholds read in the agreement's currency uncrossed, which puts the
    received threshold at 2,000,000 dollars and calls nothing on a 300,000 exposure.
    """
    dials = collateral.csa(TERMS)
    assert [collateral.required(exposure, dials, FX) for exposure in (
        300_000.0, 125_000.0, 0.0, -100_000.0, -250_000.0)] == [
        176_000.0, 1_000.0, 1_000.0, 1_000.0, -149_000.0]


def test_a_call_clears_its_minimum_strictly_on_the_side_it_falls():
    """THE MINIMUM TRANSFER IS STRICT, as the engine's scan transfers: a shortfall exactly at the
    minimum received calls nothing and one a cent over it calls the whole of it; an excess exactly
    at the minimum posted posts nothing and one over it posts the whole of it, signed as the
    settlement moves it - received positive. The minimum reported is the one on the side the
    difference falls.

    Killing mutation: `>=` for the minimum transfer, which calls 20,000 at exactly the minimum.
    """
    dials = collateral.csa(TERMS)
    at, over = (collateral.call(exposure, {}, dials, FX) for exposure in (144_000.0, 144_000.5))
    assert (at['required'], at['balance'], at['call'], at['direction'], at['minimum_transfer']) \
        == (20_000.0, 0.0, 0.0, None, 20_000.0)
    assert (over['call'], over['direction']) == (20_000.5, collateral.CALL)
    held = {'EUR': 24_800.0}
    at, over = (collateral.call(0.0, holding, dials, FX) for holding in (
        held, {'EUR': 24_800.5}))
    assert (at['balance'], at['call'], at['direction'], at['minimum_transfer']) == (
        31_000.0, 0.0, None, 30_000.0)
    assert (over['call'], over['direction']) == (-30_000.625, collateral.POST)


def test_a_haircut_is_the_engine_s_haircut_posted_on_either_side():
    """A HAIRCUT IS `Haircut_Posted`, WHICHEVER SIDE HOLDS THE ASSET, as the engine's balance reads
    every cash row: what the bank holds and what it posted are each worth their crossed amount less
    it, `Haircut_Received` read by neither, and an asset the terms declare no haircut for is worth
    its crossed amount.

    Killing mutation: `Haircut_Received` taken on what the bank holds, which values 100,000 dollars
    held at 75,000.
    """
    dials = collateral.csa(HAIRCUTS)
    assert collateral.valued({'USD': 100_000.0}, dials, FX) == 50_000.0
    assert collateral.valued({'USD': -100_000.0}, dials, FX) == -50_000.0
    assert collateral.valued({'EUR': 8.0, 'USD': 4.0}, dials, FX) == 12.0


def test_collateral_and_margin_are_two_balances_and_the_call_reads_the_first():
    """COLLATERAL AND MARGIN ARE KEPT APART: what is held under each agreement is its standing
    collateral and margin movements summed per asset, each kind on its own, every agreement in one
    pass, as of the value date asked - a movement dated after it and money of another kind held
    nowhere - and the call reads the collateral alone, so margin posted beside it moves no call.

    Killing mutation: margin summed into collateral, which posts the half million of margin back as
    collateral held in excess.
    """
    def moved(kind, subject, asset, amount, day):
        return {'kind': kind, 'subject': subject, 'asset': asset, 'amount': amount,
                'effective_time': day + 'T00:00:00.000000Z'}

    movements = [moved('collateral', 'CSA-1', 'EUR', 20_000.0, '2024-06-27'),
                 moved('collateral', 'CSA-1', 'EUR', 4_800.0, BASE),
                 moved('collateral', 'CSA-1', 'ZAR', 160_000.0, BASE),
                 moved('margin', 'CSA-1', 'EUR', 500_000.0, '2024-06-27'),
                 moved('collateral', 'CSA-1', 'EUR', 999.0, NEXT),
                 moved('collateral', 'CSA-2', 'EUR', 7.0, BASE),
                 moved('fee', 'CSA-1', 'EUR', 3.0, BASE)]
    balances = collateral.held(movements, BASE)
    assert balances == {'CSA-1': {'collateral': {'EUR': 24_800.0, 'ZAR': 160_000.0},
                                  'margin': {'EUR': 500_000.0}},
                        'CSA-2': {'collateral': {'EUR': 7.0}, 'margin': {}}}
    assert collateral.held(movements)['CSA-1']['collateral']['EUR'] == 25_799.0
    held = balances['CSA-1']
    said = collateral.call(0.0, held['collateral'], collateral.csa(TERMS), FX)
    assert (said['balance'], said['call'], said['direction']) == (41_000.0, -40_000.0,
                                                                  collateral.POST)


def test_nothing_is_rounded_where_the_terms_declare_no_rounding():
    """A CALL IS ROUNDED WHERE THE TERMS SAY, and a netting set declares no rounding: the call is
    the difference itself, to the last digit the arithmetic carries.

    Killing mutation: the call rounded to the cent.
    """
    declared = declared_fields(NettingCollateralSet).values()
    names = [field.key for field in declared] + [
        part.key for field in declared for part in field.sub_fields or ()]
    assert not [name for name in names if 'round' in name.lower()], \
        'a rounding the terms declare is one the call must read'
    dials = collateral.csa(TERMS)
    said = collateral.call(144_000.123, {}, dials, FX)
    assert said['call'] == collateral.required(144_000.123, dials, FX) != round(said['call'], 2)


def test_a_mark_nobody_has_is_named_and_calls_nothing():
    """A CALL NOBODY CAN WORK OUT IS NAMED, NEVER READ AS ZERO. Where the P&L has no mark for a
    position under an agreement, the read answers that agreement's exposure and call null and
    names the mark under `unknown`, in the P&L's own words, beside the other agreements' calls
    worked out as ever; a spot the close lacks for an asset held is named the same way, and so is
    an agreement currency the terms leave out.

    Killing mutations: the P&L's unknowns dropped from the call, which works the arithmetic on a
    null and raises; a missing mark read as zero, which calls on the half of the exposure known; a
    currency the terms leave out sought as a spot, which raises.
    """
    close = {'day': BASE, 'rates': dict(FX), 'document': engine_job(TERMS)}
    valued = {'rows': [
        {'instrument': 'I1', 'agreement': 'CSA-1', 'value_end': 300_000.0, 'quantity_end': 1.0,
         'paid_end': 0.0},
        {'instrument': 'I2', 'agreement': 'CSA-1', 'value_end': None, 'quantity_end': 1.0,
         'paid_end': 0.0},
        {'instrument': 'I3', 'agreement': 'CSA-2', 'value_end': 300_000.0, 'quantity_end': 1.0,
         'paid_end': 0.0}],
        'unknown': [{'instrument': 'I2', 'what': 'no mark at the end'}]}
    agreements = [{'agreement': name, 'entity': 'LEI-1', 'terms': dict(TERMS, Reference=name)}
                  for name in ('CSA-1', 'CSA-2', 'CSA-3')] + [{'agreement': 'CSA-4', 'entity':
                  'LEI-1', 'terms': dict(TERMS, Agreement_Currency='', Balance_Currency='')}]
    gold = [{'kind': 'collateral', 'subject': agreement, 'asset': 'XAU', 'amount': 1.0,
             'effective_time': BASE + 'T00:00:00.000000Z'} for agreement in ('CSA-3', 'CSA-4')]
    unknown, known, uncrossed, unstated = reading.calls(close, valued, agreements, gold, BASE)
    assert (unknown['exposure'], unknown['call'], unknown['direction'], unknown['unknown']) == (
        None, None, None, valued['unknown'])
    assert (known['exposure'], known['direction'], known['unknown']) == (
        300_000.0 / FX['ZAR'], collateral.CALL, []), 'in rand, the agreement currency'
    assert (uncrossed['call'], uncrossed['unknown']) == (None, [
        {'instrument': None, 'what': 'the close carries no spot for XAU'}])
    assert (unstated['call'], unstated['unknown']) == (None, [
        {'instrument': None, 'what': 'the close carries no spot for XAU'},
        {'instrument': None, 'what': 'the terms of CSA-4 state no Agreement_Currency'}])


def engine_job(terms):
    """A one-date collateralised credit Monte Carlo: 100,000 units of a year's equity forward
    struck near its forward, under the CSA `terms`, on the template's dollar market with the rand
    and the euro static beside it, the equity the one factor simulated - so the exposure a day
    out spans both thresholds and both minimums. The file's own balance is 5 euros."""
    with open(TEMPLATE) as handle:
        job = json.load(handle)
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    flat = {'.Curve': {'meta': [], 'data': [[0.0, 0.02], [5.0, 0.02]]}}
    for currency in ('ZAR', 'EUR'):
        market['Price Factors']['FxRate.' + currency] = {
            'Domestic_Currency': None, 'Interest_Rate': currency, 'Spot': FX[currency]}
        market['Price Factors']['InterestRate.' + currency] = {
            'Currency': currency, 'Day_Count': 'ACT_365', 'Sub_Type': None, 'Curve': flat}
    market['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': 0.3, 'Drift': 0.0}}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'EquityPrice': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    forward = {'Object': 'EquityForwardDeal', 'Reference': 'FWD', 'Equity': 'EQ',
               'Currency': 'USD', 'Discount_Rate': 'USD', 'Payoff_Currency': 'USD',
               'Buy_Sell': 'Buy', 'Units': 100_000.0, 'Forward_Price': 103.0,
               'Maturity_Date': {'.Timestamp': '2025-06-27'}}
    job['Calc']['Deals'] = {'Reference': 'desk', 'Deals': {'Children': [{
        'Instrument': {'.Deal': dict(terms, Opening_Balance=5.0)},
        'Children': [{'Instrument': {'.Deal': forward}}]}]}}
    job['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': BASE}, 'Currency': 'USD',
        'Time_grid': '0d 1d 12m', 'Batch_Size': 4096, 'Simulation_Batches': 1, 'Random_Seed': 1,
        'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD'}
    return job


def balance_of(job):
    return job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']['Opening_Balance']


def test_the_record_s_balance_opens_the_engine_s_recursion_and_the_call_is_its_transfer(
        tmp_path, monkeypatch):
    """ONE FORMULA, THE ENGINE'S. The record holds collateral under a declared agreement - euros,
    rand and dollars posted, a margin beside it and a movement dated the day after - and the
    compiled job writes the collateral held by its base date as the set's `Opening_Balance`, in
    its `Balance_Currency` at the job's own spots, the margin and the later movement left out; a
    credit Monte Carlo over that job opens its recursion on exactly that balance. A day on, on
    every one of 4,096 paths spanning both thresholds and both minimums, the spine's call on the
    engine's own gross exposure moves the balance where the engine's scan moves it - to the
    support the spine requires, exactly on these dyadic spots and to rounding at the
    minimum-transfer edge on others - and leaves it where the engine leaves it. A base valuation, which runs no recursion, is compiled with no balance. An asset the job's
    spots cannot cross refuses by name wherever the book is valued - a plan, a read, a base
    valuation - while a diary, which values nothing, compiles past it; it needs no spot once
    returned in full, and with nothing held the job is the file.

    Killing mutations: the balance written in `Agreement_Currency`, where the engine multiplies by
    `Balance_Currency`'s spot; a movement after the base date written into the plan; margin
    summed into the balance; the thresholds uncrossed; the two minimums swapped; the early return
    kept where a balance exists; the balance written into a base valuation; an asset the spots
    cannot cross let through a read, or through a base valuation; a diary refused over it; an
    asset returned in full still wanting a spot.
    """
    home = tmp_path / 'spine'
    init_home(home, ACTOR)
    monkeypatch.setenv('DV_SPINE_HOME', str(home))
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    job = engine_job(TERMS)
    assert spine.compiled_job(job) is job, 'nothing held, and the file is the plan'
    spine.declare_entity('LEI-1', 'Client One')
    spine.declare_agreement('CSA-1', 'LEI-1', 'ISDA 2002 with CSA', TERMS)
    holding = {'EUR': 20_000.0, 'ZAR': 160_000.0, 'USD': -8_000.0}
    for reference, (asset, amount, kind, day) in enumerate((
            ('EUR', 20_000.0, 'collateral', '2024-06-27'), ('ZAR', 160_000.0, 'collateral', BASE),
            ('USD', -8_000.0, 'collateral', BASE), ('EUR', 500_000.0, 'margin', BASE),
            ('EUR', 999.0, 'collateral', NEXT))):
        spine.transition('CSA-1', 'settled', amount=amount, asset=asset, kind=kind,
                         reference='MOVE-{}'.format(reference),
                         effective_time=day + 'T00:00:00.000000Z')

    compiled = spine.compiled_job(job)
    assert balance_of(compiled) == pytest.approx(21_600.0, rel=1e-15), \
        'the collateral held by the base date, in euros'
    context = derivus.Context()
    context.load_json((json.dumps(compiled), 'collateral'))
    calc, out = derivus.run_cmc(context.current_cfg, prec=torch.float64)
    netting = out['Netting'].sub_structures[0].obj.Calc_res
    dates = sorted(calc.time_grid.mtm_dates)
    reported = list(np.array(dates)[calc.time_grid.report_index])
    opened, later = (reported.index(pd.Timestamp(day)) for day in (BASE, NEXT))
    held = netting['Collateral'][0]
    gross = netting['GrossMTM'][0][dates.index(pd.Timestamp(NEXT))]
    assert held[opened] == pytest.approx(21_600.0 * FX['EUR'], rel=1e-14), \
        'the engine opens on the balance written'

    dials = collateral.csa(TERMS)
    said = [collateral.call(float(exposure), holding, dials, FX) for exposure in gross]
    required = np.array([call['required'] for call in said])
    moved = held[later] != held[opened]
    assert np.array_equal(moved, [call['direction'] is not None for call in said]), \
        'the spine calls where the engine does not transfer, or the other way'
    assert np.array_equal(held[later][moved], (required[moved] / FX['EUR']) * FX['EUR']), \
        "the spine's required support is not the engine's"
    assert said[0]['balance'] == 27_000.0
    directions = [call['direction'] for call in said]
    regimes = np.where(gross > 125_000.0, 1, np.where(gross < -100_000.0, -1, 0))
    for count in ([directions.count(side) for side in (collateral.CALL, collateral.POST, None)]
                  + [int(np.sum(regimes == side)) for side in (1, 0, -1)]):
        assert count >= 200, 'the paths do not span both thresholds and both minimums'

    spine.transition('CSA-1', 'settled', amount=1.0, asset='XAU', kind='collateral',
                     reference='MOVE-XAU', effective_time=BASE + 'T00:00:00.000000Z')
    with pytest.raises(spine.SpineRefused, match='XAU as collateral under'):
        spine.compiled_job(job)
    for runs in (None, 'BaseValuation'):
        with pytest.raises(spine.SpineRefused, match='a market the book lacks; install the spot'):
            spine.compiled_job(job, strict=False, runs=runs)
    assert spine.compiled_job(job, strict=False, runs=False) is job, 'a diary reads no balance'
    spine.transition('CSA-1', 'settled', amount=-1.0, asset='XAU', kind='collateral',
                     reference='MOVE-XAU-BACK', effective_time=BASE + 'T00:00:00.000000Z')
    assert balance_of(spine.compiled_job(job)) == pytest.approx(21_600.0, rel=1e-15), \
        'returned in full, it needs no spot'
    assert spine.compiled_job(job, runs='BaseValuation') is job, 'a base valuation reads none'



def test_the_day_s_payment_is_read_as_the_set_s_own_recursion_reads_it():
    """THE DAY'S PAYMENT IS READ AS THE NETTING SET READS IT. A second forward under the set
    matures on the call date, so the gross exposure there carries what it pays that day, path by
    path. The call's exposure is the P&L's value - the gross less that payment - with the payment
    added back where the set holds it, `Exclude_Paid_Today` off as the engine defaults it, and
    standing where the set excludes it: under either setting, on every one of 4,096 paths, the
    spine's call moves the balance where the engine's scan moves it and to the support it requires,
    to rounding, while the other reading parts from the engine on hundreds of paths.

    Killing mutations: the day's payment never added back, which parts from the engine's default;
    the payment added back under `Exclude_Paid_Today` too; the setting read inverted.
    """
    holding, dials = {'EUR': 21_600.0}, collateral.csa(TERMS)
    for excluded in (False, True):
        job = engine_job(TERMS)
        netting = job['Calc']['Deals']['Deals']['Children'][0]
        netting['Instrument']['.Deal']['Opening_Balance'] = holding['EUR']
        netting['Children'].append({'Instrument': {'.Deal': dict(
            netting['Children'][0]['Instrument']['.Deal'], Reference='FWD-NEXT',
            Forward_Price=100.0, Maturity_Date={'.Timestamp': NEXT})}})
        if excluded:
            job['Calc']['MergeMarketData']['ExplicitMarketData']['Valuation Configuration'] = {
                'NettingCollateralSet': {'Exclude_Paid_Today': True}}
        context = derivus.Context()
        context.load_json((json.dumps(job), 'paid today'))
        calc, out = derivus.run_cmc(context.current_cfg, prec=torch.float64)
        result = out['Netting'].sub_structures[0].obj.Calc_res
        dates = sorted(calc.time_grid.mtm_dates)
        reported = list(np.array(dates)[calc.time_grid.report_index])
        held = result['Collateral'][0]
        later = held[reported.index(pd.Timestamp(NEXT))]
        moved = later != held[reported.index(pd.Timestamp(BASE))]
        gross = result['GrossMTM'][0][dates.index(pd.Timestamp(NEXT))]
        paid = out['Results']['cashflows']['USD'].loc[pd.Timestamp(NEXT)].to_numpy()
        assert reading.paid_today(job) is not excluded

        def calls(today):
            # each path a P&L row: four units, each worth the gross less the day's payment
            return [collateral.call(reading.exposure('CSA-1', {'rows': [{
                'agreement': 'CSA-1', 'value_end': float(total - cash), 'quantity_end': 4.0,
                'paid_end': float(cash) / 4.0}]}, FX, 'USD', today), holding, dials, FX)
                for total, cash in zip(gross, paid)]

        said, other = calls(reading.paid_today(job)), calls(not reading.paid_today(job))
        assert np.array_equal(moved, [call['direction'] is not None for call in said]), \
            'the spine calls on another exposure than the set recurses on'
        assert later[moved] == pytest.approx(
            np.array([call['required'] for call in said])[moved], rel=1e-12, abs=1e-6)
        assert np.sum(moved != [call['direction'] is not None for call in other]) >= 200, \
            "the day's payment moves no call here"
