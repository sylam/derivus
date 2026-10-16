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


import logging
from functools import partial, reduce

from . import utils, pricing
from .schema import (
    Cash, DAY_COUNTS, F, Observes, REQUIRED, Row, own, DealFields, declared_factors, declared_settlements,
    engine_default, settlement_days,
    ADMIN, FX_ADMIN, FX_AXIS, CASHFLOWLISTDEAL, COMMODITY_TOUCH, EQUITYOPTIONBASE, EQUITY_TOUCH, EXPIRY_PRINT,
    FX_TOUCH, FXOPTIONBASE, FX_OPTION_STYLE, PRICING, FLOATING_INDEX, COUPON_SCHEDULE, CAP_FLOOR, OPTION_STYLE, TOUCH,
    QEDI_CUSTOMAUTOCALLSWAP, QEDI_CUSTOMSWAP)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn


def get_business_day_offsets(calendar_names, calendars, business_days=2):
    def calendar_business_day(calendar_name, calendars):
        """The business-day offset for a named calendar, Monday-Friday when it is unknown."""
        return calendars.get(
            calendar_name,
            {'businessday': pd.offsets.BDay(1)}
        )['businessday']

    """Return the two business-day offsets used for settlement adjustment.

    The adjustment assumes two offsets applied as:
        bus_day_offsets[0].rollforward(date + bus_day_offsets[1])

    If one calendar is supplied, use it for both sides.
    If none is supplied, use a standard Monday-Friday calendar.
    """
    if calendar_names:
        names = [x.strip() for x in calendar_names.split(',') if x.strip()]
        offsets = [business_days * calendar_business_day(name, calendars) for name in names]

        if len(offsets) == 1:
            offsets = [offsets[0]]

        return offsets[:2]

    return [pd.offsets.BDay(business_days)]


def forward_settlement_date(date_to_roll, calendar_names, calendars, business_days=2):
    """Return expiry plus settlement lag, rolled using the configured calendars."""
    bus_day_offsets = get_business_day_offsets(calendar_names, calendars, business_days=business_days)
    if len(bus_day_offsets) > 1:
        return utils.adjust_date(bus_day_offsets[0], 'Following', date_to_roll + bus_day_offsets[1])
    else:
        return date_to_roll + bus_day_offsets[0]


def option_date_info(deal, base_date, calendars, business_days=2):
    """The expiry, settlement and forward-settlement day offsets of an option-style deal.

    Forward_Settlement is expiry plus the settlement lag rolled on the configured calendars, used as
    T in forward calculations; Settlement is the day the deal's own declaration settles on.
    """
    expiry_date = deal.field['Expiry_Date']
    calendar_names = deal.field.get('Calendars')

    adjusted_forward_settlement_date = forward_settlement_date(
        expiry_date, calendar_names, calendars, business_days=business_days)

    # the day the deal pays: the last its reset registered, an option's declaration settling once
    settlement_date = max(max(dates) for dates in deal.settlement_currencies.values())
    expiry = (expiry_date - base_date).days
    settlement = (settlement_date - base_date).days
    forward_settlement = (adjusted_forward_settlement_date - base_date).days

    return expiry, settlement, forward_settlement


def expiry_fixing(field, base_date, time_grid, deal_type):
    """The expiry's print as a one-row reset, which a deal expired before the base date states in
    its `Price_Fixing` table: a payoff settling after the expiry is on that print, not on today's spot."""
    expiry = field['Expiry_Date']
    stated = dict(field.get('Price_Fixing') or []).get(expiry)
    if expiry < base_date and not stated:
        raise utils.UnpriceableSchedule(
            '{}: Expiry_Date {:%Y-%m-%d} is behind the base date and Price_Fixing states no print for '
            'it, so the payoff it settles has no level. Add the row [Expiry_Date, print]'.format(
                field.get('Reference', deal_type), expiry))
    return utils.TensorResets.from_observations(base_date, time_grid, [[expiry, stated or 0.0]])


def refuse_consequence_field(field, name, deal_type, remedy):
    """Refuse a document declaring a CONSEQUENCE the engine folds for itself.

    A knock, an expiry, an accrual is what the observations already say; a field asserting one is a
    second, unfalsifiable source for the same answer. The spine's vocabulary closure at the deal
    schema (`derivus_spine.vocabulary`). A blank is no statement, so it passes.
    """
    if field.get(name) not in (None, ''):
        raise utils.UnpriceableSchedule(
            '{}: {!r} declares a CONSEQUENCE ({}={!r}), and the state it asserts is DERIVED from '
            'the observations rather than declared beside them - a document that can say both can '
            'disagree with itself. Delete the field; {}'.format(
                field.get('Reference', deal_type), name, name, field[name], remedy))


def refuse_unpaired_schedule(field, rows, deal_type):
    """Refuse a `(fixing, settlement, ...)` schedule whose rows do not pair one with one.

    An OSS strip walks its j-th fixing and pays it at its j-th settlement, so a repeated fixing
    date or a settlement running backwards re-dates a payment silently.
    """
    if any(x[1] < x[0] for x in rows) or \
            any(a[0] >= b[0] or a[1] > b[1] for a, b in zip(rows, rows[1:])):
        raise ValueError('{}: {} fixings must be strictly increasing with non-decreasing '
                         'settlements on or after their fixing'.format(
            field.get('Reference', deal_type), deal_type))


def refuse_zero_payoff(field, deal_type, fieldname='Cash_Payoff'):
    """Refuse a digital whose payoff amount `fieldname` is exactly 0.0.

    The leg pays nothing on every path it is in the money, so it prices to zero and the structure
    it was booked to CANCEL keeps its whole payoff - a leg of no value is not a contra, and the
    amount is REQUIRED precisely because 0.0 is falsy and reads as unauthored. A named SKIP, not
    the fatal class: the netting set prices without the leg and the log names it.
    """
    if field[fieldname] == 0.0:
        raise ValueError(
            '{}: {} declares {} 0.0, and a digital that pays nothing is not a deal - it '
            'prices to exactly zero on every path and the leg it was booked to cancel keeps its '
            'whole payoff. Author the amount the leg cancels (derivus_compact_autocalls.py names '
            'it on a six-leg structure), or delete the leg'.format(
                field.get('Reference', deal_type), deal_type, fieldname))


def became(source, deal_type, calendars, valuation_options, **overrides):
    """`source` as a live `deal_type`, carrying every field that type declares plus `overrides`.

    The substitute is built the way a document of that type is - through `construct_instrument`,
    under that type's OWN valuation block - so it prices as its own document rather than as a
    branch inside the deal it replaced.
    """
    cls = globals()[deal_type]
    declared = {f.key for group in cls.fields for f in group.fields}
    field = dict(source.field)
    field.update(overrides)
    field = {k: v for k, v in field.items() if k in declared}
    field['Object'] = deal_type
    deal = construct_instrument(field, valuation_options)
    deal.base_currency = source.base_currency
    deal.reset(calendars)
    return deal


def barrier_monitoring_rows(rows):
    """A `Barrier_Dates` table as `[(date, observed-or-None)]`.

    One column is a bare date and reads blank; two is the date and its `Observed` close. Both
    shapes load, so a document written before the column existed is unchanged on the wire.
    """
    out = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            observed = row[1] if len(row) > 1 and row[1] not in ('', None) else None
            out.append((row[0], None if observed is None else float(observed)))
        else:
            out.append((row, None))
    return out


def barrier_state_at(rows, base_date, barrier, barrier_up, deal_type, reference):
    """The monitoring schedule's resolved verdict at `base_date`: `(hit, date)`, `(False, None)`
    where nothing observed has crossed.

    A monitoring date on or before the base date with no `Observed` REFUSES - the absence of a
    fixing is not the absence of a hit, and a schedule the engine silently walked past is a deal
    priced alive on no evidence.
    """
    hit_on = None
    for date, observed in sorted(barrier_monitoring_rows(rows), key=lambda row: row[0]):
        if date > base_date:
            break
        if observed is None:
            raise utils.UnpriceableSchedule(
                '{}: the {} monitoring date {:%Y-%m-%d} is on or before the base date '
                '{:%Y-%m-%d} and its Observed close is blank. The absence of a fixing is not the '
                'absence of a hit, and the engine will not price the deal alive on no evidence - '
                'author the close that date fixed at in the Barrier_Dates row, or delete the row '
                'if the level was never watched then'.format(
                    reference or deal_type, deal_type, date, base_date))
        if hit_on is None and (observed >= barrier if barrier_up else observed <= barrier):
            hit_on = date
    return hit_on is not None, hit_on


def resolve_discrete_barrier(deal, base_date, calendars, valuation_options, vanilla_type,
                             **vanilla):
    """A discretely-monitored barrier deal folded against the `Observed` closes already in.

    No decisions left and the level was crossed: the deal IS the vanilla (knock-in) - `vanilla`
    naming what that type calls by another name - or IS its rebate cashflow at the crossing
    (knock-out), built as a document of that type. Decisions remain - or nothing crossed - and the
    deal is unchanged, its resolved rows already behind the pricer's first monitored date.

    A t0-folded state registers no boundary set, so nothing here reaches the sensitivity
    architecture. Continuous monitoring (no `Barrier_Dates`) needs a daily bar series to fold
    over, which nothing files yet, so it is returned untouched.
    """
    rows = deal.field.get('Barrier_Dates', [])
    if not rows:
        return deal
    barrier_type = deal.field['Barrier_Type']
    hit, hit_on = barrier_state_at(
        rows, base_date, float(deal.field['Barrier_Price']), 'Up' in barrier_type,
        type(deal).__name__, deal.field.get('Reference'))
    if not hit:
        return deal
    if 'In' in barrier_type:
        return became(deal, vanilla_type, calendars, valuation_options,
                      Settlement_Date=deal.field.read('Settlement_Date', deal.field['Expiry_Date']),
                      **vanilla)
    # the rebate is per DEAL and signed by Buy_Sell, which `FixedCashflowDeal` reads off `Amount`
    return became(deal, 'FixedCashflowDeal', calendars, valuation_options,
                  Currency=deal.payoff_ccy,
                  Discount_Rate=deal.field.read('Discount_Rate', deal.field['Currency']),
                  Amount=(1.0 if deal.field['Buy_Sell'] == 'Buy' else -1.0) *
                         float(deal.field.get('Cash_Rebate', 0.0)),
                  Payment_Date=hit_on)


def calc_factor_index(field, static_offsets, stochastic_offsets, all_tenors={}):
    """Return a factor's offset in the scenario block, whether it is static or stochastic."""
    if static_offsets.get(field) is not None:
        return tuple([False, field, static_offsets[field].get_subtype()] + all_tenors.get(field, []))
    elif stochastic_offsets.get(field) is not None:
        return tuple([True, field, stochastic_offsets[field].factor.get_subtype()] + all_tenors.get(field, []))
    else:
        raise Exception('Cannot find {}'.format(utils.check_tuple_name(field)))


def calc_factor_code_chain(head, fieldname, static_offsets, stochastic_offsets, all_tenors={}):
    """Positional prefix-chain code: period 1 is the `head` factor, each longer prefix the head's
    `utils.NESTED` tail named by the whole chain (curve: itself; 0D spot: ObservedBasis). See docs."""
    return [calc_factor_index(utils.Factor(head if x == 1 else utils.NESTED[head], fieldname[:x]),
                              static_offsets, stochastic_offsets, all_tenors)
            for x in range(1, len(fieldname) + 1)]


def calc_factor_value(field, static_offsets, stochastic_offsets, all_factors):
    """Return a factor's current value, whether it is static or stochastic."""
    if static_offsets.get(field) is not None:
        return all_factors[field].current_value()
    elif stochastic_offsets.get(field) is not None:
        return all_factors[field].factor.current_value()
    else:
        raise Exception('Cannot find value for {}'.format(utils.check_tuple_name(field)))


def get_interest_rate_currency(name, all_factors):
    """Read the Currency on an Interest Rate Price Factor"""
    ir_curve = all_factors.get(utils.Factor('InterestRate', name))
    return ir_curve.factor.get_currency() if hasattr(ir_curve, 'factor') else ir_curve.get_currency()


def get_inflation_index_name(fieldname, all_factors):
    """Read the Name of the Price Index price factor linked to this inflation index"""
    inflation = all_factors.get(utils.Factor('InflationRate', fieldname))
    return inflation.factor.param.get('Price_Index') if hasattr(inflation, 'factor') \
        else inflation.param.get('Price_Index')


def get_inflation_index_objects(inflation_name, index_name, all_factors):
    """Read the inflation rate and Price Index factor objects behind an inflation index"""
    inflation = all_factors.get(utils.Factor('InflationRate', inflation_name))
    inflation_factor = inflation.factor if hasattr(inflation, 'factor') else inflation
    index = all_factors.get(utils.Factor('PriceIndex', index_name))
    index_factor = index.factor if hasattr(index, 'factor') else index
    return inflation_factor, index_factor


def get_fxrate_factor(fieldname, static_offsets, stochastic_offsets):
    """Read the (basis-aware) code of the FX rate price factor"""
    return calc_factor_code_chain('FxRate', fieldname, static_offsets, stochastic_offsets)


def get_cross_barrier_underlying(pair, stochastic_offsets):
    """The FxRate whose simulated log-variance is the cross of the two currencies `pair`, or None:
    `t_Bridge_Variance_Rate` is keyed per factor, so only a cross with exactly one live leg and no
    basis chain has one, and None leaves the pricer observing endpoints."""
    legs = [utils.Factor('FxRate', x) for x in pair if len(x) == 1]
    live = [x for x in legs if stochastic_offsets.get(x) is not None]
    return live[0] if len(legs) == 2 and len(live) == 1 else None


def get_equity_barrier_underlying(fieldname):
    """The single factor whose simulated log-variance governs crossings between grid dates, or None.

    The equity leg is monitored directly, so the only guard is a composed basis chain: the head
    factor's variance alone understates the composed spot's, and `t_Bridge_Variance_Rate` is keyed
    per factor. None means the pricer observes endpoints, the conservative direction.
    """
    return None if len(fieldname) > 1 else utils.Factor('EquityPrice', fieldname)


def get_fxrate_spot(fieldname, static_offsets, stochastic_offsets, all_factors):
    """Read the spot of the FX rate price factor"""
    return calc_factor_value(
        utils.Factor('FxRate', fieldname), static_offsets, stochastic_offsets, all_factors)[0]


def get_forwardprice_sampling(fieldname, all_factors):
    """Read the sampling offset for the Sampling_type price factor"""
    return all_factors.get(utils.Factor('ForwardPriceSample', fieldname))


def get_forwardprice_factor(cashflow_currency, static_offsets, stochastic_offsets,
                            all_tenors, all_factors, reference_factor, forward_factor,
                            base_date):
    """Read the Forward price factor of a reference Factor - adjusts the all_tenors lookup to
    include the excel_date version of the base_date"""
    if reference_factor is not None:
        forward_price = reference_factor.get_forwardprice()
        # note that in future we could stack forward prices together
        forward_offset = [calc_factor_index(
            utils.Factor('ForwardPrice', forward_price), static_offsets, stochastic_offsets, all_tenors)]
    else:
        forward_offset = None

    forward_fx_ofs = get_fx_and_zero_rate_factor(
        forward_factor.get_currency(), static_offsets, stochastic_offsets, all_tenors, all_factors)

    if cashflow_currency != forward_factor.get_currency():
        cashflow_fx_ofs = get_fx_and_zero_rate_factor(
            cashflow_currency, static_offsets, stochastic_offsets, all_tenors, all_factors)
        return [forward_offset, forward_fx_ofs, cashflow_fx_ofs]
    else:
        return [forward_offset, forward_fx_ofs, forward_fx_ofs]


def get_reference_factor(fieldname, all_factors):
    """Read the Reference and Forward price factors"""
    reference = all_factors.get(utils.Factor('ReferencePrice', fieldname))
    return reference.factor if hasattr(reference, 'factor') else reference


def get_factor_component(componentname, all_factors):
    # repo/carry/dividend/currency live on the primary spot = the first period of the name
    primary = componentname.name[:1] if componentname.type in utils.BASIS_COMPOSABLE_TYPES else componentname.name
    underlying = all_factors.get(utils.Factor(componentname.type, primary))
    under_factor = underlying.factor if hasattr(underlying, "factor") else underlying
    return under_factor


def get_equity_component(fieldname, all_factors):
    return get_factor_component(utils.Factor('EquityPrice', fieldname), all_factors)


def get_commodity_component(fieldname, all_factors):
    return get_factor_component(utils.Factor('CommodityPrice', fieldname), all_factors)


def get_survival_component(fieldname, all_factors):
    """The Survival Probability factor object - a credit pricer reads its Recovery Rate at eval"""
    return get_factor_component(utils.Factor('SurvivalProb', fieldname), all_factors)


def get_reference_factor_objects(fieldname, all_factors):
    """Read the Reference and Forward price factors"""
    reference_factor = get_reference_factor(fieldname, all_factors)
    forward_price_name = reference_factor.get_forwardprice()
    forward = all_factors.get(utils.Factor('ForwardPrice', forward_price_name))
    forward_factor = (forward.factor if hasattr(forward, 'factor') else forward)
    return reference_factor, forward_factor


def get_implied_correlation(rate1, rate2, all_factors):
    """The Correlation factor object a rate pair names, or None when the pair is unauthored -
    `utils.implied_correlation` reads the number off it at eval"""
    correlation_name = rate1[:-1] + ('{0}/{1}'.format(rate1[-1], rate2[0]),) + rate2[1:]
    if rate1[0] == 'FxRate':
        correlation_name += (rate1[1],)
    return all_factors.get(utils.Factor('Correlation', correlation_name))


def equity_fx_correlation(field, all_factors):
    """An equity's compo or quanto correlation, `Correlation.EquityPrice.<vol>/FxRate.<sorted pair>`,
    and the sign the deal's own direction reads it with, the pair being named sorted."""
    if not field.get('Equity_Volatility'):
        raise ValueError('a compo or quanto payoff names its Equity_Volatility, which names the pair\'s correlation')
    corr_sign, fx_lookup = utils.check_fx_name([field['Currency'][0], field['Payoff_Currency'][0]])
    return get_implied_correlation(
        ('EquityPrice',) + field['Equity_Volatility'], ('FxRate',) + fx_lookup, all_factors), corr_sign


def unnamed_surface(deal, paid_in):
    """The booking message of a compo or quanto paid across the pair - in `paid_in`, blank being the
    deal's own currency - that names no Equity_Volatility, which `equity_fx_correlation` refuses at
    compile; the surface is otherwise optional on a deal pricing no option of its own."""
    currency = deal.field.get('Currency')
    crossed = deal.field.read(paid_in, currency) != currency
    if deal.payoff_type != 'Standard' and crossed and not deal.check_factor_name('Equity_Volatility'):
        yield 'Equity_Volatility is not stated, and a {} payoff reads the pair\'s correlation off it'.format(
            deal.payoff_type)


def reference_fx_correlation(field, all_factors):
    """A reference price's, as the energy family declares it (`config.conditional_fields`):
    `Correlation.FxRate.<local>.<payoff>/ReferencePrice.<reference>.<local>`, named local to payoff
    and so read unsigned."""
    return get_implied_correlation(('FxRate',) + field['Currency'] + field['Payoff_Currency'],
                                   ('ReferencePrice',) + field['Reference_Type'], all_factors), 1.0


def get_commodity_rate_factor(fieldname, static_offsets, stochastic_offsets):
    """Read the (basis-aware) code of the commodity spot price factor"""
    return calc_factor_code_chain('CommodityPrice', fieldname, static_offsets, stochastic_offsets)


def get_forward_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    return [calc_factor_index(utils.Factor('ForwardRate', fieldname), static_offsets, stochastic_offsets, all_tenors)]


def get_observed_basis_decay(commodity, all_factors):
    """`(phi, lam, mu code)` - how the tail of a composed spot's code decays from its current level.

    `lam` is the slow mean's own EWMA coefficient, `0.0` when that extension is off; a projection
    needs both, because with the mean recursion the simulated `(b, mu)` pair is row-stochastic and a
    deviation decays at `lam*phi` toward the pair's conserved mixture rather than at `phi` toward
    the mean. `mu` is the spot-shaped code reading the `(key, 'basis_mu')` series the process
    publishes, `None` when the recursion is off. A tail nothing simulates has no dynamics, so
    `phi = 1` leaves it where it is. The declared parameters ride `Factor_dep` as plain floats, not
    AAD leaves. Valid inside an inner-MC fork, which republishes every path series.
    """
    if len(commodity) < 2 or not commodity[-1][utils.FACTOR_INDEX_Stoch]:
        return 1.0, 0.0, None
    key = commodity[-1][utils.FACTOR_INDEX_Offset]
    # the PROCESS declares its own decay pair (`basis_decay`) - one model per payoff: a basis
    # law that cannot state the (phi, lam) projection raises there, never prices as another's
    phi, lam = all_factors[key].basis_decay()
    return phi, lam, ([(True, (key, 'basis_mu'), None)] if lam else None)


def get_equity_rate_factor(fieldname, static_offsets, stochastic_offsets):
    """Read the (basis-aware) code of the equity spot price factor"""
    return calc_factor_code_chain('EquityPrice', fieldname, static_offsets, stochastic_offsets)


def get_equity_spot(fieldname, static_offsets, stochastic_offsets, all_factors):
    """Read the spot of the equity price factor"""
    return calc_factor_value(
        utils.Factor('EquityPrice', fieldname), static_offsets, stochastic_offsets, all_factors)[0]


def get_equity_currency_factor(fieldname, static_offsets, stochastic_offsets, all_factors):
    """Read the index of the Equity's Currency price factor (off the primary spot)"""
    fxrate = get_equity_component(fieldname, all_factors).get_currency()
    return [calc_factor_index(utils.Factor('FxRate', fxrate),
                              static_offsets, stochastic_offsets)]


def get_dividend_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the dividend rate price factor"""
    return [calc_factor_index(utils.Factor('DividendRate', fieldname), static_offsets,
                              stochastic_offsets, all_tenors)]


def get_interest_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the interest rate price factor (a curve + its basis-spread chain)"""
    return calc_factor_code_chain('InterestRate', fieldname, static_offsets, stochastic_offsets, all_tenors)


def get_equity_zero_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
    """Read the equity's interest rate price factor (off the primary spot)"""
    ir_curve = get_equity_component(fieldname, all_factors).get_repo_curve_name()
    return get_interest_factor(ir_curve, static_offsets, stochastic_offsets, all_tenors)


def get_commodity_zero_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
    """Read the commodity's interest rate price factor (off the primary spot)"""
    ir_curve = get_commodity_component(fieldname, all_factors).get_repo_curve_name()
    return get_interest_factor(ir_curve, static_offsets, stochastic_offsets, all_tenors)


def get_fx_zero_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
    """Read the Currency's interest rate price factor (off the primary spot)"""
    ir_curve = get_factor_component(
        utils.Factor('FxRate', fieldname), all_factors).get_repo_curve_name(fieldname[:1])
    return get_interest_factor(ir_curve, static_offsets, stochastic_offsets, all_tenors)


def get_fx_and_zero_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
    """Get the FX rate and it's repo curve"""
    return [get_fxrate_factor(fieldname, static_offsets, stochastic_offsets),
            get_fx_zero_rate_factor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors)]


def get_price_index_factor(fieldname, static_offsets, stochastic_offsets):
    """Read the index of the Price Index price factor"""
    return [calc_factor_index(utils.Factor('PriceIndex', fieldname), static_offsets, stochastic_offsets)]


def get_survival_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the survival probability price factor"""
    return [calc_factor_index(utils.Factor('SurvivalProb', fieldname), static_offsets,
                              stochastic_offsets, all_tenors)]


def get_inflation_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the inflation rate price factor (a curve + its prefix chain)"""
    return calc_factor_code_chain('InflationRate', fieldname, static_offsets, stochastic_offsets, all_tenors)


def get_vol_factor(factor_type, fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of a vol surface of `factor_type` (`FXVol`, `EquityPriceVol`); only one vol
    surface is supported. An SVI or Skew surface answers its parameter sub-factors, which inherit
    the type tag, so a parametric surface's parameters carry it too."""
    factor_name = utils.Factor(factor_type, fieldname)

    def check_surface_type(subtype, factor, stoch):
        params = {'SVI': factor.svi_params, 'Skew': factor.skew_params}.get(subtype[0])
        names = factor_name if params is None else [
            utils.Factor(factor_name.type, factor_name.name + (param,)) for param in params]
        return [tuple([stoch, names, subtype] + all_tenors.get(factor_name, []))]

    if static_offsets.get(factor_name) is not None:
        subtype = static_offsets[factor_name].get_subtype()
        return check_surface_type(subtype, static_offsets[factor_name], False)
    elif stochastic_offsets.get(factor_name) is not None:
        # not yet implemented but doable
        subtype = stochastic_offsets[factor_name].factor.get_subtype()
        return check_surface_type(subtype, stochastic_offsets[factor_name].factor, True)
    else:
        raise Exception('Cannot find {}'.format(utils.check_tuple_name(factor_name)))


def get_spot_model_params_factor(spot_model, name, all_factors, static_offsets, stochastic_offsets):
    """A non-GBM spot model's parameter factor, by NAMING CONVENTION off the factor whose law it
    describes: `<spot_model>ModelParameters.<name>`. Model-agnostic.

    An equity deal names its equity; an FX deal names the pair's key
    (`utils.spot_model_currency`) - its NON-BASE token, or `<underlying>.<priced in>` for a cross,
    the same key the calibration writes.

    Returns the SVI-shaped index `[(stoch, [per-parameter sub-factors], spot_model,
    {curve parameter: knots})]`, subtype tagged with spot_model for the pricer's branch, or None for
    `spot_model == 'None'` (GBM). The fourth slot carries every structural fact the factor answers -
    each curve's knots and the non-leaf scalars its kit reads - the leaves themselves riding the
    static buffer and resolving on the tensor side.

    The value is validated against `Deal.spot_models` at construction, so what is left here is
    presence: a model switched on but absent from the market data raises KeyError, which the
    dependency loop turns into a skipped deal, never a silent GBM fallback."""
    if spot_model == 'None':
        return None
    mp = utils.spot_model_factor(spot_model, name)
    if static_offsets.get(mp) is not None:
        stoch = False
    elif stochastic_offsets.get(mp) is not None:
        stoch = True
    else:
        raise KeyError('SpotModel=%s but %s not in the market data'
                       % (spot_model, utils.check_tuple_name(mp)))
    factor = get_factor_component(mp, all_factors)
    return [tuple([stoch, [utils.Factor(mp.type, mp.name + (param,))
                           for param in factor.current_value()], spot_model,
                   factor.curve_tenors()])]


def set_spot_model_index(field_index, model, options, reference):
    """Write a declared spot model's parameter factor, refusing a deal that re-declares its clock.

    The trading-day clock is what the fitted parameters MEAN, so it is the factor's and the kit
    reads it there; a `Steps_Per_Year` on the deal beside a different one on the block is two
    clocks for one walk. A GBM deal writes nothing and keeps its own declaration."""
    if model is not None:
        field_index['Spot_Model'] = model
        fitted = float(model[0][utils.FACTOR_INDEX_Tenor_Index]['Steps_Per_Year'])
        declared = options.get('Steps_Per_Year')
        if declared is not None and float(declared) != fitted:
            raise utils.UnpriceableSchedule(
                '{}: the Valuation Configuration declares Steps_Per_Year {:g} and the {} factor '
                'this deal prices on was fitted on {:g}. The step clock is what the fitted '
                'parameters mean - the walk reads the block\'s own number and a deal cannot '
                're-declare it, or one document values the same parameters on two grids. Drop '
                'Steps_Per_Year from the Valuation Configuration, or refit the block on {:g}'.format(
                    reference, float(declared), model[0][utils.FACTOR_INDEX_SubType], fitted,
                    float(declared)))


def spot_model_reciprocal_axis(spot_model, underlying, currency, base, reference):
    """Does this FX deal pay on the RECIPROCAL of the axis its spot model was fitted on - and, for a
    family that cannot be carried there, the refusal.

    A pair's law describes one token PRICED IN another (`utils.spot_model_pair`), so a deal whose
    underlying IS that second token pays on `1/s` and settles in the other currency. It is the base
    for a pair with a base leg and the alphabetically earlier token for a cross - one question
    either way. LogVar2FJ transports to that numeraire as a measure change inside its walk
    (`utils.LogVar2FJ.walk`).

    An ALLOW-LIST, so a family added without a carry refuses rather than pricing a fit on the
    wrong axis. Compared on `check_rate_name` tuples, the same spelling-blind test
    `utils.spot_model_currency` makes, or a mixed call misses the inversion.
    """
    priced_in = utils.spot_model_pair(underlying, currency, base)[1] or base
    if utils.check_rate_name(underlying) != utils.check_rate_name(priced_in):
        return False
    if spot_model != 'LogVar2FJ':
        raise utils.UnpriceableSchedule(
            '{0}: SpotModel={1!r} on a deal whose Underlying_Currency {2} IS the currency its '
            'pair\'s law is priced in. The fit describes {3} priced in {2} - the token a rate is '
            'priced in has no law of its own - and this deal pays on that reciprocal, settled in '
            '{3}. LogVar2FJ carries to that numeraire with a measure change inside its walk '
            '(utils.LogVar2FJ.walk); {1} carries no such derivation. Declare LogVar2FJ, or quote the pair '
            'the other way up so the deal is written on {3} and no axis is crossed'.format(
                reference, spot_model, '.'.join(utils.check_rate_name(underlying)),
                '.'.join(utils.check_rate_name(currency))))
    return True


def get_interest_vol_factor(fieldname, tenor, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the interest vol price factor"""
    pricefactor = 'InterestRateVol' if pd.Timestamp('1900-01-01') + tenor <= pd.Timestamp('1900-01-01') + pd.DateOffset(
        years=1) else 'InterestYieldVol'
    return [calc_factor_index(utils.Factor(pricefactor, fieldname), static_offsets,
                              stochastic_offsets, all_tenors)]


def get_commodity_vol_factor(fieldname, static_offsets, stochastic_offsets, all_tenors):
    """Read the index of the commodity vol price factor - this getter OWNS `CommodityPriceVol`."""
    return [calc_factor_index(utils.Factor('CommodityPriceVol', fieldname), static_offsets, stochastic_offsets,
                              all_tenors)]


class Deal(object):
    """Base class for a trade or deal.

    Aggregates sub-deals - a netting set is itself a "deal" - and calculates the dynamic dates a
    deal resets on.
    """
    documentation = ''

    #: A deal that breaks down into simpler instruments carries them as JSON `Children` and prices
    #: them in `post_process`; a leaf - an option, a cashflow list, a deposit - cannot. The engine
    #: recurses on `Children` being PRESENT, never on the type.
    accepts_children = False

    #: What the type observes (`schema.Observes`) - the index a fixing is of, the table a print
    #: fills, the choice its expiry vests. Read through `schema.OBSERVES`, which holds each type's
    #: own declaration and never a parent's, as `vernacular` is read.
    observes = None

    #: The `SpotModel` values this deal type HONOURS, declared the way a process declares
    #: `factor_types`. The base declares GBM only, so a type that never wrote a non-GBM pricer
    #: refuses the switch instead of pricing GBM under its name.
    spot_models = ('None',)

    #: `System Parameters.Base_Currency` as a checked name, stamped by the compile driver
    #: (`Calculation.set_deal_structures`) because a deal is constructed before the book it prices
    #: against is known. The one seam by which base-vs-foreign reaches a deal's compile surface.
    base_currency = None

    #: Whether the deal is booked over `Children`, stamped by the compile driver as `base_currency`
    #: is: a type whose `post_process` prices its children compiles no legs off its own terms then,
    #: so a reset those terms leave unstated warns nobody.
    has_children = False

    #: The field a market quote of this type lands in, in that field's own unit - a benchmark's
    #: declaration of itself, None for a type no quote set authors. `quote` writes it; a type
    #: whose quote reaches more than one place says how.
    quoted = None

    #: valuation options this type reads, at the value an unstated one means
    option_defaults = {}
    #: `{field: [factor types]}` - read off the type's declarations (`F(..., factors=)`), never
    #: written beside them
    factor_fields = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.factor_fields = declared_factors(cls)

    @classmethod
    def quote(cls, deal, value):
        """Write `value`, a quote, into the authored block `deal` where `quoted` says."""
        deal[cls.quoted] = value

    def __init__(self, params, valuation_options):
        # a declared model this type cannot honour is a malformed program, not a market-data miss
        spot_model = valuation_options.get('SpotModel', 'None')
        if spot_model not in self.spot_models:
            raise ValueError('{0} does not honour SpotModel={1!r}; it accepts {2}'.format(
                type(self).__name__, spot_model, self.spot_models))
        self.options = {**self.option_defaults, **valuation_options}
        # the authored block, answering a read by name from this type's own declarations - the one
        # seam a `default=` reaches a deal through, on `declared_defaults`' ruling for calculations
        self.field = DealFields(params, type(self))
        self.path_dependent = False
        # evaluate child dependencies through the accumulator
        self.accum_dependencies = False

    def reset(self, calendars=None):
        self.reval_dates = set()
        self.settlement_currencies = {}

    def resolve_history(self, base_date, calendars, valuation_options):
        """This deal as its OBSERVED history has already made it, or `self` where decisions remain.

        Called once per deal by the compile driver. A deal with no decisions left is not this deal
        priced through a dead branch, it is the simpler deal it became.
        """
        return self

    def pays_on_grid(self):
        """Whether this deal can pay on a grid date before its expiry - a touch, a rebate, an exercise."""
        return False

    def add_grid_dates(self, parser, base_date, grid):
        if not self.pays_on_grid():
            return
        if isinstance(grid, str):
            grid_dates = parser(base_date, self.field['Expiry_Date'], grid)
            self.reval_dates.update(grid_dates)
            self.settlement_currencies.setdefault(self.payoff_ccy, set()).update(grid_dates)
        else:
            # a grid stated as dates: every date before a currency's last payment pays in it
            for cash_flow in self.settlement_currencies.values():
                last_pmt = max(cash_flow)
                delta = {x for x in grid if x < last_pmt}
                cash_flow.update(delta)
                self.reval_dates.update(delta)

    def add_reval_date_offset(self, offset, relative_to_settlement=True):
        if relative_to_settlement:
            for fixings in self.settlement_currencies.values():
                self.reval_dates.update([x + pd.DateOffset(days=offset) for x in fixings])
        else:
            fixings = reduce(
                set.union, [{x + pd.DateOffset(days=ofs) for ofs in offset}
                            for x in self.reval_dates])
            self.reval_dates.update(fixings)

    def add_reval_dates(self, dates, currency=None):
        self.reval_dates.update(dates)
        if currency:
            self.settlement_currencies.setdefault(currency, set()).update(dates)

    def check_factor_name(self, key, otherwise=None):
        """The factor `key` names (`DealFields.read`), normalised to its tuple, or `otherwise`."""
        value = self.field.read(key)
        return utils.check_rate_name(value) if value else otherwise

    def coupon_schedule(self, calendars, effective, maturity, frequency, leg='',
                        accrual='Accrual_Calendars', timing='Payment_Timing', method='Modified_Following',
                        forward=False):
        """A coupon leg's accrual dates and the day each of its periods pays, off the conventions
        its fields (each prefixed `leg`) state: the dates `utils.coupon_dates` spells from its coupon
        dates, `method` and `forward` on the accrual calendar, and a period paying `Payment_Offset`
        business days of the payment calendar after its end, or its start under `Begin`. A blank
        calendar reads the other one, and an offset on neither counts weekdays."""
        def read(key):
            return self.field.read(leg + key)

        accrual_calendar = read(accrual) or read('Payment_Calendars')
        rolls, pays = (calendars.get(name, {}).get('businessday') for name in (
            accrual_calendar, read('Payment_Calendars') or accrual_calendar))
        dates = utils.coupon_dates(maturity, effective, frequency, rolls, read('First_Coupon_Date'),
                                   read('Penultimate_Coupon_Date'), method, forward)
        offset = int(read('Payment_Offset') or 0)
        return dates, pd.DatetimeIndex([
            day + offset * (pays or pd.offsets.BDay()) if offset else
            utils.adjust_date(pays, 'Modified_Following', day)
            for day in (dates[:-1] if read(timing) == 'Begin' else dates[1:])])

    def float_index(self, calendars):
        """How the deal's floating index fixes (`utils.FloatIndex`), off its `Index_Day_Count`,
        `Reset_Type`, `Index_Offset` and `Index_Calendars`."""
        return utils.FloatIndex(
            utils.DayCount.code(self.field.read('Index_Day_Count')), self.field.read('Reset_Type') == 'Arrears',
            int(self.field.read('Index_Offset', 0)),
            calendars.get(self.field.read('Index_Calendars'), {}).get('businessday'))

    def equity_swaplets(self, field, paid, items, principal, settlement_days, base_date, static_offsets,
                        stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
        """The factor index `pricing.pv_equity_leg` prices the swaplet rows `items` off, paid in
        `paid` on `field`'s `Currency` and `Payoff_Currency` pair (a compo or quanto through
        `check_option_data`) and discounted on its `Discount_Rate`: on a principal where `principal`,
        else on shares, each forward read `settlement_days` business days of the accrual calendar
        after its day. A compo or quanto reinvesting its dividends is not built and refuses by name."""
        field.update(Equity=utils.check_rate_name(self.field['Equity']),
                     Equity_Volatility=self.check_factor_name('Equity_Volatility'))
        bus_day = calendars.get(
            self.field.get('Accrual_Calendars'), {'businessday': pd.offsets.Day(0)})['businessday']
        field_index = {
            'PrincipleNotShares': int(principal),
            'SettleCurrency': paid,
            'Currency': get_fxrate_factor(field['Payoff_Currency'], static_offsets, stochastic_offsets),
            'Discount': get_interest_factor(field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
            'Dividend_Yield': get_dividend_rate_factor(
                field['Equity'][:1], static_offsets, stochastic_offsets, all_tenors),
            'Equity_Zero': get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Expiry': max((x['Payment_Date'] - base_date).days for x in items)}
        field_index['Flows'], field_index['Bus_Ofs'] = utils.TensorCashFlows.equity_swaplet(
            base_date, time_grid, 1 if self.field['Buy_Sell'] == 'Buy' else -1, {'Items': items},
            get_equity_spot(field['Equity'], static_offsets, stochastic_offsets, all_factors),
            settlement_days * bus_day)
        self.check_option_data(field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors,
                               equity_fx_correlation, vol_field='Equity_Volatility')
        if field_index['Check_Payoff_Type'] and any(x.get('Dividend_Multiplier', 1.0) for x in items):
            raise ValueError('{}: {} pays a compo or quanto on a swap reinvesting its dividends, which is not '
                             'built - a price return states Dividend_Multiplier 0 on every row'.format(
                                 self.field.get('Reference'), type(self).__name__))
        return field_index

    def closeout_window(self):
        """The liquidation and settlement periods this deal closes out over, None where it holds no
        collateral."""
        return None

    @property
    def payoff_ccy(self):
        """The currency the payoff is paid in: `Payoff_Currency`, else the deal's own `Currency`."""
        return self.field.read('Payoff_Currency', self.field['Currency'])

    @property
    def payoff_type(self):
        """How the payoff reaches `payoff_ccy`: `Payoff_Type`, else `Standard`."""
        return self.field.read('Payoff_Type', 'Standard')

    def add_settlement_dates(self):
        """Register every day the deal's own declaration settles on under the currency its `settles`
        marker names - the marker's field, else the deal's own - a blank day read as the day the
        marker names in its place."""
        for _, _, cash, _, day in settlement_days(self.field, declared_settlements(type(self))):
            if day is not None:
                self.add_reval_dates({day}, self.field.read(cash.currency, self.field['Currency']))

    def get_reval_dates(self, clip_expiry=False):
        if clip_expiry and bool(self.settlement_currencies):
            last_cashflow = max([max(fixing) for fixing in self.settlement_currencies.values()])
            return {x for x in self.reval_dates if x <= last_cashflow}
        else:
            return self.reval_dates

    def get_report_dates(self, time_grid, base_date):
        """Every MTM date of the grid up to the last this deal revalues on, the base date at least."""
        last = max(set(self.get_reval_dates()).union({base_date}))
        return sorted([x for x in time_grid.mtm_dates if x <= last])

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        if self.path_dependent:
            self.add_grid_dates(parser, base_date, grid if node_children is None else node_resets)

        node_resets.update(self.get_reval_dates())
        for settlement_currency, settlement_dates in self.get_settlement_currencies().items():
            node_settlements.setdefault(settlement_currency, set()).update(settlement_dates)

    def get_settlement_currencies(self):
        return self.settlement_currencies

    def leg_descriptors(self, deal_data):
        """Static (batch-independent) cashflow descriptors: (summed |notional|, latest pay day
        offset). Reads this deal's own `Factor_dep['Cashflows']`, defaulting to (0.0, None) for
        deals with no cashflow schedule."""
        cf = deal_data.Factor_dep.get('Cashflows')
        return (cf.total_abs_nominal(), cf.last_pay_day()) if cf is not None else (0.0, None)

    def calculate(self, shared, time_grid, deal_data):
        """Generate the theo price and interpolate it onto the report grid. A deal the compile
        kept unread, its `Factor_dep` None, marks zero on every date; a pricing failure is swallowed into a scalar-0 mark,
        counted once under `Deals Skipped`, unless `utils.is_fatal_pricing_error` names it a
        framework fault, which an inner-MC fork would read as an expired contract."""
        if deal_data.Factor_dep is None:
            return pricing.interpolate(shared.one.new_zeros(
                (deal_data.Time_dep.deal_time_grid.size, shared.simulation_batch)),
                shared, time_grid, deal_data)
        try:
            mtm = self.generate(shared, time_grid, deal_data)
            return pricing.interpolate(mtm, shared, time_grid, deal_data)
        except Exception as e:
            logging.critical('Deal {} skipped - {}'.format(
                deal_data.Instrument.field.get("Reference"), e.args))
            if utils.is_fatal_pricing_error(e):
                raise
            unpriced = getattr(shared, 'unpriced', None)
            if unpriced is not None and self not in unpriced:
                unpriced.add(self)
                self.unpriced_because = e
                shared.calc_stats['Deals Skipped'] = shared.calc_stats.get('Deals Skipped', 0) + 1
            return 0.0 * shared.one

    def sum_children(self, accum, shared, time_grid, deal_data, child_dependencies):
        """The `post_process` of a deal booked as the sum of its `Children`: each child priced on
        this deal's grid, the sum interpolated onto it."""
        mtm_list = []

        for child in child_dependencies:
            # make the child price to the same grid as the parent
            child.Time_dep.assign(deal_data.Time_dep)
            mtm_list.append(child.Instrument.calculate(shared, time_grid, child))

        mtm = torch.sum(torch.stack(mtm_list), dim=0)

        return pricing.interpolate(mtm, shared, time_grid, deal_data, interpolate_grid=False)

    def children_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        """The `finalize_dates` of a deal booked as the sum of its `Children`: over them it takes
        their settlement dates, stated by its terms it keeps its own."""
        if node_children is not None:
            Deal.reset(self)
            for currency, dates in node_settlements.items():
                self.add_reval_dates(dates, currency)
        return Deal.finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements)

    def build_features(self, shared, time_grid, deal_data):
        try:
            return self.hedge_features(shared, time_grid, deal_data)
        except Exception as e:
            logging.critical('Deal {} skipped - {}'.format(
                deal_data.Instrument.field.get("Reference"), e.args))
            # Same rule as `calculate`: a swallowed leg here drops silently out of the liability
            # accumulation, where a second leg's mark broadcasts over the gap.
            if utils.is_fatal_pricing_error(e):
                raise
            return {}

    def generate(self, shared, time_grid, deal_data):
        raise NotImplementedError('generate in class {} not implemented yet for deal {}'.format(
            self.__class__.__name__, self.field.get('Reference'))
        )

    def hedge_features(self, shared, time_grid, deal_data):
        """The default hedge mark, accumulated by `resolve_hedge_structure` into the liability MTM.

        The same price `calculate` produces, without the per-batch GPU->CPU save_results copy. The
        hedge path consumes only this MTM; a deal type may override to expose extra features."""
        return {'mtm': self.calculate(shared, time_grid, deal_data)}

    def check_option_data(self, field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors,
                          correlation, vol_field=None):
        """Set the FX vol and implied-correlation dependencies of a compo or quanto payoff, paid in
        `field['Payoff_Currency']` on an underlying quoted in `field['Currency']`.

        `correlation(field, all_factors)` names the family's own pair and the sign it is read with -
        `equity_fx_correlation`, `reference_fx_correlation`. `vol_field` names the field holding the
        equity surface a deal pricing no option of its own reads for the quanto carry's ATM vol.
        """
        payoff_type = self.payoff_type
        if payoff_type != 'Standard' and field['Payoff_Currency'] != field['Currency']:
            field_index['Check_Payoff_Type'] = True
            _, fx_lookup = utils.check_fx_name([field['Currency'][0], field['Payoff_Currency'][0]])
            field_index['FXVol'] = get_vol_factor('FXVol', fx_lookup, static_offsets, stochastic_offsets, all_tenors)
            factor, sign = correlation(field, all_factors)
            field_index['Correlation_Sign'] = sign
            field_index['{}ImpliedCorrelation'.format(payoff_type)] = factor

            if payoff_type == 'Compo':
                # needed to calculate fx forwards
                field_index['Local'] = get_fx_and_zero_rate_factor(
                    field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
                field_index['Other'] = get_fx_and_zero_rate_factor(
                    field['Payoff_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
                # the compo is monitored on S*X, so a bridge needs the cross's variance too
                field_index['Compo_Underlying'] = get_cross_barrier_underlying(
                    (field['Currency'], field['Payoff_Currency']), stochastic_offsets)
            if vol_field is not None:
                field_index['Volatility'] = get_vol_factor(
                    'EquityPriceVol', field[vol_field], static_offsets, stochastic_offsets, all_tenors)
        else:
            field_index['Check_Payoff_Type'] = False


def scan_collateral_balance(opening, required, recv_band, post_band, call_mask,
                            start=0, collect_gaps=False):
    """Walk the collateral balance through the margin calls: hold it until it leaves the band
    [recv_band, post_band], then reset it to the required amount.

    Returns `(balance_path, gaps)`; `gaps` is empty unless `collect_gaps`, when it carries
    `(call_index, receive_gap, post_gap, previous, required)` per call date with the autograd graph
    intact. `start` and `opening` restart the same recursion at a margin date from a forced balance,
    which is the whole of a counterfactual replay.

    The transfer test must stay `previous < recv | previous > post` rather than be re-derived from
    the gaps: the two agree in exact arithmetic and not in floating point, and the forward valuation
    must not move when the boundary machinery is switched on.
    """
    path = [opening]
    gaps = []
    for index in range(start + 1, required.shape[0]):
        previous = path[-1]
        if not call_mask[index]:
            path.append(previous)
            continue
        recv, post = recv_band[index], post_band[index]
        transfer = (previous < recv) | (previous > post)
        if collect_gaps:
            gaps.append((index, recv - previous, previous - post, previous, required[index]))
        path.append(required[index] * transfer + previous * (~transfer))
    return torch.stack(path), gaps


def collateral_bands(required, face, cut, fx, min_received, min_posted):
    """`(balance, recv_band, post_band)` for `scan_collateral_balance`: the units of the collateral
    the support `required` asks for and the two edges past which it is transferred, each in the
    unit of the side it falls on - support held counting at `face`, support posted at `face` less
    `cut`, what one less `Haircut_Posted` takes off a unit."""
    side = (required < 0).to(face.dtype)
    unit = face - cut * side
    balance, per_unit = required / unit, fx / unit

    def carried(edge, amount):
        # the ratio of the two sides' units, spelled as one plus the cut's share so that it is
        # exactly one, gradient and all, where nothing is cut
        other = (amount < 0).to(face.dtype)
        return edge * (1.0 + cut * (other - side) / (face - cut * other))

    return (balance, carried(balance - min_received * per_unit, required - min_received * fx),
            carried(min_posted * per_unit + balance, required + min_posted * fx))


def collateral_credit(balance, face, cut):
    """What `balance` units of the collateral are credited at a closeout: held at `face` less `cut`,
    what one less `Haircut_Received` takes off a unit, and posted at `face`, the bank's own asset
    in full."""
    return balance * (face - cut * (balance > 0))


class NettingCollateralSet(Deal):
    accepts_children = True
    vernacular = 'netting set, CSA, client, counterparty agreement'
    #: One row of a credit support list, rating 1 at amount 0, as the wire spells it: no threshold,
    #: no minimum transfer and no independent amount.
    ZERO_SUPPORT = {'.CreditSupportList': [[1, 0.0]]}
    #: The two haircuts every collateral row declares, a row stating neither cutting nothing.
    HAIRCUTS = (F('Haircut_Posted', 'Percent', default=0.0, convention=True),
                F('Haircut_Received', 'Percent', default=0.0, convention=True))
    fields = [ADMIN, own('NettingCollateralSet', [
        F('Agreement_Currency', 'Text', default='', convention=True, factors=['FxRate']),
        F('Apply_Closeout_When_Uncollateralized', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Balance_Currency', 'Text', default='', convention=True, factors=['FxRate']),
        F('Opening_Balance', 'Float', default=0.0, convention=True),
        F('Base_Collateral_Call_Date', 'Date', default='', convention=True),
        F('Calendars', 'Text', default='', convention=True),
        F('Collateral_Assets', 'Container',
          convention=True, sub_fields=[
              F('Cash_Collateral', 'Table', default='null', row=Row([
                  F('Currency', 'Text', factors=['FxRate']), F('Amount', 'Float'), *HAIRCUTS,
                  F('Collateral_Rate', 'Text', factors=['InterestRate']),
                  F('Funding_Rate', 'Text', factors=['InterestRate']), F('Liquidation_Period', 'Integer')])),
              F('Bond_Collateral', 'Table', default='null', row=Row([
                  *HAIRCUTS, F('Liquidation_Period', 'Integer'),
                  F('Issuer', 'Text'), F('Currency', 'Text', factors=['FxRate']),
                  F('Discount_Rate', 'Text', factors=['InterestRate']), F('Maturity', 'Period'),
                  F('Principal', 'Float'), F('Coupon_Rate', 'Percent'), F('Coupon_Interval', 'Period'),
                  F('Collateral_Rate', 'Text'), F('Funding_Rate', 'Text')])),
              F('Equity_Collateral', 'Table', default='null', row=Row([
                  F('Equity', 'Text', factors=['EquityPrice', 'DividendRate']), F('Units', 'Float'),
                  *HAIRCUTS, F('Collateral_Rate', 'Text'),
                  F('Funding_Rate', 'Text'), F('Liquidation_Period', 'Integer')])),
              F('Commodity_Collateral', 'Table', default='null', row=Row([
                  F('Commodity', 'Text'), F('Units', 'Float'), *HAIRCUTS,
                  F('Collateral_Rate', 'Text'), F('Funding_Rate', 'Text'),
                  F('Liquidation_Period', 'Integer')]))]),
        F('Collateral_Call_Frequency', 'Period', default='1D', convention=True),
        F('Collateralized', 'Text', default='False', convention=True, values=['True', 'False']),
        F('Netted', 'Text', default='True', convention=True, values=['True', 'False']),
        F('Credit_Support_Amounts', 'Container',
          convention=True, sub_fields=[
              F('Bank', 'Text', default=''),
              F('Counterparty', 'Text', default='', factors=['SurvivalProb']),
              F('Independent_Amount', 'CreditSupportList', default=ZERO_SUPPORT,
                row=Row([F('Independent Amount', 'Float'), F('Credit Rating', 'Integer')])),
              F('Received_Threshold', 'CreditSupportList', default=ZERO_SUPPORT,
                row=Row([F('Threshold Amount', 'Float'), F('Credit Rating', 'Integer')])),
              F('Posted_Threshold', 'CreditSupportList', default=ZERO_SUPPORT,
                row=Row([F('Threshold Amount', 'Float'), F('Credit Rating', 'Integer')])),
              F('Minimum_Received', 'CreditSupportList', default=ZERO_SUPPORT,
                row=Row([F('Minimum Amount', 'Float'), F('Credit Rating', 'Integer')])),
              F('Minimum_Posted', 'CreditSupportList', default=ZERO_SUPPORT,
                row=Row([F('Minimum Posted', 'Float'), F('Credit Rating', 'Integer')]))]),
        F('Funding_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Liquidation_Period', 'Integer', default=0, convention=True),
        F('Settlement_Period', 'Integer', default=0, convention=True),
        F('Post_Regulations', 'Text', default='', convention=True,
          description='The regulatory regimes initial margin is POSTED under this agreement by, as '
                      'the CRIF carries them (comma-separated where several), blank where none is '
                      'declared'),
        F('Collect_Regulations', 'Text', default='', convention=True,
          description='The regimes initial margin is COLLECTED under this agreement by, as the '
                      'CRIF carries them; a term of the agreement, blank where none is declared')
    ])]

    documentation = ('Container', [
        'Collateral agreements (CSA\'s) are represented using a container instrument called a netting ',
        'collateral set. The idea is to first model the effect of an uncollateralized net portfolio ',
        '$V(t)$ and then, per scenario, transform this to a collateralized portfolio $\\hat V(t)$.',
        '',
        'The transformation runs only in a simulation of more than one scenario; a single-scenario',
        'run reports the uncollateralised net.',
        '',
        'In addition to posting and receiving collateral there are still two residual risks viz.',
        '',
        '### Settlement Risk',
        '',
        'This arises when counterparty default occurs unexpectedly. Potentially, a party may make ',
        'payment (or post collateral) to the counterparty without receiving the corresponding collateral',
        '(or payment) in return.',
        '',
        '### Liquidity risk',
        '',
        'This refers to the basis risk between the market cost of closing out or replacing the counterparty',
        'portfolio against the realized market value of the collateral held.',
        '',
        'The general approach to simulating collateral is as follows:',
        '',
        'Define:',
        '',
        '- $t$ the simulation time',
        '- $V(t)$ a realization of the uncollateralized portfolio for a scenario',
        '- $\\hat V(t)$ a realization of collateralized portfolio for a scenario',
        '- $B(t)$ the number of units of the collateral portfolio that should be held if the collateral '
        'agreement was honoured by both parties. This includes minimum transfer amounts and is piecewise '
        'constant between collateral call dates',
        '- $S(t)$ is the value of one unit of the collateral portfolio in base currency for a scenario',
        '- $\\delta_s$ the length of the settlement period',
        '- $\\delta_l$ the length of the liquidation period',
        '- $t_s$ the start of the settlement period',
        '- $t_l$ the start of the liquidation period',
        '- $t_e$ the end of the liquidation period',
        '- $C(t_1,t_2)$ the value in base currency of cash in all currencies accumulated by the portfolio'
        ' over the interval $[t_1,t_2]$',
        '',
        'The closeout period (when the counterparty becomes insolvent) starts at $t_s$ and ends at $t_e$',
        '(when the position and collateral have been liquidated). This period is further divided into a',
        'settlement period $\\delta_s$ followed by a liquidation period $\\delta_l$ such that the',
        'liquidation begins at $t_l$ when the settlement period ends. After the settlement period,',
        'neither party pays cashflows or transfers collateral but the market risk on the portfolio and',
        'collateral continue until the end of liquidation period [@gregory2015; @andersen2017].',
        '',
        'The collateralized portfolio at time $t$ is the difference between the sum of the liquidated',
        'portfolio value and the cash accumulated during the closeout period $C(t_s,t_e)$ and the',
        'credit for the $\\underline B=\\min\\{B(u):t_s\\le u\\le t_l\\}$ units of the collateral',
        'portfolio: collateral held is credited at its liquidated value less its received haircuts,',
        'and collateral posted stays the bank\'s asset at its full value:',
        '',
        '$$\\hat V(t)=V(t_e)+C(t_s,t_e)-\\underline B\\,S_c(t_e),\\qquad S_c=\\begin{cases}S_r, '
        '&\\underline B>0\\\\ S, &\\text{otherwise}\\end{cases}$$',
        '',
        'The calculation of $C(t_s,t_e)$, $B(t)$, $S(t)$ and $S_r(t)$ is described below; the',
        'collateral reported is the balance at its full value, $B(t)S(t)$.',
        '',
        '### Standard and Forward looking closeout',
        '',
        'Usually exposure is reported at the end of the closeout period i.e. :',
        '',
        '- $t_s=t-\\delta_l-\\delta_s$',
        '- $t_l=t-\\delta_l$',
        '- $t_e=t$',
        '',
        'However, it is also possible to model the settlement and liquidation periods to come after $t$.',
        'This forward looking closeout implies:',
        '',
        '- $t_s=t$',
        '- $t_l=t+\\delta_s$',
        '- $t_e=t+\\delta_s+\\delta_l$',
        '',
        'Note that Standard closeout causes exposure to be reported one closeout period after portfolio',
        'maturity (as the mechanics of default are still present).',
        '',
        'Every collateralised set in one job closes out over the same $\\delta_l$ and $\\delta_s$; a job',
        'whose sets differ is refused by name. Beside an uncollateralised set, which reports every date',
        'of the grid, a collateralised set contributes zero on its own $t_l$ and $t_s$ dates.',
        '',
        '### Cashflow Accounting',
        '',
        'Define (for time $t$):',
        '',
        '- $C_r(t)$ the unsigned cumulative base currency amount of cash received per scenario',
        '- $C_p(t)$ the unsigned cumulative base currency amount of cash paid per scenario',
        '- $C_i(t)$ the signed net cash amount paid per scenario per currency $i$ (positive if received,'
        ' negative if paid)',
        '- $X_i(t)$ the exchange rate from currency $i$ to base currency',
        '',
        'Interest over the closeout period is assumed negligible and hence not calculated. If **Exclude'
        ' Paid Today** is **No**,',
        'then the portfolio value at $t$ includes cash paid on $t$ and hence:',
        '',
        '$$C_r(t)=\\sum_{t_j<t}\\sum_i X_i(t_j)\\max(0,C^i(t_j))$$',
        '',
        '$$C_p(t)=\\sum_{t_j<t}\\sum_i X_i(t_j)\\max(0,-C^i(t_j))$$',
        '',
        'Otherwise (when **Exclude Paid Today** is **Yes**):',
        '',
        '$$C_r(t)=\\sum_{t_j\\le t}\\sum_i X_i(t_j)\\max(0,C^i(t_j))$$',
        '',
        '$$C_p(t)=\\sum_{t_j\\le t}\\sum_i X_i(t_j)\\max(0,-C^i(t_j))$$',
        '',
        '#### Settlement risk mechanics',
        '',
        'The **Cash Settlement Risk** mechanics are determined depending on the order of cash payments'
        ' versus collateral.',
        '',
        '- **Received Only** assumes that collateral is transferred before cash. All cash is retained'
        ' during the settlement period.',
        '',
        '$$C(t_s,t_e)=C_r(t_e)-C_p(t_e)-C_r(t_s)+C_p(t_s)$$',
        '',
        '- **Paid Only** assumes that cash is transferred before collateral. Cash is paid by both sides',
        'and no cash is retained during the settlement period.',
        '',
        '$$C(t_s,t_e)=C_r(t_e)-C_p(t_e)-C_r(t_l)+C_p(t_l)$$',
        '',
        '- **All** assumes that the ordering of cash and collateral is not fixed (i.e. paid and received'
        ' cash are at risk). Only received cash is retained during the settlement period.',
        '',
        '$$C(t_s,t_e)=C_r(t_e)-C_p(t_e)-C_r(t_s)+C_p(t_l)$$',
        '',
        '### Collateral Accounting',
        '',
        'Define:',
        '',
        '- $A(t_i)$ the agreed value in base currency of collateral that should be held per scenario if',
        ' the CSA was honoured by both parties *ignoring minimum transfer amounts*. Collateral is not',
        'held simultaneously by both parties. Received collateral is positive and posted is negative.',
        '- $A(0)$ the initial value of collateral in base currency specified by the **Opening Balance**.',
        '- $X(t)$ the exchange from the CSA currency to the base currency. All amounts on the CSA',
        '(Thresholds, minimum transfers etc.) are expressed in the **Agreement Currency**.',
        '- $I$ the independent amount of collateral in agreement currency that is either posted (negative)'
        ' or received (positive).',
        '- $H(t)$ the received threshold of collateral: if the portfolio value is above this, the agreed '
        'collateral value must be increased by the difference.',
        '- $G(t)$ the posted threshold of collateral: if the portfolio value is below this, the agreed '
        'collateral value must be decreased by the difference.',
        '- $M_r(t)$ the minimum received transfer amount. The collateral held will not increase unless '
        'the increase is more than this amount.',
        '- $M_p(t)$ the minimum posted transfer amount. The collateral posted will not increase unless '
        'the increase is more than this amount.',
        '- $W_t(b)$ the credit support a balance of $b$ units counts for at $t$: $bS(t)$ where it is',
        'held, at face, and $bS_p(t)$ where it is posted, at one less its posted haircuts.',
        '- $t_i, i>0$ are collateral call dates. Note that $t_0=0$ and that in general, $t_0$ need not be'
        ' a collateral call date.',
        '',
        'We then have the following relationships:',
        '',
        '$$A(t_i)=X(t_i)I+\\begin{cases} V(t_i)-X(t_i)H(t_i), '
        '&\\text{if } V(t_i)>X(t_i)H(t_i)\\\\ V(t_i)-X(t_i)G(t_i), '
        '&\\text{if } V(t_i)<X(t_i)G(t_i)\\\\ 0, &\\text{otherwise} \\end{cases}$$',
        '',
        'In general, the presence of minimum transfer amounts introduce a path dependency on $B(t)$ and,',
        'as such, is not a simple function of $A(t_i)$. Instead, it can be expressed via the following',
        'recurrence:',
        '',
        '$$B(t_i)=\\begin{cases} W_0^{-1}(A(0)), &\\text{if } i=0\\\\',
        'W_{t_i}^{-1}(A(t_i)), &\\text{if } A(t_i)-W_{t_i}(B(t_{i-1}))>M_r(t_i)X(t_i)'
        '\\text{ and } i>0,\\\\',
        '&\\text{or if } W_{t_i}(B(t_{i-1}))-A(t_i)>M_p(t_i)X(t_i)\\text{ and } i>0\\\\',
        'B(t_{i-1}), &\\text{otherwise}\\end{cases}$$',
        '',
        'Since $B(t_i)$ is constant between call dates, $B(t)=B(t_{i^*})$, where $t_{i^*}$ is the closest',
        'call date on or before $t$. Since $B(t)$ is path dependent, it requires calculation on all',
        'collateral call dates. Due to this being fairly prohibitive, the recurrence is only evaluated',
        'at collateral call dates associated with a particular simulation time grid. This approximation',
        'can be improved by using a finer simulation grid (also note that the path dependency dissipates',
        'as the minimum transfer amounts reduce to zero).',
        '',
        '### Collateral Portfolio',
        '',
        'Define:',
        '',
        '- $a_i$ the number of units of the $i^{th}$ asset in the collateral portfolio.',
        '- $S_i(t)$ the base currency value per unit of the $i^{th}$ asset.',
        '',
        'The collateral portfolio may hold cash (**Amount** of a currency), equities (**Units** at the',
        'equity price) and fixed-coupon bonds (**Principal**, **Coupon_Rate** every **Coupon_Interval**',
        'to **Maturity** from the base date, valued on their **Discount_Rate**), so it is sensitive to',
        'interest rates, FX and equity prices over the closeout period; **Commodity_Collateral** is',
        'declared and not read. With none stated, the collateral is cash in the **Balance_Currency**.',
        'The relative amounts of the assets are held constant, so each asset\'s share of the portfolio',
        'moves with its price. The value of the portfolio is:',
        '',
        '$$S(t)=\\sum_i a_i S_i(t),\\qquad S_p(t)=\\sum_i (1-h^p_i)a_iS_i(t),\\qquad '
        'S_r(t)=\\sum_i (1-h^r_i)a_iS_i(t)$$',
        '',
        'where $h^p_i$ and $h^r_i$ are the row\'s **Haircut_Posted** and **Haircut_Received**, nothing',
        'where it states none: collateral the bank posts is delivered at face over one less its',
        'posted haircut, and collateral it holds is delivered at face and credited at one less its',
        'received haircut. Haircuts must be strictly less than one.',
    ])

    option_defaults = {'Cash_Settlement_Risk': utils.CASH_SETTLEMENT_Received_Only,
                       'Forward_Looking_Closeout': False,
                       'Exclude_Paid_Today': False}

    def __init__(self, params, valuation_options):
        super(NettingCollateralSet, self).__init__(params, valuation_options)
        self.path_dependent = True
        self.calendar = None

    def reset(self, calendars):
        super(NettingCollateralSet, self).reset()
        self.accum_dependencies = True
        calendar = calendars.get(self.field.get('Calendars')) if self.closeout_window() else None
        if calendar:
            self.calendar = calendar['businessday']

    def calc_liquidation_settlement_dates(self, date, base_date):
        if self.options['Forward_Looking_Closeout']:
            tl = date + pd.DateOffset(days=self.field['Settlement_Period'])
            ts = date
        else:
            tl = max(date - pd.DateOffset(days=self.field['Liquidation_Period']), base_date)
            ts = max(date - pd.DateOffset(days=self.field['Settlement_Period'] + self.field['Liquidation_Period']),
                     base_date)
        return ts, tl

    def closeout_window(self):
        """The liquidation and settlement periods a collateralised set closes out over."""
        if self.field.get('Collateralized', 'False') == 'True':
            return self.field['Liquidation_Period'], self.field['Settlement_Period']

    def get_report_dates(self, time_grid, base_date):
        reval_dates = super(NettingCollateralSet, self).get_report_dates(time_grid, base_date)
        if self.field.get('Collateralized', 'False') == 'True':
            # a collateralised date reports where its settlement and liquidation dates are on the grid
            reval_dates = [date for date in reval_dates if all(
                x in time_grid.mtm_dates for x in self.calc_liquidation_settlement_dates(date, base_date))]
        return reval_dates

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        if self.field.get('Collateralized', 'False') == 'True':
            for child in node_children:
                child.add_reval_date_offset(1)
                settle_liquid = {self.field['Settlement_Period'],
                                 self.field['Settlement_Period'] + self.field['Liquidation_Period']}
                child.add_reval_date_offset(settle_liquid, relative_to_settlement=False)
                node_resets.update(child.get_reval_dates())

            grid_dates = parser(base_date, max(node_resets), grid)

            for dates in node_settlements.values():
                valid_fixings = np.clip(list(dates), base_date, max(dates))
                grid_dates.update(valid_fixings)

            node_additions = set()
            node_additions.add(base_date + pd.DateOffset(days=1))
            base_call_date = self.field['Base_Collateral_Call_Date'] if self.field.get(
                'Base_Collateral_Call_Date') else base_date
            call_freq = self.field['Collateral_Call_Frequency'] if self.field.get(
                'Collateral_Call_Frequency') else pd.DateOffset(days=1)

            if call_freq.kwds != {'days': 1}:
                coll_dates = utils.generate_dates_forward(
                    max(grid_dates), base_call_date, call_freq,
                    bus_day=self.calendar, method='Modified_Following')
                grid_dates.update(coll_dates)

            full_grid = grid_dates.union({x for x in node_resets if x >= base_date})

            for t in sorted(full_grid):
                ts, tl = self.calc_liquidation_settlement_dates(t, base_date)
                node_additions.update({t, ts, tl})

            fresh_grid = full_grid.union(node_additions)
            self.reval_dates = set()

            for date in sorted(fresh_grid):
                ts, tl = self.calc_liquidation_settlement_dates(date, base_date)
                if ts in fresh_grid and tl in fresh_grid:
                    self.reval_dates.add(date)

            node_resets.update(node_additions)

            final_grid = np.array(sorted(node_resets))
            for child in node_children:
                if child.path_dependent:
                    child.finalize_dates(parser, base_date, grid, node_children, node_resets, node_settlements)
                else:
                    child_expiry = max(child.get_reval_dates())
                    child.add_reval_dates(set(final_grid[:final_grid.searchsorted(child_expiry)]))
        else:
            for child in node_children:
                child.add_reval_date_offset(1)
                node_resets.update(child.get_reval_dates())

            self.reval_dates = node_resets

    def validate(self):
        """What a CSA cannot be without once it is collateralised, and a cash row asking for ColVA
        must fund.

        The agreement currency is read on the collateralised branch alone, so an uncollateralised
        set states none and a collateralised one cannot be priced without it - a rule spanning two
        fields, which is this method rather than a declaration. Collateral_Rate and Funding_Rate are
        each optional on the row, but post_process enters the ColVA block on Collateral_Rate alone
        and then reads Funding_Rate unconditionally. Reads defensively: validation runs BEFORE the
        contract is known to hold.
        """
        if self.field.get('Collateralized', 'False') == 'True' \
                and not self.field.get('Agreement_Currency'):
            yield 'Collateralized is True, so Agreement_Currency is required'
        rows = (self.field.get('Collateral_Assets') or {}).get('Cash_Collateral') or []
        for i, row in enumerate(rows):
            if row.get('Collateral_Rate') and not row.get('Funding_Rate'):
                yield ('Collateral_Assets.Cash_Collateral[{}]: Collateral_Rate is set, so '
                       'Funding_Rate is required'.format(i))

    @classmethod
    def haircuts(cls, row):
        """A collateral row's `{Haircut_Posted, Haircut_Received}` as fractions, one it states none
        of read as the convention its column declares."""
        row = DealFields(row, declared={f.key: engine_default(f) for f in cls.HAIRCUTS})
        return {f.key: float(row[f.key]) for f in cls.HAIRCUTS}

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {}
        field_index = {}

        if 'Credit_Support_Amounts' in self.field and self.field['Credit_Support_Amounts'].get('Counterparty'):
            try:
                field_index['Survival_Prob'] = get_survival_factor(
                    utils.check_rate_name(self.field['Credit_Support_Amounts']['Counterparty']),
                    static_offsets, stochastic_offsets, all_tenors)
            except:
                # if it's missing make the Survival_Prob None (so that later we can set it to 1)
                field_index['Survival_Prob'] = None

        if self.field.get('Collateralized', 'False') == 'True':
            field['Agreement_Currency'] = utils.check_rate_name(self.field['Agreement_Currency'])
            field['Funding_Rate'] = self.check_factor_name('Funding_Rate')

            # defaults to the agreement currency, not the base currency the convention suggests
            field['Balance_Currency'] = self.check_factor_name('Balance_Currency', field['Agreement_Currency'])

            field_index['Agreement_Currency'] = get_fxrate_factor(
                field['Agreement_Currency'], static_offsets, stochastic_offsets)
            field_index['Balance_Currency'] = get_fxrate_factor(
                field['Balance_Currency'], static_offsets, stochastic_offsets)

            field_index['Settlement_Currencies'] = {}
            for currency in time_grid.CurrencyMap.keys():
                field_index['Settlement_Currencies'].setdefault(
                    currency, get_fxrate_factor(utils.check_rate_name(currency), static_offsets, stochastic_offsets))

            collateral_defined = False
            field_index['Equity_Collateral'] = []
            field_index['Bond_Collateral'] = []
            field_index['Cash_Collateral'] = []

            collateral_assets = self.field.get('Collateral_Assets')

            if collateral_assets:
                if collateral_assets.get('Equity_Collateral'):
                    collateral_equity = self.field['Collateral_Assets']['Equity_Collateral']

                    for col_equity in collateral_equity:
                        equity_rate = utils.check_rate_name(col_equity['Equity'])
                        field_index['Equity_Collateral'].append(
                            utils.Collateral(**self.haircuts(col_equity),
                                             Amount=col_equity['Units'],
                                             Currency=get_equity_currency_factor(
                                                 equity_rate, static_offsets, stochastic_offsets, all_factors),
                                             Funding_Rate=None, Collateral_Rate=None,
                                             Collateral=get_equity_rate_factor(
                                                 equity_rate, static_offsets, stochastic_offsets))
                        )
                    collateral_defined = True

                if collateral_assets.get('Bond_Collateral'):
                    collateral_bond = self.field['Collateral_Assets']['Bond_Collateral']

                    for col_bond in collateral_bond:
                        if np.array(list(col_bond['Coupon_Interval'].kwds.values())).any():
                            reset_dates = utils.generate_dates_backward(
                                base_date + col_bond['Maturity'], base_date, col_bond['Coupon_Interval'])
                        else:
                            reset_dates = np.array([base_date, base_date + col_bond['Maturity']])
                        fixed_cash = utils.TensorCashFlows.generate_fixed(
                            base_date, reset_dates, col_bond['Principal'], None, 0, float(col_bond['Coupon_Rate']))
                        # make sure there's a nominal repayment at maturity
                        fixed_cash.add_fixed_payments(base_date, 'Maturity', base_date, 0, col_bond['Principal'])
                        discount = get_interest_factor(utils.check_rate_name(col_bond['Discount_Rate']),
                                                       static_offsets, stochastic_offsets,
                                                       all_tenors)
                        time_index = np.searchsorted(time_grid.mtm_time_grid, (max(reset_dates) - base_date).days)

                        field_index['Bond_Collateral'].append(
                            utils.Collateral(**self.haircuts(col_bond),
                                             Amount=1,
                                             Currency=get_fxrate_factor(
                                                 utils.check_rate_name(col_bond['Currency']),
                                                 static_offsets, stochastic_offsets),
                                             Funding_Rate=None, Collateral_Rate=None,
                                             Collateral=utils.DealDataType(
                                                 Instrument=None,
                                                 Factor_dep={'Cashflows': fixed_cash, 'Discount': discount},
                                                 Time_dep=utils.DealTimeDependencies(
                                                     time_grid.mtm_time_grid, np.arange(time_index)),
                                                 Calc_res=None))
                        )
                    collateral_defined = True

                if collateral_assets.get('Cash_Collateral'):
                    collateral_cash = self.field['Collateral_Assets']['Cash_Collateral']
                    # `Amount` weighs an asset in the collateral the balance counts: a sole asset's
                    # balance cancels it, so a sole row stating none is a unit; among several it is not
                    sole = len(collateral_cash) == 1 and not collateral_defined
                    for col_cash in collateral_cash:
                        if col_cash.get('Amount') is None and not sole:
                            raise ValueError('{}: a Cash_Collateral row states no Amount, its weight '
                                             'among several collateral assets'.format(
                                self.field.get('Reference')))
                        field_index['Cash_Collateral'].append(
                            utils.Collateral(**self.haircuts(col_cash),
                                             Amount=1.0 if col_cash.get('Amount') is None
                                             else col_cash['Amount'],
                                             Currency=get_fxrate_factor(
                                                 utils.check_rate_name(col_cash['Currency']),
                                                 static_offsets, stochastic_offsets),
                                             Funding_Rate=get_interest_factor(
                                                 utils.check_rate_name(
                                                     col_cash['Funding_Rate']), static_offsets,
                                                 stochastic_offsets, all_tenors)
                                             if col_cash.get('Funding_Rate') else None,
                                             Collateral_Rate=get_interest_factor(
                                                 utils.check_rate_name(
                                                     col_cash['Collateral_Rate']),
                                                 static_offsets, stochastic_offsets, all_tenors)
                                             if col_cash.get('Collateral_Rate') else None,
                                             Collateral=1.0)
                        )
                    collateral_defined = True

            if not collateral_defined:
                # default the collateral to the balance currency
                field_index['Cash_Collateral'] = [
                    utils.Collateral(Haircut_Posted=0.0, Haircut_Received=0.0,
                                     Amount=1.0,
                                     Currency=get_fxrate_factor(
                                         field['Balance_Currency'], static_offsets, stochastic_offsets),
                                     Funding_Rate=None, Collateral_Rate=None,
                                     Collateral=1.0)]

            if self.field['Credit_Support_Amounts'].get('Independent_Amount'):
                field_index['Independent_Amount'] = self.field['Credit_Support_Amounts']['Independent_Amount'].value()
            else:
                field_index['Independent_Amount'] = 0.0

            t = time_grid.time_grid[:, utils.TIME_GRID_MTM]
            # collateral call dates (interpolated - assuming daily)
            base_call_date = self.field['Base_Collateral_Call_Date'] if self.field.get(
                'Base_Collateral_Call_Date') else base_date
            call_freq = self.field['Collateral_Call_Frequency'] if self.field.get(
                'Collateral_Call_Frequency') else pd.DateOffset(days=1)

            # normally this would just be the reval dates, but if we have more than 1 netting set,
            # then we need to include the global (time_grid.mtm_dates) set of dates
            reval_dates = self.get_report_dates(time_grid, base_date)

            call_dates = np.array([(x - base_date).days for x in reval_dates])
            # where each reported row sits on the global grid the root sums every set on
            field_index['Report_Rows'] = np.searchsorted(t, call_dates)
            call_mask = np.ones(time_grid.mtm_time_grid.size, dtype=np.int32)

            if call_freq.kwds != {'days': 1}:
                all_call_days = utils.generate_dates_forward(
                    max(self.get_reval_dates()), base_call_date, call_freq,
                    bus_day=self.calendar, method='Modified_Following')
                approx_calls = pd.DatetimeIndex(sorted(time_grid.mtm_dates)).intersection(all_call_days)
                mod_call_dates = np.array([(x - base_date).days for x in approx_calls])

                call_mask[np.searchsorted(time_grid.mtm_time_grid, mod_call_dates)] = 2
                call_mask -= 1

            field_index['call_mask'] = call_mask

            if self.options['Forward_Looking_Closeout']:
                field_index['Ts'] = (
                        np.searchsorted(t, call_dates, side='right').astype(np.int32) - 1
                ).clip(0, time_grid.mtm_time_grid.size - 1)
                field_index['Tl'] = (
                        np.searchsorted(t, call_dates + self.field['Settlement_Period'], side='right').astype(
                            np.int32) - 1).clip(0, time_grid.mtm_time_grid.size - 1)
                field_index['Te'] = (
                        np.searchsorted(
                            t, call_dates + self.field['Settlement_Period'] + self.field['Liquidation_Period'],
                            side='right').astype(np.int32) - 1).clip(0, time_grid.mtm_time_grid.size - 1)
            else:
                field_index['Ts'] = (
                        np.searchsorted(t,
                                        call_dates - self.field['Settlement_Period'] - self.field['Liquidation_Period'],
                                        side='right').astype(np.int32) - 1).clip(0, time_grid.mtm_time_grid.size - 1)
                field_index['Tl'] = (
                        np.searchsorted(t, call_dates - self.field['Liquidation_Period'], side='right').astype(
                            np.int32) - 1).clip(0, time_grid.mtm_time_grid.size - 1)
                field_index['Te'] = (
                        np.searchsorted(t, call_dates, side='right').astype(np.int32) - 1
                ).clip(0, time_grid.mtm_time_grid.size - 1)

        return field_index

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        """Collateralise the accumulated child MTM and report the netting set's exposure.

        Under `boundary_aad` it also publishes two counterfactual closures for the boundary-aware
        sensitivity engine, both built from detached captures so a replay produces a COEFFICIENT
        (the objective jump) rather than a differentiated quantity:

        - MTA boundary set - the balance reaches the exposure only through min_Bt, so nothing needs
          re-pricing: `replay_net_mtm` replays the captured balance-independent arithmetic on a
          different balance path, and `rescan` restarts the forward walk at a margin date.
        - `gross_to_net` - a PRICER event moves the gross, which reaches the net both through Vte
          and through the balance the scan derives from it, so the counterfactual redoes that whole
          chain (At, required balance, bands, scan). The delta arrives on the MTM grid."""
        # calc v^t = v(te) + C(ts,te) - min(B(u); ts<=u<=tl}S(te)
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        logging.info('Netting set {}'.format(self.field.get('Reference')))

        surv = None
        if hasattr(shared, 'scale_survival') and shared.scale_survival:
            if factor_dep.get('Survival_Prob'):
                surv = utils.calc_time_grid_curve_rate(factor_dep['Survival_Prob'], np.zeros((1, 3)), shared)
            else:
                logging.warning('Survival Probability for Netting set {} not found - Setting to 1.0'.format(
                    self.field.get('Reference')))

        if self.field.get('Collateralized', 'False') == 'True' and shared.simulation_batch > 1:
            C_base = 0.0

            for curr, fx_factor in factor_dep['Settlement_Currencies'].items():
                Ci = []
                T = sorted(shared.t_Cashflows[curr].keys())
                for index, pad in enumerate(np.diff([-1] + T + [time_grid.time_grid.shape[0]]) - 1):
                    if index < len(T):
                        cashflow_at_index = torch.unsqueeze(shared.t_Cashflows[curr][T[index]], dim=0)
                        Ci.append(Fn.pad(cashflow_at_index, [0, 0, pad, 0]))

                base_Ci = torch.cat(Ci, dim=0)
                padded_time_grid = time_grid.time_grid[:-pad] if pad else time_grid.time_grid
                C_base += Fn.pad(utils.calc_time_grid_spot_rate(
                    fx_factor, padded_time_grid, shared) * base_Ci, [0, 0, 0, pad])

            if not self.options['Exclude_Paid_Today']:
                C_block = C_base[:-1]
                Cf_Rec = Fn.pad(torch.cumsum(torch.relu(C_block), dim=0), [0, 0, 1, 0])
                Cf_Pay = Fn.pad(torch.cumsum(torch.relu(-C_block), dim=0), [0, 0, 1, 0])
            else:
                Cf_Rec = torch.cumsum(torch.relu(C_base), dim=0)
                Cf_Pay = torch.cumsum(torch.relu(-C_base), dim=0)

            # The time_grid is global (across all potential netting sets) - the local grid is just this netting set
            local_time_grid = time_grid.time_grid[
                time_grid.time_grid[:, utils.TIME_GRID_MTM] <= deal_time[-1][utils.TIME_GRID_MTM]]
            local_accum = accum[:len(local_time_grid)]

            assets = [(cash_col, cash_col.Amount * utils.calc_time_grid_spot_rate(
                cash_col.Currency, local_time_grid, shared)) for cash_col in factor_dep['Cash_Collateral']]

            for bond_col in factor_dep['Bond_Collateral']:
                bond = pricing.pv_fixed_cashflows(shared, time_grid, bond_col.Collateral, settle_cash=False)
                padding = local_time_grid.shape[0] - bond.shape[0]
                assets.append((bond_col, utils.calc_time_grid_spot_rate(
                    bond_col.Currency, local_time_grid, shared) * Fn.pad(bond, [0, 0, 0, padding])))

            for equity_col in factor_dep['Equity_Collateral']:
                assets.append((equity_col, equity_col.Amount * utils.calc_time_grid_spot_rate(
                    equity_col.Currency, local_time_grid, shared) * utils.calc_time_grid_spot_rate(
                    equity_col.Collateral, local_time_grid, shared)))

            # one unit of the collateral at face, and what one less each haircut takes off it
            St = posted_cut = held_cut = shared.one.new_zeros(len(local_time_grid), shared.simulation_batch)
            for col, value in assets:
                St = St + value
                posted_cut = posted_cut + col.Haircut_Posted * value
                held_cut = held_cut + col.Haircut_Received * value

            fx_base = utils.calc_time_grid_spot_rate(shared.Report_Currency, local_time_grid, shared)
            fx_agreement = utils.calc_time_grid_spot_rate(factor_dep['Agreement_Currency'], local_time_grid, shared)
            H = self.field['Credit_Support_Amounts']['Received_Threshold'].value() * fx_agreement
            G = self.field['Credit_Support_Amounts']['Posted_Threshold'].value() * fx_agreement
            min_received = self.field['Credit_Support_Amounts']['Minimum_Received'].value()
            min_posted = self.field['Credit_Support_Amounts']['Minimum_Posted'].value()

            Vt = local_accum * fx_base
            # fx_base is REBOUND further down onto the Te grid; a gross counterfactual needs the
            # local-grid one that Vt itself was built with, so keep it while it still means that.
            fx_base_local = fx_base

            if self.options['Exclude_Paid_Today']:
                # each leg is differenced against ITSELF, as the Te-grid twin below does
                mtm_today_adj = torch.cat([Cf_Rec[0].reshape(1, -1), Cf_Rec[1:] - Cf_Rec[:-1]], dim=0) - \
                                torch.cat([Cf_Pay[0].reshape(1, -1), Cf_Pay[1:] - Cf_Pay[:-1]], dim=0)
                Vt -= mtm_today_adj

            At = factor_dep['Independent_Amount'] * fx_agreement + (Vt - H) * (Vt > H) + (Vt - G) * (Vt < G)

            opening = self.field['Opening_Balance'] * utils.calc_time_grid_spot_rate(
                factor_dep['Balance_Currency'], np.array([[0, 0, 0]]), shared)
            Bt = opening / (St[0] - posted_cut[0] * (opening < 0))
            Bt_new, Mr, Mp = collateral_bands(At, St, posted_cut, fx_agreement, min_received, min_posted)

            # one recursion for both cases: an all-True call mask IS the daily schedule
            boundary_aad = getattr(shared, 'boundary_aad', False)
            Bt, mta_gaps = scan_collateral_balance(
                Bt[0], Bt_new, Mr, Mp, factor_dep['call_mask'].astype(bool),
                collect_gaps=boundary_aad)

            if boundary_aad:
                # a minimum transfer amount makes the balance jump, so the decision carries a
                # derivative ordinary AAD drops. Record both sides; the correction is assembled
                # against the CVA objective, which does not exist until later.
                mta_events = []
                for index, recv_gap, post_gap, previous, required in mta_gaps:
                    mta_events.extend([
                        utils.MTABoundaryEvent(index, 'receive', recv_gap,
                                               previous.detach(), required.detach()),
                        utils.MTABoundaryEvent(index, 'post', post_gap,
                                               previous.detach(), required.detach())])
                # one registration per run of held balance - the whole series decides it
                utils.mark_binding_calls(mta_events)
                boundary_balance = Bt.detach()

            report_time = local_time_grid[factor_dep['Te']]
            fx_base = utils.calc_time_grid_spot_rate(shared.Report_Currency, report_time, shared)

            Vte = local_accum[factor_dep['Te']] * fx_base

            if self.options['Exclude_Paid_Today']:
                mtm_today_adj = (Cf_Rec[factor_dep['Te']] -
                                 Fn.pad(Cf_Rec[factor_dep['Te'][1:] - 1], [0, 0, 1, 0])) - (
                                        Cf_Pay[factor_dep['Te']] - Fn.pad(Cf_Pay[factor_dep['Te'][1:] - 1],
                                                                          [0, 0, 1, 0]))

                Vte -= mtm_today_adj

            if self.options['Cash_Settlement_Risk'] == utils.CASH_SETTLEMENT_Received_Only:
                C_ts_te = (Cf_Rec[factor_dep['Te']] - Cf_Rec[factor_dep['Ts']]) - \
                          (Cf_Pay[factor_dep['Te']] - Cf_Pay[factor_dep['Ts']])
            elif self.options['Cash_Settlement_Risk'] == utils.CASH_SETTLEMENT_Paid_Only:
                C_ts_te = (Cf_Rec[factor_dep['Te']] - Cf_Rec[factor_dep['Tl']]) - \
                          (Cf_Pay[factor_dep['Te']] - Cf_Pay[factor_dep['Tl']])
            else:
                C_ts_te = (Cf_Rec[factor_dep['Te']] - Cf_Rec[factor_dep['Ts']]) - \
                          (Cf_Pay[factor_dep['Te']] - Cf_Pay[factor_dep['Tl']])

            # note that this should be the minimum Bt from factor_dep['Ts'] to factor_dep['Tl']
            Ste, held_cut_te = St[factor_dep['Te']], held_cut[factor_dep['Te']]
            Bte = Bt[factor_dep['Te']]
            base_i = factor_dep['Ts'][:-1]
            delta_T = factor_dep['Tl'][:-1] - base_i
            min_Bt = Bt[base_i]
            b_index = np.zeros_like(delta_T)

            for i in range(delta_T.max() if delta_T.size else 0):
                b_index[delta_T > i] = i + 1
                min_Bt = torch.min(min_Bt, Bt[base_i + b_index])

            # zero out the last row
            min_Bt = Fn.pad(min_Bt, [0, 0, 0, 1])
            expected_collateral = Bte * Ste / fx_base

            net_accum = (Vte + C_ts_te - collateral_credit(min_Bt, Ste, held_cut_te)) / fx_base

            if len(factor_dep['Cash_Collateral']) == 1 and factor_dep['Cash_Collateral'][0].Collateral_Rate is not None:
                cash_col = factor_dep['Cash_Collateral'][0]
                cash_rep = utils.calc_time_grid_spot_rate(cash_col.Currency, report_time, shared) / fx_base
                mtm_grid = report_time[:, utils.TIME_GRID_MTM]

                # calculate collateral valuation adjustment
                delta_scen_t = np.append(np.diff(mtm_grid), 0).reshape(-1, 1)
                discount_funding = utils.calc_time_grid_curve_rate(cash_col.Funding_Rate, report_time, shared)
                discount_collateral = utils.calc_time_grid_curve_rate(cash_col.Collateral_Rate, report_time, shared)
                discount_collateral_t0 = utils.calc_time_grid_curve_rate(
                    cash_col.Collateral_Rate, np.zeros((1, 3)), shared)

                Dc_over_f_tT_m1 = torch.expm1(
                    torch.squeeze(discount_funding.gather_weighted_curve(shared, delta_scen_t) -
                                  discount_collateral.gather_weighted_curve(shared, delta_scen_t), dim=1)
                )

                Dc0_T = torch.exp(
                    -torch.squeeze(discount_collateral_t0.gather_weighted_curve(
                        shared, mtm_grid.reshape(1, -1)), dim=0))

                funding = expected_collateral * cash_rep * Dc_over_f_tT_m1 * Dc0_T
                shared.save_results(deal_data.Calc_res, {'Funding': funding})

            if surv:
                St_T = torch.squeeze(torch.exp(-surv.gather_weighted_curve(
                    shared, report_time[:, utils.TIME_GRID_MTM].reshape(1, -1), multiply_by_time=False)
                                               ), dim=0)
                net_accum = net_accum * St_T

            # each of the set's rows lands on its own date among the root's report rows
            on_report = shared.upload(
                np.searchsorted(time_grid.report_index, factor_dep['Report_Rows']), torch.int64)

            def reported(rows):
                return rows.new_zeros(time_grid.report_index.size, rows.shape[1]).index_copy(0, on_report, rows)

            if boundary_aad:
                # Balance-independent, detached captures for the MTA replay (see docstring).
                b_Vte, b_C, b_Ste, b_cut = Vte.detach(), C_ts_te.detach(), Ste.detach(), held_cut_te.detach()
                b_fx, b_surv = fx_base.detach(), St_T.detach() if surv else None

                def replay_net_mtm(balance_path, vte=None, c_delta=None, base_i=base_i,
                                   delta_T=delta_T):
                    """Balance path, and optionally a different Vte or settled-cash delta, to the
                    net mtm as reported. `vte` defaults to what was reported - the margin-call case
                    where only the balance varies; a pricer counterfactual moves the gross through
                    that one term, and one that flips a payment moves cash through `c_delta`."""
                    running = balance_path[base_i]
                    step = np.zeros_like(delta_T)
                    for i in range(delta_T.max() if delta_T.size else 0):
                        step[delta_T > i] = i + 1
                        running = torch.min(running, balance_path[base_i + step])
                    running = Fn.pad(running, [0, 0, 0, 1])
                    out = ((b_Vte if vte is None else vte) +
                           (b_C if c_delta is None else b_C + c_delta) -
                           collateral_credit(running, b_Ste, b_cut)) / b_fx
                    if b_surv is not None:
                        out = out * b_surv
                    return reported(out)

                def rescan(opening, start, req=Bt_new.detach(), recv=Mr.detach(),
                           post=Mp.detach(), mask=factor_dep['call_mask'].astype(bool)):
                    """The forward walk's own recursion, restarted at a margin date from a forced
                    opening balance - which is the whole of a transfer counterfactual."""
                    return scan_collateral_balance(opening, req, recv, post, mask, start=start)[0]

                shared.boundary_sets.append(utils.MTABoundarySet(
                    events=mta_events, replay=replay_net_mtm,
                    balance=boundary_balance, rescan=rescan))

                # Gross-independent, detached captures for the whole gross->net chain
                # (At, required balance, bands, scan) - see docstring.
                g_fx, g_fx_local = fx_base.detach(), fx_base_local.detach()
                g_St, g_cut, g_fxa = St.detach(), posted_cut.detach(), fx_agreement.detach()
                g_H, g_G, g_IA = H.detach(), G.detach(), (
                        factor_dep['Independent_Amount'] * fx_agreement).detach()
                g_Vt, g_B0, g_Te = Vt.detach(), Bt[0].detach(), factor_dep['Te']
                g_mask = factor_dep['call_mask'].astype(bool)
                # the C_ts_te window edges per the declared settlement-risk convention;
                # Exclude_Paid_Today shifts the cumsum from strict-before to inclusive
                g_ept = self.options['Exclude_Paid_Today']
                g_rec_lo = factor_dep['Tl'] if self.options[
                                                   'Cash_Settlement_Risk'] == utils.CASH_SETTLEMENT_Paid_Only else \
                factor_dep['Ts']
                g_pay_lo = factor_dep['Ts'] if self.options[
                                                   'Cash_Settlement_Risk'] == utils.CASH_SETTLEMENT_Received_Only else \
                factor_dep['Tl']

                def cash_to_C(t, branch, booked):
                    """A flipped payment at mtm row `t` as its C_ts_te delta (and Vte delta).

                    The ledger is relu-split into received/paid before cumulating, so the exact
                    delta needs the booked amount beside the branch - the split of a difference is
                    not the difference of the splits. A registration declares REPORTING-currency
                    amounts while `Cf_Rec`/`Cf_Pay` cumulate base ones, so the row's own cross
                    carries it across; the two coincide on a base-reporting book and nowhere else.
                    """
                    # a row past this set's own prefix is masked out of `rec`/`pay` below, so the
                    # clamp only keeps the gather in range
                    fx_t = g_fx_local[min(t, g_fx_local.shape[0] - 1)]
                    rec_d = fx_t * (torch.relu(branch) - torch.relu(booked))
                    pay_d = fx_t * (torch.relu(-branch) - torch.relu(-booked))
                    if g_ept:
                        rec, pay = (g_rec_lo < t) & (t <= g_Te), (g_pay_lo < t) & (t <= g_Te)
                        today = branch.new((g_Te == t).astype('f8')).reshape(-1, 1)
                        vte_d = -today * (rec_d - pay_d)
                    else:
                        rec, pay = (g_rec_lo <= t) & (t < g_Te), (g_pay_lo <= t) & (t < g_Te)
                        vte_d = None
                    c_d = (branch.new(rec.astype('f8')).reshape(-1, 1) * rec_d -
                           branch.new(pay.astype('f8')).reshape(-1, 1) * pay_d)
                    return c_d, vte_d

                def net_from_gross(delta, cash=None, base_i=base_i, delta_T=delta_T):
                    """Push a GROSS-mtm delta all the way to the net, collateral response included.
                    The delta arrives on the MTM grid; this set runs on its own local prefix.

                    Vte is built from the REPORTED b_Vte, never by re-indexing g_Vt: under
                    Exclude_Paid_Today the two carry DIFFERENT cashflow adjustments, so re-deriving
                    it rebases the counterfactual by ~100x the signal it carries, invisibly to a
                    value gate. From b_Vte, gross_to_net(0) == the reported net BY CONSTRUCTION."""
                    delta = delta[:g_Vt.shape[0]]
                    Vt_cf = g_Vt + delta * g_fx_local
                    At_cf = g_IA + (Vt_cf - g_H) * (Vt_cf > g_H) + (Vt_cf - g_G) * (Vt_cf < g_G)
                    balance, _ = scan_collateral_balance(g_B0, *collateral_bands(
                        At_cf, g_St, g_cut, g_fxa, min_received, min_posted), g_mask)
                    # Vte from the REPORTED b_Vte, never re-indexed off g_Vt (see docstring). g_fx
                    # is already the Te-grid fx, so only the delta takes the Te index here.
                    vte = b_Vte + delta[g_Te] * g_fx
                    c_d = None
                    if cash:
                        parts = [cash_to_C(*entry) for entry in cash]
                        c_d = sum(p[0] for p in parts)
                        for _, vte_d in parts:
                            if vte_d is not None:
                                vte = vte + vte_d
                    return replay_net_mtm(balance, vte=vte, c_delta=c_d)

                # published, not applied: `resolve_structure` hands it to the registrations made
                # beneath THIS netting set and drops it again
                shared.gross_to_net = net_from_gross

            gross_padding = time_grid.mtm_time_grid.size - local_accum.shape[0]
            shared.save_results(
                deal_data.Calc_res, {
                    'Collateral': reported(expected_collateral),
                    'GrossMTM': Fn.pad(local_accum, [0, 0, 0, gross_padding]) if gross_padding else local_accum,
                    'Value': reported(net_accum)
                })
            # the root sums every set row by row on the global grid and reports its own dates off it
            return net_accum.new_zeros(time_grid.mtm_time_grid.size, net_accum.shape[1]).index_copy(
                0, shared.upload(factor_dep['Report_Rows'], torch.int64), net_accum)
        else:
            St_T = torch.squeeze(torch.exp(-surv.gather_weighted_curve(
                shared, time_grid.time_grid[:, utils.TIME_GRID_MTM].reshape(1, -1), multiply_by_time=False)), dim=0
                                 ) if surv else shared.one

            return pricing.interpolate(accum, shared, time_grid, deal_data, interpolate_grid=False) * St_T


class MtMCrossCurrencySwapDeal(Deal):
    accepts_children = True
    vernacular = 'mark-to-market cross-currency swap, resetting cross-currency swap, MtM CCS'
    fields = [ADMIN, own('MtMCrossCurrencySwapDeal', [
        F('Pay_Discount_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Pay_Rate_Type', 'Text', default='Fixed', convention=True, values=['Fixed', 'Floating']),
        F('Receive_Interest_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Maturity_Date', 'Date', default=''),
        F('Principal_Exchange', 'Text', default='None', values=['None', 'Start', 'Maturity', 'Start_Maturity']),
        F('Pay_Interest_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Receive_Discount_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Receive_Rate_Type', 'Text', default='Floating', convention=True, values=['Fixed', 'Floating']),
        F('Effective_Date', 'Date', default=''),
        F('Pay_Currency', 'Text', default='', factors=['FxRate']),
        F('MtM_Side', 'Text', default='Pay', values=['Pay', 'Receive']),
        F('Receive_Currency', 'Text', default='', factors=['FxRate']),
        F('Pay_Interest_Rate_Volatility', 'Text', default='', convention=True,
          factors=['InterestRateVol', 'InterestYieldVol']),
        F('Pay_Discount_Rate_Volatility', 'Text', default='', convention=True,
          factors=['InterestRateVol', 'InterestYieldVol']),
        F('Receive_Interest_Rate_Volatility', 'Text', default='', convention=True,
          factors=['InterestRateVol', 'InterestYieldVol']),
        F('Receive_Discount_Rate_Volatility', 'Text', default='', convention=True,
          factors=['InterestRateVol', 'InterestYieldVol'])
    ])]

    documentation = ('Interest Rates',
                     ['This currency swap adjusts the notional of one leg to capture any changes in the FX Spot rate',
                      'since the last reset. At each reset, the principal of the adjusted leg is set to the principal',
                      'of the unadjusted leg multiplied by the spot FX rate. MtM cross currency swaps are path',
                      'dependent.',
                      '',
                      'The unadjusted leg is either a fixed or floating interest rate list and is valued as such,',
                      'however, the floating adjusted leg is valued as',
                      '',
                      '$$\\sum_{i=1}^n(P_i(t)(L_i(t)+m)\\alpha_i+A_i(t))D(t,t_i),$$',
                      '',
                      'where',
                      '',
                      '- $A_i(t)=P_i(t)-P_{i+1}(t)$ for $i\\lt n$ and $A_n(t)=P_n(t)$',
                      '- $P_i(t)$ is the expected principal $P_i(t)=F(t,t_{i-1})\\tilde P_i$,',
                      '- $\\tilde P_i$ is the unadjusted leg principal for the $i^{th}$ period.',
                      '- $F(t,T)$ is the forward FX rate for settlement at time $T$.'
                      ])

    def reset(self, calendars):
        super(MtMCrossCurrencySwapDeal, self).reset()
        self.paydates = utils.generate_dates_backward(
            self.field['Maturity_Date'], self.field['Effective_Date'],
            self.field.get('Pay_Frequency', pd.DateOffset(months=6)))
        self.recdates = utils.generate_dates_backward(
            self.field['Maturity_Date'], self.field['Effective_Date'],
            self.field.get('Receive_Frequency', pd.DateOffset(months=6)))
        self.add_reval_dates(self.paydates, self.field['Pay_Currency'])
        self.add_reval_dates(self.recdates, self.field['Receive_Currency'])
        self.child_map = {}

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        # have to reset the original instrument and let the child node decide
        super(MtMCrossCurrencySwapDeal, self).reset()
        for currency, dates in node_settlements.items():
            if 'Start' in self.field['Principal_Exchange'] and base_date <= self.field['Effective_Date']:
                dates.add(self.field['Effective_Date'])
            self.add_reval_dates(dates, currency)
        return super(MtMCrossCurrencySwapDeal, self).finalize_dates(
            parser, base_date, grid, node_children, node_resets, node_settlements)

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

        if not self.child_map:
            # which child carries the MtM leg is known only once the children exist, and the
            # principal exchange this inserts is a schedule edit - so each child's compile
            # CONTINUES here and its tensor half is rebuilt after (`reopen` / `bind`)
            for index, child in enumerate(child_dependencies):
                # make the child price to the same grid as the parent
                child.Time_dep.assign(deal_data.Time_dep)
                cashflows = child.Factor_dep['Cashflows'].reopen()
                if child.Factor_dep['Currency'] == factor_dep[factor_dep['MTM']]['Currency']:
                    daycount = self.field.get(factor_dep['MTM'] + '_Day_Count', 'ACT_365')
                    # add a zero nominal payment at the beginning if forward starting
                    cashflows.add_mtm_payments(
                        factor_dep['base_date'], self.field['Principal_Exchange'],
                        self.field['Effective_Date'], daycount)
                    cashflows.set_future_fx_resets(
                        deal_time[:, utils.TIME_GRID_MTM].max(), time_grid)
                    self.child_map.setdefault('MTM', child)
                else:
                    daycount = self.field.get(factor_dep['Other'] + '_Day_Count', 'ACT_365')
                    capital = cashflows.schedule[-1][utils.CASHFLOW_INDEX_Nominal]
                    cashflows.add_fixed_payments(
                        factor_dep['base_date'], self.field['Principal_Exchange'],
                        self.field['Effective_Date'], daycount, capital)
                    self.child_map.setdefault('Static', child)
                cashflows.bind(shared.one)

        FX_rep = utils.calc_time_grid_spot_rate(shared.Report_Currency, deal_time, shared)
        FX_static = utils.calc_time_grid_spot_rate(
            self.child_map['Static'].Factor_dep['Currency'][0], deal_time, shared)
        FX_mtm = utils.calc_time_grid_spot_rate(
            self.child_map['MTM'].Factor_dep['Currency'][0], deal_time, shared)

        if self.child_map['Static'].Factor_dep.get('Forward'):
            static_leg = pricing.pv_float_cashflow_list(
                shared, time_grid, self.child_map['Static'], pricing.pricer_float_cashflows)
        else:
            static_leg = pricing.pv_fixed_cashflows(shared, time_grid, self.child_map['Static'])

        mtm_leg = pricing.pv_float_cashflow_list(
            shared, time_grid, self.child_map['MTM'], pricing.pricer_float_cashflows,
            mtm_currency=self.child_map['Static'].Factor_dep['Currency'])

        mtm = (static_leg * FX_static + mtm_leg * FX_mtm) / FX_rep

        return pricing.interpolate(mtm, shared, time_grid, deal_data)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {
            'Pay_Currency': utils.check_rate_name(self.field['Pay_Currency']),
            'Receive_Currency': utils.check_rate_name(self.field['Receive_Currency'])
        }

        field['Pay_Discount_Rate'] = self.check_factor_name('Pay_Discount_Rate', field['Pay_Currency'])
        field['Pay_Interest_Rate'] = self.check_factor_name('Pay_Interest_Rate', field['Pay_Discount_Rate'])
        field['Receive_Discount_Rate'] = self.check_factor_name('Receive_Discount_Rate', field['Receive_Currency'])
        field['Receive_Interest_Rate'] = self.check_factor_name('Receive_Interest_Rate', field['Receive_Discount_Rate'])

        field_index = {'Pay': {}, 'Receive': {}}
        self.isQuanto = get_interest_rate_currency(
            field['Receive_Interest_Rate'], all_factors) != field['Receive_Currency']
        if self.isQuanto:
            # TODO - Deal with Quanto Interest Rate swaps
            pass
        else:
            field_index['Pay']['Currency'] = get_fx_and_zero_rate_factor(
                field['Pay_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Pay']['Forward'] = get_interest_factor(
                field['Pay_Interest_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Pay']['Discount'] = get_interest_factor(
                field['Pay_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Receive']['Currency'] = get_fx_and_zero_rate_factor(
                field['Receive_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['Receive']['Forward'] = get_interest_factor(
                field['Receive_Interest_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Receive']['Discount'] = get_interest_factor(
                field['Receive_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors)

        # TODO - complete cashflow definitions..

        field_index['MTM'] = self.field['MtM_Side']
        field_index['Other'] = {'Pay': 'Receive', 'Receive': 'Pay'}[self.field['MtM_Side']]
        field_index['base_date'] = base_date

        return field_index


class FXNonDeliverableForward(Deal):
    vernacular = 'non-deliverable forward, NDF'
    fields = [ADMIN, own('FXNonDeliverableForward', [
        F('Sell_Currency', 'Text', default='', factors=['FxRate']),
        F('Sell_Amount', 'Float', default=0.0, sized=True),
        F('Settlement_Date', 'Date', default='', settles=Cash('Settlement_Currency')),
        F('Settlement_Currency', 'Text', default='', factors=['FxRate']),
        F('Buy_Amount', 'Float', default=0.0, sized=True),
        F('Discount_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Buy_Currency', 'Text', default='', factors=['FxRate'])
    ])]

    documentation = ('Fx And Equity',
                     ['An FX non-deliverable forward effectively an FX Forward deal that is cash settled in a',
                      '(potentially) third currency. The deal pays',
                      '',
                      '$$A \\tilde X(t)-BX(t)$$',
                      '',
                      'in **settlement currency** at the **settlement date** $T$ where:',
                      '',
                      '- $A$ is the **Buy_Amount** of **Buy_Currency**',
                      '- $B$ is the **Sell_Amount** of **Sell_Currency**',
                      '- $\\tilde X(t)$ is the price of the buy currency in settlement currency',
                      '- $X$ is the price of the sell currency in settlement currency',
                      '',
                      'The value of the deal in settlement currency at time $t$, ($t \\le T$), is',
                      '',
                      '$$ \\Big(A \\tilde F(t,T)-BF(t,T)\\Big)D(t,T),$$',
                      '',
                      'where:',
                      '',
                      '- $\\tilde F(t,T)$ is the forward price of the buy currency in settlement currency',
                      '- $F(t,T)$ is the forward price of the sell currency in settlement currency',
                      '- $D$ is the discount factor on **Discount_Rate**, the settlement currency\'s where blank'
                      ])

    def reset(self, calendars):
        super(FXNonDeliverableForward, self).reset()
        self.add_reval_dates({self.field['Settlement_Date']}, self.field['Settlement_Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Buy_Currency': utils.check_rate_name(self.field['Buy_Currency']),
                 'Sell_Currency': utils.check_rate_name(self.field['Sell_Currency']),
                 'Settlement_Currency': utils.check_rate_name(self.field['Settlement_Currency'])}

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Settlement_Currency'])

        field_index = {
            'BuyFX': get_fx_and_zero_rate_factor(
                field['Buy_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'SellFX': get_fx_and_zero_rate_factor(
                field['Sell_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleFX': get_fx_and_zero_rate_factor(
                field['Settlement_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Maturity': (self.field['Settlement_Date'] - base_date).days,
            # needed for reporting
            'Local_Currency': '{0}.{1}'.format(self.field['Buy_Currency'], self.field['Sell_Currency'])
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)

        buy_forward = utils.calc_fx_forward(
            factor_dep['BuyFX'], factor_dep['SettleFX'], factor_dep['Maturity'], deal_time, shared)
        sell_forward = utils.calc_fx_forward(
            factor_dep['SellFX'], factor_dep['SettleFX'], factor_dep['Maturity'], deal_time, shared)
        FX_rep = utils.calc_fx_cross(
            factor_dep['SettleFX'][0], shared.Report_Currency, deal_time, shared)

        discount_rates = torch.squeeze(
            utils.calc_discount_rate(
                discount,
                (factor_dep['Maturity'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1), shared),
            dim=1)
        try:
            cash = (buy_forward * self.field['Buy_Amount'] - sell_forward * self.field['Sell_Amount'])
        except:
            cash = (buy_forward * float(self.field['Buy_Amount']) - sell_forward * float(self.field['Sell_Amount']))

        pricing.cash_settle(
            shared, self.field['Settlement_Currency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])

        return cash * discount_rates * FX_rep


class FXSwapDeal(Deal):
    vernacular = 'FX swap, forex swap, near and far legs'
    fields = [ADMIN, own('FXSwapDeal', [
        F('Near_Settlement_Date', 'Date', default='', settles=(
            Cash('Near_Buy_Far_Sell_Ccy', 'Near_Buy_Amount'),
            Cash('Near_Sell_Far_Buy_Ccy', 'Near_Sell_Amount', -1.0))),
        F('Far_Settlement_Date', 'Date', default='', settles=(
            Cash('Near_Sell_Far_Buy_Ccy', 'Far_Buy_Amount'),
            Cash('Near_Buy_Far_Sell_Ccy', 'Far_Sell_Amount', -1.0))),
        F('Near_Buy_Far_Sell_Ccy', 'Text', default='', factors=['FxRate']),
        F('Near_Sell_Far_Buy_Ccy', 'Text', default='', factors=['FxRate']),
        F('Near_Buy_Far_Sell_Discount_Rate', 'Text', default=REQUIRED, factors=['InterestRate']),
        F('Near_Sell_Far_Buy_Discount_Rate', 'Text', default=REQUIRED, factors=['InterestRate']),
        F('Near_Buy_Amount', 'Float', default=0.0, sized=True),
        F('Near_Sell_Amount', 'Float', default=0.0, sized=True),
        F('Far_Buy_Amount', 'Float', default=0.0, sized=True),
        F('Far_Sell_Amount', 'Float', default=0.0, sized=True)
    ])]

    documentation = ('Fx And Equity', [
        'An FX swap is a combination of an FX forward deal with near settlement date $t_1$ and',
        'an FX forward deal in the opposite direction with far settlement date $t_2$, where $t_1 < t_2$.',
        'The reporting-currency value of the FX swap at time $t$, $t \\le t_1$, is',
        '',
        '$$A_1 \\tilde D(t,t_1) \\tilde X(t)-B_1 D(t,t_1)X(t) + B_2 D(t,t_2) X(t)-A_2 \\tilde D(t,t_2) \\tilde X(t)$$',
        '',
        'where $A_1$ is **Near_Buy_Amount** and $A_2$ **Far_Sell_Amount** of **Near_Buy_Far_Sell_Ccy**',
        '(discount factor $\\tilde D$ on **Near_Buy_Far_Sell_Discount_Rate**, price $\\tilde X$ in the',
        'reporting currency), and $B_1$ is **Near_Sell_Amount** and $B_2$ **Far_Buy_Amount** of',
        '**Near_Sell_Far_Buy_Ccy** ($D$ and $X$ likewise). The near leg drops out after $t_1$, leaving the',
        'far leg. Typically $t_1$ is the spot settlement date and $A_2 = A_1$.'])

    def reset(self, calendars):
        super(FXSwapDeal, self).reset()
        self.add_reval_dates({self.field['Near_Settlement_Date'], self.field['Far_Settlement_Date']},
                             self.field['Near_Buy_Far_Sell_Ccy'])
        self.add_reval_dates({self.field['Near_Settlement_Date'], self.field['Far_Settlement_Date']},
                             self.field['Near_Sell_Far_Buy_Ccy'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        # both discount rates are required, for the reason FXForwardDeal states: discovery reads
        # the RAW field and skips a blank, so a fallback here would name a curve never loaded
        field = {'NearBuyFarSell_Currency': utils.check_rate_name(self.field['Near_Buy_Far_Sell_Ccy']),
                 'NearBuyFarSell_DiscountRate': utils.check_rate_name(self.field['Near_Buy_Far_Sell_Discount_Rate']),
                 'NearSellFarBuy_Currency': utils.check_rate_name(self.field['Near_Sell_Far_Buy_Ccy']),
                 'NearSellFarBuy_DiscountRate': utils.check_rate_name(self.field['Near_Sell_Far_Buy_Discount_Rate'])}

        field_index = {'NearBuyFX': get_fxrate_factor(field['NearBuyFarSell_Currency'], static_offsets,
                                                      stochastic_offsets),
                       'NearBuyDiscount': get_interest_factor(field['NearBuyFarSell_DiscountRate'], static_offsets,
                                                              stochastic_offsets, all_tenors),
                       'NearSellFX': get_fxrate_factor(field['NearSellFarBuy_Currency'], static_offsets,
                                                       stochastic_offsets),
                       'NearSellDiscount': get_interest_factor(field['NearSellFarBuy_DiscountRate'], static_offsets,
                                                               stochastic_offsets, all_tenors),
                       'Maturity': (self.field['Far_Settlement_Date'] - base_date).days,
                       'Near_Maturity': (self.field['Near_Settlement_Date'] - base_date).days}

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

        mtm = 0
        remaining_tenor = (factor_dep['Maturity'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1)

        near_buy_discount = utils.calc_time_grid_curve_rate(factor_dep['NearBuyDiscount'], deal_time, shared)
        near_sell_discount = utils.calc_time_grid_curve_rate(factor_dep['NearSellDiscount'], deal_time, shared)

        if factor_dep['Near_Maturity'] >= 0:
            near_deal_time = deal_time[:np.searchsorted(
                deal_time[:, utils.TIME_GRID_MTM], factor_dep['Near_Maturity'], side='right')]
            near = (factor_dep['Near_Maturity'] - near_deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1)

            NearBuy_rep = utils.calc_fx_cross(
                factor_dep['NearBuyFX'], shared.Report_Currency, near_deal_time, shared)
            NearSell_rep = utils.calc_fx_cross(
                factor_dep['NearSellFX'], shared.Report_Currency, near_deal_time, shared)

            # the near leg is valued only up to its own date, on the curves read there
            near_sell_discount_rate = torch.squeeze(utils.calc_discount_rate(
                utils.calc_time_grid_curve_rate(factor_dep['NearSellDiscount'], near_deal_time, shared),
                near, shared), dim=1)
            near_buy_discount_rate = torch.squeeze(utils.calc_discount_rate(
                utils.calc_time_grid_curve_rate(factor_dep['NearBuyDiscount'], near_deal_time, shared),
                near, shared), dim=1)

            mtm_near = self.field['Near_Buy_Amount'] * near_buy_discount_rate * NearBuy_rep - \
                       self.field['Near_Sell_Amount'] * near_sell_discount_rate * NearSell_rep

            mtm = Fn.pad(mtm_near, [0, 0, 0, remaining_tenor.size - near.size])

            pricing.cash_settle(
                shared, self.field['Near_Buy_Far_Sell_Ccy'],
                deal_data.Time_dep.fetch_index_by_day(factor_dep['Near_Maturity']), self.field['Near_Buy_Amount'])
            pricing.cash_settle(
                shared, self.field['Near_Sell_Far_Buy_Ccy'],
                deal_data.Time_dep.fetch_index_by_day(factor_dep['Near_Maturity']), -self.field['Near_Sell_Amount'])

        FX_NearBuy_rep = utils.calc_fx_cross(
            factor_dep['NearBuyFX'], shared.Report_Currency, deal_time, shared)
        FX_NearSell_rep = utils.calc_fx_cross(
            factor_dep['NearSellFX'], shared.Report_Currency, deal_time, shared)

        pricing.cash_settle(
            shared, self.field['Near_Sell_Far_Buy_Ccy'],
            deal_data.Time_dep.fetch_index_by_day(factor_dep['Maturity']), self.field['Far_Buy_Amount'])
        pricing.cash_settle(
            shared, self.field['Near_Buy_Far_Sell_Ccy'],
            deal_data.Time_dep.fetch_index_by_day(factor_dep['Maturity']), -self.field['Far_Sell_Amount'])

        far_buy_discount_rate = torch.squeeze(
            utils.calc_discount_rate(near_sell_discount, remaining_tenor, shared), dim=1)
        far_sell_discount_rate = torch.squeeze(
            utils.calc_discount_rate(near_buy_discount, remaining_tenor, shared), dim=1)

        mtm += self.field['Far_Buy_Amount'] * far_buy_discount_rate * FX_NearSell_rep - \
               self.field['Far_Sell_Amount'] * far_sell_discount_rate * FX_NearBuy_rep

        return mtm


class FXForwardDeal(Deal):
    vernacular = 'FX forward, outright forward, deliverable forward'
    quoted = 'Buy_Amount'
    fields = [ADMIN, own('FXForwardDeal', [
        F('Sell_Currency', 'Text', default='', factors=['FxRate']),
        F('Sell_Amount', 'Float', default=0.0, sized=True),
        F('Settlement_Date', 'Date', default='',
          settles=(Cash('Buy_Currency', 'Buy_Amount'), Cash('Sell_Currency', 'Sell_Amount', -1.0))),
        F('Buy_Amount', 'Float', default=0.0, sized=True),
        F('Sell_Discount_Rate', 'Text', default=REQUIRED, factors=['InterestRate']),
        F('Buy_Currency', 'Text', default='', factors=['FxRate']),
        F('Buy_Discount_Rate', 'Text', default=REQUIRED, factors=['InterestRate'])
    ])]

    documentation = ('Fx And Equity', [
        'An FX forward is an agreement to buy an amount $A$ (**Buy_Amount**) of one currency in exchange',
        'for an amount $B$ (**Sell_Amount**) of another at settlement date $T$ (**Settlement_Date**).',
        'The value of the deal in reporting currency at time $t$, ($t \\le T$), is',
        '',
        '$$A \\tilde D(t,T) \\tilde X(t)-BD(t,T)X(t)$$',
        '',
        'where',
        '',
        '- $D$ is the sell currency discount factor on **Sell_Discount_Rate**',
        '- $\\tilde D$ is the buy currency discount factor on **Buy_Discount_Rate**',
        '- $X$ is the price of the sell currency in reporting currency',
        '- $\\tilde X$ is the price of the buy currency in reporting currency'])

    @classmethod
    def quote(cls, deal, value):
        """The quote is the FORWARD OUTRIGHT - units of `Buy_Currency` per one unit of
        `Sell_Currency` - and the amount it buys is where that number lands: with `Sell_Amount` and
        both discount curves fixed the value is exactly affine in it, an exact quote-space
        residual. Not a percent, and nothing converts it: each type's field carries its own unit."""
        deal['Buy_Amount'] = value * deal['Sell_Amount']

    def reset(self, calendars):
        super(FXForwardDeal, self).reset()
        self.add_reval_dates({self.field['Settlement_Date']}, self.field['Buy_Currency'])
        self.add_reval_dates({self.field['Settlement_Date']}, self.field['Sell_Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        # both discount rates are required: discovery reads the RAW field and skips a blank, so a
        # fallback here would name a curve that was never loaded - and the sell leg's would name
        # the BUY currency, pricing it on the wrong curve
        field = {'Buy_Currency': utils.check_rate_name(self.field['Buy_Currency']),
                 'Buy_Discount_Rate': utils.check_rate_name(self.field['Buy_Discount_Rate']),
                 'Sell_Currency': utils.check_rate_name(self.field['Sell_Currency']),
                 'Sell_Discount_Rate': utils.check_rate_name(self.field['Sell_Discount_Rate'])}

        field_index = {
            'BuyFX': get_fxrate_factor(field['Buy_Currency'], static_offsets, stochastic_offsets),
            'BuyDiscount': get_interest_factor(
                field['Buy_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'SellFX': get_fxrate_factor(field['Sell_Currency'], static_offsets, stochastic_offsets),
            'SellDiscount': get_interest_factor(
                field['Sell_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Maturity': (self.field['Settlement_Date'] - base_date).days
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        # the tenors and the scenario grid should already be loaded up in the Buffer space in constant memory
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

        FX_Buy_rep = utils.calc_fx_cross(
            factor_dep['BuyFX'], shared.Report_Currency, deal_time, shared)
        FX_Sell_rep = utils.calc_fx_cross(
            factor_dep['SellFX'], shared.Report_Currency, deal_time, shared)

        buy_discount = utils.calc_time_grid_curve_rate(factor_dep['BuyDiscount'], deal_time, shared)
        sell_discount = utils.calc_time_grid_curve_rate(factor_dep['SellDiscount'], deal_time, shared)
        remaining_tenor = (factor_dep['Maturity'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1)

        buy_discount_rate = torch.squeeze(utils.calc_discount_rate(buy_discount, remaining_tenor, shared), dim=1)
        sell_discount_rate = torch.squeeze(utils.calc_discount_rate(sell_discount, remaining_tenor, shared), dim=1)

        pricing.cash_settle(
            shared, self.field['Buy_Currency'], deal_data.Time_dep.deal_time_grid[-1], self.field['Buy_Amount'])
        pricing.cash_settle(
            shared, self.field['Sell_Currency'], deal_data.Time_dep.deal_time_grid[-1], -self.field['Sell_Amount'])

        return self.field['Buy_Amount'] * FX_Buy_rep * buy_discount_rate - \
            self.field['Sell_Amount'] * FX_Sell_rep * sell_discount_rate


class StructuredDeal(Deal):
    accepts_children = True
    vernacular = 'structure, package, strategy'
    fields = [ADMIN, own('StructuredDeal', [
        F('Currency', 'Text', default='', factors=['FxRate']),
        F('Net_Cashflows', 'Text', default='No', convention=True, values=['Yes', 'No'])
    ])]

    documentation = ('Container', [
        'A structured deal consists of multiple individual deals. In credit exposure calculations',
        'it is treated as a single deal, the values of its children netted together - an interest rate',
        'swap, for instance, booked as its two legs. Its value is the plain sum of the children\'s',
        'values, each already in the reporting currency:',
        '',
        '$$V(t)=\\sum_i V_i(t)$$',
        '',
        'There is no direction: a child carries its own **Buy_Sell**. **Currency** names an FX factor',
        'the deal depends on and does not convert the sum. **Net_Cashflows** is accepted and has no',
        'effect.'])

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        if node_children is not None:
            for child in node_children:
                child.add_reval_date_offset(1)
                node_resets.update(child.get_reval_dates())

        self.reval_dates = node_resets
        for currency, dates in node_settlements.items():
            self.add_reval_dates(dates, currency)

        return super(StructuredDeal, self).finalize_dates(
            parser, base_date, grid, node_children, node_resets, node_settlements)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {}
        field['Currency'] = utils.check_rate_name(self.field['Currency'])

        field_index = {}
        field_index['Currency'] = get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets)

        return field_index

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        # accum should be zero for an empty container like a netting set or another structured deal
        # but can be nonzero if this structured deal contains other structured deals
        net_mtm = accum

        for child in child_dependencies:
            net_mtm = net_mtm + child.Instrument.calculate(shared, time_grid, child)

        # TODO - net off cashflows across the children
        if deal_data.Instrument.field['Net_Cashflows'] == 'Yes':
            pass

        return pricing.interpolate(net_mtm, shared, time_grid, deal_data, interpolate_grid=False)

    def generate(self, shared, time_grid, deal_data):
        return shared.one.new_zeros(1, 1)


class StructuredDealBreakClause(StructuredDeal):
    documentation = ('Container', [
        'Priced exactly as `StructuredDeal`, the sum of its children; the break itself is not',
        'modelled.'])

    def __init__(self, params, valuation_options):
        super(StructuredDealBreakClause, self).__init__(params, valuation_options)


class DepositDeal(Deal):
    """A money-market deposit. **Amount** is placed at **Effective_Date** and accrues over
    **Payment_Frequency** periods on **Accrual_Day_Count**, each on the balance then outstanding:
    an **Amortisation** step is repaid at the end of the period its date falls in, the day the
    balance falls, and the balance left is repaid at **Maturity_Date**.

    The rate comes from **Interest_Rate_Schedule** (a date->rate `DateList` in percent) when that
    covers every period's unadjusted start, and is forecast off the **Interest_Rate** curve otherwise. A
    fully-pinned schedule prices as a pure fixed leg and pulls NO forecast factor, which is what
    lets a curve bootstrapper price a depo quote without depending on the curve it is solving for.
    Coverage is all-or-nothing rather than per-period, because the float path's known-rate override
    pins only resets ON OR BEFORE the reference date - the right reading for a seasoned floating
    deposit, a reset behind it with no row fixed at 0% with a warning.

    The amount lent is an outflow at the start that drops out once the deposit has started, and
    comes back as the steps and the final balance. Flows discount on **Discount_Rate**, defaulting
    to the currency's own curve.
    """
    vernacular = 'deposit, money-market deposit, placement'
    observes = Observes('Interest_Rate', 'InterestRate')
    quoted = 'Interest_Rate_Schedule'
    fields = [ADMIN, PRICING, COUPON_SCHEDULE, own('DepositDeal', [
        F('Effective_Date', 'Date', default=''),
        F('Maturity_Date', 'Date', default=''),
        F('Payment_Frequency', 'Period', default='3M', convention=True),
        F('Interest_Frequency', 'Period', default='3M', convention=True),
        F('Compounding', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Compounding_Method', 'Text', default='None', convention=True,
          values=['None', 'OIS', 'Include_Margin', 'Flat', 'Exclude_Margin']),
        F('FX_Reset_Offset', 'Integer', default=0, convention=True),
        F('Known_FX_Rates', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float')])),
        F('Amount', 'Float', default=0.0, sized=True),
        F('Interest_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Interest_Rate_Schedule', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float')]))
    ])]

    documentation = ('Interest Rates', [
        'A deposit pays interest on its balance, repays each **Amortisation** step at the end of',
        'the period it falls in and the balance left at maturity:',
        '',
        '$$V(t)=\\sum_i \\delta_i N_i r_i D(t,T_i) + \\sum_i (N_i - N_{i+1}) D(t,T_i)$$',
        '',
        'where $N_i$ is the balance accruing over period $i$, $N_{n+1}=0$, $r_i$ is the accrual',
        'rate for period $i$ - read from **Interest_Rate_Schedule** when that pins every period, a row',
        'dated on the unadjusted start of the period it pins, otherwise forecast from **Interest_Rate**',
        '- and $D(t,T)$ is the discount factor from **Discount_Rate**. The periods are the deposit\'s',
        '[coupon schedule](#coupon-schedules).',
    ])

    @classmethod
    def quote(cls, deal, value):
        """A deposit has no rate field of its own: the quote is pinned at every period's unadjusted
        start, which prices it as a fixed leg and keeps it off the forecast curve a solve builds -
        `reset` drops that dependency when the schedule covers every start."""
        starts = utils.coupon_dates(deal['Maturity_Date'], deal['Effective_Date'], deal['Payment_Frequency'],
                                    first=deal['First_Coupon_Date'] or None,
                                    penultimate=deal['Penultimate_Coupon_Date'] or None)[:-1]
        deal['Interest_Rate_Schedule'] = utils.DateList({date: value for date in starts})

    def reset(self, calendars):
        super(DepositDeal, self).reset()
        terms = self.field['Effective_Date'], self.field['Maturity_Date'], self.field['Payment_Frequency']
        self.paydates, self.pays = self.coupon_schedule(calendars, *terms)
        self.add_reval_dates(self.paydates.union(self.pays), self.field['Currency'])
        # a pinned rate is stated on its period's unadjusted start, the day `quote` can name
        self.starts = self.coupon_schedule({}, *terms)[0][:-1]
        schedule = self.field['Interest_Rate_Schedule']
        # full coverage means no forecast curve is read
        self.is_fixed = bool(schedule and set(self.starts).issubset(schedule.data))
        if self.is_fixed:
            # discovery reads factor_fields off the INSTANCE and reset() runs first, so dropping the
            # forecast reference here keeps a pinned depo off the curve a bootstrapper is solving for
            self.factor_fields = {k: v for k, v in DepositDeal.factor_fields.items()
                                  if k != 'Interest_Rate'}

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'SettleCurrency': self.field['Currency'],
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors)}

        daycount = utils.DayCount.code(self.field['Accrual_Day_Count'])
        schedule = self.field['Interest_Rate_Schedule']

        if self.is_fixed:
            cashflows = utils.TensorCashFlows.generate_fixed(
                base_date, self.paydates, self.field['Amount'], self.field['Amortisation'], daycount, 0.0,
                self.pays)
            for row, start in zip(cashflows.schedule, self.starts):
                row[utils.CASHFLOW_INDEX_FixedRate] = schedule.data[start] / 100.0
        else:
            field['Interest_Rate'] = self.check_factor_name('Interest_Rate', field['Discount_Rate'])
            field_index['Forward'] = get_interest_factor(
                field['Interest_Rate'], static_offsets, stochastic_offsets, all_tenors)
            cashflows = utils.TensorCashFlows.generate_float(
                base_date, time_grid, self.paydates, self.field['Amount'], self.field['Amortisation'],
                schedule, self.field['Interest_Frequency'], self.field['Payment_Frequency'], daycount, 0.0,
                self.field.get('Reference'), self.pays)
            field_index['Model'] = pricing.pricer_float_cashflows

        # each period repays the step its successor's balance is smaller by, the last period the
        # balance then outstanding; `add_fixed_payments` places the amount lent
        nominal = cashflows.schedule[:, utils.CASHFLOW_INDEX_Nominal]
        cashflows.schedule[:, utils.CASHFLOW_INDEX_FixedAmt] = nominal - np.append(nominal[1:], 0.0)
        cashflows.add_fixed_payments(
            base_date, 'Start', self.field['Effective_Date'],
            self.field['Accrual_Day_Count'], self.field['Amount'])

        field_index['Cashflows'] = cashflows
        field_index['Compounding'] = self.field['Compounding'] == 'Yes'
        field_index['CompoundingMethod'] = self.field['Compounding_Method']

        return field_index

    def generate(self, shared, time_grid, deal_data):
        pv = pricing.pv_fixed_leg if self.is_fixed else pricing.pv_float_leg
        return pv(shared, time_grid, deal_data)


class SwapInterestDeal(Deal):
    accepts_children = True
    vernacular = 'interest rate swap, IRS, vanilla swap, basis swap, payer swap, receiver swap'
    observes = Observes('Interest_Rate', 'InterestRate')
    quoted = 'Swap_Rate'
    fields = [ADMIN, PRICING, FLOATING_INDEX, own('SwapInterestDeal', [
        F('Index_Frequency', 'Period', default='0M', convention=True),
        F('Rate_Multiplier', 'Float', default=1.0, convention=True),
        F('Known_Rates', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float')])),
        F('Maturity_Date', 'Date', default=''),
        F('Interest_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Interest_Rate_Volatility', 'Text', default='', convention=True,
          factors=['InterestRateVol', 'InterestYieldVol']),
        F('Effective_Date', 'Date', default=''),
        F('Floating_Margin', 'Float', default=0.0, convention=True),
        F('Fixed_Compounding', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Rate_Constant', 'Percent', default=0.0, convention=True),
        F('Compounding_Method', 'Text', default='None', convention=True,
          values=['None', 'OIS', 'Include_Margin', 'Flat', 'Exclude_Margin']),
        F('Amortisation', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float', sized='magnitude')])),
        F('Discount_Rate_Volatility', 'Text', default='', convention=True,
          factors=['InterestRateVol', 'InterestYieldVol']),
        F('Swap_Rate', 'Float', default=0.0),
        F('Principal', 'Float', default=0.0, sized=True)
    ]), own('SwapInterestDeal', [
        F('Pay_Rate_Type', 'Text', default='Fixed', values=['Fixed', 'Floating']),
        F('Pay_First_Coupon_Date', 'Date', default='', convention=True),
        F('Pay_Timing', 'Text', default='End', convention=True, values=['End', 'Begin', 'Discounted']),
        F('Pay_Payment_Offset', 'Integer', default=0, convention=True),
        F('Pay_Interest_Frequency', 'Period', default='3M', convention=True),
        F('Pay_Penultimate_Coupon_Date', 'Date', default='', convention=True),
        F('Pay_Accrual_Calendars', 'Text', default='', convention=True),
        F('Pay_Frequency', 'Period', default='3M', convention=True),
        F('Pay_Payment_Calendars', 'Text', default='', convention=True),
        F('Pay_Day_Count', 'Text', default='ACT_365', convention=True, values=DAY_COUNTS)
    ], 'Pay', 'Pay leg'), own('SwapInterestDeal', [
        F('Receive_Payment_Calendars', 'Text', default='', convention=True),
        F('Receive_Day_Count', 'Text', default='ACT_365', convention=True, values=DAY_COUNTS),
        F('Receive_Accrual_Calendars', 'Text', default='', convention=True),
        F('Receive_First_Coupon_Date', 'Date', default='', convention=True),
        F('Receive_Penultimate_Coupon_Date', 'Date', default='', convention=True),
        F('Receive_Interest_Frequency', 'Period', default='3M', convention=True),
        F('Receive_Timing', 'Text', default='End', convention=True, values=['End', 'Begin', 'Discounted']),
        F('Receive_Frequency', 'Period', default='3M', convention=True),
        F('Receive_Payment_Offset', 'Integer', default=0, convention=True)
    ], 'Receive', 'Receive leg')]

    documentation = ('Interest Rates', [
        'A fixed-float interest rate swap built from its own fields: a fixed leg at **Swap_Rate**',
        '(percent) and a floating leg forecast off **Interest_Rate** plus **Floating_Margin** (basis',
        'points), both on **Principal** less any **Amortisation**, each valued as a cashflow list',
        '([fixed](#fixed-interest-cashflows), [floating](#floating-interest-cashflows)) and discounted',
        'on **Discount_Rate**. **Pay_Rate_Type** `Fixed` pays the fixed leg on the `Pay_` schedule and',
        'receives the floating leg on the `Receive_` schedule; `Floating` the reverse. Each is the',
        '[coupon schedule](#coupon-schedules) its leg\'s conventions spell.',
        '',
        'An **Interest_Rate** curve in a currency other than **Currency** (a quanto swap) is not built:',
        'the deal is skipped. Booked over `Children`, its legs, it is their sum, revalued on their',
        'days, and builds no legs of its own.'])

    def reset(self, calendars):
        super(SwapInterestDeal, self).reset()
        self.legs = {leg: self.coupon_schedule(
            calendars, self.field['Effective_Date'], self.field['Maturity_Date'],
            self.field[leg + 'Frequency'], leg, timing='Timing') for leg in ('Pay_', 'Receive_')}
        for dates, pays in self.legs.values():
            self.add_reval_dates(dates.union(pays), self.field['Currency'])
        # this swap could be quantoed - TODO
        self.isQuanto = None

    finalize_dates = Deal.children_dates
    post_process = Deal.sum_children

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['Interest_Rate'] = self.check_factor_name('Interest_Rate', field['Discount_Rate'])

        field_index = {'SettleCurrency': self.field['Currency']}
        self.isQuanto = get_interest_rate_currency(field['Interest_Rate'], all_factors) != field['Currency']
        if self.isQuanto:
            # TODO - Deal with Quanto Interest Rate swaps
            pass
        else:
            field_index['Forward'] = get_interest_factor(
                field['Interest_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Discount'] = get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Currency'] = get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)

        # on its legs it prices them (`post_process`): legs of its own would go unread
        if self.has_children:
            return field_index

        fixed, floating, sign = ('Pay_', 'Receive_', -1) if self.field['Pay_Rate_Type'] == 'Fixed' else (
            'Receive_', 'Pay_', 1)
        dates, pays = self.legs[fixed]
        field_index['FixedCashflows'] = utils.TensorCashFlows.generate_fixed(
            base_date, dates, sign * self.field['Principal'], self.field['Amortisation'],
            utils.DayCount.code(self.field[fixed + 'Day_Count']), self.field['Swap_Rate'] / 100.0, pays)
        dates, pays = self.legs[floating]
        field_index['FloatCashflows'] = utils.TensorCashFlows.generate_float(
            base_date, time_grid, dates, -sign * self.field['Principal'], self.field['Amortisation'],
            self.field['Known_Rates'], self.field[floating + 'Interest_Frequency'], self.field['Index_Tenor'],
            utils.DayCount.code(self.field[floating + 'Day_Count']), self.field['Floating_Margin'] / 10000.0,
            self.field.get('Reference'), pays, self.float_index(calendars))

        field_index['Compounding'] = self.field['Fixed_Compounding'] == 'Yes'
        field_index['CompoundingMethod'] = self.field['Compounding_Method']
        # read where a coupon owns several resets, which its Compounding_Method folds: None
        # averages them at generate_float's 1/n and the others compound them
        field_index['AveragingMethod'] = 'Average_Rate'

        return field_index

    def generate(self, shared, time_grid, deal_data):
        fixed = deal_data.Factor_dep.copy()
        fixed['Cashflows'] = fixed['FixedCashflows']
        fixed_leg = pricing.pv_fixed_leg(shared, time_grid, utils.DealDataType(
            Instrument=deal_data.Instrument, Factor_dep=fixed,
            Time_dep=deal_data.Time_dep, Calc_res=deal_data.Calc_res))

        float = deal_data.Factor_dep.copy()
        float['Cashflows'] = float['FloatCashflows']
        float['Model'] = pricing.pricer_float_cashflows
        float_leg = pricing.pv_float_leg(shared, time_grid, utils.DealDataType(
            Instrument=deal_data.Instrument, Factor_dep=float,
            Time_dep=deal_data.Time_dep, Calc_res=deal_data.Calc_res))

        return fixed_leg + float_leg


class CFFixedInterestListDeal(Deal):
    vernacular = 'fixed-rate bond, fixed leg, fixed coupons'
    quoted = 'Cashflows'
    fields = [ADMIN, PRICING, CASHFLOWLISTDEAL, own('CFFixedInterestListDeal', [
        F('Cashflows', 'Container',
          sub_fields=[
              F('Compounding', 'Text', default='No', convention=True, values=['Yes', 'No']),
              F('Items', 'Table', default='null', row=Row([
                  F('Payment_Date', 'Date'), F('Notional', 'Float', sized=True), F('Rate', 'Percent'),
                  F('Accrual_Start_Date', 'Date'), F('Accrual_End_Date', 'Date'),
                  F('Accrual_Day_Count', 'Text', values=DAY_COUNTS), F('Accrual_Year_Fraction', 'Float'),
                  F('Fixed_Amount', 'Float', sized=True), F('Discounted', 'Text', values=['Yes', 'No']),
                  F('FX_Reset_Date', 'Date'), F('Known_FX_Rate', 'Float')]))]),
        F('Settlement_Amount', 'Float', default=0.0, sized=True),
        F('Calendars', 'Text', default='', convention=True)
    ])]

    documentation = (
        'Interest Rates', ['A series of fixed interest cashflows as described [here](#fixed-interest-cashflows)'])

    @classmethod
    def quote(cls, deal, value):
        """The fixed leg of a two-leg benchmark carries the quote on every row of its schedule."""
        for item in deal['Cashflows']['Items']:
            item['Rate'] = utils.Percent(value)

    def reset(self, calendars):
        super(CFFixedInterestListDeal, self).reset()

        paydates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        if self.field.get('Settlement_Date'):
            resetdates = {x for x in paydates if x < self.field['Settlement_Date']}
            resetdates.add(self.field['Settlement_Date'])
        else:
            resetdates = paydates

        self.add_reval_dates(resetdates, self.field['Currency'])

    def validate(self):
        """A settlement amount is paid ON a date.

        calc_dependencies leaves Settlement_Date as None when it is not supplied, and
        pv_fixed_cashflows reaches `factor_dep['Settlement_Date'] - time_block` under
        `if settlement_amt:` - so a non-zero amount without a date meets None.
        """
        if self.field.get('Settlement_Amount') and not self.field.get('Settlement_Date'):
            yield 'Settlement_Amount is set, so Settlement_Date is required'

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['Repo_Rate'] = self.check_factor_name('Repo_Rate', field['Discount_Rate'])
        field['Settlement_Rate'] = self.check_factor_name('Settlement_Rate', field['Discount_Rate'])

        field_index = {}
        buy_sell = 1 if self.field['Buy_Sell'] == 'Buy' else -1
        field_index['SettleCurrency'] = self.field['Currency']
        field_index['Currency'] = get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
        field_index['Discount'] = get_interest_factor(
            field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors)
        field_index['Repo_Rate'] = get_interest_factor(
            field['Repo_Rate'], static_offsets, stochastic_offsets, all_tenors)
        field_index['Settlement_Rate'] = get_interest_factor(
            field['Settlement_Rate'], static_offsets, stochastic_offsets, all_tenors)
        field_index['Cashflows'] = utils.TensorCashFlows.fixed(
            base_date, buy_sell, self.field['Cashflows'], self.field.get('Settlement_Date'))

        field_index['Compounding'] = self.field['Cashflows']['Compounding'] == 'Yes'
        field_index['Settlement_Date'] = (self.field.get('Settlement_Date') -
                                          base_date).days if self.field.get('Settlement_Date') else None
        field_index['Settlement_Amount'] = self.field.get('Settlement_Amount', 0.0) * buy_sell

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_fixed_leg(shared, time_grid, deal_data)


class CFFixedListDeal(Deal):
    vernacular = 'fixed cashflows, cashflow schedule, known payments'
    fields = [ADMIN, PRICING, own('CFFixedListDeal', [
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Description', 'Text', default='', convention=True),
        F('Cashflows', 'Container',
          sub_fields=[
              F('Items', 'Table', default='null',
                row=Row([F('Payment_Date', 'Date'), F('Fixed_Amount', 'Float', sized=True)]))])
    ])]

    documentation = (
        'Interest Rates',
        ['The value of a fixed cashflow list deal at valuation date t is',
         '',
         '$$\\delta \\sum_i C_i D(t, T_i)[t \\leq T_i], $$',
         '',
         '$C_i$ is the fixed amount and $T_i$ is the payment date of the $i^{th}$ cashflow in the list and ',
         '$\\delta$ is 1 if the deal is a **Buy** else -1 if the deal is a **Sell**']
    )

    def reset(self, calendars):
        super(CFFixedListDeal, self).reset()
        self.paydates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        self.add_reval_dates(self.paydates, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'SettleCurrency': self.field['Currency'], 'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Cashflows': utils.TensorCashFlows.simple_fixed(
                base_date, 1 if self.field['Buy_Sell'] == 'Buy' else -1, self.field['Cashflows'],
                lambda cashflow: cashflow['Fixed_Amount'], aggregate=True)
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_fixed_leg(shared, time_grid, deal_data)


class FixedCashflowDeal(Deal):
    vernacular = 'fixed cashflow, single payment, fee'
    fields = [ADMIN, PRICING, own('FixedCashflowDeal', [
        F('Calendars', 'Text', default='', convention=True),
        F('Amount', 'Float', default=0.0, sized=True),
        F('Payment_Date', 'Date', default='', settles=Cash('Currency', 'Amount'))
    ])]

    documentation = (
        'Interest Rates', ['The time $t$ value of a fixed cashflow amount $C$ paid at time $T$ is $D(t,T)C$.'])

    def reset(self, calendars):
        super(FixedCashflowDeal, self).reset()
        self.add_reval_dates({self.field['Payment_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Amount': (1 if self.field.get('Buy_Sell', 'Buy') == 'Buy' else -1) * self.field['Amount'],
            'Payment_Date': (self.field['Payment_Date'] - base_date).days,
            'Local_Currency': self.field['Currency']
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
        fx_rep = utils.calc_fx_cross(
            factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        discount_rates = torch.squeeze(
            utils.calc_discount_rate(
                discount, (factor_dep['Payment_Date'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1),
                shared),
            dim=1)

        mtm = factor_dep['Amount'] * discount_rates * fx_rep

        pricing.cash_settle(
            shared, self.field['Currency'], deal_data.Time_dep.deal_time_grid[-1], factor_dep['Amount'])

        return mtm


class CFFloatingInterestListDeal(Deal):
    vernacular = 'floating-rate note, floating leg, FRN'
    observes = Observes('Forecast_Rate', 'InterestRate')
    fields = [ADMIN, PRICING, CASHFLOWLISTDEAL, own('CFFloatingInterestListDeal', [
        F('Discount_Rate_Swaption_Volatility', 'Text', default='', convention=True,
          factors=['InterestYieldVol']),
        F('Rate_Adjustment_Method', 'Text', default='None', convention=True,
          values=['None', 'Modified_Following', 'Following', 'Preceding', 'Modified_Preceding']),
        F('Forecast_Rate_Swaption_Volatility', 'Text', default='', convention=True,
          factors=['InterestYieldVol']),
        F('Settlement_Amount', 'Float', default=0.0),
        F('Cashflows', 'Container',
          sub_fields=[
              F('Properties', 'Table', default='null', row=Row([
                  F('Digital_Payoff_Rate', 'Percent'), F('Cap_Multiplier', 'Float'), F('Cap_Strike', 'Percent'),
                  F('Floor_Multiplier', 'Float'), F('Floor_Strike', 'Percent')])),
              F('Compounding_Method', 'Text', default='None', convention=True,
                values=['None', 'OIS', 'Include_Margin', 'Flat', 'Exclude_Margin']),
              F('Averaging_Method', 'Text', default='Average_Rate', convention=True,
                values=['Average_Interest', 'Average_Rate', 'Pre_Aggregation', 'Post_Aggregation']),
              F('Items', 'Table', default='null', row=Row([
                  F('Payment_Date', 'Date'), F('Notional', 'Float', sized=True), F('Accrual_Start_Date', 'Date'),
                  F('Accrual_End_Date', 'Date'), F('Accrual_Day_Count', 'Text', values=DAY_COUNTS),
                  F('Accrual_Year_Fraction', 'Float'), F('Resets', 'Table'), F('Margin', 'Basis'),
                  F('Fixed_Amount', 'Float', sized=True), F('FX_Reset_Date', 'Date'), F('Known_FX_Rate', 'Float')]))]),
        F('Forecast_Rate_Cap_Volatility', 'Text', default='', convention=True,
          factors=['InterestRateVol']),
        F('Discount_Rate_Cap_Volatility', 'Text', default='', convention=True,
          factors=['InterestRateVol']),
        F('Rate_Calendars', 'Text', default='', convention=True),
        F('Forecast_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Rate_Sticky_Month_End', 'Text', default='Yes', convention=True, values=['Yes', 'No']),
        F('Accrual_Calendars', 'Text', default='', convention=True),
        F('Rate_Offset', 'Integer', default=0, convention=True)
    ])]

    documentation = (
        'Interest Rates', [
            'A series of floating interest cashflows as described [here](#floating-interest-cashflows)',
            '',
            'A cashflow Properties row with a non-zero **Cap_Multiplier** or **Floor_Multiplier**',
            'prices the leg as caplets or floorlets; **Digital_Payoff_Rate** makes each one a',
            'digital paying that rate. If the **Digital_Spread** Valuation Configuration option is',
            'set (> 0), the digital is priced as a call/put spread of ABSOLUTE half-width',
            '`Digital_Spread` in rate either side of the strike, each leg reading the cap surface',
            'at its own strike, so the smile is picked up automatically.',
            '',
            'On an aggregated leg (several resets per cashflow), **Compounding_Method** folds the',
            'resets to the period rate - `None` averages them and the others compound them, the',
            'margin placed as each places it across cashflows, `Flat` refusing a payment date that',
            'gathers several cashflows, one of several resets - and **Averaging_Method** picks the',
            'cap convention: `Pre_Aggregation` prices every reset as its own optionlet at its own',
            'expiry and accrual before summing back to the period (the daily capped RFR shape, and',
            'with a digital payoff a range accrual), placing no margin and so reading `None` or',
            '`OIS` alone, while `Post_Aggregation` folds first and prices one',
            'option on the period $[T_s,T_e]$ rate, vol read at the period end $T_e$ and the Black',
            'time the averaging decay of a backward-looking rate (Lyashenko-Mercurio [@lyashenko2019]):',
            '',
            '$$T_{\\text{eff}}=\\begin{cases}(T_s-t)+\\tfrac13(T_e-T_s), & t<T_s\\\\'
            '\\dfrac{(T_e-t)^3}{3(T_e-T_s)^2}, & T_s\\le t<T_e\\end{cases}$$',
            '',
            'Any other value folds and prices at the reset-start expiry.'])

    option_defaults = {'Digital_Spread': 0.0}

    def reset(self, calendars):
        super(CFFloatingInterestListDeal, self).reset()
        reset_dates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        self.add_reval_dates(reset_dates, self.field['Currency'])

    def validate(self):
        """A floating list settles no forward: its pricer reads no settlement field or curve."""
        for key in ('Settlement_Date', 'Settlement_Amount', 'Repo_Rate', 'Settlement_Rate'):
            if self.field.get(key):
                yield '{} is stated, and a floating cashflow list settles no forward'.format(key)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['Forecast_Rate'] = self.check_factor_name('Forecast_Rate', field['Discount_Rate'])

        field_index = {
            'Digital_Spread': self.options['Digital_Spread'],
            'SettleCurrency': self.field['Currency'],
            'Forward': get_interest_factor(
                field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'VolSurface': np.zeros(1, dtype=np.int32),
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
        }

        float_cashflows = utils.TensorCashFlows.float(
            base_date, time_grid, 1 if self.field['Buy_Sell'] == 'Buy' else -1,
            self.field['Cashflows'], self.field.get('Reference'))

        field_index['CompoundingMethod'] = self.field['Cashflows']['Compounding_Method']
        field_index['AveragingMethod'] = self.field['Cashflows']['Averaging_Method']

        if field_index['CompoundingMethod'] is None:
            field_index['CompoundingMethod'] = 'None'

        # potentially compress the cashflow list for faster computation
        if field_index['CompoundingMethod'] == 'None' and self.options.get('OIS_Cashflow_Group_Size', 0) > 0:
            field_index['Cashflows'] = float_cashflows.compress_no_compounding(
                self.options['OIS_Cashflow_Group_Size'])
        elif field_index['CompoundingMethod'] == 'OIS':
            # (groupsize -1 means group resets per cashflow - not 1 cashflow 1 reset)
            field_index['Cashflows'] = float_cashflows.compress_no_compounding(
                groupsize=-1, check_resets=False)
        else:
            field_index['Cashflows'] = float_cashflows

        field_index['Model'] = pricing.pricer_float_cashflows
        if self.field['Cashflows'].get('Properties'):
            first_prop = self.field['Cashflows']['Properties'][0]
            # cap vs floor is decided on a NON-ZERO multiplier, the test the branch is entered
            # on: an authored floor carries Cap_Multiplier 0.0, which is not a cap
            is_cap = bool(first_prop.get('Cap_Multiplier', 0.0))
            if is_cap or first_prop.get('Floor_Multiplier', 0.0):
                if first_prop.get('Digital_Payoff_Rate') is not None:
                    field_index['Digital_Payoff_Rate'] = first_prop['Digital_Payoff_Rate']
                field_index['Model'] = pricing.pricer_cap if is_cap else pricing.pricer_floor
                field_index['VolSurface'] = get_interest_vol_factor(
                    utils.check_rate_name(self.field['Forecast_Rate_Cap_Volatility']), pd.DateOffset(months=3),
                    static_offsets, stochastic_offsets, all_tenors)

                field_index['Cashflows'].overwrite_rate(
                    utils.CASHFLOW_INDEX_Strike,
                    float(first_prop['Cap_Strike'] if is_cap else first_prop['Floor_Strike']))

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_float_leg(shared, time_grid, deal_data)


class YieldInflationCashflowListDeal(Deal):
    vernacular = 'inflation-linked bond, linker, inflation cashflows'
    fields = [ADMIN, PRICING, CASHFLOWLISTDEAL, own('YieldInflationCashflowListDeal', [
        F('Index', 'Text', default='', factors=['InflationRate']),
        F('Index_Reference', 'Container',
          sub_fields=[
              F('Months_Lag', 'Integer', default=1),
              F('Quarters_Lag', 'Integer', default=0),
              F('Quarter_Reference_Month', 'Integer', default=1),
              F('Reference_Type', 'Text', default='Interpolated', values=['Single', 'Interpolated', 'Average']),
              F('Reference_Day', 'Integer', default=1),
              F('Days_In_Period', 'Integer', default=0)]),
        F('Cashflows', 'Container',
          sub_fields=[
              F('Items', 'Table', default='null', row=Row([
                  F('Payment_Date', 'Date'), F('Notional', 'Float', sized=True), F('Base_Reference_Date', 'Date'),
                  F('Base_Reference_Value', 'Float'), F('Final_Reference_Date', 'Date'),
                  F('Final_Reference_Value', 'Float'), F('Accrual_Start_Date', 'Date'), F('Accrual_End_Date', 'Date'),
                  F('Accrual_Day_Count', 'Text', values=DAY_COUNTS), F('Accrual_Year_Fraction', 'Float'),
                  F('Yield', 'Percent'), F('Margin', 'Basis'), F('Rate_Multiplier', 'Float'),
                  F('Is_Coupon', 'Text', values=['Yes', 'No'])]))]),
        F('Calendars', 'Text', default='', convention=True),
        F('Is_Forward_Deal', 'Text', default='No', convention=True, values=['Yes', 'No'])
    ])]

    documentation = ('Inflation', ['This pays a fixed coupon on an inflation indexed principal. Define the following: ',
                                   '',
                                   '- $P$ the principal amount',
                                   '- $T_b$ the base reference date',
                                   '- $T_f$ the final reference date',
                                   '- $t_1$ the accrual start date',
                                   '- $t_2$ the accrual end date',
                                   '- $\\alpha$ the accrual daycount from $t_1$ to $t_2$',
                                   '- $r$ the fixed yield',
                                   '- $A\\ne 0$ is the rate multiplier',
                                   '',
                                   'The cashflow payoff is',
                                   '',
                                   '$$P\\Big(A\\frac{I_R(t,T_f)}{I_R(t,T_b)}\\Big)r\\alpha$$',
                                   '',
                                   'Under a simulation a price index with no model is constant: the deal is priced',
                                   'only where the calculation\'s `NoModel` is `RiskNeutral` - the print in force',
                                   'on every row that one constant, the inflation curve read forward from the row, or',
                                   'off its own model where it has one - and is skipped by name otherwise, as any',
                                   'deal its pricer cannot value.',
                                   ])

    def reset(self, calendars):
        super(YieldInflationCashflowListDeal, self).reset()

        paydates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        if self.field.get('Is_Forward_Deal', 'No') == 'Yes':
            resetdates = {x for x in paydates if x < self.field['Settlement_Date']}
            resetdates.add(self.field['Settlement_Date'])
        else:
            resetdates = paydates

        self.add_reval_dates(resetdates, self.field['Currency'])

    def validate(self):
        """Each cashflow must pin both index references: the known VALUE, or the DATE to read it at.

        TensorCashFlows.index flags a reference stating no value and reads it off its date, the
        base date where none was given - so supplying neither prices against the base date's
        reference rather than failing.
        """
        for i, cashflow in enumerate((self.field.get('Cashflows') or {}).get('Items') or []):
            for ref in ('Base', 'Final'):
                if not (cashflow.get(ref + '_Reference_Value') or cashflow.get(ref + '_Reference_Date')):
                    yield ('Cashflows.Items[{}]: supply {}_Reference_Value or '
                           '{}_Reference_Date'.format(i, ref, ref))

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {
            'Currency': utils.check_rate_name(self.field['Currency']),
            'Index': utils.check_rate_name(self.field['Index'])
        }

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['Repo_Rate'] = self.check_factor_name('Repo_Rate', field['Discount_Rate'])
        field['Settlement_Rate'] = self.check_factor_name('Settlement_Rate', field['Discount_Rate'])
        field['PriceIndex'] = utils.check_rate_name(get_inflation_index_name(field['Index'], all_factors))

        field_index = {
            'ForecastIndex': get_inflation_factor(
                field['Index'], static_offsets, stochastic_offsets, all_tenors),
            'PriceIndex': get_price_index_factor(
                field['PriceIndex'], static_offsets, stochastic_offsets),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Repo_Rate': get_interest_factor(
                field['Repo_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Settlement_Rate': get_interest_factor(
                field['Settlement_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets)
        }

        inflation_factor, index_factor = get_inflation_index_objects(field['Index'], field['PriceIndex'], all_factors)
        # rule and lag stay separate: any lag is valid, and the pricer needs only how many index
        # observations a cashflow carries
        interpolated = self.field['Index_Reference']['Reference_Type'] == 'Interpolated'
        months_lag = int(self.field['Index_Reference']['Months_Lag'])
        field_index['Resets_Per_Cashflow'] = 2 if interpolated else 1

        field_index['Cashflows'], field_index['Base_Resets'], field_index['Final_Resets'] = utils.TensorCashFlows.index(
            base_date, time_grid, 1 if self.field['Buy_Sell'] == 'Buy' else -1, self.field['Cashflows'],
            inflation_factor, index_factor, self.field.get('Settlement_Date'), months_lag, interpolated)

        field_index['SettleCurrency'] = self.field['Currency']
        field_index['Settlement_Date'] = (self.field.get('Settlement_Date') -
                                          base_date).days if self.field.get('Settlement_Date') else None

        return field_index

    def generate(self, shared, time_grid, deal_data):
        if not deal_data.Factor_dep['PriceIndex'][0][utils.FACTOR_INDEX_Stoch] and len(
                deal_data.Time_dep.deal_time_grid) > 1 and not shared.riskneutral:
            # skipped and counted as any deal its pricer cannot value
            raise ValueError('the price index of {} has no model and NoModel is Constant: give the '
                             'index a model, or set NoModel to RiskNeutral, which reads the inflation '
                             'curve forward from each row, or off its own model where it has '
                             'one'.format(self.field['Index']))
        return pricing.pv_index_leg(shared, time_grid, deal_data)


class CapDeal(Deal):
    accepts_children = True
    vernacular = 'interest rate cap, cap, caplets'
    observes = Observes('Forecast_Rate', 'InterestRate')
    fields = [ADMIN, PRICING, FLOATING_INDEX, COUPON_SCHEDULE, CAP_FLOOR, own('CapDeal', [
        F('Cap_Rate', 'Float', default=0.0)
    ])]

    #: what each period is an option on: the strike the schedule reads and the pricer it takes
    optionlet = ('Cap_Rate', pricing.pricer_cap)

    documentation = (
        'Interest Rates', ['A series of caplets as described [here](#capletsfloorlets), on the',
                           '[coupon schedule](#coupon-schedules) its terms spell'])

    def reset(self, calendars):
        super(CapDeal, self).reset()
        self.resetdates, self.pays = self.coupon_schedule(
            calendars, self.field['Effective_Date'], self.field['Maturity_Date'], self.field['Payment_Interval'])
        self.add_reval_dates(self.resetdates.union(self.pays), self.field['Currency'])
        # this swap could be quantoed
        self.isQuanto = None

    finalize_dates = Deal.children_dates
    post_process = Deal.sum_children

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        strike, pricer = self.optionlet
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['Forecast_Rate'] = self.check_factor_name('Forecast_Rate', field['Discount_Rate'])
        field['Forecast_Rate_Volatility'] = utils.check_rate_name(self.field['Forecast_Rate_Volatility'])

        field_index = {}
        Principal = self.field['Principal']
        Amortisation = self.field.get('Amortisation')
        Known_Rates = self.field.get('Known_Rates')

        self.isQuanto = get_interest_rate_currency(field['Forecast_Rate'], all_factors) != field['Currency']
        if self.isQuanto:
            # TODO - Deal with Quanto Interest Rate swaps
            pass
        else:
            field_index['Forward'] = get_interest_factor(
                field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Discount'] = get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors)
            field_index['Currency'] = get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
            field_index['VolSurface'] = get_interest_vol_factor(
                field['Forecast_Rate_Volatility'], self.field['Payment_Interval'],
                static_offsets, stochastic_offsets, all_tenors)
            # on its optionlet list it prices the list (`post_process`): its own caplets go unread
            if not self.has_children:
                field_index['Cashflows'] = utils.TensorCashFlows.generate_float(
                    base_date, time_grid, self.resetdates,
                    (1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0) * Principal,
                    Amortisation, Known_Rates, self.field['Index_Tenor'], self.field['Reset_Frequency'],
                    utils.DayCount.code(self.field['Accrual_Day_Count']), self.field[strike] / 100.0,
                    self.field.get('Reference'), self.pays, self.float_index(calendars))
            field_index.update(
                SettleCurrency=self.field['Currency'], CompoundingMethod='None',
                AveragingMethod=self.field['Averaging_Method'], Model=pricer)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_float_leg(shared, time_grid, deal_data)


class FloorDeal(CapDeal):
    vernacular = 'interest rate floor, floor, floorlets'
    # the schema reads each type's own declarations, never a parent's
    observes = Observes('Forecast_Rate', 'InterestRate')
    fields = [ADMIN, PRICING, FLOATING_INDEX, COUPON_SCHEDULE, CAP_FLOOR, own('FloorDeal', [
        F('Floor_Rate', 'Float', default=0.0)
    ])]

    optionlet = ('Floor_Rate', pricing.pricer_floor)

    documentation = (
        'Interest Rates', ['A series of floorlets as described [here](#capletsfloorlets), on the',
                           '[coupon schedule](#coupon-schedules) its terms spell'])


class SwaptionDeal(Deal):
    accepts_children = True
    # its legs are the underlying the option is written on: a position sizes them, flips the option
    option_on_children = True
    vernacular = 'swaption, payer swaption, receiver swaption'
    fields = [ADMIN, PRICING, FLOATING_INDEX, own('SwaptionDeal', [
        F('Floating_Margin', 'Float', default=0.0, convention=True),
        F('Rate_Schedule', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float')])),
        F('Settlement_Style', 'Text', default='Physical', convention=True, values=['Physical', 'Cash']),
        F('Cash_Settlement_Method', 'Text', default='Cash_Price', convention=True,
          values=['Cash_Price', 'Par_Yield_Curve'],
          description='The annuity a cash settlement is struck on: the swap\'s own, or its fixed '
                      'coupons discounted at the swap rate.'),
        F('Swap_Rate', 'Float', default=0.0),
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Option_Expiry_Date', 'Date', default=''),
        F('Forecast_Rate_Volatility', 'Text', default='', convention=True,
          factors=['InterestRateVol', 'InterestYieldVol']),
        F('Settlement_Date', 'Date', default=''),
        F('Margin_Schedule', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float')])),
        F('Principal', 'Float', default=0.0, sized=True),
        F('Swap_Maturity_Date', 'Date', default=''),
        F('Swap_Effective_Date', 'Date', default=REQUIRED),
        F('Forecast_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Payer_Receiver', 'Text', default='Payer', values=['Payer', 'Receiver'])
    ]), own('SwaptionDeal', [
        F('Pay_Amortisation', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float', sized='magnitude')])),
        F('Pay_First_Coupon_Date', 'Date', default='', convention=True),
        F('Pay_Timing', 'Text', default='End', convention=True, values=['End', 'Begin', 'Discounted']),
        F('Pay_Payment_Offset', 'Integer', default=0, convention=True),
        F('Pay_Penultimate_Coupon_Date', 'Date', default='', convention=True),
        F('Pay_Payment_Calendars', 'Text', default='', convention=True),
        F('Pay_Frequency', 'Period', default='3M', convention=True),
        F('Pay_Day_Count', 'Text', default='ACT_365', convention=True, values=DAY_COUNTS),
        F('Pay_Calendars', 'Text', default='', convention=True)
    ], 'Pay', 'Pay leg'), own('SwaptionDeal', [
        F('Receive_Penultimate_Coupon_Date', 'Date', default='', convention=True),
        F('Receive_Day_Count', 'Text', default='ACT_365', convention=True, values=DAY_COUNTS),
        F('Receive_First_Coupon_Date', 'Date', default='', convention=True),
        F('Receive_Amortisation', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float', sized='magnitude')])),
        F('Receive_Payment_Calendars', 'Text', default='', convention=True),
        F('Receive_Timing', 'Text', default='End', convention=True, values=['End', 'Begin', 'Discounted']),
        F('Receive_Frequency', 'Period', default='3M', convention=True),
        F('Receive_Calendars', 'Text', default='', convention=True),
        F('Receive_Payment_Offset', 'Integer', default=0, convention=True)
    ], 'Receive', 'Receive leg')]

    documentation = ('Interest Rates', [
        'Let $t_0, T_1$ and $T_2$ be the **Option_Expiry_Date**, **Swap_Effective_Date** and',
        '**Swap_Maturity_Date** ($t_0 \\le T_1 \\lt T_2$). The underlying swap is the deal\'s two',
        '`Children`, a `CFFixedInterestListDeal` and a `CFFloatingInterestListDeal`, each signed by its',
        'own **Buy_Sell**. With $V_{fixed}$, $V_{float}$ their values, $V_{margin}$ the value of the',
        'floating leg\'s margin cashflows and $A(t)$ the fixed leg\'s annuity (its principals times',
        'accrual fractions, discounted),',
        '',
        '$$s(t)=\\frac{V_{float}(t)-V_{margin}(t)}{A(t)},\\qquad K(t)=\\frac{V_{fixed}(t)-V_{margin}(t)}{A(t)}$$',
        '',
        'and the underlying swap is worth $U(t)=\\delta(s(t)-K(t))A(t)$, $\\delta=+1$ for a **Payer** and',
        '$-1$ for a **Receiver**. Before expiry the option is',
        '',
        '$$V(t)=\\omega\\,A(t)\\,\\mathcal B_\\delta\\big(s(t),\\,k,\\,\\sigma\\sqrt{t_0-t}\\big)$$',
        '',
        'with $\\omega=\\pm1$ for **Buy**/**Sell** and $k$ = **Swap_Rate**/100: the strike is the deal\'s',
        'own rate, while the moneyness the vol is read at is $s(t)-K(t)$. $\\mathcal B_\\delta$ is Black on a',
        '`Lognormal` surface (displaced by its **Shift**) and Bachelier on a `Normal` one. $\\sigma$ is the',
        '**Forecast_Rate_Volatility** surface at moneyness $s-K$ (decimal), expiry $t_0-t$ in the discount',
        'curve\'s day count, and tenor $T_2-T_1$ in **Index_Day_Count**.',
        '',
        '#### Physically Settled Swaptions',
        '',
        '**Settlement_Style** `Physical`: the holder exercises where $\\delta(s-K)\\ge0$ on the last scenario',
        'date before expiry, and on those paths the deal is $\\omega U(t)$ from then on, its cash what the',
        'swap\'s legs pay.',
        '',
        '#### Cash Settled Swaptions',
        '',
        '**Settlement_Style** `Cash` pays the option\'s expiry value on **Settlement_Date** $T$ (the expiry',
        'where blank), held at that value after $t_0$, struck on the annuity **Cash_Settlement_Method** names.',
        '`Cash_Price` is the swap\'s own annuity $A$, ISDA\'s Cash Price (Collateralized Cash Price where the',
        'discount curve is the collateral\'s), and the deal is $V(t)D(t,T)/D(t,t_0)$. `Par_Yield_Curve`',
        'discounts the fixed leg\'s own coupons at the swap rate,',
        '$A_c(s)=\\sum_k N_k\\tau_k\\prod_{j\\le k}(1+\\tau_j s)^{-1}$, which on a regular 30/360 leg is ISDA\'s',
        'unadjusted par yield annuity, and is',
        '',
        '$$V(t)=\\omega\\,D(t,T)\\,A_c\\big(s(t)\\big)\\,\\mathcal B_\\delta\\big(s(t),k,\\sigma\\sqrt{t_0-t}\\big),$$',
        '',
        'the market formula, $A_c$ frozen at the forward swap rate. Against a one-factor Hull-White fitted',
        'to the same vol, a 1Y into 5Y payer at the money and 100bp either side reads 0.1% to 0.3% of its',
        'value low (under 0.6bp of notional) on a curve sloped from 1% to 6%, where the swap\'s own annuity',
        'reads it 2.3% to 2.4% high.',
        '',
        '#### Limitations',
        '',
        'An expiry before the base date is refused by name. A swap effective after the expiry (forward',
        'start) or amortising exercises into exactly the swap its legs make, but reads its vol at the',
        'vanilla point above, expiry $t_0$ and tenor $T_2-T_1$, unadjusted. Against a one-factor Hull-White',
        'fitted to that point at 5% mean reversion, a 1Y option on a 5Y swap amortising to a tenth marks',
        '3.5% low and one on a 5Y swap starting a year after its expiry 5.1% high on a flat 4% curve, and',
        '21% low and 29% high on a curve sloped from 1% to 6%.'])

    def reset(self, calendars):
        super(SwaptionDeal, self).reset()
        self.add_reval_dates({self.field['Option_Expiry_Date']})
        self.legs = {leg: self.coupon_schedule(
            calendars, self.field['Swap_Effective_Date'], self.field['Swap_Maturity_Date'],
            self.field[leg + 'Frequency'], leg, 'Calendars', 'Timing') for leg in ('Pay_', 'Receive_')}

        if self.field['Settlement_Style'] == 'Physical':
            for dates, pays in self.legs.values():
                self.add_reval_dates(dates.union(pays), self.field['Currency'])
        else:
            settlement = self.field['Settlement_Date'] or self.field['Option_Expiry_Date']
            self.add_reval_dates({settlement}, self.field['Currency'])

    def finalize_dates(self, parser, base_date, grid, node_children, node_resets, node_settlements):
        if self.field['Settlement_Style'] == 'Cash':
            # the legs only price the option, which pays one amount on its own settlement day
            node_resets.clear()
            node_settlements.clear()
        elif node_children is not None:
            # exercised, it is the swap its legs pay, held from the day after the expiry
            expiry = self.field['Option_Expiry_Date']
            super(SwaptionDeal, self).reset()
            self.add_reval_dates({expiry, expiry + pd.DateOffset(days=1)})
            for currency, dates in node_settlements.items():
                self.add_reval_dates(dates, currency)

        return super(SwaptionDeal, self).finalize_dates(parser, base_date, grid, node_children, node_resets,
                                                        node_settlements)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['Forecast_Rate'] = self.check_factor_name('Forecast_Rate', field['Discount_Rate'])
        field['Forecast_Rate_Volatility'] = utils.check_rate_name(self.field['Forecast_Rate_Volatility'])

        field_index = {'SettleCurrency': self.field['Currency'],
                       'Forward': get_interest_factor(
                           field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors),
                       'Discount': get_interest_factor(
                           field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
                       'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'VolSurface': get_interest_vol_factor(
                           field['Forecast_Rate_Volatility'], pd.DateOffset(years=2),
                           static_offsets, stochastic_offsets, all_tenors),
                       'Expiry': (self.field['Option_Expiry_Date'] - base_date).days}
        if field_index['Expiry'] < 0:
            raise utils.UnpriceableSchedule('{}: Option_Expiry_Date {:%Y-%m-%d} is behind the base date, {}'.format(
                self.field.get('Reference'), self.field['Option_Expiry_Date'],
                'and the cash amount it fixed is stated nowhere' if self.field['Settlement_Style'] == 'Cash' else
                'so it is the swap it delivered or nothing: book that swap, or nothing if it lapsed'
                ))
        if self.field['Settlement_Style'] == 'Cash':
            # the day it pays, the one its reset registered
            field_index['Settlement'] = (max(self.settlement_currencies[self.field['Currency']]) - base_date).days
            field_index['Par_Yield'] = self.field['Cash_Settlement_Method'] == 'Par_Yield_Curve'
        field_index['Cash_Settled'] = self.field['Settlement_Style'] != 'Physical'
        field_index['Underlying_Swap_maturity'] = utils.DayCount.accrual(
            base_date, (self.field['Swap_Maturity_Date'] - self.field['Swap_Effective_Date']).days,
            utils.DayCount.code(self.field['Index_Day_Count']))
        # on its legs it prices them (`post_process`): the swap its own terms spell goes unread
        if self.has_children:
            return field_index

        fixed, floating, sign = ('Pay_', 'Receive_', -1) if self.field['Payer_Receiver'] == 'Payer' else (
            'Receive_', 'Pay_', 1)
        dates, pays = self.legs[fixed]
        field_index['FixedCashflows'] = utils.TensorCashFlows.generate_fixed(
            base_date, dates, sign * self.field['Principal'], self.field[fixed + 'Amortisation'],
            utils.DayCount.code(self.field[fixed + 'Day_Count']), self.field['Swap_Rate'] / 100.0, pays)
        dates, pays = self.legs[floating]
        field_index['FloatCashflows'] = utils.TensorCashFlows.generate_float(
            base_date, time_grid, dates, -sign * self.field['Principal'], self.field[floating + 'Amortisation'],
            None, self.field[floating + 'Frequency'], self.field['Index_Tenor'],
            utils.DayCount.code(self.field[floating + 'Day_Count']), self.field['Floating_Margin'] / 10000.0,
            self.field.get('Reference'), pays, self.float_index(calendars))

        if self.field['Settlement_Style'] == 'Physical':
            # remember to potentially deliver the underlying swap if it's in the money
            field_index['FixedStartIndex'] = field_index['FixedCashflows'].get_cashflow_start_index(time_grid.time_grid)
            field_index['FloatStartIndex'] = field_index['FloatCashflows'].get_cashflow_start_index(time_grid.time_grid)
        else:
            field_index['FixedStartIndex'] = np.zeros(1, dtype=np.int32)
            field_index['FloatStartIndex'] = np.zeros(1, dtype=np.int32)

        return field_index

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        daycount_fn = factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount]
        child_map = {}
        buy_sell = {}

        for child in child_dependencies:
            # make the child price to the same grid as the parent
            child.Time_dep.assign(deal_data.Time_dep)
            child_map[child.Instrument.field['Object']] = child
            buy_sell[child.Instrument.field['Object']] = -1.0 if child.Instrument.field['Buy_Sell'] == 'Sell' else 1.0

        FX_rep = utils.calc_fx_cross(factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        buysell = 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0
        delta = 1.0 if self.field['Payer_Receiver'] == 'Payer' else -1.0
        # what each leg pays, signed as the holder of the swap receives it, booked where exercised
        swap_cash = []

        def swap_leg(sign):
            return not factor_dep['Cash_Settled'] and (
                lambda _, currency, index, cash: swap_cash.append((currency, index, buysell * sign * cash))
                )

        # pvbp, vfixed and vfloat should all be positive
        pvbp = buy_sell['CFFixedInterestListDeal'] * pricing.pv_fixed_cashflows(
            shared, time_grid, child_map['CFFixedInterestListDeal'], ignore_fixed_rate=True, settle_cash=False)
        vfixed = buy_sell['CFFixedInterestListDeal'] * pricing.pv_fixed_cashflows(
            shared, time_grid, child_map['CFFixedInterestListDeal'],
            settle_cash=swap_leg(-delta * buy_sell['CFFixedInterestListDeal'])
            )
        vfloat = buy_sell['CFFloatingInterestListDeal'] * pricing.pv_float_cashflow_list(
            shared, time_grid, child_map['CFFloatingInterestListDeal'], pricing.pricer_float_cashflows,
            settle_cash=swap_leg(delta * buy_sell['CFFloatingInterestListDeal'])
            )

        if child_map['CFFloatingInterestListDeal'].Factor_dep[
               'Cashflows'].schedule[:, utils.CASHFLOW_INDEX_FloatMargin].any():
            vmargin = buy_sell['CFFloatingInterestListDeal'] * pricing.pv_fixed_cashflows(
                shared, time_grid, child_map['CFFloatingInterestListDeal'], settle_cash=False)
        else:
            vmargin = 0.0

        st = (vfloat - vmargin) / pvbp
        Kt = (vfixed - vmargin) / pvbp
        mn = st - Kt
        # get the vol subtype (distribution_type, shift) tuple - only on the base (first vol factor)
        dist, shf = factor_dep['VolSurface'][0][utils.FACTOR_INDEX_SubType]
        pricing_fn = utils.black_european_option if dist == 'Lognormal' else utils.bachelier_european_option
        tenor_in_days = factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]
        tenor = daycount_fn(tenor_in_days)

        if factor_dep['Cash_Settled']:
            vols = utils.VolSurface.tenor_rate(
                factor_dep['VolSurface'], mn, tenor,
                factor_dep['Underlying_Swap_maturity'], shared)

            theo_price = pricing_fn(
                st, self.field['Swap_Rate'] / 100.0, vols, tenor, buysell, delta, shared, shift=shf.amount)

            # the amount the expiry fixes, held past it and paid on the settlement day
            discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
            to_settlement, to_expiry = (
                torch.squeeze(utils.calc_discount_rate(discount, days.clip(min=0).reshape(-1, 1), shared), dim=1)
                for days in (factor_dep['Settlement'] - deal_time[:, utils.TIME_GRID_MTM], tenor_in_days)
                )
            if factor_dep['Par_Yield']:
                # the fixed coupons discounted at the swap rate, compounded period by period, frozen
                # at the forward: Black on the cash annuity, the market formula
                fixed = child_map['CFFixedInterestListDeal'].Factor_dep['Cashflows'].dual()
                accrual = fixed.tn[:, utils.CASHFLOW_INDEX_Year_Frac]
                annuity = torch.sum(
                    buy_sell['CFFixedInterestListDeal'] * fixed.tn[:, utils.CASHFLOW_INDEX_Nominal] * accrual *
                    torch.exp(-torch.cumsum(torch.log1p(st.unsqueeze(-1) * accrual), dim=-1)), dim=-1)
                value = utils.held_past_expiry(annuity * theo_price, tenor_in_days, None) * to_settlement
            else:
                value = utils.held_past_expiry(pvbp * theo_price, tenor_in_days, None) * to_settlement / to_expiry
            pricing.cash_settle(shared, factor_dep['SettleCurrency'], deal_data.Time_dep.deal_time_grid[-1], value[-1])

            mtm = FX_rep * value
        else:
            expiry = tenor[tenor >= 0.0]
            counts = (expiry.size, tenor.size - expiry.size)

            F_option, F_swap = torch.split(pvbp, counts)
            spot_option, spot_swap = torch.split(st, counts)
            mn_option, mn_swap = torch.split(mn, counts)

            vols = utils.VolSurface.tenor_rate(
                factor_dep['VolSurface'], mn_option, expiry,
                factor_dep['Underlying_Swap_maturity'], shared)

            theo_price = pricing_fn(
                spot_option, self.field['Swap_Rate'] / 100.0, vols, expiry, buysell, delta, shared, shift=shf.amount)

            value = F_option * theo_price
            Ut_swap = delta * mn_swap * F_swap

            if Ut_swap.shape[0]:
                decision = delta * mn_option[-1]
                exercised = decision >= 0
                Ut_mask = Ut_swap * exercised
                mtm = FX_rep * torch.cat([value, buysell * Ut_mask], dim=0)
                # what the swap pays, on the paths that exercised into it
                for currency, index, cash in swap_cash:
                    pricing.cash_settle(shared, currency, index, cash * exercised)
                if getattr(shared, 'boundary_aad', False) and time_grid.report_index is not None:
                    self.register_exercise_boundary(
                        shared, time_grid, deal_data, FX_rep, value,
                        buysell * Ut_swap, decision, exercised, swap_cash)
            else:
                mtm = FX_rep * value

        # vanilla swaptions only: an amortisation schedule would need the underlying tenor
        # adjusted, which is not done here

        return pricing.interpolate(mtm, shared, time_grid, deal_data)

    @staticmethod
    def register_exercise_boundary(shared, time_grid, deal_data, fx_rep, option, swap, gap, exercised, swap_cash):
        """Record the frozen exercise decision so its derivative can be restored.

        Physical settlement is genuinely path dependent, so the jump is real product economics and
        must not be smoothed; what ordinary AAD drops is the flux of scenarios across the exercise
        boundary. The gap IS the DECISION the pricer took - `delta * mn_option[-1]`, the moneyness
        on the last pre-expiry row - so `fired` is `gap >= 0` by construction; a gap taken on the
        first post-expiry row would put the flux at a boundary nothing was decided on. Both branches
        are the two sides of the mask the pricer just evaluated, so nothing is re-priced.

        Registered as a single-decision LatchedBoundarySet, the shape a discrete barrier uses:
        `triggered` holds the swap, `untriggered` lets the option lapse. The decision is a property
        of the SCENARIO, resolved for every row alike, so `obs_before` needs no row labels. Each
        payment of the swap, `swap_cash`, is the decision's own: made if it fired, nothing if not.

        `gap_disagrees` on the debug line is the only place a wrongly sourced gap is visible: the
        reported profile is the same either way and only the gradient moves.
        """
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug('SWAPTION BOUNDARY %s fired=%d/%d gap_disagrees=%d',
                          deal_data.Instrument.field.get('Reference', 'unnamed'),
                          int(exercised.sum()), exercised.numel(),
                          int(((gap.detach() >= 0) != exercised).sum()))
        triggered = torch.cat([option, swap], dim=0).detach()
        cash_events = []
        for _, index, cash in swap_cash:
            row = np.searchsorted(deal_data.Time_dep.deal_time_grid, index)
            paid = (pricing.report_fx(fx_rep, row) * cash + torch.zeros_like(gap)).detach()
            cash_events.append((int(index), 0, paid, torch.zeros_like(paid), paid * exercised))
        shared.boundary_sets.append(utils.LatchedBoundarySet(
            gaps=[gap], fired=[exercised.detach()],
            obs_before=np.ones(triggered.shape[0], dtype=np.int64),
            triggered=triggered,
            untriggered=torch.cat([option, torch.zeros_like(swap)], dim=0).detach(),
            cash_events=cash_events or None,
            to_mtm=pricing.deal_to_mtm_grid(time_grid, deal_data, fx_rep),
            report_index=time_grid.report_index))

    def generate(self, shared, time_grid, deal_data):
        # TODO - move the post_process pricing code into the pricing module and call it from
        # both here and post_process
        raise Exception('generate in {0} - Not implemented yet'.format(self.__class__.__name__))


class FXDiscreteExplicitAsianOption(Deal):
    vernacular = 'FX Asian option, average-rate option'
    observes = Observes(('Underlying_Currency', 'Currency'), 'FxRate', 'Sampling_Data', 1)
    fields = [ADMIN, FX_ADMIN, FXOPTIONBASE, own('FXDiscreteExplicitAsianOption', [
        F('Expiry_Date', 'Date', default=''),
        F('Is_Digital', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Underlying_Amount', 'Float', default=0.0, sized=True),
        F('Sampling_Data', 'Table', default='null', description='Sampling_Data',
          row=Row([F('Date', 'Date'), F('Price', 'Float'), F('Weight', 'Float')]))
    ])]

    documentation = ('Fx And Equity', [
        'A discretely sampled FX Asian option on **Underlying_Amount** units of **Underlying_Currency**,',
        'priced in **Currency** as described in [Discrete Asian Options](#discrete-asian-options).'])

    pricer = staticmethod(pricing.pv_discrete_asian_option)

    def reset(self, calendars):
        super(FXDiscreteExplicitAsianOption, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['FX_Volatility'] = utils.check_rate_name(self.field['FX_Volatility'])

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_vol_factor('FXVol',
                                         field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Invert_Moneyness': 1 if field['Currency'][0] == field['FX_Volatility'][0] else 0,
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])
        }
        field_index.update(self.samples(base_date, time_grid))
        return field_index

    def samples(self, base_date, time_grid):
        """The averaged fixings and the strike, as this type's pricer reads them."""
        missing_fixings = [x for x in self.field['Sampling_Data'] if x[0] < base_date and not x[1]]
        if missing_fixings:
            logging.error('Past fixings not defined - please specify fixings for {}'.format(
                ', '.join([str(x[0]) for x in missing_fixings])))
        return {'Digital': self.field.get('Is_Digital', 'No') == 'Yes',
                'Samples': utils.TensorResets.from_observations(
                    base_date, time_grid, self.field['Sampling_Data'], weighted=True),
                'Strike': self.field['Strike_Price']}

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        FX_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
        spot = utils.calc_fx_cross(
            deal_data.Factor_dep['Underlying_Currency'][0],
            deal_data.Factor_dep['Currency'][0], deal_time, shared)
        forward = utils.calc_fx_forward(
            deal_data.Factor_dep['Underlying_Currency'], deal_data.Factor_dep['Currency'],
            deal_data.Factor_dep['Expiry'], deal_time, shared)

        mtm = self.pricer(
            shared, time_grid, deal_data, self.field['Underlying_Amount'], spot, forward,
            (deal_data.Factor_dep['Currency'][1], deal_data.Factor_dep['Underlying_Currency'][1]),
            [deal_data.Factor_dep['Underlying_Currency'][0], deal_data.Factor_dep['Currency'][0]],
            invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'], use_forwards=True) * FX_rep

        return mtm


class FXDiscreteExplicitDoubleAsianOption(FXDiscreteExplicitAsianOption):
    vernacular = 'FX double Asian, average-strike average-rate option'
    fields = [ADMIN, FX_ADMIN, FXOPTIONBASE, own('FXDiscreteExplicitDoubleAsianOption', [
        F('Expiry_Date', 'Date', default=''),
        F('Underlying_Amount', 'Float', default=0.0, sized=True),
        F('Strike_Multiplier', 'Float', default=1.0, convention=True),
        F('Sampling_Data_1', 'Table', default='null', description='Sampling_Data_1',
          row=Row([F('Date', 'Date'), F('Price', 'Float'), F('Weight', 'Float')])),
        F('Sampling_Multiplier_1', 'Float', default=1.0, convention=True),
        F('Sampling_Data_2', 'Table', default='null', description='Sampling_Data_2',
          row=Row([F('Date', 'Date'), F('Price', 'Float'), F('Weight', 'Float')])),
        F('Sampling_Multiplier_2', 'Float', default=1.0, convention=True)
    ])]

    observes = None
    documentation = ('Fx And Equity', [
        'A spread option on two discretely sampled FX averages, specified in',
        '[Discrete Double Asian Options](#discrete-double-asian-options).'])

    pricer = staticmethod(pricing.pv_discrete_double_asian_option)

    def samples(self, base_date, time_grid):
        """The two averaged strips, their multipliers and the strike."""
        return {'Alpha_0': self.field.get('Strike_Multiplier', 1.0),
                'Alpha_1': self.field.get('Sampling_Multiplier_1', 1.0),
                'Alpha_2': self.field.get('Sampling_Multiplier_2', 1.0),
                'Samples_1': utils.TensorResets.from_observations(
                    base_date, time_grid, self.field['Sampling_Data_1'], weighted=True),
                'Samples_2': utils.TensorResets.from_observations(
                    base_date, time_grid, self.field['Sampling_Data_2'], weighted=True),
                'Strike': self.field.get('Strike_Price', 0.0)}


class EquityDiscreteExplicitAsianOption(Deal):
    vernacular = 'equity Asian option, average-rate option'
    # a Compo's samples are S*X, which no print of the equity is
    observes = Observes('Equity', 'EquityPrice', 'Sampling_Data', 1,
                        unless=('Payoff_Type', 'Compo'))
    fields = [ADMIN, PRICING, EQUITYOPTIONBASE, own('EquityDiscreteExplicitAsianOption', [
        F('Is_Digital', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Units', 'Float', default=0.0, sized=True),
        F('Sampling_Data', 'Table', default='null', description='Sampling_Data',
          row=Row([F('Date', 'Date'), F('Price', 'Float'), F('Weight', 'Float')])),
        F('Payoff_Type', 'Text', default='Standard', convention=True, values=['Standard', 'Quanto', 'Compo'])
    ])]

    documentation = ('Fx And Equity', [
        'A path independent option described [here](#discrete-asian-options)',
        '',
        'Under **Payoff_Type** `Compo` the average is on $S\\cdot X$: every future sample takes the',
        'composite forward, carry and vol at its own tenor, the realised fixings the document carries',
        'ARE composite prices and nothing converts them, and the local smile is read at the translated',
        'strike $K/F_X(T)$.'])

    def reset(self, calendars):
        super(EquityDiscreteExplicitAsianOption, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.payoff_ccy)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Payoff_Currency': utils.check_rate_name(self.payoff_ccy),
                 'Equity': utils.check_rate_name(self.field['Equity']),
                 'Equity_Volatility': utils.check_rate_name(self.field['Equity_Volatility'])}

        field['Dividends'] = self.check_factor_name('Dividends', field['Equity'][:1])
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'Payoff_Currency': get_fxrate_factor(field['Payoff_Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.payoff_ccy,
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
            'Equity_Zero': get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Dividend_Yield': get_dividend_rate_factor(
                field['Dividends'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Digital': self.field.get('Is_Digital', 'No') == 'Yes',
            'Volatility': get_vol_factor('EquityPriceVol',
                                         field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Samples': utils.TensorResets.from_observations(
                base_date, time_grid, self.field['Sampling_Data'], weighted=True),
            'Strike': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0
        }

        missing_fixings = [x for x in self.field['Sampling_Data'] if x[0] < base_date and not x[1]]
        if missing_fixings:
            logging.error('Past fixings not defined - please specify fixings for {}'.format(
                ', '.join([str(x[0]) for x in missing_fixings])))

        self.check_option_data(field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors,
                               equity_fx_correlation)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        FX_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)
        spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)
        forward = utils.calc_eq_forward(
            deal_data.Factor_dep['Equity'], deal_data.Factor_dep['Equity_Zero'],
            deal_data.Factor_dep['Dividend_Yield'], deal_data.Factor_dep['Expiry'], deal_time, shared)

        mtm = pricing.pv_discrete_asian_option(
            shared, time_grid, deal_data, self.field['Units'], spot, forward,
            (deal_data.Factor_dep['Equity_Zero'], deal_data.Factor_dep['Dividend_Yield']),
            [deal_data.Factor_dep['Equity']]) * FX_rep

        return mtm


class EquityBarrierBinaryOption(Deal):
    vernacular = 'barrier digital, knock-in digital, knock-out binary'
    observes = Observes('Equity', 'EquityPrice', 'Barrier_Dates', 1, expires='Expiry_Date',
                        monitors=True)
    fields = [ADMIN, PRICING, EQUITYOPTIONBASE, own('EquityBarrierBinaryOption', [
        F('Barrier_Dates', 'Table', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Observed', 'Float')])),
        F('Cash_Payoff', 'Float', default=REQUIRED, sized=True),
        F('Barrier_Type', 'Text', default='Down_And_In',
          values=['Down_And_In', 'Down_And_Out', 'Up_And_In', 'Up_And_Out']),
        F('Barrier_Price', 'Float', default=0.0),
        F('Settlement_Date', 'Date', default='',
          settles=Cash('Payoff_Currency', otherwise='Expiry_Date'))
    ])]

    spot_models = ('None', 'LogVar2FJ')

    documentation = ('Fx And Equity', [
        'A discrete barrier binary (digital) option priced using the same One-Step Survival (OSS)',
        'Monte Carlo approach as `EquityBarrierOption`, with `isdigital=True` so the terminal payoff',
        'is a fixed cash amount rather than a vanilla call/put payoff.',
        'See `EquityBarrierOption` for the full OSS methodology, the `Barrier_Dates` `Observed`',
        'column and its compile-time fold, and the **SpotModel** valuation option shared by both',
        'deals - `LogVar2FJ`, which walks its own internal step and is exact where the',
        'observation dates lie on that grid. A knocked-in binary compiles as the plain',
        '`EquityBinaryOption` it now is.'
    ])

    def resolve_history(self, base_date, calendars, valuation_options):
        # the plain binary pays its cash as `Payoff`, the style it defaults to
        return resolve_discrete_barrier(
            self, base_date, calendars, valuation_options, 'EquityBinaryOption',
            Payoff=self.field['Cash_Payoff'])

    def reset(self, calendars):
        super(EquityBarrierBinaryOption, self).reset()
        barrierdates = {d for d, _ in barrier_monitoring_rows(self.field['Barrier_Dates'])}
        self.add_reval_dates(barrierdates.union({self.field['Expiry_Date']}), self.payoff_ccy)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        refuse_zero_payoff(self.field, 'EquityBarrierBinaryOption')
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Payoff_Currency': utils.check_rate_name(self.payoff_ccy),
                 'Equity': utils.check_rate_name(self.field['Equity']),
                 'Equity_Volatility': utils.check_rate_name(self.field['Equity_Volatility'])}

        field['Dividends'] = self.check_factor_name('Dividends', field['Equity'][:1])
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        ab = {d for d, _ in barrier_monitoring_rows(self.field.get('Barrier_Dates', []))}
        all_dates = sorted(ab.union([self.field['Expiry_Date']]))

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'Payoff_Currency': get_fxrate_factor(field['Payoff_Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.payoff_ccy,
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Equity': utils.declared_spot(get_equity_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets), field['Equity']),
            'Equity_Zero': get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Dividend_Yield': get_dividend_rate_factor(
                field['Dividends'], static_offsets, stochastic_offsets, all_tenors),
            'Volatility': get_vol_factor('EquityPriceVol',
                                         field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Observation_Dates': utils.TensorResets.from_observations(
                base_date, time_grid, [[x, 0] for x in all_dates]),
            'Barrier_Dates': [1 if x in ab else -1 for x in all_dates],
            'Strike_Price': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Expiry': (self.field['Expiry_Date'] - base_date).days
        }

        self.check_option_data(field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors,
                               equity_fx_correlation)

        # non-GBM spot model, by naming convention off the equity underlying and with no deal
        # field: <SpotModel>ModelParameters.<equity>; off or absent gives None (GBM)
        hn = get_spot_model_params_factor(
            self.options.get('SpotModel', 'None'), field['Equity'],
            all_factors, static_offsets, stochastic_offsets)
        set_spot_model_index(field_index, hn, self.options, self.field.get('Reference'))

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

        spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)
        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)

        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
        b = torch.squeeze(pricing.deal_carry_rate(
            deal_data.Factor_dep,
            (deal_data.Factor_dep['Equity_Zero'], deal_data.Factor_dep['Dividend_Yield']),
            tau.reshape(-1, 1), deal_time, shared), dim=1)

        spot = utils.spot_on_deal_grid(spot, deal_time, shared)

        pv = pricing.pv_discrete_barrier_option(
            shared, time_grid, deal_data, spot, b, tau, fx_rep, isdigital=True)

        mtm = pv * fx_rep

        return mtm


class EquityOptionDeal(Deal):
    vernacular = 'equity option, stock option, vanilla call or put'
    observes = Observes('Equity', 'EquityPrice', 'Price_Fixing', 1, elects='Settlement_Style',
                        expires='Expiry_Date')
    fields = [ADMIN, PRICING, EQUITYOPTIONBASE, OPTION_STYLE, own('EquityOptionDeal', [
        F('Units', 'Float', default=0.0, sized=True),
        F('Forward_Price_Date', 'Date', default='', convention=True),
        F('Payoff_Type', 'Text', default='Standard', convention=True, values=['Standard', 'Quanto', 'Compo']),
        F('Settlement_Date', 'Date', default='', convention=True,
          settles=Cash('Payoff_Currency', otherwise='Expiry_Date')),
        EXPIRY_PRINT
    ])]

    documentation = ('Fx And Equity', [
        'A vanilla option on **Units** shares, priced as described [here](./definitions.md#european-options)',
        'with the conventions listed there: Black on the forward to expiry plus two business days,',
        'the vol at the deal\'s moneyness and expiry, discounted from the settlement date.',
        '',
        'The payoff is paid on **Settlement_Date**, the expiry where blank. Past the expiry the',
        'underlying is what the expiry fixed: the path\'s own level on the expiry row or, where the',
        'expiry is behind the base date, the print **Price_Fixing** states for it, which it must.',
        '',
        'Under **Payoff_Type** `Quanto` or `Compo` the European prices as in',
        '[Quanto and Compo Payoffs](./definitions.md#quanto-and-compo-payoffs); under `Compo` the',
        'terminal law is the product $S\\cdot X$ and the local smile is read at the translated strike',
        '$K/F_X(T)$.',
        '',
        '**Option_Style** `American` is priced by the Bjerksund-Stensland trigger approximation [@bjerksund1993]',
        'with $r$ the discount curve\'s zero rate to expiry, $b=\\log(F/S)/\\tau$ and the same vol. For a call',
        'with $\\beta=\\tfrac12-\\tfrac{b}{\\sigma^2}+\\sqrt{(\\tfrac{b}{\\sigma^2}-\\tfrac12)^2+\\tfrac{2r}{\\sigma^2}}$, the',
        'flat exercise trigger is',
        '',
        '$$I=B_0+(B_\\infty-B_0)\\big(1-e^{h(\\tau)}\\big),\\quad h(\\tau)=-(b\\tau+2\\sigma\\sqrt\\tau)\\frac{B_0}{B_\\infty-B_0},\\quad B_0=K\\max\\Big(\\frac{r}{r-b},1\\Big),\\quad B_\\infty=\\frac{\\beta K}{\\beta-1}$$',
        '',
        'and the value below the trigger is the published closed form in $(I,\\beta)$, floored at the Black',
        'European; at or above it, $S-K$. A put is the call on $(K,S)$ with $r\\to r-b$, $b\\to-b$. Where',
        '$b\\ge r$ there is no early-exercise premium and the value is the Black European.',
        '',
        '- On a scenario path the option is monitored on the deal\'s grid dates: the first date the spot',
        '  reaches $I$ exercises, the intrinsic settles that day and the deal is worth 0 after.',
        '- A **Settlement_Date** on an American option is refused by name: it settles when exercised.',
        '- The American arm ignores **Payoff_Type**: no quanto or compo adjustment is applied.',
        '- Accuracy against a lattice or PDE: unmeasured.'])

    def __init__(self, params, valuation_options):
        super(EquityOptionDeal, self).__init__(params, valuation_options)
        # the binary subclass declares no Option_Style, so its absence reads European here
        self.path_dependent = self.field.get('Option_Style', 'European') == 'American'

    def reset(self, calendars):
        super(EquityOptionDeal, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']})
        self.add_settlement_dates()

    def pays_on_grid(self):
        # only an American option is path dependent: it can be exercised, and so settle, on any
        # grid date
        return self.path_dependent

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Equity': utils.check_rate_name(self.field['Equity']),
                 'Payoff_Currency': utils.check_rate_name(self.payoff_ccy),
                 'Equity_Volatility': utils.check_rate_name(self.field['Equity_Volatility'])}

        expiry, settlement, forward_settlement = option_date_info(self, base_date, calendars)
        if self.path_dependent and self.field['Settlement_Date']:
            raise utils.UnpriceableSchedule(
                '{}: an American option settles on the day it is exercised, so Settlement_Date is not '
                'its to state'.format(self.field.get('Reference', 'EquityOptionDeal')))

        field['Dividends'] = self.check_factor_name('Dividends', field['Equity'][:1])
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'Payoff_Currency': get_fxrate_factor(field['Payoff_Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.payoff_ccy,
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
            'Equity_Zero': get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Dividend_Yield': get_dividend_rate_factor(
                field['Dividends'], static_offsets, stochastic_offsets, all_tenors),
            'Volatility': get_vol_factor('EquityPriceVol', field['Equity_Volatility'], static_offsets,
                                         stochastic_offsets, all_tenors),
            'Strike_Price': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field.get('Buy_Sell', 'Buy') == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field.get('Option_Type', 'Call') == 'Call' else -1.0,
            'Option_Style': self.field.get('Option_Style', 'European'),
            'Expiry': expiry,
            'Settlement': settlement,
            'Forward_Settlement': forward_settlement,
            'Expiry_Fixing': expiry_fixing(self.field, base_date, time_grid, 'EquityOptionDeal')
            }

        self.check_option_data(field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors,
                               equity_fx_correlation)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        fx_rep = utils.calc_fx_cross(
            factor['Payoff_Currency'], shared.Report_Currency, deal_time, shared)

        strike = pricing.compo_strike(factor, deal_time, shared, factor['Strike_Price'] * shared.one)
        spot = utils.calc_time_grid_spot_rate(factor['Equity'], deal_time, shared)
        forward = utils.calc_eq_forward(
            factor['Equity'], factor['Equity_Zero'],
            factor['Dividend_Yield'], factor['Forward_Settlement'], deal_time, shared)
        moneyness = pricing.calc_moneyness(strike, spot, forward, deal_data)

        if factor['Option_Style'] == 'European':
            mtm = pricing.pv_european_option(
                shared, time_grid, deal_data, self.field['Units'], moneyness, forward) * fx_rep
        else:
            mtm = pricing.pv_american_option(
                shared, time_grid, deal_data, self.field['Units'], moneyness, spot, forward) * fx_rep

        return mtm


class EquityBinaryOption(EquityOptionDeal):
    vernacular = 'equity digital, binary option, cash-or-nothing, asset-or-nothing'
    observes = Observes('Equity', 'EquityPrice', 'Price_Fixing', 1, expires='Expiry_Date')
    fields = [ADMIN, PRICING, EQUITYOPTIONBASE, own('EquityBinaryOption', [
        F('Payoff', 'Float', default=REQUIRED, sized=True),
        F('Payoff_Style', 'Text', default='Cash', convention=True, values=['Cash', 'Asset']),
        F('Settlement_Date', 'Date', default='', convention=True,
          settles=Cash('Payoff_Currency', otherwise='Expiry_Date')),
        EXPIRY_PRINT
    ])]

    documentation = ('Fx And Equity', [
        'A European digital, priced with the conventions of [European Options](definitions.md#european-options).',
        '',
        'In the money at expiry it pays **Payoff**: that amount of cash under **Payoff_Style**',
        '`Cash` (cash-or-nothing), that many units of the equity under `Asset` (asset-or-nothing),',
        'on **Settlement_Date** - the expiry where blank - off the level the expiry fixed, which',
        'an expiry behind the base date states in **Price_Fixing**.',
        '',
        'Under **Payoff_Type** `Compo` the terminal law is the product $S\\cdot X$ and the local smile',
        'is read at the translated strike $K/F_X(T)$, each spread leg at its own.',
        '',
        'If the **Relative_Digital_Spread** Valuation Configuration option is set (> 0), the',
        'digital is priced as a call/put spread of width `Strike * Relative_Digital_Spread`',
        'either side of the strike, rather than the single-vol closed form, so the vol surface',
        'smile is picked up automatically.',
        '',
        'A **Payoff** of exactly 0.0 is refused by name at compile: the leg would price to zero',
        'on every path and the structure it was booked to cancel would keep its whole payoff.'])

    option_defaults = {'Relative_Digital_Spread': 0.0}

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        refuse_zero_payoff(self.field, 'EquityBinaryOption', fieldname='Payoff')
        return super(EquityBinaryOption, self).calc_dependencies(
            base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars)

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)

        strike = pricing.compo_strike(deal_data.Factor_dep, deal_time, shared,
                                      deal_data.Factor_dep['Strike_Price'] * shared.one)
        spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)
        forward = utils.calc_eq_forward(
            deal_data.Factor_dep['Equity'], deal_data.Factor_dep['Equity_Zero'],
            deal_data.Factor_dep['Dividend_Yield'], deal_data.Factor_dep['Forward_Settlement'], deal_time, shared)
        moneyness = pricing.calc_moneyness(strike, spot, forward, deal_data)

        # the spread legs' moneyness is authored HERE, where the deal's convention arguments
        # live - calc_moneyness stays the single owner of the surface's moneyness rule
        eps = self.options['Relative_Digital_Spread']
        spread = (eps,
                  pricing.calc_moneyness(strike * (1.0 - eps), spot, forward, deal_data),
                  pricing.calc_moneyness(strike * (1.0 + eps), spot, forward, deal_data)) if eps else None

        mtm = pricing.pv_european_option(
            shared, time_grid, deal_data, self.field['Payoff'], moneyness, forward,
            binary=True, digital_spread=spread,
            payoff_style=self.field['Payoff_Style'].lower()) * fx_rep

        return mtm


class QEDI_CustomAutoCallSwap(Deal):
    vernacular = 'autocall, autocallable, phoenix, snowball'
    observes = Observes('Equity', 'EquityPrice', 'Price_Fixing', 1, monitors=True)
    fields = [ADMIN, PRICING, EQUITYOPTIONBASE, OPTION_STYLE, QEDI_CUSTOMAUTOCALLSWAP]

    spot_models = ('None', 'LogVar2FJ')

    documentation = ('Fx And Equity', [
        'An autocall on one equity: contingent coupons that end the deal when the coupon window\'s',
        'average reaches its threshold, and a put paying $A/K-1+$**Rebate** of **Units** where the barrier',
        'is breached. Payoff, windows, valuation, refusals and limitations are specified in',
        '[Autocallable Swaps](#autocallable-swaps); this type has NO floating leg.',
        '',
        '**Barrier** is RELATIVE here: the put barrier is $B=$ **Barrier** $\\times$ **Strike_Price**.',
        '',
        '**Valuation options** (set in the Valuation Configuration section, per deal type)',
        '',
        '- **SpotModel**: `None` (default - lognormal dynamics off the implied vol surface) or',
        '`LogVar2FJ`, the two-factor log-variance model with an NIG residual',
        '([Stochastic Volatility](../theory/stochastic_volatility.md)), which drives the OSS simulation',
        'and its analytic legs. Its parameters resolve by naming convention from the',
        '`<SpotModel>ModelParameters.<underlying>` price factor (e.g. `LogVar2FJModelParameters.SPX`);',
        'it walks its own internal step and hands each fixing interval one Gaussian block law. Switching',
        'the model on without that factor in the market data is a loud skip, never a silent lognormal',
        'fallback. A `Quanto` payoff enters the walk as the per-step drift $-\\rho\\,\\sigma_{FX,k}\\sqrt{V_k}$,',
        '$\\rho$ the marked `Correlation.EquityPrice.<eq>/FxRate.<pair>`; declaring one without that',
        'correlation is a loud skip. `Compo` is refused: it is the product $S\\cdot X$ and this arm walks',
        'the equity alone.',
        '- **Steps_Per_Year**: the trading-day clock belongs to the parameter FACTOR and not to the',
        'deal - a Valuation Configuration declaring one that differs from the fitted block is refused',
        'by name.'])

    def floating_dates(self):
        """The dates a swap leg pays on - none here, the swap version declaring the leg."""
        return set()

    def reset(self, calendars):
        super(QEDI_CustomAutoCallSwap, self).reset()
        coupondates = set([x[0] for x in self.field['Autocall_Coupons']])
        self.add_reval_dates(coupondates.union(self.floating_dates()), self.payoff_ccy)

    def coupon_windows(self, fixings, coupons):
        """Which fixings each coupon is observed on: the observations, and each coupon's LAST one.

        DECLARED by `Coupon_Observations`, one row per coupon, which is the whole pairing where it
        is authored. Otherwise DERIVED: a coupon's window is every fixing after its predecessor's
        date up to and including its own (`pricing.oss_window_ends`), and where every window after
        the first holds exactly ONE fixing the deal fixes once per coupon, so the fixings before
        the first coupon's own observation are its strike observation and no coupon's and are
        dropped. Fixings that do not partition into one non-empty window per coupon are refused.
        """
        declared = dict(self.field.get('Coupon_Observations', []))
        if declared:
            ref = self.field.get('Reference', 'this QEDI_CustomAutoCallSwap')
            authored = {x[0] for x in self.field['Price_Fixing']}
            named = [declared.get(c) for c in coupons]
            for coupon, observed in zip(coupons, named):
                if observed is None:
                    raise utils.UnpriceableSchedule(
                        '{}: Coupon_Observations names no fixing for the coupon dated {:%Y-%m-%d}. '
                        'The table IS the pairing where it is authored - one row per coupon - so a '
                        'coupon it leaves out has no observation at all. Name that coupon\'s '
                        'fixing, or delete the table and let the schedule derive every window'
                        .format(ref, coupon))
                if observed not in authored:
                    raise utils.UnpriceableSchedule(
                        '{}: Coupon_Observations dates the coupon of {:%Y-%m-%d} on a fixing of '
                        '{:%Y-%m-%d}, which is not a row of Price_Fixing - an observation the deal '
                        'never books is not a level. Add that fixing, or name one that is booked'
                        .format(ref, coupon, observed))
                if observed > coupon:
                    raise utils.UnpriceableSchedule(
                        '{}: Coupon_Observations dates the coupon of {:%Y-%m-%d} on a fixing of '
                        '{:%Y-%m-%d}, which falls AFTER it - a coupon is decided on a level the '
                        'market has already printed. Name a fixing on or before the coupon'
                        .format(ref, coupon, observed))
            if any(b <= a for a, b in zip(named, named[1:])):
                raise utils.UnpriceableSchedule(
                    '{}: Coupon_Observations pairs the coupons {} with the fixings {}, which do '
                    'not run in the same order - a fixing named by two coupons, or a pairing that '
                    'crosses. Each coupon observes a fixing of its own, later than the one before '
                    'it'.format(ref, [x.strftime('%Y-%m-%d') for x in coupons],
                                [x.strftime('%Y-%m-%d') for x in named]))
            for previous, coupon, observed in zip(coupons, coupons[1:], named[1:]):
                if observed <= previous:
                    raise utils.UnpriceableSchedule(
                        '{}: Coupon_Observations dates the coupon of {:%Y-%m-%d} on a fixing of '
                        '{:%Y-%m-%d}, on or before the coupon of {:%Y-%m-%d} - a level the coupon '
                        'before it has already decided on. Name a fixing after that coupon'
                        .format(ref, coupon, observed, previous))
            return named, np.arange(len(coupons))

        ends = pricing.oss_window_ends(fixings, coupons)
        if ends is None:
            raise utils.UnpriceableSchedule(
                '{}: the price fixings {} do not give each coupon {} a window of its own - every '
                'coupon decides on at least one fixing after the coupon before it, and no fixing '
                'falls after the last coupon. Add the fixing a coupon observes, or delete the one '
                'no coupon reads'.format(
                    self.field.get('Reference', 'this QEDI_CustomAutoCallSwap'),
                    [x.strftime('%Y-%m-%d') for x in fixings],
                    [x.strftime('%Y-%m-%d') for x in coupons]))
        step = np.diff(ends)
        return ((fixings[ends[0]:], ends - ends[0]) if step.size and (step == 1).all()
                else (fixings, ends))

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        """Resolve the autocall's factor dependencies, date lookups and pricing-model switches.

        The non-GBM spot model resolves by NAMING CONVENTION off the equity underlying, with no
        deal field: <SpotModel>ModelParameters.<equity>; off or absent gives None (GBM). A QUANTO
        payoff prices under it - the measure change is a drift the walk carries per step off its own variance
        budget - but a COMPO raises, being the product S*X the arm has no second asset for, and so
        does a quanto naming no `Correlation` factor, an unauthored pair reading as uncorrelated
        and pricing with no quanto drift at all. Those refusals land in the deal-skip path, so a
        refusal is an attributable value loss rather than a wrong number.

        THE FIXING-TO-COUPON PAIRING is `coupon_windows`, declared or derived from the schedule
        with no free parameter: which fixings are observations, and which coupon's trigger level
        each fixing's vol strip is read at. Only the PAST coupons filter the fixings - a coupon
        before the base date closes every window it opened - and the first of them whose window
        printed at or above its trigger (`pricing.autocall_called`) called the note, which raises
        `InstrumentExpired` naming that date, the skip of an expired deal. A BARRIER DATE IS A
        FIXING: dated on a coupon it names that coupon, dated on a fixing - printed or not - it
        names the coupon whose window holds it, and either way it is observed on the window and
        settled on the coupon date.

        THE ZERO-COUPON ROW IS THE FATAL REFUSAL (`UnpriceableSchedule`, per
        `utils.is_fatal_pricing_error`): it says the DOCUMENT is wrong, not that the engine cannot
        reach it. So are fixings that do not give each coupon a window of its own, a barrier dated
        on no coupon and no fixing, an `'Average'` barrier date whose coupon window holds no fixing
        - the mean it would read is a mean of nothing - a `Coupon_Observations` table that does not
        pair, and a `BarrierIsHit`, the put's state being its barrier date's print. The
        max(all_dates) <= Expiry_Date warning is currently DISABLED."""
        refuse_consequence_field(self.field, 'BarrierIsHit', 'QEDI_CustomAutoCallSwap', (
            'date the put\'s barrier on the fixing or coupon it observes and record that close in '
            'Price_Fixing'))
        field = {
            'Currency': utils.check_rate_name(self.field['Currency']),
            'Payoff_Currency': utils.check_rate_name(self.payoff_ccy),
            'Equity': utils.check_rate_name(self.field['Equity']),
            'Equity_Volatility': utils.check_rate_name(self.field['Equity_Volatility'])
        }

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['Dividends'] = self.check_factor_name('Dividends', field['Equity'][:1])

        coupon_dates = [x[0] for x in self.field['Autocall_Coupons']]
        fixing_dates = [x[0] for x in self.field['Price_Fixing']]

        pf = dict(self.field['Price_Fixing'])
        ac = dict(self.field['Autocall_Coupons'])
        at = dict(self.field['Autocall_Thresholds'])
        floats = self.floating_dates()
        ab = set(self.field.get('Barrier_Dates', []))
        # a threshold row is the coupon row of its own POSITION whatever date it carries - the
        # book dates them on the observation and the repo on the coupon
        tl = {c: at[t] for c, t in zip(ac, at)}
        if min(tl.values()) <= 0.0:
            logging.error(
                'AutoCall has some thresholds <=0 - please map the correct thresholds (defaulting to 1.0)')
            tl = {k: v if v else 1.0 for k, v in tl.items()}

        # A COUPON BEHIND THE BASE DATE THAT CALLED ended the note on its date: its window's prints
        # decide it as the pricer decides an observed coupon, and nothing is left to value
        fixings, ends = self.coupon_windows(sorted(pf), sorted(ac))
        for coupon, window in zip(sorted(x for x in ac if x < base_date), np.split(
                np.array([pf[x] for x in fixings]), ends[:-1] + 1)):
            if pricing.autocall_called(window.sum() / window.size, tl[coupon],
                                       self.field['Strike_Price']):
                raise utils.InstrumentExpired('{}: autocalled on {:%Y-%m-%d}'.format(
                    self.field.get('Reference', 'this QEDI_CustomAutoCallSwap'), coupon))

        # DISABLED: warn when max(all_dates) > self.field['Expiry_Date'] - see docstring

        # EVERY COUPON OWNS A WINDOW of fixings: a window of ONE is the spot at that fixing, and a
        # longer one the arithmetic average (`pv_MC_AutoCallSwap`)
        spot_model = self.options.get('SpotModel', 'None')
        ac_dates = sorted([x for x in ac if x >= base_date])
        prior = [x for x in ac if x < min(ac_dates)]
        pf_dates = sorted([x for x in pf if not prior or x > max(prior)])
        pf_dates, ends = self.coupon_windows(pf_dates, ac_dates)
        # A BARRIER DATE IS A FIXING: observed on its coupon's window, settled on the coupon date.
        # A row dated on a coupon names that coupon; one dated on a fixing names the coupon whose
        # window holds it - printed or not - and its flag lands on the coupon's row
        starts = np.concatenate([[0], ends[:-1] + 1])
        window_of = {f: c for k, c in enumerate(ac_dates) for f in pf_dates[starts[k]:ends[k] + 1]}
        ab = {x if x in coupon_dates else window_of.get(x, x) for x in ab}
        off = sorted(x for x in ab if x >= base_date and x not in coupon_dates)
        if off:
            raise utils.UnpriceableSchedule(
                '{}: the barrier dated {} falls on no coupon date and no price fixing, and a '
                'barrier observes a print. Date it on the coupon or the fixing it observes, or add '
                'that fixing'.format(self.field.get('Reference', 'this QEDI_CustomAutoCallSwap'),
                                     ', '.join(x.strftime('%Y-%m-%d') for x in off)))
        # every fixing is a grid date of its own: a block whose rows straddle one would read its
        # print on the rows before it as well as after
        all_dates = sorted(reduce(set.union, [set(coupon_dates), ab, floats, set(fixing_dates)]))

        if self.field['Barrier_Observation'] == 'Average':
            # AN `Average` BARRIER READS ITS COUPON'S WINDOW, which opens after that coupon's
            # predecessor, and a mean of no fixings is not a level
            for barrier_date in sorted(x for x in ab if x >= base_date):
                opens = max([x for x in coupon_dates if x < barrier_date] or [pd.Timestamp.min])
                if not any(opens < x <= barrier_date for x in fixing_dates):
                    raise utils.UnpriceableSchedule(
                        "{}: Barrier_Observation is 'Average' and no price fixing falls in the "
                        'coupon window the barrier dated {:%Y-%m-%d} is observed in, so the mean '
                        'it reads is a mean of nothing. Move that barrier date onto or after an '
                        "observation, or declare Barrier_Observation: 'Spot', which reads the "
                        "barrier date's own spot".format(
                            self.field.get('Reference', 'this QEDI_CustomAutoCallSwap'),
                            barrier_date))

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'Payoff_Currency': get_fxrate_factor(field['Payoff_Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.payoff_ccy,
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
            'Equity_Zero': get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Dividend_Yield': get_dividend_rate_factor(
                field['Dividends'], static_offsets, stochastic_offsets, all_tenors),
            'Volatility': get_vol_factor('EquityPriceVol',
                                         field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Strike_Price': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Barrier': self.field.get('Barrier', 0.0) * self.field['Strike_Price'],
            'Barrier_Observation': self.field['Barrier_Observation'],
            'Expiry': (self.field['Expiry_Date'] - base_date).days
        }

        # A ZERO-COUPON ROW IS REFUSED, not priced. This arm advances both the spot and the
        # price-fixing index INSIDE the coupon block (`if coup > 0` in pv_MC_AutoCallSwap), so
        # a row quoted zero runs no block and the coupon after it takes this row's interval -
        # 4.41%, or 24.2% where a barrier dated on it then reads a stale spot too.
        for coupon_date in ac_dates:
            if ac[coupon_date] <= 0.0:
                stale = 'the barrier dated on it reads the PREVIOUS fixing\'s spot and ' \
                    if coupon_date in ab else ''
                raise utils.UnpriceableSchedule(
                    '{}: the Autocall_Coupons row dated {:%Y-%m-%d} is quoted {:g}, and a '
                    'coupon of zero is not a coupon - the row runs no coupon block at all, so '
                    '{}the coupon after it takes this row\'s interval in place of its own. '
                    'Author a real coupon on that row, or delete the row'.format(
                        self.field.get('Reference', 'this QEDI_CustomAutoCallSwap'),
                        coupon_date, ac[coupon_date], stale))

        if np.any([k <= base_date and v == 0 for k, v in pf.items() if k in pf_dates]):
            logging.error('AutoCall has past fixing set to 0 - please map the correct fixing')

        field_index.update({
            'Price_Fixing': utils.TensorResets.from_observations(
                base_date, time_grid, [[x, pf[x]] for x in pf_dates]),
            'Coupon_Fixing': utils.TensorResets.from_observations(
                base_date, time_grid, [[x, ac[x]] for x in ac_dates]),
            'Autocall_Thresholds': [tl.get(x, -1) for x in all_dates],
            'Coupon_Windows': ends,
            # every fixing carries the trigger level of the coupon whose window it is in, which
            # is the strike that coupon's digital is read at
            'Fixing_Thresholds': np.repeat([tl[c] for c in ac_dates], np.diff(ends, prepend=-1))
        })

        field_index.update({
            'Fixings': utils.TensorResets.from_observations(
                base_date, time_grid, [[x, pf.get(x, -1)] for x in all_dates]),
            'Barrier_Dates': [1 if x in ab else -1 for x in all_dates],
            # a row of the declared leg's table is a payment date whatever its value
            'Autocall_Floating': [1 if x in floats else -1 for x in all_dates],
            'Autocall_Coupons': [ac.get(x, -1) for x in all_dates]
        })

        self.check_option_data(field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors,
                               equity_fx_correlation)

        # non-GBM spot model, by naming convention off the equity - see docstring
        if spot_model != 'None' and field_index['Check_Payoff_Type']:
            pair = utils.check_fx_name([field['Currency'][0], field['Payoff_Currency'][0]])[1]
            if self.payoff_type != 'Quanto':
                raise ValueError(
                    'SpotModel=%s cannot price Payoff_Type=%s settled in %s: a compo payoff is the '
                    'PRODUCT S*X, so the fx rate has to be simulated JOINTLY with the equity and '
                    'this arm walks the equity alone. The quanto measure change is a drift, which '
                    'the walk carries per step off its own variance; a compo is a second asset, '
                    'which it does not have. Settle the autocall in %s, book it Quanto if the fx '
                    'is fixed, or drop the SpotModel'
                    % (spot_model, self.payoff_type, self.payoff_ccy, self.field['Currency']))
            if field_index['QuantoImpliedCorrelation'] is None:
                raise ValueError(
                    'SpotModel=%s prices Payoff_Type=Quanto off the marked equity/fx correlation '
                    'and Correlation.EquityPrice.%s/FxRate.%s is not in the market data. An '
                    'unauthored pair reads as UNCORRELATED, which is a quanto drift of exactly '
                    'zero rather than a missing one. Author the correlation, settle the autocall '
                    'in %s, or drop the SpotModel'
                    % (spot_model, '.'.join(field['Equity_Volatility']), '.'.join(pair),
                       self.field['Currency']))
        hn = get_spot_model_params_factor(
            spot_model, field['Equity'], all_factors, static_offsets, stochastic_offsets)
        set_spot_model_index(field_index, hn, self.options, self.field.get('Reference'))

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)

        strike = deal_data.Factor_dep['Strike_Price']
        spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)
        forward = utils.calc_eq_forward(
            deal_data.Factor_dep['Equity'], deal_data.Factor_dep['Equity_Zero'],
            deal_data.Factor_dep['Dividend_Yield'], deal_data.Factor_dep['Expiry'], deal_time, shared)
        moneyness = pricing.calc_moneyness(strike * shared.one, spot, forward, deal_data)

        if spot.shape[0] != deal_time.shape[0]:
            # skipped and counted as any deal its pricer cannot value
            raise ValueError('the equity {} is static and the autocall walks a simulated one - '
                             'give it a model'.format(self.field['Equity']))
        return pricing.pv_MC_AutoCallSwap(shared, time_grid, deal_data, spot, moneyness, fx_rep) * fx_rep


class QEDI_CustomAutoCallSwap_V2(QEDI_CustomAutoCallSwap):
    vernacular = 'autocall swap, autocallable swap, phoenix swap'
    observes = Observes('Equity', 'EquityPrice', 'Price_Fixing', 1, monitors=True)
    fields = [ADMIN, PRICING, EQUITYOPTIONBASE, OPTION_STYLE, QEDI_CUSTOMSWAP, QEDI_CUSTOMAUTOCALLSWAP]

    documentation = ('Fx And Equity', [
        'The autocall of `QEDI_CustomAutoCallSwap` written as a swap: the same coupons, put, windows,',
        'valuation options and refusals ([Autocallable Swaps](#autocallable-swaps)), plus a floating leg',
        'the **Buy** side pays on each **Autocall_Floating** date while the deal is alive, in place of an',
        'upfront premium. Two differences from the parent:',
        '',
        '- **Barrier** is ABSOLUTE here: the put barrier is $B=$ **Barrier**, a price level, where the',
        '  parent reads a fraction of **Strike_Price**.',
        '- The floating leg: periods between consecutive **Autocall_Floating** dates, the first starting',
        '  **Reset_Frequency** before the first; each pays $(\\ell_k+m)\\alpha_k$ per unit of **Units**, with',
        '  $\\ell_k$ the simply compounded **Forecast_Rate** forward over the period (ACT/365), $m$ the',
        '  **Floating_Margin** in basis points and $\\alpha_k$ the **Forecast_Rate** curve\'s day count. A',
        '  period started before the base date pays its table value instead, the period\'s amount per unit',
        '  notional with the margin in it.',
        '',
        'The one-step-survival construction for autocallables is that of Alm, Harrach, Harrach and Keller',
        '[@alm2013].'])

    def floating_dates(self):
        return {x[0] for x in self.field['Autocall_Floating']}

    def calc_dependencies(
            self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
        field_index = super(QEDI_CustomAutoCallSwap_V2, self).calc_dependencies(
            base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars)

        field_index['Forward'] = get_interest_factor(
            utils.check_rate_name(self.field['Forecast_Rate']), static_offsets, stochastic_offsets, all_tenors)

        # barrier is now absolute (not relative to strike)
        field_index['Barrier'] = self.field['Barrier']

        daycount = field_index['Forward'][0][utils.FACTOR_INDEX_Daycount]
        floating_pay_dates = [x[0] for x in self.field['Autocall_Floating']]
        # need to add the previous date to calculate the reset
        prev_floating_date = min(floating_pay_dates) - self.field['Reset_Frequency']
        floating_pay_dates = [prev_floating_date] + floating_pay_dates

        cashflows = {'Items':
                         [{'Accrual_Start_Date': start,
                           'Accrual_End_Date': end,
                           'Payment_Date': end,
                           'Accrual_Year_Fraction': daycount((end - start).days),
                           'Notional': 1.0,
                           'Resets': [[
                               start, start, end, daycount((end - start).days),
                               self.field['Reset_Frequency'], 'ACT_365', None, 0.0, 'No',
                               utils.Percent(
                                   100.0 * (reset_val / daycount((end - start).days) -
                                            self.field['Floating_Margin'].amount) if start < base_date else 0.0)
                           ]],
                           'Margin': self.field['Floating_Margin']}
                          for start, end, (_, reset_val) in zip(
                             floating_pay_dates[:-1], floating_pay_dates[1:], self.field['Autocall_Floating'])]
                     }

        field_index['Cashflows'] = utils.TensorCashFlows.float(
            base_date, time_grid, 1.0, cashflows, self.field.get('Reference'))

        return field_index


class TouchLifecycle:
    """What a one-touch registers, equity or FX: its payout on the touch or at expiry, a third
    value refused at construction."""
    #: paid where the barrier WAS touched - a no-touch pays where it never was
    no_touch = False

    @property
    def payment_timing(self):
        """When the payout is paid - at the `Touch` or at `Expiry` - which a no-touch fixes."""
        return self.field['Payment_Timing']

    def __init__(self, params, valuation_options):
        super().__init__(params, valuation_options)
        self.path_dependent = True
        # the pricer's closed-form chain has exactly these two branches, and `reset` /
        # `add_grid_dates` read the field before dependencies run - a third value is a malformed
        # program, refused at construction
        if self.payment_timing not in ('Touch', 'Expiry'):
            raise ValueError('{} Payment_Timing must be Touch or Expiry, not {!r}'.format(
                type(self).__name__, self.payment_timing))

    def reset(self, calendars):
        super().reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.payoff_ccy)

    def pays_on_grid(self):
        return self.payment_timing == 'Touch'

    def barrier_monitoring(self, base_date):
        """The Broadie-Glasserman-Kou shift's spacing term: `Barrier_Dates`' mean gap where a type
        declares and states them, else `Barrier_Monitoring_Frequency` (`0M` continuous)."""
        # NOT a monitoring schedule the pricer walks: the touch is monitored CONTINUOUSLY and the
        # dates only estimate the spacing the continuity correction is sized on
        dates = self.field.get('Barrier_Dates', [])
        days = np.mean([x.days for x in np.diff(sorted(set(dates)))]) if dates else             (base_date + self.field['Barrier_Monitoring_Frequency'] - base_date).days
        return 0.5826 * np.sqrt(days / 365.0)

    def add_reval_date_offset(self, offset, relative_to_settlement=True):
        # a touch already pays on the grid, so it takes no offset dates past its settlements
        if relative_to_settlement and self.payment_timing == 'Touch':
            return
        super().add_reval_date_offset(offset, relative_to_settlement)


class EquityOneTouchOption(TouchLifecycle, Deal):
    vernacular = 'equity one-touch, one touch, touch option'
    fields = [ADMIN, PRICING, EQUITY_TOUCH, TOUCH, own('EquityOneTouchOption', [
        F('Payment_Timing', 'Text', default='Expiry', convention=True, values=['Touch', 'Expiry'])
    ], role='Timing', title='Payment timing')]

    documentation = ('Fx And Equity', [
        'A path dependent Equity Option described [here](#one-touch-and-no-touch-binary-options-and-rebates)',
        '',
        'Monitoring is continuous. **Barrier_Dates**, where given, are not an observation schedule: they',
        'only size the Broadie-Glasserman-Kou spacing $\\delta$ as their mean gap; without them $\\delta$ is',
        '**Barrier_Monitoring_Frequency** (`0M` continuous).',
        '',
        'Under **Payoff_Type** `Compo` the priced and MONITORED process is the product $S\\cdot X$ -',
        'spot at the fx cross, carry including the fx forward\'s own, vol composed of the two legs -',
        'and the barrier is the payoff-currency level on that product it is declared as, with the',
        'local smile read at the translated level $H/F_X(T)$.'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Payoff_Currency': utils.check_rate_name(self.payoff_ccy),
                 'Equity': utils.check_rate_name(self.field['Equity']),
                 'Equity_Volatility': utils.check_rate_name(self.field['Equity_Volatility'])}

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {'Currency': get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleCurrency': self.payoff_ccy,
            'Payoff_Currency': get_fxrate_factor(
                field['Payoff_Currency'], static_offsets, stochastic_offsets),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Equity': utils.declared_spot(get_equity_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets), field['Equity']),
            'Equity_Zero': get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Dividend_Yield': get_dividend_rate_factor(
                field['Equity'][:1], static_offsets, stochastic_offsets, all_tenors),
            'Volatility': get_vol_factor(
                'EquityPriceVol', field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Barrier_Underlying': get_equity_barrier_underlying(field['Equity']),
            'Expiry': (self.field['Expiry_Date'] - base_date).days}

        field_index['Barrier_Monitoring'] = self.barrier_monitoring(base_date)

        self.check_option_data(field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors,
                               equity_fx_correlation)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        nominal = self.field['Cash_Payoff']
        payoff_currency = self.payoff_ccy
        spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)
        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)

        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
        b = torch.squeeze(pricing.deal_carry_rate(
            deal_data.Factor_dep,
            (deal_data.Factor_dep['Equity_Zero'], deal_data.Factor_dep['Dividend_Yield']),
            tau.reshape(-1, 1), deal_time, shared), dim=1)

        spot = utils.spot_on_deal_grid(spot, deal_time, shared)

        pv = pricing.pv_one_touch_option(
            shared, time_grid, deal_data, nominal, spot, b, tau, payoff_currency)

        mtm = pv * fx_rep

        return mtm


class EquityNoTouchOption(EquityOneTouchOption):
    vernacular = 'equity no-touch, no touch'
    fields = [ADMIN, PRICING, EQUITY_TOUCH, TOUCH]

    no_touch = True
    #: paid at expiry alone, where the barrier was never touched
    payment_timing = 'Expiry'

    documentation = ('Fx And Equity', [
        'A path dependent Equity Option described [here](#one-touch-and-no-touch-binary-options-and-rebates),',
        'paying **Cash_Payoff** at expiry where the barrier was never touched - the one-touch paid at',
        'expiry on the same terms is its complement, the two together the discounted payout.'])


class EquityBarrierOption(Deal):
    vernacular = 'equity barrier option, knock-in, knock-out'
    observes = Observes('Equity', 'EquityPrice', 'Barrier_Dates', 1, expires='Expiry_Date',
                        monitors=True)
    fields = [ADMIN, PRICING, EQUITYOPTIONBASE, own('EquityBarrierOption', [
        F('Cash_Rebate', 'Float', default=0.0, convention=True, sized=True),
        F('Units', 'Float', default=0.0, sized=True),
        F('Barrier_Dates', 'Table', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Observed', 'Float')])),
        F('Barrier_Monitoring_Frequency', 'Period', default='0M', convention=True),
        F('Barrier_Type', 'Text', default='Down_And_In',
          values=['Down_And_In', 'Down_And_Out', 'Up_And_In', 'Up_And_Out']),
        F('Barrier_Price', 'Float', default=0.0),
        F('Payoff_Type', 'Text', default='Standard', convention=True, values=['Standard', 'Quanto', 'Compo'])
    ])]

    spot_models = ('None', 'LogVar2FJ')

    documentation = ('Fx And Equity', [
        'A barrier option on **Units** shares with two arms, chosen by **Barrier_Dates**:',
        '',
        '- WITH **Barrier_Dates** the barrier is observed on those dates only, priced by One-Step',
        '  Survival (OSS) Monte Carlo ([One-Step Survival Methodology](../theory/mc_simulation.md));',
        '  between observation dates the spot evolves freely and only at barrier dates does OSS',
        '  truncation apply. The rest of this section describes this arm.',
        '- WITHOUT them the barrier is continuous, priced by the closed form of',
        '  [Single Barrier Options](#single-barrier-options) on every scenario date (vol floored at 5%,',
        '  Brownian-bridge touch probability, Broadie-Glasserman-Kou shift for',
        '  **Barrier_Monitoring_Frequency**); a non-`None` **SpotModel** is refused by name there.',
        '',
        '**Observed history**',
        '',
        'A `Barrier_Dates` row is `[date, Observed]`, the close that date fixed at, blank until it',
        'is known. A row on or before the base date with a blank `Observed` is REFUSED: the absence',
        'of a fixing is not the absence of a hit. Any observed close crossing the level resolves the',
        'deal at compile - a knocked-in option compiles as the plain `EquityOptionDeal` it now is,',
        'and a knocked-out one as the `FixedCashflowDeal` paying its rebate on the crossing date.',
        'There is no hit flag to declare; the state is derived from the rows.',
        '',
        'The two readings differ on a level touched exactly: an `Observed` close EQUAL to the barrier',
        'counts as a crossing at the compile-time fold, while on a simulated path (the outer hit mask',
        'and the OSS step alike) a spot exactly on the barrier survives - crossing there is strict.',
        '',
        '**Knock-Out**',
        '',
        'At each barrier observation date $t_j$, the weight $(1-p_j)L_{j-1}$ crosses the barrier',
        'and pays the cash rebate $R$ discounted to $t_j$. Surviving paths contribute the vanilla',
        'payoff at expiry:',
        '',
        '$$V_{\\text{KO}} = \\sum_{j \\in \\mathcal{B}} (1-p_j)\\,L_{j-1}\\,R\\,D_j + L_T\\,g(S_T)\\,D_T$$',
        '',
        '**Knock-In** (via in-out parity)',
        '',
        'The exact identity $V_{\\text{KI}} = V_{\\text{vanilla}} - V_{\\text{KO,pure}} + R\\,E[L_T D_T]$',
        'is used. $V_{\\text{vanilla}}$ is the declared model\'s closed form, evaluated at each',
        'reporting row and constant across inner paths; $V_{\\text{KO,pure}}$ is the',
        'survival-weighted MC estimate. The per-path estimator is therefore:',
        '',
        '$$P = V_{\\text{vanilla}} - L_T\\,D_T\\bigl(g(S_T) - R\\bigr)$$',
        '',
        'The sample mean is clamped to $\\geq 0$ to enforce no-arbitrage.',
        '',
        '**CVA / XVA Path-State Tracking**',
        '',
        'In the outer CVA simulation, each outer scenario spot is compared against the barrier at',
        'every past observation date to build a per-row, per-scenario hit mask. Once a scenario has',
        'hit the barrier, subsequent MTM values are replaced analytically:',
        '',
        '| Barrier Type | Hit MTM |',
        '|---|---|',
        '| Knock-Out | 0 |',
        '| Knock-In | vanilla at the current outer spot, in the declared model |',
        '',
        'That leg and the in-out-parity leg above value the *same* European on the *same* state, so',
        'they take the same model and the same forward - one `total_log_forward` over the remaining',
        'fixing strip.',
        '',
        'When *all* outer scenarios have resolved, the inner OSS simulation is skipped entirely and',
        'that closed form is the whole reported PV.',
        '',
        '**Valuation options** (set in the Valuation Configuration section, per deal type)',
        '',
        '- **SpotModel**: `None` (default — lognormal dynamics off the implied vol surface)',
        'or `LogVar2FJ`, which drives the OSS simulation and walks its own INTERNAL step,',
        'giving each observation interval one Gaussian block law - exact where the dates lie on',
        'that grid, as a daily-monitored barrier does - and prices the barrier-hit and',
        'in-out-parity legs as its conditional Black over that walk rather than Black-Scholes.',
        'Parameters are resolved by naming convention from the',
        '`<SpotModel>ModelParameters.<underlying>` price factor (e.g.',
        '`LogVar2FJModelParameters.SPX`). Switching the model on without that factor in the',
        'market data is a loud skip, never a silent lognormal fallback. Requires a',
        'single-currency payoff: a Quanto/Compo carry is a lognormal quantity, so declaring',
        'one alongside a non-`None` SpotModel is the same loud skip.',
        '- **Steps_Per_Year**: the trading-day clock belongs to the parameter FACTOR and not',
        'to the deal - a Valuation Configuration declaring one that differs from the',
        'fitted block is refused by name.',
        '',
        '**Payoff_Type Compo**',
        '',
        'A composite payoff is written on $S\\cdot X$, so that product is what is priced AND',
        'monitored on both arms: spot at the fx cross, forward at the outright fx forward, carry',
        'the local one plus the fx forward\'s own, vol the composition of the two legs, and',
        'strike and barrier the payoff-currency levels on $S\\cdot X$ they are declared as. On the',
        'continuously monitored arm the crossing test, the Brownian bridge and the',
        'Broadie-Glasserman-Kou shift all read that composite spot and the local smile is read',
        'at the translated strike $K/F_X(T)$, the local strike a payoff-currency strike on',
        '$S\\cdot X$ is worth; the bridge composes the two factors\' published variance rates',
        'with the implied correlation and observes endpoints where either publishes none.'])

    def __init__(self, params, valuation_options):
        super(EquityBarrierOption, self).__init__(params, valuation_options)
        self.path_dependent = True

    def resolve_history(self, base_date, calendars, valuation_options):
        return resolve_discrete_barrier(
            self, base_date, calendars, valuation_options, 'EquityOptionDeal')

    def reset(self, calendars):
        super(EquityBarrierOption, self).reset()
        if self.field.get('Barrier_Dates', []):
            barrierdates = {d for d, _ in barrier_monitoring_rows(self.field['Barrier_Dates'])}
            self.add_reval_dates(barrierdates.union({self.field['Expiry_Date']}), self.payoff_ccy)
        else:
            self.add_reval_dates({self.field['Expiry_Date']}, self.payoff_ccy)

    def pays_on_grid(self):
        # a cash rebate is paid on touch if the option knocks out
        return bool(self.field['Cash_Rebate'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        """Resolve the barrier option's factor dependencies and pricing-model switches.

        The non-GBM spot model resolves by NAMING CONVENTION off the equity underlying, with no
        deal field: <SpotModel>ModelParameters.<equity>; off or absent gives None (GBM). Only the
        DISCRETE (Barrier_Dates) OSS pricer carries the non-GBM branch, hence the raise; a
        Quanto/Compo payoff raises too - a compo is the product S*X the arm has no second asset
        for, and the walk's quanto drift is passed a loading by the autocall alone. Both refusals
        land in the deal-skip path, so a refusal is an attributable value loss rather than a wrong
        number."""

        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Equity': utils.check_rate_name(self.field['Equity']),
                 'Equity_Volatility': utils.check_rate_name(self.field['Equity_Volatility']),
                 'Payoff_Currency': utils.check_rate_name(self.payoff_ccy)}

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['Dividends'] = self.check_factor_name('Dividends', field['Equity'][:1])

        field_index = {'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'Payoff_Currency': get_fxrate_factor(
                           field['Payoff_Currency'], static_offsets, stochastic_offsets),
                       'SettleCurrency': self.payoff_ccy,
                       'Discount': get_interest_factor(
                           field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
                       'Equity': utils.declared_spot(get_equity_rate_factor(
                           field['Equity'], static_offsets, stochastic_offsets), field['Equity']),
                       # names the factor the barrier is monitored ON, so the pricer can ask what
                       # the simulation did between grid dates rather than only at them
                       'Barrier_Underlying': get_equity_barrier_underlying(field['Equity']),
                       'Equity_Zero': get_equity_zero_rate_factor(
                           field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Dividend_Yield': get_dividend_rate_factor(
                           field['Dividends'], static_offsets, stochastic_offsets, all_tenors),
                       'Volatility': get_vol_factor('EquityPriceVol',
                                                    field['Equity_Volatility'], static_offsets, stochastic_offsets,
                                                    all_tenors),
                       'Expiry': (self.field['Expiry_Date'] - base_date).days,
                       'Strike_Price': self.field['Strike_Price'],
                       'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
                       'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0
                       }

        if self.field.get('Barrier_Dates', []):
            ab = {d for d, _ in barrier_monitoring_rows(self.field['Barrier_Dates'])}
            all_dates = sorted(ab.union([self.field['Expiry_Date']]))
            field_index['Observation_Dates'] = utils.TensorResets.from_observations(
                base_date, time_grid, [[x, 0] for x in all_dates])
            field_index['Barrier_Dates'] = [1 if x in ab else -1 for x in all_dates]
        else:
            field_index['Barrier_Monitoring'] = 0.5826 * np.sqrt(
                (base_date + self.field['Barrier_Monitoring_Frequency'] - base_date).days / 365.0)

        self.check_option_data(field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors,
                               equity_fx_correlation)

        # non-GBM spot model, by naming convention off the equity - see docstring
        spot_model = self.options.get('SpotModel', 'None')
        if spot_model != 'None' and 'Barrier_Dates' not in field_index:
            raise ValueError('SpotModel=%s requires the discrete (Barrier_Dates) barrier; the '
                             'continuous Barrier_Monitoring variant has no non-GBM path' % spot_model)
        if spot_model != 'None' and field_index['Check_Payoff_Type']:
            raise ValueError('SpotModel=%s cannot price Payoff_Type=%s settled in %s. A COMPO is '
                             'the product S*X and this arm walks the equity alone; a QUANTO is a '
                             'per-step drift the walk does carry, but only the autocall passes it '
                             'the loading today (`pricing.quanto_step_loading`) - settle the '
                             'barrier in %s or drop the SpotModel'
                             % (spot_model, self.payoff_type, self.payoff_ccy,
                                self.field['Currency']))
        hn = get_spot_model_params_factor(
            spot_model, field['Equity'], all_factors, static_offsets, stochastic_offsets)
        set_spot_model_index(field_index, hn, self.options, self.field.get('Reference'))

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        nominal = self.field['Units']
        payoff_currency = self.payoff_ccy
        spot = utils.calc_time_grid_spot_rate(deal_data.Factor_dep['Equity'], deal_time, shared)
        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)

        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
        b = torch.squeeze(pricing.deal_carry_rate(
            deal_data.Factor_dep,
            (deal_data.Factor_dep['Equity_Zero'], deal_data.Factor_dep['Dividend_Yield']),
            tau.reshape(-1, 1), deal_time, shared), dim=1)

        spot = utils.spot_on_deal_grid(spot, deal_time, shared)

        if 'Barrier_Dates' in deal_data.Factor_dep:
            pv = pricing.pv_discrete_barrier_option(
                shared, time_grid, deal_data, spot, b, tau, fx_rep, isdigital=False)
        else:
            pv = pricing.pv_barrier_option(
                shared, time_grid, deal_data, nominal, spot, b, tau, payoff_currency)

        mtm = pv * fx_rep

        return mtm


class CommodityDigitalOption(Deal):
    vernacular = 'commodity digital, energy digital, binary option on a reference price'
    observes = Observes('Reference_Type', 'ReferencePrice', 'Price_Fixing', 1, expires='Expiry_Date')
    fields = [ADMIN, PRICING, own('CommodityDigitalOption', [
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Option_Type', 'Text', default='Call', values=['Call', 'Put']),
        F('Expiry_Date', 'Date', default=''),
        F('Settlement_Date', 'Date', default='', convention=True,
          settles=Cash('Currency', otherwise='Expiry_Date')),
        F('Strike_Price', 'Float', default=0.0),
        F('Payoff', 'Float', default=REQUIRED, sized=True),
        F('Payoff_Style', 'Text', default='Cash', convention=True, values=['Cash', 'Asset']),
        F('Payoff_Type', 'Text', default='Standard', convention=True, values=['Standard', 'Quanto', 'Compo']),
        F('Reference_Type', 'Text', default='', factors=['ReferencePrice']),
        F('Reference_Volatility', 'Text', default='', factors=['CommodityPriceVol']),
        EXPIRY_PRINT
    ])]

    documentation = ('Energy', [
        'A European digital on a commodity reference price, described',
        '[here](./definitions.md#european-options). In the money at expiry it pays **Payoff**: that',
        'amount in **Currency** under **Payoff_Style** `Cash` (cash-or-nothing), that many units of',
        'the reference under `Asset` (asset-or-nothing).',
        '',
        'The underlying is the forward of the **Reference_Type** for the delivery date its fixing',
        'curve maps **Expiry_Date** to, read off the ForwardPrice curve on every scenario date, and',
        'its vol is the **Reference_Volatility** surface at the forward\'s moneyness over the expiry.',
        'The payoff is paid on **Settlement_Date**, the expiry where blank. Past the expiry the',
        'underlying is what the expiry fixed: the path\'s own forward on the expiry row or, where the',
        'expiry is behind the base date, the print **Price_Fixing** states for it, which it must.',
        '',
        'Paid in a **Currency** other than the forward curve\'s own, **Payoff_Type** says how, as it',
        'does for an equity: `Compo` prices the product $S\\cdot X$ - the strike a payoff-currency',
        'level, the local smile read at the translated strike $K/F_X(T)$ and the vol composed with',
        'the pair\'s - `Quanto` carries the local forward by $-\\rho\\sigma_S\\sigma_X$ and pays at one,',
        'and `Standard`, what omission means, pays the local digital\'s value in **Currency** at one',
        'with no adjustment. The pair\'s `FXVol` and the correlation',
        '`Correlation.FxRate.<local>.<payoff>/ReferencePrice.<reference>.<local>` are the ones the',
        'energy family declares.',
        '',
        'If the **Relative_Digital_Spread** Valuation Configuration option is set (> 0), the',
        'digital is priced as a call/put spread of width `Strike * Relative_Digital_Spread`',
        'either side of the strike, rather than the single-vol closed form, so the vol surface',
        'smile is picked up automatically.'])

    option_defaults = {'Relative_Digital_Spread': 0.0}

    def reset(self, calendars):
        super(CommodityDigitalOption, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']})
        self.add_settlement_dates()

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        refuse_zero_payoff(self.field, 'CommodityDigitalOption', fieldname='Payoff')
        reference_factor, forward_factor = get_reference_factor_objects(
            utils.check_rate_name(self.field['Reference_Type']), all_factors)
        # the option seam's roles: `Currency` the forward curve's own, `Payoff_Currency` the deal's
        field = {
            'Currency': forward_factor.get_currency(),
            'Payoff_Currency': utils.check_rate_name(self.field['Currency']),
            'Reference_Type': utils.check_rate_name(self.field['Reference_Type']),
            'Reference_Volatility': utils.check_rate_name(self.field['Reference_Volatility'])
            }
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Payoff_Currency'])

        expiry, settlement, _ = option_date_info(self, base_date, calendars)
        field_index = {
            'Currency': get_fxrate_factor(field['Payoff_Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Volatility': get_commodity_vol_factor(
                field['Reference_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Strike_Price': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Expiry': expiry,
            'Settlement': settlement,
            'Expiry_Fixing': expiry_fixing(self.field, base_date, time_grid, 'CommodityDigitalOption'),
            # the forward curve's coordinate of the fixing: the delivery date the reference maps the expiry to
            'Delivery': float(reference_factor.get_fixings((self.field['Expiry_Date'] - utils.excel_offset).days))
            }
        field_index['ForwardPrice'], _, _ = get_forwardprice_factor(
            field['Payoff_Currency'], static_offsets, stochastic_offsets, all_tenors,
            all_factors, reference_factor, forward_factor, base_date)
        self.check_option_data(
            field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors, reference_fx_correlation)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        fx_rep = utils.calc_fx_cross(factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        forward_curve = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], deal_time, shared)
        delivery = np.full((deal_time.shape[0], 1), factor_dep['Delivery'])
        forward = torch.squeeze(forward_curve.gather_weighted_curve(shared, delivery, multiply_by_time=False), dim=1)
        strike = pricing.compo_strike(factor_dep, deal_time, shared, factor_dep['Strike_Price'] * shared.one)
        moneyness = pricing.calc_moneyness(strike, forward, forward, deal_data, use_forward=True)

        # spread legs at the deal's own moneyness convention (see EquityBinaryOption.generate)
        eps = self.options['Relative_Digital_Spread']
        spread = (eps,
                  pricing.calc_moneyness(strike * (1.0 - eps), forward, forward, deal_data, use_forward=True),
                  pricing.calc_moneyness(strike * (1.0 + eps), forward, forward, deal_data, use_forward=True)
                  ) if eps else None

        return pricing.pv_european_option(
            shared, time_grid, deal_data, self.field['Payoff'], moneyness, forward, binary=True,
            digital_spread=spread, payoff_style=self.field['Payoff_Style'].lower()) * fx_rep


class CommodityOneTouchOption(TouchLifecycle, Deal):
    vernacular = 'commodity one-touch, energy one-touch, one touch on a reference price'
    fields = [ADMIN, PRICING, COMMODITY_TOUCH, TOUCH, own('CommodityOneTouchOption', [
        F('Payment_Timing', 'Text', default='Expiry', convention=True, values=['Touch', 'Expiry'])
    ], role='Timing', title='Payment timing')]

    documentation = ('Energy', [
        'A one-touch on a commodity reference price, priced by the closed form and bridge described',
        '[here](./fx_and_equity.md#one-touch-and-no-touch-binary-options-and-rebates) for equity and FX.',
        "The monitored price on each date is the reference's fixing there, the forward curve read at",
        'the delivery its `Fixing_Curve` maps that date to, and its carry to the expiry is',
        "$b = \\ln\\big(F(t,D(T))/F(t,D(t))\\big)/(T-t)$, so the closed form's forward is the forward of",
        "the expiry's delivery. That holds while the delivery stays put (one contract) or moves with the",
        'date (a spot-like reference); a reference that rolls from one contract to another before the',
        'expiry jumps at the roll, which the closed form cannot price, and is refused by name. The',
        'bridge observes endpoints: the forward-price process publishes no bridge variance.',
        '',
        'Paid in another currency than the forward curve\'s, a `Quanto` or `Compo` **Payoff_Type** reads',
        '`Correlation.FxRate.<forward>.<payoff>/ReferencePrice.<reference>.<forward>`, stated local to',
        'payoff and read as stated - the opposite sign to the equity touch\'s sorted-pair spelling.'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        reference_factor, forward_factor = get_reference_factor_objects(
            utils.check_rate_name(self.field['Reference_Type']), all_factors)
        base, expiry = ((d - utils.excel_offset).days for d in (base_date, self.field['Expiry_Date']))
        step = np.diff(reference_factor.get_fixings(np.arange(base, expiry + 1)))
        if step.size and not (np.allclose(step, 0.0) or np.allclose(step, 1.0)):
            raise ValueError(
                '{}: reference {} rolls from one delivery to another before the expiry {}, so the '
                'fixing it watches jumps at the roll - state an option on the contract itself'.format(
                    self.field['Reference'], self.field['Reference_Type'], self.field['Expiry_Date'].date()))
        # the option seam's roles: `Currency` the forward curve's own, `Payoff_Currency` the deal's
        field = {'Currency': forward_factor.get_currency(),
                 'Payoff_Currency': utils.check_rate_name(self.field['Currency']),
                 'Reference_Type': utils.check_rate_name(self.field['Reference_Type']),
                 'Reference_Volatility': utils.check_rate_name(self.field['Reference_Volatility'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Payoff_Currency'])

        curve = reference_factor.param['Fixing_Curve'].array
        field_index = {
            'Payoff_Currency': get_fxrate_factor(field['Payoff_Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_interest_factor(field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Volatility': get_commodity_vol_factor(
                field['Reference_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Barrier_Underlying': None,
            'Barrier_Monitoring': self.barrier_monitoring(base_date),
            'Expiry': expiry - base,
            # the delivery each row's fixing reads, off the days since the base date
            'Fixing_Curve': (curve[:, 0] - base, curve[:, 1]),
            'Delivery': float(reference_factor.get_fixings(expiry))}
        field_index['ForwardPrice'], _, _ = get_forwardprice_factor(
            field['Payoff_Currency'], static_offsets, stochastic_offsets, all_tenors,
            all_factors, reference_factor, forward_factor, base_date)
        self.check_option_data(
            field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors, reference_fx_correlation)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        fx_rep = utils.calc_fx_cross(factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)
        curve = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], deal_time, shared)
        days = deal_time[:, utils.TIME_GRID_MTM]
        observed = np.interp(days, *factor_dep['Fixing_Curve']).reshape(-1, 1)
        # a static forward curve answers one column; the pricer assumes one per scenario
        spot = torch.squeeze(curve.gather_weighted_curve(shared, observed, multiply_by_time=False), dim=1).expand(
            -1, shared.simulation_batch)
        forward = torch.squeeze(curve.gather_weighted_curve(
            shared, np.full_like(observed, factor_dep['Delivery']), multiply_by_time=False), dim=1)
        tau = factor_dep['Expiry'] - days
        years = spot.new(factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](tau)).reshape(-1, 1)
        # the carry growing the watched fixing into the expiry's forward; on the expiry row the two
        # are one read, so its log is 0 over a year fraction held off 0
        b = torch.log(forward / spot) / torch.where(years > 0.0, years, 1.0)

        return pricing.pv_one_touch_option(
            shared, time_grid, deal_data, self.field['Cash_Payoff'], spot, b, tau, self.payoff_ccy) * fx_rep


class CommodityNoTouchOption(CommodityOneTouchOption):
    vernacular = 'commodity no-touch, energy no-touch, no touch on a reference price'
    fields = [ADMIN, PRICING, COMMODITY_TOUCH, TOUCH]

    no_touch = True
    #: paid at expiry alone, where the barrier was never touched
    payment_timing = 'Expiry'

    documentation = ('Energy', [
        'A no-touch on a commodity reference price, the one-touch above paying **Cash_Payoff** at expiry',
        'where the barrier was never touched - the one-touch paid at expiry on the same terms is its',
        'complement, the two together the discounted payout.'])


class CommodityForwardDeal(Deal):
    vernacular = 'commodity forward, metal forward'
    fields = [ADMIN, PRICING, own('CommodityForwardDeal', [
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Forward_Date', 'Date', default='', convention=True),
        F('Maturity_Date', 'Date', default='', settles=Cash('Currency')),
        F('Commodity', 'Text', default='', factors=['CommodityPrice']),
        F('Units', 'Float', default=0.0, sized=True),
        F('Reference_Type', 'Text', default='', factors=['ReferencePrice'])
    ])]

    documentation = ('Energy',
                     ['A commodity funded at its own repo rate, which may differ from the carry implied by',
                      'the forward curve (a physical holding can earn a yield the market does not). No strike',
                      'is exchanged: at **Maturity_Date** $T$ the deal is worth the funded price. With',
                      '$t\'=\\max(t,f)$, $f$ the **Forward_Date** (the base date where blank),',
                      '',
                      '$$V(t)=\\pm N\\,F_R(t,t\')\\,e^{r_c(t,\\,T-t\')\\,(T-t\')}\\,D(t,T)$$',
                      '',
                      'where $\\pm$ is the **Buy_Sell** sign, $N$ **Units**, $F_R(t,t\')$ the **Reference_Type**',
                      'forward price for delivery $t\'$ read off its ForwardPrice curve on the scenario date,',
                      '$r_c$ the zero rate of the **Commodity** price factor\'s repo curve over $T-t\'$, and',
                      '$D(t,T)$ the discount factor on **Discount_Rate** (the **Currency** curve where blank).'
                      ])

    def reset(self, calendars):
        super(CommodityForwardDeal, self).reset()
        self.add_reval_dates({self.field['Maturity_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Commodity': utils.check_rate_name(self.field['Commodity']),
                 'Reference_Type': utils.check_rate_name(self.field['Reference_Type'])}

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'Discount': get_interest_factor(
                           field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
                       'Commodity_Zero': get_commodity_zero_rate_factor(
                           field['Commodity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Expiry': (self.field['Maturity_Date'] - base_date).days}

        reference_factor, forward_factor = get_reference_factor_objects(field['Reference_Type'], all_factors)
        field_index['base_index'] = (base_date - utils.excel_offset).days
        field_index['ForwardPrice'], field_index['ForwardFX'], field_index['CashFX'] = get_forwardprice_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors,
            all_factors, reference_factor, forward_factor, base_date)

        # check if this leg is a forward (else spot)
        field_index['Forward_Date'] = (
                self.field['Forward_Date'] - base_date).days if self.field.get('Forward_Date') else 0

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
        forward_curve = utils.calc_time_grid_curve_rate(factor_dep['ForwardPrice'], deal_time, shared)
        remaining_tenor = np.array([max(x[utils.TIME_GRID_MTM], factor_dep['Forward_Date']) for x in deal_time])
        spot_date_index = (factor_dep['base_index'] + remaining_tenor).reshape(-1, 1)
        energy_spot = forward_curve.gather_weighted_curve(shared, spot_date_index, multiply_by_time=False)
        T_t = factor_dep['Expiry'] - remaining_tenor.reshape(-1, 1)
        curve_grid = utils.calc_time_grid_curve_rate(factor_dep['Commodity_Zero'], deal_time, shared)
        forward = torch.squeeze(energy_spot * torch.exp(curve_grid.gather_weighted_curve(shared, T_t)), dim=1)
        fx_rep = utils.calc_fx_cross(factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        nominal = (1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0) * self.field['Units']

        discount_rates = torch.squeeze(utils.calc_discount_rate(
            discount, (factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1), shared),
            dim=1)

        cash = nominal * forward

        pricing.cash_settle(shared, self.field['Currency'], deal_data.Time_dep.deal_time_grid[-1], cash[-1])

        return cash * discount_rates * fx_rep


class CommodityFutureDeal(Deal):
    vernacular = 'commodity future, futures contract'
    fields = [ADMIN, own('CommodityFutureDeal', [
        F('Commodity', 'Text', default='', factors=['CommodityPrice']),
        F('Maturity_Date', 'Date', default=''),
        F('Repo_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Carry', 'Text', default='', factors=['ForwardRate']),
        F('Currency', 'Text', default='', factors=['FxRate']),
        F('Units', 'Float', default=1.0, convention=True, sized=True,
          description='The number of contracts, one when omitted')
    ])]

    documentation = ('Energy', [
        'A standardised futures contract on a commodity. **Commodity** is the pricing spot — a',
        'plain spot, or a composed name (primary + ObservedBasis chain, e.g.',
        '`PLATINUM_CME.LME_CME`) whose periods are summed by the basis-aware spot lookup.',
        '',
        '$$F(t,T)=S(t)\\exp\\Big(c(t,T)\\,(T-t) + \\int_t^T r(t,u)du\\Big)$$',
        '',
        'where $S(t)$ is the (possibly composed) spot, $r(t,u)$ is the forward repo rate from',
        '**Repo_Rate**, and $c(t,T)$ is the carry rate at the contract\'s expiry date $T$ from the',
        '**Carry** factor (a `ForwardPrice`-shaped factor with absolute-date tenors). Output is',
        '**Units** contracts, converted to the report currency via **Currency**.',
    ])

    def reset(self, calendars):
        super(CommodityFutureDeal, self).reset()
        self.add_reval_dates({self.field['Maturity_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        # Commodity is the (possibly composed) pricing-spot name; get_commodity_rate_factor is
        # basis-aware, so the code is one element for a plain spot or primary + basis chain.
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Carry': utils.check_rate_name(self.field['Carry']),
                 'Repo_Rate': utils.check_rate_name(self.field['Repo_Rate']),
                 'Commodity': utils.check_rate_name(self.field['Commodity'])}

        field_index = {'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'Commodity': get_commodity_rate_factor(
                           field['Commodity'], static_offsets, stochastic_offsets),
                       'Repo': get_interest_factor(
                           field['Repo_Rate'], static_offsets, stochastic_offsets, all_tenors),
                       'Carry': get_forward_rate_factor(
                           field['Carry'], static_offsets, stochastic_offsets, all_tenors),
                       'base_index': (base_date - utils.excel_offset).days,
                       'Expiry': (self.field['Maturity_Date'] - base_date).days}

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        spot = utils.calc_time_grid_spot_rate(factor_dep['Commodity'], deal_time, shared)
        repo = utils.calc_time_grid_curve_rate(factor_dep['Repo'], deal_time, shared)
        carry = utils.calc_time_grid_curve_rate(factor_dep['Carry'], deal_time, shared)
        T_t = factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM].reshape(-1, 1)
        # Carry is a forward-curve-shaped factor (absolute-date tenor); query at the contract's
        # absolute expiry-date Excel offset so the lookup lands on a knot, then multiply by the
        # remaining tenor in years to integrate the rate
        expiry_date_index = np.full_like(T_t, factor_dep['base_index'] + factor_dep['Expiry'])
        carry_rate = carry.gather_weighted_curve(shared, expiry_date_index, multiply_by_time=False)
        T_t_years = shared.one.new_tensor(T_t / utils.DayCount.DAYS_IN_YEAR).unsqueeze(-1)

        fx_rep = utils.calc_fx_cross(
            factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        # cost-of-carry forward on the (possibly composed) spot
        mtm = spot * torch.exp(
            carry_rate * T_t_years + repo.gather_weighted_curve(shared, T_t)).squeeze(1)

        return mtm * fx_rep * self.field['Units']


class CommodityAveragePriceSwapDeal(Deal):
    vernacular = 'average price swap, commodity swap, Asian swap'
    observes = Observes('Commodity', 'CommodityPrice', 'Sampling_Data', 1)
    fields = [ADMIN, PRICING, own('CommodityAveragePriceSwapDeal', [
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Commodity', 'Text', default=REQUIRED, factors=['CommodityPrice']),
        F('Carry', 'Text', default=REQUIRED, factors=['ForwardRate']),
        F('Units', 'Float', default=0.0, sized=True),
        F('Fixed_Price', 'Float', default=0.0),
        F('Settlement_Date', 'Date', default=REQUIRED, settles=Cash('Currency')),
        F('Sampling_Data', 'Table', default=REQUIRED, description='Sampling_Data',
          row=Row([F('Date', 'Date'), F('Price', 'Float'), F('Weight', 'Float')]))
    ])]

    documentation = ('Energy', [
        'A cash-settled average-price swap on a commodity fix. **Commodity** is the sampled spot -',
        'a plain name, or a composed one (primary + `ObservedBasis` chain, e.g.',
        '`PLATINUM_CME.LBMA`) whose periods the basis-aware spot lookup sums.',
        '',
        '$$V(t)=\\omega N D(t,T_{pay})\\Big(\\sum_j w_j \\tilde S(t,T_j) - K\\Big)$$',
        '',
        'where $\\omega=\\pm1$ is **Buy_Sell**, $N$ is **Units**, $K$ is **Fixed_Price** and the',
        'weights come from **Sampling_Data**, normalised. A fixing already observed at $t$ enters',
        'at its realised level - the declared `Price` before the base date, the simulated spot at',
        'that fixing\'s own scenario row after it. One still to come enters as the cost-of-carry',
        'forward on the PRIMARY spot plus the conditional mean of the basis,',
        '',
        '$$\\tilde S(t,T_j)=S(t)e^{c(T_j)(T_j-t)+\\int_t^{T_j}r}+\\mu_t+\\phi^{n_j}(b_t-\\mu_t)$$',
        '',
        'with $c$ from **Carry**, $r$ the primary\'s own repo curve, and $(\\phi,\\mu_t)$ the basis',
        'model\'s declared AR coefficient and published slow mean. $n_j$ is measured in SIMULATION',
        'STEPS, which is what that model applies $\\phi$ to.',
    ])

    def validate(self):
        """Settlement cannot precede the average it settles.

        `calc_dependencies` measures the discount tenor from `Settlement_Date` and `reset` makes it
        the deal grid's last row, so a settlement inside the sampling window discounts the later
        fixings from BEFORE they fix and stops the grid short of them.
        """
        dates = [x[0] for x in self.field.get('Sampling_Data') or []]
        if dates and self.field.get('Settlement_Date') and self.field['Settlement_Date'] < max(dates):
            yield 'Settlement_Date must be on or after the last Sampling_Data date'

    def reset(self, calendars=None):
        super(CommodityAveragePriceSwapDeal, self).reset()
        # the fixings are reval dates so every one of them is a row: the realised half of the
        # average is read AT them, and a grid that steps over a fixing interpolates across it
        self.add_reval_dates({x[0] for x in self.field['Sampling_Data']})
        self.add_reval_dates({self.field['Settlement_Date']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        """Resolve the average-price swap's dependencies.

        `Commodity` may name a composed spot; `get_commodity_rate_factor` is basis-aware, so the
        code is one element for a plain spot and primary + basis chain for a composed one. The
        pricer needs the two halves separately - the whole chain for a realised fixing, the primary
        alone under the forward's exponential - so both slices are filed. The repo curve comes off
        the primary spot while the carry is declared, as for `CommodityFutureDeal`.

        DECLARED LIMITATION: `CurveTenor` CLIPS a query to the knot bracket, so a `Carry` whose last
        knot falls before the last fixing prices that fixing off a flat log-carry, silently.
        """
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Commodity': utils.check_rate_name(self.field['Commodity']),
                 'Carry': utils.check_rate_name(self.field['Carry'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        commodity = get_commodity_rate_factor(field['Commodity'], static_offsets, stochastic_offsets)
        phi, lam, mu = get_observed_basis_decay(commodity, all_factors)

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'Commodity': commodity,
            'Spot': commodity[:1],
            'Basis': commodity[1:],
            'Basis_Phi': phi,
            'Basis_Lambda': lam,
            'Basis_Mu': mu,
            'Repo': get_commodity_zero_rate_factor(
                field['Commodity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Carry': get_forward_rate_factor(
                field['Carry'], static_offsets, stochastic_offsets, all_tenors),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Samples': utils.TensorResets.from_observations(
                base_date, time_grid, self.field['Sampling_Data'], weighted=True),
            'base_index': (base_date - utils.excel_offset).days,
            'Settlement': (self.field['Settlement_Date'] - base_date).days,
            'Strike': self.field['Fixed_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'SettleCurrency': self.field['Currency']
        }

        missing_fixings = [x for x in self.field['Sampling_Data'] if x[0] < base_date and not x[1]]
        if missing_fixings:
            logging.error('Past fixings not defined - please specify fixings for {}'.format(
                ', '.join([str(x[0]) for x in missing_fixings])))

        return field_index

    def leg_descriptors(self, deal_data):
        """`(|Units|, settlement day offset)` - the pair `Bundle`'s utility scale reads, off THIS
        deal's own resolved dependencies rather than a `Cashflows` schedule it does not build.

        The base implementation's `(0.0, None)` default is right for a deal with no schedule and
        wrong for this one: it pays exactly one cashflow, of `Units` ounces, at `Settlement`."""
        return abs(self.field['Units']), float(deal_data.Factor_dep['Settlement'])

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_average_price_swap(shared, time_grid, deal_data)


class EquityForwardDeal(Deal):
    vernacular = 'equity forward, stock forward'
    fields = [ADMIN, PRICING, own('EquityForwardDeal', [
        F('Forward_Price', 'Float', default=0.0),
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Payoff_Type', 'Text', default='Standard', convention=True, values=['Standard', 'Quanto', 'Compo']),
        F('Equity_Volatility', 'Text', default='', convention=True, factors=['EquityPriceVol']),
        F('Maturity_Date', 'Date', default='', settles=Cash('Currency')),
        F('Equity', 'Text', default='', factors=['EquityPrice', 'DividendRate']),
        F('Units', 'Float', default=0.0, sized=True),
        F('Payoff_Currency', 'Text', default='', convention=True, factors=['FxRate'])
    ])]

    documentation = ('Fx And Equity', [
        'Described [here](definitions.md#forwards)',
        '',
        'Under **Payoff_Type** `Compo` the forward is the product $S\\cdot X$\'s, the equity\'s at the',
        'outright fx forward into **Payoff_Currency** and **Forward_Price** a level in it; under `Quanto`',
        'the equity\'s forward is carried by $-\\rho\\sigma_S\\sigma_X$ and paid in **Payoff_Currency** at',
        'one. Either names **Equity_Volatility**, which names the pair\'s correlation.'])

    def validate(self):
        """A Quanto or Compo paid in a Payoff_Currency other than its Currency names its surface."""
        return unnamed_surface(self, 'Payoff_Currency')

    def reset(self, calendars):
        super(EquityForwardDeal, self).reset()
        self.add_reval_dates({self.field['Maturity_Date']}, self.payoff_ccy)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Payoff_Currency': utils.check_rate_name(self.payoff_ccy),
                 'Equity': utils.check_rate_name(self.field['Equity']),
                 'Equity_Volatility': self.check_factor_name('Equity_Volatility')}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'Payoff_Currency': get_fxrate_factor(field['Payoff_Currency'], static_offsets, stochastic_offsets),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Equity': get_equity_rate_factor(field['Equity'], static_offsets, stochastic_offsets),
            'Equity_Zero': get_equity_zero_rate_factor(
                field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Dividend_Yield': get_dividend_rate_factor(
                field['Equity'][:1], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (self.field['Maturity_Date'] - base_date).days
            }

        self.check_option_data(
            field, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors, equity_fx_correlation,
            vol_field='Equity_Volatility')

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
        forward = utils.calc_eq_forward(
            factor_dep['Equity'], factor_dep['Equity_Zero'],
            factor_dep['Dividend_Yield'], factor_dep['Expiry'], deal_time, shared)

        if factor_dep['Check_Payoff_Type']:
            # the compo's forward at the outright fx forward, the quanto's carried
            expiry = discount.code[0][utils.FACTOR_INDEX_Daycount](
                factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
            atm_vol = utils.VolSurface.rate(factor_dep['Volatility'], None, expiry, shared)
            adj = pricing.calc_vol_adjustment(factor_dep, deal_time, expiry, atm_vol, shared)
            forward = adj['s_adj'] * forward * torch.exp(adj['b_adj'] * shared.one.new(expiry.reshape(-1, 1)))

        fx_rep = utils.calc_fx_cross(
            factor_dep['Payoff_Currency'], shared.Report_Currency, deal_time, shared)
        nominal = (1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0) * self.field['Units']

        discount_rates = torch.squeeze(utils.calc_discount_rate(
            discount, (factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM]).reshape(-1, 1), shared),
            dim=1)

        cash = nominal * (forward - self.field['Forward_Price'])

        pricing.cash_settle(shared, self.payoff_ccy, deal_data.Time_dep.deal_time_grid[-1], cash[-1])

        return cash * discount_rates * fx_rep


class CashAccountDeal(Deal):
    vernacular = 'cash account, call account, cash balance'
    fields = [ADMIN, PRICING, own('CashAccountDeal', [
        F('Investment_Horizon', 'Date', default=''),
        F('Units', 'Float', default=0.0, sized=True)
    ])]

    documentation = ('Fx And Equity', [
        '**Units** of cash to **Investment_Horizon**, each interval of the grid accruing at the rate',
        'its start reads off **Discount_Rate**: $V(t_i)=\\text{Units}/\\prod_{j<i}D(t_j,t_{j+1})$.'
        ])

    def reset(self, calendars):
        super(CashAccountDeal, self).reset()
        self.add_reval_dates({self.field['Investment_Horizon']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency'])}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'Discount': get_interest_factor(
                           field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
                       'Expiry': (self.field['Investment_Horizon'] - base_date).days}

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        discount = utils.calc_time_grid_curve_rate(factor_dep['Discount'], deal_time, shared)
        # each interval accrues at the rate its start reads, which is all that is known then
        step = utils.calc_discount_rate(
            discount, np.append(np.diff(deal_time[:, utils.TIME_GRID_MTM]), 0).reshape(-1, 1), shared
            ).squeeze(1)
        discount = torch.cat([torch.ones_like(step[:1]), step[:-1]]).cumprod(0)

        fx_rep = utils.calc_fx_cross(factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        cash = self.field['Units'] / discount
        return cash * fx_rep


class EquityDeal(Deal):
    vernacular = 'shares, stock position, equity holding'
    fields = [ADMIN, own('EquityDeal', [
        F('Equity', 'Text', default='', factors=['EquityPrice']),
        F('Investment_Horizon', 'Date', default=''),
        F('Currency', 'Text', default='', factors=['FxRate']),
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Units', 'Float', default=0.0, sized=True)
    ])]

    documentation = ('Fx And Equity', [
        'A holding of **Units** shares, not a forward: no strike is exchanged and no dividend is paid.',
        'To **Investment_Horizon**, in the reporting currency,',
        '',
        '$$V(t)=\\pm N\\,S(t)\\,X(t)$$',
        '',
        'with $\\pm$ the **Buy_Sell** sign, $N$ **Units**, $S$ the equity price and $X$ the price of',
        '**Currency** in the reporting currency.'])

    def reset(self, calendars):
        super(EquityDeal, self).reset()
        self.add_reval_dates({self.field['Investment_Horizon']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Equity': utils.check_rate_name(self.field['Equity'])}

        field_index = {'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
                       'Equity': utils.declared_spot(get_equity_rate_factor(
                           field['Equity'], static_offsets, stochastic_offsets), field['Equity']),
                       'Expiry': (self.field['Investment_Horizon'] - base_date).days}
        # TODO - Add more detail for dividend payments etc.
        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor_dep = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        spot = utils.calc_time_grid_spot_rate(factor_dep['Equity'], deal_time, shared)
        spot = utils.spot_on_deal_grid(spot, deal_time, shared)
        fx_rep = utils.calc_fx_cross(factor_dep['Currency'], shared.Report_Currency, deal_time, shared)
        nominal = (1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0) * self.field['Units']
        cash = nominal * spot
        return cash * fx_rep


class EquitySwapletListDeal(Deal):
    vernacular = 'equity swap, total return swap, TRS'
    fields = [ADMIN, PRICING, own('EquitySwapletListDeal', [
        F('Accrual_Calendars', 'Text', default='', convention=True),
        F('Settlement_Days', 'Integer', default=0, convention=True),
        F('Equity', 'Text', default='', factors=['EquityPrice', 'DividendRate']),
        F('Equity_Currency', 'Text', default='', factors=['FxRate']),
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Payoff_Type', 'Text', default='Standard', convention=True, values=['Standard', 'Quanto', 'Compo']),
        F('Amount_Type', 'Text', default='Principal', convention=True, description='Amount_Type',
          values=['Principal', 'Shares']),
        F('Cashflows', 'Container',
          sub_fields=[
              F('Items', 'Table', default='null', row=Row([
                  F('Start_Date', 'Date'), F('End_Date', 'Date'), F('Payment_Date', 'Date'),
                  F('Amount', 'Float', sized=True), F('Start_Multiplier', 'Float'), F('End_Multiplier', 'Float'),
                  F('Dividend_Multiplier', 'Float'), F('Known_Start_Price', 'Float'), F('Known_End_Price', 'Float'),
                  F('Known_Start_FX_Rate', 'Float'), F('Known_End_FX_Rate', 'Float'), F('Quanto_FX_Rate', 'Float'),
                  F('Known_Dividend_Sum', 'Float')]))]),
        F('Equity_Volatility', 'Text', default='', convention=True, factors=['EquityPriceVol'])
    ])]

    documentation = ('Fx And Equity', [
        'Described [here](#equity-swaps)',
        '',
        'Paid in a **Currency** other than the equity\'s own, **Payoff_Type** says how: `Compo` prices the',
        'product $S\\cdot X$ - an observed level at the fx rate stated beside it, a forward one at the',
        'outright fx forward - and `Quanto` carries each forward by $-\\rho\\sigma_S\\sigma_X$ and pays at',
        'one. Both are the forward calculation on a price return: a row reinvesting its dividends refuses',
        'by name.'])

    def validate(self):
        """A Quanto or Compo in a Currency other than the Equity_Currency names its surface - the
        compile reads the equity's currency off its price factor, which booking cannot see."""
        return unnamed_surface(self, 'Equity_Currency')

    def reset(self, calendars):
        super(EquitySwapletListDeal, self).reset()
        self.paydates = set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
        self.add_reval_dates(self.paydates, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        # the option seam's roles: `Currency` the equity's own, `Payoff_Currency` the swap's
        paid = utils.check_rate_name(self.field['Currency'])
        equity = get_equity_component(utils.check_rate_name(self.field['Equity']), all_factors)
        field = {'Currency': equity.get_currency(), 'Payoff_Currency': paid,
                 'Discount_Rate': self.check_factor_name('Discount_Rate', paid)}
        return self.equity_swaplets(
            field, self.field['Currency'], self.field['Cashflows']['Items'], self.field['Amount_Type'] == 'Principal',
            self.field['Settlement_Days'], base_date, static_offsets, stochastic_offsets, all_factors, all_tenors,
            time_grid, calendars)

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_equity_leg(shared, time_grid, deal_data)


class EquitySwapLeg(Deal):
    vernacular = 'equity swap leg, total return leg'
    fields = [ADMIN, PRICING, own('EquitySwapLeg', [
        F('Accrual_Calendars', 'Text', default='', convention=True),
        F('Rate_Adjustment_Method', 'Text', default='None', convention=True,
          values=['None', 'Modified_Following', 'Following', 'Preceding', 'Modified_Preceding']),
        F('Dividend_Timing', 'Text', default='Terminal', convention=True, values=['Continuous', 'Terminal']),
        F('Equity', 'Text', default='', factors=['EquityPrice', 'DividendRate']),
        F('Equity_Volatility', 'Text', default='', convention=True, factors=['EquityPriceVol']),
        F('Equity_Known_Prices', 'DateEqualList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Asset Price', 'Float'), F('FX Rate', 'Float')])),
        F('Effective_Date', 'Date', default=''),
        F('First_Coupon_Date', 'Date', default='', convention=True),
        F('Known_Dividends', 'DateEqualList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Value', 'Float')])),
        F('Maturity_Date', 'Date', default=''),
        F('Payment_Calendars', 'Text', default='', convention=True,
          description='The calendar the payment offset is rolled on; blank is the leg\'s own '
                      'Accrual_Calendars, and blank on both is Monday to Friday'),
        F('Payment_Frequency', 'Period', default='3M', convention=True),
        F('Payment_Offset', 'Integer', default=0, convention=True),
        F('Penultimate_Coupon_Date', 'Date', default='', convention=True),
        F('Principal_Fixed_Variable', 'Text', default='Variable', convention=True, values=['Principal', 'Variable']),
        F('Roll_Direction', 'Text', default='Forward', convention=True, values=['Forward', 'Backward']),
        F('Units', 'Float', default=0.0, sized=True),
        F('Include_Dividends', 'Text', default='Yes', convention=True, values=['Yes', 'No']),
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Principal', 'Float', default=0.0, sized=True),
        F('Payoff_Currency', 'Text', default='', convention=True, factors=['FxRate']),
        F('Payoff_Type', 'Text', default='Standard', convention=True, values=['Standard', 'Quanto', 'Compo']),
        F('Reset_Calendars', 'Text', default='', convention=True),
        F('Reset_Offset', 'Integer', default=0, convention=True),
        F('Equity_Currency', 'Text', default='', factors=['FxRate'])
    ])]

    documentation = ('Fx And Equity', [
        'The [equity swaplets](#equity-swaps) of its schedule\'s periods, on **Units** shares where',
        '**Principal_Fixed_Variable** is `Variable` and on **Principal** where it is `Principal`, with $b=1$',
        'where **Include_Dividends** is `Yes`. The periods are the',
        '[coupon schedule](interest_rates.md#coupon-schedules) **Payment_Frequency** and the coupon dates',
        'spell, but **Roll_Direction** `Forward` runs the dates forward from the **Effective_Date**, a short',
        'last period where they do not land on the **Maturity_Date**, and `Backward` back from it; and the',
        'dates roll on the accrual calendar by the business-day convention **Rate_Adjustment_Method** names,',
        '`None` leaving them unrolled. A period fixes its start and end **Reset_Offset** business days of',
        '**Reset_Calendars** before its accrual dates (the accrual calendar where blank, weekdays where both',
        'are). A fixing reads the **Equity_Known_Prices** row stated on its day, else, before the base date,',
        'the latest stated before it, refusing by name where there is none, and on the base date the spot;',
        'a period carries the **Known_Dividends** paid between its start and the base date.',
        '',
        'Paid in a **Payoff_Currency** other than its **Currency**, **Payoff_Type** reads as the swaplet',
        'list\'s, so a compo or quanto leg states **Include_Dividends** `No`.'])

    def validate(self):
        """A Quanto or Compo paid in a Payoff_Currency other than its Currency names its surface."""
        return unnamed_surface(self, 'Payoff_Currency')

    def reset(self, calendars):
        super(EquitySwapLeg, self).reset()
        dates, self.pays = self.coupon_schedule(
            calendars, self.field['Effective_Date'], self.field['Maturity_Date'], self.field['Payment_Frequency'],
            method=self.field['Rate_Adjustment_Method'], forward=self.field['Roll_Direction'] == 'Forward')
        offset = int(self.field['Reset_Offset'])
        resets = calendars.get(self.field['Reset_Calendars'] or self.field['Accrual_Calendars'] or
                               self.field['Payment_Calendars'], {}).get('businessday') or pd.offsets.BDay()
        # no offset is no move: a zero multiple of a business day rolls a holiday forward
        self.fixings = [day - offset * resets for day in dates] if offset else list(dates)
        self.add_reval_dates(set(self.pays), self.payoff_ccy)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Payoff_Currency': utils.check_rate_name(self.payoff_ccy)}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        known = self.field['Equity_Known_Prices'].data if self.field['Equity_Known_Prices'] else {}
        spot = get_equity_spot(utils.check_rate_name(self.field['Equity']), static_offsets, stochastic_offsets,
                               all_factors)
        cross = 1.0 if field['Currency'] == field['Payoff_Currency'] else get_fxrate_spot(
            field['Currency'], static_offsets, stochastic_offsets, all_factors) / get_fxrate_spot(
            field['Payoff_Currency'], static_offsets, stochastic_offsets, all_factors)

        def fixed(day):
            """The price and fx rate a fixing reads: the row stated on its day, else, before the base
            date, the latest stated before it, and from the base date on the spot and the fx cross."""
            if day >= base_date and day not in known:
                return spot, cross
            stated = [date for date in known if date <= day]
            return known[max(stated)] if stated else (None, None)

        items = []
        for start, end, pay in zip(self.fixings[:-1], self.fixings[1:], self.pays):
            (start_price, start_fx), (end_price, end_fx) = fixed(start), fixed(end)
            if start_price is None:
                raise ValueError('Equity_Known_Prices states no price on or before the fixing {:%Y-%m-%d}, '
                                 'before the base date'.format(start))
            items.append({
                'Start_Date': start, 'End_Date': end, 'Payment_Date': pay, 'Start_Multiplier': 1.0,
                'End_Multiplier': 1.0, 'Known_Start_Price': start_price, 'Known_Start_FX_Rate': start_fx,
                'Known_End_Price': end_price, 'Known_End_FX_Rate': end_fx,
                'Known_Dividend_Sum': self.field['Known_Dividends'].sum_range(base_date, start, 0)
                if self.field['Known_Dividends'] else 0.0,
                'Dividend_Multiplier': 1.0 if self.field['Include_Dividends'] == 'Yes' else 0.0,
                'Amount': self.field['Units'] if self.field['Principal_Fixed_Variable'] == 'Variable'
                else self.field['Principal']})
        return self.equity_swaplets(
            field, self.payoff_ccy, items, self.field['Principal_Fixed_Variable'] == 'Principal', 0, base_date,
            static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars)

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_equity_leg(shared, time_grid, deal_data)


class FXOneTouchOption(TouchLifecycle, Deal):
    vernacular = 'FX one-touch, one touch, touch option'
    fields = [ADMIN, PRICING, FX_ADMIN, FX_TOUCH, TOUCH, own('FXOneTouchOption', [
        F('Payment_Timing', 'Text', default='Expiry', convention=True, values=['Touch', 'Expiry'])
    ], role='Timing', title='Payment timing')]

    documentation = ('Fx And Equity', [
        'A path dependent FX Option described [here](#one-touch-and-no-touch-binary-options-and-rebates)'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency']),
                 'FX_Volatility': utils.check_rate_name(self.field['FX_Volatility'])}

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {'Currency': get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_vol_factor('FXVol',
                                         field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Barrier_Underlying': get_cross_barrier_underlying(
                (field['Underlying_Currency'], field['Currency']), stochastic_offsets),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Invert_Moneyness': 1 if field['Currency'][0] == field['FX_Volatility'][0] else 0,
            'Barrier_Monitoring': self.barrier_monitoring(base_date),
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])}

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        nominal = self.field['Cash_Payoff']
        payoff_currency = self.payoff_ccy
        fx_rep = utils.calc_fx_cross(deal_data.Factor_dep['Currency'][0],
                                     shared.Report_Currency, deal_time, shared)

        spot = utils.spot_on_deal_grid(utils.calc_fx_cross(
            deal_data.Factor_dep['Underlying_Currency'][0], deal_data.Factor_dep['Currency'][0],
            deal_time, shared), deal_time, shared)

        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])

        b = torch.squeeze(pricing.deal_carry_rate(
            deal_data.Factor_dep,
            (deal_data.Factor_dep['Currency'][1], deal_data.Factor_dep['Underlying_Currency'][1]),
            tau.reshape(-1, 1), deal_time, shared), dim=1)

        pv = pricing.pv_one_touch_option(
            shared, time_grid, deal_data, nominal, spot, b, tau, payoff_currency,
            invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'], use_forwards=True)

        mtm = pv * fx_rep

        return mtm


class FXNoTouchOption(FXOneTouchOption):
    vernacular = 'FX no-touch, no touch'
    fields = [ADMIN, PRICING, FX_ADMIN, FX_TOUCH, TOUCH]

    no_touch = True
    #: paid at expiry alone, where the barrier was never touched
    payment_timing = 'Expiry'

    documentation = ('Fx And Equity', [
        'A path dependent FX Option described [here](#one-touch-and-no-touch-binary-options-and-rebates),',
        'paying **Cash_Payoff** at expiry where the barrier was never touched - the one-touch paid at',
        'expiry on the same terms is its complement, the two together the discounted payout.'])


class FXBarrierOption(Deal):
    vernacular = 'FX barrier option, knock-in, knock-out'
    fields = [ADMIN, FX_ADMIN, FXOPTIONBASE, own('FXBarrierOption', [
        F('Underlying_Amount', 'Float', default=0.0, sized=True),
        F('Barrier_Monitoring_Frequency', 'Period', default='0M', convention=True),
        F('Payoff_Currency', 'Text', default='', convention=True, factors=['FxRate']),
        F('Barrier_Price', 'Float', default=0.0,
          description=FX_AXIS.format('Barrier price')),
        F('Cash_Rebate', 'Float', default=0.0, convention=True, sized=True),
        F('Barrier_Type', 'Text', default='Down_And_In',
          values=['Down_And_In', 'Down_And_Out', 'Up_And_In', 'Up_And_Out']),
        F('Expiry_Date', 'Date', default='')
    ])]

    documentation = ('Fx And Equity', [
        'A continuously monitored FX barrier option on **Underlying_Amount** units of',
        '**Underlying_Currency**, priced in **Currency** by the closed form of',
        '[Single Barrier Options](#single-barrier-options) on every scenario date: vol floored at 5%,',
        'Brownian-bridge touch probability, Broadie-Glasserman-Kou shift for',
        '**Barrier_Monitoring_Frequency**. **Cash_Rebate** is an absolute amount. A **Payoff_Currency**',
        'other than **Currency** is not quanto-adjusted (a warning is logged).'])

    def __init__(self, params, valuation_options):
        super(FXBarrierOption, self).__init__(params, valuation_options)
        self.path_dependent = True

    def reset(self, calendars):
        super(FXBarrierOption, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']}, self.payoff_ccy)

    def pays_on_grid(self):
        # a cash rebate is paid on touch if the option knocks out
        return bool(self.field['Cash_Rebate'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):

        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Payoff_Currency': utils.check_rate_name(self.payoff_ccy),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency']),
                 'FX_Volatility': utils.check_rate_name(self.field['FX_Volatility'])}

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Payoff_Currency': get_fx_and_zero_rate_factor(
                field['Payoff_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_vol_factor('FXVol',
                                         field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Barrier_Underlying': get_cross_barrier_underlying(
                (field['Underlying_Currency'], field['Currency']), stochastic_offsets),
            'Barrier_Monitoring': 0.5826 * np.sqrt(
                (base_date + self.field['Barrier_Monitoring_Frequency'] - base_date).days / 365.0),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Invert_Moneyness': 1 if field['Currency'][0] == field['FX_Volatility'][0] else 0,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])}

        if field['Payoff_Currency'] != field['Currency']:
            logging.warning("Quanto deal. TODO!!!!!!!")

        # discrete barrier monitoring requires adjusting the barrier by 0.58
        # ( -(scipy.special.zetac(0.5)+1)/np.sqrt(2.0*np.pi) ) * sqrt (monitoring freq)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        nominal = self.field['Underlying_Amount']
        payoff_currency = self.payoff_ccy

        spot = utils.spot_on_deal_grid(utils.calc_fx_cross(
            deal_data.Factor_dep['Underlying_Currency'][0], deal_data.Factor_dep['Currency'][0],
            deal_time, shared), deal_time, shared)

        tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
        b = torch.squeeze(pricing.deal_carry_rate(
            deal_data.Factor_dep,
            (deal_data.Factor_dep['Currency'][1], deal_data.Factor_dep['Underlying_Currency'][1]),
            tau.reshape(-1, 1), deal_time, shared), dim=1)

        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)

        pv = pricing.pv_barrier_option(
            shared, time_grid, deal_data, nominal, spot, b, tau, payoff_currency,
            invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'], use_forwards=True)

        mtm = pv * fx_rep

        return mtm


class FXPartialTimeBarrierOption(Deal):
    vernacular = 'partial barrier, window barrier, partial-time barrier'
    fields = [ADMIN, FX_ADMIN, FXOPTIONBASE, own('FXPartialTimeBarrierOption', [
        F('Expiry_Date', 'Date', default=''),
        F('Underlying_Amount', 'Float', default=0.0, sized=True),
        F('Payoff_Currency', 'Text', default='', convention=True, factors=['FxRate']),
        F('Barrier_Price', 'Float', default=0.0,
          description=FX_AXIS.format('Barrier price')),
        F('Barrier_Type', 'Text', default='Down_And_In',
          values=['Down_And_In', 'Down_And_Out', 'Up_And_In', 'Up_And_Out']),
        F('Barrier_At_Start', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Barrier_Limit_Date', 'Date', default=''),
        F('Barrier_Monitoring_Frequency', 'Period', default='0M', convention=True),
        F('Cash_Rebate', 'Float', default=0.0, convention=True, sized=True)
    ])]

    documentation = ('Fx And Equity', [
        'An FX barrier option monitored over a window only, on **Underlying_Amount** units of',
        '**Underlying_Currency** and priced in **Currency**: specified in',
        '[Partial Barrier Options](#partial-barrier-options).'])

    def __init__(self, params, valuation_options):
        super(FXPartialTimeBarrierOption, self).__init__(params, valuation_options)
        self.path_dependent = True

    def reset(self, calendars):
        super(FXPartialTimeBarrierOption, self).reset()
        self.add_reval_dates({self.field['Expiry_Date'], self.field['Barrier_Limit_Date']}, self.payoff_ccy)

    def pays_on_grid(self):
        # a cash rebate is paid on touch if the option knocks out; an untouched knock-in pays its
        # rebate at expiry, which `reset` already declares
        return bool(self.field['Cash_Rebate']) and 'Out' in self.field['Barrier_Type']

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):

        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency']),
                 'FX_Volatility': utils.check_rate_name(self.field['FX_Volatility'])
                 }

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {'Currency': get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_vol_factor('FXVol',
                                         field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Barrier_Monitoring': 0.5826 * np.sqrt(
                (base_date + self.field['Barrier_Monitoring_Frequency'] - base_date).days / 365.0),
            'Barrier_Underlying': get_cross_barrier_underlying(
                (field['Underlying_Currency'], field['Currency']), stochastic_offsets),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Limit_Date': (self.field['Barrier_Limit_Date'] - base_date).days,
            'Invert_Moneyness': 1 if field['Currency'][0] == field['FX_Volatility'][0] else 0,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])}

        # discrete barrier monitoring requires adjusting the barrier by 0.58
        # ( -(scipy.special.zetac(0.5)+1)/np.sqrt(2.0*np.pi) ) * sqrt (monitoring freq)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        nominal = self.field['Underlying_Amount']
        payoff_currency = self.payoff_ccy

        spot = utils.spot_on_deal_grid(utils.calc_fx_cross(
            deal_data.Factor_dep['Underlying_Currency'][0], deal_data.Factor_dep['Currency'][0],
            deal_time, shared), deal_time, shared)

        # need to adjust if there's just 1 timepoint - i.e. base reval
        if time_grid.mtm_time_grid.size > 1:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
            tau1 = (deal_data.Factor_dep['Limit_Date'] - deal_time[:, utils.TIME_GRID_MTM])[:-1]
        else:
            tau = (deal_data.Factor_dep['Expiry'] - deal_time[:, utils.TIME_GRID_MTM])
            tau1 = (deal_data.Factor_dep['Limit_Date'] - deal_time[:, utils.TIME_GRID_MTM])

        b = torch.squeeze(pricing.deal_carry_rate(
            deal_data.Factor_dep,
            (deal_data.Factor_dep['Currency'][1], deal_data.Factor_dep['Underlying_Currency'][1]),
            tau.reshape(-1, 1), deal_time[:tau.size], shared), dim=1)

        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)

        pv = pricing.pv_partial_barrier_option(
            shared, time_grid, deal_data, nominal, spot, b, tau, tau1, payoff_currency, fx_rep,
            invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'])

        mtm = pv * fx_rep

        return mtm


class FXTARFOptionDeal(Deal):
    vernacular = 'TARF, target redemption forward, target forward'
    observes = Observes(('Underlying_Currency', 'Currency'), 'FxRate', 'TARF_ExpiryDates', 2)
    fields = [ADMIN, FX_ADMIN, FXOPTIONBASE, own('FXTARFOptionDeal', [
        F('Expiry_Date', 'Date', default=''),
        F('Underlying_Amount', 'Float', default=0.0, sized=True),
        F('Settlement_Style', 'Text', default='Physical', convention=True, values=['Physical', 'Cash']),
        F('Option_Style', 'Text', default='European', convention=True, values=['European']),
        F('InvertedTarget', 'Text', default='', convention=True),
        F('LeverageNotional', 'Float', default=0.0, sized=True),
        F('TargetLevel', 'Float', default=0.0),
        # NO `tag`: a tag names the container the WIRE form uses, and this table is read by
        # iterating rows. Tagged `DateEqualList`, the decoder hands over a `utils.DateEqualList`,
        # which is not iterable.
        F('TARF_ExpiryDates', 'Table', default='null', row=Row([
            F('Fixing Date', 'Date'), F('Settlement Date', 'Date', settles=Cash('Currency')), F('Value', 'Float')])),
        F('Barrier', 'Float', default=0.0)
    ])]

    spot_models = ('None', 'LogVar2FJ')

    documentation = (
        'Fx And Equity', [
            'A Monte Carlo priced FX Target-Accrual Redemption Forward (TARF), using the',
            'One-Step Survival (OSS) technique. See [One-Step Survival Methodology](../theory/mc_simulation.md)',
            'for the general OSS theory.',
            '',
            'The TARF knocks out when the accumulated profit to the client reaches a target',
            'level $T^*$. The OSS barrier is the *remaining target* at each fixing.',
            '',
            '**Payoff.** With $c=+1$ for a Call and $-1$ for a Put, $K$ **Strike_Price**, $N_1$',
            '**Underlying_Amount**, $N_2$ **LeverageNotional**, $H$ **Barrier**, $T^*$ **TargetLevel** and',
            '$\\omega=\\pm1$ for **Buy**/**Sell**, fixing $j$ (row of **TARF_ExpiryDates**) accrues',
            '',
            '$$a_j=\\max(c(S_j-K),0)\\quad\\text{or, under a non-blank InvertedTarget,}\\quad a_j=\\max\\big(c(1/K-1/S_j),0\\big)$$',
            '',
            'and pays on its settlement date $\\omega\\big(N_1\\min(a_j,T^*-A_{j-1})-N_2\\,u_j\\,\\mathbf 1_{\\text{KI}}\\big)$, where',
            '$A_j=A_{j-1}+a_j$, $u_j$ is the out-of-the-money amount $\\max(-c(S_j-K),0)$ (its reciprocal form',
            'under **InvertedTarget**) and $\\mathbf 1_{\\text{KI}}$ is $c(H-S_j)\\ge0$ where $H>0$, else 1. The',
            'deal redeems at the first fixing where $A_j\\ge T^*$, that fixing paying exactly the remainder.',
            '$T^*$ is in the accrual\'s own units (price per unit of $N_1$). **Option_Style** and',
            '**Settlement_Style** are accepted and not read. A fixing dated before the base date must carry',
            'its observed rate (refused by name otherwise).',
            '',
            'Let $A_{j-1}$ be the accumulated accrual entering fixing $j$ and $\\Delta_j$ the ITM accrual',
            'for that step. The barrier in the OSS sense is:',
            '',
            '$$H_j = T^* - A_{j-1}$$',
            '',
            'The survival probability $p_j$ is the probability that the step accrual $\\Delta_j < H_j$',
            '(i.e. the TARF does not terminate at fixing $j$). The knocked-out weight',
            '$(1-p_j)L_{j-1}$ contributes the partial accrual needed to exactly reach $T^*$ plus any',
            'OTM payments from the terminated fixing. Surviving paths continue with $A_j = A_{j-1} + \\Delta_j$',
            'drawn from the truncated distribution.',
            '',
            'A SEASONED deal is the same deal with less of it left: a row whose SETTLEMENT date is',
            'behind the base date has paid, so its `Value` accrues into $A_0$ and its dates leave',
            'the grid - the strip prices against $T^* - A_0$ and never pays that row again, and a',
            'deal whose settled rows reach $T^*$ between them is redeemed and worth zero. A row',
            'FIXED but not yet settled is still the strip\'s: it pays in-loop on its `Value` at',
            'survival 1, banking its payment at its own settlement date, and the strip walks on',
            'from the row\'s spot.',
            '',
            'Inner paths: **MCMC_Simulations** per outer scenario and its antithetic mirror. Declared',
            'limitation: under `Branch_And_Weight` the second-order (gamma) estimator\'s variance grows',
            'like $s^{-3}$ in the fixing spacing $s$ - usable at monthly or quarterly fixings, rough at',
            'daily ones; the size is unmeasured here.',
            '',
            '**Valuation options** (set in the Valuation Configuration section, per deal type)',
            '',
            '- **SpotModel**: `None` (default — lognormal dynamics off the implied vol surface)',
            'or `LogVar2FJ`, which drives the OSS fixing-to-fixing simulation, walking its own INTERNAL',
            'step and handing each fixing interval one Gaussian block law that the PnL-cap survival and',
            'the integrated knock-in tail both read. Parameters are resolved by naming convention from',
            'the `<SpotModel>ModelParameters.<pair key>` price factor, which is what the calibration',
            'writes: the pair\'s NON-BASE token where one leg is the book\'s base currency (e.g.',
            '`LogVar2FJModelParameters.EUR` for an EURUSD leg on a USD book), and',
            '`<later>.<earlier>` alphabetically for a CROSS - `LogVar2FJModelParameters.ZAR.EUR`, the',
            'rand priced in the euro, for an EURZAR leg on a USD book - whichever side the deal is',
            'written from. Switching the model on without that',
            'factor in the market data is a loud skip, never a silent lognormal fallback.',
            '- **Steps_Per_Year**: the trading-day clock belongs to the parameter FACTOR and not to',
            'the deal - a Valuation Configuration declaring one that differs from the fitted',
            'block is refused by name.'])

    def reset(self, calendars):
        super(FXTARFOptionDeal, self).reset()
        # TARF dates are Fixing, then settlement
        self.add_reval_dates(
            set([x[1] for x in self.field['TARF_ExpiryDates']]), self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        """Resolve the TARF's factor dependencies and the opt-in spot-model switch.

        A row whose SETTLEMENT is behind the base date is history: it leaves the grid and its
        observed value goes to `Settled_Fixings`, which the pricer folds into the pot the strip
        opens on. What is left is one fixing per settlement, which is what the strip walks.

        The switch swaps the (moneyness, vol-surface) lookup for the model's own walk in the
        pricer, leaving the GBM `field_index['Volatility']` path untouched when off or absent. No
        deal field: the params factor resolves by NAMING CONVENTION off the pair's NON-BASE token,
        which for a TARF forced onto the base is `Currency` rather than `Underlying_Currency`.
        Where the underlying IS the base, `Invert_Spot` carries that one law to this deal's own
        axis and numeraire."""
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency']),
                 'FX_Volatility': utils.check_rate_name(self.field['FX_Volatility'])}

        schedule = sorted(self.field['TARF_ExpiryDates'], key=lambda x: x[0])
        active = [x for x in schedule if x[1] >= base_date]
        if not active:
            raise utils.InstrumentExpired(self.field.get('Reference', 'FXTARFOptionDeal'))
        refuse_unpaired_schedule(self.field, active, 'FXTARFOptionDeal')
        if any(not x[-1] for x in schedule if x[0] < base_date):
            raise utils.UnpriceableSchedule(
                '{}: a TARF fixing whose date has PASSED must carry the rate observed on it - the '
                'schedule is the record of what the deal has already accrued, and a blank row '
                'prices it against a target nothing has been taken out of'.format(
                    self.field.get('Reference', 'FXTARFOptionDeal')))

        pf = {x[0]: (x[-1] if (x[-1] is not None) else 0.0) for x in active}
        pf_dates = [x[0] for x in active]
        sd_dates = [x[1] for x in active]
        all_dates = sorted(set(sd_dates).union(pf_dates))

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_vol_factor(
                'FXVol', field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (self.field['Expiry_Date'] - base_date).days,
            'Invert_Moneyness': field['Currency'][0] == field['FX_Volatility'][0],
            'Strike_Price': self.field['Strike_Price'],
            'Fixings': utils.TensorResets.from_observations(
                base_date, time_grid, [[x, pf.get(x, -1)] for x in all_dates]),
            'Price_Fixings': utils.TensorResets.from_observations(base_date, time_grid, [[x, pf[x]] for x in pf_dates]),
            'Settlement': np.array([(x - base_date).days for x in sd_dates]),
            'Settled_Fixings': [x[-1] for x in schedule if x[1] < base_date],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Notional1': self.field['Underlying_Amount'],
            'Notional2': self.field['LeverageNotional'],
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])
        }

        # opt-in spot model, by naming convention off the pair's key. It resolves only under the
        # switch: it needs the book's base, and a GBM deal is priced on compile paths that never
        # knew one
        spot_model = self.options.get('SpotModel', 'None')
        hn = None if spot_model == 'None' else get_spot_model_params_factor(
            spot_model,
            utils.spot_model_currency(
                field['Underlying_Currency'], field['Currency'], self.base_currency),
            all_factors, static_offsets, stochastic_offsets)
        if hn is not None:
            set_spot_model_index(field_index, hn, self.options, self.field.get('Reference'))
            if spot_model_reciprocal_axis(
                    self.options['SpotModel'], field['Underlying_Currency'], field['Currency'],
                    self.base_currency, self.field.get('Reference')):
                field_index['Invert_Spot'] = True

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        FX_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)

        spot = utils.spot_on_deal_grid(utils.calc_fx_cross(
            deal_data.Factor_dep['Underlying_Currency'][0],
            deal_data.Factor_dep['Currency'][0], deal_time, shared), deal_time, shared)

        mtm = pricing.pv_MC_Tarf(
            shared, time_grid, deal_data, spot, FX_rep) * FX_rep

        return mtm


class FXAccumulatorOptionDeal(Deal):
    vernacular = 'FX accumulator, decumulator, accu'
    observes = Observes(('Underlying_Currency', 'Currency'), 'FxRate', 'Accumulator_ExpiryDates', 2)
    fields = [ADMIN, FX_ADMIN, FXOPTIONBASE, own('FXAccumulatorOptionDeal', [
        F('Underlying_Amount', 'Float', default=0.0, sized=True),
        F('LeverageNotional', 'Float', default=0.0, sized=True),
        F('Barrier_Type', 'Text', default='Up_And_Out', values=['Up_And_Out', 'Down_And_Out']),
        F('Barrier_Price', 'Float', default=REQUIRED,
          description=FX_AXIS.format('Barrier price')),
        # NO `tag`: a tag names the utils container the wire form uses, and a tagged table
        # arrives as that object. This schedule is read by iterating rows.
        F('Accumulator_ExpiryDates', 'Table', default='null', row=Row([
            F('Fixing Date', 'Date'), F('Settlement Date', 'Date', settles=Cash('Currency')), F('Value', 'Float')]))
    ])]

    spot_models = ('None', 'LogVar2FJ')

    documentation = (
        'Fx And Equity', [
            'A fixing-observed knock-out FX accumulator priced by One-Step Survival (OSS) Monte',
            'Carlo. See [One-Step Survival Methodology](../theory/mc_simulation.md) for the theory.',
            '',
            'At each fixing $t_j$, provided the knock-out barrier has not been hit, the signed',
            'payoff is',
            '',
            '$$\\mathrm{BS}\\left[N_1\\max(c(S_j-K),0)-N_2\\max(-c(S_j-K),0)\\right],$$',
            '',
            'paid at that fixing\'s own settlement date, where $c=+1$ for Call and $c=-1$ for Put,',
            '$N_1$ is Underlying_Amount, $N_2$ is LeverageNotional, and BS is +1/-1 for Buy/Sell.',
            '',
            'The barrier is observed on the fixing dates themselves. For Up_And_Out the deal',
            'survives a fixing only when $S_j < H$; for Down_And_Out only when $S_j > H$. The',
            'fixing that breaches pays no cashflow and every later fixing is cancelled; cashflows',
            'accrued before the breach still settle. Survival at a future fixing is integrated',
            'analytically and the surviving spot is drawn from the truncated distribution, so the',
            'knock-out decision stays smooth through the AAD tape; an already-observed fixing is',
            'resolved by its exact indicator.',
            '',
            'Accumulator_ExpiryDates rows are [fixing date, settlement date, observed fixing],',
            'with the observed fixing blank until it is known. The knock-out state is DERIVED',
            'from those rows - a settled fixing whose recorded value breaches the barrier ends',
            'the deal - and there is no flag to declare it with.',
            '',
            '**Valuation options** (set in the Valuation Configuration section, per deal type)',
            '',
            '- **SpotModel**: `None` (default - lognormal dynamics off the implied vol surface) or',
            '`LogVar2FJ`, resolved by naming convention from the',
            '`<SpotModel>ModelParameters.<pair key>` price factor exactly as for the FX TARF - the',
            'non-base token, or `<later>.<earlier>` alphabetically for a cross; it walks',
            'its own INTERNAL step and hands each fixing interval the block law this loop truncates at.',
            '- **Steps_Per_Year**: the trading-day clock belongs to the parameter FACTOR and not to',
            'the deal - a Valuation Configuration declaring one that differs from the fitted',
            'block is refused by name.'])

    def __init__(self, params, valuation_options):
        super(FXAccumulatorOptionDeal, self).__init__(params, valuation_options)
        # `path_dependent` without an `add_grid_dates`: `reset` already puts every fixing and
        # settlement in `reval_dates`. What the flag buys is its other meaning - inside a netting
        # set a path-dependent child keeps its own dates instead of inheriting every grid row
        self.path_dependent = True

    def reset(self, calendars):
        super(FXAccumulatorOptionDeal, self).reset()
        # fixings are state-observation rows; only settlement dates are cashflow dates
        self.reval_dates.update({x[0] for x in self.field['Accumulator_ExpiryDates']})
        self.add_reval_dates(
            {x[1] for x in self.field['Accumulator_ExpiryDates']}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        """Resolve the accumulator's factor dependencies and the opt-in spot-model switch.

        The knock-out state carried in from before the base date is a FOLD over the schedule:
        every settled fixing's own breach test, and nothing else. Unsettled and future fixings are
        the pricer's business. The spot-model switch is the TARF's - see
        FXTARFOptionDeal.calc_dependencies."""
        refuse_consequence_field(self.field, 'Barrier_Hit', 'FXAccumulatorOptionDeal', (
            'record the fixing that knocked it out in Accumulator_ExpiryDates - its date, its '
            'settlement date and the value that breached'))
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency']),
                 'FX_Volatility': utils.check_rate_name(self.field['FX_Volatility'])}

        schedule = sorted(self.field['Accumulator_ExpiryDates'], key=lambda x: x[0])
        active = [x for x in schedule if x[1] >= base_date]
        if not active:
            raise utils.InstrumentExpired(self.field.get('Reference', 'FXAccumulatorOptionDeal'))

        # not defensive: the pricer's block search assumes settlements monotone in fixing order,
        # and a duplicated fixing date would silently collapse in the lookup
        refuse_unpaired_schedule(self.field, active, 'FXAccumulatorOptionDeal')

        pf = {x[0]: (x[-1] if (x[-1] is not None) else 0.0) for x in active}
        fixing_dates = [x[0] for x in active]
        settlement_dates = [x[1] for x in active]
        all_dates = sorted(set(fixing_dates).union(settlement_dates))

        barrier = float(self.field['Barrier_Price'])
        barrier_up = 'Up' in self.field['Barrier_Type']
        # a knock-out barrier at or below zero is unreachable from below and instantly breached
        # from above: an Up_And_Out deal with the field left blank survives NO fixing and prices
        # to a silent scalar zero
        if barrier <= 0.0:
            raise ValueError('{}: Barrier_Price must be positive (got {}); an accumulator with no '
                             'knock-out is authored with the barrier far out of the money'.format(
                self.field.get('Reference', 'FXAccumulatorOptionDeal'), barrier))
        settled_breach = any(
            (x[-1] >= barrier if barrier_up else x[-1] <= barrier)
            for x in schedule if x[1] < base_date and x[-1] is not None)

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_vol_factor('FXVol',
                                         field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (settlement_dates[-1] - base_date).days,
            'Invert_Moneyness': field['Currency'][0] == field['FX_Volatility'][0],
            'Strike_Price': self.field['Strike_Price'],
            'Fixings': utils.TensorResets.from_observations(
                base_date, time_grid, [[x, pf.get(x, -1)] for x in all_dates]),
            'Price_Fixings': utils.TensorResets.from_observations(
                base_date, time_grid, [[x, pf[x]] for x in fixing_dates]),
            'Settlement': np.array([(x - base_date).days for x in settlement_dates]),
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Notional1': self.field['Underlying_Amount'],
            'Notional2': self.field['LeverageNotional'],
            'Barrier_Price': barrier,
            'Barrier_Up': barrier_up,
            'Prefix_Breached': settled_breach,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])
        }

        # opt-in spot model, by naming convention off the pair's key. It resolves only under the
        # switch: it needs the book's base, and a GBM deal is priced on compile paths that never
        # knew one
        spot_model = self.options.get('SpotModel', 'None')
        hn = None if spot_model == 'None' else get_spot_model_params_factor(
            spot_model,
            utils.spot_model_currency(
                field['Underlying_Currency'], field['Currency'], self.base_currency),
            all_factors, static_offsets, stochastic_offsets)
        if hn is not None:
            set_spot_model_index(field_index, hn, self.options, self.field.get('Reference'))
            if spot_model_reciprocal_axis(
                    self.options['SpotModel'], field['Underlying_Currency'], field['Currency'],
                    self.base_currency, self.field.get('Reference')):
                field_index['Invert_Spot'] = True

        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        FX_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)

        spot = utils.spot_on_deal_grid(utils.calc_fx_cross(
            deal_data.Factor_dep['Underlying_Currency'][0],
            deal_data.Factor_dep['Currency'][0], deal_time, shared), deal_time, shared)

        return pricing.pv_MC_Accumulator(
            shared, time_grid, deal_data, spot, FX_rep) * FX_rep


class FXExtendableForwardDeal(Deal):
    vernacular = 'extendable forward, extendible forward'
    observes = Observes(('Underlying_Currency', 'Currency'), 'FxRate', 'Extendable_ExpiryDates', 2)
    fields = [ADMIN, FX_ADMIN, FXOPTIONBASE, own('FXExtendableForwardDeal', [
        F('Extension_Strike', 'Float', default=REQUIRED),
        F('Extension_Date', 'Date', default=REQUIRED),
        F('Extension_Style', 'Text', default='Strip', convention=True, values=['Strip', 'Rolling']),
        # WHOSE right the extension is, from the REPORTED book: 'Bank' is the reporter and
        # 'Counterparty' the mirror booking, where the exerciser optimises against this book;
        # a short position flips it with Buy_Sell.
        F('Exercised_By', 'Text', default='Bank', values=['Bank', 'Counterparty'], side=True),
        F('Underlying_Amount', 'Float', default=0.0, sized=True),
        # NO `tag`, read by iterating rows - the accumulator's schedule states the reason.
        # `Extended` is a lifecycle fact (Yes/No once the exerciser has decided, blank otherwise),
        # never a model output.
        F('Extendable_ExpiryDates', 'Table', default='null', row=Row([
            F('Fixing Date', 'Date'), F('Settlement Date', 'Date', settles=Cash('Currency')), F('Value', 'Float'),
            F('Extended', 'Text')]))
    ])]

    documentation = (
        'Fx And Equity', [
            'An FX extendable forward with periodic fixing cashflows, exercisable by either side.',
            '',
            '**Exercised_By** names whose right the extension is, read from the REPORTED book.',
            '*Bank* (the default) is the reporter: the deal is optimised from this book, as a',
            'bought one and a sold one both are. *Counterparty* is the MIRROR booking - the client',
            'side of a bank-exercisable deal - where the exerciser optimises AGAINST this book, so',
            'the extension is kept when the continuation is worth less than nothing here. A deal',
            'and its mirror (opposite **Buy_Sell**, opposite **Exercised_By**) sum to zero.',
            '',
            'Fixings up to and including **Extension_Date** are guaranteed at **Strike_Price**',
            '(K1). Any fixing created by an extension is struck at **Extension_Strike** (K2). The',
            'fixing on an extension decision date is already in force and therefore pays before',
            'the decision to continue or terminate the remaining schedule is applied.',
            '',
            '**Extension_Style = Strip**: the fixing count must be even and Extension_Date is the',
            'last fixing of the first half. At that date the exerciser either activates the whole',
            'second half at K2 or terminates the deal after the current K1 fixing.',
            '',
            '**Extension_Style = Rolling**: starting at Extension_Date, each fixing carries a',
            'decision whether to create exactly the next fixing at K2. The first decision',
            'not to extend terminates the deal permanently after the current fixing.',
            '',
            '**Extendable_ExpiryDates** rows are [fixing date, settlement date, observed fixing,',
            'observed extension decision]. The decision is a lifecycle fact - and the SAME fact on',
            'both sides of the mirror: Yes/No once taken (mandatory once the fixing is historical),',
            'blank otherwise - future blanks are decided scenario-by-scenario by the exercise rule.',
            '',
            'GBM off the implied surface only, every transition variance read from the smile frozen',
            'at the **Extension_Strike** moneyness (an approximation, unmeasured). The exercise',
            'boundary $H_j$ at each Rolling decision solves $C_j(H_j)=0$ by backward Gauss-Hermite',
            'quadrature of the continuation value. The Strip boundary is closed form, the spot at which',
            'the second half\'s strip of $K_2$ forwards is worth zero: $K_2B/A$ with',
            '$A=\\sum_kD(t_e,T_k)F(t_e,t_k)/S(t_e)$ and $B=\\sum_kD(t_e,T_k)$ over those fixings $t_k$ paid',
            'on $T_k$, $t_e$ the **Extension_Date**. The',
            'forward pass draws only the continuing branch at each decision and weights it by the',
            'analytic continuation probability (One-Step Survival, see',
            '[One-Step Survival Methodology](../theory/mc_simulation.md)).',
            '',
            '**Valuation options** (set in the Valuation Configuration section, per deal type)',
            '',
            '- **Boundary_Grid_Size** / **Boundary_Quadrature** / **Boundary_Batch_Chunk** /',
            '**Boundary_Std_Width** / **Boundary_Min_Log_Width**: resolution dials of the',
            'backward boundary solver (defaults 501 / 32 / 256 / 6.0 / 2.0).'
        ])

    option_defaults = {'Boundary_Grid_Size': 501, 'Boundary_Quadrature': 32,
                       'Boundary_Batch_Chunk': 256, 'Boundary_Std_Width': 6.0,
                       'Boundary_Min_Log_Width': 2.0}

    def __init__(self, params, valuation_options):
        super(FXExtendableForwardDeal, self).__init__(params, valuation_options)
        self.path_dependent = True

    def reset(self, calendars):
        super(FXExtendableForwardDeal, self).reset()
        schedule = self.field['Extendable_ExpiryDates']
        # every fixing is a reval date - a decision is reconstructed at its own fixing row
        self.reval_dates.update({x[0] for x in schedule})
        self.add_reval_dates({x[1] for x in schedule}, self.field['Currency'])

    @staticmethod
    def _decision_code(value):
        """A lifecycle Yes/No, or -1 for not-yet-decided."""
        if value is None or str(value).strip() == '':
            return -1
        text = str(value).strip().lower()
        if text in ('yes', 'no'):
            return int(text == 'yes')
        raise ValueError('Extension decision must be Yes, No or blank, not {!r}'.format(value))

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency']),
                 'FX_Volatility': utils.check_rate_name(self.field['FX_Volatility'])}

        if float(self.field['Strike_Price']) <= 0.0 or float(self.field['Extension_Strike']) <= 0.0:
            raise ValueError('FXExtendableForwardDeal requires positive K1 and K2 strikes')

        style = str(self.field['Extension_Style']).strip().lower()
        exercised_by = str(self.field.read('Exercised_By', 'Bank')).strip()
        if exercised_by not in ('Bank', 'Counterparty'):
            raise ValueError(
                'Exercised_By must be Bank or Counterparty, not {!r} - book the mirror side as '
                'Counterparty and leave it blank for the reporter'.format(exercised_by))
        schedule = sorted(self.field['Extendable_ExpiryDates'], key=lambda x: x[0])
        if len(schedule) < 2:
            raise ValueError('FXExtendableForwardDeal requires at least two fixings')
        fixing_dates = [x[0] for x in schedule]
        settlement_dates = [x[1] for x in schedule]
        if any(b < a for a, b in zip(fixing_dates, settlement_dates)):
            raise ValueError('Extendable forward settlement date must be on or after its fixing date')
        if any(fixing_dates[i] >= fixing_dates[i + 1] for i in range(len(fixing_dates) - 1)):
            raise ValueError('Extendable forward fixing dates must be strictly increasing')
        if any(settlement_dates[i] > settlement_dates[i + 1] for i in range(len(settlement_dates) - 1)):
            raise ValueError('Extendable forward settlement dates must be non-decreasing')
        if settlement_dates[-1] < base_date:
            raise utils.InstrumentExpired(self.field.get('Reference', 'FXExtendableForwardDeal'))

        extension_date = self.field['Extension_Date']
        if extension_date not in fixing_dates:
            raise ValueError('Extension_Date must be one of the fixing dates')
        extension_index = fixing_dates.index(extension_date)
        if extension_index >= len(fixing_dates) - 1:
            raise ValueError('Extension_Date must be before the final fixing')

        if style == 'strip':
            if len(fixing_dates) % 2:
                raise ValueError('Strip style requires an even number of fixings')
            if extension_index != len(fixing_dates) // 2 - 1:
                raise ValueError('Strip style Extension_Date must be the last fixing of the first half')
            decision_indices = [extension_index]
        else:
            decision_indices = list(range(extension_index, len(fixing_dates) - 1))

        fixing_values = [x[2] if len(x) > 2 and x[2] is not None else 0.0 for x in schedule]
        decision_codes = np.full(len(schedule), -1, dtype=np.int8)
        for i, row in enumerate(schedule):
            if len(row) > 3:
                decision_codes[i] = self._decision_code(row[3])
        invalid = [i for i, code in enumerate(decision_codes) if code >= 0 and i not in decision_indices]
        if invalid:
            raise ValueError('Extension decisions supplied on non-decision fixing(s): {}'.format(invalid))
        # a decision is a lifecycle FACT: mandatory once its fixing is historical, refused on a
        # future one, and never demanded after the first No - decisions the termination made moot
        # were never taken
        seen_no = False
        for i in decision_indices:
            if seen_no and decision_codes[i] == 1:
                raise ValueError('A Rolling extension cannot restart after the first No decision')
            if not seen_no and fixing_dates[i] < base_date and decision_codes[i] < 0:
                raise ValueError('Historical extension decision on {} must be supplied'.format(fixing_dates[i]))
            if fixing_dates[i] > base_date and decision_codes[i] >= 0:
                raise ValueError('Extension decision on {} is in the future'.format(fixing_dates[i]))
            if decision_codes[i] == 0:
                seen_no = True

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_vol_factor('FXVol',
                                         field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': (settlement_dates[-1] - base_date).days,
            'Invert_Moneyness': field['Currency'][0] == field['FX_Volatility'][0],
            'Strike_Price': float(self.field['Strike_Price']),
            'Extension_Strike': float(self.field['Extension_Strike']),
            'Fixing_Days': np.array([(x - base_date).days for x in fixing_dates], dtype=np.int64),
            'Price_Fixings': utils.TensorResets.from_observations(
                base_date, time_grid, [[d, v] for d, v in zip(fixing_dates, fixing_values)]),
            'Settlement': np.array([(x - base_date).days for x in settlement_dates], dtype=np.int64),
            'Decision_Codes': decision_codes,
            'Decision_Indices': np.array(decision_indices, dtype=np.int64),
            'Extension_Index': extension_index,
            'Extension_Style': style,
            'Exercise_Sign': 1.0 if exercised_by == 'Bank' else -1.0,
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field.get('Option_Type', 'Call') == 'Call' else -1.0,
            'Notional': float(self.field['Underlying_Amount']),
            'Boundary_Grid_Size': int(self.options['Boundary_Grid_Size']),
            'Boundary_Quadrature': int(self.options['Boundary_Quadrature']),
            'Boundary_Batch_Chunk': int(self.options['Boundary_Batch_Chunk']),
            'Boundary_Std_Width': float(self.options['Boundary_Std_Width']),
            'Boundary_Min_Log_Width': float(self.options['Boundary_Min_Log_Width'])
        }
        return field_index

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
        spot = utils.spot_on_deal_grid(utils.calc_fx_cross(
            deal_data.Factor_dep['Underlying_Currency'][0],
            deal_data.Factor_dep['Currency'][0], deal_time, shared), deal_time, shared)
        return pricing.pv_MC_ExtendableForward(shared, time_grid, deal_data, spot, fx_rep) * fx_rep


class FXOptionDeal(Deal):
    vernacular = 'FX option, currency option, vanilla call or put'
    observes = Observes(('Underlying_Currency', 'Currency'), 'FxRate', 'Price_Fixing', 1,
                        elects='Settlement_Style', expires='Expiry_Date')
    fields = [ADMIN, FX_ADMIN, FXOPTIONBASE, FX_OPTION_STYLE, own('FXOptionDeal', [
        F('Underlying_Amount', 'Float', default=0.0, sized=True),
        F('Expiry_Date', 'Date', default=''),
        F('Forward_Price_Date', 'Date', default='', convention=True),
        EXPIRY_PRINT
    ])]

    documentation = (
        'Fx And Equity', ['A path independent vanilla FX Option on **Underlying_Amount** units of',
                          '**Underlying_Currency**, priced in **Currency** by Black on the FX forward',
                          '(Garman-Kohlhagen [@garman1983]), with the conventions of',
                          '[European Options](./definitions.md#european-options): the forward to expiry plus',
                          'two business days, the vol at the forward\'s moneyness and the expiry, discounted',
                          'on **Discount_Rate**.',
                          '',
                          'The payoff is paid on **Delivery_Date**, the expiry where blank. Past the expiry the',
                          'forward is held at what the expiry fixed: the path\'s own level on the expiry row or,',
                          'where the expiry is behind the base date, the print **Price_Fixing** states for it,',
                          'which it must.',
                          '',
                          '**Option_Style** offers `European` alone, the one style this prices;',
                          '**Settlement_Style**, **Forward_Price_Date** and **Option_On_Forward** do not enter',
                          'the price.'])

    def reset(self, calendars):
        super(FXOptionDeal, self).reset()
        self.add_reval_dates({self.field['Expiry_Date']})
        self.add_settlement_dates()

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Underlying_Currency': utils.check_rate_name(self.field['Underlying_Currency']),
                 'FX_Volatility': utils.check_rate_name(self.field['FX_Volatility'])}

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        expiry, settlement, forward_settlement = option_date_info(self, base_date, calendars)

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Underlying_Currency': get_fx_and_zero_rate_factor(
                field['Underlying_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Volatility': get_vol_factor('FXVol',
                                         field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors),
            'Expiry': expiry,
            'Settlement': settlement,
            'Forward_Settlement': forward_settlement,
            'Expiry_Fixing': expiry_fixing(self.field, base_date, time_grid, 'FXOptionDeal'),
            'Invert_Moneyness': field['Currency'][0] == field['FX_Volatility'][0],
            'Strike_Price': self.field['Strike_Price'],
            'Buy_Sell': 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0,
            'Option_Type': 1.0 if self.field['Option_Type'] == 'Call' else -1.0,
            'Local_Currency': '{0}.{1}'.format(self.field['Underlying_Currency'], self.field['Currency'])
        }

        return field_index

    def generate(self, shared, time_grid, deal_data):
        factor = deal_data.Factor_dep
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]

        fx_rep = utils.calc_fx_cross(
            factor['Currency'][0], shared.Report_Currency, deal_time, shared)
        forward = utils.calc_fx_forward(
            factor['Underlying_Currency'], deal_data.Factor_dep['Currency'],
            factor['Forward_Settlement'], deal_time, shared)
        moneyness = pricing.calc_moneyness(
            factor['Strike_Price'], forward, forward,
            deal_data, use_forward=True, invert_moneyness=deal_data.Factor_dep['Invert_Moneyness'])

        mtm = pricing.pv_european_option(
            shared, time_grid, deal_data, self.field['Underlying_Amount'], moneyness, forward) * fx_rep

        return mtm


class FXEuropeanOption(FXOptionDeal):
    def __init__(self, params, valuation_options):
        super(FXEuropeanOption, self).__init__(params, valuation_options)


class FXBinaryOption(FXOptionDeal):
    vernacular = 'FX digital, binary option, cash-or-nothing, asset-or-nothing'
    observes = Observes(('Underlying_Currency', 'Currency'), 'FxRate', 'Price_Fixing', 1,
                        expires='Expiry_Date')
    fields = [ADMIN, FX_ADMIN, FXOPTIONBASE, own('FXBinaryOption', [
        F('Payoff', 'Float', default=REQUIRED, sized=True),
        F('Payoff_Style', 'Text', default='Cash', convention=True, values=['Cash', 'Asset']),
        F('Settlement_Style', 'Text', default='Physical', convention=True, values=['Physical', 'Cash']),
        F('Expiry_Date', 'Date', default=''),
        EXPIRY_PRINT
    ])]

    documentation = (
        'Fx And Equity', ['A path independent vanilla FX binary (digital) option described'
                          ' [here](./definitions.md#european-options). In the money at expiry it'
                          ' pays **Payoff**: that amount in **Currency** under **Payoff_Style**'
                          ' `Cash` (cash-or-nothing), that many units of **Underlying_Currency**'
                          ' under `Asset` (asset-or-nothing). Unlike `FXOptionDeal`, its forward runs to',
                          'the expiry date itself, with no spot lag.',
                          '',
                          'If the **Relative_Digital_Spread** Valuation Configuration option is set',
                          '(> 0), the digital is priced as a call/put spread of width',
                          '`Strike * Relative_Digital_Spread` either side of the strike, rather than',
                          'the single-vol closed form, so the vol surface smile is picked up',
                          'automatically.'])

    option_defaults = {'Relative_Digital_Spread': 0.0}

    def generate(self, shared, time_grid, deal_data):
        deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
        fx_rep = utils.calc_fx_cross(
            deal_data.Factor_dep['Currency'][0], shared.Report_Currency, deal_time, shared)
        forward = utils.calc_fx_forward(
            deal_data.Factor_dep['Underlying_Currency'], deal_data.Factor_dep['Currency'],
            deal_data.Factor_dep['Expiry'], deal_time, shared)
        invert_moneyness = deal_data.Factor_dep['Invert_Moneyness']
        strike = deal_data.Factor_dep['Strike_Price']
        moneyness = pricing.calc_moneyness(
            strike, forward, forward, deal_data, use_forward=True, invert_moneyness=invert_moneyness)

        # spread legs at the deal's own moneyness convention (see EquityBinaryOption.generate)
        eps = self.options['Relative_Digital_Spread']
        spread = (eps,
                  pricing.calc_moneyness(
                      strike * (1.0 - eps), forward, forward, deal_data,
                      use_forward=True, invert_moneyness=invert_moneyness),
                  pricing.calc_moneyness(
                      strike * (1.0 + eps), forward, forward, deal_data,
                      use_forward=True, invert_moneyness=invert_moneyness)) if eps else None

        mtm = pricing.pv_european_option(
            shared, time_grid, deal_data, self.field['Payoff'], moneyness, forward, binary=True,
            digital_spread=spread, payoff_style=self.field['Payoff_Style'].lower()) * fx_rep

        return mtm


class CreditNthToDefault(Deal):
    vernacular = 'nth-to-default, first-to-default, basket default swap'
    fields = [ADMIN, PRICING, own('CreditNthToDefault', [
        F('Names', 'Table', default='null', row=Row([F('Name', 'Text')]), factors=['SurvivalProb']),
        F('Effective_Date', 'Date', default=''),
        F('Maturity_Date', 'Date', default=''),
        F('Correlation', 'Float', default=0.0, convention=True),
        F('Quadrature_Points', 'Integer', default=81, convention=True,
          description='Gauss-Hermite nodes the copula common-factor integral is taken on'),
        F('Calendars', 'Text', default='', convention=True),
        F('Pay_Frequency', 'Period', default='3M', convention=True),
        F('Pay_Rate', 'Basis', default=0.0),
        F('Max_Defaults', 'Integer', default=3),
        F('Defaults_So_Far', 'Integer', default=0, convention=True),
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Principal', 'Float', default=0.0, sized=True),
        F('Accrual_Day_Count', 'Text', default='ACT_365', convention=True,
          values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', 'ACT_ACT_ICMA'],
          description='Accrual day count, ACT family only'),
        F('Amortisation', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float', sized='magnitude')])),
        F('CDS_Index', 'Text', default='', factors=['SurvivalProb'])
    ])]

    documentation = (
        'Credit',
        ['This is the PREMIUM leg of an $n$-th-to-default basket on a list of reference names: the',
         'buyer pays a periodic coupon that STEPS DOWN as the basket loses names, and pays nothing',
         'once $n$ (*Max_Defaults*) of them have defaulted. Writing $k$ for the number of names',
         'defaulted by time $t$ and $k_0$ for *Defaults_So_Far*, the rate earned is',
         '',
         '$$c\\left(1-\\frac{k_0+k}{n}\\right)1_{\\{k_0+k<n\\}}$$',
         '',
         'where $c$ is *Pay_Rate*. Only at $n=1$, $k_0=0$ does this reduce to $c$ times the',
         'probability that no name has defaulted.',
         '',
         'Names default under a one-factor Gaussian copula [@li2000; @laurent2005] with a flat pairwise',
         '*Correlation* $\\rho$ on marginal default probabilities $1-e^{-\\Lambda_j}$, each name\'s cumulative',
         'hazard over the horizon $h$ scaled by the *CDS_Index* proxy,',
         '',
         '$$\\Lambda_j(t,t+h)=\\Lambda^{\\text{name}}_j(t,t+h)\\,\\frac{\\Lambda_{\\text{idx}}(t,t+h)}{\\Lambda_{\\text{idx}}(0,h)},$$',
         '',
         'the index ratio being the scenario\'s index curve over today\'s for the same horizon.',
         'Conditional on the common factor the names are independent; the factor integral is taken by',
         'Gauss-Hermite quadrature on *Quadrature_Points* nodes (default 81).',
         '',
         'Each period\'s expected accrual $\\int E[\\text{rate}]\\,dt$ is integrated by trapezoid on',
         'hazard samples 30 days apart, from the period\'s own start date to its payment date, and',
         'is measured in *Accrual_Day_Count* year fractions - the discount curve keeps its own day',
         'count. The result is weighted by that period\'s notional, which carries *Principal*, the',
         '*Buy_Sell* sign and any *Amortisation*, and discounted to the valuation date. Note that',
         'the default (protection) leg is not valued here.'])

    def reset(self, calendars):
        super(CreditNthToDefault, self).reset()
        bus_day = calendars.get(self.field['Calendars'], {'businessday': pd.offsets.Day(1)})['businessday']
        if list(self.field['Pay_Frequency'].kwds.values()) == [0]:
            self.resetdates = pd.DatetimeIndex([self.field['Effective_Date'], self.field['Maturity_Date']])
        else:
            self.resetdates = utils.generate_dates_backward(
                self.field['Maturity_Date'], self.field['Effective_Date'],
                self.field['Pay_Frequency'], bus_day=bus_day)
        self.add_reval_dates(self.resetdates, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'CDS_Index': utils.check_rate_name(self.field['CDS_Index']),
                 'Names': [utils.check_rate_name(name) for name in self.field['Names']]}

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        pay_rate = self.field['Pay_Rate'] / 100.0 if isinstance(
            self.field['Pay_Rate'], float) else self.field['Pay_Rate'].amount
        accrual_daycount = utils.DayCount.code(self.field['Accrual_Day_Count'])

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.field['Currency'],
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Name': get_survival_factor(field['CDS_Index'], static_offsets, stochastic_offsets, all_tenors),
            'Names': [get_survival_factor(
                name, static_offsets, stochastic_offsets, all_tenors) for name in field['Names']],
            'Coupon': pay_rate,
            'Correlation': self.field.get('Correlation', 0.0),
            'Max_Defaults': self.field['Max_Defaults'],
            'Defaults_So_Far': self.field.get('Defaults_So_Far', 0),
            'Quadrature_Points': self.field.get('Quadrature_Points', 81),
            # the accrual measure the coupon is quoted against - NOT the discount curve's
            'Accrual_Daycount': partial(utils.DayCount.accrual, base_date, code=accrual_daycount)
        }

        # signed as the coupon moves: the protection buyer PAYS it
        field_index['Cashflows'] = utils.TensorCashFlows.generate_fixed(
            base_date, self.resetdates,
            (-1 if self.field['Buy_Sell'] == 'Buy' else 1) * self.field['Principal'],
            self.field['Amortisation'], accrual_daycount, pay_rate)

        # include the maturity date in the daycount
        field_index['Cashflows'].add_maturity_accrual(base_date, accrual_daycount)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_credit_step_down_leg(shared, time_grid, deal_data)


class DealDefaultSwap(Deal):
    vernacular = 'credit default swap, CDS, single-name CDS'
    fields = [ADMIN, PRICING, own('DealDefaultSwap', [
        F('Upfront_Date', 'Date', default='', convention=True),
        F('Upfront', 'Percent', default=0.0, convention=True),
        F('Penultimate_Coupon_Date', 'Date', default='', convention=True),
        F('Pay_Rate', 'Basis', default=0.0),
        F('Pay_Frequency', 'Period', default='3M', convention=True),
        F('Name', 'Text', default='', factors=['SurvivalProb']),
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Amortisation', 'DateList', default='null', convention=True,
          row=Row([F('Date', 'Date'), F('Amount', 'Float', sized='magnitude')])),
        F('Calendars', 'Text', default='', convention=True),
        F('Accrual_Day_Count', 'Text', default='ACT_365', convention=True, values=DAY_COUNTS),
        F('Is_Digital', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Effective_Date', 'Date', default=''),
        F('Maturity_Date', 'Date', default=''),
        F('Digital_Recovery', 'Percent', default=0.0, convention=True),
        F('Accrue_Fee', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Principal', 'Float', default=0.0, sized=True)
    ])]

    documentation = ('Credit', [
        'A single-name credit default swap [@hull2000; @okane2008]. The protection buyer pays a periodic fee',
        '**Pay_Rate** (basis points) on the **Principal** (less any **Amortisation**) while the **Name** survives;',
        'on default the seller pays $P(1-R)$, $R$ the recovery on the name\'s survival factor, or',
        '**Digital_Recovery** in its place where **Is_Digital** is `Yes`. For a protection buyer, with',
        '$\\tilde t_i=\\max(t_i,t)$, $t_i$ the period boundaries, $T_i$ the payment dates and $S(t,\\cdot)$ the',
        'survival curve,',
        '',
        '$$V(t)=\\sum_{i}P_i(1-R)V_i(t)-\\sum_{i}P_ic\\,\\alpha_iD(t,T_i)\\big(S(t,T_i)+a_i\\,p_i\\big)-UP\\,D(t,T_u)\\mathbf 1_{\\{t\\le T_u\\}}$$',
        '',
        'and a protection seller holds $-V$. Here $c$ is the fee rate, $\\alpha_i$ the accrual fraction in',
        '**Accrual_Day_Count**, $p_i=S(t,\\tilde t_{i-1})-S(t,T_i)$ the period\'s default probability, and',
        '$U$ the **Upfront** (percent of $P$) paid by the buyer on **Upfront_Date** (the effective date',
        'where blank), counted while $t\\le T_u$ whatever the name does. The protection leg uses the',
        'midpoint form',
        '',
        '$$V_i(t)=\\frac{D(t,\\tilde t_{i-1})+D(t,T_i)}{2}\\Big(S(t,\\tilde t_{i-1})-S(t,T_i)\\Big)$$',
        '',
        'of the exact flat-hazard, flat-rate period value',
        '$\\frac{\\bar h_i}{f_i+\\bar h_i}\\big(D(t,\\tilde t_{i-1})S(t,\\tilde t_{i-1})-D(t,T_i)S(t,T_i)\\big)$, with',
        '$\\bar h_i$ and $f_i$ the period\'s average hazard and forward rate. Its relative error per period',
        'against that exact form, computed analytically: $7.8\\times10^{-6}$ ($f=5\\%$, $\\bar h=2\\%$,',
        'quarterly), $-6.5\\times10^{-5}$ ($f=5\\%$, $\\bar h=30\\%$, quarterly), $-8.3\\times10^{-4}$',
        '($f=10\\%$, $\\bar h=50\\%$, semi-annual).',
        '',
        '**Accrue_Fee** `Yes` adds the fee accrued to default, $a_i=\\alpha^A_i+\\tfrac12\\alpha^R_i$, with',
        '$\\alpha^A_i$ the fraction of the period already accrued and $\\alpha^R_i=1-\\alpha^A_i$ the fraction',
        'remaining (so $a_i=\\tfrac12$ for a period not yet started); `No` sets $a_i=0$.'])

    def reset(self, calendars):
        super(DealDefaultSwap, self).reset()
        bus_day = calendars.get(self.field['Calendars'], {'businessday': pd.offsets.Day(1)})['businessday']
        if list(self.field['Pay_Frequency'].kwds.values()) == [0]:
            self.resetdates = pd.DatetimeIndex([self.field['Effective_Date'], self.field['Maturity_Date']])
        else:
            if self.field.get('Penultimate_Coupon_Date'):
                self.resetdates = utils.generate_dates_backward(
                    self.field['Penultimate_Coupon_Date'], self.field['Effective_Date'],
                    self.field['Pay_Frequency'], bus_day=bus_day)
                if self.field['Maturity_Date'] > self.field['Penultimate_Coupon_Date']:
                    self.resetdates = self.resetdates.append(pd.DatetimeIndex([self.field['Maturity_Date']]))
            else:
                self.resetdates = utils.generate_dates_backward(
                    self.field['Maturity_Date'], self.field['Effective_Date'],
                    self.field['Pay_Frequency'], bus_day=bus_day)

        self.add_reval_dates(self.resetdates, self.field['Currency'])
        # the upfront is paid on its own day, the premium leg's effective date where none is stated
        self.upfront_day = self.field.read('Upfront_Date', self.field['Effective_Date'])
        if self.percent('Upfront'):
            self.add_reval_dates({self.upfront_day}, self.field['Currency'])

    def percent(self, key):
        """A stated `Percent` as a fraction, a bare number read as the percent it is declared, as
        the coupon's is."""
        stated = self.field[key] or 0.0
        return stated / 100.0 if isinstance(stated, (int, float)) else stated.amount

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Name': utils.check_rate_name(self.field['Name'])}

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {
            'Currency': get_fxrate_factor(field['Currency'], static_offsets, stochastic_offsets),
            'SettleCurrency': self.field['Currency'],
            'Accrue_Fee': self.field.get('Accrue_Fee', 'No') == 'Yes',
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Name': get_survival_factor(field['Name'], static_offsets, stochastic_offsets, all_tenors),
            'Recovery_Rate': get_survival_component(field['Name'], all_factors)
        }
        if self.field['Is_Digital'] == 'Yes':
            # a digital pays a stated share of the notional, not the name's own recovery
            field_index['Digital_Recovery'] = self.percent('Digital_Recovery')

        pay_rate = self.field['Pay_Rate'] / 100.0 if isinstance(
            self.field['Pay_Rate'], float) else self.field['Pay_Rate'].amount
        # signed as the premium moves: the protection buyer PAYS it
        nominal = (-1 if self.field['Buy_Sell'] == 'Buy' else 1) * self.field['Principal']
        daycount = utils.DayCount.code(self.field['Accrual_Day_Count'])

        field_index['Cashflows'] = utils.TensorCashFlows.generate_fixed(
            base_date, self.resetdates, nominal, self.field['Amortisation'], daycount, pay_rate)

        # include the maturity date in the daycount
        field_index['Cashflows'].add_maturity_accrual(base_date, daycount)

        # a positive upfront is paid by the protection buyer; one behind the base date is paid
        upfront = self.percent('Upfront')
        if upfront and self.upfront_day >= base_date:
            field_index['Upfront'] = utils.TensorCashFlows([utils.TensorCashFlows.make_cashflow(
                base_date, self.upfront_day, self.upfront_day, self.upfront_day, 0.0, daycount,
                upfront * nominal, 0.0)], np.zeros((1, 3)))

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_cds_leg(shared, time_grid, deal_data)


class FRADeal(Deal):
    vernacular = 'forward rate agreement, FRA'
    observes = Observes('Interest_Rate', 'InterestRate')
    quoted = 'FRA_Rate'
    fields = [ADMIN, PRICING, own('FRADeal', [
        F('Use_Known_Rate', 'Text', default='No', convention=True, values=['Yes', 'No'],
          description='Yes where the reset has fixed at Known_Rate, which a reset behind the base date must'),
        F('Known_Rate', 'Percent', default=0.0, convention=True),
        F('Payment_Timing', 'Text', default='End', convention=True, values=['End', 'Begin', 'Discounted']),
        F('Principal', 'Float', default=0.0, sized=True),
        F('Interest_Rate', 'Text', default='', convention=True, factors=['InterestRate']),
        F('Day_Count', 'Text', default='ACT_365', convention=True, values=DAY_COUNTS),
        F('Calendars', 'Text', default='', convention=True),
        F('Reset_Date', 'Date', default='', convention=True),
        F('Effective_Date', 'Date', default=''),
        F('Maturity_Date', 'Date', default=''),
        F('Borrower_Lender', 'Text', default='Borrower', values=['Borrower', 'Lender'], side=True),
        F('FRA_Rate', 'Float', default=0.0)
    ])]

    documentation = ('Interest Rates',
                     ['A forward rate agreement (FRA) pays the difference between a floating rate and a fixed',
                      'rate over **Effective_Date** to **Maturity_Date**. It is one',
                      '[floating interest cashflow](#floating-interest-cashflows) with margin $-K$:',
                      '',
                      '$$V(t)=\\pm P\\,\\big(L(t)-K\\big)\\,\\alpha\\,D(t,T_{PV})$$',
                      '',
                      'with $\\pm$ $+$ for a **Borrower** and $-$ for a **Lender**, $P$ **Principal**, $K$ **FRA_Rate**,',
                      '$\\alpha$ the accrual fraction in **Day_Count**, $L$ the forward on **Interest_Rate** (or',
                      '**Known_Rate** where **Use_Known_Rate** is `Yes`) reset on **Reset_Date** (the effective date',
                      'where blank). **Payment_Timing** `Begin` values and pays the amount at the effective date',
                      'with no $1/(1+L\\alpha)$ discount; `End` and `Discounted` both value from maturity',
                      '($T_{PV}$ = maturity), `End` paying there and `Discounted` paying the amount discounted back',
                      'to the effective date.'
                      ])

    def reset(self, calendars):
        super(FRADeal, self).reset()
        self.pay_date = self.field['Maturity_Date'] if self.field.get(
            'Payment_Timing', 'End') == 'End' else self.field['Effective_Date']
        self.add_reval_dates({self.pay_date}, self.field['Currency'])

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {
            'Currency': utils.check_rate_name(self.field['Currency']),
            'Reset_Date': self.field['Reset_Date'] if self.field.get('Reset_Date') else self.field['Effective_Date']
        }

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])
        field['Interest_Rate'] = self.check_factor_name('Interest_Rate', field['Discount_Rate'])

        field_index = {
            'Currency': get_fx_and_zero_rate_factor(
                field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
            'Discount': get_interest_factor(
                field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Forward': get_interest_factor(
                field['Interest_Rate'], static_offsets, stochastic_offsets, all_tenors),
            'Daycount': utils.DayCount.code(self.field['Day_Count']),
            'CompoundingMethod': 'None',
            'SettleCurrency': self.field['Currency']
        }

        Accrual_fraction = utils.DayCount.accrual(
            base_date, (self.field['Maturity_Date'] - self.field['Effective_Date']).days, field_index['Daycount'])

        # the PV date, NOT the settlement date: only Begin values from the start of the rate
        # period. End and Discounted both PV from maturity and differ only in where the cash
        # lands, which `reset`'s pay_date states.
        timing = self.field.get('Payment_Timing', 'End')
        discount_date = self.field['Effective_Date'] if timing == 'Begin' else self.field['Maturity_Date']

        cashflows = {'Items':
            [{
                'Payment_Date': discount_date,
                'Accrual_Start_Date': self.field['Effective_Date'],
                'Accrual_End_Date': self.field['Maturity_Date'],
                'Accrual_Year_Fraction': Accrual_fraction,
                'Notional': self.field['Principal'],
                'Margin': utils.Basis(-100.0 * self.field['FRA_Rate']),
                # the reset fixed at Known_Rate where Use_Known_Rate says so, and states no rate where not
                'Resets': [
                    [field['Reset_Date'], self.field['Effective_Date'],
                     self.field['Maturity_Date'], Accrual_fraction, self.field['Use_Known_Rate'],
                     self.field['Known_Rate'] if self.field['Use_Known_Rate'] == 'Yes' else None]
                ]
            }]
        }

        field_index['VolSurface'] = np.zeros(1, dtype=np.int32)
        field_index['Cashflows'] = utils.TensorCashFlows.float(
            base_date, time_grid, 1 if self.field['Borrower_Lender'] == 'Borrower' else -1,
            cashflows, self.field.get('Reference'))

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_fra_leg(shared, time_grid, deal_data)


class FloatingEnergyDeal(Deal):
    vernacular = 'floating energy leg, floating-price energy swap'
    fields = [ADMIN, PRICING, own('FloatingEnergyDeal', [
        F('Sampling_Type', 'Text', default='', factors=['ForwardPriceSample']),
        F('FX_Sampling_Type', 'Text', default='', convention=True, factors=['ForwardPriceSample']),
        F('Average_FX', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Payer_Receiver', 'Text', default='Payer', values=['Payer', 'Receiver'], side=True),
        F('Payments', 'Container',
          sub_fields=[
              F('Items', 'Table', default='null', row=Row([
                  F('Payment_Date', 'Date'), F('Period_Start', 'Date'), F('Period_End', 'Date'),
                  F('Volume', 'Float', sized=True), F('Fixed_Basis', 'Float'), F('Price_Multiplier', 'Float'),
                  F('Realized_Average_Date', 'Date'), F('Realized_Average', 'Float'), F('FX_Period_Start', 'Date'),
                  F('FX_Period_End', 'Date'), F('FX_Realized_Average', 'Float')]))]),
        F('Reference_Type', 'Text', default='', factors=['ReferencePrice']),
        F('Payoff_Currency', 'Text', default='', convention=True, factors=['FxRate']),
        F('Commodity', 'Text', default='', factors=['CommodityPrice'])
    ])]

    documentation = ('Energy', [
        'The time $t$ value of an energy cashflow paid at $T$ indexed to volume $V$ of energy at a price determined by',
        'the reference price $S^R$ is',
        '',
        '$$V (A S^R(t,t_s^s,t_e^s,\\mathcal S)+b)D(t,T)$$',
        '',
        'where $A$ is the **Price Multiplier** and $b$ is the **Fixed Basis**.',
        '',
        '**Valuation options** (set in the Valuation Configuration section, per deal type)',
        '',
        '- **ForwardCurve**: `Full` (default — sample the simulated `ForwardPrice` curve directly) or',
        '`Components`. With `Components` the forward curve is *derived* from the simulated spot',
        'components instead: the deal reads its **Commodity** (`CommodityPrice`) factor and builds the',
        'curve with $F(t,t)=S(t)$, so a spot model (e.g. a GARCH or regime world) drives the fixings',
        'without a separately simulated forward curve.'
    ])

    def reset(self, calendars):
        super(FloatingEnergyDeal, self).reset()
        paydates = set([x['Payment_Date'] for x in self.field['Payments']['Items']])
        self.add_reval_dates(paydates, self.payoff_ccy)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        """Resolve the floating energy leg's factor dependencies.

        Under ForwardCurve='Components' the forward curve is built from the components already
        simulated, and Commodity may name a composed spot: `get_commodity_rate_factor` is
        basis-aware and returns the summed code, while `get_factor_component` resolves repo and
        carry off the primary."""
        field = {
            'Currency': utils.check_rate_name(self.field['Currency']),
            'Sampling_Type': utils.check_rate_name(self.field['Sampling_Type']),
            'FX_Sampling_Type': self.check_factor_name('FX_Sampling_Type'),
            'Reference_Type': utils.check_rate_name(self.field['Reference_Type']),
            'Payoff_Currency': utils.check_rate_name(self.payoff_ccy)
        }

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {}

        if self.options.get('ForwardCurve', 'Full') == 'Components':
            # forwardcurve from the already-simulated components; Commodity may be composed - see docstring
            field["Commodity"] = utils.check_rate_name(self.field["Commodity"])
            reference_factor = get_reference_factor(field["Reference_Type"], all_factors)
            commodity_factor = get_commodity_component(field["Commodity"], all_factors)
            field_index['Commodity'] = get_commodity_rate_factor(
                field['Commodity'], static_offsets, stochastic_offsets)
            field_index["ForwardPrice"], field_index["ForwardFX"], field_index["CashFX"] = get_forwardprice_factor(
                field['Payoff_Currency'], static_offsets, stochastic_offsets, all_tenors,
                all_factors, None, commodity_factor, base_date)
            field_index["Repo_Rate"] = get_interest_factor(
                commodity_factor.get_repo_curve_name(), static_offsets, stochastic_offsets, all_tenors)
            field_index["Carry_Rate"] = get_forward_rate_factor(
                commodity_factor.get_carry_curve_name(), static_offsets, stochastic_offsets, all_tenors)
            field_index['ForwardStart'] = np.array(
                sorted([(x - reference_factor.start_date).days for x in time_grid.mtm_dates]))
        else:
            reference_factor, forward_factor = get_reference_factor_objects(field['Reference_Type'], all_factors)
            field_index['ForwardPrice'], field_index['ForwardFX'], field_index['CashFX'] = get_forwardprice_factor(
                field['Payoff_Currency'], static_offsets, stochastic_offsets, all_tenors,
                all_factors, reference_factor, forward_factor, base_date)

        fx_sample = get_forwardprice_sampling(
            field["FX_Sampling_Type"], all_factors) if field["FX_Sampling_Type"] else None
        forward_sample = get_forwardprice_sampling(field["Sampling_Type"], all_factors)
        field_index['Discount'] = get_interest_factor(
            field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors)
        field_index['Cashflows'] = utils.TensorCashFlows.energy(
            base_date, time_grid, -1.0 if self.field['Payer_Receiver'] == 'Payer' else 1.0,
            self.field['Payments'], reference_factor, forward_sample, fx_sample, calendars)
        field_index['Currency'] = get_fx_and_zero_rate_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)
        field_index['SettleCurrency'] = self.payoff_ccy

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_energy_leg(shared, time_grid, deal_data)


class FixedEnergyDeal(Deal):
    vernacular = 'fixed energy leg, fixed-price energy swap'
    fields = [ADMIN, PRICING, own('FixedEnergyDeal', [
        F('Payoff_Currency', 'Text', default='', convention=True, factors=['FxRate']),
        F('Payer_Receiver', 'Text', default='Payer', values=['Payer', 'Receiver'], side=True),
        F('Payments', 'Container',
          sub_fields=[
              F('Items', 'Table', default='null',
                row=Row([F('Payment_Date', 'Date'), F('Volume', 'Float', sized=True), F('Fixed_Price', 'Float')]))])
    ])]

    documentation = ('Energy', [
        'The time $t$ value of a fixed energy cashflow paid at $T$ indexed to volume $V$ of energy at a fixed price $K$ is',
        '',
        '$$V K D(t,T)$$',
    ])

    def reset(self, calendars):
        super(FixedEnergyDeal, self).reset()
        self.paydates = set([x['Payment_Date'] for x in self.field['Payments']['Items']])
        self.add_reval_dates(self.paydates, self.payoff_ccy)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {'Currency': utils.check_rate_name(self.field['Currency']),
                 'Payoff_Currency': utils.check_rate_name(self.payoff_ccy)}
        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field_index = {'SettleCurrency': self.payoff_ccy,
                       'Discount': get_interest_factor(
                           field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors),
                       'Currency': get_fx_and_zero_rate_factor(
                           field['Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors),
                       'Cashflows': utils.TensorCashFlows.simple_fixed(
                           base_date, -1.0 if self.field['Payer_Receiver'] == 'Payer' else 1.0,
                           self.field['Payments'],
                           lambda cashflow: cashflow['Volume'] * cashflow['Fixed_Price'])}

        if field['Payoff_Currency'] != field['Currency']:
            field_index['SettleFX'] = get_fx_and_zero_rate_factor(
                field['Payoff_Currency'], static_offsets, stochastic_offsets, all_tenors, all_factors)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_fixed_leg(shared, time_grid, deal_data)


class EnergySingleOption(Deal):
    vernacular = 'energy option, commodity option'
    fields = [ADMIN, PRICING, own('EnergySingleOption', [
        F('Buy_Sell', 'Text', default='Buy', values=['Buy', 'Sell'], side=True),
        F('Sampling_Type', 'Text', default='', factors=['ForwardPriceSample']),
        F('FX_Sampling_Type', 'Text', default='', convention=True, factors=['ForwardPriceSample']),
        F('Average_FX', 'Text', default='No', convention=True, values=['Yes', 'No']),
        F('Settlement_Date', 'Date', default=''),
        F('Period_Start', 'Date', default=''),
        F('Period_End', 'Date', default=''),
        F('Strike', 'Float', default=0.0),
        F('Realized_Average', 'Float', default=0.0),
        F('Option_Type', 'Text', default='Call', values=['Call', 'Put']),
        F('FX_Period_Start', 'Date', default=''),
        F('FX_Period_End', 'Date', default=''),
        F('FX_Realized_Average', 'Float', default=0.0),
        F('Volume', 'Float', default=0.0, sized=True),
        F('Reference_Type', 'Text', default='', factors=['ReferencePrice']),
        F('Reference_Volatility', 'Text', default='', factors=['CommodityPriceVol']),
        F('Payoff_Type', 'Text', default='Compo', convention=True, values=['Compo', 'Quanto']),
        F('Payoff_Currency', 'Text', default='', convention=True, factors=['FxRate'])
    ])]

    documentation = ('Energy', [
        'A European option on an energy forward contract can be priced using the Black model with a volatility derived',
        'using the moment matching approach described earlier. Consider a European option with a reference price',
        '$S^R(t,t_s^s,t_e^s,\\mathcal S)$, where $t_s^s$ is the usual start of the sampling period and $t_e^s$ is the',
        'end of the period and also the expiry date of the option. The value of the option with strike $K$ and',
        'settlement date $T$ is',
        '',
        '$$\\mathcal B_\\delta (S^R(t,t_s^s,t_e^s,\\mathcal S),K,w(t,t_e^s,t_s^s,t_e^s,\\mathcal S))D(t,T)$$',
        '',
        'where $w(t,t_e^s,t_s^s,t_e^s,\\mathcal S)$ is the standard deviation of $S^R(t,t_s^s,t_e^s,\\mathcal S)$',
        'and $\\mathcal B_\\delta$ is the Black formula [@black1976].',
        '',
        'Paid in a **Currency** other than the forward curve\'s own, **Payoff_Type** says how: `Compo`, what',
        'omission means, averages the product $S\\cdot X$ - every sample at its own fx forward and its variance',
        'composed with the pair\'s - and `Quanto` averages the local samples, each forward one carried by',
        '$-\\rho\\sigma_S\\sigma_X$ over its own tenor, and pays at one. The pair\'s `FXVol` and the correlation',
        '`Correlation.FxRate.<local>.<payoff>/ReferencePrice.<reference>.<local>` are the ones the energy',
        'family declares.'
    ])

    def reset(self, calendars):
        super(EnergySingleOption, self).reset()
        self.paydates = {self.field['Settlement_Date']}
        self.add_reval_dates(self.paydates, self.payoff_ccy)

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                          calendars):
        field = {
            'Currency': utils.check_rate_name(self.field['Currency']),
            'Payoff_Currency': utils.check_rate_name(self.payoff_ccy),
            'Sampling_Type': utils.check_rate_name(self.field['Sampling_Type']),
            'FX_Sampling_Type': self.check_factor_name('FX_Sampling_Type'),
            'Reference_Type': utils.check_rate_name(self.field['Reference_Type']),
            'Reference_Volatility': utils.check_rate_name(self.field['Reference_Volatility'])
        }

        field['Discount_Rate'] = self.check_factor_name('Discount_Rate', field['Currency'])

        field['cashflow'] = {'Payment_Date': self.field['Settlement_Date'],
                             'Period_Start': self.field['Period_Start'],
                             'Period_End': self.field['Period_End'],
                             'Volume': 1.0,
                             'Fixed_Price': self.field['Strike'],
                             'Realized_Average': self.field['Realized_Average'] or 0.0,
                             'FX_Period_Start': self.field['FX_Period_Start'],
                             'FX_Period_End': self.field['FX_Period_End'],
                             'FX_Realized_Average': self.field['FX_Realized_Average'] or 0.0}

        field_index = {}
        reference_factor, forward_factor = get_reference_factor_objects(field['Reference_Type'], all_factors)
        forward_sample = get_forwardprice_sampling(field['Sampling_Type'], all_factors)
        fx_sample = get_forwardprice_sampling(field['FX_Sampling_Type'], all_factors) if field[
            'FX_Sampling_Type'] else None
        cashflow = utils.TensorCashFlows.energy(
            base_date, time_grid, 1, {'Items': [field['cashflow']]},
            reference_factor, forward_sample, fx_sample, calendars)
        field_index['Cashflow'] = cashflow

        field_index['ForwardPrice'], field_index['ForwardFX'], field_index['CashFX'] = get_forwardprice_factor(
            field['Currency'], static_offsets, stochastic_offsets, all_tenors,
            all_factors, reference_factor, forward_factor, base_date)

        field_index['Payoff_Currency'] = get_fxrate_factor(
            field['Payoff_Currency'], static_offsets, stochastic_offsets)
        field_index['Discount'] = get_interest_factor(
            field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors)
        field_index['Volatility'] = get_commodity_vol_factor(
            field['Reference_Volatility'], static_offsets, stochastic_offsets, all_tenors)
        field_index['SettleCurrency'] = self.payoff_ccy
        field_index['Buy_Sell'] = 1.0 if self.field['Buy_Sell'] == 'Buy' else -1.0
        field_index['Option_Type'] = 1.0 if self.field['Option_Type'] == 'Call' else -1.0
        field_index['Expiry'] = (self.field['Settlement_Date'] - base_date).days
        field_index['Strike'] = self.field['Strike']

        # the option seam's roles: `Currency` the forward curve's own, `Payoff_Currency` the deal's
        pair = {
            'Currency': forward_factor.get_currency(),
            'Payoff_Currency': field['Currency'],
            'Reference_Type': field['Reference_Type']
            }
        self.check_option_data(
            pair, field_index, static_offsets, stochastic_offsets, all_tenors, all_factors, reference_fx_correlation)

        return field_index

    def generate(self, shared, time_grid, deal_data):
        return pricing.pv_energy_option(shared, time_grid, deal_data, self.field['Volume'])


def accepts_children(deal_type):
    """Whether a deal of this type breaks down into simpler instruments.

    A container prices its children in `post_process`; a leaf cannot. The engine recurses on
    `Children` being PRESENT and never on the type, so this is an AUTHORING question: a UI that
    offers a leaf children builds a tree the pricer will not read.
    """
    return getattr(globals().get(deal_type), 'accepts_children', False)


def construct_instrument(param, all_valuation_options):
    """The deal `param` names, or the bare `{'Object': ...}` an unknown type loads as.

    A node that never became a `Deal` keeps the name it was authored with and nothing else, which
    is what `Config.nameless_deals` refuses it by - the payload goes, a misspelt type carrying no
    terms anyone can price.
    """
    if param.get('Object') not in globals():
        logging.error('Instrument {0} not defined'.format(param.get('Object')))
        return {'Object': param.get('Object')}
    else:
        deal_options = all_valuation_options.get(param.get('Object'), {})
        return globals().get(param.get('Object'))(param, deal_options)


def deal_node(block, all_valuation_options):
    """A deal-tree node from a wire block, its `Children` blocks nodes under it - the shape
    `Config.set_calculation_children` takes, the one adapter a block reaches a constructed tree by."""
    node = {'Instrument': construct_instrument(
        {key: value for key, value in block.items() if key != 'Children'}, all_valuation_options)}
    if block.get('Children'):
        node['Children'] = [deal_node(child, all_valuation_options) for child in block['Children']]
    return node


if __name__ == '__main__':
    pass
