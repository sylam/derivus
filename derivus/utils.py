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

import calendar
import math
from functools import reduce, wraps
from collections import namedtuple, deque
from typing import Tuple, List
from dataclasses import dataclass

import logging
import scipy.stats
import pandas as pd
import numpy as np

import torch

# For dealing with excel dates and dataframes
excel_offset = pd.Timestamp('1899-12-30 00:00:00')


def array_type(x): return np.array(x)


# Days in year - could set this to 365.0 or 365.25 if you want that bit extra time
DAYS_IN_YEAR = 365.25

# daycount codes
DAYCOUNT_None = -1
DAYCOUNT_ACT365 = 0
DAYCOUNT_ACT360 = 1
DAYCOUNT_ACT365IDSA = 2
DAYCOUNT_ACT30_360 = 3
DAYCOUNT_ACT30_E360 = 4
DAYCOUNT_ACTACTICMA = 5

# factor codes
FACTOR_INDEX_Stoch = 0  # either True for stochastic or False for static
FACTOR_INDEX_Offset = 1  # index to get the factor name
FACTOR_INDEX_SubType = 2  # index to get the factor subtype (if any)
# these are indices to get the tenors relevant to interpolate the risk factor in question
FACTOR_INDEX_Tenor_Index = 3
FACTOR_INDEX_Daycount = 4  # daycount code
FACTOR_INDEX_ExcelCalcDate = 4
FACTOR_INDEX_Moneyness_Index = 3
FACTOR_INDEX_Expiry_Index = 4
FACTOR_INDEX_VolTenor_Index = 5
FACTOR_INDEX_Flat_Index = 5
FACTOR_INDEX_Surface_Flat_Index = 6

# cashflow codes 
CASHFLOW_INDEX_Start_Day = 0
CASHFLOW_INDEX_End_Day = 1
CASHFLOW_INDEX_Pay_Day = 2

CASHFLOW_INDEX_Year_Frac = 3
# can also use this index for equity swaplet multipliers
CASHFLOW_INDEX_Start_Mult = 3

CASHFLOW_INDEX_Nominal = 4
# can also use this index for equity swaplet multipliers
CASHFLOW_INDEX_End_Mult = 4

CASHFLOW_INDEX_FixedAmt = 5

# Cashflow code for Float payments
CASHFLOW_INDEX_FloatMargin = 6
# Cashflow code for Fixed payments
CASHFLOW_INDEX_FixedRate = 6
# Cashflow code for caps/floor payments
CASHFLOW_INDEX_Strike = 6
# Cashflow code for equity swaplet multipliers
CASHFLOW_INDEX_Dividend_Mult = 6
# Cashflow code for possible FX resets
CASHFLOW_INDEX_FXResetDate = 7
# for equity swaps, we need to adjust days based on settlement
CASHFLOW_INDEX_Start_Adj = 7
CASHFLOW_INDEX_FXResetValue = 8
CASHFLOW_INDEX_End_Adj = 8

# used by inflation cashflows
CASHFLOW_INDEX_BaseReference = 9
CASHFLOW_INDEX_FinalReference = 10
CASHFLOW_OFFSET_Settle = 2

# Number of resets/fixings for this cashflow (0 for fixed cashflows)
CASHFLOW_INDEX_NumResets = 9
# offset in the reset/fixings array for this cashflow
CASHFLOW_INDEX_ResetOffset = 10
# Boolean (0 or 1) value that determines if this cashflow is settled (1) or accumulated (0)
CASHFLOW_INDEX_Settle = 11

# Cashflow calculation methods 

CASHFLOW_METHOD_Equity_Shares = 0
CASHFLOW_METHOD_Equity_Principal = 1
CASHFLOW_METHOD_Average_Interest = 0

CASHFLOW_METHOD_Compounding_Include_Margin = 2
CASHFLOW_METHOD_Compounding_Flat = 3
CASHFLOW_METHOD_Compounding_Exclude_Margin = 4
CASHFLOW_METHOD_Compounding_None = 5

CASHFLOW_METHOD_Fixed_Compounding_No = 0
CASHFLOW_METHOD_Fixed_Compounding_Yes = 1


# reset codes - note that the first 3 fields correspond with the TIME_GRID
# (so that a reset can be treated as a timepoint)
RESET_INDEX_Time_Grid = 0
RESET_INDEX_Reset_Day = 1
RESET_INDEX_Scenario = 2
RESET_INDEX_Start_Day = 3
RESET_INDEX_End_Day = 4
RESET_INDEX_Weight = 5
RESET_INDEX_Value = 6
# used to store the reset accrual period
RESET_INDEX_Accrual = 7
# used to store any fx averaging (can't be used with accrual periods)
RESET_INDEX_FXValue = 7

# modifiers for dealing with a sequence of cashflows
SCENARIO_CASHFLOWS_FloatLeg = 0
SCENARIO_CASHFLOWS_Cap = 1
SCENARIO_CASHFLOWS_Floor = 2
SCENARIO_CASHFLOWS_Energy = 3
SCENARIO_CASHFLOWS_Index = 4
SCENARIO_CASHFLOWS_Equity = 5

# Constants for the time grid
TIME_GRID_PriorScenarioDelta = 0
TIME_GRID_MTM = 1
TIME_GRID_ScenarioPriorIndex = 2

# Collateral Cash Valuation mode
CASH_SETTLEMENT_Received_Only = 0
CASH_SETTLEMENT_Paid_Only = 1
CASH_SETTLEMENT_All = 2

# Factor sizes
FACTOR_SIZE_CURVE = 4
FACTOR_SIZE_RATE = 2

# Named tuples to make life easier
Factor = namedtuple('Factor', 'type name')
RateInfo = namedtuple('RateInfo', 'model_name archive_name calibration')
CalibrationInfo = namedtuple('CalibrationInfo', 'param correlation delta')
# One equation's fit from `stochasticprocess.arx1_t_mle`. `sigma` is the innovation standard
# DEVIATION - scalar under constant variance, the path sigma_t under `garch=True`, where
# `sigma[-1]**2` is the NEXT observation's variance. `resid` is standardised by `sigma`.
ARX1Fit = namedtuple('ARX1Fit', 'phi mu sigma gamma resid garch')
DealDataType = namedtuple('DealDataType', 'Instrument Factor_dep Time_dep Calc_res')
Partition = namedtuple('Partition', 'DealMTMs Collateral_Cash Funding_Cost Cashflows')
Collateral = namedtuple('Collateral', 'Haircut Amount Currency Funding_Rate Collateral_Rate Collateral')

# define 1, 2 and 3d risk factors - add more as development proceeds
DimensionLessFactors = ['ReferenceVol', 'Correlation']
OneDimensionalFactors = ['InterestRate', 'InflationRate', 'DividendRate', 'SurvivalProb', 'ForwardPrice', 'ForwardRate']
#: Every (moneyness, expiry) vol surface. ONE implementation - `riskfactors.VolatilityGrid` - with
#: an asset-class TAG per member, the risk-class partition below being a pure function of the factor
#: type. The untagged name is transitional: see `resolve_factor_key`.
TwoDimensionalFactors = ['FXVol', 'EquityPriceVol', 'CommodityPriceVol', 'VolatilityGrid']
ThreeDimensionalFactors = ['InterestRateVol', 'InterestYieldVol', 'ForwardPriceVol']

#: The CRIF-style risk class of every declared factor type, as data. A sensitivity is reported under
#: the class of the factor it is differentiated against, so the partition is TOTAL over the Factor
#: store and a pure function of `factor.type`; `CrossClass` is a factor whose class is inherited.
FactorRiskClass = {
    'InterestRate': 'InterestRate', 'InflationRate': 'InterestRate', 'PriceIndex': 'InterestRate',
    'InterestRateVol': 'InterestRate', 'InterestYieldVol': 'InterestRate',
    'HullWhite2FactorModelParameters': 'InterestRate',
    'FxRate': 'FX', 'FXVol': 'FX',
    'EquityPrice': 'Equity', 'EquityPriceVol': 'Equity', 'DividendRate': 'Equity',
    'CommodityPrice': 'Commodity', 'CommodityPriceVol': 'Commodity', 'FuturesPrice': 'Commodity',
    'ForwardPrice': 'Commodity', 'ForwardRate': 'Commodity', 'ForwardPriceVol': 'Commodity',
    'ForwardPriceSample': 'Commodity', 'ReferencePrice': 'Commodity', 'ReferenceVol': 'Commodity',
    'CSForwardPriceModelParameters': 'Commodity',
    'SurvivalProb': 'Credit',
    'Correlation': 'CrossClass', 'ObservedBasis': 'CrossClass',
    'GBMAssetPriceTSModelParameters': 'CrossClass', 'HestonNandiModelParameters': 'CrossClass',
    # TRANSITIONAL: an untagged surface cannot decide its own class. Here only because the store
    # still declares the name `resolve_factor_key` reads, and it retires with that shim.
    'VolatilityGrid': 'CrossClass'
}

# weekends and weekdays
WeekendMap = {'Friday and Saturday': 'Sun Mon Tue Wed Thu',
              'Saturday and Sunday': 'Mon Tue Wed Thu Fri',
              'Sunday': 'Mon Tue Wed Thu Fri Sat',
              'Saturday': 'Sun Mon Tue Wed Thu Fri',
              'Friday': 'Sat Sun Mon Tue Wed Thu'}


@dataclass
class DeferredDeal:
    payload: dict


@dataclass
class MTABoundarySet:
    """One netting set's transfer decisions, plus what is needed to price their counterfactuals.

    `replay` maps an alternative balance path to the netting-set MTM exactly as reported; `rescan`
    restarts the forward walk's own recursion at a margin date from a forced opening balance. Both
    are built inside post_process, so they capture only what does not depend on the balance.
    """
    events: list
    replay: object                  # callable: balance path -> netting-set MTM
    balance: object                 # the realised path, detached; the prefix a replay keeps
    rescan: object                  # callable: (opening, start) -> the balance path from there on

    # Who registered it - `BoundarySet.deal`'s slot, carried here too because this is not one of
    # those (it shares only `objective_jumps`) and `boundary_sets` holds both. A netting SET stamps
    # itself here: the decision is the set's, not any one deal's.
    deal = None

    def objective_jumps(self, score):
        """Per margin call, the gap and the change in the OBJECTIVE its transfer decision produces.

        The counterfactual is the SAME collateral recursion restarted from a forced opening balance
        - transfer, then hold - so nothing is re-simulated, re-priced or bumped: only the cheap
        balance scan is replayed. A replay yields this SET's net while `score` consumes a change to
        the PORTFOLIO, so the realised balance's own replay is the baseline - one extra scan per
        set, not per event.

        The receive and post sides of one call share a jump: D = 1 means "a transfer happened" for
        both, so only the gap differs. Computing it once halves the replays.

        The jump is masked by the event's `live`: a run of calls over which the balance is HELD
        publishes ONE transfer decision, and the coefficient carries it once. Masking the jump
        masks the kernel weight it multiplies, the product being the same three factors - and the
        amplification refusal, which reads the weights the SOLVE produced, sees the gaps either way.
        """
        unstamped = sum(event.live is None for event in self.events)
        if unstamped:
            raise ValueError(
                '{} of {} margin-call events reached the correction with no `live` mask: '
                '`mark_binding_calls` decides it over the whole series, so every site that builds '
                'events owes it one call'.format(unstamped, len(self.events)))
        with torch.no_grad():
            reported = self.replay(self.balance)
        jumps = {}
        for event in self.events:
            if event.call_index not in jumps:
                with torch.no_grad():
                    prefix = self.balance[:event.call_index]
                    transferred, held = [
                        score(self.replay(torch.cat(
                            [prefix, self.rescan(opening, event.call_index)], dim=0)) - reported)
                        for opening in (event.required_balance, event.previous_balance)]
                    jumps[event.call_index] = transferred - held
            # outside the block: `no_grad` is thread-local, and a generator suspended inside one
            # leaves it set in whoever resumes it - handing the caller a gap that carries no graph
            yield event.gap, jumps[event.call_index] * event.live


@dataclass
class BoundarySet:
    """One deal's decisions taken on SIMULATED state, and what a counterfactual needs of them.

    The value jump at such a decision is real - a knocked-out deal IS worth nothing - so the
    estimate is not what is wrong. What ordinary AAD drops is the FLUX: as a factor moves, scenarios
    cross the trigger, and the indicator recording it has zero derivative almost everywhere.

    Subclasses differ in ONE thing: how far a decision reaches - every row from the decision onward,
    or inside a pricer's own inner Monte Carlo. A decision must register as ONE counterfactual
    carrying its whole reach, because an objective with a kink (a collateralised net sits at the relu
    by construction) scores the sum of two partial counterfactuals differently from the
    counterfactual of the sum.

    They share the estimator's contract: `objective_jumps(score)` yields, per decision, the gap
    (graph retained, signed so gap > 0 means the trigger FIRED) and the objective's response. This
    base is the plumbing every form consumes: the gross-to-net chain and `portfolio_delta`.

    Branch values are registered on the PRICER's own grid and currency, and `to_mtm` is the deal's
    own map onto the MTM grid - `fx_rep` then INTERPOLATION, never a tail pad, since another deal's
    mtm date inside this deal's life inserts a row in the middle.
    """
    gaps: list                      # per decision, tensors, graph retained
    report_index: object            # MTM grid -> report grid, for the additive route only
    to_mtm: object                  # pricer-grid profile -> MTM-grid rows, detached

    # The netting set this registration sits beneath stamps its own gross-to-net chain here; None
    # (the class default, and what an ADDITIVE set leaves alone) means the counterfactual is the
    # reported MTM plus the delta. It rides on the SET because `boundary_sets` accumulates globally.
    net_from_gross = None
    # What the chain returns at a zero delta - this set's own level, the baseline a change is
    # measured from. Cached on first use: it costs one balance scan per registration.
    net_at_zero = None
    # Who registered it, stamped by `stamp_boundary_sets` off the structure walk. A slot rather than
    # a field because the subclasses declare non-default fields of their own, which cannot follow a
    # defaulted one. Read by the second-derivative refusal, which names the deals it refuses over.
    deal = None

    def portfolio_delta(self, delta, cash=None):
        """This registration's deal-mtm delta as a change to the reported PORTFOLIO.

        The chain returns this SET's net LEVEL while the objective consumes the root sum over every
        netting set, so what the portfolio gains is the chain at this delta LESS the chain at zero -
        true whatever the set's own arithmetic is. An additive set publishes no chain and only has
        to reach the report grid, and reads no ledger, so `cash` (a list of
        `(time_index, branch, booked)` rows for the branch being scored) reaches only the collateral
        chain.
        """
        if self.net_from_gross is None:
            return delta[self.report_index]
        if self.net_at_zero is None:
            self.net_at_zero = self.net_from_gross(torch.zeros_like(delta))
        return self.net_from_gross(delta, cash) - self.net_at_zero


@dataclass
class LatchedBoundarySet(BoundarySet):
    """One deal's LATCHING decisions taken on simulated state, and what a counterfactual needs.

    Two pricers register this shape: a discretely monitored barrier latches "has crossed" at each
    observation date; a physically settled swaption latches "was exercised" once, at expiry, and
    carries it over every later row.

    The counterfactual is cheap because flipping the decision does not change the SIMULATION, only
    the state read off it: `untriggered` and `triggered` are the two sides of the selection the
    pricer already evaluates, so nothing is re-simulated, re-priced or re-drawn.

    `gaps` retain their graph - log(spot/barrier) at each barrier observation, the underlying swap
    value at a swaption's expiry - and are signed so gap > 0 means the trigger FIRED, matching a
    jump of J(fired) - J(did not). Everything else is DETACHED: it seeds a counterfactual whose
    result is a coefficient, not a differentiated quantity.
    """
    fired: list                     # per decision, detached bool (B,)
    obs_before: object              # (T,) int: decisions strictly before each row, PRICER grid
    untriggered: object             # (T, B) detached, PRICER grid: value while it has not fired
    triggered: object               # (T, B) detached, PRICER grid: value once it has
    # The rows a decision forks that `obs_before` does not reach: None, or per decision a LIST of
    # `(pricer_row, value_if_fired, value_if_not)`, detached. A LAGGED settlement forks every row
    # of its block, not only the observation's. One decision is still ONE counterfactual.
    own_row: list = None
    # Settled-cash-in-transit a DEAD row still carries: None, or one entry per PRICER row - None or
    # `(first_decision, (m, B) tensor)` whose slice t is the row-present value of decision
    # `first_decision + t`'s fixed-but-unsettled payoff. Detached.
    pending: list = None
    # Per PAYMENT, `(mtm_row, decision, if_fired, if_not, booked)`: what the deal pays at that row
    # in the two states of the decision that GATES it, and the ledger as booked - detached
    # reporting-currency (B,) amounts, `decision` an index into `gaps` (-1 for cash no decision can
    # touch). A trigger's own payment declares `(amount, 0)` at its own decision; a STREAM's fixing
    # declares `(0, amount)` at the last decision that can kill it. Many payments may share a row,
    # and a decision may gate many. The collateral chain reads them through C_ts_te AGAINST THE
    # LEDGER `cash_settle` BUILT, so only settled cash may be declared - a payment the pricer priced
    # but never booked replays a ledger the reported world has not got. The additive route ignores
    # them.
    cash_events: list = None
    # Per WHOLE-VALUE SETTLEMENT, `(mtm_row, booked)`: a row that pays out everything the deal is
    # still worth, and the amount booked there - detached reporting-currency (B,). THE ROW'S MTM IS
    # THE SETTLEMENT, which is what lets a branch declare `booked + branch[row]`; a row that settles
    # one cashflow of a STREAM cannot use this - there `branch[row]` is the change in what remains,
    # not in what was paid, and the two differ by the whole of the deal's future.
    settles: list = None

    def branch_deltas(self):
        """Per decision, the deal-mtm delta with that decision forced ON and forced OFF.

        BOTH branches for EVERY scenario, not "what happened" against one alternative: the estimator
        wants E[jump | gap = 0], and near the boundary the scenarios that did not fire are as
        numerous as those that did, so scoring the former as a zero jump HALVES the conditional
        expectation. Re-deriving the value from the two branches the pricer already evaluated
        re-simulates and re-prices nothing.

        The selection happens on the PRICER grid and `to_mtm` is applied to the whole branch profile
        afterwards, not to the two branches separately. The two orders differ on an INTERPOLATED
        row, and selecting first reproduces that row's blend exactly because the map is linear.
        """
        with torch.no_grad():
            # latched state after each decision, and the value that state reports
            prefix = [torch.zeros_like(self.fired[0])]
            for flag in self.fired:
                prefix.append(prefix[-1] | flag)
            reported = self.to_mtm(self.select(prefix))
        for k, gap in enumerate(self.gaps):
            with torch.no_grad():
                on, off = [], []
                run_on = run_off = torch.zeros_like(self.fired[0])
                on.append(run_on)
                off.append(run_off)
                for j, flag in enumerate(self.fired):
                    run_on = run_on | (torch.ones_like(flag) if j == k else flag)
                    run_off = run_off | (torch.zeros_like(flag) if j == k else flag)
                    on.append(run_on)
                    off.append(run_off)
                own = (self.own_row[k] if self.own_row else None) or ()
                deltas = []
                for state, side in ((on, 1), (off, 2)):
                    profile = self.select(state)
                    for fork in own:
                        profile[fork[0]] = fork[side]
                    deltas.append(self.to_mtm(profile) - reported)
            yield (gap,) + tuple(deltas)

    def select(self, state):
        """The value profile a latch state reports: triggered where dead, untriggered where
        alive - plus, at a dead row with `pending` entries, the payoffs of the fixings that path
        SURVIVED whose settlements have not landed. `state[i]` is the dead prefix through the
        first `i` decisions, which makes the survived weight for decision j `~state[j + 1]`."""
        profile = torch.where(torch.stack(state)[self.obs_before],
                              self.triggered, self.untriggered)
        if self.pending is not None:
            for r, entry in enumerate(self.pending):
                if entry is None:
                    continue
                j0, c = entry
                survived = torch.stack(
                    [(~state[j + 1]).to(c.dtype) for j in range(j0, j0 + c.shape[0])])
                profile[r] = torch.where(state[self.obs_before[r]],
                                         self.triggered[r] + (c * survived).sum(dim=0),
                                         profile[r])
        return profile

    def objective_jumps(self, score):
        """Per decision, the gap and the change in the OBJECTIVE that decision produces.

        `score` maps a change to the reported portfolio to the per-scenario objective of the
        counterfactual. A latched decision moves one scenario's reported value by a FINITE amount,
        so the objective's response is a difference across that amount.

        Each counterfactual's ledger reach is derived from `cash_events` as it is scored, on ONE law
        whichever family declared them: a payment gated by decision d survives iff no decision up to
        d fired, and pays `if_fired` or `if_not` according to d's own state. So forcing d ON kills
        every payment gated at or after it and makes d's own; forcing it OFF pays them, each on the
        prefix that excludes d. A payment gated BEFORE the decision is identical in both worlds and
        is carried only to keep its row's total whole.

        WHOLE ROWS, because `cash_to_C` relu-splits received from paid: the split of a sum is not
        the sum of the splits, so payments sharing a row are added before the row is declared.

        A declared `settles` row adds the other half of the reach: the deal's whole remaining value
        at that row is what it pays there, so the branch settles `booked + branch[row]`. Without it
        the chain folds the REALISED cash into both branches while the balance scan follows the
        counterfactual, and the two disagree from the settlement onward.

        The yield is OUTSIDE the no_grad block on purpose: `no_grad` is thread-local, and a
        generator suspended inside one leaves it set in whoever resumes it - which would hand the
        caller a gap whose `gap - gap.detach()` carries no graph.
        """
        for k, (gap, on, off) in enumerate(self.branch_deltas()):
            with torch.no_grad():
                rows, changed = {}, []
                for t, branch_on, branch_off, booked in self.ledger(k):
                    if t not in rows:
                        rows[t] = [0.0, 0.0, 0.0]
                    entry = rows[t]
                    entry[0] = entry[0] + (booked if branch_on is None else branch_on)
                    entry[1] = entry[1] + (booked if branch_off is None else branch_off)
                    entry[2] = entry[2] + booked
                    if branch_on is not None and t not in changed:
                        changed.append(t)
                for t, booked in (self.settles or ()):
                    if t not in rows:
                        rows[t] = [0.0, 0.0, 0.0]
                    entry = rows[t]
                    entry[0], entry[1] = entry[0] + booked + on[t], entry[1] + booked + off[t]
                    entry[2] = entry[2] + booked
                    if t not in changed:
                        changed.append(t)
                on_cash = [(t, rows[t][0], rows[t][2]) for t in changed]
                off_cash = [(t, rows[t][1], rows[t][2]) for t in changed]
                jump = (score(self.portfolio_delta(on, on_cash)) -
                        score(self.portfolio_delta(off, off_cash)))
            yield gap, jump

    def ledger(self, k):
        """Per declared payment decision `k` can move, `(mtm_row, forced_on, forced_off, booked)`.

        A branch of None means "whatever was booked" - a payment this decision cannot reach, yielded
        so its row's total stays whole for the relu split, and never on its own account.
        """
        excl, before = torch.zeros_like(self.fired[0]), []
        for j, flag in enumerate(self.fired):
            before.append(excl)
            if j != k:
                excl = excl | flag
        for t, d, if_fired, if_not, booked in (self.cash_events or ()):
            if d < k:
                # gated earlier, or by nothing at all (-1) - the same cash in both worlds
                yield t, None, None, booked
            elif d == k:
                # the decision's own payment: forced ON it is made if nothing earlier killed the
                # deal, forced OFF it is what the other side of that decision pays there
                yield t, if_fired * ~before[d], if_not * ~before[d], booked
            else:
                # forced ON, decision k killed the deal before this payment's gate could pass it
                yield (t, torch.zeros_like(booked),
                       torch.where(self.fired[d], if_fired, if_not) * ~before[d], booked)


@dataclass
class InnerBoundarySet(BoundarySet):
    """Triggers taken INSIDE a pricer's own Monte Carlo - one decision per inner path.

    A TARF's knock-in is read off `Sj`, so the decision is per (scenario, inner path) and the
    reported row is the MEAN over those paths. That makes this a third shape rather than a variant:
    the other two move ONE SCENARIO's reported value by a finite amount, where an inner path moves
    the reported row by 1/n of itself. The objective's response to a perturbation that small is its
    DERIVATIVE, and a difference taken over a jump the reported value never takes measures its
    curvature instead - which CVA's `relu` has plenty of.

    So this shape carries the per-inner-path jump and `objective_jumps` differentiates `score` ONCE
    at the reported value and multiplies. The jump is the UNDIVIDED change to the row's own
    accumulator, the kernel's density already dividing by the pooled sample size B*n - which is also
    why the gaps go in as one (B, n) tensor rather than n separate decisions.

    Requires the objective to be SEPARABLE per scenario, which every exposure measure here is: the
    sensitivity is read off one backward pass over `score(...).sum()`, and that is the per-scenario
    derivative only because scenario b's objective depends on column b alone.
    """
    rows: list                      # per event, int: the PRICER-grid row the jump lands on
    jumps: list                     # per event, (B, n_inner) detached: the change in that row's
                                    # own accumulator if THAT inner path's trigger flips,
                                    # J(fired) - J(did not), undivided by n
    reported: object                # (T, B) detached, PRICER grid: the value as reported

    def objective_jumps(self, score):
        """Per decision, the gap and the objective's response to that inner path's jump.

        The multiplier is d(objective)/d(this pricer row) by the chain rule through the deal's own
        map. That map is LINEAR, so the image of a unit row IS its column of the Jacobian, and one
        backward pass through `score` supplies the other half - nothing re-priced, nothing drawn.
        """
        for gap, row, jump in zip(self.gaps, self.rows, self.jumps):
            with torch.enable_grad():
                unit = torch.zeros_like(self.reported)
                unit[row] = 1.0
                weight = self.to_mtm(unit)
                probe = torch.zeros_like(weight).requires_grad_(True)
                sensitivity, = torch.autograd.grad(score(self.portfolio_delta(probe)).sum(), probe)
            # outside the block: `enable_grad` is thread-local too, and a generator suspended
            # inside one hands the caller a context it never asked for
            yield gap, (sensitivity * weight).sum(dim=0).unsqueeze(1) * jump


def claim_boundary_sets(shared, mark):
    """Hand a netting set's gross-to-net chain to the registrations made beneath it.

    `post_process` runs only after its children are priced, so everything the set is answerable for
    is the TAIL of `boundary_sets` added since its structure was entered - hence the mark is taken
    there rather than here.

    Only a set that PUBLISHES a chain claims: an uncollateralised netting set passes a deal's value
    through additively, and a swaption's post_process is not a netting set at all, so both leave the
    registration for whatever sits above. An inner collateralised set that already stamped one is
    the closer of the two and keeps it.
    """
    chain = shared.__dict__.pop('gross_to_net', None)
    if chain is not None:
        for bset in shared.boundary_sets[mark:]:
            if isinstance(bset, BoundarySet) and bset.net_from_gross is None:
                bset.net_from_gross = chain


def stamp_boundary_sets(shared, mark, name):
    """Name the registrations made since `mark`, the same tail-since-a-mark idiom as the claim.

    A pricer knows nothing about the tree it is in, so the WALK names it: the deal loop stamps each
    deal as it prices, and the structure stamps whatever its own `post_process` added. Innermost
    wins - already-named sets are left alone.
    """
    for bset in shared.boundary_sets[mark:]:
        if bset.deal is None:
            bset.deal = name


@dataclass
class MTABoundaryEvent:
    """One margin call's transfer decision, recorded so its derivative can be recovered.

    A minimum transfer amount makes the collateral balance jump discontinuously, so ordinary AAD
    differentiates the netting set with the decision FROZEN. `gap` is the margin the decision was
    made by and RETAINS its graph - built from the whole netting-set MTM, so its derivative already
    carries every shared-factor and cross-deal effect. The balances are DETACHED: they seed a replay
    whose result is a coefficient, not a differentiated quantity.
    """
    call_index: int
    side: str                       # 'receive' | 'post'
    gap: object                     # tensor, graph retained
    previous_balance: object        # tensor, detached
    required_balance: object        # tensor, detached
    # Per scenario, whether THIS call is the one its run of held balance registers - stamped by
    # `mark_binding_calls` over the whole series, no call being able to decide it alone.
    live: object = None


def mark_binding_calls(events):
    """Stamp each event's `live` mask: one registration per run of calls the balance is HELD across,
    receive and post separately.

    A constant `previous_balance` means every call in the run publishes the same transfer decision,
    and the balance can make that transfer once. The binding call is the one whose gap is largest;
    ties keep the first. Where every call transfers, each run is one call and the mask is all True.
    """
    for side in sorted({event.side for event in events}):
        _mark_side([event for event in events if event.side == side])


def _mark_side(events):
    """One side's binding calls: the largest gap in each run of held balance, ties to the first."""
    if not events:
        return
    previous = torch.stack([event.previous_balance for event in events])
    gaps = torch.stack([event.gap.detach() for event in events])
    new_run = torch.ones_like(previous, dtype=torch.bool)
    new_run[1:] = previous[1:] != previous[:-1]

    # the run's largest gap at every one of its events: a forward running maximum, then swept back
    run_max = torch.empty_like(gaps)
    running = torch.full_like(gaps[0], float('-inf'))
    for k in range(gaps.shape[0]):
        running = torch.where(new_run[k], gaps[k], torch.maximum(running, gaps[k]))
        run_max[k] = running
    for k in range(gaps.shape[0] - 2, -1, -1):
        run_max[k] = torch.where(new_run[k + 1], run_max[k], run_max[k + 1])

    keep = gaps == run_max
    taken = torch.zeros_like(keep[0])
    for k, event in enumerate(events):
        taken = torch.where(new_run[k], torch.zeros_like(taken), taken)
        keep[k] = keep[k] & ~taken
        taken = taken | keep[k]
        event.live = keep[k]
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug('MTA BINDING side=%s events=%d runs=%.4g live=%.4g', events[0].side,
                      len(events), float(new_run.to(gaps.dtype).sum(dim=0).max()),
                      float(keep.to(gaps.dtype).sum(dim=0).max()))


# Custom Exceptions
class InstrumentExpired(Exception):
    def __init__(self, message):
        self.message = message


class ScheduleLifecycleError(Exception):
    """A schedule was touched out of order: priced before `bind` copied it to the device, or edited
    after. Either says the calculation did not reach it, which is a framework fault rather than a
    property of the deal."""


class UnpriceableSchedule(Exception):
    """An authored schedule the engine has no number for, refused BY NAME where it is read.

    The deal loads and its fields are the fields the schema declares; what is missing is a quantity
    the AUTHOR did not state and no rule recovers, so the answer is the name of the thing and the
    remedy, never a guess. `make_float_cashflows`' zero-length rate window is the first: a reset
    whose rate start equals its rate end has no forward rate to read, and no `Row` declares the
    tenor that would define one. A schedule saying too MUCH is refused here too: a consequence
    field beside the observations that already fold to it (`refuse_consequence_field`).

    FATAL by `is_fatal_pricing_error`, which is the whole point: a compile guard's canonical
    response is to log and increment `Deals Skipped`, and a skipped deal marks at nothing while the
    job SUCCEEDS. A refusal that turns into a zero mark is not a refusal, so this re-raises out of
    both guards."""


class SecondOrderRefused(Exception):
    """A second-order block will not be answered on this portfolio, because the answer would be a
    plausible wrong number rather than a failure.

    Three sites raise it: `Base_Revaluation.execute` refuses `Greeks: 'All'` on a registered
    `BoundarySet`; `Credit_Monte_Carlo.execute` refuses `Hessian: 'Yes'` on the same thing, and
    `pricing.exposure_kink_term` on a reporting row whose bandwidth ladder DIVERGES.

    Named so a caller can FALL BACK rather than lose the run: the value and the first-order block
    are unaffected, so re-running at `Greeks: 'First'` / `Hessian: 'No'` keeps everything except the
    thing refused. A blanket `except` cannot tell that from any other valuation exception."""


class CalibrationStale(Exception):
    """`Quote_Propagation: 'Linear'` will not carry this tick, for one of the two reasons a ride
    can fail to be a number (`InterestRateCurveParameters.propagate`): the ridden theta no longer
    reprices the set's own benchmarks inside its `Drift_Tolerance`, or NO ARTIFACT answers to the
    plan at all - a cold process, an evicted slot, a re-authored strip.

    Both refuse rather than falling back: the fallback is a plausible wrong curve and the replay
    tuple cannot tell it from the right one (two runs agreeing on `plan_hash`, `values_hash`, the
    version and the seed disagreed by 13.4% on a mark once one lost its artifact).

    Named so a caller can REFIT rather than lose the run: no number has been reported and the
    artifact is simply missing or older than the move it was asked to carry, so
    `Config.bootstrap()` publishes a fresh one under the same slot and the same EXECUTE runs."""


def is_fatal_pricing_error(e):
    """Exceptions a deal-level guard must NOT swallow into a scalar-0 mark: the machine running out
    of memory, a schedule the calculation never bound, and a schedule the engine refused BY NAME.

    The first two say the FRAMEWORK is wrong rather than the deal, and each produces a silently
    missing mark if caught — inside an inner-MC fork a missing tradable mark reads as an expired
    contract and retires the instrument from the hedge set. `UnpriceableSchedule` says the DOCUMENT
    is wrong, and a named refusal swallowed into a zero mark on a job that then succeeds has said
    nothing at all. Everything else keeps the canonical skip.

    Read by all four guards over a deal — `Deal.calculate`, `Deal.build_features` and both compile
    guards in `DealStructure` — so one predicate decides everywhere the answer would otherwise be a
    quiet zero."""
    return isinstance(e, (MemoryError, torch.cuda.OutOfMemoryError, ScheduleLifecycleError,
                          UnpriceableSchedule)) or (
        isinstance(e, RuntimeError) and 'out of memory' in str(e).lower())


def log_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError:
            # we are okay to pass keyerrors back to the calling code
            raise
        except Exception:
            # Log the exception with traceback and context
            logging.exception("An error occurred in function '%s'", func.__name__)
            # Re-raise the exception
            raise

    return wrapper


# Defined types - things like percentages, basis points etc.

class Descriptor:
    """Useful for arbitrary storage values"""

    def __init__(self, value):
        self.data = value
        self.descriptor_type = 'X'

    def __str__(self):
        return self.descriptor_type.join([str(x) for x in self.data])


class Scaled(object):
    """A number authored in one unit and stored in another; `divisor` is the whole difference.

    Percent and Basis stay distinct NAMES because the JSON encoder picks its tag by isinstance, and
    are siblings rather than parent/child so neither is an instance of the other.
    """
    divisor = 1.0
    suffix = ''

    def __init__(self, amount):
        self.amount = amount / self.divisor

    def __str__(self):
        return '%g%s' % (self.amount * self.divisor, self.suffix)

    def __float__(self):
        return self.amount

    def __lt__(self, other):
        return self.amount < other.amount

    def __eq__(self, other):
        return self.amount == other.amount if isinstance(other, type(self)) else NotImplemented

    def __hash__(self):
        return hash(self.amount)

    def __mul__(self, other):
        return self.amount * other

    def __add__(self, other):
        return self.amount + other

    def __repr__(self):
        return str(self)

    __rmul__ = __mul__


class Percent(Scaled):
    divisor, suffix = 100.0, '%'


class Basis(Scaled):
    divisor, suffix = 10000.0, ' bp'


class Curve:
    def __init__(self, meta, data):
        self.meta = meta
        self.array = array_type(sorted(data)) if isinstance(data, list) else data

    def __str__(self):
        def format1darray(data):
            return '(%s)' % ','.join(['%.12g' % y for y in data])

        array_rep = format1darray(self.array) if len(self.array.shape) == 1 else ','.join(
            [format1darray(x) for x in self.array])
        meta_rep = ','.join([str(x) for x in self.meta])
        return '[%s,%s]' % (meta_rep, array_rep) if meta_rep else '[%s]' % array_rep


class Offsets:
    lookup = {'months': 'm', 'days': 'd', 'years': 'y', 'weeks': 'w'}

    def __init__(self, data):
        self.grid = isinstance(data[0], list)
        self.data = data

    def __str__(self):
        ofs_fmt = lambda ofs: ''.join(['%d%s' % (v, Offsets.lookup[k]) for k, v in ofs.kwds.items()])
        if self.grid:
            periods = [ofs_fmt(value[0]) if len(value) == 1 else '{0}({1})'.format(*map(ofs_fmt, value)) for value in
                       self.data]
            return '{0}'.format(' '.join(periods))
        else:
            periods = [ofs_fmt(value) for value in self.data]
            return '[{0}]'.format(','.join(periods))


class DateList:
    def __init__(self, data):
        self.data = dict(data)
        self.dates = set()

    def __str__(self):
        return '\\'.join(
            ['%s=%.12g' % ('%02d%s%04d' % (x[0].day, calendar.month_abbr[x[0].month], x[0].year), x[1]) for x in
             self.data.items()]) + '\\'

    def sum_range(self, run_date, cuttoff_date):
        return sum([val for date, val in self.data.items() if run_date > date > cuttoff_date], 0.0)

    def prepare_dates(self):
        self.dates = set(self.data.keys())

    def consume(self, cuttoff, date):
        datelist = set([x for x in self.dates if x >= cuttoff]) if cuttoff else self.dates
        if datelist:
            closest_date = min(datelist, key=lambda x: np.abs((x - date).days))
            if closest_date <= date:
                self.dates.remove(closest_date)
            return closest_date, self.data[closest_date]
        else:
            return None, 0.0


class CreditSupportList:
    def __init__(self, data):
        self.data = dict(data)

    def value(self):
        return next(iter(self.data.values()))

    def __str__(self):
        return '\\'.join(['%d=%.12g' % (rating, amount) for rating, amount in self.data.items()]) + '\\'


class DateEqualList:
    def __init__(self, data):
        self.data = {x[0]: x[1:] for x in data}

    def value(self):
        return self.data.values()

    def get(self, field):
        return self.data.get(field)

    def sum_range(self, run_date, cuttoff_date, index):
        return sum([val[index] for date, val in self.data.items() if run_date > date > cuttoff_date], 0.0)

    def __str__(self):
        return '[' + ','.join(['%s=%s' % (
            '%02d%s%04d' % (date.day, calendar.month_abbr[date.month], date.year), '='.join([str(y) for y in value]))
                               for date, value in self.data.items()]) + ']'


def select_rows(operand, pos):
    """Row subset of a per-time-row operand (indices, interp weights, tenors, alpha) for a routed
    group. A leading dim of 1 is broadcasting against the time axis and must be left alone."""
    return operand if pos is None or not torch.is_tensor(operand) or operand.shape[0] == 1 \
        else operand[pos]


class ScenarioBlock(object):
    """One physical scenario tensor and where it sits in the LOGICAL grid.

    `first_row` is the block's first logical scenario row. `batch_index` maps each logical batch
    column to the column of THIS block that supplies it — `None` when the block is already at the
    logical width. An inner-MC fork's realized past holds one outer column per `Inner_Sub_Batch`
    flat columns, so its map is the flattening the fork performed: passing it as data is what stops
    the two ends having to agree by arithmetic.
    """

    def __init__(self, tensor, first_row=0, batch_index=None):
        self.tensor = tensor
        self.first_row = first_row
        self.batch_index = batch_index
        self.n_rows = tensor.shape[0]

    def project(self, val):
        """A read at this block's width, taken up to the logical grid's.

        Applied to the RESULT of a read, never to the stored tensor: projecting the tensor would
        materialize the block at the logical width and hand back the memory the block split exists
        to save. A batch-axis gather, so it commutes with the time blend and with `combine`."""
        return val if self.batch_index is None else val.index_select(-1, self.batch_index)

    def __mul__(self, other):
        return ScenarioBlock(self.tensor * other, self.first_row, self.batch_index)


class ScenarioSource(object):
    """A factor's scenario grid as the pricer sees it: a SEQUENCE of `ScenarioBlock`s under one
    logical shape, each at its own batch width.

    Ordinary generation publishes a bare tensor and never builds one of these. An inner-MC fork
    publishes TWO blocks: the outer-realized past at `Batch_Size`, then the forked rows at
    `Batch_Size x Inner_Sub_Batch`. Every past row is identical across the inner draws, so joining
    them into one tensor writes the realized past out `Inner_Sub_Batch` times — 98% of the stuffed
    buffer at the production operating point.

    Write-once and read-only: built after every process's `generate` has published, and carrying
    only the operations `make_curve_tensor` performs on a raw buffer value, so anything else fails
    loud rather than silently materializing.
    """

    def __init__(self, *blocks):
        self.blocks = blocks
        self.cuts = np.cumsum([b.n_rows for b in blocks[:-1]], dtype=np.int64)
        self.shape = (sum(b.n_rows for b in blocks),) + tuple(blocks[-1].tensor.shape[1:])

    def new(self, *args, **kwargs):
        return self.blocks[-1].tensor.new(*args, **kwargs)

    def __mul__(self, other):
        # the LinearRT/HermiteRT tenor rescale — elementwise over the tenor axis, so per block
        return ScenarioSource(*[b * other for b in self.blocks])


class Interpolation(object):
    """Tenor and time interpolation over ONE physical scenario tensor.

    A leaf, and the only class base valuation / credit Monte Carlo / the outer hedge loop ever
    build. `build` prepares what a given interpolation kind stores — an RT kind folds the tenor into
    the values, Hermite derives its coefficient pair. Dividend curves are plain here; what makes
    them different lives in `CurveTenor`.

    It knows nothing about inner MC, block boundaries, logical rows or batch fan-out: the rows
    reaching it are already in its own frame, and it flattens them against its OWN tenor stride,
    which is what lets a tenor segment be the same kind of object.
    """

    def __init__(self, tensor, interp_params):
        self.tensor = tensor
        self.shape = tuple(tensor.shape)
        self.indexed_tensor = tensor.reshape(-1, tensor.shape[-1])
        self.interp_params = [p.reshape(-1, p.shape[-1]) for p in interp_params]

    @classmethod
    def build(cls, tensor, kind, tenor):
        """What an interpolation of `kind` stores: the values, and whatever it derives from them.
        Rate*time folds the tenor into the values; Hermite derives its coefficient pair."""
        if kind == 'Linear':
            return cls(tensor, [])
        t = tensor.new(tenor[:tensor.shape[1]]).reshape(1, -1, 1)
        if kind in ('Hermite', 'HermiteRT'):
            values = tensor * t if kind == 'HermiteRT' else tensor
            return cls(values, hermite_interpolation_tensor(t, values))
        if kind == 'LinearRT':
            return cls(tensor * t, [])
        return cls(tensor, [])

    def route(self, index, has_alpha):
        """A leaf IS the whole grid — there is nothing to route."""
        return None

    def read_at(self, tenor_data, rows, i1, i2, w2):
        """The RAW value at one time point — before the rate*time scaling, which `combine` applies
        after any time blend.

        Scenario rows are flattened into this tensor's (row, tenor) frame HERE, against its OWN
        stride, which is why `CurveTensor` hands out rows rather than a flat offset. `rows is None`
        means every row is row 0 — a static curve, or a stochastic one gathered only at the base
        date — and skips the add entirely."""
        base = None if rows is None else rows.reshape(-1, 1) * self.shape[1]
        i0, i1x = (i1, i2) if base is None else (base + i1, base + i2)
        if tenor_data[0].startswith('Hermite'):
            g, c = self.interp_params
            return calc_hermite_curve(
                w2, g[i0,], c[i0,], self.indexed_tensor[i0,], self.indexed_tensor[i1x,])
        # default to linear
        return self.indexed_tensor[i0,] * (1.0 - w2) + self.indexed_tensor[i1x,] * w2

    def blend(self, raw, nxt, alpha):
        """Linear time interpolation between two raw reads."""
        return (1 - alpha) * raw + alpha * nxt

    def project(self, block, raw):
        """A raw read taken up to the logical batch width by the block that produced it."""
        return block.project(raw)

    def combine(self, raw, tenor_data, i2, tnr, time_factor):
        """Raw read -> curve value: the rate*time scaling this kind asks for. Elementwise in `raw`,
        so it commutes with the time blend and with a block projection — which is why it runs ONCE,
        after both, rather than inside either."""
        kind, tnr_min, tnr_max = tenor_data
        tenors = tnr.unsqueeze(-1)
        mult = tenors if time_factor else 1.0
        if kind.endswith('RT'):
            mult = mult / tenors.clamp(tnr_min, tnr_max)
        return raw * mult

    def eval(self, tenor_data, index, index_next, alpha, i1, i2, w2, tnr, time_factor, route=None):
        raw = self.read_at(tenor_data, index, i1, i2, w2)
        if alpha is not None:
            # the t+1 read is taken BEFORE either weighting, so no full-width term is held across it
            raw = self.blend(raw, self.read_at(tenor_data, index_next, i1, i2, w2), alpha)
        return self.combine(raw, tenor_data, i2, tnr, time_factor)

    def gather_rows(self, index, index_next, alpha, route=None):
        """Whole rows at `index` — the 0D spot path."""
        if alpha is None:
            return self.tensor[index]
        return self.tensor[index] * (1 - alpha) + self.tensor[index_next] * alpha


class SegmentedInterpolation(object):
    """A curve whose tenor axis is split at a near index, each side interpolated its own way
    (`Near_Interpolation`). A SIBLING of `Interpolation`, not a subclass: it composes leaves over
    TENOR as `RoutedInterpolation` composes strategies over SCENARIO ROWS, and the two compositions
    are orthogonal.

    Segments are middle-dim slices with their own tenor divisors, so each owns its own flat stride —
    the reason `CurveTensor` hands out scenario ROWS and lets the strategy flatten them.
    """

    def __init__(self, tensor, spec, tenor):
        self.tensor = tensor
        self.shape = tuple(tensor.shape)
        self.indexed_tensor = tensor.reshape(-1, tensor.shape[-1])
        self.spec = spec
        # this only works for 2 segments - checked when the factor is built
        self.cutoff = spec[0][1]
        self.segments = [Interpolation.build(tensor[:, s:e + 1, :], kind, tenor[s:e + 1])
                         for s, e, kind in spec]


    def route(self, index, has_alpha):
        return None

    def seg_tenors(self, seg_i, i1, i2):
        """`i1, i2` in segment `seg_i`'s own tenor frame."""
        s, e, _kind = self.spec[seg_i]
        if seg_i == 0:
            return i1.clamp(max=e), i2.clamp(max=e)
        return (i1 - s).clamp(min=0), (i2 - s).clamp(min=0)

    def read_at(self, tenor_data, rows, i1, i2, w2):
        """One raw read PER SEGMENT — evaluated on the full tenor set and selected in `combine`.
        More work than needed, but it keeps every segment a plain leaf."""
        return [seg.read_at((seg_spec[2], tnr_min, tnr_max), rows,
                            *self.seg_tenors(k, i1, i2), w2)
                for k, (seg, (seg_spec, tnr_min, tnr_max))
                in enumerate(zip(self.segments, zip(*tenor_data)))]

    def blend(self, raw, nxt, alpha):
        return [seg.blend(a, b, alpha) for seg, a, b in zip(self.segments, raw, nxt)]

    def project(self, block, raw):
        return [seg.project(block, v) for seg, v in zip(self.segments, raw)]

    def combine(self, raw, tenor_data, i2, tnr, time_factor):
        """Each segment's own scaling, then the tenor select between them. `tenor_data` is the
        `(spec, (min, split), (split, max))` triple, so `zip(*tenor_data)` is one segment's
        `((start, end, kind), tnr_min, tnr_max)`."""
        vals = [seg.combine(v, (seg_spec[2], tnr_min, tnr_max),
                            self.seg_tenors(k, i2, i2)[1], tnr, time_factor)
                for k, (seg, v, (seg_spec, tnr_min, tnr_max))
                in enumerate(zip(self.segments, raw, zip(*tenor_data)))]
        return torch.where((i2 <= self.cutoff).unsqueeze(-1), vals[0], vals[1])

    def eval(self, tenor_data, index, index_next, alpha, i1, i2, w2, tnr, time_factor, route=None):
        raw = self.read_at(tenor_data, index, i1, i2, w2)
        if alpha is not None:
            raw = self.blend(raw, self.read_at(tenor_data, index_next, i1, i2, w2), alpha)
        return self.combine(raw, tenor_data, i2, tnr, time_factor)

    def gather_rows(self, index, index_next, alpha, route=None):
        if alpha is None:
            return self.tensor[index]
        return self.tensor[index] * (1 - alpha) + self.tensor[index_next] * alpha


class RoutedInterpolation(object):
    """One logical scenario grid over several physical blocks — an inner-MC fork's realized past and
    its forked rows — each carrying its OWN interpolation, built recursively from the same curve
    tenor. A segmented curve inside a fork is a `RoutedInterpolation` of `SegmentedInterpolation`s
    and needs no special case here.

    It owns exactly the composite concerns: which block holds a row, rebasing a logical row into
    that block's frame, projecting a narrow block's read up to the logical batch width, and
    reassembling the groups in the caller's row order. The interpolations stay unaware of it.
    """

    def __init__(self, source, curve_tenor):
        self.blocks = source.blocks
        self.shape, self.cuts = source.shape, source.cuts
        self.strategies = tuple(build_interpolation(b.tensor, curve_tenor) for b in source.blocks)
        # the last block is the one already at the logical batch width, so it answers the
        # tensor-shaped questions a leaf answers for itself
        self.tensor = source.blocks[-1].tensor
        self.indexed_tensor = self.strategies[-1].indexed_tensor

    def route(self, index, has_alpha):
        """Group a gather's rows by the block that owns each of its two reads: `(row positions,
        block for the t read, block for the t+1 read)`, positions `None` when ONE group covers every
        row. A time-interpolated read reaches `index + 1`, so a row just below a cut reads ACROSS it
        and names two blocks — classify on where a read ENDS, not where it starts. Decided from the
        numpy indices a `CurveTensor` already holds, so it costs no device sync."""
        hi = np.minimum(index + 1, self.shape[0] - 1) if has_alpha else index
        at_t = np.searchsorted(self.cuts, index, side='right')
        at_t1 = np.searchsorted(self.cuts, hi, side='right')
        # an empty gather (a step with no resets in range) names no rows: one group, empty
        pairs = np.unique(np.stack([at_t, at_t1]), axis=1) if index.size else np.zeros((2, 1), int)
        if pairs.shape[1] == 1:
            return ((None, int(pairs[0, 0]), int(pairs[1, 0])),)
        return tuple((torch.tensor(np.flatnonzero((at_t == t0) & (at_t1 == t1)),
                                   dtype=torch.int64, device=self.tensor.device), int(t0), int(t1))
                     for t0, t1 in pairs.T)

    def routed(self, route, n_rows, read):
        """Run `read` per routed group and put the groups back in the caller's row order. A group
        covering every row answers directly."""
        out = None
        for pos, at_t, at_t1 in route:
            val = read(pos, at_t, at_t1)
            if pos is None:
                return val
            out = val.new_empty((n_rows,) + tuple(val.shape[1:])) if out is None else out
            out.index_copy_(0, pos, val)
        return out

    def local(self, rows, block):
        """Logical scenario rows in `block`'s own frame."""
        return rows if rows is None or not block.first_row else rows - block.first_row

    def eval(self, tenor_data, index, index_next, alpha, i1, i2, w2, tnr, time_factor, route):
        def group(pos, at_t, at_t1):
            b0, b1 = self.blocks[at_t], self.blocks[at_t1]
            s0, s1 = self.strategies[at_t], self.strategies[at_t1]
            rows = self.local(select_rows(index, pos), b0)
            weight, t1, t2 = (select_rows(x, pos) for x in (w2, i1, i2))
            nxt = None if index_next is None else self.local(select_rows(index_next, pos), b1)
            # the read is per block, but the SPEC is the curve's
            raw = s0.project(b0, s0.read_at(tenor_data, rows, t1, t2, weight))
            if alpha is not None:
                # projection and the time blend are both linear and `combine` runs after both, so
                # the routed path is the same arithmetic in the same order as an unrouted one
                raw = s0.blend(raw, s1.project(b1, s1.read_at(tenor_data, nxt, t1, t2, weight)),
                               select_rows(alpha, pos))
            return s0.combine(raw, tenor_data, t2, select_rows(tnr, pos), time_factor)

        return self.routed(route, (i1 if index is None else index).shape[0], group)

    def gather_rows(self, index, index_next, alpha, route):
        """Whole-row gather — the 0D spot path. The same block routing as `eval`, on the
        scenario-row axis rather than the flattened (row, tenor) one."""
        def group(pos, at_t, at_t1):
            b0, b1 = self.blocks[at_t], self.blocks[at_t1]
            if alpha is None:
                return b0.project(
                    self.strategies[at_t].tensor[self.local(select_rows(index, pos), b0)])
            a = select_rows(alpha, pos)
            return b0.project(
                self.strategies[at_t].tensor[self.local(select_rows(index, pos), b0)]) * (1 - a) + \
                b1.project(
                    self.strategies[at_t1].tensor[
                        self.local(select_rows(index_next, pos), b1)]) * a

        return self.routed(route, index.shape[0], group)


def build_interpolation(value, curve_tenor):
    """The one constructor for a curve's interpolation, recursive in the scenario axis.

        bare tensor    + a kind string  -> Interpolation
        bare tensor    + a segment list -> SegmentedInterpolation
        ScenarioSource + either         -> RoutedInterpolation, whose per-block children are built
                                           by this function again

    So a segmented curve inside an inner-MC fork composes rather than special-cases."""
    if isinstance(value, ScenarioSource):
        return RoutedInterpolation(value, curve_tenor)
    if isinstance(curve_tenor.type, str):
        return Interpolation.build(value, curve_tenor.type, curve_tenor.tenor)
    return SegmentedInterpolation(value, curve_tenor.type, curve_tenor.tenor)


class CurveTenor(object):
    def __init__(self, tenor_points, interp):
        # linear interpolation by default
        points = np.array(tenor_points)
        min_tenor = points.min()
        max_tenor = points.max()
        # check that dividends are defined >0
        if interp == 'Dividend':
            tenor_delta = (1.0 / np.array(tenor_points[:-1]).clip(1e-5, np.inf)) - \
                          (1.0 / np.array(tenor_points[1:]).clip(1e-5, np.inf))
            min_tenor = max(1e-5, min_tenor)
            max_tenor = max(1e-5, max_tenor)
        else:
            tenor_delta = np.diff(points)

        self.tenor = points
        self.delta = np.append(tenor_delta, 1.0)
        self.type = interp
        self.min = min_tenor
        self.max = max_tenor
        self.max_index = max(points.shape[0] - 1, 0)
        self.tensor_cache = {}

    def get_index(self, tenor_points_in_years):
        if isinstance(tenor_points_in_years, torch.Tensor):
            clipped_points = tenor_points_in_years.clip(self.min, self.max)
            if not self.tensor_cache:
                self.tensor_cache['tenor'] = tenor_points_in_years.new(self.tenor)
                self.tensor_cache['delta'] = tenor_points_in_years.new(self.delta)
            tenor = self.tensor_cache['tenor']
            delta = self.tensor_cache['delta']
            index = torch.searchsorted(tenor, clipped_points, right=True) - 1
        else:
            clipped_points = np.clip(tenor_points_in_years, self.min, self.max)
            tenor = self.tenor
            delta = self.delta
            index = tenor.searchsorted(clipped_points, side='right') - 1

        if 'Extrapolate' in self.type:
            # extend the end segments: clamp the index to a real segment and leave alpha
            # unclipped - a linear blend with alpha outside [0, 1] IS linear extrapolation
            index = index.clip(0, max(self.max_index - 1, 0))
            index_next = (index + 1).clip(0, self.max_index)
            alpha = (tenor_points_in_years - tenor[index]) / delta[index]
            return index, index_next, alpha

        index_next = (index + 1).clip(0, self.max_index)

        if self.type == 'Dividend':
            alpha = (1.0 / tenor[index].clip(min=1e-5) -
                     1.0 / clipped_points) / delta[index]
        else:
            alpha = (clipped_points - tenor[index]) / delta[index]

        return index, index_next, alpha


@torch.jit.script
class Calculation_State(object):
    """
    Note that all pricing functions depend on this class being correctly setup. All calculations
    should inherit from this calculation state and extend accordingly
    """

    def __init__(self, static_buffer, unit, mcmc_sims, report_currency: List[Tuple[bool, int]],
                 nomodel: str, simulation_batch: int, keep_tensor: bool):
        # these are tensors
        self.t_Buffer = {}
        self.t_Static_Buffer = static_buffer
        # storing a unit tensor allows the dtype and device to be encoded in the calculation state
        self.one = unit
        self.fillvalue = unit.new_zeros((0, 1, simulation_batch))
        self.simulation_batch = simulation_batch
        self.Report_Currency = report_currency
        self.t_Cashflows = None
        # these are shared parameter states
        self.riskneutral = nomodel == 'RiskNeutral'
        self.MCMC_sims = mcmc_sims
        # keep individual calculation results per dependency?
        self.keep_tensor = keep_tensor
        # Recompute a Monte Carlo pricer's inner simulation in backward() rather than taping it
        # (`Recompute_Inner_MC`); off is the taped path bit for bit. Declared here rather than by
        # the calculations that set it, so every pricer reads it without a fallback.
        self.recompute_inner_mc = False
        # Second derivatives are wanted (`Greeks: 'All'`, base valuation only), so the reverse
        # sweep runs with `create_graph`. Declared here for the same reason as the switch above.
        self.gamma = False
        # The smooth (branch-and-weight) value estimator is wanted (`Branch_And_Weight`, base
        # valuation only); off is the crisp one-step-survival path bit for bit. False here is what
        # keeps `Credit_Monte_Carlo`, which declares no such field, on the crisp estimator.
        self.branch_and_weight = False
        # THE STRIDE is wanted (`HN_Stride`, base valuation only): the component Heston-Nandi
        # k-step conditional law in place of the daily walk between fixings. ONE field governs every
        # consumer, because all consent to the SAME declared approximation - the carried state
        # across the jump. Not a speed lever (`pricing.ComponentHestonNandiKit.substeps`); off is
        # the daily walk bit for bit.
        self.hn_stride = False
        # where the memoized quasi-random stream stands, per (dimension, sample_size) - only
        # `CMC_State.quasi_rng` advances it, but `rng_position` seeks every state's streams
        self.t_quasi_rng_batch = {}


def rng_position(shared, position=None):
    """Where every random stream a calculation draws from STANDS, and optionally a seek.

    Returns the position it was at, and seeks to `position` FIRST if one is given - so one call both
    rewinds and records where to rewind back to, the whole idiom a recompute needs
    (`pricing.InnerMCRecompute`). A free function because `Calculation_State` is
    `torch.jit.script`ed and none of this compiles.

    Two streams reach a pricer, both read inside one inner Monte Carlo. Sobol draws are MEMOIZED, so
    their position is a counter per (dimension, sample_size) and seeking it makes the next draw
    return the very same tensor - the replay is exact by identity. The regular generator has no
    memo, so its position is its own state; `torch.rand` and the Heston-Nandi unmonitored sub-steps
    both draw from it.

    The device generator is only asked for its state on a device, which copies back to the host and
    synchronises - hence once per pricing block.
    """
    was = (dict(shared.t_quasi_rng_batch), torch.get_rng_state(),
           torch.cuda.get_rng_state(shared.one.device) if shared.one.is_cuda else None)
    if position is not None:
        counters, cpu_state, device_state = position
        shared.t_quasi_rng_batch = dict(counters)
        torch.set_rng_state(cpu_state)
        if device_state is not None:
            torch.cuda.set_rng_state(device_state, shared.one.device)
    return was


# often we need a numpy array and its tensor equivalent at the same time
class DualArray:
    def __init__(self, tensor, ndarray):
        self.np = ndarray
        self.tn = tensor

    def __getitem__(self, x):
        return DualArray(self.tn[x], self.np[x])


def bind_schedules(compiled, unit):
    """Bind every `TensorSchedule` reachable in a deal's compiled `Factor_dep`, and return it.

    A deal files its schedules under whatever key it likes and nests them, so this reads the
    compiled output rather than a list of key names. It WRAPS `calc_dependencies` on the walk that
    already builds the deal tree: binding is part of compiling, not a second pass over it.
    """
    if isinstance(compiled, TensorSchedule):
        compiled.bind(unit)
    else:
        for value in (compiled.values() if isinstance(compiled, dict) else
                      compiled if isinstance(compiled, (list, tuple)) else ()):
            bind_schedules(value, unit)
    return compiled


# Tensor specific classes that's used internally
class TensorSchedule(object):
    """A cashflow or reset schedule: numpy while it compiles, a device tensor once it is BOUND.

    The DUAL representation is the point: index columns are checked in fast numpy (`np.unique`,
    `searchsorted`, masks) and only the copy the arithmetic runs on goes to the device. `carry` is
    the one seam where the tensor half can differ from that copy.

    `bind` separates the two halves in TIME, and is the whole lifecycle. Before it the numpy half is
    authoritative and the compile-time edits are the only writes; after it the device copy is
    authoritative, every accessor serves that ONE copy, and an edit raises rather than silently
    failing to reach it. `derived` is the run-scoped home for anything a pricer builds off the copy.
    """

    def __init__(self, schedule, offsets):
        self.schedule = np.array(schedule)
        self.offsets = np.array(offsets)
        #: the bound tensor half, and the tensors pricers derive from it - both minted by `bind`
        self.bound = None
        self.derived = {}
        self.unit = None
        self.overlay = None

    def __repr__(self):
        return '{}({} rows)'.format(type(self).__name__, self.schedule.shape[0])

    def bind(self, unit):
        """Copy the numpy half to the device and make that copy authoritative - the lifecycle event.

        Whatever overlay `carry` attached is spliced in HERE, and re-binding is how a second run
        starts clean: a fresh copy for the new unit, every tensor derived from the old one dropped.
        """
        array = self._bound_array()
        self.unit = unit
        self.bound = DualArray(self._spliced(unit.new_tensor(array)), array)
        self.derived = {}
        return self

    def _bound_array(self):
        """The numpy half the device copy is made of - schedule and offsets side by side."""
        return np.concatenate((self.schedule, self.offsets), axis=1)

    def _compiling(self):
        """Refuse a compile-time edit once bound: the device copy is what the arithmetic reads, so
        an edit to the numpy half here would not reach it."""
        if self.bound is not None:
            raise ScheduleLifecycleError(
                '{} is bound - compile-time edits must run before bind'.format(self))

    def reopen(self):
        """Drop the binding so a compile can CONTINUE - re-bind when it has. The one caller is
        `MtMCrossCurrencySwapDeal.post_process`, which cannot know which child carries the MtM leg
        until the children exist."""
        self.bound = None
        return self

    def carry(self, overlay):
        """Attach differentiable VALUE columns, `{column: tensor}`, to the tensor half.

        A schedule's value columns - a fixed rate, a margin - are where a market QUOTE lands, and
        `new_tensor` copies them across as data, which is where a quote stops being differentiable.
        An overlay column is spliced into that copy instead, so a quote reaches a pricer with its
        graph intact. Index columns are never overlaid, being read off `.np`, so every existing
        consumer is untouched.
        """
        self._compiling()
        self.overlay = overlay
        return self

    def _spliced(self, tensor):
        """`tensor` with the overlay's columns replaced, out of place so the graph survives."""
        if not self.overlay:
            return tensor
        return tensor.index_copy(
            1, torch.tensor(list(self.overlay), dtype=torch.int64, device=tensor.device),
            torch.stack(list(self.overlay.values()), dim=1))

    def __getitem__(self, x):
        return self.schedule[x]

    def __len__(self):
        return len(self.schedule)

    def dual(self, index=0):
        """The bound dual from row `index` on."""
        if self.bound is None:
            raise ScheduleLifecycleError('{} was never bound to a calculation'.format(self))
        return self.bound[index:]

    def declared_values(self, index=RESET_INDEX_Value, filter_index=RESET_INDEX_Reset_Day):
        """The already-fixed rows' `index` column as plain scalars, in schedule order."""
        return [x[index] for x in self.schedule if x[filter_index] < 0.0]

    def known_resets(self, num_scenarios, index=RESET_INDEX_Value,
                     filter_index=RESET_INDEX_Reset_Day, include_today=False):
        """The already-fixed rows' VALUE column, one `(1, num_scenarios)` tensor each.
        `include_today` keeps a row resetting today, which only an equity reset wants."""
        key = ('known_resets', num_scenarios, index, include_today)
        if self.derived.get(key) is None:
            values = ([x[index] for x in self.schedule
                       if x[filter_index] <= 0.0 and x[index] > 0] if include_today
                      else self.declared_values(index, filter_index))
            self.derived[key] = [self.unit.new_full((1, num_scenarios), x) for x in values]
        return self.derived[key]


class DealTimeDependencies(object):
    def __init__(self, mtm_time_grid, deal_time_grid):
        self.mtm_time_grid = mtm_time_grid
        self.delta = np.hstack(((mtm_time_grid[deal_time_grid[1:]] -
                                 mtm_time_grid[deal_time_grid[:-1]]), [1]))
        self.interp = mtm_time_grid[mtm_time_grid <= mtm_time_grid[deal_time_grid[-1]]]
        self.deal_time_grid = deal_time_grid
        # store the indices for linear interpolation
        self.update_indices()

    def assign(self, time_dependencies):
        # only assign up to the max of this set of dependencies
        expiry = self.deal_time_grid[-1]
        query = time_dependencies.deal_time_grid <= expiry
        self.delta = time_dependencies.delta[query]
        self.deal_time_grid = time_dependencies.deal_time_grid[query]
        self.interp = self.mtm_time_grid[self.mtm_time_grid <= self.mtm_time_grid[expiry]]
        # store the indices for linear interpolation
        self.update_indices()

    def copy_restricted(self, cutoff_mtm_index):
        """Fresh DealTimeDependencies covering only deal events at mtm positions >=
        cutoff_mtm_index; delta/interp/indices/alpha are recomputed for the sliced view, so the
        interpolate path stays aligned with mtm_time_grid. None if every event is past the
        cutoff."""
        keep = self.deal_time_grid >= cutoff_mtm_index
        if not keep.any():
            return None
        return type(self)(self.mtm_time_grid, self.deal_time_grid[keep])

    def copy_window(self, from_mtm_index, to_mtm_index):
        """Fresh DealTimeDependencies covering only deal events at mtm positions in
        [from_mtm_index, to_mtm_index] — the one-step inner-MC fork prices at exactly {t, t+1}, so
        the AAD tape and the scenario buffer stop at t+1. Assumes hedge-mode deals reval on every
        mtm date. None if no event falls inside the window."""
        keep = (self.deal_time_grid >= from_mtm_index) & (self.deal_time_grid <= to_mtm_index)
        if not keep.any():
            return None
        return type(self)(self.mtm_time_grid, self.deal_time_grid[keep])

    def update_indices(self):
        self.index = np.searchsorted(self.deal_time_grid, np.arange(self.interp.size), side='right') - 1
        self.index_next = (self.index + 1).clip(0, self.deal_time_grid.size - 1)
        self.alpha = (np.array((self.interp - self.interp[self.deal_time_grid[self.index]]) /
                               self.delta[self.index]).reshape(-1, 1))
        self.t_alpha = None

    def fetch_index_by_day(self, days):
        return self.interp.searchsorted(days)


# calculation time grid
class TimeGrid(object):
    def __init__(self, scenario_dates, MTM_dates, base_MTM_dates):
        self.scenario_dates = scenario_dates
        self.base_MTM_dates = base_MTM_dates
        self.CurrencyMap = {}
        self.report_index = None
        self.mtm_dates = MTM_dates
        self.date_lookup = dict([(x, i) for i, x in enumerate(sorted(MTM_dates))])

    def set_report_dates(self, base_date, report_dates):
        report_days = [(x - base_date).days for x in sorted(report_dates)]
        self.report_index = (self.mtm_time_grid.searchsorted(
            report_days, side='right') - 1).clip(0, self.mtm_time_grid.size - 1)

    def calc_time_grid(self, time_in_days):
        dvt = np.concatenate(([1], np.diff(self.scen_time_grid), [1]))
        scen_index = self.scen_time_grid.searchsorted(time_in_days, side='right')
        index = (scen_index - 1).clip(0, self.scen_time_grid.size - 1)
        alpha = ((time_in_days - self.scen_time_grid[index]) / dvt[scen_index]).clip(0, 1)
        return np.dstack([alpha, time_in_days, index])[0]

    def set_base_date(self, base_date, delta=None):
        # grids in days. The scenario_dates may equal the mtm_dates - a finer margin period of risk
        # on collateralized netting sets
        self.mtm_time_grid = np.array([(x - base_date).days for x in sorted(self.mtm_dates)])
        self.scen_time_grid = np.array([(x - base_date).days for x in sorted(self.scenario_dates)])

        self.base_time_grid = set([self.date_lookup[x] for x in self.base_MTM_dates])
        self.time_grid = self.calc_time_grid(self.mtm_time_grid)

        # store the scenario time_grid
        self.scenario_grid = np.zeros((self.scen_time_grid.size, 3))
        self.scenario_grid[:, TIME_GRID_MTM] = self.scen_time_grid
        self.scenario_grid[:, TIME_GRID_ScenarioPriorIndex] = np.arange(self.scen_time_grid.size)

        # a very fine time_grid, done AFTER the scenario_grid: a non-null delta generates scenarios
        # without calculating the whole risk factor
        if delta is not None:
            delta_days, delta_tenors = delta
            delta_grid = np.union1d(np.arange(0, self.scen_time_grid.max(), delta_days), delta_tenors.round())
            self.scen_time_grid = np.union1d(self.scen_time_grid, delta_grid)

        self.time_grid_years = self.scen_time_grid / DAYS_IN_YEAR

    def get_scenario_offset(self, days_from_base):
        prev_scen_index = self.scen_time_grid[self.scen_time_grid <= days_from_base].size - 1
        scenario_grid_delta = np.float64(
            (self.scen_time_grid[prev_scen_index + 1] - self.scen_time_grid[prev_scen_index]) if (
                    self.scen_time_grid.size > 1 and self.scen_time_grid.size > prev_scen_index + 1) else 1.0)
        return (days_from_base - self.scen_time_grid[prev_scen_index]) / scenario_grid_delta, prev_scen_index

    def set_currency_settlement(self, currencies):
        self.CurrencyMap = {}
        for currency, dates in currencies.items():
            settlement_dates = sorted([self.date_lookup[x] for x in dates if x in self.date_lookup])
            if settlement_dates:
                currency_lookup = np.zeros(self.mtm_time_grid.size, dtype=np.int32) - 1
                currency_lookup[settlement_dates] = np.arange(len(settlement_dates))
                self.CurrencyMap.setdefault(currency, currency_lookup)

    def truncate_to(self, original_base_date, t_days):
        """A new TimeGrid covering [t_days, T] of the original, base shifted forward by t_days -
        the truncated horizon a nested simulation starts from at an outer timestep.
        `original_base_date` is the caller's responsibility: TimeGrid does not store its own."""
        new_base_date = original_base_date + pd.Timedelta(days=int(t_days))
        new_scenario_dates = [d for d in sorted(self.scenario_dates) if d >= new_base_date]
        new_mtm_dates = [d for d in sorted(self.mtm_dates) if d >= new_base_date]
        new_base_mtm = [d for d in self.base_MTM_dates if d in new_mtm_dates]
        new_grid = TimeGrid(new_scenario_dates, new_mtm_dates, new_base_mtm)
        if new_scenario_dates:
            new_grid.set_base_date(new_base_date)
        else:
            # Past-end caller's grid: keep `scen_time_grid` queryable (empty) so size
            # checks like `grid.scen_time_grid.size < 2` work without AttributeError.
            new_grid.scen_time_grid = np.array([], dtype=np.int64)
            new_grid.time_grid_years = np.array([], dtype=np.float64)
        return new_grid

    def calc_deal_grid(self, dates):
        try:
            dynamic_dates = self.base_time_grid.union([self.date_lookup[x] for x in dates])
        except KeyError as e:
            # if there is at least one reset date in the set of dates, then return it, else the deal has expired
            r = [self.date_lookup.get(x, max(self.date_lookup.values())) for x in dates]
            if r:
                dynamic_dates = self.base_time_grid.union(r)
            else:
                if max(dates) < min(self.date_lookup.keys()):
                    raise InstrumentExpired(e)

                # include this instrument but don't bother pricing it through time
                return DealTimeDependencies(self.mtm_time_grid, np.array([0]))

        # now construct the full deal grid
        deal_time_grid = np.array(sorted(dynamic_dates))
        # find the last dynamic date - should be the expiry date or the end of the grid
        expiry = self.date_lookup.get(max(dates), max(self.date_lookup.values()))
        # calculate the interpolation points etc.
        return DealTimeDependencies(self.mtm_time_grid, deal_time_grid[deal_time_grid <= expiry])


class TensorResets(TensorSchedule):
    def __init__(self, schedule, offsets):
        super(TensorResets, self).__init__(schedule, offsets)

        # Assign the offsets directly to the resets
        self.schedule[:, RESET_INDEX_Scenario] = self.offsets

    def _bound_array(self):
        """A reset's offset IS its scenario column, written there at construction, so the schedule
        already carries it and there is nothing to splice in."""
        return self.schedule

    def get_simulated_resets(self, max_time, forward, shared):
        within_horizon = (self.offsets > -1) & (self.schedule[:, RESET_INDEX_Reset_Day] <= max_time)
        sim_resets = self.dual()[within_horizon]
        known_resets = self.known_resets(shared.simulation_batch)
        old_resets = calc_time_grid_curve_rate(
            forward, sim_resets.np[:, :RESET_INDEX_Scenario + 1], shared)
        delta_start = (sim_resets.np[:, RESET_INDEX_Start_Day] -
                       sim_resets.np[:, RESET_INDEX_Reset_Day]).reshape(-1, 1)
        delta_end = (sim_resets.np[:, RESET_INDEX_End_Day] -
                     sim_resets.np[:, RESET_INDEX_Reset_Day]).reshape(-1, 1)
        reset_weights = (sim_resets.tn[:, RESET_INDEX_Weight] /
                         sim_resets.tn[:, RESET_INDEX_Accrual]).reshape(-1, 1, 1)

        reset_values = torch.expm1(
            old_resets.gather_weighted_curve(shared, delta_end, delta_start)) * reset_weights \
            if sim_resets.np.any() else shared.fillvalue

        # fetch all fixed resets
        return torch.squeeze(
            torch.concat(
                [shared.fillvalue if not known_resets else torch.stack(known_resets), reset_values], dim=0)
            , dim=1)

    def split_block_resets(self, reset_offset, t, date_offset=0):
        all_resets = self.schedule[reset_offset:]
        future_resets = np.searchsorted(all_resets[:, RESET_INDEX_Reset_Day] - date_offset, t)
        return future_resets

    def get_start_index(self, time_grid, offset=0):
        """Read the start index (relative to the time_grid) of each reset"""
        return np.searchsorted(self.schedule[:, RESET_INDEX_Reset_Day] - offset,
                               time_grid[:, TIME_GRID_MTM]).astype(np.int64)

    def split_groups(self, group_size):
        if self.derived.get(('groups', group_size)) is None:
            groups = []
            for i in range(group_size):
                group = TensorResets(self.schedule[i::group_size], self.offsets[i::group_size])
                groups.append(group.bind(self.unit))
            self.derived[('groups', group_size)] = groups
        return self.derived.get(('groups', group_size))


class TensorCashFlows(TensorSchedule):
    def __init__(self, schedule, offsets):
        # check which cashflows are settlements (as opposed to accumulations)
        for cashflow, next_cashflow, cash_ofs in zip(schedule[:-1], schedule[1:], offsets[:-1]):
            if (next_cashflow[CASHFLOW_INDEX_Pay_Day] != cashflow[CASHFLOW_INDEX_Pay_Day]) or (
                    cashflow[CASHFLOW_INDEX_FixedAmt] != 0):
                cash_ofs[CASHFLOW_OFFSET_Settle] = 1

        # last cashflow always settles (if it's not marked as such) otherwise, it's a forward
        if offsets[-1][CASHFLOW_OFFSET_Settle] == 0:
            offsets[-1][CASHFLOW_OFFSET_Settle] = 1

        # Add Resets field
        self.Resets = None
        # call superclass
        super(TensorCashFlows, self).__init__(schedule, offsets)

    def bind(self, unit):
        """A cashflow's resets are part of it, so they bind with it."""
        if self.Resets is not None:
            self.Resets.bind(unit)
        return super(TensorCashFlows, self).bind(unit)

    def total_abs_nominal(self):
        """Summed |notional| across the schedule."""
        return float(np.abs(self.schedule[:, CASHFLOW_INDEX_Nominal]).sum())

    def last_pay_day(self):
        """Latest payment day (offset in days from base_date)."""
        return float(self.schedule[:, CASHFLOW_INDEX_Pay_Day].max())

    def get_par_swap_rate(self, base_date, ir_curve):
        """Used to calculate the par swap rate for these cashflows given an interest rate curve"""
        Dt = ir_curve.get_day_count_accrual(base_date, self.schedule[:, CASHFLOW_INDEX_Pay_Day])
        D = np.exp(-ir_curve.current_value(Dt) * Dt) * self.schedule[:, CASHFLOW_INDEX_Year_Frac]
        if self.Resets is not None:
            T = ir_curve.get_day_count_accrual(base_date, self.Resets.schedule[:, RESET_INDEX_End_Day])
            t = ir_curve.get_day_count_accrual(base_date, self.Resets.schedule[:, RESET_INDEX_Start_Day])
            a = self.Resets.schedule[:, RESET_INDEX_Accrual]
            r = (np.exp(ir_curve.current_value(T) * T - ir_curve.current_value(t) * t) - 1.0) / a
            return (D * r).sum() / D.sum(), D.sum()
        else:
            return D.sum()

    def insert_cashflow(self, cashflow):
        """Inserts a cashflow at the beginning of the cashflow schedule - useful to model a fixed payment at the
        beginning of a schedule of cashflows"""
        self._compiling()
        self.schedule = np.vstack((cashflow, self.schedule))
        self.offsets = np.vstack(([0, 0, 1], self.offsets))

    def set_fixed_amount(self, rate):
        """sets the fixed amount to the rate provided"""
        self._compiling()
        self.schedule[:, CASHFLOW_INDEX_FixedAmt] = rate * self.schedule[:, CASHFLOW_INDEX_Nominal] * \
                                                    self.schedule[:, CASHFLOW_INDEX_Year_Frac]

    def add_maturity_accrual(self, reference_date, daycount_code):
        """Adjusts the last cashflow's daycount accrual fraction to include the maturity date"""
        self._compiling()
        last_cashflow = self.schedule[-1]
        last_cashflow[CASHFLOW_INDEX_Year_Frac] = get_day_count_accrual(
            reference_date + pd.offsets.Day(last_cashflow[CASHFLOW_INDEX_End_Day]),
            last_cashflow[CASHFLOW_INDEX_End_Day] - last_cashflow[CASHFLOW_INDEX_Start_Day] + 1, daycount_code)

    def set_resets(self, schedule, offsets):
        self._compiling()
        self.Resets = TensorResets(schedule, offsets)

    def overwrite_rate(self, attribute_index, value):
        """
        Overwrites the strike/fixed_amount/float_rate defined in the cashflow schedule
        """
        self._compiling()
        for cashflow in self.schedule:
            cashflow[attribute_index] = value

    def set_future_fx_resets(self, max_time, time_grid):
        FXResets = []
        valid = (self.schedule[:, CASHFLOW_INDEX_FXResetDate] <= max_time) & (
                self.schedule[:, CASHFLOW_INDEX_FXResetDate] >= 0)
        for cashflow in self.schedule:
            Reset_Day = cashflow[CASHFLOW_INDEX_FXResetDate]
            Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
            FXResets.append([Time_Grid, Reset_Day, Scenario])
        self.FXResets = np.array(FXResets)[valid]

    def add_mtm_payments(self, base_date, principal_exchange, effective_date, day_count):
        ''' MTM CCIRS's only need a zero marker for the nominal should the effective date be in the future '''
        if (principal_exchange in ['Start_Maturity', 'Start']) and base_date <= effective_date:
            dummy_cashflow = make_cashflow(
                base_date, base_date - pd.offsets.Day(1), effective_date,
                effective_date, 0.0, get_day_count(day_count), 0.0, 0.0)
            self.insert_cashflow(dummy_cashflow)

    def add_fixed_payments(self, base_date, principal_exchange, effective_date, day_count, principal):
        ''' Regular CCIRS's might need to exchange principle at the start and end '''
        if (principal_exchange in ['Start_Maturity', 'Start']) and base_date <= effective_date:
            self.insert_cashflow(
                make_cashflow(base_date, effective_date, effective_date, effective_date, 0.0, get_day_count(day_count),
                              -principal, 0.0))

        if principal_exchange in ['Start_Maturity', 'Maturity']:
            self._compiling()
            self.schedule[-1][CASHFLOW_INDEX_FixedAmt] = principal

    def get_cashflow_start_index(self, time_grid, field_index=CASHFLOW_INDEX_Pay_Day, last_payment=None):
        """Read the start index (relative to the time_grid) of each cashflow"""
        t_grid = time_grid[:, TIME_GRID_MTM]
        if last_payment:
            t_grid = time_grid[:, TIME_GRID_MTM].copy()
            t_grid[t_grid > last_payment] = self.schedule[:, CASHFLOW_INDEX_Pay_Day].max() + 1
        return np.searchsorted(self.schedule[:, field_index], t_grid).astype(np.int64)


def split_tensor(tensor, counts):
    return torch.split(tensor, tuple(counts)) if tensor.shape[0] == counts.sum() else [tensor] * counts.size


def split_array(array, counts):
    """`split_tensor` on the numpy side — keeps a CurveTensor's CPU scenario indices in step with its
    device ones, so a per-deal slice re-derives its row routing without a device sync."""
    return np.split(array, counts.cumsum()[:-1]) if array.shape[0] == counts.sum() \
        else [array] * counts.size


# @torch.jit.script
def calc_hermite_curve(t_a, g, c, curve_t0, curve_t1):
    one_minus_ta = (1.0 - t_a)
    return curve_t0 * one_minus_ta + t_a * (curve_t1 + one_minus_ta * (g + t_a * c))


class CurveTensor(object):
    '''A view into the simulation grid: a curve carries tenor points per timepoint per scenario, and
    this indexes that grid while keeping track of the indices and any non-linear interpolation.
    Used directly by TensorBlock.
    '''

    def __init__(self, interp_obj, index, alpha, np_index=None):
        self.interp_obj = interp_obj
        self.np_index = index if isinstance(index, np.ndarray) else np_index
        self.index = torch.tensor(
            index, dtype=torch.int64, device=interp_obj.tensor.device) if isinstance(index, np.ndarray) else index
        # SCENARIO ROWS, not a flattened (row, tenor) offset: the strategy owns that flattening,
        # a tenor SEGMENT having its own stride
        if alpha is not None:
            self.alpha = self.interp_obj.tensor.new(alpha) if isinstance(alpha, np.ndarray) else alpha
            self.index_next = (self.index + 1).clamp(0, self.interp_obj.shape[0] - 1)
        else:
            self.alpha = self.index_next = None
        # A curve every one of whose rows is row 0 (a static factor, or a stochastic one gathered
        # only at the base date) skips the flattening add. Decided off the NUMPY indices, so asking
        # costs no device sync.
        self.rows = None if not self.np_index.any() and self.alpha is None else self.index
        # Which of the source's row blocks owns each row this gather reads — also off the CPU-side
        # indices, once per CurveTensor rather than per gather. A leaf answers with its whole grid.
        self.route = self.interp_obj.route(self.np_index, self.alpha is not None)

    def interp_value(self):
        return self.interp_obj.gather_rows(self.index, self.index_next, self.alpha, self.route)

    def split(self, counts):
        sub_alpha = split_tensor(self.alpha, counts) if self.alpha is not None else [None] * counts.size
        sub_index = split_tensor(self.index, counts)
        return [CurveTensor(self.interp_obj, sub_index, sub_alpha, np_index=sub_np)
                for sub_index, sub_alpha, sub_np in
                zip(sub_index, sub_alpha, split_array(self.np_index, counts))]

    def interpolate_risk_neutral(self, curve_component, points, time_grid, time_multiplier):
        t = time_grid[:, 1].reshape(-1, 1)
        T = points + t
        return self.interpolate_curve(
            curve_component, T, time_multiplier) - self.interpolate_curve(
            curve_component, t, time_multiplier)

    def interpolate_curve(self, curve_component, points, time_factor):
        # our tensor object
        tensor = self.interp_obj.indexed_tensor
        # check the points being queried
        time_size, point_size = points.shape

        if point_size > 0:
            # get the points in years
            tenor_points_in_years = tensor.new(curve_component[FACTOR_INDEX_Daycount](points))
            curve_tenor = curve_component[FACTOR_INDEX_Tenor_Index]
            i1, i2, a = curve_tenor.get_index(tenor_points_in_years)

            if isinstance(curve_tenor.type, str):
                tenor_data = (curve_tenor.type, curve_tenor.min, curve_tenor.max)
            else:
                split_tenor = curve_tenor.tenor[curve_tenor.type[0][1]]
                tenor_data = (curve_tenor.type, (curve_tenor.min, split_tenor),
                              (split_tenor, curve_tenor.max))

            return self.interp_obj.eval(
                tenor_data, self.rows, self.index_next, self.alpha, i1, i2, a.unsqueeze(dim=-1),
                tenor_points_in_years, time_factor, route=self.route)
        else:
            # return a null tensor
            return tensor.new_zeros([time_size, 0, tensor.shape[-1]])


class TensorBlock(object):
    def __init__(self, code, tensors: List[CurveTensor], time_grid: np.ndarray):
        self.code = code
        self.time_grid = time_grid
        self.curve_tensors = tensors
        self.local_cache = {}

    def split_counts(self, counts, shared):

        key_code = ('tensorblock', tuple([x[:2] for x in self.code]),
                    tuple(self.time_grid[:, TIME_GRID_MTM]),
                    tuple(counts))

        if key_code not in shared.t_Buffer:
            rate_tensor = zip(*[sub_tensor.split(counts) for sub_tensor in self.curve_tensors])
            time_block = np.split(self.time_grid, counts.cumsum())
            shared.t_Buffer[key_code] = [TensorBlock(self.code, tensor, time_t)
                                         for tensor, time_t in zip(rate_tensor, time_block)]

        return shared.t_Buffer[key_code]

    def gather_weighted_curve(self, shared, end_points,
                              start_points=None, multiply_by_time=True):

        # @torch.jit.script
        def calc_curve(time_multiplier, points):
            temp_curve = None
            for curve_tensor, curve_component in zip(self.curve_tensors, self.code):
                # handle static curves
                if not curve_component[FACTOR_INDEX_Stoch] and shared.riskneutral:
                    scaled_val = curve_tensor.interpolate_risk_neutral(
                        curve_component, end_points, self.time_grid, time_multiplier)
                else:
                    scaled_val = curve_tensor.interpolate_curve(curve_component, points, time_multiplier)

                if temp_curve is None:
                    temp_curve = scaled_val
                else:
                    temp_curve += scaled_val

            return temp_curve
        
        local_cache_key = (end_points.shape, end_points.tobytes(),
                           (start_points.shape, start_points.tobytes()) if start_points is not None else None,
                           multiply_by_time)

        if local_cache_key not in self.local_cache:

            curve_points = calc_curve(1 if multiply_by_time else 0, end_points)

            if start_points is not None:
                curve_points -= calc_curve(1 if multiply_by_time else 0, start_points)
            self.local_cache[local_cache_key] = curve_points

        return self.local_cache[local_cache_key]

    def reduce_deflate(self, delta_scen_t, time_points, shared):
        DtT = torch.exp(-torch.squeeze(self.gather_weighted_curve(shared, delta_scen_t)).cumsum(dim=0))
        # we need the index just prior - note this needs to be checked in the calling code
        indices = self.time_grid[:, TIME_GRID_MTM].searchsorted(time_points) - 1
        return {t: DtT[index] for t, index in zip(time_points, indices)}


class DerivedForwardCurve(object):
    '''A forward curve rebuilt from simulated components, F(t,T) = S(t) exp(c(T)(T-t) + r(t,T)(T-t)):
    S a spot tensor (time, batch), c a carry TensorBlock at absolute (excel date) tenors, r a
    repo/funding TensorBlock at relative year tenors. `t_excel` maps each time row to its excel date
    offset, so gathers take the same absolute-date end_points a ForwardPrice factor does. Duck-types
    the TensorBlock surface curve pricing uses. F(t,t) = S(t) exactly.
    '''

    def __init__(self, spot, carry, repo, t_excel, time_grid):
        self.spot = spot
        self.carry = carry
        self.repo = repo
        self.t_excel = t_excel
        self.time_grid = time_grid

    def split_counts(self, counts, shared):
        cum_counts = counts.cumsum()
        return [DerivedForwardCurve(*sub_block) for sub_block in zip(
            torch.split(self.spot, tuple(counts)),
            self.carry.split_counts(counts, shared), self.repo.split_counts(counts, shared),
            np.split(self.t_excel, cum_counts), np.split(self.time_grid, cum_counts))]

    def gather_weighted_curve(self, shared, end_points, start_points=None, multiply_by_time=False):
        tenor_in_days = end_points - self.t_excel.reshape(-1, 1)
        cost_of_carry = self.carry.gather_weighted_curve(
            shared, end_points, multiply_by_time=False) * self.spot.new_tensor(
            tenor_in_days / DAYS_IN_YEAR).unsqueeze(-1) + self.repo.gather_weighted_curve(shared, tenor_in_days)
        return self.spot.unsqueeze(1) * torch.exp(cost_of_carry)


# date generation utils

def cds_dates(base, num_months):
    base_month = base.month
    initial = pd.DateOffset(months=(3 - base_month % 3) % 3, day=20)
    months = pd.DateOffset(months=3)
    last_date = (base + initial) if base.day < 20 else (base + initial + months)
    res = [last_date]

    while last_date < base + pd.DateOffset(months=num_months):
        last_date = last_date + months
        res.append(last_date)

    return res


def calc_cds_rates(R, survival, discount, base_date, CDS_tenors, all_factors, bump=0.01 * 0.01):
    def calc_par_cds(S_j, cds_tenor, delta=0.0, start_time=None, end_time=None):
        if delta:
            S_vals = S_j.copy()
            S_vals[start_time: end_time] += delta * (S_ti[start_time: end_time] - S_ti[start_time])
        else:
            S_vals = S_j

        h = (S_vals[1:] - S_vals[:-1]) / (S_ti[1:] - S_ti[:-1])
        S = np.exp(-S_vals)
        F = D * S
        V_prot = ((F[:-1] - F[1:]) * h) / (h + f)

        cds_pay_dates = cds_dates(base_date, int(cds_tenor * 12))
        # insert the previous standard date (3 months prior)
        cds_pay_dates.insert(0, cds_pay_dates[0] - pd.DateOffset(months=3))
        tau = np.array([survival[FACTOR_INDEX_Daycount]((x - base_date).days) for x in cds_pay_dates])
        alpha = tau[1:] - tau[:-1]
        n = S_ti.searchsorted(tau[1:])
        v_fee = -tau[0]
        prev_n = 0

        for alpha_j, prev_tau, n_j in zip(alpha, tau[:-1], n):
            sub_i = slice(prev_n, n_j)
            sub_i_p1 = slice(prev_n + 1, n_j + 1)
            h_plus_f = h[sub_i] + f[sub_i]
            A_j = ((1 + h_plus_f * (S_ti[sub_i] - prev_tau)) * F[sub_i] - (
                    1 + h_plus_f * (S_ti[sub_i_p1] - prev_tau)) * F[sub_i_p1]) * h[sub_i] / h_plus_f ** 2
            v_fee += alpha_j * D[n_j] * S[n_j] + A_j.sum()
            prev_n = n_j

        v_prot = (1.0 - R) * V_prot[:n_j].sum()
        return v_prot / v_fee, n[-1]

    max_cds_dates = cds_dates(base_date, int(max(CDS_tenors) * 12))
    time_to_add = [survival[FACTOR_INDEX_Daycount]((x - base_date).days) for x in max_cds_dates]

    S_proc = all_factors[survival[FACTOR_INDEX_Offset]]
    D_proc = all_factors[discount[FACTOR_INDEX_Offset]]
    S_factor = S_proc.factor if hasattr(S_proc, 'factor') else S_proc
    D_factor = D_proc.factor if hasattr(D_proc, 'factor') else D_proc

    # calculate the piecewise hazard rate, forward rate and survival and discount curves
    S_ti = np.union1d(S_factor.get_tenor(), time_to_add)
    D_vals = D_factor.current_value(S_ti) * S_ti
    f = (D_vals[1:] - D_vals[:-1]) / (S_ti[1:] - S_ti[:-1])
    D = np.exp(-D_vals)

    S_vals_0 = S_factor.current_value(S_ti)
    CDS_rates = {}
    for tenor in CDS_tenors:
        CDS_rates[tenor] = calc_par_cds(S_vals_0, tenor)

    if bump:
        S_j = [S_vals_0]
        start = 0

        for k, v in CDS_rates.items():
            end = v[1] + 1
            delta_j = scipy.optimize.brentq(
                lambda x: calc_par_cds(S_vals_0, k, delta=x, start_time=start, end_time=end)[0] - (v[0] + bump), -0.1,
                0.1)

            S_j.append(S_vals_0.copy())
            S_j[-1][start: end] += delta_j * (S_ti[start: end] - S_ti[start])
            start = v[1]

        return {k: v[0] for k, v in CDS_rates.items()}, S_ti, S_j
    else:
        return {k: v[0] for k, v in CDS_rates.items()}


def calc_par_cds(R, D, f, S_ti, S_j, tau, delta=0.0, start_time=None, end_time=None):
    if delta:
        S_vals = S_j.copy()
        S_vals[start_time: end_time] += delta * S_ti[start_time: end_time]
    else:
        S_vals = S_j

    h = (S_vals[1:] - S_vals[:-1]) / (S_ti[1:] - S_ti[:-1])
    S = np.exp(-S_vals)
    F = D * S
    V_prot = ((F[:-1] - F[1:]) * h) / (h + f)

    alpha = tau[1:] - tau[:-1]
    n = S_ti.searchsorted(tau[1:])
    v_fee = -tau[0]
    prev_n = 0

    for alpha_j, prev_tau, n_j in zip(alpha, tau[:-1], n):
        sub_i = slice(prev_n, n_j)
        sub_i_p1 = slice(prev_n + 1, n_j + 1)
        h_plus_f = h[sub_i] + f[sub_i]
        A_j = ((1 + h_plus_f * (S_ti[sub_i] - prev_tau)) * F[sub_i] - (
                1 + h_plus_f * (S_ti[sub_i_p1] - prev_tau)) * F[sub_i_p1]) * h[sub_i] / h_plus_f ** 2
        v_fee += alpha_j * D[n_j] * S[n_j] + A_j.sum()
        prev_n = n_j

    v_prot = (1.0 - R) * V_prot[:n_j].sum()
    return v_prot / v_fee


def index_cds_par_spread(
    H0_names, tau, D, R, f, S_ti, hazard_scale, eps=1e-14
):
    H = hazard_scale * H0_names                 # (N,M)
    N, M = H.shape

    dt = S_ti[1:] - S_ti[:-1]                   # (M-1,)
    h = (H[:, 1:] - H[:, :-1]) / dt             # (N,M-1)

    S = np.exp(-H)                               # (N,M)
    F = S * D[None, :]                           # (N,M)

    hp = h + f[None, :]
    hp = np.where(np.abs(hp) < eps, np.sign(hp) * eps + eps, hp)

    V_prot = ((F[:, :-1] - F[:, 1:]) * h) / hp   # (N,M-1)

    alpha = tau[1:] - tau[:-1]
    n = S_ti.searchsorted(tau[1:])               # match calc_par_cds
    # Optional: assert tau points are on-grid (safer)
    # (need to check indices bounds first)
    if np.any(n >= len(S_ti)):
        raise ValueError("tau contains points beyond S_ti range")
    if not np.all(S_ti[n] == tau[1:]):
        raise ValueError("tau[1:] must be exact grid points in S_ti")

    v_fee = -tau[0] * N
    prev_n = 0

    for alpha_j, prev_tau, n_j in zip(alpha, tau[:-1], n):
        sub_i = slice(prev_n, n_j)
        sub_i_p1 = slice(prev_n + 1, n_j + 1)

        hp_seg = h[:, sub_i] + f[sub_i][None, :]
        hp_seg = np.where(np.abs(hp_seg) < eps, np.sign(hp_seg) * eps + eps, hp_seg)

        term0 = (1.0 + hp_seg * (S_ti[sub_i][None, :] - prev_tau)) * F[:, sub_i]
        term1 = (1.0 + hp_seg * (S_ti[sub_i_p1][None, :] - prev_tau)) * F[:, sub_i_p1]
        A_j = (term0 - term1) * h[:, sub_i] / (hp_seg ** 2)

        v_fee += alpha_j * np.sum(D[n_j] * S[:, n_j]) + np.sum(A_j)
        prev_n = n_j

    n_last = n[-1]
    v_prot_total = (1.0 - R) * np.sum(V_prot[:, :n_last])

    return v_prot_total / v_fee


def filter_data_frame(df, from_date, to_date, rate=None):
    index1 = (pd.Timestamp(from_date) - excel_offset).days
    index2 = (pd.Timestamp(to_date) - excel_offset).days
    return df.loc[index1:index2] if rate is None else df.loc[index1:index2][
        [col for col in df.columns if col.startswith(rate)]]


def bars_touched(bars, level, barrier_up):
    """Whether a continuously-monitored level was touched over a daily `(low, high)` bar series.

    A bar BRACKETS every intraday print of its day, so the verdict is exact without the prints:
    `Up` touches iff any high reaches the level, `Down` iff any low reaches it. Touching IS
    crossing - the inequalities are weak, and an exact-touch day is a hit.

    Pure and source-free. WHERE THE BARS COME FROM - hydrating `(index, date, source)` facts from
    the log or the market data - is spine increment 4's, not this function's.
    """
    for low, high in bars:
        if low > high:
            raise UnpriceableSchedule(
                'a daily bar has low {:g} above high {:g}, which brackets nothing - a bar is the '
                'range every print of its day fell inside'.format(low, high))
        if (high >= level) if barrier_up else (low <= level):
            return True
    return False


# Math Type stuff

def hermite_interpolation(tenors, rates):
    def calc_ri(t, r):
        r_i = ((np.diff(r[:-1]) * np.diff(t[1:])) / np.diff(t[:-1]) +
               (np.diff(r[1:]) * np.diff(t[:-1])) / np.diff(t[1:])) / (t[2:] - t[:-2])
        r_1 = (((r[1] - r[0]) * (t[2] + t[1] - 2.0 * t[0])) / (t[1] - t[0]) -
               (r[2] - r[1]) * (t[1] - t[0]) / (t[2] - t[1])) / (t[2] - t[0])
        r_n = -1.0 / (t[-1] - t[-3]) * ((r[-2] - r[-3]) * (t[-1] - t[-2]) / (t[-2] - t[-3]) -
                                        (r[-1] - r[-2]) * (2.0 * t[-1] - t[-2] - t[-3]) / (t[-1] - t[-2]))
        return np.append(np.append(r_1, r_i), r_n)

    def calc_gi(t, r, ri):
        return np.append(np.diff(t), 0.0) * ri - np.append(np.diff(r), 0.0)

    def calc_ci(t, r, ri):
        return np.append(2.0 * np.diff(r) - np.diff(t) * (ri[:-1] + ri[1:]), 0.0)

    ri = calc_ri(tenors, rates)
    gi = calc_gi(tenors, rates, ri)
    ci = calc_ci(tenors, rates, ri)
    return gi, ci


# @torch.jit.script
def norm_cdf(x):
    return 0.5 * (torch.erfc(x * -0.7071067811865475))


def norm_pdf(x):
    return 0.3989422804014327 * torch.exp(-0.5 * x * x)


def norm_icdf(x):
    return 1.4142135623730951 * torch.erfinv(2.0 * x - 1.0)


def BivN(P, Q, rho):
    from scipy.stats import multivariate_normal
    mvn = np.vectorize(lambda x: multivariate_normal(cov=[[1.0, x], [x, 1.0]]))
    z2 = mvn(rho)
    cdf = np.vectorize(lambda z, x, y: z.cdf([x, y]))
    return cdf(z2, P, Q)


def ApproxBivN(P, Q, rho):
    """Bivariate normal integral, accurate to about 4 decimal places - Tsay and Ke, "A Simple
    Approximation for Bivariate Normal Integral Based on Error Function". Chosen for being fully
    vectorized rather than for accuracy.
    """
    # work out the cases
    denom = torch.sqrt(1.0 - rho * rho)
    a = -rho / denom
    b = P / denom
    numer = a * Q + b

    case1 = (a > 0.0) & (numer >= 0.0)
    case2 = (a > 0.0) & (numer < 0.0)
    case3 = (a < 0.0) & (numer >= 0.0)
    case4 = (a < 0.0) & (numer < 0.0)

    c1 = -1.0950081470333
    c2 = -0.75651138383854
    r2 = 1.4142135623730951
    ma2c2 = 1.0 - a * a * c2
    two_sq_ma2c2 = 2.0 * torch.sqrt(ma2c2)
    a2c1_2 = a * a * c1 * c1
    q_part = r2 * (Q - a * c2 * (a * Q + b))
    root4_p = torch.exp((a2c1_2 + 2 * b * (r2 * c1 + b * c2)) / (4.0 * ma2c2)) / (2.0 * two_sq_ma2c2)
    root4_m = torch.exp((a2c1_2 - 2 * b * (r2 * c1 - b * c2)) / (4.0 * ma2c2)) / (2.0 * two_sq_ma2c2)
    erf2_p = torch.erf((q_part + a * c1) / two_sq_ma2c2)
    erf2_m = torch.erf((q_part - a * c1) / two_sq_ma2c2)
    erf_p1 = (r2 * b) / (a * two_sq_ma2c2)
    erf_p2 = (a * a * c1) / (a * two_sq_ma2c2)
    erf1 = torch.erf(erf_p1 + erf_p2)
    erf3 = torch.erf(erf_p1 - erf_p2)
    final = norm_cdf(P) * norm_cdf(Q)

    for c, f in enumerate([case1, case2, case3, case4]):
        if f.any():
            if c == 0:
                case = .5 * (
                        torch.erf(Q / r2) + torch.erf(b / (r2 * a))) + root4_m * (
                               1.0 - erf3) - root4_p * (erf2_m + erf1)
            elif c == 1:
                case = root4_m * (1 + erf2_p)
            elif c == 2:
                case = .5 * (1 + torch.erf(Q / r2)) - root4_p * (1.0 + erf2_m)
            else:
                case = .5 * (1 - torch.erf(b / (r2 * a))) - root4_p * (1.0 - erf1) + root4_m * (erf2_p + erf3)

            final[f] = case[f]

    return final


def black_european_option_price(F, X, r, vol, tenor, buyOrSell, callOrPut):
    stddev = vol * np.sqrt(tenor)
    sign = 1.0 if (F > 0.0 and X > 0.0) else -1.0
    d1 = (np.log(F / X) + 0.5 * stddev * stddev) / stddev
    d2 = d1 - stddev
    return buyOrSell * callOrPut * (F * scipy.stats.norm.cdf(callOrPut * sign * d1) -
                                    X * scipy.stats.norm.cdf(callOrPut * sign * d2)) * np.exp(-r * tenor)


def bachelier_european_option_price(F, X, r, vol, tenor, buyOrSell, callOrPut):
    """The numpy twin of `bachelier_european_option`, and the same signature as the numpy Black.

    A vol quoted NORMAL is an absolute rate move, so the premium is
    ``P = e^{-rT}[mu*Phi(mu/s) + s*phi(mu/s)]`` with ``mu = omega(F-X)`` and ``s = sigma_N sqrt(T)``.

    THE GENERAL FORM, not the at-the-money one: `create_market_swaps` strikes its benchmarks at par
    and so always calls this at ``F = X``, where it collapses to ``A sigma_N sqrt(T/2 pi)`` - but
    baking that collapse in would price an off-market strike as an ATM one in silence.

    ``mu = omega(F-X)`` carries both directions in one expression because ``phi`` is even, and it is
    the SAME expression `bachelier_european_option` evaluates in tensors - one formula in two
    precisions, as the Black pair is. Measured at the money, the two are BIT-IDENTICAL and both sit
    2.2e-16 relative from the closed form. Gate: `tests/test_hw2f_analytic.py`'s
    `test_the_two_conventions_are_two_prices_and_the_normal_one_is_the_bachelier_premium`.
    """
    stddev = vol * np.sqrt(tenor)
    mu = callOrPut * (F - X)
    return buyOrSell * (mu * scipy.stats.norm.cdf(mu / stddev) +
                        stddev * scipy.stats.norm.pdf(mu / stddev)) * np.exp(-r * tenor)


# ======================================================================================
# Characteristic-function (Fourier) inversion primitive for affine option pricers: a MODEL-AGNOSTIC
# European vanilla / digital pricer for any model whose aggregate log-return R = log(S_T/S_t) has a
# known characteristic function. A model supplies ONLY its log-CF (see `cf_european_probabilities`
# for the plug-in contract). Float64 is mandatory: the S*P1 - K*e^{-rn}*P2 assembly is a
# cancellation of two O(1) probabilities and float32 destroys it.
# ======================================================================================

def gauss_legendre(a, b, panels, order=8, dtype=torch.float64, device='cpu'):
    """Composite Gauss-Legendre nodes/weights on [a, b], ASCENDING, endpoints excluded.

    ``panels`` sub-intervals each carry an ``order``-point rule. The panel edges - hence ``a`` and
    ``b`` - are never sampled, so an integrand with a removable singularity at an endpoint (the
    ``1/(i*phi)`` of a Fourier inversion at ``phi = 0``) integrates on ``[0, phi_max]`` with no hole.
    """
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    nodes = (mid + half * x[None, :]).ravel()
    wts = (half * w[None, :]).ravel()
    o = np.argsort(nodes)
    return (torch.tensor(nodes[o], dtype=dtype, device=device),
            torch.tensor(wts[o], dtype=dtype, device=device))


def gauss_legendre_dyadic(phi_max, panels, order=8, dtype=torch.float64, device='cpu', start=8.0):
    """Gauss-Legendre nodes on ``[0, phi_max]`` NESTED across the doubling ladder of
    :func:`cf_adaptive_phi_max`.  Returns ``(nodes, weights, cuts)``.

    Fixed blocks ``[0, start]``, ``[start, 2*start]``, ``[2*start, 4*start]``, ..., the first at
    ``panels`` panels and each doubling block at half that, so EVERY block carries at least the
    panel width ``panels`` uniform panels buy over that block's own upper bound - accuracy at or
    above :func:`gauss_legendre` on every rung, by construction. The grid for a smaller bound is
    then a PREFIX of the grid for a larger one: ``cuts[rung]`` is how many leading nodes integrate
    to ``rung``, so one backward recursion over the widest bound serves every contract under it.
    """
    nodes, wts, cuts, lo, hi, blocks = [], [], {}, 0.0, float(start), int(panels)
    while True:
        n, w = gauss_legendre(lo, hi, blocks, order, dtype, device)
        nodes.append(n)
        wts.append(w)
        cuts[hi] = cuts.get(lo, 0) + len(n)
        if hi >= phi_max:
            return torch.cat(nodes), torch.cat(wts), cuts
        lo, hi, blocks = hi, hi * 2.0, max(int(panels) // 2, 1)


def complex_log_unwrap(w, dim=-1):
    """Complex log with the branch fixed by continuity ALONG ``dim`` (the phi grid).

    ``dim`` must be an axis along which phi varies smoothly and monotonically, anchored at its first
    entry (smallest phi, where ``w`` is near ``1+0j``). The general guard against the discrete
    "Heston trap": the principal branch of ``log(1 - 2*alpha*B)`` taken independently at each
    backward step is wrong whenever that argument winds around the origin. ``torch.round`` carries
    zero gradient, correctly - the winding correction is a locally-constant integer. Reduces to the
    principal branch at size 1 along ``dim``.
    """
    two_pi = 2.0 * np.pi
    ang = torch.angle(w)
    if w.shape[dim] > 1:
        d = torch.diff(ang, dim=dim)
        d = d - two_pi * torch.round(d / two_pi)
        first = ang.narrow(dim, 0, 1)
        ang = torch.cat([first, first + torch.cumsum(d, dim=dim)], dim=dim)
    return torch.complex(torch.log(torch.abs(w)), ang)


def cf_phi_max_ladder(start=8.0, cap=2.0 ** 24):
    """The doubling rungs :func:`cf_adaptive_phi_max` scans - the ONE spelling of them, so a caller
    that pre-computes its log-CF ON the ladder scans the same rungs it did."""
    rungs, phi = [], float(start)
    while phi < cap:
        rungs.append(phi)
        phi *= 2.0
    return rungs


def cf_adaptive_phi_max(logcf, carry, dtype=torch.float64, device='cpu',
                        log_tol=-40.0, start=8.0, cap=2.0 ** 24):
    """Smallest power-of-two ``phi_max`` at which the inversion integrand has decayed.

    The criterion is ``Re(logcf) - ln(phi) < log_tol`` on BOTH inversion contours (``i*phi`` and
    ``i*phi + 1``), the +1 share-measure contour normalised by the log forward-growth ``carry``.
    ``logcf`` must already be reduced to the SLOWEST-DECAYING states in the batch. A closed-form
    cutoff is wrong here: the envelope decays slower than the pure-Gaussian ``exp(-phi^2 V/2)``.
    Runs under ``no_grad``.

    THE WHOLE LADDER IS ONE EVALUATION. The doubling rungs are independent questions and the
    backward recursion behind ``logcf`` is ELEMENTWISE in phi, so every candidate and both contours
    ride one pass as a single tensor and the sequential test then reads the answers off. The branch
    unwrap is anchored on the trailing axis, which stays length one, so each element is asked the
    same question one at a time or batched. THE ANSWER IS BIT-IDENTICAL AS MEASURED, NOT BY
    CONSTRUCTION: elementwise on CUDA, while on CPU torch dispatches a different complex kernel and
    three or four rungs in twenty land 1-2 ulp apart. What survives that is the BOUND, because the
    criterion is a threshold on a power-of-two ladder: consecutive rungs are whole units of the
    metric apart and the perturbation is 1e-15 of one.

    THE PLUG-IN CONTRACT: ``logcf`` receives a complex phi of shape ``(rung, 1)`` and must
    broadcast its state axes in FRONT of it, returning ``(*state, rung, 1)``.
    """
    with torch.no_grad():
        rungs = cf_phi_max_ladder(start, cap)
        if not rungs:
            return float(start)            # ``start`` is already at the cap: nothing to scan
        phi = rungs[-1] * 2.0
        ladder = torch.tensor(rungs, dtype=dtype, device=device).reshape(-1, 1)
        m0 = logcf(ladder * 1j).real
        m1 = logcf(ladder * 1j + 1.0).real - carry
        top = torch.maximum(m0, m1).movedim(-2, 0).reshape(len(rungs), -1).amax(-1).tolist()
        for rung, m in zip(rungs, top):
            if m - np.log(rung) < log_tol:
                return rung
        return phi                         # nothing decayed: the first rung past the cap


def cf_european_probabilities(logcf, log_moneyness, carry, phi_max, panels=256, order=8,
                              dtype=torch.float64, device='cpu', want=3, grid=None):
    """The two exercise probabilities P1, P2 of a European claim, by Fourier inversion.

    MODEL-AGNOSTIC.  Given a model whose aggregate log-return ``R = log(S_T/S_t)`` has the
    generalised (complex-phi) characteristic function ``E_t[(S_T/S_t)^phi] = exp(logcf(phi))``,

        P2 = 1/2 + (1/pi) Int_0^inf Re[ e^{-i phi m} exp(logcf(i phi))            / (i phi) ] d phi
        P1 = 1/2 + (1/pi) Int_0^inf Re[ e^{-i phi m} exp(logcf(i phi + 1) - carry)/ (i phi) ] d phi

    with ``m = ln(K/S)`` and ``carry = ln E_t[S_T/S_t]``.  A vanilla is then priced by the caller as
    ``S*P1 - K*e^{-carry}*P2`` and a digital/CDF by ``Q(R <= b) = 1 - P2`` at ``m = b``, spot-free by
    construction.  ``want`` is a bit mask: 1 = P1, 2 = P2, 3 = both, so a CDF is half the cost.

    THE PLUG-IN CONTRACT.  ``logcf(phi)`` receives a complex tensor whose trailing axis is the
    quadrature grid and must return ``log E_t[(S_T/S_t)^phi]`` broadcasting to ``(batch, node)`` -
    the state is captured by the closure.  For an AFFINE model that log-CF is ``A(phi) + B(phi)*V_t``
    from the model's own backward recursion (use :func:`complex_log_unwrap` there for the branch of
    any ``log(1 - ...)`` term); a Levy model returns ``A(phi)`` alone.  The caller also resolves
    ``phi_max`` via :func:`cf_adaptive_phi_max` on the same closure at the worst-case state.

    ``grid`` : an optional precomputed ``(nodes, weights)`` in place of the internal build, for a
    caller integrating many contracts on ONE shared grid (``phi_max``/``panels``/``order`` are then
    unread).

    Differentiable w.r.t. every leaf reachable through ``logcf`` and ``carry``; float64 is required
    for the P1-P2 cancellation.
    """
    lm = log_moneyness.unsqueeze(-1)
    nodes, wts = grid if grid is not None else gauss_legendre(
        0.0, phi_max, panels, order, dtype, device)
    iphi = nodes * 1j
    shift = torch.exp(-1j * nodes * lm) / iphi                # K^{-i phi} S^{i phi} / (i phi)
    out = []
    for bit, off, disc in ((1, 1.0, carry), (2, 0.0, 0.0)):
        if not (want & bit):
            out.append(None)
            continue
        d = (shift * torch.exp(logcf(iphi + off) - disc)).real
        out.append(0.5 + (d * wts).sum(-1) / np.pi)
    return out[0], out[1]


# ======================================================================================
# Heston-Nandi GARCH(1,1): params + A/B recursion + daily-step recursion + semi-analytic pricing.
# The math is FREE FUNCTIONS taking the GARCH params as explicit trailing args (omega, alpha, beta,
# gamma_star, r); each consumer unpacks its own name->tensor mapping into those args by the
# canonical names below. Theory: HestonNandiImpliedSpotModel.documentation (stochasticprocess).
# ======================================================================================

# The HestonNandiModelParameters price factor's parameters, in canonical order - the SINGLE source
# of that name set, shared with riskfactors (the dependency edge only goes DOWN: utils never imports
# riskfactors). ``r``, the per-step cost of carry, is NOT a factor parameter - only these five are.
HN_PARAM_NAMES = ('Omega', 'Alpha', 'Beta', 'Gamma_Star', 'H0')


def hn_persistence(alpha, beta, gamma_star):
    """psi = beta + alpha * gamma*^2 (the GARCH persistence; must be < 1 for stationarity)."""
    return beta + alpha * gamma_star ** 2


def hn_stationary_var(omega, alpha, beta, gamma_star):
    """E[h] = (omega + alpha) / (1 - psi), the per-step stationary variance."""
    return (omega + alpha) / (1.0 - hn_persistence(alpha, beta, gamma_star))


def hn_ann_vol(omega, alpha, beta, gamma_star, steps_per_year=252.0):
    """Long-run annualised vol sqrt(E[h] * steps_per_year); float or tensor per the inputs."""
    v = hn_stationary_var(omega, alpha, beta, gamma_star) * steps_per_year
    return float(v) ** 0.5 if not torch.is_tensor(v) else v.sqrt()


def hn_ab(phi, n_steps, omega, alpha, beta, gamma_star, r, unwrap=True, phi_dim=-1):
    """Backward A/B recursion for ``n_steps`` steps.  Returns ``(A, B)``.

    ``phi`` : real OR complex tensor.  If complex it is assumed to vary smoothly and ascending
              along ``phi_dim`` (needed for the branch unwrap).
    Result satisfies E_t[S_{t+n}^phi] = S_t^phi * exp(A + B * h_{t+1}); i.e. the HN affine log-CF
    of the aggregate log-return is ``A + B * h1`` (the closure handed to the model-agnostic
    inversion primitive :func:`cf_european_probabilities`).
    """
    A = torch.zeros_like(phi)
    B = torch.zeros_like(phi)
    lin = phi * (gamma_star - 0.5) - 0.5 * gamma_star ** 2   # <-- the -phi/2 is the LRNVR drift
    half_sq = 0.5 * (phi - gamma_star) ** 2
    phir = phi * r
    for _ in range(int(n_steps)):
        w = 1.0 - 2.0 * alpha * B
        logw = complex_log_unwrap(w, dim=phi_dim) if (unwrap and w.is_complex()) else torch.log(w)
        A = A + phir + B * omega - 0.5 * logw
        B = lin + beta * B + half_sq / w
    return A, B


def hn_logmgf(phi, n_steps, h1, omega, alpha, beta, gamma_star, r, **kw):
    """log E_t[exp(phi * R_n)] where R_n = log(S_{t+n}/S_t).  = A + B*h1."""
    A, B = hn_ab(phi, n_steps, omega, alpha, beta, gamma_star, r, **kw)
    return A + B * h1


def auto_phi_max(n_steps, h1, omega, alpha, beta, gamma_star, r,
                 log_tol=-40.0, start=8.0, cap=2.0 ** 24):
    """Smallest power-of-two phi_max with Re(A + B*h1) - ln(phi) < log_tol.

    The HN glue for :func:`cf_adaptive_phi_max`: it reduces the batch to the extreme h1 (the
    smallest, whose integrand decays slowest) so the scan runs on a 2-element state.
    """
    h1t = torch.as_tensor(h1).detach()
    hs = torch.stack([h1t.min(), h1t.max()]).to(omega.dtype).reshape(-1, 1, 1)
    carry = torch.as_tensor(r).detach() * int(n_steps)
    return cf_adaptive_phi_max(
        lambda z: hn_logmgf(z, n_steps, hs, omega, alpha, beta, gamma_star, r), carry,
        omega.dtype, omega.device, log_tol, start, cap)


def _p1_p2(logm, n_steps, h1, omega, alpha, beta, gamma_star, r,
           phi_max, panels, order, unwrap, want=3):
    """P1, P2 for log-moneyness ``logm`` = ln(K/S).  ``logm``/``h1`` broadcast together.

    Thin HN glue over :func:`cf_european_probabilities`: the HN affine log-CF ``A + B*h1`` as the
    ``logcf`` closure and ``r*n`` as the P1-contour normalisation.  ``want``: 1 = P1, 2 = P2.
    """
    logm = torch.as_tensor(logm, dtype=omega.dtype, device=omega.device)
    h1 = torch.as_tensor(h1, dtype=omega.dtype, device=omega.device)
    logm, h1 = torch.broadcast_tensors(logm, h1)
    if phi_max is None:
        phi_max = auto_phi_max(n_steps, h1, omega, alpha, beta, gamma_star, r)
    if panels is None:
        panels = 256
    hh = h1.unsqueeze(-1)

    def logcf(phi):
        A, B = hn_ab(phi, n_steps, omega, alpha, beta, gamma_star, r, unwrap=unwrap)
        return A + B * hh

    return cf_european_probabilities(
        logcf, logm, r * n_steps, phi_max, panels, order, omega.dtype, omega.device, want)


def hn_call(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r,
            phi_max=None, panels=None, order=8, unwrap=True):
    """European CALL, ``n_steps`` steps to expiry, spot ``S``, strike ``K``.

    ``h1`` is the (predictable) variance of the FIRST step; ``r`` the PER-STEP cost of carry.
    Differentiable w.r.t. (omega, alpha, beta, gamma_star, r, h1, S, K).
    """
    S = torch.as_tensor(S, dtype=omega.dtype, device=omega.device)
    K = torch.as_tensor(K, dtype=omega.dtype, device=omega.device)
    P1, P2 = _p1_p2(torch.log(K / S), n_steps, h1, omega, alpha, beta, gamma_star, r,
                    phi_max, panels, order, unwrap)
    return S * P1 - K * torch.exp(-r * n_steps) * P2


def hn_put(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r, **kw):
    """European PUT.  By put-call parity off :func:`hn_call` (the parity residual of the inversion
    itself is tested separately via the phi=1 martingale identity)."""
    S = torch.as_tensor(S, dtype=omega.dtype, device=omega.device)
    K = torch.as_tensor(K, dtype=omega.dtype, device=omega.device)
    return (hn_call(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r, **kw)
            - S + K * torch.exp(-r * n_steps))


def hn_cdf_logret(x, n_steps, h1, omega, alpha, beta, gamma_star, r,
                  phi_max=None, panels=None, order=8, unwrap=True):
    """EXACT  Q( R_n <= x )  where R_n = log(S_{t+n}/S_t), by Fourier inversion - what the
    one-step-survival loop needs for an UP barrier at S*exp(x). Spot-free by construction; ``x``
    and ``h1`` broadcast together.
    """
    _, P2 = _p1_p2(x, n_steps, h1, omega, alpha, beta, gamma_star, r,
                   phi_max, panels, order, unwrap, want=2)
    return 1.0 - P2


# The predictable-variance recursion h_{t+1} = omega + beta*h_t + alpha*(z_t - gamma*sqrt(h_t))^2
# lives ONLY in ``hn_variance_step``, which every consumer routes through - the OSS pricers in
# ``pricing.py``, the ``HestonNandiImpliedSpotModel`` diffusion, and ``tests/hn_reference.py``.

def hn_variance_step(h, sh, z, omega, alpha, beta, gamma_star):
    """The HN predictable-variance recursion h_{t+1} = omega + beta*h + alpha*(z - gamma*sqrt(h))^2.

    ``sh`` = sqrt(h) is passed in (the caller already needs it for the log-spot step), so the
    square root is computed exactly once.  All args broadcast on the simulation axis.
    """
    return omega + beta * h + alpha * (z - gamma_star * sh) ** 2


def hn_daily_advance(Sj, h, b_step, z, omega, alpha, beta, gamma_star):
    """One daily Heston-Nandi step under the risk-neutral (LRNVR) measure. Returns (Sj, h).

    Advances the log-spot by ``(b_step - 0.5*h) + sqrt(h)*z`` and recurses the predictable variance.
    ``z`` is either a fresh unconditional normal or the survival-truncated final draw of a monitored
    interval; in BOTH cases the recursion is fed the REALISED z - the survival-conditioned law is
    leverage-asymmetric under truncation, so DO NOT 'fix' it back to an unconditional draw.
    ``b_step`` is the per-step cost-of-carry (r-q); all args broadcast on the trailing sim axis.
    """
    sh = torch.sqrt(h)
    Sj = Sj * torch.exp((b_step - 0.5 * h) + sh * z)
    h = hn_variance_step(h, sh, z, omega, alpha, beta, gamma_star)
    return Sj, h


def hn_log_substep(log_S, h, z, b_step, omega, alpha, beta, gamma_star):
    """One unmonitored HN day, accumulating the LOG increment: the same step as
    :func:`hn_daily_advance` with the exponential left to the caller.

    Kept separate because it is the chain the OSS pricers repeat n_sub times per fixing, and at
    their batch shapes it is bandwidth-bound - ~13 elementwise kernels over a multi-million element
    tensor. Fused it is one kernel and 5.9x faster, bit-identical forward and gradient.
    """
    sh = torch.sqrt(h)
    return (log_S + (b_step - 0.5 * h) + sh * z,
            hn_variance_step(h, sh, z, omega, alpha, beta, gamma_star))


#: The fused build, and the one every ordinary run takes. It is NOT twice differentiable
#: (AOTAutograd's compiled backward raises `does not currently support double backward`), so a
#: `Greeks: 'All'` valuation walks the eager function above - same numbers, ~5.9x slower.
hn_log_substep_fused = torch.compile(hn_log_substep, dynamic=True)


def declared_spot(code, name):
    """Pass a resolved spot code through, saying ONCE whether it is simulated.

    A static spot is held flat across the time grid at pricing - legitimate, but it makes the
    exposure profile a deterministic forward. Said in calc_dependencies because it is a compile-time
    fact; the alternative is a warning that repeats every batch.
    """
    if not code[0][FACTOR_INDEX_Stoch]:
        logging.warning('%s is not simulated - spot is held flat across the time grid',
                        check_tuple_name(code[0][FACTOR_INDEX_Offset]))
    return code


def spot_on_deal_grid(spot, deal_time, shared):
    """Give ``spot`` the shape every pricer assumes: (len(deal_time), n_scenarios).

    A SIMULATED spot already has it; a static one arrives as a single row and is tiled up. The test
    is on ROWS - the axis being corrected. Testing columns instead reads a legitimate broadcast pair
    as a defect and tiles the ROWS by len(deal_time), squaring the grid.
    """
    return spot if spot.shape[0] == len(deal_time) else spot.tile(
        len(deal_time), shared.simulation_batch)


def bridge_interval_variance(shared, factor_dep, deal_time):
    """Per-row SIMULATION log-variance spanning each step of a deal's own time axis, for the bridge.

    Elapsed time comes off the DEAL's axis: its dates need not be adjacent, or even start, on the
    scenario grid the rate was published against. The leading zero leaves the first date observing
    endpoints, and a factor with no published rate leaves every date so.
    """
    rate = getattr(shared, 't_Bridge_Variance_Rate', {}).get(factor_dep.get('Barrier_Underlying'))
    days = deal_time[:, TIME_GRID_MTM]
    return (rate or 0.0) / DAYS_IN_YEAR * np.diff(days, prepend=days[0])


def barrier_touched(prev_touched, prev_spot, s_t, barrier, variance, up):
    """Running PROBABILITY that the path has touched ``barrier`` at or before now.

    An endpoint test only asks whether the spot sits beyond the barrier ON a grid date, so a path
    that crossed and came back is recorded as never having touched - while the closed forms applied
    to that state assume CONTINUOUS monitoring. An endpoint already beyond the barrier is a certain
    touch, and with both endpoints inside, the Brownian-bridge crossing probability is exact for a
    lognormal step.

    ``variance`` is the SIMULATION log-variance spanning the interval. Falsy observes endpoints
    only, covering the three cases that must: the first date on any grid, a process with no
    lognormal interval law, and two coincident dates - whose zero would otherwise put a 0/0 in the
    discarded branch of the `where`, whose gradient is nan even when its value is thrown away.
    """
    beyond = ((s_t > barrier) if up else (s_t < barrier)).to(s_t.dtype)
    if not variance:
        return (prev_touched + beyond).clip(max=1.0)

    if up:
        d0, d1 = torch.log(barrier / prev_spot), torch.log(barrier / s_t)
    else:
        d0, d1 = torch.log(prev_spot / barrier), torch.log(s_t / barrier)
    crossed = torch.where((d0 > 0) & (d1 > 0),
                          torch.exp((-2.0 * d0 * d1 / variance).clamp(max=0.0)),
                          torch.ones_like(s_t))
    return prev_touched + (1.0 - prev_touched) * torch.maximum(beyond, crossed)


def hn_unmonitored_substeps(Sj, h, b_step, n_steps, hn_params, shared, num_sims, antithetic):
    """Advance (Sj, h) through ``n_steps`` UNCONDITIONAL (unmonitored) daily HN steps.

    These carry no barrier - the OSS truncation applies only on the monitored final step, done by
    the caller - so a monitored interval of n_sub days passes ``n_steps = n_sub - 1`` and a
    non-monitored one the full ``n_sub``. Fresh regular-stream normals per step; with ``antithetic``
    the normal is negated on the paired half to align with the u<->1-u halves of the truncated final
    draw. ``hn_params`` = (omega, alpha, beta, gamma_star).

    Nothing observes the spot between these steps, so the walk runs in log space and exponentiates
    ONCE.
    """
    if not n_steps:                                              # a daily fixing walks nothing
        return Sj, h
    # picked once per interval: the fused kernel has no double backward, so a second-derivative
    # run walks the eager spelling of it
    substep = hn_log_substep if shared.gamma else hn_log_substep_fused
    log_S = torch.zeros_like(b_step)
    for _ in range(n_steps):
        zc = torch.randn([shared.simulation_batch, num_sims],
                         dtype=shared.one.dtype, device=shared.one.device)
        z = torch.cat([zc, -zc], dim=-1) if antithetic else zc
        log_S, h = substep(log_S, h, z, b_step, *hn_params)
    return Sj * log_S.exp(), h


# ======================================================================================
# COMPONENT Heston-Nandi (Christoffersen-Jacobs-Ornthanalai-Wang 2008): the variance splits into a
# long-run component q_t and a short-run deviation that is a pure AR(1) in beta. The centered
# Q-measure recursions, the L-curve identity and the plain-family nesting map are in
# docs_src/developer/market_prices.md#hestonnandi-component; `hn_component_to_plain` is the map.
# On this framework's step convention h_t is the PREDICTABLE variance of the step from t to t+1 and
# z_t the innovation driving both that return and the update.
# ======================================================================================

#: The `HestonNandiComponentModelParameters` price factor's SCALAR parameters, in canonical order -
#: the single source of that name set, shared with the riskfactors class and every consumption site.
#: There is no Omega (a function of the L curve) and no Q0 (q_0 is L(0) by the anchoring).
HN_COMPONENT_PARAM_NAMES = ('Alpha', 'Beta', 'Gamma_1', 'Rho', 'Phi', 'Gamma_2', 'H0')

#: The CURVE parameter's name. Its VALUES are fitted leaves; its knots are structural.
HN_COMPONENT_CURVE_NAME = 'L_Curve'


def hn_component_l_path(knots, values, n_steps, steps_per_year=252.0):
    """The long-run variance path ``L_t`` for t = 0..n_steps, on the trading-day clock.

    PIECEWISE-LINEAR IN t BETWEEN PILLAR KNOTS, flat outside them, and the choice is the model:
    omega_t = L_{t+1} - rho*L_t differences this curve, so a piecewise CONSTANT L would spike
    omega_t at each pillar and a spline would oscillate between pillars nobody quoted. Linear in t
    makes omega_t affine within a pillar and kinked only AT one.

    ``knots`` are tenors in YEARS (structural, numpy), ``values`` the per-step variances at them (a
    differentiable tensor - the fitted leaf). Returns an (n_steps+1,) tensor including L_0, which
    IS q_0 by the anchoring.
    """
    return curve_at(knots, values, torch.arange(
        int(n_steps) + 1, dtype=values.dtype, device=values.device) / steps_per_year)


def curve_at(knots, values, t):
    """A fitted curve's values at arbitrary times ``t`` in years: PIECEWISE-LINEAR IN t between the
    knots and flat outside them. ``knots`` is structural (numpy), ``values`` the differentiable
    leaf, and the answer carries ``t``'s shape."""
    k = torch.as_tensor(np.ascontiguousarray(knots, dtype=float),
                        dtype=values.dtype, device=values.device)
    if k.numel() == 1:
        return values[0].expand(t.shape).clone()
    # right-hand knot of the bracketing segment, clamped so both ends flat-extrapolate
    j = torch.clamp(torch.searchsorted(k, t), 1, k.numel() - 1)
    lo, hi = k[j - 1], k[j]
    frac = ((t - lo) / (hi - lo)).clamp(0.0, 1.0)
    return values[j - 1] + frac * (values[j] - values[j - 1])


#: Slack in years matching a walk time to a bucket knot. A grid's ACCUMULATED cumsum lands a
#: boundary a few ulps low - 252 daily steps reach 1 - 3.1e-15 - and would start its bucket a step
#: late; buckets are calendar dates and never sit within the 30 ms this allows.
BUCKET_TOL = 1.0e-9


def bucket_at(knots, values, t):
    """A bucketed parameter's value in force at times ``t`` in years: PIECEWISE CONSTANT, the last
    knot at or before ``t`` (within ``BUCKET_TOL``) and the first value before the first knot.
    ``knots`` is structural (numpy), ``values`` the differentiable leaf, and the answer carries
    ``t``'s shape."""
    k = torch.as_tensor(np.ascontiguousarray(knots, dtype=float),
                        dtype=values.dtype, device=values.device) - BUCKET_TOL
    return values[torch.clamp(torch.searchsorted(k, t, right=True) - 1, min=0)]


def hn_component_omega_path(l_path, rho):
    """``omega_t = L_{t+1} - rho*L_t`` for t = 0..n-1, from an (n+1,) L path - the whole content of
    the L parametrisation.

    A NEGATIVE entry is a long-run variance demanded to fall FASTER than rho decays it, which drives
    q (and hence h) negative; the calibration refuses or floors it by name
    (``HestonNandiComponentModelParameters.negative_omega``).
    """
    return l_path[1:] - rho * l_path[:-1]


# The component pair recursion lives ONLY in ``hn_component_variance_step``, which every consumer
# routes through - the OSS pricers, the ``HestonNandiComponentImpliedSpotModel`` diffusion, the log
# sub-step below, and ``tests/hn_reference.py``'s ``hnc_simulate``.

#: THE VARIANCE FLOOR, a DECLARED PROPERTY OF THIS MODEL rather than a quiet repair: unlike plain
#: Heston-Nandi the CJOW pair has NO positivity guarantee for phi > 0, and no parameter box closes
#: it - the calibration REPORTS `worst_case_variance_drift` instead. 1e-12 per step is 0.16 basis
#: points of annualised vol, numerically zero but a number `sqrt` can take; without it a tail path
#: returns NaN (measured 2 of 8192 inner paths over 248 daily steps). The closed form does NOT
#: floor, so the two agree only where it does not bind - gated by
#: `test_the_closed_form_matches_day_stepped_monte_carlo`.
HN_COMPONENT_VARIANCE_FLOOR = 1.0e-12


def hn_component_variance_step(h, q, sh, z, omega_t, alpha, beta, gamma1, rho, phi, gamma2):
    """The component recursion, returning ``(h_{t+1}, q_{t+1})``, both floored at
    :data:`HN_COMPONENT_VARIANCE_FLOOR` (the floor is the model's, not a patch).

    ``sh`` = sqrt(h) is passed in, so the square root is computed once. ``omega_t`` is THIS step's
    long-run intercept, ``L_{t+1} - rho*L_t``; every other argument is a global.

    q IS COMPUTED FIRST and h reads it, because the CJOW form defines h_{t+1} off q_{t+1}: the
    long-run level moves and the short-run deviation is measured against the level it moved TO.
    Reading q_t there instead is a different model, with deviation persistence beta - rho.

    The clamp returns x ITSELF above the floor, so on the nested face this is bit-identical to the
    unfloored recursion.
    """
    e1 = z - gamma1 * sh
    e2 = z - gamma2 * sh
    q_next = omega_t + rho * q + phi * (e2 * e2 - (1.0 + gamma2 * gamma2 * h))
    h_next = q_next + beta * (h - q) + alpha * (e1 * e1 - (1.0 + gamma1 * gamma1 * h))
    return (h_next.clamp(min=HN_COMPONENT_VARIANCE_FLOOR),
            q_next.clamp(min=HN_COMPONENT_VARIANCE_FLOOR))


def hn_component_daily_advance(Sj, h, q, b_step, z, omega_t, alpha, beta, gamma1, rho, phi, gamma2):
    """One daily component step under the risk-neutral (LRNVR) measure. Returns ``(Sj, h, q)``.

    The log-spot advance is IDENTICAL to :func:`hn_daily_advance` - the component structure lives
    entirely in the variance recursion - so the same note on the REALISED ``z`` applies.
    """
    sh = torch.sqrt(h)
    Sj = Sj * torch.exp((b_step - 0.5 * h) + sh * z)
    h, q = hn_component_variance_step(h, q, sh, z, omega_t, alpha, beta, gamma1, rho, phi, gamma2)
    return Sj, h, q


def hn_component_log_substep(log_S, h, q, z, b_step, omega_t, alpha, beta, gamma1, rho, phi, gamma2):
    """One unmonitored component day, accumulating the LOG increment - the same step as
    :func:`hn_component_daily_advance` with the exponential left to the caller. Kept separate for
    the reason its plain sibling is: the chain the OSS pricers repeat n_sub times per fixing.

    THE INCREMENT IS BUILT FIRST, from the PREDICTABLE h_t, and only then is the state advanced. A
    spelling that rebinds `h` before the `-0.5*h` reads it uses NEXT step's variance for THIS step's
    return, a different model - 1.9e-3 relative on the log path over 40 steps.
    """
    sh = torch.sqrt(h)
    increment = log_S + (b_step - 0.5 * h) + sh * z
    h, q = hn_component_variance_step(h, q, sh, z, omega_t, alpha, beta, gamma1, rho, phi, gamma2)
    return increment, h, q


#: The fused build, and the one every ordinary run takes. SAME RULE AS THE PLAIN SUB-STEP: NOT
#: twice differentiable (AOTAutograd's compiled backward raises `does not currently support double
#: backward`), so a `Greeks: 'All'` valuation walks the eager function above - same numbers.
hn_component_log_substep_fused = torch.compile(hn_component_log_substep, dynamic=True)


def hn_component_unmonitored_substeps(Sj, h, q, b_step, omegas, hnc_params, shared,
                                      num_sims, antithetic):
    """Advance ``(Sj, h, q)`` through ``len(omegas)`` UNCONDITIONAL daily component steps.

    The sibling of :func:`hn_unmonitored_substeps`, and every note there holds. ``omegas`` is the
    per-step intercept strip (its LENGTH is the step count, so a daily fixing passes an empty one);
    ``hnc_params`` = (alpha, beta, gamma1, rho, phi, gamma2).
    """
    if not len(omegas):                                          # a daily fixing walks nothing
        return Sj, h, q
    # picked once per interval, not per step, exactly as the plain sub-step picks it
    substep = hn_component_log_substep if shared.gamma else hn_component_log_substep_fused
    log_S = torch.zeros_like(b_step)
    for omega_t in omegas:
        zc = torch.randn([shared.simulation_batch, num_sims],
                         dtype=shared.one.dtype, device=shared.one.device)
        z = torch.cat([zc, -zc], dim=-1) if antithetic else zc
        log_S, h, q = substep(log_S, h, q, z, b_step, omega_t, *hnc_params)
    return Sj * log_S.exp(), h, q


# The component A/B/C recursion + semi-analytic pricing. A SIBLING of ``hn_ab`` / ``_p1_p2``, NOT a
# generalisation they route through: carrying a second coefficient and a per-step omega through the
# plain loop body would reassociate plain HN's own numbers, and plain HN is the ORACLE for the
# nesting gate. Both hand their log-CF to the same ``cf_european_probabilities`` primitive.

def hn_component_abc(phi, omegas, alpha, beta, gamma1, rho, phi_q, gamma2, r,
                     unwrap=True, phi_dim=-1, terminal=None):
    """``(A, B, C)`` after ``len(omegas)`` backward steps - the last row of
    :func:`hn_component_abc_strip` - satisfying

        E_t[S_{t+n}^phi] = S_t^phi * exp(A + B*h_t + C*q_t)

    so the component affine log-CF of the aggregate log-return is ``A + B*h_0 + C*q_0`` - the
    closure handed to :func:`cf_european_probabilities`.

    ``omegas`` is consumed in REVERSE (the backward induction reaches step t last), so ``omegas[0]``
    is the intercept of the FIRST step - the orientation :func:`hn_component_omega_path` emits.

    ``phi`` : real OR complex tensor; if complex it must vary smoothly and ascending along
    ``phi_dim`` for the branch unwrap of ``log(w)``.

    ``terminal`` : optional ``(u, v)``, the terminal condition ``(B_0, C_0)`` in place of ``(0, 0)``,
    which turns this into the JOINT transform of the return and the state it lands in:

        E_t[exp(phi*R_n + u*h_{t+n} + v*q_{t+n})] = exp(A + B*h_t + C*q_t)

    the one source the stride's carried-state moments are autodiffed out of
    (:func:`hn_component_stride_strip`).  The default ``(0, 0)`` integrates the state out.
    """
    n = len(omegas)
    strip = hn_component_abc_strip(phi, n, alpha, beta, gamma1, rho, phi_q, gamma2, unwrap, phi_dim,
                                   terminal)
    return hn_component_strip_a(strip, omegas, r), strip[3][n], strip[4][n]


def hn_component_logmgf(phi, omegas, h0, q0, alpha, beta, gamma1, rho, phi_q, gamma2, r, **kw):
    """log E_t[exp(phi * R_n)] where R_n = log(S_{t+n}/S_t).  = A + B*h0 + C*q0."""
    A, B, C = hn_component_abc(phi, omegas, alpha, beta, gamma1, rho, phi_q, gamma2, r, **kw)
    return A + B * h0 + C * q0


def hn_component_auto_phi_max(omegas, h0, q0, alpha, beta, gamma1, rho, phi_q, gamma2, r,
                              log_tol=-40.0, start=8.0, cap=2.0 ** 24):
    """Smallest power-of-two phi_max with Re(A + B*h0 + C*q0) - ln(phi) < log_tol.

    The component glue for :func:`cf_adaptive_phi_max`, reducing the batch to the EXTREME states so
    the scan runs on a 4-element state.

    ALL FOUR CORNERS OF THE (h0, q0) BOX, not the two diagonal ones: B and C are free to carry
    OPPOSITE signs, so the slowest-decaying state can be (h.max, q.min), which pairing h with q by
    rank never probes.

    EVERY PRICE DERIVES ITS OWN BOUND - never reuse one across a ladder, because A LARGER BOUND IS
    NOT CONSERVATIVE FOR THIS MODEL: past a parameter- and step-count-dependent point the A/B/C
    recursion DIVERGES rather than decaying. Measured on a converged four-pillar fit, a 126-step
    price is 0.7353 at phi_max 128/256/512, 0.7323 at 1024 and 9.4e+55 at 2048, while the 21-step
    contract in the SAME strip wants 512. :func:`hn_component_strip_phi_max` derives the same
    bound off a strip instead of scanning for it.
    """
    hs, qs = hn_component_state_corners(h0, q0, alpha.dtype, 2)
    carry = torch.as_tensor(r).detach() * len(omegas)
    return cf_adaptive_phi_max(
        lambda z: hn_component_logmgf(z, omegas, hs, qs, alpha, beta, gamma1, rho, phi_q,
                                      gamma2, r), carry, alpha.dtype, alpha.device,
        log_tol, start, cap)


def hn_component_state_corners(h0, q0, dtype, trailing):
    """ALL FOUR ``(h, q)`` corners of a state block, ``(4, 1, ...)`` with ``trailing`` unit axes -
    the two diagonal ones do not bound the decay criterion."""
    h = torch.as_tensor(h0).detach().to(dtype).reshape(-1)
    q = torch.as_tensor(q0).detach().to(dtype).reshape(-1)
    shape = (-1,) + (1,) * trailing
    return (torch.stack([h.min(), h.min(), h.max(), h.max()]).reshape(shape),
            torch.stack([q.min(), q.max(), q.min(), q.max()]).reshape(shape))


def hn_component_abc_strip(phi, n_steps, alpha, beta, gamma1, rho, phi_q, gamma2,
                           unwrap=True, phi_dim=-1, terminal=None):
    """EVERY maturity's ``(A, B, C)`` from ONE backward pass.  Returns ``(phi, a, d, B, C)``.

    THE ALGEBRA, one step back, writing D = B + C, b = alpha*B + D*phi_q and
    G = alpha*B*gamma1 + D*phi_q*gamma2 (so w = 1 - 2b is the Gaussian normalisation of the
    combined quadratic in z, whose two centers gamma1 and gamma2 do NOT coincide):

        A <- A + phi*r + D*omega_t - b - 0.5*log(w)
        B <- -phi/2 + beta*B + (phi - 2G)^2 / (2w)
        C <- D*rho - beta*B                       (B on the RIGHT is the OLD B)

    The ``-b`` is what the two CENTERING subtractions leave once the (1 + gamma^2 h) terms have
    cancelled the quadratics' own h-coefficients; drop it and the price is wrong by a factor that
    grows with the step count. B and C never read the curve and are time-homogeneous, and A is
    affine in it, so with ``a_k`` the cumulative ``-b - 0.5*log(w)`` after k+1 steps and
    ``d_k = B_k + C_k``,

        A_n = a[n-1] + n*phi*r + sum_k d[k] * omegas[n-1-k],   (B_n, C_n) = (B[n], C[n])

    for every ``n <= n_steps`` (:func:`hn_component_strip_a`). ``a`` is ``(n_steps, *phi.shape)``,
    ``B``/``C`` are ``(n_steps+1, *phi.shape)``, and ``d`` is ``B + C`` over the steps flattened to
    ``(n_steps, -1)`` - the matrix the omega curve dots into. ``terminal`` seeds ``(B_0, C_0)``.
    """
    # seeded at the loop's own broadcast shape, as the running recursion widened to on its first step
    B = C = acc = phi.new_zeros(torch.broadcast_shapes(phi.shape, *(
        getattr(p, 'shape', ()) for p in (alpha, beta, gamma1, rho, phi_q, gamma2))))
    if terminal is not None:
        B, C = B + terminal[0], C + terminal[1]
    a, Bs, Cs = [], [B], [C]
    half_phi = 0.5 * phi
    for _ in range(int(n_steps)):
        D = B + C
        Bq = D * phi_q
        b = alpha * B + Bq
        G = alpha * B * gamma1 + Bq * gamma2
        w = 1.0 - 2.0 * b
        logw = complex_log_unwrap(w, dim=phi_dim) if (unwrap and w.is_complex()) else torch.log(w)
        acc = acc - b - 0.5 * logw
        a.append(acc)
        B, C = -half_phi + beta * B + (phi - 2.0 * G) ** 2 / (2.0 * w), D * rho - beta * B
        Bs.append(B)
        Cs.append(C)
    B, C = torch.stack(Bs), torch.stack(Cs)
    n = int(n_steps)
    return phi, torch.stack(a), (B[:n] + C[:n]).reshape(n, -1), B, C


def hn_component_strip_a(strip, omegas, r):
    """``A`` at ``len(omegas)`` steps off :func:`hn_component_abc_strip`: one prefix read and one dot
    product, the curve entering as a tensor so a per-step tensor keeps its graph."""
    phi, a, d, B, _ = strip
    n = len(omegas)
    w = (omegas if torch.is_tensor(omegas) else
         torch.stack([torch.as_tensor(o) for o in omegas]) if n else torch.zeros(0))
    w = w.reshape(-1).flip(0).to(d.dtype)
    return (a[n - 1] if n else 0.0) + (n * r) * phi + (w @ d[:n]).reshape(B.shape[1:])


def hn_component_strip_logcf(strip, omegas, h0, q0, r):
    """``A + B*h0 + C*q0`` at ``len(omegas)`` steps, off :func:`hn_component_abc_strip`. ``h0`` and
    ``q0`` broadcast against the strip's own ``phi`` shape."""
    n = len(omegas)
    return hn_component_strip_a(strip, omegas, r) + strip[3][n] * h0 + strip[4][n] * q0


def hn_component_strip_phi_max(strip, omegas, h0, q0, r, log_tol=-40.0):
    """:func:`hn_component_auto_phi_max`'s bound, READ off a rung strip instead of scanned.

    ``strip`` is :func:`hn_component_abc_strip` over :func:`cf_phi_max_ladder` with both
    inversion contours stacked on axis -3; the primitive names the one it wants by the real offset
    it adds to the ladder.
    EVERY PRICE STILL DERIVES ITS OWN BOUND - the criterion is affine in the omega curve and in the
    state, so only the recursion behind it is shared, never the answer.
    """
    hs, qs = hn_component_state_corners(h0, q0, strip[0].real.dtype, 3)
    if bool(hs[0] == hs[-1]) and bool(qs[0] == qs[-1]):
        hs, qs = hs[:1], qs[:1]            # a scalar state: the four corners are one
    return cf_adaptive_phi_max(
        lambda z: hn_component_strip_logcf(strip, omegas, hs, qs, r)[:, int(z.reshape(-1)[0].real)],
        torch.as_tensor(r).detach() * len(omegas), strip[0].real.dtype, strip[0].device, log_tol)


def _component_p1_p2(logm, omegas, h0, q0, alpha, beta, gamma1, rho, phi_q, gamma2, r,
                     phi_max, panels, order, unwrap, want=3):
    """P1, P2 for log-moneyness ``logm`` = ln(K/S).  ``logm``/``h0``/``q0`` broadcast together.

    The component sibling of :func:`_p1_p2`: the affine log-CF ``A + B*h0 + C*q0`` as the ``logcf``
    closure, ``r*n`` as the P1-contour normalisation. ``want``: 1 = P1, 2 = P2.
    """
    logm = torch.as_tensor(logm, dtype=alpha.dtype, device=alpha.device)
    h0 = torch.as_tensor(h0, dtype=alpha.dtype, device=alpha.device)
    q0 = torch.as_tensor(q0, dtype=alpha.dtype, device=alpha.device)
    logm, h0, q0 = torch.broadcast_tensors(logm, h0, q0)
    if phi_max is None:
        phi_max = hn_component_auto_phi_max(
            omegas, h0, q0, alpha, beta, gamma1, rho, phi_q, gamma2, r)
    if panels is None:
        panels = 256
    hh, qq = h0.unsqueeze(-1), q0.unsqueeze(-1)

    def logcf(phi):
        A, B, C = hn_component_abc(phi, omegas, alpha, beta, gamma1, rho, phi_q, gamma2, r,
                                   unwrap=unwrap)
        return A + B * hh + C * qq

    return cf_european_probabilities(
        logcf, logm, r * len(omegas), phi_max, panels, order, alpha.dtype, alpha.device, want)


def hn_component_call(S, K, omegas, h0, q0, alpha, beta, gamma1, rho, phi_q, gamma2, r,
                      phi_max=None, panels=None, order=8, unwrap=True):
    """European CALL under the component model, ``len(omegas)`` steps to expiry.

    ``h0``/``q0`` are the (predictable) short- and long-run variances of the FIRST step - q0 is L(0)
    by the anchoring - and ``r`` the PER-STEP cost of carry. Differentiable w.r.t. every parameter,
    the omega strip (hence the L curve behind it), h0, q0, S and K.
    """
    S = torch.as_tensor(S, dtype=alpha.dtype, device=alpha.device)
    K = torch.as_tensor(K, dtype=alpha.dtype, device=alpha.device)
    P1, P2 = _component_p1_p2(torch.log(K / S), omegas, h0, q0, alpha, beta, gamma1, rho,
                              phi_q, gamma2, r, phi_max, panels, order, unwrap)
    return S * P1 - K * torch.exp(-r * len(omegas)) * P2


def hn_component_put(S, K, omegas, h0, q0, alpha, beta, gamma1, rho, phi_q, gamma2, r, **kw):
    """European PUT.  By put-call parity off :func:`hn_component_call`, exactly as the plain
    model's put is."""
    S = torch.as_tensor(S, dtype=alpha.dtype, device=alpha.device)
    K = torch.as_tensor(K, dtype=alpha.dtype, device=alpha.device)
    return (hn_component_call(S, K, omegas, h0, q0, alpha, beta, gamma1, rho, phi_q, gamma2, r,
                              **kw) - S + K * torch.exp(-r * len(omegas)))


def hn_component_cdf_logret(x, omegas, h0, q0, alpha, beta, gamma1, rho, phi_q, gamma2, r,
                            phi_max=None, panels=None, order=8, unwrap=True):
    """EXACT  Q( R_n <= x )  where R_n = log(S_{t+n}/S_t), by Fourier inversion - the component
    sibling of :func:`hn_cdf_logret`. Spot-free by construction; ``x``, ``h0`` and ``q0`` broadcast
    together.
    """
    _, P2 = _component_p1_p2(x, omegas, h0, q0, alpha, beta, gamma1, rho, phi_q, gamma2, r,
                             phi_max, panels, order, unwrap, want=2)
    return 1.0 - P2


def hn_component_from_plain(omega, alpha, beta, gamma_star):
    """The plain HN parameters as a point of the COMPONENT parameter space - the exact map the
    nesting rests on.  Returns ``(alpha, beta_c, gamma1, L)`` with phi = 0 and L FLAT.

    beta_c is the plain PERSISTENCE psi = beta + alpha*gamma*^2, because the component's beta is the
    short-run deviation's own AR(1) coefficient and under the plain recursion that deviation decays
    at psi. L is the plain STATIONARY variance, where a flat long-run component must sit.
    """
    psi = hn_persistence(alpha, beta, gamma_star)
    return alpha, psi, gamma_star, (omega + alpha) / (1.0 - psi)


def hn_component_to_plain(alpha, beta, gamma1, level):
    """The inverse of :func:`hn_component_from_plain` - a flat-L, phi=0 component parameter set as
    the plain ``(omega, alpha, beta, gamma_star)`` it is identical to."""
    return level * (1.0 - beta) - alpha, alpha, beta - alpha * gamma1 ** 2, gamma1


# ======================================================================================
# THE STRIDE - the k-step conditional law of ln S given (h, q), cached and exactly differentiable.
#
# The backward A/B/C recursion run k steps gives the k-step conditional log-CF
# exp(A_k + B_k*h + C_k*q) EXACTLY, and A/B/C depend on the parameters, the calendar position and
# the transform node - never on the state - so they are cached once per (fixing interval x
# quadrature node) and every per-path question after that is a dot product over the cache. The
# omega strip enters A alone, so B_k/C_k are reusable across intervals of equal length.
#
# THE CASE SPLIT IS A RULE: a daily-monitored contract keeps the daily path and the one-day
# probability permanently, the daily advance being the reference implementation. Where a pricer
# strides the WHOLE fixing interval (`pv_MC_Tarf`, `pv_MC_Accumulator`) such a contract's interval
# is one day and the stride is INERT rather than absent - the one-step law IS the daily law exactly,
# 1.8e-11 relative on/off, against 2.9e-2 on the monthly schedule the stride is for.
# ======================================================================================

#: The mixed partial derivatives of the joint transform at the origin that the carried-state
#: approximation is pinned by, in the order :data:`HNComponentStride.mom` stores them. ``p`` is
#: d/dphi (the return), ``u`` d/du (the short-run variance it lands in), ``v`` d/dv (the long-run
#: one); a key is the multiset of derivatives, so ``upp`` is d3/du dphi^2. Every one is a JOINT
#: CUMULANT of (R_k, h_k, q_k), the transform being a log-MGF - which is why the normal equations
#: below are written in central moments and read straight off this block.
HN_STRIDE_MOMENT_KEYS = ('p', 'pp', 'ppp', 'pppp',
                         'u', 'up', 'upp', 'uu',
                         'v', 'vp', 'vpp', 'vv', 'uv')

#: One fixing interval's cached stride. ``nodes``/``wts`` are that interval's own Gauss-Legendre
#: quadrature (its own ``phi_max`` - a bound is NOT transferable between step counts, see
#: :func:`hn_component_auto_phi_max`); ``A``, ``B``, ``C`` are the complex coefficient strips on the
#: ``i*phi`` inversion contour, shape ``(node,)``; ``mom`` is the (13, 3) real block of origin
#: derivatives - row per :data:`HN_STRIDE_MOMENT_KEYS`, column ``(A-part, B-part, C-part)`` - so a
#: per-path moment is ``mom[i,0] + mom[i,1]*h + mom[i,2]*q``.
HNComponentStride = namedtuple(
    'HNComponentStride', 'n_steps nodes wts phi_max A B C mom r')

#: The quadratic carried-state fit for one cube of paths: ``h_k ~ a_h + b_h*y + c_h*y^2`` in the
#: CENTERED return ``y = x - mean_x``, plus the residual scale/correlation.
HNComponentStrideCarry = namedtuple(
    'HNComponentStrideCarry', 'mean_x a_h b_h c_h a_q b_q c_q sd_h sd_q corr')

#: Floor on a residual standard deviation before it divides into a correlation. Numerically zero.
HN_STRIDE_TINY = 1.0e-300

#: Complex elements per block inside the inversion loop, the ONE place the stride can bound its own
#: footprint (it runs under ``no_grad``, so a block frees before the next). EVERYTHING ELSE IS
#: O(paths x nodes) AND THAT, NOT TIME, IS WHAT LIMITS A CUBE: a differentiable draw holds about six
#: (paths, node) complex128 buffers alive for the backward pass, so a 512-node strip fits ~2^18
#: paths on a 24 GB device. 2^24 here is 256 MB a block.
HN_STRIDE_INVERT_BLOCK = 1 << 24


def _hn_stride_moment_block(omegas, hnc_params, r):
    """The (13, 3) origin-derivative block of the JOINT transform, by autograd on
    :func:`hn_component_abc` - no hand-written moment recursions anywhere.

    With the terminal condition ``(B_0, C_0) = (u, v)`` that recursion returns
    ``M(phi,u,v) = log E_t[exp(phi*R_k + u*h_{t+k} + v*q_{t+k})]``, so every mixed partial at the
    origin is a joint CUMULANT of ``(R_k, h_k, q_k)``. ONE autograd chain delivers all three
    coefficient parts: the recursion is ELEMENTWISE in ``phi``, so running it on a 3-vector against
    the probes ``h = (1,0,0)``, ``q = (0,1,0)`` evaluates ``(A+B, A+C, A)`` in parallel.
    ``create_graph`` throughout, which is what makes the carried state's gradient real rather than a
    detached constant.

    IT FORCES ``enable_grad`` AND RESTORES THE AMBIENT MODE ON THE WAY OUT: these derivatives are
    the cache's VALUE, not a gradient of the caller's computation. A valuation builds its cache
    inside ``no_grad``, where the recursion records no graph, every partial is a structural zero and
    the carry divides by ``mu2 = 0`` - measured, a NaN on all 8,192 paths at the first stride with a
    healthy Phi beside it.
    """
    ambient = torch.is_grad_enabled()
    with torch.enable_grad():
        block = _hn_stride_origin_derivatives(omegas, hnc_params, r)
    return block if ambient else block.detach()


def _hn_stride_origin_derivatives(omegas, hnc_params, r):
    """The chain itself - see :func:`_hn_stride_moment_block`, which owns the mode handling."""
    alpha = hnc_params[0]
    dt, dev = alpha.dtype, alpha.device
    z3 = torch.zeros(3, dtype=dt, device=dev)
    phi, u, v = (z3.clone().requires_grad_(True) for _ in range(3))
    hp = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)
    qp = torch.tensor([0.0, 1.0, 0.0], dtype=dt, device=dev)
    A, B, C = hn_component_abc(phi, omegas, *hnc_params, r, unwrap=False, terminal=(u, v))
    M = A + B * hp + C * qp

    def d(y, wrt):
        # a k=1 stride is EXACTLY Gaussian in the return, so the phi chain legitimately terminates
        # at the second cumulant and every higher one is identically zero - not an error condition
        if not y.requires_grad:
            return torch.zeros_like(z3)
        g = torch.autograd.grad(y.sum(), wrt, create_graph=True, allow_unused=True)[0]
        return torch.zeros_like(z3) if g is None else g

    o = {'p': d(M, phi), 'u': d(M, u), 'v': d(M, v)}
    o['pp'] = d(o['p'], phi)
    o['ppp'] = d(o['pp'], phi)
    o['pppp'] = d(o['ppp'], phi)
    o['up'] = d(o['u'], phi)
    o['upp'] = d(o['up'], phi)
    o['uu'] = d(o['u'], u)
    o['uv'] = d(o['u'], v)
    o['vp'] = d(o['v'], phi)
    o['vpp'] = d(o['vp'], phi)
    o['vv'] = d(o['v'], v)
    # probe 2 is A alone; probes 0 and 1 are A+B and A+C, so B and C come out by subtraction
    return torch.stack([torch.stack([o[k][2], o[k][0] - o[k][2], o[k][1] - o[k][2]])
                        for k in HN_STRIDE_MOMENT_KEYS])


def hn_component_stride_strip(omegas, hnc_params, r, h_box, q_box, phi_max=None, panels=None,
                              order=8, unwrap=True, moments=True):
    """Build ONE fixing interval's cached stride over the ``omegas`` strip (its length IS k).

    ``hnc_params`` = ``(alpha, beta, gamma1, rho, phi, gamma2)``, the same block every other
    component entry point takes.  ``h_box``/``q_box`` are the state RANGES the cube will reach - the
    adaptive quadrature bound is resolved once, here, off all four corners of that box, and a strip
    is only valid for states inside it (a bound is not transferable, and a LARGER one is not
    conservative: past a step-count-dependent point the recursion diverges).

    ``moments=False`` builds the Phi coefficients alone - the branch-and-weight and
    conditional-probability consumers never carry state, so they never pay for the origin block.

    THE COST OF CARRY IS A SHIFT, NOT A REBUILD: ``r`` enters A only as ``phi*r*k``, which the
    inversion multiplies against ``exp(-i phi x)``, so a strip built at ``r_0`` answers for ANY
    other per-step carry ``b`` - including a per-path one - by moving the moneyness:

        Q_b(R_k <= x | h, q)  ==  hn_component_stride_cdf(strip, x - (b - r_0)*k, h, q)

    and the carried state is untouched but for its mean, which moves by the same ``(b - r_0)*k``
    (measured 2.2e-16 on Phi, every carry loading bitwise identical). So one strip per (anchor,
    length) serves a whole book of deals whose ``b_step`` differs, which matters because ``b_step``
    reaches the OSS pricers as a per-path tensor.

    THE SHIFT RUNS BOTH WAYS, and a consumer that only moves the moneyness IN has done half of it: a
    barrier enters as ``x_cap - (b - r_0)*k`` and what comes back is a return in the strip's OWN
    ``r_0`` measure, so THE RETURN UN-SHIFTS by ``+(b - r_0)*k`` before it may move a spot. Skip
    that and the survival WEIGHT is still right while the whole survivor law sits a carry too low -
    27x the daily walk's own quantile band at k = 21, b = 4r.
    :func:`hn_component_stride_step` does the un-shift when it is handed ``b_step``.
    """
    alpha = hnc_params[0]
    dt, dev = alpha.dtype, alpha.device
    omegas = list(omegas)
    if phi_max is None:
        phi_max = hn_component_auto_phi_max(omegas, h_box, q_box, *hnc_params, r)
    if panels is None:
        panels = 256
    nodes, wts = gauss_legendre(0.0, float(phi_max), panels, order, dt, dev)
    A, B, C = hn_component_abc(nodes * 1j, omegas, *hnc_params, r, unwrap=unwrap)
    mom = _hn_stride_moment_block(omegas, hnc_params, r) if moments else None
    return HNComponentStride(len(omegas), nodes, wts, float(phi_max), A, B, C, mom, r)


def _hn_stride_logcf(strip, h, q):
    """The cached log-CF ``A + B*h + C*q`` on the node axis - shape ``(*batch, node)``."""
    return strip.A + strip.B * h.unsqueeze(-1) + strip.C * q.unsqueeze(-1)


def _hn_stride_cast(strip, *xs):
    """Broadcast the per-path arguments onto one shape in the strip's dtype/device."""
    return torch.broadcast_tensors(
        *[torch.as_tensor(x, dtype=strip.nodes.dtype, device=strip.nodes.device) for x in xs])


def hn_component_stride_factor(strip, h, q):
    """``E = exp(A + B h + C q)`` on the node axis - THE ONE STATE-DEPENDENT OBJECT the stride has.

    Everything after it is a contraction over this cache: the survival p, each ``F(x)``/``f(x)``,
    the Newton residuals, the fired branch's tilted mass, the backward polish terms. Built once per
    fixing and handed down, it costs a survival-truncated draw one build where the contractions
    would otherwise ask for five - and it is the same tensor either way, so nothing about a strip's
    numbers moves with the sharing.
    """
    h, q = _hn_stride_cast(strip, h, q)
    return torch.exp(_hn_stride_logcf(strip, h, q))


def hn_component_stride_cdf(strip, x, h, q, factor=None):
    """``Q(R_k <= x | h, q)`` over the cache - the stride's Gil-Pelaez Phi.

    THE SAME QUADRATURE `hn_component_cdf_logret` RUNS, with the A/B/C recursion replaced by the
    cached strips and nothing else changed - same nodes, weights and assembly order, so the two
    agree bit-for-bit at equal ``phi_max``/``panels``/``order`` (gated). ``x``, ``h`` and ``q``
    broadcast together. Differentiable w.r.t. ``x``, the state, and every model parameter.

    ``factor`` is this state's :func:`hn_component_stride_factor`, to avoid rebuilding it.
    """
    x, h, q = _hn_stride_cast(strip, x, h, q)
    shift = torch.exp(-1j * strip.nodes * x.unsqueeze(-1)) / (strip.nodes * 1j)
    d = (shift * (hn_component_stride_factor(strip, h, q) if factor is None else factor)).real
    return 1.0 - (0.5 + (d * strip.wts).sum(-1) / np.pi)


def hn_component_stride_pdf(strip, x, h, q, factor=None):
    """The density of the k-step log-return, ``dQ/dx`` - the same inversion without ``1/(i phi)``
    (differentiating under the integral cancels it).  Used to reattach the exact gradient to the
    inverted draw, and as the Newton slope inside the inversion."""
    x, h, q = _hn_stride_cast(strip, x, h, q)
    d = (torch.exp(-1j * strip.nodes * x.unsqueeze(-1))
         * (hn_component_stride_factor(strip, h, q) if factor is None else factor)).real
    return (d * strip.wts).sum(-1) / np.pi


def hn_component_stride_cumulants(strip, h, q):
    """The first four cumulants of the k-step log-return given ``(h, q)``, off the cached origin
    block - ``(mean, variance, skew, excess kurtosis)``. Three multiply-adds per path per cumulant,
    and the seed the inversion starts from."""
    h, q = _hn_stride_cast(strip, h, q)
    m = strip.mom
    k = [m[i, 0] + m[i, 1] * h + m[i, 2] * q for i in range(4)]
    return k[0], k[1], k[2] / k[1] ** 1.5, k[3] / k[1] ** 2


def hn_component_stride_invert(strip, p, h, q, iters=32, tol=1.0e-14, chunk=None, factor=None):
    """Solve ``Q(R_k <= x | h, q) = p`` for x, per path.  VALUE ONLY - no gradient.

    CORNISH-FISHER SEED, then safeguarded Newton. The seed is free - the cache already holds the
    exact first four cumulants - and it is what keeps the iteration count down, the inversion being
    the ONLY part of the stride that is not a single pass over the cache. Newton then runs on the
    cached density with a bisection fallback whenever a step leaves the running bracket, which is
    what makes this safe where Cornish-Fisher is not monotone.

    The state-dependent factor ``exp(A + B h + C q)`` is HOISTED OUT of the loop and the rotation
    ``exp(-i phi x)`` is shared by the residual and the slope, so an iteration is ONE complex
    exponential and two dot products over the node axis - a reassociation of
    :func:`hn_component_stride_cdf`, not a second formula.

    The gradient is not lost, it is DEFERRED: :func:`hn_component_stride_draw` reattaches it by two
    graph-carrying Newton steps at this root - one for the implicit function theorem, the second for
    the O(1) term a detached starting point drops at second order.

    IT REFUSES RATHER THAN RETURNING AN UNPOLISHED ROOT: every second-order term downstream is built
    by differentiating THIS equation at THIS root, and the implicit function theorem says nothing
    about a point that does not solve it.

    THE REFUSAL IS PER PATH, AND ONLY OF PATHS THE LAW REACHES. The two breaks above are collective,
    so one stalled path holds every other open, and the ONE population that stalls is the one whose
    target lies outside the law: past about mean - 5 sd the strip's own quadrature error exceeds the
    probability being asked for, no bracket exists, and the widening loop walks off to an ``x`` of
    1e3 to 1e5. Those paths are MARKED as they widen and exempted, because their answer is a
    pre-existing saturation of the deep tail and not a convergence question - measured at k = 126,
    a mean - 8 sd cap and 32,768 paths: 228 such, against 0 paths that stall inside the law's reach
    in any band from the untruncated draw down to that cap. What is refused is a path whose own
    +/-4 sd bracket held the root and which still did not resolve it - a starved iteration count or
    a NaN - and one is enough to refuse the call.

    ``chunk`` blocks the path axis at :data:`HN_STRIDE_INVERT_BLOCK` complex elements. The iteration
    is elementwise, but the CONVERGENCE BREAK is collective, so a different blocking stops one
    iteration earlier or later and the roots agree to ``tol`` rather than bitwise (measured 7.1e-15
    in x between a 5,000-path solve and the same paths in blocks of 7).

    ``factor`` is the state's :func:`hn_component_stride_factor` and must ALREADY BE AT THE CAST
    SHAPE - one row per path this call solves. It is blocked alongside ``p`` and a factor that
    merely broadcasts against it would be sliced out of step, so the shape is checked rather than
    broadcast: the caller that has one has already cast its arguments together. THE GUARD IS HERE
    AND NOWHERE ELSE because this is the only verb that slices the path axis - the CDF and the
    density take a factor unchecked, having nothing to do with it but broadcast.
    """
    p, h, q = _hn_stride_cast(strip, p, h, q)
    n_node = strip.nodes.numel()
    if factor is not None and tuple(factor.shape) != tuple(p.shape) + (n_node,):
        raise ValueError(
            'HN_Stride: a state factor of shape {} was handed to a {}-path inversion, which wants '
            '{}. It is blocked with the path axis, so a broadcastable-but-narrower one would be '
            'sliced out of step - build it from the arguments AFTER they are cast '
            'together.'.format(tuple(factor.shape), p.numel(), tuple(p.shape) + (n_node,)))
    if chunk is None:
        chunk = max(1, int(HN_STRIDE_INVERT_BLOCK // n_node))
    if p.numel() > chunk:
        shape, pf, hf, qf = p.shape, p.reshape(-1), h.reshape(-1), q.reshape(-1)
        ff = None if factor is None else factor.reshape(-1, n_node)
        return torch.cat([hn_component_stride_invert(
            strip, pf[i:i + chunk], hf[i:i + chunk], qf[i:i + chunk], iters, tol, chunk,
            None if ff is None else ff[i:i + chunk])
            for i in range(0, pf.numel(), chunk)]).reshape(shape)
    with torch.no_grad():
        mom = strip.mom
        if mom is None:                                 # no origin block: a plain +/-N sd bracket
            mean, sd = torch.zeros_like(p), torch.ones_like(p) * float(strip.n_steps) ** 0.5
            seed = mean
        else:
            mean, var, g1, g2 = hn_component_stride_cumulants(strip, h, q)
            sd = var.clamp_min(HN_STRIDE_TINY).sqrt()
            z = norm_icdf(p.clamp(1.0e-15, 1.0 - 1.0e-15))
            w = (z + (z * z - 1.0) * g1 / 6.0 + (z * z * z - 3.0 * z) * g2 / 24.0
                 - (2.0 * z * z * z - 5.0 * z) * g1 * g1 / 36.0)
            seed = mean + sd * w.clamp(-9.0, 9.0)
        nodes, wts = strip.nodes, strip.wts
        cf = hn_component_stride_factor(strip, h, q) if factor is None else factor
        w_cdf, w_pdf = cf * wts / (nodes * 1j), cf * wts

        def rotate(xx):
            """``exp(-i phi x)`` on the node axis - the one transcendental an iterate costs."""
            return torch.exp(-1j * nodes * xx.unsqueeze(-1))

        def phi_of(xx, e=None):
            e = rotate(xx) if e is None else e
            return 1.0 - (0.5 + (e * w_cdf).real.sum(-1) / np.pi)

        def dens(xx, e=None):
            e = rotate(xx) if e is None else e
            return (e * w_pdf).real.sum(-1) / np.pi

        lo, hi = torch.minimum(seed, mean) - 4.0 * sd, torch.maximum(seed, mean) + 4.0 * sd
        beyond = torch.zeros_like(p, dtype=torch.bool)
        for _ in range(16):                       # widen until the bracket actually brackets
            below, above = phi_of(lo) > p, phi_of(hi) < p
            if not bool(below.any() or above.any()):
                break
            # A WIDENED PATH IS ONE ASKED FOR A QUANTILE THE LAW DOES NOT REACH - four standard
            # deviations past a seed exact to four cumulants already spans it - so it is marked and
            # exempted from the convergence refusal below
            beyond |= below | above
            lo = torch.where(below, mean - 2.0 * (mean - lo), lo)
            hi = torch.where(above, mean + 2.0 * (hi - mean), hi)
        x = seed.clamp(min=lo, max=hi)
        worst, n_open = float('nan'), 0
        settled = False
        for _ in range(iters):
            # F(x) AND f(x) ARE TWO CONTRACTIONS OF ONE ROTATION: they are asked at the same x
            # every iteration, and the rotation is the only transcendental in the loop
            rot = rotate(x)
            fx = phi_of(x, rot) - p
            lo = torch.where(fx <= 0.0, x, lo)
            hi = torch.where(fx > 0.0, x, hi)
            worst = float(fx.abs().max())
            if worst < tol:
                settled = True
                break
            step = x - fx / dens(x, rot).clamp_min(HN_STRIDE_TINY)
            # NON-STRICT against the bracket: a CONVERGED path has just set an endpoint to its own
            # x, and a strict test reads its Newton step as "outside" and bisects a bracket still
            # four standard deviations wide, throwing the root away
            ok = torch.isfinite(step) & (step >= lo) & (step <= hi)
            nxt = torch.where(ok, step, 0.5 * (lo + hi))
            done = bool(((nxt - x).abs() <= tol * sd).all())
            x = nxt
            if done:
                settled = True
                break
        if not settled:
            # THE BREAKS ARE COLLECTIVE, so one path stalling holds every other one open. The
            # refusal is therefore asked per path, and only of paths the law actually reaches
            fx = (phi_of(x) - p).abs()
            open_ = ~(fx < tol) & ~beyond
            settled = not bool(open_.any())
            n_open = int(open_.sum())
            worst = float(fx[open_].max()) if n_open else float(fx.max())
    if not settled:
        raise ValueError(
            'HN_Stride: {} of {} paths did not converge in {} iterations - the worst is {:.3g} '
            'away in probability against a tolerance of {:.3g}, and every one of them was asked '
            'for a quantile INSIDE the law\'s own reach. That root is not returned because every '
            'term built on it differentiates THIS equation AT a solution of it: the draw\'s '
            'gradient is the implicit function theorem and its second derivative two polish steps '
            'from the root, neither of which says anything at a point that does not solve it. A '
            'NaN in the inversion is the usual cause and means the strip is not a probability - '
            'raise the state floor (`pricing.ComponentHestonNandiKit.stride_state_floor`) or price '
            'with HN_Stride off.'.format(n_open, p.numel(), iters, worst, tol))
    return x


def hn_component_stride_draw(strip, u, h, q, x_cap=None, iters=32, tol=1.0e-14, factor=None):
    """SURVIVAL-TRUNCATED inverse-CDF draw of the k-step log-return.  Returns ``(x, phi_cap)``.

    ``u`` is uniform on (0,1) and the draw solves ``Q(R_k <= x) = u * Phi_cap`` with
    ``Phi_cap = Q(R_k <= x_cap)``, the one-sided survival mass the OSS truncation carries.
    ``x_cap=None`` draws from the untruncated law; the caller multiplies the survival weight into
    its own running product.

    THE GRADIENT IS EXACT, not a differentiated iteration. The root is found under ``no_grad``, then
    corrected by Newton steps whose terms carry the graph:

        x <- x + (u*Phi_cap - Q(x)) / q(x*)

    Each correction is numerically zero, so the VALUE is the root, while the first step's derivative
    is the implicit function theorem for this equation. The density divides as a detached constant,
    correctly - it multiplies a term that vanishes.

    TWO STEPS, AND THE SECOND IS WHAT MAKES THE SECOND DERIVATIVE EXIST. Writing
    ``g(x, theta) = Q(x, theta) - target(theta)``, one step off a DETACHED root has ``dx = 0`` going
    in, so it returns the right FIRST derivative but ``-g_thetatheta / q`` at second order, missing
    the ``g_xx dx^2 + 2 g_xtheta dx`` only a starting point CARRYING ``dx`` supplies. A second step
    starts from an ``x`` whose first derivative is already exact and closes it, at one more
    inversion pass. Measured at k = 21: one step reads ``d2x`` 3.1% out, two steps 0.002%, and the
    VALUE is the inverter's root to the last bit either way. Downstream it is worth 36% of an
    autocall's gamma.

    EACH CORRECTION IS BOUNDED BY ONE STANDARD DEVIATION of the k-step law, which never binds in
    normal running. It is there because the density is a divisor, and a quadrature underflowed to
    zero in a deep tail would turn a rounding residual into an infinity.

    ONE STATE FACTOR SERVES THE WHOLE DRAW - the cap's Phi, the inverter, the density at the root
    and both polish steps are five contractions of the same ``exp(A + B h + C q)``, built once here.
    """
    u, h, q = _hn_stride_cast(strip, u, h, q)
    e = hn_component_stride_factor(strip, h, q) if factor is None else factor
    phi_cap = (torch.ones_like(u) if x_cap is None
               else hn_component_stride_cdf(strip, x_cap, h, q, e))
    target = u * phi_cap
    root = hn_component_stride_invert(
        strip, target.detach(), h.detach(), q.detach(), iters, tol, factor=e.detach())
    dens = hn_component_stride_pdf(strip, root, h, q, e).detach().clamp_min(HN_STRIDE_TINY)
    bound = (hn_component_stride_cumulants(strip, h, q)[1].detach().clamp_min(0.0).sqrt()
             if strip.mom is not None else torch.ones_like(dens))
    x = root
    for _ in range(2):
        x = x + ((target - hn_component_stride_cdf(strip, x, h, q, e)) / dens).clamp(
            min=-bound, max=bound)
    return x, phi_cap


def sqrt_or_zero(v):
    """``sqrt(max(v, 0))`` with a ZERO gradient where ``v <= 0``: ``clamp_min(0).sqrt()`` is the
    same value but its backward is ``inf * 0`` there, a NaN on every leaf behind it - measured on a
    fit with ``Phi`` exactly 0, where the long-run component's residual variance is exactly 0."""
    positive = v > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, v, torch.ones_like(v))),
                       torch.zeros_like(v))


def hn_component_stride_carry_loadings(strip, h, q):
    """THE CARRIED-STATE APPROXIMATION, pinned.  Returns a :data:`HNComponentStrideCarry`.

    ``S_k`` is drawn exactly; the state it lands in is not, and cannot be - the conditional law of
    ``(h_k, q_k)`` given the realised return has no closed form. It is carried by QUADRATIC
    conditional matching, and quadratic is a statement about KIND: ``E[h_k | x]`` is the news-impact
    curve, an asymmetric U tilted by gamma_1, so a LINEAR carry gets the vol-of-vol convexity
    structurally wrong however well it is fitted.

    ``h_k ~ a + b*y + c*y^2`` in the centered return ``y = x - E[x]``, with (a, b, c) the exact L2
    projection onto ``span{1, y, y^2}``.  In central moments ``mu2, mu3, mu4`` of the return the
    normal equations are

        [ 1    0    mu2 ] [a]   [ E[h_k]                ]
        [ 0    mu2  mu3 ] [b] = [ Cov(h_k, y)           ]
        [ mu2  mu3  mu4 ] [c]   [ E[h_k y^2]            ]

    and EVERY entry is a joint cumulant off :func:`_hn_stride_moment_block`: mu2 = M_pp,
    mu3 = M_ppp, mu4 = M_pppp + 3 mu2^2, Cov(h_k,y) = M_up, E[h_k y^2] - E[h_k] mu2 = M_upp. So the
    3x3 collapses to a 2x2 whose determinant is the Gram determinant of (y, y^2), positive for any
    non-degenerate law.  q rides the same algebra with its own slower loadings.

    The residual is matched in VARIANCE and in the h-q residual CORRELATION off the closed-form
    covariance: the fit being an orthogonal projection, the residual variance is
    ``Var(h_k) - (b*M_up + c*M_upp)`` exactly, and the residual covariance must equal the same
    expression with the roles swapped - a free consistency check the gate reads. NOT matched: the
    residual's heteroskedasticity in x, or any shape past the second moment. That is the
    approximation, and its size is non-monotone in k.
    """
    h, q = _hn_stride_cast(strip, h, q)
    mom = strip.mom
    m = {k: mom[i, 0] + mom[i, 1] * h + mom[i, 2] * q
         for i, k in enumerate(HN_STRIDE_MOMENT_KEYS)}
    mu2, mu3 = m['pp'], m['ppp']
    mu4 = m['pppp'] + 3.0 * mu2 ** 2
    spread = mu4 - mu2 ** 2                                   # Var(y^2)
    den = mu2 * spread - mu3 ** 2                             # the (y, y^2) Gram determinant
    b_h = (m['up'] * spread - m['upp'] * mu3) / den
    c_h = (mu2 * m['upp'] - mu3 * m['up']) / den
    b_q = (m['vp'] * spread - m['vpp'] * mu3) / den
    c_q = (mu2 * m['vpp'] - mu3 * m['vp']) / den
    sd_h = sqrt_or_zero(m['uu'] - (b_h * m['up'] + c_h * m['upp']))
    sd_q = sqrt_or_zero(m['vv'] - (b_q * m['vp'] + c_q * m['vpp']))
    cov = m['uv'] - (b_q * m['up'] + c_q * m['upp'])
    return HNComponentStrideCarry(
        m['p'], m['u'] - c_h * mu2, b_h, c_h, m['v'] - c_q * mu2, b_q, c_q, sd_h, sd_q,
        (cov / (sd_h * sd_q).clamp_min(HN_STRIDE_TINY)).clamp(-1.0, 1.0))


def hn_component_stride_carry(strip, x, h, q, e1, e2, loadings=None):
    """Carry ``(h, q)`` across the stride onto the realised return ``x``.  Returns ``(h_k, q_k)``.

    ``e1``/``e2`` are independent standard normals, correlated by the 2x2 Cholesky of the residual
    covariance :func:`hn_component_stride_carry_loadings` closed-form. Both states are FLOORED at
    :data:`HN_COMPONENT_VARIANCE_FLOOR` - the same declared floor the daily recursion carries, not a
    repair of the approximation.  Pass ``loadings`` to reuse a fit across a cube.
    """
    x, h, q = _hn_stride_cast(strip, x, h, q)
    ld = hn_component_stride_carry_loadings(strip, h, q) if loadings is None else loadings
    y = x - ld.mean_x
    fit_h = ld.a_h + ld.b_h * y + ld.c_h * y * y
    fit_q = ld.a_q + ld.b_q * y + ld.c_q * y * y
    n2 = ld.corr * e1 + sqrt_or_zero(1.0 - ld.corr ** 2) * e2
    return ((fit_h + ld.sd_h * e1).clamp(min=HN_COMPONENT_VARIANCE_FLOOR),
            (fit_q + ld.sd_q * n2).clamp(min=HN_COMPONENT_VARIANCE_FLOOR))


def hn_component_stride_step(strip, Sj, h, q, u, e1, e2, x_cap=None, loadings=None, b_step=None):
    """ONE STRIDE: jump the spot k steps on a survival-truncated draw and carry the state with it.
    Returns ``(Sj, h, q, phi_cap)`` - the verb an OSS pricer calls in place of k
    :func:`hn_component_log_substep` days plus a truncated final :func:`hn_component_daily_advance`.

    ``b_step`` IS THE MEASURE THE SPOT MOVES UNDER and need not be the ``r`` the strip was built at,
    which is the point of keying the cache on calendar position alone. The draw and the CARRY both
    happen in the strip's own ``r`` measure - the loadings centre on that mean - and the RETURN
    alone is un-shifted by ``(b_step - r)*k`` on the way out. The caller passes ``x_cap`` shifted
    the other way, into that same measure.

    ``b_step=None`` leaves the return where the strip put it, right exactly when the deal's carry IS
    the strip's ``r``.
    """
    x, phi_cap = hn_component_stride_draw(strip, u, h, q, x_cap)
    h, q = hn_component_stride_carry(strip, x, h, q, e1, e2, loadings)
    if b_step is not None:
        x = x + (b_step - strip.r) * strip.n_steps
    return Sj * torch.exp(x), h, q, phi_cap


# ======================================================================================
# LOGVAR2FJ - two-factor log-variance with co-jumps, walked on an INTERNAL step and priced through
# the sample-then-Phi stride. Theory and the phase-0 gates: logvar2fj_spec.md.
#
# One Markov chain on an internal step delta; a stride from any date to any date is a BLOCK of that
# chain, so two dates give the same law however the interval between them is cut. Given every
# step's two variance shocks and its jump count the block return is EXACTLY Gaussian - which is
# what makes survival one Phi, the truncated draw one Phi^-1 and a partial moment Black, so the
# lognormal primitives in `pricing` serve this model verbatim and nothing is integrated inside a
# stride.
#
# The math is FREE FUNCTIONS taking one parameter dict keyed by the canonical names below - the set
# the price factor declares; `pricing.LogVar2FJKit` owns the unpack, the grid and the draws.
# ======================================================================================

#: The `LogVar2FJModelParameters` price factor's SCALAR leaves, in canonical order - the single
#: source of that name set, shared with the riskfactors class and every consumption site.
LV_PARAM_NAMES = ('Kappa_L', 'Sigma_L', 'Rho_L', 'Kappa_S', 'Sigma_S', 'Sigma_J', 'Nu')

#: The parameters that are STRUCTURAL rather than leaves: the jump intensity, whose whole content
#: is the law of integer counts and so is not on the tape, and the cap, a guard on log-variance
#: rather than a modelling device. A leaf at either would report a derivative nothing carries.
LV_STRUCTURAL_NAMES = ('Lambda', 'Cap_A', 'Cap_Beta')

#: The two forward-skew levers, piecewise CONSTANT on calendar-time buckets whose START times are
#: the curves' knots (spec 2.3.1) - one knot at 0 is the constant-parameter model. Both carry the
#: same buckets, which the factor asserts.
LV_BUCKET_NAMES = ('Rho_S', 'Mu_J')

#: The CURVE parameters, in the order the kit unpacks them. `L_Curve` is log annualised DIFFUSIVE
#: variance at knots in years, piecewise linear between them and flat outside; all three carry
#: structural knots and VALUES that are leaves.
LV_CURVE_NAMES = ('L_Curve',) + LV_BUCKET_NAMES

#: The floor on the idiosyncratic share c = 1 - rho_s(t)^2 - rho_l^2, asserted in every bucket at
#: parameter load (spec 2.2.2): below it the truncation conditions on nothing and a surface that
#: wants it is asking for a one-shock model.
LV_C_MIN = 0.12

#: The model's only guard, taken in each division by a block standard deviation (spec 2.8) and
#: nowhere else. Small enough to be inert, large enough to bound z at ~1e12.
LV_SIGMA_TINY = 1.0e-12

#: Jumps per internal step past which the inverse-CDF count truncates. At lam*delta ~ 0.006 the
#: truncated mass is 5e-11; at a 21-day step it is 9e-6 and drifts a 5y forward by 4e-5.
LV_MAX_JUMPS = 3


def lv_counts(u, lam, deltas):
    """Poisson(lam*delta) counts per step by inverse CDF, truncated at 3 (spec 2.5).

    Integer and off the tape: lam is structural, so nothing here carries a gradient.
    """
    a = float(lam) * deltas
    pmf = torch.exp(-a)
    cdf = pmf.clone()
    N = torch.zeros(u.shape, dtype=torch.int8, device=u.device)
    for n in range(1, LV_MAX_JUMPS + 1):
        N = N + (u > cdf).to(torch.int8)
        pmf = pmf * a / n
        cdf = cdf + pmf
    return N


def lv_cap(x, a, beta):
    """Smooth cap on log-variance, a - beta*softplus((a-x)/beta) (spec 2.7); a None is uncapped."""
    if a is None:
        return x
    return a - beta * torch.nn.functional.softplus((a - x) / beta)


def lv_ou_step_weights(kappa, sigma, deltas):
    """Per-step decay phi and shock weight w of an OU factor on a non-uniform grid.

    A ZERO-length step - an MTM row landing on a remaining fixing hands the walk one - makes the
    radicand an exact zero, whose `sqrt` backward is `inf * 0`; `sqrt_or_zero` gives the weight's
    own derivative there, which is zero because the weight is identically zero in delta.
    """
    phi = torch.exp(-kappa * deltas)
    return phi, sigma * sqrt_or_zero((1.0 - phi * phi) / (2.0 * kappa))


def lv_filter_matmul(kappa, sigma, deltas):
    """Lower-triangular OU filter over a non-uniform grid, its weights and initial decay.

    F[k, j] = prod_{i=j+1..k} phi_i for j <= k; with per-step input x_j the state after step k
    is state_{k+1} = d[k]*state_0 + (x @ F.T)[..., k].
    """
    t = torch.cumsum(deltas, dim=0)
    _, w = lv_ou_step_weights(kappa, sigma, deltas)
    # Mask the exponent, not the exponential: exp(+kappa dt) above the diagonal overflows
    # for large kappa * t_n, and tril's zero gradient would then be 0 * inf.
    F = torch.tril(torch.exp(-kappa * torch.tril(t[:, None] - t[None, :])))
    return F, w, torch.exp(-kappa * t)


def lv_block_ends(block_of_step):
    """Index of the last step of each block."""
    return torch.cumsum(torch.bincount(block_of_step), dim=0) - 1


def lv_walk(params, curve_at_grid, deltas, eta_l, eta_s, counts, state0,
            method='matmul', want_paths=False, blocks=None):
    """Walk both log-variance factors and form each step's return mean and variance.

    eta_l, eta_s, counts are [batch, sims, n]; curve_at_grid is L at the n+1 grid times and
    params['Rho_S'], params['Mu_J'] are the bucket values in force at each step start;
    state0 = (l0, s0) is [batch, sims]. Returns (m_x, var, s_path, l_path); m_x carries no
    carry term (it is added per block) and the paths are None unless want_paths. 'scan' builds
    no [n, n] matrix and holds no state path unless one is asked for.

    `blocks` (the per-step block index, scan only) makes the scan ACCUMULATE each block's sums as
    it passes that block's steps and return (M, Sigma^2) of shape [..., n_blocks] in their place,
    so nothing of shape [batch, sims, n] is materialised at all. That is the shape spec 3 asks for
    and the only one a pricing call walks; `lv_block_stats` is the matmul path's own aggregation
    and the two are gated equal.
    """
    rl, sj, nu = params['Rho_L'], params['Sigma_J'], params['Nu']
    a, beta = params['Cap_A'], params['Cap_Beta']
    # the trailing axis is the grid's, the bucketed levers arriving per STEP and a scalar spreading
    # to the constant-parameter model. The leading axis keeps a step's slice DIMENSIONED: torch
    # demotes a 0-dim float64 against a float32 count, adding the jump mean in single precision.
    ones = torch.ones_like(deltas).unsqueeze(0)
    rs, mu = params['Rho_S'] * ones, params['Mu_J'] * ones
    sj2 = sj * sj
    comp = params['Lambda'] * deltas * (torch.exp(mu + 0.5 * sj2) - 1.0)
    l0, s0 = state0

    if method == 'matmul':
        F_s, w_s, d_s = lv_filter_matmul(params['Kappa_S'], params['Sigma_S'], deltas)
        F_l, w_l, d_l = lv_filter_matmul(params['Kappa_L'], params['Sigma_L'], deltas)
        Nf = counts.to(eta_l.dtype)
        x_s = w_s * eta_s + nu * Nf
        s_path = torch.cat([s0[..., None], s0[..., None] * d_s + x_s @ F_s.T], dim=-1)
        dev_l = (l0 - curve_at_grid[0])[..., None] * d_l + (w_l * eta_l) @ F_l.T
        l_path = torch.cat([l0[..., None], curve_at_grid[1:] + dev_l], dim=-1)
        V = deltas * torch.exp(lv_cap(l_path[..., :-1] + s_path[..., :-1], a, beta))
        sq = sqrt_or_zero(V)
        var = (1.0 - rs * rs - rl * rl) * V + Nf * sj2
        m_x = -0.5 * V + rl * sq * eta_l + rs * sq * eta_s + Nf * mu - comp
        return m_x, var, (s_path if want_paths else None), (l_path if want_paths else None)

    phi_s, w_s = lv_ou_step_weights(params['Kappa_S'], params['Sigma_S'], deltas)
    phi_l, w_l = lv_ou_step_weights(params['Kappa_L'], params['Sigma_L'], deltas)
    ends = set() if blocks is None else set(lv_block_ends(blocks).tolist())
    l, s, acc_m, acc_v = l0, s0, 0.0, 0.0
    ms, vs, ls, ss = [], [], [l0], [s0]
    for k in range(deltas.shape[0]):
        Nk = counts[..., k].to(eta_l.dtype)
        rs_k, mu_k = rs[..., k], mu[..., k]
        V = deltas[k] * torch.exp(lv_cap(l + s, a, beta))
        sq = sqrt_or_zero(V)
        var = (1.0 - rs_k * rs_k - rl * rl) * V + Nk * sj2
        m_x = (-0.5 * V + rl * sq * eta_l[..., k] + rs_k * sq * eta_s[..., k]
               + Nk * mu_k - comp[..., k])
        if blocks is None:
            ms.append(m_x)
            vs.append(var)
        else:
            acc_m, acc_v = acc_m + m_x, acc_v + var
            if k in ends:
                ms.append(acc_m)
                vs.append(acc_v)
                acc_m, acc_v = 0.0, 0.0
        s = phi_s[..., k] * s + w_s[..., k] * eta_s[..., k] + nu * Nk
        l = (curve_at_grid[k + 1] + phi_l[..., k] * (l - curve_at_grid[k])
             + w_l[..., k] * eta_l[..., k])
        if want_paths:
            ss.append(s)
            ls.append(l)
    paths = (torch.stack(ss, -1), torch.stack(ls, -1)) if want_paths else (None, None)
    return torch.stack(ms, -1), torch.stack(vs, -1), paths[0], paths[1]


def lv_block_stats(m_x, var, carry_blocks, block_of_step, s_path=None, l_path=None):
    """Aggregate the per-step mean and variance onto monitored blocks by one matmul.

    Returns (M, Sigma, end_states); end_states is the (l, s) grid state at each block end when
    the state paths are given, else None.
    """
    n = block_of_step.shape[0]
    nb = int(block_of_step.max()) + 1
    A = torch.zeros(nb, n, dtype=m_x.dtype, device=m_x.device)
    A[block_of_step, torch.arange(n, device=m_x.device)] = 1.0
    M = m_x @ A.T + carry_blocks
    Sigma = sqrt_or_zero(var @ A.T)
    end_states = None
    if s_path is not None:
        idx = lv_block_ends(block_of_step) + 1
        end_states = (l_path[..., idx], s_path[..., idx])
    return M, Sigma, end_states


def lv_black(forward, strike, total_sd, is_call=True):
    """Black on a TOTAL standard deviation, not an annualised vol."""
    sd = total_sd.clamp_min(LV_SIGMA_TINY) if torch.is_tensor(total_sd) else total_sd
    d1 = torch.log(forward / strike) / sd + 0.5 * total_sd
    d2 = d1 - total_sd
    if is_call:
        return forward * norm_cdf(d1) - strike * norm_cdf(d2)
    return strike * norm_cdf(-d2) - forward * norm_cdf(-d1)


def lv_vanilla(S, strike, M, Sigma, discount, is_call=True):
    """Conditional Black over the walk's shocks (spec 2.4); averages the last axis."""
    forward = S * torch.exp(M + 0.5 * Sigma * Sigma)
    return discount * lv_black(forward, strike, Sigma, is_call).mean(dim=-1)


def lv_ou_variance(sigma, kappa, t):
    """Variance at t of an OU log-variance factor started from a known value."""
    return sigma * sigma * (1.0 - torch.exp(-2.0 * kappa * t)) / (2.0 * kappa)


def lv_curve_from_forward_variance(xi, params, grid):
    """Curve knots L(t) giving each step the market forward variance xi (spec 2.6).

    The jump variance lam*(mu_J^2 + sigma_J^2) is removed from the target and the fast
    factor's compound-Poisson term of E[exp(l+s)] is carried; the cap is ignored. BUCKET-BLIND:
    ``Mu_J`` is read as ONE value, not the walk's per-step tensor, so a bucketed document maps a
    bucket at a time (spec 2.3.1).
    """
    ks, lam = params['Kappa_S'], params['Lambda']
    xi_diff = xi - lam * (params['Mu_J'] ** 2 + params['Sigma_J'] ** 2)
    var_g = (lv_ou_variance(params['Sigma_S'], ks, grid)
             + lv_ou_variance(params['Sigma_L'], params['Kappa_L'], grid))
    deltas = grid[1:] - grid[:-1]
    lag = torch.tril(grid[:, None] - grid[None, 1:], -1)
    jump = torch.tril(torch.exp(params['Nu'] * torch.exp(-ks * lag)) - 1.0, -1) @ (lam * deltas)
    return torch.log(xi_diff) - 0.5 * var_g - jump


def lv_jump_cumulants(params, T, V):
    """Cumulants of a frozen-variance one-period return: Gaussian V plus compound Poisson.

    V is the total diffusive base variance over T. The return's Gaussian part is all of V, not
    c*V: the leverage shocks carry the other (rho_l^2 + rho_s^2)*V of it. kappa_n = lam*T*E[J^n]
    for n >= 2 with J ~ N(mu_J, sigma_J^2). Returns (k2, k3, k4, skewness, excess kurtosis).
    Bucket-blind alike: ONE ``Mu_J``, so a bucketed document reads one bucket's own cumulants.
    """
    mu, sg = params['Mu_J'], params['Sigma_J']
    lt = params['Lambda'] * T
    k2 = V + lt * (mu ** 2 + sg ** 2)
    k3 = lt * (mu ** 3 + 3.0 * mu * sg ** 2)
    k4 = lt * (mu ** 4 + 6.0 * mu ** 2 * sg ** 2 + 3.0 * sg ** 4)
    return k2, k3, k4, k3 / k2 ** 1.5, k4 / (k2 * k2)


# Correlated sub-stepping -- exact within-interval dynamics between coarse scenario nodes. A coarse
# exposure grid still owes each factor the dynamics it would have had on the calibration clock:
# forwarding the variance deterministically and drawing one aggregate Gaussian is 29%-2000% wrong on
# tail probabilities at |z|=2-3, precisely the quantiles PFE reads. The interval walks its own
# sub-steps instead, and the framework's correlated draw enters as the sqrt(variance)-weighted
# combination of the sub-step normals. Freeze h and the aggregate return collapses back to
# sqrt(sum var)*z_fw, so this is a strict refinement of the mean bridge.

def substep_schedule(f):
    """Trading-time lengths spanning each interval of `f` calibration steps: whole steps, then the
    fractional remainder.  A scenario grid is a CALENDAR object and the recursion is calibrated per
    trading day, so f is essentially never an integer -- rounding it to whole days makes node
    variance a step function of grid spacing, -13% on the default CVA grid.  len == 1 is the exact
    fractional step every fine grid already took; longer is a coarse-grid walk.
    """
    schedule = []
    for x in f:
        whole = int(x)
        rem = float(x) - whole
        steps = (1.0,) * whole + ((rem,) if rem > 1e-9 else ())
        schedule.append(steps or (float(x),))       # dt == 0 (the t=0 anchor) stays one null step
    return schedule


def substep_normals(sqrt_var, z_fw):
    """n iid N(0,1) sub-step draws Z whose weighted combination REPRODUCES the framework draw:
    w'Z = z_fw exactly, w = ``sqrt_var`` normalized along the leading (sub-step) axis.

    Z = e + w*(z_fw - w'e) with e fresh iid normals: Cov(Z) = (I - ww') + ww' = I given
    z_fw ~ N(0,1) independent of e, and w is F_t-measurable so this holds conditionally, per
    interval.  ``sqrt_var`` is (n, ...batch), ``z_fw`` (...batch).

    The weights decide only WHICH linear functional of the walk carries the cross-factor correlation
    -- every marginal is invariant to them.  sqrt of the mean-forwarded variance contribution is the
    interval's own return loading, matching a correlated sibling that shares its variance profile; a
    sibling with flat per-day variance would want uniform weights.  A MODELLING CHOICE, not an
    exactness claim -- see test_weights_match_the_return_loading.
    """
    w = sqrt_var / (sqrt_var ** 2).sum(0, keepdim=True).sqrt()
    e = torch.randn_like(w)
    return e + w * (z_fw - (w * e).sum(0))


def hn_correlated_substeps(h, z_fw, sub_dt, omega, alpha, beta, gamma_star):
    """Walk one coarse scenario interval as the `sub_dt` fractional Heston-Nandi steps that span it.
    Returns (h_end, var_sum, r_sum): terminal variance, realized integrated variance (the caller's
    -1/2 convexity drift), and the innovation -- so the interval return carry - var_sum/2 + r_sum is
    a price-martingale by iterated expectations, exact at every sub-step. Each step is the same
    fractional recursion the fine grid takes, so the two branches agree in the limit.
    """
    psi = hn_persistence(alpha, beta, gamma_star)
    var_bar, mean = [], h
    for dt in sub_dt:                                        # E[h_{j+1}] = h + dt*(omega+alpha+psi*h - h)
        var_bar.append(mean * dt)
        mean = mean + dt * (omega + alpha + psi * mean - mean)
    z = substep_normals(torch.stack(var_bar).sqrt(), z_fw)
    var_sum, r_sum = torch.zeros_like(h), torch.zeros_like(h)
    for j, dt in enumerate(sub_dt):
        sh = h.sqrt()
        var_j = h * dt
        var_sum = var_sum + var_j
        r_sum = r_sum + var_j.sqrt() * z[j]
        h = h + dt * (hn_variance_step(h, sh, z[j], omega, alpha, beta, gamma_star) - h)
    return h, var_sum, r_sum


def hn_component_correlated_substeps(h, q, z_fw, sub_dt, omegas, alpha, beta, gamma1,
                                     rho, phi, gamma2):
    """Walk one coarse scenario interval as the `sub_dt` fractional COMPONENT steps that span it.
    Returns (h_end, q_end, var_sum, r_sum) - the sibling of :func:`hn_correlated_substeps`, and
    every note there holds.

    `omegas` is this interval's own intercept strip, one entry per sub-step. The FORWARDED MEAN that
    sets the correlation weights is the component one: E[q_{j+1}] = omega_j + rho*q_j and
    E[h_{j+1}] = E[q_{j+1}] + beta*(h_j-q_j), both centering terms having conditional mean zero.
    """
    var_bar, mean_h, mean_q = [], h, q
    for dt, omega_t in zip(sub_dt, omegas):
        var_bar.append(mean_h * dt)
        step_q = omega_t + rho * mean_q
        step_h = step_q + beta * (mean_h - mean_q)
        mean_h, mean_q = mean_h + dt * (step_h - mean_h), mean_q + dt * (step_q - mean_q)
    z = substep_normals(torch.stack(var_bar).sqrt(), z_fw)
    var_sum, r_sum = torch.zeros_like(h), torch.zeros_like(h)
    for j, (dt, omega_t) in enumerate(zip(sub_dt, omegas)):
        sh = h.sqrt()
        var_j = h * dt
        var_sum = var_sum + var_j
        r_sum = r_sum + var_j.sqrt() * z[j]
        step_h, step_q = hn_component_variance_step(
            h, q, sh, z[j], omega_t, alpha, beta, gamma1, rho, phi, gamma2)
        h, q = h + dt * (step_h - h), q + dt * (step_q - q)
    return h, q, var_sum, r_sum


def garch_correlated_substeps(h, z_fw, sub_dt, omega, alpha, beta, nu):
    """Walk one coarse scenario interval as the `sub_dt` fractional GARCH(1,1)-t steps that span it.
    Returns (h_end, var_sum, r_sum) with r_j = sqrt(h_j*dt_j)*eps_j: each eps_j is EXACTLY
    standardized Student-t, built by t-scaling the conditioned sub-step normals with fresh Gammas --
    the same scale mixture GARCHSpotModel.generate uses per step, so the correlated draw rides the
    interval's Gaussian kernel.  Same fractional recursion as the fine grid.
    """
    var_bar, mean = [], h
    for dt in sub_dt:                                        # E[h_{j+1}] adds alpha*E[r^2] = alpha*h*dt
        var_bar.append(mean * dt)
        mean = mean + dt * (omega - (1.0 - beta) * mean) + alpha * mean * dt
    z = substep_normals(torch.stack(var_bar).sqrt(), z_fw)
    W = torch.distributions.Gamma(nu / 2.0, 0.5).sample(z.shape).clamp_min(1.0e-6)
    eps = z * torch.sqrt(nu / W) * torch.sqrt((nu - 2.0).clamp_min(1.0e-3) / nu)
    var_sum, r_sum = torch.zeros_like(h), torch.zeros_like(h)
    for j, dt in enumerate(sub_dt):
        var_j = h * dt
        var_sum = var_sum + var_j
        r = var_j.sqrt() * eps[j]
        r_sum = r_sum + r
        h = h + dt * (omega - (1.0 - beta) * h) + alpha * r * r
    return h, var_sum, r_sum


# Black-Scholes reference + HN implied vol (the HN smile/skew diagnostic and the bootstrapper seed).

def bs_call_np(S, K, r, n, total_var):
    """BS call from TOTAL variance (r, n in per-step units) -- a thin adapter over
    ``black_european_option_price`` (F = S*e^{r*n}; vol=sqrt(tv), tenor=1 so stddev^2 = tv)."""
    return float(black_european_option_price(
        S * np.exp(r * n), K, r * n, np.sqrt(total_var), 1.0, 1.0, 1.0))



def bs_implied_total_var(price, S, K, r, n, lo=1e-12, hi=25.0, tol=1e-14, iters=200):
    """Bisection on TOTAL variance (no time units, so this is convention-free)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if bs_call_np(S, K, r, n, mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def hn_implied_vol(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r,
                   steps_per_year=252.0, **kw):
    """Annualised BS implied vol of the HN price (for the smile/skew diagnostics)."""
    c = float(hn_call(S, K, n_steps, h1, omega, alpha, beta, gamma_star, r, **kw))
    tv = bs_implied_total_var(c, float(S), float(K), float(r), int(n_steps))
    return np.sqrt(tv / (int(n_steps) / steps_per_year))


def Bjerksund_Stensland(A1, A2, B, x1, x2, K, sigma1, sigma2, rho, callOrPut):
    a = x2 + K
    b = x2 / a
    sigma1_2 = sigma1 * sigma1
    sigma2_2 = sigma2 * sigma2
    # make sure the variance is at least 1e-6
    v2 = torch.clamp(sigma1_2 - 2 * rho * sigma1 * b * sigma2 + b * b * sigma2_2, min=1e-6)
    v = torch.sqrt(v2)
    d = torch.log(x1 / a) / v
    d1 = d + v / 2
    d2 = d - (sigma1_2 - 2 * rho * sigma1 * sigma2 - b * b * sigma2_2 + 2 * b * sigma2_2) / (2 * v)
    d3 = d - (sigma1_2 - b * b * sigma2_2) / (2 * v)

    return A1 * x1 * norm_cdf(callOrPut * d1) + A2 * x2 * norm_cdf(callOrPut * d2) + B * norm_cdf(callOrPut * d3)


def bachelier_european_option(F, X, vol, tenor, buyorsell, callorput, shared, cash_payoff=0.0, shift=0.0):
    # calculates the bachelier function WITHOUT discounting
    # shift is not used but needed to have the same sig as black_european_option

    if isinstance(tenor, float):
        guard = (vol > 0.0) & (tenor > 0.0)
        stddev = vol.clamp(min=1e-5) * np.sqrt(max(tenor, 0.0))
    else:
        tenor_np = tenor.clip(min=0.0)
        tau_key = ('tenor', tenor_np.shape, tenor_np.tobytes())
        if tau_key not in shared.t_Buffer:
            shared.t_Buffer[tau_key] = vol.new(np.sqrt(tenor_np))

        tau = shared.t_Buffer[tau_key]
        guard = tau > 0.0

        if len(guard.shape) > 1:
            guard = torch.unsqueeze(guard, dim=2)
            sigma = vol * torch.unsqueeze(tau, dim=2)
        else:
            guard = torch.unsqueeze(guard, dim=1)
            sigma = vol * tau.reshape(-1, 1)

        stddev = sigma.clamp(min=1e-5)

    # Degenerate case: zero stddev (should be guarded by clamp, but keep intrinsic as fallback)
    intrinsic = torch.relu(callorput * (F - X))

    if cash_payoff:
        # Digital option under Bachelier:
        # price = cash * N(callorput * d), d = (F - K) / stddev
        d = (F - X) / stddev
        prem = cash_payoff * norm_cdf(callorput * d)
        value = cash_payoff * (callorput * (F - X) > 0) * shared.one
    else:
        mu = callorput * (F - X)  # positive when option is in-the-money
        mu_per_sig = mu / stddev
        prem = mu * norm_cdf(mu_per_sig) + stddev * norm_pdf(mu_per_sig)
        value = intrinsic

    return buyorsell * torch.where(guard, prem, value)

def black_european_option(F, X, vol, tenor, buyorsell, callorput, shared, cash_payoff=0.0, shift=0.0):
    # calculates the black function WITHOUT discounting

    if isinstance(tenor, float):
        guard = (vol > 0.0) & (X > 0.0)
        stddev = vol.clamp(min=1e-5) * np.sqrt(tenor)
        strike = max(X, 1e-5) if isinstance(X, float) else X.clamp(min=1e-5)
    else:
        tenor_np = tenor.clip(min=0.0)
        tau_key = ('tenor', tenor_np.shape, tenor_np.tobytes())
        if tau_key not in shared.t_Buffer:
            shared.t_Buffer[tau_key] = vol.new(np.sqrt(tenor_np))

        tau = shared.t_Buffer[tau_key]
        guard = tau > 0.0

        if len(guard.shape) > 1:
            guard = torch.unsqueeze(guard, dim=2)
            sigma = vol * torch.unsqueeze(tau, dim=2)
        else:
            guard = torch.unsqueeze(guard, dim=1)
            sigma = vol * tau.reshape(-1, 1)

        stddev = sigma.clamp(min=1e-5)
        strike = X

    # make sure the forward is always >1e-5
    forward = torch.clamp(F, min=1e-5)

    if isinstance(strike, float) and strike == 0 and not shift:
        # need to check if this is a put option (value is 0)
        # or a call option (value is just the forward)
        adjustment = 1.0 if callorput == 1.0 else 0.0
        prem = forward * adjustment
        value = forward * adjustment
    else:
        # handle shifted vol surfaces
        if shift:
            forward = torch.clamp(F, min=1e-5-shift) + shift
            strike = strike + shift
        d1 = torch.log(forward / strike) / stddev + 0.5 * stddev
        d2 = d1 - stddev
        if cash_payoff:
            prem = cash_payoff * norm_cdf(callorput * d2)
            value = cash_payoff * (callorput * (forward - strike) > 0) * shared.one
        else:
            prem = callorput * (forward * norm_cdf(callorput * d1) - strike * norm_cdf(callorput * d2))
            value = torch.relu(callorput * (forward - strike))
    return buyorsell * torch.where(guard, prem, value)


# tenor manipulation
def get_tenors(factor_dict):
    all_tenor = {}
    for factor_name, data in factor_dict.items():
        factor = data.factor if hasattr(data, 'factor') else data
        if hasattr(factor, 'get_tenor_indices'):
            indices = factor.get_tenor_indices()
            if isinstance(indices, dict):
                for k, v in indices.items():
                    new_factor_name = Factor(factor_name.type, factor_name.name + (k,))
                    all_tenor.setdefault(check_scope_name(new_factor_name), v)
            else:
                all_tenor.setdefault(check_scope_name(factor_name), indices)
    return all_tenor


def tenor_diff(tenor_points, interp='Linear'):
    return CurveTenor(tenor_points, interp)


def update_tenors(base_date, all_factors):
    def daycount_fn(base_date, daycount):
        def calc_daycount(time_in_days):
            return get_day_count_accrual(base_date, time_in_days, daycount)

        return calc_daycount

    all_tenors = {}
    for factor, factor_obj in all_factors.items():
        risk_factor = factor_obj.factor if hasattr(factor_obj, 'factor') else factor_obj

        if factor.type in OneDimensionalFactors or (
                factor.type in TwoDimensionalFactors and risk_factor.get_subtype()[0] in ['SVI', 'Skew']):
            tenor_points = risk_factor.get_tenor()

            if factor.type == 'DividendRate':
                tenor_data = tenor_diff(tenor_points, 'Dividend')
            elif factor.type in ['InterestRate', 'InflationRate', 'ForwardRate']:
                if len(risk_factor.interpolation)>1:
                    interpolation_type = tuple([(x[0], x[1], x[2][0]) for x in risk_factor.interpolation])
                else:
                    interpolation_type = risk_factor.interpolation[0][0]
                tenor_data = tenor_diff(tenor_points, interpolation_type)
            else:
                tenor_data = tenor_diff(tenor_points)

            daycount = risk_factor.get_day_count()
            all_tenors[factor] = [tenor_data, daycount_fn(base_date, daycount)]

        # this is a surface of some kind
        elif factor.type in TwoDimensionalFactors:
            # we're going to dynamically interpolate when needed
            expiry_map = []
            for moneyness_points in risk_factor.index_map.values():
                expiry_map.append(tenor_diff(moneyness_points))
            # store the moneyness and expiry first
            all_tenors[factor] = [tenor_diff(risk_factor.get_moneyness()),
                                  tenor_diff(risk_factor.get_expiry()), expiry_map]

        elif factor.type in ThreeDimensionalFactors:
            if factor.type == 'ForwardPriceVol':
                # can interpolate dynamically when needed
                expiry_map = []
                for expiry_points in risk_factor.index_map[risk_factor.EXPIRY_INDEX]:
                    expiry_map.append(tenor_diff(expiry_points[0]))
                moneyness_map = []
                for moneyness_points in risk_factor.index_map[risk_factor.MONEYNESS_INDEX]:
                    moneyness_map.append(tenor_diff(moneyness_points[0]))
                # store the moneyness, expiry and tenor points
                all_tenors[factor] = [moneyness_map, expiry_map,
                                      tenor_diff(risk_factor.get_tenor()), risk_factor.index_map]
            else:
                # full surface defined - do not interpolate dynamically
                for dim_index, data in enumerate(
                        [risk_factor.get_moneyness(), risk_factor.get_expiry(), risk_factor.get_tenor()]):
                    all_tenors.setdefault(factor, [0, 0, 0])[dim_index] = tenor_diff(data)

    return all_tenors


# indexing ops manipulating large tensors
def interpolate_tensor(t, tenor, rate_tensor):
    dvt = np.concatenate(([1], np.diff(tenor), [1]))
    tenor_index = tenor.searchsorted(t, side='right')
    index = (tenor_index - 1).clip(0, tenor.size - 1)
    index_next = tenor_index.clip(0, tenor.size - 1)
    alpha = rate_tensor.new(((t - tenor[index]) / dvt[tenor_index]).clip(0, 1))
    return rate_tensor[index] * (1 - alpha) + rate_tensor[index_next] * alpha



def gather_interp_matrix(mtm, deal_time_dep):
    if deal_time_dep.alpha.any():
        if deal_time_dep.t_alpha is None:
            deal_time_dep.t_alpha = mtm.new(deal_time_dep.alpha)
        return mtm[deal_time_dep.index] * (1 - deal_time_dep.t_alpha) + \
            mtm[deal_time_dep.index_next] * deal_time_dep.t_alpha
    else:
        return mtm[deal_time_dep.index]


def gather_scenario_interp(interp_obj, time_grid, shared, as_curve_tensor=True):
    # calc the time interpolation weights
    index = time_grid[:, TIME_GRID_ScenarioPriorIndex].astype(np.int64)
    alpha_shape = tuple([-1] + [1] * (len(interp_obj.shape) - 1))
    alpha = time_grid[:, TIME_GRID_PriorScenarioDelta].reshape(alpha_shape)
    curve_tensor = CurveTensor(interp_obj, index, alpha if alpha.any() else None)
    return curve_tensor if as_curve_tensor else curve_tensor.interp_value()


def split_counts(rates, counts, shared):
    splits = []
    for rate in rates:
        if isinstance(rate, torch.Tensor):
            splits.append(split_tensor(rate, counts))
        else:
            splits.append(rate.split_counts(counts, shared))

    return zip(*splits)


def calc_fx_cross(rate1, rate2, time_grid, shared):
    key_code = ('fxcross', rate1[0], rate2[0], time_grid[:, TIME_GRID_MTM].tobytes())
    if rate1 != rate2:
        if key_code not in shared.t_Buffer:
            shared.t_Buffer[key_code] = calc_time_grid_spot_rate(
                rate1, time_grid, shared) / calc_time_grid_spot_rate(
                rate2, time_grid, shared)
    else:
        shared.t_Buffer[key_code] = shared.one
    return shared.t_Buffer[key_code]


def calc_discount_rate(block, tenors_in_days, shared, multiply_by_time=True):
    key_code = ('discount', tuple([x[:2] for x in block.code]),
                tuple(block.time_grid[:, TIME_GRID_MTM]),
                tenors_in_days.shape, tuple(tenors_in_days.ravel()))

    if key_code not in shared.t_Buffer:
        discount_rates = torch.exp(-block.gather_weighted_curve(
            shared, tenors_in_days, multiply_by_time=multiply_by_time))
        shared.t_Buffer[key_code] = discount_rates

    return shared.t_Buffer[key_code]


def calc_spot_forward(curve, T, time_grid, shared, only_diag):
    """
    Function for calculating the forward price of FX or EQ rates taking
    into account risk neutrality for static curves
    """
    curve_grid = calc_time_grid_curve_rate(curve, time_grid, shared)
    T_t = T - time_grid[:, TIME_GRID_MTM].reshape(-1, 1)
    weights = np.diag(T_t).reshape(-1, 1) if only_diag else T_t
    return curve_grid.gather_weighted_curve(shared, weights)


def calc_dividend_samples(start_day, samples, time_grid):
    reset_start_day = start_day.clip(min=0)
    time_grid_scenario = [time_grid.get_scenario_offset(x) for x in reset_start_day]
    scenario = [x[1] for x in time_grid_scenario]
    time_interp = [x[0] for x in time_grid_scenario]
    resets = [TensorResets([[Time_Grid, reset_start, -1, reset_start, reset_end, 0.0, 0.0, 0.0]
                            for reset_end in samples], [scenario_offset] * len(samples))
              for Time_Grid, reset_start, scenario_offset in zip(time_interp, reset_start_day, scenario)]
    return resets


def calc_realized_dividends(s_t0, repo, div_yield, div_reset_stack, shared):
    # Calculate exp(sr) * (1 - exp(-sq))
    sr_minus_sq = torch.stack([
        torch.exp(torch.squeeze(calc_spot_forward(
            repo, div_resets[:, RESET_INDEX_End_Day], div_resets, shared, True), dim=1)
        ) * (1.0 - torch.exp(
            -torch.squeeze(calc_spot_forward(
                div_yield, div_resets[:, RESET_INDEX_End_Day], div_resets, shared, True), dim=1))
             )
        for div_resets in div_reset_stack], dim=1)

    return s_t0 * sr_minus_sq


def calc_eq_drift(repo, div_yield, weights, time_grid, shared, multiply_by_time=True):
    repo_curve_grid = calc_time_grid_curve_rate(repo, time_grid, shared)
    div_curve_grid = calc_time_grid_curve_rate(div_yield, time_grid, shared)
    return repo_curve_grid.gather_weighted_curve(
        shared, weights, multiply_by_time=multiply_by_time) - div_curve_grid.gather_weighted_curve(
        shared, weights, multiply_by_time=multiply_by_time)


def calc_eq_forward(equity, repo, div_yield, T, time_grid, shared, only_diag=False):
    T_scalar = isinstance(T, int)
    key_code = ('eqforward', equity[0], div_yield[0][:2], only_diag,
                T if T_scalar else tuple(T),
                time_grid[:, TIME_GRID_MTM].tobytes())

    if key_code not in shared.t_Buffer:
        T_t = T - time_grid[:, TIME_GRID_MTM].reshape(-1, 1)
        spot = calc_time_grid_spot_rate(equity, time_grid, shared)

        if T_t.any():
            drift = torch.exp(
                calc_spot_forward(repo, T, time_grid, shared, only_diag) -
                calc_spot_forward(div_yield, T, time_grid, shared, only_diag))
        else:
            drift = shared.one.new_ones(
                [time_grid.shape[0], 1 if only_diag else T_t.size, 1])

        shared.t_Buffer[key_code] = spot * torch.squeeze(drift, dim=1) \
            if T_scalar else torch.unsqueeze(spot, dim=1) * drift

    return shared.t_Buffer[key_code]


def calc_fx_drift(local, other, weights, time_grid, shared, multiply_by_time=True):
    repo_local = calc_time_grid_curve_rate(local[1], time_grid, shared)
    repo_other = calc_time_grid_curve_rate(other[1], time_grid, shared)
    return repo_other.gather_weighted_curve(
        shared, weights, multiply_by_time=multiply_by_time) - repo_local.gather_weighted_curve(
        shared, weights, multiply_by_time=multiply_by_time)


def calc_fx_forward(local, other, T, time_grid, shared, only_diag=False):
    T_scalar = isinstance(T, int)
    key_code = ('fxforward', local[0][0], other[0][0], only_diag,
                T if T_scalar else tuple(T),
                time_grid[:, TIME_GRID_MTM].tobytes())
    if key_code not in shared.t_Buffer:
        if local[0] != other[0]:
            T_t = T - time_grid[:, TIME_GRID_MTM].reshape(-1, 1)
            fx_spot = calc_fx_cross(local[0], other[0], time_grid, shared)

            if T_t.any():
                weights = np.diag(T_t).reshape(-1, 1) if only_diag else T_t
                drift = torch.exp(calc_fx_drift(local, other, weights, time_grid, shared))
            else:
                drift = fx_spot.new_ones([time_grid.shape[0], 1 if only_diag else T_t.size, 1])

            shared.t_Buffer[key_code] = fx_spot * torch.squeeze(drift, dim=1) \
                if T_scalar else torch.unsqueeze(fx_spot, dim=1) * drift
        else:
            shared.t_Buffer[key_code] = shared.one

    return shared.t_Buffer[key_code]


def gather_flat_surface(flat_surface, code, expiry, shared, calc_std):
    # cache the time surface interpolation matrix
    time_code = ('surface_flat', code[:2], tuple(expiry), calc_std)

    if time_code not in shared.t_Buffer:
        expiry_tenor = code[FACTOR_INDEX_Expiry_Index]
        moneyness_max_index = np.array([x.tenor.shape[0] for x in code[FACTOR_INDEX_Flat_Index]])
        exp_index = np.cumsum(np.append(0, moneyness_max_index[:-1]))
        time_modifier = np.sqrt(expiry).reshape(-1, 1) if calc_std else 1.0
        index, index_next, alpha = expiry_tenor.get_index(expiry)
        alpha = flat_surface.new(alpha.reshape(-1, 1, 1))
        subset = np.union1d(index, index_next)

        block_indices, block_alphas = [], []
        new_moneyness_tenor = reduce(np.union1d, [code[FACTOR_INDEX_Flat_Index][x].tenor for x in subset])

        for tenor_index in subset:
            moneyness_tenor = code[FACTOR_INDEX_Flat_Index][tenor_index]
            moneyness_index, moneyness_index_next, moneyness_alpha = moneyness_tenor.get_index(
                new_moneyness_tenor)

            block_indices.append(exp_index[tenor_index] + np.stack([moneyness_index, moneyness_index_next]))
            block_alphas.append(np.stack([1.0 - moneyness_alpha, moneyness_alpha]))

        # need to interpolate back to the tenor level
        money_indices, money_alpha = np.array(block_indices), np.array(block_alphas)
        subset_index = subset.searchsorted(index)
        tenor_money_indices = flat_surface.new_tensor(money_indices[subset_index], dtype=torch.int64)
        tenor_money_alpha = flat_surface.new(money_alpha[subset_index])
        subset_index_next = subset.searchsorted(index_next)
        tenor_money_alpha_next = flat_surface.new(money_alpha[subset_index_next])
        tenor_money_indices_next = flat_surface.new_tensor(money_indices[subset_index_next], dtype=torch.int64)

        if code[FACTOR_INDEX_SubType][0] == 'Malz':
            # interpolate along variance for term
            term_prior = flat_surface.new(expiry_tenor.tenor[index].reshape(-1, 1, 1))
            term_post = flat_surface.new(expiry_tenor.tenor[index_next].reshape(-1, 1, 1))
            t_expiry = flat_surface.new(expiry.clip(min=expiry_tenor.min).reshape(-1, 1))
            var_prior = term_prior * flat_surface.take(tenor_money_indices)**2
            var_post = term_post * flat_surface.take(tenor_money_indices_next)**2
            var_surface = time_modifier * torch.sum(
                var_prior * tenor_money_alpha * (1.0 - alpha) +
                var_post * tenor_money_alpha_next * alpha, dim=1)
            surface = torch.sqrt(var_surface/t_expiry)
        else:
            # interpolate along volatility
            surface = time_modifier * torch.sum(
                flat_surface.take(tenor_money_indices) * tenor_money_alpha * (1.0 - alpha) +
                flat_surface.take(tenor_money_indices_next) * tenor_money_alpha_next * alpha, dim=1)

        shared.t_Buffer[time_code] = (surface.reshape(-1), code, tenor_diff(new_moneyness_tenor))

    return shared.t_Buffer[time_code]


def gather_surface_interp(surface, code, expiry, shared, calc_std):
    # cache the time surface interpolation matrix
    time_code = ('surface_interp', code[:2], tuple(expiry), calc_std)

    if time_code not in shared.t_Buffer:
        expiry_tenor = code[FACTOR_INDEX_Expiry_Index]
        index, index_next, alpha = expiry_tenor.get_index(expiry)
        time_modifier = np.sqrt(expiry) if calc_std else 1.0
        alpha = surface.new(alpha).reshape(-1, 1)

        shared.t_Buffer[time_code] = (surface[index] * (1 - alpha) + surface[index_next] * alpha) * time_modifier

    return shared.t_Buffer[time_code]


def calc_moneyness_vol_rate(moneyness, expiry, key_code, shared):
    def calc_skew(x, t, atm_vol, s, L, R, C, D, lam, rho):
        skew_key = ('skew_params', t) + key_code[FACTOR_INDEX_Offset][0]

        if skew_key not in shared.t_Buffer:
            s2LC = s + 2.0 * L * C
            gamma = s2LC / (-2.0 * C * lam)
            beta = s2LC * (1.0 + 1.0 / lam)
            alpha = atm_vol + C * ((s - beta) + C * (L - gamma))

            # Right wing
            s2RD = s + 2.0 * R * D
            gamma_r = s2RD / (-2.0 * D * rho)
            beta_r = s2RD * (1.0 + 1.0 / rho)
            alpha_r = atm_vol + D * ((s - beta_r) + D * (R - gamma_r))

            shared.t_Buffer[skew_key] = (gamma, beta, alpha, gamma_r, beta_r, alpha_r)

        gamma, beta, alpha, gamma_r, beta_r, alpha_r = shared.t_Buffer[skew_key]
        lam_ok = lam.all()
        rho_ok = rho.all()

        # the 6 regions of the skew - check for 0 lam and rho - hold flat
        r1 = torch.ones_like(x) * (
            (alpha + C * (beta * (1.0 + lam) + gamma * (1.0 + lam) ** 2 * C)) if lam_ok else (atm_vol + C * (s + L * C)))
        r2 = alpha + x * (beta + gamma * x) if lam_ok else atm_vol + C * (s + L * C)
        r3 = atm_vol + x * (s + L * x)
        r4 = atm_vol + x * (s + R * x)
        r5 = alpha_r + x * (beta_r + gamma_r * x) if rho_ok else atm_vol + D * (s + R * D)
        r6 = torch.ones_like(x) * (
            (alpha_r + D * (beta_r * (1.0 + rho) + gamma_r * (1.0 + rho) ** 2 * D)) if rho_ok else (atm_vol + D * (s + R * D)))

        return torch.where(
            x <= (1 + lam) * C, r1,
                torch.where(x <= C, r2,
                            torch.where(x<=0, r3,
                                        torch.where(x<=D, r4,
                                                    torch.where(x<(1+rho)*D, r5, r6)
                                                    )
                                        )
                            )
                )

    if key_code[0] == 'vol_time_grid' and key_code[FACTOR_INDEX_Offset][0][0] in ['SVI', 'Skew']:
        surface, rate_code, calc_std = shared.t_Buffer[key_code]
        expiry_tenor = rate_code[FACTOR_INDEX_Tenor_Index]
        time_modifier = np.sqrt(expiry).reshape(-1, 1) if calc_std else 1.0
        index, index_next, alpha = expiry_tenor.get_index(expiry)
        alpha = shared.one.new(alpha.reshape(-1, 1))

        # need to calculate the correct way to query the vol surface
        if moneyness is None:
            moneyness = 0.0 * shared.one
        else:
            if rate_code[FACTOR_INDEX_SubType][1] == 'Sticky_Strike':
                atm_ref = surface['ATM_Ref'][index] * (1 - alpha) + surface['ATM_Ref'][index_next] * alpha
                moneyness = torch.log(moneyness / atm_ref)

        if rate_code[FACTOR_INDEX_SubType][0] == 'Skew':
            vol_prior = calc_skew(moneyness, tuple(index), surface['ATM_Vol'][index], surface['s'][index],
                                  surface['L'][index], surface['R'][index], surface['C'][index],
                                  surface['D'][index], surface['lam'][index], surface['rho'][index])
            vol_post = calc_skew(moneyness, tuple(index_next), surface['ATM_Vol'][index_next], surface['s'][index_next],
                                  surface['L'][index_next], surface['R'][index_next], surface['C'][index_next],
                                  surface['D'][index_next], surface['lam'][index_next], surface['rho'][index_next])
            vol = vol_prior * (1 - alpha) + vol_post * alpha
            return vol * time_modifier

        elif rate_code[FACTOR_INDEX_SubType][0] == 'SVI':
            k_m_prior = moneyness - surface['m'][index]
            var_prior = surface['a'][index] + surface['b'][index] * (
                    surface['rho'][index] * k_m_prior + torch.sqrt(k_m_prior ** 2 + surface['sigma'][index] ** 2))
            k_m_post = moneyness - surface['m'][index_next]
            var_post = surface['a'][index_next] + surface['b'][index_next] * (
                    surface['rho'][index_next] * k_m_post + torch.sqrt(
                k_m_post ** 2 + surface['sigma'][index_next] ** 2))
            variance = var_prior * (1 - alpha) + var_post * alpha
            return torch.sqrt(variance) * time_modifier
    else:
        surface, rate_code, moneyness_tenor = shared.t_Buffer[key_code]
        max_index = np.prod(surface.shape) - 1
        if moneyness is None:
            moneyness = shared.one * (0.0 if rate_code[FACTOR_INDEX_SubType][0]=='Malz' else 0.0)
        index, _, alpha = moneyness_tenor.get_index(moneyness)
        expiry_indices = np.arange(expiry.size).astype(np.int32)
        expiry_index_key = ('expiry_tenor', tuple(expiry_indices), moneyness_tenor.tenor.size)

        if expiry_index_key not in shared.t_Buffer:
            shared.t_Buffer[expiry_index_key] = shared.one.new_tensor(
                np.array([expiry_indices * moneyness_tenor.tenor.size]),
                dtype=torch.int32).T

        expiry_offsets = shared.t_Buffer[expiry_index_key]
        vol_index = index + expiry_offsets

        vol_index_next = torch.clamp(vol_index + 1, 0, max_index)
        vols = surface[vol_index] * (1.0 - alpha) + surface[vol_index_next] * alpha
        return vols


def calc_time_grid_vol_rate(code, moneyness, expiry, shared, calc_std=False):
    keys = []
    for rate in code:
        if rate[FACTOR_INDEX_SubType][0] in ['SVI', 'Skew']:
            keys.append((rate[FACTOR_INDEX_SubType][0], tuple(rate[:1] + tuple(rate[1]))))
        else:
            keys.append(('vol2d', rate[:2]))

    key_code = ('vol_time_grid', tuple(keys), tuple(expiry), calc_std)

    if key_code not in shared.t_Buffer:
        spread = None
        # We only support one vol stack at the moment - but can extend this to 2 or more
        for rate in code:
            # Only static moneyness/expiry vol surfaces are supported for now
            if rate[FACTOR_INDEX_Stoch]:
                raise Exception("Stochastic vol surfaces not yet implemented")
            else:
                if rate[FACTOR_INDEX_SubType][0] in ['SVI', 'Skew']:
                    spread = {x.name[-1]: shared.t_Static_Buffer[x].reshape(-1, 1) for x in rate[FACTOR_INDEX_Offset]}
                else:
                    spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                break

        # either interpolate a flat vol surface or a svi/skew vol param
        if code[0][FACTOR_INDEX_SubType][0] in ['SVI', 'Skew']:
            shared.t_Buffer[key_code] = (spread, code[0], calc_std)
        else:
            shared.t_Buffer[key_code] = gather_flat_surface(
                spread, code[0], expiry, shared, calc_std)

    return calc_moneyness_vol_rate(moneyness, expiry, key_code, shared)


def calc_tenor_time_grid_vol_rate(code, moneyness, expiry, tenor, shared, calc_std=False):
    key_code = ('vol3d', tuple([x[:2] for x in code]),
                tuple(expiry.flatten()), tenor, calc_std)

    if key_code not in shared.t_Buffer:
        vol_spread = None

        for rate in code:
            # Only static moneyness/expiry vol surfaces are supported for now
            if rate[FACTOR_INDEX_Stoch]:
                raise Exception("Stochastic vol surfaces not yet implemented")
            else:
                vol_spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                break

        tenor_index = code[0][FACTOR_INDEX_VolTenor_Index]
        space = vol_spread.reshape(tenor_index.tenor.size, -1)
        index, index_next, alpha = tenor_index.get_index(tenor)

        spread = (1.0 - alpha) * space[index] + alpha * space[index_next]

        surface = spread.reshape(-1, code[0][FACTOR_INDEX_Moneyness_Index].tenor.size)
        flat_vol_time = gather_surface_interp(surface, code[0], expiry, shared, calc_std).reshape(-1, )

        shared.t_Buffer[key_code] = (flat_vol_time, code[0], code[0][FACTOR_INDEX_Moneyness_Index])

    return calc_moneyness_vol_rate(moneyness, expiry, key_code, shared)


def calc_tenor_cap_time_grid_vol_rate(code, moneyness, expiry, tenor, shared, calc_std=False):
    key_code = ('vol3d_cap', tuple([x[:2] for x in code]), tenor, calc_std, tuple(expiry.flatten()))

    if key_code not in shared.t_Buffer:
        vol_spread = None

        for rate in code:
            # Only static moneyness/expiry vol surfaces are supported for now
            if rate[FACTOR_INDEX_Stoch]:
                raise Exception("Stochastic vol surfaces not yet implemented")
            else:
                vol_spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                break

        tenor_index = code[0][FACTOR_INDEX_VolTenor_Index]
        space = vol_spread.reshape(tenor_index.tenor.size, -1)
        index, index_next, alpha = tenor_index.get_index(tenor)

        spread = space[index] * (1.0 - alpha) + space[index_next] * alpha
        shared.t_Buffer[key_code] = spread.reshape(-1, code[0][FACTOR_INDEX_Moneyness_Index].tenor.size)

    surface = shared.t_Buffer[key_code]
    result = []
    for exp, mon in zip(expiry, moneyness):
        time_exp = key_code[:-1] + tuple(exp)
        if time_exp not in shared.t_Buffer:
            flat_vol_time = gather_surface_interp(
                surface, code[0], exp, shared, calc_std).reshape(-1)
            shared.t_Buffer[time_exp] = (flat_vol_time, code[0], code[0][FACTOR_INDEX_Moneyness_Index])
        result.append(calc_moneyness_vol_rate(mon, exp, time_exp, shared))

    return torch.stack(result)


def calc_delivery_time_grid_vol_rate(code, moneyness, expiry, delivery, time_grid, shared):
    # can't cache this function as moneyness is generally stochastic
    vol_spread = None

    for rate in code:
        # Only static moneyness/expiry vol surfaces are supported for now
        if rate[FACTOR_INDEX_Stoch]:
            raise Exception("Stochastic vol surfaces not yet implemented")
        else:
            vol_spread = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
            break

    index_map = code[0][FACTOR_INDEX_Surface_Flat_Index]
    tenor_index = code[0][FACTOR_INDEX_VolTenor_Index]
    expiry_index = code[0][FACTOR_INDEX_Expiry_Index]
    money_index = code[0][FACTOR_INDEX_Moneyness_Index]

    # need to know the moneyness offset for a particular expiry offset
    expiry_offset = np.cumsum([0] + [x.tenor.size for x in expiry_index])
    t_index, t_index_next, alpha = tenor_index.get_index(delivery)
    alpha_tensor = vol_spread.new(alpha).unsqueeze(2)

    space = []
    tenor_cache = {}
    for current_tenor_index in [t_index, t_index_next]:
        result = []
        for tenor_sub_index, exp, mon in zip(current_tenor_index, expiry, moneyness):
            expiry_tenor_map = [expiry_index[to].get_index(e) for to, e in zip(tenor_sub_index, exp)]
            time_slice = []
            for tenor_offset, (e_index, e_index_next, e_alpha) in zip(tenor_sub_index, expiry_tenor_map):
                tenor_exp_key = (tenor_offset, e_index, e_index_next)
                if tenor_exp_key not in tenor_cache:
                    if expiry_index[tenor_offset].tenor.size > 1:
                        # need to interpolate the expiry
                        moneyness_00 = expiry_offset[tenor_offset] + e_index
                        moneyness_01 = expiry_offset[tenor_offset] + e_index_next

                        m_prior = vol_spread[slice(*index_map[2][moneyness_00][1:])]
                        m_next = vol_spread[slice(*index_map[2][moneyness_01][1:])]

                        # grab 2 moneyness layers
                        m_index_1, m_index_next_1, m_alpha_1 = money_index[moneyness_00].get_index(mon)
                        m_index_2, m_index_next_2, m_alpha_2 = money_index[moneyness_01].get_index(mon)

                        exp_prior = m_prior[m_index_1] * (1 - m_alpha_1) + m_prior[m_index_next_1] * m_alpha_1
                        exp_next = m_next[m_index_2] * (1 - m_alpha_2) + m_next[m_index_next_2] * m_alpha_2
                        tenor_cache[tenor_exp_key] = exp_prior * (1 - e_alpha) + exp_next * e_alpha

                    else:
                        # go straight to moneyness
                        moneyness_0 = expiry_offset[tenor_offset]
                        m_slice = vol_spread[slice(*index_map[2][moneyness_0][1:])]
                        m_index, m_index_next, m_alpha = money_index[moneyness_0].get_index(mon)
                        tenor_cache[tenor_exp_key] = m_slice[m_index] * (1 - m_alpha) + m_slice[m_index_next] * m_alpha

                time_slice.append(tenor_cache[tenor_exp_key])
            result.append(torch.stack(time_slice))
        space.append(result)

    interpolated_vols = [prior * (1 - a) + next * a for prior, next, a in zip(space[0], space[1], alpha_tensor)]

    return torch.stack(interpolated_vols)


def hermite_interpolation_tensor(t, rate_tensor):
    rate_diff = (rate_tensor[:, 1:, :] - rate_tensor[:, :-1, :])
    time_diff = t[:, 1:, :] - t[:, :-1, :]

    # calc r_i
    r_i = ((rate_diff[:, :-1, :] * time_diff[:, 1:, :]) / time_diff[:, :-1, :] +
           (rate_diff[:, 1:, :] * time_diff[:, :-1, :]) / time_diff[:, 1:, :]) / (
                  t[:, 2:, :] - t[:, :-2, :])
    r_1 = ((rate_diff[:, 0] * (t[:, 2, :] + t[:, 1, :] - 2.0 * t[:, 0, :])) / time_diff[:, 0, :] -
           (rate_diff[:, 1] * time_diff[:, 0, :]) / time_diff[:, 1, :]) / (t[:, 2, :] - t[:, 0, :])

    r_n = (-1.0 / (t[:, -1, :] - t[:, -3, :])) * (
            (rate_diff[:, -2] * time_diff[:, -1, :]) / time_diff[:, -2, :] -
            (rate_diff[:, -1] * (2.0 * t[:, -1, :] - t[:, -2, :] - t[:, -3, :])) / time_diff[:, -1, :])

    ri = torch.cat([torch.unsqueeze(r_1, dim=1), r_i, torch.unsqueeze(r_n, dim=1)], dim=1)

    # zero
    zero = torch.unsqueeze(torch.zeros_like(r_1), dim=1)
    # calc g_i
    gi = torch.cat([time_diff * ri[:, :-1, :] - rate_diff, zero], dim=1)
    # calc c_i
    ci = torch.cat([2.0 * rate_diff - time_diff * (ri[:, :-1, :] + ri[:, 1:, :]), zero], dim=1)

    return gi, ci


def make_curve_tensor(tensor, curve_component, time_grid, shared, n_batch_dims=1):
    """Build (and cache) the interpolation for a curve, then gather it onto `time_grid`. A None
    `time_grid` skips the gather and hands back the bare CurveTensor.

    `n_batch_dims` > 1 means the curve carries multiple trailing batch axes - a nested inner-MC
    curve shaped (scen, n_tenors, B, B2). They are collapsed into ONE batch axis up front so the
    rest of the curve stack stays rank-agnostic; the caller reshapes the gathered result's trailing
    axis back. The default of 1 preserves the single-batch path exactly.

    Those (B, B2) gathers all happen inside a process's `generate`, BEFORE a fork publishes its
    block sequence, so a multi-block source never reaches here and would say so.

    Interpolation is built by ONE recursive factory: a bare tensor becomes a leaf (or a
    tenor-segmented composite of leaves), a fork's `ScenarioSource` a scenario-routed composite
    whose per-block children are built by the same call.
    """
    if n_batch_dims > 1:
        tensor = tensor.reshape(*tensor.shape[:-n_batch_dims], -1)
    curve_tenor = curve_component[FACTOR_INDEX_Tenor_Index]
    key_code = (curve_tenor.type, curve_component[:2], tuple(tensor.shape))

    if key_code not in shared.t_Buffer:
        shared.t_Buffer[key_code] = build_interpolation(tensor, curve_tenor)

    if time_grid is not None:
        return gather_scenario_interp(shared.t_Buffer[key_code], time_grid, shared)
    else:
        return CurveTensor(shared.t_Buffer[key_code], np.zeros(1, dtype=np.int64), None)


def calc_time_grid_curve_rate(code, time_grid, shared, n_batch_dims=1):
    """Gather every curve factor named by `code` onto `time_grid`, as one cached TensorBlock.

    `n_batch_dims` > 1 gathers a curve whose simulated state carries extra trailing batch axes
    (nested inner-MC). It is threaded to make_curve_tensor, which collapses them to one batch axis,
    so the gathered result's trailing axis is B*B2 and the caller reshapes it back.
    """
    time_hash = time_grid[:, TIME_GRID_MTM].tobytes()
    code_hash = tuple(x[:2] for x in code)

    key_code = ('curve', code_hash, time_hash, n_batch_dims)

    if key_code not in shared.t_Buffer:
        value = []

        for rate in code:
            rate_code = ('curve_factor', rate[:2], time_hash, n_batch_dims)

            # check if the curve factors are already available
            if rate_code not in shared.t_Buffer:
                if rate[FACTOR_INDEX_Stoch]:
                    tensor = shared.t_Scenario_Buffer[rate[FACTOR_INDEX_Offset]]
                    spread = make_curve_tensor(tensor, rate, time_grid, shared, n_batch_dims=n_batch_dims)
                else:
                    # static curve: no scenario batch axes, n_batch_dims is irrelevant.
                    tensor = shared.t_Static_Buffer[rate[FACTOR_INDEX_Offset]]
                    spread = make_curve_tensor(tensor.reshape(1, -1, 1), rate, None, shared)

                # store it
                shared.t_Buffer[rate_code] = spread

            # append the curve and its (possible) interpolation parameters
            value.append(shared.t_Buffer[rate_code])

        shared.t_Buffer[key_code] = TensorBlock(code=code, tensors=value, time_grid=time_grid)

    return shared.t_Buffer[key_code]


def calc_time_grid_spot_rate(rate, time_grid, shared):
    """Gather the composed spot rate onto `time_grid`.

    `rate` is a CODE (a list of resolved factor indices), mirroring calc_time_grid_curve_rate:
    element 0 is the primary spot and any tail elements are ObservedBasis components, so the spot is
    the SUM of the gathered components. A single-element code is the plain spot - the same ops in
    the same order, hence bit-identical.
    """
    key_code = ('spot', tuple(tuple(r[:2]) for r in rate), time_grid[:, TIME_GRID_MTM].tobytes())

    if key_code not in shared.t_Buffer:
        value = None
        for r in rate:
            if r[FACTOR_INDEX_Stoch]:
                tensor = shared.t_Scenario_Buffer[r[FACTOR_INDEX_Offset]]
                component = gather_scenario_interp(
                    build_interpolation(tensor, tenor_diff(np.zeros(1))),
                    time_grid, shared, as_curve_tensor=False)
            else:
                tensor = shared.t_Static_Buffer[r[FACTOR_INDEX_Offset]]
                component = tensor.reshape(1, -1)
            value = component if value is None else value + component

        shared.t_Buffer[key_code] = value

    return shared.t_Buffer[key_code]


def calc_curve_forwards(factor, tensor, time_grid_years, shared, mul_time=True):
    """Forward rates off a curve, for one calibrated curve or a batch of per-path curves.

    `tensor` is the curve: (n_tenors,) calibrated, or (n_tenors, B) for a BATCH. Every op below is
    elementwise or a tenor-axis gather, so the batch axis rides along as a trailing broadcast dim -
    no reduction reassociates, and the batched result is bitwise equal to looping the columns.
    `nb == 0` makes every `_bcast` a no-op reshape.
    """
    nb = tensor.dim() - 1

    def _bcast(x):
        """Right-pad tenor/time-shaped `x` with the curve's trailing batch axes."""
        return x.reshape(*x.shape, *([1] * nb))

    def prepare_tenors(factor_tenor, time_grid, extrapolate):
        """Prepare tenor grid with optional extrapolation."""
        tnr = factor_tenor.copy()
        amended_tensor = tensor
        if extrapolate:
            max_tenor = time_grid.max() + factor_tenor.max()
            tnr = np.append(tnr, max_tenor)
            # flat extrapolated gradient
            point_at_inf = tensor[-1:] + time_grid.max() * (tensor[-1:] - tensor[-2:-1]) / (
                    factor_tenor[-1] - factor_tenor[-2])
            amended_tensor = torch.cat([tensor, point_at_inf])

        tnr_d = np.diff(tnr, append=tnr.max() + 1)
        return tensor.new(tnr), tensor.new(tnr_d), amended_tensor

    def scale_for_rt(tnr, tensor, is_rt):
        """Scale tensor for rate*time interpolation."""
        if is_rt:
            return tensor * _bcast(tnr)
        return tensor

    def calculate_interp_params(tnr, tnr_d, time_grid):
        """Vectorized calculation of interpolation indices and weights."""
        #get the max index
        max_tnr_index = tnr.size()[0] - 1
        # Batch calculate all time + tenor combinations
        time_tenor = time_grid.view(-1, 1) + tnr.view(1, -1)

        # Find interpolation indices
        left_idx = (torch.searchsorted(tnr, time_tenor, right=True) - 1).clamp(min=0)
        right_idx = (left_idx + 1).clamp(max=max_tnr_index)

        left_time_idx = (torch.searchsorted(tnr, time_grid, right=True) - 1).clamp(min=0)
        right_time_idx = (left_time_idx + 1).clamp(max=max_tnr_index)

        alpha_1 = (time_tenor.clamp(max=tnr.max()) - tnr[left_idx]) / tnr_d[left_idx]
        alpha_2 = (time_grid - tnr[left_time_idx]).clamp(min=0.0) / tnr_d[left_time_idx]

        return (alpha_1, left_idx, right_idx), (alpha_2, left_time_idx, right_time_idx)

    def hermite_interpolation_new(tensor, tnr, is_rt, mul_time, full_tnr=None):

        def interp(values, indices_t):
            norm = _bcast(values) if mul_time else 1.0
            alpha, ten_t, ten_t_next = indices_t
            if is_rt:
                norm = norm / _bcast(values.clamp(full_tnr.min(), full_tnr.max()))
            return calc_hermite_curve(
                _bcast(alpha), g[ten_t], c[ten_t], tensor[ten_t], tensor[ten_t_next]) * norm

        """Handle Hermite interpolation variants."""
        t = tnr.view(1, -1, 1)
        if full_tnr is None:
            full_tnr = tnr
        # (1, n_tenors, 1) calibrated / (1, n_tenors, B) batched. Squeeze the leading axis only when
        # batched — a plain squeeze() would also eat the batch axis at B == 1
        gc = hermite_interpolation_tensor(t, tensor.reshape(1, tensor.shape[0], -1))
        g, c = [x.squeeze(0) for x in gc] if nb else [torch.squeeze(x) for x in gc]

        return interp

    def linear_interpolation_new(tensor, tnr, is_rt, mul_time, extrapolate, full_tnr=None):

        def interp(values, indices_t):
            norm = _bcast(values) if mul_time else 1.0
            alpha, ten_t, ten_t_next = indices_t

            if is_rt:
                norm = norm / _bcast(values.clamp(full_tnr.min(), full_tnr.max()))
            alpha = _bcast(alpha)
            val = alpha * tensor[ten_t_next] + (1 - alpha) * tensor[ten_t]

            # `> 1 + nb` is the tenor axis test: it selects the (time x tenor) call and skips the
            # time-only one
            if extrapolate and val.dim() > 1 + nb:
                val = val[:, :-1]
            return val * norm

        if extrapolate:
            tnr = tnr[:-1]

        if full_tnr is None:
            full_tnr = tnr

        return interp

    def calc_fwd_interpolated_new(method, l_tnr,  l_tensor, full_tnr=None):
        is_rt = method.endswith('RT')
        is_hermite = method.startswith('Hermite')

        # Handle RT scaling
        tensor = scale_for_rt(l_tnr, l_tensor, is_rt)

        # Perform interpolation
        # basic idea - (tenor_pts+t)*f(tenor_pts+t) - t*f(t) for t in the time_grid
        if is_hermite:
            return hermite_interpolation_new(tensor, l_tnr, is_rt, mul_time, full_tnr=full_tnr)
        else:
            return linear_interpolation_new(tensor, l_tnr, is_rt, mul_time, extrapolate, full_tnr=full_tnr)

    # Preprocess tensors
    if len(factor.interpolation)>1:
        interp_method = [x[-1] for x in factor.interpolation]
        extrapolate = 'Extrapolate' in interp_method[-1][0]
    else:
        interp_method = factor.interpolation[0][0]
        extrapolate = 'Extrapolate' in interp_method

    factor_tenor = factor.get_tenor()
    time_grid = tensor.new(time_grid_years)
    tnr, tnr_d, tensor = prepare_tenors(factor_tenor, time_grid_years, extrapolate)
    # Calculate interpolation indices and weights
    indices_t, indices_time = calculate_interp_params(tnr, tnr_d, time_grid)

    """Compute interpolated forward curves with support for multiple interpolation methods."""
    # see if we have more than 1 interpolation object defined
    M = time_grid.view(-1, 1) + tnr.view(1, -1)
    if len(factor.interpolation)==1:
        f = calc_fwd_interpolated_new(interp_method, tnr, tensor)
        # insert the tenor axis: (T,) -> (T, 1) calibrated, (T, B) -> (T, 1, B) batched
        t_leg = f(time_grid, indices_time)
        return f(M, indices_t) - t_leg.reshape(t_leg.shape[0], 1, *t_leg.shape[1:])
    elif len(factor.interpolation)==2:
        cuttoff_index = factor.interpolation[0][1]
        cuttoff_tenor = tnr[cuttoff_index]
        # near leg
        n = calc_fwd_interpolated_new(interp_method[0][0], tnr[:cuttoff_index+1], tensor[:cuttoff_index+1])
        near_tT = n(
            M,
            (indices_t[0],indices_t[1].clamp(max=cuttoff_index), indices_t[2].clamp(max=cuttoff_index)))
        near_t = n(
            time_grid,
            (indices_time[0], indices_time[1].clamp(max=cuttoff_index), indices_time[2].clamp(max=cuttoff_index)))
        # far leg
        f = calc_fwd_interpolated_new(interp_method[1][0], tnr[cuttoff_index:], tensor[cuttoff_index:])
        far_tT = f(
            M,
            (indices_t[0],(indices_t[1]-cuttoff_index).clamp(min=0), (indices_t[2]-cuttoff_index).clamp(min=0)))
        far_t = f(
            time_grid,
            (indices_time[0], (indices_time[1]-cuttoff_index).clamp(min=0), (indices_time[2]-cuttoff_index).clamp(min=0)))
        mask_near = _bcast(M <= cuttoff_tenor)
        time_t = torch.where(_bcast(time_grid <= cuttoff_tenor), near_t, far_t)
        return (torch.where(mask_near, near_tT, far_tT)
                - time_t.reshape(time_t.shape[0], 1, *time_t.shape[1:]))
    else:
        raise ValueError("More than 2 Interpolation Segments not supported")


def PCA(matrix, num_redim=0):
    # Compute eigenvalues and sort into descending order
    evals, evecs = np.linalg.eig(matrix)
    indices = np.argsort(evals)[::-1]
    evecs = evecs[:, indices]
    evals = evals[indices]

    if num_redim > 0:
        evecs = evecs[:, :num_redim]
        evals = evals[:num_redim]

    var = np.diag(matrix)
    aki = evecs * np.sqrt(var.reshape(-1, 1).dot(1.0 / evals.reshape(1, -1)))
    # correlation = (np.identity(var.size)/np.sqrt(var)).dot(evecs).dot(np.identity(evals.size)*np.sqrt(evals))

    return aki, evecs, evals


def calc_statistics(data_frame, method='Log', num_business_days=252.0, frequency=1, max_alpha=4.0):
    """Currently only frequency==1 is supported"""

    def calc_alpha(x, y):
        return (-num_business_days * np.log(
            1.0 + ((x - x.mean(axis=0)) * (y - y.mean(axis=0))).mean(axis=0) / ((y - y.mean(axis=0)) ** 2.0).mean(
                axis=0))).clip(0.001, max_alpha)

    def calc_sigma2(x, y, alpha):
        return (x.var(axis=0) - ((1 - np.exp(-alpha / num_business_days)) ** 2) * y.var(axis=0)) * (
                (2.0 * alpha) / (1 - np.exp(-2.0 * alpha / num_business_days)))

    def calc_theta(x, y, alpha):
        return y.mean(axis=0) + x.mean(axis=0) / (1.0 - np.exp(-alpha / num_business_days))

    def calc_log_theta(theta, sigma2, alpha):
        return np.exp(theta + sigma2 / (4.0 * alpha))

    # TODO - implement weighting
    # delta = frequency / num_business_days

    transform = {'Diff': lambda x: x, 'Log': lambda x: np.log(x.clip(0.0001, np.inf))}[method]
    transformed_df = transform(data_frame)

    # can implement decay weights here if needed

    data = transformed_df.diff(frequency).shift(-frequency)
    y = transformed_df  #
    alpha = calc_alpha(data, y)
    theta = calc_theta(data, y, alpha)
    sigma2 = calc_sigma2(data, y, alpha)

    if method == 'Log':
        theta = calc_log_theta(theta, sigma2, alpha)
        # get rid of any infs
        theta.replace([np.inf, -np.inf], np.nan, inplace=True)

        # ignore any outlier greater than 2 std deviations from the median
        median = theta.median()
        theta[np.abs(theta - median) > (2 * theta.std())] = np.nan

    stats = pd.DataFrame({
        'Volatility': data.std(axis=0) * np.sqrt(num_business_days),
        'Drift': data.mean(axis=0) * num_business_days,
        'Mean Reversion Speed': alpha,
        'Long Run Mean': theta,
        'Reversion Volatility': np.sqrt(sigma2)
    })

    correlation = data.corr()
    return stats, correlation, data


# Graph operations - needed for dependency solving

def traverse_dependents(x, adj):
    seen = set(adj[x])
    queue = deque(adj[x])
    while queue:
        i = queue.popleft()
        yield i
        for t in adj[i]:
            if t not in seen:
                seen.add(t)
                queue.append(t)


def topological_sort(graph_unsorted):
    """Move each node whose edges are all resolved onto the sorted sequence, repeatedly. DESTROYS
    `graph_unsorted` and returns the sorted keys.
    """

    graph_sorted = []

    # Run until the unsorted graph is empty.
    while graph_unsorted:

        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append(node)

        if not acyclic:
            raise RuntimeError("A cyclic dependency occurred")

    return graph_sorted


# Data transformation utilities for constructing cashflows, calculating accruals etc.

def get_day_count(code):
    if code == 'ACT_365':
        return DAYCOUNT_ACT365
    elif code == 'ACT_360':
        return DAYCOUNT_ACT360
    elif code == '_30_360':
        return DAYCOUNT_ACT30_360
    elif code == '_30E_360':
        return DAYCOUNT_ACT30_E360
    elif code == 'ACT_365_ISDA':
        return DAYCOUNT_ACT365IDSA
    elif code == 'ACT_ACT_ICMA':
        return DAYCOUNT_ACTACTICMA
    else:
        raise Exception('Daycount {} Not implemented'.format(code))


def get_day_count_accrual(reference_date, time_in_days, code):
    """Need to complete this implementation. time_in_days is incremental"""

    if code == DAYCOUNT_ACT360:
        return time_in_days / 360.0
    elif code == DAYCOUNT_ACT365:
        return time_in_days / 365.0
    elif code in (DAYCOUNT_ACT365IDSA, DAYCOUNT_ACTACTICMA):
        # TODO
        return time_in_days / 365.0
    elif code == DAYCOUNT_ACT30_360:
        e1 = min(reference_date.day, 30)
        new_date = end_date = reference_date
        if isinstance(time_in_days, np.ndarray):
            ret = []
            for ed in time_in_days.tolist():
                end_date += pd.DateOffset(days=ed)
                e2 = 30 if end_date.day >= 30 and new_date.day >= 30 else end_date.day
                ret.append(((e2 - e1) + 30 * (end_date.month - new_date.month) +
                            360 * (end_date.year - new_date.year)) / 360.0)
                new_date = end_date
            return ret
        else:
            end_date = reference_date + pd.DateOffset(days=time_in_days)
            e2 = 30 if end_date.day >= 30 and reference_date.day >= 30 else end_date.day
            return ((e2 - e1) + 30 * (end_date.month - reference_date.month) +
                    360 * (end_date.year - reference_date.year)) / 360.0
    elif code == DAYCOUNT_ACT30_E360:
        e1 = min(reference_date.day, 30)
        new_date = end_date = reference_date
        if isinstance(time_in_days, np.ndarray):
            ret = []
            for ed in time_in_days.tolist():
                end_date += pd.DateOffset(days=ed)
                e2 = min(end_date.day, 30)
                ret.append(((e2 - e1) + 30 * (end_date.month - new_date.month) +
                            360 * (end_date.year - new_date.year)) / 360.0)
                new_date = end_date
            return ret
        else:
            end_date = reference_date + pd.DateOffset(days=time_in_days)
            e2 = min(end_date.day, 30)
            return ((e2 - e1) + 30 * (end_date.month - reference_date.month) +
                    360 * (end_date.year - reference_date.year)) / 360.0
    elif code == DAYCOUNT_None:
        return time_in_days


def get_fieldname(field, obj):
    """Needed to evaluate nested fields - e.g. collateral fields"""
    if isinstance(field, tuple):
        if len(field) == 1:
            try:
                return [element.get(field[0]) for element in obj if element.get(field[0])]
            except:
                return [obj[field[0]]] if obj.get(field[0]) else []
        else:
            return get_fieldname(field[1:], obj[field[0]] if obj.get(field[0]) else ({} if len(field) > 2 else [{}]))
    else:
        return [obj[field]] if obj.get(field) else []


def check_rate_name(name):
    """Name as a tuple; rate names are upper case."""
    return tuple([x.upper() for x in name]) if type(name) == tuple else tuple(name.split('.'))


def check_tuple_name(factor):
    """Opposite of check_rate_name - used to make sure the name is a flat name"""
    return '.'.join((factor.type,) + factor.name) if type(factor.name) == tuple else factor


def resolve_factor_key(factor, price_factors):
    """The `Price Factors` key holding this factor's block, which is its own name unless a vol
    surface was written under a SIBLING 2D name.

    TRANSITIONAL, ONE RELEASE: market data exists under both the tagged and untagged spellings. A
    factor in `TwoDimensionalFactors` falls back to the untagged `VolatilityGrid.<name>` block AND
    TO NOTHING ELSE - never to a sibling tag, which would let one asset class price off another's
    surface. The requested type still decides which class is built, so the typed name stays
    canonical on write.

    RETIREMENT: once no market data carries `VolatilityGrid.*`, delete this function, drop
    `VolatilityGrid` from `TwoDimensionalFactors` and `FactorRiskClass`, and put `check_tuple_name`
    back in `riskfactors.construct_factor` and `Config.factor_universe`.
    """
    name = check_tuple_name(factor)
    if name in price_factors or factor.type not in TwoDimensionalFactors:
        return name
    # PRE-TAG spelling only: a cross-tag fallback would let an FX request silently price off an
    # equity block - right gradient label, wrong number (measured)
    legacy = check_tuple_name(Factor('VolatilityGrid', factor.name))
    return legacy if legacy in price_factors else name


def payoff_currency(field):
    """A deal's payoff currency, which is its own currency unless it says otherwise.

    Optional currency fields are declared with an EMPTY default, so 'not specified' reaches the
    engine as a present empty string from a UI and as an absent key from hand-written JSON. Both
    mean the same thing, so the test is on the VALUE rather than on presence.
    """
    return field.get('Payoff_Currency') or field['Currency']


# 0D spot factor types whose NAME may carry a composed reference: a primary spot plus one or
# more ObservedBasis periods, positional like the InterestRate curve+basis parent chain
# (InterestRate.USD_SOFR.FUNDING; here CommodityPrice.PLATINUM_CME.LME_CME).
BASIS_COMPOSABLE_TYPES = ('FxRate', 'EquityPrice', 'CommodityPrice')


def check_scope_name(factor):
    """Uses check_tuple_name but makes sure TF can use the result as a scope name"""
    return check_tuple_name(factor).translate(
        str.maketrans({'#': '_', ':': '_', ' ': '_', '(': '_', '/': '_', '+': '_', '%': '_', '*': '_', ')': '_'}))


def check_fx_name(fx_correlation):
    """The sorted pair name and the sign to read a correlation with: FX rates are named
    alphabetically, so a pair the other way round reads -rho off the sorted factor."""
    ccy1, ccy2 = fx_correlation
    return (1.0, (ccy1, ccy2)) if ccy1 < ccy2 else (-1.0, (ccy2, ccy1))


def hn_reciprocal_gamma(gamma_star):
    """The plain Heston-Nandi leverage parameter carried to the RECIPROCAL axis, under that axis'
    own numeraire.

    ONE law, two currencies: `FxRate.<ccy>` IS the density that changes numeraire, so the change
    shifts the innovation by exactly one standard deviation and a fit for `s` describes `1/s` as
    `(omega, alpha, beta, 1 - gamma*)` at that deal's own carry. A derivation, never a second fit.
    The COMPONENT family does not transport this way - its long-run intercept picks up a
    state-dependent term and leaves the family.
    """
    return 1.0 - gamma_star


def spot_model_currency(underlying, currency, base):
    """The leg of an FX pair a spot model's parameters are named for: the NON-BASE one.

    An `FxRate` is that currency priced in the base, so the base leg is the numeraire and has no law
    of its own; a CROSS - neither leg the base - keeps the underlying. The answer comes back in the
    caller's own spelling, but the comparison is on `check_rate_name` tuples, so a flat name and a
    checked one cannot disagree.

    An UNKNOWN base REFUSES: the token is not resolvable without it, and answering the underlying
    anyway is the defect - a runner would pin a model the engine looks up under the other name, and
    the deal marks at nothing.
    """
    if base is None:
        raise ValueError(
            'a spot model is keyed off the pair\'s NON-BASE token, and this book\'s base currency '
            'is not known here, so {}/{} cannot be resolved. A book declares it at System '
            'Parameters.Base_Currency (in the ExplicitMarketData block a quote reads); a deal is '
            'stamped with it by Calculation.set_deal_structures'.format(
                '.'.join(check_rate_name(underlying)), '.'.join(check_rate_name(currency))))
    return currency if check_rate_name(underlying) == check_rate_name(base) else underlying


def implied_correlation(factor, sign=1.0):
    """The market implied correlation between a rate pair, read off the `Correlation` price factor
    `Factor_dep` carries. `None` is an unauthored pair, which is uncorrelated.

    `sign` is the reverse-pair flip `check_fx_name` resolved at compile: a correlation is named on
    the sorted currency pair, so a deal running the other way reads -rho off the same factor."""
    return sign * factor.current_value()[0] if factor is not None else 0.0


def make_cashflow(reference_date, start_date, end_date, pay_date, nominal, daycount_code, fixed_amount, spread_or_rate):
    """One cashflow vector - for manually constructing a nominal or fixed payment."""
    cashflow_days = [(x - reference_date).days for x in [start_date, end_date, pay_date]]
    return np.array(
        cashflow_days + [get_day_count_accrual(reference_date, cashflow_days[1] - cashflow_days[0], daycount_code),
                         nominal, fixed_amount, spread_or_rate, 0, 0])


def get_cashflows(reference_date, reset_dates, nominal, amort, daycount_code, spread_or_rate):
    """Start_day, End_day, Pay_day, Year_Frac, Nominal, FixedAmount (=0) and rate/spread, as days
    and nominals relative to the reference date. The nominal array must be one shorter than
    `reset_dates` (there is no nominal on the effective date), or a single number for a constant
    profile.
    """

    amort_offsets = np.array([((k - reference_date).days, v) for k, v in amort.data.items()] if amort else [])
    day_offsets = np.array([(x - reference_date).days for x in reset_dates])

    nominal_amount, nominal_sign = [np.abs(nominal)], 1 if nominal > 0 else -1
    amort_index = 0
    for offset in day_offsets[1:]:
        amort_to_add = 0.0
        while amort_index < amort_offsets.shape[0] and amort_offsets[amort_index][0] <= offset:
            amort_to_add += amort_offsets[amort_index][1]
            amort_index += 1
        nominal_amount.append(nominal_amount[-1] - amort_to_add)
    nominal_amount = nominal_sign * np.array(nominal_amount)

    # we want the earliest negative number
    last_payment = np.where(day_offsets >= 0)[0]

    # calculate the index of the earliest cashflow
    previous_index = max(last_payment[0] - 1 if last_payment.size else day_offsets.size, 0)
    cashflows_left = day_offsets[previous_index:]
    rates = spread_or_rate if isinstance(nominal, np.ndarray) else [spread_or_rate] * (reset_dates.size - 1)
    ref_date = (reference_date + pd.offsets.Day(cashflows_left[0])) \
        if cashflows_left.any() else reference_date

    # order is start_day, end_day, pay_day, daycount_accrual, nominal, fixed amount, FxResetDate, FXResetValue

    return zip(cashflows_left[:-1], cashflows_left[1:], cashflows_left[1:],
               get_day_count_accrual(ref_date, np.diff(cashflows_left), daycount_code),
               nominal_amount[previous_index:], np.zeros(cashflows_left.size - 1), rates[previous_index:],
               np.zeros(cashflows_left.size - 1), np.zeros(cashflows_left.size - 1))


def generate_float_cashflows(reference_date, time_grid, reset_dates, nominal, amort, known_rate_list, reset_tenor,
                             reset_frequency, daycount_code, spread):
    """`get_cashflows`' schedule plus the reset structure. The nominal array must be one shorter
    than `reset_dates`, or a single number for a constant profile.
    """

    cashflow_schedule = list(get_cashflows(reference_date, reset_dates, nominal, amort, daycount_code, spread))
    cashflow_reset_offsets = []
    all_resets = []
    reset_scenario_offsets = []

    # prepare to consume reset dates
    known_rates = known_rate_list if known_rate_list is not None else DateList({})
    known_rates.prepare_dates()

    min_date = None
    for cashflow in cashflow_schedule:
        r = []
        if next(iter(reset_frequency.kwds.values())) == 0.0:
            reset_days = np.array([reference_date + pd.DateOffset(days=int(cashflow[CASHFLOW_INDEX_Start_Day]))])
            reset_tenor = pd.offsets.Day(cashflow[CASHFLOW_INDEX_End_Day] - cashflow[CASHFLOW_INDEX_Start_Day])
        else:
            reset_days = pd.date_range(reference_date + pd.DateOffset(days=int(cashflow[CASHFLOW_INDEX_Start_Day])),
                                       reference_date + pd.DateOffset(days=int(cashflow[CASHFLOW_INDEX_End_Day])),
                                       freq=reset_frequency, inclusive='left')
            reset_tenor = reset_frequency if next(iter(reset_tenor.kwds.values())) == 0.0 else reset_tenor

        for reset_day in reset_days:
            Reset_Day = (reset_day - reference_date).days
            Start_Day = (reset_day - reference_date).days
            End_Day = (reset_day + reset_tenor - reference_date).days
            Accrual = get_day_count_accrual(reference_date, End_Day - Start_Day, daycount_code)
            Weight = 1.0 / reset_days.size
            Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)

            # match the closest reset
            closest_date, Value = known_rates.consume(min_date, reset_day)
            if closest_date is not None:
                min_date = closest_date if min_date is None else max(min_date, closest_date)

            # only add a reset if it's in the past
            r.append([Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight,
                      Value / 100.0 if reset_day < reference_date else 0.0, Accrual])
            reset_scenario_offsets.append(Scenario)

            if Start_Day == End_Day:
                raise Exception("Reset Start and End Days coincide")

        # attach the reset_offsets to the cashflow - assume each cashflow is a settled one (not accumulated)
        cashflow_reset_offsets.append([len(r), len(all_resets), 1])
        # store resets
        all_resets.extend(r)

    cashflows = TensorCashFlows(cashflow_schedule, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)

    return cashflows


def generate_fixed_cashflows(reference_date, reset_dates, nominal, amort, daycount_code, fixed_rate):
    """`get_cashflows`' schedule with null resets. The nominal array must be one shorter than
    `reset_dates`, or a single number for a constant profile.
    """
    cashflow_schedule = list(get_cashflows(reference_date, reset_dates, nominal, amort, daycount_code, fixed_rate))
    # Add the null resets to the end
    dummy_resets = np.zeros((len(cashflow_schedule), 3))

    return TensorCashFlows(cashflow_schedule, dummy_resets)


def make_fixed_cashflows(reference_date, position, cashflows, settlement_date):
    """Fixed cashflows from a data source, taking nominal amounts into account."""
    cash = []
    reset_offsets = []

    for cashflow in sorted(
            cashflows['Items'], key=lambda x: (x['Payment_Date'], x.get('Accrual_Start_Date', x['Payment_Date']))):
        rate = cashflow['Rate'] if isinstance(cashflow['Rate'], float) else cashflow['Rate'].amount
        if cashflow['Payment_Date'] >= reference_date and (
                (cashflow['Payment_Date'] >= settlement_date) if settlement_date else True):
            # check the accrual dates - if none set it to the payment date
            Accrual_Start_Date = cashflow['Accrual_Start_Date'] if cashflow[
                'Accrual_Start_Date'] else cashflow['Payment_Date']
            Accrual_End_Date = cashflow['Accrual_End_Date'] if cashflow[
                'Accrual_End_Date'] else cashflow['Payment_Date']

            cash.append([(Accrual_Start_Date - reference_date).days, (Accrual_End_Date - reference_date).days,
                         (cashflow['Payment_Date'] - reference_date).days,
                         cashflow['Accrual_Year_Fraction'], position * cashflow['Notional'],
                         position * cashflow.get('Fixed_Amount', 0.0), rate, 0.0, 0.0])

            # needed to deal with forward settlement
            reset_offsets.append([0, 0, 0 if settlement_date is None else -(settlement_date - reference_date).days])

    return TensorCashFlows(cash, reset_offsets)


def make_sampling_data(reference_date, time_grid, samples):
    all_resets = []
    reset_scenario_offsets = []
    D = float(sum([x[-1] for x in samples]))

    for sample in sorted(samples):
        Reset_Day = (sample[0] - reference_date).days
        Start_Day = Reset_Day
        End_Day = Reset_Day
        Weight = sample[-1] / D
        Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
        # only add a reset if its in the past
        all_resets.append(
            [Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight,
             sample[-2] if sample[0] < reference_date else 0.0, 0.0])
        reset_scenario_offsets.append(Scenario)

    return TensorResets(all_resets, reset_scenario_offsets)


def make_fixing_data(reference_date, time_grid, fixings):
    all_resets = []
    reset_scenario_offsets = []

    for fixing in sorted(fixings):
        Reset_Day = (fixing[0] - reference_date).days
        Start_Day = Reset_Day
        End_Day = Reset_Day
        Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
        # only add a reset if it's in the past
        all_resets.append(
            [Time_Grid, Reset_Day, -1, Start_Day, End_Day, 1.0,
             fixing[-1] if fixing[0] < reference_date else 0.0, 0.0])
        reset_scenario_offsets.append(Scenario)

    return TensorResets(all_resets, reset_scenario_offsets)


def make_simple_fixed_cashflows(reference_date, position, cashflows):
    """Fixed cashflows from a data source, reading the fixed value alone."""
    cash = {}
    for cashflow in sorted(cashflows['Items'], key=lambda x: x['Payment_Date']):
        if cashflow['Payment_Date'] >= reference_date:
            tenor = (cashflow['Payment_Date'] - reference_date).days
            if tenor in cash:
                cash[tenor][5] += position * cashflow['Fixed_Amount']
            else:
                cash.setdefault(tenor, [tenor, tenor, tenor, 1.0, 0.0,
                                        position * cashflow['Fixed_Amount'], 0.0, 0.0, 0.0])

    # Add the null resets to the end
    dummy_resets = np.zeros((len(cash), 3))

    return TensorCashFlows(list(cash.values()), dummy_resets)


def make_energy_fixed_cashflows(reference_date, position, cashflows):
    """Energy fixed cashflows from a data source, reading the fixed value alone."""
    cash = []
    for cashflow in sorted(cashflows['Items'], key=lambda x: x['Payment_Date']):
        if cashflow['Payment_Date'] >= reference_date:
            cash.append(
                [(cashflow['Payment_Date'] - reference_date).days, (cashflow['Payment_Date'] - reference_date).days,
                 (cashflow['Payment_Date'] - reference_date).days,
                 1.0, 0.0, position * cashflow['Volume'] * cashflow['Fixed_Price'], 0.0, 0.0, 0.0])

    # Add the null resets to the end
    dummy_resets = np.zeros((len(cash), 3))

    return TensorCashFlows(cash, dummy_resets)


def make_equity_swaplet_cashflows(base_date, time_grid, position, cashflows, current_spot, busday):
    """Equity cashflows from a data source."""
    cash = []
    all_resets = []
    cashflow_reset_offsets = []
    reset_scenario_offsets = []

    for cashflow in sorted(cashflows['Items'], key=lambda x: (x['Payment_Date'], x['End_Date'], x['Start_Date'])):
        if cashflow['Payment_Date'] >= base_date:
            cash.append([(cashflow['Start_Date'] - base_date).days, (cashflow['End_Date'] - base_date).days,
                         (cashflow['Payment_Date'] - base_date).days, cashflow.get('Start_Multiplier', 1.0),
                         cashflow.get('End_Multiplier', 1.0), position * cashflow['Amount'],
                         cashflow.get('Dividend_Multiplier', 1.0),
                         (cashflow['Start_Date'] + busday - base_date).days,
                         (cashflow['End_Date'] + busday - base_date).days])

            r = []
            for reset in ['Start', 'End']:
                Reset_Day = (cashflow[reset + '_Date'] - base_date).days
                Start_Day = Reset_Day
                # we map the weight of the reset with the prior dividends
                Weight = cashflow.get('Known_Dividend_Sum', 0.0)

                # Need to use this reset to estimate future dividends
                Time_Grid, Scenario = time_grid.get_scenario_offset(max(Reset_Day, 0))

                # only add a reset if it's in the past - if its 0, then replace it with the current spot
                if Start_Day <= 0:
                    known_price = cashflow.get('Known_' + reset + '_Price', 0.0)
                    if Start_Day == 0 and not known_price:
                        logging.warning(
                            'Known_{}_Price not set at base_date - setting to current spot'.format(reset))
                        reset_price = current_spot
                    else:
                        reset_price = known_price
                else:
                    reset_price = 0.0

                r.append([Time_Grid, Reset_Day, -1, Start_Day, Start_Day, Weight,
                          reset_price,
                          cashflow.get('Known_' + reset + '_FX_Rate', 0.0) if Start_Day <= 0 else 0.0])
                reset_scenario_offsets.append(Scenario)

            # attach the reset_offsets to the cashflow
            cashflow_reset_offsets.append([len(r), len(all_resets), 0])
            # store resets
            all_resets.extend(r)

    cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)
    # calculate the business day ajustment on the mtm time grid
    bus_offset = np.array([((x + busday) - x).days for x in sorted(time_grid.mtm_dates)])
    return cashflows, bus_offset


def index_reference_samples(pricing_date, months_lag, interpolated):
    """The (date, weight) index observations an inflation reference reads.

    A non-interpolated reference reads one month-start, `months_lag` months back; an interpolated
    one straddles two, weighted by how far into its own month the pricing date sits. Keeping the
    rule and the lag separate admits any lag, which is what the schema always declared.
    """
    if not interpolated:
        return [((pricing_date - pd.DateOffset(months=months_lag)).to_period('M').to_timestamp('D'), 1.0)]

    month_start = pricing_date.to_period('M').to_timestamp('D')
    w = (pricing_date - month_start).days / float(
        ((month_start + pd.DateOffset(months=1)) - month_start).days)
    return [((pricing_date - pd.DateOffset(months=lag)).to_period('M').to_timestamp('D'), weight)
            for lag, weight in ((months_lag, 1.0 - w), (months_lag - 1, w))]


def make_index_cashflows(base_date, time_grid, position, cashflows, price_index, index_rate,
                         settlement_date, months_lag, interpolated, isBond=True):
    """Index-linked cashflows from a data source, against the price_index and index_rate factors."""

    def index_reference(pricing_date, lagged_date, resets, offsets):
        for Day, Weight in index_reference_samples(pricing_date, months_lag, interpolated):
            Rel_Day = (Day - lagged_date).days
            Value = index_rate.get_reference_value(Day) if Day <= lagged_date else 0.0
            Time_Grid, Scenario = time_grid.get_scenario_offset(Rel_Day) if Rel_Day >= 0.0 else (0, -1)
            resets.append([Time_Grid, Rel_Day, -1, Rel_Day, Rel_Day, Weight, Value, 0.0])
            offsets.append(Scenario)


    cash = []
    cashflow_reset_offsets = []
    # resets at different points in time
    time_resets = []
    time_scenario_offsets = []
    # resets per cashflow
    base_resets = []
    base_scenario_offsets = []
    final_resets = []
    final_scenario_offsets = []

    for cashflow in sorted(cashflows['Items'], key=lambda x: x['Payment_Date']):
        if cashflow['Payment_Date'] >= base_date and (
                (cashflow['Payment_Date'] >= settlement_date) if settlement_date else True):
            Pay_Date = (cashflow['Payment_Date'] - base_date).days
            Accrual_Start_Date = (cashflow['Accrual_Start_Date'] - base_date).days \
                if cashflow.get('Accrual_Start_Date') else Pay_Date
            Accrual_End_Date = (cashflow['Accrual_End_Date'] - base_date).days \
                if cashflow.get('Accrual_End_Date') else Pay_Date
            base_reference_date = cashflow.get('Base_Reference_Date') \
                if cashflow.get('Base_Reference_Date') else base_date
            final_reference_date = cashflow.get('Final_Reference_Date') \
                if cashflow.get('Final_Reference_Date') else base_date

            cash.append([Accrual_Start_Date, Accrual_End_Date, Pay_Date, cashflow['Accrual_Year_Fraction'],
                         position * cashflow['Notional'], cashflow['Rate_Multiplier'], cashflow['Yield'].amount, 0.0,
                         0.0])

            # attach the base and final reference dates to the cashflow
            cashflow_reset_offsets.append(
                [cashflow['Base_Reference_Value'] if cashflow['Base_Reference_Value'] else -(
                        base_reference_date - base_date).days,
                 cashflow['Final_Reference_Value'] if cashflow['Final_Reference_Value'] else -(
                         final_reference_date - base_date).days,
                 Pay_Date if settlement_date is None else -(settlement_date - base_date).days])

            if isBond:
                index_reference(
                    base_reference_date, base_date, base_resets, base_scenario_offsets)
                index_reference(
                    final_reference_date, base_date, final_resets, final_scenario_offsets)

    # set the cashflows
    cashflows = TensorCashFlows(sorted(cash), cashflow_reset_offsets)
    # check if the paydays are still sorted
    if (cashflows.schedule[:, CASHFLOW_INDEX_Pay_Day] != sorted(cashflows.schedule[:, CASHFLOW_INDEX_Pay_Day])).any():
        logging.error("Cashflow Pay Day not in sorted order - check accrual dates")

    if isBond:
        mtm_grid = time_grid.time_grid[:, TIME_GRID_MTM]

        for last_published_date in index_rate.get_last_publication_dates(base_date, mtm_grid):
            # calc the number of days since last published date to the base date
            Rel_Day = (last_published_date - base_date).days
            Value = index_rate.get_reference_value(last_published_date) if last_published_date <= index_rate.param[
                'Last_Period_Start'] else 0.0

            time_resets.append([0.0, Rel_Day, Rel_Day, Rel_Day, -1, 1.0, Value, 0.0])
            time_scenario_offsets.append(0)

        cashflows.set_resets(time_resets, time_scenario_offsets)

        return cashflows, TensorResets(base_resets, base_scenario_offsets), TensorResets(
            final_resets, final_scenario_offsets)

    else:
        for eval_time in time_grid.time_grid[:, TIME_GRID_MTM]:
            actual_time = base_date + pd.DateOffset(days=eval_time)

            index_reference(
                actual_time, index_rate.param['Last_Period_Start'], time_resets, time_scenario_offsets)

        cashflows.set_resets(time_resets, time_scenario_offsets)

        return cashflows


def make_float_cashflows(reference_date, time_grid, position, cashflows, reference=None):
    """Floating cashflows from a data source.

    `reference` is the deal's own, here for the refusal below: a reset with a zero-length rate
    window is refused BY NAME, and a refusal that cannot say which deal it is about is one a desk
    cannot act on.
    """
    cash = []
    all_resets = []
    cashflow_reset_offsets = []
    reset_scenario_offsets = []

    for cashflow in sorted(
            cashflows['Items'], key=lambda x: (x['Payment_Date'], x['Accrual_End_Date'], x['Accrual_Start_Date'])):

        if cashflow['Payment_Date'] >= reference_date:
            # potential FX resets
            fx_reset_date = (cashflow.get('FX_Reset_Date') - reference_date).days \
                if cashflow.get('FX_Reset_Date') else 0.0
            fx_reset_val = cashflow.get('Known_FX_Rate', 0.0)

            cash.append([(cashflow['Accrual_Start_Date'] - reference_date).days,
                         (cashflow['Accrual_End_Date'] - reference_date).days,
                         (cashflow['Payment_Date'] - reference_date).days,
                         cashflow['Accrual_Year_Fraction'], position * cashflow['Notional'],
                         position * cashflow.get('Fixed_Amount', 0.0), cashflow['Margin'].amount,
                         fx_reset_date, fx_reset_val])

            r = []
            for reset in cashflow['Resets']:
                # A DEGENERATE RATE WINDOW IS REFUSED, not derived: the rate tenor is a quantity the
                # author did not state and no rule recovers - the accrual window is the period's,
                # not the rate's - so widening one would be a number nobody quoted
                if reset[2] == reset[1]:
                    raise UnpriceableSchedule(
                        '{}: the reset fixing {:%Y-%m-%d} on the cashflow paying {:%Y-%m-%d} has a '
                        'rate window that starts and ends on {:%Y-%m-%d}. A zero-length window has '
                        'no forward rate to read and the schedule states no tenor to widen it to. '
                        'Author the reset\'s rate end date after its rate start (the accrual end '
                        'is the usual one), or drop the reset.'.format(
                            reference or 'this CashflowListDeal', reset[0],
                            cashflow['Payment_Date'], reset[1]))

                # create the reset vector
                Reset_Day = (reset[0] - reference_date).days
                Start_Day = (reset[1] - reference_date).days
                End_Day = (reset[2] - reference_date).days
                Accrual = reset[3]
                Weight = 1.0 / len(cashflow['Resets'])
                Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
                # only add a reset if it's in the past
                r.append([Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight,
                          reset[-1].amount if reset[0] < reference_date else 0.0, Accrual])
                reset_scenario_offsets.append(Scenario)

            # attach the reset_offsets to the cashflow
            cashflow_reset_offsets.append([len(r), len(all_resets), 0])
            # store resets
            all_resets.extend(r)

    cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)

    return cashflows


def make_energy_cashflows(reference_date, time_grid, position, cashflows, reference, forwardsample, fxsample,
                          calendars):
    """Floating/fixed cashflows from a data source under the energy model.
    TODO: allow an fxSample different from the forwardsample.
    """
    cash = []
    all_resets = []
    cashflow_reset_offsets = []
    reset_scenario_offsets = []
    forward_calendar_bday = calendars.get(forwardsample.get_holiday_calendar(), {'businessday': 'B'})['businessday']

    for cashflow in sorted(cashflows['Items'], key=lambda x: (x['Payment_Date'], x['Period_End'], x['Period_Start'])):
        if cashflow['Payment_Date'] >= reference_date:
            cash.append(
                [(cashflow['Period_Start'] - reference_date).days, (cashflow['Period_End'] - reference_date).days,
                 (cashflow['Payment_Date'] - reference_date).days, cashflow.get('Price_Multiplier', 1.0),
                 position * cashflow['Volume'], 0.0, cashflow.get('Fixed_Basis', 0.0), 0.0, 0.0])

            r = []
            bunsiness_dates = pd.date_range(
                cashflow['Period_Start'], cashflow['Period_End'], freq=forward_calendar_bday)

            if forwardsample.get_sampling_convention() == 'ForwardPriceSampleDaily':
                # create daily samples
                reset_dates = bunsiness_dates

            elif forwardsample.get_sampling_convention() == 'ForwardPriceSampleBullet':
                # create one sample
                reset_dates = [bunsiness_dates[-1]]

            resets_in_excel_format = np.array([(x - reference.start_date).days for x in reset_dates])
            reference_date_excel = (reference_date - reference.start_date).days

            # retrieve the fixing dates from the reference curve and adding an offset
            fixing_dates = reference.get_fixings(resets_in_excel_format + forwardsample.param.get('Offset', 0))

            for reset_day, fixing_day in zip(resets_in_excel_format, fixing_dates):
                Reset_Day = reset_day - reference_date_excel
                # Start_Day = reset_day - reference_date_excel
                Start_Day = reset_day
                End_Day = fixing_day
                Weight = 1.0 / len(reset_dates)
                Time_Grid, Scenario = time_grid.get_scenario_offset(Reset_Day)
                # only add a reset if its in the past
                r.append([Time_Grid, Reset_Day, -1, Start_Day, End_Day, Weight,
                          cashflow['Realized_Average'] or 0.0, cashflow['FX_Realized_Average'] or 0.0])
                reset_scenario_offsets.append(Scenario)

            # attach the reset_offsets to the cashflow
            cashflow_reset_offsets.append([len(r), len(all_resets), 0])
            # store resets
            all_resets.extend(r)

    cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
    cashflows.set_resets(all_resets, reset_scenario_offsets)

    return cashflows


def compress_deal_data(deals):
    def filter_deals(deals, values):
        filtered = []
        unfiltered = []
        for deal in deals:
            (filtered if deal['Instrument'].field['Reference'] in values else unfiltered).append(deal)
        return filtered, unfiltered

    def compress_CFFloatingInterestListDeal(unders, ref, use_ref_as_tag=False):
        compressed = []
        all_margin = {}
        all_notional = {}
        for deal in unders:
            buy_sell = 1.0 if deal['Instrument'].field['Buy_Sell'] == 'Buy' else -1.0
            prop_key = tuple(sorted(
                [(k, v) for k, v in deal['Instrument'].field['Cashflows'].items() if k != 'Items']))
            margin_list = all_margin.setdefault(prop_key, {})
            notional_list = all_notional.setdefault(prop_key, {})
            for cf in deal['Instrument'].field['Cashflows']['Items']:
                cf_key = tuple(sorted(
                    [(k, v) for k, v in cf.items() if k not in ['Notional', 'Resets', 'Margin']]))
                reset_key = tuple(sorted([tuple(x) for x in cf['Resets']]))
                key = (cf_key, reset_key)
                notional = buy_sell * cf['Notional']
                margin_list[key] = margin_list.setdefault(key, 0.0) + cf['Margin'] * notional
                notional_list[key] = notional_list.setdefault(key, 0.0) + notional

        # finish this off
        prop_index = 0
        for cf_prop, margin_list in all_margin.items():
            leg = []
            existing_deals = unders[prop_index:]
            notional_list = all_notional[cf_prop]
            for key, val in margin_list.items():
                notional = notional_list[key]
                cashflow = dict(key[0])
                cashflow['Resets'] = [list(x) for x in list(key[1])]
                if notional:
                    cashflow['Notional'] = notional
                    cashflow['Margin'] = Basis(10000.0 * val / notional)
                    leg.append(cashflow)
                elif val:
                    cashflow['Notional'] = val
                    cashflow['Margin'] = Basis(10000.0)
                    leg.append(cashflow)
                    logging.warning('Float Cashflow Nominal compressed to 0.0 and margin is not 0 - TEST')
                else:
                    logging.info('Float Cashflow Nominal compressed to 0.0 and margin is 0 - will be skipped')

            # check that there are no overlapping resets (if so - create a new leg)
            final = sorted(leg, key=lambda x: (x['Payment_Date'], x['Accrual_Start_Date'], x['Accrual_End_Date']))
            # can just check the first reset because we sorted them earlier
            splits = [i + 1 for i, (x, y) in enumerate(
                zip(final[:-1], final[1:])) if x['Resets'][0][0] > y['Resets'][0][0]]

            if len(splits) >= len(existing_deals):
                # can happen with e.g. prime linked swaps (many resets per day)
                # check to see if we must edit the tag
                for deal in existing_deals:
                    if use_ref_as_tag:
                        deal['Instrument'].field['Tags'] = list(ref)
                    # add the deal uncompressed
                    compressed.append(deal)
            else:
                for i, (deal, m, n) in enumerate(zip(existing_deals, [0] + splits, splits + [None])):
                    legnum = '_Leg{}'.format(i) if splits else ''
                    deal['Instrument'].field['Buy_Sell'] = 'Buy'
                    deal['Instrument'].field['Cashflows'] = dict(cf_prop)
                    deal['Instrument'].field['Cashflows']['Items'] = final[m:n]
                    if use_ref_as_tag:
                        deal['Instrument'].field['Reference'] = 'Compressed_CFFloat_{}_{}{}'.format(
                            'Buy', deal['Instrument'].field['Currency'], legnum)
                        deal['Instrument'].field['Tags'] = list(ref)
                    else:
                        deal['Instrument'].field['Reference'] = 'Compressed_CFFloat_{}_{}{}'.format('Buy', ref, legnum)

                    compressed.append(deal)

                # move the existing deal index forward
                prop_index += i + 1

        return compressed

    # return this as our compressed portfolio
    reduced_deals = deals
    # first try and compress equity_swaps
    equity_swaps = [x for x in reduced_deals if x['Instrument'].field['Object'] == 'EquitySwapletListDeal']
    # don't bother if there are less than 400 swaps
    if equity_swaps and len(equity_swaps) > 400:
        logging.info('Compressing {} EquitySwaplets'.format(len(equity_swaps)))
        eq_unders = {}
        ir_unders = {}
        eq_swap_ref = {x['Instrument'].field['Reference']: x['Instrument'].field['Equity'] for x in equity_swaps}
        all_eq_swap, all_other = filter_deals(reduced_deals, eq_swap_ref.keys())

        # first load all compressible deals
        for k in all_eq_swap:
            key = tuple(
                sorted([(field, tuple(value) if isinstance(value, list) else value)
                        for field, value in k['Instrument'].field.items()
                        if field not in ['Reference', 'Buy_Sell', 'Cashflows']]))

            if k['Instrument'].field['Object'] == 'EquitySwapletListDeal':
                # need to split buys and sells because there could be at different prices for the same day
                buy_sell = (('Buy_Sell', k['Instrument'].field['Buy_Sell']),)
                eq_unders.setdefault(key + buy_sell, []).append(k)
            else:
                # pair up with the equity leg so that it's easy to track funding per stock
                under_eq = eq_swap_ref[k['Instrument'].field['Reference']]
                ir_unders.setdefault(key + (under_eq,), []).append(k)

        # now compress
        eq_compressed = {}
        for k, unders in eq_unders.items():
            cf_list = {}
            for deal in unders:
                for cf in deal['Instrument'].field['Cashflows']['Items']:
                    key = tuple([(k, v) for k, v in cf.items() if k != 'Amount'])
                    cf_list[key] = cf_list.setdefault(key, 0.0) + cf['Amount']

            # edit the last deal
            deal['Instrument'].field['Cashflows']['Items'] = [dict(k + (('Amount', v),)) for k, v in cf_list.items()]
            deal['Instrument'].field['Reference'] = 'Compressed_EQSwaplet_{}_{}'.format(
                deal['Instrument'].field['Buy_Sell'], deal['Instrument'].field['Equity'])
            eq_compressed.setdefault(deal['Instrument'].field['Equity'], []).append(deal)

        ir_compressed = {}
        for k, unders in ir_unders.items():
            ir_compressed.setdefault(k[-1], []).extend(compress_CFFloatingInterestListDeal(unders, k[-1]))

        for k, v in eq_compressed.items():
            all_other.extend(v)
            all_other.extend(ir_compressed[k])

        reduced_deals = all_other

    return reduced_deals


def compress_no_compounding(cashflows, groupsize, check_resets=True):
    '''Approximate many resets by fewer groups, or return the cashflows unchanged.

    :param groupsize: -1 keeps every reset and only regroups them; otherwise sample this many
        groups per cashflow
    :param check_resets: require every reset to be in the future
    '''
    cash_pmts, cash_index, cash_counts = np.unique(
        cashflows.schedule[:, CASHFLOW_INDEX_Pay_Day], return_index=True, return_counts=True)

    if (cashflows.offsets[:, 0] == 1).all():
        if (cash_counts > abs(groupsize)).any():
            # can compress
            cash, cashflow_reset_offsets = [], []
            all_resets, reset_scenario_offsets = [], []
            for pay_day, index, num_cf in zip(*[cash_pmts, cash_index, cash_counts]):
                cashflow_schedule = cashflows.schedule[index:index + num_cf]
                cashflow_offsets = cashflows.offsets[index:index + num_cf]
                reset_offset = cashflows.offsets[index:index + num_cf, 1]
                nominals = np.unique(cashflow_schedule[:, CASHFLOW_INDEX_Nominal])
                margins = np.unique(cashflow_schedule[:, CASHFLOW_INDEX_FloatMargin])

                if groupsize == -1 and nominals.size == 1 and margins.size == 1:
                    # we can compress this
                    cash.append(
                        [cashflow_schedule[0, CASHFLOW_INDEX_Start_Day],
                         cashflow_schedule[-1, CASHFLOW_INDEX_End_Day],
                         pay_day,
                         cashflow_schedule[:, CASHFLOW_INDEX_Year_Frac].sum(),
                         cashflow_schedule[:, CASHFLOW_INDEX_Nominal].mean(),
                         cashflow_schedule[:, CASHFLOW_INDEX_FixedAmt].sum(),
                         cashflow_schedule[:, CASHFLOW_INDEX_FloatMargin].mean(),
                         cashflow_schedule[0, CASHFLOW_INDEX_FXResetDate],
                         cashflow_schedule[0, CASHFLOW_INDEX_FXResetValue]])

                    cashflow_reset_offsets.append([num_cf, index, 1])
                    all_resets.extend(cashflows.Resets[reset_offset].tolist())
                    reset_scenario_offsets.extend(cashflows.Resets.offsets[reset_offset].tolist())

                elif nominals.size <= groupsize and margins.size <= groupsize and (check_resets and not (
                        cashflows.Resets[reset_offset, RESET_INDEX_Reset_Day] < 0).any() or not check_resets):
                    # we can compress this
                    for cash_group, ofs_group in zip(*map(
                            lambda x: np.array_split(x, groupsize), [cashflow_schedule, cashflow_offsets])):
                        cash.append(
                            [cash_group[0, CASHFLOW_INDEX_Start_Day],
                             cash_group[-1, CASHFLOW_INDEX_End_Day],
                             pay_day,
                             cash_group[:, CASHFLOW_INDEX_Year_Frac].sum(),
                             # not strictly correct - need to break this up - TODO
                             cash_group[:, CASHFLOW_INDEX_Nominal].mean(),
                             cash_group[:, CASHFLOW_INDEX_FixedAmt].sum(),
                             # not strictly correct - need to break this up - TODO
                             cash_group[:, CASHFLOW_INDEX_FloatMargin].mean(),
                             cash_group[0, CASHFLOW_INDEX_FXResetDate],
                             cash_group[0, CASHFLOW_INDEX_FXResetValue]])

                        reset_index = ofs_group[ofs_group[:, 1].size // 2, 1]
                        cashflow_reset_offsets.append([1, len(all_resets), 0])
                        reset_scenario_offsets.append(cashflows.Resets.offsets[reset_index])
                        all_resets.append(cashflows.Resets[reset_index].tolist())

                else:
                    # copy as is
                    cash.extend(cashflow_schedule.tolist())
                    all_resets.extend(cashflows.Resets[reset_offset].tolist())
                    reset_scenario_offsets.extend(cashflows.Resets.offsets[reset_offset].tolist())
                    cashflow_reset_offsets.extend(cashflows.offsets[index:index + num_cf].tolist())

            approx_cashflows = TensorCashFlows(cash, cashflow_reset_offsets)
            approx_cashflows.set_resets(all_resets, reset_scenario_offsets)
            if len(cashflows.Resets) == len(approx_cashflows.Resets):
                logging.warning('Cashflows rebased from {} resets'.format(len(cashflows.Resets)))
            else:
                logging.warning('Cashflows reduced from {} resets to {} resets'.format(
                    len(cashflows.Resets), len(approx_cashflows.Resets)))
            return approx_cashflows

    return cashflows


if __name__ == '__main__':
    pass
