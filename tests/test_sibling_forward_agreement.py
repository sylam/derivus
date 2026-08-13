"""SIBLING AGREEMENT: one financial quantity, one expression.

The defect this gate exists for was not a wrong formula - it was TWO formulas for the forward to
expiry inside ``pv_discrete_barrier_option``, one of them unexercised and drifted. The in-out-parity
leg inside ``sim_spot_oss`` and the already-hit KI leg value the SAME European on the SAME state, and
the already-hit one summed annualised RATES with no ``dt`` while adding a half-variance whose
cancelling subtraction lived on the other branch. It read 106.7% high in log-forward on this repo's
own fixture and shipped for as long as the deal existed.

Nothing in the suite could see it, and the reason generalises: every barrier gate priced under BASE
VALUATION (one deal-time row, so the hit mask is all-False and this leg is never evaluated), the one
barrier that ran an exposure grid was ``Down_And_Out`` (where the leg is the model-free zeros
branch), and every fixture set ``r = q = 0``, which zeroes the missing ``dt``. Three independent
degeneracies. An ORACLE gate would have needed all three removed at once; an INVARIANCE gate would
have needed the path executed with a discriminating fixture. A CONSISTENCY gate needs neither - no
reference value, no market data, no Monte Carlo. It only asks whether two routes to one number
agree, and it is the cheapest strong gate in the box.

WHAT IS ASSERTED, in rising order of how much of the pricer it reaches:

  1. ``total_log_forward`` is rank-polymorphic. One MTM row and a whole block are the same
     expression, which is what lets both legs call it.
  2. Both legs of the barrier read that one expression on the same inputs. With the shared helper
     this holds by construction - the gate's job is to notice the day someone inlines it back.
  3. The forward the PAYOFF is valued at equals the forward the VOL SURFACE is read at. These are
     genuinely independent routes: one gathers the carry curve once at ``tau``, the other gathers it
     at every fixing and integrates over the strip. This is the assertion that would have killed the
     original defect on its own, and it needs no hit, no barrier crossing and no second model.
  4. The same, on a SLOPED carry curve.
  5. The strip primitive against its own definition - pure algebra, no pricer.
  6. The VOL strip's last forward is that same forward. Its fixings are read at their own forwards
     and the last fixing is expiry, so the surface's per-fixing route has to land on the payoff's.

Assertion (3) is EXACT (``torch.equal``) on a flat curve and it is not a placebo there: ``r = 5%``
against ``q = 1%`` makes the forward 1.0408 at the first row, not 1.0, so a dropped ``dt`` or a
spurious half-variance moves it.

(4) WAS A STRICT XFAIL: a second, independent defect this gate found and did not fix. The
per-interval carry was built as ``drifts * sample_ts`` - the AVERAGE rate to a fixing times the
length of a different, shorter window - which is only the interval integral on a flat curve or at
the first fixing. Measured 4.276e-02 on this fixture. ``pricing.forward_carry_rate`` now builds the
strip by differencing the cumulative integrals, which is what the two siblings always did and what
all FOUR adopters now read; the disagreement is 2.220e-16 and the xfail is a live assertion. In
PRICE, on a never-knocking barrier against Black on the same sloped world: -20.10% before, +0.045%
after.

MUTATION MATRIX (each applied to derivus/pricing.py or its module globals, run, reverted; all six
re-measured against the five gates as they stand):

    mutation                                                    1    2    3    4    5    6
    (1) total_log_forward drops `times` (the original defect)  DIED  --  DIED DIED DIED DIED
    (2) it sums over the batch axis instead of the fixings     DIED DIED DIED DIED DIED DIED
    (3) already-hit call site scales sample_ts by 1.001         --  DIED   --   --    --   --
    (4) parity call site scales carry by 1.001                  --  DIED DIED DIED   --   --
    (5) forward_carry_rate returns carry_rate (the 2nd defect)  --   --    --  DIED DIED  --
    (6) barrier theta passes `drifts` for `fwd_drifts`          --  DIED   --  DIED   --   --
    (7) the VOL strip built on the INTERVAL carry               --   --    --   --    --  DIED

(7) is the sibling gate 6 exists for and it is caught by NOTHING else in the repo: the strip drives
the surface read, not the payoff, so every price gate on a flat-in-moneyness fixture is blind to it
and the carry gates above never look at it. 8.963e-02 on the sloped fixture, 0.0 on the flat one.

(1) is the one worth reading twice: both legs share the wrong expression, so gate 2 - the sibling
gate proper - stays GREEN. Only the gates that compare against a route which does not go through
the helper at all die. A consistency gate between two spellings cannot catch a defect they agree
on; it needs a third route that was never a sibling.

(5) and (6) are the same lesson at the seam below, and they are why gate 4 exists as its own test
rather than as a tolerance on gate 3: gate 3 stays GREEN under both, because a FLAT fixture makes
the wrong strip equal to the right one to within a few ULPs. (6) is caught by gate 2 as well only
because it desynchronises the two legs - mutate BOTH call sites, as (5) does, and every flat gate
in this file goes green on a pricer whose simulated E[S_T] is not F(t,T).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import pricing, utils
from derivus.config import Config
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
SPOT, STRIKE = 100.0, 100.0
FLAT_R = [[0.0, 0.05], [5.0, 0.05]]
FLAT_Q = [[0.01, 0.01], [5.0, 0.01]]
# a carry curve with real shape: the zero rate runs 0.5% -> 9% and the dividend 4% -> 0.5%, so the
# average rate to a fixing is nowhere near the forward rate over the interval preceding it
SLOPED_R = [[0.0, 0.005], [0.5, 0.03], [1.0, 0.07], [5.0, 0.09]]
SLOPED_Q = [[0.01, 0.04], [0.5, 0.02], [1.0, 0.005], [5.0, 0.005]]


def _cfg(barrier_type, barrier, rate=FLAT_R, div=FLAT_Q, drift=0.02):
    """Monthly-monitored equity barrier on a plain GBM scenario. Nothing here is tuned to the
    quantity under test: the gate reads the pricer's own intermediate forwards, so the deal only
    has to price."""
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': STRIKE, 'Expiry_Date': BASE + pd.Timedelta(days=365), 'Units': 100.0,
        'Barrier_Type': barrier_type, 'Barrier_Price': barrier, 'Cash_Rebate': 0.0,
        'Barrier_Dates': [BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=1),
    }
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], rate)},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None, 'Curve': utils.Curve([], div)},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, 0.25] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
    }
    c.params['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': 0.25, 'Drift': drift}}
    c.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceModel')
    c.params['Valuation Configuration'] = {}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(field, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _run_recording(monkeypatch, cfg, batch=8, sims=64):
    """Price the deal with both forward routes recorded.

    ``calc_moneyness`` receives the forward the SURFACE is read at; ``total_log_forward`` returns the
    forward the PAYOFF is valued at - once per block as ``[N_block, batch]`` from the already-hit
    leg, once per MTM row as ``[batch]`` from the in-out-parity leg. The rank is the discriminator,
    and the parity leg is called for every row in order, so its calls index the deal time grid."""
    surface, payoff = [], []
    real_cm, real_fwd = pricing.calc_moneyness, pricing.total_log_forward

    def spy_cm(strike, spot, forward, deal_data, use_forward=False, invert_moneyness=False):
        surface.append((spot.detach().clone(), forward.detach().clone()))
        return real_cm(strike, spot, forward, deal_data, use_forward, invert_moneyness)

    def spy_fwd(carry_rate, times):
        out = real_fwd(carry_rate, times)
        payoff.append(out.detach().clone())
        return out

    monkeypatch.setattr(pricing, 'calc_moneyness', spy_cm)
    monkeypatch.setattr(pricing, 'total_log_forward', spy_fwd)
    params = {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2d 1w(1w) 3m(1m)',
              'Batch_Size': batch, 'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD',
              'MCMC_Simulations': sims, 'Tenor_Offset': 0.0, 'Deflation_Interest_Rate': 'USD'}
    derivus.run_cmc(cfg, prec=DTYPE, overrides=params)
    spot, fwd = surface[0]
    # the STRIP's calls are rank 3 ([N_block, N_fix, batch]) and the preamble's expiry read rank 2,
    # so the two routes into `calc_moneyness` separate on shape alone
    strip = [(s, f) for s, f in surface if f.dim() == 3]
    return fwd / spot, payoff, strip       # surface growth exp(b*tau) [N_mtm, batch], payoff calls


def _aligned(rows, growth):
    """The two forwards on one shape. The carry curves are static, so the payoff route carries a
    batch of 1 and broadcasts; the surface route rides the simulated spot's batch."""
    return torch.exp(torch.stack(rows)).expand_as(growth), growth


def _disagreement(rows, growth):
    payoff, surface = _aligned(rows, growth)
    rel = (payoff / surface - 1.0).abs()
    return 'payoff forward vs surface forward: max rel %.3e at row %d (%.10f vs %.10f)' % (
        float(rel.max()), int(rel.max(dim=1).values.argmax()),
        float(payoff.flatten()[int(rel.argmax())]), float(surface.flatten()[int(rel.argmax())]))


def test_total_log_forward_is_rank_polymorphic():
    """A block of MTM rows and one row of that block must be the same expression.

    Pure algebra - no pricer, no market data. This is what makes ONE helper serviceable to a leg
    that values a whole block and a leg that values a single row, and an axis mistake in it would
    put the two legs back on different numbers while both still 'called the helper'."""
    g = torch.Generator().manual_seed(3)
    carry = torch.rand(5, 7, 4, generator=g, dtype=DTYPE) * 0.08 - 0.02
    times = torch.rand(5, 7, generator=g, dtype=DTYPE) * 0.3

    block = pricing.total_log_forward(carry, times)
    assert block.shape == (5, 4)
    for row in range(5):
        one = pricing.total_log_forward(carry[row], times[row])
        assert one.shape == (4,)
        assert torch.equal(block[row], one), (
            'row %d: block form %r vs row form %r' % (row, block[row].tolist(), one.tolist()))
    # and it really is the carry integral, not something that merely has the right shape
    assert torch.allclose(block, torch.einsum('bfn,bf->bn', carry, times), rtol=0, atol=1e-15)


def test_both_barrier_legs_read_one_forward(monkeypatch):
    """The already-hit KI leg and the in-out-parity leg must value the same forward, row by row.

    A ``Down_And_In`` whose barrier the GBM scenario actually crosses, so on every block after the
    first crossing BOTH legs are built: the already-hit leg once for the block, the parity leg once
    per row inside ``sim_spot_oss``. The already-hit leg is built first, so each ``[N_block, batch]``
    record opens a group and the ``[batch]`` records that follow are its rows.

    The two are equal by construction while both call ``total_log_forward`` - that is the point of
    the refactor, and this gate is what notices the day one of them is inlined again. It dies on any
    edit to either call site: dropping ``sample_ts`` at one of them, or passing the block's slice
    where the strip belongs."""
    _, payoff, _ = _run_recording(monkeypatch, _cfg('Down_And_In', 90.0))

    groups, current = [], None
    for rec in payoff:
        if rec.dim() == 2:
            current = (rec, [])
            groups.append(current)
        elif current is not None:
            current[1].append(rec)
    paired = [(blk, rows) for blk, rows in groups if rows]
    assert paired, 'no block built BOTH legs - the fixture never knocked in, gate is vacuous'
    assert sum(len(r) for _, r in paired) >= 8, 'too few paired rows to be a gate'

    for blk, rows in paired:
        assert len(rows) == blk.shape[0], (
            'block has %d rows but the parity leg was called %d times' % (blk.shape[0], len(rows)))
        for i, row in enumerate(rows):
            assert torch.equal(blk[i], row), (
                'row %d: already-hit leg log-forward %r vs parity leg %r'
                % (i, blk[i].tolist(), row.tolist()))


def test_payoff_forward_equals_the_forward_the_surface_is_read_at(monkeypatch):
    """The forward the payoff is valued at IS the forward the vol surface is read at.

    Two independent routes to one number. The surface's is ``spot * exp(b * tau)``: ONE gather of
    the carry curve at the deal's own tenor, built in ``pv_discrete_barrier_option``'s preamble and
    handed to ``calc_moneyness``. The payoff's is the carry gathered at EVERY fixing and integrated
    over the strip. Nothing forces them to agree except being the same financial quantity, and on a
    flat curve they agree to the last bit.

    NOT A PLACEBO: ``r = 5%`` against ``q = 1%`` puts the growth at 1.0083 on the first row, so the
    original defect - annualised rates summed with no ``dt``, plus a half-variance - lands 106.7%
    out and dies here. ``r = q`` would have made the forward exactly 1.0 and this gate a placebo,
    which is precisely the degeneracy every pre-existing HN barrier fixture had.

    The barrier sits at 40 against a spot of 100, so no scenario ever crosses it: the already-hit
    leg is never built and every one of the 37 rows is carried by the parity leg alone. The gate
    wants a forward, not a crossing - it is discriminating without any hit at all, which is exactly
    what a consistency gate buys over an oracle one."""
    growth, payoff, _ = _run_recording(monkeypatch, _cfg('Down_And_In', 40.0))
    rows = [r for r in payoff if r.dim() == 1]
    assert len(rows) == growth.shape[0], (
        'parity leg ran on %d rows, the deal grid has %d' % (len(rows), growth.shape[0]))
    assert float(growth.max()) > 1.005, 'flat-curve fixture has no carry - gate would be a placebo'
    assert torch.equal(*_aligned(rows, growth)), _disagreement(rows, growth)


def test_payoff_forward_survives_a_sloped_carry_curve(monkeypatch):
    """The same assertion on a SLOPED carry curve, which is where the strip's shape is observable.

    This was a strict xfail carrying a second, independent defect this file found and did not fix:
    ``sim_spot_oss`` built its per-interval carry as ``drifts * sample_ts``, multiplying the AVERAGE
    rate over ``[t, T_j]`` by the length of the interval ENDING at ``T_j``. Those are the same
    number for the first interval and on a flat curve, and every barrier fixture in this repo is
    flat, so nothing here could see it. ``pricing.forward_carry_rate`` now builds the strip by
    DIFFERENCING the cumulative integrals, which is what ``pv_MC_Tarf`` and ``pv_MC_AutoCallSwap``
    always did, and all four adopters read it from one expression.

    MEASURED on this fixture: the payoff forward ran 4.276e-02 below the reference before, and
    2.220e-16 after - fourteen orders, and four below the 1e-12 bar, which is why the tolerance
    cannot launder the defect it is written for. It is a tolerance rather than ``torch.equal``
    because the two routes genuinely do different float work once the curve has shape: one gathers
    the curve once at ``tau``, the other gathers at twelve fixings and telescopes.

    THE FORWARD IS NOT THE PRICE, and the strip drives the pricer's own simulation, so the number
    that settles it is a VALUE against a closed form: a never-knocking ``Down_And_Out`` call on
    this same sloped world is a European, and against Black at ``F = S*exp(0.065)``, ``DF =
    exp(-0.07)`` the pricer read **-20.10%** before and **+0.045%** after, at 65536 inner paths.
    That gate is not here because a Black oracle on a barrier fixture belongs beside the barrier's
    own suite; this file's job is the consistency statement, which needs no reference value.

    THE MUTATION THAT MUST KILL IT is the defect itself - passing ``drifts`` where ``fwd_drifts``
    goes at either barrier call site, or making ``forward_carry_rate`` return ``carry_rate``.
    Verified: 4.276e-02 against a 1e-12 bar."""
    growth, payoff, _ = _run_recording(
        monkeypatch, _cfg('Down_And_In', 40.0, rate=SLOPED_R, div=SLOPED_Q))
    rows = [r for r in payoff if r.dim() == 1]
    payoff_fwd, surface_fwd = _aligned(rows, growth)
    assert float((payoff_fwd / surface_fwd - 1.0).abs().max()) > 0.0, (
        'the sloped fixture reproduced the flat one exactly - the curve has no shape and the '
        'assertion below is the flat-curve gate a second time')
    assert torch.allclose(payoff_fwd, surface_fwd, rtol=1e-12, atol=0.0), _disagreement(rows, growth)


def test_the_vol_strips_last_forward_is_the_payoff_forward(monkeypatch):
    """A THIRD ROUTE, and the one the vol strip added. Every fixing of ``forward_vol_strip`` is
    read at its own forward; the LAST fixing is expiry (``instruments.py`` unions ``Expiry_Date``
    into the observation dates), so that forward must be the one the payoff is valued at.

    The routes really are different arithmetic. The strip builds ``S * exp(c_N * T_N)`` from the
    ZERO carry gathered at ``T_N`` times the cumulative tenor - one product. The payoff builds it
    by DIFFERENCING those cumulative integrals into an interval strip and summing the strip back
    up. They agree because differencing then telescoping is the identity, and nothing but that
    makes them agree: the vol surface is read at one and the option is paid at the other.

    WHY IT IS NOT THE SAME GATE AS (3) ABOVE. That one compares the payoff forward with the
    PREAMBLE's ``spot * exp(b * tau)``, a single gather at the deal's own tenor. This one compares
    it with the strip's, which rides the fixing schedule - so a strip built on the wrong tenors, or
    on the interval carry where the cumulative belongs, dies here and nowhere else. Measured: the
    strip handed the INTERVAL carry instead of the cumulative one reads 8.963e-02 against a 1e-13
    bar on the sloped fixture and passes clean on the flat one, where the two carries are the same
    number - which is why this runs on both and why a flat-only version would be a placebo.
    """
    for tag, kw in [('flat', {}), ('sloped', dict(rate=SLOPED_R, div=SLOPED_Q))]:
        growth, payoff, strip = _run_recording(monkeypatch, _cfg('Down_And_In', 40.0, **kw))
        assert strip, 'no vol strip was built - the pricer never read the surface per fixing'
        last = torch.cat([f[..., -1, :] / s[..., 0, :] for s, f in strip], dim=0)
        assert last.shape == growth.shape, (
            '%s: the strip covers %r rows and the deal grid has %r' % (
                tag, last.shape, growth.shape))
        rows = [r for r in payoff if r.dim() == 1]
        payoff_fwd, _ = _aligned(rows, growth)
        rel = (payoff_fwd / last - 1.0).abs().max()
        assert float(rel) < 1e-13, (
            '%s: the payoff forward and the vol strip\'s expiry forward disagree by %.3e' % (
                tag, float(rel)))


def test_the_interval_carry_strip_is_the_difference_of_cumulative_integrals():
    """``forward_carry_rate`` against the definition, with no pricer and no market data.

    ``carry_rate[j]`` is a ZERO rate at tenor ``T_j``, so the carry over the interval ending there
    is ``(c_j*T_j - c_{j-1}*T_{j-1}) / dt_j``, and its integral summed over the strip telescopes to
    ``c_N*T_N`` - ONE gather at expiry. Both statements are asserted, the second being what makes
    ``total_log_forward`` of this strip the forward the vol surface is read at.

    The first interval is the one place where the wrong expression is right (the cumulative window
    IS the interval), which is why a single-fixing fixture cannot see the defect; the flat-curve row
    is why no fixture in this repo could either. Both degeneracies are asserted so that a fixture
    built on either one is recognisable as a placebo."""
    g = torch.Generator().manual_seed(11)
    dt = torch.rand(4, 6, generator=g, dtype=DTYPE) * 0.3 + 0.05
    cum_t = dt.cumsum(dim=-1)
    carry = torch.rand(4, 6, 3, generator=g, dtype=DTYPE) * 0.08 - 0.02

    strip = pricing.forward_carry_rate(carry, cum_t, dt)
    integral = carry * cum_t.unsqueeze(-1)
    assert torch.equal(strip[..., 0, :], carry[..., 0, :])
    # rtol, not exact: the strip is a RATE, so re-multiplying by dt is a division round trip
    assert torch.allclose(strip[..., 1:, :] * dt[..., 1:].unsqueeze(-1),
                          integral.diff(dim=-2), rtol=1e-14, atol=0)
    # the sum telescopes to the last cumulative integral - the whole point of the shared strip
    assert torch.allclose(pricing.total_log_forward(strip, dt), integral[..., -1, :],
                          rtol=1e-14, atol=0)
    # and on a FLAT curve the defect's expression is exactly right, which is the blindness - but
    # NOT bitwise: differencing amplifies the rounding of a cumulative time by T_j/dt_j, so a flat
    # fixture moves in the last bits and cannot be gated with torch.equal. Measured 5.0 eps here
    # and 6.0 eps on the engine's monthly ACT/365 strip, against the bound below.
    flat = torch.full_like(carry, 0.037)
    rel = (pricing.forward_carry_rate(flat, cum_t, dt) / flat - 1.0).abs()
    bound = torch.finfo(DTYPE).eps * (cum_t / dt).max()
    assert float(rel.max()) <= float(bound), '%.3e against the T/dt bound %.3e' % (
        float(rel.max()), float(bound))
