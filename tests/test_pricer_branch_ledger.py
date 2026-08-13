"""THE LEDGER of pricer branches no test EXECUTES, held to the pricers by AST.

The defect this exists for was not mis-asserted, it was never RUN. The already-hit KI leg of
``pv_discrete_barrier_option`` valued its vanilla off a log-forward that summed annualised RATES
with no ``dt`` and added a half-variance whose cancelling subtraction lives on the OTHER branch it
was copied from. Measured on a fixture with ``r != q`` and a long observation strip it reported
+1432% of the leg's true value, on every ``all_hit`` row, on every BARRIER_IN deal in every
exposure/CVA/PFE run - and the whole suite was green, because three fixture properties each hid it
INDEPENDENTLY:

  1. every barrier gate priced under BASE VALUATION - one deal-time row, so the hit mask is
     all-False at row 0 and ``some_hit`` is never true (the leg is still built when
     ``boundary_aad`` is on, but only as a counterfactual nothing reports);
  2. the one barrier that ran an exposure grid was ``Down_And_Out``, where the leg is the
     model-free zeros branch;
  3. every fixture set ``r = q = 0``, which zeroes the missing-``dt`` half of the error.

WHAT THIS WOULD ACTUALLY HAVE CAUGHT, measured rather than asserted. The tree at ``a87e3b5^`` was
exported and its sixteen barrier-touching test files were re-run under the tracer. The result is a
NEAR MISS and it is worth stating precisely, because the comfortable version of this file's claim is
false:

  * the defective expression itself WAS executed - ``test_boundary_pricer_events`` reached it, via
    the ``boundary_aad`` disjunct of ``if some_hit or boundary_aad:``, which builds the leg as a
    counterfactual whose value is then discarded. Branch execution would NOT have flagged it. A
    branch can be run by a test that never looks at what it produced.
  * ``if all_hit:`` - one screen below, the row shape where that leg IS the entire reported PV and
    where the error measures +1432% - was never executed by anything. THAT is the line the census
    would have printed, and the fixture it asks for (a knock-IN on a multi-row grid where every
    scenario has already crossed) is exactly the fixture that exposes the defect.

So the census does not find wrong arithmetic; it finds the fixture that was never written. Here that
was the same fixture. It is kept honest by ``MUST_COVER`` below, which pins ``if all_hit:``.

No oracle finds code that does not run. A census prints the line.

WHAT THIS FILE IS. ``UNREACHED`` is the committed list of every branch arc in ``FAMILY`` that the
full suite never takes, each carrying the fixture property that would reach it - so it reads as a
work-list rather than as a number. It is MEASURED by ``gates/pricer_branch_census.py`` (a full
suite run under ``coverage --branch``, minutes, deliberate) and this file is the cheap half that
runs in the suite in milliseconds, with no coverage, no torch and no fixtures:

  * every ``UNREACHED`` key still resolves to a branch that EXISTS - which is what stops the ledger
    decaying into sentences about code that was edited away years ago, the failure mode that makes
    a checklist worse than useless;
  * no ``MUST_COVER`` branch is on it. That set is the policy: these arcs are load-bearing and a
    fixture reaches them TODAY, so the day a refactor stops executing one, the census run must add
    it to ``UNREACHED`` and adding it turns this file red. Coverage cannot be dropped quietly.

Keys are ``(qualname, source text, direction)`` and never line numbers, so editing anything above a
branch does not churn the ledger; editing the branch's own condition does, deliberately.
``direction`` is ``body`` (never entered), ``else`` (the alternative never taken), ``exit`` (a loop
that never completed) or ``never-called`` (a whole function or closure nothing invokes - reported
once, rather than as one finding per arc inside it), with ``#n`` appended when the same condition is
spelt more than once in the same function.

WHY NOT IN THE SUITE. The measurement needs the WHOLE suite traced, because any test may be the one
that takes an arc, and that is a multi-minute run for a fact that changes only when a pricer or a
fixture changes. Splitting it this way is the honest trade: the expensive half is run on purpose,
the half that ROTS is checked every time anybody runs the tests.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

PRICING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'derivus', 'pricing.py')

#: The pricers the census covers: the three inner-MC pricers, plus the analytic barrier/option
#: family whose closed forms the barrier legs are cross-checked against. Widening it is free at
#: measurement time - the tracer already reads the whole file - but every pricer added arrives with
#: its own unreached arcs to write sentences for, so it is widened deliberately.
FAMILY = (
    'pv_discrete_barrier_option', 'pv_MC_Tarf', 'pv_MC_AutoCallSwap',
    'pv_barrier_option', 'pv_one_touch_option', 'pv_partial_barrier_option',
    'pv_american_option', 'pv_european_option',
    'getbarrierpayoff', 'getpartialbarrierpayoff',
)

#: `{(qualname, source text, direction): what a fixture would need to reach it}`, as measured over
#: all 70 test files. Regenerate with `python gates/pricer_branch_census.py --emit`; every entry is
#: a work-list item, not an excuse.
#:
#: The shape of it: the suite prices ONE barrier direction (`Down_And_Out`, plus an `Up_And_In` that
#: only ever reaches the discrete pricer), ONE option type (`Call`), no rebate, no quanto/compo, no
#: American, no digital European, no inverted or put TARF, and an autocall with no floating leg.
#: Seven of the eight closed-form barrier payoffs and the whole analytic knock-IN leg are code the
#: suite has never run.
UNREACHED = {
    # ---- pricers and closures NOTHING in the suite calls ----------------------------------------
    ('pv_partial_barrier_option', 'def pv_partial_barrier_option', 'never-called'):
        "an FXPartialTimeBarrierOption deal. No fixture builds one, so neither the pricer nor its "
        "payoff closure has ever been executed by a test.",
    ('getpartialbarrierpayoff', 'def getpartialbarrierpayoff', 'never-called'):
        "the same deal - this is pv_partial_barrier_option's payoff, dead for the same reason.",
    ('pv_american_option', 'def pv_american_option', 'never-called'):
        "an EquityOptionDeal with Option_Style != 'European'. Every equity option fixture is "
        "European, so the Barone-Adesi-Whaley branch of EquityOptionDeal is never priced.",
    ('pv_MC_Tarf.bs_call_put_fwd', 'def bs_call_put_fwd', 'never-called'):
        "a TARF row whose remaining strip is valued by the closed form rather than simulated - "
        "defined in the pricer and called from nowhere in it. Read it as a dead helper first and a "
        "coverage gap second.",

    # ---- getbarrierpayoff: 7 of the 8 closed-form payoffs -----------------------------------
    # The selector is (direction, eta, phi, strike vs H). Every fixture that reaches it is
    # Down_And_Out / Call / strike > H, which is the LAST elif of the OUT block, so the three
    # tests above it are evaluated-and-false and everything under BARRIER_IN is unexecuted.
    ('getbarrierpayoff.barrier_option', 'if direction == BARRIER_IN:', 'body'):
        "a knock-IN barrier that routes to pv_barrier_option - i.e. an FX/Equity BarrierOption with "
        "Barrier_Type '*_And_In' and no discrete-monitoring override. The four knock-in formulas "
        "(A+E, B-C+D+E, A-B+D+E, C+E) have never been evaluated.",
    ('getbarrierpayoff.barrier_option', 'if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > H) or', 'body#1'):
        "as above, plus Option_Type 'Call' with an Up barrier and Strike_Price above it (or a Put "
        "with a Down barrier at or below the strike) - the A+E knock-in payoff.",
    ('getbarrierpayoff.barrier_option', 'if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > H) or', 'else#1'):
        "any knock-IN barrier at all; the else is the rest of the knock-in chain.",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= H) or', 'body#1'):
        "a knock-IN Call with an Up barrier at or below the strike (or Put/Down above it) - B-C+D+E.",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= H) or', 'else#1'):
        "any knock-IN barrier that is not the B-C+D+E case.",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > H) or', 'body#1'):
        "a knock-IN Put with an Up barrier below the strike (or Call/Down at or above it) - A-B+D+E.",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > H) or', 'else#1'):
        "any knock-IN barrier that is not the A-B+D+E case.",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or', 'body#1'):
        "a knock-IN Put with an Up barrier at or above the strike (or Call/Down below it) - C+E. "
        "Note this chain has no else, so an unmatched knock-in returns None and the deal dies "
        "downstream rather than here.",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or', 'else#1'):
        "a knock-IN barrier matching none of the four cases - unreachable if the enumeration is "
        "complete, which is itself worth asserting rather than assuming.",
    ('getbarrierpayoff.barrier_option', 'if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > H) or', 'body#2'):
        "a knock-OUT Call with an Up barrier above the strike (or Put/Down at or below it) - the "
        "rebate-only F payoff, where the option itself is worthless on survival.",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= H) or', 'body#2'):
        "an Up_And_Out Call struck at or above the barrier (or a Down_And_Out Put below it) - "
        "A-B+C-D+F.",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > H) or', 'body#2'):
        "an Up_And_Out Put struck above the barrier (or a Down_And_Out Call at or below it) - B-D+F.",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or', 'else#2'):
        "a knock-OUT barrier matching none of the four cases - as above, an unasserted completeness "
        "claim rather than a live path.",

    # ---- pv_discrete_barrier_option: the OSS inner MC and the already-hit leg --------------------
    ('pv_discrete_barrier_option.sim_spot_oss', 'if isdigital:', 'body#1'):
        "an EquityBarrierBinaryOption whose LAST step is integrated rather than sampled - i.e. a "
        "digital reaching the terminal fixing under GBM.",
    ('pv_discrete_barrier_option.sim_spot_oss', 'if isdigital:', 'body#2'):
        "the same digital, on the in-out-parity vanilla leg.",
    ('pv_discrete_barrier_option.sim_spot_oss', 'if float(carry_total.detach().max() - carry_total.detach().min()) > 1.0e-9:', 'body'):
        "a Heston-Nandi barrier under a STOCHASTIC discount/dividend curve, so the carry varies "
        "across scenarios. This is the refusal that stops HN pricing with a batched carry, and no "
        "test has ever made it fire - the guard is unexercised, exactly the class the "
        "boundary_weights 1e-30 threshold was in.",
    ('pv_discrete_barrier_option.sim_spot_oss', 'if isBarrierDate_block[j] > 0:', 'body#2'):
        "a Heston-Nandi discrete barrier whose block contains a barrier observation date: the HN "
        "sub-stepping arm of the OSS recursion. Every HN barrier fixture prices under base "
        "valuation, so the block never spans one.",
    ('pv_discrete_barrier_option.sim_spot_oss', 'if eta == BARRIER_UP:', 'body#2'):
        "the same HN arm with an Up barrier.",
    ('pv_discrete_barrier_option.sim_spot_oss', 'if eta == BARRIER_UP:', 'else#2'):
        "the same HN arm with a Down barrier.",
    ('pv_discrete_barrier_option.sim_spot_oss', 'if direction == BARRIER_OUT:', 'body#2'):
        "the same HN arm on a knock-OUT.",
    ('pv_discrete_barrier_option.sim_spot_oss', 'if direction == BARRIER_OUT:', 'else#2'):
        "the same HN arm on a knock-IN.",
    ('pv_discrete_barrier_option.sim_spot_oss', 'if isBarrierDate_block[j] > 0:', 'body#3'):
        "an HN block whose FINAL fixing is a barrier date - the terminal-step branch of the same "
        "recursion.",
    ('pv_discrete_barrier_option', 'if carry_spread > 1.0e-9:', 'body'):
        "an already-knocked-in HN barrier under a stochastic carry. The sibling of the "
        "sim_spot_oss refusal above, and equally unexercised.",
    ('pv_discrete_barrier_option', 'if isdigital:', 'body#1'):
        "an already-knocked-in HN digital barrier - the hn_cdf_logret leg of the hit value.",
    ('pv_discrete_barrier_option', 'if isdigital:', 'body#2'):
        "an already-knocked-in GBM digital barrier. This is the cash-payoff twin of the leg that "
        "carried the +1432% forward defect, and it is still not executed by anything.",

    # ---- pv_barrier_option: the analytic barrier ------------------------------------------------
    ('pv_barrier_option', 'if direction == BARRIER_IN:', 'body'):
        "a knock-IN routed to the analytic pricer. THE SIBLING OF THE DEFECT: this leg values the "
        "same European by in-out parity, with its own `expiry[index]` ternary and its own rebate "
        "rule, and no test executes a line of it.",
    ('pv_barrier_option', 'if cash_rebate and expiry[index] > 0.0:', 'body'):
        "a knock-OUT barrier with a non-zero Cash_Rebate on an exposure grid. Every barrier fixture "
        "sets Cash_Rebate 0.0, so the rebate is never settled and the double-count guard beside it "
        "is never tested.",
    ('pv_barrier_option', 'elif direction == BARRIER_OUT and expiry[index] == 0.0:', 'else'):
        "a knock-IN priced AT expiry with every scenario touched, so both the survival branch and "
        "the knock-out intrinsic branch are skipped and the payoff falls through to zero.",
    ('pv_barrier_option', "if factor_dep.get('Check_Payoff_Type', False):", 'body'):
        "a quanto or compo barrier. Blocked, not merely untested: calc_vol_adjustment returns a "
        "python float for b_adj under plain GBM, so the deal raises and is skipped (the pinned "
        "strict xfail). Reaching this needs the Compo defect fixed first.",
    ('pv_barrier_option', 'if expiry_years_key not in factor_dep:', 'else'):
        "the same barrier priced twice with an identical expiry tuple, so the cached tenor tensor "
        "is hit rather than built. Benign - it is a memoisation, not a payoff - but it is a branch "
        "no test takes.",

    # ---- pv_one_touch_option ---------------------------------------------------------------
    ('pv_one_touch_option', "elif deal_data.Instrument.field['Payment_Timing'] == 'Touch':", 'else'):
        "a one-touch whose Payment_Timing is neither 'Expiry' nor 'Touch'. The chain has no else, "
        "so an unknown timing silently prices as whatever the last assignment left - worth an "
        "explicit refusal rather than a fixture.",
    ('pv_one_touch_option', 'if rebate_part.any():', 'else#2'):
        "a one-touch block where no scenario touched in the interval, so nothing is cash-settled.",
    ('pv_one_touch_option', "if factor_dep.get('Check_Payoff_Type', False):", 'body'):
        "a quanto or compo one-touch - blocked by the same Compo defect as the barrier above.",
    ('pv_one_touch_option', 'if expiry_years_key not in factor_dep:', 'else'):
        "the tenor-cache hit, as in pv_barrier_option.",

    # ---- pv_european_option ----------------------------------------------------------------
    ('pv_european_option', 'if binary:', 'body'):
        "an EquityBinaryOption or FXBinaryOption. Both classes exist and route here with "
        "binary=True, and no fixture builds either, so the cash-or-nothing payoff is unpriced.",

    # ---- pv_MC_Tarf ------------------------------------------------------------------------
    ('pv_MC_Tarf.accrued', 'if inverted:', 'body'):
        "a TARF with Invert_Target set - the reciprocal accrual. Every TARF fixture is "
        "non-inverted.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if not invertedTarget:', 'else#1'):
        "the same inverted TARF, on the PnL barrier B_pnl.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if not invertedTarget:', 'else#2'):
        "the same inverted TARF, on the second B_pnl site.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if (callOrPut > 0):', 'else'):
        "a PUT TARF. The survival truncation flips side (Z >= z_max), and every TARF fixture is a "
        "Call, so half of the one-step-survival draw is unexecuted.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if reduced_samples:', 'else'):
        "a TARF MTM row past its last fixing, where the block has no remaining samples to draw.",
    ('pv_MC_Tarf', 'if sample_val:', 'body'):
        "a TARF whose Reset history carries a non-zero PAST fixing, so the accumulated target is "
        "seeded from history rather than from zero.",
    ('pv_MC_Tarf', 'if b_gaps:', 'else'):
        "a TARF priced with boundary_aad on that records no gap at all - a grid whose rows all sit "
        "past the last fixing.",

    # ---- pv_MC_AutoCallSwap ----------------------------------------------------------------
    ('pv_MC_AutoCallSwap.sim_autocall', 'if isBarrierDate[t] > 0.0:', 'body'):
        "an autocall with a put barrier (Barrier_Dates set), so breachEvent is computed. The "
        "textbook-path simulator's barrier arm is unexecuted.",
    ('pv_MC_AutoCallSwap.sim_autocall', 'if isFixingDate[t] > 0.0:', 'else'):
        "an autocall step that is NOT an averaging fixing date.",
    ('pv_MC_AutoCallSwap.sim_autocall', 'if isFloatDate[t] > 0.0:', 'body'):
        "an autocall with a FLOATING leg. No autocall fixture has one, which is the same gap as "
        "the two `if 'Forward' in factor_dep:` entries below.",
    ('pv_MC_AutoCallSwap.sim_autocall', 'if coupon[t] <= 0.0:', 'body'):
        "a float date that is not also a coupon date.",
    ('pv_MC_AutoCallSwap.sim_autocall', 'if coupon[t] <= 0.0:', 'else'):
        "a float date that IS a coupon date.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if FloatingDate > 0:', 'body'):
        "the floating leg again, this time in the OSS simulator that actually prices the deal.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if P_cf is not None:', 'body#1'):
        "the boundary-AAD counterfactual accumulator on a floating date - needs both a floating leg "
        "and boundary_aad on.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if P_cf is not None:', 'else#1'):
        "a floating date with the counterfactual off.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if coup > 0:', 'else'):
        "an autocall observation step that carries no coupon.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if tau == 0.0:', 'else#1'):
        "a zero-length coupon interval on a row that is NOT the deal's own valuation date, so the "
        "termination latch is left alone.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if fixing_aligned:', 'else'):
        "an autocall block that starts from a PAST fixing rather than the scenario spot - i.e. a "
        "reported row between two coupon observations.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if dt > 0:', 'else#2'):
        "the same past-fixing block, where dt is forced to zero and p becomes the hard indicator.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if last_fixing is None:', 'else'):
        "the same: a block whose spot comes from past_fixings.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if reduced_samples:', 'else'):
        "an autocall MTM row with no remaining coupon observations.",
    ('pv_MC_AutoCallSwap', "if 'Forward' in factor_dep:", 'body#1'):
        "an autocall with a floating leg, at the factor-resolution site.",
    ('pv_MC_AutoCallSwap', "if 'Forward' in factor_dep:", 'body#2'):
        "the same, at the forward-rate gather site.",
    ('pv_MC_AutoCallSwap', 'for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):', 'body'):
        "the same - this loop has no floating resets to walk.",
    ('pv_MC_AutoCallSwap', 'for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):', 'exit'):
        "the same loop completing at least one pass.",
    ('pv_MC_AutoCallSwap', 'if terminationDate.any():', 'else'):
        "an autocall where NO scenario triggers - a threshold high enough that the deal always "
        "runs to maturity.",
}

#: Arcs a fixture reaches TODAY and that must not stop being reached. Deliberately NOT "everything
#: currently covered" - that would be a second copy of the census, maintained by hand. These are
#: the arcs whose being unexecuted is precisely the state the barrier defect survived in: the
#: already-hit leg, the two selectors that gate it, the row shape it was measured wrong on, and the
#: in-out-parity vanilla inside the OSS that it has to agree with.
MUST_COVER = {
    ('pv_discrete_barrier_option', 'if some_hit or boundary_aad:', 'body'),
    ('pv_discrete_barrier_option', 'if direction == BARRIER_IN:', 'body'),
    ('pv_discrete_barrier_option', 'if direction == BARRIER_IN:', 'else'),
    ('pv_discrete_barrier_option', 'if all_hit:', 'body'),
    ('pv_discrete_barrier_option', 'if hn:', 'body#2'),
    ('pv_discrete_barrier_option', 'if hn:', 'else#2'),
    ('pv_discrete_barrier_option.sim_spot_oss', 'if direction == BARRIER_IN:', 'body'),
    ('pv_discrete_barrier_option.sim_spot_oss', 'if direction == BARRIER_IN:', 'else'),
}


def family_scopes():
    """`[(start, end, qualname)]` for every def inside a `FAMILY` pricer, the pricer itself
    included.

    Nested defs carry their parent in the qualname because that is where the interesting arcs live
    - `sim_spot_oss`, the payoff closures - and a bare `barrier_option` would name two different
    functions in two different pricers."""
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f'{prefix}.{child.name}' if prefix else child.name
                if prefix or child.name in FAMILY:
                    out.append((child.lineno, child.end_lineno, name))
                    walk(child, name)
                else:
                    walk(child, '')
            else:
                walk(child, prefix)
    walk(ast.parse(open(PRICING).read()), '')
    return out


def branch_sites():
    """`{line: (qualname, source text, body destination lines, alternative direction, tag)}` for
    every branch site in the family.

    The direction a missed arc names is recovered from the TREE, not from the arc: coverage reports
    `(test line, destination line)` and only the AST knows whether that destination opens the body.
    Anything else is the alternative - `else` for a conditional, `exit` for a loop, which is how
    "ran zero times" and "never finished" stay distinguishable. A ternary is recorded for anchoring
    only: coverage does not arc the two arms of an `IfExp`.

    `tag` disambiguates a condition spelt the same way more than once in one function - the barrier
    payoff selector asks `if direction == BARRIER_IN:` in two blocks and `sim_spot_oss` tests
    `isBarrierDate_block[j] > 0` four times - by appending `#n` in source order. Without it a third
    of this file's ledger would be one key standing for several different questions."""
    lines = open(PRICING).read().splitlines()
    scopes = family_scopes()
    sites = {}

    def owner(line):
        best = None
        for start, end, name in scopes:
            if start <= line <= end and (best is None or start >= best[0]):
                best = (start, end, name)
        return best[2] if best else None

    for node in ast.walk(ast.parse('\n'.join(lines))):
        if not isinstance(node, (ast.If, ast.IfExp, ast.For, ast.While, ast.AsyncFor)):
            continue
        name = owner(node.lineno)
        if name is None or (isinstance(node, ast.IfExp) and node.lineno in sites):
            continue
        body = set() if isinstance(node, ast.IfExp) else {s.lineno for s in node.body}
        other = 'else' if isinstance(node, (ast.If, ast.IfExp)) else 'exit'
        sites[node.lineno] = [name, lines[node.lineno - 1].strip(), body, other, '']

    counts = {}
    for line in sorted(sites):
        site = sites[line]
        pair = (site[0], site[1])
        counts[pair] = counts.get(pair, 0) + 1
        site[4] = f'#{counts[pair]}'
    for site in sites.values():
        if counts[(site[0], site[1])] == 1:
            site[4] = ''
    return sites


def anchors():
    """Every ledger key the CURRENT source could legitimately produce, from the AST alone."""
    keys = {(name, f'def {name.split(".")[-1]}', 'never-called') for _, _, name in family_scopes()}
    for name, text, _, other, tag in branch_sites().values():
        keys |= {(name, text, 'body' + tag), (name, text, other + tag)}
    return keys


ANCHORS = anchors()


def test_the_anchor_is_not_vacuous():
    """Every assertion below passes trivially over an empty walk, so the walk is pinned first: a
    renamed pricer or a broken AST descent would otherwise turn this file green rather than red."""
    named = {name for _, _, name in family_scopes() if '.' not in name}
    assert named == set(FAMILY), f'FAMILY names no pricer defines: {sorted(set(FAMILY) - named)}'
    assert len(ANCHORS) > 300, f'the branch walk found only {len(ANCHORS)} arcs - it stopped'


@pytest.mark.parametrize('key', sorted(UNREACHED))
def test_a_ledger_entry_still_names_a_branch_that_exists(key):
    """A sentence about a branch somebody deleted is worse than no sentence: it reads as a live
    known gap and it can never go green. Re-anchor it or drop it."""
    assert key in ANCHORS, (
        f'{key[0]} has no {key[2]} branch spelt {key[1]!r} any more - re-run the census')


@pytest.mark.parametrize('key', sorted(MUST_COVER))
def test_a_must_cover_branch_is_still_executed(key):
    """The ratchet. These arcs are reached by a fixture today; the census can only put one on the
    ledger by measuring that it stopped being reached, and doing so fails here."""
    assert key in ANCHORS, f'{key[0]} has no {key[2]} branch spelt {key[1]!r} any more'
    assert key not in UNREACHED, (
        f'{key[0]} / {key[2]} branch {key[1]!r} is no longer executed by any test')
