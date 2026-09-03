"""BRANCH-EXECUTION CENSUS: which branch of which pricer does no test in the suite RUN?

Runs the whole suite under a branch tracer restricted to ``derivus/pricing.py``, maps every arc the
run never took onto the enclosing pricer, and asserts the result against the ``UNREACHED`` ledger
below in BOTH directions:

  * an arc the ledger does not name  -> a NEW unexecuted branch has appeared;
  * a ledger entry the run now takes -> a fixture reached it, delete the line (no stale alibis).

STANDALONE MEANS STANDALONE. Two imports used to stop this file running. The ledger lived in
``tests/test_pricer_branch_ledger.py``, which went with the 2026-08-21 purge; it is inlined here
and its cheap AST half is now ``--anchors`` (no torch, no fixtures, milliseconds). And the tracer
was ``coverage``, which is in no requirements file and is not installed on this machine - so the
measurement could not be taken at all. It is ``sys.monitoring`` here (PEP 669, stdlib): LINE says
which lines ran and BRANCH reports ``(code, source offset, destination offset)`` for every
conditional jump. Every callback returns ``DISABLE`` once it has nothing left to learn at that
instruction - immediately outside ``pricing.py``, and inside it once both destinations of a jump
have been seen - so the cost is one callback per instruction rather than one per execution, and a
traced test file runs at its untraced speed.

The number this prints is NOT comparable to the pre-purge 64/65 readings: the instrument changed,
and those two could not be reconciled with each other either.

WHAT A CENSUS IS FOR. No oracle finds code that does not run. ``pricing.py`` was at high line
coverage throughout the years the barrier leg was 1432% wrong; what was missing was one ARC -
``if all_hit:``, the row shape where that leg IS the reported PV. PERCENTAGES ARE NOT THE OUTPUT;
the output is the list of arcs, each carrying the fixture property that would reach it, so it reads
as a work-list rather than as a number.

WHY THE WHOLE SUITE. Any test may be the one that takes an arc, so a subset can only OVER-report,
and an over-reported census is a work-list with invented work in it. The run is therefore long
(~50 minutes on this repo), the fact it produces changes only when a pricer or a fixture changes,
and re-analysis of an existing data directory is free - which is why this is a gate and not a test.
It traces one test FILE at a time and resumes from what is already on disk, so an interrupted run is
not a lost one and adding one test file costs one file's trace; the data lands in a fixed directory
under the system temp, never in the repo.

    CUDA_VISIBLE_DEVICES=0 python gates/pricer_branch_census.py            # measure, then assert
    CUDA_VISIBLE_DEVICES=0 python gates/pricer_branch_census.py --data D   # re-analyse D, no run
    CUDA_VISIBLE_DEVICES=0 python gates/pricer_branch_census.py --emit     # print a fresh ledger
    python gates/pricer_branch_census.py --anchors                         # AST only, no run
    python gates/pricer_branch_census.py --trace P tests/test_x.py         # one file, used above

RUN IT ON A TREE THAT IS NOT MOVING. Arcs are line numbers, so an edit to ``pricing.py`` mid-run
re-attributes every arc below it; each part carries the source's hash (line endings normalised,
because a checkout and its own worktree differ by CRLF alone) and ``measure`` refuses a directory
whose parts disagree. Where another stream owns ``pricing.py``, take the census in a
``git worktree`` pinned at the commit being measured.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PRICING = os.path.join(ROOT, 'derivus', 'pricing.py')

#: The pricers the census covers: the three inner-MC pricers, plus the analytic barrier/option
#: family whose closed forms the barrier legs are cross-checked against. Widening it is free at
#: measurement time - the tracer already reads the whole file - but every pricer added arrives with
#: its own unreached arcs to write sentences for, so it is widened deliberately.
FAMILY = (
    'pv_discrete_barrier_option', 'pv_MC_Tarf', 'pv_MC_AutoCallSwap',
    'pv_barrier_option', 'pv_one_touch_option', 'pv_partial_barrier_option',
    'pv_american_option', 'pv_european_option',
    'getbarrierpayoff', 'getpartialbarrierpayoff', 'partial_window_rebate',
    'oss_truncated_draw',
)

#: `{(qualname, source text, direction): what a fixture would need to reach it}`, as measured over
#: the whole suite. Regenerate with `python gates/pricer_branch_census.py --emit`; every entry is a
#: work-list item, not an excuse.
#:
#: The shape of what is left, after the American arm and the eight closed-form barrier payoffs were
#: closed: no PUT partial barrier and no rebate on one, no quanto/compo, no inverted or put TARF, no
#: Heston-Nandi barrier on an exposure grid, and an autocall with no floating leg. Reading at
#: 1ed927a: 59 arcs.
UNREACHED = {
    # ---- getbarrierpayoff: the two completeness elses -------------------------------------------
    # The selector is (direction, eta, phi, strike vs H) and the four arms of each block partition
    # those eight combinations two apiece, so the trailing `else` of each chain is dead BY
    # CONSTRUCTION. `test_barrier_arms_json.py` prices all eight and never falls through either.
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or', 'else#1'):
        "a knock-IN barrier matching none of the four cases - unreachable if the enumeration is "
        "complete, which it is: the four conditions cover all eight (phi, eta, strike vs H).",
    ('getbarrierpayoff.barrier_option', 'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or', 'else#2'):
        "a knock-OUT barrier matching none of the four cases - as above, dead by construction.",

    # ---- getpartialbarrierpayoff: the PUT window barrier ----------------------------------------
    ('getpartialbarrierpayoff', 'if phi == -1:', 'body'):
        "a PUT FXPartialTimeBarrierOption. Every configuration in test_partial_barrier_json.py is a "
        "Call, so the put-call transformation that reflects the whole problem is never applied.",
    ('getpartialbarrierpayoff.BarrierPutCallTransformation', 'def BarrierPutCallTransformation', 'never-called'):
        "the same put deal - this closure exists only for the phi == -1 arm above.",
    ('getpartialbarrierpayoff.partial_barrier_option', 'if eta == 0:  # type B1', 'body'):
        "a window barrier with eta 0 - the Heynen-Kat type-B1 form. No declared Barrier_Type "
        "produces it (the selector sets eta from Down or Up and nothing else), which is why the "
        "roadmap could fix this arm's inverted strike selection without any fixture moving.",

    # ---- pv_american_option ---------------------------------------------------------------------
    ('pv_american_option', 'if (K > 0.0).all():', 'else'):
        "an American option struck at ZERO, where the Bjerksund-Stensland trigger degenerates to "
        "B_0. A completeness guard, not a live path, and the ONLY arc of this pricer that "
        "test_american_option_json.py does not take.",

    # ---- pv_barrier_option ----------------------------------------------------------------------
    ('pv_barrier_option', 'elif direction == BARRIER_OUT and expiry[index] == 0.0:', 'else'):
        "a knock-IN priced AT expiry with every scenario touched, so both the survival branch and "
        "the knock-out intrinsic branch are skipped and the payoff falls through to zero.",
    ('pv_barrier_option', 'if expiry_years_key not in factor_dep:', 'else'):
        "the same barrier priced twice with an identical expiry tuple, so the cached tenor tensor "
        "is hit rather than built. Benign - a memoisation, not a payoff.",
    ('pv_barrier_option', "if factor_dep.get('Check_Payoff_Type', False):", 'body'):
        "a quanto or compo barrier. Blocked rather than untested: the analytic consumers adjust the "
        "VOL only, so this is the half-adjusted compo the roadmap leaves open.",

    # ---- pv_one_touch_option --------------------------------------------------------------------
    ('pv_one_touch_option', "elif deal_data.Instrument.field['Payment_Timing'] == 'Touch':", 'else'):
        "a one-touch whose Payment_Timing is neither 'Expiry' nor 'Touch'. Refused at CONSTRUCTION "
        "now, so the chain can no longer be reached with a third value - dead by the refusal.",
    ('pv_one_touch_option', 'if rebate_part.any():', 'else#1'):
        "a touch-paid one-touch block in which NO scenario crossed during the interval, so nothing "
        "is cash-settled on that row.",
    ('pv_one_touch_option', 'if rebate_part.any():', 'else#2'):
        "the same at the EXPIRY row: a one-touch that expired with no scenario ever touching.",
    ('pv_one_touch_option', "if factor_dep.get('Check_Payoff_Type', False):", 'body'):
        "a quanto or compo one-touch - the same half-adjusted family as the barrier above.",
    ('pv_one_touch_option', 'if expiry_years_key not in factor_dep:', 'else'):
        "the tenor-cache hit, as in pv_barrier_option.",

    # ---- pv_partial_barrier_option: the rebate and discrete monitoring --------------------------
    ('pv_partial_barrier_option', "elif barrierType in ['Up_And_Out', 'Up_And_In']:", 'else'):
        "a Barrier_Type that is neither Down_* nor Up_*, leaving eta at 0 - the completeness gap "
        "the type-B1 arm above is the other half of.",
    ('pv_partial_barrier_option', "if factor_dep['Barrier_Monitoring']:", 'body'):
        "a partial barrier with a non-zero Barrier_Monitoring_Frequency, so the Broadie-Glasserman-"
        "Kou shift applies. Every fixture declares 0M continuous monitoring.",
    ('pv_partial_barrier_option', 'if cash_rebate:', 'body'):
        "a partial barrier with a non-zero Cash_Rebate - the census's own confirmation of the "
        "roadmap row: every gate here runs rebate 0.",
    ('pv_partial_barrier_option', 'for cash_index, cash in zip(deal_data.Time_dep.deal_time_grid[1:], rebate_part):', 'body'):
        "the same rebate, at the per-row settlement loop inside it.",
    ('pv_partial_barrier_option', 'for cash_index, cash in zip(deal_data.Time_dep.deal_time_grid[1:], rebate_part):', 'exit'):
        "the same loop completing - unreachable until a fixture carries a rebate at all.",

    # ---- pv_discrete_barrier_option: the OSS inner MC and the already-hit leg -------------------
    ('pv_discrete_barrier_option.sim_spot_oss', 'if isdigital:', 'body#2'):
        "an EquityBarrierBinaryOption reaching the in-out-parity vanilla leg of the OSS recursion.",
    ('pv_discrete_barrier_option.sim_spot_oss', 'if float(carry_total.detach().max() - carry_total.detach().min()) > 1.0e-9:', 'body'):
        "a Heston-Nandi barrier under a STOCHASTIC discount/dividend curve, so the carry varies "
        "across scenarios. This is the refusal that stops HN pricing with a batched carry and no "
        "test has ever made it fire - the class the boundary_weights 1e-30 threshold was in.",
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

    # ---- pv_MC_Tarf: the inverted target, the HN step, the crisp kink ---------------------------
    ('pv_MC_Tarf.accrued', 'if inverted:', 'body'):
        "a TARF with Invert_Target set - the reciprocal accrual. Every TARF fixture is "
        "non-inverted.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if not invertedTarget:', 'else#1'):
        "the same inverted TARF, on the PnL barrier B_pnl.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if not invertedTarget:', 'else#2'):
        "the same inverted TARF, on the second B_pnl site.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if not hn:', 'else#2'):
        "a Heston-Nandi TARF at the PER-STEP vol read. The block-level read a screen above takes "
        "both arms; this one has never seen `hn` true with a positive interval.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if fix is None:', 'else#2'):
        "a TARF step at an already OBSERVED fixing, on the advance rather than the draw - a "
        "reporting row whose block opens on a past fixing.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if not integrated:', 'body'):
        "a TARF carrying its OTM kink crisply rather than integrated - the non-integrated arm of "
        "the smoothing dial.",
    ('pv_MC_Tarf.sim_spot_tarf', 'if reduced_samples:', 'else'):
        "a TARF MTM row past its last fixing, where the block has no remaining samples to draw.",
    ('pv_MC_Tarf', 'if b_gaps:', 'else'):
        "a TARF priced with boundary_aad on that records no gap at all - a grid whose rows all sit "
        "past the last fixing.",

    # ---- pv_MC_AutoCallSwap: the floating leg, and the counterfactual on it ---------------------
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
    ('pv_MC_AutoCallSwap.sim_spot', 'if P_cf is not None:', 'body#3'):
        "the counterfactual on the SMOOTHED put-barrier breach - needs a put barrier and "
        "boundary_aad in the same run.",
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
    ('pv_MC_AutoCallSwap.sim_spot', 'if putBarrier > 0.0:', 'else'):
        "the strided coupon step on an autocall with NO put barrier, so no interval is held for the "
        "barrier block below it.",
    ('pv_MC_AutoCallSwap.sim_spot', 'if logging.getLogger().isEnabledFor(logging.DEBUG):', 'body'):
        "the AUTOCALL_SETTLE debug line. Nothing runs the suite at DEBUG; a diagnostic rather than "
        "a payoff, and the one entry here no fixture should be written for.",
    ('pv_MC_AutoCallSwap', "if 'Forward' in factor_dep:", 'body#1'):
        "an autocall with a floating leg, at the factor-resolution site.",
    ('pv_MC_AutoCallSwap', "if 'Forward' in factor_dep:", 'body#2'):
        "the same, at the forward-rate gather site.",
    ('pv_MC_AutoCallSwap', 'for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):', 'body'):
        "the same - this loop has no floating resets to walk.",
    ('pv_MC_AutoCallSwap', 'for offset, size, forward_rates in zip(*[reset_ofs, reset_count, forward_blocks]):', 'exit'):
        "the same loop completing at least one pass.",
    ('pv_MC_AutoCallSwap', 'if all_fixings[row, 0] != 0.0:', 'body'):
        "a latched autocall event on a row whose fixing is a PAST observation, so the loop skips it "
        "as a re-observation rather than a new decision - needs boundary_aad on a grid whose block "
        "re-observes a fixing it has already seen.",
    ('pv_MC_AutoCallSwap', "if boundary_aad and factor_dep['no_averaging']:", 'body#3'):
        "the all-resolved block's counterfactual: boundary_aad on, and a block reached after EVERY "
        "scenario has autocalled, so the rows carry a zero counterfactual.",
}

#: Arcs a fixture reaches TODAY and that must not stop being reached. Deliberately NOT "everything
#: currently covered" - that would be a second copy of the census, maintained by hand. These are
#: the arcs whose being unexecuted is precisely the state the barrier defect survived in: the
#: already-hit leg, the two selectors that gate it, the row shape it was measured wrong on, the
#: in-out-parity vanilla inside the OSS that it has to agree with, and the eight closed-form
#: payoff arms of `getbarrierpayoff` (four IN, four OUT), each of which is one wrong formula away
#: from a silently mispriced direction.
MUST_COVER = {
    ('pv_discrete_barrier_option', 'if some_hit or boundary_aad:', 'body'),
    ('pv_discrete_barrier_option', 'if direction == BARRIER_IN:', 'body'),
    ('pv_discrete_barrier_option', 'if direction == BARRIER_IN:', 'else'),
    ('pv_discrete_barrier_option', 'if all_hit:', 'body'),
    ('pv_discrete_barrier_option', 'if hn:', 'body#2'),
    ('pv_discrete_barrier_option', 'if hn:', 'else#2'),
    ('pv_discrete_barrier_option.sim_spot_oss', 'if direction == BARRIER_IN:', 'body'),
    ('pv_discrete_barrier_option.sim_spot_oss', 'if direction == BARRIER_IN:', 'else'),
    ('pv_barrier_option', 'if direction == BARRIER_IN:', 'body'),
    ('pv_partial_barrier_option', 'def pv_partial_barrier_option', 'never-called'),
    ('getpartialbarrierpayoff', 'def getpartialbarrierpayoff', 'never-called'),
    ('pv_american_option', 'def pv_american_option', 'never-called'),
    ('getbarrierpayoff.barrier_option', 'if direction == BARRIER_IN:', 'body'),
    ('getbarrierpayoff.barrier_option',
     'if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > H) or', 'body#1'),
    ('getbarrierpayoff.barrier_option',
     'elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= H) or', 'body#1'),
    ('getbarrierpayoff.barrier_option',
     'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > H) or', 'body#1'),
    ('getbarrierpayoff.barrier_option',
     'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or', 'body#1'),
    ('getbarrierpayoff.barrier_option',
     'if ((phi == OPTION_CALL and eta == BARRIER_UP and strike > H) or', 'body#2'),
    ('getbarrierpayoff.barrier_option',
     'elif ((phi == OPTION_CALL and eta == BARRIER_UP and strike <= H) or', 'body#2'),
    ('getbarrierpayoff.barrier_option',
     'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike > H) or', 'body#2'),
    ('getbarrierpayoff.barrier_option',
     'elif ((phi == OPTION_PUT and eta == BARRIER_UP and strike <= H) or', 'body#2'),
    ('pv_barrier_option', 'if cash_rebate and expiry[index] > 0.0:', 'body'),
}


def pricing_hash():
    """The source's identity for arc purposes: line endings normalised, because a checkout on this
    repo differs from its own worktree by CRLF alone and the arcs are line numbers."""
    return hashlib.md5(open(PRICING, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()


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


def escapable(loop):
    """Can this loop be left before its iterator is exhausted - a `break` of its own, or a `return`
    or `raise` anywhere under it?

    A loop that cannot is one `measure` must not report as never-completing: CPython attributes the
    exhaustion jump to the `for` line itself, so that arc is invisible to any line-keyed tracer, and
    a loop whose body ran and that has no way out has exhausted its iterator by construction."""
    for node in ast.walk(loop):
        if isinstance(node, (ast.Return, ast.Raise)):
            return True
        if isinstance(node, ast.Break) and not any(
                node in ast.walk(inner) for inner in ast.walk(loop)
                if inner is not loop and isinstance(inner, (ast.For, ast.While, ast.AsyncFor))):
            return True
    return False


def branch_sites():
    """`{line: [qualname, source text, body lines, alternative name, tag, header lines, kind,
    else-block lines, the test contains a BoolOp]}` for every branch site in the family.

    The direction a missed arc names is recovered from the TREE, not from the arc: the tracer
    reports `(source line, destination line)` and only the AST knows whether that destination opens
    the body. Anything else is the alternative - `else` for a conditional, `exit` for a loop, which
    is how "ran zero times" and "never finished" stay distinguishable. A ternary is recorded for
    anchoring only; `measure` never reports one.

    `tag` disambiguates a condition spelt the same way more than once in one function - the barrier
    payoff selector asks `if direction == BARRIER_IN:` in two blocks and `sim_spot_oss` tests
    `isBarrierDate_block[j] > 0` four times - by appending `#n` in source order. Without it a third
    of the ledger would be one key standing for several different questions.

    `header` is the line span of the branch's own TEST, which is how a short-circuit jump inside a
    two-line condition (the barrier payoff selector's `if (A and B) or (C and D):`) is told from the
    jump that actually leaves the branch: an arc whose destination is still in the header decided
    nothing. `kind` and the BoolOp flag carry the other two exclusions `measure` needs - a loop's
    back edge and an operand's fall-through both land on the header line too."""
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
        test = node.iter if isinstance(node, (ast.For, ast.AsyncFor)) else node.test
        header = set(range(min(node.lineno, test.lineno), (test.end_lineno or test.lineno) + 1))
        kind = ('ternary' if isinstance(node, ast.IfExp) else
                'loop' if isinstance(node, (ast.For, ast.While, ast.AsyncFor)) else 'if')
        alt = set() if kind == 'ternary' else {s.lineno for s in node.orelse}
        sites[node.lineno] = [name, lines[node.lineno - 1].strip(), body, other, '',
                              header, kind, alt,
                              any(isinstance(n, ast.BoolOp) for n in ast.walk(test)),
                              kind == 'loop' and escapable(node),
                              any(isinstance(n, ast.BoolOp) for n in ast.walk(test)) and
                              not any(isinstance(n, ast.Or) for n in ast.walk(test))]

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
    for site in branch_sites().values():
        name, text, _, other, tag = site[:5]
        keys |= {(name, text, 'body' + tag), (name, text, other + tag)}
    return keys


def check_anchors():
    """`[(label, [key])]` for the three failures the AST alone can see, no trace data needed.

    A sentence about a branch somebody deleted is worse than no sentence: it reads as a live known
    gap and it can never go green. The walk is pinned first because every check below passes
    trivially over an empty one."""
    named = {name for _, _, name in family_scopes() if '.' not in name}
    assert named == set(FAMILY), f'FAMILY names no pricer defines: {sorted(set(FAMILY) - named)}'
    known = anchors()
    assert len(known) > 300, f'the branch walk found only {len(known)} arcs - it stopped'
    return (
        ('ledger entries the AST cannot resolve - re-anchor them',
         sorted(k for k in UNREACHED if k not in known)),
        ('MUST_COVER entries the AST cannot resolve - re-anchor them',
         sorted(k for k in MUST_COVER if k not in known)),
        ('MUST_COVER entries the ledger also names - one of the two is wrong',
         sorted(MUST_COVER & set(UNREACHED))),
    )


def trace_one(part, target):
    """Run one pytest file under the BRANCH monitor and write `{lines, arcs, hash}` to `part`.

    The monitor is global and pays for itself with `DISABLE`: outside `pricing.py` every callback
    fires once per instruction and switches itself off, and inside it a jump switches off once both
    of its destinations have been seen. The data is written from `finally` rather than `atexit` so a
    pytest run that hard-fails still leaves a part behind.

    The identity of the imported module is checked FIRST. Arcs are matched by filename, so a repo
    running against an editable install of a DIFFERENT checkout would trace nothing and report the
    whole family as never-called - a silent zero, not an error."""
    import derivus.pricing
    assert os.path.normcase(os.path.abspath(derivus.pricing.__file__)) == \
        os.path.normcase(os.path.abspath(PRICING)), \
        f'derivus.pricing imported from {derivus.pricing.__file__}, not {PRICING}'
    mon = sys.monitoring
    tool = next(i for i in range(6) if mon.get_tool(i) is None)
    mon.use_tool_id(tool, 'derivus-census')
    want = os.path.normcase(os.path.abspath(PRICING))
    lines, arcs, both, seen, line_of = set(), set(), set(), {}, {}

    def line_map(code):
        got = line_of.get(code)
        if got is None:
            got = {}
            for start, end, num in code.co_lines():
                if num is not None:
                    for off in range(start, end, 2):
                        got[off] = num
            line_of[code] = got
        return got

    def on_line(code, number):
        if os.path.normcase(code.co_filename) != want:
            return mon.DISABLE
        lines.add(number)
        return mon.DISABLE                       # one hit is the whole question

    def on_branch(code, offset, destination):
        if os.path.normcase(code.co_filename) != want:
            return mon.DISABLE
        table = line_map(code)
        src, dst = table.get(offset), table.get(destination)
        if src is not None and dst is not None:
            arcs.add((src, dst))
        taken = seen.setdefault((code, offset), set())
        taken.add(destination)
        if len(taken) > 1:
            both.add(src)                        # this jump went BOTH ways at least once
            return mon.DISABLE
        return None

    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.register_callback(tool, mon.events.BRANCH, on_branch)
    mon.set_events(tool, mon.events.LINE | mon.events.BRANCH)
    try:
        import pytest
        pytest.main([target, '-q', '-p', 'no:cacheprovider'])
    finally:
        mon.set_events(tool, 0)
        mon.free_tool_id(tool)
        json.dump({'hash': pricing_hash(), 'lines': sorted(lines),
                   'arcs': sorted(arcs), 'both': sorted(both)}, open(part, 'w'))


def measure(data_dir):
    """`{(qualname, text, direction)}` for every family arc the run never took.

    A scope NO test calls is reported once, as `never-called`, and its interior arcs are dropped: a
    dead closure would otherwise arrive as dozens of findings that are all one fact. Its `def` line
    is excluded from the liveness test because that line executes when the closure is CREATED, which
    says nothing about it being called - the exact distinction the barrier leg died on.

    A ternary is anchored but never reported: `IfExp` arcs are not what the ledger's sentences are
    about, and reporting them would restate one expression as two work items.

    THE BODY IS READ OFF THE LINES, NOT THE ARCS. A branch's first body statement executes only
    when that branch is entered, so `min(body) in lines` is exact and free of the jump-encoding
    traps the alternative direction has to live with: `for x in y:` enters its body by falling
    through to the loop-variable unpack, which CPython attributes to the `for` line itself, so the
    entering arc reads `(L, L)` and no destination-based rule can see it. The alternative direction
    is exact too wherever an `else`/`elif` block exists - its first line is reached only by taking
    it - and falls back to the arcs only for a branch with no `else` at all, where the three
    exclusions below are what separate a decision from a short circuit and a loop's back edge. Where
    even that cannot decide - an `and` chain whose `if` is the last statement of a loop body, so its
    else target is the back edge on the condition's own line - `both` settles it: in an `and` chain
    every jump target IS the else, so a jump that went both ways proves the condition was false."""
    lines, arcs, both, hashes, traced = set(), set(), set(), set(), set()
    for path in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
        part = json.load(open(path))
        traced.add(os.path.basename(path)[:-5])
        hashes.add(part['hash'])
        lines |= set(part['lines'])
        arcs |= {tuple(a) for a in part['arcs']}
        both |= set(part.get('both', ()))
    # a test file with no part is a test file that was never run, and its arcs would be reported as
    # unreached - the over-report that makes a census a work-list with invented work in it
    missing = sorted({os.path.basename(p)[:-3] for p in
                      glob.glob(os.path.join(ROOT, 'tests', 'test_*.py'))} - traced)
    assert not missing, f'no trace for {len(missing)} test files: {missing}'
    assert len(hashes) == 1, f'the parts in {data_dir} traced {len(hashes)} different pricing.py'
    assert hashes == {pricing_hash()}, \
        'pricing.py has changed since the trace - the arcs no longer line up, re-run'

    dead = [(s, e, n) for s, e, n in family_scopes() if not any(s < ln <= e for ln in lines)]
    outermost = [d for d in dead if not any(s < d[0] and d[1] <= e for s, e, _ in dead)]
    found = {(n, f'def {n.split(".")[-1]}', 'never-called') for _, _, n in outermost}

    for line, site in branch_sites().items():
        name, text, body, other, tag, header, kind, alt, boolop, escapes, and_only = site
        if kind == 'ternary' or any(s < line <= e for s, e, _ in dead):
            continue
        entered = min(body) in lines
        if not entered:
            found.add((name, text, 'body' + tag))
        if alt:
            taken = min(alt) in lines
        elif kind == 'loop' and entered and not escapes:
            taken = True                                 # nothing can leave it early; see `escapable`
        elif and_only and both & header:
            taken = True                    # in an `and` chain every jump target IS the else
        else:
            out = {d for f, d in arcs if f in header} - body
            out -= header - {line} if len(header) > 1 else set()   # a short circuit decides nothing
            out -= header if kind == 'loop' or boolop else set()   # a back edge, an operand
            taken = bool(out)
        if not taken:
            found.add((name, text, other + tag))
    return found


def run_suite(data_dir):
    """Trace the suite ONE TEST FILE AT A TIME into `data_dir`, one JSON part each.

    One traced `pytest tests/` would be simpler - but a run killed at 96% would produce nothing at
    all. Per file, an interruption costs one file, a re-run resumes from what is already on disk,
    and the per-file parts answer "which test reaches this arc", which no combined run can.

    The tree is hashed into every part: arcs are line numbers resolved against the source AFTER the
    run, so an edit landing mid-run reattributes every arc below it to the wrong branch, and
    `measure` refuses a directory whose parts disagree."""
    os.makedirs(data_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(ROOT, 'tests', 'test_*.py'))):
        name = os.path.basename(path)[:-3]
        part = os.path.join(data_dir, f'{name}.json')
        if os.path.exists(part):
            continue
        print(f'  tracing {name}', flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), '--trace', part,
                        f'tests/{name}.py'], cwd=ROOT, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='an existing trace directory; skips the suite run')
    ap.add_argument('--dir', help='trace INTO this directory instead of the default temp one')
    ap.add_argument('--emit', action='store_true', help='print a fresh UNREACHED ledger and stop')
    ap.add_argument('--anchors', action='store_true', help='AST checks only; no suite, no tracer')
    ap.add_argument('--trace', metavar='PART',
                    help='trace ONE test file into PART - how the suite loop invokes itself')
    ap.add_argument('target', nargs='?', help='the test file --trace runs')
    args = ap.parse_args()

    if args.trace:
        trace_one(args.trace, args.target)
        return 0

    if args.anchors:
        failures = check_anchors()
        for label, rows in failures:
            if rows:
                print(f'\nFAIL: {label}:')
                for key in rows:
                    print(f'  {key}')
        if not any(rows for _, rows in failures):
            print(f'{len(UNREACHED)} ledger and {len(MUST_COVER)} MUST_COVER entries all anchor')
        return 1 if any(rows for _, rows in failures) else 0

    # stable so an interrupted run resumes, and OUTSIDE the repo because trace data is regenerable
    # and has no business being committable
    data_dir = args.data or args.dir or os.path.join(tempfile.gettempdir(), 'derivus_census')
    if not args.data:
        run_suite(data_dir)
    found = measure(data_dir)

    if args.emit:
        print('UNREACHED = {')
        for key in sorted(found):
            print(f'    {key!r}:\n        "",')
        print('}')
        return 0

    print(f'\n{len(found)} unexecuted branch arcs across {len(FAMILY)} pricers; '
          f'the ledger names {len(UNREACHED)}\n')
    for key in sorted(found):
        print(f'  {key[0]}\n      {key[2]:<12s} {key[1][:96]}'
              f'\n      -> {UNREACHED.get(key, "*** NOT ON THE LEDGER ***")}')

    failures = check_anchors() + (
        ('unexecuted branches the ledger does not name', sorted(k for k in found
                                                                if k not in UNREACHED)),
        ('ledger entries a test now reaches - delete them', sorted(k for k in UNREACHED
                                                                   if k not in found)),
        ('MUST_COVER branches that stopped being executed', sorted(MUST_COVER & found)),
    )
    for label, rows in failures:
        if rows:
            print(f'\nFAIL: {label}:')
            for key in rows:
                print(f'  {key}')
    return 1 if any(rows for _, rows in failures) else 0


if __name__ == '__main__':
    sys.exit(main())
