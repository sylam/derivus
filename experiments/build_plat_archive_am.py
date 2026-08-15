"""Build the AM-fix-anchored platinum calibration archive, both sessions.

The model (fix-anchored, 2026-08): the LBMA AM fix is the primary state; the PM fix and the CME
futures curve hang off it,

    F_i = Fix * exp(c*tau_i + a*tau_i^2 + e)        i = 1, 2, 3   (exactly identified per day)

so each session's three front futures at the fix+3min snapshot solve the 3x3 Vandermonde for
(c, a, e) — the AM solve anchored on the AM fix, the PM solve on the PM fix. The archive rows are
framework factor names in generation order (the column header IS the contract), every value a
SAME-DAY observation:

    CommodityPrice.LBMA_AM              the AM fix (observed, never synthetic)
    ObservedBasis.LBMA_AM.PM            PM fix - AM fix  (composed CommodityPrice.LBMA_AM.PM
                                        IS the PM fixing)
    ObservedBasis.LBMA_AM.CME           fix_am*(exp(e_am)-1): AM-session CME basis, so
                                        (fix_am + b)*exp(z(tau)*tau) reproduces F_i(am) exactly
    ObservedBasis.LBMA_AM.CME.PM        fix_pm*(exp(e_pm)-1): PM-session CME basis in the same
                                        (own-fix-anchored) convention
    ForwardRate.PLATINUM_CARRY,0.5      z(0.5) = c + 0.5a   (AM solve; z(tau)*tau = c*tau +
    ForwardRate.PLATINUM_CARRY,1.0      z(1.0) = c + a       a*tau^2, total carry incl. financing)

Rows where a snapshot fails quality (crossed, wide, missing contract, tau <= 2d) leave that
session's solved column NaN; the fixes are always carried. Tenors use days/365.25 (the framework's
DAYS_IN_YEAR — the old riskflow extraction's /365.0 is corrected here).
"""
import logging
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'platinum_lbma_with_futures_basis.csv')
EXPIRY = os.path.join(ROOT, 'data', 'platinum_futures_contract_expiry.csv')
OUT = os.path.join(ROOT, 'data', 'plat_archive_am.csv')

OFFSET_LADDER = (3, 2, 4, 1, 5, 0)   # prefer the +3min anchor, fall back to the nearest minutes
MAX_SPREAD_BPS = 100.0               # junk guard only: a wide uncrossed mid is information, a
MIN_TAU = 2.0 / 365.25               # missing/crossed quote is not (30bp censored 15-23% of
                                     # post-2020 STRESS days - missing-not-at-random)
DAYS_IN_YEAR = 365.25
REF_TENORS = (0.5, 1.0)

FIX_COL = 'CommodityPrice.LBMA_AM'
PM_DIFF_COL = 'ObservedBasis.LBMA_AM.PM'
AM_CME_COL = 'ObservedBasis.LBMA_AM.CME'
PM_CME_COL = 'ObservedBasis.LBMA_AM.CME.PM'
CARRY_COL = 'ForwardRate.PLATINUM_CARRY'


def solve_session(df, expiry, session, anchor):
    """Solve (c, a, e) per day from the three front futures mids at the session's snapshot,
    anchored on that session's fix. Returns the three arrays, NaN where quality fails."""
    mids, taus = [], []
    ladder_used = np.zeros(len(OFFSET_LADDER), dtype=int)
    for i in (1, 2, 3):
        tau = df[f'PL{i}_{session}_CONTRACT'].map(expiry).sub(df['DATE']).dt.days / DAYS_IN_YEAR
        mid = pd.Series(np.nan, index=df.index)
        for rank, k in enumerate(OFFSET_LADDER):
            bid = pd.to_numeric(df[f'PL{i}_{session}_{k}min_BID'], errors='coerce')
            ask = pd.to_numeric(df[f'PL{i}_{session}_{k}min_ASK'], errors='coerce')
            mk = 0.5 * (bid + ask)
            ok = (mid.isna() & bid.notna() & ask.notna() & (ask >= bid)
                  & ((ask - bid) / mk * 1e4 <= MAX_SPREAD_BPS))
            mid[ok] = mk[ok]
            ladder_used[rank] += ok.sum()
        mids.append(mid)
        taus.append(tau)
    logging.info('%s offset ladder usage %s: %s', session, OFFSET_LADDER, ladder_used.tolist())
    F = np.stack([m.values for m in mids], axis=1)
    T = np.stack([t.values for t in taus], axis=1)
    ok = ~np.isnan(F) & (T > MIN_TAU) & anchor.notna().values[:, None]
    g = ok.all(axis=1)
    A = np.stack([T, T * T, np.ones_like(T)], axis=2)          # rows: [tau, tau^2, 1]
    y = np.log(F / anchor.values[:, None])
    sol = np.full((len(df), 3), np.nan)
    sol[g] = np.linalg.solve(A[g], y[g][:, :, None])[:, :, 0]
    c, a, e = sol.T

    # Delivery-month fallback: the expiring front goes quote-dead (100bp+ wide for weeks) while
    # the next two stay tight. e is tenor-independent by the model, so two live contracts still
    # identify (c, e) with the curvature frozen at its trailing EWMA - causal, prior rows only.
    lam, ewma_a, fell_back = 0.9, np.nan, 0
    for t in range(len(df)):
        if g[t]:
            ewma_a = a[t] if np.isnan(ewma_a) else lam * ewma_a + (1 - lam) * a[t]
        elif ok[t].sum() == 2 and not np.isnan(ewma_a):
            (i, j), a_hat = np.where(ok[t])[0], ewma_a
            cc = (y[t, j] - y[t, i] - a_hat * (T[t, j] ** 2 - T[t, i] ** 2)) / (T[t, j] - T[t, i])
            c[t], a[t], e[t] = cc, a_hat, y[t, i] - cc * T[t, i] - a_hat * T[t, i] ** 2
            fell_back += 1
    logging.info('%s two-contract fallback rows: %d', session, fell_back)
    return c, a, e


def build_archive(df, expiry):
    """Solve both sessions and shape the wide CSV. The AM solve carries the carry curve; each
    session's e lands as that session's own-fix-anchored CME basis."""
    fix = pd.to_numeric(df['AM'], errors='coerce')
    pm = pd.to_numeric(df['PM'], errors='coerce')
    c, a, e_am = solve_session(df, expiry, 'AM', fix)
    _, _, e_pm = solve_session(df, expiry, 'PM', pm)

    out = pd.DataFrame(index=df['DATE'].rename('Date'))
    out[FIX_COL] = fix.values
    out[PM_DIFF_COL] = (pm - fix).values
    out[AM_CME_COL] = fix.values * np.expm1(e_am)
    out[PM_CME_COL] = pm.values * np.expm1(e_pm)
    for tau in REF_TENORS:
        out[f'{CARRY_COL},{tau}'] = c + a * tau
    return out


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    df = pd.read_csv(DATA, parse_dates=['DATE'])
    exp = pd.read_csv(EXPIRY, parse_dates=['Expiry_Date'])
    expiry = dict(zip(exp.Contract_Code, exp.Expiry_Date))
    df = df[df.DATE >= '2010-01-04'].reset_index(drop=True)
    out = build_archive(df, expiry)
    logging.info('rows %d (%s -> %s), solved AM %d (%.1f%%) PM %d (%.1f%%)', len(out),
                 out.index.min().date(), out.index.max().date(), out[AM_CME_COL].notna().sum(),
                 100 * out[AM_CME_COL].notna().mean(), out[PM_CME_COL].notna().sum(),
                 100 * out[PM_CME_COL].notna().mean())

    # The chain links (contiguous business days, 2020+): ID = same-day AM->PM, ON = PM->next AM.
    s = out[out.index >= '2020']
    gap = s.index.to_series().diff().dt.days.values
    b_am, b_pm = s[AM_CME_COL].values, s[PM_CME_COL].values
    for lbl, x, y, contig in (
            ('ID (b_am -> b_pm, same day)', b_am, b_pm, np.ones(len(s), dtype=bool)),
            ('ON (b_pm -> next b_am)', b_pm[:-1], b_am[1:], gap[1:] <= 4),
            ('daily (b_am -> next b_am)', b_am[:-1], b_am[1:], gap[1:] <= 4)):
        m = contig & np.isfinite(x[:len(contig)]) & np.isfinite(y[:len(contig)])
        X = np.column_stack([np.ones(m.sum()), x[:len(contig)][m]])
        beta, *_ = np.linalg.lstsq(X, y[:len(contig)][m], rcond=None)
        r = y[:len(contig)][m] - X @ beta
        logging.info('link %s: n %d beta %.3f resid sd %.2f $/oz', lbl, m.sum(), beta[1], r.std())

    for lbl, lo, hi in (('2010-2019', '2010', '2020'), ('2020+', '2020', '2027'),
                        ('2023+', '2023', '2027'), ('2026', '2026', '2027')):
        p = out[(out.index >= lo) & (out.index < hi)]
        z05, z10 = p[f'{CARRY_COL},0.5'], p[f'{CARRY_COL},1.0']
        logging.info('%s: b_am mean %+6.2f sd %5.2f | b_pm sd %5.2f | pm-am sd %5.2f | '
                     'z(0.5) %+5.2f%% z(1.0) %+5.2f%% | solved %.1f%%', lbl,
                     p[AM_CME_COL].mean(), p[AM_CME_COL].std(), p[PM_CME_COL].std(),
                     p[PM_DIFF_COL].std(), 100 * z05.mean(), 100 * z10.mean(),
                     100 * p[AM_CME_COL].notna().mean())
    out.to_csv(OUT, float_format='%.6f')
    logging.info('wrote %s', OUT)


if __name__ == '__main__':
    main()
