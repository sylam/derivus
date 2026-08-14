"""Build the AM-fix-anchored platinum calibration archive.

The model (fix-anchored, 2026-08): the LBMA AM fix is the primary state; the PM fix and the CME
futures curve hang off it,

    F_i = Fix_AM * exp(c*tau_i + a*tau_i^2 + e)        i = 1, 2, 3   (exactly identified per day)

so each day's three front futures at the AM+3min snapshot solve the 3x3 Vandermonde for (c, a, e).
The archive rows are framework factor names (the column header IS the contract):

    CommodityPrice.LBMA_AM            the AM fix (observed, never synthetic)
    ObservedBasis.LBMA_AM.BASIS_PM    PM fix - AM fix  (composed PM fixing = LBMA_AM.BASIS_PM)
    ObservedBasis.LBMA_AM.CME         fix*(exp(e)-1)   (composed futures underlying = LBMA_AM.CME,
                                      so (fix + b_cme)*exp(z(tau)*tau) reproduces F_i exactly)
    ForwardRate.PLATINUM_CARRY,0.5    z(0.5) = c + 0.5a   (z(tau)*tau = c*tau + a*tau^2, total
    ForwardRate.PLATINUM_CARRY,1.0    z(1.0) = c + a       carry incl. financing, as in the sync world)

Rows where the snapshot fails quality (crossed, wide, missing contract, tau <= 2d) leave the solved
columns NaN; the fixes are always carried. Tenors use days/365.25 (the framework's DAYS_IN_YEAR --
the old riskflow extraction's /365.0 is corrected here).
"""
import logging
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'platinum_lbma_with_futures_basis.csv')
EXPIRY = os.path.join(ROOT, 'data', 'platinum_futures_contract_expiry.csv')
OUT = os.path.join(ROOT, 'data', 'plat_archive_am.csv')

FIXING = 'AM'
OFFSET_LADDER = (3, 2, 4, 1, 5, 0)   # prefer the +3min anchor, fall back to the nearest minutes
MAX_SPREAD_BPS = 100.0               # junk guard only: a wide uncrossed mid is information, a
MIN_TAU = 2.0 / 365.25               # missing/crossed quote is not (30bp censored 15-23% of
                                     # post-2020 STRESS days - missing-not-at-random)
DAYS_IN_YEAR = 365.25
REF_TENORS = (0.5, 1.0)

FIX_COL = 'CommodityPrice.LBMA_AM'
PM_BASIS_COL = 'ObservedBasis.LBMA_AM.BASIS_PM'
CME_BASIS_COL = 'ObservedBasis.LBMA_AM.CME'
CARRY_COL = 'ForwardRate.PLATINUM_CARRY'


def build_archive(df, expiry):
    """Solve (c, a, e) per day from the three AM+3min futures mids and shape the wide CSV."""
    fix = pd.to_numeric(df[FIXING], errors='coerce')
    pm = pd.to_numeric(df['PM'], errors='coerce')
    mids, taus, good = [], [], fix.notna()
    ladder_used = np.zeros(len(OFFSET_LADDER), dtype=int)
    for i in (1, 2, 3):
        tau = df[f'PL{i}_{FIXING}_CONTRACT'].map(expiry).sub(df['DATE']).dt.days / DAYS_IN_YEAR
        mid = pd.Series(np.nan, index=df.index)
        for rank, k in enumerate(OFFSET_LADDER):
            bid = pd.to_numeric(df[f'PL{i}_{FIXING}_{k}min_BID'], errors='coerce')
            ask = pd.to_numeric(df[f'PL{i}_{FIXING}_{k}min_ASK'], errors='coerce')
            mk = 0.5 * (bid + ask)
            ok = (mid.isna() & bid.notna() & ask.notna() & (ask >= bid)
                  & ((ask - bid) / mk * 1e4 <= MAX_SPREAD_BPS))
            mid[ok] = mk[ok]
            ladder_used[rank] += ok.sum()
        good &= mid.notna() & (tau > MIN_TAU)
        mids.append(mid)
        taus.append(tau)
    logging.info('offset ladder usage %s: %s', OFFSET_LADDER, ladder_used.tolist())
    F = np.stack([m.values for m in mids], axis=1)
    T = np.stack([t.values for t in taus], axis=1)
    ok = ~np.isnan(F) & (T > MIN_TAU) & fix.notna().values[:, None]
    g = ok.all(axis=1)
    A = np.stack([T, T * T, np.ones_like(T)], axis=2)          # rows: [tau, tau^2, 1]
    y = np.log(F / fix.values[:, None])
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
    logging.info('two-contract fallback rows: %d', fell_back)

    out = pd.DataFrame(index=df['DATE'].rename('Date'))
    out[FIX_COL] = fix.values
    out[PM_BASIS_COL] = (pm - fix).values
    out[CME_BASIS_COL] = fix.values * np.expm1(e)
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
    solved = out[CME_BASIS_COL].notna()
    logging.info('rows %d (%s -> %s), solved %d (%.1f%%)', len(out), out.index.min().date(),
                 out.index.max().date(), solved.sum(), 100 * solved.mean())
    for lbl, lo, hi in (('2010-2019', '2010', '2020'), ('2020+', '2020', '2027'),
                        ('2023+', '2023', '2027'), ('2026', '2026', '2027')):
        s = out[(out.index >= lo) & (out.index < hi)]
        z05, z10 = s[f'{CARRY_COL},0.5'], s[f'{CARRY_COL},1.0']
        logging.info('%s: b_cme mean %+6.2f sd %5.2f $/oz | b_pm sd %5.2f | z(0.5) %+5.2f%% '
                     'z(1.0) %+5.2f%% | solved %.1f%%', lbl, s[CME_BASIS_COL].mean(),
                     s[CME_BASIS_COL].std(), s[PM_BASIS_COL].std(), 100 * z05.mean(),
                     100 * z10.mean(), 100 * s[CME_BASIS_COL].notna().mean())
    out.to_csv(OUT, float_format='%.6f')
    logging.info('wrote %s', OUT)


if __name__ == '__main__':
    main()
