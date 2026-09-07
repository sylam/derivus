"""Compact a book's decomposed autocalls into single `QEDI_CustomAutoCallSwap_V2` deals.

An upstream export books a knock-in-put autocall as one `StructuredDeal` of up to six legs: the V2
swap with its put leg switched off (no `Barrier_Dates`, the put level written as an ABSOLUTE
number in `Barrier`, which the engine reads as a ratio of strike), a sold vanilla put at the
barrier level, optionally a sold cash-or-nothing put (the loss between strike and barrier), an
up-and-in "contra" of each that knocks in at the autocall trigger on the observation dates so the
put dies where the note calls, and an IRS shell with zero notionals. The V2's own put leg prices
all of that in one estimator with the survival weight the coupon strip already carries, so the
folded deal is the same trade priced once instead of five Monte-Carlo legs.

Per structure the legs are checked against the swap's own tables - underlying, payoff currency and
quanto flag, strikes, expiries, sides, the contras' barrier and dates, `units x strike` against the
notional, the digital's payout - and only a structure passing every check is folded:

    Barrier       = put strike / swap strike                       (a ratio)
    Barrier_Dates = [the final coupon date]
    Rebate        = 1 - Barrier - digital payout / notional        (0 = full loss from strike;
                                                                    1 - Barrier = loss below the barrier)

Anything failing a check is left untouched and named in the report; an unfunded digital contra
(`Cash_Payoff` 0.0) is a booking defect the fold repairs, and is named as one.

    python derivus_compact_autocalls.py book.json [-o out.json] [--report out.md] [--validate]
                                                  [--dry-run] [--reference REF ...]
"""
import argparse
import json
import os
import tempfile

REL = 1e-9
LEVEL = 1e-6   # a barrier level is quoted to two decimals; anything derived from the ratio carries that

ROLES = {('QEDI_CustomAutoCallSwap_V2', None, None): 'swap',
         ('EquityOptionDeal', 'Put', None): 'put',
         ('EquityBarrierOption', 'Put', 'Up_And_In'): 'put_contra',
         ('EquityBinaryOption', 'Put', None): 'digital',
         ('EquityBarrierBinaryOption', 'Put', 'Up_And_In'): 'digital_contra'}
SHELL = {'CFFixedInterestListDeal', 'CFFloatingInterestListDeal'}
OTHER_SIDE = {'Buy': 'Sell', 'Sell': 'Buy'}


def deal(node):
    return (node.get('Instrument') or {}).get('.Deal', {})


def ts(x):
    return x['.Timestamp'] if isinstance(x, dict) else str(x)


def close(a, b, rel=REL):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


def role_of(d):
    obj = d.get('Object')
    if obj == 'QEDI_CustomAutoCallSwap_V2':
        return 'swap'
    return ROLES.get((obj, d.get('Option_Type'), d.get('Barrier_Type') if 'Barrier' in obj else None))


def is_shell(node):
    legs = node.get('Children', []) or []
    return deal(node).get('Object') == 'StructuredDeal' and legs and {deal(g).get('Object') for g in legs} <= SHELL


def notionals_zero(shell):
    return not any(item.get('Notional', 0.0)
                   for leg in shell.get('Children', []) or []
                   for item in deal(leg).get('Cashflows', {}).get('Items', []))


def classify(node):
    """The legs of a StructuredDeal by role, or None where the pattern is not an autocall's."""
    roles = {'swap': [], 'put': [], 'put_contra': [], 'digital': [], 'digital_contra': [], 'shell': [], 'other': []}
    for child in node.get('Children', []) or []:
        role = role_of(deal(child)) or ('shell' if is_shell(child) else 'other')
        roles[role].append(child)
    if len(roles['swap']) != 1 or not (roles['put'] or roles['digital']):
        return None
    return roles


def check(roles):
    """Every leg against the swap's tables: (findings, fatal, ratio, digital payout)."""
    swap = deal(roles['swap'][0])
    strike, units = swap['Strike_Price'], swap['Units']
    obs = [ts(r[0]) for r in swap['Autocall_Thresholds']]
    coupons = [ts(r[0]) for r in swap['Autocall_Coupons']]
    level = deal((roles['put'] or roles['digital'])[0])['Strike_Price']
    ratio = level / strike
    findings, fatal = [], []

    def expect(cond, text):
        (findings if cond else fatal).append(('ok ' if cond else 'FAIL ') + text)

    for role in ('put', 'put_contra', 'digital', 'digital_contra'):
        for leg in roles[role]:
            d = deal(leg)
            expect(d['Equity'] == swap['Equity'] and d['Payoff_Currency'] == swap['Payoff_Currency']
                   and d.get('Payoff_Type') == swap.get('Payoff_Type'),
                   '%s %s on the swap\'s underlying, payoff currency and quanto flag' % (role, d.get('Reference')))
            expect(close(d['Strike_Price'], level), '%s strike %g = the barrier level %g' % (role, d['Strike_Price'], level))
            expect(ts(d['Expiry_Date']) == obs[-1], '%s expiry %s = the final observation %s' % (role, ts(d['Expiry_Date']), obs[-1]))
            expect(d['Buy_Sell'] == (swap['Buy_Sell'] if role.endswith('contra') else OTHER_SIDE[swap['Buy_Sell']]),
                   '%s side %s against the swap\'s %s' % (role, d['Buy_Sell'], swap['Buy_Sell']))
            if role.endswith('contra'):
                expect(close(d['Barrier_Price'], strike), '%s barrier %g = the swap strike, the autocall trigger' % (role, d['Barrier_Price']))
                expect(sorted(ts(r[0]) for r in d.get('Barrier_Dates', [])) == sorted(obs), '%s barrier dates = the observation dates' % role)
    for role in ('put', 'put_contra'):
        total = sum(deal(x)['Units'] for x in roles[role])
        if roles[role]:
            expect(close(total * strike, units, LEVEL), '%s units x strike = %g = the notional %g (the units imply an initial level of %.2f)' % (
                role, total * strike, units, units / total))
    payout = sum(deal(x)['Cash_Payoff'] for x in roles['digital'])
    expect(payout <= (1.0 - ratio) * units * (1.0 + LEVEL), 'digital payout %g within the strike-to-barrier gap %g' % (payout, (1.0 - ratio) * units))
    for leg in roles['digital']:
        if ts(deal(leg).get('Settlement_Date', coupons[-1])) != coupons[-1]:
            findings.append('note digital settles %s, the final coupon date is %s' % (ts(deal(leg)['Settlement_Date']), coupons[-1]))
    contra = sum(deal(x)['Cash_Payoff'] for x in roles['digital_contra'])
    if roles['digital'] and roles['digital_contra']:
        if contra == 0.0:
            findings.append('DEFECT digital contra Cash_Payoff is 0.0 (should be %g): the sold digital was carried as if the note could never call' % payout)
        else:
            expect(close(contra, payout, LEVEL), 'digital contra pays %g = the digital' % contra)
    expect(bool(roles['put_contra']) == bool(roles['put']) and bool(roles['digital_contra']) == bool(roles['digital']),
           'every sold leg has its contra')
    expect(not swap.get('Barrier_Dates'), 'the swap\'s own put leg is off (no Barrier_Dates)')
    if swap.get('Barrier', 0.0) > 1.0:
        findings.append('note the swap\'s Barrier field holds the ABSOLUTE level %g; the engine reads a ratio of strike' % swap['Barrier'])
    for shell in roles['shell']:
        expect(notionals_zero(shell), 'IRS shell %s has zero notionals on every row' % deal(shell).get('Reference'))
    expect(not roles['other'], 'no leg outside the pattern (%s)' % ', '.join(deal(x).get('Object', '?') for x in roles['other']))
    return findings, fatal, ratio, payout


def fold(node, roles, ratio, payout):
    swap = deal(roles['swap'][0])
    legs = roles['put'] + roles['put_contra'] + roles['digital'] + roles['digital_contra'] + roles['shell']
    swap['Barrier'] = round(ratio, 10)
    swap['Barrier_Dates'] = [{'.Timestamp': ts(swap['Autocall_Coupons'][-1][0])}]
    rebate = 1.0 - ratio - payout / swap['Units']
    swap['Rebate'] = 0.0 if abs(rebate) < LEVEL else round(rebate, 10)
    swap['MtM'] = sum(deal(x).get('MtM') or 0.0 for x in roles['swap'] + legs)
    node['Children'] = [c for c in node['Children'] if c not in legs]
    return [(deal(x)['Object'], deal(x).get('Reference'), deal(x).get('MtM')) for x in legs]


def structures(node, out):
    if isinstance(node, dict):
        if deal(node).get('Object') == 'StructuredDeal' and classify(node):
            out.append(node)
        for v in node.values():
            structures(v, out)
    elif isinstance(node, list):
        for v in node:
            structures(v, out)
    return out


def prune_empty_legs(node, out):
    """Drop every cashflow-list leg whose notionals are all zero, wherever it sits, and any
    StructuredDeal that is left with no children; returns the references dropped."""
    if isinstance(node, dict):
        kids = node.get('Children')
        if isinstance(kids, list):
            keep = []
            for child in kids:
                d = deal(child)
                dead = (d.get('Object') in SHELL and not any(
                    item.get('Notional', 0.0) for item in d.get('Cashflows', {}).get('Items', [])))
                if not dead:
                    prune_empty_legs(child, out)
                    dead = d.get('Object') == 'StructuredDeal' and not child.get('Children')
                (out if dead else keep).append(d.get('Reference') if dead else child)
            node['Children'] = keep
        else:
            for v in node.values():
                prune_empty_legs(v, out)
    elif isinstance(node, list):
        for v in node:
            prune_empty_legs(v, out)
    return out


def compact(doc, references=None, dry_run=False):
    """Fold every qualifying structure in `doc` in place, then prune the dead legs; returns the
    report lines."""
    lines, folded, removed = [], 0, 0
    found = [n for n in structures(doc, []) if not references or str(deal(n).get('Reference')) in references]
    lines.append('%d decomposed autocall structures found' % len(found))
    for node in found:
        roles = classify(node)
        swap = deal(roles['swap'][0])
        findings, fatal, ratio, payout = check(roles)
        lines += ['', '## %s - %s' % (deal(node).get('Reference'), swap.get('Reference')),
                  'underlying %s, notional %g %s, strike %g, barrier %.4f of strike, %d observations %s .. %s' % (
                      swap['Equity'], swap['Units'], swap['Payoff_Currency'], swap['Strike_Price'], ratio,
                      len(swap['Autocall_Thresholds']), ts(swap['Autocall_Thresholds'][0][0]), ts(swap['Autocall_Thresholds'][-1][0]))]
        lines += ['- ' + f for f in findings + fatal]
        if fatal:
            lines.append('LEFT UNTOUCHED: %d checks failed' % len(fatal))
            continue
        if dry_run:
            lines.append('WOULD FOLD: Barrier %.10g, Barrier_Dates [%s], Rebate %.10g' % (
                ratio, ts(swap['Autocall_Coupons'][-1][0]), 1.0 - ratio - payout / swap['Units']))
            continue
        dropped = fold(node, roles, ratio, payout)
        folded, removed = folded + 1, removed + len(dropped)
        lines.append('FOLDED into one V2: Barrier %.10g, Barrier_Dates [%s], Rebate %.10g; MtM carried %g' % (
            swap['Barrier'], swap['Barrier_Dates'][0]['.Timestamp'], swap['Rebate'], swap['MtM']))
        lines += ['  - dropped %s %s (MtM %s)' % x for x in dropped]
    pruned = [] if dry_run else prune_empty_legs(doc, [])
    lines += ['', '%d structures folded, %d legs removed; %d zero-notional legs pruned elsewhere%s' % (
        folded, removed, len(pruned), ': ' + ', '.join(str(x) for x in pruned) if pruned else '')]
    return lines


def validate(doc):
    """Compile the folded book through the engine, without the market-data files it points at."""
    import derivus
    trimmed = json.loads(json.dumps(doc))
    trimmed['Calc'].get('MergeMarketData', {}).pop('MarketDataFile', None)
    trimmed['Calc'].pop('CalendDataFile', None)
    fd, path = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(trimmed, f)
    try:
        cx = derivus.Context()
        cx.load_json(path)
        report = cx.validate()
    finally:
        os.remove(path)
    return ['', 'engine validation: %d deals with messages, %d factors missing' % (len(report['deals']), len(report['factors']))] + [
        '  - %s: %s' % (ref, '; '.join(msgs)) for ref, msgs in report['deals'].items()] + ['  - missing %s' % f for f in report['factors']]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('book')
    ap.add_argument('-o', '--out', help='the folded book (default: <book>_compact.json)')
    ap.add_argument('--report', help='the report (default: beside the output, .md)')
    ap.add_argument('--reference', nargs='*', help='fold only these StructuredDeal references')
    ap.add_argument('--dry-run', action='store_true', help='report what would fold; write nothing')
    ap.add_argument('--validate', action='store_true', help='compile the folded book through derivus')
    args = ap.parse_args()
    doc = json.load(open(args.book, encoding='utf-8'))
    lines = ['# Autocall compaction: %s' % args.book, ''] + compact(doc, set(args.reference or []), args.dry_run)
    if args.validate and not args.dry_run:
        lines += validate(doc)
    print('\n'.join(lines))
    if not args.dry_run:
        out = args.out or os.path.splitext(args.book)[0] + '_compact.json'
        json.dump(doc, open(out, 'w', encoding='utf-8'), indent=1)
        open(args.report or os.path.splitext(out)[0] + '.md', 'w', encoding='utf-8').write('\n'.join(lines))
        print('written', out)


if __name__ == '__main__':
    main()
