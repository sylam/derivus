"""SEAM READERS - every function reading a primitive beside the seam that is its one spelling.

    python gates/seam_readers.py            # the readers outside each seam, and the verdict
    python gates/seam_readers.py --json

A SEAM is the function that is the one spelling of a read (`pricing.calc_moneyness`, where a vol
surface is read), declared in `SEAMS` with the primitives it wraps: the reads a function makes in
its place where it does not take it. Off `reach.py`'s graph - its functions, import table and class
hierarchy - every outermost function under `derivus/`, a method or a module function with the defs
nested in it, that references a seam's primitive and not the seam is a READER OUTSIDE IT. A
primitive is a module function (`utils.curve_spread`), a method on a receiver naming its class or a
subclass (`utils.VolSurface.rate`), or `.name`, an attribute of any receiver (`.get_vols`); an
attribute of `self` or `cls` is its owner's own and reads nothing.

`KNOWN` accepts a reader by name, with the reason it stands outside the seam. The run fails on a
reader neither its seam nor `KNOWN` explains, on a `KNOWN` entry with no reason or no longer reading
outside its seam, and on a seam declaring no primitive or one naming nothing in the code.

BLIND to a read through a callable passed as a value or `getattr`, to a coordinate computed by a
caller and handed in (`KNOWN` says where), and to a raw read beside a call of its seam in one
function: referencing the seam once explains every read beside it.
"""
import argparse
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reach  # noqa: E402

#: `{seam: primitives}`: each seam the one spelling of a read, and the reads it wraps.
SEAMS = {
    # where a vol surface is read: the coordinate its subtype names
    'pricing.calc_moneyness': ('riskfactors.Factor2D.current_value', '.get_vols', '.moneyness',
                               'utils.VolSurface.rate'),
    # a carry: the curves' integral over the year fraction its consumer multiplies it by
    'pricing.zero_carry_rate': ('utils.curve_spread',),
    # the interval carry: the zero carry differenced over each fixing interval
    'pricing.forward_carry_rate': ('pricing.zero_carry_rate',),
    # the log forward: the interval strip integrated over the fixings
    'pricing.total_log_forward': ('pricing.forward_carry_rate',),
    # the vol at each fixing's tenor, at the moneyness of the forward to it
    'pricing.forward_vol_strip': ('pricing.zero_carry_rate',),
    # the interval vol: the implied strip differenced over each fixing interval
    'pricing.forward_vol_rate': ('pricing.forward_vol_strip',),
}

_ATM = 'an ATM vol: a None coordinate, which `VolSurface.rate` places by the surface\'s subtype'
_WALK = 'steps the interval strip along its path, and no forward is totalled'
_DIGITALS = 'prices each observation as a European digital on its own forward and its own implied ' \
            'vol: no interval is stepped'
_AVERAGE = 'forwards each sample on its zero carry and moment-matches on its implied vol: an ' \
           'average of forwards steps no interval'

#: `{(seam, reader): reason}`: a reader outside its seam accepted by name.
KNOWN = {
    ('pricing.calc_moneyness', 'bootstrappers.OptionQuoteFamily.fx_surface_block'):
        'reads the surface at `OptionQuoteFamily.moneyness`, the family\'s call of the seam; its '
        '`.moneyness` is the grid check\'s node count',
    ('pricing.calc_moneyness', 'bootstrappers.GBMAssetPriceTSModelParameters.atm_column'):
        'reads the surface at `OptionQuoteFamily.moneyness`, the family\'s call of the seam, at a '
        'strike on the forward; its `.moneyness` and `.get_vols` are the grid that coordinate is '
        'interpolated on',
    ('pricing.calc_moneyness', 'instruments.EquityForwardDeal.generate'): _ATM,
    ('pricing.calc_moneyness', 'pricing.calc_vol_adjustment'): _ATM,
    ('pricing.calc_moneyness', 'pricing.pv_equity_cashflows'): _ATM,
    ('pricing.calc_moneyness', 'pricing.quanto_step_loading'): _ATM,
    ('pricing.calc_moneyness', 'pricing.pv_european_option'):
        'reads at the moneyness its deal\'s `generate` computes through the seam, a digital\'s '
        'spread legs included',
    ('pricing.calc_moneyness', 'pricing.pv_american_option'):
        'reads at the moneyness its deal\'s `generate` computes through the seam',
    ('pricing.calc_moneyness', 'pricing.pv_MC_AutoCallSwap'):
        'reads its expiry vol at the moneyness `QEDI_CustomAutoCallSwap.generate` computes '
        'through the seam, and its strips through `forward_vol_strip`',
    ('pricing.zero_carry_rate', 'utils.calc_fx_forward'):
        'the outright forward, `exp` of the integral the zero carry divides by its year fraction',
    ('pricing.zero_carry_rate', 'pricing.pv_MC_ExtendableForward'):
        'spells its log forward off `curve_spread`, its per-fixing vol and its interval carry and '
        'variance inline beside the seams; the fold onto them is a roadmap row, not yet landed',
    ('pricing.forward_carry_rate', 'pricing.deal_carry_rate'):
        'the zero carry over its deal\'s day count, for a pricer reading one expiry',
    ('pricing.forward_vol_strip', 'pricing.deal_carry_rate'): 'returns a carry and reads no vol',
    ('pricing.forward_carry_rate', 'pricing.pv_discrete_asian_option'): _AVERAGE,
    ('pricing.forward_carry_rate', 'pricing.pv_discrete_double_asian_option'): _AVERAGE,
    ('pricing.forward_vol_rate', 'pricing.pv_discrete_asian_option'): _AVERAGE,
    ('pricing.forward_vol_rate', 'pricing.pv_discrete_double_asian_option'): _AVERAGE,
    ('pricing.forward_carry_rate', 'pricing.pv_fx_range_accrual'): _DIGITALS,
    ('pricing.forward_vol_rate', 'pricing.pv_fx_range_accrual'): _DIGITALS,
    ('pricing.total_log_forward', 'pricing.pv_MC_Accumulator'): _WALK,
    ('pricing.total_log_forward', 'pricing.pv_MC_AutoCallSwap'): _WALK,
    ('pricing.total_log_forward', 'pricing.pv_MC_Tarf'): _WALK,
}


def units(g):
    """`{key: node}` for every outermost function: a module function or a class's method."""
    return {k: v for k, v in g.node.items() if v['kind'] == 'func' and (
        k.rsplit('.', 1)[0] in g.mods or g.node[k.rsplit('.', 1)[0]]['kind'] == 'class')}


def references(g, node):
    """`(names, attributes)` a function references, the defs nested in it included: each name as
    the key it resolves to, each attribute as `(receiver, name)`, `self`, `cls` and `super()` as
    `self`."""
    mod, imports = node['mod'], g.imports[node['mod']]
    names, attributes = set(), set()
    for n in ast.walk(node['ast']):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            names.update((imports.get(n.id), '{}.{}'.format(mod, n.id)))
        elif isinstance(n, ast.Attribute):
            receiver = ast.unparse(n.value)
            attributes.add(('self' if receiver in ('self', 'cls') or receiver.startswith('super(')
                            else receiver, n.attr))
    return names, attributes


def refers(g, refs, spec):
    """Whether `refs`, a function's `references`, reach `spec` on another object than its own."""
    names, attributes = refs
    head, _, attr = spec.rpartition('.')
    hierarchy = {c.rsplit('.', 1)[1] for c in ({head} | g._all_bases(head) | g.sub.get(head, set()))
                 } if head in g.node and g.node[head]['kind'] == 'class' else set()
    return spec in names or any(
        at == attr and receiver != 'self' and (
            not head or receiver == head or receiver.split('.')[-1] in hierarchy)
        for receiver, at in attributes)


def resolves(g, spec):
    """Whether `spec` names something in the code: a function or method, or an attribute a class
    defines or assigns on `self`."""
    if not spec.startswith('.'):
        return spec in g.node
    return any(k.endswith(spec) for k in g.node) or any(
        isinstance(n, ast.Attribute) and n.attr == spec[1:] and isinstance(n.ctx, ast.Store)
        for tree in g.mods.values() for n in ast.walk(tree))


def census(g):
    """`{seam: {reader: [primitives]}}`: every function referencing a primitive of a seam it does
    not reference, the seam's own nested defs aside."""
    refs = {key: references(g, node) for key, node in units(g).items()}
    out = {}
    for seam, primitives in SEAMS.items():
        found = out.setdefault(seam, {})
        for key, ref in refs.items():
            hits = [p for p in primitives if refers(g, ref, p)]
            if hits and key != seam and not refers(g, ref, seam):
                found[key] = hits
    return out


def failures(g, found):
    """`[(label, [rows])]`: what fails the run."""
    return (
        ('seams declaring no primitive', sorted(s for s, p in SEAMS.items() if not p)),
        ('seams or primitives naming nothing in derivus/', sorted(
            s for seam, prims in SEAMS.items() for s in (seam,) + tuple(prims)
            if not resolves(g, s))),
        ('KNOWN entries giving no reason', sorted(k for k, why in KNOWN.items()
                                                  if not str(why or '').strip())),
        ('KNOWN entries reading nothing outside their seam - delete them', sorted(
            k for k in KNOWN if k[1] not in found.get(k[0], {}))),
        ('readers outside a seam that nothing explains', sorted(
            (seam, reader) for seam, readers in found.items() for reader in readers
            if (seam, reader) not in KNOWN)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=reach.ROOT)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    g = reach.Graph(os.path.abspath(args.repo))
    found = census(g)
    bad = failures(g, found)
    if args.json:
        print(json.dumps({'readers': found, 'failures': dict(bad)}, indent=1, default=list))
    else:
        for seam, readers in found.items():
            print('{}  ({})'.format(seam, ', '.join(SEAMS[seam])))
            for reader, hits in sorted(readers.items()):
                print('  {:<62} {}\n      -> {}'.format(reader, ', '.join(hits), KNOWN.get(
                    (seam, reader), '*** NOT EXPLAINED ***')))
        for label, rows in bad:
            if rows:
                print('\nFAIL: {}:'.format(label))
                for row in rows:
                    print('  {}'.format(row))
    return 1 if any(rows for _, rows in bad) else 0


if __name__ == '__main__':
    sys.exit(main())
