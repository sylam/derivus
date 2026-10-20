"""ADMITTED-VALUES CENSUS - a value a reader admits that no document reaching it states is a branch
no data takes.

    python gates/declared_values.py tests/...     # self-driving; --values-json PATH
    ... --values-strict                           # also fail on a hole, not only an unmet MUST

WHAT IS CENSUSED. Every declared menu - each `F(..., values=[...])` of every store `schema.mapping`
publishes, a container's child and a table's column included - per type, a deal type taking its
sections' menus; and every class-level ADMISSION in `ADMISSIONS`, a tuple or dict of values a
reader compares a stated field against (`OptionQuoteFamily.tabular_surfaces`, a deal's
`spot_models`). A row is `(reader, field)`: the values it admits, the values the documents reaching
it state, and its HOLES, the admitted values none states. A field a document omits or states blank
states the default a read completes it to, a deal's convention or another store's default; a
table's column declares none.

HOW A DOCUMENT IS PLACED. A pytest plugin in `fixture_degeneracy.py`'s shape. A document is what
`Context.load_json` returns and what each run entry point is handed (`run_baseval`, `run_cmc`,
`run_hedgemontecarlo`, `run_simm`, `run_saccr`): its market data, the file it names included, its
deals, its calculation block and the mapping file that names; a run also says which factor types it
simulated. A menu's reader is its type, which a document reaches by stating an entry of it. An
admission's readers are the functions referencing it, off `reach.py`'s graph, or where every deal
type declares its own (`spot_models`) the type; a function is placed by the tests it ran in, traced
by a PY_START event on its code alone as `reach.py`'s document map traces a document, and a
document reaches it in the test it was seen in. A function that ran in a test where no document
was seen is BLIND there, counted per row and never a pass; a row nothing in the selection reaches
is a scoped run, not a finding, and a store no seam states (`Calibration`) is blind whole.

WHAT FAILS. An unmet `MUST`: a `(reader, field, value)` some document reaching the reader states,
the field read wherever a store declares it, in a run simulating the factor type `simulated` names
where it names one - the conjunction one run must hold, as `fixture_degeneracy.JOINT` spells it. A
`MUST` whose reader nothing in the selection runs is skipped, and one whose reader runs is held, so
a selection reaching the reader carries the module meeting it (`tests/test_cross_spot_model.py`
the emitter's, `tests/test_inflation_list.py` the linker's). `--values-strict` fails a hole too;
the table is otherwise a report.

WHAT IT CANNOT DO. A test is one placement: two documents seen in one test reaching one function
are both credited with what either states, which a `MUST` naming `simulated` narrows to the
document whose run meets it. It reads what a document states, not what the reader reads off it.
"""
import importlib
import json
import os
import sys
import tempfile
from collections import defaultdict, namedtuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reach  # noqa: E402
import seam_readers  # noqa: E402

#: A class-level admission: `attribute` a `module.Class.name` (a dict's entry by a further key, `*`
#: every deal type declaring other than `Deal`'s), the `field` it is compared against at its path
#: in an entry (`*` any key), the `section` stating it and the entry `types` (None for any), and
#: the `default` an entry omitting it is read as where no store declares one.
Admission = namedtuple('Admission', 'attribute field section types default', defaults=(None, None))
Row = namedtuple('Row', 'reader field admitted where default placed')
Must = namedtuple('Must', 'reader field value simulated why')

VOLS = ('FXVol', 'EquityPriceVol', 'CommodityPriceVol')

ADMISSIONS = (
    Admission('bootstrappers.OptionQuoteFamily.tabular_surfaces', 'Surface_Type', 'Price Factors',
              VOLS),
    Admission('bootstrappers.OptionQuoteFamily.quote_type_references', 'Quote_Type',
              'Market Prices', ('LogVar2FJModelPrices',)),
    Admission('instruments.DealDefaultSwap.quoted', 'Points.Quote_Type', 'Market Prices',
              ('SurvivalProbPrices',)),
    Admission('instruments.*.spot_models', 'SpotModel', 'Valuation Configuration', None, 'None'),
    Admission('calculation.SIMM.VOL_PILLARS', 'Surface_Type', 'Price Factors', VOLS),
    Admission('calculation.SIMM.OUTRIGHTS', 'Points.DealType', 'Market Prices',
              ('InterestRatePrices',)),
    Admission('calculation.SIMM.MOVES', 'Factors.*.SIMM.RiskType', 'Mapping'),
    Admission('calculation.SACCR.SUBCLASSES', 'SACCR.Asset_Class.*', 'Mapping'),
    Admission('calculation.SACCR.SUBCLASSES.Credit', 'Factors.*.SACCR.Rating', 'Mapping'),
    Admission('calculation.SACCR.EQUITY_INDICES', 'Factors.*.SIMM.Bucket', 'Mapping'),
)

MUST = (
    Must('bootstrappers.OptionQuoteFamily.fx_surface_block', 'Surface_Type', 'Explicit', None,
         'the emitter reads an Explicit surface at the ratio of forward to strike, its own axis'),
    Must('bootstrappers.OptionQuoteFamily.fx_surface_block', 'Surface_Type', 'Relative_Forward',
         None, 'the emitter reads a Relative_Forward surface at the strike\'s distance from the '
                'forward over it'),
    Must('pricing.pv_index_cashflows', 'Is_Forward_Deal', 'Yes', 'PriceIndex',
         'the reset filter of a forward linker reads the index path to its settlement'),
)

#: `{store: the section its entries are stated in}`; a store missing here no seam states.
SECTIONS = {'Factor': 'Price Factors', 'Process': 'Price Models', 'MarketPrices': 'Market Prices',
            'Calculation': 'Calculation', 'Configuration': 'Bootstrapper Configuration',
            'System': 'System Parameters', 'Instrument': 'Deals'}


def descriptors(mapping):
    """`[(store, type, {key: descriptor})]` for every type of every store `mapping` publishes."""
    for store, body in mapping.items():
        if store == 'Instrument':
            for kind, sections in body['types'].items():
                yield store, kind, {k: d for s in sections for k, d in body['sections'][s].items()}
        elif store == 'Configuration':
            for kind, entry in body['Bootstrapper Configuration']['types'].items():
                yield store, kind, entry['fields']
        elif isinstance(body, dict) and isinstance(body.get('fields'), dict):
            yield store, next(iter(body.get('types', {})), store), body['fields']
        elif isinstance(body, dict) and isinstance(body.get('types'), dict):
            for kind, fields in body['types'].items():
                if isinstance(fields, dict):
                    yield store, kind, fields.get('fields', fields)


def menus(fields, deal, path=()):
    """`[(path, values, default)]` for every menu among `fields`, its containers and tables, the
    default what a read completes an omission to: a deal's convention, any other store's default,
    and a table column nothing."""
    for key, d in fields.items():
        if not isinstance(d, dict):
            continue
        if d.get('values') is not None:
            yield path + (key,), tuple(d['values']), d.get('value') if (
                d.get('convention') if deal else not d.get('required')) else None
        for column, kind in zip(d.get('col_names', ()), d.get('sub_types', ())):
            if kind.get('source') is not None:
                yield path + (key, column), tuple(kind['source']), None
        yield from menus(d.get('sub_fields') or {}, deal, path + (key,))


def admitted(spec):
    """The values an admission's `module.Class.name[.key]` holds: a dict's keys, a tuple whole."""
    module, owner, name, *key = spec.split('.')
    value = getattr(getattr(importlib.import_module('derivus.' + module), owner), name)
    for k in key:
        value = value[k]
    return tuple(value)


def build_rows(g, mapping):
    """Every census row: each store type's menus, each admission's readers, each `MUST`'s."""
    rows, defaults = [], {}
    for store, kind, fields in descriptors(mapping):
        for path, values, default in menus(fields, store == 'Instrument'):
            rows.append(Row('{}.{}'.format(store, kind), '.'.join(path), values,
                            ((SECTIONS.get(store), kind, path),), default, 'type'))
            defaults[(SECTIONS.get(store), kind, path)] = default
    instruments = importlib.import_module('derivus.instruments')
    funcs = {key: seam_readers.references(g, node) for key, node in seam_readers.units(g).items()}
    for a in ADMISSIONS:
        path = tuple(a.field.split('.'))
        if a.attribute.startswith('instruments.*.'):
            name = a.attribute.rsplit('.', 1)[1]
            base = getattr(instruments.Deal, name)
            for kind in mapping['Instrument']['types']:
                values = tuple(getattr(getattr(instruments, kind), name))
                if values != tuple(base):
                    rows.append(Row('Instrument.' + kind, a.field, values,
                                    ((a.section, kind, path),), a.default, 'type'))
            continue
        owner = '.'.join(a.attribute.split('.')[:2])
        attribute = a.attribute.split('.')[2]
        family = {owner} | g.sub.get(owner, set())
        named = {k.rsplit('.', 1)[1] for k in family}
        where = tuple((a.section, kind, path) for kind in (a.types or (None,)))
        for key, (_, attributes) in sorted(funcs.items()):
            if any(at == attribute and (receiver.split('.')[-1] in named or (
                    receiver == 'self' and g.node[key]['cls'] in family))
                   for receiver, at in attributes):
                rows.append(Row(key, a.field, admitted(a.attribute), where, a.default if
                                a.default is not None else defaults.get(where[0]), 'function'))
    for m in MUST:
        if not any(r.reader == m.reader and r.field == m.field for r in rows):
            where = tuple(w for r in rows if r.placed == 'type' and r.field == m.field
                          for w in r.where)
            rows.append(Row(m.reader, m.field, next(r.admitted for r in rows if r.field == m.field),
                            where, defaults.get(where[0]), 'function'))
    return rows


# ======================================================================================
# observation
# ======================================================================================

STATE = {'rows': [], 'index': {}, 'codes': {}, 'must': [], 'node': '<setup>'}
SEEN = {'values': defaultdict(set), 'reached': set(), 'blind': defaultdict(int), 'met': set(),
        'exercised': set(), 'documents': 0, 'errors': []}
TEST = {}


def reset():
    TEST.update(stated=defaultdict(set), present=set(), satisfied=set(), ran=set(), seen=0)


def sections(cfg):
    """`[(section, type, block)]` for every entry a loaded document states."""
    params, deals = cfg.params, cfg.deals
    booked = [d.field for d in cfg.walk_deals() if hasattr(d, 'field')]
    out = [('Deals', d.get('Object'), d) for d in booked]
    out += [('Valuation Configuration', t, params.get('Valuation Configuration', {}).get(t, {}))
            for t in {d.get('Object') for d in booked}]
    for section in ('Price Factors', 'Price Models', 'Market Prices'):
        out += [(section, name.split('.')[0], block.get('instrument', block)
                 if section == 'Market Prices' else block)
                for name, block in params.get(section, {}).items() if isinstance(block, dict)]
    out += [('Bootstrapper Configuration', name, block)
            for name, block in params.get('Bootstrapper Configuration', {}).items()]
    out.append(('System Parameters', 'Config', params.get('System Parameters', {})))
    calc = deals.get('Calculation') or {}
    if calc:
        out.append(('Calculation', calc.get('Object'), calc))
    if isinstance(calc.get('Mapping'), str) and os.path.isfile(calc['Mapping']):
        with open(calc['Mapping'], encoding='utf-8') as f:
            out.append(('Mapping', None, json.load(f)))
    return out


def stated_at(block, path):
    """Every value `block` states at `path`, a table stepping into each of its rows."""
    nodes = [block]
    for key in path:
        nodes = [v for n in nodes for x in (n if isinstance(n, list) else [n]) if isinstance(x, dict)
                 for v in (x.values() if key == '*' else [x[key]] if key in x else [])]
    return [v for n in nodes for v in (n if isinstance(n, list) else [n])]


def observe(cfg, simulated=()):
    """Fold one document into the running test: what it states per row, and the `MUST`s it
    meets."""
    stated = defaultdict(set)
    for section, kind, block in sections(cfg):
        for row_id, path in STATE['index'].get((section, kind), []) + (
                STATE['index'].get((section, None), []) if kind is not None else []):
            row = STATE['rows'][row_id]
            values = [v for v in stated_at(block, path) if v not in (None, '') or '' in row.admitted]
            if not values and row.default is not None:
                values = [row.default]
            TEST['present'].add(row_id)
            stated[row_id].update(str(v) for v in values)
    for row_id, values in stated.items():
        TEST['stated'][row_id] |= values
    for i, (m, rows) in enumerate(STATE['must']):
        if any(m.value in stated[r] for r in rows) and m.simulated in {None} | set(simulated):
            TEST['satisfied'].add(i)
    TEST['seen'] += 1


def fold():
    """Fold the finished test into the census."""
    rows = STATE['rows']
    for row_id, row in enumerate(rows):
        if row.placed == 'type':
            reached = row_id in TEST['present']
        elif row.reader in TEST['ran']:
            reached = bool(TEST['seen'])
            SEEN['blind'][row_id] += not reached
        else:
            reached = False
        if reached:
            SEEN['reached'].add(row_id)
            SEEN['values'][row_id] |= TEST['stated'].get(row_id, set())
    for i, (m, _) in enumerate(STATE['must']):
        if m.reader in TEST['ran']:
            SEEN['exercised'].add(i)
            if i in TEST['satisfied']:
                SEEN['met'].add(i)
    SEEN['documents'] += TEST['seen']
    reset()


def code_of(key):
    """The code object a reader key names: a module function, a method, a class's own."""
    module, *names = key.split('.')
    obj = importlib.import_module('derivus' if module == '__init__' else 'derivus.' + module)
    for name in names:
        obj = vars(obj)[name] if isinstance(obj, type) else getattr(obj, name)
    return getattr(obj, '__func__', getattr(obj, 'fget', obj)).__code__


def census():
    """`[(row, values, holes, blind)]` for every row the selection reached or was blind to, the
    tests it was blind in, or None for a store no seam states."""
    out = []
    for row_id, row in enumerate(STATE['rows']):
        blind = SEEN['blind'].get(row_id, 0) if row.where[0][0] else None
        if row_id in SEEN['reached'] or blind != 0:
            values = SEEN['values'][row_id]
            out.append((row, sorted(values), [v for v in row.admitted if v not in values], blind))
    return out


# ======================================================================================
# pytest plugin
# ======================================================================================

def pytest_addoption(parser):
    g = parser.getgroup('declared values')
    g.addoption('--values-census', action='store_true', help='run the admitted-values census')
    g.addoption('--values-strict', action='store_true',
                help='also fail on a hole, not only an unmet MUST')
    g.addoption('--values-json',
                default=os.path.join(tempfile.gettempdir(), 'derivus_values_census.json'))


def pytest_configure(config):
    if not config.getoption('values_census'):
        return
    import derivus
    from derivus import schema
    # an editable install of another checkout would place nothing and read as a selection
    # reaching no reader - a silent zero, not an error
    assert os.path.normcase(derivus.__file__).startswith(os.path.normcase(reach.ROOT)), \
        derivus.__file__
    STATE['rows'] = rows = build_rows(reach.Graph(reach.ROOT), schema.mapping)
    for row_id, row in enumerate(rows):
        for section, kind, path in row.where:
            STATE['index'].setdefault((section, kind), []).append((row_id, path))
    STATE['must'] = [(m, [i for i, r in enumerate(rows) if r.placed == 'type' and r.field == m.field])
                     for m in MUST]
    STATE['codes'] = {code_of(r.reader): r.reader for r in rows if r.placed == 'function'}
    reset()

    mon = sys.monitoring
    tool = next(i for i in range(6) if mon.get_tool(i) is None)
    mon.use_tool_id(tool, 'derivus-values')

    def on_start(code, _offset):
        TEST['ran'].add(STATE['codes'][code])
        return mon.DISABLE                              # once a test; `restart_events` re-arms
    mon.register_callback(tool, mon.events.PY_START, on_start)
    for code in STATE['codes']:
        mon.set_local_events(tool, code, mon.events.PY_START)

    def wrap(fn, run):
        def inner(context, *a, **kw):
            out = fn(context, *a, **kw)
            calc = out[0] if run and isinstance(out, tuple) else None
            try:
                observe(getattr(context, 'current_cfg', context),
                        {f.type for f in getattr(calc, 'stoch_factors', {})})
            except Exception as e:                      # a census must never fail a real gate
                SEEN['errors'].append('{}: {!r}'.format(STATE['node'], e))
            return out
        return inner

    derivus.Context.load_json = wrap(derivus.Context.load_json, False)
    for name in ('run_baseval', 'run_cmc', 'run_hedgemontecarlo', 'run_simm', 'run_saccr'):
        setattr(derivus, name, wrap(getattr(derivus, name), True))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if STATE['rows']:
        fold()
        sys.monitoring.restart_events()
    STATE['node'] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption('values_census'):
        return
    fold()
    rows = census()
    path = session.config.getoption('values_json')
    with open(path, 'w') as f:
        json.dump([{'reader': r.reader, 'field': r.field, 'admitted': r.admitted, 'stated': v,
                    'holes': h, 'blind': b} for r, v, h, b in rows], f, indent=1)
    holes = [(r, h) for r, _, h, b in rows if h and b is not None]
    print('\n\nADMITTED-VALUES CENSUS  ({} documents seen, {} of {} rows reached, rows -> {})\n'
          .format(SEEN['documents'], len(SEEN['reached']), len(STATE['rows']), path))
    for r, v, h, b in rows:
        if b is not None and (h or b):
            print('  {:<58} {:<26} holes {}{}'.format(r.reader, r.field, h, '' if not b else
                  '  BLIND in {} test(s): it ran where no document was seen'.format(b)))
    unseen = sorted({r.reader.split('.')[0] for r, _, _, b in rows if b is None})
    print('\nBLIND whole, no seam stating them: {}'.format(', '.join(
        '{} ({} rows)'.format(s, sum(r.reader.split('.')[0] == s for r, *_ in rows)) for s in unseen)))
    musts = STATE['must']
    unmet = [i for i in SEEN['exercised'] if i not in SEEN['met']]
    print('\nholes: {} on {} rows; unmet MUST: {} of {} ({} not exercised by this selection)'.format(
        sum(len(h) for _, h in holes), len(holes), len(unmet), len(musts),
        len(musts) - len(SEEN['exercised'])))
    for i in sorted(unmet):
        m = musts[i][0]
        print('  UNMET  {} {}={}{}\n         {}'.format(m.reader, m.field, m.value, (
            ' in a run simulating ' + m.simulated) if m.simulated else '', m.why))
    for error in SEEN['errors']:
        print('  UNREAD  {}'.format(error))
    if unmet or (holes and session.config.getoption('values_strict')):
        session.exitstatus = 1


if __name__ == '__main__':
    sys.path.insert(0, reach.ROOT)
    args = sys.argv[1:] or ['tests/']
    raise SystemExit(pytest.main(['-q', '--values-census'] + args, plugins=[sys.modules[__name__]]))
