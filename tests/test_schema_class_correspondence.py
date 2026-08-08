"""`fields.mapping` is a third store of field knowledge, keyed by name by CONVENTION, and nothing
checks that convention against the classes it describes.

The engine never reads `fields.py` - `construct_instrument` takes the raw JSON - so every drift here
is invisible to the valuation suite and shows up only as an authoring failure: a deal type the UI
offers but `globals()` cannot dispatch (logged, then dropped as `{}`), a field a pricer reads but no
schema-authored deal can emit, a widget that reads a JSON key nobody writes.

Each test below is one leg of the class<->schema correspondence, and each exemption list names WHY
a case is deliberate. `ALIASED_KEYS` used to be the interesting one - every field whose descriptor
key had to be invented because a name-keyed dict cannot give two deals different descriptors for
the same name. It is empty: Instrument is keyed per SECTION and every other store per TYPE, so no
descriptor has to invent a key any more.
"""
import ast
import inspect
import pathlib
import re

import pytest

from derivus import (bootstrappers, calculation, fields, instruments, riskfactors, stochasticprocess,
                      utils)

MAPPING = fields.mapping
INSTRUMENT = MAPPING['Instrument']

# Deal subclasses that legitimately have no schema row of their own.
UNDECLARED_DEALS = {
    'Deal': 'abstract base',
    'FXEuropeanOption': 'alias subclass of FXOptionDeal - no own fields, authored as the parent',
    'StructuredDealBreakClause': 'alias subclass of StructuredDeal - no own fields',
}

# Descriptor keys that are NOT the JSON field name, per store -> the key they actually read.
# A name-keyed dict admits one descriptor per name, so a field needing different valid values in
# two deals must invent a key and carry the real name elsewhere. It is EMPTY now: every store that
# holds descriptors is keyed per type or per section, so a 2D `Surface` and a 3D one each hold
# their own, a scalar `Sigma` no longer has to be filed as `sigma` beside a curve one, and the last
# entry went with the Calibration store's `Number_PCA_Factors`, which no calibration read.
ALIASED_KEYS = {}


def deal_classes():
    return {n: c for n, c in vars(instruments).items()
            if inspect.isclass(c) and issubclass(c, instruments.Deal)}


def declared_fields(deal_type):
    """Every field name a schema-authored deal of this type can carry."""
    return {f for section in INSTRUMENT['types'][deal_type] for f in INSTRUMENT['sections'][section]}


def test_every_declared_deal_type_is_dispatchable():
    """`construct_instrument` does `globals().get(param['Object'])`, and on a miss LOGS AN ERROR and
    returns `{}` - the deal vanishes from the portfolio rather than raising. So a schema type naming
    no class is a deal the UI offers, the docs document, and the engine silently drops.

    `SwapBasisDeal` and `SwapCurrencyDeal` sat in exactly that state, offered under two menus with
    128 descriptors between them and no class in this repo or the one it came from. A type is now a
    class that declares `fields`, so the state is unreachable rather than merely absent."""
    undispatchable = sorted(set(INSTRUMENT['types']) - set(deal_classes()))
    assert not undispatchable, (
        f'schema offers deal types that instruments.globals() cannot construct: {undispatchable}')


def test_every_deal_class_is_declarable():
    """The converse: a Deal subclass with no schema row cannot be authored from the UI or the docs
    at all, however well the pricer works."""
    missing = sorted(set(deal_classes()) - set(INSTRUMENT['types']) - set(UNDECLARED_DEALS))
    assert not missing, f'Deal subclasses no schema can author: {missing}'


def test_every_group_member_is_a_declared_type():
    """`groups` drives the UI's create-deal context menu. A member naming no declared type offers a
    deal the panel cannot build. This caught a missing comma between two adjacent entries, which
    Python concatenated into `FXSwapDealMtMCrossCurrencySwapDeal` - removing BOTH from the menu."""
    unknown = {name: [m for m in members if m not in INSTRUMENT['types']]
               for name, members in INSTRUMENT['groups'].items()}
    unknown = {k: v for k, v in unknown.items() if v}
    assert not unknown, f'context-menu entries naming no declared type: {unknown}'


def test_factor_fields_are_declared():
    """`factor_fields` is how discovery finds a deal's price factors, so every key in it is a field
    the deal genuinely reads. One absent from the schema is the `Barrier_Price` defect: the pricer
    reads it with a hard key while no schema-authored deal can emit it."""
    undeclared = [(n, k) for n, c in sorted(deal_classes().items())
                  if n in INSTRUMENT['types']
                  for k in c.__dict__.get('factor_fields', {})
                  if (k if isinstance(k, str) else k[0]) not in declared_fields(n)]
    assert not undeclared, f'factor references no schema-authored deal can carry: {undeclared}'


@pytest.mark.parametrize('mapping_key', sorted(k for k, v in MAPPING.items() if 'fields' in v))
def test_aliasing_is_declared_not_inferred(mapping_key):
    """The UI used to recover the JSON key by reconstructing it from the DESCRIPTION
    (`description.replace(' ', '_')`, six sites), which made that one string load-bearing as a key
    and free text at the same time: prose written into it pointed the widget at a key nobody writes,
    so the field silently showed its default and edits landed somewhere else. `name` now carries the
    key explicitly and `description` is free text.

    Aliasing is legitimate but must be DECLARED, so this holds `ALIASED_KEYS` to exactly the set
    that needs it - an alias added later without a listing fails here rather than going quiet."""
    aliased = ALIASED_KEYS.get(mapping_key, {})
    for key, meta in MAPPING[mapping_key]['fields'].items():
        assert meta.get('name', key) == aliased.get(key, key), (
            f'{mapping_key}.{key} reads {meta.get("name", key)!r}; ALIASED_KEYS expects '
            f'{aliased.get(key, key)!r}')


def declared_market_factor_types():
    """The `Market Prices` type strings the schema publishes, which a bootstrapper now declares
    with `market_factor_type` and selects its work by."""
    return set(MAPPING['MarketPrices']['types'])


def test_no_bootstrapper_owns_a_market_price_literal():
    """This replaces the pair of gates that held the declared types to the literals the engine
    matched, in both directions. A bootstrapper selects work by `market_factor.type == <X>` with
    no else, so a schema name matching no literal was a quote block the engine walked straight
    past - the config looked authored, the calibration never ran, and only a downstream `wrote no
    *.* price factor` error hinted at it. The drift had reached the published docs.

    The type is now declared once, as `market_factor_type`, and the engine compares against that
    attribute - so the two cannot disagree and the old gates would be tautologies. What is
    gateable is the discipline that makes them tautologies, which is this: no bootstrapper owns
    the text. Same shape as `test_instruments_call_resolvers_not_factor_types`, and it is why
    `HullWhite2FactorModelParameters` no longer sets its type in `__init__`."""
    offenders = []
    for node in ast.walk(ast.parse(inspect.getsource(bootstrappers))):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute) \
                and node.left.attr == 'type':
            offenders += [f'{c.value} at line {node.lineno}' for c in node.comparators
                          if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    assert not offenders, f'bootstrappers comparing market_factor.type to a literal: {offenders}'


def test_every_market_price_quote_instrument_is_a_declared_deal():
    """A quote is a reference to an EXISTING instrument type, so the `Instrument` store's
    declarations ARE the quote's schema and the family only names them. A name naming no deal type
    is a quote nothing can author - and the names are what a `DealType` dropdown offers."""
    unknown = {cls.market_factor_type: sorted(set(cls.quote_instruments) - set(INSTRUMENT['types']))
               for cls in vars(bootstrappers).values()
               if isinstance(cls, type) and 'quote_instruments' in cls.__dict__}
    unknown = {k: v for k, v in unknown.items() if v}
    assert not unknown, f'quote instruments naming no declared deal type: {unknown}'


RETIRED_VOL_TYPES = ('FXVol', 'EquityPriceVol', 'CommodityPriceVol')


def test_the_retired_vol_types_stay_retired():
    """Three empty `Factor2D` subclasses whose bodies differed only in a docstring, and three schema
    declarations that had drifted apart - FX and commodity could not author the SVI/Skew surfaces
    `Factor2D.get_subtype` has always supported, and only commodity declared Currency.

    What varies is the SUBTYPE, not the asset class: asset class belongs to the UNDERLYING, whose
    types (FxRate / EquityPrice / CommodityPrice) stay distinct and are what the bootstrapper reads
    it from. One `VolatilityGrid` replaces all three, in the classes, the schema, the discovery
    registries and the deals' `factor_fields`."""
    assert hasattr(riskfactors, 'VolatilityGrid')
    back = [n for n in RETIRED_VOL_TYPES if hasattr(riskfactors, n)]
    assert not back, f'retired vol classes are back in riskfactors: {back}'

    declared = {n for n in RETIRED_VOL_TYPES
                if n in MAPPING['Factor']['types'] or n in MAPPING['Process_factor_map']}
    assert not declared, f'retired vol types are back in the schema: {sorted(declared)}'

    referenced = sorted({f'{n}.{f}' for n, c in deal_classes().items()
                         for f, types in getattr(c, 'factor_fields', {}).items()
                         if set(types) & set(RETIRED_VOL_TYPES)})
    assert not referenced, f'deals still reference a retired vol type: {referenced}'
    assert utils.TwoDimensionalFactors == ['VolatilityGrid'], (
        f'the 2D factor registry disagrees: {utils.TwoDimensionalFactors}')


def json_names(cls):
    """The keys an author writes for this deal - the alias where a field has one, plus the children
    of every container, since those are authored inside it."""
    out = set()

    def walk(f):
        out.add(f.json_name or f.name)
        for child in (f.sub_fields or []):
            walk(child)
    for group in getattr(cls, 'fields', []):
        for f in group.fields:
            walk(f)
    return out


@pytest.mark.parametrize('deal_type', sorted(
    n for n, c in deal_classes().items() if getattr(c, 'fields', None)))
def test_every_field_a_deal_reads_is_declarable(deal_type):
    """`test_factor_fields_are_declared` checks the same thing for factor REFERENCES; this is every
    field. A deal that hard-keys `self.field['X']` while no declaration offers X cannot be authored
    from the schema at all - `SwaptionDeal` read `Swap_Effective_Date` in `reset` with no guard, so
    a schema-authored swaption raised KeyError before it could price, and `EquityOneTouchOption`
    read `Barrier_Dates` for its discrete-monitoring path while only `EquityBarrierOption` declared
    it, putting that path out of reach of any UI.

    Hard keys only. A `.get` read is a field the deal is content to find missing, which is a
    different statement and is what `default=REQUIRED` and `validate()` are for."""
    src = ast.parse(inspect.getsource(getattr(instruments, deal_type)))
    read = {n.slice.value for n in ast.walk(src)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute)
            and n.value.attr == 'field' and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str)}
    undeclared = sorted(read - json_names(getattr(instruments, deal_type)))
    assert not undeclared, f'{deal_type} reads fields no schema-authored deal can carry: {undeclared}'


def test_instruments_call_resolvers_not_factor_types():
    """The `get_*` layer owns every factor-type text key; an instrument knows WHICH resolver to
    call, never the literal. Six sites violated that - two barrier deals built
    `utils.Factor('EquityPrice', ...)` raw (and for a composed equity name that named the dropped
    full-name head key, so the bridge lookup missed by accident rather than by the documented
    guard), three object hops in CreditNthToDefault's beta block, and one commodity component hop.

    Held exception, noted in the roadmap: `get_implied_correlation`'s two callers still build
    type-prefixed correlation-name tuples. Those are tuple literals, not Factor constructions, so
    this gate does not cover them - it pins Factor construction and raw opcode calls to zero."""
    src = ast.parse(inspect.getsource(instruments))
    offenders = []
    for cls in [n for n in src.body if isinstance(n, ast.ClassDef)]:
        for call in [n for n in ast.walk(cls) if isinstance(n, ast.Call)]:
            f = call.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
            literal = call.args and isinstance(call.args[0], ast.Constant)
            if (name == 'Factor' and literal) or name in ('calc_factor_index',
                                                          'calc_factor_code_chain'):
                offenders.append(f'{cls.name}:{call.lineno}')
    assert not offenders, f'instrument methods owning factor-type text: {offenders}'


def test_accepts_children_matches_what_the_class_does_with_them():
    """A deal that breaks down into simpler instruments prices them in `post_process`; a leaf has
    no `post_process` at all. `accepts_children` is the DECLARATION of that, and this holds it to
    the implementation in both directions.

    It has to be declared rather than inferred, because it is what a UI asks before offering a
    node children - and it lived in the wrong place: the create-menu's jsTree node kind, where only
    `New Structure` was a folder. So `CapDeal`, `FloorDeal`, `SwapInterestDeal`, `SwaptionDeal` and
    `MtMCrossCurrencySwapDeal` were all files, and the Workbench could not build any of them with
    the legs their `post_process` prices. The engine never cared - it recurses on `Children` being
    present - which is exactly why nothing failed and no test noticed."""
    prices_children = {n for n, c in deal_classes().items() if 'post_process' in c.__dict__}
    declares = {n for n, c in deal_classes().items() if getattr(c, 'accepts_children', False)}

    # alias subclasses inherit both the method and the declaration, so compare on own-attr for the
    # implementation and let inheritance carry the declaration
    inherited = {n for n in declares - prices_children
                 if any('post_process' in b.__dict__ for b in deal_classes()[n].__mro__[1:])}
    assert declares - prices_children - inherited == set(), (
        f'declared a container but prices no children: {sorted(declares - prices_children - inherited)}')
    assert prices_children - declares == set(), (
        f'prices children but is not declared a container: {sorted(prices_children - declares)}')


def test_discountrate_stays_retired():
    """`DiscountRate` was a factor type whose entire body was one field pointing at another factor,
    and `get_discount_factor` existed only to follow it: look up the wrapper, ask it for the name,
    resolve THAT as an InterestRate. Every block in the repo mapped a name to itself.

    The tell that it was redundant is that `add_rates_for_factor` self-healed this type and no
    other - it could synthesise the block precisely because the block held nothing the engine could
    not already derive. A deal's `Discount_Rate` field now names an InterestRate directly, which
    loses no expressiveness: discounting on a different curve was always done by naming one."""
    assert not hasattr(riskfactors, 'DiscountRate'), 'the wrapper class is back'
    assert not hasattr(instruments, 'get_discount_factor'), 'the name hop is back'
    assert 'DiscountRate' not in MAPPING['Factor']['types'], 'back in the Factor store'
    assert 'DiscountRate' not in MAPPING['Process_factor_map'], 'back in the process map'
    assert 'DiscountRate' not in utils.DimensionLessFactors, 'back in the dimensionless registry'

    referenced = sorted({f'{n}.{f}' for n, c in deal_classes().items()
                         for f, types in getattr(c, 'factor_fields', {}).items()
                         if 'DiscountRate' in types})
    assert not referenced, f'deals still reference the retired type: {referenced}'


def test_docs_publish_only_real_market_price_names():
    """The drift reached the DOCS, which is where it did its damage: a reader copying the example
    authored a block the engine walks past. Neither the schema nor the suite looked at prose, so the
    two pages disagreed with the engine, and with each other, for as long as they existed."""
    docs = pathlib.Path(__file__).parent.parent / 'docs_src'
    known = declared_market_factor_types()
    published = {(p.relative_to(docs), n) for p in docs.rglob('*.md')
                 for n in re.findall(r'\b\w*ModelPrices\b', p.read_text())}
    unknown = sorted(f'{p}: {n}' for p, n in published if n not in known)
    assert not unknown, f'docs publish market-price names the engine never matches: {unknown}'
