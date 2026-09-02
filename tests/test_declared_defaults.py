"""A declared default reaches the deal, and nothing else moves.

`Deal.__init__` took the authored block verbatim, so a `fields` declaration's `default=` was
schema-only: a pricer reading an unauthored field BY NAME raised `KeyError` inside
`calc_dependencies`, the deal was logged and SKIPPED, and the job succeeded with the deal priced at
nothing. The seam is `schema.DealFields`, which `Deal.__init__` wraps the authored block in: a read
by name falls through to `schema.deal_defaults`, and nothing else does.

COMPLETION IS AN ALLOWLIST, `schema.COMPLETABLE`, by design. A declared default is what a blank
panel SHOWS, not what the engine means: `FXBarrierOption` marks none of its economic fields
REQUIRED, so answering every read by name would replace a LOUD failure with a schema-invalid block
pricing at a plausible wrong number - measured one dropped field at a time, 741.53 for the strike
and the Down_And_In value 6.37 for a Down_And_Out, against 78.93 for the deal the author meant.
Only fields whose declared value IS the engine's own fallback are on the list, and each is gated.

THE ROADMAP ROW'S OWN SHAPE, measured both ways on one document: a block omitting
`Barrier_Monitoring_Frequency` and `Cash_Rebate` used to leave NO row in `Results['mtm']` and value
the book at 0. It prices now, on all four spellings, BIT-IDENTICAL to the same block furnishing
`0M` and a zero rebate explicitly - which is the acceptance: a default is what the author would
have written, not a second pricer. Suppressing the completion takes all four back to no row at all.

WHAT A DEFAULT DOES NOT DO IS ENTER THE PROGRAM. `get`, `in`, iteration, `len` and the JSON round
trip see exactly the authored keys, so `plan_hash`, the factor universe and a saved book are
byte-identical across every job document in the tree, hashes pinned below. That split is the
design: `field[key]` is the read that used to raise and the declaration answers it, while
`get(key, fallback)` is the READER's own statement of what an omitted field means.

A DEFAULT IS CONVERTED, not copied. A declaration is authoring metadata - a blank Table is the
string `'null'`, a Period `'3M'`, a rate a whole number of percent - so `schema.engine_default`
converts each the way the loader converts that field's wire form. Uncoerced, `'0M'` reaches
`base_date + self.field[...]` as a str and the repair swaps one skip for another.

DEGENERACY CHECKLIST for the barrier gate: r = 4% against q = 2% (non-zero, r != q), both knock
directions and both option types, and the repaired reading compared against the furnished one
rather than against zero.
"""
import copy
import glob
import json
import os
import pickle
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

import derivus
from derivus import instruments, schema, structures, utils
from derivus.config import Config, CustomJsonEncoder
from derivus.instruments import construct_instrument
from derivus.schema import REQUIRED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures')

BASE = pd.Timestamp('2024-06-28')
X0, R_USD, R_EUR, SIGMA = 1.25, 0.04, 0.02, 0.15
NOTIONAL, EXPIRY_D = 1000.0, 365

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': X0},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_EUR], [5.0, R_EUR]])},
    'FXVol.EUR.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                      'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.6, 1.0, 1.4)
                                                  for t in (0.02, 2.0)])}}

#: The two fields `pv_barrier_option` asks for by name, in the WIRE form a document carries them.
#: An author omitting these is the roadmap row's own example.
FURNISHED = {'Barrier_Monitoring_Frequency': {'.DateOffset': '0M'}, 'Cash_Rebate': 0.0}

#: `(Barrier_Type, barrier, strike, option type)` - both directions and both option types, each
#: barrier struck on the live side of its own axis.
BARRIERS = [('Down_And_Out', 1.12, 1.25, 'Call'), ('Down_And_In', 1.12, 1.25, 'Call'),
            ('Up_And_Out', 1.40, 1.30, 'Put'), ('Up_And_In', 1.40, 1.30, 'Put')]

#: The plan and the factor universe of every job document under `tests/fixtures`, as they read
#: BEFORE declared defaults reached a deal. `(plan_hash, resolved factors, missing factors)` - a
#: default entering the program moves the first and a default minting a factor moves the rest.
PINNED = {
    'autocall_job.json': (
        'a5f6560a00df96fd7f0a5b5b1087e60495c6321517431c54e6afcb13931f5d73', 5, 0),
    'commodity_aps_world.json': (
        '1847aa7c7053b57255426bee4aae1da44954b955b52116a6a14d5ad26be6b07f', 6, 0),
    'fx_accumulator_job.json': (
        '14df269ed239e6640cb8a44a01224f11a4d265815fc7c9d564cf7bf8933badc7', 5, 0),
    'fx_tarf_job.json': (
        '0413fa7e4ff497523561e44e1ab205369d6948731061e3479fd2714a099f93d1', 5, 0),
    'platinum_hedge_shipping.json': (
        'bbdbb1deabdb17e91f00bc1a627a50cc22c8ccce35cf65f8cddfb6cd67c2b037', 2, 0),
    'policy_test_simulate_only.json': (
        '7d58d418ce8a4ce3eea75d4edfb69298c3a0065011bee26bfe690f2c142d9a85', 2, 0),
}

#: The deal types whose CONSTRUCTOR writes into the authored block - `setdefault` calls that
#: predate the declarations. They still write, so every existing document's plan is unmoved; they
#: are what "now redundant" means at the class level.
CONSTRUCTOR_WRITES = {
    'NettingCollateralSet': {'Settlement_Period', 'Liquidation_Period', 'Opening_Balance'}}


def deal_classes():
    """Every constructible deal type, by name."""
    return sorted(
        (name, cls) for name, cls in vars(instruments).items()
        if isinstance(cls, type) and issubclass(cls, instruments.Deal) and getattr(cls, 'fields', None))


def declared(cls):
    """`{key: F}` for every field `cls` declares a default for, inherited declarations included."""
    return {f.key: f for group in (getattr(cls, 'fields', []) or []) for f in group.fields
            if f.default is not REQUIRED and f.default is not None}


def barrier_deal(barrier_type, barrier, strike, option_type, **extra):
    block = {'Object': 'FXBarrierOption', 'Reference': 'BR', 'Currency': 'USD',
             'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
             'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': option_type,
             'Strike_Price': strike, 'Barrier_Price': barrier, 'Barrier_Type': barrier_type,
             'Underlying_Amount': NOTIONAL,
             'Expiry_Date': {'.Timestamp': (BASE + pd.DateOffset(days=EXPIRY_D)).strftime('%Y-%m-%d')}}
    block.update(extra)
    return block


def price(block):
    """One deal through the JSON contract; its own row of `Results['mtm']`, not the book total."""
    job = {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                        'MCMC_Simulations': 1, 'Random_Seed': 1},
        'Deals': {'Tag_Titles': '', 'Reference': 'defaults',
                  'Deals': {'Children': [{'Instrument': {'.Deal': block}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': {}, 'Price Factors': FACTORS}}}}
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'defaults'))
    _, out = cx.run_job()
    rows = out['Results']['mtm']
    own = rows[rows['Reference'] == 'BR']['Value']
    # a skipped deal leaves NO row of its own - the failure mode this gate exists for
    assert len(own) == 1, 'the deal left no row in Results: it was skipped, not priced'
    return float(own.iloc[0])


# --------------------------------------------------------------------------------------------
# the repair
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('barrier_type,barrier,strike,option_type', BARRIERS)
def test_a_barrier_omitting_the_two_declared_fields_prices_instead_of_skipping(
        barrier_type, barrier, strike, option_type):
    """`pv_barrier_option`'s own block, both ways: the declaration is what the author would have
    written, so the two readings agree to the bit and neither is zero."""
    furnished = price(barrier_deal(barrier_type, barrier, strike, option_type, **FURNISHED))
    omitted = price(barrier_deal(barrier_type, barrier, strike, option_type))
    assert omitted == furnished, (barrier_type, omitted, furnished)
    assert abs(furnished) > 1.0, 'the fixture must have something to lose'


def test_the_repaired_deal_reads_the_engine_form_of_both_defaults():
    """Uncoerced, `'0M'` is a str and `base_date + str` is the next skip. The monitoring frequency
    arrives as a `DateOffset` of zero months and the rebate as the declared zero."""
    deal = construct_instrument(dict(barrier_deal(*BARRIERS[0]), Expiry_Date=BASE), {})
    frequency = deal.field['Barrier_Monitoring_Frequency']
    assert (BASE + frequency - BASE).days == 0
    assert isinstance(frequency, pd.DateOffset)
    assert deal.field['Cash_Rebate'] == 0


# --------------------------------------------------------------------------------------------
# a default answers a read; it does not enter the program
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name,cls', deal_classes())
def test_a_default_answers_a_read_and_never_enters_the_program(name, cls):
    """The seam itself, over every deal type: a COMPLETABLE default answers `field[key]` and every
    other declared default still raises, while the dict holds only what was authored - which is
    what `plan_hash`, `get_fieldname` and the JSON round trip read. Read off `DealFields` rather
    than a constructed deal: 21 classes cannot be built from `Object` alone, by the same KeyError
    this allowlist deliberately leaves in place.
    """
    block = {'Object': name}
    field = schema.DealFields(dict(block), cls)

    assert set(field) == set(block), 'a default entered the program'
    assert json.loads(json.dumps(field, cls=CustomJsonEncoder)).keys() == field.keys()

    answered = 0
    for key in declared(cls):
        if key in block:
            continue
        assert key not in field, '{}: a default answered a membership test'.format(key)
        assert field.get(key) is None, '{}: a default displaced a get() fallback'.format(key)
        if key in schema.COMPLETABLE:
            field[key]                                      # answers rather than raising
            answered += 1
        else:
            with pytest.raises(KeyError):
                field[key]

    assert set(field) == set(block), 'a read wrote the default into the block'
    assert answered == len(set(declared(cls)) & schema.COMPLETABLE)

    with pytest.raises(KeyError):
        field['A_Field_No_Declaration_Names']


def test_the_deal_constructor_is_where_the_seam_is_wired():
    """`Deal.__init__` wraps the authored block, so the completion travels with the deal rather
    than being applied at one reader."""
    deal = construct_instrument(dict(barrier_deal(*BARRIERS[0])), {})
    assert isinstance(deal.field, schema.DealFields)
    assert set(deal.field) == set(barrier_deal(*BARRIERS[0]))
    assert deal.field['Cash_Rebate'] == 0 and 'Cash_Rebate' not in deal.field


@pytest.mark.parametrize('name,cls', [(n, c) for n, c in deal_classes()
                                      if set(declared(c)) & schema.COMPLETABLE])
def test_a_completed_default_is_the_deal_s_own(name, cls):
    """A completion is deep-copied per deal: `DateList.consume` mutates, and two deals of one type
    sharing a declaration's object would consume each other's fixings. Read on the fields that are
    actually completable, which is where the copy is the only thing standing between them."""
    keys = [key for key in declared(cls) if key in schema.COMPLETABLE]
    one, two = (schema.DealFields({'Object': name}, cls) for _ in range(2))
    store = schema.deal_defaults(cls)
    for key in keys:
        assert one[key] is one[key], '{}: a read must be stable'.format(key)
        assert one[key] == store[key], '{}: the completion is not the declaration'.format(key)
        if not isinstance(store[key], (int, float, str, bytes, bool)):
            # an immutable scalar may legitimately be interned; a container may not be shared
            assert one[key] is not store[key], '{}: the class store was handed out'.format(key)
            assert one[key] is not two[key], key


def test_no_declared_default_can_mint_a_price_factor():
    """Discovery reads the raw block through `get_fieldname`, which drops a blank - so a default
    landing on a factor-naming field has to BE blank, or a deal would name a curve nobody loaded."""
    minting = []
    for name, cls in deal_classes():
        defaults = schema.deal_defaults(cls)
        for key in getattr(cls, 'factor_fields', {}) or {}:
            head = key[0] if isinstance(key, tuple) else key
            value = defaults.get(head)
            if isinstance(key, tuple):
                # a nested field is walked one level at a time and each level is dropped on falsy
                value = (value or {}).get(key[1]) if len(key) > 1 else value
            if value:
                minting.append('{}.{}={!r}'.format(name, key, value))
    assert not minting, minting


# --------------------------------------------------------------------------------------------
# the engine form of a declared default is the loader's own
# --------------------------------------------------------------------------------------------
def test_a_period_default_parses_through_the_grammar_the_loader_uses():
    """Every Period a deal declares, against `Config.parse_period` - the one spelling of that
    parse, and the form `{'.DateOffset': ...}` decodes to."""
    config = Config()
    seen = 0
    for _, cls in deal_classes():
        for key, field in declared(cls).items():
            if field.obj != 'Period':
                continue
            seen += 1
            engine = schema.deal_defaults(cls)[key]
            assert engine.kwds == config.parse_period(field.default).kwds, (cls.__name__, key)
    assert seen >= 20, 'the Period declarations went somewhere'


def test_a_table_default_is_the_empty_container_its_tag_names():
    """`'null'` is what a widget writes for an empty table; the engine reads a `utils` container or
    a list, and never the four characters."""
    kinds = {'DateList': utils.DateList, 'DateEqualList': utils.DateEqualList,
             'CreditSupportList': utils.CreditSupportList, 'DateValueList': list, None: list}
    seen = 0
    for _, cls in deal_classes():
        for key, field in declared(cls).items():
            if field.type != 'Table':
                continue
            seen += 1
            value = schema.deal_defaults(cls)[key]
            assert isinstance(value, kinds[field.tag]), (cls.__name__, key, field.tag)
            assert not (value.data if hasattr(value, 'data') else value), (cls.__name__, key)
    assert seen >= 30, 'the Table declarations went somewhere'


def test_a_rate_default_carries_its_unit():
    """A `Percent`/`Basis` declaration is a whole number of percent or of basis points, and the
    engine reads `.amount`."""
    for _, cls in deal_classes():
        for key, field in declared(cls).items():
            if field.obj not in ('Percent', 'Basis'):
                continue
            value = schema.deal_defaults(cls)[key]
            assert isinstance(value, utils.Percent if field.obj == 'Percent' else utils.Basis)
            assert float(value) == field.default / value.divisor


# --------------------------------------------------------------------------------------------
# nothing else moved
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('filename', sorted(PINNED))
def test_a_job_document_keeps_its_plan_hash_and_its_factor_universe(filename):
    """The HARD acceptance, per document: the program and the want-list are what they were."""
    context = derivus.Context()
    context.load_json(os.path.join(FIXTURES, filename))
    universe = context.current_cfg.factor_universe()
    plan, resolved, missing = PINNED[filename]
    assert context.plan_hash() == plan
    assert (len(universe['resolved']), len(universe['missing'])) == (resolved, missing)


def test_every_deal_in_every_job_document_holds_exactly_its_authored_block():
    """The invariant behind the pinned hashes, over every document in the tree that loads: the
    constructed deal's field dict IS the `.Deal` block the file carries."""
    paths = sorted(glob.glob(os.path.join(FIXTURES, '*.json')) +
                   glob.glob(os.path.join(ROOT, 'data', '*', 'job_*.json')))
    checked = 0
    for path in paths:
        with open(path, 'rt', encoding='utf-8') as handle:
            raw = handle.read()
        if '".Deal"' not in raw:
            continue
        context = derivus.Context()
        context.load_json(path)
        authored = []
        stack = [json.loads(raw)]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if isinstance(node.get('.Deal'), dict):
                    authored.append(node['.Deal'])
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        built = list(context.current_cfg.walk_deals())
        assert len(built) == len(authored), path
        wrote = [set(block) for block in authored]
        for deal in built:
            keys = set(deal.field) - CONSTRUCTOR_WRITES.get(deal.field['Object'], set())
            assert keys in wrote, '{}: {} carries keys no author wrote - {}'.format(
                path, deal.field.get('Reference'), sorted(keys.difference(*wrote)))
            checked += 1
    assert checked, 'the sweep found no deals to check'


#: The fields of the SAME `FXBarrierOption` block whose declared default is a blank panel's value
#: and not the engine's meaning. Dropping one is a schema-INVALID document, and the value beside
#: each is what completing it would have priced.
NOT_COMPLETABLE = [('Strike_Price', 741.5344181072), ('Barrier_Type', 6.3673278714),
                   ('Buy_Sell', 78.9325214257), ('Option_Type', 78.9325214257)]


@pytest.mark.parametrize('key,silent', NOT_COMPLETABLE, ids=[k for k, _ in NOT_COMPLETABLE])
def test_an_economic_field_is_not_completed_and_does_not_price(key, silent):
    """The seam's own hazard, gated. `FXBarrierOption` marks none of its economic fields REQUIRED,
    so a blanket completion prices a strikeless deal at 741.53 and flips a Down_And_Out to its In
    at 6.37, against 78.93 for the deal the author meant - which is why COMPLETABLE is an allowlist
    rather than a guard.
    """
    block = barrier_deal('Down_And_Out', 1.12, 1.25, 'Call', **FURNISHED)
    del block[key]
    try:
        priced = price(block)
    except AssertionError:
        return                                              # skipped: no row of its own, as before
    assert priced != priced, (
        '{} was completed: the deal priced {!r} where the author meant {!r}'.format(
            key, priced, silent))


def test_a_blank_date_default_keeps_its_named_refusal():
    """`Expiry_Date` declares `''`, and a blank Date is not an absent one: completed, it reaches a
    date comparison as a `str` and the deal dies four layers down on
    `'<' not supported between instances of 'Timestamp' and 'str'` instead of naming the field."""
    block = barrier_deal('Down_And_Out', 1.12, 1.25, 'Call', **FURNISHED)
    del block['Expiry_Date']
    with pytest.raises(KeyError) as refusal:
        price(block)
    assert 'Expiry_Date' in str(refusal.value)


def test_the_runner_s_explicit_furnishing_now_agrees_with_the_declaration():
    """`structures.materialize` furnishes the same two fields by hand - the local remedy the row
    names. It is now REDUNDANT rather than wrong: the values it writes and the declaration's own
    are the same engine value, so the runner's statement costs nothing and moves nothing."""
    source = open(structures.__file__, 'rt', encoding='utf-8').read()
    assert "deal.setdefault('Cash_Rebate', 0.0)" in source
    assert "deal.setdefault('Barrier_Monitoring_Frequency', {'.DateOffset': '0M'})" in source

    defaults = schema.deal_defaults(instruments.FXBarrierOption)
    assert defaults['Cash_Rebate'] == 0.0
    assert defaults['Barrier_Monitoring_Frequency'].kwds == Config().parse_period('0M').kwds


def test_the_declaration_is_the_only_source_a_deal_default_comes_from():
    """`declared_defaults` completes a CALCULATION's params and `deal_defaults` a deal's read, off
    the same `default=`. Neither reads the other's shape, and both skip `REQUIRED`."""
    assert schema.deal_defaults(instruments.FXBarrierOption).keys() <= {
        f.key for group in instruments.FXBarrierOption.fields for f in group.fields}
    for group in instruments.FXBarrierOption.fields:
        for field in group.fields:
            if field.default is REQUIRED:
                assert field.key not in schema.deal_defaults(instruments.FXBarrierOption)


def test_a_deal_survives_the_round_trips_a_job_puts_it_through():
    """A book crosses a process boundary by pickle and a solve step by `deepcopy`; both have to
    bring the completion with them, or a forked worker prices the skip again."""
    deal = construct_instrument(dict(barrier_deal(*BARRIERS[0])), {})
    authored = dict(deal.field)
    for clone in (copy.deepcopy(deal), pickle.loads(pickle.dumps(deal))):
        assert dict(clone.field) == authored
        assert clone.field['Cash_Rebate'] == 0
        assert 'Cash_Rebate' not in clone.field
