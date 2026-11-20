// Drives `src/quote.ts` where the eye cannot: what a structure's ticket asks for off its `/schema`
// declaration, what is sent, how a quote's levels read on the market's axis, and what an
// acceptance says. Every check names the MUTATION it kills - the change to the module that would
// still typecheck, still render, and still be wrong.
//
//   node web/scripts/quote_check.mjs          # exits 1 on a miss
//
// The declarations are `/schema`'s own shape for a collar and a range accrual; the quotes are a
// collar quoted on ZARUSD - a pair the book's surface quotes the other way round - and a range
// accrual with one tier cell left to the solve.

import { check, done, load } from './check.mjs';

const quote = await load('quote');

const required = (widget, description) => ({ widget, description, value: '', required: true });
const SHARED = {
  pair: required('Text', 'The market pair, base then quote'),
  expiry: required('Text', 'Tenor dealt on the book\'s Base_Date'),
  notional: required('Float', 'The amount each leg is struck on'),
  notional_currency: required('Text', 'Which side of the pair the notional is in'),
};
const COLLAR = {
  vernacular: 'zero-cost collar, range forward, cylinder',
  fields: {
    ...SHARED,
    sell_currency: { widget: 'Text', description: 'Currency the client sells', convention: true,
                     selector: 'optional' },
    buy_currency: { widget: 'Text', description: 'Currency the client buys', convention: true,
                    selector: 'optional' },
  },
  recipe: ['Price protection', 'Solve financing.Strike_Price to -Premium(protection)'],
  variations: {
    floor: { buys: 'quote', fields: { floor: required('Float', 'The protected level') },
             legs: {} },
    cap: { buys: 'base', fields: { cap: required('Float', 'The protected level') }, legs: {} },
  },
};
const CAKE = {
  vernacular: 'range accrual, wedding cake',
  fields: {
    ...SHARED,
    premium: { widget: 'Float', description: 'What the client pays', value: 0.0,
               convention: true },
    tiers: { ...required('Table', 'One row per tier'), obj: ['Float', 'Float', 'Float'],
             col_names: ['floor', 'cap', 'coupon'] },
  },
  recipe: ['Solve the one blank parameter to premium'],
  legs: {},
};

// --- the ticket asked for
check('a structure\'s sales names are its vernacular, each said whole',
      'the names split without trimming, which offers " range forward" in the menu',
      quote.salesNames(COLLAR), ['zero-cost collar', 'range forward', 'cylinder']);

check('a pair reads however a desk types it, base then quote',
      'the separator not allowed, which reads USD/ZAR as no pair; or the case kept, which names '
      + 'a usd the variations cannot find',
      ['USD/ZAR', ' usdzar ', 'USD.ZAR', 'USDZA'].map(quote.pairSides),
      [['USD', 'ZAR'], ['USD', 'ZAR'], ['USD', 'ZAR'], undefined]);

check('a ticket fills the shared parameters bar the selectors, then the chosen form\'s own',
      'the selectors kept, which asks a salesperson for sell_currency like a level; or the '
      + 'variation\'s level dropped, which quotes a collar with no floor',
      [Object.keys(quote.ticketFields(COLLAR, 'floor')), Object.keys(quote.ticketFields(COLLAR, null))],
      [['pair', 'expiry', 'notional', 'notional_currency', 'floor'],
       ['pair', 'expiry', 'notional', 'notional_currency']]);

check('each form is offered as the currency of the pair its client buys',
      'base and quote swapped, which offers a floor as the client buying dollars',
      [quote.directions(COLLAR, 'USDZAR'), quote.directions(COLLAR, '')],
      [[{ word: 'floor', buys: 'ZAR' }, { word: 'cap', buys: 'USD' }],
       [{ word: 'floor', buys: 'the quote currency' }, { word: 'cap', buys: 'the base currency' }]]);

const collarDraft = {
  structure: 'ZeroCostCollar', variation: 'cap',
  values: { pair: 'USDZAR', expiry: '6M', notional: 1000000, notional_currency: 'USD', cap: 18.1,
            floor: 17.5, sell_currency: '' },
  nettingSet: 'ISDA-KESTREL', margin: { amount: '20000', currency: 'ZAR' },
};
check('the request states the form\'s parameters, the currency its client buys, the client and the '
      + 'margin as numbers',
      'another form\'s level sent along, which the service refuses as contradicting the cap; the '
      + 'buying selector read off the wrong side; or the margin sent as the text typed',
      quote.requestOf(collarDraft, COLLAR),
      { structure: 'ZeroCostCollar',
        params: { pair: 'USDZAR', expiry: '6M', notional: 1000000, notional_currency: 'USD',
                  cap: 18.1, buy_currency: 'USD' },
        netting_set: 'ISDA-KESTREL', margin: { amount: 20000, currency: 'ZAR' } });

check('a blank states nothing: no premium, no client and no margin travel where none was typed',
      'a blank kept, which sends premium "" - a premium nobody named - and an empty netting set',
      quote.requestOf({ structure: 'RangeAccrual', variation: null,
                        values: { pair: 'USDZAR', premium: '', tiers: [{ floor: 17, cap: 18 }] },
                        nettingSet: '', margin: { amount: ' ', currency: '' } }, CAKE),
      { structure: 'RangeAccrual', params: { pair: 'USDZAR', tiers: [{ floor: 17, cap: 18 }] } });

check('the recipe says which leg\'s level it solves, and to what, off its declared steps',
      'the target\'s sign dropped, which tells a desk the financing leg is worth the protection '
      + 'premium rather than minus it; or a parameter solve left unsaid',
      [...quote.solving(COLLAR), ...quote.solving(CAKE),
       ...quote.solving({ recipe: ['Price protection', 'Price participation',
                                   'Solve financing.Strike_Price to -Premium(protection) '
                                   + '-Premium(participation)', 'Solve tarf.Strike_Price to 0'] })],
      ["The quote solves the financing leg's strike price, so that leg is worth minus the "
       + 'protection premium.',
       'The quote solves the one blank parameter to premium.',
       "The quote solves the financing leg's strike price, so that leg is worth minus the "
       + 'protection premium minus the participation premium.',
       "The quote solves the tarf leg's strike price, so that leg is worth nothing."]);

// --- the clients
const set = (reference, children = []) => ({
  Instrument: { '.Deal': { Object: 'NettingCollateralSet', Reference: reference } },
  Children: children });
const DOC = { Calc: { Calculation: {}, Deals: { Deals: { Children: [
  set('ISDA-ZULU'), { Instrument: { '.Deal': { Object: 'StructuredDeal', Reference: 'S' } },
                      Children: [set('ISDA-NESTED')] }] } } } };
const AGREEMENTS = [
  { agreement: 'ISDA-KESTREL', entity: 'KESTREL', kind: 'ISDA',
    terms: { Object: 'NettingCollateralSet' } },
  { agreement: 'ISDA-ZULU', entity: 'ZULU', kind: 'ISDA', terms: { Object: 'NettingCollateralSet' } },
];
check('a quote may be for any set the book holds, however deep, or any the record declares',
      'the top level walked alone, which loses a set nested in a container; or the paper ignored, '
      + 'which hides a client onboarded before its first trade',
      quote.clients(DOC, AGREEMENTS), ['ISDA-KESTREL', 'ISDA-NESTED', 'ISDA-ZULU']);

check('a client reads as the entity behind its set, the set beside it',
      'the agreement id read as the entity, which shows a set where a client\'s name belongs',
      [quote.clientText('ISDA-KESTREL',
                        [{ entity: 'KESTREL', name: 'Kestrel Mining Ltd', parent: null }],
                        AGREEMENTS),
       quote.clientText('ISDA-OTHER', [], AGREEMENTS), quote.clientText(null, [], [])],
      ['Kestrel Mining Ltd (KESTREL) · ISDA-KESTREL', 'ISDA-OTHER',
       'no netting set - books at the root of the book']);

// --- the ticket answered: a collar on ZARUSD, the book's surface quoted USDZAR
const leg = (role, type, side, strike) => ({
  Instrument: { '.Deal': {
    Object: 'FXOptionDeal', Reference: `ZeroCostCollar-ab_${role}`, Currency: 'USD',
    Underlying_Currency: 'ZAR', Underlying_Amount: 18000000, FX_Volatility: 'USD.ZAR',
    Option_Type: type, Buy_Sell: side, Strike_Price: strike,
    Expiry_Date: { '.Timestamp': '2027-02-16' } } } });
const ZARUSD = {
  quote_id: 'ab', structure: 'ZeroCostCollar', variation: 'floor',
  params: { pair: 'ZARUSD', expiry: '3M', notional: 18000000, notional_currency: 'ZAR', floor: 0.0555 },
  client: { buys: 'USD', sells: 'ZAR' }, netting_set: null, currency: 'USD',
  legs: [
    { reference: 'ZeroCostCollar-ab_protection', role: 'protection', buy_sell: 'Buy',
      strike_market: 0.0555, premium: 18614.53, solved: null, note: null },
    { reference: 'ZeroCostCollar-ab_financing', role: 'financing', buy_sell: 'Sell',
      strike_market: 0.058461939594854774, premium: -19958.90,
      solved: { Strike_Price: 0.058461939594854774 }, note: null }],
  net: 0, net_mid: -1344.37, edge: 1344.37, charged_on: 'Strike_Price', spread_note: null,
  spot: { source: 'book', note: null, value_market: 0.05763688760806916 },
  forward: 0.05717028779136157,
};
const DEAL = { Object: 'StructuredDeal', Children: [leg('protection', 'Put', 'Buy', 0.0555),
  leg('financing', 'Call', 'Sell', 0.058461939594854774)] };
const legs = quote.ticketLegs(ZARUSD, DEAL, null);
check('a leg reads on the pair the market quotes, by the blotter\'s rule: a ZARUSD put is a USDZAR '
      + 'call, struck at the reciprocal, and the level the recipe solved is marked',
      'the quote\'s own strike_market read, which prints 0.0555 on a desk that quotes 18.02; or '
      + 'the solved level matched to the wrong leg',
      legs.map((one) => [one.leg.role, one.read.on?.value, one.read.sense,
                         Number(one.read.strike?.value.toFixed(4)), one.size.side?.value,
                         one.solved ?? null]),
      [['protection', 'USDZAR', 'Call', 18.018, 'Buy', null],
       ['financing', 'USDZAR', 'Put', 17.1051, 'Sell', 'strike']]);

const fixing = (day, settles) => [{ '.Timestamp': day }, { '.Timestamp': settles }, 0.0];
const TARF = {
  Object: 'FXTARFOptionDeal', Reference: 'TargetRedemptionForward-ef_tarf', Currency: 'ZAR',
  Underlying_Currency: 'USD', Underlying_Amount: 500000, FX_Volatility: 'USD.ZAR',
  Option_Type: 'Call', Buy_Sell: 'Buy', Strike_Price: 16.05, TargetLevel: 1.5,
  LeverageNotional: 1000000,
  TARF_ExpiryDates: [fixing('2026-12-16', '2026-12-18'), fixing('2027-01-14', '2027-01-18'),
                     fixing('2027-02-16', '2027-02-18')] };
const TARF_SCHEMA = { Instrument: { types: { FXTARFOptionDeal: ['strip'] }, sections: { strip: {
  Underlying_Amount: { widget: 'Float', description: 'Underlying Amount', value: 0, sized: true },
  TARF_ExpiryDates: { widget: 'Table', description: 'TARF ExpiryDates', value: 'null',
                      col_names: ['Fixing Date', 'Settlement Date', 'Value'] } } } } };
check('a strip\'s fixings ride its leg on the ticket, under the columns its type declares; a vanilla '
      + 'carries none',
      'the schedule\'s declaration not looked up, which prints the fixings as bare rows no column '
      + 'names; or every leg handed a schedule, an empty table under each vanilla',
      [quote.ticketLegs({ legs: [{ role: 'tarf', solved: { Strike_Price: 16.05 } }] },
                        { Children: [{ Instrument: { '.Deal': TARF } }] }, TARF_SCHEMA)
        .map((one) => [one.schedule?.value.length, one.schedule?.declared?.col_names, one.solved,
                       one.read.terms.parts]),
       legs.map((one) => one.schedule ?? null)],
      [[[3, ['Fixing Date', 'Settlement Date', 'Value'], 'strike',
         ['3 fixings, amount per fixing', 'target 1.5 ZAR per USD', 'leverage 2×']]], [null, null]]);

check('the solved number is set apart on the market\'s axis; a solved ticket cell as the quote '
      + 'states it',
      'the engine\'s value shown, which sets 0.0585 apart on a USDZAR desk',
      [quote.solvedNumbers(legs).map((one) => [one.what, one.on, Number(one.value.toFixed(4))]),
       quote.solvedNumbers([{ leg: { role: 'tier_1', solved: { 'tiers[0].coupon': 2.85 } },
                              read: {} }])],
      [[['financing strike', 'USDZAR', 17.1051]], [{ what: 'coupon, tiers row 1', value: 2.85 }]]);

check('a solved parameter reads in words, a table\'s cell by its row counted from one',
      'the path shown as the recipe writes it, which reads a desk tiers[0].coupon; or the row '
      + 'counted from nothing, which names the first tier row 0',
      [quote.pathWords('strike_price'), quote.pathWords('tiers[0].coupon'),
       quote.pathWords('tiers[2].upper_cap'),
       quote.spreadText({ ...ZARUSD, charged_on: 'tiers[0].coupon', edge: 10, net_mid: 5 })],
      ['strike price', 'coupon, tiers row 1', 'upper cap, tiers row 3',
       "The market's two-way spread costs the client 10 USD, charged on the coupon, tiers row 1; at "
       + 'mid the legs are worth 5 USD.']);

check('the spot and the forward read on the pair as the market quotes it',
      'the level read as the ticket wrote it, which shows a USDZAR desk a spot of 0.0576',
      [quote.onMarketAxis(ZARUSD.spot.value_market, 'ZARUSD', 'USD.ZAR'),
       quote.onMarketAxis(17.35, 'USDZAR', 'USD.ZAR')]
        .map((read) => [read?.pair, Number(read?.level.toFixed(4))]),
      [['USDZAR', 17.35], ['USDZAR', 17.35]]);

check('the spread reads in words: what it cost the client, where, and the legs at mid',
      'the charged coordinate dropped, which hides that the cap moved against the client; or the '
      + "quote's own no-two-way note overwritten",
      [quote.spreadText(ZARUSD),
       quote.spreadText({ ...ZARUSD, charged_on: 'premium', edge: 10, net_mid: 5 }),
       quote.spreadText({ ...ZARUSD, spread_note: 'FXVolPrices.USD.ZAR is not on this book' })],
      ["The market's two-way spread costs the client 1,344.37 USD, charged on the strike price; at "
       + 'mid the legs are worth -1,344.37 USD.',
       "The market's two-way spread costs the client 10 USD, added to the premium; at mid the legs "
       + 'are worth 5 USD.',
       'FXVolPrices.USD.ZAR is not on this book']);

// --- quoting again
const CAKED = {
  quote_id: 'cd', structure: 'RangeAccrual', variation: null, netting_set: 'ISDA-KESTREL',
  params: { premium: 5000, pair: 'USDZAR', tiers: [[17.0, 18.2, 2.8512196637090006]] },
  legs: [{ role: 'tier_1', solved: { 'tiers[0].coupon': 2.8512196637090006 } }],
  margin: { amount: 20000, currency: 'ZAR', value: 1152.7, pricing_currency: 'USD' },
};
check('a quote quoted again starts from its ticket as priced, every cell the recipe solved blank again',
      'the solved coupon kept, which states every cell and the premium - the service refuses '
      + 'that, leaving nothing to solve; or the tiers left as the arrays the quote filed, which the '
      + 'table cannot read by column',
      [quote.draftOf(CAKED, CAKE), quote.draftOf(ZARUSD, COLLAR).values],
      [{ structure: 'RangeAccrual', variation: null,
         values: { premium: 5000, pair: 'USDZAR', tiers: [{ floor: 17, cap: 18.2, coupon: '' }] },
         nettingSet: 'ISDA-KESTREL', margin: { amount: '20000', currency: 'ZAR' } },
       ZARUSD.params]);

// --- the firm window and the acceptance
const SCHEMA = { 'Quote Policy': { fields: { firm_seconds: { widget: 'Float', value: 600 } } } };
const policy = (block) => ({ Calc: { Calculation: {}, ...(block ? { 'Quote Policy': block } : {}) } });
check('the firm window is the policy\'s own, its convention where it states none, none without one',
      'a stated window of 0 read as unstated, which shows ten minutes on a desk refusing every '
      + 'acceptance',
      [quote.firmSeconds(policy({ firm_seconds: 0 }), SCHEMA),
       quote.firmSeconds(policy({}), SCHEMA), quote.firmSeconds(policy(null), SCHEMA)],
      [0, 600, null]);

check('a quote is firm until its stamp plus the window',
      'the window added in milliseconds, which lapses a ten-minute quote in under a second',
      quote.firmUntil('2026-11-19T18:14:08.613+00:00', 600), '2026-11-19T18:24:08.613Z');

check('what holds between a quote and its acceptance is said per desk',
      'the record ignored, which tells a recording desk a quote stays bookable whatever the book '
      + 'did since',
      [quote.firmText(600, true), quote.firmText(null, true), quote.firmText(null, false)],
      ['This desk holds a quote firm for 600 seconds; an older one is refused, and is quoted again.',
       'This desk declares no firm window: a quote books while the book has not moved under it.',
       'This desk declares no firm window: a quote stays bookable until it is booked.']);

const SIZE = "the 'desk-auto' tier caps max_notional at 5000000 USD and this ticket is "
  + '6560000.0 of it - book it under a tier that admits the size, or split the clip';
const TENOR = "the 'by-size' tier caps max_tenor_years at 1 years and this ticket runs 2.25 - "
  + 'book it under a tier that admits the tenor';
check('an acceptance reads as where it landed, the service\'s words, and whether to quote '
        + 'again; a booking under a tier says on whom it waits, then each check an earlier tier '
        + 'failed it on, a line each, and nothing where it fell in the first',
      'every refusal offered a re-quote, which re-prices a client\'s price the desk\'s own tiers '
      + 'declined; a booking pending a second seat read as booked and done; the checks it fell '
      + 'past dropped, or joined into one line, or an automatic tier\'s left off; or a ticket that '
      + 'fell in the first tier given a blank line',
      [quote.verdictOf({ written: true, deal_path: '0/0', recorded: { lsn: 25 } }),
       quote.verdictOf({ written: true, deal_path: '1/1' }),
       quote.verdictOf({ written: true, deal_path: '0/1', recorded: { lsn: 31 },
                         waits_on: 'a second seat holding approve' }),
       quote.verdictOf({ written: false, booked: { lsn: null, deal_path: '1/1' },
                         note: 'booked at 1/1' }),
       quote.verdictOf({ written: false, refused: ['the book would not take it'] }),
       quote.verdictOf({ written: false, refused: ['no tier admits it'],
                         accepted: { lsn: 40, ticket: 't' } }),
       quote.verdictOf({ written: true, deal_path: '0/1', waits_on: 'a second seat holding approve',
                         tier: { name: 'second-seat', four_eyes: true,
                                 fell_past: [SIZE, TENOR] } }),
       quote.verdictOf({ written: true, deal_path: '0/2',
                         tier: { name: 'desk-auto', four_eyes: false, fell_past: [] } }),
       quote.verdictOf({ written: true, deal_path: '0/3',
                         tier: { name: 'by-size', four_eyes: false, fell_past: [SIZE] } })],
      [{ booked: { deal_path: '0/0', lsn: 25 }, words: [], requote: false },
       { booked: { deal_path: '1/1', lsn: null }, words: [], requote: false },
       { booked: { deal_path: '0/1', lsn: 31 },
         words: ['Booked pending a second seat: a second seat holding approve'], requote: false },
       { booked: { lsn: null, deal_path: '1/1' }, words: ['booked at 1/1'], requote: false },
       { words: ['the book would not take it'], requote: true },
       { words: ['no tier admits it'], requote: false },
       { booked: { deal_path: '0/1', lsn: null },
         words: ['Booked pending a second seat: a second seat holding approve',
                 `Fell past: ${SIZE}`, `Fell past: ${TENOR}`], requote: false },
       { booked: { deal_path: '0/2', lsn: null }, words: [], requote: false },
       { booked: { deal_path: '0/3', lsn: null }, words: [`Fell past: ${SIZE}`], requote: false }]);

check('a re-quote made here is kept with the quote it changes, once, the newest first',
      'a re-quote kept twice where it is made again, or its link dropped for the newest',
      quote.keep(quote.keep(quote.keep([], 'b', 'a'), 'c', 'b'), 'b', 'a'),
      [{ id: 'b', from: 'a' }, { id: 'c', from: 'b' }]);

// --- the desk's quotes
const AT = '2026-11-19T22:51:18.078+00:00';
const NOW = Date.parse('2026-11-19T22:53:00.078Z');
const given = (extra) => ({ quoted_at: AT, accepted: null, booked: null, ...extra });
check('a quote stands booked, accepted, firm until its window closes, past it, or simply quoted '
        + 'where the book declares no window',
      'a booked quote read against the window, which shows the desk\'s own trade as lapsed; a '
        + 'quote stating no time read as firm, which the acceptance refuses as aged; or the window '
        + 'read as open at its closing instant\'s far side',
      [quote.standsAt(given({ booked: { lsn: 30, deal_path: '0/2' } }), 60, NOW),
       quote.standsAt(given({ accepted: { lsn: 28, ticket: 't' } }), 600, NOW),
       quote.standsAt(given(), 600, NOW), quote.standsAt(given(), 60, NOW),
       quote.standsAt(given(), 102, NOW), quote.standsAt(given({ quoted_at: null }), 600, NOW),
       quote.standsAt(given(), null, NOW)],
      [{ word: 'booked' }, { word: 'accepted' },
       { word: 'firm', until: '2026-11-19T23:01:18.078Z' },
       { word: 'lapsed', until: '2026-11-19T22:52:18.078Z' },
       { word: 'firm', until: '2026-11-19T22:53:00.078Z' }, { word: 'lapsed', until: undefined },
       { word: 'quoted' }]);

const LISTED = [given({ quote_id: 'q3' }), given({ quote_id: 'q2', booked: { lsn: 30, deal_path: '0/20' } }),
                given({ quote_id: 'q1', booked: { lsn: null, deal_path: '0/2' } }),
                given({ quote_id: 'q4',
                        booked: { lsn: 41, deal_path: '2', reference: 'COLLAR-A' } }),
                given({ quote_id: 'q5',
                        booked: { lsn: 42, deal_path: '3', reference: 'COLLAR-B' } })];
check('a trade\'s quote is the one whose booking names its reference, wherever the file now holds '
        + 'it, and the one that landed at its path only where the booking names none',
      'a path matched by its start, so the trade at 0/2 takes the quote booked at 0/20; the first '
        + 'quote of the list taken for any trade; or the path trusted over the reference, which '
        + 'gives COLLAR-B, moved to 2 by a deletion ahead of it, the quote of COLLAR-A it replaced',
      [quote.quoteAt(LISTED, '0/2', 'X')?.quote_id, quote.quoteAt(LISTED, '0/20', 'X')?.quote_id,
       quote.quoteAt(LISTED, '1/0', 'X'), quote.quoteAt(LISTED, '2', 'COLLAR-B')?.quote_id,
       quote.quoteAt(LISTED, '3', 'COLLAR-C')],
      ['q1', 'q2', undefined, 'q5', undefined]);

const QUOTED_BOOK = [
  { Instrument: { '.Deal': { Object: 'NettingCollateralSet', Reference: 'ISDA-K' } }, Children: [
    { Instrument: { '.Deal': { Object: 'FXOptionDeal', Reference: 'CALL' } } }] },
  { Instrument: { '.Deal': { Object: 'StructuredDeal', Reference: 'COLLAR-B' } }, Children: [
    { Instrument: { '.Deal': { Object: 'FXOptionDeal', Reference: 'COLLAR-A' } } }] }];
check('a booked quote lands where the file holds its trade now: the trade under its reference, a '
        + 'leg never, nothing where the file lost it, and its path only where it names none',
      'the booked path read as the trade, which opens COLLAR-B for the quote that booked the '
        + 'deleted COLLAR-A; a leg naming the reference taken for the trade; or a netting set '
        + 'taken for its trades',
      [quote.landedAt(QUOTED_BOOK, LISTED[4].booked), quote.landedAt(QUOTED_BOOK, LISTED[3].booked),
       quote.landedAt(QUOTED_BOOK, { lsn: 1, deal_path: '0/0' }),
       quote.landedAt(QUOTED_BOOK, { lsn: 1, deal_path: '0', reference: 'ISDA-K' })],
      ['1', undefined, '0/0', undefined]);

done();
