// Drives `src/pnl.ts` where the eye cannot: how the P&L screen names a position and a risk factor,
// what it asks the verb for and when it asks again, and how it reads the explain. Every check names
// the MUTATION it kills - the change that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/pnl_check.mjs          # exits 1 on a miss

import { check, done, load } from './check.mjs';

const pnl = await load('pnl');

const [desk, file] = [pnl.noPnl(false, true), pnl.noPnl(false, false)];
check("a desk keeping no record is told so in the Paper tab's voice, and where the book's worth "
      + 'is read; a file opened here that no P&L is read of it; a desk that records nothing in '
      + "the P&L's place; and nobody a flag, a route or a variable",
      "the state dropped, which shows the service's refusal of a read no such desk answers, a "
      + "route and a variable in a red box; the state shown on a desk that records; or the "
      + "operator's words back, --book and a spine home",
      [pnl.noPnl(true, true), desk.title, /Risk tab/.test(desk.hint), file.title,
       /Risk tab/.test(file.hint),
       [desk, file].some((said) => /--book|spine|DV_|\/book|the service/.test(said.title + said.hint))],
      [null, 'This desk keeps no record, so there is no P&L to read here.', true,
       'A file opened here has no record, so no P&L is read of it.', false, false]);

const moved = (named) => ({ delta: 2, start: 1, end: 1.5, move: 0.5, pnl: 1, ...named });

check('a risk factor reads as its quote in its block, or as the factor at its coordinates',
      'a quote shown by its block alone, which files every rung of a curve under one name; or a '
      + "spot given an empty coordinate, which reads as a curve's front point",
      [pnl.factorLabel(moved({ block: 'InterestRatePrices.ZAR', quote: '3Y' })),
       pnl.factorLabel(moved({ factor: 'InterestRate.ZAR', tenor: [0.5] })),
       pnl.factorLabel(moved({ factor: 'FXVol.ZAR', tenor: [1, 0.25] })),
       pnl.factorLabel(moved({ factor: 'FxRate.ZAR', tenor: [] }))],
      ['InterestRatePrices.ZAR · 3Y', 'InterestRate.ZAR · 0.5', 'FXVol.ZAR · 1 × 0.25',
       'FxRate.ZAR']);

const EXPLAIN = { existing: -200, carry: 5, market: -190, residual: -15, reserves: null,
                  factors: [], unknown: [], note: null };
check('the explain reads in the order it adds up, the residual last',
      'the lines reordered or one dropped, which leaves a reader summing three numbers to a fourth '
      + 'that is not theirs',
      pnl.explained(EXPLAIN).map((line) => [line.label, line.value]),
      [['Held positions', -200], ['Carry', 5], ['Market', -190], ['Residual', -15]]);

check("the residual's share is of what the held positions made, whatever its sign, and none "
      + 'where nothing was made or either is unknown',
      "a share of a signed P&L, which flips a loss day's residual; or a share of nothing, "
      + 'which reads as infinite',
      [pnl.residualShare(EXPLAIN), pnl.residualShare({ ...EXPLAIN, existing: 0 }),
       pnl.residualShare({ ...EXPLAIN, residual: null }),
       pnl.residualShare({ ...EXPLAIN, existing: null })],
      [-0.075, null, null, null]);

check('the verb is asked for the window and the scope stated, a blank end being now',
      "a blank end sent as `end=`, which the verb refuses as no day; a blank scope sent, which "
      + 'narrows the book to a portfolio named nothing; or the explain asked for unasked, which '
      + 'costs three valuations',
      [pnl.pnlQuery('2024-07-01', '', { portfolio: '', agreement: 'ISDA-1', client: '' }, false),
       pnl.pnlQuery('', '2024-07-02', { portfolio: 'desk/FX' }, true)],
      ['start=2024-07-01&agreement=ISDA-1', 'end=2024-07-02&portfolio=desk%2FFX&explain=true']);

check("a read ending now, or explained, is a what-if's work, which the queue admits a seat to; a "
      + 'window between two marked days read plain is a read of the record, asked of any seat',
      'the explain dropped, which asks three valuations of a seat the queue refuses; or every '
      + 'read taken for a what-if, which holds a struck window from a seat that may read it',
      [pnl.revalues('', false), pnl.revalues('2024-07-02', true), pnl.revalues('', true),
       pnl.revalues('2024-07-02', false)],
      [true, true, true, false]);

const DAYS = ['2024-06-28', '2024-07-01', '2024-07-02'].map((day) => ({ day, lsn: 1 }));
const opens = (days, held) => {
  const window = pnl.opensOn(days, held);
  return window && [window.start, window.end, pnl.revalues(window.end, false)];
};
check('the page opens to now from the last marks for a seat that may ask a what-if; for a seat '
      + 'held from one it opens on the last two marked days, a window that asks none, and on no '
      + 'window where fewer than two days are marked',
      'the held seat still opened to now, which holds it from every window the page could show '
      + 'it; its window opened on the first two days marked; or one marked day read as a window',
      [opens(DAYS, false), opens(DAYS, true), opens(DAYS.slice(0, 1), true), opens([], true)],
      [['', '', true], ['2024-07-01', '2024-07-02', false], null, null]);
const HOLD = 'Needs validate: sales-9 holds it nowhere.';
check('a held seat with no window is told, after the line holding it, that a P&L between two '
      + 'marked days is not held from it and reads once two are marked, beside the one that is',
      'the held line left alone, which reads as though the seat may never read a P&L here; or '
      + 'the day marked left out, which leaves it to guess how many closes are owed',
      [pnl.noWindow(HOLD, DAYS.slice(0, 1)), pnl.noWindow(HOLD, [])],
      [`${HOLD} A P&L between two marked days is not held from this seat, and reads once two `
        + 'days are marked: only 2024-06-28 is.',
       `${HOLD} A P&L between two marked days is not held from this seat, and reads once two `
        + 'days are marked: none is.']);

const QUERY = pnl.pnlQuery('2024-07-01', '2024-07-02', {}, false);
check('a P&L read is of the seat reading and the query asked: another seat reads again, and so '
      + 'does another window',
      "the seat left out of the key, which leaves the last seat's rows and totals under the "
      + 'seat now named in the header; or the query left out, which reads no window chosen',
      [pnl.pnlRead('sales-1', QUERY) === pnl.pnlRead('sales-9', QUERY),
       pnl.pnlRead('sales-1', QUERY) === pnl.pnlRead('sales-1', pnl.pnlQuery('2024-06-28', '', {}, false)),
       pnl.pnlRead('sales-1', QUERY) === pnl.pnlRead('sales-1', QUERY)],
      [false, false, true]);

check('only a window ending now falls behind the book, and only once the book has moved',
      'a struck day offered a re-read, which never moves; or every write reading the window again, '
      + 'which values the whole book per tick',
      [pnl.behind('', 'e1', 'e1'), pnl.behind('', 'e1', 'e2'), pnl.behind('2024-07-02', 'e1', 'e2'),
       pnl.behind('', undefined, 'e2')],
      [false, true, false, false]);

check('a position reads by the reference it was booked under, else by its address',
      'the address shown where a reference exists, which no desk can read',
      [pnl.positionLabel({ instrument: 'a'.repeat(64), reference: 'CF-A' }),
       pnl.positionLabel({ instrument: 'b'.repeat(64), reference: null })],
      ['CF-A', `${'b'.repeat(12)}…`]);

done();
