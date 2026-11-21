import { useEffect, useReducer, useRef, useState } from 'react';
import {
  DOORBELL, failure, getBook, getBookActivity, getBookAgreements, getBookEntities, getBookMarket,
  getBookPortfolios, getBookPositions, getBookRisk, getBookStatus, getBookXva, getPaths, getSchema,
  getSeat, postDescribe,
} from './api';
import { DocumentLoader } from './components/DocumentLoader';
import { JobHeader } from './components/JobHeader';
import { ActivityStrip, ReconcileBanner, SeatControl, WorklistBanner } from './components/Record';
import { WORKSPACES } from './registry';
import { bookOwed, polled, ringsAhead, single, worklistAt } from './spine';
import {
  AppContext, INITIAL, PAPER_TABS, bookLoaded, clockOf, paperAt, paperOwed, reducer, seatClock,
  spineOf,
} from './state';

const BOOK_POLL_MS = 2000;

/** Each tab's label by its id, for a page's way back. */
const LABELS = Object.fromEntries(WORKSPACES.map((w) => [w.id, w.label]));

export function App() {
  const [state, dispatch] = useReducer(reducer, INITIAL, (initial) => ({
    ...initial,
    tab: WORKSPACES.some((w) => w.id === location.hash.slice(1))
      ? location.hash.slice(1) : initial.tab,
  }));

  // the schema, once - everything renders from it
  useEffect(() => {
    getSchema()
      .then((schema) => dispatch({ type: 'SCHEMA_LOADED', schema }))
      .catch((error) => dispatch({ type: 'SCHEMA_FAILED', error: failure(error).error }));
  }, []);

  // the live book, if the service serves one - silence (a 404) just means "open a file instead"
  useEffect(() => {
    getBook()
      .then((live) => dispatch(bookLoaded(live)))
      .catch(() => undefined);
  }, []);

  // a hidden window is nobody looking - the one thing the tab does not say
  const [visible, setVisible] = useState(document.visibilityState === 'visible');
  useEffect(() => {
    const onVisibility = () => setVisible(document.visibilityState === 'visible');
    addEventListener('visibilitychange', onVisibility);
    return () => removeEventListener('visibilitychange', onVisibility);
  }, []);

  // Where the record's strip has read up to, and the source the beat compares the stamp with:
  // refs, so the beat below is made once and never re-reads the history.
  const since = useRef(0);
  const source = useRef(state.source);
  source.current = state.source;

  // THE BEAT, over the desk's own book and never a document opened from a file (`bookOwed`): the
  // status read says where the book and the record stand, the book is fetched only where its
  // stamp moved, and the strip's next page rides along - all of it ONE transition, so a deal
  // booked by any client moves the clock once. One beat at a time (`single`), so a stalled
  // service stacks nothing; `spine` is read fresh each time, so a service restarted with a home
  // appears without a reload.
  const beat = useRef(single(async () => {
    try {
      const status = await getBookStatus();
      const [live, page] = await Promise.all([
        bookOwed(source.current, status.etag) ? getBook() : null,
        status.spine === null ? null : getBookActivity(since.current || '')]);
      if (page) since.current = page.lsn;
      dispatch({ type: 'BEAT', etag: status.etag, live, spine: status.spine, page });
    } catch { /* the beat outlives a service restart */ }
  })).current;

  // it beats while the window is shown, and once on being shown again to catch up
  const beating = polled(state.source) && visible;
  useEffect(() => {
    if (!beating) return;
    beat();
    const timer = setInterval(beat, BOOK_POLL_MS);
    return () => clearInterval(timer);
  }, [beating, beat]);

  // THE DOORBELL, where the service rings one: a beat per head move, and the same read. A ring is
  // a NOTIFICATION AND NEVER A DELIVERY, so a stream a proxy closes, a ring that arrives twice and
  // one that arrives out of order all cost latency and never an event - the beat above reaches
  // every one of them on its own. Opened only on a shown desk that records: a box keeping no
  // home 404s this, and the browser would reconnect to that forever.
  const recording = state.record.spine !== null;
  useEffect(() => {
    if (!recording || !visible) return;
    const bell = new EventSource(DOORBELL);
    bell.onmessage = (ring) => {
      if (ringsAhead(JSON.parse(ring.data), since.current)) beat();
    };
    return () => bell.close();
  }, [recording, visible, beat]);

  // The desk's two data views ride the ONE clock (`bookClock`): the book's etag the poll above
  // moves, and where the record stands - a booking, an amendment or a tick moves the first, an
  // approval or a record event the book prices off the second - and the numbers follow either.
  const clock = clockOf(state);
  // the clock each slice's answer in hand was read at: a tab taken again under the same clock asks
  // for nothing it already holds
  const answered = useRef<Record<string, string>>({});
  const owed = (slice: string, at: string, held: unknown) =>
    answered.current[slice] !== at || !held;

  // The risk verb RUNS the book on a miss, so it is priced while a desk is LOOKING at it and only
  // then: taking the tab prices it, every move of the clock under that gaze re-prices it, and a tab
  // left behind or a window gone hidden costs nothing until it is looked at again.
  const riskWanted = ['risk', 'trade', 'xva', 'client'].includes(state.tab) && visible;
  const riskAt = seatClock(state);
  useEffect(() => {
    if (clock === null || riskAt === null || !riskWanted
        || !owed('risk', riskAt, state.risk.data)) return;
    let live = true;
    dispatch({ type: 'RISK_FETCHING' });
    getBookRisk()
      .then((risk) => {
        if (!live) return;
        dispatch({ type: 'RISK_LOADED', risk, clock });
        answered.current.risk = riskAt;
      })
      .catch((error) => { if (live) dispatch({ type: 'RISK_FAILED', ...failure(error) }); });
    return () => { live = false; };
  }, [riskWanted, riskAt]);

  // The XVA view is a file read, not a run: it re-reads on FOCUS - a recalc asked for through the
  // MCP verbs lands in that file while nobody is looking at this screen - and again whenever the
  // clock moves under it, which is what says a row is of an earlier book, or an orphan.
  const onXva = ['xva', 'client'].includes(state.tab);
  useEffect(() => {
    if (clock === null || !onXva || !owed('xva', clock, state.xva.data)) return;
    let live = true;
    dispatch({ type: 'XVA_FETCHING' });
    getBookXva()
      .then((xva) => {
        if (!live) return;
        dispatch({ type: 'XVA_LOADED', xva, clock });
        answered.current.xva = clock;
      })
      .catch((error) => { if (live) dispatch({ type: 'XVA_FAILED', ...failure(error) }); });
    return () => { live = false; };
  }, [clock, onXva]);

  // The market the book prices on and where this box reads the files it names, read on the same
  // etag so a write anywhere repaints them. The market is asked with the stamp held, through a ref
  // so the read stays keyed on the source alone: it crosses the wire only where it moved.
  const stamp = useRef<string>();
  stamp.current = state.market.data?.stamp;
  useEffect(() => {
    if (state.source?.kind !== 'book') return;
    let live = true;
    getBookMarket(stamp.current)
      .then((market) => { if (live) dispatch({ type: 'MARKET_LOADED', market }); })
      .catch((error) => { if (live) dispatch({ type: 'MARKET_FAILED', ...failure(error) }); });
    getPaths()
      .then((paths) => { if (live) dispatch({ type: 'PATHS_LOADED', paths }); })
      .catch((error) => { if (live) dispatch({ type: 'PATHS_FAILED', ...failure(error) }); });
    return () => { live = false; };
  }, [state.source]);

  // The record's paper and the holdings under it, for every screen that names its clients,
  // groups the book by them or signs its tickets: each part a screen shows read while it is looked
  // at on a desk that records - the holdings once per clock and seat, the paper once per
  // declaration of it (`paperAt`) - however often the tabs that show it are taken.
  const paperWanted = PAPER_TABS.includes(state.tab) && visible && recording;
  const at = paperAt(state);
  useEffect(() => {
    if (at === null || !paperWanted) return;
    const owes = paperOwed(state.tab, state.paper.data, answered.current, at);
    if (!owes.paper && !owes.holdings) return;
    let live = true;
    dispatch({ type: 'PAPER_FETCHING' });
    Promise.all([owes.paper && Promise.all([getBookEntities(), getBookAgreements()]),
                 owes.holdings && Promise.all([getBookPositions(), getBookPortfolios()])])
      .then(([signed, held]) => {
        if (!live) return;
        dispatch({ type: 'PAPER_LOADED', paper: {
          ...(signed && { entities: signed[0].entities, agreements: signed[1].agreements,
                          unstated: signed[1].settlement_state }),
          ...(held && { positions: held[0].positions,
                        nodes: held[1].portfolios.map((node) => node.path) }) } });
        if (signed) answered.current.paper = at.paper;
        if (held) answered.current.holdings = at.holdings;
      })
      .catch((error) => { if (live) dispatch({ type: 'PAPER_FAILED', ...failure(error) }); });
    return () => { live = false; };
  }, [at?.paper, at?.holdings, paperWanted, state.tab]);

  // What the seat may do, asked again where the seat or the record's head moved. A refusal - no
  // seat named, under grants - stands in its place; any other failure leaves acts to the service.
  const grantsAt = worklistAt(spineOf(state), state.seat);
  useEffect(() => {
    if (grantsAt === null) return;
    let live = true;
    const asked = state.seat;
    const held = (grants: typeof state.grants) => {
      if (live) dispatch({ type: 'GRANTS', grants });
    };
    getSeat()
      .then((read) => held({ asked, read, refused: null }))
      .catch((error) => {
        const { status, error: refused } = failure(error);
        held(status === 422 ? { asked, read: null, refused } : null);
      });
    return () => { live = false; };
  }, [grantsAt]);

  // describe whenever the document moves
  useEffect(() => {
    if (!state.doc) return;
    postDescribe(state.doc)
      .then((describe) => dispatch({ type: 'DESCRIBED', describe }))
      .catch(() => undefined);
  }, [state.doc]);

  // tab mirrored to the hash - back/forward for free, no router (the static mount has no SPA
  // fallback, so a path-routed deep link would 404 on reload)
  useEffect(() => {
    if (location.hash.slice(1) !== state.tab) location.hash = state.tab;
  }, [state.tab]);
  useEffect(() => {
    const onHash = () => {
      const tab = location.hash.slice(1);
      if (WORKSPACES.some((w) => w.id === tab)) dispatch({ type: 'TAB', tab });
    };
    addEventListener('hashchange', onHash);
    return () => removeEventListener('hashchange', onHash);
  }, []);

  const Active = WORKSPACES.find((w) => w.id === state.tab)?.view ?? WORKSPACES[0].view;

  return (
    <AppContext.Provider value={{ state, dispatch, labels: LABELS }}>
      <div className="app">
        <header className="masthead">
          <h1><span>▮</span> derivus</h1>
          <DocumentLoader />
          <span className="spacer" />
          <SeatControl />
          {state.schema && <span className="version">engine {state.schema.engine_version}</span>}
        </header>
        <JobHeader />
        <ReconcileBanner />
        <WorklistBanner />
        <nav className="tabs">
          {WORKSPACES.map((workspace) => (
            <button
              key={workspace.id}
              className={state.tab === workspace.id ? 'active' : ''}
              onClick={() => dispatch({ type: 'TAB', tab: workspace.id })}
            >
              {workspace.label}
            </button>
          ))}
        </nav>
        {state.schemaError && (
          <div className="error-box">The service is not answering: {state.schemaError}</div>
        )}
        {state.docError && <div className="error-box">{state.docError}</div>}
        {state.market.error && <div className="error-box">{state.market.error}</div>}
        {!state.doc ? (
          <div className="placeholder" style={{ flex: 1 }}>
            <div>No document loaded.</div>
            <div className="hint">
              Start the service with <code>--book &lt;job file&gt;</code>, or open a job file above.
            </div>
          </div>
        ) : (
          <Active />
        )}
        <ActivityStrip />
      </div>
    </AppContext.Provider>
  );
}
