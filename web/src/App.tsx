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
import { polled, ringsAhead, worklistAt } from './spine';
import { AppContext, INITIAL, bookLoaded, clockOf, reducer } from './state';

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

  // Where the record's strip has read up to. A ref rather than state: the beat below is
  // recreated whenever the book moves, and a cursor that reset with it would re-read the history.
  const since = useRef(0);

  // The record's own read - the pin and how far the record has moved past it, then the events
  // after the cursor. ONE FUNCTION on two clocks, the poll's beat and the doorbell's, so the two
  // cannot read differently; held across renders, so the stream below is opened once. `spine` is
  // read fresh every time, so a service restarted with a home appears without a reload and a box
  // that keeps none is never read twice.
  const readRecord = useRef(async () => {
    const status = await getBookStatus();
    const page = status.spine === null ? null : await getBookActivity(since.current || '');
    if (page) since.current = page.lsn;
    dispatch({ type: 'RECORD_READ', spine: status.spine, page });
  }).current;

  // the etag poll, over the desk's own book and never a document opened from a file (`polled`):
  // a deal booked by ANY client (MCP, Excel, an editor on the file) appears within a tick, the
  // user's place preserved, and a moved path is another document served and loads as one
  useEffect(() => {
    if (!polled(state.source)) return;
    const { etag, path } = state.source;
    const timer = setInterval(async () => {
      try {
        const live = await getBook();
        if (live.etag !== etag || live.path !== path) {
          dispatch(bookLoaded(live, live.path === path));
        }
        await readRecord();
      } catch { /* the poll outlives a service restart */ }
    }, BOOK_POLL_MS);
    return () => clearInterval(timer);
  }, [state.source, readRecord]);

  // THE DOORBELL, where the service rings one: a beat per head move, and the same read. A beat is
  // a NOTIFICATION AND NEVER A DELIVERY, so a stream a proxy closes, a beat that arrives twice and
  // one that arrives out of order all cost latency and never an event - the poll above reaches
  // every one of them on its own beat. Opened only on a desk that records: a box keeping no home
  // 404s this, and the browser would reconnect to that forever.
  const recording = state.record.spine !== null;
  useEffect(() => {
    if (!recording) return;
    const bell = new EventSource(DOORBELL);
    bell.onmessage = (beat) => {
      if (ringsAhead(JSON.parse(beat.data), since.current)) readRecord();
    };
    return () => bell.close();
  }, [recording, readRecord]);

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
  const riskAt = `${clock}|${state.seat}`;
  useEffect(() => {
    if (clock === null || !riskWanted || !owed('risk', riskAt, state.risk.data)) return;
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

  // The positions and the paper they sit under, for every screen that groups the book by them,
  // names its clients or signs its tickets: read while one is looked at on a desk that records,
  // once per clock however often the tabs that show it are taken.
  const paperWanted = ['portfolio', 'blotter', 'quote', 'trade', 'risk', 'xva', 'curves',
    'backoffice', 'client', 'paper', 'pnl'].includes(state.tab) && visible && recording;
  useEffect(() => {
    if (clock === null || !paperWanted || !owed('paper', clock, state.paper.data)) return;
    let live = true;
    dispatch({ type: 'PAPER_FETCHING' });
    Promise.all([getBookPositions(), getBookEntities(), getBookAgreements(), getBookPortfolios()])
      .then(([held, legal, signed, tree]) => {
        if (!live) return;
        dispatch({ type: 'PAPER_LOADED', paper: {
          positions: held.positions, entities: legal.entities, agreements: signed.agreements,
          unstated: signed.settlement_state, nodes: tree.portfolios.map((node) => node.path) } });
        answered.current.paper = clock;
      })
      .catch((error) => { if (live) dispatch({ type: 'PAPER_FAILED', ...failure(error) }); });
    return () => { live = false; };
  }, [clock, paperWanted]);

  // What the seat may do, asked again where the seat or the record's head moved. A refusal - no
  // seat named, under grants - stands in its place; any other failure leaves acts to the service.
  const grantsAt = worklistAt(state.source?.kind === 'book' ? state.record.spine : null,
                              state.seat);
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
