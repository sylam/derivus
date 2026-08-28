import { useEffect, useReducer } from 'react';
import { failure, getBook, getBookRisk, getBookXva, getSchema, postDescribe } from './api';
import { DocumentLoader } from './components/DocumentLoader';
import { JobHeader } from './components/JobHeader';
import { WORKSPACES } from './registry';
import { AppContext, INITIAL, reducer } from './state';

const BOOK_POLL_MS = 2000;

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
      .catch((error) => dispatch({ type: 'SCHEMA_FAILED', error: String(error) }));
  }, []);

  // the live book, if the service serves one - silence (a 404) just means "open a file instead"
  useEffect(() => {
    getBook()
      .then((live) => dispatch({
        type: 'DOC_LOADED', doc: live.document,
        source: { kind: 'book', etag: live.etag, path: live.path },
      }))
      .catch(() => undefined);
  }, []);

  // the etag poll: a deal booked by ANY client (MCP, Excel, an editor on the file) appears here
  // within a tick, the user's place preserved
  useEffect(() => {
    if (state.source?.kind !== 'book') return;
    const etag = state.source.etag;
    const timer = setInterval(async () => {
      try {
        const live = await getBook();
        if (live.etag !== etag) {
          dispatch({
            type: 'DOC_LOADED', doc: live.document, refresh: true,
            source: { kind: 'book', etag: live.etag, path: live.path },
          });
        }
      } catch { /* the poll outlives a service restart */ }
    }, BOOK_POLL_MS);
    return () => clearInterval(timer);
  }, [state.source]);

  // The desk's two data views ride the SAME etag: the poll above is the only clock in the client,
  // and a booking, an amendment or a market tick moves the book here and the numbers follow.
  //
  // The risk verb RUNS the book on a miss, so it is fetched for a desk that is looking at it -
  // the open tab, or a tab it has already been on this session - and never speculatively for a
  // user who has not asked. `state.source` is a new object only when the etag moved, so this
  // fires once per move rather than once per poll tick.
  const riskWanted = state.tab === 'risk' || state.risk.data !== null;
  useEffect(() => {
    if (state.source?.kind !== 'book' || !riskWanted) return;
    const bookEtag = state.source.etag;
    let live = true;
    dispatch({ type: 'RISK_FETCHING' });
    getBookRisk()
      .then((risk) => { if (live) dispatch({ type: 'RISK_LOADED', risk, bookEtag }); })
      .catch((error) => { if (live) dispatch({ type: 'RISK_FAILED', ...failure(error) }); });
    return () => { live = false; };
  }, [state.source, riskWanted]);

  // The XVA view is a file read, not a run: it re-reads on FOCUS - a recalc asked for through the
  // MCP verbs lands in that file while nobody is looking at this screen - and again whenever the
  // book moves under it, which is what turns a set that has left the book into an orphan row.
  const onXva = state.tab === 'xva';
  useEffect(() => {
    if (state.source?.kind !== 'book' || !onXva) return;
    let live = true;
    dispatch({ type: 'XVA_FETCHING' });
    getBookXva()
      .then((xva) => { if (live) dispatch({ type: 'XVA_LOADED', xva }); })
      .catch((error) => { if (live) dispatch({ type: 'XVA_FAILED', ...failure(error) }); });
    return () => { live = false; };
  }, [state.source, onXva]);

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
    <AppContext.Provider value={{ state, dispatch }}>
      <div className="app">
        <header className="masthead">
          <h1><span>▮</span> derivus</h1>
          <DocumentLoader />
          <span className="spacer" />
          {state.schema && <span className="version">engine {state.schema.engine_version}</span>}
        </header>
        <JobHeader />
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
      </div>
    </AppContext.Provider>
  );
}
