import { getBook } from '../api';
import { useApp } from '../state';
import type { JobDoc } from '../types';

/** Where the document comes from: the service's live book (the default, kept fresh by the etag
 * poll in App), or a local file dropped in for ad-hoc viewing - which stops the poll until the
 * user steps back to the book. */
export function DocumentLoader() {
  const { state, dispatch } = useApp();

  async function openLocal(file: File) {
    try {
      const doc = JSON.parse(await file.text()) as JobDoc;
      dispatch({ type: 'DOC_LOADED', doc, source: { kind: 'local', name: file.name } });
    } catch (error) {
      dispatch({ type: 'DOC_FAILED', error: `${file.name} is not valid JSON: ${error}` });
    }
  }

  async function backToBook() {
    try {
      const live = await getBook();
      dispatch({
        type: 'DOC_LOADED', doc: live.document,
        source: { kind: 'book', etag: live.etag, path: live.path },
      });
    } catch {
      dispatch({ type: 'DOC_FAILED', error: 'The service is not serving a book' });
    }
  }

  return (
    <>
      {state.source?.kind === 'book' && (
        <span className="source-chip" title={state.source.path}>
          ● live book · {state.source.path.split('/').pop()}
        </span>
      )}
      {state.source?.kind === 'local' && (
        <>
          <span className="source-chip local">local · {state.source.name}</span>
          <button className="ghost" onClick={backToBook}>back to book</button>
        </>
      )}
      <label className="filelabel">
        open a job file
        <input
          type="file"
          accept=".json,application/json"
          onChange={(e) => e.target.files?.[0] && openLocal(e.target.files[0])}
        />
      </label>
    </>
  );
}
