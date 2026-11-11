import { BOOK_DOCUMENT, failure, getBook, openBook } from '../api';
import { bookLoaded, useApp } from '../state';
import type { JobDoc } from '../types';

/** Where the document comes from: the book the service serves, kept fresh by the etag poll in App
 * - its own, or a job file opened here, which the service files and serves in its place until the
 * user steps back - or, where the service will not serve that file, a read-only view of it, which
 * stops the poll until the user steps back to the book. */
export function DocumentLoader() {
  const { state, dispatch } = useApp();

  async function openFile(file: File) {
    let doc: JobDoc;
    try {
      doc = JSON.parse(await file.text()) as JobDoc;
    } catch (error) {
      dispatch({ type: 'DOC_FAILED', error: `${file.name} is not valid JSON: ${error}` });
      return;
    }
    try {
      await openBook(file.name, doc);
    } catch (error) {
      // the service will not serve it: the file as a read-only view, under the service's reason
      dispatch({ type: 'DOC_LOADED', doc, source: { kind: 'local', name: file.name } });
      dispatch({ type: 'DOC_FAILED', error: failure(error).error });
      return;
    }
    await backToBook();
  }

  async function backToBook() {
    try {
      dispatch(bookLoaded(await getBook()));
    } catch {
      dispatch({ type: 'DOC_FAILED', error: 'The service is not serving a book' });
    }
  }

  // the desk's own book served again; a refusal leaves the opened document standing
  async function closeOpened() {
    try {
      await openBook();
    } catch (error) {
      dispatch({ type: 'DOC_FAILED', error: failure(error).error });
      return;
    }
    await backToBook();
  }

  return (
    <>
      {state.source?.kind === 'book' && (
        <>
          <span className="source-chip" title={state.source.path}>
            ● {state.source.opened ? `opened · ${state.source.opened}`
              : `live book · ${state.source.path.split('/').pop()}`}
          </span>
          {state.source.opened && (
            <button className="ghost" onClick={closeOpened}>back to book</button>
          )}
          <a className="ghost" href={BOOK_DOCUMENT} download
             title="the served document as the file it is, every edit made to it included">
            download job</a>
        </>
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
          onChange={(e) => e.target.files?.[0] && openFile(e.target.files[0])}
        />
      </label>
    </>
  );
}
