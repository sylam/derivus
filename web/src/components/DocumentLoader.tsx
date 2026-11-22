import { useRef, useState } from 'react';
import { BOOK_DOCUMENT, failure, getBook, openBook } from '../api';
import { BACK_TO_BOOK, fileName, leavingAsks, recordedCopy } from '../desk';
import { bookLoaded, useApp } from '../state';
import type { JobDoc } from '../types';

/** Where the document comes from: the book the service serves, kept fresh by the etag poll in App
 * - its own, or a job file opened here, which the service files and serves in its place until the
 * user steps back - or, where the service will not serve that file, a read-only view of it, which
 * stops the poll until the user steps back to the book, a desk that records saying why in a note
 * (`recordedCopy`). Leaving an opened document that may hold edits made here asks first
 * (`leavingAsks`). */
export function DocumentLoader() {
  const { state, dispatch } = useApp();
  // what the last open here answered: the copy it filed and the etag that copy stood at
  const seen = useRef<{ path: string; etag: string } | null>(null);
  const [asking, setAsking] = useState<{ said: string; go: () => void } | null>(null);

  function leave(go: () => void) {
    const said = state.source?.kind === 'book' ? leavingAsks(state.source, seen.current) : null;
    if (said) setAsking({ said, go });
    else go();
  }

  async function openFile(file: File) {
    let doc: JobDoc;
    try {
      doc = JSON.parse(await file.text()) as JobDoc;
    } catch (error) {
      dispatch({ type: 'DOC_FAILED', error: `${file.name} is not valid JSON: ${error}` });
      return;
    }
    try {
      seen.current = await openBook(file.name, doc);
    } catch (error) {
      // the service will not serve it: the file as a read-only view, under the service's reason
      const refused = failure(error);
      const note = recordedCopy(refused);
      dispatch({ type: 'DOC_LOADED', doc, source: { kind: 'local', name: file.name, note } });
      if (!note) dispatch({ type: 'DOC_FAILED', error: refused.error });
      return;
    }
    await backToBook();
  }

  async function backToBook() {
    try {
      dispatch(bookLoaded(await getBook()));
    } catch (error) {
      dispatch({ type: 'DOC_FAILED', error: failure(error).error });
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
              : `live book · ${fileName(state.source.path)}`}
          </span>
          {state.source.opened && (
            <button className="ghost" onClick={() => leave(closeOpened)}
                    title={`serve the desk's own book again - an edit made to ${state.source.opened} here is kept only in a download of it`}>
              {BACK_TO_BOOK}
            </button>
          )}
          <a className="ghost" href={BOOK_DOCUMENT} download
             title="the served document as the file it is, every edit made to it included">
            download job</a>
        </>
      )}
      {state.source?.kind === 'local' && (
        <>
          <span className="source-chip local">local · {state.source.name}</span>
          <button className="ghost" onClick={backToBook}
                  title="leave this read-only copy and show the book the service serves">
            {BACK_TO_BOOK}
          </button>
        </>
      )}
      <label className="filelabel">
        open a job file
        <input
          type="file"
          accept=".json,application/json"
          onChange={(e) => {
            // emptied once read, so the file just chosen can be chosen again
            const file = e.target.files?.[0];
            e.target.value = '';
            if (file) leave(() => openFile(file));
          }}
        />
      </label>
      {asking && (
        <span className="banner">
          {asking.said}
          <button className="ghost" onClick={() => { setAsking(null); asking.go(); }}>go on</button>
          <button className="ghost" onClick={() => setAsking(null)}>stay</button>
        </span>
      )}
    </>
  );
}
