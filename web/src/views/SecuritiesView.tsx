import { Fragment, useState, type ReactNode } from 'react';
import { configureSecurities, failure, getSecurities, verifySecurities } from '../api';
import { DataTable } from '../components/DataTable';
import { Gated } from '../components/Desk';
import { EmptyState } from '../components/EmptyState';
import { EditableScalar, FieldView } from '../components/FieldView';
import { Navigator, type Page } from '../components/Navigator';
import { offBook } from '../desk';
import {
  USED_COLUMNS, mapPages, securityRows, mergeRequest, rejectedRows, seedDescriptor, setAt, usedRows,
  verifyRequest, withMember,
} from '../securities';
import { heldBack, runHeld, useApp, useQueued, useRead } from '../state';
import { isObject } from '../tokens';
import type { SecuritiesAnswer, SeedOutcome, UsedRow, VerifyOutcome } from '../types';

const PANES = [
  { id: 'vocabulary', label: 'the vocabulary' },
  { id: 'evidence', label: 'the evidence' },
  { id: 'prints', label: 'every knot' },
];

/** One field of the entry on screen, at its path. */
type Edit = (path: string[], value: unknown) => void;

/** A list the desk edits member by member: each an input of the shape it stands in, an ✕ that
 * drops one, and one button that appends another of the shape the last member has - its members
 * read, where nothing edits it. */
function SeedList({ path, members, onEdit }: {
  path: string[]; members: unknown[]; onEdit?: Edit;
}) {
  if (!onEdit) return <span className="mono">{members.join(', ')}</span>;
  const blank = typeof members[members.length - 1] === 'number' ? 0 : '';
  return (
    <span className="editcell">
      {members.map((member, index) => (
        <span key={index} className="editcell">
          <EditableScalar name={String(index)} value={member}
                          descriptor={seedDescriptor(String(index), member)}
                          onAmend={async (_, wire) => {
                            onEdit(path, withMember(members, index, wire));
                            return null;
                          }} />
          <button className="ghost"
                  onClick={() => onEdit(path, withMember(members, index, null))}>✕</button>
        </span>
      ))}
      <button className="ghost"
              onClick={() => onEdit(path, withMember(members, members.length, blank))}>add</button>
    </span>
  );
}

/** One value of a seed entry, dispatched on its SHAPE - the seed is the desk's own JSON and no
 * declaration of the engine's describes it, so the form a value stands in is the form an edit
 * puts back. A dict is its own rows one level in; a list edits member by member. */
function SeedValue({ path, value, onEdit }: { path: string[]; value: unknown; onEdit?: Edit }) {
  if (isObject(value)) return <SeedFields path={path} entry={value} onEdit={onEdit} />;
  if (Array.isArray(value)) return <SeedList path={path} members={value} onEdit={onEdit} />;
  return onEdit ? (
    <EditableScalar name={path[path.length - 1] ?? ''} value={value}
                    descriptor={seedDescriptor(path[path.length - 1] ?? '', value)}
                    onAmend={async (_, wire) => { onEdit(path, wire); return null; }} />
  ) : <FieldView value={value} />;
}

function SeedFields({ path, entry, onEdit }: {
  path: string[]; entry: Record<string, unknown>; onEdit?: Edit;
}) {
  return (
    <div className="fields">
      {Object.entries(entry).map(([key, value]) => (
        <Fragment key={key}>
          <div className="k">{key}</div>
          <div className="v">
            <SeedValue path={[...path, key]} value={value} onEdit={onEdit} />
          </div>
        </Fragment>
      ))}
    </div>
  );
}

/** One vocabulary entry: the candidates a desk claims under one key of one block, edited in place
 * and merged by one button - read only where the seat may not mark. A refusal is the service's
 * own sentence with the form STANDING, so the spelling it refused is still there to fix; a write
 * answers with the tickers that block now spells and the file it kept beside the one it
 * replaced. */
function SeedCard({ block, name, entry, onSaved }: {
  block: string; name: string; entry: unknown; onSaved: () => void;
}) {
  const hold = heldBack(useApp().state, 'ticker_vocabulary');
  const [edited, setEdited] = useState<unknown>(null);
  const [outcome, setOutcome] = useState<SeedOutcome | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const value = edited ?? entry;

  async function save() {
    setSaving(true);
    setRefused(null);
    setOutcome(null);
    try {
      setOutcome(await configureSecurities(mergeRequest(block, name, value)));
      setEdited(null);
      onSaved();
    } catch (error) {
      setRefused(failure(error).error);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="card">
      <h3>{block} — {name}</h3>
      <SeedValue path={[]} value={value}
                 onEdit={hold ? undefined : (at, next) => setEdited(setAt(value, at, next))} />
      {!hold && <div className="pager">
        <button className="ghost" disabled={saving || edited === null}
                onClick={() => void save()}>save {name}</button>
        {edited !== null && <span className="chip">edited</span>}
        {saving && <span className="chip running">writing</span>}
        {outcome && <span className="chip done">{outcome.candidates.length} tickers</span>}
        {outcome?.backup && <span className="mono">kept {outcome.backup}</span>}
      </div>}
      {refused && <div className="error-box">{refused}</div>}
    </section>
  );
}

/** THE VOCABULARY: the seed as the service completes it - the packaged questionnaire with the
 * desk's own over it - one page per key each block files its entries by. */
const vocabularyPages = (seed: SecuritiesAnswer['seed'], onSaved: () => void): Page[] =>
  Object.keys(seed).sort().flatMap((block) => Object.keys(seed[block] ?? {}).map((key) => ({
    id: `${block}/${key}`, folder: block, label: key,
    content: <SeedCard block={block} name={key} entry={seed[block][key]} onSaved={onSaved} />,
  })));

type Verification = { start: (scope: { block?: string }) => void; running: boolean; strip: ReactNode };

/** The terminal round trip as one queued job, polled the way an execute is - one at a time, this
 * desk having one terminal. The submission's own refusal (a workstation whose blpapi does not
 * import) is held HERE, so it is said ONCE however many buttons asked for it, and it is a notice
 * rather than an error: the vocabulary reads perfectly well on a workstation that cannot verify. */
function useVerification(onDone: () => void): Verification {
  const [scope, setScope] = useState('');
  const [outcome, setOutcome] = useState<VerifyOutcome | null>(null);
  const asking = useQueued(verifySecurities, 'Securities', (answer?: VerifyOutcome) => {
    setOutcome(answer ?? {});
    onDone();
  });
  const progress = asking.run?.progress;
  const status = asking.run?.status ?? (outcome && 'done');
  const drifted = Object.entries(outcome?.drifted ?? {});
  const added = Object.entries(outcome?.added ?? {});
  return {
    start: (asked) => {
      setOutcome(null);
      setScope(asked.block ?? 'the whole map');
      asking.start(verifyRequest(asked));
    },
    running: !!asking.run,
    strip: (
      <>
        {status && (
          <div className="statusrow">
            <span className={`chip ${status === 'done' ? 'done' : 'running'}`}>{status}</span>
            <span className="mono">{scope}</span>
            {progress && (
              <span className="mono">{progress.note}
                {progress.total ? ` ${progress.done}/${progress.total}` : ''}</span>
            )}
            {outcome && (
              <span className="hint">
                {(outcome.verified ?? []).length} re-verified · {added.length} new ·{' '}
                {drifted.length} drifted · {outcome.seconds}s
              </span>
            )}
            {(outcome?.unknown ?? []).length > 0 && (
              <span className="chip">the map does not carry {outcome!.unknown!.join(', ')}</span>
            )}
          </div>
        )}
        {asking.error && <div className="banner">{asking.error}</div>}
        {drifted.length > 0 && (
          <DataTable columns={['entry', 'security', 'drift']} index={[]}
                     data={drifted.map(([path, drift]) => [path, drift.security, drift.drift])} />
        )}
        {added.length > 0 && (
          <DataTable columns={['security', 'verdict']} index={[]} data={added} />
        )}
      </>
    ),
  };
}

/** THE EVIDENCE: what a terminal answered, filed by block - each entry under the path a drift is
 * named by, with the button that asks again where `verify` is the seat's - and the ledger of what
 * was rejected and why. */
function evidencePages(answer: SecuritiesAnswer, verify?: Verification): Page[] {
  const rejected = rejectedRows(answer.map.rejected);
  return [
    ...mapPages(answer.map.blocks).map(({ block, key, node }) => {
      const rows = securityRows(node, key === null ? [block] : [block, key]);
      return {
        id: key === null ? block : `${block}/${key}`, label: key ?? block,
        folder: key === null ? undefined : block,
        content: (
          <section className="card">
            <h3>{key === null ? block : `${block} — ${key}`} — {rows.length} verified</h3>
            {verify && (
              <div className="pager">
                <button className="ghost" disabled={verify.running}
                        onClick={() => verify.start({ block })}>verify {block}</button>
              </div>
            )}
            <DataTable columns={['entry', 'security', 'name', 'last print', 'verified']} index={[]}
                       data={rows.map(({ path, entry }) => [
                         path, entry.security, entry.name, entry.last_update, entry.verified])} />
          </section>
        ),
      };
    }),
    ...(rejected.length ? [{ id: 'rejected', label: `rejected (${rejected.length})`, content: (
      <section className="card">
        <h3>rejected — {rejected.length}</h3>
        <DataTable columns={['ticker', 'verdict', 'name', 'last print', 'error']} index={[]}
                   data={rejected.map((row) => [
                     row.security, row.verdict, row.name, row.last_update, row.error])}
                   cell={(r, c) => (c === 1
                     ? <span className="chip error">{rejected[r].verdict}</span> : undefined)} />
      </section>
    ) }] : []),
  ];
}

/** EVERY KNOT NAMES ITS PRINT: the book's own quote rows, a page per curve, with the print each was
 * solved from and whatever a terminal ever said about the security it came off. The IPV read -
 * nothing here edits anything, and the verdict is the service's word. */
const printPages = (used: UsedRow[]): Page[] =>
  [...new Set(used.map((row) => row.curve))].map((curve) => {
    const rows = usedRows(used, curve);
    return {
      id: curve, label: curve, content: (
        <section className="card">
          <h3>{curve} — {rows.length} knots</h3>
          <DataTable columns={[...USED_COLUMNS]} index={[]}
                     data={rows.map((row) => USED_COLUMNS.map((key) => row[key]))}
                     cell={(r, c) => (USED_COLUMNS[c] === 'verdict'
                       ? <span className={`chip ${rows[r].tone}`}>{rows[r].verdict}</span>
                       : undefined)} />
        </section>
      ),
    };
  });

/** The ticker vocabulary, its evidence and every knot's print - the three halves of one read. The
 * seed is what this desk CLAIMS it could quote, the map is what a terminal answered about those
 * claims, and the join is which print each knot of the book was solved from. */
export function SecuritiesView() {
  const { state, dispatch } = useApp();
  const pane = state.selection.picks.securities ?? PANES[0].id;
  const [reload, setReload] = useState(0);
  const source = state.source;
  const verify = useVerification(() => setReload((count) => count + 1));

  // the read rides the etag poll like every other screen, and a write here or a verification that
  // rewrote the map asks for it again
  const { data: answer, error } = useRead(source?.kind === 'book' ? getSecurities : null,
                                          [source, reload]);

  if (source?.kind !== 'book') {
    return <EmptyState {...offBook('the tickers this desk could quote, what a terminal verified '
                                   + 'about them and the print behind every knot')} />;
  }
  if (error) {
    return <div className="main"><div className="panel">
      <div className="error-box">{error}</div></div></div>;
  }
  if (!answer) {
    return <EmptyState title="Reading the desk's vocabulary…"
                       hint="The seed, the map and every knot's print, in one read." />;
  }

  const saved = () => setReload((count) => count + 1);
  // a seat that may not write the vocabulary or the map reads them and the evidence, told once
  // what it needs; a verification writes the map and is a job of the queue
  const [named, hold] = [heldBack(state, 'ticker_vocabulary'), heldBack(state, 'security_map')];
  const queueHold = runHeld(state);
  const panes: Record<string, { pages: Page[]; empty: string; head?: ReactNode }> = {
    vocabulary: { pages: vocabularyPages(answer.seed, saved), empty: 'the seed names nothing',
                  head: named && <div className="note">{named}</div> },
    evidence: {
      pages: evidencePages(answer, hold ?? queueHold ? undefined : verify),
      empty: answer.provisioned ? 'the map carries no entry'
        : 'this home has never been verified: the vocabulary is a claim, and nothing has '
          + 'evidenced it',
      head: (
        <>
          {hold && <div className="note">{hold}</div>}
          <div className="statusrow">
            {!hold && <Gated label="verify the whole map" hold={queueHold}>
              <button className="ghost" disabled={verify.running}
                      onClick={() => verify.start({})}>verify the whole map</button>
            </Gated>}
            <span className="hint">generated <span className="mono">
              {answer.map.generated ?? 'never'}</span></span>
          </div>
          {verify.strip}
        </>
      ),
    },
    prints: {
      pages: printPages(answer.used),
      empty: 'this book carries no curve block, so no knot names a print yet',
    },
  };
  const shown = panes[pane];

  // one pane at a time, each its own tree under its own pick, and the vocabulary's cards kept
  // mounted - an entry edited and not yet saved survives a look at another
  return (
    <Navigator
      screen={`securities.${pane}`} pages={shown.pages} empty={shown.empty}
      keep={pane === 'vocabulary'}
      side={
        <div className="sidehead">
          {PANES.map((entry) => (
            <button key={entry.id} className={pane === entry.id ? 'on' : ''}
                    onClick={() => dispatch({ type: 'PICK', screen: 'securities', id: entry.id })}>
              {entry.label}
            </button>
          ))}
        </div>
      }
      head={
        <>
          <div className="pager">
            <span className="mono">{answer.home}</span>
            <span className={`chip${answer.provisioned ? ' done' : ''}`}>
              {answer.provisioned ? 'verified' : 'never verified'}</span>
          </div>
          {shown.head}
        </>
      } />
  );
}
