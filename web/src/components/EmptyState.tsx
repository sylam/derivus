/** A view with nothing to show, saying WHY and what would fill it. The honest counterpart to the
 * JSON fallback: a screen whose data is absent states the absence, and never renders a zero, a
 * blank table or a spinner that means nothing. */
export function EmptyState({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="main">
      <div className="placeholder" style={{ flex: 1 }}>
        <div>{title}</div>
        <div className="hint" style={{ maxWidth: 520, textAlign: 'center' }}>{hint}</div>
      </div>
    </div>
  );
}
