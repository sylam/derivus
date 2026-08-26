/** The honest fallback: any value the vocabulary cannot state renders as itself. This is how the
 * shapeless descriptors (transition matrices, deal maps, regime lists) stay VISIBLE without the
 * renderer inventing widgets for them. */
export function JsonView({ value }: { value: unknown }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}
