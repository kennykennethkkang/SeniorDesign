// Small pure helpers extracted from the dashboard components so they can be
// unit-tested in isolation. Each function is a direct equivalent of the inline
// expression it replaced, so the components render the same strings as before.

/**
 * Convert an ISO timestamp like ``2026-05-22T14:25:30.123Z`` to the display
 * form ``2026-05-22 14:25:30`` used in the dashboard footer notes. Returns the
 * empty string when the input is empty.
 */
export function formatFetchedAtUtc(iso: string): string {
  if (!iso) return "";
  return iso.replace("T", " ").replace(/\..*$/, "");
}

/**
 * Build the cluster-queue footer line. Drops the "Last refreshed" prefix when
 * we have no timestamp yet so the panel still tells the user how often it
 * polls during the very first render.
 */
export function refreshNoteText(fetchedAt: string, refreshIntervalMs: number): string {
  const seconds = Math.round(refreshIntervalMs / 1000);
  if (fetchedAt) {
    return `Last refreshed ${fetchedAt} UTC. Refreshes every ${seconds} s.`;
  }
  return `Refreshes every ${seconds} s.`;
}

/**
 * Build the file-search request URL. Encodes the query so symbols like ``&``
 * or spaces in audio filenames do not break the GET request.
 */
export function searchEndpoint(baseUrl: string, query: string): string {
  return `${baseUrl}?q=${encodeURIComponent(query)}`;
}

/**
 * Defensive ``slice`` used by the file-search result list. The matches array
 * is typed as required, but older payloads occasionally omit it; falling back
 * to an empty array keeps the renderer simple.
 */
export function limitMatches<T>(matches: T[] | null | undefined, max = 12): T[] {
  return (matches ?? []).slice(0, max);
}
