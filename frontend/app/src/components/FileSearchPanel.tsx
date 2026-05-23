import { useEffect, useState } from "react";
import type { FileSearchResponse } from "../types";
import { limitMatches, searchEndpoint } from "../utils";

interface Props {
  url: string;
  debounceMs?: number;
}

/**
 * TypeScript port of the legacy bundle's FileSearchPanel. Same JSON contract.
 */
export function FileSearchPanel({ url, debounceMs = 250 }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<FileSearchResponse["results"]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query) {
      setResults([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setLoading(true);
      fetch(searchEndpoint(url, query), {
        cache: "no-store",
        headers: { Accept: "application/json" },
      })
        .then((response) => response.json() as Promise<FileSearchResponse>)
        .then((payload) => {
          if (cancelled) return;
          setResults(payload.results || []);
          setLoading(false);
        })
        .catch(() => {
          if (!cancelled) setLoading(false);
        });
    }, debounceMs);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [url, query, debounceMs]);

  return (
    <article className="panel file-search-panel">
      <div className="panel-head">
        <div>
          <h2>Find a file across runs</h2>
          <p>
            Type any part of an audio file name to see every run, output,
            and source file that mentions it.
          </p>
        </div>
      </div>
      <input
        type="search"
        className="file-search-input"
        value={query}
        placeholder="e.g. 017_call"
        onChange={(e) => setQuery(e.target.value)}
        aria-label="File search"
      />
      {query && loading ? <p className="footer-note">Searching...</p> : null}
      {query && !loading && results.length === 0 ? (
        <p className="empty">No matches for &ldquo;{query}&rdquo;.</p>
      ) : null}
      {results.length > 0 ? (
        <ul className="file-search-results">
          {results.map((group) => (
            <li key={group.stem} className="file-search-group">
              <span className="file-search-stem">{group.stem}</span>
              <span className="file-search-count">{group.match_count} match(es)</span>
              <ul className="file-search-matches">
                {limitMatches(group.matches).map((m, i) => (
                  <li
                    key={`${m.rel_path}-${i}`}
                    className={`file-search-match file-search-match--${m.kind}`}
                  >
                    {m.href ? <a href={m.href}>{m.rel_path}</a> : m.rel_path}
                    <span className="file-search-kind">
                      {" "}({m.kind}{m.run_name ? ` · ${m.run_name}` : ""})
                    </span>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}
