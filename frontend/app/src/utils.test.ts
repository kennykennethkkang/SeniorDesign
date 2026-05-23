// Unit tests for the small pure helpers used by the dashboard components.
// These run under vitest without needing jsdom, since every function under
// test is a plain string/array operation.

import { describe, expect, it } from "vitest";
import {
  formatFetchedAtUtc,
  limitMatches,
  refreshNoteText,
  searchEndpoint,
} from "./utils";

describe("formatFetchedAtUtc", () => {
  it("converts an ISO timestamp to the dashboard display form", () => {
    expect(formatFetchedAtUtc("2026-05-22T14:25:30.123Z")).toBe("2026-05-22 14:25:30");
  });

  it("drops the fractional seconds when they are present without a trailing Z", () => {
    expect(formatFetchedAtUtc("2026-05-22T14:25:30.000456")).toBe("2026-05-22 14:25:30");
  });

  it("leaves a timestamp without fractional seconds alone (apart from the T separator)", () => {
    expect(formatFetchedAtUtc("2026-05-22T14:25:30")).toBe("2026-05-22 14:25:30");
  });

  it("returns the empty string for an empty input so the footer can suppress the prefix", () => {
    expect(formatFetchedAtUtc("")).toBe("");
  });
});

describe("refreshNoteText", () => {
  it("includes the timestamp when one is available", () => {
    expect(refreshNoteText("2026-05-22 14:25:30", 5000)).toBe(
      "Last refreshed 2026-05-22 14:25:30 UTC. Refreshes every 5 s.",
    );
  });

  it("falls back to just the polling cadence on the first render", () => {
    expect(refreshNoteText("", 5000)).toBe("Refreshes every 5 s.");
  });

  it("rounds the interval to the nearest second so 2750 ms reads as 3 s", () => {
    expect(refreshNoteText("", 2750)).toBe("Refreshes every 3 s.");
  });
});

describe("searchEndpoint", () => {
  it("appends the encoded query as a single ?q= parameter", () => {
    expect(searchEndpoint("/api/file-search", "017_call")).toBe(
      "/api/file-search?q=017_call",
    );
  });

  it("URL-encodes characters that would otherwise break the query string", () => {
    expect(searchEndpoint("/api/file-search", "A & B?")).toBe(
      "/api/file-search?q=A%20%26%20B%3F",
    );
  });
});

describe("limitMatches", () => {
  it("returns at most the requested number of items (default 12)", () => {
    const items = Array.from({ length: 20 }, (_, i) => i);
    expect(limitMatches(items)).toHaveLength(12);
    expect(limitMatches(items, 5)).toEqual([0, 1, 2, 3, 4]);
  });

  it("returns an empty array when given null or undefined", () => {
    expect(limitMatches(undefined)).toEqual([]);
    expect(limitMatches(null)).toEqual([]);
  });

  it("returns the original list unchanged when it is already short enough", () => {
    expect(limitMatches([1, 2, 3])).toEqual([1, 2, 3]);
  });
});
