import { useEffect, useState } from "react";
import { ClusterQueuePanel } from "./components/ClusterQueuePanel";
import { FileSearchPanel } from "./components/FileSearchPanel";
import type { DashboardState, ClusterQueueSnapshot, FileSearchResponse } from "./types";

interface AppProps {
  initialState: DashboardState;
}

/**
 * Phase 6 scaffold root component.
 *
 * The legacy bundle (frontend/static/js/dashboard_app.js) still ships every
 * page. This Vite/React/TS bundle currently renders a single migrated
 * "/cluster-queue-preview" demo page so we can validate the build pipeline
 * end-to-end without breaking the existing UI. Subsequent migrations
 * (Overview, MediaLibrary, YouTube, Diarization, FineTuning, TrainingLabels)
 * will land alongside this component, one tab at a time.
 *
 * The legacy bundle and this bundle share the same `dashboard-state` JSON
 * payload so they can coexist while we cut over.
 */
export default function App({ initialState }: AppProps) {
  const routes = initialState.routes ?? {};
  return (
    <main className="vite-dashboard-preview">
      <header className="vite-dashboard-header">
        <h1>{initialState.appTitle ?? "ML Speech Diarization"}</h1>
        <p>
          Vite + React + TypeScript build (Phase 6 scaffold). The legacy
          createElement bundle still drives every page; this preview proves
          the new pipeline works against the same JSON state.
        </p>
      </header>

      <ClusterQueuePanel
        url={routes.clusterQueue ?? "/api/cluster-queue"}
      />

      <FileSearchPanel
        url={routes.fileSearch ?? "/api/file-search"}
      />
    </main>
  );
}
