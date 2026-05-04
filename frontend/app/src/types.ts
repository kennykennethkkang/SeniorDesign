// Shared types extracted from the dashboard JSON state shape produced by
// dashboard/app/rendering.py::frontend_state. Names mirror the JSON keys so
// migrating components from the legacy bundle is a copy + type-tighten.

export interface DashboardRoutes {
  tracking?: string;
  pageState?: string;
  clusterQueue?: string;
  runtimeEstimate?: string;
  fileSearch?: string;
  [key: string]: string | undefined;
}

export interface DashboardState {
  appTitle?: string;
  currentPath?: string;
  routes?: DashboardRoutes;
  [key: string]: unknown;
}

export interface ClusterQueueJob {
  job_id: string;
  user: string;
  state: string;
  partition: string;
  time_used: string;
  time_left: string;
  reason: string;
  name: string;
  account: string;
  cpus: string;
  nodes: string;
  is_self: boolean;
}

export interface ClusterQueueSnapshot {
  available: boolean;
  fetched_at_utc: string;
  current_user: string;
  states_filter: string;
  jobs: ClusterQueueJob[];
  message: string | null;
  filtered?: boolean;
}

export interface FileSearchMatch {
  name: string;
  kind: string;
  rel_path: string;
  size: number;
  run_name: string;
  href?: string;
}

export interface FileSearchGroup {
  stem: string;
  matches: FileSearchMatch[];
  match_count: number;
}

export interface FileSearchResponse {
  query: string;
  results: FileSearchGroup[];
  total_stems: number;
}
