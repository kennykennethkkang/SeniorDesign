import { useEffect, useMemo, useState } from "react";
import type { ClusterQueueSnapshot } from "../types";

interface Props {
  url: string;
  refreshIntervalMs?: number;
}

/**
 * TypeScript port of the legacy bundle's ClusterQueuePanel.
 *
 * Same JSON contract, same caching semantics on the backend, same "you"
 * highlighting. Rewritten as a real component with typed props so the rest
 * of the migrated dashboard can compose it instead of copy-pasting the
 * createElement version.
 */
export function ClusterQueuePanel({ url, refreshIntervalMs = 5000 }: Props) {
  const [snapshot, setSnapshot] = useState<ClusterQueueSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stateFilter, setStateFilter] = useState("PD,R,CG");
  const [partitionFilter, setPartitionFilter] = useState("");
  const [userFilter, setUserFilter] = useState("");

  useEffect(() => {
    let cancelled = false;
    const run = () => {
      const params = new URLSearchParams();
      if (stateFilter) params.set("state", stateFilter);
      if (partitionFilter) params.set("partition", partitionFilter);
      if (userFilter) params.set("user", userFilter);
      const target = params.toString() ? `${url}?${params.toString()}` : url;
      fetch(target, { cache: "no-store", headers: { Accept: "application/json" } })
        .then((response) => {
          if (!response.ok) throw new Error(`Cluster queue request failed with ${response.status}`);
          return response.json() as Promise<ClusterQueueSnapshot>;
        })
        .then((payload) => {
          if (cancelled) return;
          setSnapshot(payload);
          setError(null);
          setLoading(false);
        })
        .catch((err: Error) => {
          if (cancelled) return;
          setError(err.message);
          setLoading(false);
        });
    };
    run();
    const timer = window.setInterval(run, refreshIntervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [url, stateFilter, partitionFilter, userFilter, refreshIntervalMs]);

  const jobs = useMemo(() => snapshot?.jobs ?? [], [snapshot]);
  const me = snapshot?.current_user ?? "";
  const fetchedAt = snapshot?.fetched_at_utc
    ? snapshot.fetched_at_utc.replace("T", " ").replace(/\..*$/, "")
    : "";

  return (
    <article className="panel cluster-queue-panel">
      <div className="panel-head">
        <div>
          <h2>Cluster Queue</h2>
          <p>
            Live view of every job currently known to <code>squeue</code>.
            Your jobs are highlighted.
          </p>
        </div>
      </div>

      <div className="cluster-queue-controls">
        <label className="field">
          <span className="label">States</span>
          <select value={stateFilter} onChange={(e) => setStateFilter(e.target.value)}>
            <option value="PD,R,CG">Pending + Running + Completing</option>
            <option value="PD">Pending only</option>
            <option value="R">Running only</option>
            <option value="PD,R">Pending + Running</option>
          </select>
        </label>
        <label className="field">
          <span className="label">Partition contains</span>
          <input
            type="text"
            value={partitionFilter}
            onChange={(e) => setPartitionFilter(e.target.value)}
            placeholder="e.g. gpu"
          />
        </label>
        <label className="field">
          <span className="label">User contains</span>
          <input
            type="text"
            value={userFilter}
            onChange={(e) => setUserFilter(e.target.value)}
            placeholder={me ? `e.g. ${me}` : "username"}
          />
        </label>
      </div>

      {error ? (
        <p className="empty">{error}</p>
      ) : !snapshot?.available ? (
        <p className="empty">{snapshot?.message ?? "squeue is not available on this machine."}</p>
      ) : loading && jobs.length === 0 ? (
        <p className="empty">Loading cluster queue...</p>
      ) : jobs.length === 0 ? (
        <p className="empty">{snapshot.message ?? "Cluster queue is empty for the selected filters."}</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Job ID</th>
              <th>User</th>
              <th>State</th>
              <th>Partition</th>
              <th>Elapsed</th>
              <th>Time left</th>
              <th>Nodes</th>
              <th>CPUs</th>
              <th>Name</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr
                key={job.job_id}
                className={job.is_self ? "cluster-queue-row cluster-queue-row--self" : "cluster-queue-row"}
              >
                <td>{job.job_id}</td>
                <td>{job.is_self ? <strong>{job.user} (you)</strong> : job.user}</td>
                <td>
                  <span className={`status-pill status-${job.state.toLowerCase()}`}>{job.state}</span>
                </td>
                <td>{job.partition}</td>
                <td>{job.time_used}</td>
                <td>{job.time_left}</td>
                <td>{job.nodes}</td>
                <td>{job.cpus}</td>
                <td title={job.name}>{job.name}</td>
                <td title={job.reason}>{job.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <p className="footer-note">
        {fetchedAt
          ? `Last refreshed ${fetchedAt} UTC. Refreshes every ${Math.round(refreshIntervalMs / 1000)} s.`
          : `Refreshes every ${Math.round(refreshIntervalMs / 1000)} s.`}
      </p>
    </article>
  );
}
