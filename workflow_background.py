#!/usr/bin/env python3
"""Shared helpers for detached workflow runs launched from the local dashboard.

These utilities decouple long-running cluster jobs from the dashboard's HTTP
request cycle. Once an sbatch job is handed off to SLURM, everything here is
about polling its state and surfacing that state to the frontend.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


def utc_now_iso() -> str:
    """Produce a microsecond-free UTC ISO timestamp so metadata files stay diff-friendly."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def process_is_running(pid: int) -> bool:
    """Check process liveness by sending signal 0. No actual signal is delivered."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# Short-lived cache for slurm_job_state results. The dashboard polls once per
# active run per second, so 50+ runs means 50 squeue calls every second. SLURM
# state changes slowly (PD to R to done over minutes), so a few-second TTL
# shaves off hundreds of subprocess calls per render with no visible lag.
# CLI processes exit before the TTL matters so they are unaffected.
_SLURM_JOB_STATE_CACHE: dict[tuple[str, bool], tuple[float, str]] = {}
_SLURM_JOB_STATE_CACHE_TTL = 3.0


def _slurm_job_state_cached(job_id: str, include_accounting: bool) -> str | None:
    """Return the cached job state if still fresh, else None."""

    entry = _SLURM_JOB_STATE_CACHE.get((job_id, include_accounting))
    if entry is None:
        return None
    expires_at, value = entry
    if time.monotonic() < expires_at:
        return value
    return None


def _slurm_job_state_store(job_id: str, include_accounting: bool, value: str) -> None:
    """Memoize one slurm_job_state result with a short TTL."""

    _SLURM_JOB_STATE_CACHE[(job_id, include_accounting)] = (
        time.monotonic() + _SLURM_JOB_STATE_CACHE_TTL,
        value,
    )


def slurm_job_state(job_id: str, *, include_accounting: bool = True) -> str:
    """Get the current SLURM state for a job, falling back to sacct when it's already left squeue."""

    normalized_job_id = str(job_id or "").strip().split(".", 1)[0]
    if not normalized_job_id:
        return ""

    cached = _slurm_job_state_cached(normalized_job_id, include_accounting)
    if cached is not None:
        return cached
    if shutil.which("squeue"):
        try:
            completed = subprocess.run(
                ["squeue", "-h", "-j", normalized_job_id, "-o", "%T"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            states = [line.strip().lower() for line in completed.stdout.splitlines() if line.strip()]
            if states:
                value = states[0]
                _slurm_job_state_store(normalized_job_id, include_accounting, value)
                return value
    if not include_accounting:
        _slurm_job_state_store(normalized_job_id, include_accounting, "")
        return ""
    accounting = slurm_accounting_snapshot(normalized_job_id)
    value = str(accounting.get("state") or "").lower()
    _slurm_job_state_store(normalized_job_id, include_accounting, value)
    return value


def slurm_accounting_snapshot(job_id: str) -> dict[str, object]:
    """Pull terminal job stats from sacct for jobs that have already finished and left the live queue."""

    normalized_job_id = str(job_id or "").strip().split(".", 1)[0]
    if not normalized_job_id or not shutil.which("sacct"):
        return {}
    fields = [
        "JobIDRaw",
        "State",
        "ExitCode",
        "Elapsed",
        "Timelimit",
        "Submit",
        "Start",
        "End",
        "NodeList",
        "NNodes",
        "NCPUS",
        "Partition",
        "JobName",
    ]
    try:
        completed = subprocess.run(
            ["sacct", "-n", "-P", "-j", normalized_job_id, f"--format={','.join(fields)}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    for raw in completed.stdout.splitlines():
        parts = [part.strip() for part in raw.split("|")]
        if len(parts) < len(fields):
            continue
        record = dict(zip(fields, parts))
        row_job_id = record.get("JobIDRaw", "").split(".", 1)[0]
        if row_job_id != normalized_job_id:
            continue
        state = record.get("State", "")
        return {
            "job_id": normalized_job_id,
            "available": True,
            "state": state,
            "state_source": "sacct",
            "exit_code": record.get("ExitCode", ""),
            "time_used": record.get("Elapsed", ""),
            "time_limit": record.get("Timelimit", ""),
            "submitted_at": record.get("Submit", ""),
            "started_at": record.get("Start", ""),
            "ended_at": record.get("End", ""),
            "nodes": record.get("NodeList", ""),
            "node_count": record.get("NNodes", ""),
            "cpus": record.get("NCPUS", ""),
            "partition": record.get("Partition", ""),
            "name": record.get("JobName", ""),
            "message": f"sacct reports {state}." if state else "Job has left squeue.",
        }
    return {}


def slurm_queue_snapshot(job_id: str) -> dict[str, object]:
    """Return live state + resource details for *one* SLURM job we submitted.

    Asks ``squeue -j <jobid>`` and parses just that one row. We do not scan
    the rest of the cluster because the dashboard only cares about our own
    jobs and the old cluster-wide scan was slow enough to appear in the
    per-second polling lag.

    The "queue_position" and "jobs_ahead" fields are absent by design.
    SlurmQueueTracker handles that and just hides those rows.
    """

    normalized_job_id = str(job_id or "").strip().split(".", 1)[0]
    if not normalized_job_id:
        return {}
    if not shutil.which("squeue"):
        return {
            "job_id": normalized_job_id,
            "available": False,
            "message": "squeue is not available on this machine.",
        }
    try:
        completed = subprocess.run(
            [
                "squeue",
                "-h",
                "-j",
                normalized_job_id,
                "-o",
                "%i|%T|%R|%S|%P|%j|%u|%M|%L|%D|%C|%b",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "job_id": normalized_job_id,
            "available": False,
            "message": "Unable to query squeue right now.",
        }
    # When a job has already left the queue, squeue exits non-zero with
    # "Invalid job id specified". That is the normal terminal path; we fall
    # through to sacct instead of treating it as an error.
    stdout = (completed.stdout or "").strip()
    if completed.returncode == 0 and stdout:
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 2:
                continue
            row_job_id = parts[0].split(".", 1)[0]
            if row_job_id != normalized_job_id:
                continue
            state = parts[1]
            reason = parts[2] if len(parts) > 2 else ""
            start_time = parts[3] if len(parts) > 3 else ""
            partition = parts[4] if len(parts) > 4 else ""
            name = parts[5] if len(parts) > 5 else ""
            user = parts[6] if len(parts) > 6 else ""
            time_used = parts[7] if len(parts) > 7 else ""
            time_left = parts[8] if len(parts) > 8 else ""
            nodes = parts[9] if len(parts) > 9 else ""
            cpus = parts[10] if len(parts) > 10 else ""
            gres = parts[11] if len(parts) > 11 else ""
            details = {
                "job_id": normalized_job_id,
                "available": True,
                "state": state,
                "reason": reason,
                "estimated_start": start_time,
                "partition": partition,
                "name": name,
                "user": user,
                "time_used": time_used,
                "time_left": time_left,
                "nodes": nodes,
                "cpus": cpus,
                "gres": gres,
                "fetched_at_utc": utc_now_iso(),
                "state_source": "squeue",
            }
            state_key = state.lower()
            if state_key in {"running", "r", "completing", "cg"}:
                details["message"] = "Running now."
            elif state_key in {"pending", "pd"}:
                details["message"] = "Pending in queue."
            else:
                details["message"] = state or "In queue."
            return details

    accounting = slurm_accounting_snapshot(normalized_job_id)
    if accounting:
        return {
            **accounting,
            "reason": "",
            "estimated_start": accounting.get("started_at", ""),
            "fetched_at_utc": utc_now_iso(),
            "message": f"Job left squeue; {accounting.get('message')}",
        }

    return {
        "job_id": normalized_job_id,
        "available": True,
        "state": "not in queue",
        "reason": "",
        "estimated_start": "",
        "fetched_at_utc": utc_now_iso(),
        "state_source": "squeue",
        "message": "Job is not in squeue. It may have finished or left the queue.",
    }


def run_status(run_dir: Path) -> str:
    """Return a human-readable run state by checking exit_code, then SLURM, then metadata."""

    exit_code_path = run_dir / "exit_code.txt"
    metadata_path = run_dir / "metadata.json"
    if exit_code_path.is_file():
        try:
            exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return "unknown"
        return "succeeded" if exit_code == 0 else "failed"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        slurm_job_id = str(metadata.get("slurm_job_id") or "").strip()
        if slurm_job_id:
            include_accounting = (
                str(metadata.get("runner") or "") == "slurm"
                or str(metadata.get("submission_status") or "") == "submitted"
                or bool(metadata.get("started_at_utc"))
            )
            state = slurm_job_state(slurm_job_id, include_accounting=include_accounting)
            if state:
                if state in {"running", "completing"}:
                    return "running"
                if state in {"completed"}:
                    return "succeeded"
                if any(
                    state.startswith(prefix)
                    for prefix in (
                        "failed",
                        "cancelled",
                        "timeout",
                        "out_of_memory",
                        "node_fail",
                        "preempted",
                        "boot_fail",
                        "deadline",
                    )
                ):
                    return "failed"
                return "submitted"
            return "submitted"
        if process_is_running(int(metadata.get("pid", 0))):
            return "running"
    return "stopped"


def launch_background_command(
    *,
    command: Sequence[str],
    cwd: Path,
    run_dir: Path,
    metadata: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run one command in the background and record the usual metadata files."""

    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    exit_code_path = run_dir / "exit_code.txt"
    wrapper_path = run_dir / "run.sh"
    metadata_path = run_dir / "metadata.json"
    command_display = " ".join(shlex.quote(part) for part in command)
    wrapper_lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "rc=0",
        f"if ! {command_display}; then",
        "  rc=$?",
        "fi",
        f"printf '%s\\n' \"$rc\" > {shlex.quote(str(exit_code_path))}",
        "exit \"$rc\"",
    ]
    wrapper_path.write_text("\n".join(wrapper_lines) + "\n", encoding="utf-8")
    wrapper_path.chmod(0o755)

    launch_command = (
        f"nohup bash {shlex.quote(str(wrapper_path))} "
        f"> {shlex.quote(str(stdout_path))} "
        f"2> {shlex.quote(str(stderr_path))} "
        "< /dev/null & echo $!"
    )
    completed = subprocess.run(
        ["bash", "-lc", launch_command],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **(dict(env) if env else {})},
    )
    if completed.returncode != 0:
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        exit_code_path.write_text(str(completed.returncode), encoding="utf-8")
        raise RuntimeError("Failed to launch background command.")

    pid_text = (completed.stdout or "").strip().splitlines()
    pid = int(pid_text[-1]) if pid_text and pid_text[-1].isdigit() else 0
    run_metadata = {
        "pid": pid,
        "command": list(command),
        "cwd": str(cwd),
        "started_at_utc": utc_now_iso(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "exit_code_path": str(exit_code_path),
        "wrapper_path": str(wrapper_path),
        **(dict(metadata) if metadata else {}),
    }
    metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "pid": pid,
        "run_dir": run_dir,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "exit_code_path": exit_code_path,
        "metadata_path": metadata_path,
    }


def wait_for_file(path: Path, *, timeout_seconds: float = 5.0) -> bool:
    """Wait briefly for a file created by a background process."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()
