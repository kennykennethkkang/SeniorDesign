#!/usr/bin/env python3
"""Shared helpers for detached workflow runs launched from the local dashboard."""

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
    """Return a stable UTC timestamp for metadata files."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def process_is_running(pid: int) -> bool:
    """Check whether a local process still exists."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def slurm_job_state(job_id: str) -> str:
    """Return the current Slurm state for a submitted job when available."""

    if not job_id or not shutil.which("squeue"):
        return ""
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    states = [line.strip().lower() for line in completed.stdout.splitlines() if line.strip()]
    return states[0] if states else ""


def slurm_queue_snapshot(job_id: str) -> dict[str, object]:
    """Return queue position details for one Slurm job when `squeue` is available."""

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
            ["squeue", "-h", "-t", "PD,R,CG", "-o", "%i|%T|%R|%S"],
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
    if completed.returncode != 0:
        return {
            "job_id": normalized_job_id,
            "available": False,
            "message": (completed.stderr or completed.stdout or "squeue returned an error.").strip(),
        }

    pending_position = 0
    jobs_ahead = 0
    for line in completed.stdout.splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        row_job_id, state, reason, start_time = [part.strip() for part in parts]
        base_row_job_id = row_job_id.split(".", 1)[0]
        state_key = state.lower()
        if state_key in {"pending", "pd"}:
            pending_position += 1
        if base_row_job_id != normalized_job_id:
            if state_key in {"pending", "pd"}:
                jobs_ahead += 1
            continue
        if state_key in {"running", "r", "completing", "cg"}:
            return {
                "job_id": normalized_job_id,
                "available": True,
                "state": state,
                "queue_position": 0,
                "jobs_ahead": 0,
                "reason": reason,
                "estimated_start": start_time,
                "message": "Running now.",
            }
        return {
            "job_id": normalized_job_id,
            "available": True,
            "state": state,
            "queue_position": pending_position,
            "jobs_ahead": max(jobs_ahead, 0),
            "reason": reason,
            "estimated_start": start_time,
            "message": f"Pending position {pending_position}.",
        }

    return {
        "job_id": normalized_job_id,
        "available": True,
        "state": "not in queue",
        "queue_position": None,
        "jobs_ahead": None,
        "reason": "",
        "estimated_start": "",
        "message": "Job is not in squeue. It may have finished or left the queue.",
    }


def run_status(run_dir: Path) -> str:
    """Derive a readable run state from the standard metadata files."""

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
            state = slurm_job_state(slurm_job_id)
            if state:
                if state in {"running", "completing"}:
                    return "running"
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
