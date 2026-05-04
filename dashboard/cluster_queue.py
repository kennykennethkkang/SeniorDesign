"""Cluster-wide Slurm queue snapshot.

Distinct from ``workflow_background.slurm_queue_snapshot`` (which reports the
position of one specific job): this module returns *every* job currently
known to ``squeue``, so the dashboard can surface who else is competing for
nodes alongside the user's runs.

Output format is a plain dict so it can be cached via the WSGI app's
``cached_value`` TTL helper and serialized straight to JSON for the frontend.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from workflow_background import utc_now_iso

# Pipe-delimited squeue format. Aligned with the columns the dashboard panel
# renders. Order matters and must match the parsing below.
_SQUEUE_FORMAT = "%i|%u|%T|%P|%M|%L|%R|%j|%a|%C|%D"
_SQUEUE_FIELDS = (
    "job_id",
    "user",
    "state",
    "partition",
    "time_used",
    "time_left",
    "reason",
    "name",
    "account",
    "cpus",
    "nodes",
)
_SQUEUE_TIMEOUT_SECONDS = 5
_DEFAULT_STATES = "PD,R,CG"


def current_username() -> str:
    """Return the current Unix username if it can be determined."""

    for getter in (lambda: os.environ.get("USER"), lambda: os.environ.get("LOGNAME")):
        value = getter()
        if value:
            return value
    try:
        return os.getlogin()
    except OSError:
        return ""


def cluster_queue_snapshot(states: str = _DEFAULT_STATES) -> dict[str, Any]:
    """Return a snapshot of every Slurm job currently in the cluster queue.

    The result has shape::

        {
            "available": bool,                  # is squeue runnable here?
            "fetched_at_utc": str,              # ISO-8601 UTC timestamp
            "current_user": str,                # whoami (for "you" highlight)
            "states_filter": str,               # the squeue -t value used
            "jobs": [                           # one entry per queue row
                {
                    "job_id": "...", "user": "...", "state": "...",
                    "partition": "...", "time_used": "...",
                    "time_left": "...", "reason": "...", "name": "...",
                    "account": "...", "cpus": "...", "nodes": "...",
                    "is_self": bool,
                },
                ...
            ],
            "message": str | None,              # populated on failure / empty
        }
    """

    if not shutil.which("squeue"):
        return {
            "available": False,
            "fetched_at_utc": utc_now_iso(),
            "current_user": current_username(),
            "states_filter": states,
            "jobs": [],
            "message": "squeue is not available on this machine.",
        }

    try:
        completed = subprocess.run(
            ["squeue", "-h", "-t", states, "-o", _SQUEUE_FORMAT],
            check=False,
            capture_output=True,
            text=True,
            timeout=_SQUEUE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "fetched_at_utc": utc_now_iso(),
            "current_user": current_username(),
            "states_filter": states,
            "jobs": [],
            "message": f"squeue timed out after {_SQUEUE_TIMEOUT_SECONDS}s.",
        }
    except OSError as exc:
        return {
            "available": True,
            "fetched_at_utc": utc_now_iso(),
            "current_user": current_username(),
            "states_filter": states,
            "jobs": [],
            "message": f"Unable to invoke squeue: {exc}",
        }

    me = current_username()

    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "squeue returned an error.").strip()
        return {
            "available": True,
            "fetched_at_utc": utc_now_iso(),
            "current_user": me,
            "states_filter": states,
            "jobs": [],
            "message": message,
        }

    jobs: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < len(_SQUEUE_FIELDS):
            continue
        record = {field: parts[index].strip() for index, field in enumerate(_SQUEUE_FIELDS)}
        record["is_self"] = bool(me) and record["user"] == me
        jobs.append(record)

    return {
        "available": True,
        "fetched_at_utc": utc_now_iso(),
        "current_user": me,
        "states_filter": states,
        "jobs": jobs,
        "message": None,
    }
