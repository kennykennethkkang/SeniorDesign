"""Background auto-train runner — kicks off prepare + sbatch when labels complete.

The label completion flow can hand selected projects to this runner from the
review popup, and the older per-project auto-train toggle still uses the same
path. Either way, this module skips the manual prepare -> launch click cycle
and:

1. Acquires a per-project lock so two concurrent label completions on the same
   project don't race to launch two simultaneous training runs (which would
   collide on checkpoint dirs and produce a confusing version history).
2. Sets ``auto_train_pending: True`` in the display sidecar so the dashboard
   shows the "queued, waiting for current run" badge to the user.
3. Polls list_runs() until no run for that project is in {submitted, running}.
4. Runs prepare_project + launch_training (sbatch path) with sensible defaults.
5. Clears the pending flag so the dashboard reflects the new state.

If the sbatch submission fails we leave the pending flag clear and log the
error to the project's ``auto_train.log`` — there's no automatic retry, since
the user almost always wants to know about cluster issues themselves.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Mapping

import fine_tuning_manager as ftm
from workflow_background import utc_now_iso


# In-memory per-project locks. Single-host dashboard, single Python process —
# don't need a filesystem lock. The dict is keyed on "<backend>/<slug>".
_AUTO_TRAIN_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

# Polling cadence + ceiling. 30s feels fine for cluster jobs that take ~hour.
_POLL_INTERVAL_SECONDS = 30.0
_MAX_WAIT_SECONDS = 4 * 60 * 60  # four hours
_ACTIVE_RUN_STATUSES = {
    "configuring",
    "completing",
    "pending",
    "requeued",
    "resizing",
    "running",
    "signaling",
    "staged_out",
    "submitted",
    "suspended",
}


def _project_lock(backend: str, slug: str) -> threading.Lock:
    """Get-or-create the lock for a (backend, slug) pair."""

    key = f"{backend}/{slug}"
    with _LOCKS_GUARD:
        lock = _AUTO_TRAIN_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _AUTO_TRAIN_LOCKS[key] = lock
        return lock


def _has_active_run(project_name: str, backend: str, root: Path) -> bool:
    """Check whether any run is currently submitted or running for the project."""

    runs = ftm.list_runs(project_name, backend=backend, root=root, limit=8)
    for run in runs:
        status = str(run.get("status") or "").strip().lower()
        if status in _ACTIVE_RUN_STATUSES:
            return True
    return False


def _set_pending(project_name: str, *, backend: str, root: Path, pending: bool) -> None:
    """Update the auto_train_pending flag in display.json so the UI can surface it."""

    target = ftm.project_dir(project_name, backend=backend, root=root)
    if not target.is_dir():
        return
    ftm._write_display_sidecar(  # noqa: SLF001 — internal helper, intentional cross-module use
        target / "display.json",
        updates={"auto_train_pending": bool(pending)},
    )


def _append_log(project_name: str, *, backend: str, root: Path, message: str) -> None:
    """Append one timestamped line to the project's auto_train.log for debugging."""

    target = ftm.project_dir(project_name, backend=backend, root=root)
    if not target.is_dir():
        return
    log_path = target / "auto_train.log"
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now_iso()}] {message}\n")
    except OSError:
        # Best-effort logging; don't blow up the runner over a write failure.
        pass


def _prepare_and_launch(
    project_name: str,
    *,
    backend: str,
    root: Path,
    prepare_options: Mapping[str, object] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run prepare_project + launch_training with defaults baked into the constants.

    We deliberately call prepare_project first to regenerate manifests with the
    newly-completed sample included; if we skipped that step, the new label
    would never make it into the training set.
    """

    artifacts = ftm.prepare_project(
        project_name=project_name,
        backend=backend,
        root=root,
        **dict(prepare_options or {}),
    )
    run = ftm.launch_training(
        project_name=project_name,
        backend=backend,
        prefer_sbatch=True,
        python_bin=sys.executable,
        extra_env=dict(extra_env or {}),
        root=root,
    )
    return {
        "version_name": run.version_name,
        "job_id": run.job_id,
        "pid": run.pid,
        "samples": artifacts.sample_count,
    }


def _runner(
    project_name: str,
    *,
    backend: str,
    root: Path,
    prepare_options: Mapping[str, object] | None,
    extra_env: Mapping[str, str] | None,
) -> None:
    """Long-running thread body. Holds the project lock for the whole flow."""

    lock = _project_lock(backend, project_name)
    if not lock.acquire(blocking=False):
        # Another auto-train is already queued for this project — just mark
        # pending so the UI reflects that another label was added in the meantime.
        _set_pending(project_name, backend=backend, root=root, pending=True)
        _append_log(
            project_name,
            backend=backend,
            root=root,
            message="Skipped: another auto-train is already in flight; flagged pending.",
        )
        return
    try:
        _set_pending(project_name, backend=backend, root=root, pending=True)
        _append_log(
            project_name,
            backend=backend,
            root=root,
            message="Auto-train queued; waiting for any active runs to finish.",
        )

        deadline = time.monotonic() + _MAX_WAIT_SECONDS
        while _has_active_run(project_name, backend, root):
            if time.monotonic() > deadline:
                _append_log(
                    project_name,
                    backend=backend,
                    root=root,
                    message="Timed out waiting for active run to finish; giving up.",
                )
                _set_pending(project_name, backend=backend, root=root, pending=False)
                return
            time.sleep(_POLL_INTERVAL_SECONDS)

        try:
            outcome = _prepare_and_launch(
                project_name,
                backend=backend,
                root=root,
                prepare_options=prepare_options,
                extra_env=extra_env,
            )
        except Exception as exc:  # noqa: BLE001 — log everything in the runner thread
            _append_log(
                project_name,
                backend=backend,
                root=root,
                message=f"Auto-train failed: {exc}\n{traceback.format_exc()}",
            )
            _set_pending(project_name, backend=backend, root=root, pending=False)
            return

        _append_log(
            project_name,
            backend=backend,
            root=root,
            message=(
                f"Auto-train launched: version={outcome.get('version_name')} "
                f"job_id={outcome.get('job_id')} samples={outcome.get('samples')}"
            ),
        )
        _set_pending(project_name, backend=backend, root=root, pending=False)
    finally:
        lock.release()


def queue_auto_train(
    project_name: str,
    *,
    backend: str,
    root: Path,
    prepare_options: Mapping[str, object] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> bool:
    """Spawn a daemon thread to auto-train a project; returns False if already queued.

    Returning False is the "already queued, will pick up the new sample on the
    next pass" case — exactly what we want when multiple label completions land
    in quick succession on the same project. The currently-running thread will
    re-check ``_has_active_run`` and re-prepare with the latest sample set.
    """

    target = ftm.project_dir(project_name, backend=backend, root=root)
    if not target.is_dir():
        return False
    # If a runner is already holding the lock, set pending so the UI shows
    # "auto-train queued" and let the existing thread handle it.
    lock = _project_lock(backend, project_name)
    if lock.locked():
        _set_pending(project_name, backend=backend, root=root, pending=True)
        return False
    thread = threading.Thread(
        target=_runner,
        kwargs={
            "project_name": project_name,
            "backend": backend,
            "root": root,
            "prepare_options": prepare_options,
            "extra_env": extra_env,
        },
        name=f"auto-train-{backend}-{project_name}",
        daemon=True,
    )
    thread.start()
    return True
