"""Slurm submission helper used by site-launched workflows.

Extracted verbatim from ``workflow_dashboard.py``. Behavior is unchanged: the
same metadata files, stdout/stderr captures, and run-dir layout are produced
for every site run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from workflow_background import utc_now_iso


def submit_sbatch_job(
    *,
    sbatch_script: Path,
    cwd: Path,
    run_dir: Path,
    export_env: dict[str, str],
    metadata: dict[str, object] | None = None,
    job_label: str = "site workflow",
) -> dict[str, object]:
    """Submit a Slurm job and record the same metadata files used by site runs."""

    if not shutil.which("sbatch"):
        raise RuntimeError("sbatch is not available on this machine.")
    if not sbatch_script.is_file():
        raise RuntimeError(f"Slurm script not found: {sbatch_script}")

    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    exit_code_path = run_dir / "exit_code.txt"
    metadata_path = run_dir / "metadata.json"
    command = ["sbatch", "--parsable", str(sbatch_script)]
    run_metadata = {
        "runner": "slurm",
        "submission_status": "submitting",
        "command": command,
        "cwd": str(cwd),
        "sbatch_script": str(sbatch_script),
        "started_at_utc": utc_now_iso(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "exit_code_path": str(exit_code_path),
        **(metadata or {}),
    }
    metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")

    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **export_env},
    )
    if completed.returncode != 0:
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "Slurm submission failed.\n", encoding="utf-8")
        exit_code_path.write_text(str(completed.returncode), encoding="utf-8")
        run_metadata.update(
            {
                "submission_status": "failed",
                "sbatch_stdout": completed.stdout,
                "sbatch_stderr": completed.stderr,
            }
        )
        metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
        error_text = (completed.stderr or completed.stdout or "Slurm submission failed.").strip()
        raise RuntimeError(error_text)

    output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    raw_job_id = output_lines[-1] if output_lines else ""
    job_id = raw_job_id.split(";", 1)[0]
    stdout_path.write_text(
        f"Submitted Slurm job {job_id} for {job_label}.\n"
        f"Slurm script: {sbatch_script}\n",
        encoding="utf-8",
    )
    if completed.stderr:
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    run_metadata.update(
        {
            "submission_status": "submitted",
            "slurm_job_id": job_id,
            "sbatch_stdout": completed.stdout,
            "sbatch_stderr": completed.stderr,
        }
    )
    metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "slurm_job_id": job_id,
        "run_dir": run_dir,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "exit_code_path": exit_code_path,
        "metadata_path": metadata_path,
    }
