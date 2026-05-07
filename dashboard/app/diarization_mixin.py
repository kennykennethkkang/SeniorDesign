#!/usr/bin/env python3
"""Diarization model selection, run state, and submission handlers.

Wraps everything between "user picked some files and a model" and "sbatch
job is in flight": validating selections, resolving the chosen backend
(NeMo / pyannote / fine-tuned), composing the run directory, and handing
off to ``scheduler/run_site_diarization.sbatch``. We never call the
diarization pipeline directly here — that lives in ``run_diarization.py``
and we treat it as a black box on purpose so the dashboard side can change
without touching ML code.
"""
from __future__ import annotations


import argparse
import csv
import errno
import hashlib
import html
import heapq
import io
import importlib.util
import json
import mimetypes
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="'cgi' is deprecated and slated for removal in Python 3.13",
        category=DeprecationWarning,
    )
    import cgi

from fine_tuning_manager import (
    DEFAULT_BASE_SHIFT,
    DEFAULT_BASE_WINDOW,
    DEFAULT_CONFIG_NAME,
    DEFAULT_FINE_TUNING_BACKEND,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PYANNOTE_DURATION,
    DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK,
    DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME,
    DEFAULT_PYANNOTE_PRETRAINED_MODEL,
    DEFAULT_STEP_COUNT,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_DEVICES,
    DEFAULT_SPEAKER_MODEL,
    DEFAULT_SLURM_CPUS,
    DEFAULT_SLURM_GPUS,
    DEFAULT_SLURM_MEMORY,
    DEFAULT_SLURM_PARTITION,
    DEFAULT_SLURM_TIME,
    launch_training,
    build_sample,
    list_projects,
    normalize_backend,
    parse_rttm,
    probe_media_duration,
    prepare_project,
    run_status as fine_tuning_run_status,
    sanitize_filename,
    save_project_sample_streams,
    slugify,
)
from workflow_background import launch_background_command, run_status, slurm_queue_snapshot, utc_now_iso
from audio_numbering import AUDIO_EXTENSIONS, NUMBERED_PREFIX, normalize_audio_dir
from review_bundle import REVIEW_BUNDLE_FORMAT_VERSION, parse_srt, write_review_bundle
from workflow_cli import (
    DIARIZATION_RUNS_ROOT,
    OUTPUTS_ROOT,
    PROJECT_ROOT,
    YOUTUBE_HISTORY_INDEX,
    YOUTUBE_RUNS_ROOT,
    count_queued_urls,
    iter_audio_files,
    iter_diarization_run_directories,
    latest_diarization_directory,
    latest_directory,
)
from workflow_preferences import (
    DIARIZATION_BACKEND_LABELS,
    DIARIZATION_BACKENDS,
    load_preferences,
    normalize_diarization_backend,
    save_preferences,
)

from dashboard.constants import (
    ARTIFACT_PREVIEW_BYTE_LIMIT,
    ARTIFACT_PREVIEW_LINE_LIMIT,
    ARTIFACT_SCAN_EXCLUDE_DIRS,
    AUDIO_INVENTORY_PAGES,
    CODE_ROOT,
    DASHBOARD_REFRESH_STATUSES,
    DEFAULT_AUDIO_FOLDERS,
    DEFAULT_SERVER_MODE,
    DEFAULT_SERVER_THREADS,
    DEFAULT_TRAINING_LABEL_PROJECT,
    DEFAULT_UPLOAD_AUDIO_FOLDER,
    DEFAULT_YOUTUBE_AUDIO_FOLDER,
    DIARIZATION_ACTIVE_STATUSES,
    DIARIZATION_ARTIFACT_SUFFIXES,
    DIARIZATION_COMPLETED_STATUSES,
    DIARIZATION_LABEL_PREVIEW_LIMIT,
    DOWNLOADABLE_ROOT_NAMES,
    FRONTEND_TEMPLATE,
    IDLE_TRACKING_INTERVAL_MS,
    LIVE_TRACKING_INTERVAL_MS,
    MODEL_SELECTION_PAGES,
    NAV_PATHS,
    PAGE_ALIASES,
    PAGE_PATHS,
    PROJECT_SUMMARY_PAGES,
    RECENT_OUTPUT_PAGES,
    RECENT_SRT_PAGES,
    ROOT_AUDIO_FOLDER_VALUE,
    SECURITY_RESPONSE_HEADERS,
    TAIL_PREVIEW_SUFFIXES,
    TEXT_PREVIEW_SUFFIXES,
    TRAINING_LABEL_CONTEXT_PAGES,
    TRAINING_RTTM_SUFFIXES,
    TRAINING_SOURCE_SCAN_EXCLUDE_DIRS,
    TRAINING_TRANSCRIPT_SUFFIXES,
    UPLOAD_AUDIO_ACCEPT,
    UPLOAD_AUDIO_SUFFIXES,
    WORKSPACE_MEDIA_SUFFIXES,
    YOUTUBE_INDEX_COLUMNS,
    YOUTUBE_NO_DATA_MARKERS,
    YOUTUBE_QUEUE_PAGES,
)
from dashboard.slurm import submit_sbatch_job
from dashboard.servers import (
    ThreadedWSGIServer,
    bind_server,
    built_in_make_server,
    server_port,
    waitress_make_server,
    write_url_file,
)
from dashboard.cli import build_parser


class DiarizationMixin:
    """Diarization model selection, run state, and handlers."""

    def base_diarization_model_options(self) -> list[dict[str, str]]:
        """Return the built-in diarization choices that always exist."""

        options: list[dict[str, str]] = []
        for backend in DIARIZATION_BACKENDS:
            label = self.diarization_backend_label(backend)
            options.append(
                {
                    "key": backend,
                    "backend": backend,
                    "label": f"{label} (default)",
                    "short_label": label,
                    "kind": "default",
                    "description": f"Use the shared default {label} diarization settings.",
                    "nemo_msdd_model_path": "",
                    "pyannote_segmentation_model": "",
                }
            )
        return options

    def newest_model_artifact(
        self,
        root: Path,
        *,
        suffixes: set[str],
    ) -> Path | None:
        """Return the newest checkpoint-like file below one project experiments folder."""

        if not root.is_dir():
            return None
        newest_path: Path | None = None
        newest_mtime = -1.0
        for current_dir, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name != "__pycache__"]
            current_path = Path(current_dir)
            for filename in filenames:
                candidate = current_path / filename
                if candidate.suffix.lower() not in suffixes:
                    continue
                try:
                    modified = candidate.stat().st_mtime
                except OSError:
                    continue
                if modified >= newest_mtime:
                    newest_path = candidate
                    newest_mtime = modified
        return newest_path

    def newest_pyannote_artifact(self, experiments_dir: Path) -> Path | None:
        """Return the newest reusable pyannote artifact from a fine-tuning project."""

        if not experiments_dir.is_dir():
            return None

        newest_path: Path | None = None
        newest_mtime = -1.0
        for current_dir, dirnames, filenames in os.walk(experiments_dir):
            dirnames[:] = [name for name in dirnames if name != "__pycache__"]
            current_path = Path(current_dir)
            if "config.json" in filenames and "model.safetensors" in filenames:
                config_path = current_path / "config.json"
                weights_path = current_path / "model.safetensors"
                try:
                    modified = max(config_path.stat().st_mtime, weights_path.stat().st_mtime)
                except OSError:
                    modified = -1.0
                if modified >= newest_mtime:
                    newest_path = current_path
                    newest_mtime = modified
            for filename in filenames:
                candidate = current_path / filename
                if candidate.suffix.lower() != ".ckpt":
                    continue
                try:
                    modified = candidate.stat().st_mtime
                except OSError:
                    continue
                if modified >= newest_mtime:
                    newest_path = candidate
                    newest_mtime = modified
        return newest_path

    def fine_tuned_diarization_model_options(self) -> list[dict[str, str]]:
        """Expose reusable fine-tuning outputs as optional diarization model choices."""

        def builder() -> list[dict[str, str]]:
            options: list[dict[str, str]] = []
            for project in self.project_summaries():
                backend_raw = str(project.get("backend", "") or "").strip().lower()
                if backend_raw not in DIARIZATION_BACKENDS:
                    continue
                project_path = Path(str(project.get("path", ""))).expanduser()
                experiments_dir = project_path / "artifacts" / "experiments"
                slug = str(project.get("slug", "") or project_path.name).strip() or "project"
                label = self.diarization_backend_label(backend_raw)

                def append_option(*, key_suffix: str, display_name: str, description: str, artifact_root: Path) -> bool:
                    if backend_raw == "nemo":
                        model_artifact = self.newest_model_artifact(
                            artifact_root,
                            suffixes={".ckpt", ".nemo"},
                        )
                    else:
                        model_artifact = self.newest_pyannote_artifact(artifact_root)
                    if not model_artifact:
                        return False
                    option: dict[str, str] = {
                        "key": f"{backend_raw}:{slug}:{key_suffix}" if key_suffix else f"{backend_raw}:{slug}",
                        "backend": backend_raw,
                        "label": f"{label} fine-tuned / {slug} / {display_name}" if display_name else f"{label} fine-tuned / {slug}",
                        "short_label": display_name or slug,
                        "kind": "fine_tuned",
                        "description": description,
                        "nemo_msdd_model_path": "",
                        "pyannote_segmentation_model": "",
                    }
                    if backend_raw == "nemo":
                        option["nemo_msdd_model_path"] = str(model_artifact)
                    else:
                        option["pyannote_segmentation_model"] = str(model_artifact)
                    options.append(option)
                    return True

                version_options_added = False
                for run in project.get("recent_runs") or []:
                    if not isinstance(run, dict):
                        continue
                    experiment_dir_value = str(run.get("experiment_dir") or "").strip()
                    if not experiment_dir_value:
                        continue
                    experiment_dir = Path(experiment_dir_value)
                    if not experiment_dir.is_dir():
                        continue
                    version_name = str(run.get("version_name") or experiment_dir.name)
                    # Prefer the user's display rename; fall back to version_name
                    # so legacy runs that haven't been renamed still get a label.
                    display_label = str(run.get("display_name") or "").strip() or version_name
                    version_slug = str(run.get("version_slug") or slugify(version_name))
                    version_options_added = append_option(
                        key_suffix=version_slug,
                        display_name=display_label,
                        description=(
                            f"Fine-tuned model {display_label} (v: {version_name})."
                            if display_label != version_name
                            else f"Reusable fine-tuning artifact from version {version_name}."
                        ),
                        artifact_root=experiment_dir,
                    ) or version_options_added
                if version_options_added:
                    continue
                append_option(
                    key_suffix="",
                    display_name="",
                    description=f"Latest reusable fine-tuning artifact from project {slug}.",
                    artifact_root=experiments_dir,
                )
            options.sort(key=lambda item: (item["backend"], item["label"].lower()))
            return options

        return self.cached_value(
            "fine_tuned_diarization_model_options",
            ttl_seconds=3.0,
            builder=builder,
        )

    def diarization_model_options(self) -> list[dict[str, str]]:
        """Return every selectable diarization model profile."""

        return self.cached_value(
            "diarization_model_options",
            ttl_seconds=3.0,
            builder=lambda: [
                *self.base_diarization_model_options(),
                *self.fine_tuned_diarization_model_options(),
            ],
        )

    def diarization_model_option_lookup(self) -> dict[str, dict[str, str]]:
        """Index selectable diarization model profiles by their stable key."""

        return {
            str(option.get("key", "")): option
            for option in self.diarization_model_options()
            if str(option.get("key", "")).strip()
        }

    def resolve_diarization_model_option(
        self,
        *,
        model_key: str | None,
        fallback_backend: str | None,
    ) -> dict[str, str]:
        """Resolve a selected model key, falling back to the built-in backend choice."""

        lookup = self.diarization_model_option_lookup()
        requested_key = (model_key or "").strip()
        if requested_key and requested_key in lookup:
            return lookup[requested_key]
        backend = normalize_diarization_backend(fallback_backend or str(DEFAULT_FINE_TUNING_BACKEND))
        return lookup.get(backend, self.base_diarization_model_options()[0])

    def active_run_directories(self, root: Path) -> list[Path]:
        """Return direct child run directories that still need dashboard polling."""

        if not root.is_dir():
            return []
        active: list[Path] = []
        for path in root.iterdir():
            if path.is_dir() and run_status(path) in DASHBOARD_REFRESH_STATUSES:
                active.append(path)
        return active

    def active_diarization_run_directories(self) -> list[Path]:
        """Return active diarization runs across all backend folders."""

        return [
            path
            for path in iter_diarization_run_directories(self.diarization_runs_root)
            if run_status(path) in DASHBOARD_REFRESH_STATUSES
        ]

    def active_diarization_runs_detailed(self, *, limit: int = 4) -> list[dict[str, object]]:
        """Return ``diarization_run_details`` payloads for currently active diarization runs.

        Capped at ``limit`` to bound polling work when many runs are queued.
        Used by the frontend's tab-toggle for switching between simultaneous runs.
        """

        active_paths = self.active_diarization_run_directories()
        # Newest first so the most recently submitted run is the default tab.
        active_paths.sort(key=lambda path: path.name, reverse=True)
        details = []
        for run_dir in active_paths[:limit]:
            payload = self.diarization_run_details(run_dir)
            if payload:
                details.append(payload)
        return details

    def active_fine_tuning_runs(self) -> list[dict[str, str]]:
        """Return active fine-tuning runs without loading full project summaries."""

        active: list[dict[str, str]] = []
        projects_root = self.root / "fine_tuning" / "projects"
        if not projects_root.is_dir():
            return active

        project_dirs: list[tuple[str, Path]] = []
        for candidate in sorted(projects_root.iterdir(), key=lambda path: path.name.lower()):
            if not candidate.is_dir():
                continue
            if (candidate / "runs").is_dir():
                project_dirs.append((DEFAULT_FINE_TUNING_BACKEND, candidate))
                continue
            for nested in sorted(candidate.iterdir(), key=lambda path: path.name.lower()):
                if nested.is_dir() and (nested / "runs").is_dir():
                    project_dirs.append((candidate.name, nested))

        for backend, project_dir in project_dirs:
            runs_dir = project_dir / "runs"
            for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                if not run_dir.is_dir() or not (run_dir / "metadata.json").is_file():
                    continue
                status = fine_tuning_run_status(run_dir)
                if status not in DASHBOARD_REFRESH_STATUSES:
                    continue
                active.append(
                    {
                        "backend": backend,
                        "project": project_dir.name,
                        "run": run_dir.name,
                        "status": status,
                    }
                )
        return sorted(active, key=lambda item: (item["backend"], item["project"], item["run"]))

    def handle_tests(self, environ):
        """Run the local unit test suite from the dashboard."""

        command = [sys.executable, "workflow_cli.py", "test"]
        status, message = self.command_result_message(command)
        return self.redirect(environ, "/diarization", message=message, status=status)

    def handle_diarization_run(self, environ):
        """Submit one or more Slurm diarization runs for selected or all audio files.

        Multiple model keys may be selected; each one fans out to its own independent
        run that shares a common ``batch_id`` so sibling runs can be grouped later.
        """

        form = self.parse_form(environ)
        preferences = self.model_preferences()
        selected_audio = self.ordered_unique(form.getlist("selected_audio"))
        mode = (form.getfirst("diarization_mode") or "").strip().lower()
        run_all = mode == "all" or bool(form.getfirst("diarize_all_audio"))
        base_audio_files = self.selected_audio_names(selected_audio, run_all=run_all)

        requested_keys = self.ordered_unique(form.getlist("diarization_model_keys"))
        if not requested_keys:
            singular = (
                form.getfirst("diarization_model_key")
                or form.getfirst("diarization_backend")
                or str(preferences.get("default_diarization_model_key") or "")
            )
            if singular:
                requested_keys = [singular]
        fallback_backend = form.getfirst("diarization_backend") or str(preferences["default_backend"])

        model_options: list[dict[str, str]] = []
        seen_option_keys: set[str] = set()
        for key in requested_keys:
            option = self.resolve_diarization_model_option(
                model_key=key,
                fallback_backend=fallback_backend,
            )
            option_key = str(option.get("key") or "")
            if option_key in seen_option_keys:
                continue
            seen_option_keys.add(option_key)
            model_options.append(option)
        if not model_options:
            model_options = [
                self.resolve_diarization_model_option(
                    model_key=None,
                    fallback_backend=fallback_backend,
                )
            ]

        primary_model_key = str(model_options[0].get("key") or "")
        diarization_location = self.diarization_page_location(primary_model_key)
        include_completed = bool(form.getfirst("diarization_include_completed"))

        if not base_audio_files:
            return self.redirect(
                environ,
                diarization_location,
                message="Select at least one audio file or choose the run-all option.",
                status="error",
            )

        whisper_model = form.getfirst("diarization_whisper_model") or str(preferences["whisper_model"])
        batch_size = self.parse_int(
            form.getfirst("diarization_batch_size"),
            int(preferences["whisper_batch_size"]),
            "diarization_batch_size",
        )
        language = (form.getfirst("diarization_language") or "").strip()
        pyannote_hf_token = (
            form.getfirst("pyannote_hf_token")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or self.dashboard_secret("HF_TOKEN")
            or self.dashboard_secret("HUGGINGFACE_TOKEN")
            or self.dashboard_secret("HUGGINGFACE_HUB_TOKEN")
            or ""
        ).strip()
        needs_pyannote_token = any(option["backend"] == "pyannote" for option in model_options)
        if needs_pyannote_token and not pyannote_hf_token:
            return self.redirect(
                environ,
                diarization_location,
                message=self.notification_message(
                    "Pyannote needs a Hugging Face token before it can run.",
                    "Paste a token in Advanced Runtime Setting or start the dashboard with HF_TOKEN set.",
                    "The token must have access to the selected pyannote pipeline.",
                ),
                status="error",
            )

        history_lookup = (
            self.diarization_model_history_lookup() if not include_completed else {}
        )
        batch_id = time.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        all_model_keys = [str(option.get("key") or "") for option in model_options]

        submitted_runs: list[dict[str, object]] = []
        skipped_models: list[dict[str, str]] = []
        failed_models: list[dict[str, str]] = []

        for index, model_option in enumerate(model_options):
            backend = model_option["backend"]
            model_key = str(model_option.get("key") or "")
            model_label = str(model_option.get("label") or model_key or backend)

            if include_completed:
                audio_files = list(base_audio_files)
                skipped_completed: list[str] = []
            else:
                audio_files = []
                skipped_completed = []
                for audio_name in base_audio_files:
                    record = history_lookup.get(audio_name, {}).get(model_key)
                    if self.diarization_record_is_completed(record):
                        skipped_completed.append(audio_name)
                    else:
                        audio_files.append(audio_name)
            if not audio_files:
                skipped_models.append(
                    {"model_key": model_key, "model_label": model_label, "reason": "all_completed"}
                )
                continue

            run_name = self.run_directory_name(
                "diarization",
                backend=backend,
                count=len(audio_files),
            )
            if len(model_options) > 1:
                run_name = f"{run_name}_{secrets.token_hex(2)}"
            run_dir = self.diarization_backend_runs_root(backend) / run_name
            logs_dir = run_dir / "logs"
            run_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            selection_path = run_dir / "selected_audio.txt"
            selection_path.write_text("\n".join(audio_files) + "\n", encoding="utf-8")

            export_env = {
                "SITE_ROOT_DIR": str(self.root),
                "SITE_DIARIZATION_RUN_DIR": str(run_dir),
                "SITE_DIARIZATION_AUDIO_LIST": str(selection_path),
                "SITE_DIARIZATION_LOG_DIR": str(logs_dir),
                "SITE_AUDIO_DIR": str(self.audio_dir),
                "DIARIZATION_BACKEND": backend,
                "WHISPER_MODEL": whisper_model,
                "WHISPER_BATCH_SIZE": str(batch_size),
                "DIARIZATION_DEVICE": "cuda",
                "DIARIZATION_LANGUAGE": language,
                "DIARIZATION_SOURCE_SEPARATION": "0",
                "DIARIZATION_SKIP_REVIEW": "0",
                "PYANNOTE_PIPELINE_MODEL": str(preferences["pyannote_pipeline_model"]),
                "PYANNOTE_SEGMENTATION_MODEL": model_option.get("pyannote_segmentation_model") or str(preferences["pyannote_segmentation_model"]),
            }
            if model_option.get("nemo_msdd_model_path"):
                export_env["NEMO_MSDD_MODEL"] = model_option["nemo_msdd_model_path"]
            if pyannote_hf_token:
                export_env["HF_TOKEN"] = pyannote_hf_token
                export_env["HUGGINGFACE_HUB_TOKEN"] = pyannote_hf_token

            metadata = {
                "operation": "diarization",
                "backend": backend,
                "model_key": model_key,
                "model_label": model_label,
                "mode": "all" if run_all else "selected",
                "selected_audio_count": len(audio_files),
                "selected_audio_path": str(selection_path),
                "skipped_completed_count": len(skipped_completed),
                "include_completed": include_completed,
                "whisper_model": whisper_model,
                "whisper_batch_size": batch_size,
                "language": language,
                "nemo_msdd_model_path": model_option.get("nemo_msdd_model_path") or "",
                "pyannote_pipeline_model": str(preferences["pyannote_pipeline_model"]),
                "pyannote_segmentation_model": model_option.get("pyannote_segmentation_model") or str(preferences["pyannote_segmentation_model"]),
                "output_dir": str(run_dir),
                "log_dir": str(logs_dir),
                "review_outputs": "enabled",
                "batch_id": batch_id,
                "batch_index": index,
                "batch_size_total": len(model_options),
                "batch_model_keys": all_model_keys,
            }
            try:
                import workflow_dashboard as _wd  # lazy: honor monkey-patches in tests
                submission = _wd.submit_sbatch_job(
                    sbatch_script=self.site_diarization_sbatch,
                    cwd=self.root,
                    run_dir=run_dir,
                    export_env=export_env,
                    metadata=metadata,
                    job_label="site diarization",
                )
            except RuntimeError as exc:
                failed_models.append(
                    {"model_key": model_key, "model_label": model_label, "error": str(exc)}
                )
                continue
            submitted_runs.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "backend": backend,
                    "backend_label": self.diarization_backend_label(backend),
                    "run_name": run_dir.name,
                    "run_dir": run_dir,
                    "slurm_job_id": submission.get("slurm_job_id") or "unknown",
                    "selected_count": len(audio_files),
                    "skipped_completed_count": len(skipped_completed),
                }
            )

        self.invalidate_dashboard_cache()

        if not submitted_runs:
            if skipped_models and not failed_models:
                if len(skipped_models) == 1:
                    skipped_label = skipped_models[0]["model_label"]
                    return self.redirect(
                        environ,
                        diarization_location,
                        message=self.notification_message(
                            "All selected audio files have already been diarized.",
                            f"The selected files already have completed {skipped_label} results.",
                            "Use the 'Allow re-running already diarized files' checkbox if you intentionally want to run them again.",
                        ),
                        status="error",
                    )
                summary = ", ".join(item["model_label"] for item in skipped_models)
                return self.redirect(
                    environ,
                    diarization_location,
                    message=self.notification_message(
                        "All selected audio files have already been diarized.",
                        f"Skipped {len(skipped_models)} model(s) because their audio is already diarized: {summary}.",
                        "Use the 'Allow re-running already diarized files' checkbox if you intentionally want to run them again.",
                    ),
                    status="error",
                )
            error_lines: list[str] = []
            if skipped_models:
                summary = ", ".join(item["model_label"] for item in skipped_models)
                error_lines.append(
                    f"Skipped {len(skipped_models)} model(s) because their audio is already diarized: {summary}."
                )
            if failed_models:
                for item in failed_models:
                    error_lines.append(f"{item['model_label']}: {item['error']}")
                error_lines.append("Check that this site is running on WAVE with sbatch available.")
            if not error_lines:
                error_lines.append("Select at least one audio file or choose the run-all option.")
            return self.redirect(
                environ,
                diarization_location,
                message=self.notification_message(
                    "No diarization jobs were submitted.",
                    *error_lines,
                ),
                status="error",
            )

        if len(submitted_runs) == 1 and not skipped_models and not failed_models:
            run = submitted_runs[0]
            details = []
            if run["skipped_completed_count"]:
                details.append(
                    f"Skipped {run['skipped_completed_count']} already diarized file(s)."
                )
            return self.redirect(
                environ,
                diarization_location,
                message=self.notification_message(
                    f"Submitted Slurm diarization job {run['slurm_job_id']} for run '{run['run_name']}' with {run['selected_count']} file(s) and the {run['backend_label']} backend.",
                    f"Selected model profile: {run['model_label']}.",
                    *details,
                ),
                status="success",
            )

        summary_line = (
            f"Submitted {len(submitted_runs)} diarization job(s) (batch {batch_id})."
        )
        details: list[str] = []
        for run in submitted_runs:
            details.append(
                f"{run['model_label']}: job {run['slurm_job_id']} ({run['selected_count']} file(s))."
            )
        if skipped_models:
            for item in skipped_models:
                details.append(
                    f"Skipped {item['model_label']} because all selected audio is already diarized."
                )
        if failed_models:
            for item in failed_models:
                details.append(f"Failed to submit {item['model_label']}: {item['error']}")
        notice_status = "success" if not failed_models else "info"
        return self.redirect(
            environ,
            diarization_location,
            message=self.notification_message(summary_line, *details),
            status=notice_status,
        )

    def handle_model_preferences(self, environ):
        """Persist the dashboard defaults that drive later workflow pages."""

        form = self.parse_form(environ)
        existing = self.model_preferences()
        selected_model_option = self.resolve_diarization_model_option(
            model_key=form.getfirst("default_diarization_model_key") or str(existing.get("default_diarization_model_key") or ""),
            fallback_backend=form.getfirst("default_backend") or str(existing["default_backend"]),
        )
        pyannote_defaults = dict(existing.get("pyannote_fine_tuning", {}))
        nemo_defaults = dict(existing.get("nemo_fine_tuning", {}))
        pyannote_segmentation_model = form.getfirst("pyannote_segmentation_model")
        if pyannote_segmentation_model is None:
            pyannote_segmentation_model = str(existing["pyannote_segmentation_model"])
        else:
            pyannote_segmentation_model = pyannote_segmentation_model.strip()
        updated = {
            **existing,
            "default_backend": selected_model_option["backend"],
            "default_diarization_model_key": selected_model_option["key"],
            "whisper_model": form.getfirst("whisper_model") or existing["whisper_model"],
            "whisper_batch_size": self.parse_int(
                form.getfirst("whisper_batch_size"),
                int(existing["whisper_batch_size"]),
                "whisper_batch_size",
            ),
            "pyannote_pipeline_model": form.getfirst("pyannote_pipeline_model") or existing["pyannote_pipeline_model"],
            "pyannote_segmentation_model": pyannote_segmentation_model,
            "fine_tuning_backend": form.getfirst("fine_tuning_backend") or existing["fine_tuning_backend"],
            "pyannote_fine_tuning": {
                **pyannote_defaults,
                "pretrained_model": form.getfirst("pyannote_pretrained_model") or pyannote_defaults.get("pretrained_model", DEFAULT_PYANNOTE_PRETRAINED_MODEL),
                "duration": self.parse_float(form.getfirst("pyannote_duration"), float(pyannote_defaults.get("duration", DEFAULT_PYANNOTE_DURATION)), "pyannote_duration"),
                "max_speakers_per_chunk": self.parse_int(form.getfirst("pyannote_max_speakers_per_chunk"), int(pyannote_defaults.get("max_speakers_per_chunk", DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK)), "pyannote_max_speakers_per_chunk"),
                "max_speakers_per_frame": self.parse_int(form.getfirst("pyannote_max_speakers_per_frame"), int(pyannote_defaults.get("max_speakers_per_frame", DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME)), "pyannote_max_speakers_per_frame"),
                "max_epochs": self.parse_int(form.getfirst("pyannote_max_epochs"), int(pyannote_defaults.get("max_epochs", DEFAULT_MAX_EPOCHS)), "pyannote_max_epochs"),
                "devices": self.parse_int(form.getfirst("pyannote_devices"), int(pyannote_defaults.get("devices", DEFAULT_DEVICES)), "pyannote_devices"),
            },
            "nemo_fine_tuning": {
                **nemo_defaults,
                "config_name": form.getfirst("nemo_config_name") or nemo_defaults.get("config_name", DEFAULT_CONFIG_NAME),
                "speaker_model": form.getfirst("nemo_speaker_model") or nemo_defaults.get("speaker_model", DEFAULT_SPEAKER_MODEL),
            },
        }
        save_preferences(updated, root=self.root)
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/diarization",
            message="Saved model settings for diarization and fine-tuning.",
            status="success",
        )

    def model_preferences(self) -> dict[str, object]:
        """Load the persisted workflow defaults used by multiple dashboard pages."""

        return self.cached_value(
            "model_preferences",
            ttl_seconds=3.0,
            builder=lambda: load_preferences(root=self.root),
        )

    def normalize_diarization_status(self, status: str, error_summary: str = "") -> str:
        """Normalize per-file diarization status values from current and older runs."""

        normalized = (status or "").strip().lower().replace(" ", "_")
        if normalized == "failed" and "Whisper returned an empty transcript" in error_summary:
            return "no_speech"
        return normalized or "unknown"

    def diarization_backend_label(self, backend: str) -> str:
        """Return the readable label for a diarization backend."""

        return DIARIZATION_BACKEND_LABELS.get(backend, backend)

    def diarization_run_backend(self, run_dir: Path, metadata: dict[str, object]) -> str:
        """Infer the backend for new nested runs and older top-level run folders."""

        raw_backend = str(metadata.get("backend") or "").strip()
        if raw_backend:
            try:
                return normalize_diarization_backend(raw_backend)
            except ValueError:
                return raw_backend.lower()
        if run_dir.parent.parent == self.diarization_runs_root:
            try:
                return normalize_diarization_backend(run_dir.parent.name)
            except ValueError:
                pass
        for backend in DIARIZATION_BACKENDS:
            if f"_diarization_{backend}_" in run_dir.name or run_dir.name.endswith(f"_diarization_{backend}"):
                return backend
        return "unknown"

    def diarization_record_is_completed(self, record: object) -> bool:
        """Return whether a history record should be skipped by default."""

        if not isinstance(record, dict):
            return False
        return self.normalize_diarization_status(str(record.get("status", ""))) in DIARIZATION_COMPLETED_STATUSES

    def diarization_record_priority(self, record: object) -> int:
        """Rank history records so stored completed work survives newer failed retries."""

        if not isinstance(record, dict):
            return -1
        status = self.normalize_diarization_status(
            str(record.get("status", "")),
            str(record.get("error_summary", "")),
        )
        if status in DIARIZATION_ACTIVE_STATUSES:
            return 3
        if status in DIARIZATION_COMPLETED_STATUSES:
            return 2
        return 1

    def store_diarization_record(
        self,
        lookup: dict[str, dict[str, dict[str, object]]],
        *,
        audio_basename: str,
        backend: str,
        record: dict[str, object],
    ) -> None:
        """Keep the best record for one audio/backend pair without losing stored outputs."""

        existing = lookup.get(audio_basename, {}).get(backend)
        if existing is not None and self.diarization_record_priority(existing) >= self.diarization_record_priority(record):
            return
        lookup.setdefault(audio_basename, {})[backend] = record

    def diarization_run_model_key(self, backend: str, metadata: dict[str, object]) -> str:
        """Resolve the stable model-profile key for one historical diarization run."""

        key = str(metadata.get("model_key") or "").strip()
        return key or backend

    def diarization_run_model_label(
        self,
        *,
        backend: str,
        model_key: str,
        metadata: dict[str, object],
    ) -> str:
        """Resolve the user-facing label for one historical diarization model choice."""

        label = str(metadata.get("model_label") or "").strip()
        if label:
            return label
        option = self.diarization_model_option_lookup().get(model_key)
        if option is not None:
            return option.get("label") or self.diarization_backend_label(backend)
        return self.diarization_backend_label(backend)

    def diarization_model_history_lookup(self) -> dict[str, dict[str, dict[str, object]]]:
        """Map each audio filename to its newest result for each diarization model profile."""

        def builder() -> dict[str, dict[str, dict[str, object]]]:
            run_dirs = iter_diarization_run_directories(self.diarization_runs_root)
            run_dirs.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)

            lookup: dict[str, dict[str, dict[str, object]]] = {}
            for run_dir in run_dirs:
                metadata_path = run_dir / "metadata.json"
                metadata: dict[str, object] = {}
                if metadata_path.is_file():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        metadata = {}

                backend = self.diarization_run_backend(run_dir, metadata)
                backend_label = self.diarization_backend_label(backend)
                model_key = self.diarization_run_model_key(backend, metadata)
                model_label = self.diarization_run_model_label(
                    backend=backend,
                    model_key=model_key,
                    metadata=metadata,
                )
                summary_path = run_dir / "logs" / "runtime_summary.tsv"
                summary_rows = self.read_tsv_rows(summary_path)
                selected_audio = self.read_line_file(run_dir / "selected_audio.txt")
                run_state = run_status(run_dir)
                modified = run_dir.stat().st_mtime
                last_run = time.strftime("%Y-%m-%d %H:%M", time.localtime(modified))

                summarized_names: set[str] = set()
                for row in summary_rows:
                    audio_key = self.clean_audio_selection_value(row.get("audio_file", ""))
                    if not audio_key:
                        continue
                    error_summary = row.get("error_summary", "")
                    status = self.normalize_diarization_status(row.get("status", ""), error_summary)
                    stem = self.diarization_output_base(audio_key)
                    summarized_names.add(audio_key)
                    self.store_diarization_record(
                        lookup,
                        audio_basename=audio_key,
                        backend=model_key,
                        record={
                        "audio_file": audio_key,
                        "status": status,
                        "runtime_seconds": row.get("runtime_seconds", ""),
                        "error_summary": error_summary,
                        "run_name": run_dir.name,
                        "run_dir": run_dir,
                        "run_status": run_state,
                        "backend": backend,
                        "backend_label": backend_label,
                        "model_key": model_key,
                        "model_label": model_label,
                        "last_run": last_run,
                        "sort_key": modified,
                        "transcript_path": run_dir / f"{stem}.txt",
                        "srt_path": run_dir / f"{stem}.srt",
                        "review_path": run_dir / f"{stem}_review.html",
                        "flags_path": run_dir / f"{stem}_review_flags.tsv",
                        "summary_path": summary_path,
                        "batch_id": str(metadata.get("batch_id") or ""),
                        "batch_index": metadata.get("batch_index"),
                        "batch_size_total": metadata.get("batch_size_total"),
                        },
                    )

                if run_state not in DIARIZATION_ACTIVE_STATUSES:
                    continue
                for audio_name in selected_audio:
                    audio_basename = self.clean_audio_selection_value(audio_name)
                    if (
                        not audio_basename
                        or audio_basename in summarized_names
                    ):
                        continue
                    self.store_diarization_record(
                        lookup,
                        audio_basename=audio_basename,
                        backend=model_key,
                        record={
                        "audio_file": audio_basename,
                        "status": run_state,
                        "runtime_seconds": "",
                        "error_summary": "",
                        "run_name": run_dir.name,
                        "run_dir": run_dir,
                        "run_status": run_state,
                        "backend": backend,
                        "backend_label": backend_label,
                        "model_key": model_key,
                        "model_label": model_label,
                        "last_run": last_run,
                        "sort_key": modified,
                        "transcript_path": None,
                        "srt_path": None,
                        "review_path": None,
                        "flags_path": None,
                        "summary_path": summary_path if summary_path.is_file() else None,
                        "batch_id": str(metadata.get("batch_id") or ""),
                        "batch_index": metadata.get("batch_index"),
                        "batch_size_total": metadata.get("batch_size_total"),
                        },
                    )
            return lookup

        return self.cached_value("diarization_model_history_lookup", ttl_seconds=3.0, builder=builder)

    def diarization_history_lookup(self) -> dict[str, dict[str, object]]:
        """Map each audio filename to its newest diarization result across all model profiles."""

        lookup: dict[str, dict[str, object]] = {}
        for audio_name, records_by_backend in self.diarization_model_history_lookup().items():
            records = list(records_by_backend.values())
            records.sort(key=lambda row: float(row.get("sort_key", 0.0) or 0.0), reverse=True)
            if records:
                lookup[audio_name] = records[0]
        return lookup

    def review_model_comparisons_for_srt(self, srt_path: Path) -> list[dict[str, object]]:
        """Return all diarization model outputs that should be switchable in one review page."""

        resolved_srt = srt_path.expanduser().resolve()
        target_stem = resolved_srt.stem
        matched_records: dict[str, dict[str, object]] = {}
        history_lookup = self.diarization_model_history_lookup()

        for audio_name, records_by_model in history_lookup.items():
            for record in records_by_model.values():
                candidate_srt = record.get("srt_path")
                if isinstance(candidate_srt, Path) and candidate_srt.is_file() and candidate_srt.resolve() == resolved_srt:
                    matched_records = records_by_model
                    break
            if matched_records:
                break

        if not matched_records:
            for audio_name, records_by_model in history_lookup.items():
                if self.diarization_output_base(audio_name) == target_stem:
                    matched_records = records_by_model
                    break

        if not matched_records:
            return []

        ordered_keys = [
            str(option.get("key", ""))
            for option in self.diarization_model_options()
            if str(option.get("key", "")).strip()
        ]
        extra_keys = sorted(key for key in matched_records if key not in ordered_keys)
        comparisons: list[dict[str, object]] = []
        for model_key in [*ordered_keys, *extra_keys]:
            record = matched_records.get(model_key)
            if not isinstance(record, dict):
                continue
            candidate_srt = record.get("srt_path")
            if not isinstance(candidate_srt, Path) or not candidate_srt.is_file():
                continue
            review_path = record.get("review_path")
            comparisons.append(
                {
                    "key": str(record.get("model_key") or model_key),
                    "label": str(record.get("model_label") or record.get("backend_label") or model_key),
                    "status": str(record.get("status") or "unknown"),
                    "run_name": str(record.get("run_name") or ""),
                    "srt_path": candidate_srt,
                    "review_path": review_path if isinstance(review_path, Path) else None,
                }
            )
        return comparisons

    def preferred_review_record(
        self,
        records_by_model: dict[str, dict[str, object]],
    ) -> dict[str, object] | None:
        """Pick the best existing review artifact for the Label action."""

        ordered_keys = [
            str(option.get("key", ""))
            for option in self.diarization_model_options()
            if str(option.get("key", "")).strip()
        ]
        extra_keys = sorted(key for key in records_by_model if key not in ordered_keys)
        for model_key in [*ordered_keys, *extra_keys]:
            record = records_by_model.get(model_key)
            if not isinstance(record, dict):
                continue
            review_path = record.get("review_path")
            srt_path = record.get("srt_path")
            if isinstance(review_path, Path) and review_path.is_file():
                # An older review that was generated before find_matching_media
                # learned about subfolders / folder-prefixed stems will contain
                # the "no matching media file was found" placeholder, even
                # though the WAV is now in audio_in/youtube_links/ and would be
                # found by the current logic. Regenerate the HTML in that case
                # so the audio player actually shows up. Cheap: one file read,
                # one regen, only when the placeholder is present.
                if isinstance(srt_path, Path) and srt_path.is_file() and self._review_html_is_missing_media(review_path):
                    try:
                        write_review_bundle(
                            srt_path=srt_path,
                            output_html=review_path,
                            report_tsv=record.get("flags_path") if isinstance(record.get("flags_path"), Path) else None,
                            audio_dir=self.audio_dir,
                            training_label_records=self.load_training_label_records(),
                            model_comparisons=self.review_model_comparisons_for_srt(srt_path),
                            fine_tuning_projects=list_projects(root=self.root),
                            quiet=True,
                        )
                    except Exception:
                        pass
                return record
            if isinstance(srt_path, Path) and srt_path.is_file() and isinstance(review_path, Path):
                try:
                    write_review_bundle(
                        srt_path=srt_path,
                        output_html=review_path,
                        report_tsv=record.get("flags_path") if isinstance(record.get("flags_path"), Path) else None,
                        audio_dir=self.audio_dir,
                        training_label_records=self.load_training_label_records(),
                        model_comparisons=self.review_model_comparisons_for_srt(srt_path),
                        fine_tuning_projects=list_projects(root=self.root),
                        quiet=True,
                    )
                except Exception:
                    continue
                if review_path.is_file():
                    return record
        return None

    _REVIEW_HTML_AUDIO_SRC = re.compile(
        r'<(?:audio|video)\b[^>]*\bsrc="([^"]+)"',
        re.IGNORECASE,
    )
    # The bundle stamps its own format version into a meta tag. We use it to
    # auto-regenerate stale review HTMLs (e.g. ones built before an auto-save
    # or audio-loading fix) the next time the row is opened, so users picking
    # up a fix don't have to manually rerun diarization.
    _REVIEW_HTML_BUNDLE_VERSION = re.compile(
        r'<meta\s+name="review-bundle-version"\s+content="(\d+)"',
        re.IGNORECASE,
    )

    def _review_html_is_missing_media(self, review_path: Path) -> bool:
        """Decide whether the existing review HTML's audio reference is broken.

        Three failure modes regenerate the bundle:

        1. The HTML was generated when no media could be located, so it
           contains the explicit "No matching media file was found" marker.
        2. The HTML embeds a media ``src=...`` whose resolved file no longer
           exists on disk. This happens after a WAV cleanup + redownload
           cycle: the old src points at ``audio_in/002_clip.wav`` but the
           replacement now lives at ``audio_in/youtube_links/001_clip.wav``,
           so the player surface is technically present but loads a 404.
        3. The HTML predates the current review-bundle format. We bump
           ``REVIEW_BUNDLE_FORMAT_VERSION`` whenever the embedded JS or
           markup changes in a way old bundles need to pick up — typically
           bug fixes around auto-save, status display, or audio handling.
        """

        try:
            text = review_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if "No matching media file was found" in text:
            return True
        version_match = self._REVIEW_HTML_BUNDLE_VERSION.search(text)
        if version_match is None:
            # Pre-versioned bundle. Definitely older than the current format.
            return True
        try:
            bundle_version = int(version_match.group(1))
        except ValueError:
            return True
        if bundle_version < REVIEW_BUNDLE_FORMAT_VERSION:
            return True
        match = self._REVIEW_HTML_AUDIO_SRC.search(text)
        if not match:
            # No media tag at all means there is nothing to play; treat that
            # as "needs regen" so the new find_matching_media gets a shot.
            return True
        raw_src = unquote(match.group(1))
        try:
            embedded_path = (review_path.parent / raw_src).resolve()
        except (OSError, ValueError):
            return True
        return not embedded_path.is_file()

    def diarization_row_state(
        self,
        record: dict[str, object] | None,
        *,
        missing_label: str = "ready to diarize",
        backend_label: str = "",
    ) -> dict[str, str]:
        """Convert a backend-specific history record into a short table status."""

        if not record:
            label = backend_label or "this model"
            return {
                "queue_state": missing_label,
                "state_class": "ready",
                "selection_ready": "yes",
                "detail": f"No previous {label} site diarization run found for this filename.",
            }
        status = self.normalize_diarization_status(str(record.get("status", "")), str(record.get("error_summary", "")))
        run_name = str(record.get("run_name") or "unknown run")
        if status in DIARIZATION_COMPLETED_STATUSES:
            label = "no speech found" if status == "no_speech" else "already diarized"
            return {
                "queue_state": label,
                "state_class": "converted",
                "selection_ready": "no",
                "detail": f"Latest completed result is in {run_name}.",
            }
        if status in DIARIZATION_ACTIVE_STATUSES:
            return {
                "queue_state": "already submitted",
                "state_class": "pending",
                "selection_ready": "no",
                "detail": f"This file is already part of active run {run_name}.",
            }
        return {
            "queue_state": "needs retry",
            "state_class": "failed",
            "selection_ready": "yes",
            "detail": str(record.get("error_summary") or f"Last attempt in {run_name} did not finish cleanly."),
        }

    def diarization_model_statuses(
        self,
        records_by_model: dict[str, dict[str, object]],
        *,
        model_options: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Summarize whether each selectable model profile has touched one audio file."""

        statuses: list[dict[str, str]] = []
        ordered_keys = [str(option.get("key", "")) for option in model_options if str(option.get("key", "")).strip()]
        extra_keys = sorted(key for key in records_by_model if key not in ordered_keys)
        option_lookup = {str(option.get("key", "")): option for option in model_options}
        for model_key in [*ordered_keys, *extra_keys]:
            option = option_lookup.get(model_key, {})
            record = records_by_model.get(model_key)
            backend = str((record or {}).get("backend", "")) if isinstance(record, dict) else str(option.get("backend", ""))
            model_label = str(option.get("label") or (record or {}).get("model_label") or self.diarization_backend_label(backend))
            row_state = self.diarization_row_state(
                record if isinstance(record, dict) else None,
                missing_label="not run",
                backend_label=model_label,
            )
            statuses.append(
                {
                    "key": model_key,
                    "backend": backend,
                    "label": model_label,
                    "status": row_state["queue_state"],
                    "stateClass": row_state["state_class"],
                    "selectionReady": row_state["selection_ready"],
                    "detail": row_state["detail"],
                    "lastRun": str((record or {}).get("last_run", "")) if isinstance(record, dict) else "",
                    "runName": str((record or {}).get("run_name", "")) if isinstance(record, dict) else "",
                }
            )
        return statuses

    def diarization_audio_rows(
        self,
        audio_paths: list[Path],
        *,
        target_model_key: str,
        model_options: list[dict[str, str]],
    ) -> list[dict[str, object]]:
        """Build the selectable diarization table with per-model history status."""

        history_lookup = self.diarization_model_history_lookup()
        option_lookup = {str(option.get("key", "")): option for option in model_options}
        target_option = option_lookup.get(target_model_key) or self.resolve_diarization_model_option(
            model_key=target_model_key,
            fallback_backend="nemo",
        )
        rows: list[dict[str, object]] = []
        for index, path in enumerate(audio_paths, start=1):
            audio_name = self.audio_relative_path(path)
            records_by_model = history_lookup.get(audio_name, {})
            record = records_by_model.get(target_option["key"])
            row_state = self.diarization_row_state(
                record if isinstance(record, dict) else None,
                backend_label=target_option["label"],
            )
            model_statuses = self.diarization_model_statuses(
                records_by_model,
                model_options=model_options,
            )
            rows.append(
                {
                    "index": str(index),
                    "name": audio_name,
                    "fileName": path.name,
                    "folder": audio_name.split("/", 1)[0] if "/" in audio_name else "Unsorted Root",
                    "type": path.suffix.lower() or "file",
                    "lastRun": str((record or {}).get("last_run", "")) if isinstance(record, dict) else "",
                    "runName": str((record or {}).get("run_name", "")) if isinstance(record, dict) else "",
                    "backend": str((record or {}).get("backend", "")) if isinstance(record, dict) else "",
                    "targetBackend": target_option["backend"],
                    "targetBackendLabel": self.diarization_backend_label(target_option["backend"]),
                    "targetModelKey": target_option["key"],
                    "targetModelLabel": target_option["label"],
                    "modelStatuses": model_statuses,
                    "selectionByModel": {
                        item["key"]: item["selectionReady"]
                        for item in model_statuses
                    },
                    **row_state,
                }
            )
        return rows

    def diarization_library_summary(self, rows: list[dict[str, object]]) -> dict[str, object]:
        """Summarize the current input library by diarization readiness."""

        model_counts: list[dict[str, object]] = []
        seen_keys: set[str] = set()
        all_status_rows = [
            item
            for row in rows
            for item in list(row.get("modelStatuses", []))
            if isinstance(item, dict)
        ]
        ordered_keys = []
        for item in all_status_rows:
            key = str(item.get("key", ""))
            if key and key not in seen_keys:
                ordered_keys.append(key)
                seen_keys.add(key)
        for model_key in ordered_keys:
            statuses = [
                item
                for item in all_status_rows
                if item.get("key") == model_key
            ]
            first = statuses[0] if statuses else {}
            model_counts.append(
                {
                    "key": model_key,
                    "backend": first.get("backend", ""),
                    "label": first.get("label", model_key),
                    "diarized": sum(1 for item in statuses if item.get("stateClass") == "converted"),
                    "ready": sum(1 for item in statuses if item.get("stateClass") == "ready"),
                    "retry": sum(1 for item in statuses if item.get("stateClass") == "failed"),
                    "active": sum(1 for item in statuses if item.get("stateClass") == "pending"),
                }
            )
        return {
            "total": len(rows),
            "ready": sum(1 for row in rows if row.get("state_class") == "ready"),
            "diarized": sum(1 for row in rows if row.get("state_class") == "converted"),
            "retry": sum(1 for row in rows if row.get("state_class") == "failed"),
            "active": sum(1 for row in rows if row.get("state_class") == "pending"),
            "modelCounts": model_counts,
        }

    def diarization_history_rows(self, limit: int = 16) -> list[dict[str, object]]:
        """Return the newest per-file, per-model diarization records for Media Library."""

        rows = [
            record
            for records_by_backend in self.diarization_model_history_lookup().values()
            for record in records_by_backend.values()
        ]
        rows.sort(key=lambda row: float(row.get("sort_key", 0.0) or 0.0), reverse=True)
        return rows[:limit]

    def diarization_item_records(
        self,
        *,
        run_dir: Path,
        selected_audio: list[str],
        summary_rows: list[dict[str, str]],
    ) -> list[dict[str, object]]:
        """Match each selected audio file with transcripts, time spans, and logs."""

        summary_by_name = {
            self.clean_audio_selection_value(row.get("audio_file", "")): row
            for row in summary_rows
            if self.clean_audio_selection_value(row.get("audio_file", ""))
        }
        selected_basenames = {self.clean_audio_selection_value(name) for name in selected_audio}

        def log_path(value: str, fallback: Path) -> Path:
            if not value:
                return fallback
            candidate = Path(value)
            return candidate if candidate.is_absolute() else run_dir / value

        def audio_path_for(audio_name: str) -> Path | None:
            candidate = Path(audio_name)
            if candidate.is_absolute() and candidate.is_file():
                return candidate
            return self.audio_path_for_relative(audio_name)

        records: list[dict[str, object]] = []
        for index, audio_name in enumerate(selected_audio, start=1):
            audio_basename = self.clean_audio_selection_value(audio_name)
            row = summary_by_name.get(audio_basename, {})
            stem = self.diarization_output_base(audio_basename)
            safe_name = self.safe_log_component(audio_basename)
            stdout_path = log_path(row.get("stdout_log", ""), run_dir / "logs" / f"{safe_name}.out")
            stderr_path = log_path(row.get("stderr_log", ""), run_dir / "logs" / f"{safe_name}.err")
            status = row.get("status", "waiting")
            error_summary = row.get("error_summary", "")
            if status not in {"ok", "waiting"} and not error_summary:
                error_summary = self.error_summary(stderr_path)
            if status == "failed" and "Whisper returned an empty transcript" in error_summary:
                status = "no_speech"
            records.append(
                {
                    "index": index,
                    "audio_file": audio_basename,
                    "audio_path": audio_path_for(audio_name),
                    "status": status,
                    "runtime_seconds": row.get("runtime_seconds", ""),
                    "error_summary": error_summary,
                    "transcript_path": run_dir / f"{stem}.txt",
                    "srt_path": run_dir / f"{stem}.srt",
                    "review_path": run_dir / f"{stem}_review.html",
                    "flags_path": run_dir / f"{stem}_review_flags.tsv",
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                }
            )

        extra_rows = [
            row
            for row in summary_rows
            if self.clean_audio_selection_value(row.get("audio_file", "")) not in selected_basenames
        ]
        for row in extra_rows:
            audio_basename = self.clean_audio_selection_value(row.get("audio_file", ""))
            if not audio_basename:
                continue
            stem = self.diarization_output_base(audio_basename)
            stderr_path = log_path(row.get("stderr_log", ""), Path(""))
            status = row.get("status", "unknown")
            error_summary = row.get("error_summary", "")
            if status != "ok" and not error_summary:
                error_summary = self.error_summary(stderr_path)
            if status == "failed" and "Whisper returned an empty transcript" in error_summary:
                status = "no_speech"
            records.append(
                {
                    "index": len(records) + 1,
                    "audio_file": audio_basename,
                    "audio_path": audio_path_for(audio_basename),
                    "status": status,
                    "runtime_seconds": row.get("runtime_seconds", ""),
                    "error_summary": error_summary,
                    "transcript_path": run_dir / f"{stem}.txt",
                    "srt_path": run_dir / f"{stem}.srt",
                    "review_path": run_dir / f"{stem}_review.html",
                    "flags_path": run_dir / f"{stem}_review_flags.tsv",
                    "stdout_path": log_path(row.get("stdout_log", ""), Path("")),
                    "stderr_path": stderr_path,
                }
            )
        return records

    def diarization_run_details(self, run_dir: Path | None) -> dict[str, object] | None:
        """Load progress details for the newest site-created diarization run."""

        if not run_dir or not run_dir.is_dir():
            return None

        metadata_path = run_dir / "metadata.json"
        metadata: dict[str, object] = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}

        selection_path = run_dir / "selected_audio.txt"
        summary_path = run_dir / "logs" / "runtime_summary.tsv"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        selected_audio = self.read_line_file(selection_path)
        summary_rows = self.read_tsv_rows(summary_path)
        completed = len(summary_rows)
        item_records = self.diarization_item_records(
            run_dir=run_dir,
            selected_audio=selected_audio,
            summary_rows=summary_rows,
        )
        succeeded = sum(1 for record in item_records if (record.get("status") or "").strip() == "ok")
        failed = sum(1 for record in item_records if (record.get("status") or "").strip() == "failed")
        no_speech = sum(1 for record in item_records if (record.get("status") or "").strip() == "no_speech")
        status = run_status(run_dir)
        if status == "failed" and completed > 0 and succeeded > 0:
            status = "completed_with_errors"
        elif status == "succeeded" and no_speech > 0:
            status = "completed_with_warnings"
        slurm_job_id = str(metadata.get("slurm_job_id") or "").strip()
        slurm_queue = self.cached_slurm_queue(slurm_job_id, status)

        # Live "currently processing" inference for active runs only:
        # the first selected audio that has no completed status is treated as
        # the in-flight file. Read-only; no edits to the diarization wrapper.
        live_progress: dict[str, object] = {}
        if status in {"running", "submitted"} and selected_audio:
            for index, audio_name in enumerate(selected_audio):
                completed_names = {(row.get("audio_file") or "").strip() for row in summary_rows}
                if audio_name not in completed_names:
                    live_progress = {
                        "current_index": index + 1,
                        "current_total": len(selected_audio),
                        "current_file": audio_name,
                        "fraction": (index) / max(len(selected_audio), 1),
                    }
                    break

        return {
            "run_dir": run_dir,
            "name": run_dir.name,
            "status": status,
            "metadata": metadata,
            "slurm_queue": slurm_queue,
            "selected_audio": selected_audio,
            "selected_count": len(selected_audio),
            "completed_count": completed,
            "succeeded_count": succeeded,
            "failed_count": failed,
            "no_speech_count": no_speech,
            "remaining_count": max(len(selected_audio) - completed, 0),
            "live_progress": live_progress,
            "summary_rows": summary_rows[:10],
            "item_records": item_records,
            "summary_path": summary_path if summary_path.is_file() else None,
            "selection_path": selection_path if selection_path.is_file() else None,
            "stdout_path": stdout_path if stdout_path.is_file() else None,
            "stderr_path": stderr_path if stderr_path.is_file() else None,
            "metadata_path": metadata_path if metadata_path.is_file() else None,
            "stdout_tail": self.tail_text(stdout_path),
            "stderr_tail": self.tail_text(stderr_path),
        }
