#!/usr/bin/env python3
"""YouTube link queue, conversion, and run history."""
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
from review_bundle import parse_srt, write_review_bundle
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


class YouTubeMixin:
    """YouTube link queue, conversion, and run history."""

    def handle_youtube_links(self, environ):
        """Append deduplicated YouTube URLs to the project queue file."""

        form = self.parse_form(environ)
        raw_urls = (form.getfirst("youtube_urls") or "").splitlines()
        new_urls = [line.strip() for line in raw_urls if line.strip()]
        if not new_urls:
            return self.redirect(environ, "/youtube", message="No YouTube URLs were provided.", status="error")

        existing = set()
        if self.youtube_links_path.is_file():
            existing = {
                line.strip()
                for line in self.youtube_links_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }

        appended = [url for url in new_urls if url not in existing]
        if appended:
            try:
                self.youtube_links_path.parent.mkdir(parents=True, exist_ok=True)
                with self.youtube_links_path.open("a", encoding="utf-8") as handle:
                    for url in appended:
                        handle.write(f"{url}\n")
            except OSError as exc:
                # Disk full, perms changed mid-request, or NFS hiccup. Surface
                # a graceful error toast instead of letting the WSGI 500 page
                # take the user out of the dashboard.
                return self.redirect(
                    environ,
                    "/youtube",
                    message=self.notification_message(
                        "Could not save the queued URL(s).",
                        str(exc),
                        f"Check disk space and write permissions on {self.youtube_links_path}.",
                    ),
                    status="error",
                )
        self.invalidate_dashboard_cache()

        return self.redirect(
            environ,
            "/youtube",
            message=f"Queued {len(appended)} new YouTube URL(s).",
            status="success",
        )

    def handle_youtube_link_delete(self, environ):
        """Remove one queued YouTube URL and clean up any converted media."""

        form = self.parse_form(environ)
        url = (form.getfirst("youtube_url") or form.getfirst("url") or "").strip()
        if not url:
            return self.redirect(
                environ,
                "/youtube",
                message="Choose a YouTube link to remove.",
                status="error",
            )
        try:
            summary = self.delete_youtube_link(url)
        except ValueError as exc:
            return self.redirect(environ, "/youtube", message=str(exc), status="error")
        except OSError as exc:
            return self.redirect(
                environ,
                "/youtube",
                message=f"Could not remove that YouTube link: {exc}",
                status="error",
            )

        details = [
            f"Removed {summary['queue_entries_removed']} queue entr{'y' if summary['queue_entries_removed'] == 1 else 'ies'}.",
            f"Removed {summary['history_rows_removed']} history row(s).",
        ]
        if summary["audio_files_removed"]:
            details.append(f"Deleted {summary['audio_files_removed']} converted audio file(s) and its generated artifacts.")
        if summary["renamed_files"]:
            details.append(f"Renumbered {len(summary['renamed_files'])} remaining audio file(s) so the folder stays 001, 002, 003.")
        return self.redirect(
            environ,
            "/youtube",
            message=self.notification_message("Removed YouTube link.", url, *details),
            status="success",
        )

    def handle_review(self, environ):
        """Generate a review bundle for an existing subtitle file."""

        form = self.parse_form(environ)
        srt_value = (form.getfirst("srt_path") or "").strip()
        if not srt_value:
            return self.redirect(environ, "/uploads", message="Select an SRT file first.", status="error")
        srt_path = self.resolve_local_path(srt_value)
        media_value = (form.getfirst("media_path") or "").strip()
        media_path = self.resolve_local_path(media_value) if media_value else None
        html_path, _ = write_review_bundle(
            srt_path=srt_path,
            media_path=media_path,
            training_label_records=self.load_training_label_records(),
            model_comparisons=self.review_model_comparisons_for_srt(srt_path),
            quiet=True,
        )
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/uploads",
            message=f"Generated review bundle: {self.describe_path(html_path)}",
            status="success",
        )

    def handle_youtube_conversion(self, environ):
        """Launch a background YouTube conversion run for selected or queued URLs."""

        form = self.parse_form(environ)
        selected_urls = self.ordered_unique(form.getlist("selected_urls"))
        mode = (form.getfirst("youtube_mode") or "").strip().lower()
        run_all = mode == "all" or bool(form.getfirst("youtube_convert_all"))
        urls = self.youtube_queue_entries() if run_all else selected_urls
        if not urls:
            return self.redirect(
                environ,
                "/youtube",
                message="Select at least one queued YouTube URL or choose the convert-all option.",
                status="error",
            )
        folder_value, target_folder = self.selected_audio_folder_path(
            form,
            default_folder=DEFAULT_YOUTUBE_AUDIO_FOLDER,
        )

        run_dir = self.youtube_runs_root / self.run_directory_name(
            "youtube-conversion",
            count=len(urls),
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        selection_path = run_dir / "selected_urls.txt"
        selection_path.write_text("\n".join(urls) + "\n", encoding="utf-8")
        force_redownload = bool(form.getfirst("youtube_force_redownload"))
        metadata = {
            "operation": "youtube_conversion",
            "mode": "all" if run_all else "selected",
            "selected_url_count": len(urls),
            "selected_urls_path": str(selection_path),
            "force_redownload": force_redownload,
            "audio_folder": folder_value or ROOT_AUDIO_FOLDER_VALUE,
            "audio_output_dir": str(target_folder),
        }

        if shutil.which("sbatch") and self.site_youtube_sbatch.is_file():
            try:
                import workflow_dashboard as _wd  # lazy: honor monkey-patches in tests
                submission = _wd.submit_sbatch_job(
                    sbatch_script=self.site_youtube_sbatch,
                    cwd=self.root,
                    run_dir=run_dir,
                    export_env={
                        "SITE_ROOT_DIR": str(self.root),
                        "SITE_YOUTUBE_RUN_DIR": str(run_dir),
                        "SITE_YOUTUBE_URLS_FILE": str(selection_path),
                        "SITE_YOUTUBE_QUEUE_FILE": str(self.youtube_links_path),
                        "SITE_YOUTUBE_INDEX_FILE": str(self.youtube_history_index_path),
                        "SITE_YOUTUBE_FORCE_REDOWNLOAD": "1" if force_redownload else "0",
                        "SITE_YOUTUBE_AUDIO_DIR": str(target_folder),
                    },
                    metadata=metadata,
                    job_label="YouTube audio conversion",
                )
            except RuntimeError as exc:
                self.invalidate_dashboard_cache()
                return self.redirect(
                    environ,
                    "/youtube",
                    message=self.notification_message(
                        "YouTube conversion job was not submitted.",
                        str(exc),
                        "Check that this site is running on WAVE with sbatch available.",
                    ),
                    status="error",
                )

            self.invalidate_dashboard_cache()
            job_id = submission.get("slurm_job_id") or "unknown"
            return self.redirect(
                environ,
                "/youtube",
                message=f"Submitted Slurm YouTube audio conversion job {job_id} for run '{run_dir.name}' with {len(urls)} link(s).",
                status="success",
            )

        command = [
            sys.executable,
            "workflow_cli.py",
            "convert-youtube-selection",
            "--urls-file",
            str(selection_path),
            "--queue-file",
            str(self.youtube_links_path),
            "--run-dir",
            str(run_dir),
            "--index-file",
            str(self.youtube_history_index_path),
            "--output-dir",
            str(target_folder),
        ]
        if force_redownload:
            command.append("--force-redownload")
        import workflow_dashboard as _wd  # lazy: honor monkey-patches in tests
        _wd.launch_background_command(
            command=command,
            cwd=self.root,
            run_dir=run_dir,
            metadata={**metadata, "runner": "local_background"},
        )
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/youtube",
            message=f"Started YouTube conversion run '{run_dir.name}' for {len(urls)} link(s).",
            status="success",
        )

    def handle_youtube_reset(self, environ):
        """Clear YouTube queue state so the next bulk batch starts from a clean slate."""

        summary = self.reset_youtube_workspace()
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/youtube",
            message=(
                "Reset the YouTube workspace: "
                f"removed {summary['audio_files_removed']} media file(s), "
                f"cleared {summary['queue_entries_cleared']} queued link(s), "
                f"deleted {summary['run_items_removed']} old run item(s), and "
                f"removed {summary['failed_reports_removed']} failed-link report(s)."
            ),
            status="success",
        )

    def youtube_preview(self, limit: int = 8) -> list[str]:
        """Show only the first few queued URLs to keep the dashboard compact."""

        if not self.youtube_links_path.is_file():
            return []
        urls = [
            line.strip()
            for line in self.youtube_links_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return urls[:limit]

    def youtube_queue_entries(self) -> list[str]:
        """Return the full YouTube queue for selection tables and batch launches."""

        return self.cached_value(
            "youtube_queue_entries",
            ttl_seconds=3.0,
            builder=lambda: [
                line.strip()
                for line in self.youtube_links_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if self.youtube_links_path.is_file()
            else [],
        )

    def youtube_history_rows(self, limit: int = 12) -> list[dict[str, str]]:
        """Load the most recent YouTube conversion records from the history index."""

        def builder() -> list[dict[str, str]]:
            if not self.youtube_history_index_path.is_file():
                return []
            with self.youtube_history_index_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            rows.sort(key=lambda row: row.get("last_attempt_utc", ""), reverse=True)
            return rows[:limit]

        return self.cached_value(
            f"youtube_history_rows:{limit}",
            ttl_seconds=3.0,
            builder=builder,
        )

    def youtube_history_lookup(self) -> dict[str, dict[str, str]]:
        """Map each queued YouTube URL to its latest conversion record."""

        def builder() -> dict[str, dict[str, str]]:
            if not self.youtube_history_index_path.is_file():
                return {}
            lookup: dict[str, dict[str, str]] = {}
            for row in self.read_tsv_rows(self.youtube_history_index_path):
                url = (row.get("url") or "").strip()
                if url:
                    lookup[url] = row
            return lookup

        return self.cached_value(
            "youtube_history_lookup",
            ttl_seconds=3.0,
            builder=builder,
        )

    def youtube_queue_rows(self) -> list[dict[str, str]]:
        """Combine queued URLs with history so the table can explain each row."""

        def builder() -> list[dict[str, str]]:
            rows: list[dict[str, str]] = []
            history_lookup = self.youtube_history_lookup()
            for index, url in enumerate(self.youtube_queue_entries(), start=1):
                record = history_lookup.get(url, {})
                history_status = (record.get("status") or "").strip().lower()
                audio_path_value = (record.get("audio_path") or "").strip()
                audio_file = (record.get("audio_file") or "").strip()
                audio_exists = False
                if audio_path_value:
                    try:
                        audio_exists = self.resolve_under_root(audio_path_value).is_file()
                    except ValueError:
                        audio_exists = False
                elif audio_file:
                    audio_exists = (self.audio_dir / audio_file).is_file()

                if history_status == "ok" and audio_exists:
                    queue_state = "already converted"
                    state_class = "converted"
                    selection_ready = "no"
                    detail = audio_file or "This link already has a saved audio file."
                elif history_status in {"retry", "failed"}:
                    queue_state = "retry needed"
                    state_class = "failed"
                    selection_ready = "yes"
                    detail = (record.get("note") or "The last conversion attempt failed.").strip()
                elif history_status == "no_data":
                    queue_state = "no public data"
                    state_class = "no-data"
                    selection_ready = "no"
                    detail = (
                        (record.get("note") or "").strip()
                        or "The last attempt found no public audio data for this URL."
                    )
                else:
                    queue_state = "ready to convert"
                    state_class = "ready"
                    selection_ready = "yes"
                    detail = (
                        "History exists, but the saved audio file is missing. The next run will rebuild it."
                        if history_status == "ok"
                        else "This link has not been converted yet."
                    )

                rows.append(
                    {
                        "index": str(index),
                        "url": url,
                        "queue_state": queue_state,
                        "state_class": state_class,
                        "selection_ready": selection_ready,
                        "detail": detail,
                    }
                )
            return rows

        return self.cached_value(
            "youtube_queue_rows",
            ttl_seconds=3.0,
            builder=builder,
        )

    def youtube_failed_rows(self, limit: int = 8) -> list[dict[str, str]]:
        """Normalize the latest YouTube issue report for table rendering."""

        def builder() -> list[dict[str, str]]:
            normalized_rows: list[dict[str, str]] = []
            for row in self.read_tsv_rows(self.failed_links_path):
                summary = (row.get("summary") or row.get("error") or row.get("note") or "").strip()
                issue_category = self.youtube_issue_category(
                    summary,
                    explicit_category=(row.get("issue_category") or ""),
                )
                explicit_removed = (
                    row.get("removed_from_links_file")
                    or row.get("removed_from_list")
                    or ""
                ).strip().lower()
                queue_action = (row.get("queue_action") or "").strip() or (
                    "remove_from_queue" if issue_category == "no_data" else "keep_in_queue"
                )
                queue_result = (row.get("queue_result") or "").strip() or (
                    "removed"
                    if queue_action == "remove_from_queue" and explicit_removed == "yes"
                    else "kept"
                )
                normalized_rows.append(
                    {
                        "url": (row.get("url") or "").strip(),
                        "summary": summary,
                        "issue_category": issue_category,
                        "queue_action": queue_action,
                        "queue_result": queue_result,
                        "when": (
                            row.get("last_attempt_utc")
                            or row.get("run_utc")
                            or row.get("job_id")
                            or ""
                        ).strip(),
                        "detail_log": (row.get("detail_log") or "").strip(),
                    }
                )
            normalized_rows.sort(key=lambda row: row.get("when", ""), reverse=True)
            return normalized_rows[:limit]

        return self.cached_value(
            f"youtube_failed_rows:{limit}",
            ttl_seconds=3.0,
            builder=builder,
        )

    def active_youtube_runs_detailed(self, *, limit: int = 4) -> list[dict[str, object]]:
        """Return ``youtube_run_details`` payloads for currently active YouTube runs.

        Capped at ``limit`` to bound polling work. Used by the frontend's
        tab-toggle for switching between simultaneous runs.
        """

        active_paths = self.active_run_directories(self.youtube_runs_root)
        active_paths.sort(key=lambda path: path.name, reverse=True)
        details = []
        for run_dir in active_paths[:limit]:
            payload = self.youtube_run_details(run_dir)
            if payload:
                details.append(payload)
        return details

    def youtube_run_details(self, run_dir: Path | None) -> dict[str, object] | None:
        """Load the latest YouTube run summary, artifacts, and short log previews."""

        if not run_dir or not run_dir.is_dir():
            return None

        metadata_path = run_dir / "metadata.json"
        metadata: dict[str, object] = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}

        resolution_path = run_dir / "conversion_report.tsv"
        if not resolution_path.is_file():
            resolution_path = run_dir / "url_resolution.tsv"
        removed_path = run_dir / "queue_updates.tsv"
        if not removed_path.is_file():
            removed_path = run_dir / "removed_failed_urls.tsv"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        selected_urls_path = run_dir / "selected_urls.txt"
        resolution_rows = self.read_tsv_rows(resolution_path)
        removed_rows = self.read_tsv_rows(removed_path)

        summary = {
            "downloaded": 0,
            "skipped": 0,
            "retry": 0,
            "no_data": 0,
        }
        for row in resolution_rows:
            result = (row.get("status") or row.get("result") or "").strip().lower()
            if result == "downloaded":
                summary["downloaded"] += 1
            elif result in {"retry", "failed"}:
                summary["retry"] += 1
            elif result == "no_data":
                summary["no_data"] += 1
            elif result:
                summary["skipped"] += 1

        removed_from_queue = 0
        for row in removed_rows:
            summary_text = (row.get("summary") or row.get("error") or row.get("note") or "").strip()
            issue_category = self.youtube_issue_category(
                summary_text,
                explicit_category=(row.get("issue_category") or ""),
            )
            queue_action = (row.get("queue_action") or "").strip() or (
                "remove_from_queue" if issue_category == "no_data" else "keep_in_queue"
            )
            queue_result = (row.get("queue_result") or row.get("removed_from_list") or row.get("removed_from_links_file") or "").strip().lower()
            if queue_action == "remove_from_queue" and queue_result in {"yes", "removed"}:
                removed_from_queue += 1
        retry_preview = [
            row
            for row in resolution_rows
            if (row.get("status") or row.get("result") or "").strip().lower() in {"retry", "failed"}
        ][:6]
        no_data_preview = [
            row
            for row in resolution_rows
            if (row.get("status") or row.get("result") or "").strip().lower() == "no_data"
        ][:6]

        status = run_status(run_dir)
        slurm_job_id = str(metadata.get("slurm_job_id") or "").strip()
        import workflow_dashboard as _wd  # lazy: honor monkey-patches in tests
        slurm_queue = (
            self.cached_value(
                f"slurm_queue::{slurm_job_id}",
                ttl_seconds=2.0,
                builder=lambda: _wd.slurm_queue_snapshot(slurm_job_id),
            )
            if slurm_job_id and status in {"submitted", "running"}
            else {}
        )

        return {
            "run_dir": run_dir,
            "status": status,
            "name": run_dir.name,
            "metadata": metadata,
            "slurm_queue": slurm_queue,
            "summary": summary,
            "removed_from_queue": removed_from_queue,
            "resolution_preview": resolution_rows[:8],
            "retry_preview": retry_preview,
            "no_data_preview": no_data_preview,
            "stdout_path": stdout_path if stdout_path.is_file() else None,
            "stderr_path": stderr_path if stderr_path.is_file() else None,
            "resolution_path": resolution_path if resolution_path.is_file() else None,
            "removed_path": removed_path if removed_path.is_file() else None,
            "metadata_path": metadata_path if metadata_path.is_file() else None,
            "selected_urls_path": selected_urls_path if selected_urls_path.is_file() else None,
            "stdout_tail": self.tail_text(stdout_path),
            "stderr_tail": self.tail_text(stderr_path),
        }

    def reset_youtube_workspace(self) -> dict[str, int]:
        """Delete the existing YouTube queue, generated audio, and site run history."""

        self.ensure_default_audio_folders()
        self.youtube_runs_root.mkdir(parents=True, exist_ok=True)
        self.youtube_history_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.failed_links_path.parent.mkdir(parents=True, exist_ok=True)

        audio_files_removed = 0
        youtube_audio_dir = self.audio_folder_path(DEFAULT_YOUTUBE_AUDIO_FOLDER, create=True)
        for path in sorted(youtube_audio_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.name == ".gitkeep":
                continue
            if path.is_dir() and path.name.startswith(".youtube_tmp_"):
                shutil.rmtree(path)
                audio_files_removed += 1
                continue
            if path.is_file() and path.suffix.lower() in WORKSPACE_MEDIA_SUFFIXES:
                path.unlink()
                audio_files_removed += 1
        for path in self.audio_dir.glob(".youtube_tmp_*"):
            if path.is_dir():
                shutil.rmtree(path)
                audio_files_removed += 1
        for path in self.audio_dir.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_file() and path.suffix.lower() in WORKSPACE_MEDIA_SUFFIXES:
                path.unlink()
                audio_files_removed += 1

        history_audio_paths: set[Path] = set()
        if self.youtube_history_index_path.is_file():
            for row in self.read_tsv_rows(self.youtube_history_index_path):
                raw_path = (row.get("audio_path") or "").strip()
                if not raw_path:
                    continue
                try:
                    candidate = self.resolve_under_root(raw_path)
                    candidate.relative_to(self.audio_dir.resolve())
                except ValueError:
                    continue
                if candidate.is_file() and candidate.suffix.lower() in WORKSPACE_MEDIA_SUFFIXES:
                    history_audio_paths.add(candidate)
        for path in sorted(history_audio_paths):
            if path.is_file():
                path.unlink()
                audio_files_removed += 1

        run_items_removed = 0
        if self.youtube_runs_root.is_dir():
            for path in self.youtube_runs_root.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                run_items_removed += 1

        queue_entries_cleared = len(self.youtube_queue_entries())
        self.youtube_links_path.write_text("", encoding="utf-8")

        history_entries_removed = 0
        if self.youtube_history_index_path.is_file():
            history_entries_removed = max(len(self.read_tsv_rows(self.youtube_history_index_path)), 0)
            self.youtube_history_index_path.unlink()

        failed_reports_removed = 0
        failed_links_dir = self.failed_links_path.parent
        if failed_links_dir.is_dir():
            for path in failed_links_dir.glob("*.tsv"):
                path.unlink()
                failed_reports_removed += 1

        return {
            "audio_files_removed": audio_files_removed,
            "run_items_removed": run_items_removed,
            "queue_entries_cleared": queue_entries_cleared,
            "history_entries_removed": history_entries_removed,
            "failed_reports_removed": failed_reports_removed,
        }

    def remove_youtube_url_from_queue(self, url: str) -> int:
        """Remove exact URL matches from youtube_links.txt while preserving comments."""

        target_url = (url or "").strip()
        if not target_url or not self.youtube_links_path.is_file():
            return 0
        removed = 0
        kept_lines: list[str] = []
        for line in self.youtube_links_path.read_text(encoding="utf-8").splitlines(keepends=True):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and stripped == target_url:
                removed += 1
                continue
            kept_lines.append(line)
        if removed:
            self.youtube_links_path.write_text("".join(kept_lines), encoding="utf-8")
        return removed

    def youtube_audio_path_for_row(self, row: dict[str, str]) -> Path | None:
        """Resolve the converted audio file referenced by one YouTube index row."""

        row_path = (row.get("audio_path") or "").strip()
        if row_path:
            try:
                candidate = self.resolve_under_root(row_path)
                candidate.resolve().relative_to(self.audio_dir.resolve())
            except (OSError, ValueError):
                candidate = None
            if candidate and candidate.is_file() and candidate.suffix.lower() in WORKSPACE_MEDIA_SUFFIXES:
                return candidate.resolve()

        audio_file = (row.get("audio_file") or "").strip()
        if not audio_file:
            return None
        cleaned = self.clean_audio_selection_value(audio_file)
        if cleaned:
            direct = self.audio_path_for_relative(cleaned)
            if direct is not None:
                return direct.resolve()

        basename = Path(audio_file).name
        matches: list[Path] = []
        if basename and self.audio_dir.is_dir():
            for candidate in self.audio_dir.rglob(basename):
                if candidate.is_file() and candidate.suffix.lower() in WORKSPACE_MEDIA_SUFFIXES:
                    matches.append(candidate.resolve())
        return matches[0] if len(matches) == 1 else None

    def row_audio_path_matches(self, row: dict[str, str], old_path: Path) -> bool:
        """Return whether a TSV row points at an audio file before it was renamed."""

        old_resolved = old_path.resolve()
        row_path = (row.get("audio_path") or "").strip()
        if row_path:
            try:
                return Path(row_path).expanduser().resolve() == old_resolved
            except OSError:
                return False
        return (row.get("audio_file") or "").strip() == old_path.name

    def delete_youtube_link(self, url: str) -> dict[str, object]:
        """Delete a queued YouTube link, its converted WAV, and compact numbering."""

        target_url = (url or "").strip()
        if not target_url:
            raise ValueError("Choose a YouTube link to remove.")

        matching_rows = [
            row
            for row in self.read_tsv_rows(self.youtube_history_index_path)
            if (row.get("url") or "").strip() == target_url
        ]
        audio_paths: list[Path] = []
        for row in matching_rows:
            audio_path = self.youtube_audio_path_for_row(row)
            if audio_path is not None and audio_path not in audio_paths:
                audio_paths.append(audio_path)

        affected_folders = {path.parent.resolve() for path in audio_paths}
        audio_files_removed = 0
        artifacts_removed = 0
        summary_rows_removed = 0
        selection_lines_removed = 0
        history_rows_removed = 0
        for audio_path in audio_paths:
            if not audio_path.is_file():
                continue
            delete_summary = self.cascade_delete_audio_file(self.audio_relative_path(audio_path))
            if delete_summary.get("audio_was_present"):
                audio_files_removed += 1
            history_rows_removed += len(list(delete_summary.get("youtube_urls_cleared") or []))
            artifacts_removed += int(delete_summary.get("diarization_artifacts_removed", 0))
            summary_rows_removed += int(delete_summary.get("summary_rows_removed", 0))
            selection_lines_removed += int(delete_summary.get("selection_lines_removed", 0))

        if self.youtube_history_index_path.is_file():
            rows = self.read_tsv_rows(self.youtube_history_index_path)
            kept_rows = [
                row
                for row in rows
                if (row.get("url") or "").strip() != target_url
            ]
            history_rows_removed = len(rows) - len(kept_rows)
            if history_rows_removed:
                self.write_tsv_rows(self.youtube_history_index_path, YOUTUBE_INDEX_COLUMNS, kept_rows)

        renamed_files: list[dict[str, str]] = []
        for folder in sorted(affected_folders, key=lambda path: str(path)):
            renamed_files.extend(self.compact_audio_folder_numbering(folder))

        queue_entries_removed = self.remove_youtube_url_from_queue(target_url)
        self.invalidate_dashboard_cache()
        return {
            "url": target_url,
            "queue_entries_removed": queue_entries_removed,
            "history_rows_removed": history_rows_removed,
            "audio_files_removed": audio_files_removed,
            "diarization_artifacts_removed": artifacts_removed,
            "summary_rows_removed": summary_rows_removed,
            "selection_lines_removed": selection_lines_removed,
            "renamed_files": renamed_files,
        }
