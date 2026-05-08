#!/usr/bin/env python3
"""Rendering mixin: builds the JSON page-state blob and serves the React HTML shell.

The browser side doesn't talk to a REST API — it boots from a single page
template with a chunk of JSON shoved into ``<script id="dashboard-state">``.
This mixin is what assembles that JSON: page context, asset URLs, route
table, defaults, queue snapshots, etc. Frontend changes that need new data
end up here first, then in ``dashboard_app.js`` to consume it.
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
    find_pyannote_run_out_log,
    list_projects,
    normalize_backend,
    parse_pyannote_val_loss,
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


class RenderingMixin:
    """Assembles the per-page JSON state blob that the frontend's React components read from window.__STATE__."""

    def shared_page_context(self) -> dict[str, object]:
        """Build the small context dict that every page needs (audio count, queue size, latest run paths)."""

        return {
            "audio_count": self.audio_input_count(),
            "queue_count": self.queue_size(),
            "project_count": self.project_count(),
            "failed_links_present": self.failed_links_path.is_file(),
            "tracking": self.active_tracking_summary(),
            **self.latest_output_roots(),
        }

    def page_context(
        self,
        current_path: str,
        *,
        diarization_model_key: str = "",
    ) -> dict[str, object]:
        """Lazy-load per-page data — expensive scans (audio inventory, project list) only happen for pages that show them."""

        effective_path = current_path if current_path in PAGE_PATHS else "/"
        context = self.shared_page_context()
        if effective_path in AUDIO_INVENTORY_PAGES:
            audio_files = self.audio_inventory()
            context["audio_files"] = audio_files
            context["audio_folders"] = self.audio_folder_rows(audio_files=audio_files)
        if effective_path == "/youtube" and "audio_folders" not in context:
            # /youtube isn't in AUDIO_INVENTORY_PAGES, so we only get the pre-loaded
            # inventory if another page on the same request already triggered it.
            context["audio_folders"] = self.audio_folder_rows()
        if effective_path in RECENT_OUTPUT_PAGES:
            context["recent_outputs"] = self.recent_output_files(limit=30)
        if effective_path in RECENT_SRT_PAGES:
            context["recent_srts"] = self.recent_srt_files()
        if effective_path in PROJECT_SUMMARY_PAGES:
            context["projects"] = self.project_summaries()
        if effective_path in YOUTUBE_QUEUE_PAGES:
            context["youtube_preview"] = self.youtube_preview(limit=12)
            context["youtube_queue"] = self.youtube_queue_entries()
            context["youtube_queue_rows"] = self.youtube_queue_rows()
            queue_rows = context["youtube_queue_rows"]
            context["youtube_queue_summary"] = {
                "total": len(queue_rows),
                "ready": sum(1 for row in queue_rows if row["state_class"] == "ready"),
                "retry": sum(1 for row in queue_rows if row["state_class"] == "failed"),
                "no_data": sum(1 for row in queue_rows if row["state_class"] == "no-data"),
                "converted": sum(1 for row in queue_rows if row["state_class"] == "converted"),
            }
            context["youtube_history"] = self.youtube_history_rows(limit=16)
            context["youtube_failed_rows"] = self.youtube_failed_rows(limit=8)
            issue_rows = context["youtube_failed_rows"]
            context["youtube_retry_rows"] = [
                row for row in issue_rows if row.get("issue_category") == "retry"
            ]
            context["youtube_no_data_rows"] = [
                row for row in issue_rows if row.get("issue_category") == "no_data"
            ]
            context["youtube_latest_run"] = self.youtube_run_details(context["latest_site_youtube"])
            context["youtube_active_runs"] = self.active_youtube_runs_detailed()
        if effective_path in MODEL_SELECTION_PAGES:
            context["preferences"] = self.model_preferences()
        if effective_path in {"/uploads", "/training-labels", "/fine-tuning"}:
            # Capped at 25 — the frontend filter dropdown only displays recent
            # runs; 100 was bloating the state blob to multi-MB. Older runs
            # remain accessible from the diarization tab's run history.
            context["diarization_history"] = self.diarization_history_rows(limit=25)
            context["diarization_latest_run"] = self.diarization_run_details(context["latest_site_diarization"])
            context["diarization_active_runs"] = self.active_diarization_runs_detailed()
        if effective_path in {"/training-labels", "/fine-tuning"}:
            context["diarization_model_options"] = self.diarization_model_options()
        if effective_path in TRAINING_LABEL_CONTEXT_PAGES:
            label_records = self.load_training_label_records()
            audio_files = list(context.get("audio_files", []))
            context["training_label_records"] = label_records
            context["training_label_summary"] = self.training_label_summary(audio_files, label_records)
        if effective_path == "/stitching":
            context["stitched_summary"] = self.stitched_summary()
        if effective_path == "/diarization":
            preferences = context.get("preferences") or self.model_preferences()
            model_options = self.diarization_model_options()
            target_option = self.resolve_diarization_model_option(
                model_key=diarization_model_key or str(preferences.get("default_diarization_model_key") or ""),
                fallback_backend=str(preferences["default_backend"]),
            )
            diarization_audio_rows = self.diarization_audio_rows(
                list(context.get("audio_files", [])),
                target_model_key=target_option["key"],
                model_options=model_options,
            )
            context["diarization_model_options"] = model_options
            context["selected_diarization_model_key"] = target_option["key"]
            context["diarization_audio_rows"] = diarization_audio_rows
            context["diarization_library_summary"] = self.diarization_library_summary(diarization_audio_rows)
            context["diarization_latest_run"] = self.diarization_run_details(context["latest_site_diarization"])
            context["diarization_active_runs"] = self.active_diarization_runs_detailed()
        if effective_path == "/fine-tuning":
            projects = context.get("projects", [])
            context["training_source_files"] = self.training_source_files()
            context["fine_tuning_summary"] = self.fine_tuning_summary(projects)
        return context

    def frontend_route(self, path: str, script_name: str) -> str:
        """Build an absolute-path URL for a backend route so the frontend never hard-codes paths."""

        return self.with_prefix(path, script_name)

    def frontend_asset(self, relative_path: str, script_name: str) -> str:
        """Resolve a URL for a static asset under /assets/ so templates don't need to know the prefix.

        Appends an mtime-based ``?v=...`` cache buster. Cache-Control: no-store
        usually does the job, but every so often a browser holds onto a stale
        bundle through a refresh or two. Fingerprinting the URL itself is the
        belt-and-braces fix — when the file changes, the URL changes, and the
        browser has no choice but to refetch.
        """

        url = self.with_prefix("/assets/" + quote(relative_path, safe="/"), script_name)
        try:
            asset_path = (self.frontend_dir / relative_path).resolve()
            asset_path.relative_to(self.frontend_dir.resolve())
            mtime = int(asset_path.stat().st_mtime)
        except (OSError, ValueError):
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}v={mtime}"

    def frontend_file_record(self, path: Path, script_name: str) -> dict[str, str]:
        """Turn a workspace Path into the flat dict the React file-table components expect."""

        try:
            relative = str(path.resolve().relative_to(self.root))
        except ValueError:
            relative = str(path)
        return {
            "name": path.name,
            "path": relative,
            "type": path.suffix.lower() or "file",
            "href": self.file_link(path, script_name) if path.is_file() else "",
        }

    def frontend_artifact_link(self, label: str, path: object, script_name: str) -> dict[str, str] | None:
        """Build a clickable artifact entry for the run-details dialog, or None if the file doesn't exist yet."""

        if not isinstance(path, Path) or not path.is_file():
            return None
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return {
            "label": label,
            "href": self.file_link(path, script_name),
            "path": self.describe_path(path),
            "kind": self.artifact_kind(path, content_type),
            "previewHref": self.artifact_preview_link(path, script_name),
        }

    def frontend_youtube_history_rows(
        self,
        rows: list[dict[str, str]],
        script_name: str,
    ) -> list[dict[str, str]]:
        """Turn raw YouTube history TSV rows into the normalized dicts the React table expects, with audio hrefs resolved."""

        serialized_rows: list[dict[str, str]] = []
        for row in rows:
            raw_status = (row.get("status", "") or "").strip().lower()
            display_status = self.youtube_issue_category(row.get("note", "")) if raw_status == "failed" else raw_status
            audio_path_value = (row.get("audio_path") or "").strip()
            audio_href = ""
            if audio_path_value:
                try:
                    candidate = self.resolve_under_root(audio_path_value)
                except ValueError:
                    candidate = None
                if candidate and candidate.is_file():
                    audio_href = self.file_link(candidate, script_name)
            serialized_rows.append(
                {
                    "status": display_status.replace("_", " "),
                    "title": row.get("title", "") or row.get("video_id", ""),
                    "audioFile": row.get("audio_file", ""),
                    "audioHref": audio_href,
                    "note": row.get("note", ""),
                    "lastAttempt": row.get("last_attempt_utc", ""),
                }
            )
        return serialized_rows

    def frontend_youtube_run(self, latest_run: object, script_name: str) -> dict[str, object] | None:
        """Shape a YouTube run summary dict into the payload the details-dialog component reads."""

        if not isinstance(latest_run, dict):
            return None
        artifact_links = [
            link
            for link in [
                self.frontend_artifact_link("selected_urls.txt", latest_run.get("selected_urls_path"), script_name),
                self.frontend_artifact_link("conversion_report.tsv", latest_run.get("resolution_path"), script_name),
                self.frontend_artifact_link("queue_updates.tsv", latest_run.get("removed_path"), script_name),
                self.frontend_artifact_link("metadata.json", latest_run.get("metadata_path"), script_name),
                self.frontend_artifact_link("activity.log", latest_run.get("stdout_path"), script_name),
                self.frontend_artifact_link("error.log", latest_run.get("stderr_path"), script_name),
            ]
            if link
        ]
        return {
            "name": latest_run.get("name", ""),
            "status": latest_run.get("status", "unknown"),
            "metadata": latest_run.get("metadata") or {},
            "summary": latest_run.get("summary") or {},
            "removedFromQueue": latest_run.get("removed_from_queue", 0),
            "resolutionPreview": latest_run.get("resolution_preview") or [],
            "retryPreview": latest_run.get("retry_preview") or [],
            "noDataPreview": latest_run.get("no_data_preview") or [],
            "stdoutTail": latest_run.get("stdout_tail") or "",
            "stderrTail": latest_run.get("stderr_tail") or "",
            "slurmQueue": latest_run.get("slurm_queue") or {},
            "artifactLinks": artifact_links,
        }

    def frontend_diarization_run(self, latest_run: object, script_name: str) -> dict[str, object] | None:
        """Shape a diarization run summary (with per-file records) into the payload the run-details dialog reads."""

        if not isinstance(latest_run, dict):
            return None
        artifact_links = [
            link
            for link in [
                self.frontend_artifact_link("selected_audio.txt", latest_run.get("selection_path"), script_name),
                self.frontend_artifact_link("runtime_summary.tsv", latest_run.get("summary_path"), script_name),
                self.frontend_artifact_link("metadata.json", latest_run.get("metadata_path"), script_name),
                self.frontend_artifact_link("stdout.log", latest_run.get("stdout_path"), script_name),
                self.frontend_artifact_link("stderr.log", latest_run.get("stderr_path"), script_name),
            ]
            if link
        ]
        item_records: list[dict[str, object]] = []
        for record in latest_run.get("item_records") or []:
            if not isinstance(record, dict):
                continue
            item_links = [
                link
                for link in [
                    self.frontend_artifact_link("Source Audio", record.get("audio_path"), script_name),
                    self.frontend_artifact_link("Transcript", record.get("transcript_path"), script_name),
                    self.frontend_artifact_link("Diarized Times", record.get("srt_path"), script_name),
                    self.frontend_artifact_link("Review Page", record.get("review_path"), script_name),
                    self.frontend_artifact_link("Review Flags", record.get("flags_path"), script_name),
                    self.frontend_artifact_link("stdout", record.get("stdout_path"), script_name),
                    self.frontend_artifact_link("stderr", record.get("stderr_path"), script_name),
                ]
                if link
            ]
            item_records.append(
                {
                    "index": record.get("index", len(item_records) + 1),
                    "audioFile": record.get("audio_file", ""),
                    "audioHref": (
                        self.file_link(record["audio_path"], script_name)
                        if isinstance(record.get("audio_path"), Path) and record["audio_path"].is_file()
                        else ""
                    ),
                    "status": record.get("status", "waiting"),
                    "runtimeSeconds": record.get("runtime_seconds", ""),
                    "errorSummary": record.get("error_summary", ""),
                    "links": item_links,
                }
            )
        metadata_payload = latest_run.get("metadata") or {}
        batch_id_value = ""
        if isinstance(metadata_payload, dict):
            batch_id_value = str(metadata_payload.get("batch_id") or "")
        return {
            "name": latest_run.get("name", ""),
            "status": latest_run.get("status", "unknown"),
            "metadata": metadata_payload,
            "slurmQueue": latest_run.get("slurm_queue") or {},
            "selectedCount": latest_run.get("selected_count", 0),
            "completedCount": latest_run.get("completed_count", 0),
            "succeededCount": latest_run.get("succeeded_count", 0),
            "failedCount": latest_run.get("failed_count", 0),
            "noSpeechCount": latest_run.get("no_speech_count", 0),
            "remainingCount": latest_run.get("remaining_count", 0),
            "liveProgress": latest_run.get("live_progress") or {},
            "summaryRows": latest_run.get("summary_rows") or [],
            "stdoutTail": latest_run.get("stdout_tail") or "",
            "stderrTail": latest_run.get("stderr_tail") or "",
            "batchId": batch_id_value,
            "artifactLinks": artifact_links,
            "items": item_records,
        }

    def frontend_diarization_label_preview(self, srt_path_value: object) -> dict[str, object]:
        """Parse an SRT into the segment/transcript preview the training-label comparison panel shows."""

        if not isinstance(srt_path_value, Path) or not srt_path_value.is_file():
            return {}

        try:
            cues = parse_srt(srt_path_value)
        except (OSError, UnicodeDecodeError, ValueError):
            return {}

        def seconds_text(ms_value: int) -> str:
            formatted = f"{max(ms_value, 0) / 1000:.3f}".rstrip("0").rstrip(".")
            return formatted or "0"

        segment_lines: list[str] = []
        transcript_lines: list[str] = []
        speakers: set[str] = set()
        preview_cues = cues[:DIARIZATION_LABEL_PREVIEW_LIMIT]
        for cue in preview_cues:
            speaker = self.rttm_safe_token(cue.speaker, fallback=f"SPEAKER_{cue.index:02d}")
            speakers.add(speaker)
            segment_lines.append(f"{seconds_text(cue.start_ms)} {seconds_text(cue.end_ms)} {speaker}")
            transcript_text = cue.spoken_text or cue.text
            if transcript_text:
                transcript_lines.append(f"{speaker}: {transcript_text}")

        duration_seconds = round(max((cue.end_ms for cue in cues), default=0) / 1000, 3)
        return {
            "labelSegments": "\n".join(segment_lines),
            "transcriptPreview": "\n".join(transcript_lines),
            "segmentCount": len(cues),
            "previewSegmentCount": len(preview_cues),
            "speakerCount": len(speakers),
            "durationSeconds": duration_seconds,
            "previewTruncated": len(cues) > len(preview_cues),
        }

    def frontend_diarization_history_rows(
        self,
        rows: list[dict[str, object]],
        script_name: str,
    ) -> list[dict[str, object]]:
        """Enrich per-file diarization history rows with resolved artifact hrefs and label previews."""

        serialized_rows: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            audio_file = str(row.get("audio_file") or "")
            source_audio = self.audio_dir / audio_file if audio_file else None
            audio_href = (
                self.file_link(source_audio, script_name)
                if isinstance(source_audio, Path) and source_audio.is_file()
                else ""
            )
            links = [
                link
                for link in [
                    self.frontend_artifact_link("Source Audio", source_audio, script_name),
                    self.frontend_artifact_link("Transcript", row.get("transcript_path"), script_name),
                    self.frontend_artifact_link("Diarized Times", row.get("srt_path"), script_name),
                    self.frontend_artifact_link("Review Page", row.get("review_path"), script_name),
                    self.frontend_artifact_link("Review Flags", row.get("flags_path"), script_name),
                    self.frontend_artifact_link("Summary", row.get("summary_path"), script_name),
                ]
                if link
            ]
            label_preview = self.frontend_diarization_label_preview(row.get("srt_path"))
            serialized_rows.append(
                {
                    "audioFile": audio_file,
                    "fileName": Path(audio_file).name if audio_file else "",
                    "folder": audio_file.split("/", 1)[0] if "/" in audio_file else "Unsorted Root",
                    "audioHref": audio_href,
                    "status": row.get("status", "unknown"),
                    "runName": row.get("run_name", ""),
                    "backend": row.get("backend", ""),
                    "backendLabel": row.get("backend_label") or self.diarization_backend_label(str(row.get("backend", ""))),
                    "modelKey": row.get("model_key", row.get("backend", "")),
                    "modelLabel": row.get("model_label") or row.get("backend_label") or self.diarization_backend_label(str(row.get("backend", ""))),
                    "lastRun": row.get("last_run", ""),
                    "runtimeSeconds": row.get("runtime_seconds", ""),
                    "errorSummary": row.get("error_summary", ""),
                    "batchId": str(row.get("batch_id") or ""),
                    "links": links,
                    "srtPath": (
                        self.describe_path(row["srt_path"])
                        if isinstance(row.get("srt_path"), Path)
                        else ""
                    ),
                    **label_preview,
                }
            )
        return serialized_rows

    def frontend_diarization_runs_for_compare(self) -> list[dict[str, object]]:
        """Flatten the diarization-history lookup into one row per run for the DER calculator's dropdowns.

        Surfaces base models AND fine-tuned checkpoints — every run that landed
        an SRT on disk shows up so the user can pair any two together. We tag
        each row with ``modelKind`` ("default" / "fine_tuned" / "unknown") so
        the UI can show a badge without re-deriving it from the model key.
        """

        option_lookup = self.diarization_model_option_lookup()
        runs: dict[str, dict[str, object]] = {}
        lookup = self.diarization_model_history_lookup()
        for audio_name, per_model in lookup.items():
            if not isinstance(per_model, dict):
                continue
            for record in per_model.values():
                if not isinstance(record, dict):
                    continue
                run_dir = record.get("run_dir")
                srt_path = record.get("srt_path")
                if not isinstance(run_dir, Path):
                    continue
                # An SRT-on-disk check keeps half-finished or failed runs from
                # showing up in the dropdown; you can only score what actually
                # made it to disk.
                if not (isinstance(srt_path, Path) and srt_path.is_file()):
                    continue
                key = str(run_dir)
                model_key = str(record.get("model_key") or record.get("backend") or "")
                option = option_lookup.get(model_key) if model_key else None
                model_kind = str(option.get("kind") if isinstance(option, dict) else "") or "unknown"
                bucket = runs.setdefault(
                    key,
                    {
                        "path": self.describe_path(run_dir),
                        "name": run_dir.name,
                        "backend": str(record.get("backend", "")),
                        "backendLabel": str(record.get("backend_label") or self.diarization_backend_label(str(record.get("backend", "")))),
                        "modelKey": model_key,
                        "modelLabel": str(record.get("model_label") or record.get("backend_label") or ""),
                        "modelKind": model_kind,
                        "lastRun": str(record.get("last_run", "")),
                        "audioFiles": set(),
                        "_sortKey": float(record.get("sort_key") or 0.0),
                    },
                )
                bucket["audioFiles"].add(audio_name)
                bucket["_sortKey"] = max(float(bucket["_sortKey"]), float(record.get("sort_key") or 0.0))

        serialized = []
        for entry in runs.values():
            audio_files = sorted(entry.pop("audioFiles"))
            sort_key = entry.pop("_sortKey")
            entry["audioFiles"] = audio_files
            entry["fileCount"] = len(audio_files)
            entry["_sortKey"] = sort_key
            serialized.append(entry)
        serialized.sort(key=lambda row: row["_sortKey"], reverse=True)
        for entry in serialized:
            entry.pop("_sortKey", None)
        return serialized

    def frontend_diarization_model_options(
        self,
        rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Normalize model-option rows into the flat shape the model-picker dropdown expects."""

        serialized: list[dict[str, str]] = []
        for row in rows:
            key = str(row.get("key", "")).strip()
            if not key:
                continue
            serialized.append(
                {
                    "key": key,
                    "backend": str(row.get("backend", "")),
                    "label": str(row.get("label", "") or key),
                    "shortLabel": str(row.get("short_label", "") or row.get("label", "") or key),
                    "kind": str(row.get("kind", "") or "default"),
                    "description": str(row.get("description", "") or ""),
                }
            )
        return serialized

    def frontend_projects(self, projects: list[dict[str, object]], script_name: str) -> list[dict[str, object]]:
        """Build the fine-tuning project card payloads — includes step completion state, recent runs, and artifact links."""

        serialized: list[dict[str, object]] = []
        for project in projects:
            latest_run = project.get("latest_run") or {}
            metadata = project.get("metadata") or {}
            metrics = self.fine_tuning_project_metrics(project)
            sample_count = int(project.get("sample_count", 0))
            completed_steps = 1 if sample_count > 0 else 0
            if project.get("prepared"):
                completed_steps += 1
            if latest_run:
                completed_steps += 1

            project_path = Path(str(project.get("path", "")))
            links: list[dict[str, str]] = []
            for label, candidate in [
                ("metadata.json", project_path / "artifacts" / "metadata.json"),
                ("launch script", next(iter(sorted((project_path / "artifacts").glob("launch_*_finetune.sh"))), None)),
                ("slurm script", next(iter(sorted((project_path / "artifacts").glob("launch_*_finetune.sbatch"))), None)),
                ("training script", next(iter(sorted((project_path / "artifacts").glob("train_*.py"))), None)),
            ]:
                link = self.frontend_artifact_link(label, candidate, script_name)
                if link:
                    links.append(link)
            if latest_run:
                for label, candidate in [
                    ("stdout.log", Path(str(latest_run.get("stdout_path", "")))),
                    ("stderr.log", Path(str(latest_run.get("stderr_path", "")))),
                ]:
                    link = self.frontend_artifact_link(label, candidate, script_name)
                    if link:
                        links.append(link)

            latest_run_text = "none"
            if latest_run:
                latest_run_text = (
                    f"{latest_run.get('status', 'unknown')} "
                    f"{latest_run.get('version_name', Path(str(latest_run.get('run_dir', ''))).name)} "
                    f"({Path(str(latest_run.get('run_dir', ''))).name})"
                )
            recent_runs = []
            project_backend = str(project.get("backend", "")).lower()
            for run in (project.get("recent_runs") or [])[:5]:
                if not isinstance(run, dict):
                    continue
                version_name = str(run.get("version_name") or Path(str(run.get("run_dir", ""))).name)
                experiment_dir = Path(str(run.get("experiment_dir") or ""))
                model_available = bool(
                    experiment_dir.is_dir()
                    and not bool(run.get("model_deleted"))
                )
                # display_name was set by list_runs; fall back to version_name
                # so a renamed run shows the friendly label here too.
                # Pyannote runs get their .out scanned for val_loss points so
                # the project card can show the validation curve. NeMo logs
                # val_loss elsewhere, so we leave it empty for that backend.
                val_loss_series: list[dict[str, object]] = []
                val_loss_log_path = ""
                if project_backend == "pyannote":
                    out_log = find_pyannote_run_out_log(run, project_path)
                    if out_log is not None:
                        val_loss_series = parse_pyannote_val_loss(out_log)
                        val_loss_log_path = str(out_log)
                recent_runs.append(
                    {
                        "status": str(run.get("status", "unknown")),
                        "versionName": version_name,
                        "displayName": str(run.get("display_name") or "").strip() or version_name,
                        "versionNumber": run.get("version_number", ""),
                        "runName": Path(str(run.get("run_dir", ""))).name,
                        "runDir": str(run.get("run_dir", "")),
                        "experimentDir": str(run.get("experiment_dir") or ""),
                        "modelAvailable": model_available,
                        "modelDeleted": bool(run.get("model_deleted")),
                        "startedAt": str(run.get("started_at_utc") or ""),
                        # Surface the snapshotted base/pretrained model so the user
                        # can tell at a glance which checkpoint each fine-tuned run
                        # was built on top of. Empty for legacy runs that predate
                        # the metadata snapshot — UI just hides the row.
                        "baseModel": str(run.get("base_model") or "").strip(),
                        "baseModelKind": str(run.get("base_model_kind") or "").strip(),
                        # val_loss points parsed live from the pyannote .out file;
                        # empty for nemo or for runs where validation hasn't
                        # produced any val_loss lines yet.
                        "valLoss": val_loss_series,
                        "valLossLogPath": val_loss_log_path,
                    }
                )
            serialized.append(
                {
                    "backend": str(project.get("backend", "")),
                    "slug": str(project.get("slug", "")),
                    # Friendly display name from display.json sidecar; falls back
                    # to the slug so legacy projects without a rename still show
                    # something readable.
                    "displayName": str(project.get("display_name") or "").strip() or str(project.get("slug", "")),
                    "autoTrain": bool(project.get("auto_train")),
                    "autoTrainPending": bool(project.get("auto_train_pending")),
                    "sampleCount": sample_count,
                    "prepared": bool(project.get("prepared")),
                    "latestRunText": latest_run_text,
                    "recentRuns": recent_runs,
                    "completedSteps": completed_steps,
                    "metrics": {
                        "sampleCount": metrics.get("sample_count", sample_count),
                        "samplesWithRttm": metrics.get("samples_with_rttm", 0),
                        "samplesMissingRttm": metrics.get("samples_missing_rttm", 0),
                        "totalAudioSeconds": metrics.get("total_audio_seconds", 0.0),
                        "totalSpeechSeconds": metrics.get("total_speech_seconds", 0.0),
                        "totalActiveSpeechSeconds": metrics.get("total_active_speech_seconds", 0.0),
                        "totalOverlapSeconds": metrics.get("total_overlap_seconds", 0.0),
                        "totalNonSpeechSeconds": metrics.get("total_non_speech_seconds", 0.0),
                        "totalSegments": metrics.get("total_segments", 0),
                        "uniqueSpeakerLabels": metrics.get("unique_speaker_labels", 0),
                        "maxSpeakersPerSample": metrics.get("max_speakers_per_sample", 0),
                        "maxConcurrentSpeakers": metrics.get("max_concurrent_speakers", 0),
                        "averageSpeakersPerSample": metrics.get("average_speakers_per_sample", 0.0),
                        "averageSegmentsPerSample": metrics.get("average_segments_per_sample", 0.0),
                        "averageSegmentSeconds": metrics.get("average_segment_seconds", 0.0),
                        "speechCoverage": metrics.get("speech_coverage", 0.0),
                        "overlapCoverage": metrics.get("overlap_coverage", 0.0),
                        "speakerTurnsPerMinute": metrics.get("speaker_turns_per_minute", 0.0),
                        "dominantSpeakerShare": metrics.get("dominant_speaker_share", 0.0),
                        "trainCount": metrics.get("train_count", 0),
                        "validationCount": metrics.get("validation_count", 0),
                        "actualTrainRatio": metrics.get("actual_train_ratio", 0.0),
                    },
                    "warnings": [str(item) for item in (metadata.get("warnings") or [])[:4]],
                    "links": links,
                }
            )
        return serialized

    def refresh_training_label_review_html(
        self,
        review_path: Path,
        review_record: dict[str, object] | None,
    ) -> None:
        """Regenerate a saved training-label review page when its embedded bundle is stale."""

        if not isinstance(review_path, Path) or not review_path.is_file():
            return
        try:
            needs_refresh = self._review_html_is_missing_media(review_path)
        except Exception:
            return
        if not needs_refresh:
            return

        srt_candidates: list[Path] = []
        if isinstance(review_record, dict) and isinstance(review_record.get("srt_path"), Path):
            srt_candidates.append(review_record["srt_path"])
        if review_path.name.endswith("_review.html"):
            srt_candidates.append(review_path.with_name(f"{review_path.name[:-len('_review.html')]}.srt"))
        srt_candidates.append(review_path.with_suffix(".srt"))

        for srt_path in srt_candidates:
            if not isinstance(srt_path, Path) or not srt_path.is_file():
                continue
            flags_path = review_record.get("flags_path") if isinstance(review_record, dict) else None
            if not isinstance(flags_path, Path):
                flags_path = srt_path.with_name(f"{srt_path.stem}_review_flags.tsv")
            try:
                write_review_bundle(
                    srt_path=srt_path,
                    output_html=review_path,
                    report_tsv=flags_path,
                    audio_dir=self.audio_dir,
                    training_label_records=self.load_training_label_records(),
                    model_comparisons=self.review_model_comparisons_for_srt(srt_path),
                    fine_tuning_projects=list_projects(root=self.root),
                    quiet=True,
                )
            except Exception:
                continue
            return

    def frontend_training_label_rows(
        self,
        audio_paths: list[Path],
        records: dict[str, dict[str, object]],
        script_name: str,
    ) -> list[dict[str, object]]:
        """Combine audio-file metadata with label_status records so the training-labels page shows everything in one row.

        Cached for 10 s — same reason as ``training_label_summary``: the
        per-audio rttm_lookup does ~5 stat() calls per file on a networked
        filesystem and dominated cold-cache page renders. Mutations
        (label saves, completions, deletes) call ``invalidate_dashboard_cache``.
        """

        if not audio_paths:
            return []
        return self.cached_value(
            f"frontend_training_label_rows::{script_name}",
            ttl_seconds=60.0,
            builder=lambda: self._build_frontend_training_label_rows(audio_paths, records, script_name),
        )

    def _build_frontend_training_label_rows(
        self,
        audio_paths: list[Path],
        records: dict[str, dict[str, object]],
        script_name: str,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        diarization_records = self.diarization_model_history_lookup()
        # Same single-walk index trick as the summary path. Per-audio check
        # is now an O(1) set membership instead of a fistful of stat()s.
        rttm_index_lookup = getattr(self, "synthetic_rttm_index", None)
        rttm_index = rttm_index_lookup() if callable(rttm_index_lookup) else None
        synthetic_lookup = getattr(self, "synthetic_rttm_for_audio_via_index", None)
        for index, path in enumerate(audio_paths, start=1):
            audio_name = self.audio_relative_path(path)
            record = records.get(audio_name) or records.get(path.name) or {}
            if not record and rttm_index is not None and callable(synthetic_lookup):
                rttm_path = synthetic_lookup(path, rttm_index)
                if isinstance(rttm_path, Path):
                    source = "audio_stitching" if "/audioStitching/" in f"/{audio_name}" else "rttm_pair"
                    record = {
                        "status": "completed",
                        "source": source,
                        "backend": "both",
                        "target_backends": ["nemo", "pyannote"],
                        "target_projects": [],
                        "training_projects": [],
                        "training_usage": [],
                        "training_audio_path": self.describe_path(path),
                        "training_rttm_path": self.describe_path(rttm_path),
                        "project_name": "",
                    }
            review_record = self.preferred_review_record(diarization_records.get(audio_name, {}))
            status = self.training_label_status(record)
            review_path_value = str(record.get("review_path") or "")
            review_href = ""
            if review_path_value:
                try:
                    review_path = self.resolve_under_root(review_path_value)
                except ValueError:
                    review_path = None
                if review_path and review_path.is_file():
                    self.refresh_training_label_review_html(review_path, review_record if isinstance(review_record, dict) else None)
                    review_href = self.file_link(review_path, script_name)
            if not review_href:
                review_path = review_record.get("review_path") if isinstance(review_record, dict) else None
                if isinstance(review_path, Path) and review_path.is_file():
                    review_path_value = self.describe_path(review_path)
                    review_href = self.file_link(review_path, script_name)
            system_questions = [
                str(item)
                for item in (record.get("system_questions") or [])
                if str(item).strip()
            ]
            issue_questions = str(record.get("issue_questions") or "")
            backend = str(record.get("backend") or "")
            project_name = str(record.get("project_name") or DEFAULT_TRAINING_LABEL_PROJECT)
            target_backends = [
                str(item)
                for item in (record.get("target_backends") or [])
                if str(item).strip()
            ]
            if not target_backends:
                target_backends = self.training_label_target_backends(backend or "both")
            backend_label = (
                "NeMo + pyannote"
                if len(target_backends) > 1
                else self.diarization_backend_label(target_backends[0])
            )
            completed_training_projects = [
                str(item)
                for item in (record.get("training_projects") or [])
                if str(item).strip()
            ]
            recorded_target_projects = [
                str(item)
                for item in (record.get("target_projects") or [])
                if str(item).strip()
            ]
            target_projects = (
                recorded_target_projects
                if recorded_target_projects
                else [f"{target_backend}/{project_name}" for target_backend in target_backends] if project_name else []
            )
            queued_training_projects = [
                str(item)
                for item in (record.get("queued_training_projects") or [])
                if str(item).strip()
            ]
            training_usage = [
                item
                for item in (record.get("training_usage") or [])
                if isinstance(item, dict)
            ]
            source = str(record.get("source") or "")
            if status == "completed" and not completed_training_projects and source not in {"audio_stitching", "rttm_pair"}:
                completed_training_projects = target_projects
            detail = "Ready to label for training."
            if status == "draft":
                detail = "Saved for later."
            elif status == "needs_review":
                detail = system_questions[0] if system_questions else (issue_questions or "Needs an answer before training.")
            elif status == "completed":
                detail = (
                    f"Training sample available in {', '.join(completed_training_projects)}."
                    if completed_training_projects
                    else "RTTM pair is ready for fine-tuning."
                )
                if queued_training_projects:
                    detail += f" Auto-train requested for {', '.join(queued_training_projects)}."
            rows.append(
                {
                    "index": index,
                    "name": audio_name,
                    "fileName": path.name,
                    "folder": audio_name.split("/", 1)[0] if "/" in audio_name else "Unsorted Root",
                    "type": path.suffix.lower() or "file",
                    "audioHref": self.file_link(path, script_name) if path.is_file() else "",
                    "status": status,
                    "backend": backend,
                    "backendLabel": backend_label,
                    "targetBackends": target_backends,
                    "projectName": project_name,
                    "targetProjects": target_projects,
                    "trainingProjects": completed_training_projects,
                    "queuedTrainingProjects": queued_training_projects,
                    "trainingUsage": training_usage,
                    "labelSegments": str(record.get("label_segments") or ""),
                    "transcriptText": str(record.get("transcript_text") or ""),
                    "issueQuestions": issue_questions,
                    "systemQuestions": system_questions,
                    "source": str(record.get("source") or ""),
                    "reviewPath": review_path_value,
                    "reviewHref": review_href,
                    "completedAt": str(record.get("completed_at_utc") or ""),
                    "updatedAt": str(record.get("updated_at_utc") or ""),
                    "speakerCount": record.get("speaker_count", ""),
                    "segmentCount": record.get("segment_count", ""),
                    "trainingAudioPath": str(record.get("training_audio_path") or ""),
                    "trainingRttmPath": str(record.get("training_rttm_path") or ""),
                    "detail": detail,
                }
            )
        return rows

    def frontend_state(
        self,
        *,
        context: dict[str, object],
        current_path: str,
        script_name: str,
        message: str,
        message_status: str,
    ) -> dict[str, object]:
        """Assemble the full window.__STATE__ blob — everything React needs to render the current page."""

        projects = self.frontend_projects(list(context.get("projects", [])), script_name)
        recent_outputs = [
            self.frontend_file_record(path, script_name)
            for path in list(context.get("recent_outputs", []))
        ]
        recent_srts = [
            self.frontend_file_record(path, script_name)
            for path in list(context.get("recent_srts", []))
        ]
        raw_audio_paths = list(context.get("audio_files", []))
        raw_training_label_records = context.get("training_label_records") or {}
        if not isinstance(raw_training_label_records, dict):
            raw_training_label_records = {}
        training_label_records = {
            str(key): value
            for key, value in raw_training_label_records.items()
            if isinstance(value, dict)
        }
        training_label_rows = self.frontend_training_label_rows(
            [path for path in raw_audio_paths if isinstance(path, Path)],
            training_label_records,
            script_name,
        )
        audio_files = [
            {
                "name": self.audio_relative_path(path) if isinstance(path, Path) else str(path),
                "fileName": path.name,
                "folder": (
                    self.audio_relative_path(path).split("/", 1)[0]
                    if isinstance(path, Path) and "/" in self.audio_relative_path(path)
                    else "Unsorted Root"
                ),
                "path": self.describe_path(path) if isinstance(path, Path) else str(path),
                "type": path.suffix.lower() or "file",
            }
            for path in raw_audio_paths
            if isinstance(path, Path)
        ]
        raw_training_sources = context.get("training_source_files") or {}
        if not isinstance(raw_training_sources, dict):
            raw_training_sources = {}
        raw_training_audio_paths = list(raw_training_sources.get("audio_files", []))
        # No fallback to the full audio_in/ inventory: if validation returned no
        # paired audio, the picker stays empty so the audio and RTTM lists can
        # never drift apart. The earlier fallback caused a mismatch where the
        # audio side showed every file in audio_in/ while the RTTM side was 0.
        training_audio_paths = [
            path for path in raw_training_audio_paths if isinstance(path, Path)
        ]
        raw_rttm_audio_map = raw_training_sources.get("rttm_audio_map") or {}
        if not isinstance(raw_rttm_audio_map, dict):
            raw_rttm_audio_map = {}

        def training_rttm_record(path: Path) -> dict[str, str]:
            record = self.frontend_file_record(path, script_name)
            audio_match = raw_rttm_audio_map.get(str(path.resolve()))
            if isinstance(audio_match, Path):
                try:
                    record["audioFile"] = f"audio_in/{self.audio_relative_path(audio_match)}"
                    record["name"] = f"{audio_match.stem}.rttm"
                except ValueError:
                    pass
            return record

        training_sources = {
            "audioFiles": [
                self.frontend_file_record(path, script_name)
                for path in training_audio_paths
                if isinstance(path, Path)
            ],
            "rttmFiles": [
                training_rttm_record(path)
                for path in list(raw_training_sources.get("rttm_files", []))
                if isinstance(path, Path)
            ],
            "transcriptFiles": [
                self.frontend_file_record(path, script_name)
                for path in list(raw_training_sources.get("transcript_files", []))
                if isinstance(path, Path)
            ],
        }
        preferences = context.get("preferences") or self.model_preferences()
        stitched_rows = self.stitched_run_rows(limit=30, script_name=script_name) if current_path == "/stitching" else []
        latest = {
            key: (
                {
                    "name": value.name,
                    "path": self.describe_path(value),
                }
                if isinstance(value, Path)
                else None
            )
            for key, value in {
                "latestSiteDiarization": context.get("latest_site_diarization"),
                "latestSiteYoutube": context.get("latest_site_youtube"),
                "latestSingle": context.get("latest_single"),
                "latestBulk": context.get("latest_bulk"),
            }.items()
        }
        live_tracking, _, _, _ = self.live_tracking_details()
        return {
            "currentPath": current_path,
            "message": message,
            "messageStatus": message_status,
            "navItems": [
                {
                    "path": path,
                    "href": self.frontend_route(path, script_name),
                }
                for path in NAV_PATHS
            ],
            "routes": {
                "uploadAudio": self.frontend_route("/upload/audio", script_name),
                "createAudioFolder": self.frontend_route("/audio-folders/create", script_name),
                "renameAudioFolder": self.frontend_route("/audio-folders/rename", script_name),
                "deleteAudioFolder": self.frontend_route("/audio-folders/delete", script_name),
                "deleteAudioFile": self.frontend_route("/audio-files/delete", script_name),
                "bulkDeleteAudioFiles": self.frontend_route("/audio-files/bulk-delete", script_name),
                "moveAudioFile": self.frontend_route("/audio-files/move", script_name),
                "saveTrainingLabel": self.frontend_route("/training-labels/save", script_name),
                "uncompleteTrainingLabel": self.frontend_route("/training-labels/uncomplete", script_name),
                "youtubeLinks": self.frontend_route("/youtube-links", script_name),
                "deleteYoutubeLink": self.frontend_route("/youtube-links/delete", script_name),
                "convertYoutube": self.frontend_route("/actions/convert-youtube", script_name),
                "resetYoutube": self.frontend_route("/actions/reset-youtube-workspace", script_name),
                "stitchAudio": self.frontend_route("/actions/stitch-audio", script_name),
                "renameStitching": self.frontend_route("/stitching/rename", script_name),
                "deleteStitching": self.frontend_route("/stitching/delete", script_name),
                "review": self.frontend_route("/actions/review", script_name),
                "runDiarization": self.frontend_route("/actions/run-diarization", script_name),
                "tests": self.frontend_route("/actions/test", script_name),
                "saveModels": self.frontend_route("/models/save", script_name),
                "tracking": self.frontend_route("/api/tracking", script_name),
                "pageState": self.frontend_route("/api/page-state", script_name),
                "clusterQueue": self.frontend_route("/api/cluster-queue", script_name),
                "runtimeEstimate": self.frontend_route("/api/runtime-estimate", script_name),
                "fileSearch": self.frontend_route("/api/file-search", script_name),
                "fineTuneCreateProject": self.frontend_route("/fine-tuning/create-project", script_name),
                "fineTuneUpload": self.frontend_route("/fine-tuning/upload-sample", script_name),
                "fineTunePrepare": self.frontend_route("/fine-tuning/prepare", script_name),
                "fineTunePrepareLaunch": self.frontend_route("/fine-tuning/prepare-launch", script_name),
                "fineTuneLaunch": self.frontend_route("/fine-tuning/launch", script_name),
                "fineTuneRenameProject": self.frontend_route("/fine-tuning/rename-project", script_name),
                "fineTuneRenameRun": self.frontend_route("/fine-tuning/rename-run", script_name),
                "fineTuneDeleteRunModel": self.frontend_route("/fine-tuning/delete-run-model", script_name),
                "fineTuneScoreRun": self.frontend_route("/api/fine-tuning/score-run", script_name),
                "fineTuneCompareRuns": self.frontend_route("/api/fine-tuning/compare-runs", script_name),
                "fineTuneAutoTrain": self.frontend_route("/fine-tuning/auto-train", script_name),
            },
            "assets": {
                "styles": self.frontend_asset("static/css/dashboard.css", script_name),
                "react": self.frontend_asset("vendor/react.production.min.js", script_name),
                "reactDom": self.frontend_asset("vendor/react-dom.production.min.js", script_name),
                "app": self.frontend_asset("static/js/dashboard_app.js", script_name),
            },
            "defaults": {
                "uploadAudioAccept": UPLOAD_AUDIO_ACCEPT,
                "uploadAudioSuffixes": list(UPLOAD_AUDIO_SUFFIXES),
                "defaultUploadAudioFolder": DEFAULT_UPLOAD_AUDIO_FOLDER,
                "defaultYoutubeAudioFolder": DEFAULT_YOUTUBE_AUDIO_FOLDER,
                "rootAudioFolderValue": ROOT_AUDIO_FOLDER_VALUE,
                "diarizationBackends": [
                    {
                        "value": backend,
                        "label": self.diarization_backend_label(backend),
                    }
                    for backend in DIARIZATION_BACKENDS
                ],
                "pythonBin": sys.executable,
                "trainRatio": DEFAULT_TRAIN_RATIO,
                "baseWindow": DEFAULT_BASE_WINDOW,
                "baseShift": DEFAULT_BASE_SHIFT,
                "stepCount": DEFAULT_STEP_COUNT,
                "configName": DEFAULT_CONFIG_NAME,
                "speakerModel": DEFAULT_SPEAKER_MODEL,
                "pyannotePretrainedModel": DEFAULT_PYANNOTE_PRETRAINED_MODEL,
                "maxEpochs": DEFAULT_MAX_EPOCHS,
                "slurmPartition": DEFAULT_SLURM_PARTITION,
                "slurmTime": DEFAULT_SLURM_TIME,
                "slurmMemory": DEFAULT_SLURM_MEMORY,
                "slurmCpus": DEFAULT_SLURM_CPUS,
                "slurmGpus": DEFAULT_SLURM_GPUS,
            },
            "context": {
                "audioCount": context["audio_count"],
                "queueCount": context["queue_count"],
                "projectCount": context["project_count"],
                "failedLinksPresent": bool(context["failed_links_present"]),
                "latest": latest,
                "audioFiles": audio_files,
                "audioFolders": context.get("audio_folders", self.audio_folder_rows()),
                "trainingSources": training_sources,
                "recentOutputs": recent_outputs,
                "recentSrts": recent_srts,
                "preferences": preferences,
                "projects": projects,
                "fineTuningSummary": context.get("fine_tuning_summary", self.fine_tuning_summary(list(context.get("projects", [])))),
                "tracking": live_tracking,
                "trainingLabels": {
                    # The full row list is only rendered by /training-labels.
                    # Other pages (especially /fine-tuning) just need the
                    # summary + defaults for headers and dropdowns, so we
                    # skip the multi-MB row payload elsewhere.
                    "rows": training_label_rows if current_path == "/training-labels" else [],
                    "summary": context.get("training_label_summary", self.training_label_summary([], {})),
                    "defaultProjectName": DEFAULT_TRAINING_LABEL_PROJECT,
                },
                "stitching": {
                    "rows": stitched_rows,
                    "summary": context.get("stitched_summary", self.stitched_summary()),
                },
                "youtube": {
                    "queueRows": context.get("youtube_queue_rows", []),
                    "queueSummary": context.get("youtube_queue_summary", {}),
                    "history": self.frontend_youtube_history_rows(list(context.get("youtube_history", [])), script_name),
                    "retryRows": context.get("youtube_retry_rows", []),
                    "noDataRows": context.get("youtube_no_data_rows", []),
                    "latestRun": self.frontend_youtube_run(context.get("youtube_latest_run"), script_name),
                    "activeRuns": [
                        run for run in (
                            self.frontend_youtube_run(detail, script_name)
                            for detail in context.get("youtube_active_runs", [])
                        )
                        if run is not None
                    ],
                },
                "diarization": {
                    "audioRows": context.get("diarization_audio_rows", []),
                    "librarySummary": context.get("diarization_library_summary", {}),
                    "modelOptions": self.frontend_diarization_model_options(list(context.get("diarization_model_options", []))),
                    "selectedModelKey": context.get("selected_diarization_model_key", ""),
                    "history": self.frontend_diarization_history_rows(list(context.get("diarization_history", [])), script_name),
                    "latestRun": self.frontend_diarization_run(context.get("diarization_latest_run"), script_name),
                    "activeRuns": [
                        run for run in (
                            self.frontend_diarization_run(detail, script_name)
                            for detail in context.get("diarization_active_runs", [])
                        )
                        if run is not None
                    ],
                    "runsForCompare": self.frontend_diarization_runs_for_compare() if current_path == "/fine-tuning" else [],
                    "evaluableFiles": [
                        {
                            "name": row.get("name", ""),
                            "fileName": row.get("fileName", ""),
                            "referenceRttm": str(row.get("trainingRttmPath") or ""),
                        }
                        for row in training_label_rows
                        if current_path == "/fine-tuning"
                        and isinstance(row, dict)
                        and row.get("status") == "completed"
                        # Read the RTTM path off the row itself rather than
                        # the raw label_status records. The row builder
                        # already merges synthetic-completion paths
                        # (audioStitching sidecars, raw RTTM pairs, etc.) and
                        # handles the audio_in/<file>.wav vs renumbered
                        # 001_<file>.wav mismatch where the persistent record
                        # was keyed before normalize_audio_dir ran. Using the
                        # row makes every "completed" entry — including
                        # stitched conversations — eligible for DER.
                        and str(row.get("trainingRttmPath") or "")
                    ],
                },
            },
        }

    def render_page(
        self,
        *,
        message: str,
        message_status: str,
        script_name: str,
        current_path: str,
        diarization_model_key: str = "",
    ) -> str:
        """Produce the final HTML response — builds page context, serializes state, and injects it into the HTML shell."""

        context = self.page_context(
            current_path,
            diarization_model_key=diarization_model_key,
        )
        state = self.frontend_state(
            context=context,
            current_path=current_path,
            script_name=script_name,
            message=message,
            message_status=message_status,
        )
        state_json = (
            json.dumps(state, ensure_ascii=False)
            .replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )
        assets = state["assets"]
        try:
            template = self.dashboard_template_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise RuntimeError(f"Dashboard frontend template is missing: {self.dashboard_template_path}") from exc

        replacements = {
            "DASHBOARD_TITLE": html.escape("ML Speech Diarization"),
            "DASHBOARD_STYLES": html.escape(str(assets["styles"])),
            "DASHBOARD_STATE": state_json,
            "REACT_RUNTIME": html.escape(str(assets["react"])),
            "REACT_DOM_RUNTIME": html.escape(str(assets["reactDom"])),
            "DASHBOARD_APP": html.escape(str(assets["app"])),
        }
        for token, value in replacements.items():
            template = template.replace(f"{{{{{token}}}}}", value)
        return template
