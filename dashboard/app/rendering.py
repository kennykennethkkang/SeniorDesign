#!/usr/bin/env python3
"""Page context construction and final HTML rendering."""
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


class RenderingMixin:
    """Page context construction and final HTML rendering."""

    def shared_page_context(self) -> dict[str, object]:
        """Collect the lightweight context shared across all dashboard pages."""

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
        """Collect only the context needed by the requested page."""

        effective_path = current_path if current_path in PAGE_PATHS else "/"
        context = self.shared_page_context()
        if effective_path in AUDIO_INVENTORY_PAGES:
            audio_files = self.audio_inventory()
            context["audio_files"] = audio_files
            context["audio_folders"] = self.audio_folder_rows(audio_files=audio_files)
        if effective_path == "/youtube" and "audio_folders" not in context:
            # /youtube is not an AUDIO_INVENTORY_PAGES member, so reuse here is
            # only available when audio_inventory was loaded above. The folder
            # dropdown is small enough that a separate walk is acceptable.
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
            context["diarization_history"] = self.diarization_history_rows(limit=100)
            context["diarization_latest_run"] = self.diarization_run_details(context["latest_site_diarization"])
            context["diarization_active_runs"] = self.active_diarization_runs_detailed()
        if effective_path in {"/training-labels", "/fine-tuning"}:
            context["diarization_model_options"] = self.diarization_model_options()
        if effective_path in TRAINING_LABEL_CONTEXT_PAGES:
            label_records = self.load_training_label_records()
            audio_files = list(context.get("audio_files", []))
            context["training_label_records"] = label_records
            context["training_label_summary"] = self.training_label_summary(audio_files, label_records)
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
        """Return an unescaped route for JSON state consumed by React."""

        return self.with_prefix(path, script_name)

    def frontend_asset(self, relative_path: str, script_name: str) -> str:
        """Return a URL for a static frontend asset."""

        return self.with_prefix("/assets/" + quote(relative_path, safe="/"), script_name)

    def frontend_file_record(self, path: Path, script_name: str) -> dict[str, str]:
        """Serialize one workspace file for React tables."""

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
        """Serialize one optional artifact link."""

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
        """Serialize YouTube history rows and attach audio-file links when possible."""

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
        """Serialize the latest YouTube run for the React details dialog."""

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
        """Serialize the latest diarization run for React."""

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
        """Convert one SRT into editable label-preview text for comparison views."""

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
        """Serialize per-file diarization history rows with useful artifact links."""

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

    def frontend_diarization_model_options(
        self,
        rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Serialize selectable diarization model profiles for the React frontend."""

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
        """Serialize fine-tuning project cards and their direct artifact links."""

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
            for run in (project.get("recent_runs") or [])[:5]:
                if not isinstance(run, dict):
                    continue
                version_name = str(run.get("version_name") or Path(str(run.get("run_dir", ""))).name)
                # display_name was set by list_runs; fall back to version_name
                # so a renamed run shows the friendly label here too.
                recent_runs.append(
                    {
                        "status": str(run.get("status", "unknown")),
                        "versionName": version_name,
                        "displayName": str(run.get("display_name") or "").strip() or version_name,
                        "versionNumber": run.get("version_number", ""),
                        "runName": Path(str(run.get("run_dir", ""))).name,
                        "runDir": str(run.get("run_dir", "")),
                        "startedAt": str(run.get("started_at_utc") or ""),
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

    def frontend_training_label_rows(
        self,
        audio_paths: list[Path],
        records: dict[str, dict[str, object]],
        script_name: str,
    ) -> list[dict[str, object]]:
        """Serialize uploaded audio files with their current training-label state."""

        rows: list[dict[str, object]] = []
        diarization_records = self.diarization_model_history_lookup()
        for index, path in enumerate(audio_paths, start=1):
            audio_name = self.audio_relative_path(path)
            record = records.get(audio_name) or records.get(path.name) or {}
            status = self.training_label_status(record)
            review_path_value = str(record.get("review_path") or "")
            review_href = ""
            if review_path_value:
                try:
                    review_path = self.resolve_under_root(review_path_value)
                except ValueError:
                    review_path = None
                if review_path and review_path.is_file():
                    review_href = self.file_link(review_path, script_name)
            if not review_href:
                review_record = self.preferred_review_record(diarization_records.get(audio_name, {}))
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
            target_projects = [f"{target_backend}/{project_name}" for target_backend in target_backends] if project_name else []
            if status == "completed" and not completed_training_projects:
                completed_training_projects = target_projects
            detail = "Ready to label for training."
            if status == "draft":
                detail = "Saved for later."
            elif status == "needs_review":
                detail = system_questions[0] if system_questions else (issue_questions or "Needs an answer before training.")
            elif status == "completed":
                detail = f"Training sample available in {', '.join(completed_training_projects)}."
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
        """Build the JSON state consumed by the React frontend."""

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
        training_sources = {
            "audioFiles": [
                self.frontend_file_record(path, script_name)
                for path in raw_audio_paths
                if isinstance(path, Path)
            ],
            "rttmFiles": [
                self.frontend_file_record(path, script_name)
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
                "youtubeLinks": self.frontend_route("/youtube-links", script_name),
                "deleteYoutubeLink": self.frontend_route("/youtube-links/delete", script_name),
                "convertYoutube": self.frontend_route("/actions/convert-youtube", script_name),
                "resetYoutube": self.frontend_route("/actions/reset-youtube-workspace", script_name),
                "review": self.frontend_route("/actions/review", script_name),
                "runDiarization": self.frontend_route("/actions/run-diarization", script_name),
                "tests": self.frontend_route("/actions/test", script_name),
                "saveModels": self.frontend_route("/models/save", script_name),
                "tracking": self.frontend_route("/api/tracking", script_name),
                "pageState": self.frontend_route("/api/page-state", script_name),
                "clusterQueue": self.frontend_route("/api/cluster-queue", script_name),
                "runtimeEstimate": self.frontend_route("/api/runtime-estimate", script_name),
                "fileSearch": self.frontend_route("/api/file-search", script_name),
                "fineTuneUpload": self.frontend_route("/fine-tuning/upload-sample", script_name),
                "fineTunePrepare": self.frontend_route("/fine-tuning/prepare", script_name),
                "fineTuneLaunch": self.frontend_route("/fine-tuning/launch", script_name),
                "fineTuneRenameProject": self.frontend_route("/fine-tuning/rename-project", script_name),
                "fineTuneRenameRun": self.frontend_route("/fine-tuning/rename-run", script_name),
                "fineTuneScoreRun": self.frontend_route("/api/fine-tuning/score-run", script_name),
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
                    "rows": training_label_rows,
                    "summary": context.get("training_label_summary", self.training_label_summary([], {})),
                    "defaultProjectName": DEFAULT_TRAINING_LABEL_PROJECT,
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
        """Inject workflow state into the frontend-owned React HTML shell."""

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
