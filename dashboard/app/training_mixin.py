#!/usr/bin/env python3
"""Training-label record persistence and segment parsing.

Owns the manual-labeling side of the pipeline: serializing per-file label
records to disk in a stable format, parsing the segments out of RTTM/SRT
inputs, and bridging completed label sets into the fine-tune project
queue. The labeler UI itself lives in the React bundle; this mixin is the
WSGI/storage half that the UI talks to.
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
from urllib.parse import parse_qs, parse_qsl, quote, unquote, urlencode, urlsplit

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
    format_rttm_line,
    launch_training,
    build_sample,
    list_projects,
    normalize_backend,
    parse_rttm,
    probe_media_duration,
    prepare_project,
    project_dir as ftm_project_dir,
    read_project_display as ftm_read_project_display,
    run_status as fine_tuning_run_status,
    sanitize_filename,
    save_project_sample_links,
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


class TrainingLabelsMixin:
    """Training-label record persistence and segment parsing."""

    def handle_training_folders(self, environ):
        """Return per-folder label-status counts for the /training-labels picker.

        The page paints folder buckets first; rows are only fetched when the
        user picks one. Walking the inventory + the label_status records
        once produces all counts at once, which is far cheaper than
        shipping ~6 k row dicts on every page load.
        """

        audio_files = self.audio_inventory()
        records = self.load_training_label_records()
        rttm_index_lookup = getattr(self, "synthetic_rttm_index", None)
        rttm_index = rttm_index_lookup() if callable(rttm_index_lookup) else None
        synthetic_lookup = getattr(self, "synthetic_rttm_for_audio_via_index", None)

        empty = {"total": 0, "not_started": 0, "draft": 0, "needs_review": 0, "completed": 0}
        folders_data: dict[str, dict[str, int]] = {}
        totals = dict(empty)
        audio_root = self._resolved_audio_dir()

        for path in audio_files:
            try:
                rel = path.relative_to(audio_root).as_posix()
            except ValueError:
                continue
            folder_key = rel.split("/", 1)[0] if "/" in rel else "__root__"
            record = records.get(rel) or records.get(path.name)
            status = self.training_label_status(record)
            if status == "not_started" and rttm_index is not None and callable(synthetic_lookup):
                if synthetic_lookup(path, rttm_index) is not None:
                    status = "completed"
            bucket = folders_data.setdefault(folder_key, dict(empty))
            bucket["total"] += 1
            bucket[status] = bucket.get(status, 0) + 1
            totals["total"] += 1
            totals[status] = totals.get(status, 0) + 1

        folder_rows = []
        for folder_key, counts in folders_data.items():
            display_name = "Unsorted Root" if folder_key == "__root__" else folder_key
            folder_rows.append({"value": folder_key, "name": display_name, **counts})
        folder_rows.sort(key=lambda row: (row["value"] != "__root__", row["name"].lower()))

        return self.json_response("200 OK", {"folders": folder_rows, "totals": totals})

    def handle_training_labels_slice(self, environ):
        """Return one slice of training-label rows for the chosen folder/status.

        Query params:
          folder  - folder value from /api/training-folders (``__root__`` for
                    files directly under audio_in/). Empty = all folders.
          status  - optional filter: ``not_started``/``draft``/``needs_review``/``completed``.
          q       - optional case-insensitive substring filter on the audio path.
          limit   - max rows to return (capped at 200, default 50).
          offset  - rows to skip for pagination.
        """

        query = parse_qs(environ.get("QUERY_STRING", ""))
        folder = (query.get("folder") or [""])[0].strip()
        status_filter = (query.get("status") or [""])[0].strip().lower()
        needle = (query.get("q") or [""])[0].strip().lower()
        # hide_completed=1 drops completed rows on the server BEFORE counting,
        # so the pagination total matches what the user actually sees. The
        # frontend's "Show completed labels" toggle drives this flag; keeping
        # the filter server-side makes 'Page 3 of 12' an honest count instead
        # of 'showing 250 fetched but only 87 visible after client-side
        # filter'.
        hide_completed = (query.get("hide_completed") or [""])[0].strip().lower() in {"1", "true", "yes", "on"}
        try:
            limit = max(1, min(int((query.get("limit") or ["50"])[0]), 500))
        except ValueError:
            limit = 50
        try:
            offset = max(0, int((query.get("offset") or ["0"])[0]))
        except ValueError:
            offset = 0

        audio_files = self.audio_inventory()
        audio_root = self._resolved_audio_dir()
        records = self.load_training_label_records()
        rttm_index_lookup = getattr(self, "synthetic_rttm_index", None)
        rttm_index = rttm_index_lookup() if callable(rttm_index_lookup) else None
        synthetic_lookup = getattr(self, "synthetic_rttm_for_audio_via_index", None)

        matching: list[Path] = []
        for path in audio_files:
            try:
                rel = path.relative_to(audio_root).as_posix()
            except ValueError:
                continue
            if folder == "__root__":
                if "/" in rel:
                    continue
            elif folder:
                if not rel.startswith(folder + "/"):
                    continue
            if needle and needle not in rel.lower():
                continue
            if status_filter or hide_completed:
                record = records.get(rel) or records.get(path.name)
                row_status = self.training_label_status(record)
                if row_status == "not_started" and rttm_index is not None and callable(synthetic_lookup):
                    if synthetic_lookup(path, rttm_index) is not None:
                        row_status = "completed"
                if status_filter and row_status != status_filter:
                    continue
                if hide_completed and row_status == "completed":
                    continue
            matching.append(path)

        # Surface diarized work first so the most actionable rows land on
        # page 1 of the slice. Done before offset/limit so the ranking
        # spans the entire folder, not just the current page.
        matching = self.sort_paths_diarized_first(matching)
        total = len(matching)
        page_paths = matching[offset:offset + limit]
        rows = self._build_frontend_training_label_rows(
            page_paths,
            records,
            self.script_name(environ),
        )
        return self.json_response(
            "200 OK",
            {"rows": rows, "total": total, "offset": offset, "limit": limit},
        )

    def load_training_label_records(self) -> dict[str, dict[str, object]]:
        """Load per-upload labeling state for the training label queue.

        Caches by (mtime, size) of label_status.json. The file is ~750 KB and
        used to be re-parsed on every page render of /training-labels and
        /fine-tuning. Saving a label rewrites the file and bumps mtime, so
        the cached copy is replaced on the next call automatically.
        """

        status_path = self.training_label_status_path
        if not status_path.is_file():
            return {}
        try:
            stat = status_path.stat()
        except OSError:
            return {}
        fingerprint = (str(status_path), stat.st_mtime, stat.st_size)
        cached = getattr(self, "_training_label_records_cache", None)
        if cached is not None and cached[0] == fingerprint:
            # Hand back a shallow copy: upsert_training_label_record mutates
            # the returned dict, and we don't want that bleeding into the
            # cached snapshot. Inner records are read-only on the hot path,
            # so a shallow copy is enough.
            return dict(cached[1])

        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        raw_items = payload.get("items", payload) if isinstance(payload, dict) else {}
        if not isinstance(raw_items, dict):
            return {}
        records: dict[str, dict[str, object]] = {}
        for raw_name, raw_record in raw_items.items():
            if not isinstance(raw_record, dict):
                continue
            audio_name = self.clean_audio_selection_value(str(raw_name))
            if audio_name:
                records[audio_name] = dict(raw_record)
        self._training_label_records_cache = (fingerprint, records)
        return dict(records)

    def save_training_label_records(self, records: dict[str, dict[str, object]]) -> None:
        """Persist the label queue as stable JSON for the dashboard."""

        payload = {
            "version": 1,
            "updated_at_utc": utc_now_iso(),
            "items": {key: records[key] for key in sorted(records)},
        }
        self.training_label_status_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.training_label_status_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.training_label_status_path)

    def upsert_training_label_record(self, audio_name: str, updates: dict[str, object]) -> dict[str, object]:
        """Update one audio file's labeling state without dropping existing fields."""

        records = self.load_training_label_records()
        audio_name = self.clean_audio_selection_value(audio_name)
        existing = dict(records.get(audio_name, {}))
        now = utc_now_iso()
        record = {
            **existing,
            **updates,
            "audio_file": audio_name,
            "updated_at_utc": now,
        }
        record.setdefault("created_at_utc", now)
        records[audio_name] = record
        self.save_training_label_records(records)
        self.invalidate_dashboard_cache()
        return record

    def training_label_return_location(self, form: cgi.FieldStorage) -> str:
        """Return a safe app-local destination after saving labels."""

        raw_value = (form.getfirst("label_return_to") or "").strip()
        if not raw_value:
            return "/training-labels"
        parsed = urlsplit(raw_value)
        path = parsed.path or ""
        if path.startswith("/files/") or path in PAGE_PATHS:
            query_items = [
                (key, value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                if key not in {"message", "status"}
            ]
            query = urlencode(query_items)
            return path + (f"?{query}" if query else "")
        return "/training-labels"

    def training_label_status(self, record: dict[str, object] | None) -> str:
        """Normalize one label record into the states rendered by the queue."""

        if not record:
            return "not_started"
        status = str(record.get("status") or "").strip().lower().replace("-", "_")
        if status in {"draft", "needs_review", "completed"}:
            return status
        return "draft"

    def training_label_target_backends(self, raw_backend: str | None) -> list[str]:
        """Return backend folders that should receive a completed manual label."""

        normalized = (raw_backend or "both").strip().lower()
        if normalized in {"both", "all", "nemo+pyannote", "pyannote+nemo"}:
            return ["nemo", "pyannote"]
        return [normalize_backend(normalized)]

    def training_label_target_key(self, backend: str, project_name: str) -> str:
        """Return the backend/project key used by the completion popup and status records."""

        return f"{normalize_backend(backend)}/{slugify(project_name or DEFAULT_TRAINING_LABEL_PROJECT)}"

    def parse_training_label_target_value(self, raw_value: object) -> dict[str, str] | None:
        """Parse one backend/project form value into a normalized target mapping."""

        value = str(raw_value or "").strip()
        if not value or "/" not in value:
            return None
        raw_backend, raw_project_name = value.split("/", 1)
        project_name = slugify(raw_project_name)
        if not project_name:
            return None
        backend = normalize_backend(raw_backend)
        return {
            "backend": backend,
            "project_name": project_name,
            "key": self.training_label_target_key(backend, project_name),
        }

    def training_label_targets_from_values(self, values: list[object]) -> tuple[list[dict[str, str]], list[str]]:
        """Normalize repeated backend/project values from a submitted form."""

        targets: list[dict[str, str]] = []
        questions: list[str] = []
        seen: set[str] = set()
        for raw_value in values:
            try:
                target = self.parse_training_label_target_value(raw_value)
            except ValueError as exc:
                questions.append(str(exc))
                continue
            if not target or target["key"] in seen:
                continue
            targets.append(target)
            seen.add(target["key"])
        return targets, questions

    def training_label_targets_from_form(
        self,
        form: cgi.FieldStorage,
        *,
        fallback_backends: list[str],
        fallback_project_name: str,
    ) -> tuple[list[dict[str, str]], list[str], bool]:
        """Return explicit popup targets, or the legacy backend/project target list."""

        explicit_targets, questions = self.training_label_targets_from_values(form.getlist("label_training_targets"))
        if explicit_targets:
            return explicit_targets, questions, True
        if not fallback_project_name:
            return [], questions, False
        targets = []
        seen: set[str] = set()
        for backend in fallback_backends:
            project_key = self.training_label_target_key(backend, fallback_project_name)
            if project_key in seen:
                continue
            targets.append(
                {
                    "backend": normalize_backend(backend),
                    "project_name": fallback_project_name,
                    "key": project_key,
                }
            )
            seen.add(project_key)
        return targets, questions, False

    def training_label_summary(
        self,
        audio_paths: list[Path],
        records: dict[str, dict[str, object]],
    ) -> dict[str, int]:
        """Count queue states for the labeling page summary.

        Cached for 10 s. The inner ``rttm_lookup`` does ~5 stat() calls per
        audio file to find sidecars and label_work drafts, which on a
        networked filesystem with 6k+ files turns into 30s+ per render.
        Saving a label or completing one calls ``invalidate_dashboard_cache``,
        so the cache is never stale after a real edit.
        """

        if not audio_paths:
            return {
                "total": 0,
                "not_started": 0,
                "draft": 0,
                "needs_review": 0,
                "completed": 0,
            }
        return self.cached_value(
            "training_label_summary",
            ttl_seconds=60.0,
            builder=lambda: self._build_training_label_summary(audio_paths, records),
        )

    def _build_training_label_summary(
        self,
        audio_paths: list[Path],
        records: dict[str, dict[str, object]],
    ) -> dict[str, int]:
        summary = {
            "total": len(audio_paths),
            "not_started": 0,
            "draft": 0,
            "needs_review": 0,
            "completed": 0,
        }
        # Pre-index sidecar/label_work RTTMs so the per-audio synthetic check
        # is a set lookup instead of ~5 stat() calls per file. Index builder
        # walks each directory once.
        rttm_index_lookup = getattr(self, "synthetic_rttm_index", None)
        rttm_index = rttm_index_lookup() if callable(rttm_index_lookup) else None
        synthetic_lookup = getattr(self, "synthetic_rttm_for_audio_via_index", None)
        for path in audio_paths:
            audio_name = self.audio_relative_path(path)
            record = records.get(audio_name) or records.get(path.name)
            status = self.training_label_status(record)
            if status == "not_started" and rttm_index is not None and callable(synthetic_lookup):
                if synthetic_lookup(path, rttm_index) is not None:
                    status = "completed"
            summary[status] = summary.get(status, 0) + 1
        return summary

    def parse_training_label_time(self, raw_value: str, *, line_number: int) -> float:
        """Parse seconds, MM:SS, or HH:MM:SS timestamps used in label rows."""

        value = raw_value.strip()
        try:
            if ":" not in value:
                return float(value)
            parts = [float(part) for part in value.split(":")]
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: what time should '{raw_value}' represent?") from exc
        if len(parts) == 2:
            minutes, seconds = parts
            return (minutes * 60.0) + seconds
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return (hours * 3600.0) + (minutes * 60.0) + seconds
        raise ValueError(f"Line {line_number}: should this timestamp be seconds, MM:SS, or HH:MM:SS?")

    def normalized_speaker_label(self, raw_value: str, *, line_number: int) -> str:
        """Return a conservative speaker label suitable for RTTM output."""

        speaker = self.rttm_safe_token(raw_value)
        if not speaker:
            raise ValueError(f"Line {line_number}: which speaker label should be used?")
        return speaker

    def rttm_safe_token(self, raw_value: str, *, fallback: str = "") -> str:
        """Return one whitespace-free token that can safely occupy an RTTM field."""

        token = "".join(
            char if char.isalnum() or char in "_-" else "_"
            for char in raw_value.strip()
        ).strip("_-")
        return token or fallback

    def sort_paths_diarized_first(self, paths: list[Path]) -> list[Path]:
        """Re-order audio paths so rows the user can actually act on float up.

        Two-tier ordering:
          1. Has this file ever appeared in a diarization run's summary?
             If yes it goes in the top group; if no it stays alphabetical
             at the bottom. ('Diarized' is liberal here; even failed runs
             count, since a failed-and-rerun loop is part of the workflow.)
          2. Within the diarized group, surface drafts and questions before
             not-yet-labeled files and already-completed work. Within each
             status tier, alphabetical.

        Computes diarization + label lookups once and reuses them for every
        key call so a 6 k-file inventory sorts in tens of milliseconds.
        """

        diarization_records = self.diarization_model_history_lookup()
        label_records = self.load_training_label_records()
        rttm_index_lookup = getattr(self, "synthetic_rttm_index", None)
        rttm_index = rttm_index_lookup() if callable(rttm_index_lookup) else None
        synthetic_lookup = getattr(self, "synthetic_rttm_for_audio_via_index", None)
        audio_root = self._resolved_audio_dir()
        status_rank = {"draft": 0, "needs_review": 1, "not_started": 2, "completed": 3}

        def sort_key(path: Path) -> tuple:
            try:
                audio_name = path.relative_to(audio_root).as_posix()
            except ValueError:
                audio_name = path.name
            has_diarization = bool(diarization_records.get(audio_name))
            record = label_records.get(audio_name) or label_records.get(path.name) or {}
            status = self.training_label_status(record)
            if status == "not_started" and rttm_index is not None and callable(synthetic_lookup):
                if synthetic_lookup(path, rttm_index) is not None:
                    status = "completed"
            # Group 0 = diarized, group 1 = not. Within group 0, status_rank
            # decides; within group 1 we leave rank at 99 so files just sort
            # alphabetically below the diarized block.
            group = 0 if has_diarization else 1
            rank = status_rank.get(status, 99) if has_diarization else 99
            return (group, rank, audio_name.lower())

        return sorted(paths, key=sort_key)

    def all_known_speakers(self) -> list[str]:
        """Return every distinct speaker name the workspace has ever recorded.

        Used to populate the review HTML's speaker dropdown so a user can
        reuse names they entered for other audios. Pulls from each label
        record's parsed ``label_segments`` (skipping the diarizer's
        placeholder names like SPEAKER_00) and from the explicit
        ``known_speakers`` field if a record sets one.
        """

        records = self.load_training_label_records()
        seen: set[str] = set()
        for record in records.values():
            if not isinstance(record, dict):
                continue
            raw_segments = str(record.get("label_segments") or "")
            if raw_segments.strip():
                segments, _ = self.parse_training_label_segments(raw_segments)
                for segment in segments:
                    speaker = str(segment.get("speaker") or "").strip()
                    if speaker:
                        seen.add(speaker)
            extras = record.get("known_speakers") or []
            if isinstance(extras, (list, tuple)):
                for value in extras:
                    speaker = str(value or "").strip()
                    if speaker:
                        seen.add(speaker)
        return sorted(seen, key=lambda value: value.lower())

    def parse_training_label_segments(
        self,
        raw_segments: str,
    ) -> tuple[list[dict[str, object]], list[str]]:
        """Parse simple label rows or pasted RTTM into normalized segments."""

        segments: list[dict[str, object]] = []
        questions: list[str] = []
        for line_number, raw_line in enumerate(raw_segments.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lowered = line.lower()
            if lowered.startswith("start") or lowered.startswith("speaker,start"):
                continue
            try:
                if line.upper().startswith("SPEAKER "):
                    parts = line.split()
                    if len(parts) < 8:
                        raise ValueError(f"Line {line_number}: is this RTTM row missing fields?")
                    start = self.parse_training_label_time(parts[3], line_number=line_number)
                    duration = float(parts[4])
                    speaker = self.normalized_speaker_label(parts[7], line_number=line_number)
                else:
                    if "," in line:
                        parts = [part.strip() for part in line.split(",")]
                    elif "\t" in line:
                        parts = [part.strip() for part in line.split("\t")]
                    else:
                        parts = line.split()
                    if len(parts) < 3:
                        raise ValueError(
                            f"Line {line_number}: what are the start time, end time, and speaker?"
                        )
                    start = self.parse_training_label_time(parts[0], line_number=line_number)
                    end = self.parse_training_label_time(parts[1], line_number=line_number)
                    duration = end - start
                    speaker = self.normalized_speaker_label(" ".join(parts[2:]), line_number=line_number)
                if start < 0:
                    raise ValueError(f"Line {line_number}: should the segment start before 0 seconds?")
                if duration <= 0:
                    raise ValueError(f"Line {line_number}: should the end time be after the start time?")
            except ValueError as exc:
                questions.append(str(exc))
                continue
            segments.append(
                {
                    "start": round(start, 3),
                    "duration": round(duration, 3),
                    "speaker": speaker,
                }
            )
        if not segments and not questions:
            questions.append("Which speaker-time segments should be used for this uploaded audio?")
        return segments, questions

    def training_segments_to_rttm(self, *, audio_name: str, segments: list[dict[str, object]]) -> str:
        """Convert normalized label segments into the RTTM used by training."""

        session_id = self.rttm_safe_token(Path(sanitize_filename(audio_name)).stem, fallback="sample")
        lines = []
        for segment in sorted(segments, key=lambda item: (float(item["start"]), float(item["duration"]), str(item["speaker"]))):
            lines.append(
                format_rttm_line(
                    session_id=session_id,
                    start=float(segment["start"]),
                    duration=float(segment["duration"]),
                    speaker=str(segment["speaker"]),
                )
            )
        return "\n".join(lines) + "\n"

    def fine_tuning_project_metadata(self, project_name: str, backend: str) -> dict[str, object]:
        """Load the last prepared settings for a project, if it has been prepared before."""

        metadata_path = (
            self.root
            / "fine_tuning"
            / "projects"
            / normalize_backend(backend)
            / slugify(project_name)
            / "artifacts"
            / "metadata.json"
        )
        if not metadata_path.is_file():
            return {}
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def auto_train_prepare_options(self, project_name: str, backend: str) -> dict[str, object]:
        """Reuse project-specific prepare settings first, then dashboard fine-tuning defaults."""

        normalized_backend = normalize_backend(backend)
        metadata = self.fine_tuning_project_metadata(project_name, normalized_backend)
        preferences = self.model_preferences()
        nemo_defaults = dict(preferences.get("nemo_fine_tuning", {}))
        pyannote_defaults = dict(preferences.get("pyannote_fine_tuning", {}))

        def pick(metadata_key: str, preference_map: dict[str, object], preference_key: str, default: object) -> object:
            value = metadata.get(metadata_key)
            if value is not None and value != "":
                return value
            preference_value = preference_map.get(preference_key, default)
            return default if preference_value is None or preference_value == "" else preference_value

        if normalized_backend == "pyannote":
            options: dict[str, object] = {
                "devices": self.parse_int(
                    str(pick("devices", pyannote_defaults, "devices", DEFAULT_DEVICES)),
                    DEFAULT_DEVICES,
                    "devices",
                ),
                "max_epochs": self.parse_int(
                    str(pick("max_epochs", pyannote_defaults, "max_epochs", DEFAULT_MAX_EPOCHS)),
                    DEFAULT_MAX_EPOCHS,
                    "max_epochs",
                ),
                "pyannote_pretrained_model": str(
                    pick(
                        "pyannote_pretrained_model",
                        pyannote_defaults,
                        "pretrained_model",
                        DEFAULT_PYANNOTE_PRETRAINED_MODEL,
                    )
                ),
                "pyannote_duration": self.parse_float(
                    str(pick("pyannote_duration", pyannote_defaults, "duration", DEFAULT_PYANNOTE_DURATION)),
                    DEFAULT_PYANNOTE_DURATION,
                    "pyannote_duration",
                ),
                "pyannote_max_speakers_per_chunk": self.parse_int(
                    str(
                        pick(
                            "pyannote_max_speakers_per_chunk",
                            pyannote_defaults,
                            "max_speakers_per_chunk",
                            DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK,
                        )
                    ),
                    DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK,
                    "pyannote_max_speakers_per_chunk",
                ),
                "pyannote_max_speakers_per_frame": self.parse_int(
                    str(
                        pick(
                            "pyannote_max_speakers_per_frame",
                            pyannote_defaults,
                            "max_speakers_per_frame",
                            DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME,
                        )
                    ),
                    DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME,
                    "pyannote_max_speakers_per_frame",
                ),
            }
        else:
            options = {
                "train_ratio": self.parse_float(
                    str(pick("train_ratio", nemo_defaults, "train_ratio", DEFAULT_TRAIN_RATIO)),
                    DEFAULT_TRAIN_RATIO,
                    "train_ratio",
                ),
                "base_window": self.parse_float(
                    str(pick("base_window", nemo_defaults, "base_window", DEFAULT_BASE_WINDOW)),
                    DEFAULT_BASE_WINDOW,
                    "base_window",
                ),
                "base_shift": self.parse_float(
                    str(pick("base_shift", nemo_defaults, "base_shift", DEFAULT_BASE_SHIFT)),
                    DEFAULT_BASE_SHIFT,
                    "base_shift",
                ),
                "step_count": self.parse_int(
                    str(pick("step_count", nemo_defaults, "step_count", DEFAULT_STEP_COUNT)),
                    DEFAULT_STEP_COUNT,
                    "step_count",
                ),
                "config_name": str(pick("config_name", nemo_defaults, "config_name", DEFAULT_CONFIG_NAME)),
                "speaker_model": str(pick("speaker_model", nemo_defaults, "speaker_model", DEFAULT_SPEAKER_MODEL)),
                "devices": self.parse_int(
                    str(pick("devices", nemo_defaults, "devices", DEFAULT_DEVICES)),
                    DEFAULT_DEVICES,
                    "devices",
                ),
                "max_epochs": self.parse_int(
                    str(pick("max_epochs", nemo_defaults, "max_epochs", DEFAULT_MAX_EPOCHS)),
                    DEFAULT_MAX_EPOCHS,
                    "max_epochs",
                ),
            }
            nemo_root = str(metadata.get("nemo_root") or "").strip()
            if nemo_root:
                options["nemo_root"] = Path(nemo_root)

        options.update(
            {
                "slurm_partition": str(metadata.get("slurm_partition") or DEFAULT_SLURM_PARTITION),
                "slurm_time": str(metadata.get("slurm_time") or DEFAULT_SLURM_TIME),
                "slurm_memory": str(metadata.get("slurm_memory") or DEFAULT_SLURM_MEMORY),
                "slurm_cpus": self.parse_int(str(metadata.get("slurm_cpus") or DEFAULT_SLURM_CPUS), DEFAULT_SLURM_CPUS, "slurm_cpus"),
                "slurm_gpus": self.parse_int(str(metadata.get("slurm_gpus") or DEFAULT_SLURM_GPUS), DEFAULT_SLURM_GPUS, "slurm_gpus"),
            }
        )
        return options

    def project_sample_stem(self, audio_name: str) -> str:
        """Return the on-disk stem the project's audio/rttm/text files share.

        We use ``diarization_output_base`` so that two audio files with the
        same basename in different subfolders (``a/clip.wav`` vs.
        ``b/clip.wav``) get unique, path-flattened stems
        (``a__clip`` / ``b__clip``) and never overwrite each other. The label
        completion handler now passes this stem in for both audio and RTTM
        when calling ``save_project_sample_streams``, so the two files are
        guaranteed to pair on disk.
        """

        flattened = self.diarization_output_base(audio_name)
        if flattened:
            return flattened
        # Fallback: very old / pre-flatten records may have been saved using
        # the basename-only stem produced by sanitize_filename. The legacy
        # variant is offered separately by ``legacy_project_sample_stem`` so
        # cleanup can still find pre-flatten files.
        return Path(sanitize_filename(audio_name)).stem or "sample"

    def legacy_project_sample_stem(self, audio_name: str) -> str:
        """Old basename-only stem ``save_project_sample_streams`` used to write.

        Kept around so that re-completing or uncompleting a label whose
        previous run used the basename can still find and remove the stale
        ``001_clip.wav`` / ``001_clip.rttm`` pair, even though the new save
        path uses ``subfolder__001_clip`` instead.
        """

        return Path(sanitize_filename(audio_name)).stem or ""

    def remove_project_sample_files(
        self,
        *,
        project_name: str,
        backend: str,
        stem: str,
    ) -> dict[str, list[str]]:
        """Delete the audio/rttm/transcript files for one stem from a project.

        Returns a small report of what got removed so the caller can include it
        in the user-facing notification. We glob on stem to catch any extension
        that might have been used for the audio/transcript copy.
        """

        removed: dict[str, list[str]] = {"audio": [], "rttm": [], "transcript": []}
        if not stem:
            return removed
        try:
            normalized_backend = normalize_backend(backend)
        except ValueError:
            return removed
        target = ftm_project_dir(project_name, backend=normalized_backend, root=self.root)
        if not target.is_dir():
            return removed
        # ``audio`` and ``text`` files keep their original extension, so glob on
        # stem rather than hard-coding ".wav" or ".txt".
        for sub_dir, key in (("audio", "audio"), ("text", "transcript")):
            sub_path = target / sub_dir
            if not sub_path.is_dir():
                continue
            for path in sorted(sub_path.glob(f"{stem}.*")):
                try:
                    path.unlink()
                    removed[key].append(self.describe_path(path))
                except OSError:
                    continue
        rttm_path = target / "rttm" / f"{stem}.rttm"
        if rttm_path.is_file():
            try:
                rttm_path.unlink()
                removed["rttm"].append(self.describe_path(rttm_path))
            except OSError:
                pass
        return removed

    def remove_label_sample_from_project(
        self,
        *,
        project_name: str,
        backend: str,
        audio_name: str,
    ) -> dict[str, list[str]]:
        """Remove a label's audio/rttm/transcript copies from one project.

        Tries the current path-flattened stem first, then the legacy
        basename-only stem so records written before the pairing fix still
        get cleaned. Both variants share the same project folders, so we
        merge the reports.
        """

        merged: dict[str, list[str]] = {"audio": [], "rttm": [], "transcript": []}
        seen_stems: set[str] = set()
        for stem in (
            self.project_sample_stem(audio_name),
            self.legacy_project_sample_stem(audio_name),
        ):
            if not stem or stem in seen_stems:
                continue
            seen_stems.add(stem)
            removed = self.remove_project_sample_files(
                project_name=project_name,
                backend=backend,
                stem=stem,
            )
            for key, paths in removed.items():
                merged[key].extend(paths)
        return merged

    def cleanup_stale_training_targets(
        self,
        *,
        previous_keys: list[str],
        new_keys: list[str],
        audio_name: str,
    ) -> list[str]:
        """Drop sample files from projects this label no longer points at.

        When a user re-completes a label and switches the chosen fine-tuned
        model, the old project's ``audio/`` and ``rttm/`` folders would
        otherwise keep a stale copy of the sample and silently feed it to the
        next training run. Compare old vs. new target keys and clear anything
        the new completion isn't going to write to.
        """

        new_set = {key for key in new_keys if key}
        cleared: list[str] = []
        for key in previous_keys:
            if not key or key in new_set or "/" not in key:
                continue
            backend, project_name = key.split("/", 1)
            removed = self.remove_label_sample_from_project(
                project_name=project_name,
                backend=backend,
                audio_name=audio_name,
            )
            if any(removed.values()):
                cleared.append(key)
        return cleared

    def auto_train_extra_env(self, backend: str) -> dict[str, str]:
        """Return launch environment needed for automatic training."""

        if normalize_backend(backend) != "pyannote":
            return {}
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or self.dashboard_secret("HF_TOKEN")
            or self.dashboard_secret("HUGGINGFACE_TOKEN")
            or self.dashboard_secret("HUGGINGFACE_HUB_TOKEN")
        )
        if not hf_token:
            return {}
        return {"HF_TOKEN": hf_token, "HUGGINGFACE_HUB_TOKEN": hf_token}

    def handle_training_label_save(self, environ):
        """Save draft labels or complete an uploaded audio item as training material."""

        form = self.parse_form(environ)
        return_location = self.training_label_return_location(form)
        requested_audio = self.clean_audio_selection_value((form.getfirst("audio_file") or "").strip())
        if not requested_audio:
            return self.redirect(
                environ,
                return_location,
                message="Which uploaded audio item should be labeled?",
                status="error",
            )

        matched_audio = self.selected_audio_names([requested_audio], run_all=False)
        if not matched_audio:
            return self.redirect(
                environ,
                return_location,
                message=(
                    f"I could not find '{requested_audio}' in audio_in. "
                    "Should it be uploaded again before labeling?"
                ),
                status="error",
            )
        audio_name = matched_audio[0]
        audio_path = self.audio_dir / audio_name

        raw_backend = form.getfirst("label_backend") or "both"
        legacy_target_backends = self.training_label_target_backends(raw_backend)
        project_name = (form.getfirst("label_project_name") or DEFAULT_TRAINING_LABEL_PROJECT).strip()
        training_targets, target_questions, explicit_training_targets = self.training_label_targets_from_form(
            form,
            fallback_backends=legacy_target_backends,
            fallback_project_name=project_name,
        )
        if training_targets:
            project_name = training_targets[0]["project_name"]
        target_backends = self.ordered_unique([target["backend"] for target in training_targets]) or legacy_target_backends
        backend = "both" if len(target_backends) > 1 else target_backends[0]
        target_projects = [target["key"] for target in training_targets]
        requested_auto_targets, auto_target_questions = self.training_label_targets_from_values(
            form.getlist("label_auto_train_targets")
        )
        requested_auto_target_keys = {target["key"] for target in requested_auto_targets}
        explicit_auto_train_targets = bool(requested_auto_target_keys)
        skip_auto_train = bool(form.getfirst("label_auto_train_skip"))
        # "Start a new training run" toggle from the label-complete form. When the
        # user ticks it on, every project this label gets saved into should also
        # queue an auto-train, regardless of the project-wide auto_train flag.
        # Lets the user opt in per-completion without flipping the persistent
        # toggle on the project itself.
        auto_train_now = bool(form.getfirst("label_auto_train_now"))
        # Optional human-friendly name for the training run kicked off by this
        # specific completion. Empty string means "let launch_training pick a
        # default name" - same behavior as before this field existed.
        new_training_name = (form.getfirst("label_new_training_name") or "").strip()
        selected_training_target_label = (form.getfirst("label_training_target_label") or "").strip()
        raw_segments = (form.getfirst("label_segments") or "").strip()
        # Dialogue toggle: when off, drop the transcript before it gets stored in
        # the training sample. Diarization training never reads it, but it would
        # otherwise be exported into the project's text/ folder for the user's
        # records. Absent field (legacy callers / tests without the new UI) is
        # treated as ON so existing behavior keeps working unchanged.
        include_transcript_raw = form.getfirst("label_include_transcript")
        if include_transcript_raw is None:
            include_transcript = True
        else:
            include_transcript = str(include_transcript_raw).strip().lower() not in {"", "0", "false", "off", "no"}
        raw_transcript_text = (form.getfirst("label_transcript_text") or "").strip()
        transcript_text = raw_transcript_text if include_transcript else ""
        issue_questions = (form.getfirst("label_issue_questions") or "").strip()
        label_source = (form.getfirst("label_source") or "").strip()
        label_review_path = (form.getfirst("label_review_path") or "").strip()
        action = (form.getfirst("label_action") or "draft").strip().lower()
        # Auto-save flag means the browser is debouncing a draft save in the
        # background - skip the user-facing redirect/notification chain.
        auto_save = bool(form.getfirst("label_auto_save"))
        existing_record = self.load_training_label_records().get(audio_name, {})

        base_record = {
            "backend": backend,
            "target_backends": target_backends,
            "target_projects": target_projects,
            "project_name": project_name,
            "selected_training_target_label": selected_training_target_label,
            "label_segments": raw_segments,
            "transcript_text": transcript_text,
            "include_transcript": include_transcript,
            "issue_questions": issue_questions,
            "system_questions": [],
        }
        # We always tag transcripts as "not aligned" with the speech-time labels -
        # we don't have time-stamped dialogue, so any saved transcript is just a
        # bag of text alongside the RTTM, not a per-segment annotation.
        if include_transcript and raw_transcript_text:
            base_record["transcript_unmatched"] = True
        if label_source:
            base_record["source"] = label_source
        if label_review_path:
            try:
                base_record["review_path"] = self.describe_path(self.resolve_under_root(label_review_path))
            except ValueError:
                base_record["review_path"] = label_review_path
        if action != "complete":
            status = "needs_review" if issue_questions else "draft"
            if (
                auto_save
                and status == "draft"
                and self.training_label_status(existing_record) == "completed"
                and str(existing_record.get("label_segments") or "").strip().replace("\r\n", "\n")
                == raw_segments.replace("\r\n", "\n")
            ):
                # Opening a completed review page can trigger a background
                # draft save with unchanged form data. Preserve the completed
                # state and training-file bookkeeping unless the user makes a
                # real edit or explicitly clicks Save For Later.
                status = "completed"
            self.upsert_training_label_record(audio_name, {**base_record, "status": status})
            # Auto-save shouldn't navigate the user away - the browser is
            # debouncing this in the background while they keep editing.
            if auto_save:
                return self.json_response("200 OK", {"status": "saved", "kind": status})
            return self.redirect(
                environ,
                return_location,
                message=(
                    f"Saved '{audio_name}' for later."
                    if status == "draft"
                    else f"Saved questions for '{audio_name}' before training."
                ),
                status="success",
            )

        if target_questions or auto_target_questions:
            questions = [*target_questions, *auto_target_questions]
            self.upsert_training_label_record(
                audio_name,
                {**base_record, "status": "needs_review", "system_questions": questions[:8]},
            )
            return self.redirect(
                environ,
                return_location,
                message=self.notification_message(
                    "The selected training target could not be used.",
                    *questions[:8],
                ),
                status="error",
            )
        if not training_targets:
            self.upsert_training_label_record(
                audio_name,
                {
                    **base_record,
                    "status": "needs_review",
                    "system_questions": ["Which fine-tuning project should receive this completed label?"],
                },
            )
            return self.redirect(
                environ,
                return_location,
                message="Which fine-tuning project should receive this completed label?",
                status="error",
            )
        if issue_questions:
            self.upsert_training_label_record(
                audio_name,
                {
                    **base_record,
                    "status": "needs_review",
                    "system_questions": ["Answer or remove the open questions before marking this item complete."],
                },
            )
            return self.redirect(
                environ,
                return_location,
                message=self.notification_message(
                    "This item still has unresolved labeling questions.",
                    issue_questions,
                    "Answer or remove them before marking the sample complete for training.",
                ),
                status="error",
            )

        segments, questions = self.parse_training_label_segments(raw_segments)
        if questions:
            self.upsert_training_label_record(
                audio_name,
                {**base_record, "status": "needs_review", "system_questions": questions[:8]},
            )
            return self.redirect(
                environ,
                return_location,
                message=self.notification_message(
                    "I need answers before this item can be marked complete.",
                    *questions[:8],
                ),
                status="error",
            )

        rttm_text = self.training_segments_to_rttm(audio_name=audio_name, segments=segments)
        normalized_label_segments = "\n".join(
            f"{float(segment['start']):.3f} "
            f"{float(segment['start']) + float(segment['duration']):.3f} "
            f"{segment['speaker']}"
            for segment in segments
        )
        self.training_label_work_dir.mkdir(parents=True, exist_ok=True)
        label_stem = self.diarization_output_base(audio_name)
        review_rttm_path = self.training_label_work_dir / f"{label_stem}.rttm"
        review_rttm_path.write_text(rttm_text, encoding="utf-8")
        try:
            build_sample(audio_path, review_rttm_path, None)
        except Exception as exc:
            question = (
                f"{exc} Should the audio be replaced, converted, or should the segment times be adjusted?"
            )
            self.upsert_training_label_record(
                audio_name,
                {**base_record, "status": "needs_review", "system_questions": [question]},
            )
            return self.redirect(
                environ,
                return_location,
                message=self.notification_message(
                    "I could not verify this completed label for training.",
                    question,
                ),
                status="error",
            )

        # Drop any sample copies the previous completion left in projects this
        # label is no longer training. Without this, switching the popup choice
        # from project A to project B would silently keep training A on stale
        # data, which caused the duplicate-RTTM bug we hit before.
        previous_target_keys = [
            str(item)
            for item in (existing_record.get("training_projects") or [])
            if str(item).strip()
        ]
        new_target_keys = [target["key"] for target in training_targets]
        cleared_target_keys = self.cleanup_stale_training_targets(
            previous_keys=previous_target_keys,
            new_keys=new_target_keys,
            audio_name=audio_name,
        )

        # Audio and RTTM must share the same stem so the training pipeline can
        # pair them. Build the project-side filename from ``label_stem`` (the
        # path-flattened identifier) so that two audio files with the same
        # basename in different folders never collide. ``save_project_sample_streams``
        # uses the audio filename's stem as the canonical key, so feeding it
        # ``<label_stem><suffix>`` is the simplest way to align the pair.
        project_audio_filename = f"{label_stem}{audio_path.suffix.lower() or '.wav'}"
        completed_targets = []
        try:
            for target in training_targets:
                # Same project, possibly older naming: drop any legacy basename
                # copy first so we don't end up with two pairs (old and new)
                # in audio/ and rttm/.
                self.remove_label_sample_from_project(
                    project_name=target["project_name"],
                    backend=target["backend"],
                    audio_name=audio_name,
                )
                # Symlink the audio_in source into the project rather than
                # copying its bytes, since duplicating gigabytes of stitched WAVs
                # per project was the disk-usage cliff the user wants to
                # avoid. RTTM is small and gets rewritten in canonical form
                # via the label_work draft path we wrote a few lines up.
                sample = save_project_sample_links(
                    project_name=target["project_name"],
                    backend=target["backend"],
                    audio_path=audio_path,
                    audio_name=project_audio_filename,
                    rttm_path=review_rttm_path,
                    transcript_text=transcript_text,
                    root=self.root,
                )
                completed_targets.append({**target, "sample": sample})
        except Exception as exc:
            question = f"{exc} Should this item stay saved for later until the training sample can be written?"
            self.upsert_training_label_record(
                audio_name,
                {**base_record, "status": "needs_review", "system_questions": [question]},
            )
            return self.redirect(
                environ,
                return_location,
                message=self.notification_message(
                    "The label validated but could not be copied into the training project.",
                    question,
                ),
                status="error",
            )

        primary_sample = completed_targets[0]["sample"]
        training_projects = [target["key"] for target in completed_targets]
        training_usage = [
            {
                "backend": str(target["backend"]),
                "project_name": str(target["project_name"]),
                "project_key": str(target["key"]),
                "sample_audio_path": self.describe_path(target["sample"].audio_path),
                "sample_rttm_path": self.describe_path(target["sample"].rttm_path),
                "sample_transcript_path": self.describe_path(target["sample"].transcript_path)
                if target["sample"].transcript_path
                else "",
                "auto_train_requested": False,
                "auto_train_status": "skipped",
            }
            for target in completed_targets
        ]
        usage_by_key = {str(item["project_key"]): item for item in training_usage}
        completed_record = {
            **base_record,
            "status": "completed",
            "label_segments": normalized_label_segments,
            "completed_at_utc": utc_now_iso(),
            "training_project": training_projects[0] if len(training_projects) == 1 else ", ".join(training_projects),
            "training_projects": training_projects,
            "training_usage": training_usage,
            "queued_training_projects": [],
            "explicit_training_targets": explicit_training_targets,
            "training_audio_path": self.describe_path(primary_sample.audio_path),
            "training_rttm_path": self.describe_path(primary_sample.rttm_path),
            "training_transcript_path": self.describe_path(primary_sample.transcript_path) if primary_sample.transcript_path else "",
            "speaker_count": primary_sample.num_speakers,
            "segment_count": len(segments),
        }

        # Auto-train hook - explicit popup selections queue immediately. Legacy
        # posts still honor the per-project auto-train toggle.
        from dashboard import auto_train as _auto_train
        auto_train_messages: list[str] = []
        queued_training_projects: list[str] = []
        if skip_auto_train:
            auto_train_messages.append("Auto-train skipped for this completion.")
        for target in completed_targets:
            target_backend = str(target["backend"])
            target_project_name = str(target["project_name"])
            target_key = str(target["key"])
            usage = usage_by_key.get(target_key)
            if skip_auto_train:
                continue
            try:
                project_summary = ftm_read_project_display(
                    target_project_name,
                    backend=target_backend,
                    root=self.root,
                )
            except Exception:  # noqa: BLE001
                project_summary = {}
            should_queue = (
                target_key in requested_auto_target_keys
                if explicit_auto_train_targets
                else (auto_train_now or bool(project_summary.get("auto_train")))
            )
            if not should_queue:
                continue
            try:
                prepare_options = self.auto_train_prepare_options(target_project_name, target_backend)
                extra_env = self.auto_train_extra_env(target_backend)
            except (ValueError, OSError) as exc:
                if usage is not None:
                    usage["auto_train_requested"] = True
                    usage["auto_train_status"] = "error"
                    usage["auto_train_error"] = str(exc)
                auto_train_messages.append(
                    f"Auto-train not queued for {target_key}: {exc}"
                )
                continue
            # Only forward version_name when the user actually typed one - keeps
            # the call signature backward-compatible with anything (including
            # tests) that monkey-patches queue_auto_train without the new kwarg.
            queue_kwargs = {
                "backend": target_backend,
                "root": self.root,
                "prepare_options": prepare_options,
                "extra_env": extra_env,
            }
            if new_training_name:
                queue_kwargs["version_name"] = new_training_name
            queued = _auto_train.queue_auto_train(target_project_name, **queue_kwargs)
            if usage is not None:
                usage["auto_train_requested"] = True
                usage["auto_train_status"] = "queued" if queued else "pending"
                if new_training_name:
                    usage["requested_version_name"] = new_training_name
            queued_training_projects.append(target_key)
            named_suffix = f" as '{new_training_name}'" if new_training_name else ""
            auto_train_messages.append(
                f"Auto-train {'queued' if queued else 'pending'}{named_suffix} for {target_key}."
            )
        completed_record["training_usage"] = training_usage
        completed_record["queued_training_projects"] = queued_training_projects
        self.upsert_training_label_record(audio_name, completed_record)

        if auto_save:
            return self.json_response(
                "200 OK",
                {
                    "status": "completed",
                    "auto_train": auto_train_messages,
                },
            )

        return self.redirect(
            environ,
            return_location,
            message=self.notification_message(
                f"Marked '{audio_name}' complete and added it to {', '.join(completed_record['training_projects'])}.",
                "The completed label is now part of the fine-tuning samples used by Prepare Artifacts.",
                *auto_train_messages,
            ),
            status="success",
        )

    def handle_training_label_uncomplete(self, environ):
        """Roll a completed label back to draft and remove its training sample copies.

        The user wanted a way to take a label OUT of the fine-tuning sample
        pool, which is common after spotting a labeling mistake post-completion. We
        delete the audio/rttm/transcript copies from every project this label
        was added to (covers both the new path-flattened stem and the legacy
        basename stem), drop the staged copy in ``label_work/``, and rewind
        the record's status to ``draft`` so the user can keep editing the
        segments without retyping anything.
        """

        form = self.parse_form(environ)
        return_location = self.training_label_return_location(form)
        requested_audio = self.clean_audio_selection_value((form.getfirst("audio_file") or "").strip())
        if not requested_audio:
            return self.redirect(
                environ,
                return_location,
                message="Which completed label should be removed from training?",
                status="error",
            )
        records = self.load_training_label_records()
        record = records.get(requested_audio) or records.get(Path(requested_audio).name) or {}
        if not record:
            return self.redirect(
                environ,
                return_location,
                message=f"No saved label exists for '{requested_audio}'.",
                status="error",
            )
        audio_name = str(record.get("audio_file") or requested_audio)
        previous_target_keys = [
            str(item)
            for item in (record.get("training_projects") or [])
            if str(item).strip()
        ]

        cleared_projects: list[str] = []
        for key in previous_target_keys:
            if "/" not in key:
                continue
            backend, project_name = key.split("/", 1)
            removed = self.remove_label_sample_from_project(
                project_name=project_name,
                backend=backend,
                audio_name=audio_name,
            )
            if any(removed.values()):
                cleared_projects.append(key)

        # Drop the label_work staging copy too; it gets rewritten on the next
        # completion, so leaving it behind would just be confusing dead state.
        label_stem = self.diarization_output_base(audio_name)
        review_rttm_path = self.training_label_work_dir / f"{label_stem}.rttm"
        if review_rttm_path.is_file():
            try:
                review_rttm_path.unlink()
            except OSError:
                pass

        # Preserve segments / transcript / questions so the user can keep
        # editing; only flip the status and clear the completion bookkeeping.
        rewound = dict(record)
        rewound["status"] = "draft"
        rewound["training_projects"] = []
        rewound["training_project"] = ""
        rewound["queued_training_projects"] = []
        rewound["training_usage"] = []
        rewound["training_audio_path"] = ""
        rewound["training_rttm_path"] = ""
        rewound["training_transcript_path"] = ""
        rewound["explicit_training_targets"] = False
        rewound.pop("completed_at_utc", None)
        rewound["uncompleted_at_utc"] = utc_now_iso()
        rewound.setdefault("system_questions", [])
        self.upsert_training_label_record(audio_name, rewound)

        if cleared_projects:
            message = self.notification_message(
                f"Removed '{audio_name}' from training.",
                f"Cleared sample files in: {', '.join(cleared_projects)}.",
                "Edit and click Complete For Training again to add it back.",
            )
        else:
            message = self.notification_message(
                f"Reverted '{audio_name}' to draft.",
                "No project sample files needed clearing.",
            )
        return self.redirect(
            environ,
            return_location,
            message=message,
            status="success",
        )
