#!/usr/bin/env python3
"""Training-label record persistence and segment parsing."""
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
    launch_training,
    build_sample,
    list_projects,
    normalize_backend,
    parse_rttm,
    probe_media_duration,
    prepare_project,
    read_project_display as ftm_read_project_display,
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


class TrainingLabelsMixin:
    """Training-label record persistence and segment parsing."""

    def load_training_label_records(self) -> dict[str, dict[str, object]]:
        """Load per-upload labeling state for the training label queue."""

        if not self.training_label_status_path.is_file():
            return {}
        try:
            payload = json.loads(self.training_label_status_path.read_text(encoding="utf-8"))
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
        return records

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

    def training_label_summary(
        self,
        audio_paths: list[Path],
        records: dict[str, dict[str, object]],
    ) -> dict[str, int]:
        """Count queue states for the labeling page summary."""

        summary = {
            "total": len(audio_paths),
            "not_started": 0,
            "draft": 0,
            "needs_review": 0,
            "completed": 0,
        }
        for path in audio_paths:
            audio_name = self.audio_relative_path(path)
            status = self.training_label_status(records.get(audio_name) or records.get(path.name))
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
                " ".join(
                    [
                        "SPEAKER",
                        session_id,
                        "1",
                        f"{float(segment['start']):.3f}",
                        f"{float(segment['duration']):.3f}",
                        "<NA>",
                        "<NA>",
                        str(segment["speaker"]),
                        "<NA>",
                        "<NA>",
                    ]
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
        target_backends = self.training_label_target_backends(raw_backend)
        backend = "both" if len(target_backends) > 1 else target_backends[0]
        project_name = (form.getfirst("label_project_name") or DEFAULT_TRAINING_LABEL_PROJECT).strip()
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
        # background — skip the user-facing redirect/notification chain.
        auto_save = bool(form.getfirst("label_auto_save"))

        base_record = {
            "backend": backend,
            "target_backends": target_backends,
            "project_name": project_name,
            "label_segments": raw_segments,
            "transcript_text": transcript_text,
            "include_transcript": include_transcript,
            "issue_questions": issue_questions,
            "system_questions": [],
        }
        # We always tag transcripts as "not aligned" with the speech-time labels —
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
            self.upsert_training_label_record(audio_name, {**base_record, "status": status})
            # Auto-save shouldn't navigate the user away — the browser is
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

        if not project_name:
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

        completed_samples = []
        try:
            for target_backend in target_backends:
                with audio_path.open("rb") as audio_stream:
                    completed_samples.append(
                        save_project_sample_streams(
                            project_name=project_name,
                            backend=target_backend,
                            audio_name=audio_name,
                            audio_stream=audio_stream,
                            rttm_name=f"{label_stem}.rttm",
                            rttm_stream=io.BytesIO(rttm_text.encode("utf-8")),
                            transcript_text=transcript_text,
                            root=self.root,
                        )
                    )
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

        primary_sample = completed_samples[0]
        completed_record = {
            **base_record,
            "status": "completed",
            "label_segments": normalized_label_segments,
            "completed_at_utc": utc_now_iso(),
            "training_project": (
                f"{backend}/{project_name}"
                if len(target_backends) == 1
                else f"nemo+pyannote/{project_name}"
            ),
            "training_projects": [f"{target_backend}/{project_name}" for target_backend in target_backends],
            "training_audio_path": self.describe_path(primary_sample.audio_path),
            "training_rttm_path": self.describe_path(primary_sample.rttm_path),
            "training_transcript_path": self.describe_path(primary_sample.transcript_path) if primary_sample.transcript_path else "",
            "speaker_count": primary_sample.num_speakers,
            "segment_count": len(segments),
        }
        self.upsert_training_label_record(audio_name, completed_record)

        # Auto-train hook — if any of the projects this sample was added to has
        # the "auto-train when labels complete" toggle on, kick off a background
        # prepare + sbatch. Skipped silently when the toggle is off; per-project
        # serialization lives inside auto_train.queue_auto_train.
        from dashboard import auto_train as _auto_train
        auto_train_messages: list[str] = []
        for target_backend in target_backends:
            try:
                project_summary = ftm_read_project_display(
                    project_name,
                    backend=target_backend,
                    root=self.root,
                )
            except Exception:  # noqa: BLE001
                project_summary = {}
            if not project_summary.get("auto_train"):
                continue
            try:
                prepare_options = self.auto_train_prepare_options(project_name, target_backend)
                extra_env = self.auto_train_extra_env(target_backend)
            except (ValueError, OSError) as exc:
                auto_train_messages.append(
                    f"Auto-train not queued for {target_backend}/{project_name}: {exc}"
                )
                continue
            queued = _auto_train.queue_auto_train(
                project_name,
                backend=target_backend,
                root=self.root,
                prepare_options=prepare_options,
                extra_env=extra_env,
            )
            auto_train_messages.append(
                f"Auto-train {'queued' if queued else 'pending'} for {target_backend}/{project_name}."
            )

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
