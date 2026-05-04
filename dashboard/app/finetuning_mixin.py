#!/usr/bin/env python3
"""Fine-tuning project preparation, launch, and metrics."""
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


class FineTuningMixin:
    """Fine-tuning project preparation, launch, and metrics."""

    def save_project_sample_from_paths(
        self,
        *,
        project_name: str,
        backend: str,
        audio_path: Path,
        rttm_path: Path,
        transcript_path: Path | None = None,
        transcript_text: str = "",
    ):
        """Copy a training sample from SSH workspace files into a backend project."""

        audio_stream = audio_path.open("rb")
        rttm_stream = rttm_path.open("rb")
        transcript_stream = transcript_path.open("rb") if transcript_path is not None else None
        try:
            return save_project_sample_streams(
                project_name=project_name,
                backend=backend,
                audio_name=audio_path.name,
                audio_stream=audio_stream,
                rttm_name=rttm_path.name,
                rttm_stream=rttm_stream,
                transcript_name=transcript_path.name if transcript_path is not None else None,
                transcript_stream=transcript_stream,
                transcript_text=transcript_text,
                root=self.root,
            )
        finally:
            if transcript_stream is not None:
                transcript_stream.close()
            rttm_stream.close()
            audio_stream.close()

    def handle_finetune_upload(self, environ):
        """Store one or more fine-tuning samples composed of audio plus RTTM supervision."""

        form = self.parse_form(environ)
        preferences = self.model_preferences()
        project_name = (form.getfirst("project_name") or "").strip()
        if not project_name:
            return self.redirect(environ, "/fine-tuning", message="Provide a project name for fine-tuning uploads.", status="error")
        backend = normalize_backend(form.getfirst("fine_tuning_backend") or str(preferences["fine_tuning_backend"]))

        server_audio_paths, server_audio_errors = self.selected_workspace_paths(
            form,
            "server_audio_paths",
            label="Audio",
            suffixes=set(AUDIO_EXTENSIONS),
            required_root=self.audio_dir,
        )
        server_rttm_paths, server_rttm_errors = self.selected_workspace_paths(
            form,
            "server_rttm_paths",
            label="RTTM",
            suffixes=TRAINING_RTTM_SUFFIXES,
        )
        server_transcript_paths, server_transcript_errors = self.selected_workspace_paths(
            form,
            "server_transcript_paths",
            label="Transcript",
            suffixes=TRAINING_TRANSCRIPT_SUFFIXES,
        )
        audio_items = self.uploaded_file_items(form, "training_audio")
        rttm_items = self.uploaded_file_items(form, "training_rttm")
        transcript_items = self.uploaded_file_items(form, "training_transcript")
        transcript_text = (form.getfirst("training_transcript_text") or "").strip()

        saved_samples = []
        failed_samples: list[str] = []
        notes = []

        path_errors = [*server_audio_errors, *server_rttm_errors, *server_transcript_errors]
        if path_errors:
            return self.redirect(
                environ,
                "/fine-tuning",
                message=self.notification_message("Selected SSH training files could not be used.", *path_errors[:8]),
                status="error",
            )

        if not server_audio_paths and not audio_items:
            return self.redirect(
                environ,
                "/fine-tuning",
                message="Select at least one SSH audio file from audio_in/ for training.",
                status="error",
            )
        if not server_rttm_paths and not rttm_items:
            return self.redirect(
                environ,
                "/fine-tuning",
                message=self.notification_message(
                    "Training RTTM is required.",
                    "For audio that has not been diarized, open Training Labels, type the speaker-time labels, and mark it complete first.",
                ),
                status="error",
            )

        server_selection_used = bool(server_audio_paths or server_rttm_paths or server_transcript_paths)
        if server_selection_used:
            if not server_audio_paths:
                failed_samples.append("SSH selection: choose at least one audio file from audio_in/.")
            elif not server_rttm_paths:
                failed_samples.append("SSH selection: choose at least one RTTM file.")
            else:
                rttm_by_stem, rttm_errors = self.workspace_paths_by_stem(server_rttm_paths, label="RTTM")
                transcript_by_stem, transcript_errors = self.workspace_paths_by_stem(server_transcript_paths, label="transcript")
                if rttm_errors or transcript_errors:
                    return self.redirect(
                        environ,
                        "/fine-tuning",
                        message=self.notification_message(
                            "SSH training file selections need unique filename stems.",
                            *rttm_errors,
                            *transcript_errors,
                        ),
                        status="error",
                    )
                single_pair_fallback = len(server_audio_paths) == 1 and len(server_rttm_paths) == 1
                used_rttm_stems: set[str] = set()
                used_transcript_stems: set[str] = set()
                for audio_path in server_audio_paths:
                    audio_stem = audio_path.stem
                    rttm_path = rttm_by_stem.get(audio_stem)
                    if rttm_path is None and single_pair_fallback:
                        rttm_path = server_rttm_paths[0]
                    if rttm_path is None:
                        failed_samples.append(f"{self.describe_path(audio_path)}: no RTTM file with matching stem.")
                        continue
                    transcript_path = transcript_by_stem.get(audio_stem)
                    if transcript_path is None and len(server_audio_paths) == 1 and len(server_transcript_paths) == 1:
                        transcript_path = server_transcript_paths[0]
                    try:
                        saved_samples.append(
                            self.save_project_sample_from_paths(
                                project_name=project_name,
                                backend=backend,
                                audio_path=audio_path,
                                rttm_path=rttm_path,
                                transcript_path=transcript_path,
                                transcript_text=transcript_text,
                            )
                        )
                        used_rttm_stems.add(rttm_path.stem)
                        if transcript_path is not None:
                            used_transcript_stems.add(transcript_path.stem)
                    except Exception as exc:
                        failed_samples.append(f"{self.describe_path(audio_path)}: {exc}")
                unused_rttm_stems = sorted(set(rttm_by_stem) - used_rttm_stems)
                unused_transcript_stems = sorted(set(transcript_by_stem) - used_transcript_stems)
                if unused_rttm_stems:
                    notes.append(f"Unused selected RTTM file(s): {', '.join(unused_rttm_stems[:8])}.")
                if unused_transcript_stems:
                    notes.append(f"Unused selected transcript file(s): {', '.join(unused_transcript_stems[:8])}.")

        browser_upload_used = bool(audio_items or rttm_items or transcript_items)
        if browser_upload_used:
            if not audio_items:
                failed_samples.append("Browser upload: training audio is required.")
            elif not rttm_items:
                failed_samples.append("Browser upload: training RTTM is required.")
            else:
                _audio_by_stem, audio_errors = self.upload_items_by_stem(audio_items, label="audio")
                rttm_by_stem, rttm_errors = self.upload_items_by_stem(rttm_items, label="RTTM")
                transcript_by_stem, transcript_errors = self.upload_items_by_stem(transcript_items, label="transcript")
                pairing_errors = [*audio_errors, *rttm_errors, *transcript_errors]
                if pairing_errors:
                    return self.redirect(
                        environ,
                        "/fine-tuning",
                        message=self.notification_message("Batch upload needs unique filenames.", *pairing_errors),
                        status="error",
                    )

                single_pair_fallback = len(audio_items) == 1 and len(rttm_items) == 1
                used_rttm_stems: set[str] = set()
                used_transcript_stems: set[str] = set()
                for audio_item in audio_items:
                    audio_stem = self.upload_stem(audio_item)
                    rttm_item = rttm_by_stem.get(audio_stem)
                    if rttm_item is None and single_pair_fallback:
                        rttm_item = rttm_items[0]
                    if rttm_item is None:
                        failed_samples.append(f"{getattr(audio_item, 'filename', 'audio')}: no RTTM file with matching stem.")
                        continue
                    transcript_item = transcript_by_stem.get(audio_stem)
                    if transcript_item is None and len(audio_items) == 1 and len(transcript_items) == 1:
                        transcript_item = transcript_items[0]
                    try:
                        saved_samples.append(
                            save_project_sample_streams(
                                project_name=project_name,
                                backend=backend,
                                audio_name=str(audio_item.filename),
                                audio_stream=audio_item.file,
                                rttm_name=str(rttm_item.filename),
                                rttm_stream=rttm_item.file,
                                transcript_name=str(getattr(transcript_item, "filename", "")) if transcript_item is not None else None,
                                transcript_stream=transcript_item.file if transcript_item is not None else None,
                                transcript_text=transcript_text,
                                root=self.root,
                            )
                        )
                        used_rttm_stems.add(self.upload_stem(rttm_item))
                        if transcript_item is not None:
                            used_transcript_stems.add(self.upload_stem(transcript_item))
                    except Exception as exc:
                        failed_samples.append(f"{getattr(audio_item, 'filename', 'audio')}: {exc}")

                unused_rttm_stems = sorted(set(rttm_by_stem) - used_rttm_stems)
                unused_transcript_stems = sorted(set(transcript_by_stem) - used_transcript_stems)
                if unused_rttm_stems:
                    notes.append(f"Unused uploaded RTTM file(s): {', '.join(unused_rttm_stems[:8])}.")
                if unused_transcript_stems:
                    notes.append(f"Unused uploaded transcript file(s): {', '.join(unused_transcript_stems[:8])}.")
        if failed_samples:
            notes.extend(failed_samples[:8])
        if not saved_samples:
            return self.redirect(
                environ,
                "/fine-tuning",
                message=self.notification_message("No fine-tuning samples were added.", *notes),
                status="error",
            )

        saved_names = ", ".join(sample.stem for sample in saved_samples[:6])
        if len(saved_samples) > 6:
            saved_names += f", and {len(saved_samples) - 6} more"
        status = "info" if failed_samples or notes else "success"
        message = self.notification_message(
            f"Added {len(saved_samples)} training sample(s) to {backend}/{project_name}.",
            f"Saved sample(s): {saved_names}.",
            *notes,
        )
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/fine-tuning",
            message=message,
            status=status,
        )

    def handle_finetune_prepare(self, environ):
        """Prepare manifests and launch scripts for one fine-tuning project."""

        form = self.parse_form(environ)
        preferences = self.model_preferences()
        pyannote_defaults = preferences["pyannote_fine_tuning"]
        nemo_defaults = preferences["nemo_fine_tuning"]
        project_name = (form.getfirst("prepare_project_name") or "").strip()
        if not project_name:
            return self.redirect(environ, "/fine-tuning", message="Provide a project name to prepare fine-tuning.", status="error")
        backend = normalize_backend(form.getfirst("prepare_backend") or str(preferences["fine_tuning_backend"]))

        artifacts = prepare_project(
            project_name=project_name,
            backend=backend,
            train_ratio=self.parse_float(form.getfirst("train_ratio"), float(nemo_defaults.get("train_ratio", DEFAULT_TRAIN_RATIO)), "train_ratio"),
            base_window=self.parse_float(form.getfirst("base_window"), float(nemo_defaults.get("base_window", DEFAULT_BASE_WINDOW)), "base_window"),
            base_shift=self.parse_float(form.getfirst("base_shift"), float(nemo_defaults.get("base_shift", DEFAULT_BASE_SHIFT)), "base_shift"),
            step_count=self.parse_int(form.getfirst("step_count"), int(nemo_defaults.get("step_count", DEFAULT_STEP_COUNT)), "step_count"),
            config_name=form.getfirst("config_name") or str(nemo_defaults.get("config_name", DEFAULT_CONFIG_NAME)),
            speaker_model=form.getfirst("speaker_model") or str(nemo_defaults.get("speaker_model", DEFAULT_SPEAKER_MODEL)),
            devices=self.parse_int(
                form.getfirst("devices"),
                int(
                    pyannote_defaults.get("devices", DEFAULT_DEVICES)
                    if backend == "pyannote"
                    else nemo_defaults.get("devices", DEFAULT_DEVICES)
                ),
                "devices",
            ),
            max_epochs=self.parse_int(
                form.getfirst("max_epochs"),
                int(
                    pyannote_defaults.get("max_epochs", DEFAULT_MAX_EPOCHS)
                    if backend == "pyannote"
                    else nemo_defaults.get("max_epochs", DEFAULT_MAX_EPOCHS)
                ),
                "max_epochs",
            ),
            slurm_partition=form.getfirst("slurm_partition") or DEFAULT_SLURM_PARTITION,
            slurm_time=form.getfirst("slurm_time") or DEFAULT_SLURM_TIME,
            slurm_memory=form.getfirst("slurm_memory") or DEFAULT_SLURM_MEMORY,
            slurm_cpus=self.parse_int(form.getfirst("slurm_cpus"), DEFAULT_SLURM_CPUS, "slurm_cpus"),
            slurm_gpus=self.parse_int(form.getfirst("slurm_gpus"), DEFAULT_SLURM_GPUS, "slurm_gpus"),
            nemo_root=self.resolve_local_path(form.getfirst("nemo_root")) if (form.getfirst("nemo_root") or "").strip() else None,
            pyannote_pretrained_model=form.getfirst("pyannote_pretrained_model") or str(pyannote_defaults.get("pretrained_model", DEFAULT_PYANNOTE_PRETRAINED_MODEL)),
            pyannote_duration=self.parse_float(form.getfirst("pyannote_duration"), float(pyannote_defaults.get("duration", DEFAULT_PYANNOTE_DURATION)), "pyannote_duration"),
            pyannote_max_speakers_per_chunk=self.parse_int(form.getfirst("pyannote_max_speakers_per_chunk"), int(pyannote_defaults.get("max_speakers_per_chunk", DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK)), "pyannote_max_speakers_per_chunk"),
            pyannote_max_speakers_per_frame=self.parse_int(form.getfirst("pyannote_max_speakers_per_frame"), int(pyannote_defaults.get("max_speakers_per_frame", DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME)), "pyannote_max_speakers_per_frame"),
            root=self.root,
        )
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/fine-tuning",
            message=f"Prepared {artifacts.backend}/{artifacts.project_slug} with {artifacts.sample_count} sample(s).",
            status="success",
        )

    def handle_finetune_launch(self, environ):
        """Launch a prepared fine-tuning project locally or via Slurm."""

        form = self.parse_form(environ)
        preferences = self.model_preferences()
        project_name = (form.getfirst("launch_project_name") or "").strip()
        if not project_name:
            return self.redirect(environ, "/fine-tuning", message="Provide a project name to launch fine-tuning.", status="error")
        backend = normalize_backend(form.getfirst("launch_backend") or str(preferences["fine_tuning_backend"]))

        nemo_root_value = (form.getfirst("launch_nemo_root") or "").strip()
        extra_env: dict[str, str] = {}
        if backend == "pyannote":
            hf_token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                or self.dashboard_secret("HF_TOKEN")
                or self.dashboard_secret("HUGGINGFACE_TOKEN")
                or self.dashboard_secret("HUGGINGFACE_HUB_TOKEN")
            )
            if hf_token:
                extra_env["HF_TOKEN"] = hf_token
                extra_env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        import workflow_dashboard as _wd  # lazy: honor monkey-patches in tests
        run = _wd.launch_training(
            project_name=project_name,
            backend=backend,
            nemo_root=self.resolve_local_path(nemo_root_value) if nemo_root_value else None,
            python_bin=form.getfirst("launch_python_bin") or sys.executable,
            prefer_sbatch=not bool(form.getfirst("launch_local")),
            version_name=(form.getfirst("launch_version_name") or "").strip(),
            extra_env=extra_env,
            root=self.root,
        )
        self.invalidate_dashboard_cache()
        launch_message = (
            f"Submitted {backend} fine-tuning version '{run.version_name}' as Slurm job {run.job_id}."
            if run.job_id
            else f"Launched {backend} fine-tuning version '{run.version_name}' (pid {run.pid})."
        )
        return self.redirect(
            environ,
            "/fine-tuning",
            message=launch_message,
            status="success",
        )

    def training_source_files(self) -> dict[str, list[Path]]:
        """Return server-side label and transcript files that can be paired for training."""

        return self.cached_value(
            "training_source_files",
            ttl_seconds=3.0,
            builder=lambda: {
                "rttm_files": self.newest_files(
                    search_roots=[
                        self.training_label_work_dir,
                        self.root / "fine_tuning" / "projects",
                        self.outputs_root,
                        self.root / "job_outputs",
                    ],
                    suffixes=TRAINING_RTTM_SUFFIXES,
                    limit=250,
                    exclude_dir_names=TRAINING_SOURCE_SCAN_EXCLUDE_DIRS,
                ),
                "transcript_files": self.newest_files(
                    search_roots=[
                        self.root / "fine_tuning" / "projects",
                        self.outputs_root,
                        self.root / "job_outputs",
                    ],
                    suffixes=TRAINING_TRANSCRIPT_SUFFIXES,
                    limit=250,
                    exclude_dir_names=TRAINING_SOURCE_SCAN_EXCLUDE_DIRS,
                ),
            },
        )

    def fine_tuning_project_metrics(self, project: dict[str, object]) -> dict[str, object]:
        """Calculate presentation-ready dataset metrics for one fine-tuning project."""

        project_path = Path(str(project.get("path", "")))
        audio_dir = project_path / "audio"
        rttm_dir = project_path / "rttm"
        metadata = project.get("metadata") or {}
        metadata_samples = {
            str(sample.get("stem", "")): sample
            for sample in (metadata.get("samples") or [])
            if isinstance(sample, dict) and str(sample.get("stem", "")).strip()
        } if isinstance(metadata, dict) else {}
        metrics: dict[str, object] = {
            "sample_count": 0,
            "total_audio_seconds": 0.0,
            "total_speech_seconds": 0.0,
            "total_active_speech_seconds": 0.0,
            "total_overlap_seconds": 0.0,
            "total_non_speech_seconds": 0.0,
            "total_segments": 0,
            "speaker_labels": [],
            "max_speakers_per_sample": 0,
            "max_concurrent_speakers": 0,
            "samples_with_rttm": 0,
            "samples_missing_rttm": 0,
            "average_speakers_per_sample": 0.0,
            "average_segment_seconds": 0.0,
            "speech_coverage": 0.0,
            "overlap_coverage": 0.0,
            "speaker_turns_per_minute": 0.0,
            "dominant_speaker_share": 0.0,
            "actual_train_ratio": 0.0,
        }
        if not audio_dir.is_dir():
            return metrics

        speaker_labels: set[str] = set()
        speaker_seconds: dict[str, float] = {}
        total_audio_seconds = 0.0
        total_speech_seconds = 0.0
        total_active_speech_seconds = 0.0
        total_overlap_seconds = 0.0
        total_segments = 0
        max_speakers_per_sample = 0
        max_concurrent_speakers = 0
        total_speakers_per_sample = 0
        samples_with_rttm = 0
        sample_count = 0
        for audio_path in sorted(audio_dir.iterdir(), key=lambda item: item.name.lower()):
            if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            sample_count += 1
            metadata_sample = metadata_samples.get(audio_path.stem, {})
            duration_seconds = 0.0
            try:
                duration_seconds = float(metadata_sample.get("duration_seconds", 0.0)) if metadata_sample else 0.0
            except (TypeError, ValueError):
                duration_seconds = 0.0
            if duration_seconds <= 0:
                try:
                    duration_seconds = float(probe_media_duration(audio_path))
                except Exception:
                    duration_seconds = 0.0

            rttm_path = rttm_dir / f"{audio_path.stem}.rttm"
            segments = []
            if rttm_path.is_file():
                try:
                    segments = parse_rttm(rttm_path)
                except Exception:
                    segments = []
            if segments:
                samples_with_rttm += 1
                segment_speakers = {segment.speaker for segment in segments}
                speaker_labels.update(segment_speakers)
                max_speakers_per_sample = max(max_speakers_per_sample, len(segment_speakers))
                total_speakers_per_sample += len(segment_speakers)
                total_segments += len(segments)
                speech_seconds = sum(segment.duration for segment in segments)
                total_speech_seconds += speech_seconds
                for segment in segments:
                    speaker_seconds[segment.speaker] = speaker_seconds.get(segment.speaker, 0.0) + segment.duration
                max_segment_end = max(segment.end for segment in segments)
                if duration_seconds <= 0 or max_segment_end > duration_seconds:
                    duration_seconds = max_segment_end
                active_seconds, overlap_seconds, concurrent_speakers = self.segment_activity_stats(segments)
                total_active_speech_seconds += active_seconds
                total_overlap_seconds += overlap_seconds
                max_concurrent_speakers = max(max_concurrent_speakers, concurrent_speakers)
            total_audio_seconds += duration_seconds

        avg_segment_seconds = total_speech_seconds / total_segments if total_segments else 0.0
        speech_coverage = total_active_speech_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0
        overlap_coverage = (
            total_overlap_seconds / total_active_speech_seconds
            if total_active_speech_seconds > 0
            else 0.0
        )
        total_non_speech_seconds = max(total_audio_seconds - total_active_speech_seconds, 0.0)
        speaker_turns_per_minute = (
            total_segments / (total_audio_seconds / 60.0)
            if total_audio_seconds > 0
            else 0.0
        )
        dominant_speaker_share = (
            max(speaker_seconds.values()) / total_speech_seconds
            if speaker_seconds and total_speech_seconds > 0
            else 0.0
        )

        def metadata_int(key: str) -> int:
            if not isinstance(metadata, dict):
                return 0
            try:
                return int(metadata.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0

        train_count = metadata_int("train_count")
        validation_count = metadata_int("validation_count")
        split_total = train_count + validation_count

        return {
            "sample_count": sample_count,
            "samples_with_rttm": samples_with_rttm,
            "samples_missing_rttm": max(sample_count - samples_with_rttm, 0),
            "total_audio_seconds": round(total_audio_seconds, 3),
            "total_speech_seconds": round(total_speech_seconds, 3),
            "total_active_speech_seconds": round(total_active_speech_seconds, 3),
            "total_overlap_seconds": round(total_overlap_seconds, 3),
            "total_non_speech_seconds": round(total_non_speech_seconds, 3),
            "total_segments": total_segments,
            "speaker_labels": sorted(speaker_labels),
            "unique_speaker_labels": len(speaker_labels),
            "max_speakers_per_sample": max_speakers_per_sample,
            "max_concurrent_speakers": max_concurrent_speakers,
            "average_speakers_per_sample": round(total_speakers_per_sample / samples_with_rttm, 2) if samples_with_rttm else 0.0,
            "average_segments_per_sample": round(total_segments / sample_count, 2) if sample_count else 0.0,
            "average_segment_seconds": round(avg_segment_seconds, 3),
            "speech_coverage": round(speech_coverage, 4),
            "overlap_coverage": round(overlap_coverage, 4),
            "speaker_turns_per_minute": round(speaker_turns_per_minute, 2),
            "dominant_speaker_share": round(dominant_speaker_share, 4),
            "train_count": train_count,
            "validation_count": validation_count,
            "actual_train_ratio": round(train_count / split_total, 4) if split_total else 0.0,
        }

    def segment_activity_stats(self, segments: list[object]) -> tuple[float, float, int]:
        """Return active speech, overlapped speech, and peak concurrency for RTTM rows."""

        events: list[tuple[float, int]] = []
        for segment in segments:
            try:
                start = max(float(segment.start), 0.0)
                end = max(float(segment.end), start)
            except (TypeError, ValueError, AttributeError):
                continue
            if end <= start:
                continue
            events.append((start, 1))
            events.append((end, -1))
        if not events:
            return 0.0, 0.0, 0

        events.sort(key=lambda event: (event[0], event[1]))
        active_count = 0
        max_concurrent = 0
        previous_time: float | None = None
        active_seconds = 0.0
        overlap_seconds = 0.0
        for timestamp, delta in events:
            if previous_time is not None and timestamp > previous_time:
                span = timestamp - previous_time
                if active_count > 0:
                    active_seconds += span
                if active_count > 1:
                    overlap_seconds += span
            active_count += delta
            max_concurrent = max(max_concurrent, active_count)
            previous_time = timestamp
        return active_seconds, overlap_seconds, max_concurrent

    def fine_tuning_summary(self, projects: list[dict[str, object]]) -> dict[str, object]:
        """Summarize the current fine-tuning workspace in a few stable counts."""

        sample_projects = [project for project in projects if int(project.get("sample_count", 0)) > 0]
        prepared_projects = [project for project in sample_projects if bool(project.get("prepared"))]
        running_statuses = {"running", "submitted", "pending", "configuring"}
        active_runs = [
            project
            for project in prepared_projects
            if str((project.get("latest_run") or {}).get("status", "")).lower() in running_statuses
        ]
        project_metrics = [self.fine_tuning_project_metrics(project) for project in sample_projects]
        speaker_labels = sorted(
            {
                str(label)
                for metrics in project_metrics
                for label in list(metrics.get("speaker_labels", []))
                if str(label).strip()
            }
        )
        total_samples = sum(int(metrics.get("sample_count", 0)) for metrics in project_metrics)
        total_segments = sum(int(metrics.get("total_segments", 0)) for metrics in project_metrics)
        total_speech_seconds = sum(float(metrics.get("total_speech_seconds", 0.0)) for metrics in project_metrics)
        total_active_speech_seconds = sum(float(metrics.get("total_active_speech_seconds", 0.0)) for metrics in project_metrics)
        total_overlap_seconds = sum(float(metrics.get("total_overlap_seconds", 0.0)) for metrics in project_metrics)
        total_non_speech_seconds = sum(float(metrics.get("total_non_speech_seconds", 0.0)) for metrics in project_metrics)
        total_audio_seconds = sum(float(metrics.get("total_audio_seconds", 0.0)) for metrics in project_metrics)
        samples_with_rttm = sum(int(metrics.get("samples_with_rttm", 0)) for metrics in project_metrics)
        weighted_speaker_sample_total = sum(
            float(metrics.get("average_speakers_per_sample", 0.0)) * int(metrics.get("samples_with_rttm", 0))
            for metrics in project_metrics
        )
        return {
            "projects_with_samples": len(sample_projects),
            "prepared_projects": len(prepared_projects),
            "active_runs": len(active_runs),
            "total_samples": total_samples,
            "total_segments": total_segments,
            "total_speech_seconds": round(total_speech_seconds, 3),
            "total_active_speech_seconds": round(total_active_speech_seconds, 3),
            "total_overlap_seconds": round(total_overlap_seconds, 3),
            "total_non_speech_seconds": round(total_non_speech_seconds, 3),
            "total_audio_seconds": round(total_audio_seconds, 3),
            "unique_speaker_labels": len(speaker_labels),
            "max_concurrent_speakers": max((int(metrics.get("max_concurrent_speakers", 0)) for metrics in project_metrics), default=0),
            "speech_coverage": round(total_active_speech_seconds / total_audio_seconds, 4) if total_audio_seconds > 0 else 0.0,
            "overlap_coverage": round(total_overlap_seconds / total_active_speech_seconds, 4) if total_active_speech_seconds > 0 else 0.0,
            "average_segments_per_sample": round(total_segments / total_samples, 2) if total_samples else 0.0,
            "average_speakers_per_sample": round(weighted_speaker_sample_total / samples_with_rttm, 2) if samples_with_rttm else 0.0,
            "speaker_turns_per_minute": round(total_segments / (total_audio_seconds / 60.0), 2) if total_audio_seconds > 0 else 0.0,
        }
