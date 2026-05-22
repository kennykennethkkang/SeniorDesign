#!/usr/bin/env python3
"""Fine-tuning mixin: project prep, training launch, rename, and DER/JER scoring.

This is the longest mixin, and intentionally so; it owns the entire fine-tune flow
the dashboard exposes. Everything from "create a project shell" through
"score a checkpoint against a reference RTTM and surface the metric" lives
here. The actual training and metric implementations are imported from
``fine_tuning_manager`` and ``diarization_metrics``; this file is the WSGI
glue that drives them and serves the result back to the React frontend.
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

import fine_tuning_manager as ftm
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
    delete_run_model_artifacts,
    ensure_project_structure,
    launch_training,
    build_sample,
    list_projects,
    normalize_backend,
    parse_rttm,
    probe_media_duration,
    prepare_project,
    project_dir as fine_tune_project_dir,
    run_status as fine_tuning_run_status,
    sanitize_filename,
    save_project_sample_links,
    save_project_sample_streams,
    set_project_auto_train,
    set_project_display_name,
    set_run_display_name,
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
    """Handles all fine-tuning HTTP routes: upload, prepare, launch, rename, and score."""

    FINE_TUNE_PREPARE_FIELD_NAMES = {
        "train_ratio",
        "base_window",
        "base_shift",
        "step_count",
        "config_name",
        "base_model",
        "speaker_model",
        "devices",
        "max_epochs",
        "slurm_partition",
        "slurm_time",
        "slurm_memory",
        "slurm_cpus",
        "slurm_gpus",
        "nemo_root",
        "pyannote_pretrained_model",
        "pyannote_duration",
        "pyannote_max_speakers_per_chunk",
        "pyannote_max_speakers_per_frame",
    }

    def fine_tune_audio_sample_count(self, project_name: str, backend: str) -> int:
        """Count uploaded audio files for one project/backend without preparing artifacts."""

        project_path = fine_tune_project_dir(project_name, backend=backend, root=self.root)
        audio_dir = project_path / "audio"
        if not audio_dir.is_dir():
            return 0
        return sum(
            1
            for path in audio_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )

    def resolve_fine_tune_backend_for_samples(self, project_name: str, requested_backend: str) -> str:
        """Use the backend that actually contains samples when the form backend is stale."""

        normalized_backend = normalize_backend(requested_backend)
        if self.fine_tune_audio_sample_count(project_name, normalized_backend) > 0:
            return normalized_backend

        sampled_backends = [
            backend
            for backend in sorted(ftm.SUPPORTED_FINE_TUNING_BACKENDS)
            if backend != normalized_backend
            and self.fine_tune_audio_sample_count(project_name, backend) > 0
        ]
        if len(sampled_backends) == 1:
            return sampled_backends[0]
        return normalized_backend

    def fine_tune_prepare_form_has_options(self, form) -> bool:
        """Return whether this submission came from the prepare form rather than a compact project-card button."""

        return any(
            (form.getfirst(field_name) or "").strip()
            for field_name in self.FINE_TUNE_PREPARE_FIELD_NAMES
        )

    def fine_tune_prepare_options_from_form(self, form, backend: str) -> dict[str, object]:
        """Parse the manual prepare form into prepare_project keyword arguments."""

        preferences = self.model_preferences()
        pyannote_defaults = preferences["pyannote_fine_tuning"]
        nemo_defaults = preferences["nemo_fine_tuning"]
        base_model = (form.getfirst("base_model") or "").strip()
        return {
            "train_ratio": self.parse_float(form.getfirst("train_ratio"), float(nemo_defaults.get("train_ratio", DEFAULT_TRAIN_RATIO)), "train_ratio"),
            "base_window": self.parse_float(form.getfirst("base_window"), float(nemo_defaults.get("base_window", DEFAULT_BASE_WINDOW)), "base_window"),
            "base_shift": self.parse_float(form.getfirst("base_shift"), float(nemo_defaults.get("base_shift", DEFAULT_BASE_SHIFT)), "base_shift"),
            "step_count": self.parse_int(form.getfirst("step_count"), int(nemo_defaults.get("step_count", DEFAULT_STEP_COUNT)), "step_count"),
            "config_name": form.getfirst("config_name") or str(nemo_defaults.get("config_name", DEFAULT_CONFIG_NAME)),
            "speaker_model": (
                base_model
                if backend == "nemo" and base_model
                else form.getfirst("speaker_model")
                or str(nemo_defaults.get("speaker_model", DEFAULT_SPEAKER_MODEL))
            ),
            "devices": self.parse_int(
                form.getfirst("devices"),
                int(
                    pyannote_defaults.get("devices", DEFAULT_DEVICES)
                    if backend == "pyannote"
                    else nemo_defaults.get("devices", DEFAULT_DEVICES)
                ),
                "devices",
            ),
            "max_epochs": self.parse_int(
                form.getfirst("max_epochs"),
                int(
                    pyannote_defaults.get("max_epochs", DEFAULT_MAX_EPOCHS)
                    if backend == "pyannote"
                    else nemo_defaults.get("max_epochs", DEFAULT_MAX_EPOCHS)
                ),
                "max_epochs",
            ),
            "slurm_partition": form.getfirst("slurm_partition") or DEFAULT_SLURM_PARTITION,
            "slurm_time": form.getfirst("slurm_time") or DEFAULT_SLURM_TIME,
            "slurm_memory": form.getfirst("slurm_memory") or DEFAULT_SLURM_MEMORY,
            "slurm_cpus": self.parse_int(form.getfirst("slurm_cpus"), DEFAULT_SLURM_CPUS, "slurm_cpus"),
            "slurm_gpus": self.parse_int(form.getfirst("slurm_gpus"), DEFAULT_SLURM_GPUS, "slurm_gpus"),
            "nemo_root": self.resolve_local_path(form.getfirst("nemo_root")) if (form.getfirst("nemo_root") or "").strip() else None,
            "pyannote_pretrained_model": (
                base_model
                if backend == "pyannote" and base_model
                else form.getfirst("pyannote_pretrained_model")
                or str(pyannote_defaults.get("pretrained_model", DEFAULT_PYANNOTE_PRETRAINED_MODEL))
            ),
            "pyannote_duration": self.parse_float(form.getfirst("pyannote_duration"), float(pyannote_defaults.get("duration", DEFAULT_PYANNOTE_DURATION)), "pyannote_duration"),
            "pyannote_max_speakers_per_chunk": self.parse_int(form.getfirst("pyannote_max_speakers_per_chunk"), int(pyannote_defaults.get("max_speakers_per_chunk", DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK)), "pyannote_max_speakers_per_chunk"),
            "pyannote_max_speakers_per_frame": self.parse_int(form.getfirst("pyannote_max_speakers_per_frame"), int(pyannote_defaults.get("max_speakers_per_frame", DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME)), "pyannote_max_speakers_per_frame"),
        }

    def save_project_sample_from_paths(
        self,
        *,
        project_name: str,
        backend: str,
        audio_path: Path,
        rttm_path: Path,
        transcript_path: Path | None = None,
        transcript_text: str = "",
        link_audio: bool = False,
    ):
        """Bring a labeled audio+RTTM pair into a project's training data dir.

        ``link_audio`` swaps the audio/transcript byte-copy for a symlink so
        a project can reuse the original media already on disk. The RTTM
        always gets rewritten in canonical form regardless of the link
        choice; it's small and the trainers depend on the normalized
        session_id.
        """

        if link_audio:
            return save_project_sample_links(
                project_name=project_name,
                backend=backend,
                audio_path=audio_path,
                rttm_path=rttm_path,
                transcript_path=transcript_path,
                transcript_text=transcript_text,
                root=self.root,
            )

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

    def handle_finetune_create_project(self, environ):
        """Create an empty fine-tuning project so samples can be added later.

        The upload form requires audio + RTTM to land before a project shell
        exists on disk, which is annoying when the user just wants to register
        a name (and optionally a friendlier display name) up front and feed it
        labels from the Inspect popup over time. This handler makes the
        directory structure and, if a display name is provided, writes the
        display sidecar so the project shows up in the Fine-Tuning tab and the
        popup's "existing models" list immediately.
        """

        form = self.parse_form(environ)
        preferences = self.model_preferences()
        project_name = (form.getfirst("project_name") or "").strip()
        if not project_name:
            return self.redirect(
                environ,
                "/fine-tuning",
                message="Provide a project name for the new fine-tuned model.",
                status="error",
            )
        try:
            backend = normalize_backend(
                form.getfirst("fine_tuning_backend") or str(preferences["fine_tuning_backend"])
            )
        except ValueError as exc:
            return self.redirect(environ, "/fine-tuning", message=str(exc), status="error")

        slug = slugify(project_name)
        if not slug:
            return self.redirect(
                environ,
                "/fine-tuning",
                message="Project name must contain at least one alphanumeric character.",
                status="error",
            )
        target = fine_tune_project_dir(slug, backend=backend, root=self.root)
        already_exists = target.is_dir() and any(target.iterdir())
        try:
            ensure_project_structure(slug, backend=backend, root=self.root)
        except OSError as exc:
            return self.redirect(environ, "/fine-tuning", message=f"Could not create project folder: {exc}", status="error")

        # Drop a marker into display.json so list_projects keeps showing this
        # project even with 0 samples; otherwise the empty shell gets filtered
        # out and the user wonders where their freshly-created model went.
        display_path = target / "display.json"
        sidecar_updates: dict[str, object] = {
            "manually_created": True,
            "manually_created_at_utc": utc_now_iso(),
        }
        display_name = (form.getfirst("display_name") or "").strip()
        if display_name:
            sidecar_updates["display_name"] = display_name
        try:
            ftm._write_display_sidecar(display_path, updates=sidecar_updates)  # noqa: SLF001 - internal helper, intentional cross-module use
        except OSError as exc:
            return self.redirect(environ, "/fine-tuning", message=f"Could not write display.json: {exc}", status="error")

        self.invalidate_dashboard_cache()
        if already_exists:
            message = (
                f"Project '{display_name or slug}' already exists for {backend}; left it untouched."
                if not display_name
                else f"Updated display name for existing {backend}/{slug} to '{display_name}'."
            )
            status = "info"
        else:
            label = display_name or slug
            message = f"Created empty {backend} project '{label}'. Add samples here, from Training Labels, or via the Inspect popup."
            status = "success"
        return self.redirect(environ, "/fine-tuning", message=message, status=status)

    def handle_finetune_upload(self, environ):
        """Accept one or more labeled audio+RTTM pairs and add them to one OR MORE projects.

        The form can submit ``training_targets`` repeatedly with values like
        ``pyannote/existing-one`` to fan a single sample selection out to
        several projects in one click. The legacy single ``project_name`` +
        ``fine_tuning_backend`` fields still work and act as an "additional /
        new project" target on top of any checklist picks.
        """

        form = self.parse_form(environ)
        preferences = self.model_preferences()
        manual_project_name = (form.getfirst("project_name") or "").strip()
        manual_backend_value = form.getfirst("fine_tuning_backend") or str(preferences["fine_tuning_backend"])

        explicit_targets, target_errors = self.training_label_targets_from_values(form.getlist("training_targets"))
        if target_errors:
            return self.redirect(
                environ,
                "/fine-tuning",
                message=self.notification_message("One of the selected fine-tuning projects is malformed.", *target_errors[:8]),
                status="error",
            )

        # Build a deterministic, de-duplicated list of (backend, project)
        # targets: checklist picks first, then the manual "new/additional"
        # project last. ``backend=both`` fans out to nemo + pyannote.
        upload_targets: list[tuple[str, str]] = []
        seen_keys: set[str] = set()
        for target in explicit_targets:
            key = f"{target['backend']}/{target['project_name']}"
            if key in seen_keys:
                continue
            upload_targets.append((target["backend"], target["project_name"]))
            seen_keys.add(key)

        if manual_project_name:
            try:
                manual_backends = self.training_label_target_backends(manual_backend_value)
            except ValueError as exc:
                return self.redirect(environ, "/fine-tuning", message=str(exc), status="error")
            for backend_value in manual_backends:
                key = f"{backend_value}/{manual_project_name}"
                if key in seen_keys:
                    continue
                upload_targets.append((backend_value, manual_project_name))
                seen_keys.add(key)

        if not upload_targets:
            return self.redirect(
                environ,
                "/fine-tuning",
                message="Pick at least one existing fine-tuning project, or fill in the additional project name field.",
                status="error",
            )

        # Keep these populated for downstream messages and any single-project
        # validation paths. The save loop below fans each pair out to every
        # entry in ``upload_targets``.
        project_name = upload_targets[0][1]
        backend = upload_targets[0][0]

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
        # SSH-selected audio is always symlinked into the project so we don't
        # duplicate gigabytes of media on disk. Browser uploads still copy
        # because the source is ephemeral form data, not a stable file on disk.
        link_existing_audio = True

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
                    audio_stems = self.training_audio_pair_stems(audio_path)
                    audio_stem = audio_stems[0] if audio_stems else audio_path.stem
                    rttm_path = next(
                        (rttm_by_stem[stem] for stem in audio_stems if stem in rttm_by_stem),
                        None,
                    )
                    if rttm_path is None and single_pair_fallback:
                        rttm_path = server_rttm_paths[0]
                    if rttm_path is None:
                        failed_samples.append(f"{self.describe_path(audio_path)}: no RTTM file with matching stem.")
                        continue
                    transcript_path = next(
                        (transcript_by_stem[stem] for stem in audio_stems if stem in transcript_by_stem),
                        None,
                    )
                    if transcript_path is None and len(server_audio_paths) == 1 and len(server_transcript_paths) == 1:
                        transcript_path = server_transcript_paths[0]
                    pair_succeeded = False
                    for target_backend, target_project in upload_targets:
                        try:
                            saved_samples.append(
                                self.save_project_sample_from_paths(
                                    project_name=target_project,
                                    backend=target_backend,
                                    audio_path=audio_path,
                                    rttm_path=rttm_path,
                                    transcript_path=transcript_path,
                                    transcript_text=transcript_text,
                                    link_audio=link_existing_audio,
                                )
                            )
                            pair_succeeded = True
                        except Exception as exc:
                            failed_samples.append(
                                f"{self.describe_path(audio_path)} -> {target_backend}/{target_project}: {exc}"
                            )
                    if pair_succeeded:
                        used_rttm_stems.add(rttm_path.stem)
                        if transcript_path is not None:
                            used_transcript_stems.add(transcript_path.stem)
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
                    # Browser uploads stream straight from the cgi.FieldStorage
                    # file handles, which can only be read once. To fan a
                    # single uploaded pair out to multiple projects, snapshot
                    # the audio + rttm bytes upfront and feed BytesIO copies
                    # to each save call.
                    audio_bytes = audio_item.file.read() if hasattr(audio_item.file, "read") else b""
                    rttm_bytes = rttm_item.file.read() if hasattr(rttm_item.file, "read") else b""
                    transcript_bytes = b""
                    if transcript_item is not None and hasattr(transcript_item.file, "read"):
                        transcript_bytes = transcript_item.file.read()
                    pair_succeeded = False
                    for target_backend, target_project in upload_targets:
                        try:
                            saved_samples.append(
                                save_project_sample_streams(
                                    project_name=target_project,
                                    backend=target_backend,
                                    audio_name=str(audio_item.filename),
                                    audio_stream=io.BytesIO(audio_bytes),
                                    rttm_name=str(rttm_item.filename),
                                    rttm_stream=io.BytesIO(rttm_bytes),
                                    transcript_name=str(getattr(transcript_item, "filename", "")) if transcript_item is not None else None,
                                    transcript_stream=io.BytesIO(transcript_bytes) if transcript_item is not None else None,
                                    transcript_text=transcript_text,
                                    root=self.root,
                                )
                            )
                            pair_succeeded = True
                        except Exception as exc:
                            failed_samples.append(
                                f"{getattr(audio_item, 'filename', 'audio')} -> {target_backend}/{target_project}: {exc}"
                            )
                    if pair_succeeded:
                        used_rttm_stems.add(self.upload_stem(rttm_item))
                        if transcript_item is not None:
                            used_transcript_stems.add(self.upload_stem(transcript_item))

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
        target_label = ", ".join(f"{b}/{p}" for b, p in upload_targets)
        # Each input pair is saved once per target, so the count divides
        # cleanly when every save succeeded.
        per_target = len(saved_samples) // max(len(upload_targets), 1)
        if len(upload_targets) > 1:
            headline = f"Added {len(saved_samples)} sample-copies ({per_target} pair(s) x {len(upload_targets)} project(s)) to {target_label}."
        else:
            headline = f"Added {len(saved_samples)} training sample(s) to {target_label}."
        message = self.notification_message(
            headline,
            f"Saved sample stem(s): {saved_names}.",
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
        project_name = (form.getfirst("prepare_project_name") or "").strip()
        if not project_name:
            return self.redirect(environ, "/fine-tuning", message="Provide a project name to prepare fine-tuning.", status="error")
        backend = self.resolve_fine_tune_backend_for_samples(
            project_name,
            normalize_backend(form.getfirst("prepare_backend") or str(preferences["fine_tuning_backend"])),
        )

        try:
            prepare_options = self.fine_tune_prepare_options_from_form(form, backend)
            artifacts = prepare_project(
                project_name=project_name,
                backend=backend,
                root=self.root,
                **prepare_options,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return self.redirect(
                environ,
                "/fine-tuning",
                message=f"Could not prepare {backend}/{project_name}: {exc}",
                status="error",
            )
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/fine-tuning",
            message=f"Prepared {artifacts.backend}/{artifacts.project_slug} with {artifacts.sample_count} sample(s).",
            status="success",
        )

    def fine_tune_launch_extra_env(self, backend: str) -> dict[str, str]:
        """Return backend-specific secrets for launch without exposing them to frontend state."""

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
        return extra_env

    def handle_finetune_prepare_launch(self, environ):
        """Prepare a project and immediately submit a Slurm-preferred training run."""

        form = self.parse_form(environ)
        preferences = self.model_preferences()
        project_name = (
            form.getfirst("project_slug")
            or form.getfirst("project_name")
            or form.getfirst("prepare_project_name")
            or ""
        ).strip()
        if not project_name:
            return self.redirect(environ, "/fine-tuning", message="Choose a project to prepare and submit.", status="error")
        backend = self.resolve_fine_tune_backend_for_samples(
            project_name,
            normalize_backend(form.getfirst("backend") or form.getfirst("prepare_backend") or str(preferences["fine_tuning_backend"])),
        )
        try:
            prepare_options = (
                self.fine_tune_prepare_options_from_form(form, backend)
                if self.fine_tune_prepare_form_has_options(form)
                else self.auto_train_prepare_options(project_name, backend)
            )
            artifacts = prepare_project(
                project_name=project_name,
                backend=backend,
                root=self.root,
                **prepare_options,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return self.redirect(
                environ,
                "/fine-tuning",
                message=f"Could not prepare {backend}/{project_name}: {exc}",
                status="error",
            )

        import workflow_dashboard as _wd  # lazy: honor monkey-patches in tests
        try:
            # Match the explicit-launch handler: when the prepare-and-launch
            # form skipped the NeMo root field, fall back to the detected
            # default so the auto-launch path doesn't 1-shot bail out.
            auto_nemo_root = ftm._detect_default_nemo_root() if backend == "nemo" else None
            run = _wd.launch_training(
                project_name=artifacts.project_slug,
                backend=backend,
                nemo_root=auto_nemo_root,
                python_bin=sys.executable,
                prefer_sbatch=True,
                version_name=(form.getfirst("launch_version_name") or "").strip(),
                extra_env=self.fine_tune_launch_extra_env(backend),
                root=self.root,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return self.redirect(
                environ,
                "/fine-tuning",
                message=f"Prepared {artifacts.backend}/{artifacts.project_slug} with {artifacts.sample_count} sample(s), but launch failed: {exc}",
                status="error",
            )

        self.invalidate_dashboard_cache()
        launch_message = (
            f"Prepared {artifacts.backend}/{artifacts.project_slug} with {artifacts.sample_count} sample(s) and submitted '{run.version_name}' as Slurm job {run.job_id}."
            if run.job_id
            else f"Prepared {artifacts.backend}/{artifacts.project_slug} with {artifacts.sample_count} sample(s) and launched '{run.version_name}' locally (pid {run.pid})."
        )
        return self.redirect(environ, "/fine-tuning", message=launch_message, status="success")

    def handle_finetune_launch(self, environ):
        """Launch a prepared fine-tuning project locally or via Slurm."""

        form = self.parse_form(environ)
        preferences = self.model_preferences()
        project_name = (form.getfirst("launch_project_name") or "").strip()
        if not project_name:
            return self.redirect(environ, "/fine-tuning", message="Provide a project name to launch fine-tuning.", status="error")
        backend = normalize_backend(form.getfirst("launch_backend") or str(preferences["fine_tuning_backend"]))

        nemo_root_value = (form.getfirst("launch_nemo_root") or "").strip()
        if nemo_root_value:
            resolved_nemo_root = self.resolve_local_path(nemo_root_value)
        else:
            # Fall back to whichever NeMo checkout the manager detects on disk
            # (matches the runbook's `/WAVE/.../NeMo` location). Without this,
            # users who don't paste a path get an immediate `Set NEMO_ROOT…`
            # bail-out from the launcher even when the clone is already there.
            resolved_nemo_root = ftm._detect_default_nemo_root() if backend == "nemo" else None
        import workflow_dashboard as _wd  # lazy: honor monkey-patches in tests
        run = _wd.launch_training(
            project_name=project_name,
            backend=backend,
            nemo_root=resolved_nemo_root,
            python_bin=form.getfirst("launch_python_bin") or sys.executable,
            prefer_sbatch=not bool(form.getfirst("launch_local")),
            version_name=(form.getfirst("launch_version_name") or "").strip(),
            extra_env=self.fine_tune_launch_extra_env(backend),
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

    def handle_finetune_rename_project(self, environ):
        """Update a project's display name without touching the on-disk slug or any existing run directories.

        The slug is the persistent ID; everything that references the project
        (label_status, metadata, runs/) uses it. Only the human-readable label
        in display.json changes here.
        """
        form = self.parse_form(environ)
        project_slug = (form.getfirst("project_slug") or form.getfirst("project_name") or "").strip()
        backend_value = form.getfirst("backend") or form.getfirst("project_backend") or ""
        display_name = (form.getfirst("display_name") or "").strip()
        if not project_slug:
            return self.redirect(environ, "/fine-tuning", message="Pick a fine-tuning project to rename.", status="error")
        if not display_name:
            return self.redirect(environ, "/fine-tuning", message="Provide a new display name.", status="error")
        try:
            backend = normalize_backend(backend_value)
            set_project_display_name(
                project_slug,
                backend=backend,
                display_name=display_name,
                root=self.root,
            )
        except (FileNotFoundError, ValueError) as exc:
            return self.redirect(environ, "/fine-tuning", message=str(exc), status="error")
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/fine-tuning",
            message=f"Renamed project to '{display_name}'.",
            status="success",
        )

    def handle_finetune_active_job_status(self, environ):
        """Return the status of the most recently submitted/running fine-tuning job across all projects."""

        import datetime as _dt
        from workflow_background import slurm_job_state

        active_statuses = {"running", "submitted", "pending", "configuring", "waiting"}
        projects = list_projects(root=self.root)
        found: dict[str, object] = {}
        for project in projects:
            runs_dir = Path(str(project.get("path", ""))) / "runs"
            if not runs_dir.is_dir():
                continue
            for run_dir in sorted(runs_dir.glob("*"), reverse=True):
                if not run_dir.is_dir():
                    continue
                meta_path = run_dir / "metadata.json"
                if not meta_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                job_id = str(meta.get("job_id") or "").strip()
                if not job_id:
                    continue
                raw_slurm = slurm_job_state(job_id)
                from fine_tuning_manager import slurm_state_to_run_status
                mapped = slurm_state_to_run_status(raw_slurm)
                if mapped not in active_statuses:
                    break  # runs are newest-first; first non-active means nothing running
                found = {
                    "jobId": job_id,
                    "slurmState": raw_slurm or "UNKNOWN",
                    "status": mapped,
                    "backend": str(meta.get("backend") or ""),
                    "projectName": str(meta.get("project_name") or ""),
                    "startedAt": str(meta.get("started_at_utc") or ""),
                    "stdoutPath": str(meta.get("stdout_path") or ""),
                    "maxEpochs": int(meta.get("max_epochs") or 0),
                }
                break
            if found:
                break

        if not found:
            return self.json_response("200 OK", {"active": False})

        # Parse epoch and step progress from the stdout log
        epoch = 0
        max_epochs = int(found.get("maxEpochs") or 0)
        current_step = 0
        total_steps = 0
        stdout_path = str(found.get("stdoutPath") or "")
        if stdout_path:
            log_path = Path(stdout_path)
            if not log_path.is_file():
                # try the slurm log via job_id
                slurm_logs = Path(str(found.get("stdoutPath") or "")).parent
                candidates = sorted(slurm_logs.glob(f"*_{found['jobId']}.out")) if slurm_logs.is_dir() else []
                if candidates:
                    log_path = candidates[-1]
            if log_path.is_file():
                try:
                    # Read last 8 KB, enough for recent epoch lines
                    with log_path.open("rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 8192))
                        tail = f.read().decode("utf-8", errors="replace")
                    nemo_epoch_re = re.compile(r"Epoch\s+(\d+)[\s:/,]", re.IGNORECASE)
                    # NeMo tqdm bar: "| 11/529 ["
                    nemo_step_re = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
                    pyannote_epoch_re = re.compile(r"\bepoch[=\s]+(\d+)/(\d+)", re.IGNORECASE)
                    # Pyannote step: "step=100/500" or "step 100/500"
                    pyannote_step_re = re.compile(r"\bstep[=\s]+(\d+)/(\d+)", re.IGNORECASE)
                    epoch_found = False
                    step_found = False
                    for line in reversed(tail.splitlines()):
                        if not epoch_found:
                            m = pyannote_epoch_re.search(line)
                            if m:
                                epoch = int(m.group(1))
                                if not max_epochs:
                                    max_epochs = int(m.group(2))
                                epoch_found = True
                            else:
                                m = nemo_epoch_re.search(line)
                                if m:
                                    epoch = int(m.group(1))
                                    epoch_found = True
                        if not step_found:
                            m = nemo_step_re.search(line)
                            if m:
                                current_step = int(m.group(1))
                                total_steps = int(m.group(2))
                                step_found = True
                            else:
                                m = pyannote_step_re.search(line)
                                if m:
                                    current_step = int(m.group(1))
                                    total_steps = int(m.group(2))
                                    step_found = True
                        if epoch_found and step_found:
                            break
                except Exception:
                    pass

        # Compute elapsed seconds
        elapsed = 0
        started_at = str(found.get("startedAt") or "")
        if started_at:
            try:
                started = _dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                now = _dt.datetime.now(_dt.timezone.utc)
                elapsed = max(0, int((now - started).total_seconds()))
            except Exception:
                pass

        # Estimated start time for pending/submitted jobs (squeue --start)
        estimated_start_iso = ""
        if found["status"] in {"submitted", "pending", "configuring", "waiting"}:
            try:
                result = subprocess.run(
                    ["squeue", "--job", found["jobId"], "--start", "--format=%S", "--noheader"],
                    capture_output=True, text=True, timeout=8,
                )
                raw_est = (result.stdout or "").strip().splitlines()[0].strip() if result.returncode == 0 else ""
                if raw_est and raw_est.upper() not in {"N/A", "UNKNOWN", ""}:
                    estimated_start_iso = raw_est
            except Exception:
                pass

        return self.json_response("200 OK", {
            "active": True,
            "jobId": found["jobId"],
            "slurmState": found["slurmState"],
            "status": found["status"],
            "backend": found["backend"],
            "projectName": found["projectName"],
            "epoch": epoch,
            "maxEpochs": max_epochs,
            "currentStep": current_step,
            "totalSteps": total_steps,
            "elapsedSeconds": elapsed,
            "startedAt": found["startedAt"],
            "estimatedStartIso": estimated_start_iso,
        })

    def handle_finetune_score_run(self, environ):
        """Score a model's RTTM output against hand-labeled reference RTTM and return DER + JER metrics as JSON.

        Path-traversal check is intentional; even on a single-user dashboard
        an XSS/CSRF could forge a request pointing at /etc/passwd and we'd read
        it back. Both paths must resolve inside self.root.
        """

        # Lazy import; diarization_metrics only costs something on import, and
        # most requests never need it.
        import diarization_metrics

        query = parse_qs((environ.get("QUERY_STRING") or ""), keep_blank_values=True)
        reference_value = (query.get("reference") or [""])[0]
        hypothesis_value = (query.get("hypothesis") or [""])[0]
        if not reference_value or not hypothesis_value:
            return self.json_response(
                "400 Bad Request",
                {"error": "Both 'reference' and 'hypothesis' RTTM paths are required."},
            )

        def _resolve_within_workspace(value: str) -> Path | None:
            candidate = self.resolve_local_path(value)
            try:
                resolved = candidate.resolve()
                resolved.relative_to(self.root.resolve())
            except (OSError, ValueError):
                return None
            return resolved

        reference_path = _resolve_within_workspace(reference_value)
        hypothesis_path = _resolve_within_workspace(hypothesis_value)
        if reference_path is None or hypothesis_path is None:
            return self.json_response(
                "400 Bad Request",
                {"error": "RTTM paths must point inside the project workspace."},
            )
        if not reference_path.is_file() or not hypothesis_path.is_file():
            return self.json_response("404 Not Found", {"error": "RTTM file not found."})

        try:
            metrics = diarization_metrics.score_run(reference_path, hypothesis_path)
        except (ValueError, OSError) as exc:
            return self.json_response("500 Internal Server Error", {"error": str(exc)})
        return self.json_response("200 OK", metrics)

    def handle_finetune_compare_runs(self, environ):
        """Batch DER/JER scoring against a flexible reference + N hypothesis models.

        Query params:

        - ``reference_source``: ``"labels"`` (default) to use saved training-label
          RTTMs as the reference, or ``"run"`` to use another diarization run's
          SRTs as the reference (no hand labels required for that mode).
        - ``reference_run``: required when ``reference_source=run``; path to the
          run directory whose SRTs are treated as ground truth.
        - ``model``: repeatable. Each value is a run directory whose SRT for
          each audio file is scored against the reference. One or more models
          may be supplied; the table layout scales to N columns.
        - ``audio``: repeatable list of audio files to score.

        Response shape:

        ``{
            "reference": { "source": "labels"|"run", "run": {...} },
            "models":   [ {"key": "m0", "name": "...", "path": "..."}, ... ],
            "files":    [ {"audio": "...", "reference_rttm": "...",
                           "metrics": {"m0": {...}, "m1": {...}}}, ... ],
            "averages": {"m0": {...}, "m1": {...}}
        }``

        Pairwise mode (any model = the chosen reference) is the same shape; the
        UI is responsible for not asking the user to score a model against
        itself.
        """

        import diarization_metrics

        query = parse_qs((environ.get("QUERY_STRING") or ""), keep_blank_values=True)
        reference_source = (query.get("reference_source") or ["labels"])[0].strip().lower() or "labels"
        if reference_source not in {"labels", "run"}:
            return self.json_response("400 Bad Request", {"error": f"Unknown reference_source {reference_source!r}. Expected 'labels' or 'run'."})
        reference_run_value = (query.get("reference_run") or [""])[0].strip()
        model_values = [value.strip() for value in (query.get("model") or []) if value.strip()]
        audio_values = [name.strip() for name in (query.get("audio") or []) if name.strip()]

        if not model_values:
            return self.json_response("400 Bad Request", {"error": "Pick at least one model run to score."})
        if not audio_values:
            return self.json_response("400 Bad Request", {"error": "Pick at least one audio file to score."})
        if reference_source == "run" and not reference_run_value:
            return self.json_response("400 Bad Request", {"error": "reference_source='run' requires reference_run to point at a diarization run directory."})

        try:
            runs_root_resolved = self.diarization_runs_root.resolve()
        except OSError:
            return self.json_response("500 Internal Server Error", {"error": "Diarization runs directory is not available."})

        def _resolve_run(value: str) -> Path | None:
            if not value:
                return None
            candidate = self.resolve_local_path(value)
            try:
                resolved = candidate.resolve()
                resolved.relative_to(runs_root_resolved)
            except (OSError, ValueError):
                return None
            return resolved if resolved.is_dir() else None

        reference_run: Path | None = None
        if reference_source == "run":
            reference_run = _resolve_run(reference_run_value)
            if reference_run is None:
                return self.json_response("400 Bad Request", {"error": "reference_run must point inside outputs/diarization_runs/."})

        # Resolve and de-dupe model paths in submission order. We tag each
        # entry with a stable "m0", "m1", ... key so the response can carry a
        # per-model metrics map without leaking long disk paths into JSON keys.
        model_runs: list[dict[str, object]] = []
        seen_paths: set[str] = set()
        for raw_value in model_values:
            run_dir = _resolve_run(raw_value)
            if run_dir is None:
                return self.json_response("400 Bad Request", {"error": f"Model run {raw_value!r} must point inside outputs/diarization_runs/."})
            key_path = str(run_dir)
            if key_path in seen_paths:
                continue
            seen_paths.add(key_path)
            model_runs.append({"key": f"m{len(model_runs)}", "path": run_dir, "name": run_dir.name})

        if not model_runs:
            return self.json_response("400 Bad Request", {"error": "No valid model runs after de-duplication."})

        # Translate SRT cue text speakers (e.g. "SPEAKER_00: hi") into Interval
        # objects the metrics module understands. Cues with no parseable
        # speaker fall back to a stable per-cue label so they still count
        # toward miss/false-alarm even if the model emitted unlabeled regions.
        def _cues_to_intervals(cues, *, stem: str) -> list:
            intervals = []
            for cue in cues:
                duration_ms = cue.end_ms - cue.start_ms
                if duration_ms <= 0:
                    continue
                speaker = cue.speaker or f"{stem}_cue_{cue.index:04d}"
                intervals.append(
                    diarization_metrics.Interval(
                        start=cue.start_ms / 1000.0,
                        end=cue.end_ms / 1000.0,
                        speaker=speaker,
                    )
                )
            return intervals

        def _intervals_from_run(run_dir: Path, stem: str) -> tuple[list, str | None, Path | None]:
            """Return (intervals, error_message, srt_path). Either intervals or error_message is set."""

            srt_path = run_dir / f"{stem}.srt"
            if not srt_path.is_file():
                return [], f"No SRT for this audio in {run_dir.name}.", None
            try:
                cues = parse_srt(srt_path)
            except (ValueError, OSError) as exc:
                return [], f"Could not parse SRT: {exc}", srt_path
            return _cues_to_intervals(cues, stem=stem), None, srt_path

        def _zero_aggregate() -> dict[str, float]:
            return {
                "scored": 0.0,
                "skipped": 0.0,
                "der_numerator_seconds": 0.0,
                "ref_seconds_total": 0.0,
                "miss_total": 0.0,
                "false_alarm_total": 0.0,
                "confusion_total": 0.0,
                "jer_total": 0.0,
            }

        def _accumulate(agg: dict[str, float], metrics: dict[str, object]) -> None:
            agg["scored"] += 1
            ref_seconds = float(metrics["reference_speech_seconds"])
            agg["der_numerator_seconds"] += float(metrics["der"]) * ref_seconds
            agg["ref_seconds_total"] += ref_seconds
            agg["miss_total"] += float(metrics["miss_seconds"])
            agg["false_alarm_total"] += float(metrics["false_alarm_seconds"])
            agg["confusion_total"] += float(metrics["confusion_seconds"])
            agg["jer_total"] += float(metrics["jer"])

        def _finalize_average(agg: dict[str, float]) -> dict[str, object] | None:
            scored = int(agg["scored"])
            if scored == 0:
                return None
            return {
                "files_scored": scored,
                "files_skipped": int(agg["skipped"]),
                "weighted_der": (agg["der_numerator_seconds"] / agg["ref_seconds_total"]) if agg["ref_seconds_total"] > 0 else None,
                "macro_jer": agg["jer_total"] / scored,
                "miss_seconds": agg["miss_total"],
                "false_alarm_seconds": agg["false_alarm_total"],
                "confusion_seconds": agg["confusion_total"],
                "reference_speech_seconds": agg["ref_seconds_total"],
            }

        try:
            workspace_root = self.root.resolve()
        except OSError:
            return self.json_response("500 Internal Server Error", {"error": "Workspace root is not available."})

        label_records = self.load_training_label_records() if reference_source == "labels" else {}

        aggregates: dict[str, dict[str, float]] = {entry["key"]: _zero_aggregate() for entry in model_runs}
        file_results: list[dict[str, object]] = []

        for audio_value in audio_values:
            audio_basename = self.clean_audio_selection_value(audio_value)
            if not audio_basename:
                file_results.append({"audio": audio_value, "error": "Audio name is empty after cleanup."})
                continue
            stem = self.diarization_output_base(audio_basename)

            # Resolve the reference for this audio file. Bail with a per-file
            # error if labels are missing or the reference run has no SRT for
            # this stem; that's friendlier than a 400 that nukes the whole
            # batch.
            reference_intervals: list
            reference_description: str
            if reference_source == "labels":
                record = label_records.get(audio_basename) or label_records.get(Path(audio_basename).name) or {}
                ref_path_str = str(record.get("training_rttm_path") or "")
                if not ref_path_str:
                    file_results.append({"audio": audio_basename, "error": "No completed training label / RTTM saved for this file."})
                    continue
                try:
                    reference_path = self.resolve_local_path(ref_path_str).resolve()
                    reference_path.relative_to(workspace_root)
                except (OSError, ValueError):
                    file_results.append({"audio": audio_basename, "error": "Reference RTTM is outside the workspace."})
                    continue
                if not reference_path.is_file():
                    file_results.append({"audio": audio_basename, "error": "Reference RTTM is not on disk anymore."})
                    continue
                try:
                    reference_intervals = diarization_metrics.parse_rttm_intervals(reference_path)
                except (ValueError, OSError) as exc:
                    file_results.append({"audio": audio_basename, "error": f"Could not parse reference RTTM: {exc}"})
                    continue
                reference_description = self.describe_path(reference_path)
            else:
                ref_intervals_or_empty, ref_error, ref_srt = _intervals_from_run(reference_run, stem)
                if ref_error:
                    file_results.append({"audio": audio_basename, "error": f"Reference run: {ref_error}"})
                    continue
                reference_intervals = ref_intervals_or_empty
                reference_description = self.describe_path(ref_srt) if ref_srt else self.describe_path(reference_run)

            per_model: dict[str, dict[str, object]] = {}
            for entry in model_runs:
                key = entry["key"]
                run_dir = entry["path"]
                # Pairwise edge case: if a hypothesis model points at the same
                # run as the reference, scoring it would give DER 0 with
                # nothing useful, so flag it instead of silently passing.
                if reference_source == "run" and reference_run is not None and run_dir == reference_run:
                    per_model[key] = {"error": "This model is the reference."}
                    aggregates[key]["skipped"] += 1
                    continue
                hyp_intervals, hyp_error, srt_path = _intervals_from_run(run_dir, stem)
                if hyp_error:
                    per_model[key] = {"error": hyp_error}
                    aggregates[key]["skipped"] += 1
                    continue
                try:
                    metrics = diarization_metrics.score_intervals(reference_intervals, hyp_intervals)
                except (ValueError, OSError) as exc:
                    per_model[key] = {"error": f"Scoring failed: {exc}"}
                    aggregates[key]["skipped"] += 1
                    continue
                per_model[key] = {
                    "run_dir": self.describe_path(run_dir),
                    "run_name": run_dir.name,
                    "srt_path": self.describe_path(srt_path) if srt_path else "",
                    **metrics,
                }
                _accumulate(aggregates[key], metrics)

            file_results.append(
                {
                    "audio": audio_basename,
                    "reference_rttm": reference_description,
                    "metrics": per_model,
                }
            )

        averages = {entry["key"]: _finalize_average(aggregates[entry["key"]]) for entry in model_runs}

        return self.json_response(
            "200 OK",
            {
                "reference": {
                    "source": reference_source,
                    "run": (
                        {"path": self.describe_path(reference_run), "name": reference_run.name}
                        if reference_run is not None
                        else None
                    ),
                },
                "models": [
                    {
                        "key": entry["key"],
                        "name": entry["name"],
                        "path": self.describe_path(entry["path"]),
                    }
                    for entry in model_runs
                ],
                "files": file_results,
                "averages": averages,
            },
        )

    def handle_finetune_auto_train(self, environ):
        """Toggle the per-project "auto-train when labels complete" flag.

        The label-save handler reads this flag on every completion; it lives in
        display.json so prepare_project regeneration can't clobber it.
        """

        form = self.parse_form(environ)
        project_slug = (form.getfirst("project_slug") or form.getfirst("project_name") or "").strip()
        backend_value = form.getfirst("backend") or form.getfirst("project_backend") or ""
        # Checkbox semantics: presence == enabled, absence == disabled.
        enabled = bool(form.getfirst("auto_train_enabled"))
        if not project_slug:
            return self.redirect(environ, "/fine-tuning", message="Pick a fine-tuning project to update.", status="error")
        try:
            backend = normalize_backend(backend_value)
            set_project_auto_train(
                project_slug,
                backend=backend,
                enabled=enabled,
                root=self.root,
            )
        except (FileNotFoundError, ValueError) as exc:
            return self.redirect(environ, "/fine-tuning", message=str(exc), status="error")
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/fine-tuning",
            message=(
                f"Auto-train ON for {project_slug}: completing a label will queue a training run."
                if enabled
                else f"Auto-train OFF for {project_slug}."
            ),
            status="success",
        )

    def handle_finetune_rename_run(self, environ):
        """Update the display label for one training run so runs can be told apart without renaming the directory."""

        form = self.parse_form(environ)
        run_dir_value = (form.getfirst("run_dir") or "").strip()
        display_name = (form.getfirst("display_name") or "").strip()
        if not run_dir_value:
            return self.redirect(environ, "/fine-tuning", message="Pick a run to rename.", status="error")
        if not display_name:
            return self.redirect(environ, "/fine-tuning", message="Provide a new display name.", status="error")
        run_dir = self.resolve_local_path(run_dir_value)
        # Sanity-check that the path lands inside fine_tuning/projects/; a crafted
        # form post could otherwise write display.json anywhere on the machine.
        runs_root = (self.root / "fine_tuning" / "projects").resolve()
        try:
            resolved = run_dir.resolve()
            resolved.relative_to(runs_root)
        except (OSError, ValueError):
            return self.redirect(environ, "/fine-tuning", message="Run path is outside the fine-tuning workspace.", status="error")
        try:
            set_run_display_name(resolved, display_name=display_name)
        except (FileNotFoundError, ValueError) as exc:
            return self.redirect(environ, "/fine-tuning", message=str(exc), status="error")
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/fine-tuning",
            message=f"Renamed run to '{display_name}'.",
            status="success",
        )

    def handle_finetune_delete_run_model(self, environ):
        """Remove one fine-tuned model's checkpoint artifacts while preserving run logs."""

        form = self.parse_form(environ)
        run_dir_value = (form.getfirst("run_dir") or "").strip()
        if not run_dir_value:
            return self.redirect(environ, "/fine-tuning", message="Pick a fine-tuned model to delete.", status="error")
        run_dir = self.resolve_local_path(run_dir_value)
        runs_root = (self.root / "fine_tuning" / "projects").resolve()
        try:
            resolved = run_dir.resolve()
            resolved.relative_to(runs_root)
        except (OSError, ValueError):
            return self.redirect(environ, "/fine-tuning", message="Run path is outside the fine-tuning workspace.", status="error")
        try:
            result = delete_run_model_artifacts(resolved)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return self.redirect(environ, "/fine-tuning", message=str(exc), status="error")
        self.invalidate_dashboard_cache()
        verb = "Deleted" if result.get("removed") else "Marked deleted"
        return self.redirect(
            environ,
            "/fine-tuning",
            message=f"{verb} fine-tuned model artifacts for {resolved.name}. Run logs were kept.",
            status="success",
        )

    def training_audio_pair_stems(self, audio_path: Path) -> list[str]:
        """Return filename stems that may identify labels for one audio file."""

        stems = [audio_path.stem]
        unnumbered_stem = NUMBERED_PREFIX.sub("", audio_path.stem, count=1)
        if unnumbered_stem and unnumbered_stem not in stems:
            stems.append(unnumbered_stem)
        try:
            relative_audio = self.audio_relative_path(audio_path)
        except ValueError:
            relative_audio = ""
        if relative_audio:
            flattened = self.diarization_output_base(relative_audio)
            if flattened and flattened not in stems:
                stems.append(flattened)
        return stems

    def completed_label_rttm_candidates_for_audio(self, audio_path: Path) -> list[Path]:
        """Return RTTMs recorded by a completed manual label for this audio."""

        try:
            relative_audio = self.audio_relative_path(audio_path)
        except ValueError:
            relative_audio = ""
        records = self.load_training_label_records()
        record = records.get(relative_audio) or records.get(audio_path.name) or {}
        if self.training_label_status(record) != "completed":
            return []

        candidates: list[Path] = []
        raw_paths = [record.get("training_rttm_path")]
        for usage in record.get("training_usage") or []:
            if isinstance(usage, dict):
                raw_paths.append(usage.get("sample_rttm_path"))
        for raw_path in raw_paths:
            value = str(raw_path or "").strip()
            if not value:
                continue
            try:
                candidate = self.resolve_under_root(value)
            except ValueError:
                candidate = Path(value).expanduser()
            # Same ENAMETOOLONG guard as validated_training_rttm_for_audio;
            # KaggleBabyNoises stems concatenated through stitching push
            # paths past the kernel's NAME_MAX, and a raw is_file() then
            # crashes the whole render instead of just skipping the sample.
            try:
                is_present = candidate.is_file()
            except OSError:
                continue
            if is_present and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def synthetic_rttm_index(self) -> dict[str, object]:
        """One-shot index of where synthetic-completion RTTMs live on disk.

        Building the index costs two directory walks (audio_in for sidecars,
        label_work for in-progress drafts). Per-audio lookups against the
        index are pure set membership: O(1) and zero syscalls. The previous
        path called ``Path.is_file()`` ~5 times per audio and ate ~24 s of
        wall time per page render with 6 k+ audio files on the cluster's
        networked filesystem. Cached for 60 s; mutations call
        ``invalidate_dashboard_cache``.
        """

        return self.cached_value(
            "synthetic_rttm_index",
            ttl_seconds=60.0,
            builder=self._build_synthetic_rttm_index,
        )

    def _build_synthetic_rttm_index(self) -> dict[str, object]:
        sidecar_rttm_paths: set[Path] = set()
        if self.audio_dir.is_dir():
            try:
                for rttm in self.audio_dir.rglob("*.rttm"):
                    sidecar_rttm_paths.add(rttm)
            except OSError:
                pass
        label_work_rttm_stems: set[str] = set()
        label_work_dir = self.training_label_work_dir
        if label_work_dir.is_dir():
            try:
                for rttm in label_work_dir.glob("*.rttm"):
                    label_work_rttm_stems.add(rttm.stem)
            except OSError:
                pass
        return {
            "sidecar_rttm_paths": sidecar_rttm_paths,
            "label_work_rttm_stems": label_work_rttm_stems,
        }

    def synthetic_rttm_for_audio_via_index(
        self,
        audio_path: Path,
        index: dict[str, object] | None = None,
    ) -> Path | None:
        """Indexed variant of ``validated_training_rttm_for_audio``.

        Uses the cached ``synthetic_rttm_index`` so summary + row builders
        can answer "does this audio have a synthetic completion?" without
        per-audio stat calls. Skips ``build_sample`` validation; callers
        that ship the result to the user (e.g. the file picker) keep using
        ``validated_training_rttm_for_audio`` so a malformed RTTM doesn't
        slip through.
        """

        if index is None:
            index = self.synthetic_rttm_index()
        sidecar_paths = index.get("sidecar_rttm_paths")
        if isinstance(sidecar_paths, set):
            for stem in self.training_audio_pair_stems(audio_path):
                candidate = audio_path.with_name(f"{stem}.rttm")
                if candidate in sidecar_paths:
                    return candidate
        label_work_stems = index.get("label_work_rttm_stems")
        if isinstance(label_work_stems, set) and label_work_stems:
            stems_to_check: list[str] = []
            try:
                relative_audio = self.audio_relative_path(audio_path)
            except ValueError:
                relative_audio = ""
            if relative_audio:
                flattened = self.diarization_output_base(relative_audio)
                if flattened:
                    stems_to_check.append(flattened)
                if "/" not in relative_audio and audio_path.stem not in stems_to_check:
                    stems_to_check.append(audio_path.stem)
            else:
                stems_to_check.append(audio_path.stem)
            for stem in stems_to_check:
                if stem in label_work_stems:
                    return self.training_label_work_dir / f"{stem}.rttm"
        return None

    def validated_training_rttm_for_audio(self, audio_path: Path) -> Path | None:
        """Return the RTTM that makes one audio file ready for fine-tuning."""

        # Some KaggleBabyNoises filenames blow past the filesystem's path
        # limit when we tack on a ``.rttm`` suffix; Path.is_file() then
        # raises ENAMETOOLONG and crashes the whole /fine-tuning render.
        # Treating the OSError as "no sidecar" lets the caller move on.
        def _is_file(path: Path) -> bool:
            try:
                return path.is_file()
            except OSError:
                return False

        candidates: list[Path] = []
        for stem in self.training_audio_pair_stems(audio_path):
            sidecar = audio_path.with_name(f"{stem}.rttm")
            if _is_file(sidecar) and sidecar not in candidates:
                candidates.append(sidecar)
        label_work_stems: list[str] = []
        try:
            relative_audio = self.audio_relative_path(audio_path)
        except ValueError:
            relative_audio = ""
        if relative_audio:
            flattened = self.diarization_output_base(relative_audio)
            if flattened:
                label_work_stems.append(flattened)
            if "/" not in relative_audio and audio_path.stem not in label_work_stems:
                label_work_stems.append(audio_path.stem)
        else:
            label_work_stems.append(audio_path.stem)
        for stem in label_work_stems:
            candidate = self.training_label_work_dir / f"{stem}.rttm"
            if _is_file(candidate) and candidate not in candidates:
                candidates.append(candidate)
        for candidate in self.completed_label_rttm_candidates_for_audio(audio_path):
            if candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            try:
                build_sample(audio_path, candidate, None)
            except Exception:
                continue
            return candidate
        return None

    def ready_training_source_files(self) -> dict[str, object]:
        """Return audio_in files that have a completed training-label record.

        Strict filter: only audio with status="completed" in label_status.json
        appears here, and only when its recorded RTTM still passes
        ``build_sample`` validation. In-progress label_work drafts, raw corpus
        sidecars, and project-internal RTTMs without a completed label are
        intentionally excluded. The fine-tune picker should only surface
        samples the user has finished labeling and that are ready to train on.

        Stitched runs still flow in because ``mirror_stitched_into_media``
        writes a completed-label record for every successful stitch before
        we read the records below.
        """

        try:
            for run_dir in self.stitched_run_directories(limit=100):
                self.mirror_stitched_into_media(run_dir)
        except Exception:
            pass
        audio_files: list[Path] = []
        rttm_files: list[Path] = []
        rttm_audio_map: dict[str, Path] = {}
        records = self.load_training_label_records()
        for relative_audio, record in records.items():
            if not isinstance(record, dict):
                continue
            if self.training_label_status(record) != "completed":
                continue
            audio_path = self.audio_dir / relative_audio
            if not audio_path.is_file():
                continue
            rttm_path: Path | None = None
            for candidate in self.completed_label_rttm_candidates_for_audio(audio_path):
                try:
                    build_sample(audio_path, candidate, None)
                except Exception:
                    continue
                rttm_path = candidate
                break
            if rttm_path is None:
                continue
            audio_files.append(audio_path)
            rttm_files.append(rttm_path)
            try:
                rttm_audio_map[str(rttm_path.resolve())] = audio_path
            except OSError:
                rttm_audio_map[str(rttm_path)] = audio_path
        # Stable order for the picker so consecutive renders don't reshuffle.
        paired = sorted(
            zip(audio_files, rttm_files),
            key=lambda pair: str(pair[0]).lower(),
        )
        audio_files = [pair[0] for pair in paired]
        rttm_files = [pair[1] for pair in paired]
        return {
            "audio_files": audio_files,
            "rttm_files": rttm_files,
            "rttm_audio_map": rttm_audio_map,
        }

    def training_source_files(self) -> dict[str, object]:
        """Return server-side files that are ready to be paired for training."""

        return self.cached_value(
            "training_source_files",
            ttl_seconds=20.0,
            builder=lambda: {
                **self.ready_training_source_files(),
                "transcript_files": self.newest_files(
                    search_roots=[
                        self.audio_dir,
                        self.stitched_dir,
                        self.root / "fine_tuning" / "projects",
                        self.outputs_root,
                        self.root / "job_outputs",
                    ],
                    suffixes=TRAINING_TRANSCRIPT_SUFFIXES,
                    limit=800,
                    exclude_dir_names=TRAINING_SOURCE_SCAN_EXCLUDE_DIRS,
                ),
            },
        )

    def fine_tuning_project_metrics(self, project: dict[str, object]) -> dict[str, object]:
        """Calculate presentation-ready dataset metrics for one fine-tuning project.

        Each render previously re-parsed every RTTM and ran ffprobe per audio
        sample. We now cache by (project path, audio_dir mtime, rttm_dir
        mtime, metadata.json mtime). Adding/removing RTTMs or audio bumps the
        directory mtime, which invalidates the cache automatically.
        """

        project_path = Path(str(project.get("path", "")))
        audio_dir = project_path / "audio"
        rttm_dir = project_path / "rttm"

        def _dir_mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        fingerprint = (
            str(project_path),
            _dir_mtime(audio_dir),
            _dir_mtime(rttm_dir),
            _dir_mtime(project_path / "artifacts" / "metadata.json"),
        )
        cache_key = f"fine_tuning_project_metrics::{fingerprint}"
        return self.cached_value(
            cache_key,
            ttl_seconds=30.0,
            builder=lambda: self._build_fine_tuning_project_metrics(project),
        )

    def _build_fine_tuning_project_metrics(self, project: dict[str, object]) -> dict[str, object]:
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
