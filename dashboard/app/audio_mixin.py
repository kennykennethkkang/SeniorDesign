#!/usr/bin/env python3
"""Audio inventory, folder management, uploads, and reference rewrites."""
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


class AudioMixin:
    """Audio inventory, folder management, uploads, and reference rewrites."""

    def audio_inventory(self) -> list[Path]:
        """Return the sorted input inventory for pages that need file selection."""

        # YouTube conversion and Slurm jobs create WAVs from separate processes,
        # so an in-process cache can hide fresh media from the Media Library.
        return iter_audio_files(self.audio_dir)

    def audio_folder_rows(
        self,
        audio_files: list[Path] | None = None,
    ) -> list[dict[str, object]]:
        """Describe managed audio set folders for the frontend.

        Accepts an optional pre-fetched ``audio_files`` list so the caller
        (typically ``page_context``) does not pay for two walks of
        ``audio_in/`` per request — one for the inventory grid and one for
        the folder dropdown counters. Falls back to the unguarded scan when
        called standalone (e.g., from the AJAX live-tracking endpoint).
        """

        self.ensure_default_audio_folders()
        if audio_files is None:
            audio_files = iter_audio_files(self.audio_dir)
        # Single resolved root reused per row to avoid N filesystem stat calls
        # for `path.resolve().relative_to(self.audio_dir.resolve())` below.
        audio_root = self.audio_dir.resolve()
        counts: dict[str, int] = {"": 0}
        for path in audio_files:
            try:
                relative = path.resolve().relative_to(audio_root)
            except ValueError:
                # Outside audio_in (broken symlink that escaped); skip silently.
                continue
            folder = relative.parts[0] if len(relative.parts) > 1 else ""
            counts[folder] = counts.get(folder, 0) + 1

        folders = {folder for folder in DEFAULT_AUDIO_FOLDERS}
        if self.audio_dir.is_dir():
            folders.update(
                path.name
                for path in self.audio_dir.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            )

        rows: list[dict[str, object]] = [
            {
                "value": ROOT_AUDIO_FOLDER_VALUE,
                "name": "Unsorted Root",
                "path": self.describe_path(self.audio_dir),
                "fileCount": counts.get("", 0),
                "renamable": False,
            }
        ]
        for folder in sorted(folders, key=lambda value: value.lower()):
            folder_path = self.audio_dir / folder
            rows.append(
                {
                    "value": folder,
                    "name": folder.replace("_", " ").replace("-", " ").title(),
                    "path": self.describe_path(folder_path),
                    "fileCount": counts.get(folder, 0),
                    "renamable": True,
                }
            )
        return rows

    def audio_input_count(self) -> int:
        """Count supported input media without sorting the full inventory."""

        if not self.audio_dir.is_dir():
            return 0
        resolved_audio_dir = self.audio_dir.resolve()
        return sum(
            1
            for candidate in resolved_audio_dir.rglob("*")
            if (
                candidate.is_file()
                and candidate.suffix.lower() in WORKSPACE_MEDIA_SUFFIXES
                and "_whisper_input" not in candidate.stem
            )
        )

    def queue_size(self) -> int:
        """Count queued YouTube URLs with the same short TTL as other dashboard data."""

        return self.cached_value(
            "queue_size",
            ttl_seconds=3.0,
            builder=lambda: count_queued_urls(self.youtube_links_path),
        )

    def project_count(self) -> int:
        """Count fine-tuning projects without loading the full project cards."""

        def builder() -> int:
            return len(list_projects(root=self.root))

        return self.cached_value("project_count", ttl_seconds=3.0, builder=builder)

    def project_summaries(self) -> list[dict[str, object]]:
        """Load detailed project summaries only for pages that render them."""

        return self.cached_value(
            "project_summaries",
            ttl_seconds=3.0,
            builder=lambda: list_projects(root=self.root),
        )

    def latest_output_roots(self) -> dict[str, Path | None]:
        """Load the newest output directories once for the sidebar."""

        return self.cached_value(
            "latest_output_roots",
            ttl_seconds=3.0,
            builder=lambda: {
                "latest_site_diarization": latest_diarization_directory(self.diarization_runs_root),
                "latest_site_youtube": latest_directory(self.youtube_runs_root),
                "latest_single": latest_directory(self.single_output_root),
                "latest_bulk": latest_directory(self.bulk_output_root),
            },
        )

    def audio_folder_value(self, raw_value: str | None, *, default: str = "") -> str:
        """Return the single folder component used below audio_in."""

        value = (raw_value or "").strip()
        if not value or value == ROOT_AUDIO_FOLDER_VALUE:
            return ""
        return self.safe_folder_name(value)

    def ensure_default_audio_folders(self) -> None:
        """Create the built-in source folders used by the dashboard."""

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        for folder in DEFAULT_AUDIO_FOLDERS:
            (self.audio_dir / folder).mkdir(parents=True, exist_ok=True)

    def audio_folder_path(self, folder_value: str, *, create: bool = False) -> Path:
        """Resolve a managed audio folder below audio_in."""

        normalized = self.audio_folder_value(folder_value)
        folder_path = self.audio_dir if not normalized else self.audio_dir / normalized
        resolved_audio_dir = self.audio_dir.resolve()
        resolved_folder = folder_path.resolve()
        try:
            resolved_folder.relative_to(resolved_audio_dir)
        except ValueError as exc:
            raise ValueError(f"Audio folder is outside audio_in: {folder_value}") from exc
        if create:
            resolved_folder.mkdir(parents=True, exist_ok=True)
        return resolved_folder

    def selected_audio_folder_path(
        self,
        form: cgi.FieldStorage,
        *,
        default_folder: str,
        new_field: str = "new_audio_folder",
        selected_field: str = "audio_folder",
        create: bool = True,
    ) -> tuple[str, Path]:
        """Resolve the target audio set selected by a form."""

        requested_new = (form.getfirst(new_field) or "").strip()
        requested_existing = (form.getfirst(selected_field) or "").strip()
        folder_value = self.audio_folder_value(
            requested_new or requested_existing or default_folder,
            default=default_folder,
        )
        folder_path = self.audio_folder_path(folder_value, create=create)
        return folder_value, folder_path

    def audio_relative_path(self, path: Path) -> str:
        """Return a POSIX-style path relative to audio_in."""

        return path.resolve().relative_to(self.audio_dir.resolve()).as_posix()

    def clean_audio_selection_value(self, raw_value: str) -> str:
        """Normalize a selected audio path while rejecting path traversal."""

        value = str(raw_value or "").strip().replace("\\", "/")
        if not value:
            return ""
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(self.audio_dir.resolve()).as_posix()
            except ValueError:
                return ""
        parts = [part for part in value.split("/") if part and part != "."]
        if not parts or any(part == ".." for part in parts):
            return ""
        return "/".join(parts)

    def audio_path_for_relative(self, relative_audio: str) -> Path | None:
        """Resolve a relative audio selection below audio_in."""

        cleaned = self.clean_audio_selection_value(relative_audio)
        if not cleaned:
            return None
        candidate = (self.audio_dir / cleaned).resolve()
        try:
            candidate.relative_to(self.audio_dir.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def locate_audio_file(self, relative_audio: str) -> Path:
        """Resolve an audio selection, falling back to a basename search across audio_in.

        The dashboard caches the audio inventory, so a row clicked from a stale
        page state may point at a path that has since moved (e.g. the file was
        relocated from one subfolder to another). When the strict resolution
        fails, walk audio_in once for the basename to recover the user's intent
        instead of surfacing a confusing 'not found' error. Raises
        ``FileNotFoundError`` if nothing matches and ``ValueError`` if the
        basename matches more than one file (so we never guess between
        candidates).
        """

        cleaned = self.clean_audio_selection_value(relative_audio)
        if not cleaned:
            raise ValueError(f"Invalid audio path: {relative_audio!r}")
        primary = (self.audio_dir / cleaned).resolve()
        try:
            primary.relative_to(self.audio_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Audio path is outside audio_in: {relative_audio!r}") from exc
        if primary.is_file():
            return primary

        basename = Path(cleaned).name
        if not basename:
            raise FileNotFoundError(f"Audio file not found: {cleaned}")
        matches: list[Path] = []
        if self.audio_dir.is_dir():
            for candidate in self.audio_dir.rglob(basename):
                if candidate.is_file():
                    try:
                        candidate.resolve().relative_to(self.audio_dir.resolve())
                    except ValueError:
                        continue
                    matches.append(candidate.resolve())
        if not matches:
            raise FileNotFoundError(f"Audio file not found: {cleaned}")
        if len(matches) > 1:
            options = ", ".join(self.audio_relative_path(path) for path in matches)
            raise ValueError(
                f"Multiple files named {basename!r} exist under audio_in; clarify which one to use: {options}"
            )
        return matches[0]

    def diarization_output_base(self, audio_value: str) -> str:
        """Match run_diarization.py's artifact basename for nested library sets."""

        cleaned = self.clean_audio_selection_value(audio_value)
        if not cleaned:
            cleaned = Path(audio_value or "audio").name
        stem_path = Path(cleaned).with_suffix("")
        return "__".join(self.safe_output_component(part) for part in stem_path.parts)

    def choose_available_audio_path(self, target_path: Path) -> Path:
        """Avoid overwriting an existing upload in the selected set folder."""

        if not target_path.exists():
            return target_path
        counter = 2
        while True:
            candidate = target_path.with_name(f"{target_path.stem}_{counter}{target_path.suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def convert_uploaded_audio_to_wav(self, item: cgi.FieldStorage, target_dir: Path) -> Path:
        """Store one uploaded audio item as a diarization-ready WAV file."""

        original_name = self.safe_filename(str(getattr(item, "filename", "")))
        original_suffix = Path(original_name).suffix.lower()
        output_stem = Path(original_name).stem or "upload"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.choose_available_audio_path(target_dir / f"{output_stem}.wav")
        ffmpeg_location = self.resolve_ffmpeg_location()

        if not ffmpeg_location:
            if original_suffix != ".wav":
                raise RuntimeError("ffmpeg is required to convert non-WAV uploads.")
            with target_path.open("wb") as handle:
                shutil.copyfileobj(item.file, handle)
            return target_path.resolve()

        with tempfile.TemporaryDirectory(dir=target_dir, prefix=".upload_tmp_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            temp_input = temp_dir / original_name
            temp_output = temp_dir / f"{output_stem}_converted.wav"
            with temp_input.open("wb") as handle:
                shutil.copyfileobj(item.file, handle)
            command = [
                ffmpeg_location,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(temp_input),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(temp_output),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0 or not temp_output.is_file():
                error_text = (completed.stderr or completed.stdout or "ffmpeg conversion failed").strip()
                raise RuntimeError(error_text.splitlines()[-1] if error_text else "ffmpeg conversion failed")
            shutil.move(str(temp_output), str(target_path))
        return target_path.resolve()

    def handle_audio_folder_create(self, environ):
        """Create a user-named audio set folder."""

        form = self.parse_form(environ)
        folder_name = self.audio_folder_value(form.getfirst("folder_name"))
        if not folder_name:
            return self.redirect(environ, "/uploads", message="Enter a folder name first.", status="error")
        folder_path = self.audio_folder_path(folder_name, create=True)
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/uploads",
            message=f"Created audio folder: {self.describe_path(folder_path)}",
            status="success",
        )

    def handle_audio_folder_rename(self, environ):
        """Rename one managed audio set folder."""

        form = self.parse_form(environ)
        current_folder = self.audio_folder_value(form.getfirst("current_folder"))
        new_folder = self.audio_folder_value(form.getfirst("new_folder_name"))
        if not current_folder:
            return self.redirect(environ, "/uploads", message="The root audio folder cannot be renamed.", status="error")
        if not new_folder:
            return self.redirect(environ, "/uploads", message="Enter a new folder name first.", status="error")
        current_path = self.audio_folder_path(current_folder)
        target_path = self.audio_folder_path(new_folder)
        if not current_path.is_dir():
            return self.redirect(environ, "/uploads", message=f"Audio folder not found: {current_folder}", status="error")
        if target_path.exists():
            return self.redirect(environ, "/uploads", message=f"Audio folder already exists: {new_folder}", status="error")
        current_path.rename(target_path)
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/uploads",
            message=f"Renamed audio folder to: {self.describe_path(target_path)}",
            status="success",
        )

    def handle_audio_folder_delete(self, environ):
        """Delete one audio set folder and every WAV (plus derived artifacts) inside it."""

        form = self.parse_form(environ)
        raw_folder_value = (form.getfirst("folder_name") or "").strip()
        if not raw_folder_value:
            return self.redirect(
                environ,
                "/uploads",
                message="Choose an audio folder to delete.",
                status="error",
            )
        try:
            summary = self.cascade_delete_audio_folder(raw_folder_value)
        except ValueError as exc:
            return self.redirect(environ, "/uploads", message=str(exc), status="error")
        except OSError as exc:
            return self.redirect(
                environ,
                "/uploads",
                message=f"Could not delete audio folder '{raw_folder_value}': {exc}",
                status="error",
            )

        is_root = summary.get("folder") == ROOT_AUDIO_FOLDER_VALUE
        display_name = "Unsorted Root" if is_root else str(summary["folder"])
        if summary.get("already_missing"):
            details: list[str] = ["Folder was already gone; the Media Library has been refreshed."]
        elif is_root:
            details = [
                f"Removed {summary['files_removed']} root-level audio file(s); audio_in stays available for future uploads."
            ]
        else:
            details = [f"Removed {summary['files_removed']} audio file(s)."]
        if summary["diarization_artifacts_removed"]:
            details.append(
                f"Removed {summary['diarization_artifacts_removed']} diarization artifact file(s)."
            )
        if summary["urls_removed"]:
            details.append(
                f"Cleared {len(summary['urls_removed'])} YouTube index row(s); the URLs stay in youtube_links.txt for redownload."
            )
        return self.redirect(
            environ,
            "/uploads",
            message=self.notification_message(
                (
                    f"Audio folder already removed: {display_name}"
                    if summary.get("already_missing")
                    else f"Deleted audio folder: {display_name}"
                ),
                *details,
            ),
            status="success",
        )

    def handle_audio_file_delete(self, environ):
        """Cascade-delete one WAV file and every artifact tied to it."""

        form = self.parse_form(environ)
        relative_audio = form.getfirst("audio_path") or ""
        try:
            summary = self.cascade_delete_audio_file(relative_audio)
        except ValueError as exc:
            # Reserved for genuinely bad input (path traversal, empty value).
            # FileNotFoundError no longer surfaces here — cascade_delete_audio_file
            # treats a missing WAV as a no-op success and still sweeps orphaned
            # references, which is the experience the user actually wants when
            # they click Delete on a stale row.
            return self.redirect(environ, "/uploads", message=str(exc), status="error")

        details: list[str] = []
        if summary["youtube_url"]:
            details.append(
                f"Cleared YouTube index row for {summary['youtube_url']}; the URL stays in youtube_links.txt for redownload."
            )
        if summary["diarization_artifacts_removed"]:
            details.append(
                f"Removed {summary['diarization_artifacts_removed']} diarization artifact file(s)."
            )
        if summary["summary_rows_removed"]:
            details.append(
                f"Removed {summary['summary_rows_removed']} runtime summary row(s)."
            )
        if summary["selection_lines_removed"]:
            details.append(
                f"Removed {summary['selection_lines_removed']} selected_audio.txt line(s)."
            )
        was_present = bool(summary.get("audio_was_present", True))
        headline = (
            f"Deleted audio file: {summary['audio_relative']}"
            if was_present
            else f"Audio file was already removed: {summary['audio_relative']}"
        )
        if not was_present and not details:
            details.append("No orphaned references were found; the row has been cleared from the table.")
        elif not was_present:
            details.insert(0, "The WAV was no longer on disk; cleaned up its leftover references.")
        return self.redirect(
            environ,
            "/uploads",
            message=self.notification_message(headline, *details),
            status="success",
        )

    def handle_audio_files_bulk_delete(self, environ):
        """Cascade-delete every WAV checked in the Media Library multiselect.

        Each path is run through the same idempotent ``cascade_delete_audio_file``
        used by the per-row Delete button, so missing files (stale page state)
        and YouTube-linked files behave consistently with the single-row flow.
        Failures on one file do not abort the rest of the batch — the response
        notifies the user about both successes and failures so they know
        exactly what landed.
        """

        form = self.parse_form(environ)
        relative_audio_values = self.ordered_unique(form.getlist("audio_paths"))
        if not relative_audio_values:
            return self.redirect(
                environ,
                "/uploads",
                message="Select at least one audio file to delete.",
                status="error",
            )

        successes: list[str] = []
        failures: list[tuple[str, str]] = []
        total_artifacts = 0
        total_summary_rows = 0
        total_selection_lines = 0
        for relative_audio in relative_audio_values:
            try:
                summary = self.cascade_delete_audio_file(relative_audio)
            except (ValueError, FileNotFoundError) as exc:
                failures.append((relative_audio, str(exc)))
                continue
            successes.append(str(summary["audio_relative"]))
            total_artifacts += int(summary["diarization_artifacts_removed"])
            total_summary_rows += int(summary["summary_rows_removed"])
            total_selection_lines += int(summary["selection_lines_removed"])

        if not successes:
            return self.redirect(
                environ,
                "/uploads",
                message=self.notification_message(
                    "No audio files were deleted.",
                    *[f"{name}: {reason}" for name, reason in failures[:8]],
                ),
                status="error",
            )

        details: list[str] = []
        if total_artifacts:
            details.append(f"Removed {total_artifacts} diarization artifact file(s) across runs.")
        if total_summary_rows:
            details.append(f"Removed {total_summary_rows} runtime_summary.tsv row(s).")
        if total_selection_lines:
            details.append(f"Removed {total_selection_lines} selected_audio.txt line(s).")
        if failures:
            details.append(
                f"{len(failures)} file(s) could not be deleted: "
                + ", ".join(f"{name} ({reason})" for name, reason in failures[:5])
            )
        headline = (
            f"Deleted {len(successes)} audio file(s)."
            if not failures
            else f"Deleted {len(successes)} of {len(relative_audio_values)} audio file(s)."
        )
        notice_status = "success" if not failures else "info"
        return self.redirect(
            environ,
            "/uploads",
            message=self.notification_message(headline, *details),
            status=notice_status,
        )

    def handle_audio_file_move(self, environ):
        """Move a WAV between audio set folders and rewrite associated history."""

        form = self.parse_form(environ)
        relative_audio = form.getfirst("audio_path") or ""
        target_folder = form.getfirst("target_folder") or ""
        try:
            summary = self.move_audio_file(relative_audio, target_folder)
        except FileNotFoundError as exc:
            return self.redirect(environ, "/uploads", message=str(exc), status="error")
        except ValueError as exc:
            return self.redirect(environ, "/uploads", message=str(exc), status="error")

        details: list[str] = []
        if summary["index_row_updated"]:
            details.append("Updated YouTube history index row to the new path.")
        if summary["summary_rows_updated"]:
            details.append(
                f"Rewrote {summary['summary_rows_updated']} runtime summary row(s) to the new path."
            )
        if summary["selection_lines_updated"]:
            details.append(
                f"Rewrote {summary['selection_lines_updated']} selected_audio.txt line(s) to the new path."
            )
        return self.redirect(
            environ,
            "/uploads",
            message=self.notification_message(
                f"Moved {summary['audio_relative_old']} -> {summary['audio_relative_new']}",
                *details,
            ),
            status="success",
        )

    def handle_audio_upload(self, environ):
        """Save uploaded source media as WAV and normalize numbering immediately."""

        form = self.parse_form(environ)
        if "audio_files" not in form:
            return self.redirect(
                environ,
                "/uploads",
                message=self.notification_message(
                    "Upload failed.",
                    "No files were selected.",
                    "Accepted by the shared NeMo and pyannote workflow: "
                    f"{', '.join(UPLOAD_AUDIO_SUFFIXES)}",
                ),
                status="error",
            )
        field = form["audio_files"]
        uploads = field if isinstance(field, list) else [field]
        saved_names: list[str] = []
        rejected_names: list[str] = []
        conversion_errors: list[str] = []
        default_upload_folder = (
            DEFAULT_UPLOAD_AUDIO_FOLDER
            if "audio_folder" in form or "new_audio_folder" in form
            else ""
        )
        folder_value, target_folder = self.selected_audio_folder_path(
            form,
            default_folder=default_upload_folder,
            create=False,
        )
        for item in uploads:
            if not hasattr(item, "file") or not getattr(item, "filename", ""):
                continue
            if not self.is_supported_uploaded_audio_name(item.filename):
                rejected_names.append(Path(item.filename).name)
                continue
            try:
                target_path = self.convert_uploaded_audio_to_wav(item, target_folder)
            except RuntimeError as exc:
                conversion_errors.append(f"{Path(item.filename).name}: {exc}")
                continue
            saved_names.append(self.audio_relative_path(target_path))

        if not saved_names:
            rejected_detail = (
                f"Rejected file(s): {', '.join(rejected_names[:8])}."
                if rejected_names
                else "No supported audio files were included in the upload."
            )
            conversion_detail = (
                f"Conversion failed: {'; '.join(conversion_errors[:4])}."
                if conversion_errors
                else ""
            )
            return self.redirect(
                environ,
                "/uploads",
                message=self.notification_message(
                    "Upload failed.",
                    rejected_detail,
                    conversion_detail,
                    "Accepted by the shared NeMo and pyannote workflow: "
                    f"{', '.join(UPLOAD_AUDIO_SUFFIXES)}",
                ),
                status="error",
            )

        renamed = normalize_audio_dir(self.audio_dir)
        self.invalidate_dashboard_cache()
        rename_note = f"Renumbered {len(renamed)} file(s)." if renamed else "No renumbering was needed."
        rejected_note = (
            f"Rejected unsupported file(s): {', '.join(rejected_names[:8])}."
            if rejected_names
            else ""
        )
        conversion_note = (
            f"Conversion failed for: {'; '.join(conversion_errors[:4])}."
            if conversion_errors
            else ""
        )
        status = "success" if not rejected_names and not conversion_errors else "info"
        return self.redirect(
            environ,
            "/uploads",
            message=self.notification_message(
                "Upload complete.",
                f"Saved {len(saved_names)} supported audio file(s) as WAV in {self.describe_path(target_folder)}.",
                rename_note,
                rejected_note,
                conversion_note,
            ),
            status=status,
        )

    def uploaded_file_items(self, form: cgi.FieldStorage, field_name: str) -> list[cgi.FieldStorage]:
        """Return all non-empty uploaded file fields for one multipart name."""

        if field_name not in form:
            return []
        field = form[field_name]
        items = field if isinstance(field, list) else [field]
        return [
            item
            for item in items
            if hasattr(item, "file") and getattr(item, "filename", "")
        ]

    def upload_stem(self, item: cgi.FieldStorage) -> str:
        """Return the sanitized filename stem used for upload pairing."""

        return Path(self.safe_filename(str(getattr(item, "filename", "")))).stem

    def upload_items_by_stem(
        self,
        items: list[cgi.FieldStorage],
        *,
        label: str,
    ) -> tuple[dict[str, cgi.FieldStorage], list[str]]:
        """Index uploaded files by stem and report duplicate stems clearly."""

        by_stem: dict[str, cgi.FieldStorage] = {}
        duplicates: list[str] = []
        for item in items:
            stem = self.upload_stem(item)
            if not stem:
                continue
            if stem in by_stem:
                duplicates.append(stem)
                continue
            by_stem[stem] = item
        if duplicates:
            duplicates = sorted(set(duplicates))
            return by_stem, [f"Duplicate {label} stem(s): {', '.join(duplicates[:8])}."]
        return by_stem, []

    def selected_workspace_paths(
        self,
        form: cgi.FieldStorage,
        field_name: str,
        *,
        label: str,
        suffixes: set[str],
        required_root: Path | None = None,
    ) -> tuple[list[Path], list[str]]:
        """Resolve multi-select workspace paths from a form without allowing escapes."""

        requested_values = self.ordered_unique(
            [value.strip() for value in form.getlist(field_name) if value.strip()]
        )
        paths: list[Path] = []
        errors: list[str] = []
        normalized_required_root = required_root.resolve() if required_root else None
        for raw_value in requested_values:
            try:
                candidate = self.resolve_under_root(raw_value)
            except ValueError as exc:
                errors.append(f"{label} '{raw_value}': {exc}")
                continue
            if not candidate.is_file():
                errors.append(f"{label} '{raw_value}' was not found in the SSH workspace.")
                continue
            if normalized_required_root is not None:
                try:
                    candidate.relative_to(normalized_required_root)
                except ValueError:
                    errors.append(f"{label} '{raw_value}' must be selected from {self.describe_path(normalized_required_root)}.")
                    continue
            if candidate.suffix.lower() not in suffixes:
                errors.append(f"{label} '{raw_value}' has an unsupported file type.")
                continue
            paths.append(candidate)
        return paths, errors

    def workspace_paths_by_stem(
        self,
        paths: list[Path],
        *,
        label: str,
    ) -> tuple[dict[str, Path], list[str]]:
        """Index already-resolved workspace files by stem and report ambiguous picks."""

        by_stem: dict[str, Path] = {}
        duplicates: list[str] = []
        for path in paths:
            stem = path.stem
            if not stem:
                continue
            if stem in by_stem:
                duplicates.append(stem)
                continue
            by_stem[stem] = path
        if duplicates:
            duplicates = sorted(set(duplicates))
            return by_stem, [f"Duplicate selected {label} stem(s): {', '.join(duplicates[:8])}."]
        return by_stem, []

    def rename_path_if_present(self, source: Path, target: Path) -> bool:
        """Rename one generated artifact if it exists and the destination is free."""

        if source == target or not source.is_file():
            return False
        if target.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        return True

    def rewrite_training_label_record_reference(self, old_relative: str, new_relative: str) -> int:
        """Keep saved label drafts attached when compact renumbering changes a filename."""

        records = self.load_training_label_records()
        if not records:
            return 0
        old_basename = Path(old_relative).name
        new_basename = Path(new_relative).name
        changed = 0
        for old_key in (old_relative, old_basename):
            if old_key not in records:
                continue
            record = dict(records.pop(old_key))
            record["audio_file"] = new_relative
            records[new_relative] = record
            changed += 1
            break
        if changed:
            self.save_training_label_records(records)
        return changed

    def rewrite_youtube_run_audio_reference(self, old_path: Path, new_path: Path) -> int:
        """Update per-run YouTube reports when compact renumbering changes a WAV path."""

        if not self.youtube_runs_root.is_dir():
            return 0
        old_resolved = old_path.resolve()
        updates = 0
        for run_dir in sorted(path for path in self.youtube_runs_root.iterdir() if path.is_dir()):
            report_path = run_dir / "conversion_report.tsv"
            rows = self.read_tsv_rows(report_path)
            if rows:
                fieldnames = tuple(rows[0].keys())
                changed = False
                for row in rows:
                    row_path = (row.get("audio_path") or "").strip()
                    path_matches = False
                    if row_path:
                        try:
                            path_matches = Path(row_path).expanduser().resolve() == old_resolved
                        except OSError:
                            path_matches = False
                    if path_matches or ((row.get("audio_file") or "").strip() == old_path.name and not row_path):
                        row["audio_path"] = str(new_path)
                        row["audio_file"] = new_path.name
                        changed = True
                if changed:
                    self.write_tsv_rows(report_path, fieldnames, rows)
                    updates += 1

            resolved_path = run_dir / "resolved_audio_paths.txt"
            if resolved_path.is_file():
                lines = resolved_path.read_text(encoding="utf-8", errors="replace").splitlines()
                rewritten: list[str] = []
                changed = False
                for line in lines:
                    try:
                        if Path(line).expanduser().resolve() == old_resolved:
                            rewritten.append(str(new_path))
                            changed = True
                            continue
                    except OSError:
                        pass
                    rewritten.append(line)
                if changed:
                    resolved_path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")
                    updates += 1
        return updates

    def rewrite_audio_file_references_after_rename(
        self,
        old_path: Path,
        new_path: Path,
        *,
        old_relative: str,
        new_relative: str,
    ) -> dict[str, int]:
        """Rewrite workspace references after compact numbering renames a media file."""

        summary = {
            "youtube_index_rows_updated": 0,
            "youtube_run_files_updated": 0,
            "diarization_artifacts_renamed": 0,
            "summary_rows_updated": 0,
            "selection_lines_updated": 0,
            "training_label_records_updated": 0,
        }

        if self.youtube_history_index_path.is_file():
            rows = self.read_tsv_rows(self.youtube_history_index_path)
            changed = False
            for row in rows:
                if self.row_audio_path_matches(row, old_path):
                    row["audio_path"] = str(new_path)
                    row["audio_file"] = new_path.name
                    changed = True
                    summary["youtube_index_rows_updated"] += 1
            if changed:
                self.write_tsv_rows(self.youtube_history_index_path, YOUTUBE_INDEX_COLUMNS, rows)

        summary["youtube_run_files_updated"] = self.rewrite_youtube_run_audio_reference(old_path, new_path)

        old_stem = self.diarization_output_base(old_relative)
        new_stem = self.diarization_output_base(new_relative)
        old_log_stem = self.safe_log_component(old_relative)
        new_log_stem = self.safe_log_component(new_relative)
        if self.diarization_runs_root.is_dir():
            for run_dir in iter_diarization_run_directories(self.diarization_runs_root):
                for suffix in DIARIZATION_ARTIFACT_SUFFIXES:
                    if self.rename_path_if_present(run_dir / f"{old_stem}{suffix}", run_dir / f"{new_stem}{suffix}"):
                        summary["diarization_artifacts_renamed"] += 1
                log_dir = run_dir / "logs"
                for suffix in (".out", ".err"):
                    if self.rename_path_if_present(log_dir / f"{old_log_stem}{suffix}", log_dir / f"{new_log_stem}{suffix}"):
                        summary["diarization_artifacts_renamed"] += 1
                summary_changed, selection_changed = self._rewrite_diarization_run_references(
                    run_dir,
                    old_relative,
                    new_relative=new_relative,
                )
                summary["summary_rows_updated"] += summary_changed
                summary["selection_lines_updated"] += selection_changed

        summary["training_label_records_updated"] = self.rewrite_training_label_record_reference(
            old_relative,
            new_relative,
        )
        return summary

    def compact_audio_folder_numbering(self, folder_path: Path) -> list[dict[str, str]]:
        """Renumber one audio folder to a contiguous 001, 002, 003 sequence."""

        if not folder_path.is_dir():
            return []
        files = [
            path.resolve()
            for path in folder_path.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in WORKSPACE_MEDIA_SUFFIXES
                and "_whisper_input" not in path.stem
            )
        ]

        def sort_key(path: Path) -> tuple[int, str, str]:
            match = NUMBERED_PREFIX.match(path.name)
            number = int(match.group(1)) if match else 1_000_000
            base_name = path.name[4:] if match else path.name
            return number, base_name.lower(), path.name.lower()

        planned: list[tuple[Path, Path, str, str]] = []
        for index, source in enumerate(sorted(files, key=sort_key), start=1):
            match = NUMBERED_PREFIX.match(source.name)
            base_name = source.name[4:] if match else source.name
            target = source.with_name(f"{index:03d}_{base_name}")
            if source == target:
                continue
            planned.append(
                (
                    source,
                    target,
                    self.audio_relative_path(source),
                    target.resolve().relative_to(self.audio_dir.resolve()).as_posix(),
                )
            )
        if not planned:
            return []

        source_paths = {source for source, _, _, _ in planned}
        for _, target, _, _ in planned:
            if target.exists() and target.resolve() not in source_paths:
                raise FileExistsError(f"Cannot renumber because target already exists: {target}")

        temp_moves: list[tuple[Path, Path, str, str]] = []
        for source, target, old_relative, new_relative in planned:
            temp_path = source.with_name(f".renumber-{secrets.token_hex(8)}-{source.name}")
            source.rename(temp_path)
            temp_moves.append((temp_path, target, old_relative, new_relative))

        renamed: list[dict[str, str]] = []
        for temp_path, target, old_relative, new_relative in temp_moves:
            temp_path.rename(target)
            reference_summary = self.rewrite_audio_file_references_after_rename(
                self.audio_dir / old_relative,
                target.resolve(),
                old_relative=old_relative,
                new_relative=new_relative,
            )
            renamed.append(
                {
                    "old": old_relative,
                    "new": new_relative,
                    **{key: str(value) for key, value in reference_summary.items()},
                }
            )

        self.invalidate_dashboard_cache()
        return renamed

    def _rewrite_diarization_run_references(
        self,
        run_dir: Path,
        old_relative: str,
        *,
        new_relative: str | None,
    ) -> tuple[int, int]:
        """Rewrite (or remove) every per-audio reference inside one run folder.

        When ``new_relative`` is ``None`` the matching ``runtime_summary.tsv``
        rows and ``selected_audio.txt`` lines are dropped (used by cascading
        delete). Otherwise, those references are rewritten to the new path
        (used by move). Returns ``(summary_rows_changed, selection_lines_changed)``.
        """

        summary_rows_changed = 0
        selection_lines_changed = 0
        old_log_stem = self.safe_log_component(old_relative)
        new_log_stem = self.safe_log_component(new_relative or "")
        summary_path = run_dir / "logs" / "runtime_summary.tsv"
        if summary_path.is_file():
            rows = self.read_tsv_rows(summary_path)
            if rows:
                fieldnames = tuple(rows[0].keys())
                if new_relative is None:
                    kept_rows = [
                        row
                        for row in rows
                        if self.clean_audio_selection_value(row.get("audio_file", "")) != old_relative
                    ]
                    if len(kept_rows) != len(rows):
                        summary_rows_changed = len(rows) - len(kept_rows)
                        self.write_tsv_rows(summary_path, fieldnames, kept_rows)
                else:
                    changed = False
                    for row in rows:
                        if self.clean_audio_selection_value(row.get("audio_file", "")) == old_relative:
                            row["audio_file"] = new_relative
                            changed = True
                            summary_rows_changed += 1
                            for field in ("stdout_log", "stderr_log"):
                                raw_path = (row.get(field) or "").strip()
                                if not raw_path:
                                    continue
                                log_path = Path(raw_path)
                                if log_path.name in {f"{old_log_stem}.out", f"{old_log_stem}.err"}:
                                    row[field] = str(log_path.with_name(f"{new_log_stem}{log_path.suffix}"))
                    if changed:
                        self.write_tsv_rows(summary_path, fieldnames, rows)
        selection_path = run_dir / "selected_audio.txt"
        if selection_path.is_file():
            lines = selection_path.read_text(encoding="utf-8", errors="replace").splitlines()
            rewritten: list[str] = []
            changed = False
            for line in lines:
                if self.clean_audio_selection_value(line.strip()) == old_relative:
                    if new_relative is None:
                        changed = True
                        selection_lines_changed += 1
                        continue
                    rewritten.append(new_relative)
                    changed = True
                    selection_lines_changed += 1
                else:
                    rewritten.append(line)
            if changed:
                text = "\n".join(rewritten)
                if rewritten:
                    text += "\n"
                selection_path.write_text(text, encoding="utf-8")
        return summary_rows_changed, selection_lines_changed

    def cascade_delete_audio_file(self, relative_audio: str) -> dict[str, object]:
        """Delete a WAV plus every artifact tied to it across the workspace.

        Removes (a) the WAV itself, (b) the row in
        ``outputs/youtube_conversion_history/url_audio_index.tsv`` that points
        at it, (c) every per-stem artifact in every diarization run folder, and
        (d) the matching rows in each run's ``logs/runtime_summary.tsv`` and
        lines in ``selected_audio.txt``. The URL line in ``youtube_links.txt``
        is intentionally **kept** so the next conversion run will redownload
        the audio fresh.

        The operation is **idempotent**: if the WAV is already gone (the user
        clicked Delete on a stale row, or another process removed the file
        first) we still sweep every orphaned reference and return success.
        Surfacing a "not found" error in that case would be both confusing and
        wrong — the user's intent was "make this row stop existing," and after
        the sweep that intent is satisfied.
        """

        cleaned = self.clean_audio_selection_value(relative_audio)
        if not cleaned:
            raise ValueError(f"Invalid audio path: {relative_audio!r}")
        try:
            target: Path | None = self.locate_audio_file(relative_audio)
            cleaned = self.audio_relative_path(target)
            audio_basename = target.name
            audio_path_str = str(target)
        except FileNotFoundError:
            target = None
            audio_basename = Path(cleaned).name
            audio_path_str = ""

        summary: dict[str, object] = {
            "audio_relative": cleaned,
            "audio_basename": audio_basename,
            "audio_path": audio_path_str,
            "audio_was_present": target is not None,
            "youtube_url": "",
            "diarization_artifacts_removed": 0,
            "summary_rows_removed": 0,
            "selection_lines_removed": 0,
        }

        # Drop every url_audio_index.tsv row that still points at this WAV.
        # When the file is present we match on resolved path equality (most
        # precise). When it is already gone, fall back to the recorded basename
        # so dangling rows from a previous deletion still get swept.
        if self.youtube_history_index_path.is_file():
            target_resolved: Path | None = None
            if target is not None:
                try:
                    target_resolved = target.resolve()
                except OSError:
                    target_resolved = target
            kept_rows: list[dict[str, str]] = []
            removed_urls: list[str] = []
            for row in self.read_tsv_rows(self.youtube_history_index_path):
                row_path = (row.get("audio_path") or "").strip()
                row_matches = False
                if target_resolved is not None and row_path:
                    try:
                        row_matches = Path(row_path).resolve() == target_resolved
                    except OSError:
                        row_matches = False
                if not row_matches and audio_basename:
                    if (row.get("audio_file") or "").strip() == audio_basename:
                        row_matches = True
                if row_matches:
                    url = (row.get("url") or "").strip()
                    if url:
                        removed_urls.append(url)
                    continue
                kept_rows.append(row)
            if removed_urls:
                self.write_tsv_rows(self.youtube_history_index_path, YOUTUBE_INDEX_COLUMNS, kept_rows)
                summary["youtube_url"] = removed_urls[0]
                summary["youtube_urls_cleared"] = removed_urls

        stem = self.diarization_output_base(cleaned)
        artifacts_removed = 0
        summary_rows_removed = 0
        selection_lines_removed = 0
        runs_root = self.diarization_runs_root
        if runs_root.is_dir():
            for run_dir in iter_diarization_run_directories(runs_root):
                for suffix in DIARIZATION_ARTIFACT_SUFFIXES:
                    artifact = run_dir / f"{stem}{suffix}"
                    if artifact.is_file():
                        artifact.unlink()
                        artifacts_removed += 1
                summary_changed, selection_changed = self._rewrite_diarization_run_references(
                    run_dir,
                    cleaned,
                    new_relative=None,
                )
                summary_rows_removed += summary_changed
                selection_lines_removed += selection_changed

        if target is not None and target.is_file():
            target.unlink()
        summary["diarization_artifacts_removed"] = artifacts_removed
        summary["summary_rows_removed"] = summary_rows_removed
        summary["selection_lines_removed"] = selection_lines_removed
        self.invalidate_dashboard_cache()
        return summary

    def move_audio_file(
        self,
        relative_audio: str,
        target_folder: str,
    ) -> dict[str, object]:
        """Move a WAV into ``target_folder`` and rewrite every reference to it.

        After the move, every ``logs/runtime_summary.tsv`` row and
        ``selected_audio.txt`` line that referenced the old relative path is
        rewritten to point at the new path, and the ``url_audio_index.tsv`` row
        (if any) is updated to record the new ``audio_path``/``audio_file``.
        Falls back to a basename search across ``audio_in`` when the form's
        ``audio_path`` is stale (the file was relocated since the page rendered).
        """

        source = self.locate_audio_file(relative_audio)
        cleaned = self.audio_relative_path(source)

        target_folder_value = self.audio_folder_value(target_folder)
        target_folder_path = self.audio_folder_path(target_folder_value, create=True)
        if source.parent.resolve() == target_folder_path.resolve():
            raise ValueError("Audio file is already in that folder.")

        destination = self.choose_available_audio_path(target_folder_path / source.name)
        shutil.move(str(source), str(destination))
        new_relative = self.audio_relative_path(destination)

        summary: dict[str, object] = {
            "audio_relative_old": cleaned,
            "audio_relative_new": new_relative,
            "audio_path_new": str(destination),
            "summary_rows_updated": 0,
            "selection_lines_updated": 0,
            "index_row_updated": False,
        }

        if self.youtube_history_index_path.is_file():
            rows = self.read_tsv_rows(self.youtube_history_index_path)
            changed = False
            for row in rows:
                row_path = (row.get("audio_path") or "").strip()
                if not row_path:
                    continue
                try:
                    if Path(row_path).resolve() == source:
                        row["audio_path"] = str(destination)
                        row["audio_file"] = destination.name
                        changed = True
                except OSError:
                    continue
            if changed:
                self.write_tsv_rows(self.youtube_history_index_path, YOUTUBE_INDEX_COLUMNS, rows)
                summary["index_row_updated"] = True

        runs_root = self.diarization_runs_root
        summary_rows_updated = 0
        selection_lines_updated = 0
        if runs_root.is_dir():
            for run_dir in iter_diarization_run_directories(runs_root):
                summary_changed, selection_changed = self._rewrite_diarization_run_references(
                    run_dir,
                    cleaned,
                    new_relative=new_relative,
                )
                summary_rows_updated += summary_changed
                selection_lines_updated += selection_changed
        summary["summary_rows_updated"] = summary_rows_updated
        summary["selection_lines_updated"] = selection_lines_updated

        self.invalidate_dashboard_cache()
        return summary

    def cascade_delete_audio_folder(self, folder_value: str) -> dict[str, object]:
        """Delete every media file inside ``folder_value`` with full cascade.

        The unsorted root is a virtual folder backed by ``audio_in`` itself, so
        deleting it clears only root-level media and leaves subfolders plus the
        required ``audio_in`` directory in place.
        """

        raw_value = (folder_value or "").strip()
        if not raw_value:
            raise ValueError("Choose an audio folder to delete.")
        is_root = raw_value == ROOT_AUDIO_FOLDER_VALUE
        normalized = self.audio_folder_value(folder_value)
        folder_path = self.audio_folder_path(raw_value)
        if not folder_path.is_dir():
            return {
                "folder": ROOT_AUDIO_FOLDER_VALUE if is_root else normalized,
                "files_removed": 0,
                "diarization_artifacts_removed": 0,
                "summary_rows_removed": 0,
                "selection_lines_removed": 0,
                "urls_removed": [],
                "already_missing": True,
            }

        files_removed = 0
        diarization_artifacts_removed = 0
        summary_rows_removed = 0
        selection_lines_removed = 0
        urls_removed: list[str] = []
        candidates = folder_path.iterdir() if is_root else folder_path.rglob("*")
        for wav_path in sorted(candidates):
            if not wav_path.is_file():
                continue
            if wav_path.suffix.lower() not in WORKSPACE_MEDIA_SUFFIXES:
                continue
            relative = self.audio_relative_path(wav_path)
            file_summary = self.cascade_delete_audio_file(relative)
            files_removed += 1
            diarization_artifacts_removed += int(file_summary["diarization_artifacts_removed"])
            summary_rows_removed += int(file_summary["summary_rows_removed"])
            selection_lines_removed += int(file_summary["selection_lines_removed"])
            url = str(file_summary["youtube_url"])
            if url:
                urls_removed.append(url)

        if not is_root and folder_path.exists():
            for stray in sorted(folder_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if stray.is_file():
                    try:
                        stray.unlink()
                    except FileNotFoundError:
                        continue
            if folder_path.is_dir():
                shutil.rmtree(folder_path)

        self.invalidate_dashboard_cache()
        return {
            "folder": ROOT_AUDIO_FOLDER_VALUE if is_root else normalized,
            "files_removed": files_removed,
            "diarization_artifacts_removed": diarization_artifacts_removed,
            "summary_rows_removed": summary_rows_removed,
            "selection_lines_removed": selection_lines_removed,
            "urls_removed": urls_removed,
            "already_missing": False,
        }

    def selected_audio_names(self, requested_names: list[str], *, run_all: bool) -> list[str]:
        """Return current numbered audio names after normalizing the input library."""

        renamed = normalize_audio_dir(self.audio_dir)
        if renamed:
            self.invalidate_dashboard_cache()
        inventory = [self.audio_relative_path(path) for path in iter_audio_files(self.audio_dir)]
        if run_all:
            return inventory

        inventory_set = set(inventory)
        resolved: list[str] = []
        for requested_name in requested_names:
            cleaned = self.clean_audio_selection_value(requested_name)
            if cleaned in inventory_set:
                resolved.append(cleaned)
                continue
            numbered_matches = [
                name
                for name in inventory
                if (
                    Path(name).name == cleaned
                    or (
                        Path(name).name[4:] == cleaned
                        and len(Path(name).name) > 4
                        and Path(name).name[:3].isdigit()
                        and Path(name).name[3] == "_"
                    )
                )
            ]
            if len(numbered_matches) == 1:
                resolved.append(numbered_matches[0])
        return self.ordered_unique(resolved)
