#!/usr/bin/env python3
"""Serve a lightweight web dashboard for the ML Speech Diarization workflow.

The site deliberately remains small and dependency-light so it can run inside the
same Python environment as the rest of the project. The operational model mirrors
the local WAVE documentation: interactive browsing and setup can happen in a web
session, while longer-running GPU work should be handed off to Slurm-backed jobs.
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

CODE_ROOT = Path(__file__).resolve().parent
FRONTEND_TEMPLATE = "html/dashboard/index.html"
NAV_PATHS = [
    "/",
    "/uploads",
    "/youtube",
    "/diarization",
    "/training-labels",
    "/fine-tuning",
]
PAGE_ALIASES = {
    "/jobs": "/diarization",
    "/direct": "/diarization",
    "/models": "/diarization",
    "/review": "/uploads",
    "/labeling": "/training-labels",
    "/labels": "/training-labels",
    "/training-labeling": "/training-labels",
}
PAGE_PATHS = set(NAV_PATHS)
AUDIO_INVENTORY_PAGES = {"/uploads", "/training-labels", "/diarization", "/fine-tuning"}
PROJECT_SUMMARY_PAGES = {"/", "/training-labels", "/fine-tuning"}
RECENT_OUTPUT_PAGES = {"/", "/diarization", "/youtube"}
RECENT_SRT_PAGES: set[str] = set()
YOUTUBE_QUEUE_PAGES = {"/youtube"}
MODEL_SELECTION_PAGES = {"/training-labels", "/diarization", "/fine-tuning"}
TRAINING_LABEL_CONTEXT_PAGES = {"/training-labels", "/fine-tuning"}
DEFAULT_TRAINING_LABEL_PROJECT = "uploaded-site-training"
DIARIZATION_LABEL_PREVIEW_LIMIT = 2000
ARTIFACT_SCAN_EXCLUDE_DIRS = {"__pycache__", "audio", "rttm", "text"}
TRAINING_SOURCE_SCAN_EXCLUDE_DIRS = {"__pycache__", "artifacts", "runs", "experiments", "slurm_logs"}
TRAINING_RTTM_SUFFIXES = {".rttm"}
TRAINING_TRANSCRIPT_SUFFIXES = {".txt", ".srt", ".vtt", ".csv", ".tsv"}
WORKSPACE_MEDIA_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
    ".webm",
}
DEFAULT_UPLOAD_AUDIO_FOLDER = "file_uploads"
DEFAULT_YOUTUBE_AUDIO_FOLDER = "youtube_links"
DEFAULT_AUDIO_FOLDERS = (DEFAULT_UPLOAD_AUDIO_FOLDER, DEFAULT_YOUTUBE_AUDIO_FOLDER)
ROOT_AUDIO_FOLDER_VALUE = "__root__"
YOUTUBE_INDEX_COLUMNS = (
    "url",
    "video_id",
    "status",
    "audio_file",
    "audio_path",
    "title",
    "last_attempt_utc",
    "note",
)
DIARIZATION_ARTIFACT_SUFFIXES = (".txt", ".srt", "_review.html", "_review_flags.tsv")
YOUTUBE_NO_DATA_MARKERS = (
    "video unavailable",
    "this video is unavailable",
    "private video",
    "video has been removed",
    "the uploader has not made this video available",
    "this content isn't available",
    "content is not available",
    "members-only",
    "sign in to confirm your age",
    "age-restricted",
    "unsupported url",
)
UPLOAD_AUDIO_SUFFIXES = tuple(
    sorted(extension for extension in AUDIO_EXTENSIONS if extension not in {".mp4", ".mkv", ".webm"})
)
UPLOAD_AUDIO_ACCEPT = ",".join(UPLOAD_AUDIO_SUFFIXES)
DIARIZATION_COMPLETED_STATUSES = {"ok", "no_speech"}
DIARIZATION_ACTIVE_STATUSES = {"running", "submitted", "pending", "waiting"}
DASHBOARD_REFRESH_STATUSES = {"running", "submitted", "pending", "waiting", "configuring"}
LIVE_TRACKING_INTERVAL_MS = 1000
IDLE_TRACKING_INTERVAL_MS = 5000
DEFAULT_SERVER_MODE = "auto"
DEFAULT_SERVER_THREADS = 8
SECURITY_RESPONSE_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
)
DOWNLOADABLE_ROOT_NAMES = (
    "audio_in",
    "outputs",
    "fine_tuning",
    "job_outputs",
    "job_logs",
    "youtube_links_err",
)
TEXT_PREVIEW_SUFFIXES = {
    ".csv",
    ".err",
    ".htm",
    ".html",
    ".json",
    ".log",
    ".md",
    ".out",
    ".rttm",
    ".srt",
    ".tsv",
    ".txt",
    ".uem",
    ".yaml",
    ".yml",
}
TAIL_PREVIEW_SUFFIXES = {".err", ".log", ".out"}
ARTIFACT_PREVIEW_BYTE_LIMIT = 16000
ARTIFACT_PREVIEW_LINE_LIMIT = 80


class ThreadedWSGIServer(ThreadingMixIn):
    """Allow the built-in WSGI server to handle concurrent browser requests."""

    daemon_threads = True


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


class WorkflowWebApp:
    """WSGI application that exposes the main project workflows through one dashboard."""

    def __init__(self, root: Path = PROJECT_ROOT):
        self.root = root.resolve()
        self._dashboard_cache: dict[str, tuple[float, object]] = {}
        self._dashboard_cache_lock = threading.RLock()

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio_in"

    @property
    def youtube_links_path(self) -> Path:
        return self.root / "youtube_links.txt"

    @property
    def failed_links_path(self) -> Path:
        return self.root / "youtube_links_err" / "failed_links_latest.tsv"

    @property
    def single_output_root(self) -> Path:
        return self.root / "job_outputs" / "single_diarization"

    @property
    def bulk_output_root(self) -> Path:
        return self.root / "job_outputs" / "bulk_diarization"

    @property
    def local_dashboard_dir(self) -> Path:
        return self.root / ".local_dashboard"

    @property
    def dashboard_secrets_path(self) -> Path:
        return self.local_dashboard_dir / "secrets.env"

    @property
    def training_label_status_path(self) -> Path:
        return self.root / "fine_tuning" / "label_status.json"

    @property
    def training_label_work_dir(self) -> Path:
        return self.root / "fine_tuning" / "label_work"

    @property
    def outputs_root(self) -> Path:
        return self.root / OUTPUTS_ROOT.name

    @property
    def frontend_dir(self) -> Path:
        workspace_frontend = self.root / "frontend"
        return workspace_frontend if workspace_frontend.exists() else CODE_ROOT / "frontend"

    @property
    def dashboard_template_path(self) -> Path:
        return self.frontend_dir / FRONTEND_TEMPLATE

    @property
    def site_diarization_sbatch(self) -> Path:
        return self.root / "scheduler" / "run_site_diarization.sbatch"

    @property
    def site_youtube_sbatch(self) -> Path:
        return self.root / "scheduler" / "run_site_youtube_conversion.sbatch"

    @property
    def diarization_runs_root(self) -> Path:
        return self.outputs_root / DIARIZATION_RUNS_ROOT.name

    def diarization_backend_runs_root(self, backend: str) -> Path:
        """Return the model-specific folder used for new diarization runs."""

        return self.diarization_runs_root / normalize_diarization_backend(backend)

    @property
    def youtube_runs_root(self) -> Path:
        return self.outputs_root / YOUTUBE_RUNS_ROOT.name

    @property
    def youtube_history_index_path(self) -> Path:
        return self.outputs_root / YOUTUBE_HISTORY_INDEX.parent.name / YOUTUBE_HISTORY_INDEX.name

    def __call__(self, environ, start_response):
        """Route one HTTP request to the matching handler."""

        try:
            method = (environ.get("REQUEST_METHOD") or "GET").upper()
            head_only = method == "HEAD"
            routed_method = "GET" if head_only else method
            raw_path = self.normalize_request_path(environ)
            path = PAGE_ALIASES.get(raw_path, raw_path)
            if routed_method == "GET" and path in PAGE_PATHS:
                status, headers, body = self.page_response(environ, path)
            elif routed_method == "GET" and path == "/api/tracking":
                status, headers, body = self.handle_tracking_status(environ)
            elif routed_method == "GET" and path == "/api/page-state":
                status, headers, body = self.handle_page_state(environ)
            elif routed_method == "GET" and path == "/api/artifact-preview":
                status, headers, body = self.handle_artifact_preview(environ)
            elif routed_method == "GET" and path == "/health":
                status, headers, body = self.text_response("200 OK", "ok\n")
            elif routed_method == "GET" and path.startswith("/assets/"):
                status, headers, body = self.serve_frontend_asset(path)
            elif routed_method == "GET" and path.startswith("/files/"):
                status, headers, body = self.serve_file(path, environ)
            elif routed_method == "POST" and path == "/upload/audio":
                status, headers, body = self.handle_audio_upload(environ)
            elif routed_method == "POST" and path == "/audio-folders/create":
                status, headers, body = self.handle_audio_folder_create(environ)
            elif routed_method == "POST" and path == "/audio-folders/rename":
                status, headers, body = self.handle_audio_folder_rename(environ)
            elif routed_method == "POST" and path == "/audio-folders/delete":
                status, headers, body = self.handle_audio_folder_delete(environ)
            elif routed_method == "POST" and path == "/audio-files/delete":
                status, headers, body = self.handle_audio_file_delete(environ)
            elif routed_method == "POST" and path == "/audio-files/bulk-delete":
                status, headers, body = self.handle_audio_files_bulk_delete(environ)
            elif routed_method == "POST" and path == "/audio-files/move":
                status, headers, body = self.handle_audio_file_move(environ)
            elif routed_method == "POST" and path == "/training-labels/save":
                status, headers, body = self.handle_training_label_save(environ)
            elif routed_method == "POST" and path == "/youtube-links":
                status, headers, body = self.handle_youtube_links(environ)
            elif routed_method == "POST" and path == "/youtube-links/delete":
                status, headers, body = self.handle_youtube_link_delete(environ)
            elif routed_method == "POST" and path == "/actions/convert-youtube":
                status, headers, body = self.handle_youtube_conversion(environ)
            elif routed_method == "POST" and path == "/actions/reset-youtube-workspace":
                status, headers, body = self.handle_youtube_reset(environ)
            elif routed_method == "POST" and path == "/actions/review":
                status, headers, body = self.handle_review(environ)
            elif routed_method == "POST" and path == "/actions/run-diarization":
                status, headers, body = self.handle_diarization_run(environ)
            elif routed_method == "POST" and path == "/actions/test":
                status, headers, body = self.handle_tests(environ)
            elif routed_method == "POST" and path == "/models/save":
                status, headers, body = self.handle_model_preferences(environ)
            elif routed_method == "POST" and path == "/fine-tuning/upload-sample":
                status, headers, body = self.handle_finetune_upload(environ)
            elif routed_method == "POST" and path == "/fine-tuning/prepare":
                status, headers, body = self.handle_finetune_prepare(environ)
            elif routed_method == "POST" and path == "/fine-tuning/launch":
                status, headers, body = self.handle_finetune_launch(environ)
            else:
                status, headers, body = self.text_response("404 Not Found", "Not found\n")
        except Exception as exc:  # pragma: no cover - safety net for interactive use
            status, headers, body = self.error_response(exc, environ)

        start_response(status, headers)
        return [] if head_only else body

    def parse_form(self, environ) -> cgi.FieldStorage:
        """Parse form data for both standard posts and file uploads."""

        return cgi.FieldStorage(fp=environ["wsgi.input"], environ=environ, keep_blank_values=True)

    def get_query_message(self, environ) -> tuple[str, str]:
        """Read flash-style status messaging from the query string."""

        query = parse_qs(environ.get("QUERY_STRING", ""))
        message = query.get("message", [""])[0]
        status = query.get("status", ["info"])[0]
        return message, status

    def normalize_request_path(self, environ) -> str:
        """Normalize the request path after removing any reverse-proxy prefix."""

        path = environ.get("PATH_INFO") or "/"
        script_name = self.script_name(environ)
        if script_name and path == script_name:
            path = "/"
        elif script_name and path.startswith(f"{script_name}/"):
            path = path[len(script_name):] or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        return path

    def script_name(self, environ) -> str:
        """Return the application prefix used by reverse proxies, if any.

        Open OnDemand and similar proxy layers do not always agree on whether the
        prefix is exposed through `SCRIPT_NAME` or a forwarded header, so the web
        layer accepts the common variants used in practice. Some deployments only
        preserve the original browser URL in `REQUEST_URI`-style fields, so this
        method derives a prefix from those values when needed.
        """

        explicit_prefix = (
            environ.get("SCRIPT_NAME")
            or environ.get("HTTP_X_FORWARDED_PREFIX")
            or environ.get("HTTP_X_SCRIPT_NAME")
            or ""
        ).rstrip("/")
        if explicit_prefix:
            return explicit_prefix

        path_info = environ.get("PATH_INFO") or "/"
        for key in (
            "HTTP_X_ORIGINAL_URI",
            "HTTP_X_FORWARDED_URI",
            "REQUEST_URI",
            "RAW_URI",
        ):
            raw_uri = environ.get(key)
            if not raw_uri:
                continue
            request_path = urlsplit(raw_uri).path or "/"
            inferred_prefix = self.infer_prefix_from_request_path(
                request_path=request_path,
                path_info=path_info,
            )
            if inferred_prefix is not None:
                return inferred_prefix
        return ""

    def infer_prefix_from_request_path(self, *, request_path: str, path_info: str) -> str | None:
        """Infer a proxy prefix by comparing the browser path to the app path."""

        if not request_path.startswith("/"):
            return None

        normalized_request = request_path.rstrip("/") or "/"
        normalized_path = (path_info or "/").rstrip("/") or "/"
        if normalized_request == normalized_path:
            return ""
        if normalized_path == "/":
            return normalized_request.rstrip("/")
        if normalized_request.endswith(normalized_path):
            return normalized_request[: -len(normalized_path)].rstrip("/")
        return None

    def with_prefix(self, path: str, script_name: str) -> str:
        """Attach the reverse-proxy prefix to an application-local path."""

        if not script_name:
            return path
        return f"{script_name}{path}" if path.startswith("/") else f"{script_name}/{path}"

    def redirect(self, environ, location: str, *, message: str, status: str = "info"):
        """Redirect back to the dashboard while preserving a status message."""

        separator = "&" if "?" in location else "?"
        url = (
            f"{self.with_prefix(location, self.script_name(environ))}"
            f"{separator}{urlencode({'message': message, 'status': status})}"
        )
        return "303 See Other", [("Location", url)], [b""]

    def text_response(self, status: str, text: str):
        """Return a plain-text WSGI response."""

        body = text.encode("utf-8")
        return (
            status,
            self.response_headers(
                content_type="text/plain; charset=utf-8",
                body_length=len(body),
            ),
            [body],
        )

    def json_response(self, status: str, payload: object):
        """Return a no-store JSON WSGI response for live dashboard state."""

        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return status, self.response_headers(
            content_type="application/json; charset=utf-8",
            body_length=len(body),
        ), [body]

    def html_response(self, status: str, text: str):
        """Return an HTML WSGI response."""

        body = text.encode("utf-8")
        return (
            status,
            self.response_headers(
                content_type="text/html; charset=utf-8",
                body_length=len(body),
            ),
            [body],
        )

    def error_response(self, exc: Exception, environ):
        """Render unexpected failures inside the dashboard instead of a raw traceback."""

        body = self.render_page(
            message=str(exc),
            message_status="error",
            script_name=self.script_name(environ),
            current_path=self.normalize_request_path(environ),
        )
        return self.html_response("500 Internal Server Error", body)

    def resolve_local_path(self, raw_path: str) -> Path:
        """Resolve a possibly relative path against the project root."""

        candidate = Path(raw_path).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

    def decode_workspace_route_path(self, raw_path: str) -> str:
        """Decode route paths that may arrive percent-escaped or mojibake-encoded.

        Some WSGI servers expose `PATH_INFO` after URL decoding while preserving
        bytes through latin-1, which turns UTF-8 filenames into mojibake. This
        helper accepts either representation and returns the intended workspace
        relative path.
        """

        decoded = unquote(raw_path)
        try:
            return decoded.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return decoded

    def resolve_under_root(self, raw_path: str) -> Path:
        """Resolve a path and reject anything outside the workspace tree.

        Compares against ``self.root.resolve()`` (not the raw ``self.root``)
        so a candidate that traverses through symlinks back outside the
        project still gets rejected. Without the resolve on the comparison
        side, a path like ``/some/symlink/that/points/elsewhere`` could pass
        the relative_to check.
        """

        candidate = Path(raw_path).expanduser()
        root_resolved = self.root.resolve()
        resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"Path is outside the project workspace: {raw_path}") from exc
        return resolved

    def resolve_under_frontend(self, raw_path: str) -> Path:
        """Resolve a frontend asset path and reject anything outside `frontend/`."""

        relative_path = self.decode_workspace_route_path(raw_path).lstrip("/")
        resolved = (self.frontend_dir / relative_path).resolve()
        try:
            resolved.relative_to(self.frontend_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Path is outside the frontend asset folder: {raw_path}") from exc
        return resolved

    def response_headers(
        self,
        *,
        content_type: str,
        body_length: int,
        cache_control: str = "no-store",
    ) -> list[tuple[str, str]]:
        """Build consistent response headers for dashboard pages and downloads."""

        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(body_length)),
        ]
        if cache_control:
            headers.append(("Cache-Control", cache_control))
        headers.extend(SECURITY_RESPONSE_HEADERS)
        return headers

    def is_downloadable_path(self, file_path: Path) -> bool:
        """Allow downloads only from public artifact folders inside the workspace."""

        try:
            relative = file_path.resolve().relative_to(self.root)
        except ValueError:
            return False
        if not relative.parts or any(part.startswith(".") for part in relative.parts):
            return False
        return any(relative == Path(name) or Path(name) in relative.parents for name in DOWNLOADABLE_ROOT_NAMES)

    def serve_frontend_asset(self, path: str):
        """Serve React, CSS, and other frontend assets from the dedicated folder."""

        relative = path.removeprefix("/assets/")
        if not relative:
            return self.text_response("404 Not Found", "Not found\n")
        asset_path = self.resolve_under_frontend(relative)
        if not asset_path.is_file():
            return self.text_response("404 Not Found", "Not found\n")

        content_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
        body = asset_path.read_bytes()
        return "200 OK", self.response_headers(content_type=content_type, body_length=len(body)), [body]

    def range_response_for_file(self, file_path: Path, content_type: str, range_header: str):
        """Serve one byte range so browser media seeking works for `/files/...` audio."""

        file_size = file_path.stat().st_size
        raw_range = range_header.strip()
        if not raw_range.startswith("bytes=") or "," in raw_range:
            return None
        start_text, separator, end_text = raw_range.removeprefix("bytes=").partition("-")
        if not separator:
            return None
        try:
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else file_size - 1
            else:
                suffix_length = int(end_text)
                if suffix_length <= 0:
                    raise ValueError
                start = max(file_size - suffix_length, 0)
                end = file_size - 1
        except ValueError:
            return (
                "416 Range Not Satisfiable",
                [
                    *self.response_headers(content_type=content_type, body_length=0),
                    ("Accept-Ranges", "bytes"),
                    ("Content-Range", f"bytes */{file_size}"),
                ],
                [b""],
            )
        if start < 0 or start >= file_size or end < start:
            return (
                "416 Range Not Satisfiable",
                [
                    *self.response_headers(content_type=content_type, body_length=0),
                    ("Accept-Ranges", "bytes"),
                    ("Content-Range", f"bytes */{file_size}"),
                ],
                [b""],
            )
        end = min(end, file_size - 1)
        length = end - start + 1
        with file_path.open("rb") as handle:
            handle.seek(start)
            body = handle.read(length)
        return (
            "206 Partial Content",
            [
                *self.response_headers(content_type=content_type, body_length=len(body)),
                ("Accept-Ranges", "bytes"),
                ("Content-Range", f"bytes {start}-{end}/{file_size}"),
            ],
            [body],
        )

    def serve_file(self, path: str, environ):
        """Serve project artifacts through a constrained `/files/...` route."""

        relative = self.decode_workspace_route_path(path.removeprefix("/files/"))
        if not relative:
            return self.text_response("404 Not Found", "Not found\n")
        file_path = self.resolve_under_root(relative)
        if not file_path.is_file() or not self.is_downloadable_path(file_path):
            return self.text_response("404 Not Found", "Not found\n")
        self.refresh_review_bundle_if_needed(file_path)

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        range_header = str(environ.get("HTTP_RANGE") or "")
        if range_header:
            ranged = self.range_response_for_file(file_path, content_type, range_header)
            if ranged is not None:
                return ranged

        body = file_path.read_bytes()
        headers = [
            *self.response_headers(content_type=content_type, body_length=len(body)),
            ("Accept-Ranges", "bytes"),
        ]
        return "200 OK", headers, [body]

    def refresh_review_bundle_if_needed(self, file_path: Path) -> None:
        """Regenerate older review pages on demand so playback and labeling stay current."""

        if not file_path.name.endswith("_review.html"):
            return
        srt_path = file_path.with_name(file_path.name.removesuffix("_review.html") + ".srt")
        if not srt_path.is_file():
            return
        report_path = file_path.with_name(file_path.name.removesuffix("_review.html") + "_review_flags.tsv")
        try:
            write_review_bundle(
                srt_path=srt_path,
                output_html=file_path,
                report_tsv=report_path,
                audio_dir=self.audio_dir,
                training_label_records=self.load_training_label_records(),
                model_comparisons=self.review_model_comparisons_for_srt(srt_path),
                quiet=True,
            )
        except Exception:
            return

    def artifact_kind(self, path: Path, content_type: str = "") -> str:
        """Classify one artifact for browser previews."""

        suffix = path.suffix.lower()
        normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
        if normalized_type.startswith("audio/") or suffix in AUDIO_EXTENSIONS:
            return "audio"
        if normalized_type == "text/html" or suffix in {".html", ".htm"}:
            return "html"
        if (
            normalized_type.startswith("text/")
            or normalized_type in {"application/json", "application/x-yaml"}
            or suffix in TEXT_PREVIEW_SUFFIXES
        ):
            return "text"
        return "binary"

    def read_artifact_preview_text(self, path: Path) -> tuple[str, bool]:
        """Read a short artifact preview without loading large files fully into memory."""

        suffix = path.suffix.lower()
        prefer_tail = suffix in TAIL_PREVIEW_SUFFIXES
        limit = ARTIFACT_PREVIEW_BYTE_LIMIT
        size = 0
        with path.open("rb") as handle:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            if prefer_tail and size > limit:
                handle.seek(max(0, size - limit))
                chunk = handle.read(limit)
            else:
                chunk = handle.read(limit)
        text = chunk.decode("utf-8", errors="replace")
        truncated = size > len(chunk)
        lines = text.splitlines()
        if prefer_tail and truncated and len(lines) > 1:
            lines = lines[1:]
        if len(lines) > ARTIFACT_PREVIEW_LINE_LIMIT:
            lines = lines[-ARTIFACT_PREVIEW_LINE_LIMIT:] if prefer_tail else lines[:ARTIFACT_PREVIEW_LINE_LIMIT]
            truncated = True
        preview = "\n".join(lines).strip()
        if not preview and text.strip():
            preview = text.strip()
        return preview, truncated

    def artifact_preview_payload(self, path: Path, script_name: str) -> dict[str, object]:
        """Build a browser preview payload for one allowed workspace artifact."""

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        kind = self.artifact_kind(path, content_type)
        payload: dict[str, object] = {
            "name": path.name,
            "path": self.describe_path(path),
            "href": self.file_link(path, script_name),
            "contentType": content_type,
            "kind": kind,
            "sizeBytes": path.stat().st_size if path.is_file() else 0,
        }
        if kind == "text":
            preview_text, truncated = self.read_artifact_preview_text(path)
            payload["previewText"] = preview_text
            payload["truncated"] = truncated
        elif kind == "html":
            payload["message"] = "This generated HTML page opens through the site and includes navigation back to the dashboard."
        elif kind == "audio":
            payload["message"] = "This audio file can be played directly in the browser."
        else:
            payload["message"] = "Preview is not available for this artifact type. Use the underlying file from the workspace when a download is needed."
        return payload

    def command_result_message(self, command: list[str], *, cwd: Path | None = None) -> tuple[str, str]:
        """Execute a project command and compress its output for UI feedback."""

        completed = subprocess.run(
            command,
            cwd=str(cwd or self.root),
            check=False,
            capture_output=True,
            text=True,
        )
        chunks = [chunk.strip() for chunk in [completed.stdout, completed.stderr] if chunk and chunk.strip()]
        message = "\n".join(chunks).strip() or "Command completed."
        if len(message) > 800:
            message = message[:797] + "..."
        status = "success" if completed.returncode == 0 else "error"
        return status, message

    def page_response(self, environ, current_path: str):
        """Render one of the main GET pages."""

        message, message_status = self.get_query_message(environ)
        return self.html_response(
            "200 OK",
            self.render_page(
                message=message,
                message_status=message_status,
                script_name=self.script_name(environ),
                current_path=current_path,
                diarization_model_key=self.requested_diarization_model_key(environ),
            ),
        )

    def handle_tracking_status(self, environ):
        """Return the latest live tracker payload for browser polling."""

        return self.json_response("200 OK", self.live_tracking_payload(self.script_name(environ)))

    def handle_page_state(self, environ):
        """Return full page state JSON for in-place frontend refreshes."""

        query = parse_qs(environ.get("QUERY_STRING", ""))
        requested_path = query.get("path", ["/"])[0]
        current_path = PAGE_ALIASES.get(requested_path, requested_path)
        if current_path not in PAGE_PATHS:
            current_path = "/"
        context = self.page_context(
            current_path,
            diarization_model_key=self.requested_diarization_model_key(environ),
        )
        payload = self.frontend_state(
            context=context,
            current_path=current_path,
            script_name=self.script_name(environ),
            message="",
            message_status="info",
        )
        return self.json_response("200 OK", payload)

    def handle_artifact_preview(self, environ):
        """Return a constrained preview payload for one workspace artifact."""

        query = parse_qs(environ.get("QUERY_STRING", ""))
        requested_path = (query.get("path") or [""])[0].strip()
        if not requested_path:
            return self.text_response("404 Not Found", "Not found\n")
        try:
            artifact_path = self.resolve_under_root(requested_path)
        except ValueError:
            return self.text_response("404 Not Found", "Not found\n")
        if not artifact_path.is_file() or not self.is_downloadable_path(artifact_path):
            return self.text_response("404 Not Found", "Not found\n")
        return self.json_response(
            "200 OK",
            self.artifact_preview_payload(artifact_path, self.script_name(environ)),
        )

    def requested_diarization_model_key(self, environ) -> str:
        """Read an optional model-key override from query parameters."""

        query = parse_qs(environ.get("QUERY_STRING", ""))
        return (query.get("diarization_model_key") or [""])[0].strip()

    def diarization_page_location(self, model_key: str = "") -> str:
        """Build a diarization page URL that preserves the selected model profile."""

        selected_model_key = (model_key or "").strip()
        if not selected_model_key:
            return "/diarization"
        return f"/diarization?{urlencode({'diarization_model_key': selected_model_key})}"

    def invalidate_dashboard_cache(self) -> None:
        """Clear short-lived cached listings after a mutation."""

        with self._dashboard_cache_lock:
            self._dashboard_cache.clear()

    def cached_value(self, key: str, *, ttl_seconds: float, builder):
        """Cache expensive dashboard scans for a short interval.

        The dashboard is read-heavy and the artifact tree can grow over time, so a
        short TTL removes unnecessary repeated scans without making the UI stale for
        very long after uploads or submissions.
        """

        now = time.monotonic()
        with self._dashboard_cache_lock:
            cached = self._dashboard_cache.get(key)
        if cached is not None:
            expires_at, value = cached
            if now < expires_at:
                return value
        value = builder()
        with self._dashboard_cache_lock:
            self._dashboard_cache[key] = (now + ttl_seconds, value)
        return value

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
                    version_slug = str(run.get("version_slug") or slugify(version_name))
                    version_options_added = append_option(
                        key_suffix=version_slug,
                        display_name=version_name,
                        description=f"Reusable fine-tuning artifact from version {version_name}.",
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

    def active_tracking_summary(self) -> dict[str, object]:
        """Summarize runs that should keep the dashboard refreshing."""

        diarization_runs = sorted(self.active_diarization_run_directories(), key=lambda path: str(path))
        youtube_runs = sorted(self.active_run_directories(self.youtube_runs_root), key=lambda path: str(path))
        fine_tuning_runs = self.active_fine_tuning_runs()
        should_refresh = bool(diarization_runs or youtube_runs or fine_tuning_runs)
        return {
            "refreshIntervalMs": LIVE_TRACKING_INTERVAL_MS,
            "idleRefreshIntervalMs": IDLE_TRACKING_INTERVAL_MS,
            "activeDiarizationRuns": len(diarization_runs),
            "activeYoutubeRuns": len(youtube_runs),
            "activeFineTuningRuns": len(fine_tuning_runs),
            "activeRunCount": len(diarization_runs) + len(youtube_runs) + len(fine_tuning_runs),
            "audioInputCount": self.audio_input_count(),
            "queuedLinks": self.queue_size(),
            "fineTuningProjects": self.project_count(),
            "failedLinksPresent": self.failed_links_path.is_file(),
            "shouldRefresh": should_refresh,
            "diarizationRunNames": [path.name for path in diarization_runs[:5]],
            "youtubeRunNames": [path.name for path in youtube_runs[:5]],
            "fineTuningRuns": fine_tuning_runs[:5],
        }

    def tracking_digest_value(self, value: object) -> object:
        """Normalize live tracker data into a deterministic JSON-friendly digest."""

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self.tracking_digest_value(inner) for key, inner in sorted(value.items())}
        if isinstance(value, list):
            return [self.tracking_digest_value(item) for item in value]
        if isinstance(value, tuple):
            return [self.tracking_digest_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def tracking_fingerprint(
        self,
        *,
        tracking: dict[str, object],
        diarization_run: dict[str, object] | None,
        youtube_run: dict[str, object] | None,
    ) -> str:
        """Hash the live tracker state so the frontend can refresh on real changes."""

        digest_payload = self.tracking_digest_value(
            {
                "tracking": tracking,
                "diarizationRun": diarization_run,
                "youtubeRun": youtube_run,
            }
        )
        encoded = json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def live_tracking_details(self) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object] | None, str]:
        """Load live tracker data for both long-running dashboard workflows."""

        tracking = self.active_tracking_summary()
        diarization_run = self.diarization_run_details(latest_diarization_directory(self.diarization_runs_root))
        youtube_run = self.youtube_run_details(latest_directory(self.youtube_runs_root))
        fingerprint = self.tracking_fingerprint(
            tracking=tracking,
            diarization_run=diarization_run,
            youtube_run=youtube_run,
        )
        tracking = dict(tracking)
        tracking["fingerprint"] = fingerprint
        return tracking, diarization_run, youtube_run, fingerprint

    def live_tracking_payload(self, script_name: str) -> dict[str, object]:
        """Serialize live tracker state for the browser polling endpoint."""

        tracking, diarization_run, youtube_run, fingerprint = self.live_tracking_details()
        return {
            "fingerprint": fingerprint,
            "tracking": tracking,
            "diarization": {
                "latestRun": self.frontend_diarization_run(diarization_run, script_name),
            },
            "youtube": {
                "latestRun": self.frontend_youtube_run(youtube_run, script_name),
            },
        }

    def parse_float(self, raw_value: str | None, default: float, field_name: str) -> float:
        """Parse a numeric float field with a clear user-facing error."""

        try:
            return float(raw_value) if raw_value not in {None, ""} else default
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {field_name}: {raw_value}") from exc

    def parse_int(self, raw_value: str | None, default: int, field_name: str) -> int:
        """Parse an integer field with a clear user-facing error."""

        try:
            return int(raw_value) if raw_value not in {None, ""} else default
        except ValueError as exc:
            raise ValueError(f"Invalid integer value for {field_name}: {raw_value}") from exc

    def safe_folder_name(self, value: str) -> str:
        """Normalize one audio library folder name without allowing nesting."""

        raw_name = str(value or "").strip()
        cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in raw_name)
        cleaned = cleaned.strip("._-")
        return cleaned or "set"

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

    def safe_output_component(self, value: str) -> str:
        """Render one path component for generated artifact basenames."""

        safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
        safe = safe.strip("._")
        return safe or "audio"

    def diarization_output_base(self, audio_value: str) -> str:
        """Match run_diarization.py's artifact basename for nested library sets."""

        cleaned = self.clean_audio_selection_value(audio_value)
        if not cleaned:
            cleaned = Path(audio_value or "audio").name
        stem_path = Path(cleaned).with_suffix("")
        return "__".join(self.safe_output_component(part) for part in stem_path.parts)

    def safe_log_component(self, value: str) -> str:
        """Match workflow_cli.py's per-audio log filename component."""

        safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value or ""))
        safe = safe.strip("._")
        return safe or "run"

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

    def resolve_ffmpeg_location(self) -> str | None:
        """Find ffmpeg for upload transcoding, including imageio-ffmpeg fallback."""

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            return ffmpeg_bin
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

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
            return path + (f"?{parsed.query}" if parsed.query else "")
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
            status = self.training_label_status(records.get(path.name))
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
        transcript_text = (form.getfirst("label_transcript_text") or "").strip()
        issue_questions = (form.getfirst("label_issue_questions") or "").strip()
        label_source = (form.getfirst("label_source") or "").strip()
        label_review_path = (form.getfirst("label_review_path") or "").strip()
        action = (form.getfirst("label_action") or "draft").strip().lower()

        base_record = {
            "backend": backend,
            "target_backends": target_backends,
            "project_name": project_name,
            "label_segments": raw_segments,
            "transcript_text": transcript_text,
            "issue_questions": issue_questions,
            "system_questions": [],
        }
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
        return self.redirect(
            environ,
            return_location,
            message=self.notification_message(
                f"Marked '{audio_name}' complete and added it to {', '.join(completed_record['training_projects'])}.",
                "The completed label is now part of the fine-tuning samples used by Prepare Artifacts.",
            ),
            status="success",
        )

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
                submission = submit_sbatch_job(
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
        launch_background_command(
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
                submission = submit_sbatch_job(
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
        run = launch_training(
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

    def safe_filename(self, value: str) -> str:
        """Normalize an uploaded filename so it is safe to write under the workspace."""

        name = Path(value or "").name
        cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
        cleaned = cleaned.strip("._")
        return cleaned or "upload.bin"

    def is_supported_uploaded_audio_name(self, filename: str) -> bool:
        """Accept only the shared audio formats that the local workflow expects."""

        suffix = Path(filename or "").suffix.lower()
        return suffix in UPLOAD_AUDIO_SUFFIXES

    def notification_message(self, summary: str, *details: str) -> str:
        """Build one multi-line notification payload for the shared site banner."""

        parts = [summary.strip(), *[detail.strip() for detail in details if detail.strip()]]
        return "\n".join(part for part in parts if part)

    def dashboard_secret(self, key: str) -> str:
        """Read one local dashboard secret without exposing it in frontend state."""

        if not self.dashboard_secrets_path.is_file():
            return ""
        try:
            lines = self.dashboard_secrets_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ""
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == key:
                return value.strip().strip("\"'")
        return ""

    def youtube_issue_category(self, summary: str, explicit_category: str = "") -> str:
        """Normalize issue categories so older reports still render cleanly."""

        normalized = (explicit_category or "").strip().lower().replace(" ", "_")
        if normalized in {"retry", "no_data"}:
            return normalized
        lowered = (summary or "").strip().lower()
        if any(marker in lowered for marker in YOUTUBE_NO_DATA_MARKERS):
            return "no_data"
        return "retry"

    def file_link(self, path: Path, script_name: str) -> str:
        """Map a workspace file to the constrained download route."""

        relative = path.resolve().relative_to(self.root)
        href = self.with_prefix("/files/" + quote(str(relative).replace("\\", "/")), script_name)
        return href

    def describe_path(self, path: Path) -> str:
        """Prefer workspace-relative paths in UI messages when possible."""

        resolved = path.resolve()
        try:
            return str(resolved.relative_to(self.root))
        except ValueError:
            return str(resolved)

    def artifact_preview_link(self, path: Path, script_name: str) -> str:
        """Map one workspace file to the constrained preview API route."""

        return self.with_prefix(
            "/api/artifact-preview?" + urlencode({"path": self.describe_path(path)}),
            script_name,
        )

    def newest_files(
        self,
        *,
        search_roots: list[Path],
        suffixes: set[str],
        limit: int,
        exclude_dir_names: set[str] | None = None,
    ) -> list[Path]:
        """Return the newest matching files without sorting the entire tree.

        This reduces dashboard overhead once the workspace accumulates many logs
        and outputs across repeated diarization and fine-tuning runs.
        """

        newest: list[tuple[float, str, Path]] = []
        blocked_dirs = exclude_dir_names or set()
        for root in search_roots:
            if not root.is_dir():
                continue
            for current_root, dirnames, filenames in os.walk(root):
                if blocked_dirs:
                    dirnames[:] = [name for name in dirnames if name not in blocked_dirs]
                current_dir = Path(current_root)
                for filename in filenames:
                    candidate = current_dir / filename
                    if candidate.suffix.lower() not in suffixes:
                        continue
                    try:
                        modified = candidate.stat().st_mtime
                    except OSError:
                        continue
                    row = (modified, str(candidate), candidate)
                    if len(newest) < limit:
                        heapq.heappush(newest, row)
                    else:
                        heapq.heappushpop(newest, row)
        newest.sort(reverse=True)
        return [path for _, _, path in newest]

    def recent_output_files(self, limit: int = 18) -> list[Path]:
        """Return the newest artifacts likely to matter during review."""

        return self.cached_value(
            f"recent_output_files:{limit}",
            ttl_seconds=3.0,
            builder=lambda: self.newest_files(
                search_roots=[
                    self.outputs_root,
                    self.root / "job_outputs",
                    self.root / "job_logs",
                    self.root / "fine_tuning",
                ],
                suffixes={".txt", ".tsv", ".srt", ".html", ".json", ".log", ".sh"},
                limit=limit,
                exclude_dir_names=ARTIFACT_SCAN_EXCLUDE_DIRS,
            ),
        )

    def recent_srt_files(self) -> list[Path]:
        """Return the most recent subtitle outputs for review generation."""

        return self.cached_value(
            "recent_srt_files",
            ttl_seconds=3.0,
            builder=lambda: self.newest_files(
                search_roots=[self.outputs_root, self.root / "job_outputs"],
                suffixes={".srt"},
                limit=20,
                exclude_dir_names={"__pycache__"},
            ),
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

    def model_preferences(self) -> dict[str, object]:
        """Load the persisted workflow defaults used by multiple dashboard pages."""

        return self.cached_value(
            "model_preferences",
            ttl_seconds=3.0,
            builder=lambda: load_preferences(root=self.root),
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

    def read_tsv_rows(self, path: Path) -> list[dict[str, str]]:
        """Read a tab-separated file into a list of plain string dictionaries."""

        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows: list[dict[str, str]] = []
            for row in reader:
                rows.append(
                    {
                        str(key): "" if value is None else str(value)
                        for key, value in row.items()
                        if key is not None
                    }
                )
        return rows

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

    def tail_text(self, path: Path, *, line_count: int = 12) -> str:
        """Return the last few log lines for a dashboard preview."""

        if not path.is_file():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-line_count:]).strip()

    def error_summary(self, path: Path) -> str:
        """Return the clearest short error from a per-file log."""

        if not path.is_file():
            return ""
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        for line in reversed(lines):
            if line.startswith(("RuntimeError:", "ValueError:", "FileNotFoundError:", "ModuleNotFoundError:")):
                return line
        return lines[-1] if lines else ""

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
        slurm_queue = slurm_queue_snapshot(slurm_job_id) if slurm_job_id and status in {"submitted", "running"} else {}

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

    def write_tsv_rows(
        self,
        path: Path,
        fieldnames: tuple[str, ...] | list[str],
        rows: list[dict[str, str]],
    ) -> None:
        """Atomically rewrite a TSV file with the given header and rows."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

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

    def read_line_file(self, path: Path) -> list[str]:
        """Read one plain-text list file while skipping blank and commented rows."""

        if not path.is_file():
            return []
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

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

    def _review_html_is_missing_media(self, review_path: Path) -> bool:
        """Decide whether the existing review HTML's audio reference is broken.

        Two failure modes regenerate the bundle:

        1. The HTML was generated when no media could be located, so it
           contains the explicit "No matching media file was found" marker.
        2. The HTML embeds a media ``src=...`` whose resolved file no longer
           exists on disk. This happens after a WAV cleanup + redownload
           cycle: the old src points at ``audio_in/002_clip.wav`` but the
           replacement now lives at ``audio_in/youtube_links/001_clip.wav``,
           so the player surface is technically present but loads a 404.
        """

        try:
            text = review_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if "No matching media file was found" in text:
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
        slurm_queue = slurm_queue_snapshot(slurm_job_id) if slurm_job_id and status in {"submitted", "running"} else {}

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

    def run_directory_name(self, prefix: str, *, backend: str = "", count: int = 0) -> str:
        """Build a readable run-directory name for outputs created from the site."""

        stamp = time.strftime("%Y%m%dT%H%M%S")
        parts = [stamp, prefix]
        if backend:
            parts.append(backend)
        if count > 0:
            parts.append(f"{count:02d}-items")
        return "_".join(parts)

    def ordered_unique(self, values: list[str]) -> list[str]:
        """Deduplicate a list while preserving its original order."""

        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

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
        if effective_path in MODEL_SELECTION_PAGES:
            context["preferences"] = self.model_preferences()
        if effective_path in {"/uploads", "/training-labels", "/fine-tuning"}:
            context["diarization_history"] = self.diarization_history_rows(limit=100)
            context["diarization_latest_run"] = self.diarization_run_details(context["latest_site_diarization"])
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
                recent_runs.append(
                    {
                        "status": str(run.get("status", "unknown")),
                        "versionName": str(run.get("version_name") or Path(str(run.get("run_dir", ""))).name),
                        "versionNumber": run.get("version_number", ""),
                        "runName": Path(str(run.get("run_dir", ""))).name,
                        "startedAt": str(run.get("started_at_utc") or ""),
                    }
                )
            serialized.append(
                {
                    "backend": str(project.get("backend", "")),
                    "slug": str(project.get("slug", "")),
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
                "fineTuneUpload": self.frontend_route("/fine-tuning/upload-sample", script_name),
                "fineTunePrepare": self.frontend_route("/fine-tuning/prepare", script_name),
                "fineTuneLaunch": self.frontend_route("/fine-tuning/launch", script_name),
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
                },
                "diarization": {
                    "audioRows": context.get("diarization_audio_rows", []),
                    "librarySummary": context.get("diarization_library_summary", {}),
                    "modelOptions": self.frontend_diarization_model_options(list(context.get("diarization_model_options", []))),
                    "selectedModelKey": context.get("selected_diarization_model_key", ""),
                    "history": self.frontend_diarization_history_rows(list(context.get("diarization_history", [])), script_name),
                    "latestRun": self.frontend_diarization_run(context.get("diarization_latest_run"), script_name),
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


app = WorkflowWebApp()


def waitress_available() -> bool:
    """Report whether the optional Waitress runtime is installed."""

    return importlib.util.find_spec("waitress") is not None


def resolve_server_mode(server_mode: str) -> str:
    """Resolve the requested runtime, preferring Waitress when available."""

    normalized = server_mode.lower()
    if normalized == "auto":
        return "waitress" if waitress_available() else "threaded"
    if normalized == "waitress" and not waitress_available():
        raise ValueError("Waitress is not installed. Use --server threaded or install waitress.")
    if normalized in {"waitress", "threaded", "wsgiref"}:
        return normalized
    raise ValueError(f"Unsupported server mode: {server_mode}")


def built_in_make_server(host: str, port: int, application, *, threaded: bool):
    """Create one of the built-in WSGI server variants."""

    from wsgiref.simple_server import WSGIServer, make_server

    server_class = type(
        "ThreadedBuiltinWSGIServer" if threaded else "BuiltinWSGIServer",
        (ThreadedWSGIServer, WSGIServer) if threaded else (WSGIServer,),
        {},
    )
    return make_server(host, port, application, server_class=server_class)


def waitress_make_server(host: str, port: int, application, *, threads: int):
    """Create a Waitress server when the optional dependency is available."""

    from waitress.server import create_server

    return create_server(application, host=host, port=port, threads=threads)


def server_port(server, requested_port: int) -> int:
    """Recover the bound port from the server object when possible."""

    for attribute in ("server_port", "effective_port", "port"):
        value = getattr(server, attribute, None)
        if isinstance(value, int) and value > 0:
            return value

    socket_obj = getattr(server, "socket", None)
    getsockname = getattr(socket_obj, "getsockname", None)
    if callable(getsockname):
        try:
            return int(getsockname()[1])
        except (OSError, TypeError, ValueError):
            pass

    return requested_port


def write_url_file(destination: str | None, url: str) -> None:
    """Persist the resolved local dashboard URL when requested."""

    if not destination:
        return
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{url}\n", encoding="utf-8")


def bind_server(host: str, port: int, application, *, max_port_tries: int = 25, make_server_fn=None):
    """Bind the workflow web server, retrying nearby ports when one is already busy."""

    make_server_fn = make_server_fn or (
        lambda bind_host, bind_port, bind_app: built_in_make_server(
            bind_host,
            bind_port,
            bind_app,
            threaded=False,
        )
    )
    attempts = 1 if port == 0 else max(max_port_tries, 1)
    last_exc: OSError | None = None
    for offset in range(attempts):
        candidate_port = port if port == 0 else port + offset
        try:
            server = make_server_fn(host, candidate_port, application)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or port == 0:
                raise
            last_exc = exc
            continue
        return server, server_port(server, candidate_port)

    upper_bound = port + attempts - 1
    raise OSError(
        errno.EADDRINUSE,
        f"No free port found between {port} and {upper_bound}.",
    ) from last_exc


def build_parser() -> argparse.ArgumentParser:
    """Expose a minimal CLI for serving the dashboard locally."""

    parser = argparse.ArgumentParser(description="Serve the ML Speech Diarization dashboard.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--server",
        choices=["auto", "threaded", "waitress", "wsgiref"],
        default=DEFAULT_SERVER_MODE,
        help="Runtime server to use. 'auto' prefers Waitress when installed.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_SERVER_THREADS,
        help="Worker thread count for Waitress. Retained for auto mode selection.",
    )
    parser.add_argument(
        "--url-file",
        default=None,
        help="Optional path that receives the resolved local URL after startup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Start the dashboard with the selected WSGI runtime."""

    args = build_parser().parse_args(argv)
    host = args.host
    requested_port = args.port
    if args.threads < 1:
        print("Error: --threads must be at least 1.", file=sys.stderr)
        return 1
    try:
        server_mode = resolve_server_mode(args.server)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    display_host = host
    if host in {"0.0.0.0", "::"}:
        display_host = "127.0.0.1"
    hostname = socket.gethostname()
    make_server_fn = None
    runtime_label = "built-in WSGI"
    if server_mode == "threaded":
        make_server_fn = lambda bind_host, bind_port, bind_app: built_in_make_server(
            bind_host,
            bind_port,
            bind_app,
            threaded=True,
        )
        runtime_label = "threaded built-in WSGI"
    elif server_mode == "waitress":
        make_server_fn = lambda bind_host, bind_port, bind_app: waitress_make_server(
            bind_host,
            bind_port,
            bind_app,
            threads=args.threads,
        )
        runtime_label = f"Waitress ({args.threads} worker threads)"
    elif server_mode == "wsgiref":
        runtime_label = "standard-library WSGI"

    try:
        server, actual_port = bind_server(host, requested_port, app, make_server_fn=make_server_fn)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        raise
    local_url = f"http://{display_host}:{actual_port}"
    try:
        write_url_file(args.url_file, local_url)
        if actual_port != requested_port:
            print(
                f"Port {requested_port} is already in use. "
                f"Serving workflow hub on port {actual_port} instead."
            )
        print(f"Server runtime: {runtime_label}")
        if host in {"0.0.0.0", "::"}:
            print(f"Serving workflow hub on all interfaces, port {actual_port}")
            print(f"Open locally: {local_url}")
            print(
                "Open remotely if your HPC/network setup allows it: "
                f"http://{hostname}:{actual_port}"
            )
        else:
            print(f"Serving workflow hub on {local_url}")

        if server_mode == "waitress":
            server.run()
        else:
            with server:
                server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping workflow hub.")
    finally:
        if server_mode == "waitress":
            close = getattr(server, "close", None)
            if callable(close):
                close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
