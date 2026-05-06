#!/usr/bin/env python3
"""WSGI application core: request dispatch, response helpers, and shared utilities across all mixins."""
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


class CoreMixin:
    """Central WSGI class: composes all mixins and owns the request dispatch table."""

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
            elif routed_method == "GET" and path == "/api/cluster-queue":
                status, headers, body = self.handle_cluster_queue(environ)
            elif routed_method == "GET" and path == "/api/runtime-estimate":
                status, headers, body = self.handle_runtime_estimate(environ)
            elif routed_method == "GET" and path == "/api/file-search":
                status, headers, body = self.handle_file_search(environ)
            elif routed_method == "GET" and path == "/api/fine-tuning/score-run":
                status, headers, body = self.handle_finetune_score_run(environ)
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
            elif routed_method == "POST" and path == "/fine-tuning/rename-project":
                status, headers, body = self.handle_finetune_rename_project(environ)
            elif routed_method == "POST" and path == "/fine-tuning/rename-run":
                status, headers, body = self.handle_finetune_rename_run(environ)
            elif routed_method == "POST" and path == "/fine-tuning/auto-train":
                status, headers, body = self.handle_finetune_auto_train(environ)
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

    def handle_cluster_queue(self, environ):
        """Return the cluster-wide squeue snapshot, optionally filtered.

        Query parameters:
          state     - comma-separated squeue state codes (default ``PD,R,CG``)
          partition - case-insensitive partition substring filter
          user      - case-insensitive username substring filter
        """

        from dashboard.cluster_queue import cluster_queue_snapshot

        query = parse_qs(environ.get("QUERY_STRING", ""))
        states = (query.get("state") or ["PD,R,CG"])[0].strip() or "PD,R,CG"
        partition_filter = (query.get("partition") or [""])[0].strip().lower()
        user_filter = (query.get("user") or [""])[0].strip().lower()

        # Cache for 5 seconds so a page that polls multiple times doesn't hammer
        # the controller. Filters are applied after the cached read so distinct
        # filter combinations all share one squeue invocation.
        snapshot = self.cached_value(
            f"cluster_queue::{states}",
            ttl_seconds=5.0,
            builder=lambda: cluster_queue_snapshot(states=states),
        )

        if partition_filter or user_filter:
            filtered = []
            for job in snapshot.get("jobs", []):
                if partition_filter and partition_filter not in job["partition"].lower():
                    continue
                if user_filter and user_filter not in job["user"].lower():
                    continue
                filtered.append(job)
            snapshot = {**snapshot, "jobs": filtered, "filtered": True}
        return self.json_response("200 OK", snapshot)

    def runtime_history_rates(self) -> dict:
        """Cached cluster-wide-cheap rolling rate per diarization backend.

        TTL is generous because runtime history only changes when a run
        finishes; we don't need to re-walk the runs folder on every poll.
        """

        from dashboard.runtime_history import gather_historical_rates

        return self.cached_value(
            "runtime_history_rates",
            ttl_seconds=30.0,
            builder=lambda: gather_historical_rates(
                self.diarization_runs_root,
                audio_dir=self.audio_dir,
                local_dashboard_dir=self.local_dashboard_dir,
            ),
        )

    def handle_runtime_estimate(self, environ):
        """Estimate wallclock for a candidate diarization run.

        Query parameters:
          backend         - "nemo" / "pyannote" / etc. (required)
          audio_files     - one or more relative audio paths under audio_in/

        Multiple ``audio_files`` values are supported (parse_qs returns a list).
        """

        from dashboard.runtime_history import estimate_runtime_for_files

        query = parse_qs(environ.get("QUERY_STRING", ""))
        backend = (query.get("backend") or [""])[0].strip() or "nemo"
        audio_relatives = [v for v in query.get("audio_files", []) if v.strip()]
        # Resolve to absolute paths under audio_in/
        audio_paths = []
        for rel in audio_relatives:
            try:
                resolved = (self.audio_dir / rel).resolve()
                # Guard against escapes outside audio_dir.
                resolved.relative_to(self.audio_dir.resolve())
                audio_paths.append(resolved)
            except (ValueError, OSError):
                continue

        rates = self.runtime_history_rates()
        estimate = estimate_runtime_for_files(
            audio_paths,
            backend=backend,
            rates=rates,
            local_dashboard_dir=self.local_dashboard_dir,
        )
        # Include the per-backend rate snapshot so the UI can show "based on
        # 5 runs / 24 files" without an extra round-trip.
        estimate["backend"] = backend
        estimate["history"] = {
            backend_name: {
                "available": bool(payload.get("available")),
                "ratio": payload.get("ratio"),
                "sample_count": payload.get("sample_count"),
                "runs_used": payload.get("runs_used"),
                "last_run": payload.get("last_run"),
                "message": payload.get("message"),
            }
            for backend_name, payload in rates.items()
        }
        return self.json_response("200 OK", estimate)

    def file_index_snapshot(self):
        """Return the cross-run file index. Cached for 30 s — outputs only grow
        when a run finishes, so we don't need to re-walk on every poll."""

        from dashboard.file_index import build_file_index

        return self.cached_value(
            "file_index",
            ttl_seconds=30.0,
            builder=lambda: build_file_index(
                audio_dir=self.audio_dir,
                outputs_root=self.outputs_root,
                fine_tuning_root=self.root / "fine_tuning",
            ),
        )

    def handle_file_search(self, environ):
        """Search the cross-run file index by stem.

        Query parameters:
          q     - search query (case-insensitive substring; prefix matches rank first)
          limit - max grouped results (default 50)
        """

        from dashboard.file_index import search_index

        query = parse_qs(environ.get("QUERY_STRING", ""))
        needle = (query.get("q") or [""])[0].strip()
        try:
            limit = int((query.get("limit") or ["50"])[0])
        except ValueError:
            limit = 50
        index = self.file_index_snapshot()
        if not needle:
            return self.json_response("200 OK", {"query": "", "results": [], "total_stems": len(index)})
        results = search_index(index, needle, limit=max(1, min(limit, 200)))
        # Wrap matches with download links for the frontend.
        script_name = self.script_name(environ)
        for group in results:
            for record in group.get("matches", []):
                record["href"] = self.with_prefix("/files/" + record["rel_path"], script_name)
        return self.json_response(
            "200 OK",
            {"query": needle, "results": results, "total_stems": len(index)},
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

    def cached_slurm_queue(self, slurm_job_id: str, status: str) -> dict[str, object]:
        """Return a cached squeue snapshot for active jobs; returns {} immediately for finished ones.

        We only query squeue while the job is live — once it's terminal the queue
        snapshot is no longer useful and the overhead of calling squeue is wasted.
        """

        if not slurm_job_id or status not in {"submitted", "running"}:
            return {}
        import workflow_dashboard as _wd  # lazy: honor monkey-patches in tests
        return self.cached_value(
            f"slurm_queue::{slurm_job_id}",
            ttl_seconds=2.0,
            builder=lambda: _wd.slurm_queue_snapshot(slurm_job_id),
        )

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

    def safe_output_component(self, value: str) -> str:
        """Render one path component for generated artifact basenames."""

        safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
        safe = safe.strip("._")
        return safe or "audio"

    def safe_log_component(self, value: str) -> str:
        """Match workflow_cli.py's per-audio log filename component."""

        safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value or ""))
        safe = safe.strip("._")
        return safe or "run"

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

    def read_line_file(self, path: Path) -> list[str]:
        """Read one plain-text list file while skipping blank and commented rows."""

        if not path.is_file():
            return []
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

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
