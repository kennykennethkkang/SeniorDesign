"""Module-level constants for the dashboard.

Imported by ``workflow_dashboard`` and the ``dashboard`` submodules. Keeping the
page-route sets, file-suffix allowlists, status enums, and server defaults in
one place avoids re-declaring them across every module that touches them.

These values are extracted verbatim from the original ``workflow_dashboard.py``
to preserve behavior. ``CODE_ROOT`` is rebased to point at the project root
(parent of this package) so legacy ``CODE_ROOT / "frontend" / ...`` references
continue to resolve to the same directory they did before the split.
"""
from __future__ import annotations

from pathlib import Path

from audio_numbering import AUDIO_EXTENSIONS

CODE_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_TEMPLATE = "html/dashboard/index.html"
NAV_PATHS = [
    "/",
    "/uploads",
    "/stitching",
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
AUDIO_INVENTORY_PAGES = {"/uploads", "/stitching", "/training-labels", "/diarization", "/fine-tuning"}
PROJECT_SUMMARY_PAGES = {"/", "/stitching", "/training-labels", "/fine-tuning"}
RECENT_OUTPUT_PAGES = {"/", "/stitching", "/diarization", "/youtube"}
RECENT_SRT_PAGES: set[str] = set()
YOUTUBE_QUEUE_PAGES = {"/youtube"}
MODEL_SELECTION_PAGES = {"/stitching", "/training-labels", "/diarization", "/fine-tuning"}
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
    "stitched",
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
