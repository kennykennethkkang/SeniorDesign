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

from dashboard.app.core import CoreMixin
from dashboard.app.audio_mixin import AudioMixin
from dashboard.app.youtube_mixin import YouTubeMixin
from dashboard.app.diarization_mixin import DiarizationMixin
from dashboard.app.training_mixin import TrainingLabelsMixin
from dashboard.app.stitching_mixin import StitchingMixin
from dashboard.app.finetuning_mixin import FineTuningMixin
from dashboard.app.rendering import RenderingMixin

class WorkflowWebApp(
    CoreMixin,
    AudioMixin,
    YouTubeMixin,
    DiarizationMixin,
    TrainingLabelsMixin,
    StitchingMixin,
    FineTuningMixin,
    RenderingMixin,
):
    """WSGI application that exposes the main project workflows through one dashboard."""



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


if __name__ == "__main__":
    from dashboard.cli import main
    raise SystemExit(main())
