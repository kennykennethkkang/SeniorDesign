#!/usr/bin/env python3
"""Top-level command-line entrypoint for the ML Speech Diarization project.

The dashboard does the heavy lifting now, so this CLI deliberately stays
narrow. It supports the few actions users still need from a terminal:

* ``status``     - print a quick report of audio inputs, queued YouTube URLs,
                   tool availability, and the latest dashboard-launched runs.
* ``review``     - regenerate an HTML review page (and the matching flag TSV)
                   for an existing diarization SRT.
* ``test``       - run the local unit-test suite via ``run_tests.sh``.
* ``serve-web``  - boot the dashboard in the foreground (useful for debugging).
* ``local-web``  - start/stop/inspect the daemonized dashboard process.

The legacy ``submit-single`` / ``submit-bulk`` / ``submit-youtube`` commands
were retired with the dashboard cleanup; their scripts and helpers live under
``archive/cleanup_2026-05-01/scheduler_pre_dashboard/`` for reference.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence
from urllib.error import URLError
from urllib.request import urlopen

from review_bundle import write_review_bundle
from workflow_preferences import (
    DEFAULT_PYANNOTE_PIPELINE_MODEL,
    DEFAULT_PYANNOTE_SEGMENTATION_MODEL,
    DIARIZATION_BACKENDS,
)

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_ROOT / "audio_in"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DIARIZATION_RUNS_ROOT = OUTPUTS_ROOT / "diarization_runs"
YOUTUBE_RUNS_ROOT = OUTPUTS_ROOT / "youtube_conversion_runs"
YOUTUBE_HISTORY_ROOT = OUTPUTS_ROOT / "youtube_conversion_history"
YOUTUBE_HISTORY_INDEX = YOUTUBE_HISTORY_ROOT / "url_audio_index.tsv"
SCHEDULER_DIR = PROJECT_ROOT / "scheduler"
TEST_SCRIPT = PROJECT_ROOT / "run_tests.sh"
WEB_APP_SCRIPT = PROJECT_ROOT / "workflow_dashboard.py"
YOUTUBE_BATCH_SCRIPT = PROJECT_ROOT / "youtube_audio_batch.py"
DIARIZATION_SCRIPT = PROJECT_ROOT / "all_diarization_programs" / "run_diarization.py"
YOUTUBE_LINKS = PROJECT_ROOT / "youtube_links.txt"
FAILED_LINKS = PROJECT_ROOT / "youtube_links_err" / "failed_links_latest.tsv"
LOCAL_WEB_DIR = PROJECT_ROOT / ".local_dashboard"
LOCAL_WEB_PID = LOCAL_WEB_DIR / "dashboard.pid"
LOCAL_WEB_URL = LOCAL_WEB_DIR / "dashboard.url"
LOCAL_WEB_LOG = LOCAL_WEB_DIR / "dashboard.log"
LOCAL_WEB_SECRETS = LOCAL_WEB_DIR / "secrets.env"


def ensure_local_web_dir() -> Path:
    """Create the runtime directory used by the local dashboard launcher."""

    LOCAL_WEB_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_WEB_DIR


def read_pid_file(path: Path) -> int | None:
    """Read a positive integer PID from disk when one is available."""

    if not path.is_file():
        return None
    raw_value = path.read_text(encoding="utf-8").strip()
    if not raw_value:
        return None
    try:
        pid = int(raw_value)
    except ValueError:
        return None
    return pid if pid > 0 else None


def process_running(pid: int) -> bool:
    """Check whether a process still exists on the local machine."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def local_dashboard_process_pids(
    *,
    project_root: Path = PROJECT_ROOT,
    web_app_script: Path = WEB_APP_SCRIPT,
    run_command=subprocess.run,
) -> list[int]:
    """Return running dashboard PIDs for this project, including orphaned launchers."""

    try:
        completed = run_command(
            ["ps", "-eo", "pid=,args="],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return []
    if getattr(completed, "returncode", 1) != 0:
        return []

    project_text = str(project_root)
    script_text = str(web_app_script)
    pids: list[int] = []
    for line in str(getattr(completed, "stdout", "") or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, args = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if script_text in args and project_text in args:
            pids.append(pid)
    return pids


def read_local_web_state(
    *,
    pid_path: Path = LOCAL_WEB_PID,
    url_path: Path = LOCAL_WEB_URL,
    log_path: Path = LOCAL_WEB_LOG,
    is_running_fn=process_running,
) -> dict[str, object]:
    """Describe the current background dashboard state."""

    pid = read_pid_file(pid_path)
    running = bool(pid and is_running_fn(pid))
    url = url_path.read_text(encoding="utf-8").strip() if url_path.is_file() else ""
    return {
        "pid": pid,
        "running": running,
        "url": url,
        "log_path": log_path,
        "stale_pid": bool(pid and not running),
    }


def clear_local_web_runtime_files(*paths: Path) -> None:
    """Remove launcher bookkeeping files while leaving the log behind."""

    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def build_local_web_command(
    *,
    port: int,
    server: str,
    threads: int,
    url_file: Path,
    python_bin: str = sys.executable,
) -> list[str]:
    """Build the detached localhost dashboard command."""

    return [
        python_bin,
        "-u",
        str(WEB_APP_SCRIPT),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--server",
        server,
        "--threads",
        str(threads),
        "--url-file",
        str(url_file),
    ]


def tail_file(path: Path, *, line_count: int = 12) -> str:
    """Return the most recent lines from a log file."""

    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def local_web_candidate_urls(*, preferred_port: int, max_port_tries: int = 25) -> list[str]:
    """List the localhost URLs the dashboard launcher may try during startup."""

    attempts = 1 if preferred_port == 0 else max(max_port_tries, 1)
    if preferred_port == 0:
        return ["http://127.0.0.1"]
    return [f"http://127.0.0.1:{preferred_port + offset}" for offset in range(attempts)]


def local_web_healthcheck(url: str, *, timeout_seconds: float = 0.5) -> bool:
    """Return True when the dashboard health endpoint answers cleanly."""

    health_url = f"{url.rstrip('/')}/health"
    try:
        with urlopen(health_url, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="replace").strip()
            return response.status == 200 and payload == "ok"
    except (OSError, URLError, ValueError):
        return False


def wait_for_local_web_start(
    process: subprocess.Popen,
    *,
    preferred_port: int,
    url_path: Path = LOCAL_WEB_URL,
    deadline_seconds: float = 15.0,
    max_port_tries: int = 25,
    healthcheck_fn=local_web_healthcheck,
    monotonic_fn=time.monotonic,
    sleep_fn=time.sleep,
) -> str | None:
    """Wait for a healthy dashboard and persist the resolved URL."""

    deadline = monotonic_fn() + deadline_seconds
    candidate_urls = local_web_candidate_urls(
        preferred_port=preferred_port,
        max_port_tries=max_port_tries,
    )
    while monotonic_fn() < deadline:
        recorded_url = url_path.read_text(encoding="utf-8").strip() if url_path.is_file() else ""
        probe_urls = [recorded_url] if recorded_url else []
        probe_urls.extend(url for url in candidate_urls if url and url != recorded_url)
        for candidate_url in probe_urls:
            if healthcheck_fn(candidate_url):
                url_path.write_text(f"{candidate_url}\n", encoding="utf-8")
                return candidate_url
        if process.poll() is not None:
            break
        sleep_fn(0.1)
    return None


def local_secret_is_set(key: str) -> bool:
    """Check whether a local dashboard secret exists without printing it."""

    if not LOCAL_WEB_SECRETS.is_file():
        return False
    try:
        lines = LOCAL_WEB_SECRETS.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    prefix = f"{key}="
    return any(line.strip().startswith(prefix) and line.split("=", 1)[1].strip() for line in lines)


def stop_background_process(pid: int, *, timeout_seconds: float = 5.0) -> bool:
    """Terminate a detached process, escalating to SIGKILL if needed."""

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process_running(pid):
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not process_running(pid):
            return True
        time.sleep(0.1)
    return not process_running(pid)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI and all supported subcommands."""

    parser = argparse.ArgumentParser(
        description="Unified entry point for the ML Speech Diarization workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser(
        "status", help="Show the current workspace, queue, and recent output state."
    )
    status_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many recent audio files to show.",
    )
    status_parser.set_defaults(func=cmd_status)

    review_parser = subparsers.add_parser(
        "review", help="Create an HTML review page and TSV flags from an SRT."
    )
    review_parser.add_argument("--srt", required=True, help="Path to the diarization SRT.")
    review_parser.add_argument("--media", default=None, help="Optional media path.")
    review_parser.add_argument(
        "--audio-dir",
        default=str(AUDIO_DIR),
        help="Directory to search when --media is omitted.",
    )
    review_parser.add_argument("--output-html", default=None, help="Optional HTML path.")
    review_parser.add_argument("--report-tsv", default=None, help="Optional TSV path.")
    review_parser.set_defaults(func=cmd_review)

    test_parser = subparsers.add_parser(
        "test", help="Run the local unittest suite."
    )
    test_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the command without running it.",
    )
    test_parser.set_defaults(func=cmd_test)

    web_parser = subparsers.add_parser(
        "serve-web", help="Run the local workflow dashboard."
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind host. Defaults to 127.0.0.1 (local only), which is what an "
            "SSH tunnel connects to. Pass 0.0.0.0 to expose it on the network, "
            "but only on a trusted network: the dashboard has no authentication."
        ),
    )
    web_parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    web_parser.add_argument(
        "--server",
        choices=["auto", "threaded", "waitress", "wsgiref"],
        default="auto",
        help="Runtime server to use. 'auto' prefers Waitress when installed.",
    )
    web_parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Worker thread count for Waitress-backed runs and auto mode.",
    )
    web_parser.set_defaults(func=cmd_serve_web)

    convert_parser = subparsers.add_parser(
        "convert-youtube-selection",
        help="Convert one or more queued YouTube URLs into numbered audio files.",
    )
    convert_parser.add_argument("--urls-file", required=True, help="Text file containing the URLs to process.")
    convert_parser.add_argument(
        "--queue-file",
        default=str(YOUTUBE_LINKS),
        help="Actual queue file to update when no-data URLs should be removed.",
    )
    convert_parser.add_argument(
        "--index-file",
        default=str(YOUTUBE_HISTORY_INDEX),
        help="Persistent conversion-history index (TSV).",
    )
    convert_parser.add_argument(
        "--run-dir",
        required=True,
        help="Per-run directory for logs and resolution reports.",
    )
    convert_parser.add_argument(
        "--output-dir",
        default=str(AUDIO_DIR),
        help="Folder where converted WAV files should be stored.",
    )
    convert_parser.add_argument(
        "--force-redownload",
        action="store_true",
        default=False,
        help="Download URLs again even if they already appear in the history index.",
    )
    convert_parser.set_defaults(func=cmd_convert_youtube_selection)

    diarize_selection_parser = subparsers.add_parser(
        "run-diarization-selection",
        help="Run diarization locally for a selected set of audio files.",
    )
    diarize_selection_parser.add_argument(
        "--audio-list-file",
        required=True,
        help="Text file containing one selected audio filename per line.",
    )
    diarize_selection_parser.add_argument(
        "--backend",
        required=True,
        choices=list(DIARIZATION_BACKENDS),
        help="Diarization backend to use for every selected file.",
    )
    diarize_selection_parser.add_argument(
        "--output-dir",
        required=True,
        help="Run directory where transcripts, subtitles, and review files will be written.",
    )
    diarize_selection_parser.add_argument(
        "--log-dir",
        required=True,
        help="Directory where per-file stdout/stderr logs will be written.",
    )
    diarize_selection_parser.add_argument("--device", default="auto", help="Execution device.")
    diarize_selection_parser.add_argument("--whisper-model", default="medium.en")
    diarize_selection_parser.add_argument("--batch-size", type=int, default=8)
    diarize_selection_parser.add_argument("--language", default=None)
    diarize_selection_parser.add_argument(
        "--pyannote-pipeline-model",
        default=DEFAULT_PYANNOTE_PIPELINE_MODEL,
    )
    diarize_selection_parser.add_argument(
        "--pyannote-segmentation-model",
        default=DEFAULT_PYANNOTE_SEGMENTATION_MODEL,
    )
    diarize_selection_parser.add_argument(
        "--nemo-msdd-model",
        default="",
        help="Optional NeMo MSDD checkpoint or .nemo path to use instead of the shared default model.",
    )
    diarize_selection_parser.add_argument(
        "--source-separation",
        action="store_true",
        default=False,
        help="Enable source separation before transcription.",
    )
    diarize_selection_parser.add_argument(
        "--skip-review",
        action="store_true",
        default=False,
        help="Skip review bundle generation during diarization.",
    )
    diarize_selection_parser.set_defaults(func=cmd_run_diarization_selection)

    local_web_parser = subparsers.add_parser(
        "local-web",
        help="Manage the local dashboard in the background on localhost.",
    )
    local_web_subparsers = local_web_parser.add_subparsers(
        dest="local_web_command",
        required=True,
    )

    local_web_start = local_web_subparsers.add_parser(
        "start",
        help="Start the local dashboard and print the resolved localhost URL.",
    )
    local_web_start.add_argument("--port", type=int, default=8000, help="Preferred localhost port.")
    local_web_start.add_argument(
        "--server",
        choices=["auto", "threaded", "waitress", "wsgiref"],
        default="auto",
        help="Runtime server to use. 'auto' prefers Waitress when installed.",
    )
    local_web_start.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Worker thread count for Waitress-backed runs and auto mode.",
    )
    local_web_start.add_argument(
        "--restart",
        action="store_true",
        default=False,
        help="Stop an existing local dashboard before starting a new one.",
    )
    local_web_start.set_defaults(func=cmd_local_web_start)

    local_web_status = local_web_subparsers.add_parser(
        "status",
        help="Show whether the background localhost dashboard is running.",
    )
    local_web_status.set_defaults(func=cmd_local_web_status)

    local_web_stop = local_web_subparsers.add_parser(
        "stop",
        help="Stop the background localhost dashboard.",
    )
    local_web_stop.set_defaults(func=cmd_local_web_stop)
    return parser


def iter_audio_files(audio_dir: Path) -> list[Path]:
    """List supported media inputs while excluding generated helper files."""

    if not audio_dir.is_dir():
        return []
    resolved_audio_dir = audio_dir.resolve()
    audio_suffixes = {
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

    # KaggleBabyNoises (and any future corpus drop) can contain filenames that
    # exceed the filesystem path limit. A single ENAMETOOLONG from is_file()
    # would otherwise abort the whole walk and break every page that calls us
    # (training labels save, fine-tune picker, media library...). Skip the
    # offending entry and keep going so the rest of the inventory still lists.
    def _is_audio_file(path: Path) -> bool:
        try:
            if not path.is_file():
                return False
        except OSError:
            return False
        return path.suffix.lower() in audio_suffixes and "_whisper_input" not in path.stem

    return sorted(
        (path for path in resolved_audio_dir.rglob("*") if _is_audio_file(path)),
        key=lambda item: str(item.relative_to(resolved_audio_dir)).lower(),
    )


def count_queued_urls(list_path: Path) -> int:
    """Count non-comment queue entries in the YouTube links file."""

    if not list_path.is_file():
        return 0
    return sum(
        1
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def latest_directory(root: Path) -> Path | None:
    """Return the most recently modified output directory under a root."""

    if not root.is_dir():
        return None
    directories = [path for path in root.iterdir() if path.is_dir()]
    if not directories:
        return None
    return max(directories, key=lambda path: path.stat().st_mtime)


def is_diarization_run_directory(path: Path) -> bool:
    """Return true when a directory looks like a site-created diarization run."""

    return (
        (path / "metadata.json").is_file()
        or (path / "selected_audio.txt").is_file()
        or (path / "logs" / "runtime_summary.tsv").is_file()
    )


def iter_diarization_run_directories(root: Path) -> list[Path]:
    """List legacy and backend-specific diarization run directories."""

    if not root.is_dir():
        return []
    run_dirs: list[Path] = []
    try:
        top_level = list(root.iterdir())
    except (OSError, PermissionError):
        return run_dirs
    for path in top_level:
        if not path.is_dir():
            continue
        if is_diarization_run_directory(path):
            run_dirs.append(path)
            continue
        # Per-backend nesting: outputs/diarization_runs/<backend>/<run>/.
        # Skip silently if a child folder is unreadable (broken symlink, perm
        # denied, NFS hiccup) so a single bad subdirectory cannot crash the
        # whole page state.
        try:
            children = list(path.iterdir())
        except (OSError, PermissionError):
            continue
        for child in children:
            if child.is_dir() and is_diarization_run_directory(child):
                run_dirs.append(child)
    return run_dirs


def latest_diarization_directory(root: Path) -> Path | None:
    """Return the newest diarization run, including backend subfolders.

    Tiebreaks on the directory name (timestamp-prefixed) so two runs created
    inside the same wall-clock second still produce a deterministic 'latest'
    pick, which matters on shared file systems where mtime resolution can
    collapse under load.
    """

    directories = iter_diarization_run_directories(root)
    if not directories:
        return None
    return max(directories, key=lambda path: (path.stat().st_mtime, path.name))


def read_non_comment_lines(path: Path) -> list[str]:
    """Load non-empty, non-comment lines from a plain-text selection file."""

    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def sanitize_log_component(value: str) -> str:
    """Render a filename-safe component for generated log files."""

    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    safe = safe.strip("._")
    return safe or "run"


def summarize_error_log(path: Path) -> str:
    """Return the most useful short error line from a per-file stderr log."""

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


def diarization_status_from_result(returncode: int, stderr_path: Path) -> tuple[str, bool, str]:
    """Classify one diarization item without turning expected no-speech clips into job failures."""

    if returncode == 0:
        return "ok", False, ""
    error_summary = summarize_error_log(stderr_path)
    if "Whisper returned an empty transcript" in error_summary:
        return "no_speech", False, error_summary
    return "failed", True, error_summary


def format_command(command: Sequence[str]) -> str:
    """Render a subprocess command in shell-friendly form for logging."""

    return shlex.join(command)


def run_external(command: Sequence[str], *, dry_run: bool) -> int:
    """Print and optionally execute an external command from the project root."""

    print(format_command(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def require_command(name: str) -> None:
    """Abort early when a required external command is missing from PATH."""

    if shutil.which(name):
        return
    raise SystemExit(f"Required command not found in PATH: {name}")


def print_status_block(title: str, value: str) -> None:
    """Keep status output formatting consistent across fields."""

    print(f"{title}: {value}")


def cmd_status(args: argparse.Namespace) -> int:
    """Report the current input, queue, and output state of the workspace."""

    audio_files = iter_audio_files(AUDIO_DIR)
    latest_site_diarization = latest_diarization_directory(DIARIZATION_RUNS_ROOT)
    latest_youtube_run = latest_directory(YOUTUBE_RUNS_ROOT)

    print_status_block("Workspace", str(PROJECT_ROOT))
    print_status_block("Audio inputs", f"{len(audio_files)} file(s)")
    for path in audio_files[: max(args.limit, 0)]:
        print(f"  - {path.relative_to(AUDIO_DIR)}")
    if len(audio_files) > args.limit:
        print(f"  - ... {len(audio_files) - args.limit} more")

    print_status_block("Queued YouTube URLs", str(count_queued_urls(YOUTUBE_LINKS)))
    print_status_block("sbatch", shutil.which("sbatch") or "not found")
    print_status_block("ffmpeg", shutil.which("ffmpeg") or "not found")
    token_status = "set" if os.environ.get("HF_TOKEN") or local_secret_is_set("HF_TOKEN") else "not set"
    print_status_block("HF_TOKEN", token_status)
    print_status_block("Latest site diarization run", str(latest_site_diarization) if latest_site_diarization else "none")
    print_status_block("Latest YouTube conversion run", str(latest_youtube_run) if latest_youtube_run else "none")
    print_status_block(
        "Latest failed-links report",
        str(FAILED_LINKS) if FAILED_LINKS.is_file() else "none",
    )
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Generate review artifacts for an existing subtitle output."""

    write_review_bundle(
        srt_path=Path(args.srt),
        media_path=Path(args.media) if args.media else None,
        output_html=Path(args.output_html) if args.output_html else None,
        report_tsv=Path(args.report_tsv) if args.report_tsv else None,
        audio_dir=Path(args.audio_dir),
    )
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Run the local unit test suite."""

    command = ["bash", str(TEST_SCRIPT)]
    return run_external(command, dry_run=args.dry_run)


def cmd_serve_web(args: argparse.Namespace) -> int:
    """Start the lightweight workflow dashboard."""

    command = [
        sys.executable,
        str(WEB_APP_SCRIPT),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--server",
        args.server,
        "--threads",
        str(args.threads),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def cmd_convert_youtube_selection(args: argparse.Namespace) -> int:
    """Convert a selected subset of queued YouTube URLs with the shared history index."""

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    YOUTUBE_HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(YOUTUBE_BATCH_SCRIPT),
        "--list",
        str(Path(args.urls_file).expanduser().resolve()),
        "--queue-file",
        str(Path(args.queue_file).expanduser().resolve()),
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--index-file",
        str(Path(args.index_file).expanduser().resolve()),
        "--log-dir",
        str(run_dir),
        "--resolved-output-file",
        str(run_dir / "conversion_report.tsv"),
        "--resolved-audio-list",
        str(run_dir / "resolved_audio_paths.txt"),
        "--removed-failed-file",
        str(run_dir / "queue_updates.tsv"),
        "--links-error-dir",
        str(PROJECT_ROOT / "youtube_links_err"),
    ]
    if args.force_redownload:
        command.append("--force-redownload")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def cmd_run_diarization_selection(args: argparse.Namespace) -> int:
    """Run local diarization sequentially for the selected audio files."""

    selected_files = read_non_comment_lines(Path(args.audio_list_file).expanduser().resolve())
    if not selected_files:
        raise ValueError("No audio files were provided for diarization.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_path = log_dir / "runtime_summary.tsv"
    summary_path.write_text(
        "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n",
        encoding="utf-8",
    )
    failures = 0
    no_speech = 0
    for selected_file in selected_files:
        safe_name = sanitize_log_component(selected_file)
        stdout_path = log_dir / f"{safe_name}.out"
        stderr_path = log_dir / f"{safe_name}.err"
        command = [
            sys.executable,
            str(DIARIZATION_SCRIPT),
            "--audio",
            selected_file,
            "--input-dir",
            str(AUDIO_DIR),
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
            "--whisper-model",
            args.whisper_model,
            "--batch-size",
            str(args.batch_size),
            "--diarizer",
            args.backend,
            "--pyannote-pipeline-model",
            args.pyannote_pipeline_model,
            "--pyannote-segmentation-model",
            args.pyannote_segmentation_model,
        ]
        nemo_msdd_model = getattr(args, "nemo_msdd_model", "")
        if nemo_msdd_model:
            command.extend(["--nemo-msdd-model", nemo_msdd_model])
        if args.language:
            command.extend(["--language", args.language])
        command.append("--stem" if args.source_separation else "--no-stem")
        command.append("--no-generate-review" if args.skip_review else "--generate-review")

        print(f"Starting diarization: {selected_file}")
        start_time = time.monotonic()
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        runtime_seconds = time.monotonic() - start_time
        status, is_failure, error_summary = diarization_status_from_result(completed.returncode, stderr_path)
        if status == "no_speech":
            no_speech += 1
        if is_failure:
            failures += 1
        summary_row = "\t".join(
            [
                selected_file,
                status,
                f"{runtime_seconds:.2f}",
                str(stdout_path),
                str(stderr_path),
                error_summary.replace("\t", " ").replace("\n", " "),
            ]
        )
        with summary_path.open("a", encoding="utf-8") as summary_handle:
            summary_handle.write(summary_row + "\n")
        print(f"Finished diarization: {selected_file} ({status}, {runtime_seconds:.2f}s)")

    print(f"Processed {len(selected_files)} audio file(s).")
    print(f"No speech detected: {no_speech}")
    print(f"Failed: {failures}")
    print(f"Summary: {summary_path}")
    return 1 if failures else 0


def cmd_local_web_start(args: argparse.Namespace) -> int:
    """Launch the dashboard in the background on localhost."""

    ensure_local_web_dir()
    state = read_local_web_state()
    if state["stale_pid"]:
        clear_local_web_runtime_files(LOCAL_WEB_PID, LOCAL_WEB_URL)
        state = read_local_web_state()

    if state["running"]:
        if not args.restart:
            url = state["url"] or "pending startup confirmation"
            print(f"Local dashboard is already running: {url}")
            print(f"Log: {LOCAL_WEB_LOG}")
            return 0
        if not stop_background_process(int(state["pid"])):
            print("Error: unable to stop the existing local dashboard.", file=sys.stderr)
            return 1
        clear_local_web_runtime_files(LOCAL_WEB_PID, LOCAL_WEB_URL)

    if args.restart:
        for pid in local_dashboard_process_pids():
            if state["pid"] and pid == int(state["pid"]):
                continue
            stop_background_process(pid, timeout_seconds=2.0)

    clear_local_web_runtime_files(LOCAL_WEB_PID, LOCAL_WEB_URL)
    command = build_local_web_command(
        port=args.port,
        server=args.server,
        threads=args.threads,
        url_file=LOCAL_WEB_URL,
    )
    with LOCAL_WEB_LOG.open("a", encoding="utf-8") as log_handle:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"\n[{timestamp}] Starting local dashboard: {format_command(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    LOCAL_WEB_PID.write_text(f"{process.pid}\n", encoding="utf-8")
    url = wait_for_local_web_start(
        process,
        preferred_port=args.port,
        url_path=LOCAL_WEB_URL,
    )
    if url:
        print(f"Local dashboard is running at {url}")
        print(f"Log: {LOCAL_WEB_LOG}")
        return 0

    if process.poll() is None:
        stop_background_process(process.pid, timeout_seconds=2.0)
    clear_local_web_runtime_files(LOCAL_WEB_PID, LOCAL_WEB_URL)
    print("Error: local dashboard did not finish startup cleanly.", file=sys.stderr)
    recent_log = tail_file(LOCAL_WEB_LOG)
    if recent_log:
        print(recent_log, file=sys.stderr)
    return 1


def cmd_local_web_status(args: argparse.Namespace) -> int:
    """Report the background dashboard status."""

    state = read_local_web_state()
    if state["stale_pid"]:
        clear_local_web_runtime_files(LOCAL_WEB_PID, LOCAL_WEB_URL)
        state = read_local_web_state()

    if state["running"]:
        if not state["url"]:
            recovered_url = None
            for candidate_url in local_web_candidate_urls(preferred_port=8000):
                if local_web_healthcheck(candidate_url):
                    LOCAL_WEB_URL.write_text(f"{candidate_url}\n", encoding="utf-8")
                    recovered_url = candidate_url
                    break
            if recovered_url:
                state["url"] = recovered_url
        print("Local dashboard: running")
        print(f"PID: {state['pid']}")
        print(f"URL: {state['url'] or 'pending startup confirmation'}")
    else:
        print("Local dashboard: stopped")
    print(f"Log: {LOCAL_WEB_LOG}")
    return 0


def cmd_local_web_stop(args: argparse.Namespace) -> int:
    """Stop the background dashboard if it is running."""

    state = read_local_web_state()
    if state["stale_pid"]:
        clear_local_web_runtime_files(LOCAL_WEB_PID, LOCAL_WEB_URL)
        print("Local dashboard was not running.")
        print(f"Log: {LOCAL_WEB_LOG}")
        return 0
    if not state["running"] or not state["pid"]:
        print("Local dashboard is not running.")
        print(f"Log: {LOCAL_WEB_LOG}")
        return 0
    if not stop_background_process(int(state["pid"])):
        print("Error: unable to stop the local dashboard.", file=sys.stderr)
        return 1
    clear_local_web_runtime_files(LOCAL_WEB_PID, LOCAL_WEB_URL)
    print("Stopped local dashboard.")
    print(f"Log: {LOCAL_WEB_LOG}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
