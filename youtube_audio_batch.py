#!/usr/bin/env python3
"""Batch-download YouTube URLs into numbered 16 kHz mono WAV files.

Each call reads a queue file (one URL per line, ``#`` comments allowed),
downloads any URLs that are not already represented in the on-disk index,
and renumbers the results so the rest of the pipeline (diarization,
fine-tuning) sees a stable ``001_*.wav`` ordering.

Defaults:
* ``--list``        ./youtube_links.txt
* ``--output-dir``  ./audio_in/youtube_links  (matches what the dashboard
                                               batch script also targets, so
                                               CLI and site runs land in the
                                               same place)
* ``--index-file``  ./outputs/youtube_conversion_history/url_audio_index.tsv

Usage:
    python youtube_audio_batch.py [--list FILE] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import pathlib
import re
import shutil
import sys
import tempfile
from typing import Iterable

import yt_dlp

# Helpers that used to be duplicated in this module now live in their canonical
# homes so a single fix propagates everywhere.
from audio_numbering import (
    AUDIO_EXTENSIONS,
    NUMBERED_PREFIX,
    next_available_number,
)
from workflow_background import utc_now_iso

INDEX_COLUMNS = [
    "url",
    "video_id",
    "status",
    "audio_file",
    "audio_path",
    "title",
    "last_attempt_utc",
    "note",
]
CONVERSION_REPORT_COLUMNS = [
    "url",
    "video_id",
    "status",
    "queue_action",
    "audio_file",
    "audio_path",
    "title",
    "summary",
    "detail_log",
    "last_attempt_utc",
]
QUEUE_UPDATE_COLUMNS = [
    "url",
    "issue_category",
    "queue_action",
    "queue_result",
    "summary",
    "last_attempt_utc",
]
ISSUE_REPORT_COLUMNS = [
    "run_utc",
    "url",
    "video_id",
    "issue_category",
    "queue_action",
    "queue_result",
    "summary",
    "last_attempt_utc",
    "resolved_audio_path",
    "detail_log",
]
DEFAULT_YOUTUBE_FORMAT = "bestaudio[ext=m4a]/bestaudio[acodec!=none]/bestaudio/best"
DEFAULT_YOUTUBE_EXTRACTOR_ARGS = {
    "youtube": {
        # `android_vr` currently avoids the SABR-only web client path that was
        # producing repeated 403 download failures in the local workflow.
        "player_client": ["android_vr"],
    }
}
NO_DATA_ERROR_MARKERS = (
    "video unavailable",
    "this video is unavailable",
    "private video",
    "video has been removed",
    "the uploader has not made this video available",
    "this content isn't available",
    "content is not available",
    "no video formats found",
    "no working video formats found",
    "requested format is not available",
    "unsupported url",
    "incomplete youtube id",
    "members-only",
    "sign in to confirm your age",
    "age-restricted",
    "unavailable in your country",
)


def iter_urls(list_path: pathlib.Path) -> Iterable[str]:
    with list_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def resolve_ffmpeg_location() -> str | None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else ""


def load_index(index_path: pathlib.Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    if not index_path.exists():
        return records
    with index_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            url = (row.get("url") or "").strip()
            if url:
                records[url] = row
    return records


def save_index(index_path: pathlib.Path, records: dict[str, dict[str, str]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS, delimiter="\t")
        writer.writeheader()
        for url in sorted(records):
            row = {key: records[url].get(key, "") for key in INDEX_COLUMNS}
            writer.writerow(row)


def write_conversion_report(report_path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CONVERSION_REPORT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_resolved_audio_list(audio_list_path: pathlib.Path, paths: list[str]) -> None:
    audio_list_path.parent.mkdir(parents=True, exist_ok=True)
    with audio_list_path.open("w", encoding="utf-8") as fh:
        for path in paths:
            fh.write(f"{path}\n")


def prune_failed_urls_from_list(
    list_path: pathlib.Path,
    failed_urls: set[str],
) -> list[str]:
    if not failed_urls:
        return []

    with list_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    removed: list[str] = []
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped in failed_urls:
            removed.append(stripped)
            continue
        kept.append(line)

    list_path.write_text("".join(kept), encoding="utf-8")
    return removed


def write_queue_update_table(
    queue_update_path: pathlib.Path,
    failed_records: list[dict[str, str]],
    removed_urls: set[str],
) -> None:
    queue_update_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_update_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUEUE_UPDATE_COLUMNS, delimiter="\t")
        writer.writeheader()
        for record in failed_records:
            url = record.get("url", "")
            queue_action = record.get("queue_action", "keep_in_queue")
            if queue_action == "remove_from_queue":
                queue_result = "removed" if url in removed_urls else "not_found"
            else:
                queue_result = "kept"
            writer.writerow(
                {
                    "url": url,
                    "issue_category": record.get("issue_category", ""),
                    "queue_action": queue_action,
                    "queue_result": queue_result,
                    "summary": record.get("summary", ""),
                    "last_attempt_utc": record.get("last_attempt_utc", ""),
                }
            )


def write_failed_links_report(
    report_path: pathlib.Path,
    failed_records: list[dict[str, str]],
    removed_urls: set[str],
    run_utc: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ISSUE_REPORT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for record in failed_records:
            url = record.get("url", "")
            queue_action = record.get("queue_action", "keep_in_queue")
            if queue_action == "remove_from_queue":
                queue_result = "removed" if url in removed_urls else "not_found"
            else:
                queue_result = "kept"
            writer.writerow(
                {
                    "run_utc": run_utc,
                    "url": url,
                    "video_id": record.get("video_id", ""),
                    "issue_category": record.get("issue_category", ""),
                    "queue_action": queue_action,
                    "queue_result": queue_result,
                    "summary": record.get("summary", ""),
                    "last_attempt_utc": record.get("last_attempt_utc", ""),
                    "resolved_audio_path": record.get("audio_path", ""),
                    "detail_log": record.get("detail_log", ""),
                }
            )


def summarize_exception(exc: Exception) -> str:
    """Collapse an exception into one short, readable line."""

    message = re.sub(r"\s+", " ", str(exc or "")).strip()
    return message or exc.__class__.__name__


def classify_issue(summary: str) -> str:
    """Separate dead links from retryable downloader or environment errors."""

    lowered = (summary or "").strip().lower()
    if any(marker in lowered for marker in NO_DATA_ERROR_MARKERS):
        return "no_data"
    return "retry"


def queue_action_for_issue(issue_category: str) -> str:
    """Only remove URLs when the workflow is confident the link has no usable data."""

    return "remove_from_queue" if issue_category == "no_data" else "keep_in_queue"


def choose_available_output_path(target_path: pathlib.Path) -> pathlib.Path:
    """Avoid overwriting an existing audio file when yt-dlp resolves the same title."""

    if not target_path.exists():
        return target_path

    counter = 2
    while True:
        candidate = target_path.with_name(
            f"{target_path.stem}_{counter}{target_path.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def locate_downloaded_wav(
    temp_dir: pathlib.Path,
    info: dict,
    prepared_path: pathlib.Path,
) -> pathlib.Path:
    """Find the WAV file produced by yt-dlp inside the temporary workspace."""

    candidate = prepared_path.with_suffix(".wav")
    if candidate.is_file():
        return candidate

    video_id = str(info.get("id") or "")
    if video_id:
        matches = sorted(temp_dir.glob(f"*_{video_id}.wav"))
        if matches:
            return matches[0]

    wav_matches = sorted(temp_dir.glob("*.wav"))
    if len(wav_matches) == 1:
        return wav_matches[0]

    raise FileNotFoundError("yt-dlp finished but the converted WAV file was not found.")


def download_one(
    url: str,
    out_dir: pathlib.Path,
    ffmpeg_location: str,
) -> tuple[pathlib.Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=out_dir,
        prefix=".youtube_tmp_",
    ) as temp_dir_name:
        temp_dir = pathlib.Path(temp_dir_name)
        ydl_opts = {
            "outtmpl": str(temp_dir / "%(title).80s_%(id)s.%(ext)s"),
            "format": DEFAULT_YOUTUBE_FORMAT,
            "extractor_args": DEFAULT_YOUTUBE_EXTRACTOR_ARGS,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                },
            ],
            "postprocessor_args": ["-ar", "16000", "-ac", "1"],
            "ffmpeg_location": ffmpeg_location,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            prepared_path = pathlib.Path(ydl.prepare_filename(info))
            wav_path = locate_downloaded_wav(temp_dir, info, prepared_path)

        target_path = choose_available_output_path(out_dir / wav_path.name)
        shutil.move(str(wav_path), str(target_path))
        return target_path.resolve(), info


def sanitize_for_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._")
    return (safe or "url")[:80]


def find_existing_wav_for_video_id(
    out_dir: pathlib.Path, video_id: str
) -> pathlib.Path | None:
    if not video_id or not out_dir.exists():
        return None
    for candidate in sorted(out_dir.glob(f"*_{video_id}.wav")):
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_audio_path_in_dir(
    candidate_path: str, out_dir: pathlib.Path
) -> pathlib.Path | None:
    if not candidate_path:
        return None
    try:
        resolved_path = pathlib.Path(candidate_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    if not resolved_path.is_file():
        return None
    try:
        resolved_path.relative_to(out_dir)
    except ValueError:
        return None
    return resolved_path


def collect_used_prefix_numbers(out_dir: pathlib.Path) -> set[int]:
    used_numbers: set[int] = set()
    if not out_dir.exists():
        return used_numbers
    for candidate in out_dir.iterdir():
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if "_whisper_input" in candidate.stem:
            continue
        match = NUMBERED_PREFIX.match(candidate.name)
        if match:
            used_numbers.add(int(match.group(1)))
    return used_numbers


def ensure_numbered_audio_path(
    candidate_path: pathlib.Path,
    out_dir: pathlib.Path,
    used_numbers: set[int],
) -> tuple[pathlib.Path, bool]:
    resolved_path = candidate_path.expanduser().resolve()
    try:
        resolved_path.relative_to(out_dir)
    except ValueError:
        return resolved_path, False
    if not resolved_path.is_file():
        return resolved_path, False

    match = NUMBERED_PREFIX.match(resolved_path.name)
    if match:
        used_numbers.add(int(match.group(1)))
        return resolved_path, False

    # Pick the next available numeric prefix and atomically claim it. We use
    # os.link + unlink instead of rename() + exists() because two parallel
    # downloader processes could race between the exists() check and rename(),
    # silently clobbering each other's WAVs. os.link fails fast (FileExistsError)
    # if the target already lives on disk, so we can spin to the next number
    # without losing data. Falls back to plain rename on filesystems that do
    # not support hard links (rare on the WAVE storage).
    import errno
    import os as _os
    prefix_num = next_available_number(used_numbers)
    while True:
        target_path = resolved_path.with_name(f"{prefix_num:03d}_{resolved_path.name}")
        try:
            _os.link(resolved_path, target_path)
            resolved_path.unlink()
            break
        except FileExistsError:
            prefix_num += 1
            while prefix_num in used_numbers:
                prefix_num += 1
            continue
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.EXDEV, errno.ENOSYS):
                # Cross-device or unsupported link: fall back to plain rename.
                # Still racy but matches prior behavior on such filesystems.
                if target_path.exists():
                    prefix_num += 1
                    while prefix_num in used_numbers:
                        prefix_num += 1
                    continue
                resolved_path.rename(target_path)
                break
            raise
    used_numbers.add(prefix_num)
    return target_path.resolve(), True


def main() -> int:
    project_root = pathlib.Path(__file__).resolve().parent
    default_list_file = project_root / "youtube_links.txt"
    default_index_file = project_root / "outputs" / "youtube_conversion_history" / "url_audio_index.tsv"
    default_links_error_dir = project_root / "youtube_links_err"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        dest="list_file",
        default=str(default_list_file),
        help="Path to file containing YouTube URLs (default: ./youtube_links.txt)",
    )
    parser.add_argument(
        "--queue-file",
        default=None,
        help="Optional real queue file to update when no-data URLs should be removed.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "audio_in" / "youtube_links"),
        help="Where to store WAVs (default: ./audio_in/youtube_links)",
    )
    parser.add_argument(
        "--index-file",
        default=str(default_index_file),
        help="Persistent URL-to-audio mapping table (TSV)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory for per-URL conversion logs (.out/.err)",
    )
    parser.add_argument(
        "--resolved-output-file",
        default=None,
        help="Optional per-run TSV with URL resolution results",
    )
    parser.add_argument(
        "--resolved-audio-list",
        default=None,
        help="Optional per-run text file containing resolved audio paths",
    )
    parser.add_argument(
        "--keep-failed-in-list",
        action="store_true",
        default=False,
        help="Do not remove failed URLs from the list file",
    )
    parser.add_argument(
        "--removed-failed-file",
        default=None,
        help="Optional per-run TSV report of failed URLs and whether they were removed",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        default=False,
        help="Redownload URLs even if they already exist in the index",
    )
    parser.add_argument(
        "--links-error-dir",
        default=str(default_links_error_dir),
        help="Directory where failed/removed YouTube URL reports are written",
    )
    parser.add_argument(
        "--links-error-report",
        default=None,
        help="Optional explicit failed-links report path (TSV)",
    )
    args = parser.parse_args()

    list_path = pathlib.Path(args.list_file).expanduser().resolve()
    if not list_path.exists():
        print(f"List file not found: {list_path}", file=sys.stderr)
        return 1
    queue_path = (
        pathlib.Path(args.queue_file).expanduser().resolve()
        if args.queue_file
        else list_path
    )

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    index_path = pathlib.Path(args.index_file).expanduser().resolve()
    used_numbers = collect_used_prefix_numbers(out_dir)
    links_error_dir = pathlib.Path(args.links_error_dir).expanduser().resolve()
    links_error_dir.mkdir(parents=True, exist_ok=True)
    log_dir = (
        pathlib.Path(args.log_dir).expanduser().resolve()
        if args.log_dir
        else None
    )
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_location: str | None = None

    urls = list(iter_urls(list_path))
    if not urls:
        print("No URLs found in list (or all lines were comments/blank).", file=sys.stderr)
        return 1

    index_records = load_index(index_path)
    conversion_rows: list[dict[str, str]] = []
    resolved_audio_paths: list[str] = []
    failed_records: list[dict[str, str]] = []
    successes = 0
    failures = 0
    skipped = 0
    no_data_failures = 0
    retry_failures = 0
    total_urls = len(urls)
    for idx, url in enumerate(urls, start=1):
        video_id = extract_video_id(url)
        safe_name = sanitize_for_filename(url)
        detail_log_path = (
            log_dir / "item_logs" / f"download_{idx:03d}_{safe_name}.log"
            if log_dir
            else None
        )
        existing = index_records.get(url, {})
        existing_audio_path = (existing.get("audio_path") or "").strip()
        existing_audio_in_out_dir = resolve_audio_path_in_dir(existing_audio_path, out_dir)
        if (
            not args.force_redownload
            and existing.get("status") == "ok"
            and existing_audio_in_out_dir is not None
        ):
            existing_audio_in_out_dir, was_renamed = ensure_numbered_audio_path(
                existing_audio_in_out_dir,
                out_dir,
                used_numbers,
            )
            skipped += 1
            existing["last_attempt_utc"] = utc_now_iso()
            existing["note"] = (
                "already_downloaded_renumbered" if was_renamed else "already_downloaded"
            )
            existing["audio_file"] = existing_audio_in_out_dir.name
            existing["audio_path"] = str(existing_audio_in_out_dir)
            index_records[url] = existing
            save_index(index_path, index_records)
            conversion_rows.append(
                {
                    "url": url,
                    "video_id": existing.get("video_id", video_id),
                    "status": "skipped",
                    "queue_action": "kept_in_queue",
                    "audio_file": existing_audio_in_out_dir.name,
                    "audio_path": str(existing_audio_in_out_dir),
                    "title": existing.get("title", ""),
                    "summary": "Already converted. Reusing the saved WAV file.",
                    "detail_log": "",
                    "last_attempt_utc": existing["last_attempt_utc"],
                }
            )
            resolved_audio_paths.append(str(existing_audio_in_out_dir))
            if was_renamed:
                print(f"[{idx}/{total_urls}] renumbered | {existing_audio_in_out_dir.name} | {url}")
            print(f"[{idx}/{total_urls}] skipped | {existing_audio_in_out_dir.name} | {url}")
            continue

        if (
            not args.force_redownload
            and existing.get("status") == "ok"
            and existing_audio_path
            and existing_audio_in_out_dir is None
        ):
            print(
                f"[NOTICE] Ignoring indexed audio path outside output-dir for {url}: "
                f"{existing_audio_path}",
                file=sys.stderr,
            )

        existing_wav_path = (
            find_existing_wav_for_video_id(out_dir, video_id)
            if not args.force_redownload
            else None
        )
        if existing_wav_path is not None:
            existing_wav_path, was_renamed = ensure_numbered_audio_path(
                existing_wav_path,
                out_dir,
                used_numbers,
            )
            skipped += 1
            index_records[url] = {
                "url": url,
                "video_id": str(video_id),
                "status": "ok",
                "audio_file": existing_wav_path.name,
                "audio_path": str(existing_wav_path),
                "title": existing.get("title", ""),
                "last_attempt_utc": utc_now_iso(),
                "note": (
                    "already_in_audio_in_renumbered"
                    if was_renamed
                    else "already_in_audio_in"
                ),
            }
            save_index(index_path, index_records)
            conversion_rows.append(
                {
                    "url": url,
                    "video_id": str(video_id),
                    "status": "skipped",
                    "queue_action": "kept_in_queue",
                    "audio_file": existing_wav_path.name,
                    "audio_path": str(existing_wav_path),
                    "title": existing.get("title", ""),
                    "summary": "Already converted. Reusing the saved WAV file from audio_in.",
                    "detail_log": "",
                    "last_attempt_utc": index_records[url]["last_attempt_utc"],
                }
            )
            resolved_audio_paths.append(str(existing_wav_path))
            if was_renamed:
                print(f"[{idx}/{total_urls}] renumbered | {existing_wav_path.name} | {url}")
            print(f"[{idx}/{total_urls}] skipped | {existing_wav_path.name} | {url}")
            continue

        try:
            if ffmpeg_location is None:
                ffmpeg_location = resolve_ffmpeg_location()
            if not ffmpeg_location:
                raise RuntimeError(
                    "No ffmpeg binary found. Install ffmpeg or ensure imageio-ffmpeg is installed."
                )
            if detail_log_path:
                detail_log_path.parent.mkdir(parents=True, exist_ok=True)
                with detail_log_path.open("w", encoding="utf-8") as detail_fh:
                    with contextlib.redirect_stdout(detail_fh), contextlib.redirect_stderr(detail_fh):
                        wav_path, info = download_one(url, out_dir, ffmpeg_location)
            else:
                wav_path, info = download_one(url, out_dir, ffmpeg_location)
            wav_path, was_renamed = ensure_numbered_audio_path(
                wav_path,
                out_dir,
                used_numbers,
            )
            successes += 1
            index_records[url] = {
                "url": url,
                "video_id": str(info.get("id") or video_id),
                "status": "ok",
                "audio_file": wav_path.name,
                "audio_path": str(wav_path),
                "title": str(info.get("title") or ""),
                "last_attempt_utc": utc_now_iso(),
                "note": "downloaded_renumbered" if was_renamed else "downloaded",
            }
            save_index(index_path, index_records)
            conversion_rows.append(
                {
                    "url": url,
                    "video_id": str(info.get("id") or video_id),
                    "status": "downloaded",
                    "queue_action": "kept_in_queue",
                    "audio_file": wav_path.name,
                    "audio_path": str(wav_path),
                    "title": str(info.get("title") or ""),
                    "summary": "Downloaded and converted to WAV.",
                    "detail_log": "",
                    "last_attempt_utc": index_records[url]["last_attempt_utc"],
                }
            )
            resolved_audio_paths.append(str(wav_path))
            if detail_log_path and detail_log_path.exists():
                detail_log_path.unlink()
            if was_renamed:
                print(f"[{idx}/{total_urls}] renumbered | {wav_path.name} | {url}")
            print(f"[{idx}/{total_urls}] downloaded | {wav_path.name} | {url}")
        except Exception as exc:  # pragma: no cover - convenience
            failures += 1
            err_msg = summarize_exception(exc)
            issue_category = classify_issue(err_msg)
            queue_action = queue_action_for_issue(issue_category)
            detail_log_value = str(detail_log_path) if detail_log_path and detail_log_path.is_file() else ""
            if issue_category == "no_data":
                no_data_failures += 1
            else:
                retry_failures += 1
            index_records[url] = {
                "url": url,
                "video_id": existing.get("video_id") or video_id,
                "status": issue_category,
                "audio_file": existing.get("audio_file", ""),
                "audio_path": existing.get("audio_path", ""),
                "title": existing.get("title", ""),
                "last_attempt_utc": utc_now_iso(),
                "note": err_msg,
            }
            save_index(index_path, index_records)
            conversion_rows.append(
                {
                    "url": url,
                    "video_id": index_records[url].get("video_id", ""),
                    "status": issue_category,
                    "queue_action": queue_action,
                    "audio_file": "",
                    "audio_path": "",
                    "title": existing.get("title", ""),
                    "summary": err_msg,
                    "detail_log": detail_log_value,
                    "last_attempt_utc": index_records[url]["last_attempt_utc"],
                }
            )
            failed_records.append(
                {
                    "url": url,
                    "summary": err_msg,
                    "last_attempt_utc": index_records[url]["last_attempt_utc"],
                    "video_id": index_records[url].get("video_id", ""),
                    "audio_path": index_records[url].get("audio_path", ""),
                    "issue_category": issue_category,
                    "queue_action": queue_action,
                    "detail_log": detail_log_value,
                }
            )
            queue_label = "removed from queue" if queue_action == "remove_from_queue" else "kept in queue"
            print(f"[{idx}/{total_urls}] {issue_category} | {queue_label} | {url}")
            print(f"  reason: {err_msg}")

    removed_urls: set[str] = set()
    if failures > 0 and not args.keep_failed_in_list:
        failed_urls = {
            record["url"]
            for record in failed_records
            if record.get("queue_action") == "remove_from_queue"
        }
        try:
            removed = prune_failed_urls_from_list(queue_path, failed_urls)
            removed_urls = set(removed)
            if removed:
                print(f"[QUEUE] Removed {len(removed)} no-data URL(s) from {queue_path}")
                for url in removed:
                    print(f"  removed: {url}")
            elif failed_urls:
                print("[QUEUE] No-data URLs were not found in the queue file during cleanup.")
        except Exception as exc:
            print(f"[QUEUE] Failed to update {queue_path}: {exc}")

    if args.resolved_output_file:
        resolved_path = pathlib.Path(args.resolved_output_file).expanduser().resolve()
        write_conversion_report(resolved_path, conversion_rows)
    if args.resolved_audio_list:
        audio_list_path = pathlib.Path(args.resolved_audio_list).expanduser().resolve()
        write_resolved_audio_list(audio_list_path, resolved_audio_paths)
    removed_failed_path = (
        pathlib.Path(args.removed_failed_file).expanduser().resolve()
        if args.removed_failed_file
        else (log_dir / "queue_updates.tsv" if log_dir else None)
    )
    if removed_failed_path:
        write_queue_update_table(removed_failed_path, failed_records, removed_urls)

    if failed_records:
        run_utc = utc_now_iso()
        failed_links_report = (
            pathlib.Path(args.links_error_report).expanduser().resolve()
            if args.links_error_report
            else (
                links_error_dir
                / f"failed_links_{run_utc.replace(':', '').replace('+00:00', 'Z')}.tsv"
            )
        )
        latest_failed_links_report = links_error_dir / "failed_links_latest.tsv"
        write_failed_links_report(
            failed_links_report,
            failed_records,
            removed_urls,
            run_utc=run_utc,
        )
        if latest_failed_links_report != failed_links_report:
            write_failed_links_report(
                latest_failed_links_report,
                failed_records,
                removed_urls,
                run_utc=run_utc,
            )
        print(f"[REPORT] Issue report: {failed_links_report}")

    print(
        "\nDone. "
        f"{successes} downloaded, "
        f"{skipped} skipped(existing), "
        f"{retry_failures} retry-needed, "
        f"{no_data_failures} no-data. Audio files in: {out_dir}"
    )
    print(f"URL/audio index: {index_path}")
    return 0 if (successes + skipped) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
