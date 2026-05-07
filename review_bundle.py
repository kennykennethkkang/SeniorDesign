#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "audio_in"
DEFAULT_LABEL_PROJECT = "uploaded-site-training"
# Bump this whenever the generated review HTML's JS/markup changes in a way
# that needs old bundles to be regenerated. The dashboard checks this against
# a meta tag in existing HTMLs and rewrites stale ones the next time the row
# is opened, so users picking up bug fixes don't have to manually rerun.
REVIEW_BUNDLE_FORMAT_VERSION = 10
TRAINING_BACKEND_LABELS = {
    "nemo": "NeMo",
    "pyannote": "pyannote",
}
MEDIA_EXTENSIONS = {
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
    ".mov",
}
SPEAKER_PATTERN = re.compile(r"^(Speaker\s+\d+):\s*(.*)$")


def project_slug(value: object) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in str(value or "").strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "project"


def build_training_target_payload(
    fine_tuning_projects: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Serialize existing fine-tuned runs for the completion dialog."""

    targets: list[dict[str, object]] = []
    seen: set[str] = set()

    def add_target(
        *,
        backend: object,
        project_name: object,
        display_name: object = "",
        sample_count: object = 0,
        prepared: object = False,
        auto_train: object = False,
        latest_run_text: object = "",
        kind: str = "fine_tuned",
        choice_key: object = "",
        run_name: object = "",
        version_name: object = "",
        display_status: object = "",
    ) -> None:
        normalized_backend = str(backend or "").strip().lower()
        if normalized_backend not in TRAINING_BACKEND_LABELS:
            return
        slug = project_slug(project_name)
        key = f"{normalized_backend}/{slug}"
        choice_id = str(choice_key or f"{key}::{run_name or version_name or kind}").strip()
        if not choice_id:
            choice_id = key
        if choice_id in seen:
            return
        seen.add(choice_id)
        targets.append(
            {
                "key": key,
                "choiceKey": choice_id,
                "backend": normalized_backend,
                "projectName": slug,
                "displayName": str(display_name or "").strip() or slug,
                "backendLabel": TRAINING_BACKEND_LABELS[normalized_backend],
                "sampleCount": int(sample_count or 0),
                "prepared": bool(prepared),
                "autoTrain": bool(auto_train),
                "latestRunText": str(latest_run_text or ""),
                "runName": str(run_name or ""),
                "versionName": str(version_name or ""),
                "displayStatus": str(display_status or ""),
                "kind": kind,
            }
        )

    for project in fine_tuning_projects or []:
        project_runs = project.get("recent_runs")
        runs = list(project_runs) if isinstance(project_runs, (list, tuple)) else []
        latest_run = project.get("latest_run") if isinstance(project.get("latest_run"), Mapping) else {}
        if latest_run and not runs:
            runs = [latest_run]
        for run in runs:
            if not isinstance(run, Mapping):
                continue
            run_dir_name = Path(str(run.get("run_dir", ""))).name
            version_name = str(run.get("version_name") or run_dir_name or "").strip()
            display_name = str(
                run.get("display_name")
                or version_name
                or run_dir_name
                or project.get("display_name")
                or project.get("slug")
                or ""
            ).strip()
            status = str(run.get("status") or "").strip()
            latest_run_text = " ".join(part for part in [status, version_name or run_dir_name] if part).strip()
            add_target(
                backend=project.get("backend"),
                project_name=project.get("slug"),
                display_name=display_name,
                sample_count=project.get("sample_count"),
                prepared=project.get("prepared"),
                auto_train=project.get("auto_train"),
                latest_run_text=latest_run_text,
                kind="fine_tuned",
                choice_key=f"{project.get('backend')}/{project.get('slug')}::{run_dir_name or version_name}",
                run_name=run_dir_name,
                version_name=version_name,
                display_status=status,
            )

    return targets


@dataclass(frozen=True)
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(self.end_ms - self.start_ms, 0)

    @property
    def speaker(self) -> str:
        match = SPEAKER_PATTERN.match(self.text)
        return match.group(1) if match else ""

    @property
    def spoken_text(self) -> str:
        match = SPEAKER_PATTERN.match(self.text)
        if match:
            return match.group(2).strip()
        return self.text.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an HTML review page and TSV flags from a diarization SRT."
    )
    parser.add_argument("--srt", required=True, help="Path to the diarization SRT file.")
    parser.add_argument(
        "--media",
        default=None,
        help="Optional audio/video file to compare against. If omitted, a matching file is searched for.",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(DEFAULT_MEDIA_DIR),
        help="Directory to search when --media is omitted.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output HTML path. Defaults to <srt_stem>_review.html next to the SRT.",
    )
    parser.add_argument(
        "--report-tsv",
        default=None,
        help="Output TSV path. Defaults to <srt_stem>_review_flags.tsv next to the SRT.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress status prints.",
    )
    return parser


def parse_srt_timestamp(value: str) -> int:
    hours, minutes, seconds_millis = value.split(":")
    seconds, millis = seconds_millis.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )


def format_ms(value: int) -> str:
    total_seconds, millis = divmod(max(value, 0), 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def parse_srt(srt_path: Path) -> list[Cue]:
    raw_text = srt_path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw_text.strip())
    cues: list[Cue] = []

    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or "-->" not in lines[1]:
            continue

        try:
            index = int(lines[0])
        except ValueError:
            continue

        start_text, end_text = [part.strip() for part in lines[1].split("-->", maxsplit=1)]
        cue_text = " ".join(line.strip() for line in lines[2:]).strip()
        cues.append(
            Cue(
                index=index,
                start_ms=parse_srt_timestamp(start_text),
                end_ms=parse_srt_timestamp(end_text),
                text=cue_text,
            )
        )

    return cues


def normalize_text(value: str) -> str:
    cleaned = SPEAKER_PATTERN.sub(r"\2", value).lower()
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def detect_flags(cues: Sequence[Cue]) -> dict[int, list[str]]:
    flags_by_index: dict[int, list[str]] = {}

    for position, cue in enumerate(cues):
        flags: list[str] = []
        normalized_text = normalize_text(cue.text)
        word_count = len(cue.spoken_text.split())

        if cue.duration_ms < 350:
            flags.append("very_short_segment")
        if not cue.speaker:
            flags.append("missing_speaker_prefix")
        if word_count >= 10 and cue.duration_ms < 1500:
            flags.append("dense_caption")

        if position > 0:
            previous = cues[position - 1]
            if cue.start_ms < previous.end_ms:
                flags.append("overlap_previous")
            if normalized_text and normalized_text == normalize_text(previous.text):
                flags.append("duplicate_adjacent_text")

        if position + 1 < len(cues):
            following = cues[position + 1]
            if cue.end_ms > following.start_ms:
                flags.append("overlap_next")

        flags_by_index[cue.index] = flags

    return flags_by_index


_NUMERIC_PREFIX = re.compile(r"^\d{2,4}_")
_YOUTUBE_ID = re.compile(r"_([A-Za-z0-9_-]{11})$")


def _strip_numeric_prefix(stem: str) -> str:
    """Drop the audio_numbering ``NNN_`` prefix so the descriptive part is comparable."""

    return _NUMERIC_PREFIX.sub("", stem, count=1)


def _trailing_youtube_id(stem: str) -> str | None:
    """Return the trailing 11-character YouTube ID if the stem ends with one."""

    match = _YOUTUBE_ID.search(stem)
    return match.group(1) if match else None


def _stem_matches_srt(srt_stem: str, candidate_stem: str) -> bool:
    """Decide whether a media file's stem matches a diarization SRT.

    Matching has to tolerate three workspace shapes that have all been valid
    at different points in the project's life:

    1. **Exact match** — both files share the same stem (e.g. SRT and WAV
       were written side-by-side in the same directory).
    2. **Folder-prefixed stem** — the dashboard names diarization artifacts
       by joining the audio file's relative path components with ``__``,
       so an audio file at ``audio_in/youtube_links/001_clip.wav`` produces
       ``youtube_links__001_clip.srt``. We accept ``srt_stem ==
       <folder>__<candidate_stem>`` for the simple case and also strip the
       ``NNN_`` numeric prefix on both sides so a re-downloaded audio file
       (which might land at ``001_<title>`` instead of ``002_<title>``) still
       matches its older transcript.
    3. **YouTube ID match** — both stems end in ``_<11-char-video-id>``.
       This is the bulletproof fallback for YouTube-sourced media: the user
       can wipe and redownload an audio file with a brand new numeric prefix
       and the SRT will still find it as long as the YouTube ID is intact.
    """

    if srt_stem == candidate_stem:
        return True
    if srt_stem.endswith(f"__{candidate_stem}"):
        return True

    # Numeric-prefix-tolerant comparison. Use the trailing component of the
    # SRT stem after the last "__" (so folder-prefixed stems collapse first).
    srt_basename = srt_stem.rsplit("__", 1)[-1]
    if _strip_numeric_prefix(srt_basename) == _strip_numeric_prefix(candidate_stem):
        return True

    # Last resort: same YouTube ID. Avoids false positives from short titles.
    srt_video_id = _trailing_youtube_id(srt_stem)
    candidate_video_id = _trailing_youtube_id(candidate_stem)
    if srt_video_id and candidate_video_id and srt_video_id == candidate_video_id:
        return True

    return False


def find_matching_media(
    *,
    srt_path: Path,
    explicit_media: Path | None,
    audio_dir: Path,
) -> Path | None:
    if explicit_media is not None:
        media = explicit_media.expanduser().resolve()
        return media if media.is_file() else None

    srt_stem = srt_path.stem
    # First-pass search: the SRT's own folder (some legacy runs co-located the
    # WAV with the transcripts). Cheap, exact-or-folder-prefix match.
    srt_parent = srt_path.parent.resolve()
    if srt_parent.is_dir():
        for candidate in sorted(srt_parent.iterdir()):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            if _stem_matches_srt(srt_stem, candidate.stem):
                return candidate.resolve()

    # Second-pass: walk audio_dir recursively. Necessary because the dashboard
    # routes new media into ``audio_in/youtube_links/`` and ``audio_in/file_uploads/``,
    # not the top level. The previous iterdir-only scan missed every subfolder.
    audio_dir = audio_dir.expanduser().resolve()
    if not audio_dir.is_dir():
        return None
    for candidate in sorted(audio_dir.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        if _stem_matches_srt(srt_stem, candidate.stem):
            return candidate.resolve()
    return None


def write_flag_report(
    *,
    cues: Sequence[Cue],
    flags_by_index: dict[int, list[str]],
    report_tsv: Path,
) -> None:
    report_tsv.parent.mkdir(parents=True, exist_ok=True)
    with report_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cue_index",
                "start",
                "end",
                "duration_ms",
                "speaker",
                "flags",
                "text",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for cue in cues:
            writer.writerow(
                {
                    "cue_index": cue.index,
                    "start": format_ms(cue.start_ms),
                    "end": format_ms(cue.end_ms),
                    "duration_ms": cue.duration_ms,
                    "speaker": cue.speaker or "",
                    "flags": ",".join(flags_by_index.get(cue.index, [])),
                    "text": cue.text,
                }
            )


def build_summary(flags_by_index: dict[int, list[str]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for flags in flags_by_index.values():
        for flag in flags:
            counts[flag] = counts.get(flag, 0) + 1
    return sorted(counts.items())


def media_tag_name(media_path: Path | None) -> str:
    if media_path and media_path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
        return "video"
    return "audio"


def media_href_for_review(media_path: Path, output_html: Path) -> str:
    """Return a browser-safe relative URL from the review page to its media."""

    relative = os.path.relpath(media_path, output_html.parent).replace(os.sep, "/")
    return html.escape(quote(relative, safe="/._-~"))


def review_href_for_path(path: Path, output_html: Path) -> str:
    """Return a browser-safe relative URL from the review page to another artifact."""

    relative = os.path.relpath(path, output_html.parent).replace(os.sep, "/")
    return quote(relative, safe="/._-~")


def format_seconds(value_ms: int) -> str:
    return f"{value_ms / 1000:.3f}"


def format_seconds_value(value: float) -> str:
    return f"{value:.3f}"


def parse_time_value(value: str) -> float:
    raw = value.strip()
    if ":" not in raw:
        return float(raw)
    parts = [float(part.strip()) for part in raw.split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"Unsupported timestamp: {value}")


def record_text(
    label_record: Mapping[str, object] | None,
    key: str,
    default: str = "",
) -> str:
    if not label_record:
        return default
    value = label_record.get(key)
    if value is None:
        return default
    text = str(value)
    return text if text else default


def normalized_label_backend(label_record: Mapping[str, object] | None) -> str:
    backend = record_text(label_record, "backend", "both").strip().lower()
    if backend in {"both", "nemo", "pyannote"}:
        return backend
    raw_targets = label_record.get("target_backends") if label_record else None
    targets = [str(item).strip().lower() for item in raw_targets or [] if str(item).strip()]
    if "nemo" in targets and "pyannote" in targets:
        return "both"
    if targets:
        return targets[0]
    return "both"


def selected_attr(value: str, selected: str) -> str:
    return " selected" if value == selected else ""


def training_speaker_label(raw_value: str, fallback_index: int) -> str:
    """Return the speaker token shown in label rows and written to RTTM."""

    raw_label = raw_value.strip() or f"Speaker {fallback_index}"
    label = "".join(char if char.isalnum() or char in "_-" else "_" for char in raw_label).strip("_-")
    return label or f"SPEAKER_{fallback_index:02d}"


def readable_flag_label(value: str) -> str:
    """Return a short nontechnical label for a detected segment issue."""

    labels = {
        "very_short_segment": "Very short",
        "missing_speaker_prefix": "Missing speaker",
        "dense_caption": "Dense text",
        "overlap_previous": "Overlaps previous",
        "overlap_next": "Overlaps next",
        "duplicate_adjacent_text": "Repeated text",
    }
    return labels.get(value, value.replace("_", " ").title())


def parse_saved_label_segments(raw_segments: str) -> list[tuple[str, str, str]]:
    """Parse saved simple labels or RTTM rows into review form rows."""

    rows: list[tuple[str, str, str]] = []
    for raw_line in raw_segments.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[0].upper() == "SPEAKER":
            try:
                start = float(parts[3])
                end = start + float(parts[4])
            except ValueError:
                continue
            speaker = parts[7]
            rows.append((format_seconds_value(start), format_seconds_value(end), speaker))
            continue
        if len(parts) >= 3:
            try:
                start = parse_time_value(parts[0])
                end = parse_time_value(parts[1])
            except ValueError:
                continue
            speaker = " ".join(parts[2:]).strip()
            if speaker:
                rows.append((format_seconds_value(start), format_seconds_value(end), speaker))
    return rows


def cue_payload(cue: Cue, flags_by_index: dict[int, list[str]]) -> dict[str, object]:
    flags = flags_by_index.get(cue.index, [])
    speaker_label = training_speaker_label(cue.speaker, cue.index)
    return {
        "index": cue.index,
        "start": format_seconds(cue.start_ms),
        "end": format_seconds(cue.end_ms),
        "startLabel": format_ms(cue.start_ms),
        "endLabel": format_ms(cue.end_ms),
        "speaker": speaker_label,
        "flags": flags,
        "flagText": ", ".join(flags) or "-",
        "flagDisplay": ", ".join(readable_flag_label(flag) for flag in flags) or "-",
        "text": cue.spoken_text or cue.text,
    }


def build_model_payload(
    *,
    key: str,
    label: str,
    status: str,
    run_name: str,
    srt_path: Path,
    output_html: Path,
    cues: Sequence[Cue],
    flags_by_index: dict[int, list[str]],
    review_path: Path | None = None,
) -> dict[str, object]:
    summary_items = build_summary(flags_by_index)
    return {
        "key": key,
        "label": label,
        "status": status,
        "runName": run_name,
        "srtHref": review_href_for_path(srt_path, output_html) if srt_path.is_file() else "",
        "reviewHref": review_href_for_path(review_path, output_html) if review_path and review_path.is_file() else "",
        "summary": [
            {"flag": flag, "label": readable_flag_label(flag), "count": count}
            for flag, count in summary_items
        ],
        "speakers": sorted(
            {
                str(item["speaker"])
                for item in (cue_payload(cue, flags_by_index) for cue in cues)
                if str(item["speaker"]).strip()
            }
        ),
        "flags": sorted({flag for flags in flags_by_index.values() for flag in flags}),
        "cues": [cue_payload(cue, flags_by_index) for cue in cues],
    }


def build_model_comparison_payload(
    *,
    primary_cues: Sequence[Cue],
    primary_flags_by_index: dict[int, list[str]],
    primary_srt_path: Path,
    output_html: Path,
    model_comparisons: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    primary_payload: dict[str, object] | None = None

    for item in model_comparisons or []:
        raw_path = item.get("srt_path")
        if not isinstance(raw_path, Path) or not raw_path.is_file():
            continue
        srt_path = raw_path.resolve()
        if srt_path in seen_paths:
            continue
        seen_paths.add(srt_path)
        if srt_path == primary_srt_path.resolve():
            cues = list(primary_cues)
            flags_by_index = primary_flags_by_index
        else:
            try:
                cues = parse_srt(srt_path)
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            flags_by_index = detect_flags(cues)
        review_path = item.get("review_path")
        payload = build_model_payload(
            key=str(item.get("key") or srt_path.stem),
            label=str(item.get("label") or srt_path.stem),
            status=str(item.get("status") or "unknown"),
            run_name=str(item.get("run_name") or ""),
            srt_path=srt_path,
            output_html=output_html,
            cues=cues,
            flags_by_index=flags_by_index,
            review_path=review_path if isinstance(review_path, Path) else None,
        )
        if srt_path == primary_srt_path.resolve():
            primary_payload = payload
        else:
            payloads.append(payload)

    if primary_payload is None:
        primary_payload = build_model_payload(
            key=primary_srt_path.stem,
            label="Current review",
            status="loaded",
            run_name=primary_srt_path.parent.name,
            srt_path=primary_srt_path,
            output_html=output_html,
            cues=primary_cues,
            flags_by_index=primary_flags_by_index,
            review_path=output_html,
        )
    return [primary_payload, *payloads]


def build_html(
    *,
    cues: Sequence[Cue],
    flags_by_index: dict[int, list[str]],
    srt_path: Path,
    media_path: Path | None,
    output_html: Path,
    media_selection_name: str = "",
    label_record: Mapping[str, object] | None = None,
    model_comparisons: Sequence[Mapping[str, object]] | None = None,
    fine_tuning_projects: Sequence[Mapping[str, object]] | None = None,
) -> str:
    media_href = media_href_for_review(media_path, output_html) if media_path is not None else ""
    media_name = media_path.name if media_path is not None else "No matching media found"
    media_label = html.escape(media_name)
    summary_items = build_summary(flags_by_index)
    summary_html = "".join(
        f"<span class=\"summary-chip\"><strong>{count}</strong> {html.escape(readable_flag_label(flag))}</span>"
        for flag, count in summary_items
    ) or "<span class=\"summary-chip\">No segment issues found</span>"

    cue_rows_html: list[str] = []
    default_label_entries: list[tuple[str, str, str, str, str]] = []
    for cue in cues:
        flags = flags_by_index.get(cue.index, [])
        flag_text = ", ".join(flags) or "-"
        flag_display_text = ", ".join(readable_flag_label(flag) for flag in flags) or "-"
        row_class = "flagged" if flags else ""
        speaker_label = training_speaker_label(cue.speaker, cue.index)
        start_seconds = format_seconds(cue.start_ms)
        end_seconds = format_seconds(cue.end_ms)
        spoken_text = cue.spoken_text
        default_label_entries.append((str(cue.index), start_seconds, end_seconds, speaker_label, spoken_text))
        cue_rows_html.append(
            "<tr "
            f"class=\"{row_class}\" "
            "data-cue-row "
            "tabindex=\"0\" "
            f"data-index=\"{cue.index}\" "
            f"data-start=\"{start_seconds}\" "
            f"data-end=\"{end_seconds}\" "
            f"data-speaker=\"{html.escape(speaker_label, quote=True)}\" "
            f"data-flags=\"{html.escape(flag_text, quote=True)}\" "
            f"data-text=\"{html.escape(spoken_text or cue.text, quote=True)}\""
            ">"
            f"<td class=\"row-index\">{cue.index}</td>"
            f"<td><div class=\"time-readout\"><span>{html.escape(format_ms(cue.start_ms))}</span><span>{html.escape(format_ms(cue.end_ms))}</span></div></td>"
            f"<td class=\"speaker-readout\">{html.escape(speaker_label)}</td>"
            f"<td class=\"flag-readout\">{html.escape(flag_display_text)}</td>"
            f"<td class=\"action-cell\"><button class=\"play-cue\" type=\"button\" data-start=\"{start_seconds}\" data-end=\"{end_seconds}\">Listen</button></td>"
            "<td class=\"action-cell\"><button class=\"use-cue\" type=\"button\">Add Label</button></td>"
            f"<td class=\"text-cell\">{html.escape(spoken_text or cue.text)}</td>"
            "</tr>"
        )

    saved_segments = parse_saved_label_segments(record_text(label_record, "label_segments", ""))
    saved_dialogue_lines = record_text(label_record, "transcript_text", "").splitlines()
    if saved_segments:
        label_entries = [
            (str(index), start, end, speaker, saved_dialogue_lines[index - 1] if index - 1 < len(saved_dialogue_lines) else "")
            for index, (start, end, speaker) in enumerate(saved_segments, start=1)
        ]
    else:
        # New files used to seed from the diarization model's cues, but I'm
        # making the user type every Start/End by hand instead — that's the
        # whole point of a manually verified training label.
        label_entries = [("1", "", "", "", "")]

    label_rows_html: list[str] = []
    label_segment_lines: list[str] = []
    for index, start, end, speaker, note in label_entries:
        label_segment_lines.append(f"{start} {end} {speaker}")
        label_rows_html.append(
            "<tr data-label-row "
            "tabindex=\"0\" "
            f"data-start=\"{html.escape(start, quote=True)}\" "
            f"data-end=\"{html.escape(end, quote=True)}\" "
            f"data-speaker=\"{html.escape(speaker, quote=True)}\""
            ">"
            "<td class=\"row-index\">"
            "<span class=\"drag-handle\" draggable=\"true\" role=\"button\" tabindex=\"0\""
            " aria-label=\"Drag to reorder this label\" title=\"Drag to reorder\">⋮⋮</span>"
            f"<span class=\"row-number\">{html.escape(index)}</span>"
            "</td>"
            "<td><div class=\"time-edit\">"
            f"<label><span>Start</span><input class=\"label-start\" type=\"text\" inputmode=\"decimal\" value=\"{html.escape(start, quote=True)}\"></label>"
            f"<label><span>End</span><input class=\"label-end\" type=\"text\" inputmode=\"decimal\" value=\"{html.escape(end, quote=True)}\"></label>"
            "</div></td>"
            f"<td><input class=\"label-speaker\" list=\"speakerOptions\" type=\"text\" value=\"{html.escape(speaker, quote=True)}\"></td>"
            f"<td class=\"label-dialogue-cell\"><input class=\"label-dialogue\" type=\"text\" value=\"{html.escape(note, quote=True)}\" placeholder=\"Optional dialogue\"></td>"
            "<td class=\"action-cell\">"
            "<button class=\"play-label\" type=\"button\">Listen</button>"
            "<button class=\"delete-label\" type=\"button\" aria-label=\"Delete this label\">Delete</button>"
            "</td>"
            "</tr>"
        )

    media_tag = media_tag_name(media_path)
    # preload="metadata" lets the browser show duration + start playing on
    # demand without slurping the whole file up front. The old preload="auto"
    # competed with our waveform fetch for bandwidth, so the player took ages
    # to become interactive on the WAVE-mounted audio shares.
    player_html = (
        f'<{media_tag} id="media" controls preload="metadata" src="{media_href}"></{media_tag}>'
        if media_path is not None
        else "<p class=\"warning\">No matching media file was found, so segment playback is disabled.</p>"
    )
    media_file_name = html.escape(media_selection_name or (media_path.name if media_path is not None else ""), quote=True)
    default_segments = html.escape("\n".join(label_segment_lines))
    default_dialogue = html.escape("\n".join(entry[4] for entry in label_entries))
    include_transcript = bool(label_record is not None and label_record.get("include_transcript") is True)
    include_transcript_value = "1" if include_transcript else "0"
    include_transcript_checked = " checked" if include_transcript else ""
    dialogue_toggle_text = "Hide Dialogue" if include_transcript else "Show Dialogue"
    label_table_class = "label-table" + ("" if include_transcript else " hide-dialogue")
    review_workspace_path = html.escape(str(output_html), quote=True)
    selected_backend = normalized_label_backend(label_record)
    selected_create_backend = selected_backend if selected_backend in {"nemo", "pyannote"} else "pyannote"
    project_name = html.escape(record_text(label_record, "project_name", DEFAULT_LABEL_PROJECT), quote=True)
    raw_label_status = record_text(label_record, "status", "not saved").replace("_", " ")
    label_status = html.escape(raw_label_status)
    speaker_values = sorted(
        {
            value
            for value in [
                *(training_speaker_label(cue.speaker, cue.index) for cue in cues),
                *(entry[3] for entry in label_entries if entry[3]),
            ]
            if value and value != "-"
        }
    )
    flag_values = sorted({flag for flags in flags_by_index.values() for flag in flags})
    speaker_options = "".join(
        f"<option value=\"{html.escape(speaker, quote=True)}\">{html.escape(speaker)}</option>"
        for speaker in speaker_values
    )
    speaker_filter_options = "<option value=\"\">All speakers</option>" + speaker_options
    flag_filter_options = "<option value=\"\">All issues</option>" + "".join(
        f"<option value=\"{html.escape(flag, quote=True)}\">{html.escape(readable_flag_label(flag))}</option>"
        for flag in flag_values
    )
    model_payload = build_model_comparison_payload(
        primary_cues=cues,
        primary_flags_by_index=flags_by_index,
        primary_srt_path=srt_path,
        output_html=output_html,
        model_comparisons=model_comparisons,
    )
    model_payload_json = json.dumps(model_payload).replace("</", "<\\/")
    training_target_payload_json = json.dumps(
        {
            "defaultProjectName": DEFAULT_LABEL_PROJECT,
            "targets": build_training_target_payload(fine_tuning_projects),
        }
    ).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="review-bundle-version" content="{REVIEW_BUNDLE_FORMAT_VERSION}">
  <title>Diarization Review | ML Speech Diarization</title>
  <script>
    (function () {{
      try {{
        var key = "ml-speech-diarization-theme";
        var storedTheme = window.localStorage.getItem(key);
        var theme = storedTheme === "dark" || storedTheme === "light"
          ? storedTheme
          : (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
        document.documentElement.setAttribute("data-theme", theme);
      }} catch (_error) {{
        document.documentElement.setAttribute("data-theme", "light");
      }}
    }}());
  </script>
  <style>
    :root {{
      --scu-red: #a32035;
      --bronco-red: #862633;
      --sunrise: #ffb600;
      --agave: #4d8798;
      --agave-dark: #115d6f;
      --stone: #758592;
      --slate: #425766;
      --slate-dark: #2a373f;
      --white: #ffffff;
      --bg: #f7f8fa;
      --bg-ink: #1e252b;
      --panel: rgba(255, 255, 255, 0.94);
      --panel-solid: #ffffff;
      --muted: #66717c;
      --line: #d9dee3;
      --border: #cbd2d9;
      --body-grid-red: rgba(163, 32, 53, 0.08);
      --body-grid-blue: rgba(77, 135, 152, 0.08);
      --body-start: #ffffff;
      --body-end: #eef2f5;
      --accent: var(--scu-red);
      --accent-dark: var(--bronco-red);
      --flag: #fff2dc;
      --current: rgba(255, 182, 0, 0.22);
      --ok: #e4f3eb;
      --ok-ink: #245b39;
      --shadow: 0 16px 38px rgba(42, 55, 63, 0.1);
      --shadow-soft: 0 8px 22px rgba(42, 55, 63, 0.08);
      --radius: 8px;
      --radius-sm: 6px;
      color-scheme: light;
    }}
    :root[data-theme="dark"] {{
      --scu-red: #d14a5f;
      --bronco-red: #8f2a3a;
      --sunrise: #ffc247;
      --agave: #69a9ba;
      --agave-dark: #8fc6d4;
      --stone: #b2bfca;
      --slate: #6f8496;
      --slate-dark: #e7eef4;
      --bg: #0f151c;
      --bg-ink: #f4f8fb;
      --panel: rgba(24, 34, 44, 0.96);
      --panel-solid: #18222c;
      --muted: #c3ced8;
      --line: #3c4f5f;
      --border: #526677;
      --body-grid-red: rgba(209, 74, 95, 0.12);
      --body-grid-blue: rgba(105, 169, 186, 0.1);
      --body-start: #111821;
      --body-end: #0a0f15;
      --accent: #ff9aac;
      --accent-dark: #d14a5f;
      --flag: #3b261c;
      --current: rgba(255, 194, 71, 0.18);
      --ok: #1f3d2f;
      --ok-ink: #bcebcf;
      --shadow: 0 18px 44px rgba(0, 0, 0, 0.38);
      --shadow-soft: 0 10px 28px rgba(0, 0, 0, 0.28);
      color-scheme: dark;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      min-height: 100vh;
      margin: 0;
      color: var(--bg-ink);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      background:
        linear-gradient(90deg, var(--body-grid-red) 0 1px, transparent 1px 100%),
        linear-gradient(180deg, var(--body-grid-blue) 0 1px, transparent 1px 100%),
        linear-gradient(180deg, var(--body-start) 0, var(--bg) 46%, var(--body-end) 100%);
      background-size: 44px 44px, 44px 44px, auto;
    }}
    a {{
      color: var(--accent);
      font-weight: 800;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    button,
    input,
    select,
    textarea {{
      min-width: 0;
      max-width: 100%;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 9px 10px;
      color: var(--bg-ink);
      background: var(--panel-solid);
      font: inherit;
    }}
    button {{
      width: auto;
      border: 0;
      color: var(--white);
      background: linear-gradient(135deg, var(--scu-red), var(--bronco-red));
      font-weight: 800;
      cursor: pointer;
      box-shadow: var(--shadow-soft);
      line-height: 1.12;
      white-space: normal;
    }}
    button.secondary,
    .nav-link {{
      border: 1px solid var(--border);
      background: var(--panel-solid);
      color: var(--slate-dark);
      box-shadow: none;
    }}
    button.primary {{
      background: linear-gradient(135deg, var(--scu-red), var(--bronco-red));
    }}
    button.ghost {{
      border: 1px solid var(--border);
      background: transparent;
      color: var(--slate-dark);
      box-shadow: none;
    }}
    button.ghost.danger {{
      border-color: rgba(163, 32, 53, 0.42);
      color: var(--accent);
    }}
    button.ghost.danger:hover {{
      background: var(--flag);
    }}
    :root[data-theme="dark"] button:not(.secondary),
    :root[data-theme="dark"] button.primary {{
      background: linear-gradient(135deg, #c44255, var(--bronco-red));
      color: var(--white);
    }}
    input,
    select,
    textarea {{
      width: 100%;
    }}
    textarea {{
      min-height: 92px;
      resize: vertical;
    }}
    h1,
    h2 {{
      margin: 0;
      letter-spacing: 0;
      line-height: 1.14;
    }}
    h1 {{
      font-size: clamp(28px, 4vw, 44px);
    }}
    h2 {{
      font-size: 20px;
    }}
    p {{
      line-height: 1.5;
    }}
    .review-shell {{
      width: min(1500px, calc(100% - 28px));
      margin: 0 auto;
      padding: 14px 0 36px;
    }}
    .topbar,
    .player-panel,
    .panel {{
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      padding: 10px 14px;
      margin-bottom: 8px;
      background:
        linear-gradient(135deg, rgba(140, 28, 46, 0.98), rgba(111, 24, 37, 0.96) 48%, rgba(42, 55, 63, 0.98)),
        var(--scu-red);
      color: var(--white);
    }}
    .topbar h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0.01em;
    }}
    .topbar .page-title small {{
      font-size: 12px;
      opacity: 0.85;
    }}
    :root[data-theme="dark"] .topbar {{
      background:
        linear-gradient(135deg, rgba(124, 34, 47, 0.98), rgba(69, 30, 39, 0.96) 48%, rgba(12, 17, 23, 0.98)),
        #451e27;
    }}
    .page-title {{
      display: grid;
      gap: 3px;
      min-width: 0;
    }}
    .page-title small {{
      color: rgba(255, 255, 255, 0.76);
      font-size: 13px;
      font-weight: 800;
    }}
    .top-actions,
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .top-actions .nav-link,
    .top-actions button {{
      min-height: 40px;
      padding: 9px 11px;
    }}
    .player-panel {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 8px 12px;
      margin-bottom: 8px;
    }}
    .source-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      margin-bottom: 4px;
      color: var(--muted);
    }}
    .source-line span {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 7px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--panel-solid);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
    }}
    .source-line strong {{
      min-width: 0;
      color: var(--bg-ink);
      overflow-wrap: anywhere;
    }}
    audio,
    video {{
      width: 100%;
      margin: 0 0 6px;
      border-radius: var(--radius);
      background: #000;
    }}
    .now-playing {{
      display: flex;
      flex-wrap: wrap;
      gap: 2px 10px;
      align-items: baseline;
      padding: 4px 9px;
      border: 1px solid rgba(255, 182, 0, 0.36);
      border-radius: var(--radius-sm);
      background: rgba(255, 182, 0, 0.08);
      color: var(--bg-ink);
      font-weight: 700;
      font-size: 12px;
    }}
    .now-playing small {{
      color: var(--muted);
      font-weight: 600;
      font-size: 11px;
    }}
    .playback-options {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 0 0 6px;
    }}
    /* All the buttons I reach for while labeling -- play/pause, jump back
       two seconds, drop the playhead time into the focused label cell --
       sit in one row right under the audio so I'm not chasing them around
       three different sections of the page. */
    .transport-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 8px 0 6px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--panel-solid);
    }}
    /* Audio cache status row. The whole point of the cache is that after
       the first visit, the file is on this device's IndexedDB and playback
       starts instantly with zero network -- no more WAVE NFS streaming
       lag. The status here tells me whether I'm currently streaming, in
       the middle of a background save, or hitting the cache straight. */
    .audio-cache-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 0 0 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    .audio-cache-status {{
      font-weight: 700;
    }}
    .audio-cache-status.is-cached {{
      color: #2f4a18;
    }}
    .audio-cache-status.is-fetching {{
      color: var(--accent-dark, var(--accent));
    }}
    .audio-cache-status.is-error {{
      color: var(--accent);
    }}
    .audio-cache-button {{
      padding: 5px 10px;
      font-size: 11px;
    }}
    .transport-button {{
      min-width: 96px;
      padding: 9px 14px;
      font-weight: 900;
    }}
    .transport-button.is-playing {{
      background: var(--accent);
      color: var(--white);
    }}
    .mode-chip {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 5px 9px;
      border: 1px solid rgba(77, 135, 152, 0.34);
      border-radius: var(--radius-sm);
      background: rgba(77, 135, 152, 0.1);
      color: var(--bg-ink);
      font-size: 12px;
      font-weight: 900;
    }}
    .waveform-panel {{
      margin: 0 0 6px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--panel-solid);
      overflow: hidden;
    }}
    .waveform-canvas {{
      display: block;
      width: 100%;
      height: 52px;
      cursor: pointer;
      background:
        linear-gradient(180deg, rgba(77, 135, 152, 0.1), rgba(163, 32, 53, 0.04)),
        var(--panel-solid);
    }}
    .waveform-tools {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      padding: 5px 8px;
      border-top: 1px solid var(--line);
    }}
    .waveform-status {{
      min-width: 180px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    .speed-control {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      margin-left: auto;
    }}
    .speed-control span {{
      margin: 0;
      white-space: nowrap;
      font-size: 12px;
      color: var(--muted);
    }}
    .speed-control select {{
      width: 86px;
      min-height: 30px;
      padding: 4px 6px;
      font-size: 12px;
    }}
    .review-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1.12fr);
      gap: 14px;
      align-items: start;
    }}
    .panel {{
      min-width: 0;
      padding: 16px;
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .label-state {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 5px 9px;
      border-radius: var(--radius-sm);
      background: var(--ok);
      color: var(--ok-ink);
      font-size: 13px;
      font-weight: 900;
    }}
    .label-health {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin: 0 0 10px;
    }}
    .health-chip {{
      min-width: 0;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--panel-solid);
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
    }}
    .health-chip strong {{
      display: block;
      color: var(--bg-ink);
      font-size: 18px;
      line-height: 1.1;
    }}
    .health-chip.ready {{
      border-color: rgba(36, 91, 57, 0.32);
      background: var(--ok);
      color: var(--ok-ink);
    }}
    .health-chip.needs-work {{
      border-color: rgba(163, 32, 53, 0.34);
      background: var(--flag);
      color: var(--accent);
    }}
    .manual-label-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 0 0 10px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--panel-solid);
    }}
    .current-time-chip {{
      display: inline-flex;
      align-items: center;
      min-height: 36px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      color: var(--slate-dark);
      background: rgba(77, 135, 152, 0.1);
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      font-weight: 900;
      white-space: nowrap;
    }}
    .summary-chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .summary-chip {{
      display: inline-flex;
      gap: 5px;
      align-items: center;
      padding: 6px 8px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--panel-solid);
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
    }}
    .model-review-strip {{
      display: grid;
      grid-template-columns: minmax(190px, 260px) minmax(0, 1fr);
      gap: 10px;
      align-items: end;
      margin-bottom: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel-solid);
    }}
    .model-summary {{
      min-height: 42px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: rgba(77, 135, 152, 0.1);
      color: var(--bg-ink);
      font-size: 13px;
      font-weight: 800;
      line-height: 1.35;
    }}
    .model-summary strong {{
      display: block;
      color: var(--bg-ink);
      font-size: 14px;
    }}
    .tool-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-bottom: 10px;
    }}
    .label-table-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin: 0 0 8px;
    }}
    .add-segment-btn {{
      min-height: 34px;
      padding: 6px 12px;
      border-radius: var(--radius-sm);
      font-weight: 900;
    }}
    .add-segment-hint {{
      color: var(--muted);
      font-size: 12px;
    }}
    .field-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    .fine-tuned-target-fields {{
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel-solid);
    }}
    .training-target-create h4 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    .training-target-dialog {{
      width: min(760px, calc(100vw - 28px));
      max-height: min(720px, calc(100vh - 28px));
      padding: 0;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel-solid);
      color: var(--ink);
      box-shadow: var(--shadow);
    }}
    .training-target-dialog::backdrop {{
      background: rgba(15, 23, 42, 0.46);
    }}
    .training-target-card {{
      display: grid;
      gap: 14px;
      padding: 18px;
    }}
    .training-target-options {{
      display: grid;
      gap: 8px;
      max-height: min(430px, 56vh);
      overflow: auto;
      padding-right: 2px;
    }}
    .training-target-option {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: start;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--soft);
      cursor: pointer;
    }}
    .training-target-option strong {{
      display: block;
      margin-bottom: 2px;
    }}
    .training-target-option small {{
      display: block;
      color: var(--muted);
      font-weight: 700;
    }}
    .training-target-option.is-selected {{
      border-color: rgba(77, 135, 152, 0.6);
      background: rgba(77, 135, 152, 0.12);
    }}
    .training-target-rename {{
      align-self: center;
      padding: 5px 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--panel-solid);
      color: var(--ink);
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
    }}
    .training-target-rename:hover {{
      border-color: rgba(77, 135, 152, 0.5);
    }}
    .training-target-empty {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
    }}
    .training-target-subtitle {{
      margin: -8px 0 0;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }}
    .training-mode-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px;
    }}
    .training-mode-tile {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px;
      align-items: start;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--soft);
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease;
    }}
    .training-mode-tile.is-selected {{
      border-color: rgba(77, 135, 152, 0.6);
      background: rgba(77, 135, 152, 0.12);
    }}
    .training-mode-tile strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 14px;
    }}
    .training-mode-tile small {{
      display: block;
      color: var(--muted);
      font-weight: 700;
      line-height: 1.4;
    }}
    .training-mode-pane[hidden] {{
      display: none;
    }}
    .training-version-name-row {{
      display: block;
      margin-top: 10px;
    }}
    .training-target-note {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .dialog-actions {{
      justify-content: flex-end;
    }}
    label span {{
      display: block;
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel-solid);
    }}
    .review-table-wrap {{
      max-height: min(620px, 64vh);
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 0;
      table-layout: fixed;
    }}
    .review-table-wrap table {{
      min-width: 680px;
    }}
    th,
    td {{
      padding: 8px 9px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafb;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    :root[data-theme="dark"] th {{
      background: #121b24;
    }}
    tbody tr {{
      cursor: pointer;
    }}
    tbody tr.flagged {{
      background: var(--flag);
    }}
    tbody tr.current {{
      background: var(--current);
      outline: 2px solid rgba(255, 182, 0, 0.72);
      outline-offset: -2px;
    }}
    tbody tr.selected {{
      box-shadow: inset 4px 0 0 var(--accent);
    }}
    tbody tr.needs-time-fix {{
      background: rgba(163, 32, 53, 0.1);
    }}
    tbody tr.missing-speaker {{
      box-shadow: inset 4px 0 0 var(--sunrise);
    }}
    .review-table-wrap .label-table {{
      min-width: 560px;
    }}
    .label-table input {{
      padding: 7px 8px;
    }}
    .label-table.hide-dialogue .label-dialogue-col,
    .label-table.hide-dialogue .label-dialogue-cell {{
      display: none;
    }}
    .row-index {{
      width: 56px;
      color: var(--muted);
      font-weight: 900;
      vertical-align: middle;
      white-space: nowrap;
    }}
    .row-index .row-number {{
      display: inline-block;
      min-width: 18px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .drag-handle {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 22px;
      margin-right: 4px;
      color: var(--muted);
      font-size: 14px;
      letter-spacing: -1px;
      cursor: grab;
      user-select: none;
      border-radius: var(--radius-sm);
    }}
    .drag-handle:hover,
    .drag-handle:focus-visible {{
      color: var(--bg-ink);
      background: rgba(77, 135, 152, 0.12);
      outline: none;
    }}
    .drag-handle:active {{
      cursor: grabbing;
    }}
    [data-label-row].dragging {{
      opacity: 0.55;
    }}
    [data-label-row].drop-above {{
      box-shadow: inset 0 2px 0 0 var(--accent);
    }}
    [data-label-row].drop-below {{
      box-shadow: inset 0 -2px 0 0 var(--accent);
    }}
    .time-readout,
    .time-edit {{
      display: grid;
      gap: 5px;
      min-width: 0;
    }}
    .time-readout span {{
      display: inline-flex;
      width: fit-content;
      padding: 3px 6px;
      border-radius: var(--radius-sm);
      background: rgba(77, 135, 152, 0.1);
      font-variant-numeric: tabular-nums;
      font-weight: 800;
      white-space: nowrap;
    }}
    .time-edit {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .time-edit label {{
      margin: 0;
    }}
    .time-edit label span {{
      margin-bottom: 4px;
      font-size: 11px;
    }}
    .speaker-readout,
    .flag-readout {{
      font-weight: 800;
    }}
    .text-cell {{
      line-height: 1.4;
    }}
    .action-cell {{
      width: 104px;
      text-align: center;
    }}
    .play-cue,
    .play-label,
    .use-cue {{
      width: 100%;
      min-width: 0;
      padding: 8px 9px;
      font-size: 13px;
    }}
    /* Per-row "done" toggle that lives next to Listen / Delete. The auto
       save status pill at the top tells me whether the *file* has any
       saved labels; this is finer-grained -- "I've personally checked
       this single row, leave it alone." Color-shifts the whole row so
       my eye can find the unchecked work fast on long files. */
    .toggle-done-label {{
      width: 100%;
      min-width: 0;
      margin-top: 4px;
      padding: 7px 8px;
      font-size: 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--line);
      background: var(--panel-solid);
      color: var(--bg-ink);
      font-weight: 800;
      cursor: pointer;
    }}
    .toggle-done-label.is-on {{
      background: rgba(85, 113, 46, 0.18);
      color: #2f4a18;
      border-color: rgba(85, 113, 46, 0.55);
    }}
    [data-label-row].is-done > td {{
      background: rgba(85, 113, 46, 0.10);
    }}
    [data-label-row].is-done .row-number::after {{
      content: " ✓";
      color: #2f4a18;
      font-weight: 900;
    }}
    [data-label-row].is-done.current > td {{
      background: rgba(85, 113, 46, 0.20);
    }}
    .label-table .action-cell {{
      width: 132px;
    }}
    .label-table .action-cell button {{
      display: block;
      width: 100%;
    }}
    .label-table .action-cell button + button {{
      margin-top: 5px;
    }}
    .delete-label {{
      width: 100%;
      min-width: 0;
      padding: 6px 8px;
      font-size: 12px;
      color: var(--accent);
      background: transparent;
      border: 1px solid rgba(163, 32, 53, 0.36);
    }}
    .delete-label:hover,
    .delete-label:focus-visible {{
      color: var(--white);
      background: var(--accent);
      border-color: var(--accent);
      outline: none;
    }}
    .filter-row {{
      margin-bottom: 12px;
    }}
    .filter-row input[type="search"] {{
      width: min(380px, 100%);
    }}
    .filter-row label {{
      min-width: 150px;
      flex: 1 1 170px;
    }}
    .checkbox-row {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-weight: 800;
    }}
    .checkbox-row input {{
      width: auto;
    }}
    .save-status {{
      display: none;
      margin: 0 0 14px;
      border: 1px solid #afcfbb;
      background: var(--ok);
      color: var(--ok-ink);
      border-radius: var(--radius);
      padding: 10px 12px;
      font-weight: 800;
    }}
    .save-status.error {{
      border-color: rgba(163, 32, 53, 0.42);
      background: var(--flag);
      color: var(--accent);
    }}
    .warning {{
      color: var(--accent);
      font-weight: 800;
    }}
    .form-actions {{
      margin-top: 12px;
    }}
    @media (max-width: 980px) {{
      .topbar {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .review-grid {{
        grid-template-columns: 1fr;
      }}
      .player-panel {{
        position: static;
      }}
    }}
    @media (max-width: 640px) {{
      .review-shell {{
        width: min(100% - 18px, 1500px);
        padding-top: 10px;
      }}
      .panel,
      .player-panel,
      .topbar {{
        padding: 12px;
      }}
      th,
      td {{
        padding: 7px;
        font-size: 14px;
      }}
      .time-edit {{
        grid-template-columns: 1fr;
      }}
      .label-health {{
        grid-template-columns: 1fr;
      }}
      .model-review-strip {{
        grid-template-columns: 1fr;
      }}
      .action-cell {{
        width: 82px;
      }}
      .play-cue,
      .play-label,
      .use-cue {{
        min-width: 48px;
        padding: 7px 8px;
      }}
    }}
  </style>
</head>
<body>
  <main class="review-shell">
    <header class="topbar">
      <div class="page-title">
        <h1>ML Speech Diarization</h1>
        <small>Diarization review</small>
      </div>
      <nav class="top-actions" aria-label="Review navigation">
        <button id="backButton" class="secondary" type="button">Back</button>
        <a id="dashboardLink" class="nav-link" href="#">Dashboard</a>
        <a id="trainingLabelsLink" class="nav-link" href="#">Labels</a>
        <a id="fineTuningLink" class="nav-link" href="#">Fine-Tuning</a>
        <button id="themeToggle" class="secondary" type="button">Theme</button>
      </nav>
    </header>

    <p id="saveStatus" class="save-status"></p>

    <section class="player-panel" aria-label="Audio playback">
      <div class="source-line"><span>Audio</span><strong>{media_label}</strong></div>
      {player_html}
      <div class="audio-cache-row" aria-live="polite">
        <span id="audioCacheStatus" class="audio-cache-status">Local cache: checking...</span>
        <button id="cacheAudioButton" class="secondary audio-cache-button" type="button" hidden>Cache audio</button>
        <button id="cacheAllAudioButton" class="secondary audio-cache-button" type="button">Cache all audio</button>
        <button id="clearAudioCacheButton" class="ghost audio-cache-button" type="button" hidden>Clear cached audio</button>
      </div>
      <div class="waveform-panel">
        <canvas id="waveformCanvas" class="waveform-canvas" aria-label="Audio waveform"></canvas>
        <div class="waveform-tools">
          <span id="waveformStatus" class="waveform-status">Waveform loading...</span>
          <button id="loadWaveformButton" class="secondary" type="button" hidden>Reload waveform</button>
        </div>
      </div>
      <div class="playback-options">
        <input id="startAtSegment" type="checkbox" checked hidden aria-hidden="true">
        <span class="mode-chip">Listen starts at the selected segment</span>
        <label class="checkbox-row"><input id="stopAtSegmentEnd" type="checkbox" checked> Stop at segment end</label>
        <label class="speed-control" for="playbackRate"><span>Speed</span><select id="playbackRate"><option value="0.5">0.5x</option><option value="0.75">0.75x</option><option value="1" selected>1x</option><option value="1.25">1.25x</option><option value="1.5">1.5x</option><option value="2">2x</option></select></label>
      </div>
      <div id="nowPlaying" class="now-playing">
        <span>Ready to review.</span>
        <small>Click Listen to play a segment. Selecting rows only highlights them.</small>
      </div>
    </section>

    <section class="review-grid">
      <section class="panel">
        <div class="panel-head">
          <h2>Training Labels</h2>
          <span class="label-state">Status: {label_status}</span>
        </div>
        <form id="labelForm" method="post">
          <input type="hidden" name="audio_file" value="{media_file_name}">
          <input type="hidden" id="labelReturnTo" name="label_return_to" value="">
          <input type="hidden" name="label_source" value="review_page">
          <input type="hidden" name="label_review_path" value="{review_workspace_path}">
          <input type="hidden" id="labelIncludeTranscript" name="label_include_transcript" value="{include_transcript_value}">
          <textarea id="labelSegments" name="label_segments" hidden>{default_segments}</textarea>
          <textarea id="labelTranscript" name="label_transcript_text" hidden>{default_dialogue}</textarea>
          <div id="trainingTargetFields" hidden></div>
          <div id="labelHealth" class="label-health" aria-live="polite">
            <span class="health-chip"><strong>0</strong> usable labels</span>
            <span class="health-chip"><strong>0</strong> speakers</span>
            <span class="health-chip needs-work"><strong>Check</strong> ready state</span>
          </div>
          <div class="manual-label-toolbar" role="toolbar" aria-label="Manual label controls">
            <button id="playPauseButton" class="primary transport-button" type="button" aria-pressed="false">Play</button>
            <button id="rewind" class="secondary transport-button" type="button">Back 2 sec</button>
            <span id="currentTimeReadout" class="current-time-chip" role="status">Current: 0.000s</span>
            <button id="setLabelTimeFromPlayer" class="secondary transport-button" type="button">Use Current Seconds</button>
            <button id="addLabelAtCurrentTime" class="secondary transport-button" type="button" title="Create a fresh label row with start time set to wherever the audio is right now.">Add label @ current time</button>
            <label class="checkbox-row label-dialogue-toggle"><input id="showLabelDialogue" type="checkbox"{include_transcript_checked}> <span id="labelDialogueToggleText">{dialogue_toggle_text}</span></label>
          </div>
          <datalist id="speakerOptions">{speaker_options}</datalist>
          <div class="tool-grid">
            <label><span>Search labels</span><input id="labelSearch" type="search" placeholder="Speaker, dialogue, or time"></label>
            <label><span>Speaker</span><select id="labelSpeakerFilter">{speaker_filter_options}</select></label>
            <label><span>Label status</span><select id="labelIssueFilter"><option value="">All labels</option><option value="needs_time">Needs time fix</option><option value="missing_speaker">Missing speaker</option></select></label>
          </div>
          <div class="label-table-toolbar">
            <button type="button" id="addLabelRowTop" class="add-segment-btn">Add Label</button>
            <span class="add-segment-hint">Appends a blank label as the next number. Drag the handle to reorder.</span>
          </div>
          <div class="table-wrap review-table-wrap">
            <table class="{label_table_class}">
              <colgroup>
                <col style="width: 56px">
                <col style="width: 178px">
                <col style="width: 20%">
                <col class="label-dialogue-col" style="width: 26%">
                <col style="width: 132px">
              </colgroup>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Start / End</th>
                  <th>Speaker</th>
                  <th class="label-dialogue-col">Dialogue</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="labelRows">
                {"".join(label_rows_html)}
              </tbody>
            </table>
          </div>
          <div class="controls form-actions">
            <button type="button" id="addLabelRow">Add Label</button>
            <button class="secondary" type="submit" name="label_action" value="draft">Save Draft</button>
            <button class="primary" type="submit" name="label_action" value="complete">Complete For Training</button>
            <button type="button" id="uncompleteLabelButton" class="ghost danger" hidden title="Remove this sample from the fine-tuning project's training set and roll the label back to a draft.">Remove From Training</button>
          </div>
        </form>
        <dialog id="trainingTargetDialog" class="training-target-dialog" aria-labelledby="trainingTargetTitle">
          <div class="training-target-card">
            <div class="panel-head">
              <h3 id="trainingTargetTitle">Train This Sample</h3>
              <button type="button" id="closeTrainingTargetDialog" class="secondary">Cancel</button>
            </div>
            <p class="training-target-subtitle">Choose how this completed label should be used to train a diarization model.</p>
            <div class="training-mode-grid" role="radiogroup" aria-label="Training mode">
              <label class="training-mode-tile" data-training-mode-tile="existing">
                <input type="radio" name="trainingModeChoice" value="existing">
                <div>
                  <strong>Continue training an existing fine-tuned model</strong>
                  <small>Adds this label to a project's training set and starts a new run that builds on the previous samples.</small>
                </div>
              </label>
              <label class="training-mode-tile" data-training-mode-tile="base">
                <input type="radio" name="trainingModeChoice" value="base">
                <div>
                  <strong>Train a new model from the default base</strong>
                  <small>Creates a brand-new fine-tuning project for this label. Name the model — it will be available for future training runs.</small>
                </div>
              </label>
            </div>
            <section id="trainingModeExistingPane" class="training-mode-pane" hidden>
              <h4>Pick one or more fine-tuned models</h4>
              <p class="training-target-note">Tick every model that should receive this label. Each ticked project gets a fresh training run.</p>
              <div id="trainingTargetOptions" class="training-target-options"></div>
              <label class="training-version-name-row">
                <span>New trained version name (optional, applied to every selected model)</span>
                <input id="existingTrainingVersionName" type="text" placeholder="cleaned-stage-2">
              </label>
            </section>
            <section id="trainingModeBasePane" class="training-mode-pane training-target-create" hidden>
              <h4>Create a new fine-tuned model</h4>
              <div class="field-grid fine-tuned-target-fields">
                <label><span>Backend</span><select id="newTrainingBackend"><option value="pyannote"{selected_attr("pyannote", selected_create_backend)}>pyannote</option><option value="nemo"{selected_attr("nemo", selected_create_backend)}>NeMo</option></select></label>
                <label><span>Model name</span><input id="newTrainingProjectName" type="text" value="{project_name}" placeholder="speaker-lab"></label>
                <label><span>New trained version name</span><input id="newTrainingVersionName" type="text" placeholder="cleaned-stage-2"></label>
              </div>
              <p class="training-target-note">Naming the model registers it for future use so more labels can be added to it later.</p>
            </section>
            <div class="controls dialog-actions">
              <button type="button" id="skipTrainingQueue" class="secondary">Save Without Training</button>
              <button type="button" id="confirmTrainingTargets" class="primary">Train Now</button>
            </div>
          </div>
        </dialog>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>Detected Segments</h2>
        </div>
        <div class="model-review-strip" role="region" aria-label="Diarization output comparison">
          <label for="modelComparisonSelect"><span>Diarization output</span><select id="modelComparisonSelect"></select></label>
          <div id="modelComparisonSummary" class="model-summary" aria-live="polite">Loading diarization output details.</div>
        </div>
        <div id="summaryChips" class="summary-chips">{summary_html}</div>
        <div class="controls filter-row">
          <label><span>Search segments</span><input id="search" type="search" placeholder="Transcript, speaker, or flag"></label>
          <label><span>Speaker</span><select id="cueSpeakerFilter">{speaker_filter_options}</select></label>
          <label><span>Flag</span><select id="flagFilter">{flag_filter_options}</select></label>
          <label class="checkbox-row"><input id="flaggedOnly" type="checkbox"> Flagged only</label>
        </div>
        <div class="table-wrap review-table-wrap">
          <table>
            <colgroup>
              <col style="width: 42px">
              <col style="width: 132px">
              <col style="width: 96px">
              <col style="width: 130px">
                <col style="width: 88px">
                <col style="width: 116px">
              <col>
            </colgroup>
            <thead>
              <tr>
                <th>#</th>
                <th>Start / End</th>
                <th>Speaker</th>
                <th>Issues</th>
                <th>Listen</th>
                <th>Add</th>
                <th>Speech</th>
              </tr>
            </thead>
            <tbody id="rows">
              {"".join(cue_rows_html)}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  </main>
  <script id="modelComparisonData" type="application/json">{model_payload_json}</script>
  <script id="trainingTargetData" type="application/json">{training_target_payload_json}</script>
  <script>
    const THEME_STORAGE_KEY = "ml-speech-diarization-theme";
    const media = document.getElementById("media");
    const waveformCanvas = document.getElementById("waveformCanvas");
    const waveformStatus = document.getElementById("waveformStatus");
    const playbackRate = document.getElementById("playbackRate");
    const startAtSegment = document.getElementById("startAtSegment");
    const stopAtSegmentEnd = document.getElementById("stopAtSegmentEnd");
    let cueRows = Array.from(document.querySelectorAll("[data-cue-row]"));
    const cueTable = document.getElementById("rows");
    const modelComparisonDataElement = document.getElementById("modelComparisonData");
    const modelComparisonSelect = document.getElementById("modelComparisonSelect");
    const modelComparisonSummary = document.getElementById("modelComparisonSummary");
    const summaryChips = document.getElementById("summaryChips");
    const search = document.getElementById("search");
    const cueSpeakerFilter = document.getElementById("cueSpeakerFilter");
    const flagFilter = document.getElementById("flagFilter");
    const flaggedOnly = document.getElementById("flaggedOnly");
    const rewind = document.getElementById("rewind");
    const labelForm = document.getElementById("labelForm");
    const labelRows = document.getElementById("labelRows");
    const labelSegments = document.getElementById("labelSegments");
    const labelTranscript = document.getElementById("labelTranscript");
    const labelIncludeTranscript = document.getElementById("labelIncludeTranscript");
    const labelHealth = document.getElementById("labelHealth");
    const labelSearch = document.getElementById("labelSearch");
    const labelSpeakerFilter = document.getElementById("labelSpeakerFilter");
    const labelIssueFilter = document.getElementById("labelIssueFilter");
    const showLabelDialogue = document.getElementById("showLabelDialogue");
    const labelDialogueToggleText = document.getElementById("labelDialogueToggleText");
    const addLabelRow = document.getElementById("addLabelRow");
    const addLabelRowTop = document.getElementById("addLabelRowTop");
    const setLabelTimeFromPlayer = document.getElementById("setLabelTimeFromPlayer");
    const currentTimeReadout = document.getElementById("currentTimeReadout");
    const trainingTargetDataElement = document.getElementById("trainingTargetData");
    const trainingTargetFields = document.getElementById("trainingTargetFields");
    const trainingTargetDialog = document.getElementById("trainingTargetDialog");
    const trainingTargetOptions = document.getElementById("trainingTargetOptions");
    const closeTrainingTargetDialog = document.getElementById("closeTrainingTargetDialog");
    const confirmTrainingTargets = document.getElementById("confirmTrainingTargets");
    const skipTrainingQueue = document.getElementById("skipTrainingQueue");
    const newTrainingBackend = document.getElementById("newTrainingBackend");
    const newTrainingProjectName = document.getElementById("newTrainingProjectName");
    const newTrainingVersionName = document.getElementById("newTrainingVersionName");
    const existingTrainingVersionName = document.getElementById("existingTrainingVersionName");
    const trainingModeExistingPane = document.getElementById("trainingModeExistingPane");
    const trainingModeBasePane = document.getElementById("trainingModeBasePane");
    const trainingModeRadios = document.querySelectorAll("input[name='trainingModeChoice']");
    const trainingModeTiles = document.querySelectorAll(".training-mode-tile");
    let currentTrainingMode = "existing";
    const saveStatus = document.getElementById("saveStatus");
    const labelReturnTo = document.getElementById("labelReturnTo");
    const trainingLabelsLink = document.getElementById("trainingLabelsLink");
    const fineTuningLink = document.getElementById("fineTuningLink");
    const dashboardLink = document.getElementById("dashboardLink");
    const backButton = document.getElementById("backButton");
    const themeToggle = document.getElementById("themeToggle");
    const nowPlaying = document.getElementById("nowPlaying");
    let activeStopHandler = null;
    let activeSeekTimer = null;
    let playRequestId = 0;
    let selectedCueRow = null;
    let selectedLabelRow = null;
    let activeLabelTimeInput = null;
    let activePlaybackRange = null;
    let waveformPeaks = [];
    let waveformLoaded = false;
    let completeSubmitConfirmed = false;
    let labelAutoSaveTimer = null;
    let labelAutoSaveController = null;

    function appBasePath() {{
      const filesIndex = window.location.pathname.indexOf("/files/");
      return filesIndex >= 0 ? window.location.pathname.slice(0, filesIndex) : "";
    }}

    function appLocalPath() {{
      const base = appBasePath();
      const path = window.location.pathname.startsWith(base)
        ? window.location.pathname.slice(base.length)
        : window.location.pathname;
      const params = new URLSearchParams(window.location.search || "");
      params.delete("message");
      params.delete("status");
      const query = params.toString();
      return path + (query ? "?" + query : "");
    }}

    function configureAppLinks() {{
      const base = appBasePath();
      labelForm.action = base + "/training-labels/save";
      labelReturnTo.value = appLocalPath();
      trainingLabelsLink.href = base + "/training-labels";
      fineTuningLink.href = base + "/fine-tuning";
      dashboardLink.href = base + "/";
      backButton.addEventListener("click", function () {{
        // Belt-and-braces: pagehide should already flush the draft, but the
        // Back button is the most common path off this page so flush sync
        // before we navigate. flushLabelAutoSaveOnUnload uses sendBeacon so
        // it survives the location change.
        if (typeof flushLabelAutoSaveOnUnload === "function") {{
          flushLabelAutoSaveOnUnload();
        }}
        window.location.href = base + "/training-labels";
      }});
    }}

    function parseTrainingTargetData() {{
      if (!trainingTargetDataElement) return {{ defaultProjectName: "uploaded-site-training", targets: [] }};
      try {{
        const payload = JSON.parse(trainingTargetDataElement.textContent || "{{}}");
        return {{
          defaultProjectName: payload.defaultProjectName || "uploaded-site-training",
          targets: Array.isArray(payload.targets) ? payload.targets : [],
        }};
      }} catch (_error) {{
        return {{ defaultProjectName: "uploaded-site-training", targets: [] }};
      }}
    }}

    const trainingTargetData = parseTrainingTargetData();

    function trainingProjectSlug(value) {{
      let cleaned = String(value || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-");
      while (cleaned.indexOf("--") >= 0) cleaned = cleaned.replace(/--/g, "-");
      cleaned = cleaned.replace(/^-+|-+$/g, "");
      return cleaned || "project";
    }}

    function trainingBackendLabel(value) {{
      return value === "nemo" ? "NeMo" : "pyannote";
    }}

    function selectedTrainingBackends() {{
      const value = newTrainingBackend ? String(newTrainingBackend.value || "pyannote").toLowerCase() : "pyannote";
      return [value === "nemo" ? "nemo" : "pyannote"];
    }}

    function currentTrainingTargets() {{
      const projectInput = newTrainingProjectName;
      const projectName = trainingProjectSlug(projectInput ? projectInput.value : trainingTargetData.defaultProjectName);
      return selectedTrainingBackends().map(function (backend) {{
        const key = backend + "/" + projectName;
        return {{
          key: key,
          choiceKey: "new::" + key,
          backend: backend,
          projectName: projectName,
          displayName: projectName,
          backendLabel: trainingBackendLabel(backend),
          sampleCount: 0,
          prepared: false,
          autoTrain: false,
          latestRunText: "",
          runName: "",
          versionName: "",
          displayStatus: "",
          kind: "new",
          checked: true,
        }};
      }});
    }}

    function addTrainingTargetChoice(choices, target, checked) {{
      if (!target || !target.key) return;
      const key = String(target.key);
      const choiceKey = String(target.choiceKey || key);
      if (choices.has(choiceKey)) {{
        if (checked) choices.get(choiceKey).checked = true;
        return;
      }}
      choices.set(choiceKey, {{
        key: key,
        choiceKey: choiceKey,
        backend: target.backend || key.split("/")[0],
        projectName: target.projectName || key.split("/").slice(1).join("/"),
        displayName: target.displayName || target.projectName || key,
        backendLabel: target.backendLabel || trainingBackendLabel(target.backend),
        sampleCount: Number(target.sampleCount || 0),
        prepared: Boolean(target.prepared),
        autoTrain: Boolean(target.autoTrain),
        latestRunText: target.latestRunText || "",
        runName: target.runName || "",
        versionName: target.versionName || "",
        displayStatus: target.displayStatus || "",
        kind: target.kind || "existing",
        checked: Boolean(checked),
      }});
    }}

    function trainingTargetChoices() {{
      const choices = new Map();
      currentTrainingTargets().forEach(function (target) {{ addTrainingTargetChoice(choices, target, true); }});
      (trainingTargetData.targets || []).forEach(function (target) {{ addTrainingTargetChoice(choices, target, false); }});
      return Array.from(choices.values());
    }}

    function existingTrainingTargetsByProject() {{
      // Collapse the per-run target list down to one row per project so the
      // popup shows a single picker entry per fine-tuned model. Run-level
      // metadata still rides along on the chosen project — the user just
      // doesn't have to think about which version row to click.
      const byKey = new Map();
      (trainingTargetData.targets || []).forEach(function (target) {{
        if (!target || target.kind === "new") return;
        if (byKey.has(target.key)) return;
        byKey.set(target.key, target);
      }});
      return Array.from(byKey.values()).sort(function (left, right) {{
        const leftLabel = (left.displayName || left.projectName || "").toLowerCase();
        const rightLabel = (right.displayName || right.projectName || "").toLowerCase();
        return leftLabel.localeCompare(rightLabel);
      }});
    }}

    function refreshExistingRowSelection() {{
      // Tile-style "is-selected" highlight follows the checkbox state. Multi-
      // select means we can't just store one row — refresh every time anything
      // toggles so the visuals match exactly which models are queued.
      if (!trainingTargetOptions) return;
      trainingTargetOptions.querySelectorAll(".training-target-option").forEach(function (row) {{
        const box = row.querySelector("input[name='trainingExistingTarget']");
        row.classList.toggle("is-selected", Boolean(box && box.checked));
      }});
    }}

    function promptRenameExistingTarget(target) {{
      // Reuse the dashboard's rename-project endpoint so the popup and the
      // Fine-Tuning tab stay in lock-step on what each model is called.
      const current = target.displayName || target.projectName;
      const next = window.prompt("Rename fine-tuned model \\"" + current + "\\"", current);
      if (next === null) return;
      const cleaned = String(next).trim();
      if (!cleaned || cleaned === current) return;
      const formData = new FormData();
      formData.append("project_slug", target.projectName);
      formData.append("backend", target.backend);
      formData.append("display_name", cleaned);
      window.fetch(appBasePath() + "/fine-tuning/rename-project", {{
        method: "POST",
        body: formData,
        credentials: "same-origin",
        redirect: "follow",
        headers: {{ Accept: "text/html,*/*" }},
      }})
        .then(function (response) {{
          if (!response.ok && response.type !== "opaqueredirect") {{
            window.alert("Could not rename. Check the server log.");
            return;
          }}
          target.displayName = cleaned;
          (trainingTargetData.targets || []).forEach(function (entry) {{
            if (entry && entry.key === target.key) entry.displayName = cleaned;
          }});
          renderTrainingTargetChoices();
        }})
        .catch(function () {{
          window.alert("Rename request failed.");
        }});
    }}

    function renderTrainingTargetChoices() {{
      if (!trainingTargetOptions) return;
      // Preserve whatever the user had ticked across renders triggered by
      // rename actions or the auto-refresh when the new-model name changes.
      const previousKeys = new Set(
        Array.from(trainingTargetOptions.querySelectorAll("input[name='trainingExistingTarget']:checked"))
          .map(function (input) {{ return input.value; }})
      );
      trainingTargetOptions.innerHTML = "";
      const choices = existingTrainingTargetsByProject();
      if (!choices.length) {{
        const note = document.createElement("p");
        note.className = "training-target-empty";
        note.textContent = "No fine-tuned models exist yet. Switch to 'Train a new model from the default base' to create your first one.";
        trainingTargetOptions.appendChild(note);
        return;
      }}
      let firstBox = null;
      let restoredAny = false;
      choices.forEach(function (target) {{
        const row = document.createElement("label");
        row.className = "training-target-option";

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = "trainingExistingTarget";
        checkbox.value = target.key;
        checkbox.dataset.targetKey = target.key;
        checkbox.dataset.projectName = target.projectName;
        checkbox.dataset.backend = target.backend;
        checkbox.dataset.displayName = target.displayName || target.projectName;
        if (previousKeys.has(target.key)) {{
          checkbox.checked = true;
          restoredAny = true;
        }}
        checkbox.addEventListener("change", refreshExistingRowSelection);
        row.appendChild(checkbox);

        const body = document.createElement("div");
        const title = document.createElement("strong");
        title.textContent = target.backendLabel + " / " + (target.displayName || target.projectName);
        body.appendChild(title);

        const meta = document.createElement("small");
        const details = ["previous fine-tuned model"];
        if (target.prepared) details.push("prepared");
        if (target.autoTrain) details.push("auto-train saved");
        details.push(String(target.sampleCount || 0) + " sample(s)");
        if (target.latestRunText) details.push("latest: " + target.latestRunText);
        if (target.projectName && target.projectName !== (target.displayName || "")) details.push("slug: " + target.projectName);
        meta.textContent = details.join(" | ");
        body.appendChild(meta);
        row.appendChild(body);

        const renameButton = document.createElement("button");
        renameButton.type = "button";
        renameButton.className = "training-target-rename";
        renameButton.textContent = "Rename";
        renameButton.title = "Give this fine-tuned model a friendlier name. The on-disk slug stays the same.";
        renameButton.addEventListener("click", function (event) {{
          event.preventDefault();
          event.stopPropagation();
          promptRenameExistingTarget(target);
        }});
        row.appendChild(renameButton);

        trainingTargetOptions.appendChild(row);
        if (!firstBox) firstBox = checkbox;
      }});
      // Default to the first model so a user who just opens the dialog and
      // hits Train Now isn't blocked by an empty-list error. After that the
      // user's explicit selections (preserved across renders) win.
      if (!restoredAny && firstBox) firstBox.checked = true;
      refreshExistingRowSelection();
    }}

    function selectedExistingTargets() {{
      if (!trainingTargetOptions) return [];
      const seen = new Set();
      const targets = [];
      trainingTargetOptions.querySelectorAll("input[name='trainingExistingTarget']:checked").forEach(function (box) {{
        if (seen.has(box.value)) return;
        seen.add(box.value);
        targets.push({{
          key: box.value,
          backend: box.dataset.backend || "",
          projectName: box.dataset.projectName || "",
          displayName: box.dataset.displayName || box.dataset.projectName || "",
        }});
      }});
      return targets;
    }}

    function setTrainingMode(mode) {{
      currentTrainingMode = mode === "base" ? "base" : "existing";
      if (trainingModeExistingPane) trainingModeExistingPane.hidden = currentTrainingMode !== "existing";
      if (trainingModeBasePane) trainingModeBasePane.hidden = currentTrainingMode !== "base";
      trainingModeRadios.forEach(function (radio) {{
        radio.checked = radio.value === currentTrainingMode;
      }});
      trainingModeTiles.forEach(function (tile) {{
        const radio = tile.querySelector("input[name='trainingModeChoice']");
        tile.classList.toggle("is-selected", Boolean(radio && radio.checked));
      }});
    }}

    function selectedDialogTargets() {{
      if (currentTrainingMode === "existing") {{
        return selectedExistingTargets().map(function (target) {{ return target.key; }});
      }}
      return currentTrainingTargets().map(function (target) {{ return target.key; }});
    }}

    function setTrainingTargetHiddenInputs(targets, queueSelected) {{
      if (!trainingTargetFields) return;
      trainingTargetFields.innerHTML = "";
      const values = targets && targets.length ? targets : currentTrainingTargets().map(function (target) {{ return target.key; }});
      const labelByTarget = new Map();
      trainingTargetChoices().forEach(function (target) {{
        if (labelByTarget.has(target.key)) return;
        const label = target.kind === "new"
          ? "Create new " + target.backendLabel + " fine-tuned model / " + (target.displayName || target.projectName)
          : target.backendLabel + " / " + (target.displayName || target.projectName);
        labelByTarget.set(target.key, label);
      }});
      function appendHidden(name, value) {{
        const input = document.createElement("input");
        input.type = "hidden";
        input.name = name;
        input.value = value;
        trainingTargetFields.appendChild(input);
      }}
      values.forEach(function (value) {{
        appendHidden("label_training_targets", value);
      }});
      const firstValue = String(values[0] || "");
      if (firstValue.indexOf("/") > 0) {{
        const firstBackend = firstValue.split("/", 1)[0];
        const firstProject = firstValue.split("/").slice(1).join("/");
        appendHidden("label_backend", firstBackend);
        appendHidden("label_project_name", firstProject);
      }}
      const versionInput = currentTrainingMode === "existing" ? existingTrainingVersionName : newTrainingVersionName;
      const versionName = versionInput ? String(versionInput.value || "").trim() : "";
      if (versionName) {{
        appendHidden("label_new_training_name", versionName);
      }}
      values.forEach(function (value) {{
        appendHidden("label_training_target_label", labelByTarget.get(value) || value);
      }});
      if (queueSelected) {{
        values.forEach(function (value) {{
          appendHidden("label_auto_train_targets", value);
        }});
      }} else {{
        appendHidden("label_auto_train_skip", "1");
      }}
    }}

    function submitCompletedTrainingLabel() {{
      completeSubmitConfirmed = true;
      const completeButton = labelForm.querySelector("button[name='label_action'][value='complete']");
      if (labelForm.requestSubmit && completeButton) {{
        labelForm.requestSubmit(completeButton);
      }} else {{
        if (trainingTargetFields) {{
          const actionInput = document.createElement("input");
          actionInput.type = "hidden";
          actionInput.name = "label_action";
          actionInput.value = "complete";
          trainingTargetFields.appendChild(actionInput);
        }}
        labelForm.submit();
      }}
    }}

    function openTrainingTargetDialog() {{
      // Pick the default mode by what's actually available — if there are no
      // fine-tuned models yet, jump straight to the "create new" pane so the
      // user isn't staring at an empty list wondering what to do.
      const hasExisting = existingTrainingTargetsByProject().length > 0;
      setTrainingMode(hasExisting ? "existing" : "base");
      renderTrainingTargetChoices();
      if (trainingTargetDialog && typeof trainingTargetDialog.showModal === "function") {{
        trainingTargetDialog.showModal();
        return;
      }}
      // Fallback: no <dialog> support. Submit with whichever mode we picked.
      const fallbackTargets = selectedDialogTargets();
      setTrainingTargetHiddenInputs(
        fallbackTargets.length ? fallbackTargets : currentTrainingTargets().map(function (target) {{ return target.key; }}),
        true,
      );
      submitCompletedTrainingLabel();
    }}

    function setTheme(theme, persist) {{
      const normalized = theme === "dark" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", normalized);
      themeToggle.textContent = normalized === "dark" ? "Light Theme" : "Dark Theme";
      if (!persist) return;
      try {{
        window.localStorage.setItem(THEME_STORAGE_KEY, normalized);
      }} catch (_error) {{
      }}
    }}

    function configureThemeToggle() {{
      const current = document.documentElement.getAttribute("data-theme") || "light";
      setTheme(current, false);
      themeToggle.addEventListener("click", function () {{
        setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark", true);
      }});
    }}

    function showSaveStatusFromQuery() {{
      const params = new URLSearchParams(window.location.search);
      const message = params.get("message");
      if (!message) return;
      saveStatus.textContent = message;
      saveStatus.classList.toggle("error", params.get("status") === "error");
      saveStatus.style.display = "block";
    }}

    function parseModelComparisons() {{
      if (!modelComparisonDataElement) return [];
      try {{
        const parsed = JSON.parse(modelComparisonDataElement.textContent || "[]");
        return Array.isArray(parsed) ? parsed : [];
      }} catch (_error) {{
        return [];
      }}
    }}

    const modelComparisons = parseModelComparisons();

    function setSelectOptions(select, values, emptyLabel, labelLookup) {{
      if (!select) return;
      const previous = select.value;
      select.innerHTML = "";
      const emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = emptyLabel;
      select.appendChild(emptyOption);
      const cleanValues = Array.from(new Set((values || []).filter(Boolean))).sort();
      cleanValues.forEach(function (value) {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = labelLookup && labelLookup[value] ? labelLookup[value] : value;
        select.appendChild(option);
      }});
      select.value = cleanValues.includes(previous) ? previous : "";
    }}

    function renderSummaryChips(model) {{
      if (!summaryChips) return;
      summaryChips.innerHTML = "";
      const summary = Array.isArray(model && model.summary) ? model.summary : [];
      if (!summary.length) {{
        const chip = document.createElement("span");
        chip.className = "summary-chip";
        chip.textContent = "No segment issues found";
        summaryChips.appendChild(chip);
        return;
      }}
      summary.forEach(function (item) {{
        const chip = document.createElement("span");
        chip.className = "summary-chip";
        const count = document.createElement("strong");
        count.textContent = String(item.count || 0);
        chip.appendChild(count);
        chip.appendChild(document.createTextNode(" " + (item.label || item.flag || "Issue")));
        summaryChips.appendChild(chip);
      }});
    }}

    function updateModelComparisonSummary(model) {{
      if (!modelComparisonSummary || !model) return;
      const cueCount = Array.isArray(model.cues) ? model.cues.length : 0;
      const speakerCount = Array.isArray(model.speakers) ? model.speakers.length : 0;
      modelComparisonSummary.innerHTML = "";
      const title = document.createElement("strong");
      title.textContent = model.label || "Selected output";
      modelComparisonSummary.appendChild(title);
      const details = [
        cueCount + " segment(s)",
        speakerCount + " speaker(s)",
        model.status ? "status: " + model.status : "",
        model.runName ? "run: " + model.runName : "",
      ].filter(Boolean).join(" | ");
      modelComparisonSummary.appendChild(document.createTextNode(details || "No details available."));
    }}

    function updateSpeakerDatalist(model) {{
      const dataList = document.getElementById("speakerOptions");
      if (!dataList) return;
      const speakers = new Set(Array.isArray(model && model.speakers) ? model.speakers : []);
      labelRowList().forEach(function (row) {{
        if (row.dataset.speaker) speakers.add(row.dataset.speaker);
      }});
      dataList.innerHTML = "";
      Array.from(speakers).filter(Boolean).sort().forEach(function (speaker) {{
        const option = document.createElement("option");
        option.value = speaker;
        option.textContent = speaker;
        dataList.appendChild(option);
      }});
    }}

    function updateLabelSpeakerFilterOptions() {{
      const speakers = labelRowList().map(function (row) {{
        const input = row.querySelector(".label-speaker");
        return input ? input.value.trim() : row.dataset.speaker || "";
      }});
      setSelectOptions(labelSpeakerFilter, speakers, "All speakers");
    }}

    function appendCell(row, className, textValue) {{
      const cell = document.createElement("td");
      if (className) cell.className = className;
      cell.textContent = textValue || "";
      row.appendChild(cell);
      return cell;
    }}

    function buildCueRow(cue) {{
      const flags = Array.isArray(cue.flags) ? cue.flags : [];
      const row = document.createElement("tr");
      if (flags.length) row.className = "flagged";
      row.setAttribute("data-cue-row", "");
      row.tabIndex = 0;
      row.dataset.index = String(cue.index || "");
      row.dataset.start = String(cue.start || "0");
      row.dataset.end = String(cue.end || "0");
      row.dataset.speaker = String(cue.speaker || "");
      row.dataset.flags = flags.length ? flags.join(", ") : "-";
      row.dataset.text = String(cue.text || "");

      appendCell(row, "row-index", row.dataset.index);

      const timeCell = document.createElement("td");
      const timeReadout = document.createElement("div");
      timeReadout.className = "time-readout";
      const start = document.createElement("span");
      start.textContent = cue.startLabel || row.dataset.start;
      const end = document.createElement("span");
      end.textContent = cue.endLabel || row.dataset.end;
      timeReadout.appendChild(start);
      timeReadout.appendChild(end);
      timeCell.appendChild(timeReadout);
      row.appendChild(timeCell);

      appendCell(row, "speaker-readout", row.dataset.speaker);
      appendCell(row, "flag-readout", cue.flagDisplay || row.dataset.flags);

      const playCell = document.createElement("td");
      playCell.className = "action-cell";
      const playButton = document.createElement("button");
      playButton.className = "play-cue";
      playButton.type = "button";
      playButton.dataset.start = row.dataset.start;
      playButton.dataset.end = row.dataset.end;
      playButton.textContent = "Listen";
      playCell.appendChild(playButton);
      row.appendChild(playCell);

      const addCell = document.createElement("td");
      addCell.className = "action-cell";
      const addButton = document.createElement("button");
      addButton.className = "use-cue";
      addButton.type = "button";
      addButton.textContent = "Add Label";
      addCell.appendChild(addButton);
      row.appendChild(addCell);

      appendCell(row, "text-cell", row.dataset.text);
      return row;
    }}

    function applyModelComparison(modelKey) {{
      if (!cueTable || !modelComparisons.length) return;
      const model = modelComparisons.find(function (item) {{ return item.key === modelKey; }}) || modelComparisons[0];
      if (!model) return;
      if (modelComparisonSelect) modelComparisonSelect.value = model.key;
      selectedCueRow = null;
      activePlaybackRange = null;
      cueTable.innerHTML = "";
      (Array.isArray(model.cues) ? model.cues : []).forEach(function (cue) {{
        cueTable.appendChild(buildCueRow(cue));
      }});
      cueRows = Array.from(cueTable.querySelectorAll("[data-cue-row]"));
      const flagLabels = {{}};
      (Array.isArray(model.summary) ? model.summary : []).forEach(function (item) {{
        if (item.flag) flagLabels[item.flag] = item.label || item.flag;
      }});
      setSelectOptions(cueSpeakerFilter, model.speakers || [], "All speakers");
      setSelectOptions(flagFilter, model.flags || [], "All issues", flagLabels);
      renderSummaryChips(model);
      updateModelComparisonSummary(model);
      updateSpeakerDatalist(model);
      applyFilters();
      setCurrentRows();
      drawWaveform();
    }}

    function configureModelComparison() {{
      if (!modelComparisonSelect || !modelComparisons.length) return;
      modelComparisonSelect.innerHTML = "";
      modelComparisons.forEach(function (model) {{
        const option = document.createElement("option");
        option.value = model.key;
        option.textContent = model.label || model.key || "Diarization output";
        modelComparisonSelect.appendChild(option);
      }});
      modelComparisonSelect.disabled = modelComparisons.length < 2;
      modelComparisonSelect.addEventListener("change", function () {{
        applyModelComparison(modelComparisonSelect.value);
      }});
      applyModelComparison(modelComparisonSelect.value || modelComparisons[0].key);
    }}

    function labelRowList() {{
      return Array.from(labelRows.querySelectorAll("[data-label-row]"));
    }}

    function rowNumber(value) {{
      const raw = String(value || "").trim();
      if (!raw) return NaN;
      if (raw.includes(":")) {{
        const parts = raw.split(":").map(function (part) {{ return Number(part.trim()); }});
        if (parts.length === 2 && parts.every(Number.isFinite)) {{
          return parts[0] * 60 + parts[1];
        }}
        if (parts.length === 3 && parts.every(Number.isFinite)) {{
          return parts[0] * 3600 + parts[1] * 60 + parts[2];
        }}
        return NaN;
      }}
      const parsed = Number(raw);
      return Number.isFinite(parsed) ? parsed : NaN;
    }}

    function setPlaybackMessage(title, detail) {{
      nowPlaying.innerHTML = "";
      const mainLine = document.createElement("span");
      mainLine.textContent = title;
      nowPlaying.appendChild(mainLine);
      if (detail) {{
        const detailLine = document.createElement("small");
        detailLine.textContent = detail;
        nowPlaying.appendChild(detailLine);
      }}
    }}

    function clearPlaybackTimers() {{
      if (activeStopHandler) {{
        media.removeEventListener("timeupdate", activeStopHandler);
        activeStopHandler = null;
      }}
      if (activeSeekTimer) {{
        window.clearTimeout(activeSeekTimer);
        activeSeekTimer = null;
      }}
    }}

    function mediaDuration() {{
      if (!media || !Number.isFinite(media.duration) || media.duration <= 0) return 0;
      return media.duration;
    }}

    function drawWaveform() {{
      if (!waveformCanvas) return;
      const ctx = waveformCanvas.getContext("2d");
      if (!ctx) return;
      const rect = waveformCanvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(Math.floor((rect.width || 800) * dpr), 320);
      const height = Math.max(Math.floor((rect.height || 76) * dpr), 64);
      if (waveformCanvas.width !== width || waveformCanvas.height !== height) {{
        waveformCanvas.width = width;
        waveformCanvas.height = height;
      }}
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--panel-solid").trim() || "#fff";
      ctx.fillRect(0, 0, width, height);
      const mid = height / 2;
      ctx.strokeStyle = "rgba(117, 133, 146, 0.32)";
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(width, mid);
      ctx.stroke();

      const duration = mediaDuration();
      if (activePlaybackRange && duration > 0) {{
        const startX = Math.max(activePlaybackRange.start / duration, 0) * width;
        const endX = Math.min(activePlaybackRange.end / duration, 1) * width;
        ctx.fillStyle = "rgba(255, 182, 0, 0.22)";
        ctx.fillRect(startX, 0, Math.max(endX - startX, 2), height);
      }}

      if (!waveformPeaks.length) {{
        ctx.fillStyle = "rgba(117, 133, 146, 0.75)";
        ctx.font = Math.max(12 * dpr, 12) + "px sans-serif";
        ctx.fillText(waveformLoaded ? "No waveform data available" : "Loading waveform", 12 * dpr, mid);
      }} else {{
        const barWidth = Math.max(width / waveformPeaks.length, 1);
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--agave-dark").trim() || "#115d6f";
        ctx.lineWidth = Math.max(1, dpr);
        ctx.beginPath();
        waveformPeaks.forEach(function (peak, index) {{
          const x = index * barWidth;
          const minY = mid + peak.min * mid * 0.86;
          const maxY = mid + peak.max * mid * 0.86;
          ctx.moveTo(x, minY);
          ctx.lineTo(x, maxY);
        }});
        ctx.stroke();
      }}

      if (duration > 0 && media) {{
        const playheadX = Math.max(Math.min(media.currentTime / duration, 1), 0) * width;
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#a32035";
        ctx.lineWidth = Math.max(2, 2 * dpr);
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, height);
        ctx.stroke();
      }}
    }}

    function setActiveRangeFromRow(row) {{
      if (!row) {{
        activePlaybackRange = null;
      }} else {{
        const start = rowNumber(row.dataset.start);
        const end = rowNumber(row.dataset.end);
        activePlaybackRange = Number.isFinite(start) && Number.isFinite(end) && end > start
          ? {{ start: start, end: end }}
          : null;
      }}
      drawWaveform();
    }}

    function waveformPeaksFromBuffer(buffer) {{
      const data = buffer.getChannelData(0);
      const peakCount = Math.min(1400, Math.max(360, Math.floor((waveformCanvas ? waveformCanvas.clientWidth : 900) * 1.3)));
      const samplesPerPeak = Math.max(1, Math.ceil(data.length / peakCount));
      const peaks = [];
      for (let index = 0; index < peakCount; index += 1) {{
        const start = index * samplesPerPeak;
        const end = Math.min(start + samplesPerPeak, data.length);
        let min = 1;
        let max = -1;
        for (let sample = start; sample < end; sample += 1) {{
          const value = data[sample] || 0;
          if (value < min) min = value;
          if (value > max) max = value;
        }}
        peaks.push({{ min: min, max: max }});
      }}
      return peaks;
    }}

    async function loadWaveform() {{
      if (!media || !waveformCanvas || !waveformStatus) {{
        if (waveformStatus) waveformStatus.textContent = "Waveform unavailable.";
        return;
      }}
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass || !window.fetch) {{
        waveformStatus.textContent = "Waveform unavailable in this browser.";
        waveformLoaded = true;
        drawWaveform();
        return;
      }}
      try {{
        const source = media.currentSrc || media.src;
        const response = await fetch(source, {{ cache: "force-cache" }});
        if (!response.ok) throw new Error("media fetch failed");
        const payload = await response.arrayBuffer();
        const context = new AudioContextClass();
        const buffer = await context.decodeAudioData(payload);
        waveformPeaks = waveformPeaksFromBuffer(buffer);
        waveformLoaded = true;
        waveformStatus.textContent = "Waveform ready. Click the wave to jump without playing.";
        if (typeof context.close === "function") context.close();
      }} catch (_error) {{
        waveformPeaks = [];
        waveformLoaded = true;
        waveformStatus.textContent = "Waveform unavailable for this file.";
      }}
      drawWaveform();
    }}

    function safeSpeakerToken(value) {{
      return String(value || "").trim().replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^[_-]+|[_-]+$/g, "");
    }}

    function labelRowBounds(row) {{
      const startInput = row.querySelector(".label-start");
      const endInput = row.querySelector(".label-end");
      return {{
        start: rowNumber(startInput ? startInput.value : row.dataset.start),
        end: rowNumber(endInput ? endInput.value : row.dataset.end),
      }};
    }}

    function rememberLabelTimeInput(input) {{
      if (input && input.matches(".label-start, .label-end")) {{
        activeLabelTimeInput = input;
      }}
    }}

    function activeLabelTimeInputLabel(input) {{
      if (!input) return "time";
      return input.classList.contains("label-start") ? "start" : "end";
    }}

    function setActiveLabelTimeFromPlayer() {{
      if (!media || !Number.isFinite(media.currentTime)) {{
        window.alert("The audio player does not have a current time yet.");
        return;
      }}
      const focusedInput = document.activeElement && document.activeElement.matches(".label-start, .label-end")
        ? document.activeElement
        : null;
      const input = focusedInput || (activeLabelTimeInput && activeLabelTimeInput.isConnected ? activeLabelTimeInput : null);
      if (!input) {{
        window.alert("Click a Start or End time box first.");
        return;
      }}
      const row = input.closest("[data-label-row]");
      const timestamp = formatSeconds(media.currentTime);
      input.value = timestamp;
      if (row) {{
        updateLabelRowDataset(row);
        setSelectedLabelRow(row);
      }}
      syncLabelSegments();
      applyLabelFilters();
      input.focus();
      input.select();
      setPlaybackMessage(
        "Set " + activeLabelTimeInputLabel(input) + " time to " + timestamp + " seconds.",
        row ? labelSummary(row) : ""
      );
      scheduleLabelAutoSave();
    }}

    function updateLabelRowDataset(row) {{
      const bounds = labelRowBounds(row);
      const speakerInput = row.querySelector(".label-speaker");
      const dialogueInput = row.querySelector(".label-dialogue");
      row.dataset.start = Number.isFinite(bounds.start) ? bounds.start.toFixed(3) : "";
      row.dataset.end = Number.isFinite(bounds.end) ? bounds.end.toFixed(3) : "";
      row.dataset.speaker = speakerInput ? speakerInput.value.trim() : "";
      row.dataset.dialogue = dialogueInput ? dialogueInput.value.trim() : "";
      const needsTimeFix = !Number.isFinite(bounds.start) || !Number.isFinite(bounds.end) || bounds.start < 0 || bounds.end <= bounds.start;
      row.classList.toggle("needs-time-fix", needsTimeFix);
      row.classList.toggle("missing-speaker", !row.dataset.speaker);
    }}

    function updateLabelHealth(validCount, speakerCount, invalidCount) {{
      if (!labelHealth) return;
      const ready = validCount > 0 && speakerCount > 0 && invalidCount === 0;
      labelHealth.innerHTML = "";
      [
        [String(validCount), "usable labels", ""],
        [String(speakerCount), "speakers", ""],
        [ready ? "Ready" : String(invalidCount), ready ? "fine-tuning ready" : "row(s) need fixes", ready ? "ready" : "needs-work"],
      ].forEach(function (item) {{
        const chip = document.createElement("span");
        chip.className = "health-chip" + (item[2] ? " " + item[2] : "");
        const strong = document.createElement("strong");
        strong.textContent = item[0];
        chip.appendChild(strong);
        chip.appendChild(document.createTextNode(" " + item[1]));
        labelHealth.appendChild(chip);
      }});
    }}

    function syncLabelSegments(updateHealth = true) {{
      const speakers = new Set();
      let invalidCount = 0;
      const dialogueLines = [];
      const lines = labelRowList().map(function (row) {{
        updateLabelRowDataset(row);
        dialogueLines.push(row.dataset.dialogue || "");
        const bounds = labelRowBounds(row);
        const start = row.dataset.start;
        const end = row.dataset.end;
        const speaker = row.dataset.speaker;
        const speakerToken = safeSpeakerToken(speaker);
        if (Number.isFinite(bounds.start) && Number.isFinite(bounds.end) && bounds.end > bounds.start && speakerToken) {{
          speakers.add(speakerToken);
          return start + " " + end + " " + speaker;
        }}
        invalidCount += 1;
        return "";
      }}).filter(Boolean);
      labelSegments.value = lines.join("\\n");
      if (labelTranscript) labelTranscript.value = dialogueLines.join("\\n").trim();
      if (updateHealth) {{
        updateLabelHealth(lines.length, speakers.size, invalidCount);
        updateLabelSpeakerFilterOptions();
        const activeModel = modelComparisonSelect && modelComparisons.length
          ? modelComparisons.find(function (item) {{ return item.key === modelComparisonSelect.value; }}) || modelComparisons[0]
          : null;
        updateSpeakerDatalist(activeModel);
      }}
    }}

    function setSelectedCueRow(row) {{
      if (selectedCueRow && selectedCueRow !== row) selectedCueRow.classList.remove("selected");
      selectedCueRow = row;
      if (row) row.classList.add("selected");
      setActiveRangeFromRow(row);
    }}

    function setSelectedLabelRow(row) {{
      if (selectedLabelRow && selectedLabelRow !== row) selectedLabelRow.classList.remove("selected");
      selectedLabelRow = row;
      if (row) row.classList.add("selected");
      setActiveRangeFromRow(row);
    }}

    function selectedOrCurrentLabelRow() {{
      if (selectedLabelRow && selectedLabelRow.isConnected && !selectedLabelRow.hidden) return selectedLabelRow;
      const current = labelRowList().find(function (row) {{ return row.classList.contains("current") && !row.hidden; }});
      if (current) {{
        setSelectedLabelRow(current);
        return current;
      }}
      const firstVisible = labelRowList().find(function (row) {{ return !row.hidden; }});
      if (firstVisible) setSelectedLabelRow(firstVisible);
      return firstVisible || null;
    }}

    function formatSeconds(seconds) {{
      return Number.isFinite(seconds) ? Math.max(seconds, 0).toFixed(3) : "0.000";
    }}

    function activeRow(rows, time) {{
      return rows.find(function (row) {{
        const start = rowNumber(row.dataset.start);
        const end = rowNumber(row.dataset.end);
        return Number.isFinite(start) && Number.isFinite(end) && time >= start && time <= end;
      }}) || null;
    }}

    function cueSummary(row) {{
      if (!row) return "Detected segment: none";
      const index = row.dataset.index ? "#" + row.dataset.index + " " : "";
      const speaker = row.dataset.speaker || "unknown speaker";
      const flags = row.dataset.flags && row.dataset.flags !== "-" ? " (" + row.dataset.flags + ")" : "";
      return "Detected segment: " + index + speaker + flags;
    }}

    function labelSummary(row) {{
      if (!row) return "Training label: none";
      const speaker = row.dataset.speaker || "unlabeled speaker";
      return "Training label: " + speaker + " [" + (row.dataset.start || "?") + "-" + (row.dataset.end || "?") + "s]";
    }}

    function selectedRangeSummary(row, start, end) {{
      const speaker = row ? (row.dataset.speaker || "speaker") : "segment";
      return speaker + " | " + formatSeconds(start) + "s to " + formatSeconds(end) + "s";
    }}

    // Track the rows we last marked .current so we can flip just those two
    // instead of touching every row on the page each tick. Used to call
    // foreach across cueRows + labelRows -- with a few thousand rows that was
    // ~6000 classList writes per playback frame, which felt like the page was
    // frozen. Diff-based update is O(1).
    let lastCurrentCue = null;
    let lastCurrentLabel = null;
    let lastNowPlayingTime = null;
    let lastNowPlayingCueIndex = null;
    let lastNowPlayingLabelKey = null;
    function setCurrentRows() {{
      if (!media) return;
      // Important: do NOT walk rows here to refresh their datasets. That used
      // to live at the top of this function, but it ran on every timeupdate
      // (so 60x/sec via rAF) and was the dominant cost on long files. Row
      // datasets are kept in sync by the input/change handlers, which is the
      // only time they actually change.
      const time = media.currentTime;
      updateCurrentTimeReadout();
      const labelRowsNow = labelRowList();
      const currentCue = activeRow(cueRows, time);
      const currentLabel = activeRow(labelRowsNow, time);
      if (currentCue !== lastCurrentCue) {{
        if (lastCurrentCue) lastCurrentCue.classList.remove("current");
        if (currentCue) currentCue.classList.add("current");
        lastCurrentCue = currentCue;
      }}
      if (currentLabel !== lastCurrentLabel) {{
        if (lastCurrentLabel) lastCurrentLabel.classList.remove("current");
        if (currentLabel) currentLabel.classList.add("current");
        lastCurrentLabel = currentLabel;
      }}
      // Avoid rebuilding the now-playing strip every tick if nothing visible
      // would change. innerHTML rewrites force layout; skipping when the
      // underlying values match the previous frame is essentially free.
      const cueIdentity = currentCue ? (currentCue.dataset.index || "") + ":" + (currentCue.dataset.speaker || "") : "";
      const labelIdentity = currentLabel ? (currentLabel.dataset.position || "") + ":" + (currentLabel.dataset.speaker || "") : "";
      const roundedTime = Math.round(time * 10) / 10;
      if (
        roundedTime === lastNowPlayingTime &&
        cueIdentity === lastNowPlayingCueIndex &&
        labelIdentity === lastNowPlayingLabelKey
      ) {{
        return;
      }}
      lastNowPlayingTime = roundedTime;
      lastNowPlayingCueIndex = cueIdentity;
      lastNowPlayingLabelKey = labelIdentity;
      nowPlaying.innerHTML = "";
      const mainLine = document.createElement("span");
      mainLine.textContent = "Time " + formatSeconds(time) + " seconds | " + cueSummary(currentCue);
      const detailLine = document.createElement("small");
      detailLine.textContent = labelSummary(currentLabel);
      nowPlaying.appendChild(mainLine);
      nowPlaying.appendChild(detailLine);
    }}

    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      const speaker = cueSpeakerFilter.value;
      const flag = flagFilter.value;
      const flagged = flaggedOnly.checked;
      cueRows.forEach(function (row) {{
        const text = row.innerText.toLowerCase();
        const matchesQuery = !query || text.includes(query);
        const matchesSpeaker = !speaker || row.dataset.speaker === speaker;
        const matchesSpecificFlag = !flag || (row.dataset.flags || "").split(", ").includes(flag);
        const matchesFlag = !flagged || row.classList.contains("flagged");
        row.hidden = !(matchesQuery && matchesSpeaker && matchesSpecificFlag && matchesFlag);
      }});
    }}

    function applyLabelFilters() {{
      const query = labelSearch.value.trim().toLowerCase();
      const speaker = labelSpeakerFilter.value;
      const issue = labelIssueFilter.value;
      labelRowList().forEach(function (row) {{
        updateLabelRowDataset(row);
        const text = row.innerText.toLowerCase() + " " + (row.dataset.dialogue || "").toLowerCase();
        const matchesQuery = !query || text.includes(query) || (row.dataset.start || "").includes(query) || (row.dataset.end || "").includes(query);
        const matchesSpeaker = !speaker || row.dataset.speaker === speaker;
        const matchesIssue = !issue
          || (issue === "needs_time" && row.classList.contains("needs-time-fix"))
          || (issue === "missing_speaker" && row.classList.contains("missing-speaker"));
        row.hidden = !(matchesQuery && matchesSpeaker && matchesIssue);
      }});
    }}

    function syncLabelDialogueVisibility() {{
      const table = labelRows ? labelRows.closest(".label-table") : null;
      if (!table || !showLabelDialogue) return;
      table.classList.toggle("hide-dialogue", !showLabelDialogue.checked);
      if (labelIncludeTranscript) labelIncludeTranscript.value = showLabelDialogue.checked ? "1" : "0";
      if (labelDialogueToggleText) labelDialogueToggleText.textContent = showLabelDialogue.checked ? "Hide Dialogue" : "Show Dialogue";
    }}

    function updateCurrentTimeReadout() {{
      if (!currentTimeReadout) return;
      const time = media && Number.isFinite(media.currentTime) ? media.currentTime : 0;
      currentTimeReadout.textContent = "Current: " + formatSeconds(time) + "s";
    }}

    function clearLabelAutoSave() {{
      if (labelAutoSaveTimer) {{
        window.clearTimeout(labelAutoSaveTimer);
        labelAutoSaveTimer = null;
      }}
      if (labelAutoSaveController) {{
        labelAutoSaveController.abort();
        labelAutoSaveController = null;
      }}
    }}

    function setAutoSaveStatus(message, isError = false) {{
      if (!saveStatus) return;
      saveStatus.className = isError ? "save-status error" : "save-status";
      saveStatus.style.display = "block";
      saveStatus.textContent = message;
    }}

    // The dashboard's labels list polls the file-system record for status. If
    // the user edits and immediately hits Back, we want the row to show
    // "draft" rather than "not started" — so flip the visible status pill the
    // moment any edit happens, instead of waiting for the round-trip to land.
    const labelStatePill = document.querySelector(".label-state");
    const uncompleteLabelButton = document.getElementById("uncompleteLabelButton");
    let visibleStatusFlipped = false;
    function labelStatusText() {{
      if (!labelStatePill) return "";
      return String(labelStatePill.textContent || "").toLowerCase();
    }}
    function updateUncompleteVisibility() {{
      if (!uncompleteLabelButton) return;
      const completed = labelStatusText().indexOf("completed") >= 0;
      uncompleteLabelButton.hidden = !completed;
    }}
    function markVisibleStatusDraft() {{
      if (visibleStatusFlipped || !labelStatePill) return;
      visibleStatusFlipped = true;
      labelStatePill.textContent = "Status: draft";
      updateUncompleteVisibility();
    }}
    updateUncompleteVisibility();
    if (uncompleteLabelButton) {{
      uncompleteLabelButton.addEventListener("click", function () {{
        // Confirm before deleting the project sample files. Roll-back also
        // clears any auto-train queue for this label, so the user shouldn't
        // be surprised when the next Train Now click starts from scratch.
        const ok = window.confirm(
          "Remove this sample from the fine-tuning project's training set and roll the label back to a draft? "
          + "Audio and RTTM copies in the project's training set will be deleted."
        );
        if (!ok) return;
        const audioInput = labelForm.querySelector("input[name='audio_file']");
        const audioName = audioInput ? String(audioInput.value || "").trim() : "";
        if (!audioName) {{
          window.alert("This page is missing the audio file reference.");
          return;
        }}
        clearLabelAutoSave();
        if (labelAutoSaveController) {{
          labelAutoSaveController.abort();
          labelAutoSaveController = null;
        }}
        const formData = new FormData();
        formData.append("audio_file", audioName);
        formData.append("label_return_to", appLocalPath());
        setLabelFormBusy(true);
        if (saveStatus) {{
          saveStatus.className = "save-status";
          saveStatus.style.display = "block";
          saveStatus.textContent = "Removing sample from training...";
        }}
        window.fetch(appBasePath() + "/training-labels/uncomplete", {{
          method: "POST",
          body: formData,
          credentials: "same-origin",
          redirect: "follow",
          headers: {{ Accept: "text/html,*/*" }},
        }})
          .then(function (response) {{
            if (!response.ok && response.type !== "opaqueredirect") {{
              throw new Error("Uncomplete failed");
            }}
            window.location.replace(response.url || (appBasePath() + appLocalPath()));
          }})
          .catch(function () {{
            setLabelFormBusy(false);
            if (saveStatus) {{
              saveStatus.className = "save-status error";
              saveStatus.style.display = "block";
              saveStatus.textContent = "Could not remove the sample. Check the server and try again.";
            }}
          }});
      }});
    }}

    // Force a save on the very first edit so the dashboard sees "draft" right
    // away. Subsequent edits keep using the 1500 ms debounce so we don't slam
    // the server while someone is mid-typing.
    let firstEditSaved = false;

    function runLabelAutoSave() {{
      labelAutoSaveTimer = null;
      if (!window.fetch || labelForm.dataset.submitting === "true") return;
      syncLabelSegments(false);
      syncLabelDialogueVisibility();
      if (labelAutoSaveController) labelAutoSaveController.abort();
      labelAutoSaveController = new AbortController();
      const formData = new FormData(labelForm);
      formData.set("label_segments", labelSegments.value);
      if (labelIncludeTranscript) formData.set("label_include_transcript", labelIncludeTranscript.value);
      formData.set("label_action", "draft");
      formData.set("label_auto_save", "1");
      setAutoSaveStatus("Auto-saving draft...");
      window.fetch(labelForm.action, {{
        method: "POST",
        body: formData,
        credentials: "same-origin",
        redirect: "manual",
        signal: labelAutoSaveController.signal,
        headers: {{ Accept: "application/json,text/html,*/*" }},
      }})
        .then(function (response) {{
          if (
            response.type === "opaqueredirect" ||
            response.status === 0 ||
            (response.status >= 200 && response.status < 400)
          ) {{
            labelAutoSaveController = null;
            setAutoSaveStatus("Auto-saved draft at " + new Date().toLocaleTimeString() + ".");
            return;
          }}
          throw new Error("Label auto-save failed");
        }})
        .catch(function (error) {{
          if (error && error.name === "AbortError") return;
          labelAutoSaveController = null;
          setAutoSaveStatus("Auto-save failed. Press Save Draft when you are ready.", true);
        }});
    }}

    function scheduleLabelAutoSave() {{
      if (!window.fetch || labelForm.dataset.submitting === "true") return;
      markVisibleStatusDraft();
      if (!firstEditSaved) {{
        // First edit after the page loads: skip the debounce so the row flips
        // to "draft" on the dashboard before the user has a chance to navigate
        // back. Bug we hit otherwise: edit, immediately hit Back, come back,
        // and the row was still "not started" because the 1500 ms timer never
        // fired.
        firstEditSaved = true;
        if (labelAutoSaveTimer) {{
          window.clearTimeout(labelAutoSaveTimer);
          labelAutoSaveTimer = null;
        }}
        runLabelAutoSave();
        return;
      }}
      if (labelAutoSaveTimer) window.clearTimeout(labelAutoSaveTimer);
      labelAutoSaveTimer = window.setTimeout(runLabelAutoSave, 1500);
    }}

    // Last-ditch flush when the page is being torn down (Back button, tab
    // close, refresh). sendBeacon survives unload where a normal fetch would
    // be cancelled, so any change still sitting in the debounce window gets
    // delivered. fetch with keepalive is the fallback for browsers that don't
    // implement sendBeacon (older Safari).
    function flushLabelAutoSaveOnUnload() {{
      if (labelForm.dataset.submitting === "true") return;
      const hasPending = labelAutoSaveTimer || labelAutoSaveController;
      if (!hasPending && !firstEditSaved) return;
      if (labelAutoSaveTimer) {{
        window.clearTimeout(labelAutoSaveTimer);
        labelAutoSaveTimer = null;
      }}
      if (labelAutoSaveController) {{
        labelAutoSaveController.abort();
        labelAutoSaveController = null;
      }}
      try {{
        syncLabelSegments(false);
        syncLabelDialogueVisibility();
      }} catch (_err) {{}}
      const formData = new FormData(labelForm);
      formData.set("label_segments", labelSegments.value);
      if (labelIncludeTranscript) formData.set("label_include_transcript", labelIncludeTranscript.value);
      formData.set("label_action", "draft");
      formData.set("label_auto_save", "1");
      let delivered = false;
      if (navigator.sendBeacon) {{
        try {{
          delivered = navigator.sendBeacon(labelForm.action, formData);
        }} catch (_err) {{
          delivered = false;
        }}
      }}
      if (!delivered && window.fetch) {{
        try {{
          window.fetch(labelForm.action, {{
            method: "POST",
            body: formData,
            credentials: "same-origin",
            keepalive: true,
            redirect: "manual",
          }});
        }} catch (_err) {{}}
      }}
    }}

    window.addEventListener("pagehide", flushLabelAutoSaveOnUnload);
    // visibilitychange catches the case where the OS / browser puts the tab
    // in the background without unloading it (mobile, tab discard); we still
    // want any pending edits durable before the tab can be killed.
    document.addEventListener("visibilitychange", function () {{
      if (document.visibilityState === "hidden") flushLabelAutoSaveOnUnload();
    }});

    function playRange(start, end, selectedRow) {{
      if (!media) return;
      const safeStart = rowNumber(start);
      const safeEnd = rowNumber(end);
      if (!Number.isFinite(safeStart)) return;
      playRequestId += 1;
      const requestId = playRequestId;
      clearPlaybackTimers();
      if (selectedRow && selectedRow.matches("[data-label-row]")) {{
        setSelectedLabelRow(selectedRow);
      }} else if (selectedRow && selectedRow.matches("[data-cue-row]")) {{
        setSelectedCueRow(selectedRow);
      }}
      const rangeEnd = Number.isFinite(safeEnd) && safeEnd > safeStart ? safeEnd : safeStart;
      setPlaybackMessage("Loading selected segment...", selectedRangeSummary(selectedRow, safeStart, rangeEnd));

      const attachStopAtEnd = function () {{
        if (!stopAtSegmentEnd.checked || !Number.isFinite(safeEnd) || safeEnd <= safeStart) return;
        activeStopHandler = function () {{
          if (requestId !== playRequestId) return;
          if (media.currentTime >= safeEnd - 0.025) {{
            media.pause();
            media.removeEventListener("timeupdate", activeStopHandler);
            activeStopHandler = null;
            setCurrentRows();
          }}
        }};
        media.addEventListener("timeupdate", activeStopHandler);
      }};

      const startPlayback = function () {{
        if (requestId !== playRequestId) return;
        attachStopAtEnd();
        const playPromise = media.play();
        if (playPromise && typeof playPromise.catch === "function") {{
          playPromise.catch(function () {{
            if (requestId !== playRequestId) return;
            setPlaybackMessage("Press the audio player's play button.", selectedRangeSummary(selectedRow, safeStart, rangeEnd));
          }});
        }}
        setCurrentRows();
      }};

      const seekThenPlay = function () {{
        if (requestId !== playRequestId) return;
        try {{
          media.pause();
          media.currentTime = Math.max(safeStart, 0);
        }} catch (_error) {{
          setPlaybackMessage("Could not jump to that segment.", "Try opening the source audio from the media player controls.");
          return;
        }}
        let started = false;
        const startOnce = function () {{
          if (started || requestId !== playRequestId) return;
          started = true;
          media.removeEventListener("seeked", startOnce);
          if (activeSeekTimer) {{
            window.clearTimeout(activeSeekTimer);
            activeSeekTimer = null;
          }}
          startPlayback();
        }};
        media.addEventListener("seeked", startOnce, {{ once: true }});
        activeSeekTimer = window.setTimeout(startOnce, 350);
      }};

      if (media.readyState === 0) {{
        media.addEventListener("loadedmetadata", seekThenPlay, {{ once: true }});
        media.load();
      }} else {{
        seekThenPlay();
      }}
    }}

    function playCueRow(row) {{
      playRange(row.dataset.start, row.dataset.end, row);
    }}

    function playLabelRow(row) {{
      updateLabelRowDataset(row);
      playRange(row.dataset.start, row.dataset.end, row);
    }}

    function createLabelRow() {{
      const row = document.createElement("tr");
      row.setAttribute("data-label-row", "");
      row.setAttribute("tabindex", "0");
      row.innerHTML = '<td class="row-index"><span class="drag-handle" draggable="true" role="button" tabindex="0" aria-label="Drag to reorder this label" title="Drag to reorder">⋮⋮</span><span class="row-number"></span></td><td><div class="time-edit"><label><span>Start</span><input class="label-start" type="text" inputmode="decimal" value=""></label><label><span>End</span><input class="label-end" type="text" inputmode="decimal" value=""></label></div></td><td><input class="label-speaker" list="speakerOptions" type="text" value=""></td><td class="label-dialogue-cell"><input class="label-dialogue" type="text" value="" placeholder="Optional dialogue"></td><td class="action-cell"><button class="play-label" type="button">Listen</button><button class="delete-label" type="button" aria-label="Delete this label">Delete</button></td>';
      labelRows.appendChild(row);
      renumberLabelRows();
      return row;
    }}

    function setLabelRowValues(row, start, end, speaker, note) {{
      row.querySelector(".label-start").value = Number.isFinite(rowNumber(start)) ? rowNumber(start).toFixed(3) : "";
      row.querySelector(".label-end").value = Number.isFinite(rowNumber(end)) ? rowNumber(end).toFixed(3) : "";
      row.querySelector(".label-speaker").value = speaker && speaker !== "-" ? speaker : "";
      row.querySelector(".label-dialogue").value = note || "";
      updateLabelRowDataset(row);
      // Newly-populated row may match a previously-marked-done identity from
      // an earlier session (e.g. dragged the same cue back in); refresh.
      if (typeof applyDoneStateToRow === "function") applyDoneStateToRow(row);
    }}

    function useCueAsLabel(cueRow) {{
      const target = selectedOrCurrentLabelRow() || createLabelRow();
      setLabelRowValues(target, cueRow.dataset.start, cueRow.dataset.end, cueRow.dataset.speaker, cueRow.dataset.text);
      setSelectedLabelRow(target);
      syncLabelSegments();
      applyLabelFilters();
      setPlaybackMessage("Added detected segment to labels.", labelSummary(target));
      scheduleLabelAutoSave();
    }}

    cueTable.addEventListener("click", function (event) {{
      const row = event.target.closest("[data-cue-row]");
      if (!row) return;
      setSelectedCueRow(row);
      if (event.target.closest("button.use-cue")) {{
        event.stopPropagation();
        useCueAsLabel(row);
        return;
      }}
      if (event.target.closest("button.play-cue")) {{
        event.stopPropagation();
        playCueRow(row);
      }}
    }});

    cueTable.addEventListener("keydown", function (event) {{
      const row = event.target.closest("[data-cue-row]");
      if (!row || (event.key !== "Enter" && event.key !== " ")) return;
      event.preventDefault();
      setSelectedCueRow(row);
    }});

    function renumberLabelRows() {{
      labelRowList().forEach(function (row, index) {{
        const numberCell = row.querySelector(".row-number");
        if (numberCell) numberCell.textContent = String(index + 1);
        row.dataset.position = String(index + 1);
      }});
    }}

    function deleteLabelRow(row) {{
      if (!row || !row.parentNode) return;
      const wasSelected = selectedLabelRow === row;
      const sibling = row.nextElementSibling || row.previousElementSibling;
      row.parentNode.removeChild(row);
      if (wasSelected) {{
        setSelectedLabelRow(sibling && sibling.matches("[data-label-row]") ? sibling : null);
      }}
      renumberLabelRows();
      syncLabelSegments();
      applyLabelFilters();
      setCurrentRows();
      scheduleLabelAutoSave();
    }}

    // "I've already labeled this row" -- a manual checkmark per row, stored
     // in localStorage keyed by the audio file. The pipeline doesn't see it,
     // it just helps me visually track which rows I've already gone through
     // on long files. Identity is the row's content (start|end|speaker), so
     // editing a row clears its done-mark automatically (which is the right
     // behavior -- if I changed the times, I should re-confirm it).
    const DONE_LABEL_STORAGE_KEY = "doneLabels.v1." + (mediaFileNameForStorage() || "_unknown_");
    function mediaFileNameForStorage() {{
      const audioInput = labelForm.querySelector("input[name='audio_file']");
      return audioInput ? String(audioInput.value || "").trim() : "";
    }}
    function readDoneRowSet() {{
      try {{
        const raw = window.localStorage.getItem(DONE_LABEL_STORAGE_KEY);
        if (!raw) return new Set();
        const parsed = JSON.parse(raw);
        return new Set(Array.isArray(parsed) ? parsed.filter(function (v) {{ return typeof v === "string"; }}) : []);
      }} catch (_err) {{ return new Set(); }}
    }}
    function writeDoneRowSet(set) {{
      try {{
        window.localStorage.setItem(DONE_LABEL_STORAGE_KEY, JSON.stringify(Array.from(set)));
      }} catch (_err) {{}}
    }}
    let doneRowSet = readDoneRowSet();
    function rowDoneIdentity(row) {{
      if (!row) return "";
      const start = row.dataset.start || "";
      const end = row.dataset.end || "";
      const speaker = row.dataset.speaker || "";
      // Empty rows would all share the same id and would mass-toggle together.
      // Require some content before we treat a row as identifiable.
      if (!start && !end && !speaker) return "";
      return start + "|" + end + "|" + speaker;
    }}
    function applyDoneStateToRow(row) {{
      if (!row) return;
      const identity = rowDoneIdentity(row);
      const button = row.querySelector(".toggle-done-label");
      const isDone = Boolean(identity) && doneRowSet.has(identity);
      row.classList.toggle("is-done", isDone);
      if (button) {{
        button.classList.toggle("is-on", isDone);
        button.setAttribute("aria-pressed", isDone ? "true" : "false");
        button.textContent = isDone ? "Done ✓" : "Mark done";
      }}
    }}
    function applyDoneStateToAllRows() {{
      labelRowList().forEach(applyDoneStateToRow);
    }}
    function toggleRowDone(row) {{
      if (!row) return;
      // updateLabelRowDataset keeps row.dataset.start/end/speaker fresh from
      // the current input values, so identity is computed against the latest
      // text the user has typed -- not whatever was there at page load.
      updateLabelRowDataset(row);
      const identity = rowDoneIdentity(row);
      if (!identity) {{
        window.alert("Fill in start, end, or speaker before marking the row done.");
        return;
      }}
      if (doneRowSet.has(identity)) {{
        doneRowSet.delete(identity);
      }} else {{
        doneRowSet.add(identity);
      }}
      writeDoneRowSet(doneRowSet);
      applyDoneStateToRow(row);
    }}

    labelRows.addEventListener("click", function (event) {{
      const row = event.target.closest("[data-label-row]");
      if (!row) return;
      if (event.target.closest("button.delete-label")) {{
        event.stopPropagation();
        deleteLabelRow(row);
        return;
      }}
      if (event.target.closest("button.toggle-done-label")) {{
        event.stopPropagation();
        toggleRowDone(row);
        return;
      }}
      setSelectedLabelRow(row);
      if (event.target.closest("button.play-label")) {{
        event.stopPropagation();
        playLabelRow(row);
        return;
      }}
    }});

    labelRows.addEventListener("focusin", function (event) {{
      const row = event.target.closest("[data-label-row]");
      rememberLabelTimeInput(event.target);
      if (row) setSelectedLabelRow(row);
    }});

    // The big lag culprit during typing was that every keystroke ran
    // syncLabelSegments + applyLabelFilters + setCurrentRows -- each of those
    // walks every row. On a long file with hundreds of labels, that's the
    // page locking up while you type. The fix: just refresh the *one* row
    // that changed, and let the auto-save (1.5 s debounce) rebuild the
    // serialized segments before it actually POSTs. Filter visibility only
    // needs a full walk when the filter UI changes, not on every keystroke.
    labelRows.addEventListener("input", function (event) {{
      const row = event.target.closest("[data-label-row]");
      if (row) {{
        updateLabelRowDataset(row);
        // Identity is content-based, so editing start/end/speaker can clear
        // an existing "done" mark. Refresh the visual right away.
        applyDoneStateToRow(row);
      }}
      scheduleLabelAutoSave();
    }});
    labelRows.addEventListener("change", function (event) {{
      // "change" fires on blur / select-change -- we're stepping off a field,
      // so it's worth doing the slightly heavier sync once to refresh health
      // chips and filter visibility.
      const row = event.target.closest("[data-label-row]");
      if (row) {{
        updateLabelRowDataset(row);
        applyDoneStateToRow(row);
      }}
      syncLabelSegments();
      applyLabelFilters();
      scheduleLabelAutoSave();
    }});

    let dragRow = null;

    function clearDropMarkers() {{
      labelRowList().forEach(function (row) {{
        row.classList.remove("drop-above", "drop-below");
      }});
    }}

    labelRows.addEventListener("dragstart", function (event) {{
      const handle = event.target.closest(".drag-handle");
      if (!handle) {{
        event.preventDefault();
        return;
      }}
      dragRow = handle.closest("[data-label-row]");
      if (!dragRow) {{
        event.preventDefault();
        return;
      }}
      dragRow.classList.add("dragging");
      if (event.dataTransfer) {{
        event.dataTransfer.effectAllowed = "move";
        try {{ event.dataTransfer.setData("text/plain", "label-row"); }} catch (_err) {{}}
      }}
    }});

    labelRows.addEventListener("dragover", function (event) {{
      if (!dragRow) return;
      const target = event.target.closest("[data-label-row]");
      if (!target || target === dragRow) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
      const rect = target.getBoundingClientRect();
      const after = (event.clientY - rect.top) > rect.height / 2;
      clearDropMarkers();
      target.classList.add(after ? "drop-below" : "drop-above");
    }});

    labelRows.addEventListener("dragleave", function (event) {{
      const target = event.target.closest("[data-label-row]");
      if (target) target.classList.remove("drop-above", "drop-below");
    }});

    labelRows.addEventListener("drop", function (event) {{
      if (!dragRow) return;
      const target = event.target.closest("[data-label-row]");
      if (!target || target === dragRow) {{
        clearDropMarkers();
        return;
      }}
      event.preventDefault();
      const rect = target.getBoundingClientRect();
      const after = (event.clientY - rect.top) > rect.height / 2;
      target.parentNode.insertBefore(dragRow, after ? target.nextSibling : target);
      clearDropMarkers();
      renumberLabelRows();
      syncLabelSegments();
      applyLabelFilters();
      setSelectedLabelRow(dragRow);
      scheduleLabelAutoSave();
    }});

    labelRows.addEventListener("dragend", function () {{
      if (dragRow) dragRow.classList.remove("dragging");
      clearDropMarkers();
      dragRow = null;
    }});

    function appendBlankLabelRow() {{
      const row = createLabelRow();
      setSelectedLabelRow(row);
      renumberLabelRows();
      syncLabelSegments();
      row.scrollIntoView({{ block: "nearest" }});
      const startInput = row.querySelector(".label-start");
      if (startInput) startInput.focus();
      scheduleLabelAutoSave();
    }}

    addLabelRow.addEventListener("click", appendBlankLabelRow);
    if (addLabelRowTop) addLabelRowTop.addEventListener("click", appendBlankLabelRow);
    if (setLabelTimeFromPlayer) setLabelTimeFromPlayer.addEventListener("click", setActiveLabelTimeFromPlayer);

    // One-click "I just heard a speaker change, capture it now". Creates a
    // fresh row, drops the playhead time into Start, focuses the speaker
    // box so you can type the speaker name and keep listening.
    const addLabelAtCurrentTime = document.getElementById("addLabelAtCurrentTime");
    if (addLabelAtCurrentTime) {{
      addLabelAtCurrentTime.addEventListener("click", function () {{
        if (!media || !Number.isFinite(media.currentTime)) {{
          window.alert("The audio player has no current time yet -- start playback or click the waveform first.");
          return;
        }}
        const row = createLabelRow();
        const stamp = formatSeconds(media.currentTime);
        const startInput = row.querySelector(".label-start");
        if (startInput) startInput.value = stamp;
        updateLabelRowDataset(row);
        setSelectedLabelRow(row);
        renumberLabelRows();
        syncLabelSegments();
        applyLabelFilters();
        row.scrollIntoView({{ block: "nearest" }});
        const speakerInput = row.querySelector(".label-speaker");
        if (speakerInput) {{
          speakerInput.focus();
          speakerInput.select();
        }}
        scheduleLabelAutoSave();
        setPlaybackMessage("Started a new label at " + stamp + "s.", "Type the speaker name and the End time when ready.");
      }});
    }}

    function setLabelFormBusy(isBusy) {{
      labelForm.dataset.submitting = isBusy ? "true" : "";
      labelForm.setAttribute("aria-busy", isBusy ? "true" : "false");
      labelForm.querySelectorAll("button[type='submit']").forEach(function (button) {{
        button.disabled = isBusy;
      }});
    }}

    if (closeTrainingTargetDialog && trainingTargetDialog) {{
      closeTrainingTargetDialog.addEventListener("click", function () {{
        trainingTargetDialog.close();
      }});
    }}
    trainingModeRadios.forEach(function (radio) {{
      radio.addEventListener("change", function () {{
        if (radio.checked) setTrainingMode(radio.value);
      }});
    }});
    function validateBaseModeInputs() {{
      // Need a model name to register a brand-new fine-tuned project. The
      // backend slugifies it, but an empty value would slugify to "project"
      // and silently overwrite an existing default-named project — better to
      // make the user be explicit.
      const projectValue = newTrainingProjectName ? String(newTrainingProjectName.value || "").trim() : "";
      if (!projectValue) {{
        window.alert("Provide a name for the new model so future training runs can find it.");
        if (newTrainingProjectName) newTrainingProjectName.focus();
        return false;
      }}
      return true;
    }}
    if (confirmTrainingTargets) {{
      confirmTrainingTargets.addEventListener("click", function () {{
        if (currentTrainingMode === "existing" && !selectedExistingTargets().length) {{
          window.alert("Tick at least one fine-tuned model to continue training, or switch to the default base option.");
          return;
        }}
        if (currentTrainingMode === "base" && !validateBaseModeInputs()) return;
        const targets = selectedDialogTargets();
        if (!targets.length) {{
          window.alert("Pick a model to train, or cancel to keep editing.");
          return;
        }}
        setTrainingTargetHiddenInputs(targets, true);
        if (trainingTargetDialog) trainingTargetDialog.close();
        submitCompletedTrainingLabel();
      }});
    }}
    if (skipTrainingQueue) {{
      skipTrainingQueue.addEventListener("click", function () {{
        if (currentTrainingMode === "existing" && !selectedExistingTargets().length) {{
          window.alert("Tick at least one fine-tuned model to associate this label with, or switch to the default base option.");
          return;
        }}
        if (currentTrainingMode === "base" && !validateBaseModeInputs()) return;
        const targets = selectedDialogTargets();
        setTrainingTargetHiddenInputs(targets, false);
        if (trainingTargetDialog) trainingTargetDialog.close();
        submitCompletedTrainingLabel();
      }});
    }}

    labelForm.addEventListener("submit", function (event) {{
      syncLabelSegments(true);
      syncLabelDialogueVisibility();
      const submitter = event.submitter;
      if (submitter && submitter.value === "complete" && !completeSubmitConfirmed) {{
        event.preventDefault();
        openTrainingTargetDialog();
        return;
      }}
      completeSubmitConfirmed = false;
      if (!window.fetch || labelForm.dataset.submitting === "true") {{
        return;
      }}
      event.preventDefault();
      clearLabelAutoSave();
      const formData = new FormData(labelForm);
      if (submitter && submitter.name) {{
        formData.set(submitter.name, submitter.value || "");
      }}
      setLabelFormBusy(true);
      if (saveStatus) {{
        saveStatus.className = "save-status";
        saveStatus.style.display = "block";
        saveStatus.textContent = "Saving label...";
      }}
      window.fetch(labelForm.action, {{
        method: "POST",
        body: formData,
        credentials: "same-origin",
        redirect: "follow",
        headers: {{ Accept: "text/html,*/*" }},
      }})
        .then(function (response) {{
          if (!response.ok) throw new Error("Label save failed");
          window.location.replace(response.url || (appBasePath() + appLocalPath()));
        }})
        .catch(function () {{
          setLabelFormBusy(false);
          if (saveStatus) {{
            saveStatus.className = "save-status error";
            saveStatus.style.display = "block";
            saveStatus.textContent = "The label could not be saved. Check the server and try again.";
          }}
        }});
    }});
    if (typeof FormDataEvent !== "undefined") {{
      labelForm.addEventListener("formdata", function (event) {{
        syncLabelSegments(false);
        syncLabelDialogueVisibility();
        event.formData.set("label_segments", labelSegments.value);
        if (labelIncludeTranscript) event.formData.set("label_include_transcript", labelIncludeTranscript.value);
      }});
    }}

    // timeupdate fires several times a second during playback. drawWaveform
    // does a full canvas redraw and setCurrentRows walks every row, so calling
    // them straight from the event handler used to chew CPU and make the page
    // feel laggy. Coalesce the work into one rAF tick per frame instead.
    let waveformDrawScheduled = false;
    let currentRowsScheduled = false;
    function requestDrawWaveform() {{
      if (waveformDrawScheduled) return;
      waveformDrawScheduled = true;
      window.requestAnimationFrame(function () {{
        waveformDrawScheduled = false;
        drawWaveform();
      }});
    }}
    function requestSetCurrentRows() {{
      if (currentRowsScheduled) return;
      currentRowsScheduled = true;
      window.requestAnimationFrame(function () {{
        currentRowsScheduled = false;
        setCurrentRows();
      }});
    }}
    if (media) {{
      media.load();
      media.addEventListener("timeupdate", requestSetCurrentRows);
      media.addEventListener("timeupdate", requestDrawWaveform);
      media.addEventListener("seeked", setCurrentRows);
      media.addEventListener("seeked", requestDrawWaveform);
      media.addEventListener("pause", setCurrentRows);
      media.addEventListener("loadedmetadata", drawWaveform);
    }}

    /*
     * Local audio cache (IndexedDB).
     *
     * The dashboard serves WAVs over a WAVE-cluster NFS mount. Every time I
     * open a review page the browser has to stream chunks back through that
     * mount before playback can start, and that's where most of the felt
     * lag was coming from. Once the file lives in IndexedDB on this device,
     * the next visit (and every visit after) skips the network entirely:
     * we read the Blob locally, hand it to the <audio> element as a
     * blob: URL, and playback starts instantly.
     *
     * First visit: keep the streaming src so playback starts immediately.
     * The Cache audio button saves the whole file locally without forcing that
     * full download during page load.
     *
     * Cache invalidation is by URL + content-length, so if a file is
     * regenerated server-side with a different size it's redownloaded
     * instead of returning stale audio.
     */
    const AUDIO_CACHE_DB_NAME = "ml-speech-audio-cache";
    const AUDIO_CACHE_STORE = "blobs";
    const AUDIO_CACHE_DB_VERSION = 1;
    const audioCacheStatus = document.getElementById("audioCacheStatus");
    const cacheAudioButton = document.getElementById("cacheAudioButton");
    const cacheAllAudioButton = document.getElementById("cacheAllAudioButton");
    const clearAudioCacheButton = document.getElementById("clearAudioCacheButton");

    function setAudioCacheStatus(message, kind) {{
      if (!audioCacheStatus) return;
      audioCacheStatus.textContent = message;
      audioCacheStatus.classList.remove("is-cached", "is-fetching", "is-error");
      if (kind) audioCacheStatus.classList.add(kind);
    }}
    function showClearAudioCacheButton(show) {{
      if (clearAudioCacheButton) clearAudioCacheButton.hidden = !show;
    }}
    function showCacheAudioButton(show) {{
      if (cacheAudioButton) cacheAudioButton.hidden = !show;
    }}
    function setCacheAudioButtonBusy(isBusy) {{
      if (cacheAudioButton) cacheAudioButton.disabled = isBusy;
      if (cacheAllAudioButton) cacheAllAudioButton.disabled = isBusy;
    }}
    function normalizedAudioCacheUrl(href) {{
      const raw = String(href || "").trim();
      if (!raw) return "";
      try {{
        return new URL(raw, window.location.href).href;
      }} catch (_err) {{
        return "";
      }}
    }}

    function openAudioCacheDb() {{
      return new Promise(function (resolve) {{
        if (!window.indexedDB) {{ resolve(null); return; }}
        let request;
        try {{
          request = window.indexedDB.open(AUDIO_CACHE_DB_NAME, AUDIO_CACHE_DB_VERSION);
        }} catch (_err) {{ resolve(null); return; }}
        request.onupgradeneeded = function () {{
          const db = request.result;
          if (!db.objectStoreNames.contains(AUDIO_CACHE_STORE)) {{
            db.createObjectStore(AUDIO_CACHE_STORE);
          }}
        }};
        request.onsuccess = function () {{ resolve(request.result); }};
        request.onerror = function () {{ resolve(null); }};
        request.onblocked = function () {{ resolve(null); }};
      }});
    }}
    function audioCacheRead(db, key) {{
      return new Promise(function (resolve) {{
        try {{
          const tx = db.transaction([AUDIO_CACHE_STORE], "readonly");
          const get = tx.objectStore(AUDIO_CACHE_STORE).get(key);
          get.onsuccess = function () {{ resolve(get.result || null); }};
          get.onerror = function () {{ resolve(null); }};
        }} catch (_err) {{ resolve(null); }}
      }});
    }}
    function audioCacheWrite(db, key, value) {{
      return new Promise(function (resolve) {{
        try {{
          const tx = db.transaction([AUDIO_CACHE_STORE], "readwrite");
          const put = tx.objectStore(AUDIO_CACHE_STORE).put(value, key);
          put.onsuccess = function () {{ resolve(true); }};
          put.onerror = function () {{ resolve(false); }};
        }} catch (_err) {{ resolve(false); }}
      }});
    }}
    function audioCacheDelete(db, key) {{
      return new Promise(function (resolve) {{
        try {{
          const tx = db.transaction([AUDIO_CACHE_STORE], "readwrite");
          const del = tx.objectStore(AUDIO_CACHE_STORE).delete(key);
          del.onsuccess = function () {{ resolve(true); }};
          del.onerror = function () {{ resolve(false); }};
        }} catch (_err) {{ resolve(false); }}
      }});
    }}

    function formatMb(bytes) {{
      if (!Number.isFinite(bytes) || bytes <= 0) return "?";
      return (bytes / 1048576).toFixed(1) + " MB";
    }}

    function swapMediaToBlob(blob, sourceUrl) {{
      if (!media) return;
      // If the user is already listening, swapping the src would briefly
      // pause-and-restart playback, which is jarring mid-segment. Defer
      // until the next pause/end -- the blob is already saved in IDB, so
      // the next visit will pick it up instantly even if we never swap on
      // this visit.
      const performSwap = function () {{
        const wasPlaying = !media.paused && !media.ended;
        const previousTime = Number.isFinite(media.currentTime) ? media.currentTime : 0;
        if (media.dataset.cacheBlobUrl) {{
          try {{ URL.revokeObjectURL(media.dataset.cacheBlobUrl); }} catch (_e) {{}}
        }}
        const blobUrl = URL.createObjectURL(blob);
        media.dataset.cacheBlobUrl = blobUrl;
        media.dataset.cacheStreamingSrc = sourceUrl;
        media.src = blobUrl;
        media.load();
        const onReady = function () {{
          media.removeEventListener("loadedmetadata", onReady);
          try {{ media.currentTime = previousTime; }} catch (_e) {{}}
          if (wasPlaying) {{
            const playPromise = media.play();
            if (playPromise && typeof playPromise.catch === "function") playPromise.catch(function () {{}});
          }}
          requestDrawWaveform();
        }};
        media.addEventListener("loadedmetadata", onReady, {{ once: true }});
      }};
      if (media.paused || media.ended) {{
        performSwap();
      }} else {{
        media.addEventListener("pause", performSwap, {{ once: true }});
        media.addEventListener("ended", performSwap, {{ once: true }});
      }}
    }}

    async function fetchAudioWithProgress(sourceUrl) {{
      const response = await window.fetch(sourceUrl, {{ credentials: "same-origin", cache: "force-cache" }});
      if (!response.ok) {{
        throw new Error("audio fetch returned " + response.status);
      }}
      const totalHeader = response.headers.get("content-length");
      const total = totalHeader ? parseInt(totalHeader, 10) : 0;
      if (!response.body || !response.body.getReader) {{
        setAudioCacheStatus("Caching to local: downloading audio...", "is-fetching");
        const blob = await response.blob();
        return {{
          blob,
          contentLength: total || blob.size,
        }};
      }}
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;
      while (true) {{
        const step = await reader.read();
        if (step.done) break;
        chunks.push(step.value);
        received += step.value.length;
        if (total > 0) {{
          const percent = Math.min(99, Math.floor((received / total) * 100));
          setAudioCacheStatus("Caching to local: " + percent + "% (" + formatMb(received) + " / " + formatMb(total) + ")", "is-fetching");
        }} else {{
          setAudioCacheStatus("Caching to local: " + formatMb(received), "is-fetching");
        }}
      }}
      const contentType = response.headers.get("content-type") || "audio/wav";
      return {{
        blob: new Blob(chunks, {{ type: contentType }}),
        contentLength: total || received,
      }};
    }}

    async function refreshAudioCacheWithNewDb(sourceUrl) {{
      const db = await openAudioCacheDb();
      if (!db) {{
        setAudioCacheStatus("Local cache: unavailable in this browser", "is-error");
        return;
      }}
      try {{
        await refreshAudioCache(db, sourceUrl);
      }} finally {{
        try {{ db.close(); }} catch (_e) {{}}
      }}
    }}

    async function ensureAudioCached() {{
      if (!media) return;
      const sourceUrl = media.currentSrc || media.src;
      if (!sourceUrl || sourceUrl.startsWith("blob:") || sourceUrl.startsWith("data:")) {{
        setAudioCacheStatus("Local cache: not applicable");
        return;
      }}
      const db = await openAudioCacheDb();
      if (!db) {{
        setAudioCacheStatus("Local cache: unavailable in this browser", "is-error");
        return;
      }}
      try {{
        const cached = await audioCacheRead(db, sourceUrl);
        if (cached && cached.blob && cached.blob.size > 0) {{
          swapMediaToBlob(cached.blob, sourceUrl);
          setAudioCacheStatus("Cached locally · " + formatMb(cached.size || cached.blob.size) + " · zero network for playback", "is-cached");
          showCacheAudioButton(false);
          showClearAudioCacheButton(true);
          // Background validation: if the server file's size differs from
          // the cached entry, redownload silently so we don't keep stale
          // audio. HEAD is cheap and won't compete with playback. Reopen IDB
          // if the refresh is needed because this function closes its handle
          // as soon as the cache check is done.
          window.fetch(sourceUrl, {{ method: "HEAD", credentials: "same-origin" }})
            .then(function (response) {{
              if (!response.ok) return;
              const total = parseInt(response.headers.get("content-length") || "0", 10);
              if (total > 0 && cached.size && total !== cached.size) {{
                refreshAudioCacheWithNewDb(sourceUrl);
              }}
            }})
            .catch(function () {{}});
          return;
        }}
        setAudioCacheStatus("Local cache: not saved yet. Use Cache audio to make future opens faster.");
        showCacheAudioButton(true);
        showClearAudioCacheButton(false);
      }} finally {{
        try {{ db.close(); }} catch (_e) {{}}
      }}
    }}

    async function refreshAudioCache(db, sourceUrl) {{
      showCacheAudioButton(false);
      setCacheAudioButtonBusy(true);
      setAudioCacheStatus("Caching audio locally. Playback can continue while this runs...", "is-fetching");
      try {{
        const result = await fetchAudioWithProgress(sourceUrl);
        const stored = await audioCacheWrite(db, sourceUrl, {{
          blob: result.blob,
          size: result.contentLength,
          savedAt: Date.now(),
        }});
        if (stored) {{
          swapMediaToBlob(result.blob, sourceUrl);
          setAudioCacheStatus("Cached locally · " + formatMb(result.contentLength) + " · zero network for playback", "is-cached");
          showCacheAudioButton(false);
          showClearAudioCacheButton(true);
        }} else {{
          setAudioCacheStatus("Local cache: write failed (likely out of quota)", "is-error");
          showCacheAudioButton(true);
        }}
      }} catch (err) {{
        setAudioCacheStatus("Local cache: download failed, staying on streaming src", "is-error");
        showCacheAudioButton(true);
      }} finally {{
        setCacheAudioButtonBusy(false);
      }}
    }}

    async function cacheAudioSourceWithoutSwap(db, sourceUrl) {{
      const cached = await audioCacheRead(db, sourceUrl);
      if (cached && cached.blob && cached.blob.size > 0) {{
        return {{ status: "cached", bytes: cached.size || cached.blob.size }};
      }}
      const result = await fetchAudioWithProgress(sourceUrl);
      const stored = await audioCacheWrite(db, sourceUrl, {{
        blob: result.blob,
        size: result.contentLength,
        savedAt: Date.now(),
      }});
      if (!stored) throw new Error("cache write failed");
      return {{ status: "stored", bytes: result.contentLength }};
    }}

    async function loadTrainingAudioCacheUrls() {{
      const urls = [];
      const seen = new Set();
      const addUrl = function (href) {{
        const url = normalizedAudioCacheUrl(href);
        if (!url || seen.has(url) || url.startsWith("blob:") || url.startsWith("data:")) return;
        seen.add(url);
        urls.push(url);
      }};
      if (media) addUrl(media.dataset.cacheStreamingSrc || media.currentSrc || media.src);
      try {{
        const stateUrl = appBasePath() + "/api/page-state?path=/training-labels";
        const response = await window.fetch(stateUrl, {{ credentials: "same-origin", cache: "no-store" }});
        if (!response.ok) return urls;
        const payload = await response.json();
        const rows = payload && payload.context && payload.context.trainingLabels
          ? payload.context.trainingLabels.rows || []
          : [];
        rows.forEach(function (row) {{ addUrl(row.audioHref); }});
      }} catch (_err) {{}}
      return urls;
    }}

    async function cacheAllTrainingAudio() {{
      if (!window.fetch || !window.indexedDB) {{
        setAudioCacheStatus("Local cache: unavailable in this browser", "is-error");
        return;
      }}
      const urls = await loadTrainingAudioCacheUrls();
      if (!urls.length) {{
        setAudioCacheStatus("No audio files were found to cache.", "is-error");
        return;
      }}
      const db = await openAudioCacheDb();
      if (!db) {{
        setAudioCacheStatus("Local cache: unavailable in this browser", "is-error");
        return;
      }}
      let stored = 0;
      let alreadyCached = 0;
      let failed = 0;
      let bytes = 0;
      setCacheAudioButtonBusy(true);
      try {{
        for (let index = 0; index < urls.length; index += 1) {{
          const url = urls[index];
          setAudioCacheStatus("Caching all audio: " + (index + 1) + " of " + urls.length, "is-fetching");
          try {{
            const result = await cacheAudioSourceWithoutSwap(db, url);
            if (result.status === "cached") alreadyCached += 1;
            else stored += 1;
            bytes += Number(result.bytes || 0);
          }} catch (_err) {{
            failed += 1;
          }}
        }}
        setAudioCacheStatus(
          "Cache all finished: " + stored + " added, " + alreadyCached + " already cached" + (failed ? ", " + failed + " failed" : "") + ". " + formatMb(bytes) + " available locally.",
          failed ? "is-error" : "is-cached"
        );
        showCacheAudioButton(false);
        showClearAudioCacheButton(true);
      }} finally {{
        setCacheAudioButtonBusy(false);
        try {{ db.close(); }} catch (_e) {{}}
      }}
    }}

    if (cacheAudioButton) {{
      cacheAudioButton.addEventListener("click", async function () {{
        if (!media) return;
        const sourceUrl = media.dataset.cacheStreamingSrc || media.currentSrc || media.src;
        if (!sourceUrl || sourceUrl.startsWith("blob:") || sourceUrl.startsWith("data:")) return;
        await refreshAudioCacheWithNewDb(sourceUrl);
      }});
    }}
    if (cacheAllAudioButton) {{
      cacheAllAudioButton.addEventListener("click", cacheAllTrainingAudio);
    }}

    if (clearAudioCacheButton) {{
      clearAudioCacheButton.addEventListener("click", async function () {{
        if (!media) return;
        const sourceUrl = media.dataset.cacheStreamingSrc || media.currentSrc || media.src;
        if (!sourceUrl) return;
        const db = await openAudioCacheDb();
        if (!db) return;
        try {{
          await audioCacheDelete(db, sourceUrl);
          setAudioCacheStatus("Local cache cleared. Reload to re-download.");
          showCacheAudioButton(true);
          showClearAudioCacheButton(false);
        }} finally {{
          try {{ db.close(); }} catch (_e) {{}}
        }}
      }});
    }}

    if (media) {{
      // Defer the first cache check until the page is past first paint so
      // we don't fight with everything else loading. requestIdleCallback
      // is best when available, otherwise just delay a tick.
      const kickCache = function () {{ ensureAudioCached(); }};
      if (typeof window.requestIdleCallback === "function") {{
        window.requestIdleCallback(kickCache, {{ timeout: 1500 }});
      }} else {{
        window.setTimeout(kickCache, 250);
      }}
    }}
    if (playbackRate && media) {{
      playbackRate.addEventListener("change", function () {{
        const rate = rowNumber(playbackRate.value);
        media.playbackRate = Number.isFinite(rate) && rate > 0 ? rate : 1;
      }});
      media.playbackRate = rowNumber(playbackRate.value) || 1;
    }}
    if (waveformCanvas && media) {{
      waveformCanvas.addEventListener("click", function (event) {{
        const duration = mediaDuration();
        if (!duration) return;
        const rect = waveformCanvas.getBoundingClientRect();
        const ratio = Math.max(0, Math.min((event.clientX - rect.left) / rect.width, 1));
        media.currentTime = ratio * duration;
        setCurrentRows();
        drawWaveform();
      }});
      window.addEventListener("resize", requestDrawWaveform);
      // Auto-load the waveform like the original program did. The IndexedDB
      // audio cache means after the first visit the WAV is local on this
      // device, so the fetch is fast; decodeAudioData still does the bulk of
      // the work but it's a one-shot cost per page load and runs off the
      // critical path (idle callback / short delay), so the rest of the UI
      // stays responsive while it's working. The "Reload waveform" button
      // is kept as a manual escape hatch for when decoding fails.
      const loadWaveformButton = document.getElementById("loadWaveformButton");
      let waveformLoadKicked = false;
      function kickWaveformLoad() {{
        if (waveformLoadKicked) return;
        waveformLoadKicked = true;
        if (waveformStatus) waveformStatus.textContent = "Loading waveform...";
        if (loadWaveformButton) {{
          loadWaveformButton.hidden = true;
          loadWaveformButton.disabled = true;
        }}
        loadWaveform().finally(function () {{
          if (loadWaveformButton) {{
            loadWaveformButton.hidden = false;
            loadWaveformButton.disabled = false;
          }}
        }});
      }}
      if (loadWaveformButton) {{
        loadWaveformButton.addEventListener("click", function () {{
          waveformLoadKicked = false;
          waveformPeaks = [];
          waveformLoaded = false;
          requestDrawWaveform();
          kickWaveformLoad();
        }});
      }}
      // Draw the empty canvas right away so the panel isn't blank, then kick
      // off the waveform decode after the page has had a beat to paint /
      // restore audio from cache. requestIdleCallback when available keeps
      // the main thread free for first input.
      requestDrawWaveform();
      const scheduleWaveform = function () {{
        if (typeof window.requestIdleCallback === "function") {{
          window.requestIdleCallback(kickWaveformLoad, {{ timeout: 2000 }});
        }} else {{
          window.setTimeout(kickWaveformLoad, 400);
        }}
      }};
      // If the audio is already loaded enough to know its duration, kick the
      // waveform right away. Otherwise wait for metadata so the fetch can
      // benefit from the in-flight cache fill.
      if (media && media.readyState >= 1) {{
        scheduleWaveform();
      }} else if (media) {{
        media.addEventListener("loadedmetadata", scheduleWaveform, {{ once: true }});
        // Backstop in case loadedmetadata never fires (slow / failed audio):
        // the user still gets a chance at the waveform decode.
        window.setTimeout(scheduleWaveform, 6000);
      }} else {{
        scheduleWaveform();
      }}
    }} else if (waveformStatus) {{
      waveformStatus.textContent = "Waveform unavailable.";
    }}
    configureAppLinks();
    configureThemeToggle();
    showSaveStatusFromQuery();
    configureModelComparison();
    renumberLabelRows();
    syncLabelSegments();
    applyLabelFilters();
    // syncLabelSegments populated row.dataset.start/.end/.speaker, so identity
    // is now stable for every existing row. Restore checkmarks from the last
    // session's localStorage view of which rows I'd already gone through.
    applyDoneStateToAllRows();
    setCurrentRows();
    search.addEventListener("input", applyFilters);
    cueSpeakerFilter.addEventListener("change", applyFilters);
    flagFilter.addEventListener("change", applyFilters);
    flaggedOnly.addEventListener("change", applyFilters);
    labelSearch.addEventListener("input", applyLabelFilters);
    labelSpeakerFilter.addEventListener("change", applyLabelFilters);
    labelIssueFilter.addEventListener("change", applyLabelFilters);
    [newTrainingBackend, newTrainingProjectName, newTrainingVersionName].forEach(function (field) {{
      if (!field) return;
      field.addEventListener("input", renderTrainingTargetChoices);
      field.addEventListener("change", renderTrainingTargetChoices);
    }});
    if (showLabelDialogue) showLabelDialogue.addEventListener("change", function () {{
      syncLabelDialogueVisibility();
      scheduleLabelAutoSave();
    }});
    syncLabelDialogueVisibility();
    updateCurrentTimeReadout();
    rewind.addEventListener("click", function () {{
      if (!media) return;
      media.currentTime = Math.max(media.currentTime - 2, 0);
      setCurrentRows();
      drawWaveform();
    }});

    // Big play/pause button next to the other transport controls. The native
    // <audio> player still has its own controls; this just gives a larger
    // hit-target right next to "Back 2 sec" and "Use Current Seconds" so
    // labeling doesn't bounce between two corners of the page.
    const playPauseButton = document.getElementById("playPauseButton");
    function refreshPlayPauseButton() {{
      if (!playPauseButton || !media) return;
      const playing = !media.paused && !media.ended;
      playPauseButton.textContent = playing ? "Pause" : "Play";
      playPauseButton.setAttribute("aria-pressed", playing ? "true" : "false");
      playPauseButton.classList.toggle("is-playing", playing);
    }}
    if (playPauseButton && media) {{
      playPauseButton.addEventListener("click", function () {{
        if (media.paused || media.ended) {{
          const playPromise = media.play();
          if (playPromise && typeof playPromise.catch === "function") {{
            playPromise.catch(function () {{}});
          }}
        }} else {{
          media.pause();
        }}
      }});
      media.addEventListener("play", refreshPlayPauseButton);
      media.addEventListener("playing", refreshPlayPauseButton);
      media.addEventListener("pause", refreshPlayPauseButton);
      media.addEventListener("ended", refreshPlayPauseButton);
      refreshPlayPauseButton();
    }}
  </script>
</body>
</html>
"""


def write_review_bundle(
    *,
    srt_path: Path,
    media_path: Path | None = None,
    output_html: Path | None = None,
    report_tsv: Path | None = None,
    audio_dir: Path = DEFAULT_MEDIA_DIR,
    training_label_records: Mapping[str, Mapping[str, object]] | None = None,
    model_comparisons: Sequence[Mapping[str, object]] | None = None,
    fine_tuning_projects: Sequence[Mapping[str, object]] | None = None,
    quiet: bool = False,
) -> tuple[Path, Path]:
    srt_path = srt_path.expanduser().resolve()
    if not srt_path.is_file():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    output_html = (
        output_html.expanduser().resolve()
        if output_html is not None
        else srt_path.with_name(f"{srt_path.stem}_review.html")
    )
    report_tsv = (
        report_tsv.expanduser().resolve()
        if report_tsv is not None
        else srt_path.with_name(f"{srt_path.stem}_review_flags.tsv")
    )

    cues = parse_srt(srt_path)
    flags_by_index = detect_flags(cues)
    resolved_media = find_matching_media(
        srt_path=srt_path,
        explicit_media=media_path,
        audio_dir=audio_dir,
    )
    label_record: Mapping[str, object] | None = None
    media_selection_name = ""
    if resolved_media is not None and training_label_records:
        try:
            media_selection_name = resolved_media.resolve().relative_to(audio_dir.expanduser().resolve()).as_posix()
        except ValueError:
            media_selection_name = resolved_media.name
        label_record = training_label_records.get(media_selection_name) or training_label_records.get(resolved_media.name)
    elif resolved_media is not None:
        try:
            media_selection_name = resolved_media.resolve().relative_to(audio_dir.expanduser().resolve()).as_posix()
        except ValueError:
            media_selection_name = resolved_media.name

    write_flag_report(cues=cues, flags_by_index=flags_by_index, report_tsv=report_tsv)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(
            cues=cues,
            flags_by_index=flags_by_index,
            srt_path=srt_path,
            media_path=resolved_media,
            output_html=output_html,
            media_selection_name=media_selection_name,
            label_record=label_record,
            model_comparisons=model_comparisons,
            fine_tuning_projects=fine_tuning_projects,
        ),
        encoding="utf-8",
    )

    if not quiet:
        print(f"Review HTML: {output_html}")
        print(f"Flag TSV:    {report_tsv}")
        if resolved_media is not None:
            print(f"Media:       {resolved_media}")
        else:
            print("Media:       no matching media found")

    return output_html, report_tsv


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    write_review_bundle(
        srt_path=Path(args.srt),
        media_path=Path(args.media) if args.media else None,
        output_html=Path(args.output_html) if args.output_html else None,
        report_tsv=Path(args.report_tsv) if args.report_tsv else None,
        audio_dir=Path(args.audio_dir),
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
