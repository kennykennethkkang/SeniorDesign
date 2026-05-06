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
        label_entries = default_label_entries

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
    player_html = (
        f'<{media_tag} id="media" controls preload="auto" src="{media_href}"></{media_tag}>'
        if media_path is not None
        else "<p class=\"warning\">No matching media file was found, so segment playback is disabled.</p>"
    )
    media_file_name = html.escape(media_selection_name or (media_path.name if media_path is not None else ""), quote=True)
    default_segments = html.escape("\n".join(label_segment_lines))
    default_dialogue = html.escape("\n".join(entry[4] for entry in label_entries))
    include_transcript = not (label_record is not None and label_record.get("include_transcript") is False)
    include_transcript_value = "1" if include_transcript else "0"
    include_transcript_checked = " checked" if include_transcript else ""
    review_workspace_path = html.escape(str(output_html), quote=True)
    selected_backend = normalized_label_backend(label_record)
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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
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
      <div class="waveform-panel">
        <canvas id="waveformCanvas" class="waveform-canvas" aria-label="Audio waveform"></canvas>
        <div class="waveform-tools">
          <span id="waveformStatus" class="waveform-status">Waveform loading...</span>
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
          <div class="field-grid">
            <label><span>Training backend</span><select name="label_backend"><option value="both"{selected_attr("both", selected_backend)}>NeMo + pyannote</option><option value="pyannote"{selected_attr("pyannote", selected_backend)}>pyannote only</option><option value="nemo"{selected_attr("nemo", selected_backend)}>NeMo only</option></select></label>
            <label><span>Project</span><input name="label_project_name" type="text" value="{project_name}"></label>
          </div>
          <div id="labelHealth" class="label-health" aria-live="polite">
            <span class="health-chip"><strong>0</strong> usable labels</span>
            <span class="health-chip"><strong>0</strong> speakers</span>
            <span class="health-chip needs-work"><strong>Check</strong> ready state</span>
          </div>
          <datalist id="speakerOptions">{speaker_options}</datalist>
          <div class="tool-grid">
            <label><span>Search labels</span><input id="labelSearch" type="search" placeholder="Speaker, dialogue, or time"></label>
            <label><span>Speaker</span><select id="labelSpeakerFilter">{speaker_filter_options}</select></label>
            <label><span>Label status</span><select id="labelIssueFilter"><option value="">All labels</option><option value="needs_time">Needs time fix</option><option value="missing_speaker">Missing speaker</option></select></label>
          </div>
          <div class="label-table-toolbar">
            <button type="button" id="addLabelRowTop" class="add-segment-btn">Add Label</button>
            <label class="checkbox-row label-dialogue-toggle"><input id="showLabelDialogue" type="checkbox"{include_transcript_checked}> Save dialogue</label>
            <span class="add-segment-hint">Appends a blank label as the next number. Drag the handle to reorder.</span>
          </div>
          <div class="table-wrap review-table-wrap">
            <table class="label-table">
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
          </div>
        </form>
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
          <button id="rewind" class="secondary" type="button">Back 2 Seconds</button>
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
    const addLabelRow = document.getElementById("addLabelRow");
    const addLabelRowTop = document.getElementById("addLabelRowTop");
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
    let activePlaybackRange = null;
    let waveformPeaks = [];
    let waveformLoaded = false;

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
        window.location.href = base + "/training-labels";
      }});
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

    function formatClock(seconds) {{
      if (!Number.isFinite(seconds)) return "0:00.000";
      const totalMillis = Math.max(Math.round(seconds * 1000), 0);
      const millis = totalMillis % 1000;
      const totalSeconds = Math.floor(totalMillis / 1000);
      const sec = totalSeconds % 60;
      const min = Math.floor(totalSeconds / 60);
      return min + ":" + String(sec).padStart(2, "0") + "." + String(millis).padStart(3, "0");
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
      return speaker + " | " + formatClock(start) + " to " + formatClock(end);
    }}

    function setCurrentRows() {{
      if (!media) return;
      syncLabelSegments(false);
      const time = media.currentTime;
      const labelRowsNow = labelRowList();
      const currentCue = activeRow(cueRows, time);
      const currentLabel = activeRow(labelRowsNow, time);
      cueRows.forEach(function (row) {{
        row.classList.toggle("current", row === currentCue);
      }});
      labelRowsNow.forEach(function (row) {{
        row.classList.toggle("current", row === currentLabel);
      }});
      nowPlaying.innerHTML = "";
      const mainLine = document.createElement("span");
      mainLine.textContent = "Time " + formatClock(time) + " | " + cueSummary(currentCue);
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
    }}

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
    }}

    function useCueAsLabel(cueRow) {{
      const target = selectedOrCurrentLabelRow() || createLabelRow();
      setLabelRowValues(target, cueRow.dataset.start, cueRow.dataset.end, cueRow.dataset.speaker, cueRow.dataset.text);
      setSelectedLabelRow(target);
      syncLabelSegments();
      applyLabelFilters();
      setPlaybackMessage("Added detected segment to labels.", labelSummary(target));
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
    }}

    labelRows.addEventListener("click", function (event) {{
      const row = event.target.closest("[data-label-row]");
      if (!row) return;
      if (event.target.closest("button.delete-label")) {{
        event.stopPropagation();
        deleteLabelRow(row);
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
      if (row) setSelectedLabelRow(row);
    }});

    labelRows.addEventListener("input", function () {{
      syncLabelSegments();
      applyLabelFilters();
      setCurrentRows();
    }});
    labelRows.addEventListener("change", function () {{
      syncLabelSegments();
      applyLabelFilters();
      setCurrentRows();
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
    }}

    addLabelRow.addEventListener("click", appendBlankLabelRow);
    if (addLabelRowTop) addLabelRowTop.addEventListener("click", appendBlankLabelRow);

    function setLabelFormBusy(isBusy) {{
      labelForm.dataset.submitting = isBusy ? "true" : "";
      labelForm.setAttribute("aria-busy", isBusy ? "true" : "false");
      labelForm.querySelectorAll("button[type='submit']").forEach(function (button) {{
        button.disabled = isBusy;
      }});
    }}

    labelForm.addEventListener("submit", function (event) {{
      syncLabelSegments(true);
      if (!window.fetch || labelForm.dataset.submitting === "true") {{
        return;
      }}
      event.preventDefault();
      const submitter = event.submitter;
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
        event.formData.set("label_segments", labelSegments.value);
      }});
    }}

    if (media) {{
      media.load();
      media.addEventListener("timeupdate", setCurrentRows);
      media.addEventListener("timeupdate", drawWaveform);
      media.addEventListener("seeked", setCurrentRows);
      media.addEventListener("seeked", drawWaveform);
      media.addEventListener("pause", setCurrentRows);
      media.addEventListener("loadedmetadata", drawWaveform);
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
      window.addEventListener("resize", drawWaveform);
      loadWaveform();
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
    setCurrentRows();
    search.addEventListener("input", applyFilters);
    cueSpeakerFilter.addEventListener("change", applyFilters);
    flagFilter.addEventListener("change", applyFilters);
    flaggedOnly.addEventListener("change", applyFilters);
    labelSearch.addEventListener("input", applyLabelFilters);
    labelSpeakerFilter.addEventListener("change", applyLabelFilters);
    labelIssueFilter.addEventListener("change", applyLabelFilters);
    if (showLabelDialogue) showLabelDialogue.addEventListener("change", syncLabelDialogueVisibility);
    syncLabelDialogueVisibility();
    rewind.addEventListener("click", function () {{
      if (!media) return;
      media.currentTime = Math.max(media.currentTime - 2, 0);
      setCurrentRows();
      drawWaveform();
    }});
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
