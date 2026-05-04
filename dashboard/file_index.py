"""Cross-run file index for the dashboard's "what touched this audio?" search.

Walks ``audio_in/`` plus the diarization, YouTube, and fine-tuning run roots
and groups every file by its *stem* (the audio basename without extension).
Used by the searchable index in the dashboard so the user can type ``017_call``
and see every run, output, and source file that mentions it.

The index is intentionally read-only and bounded — we only crawl directories
the dashboard already exposes via ``DOWNLOADABLE_ROOT_NAMES``-style policy
and stop before descending into per-checkpoint nemo/pyannote folders that
hold thousands of files unrelated to a given audio name.
"""
from __future__ import annotations

import re
from pathlib import Path

# Suffixes we consider file-index-worthy. Mirrors the suffix policy from the
# Phase 1 constants without dragging in the page-route sets.
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm"}
_TRANSCRIPT_SUFFIXES = {".srt", ".rttm", ".vtt", ".txt"}
_REPORT_SUFFIXES = {".tsv", ".csv", ".json", ".html"}
_LOG_SUFFIXES = {".log", ".out", ".err"}

_INDEXABLE_SUFFIXES = (
    _AUDIO_SUFFIXES | _VIDEO_SUFFIXES | _TRANSCRIPT_SUFFIXES | _REPORT_SUFFIXES | _LOG_SUFFIXES
)
_EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "node_modules",
    "experiments",  # NeMo experiment dumps - thousands of unrelated files
    "checkpoints",
    "wandb",
}

# Strip the leading "001_" / "042-" prefix so a stem-search by base name still
# matches the numbered file. Both "017_call" and "call" find "017_call.wav".
_NUMBER_PREFIX = re.compile(r"^\d{2,4}[_-]")


def _classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    if suffix == ".srt":
        return "srt"
    if suffix == ".rttm":
        return "rttm"
    if suffix in {".vtt", ".txt"}:
        return "transcript"
    if suffix == ".html":
        return "review"
    if suffix in {".tsv", ".csv", ".json"}:
        return "report"
    if suffix in _LOG_SUFFIXES:
        return "log"
    return "other"


def _stems(path: Path) -> list[str]:
    """Return one or more searchable stems for a file.

    Always includes ``path.stem``. For double-suffix log files like
    ``017_call.wav.out`` we also strip the trailing audio suffix so the
    log is discoverable under ``017_call`` (and ``call``).

    If the stem starts with a numeric prefix (``017_call``) we also
    return the bare body (``call``). Empty strings are filtered out.
    """

    stems: list[str] = []
    raw = path.stem
    if raw:
        stems.append(raw)
        # Double-suffix stems like "017_call.wav" → also index as "017_call"
        inner_suffix = Path(raw).suffix.lower()
        if inner_suffix and inner_suffix in (_AUDIO_SUFFIXES | _VIDEO_SUFFIXES):
            inner = Path(raw).stem
            if inner and inner != raw:
                stems.append(inner)
        # Strip a leading "001_" / "042-" prefix
        for stem in list(stems):
            bare = _NUMBER_PREFIX.sub("", stem)
            if bare and bare != stem and bare not in stems:
                stems.append(bare)
    return stems


def iter_index_files(roots: list[Path]) -> list[Path]:
    """Walk ``roots`` once and yield every file we want to index.

    Skips dotfiles, ``__pycache__``-style folders, and the runaway
    experiment / checkpoint subtrees.
    """

    seen: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _INDEXABLE_SUFFIXES:
                continue
            if any(part.startswith(".") for part in path.parts):
                continue
            if any(part in _EXCLUDE_DIR_NAMES for part in path.parts):
                continue
            seen.append(path)
    return seen


def build_file_index(
    *,
    audio_dir: Path,
    outputs_root: Path,
    fine_tuning_root: Path,
) -> dict[str, list[dict[str, str]]]:
    """Build a stem → list-of-records index over the dashboard's data folders.

    Each record::

        {
            "name": "017_call.wav",
            "kind": "audio",
            "rel_path": "audio_in/youtube_links/017_call.wav",
            "size": 12345678,
            "run_name": "20260503T052036_diarization_nemo_05-items_63ff",  # if applicable
        }
    """

    files = iter_index_files([audio_dir, outputs_root, fine_tuning_root])
    project_root = audio_dir.parent

    index: dict[str, list[dict[str, str]]] = {}
    for path in files:
        try:
            rel = str(path.relative_to(project_root))
        except ValueError:
            rel = str(path)
        # Run folder is the first ancestor whose grandparent is one of our roots.
        run_name = ""
        for ancestor in path.parents:
            parent = ancestor.parent
            if parent and parent.name in {"diarization_runs", "youtube_runs", "youtube_conversion_runs", "runs"}:
                run_name = ancestor.name
                break
            if parent and parent.parent and parent.parent.name == "diarization_runs":
                run_name = ancestor.name
                break
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        record = {
            "name": path.name,
            "kind": _classify(path),
            "rel_path": rel,
            "size": size,
            "run_name": run_name,
        }
        for stem in _stems(path):
            index.setdefault(stem.lower(), []).append(record)
    return index


def search_index(
    index: dict[str, list[dict[str, str]]],
    query: str,
    *,
    limit: int = 50,
) -> list[dict[str, object]]:
    """Substring + prefix search over the index.

    Results are grouped by stem, returning the best matches first. Stems that
    *start with* the query rank above stems that merely *contain* it.
    """

    needle = (query or "").strip().lower()
    if not needle:
        return []

    prefix_hits: list[tuple[str, list[dict[str, str]]]] = []
    contains_hits: list[tuple[str, list[dict[str, str]]]] = []
    for stem, records in index.items():
        if stem.startswith(needle):
            prefix_hits.append((stem, records))
        elif needle in stem:
            contains_hits.append((stem, records))

    prefix_hits.sort(key=lambda pair: pair[0])
    contains_hits.sort(key=lambda pair: pair[0])

    grouped: list[dict[str, object]] = []
    for stem, records in prefix_hits + contains_hits:
        grouped.append(
            {
                "stem": stem,
                "matches": records,
                "match_count": len(records),
            }
        )
        if len(grouped) >= limit:
            break
    return grouped
