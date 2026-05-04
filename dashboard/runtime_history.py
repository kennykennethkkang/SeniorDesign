"""Time-estimate engine driven by ``logs/runtime_summary.tsv`` history.

Walks every site-launched diarization run and computes a rolling
``wallclock_seconds / audio_seconds`` ratio per backend (nemo, pyannote, ...).
Audio durations are probed once with ``fine_tuning_manager.probe_media_duration``
and cached under ``.local_dashboard/audio_durations.json`` so we do not pay
that cost on every request.

The estimate function multiplies the rolling ratio by the total audio
seconds of the selected files. When we have fewer than three completed runs
on a backend we deliberately return ``available=False`` rather than bluff a
number from a single sample.
"""
from __future__ import annotations

import csv
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Iterable

from fine_tuning_manager import probe_media_duration


_MIN_HISTORICAL_FILES = 3
_RECENT_RUNS_FOR_AVERAGE = 8
_DURATION_CACHE_LOCK = threading.Lock()


def _cache_path(local_dashboard_dir: Path) -> Path:
    return local_dashboard_dir / "audio_durations.json"


def load_duration_cache(local_dashboard_dir: Path) -> dict[str, float]:
    """Load the persistent file-mtime → audio-duration cache."""

    path = _cache_path(local_dashboard_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): float(value) for key, value in data.items() if isinstance(value, (int, float))}


def save_duration_cache(local_dashboard_dir: Path, cache: dict[str, float]) -> None:
    """Persist the audio duration cache (best-effort)."""

    path = _cache_path(local_dashboard_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        # Cache is purely an optimization; failure to persist is non-fatal.
        return


def cache_key_for_file(audio_path: Path) -> str | None:
    """Build a stable cache key combining absolute path + size + mtime.

    Returns ``None`` when the file does not exist so callers can skip it.
    """

    try:
        stat = audio_path.stat()
    except OSError:
        return None
    return f"{audio_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"


def get_audio_duration(
    audio_path: Path,
    *,
    local_dashboard_dir: Path,
    cache: dict[str, float] | None = None,
) -> float | None:
    """Return audio duration in seconds, using the persistent cache when possible.

    ``cache`` lets a caller avoid repeated disk loads when probing many files.
    Mutates ``cache`` in-place; the caller is responsible for ``save_duration_cache``.
    """

    if cache is None:
        with _DURATION_CACHE_LOCK:
            cache = load_duration_cache(local_dashboard_dir)
    key = cache_key_for_file(audio_path)
    if key is None:
        return None
    if key in cache:
        return cache[key]
    duration = probe_media_duration(audio_path)
    if duration is None or duration <= 0:
        return None
    cache[key] = float(duration)
    return float(duration)


def runtime_summary_rows(run_dir: Path) -> list[dict[str, str]]:
    """Read ``logs/runtime_summary.tsv`` for one run, returning one dict per file."""

    summary_path = run_dir / "logs" / "runtime_summary.tsv"
    if not summary_path.is_file():
        return []
    rows: list[dict[str, str]] = []
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rows.append({key: (value or "").strip() for key, value in row.items()})
    except OSError:
        return []
    return rows


def _backend_for_run(run_dir: Path) -> str:
    """Infer the backend label from the run directory layout.

    Site diarization runs live under ``outputs/diarization_runs/<backend>/<run>``,
    so the parent directory name is the backend.
    """

    parent = run_dir.parent
    if parent and parent.name and parent.name not in {"diarization_runs", "outputs"}:
        return parent.name
    name = run_dir.name
    for backend in ("nemo", "pyannote"):
        if f"_{backend}_" in name or name.startswith(backend) or name.endswith(backend):
            return backend
    return "unknown"


def gather_historical_rates(
    diarization_runs_root: Path,
    *,
    audio_dir: Path,
    local_dashboard_dir: Path,
) -> dict[str, dict[str, object]]:
    """Compute rolling wallclock/audio_seconds rate per backend.

    Walks every directory under ``diarization_runs_root`` two levels deep
    (``<backend>/<run>``) and the legacy single-level layout. Returns::

        {
            "nemo": {
                "available": True,
                "ratio": 0.42,
                "sample_count": 24,
                "runs_used": 5,
                "last_run": "20260503T052036_diarization_nemo_05-items_63ff",
                "updated_at": 1715720000,
            },
            "pyannote": { ... },
        }
    """

    if not diarization_runs_root.is_dir():
        return {}

    # Discover run directories (two layouts: <root>/<backend>/<run> or <root>/<run>).
    candidates: list[Path] = []
    for top in diarization_runs_root.iterdir():
        if not top.is_dir():
            continue
        if (top / "logs" / "runtime_summary.tsv").is_file():
            candidates.append(top)
            continue
        for nested in top.iterdir():
            if nested.is_dir() and (nested / "logs" / "runtime_summary.tsv").is_file():
                candidates.append(nested)

    by_backend: dict[str, list[tuple[Path, list[dict[str, str]]]]] = {}
    for run_dir in candidates:
        rows = runtime_summary_rows(run_dir)
        if not rows:
            continue
        by_backend.setdefault(_backend_for_run(run_dir), []).append((run_dir, rows))

    if not by_backend:
        return {}

    with _DURATION_CACHE_LOCK:
        cache = load_duration_cache(local_dashboard_dir)
        cache_dirty = False

        result: dict[str, dict[str, object]] = {}
        for backend, run_rows in by_backend.items():
            run_rows.sort(key=lambda pair: pair[0].name, reverse=True)
            recent = run_rows[:_RECENT_RUNS_FOR_AVERAGE]
            ratios: list[float] = []
            for run_dir, rows in recent:
                for row in rows:
                    if (row.get("status") or "").strip() != "ok":
                        continue
                    try:
                        wallclock = float(row.get("runtime_seconds") or 0)
                    except ValueError:
                        continue
                    if wallclock <= 0:
                        continue
                    audio_rel = (row.get("audio_file") or "").strip()
                    if not audio_rel:
                        continue
                    audio_path = (audio_dir / audio_rel).resolve()
                    duration = get_audio_duration(
                        audio_path,
                        local_dashboard_dir=local_dashboard_dir,
                        cache=cache,
                    )
                    if duration is None or duration <= 0:
                        continue
                    if cache_key_for_file(audio_path) and cache_key_for_file(audio_path) in cache:
                        cache_dirty = True
                    ratios.append(wallclock / duration)
            if len(ratios) < _MIN_HISTORICAL_FILES:
                result[backend] = {
                    "available": False,
                    "sample_count": len(ratios),
                    "runs_used": len(recent),
                    "message": f"Need at least {_MIN_HISTORICAL_FILES} historical files; have {len(ratios)}.",
                }
                continue
            result[backend] = {
                "available": True,
                "ratio": statistics.fmean(ratios),
                "ratio_median": statistics.median(ratios),
                "sample_count": len(ratios),
                "runs_used": len(recent),
                "last_run": recent[0][0].name,
                "updated_at": int(time.time()),
            }

        if cache_dirty:
            save_duration_cache(local_dashboard_dir, cache)

    return result


def estimate_runtime_for_files(
    audio_paths: Iterable[Path],
    *,
    backend: str,
    rates: dict[str, dict[str, object]],
    local_dashboard_dir: Path,
) -> dict[str, object]:
    """Estimate wallclock seconds for diarizing ``audio_paths`` on ``backend``.

    Returns a payload like::

        {
            "available": True,
            "files": 12,
            "total_audio_seconds": 3120.4,
            "ratio": 0.42,
            "estimate_seconds": 1310.6,
            "estimate_label": "~22 min",
            "based_on_runs": 5,
            "based_on_files": 24,
        }
    """

    rate = rates.get(backend) or {}
    paths = [Path(p) for p in audio_paths]
    if not paths:
        return {
            "available": False,
            "message": "No files selected.",
            "files": 0,
        }
    if not rate.get("available"):
        return {
            "available": False,
            "message": (rate.get("message") or f"No {backend} runtime history yet."),
            "files": len(paths),
        }
    with _DURATION_CACHE_LOCK:
        cache = load_duration_cache(local_dashboard_dir)
        cache_dirty = False
        total_duration = 0.0
        unknown = 0
        for audio_path in paths:
            duration = get_audio_duration(
                audio_path,
                local_dashboard_dir=local_dashboard_dir,
                cache=cache,
            )
            if duration is None:
                unknown += 1
                continue
            total_duration += duration
            cache_dirty = True
        if cache_dirty:
            save_duration_cache(local_dashboard_dir, cache)
    if total_duration <= 0:
        return {
            "available": False,
            "message": "Could not probe audio duration for any selected file.",
            "files": len(paths),
            "unknown_durations": unknown,
        }
    ratio = float(rate["ratio"])
    estimate_seconds = total_duration * ratio
    return {
        "available": True,
        "files": len(paths),
        "unknown_durations": unknown,
        "total_audio_seconds": round(total_duration, 1),
        "ratio": round(ratio, 3),
        "ratio_median": round(float(rate.get("ratio_median") or ratio), 3),
        "estimate_seconds": round(estimate_seconds, 1),
        "estimate_label": format_seconds_human(estimate_seconds),
        "based_on_runs": int(rate.get("runs_used") or 0),
        "based_on_files": int(rate.get("sample_count") or 0),
        "last_run": rate.get("last_run", ""),
    }


def format_seconds_human(seconds: float) -> str:
    """Human-friendly duration label (~3 min, ~1 h 12 min, ~7 s)."""

    if seconds <= 0:
        return "~0 s"
    if seconds < 60:
        return f"~{int(round(seconds))} s"
    if seconds < 3600:
        minutes = round(seconds / 60)
        return f"~{int(minutes)} min"
    hours = int(seconds // 3600)
    minutes = round((seconds - hours * 3600) / 60)
    if minutes == 60:
        return f"~{hours + 1} h"
    if minutes == 0:
        return f"~{hours} h"
    return f"~{hours} h {minutes} min"
