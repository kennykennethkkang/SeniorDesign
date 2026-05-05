#!/usr/bin/env python3
"""Prepare and launch diarization fine-tuning artifacts for the SeniorDesign workflow.

This module now supports both training backends exposed by the site:

- NeMo MSDD fine-tuning, which stays aligned with the existing cluster workflow.
- pyannote speaker-segmentation fine-tuning, following the official tutorial pattern
  built around `Model.from_pretrained`, `SpeakerDiarization`, and a database config.

The implementation still follows the local WAVE guidance in
`docs/reference_materials/CSEN 240_ Utilizing SCU's WAVE HPC via SLURM (1).pdf`, which emphasizes
Slurm submission for shared-cluster workloads.

Two local papers in `docs/reference_materials/` informed the fine-tuning assumptions:
`2504.18582v1.pdf` reinforced the importance of transfer learning for low-resource
speech settings, while `2302.10924v1.pdf` provided additional context on why robust
speaker segmentation and adaptation matter for real-world diarization systems.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from audio_numbering import AUDIO_EXTENSIONS
from workflow_background import utc_now_iso

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAME = "msdd_5scl_15_05_50Povl_256x3x32x2.yaml"
DEFAULT_SPEAKER_MODEL = "titanet_large"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_BASE_WINDOW = 0.5
DEFAULT_BASE_SHIFT = 0.25
DEFAULT_STEP_COUNT = 50
DEFAULT_MAX_EPOCHS = 20
DEFAULT_DEVICES = 1
DEFAULT_SLURM_PARTITION = "gpu"
DEFAULT_SLURM_TIME = "08:00:00"
DEFAULT_SLURM_MEMORY = "48G"
DEFAULT_SLURM_CPUS = 8
DEFAULT_SLURM_GPUS = 1
DEFAULT_FINE_TUNING_BACKEND = "nemo"
SUPPORTED_FINE_TUNING_BACKENDS = {"nemo", "pyannote"}
DEFAULT_PYANNOTE_PRETRAINED_MODEL = "pyannote/segmentation-3.0"
DEFAULT_PYANNOTE_DURATION = 10.0
DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK = 3
DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME = 2


@dataclass(frozen=True)
class RttmSegment:
    """Represent one RTTM speaker region after validation and normalization."""

    session_id: str
    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class TrainingSample:
    """Bundle the per-sample information reused throughout preparation.

    Keeping parsed RTTM segments and transcript text here avoids repeating disk
    reads in the manifest generation phase.
    """

    stem: str
    audio_path: Path
    rttm_path: Path
    transcript_path: Path | None
    transcript_text: str
    duration_seconds: float
    speakers: tuple[str, ...]
    segments: tuple[RttmSegment, ...]

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)


@dataclass(frozen=True)
class FineTuneArtifacts:
    """Describe the files emitted by one `prepare_project` run."""

    backend: str
    project_slug: str
    project_dir: Path
    sample_count: int
    train_count: int
    validation_count: int
    session_manifest_train: Path | None
    session_manifest_validation: Path | None
    msdd_manifest_train: Path | None
    msdd_manifest_validation: Path | None
    metadata_path: Path
    launch_script_path: Path
    sbatch_script_path: Path
    warnings: tuple[str, ...]
    primary_artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class FineTuneRun:
    """Describe one launch attempt, whether local or submitted via Slurm."""

    run_dir: Path
    stdout_path: Path
    stderr_path: Path
    metadata_path: Path
    exit_code_path: Path
    pid: int
    job_id: str = ""
    version_name: str = ""
    version_slug: str = ""
    version_number: int = 0
    experiment_dir: Path | None = None


def slugify(value: str) -> str:
    """Normalize user-facing names into directory-safe project identifiers."""

    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    return cleaned or "project"


def unique_child_path(parent: Path, slug: str) -> Path:
    """Return a non-existing child path by appending a numeric suffix if needed."""

    candidate = parent / slug
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = parent / f"{slug}-{index}"
        if not candidate.exists():
            return candidate
        index += 1


def sanitize_filename(value: str) -> str:
    """Drop path components and keep only a conservative filename alphabet."""

    name = Path(value or "").name
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    safe = safe.strip("._")
    return safe or "upload.bin"


def normalize_backend(backend: str | None) -> str:
    """Normalize the backend label used throughout the fine-tuning workspace."""

    normalized = (backend or DEFAULT_FINE_TUNING_BACKEND).strip().lower()
    if normalized not in SUPPORTED_FINE_TUNING_BACKENDS:
        raise ValueError(f"Unsupported fine-tuning backend: {backend}")
    return normalized


# Friendly display names for projects and runs live in a sidecar display.json
# next to the (regenerated-on-prepare) metadata.json. Keeping it separate means
# prepare_project can wipe metadata.json without losing the user's chosen label.
def _read_display_sidecar(path: Path) -> dict[str, object]:
    """Return parsed display.json, or {} if missing/corrupt."""

    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_display_sidecar(path: Path, *, updates: dict[str, object]) -> dict[str, object]:
    """Merge ``updates`` into the display.json at ``path`` and return the result."""

    payload = _read_display_sidecar(path)
    payload.update(updates)
    payload = {k: v for k, v in payload.items() if v not in ("", None)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def project_display_path(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the path to the display sidecar for a project."""

    return project_dir(project_name, backend=backend, root=root) / "display.json"


def read_project_display(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Read the display sidecar for a project (e.g. its renamed display label)."""

    return _read_display_sidecar(project_display_path(project_name, backend=backend, root=root))


def set_project_display_name(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    display_name: str,
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Rename a project for display purposes (the slug/dir stays put)."""

    cleaned = (display_name or "").strip()
    if not cleaned:
        raise ValueError("display_name must be a non-empty string.")
    target = project_dir(project_name, backend=backend, root=root)
    if not target.is_dir():
        raise FileNotFoundError(f"Unknown fine-tuning project: {project_name}")
    return _write_display_sidecar(target / "display.json", updates={"display_name": cleaned})


def read_run_display(run_dir: Path) -> dict[str, object]:
    """Read the display sidecar for a single training run."""

    return _read_display_sidecar(run_dir / "display.json")


def set_run_display_name(run_dir: Path, *, display_name: str) -> dict[str, object]:
    """Rename a single run for display purposes (the run dir stays put)."""

    cleaned = (display_name or "").strip()
    if not cleaned:
        raise ValueError("display_name must be a non-empty string.")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Unknown fine-tuning run: {run_dir}")
    return _write_display_sidecar(run_dir / "display.json", updates={"display_name": cleaned})


def project_dir(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the root directory reserved for a named fine-tuning project."""

    normalized_backend = normalize_backend(backend)
    return root / "fine_tuning" / "projects" / normalized_backend / slugify(project_name)


def project_paths(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Collect all stable project paths in one place to reduce path drift."""

    normalized_backend = normalize_backend(backend)
    base_dir = project_dir(project_name, backend=normalized_backend, root=root)
    return {
        "backend_dir": base_dir.parent,
        "project_dir": base_dir,
        "audio_dir": base_dir / "audio",
        "rttm_dir": base_dir / "rttm",
        "text_dir": base_dir / "text",
        "artifacts_dir": base_dir / "artifacts",
        "manifests_dir": base_dir / "artifacts" / "manifests",
        "pairwise_train_dir": base_dir / "artifacts" / "pairwise_rttm" / "train",
        "pairwise_validation_dir": base_dir / "artifacts" / "pairwise_rttm" / "validation",
        "emb_train_dir": base_dir / "artifacts" / "embeddings" / "train",
        "emb_validation_dir": base_dir / "artifacts" / "embeddings" / "validation",
        "experiments_dir": base_dir / "artifacts" / "experiments",
        "slurm_logs_dir": base_dir / "artifacts" / "slurm_logs",
        "runs_dir": base_dir / "runs",
        "metadata_path": base_dir / "artifacts" / "metadata.json",
        "launch_script_path": base_dir / "artifacts" / f"launch_{normalized_backend}_finetune.sh",
        "sbatch_script_path": base_dir / "artifacts" / f"launch_{normalized_backend}_finetune.sbatch",
        "database_config_path": base_dir / "artifacts" / "database.yml",
        "train_list_path": base_dir / "artifacts" / "lists" / "train.lst",
        "development_list_path": base_dir / "artifacts" / "lists" / "development.lst",
        "test_list_path": base_dir / "artifacts" / "lists" / "test.lst",
        "train_rttm_path": base_dir / "artifacts" / "rttm" / "train.rttm",
        "development_rttm_path": base_dir / "artifacts" / "rttm" / "development.rttm",
        "test_rttm_path": base_dir / "artifacts" / "rttm" / "test.rttm",
        "train_uem_path": base_dir / "artifacts" / "uem" / "train.uem",
        "development_uem_path": base_dir / "artifacts" / "uem" / "development.uem",
        "test_uem_path": base_dir / "artifacts" / "uem" / "test.uem",
        "training_script_path": base_dir / "artifacts" / f"train_{normalized_backend}.py",
    }


def ensure_project_structure(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Create the directory structure required by uploads, manifests, and runs."""

    paths = project_paths(project_name, backend=backend, root=root)
    for key, path in paths.items():
        if key.endswith("_path"):
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    return paths


def save_project_sample(
    *,
    project_name: str,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    audio_name: str,
    audio_bytes: bytes,
    rttm_name: str,
    rttm_bytes: bytes,
    transcript_name: str | None = None,
    transcript_bytes: bytes | None = None,
    transcript_text: str | None = None,
    root: Path = PROJECT_ROOT,
) -> TrainingSample:
    """Persist an uploaded training sample and return its parsed metadata.

    The upload contract mirrors the NeMo dataset assumptions: one audio file is
    paired with one RTTM file that shares the same logical stem, and transcript
    text is optional but retained when available for downstream inspection.
    """

    if not audio_bytes:
        raise ValueError("Audio upload is empty.")
    if not rttm_bytes:
        raise ValueError("RTTM upload is empty.")

    normalized_backend = normalize_backend(backend)
    paths = ensure_project_structure(project_name, backend=normalized_backend, root=root)
    audio_filename = sanitize_filename(audio_name)
    rttm_filename = sanitize_filename(rttm_name)
    audio_suffix = Path(audio_filename).suffix.lower()
    if audio_suffix not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported training audio type: {audio_suffix or 'unknown'}")

    stem = Path(audio_filename).stem or Path(rttm_filename).stem
    if not stem:
        raise ValueError("Unable to derive a sample name from the uploaded files.")

    audio_path = paths["audio_dir"] / f"{stem}{audio_suffix}"
    rttm_path = paths["rttm_dir"] / f"{stem}.rttm"
    audio_path.write_bytes(audio_bytes)
    rttm_path.write_bytes(rttm_bytes)

    transcript_path: Path | None = None
    if transcript_bytes:
        transcript_filename = sanitize_filename(transcript_name or f"{stem}.txt")
        transcript_suffix = Path(transcript_filename).suffix or ".txt"
        transcript_path = paths["text_dir"] / f"{stem}{transcript_suffix}"
        transcript_path.write_bytes(transcript_bytes)
    elif transcript_text and transcript_text.strip():
        transcript_path = paths["text_dir"] / f"{stem}.txt"
        transcript_path.write_text(transcript_text.strip() + "\n", encoding="utf-8")

    return build_sample(audio_path, rttm_path, transcript_path)


def save_project_sample_streams(
    *,
    project_name: str,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    audio_name: str,
    audio_stream,
    rttm_name: str,
    rttm_stream,
    transcript_name: str | None = None,
    transcript_stream=None,
    transcript_text: str | None = None,
    root: Path = PROJECT_ROOT,
) -> TrainingSample:
    """Persist a training sample by streaming uploaded files directly to disk.

    This avoids loading large media uploads fully into memory before the file is
    written, which is preferable for the browser-based dashboard path.
    """

    normalized_backend = normalize_backend(backend)
    paths = ensure_project_structure(project_name, backend=normalized_backend, root=root)
    audio_filename = sanitize_filename(audio_name)
    rttm_filename = sanitize_filename(rttm_name)
    audio_suffix = Path(audio_filename).suffix.lower()
    if audio_suffix not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported training audio type: {audio_suffix or 'unknown'}")

    stem = Path(audio_filename).stem or Path(rttm_filename).stem
    if not stem:
        raise ValueError("Unable to derive a sample name from the uploaded files.")

    audio_path = paths["audio_dir"] / f"{stem}{audio_suffix}"
    rttm_path = paths["rttm_dir"] / f"{stem}.rttm"
    with audio_path.open("wb") as handle:
        shutil.copyfileobj(audio_stream, handle)
    with rttm_path.open("wb") as handle:
        shutil.copyfileobj(rttm_stream, handle)

    if audio_path.stat().st_size == 0:
        raise ValueError("Audio upload is empty.")
    if rttm_path.stat().st_size == 0:
        raise ValueError("RTTM upload is empty.")

    transcript_path: Path | None = None
    if transcript_stream is not None and transcript_name:
        transcript_filename = sanitize_filename(transcript_name or f"{stem}.txt")
        transcript_suffix = Path(transcript_filename).suffix or ".txt"
        transcript_path = paths["text_dir"] / f"{stem}{transcript_suffix}"
        with transcript_path.open("wb") as handle:
            shutil.copyfileobj(transcript_stream, handle)
        if transcript_path.stat().st_size == 0:
            transcript_path.unlink(missing_ok=True)
            transcript_path = None
    elif transcript_text and transcript_text.strip():
        transcript_path = paths["text_dir"] / f"{stem}.txt"
        transcript_path.write_text(transcript_text.strip() + "\n", encoding="utf-8")

    return build_sample(audio_path, rttm_path, transcript_path)


def parse_rttm(rttm_path: Path) -> list[RttmSegment]:
    """Parse valid RTTM speaker rows into a normalized in-memory structure."""

    segments: list[RttmSegment] = []
    for raw_line in rttm_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0].upper() != "SPEAKER":
            continue
        try:
            start = float(parts[3])
            duration = float(parts[4])
        except ValueError:
            continue
        if duration <= 0:
            continue
        segments.append(
            RttmSegment(
                session_id=parts[1],
                start=max(start, 0.0),
                duration=duration,
                speaker=parts[7],
            )
        )
    segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker))
    return segments


def probe_media_duration(media_path: Path) -> float:
    """Estimate media duration with ffprobe first and WAV parsing as fallback."""

    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        completed = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            try:
                duration = float(completed.stdout.strip())
            except ValueError:
                duration = 0.0
            if duration > 0:
                return duration

    if media_path.suffix.lower() == ".wav":
        with wave.open(str(media_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        if frame_rate > 0:
            return frame_count / frame_rate

    raise RuntimeError(
        f"Unable to determine duration for {media_path}. Install ffprobe or use WAV files."
    )


def read_transcript_text(path: Path | None) -> str:
    """Return a transcript payload compatible with NeMo manifest expectations."""

    if path is None or not path.is_file():
        return "-"
    text = path.read_text(encoding="utf-8").strip()
    return text or "-"


def build_sample(
    audio_path: Path,
    rttm_path: Path,
    transcript_path: Path | None = None,
) -> TrainingSample:
    """Validate one sample and precompute metadata reused in later phases."""

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not rttm_path.is_file():
        raise FileNotFoundError(f"RTTM file not found: {rttm_path}")

    segments = parse_rttm(rttm_path)
    if not segments:
        raise ValueError(f"RTTM file does not contain valid speaker segments: {rttm_path}")

    transcript_text = read_transcript_text(transcript_path)
    speakers = tuple(sorted({segment.speaker for segment in segments}))
    duration_seconds = max(probe_media_duration(audio_path), max(segment.end for segment in segments))
    return TrainingSample(
        stem=audio_path.stem,
        audio_path=audio_path.resolve(),
        rttm_path=rttm_path.resolve(),
        transcript_path=transcript_path.resolve() if transcript_path and transcript_path.is_file() else None,
        transcript_text=transcript_text,
        duration_seconds=round(duration_seconds, 3),
        speakers=speakers,
        segments=tuple(segments),
    )


def discover_samples(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> list[TrainingSample]:
    """Load every valid sample that has both audio and RTTM supervision."""

    normalized_backend = normalize_backend(backend)
    paths = ensure_project_structure(project_name, backend=normalized_backend, root=root)
    samples: list[TrainingSample] = []
    for audio_path in sorted(paths["audio_dir"].iterdir(), key=lambda item: item.name.lower()):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        rttm_path = paths["rttm_dir"] / f"{audio_path.stem}.rttm"
        if not rttm_path.is_file():
            raise FileNotFoundError(
                f"Missing RTTM for training sample '{audio_path.stem}': {rttm_path}"
            )
        transcript_path = next(
            (
                candidate
                for candidate in sorted(paths["text_dir"].glob(f"{audio_path.stem}.*"))
                if candidate.is_file()
            ),
            None,
        )
        samples.append(build_sample(audio_path, rttm_path, transcript_path))
    return samples


def split_samples(
    samples: Sequence[TrainingSample],
    train_ratio: float,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    """Create a deterministic train/validation split from the available samples."""

    if not samples:
        raise ValueError("At least one training sample is required.")
    ordered = sorted(samples, key=lambda sample: sample.stem.lower())
    if len(ordered) == 1:
        return [ordered[0]], [ordered[0]]

    ratio = min(max(train_ratio, 0.1), 0.9)
    train_count = round(len(ordered) * ratio)
    train_count = min(max(train_count, 1), len(ordered) - 1)
    return ordered[:train_count], ordered[train_count:]


def session_manifest_row(sample: TrainingSample) -> dict[str, object]:
    """Map one validated sample into the session-level NeMo manifest format."""

    return {
        "audio_filepath": str(sample.audio_path),
        "offset": 0,
        "duration": sample.duration_seconds,
        "label": "infer",
        "text": sample.transcript_text,
        "num_speakers": sample.num_speakers,
        "rttm_filepath": str(sample.rttm_path),
        "uem_filepath": None,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write newline-delimited JSON because NeMo consumes manifest files in JSONL form."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_path_list(path: Path, values: Iterable[Path]) -> None:
    """Write one absolute path per line for auxiliary debugging and reproducibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def write_pairwise_rttm(
    *,
    source_segments: Sequence[RttmSegment],
    target_path: Path,
    speakers: tuple[str, str] | tuple[str, ...],
) -> None:
    """Write a two-speaker RTTM file for one speaker pairing.

    NeMo MSDD training expects pairwise supervision when sessions contain more
    than two speakers, so this helper materializes the reduced RTTM view.
    """

    speakers_set = set(speakers)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        for segment in source_segments:
            if segment.speaker not in speakers_set:
                continue
            handle.write(
                " ".join(
                    [
                        "SPEAKER",
                        segment.session_id,
                        "1",
                        f"{segment.start:.3f}",
                        f"{segment.duration:.3f}",
                        "<NA>",
                        "<NA>",
                        segment.speaker,
                        "<NA>",
                        "<NA>",
                    ]
                )
                + "\n"
            )


def merged_active_ranges(segments: Sequence[RttmSegment], *, gap_tolerance: float) -> list[tuple[float, float]]:
    """Merge nearby speech spans so training windows are not split too aggressively."""

    if not segments:
        return []
    ranges: list[tuple[float, float]] = []
    current_start = segments[0].start
    current_end = segments[0].end
    for segment in segments[1:]:
        if segment.start <= current_end + gap_tolerance:
            current_end = max(current_end, segment.end)
            continue
        ranges.append((current_start, current_end))
        current_start = segment.start
        current_end = segment.end
    ranges.append((current_start, current_end))
    return ranges


def build_msdd_rows_for_sample(
    *,
    sample: TrainingSample,
    pairwise_dir: Path,
    base_window: float,
    base_shift: float,
    step_count: int,
) -> tuple[list[dict[str, object]], list[str]]:
    """Generate MSDD window rows for one sample.

    The resulting windows are bounded by active speech regions and respect the
    base-scale window and shift parameters documented in the NeMo diarization
    dataset guide.
    """

    warnings: list[str] = []
    if sample.num_speakers < 2:
        warnings.append(
            f"Sample '{sample.stem}' has fewer than 2 speakers and was excluded from the MSDD manifest."
        )
        return [], warnings

    sample_span = base_window + (base_shift * max(step_count - 1, 0))
    stride = max(sample_span - base_shift, base_shift)
    speaker_sets = (
        list(combinations(sample.speakers, 2))
        if sample.num_speakers > 2
        else [tuple(sample.speakers)]
    )

    rows: list[dict[str, object]] = []
    transcript_text = sample.transcript_text
    for speaker_pair in speaker_sets:
        speaker_pair_set = set(speaker_pair)
        filtered_segments = [
            segment for segment in sample.segments if segment.speaker in speaker_pair_set
        ]
        if not filtered_segments:
            continue

        pair_label = "_".join(speaker_pair)
        pairwise_path = pairwise_dir / f"{sample.stem}.{pair_label}.rttm"
        write_pairwise_rttm(
            source_segments=filtered_segments,
            target_path=pairwise_path,
            speakers=speaker_pair,
        )

        for active_start, active_end in merged_active_ranges(
            filtered_segments,
            gap_tolerance=base_shift,
        ):
            cursor = max(active_start, 0.0)
            while cursor < active_end:
                duration = min(sample_span, active_end - cursor)
                if duration <= 0:
                    break
                rows.append(
                    {
                        "audio_filepath": str(sample.audio_path),
                        "offset": round(cursor, 3),
                        "duration": round(duration, 3),
                        "label": "infer",
                        "text": transcript_text,
                        "num_speakers": len(speaker_pair),
                        "rttm_filepath": str(pairwise_path),
                        "uem_filepath": None,
                    }
                )
                if active_end - cursor <= sample_span:
                    break
                cursor += stride

    if not rows:
        warnings.append(
            f"Sample '{sample.stem}' did not yield MSDD training windows and was skipped."
        )
    return rows, warnings


def write_uri_list(path: Path, samples: Sequence[TrainingSample]) -> None:
    """Write one sample stem per line for pyannote database subsets."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(f"{sample.stem}\n")


def write_pyannote_subset_rttm(path: Path, samples: Sequence[TrainingSample]) -> None:
    """Concatenate sample RTTMs while forcing session IDs to match the sample stem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            for segment in sample.segments:
                handle.write(
                    " ".join(
                        [
                            "SPEAKER",
                            sample.stem,
                            "1",
                            f"{segment.start:.3f}",
                            f"{segment.duration:.3f}",
                            "<NA>",
                            "<NA>",
                            segment.speaker,
                            "<NA>",
                            "<NA>",
                        ]
                    )
                    + "\n"
                )


def write_pyannote_subset_uem(path: Path, samples: Sequence[TrainingSample]) -> None:
    """Write one full-file annotated range per sample for pyannote training."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(f"{sample.stem} NA 0.000 {sample.duration_seconds:.3f}\n")


def write_pyannote_database_config(
    *,
    database_config_path: Path,
    project_name: str,
    project_paths_map: dict[str, Path],
    root: Path,
) -> str:
    """Generate the pyannote.database configuration used by the training script."""

    project_slug = slugify(project_name)
    protocol_name = f"SeniorDesign.SpeakerDiarization.{project_slug}"
    audio_patterns = "\n".join(
        f"    - {json.dumps(str(project_paths_map['audio_dir'] / f'{{uri}}{extension}'))}"
        for extension in sorted(AUDIO_EXTENSIONS)
    )
    config = (
        "Databases:\n"
        "  SeniorDesign:\n"
        f"{audio_patterns}\n"
        "\n"
        "Protocols:\n"
        "  SeniorDesign:\n"
        "    SpeakerDiarization:\n"
        f"      {project_slug}:\n"
        "        scope: file\n"
        "        train:\n"
        f"          uri: {json.dumps(str(project_paths_map['train_list_path'].relative_to(root)))}\n"
        f"          annotation: {json.dumps(str(project_paths_map['train_rttm_path'].relative_to(root)))}\n"
        f"          annotated: {json.dumps(str(project_paths_map['train_uem_path'].relative_to(root)))}\n"
        "        development:\n"
        f"          uri: {json.dumps(str(project_paths_map['development_list_path'].relative_to(root)))}\n"
        f"          annotation: {json.dumps(str(project_paths_map['development_rttm_path'].relative_to(root)))}\n"
        f"          annotated: {json.dumps(str(project_paths_map['development_uem_path'].relative_to(root)))}\n"
        "        test:\n"
        f"          uri: {json.dumps(str(project_paths_map['test_list_path'].relative_to(root)))}\n"
        f"          annotation: {json.dumps(str(project_paths_map['test_rttm_path'].relative_to(root)))}\n"
        f"          annotated: {json.dumps(str(project_paths_map['test_uem_path'].relative_to(root)))}\n"
    )
    database_config_path.parent.mkdir(parents=True, exist_ok=True)
    database_config_path.write_text(config, encoding="utf-8")
    return protocol_name


def write_pyannote_training_script(
    *,
    training_script_path: Path,
    database_config_path: Path,
    protocol_name: str,
    pretrained_model: str,
    duration: float,
    max_speakers_per_chunk: int,
    max_speakers_per_frame: int,
    devices: int,
    max_epochs: int,
    experiments_dir: Path,
) -> None:
    """Write the Python entrypoint that mirrors the official pyannote tutorial flow."""

    script = f"""#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.database import FileFinder, registry

DATABASE_CONFIG = Path({json.dumps(str(database_config_path))})
PROTOCOL_NAME = {json.dumps(protocol_name)}
PRETRAINED_MODEL = os.environ.get("PYANNOTE_PRETRAINED_MODEL", {json.dumps(pretrained_model)})
HF_TOKEN = os.environ.get("HF_TOKEN") or True
EXPERIMENTS_DIR = Path(os.environ.get("TRAINING_EXPERIMENT_DIR", {json.dumps(str(experiments_dir))}))

registry.load_database(str(DATABASE_CONFIG))
protocol = registry.get_protocol(PROTOCOL_NAME, preprocessors={{"audio": FileFinder()}})
model = Model.from_pretrained(PRETRAINED_MODEL, token=HF_TOKEN)
model.task = SpeakerDiarization(
    protocol,
    duration={duration},
    max_speakers_per_chunk={max_speakers_per_chunk},
    max_speakers_per_frame={max_speakers_per_frame},
)

accelerator = "gpu" if int({devices}) > 0 else "cpu"
trainer = pl.Trainer(
    devices={devices},
    max_epochs={max_epochs},
    accelerator=accelerator,
    default_root_dir=str(EXPERIMENTS_DIR),
)
trainer.fit(model)
"""
    training_script_path.parent.mkdir(parents=True, exist_ok=True)
    training_script_path.write_text(script, encoding="utf-8")
    training_script_path.chmod(0o755)


def write_launch_script(
    *,
    project_name: str,
    launch_script_path: Path,
    config_name: str,
    speaker_model: str,
    devices: int,
    max_epochs: int,
    project_paths_map: dict[str, Path],
    nemo_root: Path | None,
) -> None:
    """Write a direct launcher for environments where local execution is acceptable."""

    suggested_nemo_root = str(nemo_root.resolve()) if nemo_root else ""
    script = f"""#!/usr/bin/env bash
set -euo pipefail

NEMO_ROOT="${{NEMO_ROOT:-{suggested_nemo_root}}}"
PYTHON_BIN="${{PYTHON_BIN:-python3}}"
NEURAL_DIR="$NEMO_ROOT/examples/speaker_tasks/diarization/neural_diarizer"
TRAIN_MANIFEST="{project_paths_map['manifests_dir'] / 'train_msdd_manifest.jsonl'}"
VAL_MANIFEST="{project_paths_map['manifests_dir'] / 'validation_msdd_manifest.jsonl'}"
TRAIN_EMB_DIR="{project_paths_map['emb_train_dir']}"
VAL_EMB_DIR="{project_paths_map['emb_validation_dir']}"
EXP_DIR="{project_paths_map['experiments_dir']}"
EXP_NAME="{slugify(project_name)}"

EXP_DIR="${{TRAINING_EXPERIMENT_DIR:-$EXP_DIR}}"
EXP_NAME="${{TRAINING_VERSION_SLUG:-$EXP_NAME}}"

mkdir -p "$TRAIN_EMB_DIR" "$VAL_EMB_DIR" "$EXP_DIR"

if [ -z "$NEMO_ROOT" ]; then
  echo "Set NEMO_ROOT to your NeMo checkout before launching fine-tuning." >&2
  exit 1
fi

if [ ! -d "$NEURAL_DIR" ]; then
  echo "NeMo neural diarizer directory not found: $NEURAL_DIR" >&2
  exit 1
fi

cd "$NEURAL_DIR"
exec "$PYTHON_BIN" multiscale_diar_decoder.py \\
  --config-path="../conf/neural_diarizer" \\
  --config-name="{config_name}" \\
  trainer.devices={devices} \\
  trainer.max_epochs={max_epochs} \\
  model.base.diarizer.speaker_embeddings.model_path="{speaker_model}" \\
  model.train_ds.manifest_filepath="$TRAIN_MANIFEST" \\
  model.validation_ds.manifest_filepath="$VAL_MANIFEST" \\
  model.train_ds.emb_dir="$TRAIN_EMB_DIR" \\
  model.validation_ds.emb_dir="$VAL_EMB_DIR" \\
  exp_manager.name="$EXP_NAME" \\
  exp_manager.exp_dir="$EXP_DIR"
"""
    launch_script_path.parent.mkdir(parents=True, exist_ok=True)
    launch_script_path.write_text(script, encoding="utf-8")
    launch_script_path.chmod(0o755)


def write_sbatch_script(
    *,
    project_name: str,
    sbatch_script_path: Path,
    launch_script_path: Path,
    slurm_logs_dir: Path,
    partition: str,
    time_limit: str,
    memory: str,
    cpus_per_task: int,
    gpus: int,
    nemo_root: Path | None,
) -> None:
    """Write a Slurm batch script aligned with the local WAVE usage model."""

    suggested_nemo_root = str(nemo_root.resolve()) if nemo_root else ""
    job_name = f"msdd_ft_{slugify(project_name)}"[:60]
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={slurm_logs_dir}/msdd_finetune_%j.out",
        f"#SBATCH --error={slurm_logs_dir}/msdd_finetune_%j.err",
        f"#SBATCH --partition={partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={time_limit}",
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"mkdir -p {json.dumps(str(slurm_logs_dir))}",
            f'export NEMO_ROOT="${{NEMO_ROOT:-{suggested_nemo_root}}}"',
            'export PYTHON_BIN="${PYTHON_BIN:-python3}"',
            f"bash {json.dumps(str(launch_script_path))}",
            "",
        ]
    )
    sbatch_script_path.parent.mkdir(parents=True, exist_ok=True)
    sbatch_script_path.write_text("\n".join(lines), encoding="utf-8")
    sbatch_script_path.chmod(0o755)


def write_pyannote_launch_script(
    *,
    launch_script_path: Path,
    training_script_path: Path,
) -> None:
    """Write the local launcher for pyannote fine-tuning."""

    script = f"""#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${{PYTHON_BIN:-python3}}"
export HF_TOKEN="${{HF_TOKEN:-${{HUGGINGFACE_TOKEN:-}}}}"

exec "$PYTHON_BIN" {json.dumps(str(training_script_path))}
"""
    launch_script_path.parent.mkdir(parents=True, exist_ok=True)
    launch_script_path.write_text(script, encoding="utf-8")
    launch_script_path.chmod(0o755)


def write_pyannote_sbatch_script(
    *,
    project_name: str,
    sbatch_script_path: Path,
    launch_script_path: Path,
    slurm_logs_dir: Path,
    partition: str,
    time_limit: str,
    memory: str,
    cpus_per_task: int,
    gpus: int,
) -> None:
    """Write the Slurm launcher for pyannote fine-tuning runs."""

    job_name = f"pyannote_ft_{slugify(project_name)}"[:60]
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={slurm_logs_dir}/pyannote_finetune_%j.out",
        f"#SBATCH --error={slurm_logs_dir}/pyannote_finetune_%j.err",
        f"#SBATCH --partition={partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={time_limit}",
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"mkdir -p {json.dumps(str(slurm_logs_dir))}",
            'export PYTHON_BIN="${PYTHON_BIN:-python3}"',
            f"bash {json.dumps(str(launch_script_path))}",
            "",
        ]
    )
    sbatch_script_path.parent.mkdir(parents=True, exist_ok=True)
    sbatch_script_path.write_text("\n".join(lines), encoding="utf-8")
    sbatch_script_path.chmod(0o755)


def prepare_project(
    *,
    project_name: str,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    base_window: float = DEFAULT_BASE_WINDOW,
    base_shift: float = DEFAULT_BASE_SHIFT,
    step_count: int = DEFAULT_STEP_COUNT,
    config_name: str = DEFAULT_CONFIG_NAME,
    speaker_model: str = DEFAULT_SPEAKER_MODEL,
    devices: int = DEFAULT_DEVICES,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    slurm_partition: str = DEFAULT_SLURM_PARTITION,
    slurm_time: str = DEFAULT_SLURM_TIME,
    slurm_memory: str = DEFAULT_SLURM_MEMORY,
    slurm_cpus: int = DEFAULT_SLURM_CPUS,
    slurm_gpus: int = DEFAULT_SLURM_GPUS,
    nemo_root: Path | None = None,
    pyannote_pretrained_model: str = DEFAULT_PYANNOTE_PRETRAINED_MODEL,
    pyannote_duration: float = DEFAULT_PYANNOTE_DURATION,
    pyannote_max_speakers_per_chunk: int = DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK,
    pyannote_max_speakers_per_frame: int = DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME,
    root: Path = PROJECT_ROOT,
) -> FineTuneArtifacts:
    """Generate all files required to fine-tune the selected diarization backend.

    The preparation phase is intentionally deterministic: it rewrites generated
    manifests from scratch so repeated runs do not accumulate stale intermediate
    files from earlier experiments.
    """

    normalized_backend = normalize_backend(backend)
    if base_window <= 0:
        raise ValueError("--base-window must be greater than 0.")
    if base_shift <= 0:
        raise ValueError("--base-shift must be greater than 0.")
    if step_count <= 0:
        raise ValueError("--step-count must be greater than 0.")
    if devices <= 0:
        raise ValueError("--devices must be greater than 0.")
    if max_epochs <= 0:
        raise ValueError("--max-epochs must be greater than 0.")
    if slurm_cpus <= 0:
        raise ValueError("--slurm-cpus must be greater than 0.")
    if slurm_gpus < 0:
        raise ValueError("--slurm-gpus must be 0 or greater.")

    if pyannote_duration <= 0:
        raise ValueError("--pyannote-duration must be greater than 0.")
    if pyannote_max_speakers_per_chunk <= 0:
        raise ValueError("--pyannote-max-speakers-per-chunk must be greater than 0.")
    if pyannote_max_speakers_per_frame <= 0:
        raise ValueError("--pyannote-max-speakers-per-frame must be greater than 0.")

    paths = ensure_project_structure(project_name, backend=normalized_backend, root=root)
    generated_directories = [
        paths["manifests_dir"],
        paths["pairwise_train_dir"],
        paths["pairwise_validation_dir"],
        paths["emb_train_dir"],
        paths["emb_validation_dir"],
        paths["database_config_path"].parent,
        paths["train_list_path"].parent,
        paths["train_rttm_path"].parent,
        paths["train_uem_path"].parent,
    ]
    for generated_dir in generated_directories:
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
        generated_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(project_name, backend=normalized_backend, root=root)
    train_samples, validation_samples = split_samples(samples, train_ratio)

    session_manifest_train = paths["manifests_dir"] / "train_session_manifest.jsonl"
    session_manifest_validation = paths["manifests_dir"] / "validation_session_manifest.jsonl"
    msdd_manifest_train = paths["manifests_dir"] / "train_msdd_manifest.jsonl"
    msdd_manifest_validation = paths["manifests_dir"] / "validation_msdd_manifest.jsonl"
    warnings: list[str] = []
    primary_artifacts: list[Path] = []

    if normalized_backend == "nemo":
        write_jsonl(session_manifest_train, (session_manifest_row(sample) for sample in train_samples))
        write_jsonl(
            session_manifest_validation,
            (session_manifest_row(sample) for sample in validation_samples),
        )
        write_path_list(
            paths["manifests_dir"] / "train_audio_paths.txt",
            (sample.audio_path for sample in train_samples),
        )
        write_path_list(
            paths["manifests_dir"] / "validation_audio_paths.txt",
            (sample.audio_path for sample in validation_samples),
        )
        write_path_list(
            paths["manifests_dir"] / "train_rttm_paths.txt",
            (sample.rttm_path for sample in train_samples),
        )
        write_path_list(
            paths["manifests_dir"] / "validation_rttm_paths.txt",
            (sample.rttm_path for sample in validation_samples),
        )

        train_rows: list[dict[str, object]] = []
        validation_rows: list[dict[str, object]] = []
        for sample in train_samples:
            rows, sample_warnings = build_msdd_rows_for_sample(
                sample=sample,
                pairwise_dir=paths["pairwise_train_dir"],
                base_window=base_window,
                base_shift=base_shift,
                step_count=step_count,
            )
            train_rows.extend(rows)
            warnings.extend(sample_warnings)
        for sample in validation_samples:
            rows, sample_warnings = build_msdd_rows_for_sample(
                sample=sample,
                pairwise_dir=paths["pairwise_validation_dir"],
                base_window=base_window,
                base_shift=base_shift,
                step_count=step_count,
            )
            validation_rows.extend(rows)
            warnings.extend(sample_warnings)

        if not train_rows:
            raise ValueError("No MSDD training rows were generated. Check your RTTM labels.")
        if not validation_rows:
            raise ValueError("No MSDD validation rows were generated. Check your RTTM labels.")

        write_jsonl(msdd_manifest_train, train_rows)
        write_jsonl(msdd_manifest_validation, validation_rows)
        write_launch_script(
            project_name=project_name,
            launch_script_path=paths["launch_script_path"],
            config_name=config_name,
            speaker_model=speaker_model,
            devices=devices,
            max_epochs=max_epochs,
            project_paths_map=paths,
            nemo_root=nemo_root,
        )
        write_sbatch_script(
            project_name=project_name,
            sbatch_script_path=paths["sbatch_script_path"],
            launch_script_path=paths["launch_script_path"],
            slurm_logs_dir=paths["slurm_logs_dir"],
            partition=slurm_partition,
            time_limit=slurm_time,
            memory=slurm_memory,
            cpus_per_task=slurm_cpus,
            gpus=slurm_gpus,
            nemo_root=nemo_root,
        )
        primary_artifacts.extend(
            [
                session_manifest_train,
                session_manifest_validation,
                msdd_manifest_train,
                msdd_manifest_validation,
            ]
        )
    else:
        write_uri_list(paths["train_list_path"], train_samples)
        write_uri_list(paths["development_list_path"], validation_samples)
        write_uri_list(paths["test_list_path"], validation_samples)
        write_pyannote_subset_rttm(paths["train_rttm_path"], train_samples)
        write_pyannote_subset_rttm(paths["development_rttm_path"], validation_samples)
        write_pyannote_subset_rttm(paths["test_rttm_path"], validation_samples)
        write_pyannote_subset_uem(paths["train_uem_path"], train_samples)
        write_pyannote_subset_uem(paths["development_uem_path"], validation_samples)
        write_pyannote_subset_uem(paths["test_uem_path"], validation_samples)
        protocol_name = write_pyannote_database_config(
            database_config_path=paths["database_config_path"],
            project_name=project_name,
            project_paths_map=paths,
            root=root,
        )
        write_pyannote_training_script(
            training_script_path=paths["training_script_path"],
            database_config_path=paths["database_config_path"],
            protocol_name=protocol_name,
            pretrained_model=pyannote_pretrained_model,
            duration=pyannote_duration,
            max_speakers_per_chunk=pyannote_max_speakers_per_chunk,
            max_speakers_per_frame=pyannote_max_speakers_per_frame,
            devices=devices,
            max_epochs=max_epochs,
            experiments_dir=paths["experiments_dir"],
        )
        write_pyannote_launch_script(
            launch_script_path=paths["launch_script_path"],
            training_script_path=paths["training_script_path"],
        )
        write_pyannote_sbatch_script(
            project_name=project_name,
            sbatch_script_path=paths["sbatch_script_path"],
            launch_script_path=paths["launch_script_path"],
            slurm_logs_dir=paths["slurm_logs_dir"],
            partition=slurm_partition,
            time_limit=slurm_time,
            memory=slurm_memory,
            cpus_per_task=slurm_cpus,
            gpus=slurm_gpus,
        )
        primary_artifacts.extend(
            [
                paths["database_config_path"],
                paths["train_list_path"],
                paths["development_list_path"],
                paths["training_script_path"],
            ]
        )
        session_manifest_train = None
        session_manifest_validation = None
        msdd_manifest_train = None
        msdd_manifest_validation = None

    metadata = {
        "backend": normalized_backend,
        "project_name": project_name,
        "project_slug": slugify(project_name),
        "generated_at_utc": utc_now_iso(),
        "sample_count": len(samples),
        "train_count": len(train_samples),
        "validation_count": len(validation_samples),
        "train_ratio": train_ratio,
        "devices": devices,
        "max_epochs": max_epochs,
        "slurm_partition": slurm_partition,
        "slurm_time": slurm_time,
        "slurm_memory": slurm_memory,
        "slurm_cpus": slurm_cpus,
        "slurm_gpus": slurm_gpus,
        "nemo_root": str(nemo_root.resolve()) if nemo_root else "",
        "warnings": warnings,
        "samples": [
            {
                "stem": sample.stem,
                "audio_path": str(sample.audio_path),
                "rttm_path": str(sample.rttm_path),
                "transcript_path": str(sample.transcript_path) if sample.transcript_path else "",
                "duration_seconds": sample.duration_seconds,
                "speakers": list(sample.speakers),
            }
            for sample in sorted(samples, key=lambda item: item.stem.lower())
        ],
    }
    if normalized_backend == "nemo":
        metadata.update(
            {
                "base_window": base_window,
                "base_shift": base_shift,
                "step_count": step_count,
                "config_name": config_name,
                "speaker_model": speaker_model,
            }
        )
    else:
        metadata.update(
            {
                "pyannote_pretrained_model": pyannote_pretrained_model,
                "pyannote_duration": pyannote_duration,
                "pyannote_max_speakers_per_chunk": pyannote_max_speakers_per_chunk,
                "pyannote_max_speakers_per_frame": pyannote_max_speakers_per_frame,
                "database_config_path": str(paths["database_config_path"]),
                "training_script_path": str(paths["training_script_path"]),
            }
        )
    paths["metadata_path"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return FineTuneArtifacts(
        backend=normalized_backend,
        project_slug=slugify(project_name),
        project_dir=paths["project_dir"],
        sample_count=len(samples),
        train_count=len(train_samples),
        validation_count=len(validation_samples),
        session_manifest_train=session_manifest_train,
        session_manifest_validation=session_manifest_validation,
        msdd_manifest_train=msdd_manifest_train,
        msdd_manifest_validation=msdd_manifest_validation,
        metadata_path=paths["metadata_path"],
        launch_script_path=paths["launch_script_path"],
        sbatch_script_path=paths["sbatch_script_path"],
        warnings=tuple(warnings),
        primary_artifacts=tuple(primary_artifacts),
    )


def process_is_running(pid: int) -> bool:
    """Check whether a locally launched fine-tuning process is still alive."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_status(run_dir: Path) -> str:
    """Resolve a human-readable run state from local logs or Slurm state."""

    exit_code_path = run_dir / "exit_code.txt"
    metadata_path = run_dir / "metadata.json"
    if exit_code_path.is_file():
        try:
            exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return "unknown"
        return "succeeded" if exit_code == 0 else "failed"

    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        job_id = str(metadata.get("job_id", "")).strip()
        if job_id:
            squeue_bin = shutil.which("squeue")
            if squeue_bin:
                completed = subprocess.run(
                    [squeue_bin, "-h", "-j", job_id, "-o", "%T"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                queue_state = completed.stdout.strip().lower()
                if queue_state:
                    return queue_state
            return "submitted"
        if metadata.get("submission_mode") == "sbatch":
            return "submitted"
        if process_is_running(int(metadata.get("pid", 0))):
            return "running"
    return "stopped"


def list_runs(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """List historical runs for one project, newest first.

    The optional limit keeps the dashboard path lightweight when it only needs
    the most recent run rather than the full experiment history.
    """

    paths = project_paths(project_name, backend=backend, root=root)
    runs_dir = paths["runs_dir"]
    if not runs_dir.is_dir():
        return []
    runs: list[dict[str, object]] = []
    for run_dir in sorted(runs_dir.glob("*"), reverse=True):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = run_status(run_dir)
        metadata["run_dir"] = str(run_dir)
        # Surface the user-chosen rename (if any). Fall back to version_name so
        # the dashboard always has SOMETHING to display for legacy runs.
        display = read_run_display(run_dir)
        metadata["display_name"] = (
            str(display.get("display_name") or "").strip()
            or str(metadata.get("version_name") or "").strip()
        )
        runs.append(metadata)
        if limit is not None and len(runs) >= limit:
            break
    return runs


def next_training_version_number(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> int:
    """Return the next user-facing training version number for a project/backend."""

    paths = project_paths(project_name, backend=backend, root=root)
    runs_dir = paths["runs_dir"]
    if not runs_dir.is_dir():
        return 1
    run_count = sum(1 for candidate in runs_dir.iterdir() if candidate.is_dir())
    return run_count + 1


def resolve_training_version(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    version_name: str | None = None,
    root: Path = PROJECT_ROOT,
) -> tuple[str, str, int, Path]:
    """Resolve display name, unique slug, number, and output directory for a run."""

    normalized_backend = normalize_backend(backend)
    paths = project_paths(project_name, backend=normalized_backend, root=root)
    version_number = next_training_version_number(
        project_name,
        backend=normalized_backend,
        root=root,
    )
    cleaned_name = (version_name or "").strip()
    if not cleaned_name:
        project_label = project_name.strip() or slugify(project_name)
        cleaned_name = f"{project_label} trained version {version_number}"
    base_slug = slugify(cleaned_name)
    experiment_dir = unique_child_path(paths["experiments_dir"], base_slug)
    return cleaned_name, experiment_dir.name, version_number, experiment_dir


def latest_run_summary(project_dir: Path) -> dict[str, object] | None:
    """Return the newest run summary for one project without loading full history."""

    runs_dir = project_dir / "runs"
    if not runs_dir.is_dir():
        return None

    for run_dir in sorted(runs_dir.glob("*"), reverse=True):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = run_status(run_dir)
        metadata["run_dir"] = str(run_dir)
        display = read_run_display(run_dir)
        metadata["display_name"] = (
            str(display.get("display_name") or "").strip()
            or str(metadata.get("version_name") or "").strip()
        )
        return metadata
    return None


def list_projects(*, root: Path = PROJECT_ROOT) -> list[dict[str, object]]:
    """Summarize every fine-tuning project present under the workspace root."""

    projects_root = root / "fine_tuning" / "projects"
    if not projects_root.is_dir():
        return []

    summaries: list[dict[str, object]] = []
    candidate_projects: list[tuple[str, Path]] = []
    for candidate in sorted(projects_root.iterdir(), key=lambda item: item.name.lower()):
        if not candidate.is_dir():
            continue
        if (candidate / "audio").is_dir():
            candidate_projects.append((DEFAULT_FINE_TUNING_BACKEND, candidate))
            continue
        for nested in sorted(candidate.iterdir(), key=lambda item: item.name.lower()):
            if nested.is_dir() and (nested / "audio").is_dir():
                candidate_projects.append((candidate.name, nested))

    for backend, candidate in candidate_projects:
        audio_dir = candidate / "audio"
        metadata_path = candidate / "artifacts" / "metadata.json"
        sample_count = (
            sum(
                1
                for path in audio_dir.iterdir()
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
            )
            if audio_dir.is_dir()
            else 0
        )
        # The slug is the persistent identifier; display_name is a free-form
        # rename the user can set from the dashboard. Fall back to the slug so
        # we always have a label even before anyone customizes it.
        display_payload = _read_display_sidecar(candidate / "display.json")
        display_name = str(display_payload.get("display_name") or "").strip() or candidate.name
        summary: dict[str, object] = {
            "backend": backend,
            "slug": candidate.name,
            "display_name": display_name,
            "path": str(candidate),
            "sample_count": sample_count,
            "prepared": metadata_path.is_file(),
            "latest_run": None,
        }
        if metadata_path.is_file():
            summary["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        recent_runs = list_runs(candidate.name, backend=backend, root=root, limit=5)
        if recent_runs:
            summary["recent_runs"] = recent_runs
        latest_run = latest_run_summary(candidate)
        if latest_run is not None:
            summary["latest_run"] = latest_run
        if sample_count == 0 and not summary.get("prepared") and latest_run is None:
            continue
        summaries.append(summary)
    return summaries


def launch_training(
    *,
    project_name: str,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    nemo_root: Path | None = None,
    python_bin: str = sys.executable,
    prefer_sbatch: bool = True,
    version_name: str | None = None,
    extra_env: dict[str, str] | None = None,
    root: Path = PROJECT_ROOT,
) -> FineTuneRun:
    """Launch a prepared project locally or submit it to Slurm when available."""

    normalized_backend = normalize_backend(backend)
    paths = ensure_project_structure(project_name, backend=normalized_backend, root=root)
    launch_script_path = paths["launch_script_path"]
    sbatch_script_path = paths["sbatch_script_path"]
    metadata_path = paths["metadata_path"]
    if not launch_script_path.is_file():
        raise FileNotFoundError(
            f"Training launcher not found for project '{project_name}'. Run prepare first."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Project metadata not found for project '{project_name}'. Run prepare first."
        )

    resolved_version_name, version_slug, version_number, experiment_dir = resolve_training_version(
        project_name,
        backend=normalized_backend,
        version_name=version_name,
        root=root,
    )
    run_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{version_slug}"
    run_dir = paths["runs_dir"] / run_name
    if run_dir.exists():
        run_dir = unique_child_path(paths["runs_dir"], run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    exit_code_path = run_dir / "exit_code.txt"
    wrapper_path = run_dir / "launch.sh"
    env_path = run_dir / "env.json"

    env_data = {
        "backend": normalized_backend,
        "NEMO_ROOT": str(nemo_root.resolve()) if nemo_root else "",
        "PYTHON_BIN": python_bin,
        "TRAINING_VERSION_NAME": resolved_version_name,
        "TRAINING_VERSION_SLUG": version_slug,
        "TRAINING_VERSION_NUMBER": version_number,
        "TRAINING_EXPERIMENT_DIR": str(experiment_dir),
        "launched_at_utc": utc_now_iso(),
        "prefer_sbatch": prefer_sbatch,
        "extra_env_keys": sorted((extra_env or {}).keys()),
    }
    env_path.write_text(json.dumps(env_data, indent=2, sort_keys=True), encoding="utf-8")
    inherited_env = {**os.environ, **(extra_env or {})}

    sbatch_bin = shutil.which("sbatch")
    # The WAVE documentation emphasizes scheduler-backed execution for shared GPU
    # workloads, so Slurm submission is preferred whenever the command is available.
    if prefer_sbatch and sbatch_bin and sbatch_script_path.is_file():
        completed = subprocess.run(
            [sbatch_bin, str(sbatch_script_path)],
            cwd=str(paths["project_dir"]),
            check=False,
            capture_output=True,
            text=True,
            env={
                **inherited_env,
                "PYTHON_BIN": python_bin,
                "TRAINING_VERSION_NAME": resolved_version_name,
                "TRAINING_VERSION_SLUG": version_slug,
                "TRAINING_VERSION_NUMBER": str(version_number),
                "TRAINING_EXPERIMENT_DIR": str(experiment_dir),
                **(
                    {"NEMO_ROOT": str(nemo_root.resolve())}
                    if nemo_root is not None
                    else {}
                ),
            },
        )
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        job_id = ""
        for token in (completed.stdout or "").split():
            if token.isdigit():
                job_id = token
        run_metadata = {
            "backend": normalized_backend,
            "project_name": project_name,
            "project_slug": slugify(project_name),
            "version_name": resolved_version_name,
            "version_slug": version_slug,
            "version_number": version_number,
            "experiment_dir": str(experiment_dir),
            "pid": 0,
            "job_id": job_id,
            "submission_mode": "sbatch",
            "started_at_utc": utc_now_iso(),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "exit_code_path": str(exit_code_path),
            "sbatch_script_path": str(sbatch_script_path),
            "nemo_root": str(nemo_root.resolve()) if nemo_root else "",
            "python_bin": python_bin,
            "submission_returncode": completed.returncode,
        }
        run_metadata_path = run_dir / "metadata.json"
        run_metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
        if completed.returncode != 0:
            exit_code_path.write_text(str(completed.returncode), encoding="utf-8")
        return FineTuneRun(
            run_dir=run_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata_path=run_metadata_path,
            exit_code_path=exit_code_path,
            pid=0,
            job_id=job_id,
            version_name=resolved_version_name,
            version_slug=version_slug,
            version_number=version_number,
            experiment_dir=experiment_dir,
        )

    wrapper_lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"export PYTHON_BIN={json.dumps(python_bin)}",
        f"export TRAINING_VERSION_NAME={json.dumps(resolved_version_name)}",
        f"export TRAINING_VERSION_SLUG={json.dumps(version_slug)}",
        f"export TRAINING_VERSION_NUMBER={json.dumps(str(version_number))}",
        f"export TRAINING_EXPERIMENT_DIR={json.dumps(str(experiment_dir))}",
    ]
    if nemo_root:
        wrapper_lines.append(f"export NEMO_ROOT={json.dumps(str(nemo_root.resolve()))}")
    wrapper_lines.extend(
        [
            "rc=0",
            f"if ! bash {json.dumps(str(launch_script_path))}; then",
            "  rc=$?",
            "fi",
            f"printf '%s\\n' \"$rc\" > {json.dumps(str(exit_code_path))}",
            "exit \"$rc\"",
        ]
    )
    wrapper_path.write_text("\n".join(wrapper_lines) + "\n", encoding="utf-8")
    wrapper_path.chmod(0o755)

    # The local fallback remains available for development and tests, but it runs
    # in a detached shell so the web or CLI caller does not block on training.
    launch_command = (
        f"nohup bash {shlex.quote(str(wrapper_path))} "
        f"> {shlex.quote(str(stdout_path))} "
        f"2> {shlex.quote(str(stderr_path))} "
        "< /dev/null & echo $!"
    )
    completed = subprocess.run(
        ["bash", "-lc", launch_command],
        cwd=str(paths["project_dir"]),
        check=False,
        capture_output=True,
        text=True,
        env=inherited_env,
    )
    if completed.returncode != 0:
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        exit_code_path.write_text(str(completed.returncode), encoding="utf-8")
        raise RuntimeError(
            f"Failed to launch training wrapper for project '{project_name}'."
        )
    pid_text = (completed.stdout or "").strip().splitlines()
    pid = int(pid_text[-1]) if pid_text and pid_text[-1].isdigit() else 0

    run_metadata = {
        "backend": normalized_backend,
        "project_name": project_name,
        "project_slug": slugify(project_name),
        "version_name": resolved_version_name,
        "version_slug": version_slug,
        "version_number": version_number,
        "experiment_dir": str(experiment_dir),
        "pid": pid,
        "started_at_utc": utc_now_iso(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "exit_code_path": str(exit_code_path),
        "wrapper_path": str(wrapper_path),
        "nemo_root": str(nemo_root.resolve()) if nemo_root else "",
        "python_bin": python_bin,
    }
    run_metadata_path = run_dir / "metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")

    return FineTuneRun(
        run_dir=run_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        metadata_path=run_metadata_path,
        exit_code_path=exit_code_path,
        pid=pid,
        job_id="",
        version_name=resolved_version_name,
        version_slug=version_slug,
        version_number=version_number,
        experiment_dir=experiment_dir,
    )


def print_project_status(*, root: Path = PROJECT_ROOT) -> int:
    """Print a concise project summary for command-line inspection."""

    projects = list_projects(root=root)
    if not projects:
        print(f"No fine-tuning projects found in {root / 'fine_tuning' / 'projects'}")
        return 0

    for project in projects:
        print(f"Project: {project['backend']}/{project['slug']}")
        print(f"  Path: {project['path']}")
        print(f"  Samples: {project['sample_count']}")
        print(f"  Prepared: {'yes' if project['prepared'] else 'no'}")
        latest_run = project.get("latest_run")
        if latest_run:
            print(f"  Latest run: {latest_run['status']} ({latest_run['run_dir']})")
        else:
            print("  Latest run: none")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the small CLI front door for project preparation and launch."""

    parser = argparse.ArgumentParser(
        description="Prepare and launch NeMo or pyannote fine-tuning assets for this project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="List fine-tuning projects and runs.")
    status_parser.set_defaults(func=cmd_status)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Generate backend-specific fine-tuning artifacts and launch scripts.",
    )
    prepare_parser.add_argument("--project", required=True, help="Project name or slug.")
    prepare_parser.add_argument(
        "--backend",
        choices=sorted(SUPPORTED_FINE_TUNING_BACKENDS),
        default=DEFAULT_FINE_TUNING_BACKEND,
    )
    prepare_parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    prepare_parser.add_argument("--base-window", type=float, default=DEFAULT_BASE_WINDOW)
    prepare_parser.add_argument("--base-shift", type=float, default=DEFAULT_BASE_SHIFT)
    prepare_parser.add_argument("--step-count", type=int, default=DEFAULT_STEP_COUNT)
    prepare_parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    prepare_parser.add_argument("--speaker-model", default=DEFAULT_SPEAKER_MODEL)
    prepare_parser.add_argument("--devices", type=int, default=DEFAULT_DEVICES)
    prepare_parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    prepare_parser.add_argument("--slurm-partition", default=DEFAULT_SLURM_PARTITION)
    prepare_parser.add_argument("--slurm-time", default=DEFAULT_SLURM_TIME)
    prepare_parser.add_argument("--slurm-memory", default=DEFAULT_SLURM_MEMORY)
    prepare_parser.add_argument("--slurm-cpus", type=int, default=DEFAULT_SLURM_CPUS)
    prepare_parser.add_argument("--slurm-gpus", type=int, default=DEFAULT_SLURM_GPUS)
    prepare_parser.add_argument("--nemo-root", default=None)
    prepare_parser.add_argument("--pyannote-pretrained-model", default=DEFAULT_PYANNOTE_PRETRAINED_MODEL)
    prepare_parser.add_argument("--pyannote-duration", type=float, default=DEFAULT_PYANNOTE_DURATION)
    prepare_parser.add_argument(
        "--pyannote-max-speakers-per-chunk",
        type=int,
        default=DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK,
    )
    prepare_parser.add_argument(
        "--pyannote-max-speakers-per-frame",
        type=int,
        default=DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME,
    )
    prepare_parser.set_defaults(func=cmd_prepare)

    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch fine-tuning in the background using the generated launcher script.",
    )
    launch_parser.add_argument("--project", required=True, help="Project name or slug.")
    launch_parser.add_argument(
        "--backend",
        choices=sorted(SUPPORTED_FINE_TUNING_BACKENDS),
        default=DEFAULT_FINE_TUNING_BACKEND,
    )
    launch_parser.add_argument("--nemo-root", default=None, help="Path to a NeMo checkout.")
    launch_parser.add_argument("--python-bin", default=sys.executable)
    launch_parser.add_argument(
        "--version-name",
        default="",
        help="Optional display name for this trained version.",
    )
    launch_parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run locally even when sbatch is available.",
    )
    launch_parser.set_defaults(func=cmd_launch)

    return parser


def cmd_status(args: argparse.Namespace) -> int:
    return print_project_status()


def cmd_prepare(args: argparse.Namespace) -> int:
    artifacts = prepare_project(
        project_name=args.project,
        backend=args.backend,
        train_ratio=args.train_ratio,
        base_window=args.base_window,
        base_shift=args.base_shift,
        step_count=args.step_count,
        config_name=args.config_name,
        speaker_model=args.speaker_model,
        devices=args.devices,
        max_epochs=args.max_epochs,
        slurm_partition=args.slurm_partition,
        slurm_time=args.slurm_time,
        slurm_memory=args.slurm_memory,
        slurm_cpus=args.slurm_cpus,
        slurm_gpus=args.slurm_gpus,
        nemo_root=Path(args.nemo_root).expanduser().resolve() if args.nemo_root else None,
        pyannote_pretrained_model=args.pyannote_pretrained_model,
        pyannote_duration=args.pyannote_duration,
        pyannote_max_speakers_per_chunk=args.pyannote_max_speakers_per_chunk,
        pyannote_max_speakers_per_frame=args.pyannote_max_speakers_per_frame,
    )
    print(f"Prepared project: {artifacts.backend}/{artifacts.project_slug}")
    print(f"Launch script: {artifacts.launch_script_path}")
    print(f"Slurm script: {artifacts.sbatch_script_path}")
    for artifact_path in artifacts.primary_artifacts:
        print(f"Artifact: {artifact_path}")
    if artifacts.warnings:
        print("Warnings:")
        for warning in artifacts.warnings:
            print(f"  - {warning}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    run = launch_training(
        project_name=args.project,
        backend=args.backend,
        nemo_root=Path(args.nemo_root).expanduser().resolve() if args.nemo_root else None,
        python_bin=args.python_bin,
        prefer_sbatch=not args.local,
        version_name=args.version_name,
    )
    if run.job_id:
        print(f"Submitted training version {run.version_name} as Slurm job {run.job_id}")
    else:
        print(f"Started training version {run.version_name} (pid {run.pid})")
    print(f"stdout: {run.stdout_path}")
    print(f"stderr: {run.stderr_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
