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
import re
import shutil
import shlex
import struct
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from audio_numbering import AUDIO_EXTENSIONS
from workflow_background import process_is_running, slurm_job_state, utc_now_iso

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAME = "msdd_5scl_15_05_50Povl_256x3x32x2.yaml"
DEFAULT_SPEAKER_MODEL = "titanet_large"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_BASE_WINDOW = 0.5
DEFAULT_BASE_SHIFT = 0.25
DEFAULT_STEP_COUNT = 50
DEFAULT_MAX_EPOCHS = 20
DEFAULT_DEVICES = 1
# SLURM defaults sized for the WAVE `gpu` partition (per node: 2 Volta GPUs,
# 80 CPUs, ~376 GB RAM, 2-day max). NeMo MSDD fine-tuning kept hitting the
# old 4-8 h limit before it could export the final .nemo, so the time
# budget is generous and the CPU/RAM headroom lets the dataloader keep the
# single GPU fed. One GPU on purpose: the fine-tune datasets are small and
# multi-GPU DDP sync overhead would outweigh the gain.
DEFAULT_SLURM_PARTITION = "gpu"
DEFAULT_SLURM_TIME = "12:00:00"
DEFAULT_SLURM_MEMORY = "96G"
DEFAULT_SLURM_CPUS = 16
DEFAULT_SLURM_GPUS = 1
DEFAULT_FINE_TUNING_BACKEND = "nemo"
SUPPORTED_FINE_TUNING_BACKENDS = {"nemo", "pyannote"}
DEFAULT_PYANNOTE_PRETRAINED_MODEL = "pyannote/segmentation-3.0"
DEFAULT_PYANNOTE_DURATION = 10.0
DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_CHUNK = 3
DEFAULT_PYANNOTE_MAX_SPEAKERS_PER_FRAME = 2


@dataclass(frozen=True)
class RttmSegment:
    """One validated RTTM speaker segment, stored immutably so we can't accidentally mutate training data."""

    session_id: str
    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class TrainingSample:
    """Bundle all per-sample metadata so manifest generation doesn't re-read the same files multiple times."""

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
    """Capture the paths of every artifact that prepare_project wrote, so callers don't have to guess them."""

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
    """Holds one launch attempt: local PID and SLURM job ID so we can query status from either."""

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
    """Convert a user-typed project name into a filesystem-safe slug for clean directory names."""

    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    return cleaned or "project"


def unique_child_path(parent: Path, slug: str) -> Path:
    """Find a free path under parent so we never silently overwrite a prior project with the same name."""

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
    """Strip directory components and replace unsafe characters so user-supplied filenames can't escape the project dir."""

    name = Path(value or "").name
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    safe = safe.strip("._")
    return safe or "upload.bin"


def normalize_backend(backend: str | None) -> str:
    """Validate and lowercase the backend string so every caller gets a consistent canonical value."""

    normalized = (backend or DEFAULT_FINE_TUNING_BACKEND).strip().lower()
    if normalized not in SUPPORTED_FINE_TUNING_BACKENDS:
        raise ValueError(f"Unsupported fine-tuning backend: {backend}")
    return normalized


def format_rttm_line(
    *,
    session_id: str,
    start: float | str,
    duration: float | str,
    speaker: str,
) -> str:
    """Return one canonical 10-column RTTM SPEAKER row for training."""

    return " ".join(
        [
            "SPEAKER",
            str(session_id).strip() or "sample",
            "1",
            f"{float(start):.3f}",
            f"{float(duration):.3f}",
            "<NA>",
            "<NA>",
            str(speaker).strip() or "speaker",
            "<NA>",
            "<NA>",
        ]
    )


def canonicalize_rttm_text(raw_text: str, *, session_id: str) -> str:
    """Normalize valid RTTM speaker rows without changing segment timing or speaker assignment."""

    lines: list[str] = []
    for raw_line in raw_text.splitlines():
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
        lines.append(
            format_rttm_line(
                session_id=session_id,
                start=start,
                duration=duration,
                speaker=parts[7],
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def write_canonical_rttm_text(path: Path, raw_text: str, *, session_id: str) -> None:
    """Write RTTM content in the canonical format consumed by the training jobs."""

    write_text_replacing_existing(
        path,
        canonicalize_rttm_text(raw_text, session_id=session_id),
        encoding="utf-8",
    )


def sibling_temp_path(path: Path) -> Path:
    """Return an unused temp path next to the final target for atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    for index in range(1000):
        candidate = path.with_name(f".{path.name}.tmp-{os.getpid()}-{index}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Unable to allocate temporary path for {path}")


def write_bytes_replacing_existing(path: Path, payload: bytes) -> None:
    """Write bytes without truncating an existing hardlinked target in place."""

    tmp_path = sibling_temp_path(path)
    try:
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_text_replacing_existing(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
    """Write text without mutating other hardlinks to the previous file."""

    tmp_path = sibling_temp_path(path)
    try:
        tmp_path.write_text(payload, encoding=encoding)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def copy_stream_replacing_existing(path: Path, source_stream) -> None:
    """Stream an upload to disk without mutating an existing hardlinked target."""

    tmp_path = sibling_temp_path(path)
    try:
        with tmp_path.open("wb") as handle:
            shutil.copyfileobj(source_stream, handle)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _has_audio_samples(audio_dir: Path) -> bool:
    """Return whether an audio directory contains at least one supported media file."""

    if not audio_dir.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        for path in audio_dir.iterdir()
    )


# Display names live in display.json next to metadata.json. prepare_project
# regenerates metadata.json every run, so any name stored there gets wiped.
# Keeping it in a separate sidecar means renames survive re-prepares.
def _read_display_sidecar(path: Path) -> dict[str, object]:
    """Load display.json from disk, returning {} if the file doesn't exist or is corrupted."""

    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_display_sidecar(path: Path, *, updates: dict[str, object]) -> dict[str, object]:
    """Merge new display fields into the existing display.json so we don't clobber unrelated display keys."""

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
    """Return the path to a project's display.json so nothing else hard-codes it."""

    return project_dir(project_name, backend=backend, root=root) / "display.json"


def read_project_display(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Load a project's display metadata (e.g. the user-chosen display name) from its sidecar."""

    return _read_display_sidecar(project_display_path(project_name, backend=backend, root=root))


def set_project_display_name(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    display_name: str,
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Rename a project's display label without touching the on-disk slug or any training artifacts."""

    cleaned = (display_name or "").strip()
    if not cleaned:
        raise ValueError("display_name must be a non-empty string.")
    target = project_dir(project_name, backend=backend, root=root)
    if not target.is_dir():
        raise FileNotFoundError(f"Unknown fine-tuning project: {project_name}")
    return _write_display_sidecar(target / "display.json", updates={"display_name": cleaned})


def set_project_auto_train(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    enabled: bool,
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Toggle the per-project "auto-train when labels complete" flag in display.json.

    Stored in the same sidecar as display_name so it survives prepare_project
    regeneration. Reading and writing is intentionally cheap so the label-save
    path can poll it on every completion without measurable overhead.
    """

    target = project_dir(project_name, backend=backend, root=root)
    if not target.is_dir():
        raise FileNotFoundError(f"Unknown fine-tuning project: {project_name}")
    return _write_display_sidecar(
        target / "display.json",
        updates={"auto_train": bool(enabled)},
    )


def read_run_display(run_dir: Path) -> dict[str, object]:
    """Load a run's display metadata (e.g. its friendly name) from its per-run display.json sidecar."""

    return _read_display_sidecar(run_dir / "display.json")


def set_run_display_name(run_dir: Path, *, display_name: str) -> dict[str, object]:
    """Set a friendly label on a single training run without moving or renaming the run directory."""

    cleaned = (display_name or "").strip()
    if not cleaned:
        raise ValueError("display_name must be a non-empty string.")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Unknown fine-tuning run: {run_dir}")
    return _write_display_sidecar(run_dir / "display.json", updates={"display_name": cleaned})


def delete_run_model_artifacts(run_dir: Path) -> dict[str, object]:
    """Delete one trained model's experiment artifacts while keeping run logs/history."""

    resolved_run_dir = run_dir.resolve()
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(f"Unknown fine-tuning run: {run_dir}")
    status = run_status(resolved_run_dir)
    if status in {"running", "submitted"}:
        raise ValueError(f"Cannot delete model artifacts while the run is {status}.")

    metadata_path = resolved_run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run metadata is not valid JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Run metadata must be a JSON object: {metadata_path}")

    experiment_dir_value = str(metadata.get("experiment_dir") or "").strip()
    if not experiment_dir_value:
        raise ValueError("This run does not record a model artifact directory.")
    experiment_dir = Path(experiment_dir_value).expanduser()
    if not experiment_dir.is_absolute():
        experiment_dir = (resolved_run_dir / experiment_dir).resolve()
    resolved_experiment_dir = experiment_dir.resolve()

    project_root = resolved_run_dir.parent.parent
    experiments_root = (project_root / "artifacts" / "experiments").resolve()
    try:
        resolved_experiment_dir.relative_to(experiments_root)
    except ValueError as exc:
        raise ValueError("Refusing to delete a model path outside this fine-tuning project's experiments directory.") from exc

    removed = False
    if resolved_experiment_dir.is_dir():
        shutil.rmtree(resolved_experiment_dir)
        removed = True
    elif resolved_experiment_dir.exists():
        resolved_experiment_dir.unlink()
        removed = True

    metadata.update(
        {
            "model_deleted": True,
            "model_deleted_at_utc": utc_now_iso(),
            "deleted_experiment_dir": str(resolved_experiment_dir),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "run_dir": str(resolved_run_dir),
        "experiment_dir": str(resolved_experiment_dir),
        "removed": removed,
        "status": status,
    }


def project_dir(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the canonical on-disk path for a project directory."""

    normalized_backend = normalize_backend(backend)
    slug = slugify(project_name)
    canonical = root / "fine_tuning" / "projects" / normalized_backend / slug
    legacy = root / "fine_tuning" / "projects" / slug
    if (
        normalized_backend == DEFAULT_FINE_TUNING_BACKEND
        and not canonical.exists()
        and _has_audio_samples(legacy / "audio")
    ):
        return legacy
    return canonical


def project_paths(
    project_name: str,
    *,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Return every well-known path for a project in one dict so no caller hard-codes the layout."""

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
    write_bytes_replacing_existing(audio_path, audio_bytes)
    try:
        rttm_text = rttm_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("RTTM upload must be UTF-8 text.") from exc
    write_canonical_rttm_text(rttm_path, rttm_text, session_id=stem)

    transcript_path: Path | None = None
    if transcript_bytes:
        transcript_filename = sanitize_filename(transcript_name or f"{stem}.txt")
        transcript_suffix = Path(transcript_filename).suffix or ".txt"
        transcript_path = paths["text_dir"] / f"{stem}{transcript_suffix}"
        write_bytes_replacing_existing(transcript_path, transcript_bytes)
    elif transcript_text and transcript_text.strip():
        transcript_path = paths["text_dir"] / f"{stem}.txt"
        write_text_replacing_existing(transcript_path, transcript_text.strip() + "\n", encoding="utf-8")

    return build_sample(audio_path, rttm_path, transcript_path)


def replace_path_with_symlink(target_path: Path, source_path: Path) -> None:
    """Atomically place a symlink at target_path pointing to source_path.

    Used by the dashboard's "link existing audio" upload path so a fine-tune
    project can reuse media that already lives in audio_in/ or a stitched
    folder without doubling its disk usage. The target's parent is created
    if needed; any pre-existing file or link at the target is replaced.
    """

    target_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_source = source_path.resolve()
    tmp_path = sibling_temp_path(target_path)
    try:
        tmp_path.symlink_to(resolved_source)
        os.replace(tmp_path, target_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_project_sample_links(
    *,
    project_name: str,
    backend: str = DEFAULT_FINE_TUNING_BACKEND,
    audio_path: Path,
    rttm_path: Path,
    audio_name: str | None = None,
    transcript_path: Path | None = None,
    transcript_text: str | None = None,
    root: Path = PROJECT_ROOT,
) -> TrainingSample:
    """Register a labeled sample by symlinking audio/transcript instead of copying.

    RTTM still gets rewritten in canonical form because the trainers expect a
    normalized session_id, but the big bytes (audio + transcript) point back
    at the originals. If the originals move or get deleted later, the linked
    sample will break if they move. That is the trade-off for saving disk space.

    ``audio_name`` overrides the destination filename for callers that need
    a path-flattened stem (e.g. ``folder__001_clip.wav`` derived from a
    nested audio_in/ path). When omitted, the source filename is used.
    """

    if not audio_path.is_file():
        raise ValueError(f"Audio file does not exist: {audio_path}")
    if not rttm_path.is_file():
        raise ValueError(f"RTTM file does not exist: {rttm_path}")

    normalized_backend = normalize_backend(backend)
    paths = ensure_project_structure(project_name, backend=normalized_backend, root=root)
    audio_filename = sanitize_filename(audio_name or audio_path.name)
    audio_suffix = Path(audio_filename).suffix.lower()
    if audio_suffix not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported training audio type: {audio_suffix or 'unknown'}")
    stem = Path(audio_filename).stem or sanitize_filename(rttm_path.stem)
    if not stem:
        raise ValueError("Unable to derive a sample name from the source files.")

    target_audio_path = paths["audio_dir"] / f"{stem}{audio_suffix}"
    target_rttm_path = paths["rttm_dir"] / f"{stem}.rttm"

    replace_path_with_symlink(target_audio_path, audio_path)

    try:
        rttm_source_text = rttm_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("RTTM upload must be UTF-8 text.") from exc
    write_canonical_rttm_text(target_rttm_path, rttm_source_text, session_id=stem)

    target_transcript_path: Path | None = None
    if transcript_path is not None and transcript_path.is_file():
        transcript_filename = sanitize_filename(transcript_path.name)
        transcript_suffix = Path(transcript_filename).suffix or ".txt"
        target_transcript_path = paths["text_dir"] / f"{stem}{transcript_suffix}"
        replace_path_with_symlink(target_transcript_path, transcript_path)
    elif transcript_text and transcript_text.strip():
        target_transcript_path = paths["text_dir"] / f"{stem}.txt"
        write_text_replacing_existing(
            target_transcript_path,
            transcript_text.strip() + "\n",
            encoding="utf-8",
        )

    return build_sample(target_audio_path, target_rttm_path, target_transcript_path)


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
    copy_stream_replacing_existing(audio_path, audio_stream)
    copy_stream_replacing_existing(rttm_path, rttm_stream)

    if audio_path.stat().st_size == 0:
        raise ValueError("Audio upload is empty.")
    if rttm_path.stat().st_size == 0:
        raise ValueError("RTTM upload is empty.")
    try:
        rttm_text = rttm_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("RTTM upload must be UTF-8 text.") from exc
    write_canonical_rttm_text(rttm_path, rttm_text, session_id=stem)

    transcript_path: Path | None = None
    if transcript_stream is not None and transcript_name:
        transcript_filename = sanitize_filename(transcript_name or f"{stem}.txt")
        transcript_suffix = Path(transcript_filename).suffix or ".txt"
        transcript_path = paths["text_dir"] / f"{stem}{transcript_suffix}"
        copy_stream_replacing_existing(transcript_path, transcript_stream)
        if transcript_path.stat().st_size == 0:
            transcript_path.unlink(missing_ok=True)
            transcript_path = None
    elif transcript_text and transcript_text.strip():
        transcript_path = paths["text_dir"] / f"{stem}.txt"
        write_text_replacing_existing(transcript_path, transcript_text.strip() + "\n", encoding="utf-8")

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


# Memo for probe_media_duration. Per-process, keyed by (path, mtime, size) so
# replacing a file invalidates automatically. Bounded to avoid runaway growth
# when a long-lived dashboard process scans many distinct projects over time.
_PROBE_MEDIA_DURATION_CACHE: dict[tuple[str, float, int], float] = {}
_PROBE_MEDIA_DURATION_LIMIT = 4096


def probe_media_duration(media_path: Path) -> float:
    """Read media duration cheaply when possible and fall back to ffprobe.

    Originally shelled out to ffprobe first, which burns ~10-50 ms per file.
    For a 1700-file stitch run that's a 30-60 s tax in subprocess overhead
    alone, even though most inputs are PCM WAV. Fast paths now: an in-process
    memo keyed by (path, mtime, size); WAV header read; then ffprobe.
    """

    cache_key: tuple[str, float, int] | None = None
    try:
        stat = media_path.stat()
        cache_key = (str(media_path), stat.st_mtime, stat.st_size)
    except OSError:
        cache_key = None
    if cache_key is not None:
        cached = _PROBE_MEDIA_DURATION_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if media_path.suffix.lower() == ".wav":
        try:
            value = probe_wav_duration_from_header(media_path)
            if cache_key is not None:
                _store_probe_duration(cache_key, value)
            return value
        except (OSError, ValueError, struct.error):
            pass
        try:
            with wave.open(str(media_path), "rb") as wav_file:
                frame_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
            if frame_rate > 0 and frame_count > 0:
                value = frame_count / frame_rate
                if cache_key is not None:
                    _store_probe_duration(cache_key, value)
                return value
        except (OSError, wave.Error):
            # Python 3.9's wave module rejects valid WAV variants such as IEEE
            # float WAVs. Read RIFF chunks directly before requiring ffprobe.
            pass

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
                if cache_key is not None:
                    _store_probe_duration(cache_key, duration)
                return duration

    raise RuntimeError(
        f"Unable to determine duration for {media_path}. Install ffprobe or use WAV files."
    )


def _store_probe_duration(cache_key: tuple[str, float, int], value: float) -> None:
    """Insert into the probe cache with a soft size cap."""

    if len(_PROBE_MEDIA_DURATION_CACHE) >= _PROBE_MEDIA_DURATION_LIMIT:
        _PROBE_MEDIA_DURATION_CACHE.pop(next(iter(_PROBE_MEDIA_DURATION_CACHE)), None)
    _PROBE_MEDIA_DURATION_CACHE[cache_key] = value


def probe_wav_duration_from_header(media_path: Path) -> float:
    """Return WAV duration from RIFF chunks without decoding sample data."""

    with media_path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[8:12] != b"WAVE":
            raise ValueError(f"Not a WAVE file: {media_path}")
        if header[0:4] == b"RIFF":
            endian = "<"
        elif header[0:4] == b"RIFX":
            endian = ">"
        else:
            raise ValueError(f"Not a RIFF/RIFX WAVE file: {media_path}")

        sample_rate = 0
        byte_rate = 0
        block_align = 0
        data_size = 0
        file_size = media_path.stat().st_size

        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack(f"{endian}I", chunk_header[4:8])[0]
            if chunk_id == b"fmt ":
                fmt_data = handle.read(chunk_size)
                if len(fmt_data) >= 16:
                    _, _, sample_rate, byte_rate, block_align, _ = struct.unpack(
                        f"{endian}HHIIHH",
                        fmt_data[:16],
                    )
            elif chunk_id == b"data":
                data_start = handle.tell()
                data_size = min(chunk_size, max(0, file_size - data_start))
                handle.seek(chunk_size, os.SEEK_CUR)
            else:
                handle.seek(chunk_size, os.SEEK_CUR)
            if chunk_size % 2:
                handle.seek(1, os.SEEK_CUR)

        if data_size > 0 and byte_rate > 0:
            return data_size / byte_rate
        if data_size > 0 and block_align > 0 and sample_rate > 0:
            return (data_size / block_align) / sample_rate
        raise ValueError(f"WAV duration fields are missing or empty: {media_path}")


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
    missing_rttm_warnings: list[str] | None = None,
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
            message = f"Missing RTTM for training sample '{audio_path.stem}': {rttm_path}"
            if missing_rttm_warnings is not None:
                missing_rttm_warnings.append(message)
                continue
            raise FileNotFoundError(message)
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
                format_rttm_line(
                    session_id=target_path.stem.split(".", 1)[0] or segment.session_id,
                    start=segment.start,
                    duration=segment.duration,
                    speaker=segment.speaker,
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
                        "text": "-",
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
                    format_rttm_line(
                        session_id=sample.stem,
                        start=segment.start,
                        duration=segment.duration,
                        speaker=segment.speaker,
                    )
                    + "\n"
                )


def write_pyannote_subset_uem(path: Path, samples: Sequence[TrainingSample]) -> None:
    """Write one full-file annotated range per sample for pyannote training.

    `sample.duration_seconds` (and `probe_media_duration`) trust the WAV
    header / ffprobe metadata, which works for clean files, but the workspace has
    truncated WAVs where the header claims a 7-minute clip but only 16 s of
    samples actually exist on disk. soundfile counts real frames, so use it
    to set the UEM `annotated` upper bound. Otherwise pyannote's dataloader
    samples chunks past EOF and crashes with `requested chunk … lies outside
    file bounds` partway through epoch 0.
    """

    try:
        import soundfile as sf
    except Exception:  # pragma: no cover - soundfile ships with both backends
        sf = None

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            actual_duration = 0.0
            if sf is not None:
                try:
                    info = sf.info(str(sample.audio_path))
                    if info.samplerate and info.frames:
                        actual_duration = float(info.frames) / float(info.samplerate)
                except Exception:
                    actual_duration = 0.0
            if actual_duration <= 0.0:
                try:
                    actual_duration = float(probe_media_duration(sample.audio_path))
                except Exception:
                    actual_duration = float(sample.duration_seconds)
            handle.write(f"{sample.stem} NA 0.000 {actual_duration:.3f}\n")


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
    # pyannote.database 5.x resolves protocol file paths against the YAML
    # directory (not the cwd or some workspace root). Earlier we made these
    # paths relative to the SeniorDesign root, which left pyannote searching
    # under `<artifacts>/fine_tuning/...` and bailing with FileNotFoundError
    # before the trainer ever started. Absolute paths sidestep the resolver
    # entirely so the layout works no matter where sbatch ends up running.
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
        f"          uri: {json.dumps(str(project_paths_map['train_list_path']))}\n"
        f"          annotation: {json.dumps(str(project_paths_map['train_rttm_path']))}\n"
        f"          annotated: {json.dumps(str(project_paths_map['train_uem_path']))}\n"
        "        development:\n"
        f"          uri: {json.dumps(str(project_paths_map['development_list_path']))}\n"
        f"          annotation: {json.dumps(str(project_paths_map['development_rttm_path']))}\n"
        f"          annotated: {json.dumps(str(project_paths_map['development_uem_path']))}\n"
        "        test:\n"
        f"          uri: {json.dumps(str(project_paths_map['test_list_path']))}\n"
        f"          annotation: {json.dumps(str(project_paths_map['test_rttm_path']))}\n"
        f"          annotated: {json.dumps(str(project_paths_map['test_uem_path']))}\n"
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
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message='Existing precomputed key "annotation" has been modified by a preprocessor.',
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Your `IterableDataset` has `__len__` defined.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=UserWarning,
)

# Compatibility shim for torchaudio 2.10. `AudioMetaData`, `info`, `load`,
# and `list_audio_backends` were removed/renamed, but pyannote.audio 3.4
# still calls them at import time and at runtime (its database loader calls
# `torchaudio.info` to precompute durations). The inference path is fine with
# stubbed versions, but training goes through pyannote.database so the stubs
# need real soundfile-backed implementations or the dataloader crashes.
import torch
import torchaudio
import soundfile as _sf


def _patch_torchaudio_compatibility() -> None:
    # In torchaudio 2.10 the names *exist* but `info`/`load` are
    # delegated to torchcodec, which the cluster can't load (no FFmpeg
    # libavutil on the GPU node). A "missing-only" shim lets the broken
    # torchcodec path win and pyannote dies in the dataloader. We have to
    # unconditionally replace these with soundfile-backed implementations
    # the cluster *does* have.
    class _SoundfileAudioMetaData:
        __slots__ = ("sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding")

        def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    torchaudio.AudioMetaData = _SoundfileAudioMetaData
    torchaudio.list_audio_backends = lambda: ["soundfile"]

    _BITS_BY_SUBTYPE = (
        ("PCM_8", 8), ("PCM_16", 16), ("PCM_24", 24), ("PCM_32", 32),
        ("FLOAT", 32), ("DOUBLE", 64),
    )

    def info(path, *_args, **_kwargs):
        sf_info = _sf.info(str(path))
        subtype = (sf_info.subtype or "").upper()
        bits_per_sample = 0
        for token, bits in _BITS_BY_SUBTYPE:
            if token in subtype:
                bits_per_sample = bits
                break
        return torchaudio.AudioMetaData(
            sample_rate=int(sf_info.samplerate),
            num_frames=int(sf_info.frames),
            num_channels=int(sf_info.channels),
            bits_per_sample=bits_per_sample,
            encoding=subtype or "UNKNOWN",
        )

    torchaudio.info = info

    def load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **_kwargs):
        data, sample_rate = _sf.read(
            str(path),
            start=int(frame_offset),
            frames=-1 if int(num_frames) == -1 else int(num_frames),
            dtype="float32" if normalize else "int16",
            always_2d=True,
        )
        tensor = torch.from_numpy(data)
        if channels_first:
            tensor = tensor.transpose(0, 1).contiguous()
        return tensor, int(sample_rate)

    torchaudio.load = load


_patch_torchaudio_compatibility()

# PyTorch 2.6 flipped torch.load's `weights_only` default to True, which
# refuses to unpickle the TorchVersion / dataclass blobs baked into pyannote
# checkpoints. The diarization backend already patches torch.load via
# `_trusted_torch_load_context`; mirror it here so Model.from_pretrained
# doesn't die with `_pickle.UnpicklingError: Weights only load failed`.
_original_torch_load = torch.load


def _trusted_torch_load(*args, **kwargs):
    # Force-override, do not use setdefault: lightning_fabric's pl_load
    # explicitly passes weights_only=True to torch.load, so a setdefault
    # silently loses to it and the unpickler still bails.
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _trusted_torch_load

import pytorch_lightning as pl
from pyannote.audio import Model
from pyannote.audio.core.io import get_torchaudio_info
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.database import FileFinder, registry
from torch_audiomentations import Identity

DATABASE_CONFIG = Path({json.dumps(str(database_config_path))})
PROTOCOL_NAME = {json.dumps(protocol_name)}
PRETRAINED_MODEL = os.environ.get("PYANNOTE_PRETRAINED_MODEL", {json.dumps(pretrained_model)})
HF_TOKEN = os.environ.get("HF_TOKEN") or True
EXPERIMENTS_DIR = Path(os.environ.get("TRAINING_EXPERIMENT_DIR", {json.dumps(str(experiments_dir))}))
PYANNOTE_NUM_WORKERS = max(0, int(os.environ.get("PYANNOTE_NUM_WORKERS", "4")))

class PyannoteProgressLogger(pl.Callback):
    def _total_batches(self, trainer):
        total = getattr(trainer, "num_training_batches", None)
        try:
            return str(int(total)) if total and total != float("inf") else "unknown"
        except (OverflowError, TypeError, ValueError):
            return "unknown"

    def _loss_fragment(self, outputs):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if hasattr(loss, "detach"):
            try:
                return f" loss={{float(loss.detach().cpu()):.6f}}"
            except (TypeError, ValueError):
                return ""
        return ""

    def on_train_epoch_start(self, trainer, pl_module):
        print(
            f"[pyannote-train] epoch={{trainer.current_epoch + 1}}/{{trainer.max_epochs}} "
            f"step={{trainer.global_step}} batches={{self._total_batches(trainer)}} status=start",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        completed = batch_idx + 1
        total = self._total_batches(trainer)
        should_print = completed == 1 or total == str(completed) or completed % 10 == 0
        if should_print:
            print(
                f"[pyannote-train] epoch={{trainer.current_epoch + 1}}/{{trainer.max_epochs}} "
                f"step={{trainer.global_step}} batch={{completed}}/{{total}}"
                f"{{self._loss_fragment(outputs)}}",
                flush=True,
            )

    def on_train_epoch_end(self, trainer, pl_module):
        print(
            f"[pyannote-train] epoch={{trainer.current_epoch + 1}}/{{trainer.max_epochs}} "
            f"step={{trainer.global_step}} status=end",
            flush=True,
        )

    # Validation hooks below are read-only observers. They just write a
    # val_loss line to stdout so the dashboard can plot the curve from the log.
    def _val_loss_value(self, source):
        if isinstance(source, dict):
            for key in ("val_loss", "validation_loss", "loss"):
                value = source.get(key)
                if value is None:
                    continue
                try:
                    if hasattr(value, "detach"):
                        value = value.detach().cpu()
                    return float(value)
                except (TypeError, ValueError):
                    continue
        elif source is not None:
            try:
                if hasattr(source, "detach"):
                    source = source.detach().cpu()
                return float(source)
            except (TypeError, ValueError):
                return None
        return None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        val = self._val_loss_value(outputs)
        if val is None:
            return
        print(
            f"[pyannote-train] epoch={{trainer.current_epoch + 1}}/{{trainer.max_epochs}} "
            f"step={{trainer.global_step}} val_batch={{batch_idx + 1}} val_loss={{val:.6f}}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = getattr(trainer, "callback_metrics", None) or {{}}
        # Try the common Lightning keys that different pyannote versions log
        # validation loss under. First hit wins; nothing prints if none exist.
        for key in (
            "val_loss",
            "validation_loss",
            "val_loss_epoch",
            "validation_loss_epoch",
            "loss/val",
            "loss/validation",
        ):
            val = self._val_loss_value(metrics.get(key))
            if val is None:
                continue
            print(
                f"[pyannote-train] epoch={{trainer.current_epoch + 1}}/{{trainer.max_epochs}} "
                f"step={{trainer.global_step}} val_loss={{val:.6f}} status=val-end",
                flush=True,
            )
            return


registry.load_database(str(DATABASE_CONFIG))
protocol = registry.get_protocol(
    PROTOCOL_NAME,
    preprocessors={{"audio": FileFinder(), "torchaudio.info": get_torchaudio_info}},
)
model = Model.from_pretrained(PRETRAINED_MODEL, token=HF_TOKEN)
duration = float({duration})
num_samples = max(1, round(duration * int(model.hparams.sample_rate)))
num_frames = max(1, int(model.num_frames(num_samples)))
target_rate = max(1, round(int(model.hparams.sample_rate) * num_frames / num_samples))
print(
    f"[pyannote-train] model={{PRETRAINED_MODEL}} epochs={max_epochs} devices={devices} "
    f"duration={{duration}}s target_rate={{target_rate}} num_workers={{PYANNOTE_NUM_WORKERS}}",
    flush=True,
)
model.task = SpeakerDiarization(
    protocol,
    duration=duration,
    max_speakers_per_chunk={max_speakers_per_chunk},
    max_speakers_per_frame={max_speakers_per_frame},
    num_workers=PYANNOTE_NUM_WORKERS,
    augmentation=Identity(output_type="dict", target_rate=target_rate),
)

accelerator = "gpu" if int({devices}) > 0 else "cpu"
trainer = pl.Trainer(
    devices={devices},
    max_epochs={max_epochs},
    accelerator=accelerator,
    default_root_dir=str(EXPERIMENTS_DIR),
    callbacks=[PyannoteProgressLogger()],
    log_every_n_steps=1,
)
trainer.fit(model)
checkpoint_dir = EXPERIMENTS_DIR / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
final_checkpoint = checkpoint_dir / "last.ckpt"
trainer.save_checkpoint(str(final_checkpoint))
print(f"[pyannote-train] saved_checkpoint={{final_checkpoint}}", flush=True)
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
    detected_nemo_root = _detect_default_nemo_root()
    fallback_nemo_root = str(detected_nemo_root) if detected_nemo_root else ""
    script = f"""#!/usr/bin/env bash
set -euo pipefail

# NEMO_ROOT precedence: caller's env > the value frozen at prepare-time >
# whichever well-known checkout we detected on disk when we generated this
# launcher. The detected fallback means NeMo training keeps working after a
# fresh `prepare` even if the user never fills the "NeMo root" form field.
NEMO_ROOT="${{NEMO_ROOT:-{suggested_nemo_root}}}"
if [ -z "$NEMO_ROOT" ]; then
  NEMO_ROOT={json.dumps(fallback_nemo_root)}
fi
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
# Dataloader worker count tracks the cores SLURM actually granted (falls
# back to 8 for a bare local launch). More workers keep the single GPU fed
# during MSDD embedding extraction, which is the slowest CPU-bound phase.
WORKERS="${{SLURM_CPUS_PER_TASK:-8}}"
# Reduce CUDA memory fragmentation so large embedding tensors don't OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NeMo v2.x dropped the `model.base.*` prefix that the upstream
# multiscale_diar_decoder.py docstring shows. The YAML schema now exposes
# `model.diarizer.speaker_embeddings.model_path` directly. Hydra refuses to
# override a missing struct key, so passing `model.base.*` fails the run
# with "Key 'base' is not in struct".
#
# `trainer.strategy=ddp_find_unused_parameters_true` is the other override
# we need. Lightning 2.x auto-picks DDP even on a single GPU, and the MSDD
# model holds a frozen TitaNet whose params don't participate in the loss;
# without `find_unused_parameters=True`, DDP raises mid-training with
# "It looks like your LightningModule has parameters that were not used in
# producing the loss returned by training_step."
command=(
  "$PYTHON_BIN" multiscale_diar_decoder.py
  --config-path="../conf/neural_diarizer"
  --config-name="{config_name}"
  trainer.devices={devices}
  trainer.max_epochs={max_epochs}
  trainer.strategy=ddp_find_unused_parameters_true
  model.diarizer.speaker_embeddings.model_path="{speaker_model}"
  model.train_ds.manifest_filepath="$TRAIN_MANIFEST"
  model.validation_ds.manifest_filepath="$VAL_MANIFEST"
  model.train_ds.emb_dir="$TRAIN_EMB_DIR"
  model.validation_ds.emb_dir="$VAL_EMB_DIR"
  ++model.train_ds.num_workers="$WORKERS"
  ++model.validation_ds.num_workers="$WORKERS"
  exp_manager.name="$EXP_NAME"
  exp_manager.exp_dir="$EXP_DIR"
  +exp_manager.checkpoint_callback_params.save_last=false
)

echo "Running NeMo fine-tuning command:"
printf "  %q" "${{command[@]}}"
printf "\\n"
# Not `exec` because we need to run the .nemo export safety net afterwards.
"${{command[@]}}"

# .nemo export safety net. NeMo only packages the final .nemo when training
# finishes normally; a run killed by the SLURM time limit leaves .ckpt only,
# and the diarization backend can restore a .nemo archive but NOT a raw
# Lightning .ckpt. If training finished without a .nemo, rebuild one from the
# best surviving checkpoint. Best-effort: a failure here is logged but does
# not fail the run, since training itself did complete.
if ! find "$EXP_DIR" -name "*.nemo" -print -quit 2>/dev/null | grep -q .; then
  echo "No .nemo produced by training; exporting from the best checkpoint..."
  SENIOR_DESIGN_ROOT={json.dumps(str(PROJECT_ROOT))}
  PYTHONPATH="$SENIOR_DESIGN_ROOT${{PYTHONPATH:+:$PYTHONPATH}}" \\
    "$PYTHON_BIN" -c 'import sys; from fine_tuning_manager import export_best_checkpoint_to_nemo; print("export result:", export_best_checkpoint_to_nemo(sys.argv[1]))' "$EXP_DIR" \\
    || echo "WARNING: .nemo export from checkpoint failed; model will not appear in diarization until re-trained." >&2
fi
"""
    launch_script_path.parent.mkdir(parents=True, exist_ok=True)
    launch_script_path.write_text(script, encoding="utf-8")
    launch_script_path.chmod(0o755)

# WAVE module + venv shared across the diarization pipeline.
# Loaded modules + LD_LIBRARY_PATH need to be set the same way the production
# `run_site_diarization.sbatch` does, otherwise the venv's python imports
# explode on `_sqlite3.so: undefined symbol: sqlite3_deserialize` because the
# stock cluster libs lag the GCCcore ones our binaries were linked against.
_PYTHON_MODULE = "Python/3.12.3-GCCcore-14.2.0"
_RUNTIME_ENV_SCRIPT = PROJECT_ROOT / "scheduler" / "sbatch_runtime_env.sh"
_SBATCH_RUNTIME_DIR = PROJECT_ROOT / "sbatch_runtime"
_BACKEND_VENV = {
    "nemo": _SBATCH_RUNTIME_DIR / ".venv",
    "pyannote": _SBATCH_RUNTIME_DIR / ".venv_pyannote",
}
# NeMo MSDD fine-tuning needs a real source checkout because the upstream
# `multiscale_diar_decoder.py` lives under `examples/`, not in the pip
# distribution. The runbook recommends cloning to the workspace's parent so the
# checkout outlives any single project. Auto-defaulting here means the user
# can leave the dashboard's "NeMo root" field blank as long as the clone exists.
_NEMO_ROOT_DEFAULT_CANDIDATES = (
    PROJECT_ROOT.parent / "NeMo",
    Path.home() / "NeMo",
)


def _detect_default_nemo_root() -> Path | None:
    """Return the first existing well-known NeMo checkout, if any."""

    for candidate in _NEMO_ROOT_DEFAULT_CANDIDATES:
        if (candidate / "examples" / "speaker_tasks" / "diarization" / "neural_diarizer").is_dir():
            return candidate
    return None


def _runtime_env_block(*, backend: str) -> str:
    """Return the bash snippet that prepares modules, venv, and LD_LIBRARY_PATH.

    Mirrors the working `run_site_diarization.sbatch` so the fine-tuning jobs
    pick up the same cluster runtime. The dashboard's `sys.executable` points
    at the dashboard's own venv, which has no lightning or pyannote and lacks
    the module-loaded GCCcore on its LD path.
    """

    backend_key = normalize_backend(backend)
    default_venv = _BACKEND_VENV[backend_key]
    return "\n".join(
        [
            f"module load {_PYTHON_MODULE}",
            f"_FT_DEFAULT_VENV={json.dumps(str(default_venv))}",
            f"_FT_RUNTIME_ENV_SCRIPT={json.dumps(str(_RUNTIME_ENV_SCRIPT))}",
            'PYTHON_BIN="${PYTHON_BIN:-$_FT_DEFAULT_VENV/bin/python3}"',
            # The dashboard passes its own sys.executable as PYTHON_BIN.
            # On WAVE that is /usr/bin/python3, which has none of the ML
            # packages and the GPU job dies on import. Force PYTHON_BIN back
            # to the backend venv unless the caller picked one inside it.
            'case "$PYTHON_BIN" in',
            '  "$_FT_DEFAULT_VENV/bin/"*)',
            "    ;;",
            "  *)",
            '    PYTHON_BIN="$_FT_DEFAULT_VENV/bin/python3"',
            "    ;;",
            "esac",
            "export PYTHON_BIN",
            'if [ -d "$_FT_DEFAULT_VENV" ]; then',
            "  # shellcheck disable=SC1091",
            '  source "$_FT_DEFAULT_VENV/bin/activate"',
            "fi",
            'if [ -f "$_FT_RUNTIME_ENV_SCRIPT" ]; then',
            "  # shellcheck disable=SC1090",
            '  source "$_FT_RUNTIME_ENV_SCRIPT"',
            '  if [ -n "${EBROOTGCCCORE:-}" ]; then',
            '    append_ld_library_path "$EBROOTGCCCORE/lib64"',
            "  fi",
            "  expose_venv_native_libraries",
            "fi",
        ]
    )


def export_best_checkpoint_to_nemo(experiment_dir: Path | str) -> Path | None:
    """Rebuild a loadable ``.nemo`` from the best surviving Lightning checkpoint.

    NeMo only packages the ``.nemo`` when ``trainer.fit()`` returns normally;
    a run killed by the SLURM time limit leaves ``.ckpt`` files only, and the
    diarization backend's ``NeuralDiarizer`` can restore a ``.nemo`` archive
    but not a raw checkpoint. This is the safety net: it finds the best
    checkpoint under ``experiment_dir`` and exports a ``.nemo`` next to it.

    Two real-world issues are handled:
      * MSDD-only fine-tune checkpoints drop ``speaker_model_cfg`` from their
        config, which crashes model construction. We re-inject it from the
        base ``titanet_large`` speaker model.
      * A checkpoint written while SLURM was killing the job may be corrupt.
        We walk checkpoints best-first and skip any that fail to load.

    Returns the ``.nemo`` path on success, or None if every checkpoint is
    missing/corrupt. NeMo + torch are imported lazily so the dashboard
    (which also imports this module) never pays for the heavy import.
    """

    experiment_dir = Path(experiment_dir)
    existing = next(iter(sorted(experiment_dir.rglob("*.nemo"))), None)
    if existing is not None:
        return existing
    checkpoints = [path for path in experiment_dir.rglob("*.ckpt") if path.is_file()]
    if not checkpoints:
        return None

    def _val_loss(path: Path) -> float:
        match = re.search(r"val_loss=([0-9.]+)", path.name)
        return float(match.group(1)) if match else float("inf")

    # Lowest validation loss first; newest as the tie-breaker.
    checkpoints.sort(key=lambda path: (_val_loss(path), -path.stat().st_mtime))

    import torch  # noqa: PLC0415 - lazy: heavy import, only needed at export time
    from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel  # noqa: PLC0415

    speaker_model_cfg = None
    for checkpoint in checkpoints:
        try:
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - corrupt ckpt: skip, try the next
            print(f"  skip {checkpoint.name}: cannot read checkpoint ({exc})")
            continue
        # ``hyper_parameters`` is an OmegaConf DictConfig (not a plain dict),
        # so probe it with ``in`` / item access rather than isinstance(dict).
        # NeMo stores the model config in ``hyper_parameters`` as an OmegaConf
        # DictConfig, not a plain dict with a nested "cfg" key.
        cfg = payload.get("hyper_parameters") if isinstance(payload, dict) else None
        # MSDD fine-tune configs drop speaker_model_cfg; NeuralDiarizer and
        # load_from_checkpoint both need it. Pull it off the base TitaNet.
        load_path = checkpoint
        if cfg is not None and "speaker_model_cfg" not in cfg:
            if speaker_model_cfg is None:
                from nemo.collections.asr.models import EncDecSpeakerLabelModel  # noqa: PLC0415
                speaker_model_cfg = EncDecSpeakerLabelModel.from_pretrained(
                    "titanet_large", map_location="cpu"
                ).cfg
            try:
                from omegaconf import open_dict  # noqa: PLC0415
                with open_dict(cfg):
                    cfg["speaker_model_cfg"] = speaker_model_cfg
            except Exception:  # noqa: BLE001 - plain-dict cfg: assign directly
                cfg["speaker_model_cfg"] = speaker_model_cfg
            load_path = checkpoint.with_name(checkpoint.stem + ".speakercfg.ckpt")
            torch.save(payload, str(load_path))
        try:
            model = EncDecDiarLabelModel.load_from_checkpoint(str(load_path), map_location="cpu")
        except Exception as exc:  # noqa: BLE001 - try the next checkpoint
            print(f"  skip {checkpoint.name}: model load failed ({exc})")
            continue
        out_path = checkpoint.parent / f"{experiment_dir.name}.nemo"
        model.save_to(str(out_path))
        print(f"  exported {out_path}")
        return out_path
    return None


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
    runtime_block = _runtime_env_block(backend="nemo")
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
            runtime_block,
            f"command=(bash {json.dumps(str(launch_script_path))})",
            'echo "Running fine-tuning launch command:"',
            'printf "  %q" "${command[@]}"',
            'printf "\\n"',
            '"${command[@]}"',
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

command=("$PYTHON_BIN" {json.dumps(str(training_script_path))})
echo "Running pyannote fine-tuning command:"
printf "  %q" "${{command[@]}}"
printf "\\n"
exec "${{command[@]}}"
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
    runtime_block = _runtime_env_block(backend="pyannote")
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
            runtime_block,
            f"command=(bash {json.dumps(str(launch_script_path))})",
            'echo "Running fine-tuning launch command:"',
            'printf "  %q" "${command[@]}"',
            'printf "\\n"',
            '"${command[@]}"',
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
        paths["train_list_path"].parent,
        paths["train_rttm_path"].parent,
        paths["train_uem_path"].parent,
    ]
    for generated_dir in generated_directories:
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
        generated_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    samples = discover_samples(
        project_name,
        backend=normalized_backend,
        missing_rttm_warnings=warnings,
        root=root,
    )
    if not samples:
        detail = f" First issue: {warnings[0]}" if warnings else ""
        raise ValueError(
            f"No training samples found in {paths['audio_dir']}. "
            "Add at least one audio+RTTM pair, or check that the selected backend matches the project."
            f"{detail}"
        )
    train_samples, validation_samples = split_samples(samples, train_ratio)

    session_manifest_train = paths["manifests_dir"] / "train_session_manifest.jsonl"
    session_manifest_validation = paths["manifests_dir"] / "validation_session_manifest.jsonl"
    msdd_manifest_train = paths["manifests_dir"] / "train_msdd_manifest.jsonl"
    msdd_manifest_validation = paths["manifests_dir"] / "validation_msdd_manifest.jsonl"
    primary_artifacts: list[Path] = []

    if normalized_backend == "nemo":
        msdd_train_samples = train_samples
        msdd_validation_samples = validation_samples
        if not any(sample.num_speakers >= 2 for sample in train_samples) or not any(
            sample.num_speakers >= 2 for sample in validation_samples
        ):
            msdd_candidates = [sample for sample in samples if sample.num_speakers >= 2]
            if msdd_candidates:
                msdd_train_samples, msdd_validation_samples = split_samples(
                    msdd_candidates,
                    train_ratio,
                )
                excluded_count = len(samples) - len(msdd_candidates)
                if excluded_count:
                    warnings.append(
                        f"{excluded_count} single-speaker sample(s) were kept in session manifests "
                        "but excluded from NeMo MSDD pairwise rows."
                    )
                warnings.append(
                    "Adjusted NeMo MSDD train/validation split to use the available multi-speaker sample(s)."
                )

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
        for sample in msdd_train_samples:
            rows, sample_warnings = build_msdd_rows_for_sample(
                sample=sample,
                pairwise_dir=paths["pairwise_train_dir"],
                base_window=base_window,
                base_shift=base_shift,
                step_count=step_count,
            )
            train_rows.extend(rows)
            warnings.extend(sample_warnings)
        for sample in msdd_validation_samples:
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



# Pulls val_loss values out of a pyannote .out file for the dashboard's
# per-epoch chart. Does not touch the training process at all.
_PYANNOTE_VAL_LOSS_LINE_RE = re.compile(
    r"\[pyannote-train\][^\n]*?epoch=(\d+)/(\d+)[^\n]*?"
    r"(?:step=(\d+)[^\n]*?)?"
    r"(?:val_batch=(\d+)[^\n]*?)?"
    r"val_loss=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)

# Bounded in-process memo keyed by (path, mtime, size). The fine-tuning page
# can re-render every poll while training is active; without this we'd read
# every .out file end-to-end on every render. Capped so a wild test run with
# thousands of distinct files cannot blow up RAM.
_PYANNOTE_VAL_LOSS_CACHE: dict[tuple[str, float, int], list[dict[str, object]]] = {}
_PYANNOTE_VAL_LOSS_CACHE_LIMIT = 64


def parse_pyannote_val_loss(out_path: Path, *, max_points: int = 200) -> list[dict[str, object]]:
    """Scan a pyannote training .out file and return val_loss data points.

    Each entry is ``{"epoch", "max_epochs", "step", "val_batch", "val_loss",
    "stage"}``. ``stage`` is ``"epoch"`` for end-of-validation summaries and
    ``"batch"`` for per-batch points. Newest points come last so the UI can
    plot them in chronological order.

    Cached by (path, mtime, size); a still-growing .out file invalidates the
    cache automatically because its size keeps changing, while idle/finished
    runs read from the memo for free.
    """

    if not isinstance(out_path, Path):
        out_path = Path(str(out_path))
    try:
        stat = out_path.stat()
    except OSError:
        return []
    if not out_path.is_file():
        return []
    cache_key = (str(out_path), stat.st_mtime, stat.st_size)
    cached = _PYANNOTE_VAL_LOSS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        text_blob = out_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    points: list[dict[str, object]] = []
    for match in _PYANNOTE_VAL_LOSS_LINE_RE.finditer(text_blob):
        try:
            epoch = int(match.group(1))
            max_epochs = int(match.group(2))
            val_loss = float(match.group(5))
        except (TypeError, ValueError):
            continue
        step_raw = match.group(3)
        batch_raw = match.group(4)
        try:
            step = int(step_raw) if step_raw is not None else None
        except (TypeError, ValueError):
            step = None
        try:
            val_batch = int(batch_raw) if batch_raw is not None else None
        except (TypeError, ValueError):
            val_batch = None
        points.append(
            {
                "epoch": epoch,
                "max_epochs": max_epochs,
                "step": step,
                "val_batch": val_batch,
                "val_loss": val_loss,
                "stage": "batch" if val_batch is not None else "epoch",
            }
        )
    if len(points) > max_points:
        points = points[-max_points:]
    if len(_PYANNOTE_VAL_LOSS_CACHE) >= _PYANNOTE_VAL_LOSS_CACHE_LIMIT:
        # Drop one arbitrary entry; this is a process-local cache so LRU is overkill.
        _PYANNOTE_VAL_LOSS_CACHE.pop(next(iter(_PYANNOTE_VAL_LOSS_CACHE)), None)
    _PYANNOTE_VAL_LOSS_CACHE[cache_key] = points
    return points


def find_pyannote_run_out_log(run_metadata: dict[str, object], project_dir: Path) -> Path | None:
    """Locate the pyannote .out file for a run, sbatch or local fallback.

    sbatch jobs land in ``<project>/artifacts/slurm_logs/pyannote_finetune_<jobid>.out``;
    the local-fallback wrapper writes plain ``stdout.log`` inside the run
    directory. Either layout works for parse_pyannote_val_loss.
    """

    job_id = str(run_metadata.get("job_id", "") or "").strip()
    if job_id:
        slurm_logs = project_dir / "artifacts" / "slurm_logs" / f"pyannote_finetune_{job_id}.out"
        if slurm_logs.is_file():
            return slurm_logs
    stdout_path = run_metadata.get("stdout_path")
    if stdout_path:
        candidate = Path(str(stdout_path))
        if candidate.is_file():
            return candidate
    return None


SLURM_ACTIVE_STATES = {
    "configuring",
    "completing",
    "pending",
    "requeued",
    "resizing",
    "running",
    "signaling",
    "staged_out",
    "suspended",
}
SLURM_FAILED_STATES = {
    "boot_fail",
    "cancelled",
    "deadline",
    "failed",
    "node_fail",
    "out_of_memory",
    "preempted",
    "revoked",
    "special_exit",
    "timeout",
}


def slurm_state_to_run_status(raw_state: str) -> str:
    """Convert Slurm's state names into this module's run-status vocabulary."""

    state = (raw_state or "").strip().lower().replace("-", "_")
    state_key = state.split()[0] if state else ""
    if not state_key:
        return ""
    if state_key in {"completed", "complete"}:
        return "succeeded"
    if state_key in SLURM_FAILED_STATES:
        return "failed"
    if state_key in {"pending", "requeued"}:
        return "submitted"
    if state_key in SLURM_ACTIVE_STATES:
        return "running"
    return state_key


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
            queue_status = slurm_state_to_run_status(slurm_job_state(job_id))
            if queue_status:
                return queue_status
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
            "auto_train": bool(display_payload.get("auto_train")),
            "auto_train_pending": bool(display_payload.get("auto_train_pending")),
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
        # Skip stale empty folders (legacy test fixtures, etc.) but keep
        # explicitly-created project shells that the user registered through
        # "Create Empty Project". The marker is written by the create-project
        # handler so we can tell intentional-but-empty projects from the
        # leftovers of an aborted upload.
        manually_created = bool(display_payload.get("manually_created"))
        if (
            sample_count == 0
            and not summary.get("prepared")
            and latest_run is None
            and not manually_created
        ):
            continue
        if manually_created:
            summary["manually_created"] = True
        summaries.append(summary)
    return summaries


def _read_base_model_snapshot(metadata_path: Path, normalized_backend: str) -> dict[str, object]:
    """Pull the base/pretrained model details out of a prepared project's metadata.json.

    Different backends record the starting checkpoint under different keys
    (pyannote stores `pyannote_pretrained_model`, NeMo stores `speaker_model` /
    `config_name`), so we normalize them here into one structure the runs UI
    can render without caring which backend produced it.
    """

    snapshot: dict[str, object] = {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return snapshot
    if not isinstance(metadata, dict):
        return snapshot
    if normalized_backend == "pyannote":
        model = str(metadata.get("pyannote_pretrained_model") or "").strip()
        if model:
            snapshot["base_model"] = model
            snapshot["base_model_kind"] = "pyannote_pretrained_model"
    else:
        speaker_model = str(metadata.get("speaker_model") or "").strip()
        if speaker_model:
            snapshot["base_model"] = speaker_model
            snapshot["base_model_kind"] = "nemo_speaker_model"
        config_name = str(metadata.get("config_name") or "").strip()
        if config_name:
            snapshot["base_model_config"] = config_name
    return snapshot


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
    # Snapshot the base model from the project's prepare-time metadata so the
    # run record stands on its own. The project metadata gets rewritten every
    # `prepare`, so without copying this in here we'd lose track of which
    # checkpoint a finished run was actually based on.
    base_model_snapshot = _read_base_model_snapshot(metadata_path, normalized_backend)
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
        submit_command = [sbatch_bin, str(sbatch_script_path)]
        submit_command_text = shlex.join(submit_command)
        print(f"Submitting Slurm training job: {submit_command_text}", flush=True)
        completed = subprocess.run(
            submit_command,
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
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        stdout_path.write_text(
            f"Submit command: {submit_command_text}\n{completed.stdout or ''}",
            encoding="utf-8",
        )
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
            "submit_command": submit_command,
            "submit_command_text": submit_command_text,
            "sbatch_script_path": str(sbatch_script_path),
            "nemo_root": str(nemo_root.resolve()) if nemo_root else "",
            "python_bin": python_bin,
            "submission_returncode": completed.returncode,
            **base_model_snapshot,
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
        **base_model_snapshot,
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
