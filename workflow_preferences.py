#!/usr/bin/env python3
"""Persist the dashboard defaults that shape conversion, diarization, and training."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PREFERENCES_PATH = PROJECT_ROOT / "workflow_preferences.json"
DIARIZATION_BACKENDS = ("nemo", "pyannote")
DIARIZATION_BACKEND_LABELS = {
    "nemo": "NeMo",
    "pyannote": "pyannote",
}
DEFAULT_PYANNOTE_PIPELINE_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_PYANNOTE_SEGMENTATION_MODEL = "diarizers-community/speaker-segmentation-fine-tuned-callhome-zho"

DEFAULT_PREFERENCES: dict[str, object] = {
    "default_backend": "nemo",
    "default_diarization_model_key": "nemo",
    "whisper_model": "medium.en",
    "whisper_batch_size": 8,
    "pyannote_pipeline_model": DEFAULT_PYANNOTE_PIPELINE_MODEL,
    "pyannote_segmentation_model": DEFAULT_PYANNOTE_SEGMENTATION_MODEL,
    "youtube_prefer_all": True,
    "fine_tuning_backend": "nemo",
    "pyannote_fine_tuning": {
        "pretrained_model": "pyannote/segmentation-3.0",
        "duration": 10.0,
        "max_speakers_per_chunk": 3,
        "max_speakers_per_frame": 2,
        "max_epochs": 5,
        "devices": 1,
    },
    "nemo_fine_tuning": {
        "config_name": "msdd_5scl_15_05_50Povl_256x3x32x2.yaml",
        "speaker_model": "titanet_large",
        "train_ratio": 0.8,
        "base_window": 0.5,
        "base_shift": 0.25,
        "step_count": 50,
        "max_epochs": 20,
        "devices": 1,
        "slurm_partition": "gpu",
        "slurm_time": "04:00:00",
        "slurm_memory": "32G",
        "slurm_cpus": 6,
        "slurm_gpus": 1,
    },
}


def _merge_defaults(defaults: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    """Apply user overrides onto the nested default structure."""

    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)  # type: ignore[index]
        else:
            merged[key] = value
    return merged


def load_preferences(*, root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Load persisted dashboard defaults, falling back to the baked-in set."""

    preferences_path = root / PREFERENCES_PATH.name
    if not preferences_path.is_file():
        return deepcopy(DEFAULT_PREFERENCES)
    try:
        raw = json.loads(preferences_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return deepcopy(DEFAULT_PREFERENCES)
    if not isinstance(raw, dict):
        return deepcopy(DEFAULT_PREFERENCES)
    return _merge_defaults(DEFAULT_PREFERENCES, raw)


def save_preferences(preferences: dict[str, object], *, root: Path = PROJECT_ROOT) -> Path:
    """Persist preferences in a stable JSON file at the project root."""

    merged = _merge_defaults(DEFAULT_PREFERENCES, preferences)
    preferences_path = root / PREFERENCES_PATH.name
    preferences_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return preferences_path


def normalize_diarization_backend(backend: str | None) -> str:
    """Normalize the diarization backend label used by site runs."""

    normalized = (backend or str(DEFAULT_PREFERENCES["default_backend"])).strip().lower()
    if normalized == "msdd":
        normalized = "nemo"
    if normalized not in DIARIZATION_BACKENDS:
        raise ValueError(f"Unsupported diarization backend: {backend}")
    return normalized
