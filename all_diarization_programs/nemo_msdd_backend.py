import atexit
import json
import os
import shutil
import tarfile
import tempfile
import wave

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
from omegaconf import OmegaConf


# Cached YAML block + temp dirs so we patch each fine-tuned .nemo at most once
# per Python process. NeuralDiarizer holds the loaded weights in memory after
# __init__ returns, so the patched temp file only needs to outlive that call,
# but keeping it for the process lifetime is cheaper than reference-counting.
_SPEAKER_MODEL_CFG_BLOCK: Optional[str] = None
_PATCHED_NEMO_PATHS: dict[str, str] = {}
_PATCH_TEMP_DIRS: list[str] = []


def _cleanup_patch_temp_dirs() -> None:
    while _PATCH_TEMP_DIRS:
        path = _PATCH_TEMP_DIRS.pop()
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_patch_temp_dirs)


def save_mono_wav(path: str, audio: torch.Tensor, sample_rate: int) -> None:
    """
    Save mono WAV without torchaudio/torchcodec so we avoid ffmpeg shared-lib issues
    on HPC environments.
    """
    if audio.ndim == 2:
        mono = audio[0]
    elif audio.ndim == 1:
        mono = audio
    else:
        raise ValueError(f"Expected 1D/2D audio tensor, got shape {tuple(audio.shape)}")

    mono = mono.detach().cpu().float().clamp(-1.0, 1.0).numpy()
    pcm16 = (mono * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def _extract_yaml_top_level_block(text: str, key: str) -> Optional[str]:
    """Slice a top-level YAML block starting at ``key:`` out of a config text.

    Returns ``None`` if the key isn't present. Preserves indentation so the
    extracted block can be reattached as another file's top-level entry.
    """

    lines = text.splitlines()
    start: Optional[int] = None
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        first = lines[i][:1] if lines[i] else ""
        if first and not first.isspace():
            end = i
            break
    return "\n".join(lines[start:end]) + "\n"


def _read_nemo_config_text(nemo_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Return (config text, archive member name) for the model_config.yaml inside a .nemo."""

    with tarfile.open(nemo_path, "r") as tar:
        for candidate in ("./model_config.yaml", "model_config.yaml"):
            try:
                member = tar.getmember(candidate)
            except KeyError:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                return None, None
            return handle.read().decode("utf-8"), candidate
    return None, None


def _find_base_msdd_nemo() -> Optional[Path]:
    """Return the local cache path for the base diar_msdd_telephonic .nemo, if available.

    Tries the standard NeMo cache first. If absent, asks NeMo to download it
    (one-time cost on the very first run) and re-checks the cache.
    """

    cache_root = Path.home() / ".cache" / "torch" / "NeMo"
    if cache_root.is_dir():
        for candidate in cache_root.rglob("diar_msdd_telephonic.nemo"):
            if candidate.is_file():
                return candidate
    try:
        # Force-populate the cache. We discard the model after the side effect.
        from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel
        EncDecDiarLabelModel.from_pretrained(
            model_name="diar_msdd_telephonic",
            map_location="cpu",
            strict=False,
        )
    except Exception:
        return None
    if cache_root.is_dir():
        for candidate in cache_root.rglob("diar_msdd_telephonic.nemo"):
            if candidate.is_file():
                return candidate
    return None


def _load_speaker_model_cfg_block() -> Optional[str]:
    """Lazy-load the speaker_model_cfg YAML block from the base MSDD checkpoint."""

    global _SPEAKER_MODEL_CFG_BLOCK
    if _SPEAKER_MODEL_CFG_BLOCK is not None:
        return _SPEAKER_MODEL_CFG_BLOCK
    base_path = _find_base_msdd_nemo()
    if base_path is None:
        return None
    text, _ = _read_nemo_config_text(base_path)
    if text is None:
        return None
    block = _extract_yaml_top_level_block(text, "speaker_model_cfg")
    if block:
        _SPEAKER_MODEL_CFG_BLOCK = block
    return block


def _ensure_msdd_loadable(msdd_model_path: str) -> str:
    """Patch fine-tuned MSDD .nemo files that lack ``speaker_model_cfg``.

    NeMo's ``NeuralDiarizer`` reads ``cfg.speaker_model_cfg`` to rebuild the
    embedded TitaNet submodel during ``restore_from``. MSDD-only fine-tuning
    saves drop that key, which breaks inference even though the trained
    weights are valid. We copy the block from the base diar_msdd_telephonic
    config and write a patched .nemo to a temp path, leaving the original on
    disk untouched. Result paths are cached per process so a batch of files
    only repacks each fine-tuned checkpoint once.
    """

    if not msdd_model_path:
        return msdd_model_path
    cached = _PATCHED_NEMO_PATHS.get(msdd_model_path)
    if cached is not None:
        return cached
    path = Path(msdd_model_path)
    if not path.is_file() or path.suffix.lower() != ".nemo":
        return msdd_model_path

    cfg_text, member_name = _read_nemo_config_text(path)
    if cfg_text is None:
        return msdd_model_path
    if any(line.startswith("speaker_model_cfg:") for line in cfg_text.splitlines()):
        # Already self-sufficient; load directly.
        _PATCHED_NEMO_PATHS[msdd_model_path] = msdd_model_path
        return msdd_model_path

    speaker_block = _load_speaker_model_cfg_block()
    if not speaker_block:
        # Couldn't locate a base config; defer to NeMo's original error path.
        return msdd_model_path

    if not cfg_text.endswith("\n"):
        cfg_text += "\n"
    patched_cfg_text = cfg_text + speaker_block

    work_dir = Path(tempfile.mkdtemp(prefix="msdd_patched_"))
    _PATCH_TEMP_DIRS.append(str(work_dir))
    extract_dir = work_dir / "extracted"
    extract_dir.mkdir()
    with tarfile.open(path, "r") as tar:
        tar.extractall(extract_dir)
    # Find where the config landed (some .nemo archives use ``./``-prefixed names).
    cfg_candidate = extract_dir / "model_config.yaml"
    if not cfg_candidate.is_file():
        match = next(extract_dir.rglob("model_config.yaml"), None)
        if match is None:
            return msdd_model_path
        cfg_candidate = match
    cfg_candidate.write_text(patched_cfg_text, encoding="utf-8")

    patched_path = work_dir / path.name
    arc_prefix = "./" if member_name and member_name.startswith("./") else ""
    with tarfile.open(patched_path, "w") as tar:
        for entry in sorted(extract_dir.rglob("*")):
            if not entry.is_file():
                continue
            relative = entry.relative_to(extract_dir).as_posix()
            tar.add(entry, arcname=f"{arc_prefix}{relative}" if arc_prefix else relative)

    patched_str = str(patched_path)
    _PATCHED_NEMO_PATHS[msdd_model_path] = patched_str
    return patched_str


class NemoMsddDiarizer:
    def __init__(
        self,
        device: Union[str, torch.device],
        msdd_model_path: str = "",
    ):
        # Fine-tuned MSDD checkpoints from this project are saved without
        # ``speaker_model_cfg``, which crashes NeuralDiarizer's restore path
        # before any audio is processed. Patch the .nemo on the fly.
        loadable_path = _ensure_msdd_loadable(msdd_model_path)
        self.model: NeuralDiarizer = NeuralDiarizer(
            cfg=create_config(msdd_model_path=loadable_path)
        ).to(device)

    def diarize(self, audio: torch.Tensor, sample_rate: int = 16000):
        with tempfile.TemporaryDirectory() as temp_path:
            mono_path = os.path.join(temp_path, "mono_file.wav")
            save_mono_wav(mono_path, audio, sample_rate)

            manifest_path = os.path.join(temp_path, "manifest.json")
            meta = {
                "audio_filepath": mono_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "rttm_filepath": None,
                "uem_filepath": None,
            }

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            self.model._initialize_configs(
                manifest_path=manifest_path,
                max_speakers=8,
                num_speakers=None,
                tmpdir=temp_path,
                batch_size=24,
                num_workers=0,
                verbose=True,
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.out_dir = (
                temp_path
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.manifest_filepath = (
                manifest_path
            )
            self.model.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.model.diarize()

            pred_labels_clus = rttm_to_labels(
                os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
            )

            labels = []
            for label in pred_labels_clus:
                start, end, speaker = label.split()
                start, end = float(start), float(end)
                labels.append(
                    (
                        int(start * 1000),
                        int(end * 1000),
                        int(speaker.split("_")[1]),
                    )
                )

            labels = sorted(labels, key=lambda x: x[0])

        return labels


def create_config(msdd_model_path: str = ""):
    config = OmegaConf.load(
        os.path.join(os.path.dirname(__file__), "nemo_diarization_telephonic.yaml")
    )
    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"

    config.diarizer.out_dir = None
    config.diarizer.manifest_filepath = None
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = False
    config.diarizer.clustering.parameters.oracle_num_speakers = False
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (msdd_model_path or "diar_msdd_telephonic").strip()

    return config
