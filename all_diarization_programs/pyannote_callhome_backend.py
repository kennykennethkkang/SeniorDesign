import inspect
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

from typing import Union

import torch
import torchaudio
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def _patch_torchaudio_compatibility() -> None:
    """Keep pyannote.audio 3.x importable with newer torchaudio builds."""

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            pass

        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "info"):
        def info(*_args, **_kwargs):
            raise RuntimeError("torchaudio.info is unavailable in this torchaudio build")

        torchaudio.info = info

    if not hasattr(torchaudio, "load"):
        def load(*_args, **_kwargs):
            raise RuntimeError("torchaudio.load is unavailable in this torchaudio build")

        torchaudio.load = load


_patch_torchaudio_compatibility()

from pyannote.audio import Model, Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.models.segmentation import PyanNet

DEFAULT_PYANNOTE_PIPELINE_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_PYANNOTE_SEGMENTATION_MODEL = "diarizers-community/speaker-segmentation-fine-tuned-callhome-zho"


def _load_pipeline(model_id: str, token: str) -> Pipeline:
    """Load pyannote pipelines across the 3.x and 4.x token API change."""

    parameters = inspect.signature(Pipeline.from_pretrained).parameters
    if "token" in parameters:
        return Pipeline.from_pretrained(model_id, token=token)
    return Pipeline.from_pretrained(model_id, use_auth_token=token)


def _load_model_from_pretrained(model_id_or_path: str, token: str) -> Model:
    """Load a pyannote model across the 3.x and 4.x token API change."""

    parameters = inspect.signature(Model.from_pretrained).parameters
    if "token" in parameters:
        return Model.from_pretrained(model_id_or_path, token=token)
    return Model.from_pretrained(model_id_or_path, use_auth_token=token)


@contextmanager
def _trusted_torch_load_context():
    """Load trusted pyannote checkpoints with PyTorch 2.6+ compatible defaults."""

    original_load = torch.load

    def load_compat(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = load_compat
    try:
        yield
    finally:
        torch.load = original_load


def _segmentation_snapshot(model_id_or_path: str, token: str) -> Path:
    """Resolve a Hugging Face model id or local snapshot path to files on disk."""

    local_path = Path(model_id_or_path).expanduser()
    if local_path.is_dir():
        return local_path
    return Path(snapshot_download(model_id_or_path, token=token))


def _build_segmentation_specifications(config: dict[str, object]) -> Specifications:
    max_speakers_per_frame = config.get("max_speakers_per_frame")
    max_speakers_per_chunk = int(config.get("max_speakers_per_chunk") or 0)
    if max_speakers_per_chunk < 1:
        raise RuntimeError("Custom pyannote segmentation config is missing max_speakers_per_chunk.")

    problem = (
        Problem.MULTI_LABEL_CLASSIFICATION
        if max_speakers_per_frame is None
        else Problem.MONO_LABEL_CLASSIFICATION
    )
    warm_up = config.get("warm_up") or (0.0, 0.0)
    return Specifications(
        problem=problem,
        resolution=Resolution.FRAME,
        duration=float(config.get("chunk_duration") or 10.0),
        min_duration=config.get("min_duration"),
        warm_up=tuple(float(value) for value in warm_up),
        classes=[f"speaker#{index + 1}" for index in range(max_speakers_per_chunk)],
        powerset_max_classes=max_speakers_per_frame,
        permutation_invariant=True,
    )


def _infer_lstm_layers(state_dict: dict[str, torch.Tensor]) -> int:
    layers = set()
    for key in state_dict:
        match = re.match(r"lstm\.weight_ih_l(\d+)(?:_reverse)?$", key)
        if match:
            layers.add(int(match.group(1)))
    return max(layers) + 1 if layers else 2


def _load_pyannote_segmentation_model(model_id_or_path: str, token: str, device: torch.device) -> PyanNet:
    """Load a diarizers HF segmentation snapshot into pyannote's native model."""

    snapshot_path = _segmentation_snapshot(model_id_or_path, token)
    config_path = snapshot_path / "config.json"
    weights_path = snapshot_path / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(
            f"Custom pyannote segmentation model at {snapshot_path} must contain "
            "config.json and model.safetensors."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_dict = {
        key.removeprefix("model."): value
        for key, value in load_file(weights_path).items()
        if key.startswith("model.")
    }
    if not state_dict:
        raise RuntimeError(f"Custom pyannote segmentation model at {snapshot_path} has no model weights.")

    model = PyanNet(
        sincnet={"stride": 10},
        lstm={"num_layers": _infer_lstm_layers(state_dict)},
        sample_rate=int(config.get("sample_rate") or 16000),
    )
    model.specifications = _build_segmentation_specifications(config)
    model.build()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Custom pyannote segmentation weights did not match pyannote PyanNet. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
    )
    return model.to(device).eval()


def _load_segmentation_override(model_id_or_path: str, token: str, device: torch.device):
    """Load either a Lightning checkpoint or an HF-style local snapshot override."""

    local_path = Path(model_id_or_path).expanduser()
    if local_path.is_file():
        with _trusted_torch_load_context():
            return _load_model_from_pretrained(str(local_path), token).to(device).eval()
    return _load_pyannote_segmentation_model(model_id_or_path, token, device)


class PyannoteCallhomeDiarizer:
    def __init__(
        self,
        device: Union[str, torch.device],
        hf_token: str | None = None,
        pipeline_model_id: str = DEFAULT_PYANNOTE_PIPELINE_MODEL,
        segmentation_model_id: str = DEFAULT_PYANNOTE_SEGMENTATION_MODEL,
    ):
        self.device = torch.device(device)
        pipeline_model_id = (pipeline_model_id or DEFAULT_PYANNOTE_PIPELINE_MODEL).strip()
        segmentation_model_id = (segmentation_model_id or "").strip()
        token = (
            hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )
        if not token:
            raise RuntimeError(
                "Pyannote backend requires a Hugging Face token. Set HF_TOKEN "
                "(or HUGGINGFACE_TOKEN/HUGGINGFACE_HUB_TOKEN) and ensure access "
                "to pyannote/speaker-diarization-3.1."
            )

        try:
            with _trusted_torch_load_context():
                self.pipeline = _load_pipeline(pipeline_model_id, token)
            if self.pipeline is None:
                raise RuntimeError(
                    f"Pyannote returned no pipeline for {pipeline_model_id}. "
                    "This usually means the Hugging Face token is missing access "
                    "or the pyannote model conditions have not been accepted. "
                    f"Accept access at https://huggingface.co/{pipeline_model_id} "
                    "using the same Hugging Face account that created the token."
                )
            self.pipeline.to(self.device)
            if segmentation_model_id:
                segmentation = getattr(self.pipeline, "_segmentation", None)
                if segmentation is None or not hasattr(segmentation, "model"):
                    raise RuntimeError(
                        "Selected pyannote pipeline does not expose a replaceable segmentation model."
                    )
                segmentation.model = _load_segmentation_override(
                    segmentation_model_id,
                    token,
                    self.device,
                )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize pyannote diarization backend. "
                "Confirm HF_TOKEN is set, your account has access to the selected "
                "pyannote pipeline, and the selected segmentation model is available. "
                f"Details: {exc}"
            ) from exc

    def diarize(self, audio: torch.Tensor, sample_rate: int = 16000):
        if audio.ndim == 1:
            waveform = audio.unsqueeze(0)
        elif audio.ndim == 2:
            waveform = audio
        else:
            raise ValueError(f"Expected 1D/2D audio tensor, got shape {tuple(audio.shape)}")

        waveform = waveform.detach().cpu().float()
        diarization = self.pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }
        )

        speaker_to_index: dict[str, int] = {}
        speaker_ts: list[tuple[int, int, int]] = []
        for segment, _, speaker_label in diarization.itertracks(yield_label=True):
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            if end_ms <= start_ms:
                continue
            if speaker_label not in speaker_to_index:
                speaker_to_index[speaker_label] = len(speaker_to_index)
            speaker_ts.append((start_ms, end_ms, speaker_to_index[speaker_label]))

        if not speaker_ts:
            total_ms = int((waveform.shape[-1] / sample_rate) * 1000)
            speaker_ts = [(0, max(total_ms, 1), 0)]

        return sorted(speaker_ts, key=lambda x: x[0])
