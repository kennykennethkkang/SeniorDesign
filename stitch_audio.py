#!/usr/bin/env python3
"""Create randomized stitched audio plus matching RTTM labels for training."""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from audio_numbering import AUDIO_EXTENSIONS
from fine_tuning_manager import (
    build_sample,
    normalize_backend,
    probe_media_duration,
    sanitize_filename,
    save_project_sample_streams,
    slugify,
)
from review_bundle import write_review_bundle
from workflow_background import utc_now_iso

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StitchInput:
    audio_path: Path
    audio_name: str
    speaker: str


@dataclass(frozen=True)
class TrainingTarget:
    backend: str
    project_name: str

    @property
    def key(self) -> str:
        return f"{self.backend}/{slugify(self.project_name)}"


@dataclass(frozen=True)
class StitchSegment:
    index: int
    audio_name: str
    audio_path: Path
    speaker: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class StitchResult:
    run_dir: Path
    wav_path: Path
    rttm_path: Path
    srt_path: Path
    manifest_path: Path
    metadata_path: Path
    review_path: Path | None
    review_flags_path: Path | None
    transcript_path: Path
    segments: tuple[StitchSegment, ...]
    training_usage: tuple[dict[str, object], ...]
    seed: str


def rttm_safe_token(raw_value: str, *, fallback: str = "") -> str:
    """Return one whitespace-free RTTM token."""

    token = "".join(
        char if char.isalnum() or char in "_-" else "_"
        for char in str(raw_value or "").strip()
    ).strip("_-")
    return token or fallback


def output_slug(raw_value: str) -> str:
    """Normalize the user-facing stitched output name."""

    slug = slugify(raw_value or "")
    if slug:
        return slug
    return "stitched-audio"


def unique_run_dir(stitched_root: Path, name: str) -> Path:
    """Build a timestamped run directory under ``stitched/`` without overwriting older runs."""

    stamp = time.strftime("%Y%m%dT%H%M%S")
    base = f"{stamp}_{output_slug(name)}"
    candidate = stitched_root / base
    counter = 2
    while candidate.exists():
        candidate = stitched_root / f"{base}_{counter}"
        counter += 1
    return candidate


def srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""

    total_ms = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_rttm(path: Path, *, session_id: str, segments: Sequence[StitchSegment]) -> None:
    lines = []
    for segment in segments:
        lines.append(
            " ".join(
                [
                    "SPEAKER",
                    session_id,
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
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(path: Path, segments: Sequence[StitchSegment]) -> None:
    blocks = []
    for segment in segments:
        blocks.append(
            "\n".join(
                [
                    str(segment.index),
                    f"{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}",
                    f"{segment.speaker}: {segment.audio_name}",
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_transcript(path: Path, segments: Sequence[StitchSegment]) -> None:
    lines = [
        f"{segment.speaker}: {segment.audio_name} [{segment.start:.3f}-{segment.end:.3f}]"
        for segment in segments
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, segments: Sequence[StitchSegment], *, root: Path) -> None:
    fieldnames = (
        "index",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "speaker",
        "audio_file",
        "audio_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for segment in segments:
            try:
                audio_path = segment.audio_path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                audio_path = str(segment.audio_path)
            writer.writerow(
                {
                    "index": segment.index,
                    "start_seconds": f"{segment.start:.3f}",
                    "end_seconds": f"{segment.end:.3f}",
                    "duration_seconds": f"{segment.duration:.3f}",
                    "speaker": segment.speaker,
                    "audio_file": segment.audio_name,
                    "audio_path": audio_path,
                }
            )


def wave_params(path: Path):
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as wav_file:
            params = wav_file.getparams()
    except (OSError, wave.Error):
        return None
    if params.comptype != "NONE":
        return None
    return (
        params.nchannels,
        params.sampwidth,
        params.framerate,
        params.comptype,
        params.compname,
    )


def can_stitch_with_wave(paths: Sequence[Path]) -> bool:
    """Return True when every WAV can be concatenated by copying PCM frames."""

    if not paths:
        return False
    first = wave_params(paths[0])
    if first is None:
        return False
    # Big stitch jobs benefit a lot from short-circuiting here — once any
    # input doesn't match, we can stop opening files. Saves a full sweep of
    # the 1.7k-file set in the common-case where someone mixes formats.
    for path in paths[1:]:
        params = wave_params(path)
        if params is None or params != first:
            return False
    return True


def stitch_with_wave(paths: Sequence[Path], output_path: Path) -> None:
    """Fast path for same-format PCM WAV inputs."""

    first_params = wave_params(paths[0])
    if first_params is None:
        raise ValueError("Cannot stitch non-PCM WAV files with the wave module.")
    channels, sample_width, frame_rate, comp_type, comp_name = first_params
    # 64 KiB per readframes call balances Python loop overhead against memory:
    # large enough to amortize per-call cost, small enough to keep the
    # working set tiny when stitching thousands of clips.
    chunk_frames = max(1, (64 * 1024) // max(sample_width * channels, 1))
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(frame_rate)
        output.setcomptype(comp_type, comp_name)
        for path in paths:
            with wave.open(str(path), "rb") as source:
                while True:
                    frames = source.readframes(chunk_frames)
                    if not frames:
                        break
                    output.writeframes(frames)


def ffmpeg_binary() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ffmpeg is required to stitch mixed-format audio. Install ffmpeg or use matching WAV files."
        ) from exc


def ffconcat_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def stitch_with_ffmpeg(paths: Sequence[Path], output_path: Path) -> None:
    """Normalize inputs to 16 kHz mono WAV chunks, then concatenate them."""

    ffmpeg = ffmpeg_binary()
    with tempfile.TemporaryDirectory(prefix="stitch_audio_") as tmpdir:
        temp_root = Path(tmpdir)
        normalized_paths = []
        for index, source in enumerate(paths, start=1):
            target = temp_root / f"{index:06d}.wav"
            command = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                str(target),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "ffmpeg conversion failed.").strip()
                raise RuntimeError(f"Could not prepare '{source.name}' for stitching: {detail}")
            normalized_paths.append(target)

        concat_list = temp_root / "concat.txt"
        concat_list.write_text(
            "".join(f"file '{ffconcat_escape(path)}'\n" for path in normalized_paths),
            encoding="utf-8",
        )
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "ffmpeg concat failed.").strip()
            raise RuntimeError(f"Could not write stitched WAV: {detail}")


def stitch_audio(paths: Sequence[Path], output_path: Path) -> str:
    """Write the stitched WAV and return the engine used."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if can_stitch_with_wave(paths):
        stitch_with_wave(paths, output_path)
        return "wave"
    stitch_with_ffmpeg(paths, output_path)
    return "ffmpeg"


def parse_training_target(raw_value: str) -> TrainingTarget:
    value = str(raw_value or "").strip()
    if "/" not in value:
        raise ValueError(f"Training target must be backend/project, got: {raw_value}")
    backend, project_name = value.split("/", 1)
    normalized_backend = normalize_backend(backend)
    project_slug = slugify(project_name)
    if not project_slug:
        raise ValueError(f"Training target project is empty: {raw_value}")
    return TrainingTarget(backend=normalized_backend, project_name=project_slug)


def add_to_training(
    *,
    wav_path: Path,
    rttm_path: Path,
    transcript_path: Path,
    targets: Sequence[TrainingTarget],
    root: Path,
) -> tuple[dict[str, object], ...]:
    usage = []
    for target in targets:
        with wav_path.open("rb") as audio_stream, rttm_path.open("rb") as rttm_stream:
            sample = save_project_sample_streams(
                project_name=target.project_name,
                backend=target.backend,
                audio_name=wav_path.name,
                audio_stream=audio_stream,
                rttm_name=rttm_path.name,
                rttm_stream=rttm_stream,
                transcript_text=transcript_path.read_text(encoding="utf-8") if transcript_path.is_file() else "",
                root=root,
            )
        usage.append(
            {
                "backend": target.backend,
                "project_name": target.project_name,
                "project_key": target.key,
                "sample_audio_path": str(sample.audio_path),
                "sample_rttm_path": str(sample.rttm_path),
                "sample_transcript_path": str(sample.transcript_path) if sample.transcript_path else "",
                "speaker_count": sample.num_speakers,
                "duration_seconds": sample.duration_seconds,
            }
        )
    return tuple(usage)


def metadata_base(
    *,
    root: Path,
    run_dir: Path,
    output_name: str,
    display_name: str,
    seed: str,
    inputs: Sequence[StitchInput],
    training_targets: Sequence[TrainingTarget],
) -> dict[str, object]:
    try:
        relative_run_dir = run_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative_run_dir = str(run_dir)
    return {
        "workflow": "audio_stitching",
        "runner": "local",
        "submission_status": "running",
        "started_at_utc": utc_now_iso(),
        "root": str(root),
        "run_dir": relative_run_dir,
        "output_name": output_name,
        "display_name": display_name,
        "seed": seed,
        "input_count": len(inputs),
        "training_targets": [target.key for target in training_targets],
    }


def read_existing_metadata(metadata_path: Path) -> dict[str, object]:
    if not metadata_path.is_file():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def serialize_segment(segment: StitchSegment, *, root: Path) -> dict[str, object]:
    try:
        source_path = segment.audio_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        source_path = str(segment.audio_path)
    return {
        "index": segment.index,
        "audio_file": segment.audio_name,
        "audio_path": source_path,
        "speaker": segment.speaker,
        "start": round(segment.start, 3),
        "duration": round(segment.duration, 3),
        "end": round(segment.end, 3),
    }


def run_stitch(
    *,
    inputs: Sequence[StitchInput],
    output_name: str = "",
    seed: str = "",
    root: Path = PROJECT_ROOT,
    run_dir: Path | None = None,
    training_targets: Sequence[TrainingTarget] = (),
    create_review: bool = True,
) -> StitchResult:
    """Randomize the inputs, stitch them, write timing artifacts, and optionally add the result to training."""

    root = root.expanduser().resolve()
    stitched_root = root / "stitched"
    seed_value = str(seed or secrets.token_hex(8))
    raw_output_name = str(output_name or "").strip()
    target_name = output_slug(raw_output_name)
    display_name = raw_output_name or target_name
    run_dir = run_dir.expanduser().resolve() if run_dir is not None else unique_run_dir(stitched_root, target_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    existing_metadata = read_existing_metadata(metadata_path)
    metadata = {
        **metadata_base(
            root=root,
            run_dir=run_dir,
            output_name=target_name,
            display_name=display_name,
            seed=seed_value,
            inputs=inputs,
            training_targets=training_targets,
        ),
        **existing_metadata,
        "submission_status": "running",
        "stitch_started_at_utc": utc_now_iso(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    exit_code_path = run_dir / "exit_code.txt"

    wav_path = run_dir / f"{target_name}.wav"
    rttm_path = run_dir / f"{target_name}.rttm"
    srt_path = run_dir / f"{target_name}.srt"
    manifest_path = run_dir / f"{target_name}_segments.tsv"
    transcript_path = run_dir / f"{target_name}.txt"
    review_path = run_dir / f"{target_name}_review.html"
    review_flags_path = run_dir / f"{target_name}_review_flags.tsv"

    try:
        if len(inputs) < 2:
            raise ValueError("Select at least two audio files to stitch.")
        for item in inputs:
            if not item.audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {item.audio_path}")
            if item.audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError(f"Unsupported audio type for {item.audio_path.name}.")
            if not rttm_safe_token(item.speaker):
                raise ValueError(f"Speaker label is empty for {item.audio_name}.")

        shuffled = list(inputs)
        random.Random(seed_value).shuffle(shuffled)
        offset = 0.0
        segments = []
        for index, item in enumerate(shuffled, start=1):
            duration = float(probe_media_duration(item.audio_path))
            if duration <= 0:
                raise ValueError(f"Could not determine duration for {item.audio_name}.")
            speaker = rttm_safe_token(item.speaker, fallback=f"Speaker_{index - 1}")
            segments.append(
                StitchSegment(
                    index=index,
                    audio_name=item.audio_name,
                    audio_path=item.audio_path,
                    speaker=speaker,
                    start=round(offset, 3),
                    duration=round(duration, 3),
                )
            )
            offset += duration
        segment_tuple = tuple(segments)

        stitch_engine = stitch_audio([segment.audio_path for segment in segment_tuple], wav_path)
        session_id = rttm_safe_token(Path(sanitize_filename(wav_path.name)).stem, fallback="stitched")
        write_rttm(rttm_path, session_id=session_id, segments=segment_tuple)
        write_srt(srt_path, segment_tuple)
        write_manifest(manifest_path, segment_tuple, root=root)
        write_transcript(transcript_path, segment_tuple)

        # Validate before adding to any training project so bad timings never enter the sample pool.
        sample = build_sample(wav_path, rttm_path, transcript_path)
        review_created = False
        if create_review:
            try:
                write_review_bundle(
                    srt_path=srt_path,
                    media_path=wav_path,
                    output_html=review_path,
                    report_tsv=review_flags_path,
                    audio_dir=run_dir,
                    quiet=True,
                )
                review_created = True
            except Exception as exc:  # noqa: BLE001
                metadata["review_error"] = str(exc)

        training_usage = add_to_training(
            wav_path=wav_path,
            rttm_path=rttm_path,
            transcript_path=transcript_path,
            targets=training_targets,
            root=root,
        )

        metadata.update(
            {
                "submission_status": "succeeded",
                "completed_at_utc": utc_now_iso(),
                "stitch_engine": stitch_engine,
                "audio_path": str(wav_path.resolve().relative_to(root.resolve())),
                "rttm_path": str(rttm_path.resolve().relative_to(root.resolve())),
                "srt_path": str(srt_path.resolve().relative_to(root.resolve())),
                "manifest_path": str(manifest_path.resolve().relative_to(root.resolve())),
                "transcript_path": str(transcript_path.resolve().relative_to(root.resolve())),
                "review_path": str(review_path.resolve().relative_to(root.resolve())) if review_created else "",
                "review_flags_path": str(review_flags_path.resolve().relative_to(root.resolve())) if review_created else "",
                "duration_seconds": sample.duration_seconds,
                "speaker_count": sample.num_speakers,
                "segment_count": len(segment_tuple),
                "segments": [serialize_segment(segment, root=root) for segment in segment_tuple],
                "training_usage": list(training_usage),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        exit_code_path.write_text("0\n", encoding="utf-8")
        return StitchResult(
            run_dir=run_dir,
            wav_path=wav_path,
            rttm_path=rttm_path,
            srt_path=srt_path,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            review_path=review_path if review_created else None,
            review_flags_path=review_flags_path if review_created else None,
            transcript_path=transcript_path,
            segments=segment_tuple,
            training_usage=training_usage,
            seed=seed_value,
        )
    except Exception as exc:
        metadata.update(
            {
                "submission_status": "failed",
                "failed_at_utc": utc_now_iso(),
                "error": str(exc),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        exit_code_path.write_text("1\n", encoding="utf-8")
        raise


def load_inputs_json(path: Path, *, root: Path) -> list[StitchInput]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError("Inputs JSON must contain an items list.")
    items: list[StitchInput] = []
    root_resolved = root.resolve()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        raw_path = str(raw.get("audio_path") or raw.get("path") or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        audio_path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        try:
            audio_path.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"Input is outside the project workspace: {raw_path}") from exc
        audio_name = str(raw.get("audio_name") or raw.get("name") or audio_path.name)
        speaker = str(raw.get("speaker") or "Speaker_0")
        items.append(StitchInput(audio_path=audio_path, audio_name=audio_name, speaker=speaker))
    return items


def parse_targets(values: Iterable[str]) -> tuple[TrainingTarget, ...]:
    targets = []
    seen = set()
    for value in values:
        if not str(value or "").strip():
            continue
        target = parse_training_target(value)
        if target.key in seen:
            continue
        targets.append(target)
        seen.add(target.key)
    return tuple(targets)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomize selected audio files and create a stitched WAV + RTTM pair.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--training-target", action="append", default=[])
    parser.add_argument("--skip-review", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    inputs = load_inputs_json(args.inputs_json, root=root)
    targets = parse_targets(args.training_target)
    result = run_stitch(
        inputs=inputs,
        output_name=args.output_name,
        seed=args.seed,
        root=root,
        run_dir=args.run_dir,
        training_targets=targets,
        create_review=not args.skip_review,
    )
    print(f"Stitched WAV: {result.wav_path}")
    print(f"RTTM: {result.rttm_path}")
    if result.review_path:
        print(f"Review: {result.review_path}")
    if result.training_usage:
        print("Added to training targets:")
        for usage in result.training_usage:
            print(f"  {usage['project_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
