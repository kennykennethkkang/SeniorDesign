#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

SENIOR_DESIGN_ROOT = Path(__file__).resolve().parents[1]
if str(SENIOR_DESIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(SENIOR_DESIGN_ROOT))

from workflow_preferences import (
    DEFAULT_PYANNOTE_PIPELINE_MODEL,
    DEFAULT_PYANNOTE_SEGMENTATION_MODEL,
    DIARIZATION_BACKENDS,
)

DEFAULT_INPUT_DIR = SENIOR_DESIGN_ROOT / "audio_in"
DEFAULT_OUTPUT_DIR = SENIOR_DESIGN_ROOT / "job_outputs" / "manual_run"
DIARIZER_CHOICES = tuple(dict.fromkeys([*DIARIZATION_BACKENDS, "msdd"]))

AUDIO_EXTENSIONS = {
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
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Whisper transcription plus speaker diarization on one audio file."
    )
    parser.add_argument(
        "-a",
        "--audio",
        help="Name or path of the target audio file. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing input audio files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write diarized outputs.",
    )
    parser.add_argument(
        "--list-audio",
        action="store_true",
        default=False,
        help="List audio files in --input-dir and exit without running diarization.",
    )

    stem_group = parser.add_mutually_exclusive_group()
    stem_group.add_argument(
        "--stem",
        action="store_true",
        dest="stemming",
        help="Enable source separation before transcription.",
    )
    stem_group.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        help="Skip source separation. This is the default on WAVE.",
    )
    parser.set_defaults(stemming=False)

    parser.add_argument(
        "--suppress-numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help=(
            "Suppress numerical digits. This can improve diarization alignment but "
            "spells digits out as words."
        ),
    )
    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="medium.en",
        help="Name of the Whisper model to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=8,
        help=(
            "Batch size for Whisper inference. Reduce if you run out of memory; "
            "set to 0 for non-batched longform inference."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language spoken in the audio. Omit to let Whisper detect it.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or a specific CUDA device like cuda:0.",
    )
    parser.add_argument(
        "--diarizer",
        default="nemo",
        choices=list(DIARIZER_CHOICES),
        help="Choose diarization backend.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for pyannote gated models (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--pyannote-pipeline-model",
        default=DEFAULT_PYANNOTE_PIPELINE_MODEL,
        help="Pyannote pipeline model ID.",
    )
    parser.add_argument(
        "--pyannote-segmentation-model",
        default=DEFAULT_PYANNOTE_SEGMENTATION_MODEL,
        help="Custom segmentation model. Leave blank for the pyannote pipeline default.",
    )
    parser.add_argument(
        "--nemo-msdd-model",
        default="",
        help="Optional NeMo MSDD checkpoint or .nemo path to use instead of the shared default model.",
    )
    parser.add_argument(
        "--save-whisper-input",
        action="store_true",
        default=False,
        help="Copy the Whisper input audio to output_dir as <name>_whisper_input.<ext>.",
    )

    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument(
        "--generate-review",
        action="store_true",
        dest="generate_review",
        help="Write an HTML review page and TSV flags next to the transcript outputs.",
    )
    review_group.add_argument(
        "--no-generate-review",
        action="store_false",
        dest="generate_review",
        help="Skip HTML/TSV review bundle generation.",
    )
    parser.set_defaults(generate_review=True)
    return parser


def list_audio_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    resolved_input_dir = input_dir.resolve()
    return sorted(
        (
            entry.resolve()
            for entry in resolved_input_dir.rglob("*")
            if entry.is_file()
            and entry.suffix.lower() in AUDIO_EXTENSIONS
            and "_whisper_input" not in entry.stem
        ),
        key=lambda path: str(path.relative_to(resolved_input_dir)).lower(),
    )


def prompt_for_audio(input_dir: Path) -> Path:
    files = list_audio_files(input_dir)
    if not files:
        raise SystemExit(
            f"No audio files found in '{input_dir}'. Put an audio file there or pass --audio."
        )

    print("Select an audio file to diarize:")
    for index, path in enumerate(files, start=1):
        print(f"  {index}) {path.relative_to(input_dir.resolve())}")

    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(files):
                return files[selected - 1]
        print("Invalid selection. Try again.")


def resolve_audio_path(audio_arg: str | None, input_dir: Path) -> Path:
    if audio_arg:
        audio_candidate = Path(audio_arg).expanduser()
        if audio_candidate.is_absolute():
            return audio_candidate.resolve()
        input_match = (input_dir / audio_candidate).resolve()
        if input_match.is_file():
            return input_match
        direct_match = Path.cwd() / audio_candidate
        if direct_match.is_file():
            return direct_match.resolve()
        return input_match

    if not sys.stdin.isatty():
        raise SystemExit("No --audio provided and no TTY available. Provide --audio.")
    return prompt_for_audio(input_dir)


def safe_output_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    safe = safe.strip("._")
    return safe or "audio"


def output_base_for_audio(audio_path: Path, input_dir: Path) -> str:
    try:
        relative = audio_path.resolve().relative_to(input_dir.resolve())
    except ValueError:
        return safe_output_component(audio_path.stem)
    stem_path = relative.with_suffix("")
    return "__".join(safe_output_component(part) for part in stem_path.parts)


def save_whisper_input(src_path: Path, output_dir: Path, output_base: str) -> Path | None:
    if not src_path.is_file():
        logging.warning("Whisper input audio not found, skipping copy: %s", src_path)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = output_dir / f"{output_base}_whisper_input{src_path.suffix or '.wav'}"
    if src_path.resolve() != dest_path.resolve():
        shutil.copy2(src_path, dest_path)
    return dest_path


def resolve_device_choice(requested_device: str) -> str:
    import torch

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def resolve_compute_type(device: str) -> str:
    return "float16" if device.startswith("cuda") else "int8"


def empty_cuda_cache(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def patch_ctc_aligner_transformers_dtype() -> None:
    """Keep the current CTC aligner usable with Transformers 4.x."""

    try:
        import transformers
        from ctc_forced_aligner import alignment_utils
    except Exception:
        return

    version_head = transformers.__version__.split(".", 1)[0]
    if not version_head.isdigit() or int(version_head) >= 5:
        return

    auto_model = alignment_utils.AutoModelForCTC
    original_from_pretrained = auto_model.from_pretrained
    if getattr(original_from_pretrained, "_senior_design_dtype_patch", False):
        return

    @classmethod
    def from_pretrained_compat(cls, *args, **kwargs):
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return original_from_pretrained(*args, **kwargs)

    from_pretrained_compat._senior_design_dtype_patch = True  # type: ignore[attr-defined]
    auto_model.from_pretrained = from_pretrained_compat


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 0:
        raise SystemExit("--batch-size must be 0 or greater.")


def load_runtime_dependencies():
    import faster_whisper
    import torch

    patch_ctc_aligner_transformers_dtype()

    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        load_alignment_model,
        postprocess_results,
        preprocess_text,
    )
    from deepmultilingualpunctuation import PunctuationModel

    from diarization_helpers import (
        find_numeral_symbol_tokens,
        get_realigned_ws_mapping_with_punctuation,
        get_sentences_speaker_mapping,
        get_speaker_aware_transcript,
        get_words_speaker_mapping,
        langs_to_iso,
        process_language_arg,
        punct_model_langs,
        write_srt,
    )

    return {
        "PunctuationModel": PunctuationModel,
        "faster_whisper": faster_whisper,
        "find_numeral_symbol_tokens": find_numeral_symbol_tokens,
        "generate_emissions": generate_emissions,
        "get_alignments": get_alignments,
        "get_realigned_ws_mapping_with_punctuation": get_realigned_ws_mapping_with_punctuation,
        "get_sentences_speaker_mapping": get_sentences_speaker_mapping,
        "get_speaker_aware_transcript": get_speaker_aware_transcript,
        "get_spans": get_spans,
        "get_words_speaker_mapping": get_words_speaker_mapping,
        "langs_to_iso": langs_to_iso,
        "load_alignment_model": load_alignment_model,
        "postprocess_results": postprocess_results,
        "preprocess_text": preprocess_text,
        "process_language_arg": process_language_arg,
        "punct_model_langs": punct_model_langs,
        "torch": torch,
        "write_srt": write_srt,
    }


def run_source_separation(
    audio_path: Path,
    device: str,
    working_dir: Path,
) -> Path:
    stem_output_dir = working_dir / "stem_outputs"
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs",
        "--two-stems=vocals",
        str(audio_path),
        "-o",
        str(stem_output_dir),
        "--device",
        device,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        logging.warning(
            "Source separation failed (exit code %s); using original audio.",
            result.returncode,
        )
        return audio_path

    vocal_target = (
        stem_output_dir / "htdemucs" / audio_path.stem / "vocals.wav"
    )
    if vocal_target.is_file():
        return vocal_target

    logging.warning(
        "Source separation completed but expected vocals file was not found: %s",
        vocal_target,
    )
    return audio_path


def maybe_generate_review_bundle(
    *,
    audio_path: Path,
    output_srt: Path,
    output_dir: Path,
    output_base: str,
) -> tuple[Path, Path] | None:
    from review_bundle import write_review_bundle

    review_html = output_dir / f"{output_base}_review.html"
    review_flags = output_dir / f"{output_base}_review_flags.tsv"
    write_review_bundle(
        srt_path=output_srt,
        media_path=audio_path,
        output_html=review_html,
        report_tsv=review_flags,
        quiet=True,
    )
    return review_html, review_flags


def run_diarization(args: argparse.Namespace) -> int:
    deps = load_runtime_dependencies()
    torch = deps["torch"]
    faster_whisper = deps["faster_whisper"]

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = resolve_audio_path(args.audio, input_dir)
    if not audio_path.is_file():
        raise SystemExit(f"Audio file not found: {audio_path}")

    output_base = output_base_for_audio(audio_path, input_dir)
    output_txt = output_dir / f"{output_base}.txt"
    output_srt = output_dir / f"{output_base}.srt"
    device = resolve_device_choice(args.device)
    language = deps["process_language_arg"](args.language, args.model_name)

    with tempfile.TemporaryDirectory(prefix="diarize_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        whisper_input = (
            run_source_separation(audio_path, device, temp_dir)
            if args.stemming
            else audio_path
        )

        if args.save_whisper_input:
            save_whisper_input(whisper_input, output_dir, output_base)

        whisper_model = faster_whisper.WhisperModel(
            args.model_name,
            device=device,
            compute_type=resolve_compute_type(device),
        )
        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
        audio_waveform = faster_whisper.decode_audio(str(whisper_input))
        suppress_tokens = (
            deps["find_numeral_symbol_tokens"](whisper_model.hf_tokenizer)
            if args.suppress_numerals
            else [-1]
        )

        if args.batch_size > 0:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                batch_size=args.batch_size,
            )
        else:
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )

        full_transcript = "".join(segment.text for segment in transcript_segments).strip()
        if not full_transcript:
            raise RuntimeError("Whisper returned an empty transcript.")

        del whisper_model, whisper_pipeline
        empty_cuda_cache(torch, device)

        alignment_model, alignment_tokenizer = deps["load_alignment_model"](
            device,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        )

        emissions, stride = deps["generate_emissions"](
            alignment_model,
            torch.from_numpy(audio_waveform)
            .to(alignment_model.dtype)
            .to(alignment_model.device),
            batch_size=args.batch_size,
        )

        del alignment_model
        empty_cuda_cache(torch, device)

        alignment_language = deps["langs_to_iso"].get(info.language)
        if not alignment_language:
            raise RuntimeError(
                f"Unsupported language for alignment output: {info.language!r}"
            )

        tokens_starred, text_starred = deps["preprocess_text"](
            full_transcript,
            romanize=True,
            language=alignment_language,
        )

        segments, scores, blank_token = deps["get_alignments"](
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )

        spans = deps["get_spans"](tokens_starred, segments, blank_token)
        word_timestamps = deps["postprocess_results"](
            text_starred,
            spans,
            stride,
            scores,
        )

        diarizer_choice = "nemo" if args.diarizer == "msdd" else args.diarizer
        if diarizer_choice == "nemo":
            from all_diarization_programs import NemoMsddDiarizer

            diarizer_model = NemoMsddDiarizer(
                device=device,
                msdd_model_path=args.nemo_msdd_model,
            )
        elif diarizer_choice == "pyannote":
            from all_diarization_programs import PyannoteCallhomeDiarizer

            diarizer_model = PyannoteCallhomeDiarizer(
                device=device,
                hf_token=args.hf_token,
                pipeline_model_id=args.pyannote_pipeline_model,
                segmentation_model_id=args.pyannote_segmentation_model,
            )
        else:
            raise SystemExit(f"Unsupported diarizer backend: {args.diarizer}")

        speaker_ts = diarizer_model.diarize(
            torch.from_numpy(audio_waveform).unsqueeze(0),
            sample_rate=16000,
        )
        del diarizer_model
        empty_cuda_cache(torch, device)

        word_speaker_mapping = deps["get_words_speaker_mapping"](
            word_timestamps,
            speaker_ts,
            "start",
        )

        if info.language in deps["punct_model_langs"]:
            punct_model = deps["PunctuationModel"](model="kredor/punctuate-all")
            words_list = [entry["word"] for entry in word_speaker_mapping]
            labelled_words = punct_model.predict(words_list, chunk_size=230)
            ending_puncts = ".?!"
            model_puncts = ".,;:!?"

            for word_dict, labeled_tuple in zip(word_speaker_mapping, labelled_words):
                word = word_dict["word"]
                is_acronym = bool(word) and word.endswith(".") and word.count(".") >= 2
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym)
                ):
                    punctuated = f"{word}{labeled_tuple[1]}"
                    if punctuated.endswith(".."):
                        punctuated = punctuated.rstrip(".")
                    word_dict["word"] = punctuated
        else:
            logging.warning(
                "Punctuation restoration is not available for %s. Using original punctuation.",
                info.language,
            )

        realigned_mapping = deps["get_realigned_ws_mapping_with_punctuation"](
            word_speaker_mapping
        )
        sentence_mapping = deps["get_sentences_speaker_mapping"](
            realigned_mapping,
            speaker_ts,
        )

        with output_txt.open("w", encoding="utf-8-sig") as transcript_fh:
            deps["get_speaker_aware_transcript"](sentence_mapping, transcript_fh)

        with output_srt.open("w", encoding="utf-8-sig") as srt_fh:
            deps["write_srt"](sentence_mapping, srt_fh)

    print(f"Transcript: {output_txt}")
    print(f"Subtitles:  {output_srt}")
    if args.generate_review:
        try:
            review_paths = maybe_generate_review_bundle(
                audio_path=audio_path,
                output_srt=output_srt,
                output_dir=output_dir,
                output_base=output_base,
            )
        except Exception as exc:  # pragma: no cover - review bundle is non-critical
            logging.warning("Review bundle generation failed: %s", exc)
        else:
            if review_paths is not None:
                review_html, review_flags = review_paths
                print(f"Review:     {review_html}")
                print(f"Flags:      {review_flags}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    input_dir = Path(args.input_dir).expanduser().resolve()
    if args.list_audio:
        files = list_audio_files(input_dir)
        if not files:
            print(f"No audio files found in {input_dir}", file=sys.stderr)
            return 1
        for path in files:
            print(path.name)
        return 0

    try:
        return run_diarization(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
