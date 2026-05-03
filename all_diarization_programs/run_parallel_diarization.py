import argparse
import logging
import multiprocessing as mp
import os
import re
import shutil
import sys

import faster_whisper
import torch

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
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

SENIOR_DESIGN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SENIOR_DESIGN_ROOT not in sys.path:
    sys.path.insert(0, SENIOR_DESIGN_ROOT)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "audio_in"))
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "job_outputs", "manual_parallel_run")
)

from all_diarization_programs import NemoMsddDiarizer


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


def list_audio_files(input_dir):
    if not os.path.isdir(input_dir):
        return []
    return [
        entry
        for entry in sorted(os.listdir(input_dir))
        if os.path.isfile(os.path.join(input_dir, entry))
        and os.path.splitext(entry)[1].lower() in AUDIO_EXTENSIONS
    ]


def prompt_for_audio(input_dir):
    files = list_audio_files(input_dir)
    if not files:
        raise SystemExit(
            f"No audio files found in '{input_dir}'. Put an audio file there or pass --audio with a path."
        )

    print("Select an audio file to diarize:")
    for idx, name in enumerate(files, start=1):
        print(f"  {idx}) {name}")

    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(files):
                return os.path.join(input_dir, files[index - 1])
        print("Invalid selection. Try again.")


def resolve_audio_path(audio_arg, input_dir):
    if audio_arg:
        if os.path.isabs(audio_arg) or os.path.sep in audio_arg:
            return audio_arg
        if os.path.altsep and os.path.altsep in audio_arg:
            return audio_arg
        if os.path.isfile(audio_arg):
            return audio_arg
        return os.path.join(input_dir, audio_arg)

    if not sys.stdin.isatty():
        raise SystemExit("No --audio provided and no TTY available. Provide --audio.")

    return prompt_for_audio(input_dir)


def save_whisper_input(src_path, output_dir, output_base):
    if not os.path.isfile(src_path):
        logging.warning("Whisper input audio not found, skipping copy: %s", src_path)
        return None
    ext = os.path.splitext(src_path)[1] or ".wav"
    dest_path = os.path.join(output_dir, f"{output_base}_whisper_input{ext}")
    if os.path.abspath(src_path) != os.path.abspath(dest_path):
        shutil.copy2(src_path, dest_path)
    return dest_path


def diarize_parallel(audio: torch.Tensor, device, queue: mp.Queue):
    model = NemoMsddDiarizer(device=device)
    result = model.diarize(audio, sample_rate=16000)
    queue.put(result)


mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    mtypes = {"cpu": "int8", "cuda": "float16"}
    temp_outputs_dir = "temp_outputs"
    temp_path = os.path.join(os.getcwd(), temp_outputs_dir)
    if os.path.isdir(temp_path):
        cleanup(temp_path)
    os.makedirs(temp_path, exist_ok=True)

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--audio",
        help="name or path of the target audio file. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="directory containing input audio files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="directory to write diarized outputs",
    )
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation."
        "This helps with long files that don't contain a lot of music.",
    )

    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help="Suppresses Numerical Digits."
        "This helps the diarization accuracy but converts all digits into written text.",
    )

    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="large-v2",
        help="name of the Whisper model to use",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=4,
        help="Batch size for batched inference, reduce if you run out of memory, "
        "set to 0 for original whisper longform inference",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=whisper_langs,
        help="Language spoken in the audio, specify None to perform language detection",
    )

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )

    parser.add_argument(
        "--diarizer",
        default="msdd",
        choices=["msdd"],
        help="Choose the diarization model to use",
    )
    parser.add_argument(
        "--save-whisper-input",
        action="store_true",
        default=False,
        help="Copy the Whisper input audio to output_dir as <name>_whisper_input.<ext>",
    )

    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    audio_path = resolve_audio_path(args.audio, input_dir)
    if not os.path.isfile(audio_path):
        raise SystemExit(f"Audio file not found: {audio_path}")

    output_base = os.path.splitext(os.path.basename(audio_path))[0]
    output_txt = os.path.join(output_dir, f"{output_base}.txt")
    output_srt = os.path.join(output_dir, f"{output_base}.srt")

    language = process_language_arg(args.language, args.model_name)

    if args.stemming:
        # Isolate vocals from the rest of the audio

        return_code = os.system(
            f'"{sys.executable}" -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "{temp_outputs_dir}" --device "{args.device}"'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use --no-stem argument to disable it."
            )
            vocal_target = audio_path
        else:
            vocal_target = os.path.join(
                temp_outputs_dir,
                "htdemucs",
                os.path.splitext(os.path.basename(audio_path))[0],
                "vocals.wav",
            )
    else:
        vocal_target = audio_path

    if args.save_whisper_input:
        save_whisper_input(vocal_target, output_dir, output_base)

    audio_waveform = faster_whisper.decode_audio(vocal_target)

    logging.info("Starting Nemo process with vocal_target: ", vocal_target)
    results_queue = mp.Queue()
    nemo_process = mp.Process(
        target=diarize_parallel,
        args=(
            torch.from_numpy(audio_waveform).unsqueeze(0),
            args.device,
            results_queue,
        ),
    )
    nemo_process.start()
    # Transcribe the audio file

    whisper_model = faster_whisper.WhisperModel(
        args.model_name, device=args.device, compute_type=mtypes[args.device]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
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

    full_transcript = "".join(segment.text for segment in transcript_segments)

    # clear gpu vram
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # Forced Alignment
    alignment_model, alignment_tokenizer = load_alignment_model(
        args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device),
        batch_size=args.batch_size,
    )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    nemo_process.join()
    if results_queue.empty():
        raise RuntimeError("Diarization process did not return any results.")

    speaker_ts = results_queue.get_nowait()

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
            " Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    with open(output_txt, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(output_srt, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)
