#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_single_diarization.sbatch"
INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/audio_in}"
NUMBER_AUDIO_SCRIPT="$PROJECT_ROOT/audio_numbering.py"

selected_audio=""
selected_backend=""

usage() {
  cat <<'USAGE'
Usage:
  ./submit_single_diarization_job.sh
  ./submit_single_diarization_job.sh --file <audio_file_or_path_in_audio_in>
  ./submit_single_diarization_job.sh --backend <nemo|pyannote>
  ./submit_single_diarization_job.sh --file <audio> --backend <nemo|pyannote>

Description:
  Prompts for diarization backend and one audio file (1,2,3,...) from audio_in,
  then submits the single diarization sbatch job.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --file)
      [[ -n "${2:-}" ]] || { echo "Missing value for --file" >&2; exit 1; }
      selected_audio="$2"
      shift 2
      ;;
    --backend)
      [[ -n "${2:-}" ]] || { echo "Missing value for --backend" >&2; exit 1; }
      selected_backend="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$NUMBER_AUDIO_SCRIPT" ]]; then
  echo "Missing audio numbering script: $NUMBER_AUDIO_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

INPUT_DIR="$(cd "$INPUT_DIR" && pwd -P)"
python3 "$NUMBER_AUDIO_SCRIPT" --audio-dir "$INPUT_DIR"

if [[ -z "$selected_backend" ]]; then
  echo "Select diarization backend:"
  echo "  1) nemo"
  echo "  2) pyannote"
  while true; do
    read -r -p "Enter selection [1-2]: " backend_choice
    case "$backend_choice" in
      1)
        selected_backend="nemo"
        break
        ;;
      2)
        selected_backend="pyannote"
        break
        ;;
      *)
        echo "Invalid selection. Try again."
        ;;
    esac
  done
fi

case "$selected_backend" in
  nemo|pyannote)
    ;;
  msdd)
    selected_backend="nemo"
    ;;
  *)
    echo "Unsupported backend '$selected_backend'. Use nemo or pyannote." >&2
    exit 1
    ;;
esac

if [[ -z "$selected_audio" ]]; then
  files=()
  while IFS= read -r -d '' path; do
    files+=("$path")
  done < <(
    find "$INPUT_DIR" -maxdepth 1 -type f \
      \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.m4a" -o -iname "*.flac" \
         -o -iname "*.ogg" -o -iname "*.opus" -o -iname "*.aac" -o -iname "*.wma" \
         -o -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.webm" \) \
      ! -iname "*_whisper_input.*" \
      -print0 | sort -z
  )

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No audio files found in: $INPUT_DIR" >&2
    exit 1
  fi

  echo "Select one audio file to process:"
  for i in "${!files[@]}"; do
    printf "  %d) %s\n" "$((i + 1))" "$(basename "${files[$i]}")"
  done

  while true; do
    read -r -p "Enter selection [1-${#files[@]}]: " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#files[@]} )); then
      selected_audio="$(basename "${files[$((choice - 1))]}")"
      break
    fi
    echo "Invalid selection. Try again."
  done
fi

if [[ -f "$selected_audio" ]]; then
  resolved_selected="$(readlink -f "$selected_audio")"
  case "$resolved_selected" in
    "$INPUT_DIR"/*)
      selected_audio="$(basename "$resolved_selected")"
      ;;
    *)
      echo "Selected audio must be inside $INPUT_DIR, got: $selected_audio" >&2
      exit 1
      ;;
  esac
fi

if [[ ! -f "$INPUT_DIR/$selected_audio" ]]; then
  matches=()
  while IFS= read -r -d '' candidate; do
    candidate_base="$(basename "$candidate")"
    if [[ "$candidate_base" =~ ^[0-9]{3}_(.*)$ ]] && [[ "${BASH_REMATCH[1]}" == "$selected_audio" ]]; then
      matches+=("$candidate_base")
    fi
  done < <(find "$INPUT_DIR" -maxdepth 1 -type f -print0)

  if [[ ${#matches[@]} -eq 1 ]]; then
    selected_audio="${matches[0]}"
    echo "Resolved unnumbered file name to numbered file: $selected_audio"
  elif [[ ${#matches[@]} -gt 1 ]]; then
    echo "Ambiguous selection '$selected_audio'. Matching numbered files: ${matches[*]}" >&2
    exit 1
  fi
fi

if [[ ! -f "$INPUT_DIR/$selected_audio" ]]; then
  echo "Selected audio not found in $INPUT_DIR: $selected_audio" >&2
  exit 1
fi

echo "Submitting single diarization for: $selected_audio"
echo "Backend: $selected_backend"
submit_output="$(
  sbatch --export="ALL,INPUT_DIR=$INPUT_DIR,INPUT_AUDIO=$selected_audio,DIARIZATION_BACKEND=$selected_backend" "$SBATCH_SCRIPT"
)"
echo "$submit_output"

job_id="$(echo "$submit_output" | awk '{print $NF}')"
if [[ "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Check paths after start: $PROJECT_ROOT/job_logs/single_diarization_job_${job_id}/locations.tsv"
fi
