#!/bin/bash -l
#
# Single-file Whisper diarization job.
# Select backend with DIARIZATION_BACKEND=nemo|pyannote (default: nemo)
#
#SBATCH --job-name=single_diarization
#SBATCH --output=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/single_diarization_%j.out
#SBATCH --error=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/single_diarization_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

ROOT_DIR="/WAVE/users2/unix/kkang/SeniorDesign"
PROJECT_ROOT="$ROOT_DIR/all_diarization_programs"
SBATCH_RUNTIME="$ROOT_DIR/sbatch_runtime"
JOB_LOG_ROOT="$ROOT_DIR/job_logs"
JOB_SLURM_LOGS="$JOB_LOG_ROOT/slurm"
JOB_OUTPUT_ROOT="$ROOT_DIR/job_outputs/single_diarization"
JOB_PREFIX="single_diarization_job_${SLURM_JOB_ID:-manual}"
JOB_LOG_DIR="$JOB_LOG_ROOT/$JOB_PREFIX"
OUTPUT_DIR="$JOB_OUTPUT_ROOT/$JOB_PREFIX"
SLURM_OUT_FILE="$JOB_SLURM_LOGS/single_diarization_${SLURM_JOB_ID:-manual}.out"
SLURM_ERR_FILE="$JOB_SLURM_LOGS/single_diarization_${SLURM_JOB_ID:-manual}.err"
LOCATIONS_FILE="$JOB_LOG_DIR/locations.tsv"
INPUT_DIR="${INPUT_DIR:-$ROOT_DIR/audio_in}"
INPUT_AUDIO="${INPUT_AUDIO:-audiotest.wav}"
DIARIZATION_BACKEND="${DIARIZATION_BACKEND:-nemo}"
PYANNOTE_PIPELINE_MODEL="${PYANNOTE_PIPELINE_MODEL:-pyannote/speaker-diarization-3.1}"
PYANNOTE_SEGMENTATION_MODEL="${PYANNOTE_SEGMENTATION_MODEL:-diarizers-community/speaker-segmentation-fine-tuned-callhome-zho}"
NUMBER_AUDIO_SCRIPT="$ROOT_DIR/audio_numbering.py"
REVIEW_SCRIPT="$ROOT_DIR/review_bundle.py"
REQ_CONSTRAINTS_FILE="$PROJECT_ROOT/constraints.txt"
REQ_COMMON_FILE="$PROJECT_ROOT/requirements_common.txt"
REQ_BACKEND_FILE="$PROJECT_ROOT/requirements_nemo.txt"
VENV_PATH="$SBATCH_RUNTIME/.venv"
REQ_HASH_FILE="$SBATCH_RUNTIME/.requirements_nemo.sha256"
RUNTIME_ENV_SCRIPT="$ROOT_DIR/scheduler/sbatch_runtime_env.sh"
DIARIZATION_CUDA_MODULE="${DIARIZATION_CUDA_MODULE:-}"
CACHE_BASE="${HOME}/.cache"
TRANSFORMERS_CACHE="${CACHE_BASE}/huggingface"
PIP_CACHE_DIR="${CACHE_BASE}/pip"
TMPDIR="$SBATCH_RUNTIME/tmp"
export PIP_INDEX_URL="https://pypi.org/simple"
export PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"

case "$DIARIZATION_BACKEND" in
  nemo|pyannote)
    ;;
  msdd)
    DIARIZATION_BACKEND="nemo"
    ;;
  *)
    echo "Unsupported DIARIZATION_BACKEND: $DIARIZATION_BACKEND (use nemo or pyannote)" >&2
    exit 1
    ;;
esac

if [ "$DIARIZATION_BACKEND" = "pyannote" ]; then
  REQ_BACKEND_FILE="$PROJECT_ROOT/requirements_pyannote.txt"
  VENV_PATH="$SBATCH_RUNTIME/.venv_pyannote"
  REQ_HASH_FILE="$SBATCH_RUNTIME/.requirements_pyannote.sha256"
fi

cd "$PROJECT_ROOT" || exit 1
if [ ! -d "$INPUT_DIR" ]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi
if [ ! -f "$NUMBER_AUDIO_SCRIPT" ]; then
  echo "Audio numbering script not found: $NUMBER_AUDIO_SCRIPT" >&2
  exit 1
fi
if [ ! -f "$REVIEW_SCRIPT" ]; then
  echo "Review script not found: $REVIEW_SCRIPT" >&2
  exit 1
fi
if [ ! -f "$REQ_CONSTRAINTS_FILE" ]; then
  echo "Constraints file not found: $REQ_CONSTRAINTS_FILE" >&2
  exit 1
fi
if [ ! -f "$REQ_COMMON_FILE" ]; then
  echo "Common requirements file not found: $REQ_COMMON_FILE" >&2
  exit 1
fi
if [ ! -f "$REQ_BACKEND_FILE" ]; then
  echo "Backend requirements file not found: $REQ_BACKEND_FILE" >&2
  exit 1
fi
if [ ! -f "$RUNTIME_ENV_SCRIPT" ]; then
  echo "Runtime environment helper not found: $RUNTIME_ENV_SCRIPT" >&2
  exit 1
fi
INPUT_DIR="$(cd "$INPUT_DIR" && pwd -P)"

mkdir -p \
  "$JOB_SLURM_LOGS" \
  "$JOB_LOG_ROOT" \
  "$JOB_LOG_DIR" \
  "$JOB_OUTPUT_ROOT" \
  "$OUTPUT_DIR" \
  "$TRANSFORMERS_CACHE" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR"

{
  echo -e "key\tvalue"
  echo -e "job_id\t${SLURM_JOB_ID:-manual}"
  echo -e "job_type\tsingle_diarization"
  echo -e "input_dir\t$INPUT_DIR"
  echo -e "input_audio\t$INPUT_AUDIO"
  echo -e "diarization_backend\t$DIARIZATION_BACKEND"
  echo -e "pyannote_pipeline_model\t$PYANNOTE_PIPELINE_MODEL"
  echo -e "pyannote_segmentation_model\t$PYANNOTE_SEGMENTATION_MODEL"
  echo -e "venv_path\t$VENV_PATH"
  echo -e "requirements_common\t$REQ_COMMON_FILE"
  echo -e "requirements_backend\t$REQ_BACKEND_FILE"
  echo -e "output_dir\t$OUTPUT_DIR"
  echo -e "job_log_dir\t$JOB_LOG_DIR"
  echo -e "slurm_stdout\t$SLURM_OUT_FILE"
  echo -e "slurm_stderr\t$SLURM_ERR_FILE"
} > "$LOCATIONS_FILE"

echo "Output directory: $OUTPUT_DIR"
echo "Job log directory: $JOB_LOG_DIR"
echo "Diarization backend: $DIARIZATION_BACKEND"
echo "Slurm stdout/stderr: $SLURM_OUT_FILE | $SLURM_ERR_FILE"

module load Python/3.12.3-GCCcore-14.2.0
PYTHON_GCCCORE_LIB64="${EBROOTGCCCORE:+$EBROOTGCCCORE/lib64}"
if [ -n "$DIARIZATION_CUDA_MODULE" ]; then
  module load "$DIARIZATION_CUDA_MODULE"
fi
if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
export TRANSFORMERS_CACHE HF_HOME="$TRANSFORMERS_CACHE" XDG_CACHE_HOME="$CACHE_BASE" PIP_CACHE_DIR TMPDIR
source "$RUNTIME_ENV_SCRIPT"
append_ld_library_path "$PYTHON_GCCCORE_LIB64"
expose_venv_native_libraries
REQ_HASH="$(
  cat "$REQ_CONSTRAINTS_FILE" "$REQ_COMMON_FILE" "$REQ_BACKEND_FILE" \
    | sha256sum | awk '{print $1}'
)"
CURRENT_HASH="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

if [ "$REQ_HASH" != "$CURRENT_HASH" ]; then
  echo "Installing/updating Python dependencies..."
  python -m pip install --upgrade pip \
    >"$JOB_LOG_DIR/pip_bootstrap.out" \
    2>"$JOB_LOG_DIR/pip_bootstrap.err"
  python -m pip install \
    --extra-index-url "$PIP_EXTRA_INDEX_URL" \
    -c "$REQ_CONSTRAINTS_FILE" \
    -r "$REQ_COMMON_FILE" \
    -r "$REQ_BACKEND_FILE" \
    >"$JOB_LOG_DIR/pip_requirements.out" \
    2>"$JOB_LOG_DIR/pip_requirements.err"
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
else
  echo "Dependencies unchanged; skipping pip install."
fi

python "$NUMBER_AUDIO_SCRIPT" --audio-dir "$INPUT_DIR" \
  >"$JOB_LOG_DIR/audio_renumber_pre.out" \
  2>"$JOB_LOG_DIR/audio_renumber_pre.err"

if [[ "$INPUT_AUDIO" == *"/"* ]]; then
  if [ ! -f "$INPUT_AUDIO" ]; then
    echo "Input audio path does not exist: $INPUT_AUDIO" >&2
    exit 1
  fi
  resolved_input_audio="$(readlink -f "$INPUT_AUDIO")"
  case "$resolved_input_audio" in
    "$INPUT_DIR"/*)
      INPUT_AUDIO="$(basename "$resolved_input_audio")"
      ;;
    *)
      echo "Input audio must be selected from $INPUT_DIR, got: $INPUT_AUDIO" >&2
      exit 1
      ;;
  esac
fi

if [ ! -f "$INPUT_DIR/$INPUT_AUDIO" ]; then
  matches=()
  while IFS= read -r -d '' candidate; do
    candidate_base="$(basename "$candidate")"
    if [[ "$candidate_base" =~ ^[0-9]{3}_(.*)$ ]] && [ "${BASH_REMATCH[1]}" = "$INPUT_AUDIO" ]; then
      matches+=("$candidate_base")
    fi
  done < <(find "$INPUT_DIR" -maxdepth 1 -type f -print0)

  if [ "${#matches[@]}" -eq 1 ]; then
    INPUT_AUDIO="${matches[0]}"
    echo "Resolved unnumbered input name to numbered file: $INPUT_AUDIO"
  elif [ "${#matches[@]}" -gt 1 ]; then
    echo "Ambiguous input name '$INPUT_AUDIO'. Matching numbered files: ${matches[*]}" >&2
    exit 1
  fi
fi

if [ ! -f "$INPUT_DIR/$INPUT_AUDIO" ]; then
  echo "Input audio not found in $INPUT_DIR: $INPUT_AUDIO" >&2
  exit 1
fi

WHISPER_MODEL="${WHISPER_MODEL:-small.en}"
WHISPER_BATCH_SIZE="${WHISPER_BATCH_SIZE:-16}"

audio_base="$(basename "$INPUT_AUDIO")"
audio_base="${audio_base%.*}"
run_out="$JOB_LOG_DIR/diarize_${audio_base}.out"
run_err="$JOB_LOG_DIR/diarize_${audio_base}.err"
start_ts="$(date +%s)"

python run_diarization.py \
  -a "$INPUT_AUDIO" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --whisper-model "$WHISPER_MODEL" \
  --batch-size "$WHISPER_BATCH_SIZE" \
  --device "cuda" \
  --no-stem \
  --diarizer "$DIARIZATION_BACKEND" \
  --pyannote-pipeline-model "$PYANNOTE_PIPELINE_MODEL" \
  --pyannote-segmentation-model "$PYANNOTE_SEGMENTATION_MODEL" \
  >"$run_out" \
  2>"$run_err"

end_ts="$(date +%s)"
runtime_sec="$((end_ts - start_ts))"
echo -e "audio_file\truntime_seconds\tstdout_log\tstderr_log" > "$JOB_LOG_DIR/runtime_summary.tsv"
echo -e "${audio_base}\t${runtime_sec}\t${run_out}\t${run_err}" >> "$JOB_LOG_DIR/runtime_summary.tsv"

if [ ! -s "$OUTPUT_DIR/${audio_base}.txt" ] || [ ! -s "$OUTPUT_DIR/${audio_base}.srt" ]; then
  echo "Diarization finished without expected outputs: $OUTPUT_DIR/${audio_base}.txt and $OUTPUT_DIR/${audio_base}.srt" >&2
  exit 1
fi

python "$REVIEW_SCRIPT" \
  --srt "$OUTPUT_DIR/${audio_base}.srt" \
  --media "$INPUT_DIR/$INPUT_AUDIO" \
  --output-html "$OUTPUT_DIR/${audio_base}_review.html" \
  --report-tsv "$OUTPUT_DIR/${audio_base}_review_flags.tsv" \
  --quiet

echo "Review HTML: $OUTPUT_DIR/${audio_base}_review.html"
echo "Review flags: $OUTPUT_DIR/${audio_base}_review_flags.tsv"

echo "Done. Outputs in $OUTPUT_DIR"
echo "Per-run logs in $JOB_LOG_DIR"
ls -lh "$OUTPUT_DIR"
