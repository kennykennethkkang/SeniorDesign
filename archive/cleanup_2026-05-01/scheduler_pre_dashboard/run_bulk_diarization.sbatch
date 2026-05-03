#!/bin/bash -l
#
# Batch: diarize all audio files already present in audio_in/.
# YouTube conversion is separate: run `run_youtube_audio_download.sbatch` first if needed.
# Select backend with DIARIZATION_BACKEND=nemo|pyannote (default: nemo)
#
#SBATCH --job-name=bulk_diarization
#SBATCH --output=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/bulk_diarization_%j.out
#SBATCH --error=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/bulk_diarization_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

SELF_DIR="/WAVE/users2/unix/kkang/SeniorDesign"
PROJECT_ROOT="$SELF_DIR/all_diarization_programs"
WAV_DIR="$SELF_DIR/audio_in"
SBATCH_RUNTIME="$SELF_DIR/sbatch_runtime"
JOB_LOG_ROOT="$SELF_DIR/job_logs"
JOB_SLURM_LOGS="$JOB_LOG_ROOT/slurm"
JOB_OUTPUT_ROOT="$SELF_DIR/job_outputs/bulk_diarization"
JOB_PREFIX="bulk_diarization_job_${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="$JOB_OUTPUT_ROOT/$JOB_PREFIX"
VENV_PATH="$SBATCH_RUNTIME/.venv"
JOB_LOG_DIR="$JOB_LOG_ROOT/$JOB_PREFIX"
SLURM_OUT_FILE="$JOB_SLURM_LOGS/bulk_diarization_${SLURM_JOB_ID:-manual}.out"
SLURM_ERR_FILE="$JOB_SLURM_LOGS/bulk_diarization_${SLURM_JOB_ID:-manual}.err"
LOCATIONS_FILE="$JOB_LOG_DIR/locations.tsv"
WHISPER_MODEL="${WHISPER_MODEL:-small.en}"
WHISPER_BATCH_SIZE="${WHISPER_BATCH_SIZE:-16}"
DIARIZATION_BACKEND="${DIARIZATION_BACKEND:-nemo}"
PYANNOTE_PIPELINE_MODEL="${PYANNOTE_PIPELINE_MODEL:-pyannote/speaker-diarization-3.1}"
PYANNOTE_SEGMENTATION_MODEL="${PYANNOTE_SEGMENTATION_MODEL:-diarizers-community/speaker-segmentation-fine-tuned-callhome-zho}"
CACHE_BASE="${HOME}/.cache"
TRANSFORMERS_CACHE="${CACHE_BASE}/huggingface"
PIP_CACHE_DIR="${CACHE_BASE}/pip"
TMPDIR="$SBATCH_RUNTIME/tmp"
NUMBER_AUDIO_SCRIPT="$SELF_DIR/audio_numbering.py"
REVIEW_SCRIPT="$SELF_DIR/review_bundle.py"
REQ_CONSTRAINTS_FILE="$PROJECT_ROOT/constraints.txt"
REQ_COMMON_FILE="$PROJECT_ROOT/requirements_common.txt"
REQ_BACKEND_FILE="$PROJECT_ROOT/requirements_nemo.txt"
REQ_HASH_FILE="$SBATCH_RUNTIME/.requirements_nemo.sha256"
RUNTIME_ENV_SCRIPT="$SELF_DIR/scheduler/sbatch_runtime_env.sh"
DIARIZATION_CUDA_MODULE="${DIARIZATION_CUDA_MODULE:-}"
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
  VENV_PATH="$SBATCH_RUNTIME/.venv_pyannote"
  REQ_BACKEND_FILE="$PROJECT_ROOT/requirements_pyannote.txt"
  REQ_HASH_FILE="$SBATCH_RUNTIME/.requirements_pyannote.sha256"
fi

mkdir -p \
  "$WAV_DIR" \
  "$OUTPUT_DIR" \
  "$JOB_LOG_ROOT" \
  "$JOB_SLURM_LOGS" \
  "$JOB_LOG_DIR" \
  "$JOB_OUTPUT_ROOT" \
  "$TRANSFORMERS_CACHE" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR"

{
  echo -e "key\tvalue"
  echo -e "job_id\t${SLURM_JOB_ID:-manual}"
  echo -e "job_type\tbulk_diarization"
  echo -e "diarization_backend\t$DIARIZATION_BACKEND"
  echo -e "pyannote_pipeline_model\t$PYANNOTE_PIPELINE_MODEL"
  echo -e "pyannote_segmentation_model\t$PYANNOTE_SEGMENTATION_MODEL"
  echo -e "venv_path\t$VENV_PATH"
  echo -e "requirements_common\t$REQ_COMMON_FILE"
  echo -e "requirements_backend\t$REQ_BACKEND_FILE"
  echo -e "audio_input_dir\t$WAV_DIR"
  echo -e "output_dir\t$OUTPUT_DIR"
  echo -e "job_log_dir\t$JOB_LOG_DIR"
  echo -e "slurm_stdout\t$SLURM_OUT_FILE"
  echo -e "slurm_stderr\t$SLURM_ERR_FILE"
} > "$LOCATIONS_FILE"

echo "Output directory: $OUTPUT_DIR"
echo "Job log directory: $JOB_LOG_DIR"
echo "Diarization backend: $DIARIZATION_BACKEND"
echo "Slurm stdout/stderr: $SLURM_OUT_FILE | $SLURM_ERR_FILE"

if [ ! -d "$PROJECT_ROOT" ]; then
  echo "Project directory not found: $PROJECT_ROOT" >&2
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

python "$NUMBER_AUDIO_SCRIPT" --audio-dir "$WAV_DIR" \
  >"$JOB_LOG_DIR/audio_renumber_pre.out" \
  2>"$JOB_LOG_DIR/audio_renumber_pre.err"

mapfile -d '' -t wav_files < <(
  find "$WAV_DIR" -maxdepth 1 -type f \
    \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.m4a" -o -iname "*.flac" \
       -o -iname "*.ogg" -o -iname "*.opus" -o -iname "*.aac" -o -iname "*.wma" \
       -o -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.webm" \) \
    ! -iname "*_whisper_input.*" \
    -print0 | sort -z
)

if [ ${#wav_files[@]} -eq 0 ]; then
  echo "No audio files found in $WAV_DIR. Run run_youtube_audio_download.sbatch first or add files manually." >&2
  exit 1
fi

echo "Found ${#wav_files[@]} audio file(s) in $WAV_DIR"

SUMMARY_FILE="$JOB_LOG_DIR/runtime_summary.tsv"
echo -e "audio_file\truntime_seconds\tstdout_log\tstderr_log" > "$SUMMARY_FILE"

cd "$PROJECT_ROOT"
for wav in "${wav_files[@]}"; do
  wav_real="$(readlink -f "$wav")"
  case "$wav_real" in
    "$WAV_DIR"/*)
      ;;
    *)
      echo "Audio path is outside audio_in ($WAV_DIR): $wav_real" >&2
      exit 1
      ;;
  esac

  base="$(basename "$wav")"
  safe_base="$(echo "$base" | sed 's/[^A-Za-z0-9._-]/_/g')"
  run_out="$JOB_LOG_DIR/diarize_${safe_base}.out"
  run_err="$JOB_LOG_DIR/diarize_${safe_base}.err"

  echo "Diarizing $base ..."
  start_ts="$(date +%s)"
  python run_diarization.py \
    -a "$wav" \
    --input-dir "$WAV_DIR" \
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
  echo -e "${base}\t${runtime_sec}\t${run_out}\t${run_err}" >> "$SUMMARY_FILE"

  output_base="${base%.*}"
  if [ ! -s "$OUTPUT_DIR/${output_base}.txt" ] || [ ! -s "$OUTPUT_DIR/${output_base}.srt" ]; then
    echo "Missing diarization outputs for $base: expected $OUTPUT_DIR/${output_base}.txt and .srt" >&2
    exit 1
  fi

  python "$REVIEW_SCRIPT" \
    --srt "$OUTPUT_DIR/${output_base}.srt" \
    --media "$wav_real" \
    --output-html "$OUTPUT_DIR/${output_base}_review.html" \
    --report-tsv "$OUTPUT_DIR/${output_base}_review_flags.tsv" \
    --quiet
done

echo "All done. Outputs in $OUTPUT_DIR"
echo "Per-run logs in $JOB_LOG_DIR"
ls -lh "$OUTPUT_DIR"
