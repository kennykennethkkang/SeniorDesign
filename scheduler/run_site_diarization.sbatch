#!/bin/bash -l
#
# Site-launched diarization job.
# The dashboard writes the selected audio list and output folder first, then
# submits this script with environment variables so every run stays isolated.
#
#SBATCH --job-name=site_diarization
#SBATCH --output=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/site_diarization_%j.out
#SBATCH --error=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/site_diarization_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

ROOT_DIR="${SITE_ROOT_DIR:-/WAVE/users2/unix/kkang/SeniorDesign}"
PROJECT_ROOT="$ROOT_DIR/all_diarization_programs"
RUN_DIR="${SITE_DIARIZATION_RUN_DIR:?SITE_DIARIZATION_RUN_DIR is required}"
AUDIO_LIST_FILE="${SITE_DIARIZATION_AUDIO_LIST:?SITE_DIARIZATION_AUDIO_LIST is required}"
LOG_DIR="${SITE_DIARIZATION_LOG_DIR:-$RUN_DIR/logs}"
AUDIO_DIR="${SITE_AUDIO_DIR:-$ROOT_DIR/audio_in}"
SBATCH_RUNTIME="$ROOT_DIR/sbatch_runtime"
JOB_LOG_ROOT="$ROOT_DIR/job_logs"
JOB_SLURM_LOGS="$JOB_LOG_ROOT/slurm"
LOCATIONS_FILE="$LOG_DIR/locations.tsv"
NUMBER_AUDIO_SCRIPT="$ROOT_DIR/audio_numbering.py"
CLI_SCRIPT="$ROOT_DIR/workflow_cli.py"
REQ_CONSTRAINTS_FILE="$PROJECT_ROOT/constraints.txt"
REQ_COMMON_FILE="$PROJECT_ROOT/requirements_common.txt"
DIARIZATION_BACKEND="${DIARIZATION_BACKEND:-nemo}"
WHISPER_MODEL="${WHISPER_MODEL:-small.en}"
WHISPER_BATCH_SIZE="${WHISPER_BATCH_SIZE:-16}"
DIARIZATION_DEVICE="${DIARIZATION_DEVICE:-cuda}"
PYANNOTE_PIPELINE_MODEL="${PYANNOTE_PIPELINE_MODEL:-pyannote/speaker-diarization-3.1}"
PYANNOTE_SEGMENTATION_MODEL="${PYANNOTE_SEGMENTATION_MODEL:-diarizers-community/speaker-segmentation-fine-tuned-callhome-zho}"
NEMO_MSDD_MODEL="${NEMO_MSDD_MODEL:-}"
DIARIZATION_LANGUAGE="${DIARIZATION_LANGUAGE:-}"
DIARIZATION_SOURCE_SEPARATION="${DIARIZATION_SOURCE_SEPARATION:-0}"
DIARIZATION_SKIP_REVIEW="${DIARIZATION_SKIP_REVIEW:-0}"
CACHE_BASE="${HOME}/.cache"
TRANSFORMERS_CACHE="${CACHE_BASE}/huggingface"
PIP_CACHE_DIR="${CACHE_BASE}/pip"
TMPDIR="$SBATCH_RUNTIME/tmp"
VENV_PATH="$SBATCH_RUNTIME/.venv"
REQ_BACKEND_FILE="$PROJECT_ROOT/requirements_nemo.txt"
REQ_HASH_FILE="$SBATCH_RUNTIME/.requirements_nemo.sha256"
RUNTIME_ENV_SCRIPT="$ROOT_DIR/scheduler/sbatch_runtime_env.sh"
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
  "$RUN_DIR" \
  "$LOG_DIR" \
  "$AUDIO_DIR" \
  "$JOB_LOG_ROOT" \
  "$JOB_SLURM_LOGS" \
  "$TRANSFORMERS_CACHE" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR"

finish() {
  rc=$?
  trap - EXIT
  printf "%s\n" "$rc" > "$RUN_DIR/exit_code.txt"
  exit "$rc"
}
trap finish EXIT

exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)

{
  echo -e "key\tvalue"
  echo -e "job_id\t${SLURM_JOB_ID:-manual}"
  echo -e "job_type\tsite_diarization"
  echo -e "run_dir\t$RUN_DIR"
  echo -e "audio_list_file\t$AUDIO_LIST_FILE"
  echo -e "audio_input_dir\t$AUDIO_DIR"
  echo -e "log_dir\t$LOG_DIR"
  echo -e "diarization_backend\t$DIARIZATION_BACKEND"
  echo -e "whisper_model\t$WHISPER_MODEL"
  echo -e "whisper_batch_size\t$WHISPER_BATCH_SIZE"
  echo -e "device\t$DIARIZATION_DEVICE"
  echo -e "pyannote_pipeline_model\t$PYANNOTE_PIPELINE_MODEL"
  echo -e "pyannote_segmentation_model\t$PYANNOTE_SEGMENTATION_MODEL"
  echo -e "nemo_msdd_model\t$NEMO_MSDD_MODEL"
  echo -e "venv_path\t$VENV_PATH"
  echo -e "requirements_common\t$REQ_COMMON_FILE"
  echo -e "requirements_backend\t$REQ_BACKEND_FILE"
} > "$LOCATIONS_FILE"

echo "Site diarization Slurm job: ${SLURM_JOB_ID:-manual}"
echo "Run directory: $RUN_DIR"
echo "Selected audio list: $AUDIO_LIST_FILE"
echo "Log directory: $LOG_DIR"
echo "Backend: $DIARIZATION_BACKEND"

for required_path in \
  "$PROJECT_ROOT" \
  "$AUDIO_LIST_FILE" \
  "$NUMBER_AUDIO_SCRIPT" \
  "$CLI_SCRIPT" \
  "$REQ_CONSTRAINTS_FILE" \
  "$REQ_COMMON_FILE" \
  "$REQ_BACKEND_FILE" \
  "$RUNTIME_ENV_SCRIPT"; do
  if [ ! -e "$required_path" ]; then
    echo "Required path not found: $required_path" >&2
    exit 1
  fi
done

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

nemo_runtime_is_healthy() {
  python - <<'PY' >/dev/null 2>&1
from numba import cuda  # Triggers Numba's version guard for NumPy compatibility.
PY
}

REQ_HASH="$(
  cat "$REQ_CONSTRAINTS_FILE" "$REQ_COMMON_FILE" "$REQ_BACKEND_FILE" \
    | sha256sum | awk '{print $1}'
)"
CURRENT_HASH="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"
REPAIR_RUNTIME=0

if [ "$DIARIZATION_BACKEND" = "nemo" ] && [ "$REQ_HASH" = "$CURRENT_HASH" ]; then
  if ! nemo_runtime_is_healthy; then
    echo "NeMo runtime compatibility check failed; reinstalling constrained dependencies."
    REPAIR_RUNTIME=1
  fi
fi

if [ "$REQ_HASH" != "$CURRENT_HASH" ] || [ "$REPAIR_RUNTIME" = "1" ]; then
  echo "Installing or updating diarization dependencies..."
  python -m pip install --upgrade pip \
    >"$LOG_DIR/pip_bootstrap.out" \
    2>"$LOG_DIR/pip_bootstrap.err"
  python -m pip install \
    --extra-index-url "$PIP_EXTRA_INDEX_URL" \
    -c "$REQ_CONSTRAINTS_FILE" \
    -r "$REQ_COMMON_FILE" \
    -r "$REQ_BACKEND_FILE" \
    >"$LOG_DIR/pip_requirements.out" \
    2>"$LOG_DIR/pip_requirements.err"
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
else
  echo "Diarization dependencies are unchanged; skipping pip install."
fi

python "$NUMBER_AUDIO_SCRIPT" --audio-dir "$AUDIO_DIR" \
  >"$LOG_DIR/audio_renumber_pre.out" \
  2>"$LOG_DIR/audio_renumber_pre.err"

command=(
  python "$CLI_SCRIPT"
  run-diarization-selection
  --audio-list-file "$AUDIO_LIST_FILE"
  --backend "$DIARIZATION_BACKEND"
  --output-dir "$RUN_DIR"
  --log-dir "$LOG_DIR"
  --device "$DIARIZATION_DEVICE"
  --whisper-model "$WHISPER_MODEL"
  --batch-size "$WHISPER_BATCH_SIZE"
  --pyannote-pipeline-model "$PYANNOTE_PIPELINE_MODEL"
  --pyannote-segmentation-model "$PYANNOTE_SEGMENTATION_MODEL"
)

if [ -n "$NEMO_MSDD_MODEL" ]; then
  command+=(--nemo-msdd-model "$NEMO_MSDD_MODEL")
fi

if [ -n "$DIARIZATION_LANGUAGE" ]; then
  command+=(--language "$DIARIZATION_LANGUAGE")
fi
if [ "$DIARIZATION_SOURCE_SEPARATION" = "1" ]; then
  command+=(--source-separation)
fi
if [ "$DIARIZATION_SKIP_REVIEW" = "1" ]; then
  command+=(--skip-review)
fi

echo "Running selected diarization command:"
printf "  %q" "${command[@]}"
printf "\n"
"${command[@]}"

echo "Site diarization complete."
echo "Outputs are in $RUN_DIR"
