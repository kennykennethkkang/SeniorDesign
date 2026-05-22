#!/bin/bash -l
#
# End-to-end test: fine-tuned NeMo MSDD v11 (.nemo) → diarization → SRT output
#
#SBATCH --job-name=finetune_v11_test
#SBATCH --output=/WAVE/projects2/sd-mlspeechdiarization-2025/SeniorDesign/test_finetune_v11/slurm_%j.out
#SBATCH --error=/WAVE/projects2/sd-mlspeechdiarization-2025/SeniorDesign/test_finetune_v11/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

ROOT_DIR="/WAVE/projects2/sd-mlspeechdiarization-2025/SeniorDesign"
PROJECT_ROOT="$ROOT_DIR/all_diarization_programs"
TEST_DIR="$ROOT_DIR/test_finetune_v11"
RUN_DIR="$TEST_DIR/run_output"
LOG_DIR="$RUN_DIR/logs"
AUDIO_LIST_FILE="$TEST_DIR/audio_list.txt"
AUDIO_DIR="$ROOT_DIR/audio_in"
SBATCH_RUNTIME="$ROOT_DIR/sbatch_runtime"
VENV_PATH="$SBATCH_RUNTIME/.venv"
CLI_SCRIPT="$ROOT_DIR/workflow_cli.py"
NUMBER_AUDIO_SCRIPT="$ROOT_DIR/audio_numbering.py"
REQ_CONSTRAINTS_FILE="$PROJECT_ROOT/constraints.txt"
REQ_COMMON_FILE="$PROJECT_ROOT/requirements_common.txt"
REQ_BACKEND_FILE="$PROJECT_ROOT/requirements_nemo.txt"
REQ_HASH_FILE="$SBATCH_RUNTIME/.requirements_nemo.sha256"
RUNTIME_ENV_SCRIPT="$ROOT_DIR/scheduler/sbatch_runtime_env.sh"

NEMO_MSDD_MODEL="$ROOT_DIR/fine_tuning/projects/nemo/nemo-test1/artifacts/experiments/nemo-test1-trained-version-11/nemo-test1-trained-version-11/2026-05-07_18-25-19/checkpoints/nemo-test1-trained-version-11.nemo"

CACHE_BASE="${HOME}/.cache"
TRANSFORMERS_CACHE="${CACHE_BASE}/huggingface"
PIP_CACHE_DIR="${CACHE_BASE}/pip"
TMPDIR="$SBATCH_RUNTIME/tmp"

export PIP_INDEX_URL="https://pypi.org/simple"
export PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$TMPDIR" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"

finish() {
  rc=$?
  trap - EXIT
  printf "%s\n" "$rc" > "$RUN_DIR/exit_code.txt"
  if [ "$rc" -eq 0 ]; then
    echo "TEST PASSED: fine-tune v11 full pipeline completed successfully."
  else
    echo "TEST FAILED: pipeline exited with code $rc." >&2
  fi
  exit "$rc"
}
trap finish EXIT

exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)

echo "=== Fine-tune v11 integration test ==="
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Run dir: $RUN_DIR"
echo "Audio list: $AUDIO_LIST_FILE"
echo "NeMo MSDD model: $NEMO_MSDD_MODEL"

# Verify the .nemo file exists before spending GPU time
if [ ! -f "$NEMO_MSDD_MODEL" ]; then
  echo "ERROR: .nemo file not found: $NEMO_MSDD_MODEL" >&2
  exit 1
fi
echo "Model file found: $(du -sh "$NEMO_MSDD_MODEL" | cut -f1)"

module load Python/3.12.3-GCCcore-14.2.0
PYTHON_GCCCORE_LIB64="${EBROOTGCCCORE:+$EBROOTGCCCORE/lib64}"

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
from numba import cuda
PY
}

REQ_HASH="$(
  cat "$REQ_CONSTRAINTS_FILE" "$REQ_COMMON_FILE" "$REQ_BACKEND_FILE" \
    | sha256sum | awk '{print $1}'
)"
CURRENT_HASH="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"
REPAIR_RUNTIME=0

if [ "$REQ_HASH" = "$CURRENT_HASH" ]; then
  if ! nemo_runtime_is_healthy; then
    echo "NeMo runtime compatibility check failed; reinstalling constrained dependencies."
    REPAIR_RUNTIME=1
  fi
fi

if [ "$REQ_HASH" != "$CURRENT_HASH" ] || [ "$REPAIR_RUNTIME" = "1" ]; then
  echo "Installing/updating diarization dependencies..."
  python -m pip install --upgrade pip >"$LOG_DIR/pip_bootstrap.out" 2>"$LOG_DIR/pip_bootstrap.err"
  python -m pip install \
    --extra-index-url "$PIP_EXTRA_INDEX_URL" \
    -c "$REQ_CONSTRAINTS_FILE" \
    -r "$REQ_COMMON_FILE" \
    -r "$REQ_BACKEND_FILE" \
    >"$LOG_DIR/pip_requirements.out" \
    2>"$LOG_DIR/pip_requirements.err"
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
else
  echo "Dependencies unchanged; skipping pip install."
fi

# Quick import check: confirm the .nemo loads into NeuralDiarizer without error
echo "--- Step 1: model load check ---"
SENIOR_DESIGN_ROOT="$ROOT_DIR" python - "$NEMO_MSDD_MODEL" <<'PY'
import sys, os
from pathlib import Path

nemo_path = sys.argv[1]
print(f"Loading model from: {nemo_path}")

senior_root = os.environ["SENIOR_DESIGN_ROOT"]
sys.path.insert(0, senior_root)

from all_diarization_programs.nemo_msdd_backend import _ensure_msdd_loadable, create_config
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

loadable_path = _ensure_msdd_loadable(nemo_path)
print(f"Loadable path (patched if needed): {loadable_path}")

cfg = create_config(msdd_model_path=loadable_path)
model = NeuralDiarizer(cfg=cfg).to(device)
print("Model loaded successfully.")
print(f"Model type: {type(model).__name__}")
PY

echo "--- Step 2: full diarization pipeline ---"
python "$CLI_SCRIPT" run-diarization-selection \
  --audio-list-file "$AUDIO_LIST_FILE" \
  --backend nemo \
  --output-dir "$RUN_DIR" \
  --log-dir "$LOG_DIR" \
  --device cuda \
  --whisper-model small.en \
  --batch-size 8 \
  --nemo-msdd-model "$NEMO_MSDD_MODEL"

echo "--- Step 3: output validation ---"
python - "$RUN_DIR" <<'PY'
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
srt_files = list(run_dir.rglob("*.srt"))
rttm_files = list(run_dir.rglob("*.rttm"))
print(f"SRT outputs found:  {len(srt_files)}")
print(f"RTTM outputs found: {len(rttm_files)}")
for f in srt_files:
    lines = f.read_text(encoding="utf-8").strip().splitlines()
    print(f"  {f.name}: {len(lines)} lines")
if not srt_files:
    print("ERROR: no SRT output produced!", file=sys.stderr)
    sys.exit(1)
print("Output validation passed.")
PY

echo "=== All steps passed. Fine-tune v11 is working correctly. ==="
