#!/bin/bash -l
#
# Site-launched audio stitching job.
# The dashboard writes the selected audio+speaker JSON first, then submits this
# script so large stitched RTTM samples can be created without tying up the web request.
#
#SBATCH --job-name=site_audio_stitch
#SBATCH --output=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/site_audio_stitch_%j.out
#SBATCH --error=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/site_audio_stitch_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT_DIR="${SITE_ROOT_DIR:-/WAVE/users2/unix/kkang/SeniorDesign}"
RUN_DIR="${SITE_STITCH_RUN_DIR:?SITE_STITCH_RUN_DIR is required}"
INPUTS_JSON="${SITE_STITCH_INPUTS_JSON:?SITE_STITCH_INPUTS_JSON is required}"
OUTPUT_NAME="${SITE_STITCH_OUTPUT_NAME:-stitched-audio}"
SEED="${SITE_STITCH_SEED:-}"
TRAINING_TARGETS_JSON="${SITE_STITCH_TRAINING_TARGETS_JSON:-[]}"
LOG_DIR="$RUN_DIR/logs"
JOB_LOG_ROOT="$ROOT_DIR/job_logs"
JOB_SLURM_LOGS="$JOB_LOG_ROOT/slurm"
LOCATIONS_FILE="$LOG_DIR/locations.tsv"
STITCH_SCRIPT="$ROOT_DIR/stitch_audio.py"
SBATCH_RUNTIME="$ROOT_DIR/sbatch_runtime"
TMPDIR="$SBATCH_RUNTIME/tmp"

mkdir -p \
  "$RUN_DIR" \
  "$LOG_DIR" \
  "$JOB_LOG_ROOT" \
  "$JOB_SLURM_LOGS" \
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
  echo -e "job_type\tsite_audio_stitching"
  echo -e "run_dir\t$RUN_DIR"
  echo -e "inputs_json\t$INPUTS_JSON"
  echo -e "output_name\t$OUTPUT_NAME"
  echo -e "seed\t$SEED"
  echo -e "training_targets\t$TRAINING_TARGETS_JSON"
} > "$LOCATIONS_FILE"

echo "Site audio stitching Slurm job: ${SLURM_JOB_ID:-manual}"
echo "Run directory: $RUN_DIR"
echo "Selected audio JSON: $INPUTS_JSON"

for required_path in "$INPUTS_JSON" "$STITCH_SCRIPT"; do
  if [ ! -e "$required_path" ]; then
    echo "Required path not found: $required_path" >&2
    exit 1
  fi
done

module load Python/3.12.3-GCCcore-14.2.0
export TMPDIR

mapfile -t training_targets < <(
  python3 - "$TRAINING_TARGETS_JSON" <<'PY'
import json
import sys

try:
    values = json.loads(sys.argv[1])
except json.JSONDecodeError:
    values = []
for value in values if isinstance(values, list) else []:
    print(str(value))
PY
)

command=(
  python3 "$STITCH_SCRIPT"
  --root "$ROOT_DIR"
  --inputs-json "$INPUTS_JSON"
  --run-dir "$RUN_DIR"
  --output-name "$OUTPUT_NAME"
)

if [ -n "$SEED" ]; then
  command+=(--seed "$SEED")
fi

for target in "${training_targets[@]}"; do
  command+=(--training-target "$target")
done

echo "Running audio stitching command:"
printf "  %q" "${command[@]}"
printf "\n"
"${command[@]}"

echo "Audio stitching complete."
echo "Outputs are in $RUN_DIR"
