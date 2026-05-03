#!/bin/bash -l
#
# Site-launched YouTube audio conversion job.
# The dashboard writes the selected URL list and output folder first, then
# submits this script with environment variables so every run stays isolated.
#
#SBATCH --job-name=site_youtube_audio
#SBATCH --output=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/site_youtube_audio_%j.out
#SBATCH --error=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/site_youtube_audio_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00

set -euo pipefail

ROOT_DIR="${SITE_ROOT_DIR:-/WAVE/users2/unix/kkang/SeniorDesign}"
RUN_DIR="${SITE_YOUTUBE_RUN_DIR:?SITE_YOUTUBE_RUN_DIR is required}"
URLS_FILE="${SITE_YOUTUBE_URLS_FILE:?SITE_YOUTUBE_URLS_FILE is required}"
QUEUE_FILE="${SITE_YOUTUBE_QUEUE_FILE:-$ROOT_DIR/youtube_links.txt}"
INDEX_FILE="${SITE_YOUTUBE_INDEX_FILE:-$ROOT_DIR/outputs/youtube_conversion_history/url_audio_index.tsv}"
FORCE_REDOWNLOAD="${SITE_YOUTUBE_FORCE_REDOWNLOAD:-0}"
AUDIO_DIR="${SITE_YOUTUBE_AUDIO_DIR:-$ROOT_DIR/audio_in/youtube_links}"
LOG_DIR="$RUN_DIR/logs"
SBATCH_RUNTIME="$ROOT_DIR/sbatch_runtime"
JOB_LOG_ROOT="$ROOT_DIR/job_logs"
JOB_SLURM_LOGS="$JOB_LOG_ROOT/slurm"
LOCATIONS_FILE="$LOG_DIR/locations.tsv"
CLI_SCRIPT="$ROOT_DIR/workflow_cli.py"
VENV_PATH="$SBATCH_RUNTIME/.yt_audio_venv"
CACHE_BASE="${HOME}/.cache"
PIP_CACHE_DIR="${CACHE_BASE}/pip"
TMPDIR="$SBATCH_RUNTIME/tmp"
export PIP_INDEX_URL="https://pypi.org/simple"

mkdir -p \
  "$RUN_DIR" \
  "$LOG_DIR" \
  "$AUDIO_DIR" \
  "$JOB_LOG_ROOT" \
  "$JOB_SLURM_LOGS" \
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
  echo -e "job_type\tsite_youtube_audio_conversion"
  echo -e "run_dir\t$RUN_DIR"
  echo -e "selected_urls_file\t$URLS_FILE"
  echo -e "queue_file\t$QUEUE_FILE"
  echo -e "history_index\t$INDEX_FILE"
  echo -e "audio_output_dir\t$AUDIO_DIR"
  echo -e "force_redownload\t$FORCE_REDOWNLOAD"
  echo -e "venv_path\t$VENV_PATH"
} > "$LOCATIONS_FILE"

echo "Site YouTube audio conversion Slurm job: ${SLURM_JOB_ID:-manual}"
echo "Run directory: $RUN_DIR"
echo "Selected URL list: $URLS_FILE"
echo "Queue file: $QUEUE_FILE"

for required_path in "$URLS_FILE" "$CLI_SCRIPT"; do
  if [ ! -e "$required_path" ]; then
    echo "Required path not found: $required_path" >&2
    exit 1
  fi
done

module load Python/3.12.3-GCCcore-14.2.0
if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
export PIP_CACHE_DIR TMPDIR

REQ_HASH_FILE="$SBATCH_RUNTIME/.yt_audio_requirements.sha256"
REQ_HASH="$(printf '%s\n' 'yt-dlp>=2025.1.0' 'imageio-ffmpeg>=0.5.1' | sha256sum | awk '{print $1}')"
CURRENT_HASH="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

if [ "$REQ_HASH" != "$CURRENT_HASH" ]; then
  echo "Installing or updating YouTube audio dependencies..."
  python -m pip install --upgrade pip \
    >"$LOG_DIR/pip_bootstrap.out" \
    2>"$LOG_DIR/pip_bootstrap.err"
  python -m pip install --upgrade yt-dlp imageio-ffmpeg \
    >"$LOG_DIR/pip_requirements.out" \
    2>"$LOG_DIR/pip_requirements.err"
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
else
  echo "YouTube audio dependencies are unchanged; skipping pip install."
fi

command=(
  python "$CLI_SCRIPT"
  convert-youtube-selection
  --urls-file "$URLS_FILE"
  --queue-file "$QUEUE_FILE"
  --run-dir "$RUN_DIR"
  --index-file "$INDEX_FILE"
  --output-dir "$AUDIO_DIR"
)

if [ "$FORCE_REDOWNLOAD" = "1" ]; then
  command+=(--force-redownload)
fi

echo "Running YouTube audio conversion command:"
printf "  %q" "${command[@]}"
printf "\n"
"${command[@]}"

echo "Site YouTube audio conversion complete."
echo "Outputs and reports are in $RUN_DIR"
