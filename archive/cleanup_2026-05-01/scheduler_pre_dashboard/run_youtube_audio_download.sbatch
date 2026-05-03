#!/bin/bash -l
#
# Batch: standalone YouTube-link to audio conversion (stored as WAV, no diarization).
# Submit from anywhere: sbatch /WAVE/users2/unix/kkang/SeniorDesign/run_youtube_audio_download.sbatch
#
#SBATCH --job-name=youtube_to_audio
#SBATCH --output=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/youtube_to_audio_%j.out
#SBATCH --error=/WAVE/users2/unix/kkang/SeniorDesign/job_logs/slurm/youtube_to_audio_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00

set -euo pipefail

SELF_DIR="/WAVE/users2/unix/kkang/SeniorDesign"
YOUTUBE_TO_AUDIO_SCRIPT="$SELF_DIR/youtube_audio_batch.py"
WAV_DIR="$SELF_DIR/audio_in"
LINK_FILE="$SELF_DIR/youtube_links.txt"
SBATCH_RUNTIME="$SELF_DIR/sbatch_runtime"
JOB_LOG_ROOT="$SELF_DIR/job_logs"
JOB_SLURM_LOGS="$JOB_LOG_ROOT/slurm"
JOB_PREFIX="youtube_to_audio_job_${SLURM_JOB_ID:-manual}"
JOB_LOG_DIR="$JOB_LOG_ROOT/$JOB_PREFIX"
LOCATIONS_FILE="$JOB_LOG_DIR/locations.tsv"
SLURM_OUT_FILE="$JOB_SLURM_LOGS/youtube_to_audio_${SLURM_JOB_ID:-manual}.out"
SLURM_ERR_FILE="$JOB_SLURM_LOGS/youtube_to_audio_${SLURM_JOB_ID:-manual}.err"
VENV_PATH="$SBATCH_RUNTIME/.yt_audio_venv"
URL_INDEX_FILE="$SBATCH_RUNTIME/url_audio_index.tsv"
URL_RESOLUTION_FILE="$JOB_LOG_DIR/conversion_report.tsv"
RESOLVED_AUDIO_LIST="$JOB_LOG_DIR/resolved_audio_paths.txt"
REMOVED_FAILED_FILE="$JOB_LOG_DIR/queue_updates.tsv"
YOUTUBE_LINK_ERR_DIR="$SELF_DIR/youtube_links_err"
YOUTUBE_LINK_ERR_REPORT="$YOUTUBE_LINK_ERR_DIR/failed_links_${SLURM_JOB_ID:-manual}.tsv"
CACHE_BASE="${HOME}/.cache"
PIP_CACHE_DIR="${CACHE_BASE}/pip"
TMPDIR="$SBATCH_RUNTIME/tmp"
NUMBER_AUDIO_SCRIPT="$SELF_DIR/audio_numbering.py"

mkdir -p \
  "$WAV_DIR" \
  "$JOB_LOG_ROOT" \
  "$JOB_SLURM_LOGS" \
  "$JOB_LOG_DIR" \
  "$YOUTUBE_LINK_ERR_DIR" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR"

{
  echo -e "key\tvalue"
  echo -e "job_id\t${SLURM_JOB_ID:-manual}"
  echo -e "job_type\tyoutube_to_audio"
  echo -e "audio_output_dir\t$WAV_DIR"
  echo -e "youtube_links_file\t$LINK_FILE"
  echo -e "youtube_to_audio_script\t$YOUTUBE_TO_AUDIO_SCRIPT"
  echo -e "job_log_dir\t$JOB_LOG_DIR"
  echo -e "slurm_stdout\t$SLURM_OUT_FILE"
  echo -e "slurm_stderr\t$SLURM_ERR_FILE"
  echo -e "url_resolution_file\t$URL_RESOLUTION_FILE"
  echo -e "resolved_audio_list\t$RESOLVED_AUDIO_LIST"
  echo -e "removed_failed_file\t$REMOVED_FAILED_FILE"
  echo -e "youtube_links_err_dir\t$YOUTUBE_LINK_ERR_DIR"
  echo -e "youtube_links_err_report\t$YOUTUBE_LINK_ERR_REPORT"
} > "$LOCATIONS_FILE"

echo "Audio output directory: $WAV_DIR"
echo "Job log directory: $JOB_LOG_DIR"
echo "Slurm stdout/stderr: $SLURM_OUT_FILE | $SLURM_ERR_FILE"

if [ ! -f "$LINK_FILE" ]; then
  echo "URL list file not found: $LINK_FILE" >&2
  exit 1
fi
if [ ! -f "$YOUTUBE_TO_AUDIO_SCRIPT" ]; then
  echo "YouTube-to-audio script not found: $YOUTUBE_TO_AUDIO_SCRIPT" >&2
  exit 1
fi
if [ ! -f "$NUMBER_AUDIO_SCRIPT" ]; then
  echo "Audio numbering script not found: $NUMBER_AUDIO_SCRIPT" >&2
  exit 1
fi

module load Python/3.12.3-GCCcore-14.2.0
if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
export PIP_INDEX_URL="https://pypi.org/simple"
export PIP_CACHE_DIR TMPDIR
REQ_HASH_FILE="$SBATCH_RUNTIME/.yt_audio_requirements.sha256"
REQ_HASH="$(printf '%s\n' 'yt-dlp>=2025.1.0' 'imageio-ffmpeg>=0.5.1' | sha256sum | awk '{print $1}')"
CURRENT_HASH="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

if [ "$REQ_HASH" != "$CURRENT_HASH" ]; then
  echo "Installing/updating downloader dependencies..."
  python -m pip install --upgrade pip \
    >"$JOB_LOG_DIR/pip_bootstrap.out" \
    2>"$JOB_LOG_DIR/pip_bootstrap.err"
  python -m pip install --upgrade yt-dlp imageio-ffmpeg \
    >"$JOB_LOG_DIR/pip_requirements.out" \
    2>"$JOB_LOG_DIR/pip_requirements.err"
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
else
  echo "Downloader dependencies unchanged; skipping pip install."
fi

# Ensure all existing audio_in files are numbered before download.
python "$NUMBER_AUDIO_SCRIPT" --audio-dir "$WAV_DIR" \
  >"$JOB_LOG_DIR/audio_renumber_pre.out" \
  2>"$JOB_LOG_DIR/audio_renumber_pre.err"

cd "$SELF_DIR"
python "$YOUTUBE_TO_AUDIO_SCRIPT" \
  --list "$LINK_FILE" \
  --queue-file "$LINK_FILE" \
  --output-dir "$WAV_DIR" \
  --index-file "$URL_INDEX_FILE" \
  --resolved-output-file "$URL_RESOLUTION_FILE" \
  --resolved-audio-list "$RESOLVED_AUDIO_LIST" \
  --removed-failed-file "$REMOVED_FAILED_FILE" \
  --links-error-dir "$YOUTUBE_LINK_ERR_DIR" \
  --links-error-report "$YOUTUBE_LINK_ERR_REPORT" \
  --log-dir "$JOB_LOG_DIR" \
  >"$JOB_LOG_DIR/download_youtube.out" \
  2>"$JOB_LOG_DIR/download_youtube.err"

if [ -s "$RESOLVED_AUDIO_LIST" ]; then
  while IFS= read -r wav; do
    [ -z "$wav" ] && continue
    if [ ! -f "$wav" ]; then
      echo "Resolved audio path does not exist: $wav" >&2
      exit 1
    fi
    wav_real="$(readlink -f "$wav")"
    case "$wav_real" in
      "$WAV_DIR"/*) ;;
      *)
        echo "Resolved audio path is outside audio_in ($WAV_DIR): $wav_real" >&2
        exit 1
        ;;
    esac
  done < "$RESOLVED_AUDIO_LIST"
fi

echo "YouTube-to-audio conversion complete."
echo "Audio files are in $WAV_DIR"
echo "Per-run logs are in $JOB_LOG_DIR"
