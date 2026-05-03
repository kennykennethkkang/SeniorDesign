#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_bulk_diarization.sbatch"
selected_backend="${DIARIZATION_BACKEND:-}"

usage() {
  cat <<'USAGE'
Usage:
  ./submit_bulk_diarization_job.sh
  ./submit_bulk_diarization_job.sh --backend <nemo|pyannote>

Description:
  Prompts for diarization backend and submits bulk diarization job.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
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

if [[ -z "$selected_backend" ]]; then
  echo "Select diarization backend for bulk job:"
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

echo "Submitting bulk diarization"
echo "Backend: $selected_backend"
submit_output="$(
  sbatch --export="ALL,DIARIZATION_BACKEND=$selected_backend" "$SBATCH_SCRIPT"
)"
echo "$submit_output"

job_id="$(echo "$submit_output" | awk '{print $NF}')"
if [[ "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Check paths after start: $PROJECT_ROOT/job_logs/bulk_diarization_job_${job_id}/locations.tsv"
fi
