#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONDONTWRITEBYTECODE=1

python3 -m py_compile \
  workflow_dashboard.py \
  workflow_background.py \
  workflow_cli.py \
  fine_tuning_manager.py \
  diarization_metrics.py \
  review_bundle.py \
  audio_numbering.py \
  youtube_audio_batch.py \
  all_diarization_programs/run_diarization.py

npm run check --prefix frontend

for script in scheduler/*.sbatch scheduler/*.sh; do
  if [ -e "$script" ]; then
    bash -n "$script"
  fi
done

python3 -m unittest discover -s tests -v
