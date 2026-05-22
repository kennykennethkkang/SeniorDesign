# Scheduler Folder

The dashboard submits its long-running work through Slurm. Three batch
scripts and one shared helper live here; everything else has been moved to
`archive/cleanup_2026-05-01/scheduler_pre_dashboard/` because the dashboard
now drives those flows from the browser.

Active scripts:

- `run_site_diarization.sbatch`: diarization runs launched from the dashboard's Diarization tab.
  Reads its inputs from environment variables (`SITE_DIARIZATION_RUN_DIR`,
  `SITE_DIARIZATION_AUDIO_LIST`, `DIARIZATION_BACKEND`, `WHISPER_MODEL`,
  `PYANNOTE_*` model overrides, etc.) that `workflow_dashboard.py` sets up
  before calling `sbatch`.
- `run_site_youtube_conversion.sbatch`: YouTube to WAV runs launched from the
  YouTube Audio Conversion tab. Routes downloads into
  `audio_in/youtube_links/` by default and updates
  `outputs/youtube_conversion_history/url_audio_index.tsv` so future runs
  know what has already been fetched.
- `run_site_stitching.sbatch`: audio stitching runs launched from the
  Stitching tab. Drives `stitch_audio.py` to build longer multi-speaker
  clips and mirrors the result back into `audio_in/`.
- `sbatch_runtime_env.sh`: sourced by the batch scripts to activate the
  shared `sbatch_runtime/.venv` virtual environment. Avoids duplicating the
  Python/CUDA/ffmpeg setup in each script.

Everyday use should go through the dashboard. If you need to run one of these
scripts by hand for debugging, set the same environment variables the
dashboard passes (see the per-script comments) before calling `sbatch`.
