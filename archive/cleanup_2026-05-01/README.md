# Archive — 2026-05-01 Cleanup Pass

Snapshot of files retired during the May 2026 dashboard cleanup. Nothing here
is on the active dashboard or CLI path; the contents are kept so you can
recover or reference them without having to dig through Git history.

## Contents

- `cli/senior_design_cli.py`
  Original umbrella CLI. Superseded by `workflow_cli.py`, which is the
  canonical entrypoint and is what the README documents.

- `scheduler_pre_dashboard/`
  Slurm batch scripts and shell helpers that pre-dated the dashboard. The
  dashboard now drives both diarization and YouTube conversion through
  `scheduler/run_site_diarization.sbatch` and
  `scheduler/run_site_youtube_conversion.sbatch`, which use environment
  variables instead of these positional shell wrappers. Includes:
  `submit_single_diarization.sh`, `submit_bulk_diarization.sh`, the matching
  `_job.sh` helpers, `bulk_diarization.sbatch`, `single_diarization.sbatch`,
  `run_single_diarization.sbatch`, `run_bulk_diarization.sbatch`,
  `run_youtube_audio_download.sbatch`, and `youtube_to_audio_only.sbatch`.

- `legacy/`
  Earlier project iterations: an older Whisper + NeMo prototype tree and
  miscellaneous reference assets from before the dashboard surface existed.

- `job_outputs/`
  Per-run output folders written by the legacy `submit-single` / `submit-bulk`
  CLI commands. Dashboard-launched runs now write into
  `outputs/diarization_runs/<backend>/<run>/` instead.

## When to use this folder

Treat this as cold storage. If you find yourself wanting to "bring something
back," the right move is usually to copy the relevant logic into one of the
active modules (and update the tests) rather than to start invoking these
scripts again.
