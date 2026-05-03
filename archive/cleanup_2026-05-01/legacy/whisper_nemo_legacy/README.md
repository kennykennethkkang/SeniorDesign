# Whisper-NeMo Legacy Utilities

This folder now contains only an optional one-off YouTube downloader:

- `youtube_downloader.py`

The batch YouTube conversion program is now:

- `../youtube_audio_batch.py` (used by `../run_youtube_audio_download.sbatch`)

All diarization code has been consolidated into:

- `../all_diarization_programs/`

Use these entrypoints from `SeniorDesign/`:

```bash
./submit_single_diarization_job.sh
./submit_bulk_diarization_job.sh
sbatch run_youtube_audio_download.sbatch
```
