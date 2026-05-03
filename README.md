# ML Speech Diarization

This repository is organized around a local dashboard and a small set of CLI tools for:

- collecting source media
- converting queued YouTube links into numbered audio files
- running single-file or multi-file diarization
- generating review bundles
- preparing and launching fine-tuning runs for NeMo or pyannote

The current workflow is built around these dashboard tabs:

- `Overview`
- `Media Library`
- `Training Labels`
- `YouTube Audio Conversion`
- `Diarization`
- `Fine-Tuning`

Legacy CLI commands and older output folders still exist for compatibility, but the names above are the primary workflow going forward.

## Project layout

- `workflow_cli.py`: top-level CLI for status, tests, review generation, and the local web launcher
- `workflow_dashboard.py`: backend routes, JSON page state, and local web launcher integration for the dashboard
- `frontend/`: dashboard website files split into `html/`, `static/css/`, `static/js/`, and local React runtime files
- `workflow_preferences.py`: persistent defaults used by the diarization and fine-tuning pages
- `workflow_background.py`: shared helpers for detached and Slurm-backed runs launched from the site
- `fine_tuning_manager.py`: upload, prepare, and launch logic for NeMo and pyannote fine-tuning projects
- `review_bundle.py`: review HTML and TSV flag generation from `.srt` files
- `audio_numbering.py`: numbering utility for `audio_in/`
- `youtube_audio_batch.py`: YouTube URL to audio downloader and conversion-history indexer used by the local CLI
- `youtube_to_audio_batch.py`: same downloader, invoked from the dashboard's Slurm wrapper
- `all_diarization_programs/`: diarization entrypoints and backend-specific code
- `scheduler/`: Slurm batch scripts for the dashboard's diarization and YouTube conversion jobs
- `docs/`: local guides and reference papers for the project
- `archive/`: dated snapshots of code, scripts, and runs that are no longer part of the main workflow but kept for reference
- `audio_in/`: numbered source media inputs (with `youtube_links/` and `file_uploads/` subfolders for site-created assets)
- `outputs/`: site-created diarization runs, YouTube conversion runs, and the URL-to-audio history index
- `fine_tuning/projects/<backend>/<project-slug>/`: backend-specific training workspaces
- `workflow_preferences.json`: saved model defaults created after the first settings save
- `youtube_links.txt`: queued YouTube URLs

## Basic status check

From the project root:

```bash
cd /WAVE/users2/unix/kkang/SeniorDesign
python3 workflow_cli.py status
```

This reports the current audio count, queued YouTube count, tool availability, and the newest output folders.

## Start the site

Run:

```bash
python3 workflow_cli.py local-web start
```

Useful follow-up commands:

```bash
python3 workflow_cli.py local-web status
python3 workflow_cli.py local-web stop
```

Important behavior:

- the launcher binds to `127.0.0.1` for local-only use
- if port `8000` is busy, the launcher automatically moves to the next free port
- the active URL is written to `.local_dashboard/dashboard.url`
- startup logs are written to `.local_dashboard/dashboard.log`

If you want the original foreground server for debugging:

```bash
python3 workflow_cli.py serve-web --port 8000 --server auto
```

## Site workflow

### 1. Media Library

Use `Media Library` to upload supported audio files into `audio_in/`.

What happens:

- files are copied into `audio_in/`
- files are renumbered into the expected format such as `001_example.wav`
- later tabs see a clean, stable numbered inventory

CLI equivalent:

```bash
python3 audio_numbering.py --audio-dir ./audio_in
```

### 2. YouTube Audio Conversion

Use `YouTube Audio Conversion` to:

- append one or more links to the queue
- select specific queued links
- convert the full queue with one action
- avoid duplicate conversions through a persistent history index

The page now supports single selection or full-queue conversion without switching tools.

Outputs from each site-launched conversion run:

- numbered audio files in `audio_in/`
- run folder in `outputs/youtube_conversion_runs/<timestamp>_youtube-conversion_<count>/`
- `selected_urls.txt`
- `conversion_report.tsv`
- `resolved_audio_paths.txt`
- `queue_updates.tsv`
- `item_logs/` for per-URL debug logs only when a download attempt needs extra inspection
- `stdout.log` as the main activity log and `stderr.log` for wrapper-level errors

Persistent conversion history:

- `outputs/youtube_conversion_history/url_audio_index.tsv`

Issue reports:

- `youtube_links_err/failed_links_latest.tsv`

Queue behavior:

- successful downloads stay indexed and keep their audio path in `audio_in/`
- retryable downloader or environment errors stay queued so you can run them again
- only URLs that clearly have no public audio data are removed from the queue automatically

You can still edit the raw queue directly:

```text
youtube_links.txt
```

### 3. Diarization

Use `Diarization` to:

- select one file
- select several files
- run the whole numbered library
- choose between `nemo` and `pyannote`
- save the default backend, Whisper model, batch size, and pyannote model IDs

Each new site-launched diarization run gets its own output folder:

- `outputs/diarization_runs/<timestamp>_diarization_<backend>_<count>/`

The site submits diarization through Slurm with:

- `scheduler/run_site_diarization.sbatch`

Each run folder includes the selected audio list, metadata, wrapper logs, per-file logs, transcripts, SRT diarized time spans, review HTML, and review flag reports when those artifacts are produced.

Typical contents of one run folder:

- diarization outputs generated by `run_diarization.py`
- `selected_audio.txt`
- `stdout.log` and `stderr.log`
- `logs/runtime_summary.tsv`
- per-file stdout and stderr logs under `logs/`

This keeps every run separate, which makes backend comparisons and reruns easier to inspect.

### 4. Review

Use `Review` to generate:

- `<name>_review.html`
- `<name>_review_flags.tsv`

The page works from an existing `.srt` file and can optionally attach a matching media path.

CLI equivalent:

```bash
python3 workflow_cli.py review --srt path/to/file.srt --media path/to/file.wav
```

### 5. Fine-Tuning

Use `Fine-Tuning` to manage backend-specific training workspaces under:

- `fine_tuning/projects/nemo/<project-slug>/`
- `fine_tuning/projects/pyannote/<project-slug>/`

Each project follows the same high-level sequence:

1. Upload one or more aligned audio + RTTM sample pairs.
2. Prepare backend-specific training artifacts.
3. Launch locally or submit through Slurm.

Uploaded sample storage:

- `audio/`
- `rttm/`
- optional `text/`

Prepared NeMo artifacts include:

- train and validation session manifests
- train and validation MSDD manifests
- pairwise RTTM files when needed
- local and Slurm launch scripts

Prepared pyannote artifacts include:

- `database.yml`
- `train.lst`, `development.lst`, and `test.lst`
- subset RTTM and UEM files for each split
- `train_pyannote.py`
- local and Slurm launch scripts

Training runs are stored under:

- `fine_tuning/projects/<backend>/<project-slug>/runs/<timestamp>/`

Status command:

```bash
python3 fine_tuning_manager.py status
```

Example prepare and launch commands:

```bash
python3 fine_tuning_manager.py prepare --backend nemo --project my-project
python3 fine_tuning_manager.py launch --backend nemo --project my-project
python3 fine_tuning_manager.py prepare --backend pyannote --project my-project
python3 fine_tuning_manager.py launch --backend pyannote --project my-project
```

The pyannote preparation flow is aligned with the official pyannote training tutorial structure: it generates a `pyannote.database` configuration, prepares split metadata, and writes a training script that loads the dataset registry before building a `SpeakerDiarization` task from a pretrained model.

## Fine-Tuning Breakdown

For an exact training runbook with the current workspace audio names, recommended
project names, field values, launch commands, and troubleshooting checks, open:

```text
docs/fine_tuning_training_runbook.md
```

### What data is required

For both NeMo and pyannote in this project, the required supervision is the same:

- one audio file per sample
- one matching RTTM file per sample

Optional data:

- transcript text file
- transcript notes typed into the site

The optional transcript material is helpful for human review, but the diarization fine-tuning code here is driven by the audio and RTTM labels.

### What a good sample looks like

Each uploaded pair should satisfy these rules:

- the audio file and RTTM refer to the same recording
- speaker turns in the RTTM line up with the real start and end times in the audio
- speaker labels are consistent within the file
- the file stem is stable enough that you can track it across uploads, preparation, and review

### What makes a dataset strong enough to fine-tune

Minimum expectations:

- multiple recordings, not just one long file
- multiple speakers across the project
- enough labeled speech to cover the speaking styles you expect at inference time
- a clean validation split so you can tell whether the model is actually improving

Practical guidance:

- if the project audio is noisy, overlapping, or highly conversational, your fine-tuning data should include the same conditions
- if you plan to diarize phone calls, meetings, or interviews, do not fine-tune only on clean studio speech
- more high-quality RTTM supervision beats more weakly labeled data

### What the site does at each step

1. `Add Training Sample`
   This stores audio under `audio/`, RTTM under `rttm/`, and optional text under `text/`.
2. `Prepare Training Artifacts`
   This builds manifests, split lists, launcher scripts, and backend-specific training files.
3. `Launch Background Training Run`
   This starts a local background run or submits the Slurm script when scheduler tools are available.

### Backend-specific notes

NeMo:

- this workflow prepares session manifests and MSDD manifests
- local runs may need `NEMO_ROOT` if the NeMo checkout is outside the current environment
- the project uses the saved NeMo config name, speaker model, and window settings during preparation

pyannote:

- this workflow prepares `database.yml`, split lists, RTTM files, UEM files, and a generated training script
- pyannote runs require `HF_TOKEN`
- the selected pretrained pyannote model and chunk settings are saved in the model preferences

### How to verify that data loaded correctly

The site now exposes these checks directly:

- upload forms show the selected filename immediately after you choose a file
- the fine-tuning page shows progress cards for uploaded projects, prepared projects, and active runs
- each project card shows a three-step progress bar: upload, prepare, launch
- the diarization and YouTube pages show selection counters before you launch a run
- the latest run panels show live status, logs, and summary tables

## Output locations

### Site-created YouTube conversion runs

- `outputs/youtube_conversion_runs/`
- `outputs/youtube_conversion_history/url_audio_index.tsv`

### Site-created diarization runs

- `outputs/diarization_runs/`

### Fine-tuning runs

- `fine_tuning/projects/<backend>/<project-slug>/runs/`

## Support folders

- `scheduler/`: Slurm batch scripts and the shared runtime helper used by the dashboard
- `docs/integration_testing_guide.md`: manual integration checklist
- `docs/reference_materials/`: local PDF references used while shaping the workflow
- `archive/`: dated snapshots of pre-dashboard scripts, output folders, and earlier prototypes — kept for reference but not part of the active workflow

## Testing

The test cases live in `tests/`. Run the full local check with:

```bash
bash tests/run_all_tests.sh
```

The full check runs Python syntax checks, frontend JavaScript syntax checks, Slurm script syntax checks, and the Python unit test suite. You can also use `python3 workflow_cli.py test`, which calls the same runner through `run_tests.sh`.

For details about each test file, open `tests/README.md`.

## Common issues

### Port already in use

Start the site normally:

```bash
python3 workflow_cli.py local-web start
```

If `8000` is busy, the launcher automatically chooses the next free port.

### pyannote authentication

Pyannote runs require a Hugging Face token and access to the gated pyannote models.

Set:

```bash
export HF_TOKEN=...
```

### NeMo local fine-tuning

Local NeMo launches need a valid `NEMO_ROOT` or the matching path supplied in the launch form.

### Runtime environment problems

If an environment under `sbatch_runtime/.venv` behaves oddly, open a fresh shell and run:

```bash
cd /WAVE/users2/unix/kkang/SeniorDesign
python3 workflow_cli.py status
```

## Quick command reference

```bash
cd /WAVE/users2/unix/kkang/SeniorDesign
python3 workflow_cli.py status
python3 workflow_cli.py local-web start
python3 workflow_cli.py local-web status
python3 workflow_cli.py local-web stop
python3 workflow_cli.py serve-web --port 8000 --server auto
python3 workflow_cli.py test
python3 workflow_cli.py review --srt path/to/file.srt --media path/to/file.wav
python3 fine_tuning_manager.py status
python3 fine_tuning_manager.py prepare --backend pyannote --project my-project
python3 fine_tuning_manager.py launch --backend pyannote --project my-project
python3 fine_tuning_manager.py prepare --backend nemo --project my-project
python3 fine_tuning_manager.py launch --backend nemo --project my-project
```

## Project history

Earlier iterations of this project shipped a Slurm-oriented CLI (`submit-single`, `submit-bulk`, `submit-youtube`) and a separate `senior_design_cli.py` entrypoint. Those have been retired in favor of the dashboard, which now handles single or bulk YouTube conversion, single or bulk diarization selection, backend choice from the UI, and backend-specific fine-tuning preparation. The retired scripts and their output folders live under `archive/cleanup_2026-05-01/` for reference.
