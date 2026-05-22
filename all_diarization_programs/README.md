# All Diarization Programs

This directory contains the direct diarization entrypoints and backend-specific logic used by the SeniorDesign workflow.

## Main files

- `run_diarization.py`: direct single-audio runner
- `run_parallel_diarization.py`: parallel variant used for wider batch-style execution
- `nemo_msdd_backend.py`: NeMo MSDD backend integration
- `nemo_diarization_telephonic.yaml`: NeMo diarizer config used by the MSDD backend
- `pyannote_callhome_backend.py`: pyannote-based backend integration
- `diarization_helpers.py`: output formatting and helper functions
- `constraints.txt`: dependency constraints for the batch runtime
- `requirements_common.txt`: shared runtime requirements
- `requirements_nemo.txt`: NeMo-specific runtime requirements
- `requirements_pyannote.txt`: pyannote-specific runtime requirements

## When to use this folder directly

Use the direct runner when you want:

- a fast local experiment
- a backend comparison on one file
- a debugging path that bypasses the Slurm wrappers

Use the web site or `workflow_cli.py` when you want the guided project workflow.

## Step-by-step direct run

From the project root:

```bash
cd /WAVE/users2/unix/kkang/SeniorDesign
```

### 1. Confirm which audio files are available

```bash
python3 all_diarization_programs/run_diarization.py --list-audio
```

### 2. Run one file through the default direct pipeline

```bash
python3 all_diarization_programs/run_diarization.py \
  --audio 001_example.wav \
  --input-dir ./audio_in \
  --output-dir ./job_outputs/manual_run
```

### 3. Choose a backend explicitly when needed

NeMo:

```bash
python3 all_diarization_programs/run_diarization.py \
  --audio 001_example.wav \
  --input-dir ./audio_in \
  --output-dir ./job_outputs/manual_run \
  --diarizer nemo
```

Pyannote:

```bash
python3 all_diarization_programs/run_diarization.py \
  --audio 001_example.wav \
  --input-dir ./audio_in \
  --output-dir ./job_outputs/manual_run \
  --diarizer pyannote
```

## Useful options

- `--list-audio`: print the current audio inventory and exit
- `--stem`: enable source separation before transcription
- `--no-stem`: skip source separation
- `--generate-review`: write review HTML and TSV files
- `--no-generate-review`: skip review generation
- `--device auto|cpu|cuda|cuda:0`
- `--whisper-model <name>`
- `--batch-size <n>`
- `--language <code>`
- `--hf-token <token>`: useful for pyannote if `HF_TOKEN` is not already exported

## Outputs from a direct run

The direct runner writes:

- `<name>.txt`
- `<name>.srt`
- `<name>_review.html`
- `<name>_review_flags.tsv`

Default output root for the examples above:

```text
job_outputs/manual_run/
```

## Backend notes

### NeMo

Use `nemo` when you want the project's existing NeMo MSDD diarization path.

### Pyannote

Current defaults in this workflow:

- pipeline: `pyannote/speaker-diarization-3.1`
- segmentation model: `diarizers-community/speaker-segmentation-fine-tuned-callhome-zho`

Pyannote requires:

- a Hugging Face token
- access to the gated pyannote pipeline model

The CallHome ZHO segmentation model is public, and the runner downloads its
`config.json` and `model.safetensors` through the Hugging Face cache. Leave the
segmentation model setting blank only when you want the selected pyannote
pipeline's built-in segmentation model.

Set:

```bash
export HF_TOKEN=...
```

## Relationship to the rest of the repo

- the dashboard's `Diarization` page builds its command from this runner
- `workflow_cli.py` handles the broader project workflows
- Slurm batch scripts prepare runtime environments and use the same backend code paths
