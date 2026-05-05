# Fine-Tuning Training Runbook

This runbook is written for the current SeniorDesign workspace. The key point is
that training cannot start from the `.srt` diarization outputs alone. The
fine-tuning code needs verified audio + RTTM pairs.

## Current Workspace State

The dashboard is available at:

```text
http://127.0.0.1:8000
```

Current source audio files in `audio_in/`:

```text
001_test.wav
002_Kids Give Adults Advice on Dating_B08UvVwjpG4.wav
003_Parents & Kids Play Truth or Drink ｜ Truth or Drink ｜ Cut_iIaYJqKBK44.wav
004_Seniors vs Children ｜ Middle Ground_CxL1b7nEFxA.wav
005_Adorable moment of father and toddler having conversation ｜ ABC News_Yn8j4XRxSck.wav
006_Babies Arguing With Their Parents ｜ Hilarious Baby Compilation_T97wA5oW8As.wav
```

Current useful review outputs are under:

```text
outputs/diarization_runs/pyannote/20260427T170608_diarization_pyannote_06-items/
```

That run currently has review pages, transcripts, and SRT files for `002` through
`006`. There are no fine-tuning projects yet, and there are no RTTM files yet.

## Recommended Project Names

Use one shared project name for both backends:

```text
conversation-diarization-v1
```

When you mark a label complete with backend `NeMo + pyannote`, the site creates
both of these project folders:

```text
fine_tuning/projects/nemo/conversation-diarization-v1/
fine_tuning/projects/pyannote/conversation-diarization-v1/
```

Use these trained version names:

```text
conversation-pyannote-v1
conversation-nemo-v1
```

## Step 1: Make Training Labels

Open the site and go to `Training Labels`.

For each useful audio file, use:

```text
Backend: NeMo + pyannote
Fine-tuning project: conversation-diarization-v1
```

Start with these files because they have recent review pages:

```text
002_Kids Give Adults Advice on Dating_B08UvVwjpG4.wav
003_Parents & Kids Play Truth or Drink ｜ Truth or Drink ｜ Cut_iIaYJqKBK44.wav
004_Seniors vs Children ｜ Middle Ground_CxL1b7nEFxA.wav
005_Adorable moment of father and toddler having conversation ｜ ABC News_Yn8j4XRxSck.wav
006_Babies Arguing With Their Parents ｜ Hilarious Baby Compilation_T97wA5oW8As.wav
```

`001_test.wav` can be labeled later if it is useful training data. Do not include
it just to increase the count.

In `Speaker-time labels`, enter one segment per line:

```text
start end speaker
```

Use stable labels with no spaces:

```text
00:00:00.132 00:00:01.160 SPEAKER_00
00:00:01.980 00:00:02.080 SPEAKER_00
00:00:19.400 00:00:22.580 SPEAKER_02
```

The parser accepts seconds, `MM:SS`, or `HH:MM:SS`. It also accepts pasted RTTM
rows. Do not paste a full SRT file directly; convert the relevant speaker turns
into the simple rows above.

For each file:

1. Open its review page or audio playback.
2. Correct speaker labels so the same real person keeps the same label within
   that file.
3. Remove obvious bad rows: music-only, silence, hallucinated speech, duplicate
   tiny fragments that do not represent a real turn.
4. Keep overlaps when two people really talk at the same time. Overlap helps the
   training data if it is labeled correctly.
5. Click `Mark Complete`.

Expected created files after each completed label:

```text
fine_tuning/label_work/<audio-stem>.rttm
fine_tuning/projects/nemo/conversation-diarization-v1/audio/<audio>.wav
fine_tuning/projects/nemo/conversation-diarization-v1/rttm/<audio-stem>.rttm
fine_tuning/projects/pyannote/conversation-diarization-v1/audio/<audio>.wav
fine_tuning/projects/pyannote/conversation-diarization-v1/rttm/<audio-stem>.rttm
```

## Step 2: Confirm Dataset Metrics

Go to `Fine-Tuning`.

You should see two project cards:

```text
NeMo / conversation-diarization-v1
pyannote / conversation-diarization-v1
```

Check these values before preparing:

```text
Samples: should match the number of completed audio labels
Reference speech: should be more than a few seconds
Active speech coverage: should not be 0%
Overlap: useful if your target audio has interruption or crosstalk
Speaker labels: should be more than 1 for diarization training
Max concurrent speakers: use this to set pyannote max speakers/frame
```

For a first training attempt, use at least 5 labeled samples. For a better
project, add more recordings and label enough speech that the validation split
contains speakers and acoustic conditions similar to the training split.

## Step 3: Prepare pyannote First

pyannote is the best first target in this workspace because the dashboard has a
Hugging Face token saved locally. NeMo can also work, but only after a real NeMo
checkout exists.

On `Fine-Tuning`, in `2. Prepare Artifacts`, use:

```text
Project with uploaded samples: pyannote / conversation-diarization-v1
Backend: pyannote
Project name: conversation-diarization-v1
Train ratio: 0.8
Devices: 1
pyannote pretrained model: pyannote/segmentation-3.0
pyannote chunk duration: 10
Max speakers per chunk: 3
Max speakers per frame: 2
Max epochs: 5
Slurm partition: gpu
Slurm time: 08:00:00
Slurm memory: 48G
Slurm CPUs: 8
Slurm GPUs: 1
```

If the Fine-Tuning card later shows `Max concurrent speakers` greater than `2`,
increase `Max speakers per frame` to that value before preparing again.

Click:

```text
Prepare Training Artifacts
```

Expected pyannote artifacts:

```text
fine_tuning/projects/pyannote/conversation-diarization-v1/artifacts/database.yml
fine_tuning/projects/pyannote/conversation-diarization-v1/artifacts/lists/train.lst
fine_tuning/projects/pyannote/conversation-diarization-v1/artifacts/lists/development.lst
fine_tuning/projects/pyannote/conversation-diarization-v1/artifacts/rttm/train.rttm
fine_tuning/projects/pyannote/conversation-diarization-v1/artifacts/uem/train.uem
fine_tuning/projects/pyannote/conversation-diarization-v1/artifacts/train_pyannote.py
fine_tuning/projects/pyannote/conversation-diarization-v1/artifacts/launch_pyannote_finetune.sh
fine_tuning/projects/pyannote/conversation-diarization-v1/artifacts/launch_pyannote_finetune.sbatch
```

CLI equivalent:

```bash
python3 fine_tuning_manager.py prepare \
  --backend pyannote \
  --project conversation-diarization-v1 \
  --train-ratio 0.8 \
  --devices 1 \
  --max-epochs 5 \
  --pyannote-pretrained-model pyannote/segmentation-3.0 \
  --pyannote-duration 10 \
  --pyannote-max-speakers-per-chunk 3 \
  --pyannote-max-speakers-per-frame 2 \
  --slurm-partition gpu \
  --slurm-time 08:00:00 \
  --slurm-memory 48G \
  --slurm-cpus 8 \
  --slurm-gpus 1
```

## Step 4: Launch pyannote Training

On `Fine-Tuning`, in `3. Launch Training`, use:

```text
Prepared project: pyannote / conversation-diarization-v1
Backend: pyannote
Prepared project name: conversation-diarization-v1
Trained version name: conversation-pyannote-v1
Python binary: python3
Force local launch instead of Slurm submission: unchecked
```

Click:

```text
Launch Background Training Run
```

Expected run folder:

```text
fine_tuning/projects/pyannote/conversation-diarization-v1/runs/<timestamp>_conversation-pyannote-v1/
```

Watch these links on the project card:

```text
stdout.log
stderr.log
metadata.json
```

If Slurm accepted it, the latest run will show `submitted`. If it runs locally,
it will show a process id. When it finishes successfully, it should show
`succeeded`.

CLI equivalent:

```bash
python3 fine_tuning_manager.py launch \
  --backend pyannote \
  --project conversation-diarization-v1 \
  --version-name conversation-pyannote-v1 \
  --python-bin python3
```

Use `--local` only for a small smoke test:

```bash
python3 fine_tuning_manager.py launch \
  --backend pyannote \
  --project conversation-diarization-v1 \
  --version-name conversation-pyannote-v1-local-smoke \
  --python-bin python3 \
  --local
```

## Step 5: Use the pyannote Trained Model

After training produces a reusable artifact under the project `artifacts/experiments/`
or the specific run experiment directory, the `Diarization` page model selector
should include an option like:

```text
pyannote fine-tuned / conversation-diarization-v1 / conversation-pyannote-v1
```

Select that model and run diarization on a held-out file. Do not evaluate on the
same audio you trained on unless you are only checking that the model loads.

Good first hold-out choices:

```text
001_test.wav
```

or any newly uploaded audio that was not included in the completed labels.

## Step 6: Prepare NeMo Only After NeMo Exists

This workspace currently does not show a full NeMo source checkout under the
project root. NeMo training expects a checkout with:

```text
examples/speaker_tasks/diarization/neural_diarizer/
examples/speaker_tasks/diarization/conf/neural_diarizer/
```

Once that exists, use this `NEMO_ROOT` value in the site:

```text
/path/to/NeMo
```

For example, if you clone it at `/WAVE/users2/unix/kkang/NeMo`, use:

```text
/WAVE/users2/unix/kkang/NeMo
```

On `Fine-Tuning`, in `2. Prepare Artifacts`, use:

```text
Project with uploaded samples: NeMo / conversation-diarization-v1
Backend: NeMo
Project name: conversation-diarization-v1
Train ratio: 0.8
Devices: 1
NeMo base window: 0.5
NeMo base shift: 0.25
NeMo step count: 50
NeMo config name: msdd_5scl_15_05_50Povl_256x3x32x2.yaml
NeMo speaker model: titanet_large
Max epochs: 20
Optional NeMo root: /path/to/NeMo
Slurm partition: gpu
Slurm time: 08:00:00
Slurm memory: 48G
Slurm CPUs: 8
Slurm GPUs: 1
```

Click `Prepare Training Artifacts`.

Expected NeMo artifacts:

```text
fine_tuning/projects/nemo/conversation-diarization-v1/artifacts/manifests/train_session_manifest.jsonl
fine_tuning/projects/nemo/conversation-diarization-v1/artifacts/manifests/validation_session_manifest.jsonl
fine_tuning/projects/nemo/conversation-diarization-v1/artifacts/manifests/train_msdd_manifest.jsonl
fine_tuning/projects/nemo/conversation-diarization-v1/artifacts/manifests/validation_msdd_manifest.jsonl
fine_tuning/projects/nemo/conversation-diarization-v1/artifacts/pairwise_rttm/
fine_tuning/projects/nemo/conversation-diarization-v1/artifacts/launch_nemo_finetune.sh
fine_tuning/projects/nemo/conversation-diarization-v1/artifacts/launch_nemo_finetune.sbatch
```

CLI equivalent:

```bash
python3 fine_tuning_manager.py prepare \
  --backend nemo \
  --project conversation-diarization-v1 \
  --train-ratio 0.8 \
  --base-window 0.5 \
  --base-shift 0.25 \
  --step-count 50 \
  --config-name msdd_5scl_15_05_50Povl_256x3x32x2.yaml \
  --speaker-model titanet_large \
  --devices 1 \
  --max-epochs 20 \
  --nemo-root /path/to/NeMo \
  --slurm-partition gpu \
  --slurm-time 08:00:00 \
  --slurm-memory 48G \
  --slurm-cpus 8 \
  --slurm-gpus 1
```

Then launch:

```bash
python3 fine_tuning_manager.py launch \
  --backend nemo \
  --project conversation-diarization-v1 \
  --version-name conversation-nemo-v1 \
  --nemo-root /path/to/NeMo \
  --python-bin python3
```

## What To Calculate And Report

The site now calculates these automatically on the Fine-Tuning page:

```text
Total audio seconds
Reference speech seconds
Active speech seconds
Non-speech seconds
Overlap seconds
Speech coverage = active speech / total audio
Overlap coverage = overlap / active speech
Speaker turns per minute = segment count / audio minutes
Max concurrent speakers
Train/validation split counts
DER = (missed speech + false alarm + speaker confusion) / reference speech
Absolute DER reduction = baseline DER - fine-tuned DER
Relative DER reduction = (baseline DER - fine-tuned DER) / baseline DER
```

Use `Reference speech seconds` as the DER denominator.

## Minimum Practical Training Checklist

Before launching a real run, confirm:

```text
At least 5 completed samples
At least 2 real speakers in most samples
Validation count is not 0
No unresolved Training Labels questions
No missing RTTM files
HF_TOKEN or HUGGINGFACE_HUB_TOKEN is available for pyannote
NEMO_ROOT points to a real NeMo checkout for NeMo
```

## Useful References

- NeMo speaker diarization configuration and MSDD training docs:
  <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html>
- NeMo diarization dataset notes:
  <https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/asr/speaker_diarization/datasets.html>
- pyannote model card:
  <https://huggingface.co/pyannote/speaker-diarization-community-1>
- pyannote metrics DER reference:
  <https://pyannote.github.io/pyannote-metrics/reference.html>
- Hugging Face diarizers examples:
  <https://github.com/huggingface/diarizers>
