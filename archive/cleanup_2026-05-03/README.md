# Cleanup snapshot — 2026-05-03

This batch removed ten legacy symlinks plus one redundant shim module. No
caller in the repo (Python, sbatch, shell, Markdown, or test) referenced any
of them. The canonical files they pointed at are unchanged and still in their
original locations.

## Top-level symlinks removed

| Old name                       | Pointed to                |
|--------------------------------|---------------------------|
| `fine_tuning.py`               | `fine_tuning_manager.py`  |
| `number_audio_in_files.py`     | `audio_numbering.py`      |
| `review_outputs.py`            | `review_bundle.py`        |
| `workflow_web.py`              | `workflow_dashboard.py`   |
| `youtube_to_audio_batch.py`    | `youtube_audio_batch.py`  |

## `all_diarization_programs/` symlinks removed

| Old name                                       | Pointed to                              |
|------------------------------------------------|-----------------------------------------|
| `diarize.py`                                   | `run_diarization.py`                    |
| `diarize_parallel.py`                          | `run_parallel_diarization.py`           |
| `helpers.py`                                   | `diarization_helpers.py`                |
| `nemo_msdd.py`                                 | `nemo_msdd_backend.py`                  |
| `nemo_diar_infer_telephonic.yaml`              | `nemo_diarization_telephonic.yaml`      |

## `all_diarization_programs/pyannote_callhome.py` removed

A 4-line shim that re-exported `PyannoteCallhomeDiarizer` from
`pyannote_callhome_backend.py`. The package `__init__.py` already exposes
`PyannoteCallhomeDiarizer` via `__getattr__`, so the shim was redundant and
unreferenced anywhere in the repo.

## How it was verified safe to remove

```
# top-level: zero hits in code
grep -rn "fine_tuning.py\|number_audio_in_files.py\|review_outputs.py\|\
workflow_web.py\|youtube_to_audio_batch.py" \
  --exclude-dir=archive --exclude-dir=__pycache__ --exclude-dir=.git \
  --include="*.py" --include="*.sh" --include="*.sbatch"

# dashboard sbatch scripts only invoke workflow_cli.py
# workflow_cli.py references the canonical youtube_audio_batch.py
# tests reference only canonical names
```

If something turns out to need an old name, restore the symlink:

```
ln -s fine_tuning_manager.py fine_tuning.py
```

## Restore one removed symlink at any time

`git show 12d9acf:fine_tuning.py` (initial commit) shows the original target.
