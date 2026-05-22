# Test Cases

All automated test cases for this project live in this folder.

## What Each File Covers

- `test_workflow_dashboard.py`: local website routes, React state, uploads, YouTube conversion actions, diarization Slurm submission, fine-tuning routes, and asset serving.
- `test_workflow_cli.py`: command construction, local dashboard startup checks, and PID/status handling.
- `test_fine_tuning_manager.py`: NeMo and pyannote fine-tuning sample upload, artifact preparation, and launch behavior.
- `test_integration_pipeline.py`: end-to-end behavior between audio numbering and YouTube conversion indexing.
- `test_project_utilities.py`: audio numbering helpers and YouTube conversion utility behavior.
- `test_review_bundle.py`: review HTML and flag report generation from SRT files.
- `test_run_diarization.py`: diarization CLI behavior that does not require GPU/runtime dependencies.
- `test_diarization_metrics.py`: DER and companion metrics math (missed speech, false alarm, speaker confusion, JER).
- `test_workflow_background.py`: local and Slurm background-run status helpers.
- `test_cluster_queue.py`: the cluster-wide `squeue` snapshot helper and its HTTP endpoint.
- `test_file_index.py`: the cross-run file index used by the dashboard.
- `test_runtime_history.py`: the runtime-history time-estimate engine.

## Run Everything

From the project root:

```bash
bash tests/run_all_tests.sh
```

This runs Python syntax checks, frontend JavaScript syntax checks, Slurm script syntax checks, and the full Python unit test suite.

## Run Only Unit Tests

```bash
python3 -m unittest discover -s tests -v
```

These tests use temporary folders and mocks, so they do not download YouTube videos, run real Slurm jobs, or require GPU diarization packages.
