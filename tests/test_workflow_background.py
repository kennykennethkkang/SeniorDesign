"""Tests for local/Slurm background-run status helpers."""
from __future__ import annotations

import subprocess
import unittest
from unittest import mock

import workflow_background


def _completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(args=["slurm"], returncode=returncode, stdout=stdout, stderr=stderr)


class SlurmQueueSnapshotTests(unittest.TestCase):
    def test_slurm_queue_snapshot_returns_state_and_resources_for_our_job(self):
        # squeue is now invoked with -j <jobid>, so the response contains only
        # our row. The snapshot intentionally drops cluster-wide queue position
        # fields; we don't track other people's jobs anymore.
        squeue_stdout = (
            "123|PD|Priority|2026-05-04T10:05:00|gpu|site_diarization|kkang|0:00|2:00:00|1|6|gpu:1\n"
        )
        with mock.patch.object(workflow_background.shutil, "which", return_value="/usr/bin/squeue"), mock.patch.object(
            workflow_background.subprocess,
            "run",
            return_value=_completed(stdout=squeue_stdout),
        ) as run_mock:
            snapshot = workflow_background.slurm_queue_snapshot("123.batch")

        # The key cost-saver: -j <jobid> means squeue only walks our row.
        called_args = run_mock.call_args[0][0]
        self.assertIn("-j", called_args)
        self.assertIn("123", called_args)

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["job_id"], "123")
        self.assertEqual(snapshot["state"], "PD")
        self.assertEqual(snapshot["partition"], "gpu")
        self.assertEqual(snapshot["cpus"], "6")
        self.assertEqual(snapshot["gres"], "gpu:1")
        self.assertEqual(snapshot["state_source"], "squeue")
        # Cluster-wide fields are no longer reported.
        self.assertNotIn("queue_position", snapshot)
        self.assertNotIn("jobs_ahead", snapshot)

    def test_slurm_queue_snapshot_uses_sacct_after_job_leaves_squeue(self):
        sacct_stdout = (
            "123|COMPLETED|0:0|00:01:20|02:00:00|2026-05-04T09:55:00|"
            "2026-05-04T10:00:00|2026-05-04T10:01:20|node07|1|6|gpu|site_diarization\n"
        )

        def fake_which(name):
            return f"/usr/bin/{name}" if name in {"squeue", "sacct"} else None

        def fake_run(command, **_kwargs):
            if command[0] == "squeue":
                return _completed(stdout="")
            if command[0] == "sacct":
                return _completed(stdout=sacct_stdout)
            raise AssertionError(command)

        with mock.patch.object(workflow_background.shutil, "which", side_effect=fake_which), mock.patch.object(
            workflow_background.subprocess,
            "run",
            side_effect=fake_run,
        ):
            snapshot = workflow_background.slurm_queue_snapshot("123")

        self.assertEqual(snapshot["state"], "COMPLETED")
        self.assertEqual(snapshot["state_source"], "sacct")
        self.assertEqual(snapshot["exit_code"], "0:0")
        self.assertEqual(snapshot["time_used"], "00:01:20")
        self.assertEqual(snapshot["partition"], "gpu")
        self.assertIn("left squeue", snapshot["message"])


if __name__ == "__main__":
    unittest.main()
