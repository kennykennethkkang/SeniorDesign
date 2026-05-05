import io
import json
import pathlib
import tempfile
import time
import unittest
import wave
from contextlib import redirect_stderr

import fine_tuning_manager as fine_tuning


def wav_bytes(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    payload = b"\x00\x00" * frame_count
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)
    return buffer.getvalue()


class FineTuningTests(unittest.TestCase):
    def assert_default_slurm_resources(self, sbatch_path: pathlib.Path):
        sbatch_text = sbatch_path.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=8", sbatch_text)
        self.assertIn("#SBATCH --mem=48G", sbatch_text)
        self.assertIn("#SBATCH --time=08:00:00", sbatch_text)
        self.assertIn("#SBATCH --gres=gpu:1", sbatch_text)

    def test_prepare_project_generates_session_and_msdd_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            fine_tuning.save_project_sample(
                project_name="Demo Project",
                audio_name="a_duo.wav",
                audio_bytes=wav_bytes(4.0),
                rttm_name="a_duo.rttm",
                rttm_bytes=(
                    b"SPEAKER duo 1 0.000 1.200 <NA> <NA> speaker_0 <NA> <NA>\n"
                    b"SPEAKER duo 1 1.300 1.100 <NA> <NA> speaker_1 <NA> <NA>\n"
                ),
                transcript_text="first sample",
                root=root,
            )
            fine_tuning.save_project_sample(
                project_name="Demo Project",
                audio_name="b_multi.wav",
                audio_bytes=wav_bytes(6.0),
                rttm_name="b_multi.rttm",
                rttm_bytes=(
                    b"SPEAKER multi 1 0.000 1.000 <NA> <NA> speaker_a <NA> <NA>\n"
                    b"SPEAKER multi 1 1.100 1.000 <NA> <NA> speaker_b <NA> <NA>\n"
                    b"SPEAKER multi 1 2.250 1.000 <NA> <NA> speaker_c <NA> <NA>\n"
                ),
                transcript_text="second sample",
                root=root,
            )

            artifacts = fine_tuning.prepare_project(
                project_name="Demo Project",
                train_ratio=0.5,
                root=root,
            )

            self.assertTrue(artifacts.session_manifest_train.exists())
            self.assertTrue(artifacts.session_manifest_validation.exists())
            self.assertTrue(artifacts.msdd_manifest_train.exists())
            self.assertTrue(artifacts.msdd_manifest_validation.exists())
            self.assertTrue(artifacts.launch_script_path.exists())
            self.assertTrue(artifacts.sbatch_script_path.exists())
            self.assert_default_slurm_resources(artifacts.sbatch_script_path)

            train_session_rows = artifacts.session_manifest_train.read_text(encoding="utf-8").strip().splitlines()
            validation_session_rows = artifacts.session_manifest_validation.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(train_session_rows), 1)
            self.assertEqual(len(validation_session_rows), 1)

            train_msdd_rows = [json.loads(line) for line in artifacts.msdd_manifest_train.read_text(encoding="utf-8").splitlines()]
            validation_msdd_rows = [json.loads(line) for line in artifacts.msdd_manifest_validation.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(train_msdd_rows), 1)
            self.assertGreaterEqual(len(validation_msdd_rows), 1)

            pairwise_files = sorted(
                path.name
                for path in (root / "fine_tuning" / "projects" / "nemo" / "demo-project" / "artifacts" / "pairwise_rttm").rglob("*.rttm")
            )
            self.assertIn("b_multi.speaker_a_speaker_b.rttm", pairwise_files)
            self.assertIn("b_multi.speaker_a_speaker_c.rttm", pairwise_files)
            self.assertIn("b_multi.speaker_b_speaker_c.rttm", pairwise_files)

    def test_prepare_project_generates_pyannote_training_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            fine_tuning.save_project_sample(
                project_name="Pyannote Project",
                backend="pyannote",
                audio_name="sample.wav",
                audio_bytes=wav_bytes(4.0),
                rttm_name="sample.rttm",
                rttm_bytes=(
                    b"SPEAKER sample 1 0.000 1.500 <NA> <NA> speaker_0 <NA> <NA>\n"
                    b"SPEAKER sample 1 1.700 1.100 <NA> <NA> speaker_1 <NA> <NA>\n"
                ),
                transcript_text="pyannote sample",
                root=root,
            )

            artifacts = fine_tuning.prepare_project(
                project_name="Pyannote Project",
                backend="pyannote",
                root=root,
            )

            self.assertEqual(artifacts.backend, "pyannote")
            self.assertTrue((artifacts.project_dir / "artifacts" / "database.yml").exists())
            self.assertTrue((artifacts.project_dir / "artifacts" / "lists" / "train.lst").exists())
            self.assertTrue((artifacts.project_dir / "artifacts" / "train_pyannote.py").exists())
            self.assertTrue(artifacts.launch_script_path.exists())
            self.assertTrue(artifacts.sbatch_script_path.exists())
            self.assert_default_slurm_resources(artifacts.sbatch_script_path)

    def test_launch_training_runs_generated_script_in_background(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            fine_tuning.save_project_sample(
                project_name="Launch Project",
                audio_name="launch.wav",
                audio_bytes=wav_bytes(3.0),
                rttm_name="launch.rttm",
                rttm_bytes=(
                    b"SPEAKER launch 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>\n"
                    b"SPEAKER launch 1 1.200 1.000 <NA> <NA> speaker_1 <NA> <NA>\n"
                ),
                transcript_text="launch sample",
                root=root,
            )

            fake_nemo_root = root / "NeMo"
            neural_dir = fake_nemo_root / "examples" / "speaker_tasks" / "diarization" / "neural_diarizer"
            neural_dir.mkdir(parents=True)
            (fake_nemo_root / "examples" / "speaker_tasks" / "diarization" / "conf" / "neural_diarizer").mkdir(parents=True)
            (neural_dir / "multiscale_diar_decoder.py").write_text(
                "import sys\n"
                "print('fake train invoked')\n"
                "print('ARGS=' + ' '.join(sys.argv[1:]))\n",
                encoding="utf-8",
            )

            fine_tuning.prepare_project(
                project_name="Launch Project",
                root=root,
                nemo_root=fake_nemo_root,
            )

            run = fine_tuning.launch_training(
                project_name="Launch Project",
                nemo_root=fake_nemo_root,
                prefer_sbatch=False,
                root=root,
            )

            for _ in range(40):
                if run.exit_code_path.exists():
                    break
                time.sleep(0.05)

            self.assertTrue(run.exit_code_path.exists(), "training run did not finish in time")
            self.assertEqual(run.exit_code_path.read_text(encoding="utf-8").strip(), "0")
            self.assertEqual(fine_tuning.run_status(run.run_dir), "succeeded")
            stdout_text = run.stdout_path.read_text(encoding="utf-8")
            self.assertIn("fake train invoked", stdout_text)
            self.assertIn("--config-name", stdout_text)
            self.assertEqual(run.version_name, "Launch Project trained version 1")
            self.assertEqual(run.version_number, 1)
            self.assertTrue(run.experiment_dir.exists())
            self.assertIn("launch-project-trained-version-1", stdout_text)

            named_run = fine_tuning.launch_training(
                project_name="Launch Project",
                nemo_root=fake_nemo_root,
                prefer_sbatch=False,
                version_name="Clean Speaker Version",
                root=root,
            )
            for _ in range(40):
                if named_run.exit_code_path.exists():
                    break
                time.sleep(0.05)

            self.assertEqual(named_run.version_name, "Clean Speaker Version")
            self.assertEqual(named_run.version_number, 2)
            self.assertIn("clean-speaker-version", named_run.run_dir.name)
            metadata = json.loads(named_run.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["version_name"], "Clean Speaker Version")
            self.assertEqual(metadata["version_number"], 2)
            self.assertTrue(pathlib.Path(metadata["experiment_dir"]).name.startswith("clean-speaker-version"))

    def test_main_returns_clean_error_for_expected_runtime_failures(self):
        stderr = io.StringIO()

        def failing_launch_training(**_kwargs):
            raise FileNotFoundError("Run prepare first.")

        original = fine_tuning.launch_training
        fine_tuning.launch_training = failing_launch_training
        try:
            with redirect_stderr(stderr):
                exit_code = fine_tuning.main(["launch", "--project", "demo"])
        finally:
            fine_tuning.launch_training = original

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Run prepare first.", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
