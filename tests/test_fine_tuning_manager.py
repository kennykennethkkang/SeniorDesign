import io
import json
import pathlib
import struct
import tempfile
import time
import unittest
import wave
from contextlib import redirect_stderr

import fine_tuning_manager as fine_tuning
from dashboard import auto_train


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


def float_wav_bytes(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    data_size = frame_count * 4
    riff_size = 4 + (8 + 16) + (8 + data_size)
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 3, 1, sample_rate, sample_rate * 4, 4, 32),
            b"data",
            struct.pack("<I", data_size),
            b"\x00" * data_size,
        ]
    )


class FineTuningTests(unittest.TestCase):
    def assert_default_slurm_resources(self, sbatch_path: pathlib.Path):
        sbatch_text = sbatch_path.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=8", sbatch_text)
        self.assertIn("#SBATCH --mem=48G", sbatch_text)
        self.assertIn("#SBATCH --time=08:00:00", sbatch_text)
        self.assertIn("#SBATCH --gres=gpu:1", sbatch_text)

    def test_prepare_project_accepts_ieee_float_wav_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            sample = fine_tuning.save_project_sample(
                project_name="Float WAV Project",
                audio_name="float_clip.wav",
                audio_bytes=float_wav_bytes(2.5),
                rttm_name="float_clip.rttm",
                rttm_bytes=(
                    b"SPEAKER float_clip 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>\n"
                    b"SPEAKER float_clip 1 1.100 0.800 <NA> <NA> speaker_1 <NA> <NA>\n"
                ),
                transcript_text="float wav sample",
                root=root,
            )

            self.assertAlmostEqual(sample.duration_seconds, 2.5, places=3)
            artifacts = fine_tuning.prepare_project(
                project_name="Float WAV Project",
                root=root,
            )

            self.assertEqual(artifacts.sample_count, 1)
            projects = fine_tuning.list_projects(root=root)
            self.assertEqual(projects[0]["sample_count"], 1)

    def test_save_project_sample_canonicalizes_uploaded_rttm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            fine_tuning.save_project_sample(
                project_name="Canonical RTTM",
                backend="nemo",
                audio_name="canonical.wav",
                audio_bytes=wav_bytes(3.0),
                rttm_name="mismatched_name.rttm",
                rttm_bytes=(
                    b"# comment is not part of the training RTTM\n"
                    b"SPEAKER wrong_session 7 0 1 <NA> <NA> Speaker_A confidence extra\n"
                    b"SPEAKER wrong_session 7 1.2 0.8 <NA> <NA> Speaker_B confidence extra\n"
                ),
                root=root,
            )

            rttm_path = (
                root
                / "fine_tuning"
                / "projects"
                / "nemo"
                / "canonical-rttm"
                / "rttm"
                / "canonical.rttm"
            )
            self.assertEqual(
                rttm_path.read_text(encoding="utf-8").splitlines(),
                [
                    "SPEAKER canonical 1 0.000 1.000 <NA> <NA> Speaker_A <NA> <NA>",
                    "SPEAKER canonical 1 1.200 0.800 <NA> <NA> Speaker_B <NA> <NA>",
                ],
            )

    def test_save_project_sample_streams_canonicalizes_uploaded_rttm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            fine_tuning.save_project_sample_streams(
                project_name="Stream RTTM",
                backend="pyannote",
                audio_name="stream_clip.wav",
                audio_stream=io.BytesIO(wav_bytes(3.0)),
                rttm_name="other.rttm",
                rttm_stream=io.BytesIO(
                    b"SPEAKER other 1 0.000 1.000 <NA> <NA> Speaker_A <NA> <NA>\n"
                    b"SPEAKER other 1 1.250 0.500 <NA> <NA> Speaker_B <NA> <NA>\n"
                ),
                root=root,
            )

            rttm_path = (
                root
                / "fine_tuning"
                / "projects"
                / "pyannote"
                / "stream-rttm"
                / "rttm"
                / "stream_clip.rttm"
            )
            self.assertEqual(
                rttm_path.read_text(encoding="utf-8").splitlines(),
                [
                    "SPEAKER stream_clip 1 0.000 1.000 <NA> <NA> Speaker_A <NA> <NA>",
                    "SPEAKER stream_clip 1 1.250 0.500 <NA> <NA> Speaker_B <NA> <NA>",
                ],
            )

    def test_prepare_project_accepts_legacy_default_backend_project_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            project_dir = root / "fine_tuning" / "projects" / "legacy-lab"
            (project_dir / "audio").mkdir(parents=True)
            (project_dir / "rttm").mkdir()
            (project_dir / "text").mkdir()
            (project_dir / "audio" / "legacy.wav").write_bytes(wav_bytes(3.0))
            (project_dir / "rttm" / "legacy.rttm").write_text(
                "SPEAKER legacy 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>\n"
                "SPEAKER legacy 1 1.200 1.000 <NA> <NA> speaker_1 <NA> <NA>\n",
                encoding="utf-8",
            )

            artifacts = fine_tuning.prepare_project(project_name="legacy-lab", root=root)

            self.assertEqual(artifacts.project_dir, project_dir)
            self.assertEqual(artifacts.sample_count, 1)
            self.assertTrue((project_dir / "artifacts" / "metadata.json").is_file())
            self.assertFalse((root / "fine_tuning" / "projects" / "nemo" / "legacy-lab").exists())

    def test_prepare_project_warns_and_skips_samples_missing_rttm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            project_dir = root / "fine_tuning" / "projects" / "nemo" / "partial-lab"
            (project_dir / "audio").mkdir(parents=True)
            (project_dir / "rttm").mkdir()
            (project_dir / "text").mkdir()
            (project_dir / "audio" / "valid.wav").write_bytes(wav_bytes(3.0))
            (project_dir / "audio" / "missing.wav").write_bytes(wav_bytes(3.0))
            (project_dir / "rttm" / "valid.rttm").write_text(
                "SPEAKER valid 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>\n"
                "SPEAKER valid 1 1.200 1.000 <NA> <NA> speaker_1 <NA> <NA>\n",
                encoding="utf-8",
            )

            artifacts = fine_tuning.prepare_project(
                project_name="partial-lab",
                backend="nemo",
                root=root,
            )

            self.assertEqual(artifacts.sample_count, 1)
            self.assertEqual(len(artifacts.warnings), 1)
            self.assertIn("Missing RTTM for training sample 'missing'", artifacts.warnings[0])

    def test_prepare_project_splits_nemo_msdd_on_available_multispeaker_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            fine_tuning.save_project_sample(
                project_name="Sparse Multispeaker",
                backend="nemo",
                audio_name="a_single.wav",
                audio_bytes=wav_bytes(3.0),
                rttm_name="a_single.rttm",
                rttm_bytes=b"SPEAKER a_single 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>\n",
                root=root,
            )
            fine_tuning.save_project_sample(
                project_name="Sparse Multispeaker",
                backend="nemo",
                audio_name="z_multi.wav",
                audio_bytes=wav_bytes(3.0),
                rttm_name="z_multi.rttm",
                rttm_bytes=(
                    b"SPEAKER z_multi 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>\n"
                    b"SPEAKER z_multi 1 1.200 1.000 <NA> <NA> speaker_1 <NA> <NA>\n"
                ),
                root=root,
            )

            artifacts = fine_tuning.prepare_project(
                project_name="Sparse Multispeaker",
                backend="nemo",
                train_ratio=0.5,
                root=root,
            )

            self.assertEqual(artifacts.sample_count, 2)
            self.assertTrue(artifacts.msdd_manifest_train.read_text(encoding="utf-8").strip())
            self.assertTrue(artifacts.msdd_manifest_validation.read_text(encoding="utf-8").strip())
            self.assertTrue(
                any("single-speaker sample(s)" in warning for warning in artifacts.warnings)
            )

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

            # Same cluster-runtime fixes the diarization sbatch ships need to
            # carry over to NeMo fine-tuning too — without `module load` and
            # `EBROOTGCCCORE/lib64` on LD_LIBRARY_PATH, the venv's _sqlite3.so
            # can't resolve `sqlite3_deserialize`.
            sbatch_text = artifacts.sbatch_script_path.read_text(encoding="utf-8")
            self.assertIn("module load Python/3.12.3-GCCcore-14.2.0", sbatch_text)
            self.assertIn("sbatch_runtime_env.sh", sbatch_text)
            self.assertIn("expose_venv_native_libraries", sbatch_text)
            self.assertIn("/sbatch_runtime/.venv", sbatch_text)

            train_session_rows = artifacts.session_manifest_train.read_text(encoding="utf-8").strip().splitlines()
            validation_session_rows = artifacts.session_manifest_validation.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(train_session_rows), 1)
            self.assertEqual(len(validation_session_rows), 1)

            train_msdd_rows = [json.loads(line) for line in artifacts.msdd_manifest_train.read_text(encoding="utf-8").splitlines()]
            validation_msdd_rows = [json.loads(line) for line in artifacts.msdd_manifest_validation.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(train_msdd_rows), 1)
            self.assertGreaterEqual(len(validation_msdd_rows), 1)
            self.assertEqual(train_msdd_rows[0]["text"], "-")

            pairwise_files = sorted(
                path.name
                for path in (root / "fine_tuning" / "projects" / "nemo" / "demo-project" / "artifacts" / "pairwise_rttm").rglob("*.rttm")
            )
            self.assertIn("b_multi.speaker_a_speaker_b.rttm", pairwise_files)
            self.assertIn("b_multi.speaker_a_speaker_c.rttm", pairwise_files)
            self.assertIn("b_multi.speaker_b_speaker_c.rttm", pairwise_files)

    def test_nemo_launch_script_falls_back_to_detected_nemo_root(self):
        """A fresh prepare with no nemo_root must still embed a usable default.

        We mimic the runbook's `<workspace-parent>/NeMo` layout in the temp
        sandbox so `_detect_default_nemo_root` finds it. If this regresses to
        the old "Set NEMO_ROOT to your NeMo checkout" bail-out, every dashboard
        Launch click breaks again.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            fake_nemo = root.parent / "NeMo"
            (fake_nemo / "examples" / "speaker_tasks" / "diarization" / "neural_diarizer").mkdir(parents=True, exist_ok=True)
            try:
                # Patch the candidate list so the lookup actually finds the
                # tempdir-adjacent NeMo we just stubbed out.
                original_candidates = fine_tuning._NEMO_ROOT_DEFAULT_CANDIDATES
                fine_tuning._NEMO_ROOT_DEFAULT_CANDIDATES = (fake_nemo,)
                fine_tuning.save_project_sample(
                    project_name="Default Nemo Root",
                    audio_name="default.wav",
                    audio_bytes=wav_bytes(4.0),
                    rttm_name="default.rttm",
                    rttm_bytes=(
                        b"SPEAKER default 1 0.000 1.000 <NA> <NA> speaker_a <NA> <NA>\n"
                        b"SPEAKER default 1 1.200 1.000 <NA> <NA> speaker_b <NA> <NA>\n"
                    ),
                    transcript_text="default sample",
                    root=root,
                )
                artifacts = fine_tuning.prepare_project(
                    project_name="Default Nemo Root",
                    root=root,
                )
            finally:
                fine_tuning._NEMO_ROOT_DEFAULT_CANDIDATES = original_candidates

            launch_text = artifacts.launch_script_path.read_text(encoding="utf-8")
            self.assertIn(str(fake_nemo), launch_text)
            self.assertIn("NEMO_ROOT=", launch_text)

            # NeMo v2.x renamed the speaker-embeddings override key. The
            # upstream multiscale_diar_decoder.py docstring still advertises
            # `model.base.diarizer.*`, but the actual config schema exposes
            # `model.diarizer.*` directly — Hydra refuses to override a
            # missing struct, so the old form fails fast with
            # "Key 'base' is not in struct".
            self.assertIn("model.diarizer.speaker_embeddings.model_path", launch_text)
            self.assertNotIn("model.base.diarizer", launch_text)
            # Lightning 2.x auto-picks DDP even on a single GPU, and MSDD's
            # frozen TitaNet has params that don't participate in the loss.
            # Without find_unused_parameters=True, training crashes mid-step.
            self.assertIn("trainer.strategy=ddp_find_unused_parameters_true", launch_text)
            # Lightning saves both a "best" and a "last" checkpoint by default.
            # On the WAVE cluster the last checkpoint is saved via an atomic
            # temp→dest shutil.move that crosses the /local/scratch → /WAVE
            # device boundary; when the user quota is tight this copy hits
            # EDQUOT and crashes the job even though the best checkpoint was
            # already written.  Disabling save_last cuts checkpoint storage in
            # half and avoids the cross-device quota failure.
            self.assertIn("exp_manager.checkpoint_callback_params.save_last=false", launch_text)

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
            db_yml_path = artifacts.project_dir / "artifacts" / "database.yml"
            train_lst_path = artifacts.project_dir / "artifacts" / "lists" / "train.lst"
            train_py_path = artifacts.project_dir / "artifacts" / "train_pyannote.py"
            self.assertTrue(db_yml_path.exists())
            self.assertTrue(train_lst_path.exists())
            self.assertTrue(train_py_path.exists())
            self.assertTrue(artifacts.launch_script_path.exists())
            self.assertTrue(artifacts.sbatch_script_path.exists())
            self.assert_default_slurm_resources(artifacts.sbatch_script_path)

            # Lock in the cluster fixes so future trainings keep them: the
            # database.yml must point pyannote.database at absolute paths (it
            # otherwise resolves them relative to the YAML directory and can't
            # find lists/rttm/uem), the training script must keep the torchaudio
            # AudioMetaData shim, and the sbatch must load the GCCcore-tied
            # Python module so the venv's _sqlite3 finds its symbols.
            db_yml_text = db_yml_path.read_text(encoding="utf-8")
            self.assertIn(str(train_lst_path), db_yml_text)

            train_py_text = train_py_path.read_text(encoding="utf-8")
            self.assertIn("_patch_torchaudio_compatibility", train_py_text)
            self.assertIn("AudioMetaData", train_py_text)
            # PyTorch 2.6 made torch.load default to weights_only=True, which
            # rejects the TorchVersion / dataclass payloads in pyannote
            # checkpoints. The training script needs the same trusted-load
            # patch the diarization backend uses (`_trusted_torch_load_context`)
            # or `Model.from_pretrained` dies with `_pickle.UnpicklingError:
            # Weights only load failed` before training starts.
            self.assertIn("weights_only", train_py_text)
            self.assertIn("torch.load", train_py_text)
            # pyannote.database calls `torchaudio.info` for every training
            # file to precompute durations. With torchaudio 2.10's `info`/
            # `load`/`AudioMetaData` removed, a stub-only shim makes the
            # dataloader raise during the very first epoch — the script must
            # back the stubs with real soundfile-driven implementations.
            self.assertIn("import soundfile", train_py_text)
            self.assertIn("_sf.info", train_py_text)
            self.assertIn("_sf.read", train_py_text)
            # In torchaudio 2.10 the names exist but delegate to torchcodec,
            # which fails on the cluster (no FFmpeg libavutil). The shim has
            # to override unconditionally — a hasattr guard lets the broken
            # torchcodec path win and pyannote dies in the dataloader.
            self.assertNotIn('if not hasattr(torchaudio, "load")', train_py_text)
            self.assertNotIn('if not hasattr(torchaudio, "info")', train_py_text)

    def test_pyannote_uem_clips_to_actual_audio_frames(self):
        """A WAV whose RIFF header lies about its length must not push the
        UEM annotated range past the real audio. Otherwise pyannote's
        dataloader samples chunks past EOF mid-epoch with
        `requested chunk … lies outside file bounds`. Fabricating a
        truncated WAV (header claims much more data than is on disk) is
        the cheapest way to lock this in.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            real_audio_seconds = 1.5
            sample_rate = 16000
            # Build a normal 1.5 s PCM WAV, then rewrite the data-chunk
            # length field to lie about being 60 s long.
            audio = wav_bytes(real_audio_seconds, sample_rate=sample_rate)
            inflated_data_size = (60 * sample_rate) * 2
            data_marker = b"data"
            data_pos = audio.index(data_marker)
            length_pos = data_pos + len(data_marker)
            corrupted = (
                audio[:length_pos]
                + struct.pack("<I", inflated_data_size)
                + audio[length_pos + 4:]
            )
            fine_tuning.save_project_sample(
                project_name="Truncated WAV",
                backend="pyannote",
                audio_name="trunc.wav",
                audio_bytes=corrupted,
                rttm_name="trunc.rttm",
                rttm_bytes=(
                    b"SPEAKER trunc 1 0.000 0.500 <NA> <NA> Speaker_A <NA> <NA>\n"
                    b"SPEAKER trunc 1 0.700 0.500 <NA> <NA> Speaker_B <NA> <NA>\n"
                ),
                root=root,
            )
            artifacts = fine_tuning.prepare_project(
                project_name="Truncated WAV",
                backend="pyannote",
                root=root,
            )

            train_uem = (artifacts.project_dir / "artifacts" / "uem" / "train.uem")
            dev_uem = (artifacts.project_dir / "artifacts" / "uem" / "development.uem")
            uem_text = train_uem.read_text(encoding="utf-8") + dev_uem.read_text(encoding="utf-8")
            # The UEM line is `<stem> NA 0.000 <duration>`. Anything > the
            # real audio length means we've trusted the lying header.
            for line in uem_text.splitlines():
                parts = line.split()
                if len(parts) == 4 and parts[0] == "trunc":
                    self.assertLessEqual(
                        float(parts[3]),
                        real_audio_seconds + 0.05,
                        f"UEM annotated range {parts[3]} exceeds real audio {real_audio_seconds}s",
                    )

            sbatch_text = artifacts.sbatch_script_path.read_text(encoding="utf-8")
            self.assertIn("module load Python/3.12.3-GCCcore-14.2.0", sbatch_text)
            self.assertIn("sbatch_runtime_env.sh", sbatch_text)
            self.assertIn("expose_venv_native_libraries", sbatch_text)
            self.assertIn(".venv_pyannote", sbatch_text)

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

    def test_set_project_display_name_persists_in_sidecar_and_list(self):
        # The sidecar lives next to the (regen-on-prepare) metadata.json so a
        # user-chosen rename has to survive both list_projects() and a future
        # prepare round-trip.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            project_path = root / "fine_tuning" / "projects" / "nemo" / "site-training"
            (project_path / "audio").mkdir(parents=True)
            (project_path / "audio" / "001_clip.wav").write_bytes(wav_bytes())

            fine_tuning.set_project_display_name(
                "site-training", backend="nemo", display_name="Senior Design Run", root=root
            )
            sidecar = json.loads((project_path / "display.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["display_name"], "Senior Design Run")

            display = fine_tuning.read_project_display("site-training", backend="nemo", root=root)
            self.assertEqual(display["display_name"], "Senior Design Run")

            projects = fine_tuning.list_projects(root=root)
            named = next(p for p in projects if p["slug"] == "site-training")
            self.assertEqual(named["display_name"], "Senior Design Run")

            with self.assertRaises(ValueError):
                fine_tuning.set_project_display_name(
                    "site-training", backend="nemo", display_name="  ", root=root
                )
            with self.assertRaises(FileNotFoundError):
                fine_tuning.set_project_display_name(
                    "missing-project", backend="nemo", display_name="x", root=root
                )

    def test_set_run_display_name_persists_and_surfaces_in_list_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            project_path = root / "fine_tuning" / "projects" / "nemo" / "site-training"
            run_dir = project_path / "runs" / "20260505T100000Z_v001"
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                json.dumps({"version_name": "site-training trained version 1"}), encoding="utf-8"
            )

            fine_tuning.set_run_display_name(run_dir, display_name="Best v1 (lowest DER)")

            display = fine_tuning.read_run_display(run_dir)
            self.assertEqual(display["display_name"], "Best v1 (lowest DER)")
            runs = fine_tuning.list_runs("site-training", backend="nemo", root=root)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["display_name"], "Best v1 (lowest DER)")
            # Without an explicit display name, list_runs falls back to version_name.
            (run_dir / "display.json").unlink()
            runs_again = fine_tuning.list_runs("site-training", backend="nemo", root=root)
            self.assertEqual(runs_again[0]["display_name"], "site-training trained version 1")

    def test_set_project_auto_train_persists_and_surfaces_in_list(self):
        # Auto-train is a sticky per-project flag stored in display.json so it
        # survives prepare_project regeneration. list_projects must surface the
        # current value so the dashboard checkbox reflects truth.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            project_path = root / "fine_tuning" / "projects" / "pyannote" / "auto-train-demo"
            (project_path / "audio").mkdir(parents=True)
            (project_path / "audio" / "001_clip.wav").write_bytes(wav_bytes())

            fine_tuning.set_project_auto_train(
                "auto-train-demo", backend="pyannote", enabled=True, root=root
            )
            sidecar = json.loads((project_path / "display.json").read_text(encoding="utf-8"))
            self.assertTrue(sidecar.get("auto_train"))

            projects = fine_tuning.list_projects(root=root)
            named = next(p for p in projects if p["slug"] == "auto-train-demo")
            self.assertTrue(named["auto_train"])
            self.assertFalse(named["auto_train_pending"])

            fine_tuning.set_project_auto_train(
                "auto-train-demo", backend="pyannote", enabled=False, root=root
            )
            projects_off = fine_tuning.list_projects(root=root)
            named_off = next(p for p in projects_off if p["slug"] == "auto-train-demo")
            self.assertFalse(named_off["auto_train"])

    def test_run_status_uses_slurm_accounting_terminal_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            run_dir = root / "fine_tuning" / "projects" / "nemo" / "demo" / "runs" / "run-01"
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(json.dumps({"job_id": "12345"}), encoding="utf-8")

            original = fine_tuning.slurm_job_state
            try:
                fine_tuning.slurm_job_state = lambda _job_id: "COMPLETED"
                self.assertEqual(fine_tuning.run_status(run_dir), "succeeded")

                fine_tuning.slurm_job_state = lambda _job_id: "FAILED"
                self.assertEqual(fine_tuning.run_status(run_dir), "failed")

                fine_tuning.slurm_job_state = lambda _job_id: "PENDING"
                self.assertEqual(fine_tuning.run_status(run_dir), "submitted")
            finally:
                fine_tuning.slurm_job_state = original

    def test_auto_train_treats_slurm_pending_as_active(self):
        original = auto_train.ftm.list_runs
        try:
            auto_train.ftm.list_runs = lambda *_args, **_kwargs: [{"status": "pending"}]
            self.assertTrue(auto_train._has_active_run("demo", "nemo", pathlib.Path(".")))
        finally:
            auto_train.ftm.list_runs = original

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
