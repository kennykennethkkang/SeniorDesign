import contextlib
import importlib.util
import io
import pathlib
import re
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


diarize = load_module("diarize_cli_test", "all_diarization_programs/run_diarization.py")


class DiarizeCliTests(unittest.TestCase):
    def test_help_exits_cleanly_without_runtime_dependencies(self):
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as exc:
            with contextlib.redirect_stdout(stdout):
                diarize.main(["--help"])

        self.assertEqual(exc.exception.code, 0)
        self.assertIn("Run Whisper transcription plus speaker diarization", stdout.getvalue())

    def test_list_audio_mode_does_not_require_diarization_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            (temp_path / "001_clip.wav").write_text("x", encoding="utf-8")
            (temp_path / "notes.txt").write_text("x", encoding="utf-8")
            (temp_path / "001_clip_whisper_input.wav").write_text("x", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = diarize.main(["--input-dir", str(temp_path), "--list-audio"])

        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue().strip(), "001_clip.wav")

    def test_pyannote_requirements_avoid_diarizers_resolver_stack(self):
        requirements = (
            PROJECT_ROOT / "all_diarization_programs" / "requirements_pyannote.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("pyannote.audio", requirements)
        self.assertIn("huggingface_hub>=0.36,<1", requirements)
        self.assertIn("transformers==4.40.0", requirements)
        self.assertIsNone(re.search(r"(^|/)diarizers(\\.git)?($|\\s)", requirements))

    def test_ctc_aligner_dtype_patch_translates_for_transformers_four(self):
        class FakeAutoModelForCTC:
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return object()

        fake_transformers = type(sys)("transformers")
        fake_transformers.__version__ = "4.40.0"
        fake_alignment_utils = type(sys)("ctc_forced_aligner.alignment_utils")
        fake_alignment_utils.AutoModelForCTC = FakeAutoModelForCTC
        fake_ctc = type(sys)("ctc_forced_aligner")
        fake_ctc.alignment_utils = fake_alignment_utils

        originals = {
            name: sys.modules.get(name)
            for name in (
                "transformers",
                "ctc_forced_aligner",
                "ctc_forced_aligner.alignment_utils",
            )
        }
        try:
            sys.modules["transformers"] = fake_transformers
            sys.modules["ctc_forced_aligner"] = fake_ctc
            sys.modules["ctc_forced_aligner.alignment_utils"] = fake_alignment_utils

            diarize.patch_ctc_aligner_transformers_dtype()
            fake_alignment_utils.AutoModelForCTC.from_pretrained("model-id", dtype="float16")
        finally:
            for name, module in originals.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(FakeAutoModelForCTC.calls, [(("model-id",), {"torch_dtype": "float16"})])


if __name__ == "__main__":
    unittest.main()
