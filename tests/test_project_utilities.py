import contextlib
import importlib.util
import io
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str, extra_modules=None):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    extra_modules = extra_modules or {}
    saved_modules = {}

    try:
        for name, replacement in extra_modules.items():
            saved_modules[name] = sys.modules.get(name)
            sys.modules[name] = replacement
        spec.loader.exec_module(module)
    finally:
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original 

    return module


class DummyYoutubeDL:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


number_audio = load_module("number_audio_in_files_test", "audio_numbering.py")
youtube_batch = load_module(
    "youtube_to_audio_batch_test",
    "youtube_audio_batch.py",
    extra_modules={"yt_dlp": types.SimpleNamespace(YoutubeDL=DummyYoutubeDL)},
)


class NumberAudioInFilesTests(unittest.TestCase):
    def test_is_audio_file_accepts_audio_and_rejects_generated_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            audio_path = temp_path / "clip.wav"
            generated_path = temp_path / "clip_whisper_input.wav"
            text_path = temp_path / "notes.txt"
            audio_path.write_text("x", encoding="utf-8")
            generated_path.write_text("x", encoding="utf-8")
            text_path.write_text("x", encoding="utf-8")

            self.assertTrue(number_audio.is_audio_file(audio_path))
            self.assertFalse(number_audio.is_audio_file(generated_path))
            self.assertFalse(number_audio.is_audio_file(text_path))

    def test_list_audio_files_returns_sorted_audio_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            for name in ["b.wav", "A.mp3", "ignore.txt", "z_whisper_input.wav"]:
                (temp_path / name).write_text("x", encoding="utf-8")

            audio_names = [path.name for path in number_audio.list_audio_files(temp_path)]
            self.assertEqual(audio_names, ["A.mp3", "b.wav"])

    def test_next_available_number_returns_first_gap(self):
        self.assertEqual(number_audio.next_available_number({1, 2, 4}), 3)

    def test_main_renumbers_unprefixed_audio_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            (temp_path / "song.wav").write_text("x", encoding="utf-8")
            (temp_path / "002_existing.wav").write_text("x", encoding="utf-8")

            stdout = io.StringIO()
            with mock.patch.object(
                sys,
                "argv",
                ["audio_numbering.py", "--audio-dir", str(temp_path)],
            ):
                with contextlib.redirect_stdout(stdout):
                    result = number_audio.main()

            self.assertEqual(result, 0)
            self.assertTrue((temp_path / "001_song.wav").exists())
            self.assertTrue((temp_path / "002_existing.wav").exists())
            self.assertIn("1 renamed", stdout.getvalue())


class YoutubeToAudioBatchTests(unittest.TestCase):
    def test_extract_video_id_handles_standard_and_short_urls(self):
        self.assertEqual(
            youtube_batch.extract_video_id("https://www.youtube.com/watch?v=abcdefghijk"),
            "abcdefghijk",
        )
        self.assertEqual(
            youtube_batch.extract_video_id("https://youtu.be/ZYXWVUTSRQP"),
            "ZYXWVUTSRQP",
        )

    def test_classify_issue_separates_no_data_from_retryable_errors(self):
        self.assertEqual(
            youtube_batch.classify_issue("This video is unavailable"),
            "no_data",
        )
        self.assertEqual(
            youtube_batch.classify_issue("ERROR: unable to download video data: HTTP Error 403: Forbidden"),
            "retry",
        )

    def test_prune_failed_urls_from_list_removes_only_failed_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            list_path = pathlib.Path(tmpdir) / "youtube_links.txt"
            list_path.write_text(
                "# keep comment\n"
                "https://youtu.be/abcdefghijk\n"
                "\n"
                "https://youtu.be/ZYXWVUTSRQP\n",
                encoding="utf-8",
            )

            removed = youtube_batch.prune_failed_urls_from_list(
                list_path,
                {"https://youtu.be/ZYXWVUTSRQP"},
            )

            self.assertEqual(removed, ["https://youtu.be/ZYXWVUTSRQP"])
            self.assertEqual(
                list_path.read_text(encoding="utf-8"),
                "# keep comment\nhttps://youtu.be/abcdefghijk\n\n",
            )

    def test_resolve_audio_path_in_dir_accepts_only_files_inside_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            out_dir = temp_path / "audio_in"
            out_dir.mkdir()
            inside = out_dir / "001_clip.wav"
            outside = temp_path / "outside.wav"
            inside.write_text("x", encoding="utf-8")
            outside.write_text("x", encoding="utf-8")

            self.assertEqual(
                youtube_batch.resolve_audio_path_in_dir(str(inside), out_dir),
                inside.resolve(),
            )
            self.assertIsNone(
                youtube_batch.resolve_audio_path_in_dir(str(outside), out_dir)
            )

    def test_collect_used_prefix_numbers_ignores_invalid_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            for name in [
                "001_first.wav",
                "007_video.webm",
                "003_clip_whisper_input.wav",
                "notes.txt",
            ]:
                (out_dir / name).write_text("x", encoding="utf-8")

            self.assertEqual(youtube_batch.collect_used_prefix_numbers(out_dir), {1, 7})

    def test_ensure_numbered_audio_path_renames_unprefixed_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            source = out_dir / "fresh.wav"
            source.write_text("x", encoding="utf-8")

            renamed_path, renamed = youtube_batch.ensure_numbered_audio_path(
                source,
                out_dir,
                {2},
            )

            self.assertTrue(renamed)
            self.assertEqual(renamed_path.name, "001_fresh.wav")
            self.assertTrue(renamed_path.exists())
            self.assertFalse(source.exists())


if __name__ == "__main__":
    unittest.main()
