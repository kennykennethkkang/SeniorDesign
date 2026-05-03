import csv
import importlib.util
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


number_audio = load_module("number_audio_integration_test", "audio_numbering.py")
youtube_batch = load_module(
    "youtube_to_audio_integration_test",
    "youtube_audio_batch.py",
    extra_modules={"yt_dlp": types.SimpleNamespace(YoutubeDL=DummyYoutubeDL)},
)


class PipelineIntegrationTests(unittest.TestCase):
    def test_audio_numbering_and_youtube_indexing_work_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            audio_dir = temp_path / "audio_in"
            audio_dir.mkdir()
            list_path = temp_path / "youtube_links.txt"
            index_path = temp_path / "url_audio_index.tsv"
            resolved_output = temp_path / "url_resolution.tsv"
            resolved_audio_list = temp_path / "resolved_audio_paths.txt"
            removed_failed = temp_path / "removed_failed.tsv"
            links_error_dir = temp_path / "youtube_links_err"

            existing_audio = audio_dir / "speaker_sample.wav"
            existing_audio.write_text("existing", encoding="utf-8")

            url = "https://youtu.be/abcdefghijk"
            list_path.write_text(f"{url}\n", encoding="utf-8")

            with mock.patch.object(
                sys,
                "argv",
                ["audio_numbering.py", "--audio-dir", str(audio_dir)],
            ):
                self.assertEqual(number_audio.main(), 0)

            def fake_download_one(
                requested_url: str, out_dir: pathlib.Path, ffmpeg_location: str
            ):
                self.assertEqual(requested_url, url)
                self.assertEqual(out_dir, audio_dir)
                self.assertEqual(ffmpeg_location, "/usr/bin/ffmpeg")
                downloaded = out_dir / "fresh_download.wav"
                downloaded.write_text("downloaded", encoding="utf-8")
                return downloaded, {"id": "abcdefghijk", "title": "Integration Sample"}

            with mock.patch.object(
                youtube_batch,
                "resolve_ffmpeg_location",
                return_value="/usr/bin/ffmpeg",
            ), mock.patch.object(
                youtube_batch,
                "download_one",
                side_effect=fake_download_one,
            ), mock.patch.object(
                youtube_batch,
                "utc_now_iso",
                return_value="2026-03-10T12:00:00+00:00",
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "youtube_audio_batch.py",
                    "--list",
                    str(list_path),
                    "--output-dir",
                    str(audio_dir),
                    "--index-file",
                    str(index_path),
                    "--resolved-output-file",
                    str(resolved_output),
                    "--resolved-audio-list",
                    str(resolved_audio_list),
                    "--removed-failed-file",
                    str(removed_failed),
                    "--links-error-dir",
                    str(links_error_dir),
                ],
            ):
                self.assertEqual(youtube_batch.main(), 0)

            self.assertEqual(
                sorted(path.name for path in audio_dir.iterdir()),
                ["001_speaker_sample.wav", "002_fresh_download.wav"],
            )

            with index_path.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["url"], url)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["audio_file"], "002_fresh_download.wav")
            self.assertEqual(rows[0]["title"], "Integration Sample")
            self.assertEqual(rows[0]["note"], "downloaded_renumbered")

            with resolved_output.open("r", encoding="utf-8", newline="") as fh:
                resolved_rows = list(csv.DictReader(fh, delimiter="\t"))
            self.assertEqual(len(resolved_rows), 1)
            self.assertEqual(resolved_rows[0]["status"], "downloaded")
            self.assertEqual(resolved_rows[0]["queue_action"], "kept_in_queue")
            self.assertEqual(resolved_rows[0]["audio_file"], "002_fresh_download.wav")
            self.assertTrue(
                resolved_rows[0]["audio_path"].endswith("002_fresh_download.wav")
            )

            self.assertEqual(
                resolved_audio_list.read_text(encoding="utf-8").strip().splitlines(),
                [str(audio_dir / "002_fresh_download.wav")],
            )

            self.assertTrue(removed_failed.exists())
            self.assertEqual(list(links_error_dir.glob("*.tsv")), [])

    def test_retryable_youtube_error_stays_in_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            audio_dir = temp_path / "audio_in"
            audio_dir.mkdir()
            list_path = temp_path / "youtube_links.txt"
            queue_path = temp_path / "youtube_links_master.txt"
            index_path = temp_path / "url_audio_index.tsv"
            resolved_output = temp_path / "conversion_report.tsv"
            queue_updates = temp_path / "queue_updates.tsv"
            links_error_dir = temp_path / "youtube_links_err"
            url = "https://youtu.be/abcdefghijk"
            list_path.write_text(f"{url}\n", encoding="utf-8")
            queue_path.write_text(f"{url}\n", encoding="utf-8")

            with mock.patch.object(
                youtube_batch,
                "resolve_ffmpeg_location",
                return_value="/usr/bin/ffmpeg",
            ), mock.patch.object(
                youtube_batch,
                "download_one",
                side_effect=RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden"),
            ), mock.patch.object(
                youtube_batch,
                "utc_now_iso",
                return_value="2026-03-10T12:00:00+00:00",
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "youtube_audio_batch.py",
                    "--list",
                    str(list_path),
                    "--queue-file",
                    str(queue_path),
                    "--output-dir",
                    str(audio_dir),
                    "--index-file",
                    str(index_path),
                    "--resolved-output-file",
                    str(resolved_output),
                    "--removed-failed-file",
                    str(queue_updates),
                    "--links-error-dir",
                    str(links_error_dir),
                ],
            ):
                self.assertEqual(youtube_batch.main(), 1)

            self.assertEqual(queue_path.read_text(encoding="utf-8"), f"{url}\n")
            with resolved_output.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            self.assertEqual(rows[0]["status"], "retry")
            self.assertEqual(rows[0]["queue_action"], "keep_in_queue")
            with queue_updates.open("r", encoding="utf-8", newline="") as fh:
                update_rows = list(csv.DictReader(fh, delimiter="\t"))
            self.assertEqual(update_rows[0]["queue_result"], "kept")

    def test_no_data_youtube_error_is_removed_from_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            audio_dir = temp_path / "audio_in"
            audio_dir.mkdir()
            list_path = temp_path / "youtube_links.txt"
            queue_path = temp_path / "youtube_links_master.txt"
            index_path = temp_path / "url_audio_index.tsv"
            resolved_output = temp_path / "conversion_report.tsv"
            queue_updates = temp_path / "queue_updates.tsv"
            links_error_dir = temp_path / "youtube_links_err"
            url = "https://youtu.be/abcdefghijk"
            list_path.write_text(f"{url}\n", encoding="utf-8")
            queue_path.write_text(f"{url}\n", encoding="utf-8")

            with mock.patch.object(
                youtube_batch,
                "resolve_ffmpeg_location",
                return_value="/usr/bin/ffmpeg",
            ), mock.patch.object(
                youtube_batch,
                "download_one",
                side_effect=RuntimeError("This video is unavailable"),
            ), mock.patch.object(
                youtube_batch,
                "utc_now_iso",
                return_value="2026-03-10T12:00:00+00:00",
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "youtube_audio_batch.py",
                    "--list",
                    str(list_path),
                    "--queue-file",
                    str(queue_path),
                    "--output-dir",
                    str(audio_dir),
                    "--index-file",
                    str(index_path),
                    "--resolved-output-file",
                    str(resolved_output),
                    "--removed-failed-file",
                    str(queue_updates),
                    "--links-error-dir",
                    str(links_error_dir),
                ],
            ):
                self.assertEqual(youtube_batch.main(), 1)

            self.assertEqual(queue_path.read_text(encoding="utf-8"), "")
            with resolved_output.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            self.assertEqual(rows[0]["status"], "no_data")
            self.assertEqual(rows[0]["queue_action"], "remove_from_queue")
            with queue_updates.open("r", encoding="utf-8", newline="") as fh:
                update_rows = list(csv.DictReader(fh, delimiter="\t"))
            self.assertEqual(update_rows[0]["queue_result"], "removed")


if __name__ == "__main__":
    unittest.main()
