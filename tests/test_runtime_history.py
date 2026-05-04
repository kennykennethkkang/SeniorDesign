"""Tests for the runtime-history time-estimate engine."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dashboard import runtime_history


def _write_summary(run_dir: Path, rows: list[tuple[str, str, float]]) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    summary = logs / "runtime_summary.tsv"
    lines = ["audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary"]
    for audio_file, status, runtime in rows:
        lines.append(f"{audio_file}\t{status}\t{runtime}\t/tmp/x.out\t/tmp/x.err\t")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Mark the run as completed so it shows up in history.
    (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")


class FormatSecondsTests(unittest.TestCase):
    def test_under_a_minute(self):
        self.assertEqual(runtime_history.format_seconds_human(0), "~0 s")
        self.assertEqual(runtime_history.format_seconds_human(7), "~7 s")
        self.assertEqual(runtime_history.format_seconds_human(59), "~59 s")

    def test_minutes(self):
        self.assertEqual(runtime_history.format_seconds_human(60), "~1 min")
        self.assertEqual(runtime_history.format_seconds_human(125), "~2 min")
        self.assertEqual(runtime_history.format_seconds_human(3599), "~60 min")  # rounds to 60 min

    def test_hours(self):
        self.assertEqual(runtime_history.format_seconds_human(3600), "~1 h")
        self.assertEqual(runtime_history.format_seconds_human(3660), "~1 h 1 min")
        self.assertEqual(runtime_history.format_seconds_human(7320), "~2 h 2 min")


class GatherHistoricalRatesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.audio_dir = self.root / "audio_in"
        self.audio_dir.mkdir(parents=True)
        self.local = self.root / ".local_dashboard"
        self.local.mkdir(parents=True)
        self.runs = self.root / "outputs" / "diarization_runs"
        self.runs.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_unavailable_when_root_missing(self):
        empty_root = self.root / "no-runs-here"
        result = runtime_history.gather_historical_rates(
            empty_root, audio_dir=self.audio_dir, local_dashboard_dir=self.local
        )
        self.assertEqual(result, {})

    def test_computes_rolling_ratio_per_backend(self):
        # Two nemo runs, three files each, with mocked durations.
        nemo_dir = self.runs / "nemo"
        nemo_dir.mkdir()
        for run_name, files in [
            ("20260101T000000_diarization_nemo_03-items", [
                ("youtube_links/001_a.wav", "ok", 60.0),
                ("youtube_links/002_b.wav", "ok", 90.0),
                ("youtube_links/003_c.wav", "ok", 120.0),
            ]),
            ("20260102T000000_diarization_nemo_03-items", [
                ("youtube_links/004_d.wav", "ok", 30.0),
                ("youtube_links/005_e.wav", "failed", 5.0),  # excluded (not "ok")
                ("youtube_links/006_f.wav", "ok", 45.0),
            ]),
        ]:
            run_dir = nemo_dir / run_name
            run_dir.mkdir()
            _write_summary(run_dir, files)

        # Pretend every audio file is exactly 200 s long, regardless of stat hash.
        with mock.patch.object(runtime_history, "probe_media_duration", return_value=200.0):
            # cache_key_for_file expects the file to exist; touch synthetic files.
            for rel in ["001_a.wav", "002_b.wav", "003_c.wav", "004_d.wav", "005_e.wav", "006_f.wav"]:
                (self.audio_dir / "youtube_links").mkdir(exist_ok=True)
                (self.audio_dir / "youtube_links" / rel).write_bytes(b"fake")
            result = runtime_history.gather_historical_rates(
                self.runs, audio_dir=self.audio_dir, local_dashboard_dir=self.local
            )

        self.assertIn("nemo", result)
        nemo = result["nemo"]
        self.assertTrue(nemo["available"])
        # 5 ok rows (one failed excluded), each 200s audio
        self.assertEqual(nemo["sample_count"], 5)
        # Ratios: 60/200, 90/200, 120/200, 30/200, 45/200 = 0.3, 0.45, 0.6, 0.15, 0.225
        # Mean = 0.345
        self.assertAlmostEqual(nemo["ratio"], 0.345, places=3)
        self.assertEqual(nemo["runs_used"], 2)

    def test_returns_unavailable_when_too_few_samples(self):
        nemo_dir = self.runs / "nemo"
        nemo_dir.mkdir()
        run_dir = nemo_dir / "20260101T000000_diarization_nemo_02-items"
        run_dir.mkdir()
        _write_summary(run_dir, [("youtube_links/001.wav", "ok", 50.0)])

        with mock.patch.object(runtime_history, "probe_media_duration", return_value=200.0):
            (self.audio_dir / "youtube_links").mkdir(exist_ok=True)
            (self.audio_dir / "youtube_links" / "001.wav").write_bytes(b"fake")
            result = runtime_history.gather_historical_rates(
                self.runs, audio_dir=self.audio_dir, local_dashboard_dir=self.local
            )
        self.assertFalse(result["nemo"]["available"])
        self.assertEqual(result["nemo"]["sample_count"], 1)
        self.assertIn("Need at least", result["nemo"]["message"])


class EstimateRuntimeForFilesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.local = self.root / ".local_dashboard"
        self.local.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_estimates_seconds_when_history_is_available(self):
        rates = {
            "nemo": {
                "available": True,
                "ratio": 0.5,
                "ratio_median": 0.5,
                "sample_count": 12,
                "runs_used": 3,
                "last_run": "20260101T000000_nemo_03",
            },
        }
        path1 = self.root / "001.wav"
        path2 = self.root / "002.wav"
        path1.write_bytes(b"fake")
        path2.write_bytes(b"fake")

        with mock.patch.object(runtime_history, "probe_media_duration", return_value=120.0):
            result = runtime_history.estimate_runtime_for_files(
                [path1, path2], backend="nemo", rates=rates, local_dashboard_dir=self.local
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["files"], 2)
        self.assertEqual(result["total_audio_seconds"], 240.0)
        # 240s * 0.5 = 120s
        self.assertEqual(result["estimate_seconds"], 120.0)
        self.assertEqual(result["estimate_label"], "~2 min")
        self.assertEqual(result["based_on_files"], 12)

    def test_unavailable_when_no_history(self):
        rates = {"nemo": {"available": False, "sample_count": 1, "message": "Need 3"}}
        path = self.root / "x.wav"
        path.write_bytes(b"fake")
        result = runtime_history.estimate_runtime_for_files(
            [path], backend="nemo", rates=rates, local_dashboard_dir=self.local
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["message"], "Need 3")

    def test_unavailable_when_no_files_selected(self):
        result = runtime_history.estimate_runtime_for_files(
            [], backend="nemo", rates={}, local_dashboard_dir=self.local
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["files"], 0)


if __name__ == "__main__":
    unittest.main()
