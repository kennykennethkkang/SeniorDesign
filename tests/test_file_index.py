"""Tests for the cross-run file index."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dashboard import file_index


class FileIndexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.audio = self.root / "audio_in" / "youtube_links"
        self.audio.mkdir(parents=True)
        self.outputs = self.root / "outputs" / "diarization_runs" / "nemo" / "20260101T000000_run"
        self.outputs.mkdir(parents=True)
        (self.outputs / "logs").mkdir(parents=True)
        self.fine_tuning = self.root / "fine_tuning"
        self.fine_tuning.mkdir(parents=True)

        # Sample artifacts that share a stem.
        (self.audio / "017_call.wav").write_bytes(b"fake")
        (self.outputs / "017_call.srt").write_text("1\n", encoding="utf-8")
        (self.outputs / "logs" / "017_call.wav.out").write_text("ok\n", encoding="utf-8")
        # An unrelated file we should still index.
        (self.audio / "099_other.wav").write_bytes(b"fake")
        # A nemo experiment dump that should be SKIPPED.
        excluded = self.outputs / "experiments" / "nemo_msdd"
        excluded.mkdir(parents=True)
        (excluded / "017_call.wav").write_bytes(b"junk")

    def tearDown(self):
        self._tmp.cleanup()

    def test_index_groups_files_by_stem(self):
        index = file_index.build_file_index(
            audio_dir=self.audio.parent,
            outputs_root=self.root / "outputs",
            fine_tuning_root=self.fine_tuning,
        )
        self.assertIn("017_call", index)
        kinds = {record["kind"] for record in index["017_call"]}
        # We should have at least the wav, the srt, and the log
        self.assertIn("audio", kinds)
        self.assertIn("srt", kinds)
        self.assertIn("log", kinds)
        # Excluded experiment dir must not contribute records.
        for record in index["017_call"]:
            self.assertNotIn("experiments", record["rel_path"])

    def test_index_strips_numeric_prefix(self):
        index = file_index.build_file_index(
            audio_dir=self.audio.parent,
            outputs_root=self.root / "outputs",
            fine_tuning_root=self.fine_tuning,
        )
        # The bare "call" stem should also work because of the prefix-stripping.
        self.assertIn("call", index)
        self.assertEqual(index["call"][0]["name"], "017_call.wav")

    def test_search_prefix_ranks_above_substring(self):
        index = {
            "calling_extra": [{"name": "x", "kind": "audio", "rel_path": "x", "size": 0, "run_name": ""}],
            "017_call": [{"name": "y", "kind": "audio", "rel_path": "y", "size": 0, "run_name": ""}],
            "scallop": [{"name": "z", "kind": "audio", "rel_path": "z", "size": 0, "run_name": ""}],
        }
        results = file_index.search_index(index, "call", limit=10)
        self.assertEqual([r["stem"] for r in results], ["calling_extra", "017_call", "scallop"])

    def test_search_empty_query_returns_empty(self):
        results = file_index.search_index({"a": [{"name": "n", "kind": "audio", "rel_path": "x", "size": 0, "run_name": ""}]}, "", limit=10)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
