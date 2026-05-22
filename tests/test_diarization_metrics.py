"""Tests for the senior-design DER + companion metrics module.

Each test builds tiny hand-checkable RTTM payloads so the expected numbers can
be verified on paper. The point isn't to retest scipy; it's to make sure the
glue between RTTM parsing, speaker mapping, and DER component bookkeeping is
right, because that's where this kind of code usually drifts.
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest

import diarization_metrics as metrics


def write_rttm(path: pathlib.Path, segments) -> None:
    """Helper that turns (start, duration, speaker) tuples into RTTM lines."""

    lines = []
    for start, duration, speaker in segments:
        lines.append(
            f"SPEAKER clip 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class DiarizationMetricsTests(unittest.TestCase):
    def test_identical_rttm_returns_zero_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ref = root / "ref.rttm"
            hyp = root / "hyp.rttm"
            segments = [(0.0, 5.0, "S1"), (6.0, 4.0, "S2")]
            write_rttm(ref, segments)
            write_rttm(hyp, segments)
            payload = metrics.score_run(ref, hyp)
            self.assertAlmostEqual(payload["der"], 0.0)
            self.assertAlmostEqual(payload["miss_seconds"], 0.0)
            self.assertAlmostEqual(payload["false_alarm_seconds"], 0.0)
            self.assertAlmostEqual(payload["confusion_seconds"], 0.0)
            self.assertAlmostEqual(payload["jer"], 0.0)
            self.assertEqual(payload["reference_speaker_count"], 2)
            self.assertEqual(payload["hypothesis_speaker_count"], 2)
            self.assertEqual(payload["speaker_count_diff"], 0)

    def test_speaker_label_swap_is_resolved_by_mapping_not_treated_as_confusion(self):
        # If the model labels the same time correctly but uses different
        # speaker names ("Speaker_0" vs "Speaker_1"), best_speaker_mapping
        # should fold those together so DER stays zero.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ref = root / "ref.rttm"
            hyp = root / "hyp.rttm"
            write_rttm(ref, [(0.0, 5.0, "Alice"), (6.0, 4.0, "Bob")])
            write_rttm(hyp, [(0.0, 5.0, "Speaker_0"), (6.0, 4.0, "Speaker_1")])
            payload = metrics.score_run(ref, hyp)
            self.assertAlmostEqual(payload["der"], 0.0)
            self.assertAlmostEqual(payload["confusion_seconds"], 0.0)

    def test_full_miss_when_hypothesis_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ref = root / "ref.rttm"
            hyp = root / "hyp.rttm"
            write_rttm(ref, [(0.0, 5.0, "S1")])
            write_rttm(hyp, [])
            payload = metrics.score_run(ref, hyp)
            self.assertAlmostEqual(payload["miss_seconds"], 5.0)
            self.assertAlmostEqual(payload["false_alarm_seconds"], 0.0)
            self.assertAlmostEqual(payload["confusion_seconds"], 0.0)
            self.assertAlmostEqual(payload["der"], 1.0)
            self.assertEqual(payload["hypothesis_speaker_count"], 0)
            self.assertAlmostEqual(payload["jer"], 1.0)

    def test_pure_false_alarm_when_reference_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ref = root / "ref.rttm"
            hyp = root / "hyp.rttm"
            write_rttm(ref, [])
            write_rttm(hyp, [(0.0, 3.0, "S1")])
            payload = metrics.score_run(ref, hyp)
            self.assertAlmostEqual(payload["false_alarm_seconds"], 3.0)
            self.assertEqual(payload["der"], float("inf"))

    def test_two_overlapping_reference_speakers_against_single_hypothesis_is_a_miss(self):
        # Person-time DER: ref has two speakers concurrently (S1 + S2 both
        # in [0,2]), hyp has only one speaker. Best mapping pairs one of
        # them to H1; the other contributes 2 person-seconds of MISS, since
        # there's no hypothesis slot left for it.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ref = root / "ref.rttm"
            hyp = root / "hyp.rttm"
            write_rttm(ref, [(0.0, 2.0, "S1"), (0.0, 2.0, "S2")])
            write_rttm(hyp, [(0.0, 2.0, "H1")])
            payload = metrics.score_run(ref, hyp)
            self.assertAlmostEqual(payload["miss_seconds"], 2.0, places=5)
            self.assertAlmostEqual(payload["false_alarm_seconds"], 0.0)
            self.assertAlmostEqual(payload["confusion_seconds"], 0.0)
            # ref is 2 speakers * 2s = 4 person-seconds, total error 2 -> DER 0.5.
            self.assertAlmostEqual(payload["der"], 0.5, places=5)

    def test_confusion_when_mapped_pair_is_inactive_but_other_pair_is_active(self):
        # Hyp uses ONE speaker label for two non-overlapping ref speakers.
        # Mapping picks whichever ref speaker has more overlap with H1
        # (here it's a tie at 2.0s; whichever wins, the other 2s slice has
        # n_r=1, n_h=1 but n_correct=0 -> 2 person-seconds of confusion).
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ref = root / "ref.rttm"
            hyp = root / "hyp.rttm"
            write_rttm(ref, [(0.0, 2.0, "S1"), (3.0, 2.0, "S2")])
            write_rttm(hyp, [(0.0, 2.0, "H1"), (3.0, 2.0, "H1")])
            payload = metrics.score_run(ref, hyp)
            self.assertAlmostEqual(payload["confusion_seconds"], 2.0, places=5)
            self.assertAlmostEqual(payload["miss_seconds"], 0.0)
            self.assertAlmostEqual(payload["false_alarm_seconds"], 0.0)
            # ref is 4 person-seconds; confusion 2s -> DER 0.5.
            self.assertAlmostEqual(payload["der"], 0.5, places=5)

    def test_jer_is_one_for_a_completely_missed_speaker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ref = root / "ref.rttm"
            hyp = root / "hyp.rttm"
            write_rttm(ref, [(0.0, 5.0, "Talker"), (10.0, 5.0, "Quiet")])
            # Hypothesis only catches one of the two reference speakers.
            write_rttm(hyp, [(0.0, 5.0, "H_only")])
            payload = metrics.score_run(ref, hyp)
            self.assertAlmostEqual(payload["per_speaker_jer"]["Talker"], 0.0)
            self.assertAlmostEqual(payload["per_speaker_jer"]["Quiet"], 1.0)
            self.assertAlmostEqual(payload["jer"], 0.5)
            self.assertEqual(payload["speaker_count_diff"], 1)


class RttmParsingTests(unittest.TestCase):
    def test_zero_or_negative_duration_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ref = root / "ref.rttm"
            ref.write_text(
                "\n".join(
                    [
                        "SPEAKER clip 1 0.000 0.000 <NA> <NA> S1 <NA> <NA>",
                        "SPEAKER clip 1 1.000 -0.500 <NA> <NA> S1 <NA> <NA>",
                        "SPEAKER clip 1 2.000 1.000 <NA> <NA> S1 <NA> <NA>",
                        "# a comment line",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            intervals = metrics.parse_rttm_intervals(ref)
            self.assertEqual(len(intervals), 1)
            self.assertEqual(intervals[0].speaker, "S1")
            self.assertAlmostEqual(intervals[0].start, 2.0)
            self.assertAlmostEqual(intervals[0].end, 3.0)


class ScoreIntervalsTests(unittest.TestCase):
    """The DER calculator panel feeds in already-parsed intervals (cues from
    SRT files), so the path-free entry point needs to keep parity with score_run."""

    def test_score_intervals_matches_score_run_on_identical_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ref_path = root / "ref.rttm"
            hyp_path = root / "hyp.rttm"
            ref_segments = [(0.0, 4.0, "Alice"), (5.0, 3.0, "Bob")]
            hyp_segments = [(0.0, 4.0, "S0"), (5.5, 2.5, "S1")]
            write_rttm(ref_path, ref_segments)
            write_rttm(hyp_path, hyp_segments)
            ref_intervals = metrics.parse_rttm_intervals(ref_path)
            hyp_intervals = metrics.parse_rttm_intervals(hyp_path)

            inline = metrics.score_intervals(ref_intervals, hyp_intervals)
            from_files = metrics.score_run(ref_path, hyp_path)

            for key in (
                "der",
                "miss_seconds",
                "false_alarm_seconds",
                "confusion_seconds",
                "reference_speech_seconds",
                "hypothesis_speech_seconds",
                "jer",
                "reference_speaker_count",
                "hypothesis_speaker_count",
                "speaker_count_diff",
            ):
                self.assertAlmostEqual(inline[key], from_files[key], msg=f"key={key}")

    def test_score_intervals_with_empty_hypothesis_is_pure_miss(self):
        # If the model didn't say anything but the reference has speech, every
        # reference second is missed and false-alarm is zero.
        ref_intervals = [
            metrics.Interval(start=0.0, end=2.0, speaker="A"),
            metrics.Interval(start=3.0, end=5.0, speaker="B"),
        ]
        payload = metrics.score_intervals(ref_intervals, [])
        self.assertAlmostEqual(payload["miss_seconds"], 4.0)
        self.assertAlmostEqual(payload["false_alarm_seconds"], 0.0)
        self.assertAlmostEqual(payload["confusion_seconds"], 0.0)
        self.assertAlmostEqual(payload["der"], 1.0)
        self.assertEqual(payload["hypothesis_speaker_count"], 0)


if __name__ == "__main__":
    unittest.main()
