import csv
import importlib.util
import pathlib
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


review_outputs = load_module("review_outputs_test", "review_bundle.py")


class ReviewOutputsTests(unittest.TestCase):
    def test_write_review_bundle_generates_html_and_flag_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            srt_path = temp_path / "clip.srt"
            media_path = temp_path / "clip.wav"
            html_path = temp_path / "clip_review.html"
            report_path = temp_path / "clip_review_flags.tsv"

            srt_path.write_text(
                "1\n"
                "00:00:00,000 --> 00:00:00,200\n"
                "Speaker 0: Hi.\n\n"
                "2\n"
                "00:00:00,150 --> 00:00:00,300\n"
                "Speaker 0: Hi.\n",
                encoding="utf-8",
            )
            media_path.write_text("media", encoding="utf-8")

            review_outputs.write_review_bundle(
                srt_path=srt_path,
                media_path=media_path,
                output_html=html_path,
                report_tsv=report_path,
                fine_tuning_projects=[
                    {
                        "backend": "pyannote",
                        "slug": "speaker-lab",
                        "display_name": "Speaker Lab",
                        "sample_count": 3,
                        "prepared": True,
                        "auto_train": True,
                        "recent_runs": [
                            {
                                "status": "completed",
                                "version_name": "speaker-lab-v2",
                                "display_name": "Speaker Lab v2",
                                "run_dir": str(temp_path / "fine_tuning" / "projects" / "pyannote" / "speaker-lab" / "runs" / "20260417T120000_speaker-lab-v2"),
                            }
                        ],
                    }
                ],
                quiet=True,
            )

            self.assertTrue(html_path.exists())
            self.assertTrue(report_path.exists())
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("clip.wav", html)
            self.assertIn("playRange", html)
            self.assertIn("Training Labels", html)
            self.assertIn("Detected Segments", html)
            self.assertIn(">Back</button>", html)
            self.assertIn("nowPlaying", html)
            self.assertIn("class=\"play-label\"", html)
            self.assertIn("id=\"startAtSegment\"", html)
            self.assertIn("id=\"stopAtSegmentEnd\"", html)
            self.assertIn("id=\"labelSearch\"", html)
            self.assertIn("id=\"cueSpeakerFilter\"", html)
            self.assertIn("class=\"use-cue\"", html)
            self.assertIn("Add Label", html)
            self.assertIn("Listen starts at the selected segment", html)
            self.assertIn("clearPlaybackTimers", html)
            self.assertIn("id=\"waveformCanvas\"", html)
            self.assertIn("id=\"playbackRate\"", html)
            self.assertIn("loadWaveform", html)
            self.assertIn("id=\"setLabelTimeFromPlayer\"", html)
            self.assertIn("setActiveLabelTimeFromPlayer", html)
            self.assertIn("Use Current Seconds", html)
            self.assertIn("manual-label-toolbar", html)
            self.assertIn("id=\"currentTimeReadout\"", html)
            self.assertIn("formatSeconds", html)
            self.assertIn('id="labelIncludeTranscript" name="label_include_transcript" value="0"', html)
            self.assertIn('id="showLabelDialogue" type="checkbox"> <span id="labelDialogueToggleText">Show Dialogue</span>', html)
            self.assertIn('<table class="label-table hide-dialogue">', html)
            self.assertIn("scheduleLabelAutoSave", html)
            self.assertIn("label_auto_save", html)
            self.assertIn("flushLabelAutoSaveOnUnload", html)
            self.assertIn("sendBeacon", html)
            self.assertIn("firstEditSaved", html)
            self.assertIn('id="cacheAudioButton"', html)
            self.assertIn('id="cacheAllAudioButton"', html)
            self.assertIn("cacheAllTrainingAudio", html)
            self.assertIn("fetchAudioWithProgress", html)
            self.assertIn("response.blob()", html)
            self.assertNotIn("Adjust selected label", html)
            self.assertNotIn("Set Start Here", html)
            self.assertIn("id=\"labelHealth\"", html)
            self.assertIn("Speaker_0", html)
            self.assertIn("review-grid", html)
            self.assertIn("label_action", html)
            self.assertIn("class=\"play-cue\"", html)
            self.assertIn("id=\"trainingTargetDialog\"", html)
            self.assertIn("id=\"trainingTargetData\"", html)
            self.assertIn("\"key\": \"pyannote/speaker-lab\"", html)
            self.assertIn("Speaker Lab v2", html)
            self.assertIn("previous fine-tuned model", html)
            self.assertIn("Create new", html)
            self.assertIn("id=\"newTrainingBackend\"", html)
            self.assertIn("id=\"newTrainingProjectName\"", html)
            self.assertIn("id=\"newTrainingVersionName\"", html)
            self.assertIn("Queue Selected", html)

            with report_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

            self.assertEqual(len(rows), 2)
            self.assertIn("very_short_segment", rows[0]["flags"])
            self.assertIn("overlap_previous", rows[1]["flags"])
            self.assertIn("duplicate_adjacent_text", rows[1]["flags"])

    def test_write_review_bundle_keeps_saved_dialogue_toggle_on(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            srt_path = temp_path / "clip.srt"
            media_path = temp_path / "clip.wav"
            html_path = temp_path / "clip_review.html"

            srt_path.write_text(
                "1\n"
                "00:00:00,000 --> 00:00:00,500\n"
                "Speaker 0: Saved line.\n",
                encoding="utf-8",
            )
            media_path.write_text("media", encoding="utf-8")

            review_outputs.write_review_bundle(
                srt_path=srt_path,
                media_path=media_path,
                output_html=html_path,
                training_label_records={
                    "clip.wav": {
                        "include_transcript": True,
                        "label_segments": "0.000 0.500 SPEAKER_00",
                        "transcript_text": "Saved line.",
                    }
                },
                quiet=True,
            )

            html = html_path.read_text(encoding="utf-8")
            self.assertIn('id="labelIncludeTranscript" name="label_include_transcript" value="1"', html)
            self.assertIn('id="showLabelDialogue" type="checkbox" checked> <span id="labelDialogueToggleText">Hide Dialogue</span>', html)
            self.assertIn('<table class="label-table">', html)
            self.assertNotIn('<table class="label-table hide-dialogue">', html)

    def test_write_review_bundle_can_auto_match_media_by_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            audio_dir = temp_path / "audio_in"
            audio_dir.mkdir()
            srt_path = temp_path / "sample.srt"
            media_path = audio_dir / "sample.wav"

            srt_path.write_text(
                "1\n"
                "00:00:01,000 --> 00:00:02,000\n"
                "Speaker 1: Example line.\n",
                encoding="utf-8",
            )
            media_path.write_text("media", encoding="utf-8")

            html_path, report_path = review_outputs.write_review_bundle(
                srt_path=srt_path,
                audio_dir=audio_dir,
                quiet=True,
            )

            self.assertTrue(html_path.exists())
            self.assertTrue(report_path.exists())
            self.assertIn("sample.wav", html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
