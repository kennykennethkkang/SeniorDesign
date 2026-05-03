import importlib.util
import pathlib
import sys
import tempfile
import types
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


workflow_cli = load_module("workflow_cli_test", "workflow_cli.py")


class WorkflowCliTests(unittest.TestCase):
    def test_build_local_web_command_uses_localhost_and_url_file(self):
        command = workflow_cli.build_local_web_command(
            port=8123,
            server="threaded",
            threads=4,
            url_file=pathlib.Path("/tmp/dashboard.url"),
            python_bin="/usr/bin/python3",
        )

        self.assertEqual(command[0], "/usr/bin/python3")
        self.assertEqual(command[1], "-u")
        self.assertEqual(command[2], str(PROJECT_ROOT / "workflow_dashboard.py"))
        self.assertIn("--host", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--url-file", command)
        self.assertEqual(command[-1], "/tmp/dashboard.url")

    def test_wait_for_local_web_start_falls_back_to_health_endpoint_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            url_path = temp_path / "dashboard.url"

            class FakeProcess:
                def poll(self):
                    return None

            tick_values = iter([0.0, 0.01, 0.02, 0.03])
            seen_urls = []

            def fake_healthcheck(url: str) -> bool:
                seen_urls.append(url)
                return url == "http://127.0.0.1:8001"

            url = workflow_cli.wait_for_local_web_start(
                FakeProcess(),
                preferred_port=8000,
                url_path=url_path,
                deadline_seconds=0.2,
                max_port_tries=3,
                healthcheck_fn=fake_healthcheck,
                monotonic_fn=lambda: next(tick_values, 0.5),
                sleep_fn=lambda _seconds: None,
            )

            self.assertEqual(url, "http://127.0.0.1:8001")
            self.assertEqual(
                url_path.read_text(encoding="utf-8").strip(),
                "http://127.0.0.1:8001",
            )
            self.assertEqual(
                seen_urls,
                [
                    "http://127.0.0.1:8000",
                    "http://127.0.0.1:8001",
                ],
            )

    def test_read_local_web_state_marks_stale_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            pid_path = temp_path / "dashboard.pid"
            url_path = temp_path / "dashboard.url"
            log_path = temp_path / "dashboard.log"
            pid_path.write_text("4321\n", encoding="utf-8")
            url_path.write_text("http://127.0.0.1:8005\n", encoding="utf-8")

            state = workflow_cli.read_local_web_state(
                pid_path=pid_path,
                url_path=url_path,
                log_path=log_path,
                is_running_fn=lambda pid: False,
            )

            self.assertEqual(state["pid"], 4321)
            self.assertFalse(state["running"])
            self.assertTrue(state["stale_pid"])
            self.assertEqual(state["url"], "http://127.0.0.1:8005")

    def test_read_pid_file_rejects_invalid_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = pathlib.Path(tmpdir) / "dashboard.pid"
            pid_path.write_text("abc\n", encoding="utf-8")

            self.assertIsNone(workflow_cli.read_pid_file(pid_path))

    def test_local_dashboard_process_pids_finds_only_project_dashboard(self):
        stdout = "\n".join(
            [
                f"111 /usr/bin/python3 -u {PROJECT_ROOT / 'workflow_dashboard.py'} --host 127.0.0.1 --port 8000",
                "222 /usr/bin/python3 -u /tmp/other/workflow_dashboard.py --host 127.0.0.1 --port 8000",
                "333 /usr/bin/python3 unrelated.py",
            ]
        )

        def fake_run(*_args, **_kwargs):
            return types.SimpleNamespace(returncode=0, stdout=stdout)

        self.assertEqual(
            workflow_cli.local_dashboard_process_pids(run_command=fake_run),
            [111],
        )

    def test_diarization_status_treats_empty_transcript_as_no_speech(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stderr_path = pathlib.Path(tmpdir) / "clip.err"
            stderr_path.write_text(
                "Traceback details\nRuntimeError: Whisper returned an empty transcript.\n",
                encoding="utf-8",
            )

            status, is_failure, summary = workflow_cli.diarization_status_from_result(1, stderr_path)

            self.assertEqual(status, "no_speech")
            self.assertFalse(is_failure)
            self.assertIn("empty transcript", summary)

    def test_diarization_selection_fails_when_any_real_item_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            audio_list = temp_path / "selected_audio.txt"
            output_dir = temp_path / "outputs"
            log_dir = temp_path / "logs"
            audio_list.write_text(
                "001_ok.wav\n002_failed.wav\n003_silent.wav\n",
                encoding="utf-8",
            )

            def fake_run(command, **kwargs):
                selected_audio = command[command.index("--audio") + 1]
                stderr_handle = kwargs["stderr"]
                if selected_audio == "002_failed.wav":
                    stderr_handle.write("RuntimeError: model failed\n")
                    return types.SimpleNamespace(returncode=1)
                if selected_audio == "003_silent.wav":
                    stderr_handle.write(
                        "RuntimeError: Whisper returned an empty transcript.\n"
                    )
                    return types.SimpleNamespace(returncode=1)
                return types.SimpleNamespace(returncode=0)

            original_run = workflow_cli.subprocess.run
            original_audio_dir = workflow_cli.AUDIO_DIR
            try:
                workflow_cli.subprocess.run = fake_run
                workflow_cli.AUDIO_DIR = temp_path / "audio_in"
                args = types.SimpleNamespace(
                    audio_list_file=str(audio_list),
                    backend="pyannote",
                    output_dir=str(output_dir),
                    log_dir=str(log_dir),
                    device="cuda",
                    whisper_model="tiny.en",
                    batch_size=1,
                    language=None,
                    pyannote_pipeline_model="pyannote/speaker-diarization-3.1",
                    pyannote_segmentation_model="",
                    source_separation=False,
                    skip_review=False,
                )

                exit_code = workflow_cli.cmd_run_diarization_selection(args)
            finally:
                workflow_cli.subprocess.run = original_run
                workflow_cli.AUDIO_DIR = original_audio_dir

            self.assertEqual(exit_code, 1)
            summary = (log_dir / "runtime_summary.tsv").read_text(encoding="utf-8")
            self.assertIn("002_failed.wav\tfailed\t", summary)
            self.assertIn("003_silent.wav\tno_speech\t", summary)


if __name__ == "__main__":
    unittest.main()
