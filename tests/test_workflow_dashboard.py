import contextlib
import io
import errno
import json
import os
import pathlib
import tempfile
import unittest
import wave
from types import SimpleNamespace
from typing import Optional
from urllib.parse import urlencode

import workflow_dashboard as workflow_web


@contextlib.contextmanager
def stub_auto_train_queue():
    """Stop dashboard.auto_train from spawning a real daemon thread mid-test.

    The thread races with TemporaryDirectory cleanup (mkdir's project dirs
    that the test then can't rmtree). Stub it out for any test that uses
    add_to_training so cleanup is deterministic.
    """

    from dashboard import auto_train as _auto_train

    captured: list[tuple[str, str]] = []

    def fake_queue(project_name, **kwargs):
        captured.append((kwargs.get("backend") or "", project_name))
        return True

    original = _auto_train.queue_auto_train
    _auto_train.queue_auto_train = fake_queue
    try:
        yield captured
    finally:
        _auto_train.queue_auto_train = original


def wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    payload = b"\x00\x00" * frame_count
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)
    return buffer.getvalue()


def run_wsgi(
    app,
    *,
    method: str,
    path: str,
    body: bytes = b"",
    content_type: str = "",
    query: str = "",
    environ_overrides: Optional[dict] = None,
):
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": query,
        "CONTENT_LENGTH": str(len(body)),
        "CONTENT_TYPE": content_type,
        "SERVER_NAME": "localhost",
        "SERVER_PORT": "80",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(body),
        "wsgi.errors": io.StringIO(),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }
    if environ_overrides:
        environ.update(environ_overrides)
    response_body = b"".join(app(environ, start_response))
    return captured["status"], captured["headers"], response_body


def page_state(body: bytes) -> dict:
    html = body.decode("utf-8")
    marker = '<script id="dashboard-state" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def multipart_body(fields, files):
    boundary = "----DashboardTestBoundary"
    chunks = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, filename, payload, content_type in files:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                payload,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    return body, f"multipart/form-data; boundary={boundary}"


class WorkflowWebTests(unittest.TestCase):
    def test_resolve_server_mode_prefers_threaded_when_waitress_is_missing(self):
        original = workflow_web.waitress_available
        try:
            workflow_web.waitress_available = lambda: False
            self.assertEqual(workflow_web.resolve_server_mode("auto"), "threaded")
        finally:
            workflow_web.waitress_available = original

    def test_resolve_server_mode_prefers_waitress_when_available(self):
        original = workflow_web.waitress_available
        try:
            workflow_web.waitress_available = lambda: True
            self.assertEqual(workflow_web.resolve_server_mode("auto"), "waitress")
        finally:
            workflow_web.waitress_available = original

    def test_resolve_server_mode_rejects_missing_explicit_waitress(self):
        original = workflow_web.waitress_available
        try:
            workflow_web.waitress_available = lambda: False
            with self.assertRaisesRegex(ValueError, "Waitress is not installed"):
                workflow_web.resolve_server_mode("waitress")
        finally:
            workflow_web.waitress_available = original

    def test_bind_server_retries_when_requested_port_is_busy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            app = workflow_web.WorkflowWebApp(root=root)
            calls = []

            class FakeServer:
                def __init__(self, port):
                    self.server_port = port

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            def fake_make_server(host, port, application):
                calls.append((host, port, application))
                if port == 8000:
                    raise OSError(errno.EADDRINUSE, "Address already in use")
                return FakeServer(port)

            server, actual_port = workflow_web.bind_server(
                "127.0.0.1",
                8000,
                app,
                max_port_tries=3,
                make_server_fn=fake_make_server,
            )

            self.assertEqual(actual_port, 8001)
            self.assertEqual(calls[0][1], 8000)
            self.assertEqual(calls[1][1], 8001)
            self.assertEqual(server.server_port, 8001)

    def test_bind_server_uses_alternative_server_port_attribute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            app = workflow_web.WorkflowWebApp(root=root)

            class FakeServer:
                def __init__(self, port):
                    self.effective_port = port + 2

            server, actual_port = workflow_web.bind_server(
                "127.0.0.1",
                8000,
                app,
                make_server_fn=lambda host, port, application: FakeServer(port),
            )

            self.assertEqual(actual_port, 8002)
            self.assertEqual(server.effective_port, 8002)

    def test_dashboard_renders_for_empty_temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            html = body.decode("utf-8")
            self.assertIn("ML Speech Diarization", html)
            self.assertIn("/assets/static/js/dashboard_app.js", html)
            self.assertIn("/assets/static/css/dashboard.css", html)
            state = page_state(body)
            self.assertEqual(state["currentPath"], "/")
            self.assertIn("/youtube", [item["path"] for item in state["navItems"]])
            self.assertNotIn("/outputs", [item["path"] for item in state["navItems"]])
            self.assertEqual(
                [item["path"] for item in state["navItems"]],
                ["/", "/uploads", "/stitching", "/youtube", "/diarization", "/training-labels", "/fine-tuning"],
            )

    def test_stitching_tab_creates_rttm_and_training_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir, stub_auto_train_queue():
            root = pathlib.Path(tmpdir)
            audio_set = root / "audio_in" / "clips"
            audio_set.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (audio_set / "001_child.wav").write_bytes(wav_bytes(1.0))
            (audio_set / "002_adult.wav").write_bytes(wav_bytes(2.0))

            app = workflow_web.WorkflowWebApp(root=root)
            status, _, body = run_wsgi(app, method="GET", path="/stitching")
            self.assertEqual(status, "200 OK")
            state = page_state(body)
            self.assertEqual(state["currentPath"], "/stitching")
            self.assertEqual(state["routes"]["stitchAudio"], "/actions/stitch-audio")
            self.assertEqual(state["routes"]["renameStitching"], "/stitching/rename")
            self.assertEqual(len(state["context"]["audioFiles"]), 2)

            fields = [
                ("stitch_name", "stitch-demo"),
                ("stitch_seed", "fixed-seed"),
                ("add_to_training", "1"),
                ("fine_tuning_backend", "pyannote"),
                ("project_name", "stitch-test"),
                ("selected_audio", "clips/001_child.wav"),
                ("speaker_labels", "Speaker_0"),
                ("selected_audio", "clips/002_adult.wav"),
                ("speaker_labels", "Speaker_1"),
            ]
            post_status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/actions/stitch-audio",
                body=urlencode(fields).encode("utf-8"),
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(post_status, "303 See Other")
            self.assertIn("/stitching", headers["Location"])
            run_dirs = sorted((root / "stitched").iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            wav_path = next(run_dir.glob("*.wav"))
            rttm_path = next(run_dir.glob("*.rttm"))
            manifest_path = next(run_dir.glob("*_segments.tsv"))
            review_path = next(run_dir.glob("*_review.html"))
            self.assertTrue(wav_path.is_file())
            self.assertTrue(rttm_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(review_path.is_file())

            rttm_lines = rttm_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rttm_lines), 2)
            rows = []
            for line in rttm_lines:
                parts = line.split()
                rows.append((float(parts[3]), float(parts[4]), parts[7]))
            self.assertEqual(rows[0][0], 0.0)
            self.assertAlmostEqual(rows[1][0], rows[0][0] + rows[0][1], places=3)
            self.assertEqual({row[2] for row in rows}, {"Speaker_0", "Speaker_1"})

            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "stitch-test"
            self.assertEqual(len(list((project_dir / "audio").glob("*.wav"))), 1)
            self.assertEqual(len(list((project_dir / "rttm").glob("*.rttm"))), 1)

            # Mirror should land in audio_in/audioStitching/<run-name>/ so
            # the Media Library + the diarization tab + the stitching tab
            # all surface the new audio without manual intervention.
            mirror_dir = root / "audio_in" / "audioStitching" / run_dir.name
            self.assertTrue(mirror_dir.is_dir())
            self.assertTrue(any(mirror_dir.glob("*.wav")))
            self.assertTrue(any(mirror_dir.glob("*.rttm")))

            # Pre-completed label record so the Training Labels page treats
            # the stitched sample as already labeled.
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            mirror_audio_key = next(
                (key for key in label_status.get("items", {}) if key.startswith(f"audioStitching/{run_dir.name}/")),
                None,
            )
            self.assertIsNotNone(mirror_audio_key)
            entry = label_status["items"][mirror_audio_key]
            self.assertEqual(entry["status"], "completed")
            self.assertEqual(entry["source"], "audio_stitching")
            self.assertIn("pyannote/stitch-test", entry["target_projects"])
            self.assertIn(" ", entry["label_segments"])  # at least one timing line

    def test_stitching_tab_can_rename_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_set = root / "audio_in" / "clips"
            audio_set.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (audio_set / "001_child.wav").write_bytes(wav_bytes(1.0))
            (audio_set / "002_adult.wav").write_bytes(wav_bytes(1.0))

            app = workflow_web.WorkflowWebApp(root=root)
            fields = [
                ("stitch_name", "Age Class Mix"),
                ("stitch_seed", "fixed-seed"),
                ("selected_audio", "clips/001_child.wav"),
                ("speaker_labels", "Speaker_1"),
                ("selected_audio", "clips/002_adult.wav"),
                ("speaker_labels", "Speaker_0"),
            ]
            post_status, _, _ = run_wsgi(
                app,
                method="POST",
                path="/actions/stitch-audio",
                body=urlencode(fields).encode("utf-8"),
                content_type="application/x-www-form-urlencoded",
            )
            self.assertEqual(post_status, "303 See Other")
            run_dir = next((root / "stitched").iterdir())
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["output_name"], "age-class-mix")
            self.assertEqual(metadata["display_name"], "Age Class Mix")

            rename_status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/stitching/rename",
                body=urlencode(
                    [
                        ("run_dir", str(run_dir.relative_to(root))),
                        ("display_name", "Adult Child Infant Set"),
                    ]
                ).encode("utf-8"),
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(rename_status, "303 See Other")
            self.assertIn("/stitching", headers["Location"])
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["output_name"], "age-class-mix")
            self.assertEqual(metadata["display_name"], "Adult Child Infant Set")
            status, _, body = run_wsgi(app, method="GET", path="/stitching")
            self.assertEqual(status, "200 OK")
            state = page_state(body)
            self.assertEqual(state["context"]["stitching"]["rows"][0]["displayName"], "Adult Child Infant Set")
            self.assertEqual(state["context"]["stitching"]["rows"][0]["outputName"], "age-class-mix")

    def test_finetune_upload_can_fan_out_to_multiple_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_set = root / "audio_in" / "clips"
            audio_set.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            wav = audio_set / "001_clip.wav"
            rttm_dir = root / "fine_tuning" / "label_work"
            rttm_dir.mkdir(parents=True)
            rttm_file = rttm_dir / "001_clip.rttm"
            wav.write_bytes(wav_bytes(2.0))
            rttm_file.write_text(
                "SPEAKER 001_clip 1 0.000 1.000 <NA> <NA> Speaker_1 <NA> <NA>\n",
                encoding="utf-8",
            )
            (root / "fine_tuning" / "projects" / "pyannote" / "alpha").mkdir(parents=True)
            (root / "fine_tuning" / "projects" / "nemo" / "beta").mkdir(parents=True)

            app = workflow_web.WorkflowWebApp(root=root)
            fields = [
                # Existing-project checklist picks (different backends).
                ("training_targets", "pyannote/alpha"),
                ("training_targets", "nemo/beta"),
                # And a manual additional NEW project named gamma on pyannote.
                ("fine_tuning_backend", "pyannote"),
                ("project_name", "gamma"),
                ("server_audio_paths", "audio_in/clips/001_clip.wav"),
                ("server_rttm_paths", "fine_tuning/label_work/001_clip.rttm"),
            ]
            post_status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/fine-tuning/upload-sample",
                body=urlencode(fields).encode("utf-8"),
                content_type="application/x-www-form-urlencoded",
            )
            self.assertEqual(post_status, "303 See Other")
            location = headers["Location"]
            self.assertIn("/fine-tuning", location)
            self.assertIn("status=success", location)

            for backend, project in [("pyannote", "alpha"), ("nemo", "beta"), ("pyannote", "gamma")]:
                project_dir = root / "fine_tuning" / "projects" / backend / project
                self.assertEqual(
                    len(list((project_dir / "audio").glob("*.wav"))),
                    1,
                    msg=f"audio missing in {backend}/{project}",
                )
                self.assertEqual(
                    len(list((project_dir / "rttm").glob("*.rttm"))),
                    1,
                    msg=f"rttm missing in {backend}/{project}",
                )

    def test_stitching_run_can_be_deleted_and_clears_training_copies(self):
        with tempfile.TemporaryDirectory() as tmpdir, stub_auto_train_queue():
            root = pathlib.Path(tmpdir)
            audio_set = root / "audio_in" / "clips"
            audio_set.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (audio_set / "001_child.wav").write_bytes(wav_bytes(1.0))
            (audio_set / "002_adult.wav").write_bytes(wav_bytes(1.0))
            (root / "fine_tuning" / "projects" / "pyannote" / "del-test").mkdir(parents=True)

            app = workflow_web.WorkflowWebApp(root=root)
            fields = [
                ("stitch_name", "delete-me"),
                ("stitch_seed", "fixed"),
                ("add_to_training", "1"),
                ("training_targets", "pyannote/del-test"),
                ("selected_audio", "clips/001_child.wav"),
                ("speaker_labels", "Speaker_0"),
                ("selected_audio", "clips/002_adult.wav"),
                ("speaker_labels", "Speaker_1"),
            ]
            run_wsgi(
                app,
                method="POST",
                path="/actions/stitch-audio",
                body=urlencode(fields).encode("utf-8"),
                content_type="application/x-www-form-urlencoded",
            )
            run_dir = next((root / "stitched").iterdir())
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "del-test"
            self.assertTrue(any((project_dir / "audio").iterdir()))
            self.assertTrue(any((project_dir / "rttm").iterdir()))

            del_status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/stitching/delete",
                body=urlencode([("run_dir", str(run_dir.relative_to(root)))]).encode("utf-8"),
                content_type="application/x-www-form-urlencoded",
            )
            self.assertEqual(del_status, "303 See Other")
            self.assertIn("/stitching", headers["Location"])
            self.assertFalse(run_dir.exists())
            # The pushed sample copies should be gone too — leaving them
            # behind would make the project look like it still owns the
            # stitched sample.
            self.assertFalse(any((project_dir / "audio").iterdir()))
            self.assertFalse(any((project_dir / "rttm").iterdir()))

    def test_stitching_run_queues_auto_train_for_each_training_target(self):
        with tempfile.TemporaryDirectory() as tmpdir, stub_auto_train_queue() as queued_calls:
            root = pathlib.Path(tmpdir)
            audio_set = root / "audio_in" / "clips"
            audio_set.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (audio_set / "001_child.wav").write_bytes(wav_bytes(1.0))
            (audio_set / "002_adult.wav").write_bytes(wav_bytes(1.0))
            (root / "fine_tuning" / "projects" / "pyannote" / "auto-target").mkdir(parents=True)
            (root / "fine_tuning" / "projects" / "nemo" / "auto-target").mkdir(parents=True)

            app = workflow_web.WorkflowWebApp(root=root)
            fields = [
                ("stitch_name", "auto-train-demo"),
                ("stitch_seed", "fixed"),
                ("add_to_training", "1"),
                ("training_targets", "pyannote/auto-target"),
                ("training_targets", "nemo/auto-target"),
                ("selected_audio", "clips/001_child.wav"),
                ("speaker_labels", "Speaker_0"),
                ("selected_audio", "clips/002_adult.wav"),
                ("speaker_labels", "Speaker_1"),
            ]
            run_wsgi(
                app,
                method="POST",
                path="/actions/stitch-audio",
                body=urlencode(fields).encode("utf-8"),
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(
                sorted(queued_calls),
                sorted([("nemo", "auto-target"), ("pyannote", "auto-target")]),
            )
            run_dir = next((root / "stitched").iterdir())
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata.get("auto_train_triggered"))
            self.assertEqual(
                sorted(metadata.get("auto_train_targets") or []),
                sorted(["nemo/auto-target", "pyannote/auto-target"]),
            )

    def test_stitching_tab_can_add_sample_to_multiple_training_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir, stub_auto_train_queue():
            root = pathlib.Path(tmpdir)
            audio_set = root / "audio_in" / "clips"
            audio_set.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (audio_set / "001_child.wav").write_bytes(wav_bytes(1.0))
            (audio_set / "002_adult.wav").write_bytes(wav_bytes(1.5))
            (root / "fine_tuning" / "projects" / "pyannote" / "existing-one").mkdir(parents=True)
            (root / "fine_tuning" / "projects" / "nemo" / "existing-two").mkdir(parents=True)

            app = workflow_web.WorkflowWebApp(root=root)
            fields = [
                ("stitch_name", "multi-target-demo"),
                ("stitch_seed", "fixed-seed"),
                ("add_to_training", "1"),
                ("training_targets", "pyannote/existing-one"),
                ("training_targets", "nemo/existing-two"),
                ("selected_audio", "clips/001_child.wav"),
                ("speaker_labels", "Speaker_0"),
                ("selected_audio", "clips/002_adult.wav"),
                ("speaker_labels", "Speaker_1"),
            ]
            post_status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/actions/stitch-audio",
                body=urlencode(fields).encode("utf-8"),
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(post_status, "303 See Other")
            self.assertIn("/stitching", headers["Location"])
            run_dir = next((root / "stitched").iterdir())
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["training_targets"], ["pyannote/existing-one", "nemo/existing-two"])
            self.assertEqual(
                [item["project_key"] for item in metadata["training_usage"]],
                ["pyannote/existing-one", "nemo/existing-two"],
            )
            for backend, project in [("pyannote", "existing-one"), ("nemo", "existing-two")]:
                project_dir = root / "fine_tuning" / "projects" / backend / project
                self.assertEqual(len(list((project_dir / "audio").glob("*.wav"))), 1)
                self.assertEqual(len(list((project_dir / "rttm").glob("*.rttm"))), 1)

    def test_frontend_assets_are_served_from_organized_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            js_status, js_headers, js_body = run_wsgi(
                app,
                method="GET",
                path="/assets/static/js/dashboard_app.js",
            )
            css_status, css_headers, css_body = run_wsgi(
                app,
                method="GET",
                path="/assets/static/css/dashboard.css",
            )
            old_status, _, _ = run_wsgi(
                app,
                method="GET",
                path="/assets/static/dashboard_app.js",
            )

            self.assertEqual(js_status, "200 OK")
            self.assertIn("javascript", js_headers["Content-Type"])
            self.assertIn(b"ReactDOM.createRoot", js_body)
            self.assertEqual(css_status, "200 OK")
            self.assertEqual(css_headers["Content-Type"], "text/css")
            self.assertIn(b".workspace", css_body)
            self.assertEqual(old_status, "404 Not Found")

    def test_file_route_serves_public_artifacts_and_blocks_runtime_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            secrets_dir = root / ".local_dashboard"
            secrets_dir.mkdir()
            (secrets_dir / "secrets.env").write_text("HF_TOKEN=hf_secret\n", encoding="utf-8")
            (root / "workflow_dashboard.py").write_text("print('not public')\n", encoding="utf-8")

            app = workflow_web.WorkflowWebApp(root=root)
            public_status, public_headers, public_body = run_wsgi(
                app,
                method="GET",
                path="/files/audio_in/001_clip.wav",
            )
            secret_status, _, _ = run_wsgi(
                app,
                method="GET",
                path="/files/.local_dashboard/secrets.env",
            )
            source_status, _, _ = run_wsgi(
                app,
                method="GET",
                path="/files/workflow_dashboard.py",
            )
            cached_status, cached_headers, cached_body = run_wsgi(
                app,
                method="GET",
                path="/files/audio_in/001_clip.wav",
                environ_overrides={"HTTP_IF_NONE_MATCH": public_headers["ETag"]},
            )

            self.assertEqual(public_status, "200 OK")
            # Audio gets a short browser cache so the labeling dialog stops
            # re-fetching the same WAV every time the user reopens it; other
            # artifact types still get no-store from _cache_control_for.
            self.assertEqual(public_headers["Cache-Control"], "private, max-age=3600")
            self.assertIn("ETag", public_headers)
            self.assertIn("Last-Modified", public_headers)
            self.assertEqual(public_headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(public_headers["X-Frame-Options"], "DENY")
            self.assertEqual(secret_status, "404 Not Found")
            self.assertEqual(source_status, "404 Not Found")
            self.assertTrue(public_body)
            self.assertEqual(cached_status, "304 Not Modified")
            self.assertEqual(cached_headers["ETag"], public_headers["ETag"])
            self.assertEqual(cached_body, b"")

    def test_file_route_supports_range_requests_for_media_seeking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            payload = wav_bytes()
            (audio_dir / "001_clip.wav").write_bytes(payload)
            initial_status, initial_headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="GET",
                path="/files/audio_in/001_clip.wav",
            )

            status, headers, body = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="GET",
                path="/files/audio_in/001_clip.wav",
                environ_overrides={
                    "HTTP_RANGE": "bytes=10-29",
                    "HTTP_IF_NONE_MATCH": initial_headers["ETag"],
                },
            )

            self.assertEqual(initial_status, "200 OK")
            self.assertEqual(status, "206 Partial Content")
            self.assertEqual(headers["Accept-Ranges"], "bytes")
            self.assertEqual(headers["Content-Range"], f"bytes 10-29/{len(payload)}")
            self.assertEqual(headers["Content-Length"], "20")
            self.assertEqual(body, payload[10:30])

    def test_artifact_preview_api_returns_text_preview_and_blocks_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            log_dir = root / "outputs" / "diarization_runs" / "nemo" / "run-01" / "logs"
            log_dir.mkdir(parents=True)
            log_path = log_dir / "stdout.log"
            log_path.write_text(
                "\n".join(f"line {index}" for index in range(1, 121)) + "\n",
                encoding="utf-8",
            )
            secrets_dir = root / ".local_dashboard"
            secrets_dir.mkdir()
            (secrets_dir / "secrets.env").write_text("HF_TOKEN=hf_secret\n", encoding="utf-8")

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(
                app,
                method="GET",
                path="/api/artifact-preview",
                query=urlencode({"path": "outputs/diarization_runs/nemo/run-01/logs/stdout.log"}),
            )
            secret_status, _, _ = run_wsgi(
                app,
                method="GET",
                path="/api/artifact-preview",
                query=urlencode({"path": ".local_dashboard/secrets.env"}),
            )

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
            payload = json.loads(body)
            self.assertEqual(payload["kind"], "text")
            self.assertEqual(payload["path"], "outputs/diarization_runs/nemo/run-01/logs/stdout.log")
            self.assertTrue(payload["previewText"].startswith("line 41"))
            self.assertIn("line 120", payload["previewText"])
            self.assertTrue(payload["truncated"])
            self.assertEqual(secret_status, "404 Not Found")

    def test_component_page_renders_for_uploads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/uploads")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            state = page_state(body)
            self.assertEqual(state["currentPath"], "/uploads")
            self.assertEqual(state["routes"]["uploadAudio"], "/upload/audio")
            self.assertEqual(state["defaults"]["uploadAudioAccept"], ".aac,.flac,.m4a,.mp3,.ogg,.opus,.wav,.wma")
            self.assertNotIn("/review", [item["path"] for item in state["navItems"]])

            review_status, _, review_body = run_wsgi(app, method="GET", path="/review")
            self.assertEqual(review_status, "200 OK")
            self.assertEqual(page_state(review_body)["currentPath"], "/uploads")

    def test_media_library_reflects_youtube_wav_created_after_initial_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            app = workflow_web.WorkflowWebApp(root=root)

            status, _, body = run_wsgi(app, method="GET", path="/uploads")
            self.assertEqual(status, "200 OK")
            self.assertEqual(page_state(body)["context"]["audioFiles"], [])

            youtube_folder = root / "audio_in" / "youtube_links"
            youtube_folder.mkdir(parents=True, exist_ok=True)
            (youtube_folder / "001_converted.wav").write_bytes(wav_bytes())

            status, _, body = run_wsgi(app, method="GET", path="/uploads")
            self.assertEqual(status, "200 OK")
            audio_files = page_state(body)["context"]["audioFiles"]
            self.assertEqual([row["name"] for row in audio_files], ["youtube_links/001_converted.wav"])

    def test_diarization_alias_routes_render_selection_page(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)

            for path in ("/diarization", "/jobs", "/direct"):
                status, headers, body = run_wsgi(app, method="GET", path=path)
                self.assertEqual(status, "200 OK")
                self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
                state = page_state(body)
                self.assertEqual(state["currentPath"], "/diarization")
                self.assertEqual(state["routes"]["runDiarization"], "/actions/run-diarization")
                self.assertIn("/diarization", [item["path"] for item in state["navItems"]])

    def test_diarization_page_renders_selection_toolbar_and_run_monitor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            run_dir = root / "outputs" / "diarization_runs" / "20260414T190000_diarization_nemo_01-items"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
                encoding="utf-8",
            )
            (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "selected_audio.txt").write_text("001_clip.wav\n", encoding="utf-8")
            (run_dir / "001_clip.txt").write_text("speaker transcript\n", encoding="utf-8")
            (run_dir / "001_clip.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
            (run_dir / "001_clip_review.html").write_text("<html>review</html>\n", encoding="utf-8")
            (run_dir / "001_clip_review_flags.tsv").write_text("row\tissue\n", encoding="utf-8")
            (run_dir / "logs" / "001_clip.wav.out").write_text("file stdout\n", encoding="utf-8")
            (run_dir / "logs" / "001_clip.wav.err").write_text("", encoding="utf-8")
            (run_dir / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\n"
                f"001_clip.wav\tok\t3.21\t{run_dir / 'logs' / '001_clip.wav.out'}\t{run_dir / 'logs' / '001_clip.wav.err'}\n",
                encoding="utf-8",
            )
            (run_dir / "stdout.log").write_text("processed 001_clip.wav\n", encoding="utf-8")
            (run_dir / "stderr.log").write_text("", encoding="utf-8")

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/diarization")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            state = page_state(body)
            self.assertEqual(state["context"]["audioFiles"][0]["name"], "001_clip.wav")
            latest_run = state["context"]["diarization"]["latestRun"]
            self.assertEqual(latest_run["status"], "succeeded")
            self.assertEqual(latest_run["summaryRows"][0]["audio_file"], "001_clip.wav")
            self.assertIn("runtime_summary.tsv", [link["label"] for link in latest_run["artifactLinks"]])

            media_status, media_headers, media_body = run_wsgi(app, method="GET", path="/uploads")
            self.assertEqual(media_status, "200 OK")
            self.assertEqual(media_headers["Content-Type"], "text/html; charset=utf-8")
            media_history = page_state(media_body)["context"]["diarization"]["history"]
            item_links = [link["label"] for link in media_history[0]["links"]]
            self.assertEqual(media_history[0]["audioHref"], "/files/audio_in/001_clip.wav")
            self.assertEqual(media_history[0]["fileName"], "001_clip.wav")
            self.assertEqual(media_history[0]["folder"], "Unsorted Root")
            self.assertIn("Source Audio", item_links)
            self.assertIn("Transcript", item_links)
            self.assertIn("Diarized Times", item_links)
            self.assertIn("Review Page", item_links)

    def test_review_file_route_refreshes_playback_and_label_ui(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            run_dir = root / "outputs" / "diarization_runs" / "20260414T190000_diarization_nemo_01-items"
            run_dir.mkdir(parents=True)
            (run_dir / "001_clip.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nSpeaker 0: hello\n",
                encoding="utf-8",
            )
            review_path = run_dir / "001_clip_review.html"
            review_path.write_text("<html>old review</html>\n", encoding="utf-8")
            (root / "fine_tuning").mkdir()
            (root / "fine_tuning" / "label_status.json").write_text(
                json.dumps(
                    {
                        "items": {
                            "001_clip.wav": {
                                "status": "draft",
                                "backend": "pyannote",
                                "project_name": "existing-review-project",
                                "label_segments": "0.200 0.900 Saved_Speaker",
                                "transcript_text": "saved transcript note",
                                "issue_questions": "needs a speaker check",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            status, headers, body = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="GET",
                path="/files/outputs/diarization_runs/20260414T190000_diarization_nemo_01-items/001_clip_review.html",
            )

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html")
            html = body.decode("utf-8")
            self.assertIn("playRange", html)
            self.assertIn("Training Labels", html)
            self.assertIn("Detected Segments", html)
            self.assertIn(">Back</button>", html)
            self.assertIn("Saved_Speaker", html)
            self.assertIn("existing-review-project", html)
            self.assertIn("saved transcript note", html)
            self.assertIn("id=\"startAtSegment\"", html)
            self.assertIn("id=\"labelIssueFilter\"", html)
            self.assertIn("class=\"use-cue\"", html)
            self.assertIn("id=\"waveformCanvas\"", html)
            self.assertIn("id=\"playbackRate\"", html)
            self.assertIn("Dialogue</th>", html)
            self.assertIn("Speech</th>", html)
            self.assertIn("id=\"showLabelDialogue\"", html)
            self.assertNotIn("Adjust selected label", html)
            self.assertNotIn("Transcript / notes", html)
            self.assertNotIn("Open questions", html)
            self.assertIn("training-labels/save", html)
            self.assertIn("001_clip.wav", html)

    def test_review_page_label_save_returns_to_review_and_tracks_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            run_dir = root / "outputs" / "diarization_runs" / "20260414T190000_diarization_nemo_01-items"
            run_dir.mkdir(parents=True)
            review_path = run_dir / "001_clip_review.html"
            review_path.write_text("<html>review</html>\n", encoding="utf-8")
            return_to = "/files/outputs/diarization_runs/20260414T190000_diarization_nemo_01-items/001_clip_review.html"

            body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "review-lab"),
                    ("label_segments", "0.00 0.50 Speaker_0"),
                    ("label_action", "complete"),
                    ("label_return_to", return_to),
                    ("label_source", "review_page"),
                    ("label_review_path", str(review_path)),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertTrue(headers["Location"].startswith(return_to))
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_clip.wav"]
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["source"], "review_page")
            self.assertEqual(record["review_path"], "outputs/diarization_runs/20260414T190000_diarization_nemo_01-items/001_clip_review.html")

    def test_review_page_label_save_persists_new_manual_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            nested_audio_dir = audio_dir / "field_uploads"
            nested_audio_dir.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (nested_audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            run_dir = root / "outputs" / "diarization_runs" / "20260414T190000_diarization_nemo_01-items"
            run_dir.mkdir(parents=True)
            (run_dir / "field_uploads__001_clip.srt").write_text(
                "1\n00:00:00,000 --> 00:00:00,500\nSpeaker 0: hello\n",
                encoding="utf-8",
            )
            review_path = run_dir / "field_uploads__001_clip_review.html"
            review_path.write_text("<html>review</html>\n", encoding="utf-8")
            return_to = "/files/outputs/diarization_runs/20260414T190000_diarization_nemo_01-items/field_uploads__001_clip_review.html?message=old&status=success"
            clean_return_to = return_to.split("?", 1)[0]
            manual_segments = "0.00 0.50 Speaker_0\n0.60 0.90 Speaker_1"

            body = urlencode(
                [
                    ("audio_file", "field_uploads/001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "review-lab"),
                    ("label_segments", manual_segments),
                    ("label_action", "complete"),
                    ("label_return_to", return_to),
                    ("label_source", "review_page"),
                    ("label_review_path", str(review_path)),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertTrue(headers["Location"].startswith(clean_return_to))
            self.assertNotIn("old", headers["Location"])
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            self.assertEqual(
                label_status["items"]["field_uploads/001_clip.wav"]["label_segments"],
                "0.000 0.500 Speaker_0\n0.600 0.900 Speaker_1",
            )
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "review-lab"
            # Audio and RTTM are written under the same path-flattened stem so
            # two files named ``001_clip.wav`` in different subfolders can both
            # land in the same project without overwriting each other.
            self.assertTrue((project_dir / "audio" / "field_uploads__001_clip.wav").is_file())
            self.assertTrue((project_dir / "rttm" / "field_uploads__001_clip.rttm").is_file())

            status, _, body = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="GET",
                path=clean_return_to,
            )

            self.assertEqual(status, "200 OK")
            html = body.decode("utf-8")
            self.assertEqual(html.count("<tr data-label-row"), 2)
            self.assertIn("Speaker_1", html)
            self.assertIn("<span class=\"row-number\">2</span>", html)
            self.assertIn('name="audio_file" value="field_uploads/001_clip.wav"', html)

            status, _, body = run_wsgi(workflow_web.WorkflowWebApp(root=root), method="GET", path="/fine-tuning")
            self.assertEqual(status, "200 OK")
            project = page_state(body)["context"]["projects"][0]
            self.assertEqual(project["backend"], "pyannote")
            self.assertEqual(project["slug"], "review-lab")
            self.assertEqual(project["sampleCount"], 1)

    def test_review_page_keeps_dialogue_toggle_off_after_saved_draft(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            run_dir = root / "outputs" / "diarization_runs" / "20260414T190000_diarization_nemo_01-items"
            run_dir.mkdir(parents=True)
            (run_dir / "001_clip.srt").write_text(
                "1\n00:00:00,000 --> 00:00:00,500\nSpeaker 0: hello\n",
                encoding="utf-8",
            )
            review_path = run_dir / "001_clip_review.html"
            review_path.write_text("<html>review</html>\n", encoding="utf-8")
            return_to = "/files/outputs/diarization_runs/20260414T190000_diarization_nemo_01-items/001_clip_review.html"

            body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "review-lab"),
                    ("label_segments", "0.00 0.50 Speaker_0"),
                    ("label_transcript_text", "this should stay excluded"),
                    ("label_include_transcript", "0"),
                    ("label_action", "draft"),
                    ("label_return_to", return_to),
                    ("label_source", "review_page"),
                    ("label_review_path", str(review_path)),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertTrue(headers["Location"].startswith(return_to))
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_clip.wav"]
            self.assertFalse(record["include_transcript"])
            self.assertEqual(record["transcript_text"], "")

            status, _, body = run_wsgi(workflow_web.WorkflowWebApp(root=root), method="GET", path=return_to)
            self.assertEqual(status, "200 OK")
            html = body.decode("utf-8")
            self.assertIn('id="labelIncludeTranscript" name="label_include_transcript" value="0"', html)
            self.assertIn('id="showLabelDialogue" type="checkbox"> <span id="labelDialogueToggleText">Show Dialogue</span>', html)
            self.assertIn('<table class="label-table hide-dialogue">', html)

    def test_diarization_page_tracks_completed_and_retryable_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_done.wav").write_bytes(wav_bytes())
            (audio_dir / "002_retry.wav").write_bytes(wav_bytes())
            (audio_dir / "003_new.wav").write_bytes(wav_bytes())

            run_dir = root / "outputs" / "diarization_runs" / "20260417T120000_diarization_nemo_02-items"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 2, "pid": 0}',
                encoding="utf-8",
            )
            (run_dir / "exit_code.txt").write_text("1\n", encoding="utf-8")
            (run_dir / "selected_audio.txt").write_text("001_done.wav\n002_retry.wav\n", encoding="utf-8")
            (run_dir / "001_done.txt").write_text("speaker transcript\n", encoding="utf-8")
            (run_dir / "001_done.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
            (run_dir / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_done.wav\tok\t2.00\t\t\t\n"
                "002_retry.wav\tfailed\t1.00\t\t\tRuntimeError: model failed\n",
                encoding="utf-8",
            )

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/diarization")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            diarization = page_state(body)["context"]["diarization"]
            self.assertEqual(diarization["librarySummary"]["diarized"], 1)
            self.assertEqual(diarization["librarySummary"]["retry"], 1)
            self.assertEqual(diarization["librarySummary"]["ready"], 1)
            rows = {row["name"]: row for row in diarization["audioRows"]}
            self.assertEqual(rows["001_done.wav"]["state_class"], "converted")
            self.assertEqual(rows["001_done.wav"]["selection_ready"], "no")
            model_statuses = {
                item["backend"]: item
                for item in rows["001_done.wav"]["modelStatuses"]
            }
            self.assertEqual(model_statuses["nemo"]["status"], "already diarized")
            self.assertEqual(model_statuses["pyannote"]["status"], "not run")
            self.assertEqual(rows["001_done.wav"]["selectionByModel"]["nemo"], "no")
            self.assertEqual(rows["001_done.wav"]["selectionByModel"]["pyannote"], "yes")
            self.assertEqual(rows["002_retry.wav"]["state_class"], "failed")
            self.assertEqual(rows["002_retry.wav"]["selection_ready"], "yes")
            self.assertEqual(rows["003_new.wav"]["state_class"], "ready")
            model_counts = {
                item["key"]: item
                for item in diarization["librarySummary"]["modelCounts"]
            }
            self.assertEqual(model_counts["nemo"]["diarized"], 1)
            self.assertEqual(model_counts["pyannote"]["ready"], 3)

            upload_status, _, upload_body = run_wsgi(app, method="GET", path="/uploads")
            self.assertEqual(upload_status, "200 OK")
            media_history = page_state(upload_body)["context"]["diarization"]["history"]
            self.assertEqual(media_history[0]["audioFile"], "001_done.wav")
            self.assertIn("Diarized Times", [link["label"] for link in media_history[0]["links"]])
            transcript_link = next(link for link in media_history[0]["links"] if link["label"] == "Transcript")
            self.assertEqual(transcript_link["kind"], "text")
            self.assertIn("/api/artifact-preview?", transcript_link["previewHref"])

    def test_diarization_page_marks_partial_results_and_no_speech_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_silent.wav").write_bytes(wav_bytes())
            (audio_dir / "002_clip.wav").write_bytes(wav_bytes())

            run_dir = root / "outputs" / "diarization_runs" / "20260414T191000_diarization_nemo_02-items"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 2, "pid": 0}',
                encoding="utf-8",
            )
            (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "selected_audio.txt").write_text("001_silent.wav\n002_clip.wav\n", encoding="utf-8")
            (run_dir / "002_clip.txt").write_text("speaker transcript\n", encoding="utf-8")
            (run_dir / "002_clip.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
            (run_dir / "002_clip_review.html").write_text("<html>review</html>\n", encoding="utf-8")
            (run_dir / "logs" / "001_silent.wav.out").write_text("", encoding="utf-8")
            (run_dir / "logs" / "001_silent.wav.err").write_text(
                "RuntimeError: Whisper returned an empty transcript.\n",
                encoding="utf-8",
            )
            (run_dir / "logs" / "002_clip.wav.out").write_text("ok\n", encoding="utf-8")
            (run_dir / "logs" / "002_clip.wav.err").write_text("", encoding="utf-8")
            (run_dir / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                f"001_silent.wav\tno_speech\t1.23\t{run_dir / 'logs' / '001_silent.wav.out'}\t{run_dir / 'logs' / '001_silent.wav.err'}\tRuntimeError: Whisper returned an empty transcript.\n"
                f"002_clip.wav\tok\t3.21\t{run_dir / 'logs' / '002_clip.wav.out'}\t{run_dir / 'logs' / '002_clip.wav.err'}\t\n",
                encoding="utf-8",
            )
            (run_dir / "stdout.log").write_text("Processed 2 audio file(s).\n", encoding="utf-8")
            (run_dir / "stderr.log").write_text("", encoding="utf-8")

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/diarization")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            latest_run = page_state(body)["context"]["diarization"]["latestRun"]
            self.assertEqual(latest_run["status"], "completed_with_warnings")
            self.assertEqual(latest_run["noSpeechCount"], 1)
            self.assertEqual(latest_run["succeededCount"], 1)
            silent_item = latest_run["items"][0]
            self.assertEqual(silent_item["status"], "no_speech")
            self.assertIn("empty transcript", silent_item["errorSummary"])
            self.assertEqual(silent_item["audioHref"], "/files/audio_in/001_silent.wav")

    def test_diarization_page_exposes_slurm_queue_tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            run_dir = root / "outputs" / "diarization_runs" / "20260414T191500_diarization_nemo_01-items"
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "slurm_job_id": "12345"}',
                encoding="utf-8",
            )
            (run_dir / "selected_audio.txt").write_text("001_clip.wav\n", encoding="utf-8")

            original = workflow_web.slurm_queue_snapshot
            workflow_web.slurm_queue_snapshot = lambda job_id: {
                "job_id": job_id,
                "available": True,
                "state": "PENDING",
                "queue_position": 4,
                "jobs_ahead": 3,
                "reason": "Priority",
                "estimated_start": "2026-04-17T02:00:00",
                "message": "Pending position 4.",
            }
            try:
                status, headers, body = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="GET",
                    path="/diarization",
                )
                upload_status, _, upload_body = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="GET",
                    path="/uploads",
                )
            finally:
                workflow_web.slurm_queue_snapshot = original

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            latest_run = page_state(body)["context"]["diarization"]["latestRun"]
            self.assertEqual(latest_run["status"], "submitted")
            self.assertEqual(latest_run["slurmQueue"]["job_id"], "12345")
            self.assertEqual(latest_run["slurmQueue"]["queue_position"], 4)
            self.assertEqual(latest_run["slurmQueue"]["jobs_ahead"], 3)
            self.assertEqual(upload_status, "200 OK")
            upload_run = page_state(upload_body)["context"]["diarization"]["latestRun"]
            self.assertEqual(upload_run["slurmQueue"]["job_id"], "12345")

    def test_diarization_page_keeps_completed_outputs_when_newer_retry_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_done.wav").write_bytes(wav_bytes())

            completed_run = root / "outputs" / "diarization_runs" / "nemo" / "20260417T110000_diarization_nemo_01-items"
            (completed_run / "logs").mkdir(parents=True)
            (completed_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
                encoding="utf-8",
            )
            (completed_run / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (completed_run / "selected_audio.txt").write_text("001_done.wav\n", encoding="utf-8")
            (completed_run / "001_done.txt").write_text("speaker transcript\n", encoding="utf-8")
            (completed_run / "001_done.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
            (completed_run / "001_done_review.html").write_text("<html>review</html>\n", encoding="utf-8")
            (completed_run / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_done.wav\tok\t2.00\t\t\t\n",
                encoding="utf-8",
            )

            failed_run = root / "outputs" / "diarization_runs" / "nemo" / "20260417T120000_diarization_nemo_01-items"
            (failed_run / "logs").mkdir(parents=True)
            (failed_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
                encoding="utf-8",
            )
            (failed_run / "exit_code.txt").write_text("1\n", encoding="utf-8")
            (failed_run / "selected_audio.txt").write_text("001_done.wav\n", encoding="utf-8")
            (failed_run / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_done.wav\tfailed\t1.00\t\t\tRuntimeError: retry failed\n",
                encoding="utf-8",
            )

            os.utime(completed_run, (100.0, 100.0))
            os.utime(failed_run, (200.0, 200.0))

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/diarization")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            diarization = page_state(body)["context"]["diarization"]
            row = diarization["audioRows"][0]
            self.assertEqual(row["name"], "001_done.wav")
            self.assertEqual(row["state_class"], "converted")
            self.assertEqual(row["selection_ready"], "no")

            upload_status, _, upload_body = run_wsgi(app, method="GET", path="/uploads")
            self.assertEqual(upload_status, "200 OK")
            media_history = page_state(upload_body)["context"]["diarization"]["history"]
            self.assertEqual(media_history[0]["runName"], "20260417T110000_diarization_nemo_01-items")
            self.assertIn("Transcript", [link["label"] for link in media_history[0]["links"]])

    def test_diarization_tracking_counts_active_runs_across_backends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            active_run = root / "outputs" / "diarization_runs" / "nemo" / "20260414T191500_diarization_nemo_01-items"
            active_run.mkdir(parents=True)
            (active_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "slurm_job_id": "12345"}',
                encoding="utf-8",
            )
            (active_run / "selected_audio.txt").write_text("001_clip.wav\n", encoding="utf-8")

            completed_run = root / "outputs" / "diarization_runs" / "pyannote" / "20260414T192000_diarization_pyannote_01-items"
            (completed_run / "logs").mkdir(parents=True)
            (completed_run / "metadata.json").write_text(
                '{"backend": "pyannote", "selected_audio_count": 1, "slurm_job_id": "99999"}',
                encoding="utf-8",
            )
            (completed_run / "selected_audio.txt").write_text("001_clip.wav\n", encoding="utf-8")
            (completed_run / "exit_code.txt").write_text("0\n", encoding="utf-8")

            status, headers, body = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="GET",
                path="/diarization",
            )

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            context = page_state(body)["context"]
            self.assertEqual(context["diarization"]["latestRun"]["status"], "succeeded")
            self.assertTrue(context["tracking"]["shouldRefresh"])
            self.assertEqual(context["tracking"]["refreshIntervalMs"], workflow_web.LIVE_TRACKING_INTERVAL_MS)
            self.assertTrue(context["tracking"]["fingerprint"])
            self.assertEqual(context["tracking"]["activeDiarizationRuns"], 1)
            self.assertIn(active_run.name, context["tracking"]["diarizationRunNames"])

    def test_diarization_page_exposes_all_active_model_sbatch_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            run_specs = [
                ("nemo", "20260414T191500_diarization_nemo_01-items", "71111"),
                ("pyannote", "20260414T191501_diarization_pyannote_01-items", "72222"),
            ]
            for backend, run_name, job_id in run_specs:
                run_dir = root / "outputs" / "diarization_runs" / backend / run_name
                run_dir.mkdir(parents=True)
                (run_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "backend": backend,
                            "selected_audio_count": 1,
                            "slurm_job_id": job_id,
                            "batch_id": "batch-test",
                            "batch_size_total": 2,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "selected_audio.txt").write_text("001_clip.wav\n", encoding="utf-8")

            original = workflow_web.slurm_queue_snapshot
            workflow_web.slurm_queue_snapshot = lambda job_id: {
                "job_id": str(job_id),
                "available": True,
                "state": "PENDING",
                "queue_position": 1,
                "jobs_ahead": 0,
                "reason": "Priority",
                "estimated_start": "2026-04-17T02:00:00",
                "message": "Pending position 1.",
            }
            try:
                status, headers, body = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="GET",
                    path="/diarization",
                )
            finally:
                workflow_web.slurm_queue_snapshot = original

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            diarization = page_state(body)["context"]["diarization"]
            self.assertEqual(len(diarization["activeRuns"]), 2)
            job_ids = {run["slurmQueue"]["job_id"] for run in diarization["activeRuns"]}
            self.assertEqual(job_ids, {"71111", "72222"})
            batch_ids = {run["batchId"] for run in diarization["activeRuns"]}
            self.assertEqual(batch_ids, {"batch-test"})

    def test_live_tracking_api_reports_diarization_and_youtube_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (root / "youtube_links.txt").write_text("https://youtu.be/example-one\n", encoding="utf-8")
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            diarization_run = root / "outputs" / "diarization_runs" / "nemo" / "20260414T191500_diarization_nemo_01-items"
            diarization_run.mkdir(parents=True)
            (diarization_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "slurm_job_id": "12345"}',
                encoding="utf-8",
            )
            (diarization_run / "selected_audio.txt").write_text("001_clip.wav\n", encoding="utf-8")

            youtube_run = root / "outputs" / "youtube_conversion_runs" / "20260416T193000_youtube-conversion_01-items"
            youtube_run.mkdir(parents=True)
            (youtube_run / "metadata.json").write_text(
                '{"mode": "selected", "selected_url_count": 1, "slurm_job_id": "98765"}',
                encoding="utf-8",
            )
            (youtube_run / "selected_urls.txt").write_text("https://youtu.be/example-one\n", encoding="utf-8")

            original = workflow_web.slurm_queue_snapshot
            workflow_web.slurm_queue_snapshot = lambda job_id: {
                "job_id": str(job_id),
                "available": True,
                "state": "PENDING",
                "queue_position": 1,
                "jobs_ahead": 0,
                "reason": "Priority",
                "estimated_start": "2026-04-17T02:00:00",
                "message": "Pending position 1.",
            }
            try:
                status, headers, body = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="GET",
                    path="/api/tracking",
                )
            finally:
                workflow_web.slurm_queue_snapshot = original

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
            self.assertEqual(headers["Cache-Control"], "no-store")
            payload = json.loads(body)
            self.assertTrue(payload["fingerprint"])
            self.assertEqual(payload["tracking"]["refreshIntervalMs"], workflow_web.LIVE_TRACKING_INTERVAL_MS)
            self.assertTrue(payload["tracking"]["shouldRefresh"])
            self.assertEqual(payload["tracking"]["activeDiarizationRuns"], 1)
            self.assertEqual(payload["tracking"]["activeYoutubeRuns"], 1)
            self.assertEqual(payload["diarization"]["latestRun"]["status"], "submitted")
            self.assertEqual(payload["diarization"]["latestRun"]["slurmQueue"]["job_id"], "12345")
            self.assertEqual(payload["youtube"]["latestRun"]["status"], "submitted")
            self.assertEqual(payload["youtube"]["latestRun"]["slurmQueue"]["job_id"], "98765")

    def test_live_tracking_api_reports_fine_tuning_runs_and_workspace_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (root / "youtube_links.txt").write_text("https://youtu.be/example-one\n", encoding="utf-8")
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            (project_dir / "audio").mkdir(parents=True)
            (project_dir / "audio" / "001_clip.wav").write_bytes(wav_bytes())
            run_dir = project_dir / "runs" / "20260417T210000_speaker-lab-v1"
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                '{"submission_mode": "sbatch", "version_name": "speaker lab v1"}',
                encoding="utf-8",
            )

            status, headers, body = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="GET",
                path="/api/tracking",
            )

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
            payload = json.loads(body)
            tracking = payload["tracking"]
            self.assertTrue(tracking["shouldRefresh"])
            self.assertEqual(tracking["idleRefreshIntervalMs"], workflow_web.IDLE_TRACKING_INTERVAL_MS)
            self.assertEqual(tracking["activeFineTuningRuns"], 1)
            self.assertEqual(tracking["activeRunCount"], 1)
            self.assertEqual(tracking["audioInputCount"], 1)
            self.assertEqual(tracking["queuedLinks"], 1)
            self.assertEqual(tracking["fineTuningProjects"], 1)
            self.assertEqual(tracking["fineTuningRuns"][0]["project"], "speaker-lab")
            self.assertEqual(tracking["fineTuningRuns"][0]["status"], "submitted")

    def test_page_state_api_returns_requested_page_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            status, headers, body = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="GET",
                path="/api/page-state",
                query=urlencode({"path": "/uploads"}),
            )

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
            self.assertEqual(headers["Cache-Control"], "no-store")
            payload = json.loads(body)
            self.assertEqual(payload["currentPath"], "/uploads")
            self.assertEqual(payload["routes"]["pageState"], "/api/page-state")
            self.assertEqual(payload["context"]["audioCount"], 0)

    def test_page_state_api_preserves_selected_fine_tuned_diarization_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            (project_dir / "audio").mkdir(parents=True)
            (project_dir / "artifacts" / "experiments" / "checkpoints").mkdir(parents=True)
            (project_dir / "audio" / "sample.wav").write_bytes(wav_bytes())
            (project_dir / "artifacts" / "metadata.json").write_text(
                '{"warnings": []}',
                encoding="utf-8",
            )
            (project_dir / "artifacts" / "experiments" / "checkpoints" / "final.ckpt").write_text(
                "checkpoint\n",
                encoding="utf-8",
            )

            run_dir = root / "outputs" / "diarization_runs" / "pyannote" / "20260417T121500_diarization_pyannote_01-items"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "backend": "pyannote",
                        "model_key": "pyannote:speaker-lab",
                        "model_label": "pyannote fine-tuned / speaker-lab",
                        "selected_audio_count": 1,
                        "pid": 0,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "selected_audio.txt").write_text("001_clip.wav\n", encoding="utf-8")
            (run_dir / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_clip.wav\tok\t2.00\t\t\t\n",
                encoding="utf-8",
            )

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(
                app,
                method="GET",
                path="/api/page-state",
                query=urlencode(
                    {
                        "path": "/diarization",
                        "diarization_model_key": "pyannote:speaker-lab",
                    }
                ),
            )

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
            payload = json.loads(body)
            diarization = payload["context"]["diarization"]
            self.assertEqual(diarization["selectedModelKey"], "pyannote:speaker-lab")
            self.assertIn(
                "pyannote:speaker-lab",
                {item["key"] for item in diarization["modelOptions"]},
            )
            self.assertEqual(diarization["audioRows"][0]["targetModelKey"], "pyannote:speaker-lab")
            self.assertEqual(diarization["audioRows"][0]["state_class"], "converted")

            upload_status, _, upload_body = run_wsgi(app, method="GET", path="/uploads")
            self.assertEqual(upload_status, "200 OK")
            media_history = page_state(upload_body)["context"]["diarization"]["history"]
            self.assertEqual(media_history[0]["modelKey"], "pyannote:speaker-lab")
            self.assertEqual(media_history[0]["modelLabel"], "pyannote fine-tuned / speaker-lab")

    def test_models_route_aliases_to_diarization_model_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/models")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            state = page_state(body)
            self.assertEqual(state["currentPath"], "/diarization")
            self.assertEqual(state["routes"]["saveModels"], "/models/save")
            self.assertNotIn("/models", [item["path"] for item in state["navItems"]])
            self.assertIn("pyannote_fine_tuning", state["context"]["preferences"])

    def test_upload_page_skips_unneeded_output_and_project_scans(self):
        class GuardedApp(workflow_web.WorkflowWebApp):
            def recent_output_files(self, limit: int = 18):
                raise AssertionError("uploads page should not scan recent outputs")

            def recent_srt_files(self):
                raise AssertionError("uploads page should not scan subtitle outputs")

            def project_summaries(self):
                raise AssertionError("uploads page should not load full project summaries")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = GuardedApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/uploads")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            self.assertEqual(page_state(body)["currentPath"], "/uploads")

    def test_audio_upload_route_saves_and_renumbers_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            body, content_type = multipart_body(
                fields={},
                files=[("audio_files", "clip.wav", wav_bytes(), "audio/wav")],
            )

            status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/upload/audio",
                body=body,
                content_type=content_type,
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertIn("Upload+complete.", headers["Location"])
            self.assertIn("Saved+1+supported+audio+file", headers["Location"])
            self.assertTrue((root / "audio_in" / "001_clip.wav").exists())

    def test_audio_upload_route_reports_partial_success_with_notice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            body, content_type = multipart_body(
                fields={},
                files=[
                    ("audio_files", "clip.wav", wav_bytes(), "audio/wav"),
                    ("audio_files", "notes.txt", b"hello\n", "text/plain"),
                ],
            )

            status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/upload/audio",
                body=body,
                content_type=content_type,
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=info", headers["Location"])
            self.assertIn("Upload+complete.", headers["Location"])
            self.assertIn("Rejected+unsupported+file", headers["Location"])
            self.assertTrue((audio_dir / "001_clip.wav").exists())
            self.assertFalse((audio_dir / "notes.txt").exists())

    def test_audio_upload_route_rejects_unsupported_file_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            body, content_type = multipart_body(
                fields={},
                files=[("audio_files", "notes.txt", b"hello\n", "text/plain")],
            )

            status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/upload/audio",
                body=body,
                content_type=content_type,
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=error", headers["Location"])
            self.assertIn("Upload+failed.", headers["Location"])
            self.assertIn("Accepted+by+the+shared+NeMo+and+pyannote+workflow", headers["Location"])
            self.assertFalse(any(path.name != ".gitkeep" for path in audio_dir.iterdir()))

    def test_files_route_serves_url_escaped_workspace_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            artifact_dir = root / "job_outputs" / "example"
            artifact_dir.mkdir(parents=True)
            artifact_path = artifact_dir / "my file #1.txt"
            artifact_path.write_text("hello\n", encoding="utf-8")

            app = workflow_web.WorkflowWebApp(root=root)
            href = app.file_link(artifact_path, "")
            status, headers, body = run_wsgi(app, method="GET", path=href)

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/plain")
            self.assertEqual(body, b"hello\n")

    def test_files_route_repairs_mojibake_path_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            artifact_dir = root / "job_outputs" / "example"
            artifact_dir.mkdir(parents=True)
            artifact_name = "notes ｜ draft.txt"
            artifact_path = artifact_dir / artifact_name
            artifact_path.write_text("fixed\n", encoding="utf-8")

            mojibake_name = artifact_name.encode("utf-8").decode("latin-1")
            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(
                app,
                method="GET",
                path=f"/files/job_outputs/example/{mojibake_name}",
            )

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/plain")
            self.assertEqual(body, b"fixed\n")

    def test_youtube_conversion_route_launches_background_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            (root / "youtube_links.txt").write_text(
                "https://youtu.be/example-one\nhttps://youtu.be/example-two\n",
                encoding="utf-8",
            )

            captured = {}
            original = workflow_web.launch_background_command

            def fake_launch_background_command(**kwargs):
                captured.update(kwargs)
                return {"pid": 4321}

            workflow_web.launch_background_command = fake_launch_background_command
            try:
                body = urlencode(
                    [("selected_urls", "https://youtu.be/example-one")]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/convert-youtube",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.launch_background_command = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(captured["cwd"], root)
            self.assertEqual(captured["command"][1], "workflow_cli.py")
            self.assertIn("convert-youtube-selection", captured["command"])
            self.assertIn("--queue-file", captured["command"])
            self.assertIn(str(root / "youtube_links.txt"), captured["command"])
            self.assertEqual(
                captured["command"][captured["command"].index("--output-dir") + 1],
                str(root / "audio_in" / "youtube_links"),
            )
            self.assertTrue(
                str(captured["run_dir"]).startswith(
                    str(root / "outputs" / "youtube_conversion_runs")
                )
            )
            selection_path = pathlib.Path(captured["metadata"]["selected_urls_path"])
            self.assertEqual(
                selection_path.read_text(encoding="utf-8"),
                "https://youtu.be/example-one\n",
            )

    def test_youtube_conversion_route_submits_sbatch_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_youtube_conversion.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "youtube_links.txt").write_text(
                "https://youtu.be/example-one\nhttps://youtu.be/example-two\n",
                encoding="utf-8",
            )

            captured = {}
            original_submit = workflow_web.submit_sbatch_job
            original_which = workflow_web.shutil.which

            def fake_submit_sbatch_job(**kwargs):
                captured.update(kwargs)
                metadata = dict(kwargs["metadata"])
                metadata["slurm_job_id"] = "98765"
                kwargs["run_dir"].mkdir(parents=True, exist_ok=True)
                (kwargs["run_dir"] / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                return {"slurm_job_id": "98765"}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            workflow_web.shutil.which = lambda name: "/usr/bin/sbatch" if name == "sbatch" else original_which(name)
            try:
                body = urlencode(
                    [
                        ("selected_urls", "https://youtu.be/example-one"),
                        ("youtube_force_redownload", "on"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/convert-youtube",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original_submit
                workflow_web.shutil.which = original_which

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(captured["cwd"], root)
            self.assertEqual(captured["sbatch_script"], root / "scheduler" / "run_site_youtube_conversion.sbatch")
            self.assertEqual(captured["job_label"], "YouTube audio conversion")
            self.assertEqual(captured["export_env"]["SITE_YOUTUBE_QUEUE_FILE"], str(root / "youtube_links.txt"))
            self.assertEqual(captured["export_env"]["SITE_YOUTUBE_FORCE_REDOWNLOAD"], "1")
            self.assertTrue(captured["metadata"]["force_redownload"])
            self.assertEqual(
                pathlib.Path(captured["metadata"]["selected_urls_path"]).read_text(encoding="utf-8"),
                "https://youtu.be/example-one\n",
            )

    def test_youtube_page_renders_run_monitor_and_reset_controls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (root / "youtube_links.txt").write_text(
                "https://youtu.be/example-one\nhttps://youtu.be/example-two\n",
                encoding="utf-8",
            )
            audio_path = audio_dir / "001_example.wav"
            audio_path.write_bytes(wav_bytes())

            history_dir = root / "outputs" / "youtube_conversion_history"
            history_dir.mkdir(parents=True)
            (history_dir / "url_audio_index.tsv").write_text(
                "\t".join(
                    [
                        "url",
                        "video_id",
                        "status",
                        "audio_file",
                        "audio_path",
                        "title",
                        "last_attempt_utc",
                        "note",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "https://youtu.be/example-one",
                        "abc123xyz01",
                        "ok",
                        "001_example.wav",
                        str(audio_path),
                        "Example Title",
                        "2026-04-10T18:00:00+00:00",
                        "downloaded",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            run_dir = root / "outputs" / "youtube_conversion_runs" / "20260410T180500_youtube-conversion_02-items"
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                '{"mode": "selected", "selected_url_count": 2, "started_at_utc": "2026-04-10T18:05:00+00:00", "pid": 0}',
                encoding="utf-8",
            )
            (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "selected_urls.txt").write_text(
                "https://youtu.be/example-one\nhttps://youtu.be/example-two\n",
                encoding="utf-8",
            )
            (run_dir / "conversion_report.tsv").write_text(
                "url\tvideo_id\tstatus\tqueue_action\taudio_file\taudio_path\ttitle\tsummary\tdetail_log\tlast_attempt_utc\n"
                f"https://youtu.be/example-one\tabc123xyz01\tskipped\tkept_in_queue\t001_example.wav\t{audio_path}\tExample Title\tAlready converted.\t\t2026-04-10T18:05:05+00:00\n"
                "https://youtu.be/example-two\tabc123xyz02\tno_data\tremove_from_queue\t\t\t\tThis video is unavailable\t\t2026-04-10T18:05:10+00:00\n",
                encoding="utf-8",
            )
            (run_dir / "queue_updates.tsv").write_text(
                "url\tissue_category\tqueue_action\tqueue_result\tsummary\tlast_attempt_utc\n"
                "https://youtu.be/example-two\tno_data\tremove_from_queue\tremoved\tThis video is unavailable\t2026-04-10T18:05:10+00:00\n",
                encoding="utf-8",
            )
            (run_dir / "stdout.log").write_text("download started\n", encoding="utf-8")
            (run_dir / "stderr.log").write_text("", encoding="utf-8")

            failed_dir = root / "youtube_links_err"
            failed_dir.mkdir()
            (failed_dir / "failed_links_latest.tsv").write_text(
                "run_utc\turl\tvideo_id\tissue_category\tqueue_action\tqueue_result\tsummary\tlast_attempt_utc\tresolved_audio_path\tdetail_log\n"
                "2026-04-10T18:05:10+00:00\thttps://youtu.be/example-two\tabc123xyz02\tno_data\tremove_from_queue\tremoved\tThis video is unavailable\t2026-04-10T18:05:10+00:00\t\t\n",
                encoding="utf-8",
            )

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/youtube")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            state = page_state(body)
            self.assertEqual(state["currentPath"], "/youtube")
            self.assertEqual(state["routes"]["convertYoutube"], "/actions/convert-youtube")
            self.assertEqual(state["routes"]["deleteYoutubeLink"], "/youtube-links/delete")
            self.assertEqual(state["routes"]["resetYoutube"], "/actions/reset-youtube-workspace")
            youtube = state["context"]["youtube"]
            self.assertEqual(youtube["queueRows"][0]["queue_state"], "already converted")
            self.assertEqual(youtube["latestRun"]["summary"]["no_data"], 1)
            self.assertIn("queue_updates.tsv", [link["label"] for link in youtube["latestRun"]["artifactLinks"]])
            self.assertEqual(youtube["noDataRows"][0]["queue_result"], "removed")

    def test_youtube_link_delete_removes_audio_and_compacts_numbering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            first_audio = self._seed_audio_with_youtube_url(
                root,
                folder="youtube_links",
                audio_basename="001_first.wav",
                url="https://youtu.be/first111111",
            )
            removed_audio = self._seed_audio_with_youtube_url(
                root,
                folder="youtube_links",
                audio_basename="002_remove.wav",
                url="https://youtu.be/remove22222",
            )
            kept_audio = self._seed_audio_with_youtube_url(
                root,
                folder="youtube_links",
                audio_basename="003_keep.wav",
                url="https://youtu.be/keep333333",
            )
            run_dir = self._seed_diarization_run_for_audio(
                root,
                run_subdir="nemo/20260430T100200_diarization_nemo_01-items",
                audio_relative="youtube_links/003_keep.wav",
                stem="youtube_links__003_keep",
            )
            youtube_run = root / "outputs" / "youtube_conversion_runs" / "20260501T120000_youtube-conversion_03-items"
            youtube_run.mkdir(parents=True)
            (youtube_run / "conversion_report.tsv").write_text(
                "url\tvideo_id\tstatus\tqueue_action\taudio_file\taudio_path\ttitle\tsummary\tdetail_log\tlast_attempt_utc\n"
                f"https://youtu.be/remove22222\tremove22222\tdownloaded\tkept_in_queue\t002_remove.wav\t{removed_audio}\tRemove\tDownloaded.\t\t2026-05-01T12:00:00+00:00\n"
                f"https://youtu.be/keep333333\tkeep333333\tdownloaded\tkept_in_queue\t003_keep.wav\t{kept_audio}\tKeep\tDownloaded.\t\t2026-05-01T12:00:01+00:00\n",
                encoding="utf-8",
            )
            (youtube_run / "resolved_audio_paths.txt").write_text(f"{removed_audio}\n{kept_audio}\n", encoding="utf-8")

            body = urlencode([("youtube_url", "https://youtu.be/remove22222")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/youtube-links/delete",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            queue_text = (root / "youtube_links.txt").read_text(encoding="utf-8")
            self.assertIn("https://youtu.be/first111111", queue_text)
            self.assertNotIn("https://youtu.be/remove22222", queue_text)
            self.assertIn("https://youtu.be/keep333333", queue_text)
            self.assertTrue(first_audio.exists())
            self.assertFalse(removed_audio.exists())
            self.assertFalse(kept_audio.exists())
            compacted_keep = root / "audio_in" / "youtube_links" / "002_keep.wav"
            self.assertTrue(compacted_keep.exists())

            index_text = (root / "outputs" / "youtube_conversion_history" / "url_audio_index.tsv").read_text(encoding="utf-8")
            self.assertNotIn("https://youtu.be/remove22222", index_text)
            self.assertIn("002_keep.wav", index_text)
            self.assertNotIn(f"\t{kept_audio}\t", index_text)
            self.assertNotIn("\t003_keep.wav\t", index_text)

            summary_text = (run_dir / "logs" / "runtime_summary.tsv").read_text(encoding="utf-8")
            self.assertIn("youtube_links/002_keep.wav", summary_text)
            self.assertNotIn("youtube_links/003_keep.wav", summary_text)
            selection_text = (run_dir / "selected_audio.txt").read_text(encoding="utf-8")
            self.assertIn("youtube_links/002_keep.wav", selection_text)
            self.assertNotIn("youtube_links/003_keep.wav", selection_text)
            self.assertFalse((run_dir / "youtube_links__003_keep.txt").exists())
            self.assertTrue((run_dir / "youtube_links__002_keep.txt").exists())

            report_text = (youtube_run / "conversion_report.tsv").read_text(encoding="utf-8")
            self.assertIn("002_keep.wav", report_text)
            self.assertNotIn("003_keep.wav", report_text)
            resolved_text = (youtube_run / "resolved_audio_paths.txt").read_text(encoding="utf-8")
            self.assertIn(str(compacted_keep), resolved_text)
            self.assertNotIn(str(kept_audio), resolved_text)

    def test_youtube_page_exposes_slurm_queue_tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            (root / "youtube_links.txt").write_text("https://youtu.be/example-one\n", encoding="utf-8")

            run_dir = root / "outputs" / "youtube_conversion_runs" / "20260416T193000_youtube-conversion_01-items"
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                '{"mode": "selected", "selected_url_count": 1, "slurm_job_id": "98765"}',
                encoding="utf-8",
            )
            (run_dir / "selected_urls.txt").write_text("https://youtu.be/example-one\n", encoding="utf-8")
            (run_dir / "stdout.log").write_text("submitted youtube conversion\n", encoding="utf-8")
            (run_dir / "stderr.log").write_text("", encoding="utf-8")

            original = workflow_web.slurm_queue_snapshot
            workflow_web.slurm_queue_snapshot = lambda job_id: {
                "job_id": job_id,
                "available": True,
                "state": "PENDING",
                "queue_position": 2,
                "jobs_ahead": 1,
                "reason": "Resources",
                "estimated_start": "2026-04-17T03:30:00",
                "message": "Pending position 2.",
            }
            try:
                status, headers, body = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="GET",
                    path="/youtube",
                )
            finally:
                workflow_web.slurm_queue_snapshot = original

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            latest_run = page_state(body)["context"]["youtube"]["latestRun"]
            self.assertEqual(latest_run["status"], "submitted")
            self.assertEqual(latest_run["slurmQueue"]["job_id"], "98765")
            self.assertEqual(latest_run["slurmQueue"]["queue_position"], 2)
            self.assertEqual(latest_run["slurmQueue"]["jobs_ahead"], 1)

    def test_youtube_tracking_counts_active_conversion_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            (root / "youtube_links.txt").write_text("https://youtu.be/example-one\n", encoding="utf-8")

            active_run = root / "outputs" / "youtube_conversion_runs" / "20260416T193000_youtube-conversion_01-items"
            active_run.mkdir(parents=True)
            (active_run / "metadata.json").write_text(
                '{"mode": "selected", "selected_url_count": 1, "slurm_job_id": "98765"}',
                encoding="utf-8",
            )
            (active_run / "selected_urls.txt").write_text("https://youtu.be/example-one\n", encoding="utf-8")

            completed_run = root / "outputs" / "youtube_conversion_runs" / "20260416T194000_youtube-conversion_01-items"
            completed_run.mkdir(parents=True)
            (completed_run / "metadata.json").write_text(
                '{"mode": "selected", "selected_url_count": 1, "slurm_job_id": "99999"}',
                encoding="utf-8",
            )
            (completed_run / "exit_code.txt").write_text("0\n", encoding="utf-8")

            status, headers, body = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="GET",
                path="/youtube",
            )

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            context = page_state(body)["context"]
            self.assertEqual(context["youtube"]["latestRun"]["status"], "succeeded")
            self.assertTrue(context["tracking"]["shouldRefresh"])
            self.assertEqual(context["tracking"]["refreshIntervalMs"], workflow_web.LIVE_TRACKING_INTERVAL_MS)
            self.assertTrue(context["tracking"]["fingerprint"])
            self.assertEqual(context["tracking"]["activeYoutubeRuns"], 1)
            self.assertIn(active_run.name, context["tracking"]["youtubeRunNames"])

    def test_youtube_reset_route_clears_generated_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (audio_dir / ".gitkeep").write_text("", encoding="utf-8")
            (audio_dir / "001_old.wav").write_bytes(wav_bytes())
            (root / "job_outputs").mkdir()
            (root / "youtube_links.txt").write_text(
                "https://youtu.be/example-one\nhttps://youtu.be/example-two\n",
                encoding="utf-8",
            )

            run_dir = root / "outputs" / "youtube_conversion_runs" / "old_run"
            run_dir.mkdir(parents=True)
            (run_dir / "stdout.log").write_text("old log\n", encoding="utf-8")

            history_dir = root / "outputs" / "youtube_conversion_history"
            history_dir.mkdir(parents=True)
            (history_dir / "url_audio_index.tsv").write_text(
                "url\tvideo_id\tstatus\taudio_file\taudio_path\ttitle\tlast_attempt_utc\tnote\n",
                encoding="utf-8",
            )

            failed_dir = root / "youtube_links_err"
            failed_dir.mkdir()
            (failed_dir / "failed_links_latest.tsv").write_text(
                "run_utc\turl\tvideo_id\terror\tremoved_from_links_file\tlast_attempt_utc\tresolved_audio_path\n",
                encoding="utf-8",
            )

            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/actions/reset-youtube-workspace",
                body=b"",
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertTrue((audio_dir / ".gitkeep").exists())
            self.assertFalse((audio_dir / "001_old.wav").exists())
            self.assertEqual((root / "youtube_links.txt").read_text(encoding="utf-8"), "")
            self.assertFalse(run_dir.exists())
            self.assertFalse((history_dir / "url_audio_index.tsv").exists())
            self.assertFalse((failed_dir / "failed_links_latest.tsv").exists())

    def test_fine_tuning_page_disables_prepare_and_launch_without_projects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/fine-tuning")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            state = page_state(body)
            self.assertEqual(state["currentPath"], "/fine-tuning")
            self.assertEqual(state["routes"]["fineTunePrepare"], "/fine-tuning/prepare")
            self.assertEqual(state["routes"]["fineTuneLaunch"], "/fine-tuning/launch")
            self.assertEqual(state["context"]["fineTuningSummary"]["projects_with_samples"], 0)
            self.assertEqual(state["context"]["projects"], [])

    def test_fine_tuning_page_lists_server_side_training_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            label_dir = root / "fine_tuning" / "label_work"
            output_dir = root / "outputs" / "diarization_runs" / "pyannote" / "run-01"
            audio_dir.mkdir()
            label_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            (label_dir / "001_clip.rttm").write_text(
                "SPEAKER 001_clip 1 0.000 0.500 <NA> <NA> speaker_0 <NA> <NA>\n",
                encoding="utf-8",
            )
            (output_dir / "001_clip.txt").write_text("short transcript\n", encoding="utf-8")

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/fine-tuning")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            context = page_state(body)["context"]
            self.assertEqual(context["trainingLabels"]["summary"]["total"], 1)
            self.assertEqual(context["trainingLabels"]["rows"][0]["status"], "not_started")
            self.assertEqual(
                [row["path"] for row in context["trainingSources"]["audioFiles"]],
                ["audio_in/001_clip.wav"],
            )
            self.assertEqual(
                [row["path"] for row in context["trainingSources"]["rttmFiles"]],
                ["fine_tuning/label_work/001_clip.rttm"],
            )
            self.assertEqual(
                [row["path"] for row in context["trainingSources"]["transcriptFiles"]],
                ["outputs/diarization_runs/pyannote/run-01/001_clip.txt"],
            )

    def test_training_labels_page_lists_uploaded_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/training-labels")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            state = page_state(body)
            self.assertEqual(state["currentPath"], "/training-labels")
            self.assertEqual(state["routes"]["saveTrainingLabel"], "/training-labels/save")
            self.assertEqual(state["context"]["trainingLabels"]["summary"]["total"], 1)
            row = state["context"]["trainingLabels"]["rows"][0]
            self.assertEqual(row["name"], "001_clip.wav")
            self.assertEqual(row["status"], "not_started")
            self.assertEqual(row["backendLabel"], "NeMo + pyannote")
            self.assertEqual(row["targetProjects"], ["nemo/uploaded-site-training", "pyannote/uploaded-site-training"])

    def test_training_label_saved_review_path_regenerates_stale_inspect_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            run_dir = root / "outputs" / "diarization_runs" / "pyannote" / "run-01"
            audio_dir.mkdir()
            run_dir.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            (run_dir / "001_clip.srt").write_text(
                "1\n00:00:00,000 --> 00:00:00,500\nSpeaker 0: hello\n",
                encoding="utf-8",
            )
            review_path = run_dir / "001_clip_review.html"
            review_path.write_text(
                '<html><head><meta name="review-bundle-version" content="1"></head><body>old inspect</body></html>',
                encoding="utf-8",
            )
            label_status_path = root / "fine_tuning" / "label_status.json"
            label_status_path.parent.mkdir(parents=True)
            label_status_path.write_text(
                json.dumps(
                    {
                        "items": {
                            "001_clip.wav": {
                                "status": "draft",
                                "review_path": "outputs/diarization_runs/pyannote/run-01/001_clip_review.html",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            status, _, body = run_wsgi(workflow_web.WorkflowWebApp(root=root), method="GET", path="/training-labels")

            self.assertEqual(status, "200 OK")
            row = page_state(body)["context"]["trainingLabels"]["rows"][0]
            self.assertTrue(row["reviewHref"].endswith("/files/outputs/diarization_runs/pyannote/run-01/001_clip_review.html"))
            refreshed_html = review_path.read_text(encoding="utf-8")
            self.assertIn("manual-label-toolbar", refreshed_html)
            self.assertIn('id="currentTimeReadout"', refreshed_html)
            self.assertIn("Complete For Training", refreshed_html)

    def test_training_label_can_be_saved_for_later(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "speaker-lab"),
                    ("label_segments", "0.00 0.50 SPEAKER_00"),
                    ("label_action", "draft"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_clip.wav"]
            self.assertEqual(record["status"], "draft")
            self.assertEqual(record["label_segments"], "0.00 0.50 SPEAKER_00")
            self.assertFalse((root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab" / "audio").exists())

    def test_completed_training_label_creates_fine_tuning_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "speaker-lab"),
                    ("label_segments", "0.00 0.50 SPEAKER_00\n0.50 0.90 SPEAKER_01"),
                    ("label_transcript_text", "short transcript"),
                    ("label_action", "complete"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            self.assertTrue((project_dir / "audio" / "001_clip.wav").is_file())
            self.assertTrue((project_dir / "rttm" / "001_clip.rttm").is_file())
            self.assertTrue((project_dir / "text" / "001_clip.txt").is_file())
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            self.assertEqual(label_status["items"]["001_clip.wav"]["status"], "completed")

            status, _, body = run_wsgi(workflow_web.WorkflowWebApp(root=root), method="GET", path="/fine-tuning")
            self.assertEqual(status, "200 OK")
            project = page_state(body)["context"]["projects"][0]
            self.assertEqual(project["backend"], "pyannote")
            self.assertEqual(project["slug"], "speaker-lab")
            self.assertEqual(project["sampleCount"], 1)

            status, _, body = run_wsgi(workflow_web.WorkflowWebApp(root=root), method="GET", path="/training-labels")
            self.assertEqual(status, "200 OK")
            row = page_state(body)["context"]["trainingLabels"]["rows"][0]
            self.assertEqual(row["backendLabel"], "pyannote")
            self.assertEqual(row["trainingProjects"], ["pyannote/speaker-lab"])

    def test_dialogue_toggle_off_drops_transcript_from_completed_sample(self):
        # When the dialogue checkbox is OFF (form sends "0"), the transcript
        # text must NOT be written into the project's text/ folder, and the
        # label record should reflect include_transcript=False.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "speaker-lab"),
                    ("label_segments", "0.00 0.50 SPEAKER_00"),
                    ("label_transcript_text", "user typed this but toggled off"),
                    ("label_include_transcript", "0"),
                    ("label_action", "complete"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            self.assertTrue((project_dir / "audio" / "001_clip.wav").is_file())
            self.assertTrue((project_dir / "rttm" / "001_clip.rttm").is_file())
            # Toggle was OFF, so the transcript file must not exist.
            self.assertFalse((project_dir / "text" / "001_clip.txt").is_file())

            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_clip.wav"]
            self.assertEqual(record["status"], "completed")
            self.assertFalse(record.get("include_transcript"))
            self.assertEqual(record.get("transcript_text", ""), "")

    def test_auto_save_draft_returns_json_without_redirect(self):
        # The auto-save path is invoked by the browser-side debounce; it must
        # save the draft AND return a JSON status instead of a redirect, so the
        # user stays on the page they're typing into.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "speaker-lab"),
                    ("label_segments", "0.00 0.50 SPEAKER_00"),
                    ("label_action", "draft"),
                    ("label_auto_save", "1"),
                ]
            ).encode("utf-8")
            status, headers, response_body = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "200 OK")
            self.assertNotIn("Location", headers)
            self.assertIn("application/json", headers.get("Content-Type", ""))
            payload = json.loads(response_body.decode("utf-8"))
            self.assertEqual(payload.get("status"), "saved")
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            self.assertEqual(label_status["items"]["001_clip.wav"]["status"], "draft")

    def test_auto_save_draft_does_not_downgrade_completed_label_when_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            segments = "0.000 0.500 SPEAKER_00"

            complete_body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "speaker-lab"),
                    ("label_segments", segments),
                    ("label_action", "complete"),
                ]
            ).encode("utf-8")
            status, _, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=complete_body,
                content_type="application/x-www-form-urlencoded",
            )
            self.assertEqual(status, "303 See Other")

            autosave_body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "speaker-lab"),
                    ("label_segments", segments),
                    ("label_action", "draft"),
                    ("label_auto_save", "1"),
                ]
            ).encode("utf-8")
            status, _, response_body = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=autosave_body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "200 OK")
            payload = json.loads(response_body.decode("utf-8"))
            self.assertEqual(payload.get("kind"), "completed")
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            self.assertEqual(label_status["items"]["001_clip.wav"]["status"], "completed")
            self.assertTrue((root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab" / "rttm" / "001_clip.rttm").is_file())

    def test_auto_train_uses_project_prepare_settings_and_hf_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            secrets_dir = root / ".local_dashboard"
            secrets_dir.mkdir()
            (secrets_dir / "secrets.env").write_text("HF_TOKEN=secret-token\n", encoding="utf-8")

            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            (project_dir / "artifacts").mkdir(parents=True)
            (project_dir / "display.json").write_text(
                json.dumps({"auto_train": True}),
                encoding="utf-8",
            )
            (project_dir / "artifacts" / "metadata.json").write_text(
                json.dumps(
                    {
                        "pyannote_pretrained_model": "local/checkpoints/speaker-v1",
                        "pyannote_duration": 12.5,
                        "pyannote_max_speakers_per_chunk": 4,
                        "pyannote_max_speakers_per_frame": 3,
                        "devices": 1,
                        "max_epochs": 7,
                        "slurm_partition": "gpu",
                        "slurm_time": "02:00:00",
                        "slurm_memory": "24G",
                        "slurm_cpus": 6,
                        "slurm_gpus": 1,
                    }
                ),
                encoding="utf-8",
            )

            captured = {}
            from dashboard import auto_train as auto_train_module

            def fake_queue_auto_train(project_name, *, backend, root, prepare_options=None, extra_env=None):
                captured.update(
                    {
                        "project_name": project_name,
                        "backend": backend,
                        "root": root,
                        "prepare_options": dict(prepare_options or {}),
                        "extra_env": dict(extra_env or {}),
                    }
                )
                return True

            original = auto_train_module.queue_auto_train
            auto_train_module.queue_auto_train = fake_queue_auto_train
            try:
                body = urlencode(
                    [
                        ("audio_file", "001_clip.wav"),
                        ("label_backend", "pyannote"),
                        ("label_project_name", "speaker-lab"),
                        ("label_segments", "0:00.000 0:00.500 SPEAKER_00"),
                        ("label_action", "complete"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/training-labels/save",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                auto_train_module.queue_auto_train = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(captured["project_name"], "speaker-lab")
            self.assertEqual(captured["backend"], "pyannote")
            self.assertEqual(captured["extra_env"]["HF_TOKEN"], "secret-token")
            self.assertEqual(captured["extra_env"]["HUGGINGFACE_HUB_TOKEN"], "secret-token")
            self.assertEqual(captured["prepare_options"]["pyannote_pretrained_model"], "local/checkpoints/speaker-v1")
            self.assertEqual(captured["prepare_options"]["pyannote_duration"], 12.5)
            self.assertEqual(captured["prepare_options"]["pyannote_max_speakers_per_chunk"], 4)
            self.assertEqual(captured["prepare_options"]["pyannote_max_speakers_per_frame"], 3)
            self.assertEqual(captured["prepare_options"]["max_epochs"], 7)
            self.assertEqual(captured["prepare_options"]["slurm_memory"], "24G")

    def test_completed_training_label_queues_selected_training_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            existing_project = root / "fine_tuning" / "projects" / "pyannote" / "existing-lab" / "artifacts"
            existing_project.mkdir(parents=True)
            (existing_project / "metadata.json").write_text(
                json.dumps(
                    {
                        "pyannote_pretrained_model": "local/checkpoints/existing-v1",
                        "pyannote_duration": 11.0,
                        "devices": 1,
                        "max_epochs": 9,
                        "slurm_memory": "28G",
                    }
                ),
                encoding="utf-8",
            )

            queue_calls = []
            from dashboard import auto_train as auto_train_module

            def fake_queue_auto_train(project_name, *, backend, root, prepare_options=None, extra_env=None):
                queue_calls.append(
                    {
                        "project_name": project_name,
                        "backend": backend,
                        "prepare_options": dict(prepare_options or {}),
                    }
                )
                return True

            original = auto_train_module.queue_auto_train
            auto_train_module.queue_auto_train = fake_queue_auto_train
            try:
                body = urlencode(
                    [
                        ("audio_file", "001_clip.wav"),
                        ("label_backend", "pyannote"),
                        ("label_project_name", "existing-lab"),
                        ("label_training_targets", "pyannote/existing-lab"),
                        ("label_training_targets", "nemo/new-default-lab"),
                        ("label_auto_train_targets", "pyannote/existing-lab"),
                        ("label_auto_train_targets", "nemo/new-default-lab"),
                        ("label_segments", "0.00 0.50 SPEAKER_00"),
                        ("label_action", "complete"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/training-labels/save",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                auto_train_module.queue_auto_train = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(
                [(call["backend"], call["project_name"]) for call in queue_calls],
                [("pyannote", "existing-lab"), ("nemo", "new-default-lab")],
            )
            self.assertEqual(queue_calls[0]["prepare_options"]["pyannote_pretrained_model"], "local/checkpoints/existing-v1")
            self.assertEqual(queue_calls[0]["prepare_options"]["max_epochs"], 9)
            self.assertEqual(queue_calls[0]["prepare_options"]["slurm_memory"], "28G")
            self.assertTrue((root / "fine_tuning" / "projects" / "pyannote" / "existing-lab" / "audio" / "001_clip.wav").is_file())
            self.assertTrue((root / "fine_tuning" / "projects" / "nemo" / "new-default-lab" / "audio" / "001_clip.wav").is_file())

            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_clip.wav"]
            self.assertEqual(record["training_projects"], ["pyannote/existing-lab", "nemo/new-default-lab"])
            self.assertEqual(record["queued_training_projects"], ["pyannote/existing-lab", "nemo/new-default-lab"])
            self.assertEqual(
                [(item["project_key"], item["auto_train_status"]) for item in record["training_usage"]],
                [("pyannote/existing-lab", "queued"), ("nemo/new-default-lab", "queued")],
            )

    def test_uncomplete_training_label_removes_project_sample_and_rewinds_status(self):
        # End-to-end: complete a label so a sample lands in the project, then
        # POST to /training-labels/uncomplete and verify the audio + rttm are
        # removed and the record is rolled back to draft.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            complete_body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "speaker-lab"),
                    ("label_segments", "0.00 0.50 SPEAKER_00"),
                    ("label_action", "complete"),
                ]
            ).encode("utf-8")
            run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=complete_body,
                content_type="application/x-www-form-urlencoded",
            )

            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            self.assertTrue((project_dir / "audio" / "001_clip.wav").is_file())
            self.assertTrue((project_dir / "rttm" / "001_clip.rttm").is_file())

            uncomplete_body = urlencode([("audio_file", "001_clip.wav")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/uncomplete",
                body=uncomplete_body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            # Project copies are gone, draft state restored, segments preserved.
            self.assertFalse((project_dir / "audio" / "001_clip.wav").exists())
            self.assertFalse((project_dir / "rttm" / "001_clip.rttm").exists())
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_clip.wav"]
            self.assertEqual(record["status"], "draft")
            self.assertEqual(record["training_projects"], [])
            self.assertEqual(record.get("training_audio_path", ""), "")
            self.assertEqual(record["label_segments"], "0.000 0.500 SPEAKER_00")

    def test_recomplete_training_label_removes_stale_project_copy(self):
        # Switching the chosen project on a re-completion must clear the
        # previous project's audio/rttm so the old copy doesn't keep feeding
        # the abandoned project's training set.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            first_body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_training_targets", "pyannote/old-lab"),
                    ("label_segments", "0.00 0.50 SPEAKER_00"),
                    ("label_auto_train_skip", "1"),
                    ("label_action", "complete"),
                ]
            ).encode("utf-8")
            run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=first_body,
                content_type="application/x-www-form-urlencoded",
            )
            old_project = root / "fine_tuning" / "projects" / "pyannote" / "old-lab"
            self.assertTrue((old_project / "audio" / "001_clip.wav").is_file())
            self.assertTrue((old_project / "rttm" / "001_clip.rttm").is_file())

            second_body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_training_targets", "pyannote/new-lab"),
                    ("label_segments", "0.00 0.50 SPEAKER_00"),
                    ("label_auto_train_skip", "1"),
                    ("label_action", "complete"),
                ]
            ).encode("utf-8")
            run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=second_body,
                content_type="application/x-www-form-urlencoded",
            )
            new_project = root / "fine_tuning" / "projects" / "pyannote" / "new-lab"
            self.assertTrue((new_project / "audio" / "001_clip.wav").is_file())
            self.assertTrue((new_project / "rttm" / "001_clip.rttm").is_file())
            # Stale copy in the old project must be cleared.
            self.assertFalse((old_project / "audio" / "001_clip.wav").exists())
            self.assertFalse((old_project / "rttm" / "001_clip.rttm").exists())

    def test_completed_training_label_pairs_audio_and_rttm_for_nested_audio(self):
        # When the audio lives in a subfolder, the project copy uses the
        # path-flattened stem for both audio and RTTM so the pair stays
        # unambiguous and two ``001_clip.wav`` files in different folders
        # can both land in the same project.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            (audio_dir / "alpha").mkdir(parents=True)
            (audio_dir / "beta").mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (audio_dir / "alpha" / "001_clip.wav").write_bytes(wav_bytes())
            (audio_dir / "beta" / "001_clip.wav").write_bytes(wav_bytes())

            for folder in ("alpha", "beta"):
                body = urlencode(
                    [
                        ("audio_file", f"{folder}/001_clip.wav"),
                        ("label_training_targets", "pyannote/shared-lab"),
                        ("label_auto_train_skip", "1"),
                        ("label_segments", "0.00 0.50 SPEAKER_00"),
                        ("label_action", "complete"),
                    ]
                ).encode("utf-8")
                run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/training-labels/save",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )

            project = root / "fine_tuning" / "projects" / "pyannote" / "shared-lab"
            for stem in ("alpha__001_clip", "beta__001_clip"):
                self.assertTrue((project / "audio" / f"{stem}.wav").is_file())
                self.assertTrue((project / "rttm" / f"{stem}.rttm").is_file())

    def test_create_empty_fine_tuning_project_endpoint(self):
        # The Fine-Tuning tab's "Create Empty Project" form lets the user
        # register a project name without uploading anything yet. The endpoint
        # creates the directory shell and writes the optional display name.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()

            body = urlencode(
                [
                    ("fine_tuning_backend", "pyannote"),
                    ("project_name", "Empty Test Lab"),
                    ("display_name", "Empty Test Lab"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/fine-tuning/create-project",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "empty-test-lab"
            self.assertTrue((project_dir / "audio").is_dir())
            self.assertTrue((project_dir / "rttm").is_dir())
            self.assertTrue((project_dir / "text").is_dir())
            # No samples should be present yet.
            self.assertEqual(list((project_dir / "audio").iterdir()), [])
            self.assertEqual(list((project_dir / "rttm").iterdir()), [])
            display_path = project_dir / "display.json"
            self.assertTrue(display_path.is_file())
            display = json.loads(display_path.read_text(encoding="utf-8"))
            self.assertEqual(display.get("display_name"), "Empty Test Lab")
            self.assertTrue(display.get("manually_created"))

            # Empty manually-created projects must still appear in the
            # Fine-Tuning tab so the user can see what they registered. The
            # popup pulls from the same context.projects list.
            status, _, body = run_wsgi(workflow_web.WorkflowWebApp(root=root), method="GET", path="/fine-tuning")
            self.assertEqual(status, "200 OK")
            projects = page_state(body)["context"]["projects"]
            slugs = sorted((p["backend"], p["slug"]) for p in projects)
            self.assertIn(("pyannote", "empty-test-lab"), slugs)

    def test_create_empty_fine_tuning_project_rejects_blank_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            body = urlencode([("fine_tuning_backend", "pyannote"), ("project_name", "   ")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/fine-tuning/create-project",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )
            self.assertEqual(status, "303 See Other")
            self.assertIn("status=error", headers["Location"])
            # Nothing was created.
            self.assertFalse((root / "fine_tuning" / "projects").exists())

    def test_completed_training_label_queues_named_training_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            project_dir.mkdir(parents=True)

            queue_calls = []
            from dashboard import auto_train as auto_train_module

            def fake_queue_auto_train(project_name, *, backend, root, prepare_options=None, extra_env=None, version_name=None):
                queue_calls.append(
                    {
                        "project_name": project_name,
                        "backend": backend,
                        "version_name": version_name,
                    }
                )
                return True

            original = auto_train_module.queue_auto_train
            auto_train_module.queue_auto_train = fake_queue_auto_train
            try:
                body = urlencode(
                    [
                        ("audio_file", "001_clip.wav"),
                        ("label_backend", "pyannote"),
                        ("label_project_name", "speaker-lab"),
                        ("label_training_targets", "pyannote/speaker-lab"),
                        ("label_auto_train_targets", "pyannote/speaker-lab"),
                        ("label_new_training_name", "cleaned-stage-2"),
                        ("label_segments", "0.00 0.50 SPEAKER_00"),
                        ("label_action", "complete"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/training-labels/save",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                auto_train_module.queue_auto_train = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(queue_calls, [{"project_name": "speaker-lab", "backend": "pyannote", "version_name": "cleaned-stage-2"}])
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_clip.wav"]
            self.assertEqual(record["training_usage"][0]["requested_version_name"], "cleaned-stage-2")

    def test_completed_training_label_can_skip_auto_train_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            project_dir.mkdir(parents=True)
            (project_dir / "display.json").write_text(json.dumps({"auto_train": True}), encoding="utf-8")

            from dashboard import auto_train as auto_train_module

            def fail_queue_auto_train(*_args, **_kwargs):
                raise AssertionError("skip should not queue auto-training")

            original = auto_train_module.queue_auto_train
            auto_train_module.queue_auto_train = fail_queue_auto_train
            try:
                body = urlencode(
                    [
                        ("audio_file", "001_clip.wav"),
                        ("label_backend", "pyannote"),
                        ("label_project_name", "speaker-lab"),
                        ("label_training_targets", "pyannote/speaker-lab"),
                        ("label_auto_train_skip", "1"),
                        ("label_segments", "0.00 0.50 SPEAKER_00"),
                        ("label_action", "complete"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/training-labels/save",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                auto_train_module.queue_auto_train = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_clip.wav"]
            self.assertEqual(record["training_projects"], ["pyannote/speaker-lab"])
            self.assertEqual(record["queued_training_projects"], [])
            self.assertEqual(record["training_usage"][0]["auto_train_status"], "skipped")

    def test_completed_training_label_writes_training_ready_rttm_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "Kids Give Advice.wav").write_bytes(wav_bytes())

            body = urlencode(
                [
                    ("audio_file", "Kids Give Advice.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "speaker-lab"),
                    ("label_segments", "0.00 0.50 Speaker 0"),
                    ("label_action", "complete"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            rttm_path = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab" / "rttm" / "001_Kids_Give_Advice.rttm"
            self.assertTrue(rttm_path.is_file())
            self.assertEqual(
                rttm_path.read_text(encoding="utf-8"),
                "SPEAKER 001_Kids_Give_Advice 1 0.000 0.500 <NA> <NA> Speaker_0 <NA> <NA>\n",
            )
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_Kids Give Advice.wav"]
            self.assertEqual(record["speaker_count"], 1)
            self.assertEqual(record["segment_count"], 1)

    def test_training_label_with_open_questions_is_not_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "pyannote"),
                    ("label_project_name", "speaker-lab"),
                    ("label_segments", "0.00 0.50 SPEAKER_00"),
                    ("label_issue_questions", "Who is speaking after 0.50 seconds?"),
                    ("label_action", "complete"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=error", headers["Location"])
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            self.assertEqual(label_status["items"]["001_clip.wav"]["status"], "needs_review")
            self.assertFalse((root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab" / "audio").exists())

    def test_training_label_can_target_both_fine_tuning_backends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            body = urlencode(
                [
                    ("audio_file", "001_clip.wav"),
                    ("label_backend", "both"),
                    ("label_project_name", "shared-lab"),
                    ("label_segments", "0.00 0.50 SPEAKER_00\n0.50 0.90 SPEAKER_01"),
                    ("label_action", "complete"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/training-labels/save",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            for backend in ("nemo", "pyannote"):
                project_dir = root / "fine_tuning" / "projects" / backend / "shared-lab"
                self.assertTrue((project_dir / "audio" / "001_clip.wav").is_file())
                self.assertTrue((project_dir / "rttm" / "001_clip.rttm").is_file())
            label_status = json.loads((root / "fine_tuning" / "label_status.json").read_text(encoding="utf-8"))
            record = label_status["items"]["001_clip.wav"]
            self.assertEqual(record["backend"], "both")
            self.assertEqual(record["training_projects"], ["nemo/shared-lab", "pyannote/shared-lab"])

            status, _, body = run_wsgi(workflow_web.WorkflowWebApp(root=root), method="GET", path="/training-labels")
            self.assertEqual(status, "200 OK")
            row = page_state(body)["context"]["trainingLabels"]["rows"][0]
            self.assertEqual(row["backendLabel"], "NeMo + pyannote")
            self.assertEqual(row["trainingProjects"], ["nemo/shared-lab", "pyannote/shared-lab"])

    def test_fine_tuning_page_renders_project_shortcuts_and_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            (project_dir / "audio").mkdir(parents=True)
            (project_dir / "rttm").mkdir()
            (project_dir / "text").mkdir()
            (project_dir / "artifacts").mkdir()
            (project_dir / "runs" / "20260414T190500Z").mkdir(parents=True)
            (project_dir / "audio" / "sample.wav").write_bytes(wav_bytes())
            (project_dir / "artifacts" / "metadata.json").write_text(
                '{"warnings": []}',
                encoding="utf-8",
            )
            (project_dir / "runs" / "20260414T190500Z" / "metadata.json").write_text(
                '{"pid": 0, "status": "submitted"}',
                encoding="utf-8",
            )
            (project_dir / "runs" / "20260414T190500Z" / "exit_code.txt").write_text(
                "0\n",
                encoding="utf-8",
            )

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="GET", path="/fine-tuning")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            state = page_state(body)
            project = state["context"]["projects"][0]
            self.assertEqual(project["backend"], "pyannote")
            self.assertEqual(project["slug"], "speaker-lab")
            self.assertTrue(project["prepared"])
            self.assertEqual(project["completedSteps"], 3)
            self.assertIn("metadata.json", [link["label"] for link in project["links"]])

    def test_fine_tuning_upload_accepts_multiple_matched_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            body, content_type = multipart_body(
                {"project_name": "speaker-lab", "fine_tuning_backend": "pyannote"},
                [
                    ("training_audio", "sample_a.wav", wav_bytes(2.0), "audio/wav"),
                    ("training_audio", "sample_b.wav", wav_bytes(3.0), "audio/wav"),
                    (
                        "training_rttm",
                        "sample_a.rttm",
                        b"SPEAKER sample_a 1 0.000 0.500 <NA> <NA> speaker_0 <NA> <NA>\n"
                        b"SPEAKER sample_a 1 0.700 0.400 <NA> <NA> speaker_1 <NA> <NA>\n",
                        "text/plain",
                    ),
                    (
                        "training_rttm",
                        "sample_b.rttm",
                        b"SPEAKER sample_b 1 0.000 0.800 <NA> <NA> speaker_0 <NA> <NA>\n"
                        b"SPEAKER sample_b 1 1.000 0.600 <NA> <NA> speaker_2 <NA> <NA>\n",
                        "text/plain",
                    ),
                ],
            )
            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/fine-tuning/upload-sample",
                body=body,
                content_type=content_type,
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            self.assertTrue((project_dir / "audio" / "sample_a.wav").is_file())
            self.assertTrue((project_dir / "audio" / "sample_b.wav").is_file())
            self.assertTrue((project_dir / "rttm" / "sample_a.rttm").is_file())
            self.assertTrue((project_dir / "rttm" / "sample_b.rttm").is_file())

            status, _, body = run_wsgi(app, method="GET", path="/fine-tuning")
            self.assertEqual(status, "200 OK")
            state = page_state(body)
            project = state["context"]["projects"][0]
            self.assertEqual(project["sampleCount"], 2)
            self.assertEqual(project["metrics"]["totalSegments"], 4)
            self.assertAlmostEqual(project["metrics"]["totalSpeechSeconds"], 2.3)
            self.assertAlmostEqual(project["metrics"]["totalActiveSpeechSeconds"], 2.3)
            self.assertAlmostEqual(project["metrics"]["totalOverlapSeconds"], 0.0)
            self.assertAlmostEqual(project["metrics"]["speechCoverage"], 0.46)
            self.assertEqual(project["metrics"]["uniqueSpeakerLabels"], 3)
            self.assertEqual(project["metrics"]["maxConcurrentSpeakers"], 1)
            self.assertAlmostEqual(project["metrics"]["speakerTurnsPerMinute"], 48.0)
            summary = state["context"]["fineTuningSummary"]
            self.assertEqual(summary["total_samples"], 2)
            self.assertEqual(summary["total_segments"], 4)
            self.assertAlmostEqual(summary["total_speech_seconds"], 2.3)
            self.assertAlmostEqual(summary["total_active_speech_seconds"], 2.3)
            self.assertAlmostEqual(summary["total_overlap_seconds"], 0.0)
            self.assertEqual(summary["max_concurrent_speakers"], 1)

    def test_fine_tuning_metrics_calculate_overlap_and_active_speech(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            body, content_type = multipart_body(
                {"project_name": "overlap-lab", "fine_tuning_backend": "pyannote"},
                [
                    ("training_audio", "meeting.wav", wav_bytes(4.0), "audio/wav"),
                    (
                        "training_rttm",
                        "meeting.rttm",
                        b"SPEAKER meeting 1 0.000 2.000 <NA> <NA> speaker_0 <NA> <NA>\n"
                        b"SPEAKER meeting 1 1.000 2.000 <NA> <NA> speaker_1 <NA> <NA>\n"
                        b"SPEAKER meeting 1 3.500 0.400 <NA> <NA> speaker_0 <NA> <NA>\n",
                        "text/plain",
                    ),
                ],
            )
            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/fine-tuning/upload-sample",
                body=body,
                content_type=content_type,
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])

            status, _, body = run_wsgi(app, method="GET", path="/fine-tuning")
            self.assertEqual(status, "200 OK")
            state = page_state(body)
            metrics = state["context"]["projects"][0]["metrics"]
            self.assertAlmostEqual(metrics["totalSpeechSeconds"], 4.4)
            self.assertAlmostEqual(metrics["totalActiveSpeechSeconds"], 3.4)
            self.assertAlmostEqual(metrics["totalOverlapSeconds"], 1.0)
            self.assertAlmostEqual(metrics["totalNonSpeechSeconds"], 0.6)
            self.assertAlmostEqual(metrics["speechCoverage"], 0.85)
            self.assertAlmostEqual(metrics["overlapCoverage"], 0.2941)
            self.assertEqual(metrics["maxConcurrentSpeakers"], 2)
            self.assertAlmostEqual(metrics["speakerTurnsPerMinute"], 45.0)
            self.assertAlmostEqual(metrics["dominantSpeakerShare"], 0.5455)

            summary = state["context"]["fineTuningSummary"]
            self.assertAlmostEqual(summary["total_active_speech_seconds"], 3.4)
            self.assertAlmostEqual(summary["total_overlap_seconds"], 1.0)
            self.assertAlmostEqual(summary["total_non_speech_seconds"], 0.6)
            self.assertAlmostEqual(summary["overlap_coverage"], 0.2941)
            self.assertEqual(summary["max_concurrent_speakers"], 2)

    def test_fine_tuning_upload_accepts_server_side_workspace_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            label_dir = root / "fine_tuning" / "label_work"
            transcript_dir = root / "outputs" / "diarization_runs" / "pyannote" / "run-01"
            audio_dir.mkdir()
            label_dir.mkdir(parents=True)
            transcript_dir.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes(1.5))
            (label_dir / "001_clip.rttm").write_text(
                "SPEAKER 001_clip 1 0.000 0.500 <NA> <NA> speaker_0 <NA> <NA>\n"
                "SPEAKER 001_clip 1 0.600 0.400 <NA> <NA> speaker_1 <NA> <NA>\n",
                encoding="utf-8",
            )
            (transcript_dir / "001_clip.txt").write_text("short transcript\n", encoding="utf-8")

            body = urlencode(
                [
                    ("project_name", "speaker-lab"),
                    ("fine_tuning_backend", "pyannote"),
                    ("server_audio_paths", "audio_in/001_clip.wav"),
                    ("server_rttm_paths", "fine_tuning/label_work/001_clip.rttm"),
                    ("server_transcript_paths", "outputs/diarization_runs/pyannote/run-01/001_clip.txt"),
                ]
            ).encode("utf-8")
            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/fine-tuning/upload-sample",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            project_dir = root / "fine_tuning" / "projects" / "pyannote" / "speaker-lab"
            self.assertTrue((project_dir / "audio" / "001_clip.wav").is_file())
            self.assertTrue((project_dir / "rttm" / "001_clip.rttm").is_file())
            self.assertTrue((project_dir / "text" / "001_clip.txt").is_file())

            status, _, body = run_wsgi(app, method="GET", path="/fine-tuning")
            self.assertEqual(status, "200 OK")
            project = page_state(body)["context"]["projects"][0]
            self.assertEqual(project["sampleCount"], 1)
            self.assertEqual(project["metrics"]["totalSegments"], 2)
            self.assertAlmostEqual(project["metrics"]["totalSpeechSeconds"], 0.9)

    def test_fine_tuning_launch_passes_dashboard_hf_token_for_pyannote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            secrets_dir = root / ".local_dashboard"
            secrets_dir.mkdir()
            (secrets_dir / "secrets.env").write_text("HF_TOKEN=secret-token\n", encoding="utf-8")

            captured = {}

            def fake_launch_training(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(version_name="conversation-pyannote-v1", job_id="", pid=1234)

            original = workflow_web.launch_training
            workflow_web.launch_training = fake_launch_training
            try:
                body = urlencode(
                    [
                        ("launch_project_name", "conversation-diarization-v1"),
                        ("launch_backend", "pyannote"),
                        ("launch_version_name", "conversation-pyannote-v1"),
                        ("launch_local", "on"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/fine-tuning/launch",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.launch_training = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(captured["backend"], "pyannote")
            self.assertEqual(captured["extra_env"]["HF_TOKEN"], "secret-token")
            self.assertEqual(captured["extra_env"]["HUGGINGFACE_HUB_TOKEN"], "secret-token")

    def test_diarization_route_submits_sbatch_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            captured = {}
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                captured.update(kwargs)
                metadata = dict(kwargs["metadata"])
                metadata["slurm_job_id"] = "12345"
                kwargs["run_dir"].mkdir(parents=True, exist_ok=True)
                (kwargs["run_dir"] / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                return {"slurm_job_id": "12345"}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_clip.wav"),
                        ("diarization_backend", "pyannote"),
                        ("pyannote_hf_token", "hf_test_token"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(captured["cwd"], root)
            self.assertEqual(captured["sbatch_script"], root / "scheduler" / "run_site_diarization.sbatch")
            self.assertEqual(captured["export_env"]["DIARIZATION_BACKEND"], "pyannote")
            self.assertEqual(captured["export_env"]["HF_TOKEN"], "hf_test_token")
            self.assertNotIn("hf_test_token", json.dumps(captured["metadata"]))
            self.assertEqual(
                captured["export_env"]["PYANNOTE_SEGMENTATION_MODEL"],
                "diarizers-community/speaker-segmentation-fine-tuned-callhome-zho",
            )
            self.assertEqual(captured["export_env"]["SITE_AUDIO_DIR"], str(audio_dir))
            self.assertTrue(
                str(captured["run_dir"]).startswith(
                    str(root / "outputs" / "diarization_runs")
                )
            )
            self.assertEqual(captured["run_dir"].parent, root / "outputs" / "diarization_runs" / "pyannote")
            selection_path = pathlib.Path(captured["metadata"]["selected_audio_path"])
            self.assertEqual(selection_path.read_text(encoding="utf-8"), "001_clip.wav\n")

    def test_diarization_route_uses_selected_fine_tuned_nemo_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            project_dir = root / "fine_tuning" / "projects" / "nemo" / "campus-lab"
            (project_dir / "audio").mkdir(parents=True)
            (project_dir / "artifacts" / "experiments" / "checkpoints").mkdir(parents=True)
            (project_dir / "audio" / "sample.wav").write_bytes(wav_bytes())
            (project_dir / "artifacts" / "metadata.json").write_text(
                '{"warnings": []}',
                encoding="utf-8",
            )
            model_path = project_dir / "artifacts" / "experiments" / "checkpoints" / "final.nemo"
            model_path.write_text("nemo-model\n", encoding="utf-8")

            captured = {}
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                captured.update(kwargs)
                return {"slurm_job_id": "12345"}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_clip.wav"),
                        ("diarization_model_key", "nemo:campus-lab"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertIn("diarization_model_key=nemo%3Acampus-lab", headers["Location"])
            self.assertEqual(captured["export_env"]["DIARIZATION_BACKEND"], "nemo")
            self.assertEqual(captured["export_env"]["NEMO_MSDD_MODEL"], str(model_path))
            self.assertEqual(captured["metadata"]["model_key"], "nemo:campus-lab")
            self.assertEqual(captured["metadata"]["model_label"], "NeMo fine-tuned / campus-lab")
            self.assertEqual(captured["metadata"]["nemo_msdd_model_path"], str(model_path))
            self.assertEqual(captured["run_dir"].parent, root / "outputs" / "diarization_runs" / "nemo")

    def test_diarization_route_rejects_pyannote_without_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            old_tokens = {
                key: os.environ.pop(key, None)
                for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
            }
            try:
                body = urlencode(
                    [("selected_audio", "001_clip.wav"), ("diarization_backend", "pyannote")]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                for key, value in old_tokens.items():
                    if value is not None:
                        os.environ[key] = value

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=error", headers["Location"])
            self.assertIn("Pyannote+needs+a+Hugging+Face+token", headers["Location"])

    def test_diarization_route_uses_local_dashboard_secret_for_pyannote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            secrets_dir = root / ".local_dashboard"
            secrets_dir.mkdir()
            (secrets_dir / "secrets.env").write_text("HF_TOKEN=hf_local_test\n", encoding="utf-8")
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            captured = {}
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                captured.update(kwargs)
                return {"slurm_job_id": "12345"}

            old_tokens = {
                key: os.environ.pop(key, None)
                for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
            }
            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [("selected_audio", "001_clip.wav"), ("diarization_backend", "pyannote")]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original
                for key, value in old_tokens.items():
                    if value is not None:
                        os.environ[key] = value

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(captured["export_env"]["HF_TOKEN"], "hf_local_test")
            self.assertNotIn("hf_local_test", json.dumps(captured["metadata"]))

    def test_diarization_route_skips_completed_audio_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_done.wav").write_bytes(wav_bytes())
            (audio_dir / "002_new.wav").write_bytes(wav_bytes())

            previous_run = root / "outputs" / "diarization_runs" / "20260417T110000_diarization_nemo_01-items"
            (previous_run / "logs").mkdir(parents=True)
            (previous_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
                encoding="utf-8",
            )
            (previous_run / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (previous_run / "selected_audio.txt").write_text("001_done.wav\n", encoding="utf-8")
            (previous_run / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_done.wav\tok\t2.00\t\t\t\n",
                encoding="utf-8",
            )

            captured = {}
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                captured.update(kwargs)
                return {"slurm_job_id": "12345"}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_done.wav"),
                        ("selected_audio", "002_new.wav"),
                        ("diarization_backend", "nemo"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            selection_path = pathlib.Path(captured["metadata"]["selected_audio_path"])
            self.assertEqual(selection_path.read_text(encoding="utf-8"), "002_new.wav\n")
            self.assertEqual(captured["metadata"]["selected_audio_count"], 1)
            self.assertEqual(captured["metadata"]["skipped_completed_count"], 1)

    def test_diarization_route_still_skips_completed_audio_after_newer_failed_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_done.wav").write_bytes(wav_bytes())

            completed_run = root / "outputs" / "diarization_runs" / "nemo" / "20260417T110000_diarization_nemo_01-items"
            (completed_run / "logs").mkdir(parents=True)
            (completed_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
                encoding="utf-8",
            )
            (completed_run / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (completed_run / "selected_audio.txt").write_text("001_done.wav\n", encoding="utf-8")
            (completed_run / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_done.wav\tok\t2.00\t\t\t\n",
                encoding="utf-8",
            )

            failed_run = root / "outputs" / "diarization_runs" / "nemo" / "20260417T120000_diarization_nemo_01-items"
            (failed_run / "logs").mkdir(parents=True)
            (failed_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
                encoding="utf-8",
            )
            (failed_run / "exit_code.txt").write_text("1\n", encoding="utf-8")
            (failed_run / "selected_audio.txt").write_text("001_done.wav\n", encoding="utf-8")
            (failed_run / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_done.wav\tfailed\t1.00\t\t\tRuntimeError: retry failed\n",
                encoding="utf-8",
            )

            os.utime(completed_run, (100.0, 100.0))
            os.utime(failed_run, (200.0, 200.0))

            original = workflow_web.submit_sbatch_job
            workflow_web.submit_sbatch_job = lambda **kwargs: self.fail("completed audio should not be resubmitted")
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_done.wav"),
                        ("diarization_backend", "nemo"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("All+selected+audio+files+have+already+been+diarized.", headers["Location"])

    def test_diarization_route_tracks_completion_per_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_done.wav").write_bytes(wav_bytes())

            previous_run = root / "outputs" / "diarization_runs" / "nemo" / "20260417T110000_diarization_nemo_01-items"
            (previous_run / "logs").mkdir(parents=True)
            (previous_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
                encoding="utf-8",
            )
            (previous_run / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (previous_run / "selected_audio.txt").write_text("001_done.wav\n", encoding="utf-8")
            (previous_run / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_done.wav\tok\t2.00\t\t\t\n",
                encoding="utf-8",
            )

            captured = {}
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                captured.update(kwargs)
                return {"slurm_job_id": "12345"}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_done.wav"),
                        ("diarization_backend", "pyannote"),
                        ("pyannote_hf_token", "hf_test_token"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            selection_path = pathlib.Path(captured["metadata"]["selected_audio_path"])
            self.assertEqual(selection_path.read_text(encoding="utf-8"), "001_done.wav\n")
            self.assertEqual(captured["metadata"]["backend"], "pyannote")
            self.assertEqual(captured["metadata"]["skipped_completed_count"], 0)
            self.assertEqual(captured["run_dir"].parent, root / "outputs" / "diarization_runs" / "pyannote")

    def test_diarization_route_allows_rerunning_completed_audio_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_done.wav").write_bytes(wav_bytes())

            previous_run = root / "outputs" / "diarization_runs" / "20260417T110000_diarization_nemo_01-items"
            (previous_run / "logs").mkdir(parents=True)
            (previous_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
                encoding="utf-8",
            )
            (previous_run / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (previous_run / "selected_audio.txt").write_text("001_done.wav\n", encoding="utf-8")
            (previous_run / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_done.wav\tok\t2.00\t\t\t\n",
                encoding="utf-8",
            )

            captured = {}
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                captured.update(kwargs)
                return {"slurm_job_id": "12345"}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_done.wav"),
                        ("diarization_backend", "nemo"),
                        ("diarization_include_completed", "on"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            selection_path = pathlib.Path(captured["metadata"]["selected_audio_path"])
            self.assertEqual(selection_path.read_text(encoding="utf-8"), "001_done.wav\n")
            self.assertTrue(captured["metadata"]["include_completed"])

    def test_diarization_route_multi_backend_shares_batch_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            captured: list[dict] = []
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                metadata = dict(kwargs["metadata"])
                kwargs["run_dir"].mkdir(parents=True, exist_ok=True)
                (kwargs["run_dir"] / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                captured.append({"run_dir": kwargs["run_dir"], "metadata": metadata, "export_env": kwargs["export_env"]})
                return {"slurm_job_id": str(10000 + len(captured))}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_clip.wav"),
                        ("diarization_model_keys", "nemo"),
                        ("diarization_model_keys", "pyannote"),
                        ("pyannote_hf_token", "hf_test_token"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(len(captured), 2)
            backends = sorted(call["metadata"]["backend"] for call in captured)
            self.assertEqual(backends, ["nemo", "pyannote"])
            batch_ids = {call["metadata"]["batch_id"] for call in captured}
            self.assertEqual(len(batch_ids), 1)
            batch_id = batch_ids.pop()
            self.assertTrue(batch_id)
            indices = sorted(call["metadata"]["batch_index"] for call in captured)
            self.assertEqual(indices, [0, 1])
            for call in captured:
                self.assertEqual(call["metadata"]["batch_size_total"], 2)
                self.assertEqual(call["metadata"]["batch_model_keys"], ["nemo", "pyannote"])
            run_parents = {call["run_dir"].parent for call in captured}
            self.assertIn(root / "outputs" / "diarization_runs" / "nemo", run_parents)
            self.assertIn(root / "outputs" / "diarization_runs" / "pyannote", run_parents)

    def test_diarization_route_multi_backend_select_all_includes_fine_tuned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            project_dir = root / "fine_tuning" / "projects" / "nemo" / "campus-lab"
            (project_dir / "audio").mkdir(parents=True)
            (project_dir / "artifacts" / "experiments" / "checkpoints").mkdir(parents=True)
            (project_dir / "audio" / "sample.wav").write_bytes(wav_bytes())
            (project_dir / "artifacts" / "metadata.json").write_text(
                '{"warnings": []}',
                encoding="utf-8",
            )
            (project_dir / "artifacts" / "experiments" / "checkpoints" / "final.nemo").write_text(
                "nemo-model\n", encoding="utf-8"
            )

            captured: list[dict] = []
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                metadata = dict(kwargs["metadata"])
                captured.append({"run_dir": kwargs["run_dir"], "metadata": metadata})
                return {"slurm_job_id": str(20000 + len(captured))}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_clip.wav"),
                        ("diarization_model_keys", "nemo"),
                        ("diarization_model_keys", "pyannote"),
                        ("diarization_model_keys", "nemo:campus-lab"),
                        ("pyannote_hf_token", "hf_test_token"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertEqual(len(captured), 3)
            model_keys = sorted(call["metadata"]["model_key"] for call in captured)
            self.assertEqual(model_keys, ["nemo", "nemo:campus-lab", "pyannote"])
            batch_ids = {call["metadata"]["batch_id"] for call in captured}
            self.assertEqual(len(batch_ids), 1)
            for call in captured:
                self.assertEqual(call["metadata"]["batch_size_total"], 3)

    def test_diarization_route_multi_backend_atomic_pyannote_token_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_clip.wav").write_bytes(wav_bytes())

            old_tokens = {
                key: os.environ.pop(key, None)
                for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
            }
            captured: list[dict] = []
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                captured.append(kwargs)
                return {"slurm_job_id": "30000"}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_clip.wav"),
                        ("diarization_model_keys", "nemo"),
                        ("diarization_model_keys", "pyannote"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original
                for key, value in old_tokens.items():
                    if value is not None:
                        os.environ[key] = value

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=error", headers["Location"])
            self.assertIn("Pyannote+needs+a+Hugging+Face+token", headers["Location"])
            self.assertEqual(captured, [])

    def test_diarization_route_multi_backend_skips_saturated_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            audio_dir = root / "audio_in"
            audio_dir.mkdir()
            (root / "scheduler").mkdir()
            (root / "scheduler" / "run_site_diarization.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "job_outputs").mkdir()
            (audio_dir / "001_done.wav").write_bytes(wav_bytes())

            previous_run = root / "outputs" / "diarization_runs" / "nemo" / "20260417T110000_diarization_nemo_01-items"
            (previous_run / "logs").mkdir(parents=True)
            (previous_run / "metadata.json").write_text(
                '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
                encoding="utf-8",
            )
            (previous_run / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (previous_run / "selected_audio.txt").write_text("001_done.wav\n", encoding="utf-8")
            (previous_run / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "001_done.wav\tok\t2.00\t\t\t\n",
                encoding="utf-8",
            )

            captured: list[dict] = []
            original = workflow_web.submit_sbatch_job

            def fake_submit_sbatch_job(**kwargs):
                metadata = dict(kwargs["metadata"])
                captured.append({"run_dir": kwargs["run_dir"], "metadata": metadata})
                return {"slurm_job_id": str(40000 + len(captured))}

            workflow_web.submit_sbatch_job = fake_submit_sbatch_job
            try:
                body = urlencode(
                    [
                        ("selected_audio", "001_done.wav"),
                        ("diarization_model_keys", "nemo"),
                        ("diarization_model_keys", "pyannote"),
                        ("pyannote_hf_token", "hf_test_token"),
                    ]
                ).encode("utf-8")
                status, headers, _ = run_wsgi(
                    workflow_web.WorkflowWebApp(root=root),
                    method="POST",
                    path="/actions/run-diarization",
                    body=body,
                    content_type="application/x-www-form-urlencoded",
                )
            finally:
                workflow_web.submit_sbatch_job = original

            self.assertEqual(status, "303 See Other")
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["metadata"]["backend"], "pyannote")
            self.assertEqual(captured[0]["metadata"]["batch_size_total"], 2)
            self.assertIn("Skipped", headers["Location"])

    def _seed_audio_with_youtube_url(self, root, *, folder, audio_basename, url):
        audio_dir = root / "audio_in" / folder
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / audio_basename
        audio_path.write_bytes(wav_bytes())

        history_dir = root / "outputs" / "youtube_conversion_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        index_path = history_dir / "url_audio_index.tsv"
        existing_lines = []
        if index_path.is_file():
            existing_lines = index_path.read_text(encoding="utf-8").splitlines()
        if not existing_lines:
            existing_lines.append(
                "url\tvideo_id\tstatus\taudio_file\taudio_path\ttitle\tlast_attempt_utc\tnote"
            )
        existing_lines.append(
            f"{url}\tvid_{audio_basename}\tok\t{audio_basename}\t{audio_path}\tTitle\t2026-04-30T00:00:00+00:00\tdownloaded"
        )
        index_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")

        queue_path = root / "youtube_links.txt"
        existing_queue = queue_path.read_text(encoding="utf-8") if queue_path.is_file() else ""
        queue_path.write_text(existing_queue + url + "\n", encoding="utf-8")
        return audio_path

    def _seed_diarization_run_for_audio(self, root, *, run_subdir, audio_relative, stem):
        run_dir = root / "outputs" / "diarization_runs" / run_subdir
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata.json").write_text(
            '{"backend": "nemo", "selected_audio_count": 1, "pid": 0}',
            encoding="utf-8",
        )
        (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
        (run_dir / "selected_audio.txt").write_text(audio_relative + "\n", encoding="utf-8")
        (run_dir / "logs" / "runtime_summary.tsv").write_text(
            "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
            f"{audio_relative}\tok\t2.00\t\t\t\n",
            encoding="utf-8",
        )
        (run_dir / f"{stem}.txt").write_text("transcript\n", encoding="utf-8")
        (run_dir / f"{stem}.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        (run_dir / f"{stem}_review.html").write_text("<html></html>", encoding="utf-8")
        (run_dir / f"{stem}_review_flags.tsv").write_text("flag\n", encoding="utf-8")
        return run_dir

    def test_audio_file_delete_route_cascades_artifacts_but_keeps_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            audio_path = self._seed_audio_with_youtube_url(
                root,
                folder="youtube_links",
                audio_basename="001_video.wav",
                url="https://youtu.be/abc123xyz01",
            )
            run_dir = self._seed_diarization_run_for_audio(
                root,
                run_subdir="nemo/20260430T100000_diarization_nemo_01-items",
                audio_relative="youtube_links/001_video.wav",
                stem="youtube_links__001_video",
            )

            body = urlencode([("audio_path", "youtube_links/001_video.wav")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-files/delete",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertFalse(audio_path.exists())
            queue_text = (root / "youtube_links.txt").read_text(encoding="utf-8")
            self.assertIn("https://youtu.be/abc123xyz01", queue_text)
            index_path = root / "outputs" / "youtube_conversion_history" / "url_audio_index.tsv"
            self.assertNotIn("https://youtu.be/abc123xyz01", index_path.read_text(encoding="utf-8"))
            removed_path = root / "outputs" / "youtube_conversion_history" / "removed_links.tsv"
            self.assertFalse(removed_path.exists())
            for suffix in (".txt", ".srt", "_review.html", "_review_flags.tsv"):
                self.assertFalse((run_dir / f"youtube_links__001_video{suffix}").exists())
            summary_text = (run_dir / "logs" / "runtime_summary.tsv").read_text(encoding="utf-8")
            self.assertNotIn("youtube_links/001_video.wav", summary_text)
            selection_text = (run_dir / "selected_audio.txt").read_text(encoding="utf-8")
            self.assertNotIn("youtube_links/001_video.wav", selection_text)

    def test_audio_file_delete_route_handles_uploads_without_youtube_link(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "job_outputs").mkdir()
            uploads_dir = root / "audio_in" / "file_uploads"
            uploads_dir.mkdir(parents=True)
            audio_path = uploads_dir / "001_uploaded.wav"
            audio_path.write_bytes(wav_bytes())

            body = urlencode([("audio_path", "file_uploads/001_uploaded.wav")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-files/delete",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertFalse(audio_path.exists())
            removed_path = root / "outputs" / "youtube_conversion_history" / "removed_links.tsv"
            self.assertFalse(removed_path.is_file())

    def test_audio_file_delete_succeeds_when_wav_is_already_gone(self):
        """Delete must be idempotent — stale UI rows should not surface 'not found' errors.

        Reproduces the bug the user reported after the bulk WAV cleanup: their
        browser still showed inventory rows whose underlying WAVs were already
        deleted. Clicking Delete on one of those rows used to surface 'Audio
        file not found' even though the user's intent ('make this row stop
        existing') was already satisfied. The route should succeed and quietly
        sweep any orphaned references that still mention the basename.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in" / "youtube_links").mkdir(parents=True)
            (root / "job_outputs").mkdir()

            # Seed a leftover index row + diarization run pointing at a WAV that
            # never actually exists on disk. This is exactly what the bulk
            # cleanup leaves behind for any reference that didn't get swept.
            history_dir = root / "outputs" / "youtube_conversion_history"
            history_dir.mkdir(parents=True)
            (history_dir / "url_audio_index.tsv").write_text(
                "url\tvideo_id\tstatus\taudio_file\taudio_path\ttitle\tlast_attempt_utc\tnote\n"
                "https://youtu.be/ghost\tghost\tok\t999_missing.wav\t/never/existed/999_missing.wav\tGhost\t2026-04-30T00:00:00+00:00\tdownloaded\n",
                encoding="utf-8",
            )
            run_dir = root / "outputs" / "diarization_runs" / "nemo" / "20260430T100000_diarization_nemo_01-items"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "logs" / "runtime_summary.tsv").write_text(
                "audio_file\tstatus\truntime_seconds\tstdout_log\tstderr_log\terror_summary\n"
                "youtube_links/999_missing.wav\tok\t1.0\t\t\t\n",
                encoding="utf-8",
            )
            (run_dir / "selected_audio.txt").write_text("youtube_links/999_missing.wav\n", encoding="utf-8")

            body = urlencode([("audio_path", "youtube_links/999_missing.wav")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-files/delete",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            # The leftover index row should be gone now (matched by basename).
            self.assertNotIn(
                "999_missing.wav",
                (history_dir / "url_audio_index.tsv").read_text(encoding="utf-8"),
            )
            # Same for the diarization summary + selection references.
            self.assertNotIn(
                "999_missing.wav",
                (run_dir / "logs" / "runtime_summary.tsv").read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "999_missing.wav",
                (run_dir / "selected_audio.txt").read_text(encoding="utf-8"),
            )

    def test_audio_file_delete_route_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            body = urlencode([("audio_path", "../etc/passwd")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-files/delete",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=error", headers["Location"])

    def test_audio_folder_delete_route_cascades_every_file_inside(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            self._seed_audio_with_youtube_url(
                root, folder="youtube_links", audio_basename="001_a.wav", url="https://youtu.be/aaa",
            )
            self._seed_audio_with_youtube_url(
                root, folder="youtube_links", audio_basename="002_b.wav", url="https://youtu.be/bbb",
            )

            body = urlencode([("folder_name", "youtube_links")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-folders/delete",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertFalse((root / "audio_in" / "youtube_links").exists())
            queue_text = (root / "youtube_links.txt").read_text(encoding="utf-8")
            self.assertIn("https://youtu.be/aaa", queue_text)
            self.assertIn("https://youtu.be/bbb", queue_text)
            removed_path = root / "outputs" / "youtube_conversion_history" / "removed_links.tsv"
            self.assertFalse(removed_path.exists())
            index_path = root / "outputs" / "youtube_conversion_history" / "url_audio_index.tsv"
            index_text = index_path.read_text(encoding="utf-8")
            self.assertNotIn("https://youtu.be/aaa", index_text)
            self.assertNotIn("https://youtu.be/bbb", index_text)

    def test_audio_folder_delete_treats_stale_missing_folder_as_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            stale_folder = root / "audio_in" / "youtube_links"
            stale_folder.mkdir(parents=True)
            (root / "job_outputs").mkdir()
            app = workflow_web.WorkflowWebApp(root=root)
            run_wsgi(app, method="GET", path="/uploads")
            stale_folder.rmdir()

            body = urlencode([("folder_name", "youtube_links")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                app,
                method="POST",
                path="/audio-folders/delete",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertIn("already+removed", headers["Location"])

    def test_audio_folder_delete_route_clears_root_media_but_keeps_library(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            self._seed_audio_with_youtube_url(
                root, folder="", audio_basename="001_root.wav", url="https://youtu.be/root",
            )
            self._seed_audio_with_youtube_url(
                root, folder="youtube_links", audio_basename="002_nested.wav", url="https://youtu.be/nested",
            )
            run_dir = self._seed_diarization_run_for_audio(
                root,
                run_subdir="nemo/20260430T100100_diarization_nemo_01-items",
                audio_relative="001_root.wav",
                stem="001_root",
            )

            body = urlencode([("folder_name", workflow_web.ROOT_AUDIO_FOLDER_VALUE)]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-folders/delete",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertTrue((root / "audio_in").is_dir())
            self.assertFalse((root / "audio_in" / "001_root.wav").exists())
            self.assertTrue((root / "audio_in" / "youtube_links" / "002_nested.wav").is_file())
            self.assertFalse((run_dir / "001_root.txt").exists())
            index_text = (
                root / "outputs" / "youtube_conversion_history" / "url_audio_index.tsv"
            ).read_text(encoding="utf-8")
            self.assertNotIn("https://youtu.be/root", index_text)
            self.assertIn("https://youtu.be/nested", index_text)

    def test_audio_file_move_route_rewrites_index_summary_and_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()
            self._seed_audio_with_youtube_url(
                root, folder="youtube_links", audio_basename="003_video.wav", url="https://youtu.be/zzz",
            )
            (root / "audio_in" / "file_uploads").mkdir(parents=True, exist_ok=True)
            run_dir = self._seed_diarization_run_for_audio(
                root,
                run_subdir="nemo/20260430T100100_diarization_nemo_01-items",
                audio_relative="youtube_links/003_video.wav",
                stem="youtube_links__003_video",
            )

            body = urlencode(
                [
                    ("audio_path", "youtube_links/003_video.wav"),
                    ("target_folder", "file_uploads"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-files/move",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertFalse((root / "audio_in" / "youtube_links" / "003_video.wav").exists())
            self.assertTrue((root / "audio_in" / "file_uploads" / "003_video.wav").exists())
            index_text = (root / "outputs" / "youtube_conversion_history" / "url_audio_index.tsv").read_text(encoding="utf-8")
            self.assertIn(str(root / "audio_in" / "file_uploads" / "003_video.wav"), index_text)
            self.assertNotIn(str(root / "audio_in" / "youtube_links" / "003_video.wav"), index_text)
            summary_text = (run_dir / "logs" / "runtime_summary.tsv").read_text(encoding="utf-8")
            self.assertIn("file_uploads/003_video.wav", summary_text)
            self.assertNotIn("youtube_links/003_video.wav", summary_text)
            selection_text = (run_dir / "selected_audio.txt").read_text(encoding="utf-8")
            self.assertIn("file_uploads/003_video.wav", selection_text)
            self.assertNotIn("youtube_links/003_video.wav", selection_text)

    def test_audio_file_move_route_rejects_same_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            uploads_dir = root / "audio_in" / "file_uploads"
            uploads_dir.mkdir(parents=True)
            (uploads_dir / "001_x.wav").write_bytes(wav_bytes())
            (root / "job_outputs").mkdir()

            body = urlencode(
                [
                    ("audio_path", "file_uploads/001_x.wav"),
                    ("target_folder", "file_uploads"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-files/move",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=error", headers["Location"])
            self.assertTrue((uploads_dir / "001_x.wav").exists())

    def test_audio_file_move_finds_drifted_file_via_basename_search(self):
        """If the form's path is stale (file already lives in a different folder), locate it by basename."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "job_outputs").mkdir()
            (root / "audio_in" / "youtube_links").mkdir(parents=True)
            actual_dir = root / "audio_in" / "file_uploads"
            actual_dir.mkdir(parents=True)
            actual_path = actual_dir / "001_drifted.wav"
            actual_path.write_bytes(wav_bytes())

            body = urlencode(
                [
                    ("audio_path", "youtube_links/001_drifted.wav"),
                    ("target_folder", "youtube_links"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-files/move",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertFalse(actual_path.exists())
            self.assertTrue((root / "audio_in" / "youtube_links" / "001_drifted.wav").exists())

    def test_audio_file_delete_finds_drifted_file_via_basename_search(self):
        """Cascading delete should also recover when the form's audio_path is stale."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "job_outputs").mkdir()
            (root / "audio_in" / "youtube_links").mkdir(parents=True)
            actual_dir = root / "audio_in" / "file_uploads"
            actual_dir.mkdir(parents=True)
            actual_path = actual_dir / "001_drifted.wav"
            actual_path.write_bytes(wav_bytes())

            body = urlencode([("audio_path", "youtube_links/001_drifted.wav")]).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-files/delete",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=success", headers["Location"])
            self.assertFalse(actual_path.exists())

    def test_audio_file_move_refuses_when_basename_is_ambiguous(self):
        """If two folders both contain the basename, surface an error rather than guessing."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "job_outputs").mkdir()
            youtube_dir = root / "audio_in" / "youtube_links"
            youtube_dir.mkdir(parents=True)
            uploads_dir = root / "audio_in" / "file_uploads"
            uploads_dir.mkdir(parents=True)
            (youtube_dir / "001_clip.wav").write_bytes(wav_bytes())
            (uploads_dir / "001_clip.wav").write_bytes(wav_bytes())

            body = urlencode(
                [
                    ("audio_path", "missing_folder/001_clip.wav"),
                    ("target_folder", "youtube_links"),
                ]
            ).encode("utf-8")
            status, headers, _ = run_wsgi(
                workflow_web.WorkflowWebApp(root=root),
                method="POST",
                path="/audio-files/move",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )

            self.assertEqual(status, "303 See Other")
            self.assertIn("status=error", headers["Location"])
            self.assertTrue((youtube_dir / "001_clip.wav").exists())
            self.assertTrue((uploads_dir / "001_clip.wav").exists())

    def test_youtube_audio_batch_default_output_dir_is_youtube_links(self):
        import re
        source_path = pathlib.Path(__file__).resolve().parent.parent / "youtube_audio_batch.py"
        source = source_path.read_text(encoding="utf-8")
        match = re.search(r'"--output-dir",\s*default=str\(([^)]+)\)', source)
        self.assertIsNotNone(match, "Could not locate --output-dir default in youtube_audio_batch.py")
        self.assertIn("youtube_links", match.group(1))

    def test_dashboard_supports_head_requests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            status, headers, body = run_wsgi(app, method="HEAD", path="/")

            self.assertEqual(status, "200 OK")
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            self.assertEqual(body, b"")

    def test_dashboard_respects_script_name_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            status, _, body = run_wsgi(
                app,
                method="GET",
                path="/proxy/app/uploads",
                environ_overrides={"SCRIPT_NAME": "/proxy/app"},
            )

            self.assertEqual(status, "200 OK")
            state = page_state(body)
            self.assertEqual(state["routes"]["uploadAudio"], "/proxy/app/upload/audio")
            self.assertEqual(state["assets"]["app"].split("?", 1)[0], "/proxy/app/assets/static/js/dashboard_app.js")

            status, _, body = run_wsgi(
                app,
                method="GET",
                path="/proxy/app/fine-tuning",
                environ_overrides={"SCRIPT_NAME": "/proxy/app"},
            )

            self.assertEqual(status, "200 OK")
            state = page_state(body)
            self.assertEqual(state["routes"]["fineTuneLaunch"], "/proxy/app/fine-tuning/launch")

    def test_dashboard_derives_prefix_from_request_uri_when_script_name_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "audio_in").mkdir()
            (root / "job_outputs").mkdir()

            app = workflow_web.WorkflowWebApp(root=root)
            status, _, body = run_wsgi(
                app,
                method="GET",
                path="/uploads",
                environ_overrides={"REQUEST_URI": "/node/abc/proxy/8044/uploads"},
            )

            self.assertEqual(status, "200 OK")
            state = page_state(body)
            youtube_nav = next(item for item in state["navItems"] if item["path"] == "/youtube")
            self.assertEqual(youtube_nav["href"], "/node/abc/proxy/8044/youtube")
            self.assertEqual(state["routes"]["uploadAudio"], "/node/abc/proxy/8044/upload/audio")


if __name__ == "__main__":
    unittest.main()
