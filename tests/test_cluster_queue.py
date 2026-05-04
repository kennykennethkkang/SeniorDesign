"""Tests for the cluster-wide ``squeue`` snapshot helper and HTTP endpoint."""
from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import workflow_dashboard as workflow_web
from dashboard import cluster_queue


_FAKE_SQUEUE_STDOUT = (
    "1234567|alice|R|gpu|00:14:32|00:45:28|node07|train_pyannote|stat|8|1\n"
    "1234568|bob|PD|gpu|00:00:00|01:00:00|Resources|nemo_msdd|stat|16|2\n"
    "1234569|kkang|R|cpu|00:02:11|UNLIMITED|node12|site_youtube|edu|4|1\n"
    "\n"
    "  \n"
)


def _fake_completed(stdout: str = _FAKE_SQUEUE_STDOUT, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(args=["squeue"], returncode=returncode, stdout=stdout, stderr=stderr)


def run_wsgi(app, *, path: str, query: str = ""):
    captured: dict = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": path,
        "QUERY_STRING": query,
        "CONTENT_LENGTH": "0",
        "CONTENT_TYPE": "",
        "SERVER_NAME": "localhost",
        "SERVER_PORT": "80",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(b""),
        "wsgi.errors": io.StringIO(),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }
    body = b"".join(app(environ, start_response))
    return captured["status"], captured["headers"], body


class ClusterQueueSnapshotTests(unittest.TestCase):
    def test_snapshot_parses_squeue_pipe_output(self):
        with mock.patch.object(cluster_queue, "shutil") as shutil_mock, mock.patch.object(
            cluster_queue, "subprocess"
        ) as subprocess_mock, mock.patch.dict("os.environ", {"USER": "kkang"}, clear=False):
            shutil_mock.which.return_value = "/usr/bin/squeue"
            subprocess_mock.run.return_value = _fake_completed()

            snapshot = cluster_queue.cluster_queue_snapshot()

            self.assertTrue(snapshot["available"])
            self.assertEqual(snapshot["current_user"], "kkang")
            self.assertEqual(len(snapshot["jobs"]), 3)

            alice, bob, me = snapshot["jobs"]
            self.assertEqual(alice["user"], "alice")
            self.assertFalse(alice["is_self"])
            self.assertEqual(bob["state"], "PD")
            self.assertEqual(bob["nodes"], "2")
            self.assertEqual(me["user"], "kkang")
            self.assertTrue(me["is_self"])

    def test_snapshot_returns_unavailable_when_squeue_missing(self):
        with mock.patch.object(cluster_queue, "shutil") as shutil_mock:
            shutil_mock.which.return_value = None

            snapshot = cluster_queue.cluster_queue_snapshot()

            self.assertFalse(snapshot["available"])
            self.assertIn("not available", snapshot["message"])
            self.assertEqual(snapshot["jobs"], [])

    def test_snapshot_handles_nonzero_returncode(self):
        with mock.patch.object(cluster_queue, "shutil") as shutil_mock, mock.patch.object(
            cluster_queue, "subprocess"
        ) as subprocess_mock:
            shutil_mock.which.return_value = "/usr/bin/squeue"
            subprocess_mock.run.return_value = _fake_completed(
                stdout="", returncode=1, stderr="slurm controller down\n"
            )

            snapshot = cluster_queue.cluster_queue_snapshot()

            self.assertTrue(snapshot["available"])
            self.assertEqual(snapshot["jobs"], [])
            self.assertIn("slurm controller down", snapshot["message"])

    def test_snapshot_handles_timeout(self):
        with mock.patch.object(cluster_queue, "shutil") as shutil_mock, mock.patch.object(
            cluster_queue, "subprocess"
        ) as subprocess_mock:
            shutil_mock.which.return_value = "/usr/bin/squeue"
            subprocess_mock.TimeoutExpired = subprocess.TimeoutExpired
            subprocess_mock.run.side_effect = subprocess.TimeoutExpired(cmd="squeue", timeout=5)

            snapshot = cluster_queue.cluster_queue_snapshot()

            self.assertTrue(snapshot["available"])
            self.assertIn("timed out", snapshot["message"])
            self.assertEqual(snapshot["jobs"], [])


class ClusterQueueRouteTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.app = workflow_web.WorkflowWebApp(root=self.root)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _patched_run(self, stdout: str = _FAKE_SQUEUE_STDOUT, returncode: int = 0):
        return mock.patch.multiple(
            cluster_queue,
            shutil=mock.DEFAULT,
            subprocess=mock.DEFAULT,
        )

    def test_route_returns_full_snapshot(self):
        with mock.patch.object(cluster_queue, "shutil") as shutil_mock, mock.patch.object(
            cluster_queue, "subprocess"
        ) as subprocess_mock, mock.patch.dict("os.environ", {"USER": "kkang"}, clear=False):
            shutil_mock.which.return_value = "/usr/bin/squeue"
            subprocess_mock.run.return_value = _fake_completed()

            status, _headers, body = run_wsgi(self.app, path="/api/cluster-queue")

        self.assertEqual(status, "200 OK")
        payload = json.loads(body)
        self.assertTrue(payload["available"])
        self.assertEqual(len(payload["jobs"]), 3)
        users = [job["user"] for job in payload["jobs"]]
        self.assertEqual(users, ["alice", "bob", "kkang"])
        self.assertTrue(payload["jobs"][2]["is_self"])

    def test_route_filters_by_partition_substring(self):
        with mock.patch.object(cluster_queue, "shutil") as shutil_mock, mock.patch.object(
            cluster_queue, "subprocess"
        ) as subprocess_mock:
            shutil_mock.which.return_value = "/usr/bin/squeue"
            subprocess_mock.run.return_value = _fake_completed()

            status, _headers, body = run_wsgi(self.app, path="/api/cluster-queue", query="partition=cpu")

        self.assertEqual(status, "200 OK")
        payload = json.loads(body)
        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(payload["jobs"][0]["user"], "kkang")
        self.assertTrue(payload["filtered"])

    def test_route_filters_by_user_substring(self):
        with mock.patch.object(cluster_queue, "shutil") as shutil_mock, mock.patch.object(
            cluster_queue, "subprocess"
        ) as subprocess_mock:
            shutil_mock.which.return_value = "/usr/bin/squeue"
            subprocess_mock.run.return_value = _fake_completed()

            status, _headers, body = run_wsgi(self.app, path="/api/cluster-queue", query="user=ali")

        payload = json.loads(body)
        self.assertEqual([job["user"] for job in payload["jobs"]], ["alice"])


if __name__ == "__main__":
    unittest.main()
