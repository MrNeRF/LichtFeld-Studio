# SPDX-License-Identifier: GPL-3.0-or-later
"""Publication is bounded, serialized and optional even on failure/shutdown."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/python"))
from lfs_plugins.training_runtime import producer, transport as t
from test_shared_sessions import arrays


class PublicationWorker(unittest.TestCase):
    def stream(self, publisher):
        stream = producer.AutoStream.__new__(producer.AutoStream)
        stream.publisher = publisher
        stream.metadata = {}
        stream.last_status = stream.last = 0
        stream.interval = 0
        stream.writer = stream.pending = None
        stream.closed = False
        return stream

    def test_slow_writer_never_blocks_step_or_queues_more_snapshots(self):
        gate = threading.Event()
        started = threading.Event()
        finished = threading.Event()
        calls = []
        def publish(*args, **kwargs):
            started.set()
            gate.wait(5)
            calls.append("publish")
        def finish(*args, **kw):
            calls.append("finish")
            finished.set()
        publisher = types.SimpleNamespace(publish=publish, update=Mock(), clients=lambda: [1],
            finish=finish, pool=types.SimpleNamespace(close=lambda: calls.append("close")))
        stream = self.stream(publisher)
        fake = types.SimpleNamespace(get_scene=lambda: types.SimpleNamespace(training_model=lambda: object()))
        try:
            with patch.dict(sys.modules, lichtfeld=fake), patch.object(producer, "numerical_runtime"), patch.object(producer, "capture", return_value=arrays()) as capture:
                stream.step({"iteration": 1, "num_gaussians": 1000000})
                self.assertTrue(started.wait(2))
                for step in range(2, 100):
                    stream.step({"iteration": step, "num_gaussians": 1000000})
                capture.assert_called_once()
                self.assertEqual(calls, [])
                stream.end({"iteration": 99, "terminal_state": "completed"})
                self.assertFalse(finished.is_set())
                stream.step({"iteration": 100, "num_gaussians": 1000000})
                capture.assert_called_once()
            gate.set()
            stream.writer.shutdown(wait=True)
            self.assertEqual(calls, ["publish", "finish", "close"])
        finally:
            gate.set()
            if stream.writer:
                stream.writer.shutdown(wait=True)

    def test_executor_shutdown_during_publication_still_retires_in_order(self):
        gate, started = threading.Event(), threading.Event()
        calls = []
        def publish(*args, **kwargs):
            started.set()
            gate.wait(5)
            calls.append("publish")
        publisher = types.SimpleNamespace(publish=publish, update=Mock(), clients=lambda: [1],
            finish=lambda *args, **kw: calls.append("finish"), pool=types.SimpleNamespace(close=lambda: calls.append("close")))
        stream = self.stream(publisher)
        fake = types.SimpleNamespace(get_scene=lambda: types.SimpleNamespace(training_model=lambda: object()))
        try:
            with patch.dict(sys.modules, lichtfeld=fake), patch.object(producer, "numerical_runtime"), patch.object(producer, "capture", return_value=arrays()):
                stream.step({"iteration": 1, "num_gaussians": 3})
                self.assertTrue(started.wait(2))
                stream.writer.shutdown(wait=False)
                stream.end({"iteration": 1})
                self.assertEqual(calls, [])
            gate.set()
            stream.writer.shutdown(wait=True)
            self.assertEqual(calls, ["publish", "finish", "close"])
            self.assertIsNone(stream.pending)
        finally:
            gate.set()
            if stream.writer:
                stream.writer.shutdown(wait=True)

    def test_failed_publication_does_not_prevent_terminal_outcome_and_cleanup(self):
        publisher = types.SimpleNamespace(publish=Mock(side_effect=OSError("storage lost")), update=Mock(),
            clients=lambda: [1], finish=Mock(), pool=types.SimpleNamespace(close=Mock()))
        stream = self.stream(publisher)
        fake = types.SimpleNamespace(get_scene=lambda: types.SimpleNamespace(training_model=lambda: object()))
        with patch.dict(sys.modules, lichtfeld=fake), patch.object(producer, "numerical_runtime"), patch.object(producer, "capture", return_value=arrays()):
            stream.step({"iteration": 1, "num_gaussians": 10})
            try:
                stream.pending.result(timeout=5)
            except OSError:
                pass
            stream.end({"iteration": 1, "terminal_state": "failed"})
            stream.writer.shutdown(wait=True)
        publisher.finish.assert_called_once_with("failed", iteration=1)
        publisher.pool.close.assert_called_once()

    def test_retirement_reports_existing_saved_checkpoint_without_export(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project.licht"
            project.write_bytes(b"saved by native trainer")
            publisher = types.SimpleNamespace(finish=Mock(), pool=types.SimpleNamespace(close=Mock()))
            stream = self.stream(publisher)
            stream.metadata["output"] = directory
            stream.end({"iteration": 20, "terminal_state": "stopped"})
            publisher.finish.assert_called_once_with("stopped", iteration=20, saved_project=str(project))
            self.assertEqual(project.read_bytes(), b"saved by native trainer")

    def test_metadata_error_recovers_without_reimporting_a_generation(self):
        from lfs_plugins.training_runtime.reader import Observer
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({"label": "fixture"})
            observer = Observer(publisher.root)
            def until(predicate):
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if predicate():
                        return
                    time.sleep(.01)
                self.fail(observer.error)
            try:
                until(lambda: observer.status.get("id") == publisher.id)
                observer.last = "displayed"
                publisher.update(iteration="bad")
                until(lambda: observer.error)
                publisher.update(iteration=1)
                until(lambda: not observer.error)
                self.assertEqual(observer.last, "displayed")
            finally:
                observer.close()
                observer.thread.join(5)
                publisher.finish()
                publisher.pool.close()

    def test_late_snapshot_does_not_roll_back_training_progress(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({"label": "fixture"})
            try:
                publisher.update(iteration=100, count=1000000, model_count=1000000)
                publisher.publish(arrays(3), 50, update_progress=False)
                status = t.read_json(publisher.root / "session.json")
                self.assertEqual(status["iteration"], 100)
                self.assertEqual(status["count"], 1000000)
                self.assertEqual(status["snapshot"]["iteration"], 50)
                self.assertEqual(status["snapshot"]["count"], 3)
            finally:
                publisher.finish()
                publisher.pool.close()

    def test_replaced_run_keeps_its_last_progress(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({"label": "fixture"})
            publisher.update(iteration=42)
            stream = self.stream(publisher)
            stream.hooks = types.SimpleNamespace(clear=Mock())
            with patch.object(producer, "_stream", stream), patch.object(producer, "AutoStream"):
                producer.install()
            status = t.read_json(publisher.root / "session.json")
            self.assertEqual(status["state"], "superseded")
            self.assertEqual(status["iteration"], 42)

    def test_closed_reader_drops_queued_payload(self):
        from lfs_plugins.training_runtime.reader import Observer
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({"label": "fixture"})
            publisher.publish(arrays(), 1)
            observer = Observer(publisher.root)
            deadline = time.monotonic() + 5
            while observer.frames.empty() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertFalse(observer.frames.empty())
            observer.close()
            observer.thread.join(5)
            self.assertFalse(observer.thread.is_alive())
            self.assertTrue(observer.frames.empty())
            self.assertIsNone(observer.inflight)
            publisher.finish()
            publisher.pool.close()
