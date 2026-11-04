# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import threading
import types
import unittest
from unittest import mock
import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "src/python"
sys.path.insert(0, str(SOURCE))
from lfs_plugins.training_runtime import transport as t


def arrays(count=3, degree=3):
    return {"means": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
            "scale": np.zeros((count, 3), dtype=np.float32),
            "rotation": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (count, 1)),
            "opacity": np.zeros((count, 1), dtype=np.float32),
            "sh": np.arange(count * (degree + 1) ** 2 * 3, dtype=np.float32).reshape(count, -1, 3)}


class SharedSessions(unittest.TestCase):
    def test_discovery_hides_stale_and_terminal_records(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            base = {"protocol": t.PROTOCOL, "pid": os.getpid(), "heartbeat": time.time()}
            states = [("running", {"state": "training"}), ("paused", {"state": "paused"}),
                      ("finished", {"state": "completed"}),
                      ("crashed", {"state": "training", "heartbeat": time.time() - 30})]
            for index, (name, values) in enumerate(states):
                t.atomic_json(Path(root) / name / "session.json", {**base, "pid":100+index, "id": name, **values})
            with mock.patch.object(t,'process_alive',return_value=True):
                self.assertEqual({item["id"] for item in t.discover()}, {"running", "paused"})
                self.assertEqual(len(t.discover()), 2)

    def test_newest_run_supersedes_same_process_including_terminal_registration(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ,LFS_TRAINING_SESSIONS_DIR=root):
            base={"protocol":t.PROTOCOL,"pid":os.getpid(),"heartbeat":time.time(),"state":"training"}
            for index in range(6):
                name=str(index)
                t.atomic_json(Path(root)/name/'session.json',{**base,'id':name,'started':index})
            self.assertEqual([r['id'] for r in t.discover()],['5'])
            t.atomic_json(Path(root)/'5'/'session.json',{**base,'id':'5','started':5,'state':'completed'})
            self.assertEqual(t.discover(),[])

    def test_cross_process_snapshot_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = t.SnapshotPool(directory)
            try:
                original = arrays()
                record = pool.write(original, 42)
                original["means"][:] = -99
                code = "import json,sys; from lfs_plugins.training_runtime.transport import read_snapshot; a=read_snapshot(json.loads(sys.argv[1])); print(float(a['means'].sum()))"
                env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(SOURCE), os.environ.get("PYTHONPATH", ""))))}
                result = subprocess.run([sys.executable, "-c", code, json.dumps(record)],
                                        env=env, capture_output=True, text=True, check=True)
                self.assertEqual(float(result.stdout.strip()), 36)
                bad = copy.deepcopy(record)
                bad["sha256"] = "bad"
                with self.assertRaises(ValueError):
                    t.read_snapshot(bad)
                bad = copy.deepcopy(record)
                bad["layout"]["rotation"]["offset"] += 4
                with self.assertRaises(ValueError):
                    t.read_snapshot(bad)
            finally:
                pool.close()
            with self.assertRaises(FileNotFoundError):
                t.read_snapshot(record)

    def test_bounded_generations_and_owned_reader_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = t.SnapshotPool(directory, retain=2)
            try:
                first = pool.write(arrays(), 1)
                copied = t.read_snapshot(first)
                pool.write(arrays(), 2)
                pool.write(arrays(), 3)
                self.assertEqual(len(pool.mappings), 2)
                with self.assertRaises(FileNotFoundError):
                    t.read_snapshot(first)
                self.assertEqual(float(copied["means"].sum()), 36)
            finally:
                pool.close()

    def test_malformed_snapshot_is_rejected_before_opening_os_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = t.SnapshotPool(directory)
            try:
                record = pool.write(arrays(), 1)
                mutations = [
                    lambda r: r.update(count=2**62),
                    lambda r: r.update(bytes=r['bytes']+4),
                    lambda r: r.update(layout=[]),
                    lambda r: r['layout']['means'].update(shape=[True, 3]),
                    lambda r: r['layout']['means'].update(offset=False),
                    lambda r: r['layout']['sh'].update(shape=[3, 1, 3]),
                    lambda r: r.update(sha256='invalid'),
                    lambda r: r.update(transport=[]),
                    lambda r: r.update(**{'name' if os.name=='nt' else 'path': 'bad\0location'}),
                ]
                for mutate in mutations:
                    invalid = copy.deepcopy(record)
                    mutate(invalid)
                    guard = (mock.patch.object(t.ctypes, 'WinDLL', side_effect=AssertionError('OS mapping opened'))
                             if os.name=='nt' else mock.patch.object(Path, 'open', side_effect=AssertionError('File opened')))
                    with guard, self.assertRaises(ValueError):
                        t.read_snapshot(invalid)
            finally:
                pool.close()

    def test_discovery_multiple_readers_and_disconnect(self):
        previous = os.environ.get("LFS_TRAINING_SESSIONS_DIR")
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["LFS_TRAINING_SESSIONS_DIR"] = directory
                publisher = t.SessionPublisher({"label": "Contract"})
                one = t.SessionClient(publisher.root, "One")
                two = t.SessionClient(publisher.root, "Two")
                try:
                    self.assertTrue(t.discover()[0]["alive"])
                    self.assertEqual(len(publisher.clients()), 2)
                    one.close()
                    time.sleep(0.55)
                    self.assertEqual(len(publisher.clients()), 1)
                    record = publisher.publish(arrays(), 3)
                    self.assertEqual(two.status()["snapshot"]["id"], record["id"])
                finally:
                    one.close()
                    two.close()
                    publisher.finish()
                    publisher.pool.close()
                self.assertEqual(t.discover(), [])
        finally:
            if previous is None:
                os.environ.pop("LFS_TRAINING_SESSIONS_DIR", None)
            else:
                os.environ["LFS_TRAINING_SESSIONS_DIR"] = previous

    def test_metadata_readers_do_not_block_atomic_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            t.atomic_json(path, {"generation": 0})
            failures = []
            seen = []
            finished = threading.Event()
            def reader():
                while not finished.is_set():
                    record = t.read_json(path)
                    if record is not None:
                        seen.append(record)
                    if record is not None and not isinstance(record.get("generation"), int):
                        failures.append(record)
            workers = [threading.Thread(target=reader) for _ in range(3)]
            for worker in workers:
                worker.start()
            try:
                for index in range(100):
                    t.atomic_json(path, {"generation": index})
            finally:
                finished.set()
                for worker in workers:
                    worker.join()
            self.assertEqual(failures, [])
            self.assertGreater(len(seen), 0)
            unicode_path = Path(directory) / "stato-è-🎥.json"
            t.atomic_json(unicode_path, {"value": "coerente"})
            self.assertEqual(t.read_json(unicode_path), {"value": "coerente"})

    def test_optional_observer_failure_does_not_escape_training_callbacks(self):
        from lfs_plugins.training_runtime.producer import AutoStream
        def unavailable(*args, **kwargs):
            raise PermissionError("Observer registry unavailable")
        stream = AutoStream.__new__(AutoStream)
        stream.publisher = types.SimpleNamespace(update=unavailable, clients=unavailable, finish=unavailable)
        stream.last_status = stream.last = 0
        stream.interval = 2
        stream.had_subscribers = False
        stream.step({"iteration": 1, "num_gaussians": 3})
        self.assertEqual(stream.interval, 30)
        stream.end({"iteration": 1})

    def test_starting_another_run_does_not_rewrite_completed_state(self):
        from lfs_plugins.training_runtime import producer
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ):
            os.environ["LFS_TRAINING_SESSIONS_DIR"] = directory
            publisher = t.SessionPublisher({"label": "Completed run"})
            publisher.finish("completed")
            old = producer.AutoStream.__new__(producer.AutoStream)
            old.publisher = publisher
            old.hooks = types.SimpleNamespace(clear=mock.Mock())
            old.closed = True
            publisher.pool.close()
            with mock.patch.object(producer, "_stream", old), mock.patch.object(producer, "AutoStream"):
                producer.install()
            retained = t.read_json(publisher.root / "session.json")
            self.assertEqual(retained["state"], "completed")
            self.assertNotIn("result", retained)
            old.hooks.clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
