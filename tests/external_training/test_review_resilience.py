# SPDX-License-Identifier: GPL-3.0-or-later
"""Fault isolation and status updates that cannot depend on another train step."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/python'))
from lfs_plugins.training_runtime import geometry, transport as t
from lfs_plugins.training_runtime.reader import Observer
from test_shared_sessions import arrays


class ReviewResilience(unittest.TestCase):
    def until(self, predicate, message):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail(message())

    def test_heartbeat_shows_pause_without_new_snapshot_and_terminal_wins(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            state = {'state':'active', 'iteration':17, 'iterations':100}
            publisher = t.SessionPublisher({}, status_provider=lambda: dict(state))
            try:
                state['state'] = 'paused'
                publisher.heartbeat()
                record = t.read_json(publisher.root / 'session.json')
                self.assertEqual((record['state'], record['iteration']), ('paused', 17))
                publisher.finish('stopped', iteration=18)
                state['state'] = 'active'
                publisher.heartbeat()
                self.assertEqual(t.read_json(publisher.root / 'session.json')['state'], 'stopped')
            finally:
                publisher.finish('stopped')
                publisher.pool.close()

    def test_inflight_status_probe_cannot_revive_terminal_session(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({})
            entered, release = threading.Event(), threading.Event()
            def probe():
                entered.set()
                release.wait(3)
                return {'state':'active', 'iteration':20}
            publisher.status_provider = probe
            thread = threading.Thread(target=publisher.heartbeat)
            try:
                thread.start()
                self.assertTrue(entered.wait(2))
                publisher.finish('failed', iteration=19)
                release.set()
                thread.join(3)
                record = t.read_json(publisher.root / 'session.json')
                self.assertEqual((record['state'], record['iteration']), ('failed',19))
            finally:
                release.set()
                thread.join(3)
                publisher.finish('failed')
                publisher.pool.close()

    def test_bad_camera_metadata_does_not_block_snapshot_and_recovers(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({})
            publisher.update(geometry={'path':'view_geometry.json', 'bytes':1, 'sha256':'bad'})
            publisher.publish(arrays(), 1)
            observer = Observer(publisher.root)
            frame = None
            try:
                def take():
                    nonlocal frame
                    frame = frame or observer.take()
                    return frame is not None
                self.until(take, lambda: observer.error)
                self.assertFalse(observer.error)
                self.assertTrue(observer.status['geometry_error'])
                self.assertEqual(frame['record']['iteration'], 1)
                observer.acknowledge(frame['record']['id'])
                publisher.update(state='paused')
                geometry.publish(publisher, {'schema':1, 'cameras':[], 'model_to_data':geometry.IDENTITY})
                observer.geometry_retry_at = 0
                self.until(lambda: observer.geometry is not None and not observer.status.get('geometry_error'), lambda: observer.status)
                repaired = None
                def take_repaired():
                    nonlocal repaired
                    repaired = repaired or observer.take()
                    return repaired is not None
                self.until(take_repaired, lambda: observer.error)
                self.assertEqual(repaired['record']['id'], frame['record']['id'])
                self.assertIsNotNone(repaired['geometry'])
                observer.acknowledge(repaired['record']['id'], geometry_ref=None)
                self.until(lambda: observer.ack.empty(), lambda: observer.error)
                self.assertTrue(observer.pending)  # An old geometry ACK cannot retire this frame.
                observer.acknowledge(repaired['record']['id'], 'installation failed')
                self.until(lambda: observer.error == 'installation failed', lambda: observer.error)
                observer.retry_at = 0
                repaired = None
                self.until(take_repaired, lambda: observer.error)
                observer.acknowledge(repaired['record']['id'])
                self.until(lambda: observer.last_geometry_ref == repaired['geometry_ref'], lambda: observer.error)
                publisher.update(geometry=None)
                repaired = None
                self.until(take_repaired, lambda: observer.error)
                self.assertIsNone(repaired['geometry'])
                self.assertEqual(repaired['record']['id'], frame['record']['id'])
                observer.acknowledge(repaired['record']['id'])
                self.until(lambda: observer.last_geometry_ref is None, lambda: observer.error)
                self.assertFalse(observer.error)
            finally:
                observer.close()
                observer.thread.join(5)
                publisher.finish()
                publisher.pool.close()


if __name__ == '__main__':
    unittest.main()
