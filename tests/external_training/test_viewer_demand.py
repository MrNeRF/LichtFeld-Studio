# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU-only contracts: background polling must not keep an unchanged viewer awake."""
from pathlib import Path
import queue
import sys
import threading
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src/python'))
from lfs_plugins.training_runtime.viewer import Viewer


class ViewerDemand(unittest.TestCase):
    def fixture(self):
        value=Viewer.__new__(Viewer)
        value.node_uuid='owned-preview'
        value.status_pending=False
        value.last_status_signature=None
        value.next_status_at=0
        value.copy_busy=False
        value.copy_error=value.camera_error=''
        value.reader=types.SimpleNamespace(status={'state':'paused','pid':123,'iteration':17},error='')
        return value

    def test_unchanged_metadata_and_heartbeats_never_queue_ui_work(self):
        value=self.fixture()
        native=types.SimpleNamespace(ui=types.SimpleNamespace(schedule_on_ui_thread=mock.Mock(),request_redraw=mock.Mock()))
        value._update_status=mock.Mock()
        with mock.patch.dict(sys.modules,lichtfeld=native), mock.patch('lfs_plugins.training_runtime.viewer.time.monotonic',return_value=1):
            value._schedule_status_change()
            native.ui.schedule_on_ui_thread.assert_called_once()
            value.status_pending=False
            native.ui.schedule_on_ui_thread.reset_mock()
            for heartbeat in range(100):
                value.reader.status={**value.reader.status,'heartbeat':heartbeat,'created':heartbeat,'snapshot':{'id':str(heartbeat)}}
                value._schedule_status_change()
            native.ui.schedule_on_ui_thread.assert_not_called()
            native.ui.request_redraw.assert_not_called()

    def test_changes_coalesce_until_next_deadline_and_pending_callback(self):
        value=self.fixture()
        native=types.SimpleNamespace(ui=types.SimpleNamespace(schedule_on_ui_thread=mock.Mock()))
        value._update_status=mock.Mock()
        with mock.patch.dict(sys.modules,lichtfeld=native), mock.patch('lfs_plugins.training_runtime.viewer.time.monotonic') as clock:
            clock.return_value=1
            value._schedule_status_change()
            value.status_pending=False
            native.ui.schedule_on_ui_thread.reset_mock()
            value.reader.status['iteration']=18
            clock.return_value=1.1
            value._schedule_status_change()
            native.ui.schedule_on_ui_thread.assert_not_called()
            value.reader.status['iteration']=19
            clock.return_value=1.5
            value._schedule_status_change()
            native.ui.schedule_on_ui_thread.assert_called_once()
            value.copy_busy=True
            clock.return_value=2.5
            value._schedule_status_change()
            native.ui.schedule_on_ui_thread.assert_called_once()
            value.status_pending=False
            value._schedule_status_change()
            self.assertEqual(native.ui.schedule_on_ui_thread.call_count,2)
            value.status_pending=False
            value.reader.error='Connection unavailable'
            clock.return_value=3
            value._schedule_status_change()
            self.assertEqual(native.ui.schedule_on_ui_thread.call_count,3)

    def test_live_worker_does_not_schedule_periodic_redraws_for_a_paused_trainer(self):
        value=self.fixture()
        value.stop=threading.Event()
        value.retired_arrays=queue.SimpleQueue()
        value.reader.take=mock.Mock(return_value=None)
        value.close=mock.Mock(side_effect=value.stop.set)
        callbacks=queue.Queue()
        native=types.SimpleNamespace(ui=types.SimpleNamespace(schedule_on_ui_thread=callbacks.put,request_redraw=mock.Mock()))
        value._update_status=lambda:setattr(value,'status_pending',False)
        thread=threading.Thread(target=value._watch)
        with mock.patch.dict(sys.modules,lichtfeld=native):
            try:
                thread.start()
                callbacks.get(timeout=2)()
                started=value.reader.take.call_count
                deadline=time.monotonic()+2
                while value.reader.take.call_count<started+30 and time.monotonic()<deadline:
                    value.reader.status['heartbeat']=time.time()
                    time.sleep(.01)
                self.assertGreaterEqual(value.reader.take.call_count,started+30)
                self.assertTrue(callbacks.empty(),'Unchanged status kept waking the UI')
                self.assertFalse(value.reader.error)
                native.ui.request_redraw.assert_not_called()
            finally:
                value.stop.set();thread.join(3)
            self.assertFalse(thread.is_alive())


if __name__=='__main__':unittest.main()
