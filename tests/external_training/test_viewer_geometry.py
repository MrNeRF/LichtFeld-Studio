# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional camera preparation cannot interrupt model previews or hold UI locks."""
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/python"))
from lfs_plugins.training_runtime import geometry, viewer


class ViewerGeometryContracts(unittest.TestCase):
    def fixture(self):
        value = viewer.Viewer.__new__(viewer.Viewer)
        value.reader = types.SimpleNamespace(geometry={"cameras": [], "model_to_data": geometry.IDENTITY},
                                            geometry_ref={"sha256": "fixture"})
        value.camera_ref = None
        value.camera_error = ""
        value.camera_lock = threading.Lock()
        value.prepared_cameras = None
        value.stop = threading.Event()
        value._native_prepare = mock.Mock(return_value=object())
        return value

    def test_camera_failure_does_not_block_model_preparation(self):
        value = self.fixture()
        native = types.SimpleNamespace(scene=types.SimpleNamespace(
            prepare_viewer_cameras=mock.Mock(side_effect=RuntimeError("camera allocation failed"))))
        with mock.patch.dict(sys.modules, lichtfeld=native), mock.patch.object(viewer, "numerical_runtime", return_value=np):
            result = value._prepare({"means": np.array([[2, 3, 11]], dtype=np.float32)})
        self.assertIs(result, value._native_prepare.return_value)
        self.assertIn("camera allocation failed", value.camera_error)
        self.assertIsNone(value.camera_ref)
        self.assertIsNone(value.prepared_cameras)

    def test_closed_camera_payload_retires_outside_ui_lock(self):
        value = self.fixture()
        value.stop.set()
        def release():
            self.assertTrue(value.camera_lock.acquire(blocking=False))
            value.camera_lock.release()
        payload = types.SimpleNamespace(release=mock.Mock(side_effect=release))
        native = types.SimpleNamespace(scene=types.SimpleNamespace(prepare_viewer_cameras=lambda _: payload))
        with mock.patch.dict(sys.modules, lichtfeld=native), mock.patch.object(viewer, "numerical_runtime", return_value=np):
            value._prepare({"means": np.array([[2, 3, 11]], dtype=np.float32)})
        payload.release.assert_called_once()
        self.assertIsNone(value.prepared_cameras)

    def test_closed_prepare_cannot_retain_new_cpu_arrays(self):
        value = self.fixture()
        value.stop.set()
        value.prepared_arrays = value.prepared_geometry = None
        native = types.SimpleNamespace(scene=types.SimpleNamespace(
            prepare_viewer_cameras=lambda _: types.SimpleNamespace(release=lambda: None)))
        with mock.patch.dict(sys.modules, lichtfeld=native), mock.patch.object(viewer, 'numerical_runtime', return_value=np):
            value._prepare({'means': np.array([[2, 3, 11]], dtype=np.float32)})
        self.assertIsNone(value.prepared_arrays)
        self.assertIsNone(value.prepared_geometry)

    def test_removed_geometry_stages_empty_cameras_and_resets_model_transform(self):
        value = self.fixture()
        value.camera_ref = value.reader.geometry_ref
        value.reader.geometry = value.reader.geometry_ref = None
        payload = object()
        prepare = mock.Mock(return_value=payload)
        native = types.SimpleNamespace(scene=types.SimpleNamespace(prepare_viewer_cameras=prepare))
        with mock.patch.dict(sys.modules, lichtfeld=native), mock.patch.object(viewer, 'numerical_runtime', return_value=np):
            value._prepare({'means': np.array([[2,3,11]], dtype=np.float32)})
        self.assertEqual(value.prepared_cameras, (None, payload, 0))
        self.assertEqual(prepare.call_args.args[0].shape, (0,22))
        self.assertIsNone(value._native_prepare.call_args.args[1])

    def test_reconnect_retires_old_reader_without_overwriting_replacement_status(self):
        value = self.fixture()
        value.node_uuid = 'retained-node'
        value.reader.close = mock.Mock()
        native = types.SimpleNamespace(ui=types.SimpleNamespace(
            schedule_on_ui_thread=mock.Mock(), request_redraw=mock.Mock()))
        with mock.patch.dict(sys.modules, lichtfeld=native):
            value.close(update_status=False)
        self.assertTrue(value.stop.is_set())
        value.reader.close.assert_called_once()
        native.ui.schedule_on_ui_thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
