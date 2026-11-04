# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise real native guards and editable-copy behavior in the test GUI."""
import os
import json
from pathlib import Path
import threading
import time
import numpy as np
import lichtfeld as lf
import lfs_plugins.training_runtime.panel as panel

viewer = panel._viewer
assert viewer and viewer.node_uuid
results, errors = [], []
done = threading.Event()


def check():
    try:
        scene = lf.get_scene()
        node = scene.get_node(viewer.name)
        before = np.array(node.local_transform)
        def rejected(label, action):
            try:
                action()
            except RuntimeError:
                results.append(label)
            else:
                raise AssertionError(label + " was not rejected")
        rejected("property unlock", lambda: node.set("locked", False))
        rejected("mutable tensor access", node.splat_data)
        rejected("combined mutable access", scene.combined_model)
        rejected("make persistent", lambda: scene.set_node_transient(viewer.name, False))
        rejected("training binding", lambda: scene.set_training_model_node(viewer.name))
        rejected("delete", lambda: scene.remove_node(viewer.name))
        assert not scene.rename_node(viewer.name, "should not rename")
        assert not scene.duplicate_node(viewer.name)
        attempted = np.eye(4, dtype=np.float32)
        attempted[0, 3] = 123
        scene.set_node_transform(viewer.name, attempted)
        np.testing.assert_allclose(node.local_transform, before)
        assert node.get("locked")
        node.set("visible", False)
        assert not node.get("visible")
        node.set("visible", True)
        results.extend(["rename", "duplicate", "transform", "visibility allowed"])
    except Exception as error:
        errors.append(repr(error))
    finally:
        done.set()


lf.ui.schedule_on_ui_thread(check)
lf.ui.request_redraw()
assert done.wait(30) and not errors, errors
previous = {node.name for node in lf.get_scene().get_nodes()}
viewer.create_local_copy()
deadline = time.monotonic() + 30
while viewer.copy_busy and time.monotonic() < deadline:
    time.sleep(.02)  # Editor worker only, never UI.
assert not viewer.copy_busy and not viewer.copy_error, viewer.copy_error
copies = [node for node in lf.get_scene().get_nodes() if node.name not in previous and node.name.startswith("Local snapshot ")]
assert len(copies) == 1, copies
copy = copies[0]
assert copy.splat_data() is not None and not copy.get("locked")
transform = np.eye(4, dtype=np.float32)
transform[0, 3] = 5
lf.get_scene().set_node_transform(copy.name, transform)
np.testing.assert_allclose(copy.local_transform, transform)
assert str(copy.uuid) != viewer.node_uuid
record = {"guards": results, "local_copy": copy.name, "count": copy.gaussian_count,
          "external_uuid": viewer.node_uuid, "copy_uuid": str(copy.uuid)}
(Path(os.environ["LFS_TEST_AREA"]) / "renderer-readonly-proof.json").write_text(json.dumps(record))
print("EXTERNAL_READONLY_AND_EDITABLE_COPY_OK", record)
