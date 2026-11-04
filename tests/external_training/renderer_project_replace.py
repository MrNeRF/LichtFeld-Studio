# SPDX-License-Identifier: GPL-3.0-or-later
"""Run in the disposable GUI editor after connecting an external viewer."""
import os
import json
import time
import threading
from pathlib import Path
import lichtfeld as lf
import lfs_plugins.training_runtime.panel as panel

viewer = panel._viewer
deadline = time.monotonic() + 10
while viewer.display_arrays is None and time.monotonic() < deadline:
    time.sleep(.02)
assert viewer.display_arrays is not None
gate, entered = threading.Event(), threading.Event()
prepare = viewer._native_prepare
def delayed(arrays, geometry=None):
    entered.set()
    assert gate.wait(5)
    return prepare(arrays, geometry)
viewer._native_prepare = delayed
try:
    viewer.create_local_copy()
    assert entered.wait(2), "Copy worker did not start"
    lf.get_scene().clear()
    gate.set()
    deadline = time.monotonic() + 5
    while (not viewer.stop.is_set() or viewer.copy_busy or viewer.display_arrays is not None) and time.monotonic() < deadline:
        time.sleep(.02)  # Editor worker; the UI remains responsive.
    assert viewer.stop.is_set(), viewer.status
    assert not viewer.copy_busy, "Local copy remained pending after project replacement"
    assert viewer.copy_error, "A delayed copy unexpectedly entered the replacement project"
    assert viewer.display_arrays is None and viewer.prepared_arrays is None
    assert viewer.display_geometry is None and viewer.display_bounds is None
    assert not lf.get_scene().get_nodes()
finally:
    gate.set()
    viewer._native_prepare = prepare

# A disconnected preview must also release its CPU payload after project clear,
# even though its reader/watch threads have stopped and its panel is not needed.
retained = type(viewer)(viewer.client.root)
deadline = time.monotonic() + 10
while retained.display_arrays is None and time.monotonic() < deadline:
    time.sleep(.02)
assert retained.display_arrays is not None, retained.status
retained.close()
retained.thread.join(3)  # Editor worker, never the UI thread.
lf.get_scene().clear()
deadline = time.monotonic() + 5
while retained.display_arrays is not None and time.monotonic() < deadline:
    time.sleep(.02)
assert retained.display_arrays is None and retained.prepared_arrays is None
assert retained.display_geometry is None and retained.display_bounds is None
assert not retained._scene_watch.alive
(Path(os.environ["LFS_TEST_AREA"]) / "project-replace-readonly-proof.json").write_text(
    json.dumps({"disconnected": True, "delayed_copy_discarded": True, "status": viewer.status}))
print("PROJECT_REPLACEMENT_DISCONNECTS_OBSERVER_OK")
