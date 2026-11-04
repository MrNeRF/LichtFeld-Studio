# SPDX-License-Identifier: GPL-3.0-or-later
import os
import json
import os
from pathlib import Path
import lichtfeld as lf
import lfs_plugins.training_runtime.panel as panel

viewer = panel._viewer
assert viewer and viewer.node_id is not None, viewer.status if viewer else "Disconnected"
node = lf.get_scene().get_node(viewer.name)
assert node.gaussian_count > 0
record = {"viewer_pid": os.getpid(), "node_id": node.id, "uuid": node.uuid,
          "generation": viewer.generation, "count": node.gaussian_count,
          "status": viewer.status, "ui_max_ms": viewer.ui_max_ms}
(Path(os.environ["LFS_TEST_AREA"]) / "renderer-status.json").write_text(json.dumps(record))
lf.ui.set_main_panel_active_tab("lfs.training_sessions")
lf.ui.request_redraw()
print("RENDERER_STATUS_OK", json.dumps(record))
