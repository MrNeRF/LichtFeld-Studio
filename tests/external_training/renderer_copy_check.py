# SPDX-License-Identifier: GPL-3.0-or-later
import os
import json
from pathlib import Path
import numpy as np
import lichtfeld as lf
import lfs_plugins.training_runtime.panel as panel

proof = json.loads((Path(os.environ["LFS_TEST_AREA"]) / "renderer-readonly-proof.json").read_text())
copy = lf.get_scene().get_node(proof["local_copy"])
assert str(copy.uuid) == proof["copy_uuid"]
assert copy.gaussian_count == proof["count"]
assert abs(np.array(copy.local_transform)[0, 3] - 5) < 1e-6
source = lf.get_scene().get_node(panel._viewer.name)
assert str(source.uuid) == proof["external_uuid"] and source.get("locked")
assert np.max(np.abs(np.array(source.local_transform) - np.eye(4))) < 1e-6
print("COPY_REMAINS_INDEPENDENT_WHILE_EXTERNAL_UPDATES_OK")
