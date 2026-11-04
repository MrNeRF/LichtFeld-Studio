# SPDX-License-Identifier: GPL-3.0-or-later
"""Run in the test GUI editor worker; exercise ready-model UI installation."""
import os
import json
from pathlib import Path
import threading
import time
import numpy as np
import lichtfeld as lf

scene = lf.get_scene()
name = "Prepared renderer stress fixture"
proof = []
identity = None
for count in (100000, 1000000):
    means = np.zeros((count, 3), dtype=np.float32)
    means[:, 0] = np.linspace(-1, 1, count)
    prepared = lf.scene.prepare_viewer_splat(means, np.zeros((count, 1, 3), np.float32),
        np.zeros((count, 3, 3), np.float32), np.full((count, 3), -5, np.float32),
        np.tile(np.array([1, 0, 0, 0], np.float32), (count, 1)), np.zeros((count, 1), np.float32), 1)
    done = threading.Event()
    errors = []
    def install():
        started = time.perf_counter()
        try:
            current = lf.get_scene()
            current.install_prepared_viewer(name, prepared)
            node = current.get_node(name)
            proof.append({"count": node.gaussian_count, "uuid": str(node.uuid),
                          "ui_ms": (time.perf_counter() - started) * 1000})
            lf.ui.request_redraw()
        except Exception as error:
            errors.append(str(error))
        finally:
            done.set()
    lf.ui.schedule_on_ui_thread(install)
    lf.ui.request_redraw()
    assert done.wait(30), "UI installation timed out"
    assert not errors, errors
    prepared.release()  # Retirement on editor worker, after UI acknowledgement.
    assert proof[-1]["count"] == count
assert proof[0]["uuid"] == proof[1]["uuid"], proof
(Path(os.environ["LFS_TEST_AREA"]) / "renderer-staging-proof.json").write_text(json.dumps(proof))
print("PREPARED_RENDERER_STAGING_OK", proof)
