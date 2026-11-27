# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generated-fixture apps for the ROI and Crop runtime tests."""

import json
import math
import struct
from contextlib import contextmanager

from runtime_test_app import isolated_app
from test_mcp_dataset_load_training import _dataset
from test_render_on_demand_idle import _call, _tool

CROP_TYPES = ("crop_box", "ellipsoid", "crop_polygon")


@contextmanager
def pretraining_app(directory):
    """The app before training: a generated dataset with its point cloud selected."""
    dataset = _dataset(directory / "dataset", 80)
    with isolated_app(directory) as endpoint:
        loaded = _tool(endpoint, "scene_load_dataset", {"path": dataset, "output_path": str(directory / "output"),
                                                        "min_track_length": 0})
        assert loaded["success"], loaded
        job = _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive", "timeout_ms": 90000})
        assert job["success"] and job["outcome"] == "completed", job
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]
        yield endpoint


def grid_model(path, side=12):
    """A flat side x side grid of opaque splats."""
    properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                  "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {side * side}\n"
    header += "".join(f"property float {name}\n" for name in properties) + "end_header\n"
    with path.open("wb") as output:
        output.write(header.encode())
        for y in range(side):
            for x in range(side):
                output.write(struct.pack("<17f", (x - side / 2) * .1, (y - side / 2) * .1, 0, 0, 0, 0,
                                         2., 2., 2., 4., *([math.log(.04)] * 3), 1, 0, 0, 0))


def run(endpoint, code, timeout_ms=30000):
    result = _tool(endpoint, "editor_run", {"code": code, "wait_for_completion": True,
                                             "timeout_ms": timeout_ms, "show_console": False})
    output = result["output"]["text"]
    assert result["success"] and "Traceback" not in output, result
    return output


def dispatch(endpoint, action, value=""):
    """A viewport toolbar button."""
    run(endpoint, "from lfs_plugins.toolbar import _GizmoToolbarController\n"
        f"_GizmoToolbarController().dispatch({action!r}, {value!r})\nprint('done')")


def toolbar(endpoint):
    return json.loads(run(endpoint, "import json\nfrom lfs_plugins.toolbar import _GizmoToolbarController\n"
                                    "print(json.dumps(_GizmoToolbarController().snapshot()))"))


def resource(endpoint, uri):
    return json.loads(_call(endpoint, "resources/read", {"uri": uri})["contents"][0]["text"])


def nodes(endpoint):
    return resource(endpoint, "lichtfeld://scene/nodes")["nodes"]


def crops(endpoint):
    return [node for node in nodes(endpoint) if node["type"] in CROP_TYPES]


def selected(endpoint):
    return resource(endpoint, "lichtfeld://scene/selected_nodes")["nodes"][0]


def volume(endpoint, role, owner="PointCloud"):
    return _tool(endpoint, "crop_volume_get", {"owner": owner, "role": role})["volume"]


def set_volume(endpoint, role, owner="PointCloud", **changes):
    """crop_volume_set, returning the raw response so callers can check rejections."""
    return _call(endpoint, "tools/call", {"name": "crop_volume_set",
                                          "arguments": {"owner": owner, "role": role, **changes}})
