# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run in original Blender 5.3 with factory startup; no add-on imports."""
import bpy
import json
import math
import pathlib
import sys
args = sys.argv[sys.argv.index("--") + 1:]
manifest = json.loads(pathlib.Path(args[0]).read_text(encoding="utf-8"))
results = []
expected = ((1.6, 1.6, -4.4), (5.8, 6.6, -1.0))
for item in manifest["exports"]:
    bpy.ops.wm.ply_import(filepath=item["path"])
    obj = bpy.context.object
    assert obj.type == "POINTCLOUD", obj.type
    assert obj.data.type == "GAUSSIAN_SPLAT", obj.data.type
    assert len(obj.data.points) == 2
    attrs = obj.data.attributes
    error = max(abs(actual - target) for point, target_point in zip(attrs["position"].data, expected)
                for actual, target in zip(point.vector, target_point))
    assert error < 2e-5, error
    degree = item["degree"]
    assert sum(a.name.startswith("radiance:sh_") for a in attrs) == (degree + 1) ** 2 - 1
    base = attrs["radiance:base"].data[0].vector
    assert abs(base[0] - .1) < 2e-5
    assert abs(base[3] - 1 / (1 + math.exp(1))) < 2e-5
    results.append({"degree": degree, "points": 2, "position_error": error, "native_import": "passed"})
pathlib.Path(args[1]).write_text(json.dumps({"blender": bpy.app.version_string,
    "hash": bpy.app.build_hash.decode(), "addons_used": False, "results": results}, indent=2), encoding="utf-8")
print("PASS original Blender native splat PLY import, SH0-3, no add-on")
