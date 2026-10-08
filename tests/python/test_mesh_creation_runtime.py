# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh color validation and native project roundtrips through the editor."""

import json
import struct
import time

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

SETUP = '''
import lichtfeld as lf, json
scene = lf.get_scene()
vertices = lf.Tensor.zeros([3, 3], device="cpu", dtype="float32")
vertices[1, 0] = 1.25
vertices[2, 1] = 2.5
indices = lf.Tensor.zeros([1, 3], device="cpu", dtype="int32")
indices[0, 1] = 1
indices[0, 2] = 2
'''


def _mesh_payload(path):
    """Read the one small MESH chunk in this freshly saved test fixture."""
    data = path.read_bytes()
    payloads = []
    for offset in range(65536, len(data) - 64, 64):
        if data[offset:offset + 4] != b"MESH":
            continue
        if struct.unpack_from("<HH", data, offset + 4) != (1, 64):
            continue
        flags, compression, block_crc, stored, raw = struct.unpack_from("<IHHQQ", data, offset + 24)
        assert block_crc == 0
        alignment = 4096 if flags & 1 else 64
        start = ((offset + 64 + alignment - 1) // alignment) * alignment
        assert stored > 0 and start + stored <= len(data)
        payloads.append((compression, raw, data[start:start + stored]))
    assert len(payloads) == 1
    return payloads[0]


@pytest.mark.parametrize("components", [2, 3])
def test_mesh_rejects_non_rgba_colors(camera_runtime, tmp_path, components):
    ep = camera_runtime
    snapshot = tmp_path / "validation.json"
    _editor(ep, SETUP)
    history = _tool(ep, "history_get", {})["undo_count"]
    _editor(ep, f'''
count = len(scene.get_nodes())
colors = lf.Tensor.zeros([3, {components}], device="cpu", dtype="float32")
result = {{"rejected": False}}
try:
    scene.add_mesh("invalid colors", vertices, indices, colors=colors)
except ValueError as error:
    result = {{"rejected": True, "message": str(error)}}
result["unchanged"] = len(scene.get_nodes()) == count
open({str(snapshot)!r}, "w").write(json.dumps(result))
''')
    result = json.loads(snapshot.read_text())
    if not result["rejected"]:
        saved = _call(ep, "tools/call", {"name": "project_save_as", "arguments": {
            "path": str(tmp_path / "invalid.licht"),
        }})
        (tmp_path / "invalid-save.json").write_text(json.dumps(saved, indent=2))
    assert result["rejected"], "non-RGBA mesh colors were accepted at creation"
    assert result["message"] == "colors must have shape [N, 4] matching vertices"
    assert result["unchanged"]
    assert _tool(ep, "history_get", {})["undo_count"] == history


@pytest.mark.parametrize("dtype", [None, "float32", "int32"])
def test_mesh_colors_save_reopen_exact(camera_runtime, tmp_path, dtype):
    ep = camera_runtime
    _editor(ep, SETUP)
    _editor(ep, f'''
colors = None
if {dtype!r} is not None:
    colors = lf.Tensor.zeros([3, 4], device="cpu", dtype={dtype!r})
    colors[:, 3] = 1
    colors[0, 0] = 1
    colors[1, 1] = 1
    colors[2, 2] = 1
scene.add_mesh("mesh", vertices, indices, colors=colors)
def mesh_state():
    n = lf.get_scene().get_node("mesh")
    m = n.mesh() if n else None
    return {{"uuid": n.uuid, "vertices": m.vertex_count, "faces": m.face_count,
             "normals": m.has_normals}} if m else None
''')
    snapshot = tmp_path / "mesh.json"
    capture = f"open({str(snapshot)!r}, 'w').write(json.dumps(mesh_state()))"
    _editor(ep, capture)
    before = json.loads(snapshot.read_text())
    project = tmp_path / "mesh.licht"
    _tool(ep, "project_save_as", {"path": str(project)})
    _tool(ep, "project_open", {"path": str(project), "discard_changes": True})
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        _editor(ep, capture)
        after = json.loads(snapshot.read_text())
        if after is not None:
            break
        time.sleep(.05)
    assert after == before
    reopened = tmp_path / "reopened.licht"
    _tool(ep, "project_save_as", {"path": str(reopened)})
    assert _mesh_payload(reopened) == _mesh_payload(project)
