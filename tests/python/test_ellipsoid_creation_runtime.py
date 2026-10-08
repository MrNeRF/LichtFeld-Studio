# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive ellipsoids on degenerate bounds remain saveable."""

import json
import math
import time

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("shape", ["single", "flat", "volume", "tiny", "existing"])
def test_interactive_ellipsoid_save_reopen(camera_runtime, tmp_path, shape):
    ep = camera_runtime
    _editor(ep, f'''
import lichtfeld as lf, json
scene = lf.get_scene()
f = lambda shape: lf.Tensor.zeros(shape, device="cpu", dtype="float32")
positions = [(1., 2., 3.)] if {shape!r} == "single" else [(0., 0., 0.), (2., 0., 0.), (0., 4., {0. if shape == "flat" else 6.})]
means = f([len(positions), 3])
for row, point in enumerate(positions):
    for axis, value in enumerate(point):
        means[row, axis] = value * {1e-6 if shape == "tiny" else 1.}
rotations = f([len(positions), 4]); rotations[:, 0] = 1.
scene.add_splat("model", means, f([len(positions),1,3]), f([len(positions),0,3]), f([len(positions),3]), rotations, f([len(positions),1]), sh_degree=0)
lf.ui.add_ellipsoid("model")
''')
    if shape == "existing":
        _editor(ep, 'scene.get_node("model_ellipsoid").ellipsoid().radii = (1e-6, 2e-6, 3e-6)')
    _editor(ep, '''
def ellipsoid_state():
    s = lf.get_scene()
    n = s.get_node("model_ellipsoid")
    parent = s.get_node("model")
    model = parent.splat_data() if parent else None
    if n is None or model is None:
        return None
    e = n.ellipsoid()
    return {"uuid": n.uuid, "radii": list(e.radii), "enabled": e.enabled,
            "means": model.means_raw.cpu().tolist()}
''')
    snapshot = tmp_path / "ellipsoid.json"
    capture = f"open({str(snapshot)!r}, 'w').write(json.dumps(ellipsoid_state()))"
    _editor(ep, capture)
    before = json.loads(snapshot.read_text())
    (tmp_path / "before.json").write_text(json.dumps(before, indent=2))
    project = tmp_path / "ellipsoid.licht"
    saved = _call(ep, "tools/call", {"name": "project_save_as", "arguments": {"path": str(project)}})
    (tmp_path / "save.json").write_text(json.dumps(saved, indent=2))
    assert not saved.get("isError"), saved
    assert all(radius > 0 for radius in before["radii"])
    if shape == "single":
        assert before["radii"] == pytest.approx([1e-4] * 3)
    elif shape == "flat":
        assert before["radii"] == pytest.approx([math.sqrt(3), 2 * math.sqrt(3), 1e-4])
    elif shape == "tiny":
        assert before["radii"] == pytest.approx([math.sqrt(3) * v * 1e-6 for v in (1, 2, 3)])
    elif shape == "volume":
        assert before["radii"] == pytest.approx([math.sqrt(3) * v for v in (1, 2, 3)])
    else:
        assert before["radii"] == pytest.approx([1e-6, 2e-6, 3e-6])
    _tool(ep, "project_open", {"path": str(project), "discard_changes": True})
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        _editor(ep, capture)
        after = json.loads(snapshot.read_text())
        if after is not None:
            break
        time.sleep(.05)
    assert after == before
    _editor(ep, 'lf.ui.add_ellipsoid("model")')
    _editor(ep, capture)
    assert json.loads(snapshot.read_text()) == before
    _tool(ep, "project_save_as", {"path": str(tmp_path / "reopened.licht")})
