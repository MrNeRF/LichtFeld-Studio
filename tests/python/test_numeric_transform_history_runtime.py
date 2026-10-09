# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native history coverage for numeric transform preview gestures."""

import os

import pytest

from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("space", [0, 1])
@pytest.mark.parametrize("tool,group", [("translate", "pos"), ("rotate", "rot"), ("scale", "scale")])
def test_numeric_transform_native_history(count, space, tool, group):
    endpoint = os.environ.get("LFS_TEST_MCP_ENDPOINT")
    if not endpoint:
        pytest.skip("requires an isolated running app at LFS_TEST_MCP_ENDPOINT")
    setup_code = f'''
import lichtfeld as lf
scene = lf.get_scene()
names = ["numeric_history_probe_" + str(i) for i in range({count})]
for name in names:
    if scene.get_node(name) is None:
        scene.add_group(name)
    lf.set_node_transform(name, [1,0,0,0,0,1,0,0,0,0,1,0,.1,0,0,1])
lf.select_nodes(names)
'''
    setup = _tool(endpoint, "editor_run", {
        "code": setup_code, "show_console": False,
        "wait_for_completion": True, "timeout_ms": 10000,
    })
    assert setup["completed"] and setup["success"], setup
    assert "Traceback" not in setup["output"]["text"], setup
    # Let the app publish selection before the panel checks editability.
    code = f'''
import lichtfeld as lf
from lfs_plugins.transform_controls import TransformControlsController
from types import SimpleNamespace
names = ["numeric_history_probe_" + str(i) for i in range({count})]
assert lf.can_transform_selection(), lf.get_selected_node_names()
lf.undo.clear()
before = [lf.get_node_transform(name) for name in names]
panel = TransformControlsController()
panel._selected = names
panel._active_tool = "builtin.{tool}"
panel._transform_space = {space}
if len(names) == 1:
    panel._update_single_node()
event = SimpleNamespace(current_target=lambda: None)
panel._on_input_focus(event, "transform_{group}_x_str")
for text in ("0", "0.", "0.2", "0.25", "0.250"):
    panel._set_value("{group}", 0, text)
after = [lf.get_node_transform(name) for name in names]
panel._on_input_blur(event, "transform_{group}_x_str")
assert before != after
assert len(lf.undo.undo_names()) == 1, lf.undo.undo_names()
assert lf.undo.undo()
assert [lf.get_node_transform(name) for name in names] == before
assert lf.undo.redo()
assert [lf.get_node_transform(name) for name in names] == after
print("NUMERIC_HISTORY_PASS")
'''
    result = _tool(endpoint, "editor_run", {
        "code": code,
        "show_console": False,
        "wait_for_completion": True,
        "wait_for_output": True,
        "timeout_ms": 10000,
        "output_max_chars": 5000,
    })
    assert result["completed"] and result["success"], result
    assert "NUMERIC_HISTORY_PASS" in str(result), result
