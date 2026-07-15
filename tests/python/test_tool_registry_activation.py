# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for Python toolbar tool activation."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys


def _import_tools_with_runtime_stub(monkeypatch, lf_stub):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) in sys.path:
        sys.path.remove(str(source_python))
    sys.path.insert(0, str(source_python))

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    sys.modules.pop("lfs_plugins.tools", None)
    sys.modules.pop("lfs_plugins.op_context", None)

    tools = import_module("lfs_plugins.tools")
    op_context = import_module("lfs_plugins.op_context")
    monkeypatch.setattr(
        op_context,
        "get_context",
        lambda: SimpleNamespace(has_scene=True, num_gaussians=100),
        raising=False,
    )
    tools.ToolRegistry._active_tool_id = ""
    tools.ToolRegistry._custom_tools.clear()
    return tools


def test_builtin_select_activation_uses_toolbar_tool_and_resets_brush_mode(monkeypatch):
    calls = []
    state = SimpleNamespace(active_submode="rectangle")
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        ops=SimpleNamespace(cancel_modal=lambda: calls.append(("cancel_modal",))),
        get_content_type=lambda: "splat_files",
        clear_gizmo=lambda: calls.append(("clear_gizmo",)),
        set_active_tool=lambda tool_id: calls.append(("set_active_tool", tool_id)),
        set_active_operator=lambda tool_id, gizmo="": calls.append(("set_active_operator", tool_id, gizmo)),
        set_selection_mode=lambda mode: (
            setattr(state, "active_submode", mode),
            calls.append(("set_selection_mode", mode)),
        ),
        clear_active_operator=lambda: calls.append(("clear_active_operator",)),
    )
    lf_stub.can_transform_selection = lambda: True

    tools = _import_tools_with_runtime_stub(monkeypatch, lf_stub)

    assert tools.ToolRegistry.set_active("builtin.select") is True
    assert ("set_active_tool", "builtin.select") in calls
    assert not any(call[0] == "set_active_operator" for call in calls)
    assert ("set_selection_mode", "centers") in calls
    assert state.active_submode == "centers"


def test_crop_activation_uses_active_operator(monkeypatch):
    calls = []
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        ops=SimpleNamespace(cancel_modal=lambda: calls.append(("cancel_modal",))),
        get_content_type=lambda: "splat_files",
        clear_gizmo=lambda: calls.append(("clear_gizmo",)),
        set_active_tool=lambda tool_id: calls.append(("set_active_tool", tool_id)),
        set_active_operator=lambda tool_id, gizmo="": calls.append(("set_active_operator", tool_id, gizmo)),
        set_gizmo_type=lambda gizmo: calls.append(("set_gizmo_type", gizmo)),
        clear_active_operator=lambda: calls.append(("clear_active_operator",)),
    )
    lf_stub.can_transform_selection = lambda: True

    tools = _import_tools_with_runtime_stub(monkeypatch, lf_stub)

    assert tools.ToolRegistry.set_active("builtin.cropbox") is True
    assert ("set_active_operator", "builtin.cropbox", "translate") in calls
    assert ("set_gizmo_type", "translate") in calls
    assert not any(call == ("set_active_tool", "builtin.cropbox") for call in calls)


def test_clear_active_clears_cpp_toolbar_tool(monkeypatch):
    calls = []
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        ops=SimpleNamespace(cancel_modal=lambda: calls.append(("cancel_modal",))),
        clear_gizmo=lambda: calls.append(("clear_gizmo",)),
        set_tool=lambda tool: calls.append(("set_tool", tool)),
        clear_active_operator=lambda: calls.append(("clear_active_operator",)),
    )

    tools = _import_tools_with_runtime_stub(monkeypatch, lf_stub)
    tools.ToolRegistry._active_tool_id = "builtin.select"

    tools.ToolRegistry.clear_active()

    assert ("set_tool", "none") in calls
    assert ("clear_active_operator",) in calls
    assert tools.ToolRegistry.get_active_id() == ""
