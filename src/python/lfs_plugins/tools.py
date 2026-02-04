# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Toolbar tools using declarative ToolDef system."""

from typing import Optional

import lichtfeld as lf

from .tool_defs.builtin import BUILTIN_TOOLS, get_tool_by_id
from .tool_defs.definition import ToolDef


class ToolRegistry:
    """Manages tool definitions (builtin + custom) and activation state."""

    _active_tool_id: str = ""
    _custom_tools: dict[str, ToolDef] = {}

    @classmethod
    def register_tool(cls, tool: ToolDef) -> None:
        cls._custom_tools[tool.id] = tool

    @classmethod
    def unregister_tool(cls, tool_id: str) -> None:
        cls._custom_tools.pop(tool_id, None)
        if cls._active_tool_id == tool_id:
            cls.set_active("builtin.select")

    @classmethod
    def get(cls, tool_id: str) -> Optional[ToolDef]:
        """Get tool definition by ID (builtins first, then custom)."""
        tool = get_tool_by_id(tool_id)
        if tool:
            return tool
        return cls._custom_tools.get(tool_id)

    @classmethod
    def get_all(cls) -> list[ToolDef]:
        """Get all tool definitions. Builtins keep declaration order; customs appended sorted."""
        if not cls._custom_tools:
            return list(BUILTIN_TOOLS)
        custom = sorted(cls._custom_tools.values(), key=lambda t: (t.group, t.order))
        return list(BUILTIN_TOOLS) + custom

    @classmethod
    def set_active(cls, tool_id: str) -> bool:
        """Set the active tool by ID."""
        tool = cls.get(tool_id)
        if not tool:
            return False

        from .op_context import get_context

        context = get_context()
        if not tool.can_activate(context):
            return False

        lf.ui.ops.cancel_modal()
        lf.ui.clear_gizmo()

        cls._active_tool_id = tool_id

        gizmo = tool.gizmo or ""
        lf.ui.set_active_operator(tool_id, gizmo)

        if tool_id == "builtin.select" and not lf.ui.get_active_submode():
            lf.ui.set_selection_mode("centers")

        if tool.gizmo and not tool.operator:
            lf.ui.set_gizmo_type(tool.gizmo)
        elif tool.operator:
            lf.ui.ops.invoke(tool.operator)

        return True

    @classmethod
    def clear_active(cls):
        lf.ui.ops.cancel_modal()
        lf.ui.clear_gizmo()
        lf.ui.clear_active_operator()
        cls._active_tool_id = ""

    @classmethod
    def get_active(cls) -> Optional[ToolDef]:
        """Get the active tool definition."""
        return cls.get(cls._active_tool_id)

    @classmethod
    def get_active_id(cls) -> str:
        """Get the active tool ID."""
        return cls._active_tool_id

    @classmethod
    def clear(cls):
        """Clear active tool state and custom tools."""
        cls._active_tool_id = ""
        cls._custom_tools.clear()


def register():
    """Initialize tools system."""
    if BUILTIN_TOOLS:
        ToolRegistry.set_active("builtin.select")


def unregister():
    """Cleanup tools system."""
    ToolRegistry.clear()
