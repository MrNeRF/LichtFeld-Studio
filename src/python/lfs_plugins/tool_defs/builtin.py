# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Builtin tool definitions.

This module defines all builtin tools as ToolDef instances. Tools are
organized by group and sorted by order within each group.
"""

from __future__ import annotations

from .definition import ToolDef, SubmodeDef, PivotModeDef


def _poll_has_scene(context) -> bool:
    return getattr(context, "has_scene", False)


def _poll_has_gaussians(context) -> bool:
    return (
        getattr(context, "has_scene", False)
        and getattr(context, "num_gaussians", 0) > 0
    )


BUILTIN_TOOLS: tuple[ToolDef, ...] = (
    ToolDef(
        id="builtin.select",
        label="Select",
        icon="selection",
        group="select",
        order=10,
        description="Select gaussians",
        shortcut="1",
        submodes=(
            SubmodeDef("centers", "Centers", "circle-dot"),
            SubmodeDef("rectangle", "Rectangle", "rectangle"),
            SubmodeDef("polygon", "Polygon", "polygon"),
            SubmodeDef("lasso", "Lasso", "lasso"),
            SubmodeDef("rings", "Rings", "ring"),
            SubmodeDef("color", "Color", "color-picker"),
        ),
        poll=_poll_has_gaussians,
    ),
    ToolDef(
        id="builtin.translate",
        label="Move",
        icon="translation",
        group="transform",
        order=20,
        description="Move selection",
        shortcut="2",
        gizmo="translate",
        submodes=(
            SubmodeDef("local", "Local", "local"),
            SubmodeDef("world", "World", "world"),
        ),
        pivot_modes=(
            PivotModeDef("origin", "Origin", "circle-dot"),
            PivotModeDef("bounds", "Bounds", "box"),
        ),
        poll=_poll_has_scene,
    ),
    ToolDef(
        id="builtin.rotate",
        label="Rotate",
        icon="rotation",
        group="transform",
        order=30,
        description="Rotate selection",
        shortcut="3",
        gizmo="rotate",
        submodes=(
            SubmodeDef("local", "Local", "local"),
            SubmodeDef("world", "World", "world"),
        ),
        pivot_modes=(
            PivotModeDef("origin", "Origin", "circle-dot"),
            PivotModeDef("bounds", "Bounds", "box"),
        ),
        poll=_poll_has_scene,
    ),
    ToolDef(
        id="builtin.scale",
        label="Scale",
        icon="scaling",
        group="transform",
        order=40,
        description="Scale selection",
        shortcut="4",
        gizmo="scale",
        submodes=(
            SubmodeDef("local", "Local", "local"),
            SubmodeDef("world", "World", "world"),
        ),
        pivot_modes=(
            PivotModeDef("origin", "Origin", "circle-dot"),
            PivotModeDef("bounds", "Bounds", "box"),
        ),
        poll=_poll_has_scene,
    ),
    ToolDef(
        id="builtin.mirror",
        label="Mirror",
        icon="mirror",
        group="transform",
        order=50,
        description="Mirror selection",
        shortcut="5",
        submodes=(
            SubmodeDef("x", "X Axis", "mirror-x"),
            SubmodeDef("y", "Y Axis", "mirror-y"),
            SubmodeDef("z", "Z Axis", "mirror-z"),
        ),
        poll=_poll_has_gaussians,
    ),
    ToolDef(
        id="builtin.cropbox",
        label="Crop",
        icon="cropbox",
        group="utility",
        order=70,
        description="Crop objects",
        gizmo="translate",
        poll=_poll_has_scene,
    ),
    ToolDef(
        id="builtin.align",
        label="Align",
        icon="align",
        group="utility",
        order=80,
        description="Align to world axes",
        shortcut="6",
        poll=_poll_has_scene,
    ),
)


def get_tool_by_id(tool_id: str) -> ToolDef | None:
    """Get a builtin tool by its ID.

    Args:
        tool_id: Tool ID (e.g., "builtin.translate").

    Returns:
        The ToolDef if found, None otherwise.
    """
    for tool in BUILTIN_TOOLS:
        if tool.id == tool_id:
            return tool
    return None


def get_tools_by_group(group: str) -> list[ToolDef]:
    """Get all builtin tools in a group, sorted by order.

    Args:
        group: Group name (e.g., "transform").

    Returns:
        List of tools in the group, sorted by order.
    """
    return sorted(
        [t for t in BUILTIN_TOOLS if t.group == group],
        key=lambda t: t.order,
    )


def get_all_groups() -> list[str]:
    """Get all unique group names, in order of first appearance."""
    seen = set()
    groups = []
    for tool in BUILTIN_TOOLS:
        if tool.group not in seen:
            seen.add(tool.group)
            groups.append(tool.group)
    return groups
