# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Builtin tool definitions.

This module defines all builtin tools as ToolDef instances. Tools are
organized by group and sorted by order within each group.
"""

from __future__ import annotations

from .definition import ToolDef, SubmodeDef, PivotModeDef


_CROP_VOLUME_TYPES = {"CROPBOX", "ELLIPSOID"}
_CROPBOX_TARGET_TYPES = {"SPLAT", "POINTCLOUD"} | _CROP_VOLUME_TYPES


def _poll_builtin_tool_available(tool_id: str) -> bool:
    try:
        import lichtfeld as lf

        is_tool_available = getattr(getattr(lf, "ui", None), "is_tool_available", None)
        return bool(is_tool_available(tool_id)) if callable(is_tool_available) else False
    except Exception:
        return False


def _node_type_name(node) -> str:
    try:
        # nanobind enums: str() returns "NodeType.SPLAT", extract the suffix
        return str(node.type).split(".")[-1]
    except Exception:
        return ""


def _current_scene():
    try:
        import lichtfeld as lf

        get_scene = getattr(lf, "get_scene", None)
        return get_scene() if callable(get_scene) else None
    except Exception:
        return None


def _node_contains_cropbox_target(scene, node) -> bool:
    if node is None:
        return False
    if _node_type_name(node) in _CROPBOX_TARGET_TYPES:
        return True
    if _node_type_name(node) != "DATASET":
        return False

    def contains_model_target(candidate) -> bool:
        if candidate is None:
            return False
        if _node_type_name(candidate) in {"SPLAT", "POINTCLOUD"}:
            return True
        for nested_id in getattr(candidate, "children", []) or []:
            if contains_model_target(scene.get_node_by_id(nested_id)):
                return True
        return False

    for child_id in getattr(node, "children", []) or []:
        if contains_model_target(scene.get_node_by_id(child_id)):
            return True
    return False


def _selected_node_types() -> tuple[str, ...]:
    try:
        import lichtfeld as lf

        scene = _current_scene()
        if scene is None:
            return ()
        selected_names = lf.get_selected_node_names() or []
        node_types: list[str] = []
        for name in selected_names:
            node = scene.get_node(name)
            node_type = _node_type_name(node)
            if node_type:
                node_types.append(node_type)
        return tuple(node_types)
    except Exception:
        return ()


def _selection_is_crop_volume() -> bool:
    return any(node_type in _CROP_VOLUME_TYPES for node_type in _selected_node_types())


def _selection_has_cropbox_target() -> bool:
    try:
        import lichtfeld as lf

        scene = _current_scene()
        if scene is None:
            return False
        selected_names = lf.get_selected_node_names() or []
        return any(
            _node_contains_cropbox_target(scene, scene.get_node(name))
            for name in selected_names
        )
    except Exception:
        return False


def _poll_has_scene(context) -> bool:
    return getattr(context, "has_scene", False)


def _poll_has_gaussians(context) -> bool:
    return (
        getattr(context, "has_scene", False)
        and getattr(context, "num_gaussians", 0) > 0
    )


def _poll_can_select(context) -> bool:
    return _poll_has_gaussians(context) and not _selection_is_crop_volume()


def _poll_can_transform(context) -> bool:
    return bool(getattr(context, "can_transform", False)) and not _selection_is_crop_volume()


def _poll_can_mirror(_context) -> bool:
    return _poll_builtin_tool_available("builtin.mirror") and not _selection_is_crop_volume()


def _poll_can_align(_context) -> bool:
    return _poll_builtin_tool_available("builtin.align") and not _selection_is_crop_volume()


def _poll_can_cropbox(context) -> bool:
    if not _poll_has_scene(context):
        return False
    if _selection_is_crop_volume():
        return True
    if _poll_can_transform(context):
        return True
    return _selection_has_cropbox_target()


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
            SubmodeDef("box", "Box", "box"),
            SubmodeDef("sphere", "Sphere", "sphere"),
        ),
        poll=_poll_can_select,
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
        poll=_poll_can_transform,
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
        poll=_poll_can_transform,
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
        poll=_poll_can_transform,
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
        poll=_poll_can_mirror,
    ),
    ToolDef(
        id="builtin.cropbox",
        label="Crop",
        icon="cropbox",
        group="utility",
        order=70,
        description="Crop objects",
        gizmo="translate",
        poll=_poll_can_cropbox,
    ),
    ToolDef(
        id="builtin.align",
        label="Align",
        icon="align",
        group="utility",
        order=80,
        description="Align to world axes",
        shortcut="6",
        poll=_poll_can_align,
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
