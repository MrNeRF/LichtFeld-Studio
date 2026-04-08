# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dataclass definitions for tools.

Tools are defined as immutable dataclasses rather than classes with inheritance.
This makes tool definitions declarative and easy to reason about.

Example:
    tool = ToolDef(
        id="builtin.translate",
        label="Move",
        icon="translation",
        group="transform",
        gizmo="translate",
        submodes=(
            SubmodeDef("local", "Local", "local"),
            SubmodeDef("world", "World", "world"),
        ),
        pivot_modes=(
            PivotModeDef("origin", "Origin", "circle-dot"),
            PivotModeDef("bounds", "Bounds Center", "box"),
        ),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ContextLike(Protocol):
    """Protocol for context objects passed to poll functions."""

    has_scene: bool
    num_gaussians: int


@dataclass(frozen=True)
class SubmodeDef:
    """Definition for a tool submode.

    Submodes are tool variations that share the same base tool but have
    different behaviors (e.g., local vs world coordinates).

    Args:
        id: Unique identifier for this submode.
        label: Display label.
        icon: Icon name.
        shortcut: Optional keyboard shortcut.
    """

    id: str
    label: str
    icon: str
    shortcut: str = ""


@dataclass(frozen=True)
class PivotModeDef:
    """Definition for a tool pivot mode.

    Pivot modes define the center of transformation operations.

    Args:
        id: Unique identifier for this pivot mode.
        label: Display label.
        icon: Icon name.
    """

    id: str
    label: str
    icon: str


@dataclass(frozen=True)
class ToolDef:
    """Immutable definition of a toolbar tool.

    ToolDef replaces the class-based Tool inheritance pattern with a
    declarative dataclass. All tool metadata is explicit in the definition.

    Args:
        id: Unique tool identifier (e.g., "builtin.translate").
        label: Display label.
        icon: Icon name.
        group: Toolbar group for organization ("select", "transform", "paint", "utility").
        order: Sort order within group.
        description: Tooltip text.
        shortcut: Keyboard shortcut.
        gizmo: Gizmo type ("translate", "rotate", "scale", or "").
        operator: Operator to invoke when tool is activated.
        submodes: Available submodes for this tool.
        pivot_modes: Available pivot modes for transform tools.
        poll: Optional callable to check if tool is available.
        plugin_name: Plugin name for custom icons.
        plugin_path: Plugin path for custom icons.
        action_only: Invoke operator without becoming the active tool.
        selected: Optional callable for custom toolbar highlight state.
    """

    id: str
    label: str
    icon: str
    group: str = "default"
    order: int = 100
    description: str = ""
    shortcut: str = ""
    gizmo: str = ""
    operator: str = ""
    submodes: tuple[SubmodeDef, ...] = ()
    pivot_modes: tuple[PivotModeDef, ...] = ()
    poll: Callable[[Any], bool] | None = None
    plugin_name: str = ""
    plugin_path: str = ""
    action_only: bool = False
    selected: Callable[[Any], bool] | None = None

    def can_activate(self, context: Any) -> bool:
        """Check if this tool can be activated in the given context.

        Args:
            context: Application context object.

        Returns:
            True if tool can be activated.
        """
        if self.poll is None:
            return True
        return self.poll(context)

    def to_dict(self) -> dict:
        """Convert to dictionary for C++ interop."""
        return {
            "id": self.id,
            "label": self.label,
            "icon": self.icon,
            "group": self.group,
            "order": self.order,
            "description": self.description,
            "shortcut": self.shortcut,
            "gizmo": self.gizmo,
            "operator": self.operator,
            "submodes": [
                {"id": s.id, "label": s.label, "icon": s.icon}
                for s in self.submodes
            ],
            "pivot_modes": [
                {"id": p.id, "label": p.label, "icon": p.icon}
                for p in self.pivot_modes
            ],
            "plugin_name": self.plugin_name,
            "plugin_path": self.plugin_path,
            "action_only": self.action_only,
        }
