# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Menu bar structure and registry.

Menus are defined declaratively and registered automatically without
needing lf.register_class() calls.
"""

from __future__ import annotations

from typing import Any

import lichtfeld as lf

# Global list of menu classes - populated at import time
_MENU_CLASSES: list[type] = []


def register_menu(menu_class: type) -> type:
    """Decorator to register a menu class.

    This replaces the need to call lf.register_class(MenuClass).
    Menus are registered in the order they are defined.

    Usage:
        @register_menu
        class FileMenu:
            label = "menu.file"
            location = "MENU_BAR"
            order = 10

            def menu_items(self):
                ...
    """
    _MENU_CLASSES.append(menu_class)
    return menu_class


def unregister_module(module_name: str) -> None:
    """Remove all menu classes registered by a module."""
    _MENU_CLASSES[:] = [
        cls for cls in _MENU_CLASSES
        if getattr(cls, "__module__", "") != module_name
    ]


def menu_separator() -> dict[str, Any]:
    """Create a separator entry for a declarative menu schema."""
    return {"type": "separator"}


def menu_submenu(label: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a submenu entry for a declarative menu schema."""
    return {"type": "submenu", "label": label, "items": list(items)}


def menu_operator(operator_cls_or_id: Any, label: str = "") -> dict[str, Any]:
    """Create an operator-backed menu entry.

    Accepts either an operator class or an operator id string.
    """
    operator_id = operator_cls_or_id
    resolved_label = label

    if hasattr(operator_cls_or_id, "_class_id"):
        operator_id = operator_cls_or_id._class_id()
        if not resolved_label:
            resolved_label = lf.ui.tr(getattr(operator_cls_or_id, "label", operator_id))

    return {
        "type": "operator",
        "operator_id": str(operator_id),
        "label": resolved_label,
    }


def menu_action(label: str, callback: Any, shortcut: str = "", enabled: bool = True) -> dict[str, Any]:
    """Create a plain callback-backed menu entry."""
    return {
        "type": "item",
        "label": label,
        "callback": callback,
        "shortcut": shortcut,
        "enabled": enabled,
    }


def menu_toggle(label: str, callback: Any, selected: bool,
                shortcut: str = "", enabled: bool = True) -> dict[str, Any]:
    """Create a toggle menu entry with a checkmark state."""
    return {
        "type": "toggle",
        "label": label,
        "callback": callback,
        "shortcut": shortcut,
        "enabled": enabled,
        "selected": selected,
    }


def get_menu_classes() -> list[type]:
    """Get all registered menu classes in order.

    Returns:
        List of menu classes sorted by their 'order' attribute.
    """
    # Import menu modules to trigger @register_menu decorators
    from .. import file_menu, edit_menu, tools_menu, view_menu, help_menu  # noqa: F401

    return sorted(_MENU_CLASSES, key=lambda m: getattr(m, "order", 100))


def get_menu_bar_entries() -> list[tuple[str, str, int, type]]:
    """Get menu bar entries for C++ rendering.

    Returns:
        List of (idname, label, order, menu_class) tuples.
    """
    result = []
    for cls in get_menu_classes():
        location = getattr(cls, "location", "")
        if location == "MENU_BAR":
            idname = f"{cls.__module__}.{cls.__qualname__}"
            label = getattr(cls, "label", "")
            order = getattr(cls, "order", 100)
            result.append((idname, label, order, cls))
    return result


def _clear_menus():
    """Clear all registered menus (for testing/reload)."""
    _MENU_CLASSES.clear()
