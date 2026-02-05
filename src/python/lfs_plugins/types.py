# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base types for LichtFeld plugins."""

from typing import Set

from .props import PropertyGroup


class Event:
    """Event wrapper for modal operators.

    Attributes:
        type: Event type string ('MOUSEMOVE', 'LEFTMOUSE', 'RIGHTMOUSE',
              'MIDDLEMOUSE', 'KEY_A'-'KEY_Z', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
              'ESC', 'RET', 'SPACE', etc.)
        value: Event value ('PRESS', 'RELEASE', 'NOTHING')
        mouse_x: Mouse X position (viewport coordinates)
        mouse_y: Mouse Y position (viewport coordinates)
        mouse_region_x: Mouse X position relative to region
        mouse_region_y: Mouse Y position relative to region
        delta_x: Mouse delta X (for MOUSEMOVE events)
        delta_y: Mouse delta Y (for MOUSEMOVE events)
        scroll_x: Scroll X offset (for WHEELUPMOUSE/WHEELDOWNMOUSE)
        scroll_y: Scroll Y offset (for WHEELUPMOUSE/WHEELDOWNMOUSE)
        shift: True if Shift modifier is held
        ctrl: True if Ctrl modifier is held
        alt: True if Alt modifier is held
        pressure: Tablet pressure (1.0 for mouse)
        over_gui: True if mouse is over a GUI element
        key_code: Raw GLFW key code for KEY events
    """

    type: str
    value: str
    mouse_x: float
    mouse_y: float
    mouse_region_x: float
    mouse_region_y: float
    delta_x: float
    delta_y: float
    scroll_x: float
    scroll_y: float
    shift: bool
    ctrl: bool
    alt: bool
    pressure: float
    over_gui: bool
    key_code: int


class Operator(PropertyGroup):
    """Base class for operators.

    Operators can have properties defined as class attributes using Property types.
    These properties are automatically registered and accessible via layout.prop().

    Attributes:
        label: Display label
        description: Tooltip
        options: Options like {'UNDO', 'BLOCKING'}

    Return Values:
        execute() and invoke() can return either:
        - Set format (legacy): {'FINISHED'}, {'CANCELLED'}, {'RUNNING_MODAL'}, {'PASS_THROUGH'}
        - Dict format (rich returns): {'status': 'FINISHED', 'key1': value1, ...}
    """

    label: str = ""
    description: str = ""
    options: Set[str] = set()

    @classmethod
    def _class_id(cls) -> str:
        return f"{cls.__module__}.{cls.__qualname__}"

    @classmethod
    def poll(cls, context) -> bool:
        """Check if the operator can run in the current context."""
        return True

    def invoke(self, context, event: Event) -> set:
        """Called when the operator is invoked."""
        return self.execute(context)

    def execute(self, context) -> set:
        """Execute the operator."""
        return {"FINISHED"}

    def modal(self, context, event: Event) -> set:
        """Handle modal events during operator execution."""
        return {"FINISHED"}

    def cancel(self, context):
        """Called when the operator is cancelled."""
        pass


class Panel:
    """Base class for UI panels.

    This is a minimal base class. Panels define:
    - label: Display name
    - draw(layout): Render the panel content
    - poll(context): Optional visibility check
    """

    label: str = ""

    @classmethod
    def _class_id(cls) -> str:
        return f"{cls.__module__}.{cls.__qualname__}"

    @classmethod
    def poll(cls, context) -> bool:
        """Check if the panel should be drawn."""
        return True

    def draw(self, layout):
        """Draw the panel contents."""
        pass


class Menu:
    """Base class for menu definitions."""

    label: str = ""
    location: str = "FILE"
    order: int = 100

    def draw(self, layout):
        pass


