# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""RmlUI color, property-binding, section animation, and model update helpers."""

from dataclasses import dataclass
import math
from typing import Any, Callable


def clamp_unit_channel(value):
    try:
        channel = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(channel):
        return 0.0
    return max(0.0, min(1.0, channel))


def color_channel_byte(color, index):
    try:
        value = color[index]
    except (IndexError, TypeError):
        value = 0.0
    return max(
        0,
        min(255, int(round(clamp_unit_channel(value) * 255.0))),
    )


def color_channel_text(color, index):
    return str(color_channel_byte(color, index))


def color_to_hex(color):
    return (
        f"#{color_channel_byte(color, 0):02x}"
        f"{color_channel_byte(color, 1):02x}"
        f"{color_channel_byte(color, 2):02x}"
    )


def hex_to_color(value):
    text = str(value or "").strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return (
            int(text[0:2], 16) / 255.0,
            int(text[2:4], 16) / 255.0,
            int(text[4:6], 16) / 255.0,
        )
    except ValueError:
        return None


def parse_color_channel(value):
    text = str(value or "").strip()
    if not text:
        return None
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    if "." in text and 0.0 <= parsed <= 1.0:
        parsed *= 255.0
    byte_value = max(0, min(255, int(round(parsed))))
    return byte_value / 255.0


def normalize_color(color):
    channels = []
    for index in range(3):
        try:
            channels.append(clamp_unit_channel(color[index]))
        except (IndexError, TypeError):
            channels.append(0.0)
    return tuple(channels)


def color_component_label(prefix, color, index):
    return f"{prefix}:{color_channel_byte(color, index):>3d}"


def find_ancestor_with_attribute(element, attribute, stop=None):
    """Walk up the DOM tree looking for an element with the given attribute."""
    while element is not None and element != stop:
        if element.has_attribute(attribute):
            return element
        element = element.parent()
    return None


def _select_all_text(element):
    if element is None:
        return False
    try:
        return bool(element.select())
    except Exception:
        return False


def bind_select_all_on_focus(element):
    """Select all text when the given input element receives focus."""
    if element is None:
        return None
    if element.get_attribute("data-select-all-bound", "") == "1":
        return element

    element.set_attribute("data-select-all-bound", "1")
    element.add_event_listener("focus", lambda _event, el=element: _select_all_text(el))
    return element


def bind_committed_text_input(
    element,
    key,
    *,
    escape_revert=None,
    capture=None,
    restore=None,
    commit=None,
    on_focus=None,
    on_blur=None,
):
    """Bind common panel-style text input behavior to a retained input element."""
    if element is None:
        return None

    bind_select_all_on_focus(element)

    if escape_revert is not None and capture is not None and restore is not None:
        escape_revert.bind(element, key, capture, restore)

    if on_focus is not None:
        element.add_event_listener("focus", lambda _event, k=str(key): on_focus(k))

    if commit is not None:
        def _commit_on_linebreak(event, k=str(key)):
            if not event.get_bool_parameter("linebreak", False):
                return
            commit(k)

        def _commit_on_blur(_event, k=str(key)):
            commit(k)
            if on_blur is not None:
                on_blur(k)

        element.add_event_listener("change", _commit_on_linebreak)
        element.add_event_listener("blur", _commit_on_blur)
    elif on_blur is not None:
        element.add_event_listener("blur", lambda _event, k=str(key): on_blur(k))

    return element


@dataclass
class _EscapeRevertBinding:
    element: object
    capture: Callable[[], Any]
    restore: Callable[[Any], None]
    snapshot: Any = None


class EscapeRevertController:
    """Restore focused text inputs to their pre-edit value on host-dispatched cancel."""

    def __init__(self):
        self._bindings = {}

    def clear(self):
        self._bindings.clear()

    def bind(self, element, key, capture, restore):
        if element is None:
            return None

        binding_key = str(key)
        self._bindings[binding_key] = _EscapeRevertBinding(
            element=element,
            capture=capture,
            restore=restore,
        )
        element.add_event_listener("focus", lambda _event, k=binding_key: self._capture_binding(k))
        element.add_event_listener("blur", lambda _event, k=binding_key: self._clear_binding(k))
        element.add_event_listener("escapecancel", lambda event, k=binding_key: self._restore_binding(k, event))
        return element

    def recapture(self, key):
        """Re-take the pre-edit snapshot for an already-bound key.

        Callers that retarget a live edit at different underlying state need the
        Escape snapshot to follow it; without this they would have to reach into
        the private capture.
        """
        self._capture_binding(str(key))

    def _restore_binding(self, key, event):
        binding = self._bindings.get(key)
        if binding is None or binding.element.parent() is None:
            return False

        snapshot = binding.snapshot if binding.snapshot is not None else binding.capture()
        binding.restore(snapshot)
        event.stop_propagation()
        return True

    def _capture_binding(self, key):
        binding = self._bindings.get(key)
        if binding is None:
            return
        binding.snapshot = binding.capture()

    def _clear_binding(self, key):
        binding = self._bindings.get(key)
        if binding is not None:
            binding.snapshot = None


def _section_duration(height_px):
    height_px = max(0.0, float(height_px))
    return min(0.28, 0.16 + height_px / 2200.0)


def _apply_section_visual_state(expanded, header_element=None, arrow_element=None):
    if header_element:
        header_element.set_class("is-expanded", expanded)
        header_element.set_class("is-collapsed", not expanded)
    if arrow_element:
        arrow_element.set_text(chr(0x25B6))
        arrow_element.set_class("is-expanded", expanded)
        arrow_element.set_class("is-collapsed", not expanded)


def sync_section_state(content_element, expanded, header_element=None, arrow_element=None):
    """Apply the steady-state visual state for a collapsible section."""
    if not content_element:
        return

    _apply_section_visual_state(expanded, header_element, arrow_element)
    content_element.set_class("collapsed", not expanded)

    content_element.remove_property("max-height")
    content_element.remove_property("opacity")
    content_element.remove_property("pointer-events")


def animate_section_toggle(content_element, expanding, arrow_element=None,
                           header_element=None):
    """Animate a section open/close with synchronized arrow and header state."""
    if not content_element:
        return

    _apply_section_visual_state(expanding, header_element, arrow_element)

    if expanding:
        content_element.set_class("collapsed", False)
        content_element.remove_property("pointer-events")

        current_h = max(content_element.client_height, 0)
        target_h = max(content_element.scroll_height, current_h)
        if target_h <= 0:
            sync_section_state(content_element, True, header_element, arrow_element)
            return
        duration = _section_duration(target_h)
        fade_duration = max(0.1, min(0.18, duration * 0.7))
        content_element.animate("max-height", f"{target_h}px", duration, "cubic-out",
                                f"{current_h}px" if current_h > 0 else "0px",
                                remove_on_complete=True)
        content_element.animate("opacity", "1", fade_duration, "quadratic-out",
                                remove_on_complete=True)
        return

    content_element.set_property("pointer-events", "none")
    current_h = max(content_element.client_height, 0)
    target_h = max(content_element.scroll_height, current_h)
    if target_h <= 0:
        sync_section_state(content_element, False, header_element, arrow_element)
        return
    duration = _section_duration(target_h)
    fade_duration = max(0.1, min(0.18, duration * 0.7))
    content_element.animate("max-height", "0px", duration, "cubic-in-out",
                            f"{target_h}px")
    content_element.animate("opacity", "0", fade_duration, "quadratic-out", "1")




def request_model_update(handle):
    """Schedule a dirty-policy panel update without dirtying every model variable."""
    request_update = getattr(handle, "request_update", None)
    if callable(request_update):
        request_update()
    else:
        handle.dirty_all()
