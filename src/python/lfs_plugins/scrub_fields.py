# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reusable scrub-field controller for retained RmlUI panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import rml_widgets as w
from .rml_keys import KI_ESCAPE


DRAG_THRESHOLD_PX = 4.0


@dataclass(frozen=True)
class ScrubFieldSpec:
    min_value: float
    max_value: float
    step: float
    fmt: str
    data_type: type = float
    pixels_per_step: float = 12.0


@dataclass
class _ScrubFieldState:
    prop: str
    spec: ScrubFieldSpec
    field: object
    fill: object
    display: object
    input_el: object
    dragging: bool = False
    editing: bool = False
    drag_start_mouse_x: float = 0.0
    edit_value_before: float = 0.0
    commit_on_blur: bool = False
    cancel_on_blur: bool = False


class ScrubFieldController:
    """Turns native range inputs into Blender-style scrub fields."""

    def __init__(self, specs: dict[str, ScrubFieldSpec],
                 get_value: Callable[[str], float],
                 set_value: Callable[[str, float], None]):
        self._specs = dict(specs)
        self._get_value = get_value
        self._set_value = set_value
        self._doc = None
        self._fields: dict[str, _ScrubFieldState] = {}
        self._active_prop: str | None = None

    def mount(self, doc) -> None:
        self._doc = doc
        body = doc.get_element_by_id("body")
        target = body if body is not None else doc
        target.add_event_listener("mousemove", self._on_body_mousemove)
        target.add_event_listener("mouseup", self._on_body_mouseup)
        doc.add_event_listener("keydown", self._on_keydown)
        self.sync_all()

    def unmount(self) -> None:
        self._fields.clear()
        self._active_prop = None
        self._doc = None

    def set_spec(self, prop: str, spec: ScrubFieldSpec) -> bool:
        self._specs[prop] = spec
        state = self._fields.get(prop)
        if state is None:
            return False
        state.spec = spec
        if state.editing:
            return True
        return self._sync_field(state)

    def sync_all(self) -> bool:
        if self._doc is None:
            return False

        changed = False
        changed |= self._prune_stale_fields()
        changed |= self._discover_fields()
        for state in self._fields.values():
            if state.editing:
                continue
            changed |= self._sync_field(state)
        return changed

    def _prune_stale_fields(self) -> bool:
        stale_props = [
            prop for prop, state in self._fields.items()
            if state.field.parent() is None or state.display.parent() is None
        ]
        if not stale_props:
            return False
        for prop in stale_props:
            if self._active_prop == prop:
                self._active_prop = None
            self._fields.pop(prop, None)
        return True

    def _discover_fields(self) -> bool:
        if self._doc is None:
            return False

        changed = False
        for range_input in self._doc.query_selector_all('input.setting-slider[type="range"]'):
            prop = range_input.get_attribute("data-value", "")
            if not prop or prop in self._fields or prop not in self._specs:
                continue

            row = range_input.parent()
            if row is None:
                continue

            value_label = row.query_selector(".slider-value")
            state = self._build_field(row, range_input, value_label, prop, self._specs[prop])
            if state is None:
                continue

            self._fields[prop] = state
            changed = True

        return changed

    def _build_field(self, row, range_input, value_label, prop: str, spec: ScrubFieldSpec):
        if self._doc is None:
            return None

        field = self._doc.create_element("div")
        field.set_class_names("scrub-field")
        field.set_id(f"scrub-{prop.replace('_', '-')}")
        field.set_attribute("data-prop", prop)

        fill = field.append_child("div")
        fill.set_class_names("scrub-field-fill")

        display = field.append_child("span")
        display.set_class_names("scrub-field-display")

        input_el = field.append_child("input")
        input_el.set_class_names("scrub-field-input")
        input_el.set_attribute("type", "text")
        input_el.set_attribute("data-prop", prop)
        w.bind_select_all_on_focus(input_el)

        row.insert_before(field, range_input)
        row.remove_child(range_input)
        if value_label is not None:
            row.remove_child(value_label)

        state = _ScrubFieldState(
            prop=prop,
            spec=spec,
            field=field,
            fill=fill,
            display=display,
            input_el=input_el,
        )

        field.add_event_listener("mousedown", lambda ev, p=prop: self._on_field_mousedown(p, ev))
        input_el.add_event_listener("change", lambda ev, p=prop: self._on_input_change(p, ev))
        input_el.add_event_listener("blur", lambda ev, p=prop: self._on_input_blur(p, ev))

        self._sync_field(state)
        return state

    def _on_field_mousedown(self, prop: str, event) -> None:
        state = self._fields.get(prop)
        if state is None or state.editing:
            return
        if int(event.get_parameter("button", "0")) != 0:
            return

        self._cancel_other_edits(prop)
        if self._active_prop is not None and self._active_prop != prop:
            self._finish_active_drag(False)

        mx = self._event_mouse_x(event)
        state.dragging = False
        state.drag_start_mouse_x = mx
        state.edit_value_before = self._safe_value(prop)
        self._active_prop = prop

    def _on_body_mousemove(self, event) -> None:
        if self._active_prop is None:
            return

        state = self._fields.get(self._active_prop)
        if state is None:
            self._active_prop = None
            return

        mx = self._event_mouse_x(event)
        drag_dx = mx - state.drag_start_mouse_x
        if not state.dragging and abs(drag_dx) < DRAG_THRESHOLD_PX:
            return

        if not state.dragging:
            state.dragging = True
            state.field.set_class("is-dragging", True)

        self._apply_drag_position(state, mx)
        event.stop_propagation()

    def _on_body_mouseup(self, event) -> None:
        if self._active_prop is None:
            return
        open_editor = self._fields.get(self._active_prop) is not None and not self._fields[self._active_prop].dragging
        self._finish_active_drag(open_editor)
        event.stop_propagation()

    def _on_keydown(self, event) -> None:
        key = int(event.get_parameter("key_identifier", "0"))
        if key != KI_ESCAPE:
            return

        if self._active_prop is not None:
            state = self._fields.get(self._active_prop)
            if state is not None:
                self._restore_edit_value(state)
                state.dragging = False
                state.field.set_class("is-dragging", False)
                self._active_prop = None
                event.stop_propagation()
                return

        for state in self._fields.values():
            if not state.editing:
                continue
            self._restore_edit_value(state)
            state.cancel_on_blur = True
            state.input_el.blur()
            event.stop_propagation()
            return

    def _on_input_change(self, prop: str, event) -> None:
        state = self._fields.get(prop)
        if state is None or not state.editing:
            return
        if not event.get_bool_parameter("linebreak", False):
            return

        state.commit_on_blur = True
        self._commit_edit_value(state)
        state.input_el.blur()

    def _on_input_blur(self, prop: str, _event) -> None:
        state = self._fields.get(prop)
        if state is None or not state.editing:
            return

        if state.cancel_on_blur:
            self._exit_edit(state)
            return

        if state.commit_on_blur:
            self._exit_edit(state)
            return

        self._exit_edit(state)

    def _cancel_other_edits(self, current_prop: str) -> None:
        for prop, state in self._fields.items():
            if prop == current_prop or not state.editing:
                continue
            state.cancel_on_blur = True
            state.input_el.blur()

    def _finish_active_drag(self, open_editor: bool) -> None:
        if self._active_prop is None:
            return

        prop = self._active_prop
        self._active_prop = None
        state = self._fields.get(prop)
        if state is None:
            return

        was_dragging = state.dragging
        state.dragging = False
        state.field.set_class("is-dragging", False)

        if open_editor and not was_dragging and not state.editing:
            self._enter_edit(state)

    def _enter_edit(self, state: _ScrubFieldState) -> None:
        state.editing = True
        state.commit_on_blur = False
        state.cancel_on_blur = False
        state.edit_value_before = self._safe_value(state.prop)
        state.field.set_class("is-editing", True)
        text = self._format_value(state.spec, state.edit_value_before)
        state.input_el.set_attribute("value", text)
        state.input_el.focus()
        state.input_el.select()

    def _exit_edit(self, state: _ScrubFieldState) -> None:
        state.editing = False
        state.commit_on_blur = False
        state.cancel_on_blur = False
        state.field.set_class("is-editing", False)
        self._sync_field(state)

    def _restore_edit_value(self, state: _ScrubFieldState) -> None:
        self._apply_value(state, state.edit_value_before)
        state.input_el.set_attribute("value", self._format_value(state.spec, state.edit_value_before))

    def _commit_edit_value(self, state: _ScrubFieldState) -> None:
        text = state.input_el.get_attribute("value", "").strip()
        if not text:
            return

        try:
            if state.spec.data_type is int:
                value = int(float(text))
            else:
                value = float(text)
        except (TypeError, ValueError):
            return

        self._apply_value(state, value)

    def _apply_drag_position(self, state: _ScrubFieldState, mouse_x: float) -> None:
        width = max(float(getattr(state.field, "absolute_width", 0.0)), 1.0)
        left = float(getattr(state.field, "absolute_left", 0.0))
        x = mouse_x - left
        t = max(0.0, min(1.0, x / width))
        value = state.spec.min_value + t * (state.spec.max_value - state.spec.min_value)
        value = self._snap_value(state.spec, value)
        self._apply_value(state, value)

    def _apply_value(self, state: _ScrubFieldState, value: float) -> None:
        next_value = self._clamp_value(state.spec, value)
        current_value = self._safe_value(state.prop)
        if self._values_equal(state.spec, current_value, next_value):
            self._sync_field(state)
            return

        try:
            self._set_value(state.prop, next_value)
        except (RuntimeError, TypeError, ValueError):
            return

        self._sync_field(state)

    def _sync_field(self, state: _ScrubFieldState) -> bool:
        value = self._safe_value(state.prop)
        text = self._format_value(state.spec, value)
        pct = self._fill_percent(state.spec, value)
        changed = False

        if state.display.get_inner_rml() != text:
            state.display.set_text(text)
            changed = True

        if state.input_el.get_attribute("value", "") != text:
            state.input_el.set_attribute("value", text)
            changed = True

        pct_str = f"{pct:.3f}%"
        if state.fill.get_attribute("data-fill-width", "") != pct_str:
            state.fill.set_attribute("data-fill-width", pct_str)
            state.fill.set_property("width", pct_str)
            changed = True

        return changed

    def _safe_value(self, prop: str) -> float:
        try:
            return self._clamp_value(self._specs[prop], self._get_value(prop))
        except (KeyError, RuntimeError, TypeError, ValueError):
            spec = self._specs[prop]
            return self._clamp_value(spec, spec.min_value)

    @staticmethod
    def _clamp_value(spec: ScrubFieldSpec, value: float) -> float:
        clamped = max(spec.min_value, min(spec.max_value, float(value)))
        if spec.data_type is int:
            return int(round(clamped))
        return clamped

    @staticmethod
    def _values_equal(spec: ScrubFieldSpec, left: float, right: float) -> bool:
        if spec.data_type is int:
            return int(round(left)) == int(round(right))
        return abs(float(left) - float(right)) <= max(abs(spec.step) * 1.0e-4, 1.0e-6)

    @staticmethod
    def _fill_percent(spec: ScrubFieldSpec, value: float) -> float:
        span = spec.max_value - spec.min_value
        if abs(span) < 1.0e-9:
            return 100.0
        t = (float(value) - spec.min_value) / span
        t = max(0.0, min(1.0, t))
        return t * 100.0

    @staticmethod
    def _format_value(spec: ScrubFieldSpec, value: float) -> str:
        if spec.data_type is int:
            return spec.fmt % int(round(value))
        return spec.fmt % float(value)

    @staticmethod
    def _snap_value(spec: ScrubFieldSpec, value: float) -> float:
        if spec.step <= 0.0:
            return ScrubFieldController._clamp_value(spec, value)
        steps = round((float(value) - spec.min_value) / spec.step)
        snapped = spec.min_value + steps * spec.step
        return ScrubFieldController._clamp_value(spec, snapped)

    @staticmethod
    def _event_mouse_x(event) -> float:
        try:
            return float(event.get_parameter("mouse_x", "0"))
        except (TypeError, ValueError):
            return 0.0
