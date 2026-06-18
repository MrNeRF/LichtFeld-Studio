# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth-map controls controller for the viewport overlay."""

import math

import lichtfeld as lf

from .scrub_fields import ScrubFieldController, ScrubFieldSpec


_DEPTH_MIN = 0.0
_DEPTH_MAX = 10000.0
_DEPTH_GAP = 0.01
_DEFAULT_DEPTH_NEAR = 0.1
_DEFAULT_DEPTH_FAR = 100.0
_DEFAULT_MODE = "palette"

_DEPTH_SCRUB_SPECS = {
    "depth_view_near_value": ScrubFieldSpec(_DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP, 0.05, "%.2f"),
    "depth_view_far_value": ScrubFieldSpec(_DEPTH_MIN + _DEPTH_GAP, _DEPTH_MAX, 0.05, "%.2f"),
}


def _ui_label(key: str, fallback: str) -> str:
    tr = getattr(lf.ui, "tr", None)
    if not callable(tr):
        return fallback
    try:
        value = tr(key)
    except Exception:
        return fallback
    if value and value != key:
        return value
    return fallback


def _parse_float(value, fallback):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return parsed


def _clamp(value, lower, upper):
    return min(max(value, lower), upper)


def _normalize_mode(value):
    value = str(value or "").strip().lower()
    if value in {"gray", "grayscale"}:
        return "gray"
    return _DEFAULT_MODE


class DepthViewControlsController:
    _DIRTY_FIELDS = (
        "depth_view_tool_label",
        "depth_view_mode_value",
        "depth_view_has_scene",
        "depth_view_near_value",
        "depth_view_near_slider_min",
        "depth_view_near_slider_max",
        "depth_view_far_value",
        "depth_view_far_slider_min",
        "depth_view_far_slider_max",
        "depth_view_disable_label",
    )

    def __init__(self):
        self._handle = None
        self._visible = False
        self._has_scene = False
        self._depth_near = _DEFAULT_DEPTH_NEAR
        self._depth_far = _DEFAULT_DEPTH_FAR
        self._mode = _DEFAULT_MODE
        self._last_state_key = None
        self._last_state_items = None
        self._scrub_fields = ScrubFieldController(
            _DEPTH_SCRUB_SPECS,
            self._get_scrub_value,
            self._set_scrub_value,
        )

    @property
    def visible(self):
        return self._visible

    def bind_model(self, model):
        model.bind_func("depth_view_tool_label", lambda: _ui_label("toolbar.depth_map", "Depth Map"))
        model.bind_func("depth_view_has_scene", lambda: self._has_scene)
        model.bind_func("depth_view_disable_label", lambda: "Disable Depth Map")
        model.bind(
            "depth_view_mode_value",
            lambda: self._mode,
            self._set_mode,
        )
        model.bind(
            "depth_view_near_value",
            lambda: f"{self._depth_near:.3f}",
            self._set_depth_near,
        )
        model.bind_func("depth_view_near_slider_min", lambda: f"{self._near_slider_bounds()[0]:.3f}")
        model.bind_func("depth_view_near_slider_max", lambda: f"{self._near_slider_bounds()[1]:.3f}")
        model.bind(
            "depth_view_far_value",
            lambda: f"{self._depth_far:.3f}",
            self._set_depth_far,
        )
        model.bind_func("depth_view_far_slider_min", lambda: f"{self._far_slider_bounds()[0]:.3f}")
        model.bind_func("depth_view_far_slider_max", lambda: f"{self._far_slider_bounds()[1]:.3f}")
        model.bind_event("depth_view_action", self._on_action)

        self._handle = model.get_handle()

    def mount(self, doc):
        self._visible = False
        self._last_state_key = None
        self._last_state_items = None

        wrap = doc.get_element_by_id("depth-view-block")
        if wrap:
            wrap.set_class("hidden", True)

        self._scrub_fields.mount(doc)

    def update(self, doc):
        dirty = False
        dirty_reasons = []
        visible = self._depth_view_active()
        wrap = doc.get_element_by_id("depth-view-block")
        if wrap:
            wrap.set_class("hidden", not visible)

        if visible != self._visible:
            self._visible = visible
            dirty = True
            dirty_reasons.append("visibility")

        if not visible:
            self._last_state_key = None
            self._last_state_items = None
            return ",".join(dirty_reasons) if dirty else None

        self._refresh_state()
        self._scrub_fields.sync_all()
        state_items = self._state_items()
        state_key = self._state_key(state_items)
        if state_key != self._last_state_key:
            changed_fields = self._changed_state_fields(state_items)
            self._last_state_key = state_key
            self._last_state_items = state_items
            self._dirty_changed_fields(changed_fields)
            dirty = True
            dirty_reasons.append(f"state:{'+'.join(changed_fields)}")
        return ",".join(dirty_reasons) if dirty else None

    def unmount(self):
        self._handle = None
        self._visible = False
        self._last_state_key = None
        self._last_state_items = None
        self._scrub_fields.unmount()

    def _depth_view_active(self):
        getter = getattr(lf, "get_depth_view", None)
        if not callable(getter):
            return False
        try:
            return bool(getter())
        except Exception:
            return False

    def _refresh_state(self):
        self._has_scene = self._scene_available()
        try:
            near, far = lf.get_depth_view_range()
        except Exception:
            near = self._depth_near
            far = self._depth_far
        try:
            mode = lf.get_depth_view_mode()
        except Exception:
            mode = self._mode

        self._depth_near = _clamp(_parse_float(near, _DEFAULT_DEPTH_NEAR), _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        self._depth_far = _clamp(
            _parse_float(far, _DEFAULT_DEPTH_FAR),
            self._depth_near + _DEPTH_GAP,
            _DEPTH_MAX,
        )
        self._mode = _normalize_mode(mode)

    def _state_items(self):
        return (
            ("has_scene", self._has_scene),
            ("depth_near", round(self._depth_near, 3)),
            ("depth_far", round(self._depth_far, 3)),
            ("mode", self._mode),
        )

    def _state_key(self, state_items=None):
        if state_items is None:
            state_items = self._state_items()
        return tuple(value for _name, value in state_items)

    def _changed_state_fields(self, state_items):
        if self._last_state_items is None:
            return ["initial"]
        previous = dict(self._last_state_items)
        return [name for name, value in state_items if previous.get(name) != value]

    def _near_slider_bounds(self):
        return _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP

    def _far_slider_bounds(self):
        return _DEPTH_MIN + _DEPTH_GAP, _DEPTH_MAX

    def _scene_available(self):
        getter = getattr(lf, "has_scene", None)
        if callable(getter):
            try:
                return bool(getter())
            except Exception:
                pass
        scene_getter = getattr(lf, "get_scene", None)
        if callable(scene_getter):
            try:
                return scene_getter() is not None
            except Exception:
                return False
        return False

    def _set_mode(self, value):
        mode = _normalize_mode(value)
        self._mode = mode
        try:
            lf.set_depth_view_mode(mode)
        except Exception as exc:
            self._report_error(str(exc).strip() or "Could not update depth-map mode.")
        self._dirty_all()

    def _get_scrub_value(self, prop):
        if prop == "depth_view_far_value":
            return self._depth_far
        return self._depth_near

    def _set_scrub_value(self, prop, value):
        if prop == "depth_view_far_value":
            self._set_depth_far(value)
        else:
            self._set_depth_near(value)

    def _set_depth_near(self, value):
        self._refresh_state()
        near = _clamp(_parse_float(value, self._depth_near), _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        far = max(self._depth_far, near + _DEPTH_GAP)
        self._apply_depth_range(near, far)

    def _set_depth_far(self, value):
        self._refresh_state()
        far = _clamp(_parse_float(value, self._depth_far), _DEPTH_MIN + _DEPTH_GAP, _DEPTH_MAX)
        near = min(self._depth_near, far - _DEPTH_GAP)
        self._apply_depth_range(near, far)

    def _apply_depth_range(self, near, far):
        self._depth_near = _clamp(near, _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        self._depth_far = _clamp(far, self._depth_near + _DEPTH_GAP, _DEPTH_MAX)
        try:
            lf.set_depth_view_range(self._depth_near, self._depth_far)
        except Exception as exc:
            self._report_error(str(exc).strip() or "Could not update depth-map range.")
        self._scrub_fields.sync_all()
        self._dirty_all()

    def _on_action(self, handle, event, args):
        del handle, event, args
        try:
            lf.set_depth_view(False)
        except Exception as exc:
            self._report_error(str(exc).strip() or "Could not disable depth map.")
        self._dirty_all()

    def _report_error(self, message):
        dialog = getattr(lf.ui, "message_dialog", None)
        if callable(dialog):
            try:
                dialog(_ui_label("depth_view.operation_failed", "Depth Map Failed"), message, style="error")
            except Exception:
                pass

    def _dirty_all(self):
        if not self._handle:
            return
        for field in self._DIRTY_FIELDS:
            self._handle.dirty(field)

    def _dirty_changed_fields(self, changed_fields):
        if not self._handle:
            return

        field_map = {
            "has_scene": ("depth_view_has_scene",),
            "depth_near": (
                "depth_view_near_value",
                "depth_view_near_slider_min",
                "depth_view_near_slider_max",
                "depth_view_far_slider_min",
            ),
            "depth_far": (
                "depth_view_far_value",
                "depth_view_far_slider_min",
                "depth_view_far_slider_max",
                "depth_view_near_slider_max",
            ),
            "mode": ("depth_view_mode_value",),
        }

        if "initial" in changed_fields:
            self._dirty_all()
            return

        dirty_fields = []
        for changed in changed_fields:
            dirty_fields.extend(field_map.get(changed, ()))

        for field in dict.fromkeys(dirty_fields):
            self._handle.dirty(field)
