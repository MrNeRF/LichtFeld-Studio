# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Selection controls controller for the viewport selection overlay."""

import math

import lichtfeld as lf

from . import rml_widgets as w


_SELECTION_TOOL_ID = "builtin.select"
_DEPTH_MIN = 0.0
_DEPTH_MAX = 1000.0
_DEPTH_GAP = 0.01
_DEPTH_SLIDER_HALF_WINDOW = 20.0
_DEPTH_SLIDER_MIN_SPAN = 1.0
_DEFAULT_DEPTH_NEAR = 0.0
_DEFAULT_DEPTH_FAR = 5.3
_DEFAULT_FRUSTUM_HALF_WIDTH = 1.35

_MODE_LABELS = {
    "centers": "Brush",
    "rectangle": "Rectangle",
    "polygon": "Polygon",
    "lasso": "Lasso",
    "rings": "Rings",
    "color": "Color",
}


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


def _slider_bounds(center, lower, upper):
    lower = min(lower, upper)
    center = _clamp(center, lower, upper)
    available = upper - lower
    if available <= 0:
        return lower, upper

    min_span = min(_DEPTH_SLIDER_MIN_SPAN, available)
    low = max(lower, center - _DEPTH_SLIDER_HALF_WINDOW)
    high = min(upper, center + _DEPTH_SLIDER_HALF_WINDOW)
    if high - low < min_span:
        deficit = min_span - (high - low)
        low = max(lower, low - deficit * 0.5)
        high = min(upper, high + deficit * 0.5)
    if high - low < min_span:
        if low <= lower:
            high = min(upper, lower + min_span)
        else:
            low = max(lower, upper - min_span)
    return low, high


def _execute_stage(stage):
    result = stage.execute()
    result_get = getattr(result, "get", None)
    if result_get is None:
        return "Operation returned an invalid result."
    if bool(result_get("ok", False)):
        return None
    error = str(result_get("error", "") or "").strip()
    return error or "Operation failed."


class SelectionControlsController:
    _DIRTY_FIELDS = (
        "selection_tool_label",
        "selection_mode_label",
        "selection_depth_mode_active",
        "selection_has_scene",
        "selection_has_selection",
        "selection_can_delete",
        "selection_can_undo",
        "selection_can_redo",
        "selection_depth_near_str",
        "selection_depth_near_value",
        "selection_depth_near_slider_min",
        "selection_depth_near_slider_max",
        "selection_depth_far_str",
        "selection_depth_far_value",
        "selection_depth_far_slider_min",
        "selection_depth_far_slider_max",
        "selection_depth_toggle_label",
        "selection_delete_label",
        "selection_undo_label",
        "selection_redo_label",
        "selection_invert_label",
        "selection_select_all_label",
        "selection_unselect_label",
    )

    def __init__(self):
        self._handle = None
        self._visible = False
        self._active_tool = ""
        self._active_mode = ""
        self._has_scene = False
        self._has_selection = False
        self._can_undo = False
        self._can_redo = False
        self._depth_enabled = False
        self._depth_near = _DEFAULT_DEPTH_NEAR
        self._depth_far = _DEFAULT_DEPTH_FAR
        self._frustum_half_width = _DEFAULT_FRUSTUM_HALF_WIDTH
        self._last_state_key = None

    def bind_model(self, model):
        model.bind_func("selection_tool_label", lambda: "Select")
        model.bind_func("selection_mode_label", self._mode_label)
        model.bind_func("selection_depth_mode_active", lambda: self._depth_enabled)
        model.bind_func("selection_has_scene", lambda: self._has_scene)
        model.bind_func("selection_has_selection", lambda: self._has_selection)
        model.bind_func("selection_can_delete", lambda: self._has_selection)
        model.bind_func("selection_can_undo", lambda: self._can_undo)
        model.bind_func("selection_can_redo", lambda: self._can_redo)
        model.bind_func(
            "selection_depth_toggle_label",
            lambda: "Disable Depth Mode" if self._depth_enabled else "Enable Depth Mode",
        )
        model.bind_func("selection_delete_label", lambda: "Delete Selection")
        model.bind_func("selection_undo_label", lambda: "Undo")
        model.bind_func("selection_redo_label", lambda: "Redo")
        model.bind_func("selection_invert_label", lambda: "Invert Selection")
        model.bind_func("selection_select_all_label", lambda: "Select All")
        model.bind_func("selection_unselect_label", lambda: "Unselect")

        model.bind(
            "selection_depth_near_str",
            lambda: f"{self._depth_near:.2f}",
            self._set_depth_near,
        )
        model.bind(
            "selection_depth_near_value",
            lambda: f"{self._depth_near:.3f}",
            self._set_depth_near,
        )
        model.bind_func("selection_depth_near_slider_min", lambda: f"{self._near_slider_bounds()[0]:.3f}")
        model.bind_func("selection_depth_near_slider_max", lambda: f"{self._near_slider_bounds()[1]:.3f}")
        model.bind(
            "selection_depth_far_str",
            lambda: f"{self._depth_far:.2f}",
            self._set_depth_far,
        )
        model.bind(
            "selection_depth_far_value",
            lambda: f"{self._depth_far:.3f}",
            self._set_depth_far,
        )
        model.bind_func("selection_depth_far_slider_min", lambda: f"{self._far_slider_bounds()[0]:.3f}")
        model.bind_func("selection_depth_far_slider_max", lambda: f"{self._far_slider_bounds()[1]:.3f}")
        model.bind_event("selection_action", self._on_action)

        self._handle = model.get_handle()

    def mount(self, doc):
        self._visible = False
        self._last_state_key = None

        wrap = doc.get_element_by_id("selection-block")
        if wrap:
            wrap.set_class("hidden", True)

        for input_id in ("selection-depth-near", "selection-depth-far"):
            w.bind_select_all_on_focus(doc.get_element_by_id(input_id))

    def update(self, doc):
        dirty = False
        self._active_tool = self._get_active_tool()
        visible = self._active_tool == _SELECTION_TOOL_ID

        wrap = doc.get_element_by_id("selection-block")
        if visible != self._visible:
            self._visible = visible
            if wrap:
                wrap.set_class("hidden", not visible)
            dirty = True

        if not visible:
            if wrap:
                wrap.set_class("hidden", True)
            self._last_state_key = None
            return dirty

        self._refresh_state()
        state_key = self._state_key()
        if state_key != self._last_state_key:
            self._last_state_key = state_key
            self._dirty_all()
            dirty = True
        return dirty

    def unmount(self):
        self._handle = None
        self._visible = False
        self._last_state_key = None

    def _mode_label(self):
        return _MODE_LABELS.get(self._active_mode, "Selection")

    def _get_active_tool(self):
        getter = getattr(lf.ui, "get_active_tool", None)
        if not callable(getter):
            return ""
        try:
            return getter() or ""
        except Exception:
            return ""

    def _get_active_mode(self):
        getter = getattr(lf.ui, "get_active_submode", None)
        if not callable(getter):
            return ""
        try:
            return getter() or ""
        except Exception:
            return ""

    def _refresh_state(self):
        self._active_mode = self._get_active_mode()
        self._has_scene = self._scene_available()
        self._has_selection = self._scene_has_selection()
        self._can_undo = self._undo_available()
        self._can_redo = self._redo_available()
        self._refresh_depth_state()

    def _refresh_depth_state(self):
        try:
            enabled, near, far, width = lf.selection.get_depth_filter_range()
        except Exception:
            enabled = self._depth_enabled
            near = self._depth_near
            far = self._depth_far
            width = self._frustum_half_width

        self._depth_enabled = bool(enabled)
        self._depth_near = _clamp(_parse_float(near, _DEFAULT_DEPTH_NEAR), _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        self._depth_far = _clamp(
            _parse_float(far, _DEFAULT_DEPTH_FAR),
            self._depth_near + _DEPTH_GAP,
            _DEPTH_MAX,
        )
        self._frustum_half_width = max(_parse_float(width, _DEFAULT_FRUSTUM_HALF_WIDTH), 0.05)

    def _state_key(self):
        return (
            self._active_tool,
            self._active_mode,
            self._has_scene,
            self._has_selection,
            self._can_undo,
            self._can_redo,
            self._depth_enabled,
            round(self._depth_near, 3),
            round(self._depth_far, 3),
            round(self._frustum_half_width, 3),
        )

    def _near_slider_bounds(self):
        return _slider_bounds(self._depth_near, _DEPTH_MIN, self._depth_far - _DEPTH_GAP)

    def _far_slider_bounds(self):
        return _slider_bounds(self._depth_far, self._depth_near + _DEPTH_GAP, _DEPTH_MAX)

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

    def _scene_has_selection(self):
        scene_getter = getattr(lf, "get_scene", None)
        if not callable(scene_getter):
            return False
        try:
            scene = scene_getter()
        except Exception:
            return False
        if scene is None:
            return False
        has_selection = getattr(scene, "has_selection", None)
        if callable(has_selection):
            try:
                return bool(has_selection())
            except Exception:
                return False
        return getattr(scene, "selection_mask", None) is not None

    def _undo_available(self):
        try:
            return bool(lf.undo.can_undo())
        except Exception:
            return False

    def _redo_available(self):
        try:
            return bool(lf.undo.can_redo())
        except Exception:
            return False

    def _set_depth_near(self, value):
        self._refresh_depth_state()
        near = _clamp(_parse_float(value, self._depth_near), _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        far = max(self._depth_far, near + _DEPTH_GAP)
        self._apply_depth_range(self._depth_enabled, near, far)

    def _set_depth_far(self, value):
        self._refresh_depth_state()
        far = _clamp(_parse_float(value, self._depth_far), self._depth_near + _DEPTH_GAP, _DEPTH_MAX)
        self._apply_depth_range(self._depth_enabled, self._depth_near, far)

    def _apply_depth_range(self, enabled, near, far):
        self._depth_enabled = bool(enabled)
        self._depth_near = _clamp(near, _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        self._depth_far = _clamp(far, self._depth_near + _DEPTH_GAP, _DEPTH_MAX)

        try:
            lf.selection.set_depth_filter_range(
                self._depth_enabled,
                self._depth_near,
                self._depth_far,
                self._frustum_half_width,
            )
        except Exception as exc:
            self._report_error(str(exc).strip() or "Could not update selection depth filter.")

        self._dirty_all()

    def _on_action(self, handle, event, args):
        del handle, event
        if not args:
            return

        action = str(args[0])
        if action == "toggle_depth":
            self._refresh_depth_state()
            self._apply_depth_range(not self._depth_enabled, self._depth_near, self._depth_far)
        elif action == "delete":
            self._execute_selection_stage(lambda: lf.pipeline.edit.delete_())
        elif action == "select_all":
            self._execute_selection_stage(lambda: lf.pipeline.select.all())
        elif action == "unselect":
            self._execute_selection_stage(lambda: lf.pipeline.select.none())
        elif action == "undo":
            try:
                if lf.undo.can_undo():
                    lf.undo.undo()
            except Exception as exc:
                self._report_error(str(exc).strip() or "Undo failed.")
        elif action == "redo":
            try:
                if lf.undo.can_redo():
                    lf.undo.redo()
            except Exception as exc:
                self._report_error(str(exc).strip() or "Redo failed.")
        elif action == "invert":
            self._execute_selection_stage(lambda: lf.pipeline.select.invert())

        self._refresh_state()
        self._dirty_all()

    def _execute_selection_stage(self, factory):
        try:
            error = _execute_stage(factory())
        except Exception as exc:
            error = str(exc).strip() or "Operation failed."
        if error:
            self._report_error(error)

    def _report_error(self, message):
        dialog = getattr(lf.ui, "message_dialog", None)
        if callable(dialog):
            try:
                dialog("Selection Operation Failed", message, style="error")
            except Exception:
                pass

    def _dirty_all(self):
        if not self._handle:
            return
        for field in self._DIRTY_FIELDS:
            self._handle.dirty(field)
