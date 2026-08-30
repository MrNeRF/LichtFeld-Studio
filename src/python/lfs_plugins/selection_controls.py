# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Selection controls controller for the viewport selection overlay."""

import math

import lichtfeld as lf

from . import rml_widgets as w
from .ui import RuntimeState

try:
    from .ui import native_value as _native_store_value
except Exception:
    def _native_store_value(_field, fallback):
        return fallback


_SELECTION_TOOL_ID = "builtin.select"
_DEPTH_MIN = 0.0
_DEPTH_MAX = 1000.0
_DEPTH_GAP = 0.01
_DEPTH_SLIDER_HALF_WINDOW = 20.0
_DEPTH_SLIDER_MIN_SPAN = 1.0
_DEFAULT_DEPTH_NEAR = 0.0
_DEFAULT_DEPTH_FAR = 6.0
_DEFAULT_FRUSTUM_HALF_WIDTH = 1.35
_DEFAULT_WINDOW_SCALE = 0.35
_DEFAULT_WINDOW_OFFSET = 0.0
_DEFAULT_VIZ_MODE = 1
_SCALE_PERCENT_MIN = 5.0
_SCALE_PERCENT_MAX = 100.0
_OFFSET_PERCENT_MIN = -100.0
_OFFSET_PERCENT_MAX = 100.0
_MISSING = object()
_VIZ_MODE_ICONS = {
    0: "../icon/scene/visible.png",
    1: "../icon/contrast.png",
    2: "../icon/scene/hidden.png",
}
_VIZ_MODE_LABELS = {
    0: ("main_panel.depth_filter_viz_mode_off", "Off"),
    1: ("main_panel.depth_filter_viz_mode_dim", "Dim outside"),
    2: ("main_panel.depth_filter_viz_mode_hide", "Hide outside"),
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
        return _ui_label("selection.operation_failed_generic", "Operation failed.")
    if bool(result_get("ok", False)):
        return None
    error = str(result_get("error", "") or "").strip()
    return error or "Operation failed."


class SelectionControlsController:
    _DIRTY_FIELDS = (
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
        "selection_depth_scale_str",
        "selection_depth_scale_value",
        "selection_depth_scale_slider_min",
        "selection_depth_scale_slider_max",
        "selection_depth_offset_x_str",
        "selection_depth_offset_x_value",
        "selection_depth_offset_x_slider_min",
        "selection_depth_offset_x_slider_max",
        "selection_depth_offset_y_str",
        "selection_depth_offset_y_value",
        "selection_depth_offset_y_slider_min",
        "selection_depth_offset_y_slider_max",
        "selection_viz_mode_label",
        "selection_viz_mode_icon",
        "selection_depth_toggle_label",
        "ui_size_label",
        "ui_offset_x_label",
        "ui_offset_y_label",
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
        self._window_scale = _DEFAULT_WINDOW_SCALE
        self._offset_x = _DEFAULT_WINDOW_OFFSET
        self._offset_y = _DEFAULT_WINDOW_OFFSET
        self._viz_mode = _DEFAULT_VIZ_MODE
        self._last_state_key = None
        self._last_state_items = None
        self._depth_echo_holdoff = 0
        self._depth_text_bufs = {
            "selection_depth_near_str": None,
            "selection_depth_far_str": None,
            "selection_depth_scale_str": None,
            "selection_depth_offset_x_str": None,
            "selection_depth_offset_y_str": None,
        }
        self._editing_depth_text = set()
        self._escape_revert = w.EscapeRevertController()

    def bind_model(self, model):
        model.bind_func("selection_depth_mode_active", lambda: self._depth_enabled)
        model.bind_func("selection_has_scene", lambda: self._has_scene)
        model.bind_func("selection_has_selection", lambda: self._has_selection)
        model.bind_func("selection_can_delete", lambda: self._has_selection)
        model.bind_func("selection_can_undo", lambda: self._can_undo)
        model.bind_func("selection_can_redo", lambda: self._can_redo)
        model.bind_func(
            "selection_depth_toggle_label",
            lambda: _ui_label("toolbar.depth_mode_disable", "Disable Depth Mode")
            if self._depth_enabled
            else _ui_label("toolbar.depth_mode_enable", "Enable Depth Mode"),
        )
        model.bind_func("selection_delete_label", lambda: _ui_label("toolbar.delete_selection", "Delete Selection"))
        model.bind_func("selection_undo_label", lambda: _ui_label("toolbar.undo", "Undo"))
        model.bind_func("selection_redo_label", lambda: _ui_label("toolbar.redo", "Redo"))
        model.bind_func("selection_invert_label", lambda: _ui_label("toolbar.invert_selection", "Invert Selection"))
        model.bind_func("selection_select_all_label", lambda: _ui_label("toolbar.select_all", "Select All"))
        model.bind_func("selection_unselect_label", lambda: _ui_label("toolbar.unselect", "Unselect"))

        model.bind(
            "selection_depth_near_str",
            lambda: self._depth_text_value("selection_depth_near_str"),
            lambda value: self._set_depth_text_value("selection_depth_near_str", value),
        )
        model.bind(
            "selection_depth_near_value",
            lambda: f"{self._depth_near:.3f}",
            self._set_depth_near_from_slider,
        )
        model.bind_func("selection_depth_near_slider_min", lambda: f"{self._near_slider_bounds()[0]:.3f}")
        model.bind_func("selection_depth_near_slider_max", lambda: f"{self._near_slider_bounds()[1]:.3f}")
        model.bind(
            "selection_depth_far_str",
            lambda: self._depth_text_value("selection_depth_far_str"),
            lambda value: self._set_depth_text_value("selection_depth_far_str", value),
        )
        model.bind(
            "selection_depth_far_value",
            lambda: f"{self._depth_far:.3f}",
            self._set_depth_far_from_slider,
        )
        model.bind_func("selection_depth_far_slider_min", lambda: f"{self._far_slider_bounds()[0]:.3f}")
        model.bind_func("selection_depth_far_slider_max", lambda: f"{self._far_slider_bounds()[1]:.3f}")
        model.bind(
            "selection_depth_scale_str",
            lambda: self._depth_text_value("selection_depth_scale_str"),
            lambda value: self._set_depth_text_value("selection_depth_scale_str", value),
        )
        model.bind(
            "selection_depth_scale_value",
            lambda: f"{self._scale_percent():.0f}",
            self._set_depth_scale_percent_from_slider,
        )
        model.bind_func("selection_depth_scale_slider_min", lambda: f"{_SCALE_PERCENT_MIN:.0f}")
        model.bind_func("selection_depth_scale_slider_max", lambda: f"{_SCALE_PERCENT_MAX:.0f}")
        model.bind(
            "selection_depth_offset_x_str",
            lambda: self._depth_text_value("selection_depth_offset_x_str"),
            lambda value: self._set_depth_text_value("selection_depth_offset_x_str", value),
        )
        model.bind(
            "selection_depth_offset_x_value",
            lambda: f"{self._offset_percent(self._offset_x):.0f}",
            self._set_depth_offset_x_percent_from_slider,
        )
        model.bind_func("selection_depth_offset_x_slider_min", lambda: f"{_OFFSET_PERCENT_MIN:.0f}")
        model.bind_func("selection_depth_offset_x_slider_max", lambda: f"{_OFFSET_PERCENT_MAX:.0f}")
        model.bind(
            "selection_depth_offset_y_str",
            lambda: self._depth_text_value("selection_depth_offset_y_str"),
            lambda value: self._set_depth_text_value("selection_depth_offset_y_str", value),
        )
        model.bind(
            "selection_depth_offset_y_value",
            lambda: f"{self._offset_percent(self._offset_y):.0f}",
            self._set_depth_offset_y_percent_from_slider,
        )
        model.bind_func("selection_depth_offset_y_slider_min", lambda: f"{_OFFSET_PERCENT_MIN:.0f}")
        model.bind_func("selection_depth_offset_y_slider_max", lambda: f"{_OFFSET_PERCENT_MAX:.0f}")
        model.bind_func("selection_viz_mode_label", self._viz_mode_label)
        model.bind_func("selection_viz_mode_icon", self._viz_mode_icon)
        model.bind_func("ui_size_label", lambda: _ui_label("ui.selection_depth_size", "Size"))
        model.bind_func("ui_offset_x_label", lambda: _ui_label("ui.selection_depth_offset_x", "X"))
        model.bind_func("ui_offset_y_label", lambda: _ui_label("ui.selection_depth_offset_y", "Y"))
        model.bind_event("selection_action", self._on_action)

        self._handle = model.get_handle()

    def mount(self, doc):
        self._visible = False
        self._last_state_key = None
        self._last_state_items = None

        wrap = doc.get_element_by_id("selection-block")
        if wrap:
            wrap.set_class("hidden", True)

        self._mount_depth_text_input(doc, "selection-depth-near", "selection_depth_near_str")
        self._mount_depth_text_input(doc, "selection-depth-far", "selection_depth_far_str")
        self._mount_depth_text_input(doc, "selection-depth-scale", "selection_depth_scale_str")
        self._mount_depth_text_input(doc, "selection-depth-offset-x", "selection_depth_offset_x_str")
        self._mount_depth_text_input(doc, "selection-depth-offset-y", "selection_depth_offset_y_str")

    def update(self, doc):
        dirty = False
        dirty_reasons = []
        self._active_tool = self._get_active_tool()
        visible = self._active_tool == _SELECTION_TOOL_ID
        wrap = doc.get_element_by_id("selection-block")
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

        previous_depth = self._depth_window_state()
        self._refresh_state()
        # RmlUi range inputs echo stale values when attributes update in the same frame.
        # The window size and the two offsets are driven by the same kind of range
        # input as near/far, so a change to any of the five has to arm the holdoff;
        # arming on near/far alone leaves the window sliders unprotected.
        if self._depth_window_state() != previous_depth:
            self._depth_echo_holdoff = 2
        else:
            self._depth_echo_holdoff = max(0, self._depth_echo_holdoff - 1)
        self._sync_depth_text_bufs()
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
        self._depth_echo_holdoff = 0
        self._editing_depth_text.clear()
        self._escape_revert.clear()

    def _get_active_tool(self):
        value = _native_store_value("active_tool", _MISSING)
        if value is not _MISSING:
            return value or ""
        getter = getattr(lf.ui, "get_active_tool", None)
        if not callable(getter):
            return ""
        try:
            return getter() or ""
        except Exception:
            return ""

    def _get_active_mode(self):
        value = _native_store_value("active_submode", _MISSING)
        if value is not _MISSING:
            return value or ""
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

        window_getter = getattr(lf.selection, "get_depth_filter_window", None)
        if callable(window_getter):
            try:
                w_enabled, w_near, w_far, scale, offset_x, offset_y = window_getter()
                enabled = w_enabled
                near = w_near
                far = w_far
                self._window_scale = _clamp(
                    _parse_float(scale, _DEFAULT_WINDOW_SCALE), 0.05, 1.0
                )
                self._offset_x = _clamp(
                    _parse_float(offset_x, _DEFAULT_WINDOW_OFFSET), -1.0, 1.0
                )
                self._offset_y = _clamp(
                    _parse_float(offset_y, _DEFAULT_WINDOW_OFFSET), -1.0, 1.0
                )
            except Exception:
                pass

        self._depth_enabled = bool(enabled)
        self._depth_near = _clamp(_parse_float(near, _DEFAULT_DEPTH_NEAR), _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        self._depth_far = _clamp(
            _parse_float(far, _DEFAULT_DEPTH_FAR),
            self._depth_near + _DEPTH_GAP,
            _DEPTH_MAX,
        )
        self._frustum_half_width = max(_parse_float(width, _DEFAULT_FRUSTUM_HALF_WIDTH), 0.05)
        self._refresh_viz_mode()

    def _depth_window_state(self):
        """Every field the depth sliders write, for echo-holdoff comparison."""
        return (
            self._depth_near,
            self._depth_far,
            self._window_scale,
            self._offset_x,
            self._offset_y,
        )

    def _state_items(self):
        return (
            ("language_generation", RuntimeState.language_generation.value),
            ("active_tool", self._active_tool),
            ("active_mode", self._active_mode),
            ("has_scene", self._has_scene),
            ("has_selection", self._has_selection),
            ("can_undo", self._can_undo),
            ("can_redo", self._can_redo),
            ("depth_enabled", self._depth_enabled),
            ("depth_near", round(self._depth_near, 3)),
            ("depth_far", round(self._depth_far, 3)),
            ("window_scale", round(self._window_scale, 4)),
            ("offset_x", round(self._offset_x, 4)),
            ("offset_y", round(self._offset_y, 4)),
            ("viz_mode", int(self._viz_mode)),
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

    # The five _set_depth_* methods below are the text-commit path: they apply a
    # deliberate user edit and do not consult the echo holdoff. The range inputs
    # bind the _from_slider wrappers instead, which carry that check, so each
    # origin keeps its own entry point and neither call site has to be told
    # which it is.
    # Splitting the origins is not on its own enough: applying a commit dirties
    # every slider-bound value, and RmlUi answers that by replaying the
    # pre-commit slider position. _commit_depth_text_key therefore arms the
    # holdoff before it dispatches, so the wrappers reject that echo.
    # `not self._visible` and `_last_state_key is None` still gate both paths.

    def _set_depth_near(self, value):
        if not self._visible or self._last_state_key is None:
            return
        self._refresh_depth_state()
        near = _clamp(_parse_float(value, self._depth_near), _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        far = max(self._depth_far, near + _DEPTH_GAP)
        self._apply_depth_range(self._depth_enabled, near, far)

    def _set_depth_far(self, value):
        if not self._visible or self._last_state_key is None:
            return
        self._refresh_depth_state()
        far = _clamp(_parse_float(value, self._depth_far), self._depth_near + _DEPTH_GAP, _DEPTH_MAX)
        self._apply_depth_range(self._depth_enabled, self._depth_near, far)

    def _scale_percent(self):
        return _clamp(round(self._window_scale * 100.0), _SCALE_PERCENT_MIN, _SCALE_PERCENT_MAX)

    def _offset_percent(self, offset):
        return _clamp(round(offset * 100.0), _OFFSET_PERCENT_MIN, _OFFSET_PERCENT_MAX)

    def _refresh_viz_mode(self):
        settings_getter = getattr(lf, "get_render_settings", None)
        if not callable(settings_getter):
            return
        try:
            settings = settings_getter()
        except Exception:
            return
        if settings is None:
            return
        try:
            self._viz_mode = int(_clamp(int(getattr(settings, "depth_filter_viz_mode", self._viz_mode)), 0, 2))
        except Exception:
            pass

    def _viz_mode_label(self):
        key, fallback = _VIZ_MODE_LABELS.get(int(self._viz_mode), _VIZ_MODE_LABELS[_DEFAULT_VIZ_MODE])
        return _ui_label(key, fallback)

    def _viz_mode_icon(self):
        return _VIZ_MODE_ICONS.get(int(self._viz_mode), _VIZ_MODE_ICONS[_DEFAULT_VIZ_MODE])

    def _set_depth_scale_percent(self, value):
        if not self._visible or self._last_state_key is None:
            return
        self._refresh_depth_state()
        if isinstance(value, str):
            value = value.strip().rstrip("%")
        percent = _clamp(_parse_float(value, self._scale_percent()), _SCALE_PERCENT_MIN, _SCALE_PERCENT_MAX)
        self._apply_depth_window(self._depth_enabled, self._depth_near, self._depth_far, percent / 100.0, self._offset_x, self._offset_y)

    def _set_depth_offset_x_percent(self, value):
        if not self._visible or self._last_state_key is None:
            return
        self._refresh_depth_state()
        percent = _clamp(_parse_float(value, self._offset_percent(self._offset_x)), _OFFSET_PERCENT_MIN, _OFFSET_PERCENT_MAX)
        self._apply_depth_window(self._depth_enabled, self._depth_near, self._depth_far, self._window_scale, percent / 100.0, self._offset_y)

    def _set_depth_offset_y_percent(self, value):
        if not self._visible or self._last_state_key is None:
            return
        self._refresh_depth_state()
        percent = _clamp(_parse_float(value, self._offset_percent(self._offset_y)), _OFFSET_PERCENT_MIN, _OFFSET_PERCENT_MAX)
        self._apply_depth_window(self._depth_enabled, self._depth_near, self._depth_far, self._window_scale, self._offset_x, percent / 100.0)

    # Range-input entry points. RmlUi replays a slider's pre-update position into
    # its setter when the bound attributes change in the same frame, so these
    # drop the value while the echo holdoff is armed. They are bound in
    # bind_model; the text-commit path calls the cores above directly.

    def _set_depth_near_from_slider(self, value):
        if self._depth_echo_holdoff > 0:
            return
        self._set_depth_near(value)

    def _set_depth_far_from_slider(self, value):
        if self._depth_echo_holdoff > 0:
            return
        self._set_depth_far(value)

    def _set_depth_scale_percent_from_slider(self, value):
        if self._depth_echo_holdoff > 0:
            return
        self._set_depth_scale_percent(value)

    def _set_depth_offset_x_percent_from_slider(self, value):
        if self._depth_echo_holdoff > 0:
            return
        self._set_depth_offset_x_percent(value)

    def _set_depth_offset_y_percent_from_slider(self, value):
        if self._depth_echo_holdoff > 0:
            return
        self._set_depth_offset_y_percent(value)

    def _apply_depth_range(self, enabled, near, far):
        self._apply_depth_window(enabled, near, far, self._window_scale, self._offset_x, self._offset_y)

    def _apply_depth_window(self, enabled, near, far, scale, offset_x, offset_y):
        self._depth_enabled = bool(enabled)
        self._depth_near = _clamp(near, _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        self._depth_far = _clamp(far, self._depth_near + _DEPTH_GAP, _DEPTH_MAX)
        self._window_scale = _clamp(scale, 0.05, 1.0)
        self._offset_x = _clamp(offset_x, -1.0, 1.0)
        self._offset_y = _clamp(offset_y, -1.0, 1.0)

        window_setter = getattr(lf.selection, "set_depth_filter_window", None)
        try:
            if callable(window_setter):
                window_setter(
                    self._depth_enabled,
                    self._depth_near,
                    self._depth_far,
                    self._window_scale,
                    self._offset_x,
                    self._offset_y,
                )
            else:
                lf.selection.set_depth_filter_range(
                    self._depth_enabled,
                    self._depth_near,
                    self._depth_far,
                    self._frustum_half_width,
                )
        except Exception as exc:
            self._report_error(
                str(exc).strip()
                or _ui_label("selection.update_depth_failed", "Could not update selection depth filter.")
            )

        self._sync_depth_text_bufs()
        self._dirty_all()

    def _set_viz_mode(self, mode):
        self._viz_mode = int(_clamp(int(mode), 0, 2))
        settings_getter = getattr(lf, "get_render_settings", None)
        if callable(settings_getter):
            try:
                settings = settings_getter()
                if settings is not None:
                    settings.depth_filter_viz_mode = self._viz_mode
            except Exception as exc:
                self._report_error(
                    str(exc).strip()
                    or _ui_label("selection.update_depth_failed", "Could not update selection depth filter.")
                )
        self._dirty_all()

    def _cycle_viz_mode(self):
        self._refresh_viz_mode()
        self._set_viz_mode((int(self._viz_mode) + 1) % 3)

    def _mount_depth_text_input(self, doc, input_id, key):
        w.bind_committed_text_input(
            doc.get_element_by_id(input_id),
            key,
            escape_revert=self._escape_revert,
            capture=lambda k=key: self._capture_depth_text_snapshot(k),
            restore=lambda snapshot, k=key: self._restore_depth_text_snapshot(k, snapshot),
            commit=self._commit_depth_text_key,
            on_focus=self._begin_depth_text_edit,
            on_blur=self._end_depth_text_edit,
        )

    def _depth_text_value(self, key):
        value = self._depth_text_bufs.get(key)
        if value is not None:
            return value
        return self._canonical_depth_text_value(key)

    def _set_depth_text_value(self, key, value):
        self._depth_text_bufs[key] = str(value)

    def _begin_depth_text_edit(self, key):
        self._editing_depth_text.add(key)

    def _end_depth_text_edit(self, key):
        self._editing_depth_text.discard(key)

    def _capture_depth_text_snapshot(self, key):
        return self._canonical_depth_text_value(key)

    def _restore_depth_text_snapshot(self, key, snapshot):
        # Escape restores the pre-edit text and the host blurs immediately after
        # (cancelFocusedElement in rml_input_utils.hpp), so that blur is what
        # carries the revert to the native side.
        self._depth_text_bufs[key] = str(snapshot or "")
        if self._handle:
            self._handle.dirty(key)

    def _commit_depth_text_key(self, key):
        value = self._depth_text_bufs.get(key)
        if value is not None and value.strip():
            parsed_src = value.strip().rstrip("%") if key == "selection_depth_scale_str" else value
            parsed = _parse_float(parsed_src, None)
            if parsed is None:
                self._sync_depth_text_bufs(force=True)
                return
            # The cores below refuse while the panel is hidden or before the
            # first update lands. Refuse here too, ahead of the arming: arming
            # for a write that is then refused would leave the holdoff set, and a
            # hidden update returns before the decrement, so it would still be
            # armed when the panel comes back.
            if not self._visible or self._last_state_key is None:
                self._sync_depth_text_bufs(force=True)
                return
            # Applying dirties every slider-bound value, and RmlUi answers that by
            # replaying each slider's pre-commit position into its setter. Arm the
            # holdoff first so the _from_slider wrappers reject that echo whether
            # it arrives during this call or on the next model update. The cores
            # below are the text path and do not consult the holdoff, so this
            # cannot block the commit it is protecting.
            self._depth_echo_holdoff = 2
            if key == "selection_depth_near_str":
                self._set_depth_near(parsed)
            elif key == "selection_depth_far_str":
                self._set_depth_far(parsed)
            elif key == "selection_depth_scale_str":
                self._set_depth_scale_percent(parsed)
            elif key == "selection_depth_offset_x_str":
                self._set_depth_offset_x_percent(parsed)
            elif key == "selection_depth_offset_y_str":
                self._set_depth_offset_y_percent(parsed)

        self._sync_depth_text_bufs(force=True)

    def _sync_depth_text_bufs(self, force=False):
        for key in self._depth_text_bufs:
            if not force and key in self._editing_depth_text:
                continue
            canonical = self._canonical_depth_text_value(key)
            if self._depth_text_bufs.get(key) == canonical:
                continue
            self._depth_text_bufs[key] = canonical
            if self._handle:
                self._handle.dirty(key)

    def _canonical_depth_text_value(self, key):
        if key == "selection_depth_near_str":
            return f"{self._depth_near:.2f}"
        if key == "selection_depth_far_str":
            return f"{self._depth_far:.2f}"
        if key == "selection_depth_scale_str":
            return f"{self._scale_percent():.0f}%"
        if key == "selection_depth_offset_x_str":
            return f"{self._offset_percent(self._offset_x):.0f}"
        if key == "selection_depth_offset_y_str":
            return f"{self._offset_percent(self._offset_y):.0f}"
        return ""

    def _on_action(self, handle, event, args):
        del handle, event
        if not args:
            return

        action = str(args[0])
        if action == "toggle_depth":
            self._refresh_depth_state()
            self._apply_depth_range(not self._depth_enabled, self._depth_near, self._depth_far)
        elif action == "cycle_viz":
            self._cycle_viz_mode()
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
                self._report_error(str(exc).strip() or _ui_label("selection.undo_failed", "Undo failed."))
        elif action == "redo":
            try:
                if lf.undo.can_redo():
                    lf.undo.redo()
            except Exception as exc:
                self._report_error(str(exc).strip() or _ui_label("selection.redo_failed", "Redo failed."))
        elif action == "invert":
            self._execute_selection_stage(lambda: lf.pipeline.select.invert())

        self._refresh_state()
        self._dirty_all()

    def _execute_selection_stage(self, factory):
        try:
            error = _execute_stage(factory())
        except Exception as exc:
            error = str(exc).strip() or _ui_label("selection.operation_failed_generic", "Operation failed.")
        if error:
            self._report_error(error)

    def _report_error(self, message):
        dialog = getattr(lf.ui, "message_dialog", None)
        if callable(dialog):
            try:
                dialog(_ui_label("selection.operation_failed", "Selection Operation Failed"), message, style="error")
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
            "has_scene": (
                "selection_has_scene",
                "selection_depth_near_str",
                "selection_depth_near_value",
                "selection_depth_near_slider_min",
                "selection_depth_near_slider_max",
                "selection_depth_far_str",
                "selection_depth_far_value",
                "selection_depth_far_slider_min",
                "selection_depth_far_slider_max",
            ),
            "has_selection": (
                "selection_has_selection",
                "selection_can_delete",
            ),
            "can_undo": ("selection_can_undo",),
            "can_redo": ("selection_can_redo",),
            "depth_enabled": (
                "selection_depth_mode_active",
                "selection_depth_toggle_label",
            ),
            "depth_near": (
                "selection_depth_near_str",
                "selection_depth_near_value",
                "selection_depth_near_slider_min",
                "selection_depth_near_slider_max",
                "selection_depth_far_slider_min",
            ),
            "depth_far": (
                "selection_depth_far_str",
                "selection_depth_far_value",
                "selection_depth_far_slider_min",
                "selection_depth_far_slider_max",
                "selection_depth_near_slider_max",
            ),
            "window_scale": (
                "selection_depth_scale_str",
                "selection_depth_scale_value",
            ),
            "offset_x": (
                "selection_depth_offset_x_str",
                "selection_depth_offset_x_value",
            ),
            "offset_y": (
                "selection_depth_offset_y_str",
                "selection_depth_offset_y_value",
            ),
            "viz_mode": (
                "selection_viz_mode_label",
                "selection_viz_mode_icon",
            ),
        }

        if "initial" in changed_fields:
            self._dirty_all()
            return
        if not changed_fields:
            return

        dirty_fields = []
        for changed in changed_fields:
            if changed in {"active_tool", "language_generation"}:
                self._dirty_all()
                return
            dirty_fields.extend(field_map.get(changed, ()))

        for field in dict.fromkeys(dirty_fields):
            self._handle.dirty(field)
