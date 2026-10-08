# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Selection controls controller for the viewport selection overlay."""

import math
import time

import lichtfeld as lf

from . import rml_widgets as w
from .ui import RuntimeState

try:
    from .ui import native_value as _native_store_value
except Exception:
    def _native_store_value(_field, fallback):
        return fallback

def _split_view_mode():
    # Test stubs may expose only the getters they need.
    query = getattr(lf.ui, "get_split_view_mode", None)
    if not callable(query):
        return "none"
    try:
        return str(query() or "none")
    except Exception:
        return "none"

def _gt_comparison_active():
    # Broad GT-comparison-mode query; defensive because test stubs replace lf.ui
    # with a bare namespace.
    query = getattr(lf.ui, "is_gt_comparison_active", None)
    return bool(query()) if query else False

_SELECTION_TOOL_ID = "builtin.select"
_GT_COMPARISON = "gt_comparison"
# Deferred records distinguish live edits from completed blurs.
_DEFERRED_LIVE = "live"
_DEFERRED_BLURRED = "blurred"
# _commit_paired_depth_range outcomes:
# DONE: pair resolved; both records consumed.
# DEFERRED: a live member's revalidation was inconsistent; write nothing and
# keep both records for the next stable poll.
# RETARGETED: revalidation changed context, invalidating the grouped target;
# write/consume nothing and let the flush regroup.
_COMMIT_DONE = "done"
_COMMIT_DEFERRED = "deferred"
_COMMIT_RETARGETED = "retargeted"
_DEPTH_MIN = 0.0
_DEPTH_MAX = 1000.0
_DEPTH_GAP = 0.01
_DEPTH_SLIDER_HALF_WINDOW = 20.0
_DEPTH_SLIDER_MIN_SPAN = 1.0
_DEPTH_USER_EDIT_MARK_TTL = 0.75
_DEFAULT_DEPTH_NEAR = 0.0
_DEFAULT_DEPTH_FAR = 6.0
_DEFAULT_FRUSTUM_HALF_WIDTH = 1.35
_DEFAULT_WINDOW_SCALE = 0.35
_DEFAULT_WINDOW_OFFSET = 0.0
_DEFAULT_VIZ_MODE = 1
_SCALE_PERCENT_MIN = 5.0
_SCALE_PERCENT_MAX = 300.0
_REF_RATIO_EPS = 1.0e-3
_OFFSET_PERCENT_MIN = -100.0
_OFFSET_PERCENT_MAX = 100.0
_MISSING = object()
# All three icons keep the same solid depth-window box: Off adds a dotted
# frame, Dim a speckled ring, and Hide leaves the box alone. Keep Dim distinct
# from select-invert.png, which this toolbar already uses for Invert.
_VIZ_MODE_ICONS = {
    0: "../icon/depth-show.png",
    1: "../icon/depth-dim.png",
    2: "../icon/depth-hide.png",
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
        self._window_scale_y = _DEFAULT_WINDOW_SCALE
        self._ref_scale_x = {}
        self._ref_scale_y = dict(self._ref_scale_x)
        self._active_view = int(lf.ui.get_active_view_id())
        self._split_mode = "none"
        # An exhausted refresh leaves cached state untouched and forbids writes.
        self._context_read_exhausted = False
        self._offset_x = _DEFAULT_WINDOW_OFFSET
        self._offset_y = _DEFAULT_WINDOW_OFFSET
        self._viz_mode = _DEFAULT_VIZ_MODE
        self._last_state_key = None
        self._last_state_items = None
        self._depth_echo_holdoff = 0
        self._depth_user_edit_pending = {}
        self._depth_text_bufs = {
            "selection_depth_near_str": None,
            "selection_depth_far_str": None,
            "selection_depth_scale_str": None,
            "selection_depth_offset_x_str": None,
            "selection_depth_offset_y_str": None,
        }
        self._editing_depth_text = set()
        # Track each live edit's current target through focus retargeting. Blur
        # commits before clearing edit state, so validate the origin and revert stale
        # text before that commit can write to a different view.
        self._depth_text_edit_view = {}
        # Deferred records preserve two different edit lifetimes:
        # * live: read the current buffer at flush; typing/Escape and retargeting apply.
        # * blurred: freeze payload and target at blur; clear live edit state so later
        #   focus/mode changes cannot redirect the completed edit.
        # Beginning a new edit on the same key supersedes its older blurred record.
        self._deferred_depth_commits = {}
        self._escape_revert = w.EscapeRevertController()

    def bind_model(self, model):
        # No depth-window surface exists while GT comparison is active: the depth
        # slider block hides for the duration and returns untouched when it ends.
        model.bind_func(
            "selection_depth_mode_active",
            lambda: self._depth_enabled and not _gt_comparison_active(),
        )
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
        model.bind_func("selection_depth_scale_slider_min", lambda: f"{self._scale_slider_bounds()[0]:.0f}")
        model.bind_func("selection_depth_scale_slider_max", lambda: f"{self._scale_slider_bounds()[1]:.0f}")
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
        for element_id, key in (
            ("selection-depth-near-slider", "near"),
            ("selection-depth-far-slider", "far"),
            ("selection-depth-scale-slider", "scale"),
            ("selection-depth-offset-x-slider", "offset_x"),
            ("selection-depth-offset-y-slider", "offset_y"),
        ):
            slider = doc.get_element_by_id(element_id)
            if slider is not None:
                slider.add_event_listener(
                    "mousedown", lambda _event, k=key: self._mark_depth_user_edit(k)
                )
                slider.add_event_listener(
                    "focus", lambda _event, k=key: self._mark_depth_user_edit(k)
                )

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
        # Refresh reconciles references and retargets edits before rebase/state diff,
        # including callers outside update().
        self._refresh_state()
        # Flush before state diff so landed writes arm echo holdoff and dirty fields
        # in this frame, just like immediate commits.
        self._flush_deferred_depth_commits()
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
        if self._last_state_items is not None:
            changed_before = self._changed_state_fields(state_items)
            if (
                "depth_window_draw_generation" in changed_before
                or "depth_window_draw_commit" in changed_before
            ):
                # Rebase the view identified by the completed draw or undo.
                self._rebase_view_reference(self._rebase_target_view())
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
        self._depth_user_edit_pending.clear()
        self._editing_depth_text.clear()
        self._depth_text_edit_view.clear()
        self._deferred_depth_commits.clear()
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

    def _refresh_view_context(self):
        previous = self._split_mode
        old_view = self._active_view
        self._active_view = int(lf.ui.get_active_view_id())
        self._split_mode = _split_view_mode()
        if old_view != self._active_view or (previous == _GT_COMPARISON) != (self._split_mode == _GT_COMPARISON):
            self._refresh_depth_state()
            self._cancel_foreign_depth_text_edits(force=True)

    def _refresh_state(self):
        self._active_mode = self._get_active_mode()
        self._refresh_view_context()
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
                w_enabled, w_near, w_far, scale_x, scale_y, offset_x, offset_y = window_getter()
                enabled = w_enabled
                near = w_near
                far = w_far
                self._window_scale = _clamp(
                    _parse_float(scale_x, _DEFAULT_WINDOW_SCALE), 0.05, 1.0
                )
                self._window_scale_y = _clamp(
                    _parse_float(scale_y, _DEFAULT_WINDOW_SCALE), 0.05, 1.0
                )
                # Check aspect changes against the currently addressed reference.
                # Resolve its key here because setter paths also refresh depth state.
                key = self._ref_key()
                ref_x = self._ref_scale_x.get(key, _DEFAULT_WINDOW_SCALE)
                ref_y = self._ref_scale_y.get(key, _DEFAULT_WINDOW_SCALE)
                cross = abs(
                    self._window_scale * ref_y
                    - self._window_scale_y * ref_x
                )
                norm = max(
                    self._window_scale * ref_y,
                    self._window_scale_y * ref_x,
                    1.0e-6,
                )
                if cross / norm > _REF_RATIO_EPS:
                    self._ref_scale_x[key] = self._window_scale
                    self._ref_scale_y[key] = self._window_scale_y
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
            self._window_scale_y,
            self._offset_x,
            self._offset_y,
        )

    # ---- per-view Size references ------------------------------------

    def _ref_key(self):
        return self._active_view

    def _ref_scale(self, table):
        return table.get(self._ref_key(), _DEFAULT_WINDOW_SCALE)

    def _rebase_target_view(self):
        """Return the draw signal's view only when its generation matches.

        A missing or stale companion dict returns None, selecting the displayed entry.
        """
        commit = self._draw_commit_value()
        try:
            commit_generation = int(commit.get("generation", 0) or 0)
            current_generation = int(RuntimeState.depth_window_draw_generation.value)
        except (TypeError, ValueError):
            return None
        if commit_generation != current_generation:
            return None
        return self._draw_commit_view()


    def _rebase_view_reference(self, view):
        commit = self._draw_commit_value()
        key = self._active_view if view is None else view
        self._ref_scale_x[key] = float(commit.get("scale_x", self._window_scale))
        self._ref_scale_y[key] = float(commit.get("scale_y", self._window_scale_y))


    def _draw_commit_value(self):
        value = RuntimeState.depth_window_draw_commit.value
        return value if isinstance(value, dict) else {}

    def _draw_commit_view(self):
        return int(self._draw_commit_value().get("view", self._active_view))

    def _draw_commit_items(self):
        commit = self._draw_commit_value()
        return (int(commit.get("generation", 0) or 0), self._draw_commit_view())

    # ---- text-edit guard ----------------------------------------------

    def _cancel_foreign_depth_text_edits(self, force=False):
        """Retarget live text edits after focus changes without a pointer click.

        Pointer presses commit blur before changing focus. Clickless changes leave
        the input focused: replace its buffer with current canonical text, dirty it,
        recapture Escape and keep it registered as live. This protects later typing
        from polling and allows repeated retargets; later commits use canonical text
        or whatever the user types next.
        force also retargets edits when the projection changes. RmlUi applies the value update even while the input is focused.
        """
        for key in list(self._editing_depth_text):
            if not force and self._depth_text_edit_view.get(key) == self._active_view:
                continue
            self._depth_text_edit_view[key] = self._active_view
            self._depth_text_bufs[key] = self._canonical_depth_text_value(key)
            # Recapture Escape against the newly displayed view's canonical text.
            self._escape_revert.recapture(key)
            if self._handle:
                self._handle.dirty(key)
        # Keep live edits protected from ordinary text synchronization; do not force it.
        self._sync_depth_text_bufs()

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
            # The depth sliders hide while GT comparison is active.
            ("gt_comparison_active", _gt_comparison_active()),
            ("depth_near", round(self._depth_near, 3)),
            ("depth_far", round(self._depth_far, 3)),
            ("window_scale", round(self._window_scale, 4)),
            ("window_scale_y", round(self._window_scale_y, 4)),
            ("depth_window_draw_generation", RuntimeState.depth_window_draw_generation.value),
            # Compare view and generation so repeated commits on one view still register.
            ("depth_window_draw_commit", self._draw_commit_items()),
            ("offset_x", round(self._offset_x, 4)),
            ("offset_y", round(self._offset_y, 4)),
            ("viz_mode", int(self._viz_mode)),
            # Focus must dirty the chip even when both windows have identical values.
            ("active_view", self._active_view),
            ("split_mode", self._split_mode),
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
        ref = max(self._ref_scale(self._ref_scale_x), 1.0e-6)
        return _clamp(round(self._window_scale / ref * 100.0), _SCALE_PERCENT_MIN, _SCALE_PERCENT_MAX)

    def _scale_factor_bounds(self):
        ref_x = max(self._ref_scale(self._ref_scale_x), 1.0e-6)
        ref_y = max(self._ref_scale(self._ref_scale_y), 1.0e-6)
        f_min = max(0.05 / ref_x, 0.05 / ref_y)
        f_max = min(1.0 / ref_x, 1.0 / ref_y)
        return f_min, max(f_min, f_max)

    def _scale_slider_bounds(self):
        f_min, f_max = self._scale_factor_bounds()
        lo = _clamp(round(f_min * 100.0), _SCALE_PERCENT_MIN, _SCALE_PERCENT_MAX)
        hi = _clamp(round(f_max * 100.0), lo, _SCALE_PERCENT_MAX)
        return lo, hi

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
        f_min, f_max = self._scale_factor_bounds()
        factor = _clamp(percent / 100.0, f_min, f_max)
        self._apply_depth_window(
            self._depth_enabled,
            self._depth_near,
            self._depth_far,
            self._ref_scale(self._ref_scale_x) * factor,
            self._offset_x,
            self._offset_y,
            self._ref_scale(self._ref_scale_y) * factor,
        )

    def _set_depth_offset_x_percent(self, value):
        if not self._visible or self._last_state_key is None:
            return
        self._refresh_depth_state()
        percent = _clamp(_parse_float(value, self._offset_percent(self._offset_x)), _OFFSET_PERCENT_MIN, _OFFSET_PERCENT_MAX)
        self._apply_depth_window(self._depth_enabled, self._depth_near, self._depth_far, self._window_scale, percent / 100.0, self._offset_y, self._window_scale_y)

    def _set_depth_offset_y_percent(self, value):
        if not self._visible or self._last_state_key is None:
            return
        self._refresh_depth_state()
        percent = _clamp(_parse_float(value, self._offset_percent(self._offset_y)), _OFFSET_PERCENT_MIN, _OFFSET_PERCENT_MAX)
        self._apply_depth_window(self._depth_enabled, self._depth_near, self._depth_far, self._window_scale, self._offset_x, percent / 100.0, self._window_scale_y)

    # Range-input entry points. RmlUi replays a slider's pre-update position into
    # its setter when the bound attributes change in the same frame, so these
    # drop the value while the echo holdoff is armed unless the user just pressed
    # that slider (_mark_depth_user_edit). They are bound in bind_model; the
    # text-commit path calls the cores above directly.

    def _set_depth_near_from_slider(self, value):
        if not self._depth_setter_allowed("near"):
            return
        self._set_depth_near(value)

    def _set_depth_far_from_slider(self, value):
        if not self._depth_setter_allowed("far"):
            return
        self._set_depth_far(value)

    def _set_depth_scale_percent_from_slider(self, value):
        if not self._depth_setter_allowed("scale"):
            return
        self._set_depth_scale_percent(value)

    def _set_depth_offset_x_percent_from_slider(self, value):
        if not self._depth_setter_allowed("offset_x"):
            return
        self._set_depth_offset_x_percent(value)

    def _set_depth_offset_y_percent_from_slider(self, value):
        if not self._depth_setter_allowed("offset_y"):
            return
        self._set_depth_offset_y_percent(value)

    def _mark_depth_user_edit(self, key):
        self._depth_user_edit_pending[key] = time.monotonic()

    def _depth_setter_allowed(self, key):
        if not self._visible or self._last_state_key is None:
            return False
        marked_at = self._depth_user_edit_pending.pop(key, None)
        user_edit = (
            marked_at is not None
            and time.monotonic() - marked_at <= _DEPTH_USER_EDIT_MARK_TTL
        )
        return self._depth_echo_holdoff == 0 or user_edit

    def _apply_depth_range(self, enabled, near, far):
        self._apply_depth_window(enabled, near, far, self._window_scale, self._offset_x, self._offset_y, self._window_scale_y)

    def _apply_depth_window(self, enabled, near, far, scale, offset_x, offset_y, scale_y):
        self._depth_enabled = bool(enabled)
        self._depth_near = _clamp(near, _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        self._depth_far = _clamp(far, self._depth_near + _DEPTH_GAP, _DEPTH_MAX)
        self._window_scale = _clamp(scale, 0.05, 1.0)
        self._window_scale_y = _clamp(scale_y, 0.05, 1.0)
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
                    self._window_scale_y,
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
        # A new edit supersedes this key's completed blur. Keeping its frozen record
        # could overwrite the new buffer and retire the new edit during flush.
        record = self._deferred_depth_commits.get(key)
        if record is not None and record.get("kind") == _DEFERRED_BLURRED:
            self._deferred_depth_commits.pop(key, None)
        self._editing_depth_text.add(key)
        self._depth_text_edit_view[key] = self._active_view

    def _end_depth_text_edit(self, key):
        # The widget always calls commit before on_blur. Freeze any deferred value
        # and target here, including Escape-restored text, then end the live edit.
        # Keeping it live would let a later focus/mode change redirect a completed blur.
        record = self._deferred_depth_commits.get(key)
        if record is not None:
            record["kind"] = _DEFERRED_BLURRED
            record["payload"] = self._depth_text_bufs.get(key)
            record["view"] = self._depth_text_edit_view.get(key)
        self._editing_depth_text.discard(key)
        self._depth_text_edit_view.pop(key, None)

    def _flush_deferred_depth_commits(self):
        """Flush deferred commits after update() has refreshed the context.

        Live records revalidate and read the current buffer; blurred records keep
        their frozen payload and view. Pending keys remain protected from text sync.

        Group by write target. Near/Far clamp against each other, so resolve their
        latest intended pair before one combined write. Even insertion order fails
        if re-arming Near moves it behind Far: intended .20/.50 against native Near
        .90 would clamp Far to .91. A lone endpoint uses its target's current
        counterpart. Size and offsets have no such cross-field coupling.
        Recheck records against the current registry because writes or new edits can
        remove or replace them; snapshots keep removals from disrupting iteration.

        Paired and single live paths can request regrouping after revalidation.
        Regroup once; a second move leaves pending records for the next poll.
        Done continues. Deferred or an unknown outcome stops the whole flush without
        writing/consuming any remaining sibling or group. Earlier writes remain
        committed; flushing is not transactional across groups.
        """
        if self._context_read_exhausted or not self._deferred_depth_commits:
            return
        restarted = False
        while True:
            retargeted = False
            for target, records in self._group_deferred_depth_records():
                near_record = records.pop("selection_depth_near_str", None)
                far_record = records.pop("selection_depth_far_str", None)
                outcome = _COMMIT_DONE
                if near_record is not None and far_record is not None:
                    outcome = self._commit_paired_depth_range(target, near_record, far_record)
                elif near_record is not None:
                    outcome = self._commit_deferred_depth_record(
                        "selection_depth_near_str", near_record
                    )
                elif far_record is not None:
                    outcome = self._commit_deferred_depth_record(
                        "selection_depth_far_str", far_record
                    )
                if outcome == _COMMIT_DONE:
                    for key, record in list(records.items()):
                        outcome = self._commit_deferred_depth_record(key, record)
                        if outcome != _COMMIT_DONE:
                            break
                # Handle every outcome; unknown results must stop further writes.
                if outcome == _COMMIT_DONE:
                    continue
                elif outcome == _COMMIT_RETARGETED:
                    retargeted = True
                    break
                elif outcome == _COMMIT_DEFERRED:
                    # An exhausted read stops all remaining groups without consuming their records.
                    return
                else:
                    # Unknown outcomes stop the flush just like deferred ones.
                    return
            if not retargeted or restarted:
                return
            restarted = True

    def _group_deferred_depth_records(self):
        """Return (target, {key: record}) groups in first-seen target order.

        Regrouping uses the current registry, resuming after earlier writes consumed
        records.
        """
        targets = []
        grouped = {}
        for key in list(self._deferred_depth_commits):
            record = self._deferred_depth_commits.get(key)
            if record is None:
                continue
            target = self._deferred_write_target(record)
            if target not in grouped:
                grouped[target] = {}
                targets.append(target)
            grouped[target][key] = record
        return [(target, grouped[target]) for target in targets]

    def _deferred_write_target(self, record):
        return record.get("view", self._active_view) if record.get("kind") == _DEFERRED_BLURRED else self._active_view

    def _deferred_write_destination(self, record):
        """Resolve a setter target to a window identity for revalidation.

        None selects the ordinary setter, whose destination can change with focus.
        Resolve it to the current reference key before comparing across a refresh.
        """
        target = self._deferred_write_target(record)
        if target is not None:
            return target
        return self._ref_key()

    def _commit_deferred_depth_record(self, key, record):
        """Flush one record using the same outcomes as paired writes.

        A live edit revalidates its destination and registry identity before writing.
        A change returns retargeted without consuming it; the caller regroups once.
        A blurred record needs no refresh and completes against its frozen target.
        """
        if record.get("kind") == _DEFERRED_BLURRED:
            self._commit_blurred_depth_record(key, record)
            return _COMMIT_DONE
        if key in self._editing_depth_text:
            destination = self._deferred_write_destination(record)
            self._refresh_view_context()
            if self._context_read_exhausted:
                # Preserve the pending record and stop the flush on exhaustion.
                if key not in self._deferred_depth_commits:
                    self._deferred_depth_commits[key] = {"kind": _DEFERRED_LIVE}
                return _COMMIT_DEFERRED
            if self._deferred_depth_commits.get(key) is not record:
                return _COMMIT_RETARGETED
            if self._deferred_write_destination(record) != destination:
                return _COMMIT_RETARGETED
            # Do not refresh again after this check: another refresh could retarget
            # the edit beyond the destination just validated.
            self._commit_depth_text_key(key, revalidated=True)
            return _COMMIT_DONE
        self._commit_depth_text_key(key)
        return _COMMIT_DONE

    def _commit_paired_depth_range(self, view, near_record, far_record):
        """Resolve one target's intended Near/Far pair before a combined write.

        Normal refusals, such as invalid text or a hidden view, consume both records.
        An exhausted live refresh preserves the pair. After revalidation, a changed
        record identity or target returns retargeted without writing or consuming it.
        """
        pairing = (
            ("selection_depth_near_str", near_record),
            ("selection_depth_far_str", far_record),
        )
        resolved = {}
        revalidated = False
        for key, record in pairing:
            if record.get("kind") == _DEFERRED_BLURRED:
                payload = record.get("payload")
            else:
                if key in self._editing_depth_text:
                    self._refresh_view_context()
                    revalidated = True
                    if self._context_read_exhausted:
                        return _COMMIT_DEFERRED
                payload = self._depth_text_bufs.get(key)
            resolved[key] = self._parse_depth_text_value(key, payload)
        if revalidated:
            for key, record in pairing:
                if self._deferred_depth_commits.get(key) is not record:
                    return _COMMIT_RETARGETED
                if self._deferred_write_target(record) != view:
                    return _COMMIT_RETARGETED
        self._deferred_depth_commits.pop("selection_depth_near_str", None)
        self._deferred_depth_commits.pop("selection_depth_far_str", None)
        values = {key: value for key, value in resolved.items() if value is not None}
        if not values or not self._visible or self._last_state_key is None:
            self._sync_depth_text_bufs(force=True)
            return _COMMIT_DONE
        # Arm before dirtying sliders so their old positions cannot echo into setters.
        self._depth_echo_holdoff = 2
        if view == self._active_view:
            self._refresh_depth_state()
            near = _clamp(
                values.get("selection_depth_near_str", self._depth_near),
                _DEPTH_MIN,
                _DEPTH_MAX - _DEPTH_GAP,
            )
            far = _clamp(
                values.get("selection_depth_far_str", self._depth_far),
                near + _DEPTH_GAP,
                _DEPTH_MAX,
            )
            self._apply_depth_range(self._depth_enabled, near, far)
        self._sync_depth_text_bufs(force=True)
        return _COMMIT_DONE

    def _commit_blurred_depth_record(self, key, record):
        """Apply a completed blur's frozen payload to its frozen target.

        Consume the record even for invalid text or a hidden view, as ordinary
        commit refusals do not require another retry.
        """
        if record.get("view", self._active_view) != self._active_view:
            self._deferred_depth_commits.pop(key, None)
            return _COMMIT_DONE

        self._deferred_depth_commits.pop(key, None)
        parsed = self._parse_depth_text_value(key, record.get("payload"))
        if parsed is None or not self._visible or self._last_state_key is None:
            self._sync_depth_text_bufs(force=True)
            return
        # Arm before dirtying sliders so their old positions cannot echo into setters.
        self._depth_echo_holdoff = 2
        self._dispatch_depth_setter(key, parsed)
        self._sync_depth_text_bufs(force=True)

    def _capture_depth_text_snapshot(self, key):
        return self._canonical_depth_text_value(key)

    def _restore_depth_text_snapshot(self, key, snapshot):
        # Escape restores the pre-edit text and the host blurs immediately after
        # (cancelFocusedElement in rml_input_utils.hpp), so that blur is what
        # carries the revert to native state. Sidebar/dock and viewport hosts both
        # dispatch this Escape-then-blur sequence; the depth fields use the overlay.
        self._depth_text_bufs[key] = str(snapshot or "")
        if self._handle:
            self._handle.dirty(key)

    def _commit_depth_text_key(self, key, revalidated=False):
        # Validate live edit origins before reading their buffers. Focus, mode or
        # view may have changed before the poll; the shared refresh retargets to
        # canonical text first. A commit uses that value or subsequent typing.
        # The overlay's model hook runs before Context::Update(), and RmlUi applies
        # data-value even to focused inputs. Keys with no live origin keep the buffer
        # this commit is about to read.
        #
        # On exhaustion, record a deferred live commit without changing draft or edit
        # state. Another commit or the first comparable poll retries automatically;
        # quiet native state alone cannot guarantee that read. Blur freezes the value
        # and target so the completed edit still has a record to replay.
        # revalidated is set only after the flush wrapper checks context: refreshing
        # again here could invalidate its destination check.
        if not revalidated and key in self._editing_depth_text:
            self._refresh_view_context()
            if self._context_read_exhausted:
                if key not in self._deferred_depth_commits:
                    self._deferred_depth_commits[key] = {"kind": _DEFERRED_LIVE}
                return
        self._deferred_depth_commits.pop(key, None)
        value = self._depth_text_bufs.get(key)
        if value is not None and value.strip():
            parsed = self._parse_depth_text_value(key, value)
            if parsed is None:
                self._sync_depth_text_bufs(force=True)
                return
            # The cores below refuse while the view is hidden or before the
            # first update lands. Refuse here too, ahead of the arming: arming
            # for a write that is then refused would leave the holdoff set, and a
            # hidden update returns before the decrement, so it would still be
            # armed when the view comes back.
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
            self._dispatch_depth_setter(key, parsed)

        self._sync_depth_text_bufs(force=True)

    def _parse_depth_text_value(self, key, value):
        """The number a field's text carries, or None if it carries none."""
        if value is None:
            return None
        value = str(value)
        if not value.strip():
            return None
        parsed_src = (
            value.split("×", 1)[0].strip().rstrip("%")
            if key == "selection_depth_scale_str"
            else value
        )
        return _parse_float(parsed_src, None)

    def _dispatch_depth_setter(self, key, parsed):
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

    def _sync_depth_text_bufs(self, force=False):
        for key in self._depth_text_bufs:
            # Preserve pending buffers even under force=True: live flush reads them,
            # and blurred fields wait for their frozen write. A committed key's record
            # is already removed, so only its still-pending siblings are skipped.
            if key in self._deferred_depth_commits:
                continue
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
            self._depth_echo_holdoff = 0
            self._refresh_depth_state()
            self._apply_depth_range(not self._depth_enabled, self._depth_near, self._depth_far)
        elif action == "cycle_viz":
            self._cycle_viz_mode()
        elif action == "delete":
            self._execute_selection_stage(lambda: lf.pipeline.edit.delete_())
        elif action == "select_all":
            self._run_selection_command(lf.ui.select_all_gaussians)
        elif action == "unselect":
            self._run_selection_command(lf.ui.deselect_all_gaussians)
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
            self._run_selection_command(lf.ui.invert_gaussian_selection)

        self._refresh_state()
        self._dirty_all()

    def _execute_selection_stage(self, factory):
        try:
            error = _execute_stage(factory())
        except Exception as exc:
            error = str(exc).strip() or _ui_label("selection.operation_failed_generic", "Operation failed.")
        if error:
            self._report_error(error)

    def _run_selection_command(self, command):
        """The toolbar buttons run the same commands as the Select menu and their shortcuts."""
        try:
            command()
        except Exception as exc:
            self._report_error(str(exc).strip() or _ui_label("selection.operation_failed_generic", "Operation failed."))

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
            "gt_comparison_active": ("selection_depth_mode_active",),
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
                "selection_depth_scale_slider_min",
                "selection_depth_scale_slider_max",
            ),
            "window_scale_y": ("selection_depth_scale_str",),
            "depth_window_draw_generation": (
                "selection_depth_scale_str",
                "selection_depth_scale_value",
                "selection_depth_scale_slider_min",
                "selection_depth_scale_slider_max",
            ),
            "depth_window_draw_commit": (
                "selection_depth_scale_str",
                "selection_depth_scale_value",
                "selection_depth_scale_slider_min",
                "selection_depth_scale_slider_max",
            ),
            "active_view": (
                "selection_depth_scale_str",
                "selection_depth_scale_value",
                "selection_depth_scale_slider_min",
                "selection_depth_scale_slider_max",
            ),
            "split_mode": (
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
