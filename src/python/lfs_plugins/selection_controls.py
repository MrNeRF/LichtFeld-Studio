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
    # Defensive like _gt_comparison_active: test stubs replace lf.ui with a bare
    # namespace that carries only the getters a given test needs.
    query = getattr(lf.ui, "get_split_view_mode", None)
    if not callable(query):
        return "none"
    try:
        return str(query() or "none")
    except Exception:
        return "none"


def _focused_split_panel():
    query = getattr(lf.ui, "get_focused_split_panel", None)
    if not callable(query):
        return _PANEL_LEFT
    try:
        value = str(query() or _PANEL_LEFT)
    except Exception:
        return _PANEL_LEFT
    return value if value in (_PANEL_LEFT, _PANEL_RIGHT) else _PANEL_LEFT


def _depth_window_sync():
    query = getattr(lf.ui, "get_depth_window_sync", None)
    if not callable(query):
        return False
    try:
        return bool(query())
    except Exception:
        return False


def _depth_window_collapse_source():
    """Which panel the last independent-dual collapse folded, or None.

    Leaving independent-dual copies the PRE-transition focused panel's window
    into the single remaining one (rendering_manager.cpp, the collapse branch of
    applyDepthWindowModeTransitionLocked) while the split service resets the
    observable focus to Left in the SAME transition
    (split_view_service.cpp:214). A poller that saw neither the focus change nor
    the leave separately therefore cannot recover the source panel from its own
    cache or from get_focused_split_panel(); this is the native record of it.

    None means the binding does not expose it (an older module, or a test stub
    that does not need it), and the caller falls back to its cached
    pre-transition panel.
    """
    query = getattr(lf.ui, "get_depth_window_collapse_source", None)
    if not callable(query):
        return None
    try:
        value = str(query() or "")
    except Exception:
        return None
    return value if value in (_PANEL_LEFT, _PANEL_RIGHT) else None


def _depth_window_collapse_record():
    """The reference-lineage stamp: (source, generation, kind), read atomically.

    The source alone is endpoint identity: a leave -> enter -> leave cycle
    completed between two 100ms polls reports only its LAST leg, so a consumer
    reading it cannot tell that cycle apart from the single leave it observed --
    and every per-panel value it cached predates the first collapse of the
    cycle. The generation (rendering_manager.cpp,
    stampDepthWindowLineageLocked, which always holds settings_mutex_ and
    always moves source, generation and kind together) counts the writes
    that invalidate a slot-derived cache, so the delta between two reads says
    how many destructive boundaries went by. Retained-pair discards stamp too;
    valid GT/Disabled excursions and restoration do not. Writers stamp inside
    their critical section except sync undo/redo, which stamps from its call site
    (depth_window_undo_entry.cpp:100) after the restore released the lock, so
    it takes settings_mutex_ a SECOND time. The record is therefore always
    self-consistent, but it is not always written under the same lock hold as
    the slots it describes -- which is why _refresh_panel_context revalidates
    the generation around its endpoint reads instead of trusting one read.
    The kind names WHICH write stamped it -- a leave collapse, a sync-ON copy,
    a fresh-baseline restore (project load or sync undo/redo), or a retained-pair
    discard -- because recovery differs per kind and the endpoint alone does
    not identify it.

    Returns (source, generation, kind). generation and kind are None when the
    binding is not exposed (an older module, or a test stub that does not need
    it), in which case the source falls back to the single-value getter and
    callers keep the pre-lineage behaviour. A two-element tuple from a
    pre-`kind` module degrades the same way, on the kind alone.
    """
    query = getattr(lf.ui, "get_depth_window_collapse_record", None)
    if callable(query):
        source, generation, kind = None, None, None
        try:
            record = tuple(query())
        except Exception:
            record = ()
        if len(record) >= 2:
            source = str(record[0] or "")
            source = source if source in (_PANEL_LEFT, _PANEL_RIGHT) else None
            try:
                generation = int(record[1])
            except (TypeError, ValueError):
                generation = None
            if len(record) >= 3:
                kind = str(record[2] or "")
                kind = kind if kind in _LINEAGE_KINDS else None
            return source, generation, kind
    return _depth_window_collapse_source(), None, None


def _split_mode_touches_depth_window(previous_mode, new_mode):
    """Does this split-mode change mean anything to depth-window state?

    The MIRROR of the native predicate that governs the whole of
    applyDepthWindowModeTransitionLocked (rendering_manager.cpp): only crossing
    the independent-dual boundary (where per-panel windows exist at all) or the
    GT-comparison boundary (which suspends the depth filter entirely) moves any
    depth-window state. Other mode changes normally do nothing. Discarding a
    retained pair on Disabled -> PLYComparison is witnessed separately by the
    lineage stamp; without one, that edge must not cancel a legitimate edit.
    """
    if previous_mode == new_mode:
        return False
    if (previous_mode == _INDEPENDENT_DUAL) != (new_mode == _INDEPENDENT_DUAL):
        return True
    return (previous_mode == _GT_COMPARISON) != (new_mode == _GT_COMPARISON)


def _gt_comparison_active():
    # Broad GT-comparison-mode query; defensive because test stubs replace lf.ui
    # with a bare namespace.
    query = getattr(lf.ui, "is_gt_comparison_active", None)
    return bool(query()) if query else False


_SELECTION_TOOL_ID = "builtin.select"
_PANEL_LEFT = "left"
_PANEL_RIGHT = "right"
# The Size slider's 100% reference is per panel. While panel sync
# is on -- and in every mode that has only one window -- the two references
# collapse to this single shared entry.
_PANEL_SHARED = "shared"
_INDEPENDENT_DUAL = "independent_dual"
_GT_COMPARISON = "gt_comparison"
# The two kinds of deferred-commit record (_deferred_depth_commits): an edit the
# user is still in, and one they have already finished with a blur.
_DEFERRED_LIVE = "live"
_DEFERRED_BLURRED = "blurred"
# _commit_paired_depth_range's outcomes. DONE: the pair was resolved and both
# records consumed. DEFERRED: a live member's revalidating read was torn, so
# nothing was written and both records stay for the next stable poll.
# RETARGETED: that revalidation moved the context underneath the grouping, so
# the pre-computed target no longer describes where these records belong --
# nothing is written, nothing is consumed, and the flush regroups from scratch.
_COMMIT_DONE = "done"
_COMMIT_DEFERRED = "deferred"
_COMMIT_RETARGETED = "retargeted"
# Native reference invalidations (rendering_manager.hpp, DepthWindowLineageKind).
# Sync undo/redo shares the project-restore recovery of absolute windows.
_LINEAGE_LEAVE_COLLAPSE = "leave_collapse"
_LINEAGE_SYNC_COPY = "sync_copy"
_LINEAGE_PROJECT_RESTORE = "project_restore"
_LINEAGE_RETAINED_PAIR_DISCARD = "retained_pair_discard"
_LINEAGE_KINDS = (
    _LINEAGE_LEAVE_COLLAPSE,
    _LINEAGE_SYNC_COPY,
    _LINEAGE_PROJECT_RESTORE,
    _LINEAGE_RETAINED_PAIR_DISCARD,
)
# How many times _refresh_panel_context re-reads the panel context when the
# lineage generation moves underneath it. Bounded so a stamp storm cannot spin;
# if the final attempt is still torn the refresh CONSUMES NOTHING and the whole
# tick is a no-op, leaving the delta pending for the next (stable) poll.
_CONTEXT_READ_ATTEMPTS = 3
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
# One family, three frames, and the SAME solid box in all three -- the box is
# the depth window and it never changes; only what surrounds it does. Off draws
# a dotted frame around it (everything outside is still there), Dim replaces
# that frame with a ring of specks, Hide leaves the box alone on the field.
# The eye / eye-slash pair these two replaced said nothing about the depth
# window and did not read as a set with the Dim frame between them; the eye
# icons stay in use everywhere else in the app. Dim is not select-invert.png:
# the selection toolbar's INVERT button in this same panel is that exact file.
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
_PANEL_CHIP_LABELS = {
    _PANEL_LEFT: ("ui.selection_depth_panel_left", "L"),
    _PANEL_RIGHT: ("ui.selection_depth_panel_right", "R"),
}
_SYNC_ICON_ON = "../icon/layout-columns.png"
# OFF is not one frame but two. The button is only ever shown in
# independent-dual split, where the two panels hold separate depth windows and
# the sliders address exactly one of them -- so the OFF frame says WHICH, by
# filling the half of the same two-column glyph that stands for the focused
# panel. ON stays the plain unfilled glyph: with sync on there is no focused
# half to point at, both panels share one window.
_SYNC_ICON_OFF = {
    _PANEL_LEFT: "../icon/layout-columns-left.png",
    _PANEL_RIGHT: "../icon/layout-columns-right.png",
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
        "selection_panel_chip_visible",
        "selection_panel_chip_label",
        "selection_depth_sync_active",
        "selection_depth_sync_label",
        "selection_depth_sync_icon",
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
        # PER-PANEL Size references. Keyed by
        # the panel string, plus a shared entry used whenever there is only one
        # window (sync on, or any mode that is not independent-dual). A focus
        # switch selects a DIFFERENT entry; it never rewrites one, so it can
        # never read as "a new shape was drawn".
        self._ref_scale_x = {
            _PANEL_SHARED: _DEFAULT_WINDOW_SCALE,
            _PANEL_LEFT: _DEFAULT_WINDOW_SCALE,
            _PANEL_RIGHT: _DEFAULT_WINDOW_SCALE,
        }
        self._ref_scale_y = dict(self._ref_scale_x)
        # The left/right entries survive a GT excursion only while native has
        # invalidated nothing since they were retained. Keep this witness apart
        # from the current record, which advances on every successful refresh.
        self._retained_reference_generation = None
        # Observing GT without a prior pair cannot recover the user's baselines;
        # on return, baseline each restored slot from its own current window.
        self._gt_baseline_pending = False
        self._focused_panel = _PANEL_LEFT
        self._split_mode = "none"
        self._depth_sync = False
        # The reference-lineage channel, cached as ONE triple read atomically
        # from the manager (see _depth_window_collapse_record). The generation
        # counts native writes that invalidate slot-derived state;
        # comparing its delta against the transition this poll actually observed
        # is how a cycle that happened entirely between two polls becomes
        # visible, and the kind is how the recovery is chosen. None means "not
        # exposed" (an older module or a stub), and every rule below degrades to
        # the plain source-identity behaviour.
        self._collapse_source = None
        self._collapse_generation = None
        self._collapse_kind = None
        # True while the last _refresh_panel_context gave up with a torn read
        # set. The tick consumed nothing, so nothing may be written from it.
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
        # The panel an in-flight text edit is currently TARGETED AT -- the panel
        # it was started on, then whichever panel a focus change retargets it to
        # (_cancel_foreign_depth_text_edits). Blur commits the buffer before it
        # clears the edit state (rml_widgets.bind_committed_text_input), so
        # cancelling on focus change alone still races that commit; the origin
        # is what lets the cancel neutralise it.
        self._depth_text_edit_panel = {}
        # Keys whose commit was DEFERRED because the validating context read came
        # back exhausted, as key -> RECORD. Two kinds, and the difference is the
        # whole point of the table being records rather than a bare set:
        #
        # * {"kind": "live"} -- the edit is still focused (an Enter-deferred
        #   commit, no blur yet). The flush reads the BUFFER at flush time, which
        #   is the right semantics for a live field: whatever the user types
        #   next, or an Escape, legitimately wins over the pending value. These
        #   records follow the ordinary retarget/cancel machinery on a panel
        #   transition, because the field really is still being edited.
        #
        # * {"kind": "blurred", "payload": ..., "panel": ...} -- the user
        #   FINISHED the edit and the blur landed mid-storm. There is no live
        #   field left to read, so the record FREEZES what the blur carried: the
        #   buffer as it stood at blur time, and the panel the edit was targeted
        #   at then. A completed intent cannot be retargeted by anything the
        #   focus or the mode does afterwards, so the flush writes the frozen
        #   payload to the frozen panel and the live-edit bookkeeping is retired
        #   at blur time (which is also what keeps a completed blur from
        #   masquerading as a live edit to the transition guard).
        #
        # A blurred record is superseded -- dropped outright -- when the user
        # re-begins an edit on the same key: the latest intent wins, and the new
        # edit gets a clean lifecycle (_begin_depth_text_edit).
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
        # The chip and the sync toggle exist only where per-panel windows do.
        model.bind_func("selection_panel_chip_visible", self._panel_controls_visible)
        model.bind_func("selection_panel_chip_label", self._panel_chip_label)
        model.bind_func("selection_depth_sync_active", lambda: self._depth_sync)
        model.bind_func("selection_depth_sync_label", self._sync_toggle_label)
        model.bind_func("selection_depth_sync_icon", self._sync_toggle_icon)
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
        # _refresh_state() refreshes the panel context and reconciles the
        # references itself, so the entry the rest of this frame reads is
        # already the right one before the rebase and the state diff. It also
        # owns the edit-retarget guard on a focus edge, so that every OTHER
        # caller of the refresh gets it too -- see _refresh_panel_context.
        self._refresh_state()
        # Any commit the storm deferred is retried here, against the refresh this
        # poll just did. It runs BEFORE the state diff below so a landed write is
        # seen by this frame's holdoff arming and dirtying, exactly as an
        # in-frame commit would be.
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
                # Rebase the reference of the panel the SIGNAL names, not
                # whichever window the toolbar happens to display.
                # Undoing an R-panel drag while L is focused must leave L alone.
                self._rebase_panel_reference(self._rebase_target_panel())
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
        self._depth_text_edit_panel.clear()
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

    def _refresh_panel_context(self):
        """Re-read focus / split mode / sync flag and reconcile on any edge.

        THE single place the three cached panel-context values are refreshed
        from the native side. Every channel that can move them -- update()'s
        poll, the toolbar sync toggle, and toolbar Undo/Redo of a
        DepthWindowSyncUndoEntry (which restores the flag natively,
        depth_window_undo_entry.cpp:85) -- goes through here, so an observed
        sync edge reconciles the references exactly once whatever delivered it.
        _reconcile_panel_references is itself edge-guarded, so a refresh
        that moves nothing reconciles nothing and callers cannot double-fire.

        The FOCUS edge is consumed here too, for the same reason. update() used
        to own the edit-retarget guard, but it is not the only caller that moves
        the cached focus: the sync toggle refreshes twice and _on_action
        refreshes after every toolbar action. Any of those consuming the edge
        silently would leave an active text edit pointing at the panel it
        started on, and its blur or Enter would then write into the WRONG panel.
        One code path, one consumer.

        Returns the pre-refresh (panel, sync, mode) triple for callers that
        need the transition, not just the new state.
        """
        previous_panel = self._focused_panel
        previous_sync = self._depth_sync
        previous_mode = self._split_mode
        previous_generation = self._collapse_generation
        # GENERATION REVALIDATION. The endpoint (mode / focus / sync) and the
        # lineage record are FOUR separate native reads, each locked on its own,
        # and a stamp landing BETWEEN them tears the set: the reconciliation then
        # runs a fresh record against endpoint state from before the write that
        # stamped it. That is not eventually consistent. A sync undo restoring
        # {L=.60, R=.20, sync=false} read with a stale sync=true makes
        # _fresh_baseline_references take its single-window branch and seed all
        # three entries from the focused .20; the NEXT poll then observes the
        # sync-OFF edge and copies that shared .20 into both panel entries, so
        # Left reports 300% permanently.
        #
        # So the record is read FIRST, then the endpoint, then the record AGAIN.
        # The generation is a monotonic counter bumped once per slot-
        # invalidating write (writers stamp inside their own critical section,
        # except sync undo/redo from its call site under a
        # second lock -- see _depth_window_collapse_record), so a generation
        # that moved across the two reads is proof that the set is TORN.
        #
        # A moved generation retries the whole set, bounded. An UNMOVED
        # generation is not by itself proof that nothing landed -- the sync
        # undo/redo restores the slots and stamps under two SEPARATE lock holds
        # (depth_window_undo_entry.cpp:88 and :100) -- but the retry loop only
        # has to catch the tear it can see, and a stamp that lands after a
        # completed re-read stays pending as a delta until the first poll that
        # gets a COMPARABLE (equal-generation) pair of record reads bracketing
        # its endpoint reads -- which may not be the next one (see the
        # convergence note below).
        #
        # If the last attempt is still torn the set is NOT usable: the endpoint
        # can be pre-write while the record is post-write, and consuming that
        # generation would retire an advance whose endpoint was never observed.
        # (A large delta is not a safe substitute for a consistent set: a stale
        # sync=true retained beside a fresh 'project_restore' stamp of
        # {L=.60, R=.20} sends _fresh_baseline_references down its single-window
        # branch, seeding every entry from the focused .20, and the next poll's
        # sync-OFF edge then copies that .20 into both panels for good.)
        #
        # So an EXHAUSTED TICK IS A NO-OP. It consumes NOTHING: not the lineage
        # record, not the endpoint (mode / focus / sync), no reconciliation, no
        # edit retarget. Every cached field is left exactly as the last stable
        # tick left it, and the whole delta stays pending for the first poll that
        # gets a COMPARABLE (equal-generation) pair of record reads bracketing its
        # endpoint reads -- which may not be the next one, and is not guaranteed
        # by quiet native state alone (see the convergence note below). That poll
        # re-reads a stable world and then processes everything in the CORRECT
        # ORDER -- reconcile the references first, canonicalize the Size text
        # from them afterwards.
        #
        # Caching the endpoint while deferring the lineage is NOT safe, which is
        # why nothing is cached now. The retarget guard below would fire on the
        # newly cached focus/mode edge and canonicalize the Size field through
        # _refresh_depth_state / _cancel_foreign_depth_text_edits -- but those
        # read self._ref_scale_x, which the skipped reconciliation left seeded
        # from the OTHER panel. With native scale .20 against a stale shared
        # reference .90 the field is rewritten to "22%" instead of "100%", and a
        # commit landing during a second exhausted refresh then writes that 22%
        # back into native state. Deferring the guard costs one poll; caching the
        # endpoint corrupts state permanently.
        #
        # The next poll's delta is whatever the world actually shows, and the
        # normal rules process it: usually >= _CONTEXT_READ_ATTEMPTS, but not
        # necessarily -- a record read that FAILS reports generation None (see
        # _depth_window_collapse_record), so alternating failed and successful
        # reads can exhaust the loop on a single real stamp and leave the next
        # poll a delta of one. That is fine, and it is the actual invariant here:
        # the deferred consumption is processed by the ordinary delta rules on a
        # later, stable poll, WHATEVER the delta turns out to be.
        #
        # Convergence is NOT guaranteed by quiet native state alone. A record
        # read that fails reports generation None, and None != None is false only
        # by luck of pairing: alternating failed and successful reads compare
        # unequal in every attempt and exhaust the loop with the native
        # generation never moving at all. So "the first poll after the stamps
        # stop" is not the criterion -- what is required is a later poll in which
        # some attempt gets a COMPARABLE (equal-generation) pair of record reads
        # bracketing its endpoint reads. Until that happens the tick simply
        # exhausts again.
        #
        # The property that holds regardless is the safety one: an exhausted tick
        # is inert. It consumes nothing, writes nothing and caches nothing, so
        # repeated exhaustion DELAYS the pending delta without ever corrupting
        # it, and the ordinary delta rules process it correctly whenever the
        # first stable read finally lands. A deferral of the guard, the
        # reconciliation or a commit costs polls, not state.
        #
        # This lives here, in the ONE place the cached panel context is
        # refreshed, so every channel (the poll, the sync toggle, toolbar
        # undo/redo) inherits it.
        exhausted = True
        for _attempt in range(_CONTEXT_READ_ATTEMPTS):
            record = _depth_window_collapse_record()
            split_mode = _split_view_mode()
            focused_panel = _focused_split_panel()
            depth_sync = _depth_window_sync()
            revalidated = _depth_window_collapse_record()
            stable = revalidated[1] == record[1]
            # The POST-endpoint read is the one retained, always. It is the only
            # one of the two that cannot predate an endpoint value, so it is the
            # right record both when the generation moved and when it did not --
            # including the None-generation case of a binding that does not
            # expose the counter, where `None == None` ends the loop without
            # proving anything about the source: a Right-panel leave landing
            # between the two reads leaves the FIRST record naming Left while
            # the endpoint is already post-leave, and the observed-leave rule
            # would then seed shared from the wrong panel, permanently.
            record = revalidated
            if stable:
                exhausted = False
                break
        # Recorded for the commit path: a write must never be applied against a
        # torn snapshot (see _commit_depth_text_key).
        self._context_read_exhausted = exhausted
        if exhausted:
            return previous_panel, previous_sync, previous_mode
        self._split_mode = split_mode
        self._focused_panel = focused_panel
        self._depth_sync = depth_sync
        (
            self._collapse_source,
            self._collapse_generation,
            self._collapse_kind,
        ) = record
        self._reconcile_panel_references(
            previous_panel, previous_sync, previous_mode, previous_generation
        )
        # A MODE BOUNDARY retargets an active edit even when the observable
        # focus did not move. Focus equality is not a safe proxy for "the edit's
        # context is unchanged": leaving independent-dual resets focus to Left
        # (split_view_service.cpp:214), so an edit started on Left, with native
        # focus moving to Right and the mode left before one refresh, sees Left
        # both before and after -- while the field it was editing is now the
        # SINGLE global window collapsed from RIGHT. Its buffer is stale-origin
        # text, and a blur or Enter would write it into that global window.
        mode_boundary = self._mode_boundary_edge(previous_mode, previous_generation)
        if self._focused_panel != previous_panel or mode_boundary:
            # The retarget reverts each field to the canonical text of the panel
            # NOW on screen, so the cached window values have to be that panel's
            # before it runs. Re-reading them is a pure re-read of the same
            # getters update() calls a moment later, so it is idempotent.
            self._refresh_depth_state()
            # A focus edge retargets only the edits that came from ANOTHER panel
            # (the pinned cancel semantics). A mode boundary invalidates the
            # edit's context whichever panel it names, so it forces the revert.
            self._cancel_foreign_depth_text_edits(force=mode_boundary)
        return previous_panel, previous_sync, previous_mode

    def _mode_boundary_edge(self, previous_mode, previous_generation):
        """Did the context of an in-flight edit change this refresh?

        Either the split mode crossed a boundary the NATIVE side acts on (a
        no-op mode change must not cancel a legitimate typed buffer -- see
        _split_mode_touches_depth_window), or the native side stamped the
        lineage channel at least once since the last read, including writes this
        poll could not see as a mode change at all.
        """
        if _split_mode_touches_depth_window(previous_mode, self._split_mode):
            return True
        return self._lineage_delta(previous_generation) not in (None, 0)

    def _lineage_delta(self, previous_generation):
        """How many slot-invalidating native writes happened since the last read.

        None when the channel is not exposed (an older module, or a stub that
        does not need it), which every rule below degrades on: the plain
        source-identity behaviour is the correct fallback there.
        """
        if previous_generation is None or self._collapse_generation is None:
            return None
        return self._collapse_generation - previous_generation

    def _refresh_state(self):
        self._active_mode = self._get_active_mode()
        panel_context = self._refresh_panel_context()
        self._has_scene = self._scene_available()
        self._has_selection = self._scene_has_selection()
        self._can_undo = self._undo_available()
        self._can_redo = self._redo_available()
        self._refresh_depth_state()
        return panel_context

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
                # Reference-reset detection runs against the FOCUSED panel's
                # entry only. The key is resolved fresh here
                # because _refresh_depth_state also runs from the setter paths.
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

    # ---- per-panel Size references ------------------------------------

    def _panel_controls_visible(self):
        """The chip and the sync toggle exist only in independent-dual split."""
        return self._split_mode == _INDEPENDENT_DUAL

    def _ref_key(self):
        """Which reference entry the sliders currently address.

        Independent-dual with sync OFF is the only situation with two windows,
        so it is the only one that reads a per-panel entry; everything else --
        single viewport, comparison split, or sync ON -- shares one.
        """
        if self._split_mode == _INDEPENDENT_DUAL and not self._depth_sync:
            return self._focused_panel
        return _PANEL_SHARED

    def _ref_scale(self, table):
        return table.get(self._ref_key(), _DEFAULT_WINDOW_SCALE)

    def _rebase_target_panel(self):
        """Which panel the current draw-commit signal addresses, or None.

        None means "whatever entry the sliders are addressing" and covers the
        case where only the scalar generation moved -- the panel-addressed
        companion is published with it, so a mismatched generation means the
        commit dict is stale and carries no usable panel.
        """
        commit = self._draw_commit_value()
        try:
            commit_generation = int(commit.get("generation", 0) or 0)
            current_generation = int(RuntimeState.depth_window_draw_generation.value)
        except (TypeError, ValueError):
            return None
        if commit_generation != current_generation:
            return None
        return self._draw_commit_panel()

    def _panel_window_scales(self, panel, fallback=None):
        """That panel's own scale_x/scale_y, or a fallback if it cannot be read
        (no panel= support, or not a real panel).

        `fallback` overrides the default "the scales the toolbar is displaying".
        A fresh-baseline caller must pass the FRESHLY-READ native window: the
        cached displayed scales are the pre-transition ones there, and seeding a
        reference from them is exactly the staleness the baseline exists to
        clear.
        """
        getter = getattr(lf.selection, "get_depth_filter_window", None)
        if callable(getter) and panel in (_PANEL_LEFT, _PANEL_RIGHT):
            try:
                _enabled, _near, _far, scale_x, scale_y, _ox, _oy = getter(panel=panel)
                return (
                    _clamp(_parse_float(scale_x, _DEFAULT_WINDOW_SCALE), 0.05, 1.0),
                    _clamp(_parse_float(scale_y, _DEFAULT_WINDOW_SCALE), 0.05, 1.0),
                )
            except Exception:
                pass
        if fallback is not None:
            return fallback
        return self._window_scale, self._window_scale_y

    def _native_window_scales(self):
        """The window the toolbar is displaying RIGHT NOW, read fresh.

        _reconcile_panel_references runs before this refresh's
        _refresh_depth_state, so the cached scales are still the pre-transition
        ones. A resync must not seed from them.
        """
        getter = getattr(lf.selection, "get_depth_filter_window", None)
        if callable(getter):
            try:
                _enabled, _near, _far, scale_x, scale_y, _ox, _oy = getter()
                return (
                    _clamp(_parse_float(scale_x, _DEFAULT_WINDOW_SCALE), 0.05, 1.0),
                    _clamp(_parse_float(scale_y, _DEFAULT_WINDOW_SCALE), 0.05, 1.0),
                )
            except Exception:
                pass
        return self._window_scale, self._window_scale_y

    def _rebase_panel_reference(self, panel):
        """Rebase ONE entry to ITS panel's window.

        The scale the entry rebases to must come from the panel the signal
        names, not from the displayed window -- undoing an R drag while L is
        focused would otherwise stamp L's size onto R's reference.
        """
        current = self._ref_key()
        if current == _PANEL_SHARED:
            # One window: the signal's panel is irrelevant, the shared entry is
            # the only one the sliders can be reading.
            self._ref_scale_x[_PANEL_SHARED] = self._window_scale
            self._ref_scale_y[_PANEL_SHARED] = self._window_scale_y
            return
        key = panel if panel in (_PANEL_LEFT, _PANEL_RIGHT) else current
        if key == current:
            scale_x, scale_y = self._window_scale, self._window_scale_y
        else:
            scale_x, scale_y = self._panel_window_scales(key)
        self._ref_scale_x[key] = scale_x
        self._ref_scale_y[key] = scale_y

    def _seed_all_references(self, scale_x, scale_y):
        """Put every reference entry on the SAME window.

        The recovery for every lineage rule that COLLAPSES to one window: a
        leave collapse and a sync-ON copy each leave exactly one window worth
        referencing, and which entry the sliders read next depends on the
        endpoint (shared while synced or single-window, per-panel otherwise).
        Seeding all three leaves the right answer in whichever one is consulted,
        and cannot leave a stale entry behind for a later transition to promote.

        NOT every producer collapses, though: a project restore and a sync
        undo/redo restore both write two INDEPENDENT absolute windows, so the
        per-panel endpoint must baseline each entry from its own slot instead --
        see _fresh_baseline_references.
        """
        for key in (_PANEL_SHARED, _PANEL_LEFT, _PANEL_RIGHT):
            self._ref_scale_x[key] = scale_x
            self._ref_scale_y[key] = scale_y

    def _fresh_baseline_references(self, scale_x, scale_y):
        """Re-baseline every entry from the windows that EXIST right now.

        `scale_x`/`scale_y` are the freshly-read projection -- the window the
        toolbar is about to display.

        When the endpoint is the two-window one (independent-dual, sync off) the
        slots may legitimately DIFFER after the write that stamped this advance:
        a project restore or a sync undo/redo restores two absolute windows at
        once. Seeding all three entries from the focused projection would leave
        the unfocused panel's reference describing the focused panel's window,
        so focusing it next would report a bogus Size (undoing a sync of
        {L=.60, R=.20} with Right focused would show Left as 300%). Each panel
        entry is therefore baselined from ITS OWN slot, and only the shared
        entry -- the one the sliders read in every OTHER endpoint -- takes the
        projection.

        Anywhere else there is only one window, and every entry takes it.
        """
        if self._split_mode == _INDEPENDENT_DUAL and not self._depth_sync:
            for key in (_PANEL_LEFT, _PANEL_RIGHT):
                panel_x, panel_y = self._panel_window_scales(
                    key, fallback=(scale_x, scale_y)
                )
                self._ref_scale_x[key] = panel_x
                self._ref_scale_y[key] = panel_y
            self._ref_scale_x[_PANEL_SHARED] = scale_x
            self._ref_scale_y[_PANEL_SHARED] = scale_y
            return
        self._seed_all_references(scale_x, scale_y)

    def _reconcile_panel_references(
        self, previous_panel, previous_sync, previous_mode, previous_generation=None
    ):
        """Seed entries across every transition that invalidates a reference.

        Two of them are visible as ENDPOINT changes -- entering and leaving the
        two-window world -- and the rest are visible only through the native
        reference-lineage channel, which is consulted on EVERY refresh whatever
        endpoint this one ends on.

        A plain focus switch does NOTHING here -- it only selects a different
        existing entry, which is the whole point of keying them per panel.

        `previous_panel` is the PRE-transition focus. The split service resets
        focus to Left when leaving independent-dual
        (split_view_service.cpp:214), so reading the post-transition focus would
        collapse Left's reference while the manager collapsed Right's window.
        """
        entering_two = (
            self._split_mode == _INDEPENDENT_DUAL
            and not self._depth_sync
            and (previous_mode != _INDEPENDENT_DUAL or previous_sync)
        )
        leaving_two = (
            (previous_mode == _INDEPENDENT_DUAL and not previous_sync)
            and (self._split_mode != _INDEPENDENT_DUAL or self._depth_sync)
        )
        observed_leave = (
            leaving_two
            and previous_mode == _INDEPENDENT_DUAL
            and self._split_mode != _INDEPENDENT_DUAL
        )
        # EVERY lineage advance is reconciled, whatever endpoint this refresh
        # ends on. The endpoint predicates above see only the two transitions
        # that change how many references EXIST; a hidden leave -> enter cycle,
        # a hidden sync ON/OFF cycle and a project restore can all begin and end
        # in the same endpoint while invalidating every cached reference, and
        # the channel is the only witness to them.
        delta = self._lineage_delta(previous_generation)
        if delta not in (None, 0) or previous_sync != self._depth_sync:
            self._retained_reference_generation = None
            self._gt_baseline_pending = False
        if self._split_mode == _GT_COMPARISON:
            # First observed in GT: neither the dormant pair's user baselines
            # nor a preceding shared reference has been seen by this consumer.
            # A known shared reference from global-origin GT keeps normal seeding.
            if previous_generation is None:
                self._gt_baseline_pending = True
        elif self._split_mode not in ("none", _INDEPENDENT_DUAL):
            self._retained_reference_generation = None
            self._gt_baseline_pending = False
        retained_leave = (
            previous_mode == _INDEPENDENT_DUAL
            and self._split_mode in (_GT_COMPARISON, "none")
            and delta == 0
        )
        if retained_leave:
            # A plain Independent -> Disabled leave stamps a collapse. No stamp
            # means the GT park (possibly its Disabled leg too) was coalesced.
            self._retained_reference_generation = previous_generation
            if previous_sync:
                # The parked synced pair shares one baseline. Keep it in the
                # pair entries while GT/global edits may rebase shared itself.
                for key in (_PANEL_LEFT, _PANEL_RIGHT):
                    self._ref_scale_x[key] = self._ref_scale_x[_PANEL_SHARED]
                    self._ref_scale_y[key] = self._ref_scale_y[_PANEL_SHARED]
        retained_return = (
            self._split_mode == _INDEPENDENT_DUAL
            and previous_mode != _INDEPENDENT_DUAL
            and self._retained_reference_generation is not None
            and self._retained_reference_generation == self._collapse_generation
        )
        baseline_gt_return = (
            self._split_mode == _INDEPENDENT_DUAL
            and previous_mode != _INDEPENDENT_DUAL
            and self._gt_baseline_pending
        )
        if self._split_mode == _INDEPENDENT_DUAL:
            self._retained_reference_generation = None
            self._gt_baseline_pending = False
        # The ONE advance an endpoint transition fully explains: the single
        # leave collapse this refresh actually watched happen. (An unknown kind
        # is an older module reporting only a count; it keeps the plain
        # source-identity reading.) Anything else -- a bigger delta, or a
        # stamp of a kind this endpoint does not account for -- is reconciled
        # below.
        explained = (
            observed_leave
            and delta == 1
            and self._collapse_kind in (None, _LINEAGE_LEAVE_COLLAPSE)
        )
        if delta is not None and delta > 0 and not explained:
            if delta == 1 and self._collapse_kind == _LINEAGE_SYNC_COPY:
                # ONE sync-ON copy, and this refresh did not observe it as a
                # leave. Native copied the recorded source panel's window over
                # the other, so that panel's cached reference is the one that
                # still describes a window that exists -- and it now describes
                # BOTH slots as well as the shared entry.
                source = (
                    self._collapse_source
                    if self._collapse_source in (_PANEL_LEFT, _PANEL_RIGHT)
                    else self._focused_panel
                )
                self._seed_all_references(
                    self._ref_scale_x[source], self._ref_scale_y[source]
                )
            else:
                # Everything else fresh-baselines from the CURRENT native
                # window: a project restore (both slots seeded from the project,
                # so no cached entry means anything), a delta larger than the
                # transitions observed (boundaries went by and last-panel
                # identity names only the final leg), and any leave collapse
                # this refresh did not see as a leave -- the re-entry re-seeded
                # both slots, so the surviving window is the only recoverable
                # reference. Exactly as if a new shape had just been drawn --
                # except that a restore can leave the two slots DIFFERING, which
                # is why the per-panel endpoint baselines each entry from its own
                # slot rather than from the projection alone.
                scale_x, scale_y = self._native_window_scales()
                self._fresh_baseline_references(scale_x, scale_y)
            return
        if retained_return:
            if self._depth_sync:
                self._ref_scale_x[_PANEL_SHARED] = self._ref_scale_x[_PANEL_LEFT]
                self._ref_scale_y[_PANEL_SHARED] = self._ref_scale_y[_PANEL_LEFT]
            return
        if baseline_gt_return:
            self._fresh_baseline_references(*self._native_window_scales())
            return
        if entering_two:
            # Ordinary shared -> independent entry seeds both slots from
            # the global window, so its reference belongs to both as well.
            for key in (_PANEL_LEFT, _PANEL_RIGHT):
                self._ref_scale_x[key] = self._ref_scale_x[_PANEL_SHARED]
                self._ref_scale_y[key] = self._ref_scale_y[_PANEL_SHARED]
        elif leaving_two:
            # Collapsing to one window: the reference that wins must come from
            # the panel whose WINDOW the native side collapsed, and which panel
            # that is depends on WHICH edge collapsed it.
            # Anything the lineage channel could not account for was already
            # reconciled and returned above, so this path is reached only for
            # the ONE observed leave collapse, for a sync edge, or with the
            # channel unexposed.
            if observed_leave:
                # Mode leave. The split service resets focus to Left on the way
                # out (split_view_service.cpp:214), so the post-transition focus
                # is ALWAYS Left and only the PRE-transition panel is meaningful.
                # The CACHED pre-transition panel is a poll behind, though: an
                # external focus change that coalesces with the leave into one
                # refresh leaves the cache naming the panel focused BEFORE that
                # change, while the manager collapsed the one focused after it.
                # The native record of what it actually collapsed is therefore
                # authoritative; the cache is the fallback when the binding does
                # not expose it.
                candidate = self._collapse_source or previous_panel
            else:
                # Sync edge with the mode unchanged. setDepthWindowSync copies
                # the panel focused AT SET TIME
                # (rendering_manager.cpp:1270 reads split_view_service_.
                # focusedPanel()), which is the FRESHLY-READ focus -- the cached
                # one can be a poll behind when an external focus change and the
                # sync edge coalesce into a single refresh.
                candidate = self._focused_panel
            source = (
                candidate
                if candidate in (_PANEL_LEFT, _PANEL_RIGHT)
                else self._focused_panel
            )
            self._ref_scale_x[_PANEL_SHARED] = self._ref_scale_x[source]
            self._ref_scale_y[_PANEL_SHARED] = self._ref_scale_y[source]

    def _draw_commit_value(self):
        value = RuntimeState.depth_window_draw_commit.value
        return value if isinstance(value, dict) else {}

    def _draw_commit_panel(self):
        panel = str(self._draw_commit_value().get("panel", _PANEL_LEFT))
        return panel if panel in (_PANEL_LEFT, _PANEL_RIGHT) else _PANEL_LEFT

    def _draw_commit_items(self):
        commit = self._draw_commit_value()
        return (int(commit.get("generation", 0) or 0), self._draw_commit_panel())

    # ---- chip + sync toggle -------------------------------------------

    def _panel_chip_label(self):
        key, fallback = _PANEL_CHIP_LABELS.get(
            self._focused_panel, _PANEL_CHIP_LABELS[_PANEL_LEFT]
        )
        return _ui_label(key, fallback)

    def _sync_toggle_label(self):
        return _ui_label("toolbar.depth_window_sync", "Sync Panel Depth Windows")

    def _sync_toggle_icon(self):
        if self._depth_sync:
            return _SYNC_ICON_ON
        # Same source of truth as the chip beside it (`_panel_chip_label`), so
        # the two can never disagree about which panel is focused, and the
        # same fallback for a token neither name matches.
        return _SYNC_ICON_OFF.get(self._focused_panel, _SYNC_ICON_OFF[_PANEL_LEFT])

    def _toggle_depth_window_sync(self):
        # Refresh FIRST. `not self._depth_sync` computed from the cache is only
        # a toggle while the cache agrees with the manager; if the flag moved
        # externally (undo/redo from elsewhere, MCP, a project restore) since
        # the last 100ms poll, the click would re-request the value the manager
        # already holds and nothing would toggle. This refresh also reconciles
        # that pending external edge exactly once, before the request.
        self._refresh_panel_context()
        # If that refresh came back EXHAUSTED it consumed nothing, so
        # `self._depth_sync` is still the deliberately stale pre-refresh cache
        # and `not self._depth_sync` is not a toggle -- with the cache false and
        # the manager already true the click would re-request true and the user's
        # toggle would silently vanish into a no-change write. The click is
        # dropped instead, exactly as the manager drops one refused mid-drag:
        # nothing is written, the cache is left untouched, and the button
        # shows the manager's actual state at the next stable poll.
        if self._context_read_exhausted:
            return
        setter = getattr(lf.ui, "set_depth_window_sync", None)
        if callable(setter):
            try:
                setter(not self._depth_sync)
            except Exception as exc:
                self._report_error(
                    str(exc).strip()
                    or _ui_label("selection.update_depth_failed", "Could not update selection depth filter.")
                )
        # The manager refuses changes during an owned drag or parked GT. An
        # actual Disabled sync change discards retention and applies normally;
        # a same-value request preserves it. Re-read the actual flag through
        # the shared refresh rather than assuming it flipped, so references
        # reconcile only the true before/after state, including any discard.
        self._refresh_panel_context()

    # ---- text-edit guard ----------------------------------------------

    def _cancel_foreign_depth_text_edits(self, force=False):
        """Retarget any depth text edit that was started on another panel.

        The focus changes this guards are the CLICKLESS ones -- operator, MCP,
        project restore, mode boundary. A focus change made by clicking cannot
        reach it with an edit still live: the click blurs the field first
        (rml_viewport_overlay.cpp) and the blur commits and ends the edit
        before the focused panel moves, so by the time the poller sees the
        focus edge there is nothing to retarget.

        `force` retargets EVERY live edit, including one whose recorded panel
        equals the panel now on screen. A mode boundary is what needs it: the
        edit's context is the single global window collapsed from a panel the
        poller never saw focused, so panel equality proves nothing there and the
        buffer is stale-origin text regardless of the name it carries.

        The field is REVERTED in place: the buffer is replaced with the panel
        now on screen's canonical text and the key dirtied, so the data-value
        binding pushes that text back into the retained input (RmlUi 6.2's
        default data view writes the `value` attribute and the text input
        applies it with no focused-input exception). Any later commit -- blur
        or linebreak -- therefore writes the canonical value, which is
        idempotent, or whatever the user types NEXT, which is legitimate. The
        field may still be DOM-focused (a focus change delivered by the
        operator, MCP or a project restore sends no pointer input, so no new
        focus event ever arrives), which is exactly why the field has to be
        safe to keep using rather than merely fenced off.

        The edit is RETARGETED, not dropped. A key still in
        `_editing_depth_text` is by definition one for which no blur has been
        observed -- `_end_depth_text_edit` is the blur handler and the only
        thing that clears the key -- so the field is still DOM-focused and the
        edit is still live. Re-registering it against the panel now on screen
        is what keeps the in-edit skip in `_sync_depth_text_bufs` protecting
        continued typing from the 100ms poll
        (rml_viewport_overlay.hpp:215), and what makes a SECOND focused-panel
        change run this guard again.
        """
        for key in list(self._editing_depth_text):
            if not force and self._depth_text_edit_panel.get(key) == self._focused_panel:
                continue
            self._depth_text_edit_panel[key] = self._focused_panel
            self._depth_text_bufs[key] = self._canonical_depth_text_value(key)
            # The escape snapshot still holds the OLD panel's pre-edit text;
            # re-capture so an Escape after the switch reverts to the panel now
            # on screen.
            self._escape_revert.recapture(key)
            if self._handle:
                self._handle.dirty(key)
        # NOT forced: the retargeted keys stay live edits and have just been set
        # to canonical explicitly above, so forcing here would only risk
        # overwriting a field that is legitimately being typed into.
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
            # Panel-addressed companion. Compared as a tuple so a commit on the
            # SAME panel at a new generation still registers as a change.
            ("depth_window_draw_commit", self._draw_commit_items()),
            ("offset_x", round(self._offset_x, 4)),
            ("offset_y", round(self._offset_y, 4)),
            ("viz_mode", int(self._viz_mode)),
            # Polling fix: the tuple above carries VALUES only, so a
            # focus change between two identical windows -- the default case
            # right after entering split view -- would never dirty the chip.
            ("focused_panel", self._focused_panel),
            ("split_mode", self._split_mode),
            ("depth_sync", self._depth_sync),
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
        # Re-focusing a field whose earlier blur is still deferred SUPERSEDES
        # that pending intent. The stale record carries a frozen payload from an
        # edit session the user has already moved past; leaving it in place would
        # have the flush write that old value and then retire the edit the user
        # is in the middle of, so the very next poll would canonicalize their new
        # keystrokes away. The latest intent wins and the new edit starts clean.
        record = self._deferred_depth_commits.get(key)
        if record is not None and record.get("kind") == _DEFERRED_BLURRED:
            self._deferred_depth_commits.pop(key, None)
        self._editing_depth_text.add(key)
        self._depth_text_edit_panel[key] = self._focused_panel

    def _end_depth_text_edit(self, key):
        # A blur whose commit DEFERRED must not lose the user's value. The widget
        # calls commit then on_blur unconditionally (rml_widgets.bind_committed_
        # text_input), so simply forgetting the key here would leave the deferred
        # value -- the typed number, or the text an Escape revert put back -- with
        # nothing to retry it, while native state keeps whatever the previous
        # commit wrote.
        #
        # The value is preserved by FREEZING it into the deferral record rather
        # than by holding the edit open. Holding it open instead is wrong: the
        # key would stay in _editing_depth_text, so a later focus or mode
        # transition would treat a COMPLETED blur as a live edit and
        # _cancel_foreign_depth_text_edits would replace both its payload and
        # its target panel -- the flush would then write the new panel's
        # canonical value to the new panel and the finished edit would be lost.
        # A completed blur is not live and is not retargetable, so the
        # live-edit bookkeeping is retired here exactly as it is for an
        # undeferred blur, and the record carries the payload and the origin
        # panel instead.
        record = self._deferred_depth_commits.get(key)
        if record is not None:
            record["kind"] = _DEFERRED_BLURRED
            record["payload"] = self._depth_text_bufs.get(key)
            record["panel"] = self._depth_text_edit_panel.get(key)
        self._editing_depth_text.discard(key)
        self._depth_text_edit_panel.pop(key, None)

    def _flush_deferred_depth_commits(self):
        """Retry commits deferred by an exhausted context read, on a stable tick.

        Called from update() with the poll's own refresh already done, so the
        exhausted flag below is this tick's.

        A LIVE record re-validates through _commit_depth_text_key exactly as the
        original commit did -- reading the buffer as it stands now, because the
        user is still in the field. A BLURRED record bypasses that path entirely
        and writes its frozen payload to its frozen panel: there is no live field
        to re-read and no transition that may redirect a finished edit.

        Every key that stays deferred simply stays deferred; the next stable poll
        tries again. Keys are flushed one at a time, and _sync_depth_text_bufs
        never canonicalizes a key that still has a record, so a commit landing
        for one key cannot erase the buffer another key's flush is about to read.

        Near and Far are the one CLAMP-COUPLED pair in this registry: each write
        clamps against the counterpart as it stands at that moment (the live
        path's _clamp against self._depth_far / self._depth_near, and the same
        clamps in _apply_foreign_panel_depth_values). Scale and the two offsets
        are clamped only against constants and against state each of their
        setters re-reads fresh, so they carry no such coupling.

        No REPLAY ORDER is safe for that pair. An insertion-order law fixes
        the plain two-blur case but not a superseded and re-armed one:
        blur-defer Near .20, blur-defer Far .50, refocus Near (which supersedes
        and drops its record), re-blur Near .20 (which reinserts it at the tail).
        The insertion order is now [Far, Near] and chronologically defensible,
        yet replaying Far first clamps it up against the still-native Near .90 to
        .91 and the user's .50 is destroyed.

        So order-sensitivity is replaced by PAIRED RESOLUTION. Records are
        grouped by the window their write will land in (_deferred_write_target --
        the same target the individual writes use). Per target the FINAL intended
        (near, far) is resolved BEFORE anything is written: each field takes its
        latest deferred payload -- a live record still reads its buffer at flush
        time, unchanged -- and a field with no record takes that target's current
        native value. The pair is then written as ONE combined update through
        set_depth_filter_window, which carries both, so the clamp evaluates the
        user's final intent against itself and never against a stale counterpart.
        A target with only one of the two keeps the existing single write, whose
        native counterpart is by definition not stale.

        Supersession stays per-key: the fresh .get(key) below re-checks each key
        against the CURRENT registry, and the list() snapshot keeps a flush that
        pops records from mutating the sequence being walked.

        The grouping is computed from the context as it stands when the flush
        starts, and a LIVE member revalidates that context again from inside the
        pair (_commit_paired_depth_range). That inner read can MOVE the context
        -- a focus change the poll had not observed retargets the live edit --
        and once it has, the target the grouping computed no longer describes
        where these records belong: a frozen Near still aimed at Left would ride
        a no-panel write into Right, contaminating the panel the user never
        touched and consuming a record with a write that was not its own. So the
        pair reports RETARGETED instead of writing, and the whole grouping is
        RE-DERIVED from the moved context. The restart is bounded: it re-reads
        everything once, and if the context moves a second time the affected
        records are simply left deferred for the next poll rather than spun on.

        UNPAIRED records join that same protocol -- a lone live record
        revalidates from inside its own commit and can be retargeted by exactly
        the same read -- so _commit_deferred_depth_record reports the same three
        outcomes and this loop treats them identically whatever produced them.

        The three outcomes are dispatched EXHAUSTIVELY, with no path that
        continues writing on an outcome nobody matched:

        * DONE -- resolved; carry on with the group's remaining records and the
          groups after it.
        * RETARGETED -- regroup once against the context that actually holds.
        * DEFERRED -- a revalidating read came back EXHAUSTED. The whole flush
          STOPS THERE, immediately. An exhausted read is a statement about the
          world, not about one record: no sibling in this group and no later
          group may write or be consumed on this tick, and everything still
          pending waits for the next stable poll.
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
                # The dispatch below is STRUCTURALLY EXHAUSTIVE: one explicit
                # branch per outcome constant and a defensive tail that defers.
                # An outcome that is not recognised must behave like the safest
                # constant, never fall through into "keep writing".
                if outcome == _COMMIT_DONE:
                    continue
                elif outcome == _COMMIT_RETARGETED:
                    retargeted = True
                    break
                elif outcome == _COMMIT_DEFERRED:
                    # The context read came back EXHAUSTED, so the world this
                    # flush is writing into does not exist as a consistent
                    # snapshot. Stop the ENTIRE flush here: no sibling record
                    # in this group or any later one may write or be consumed
                    # on this tick, and everything still pending stays
                    # deferred for the next stable poll to resolve together.
                    return
                else:
                    # Defensive tail: an unrecognised outcome must behave like
                    # the safest constant (DEFERRED), never fall through into
                    # "keep writing".
                    return
            if not retargeted or restarted:
                return
            restarted = True

    def _group_deferred_depth_records(self):
        """The pending records grouped by the window their write will land in.

        Returns a list of (target, {key: record}) in first-seen target order.
        Records already consumed by an earlier group's write are gone from the
        registry, so re-deriving this after a retarget resumes exactly where the
        flush left off.
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
        """The window a deferred record's write will land in.

        A blurred record carries its frozen origin panel and reaches it through
        the panel= setter overload only when a no-panel write would MISS it.
        Otherwise -- and always for a live record, which is still aimed at the
        context the user is editing in -- the write goes to the current context,
        which is the target named None here.
        """
        if record.get("kind") == _DEFERRED_BLURRED:
            panel = record.get("panel")
            if self._panel_addressed_write_needed(panel):
                return panel
        return None

    def _deferred_write_destination(self, record):
        """The WINDOW a deferred record's write actually lands in.

        _deferred_write_target names the OVERLOAD a write needs -- a panel when
        the ordinary setters would miss the frozen origin, and None when they
        reach it -- which is the right key to group by but is not an identity.
        None does not name a window; it names "wherever the context points",
        so it compares EQUAL to itself across a move that sends the write
        somewhere else entirely. Resolving None to the window the ordinary
        setters currently address (_ref_key: the focused panel when two
        independent windows exist, the single shared window otherwise) gives a
        value that changes exactly when the destination changes, which is what a
        comparison across a revalidation has to detect.
        """
        target = self._deferred_write_target(record)
        if target is not None:
            return target
        return self._ref_key()

    def _commit_deferred_depth_record(self, key, record):
        """Flush ONE deferred record through the path its kind requires.

        Returns the same _COMMIT_* outcome the paired path returns, because an
        UNPAIRED record is subject to the identical hazard and joins the
        identical regroup protocol. A live record revalidates the context from
        inside its commit, that read is allowed to MOVE the context, and the
        write that follows would then land in a window this record was never
        derived for. So the destination and the record's identity in the
        registry are captured BEFORE the revalidating read and re-checked after
        it: on a move nothing is written, nothing is consumed, and
        _COMMIT_RETARGETED sends the flush back to regroup against the context
        that actually holds -- once. A second move leaves the record deferred
        for the next poll, exactly as the paired path does.

        A BLURRED record carries its own frozen payload and origin panel and
        reads no context, so it cannot be retargeted and always reports DONE.
        """
        if record.get("kind") == _DEFERRED_BLURRED:
            self._commit_blurred_depth_record(key, record)
            return _COMMIT_DONE
        if key in self._editing_depth_text:
            destination = self._deferred_write_destination(record)
            self._refresh_panel_context()
            if self._context_read_exhausted:
                # Same torn-read law as everywhere else: the record stays
                # recorded and untouched, and the flush stops on DEFERRED.
                if key not in self._deferred_depth_commits:
                    self._deferred_depth_commits[key] = {"kind": _DEFERRED_LIVE}
                return _COMMIT_DEFERRED
            if self._deferred_depth_commits.get(key) is not record:
                return _COMMIT_RETARGETED
            if self._deferred_write_destination(record) != destination:
                return _COMMIT_RETARGETED
            # The revalidation is done and its verdict is in, so the commit
            # below must not read the context a second time -- another read
            # could move it again after this check and write against a target
            # nothing verified.
            self._commit_depth_text_key(key, revalidated=True)
            return _COMMIT_DONE
        self._commit_depth_text_key(key)
        return _COMMIT_DONE

    def _commit_paired_depth_range(self, panel, near_record, far_record):
        """Write one target's final intended (near, far) as a single update.

        Both records are consumed whatever happens, exactly as the single-field
        paths consume theirs: this is the retry of commits the user already
        made, and the ordinary refusals (unparseable text, a hidden panel) are
        no-ops, not reasons to retry forever. The one exception is a live member
        whose revalidating context read comes back EXHAUSTED -- that is the same
        tearing the original commit deferred against, so the whole pair stays
        deferred and the next stable poll resolves it together.

        A live member's revalidation is also allowed to MOVE the context, which
        invalidates the grouping that produced `panel`. Every payload is
        therefore resolved first and the identities and targets re-derived
        afterwards: if either record has been replaced in the registry, or
        either one's target no longer equals the one this pair was grouped for,
        nothing is written, nothing is consumed and _COMMIT_RETARGETED sends the
        flush back to regroup against the context that actually holds now.
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
                    self._refresh_panel_context()
                    revalidated = True
                    if self._context_read_exhausted:
                        return _COMMIT_DEFERRED
                payload = self._depth_text_bufs.get(key)
            resolved[key] = self._parse_depth_text_value(key, payload)
        if revalidated:
            for key, record in pairing:
                if self._deferred_depth_commits.get(key) is not record:
                    return _COMMIT_RETARGETED
                if self._deferred_write_target(record) != panel:
                    return _COMMIT_RETARGETED
        self._deferred_depth_commits.pop("selection_depth_near_str", None)
        self._deferred_depth_commits.pop("selection_depth_far_str", None)
        values = {key: value for key, value in resolved.items() if value is not None}
        if not values or not self._visible or self._last_state_key is None:
            self._sync_depth_text_bufs(force=True)
            return _COMMIT_DONE
        # Same arming rationale as every other commit: applying dirties every
        # slider-bound value and RmlUi replays each slider's pre-commit position
        # into its setter.
        self._depth_echo_holdoff = 2
        if panel is None:
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
        else:
            self._apply_foreign_panel_depth_values(values, panel)
        self._sync_depth_text_bufs(force=True)
        return _COMMIT_DONE

    def _commit_blurred_depth_record(self, key, record):
        """Write a COMPLETED blur's frozen payload to its frozen origin panel.

        The record is consumed whatever happens: this is the retry of a commit
        the user already finished, and the ordinary refusals (unparseable text,
        a hidden panel) are the same no-ops the live path applies -- they are not
        reasons to keep retrying forever.
        """
        self._deferred_depth_commits.pop(key, None)
        parsed = self._parse_depth_text_value(key, record.get("payload"))
        if parsed is None or not self._visible or self._last_state_key is None:
            self._sync_depth_text_bufs(force=True)
            return
        # Same arming rationale as the live commit: applying dirties every
        # slider-bound value and RmlUi replays each slider's pre-commit position
        # into its setter.
        self._depth_echo_holdoff = 2
        panel = record.get("panel")
        if self._panel_addressed_write_needed(panel):
            self._apply_foreign_panel_depth_values({key: parsed}, panel)
        else:
            # The frozen panel IS the one a no-panel write reaches now (or there
            # are no separate panel windows at all), so the ordinary setters
            # address it and the displayed state follows the write as usual.
            self._dispatch_depth_setter(key, parsed)
        self._sync_depth_text_bufs(force=True)

    def _panel_addressed_write_needed(self, panel):
        """True when the frozen panel is a real panel that a no-panel write would
        MISS -- i.e. two independent windows exist and the other one is on."""
        return (
            panel in (_PANEL_LEFT, _PANEL_RIGHT)
            and self._split_mode == _INDEPENDENT_DUAL
            and not self._depth_sync
            and panel != self._focused_panel
        )

    def _apply_foreign_panel_depth_values(self, values, panel):
        """Write one or more fields into a NON-focused panel's own window.

        The panel= setter overload exists precisely for this (py_selection.cpp,
        set_depth_filter_window): it writes that panel's slot and leaves the
        displayed window alone. The rest of the window has to come from that
        panel's OWN state, not the toolbar's cached values, so it is read back
        through the panel= getter; the Size reference likewise comes from that
        panel's reference entry.

        `values` maps field key -> parsed number, and a key that is absent keeps
        the panel's current native value. Near is applied before Far so that a
        PAIRED (near, far) update clamps Far against the user's own intended
        Near rather than the native one it is about to replace; with only one of
        the two present this is the single write it always was.
        """
        getter = getattr(lf.selection, "get_depth_filter_window", None)
        setter = getattr(lf.selection, "set_depth_filter_window", None)
        if not callable(getter) or not callable(setter):
            return
        try:
            enabled, near, far, scale_x, scale_y, offset_x, offset_y = getter(panel=panel)
        except Exception:
            return
        enabled = bool(enabled)
        near = _clamp(_parse_float(near, self._depth_near), _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
        far = _clamp(_parse_float(far, self._depth_far), near + _DEPTH_GAP, _DEPTH_MAX)
        scale_x = _clamp(_parse_float(scale_x, _DEFAULT_WINDOW_SCALE), 0.05, 1.0)
        scale_y = _clamp(_parse_float(scale_y, _DEFAULT_WINDOW_SCALE), 0.05, 1.0)
        offset_x = _clamp(_parse_float(offset_x, 0.0), -1.0, 1.0)
        offset_y = _clamp(_parse_float(offset_y, 0.0), -1.0, 1.0)
        if "selection_depth_near_str" in values:
            near = _clamp(values["selection_depth_near_str"], _DEPTH_MIN, _DEPTH_MAX - _DEPTH_GAP)
            far = max(far, near + _DEPTH_GAP)
        if "selection_depth_far_str" in values:
            far = _clamp(values["selection_depth_far_str"], near + _DEPTH_GAP, _DEPTH_MAX)
        if "selection_depth_scale_str" in values:
            parsed = values["selection_depth_scale_str"]
            ref_x = max(self._ref_scale_x.get(panel, _DEFAULT_WINDOW_SCALE), 1.0e-6)
            ref_y = max(self._ref_scale_y.get(panel, _DEFAULT_WINDOW_SCALE), 1.0e-6)
            f_min = max(0.05 / ref_x, 0.05 / ref_y)
            f_max = max(f_min, min(1.0 / ref_x, 1.0 / ref_y))
            percent = _clamp(parsed, _SCALE_PERCENT_MIN, _SCALE_PERCENT_MAX)
            factor = _clamp(percent / 100.0, f_min, f_max)
            scale_x = _clamp(ref_x * factor, 0.05, 1.0)
            scale_y = _clamp(ref_y * factor, 0.05, 1.0)
        if "selection_depth_offset_x_str" in values:
            offset_x = _clamp(values["selection_depth_offset_x_str"], _OFFSET_PERCENT_MIN, _OFFSET_PERCENT_MAX) / 100.0
        if "selection_depth_offset_y_str" in values:
            offset_y = _clamp(values["selection_depth_offset_y_str"], _OFFSET_PERCENT_MIN, _OFFSET_PERCENT_MAX) / 100.0
        try:
            setter(enabled, near, far, scale_x, offset_x, offset_y, scale_y, panel=panel)
        except Exception as exc:
            self._report_error(
                str(exc).strip()
                or _ui_label("selection.update_depth_failed", "Could not update selection depth filter.")
            )

    def _capture_depth_text_snapshot(self, key):
        return self._canonical_depth_text_value(key)

    def _restore_depth_text_snapshot(self, key, snapshot):
        # Escape restores the pre-edit text and the host blurs immediately after
        # (cancelFocusedElement in rml_input_utils.hpp), so that blur is what
        # carries the revert to the native side. Both hosts that own these
        # fields dispatch escapecancel that way: the sidebar/docked panels
        # (rml_panel_host.cpp) and the viewport overlay
        # (rml_viewport_overlay.cpp), which is where the depth Near/Far/Size
        # inputs live.
        self._depth_text_bufs[key] = str(snapshot or "")
        if self._handle:
            self._handle.dirty(key)

    def _commit_depth_text_key(self, key, revalidated=False):
        # No commit is ever swallowed. A focused-panel change reverts the field
        # in place (_cancel_foreign_depth_text_edits), so whatever a later
        # commit carries is either that canonical value -- an idempotent
        # no-change write -- or text the user typed since, which is legitimate.
        # The old pre-cancel text cannot survive the revert: RmlUi 6.2's data
        # view writes the input's value even while it is focused, and the
        # plugin's model hook runs before Context::Update() in the same render
        # (rml_viewport_overlay.cpp:1124).
        #
        # The recorded edit origin is nevertheless validated against a FRESH
        # context read here, because a commit can arrive before any poll has
        # observed a focus change (Enter or blur immediately after an operator,
        # MCP or project-restore focus switch). The shared refresh applies the
        # retarget rule on a mismatch -- the buffer becomes the new panel's
        # canonical text -- so the write below lands on the panel the user is
        # actually editing toward, never a stale-origin write into the wrong
        # panel. Keys with no live edit have no recorded origin to validate, and
        # are left alone so nothing rewrites a buffer this commit is about to
        # read.
        #
        # That validation compares the split MODE and the collapse GENERATION as
        # well as the focus, because the shared refresh does: a commit arriving
        # after a coalesced focus-change + mode-leave sees equal focus on both
        # sides of the transition, and only the mode/generation edge exposes
        # that the field now addresses a different window.
        #
        # If that validating refresh comes back EXHAUSTED it observed nothing it
        # is allowed to consume, so this commit has no validated context to write
        # against. Writing anyway would apply the user's value to whichever
        # window the torn snapshot happened to name -- and re-canonicalizing the
        # buffer would replace their keystrokes with text computed from an
        # unreconciled reference. So the commit is DEFERRED, not dropped: the
        # buffer and the live-edit state are both retained untouched, and the
        # next commit -- or the flush on the first poll whose own revalidation
        # gets a COMPARABLE (equal-generation) pair of record reads, which is not
        # necessarily the next poll and not guaranteed by quiet native state
        # alone -- resolves it against a consistent world. Nothing typed is lost;
        # only the write is postponed, for as long as the tearing lasts.
        #
        # The deferral is RECORDED, not merely returned from: a blur arriving in
        # the same storm would otherwise leave nothing to retry
        # (rml_widgets.py calls commit then on_blur unconditionally).
        # _end_depth_text_edit freezes the record's payload and origin panel;
        # _flush_deferred_depth_commits retries on a stable tick.
        #
        # `revalidated` is set ONLY by the deferred-flush wrapper, which has
        # just done this read itself and acted on its verdict. Reading again
        # here would let the context move a second time, after that verdict and
        # before the write -- the very hazard the wrapper exists to close.
        if not revalidated and key in self._editing_depth_text:
            self._refresh_panel_context()
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
            # A key with a deferral record still OWES its value to native state,
            # and canonicalizing it here would destroy exactly that: a live
            # record's buffer is what its flush will read, and a blurred record's
            # field must not be rewritten before its frozen payload lands. This
            # skip holds even under force=, because force= is the commit path
            # canonicalizing the key it has just written -- and that key's record
            # is popped before this runs, so the key being written is never
            # skipped, only its still-pending siblings. Without this, flushing
            # two deferred keys wrote the first and canonicalized the second.
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
        elif action == "toggle_sync":
            self._toggle_depth_window_sync()
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
            "focused_panel": (
                "selection_panel_chip_label",
                # The OFF frame of the sync toggle names the focused panel, so
                # focus moves it exactly as it moves the chip's letter.
                "selection_depth_sync_icon",
                "selection_depth_scale_str",
                "selection_depth_scale_value",
                "selection_depth_scale_slider_min",
                "selection_depth_scale_slider_max",
            ),
            "split_mode": (
                "selection_panel_chip_visible",
                "selection_panel_chip_label",
                "selection_depth_sync_active",
                "selection_depth_sync_icon",
            ),
            "depth_sync": (
                "selection_depth_sync_active",
                "selection_depth_sync_icon",
                "selection_depth_scale_str",
                "selection_depth_scale_value",
                "selection_depth_scale_slider_min",
                "selection_depth_scale_slider_max",
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
