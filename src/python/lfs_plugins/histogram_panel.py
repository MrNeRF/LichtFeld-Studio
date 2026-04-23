# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Docked retained panel for Gaussian histogram analysis."""

from __future__ import annotations

import math
from typing import Iterable

import lichtfeld as lf

from .histogram_support import METRICS, METRIC_BY_ID, histogram_mode_available, histogram_tr
from .rml_keys import KI_A, KI_DELETE, KI_I
from .types import Panel
from .ui.state import AppState


DEFAULT_HISTOGRAM_BIN_COUNT = 56
MIN_HISTOGRAM_BIN_COUNT = 16
MAX_HISTOGRAM_BIN_COUNT = 128
DEFAULT_COMPARE_X_BIN_COUNT = 20
DEFAULT_COMPARE_Y_BIN_COUNT = 20
MIN_COMPARE_BIN_COUNT = 8
MAX_COMPARE_BIN_COUNT = 48
HISTOGRAM_BAR_GAP = 2.0
RML_KM_CTRL = 1 << 0
RML_KM_SHIFT = 1 << 1
RML_KM_ALT = 1 << 2
RML_KM_META = 1 << 3


def _tr(key: str, fallback: str) -> str:
    return histogram_tr(key, fallback)


def _trf(key: str, fallback: str, **kwargs) -> str:
    return _tr(key, fallback).format(**kwargs)


class HistogramPanel(Panel):
    id = "lfs.histogram"
    label = "Histogram"
    space = lf.ui.PanelSpace.BOTTOM_DOCK
    order = 97
    template = "rmlui/histogram_panel.rml"
    size = (860, 660)
    height_mode = lf.ui.PanelHeightMode.FILL
    update_interval_ms = 250

    def __init__(self):
        self._doc = None
        self._chart_el = None
        self._compare_chart_el = None
        self._handle = None
        self._panel_space = lf.ui.PanelSpace.BOTTOM_DOCK
        self._is_floating = False

        self._metric_id = METRICS[0].id
        self._compare_metric_id = ""
        self._log_scale_enabled = False
        self._histogram_bin_count = DEFAULT_HISTOGRAM_BIN_COUNT
        self._compare_x_bin_count = DEFAULT_COMPARE_X_BIN_COUNT
        self._compare_y_bin_count = DEFAULT_COMPARE_Y_BIN_COUNT
        self._scene_generation = -1
        self._history_generation = -1
        self._last_lang = ""
        self._trainer_state = ""
        self._show_chart = False
        self._show_compare_card = False
        self._show_compare_chart = False
        self._empty_title = _tr("histogram.empty.unavailable.title", "Histogram unavailable")
        self._empty_message = _tr(
            "histogram.empty.unavailable.message",
            "Switch to a view or edit scene with Gaussian splats to inspect a distribution.",
        )
        self._compare_empty_title = _tr("histogram.compare.empty.title", "Compare unavailable")
        self._compare_empty_message = _tr(
            "histogram.compare.empty.message",
            "Choose a second metric to see how two Gaussian properties cluster together.",
        )
        self._sample_count = "--"
        self._range_text = "--"
        self._mean_text = "--"
        self._median_text = "--"
        self._p95_text = "--"
        self._peak_text = "--"
        self._axis_min = "--"
        self._axis_max = "--"
        self._summary_text = ""
        self._compare_summary_text = ""
        self._compare_x_metric_label = ""
        self._compare_y_metric_label = ""
        self._compare_x_axis_min = "--"
        self._compare_x_axis_max = "--"
        self._compare_y_axis_min = "--"
        self._compare_y_axis_max = "--"
        self._primary_values: lf.Tensor | None = None
        self._primary_finite_mask: lf.Tensor | None = None
        self._primary_valid_values: lf.Tensor | None = None
        self._primary_histogram_min = 0.0
        self._primary_histogram_max = 1.0
        self._compare_values: lf.Tensor | None = None
        self._compare_finite_mask: lf.Tensor | None = None
        self._compare_valid_x_values: lf.Tensor | None = None
        self._compare_valid_y_values: lf.Tensor | None = None
        self._compare_x_min = 0.0
        self._compare_x_max = 1.0
        self._compare_y_min = 0.0
        self._compare_y_max = 1.0

        self._selection_bin_indices: lf.Tensor | None = None
        self._hist_counts: list[int] | None = None
        self._hist_prefix_counts: list[int] | None = None
        self._hist_edges: list[float] | None = None
        self._compare_x_bin_indices: lf.Tensor | None = None
        self._compare_y_bin_indices: lf.Tensor | None = None
        self._compare_counts: list[int] | None = None
        self._compare_x_edges: list[float] | None = None
        self._compare_y_edges: list[float] | None = None
        self._selection_owned = False
        self._pending_selection_commit = 0
        self._active_mark_source: str | None = None
        self._panel_selection_mask: lf.Tensor | None = None
        self._selected_histogram_bins: set[int] = set()
        self._selected_compare_cells: set[tuple[int, int]] = set()
        self._histogram_overlay_bounds: tuple[int, int] | None = None
        self._compare_overlay_bounds: tuple[int, int, int, int] | None = None

        self._dragging_mark = False
        self._drag_selection_mode = "replace"
        self._drag_selection_base_mask: lf.Tensor | None = None
        self._marked_bin_start: int | None = None
        self._marked_bin_end: int | None = None
        self._marked_value_min: float | None = None
        self._marked_value_max: float | None = None
        self._dragging_compare_mark = False
        self._drag_compare_selection_mode = "replace"
        self._drag_compare_selection_base_mask: lf.Tensor | None = None
        self._compare_mark_start: tuple[int, int] | None = None
        self._compare_mark_end: tuple[int, int] | None = None
        self._compare_mark_x_min: float | None = None
        self._compare_mark_x_max: float | None = None
        self._compare_mark_y_min: float | None = None
        self._compare_mark_y_max: float | None = None
        self._marked_count = 0
        self._marked_range_text = _tr("histogram.no_marked_range", "No marked range")
        self._marked_count_text = _trf("histogram.gaussian_count", "{count} Gaussians", count="0")
        self._status_hint = _tr(
            "histogram.status_drag_delete",
            "Left-drag across the histogram to mark a range, then delete it.",
        )
        self._selection_left_style = "0%"
        self._selection_width_style = "0%"
        self._compare_selection_left_style = "0%"
        self._compare_selection_top_style = "0%"
        self._compare_selection_width_style = "0%"
        self._compare_selection_height_style = "0%"

    @classmethod
    def poll(cls, context):
        return histogram_mode_available(context)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("histogram_panel")
        if model is None:
            return

        model.bind_func("panel_label", lambda: _tr("window.histogram", "Histogram"))
        model.bind_func("analysis_label", lambda: _tr("histogram.title_eyebrow", "Gaussian Analysis"))
        model.bind_func("field_label", lambda: _tr("histogram.field", "Field"))
        model.bind_func("compare_with_label", lambda: _tr("histogram.compare_with", "Compare With"))
        model.bind_func("log_scale_label", lambda: _tr("histogram.log_scale", "Log Scale"))
        model.bind_func("close_label", lambda: _tr("common.close", "Close"))
        model.bind_func("is_floating", lambda: self._is_floating)
        model.bind_func("dock_toggle_label", self._dock_toggle_label)
        model.bind_func("samples_label", lambda: _tr("histogram.samples", "Samples"))
        model.bind_func("range_label", lambda: _tr("histogram.range", "Range"))
        model.bind_func("mean_label", lambda: _tr("histogram.mean", "Mean"))
        model.bind_func("median_label", lambda: _tr("histogram.median", "Median"))
        model.bind_func("p95_label", lambda: _tr("histogram.p95", "P95"))
        model.bind_func("peak_bin_label", lambda: _tr("histogram.peak_bin", "Peak Bin"))
        model.bind_func(
            "bin_count_text",
            lambda: _trf("histogram.bin_count", "{count} bins", count=self._histogram_bin_count),
        )
        model.bind_func("histogram_bins_label", lambda: _tr("histogram.bins", "Bins"))
        model.bind_func("compare_x_bins_label", lambda: _tr("histogram.compare_x_bins", "X Bins"))
        model.bind_func("compare_y_bins_label", lambda: _tr("histogram.compare_y_bins", "Y Bins"))
        model.bind_func("histogram_bar_gap_style", lambda: f"{HISTOGRAM_BAR_GAP:.2f}px")
        model.bind_func("marked_label", lambda: _tr("histogram.marked", "Marked"))
        model.bind_func("count_label", lambda: _tr("histogram.count", "Count"))
        model.bind_func("undo_tooltip", self._undo_tooltip)
        model.bind_func("redo_tooltip", self._redo_tooltip)
        model.bind_func("clear_label", lambda: _tr("histogram.clear", "Clear"))
        model.bind_func("delete_label", lambda: _tr("histogram.delete", "Delete"))
        model.bind_func("metric_title", lambda: METRIC_BY_ID[self._metric_id].label())
        model.bind_func("metric_description", lambda: METRIC_BY_ID[self._metric_id].description())
        model.bind_func("show_chart", lambda: self._show_chart)
        model.bind_func("show_empty_state", lambda: not self._show_chart)
        model.bind_func("empty_title", lambda: self._empty_title)
        model.bind_func("empty_message", lambda: self._empty_message)
        model.bind_func("sample_count", lambda: self._sample_count)
        model.bind_func("range_text", lambda: self._range_text)
        model.bind_func("mean_text", lambda: self._mean_text)
        model.bind_func("median_text", lambda: self._median_text)
        model.bind_func("p95_text", lambda: self._p95_text)
        model.bind_func("peak_text", lambda: self._peak_text)
        model.bind_func("axis_min", lambda: self._axis_min)
        model.bind_func("axis_max", lambda: self._axis_max)
        model.bind_func("summary_text", lambda: self._summary_text)
        model.bind_func("show_histogram_card", lambda: self._show_chart and not self._show_compare_card)
        model.bind_func("show_histogram_bins_control", lambda: not bool(self._compare_metric_id))
        model.bind_func("show_compare_bins_controls", lambda: bool(self._compare_metric_id))
        model.bind_func("show_compare_card", lambda: self._show_compare_card)
        model.bind_func("show_compare_chart", lambda: self._show_compare_chart)
        model.bind_func("show_compare_empty_state", lambda: self._show_compare_card and not self._show_compare_chart)
        model.bind_func("compare_empty_title", lambda: self._compare_empty_title)
        model.bind_func("compare_empty_message", lambda: self._compare_empty_message)
        model.bind_func("compare_summary_text", lambda: self._compare_summary_text)
        model.bind_func(
            "compare_bin_count_text",
            self._format_compare_bin_count_text,
        )
        model.bind_func("compare_x_metric_label", lambda: self._compare_x_metric_label)
        model.bind_func("compare_y_metric_label", lambda: self._compare_y_metric_label)
        model.bind_func("compare_x_axis_min", lambda: self._compare_x_axis_min)
        model.bind_func("compare_x_axis_max", lambda: self._compare_x_axis_max)
        model.bind_func("compare_y_axis_min", lambda: self._compare_y_axis_min)
        model.bind_func("compare_y_axis_max", lambda: self._compare_y_axis_max)
        model.bind_func("show_selection_overlay", lambda: self._histogram_overlay_bounds is not None)
        model.bind_func("selection_left_style", lambda: self._selection_left_style)
        model.bind_func("selection_width_style", lambda: self._selection_width_style)
        model.bind_func("show_compare_selection_overlay", lambda: self._compare_overlay_bounds is not None)
        model.bind_func("compare_selection_left_style", lambda: self._compare_selection_left_style)
        model.bind_func("compare_selection_top_style", lambda: self._compare_selection_top_style)
        model.bind_func("compare_selection_width_style", lambda: self._compare_selection_width_style)
        model.bind_func("compare_selection_height_style", lambda: self._compare_selection_height_style)
        model.bind_func("marked_range_text", lambda: self._marked_range_text)
        model.bind_func("marked_count_text", lambda: self._marked_count_text)
        model.bind_func("status_hint", lambda: self._status_hint)
        model.bind_func("undo_enabled", self._can_undo)
        model.bind_func("redo_enabled", self._can_redo)
        model.bind_func("clear_enabled", self._has_any_mark)
        model.bind_func("delete_enabled", lambda: self._has_any_mark() and self._marked_count > 0)
        model.bind("metric_id", lambda: self._metric_id, self._set_metric_id)
        model.bind("compare_metric_id", lambda: self._compare_metric_id, self._set_compare_metric_id)
        model.bind("log_scale_enabled", lambda: self._log_scale_enabled, self._set_log_scale_enabled)
        model.bind("histogram_bin_count", lambda: self._histogram_bin_count, self._set_histogram_bin_count)
        model.bind("compare_x_bin_count", lambda: self._compare_x_bin_count, self._set_compare_x_bin_count)
        model.bind("compare_y_bin_count", lambda: self._compare_y_bin_count, self._set_compare_y_bin_count)
        model.bind_event("undo_history", self._on_undo_history)
        model.bind_event("redo_history", self._on_redo_history)
        model.bind_event("clear_mark", self._on_clear_mark)
        model.bind_event("delete_marked", self._on_delete_marked)
        model.bind_event("toggle_dock_mode", self._on_toggle_dock_mode)
        model.bind_event("close_panel", self._on_close_panel)
        model.bind_record_list("metric_options")
        model.bind_record_list("compare_metric_options")
        model.bind_record_list("bins")
        model.bind_record_list("compare_bins")

        self._handle = model.get_handle()
        self._rebuild_metric_options()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._chart_el = doc.get_element_by_id("histogram-bars")
        self._compare_chart_el = doc.get_element_by_id("compare-cells")
        self._sync_panel_space_state()
        if self._chart_el:
            self._chart_el.add_event_listener("mousedown", self._on_chart_mousedown)
        if self._compare_chart_el:
            self._compare_chart_el.add_event_listener("mousedown", self._on_compare_chart_mousedown)
        doc.add_event_listener("keydown", self._on_keydown)
        doc.add_event_listener("mousemove", self._on_document_mousemove)
        doc.add_event_listener("mouseup", self._on_document_mouseup)

        self._scene_generation = -1
        self._history_generation = self._history_generation_value()
        self._last_lang = lf.ui.get_current_language()
        self._trainer_state = ""
        self._rebuild_metric_options()
        self._refresh()

    def on_update(self, doc):
        del doc

        space_changed = self._sync_panel_space_state()
        scene_generation = lf.get_scene_generation()
        history_generation = self._history_generation_value()
        current_lang = lf.ui.get_current_language()
        trainer_state = AppState.trainer_state.value
        scene_changed = scene_generation != self._scene_generation
        history_changed = history_generation != self._history_generation
        sync_selection_from_scene = False
        if (scene_generation == self._scene_generation and
                history_generation == self._history_generation and
                trainer_state == self._trainer_state and
                current_lang == self._last_lang and
                not space_changed):
            return False

        if scene_changed or history_changed:
            if self._pending_selection_commit > 0:
                self._pending_selection_commit -= 1
            else:
                self._selection_owned = False
                sync_selection_from_scene = True

        self._scene_generation = scene_generation
        self._history_generation = history_generation
        self._last_lang = current_lang
        self._trainer_state = trainer_state
        self._rebuild_metric_options()
        self._refresh()
        if sync_selection_from_scene and not (self._dragging_mark or self._dragging_compare_mark):
            self._sync_panel_selection_from_scene()
        return True

    def on_scene_changed(self, doc):
        del doc
        self._scene_generation = -1

    def on_unmount(self, doc):
        self._clear_owned_scene_selection()
        self._doc = None
        self._chart_el = None
        self._compare_chart_el = None
        self._handle = None
        doc.remove_data_model("histogram_panel")

    def _set_metric_id(self, value):
        metric_id = str(value)
        if metric_id not in METRIC_BY_ID or metric_id == self._metric_id:
            return
        self._clear_all_marks(clear_scene=True)
        self._metric_id = metric_id
        if self._compare_metric_id == metric_id:
            self._compare_metric_id = ""
        self._rebuild_metric_options()
        self._refresh()

    def _set_compare_metric_id(self, value):
        metric_id = str(value)
        valid_ids = {metric.id for metric in METRICS if metric.id != self._metric_id}
        metric_id = metric_id if metric_id in valid_ids else ""
        if metric_id == self._compare_metric_id:
            return
        self._clear_compare_mark(clear_scene=self._active_mark_source == "compare")
        self._compare_metric_id = metric_id
        self._rebuild_compare_metric_options()
        self._refresh()

    def _set_log_scale_enabled(self, value):
        enabled = self._coerce_bool(value)
        if enabled == self._log_scale_enabled:
            return
        self._log_scale_enabled = enabled
        self._update_bin_records()
        self._update_compare_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _set_histogram_bin_count(self, value):
        bin_count = self._clamp_int(value, MIN_HISTOGRAM_BIN_COUNT, MAX_HISTOGRAM_BIN_COUNT)
        if bin_count == self._histogram_bin_count:
            return
        mark_bounds = self._capture_histogram_mark_value_bounds() if self._active_mark_source == "histogram" else None
        self._histogram_bin_count = bin_count
        if self._show_chart:
            self._rebuild_histogram_from_cache()
            if mark_bounds is not None and self._hist_edges is not None:
                self._restore_histogram_mark_from_value_bounds(*mark_bounds)
                self._sync_marked_range(apply_scene=False, preserve_value_bounds=True)
        if self._handle:
            self._handle.dirty_all()

    def _set_compare_x_bin_count(self, value):
        bin_count = self._clamp_int(value, MIN_COMPARE_BIN_COUNT, MAX_COMPARE_BIN_COUNT)
        if bin_count == self._compare_x_bin_count:
            return
        mark_bounds = self._capture_compare_mark_value_bounds() if self._active_mark_source == "compare" else None
        self._compare_x_bin_count = bin_count
        if self._show_compare_card:
            self._rebuild_compare_from_cache()
            if mark_bounds is not None and self._compare_x_edges is not None and self._compare_y_edges is not None:
                self._restore_compare_mark_from_value_bounds(*mark_bounds)
                self._sync_compare_mark(apply_scene=False, preserve_value_bounds=True)
        if self._handle:
            self._handle.dirty_all()

    def _set_compare_y_bin_count(self, value):
        bin_count = self._clamp_int(value, MIN_COMPARE_BIN_COUNT, MAX_COMPARE_BIN_COUNT)
        if bin_count == self._compare_y_bin_count:
            return
        mark_bounds = self._capture_compare_mark_value_bounds() if self._active_mark_source == "compare" else None
        self._compare_y_bin_count = bin_count
        if self._show_compare_card:
            self._rebuild_compare_from_cache()
            if mark_bounds is not None and self._compare_x_edges is not None and self._compare_y_edges is not None:
                self._restore_compare_mark_from_value_bounds(*mark_bounds)
                self._sync_compare_mark(apply_scene=False, preserve_value_bounds=True)
        if self._handle:
            self._handle.dirty_all()

    def _rebuild_metric_options(self):
        if not self._handle:
            return
        self._handle.update_record_list(
            "metric_options",
            [{"value": metric.id, "label": metric.label()} for metric in METRICS],
        )
        self._rebuild_compare_metric_options()

    def _rebuild_compare_metric_options(self):
        if not self._handle:
            return
        if self._compare_metric_id == self._metric_id:
            self._compare_metric_id = ""
        options = [{"value": "", "label": _tr("histogram.compare.off", "Off")}]
        options.extend(
            {"value": metric.id, "label": metric.label()}
            for metric in METRICS
            if metric.id != self._metric_id
        )
        self._handle.update_record_list("compare_metric_options", options)

    def _refresh(self):
        if not self._handle:
            return

        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            self._set_empty(
                _tr("histogram.empty.no_scene.title", "No scene loaded"),
                _tr("histogram.empty.no_scene.message", "Load a Gaussian scene, then reopen the histogram panel."),
            )
            return

        model = scene.combined_model()
        if model is None or int(getattr(model, "num_points", 0) or 0) <= 0:
            self._set_empty(
                _tr("histogram.empty.no_visible_gaussians.title", "No visible Gaussians"),
                _tr(
                    "histogram.empty.no_visible_gaussians.message",
                    "The histogram only works when the active scene exposes a visible Gaussian model.",
                ),
            )
            return

        values = self._extract_metric_values(scene, model, self._metric_id)
        if values is None:
            self._set_empty(
                _tr("histogram.empty.metric_unavailable.title", "Metric unavailable"),
                _tr(
                    "histogram.empty.metric_unavailable.message",
                    "The selected metric could not be read from the combined Gaussian model.",
                ),
            )
            return

        visible_mask = self._extract_visible_mask(model, values)
        finite_mask = values.isfinite()
        if visible_mask is not None and visible_mask.shape == values.shape:
            finite_mask = finite_mask & visible_mask

        if not self._any_true(finite_mask):
            self._set_empty(
                _tr("histogram.empty.no_visible_values.title", "No visible values"),
                _tr(
                    "histogram.empty.no_visible_values.message",
                    "The selected metric does not contain any visible finite samples to visualize.",
                ),
            )
            return

        valid_values = values[finite_mask]
        finite_values = valid_values.contiguous().cpu().to("float32")
        metric = METRIC_BY_ID[self._metric_id]
        histogram_min, histogram_max = self._histogram_bounds(finite_values, self._metric_id)
        sorted_values, _ = finite_values.sort(0, False)
        self._primary_values = values
        self._primary_finite_mask = finite_mask
        self._primary_valid_values = valid_values
        self._primary_histogram_min = histogram_min
        self._primary_histogram_max = histogram_max

        self._show_chart = True
        self._sample_count = f"{int(finite_values.shape[0]):,}"
        self._range_text = self._format_range_text(finite_values.min_scalar(), finite_values.max_scalar())
        self._mean_text = self._format_value(finite_values.mean_scalar())
        self._median_text = self._format_value(self._percentile_from_sorted(sorted_values, 50.0))
        self._p95_text = self._format_value(self._percentile_from_sorted(sorted_values, 95.0))
        self._summary_text = _trf(
            "histogram.summary",
            "{metric} distribution across {count} Gaussians",
            metric=metric.label(),
            count=f"{int(finite_values.shape[0]):,}",
        )

        self._rebuild_histogram_from_cache(reset_footer=self._active_mark_source != "compare")
        self._refresh_compare(scene, model, values, visible_mask)
        self._handle.dirty_all()

    def _set_empty(self, title: str, message: str):
        self._show_chart = False
        self._empty_title = title
        self._empty_message = message
        self._sample_count = "--"
        self._range_text = "--"
        self._mean_text = "--"
        self._median_text = "--"
        self._p95_text = "--"
        self._peak_text = "--"
        self._axis_min = "--"
        self._axis_max = "--"
        self._summary_text = ""
        self._primary_values = None
        self._primary_finite_mask = None
        self._primary_valid_values = None
        self._primary_histogram_min = 0.0
        self._primary_histogram_max = 1.0
        self._selection_bin_indices = None
        self._hist_counts = None
        self._hist_prefix_counts = None
        self._hist_edges = None
        self._hide_compare(clear_scene=self._active_mark_source == "compare")
        self._reset_marked_state(clear_scene=True)
        if self._handle:
            self._handle.update_record_list("bins", [])
            self._handle.dirty_all()

    def _rebuild_histogram_from_cache(self, reset_footer: bool = False):
        if self._primary_values is None or self._primary_finite_mask is None or self._primary_valid_values is None:
            self._selection_bin_indices = None
            self._hist_counts = None
            self._hist_prefix_counts = None
            self._hist_edges = None
            self._peak_text = "--"
            self._axis_min = "--"
            self._axis_max = "--"
            if self._handle:
                self._handle.update_record_list("bins", [])
            return

        mark_bounds = self._capture_histogram_mark_value_bounds()
        selection_bin_indices = self._build_selection_bin_indices(
            self._primary_values,
            self._primary_finite_mask,
            self._primary_histogram_min,
            self._primary_histogram_max,
            self._histogram_bin_count,
        )
        valid_bin_indices = self._bin_indices_for_values(
            self._primary_valid_values,
            self._primary_histogram_min,
            self._primary_histogram_max,
            self._histogram_bin_count,
        )
        counts, edges = self._build_histogram(
            valid_bin_indices,
            int(self._primary_valid_values.shape[0]),
            self._primary_histogram_min,
            self._primary_histogram_max,
            self._histogram_bin_count,
        )

        self._selection_bin_indices = selection_bin_indices
        self._hist_counts = counts
        self._hist_prefix_counts = self._prefix_counts(counts)
        self._hist_edges = edges
        self._peak_text = f"{max(counts, default=0):,}"
        self._axis_min = self._format_value(edges[0])
        self._axis_max = self._format_value(edges[-1])

        if mark_bounds is not None:
            self._restore_histogram_mark_from_value_bounds(*mark_bounds)
            self._sync_marked_range(apply_scene=False, preserve_value_bounds=True)
        elif self._active_mark_source == "histogram" and self._panel_selection_mask is not None:
            self._commit_histogram_mask_selection(
                self._panel_selection_mask,
                apply_scene=False,
                overlay_bounds=self._histogram_overlay_bounds,
            )
        elif reset_footer:
            self._reset_footer_mark_state(clear_scene=False)

        self._update_bin_records()

    def _rebuild_compare_from_cache(self):
        if (
            self._primary_values is None or
            self._compare_values is None or
            self._compare_finite_mask is None or
            self._compare_valid_x_values is None or
            self._compare_valid_y_values is None
        ):
            self._compare_x_bin_indices = None
            self._compare_y_bin_indices = None
            self._compare_counts = None
            self._compare_x_edges = None
            self._compare_y_edges = None
            if self._handle:
                self._handle.update_record_list("compare_bins", [])
            return

        mark_bounds = self._capture_compare_mark_value_bounds()
        x_bin_indices = self._build_selection_bin_indices(
            self._primary_values,
            self._compare_finite_mask,
            self._compare_x_min,
            self._compare_x_max,
            self._compare_x_bin_count,
        )
        y_bin_indices = self._build_selection_bin_indices(
            self._compare_values,
            self._compare_finite_mask,
            self._compare_y_min,
            self._compare_y_max,
            self._compare_y_bin_count,
        )
        valid_x_bins = self._bin_indices_for_values(
            self._compare_valid_x_values,
            self._compare_x_min,
            self._compare_x_max,
            self._compare_x_bin_count,
        )
        valid_y_bins = self._bin_indices_for_values(
            self._compare_valid_y_values,
            self._compare_y_min,
            self._compare_y_max,
            self._compare_y_bin_count,
        )
        compare_counts, x_edges, y_edges = self._build_compare_heatmap(
            valid_x_bins,
            valid_y_bins,
            int(self._compare_valid_x_values.shape[0]),
            self._compare_x_min,
            self._compare_x_max,
            self._compare_y_min,
            self._compare_y_max,
            self._compare_x_bin_count,
            self._compare_y_bin_count,
        )

        self._compare_x_bin_indices = x_bin_indices
        self._compare_y_bin_indices = y_bin_indices
        self._compare_counts = compare_counts
        self._compare_x_edges = x_edges
        self._compare_y_edges = y_edges
        self._compare_x_axis_min = self._format_value(x_edges[0])
        self._compare_x_axis_max = self._format_value(x_edges[-1])
        self._compare_y_axis_min = self._format_value(y_edges[0])
        self._compare_y_axis_max = self._format_value(y_edges[-1])

        if mark_bounds is not None:
            self._restore_compare_mark_from_value_bounds(*mark_bounds)
            self._sync_compare_mark(apply_scene=False, preserve_value_bounds=True)
        elif self._active_mark_source == "compare" and self._panel_selection_mask is not None:
            self._commit_compare_mask_selection(self._panel_selection_mask, apply_scene=False)
        elif self._active_mark_source == "compare":
            self._reset_footer_mark_state(clear_scene=False)

        self._update_compare_bin_records()

    def _build_bin_records(self, counts: list[int], edges: list[float]) -> Iterable[dict[str, object]]:
        if self._log_scale_enabled:
            display_counts = [math.log1p(float(count)) for count in counts]
        else:
            display_counts = [float(count) for count in counts]

        peak = max(max(display_counts, default=0.0), 1.0)
        selected_bins = self._selected_histogram_bins
        marked_lo, marked_hi = self._marked_bounds()
        for index, count in enumerate(counts):
            ratio = display_counts[index] / peak
            height_pct = 0.0 if count <= 0 else max(3.0, ratio * 100.0)
            alpha = 0.16 if count <= 0 else (0.34 + ratio * 0.66)
            left = self._format_value(edges[index])
            right = self._format_value(edges[index + 1])
            yield {
                "height_style": f"{height_pct:.2f}%",
                "opacity_style": f"{alpha:.3f}",
                "tooltip": _trf(
                    "histogram.bin_tooltip",
                    "Bin {index}: {left} to {right} | {count} Gaussians",
                    index=index + 1,
                    left=left,
                    right=right,
                    count=f"{int(count):,}",
                ),
                "selected": (
                    index in selected_bins or
                    (marked_lo is not None and marked_lo <= index <= marked_hi)
                ),
            }

    def _update_bin_records(self):
        if self._handle is None or self._hist_counts is None or self._hist_edges is None:
            return
        self._handle.update_record_list(
            "bins",
            list(self._build_bin_records(self._hist_counts, self._hist_edges)),
        )

    def _extract_metric_values(self, scene, model, metric_id: str | None = None) -> lf.Tensor | None:
        metric_id = self._metric_id if metric_id is None else metric_id
        try:
            if metric_id == "opacity":
                return self._float_tensor(model.get_opacity()).reshape([-1])

            if metric_id in {"position_x", "position_y", "position_z", "world_distance", "distance"}:
                world_means = self._extract_world_space_means(scene, model)
                if metric_id == "position_x":
                    return world_means[:, 0]
                if metric_id == "position_y":
                    return world_means[:, 1]
                if metric_id == "position_z":
                    return world_means[:, 2]
                if metric_id == "world_distance":
                    return self._distance_from_world_origin(world_means)
                return self._distance_from_positions(scene, world_means)

            scaling = self._float_tensor(model.get_scaling()).reshape([-1, 3])
            if metric_id == "scale_x":
                return scaling[:, 0]
            if metric_id == "scale_y":
                return scaling[:, 1]
            if metric_id == "scale_z":
                return scaling[:, 2]
            if metric_id == "scale_max":
                return scaling.max(1).reshape([-1])
            if metric_id == "volume":
                return scaling.prod(1).reshape([-1]) * (4.0 * math.pi / 3.0)
            if metric_id == "anisotropy":
                scale_min = scaling.min(1).reshape([-1])
                scale_max = scaling.max(1).reshape([-1])
                return scale_max / (scale_min + 1e-12)
            if metric_id == "erank":
                return self._effective_rank_from_scaling(scaling)
            return None
        except Exception:
            return None

    def _extract_world_space_means(self, scene, model) -> lf.Tensor:
        means = self._float_tensor(model.get_means()).reshape([-1, 3])
        if means.numel == 0:
            return lf.Tensor.zeros([0, 3], dtype="float32", device=self._device_string(means))

        world_means = self._world_space_means(scene, means)
        return means if world_means is None else world_means

    def _world_space_means(self, scene, means: lf.Tensor) -> lf.Tensor | None:
        nodes = self._visible_splat_nodes(scene)
        if not nodes:
            return None

        world_means = means.clone()
        offset = 0

        for node in nodes:
            count = int(getattr(node, "gaussian_count", 0) or 0)
            if count <= 0:
                continue

            next_offset = offset + count
            if next_offset > world_means.shape[0]:
                return None

            matrix = lf.mat4([list(row) for row in node.world_transform]).to("float32")
            matrix = self._to_device(matrix, self._device_string(world_means))
            rotation = matrix[:3, :3].transpose(0, 1)
            translation = matrix[:3, 3].unsqueeze(0).expand([count, 3])
            world_means[offset:next_offset] = world_means[offset:next_offset].matmul(rotation) + translation
            offset = next_offset

        if offset != world_means.shape[0]:
            return None
        return world_means

    def _distance_from_positions(self, scene, positions: lf.Tensor) -> lf.Tensor:
        finite_rows = positions.isfinite().all(1).reshape([-1])
        if not self._any_true(finite_rows):
            return self._nan_tensor(int(positions.shape[0]), self._device_string(positions))

        center = self._resolve_scene_center(scene, positions, finite_rows)
        distances = self._nan_tensor(int(positions.shape[0]), self._device_string(positions))
        centered = positions[finite_rows] - center.unsqueeze(0)
        distances[finite_rows] = centered.square().sum(1).sqrt().reshape([-1])
        return distances

    def _distance_from_world_origin(self, positions: lf.Tensor) -> lf.Tensor:
        finite_rows = positions.isfinite().all(1).reshape([-1])
        if not self._any_true(finite_rows):
            return self._nan_tensor(int(positions.shape[0]), self._device_string(positions))

        distances = self._nan_tensor(int(positions.shape[0]), self._device_string(positions))
        distances[finite_rows] = positions[finite_rows].square().sum(1).sqrt().reshape([-1])
        return distances

    def _visible_splat_nodes(self, scene) -> list:
        try:
            nodes = list(scene.get_nodes())
        except Exception:
            return []

        node_by_id = {}
        for node in nodes:
            try:
                node_by_id[int(node.id)] = node
            except Exception:
                continue

        visible_cache: dict[int, bool] = {}
        splat_type = getattr(getattr(lf, "NodeType", None), "SPLAT", None)
        if splat_type is None:
            splat_type = getattr(getattr(getattr(lf, "scene", None), "NodeType", None), "SPLAT", None)

        def is_effectively_visible(node) -> bool:
            node_id = int(node.id)
            if node_id in visible_cache:
                return visible_cache[node_id]

            visible = bool(getattr(node, "visible", True))
            parent_id = int(getattr(node, "parent_id", -1))
            if visible and parent_id >= 0:
                parent = node_by_id.get(parent_id)
                if parent is not None:
                    visible = is_effectively_visible(parent)

            visible_cache[node_id] = visible
            return visible

        splat_nodes = []
        for node in nodes:
            try:
                if splat_type is not None and getattr(node, "type", None) != splat_type:
                    continue
                if int(getattr(node, "gaussian_count", 0) or 0) <= 0:
                    continue
                if is_effectively_visible(node):
                    splat_nodes.append(node)
            except Exception:
                continue
        return splat_nodes

    def _resolve_scene_center(self, scene, means: lf.Tensor, finite_rows: lf.Tensor) -> lf.Tensor:
        try:
            center = getattr(scene, "scene_center", None)
            if center is not None and int(getattr(center, "ndim", 0) or 0) == 1 and tuple(center.shape) == (3,):
                center = self._float_tensor(center).reshape([-1])
                center = self._to_device(center, self._device_string(means))
                if center.isfinite().all().bool_():
                    return center
        except Exception:
            pass

        finite_means = means[finite_rows]
        if finite_means.numel == 0:
            return lf.Tensor.zeros([3], dtype="float32", device=self._device_string(means))
        return finite_means.mean(0).reshape([-1]).to("float32")

    @staticmethod
    def _float_tensor(tensor) -> lf.Tensor:
        return tensor.contiguous().to("float32")

    @staticmethod
    def _device_string(tensor: lf.Tensor) -> str:
        return "cuda" if bool(getattr(tensor, "is_cuda", False)) else "cpu"

    @staticmethod
    def _to_device(tensor: lf.Tensor, device: str) -> lf.Tensor:
        if device == "cuda":
            return tensor if tensor.is_cuda else tensor.cuda()
        return tensor.cpu() if tensor.is_cuda else tensor

    @staticmethod
    def _any_true(mask: lf.Tensor) -> bool:
        return bool(mask.count_nonzero())

    @staticmethod
    def _contiguous_span(indices: list[int]) -> tuple[int, int] | None:
        if not indices:
            return None
        ordered = sorted({int(index) for index in indices})
        if ordered[-1] - ordered[0] + 1 != len(ordered):
            return None
        return ordered[0], ordered[-1]

    @staticmethod
    def _nan_tensor(size: int, device: str) -> lf.Tensor:
        return lf.Tensor.full([size], float("nan"), dtype="float32", device=device)

    @staticmethod
    def _event_modifier_mask(event) -> int:
        for key in ("modifiers", "key_modifier_state", "key_modifiers", "modifier_state"):
            try:
                raw = event.get_parameter(key, "")
            except Exception:
                continue
            if raw is None:
                continue
            if isinstance(raw, str) and not raw.strip():
                continue
            try:
                return int(raw)
            except Exception:
                try:
                    return int(float(raw))
                except Exception:
                    continue
        return 0

    @staticmethod
    def _event_parameter_truthy(event, name: str) -> bool:
        try:
            raw = event.get_parameter(name, "")
        except Exception:
            return False
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    @classmethod
    def _event_has_modifier(cls, event, *bool_names: str, mask: int = 0) -> bool:
        for name in bool_names:
            try:
                if event.get_bool_parameter(name, False):
                    return True
            except Exception:
                pass
            if cls._event_parameter_truthy(event, name):
                return True
        return bool(mask and (cls._event_modifier_mask(event) & mask))

    @classmethod
    def _is_rml_event(cls, event) -> bool:
        """Return True only for real RmlUI events.

        Calling into global UI key state helpers (e.g. `lf.ui.is_ctrl_down()`) can
        crash in headless/unit-test contexts where no GUI is running. We gate any
        such fallbacks behind this check.
        """

        try:
            rml_module = getattr(lf.ui, "rml", None)
            rml_event_type = getattr(rml_module, "RmlEvent", None)
            return rml_event_type is not None and isinstance(event, rml_event_type)
        except Exception:
            return False

    @classmethod
    def _event_primary_shortcut_pressed(cls, event) -> bool:
        if cls._event_has_modifier(
            event,
            "ctrl_key",
            "ctrl",
            "meta_key",
            "meta",
            "super_key",
            "super",
            mask=RML_KM_CTRL | RML_KM_META,
        ):
            return True
        # Some RmlUi backends (or custom event bridges) may omit modifier parameters
        # on mouse/key events. Fall back to the global key state so Ctrl-based
        # shortcuts and selection modifiers keep working.
        if not cls._is_rml_event(event):
            return False
        try:
            return bool(lf.ui.is_ctrl_down())
        except Exception:
            return False

    def _normalize_selection_mask(self, mask: lf.Tensor | None, reference: lf.Tensor | None) -> lf.Tensor | None:
        if mask is None or reference is None:
            return None
        try:
            normalized = (mask.reshape([-1]) != 0)
        except Exception:
            return None
        expected = int(reference.shape[0]) if int(getattr(reference, "ndim", 0) or 0) > 0 else int(reference.numel)
        actual = int(normalized.shape[0]) if int(getattr(normalized, "ndim", 0) or 0) > 0 else int(normalized.numel)
        if actual != expected:
            return None
        return self._to_device(normalized.contiguous().to("bool"), self._device_string(reference))

    def _zero_mask_like(self, reference: lf.Tensor) -> lf.Tensor:
        length = int(reference.shape[0]) if int(getattr(reference, "ndim", 0) or 0) > 0 else int(reference.numel)
        return lf.Tensor.zeros([length], dtype="bool", device=self._device_string(reference))

    def _extract_visible_mask(self, model, values: lf.Tensor) -> lf.Tensor | None:
        try:
            if bool(model.has_deleted_mask()):
                deleted = getattr(model, "deleted", None)
                if deleted is None or int(getattr(deleted, "ndim", 0) or 0) != 1:
                    return None
                if int(deleted.shape[0]) != int(values.shape[0]):
                    return None
                deleted = deleted.contiguous().reshape([-1]).to("bool")
                deleted = self._to_device(deleted, self._device_string(values))
                return ~deleted
        except Exception:
            pass
        return None

    @staticmethod
    def _effective_rank_from_scaling(scaling: lf.Tensor) -> lf.Tensor:
        energy = scaling.square()
        probabilities = energy / (energy.sum(1, True) + 1e-12)
        entropy = -(probabilities * (probabilities + 1e-12).log()).sum(1)
        return entropy.exp().reshape([-1])

    def _histogram_bounds(self, values: lf.Tensor, metric_id: str | None = None) -> tuple[float, float]:
        metric_id = self._metric_id if metric_id is None else metric_id
        if metric_id == "opacity":
            return 0.0, 1.0
        if metric_id == "anisotropy":
            lo = 1.0
            hi = values.max_scalar()
            if not math.isfinite(hi):
                return lo, lo + 1.0
            if hi < lo:
                hi = lo
            if math.isclose(hi, lo, rel_tol=1e-6, abs_tol=1e-9):
                return lo, lo + 1e-3
            return lo, hi
        if metric_id == "erank":
            return 1.0, 3.0

        lo = values.min_scalar()
        hi = values.max_scalar()
        if not math.isfinite(lo) or not math.isfinite(hi):
            return 0.0, 1.0

        if math.isclose(lo, hi, rel_tol=1e-6, abs_tol=1e-9):
            padding = max(abs(lo) * 0.05, 1e-3)
            return lo - padding, hi + padding

        return lo, hi

    def _build_histogram(
        self,
        bin_indices: lf.Tensor,
        value_count: int,
        histogram_min: float,
        histogram_max: float,
        bin_count: int = DEFAULT_HISTOGRAM_BIN_COUNT,
    ) -> tuple[list[int], list[float]]:
        edges = [
            histogram_min + (histogram_max - histogram_min) * (index / bin_count)
            for index in range(bin_count + 1)
        ]

        span = histogram_max - histogram_min
        if not math.isfinite(span) or span <= 0.0:
            counts = [0] * bin_count
            if value_count > 0:
                counts[-1] = value_count
            return counts, edges

        device = self._device_string(bin_indices)
        counts_tensor = lf.Tensor.zeros([bin_count], dtype="int32", device=device)
        if value_count > 0:
            ones = lf.Tensor.ones([value_count], dtype="int32", device=device)
            counts_tensor.index_add_(0, bin_indices.contiguous().to("int32"), ones)
        counts = counts_tensor.cpu().tolist() if counts_tensor.is_cuda else counts_tensor.tolist()
        counts = [int(count) for count in counts]
        return counts, edges

    @staticmethod
    def _bin_indices_for_values(
        values: lf.Tensor,
        histogram_min: float,
        histogram_max: float,
        bin_count: int = DEFAULT_HISTOGRAM_BIN_COUNT,
    ) -> lf.Tensor:
        value_count = int(values.shape[0]) if values.ndim > 0 else int(values.numel)
        device = HistogramPanel._device_string(values)
        if value_count <= 0:
            return lf.Tensor.zeros([0], dtype="int32", device=device)

        span = histogram_max - histogram_min
        if not math.isfinite(span) or span <= 0.0:
            return lf.Tensor.full([value_count], bin_count - 1, dtype="int32", device=device)

        return (
            (((values - histogram_min) / span) * bin_count)
            .floor()
            .clamp(0.0, float(bin_count - 1))
            .reshape([-1])
            .to("int32")
        )

    def _build_selection_bin_indices(
        self,
        values: lf.Tensor,
        finite_mask: lf.Tensor,
        histogram_min: float,
        histogram_max: float,
        bin_count: int = DEFAULT_HISTOGRAM_BIN_COUNT,
    ) -> lf.Tensor:
        value_count = int(values.shape[0]) if values.ndim > 0 else int(values.numel)
        device = self._device_string(values)
        selection_bin_indices = lf.Tensor.full([value_count], -1, dtype="int32", device=device)
        if value_count <= 0 or not self._any_true(finite_mask):
            return selection_bin_indices

        selection_bin_indices[finite_mask] = self._bin_indices_for_values(
            values[finite_mask],
            histogram_min,
            histogram_max,
            bin_count,
        )
        return selection_bin_indices

    @staticmethod
    def _prefix_counts(counts: list[int]) -> list[int]:
        prefix = [0]
        running = 0
        for count in counts:
            running += int(count)
            prefix.append(running)
        return prefix

    def _selection_mask_for_current_mode(self) -> tuple[str | None, lf.Tensor | None]:
        if self._show_compare_chart and self._compare_finite_mask is not None:
            return "compare", self._compare_finite_mask
        if self._show_chart and self._primary_finite_mask is not None:
            return "histogram", self._primary_finite_mask
        return None, None

    def _sync_panel_selection_from_scene(self):
        """Keep the histogram/compare highlights in sync with external selection edits.

        Global shortcuts (e.g. Ctrl+I) and viewport selection tools update the scene
        selection mask without touching this panel's internal mark state. When that
        happens we refresh the panel's highlighted bins/cells to match the current
        scene selection.
        """

        if not self._show_chart or self._dragging_mark or self._dragging_compare_mark:
            return

        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            return

        scene_mask = getattr(scene, "selection_mask", None)
        if scene_mask is None:
            # Nothing selected (or selection cleared) -> clear panel highlight too.
            if self._has_any_mark():
                self._reset_marked_state(clear_scene=False)
            return

        if self._show_compare_chart and self._compare_finite_mask is not None and self._compare_values is not None:
            self._commit_compare_mask_selection(scene_mask, apply_scene=False)
        elif self._primary_finite_mask is not None and self._primary_values is not None:
            self._commit_histogram_mask_selection(scene_mask, apply_scene=False)

    def _selection_mode_state(self, source: str) -> tuple[lf.Tensor | None, lf.Tensor | None]:
        if source == "compare":
            return self._compare_values, self._compare_finite_mask
        return self._primary_values, self._primary_finite_mask

    def _current_selection_mask_for_source(self, source: str) -> lf.Tensor | None:
        reference, domain_mask = self._selection_mode_state(source)
        if reference is None or domain_mask is None:
            return None

        normalized_panel = self._normalize_selection_mask(self._panel_selection_mask, reference)
        if normalized_panel is not None:
            return normalized_panel & domain_mask

        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            return self._zero_mask_like(reference)

        scene_mask = self._normalize_selection_mask(getattr(scene, "selection_mask", None), reference)
        if scene_mask is None:
            return self._zero_mask_like(reference)
        return scene_mask & domain_mask

    @classmethod
    def _selection_mode_from_event(cls, event) -> str:
        shift_pressed = cls._event_has_modifier(event, "shift_key", "shift", mask=RML_KM_SHIFT)
        if not shift_pressed and cls._is_rml_event(event):
            try:
                shift_pressed = bool(lf.ui.is_shift_down())
            except Exception:
                shift_pressed = False
        if shift_pressed:
            # Match viewport selection modifiers: Shift adds, Ctrl removes.
            return "add"
        if cls._event_primary_shortcut_pressed(event):
            return "subtract"
        return "replace"

    def _compose_selection_mask(self, drag_mask: lf.Tensor | None, source: str) -> lf.Tensor | None:
        if drag_mask is None:
            return None

        if source == "compare":
            mode = self._drag_compare_selection_mode
            base_mask = self._drag_compare_selection_base_mask
        else:
            mode = self._drag_selection_mode
            base_mask = self._drag_selection_base_mask

        if mode == "replace":
            return drag_mask

        reference, domain_mask = self._selection_mode_state(source)
        if reference is None or domain_mask is None:
            return drag_mask

        if base_mask is None:
            base_mask = self._current_selection_mask_for_source(source)
        if base_mask is None:
            base_mask = self._zero_mask_like(reference)
        base_mask = self._normalize_selection_mask(base_mask, reference)
        if base_mask is None:
            base_mask = self._zero_mask_like(reference)
        base_mask = base_mask & domain_mask

        if mode == "add":
            return base_mask | drag_mask
        if mode == "subtract":
            return base_mask & ~drag_mask
        return drag_mask

    def _current_panel_selection_mask(self, source: str) -> lf.Tensor | None:
        current = self._current_selection_mask_for_source(source)
        if current is None or not self._any_true(current):
            return None
        return current

    def _selected_histogram_bins_from_mask(self, mask: lf.Tensor | None) -> set[int]:
        normalized = self._normalize_selection_mask(mask, self._primary_values)
        if normalized is None or self._selection_bin_indices is None or not self._any_true(normalized):
            return set()
        selected = self._selection_bin_indices[normalized]
        if int(selected.numel) == 0:
            return set()
        values = selected.contiguous().cpu().tolist() if selected.is_cuda else selected.tolist()
        return {int(value) for value in values if int(value) >= 0}

    def _selected_compare_cells_from_mask(self, mask: lf.Tensor | None) -> set[tuple[int, int]]:
        normalized = self._normalize_selection_mask(mask, self._compare_values)
        if (
            normalized is None or
            self._compare_x_bin_indices is None or
            self._compare_y_bin_indices is None or
            not self._any_true(normalized)
        ):
            return set()
        x_selected = self._compare_x_bin_indices[normalized]
        y_selected = self._compare_y_bin_indices[normalized]
        if int(x_selected.numel) == 0:
            return set()
        x_values = x_selected.contiguous().cpu().tolist() if x_selected.is_cuda else x_selected.tolist()
        y_values = y_selected.contiguous().cpu().tolist() if y_selected.is_cuda else y_selected.tolist()
        return {
            (int(x_bin), int(y_bin))
            for x_bin, y_bin in zip(x_values, y_values)
            if int(x_bin) >= 0 and int(y_bin) >= 0
        }

    def _refresh_compare(self, scene, model, primary_values: lf.Tensor, visible_mask: lf.Tensor | None):
        if not self._compare_metric_id:
            self._hide_compare(clear_scene=self._active_mark_source == "compare")
            return

        compare_values = self._extract_metric_values(scene, model, self._compare_metric_id)
        if compare_values is None or compare_values.shape != primary_values.shape:
            self._set_compare_empty(
                _tr("histogram.compare.empty.metric_unavailable.title", "Compare metric unavailable"),
                _tr(
                    "histogram.compare.empty.metric_unavailable.message",
                    "The comparison metric could not be read from the combined Gaussian model.",
                ),
                clear_scene=self._active_mark_source == "compare",
            )
            return

        finite_mask = primary_values.isfinite() & compare_values.isfinite()
        if visible_mask is not None and visible_mask.shape == primary_values.shape:
            finite_mask = finite_mask & visible_mask

        if not self._any_true(finite_mask):
            self._set_compare_empty(
                _tr("histogram.compare.empty.no_visible_values.title", "No comparable values"),
                _tr(
                    "histogram.compare.empty.no_visible_values.message",
                    "The selected metric pair does not contain any visible finite samples to visualize.",
                ),
                clear_scene=self._active_mark_source == "compare",
            )
            return

        x_valid = primary_values[finite_mask]
        y_valid = compare_values[finite_mask]
        x_finite = x_valid.contiguous().cpu().to("float32")
        y_finite = y_valid.contiguous().cpu().to("float32")
        x_min, x_max = self._histogram_bounds(x_finite, self._metric_id)
        y_min, y_max = self._histogram_bounds(y_finite, self._compare_metric_id)
        self._show_compare_card = True
        self._show_compare_chart = True
        self._compare_empty_title = ""
        self._compare_empty_message = ""
        self._primary_values = primary_values
        self._compare_values = compare_values
        self._compare_finite_mask = finite_mask
        self._compare_valid_x_values = x_valid
        self._compare_valid_y_values = y_valid
        self._compare_x_min = x_min
        self._compare_x_max = x_max
        self._compare_y_min = y_min
        self._compare_y_max = y_max
        self._compare_summary_text = _trf(
            "histogram.compare.summary",
            "{x_metric} vs {y_metric} across {count} Gaussians",
            x_metric=METRIC_BY_ID[self._metric_id].label(),
            y_metric=METRIC_BY_ID[self._compare_metric_id].label(),
            count=f"{int(x_finite.shape[0]):,}",
        )
        self._compare_x_metric_label = METRIC_BY_ID[self._metric_id].label()
        self._compare_y_metric_label = METRIC_BY_ID[self._compare_metric_id].label()
        self._rebuild_compare_from_cache()

    def _build_compare_heatmap(
        self,
        x_bin_indices: lf.Tensor,
        y_bin_indices: lf.Tensor,
        value_count: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        x_bin_count: int,
        y_bin_count: int,
    ) -> tuple[list[int], list[float], list[float]]:
        x_edges = [
            x_min + (x_max - x_min) * (index / x_bin_count)
            for index in range(x_bin_count + 1)
        ]
        y_edges = [
            y_min + (y_max - y_min) * (index / y_bin_count)
            for index in range(y_bin_count + 1)
        ]

        device = self._device_string(x_bin_indices)
        counts_tensor = lf.Tensor.zeros([x_bin_count * y_bin_count], dtype="int32", device=device)
        if value_count > 0:
            flat_indices = (y_bin_indices * x_bin_count + x_bin_indices).reshape([-1]).to("int32")
            ones = lf.Tensor.ones([value_count], dtype="int32", device=device)
            counts_tensor.index_add_(0, flat_indices.contiguous(), ones)
        counts = counts_tensor.cpu().tolist() if counts_tensor.is_cuda else counts_tensor.tolist()
        return [int(count) for count in counts], x_edges, y_edges

    def _set_compare_empty(self, title: str, message: str, clear_scene: bool):
        self._show_compare_card = bool(self._compare_metric_id)
        self._show_compare_chart = False
        self._compare_empty_title = title
        self._compare_empty_message = message
        self._compare_summary_text = ""
        self._compare_x_metric_label = METRIC_BY_ID[self._metric_id].label() if self._metric_id in METRIC_BY_ID else ""
        self._compare_y_metric_label = (
            METRIC_BY_ID[self._compare_metric_id].label() if self._compare_metric_id in METRIC_BY_ID else ""
        )
        self._compare_x_axis_min = "--"
        self._compare_x_axis_max = "--"
        self._compare_y_axis_min = "--"
        self._compare_y_axis_max = "--"
        self._clear_compare_cache()
        self._clear_compare_mark(clear_scene=clear_scene)
        if self._handle:
            self._handle.update_record_list("compare_bins", [])

    def _hide_compare(self, clear_scene: bool):
        self._show_compare_card = False
        self._show_compare_chart = False
        self._compare_empty_title = ""
        self._compare_empty_message = ""
        self._compare_summary_text = ""
        self._compare_x_metric_label = ""
        self._compare_y_metric_label = ""
        self._compare_x_axis_min = "--"
        self._compare_x_axis_max = "--"
        self._compare_y_axis_min = "--"
        self._compare_y_axis_max = "--"
        self._clear_compare_cache()
        self._clear_compare_mark(clear_scene=clear_scene)
        if self._handle:
            self._handle.update_record_list("compare_bins", [])

    def _clear_compare_cache(self):
        self._compare_values = None
        self._compare_finite_mask = None
        self._compare_valid_x_values = None
        self._compare_valid_y_values = None
        self._compare_x_min = 0.0
        self._compare_x_max = 1.0
        self._compare_y_min = 0.0
        self._compare_y_max = 1.0
        self._compare_x_bin_indices = None
        self._compare_y_bin_indices = None
        self._compare_counts = None
        self._compare_x_edges = None
        self._compare_y_edges = None

    def _build_compare_bin_records(self) -> Iterable[dict[str, object]]:
        if self._compare_counts is None or self._compare_x_edges is None or self._compare_y_edges is None:
            return []

        if self._log_scale_enabled:
            display_counts = [math.log1p(float(count)) for count in self._compare_counts]
        else:
            display_counts = [float(count) for count in self._compare_counts]

        peak = max(max(display_counts, default=0.0), 1.0)
        selected_cells = self._selected_compare_cells
        x_lo, x_hi, y_lo, y_hi = self._compare_marked_bounds()
        records = []
        cell_width = 100.0 / max(self._compare_x_bin_count, 1)
        cell_height = 100.0 / max(self._compare_y_bin_count, 1)
        for row_index, y_bin in enumerate(range(self._compare_y_bin_count - 1, -1, -1)):
            for x_bin in range(self._compare_x_bin_count):
                index = y_bin * self._compare_x_bin_count + x_bin
                count = self._compare_counts[index]
                ratio = display_counts[index] / peak
                opacity = 0.08 if count <= 0 else (0.20 + ratio * 0.80)
                left = self._format_range_text(self._compare_x_edges[x_bin], self._compare_x_edges[x_bin + 1])
                right = self._format_range_text(self._compare_y_edges[y_bin], self._compare_y_edges[y_bin + 1])
                records.append(
                    {
                        "style_attr": (
                            f"left: {x_bin * cell_width:.4f}%; "
                            f"top: {row_index * cell_height:.4f}%; "
                            f"width: {cell_width:.4f}%; "
                            f"height: {cell_height:.4f}%;"
                        ),
                        "opacity_style": f"{opacity:.3f}",
                        "selected": (
                            (x_bin, y_bin) in selected_cells or
                            (
                                x_lo is not None and y_lo is not None and
                                x_lo <= x_bin <= x_hi and
                                y_lo <= y_bin <= y_hi
                            )
                        ),
                        "tooltip": _trf(
                            "histogram.compare.bin_tooltip",
                            "X {x_range} | Y {y_range} | {count} Gaussians",
                            x_range=left,
                            y_range=right,
                            count=f"{int(count):,}",
                        ),
                    }
                )
        return records

    def _update_compare_bin_records(self):
        if self._handle is None:
            return
        self._handle.update_record_list("compare_bins", list(self._build_compare_bin_records()))

    def _compare_count_for_value_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float) -> int:
        mask = self._selection_mask_for_compare_value_bounds(x_min, x_max, y_min, y_max)
        if mask is None:
            return 0
        return int(mask.count_nonzero())

    def _capture_histogram_mark_value_bounds(self) -> tuple[float, float] | None:
        if self._marked_value_min is None or self._marked_value_max is None:
            return None
        return float(self._marked_value_min), float(self._marked_value_max)

    def _restore_histogram_mark_from_value_bounds(self, range_min: float, range_max: float):
        self._marked_value_min = float(min(range_min, range_max))
        self._marked_value_max = float(max(range_min, range_max))
        self._marked_bin_start = self._lower_bin_for_value(
            self._marked_value_min,
            self._primary_histogram_min,
            self._primary_histogram_max,
            self._histogram_bin_count,
        )
        self._marked_bin_end = self._upper_bin_for_value(
            self._marked_value_max,
            self._primary_histogram_min,
            self._primary_histogram_max,
            self._histogram_bin_count,
        )

    def _capture_compare_mark_value_bounds(self) -> tuple[float, float, float, float] | None:
        if (
            self._compare_mark_x_min is not None and
            self._compare_mark_x_max is not None and
            self._compare_mark_y_min is not None and
            self._compare_mark_y_max is not None
        ):
            return (
                float(self._compare_mark_x_min),
                float(self._compare_mark_x_max),
                float(self._compare_mark_y_min),
                float(self._compare_mark_y_max),
            )
        if self._compare_x_edges is None or self._compare_y_edges is None:
            return None
        x_lo, x_hi, y_lo, y_hi = self._compare_marked_bounds_from_indices(
            len(self._compare_x_edges) - 1,
            len(self._compare_y_edges) - 1,
        )
        if x_lo is None or x_hi is None or y_lo is None or y_hi is None:
            return None
        return (
            float(self._compare_x_edges[x_lo]),
            float(self._compare_x_edges[x_hi + 1]),
            float(self._compare_y_edges[y_lo]),
            float(self._compare_y_edges[y_hi + 1]),
        )

    def _restore_compare_mark_from_value_bounds(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ):
        self._compare_mark_x_min = float(min(x_min, x_max))
        self._compare_mark_x_max = float(max(x_min, x_max))
        self._compare_mark_y_min = float(min(y_min, y_max))
        self._compare_mark_y_max = float(max(y_min, y_max))
        self._compare_mark_start = (
            self._lower_bin_for_value(
                self._compare_mark_x_min,
                self._compare_x_min,
                self._compare_x_max,
                self._compare_x_bin_count,
            ),
            self._lower_bin_for_value(
                self._compare_mark_y_min,
                self._compare_y_min,
                self._compare_y_max,
                self._compare_y_bin_count,
            ),
        )
        self._compare_mark_end = (
            self._upper_bin_for_value(
                self._compare_mark_x_max,
                self._compare_x_min,
                self._compare_x_max,
                self._compare_x_bin_count,
            ),
            self._upper_bin_for_value(
                self._compare_mark_y_max,
                self._compare_y_min,
                self._compare_y_max,
                self._compare_y_bin_count,
            ),
        )

    def _marked_count_for_value_bounds(self, range_min: float, range_max: float) -> int:
        mask = self._selection_mask_for_value_bounds(range_min, range_max)
        if mask is None:
            return 0
        return int(mask.count_nonzero())

    def _selection_mask_for_histogram_bin(self, bin_index: int) -> lf.Tensor | None:
        if self._hist_edges is None or bin_index < 0 or bin_index + 1 >= len(self._hist_edges):
            return None
        return self._selection_mask_for_value_bounds(
            float(self._hist_edges[bin_index]),
            float(self._hist_edges[bin_index + 1]),
        )

    def _toggle_histogram_bin_selection(self, bin_index: int) -> bool:
        base_mask = self._drag_selection_base_mask
        if base_mask is None:
            base_mask = self._current_selection_mask_for_source("histogram")
        if base_mask is None:
            return False

        bin_mask = self._selection_mask_for_histogram_bin(bin_index)
        normalized_base = self._normalize_selection_mask(base_mask, self._primary_values)
        if normalized_base is None or bin_mask is None:
            return False

        if not self._any_true(normalized_base & bin_mask):
            return False

        next_mask = normalized_base & ~bin_mask
        if self._any_true(next_mask):
            self._commit_histogram_mask_selection(next_mask, apply_scene=True)
        else:
            self._reset_marked_state(clear_scene=True)
        return True

    def _maybe_promote_histogram_drag_selection_mode(self, event) -> bool:
        if self._drag_selection_mode != "replace":
            return False
        mode = self._selection_mode_from_event(event)
        if mode == "replace":
            return False
        self._drag_selection_mode = mode
        if self._drag_selection_base_mask is None:
            self._drag_selection_base_mask = self._current_selection_mask_for_source("histogram")
        return True

    def _apply_scene_selection_mask(self, mask: lf.Tensor | None):
        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            self._selection_owned = False
            self._pending_selection_commit = 0
            return

        normalized = self._normalize_selection_mask(mask, self._primary_values if self._primary_values is not None else self._compare_values)
        if normalized is None or not self._any_true(normalized):
            scene.clear_selection()
            self._selection_owned = False
            self._pending_selection_commit = 0
            return

        try:
            scene.set_selection_mask(normalized.contiguous())
            self._selection_owned = True
            # A histogram commit can trigger separate scene and undo/history updates.
            # Keep ownership across both so we do not immediately resync stale scene state.
            self._pending_selection_commit = 2
        except Exception:
            self._selection_owned = False
            self._pending_selection_commit = 0

    def _clear_histogram_overlay(self):
        self._histogram_overlay_bounds = None
        self._marked_bin_start = None
        self._marked_bin_end = None
        self._marked_value_min = None
        self._marked_value_max = None
        self._selection_left_style = "0%"
        self._selection_width_style = "0%"

    def _clear_compare_overlay(self):
        self._compare_overlay_bounds = None
        self._compare_mark_start = None
        self._compare_mark_end = None
        self._compare_mark_x_min = None
        self._compare_mark_x_max = None
        self._compare_mark_y_min = None
        self._compare_mark_y_max = None
        self._compare_selection_left_style = "0%"
        self._compare_selection_top_style = "0%"
        self._compare_selection_width_style = "0%"
        self._compare_selection_height_style = "0%"

    def _commit_histogram_mask_selection(
        self,
        mask: lf.Tensor | None,
        apply_scene: bool,
        force_full_domain: bool = False,
        overlay_bounds: tuple[int, int] | None = None,
        explicit_value_bounds: tuple[float, float] | None = None,
    ):
        normalized = self._normalize_selection_mask(mask, self._primary_values)
        if normalized is None:
            self._reset_marked_state(clear_scene=apply_scene)
            return

        normalized = normalized & self._primary_finite_mask
        if not self._any_true(normalized):
            self._reset_marked_state(clear_scene=apply_scene)
            return

        self._panel_selection_mask = normalized.contiguous()
        self._active_mark_source = "histogram"
        self._clear_compare_overlay()
        self._selected_compare_cells.clear()
        if force_full_domain:
            selected_bins = list(range(self._histogram_bin_count))
        else:
            selected_bins = sorted(self._selected_histogram_bins_from_mask(normalized))
        self._selected_histogram_bins = set(selected_bins)

        if force_full_domain:
            selection_span = (0, self._histogram_bin_count - 1)
        elif explicit_value_bounds is not None and overlay_bounds is not None:
            selection_span = overlay_bounds
        else:
            selection_span = self._contiguous_span(selected_bins)
        render_span = overlay_bounds if overlay_bounds is not None else selection_span
        if render_span is None:
            self._clear_histogram_overlay()
        else:
            self._histogram_overlay_bounds = render_span
            left_px, width_px = self._histogram_selection_geometry(render_span[0], render_span[1])
            self._selection_left_style = f"{left_px:.2f}px"
            self._selection_width_style = f"{max(width_px, 1.0):.2f}px"

        if force_full_domain and self._hist_edges is not None:
            selected_value_min = float(self._hist_edges[0])
            selected_value_max = float(self._hist_edges[-1])
        elif explicit_value_bounds is not None:
            selected_value_min = float(min(explicit_value_bounds))
            selected_value_max = float(max(explicit_value_bounds))
        elif selection_span is not None and self._hist_edges is not None:
            selected_value_min = float(self._hist_edges[selection_span[0]])
            selected_value_max = float(self._hist_edges[selection_span[1] + 1])
        else:
            selected_value_min = None
            selected_value_max = None

        if selection_span is None or selected_value_min is None or selected_value_max is None:
            if not self._dragging_mark:
                self._marked_bin_start = None
                self._marked_bin_end = None
            self._marked_value_min = None
            self._marked_value_max = None
            self._marked_range_text = "Multiple ranges"
        else:
            if not self._dragging_mark:
                self._marked_bin_start = selection_span[0]
                self._marked_bin_end = selection_span[1]
            self._marked_value_min = selected_value_min
            self._marked_value_max = selected_value_max
            self._marked_range_text = self._format_range_text(selected_value_min, selected_value_max)

        self._marked_count = int(normalized.count_nonzero())
        self._marked_count_text = _trf(
            "histogram.gaussian_count",
            "{count} Gaussians",
            count=f"{self._marked_count:,}",
        )
        self._status_hint = _tr(
            "histogram.status_selection",
            "Marked range becomes the active Gaussian selection.",
        )

        if apply_scene:
            self._apply_scene_selection_mask(normalized)
        self._update_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _commit_compare_mask_selection(
        self,
        mask: lf.Tensor | None,
        apply_scene: bool,
        force_full_domain: bool = False,
        overlay_bounds: tuple[int, int, int, int] | None = None,
        explicit_value_bounds: tuple[float, float, float, float] | None = None,
    ):
        normalized = self._normalize_selection_mask(mask, self._compare_values)
        if normalized is None:
            self._reset_marked_state(clear_scene=apply_scene)
            return

        normalized = normalized & self._compare_finite_mask
        if not self._any_true(normalized):
            self._reset_marked_state(clear_scene=apply_scene)
            return

        self._panel_selection_mask = normalized.contiguous()
        self._active_mark_source = "compare"
        self._clear_histogram_overlay()
        self._selected_histogram_bins.clear()
        if force_full_domain:
            selected_cells = {
                (x_bin, y_bin)
                for y_bin in range(self._compare_y_bin_count)
                for x_bin in range(self._compare_x_bin_count)
            }
        else:
            selected_cells = self._selected_compare_cells_from_mask(normalized)
        self._selected_compare_cells = selected_cells

        x_bins = sorted({x_bin for x_bin, _ in selected_cells})
        y_bins = sorted({y_bin for _, y_bin in selected_cells})
        if force_full_domain:
            x_span = (0, self._compare_x_bin_count - 1)
            y_span = (0, self._compare_y_bin_count - 1)
        elif explicit_value_bounds is not None and overlay_bounds is not None:
            x_span = (overlay_bounds[0], overlay_bounds[1])
            y_span = (overlay_bounds[2], overlay_bounds[3])
        else:
            x_span = self._contiguous_span(x_bins)
            y_span = self._contiguous_span(y_bins)
        render_bounds = overlay_bounds
        if render_bounds is None and x_span is not None and y_span is not None:
            render_bounds = (x_span[0], x_span[1], y_span[0], y_span[1])

        if render_bounds is None:
            self._clear_compare_overlay()
        else:
            self._compare_overlay_bounds = render_bounds
            left_ratio = render_bounds[0] / self._compare_x_bin_count
            width_ratio = (render_bounds[1] + 1) / self._compare_x_bin_count - left_ratio
            top_ratio = (self._compare_y_bin_count - 1 - render_bounds[3]) / self._compare_y_bin_count
            height_ratio = (render_bounds[3] - render_bounds[2] + 1) / self._compare_y_bin_count
            self._compare_selection_left_style = f"{left_ratio * 100.0:.2f}%"
            self._compare_selection_top_style = f"{top_ratio * 100.0:.2f}%"
            self._compare_selection_width_style = f"{max(width_ratio * 100.0, 1.0):.2f}%"
            self._compare_selection_height_style = f"{max(height_ratio * 100.0, 1.0):.2f}%"

        if force_full_domain and self._compare_x_edges is not None and self._compare_y_edges is not None:
            selected_x_min = float(self._compare_x_edges[0])
            selected_x_max = float(self._compare_x_edges[-1])
            selected_y_min = float(self._compare_y_edges[0])
            selected_y_max = float(self._compare_y_edges[-1])
        elif explicit_value_bounds is not None:
            selected_x_min = float(min(explicit_value_bounds[0], explicit_value_bounds[1]))
            selected_x_max = float(max(explicit_value_bounds[0], explicit_value_bounds[1]))
            selected_y_min = float(min(explicit_value_bounds[2], explicit_value_bounds[3]))
            selected_y_max = float(max(explicit_value_bounds[2], explicit_value_bounds[3]))
        elif x_span is not None and y_span is not None and self._compare_x_edges is not None and self._compare_y_edges is not None:
            selected_x_min = float(self._compare_x_edges[x_span[0]])
            selected_x_max = float(self._compare_x_edges[x_span[1] + 1])
            selected_y_min = float(self._compare_y_edges[y_span[0]])
            selected_y_max = float(self._compare_y_edges[y_span[1] + 1])
        else:
            selected_x_min = None
            selected_x_max = None
            selected_y_min = None
            selected_y_max = None

        if (
            x_span is None or
            y_span is None or
            len(selected_cells) != len(x_bins) * len(y_bins) or
            selected_x_min is None or
            selected_x_max is None or
            selected_y_min is None or
            selected_y_max is None
        ):
            if not self._dragging_compare_mark:
                self._compare_mark_start = None
                self._compare_mark_end = None
            self._compare_mark_x_min = None
            self._compare_mark_x_max = None
            self._compare_mark_y_min = None
            self._compare_mark_y_max = None
            self._marked_range_text = "Multiple regions"
        else:
            x_lo, x_hi = x_span
            y_lo, y_hi = y_span
            if not self._dragging_compare_mark:
                self._compare_mark_start = (x_lo, y_lo)
                self._compare_mark_end = (x_hi, y_hi)
            self._compare_mark_x_min = selected_x_min
            self._compare_mark_x_max = selected_x_max
            self._compare_mark_y_min = selected_y_min
            self._compare_mark_y_max = selected_y_max
            self._marked_range_text = _trf(
                "histogram.compare.range_value",
                "X {x_range} | Y {y_range}",
                x_range=self._format_range_text(selected_x_min, selected_x_max),
                y_range=self._format_range_text(selected_y_min, selected_y_max),
            )

        self._marked_count = int(normalized.count_nonzero())
        self._marked_count_text = _trf(
            "histogram.gaussian_count",
            "{count} Gaussians",
            count=f"{self._marked_count:,}",
        )
        self._status_hint = _tr(
            "histogram.compare.status_selection",
            "Marked compare region becomes the active Gaussian selection.",
        )

        if apply_scene:
            self._apply_scene_selection_mask(normalized)
        self._update_compare_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _select_all_current_mode(self):
        source, domain_mask = self._selection_mask_for_current_mode()
        if source is None or domain_mask is None:
            return
        if source == "compare":
            self._commit_compare_mask_selection(
                domain_mask,
                apply_scene=True,
                force_full_domain=True,
                overlay_bounds=(0, self._compare_x_bin_count - 1, 0, self._compare_y_bin_count - 1),
            )
        else:
            self._commit_histogram_mask_selection(
                domain_mask,
                apply_scene=True,
                force_full_domain=True,
                overlay_bounds=(0, self._histogram_bin_count - 1),
            )

    def _invert_current_mode_selection(self):
        source, domain_mask = self._selection_mask_for_current_mode()
        if source is None or domain_mask is None:
            return
        reference = self._compare_values if source == "compare" else self._primary_values
        current_mask = None
        if self._active_mark_source == source and self._panel_selection_mask is not None:
            current_mask = self._normalize_selection_mask(self._panel_selection_mask, reference)
            if current_mask is not None:
                current_mask = current_mask & domain_mask
        if current_mask is None:
            current_mask = self._current_panel_selection_mask(source)
        if current_mask is None or not self._any_true(current_mask):
            inverted = domain_mask
            force_full_domain = True
        else:
            inverted = domain_mask & ~current_mask
            force_full_domain = False
        if source == "compare":
            self._commit_compare_mask_selection(inverted, apply_scene=True, force_full_domain=force_full_domain)
        else:
            self._commit_histogram_mask_selection(inverted, apply_scene=True, force_full_domain=force_full_domain)

    def _on_keydown(self, event):
        key = int(event.get_parameter("key_identifier", "0"))
        ctrl_pressed = self._event_primary_shortcut_pressed(event)

        if ctrl_pressed and key == KI_A:
            self._select_all_current_mode()
            event.stop_propagation()
            return

        if ctrl_pressed and key == KI_I:
            self._invert_current_mode_selection()
            event.stop_propagation()
            return

        if key == KI_DELETE and self._has_any_mark():
            self._on_delete_marked(None, None, None)
            event.stop_propagation()

    @staticmethod
    def _percentile_from_sorted(sorted_values: lf.Tensor, percentile: float) -> float:
        count = int(sorted_values.shape[0])
        if count <= 0:
            return 0.0
        if count == 1:
            return sorted_values[0].item()

        position = (count - 1) * max(0.0, min(percentile, 100.0)) / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return sorted_values[lower].item()

        weight = position - lower
        lower_value = sorted_values[lower].item()
        upper_value = sorted_values[upper].item()
        return lower_value + (upper_value - lower_value) * weight

    def _on_chart_mousedown(self, event):
        if not self._show_chart or self._chart_el is None or self._hist_edges is None:
            return
        if int(event.get_parameter("button", "0")) != 0:
            return
        # Ensure key shortcuts (Ctrl+A / Ctrl+I) are routed to this panel.
        try:
            self._chart_el.focus()
        except Exception:
            pass

        self._drag_selection_mode = self._selection_mode_from_event(event)
        self._drag_selection_base_mask = self._current_selection_mask_for_source("histogram")
        self._clear_compare_mark(clear_scene=False)
        bin_index = self._bin_index_for_mouse_x(self._event_mouse_x(event))
        self._dragging_mark = True
        self._marked_bin_start = bin_index
        self._marked_bin_end = bin_index
        self._sync_marked_range(apply_scene=False)
        event.stop_propagation()

    def _on_compare_chart_mousedown(self, event):
        if not self._show_compare_chart or self._compare_chart_el is None:
            return
        if int(event.get_parameter("button", "0")) != 0:
            return
        # Ensure key shortcuts (Ctrl+A / Ctrl+I) are routed to this panel.
        try:
            self._compare_chart_el.focus()
        except Exception:
            pass

        self._drag_compare_selection_mode = self._selection_mode_from_event(event)
        self._drag_compare_selection_base_mask = None
        if self._drag_compare_selection_mode != "replace":
            self._drag_compare_selection_base_mask = self._current_selection_mask_for_source("compare")
        self._clear_histogram_mark(clear_scene=False)
        x_bin, y_bin = self._compare_bin_indices_for_mouse(
            self._event_mouse_x(event),
            self._event_mouse_y(event),
        )
        self._dragging_compare_mark = True
        self._compare_mark_start = (x_bin, y_bin)
        self._compare_mark_end = (x_bin, y_bin)
        self._sync_compare_mark(apply_scene=False)
        event.stop_propagation()

    def _on_document_mousemove(self, event):
        if self._dragging_compare_mark and self._compare_chart_el is not None:
            bins = self._compare_bin_indices_for_mouse(self._event_mouse_x(event), self._event_mouse_y(event))
            if bins == self._compare_mark_end:
                return
            self._compare_mark_end = bins
            self._sync_compare_mark(apply_scene=False)
            event.stop_propagation()
            return

        if self._dragging_mark and self._chart_el is not None:
            mode_changed = self._maybe_promote_histogram_drag_selection_mode(event)
            bin_index = self._bin_index_for_mouse_x(self._event_mouse_x(event))
            if bin_index == self._marked_bin_end and not mode_changed:
                return
            self._marked_bin_end = bin_index
            self._sync_marked_range(apply_scene=False)
            event.stop_propagation()

    def _on_document_mouseup(self, event):
        if self._dragging_compare_mark:
            self._sync_compare_mark(apply_scene=True)
            self._dragging_compare_mark = False
            self._drag_compare_selection_mode = "replace"
            self._drag_compare_selection_base_mask = None
            event.stop_propagation()
            return

        if self._dragging_mark:
            self._maybe_promote_histogram_drag_selection_mode(event)
            toggled = (
                self._drag_selection_mode == "replace" and
                self._marked_bin_start is not None and
                self._marked_bin_start == self._marked_bin_end and
                self._toggle_histogram_bin_selection(self._marked_bin_start)
            )
            if not toggled:
                self._sync_marked_range(apply_scene=True)
            self._dragging_mark = False
            self._drag_selection_mode = "replace"
            self._drag_selection_base_mask = None
            event.stop_propagation()

    def _event_mouse_x(self, event) -> float:
        try:
            return float(event.get_parameter("mouse_x", "0"))
        except Exception:
            return 0.0

    def _event_mouse_y(self, event) -> float:
        try:
            return float(event.get_parameter("mouse_y", "0"))
        except Exception:
            return 0.0

    def _bin_index_for_mouse_x(self, mouse_x: float) -> int:
        if self._chart_el is None:
            return 0
        left = float(self._chart_el.absolute_left)
        total_width, bar_width, gap_width = self._histogram_bar_geometry(self._histogram_bin_count)
        local_x = min(max(mouse_x - left, 0.0), max(total_width - 1e-6, 0.0))
        step = max(bar_width + gap_width, 1e-6)
        return min(self._histogram_bin_count - 1, max(0, int(math.floor(local_x / step))))

    def _marked_bounds_from_indices(self, bin_count: int | None = None) -> tuple[int | None, int | None]:
        if self._marked_bin_start is None or self._marked_bin_end is None:
            return None, None
        bin_count = max(1, self._histogram_bin_count if bin_count is None else int(bin_count))
        lo = max(0, min(self._marked_bin_start, self._marked_bin_end))
        hi = min(bin_count - 1, max(self._marked_bin_start, self._marked_bin_end))
        return lo, hi

    def _marked_bounds(self, bin_count: int | None = None) -> tuple[int | None, int | None]:
        bin_count = max(1, self._histogram_bin_count if bin_count is None else int(bin_count))
        if self._dragging_mark:
            return self._marked_bounds_from_indices(bin_count)
        if (
            self._marked_value_min is not None and
            self._marked_value_max is not None and
            self._primary_valid_values is not None
        ):
            return (
                self._lower_bin_for_value(
                    self._marked_value_min,
                    self._primary_histogram_min,
                    self._primary_histogram_max,
                    bin_count,
                ),
                self._upper_bin_for_value(
                    self._marked_value_max,
                    self._primary_histogram_min,
                    self._primary_histogram_max,
                    bin_count,
                ),
            )
        return self._marked_bounds_from_indices(bin_count)

    def _has_marked_range(self) -> bool:
        lo, hi = self._marked_bounds()
        return lo is not None and hi is not None

    def _compare_marked_bounds_from_indices(
        self,
        x_bin_count: int | None = None,
        y_bin_count: int | None = None,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        if self._compare_mark_start is None or self._compare_mark_end is None:
            return None, None, None, None
        x_bin_count = max(1, self._compare_x_bin_count if x_bin_count is None else int(x_bin_count))
        y_bin_count = max(1, self._compare_y_bin_count if y_bin_count is None else int(y_bin_count))
        x_lo = max(0, min(self._compare_mark_start[0], self._compare_mark_end[0]))
        x_hi = min(x_bin_count - 1, max(self._compare_mark_start[0], self._compare_mark_end[0]))
        y_lo = max(0, min(self._compare_mark_start[1], self._compare_mark_end[1]))
        y_hi = min(y_bin_count - 1, max(self._compare_mark_start[1], self._compare_mark_end[1]))
        return x_lo, x_hi, y_lo, y_hi

    def _compare_marked_bounds(
        self,
        x_bin_count: int | None = None,
        y_bin_count: int | None = None,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        x_bin_count = max(1, self._compare_x_bin_count if x_bin_count is None else int(x_bin_count))
        y_bin_count = max(1, self._compare_y_bin_count if y_bin_count is None else int(y_bin_count))
        if self._dragging_compare_mark:
            return self._compare_marked_bounds_from_indices(x_bin_count, y_bin_count)
        if (
            self._compare_mark_x_min is not None and
            self._compare_mark_x_max is not None and
            self._compare_mark_y_min is not None and
            self._compare_mark_y_max is not None and
            self._compare_valid_x_values is not None and
            self._compare_valid_y_values is not None
        ):
            return (
                self._lower_bin_for_value(self._compare_mark_x_min, self._compare_x_min, self._compare_x_max, x_bin_count),
                self._upper_bin_for_value(self._compare_mark_x_max, self._compare_x_min, self._compare_x_max, x_bin_count),
                self._lower_bin_for_value(self._compare_mark_y_min, self._compare_y_min, self._compare_y_max, y_bin_count),
                self._upper_bin_for_value(self._compare_mark_y_max, self._compare_y_min, self._compare_y_max, y_bin_count),
            )
        return self._compare_marked_bounds_from_indices(x_bin_count, y_bin_count)

    def _has_compare_marked_range(self) -> bool:
        x_lo, x_hi, y_lo, y_hi = self._compare_marked_bounds()
        return x_lo is not None and x_hi is not None and y_lo is not None and y_hi is not None

    def _has_any_mark(self) -> bool:
        return (
            self._has_marked_range() or
            self._has_compare_marked_range() or
            (self._panel_selection_mask is not None and self._any_true(self._panel_selection_mask))
        )

    def _sync_marked_range(self, apply_scene: bool, preserve_value_bounds: bool = False):
        if self._hist_counts is None or self._hist_edges is None:
            self._reset_footer_mark_state(clear_scene=apply_scene)
            return

        lo, hi = self._marked_bounds()
        if lo is None or hi is None:
            self._reset_footer_mark_state(clear_scene=apply_scene)
            return

        self._active_mark_source = "histogram"
        if not preserve_value_bounds or self._marked_value_min is None or self._marked_value_max is None:
            self._marked_value_min = float(self._hist_edges[lo])
            self._marked_value_max = float(self._hist_edges[hi + 1])
        explicit_bounds = (self._marked_value_min, self._marked_value_max)
        self._commit_histogram_mask_selection(
            self._compose_selection_mask(
                self._selection_mask_for_value_bounds(self._marked_value_min, self._marked_value_max),
                "histogram",
            ),
            apply_scene=apply_scene,
            overlay_bounds=(lo, hi),
            explicit_value_bounds=explicit_bounds if self._drag_selection_mode == "replace" else None,
        )

    def _histogram_bar_geometry(self, bin_count: int) -> tuple[float, float, float]:
        bin_count = max(1, int(bin_count))
        if self._chart_el is None:
            return float(bin_count), 1.0, 0.0

        total_width = max(float(self._chart_el.absolute_width), 1.0)
        gap_width = HISTOGRAM_BAR_GAP
        total_gap = gap_width * max(bin_count - 1, 0)
        if total_gap >= total_width:
            gap_width = 0.0
            total_gap = 0.0
        bar_width = max((total_width - total_gap) / bin_count, 1.0 / bin_count)
        return total_width, bar_width, gap_width

    def _histogram_selection_geometry(self, lo: int, hi: int) -> tuple[float, float]:
        _, bar_width, gap_width = self._histogram_bar_geometry(self._histogram_bin_count)
        left = lo * (bar_width + gap_width)
        width = ((hi - lo) + 1) * bar_width + max(0, hi - lo) * gap_width
        return left, width

    def _sync_compare_mark(self, apply_scene: bool, preserve_value_bounds: bool = False):
        if self._compare_counts is None or self._compare_x_edges is None or self._compare_y_edges is None:
            self._reset_footer_mark_state(clear_scene=apply_scene)
            return

        x_lo, x_hi, y_lo, y_hi = self._compare_marked_bounds()
        if x_lo is None or x_hi is None or y_lo is None or y_hi is None:
            self._reset_footer_mark_state(clear_scene=apply_scene)
            return

        self._active_mark_source = "compare"
        if not preserve_value_bounds or any(
            bound is None
            for bound in (
                self._compare_mark_x_min,
                self._compare_mark_x_max,
                self._compare_mark_y_min,
                self._compare_mark_y_max,
            )
        ):
            self._compare_mark_x_min = float(self._compare_x_edges[x_lo])
            self._compare_mark_x_max = float(self._compare_x_edges[x_hi + 1])
            self._compare_mark_y_min = float(self._compare_y_edges[y_lo])
            self._compare_mark_y_max = float(self._compare_y_edges[y_hi + 1])
        explicit_bounds = (
            self._compare_mark_x_min,
            self._compare_mark_x_max,
            self._compare_mark_y_min,
            self._compare_mark_y_max,
        )
        self._commit_compare_mask_selection(
            self._compose_selection_mask(
                self._selection_mask_for_compare_value_bounds(
                    self._compare_mark_x_min,
                    self._compare_mark_x_max,
                    self._compare_mark_y_min,
                    self._compare_mark_y_max,
                ),
                "compare",
            ),
            apply_scene=apply_scene,
            overlay_bounds=(x_lo, x_hi, y_lo, y_hi),
            explicit_value_bounds=explicit_bounds if self._drag_compare_selection_mode == "replace" else None,
        )

    def _reset_footer_mark_state(self, clear_scene: bool):
        self._clear_histogram_overlay()
        self._clear_compare_overlay()
        self._drag_selection_mode = "replace"
        self._drag_selection_base_mask = None
        self._drag_compare_selection_mode = "replace"
        self._drag_compare_selection_base_mask = None
        self._panel_selection_mask = None
        self._selected_histogram_bins.clear()
        self._selected_compare_cells.clear()
        self._marked_count = 0
        self._marked_range_text = _tr("histogram.no_marked_range", "No marked range")
        self._marked_count_text = _trf("histogram.gaussian_count", "{count} Gaussians", count="0")
        self._status_hint = _tr(
            "histogram.status_drag_delete",
            "Left-drag across the histogram to mark a range, then delete it.",
        )
        self._active_mark_source = None
        if clear_scene:
            self._clear_owned_scene_selection()
        else:
            self._selection_owned = False
            self._pending_selection_commit = 0

    def _reset_marked_state(self, clear_scene: bool):
        self._reset_footer_mark_state(clear_scene=clear_scene)
        self._update_bin_records()
        self._update_compare_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _clear_histogram_mark(self, clear_scene: bool):
        self._clear_histogram_overlay()
        self._selected_histogram_bins.clear()
        if self._active_mark_source == "histogram":
            self._reset_footer_mark_state(clear_scene=clear_scene)
        self._update_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _clear_compare_mark(self, clear_scene: bool):
        self._clear_compare_overlay()
        self._selected_compare_cells.clear()
        if self._active_mark_source == "compare":
            self._reset_footer_mark_state(clear_scene=clear_scene)
        self._update_compare_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _clear_all_marks(self, clear_scene: bool):
        self._reset_marked_state(clear_scene=clear_scene)

    def _selection_mask_for_value_bounds(self, range_min: float, range_max: float) -> lf.Tensor | None:
        if self._primary_values is None or self._primary_finite_mask is None:
            return None
        return self._primary_finite_mask & self._value_range_mask(
            self._primary_values,
            range_min,
            range_max,
            self._primary_histogram_max,
        )

    def _selection_mask_for_compare_value_bounds(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> lf.Tensor | None:
        if self._primary_values is None or self._compare_values is None or self._compare_finite_mask is None:
            return None
        mask = self._compare_finite_mask & self._value_range_mask(
            self._primary_values,
            x_min,
            x_max,
            self._compare_x_max,
        )
        mask = mask & self._value_range_mask(
            self._compare_values,
            y_min,
            y_max,
            self._compare_y_max,
        )
        return mask

    def _apply_scene_selection(self, range_min: float, range_max: float):
        self._apply_scene_selection_mask(self._selection_mask_for_value_bounds(range_min, range_max))

    def _apply_compare_scene_selection(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self._apply_scene_selection_mask(
            self._selection_mask_for_compare_value_bounds(x_min, x_max, y_min, y_max)
        )

    def _compare_bin_indices_for_mouse(self, mouse_x: float, mouse_y: float) -> tuple[int, int]:
        if self._compare_chart_el is None:
            return 0, 0
        left = float(self._compare_chart_el.absolute_left)
        top = float(self._compare_chart_el.absolute_top)
        width = max(float(self._compare_chart_el.absolute_width), 1.0)
        height = max(float(self._compare_chart_el.absolute_height), 1.0)
        norm_x = min(1.0, max(0.0, (mouse_x - left) / width))
        norm_y = min(1.0, max(0.0, (mouse_y - top) / height))
        x_bin = min(self._compare_x_bin_count - 1, max(0, int(math.floor(norm_x * self._compare_x_bin_count))))
        y_bin = min(
            self._compare_y_bin_count - 1,
            max(0, int(math.floor((1.0 - norm_y) * self._compare_y_bin_count))),
        )
        return x_bin, y_bin

    def _clear_owned_scene_selection(self):
        scene = lf.get_scene()
        if scene is not None and scene.is_valid() and self._selection_owned:
            scene.clear_selection()
        self._selection_owned = False
        self._pending_selection_commit = 0

    @staticmethod
    def _history_generation_value() -> int:
        try:
            return int(lf.undo.generation())
        except Exception:
            return -1

    @staticmethod
    def _can_undo() -> bool:
        try:
            return bool(lf.undo.can_undo())
        except Exception:
            return False

    @staticmethod
    def _can_redo() -> bool:
        try:
            return bool(lf.undo.can_redo())
        except Exception:
            return False

    @staticmethod
    def _history_name(get_name) -> str:
        try:
            return str(get_name() or "").strip()
        except Exception:
            return ""

    def _undo_tooltip(self) -> str:
        name = self._history_name(lf.undo.get_undo_name) if self._can_undo() else ""
        if name:
            return _trf("histogram.undo_named", "Undo: {name}", name=name)
        return _tr("histogram.undo", "Undo")

    def _dock_toggle_label(self) -> str:
        return "Dock" if self._is_floating else "Undock"

    def _redo_tooltip(self) -> str:
        name = self._history_name(lf.undo.get_redo_name) if self._can_redo() else ""
        if name:
            return _trf("histogram.redo_named", "Redo: {name}", name=name)
        return _tr("histogram.redo", "Redo")

    def _sync_panel_space_state(self) -> bool:
        info = None
        try:
            info = lf.ui.get_panel(self.id)
        except Exception:
            info = None

        panel_space = getattr(info, "space", self._panel_space)
        is_floating = panel_space == lf.ui.PanelSpace.FLOATING
        changed = panel_space != self._panel_space or is_floating != self._is_floating
        self._panel_space = panel_space
        self._is_floating = is_floating
        return changed

    def _on_clear_mark(self, _handle, _event, _args):
        self._clear_all_marks(clear_scene=True)

    def _on_toggle_dock_mode(self, _handle, _event, _args):
        target_space = lf.ui.PanelSpace.BOTTOM_DOCK if self._is_floating else lf.ui.PanelSpace.FLOATING
        try:
            changed = bool(lf.ui.set_panel_space(self.id, target_space))
        except Exception:
            changed = False
        if not changed:
            return
        self._sync_panel_space_state()
        if self._handle:
            self._handle.dirty_all()

    def _on_close_panel(self, _handle, _event, _args):
        self._clear_owned_scene_selection()
        try:
            lf.selection.clear_preview()
        except Exception:
            pass
        lf.ui.set_panel_enabled(self.id, False)

    def _on_undo_history(self, _handle, _event, _args):
        try:
            changed = self._can_undo() and lf.undo.undo()
        except Exception:
            changed = False
        if changed:
            self._scene_generation = -1
            self._history_generation = -1
            self._refresh()

    def _on_redo_history(self, _handle, _event, _args):
        try:
            changed = self._can_redo() and lf.undo.redo()
        except Exception:
            changed = False
        if changed:
            self._scene_generation = -1
            self._history_generation = -1
            self._refresh()

    def _execute_delete_pipeline(self) -> str | None:
        try:
            result = lf.pipeline.edit.delete_().execute()
        except Exception as exc:
            return str(exc).strip() or _tr(
                "histogram.delete_failed.unexpected",
                "The delete pipeline raised an unexpected error.",
            )

        result_get = getattr(result, "get", None)
        if result_get is None:
            return _tr("histogram.delete_failed.invalid_result", "The delete pipeline returned an invalid result.")

        if bool(result_get("ok", False)):
            return None

        error = str(result_get("error", "") or "").strip()
        return error or _tr(
            "histogram.delete_failed.reported_failure",
            "The delete pipeline reported a failure.",
        )

    def _on_delete_marked(self, _handle, _event, _args):
        if not self._has_any_mark() or self._marked_count <= 0:
            return

        if self._panel_selection_mask is None:
            return

        self._apply_scene_selection_mask(self._panel_selection_mask)
        error_message = self._execute_delete_pipeline()
        if error_message is not None:
            self._status_hint = _trf(
                "histogram.delete_failed.status",
                "Delete failed: {message}",
                message=error_message,
            )
            if self._handle:
                self._handle.dirty_all()
            try:
                lf.ui.message_dialog(
                    _tr("histogram.delete_failed.title", "Delete Failed"),
                    error_message,
                    style="error",
                )
            except Exception:
                pass
            return

        self._clear_all_marks(clear_scene=False)
        self._scene_generation = -1
        self._refresh()

    def _format_compare_bin_count_text(self) -> str:
        return _trf(
            "histogram.compare_bin_count",
            "{x_count} x {y_count} bins",
            x_count=self._compare_x_bin_count,
            y_count=self._compare_y_bin_count,
        )

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _clamp_int(value, minimum: int, maximum: int) -> int:
        try:
            value = int(round(float(value)))
        except Exception:
            value = minimum
        return max(minimum, min(maximum, value))

    @staticmethod
    def _lower_bin_from_ratio(start_ratio: float, bin_count: int) -> int:
        bin_count = max(1, int(bin_count))
        ratio = min(1.0, max(0.0, float(start_ratio)))
        return min(bin_count - 1, max(0, int(math.floor(ratio * bin_count))))

    @staticmethod
    def _upper_bin_from_ratio(end_ratio: float, bin_count: int) -> int:
        bin_count = max(1, int(bin_count))
        ratio = min(1.0, max(0.0, float(end_ratio)))
        return min(bin_count - 1, max(0, int(math.ceil(ratio * bin_count) - 1)))

    @classmethod
    def _lower_bin_for_value(cls, value: float, axis_min: float, axis_max: float, bin_count: int) -> int:
        span = axis_max - axis_min
        if not math.isfinite(span) or span <= 0.0:
            return max(0, int(bin_count) - 1)
        ratio = (float(value) - axis_min) / span
        return cls._lower_bin_from_ratio(ratio, bin_count)

    @classmethod
    def _upper_bin_for_value(cls, value: float, axis_min: float, axis_max: float, bin_count: int) -> int:
        span = axis_max - axis_min
        if not math.isfinite(span) or span <= 0.0:
            return max(0, int(bin_count) - 1)
        ratio = (float(value) - axis_min) / span
        return cls._upper_bin_from_ratio(ratio, bin_count)

    @staticmethod
    def _upper_bound_is_inclusive(upper: float, axis_max: float) -> bool:
        return upper >= axis_max or math.isclose(upper, axis_max, rel_tol=1e-6, abs_tol=1e-9)

    def _value_range_mask(self, values: lf.Tensor, lower: float, upper: float, axis_max: float) -> lf.Tensor:
        lo = float(min(lower, upper))
        hi = float(max(lower, upper))
        mask = values >= lo
        if self._upper_bound_is_inclusive(hi, axis_max):
            mask = mask & (values <= hi)
        else:
            mask = mask & (values < hi)
        return mask

    @staticmethod
    def _format_value(value: float) -> str:
        abs_value = abs(value)
        if abs_value == 0.0:
            return "0"
        if abs_value >= 10000.0 or abs_value < 1e-4:
            return f"{value:.2e}"
        if abs_value >= 100.0:
            return f"{value:.1f}"
        if abs_value >= 10.0:
            return f"{value:.2f}"
        if abs_value >= 1.0:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def _format_range_text(self, range_min: float, range_max: float) -> str:
        return _trf(
            "histogram.range_value",
            "{min} to {max}",
            min=self._format_value(range_min),
            max=self._format_value(range_max),
        )
