# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Floating retained panel for Gaussian histogram analysis."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

import lichtfeld as lf

from .histogram_support import METRICS, METRIC_BY_ID, histogram_mode_available, histogram_tr
from .types import Panel
from .ui.state import AppState


BIN_COUNT = 56


def _tr(key: str, fallback: str) -> str:
    return histogram_tr(key, fallback)


def _trf(key: str, fallback: str, **kwargs) -> str:
    return _tr(key, fallback).format(**kwargs)


class HistogramPanel(Panel):
    id = "lfs.histogram"
    label = "Histogram"
    space = lf.ui.PanelSpace.FLOATING
    order = 97
    template = "rmlui/histogram_panel.rml"
    size = (860, 660)
    update_interval_ms = 250

    def __init__(self):
        self._doc = None
        self._chart_el = None
        self._close_btn = None
        self._handle = None

        self._metric_id = METRICS[0].id
        self._log_scale_enabled = False
        self._scene_generation = -1
        self._history_generation = -1
        self._last_lang = ""
        self._trainer_state = ""
        self._show_chart = False
        self._empty_title = _tr("histogram.empty.unavailable.title", "Histogram unavailable")
        self._empty_message = _tr(
            "histogram.empty.unavailable.message",
            "Switch to a view or edit scene with Gaussian splats to inspect a distribution.",
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

        self._cached_values: np.ndarray | None = None
        self._visible_mask: np.ndarray | None = None
        self._hist_counts: np.ndarray | None = None
        self._hist_edges: np.ndarray | None = None
        self._applied_selection_values: np.ndarray | None = None

        self._dragging_mark = False
        self._marked_bin_start: int | None = None
        self._marked_bin_end: int | None = None
        self._marked_count = 0
        self._marked_range_text = _tr("histogram.no_marked_range", "No marked range")
        self._marked_count_text = _trf("histogram.gaussian_count", "{count} Gaussians", count="0")
        self._status_hint = _tr(
            "histogram.status_drag_delete",
            "Left-drag across the histogram to mark a range, then delete it.",
        )
        self._selection_left_style = "0%"
        self._selection_width_style = "0%"

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
        model.bind_func("log_scale_label", lambda: _tr("histogram.log_scale", "Log Scale"))
        model.bind_func("samples_label", lambda: _tr("histogram.samples", "Samples"))
        model.bind_func("range_label", lambda: _tr("histogram.range", "Range"))
        model.bind_func("mean_label", lambda: _tr("histogram.mean", "Mean"))
        model.bind_func("median_label", lambda: _tr("histogram.median", "Median"))
        model.bind_func("p95_label", lambda: _tr("histogram.p95", "P95"))
        model.bind_func("peak_bin_label", lambda: _tr("histogram.peak_bin", "Peak Bin"))
        model.bind_func("bin_count_text", lambda: _trf("histogram.bin_count", "{count} bins", count=BIN_COUNT))
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
        model.bind_func("show_selection_overlay", self._has_marked_range)
        model.bind_func("selection_left_style", lambda: self._selection_left_style)
        model.bind_func("selection_width_style", lambda: self._selection_width_style)
        model.bind_func("marked_range_text", lambda: self._marked_range_text)
        model.bind_func("marked_count_text", lambda: self._marked_count_text)
        model.bind_func("status_hint", lambda: self._status_hint)
        model.bind_func("undo_enabled", self._can_undo)
        model.bind_func("redo_enabled", self._can_redo)
        model.bind_func("clear_enabled", self._has_marked_range)
        model.bind_func("delete_enabled", lambda: self._has_marked_range() and self._marked_count > 0)
        model.bind("metric_id", lambda: self._metric_id, self._set_metric_id)
        model.bind("log_scale_enabled", lambda: self._log_scale_enabled, self._set_log_scale_enabled)
        model.bind_event("undo_history", self._on_undo_history)
        model.bind_event("redo_history", self._on_redo_history)
        model.bind_event("clear_mark", self._on_clear_mark)
        model.bind_event("delete_marked", self._on_delete_marked)
        model.bind_record_list("metric_options")
        model.bind_record_list("bins")

        self._handle = model.get_handle()
        self._rebuild_metric_options()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._chart_el = doc.get_element_by_id("histogram-bars")
        self._close_btn = doc.get_element_by_id("close-btn")
        if self._chart_el:
            self._chart_el.add_event_listener("mousedown", self._on_chart_mousedown)
        if self._close_btn:
            self._close_btn.add_event_listener("click", self._on_close_click)
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

        if not histogram_mode_available():
            lf.ui.set_panel_enabled(self.id, False)
            return False

        scene_generation = lf.get_scene_generation()
        history_generation = self._history_generation_value()
        current_lang = lf.ui.get_current_language()
        trainer_state = AppState.trainer_state.value
        if (scene_generation == self._scene_generation and
                history_generation == self._history_generation and
                trainer_state == self._trainer_state and
                current_lang == self._last_lang):
            return False

        self._scene_generation = scene_generation
        self._history_generation = history_generation
        self._last_lang = current_lang
        self._trainer_state = trainer_state
        self._rebuild_metric_options()
        self._refresh()
        return True

    def on_scene_changed(self, doc):
        del doc
        self._scene_generation = -1

    def on_unmount(self, doc):
        self._clear_owned_scene_selection()
        self._doc = None
        self._chart_el = None
        self._close_btn = None
        self._handle = None
        doc.remove_data_model("histogram_panel")

    def _set_metric_id(self, value):
        metric_id = str(value)
        if metric_id not in METRIC_BY_ID or metric_id == self._metric_id:
            return
        self._clear_marked_range(clear_scene=True)
        self._metric_id = metric_id
        self._rebuild_metric_options()
        self._refresh()

    def _set_log_scale_enabled(self, value):
        enabled = self._coerce_bool(value)
        if enabled == self._log_scale_enabled:
            return
        self._log_scale_enabled = enabled
        self._update_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _rebuild_metric_options(self):
        if not self._handle:
            return
        self._handle.update_record_list(
            "metric_options",
            [{"value": metric.id, "label": metric.label()} for metric in METRICS],
        )

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

        values = self._extract_metric_values(scene, model)
        if values is None:
            self._set_empty(
                _tr("histogram.empty.metric_unavailable.title", "Metric unavailable"),
                _tr(
                    "histogram.empty.metric_unavailable.message",
                    "The selected metric could not be read from the combined Gaussian model.",
                ),
            )
            return

        visible_mask = self._extract_visible_mask(model, values.shape[0])
        visible_values = values[visible_mask]
        finite_values = visible_values[np.isfinite(visible_values)]
        if finite_values.size == 0:
            self._set_empty(
                _tr("histogram.empty.no_visible_values.title", "No visible values"),
                _tr(
                    "histogram.empty.no_visible_values.message",
                    "The selected metric does not contain any visible finite samples to visualize.",
                ),
            )
            return

        metric = METRIC_BY_ID[self._metric_id]
        histogram_min, histogram_max = self._histogram_bounds(finite_values)
        counts, edges = np.histogram(finite_values, bins=BIN_COUNT, range=(histogram_min, histogram_max))
        peak_count = int(counts.max()) if counts.size else 0

        self._cached_values = values
        self._visible_mask = visible_mask
        self._hist_counts = counts.astype(np.int64, copy=False)
        self._hist_edges = edges.astype(np.float32, copy=False)

        self._show_chart = True
        self._sample_count = f"{finite_values.size:,}"
        self._range_text = self._format_range_text(float(finite_values.min()), float(finite_values.max()))
        self._mean_text = self._format_value(float(np.mean(finite_values)))
        self._median_text = self._format_value(float(np.median(finite_values)))
        self._p95_text = self._format_value(float(np.percentile(finite_values, 95.0)))
        self._peak_text = f"{peak_count:,}"
        self._axis_min = self._format_value(float(edges[0]))
        self._axis_max = self._format_value(float(edges[-1]))
        self._summary_text = _trf(
            "histogram.summary",
            "{metric} distribution across {count} Gaussians",
            metric=metric.label(),
            count=f"{finite_values.size:,}",
        )

        if self._has_marked_range():
            self._sync_marked_range(apply_scene=self._scene_selection_is_owned(scene))
        else:
            self._reset_marked_state(clear_scene=False)

        self._update_bin_records()
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
        self._cached_values = None
        self._visible_mask = None
        self._hist_counts = None
        self._hist_edges = None
        self._reset_marked_state(clear_scene=True)
        if self._handle:
            self._handle.update_record_list("bins", [])
            self._handle.dirty_all()

    def _build_bin_records(self, counts: np.ndarray, edges: np.ndarray) -> Iterable[dict[str, object]]:
        display_counts = counts.astype(np.float32, copy=False)
        if self._log_scale_enabled:
            display_counts = np.log1p(display_counts)

        peak = float(max(float(display_counts.max()) if display_counts.size else 0.0, 1.0))
        marked_lo, marked_hi = self._marked_bounds()
        for index, count in enumerate(counts):
            ratio = float(display_counts[index]) / peak
            height_pct = 0.0 if count <= 0 else max(3.0, ratio * 100.0)
            alpha = 0.16 if count <= 0 else (0.34 + ratio * 0.66)
            left = self._format_value(float(edges[index]))
            right = self._format_value(float(edges[index + 1]))
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
                "selected": marked_lo is not None and marked_lo <= index <= marked_hi,
            }

    def _update_bin_records(self):
        if self._handle is None or self._hist_counts is None or self._hist_edges is None:
            return
        self._handle.update_record_list(
            "bins",
            list(self._build_bin_records(self._hist_counts, self._hist_edges)),
        )

    def _extract_metric_values(self, scene, model) -> np.ndarray | None:
        try:
            if self._metric_id == "opacity":
                opacity = self._tensor_to_numpy(model.get_opacity())
                return opacity.reshape(-1)

            if self._metric_id == "distance":
                return self._extract_distance_values(scene, model)

            scaling = self._tensor_to_numpy(model.get_scaling()).reshape(-1, 3)
            if self._metric_id == "scale_x":
                return scaling[:, 0]
            if self._metric_id == "scale_y":
                return scaling[:, 1]
            if self._metric_id == "scale_z":
                return scaling[:, 2]
            if self._metric_id == "scale_max":
                return scaling.max(axis=1)
            return None
        except Exception:
            return None

    def _extract_distance_values(self, scene, model) -> np.ndarray:
        means = self._tensor_to_numpy(model.get_means()).reshape(-1, 3)
        if means.size == 0:
            return np.empty((0,), dtype=np.float32)

        world_means = self._world_space_means(scene, means)
        if world_means is None:
            world_means = means
        return self._distance_from_positions(scene, world_means)

    def _world_space_means(self, scene, means: np.ndarray) -> np.ndarray | None:
        nodes = self._visible_splat_nodes(scene)
        if not nodes:
            return None

        world_means = means.astype(np.float32, copy=True)
        offset = 0

        for node in nodes:
            count = int(getattr(node, "gaussian_count", 0) or 0)
            if count <= 0:
                continue

            next_offset = offset + count
            if next_offset > world_means.shape[0]:
                return None

            transform = np.asarray(node.world_transform, dtype=np.float32).reshape(4, 4)
            world_means[offset:next_offset] = (
                world_means[offset:next_offset] @ transform[:3, :3].T
            ) + transform[:3, 3].reshape(1, 3)
            offset = next_offset

        if offset != world_means.shape[0]:
            return None
        return world_means

    def _distance_from_positions(self, scene, positions: np.ndarray) -> np.ndarray:
        finite_rows = np.all(np.isfinite(positions), axis=1)
        if not np.any(finite_rows):
            return np.full((positions.shape[0],), np.nan, dtype=np.float32)

        center = self._resolve_scene_center(scene, positions, finite_rows)
        distances = np.full((positions.shape[0],), np.nan, dtype=np.float32)
        distances[finite_rows] = np.linalg.norm(
            positions[finite_rows] - center.reshape(1, 3),
            axis=1,
        ).astype(np.float32, copy=False)
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

    def _resolve_scene_center(self, scene, means: np.ndarray, finite_rows: np.ndarray) -> np.ndarray:
        try:
            center = self._tensor_to_numpy(scene.scene_center).reshape(-1)
            if center.size == 3 and np.all(np.isfinite(center)):
                return center.astype(np.float32, copy=False)
        except Exception:
            pass

        finite_means = means[finite_rows]
        if finite_means.size == 0:
            return np.zeros((3,), dtype=np.float32)
        return finite_means.mean(axis=0, dtype=np.float32)

    @staticmethod
    def _tensor_to_numpy(tensor, dtype=np.float32) -> np.ndarray:
        cpu_tensor = tensor.contiguous().cpu()
        values = np.asarray(cpu_tensor.numpy(copy=False))
        if dtype is None:
            return values
        return values.astype(dtype, copy=False)

    def _extract_visible_mask(self, model, value_count: int) -> np.ndarray:
        try:
            if bool(model.has_deleted_mask()):
                deleted = self._tensor_to_numpy(model.deleted, dtype=np.bool_).reshape(-1)
                if deleted.size == value_count:
                    return np.logical_not(deleted)
        except Exception:
            pass
        return np.ones((value_count,), dtype=np.bool_)

    def _histogram_bounds(self, values: np.ndarray) -> tuple[float, float]:
        if self._metric_id == "opacity":
            return 0.0, 1.0

        lo = float(values.min())
        hi = float(values.max())
        if not math.isfinite(lo) or not math.isfinite(hi):
            return 0.0, 1.0

        if math.isclose(lo, hi, rel_tol=1e-6, abs_tol=1e-9):
            padding = max(abs(lo) * 0.05, 1e-3)
            return lo - padding, hi + padding

        return lo, hi

    def _on_chart_mousedown(self, event):
        if not self._show_chart or self._chart_el is None or self._hist_edges is None:
            return
        if int(event.get_parameter("button", "0")) != 0:
            return

        bin_index = self._bin_index_for_mouse_x(self._event_mouse_x(event))
        self._dragging_mark = True
        self._marked_bin_start = bin_index
        self._marked_bin_end = bin_index
        self._sync_marked_range(apply_scene=False)
        event.stop_propagation()

    def _on_document_mousemove(self, event):
        if not self._dragging_mark or self._chart_el is None:
            return

        bin_index = self._bin_index_for_mouse_x(self._event_mouse_x(event))
        if bin_index == self._marked_bin_end:
            return
        self._marked_bin_end = bin_index
        self._sync_marked_range(apply_scene=False)
        event.stop_propagation()

    def _on_document_mouseup(self, event):
        if not self._dragging_mark:
            return

        self._dragging_mark = False
        self._sync_marked_range(apply_scene=True)
        event.stop_propagation()

    def _event_mouse_x(self, event) -> float:
        try:
            return float(event.get_parameter("mouse_x", "0"))
        except Exception:
            return 0.0

    def _bin_index_for_mouse_x(self, mouse_x: float) -> int:
        if self._chart_el is None:
            return 0
        left = float(self._chart_el.absolute_left)
        width = max(float(self._chart_el.absolute_width), 1.0)
        norm = min(1.0, max(0.0, (mouse_x - left) / width))
        return min(BIN_COUNT - 1, max(0, int(math.floor(norm * BIN_COUNT))))

    def _marked_bounds(self) -> tuple[int | None, int | None]:
        if self._marked_bin_start is None or self._marked_bin_end is None:
            return None, None
        lo = max(0, min(self._marked_bin_start, self._marked_bin_end))
        hi = min(BIN_COUNT - 1, max(self._marked_bin_start, self._marked_bin_end))
        return lo, hi

    def _has_marked_range(self) -> bool:
        lo, hi = self._marked_bounds()
        return lo is not None and hi is not None

    def _sync_marked_range(self, apply_scene: bool):
        if self._hist_counts is None or self._hist_edges is None:
            self._reset_marked_state(clear_scene=apply_scene)
            return

        lo, hi = self._marked_bounds()
        if lo is None or hi is None:
            self._reset_marked_state(clear_scene=apply_scene)
            return

        left_ratio = lo / BIN_COUNT
        width_ratio = (hi + 1) / BIN_COUNT - left_ratio
        self._selection_left_style = f"{left_ratio * 100.0:.2f}%"
        self._selection_width_style = f"{max(width_ratio * 100.0, 1.0):.2f}%"

        range_min = float(self._hist_edges[lo])
        range_max = float(self._hist_edges[hi + 1])
        selection_mask = self._selection_mask_for_range(range_min, range_max, hi == BIN_COUNT - 1)
        self._marked_count = int(selection_mask.sum()) if selection_mask is not None else 0
        self._marked_range_text = self._format_range_text(range_min, range_max)
        self._marked_count_text = _trf(
            "histogram.gaussian_count",
            "{count} Gaussians",
            count=f"{self._marked_count:,}",
        )
        self._status_hint = _tr(
            "histogram.status_selection",
            "Marked range becomes the active Gaussian selection.",
        )

        self._update_bin_records()
        if apply_scene:
            self._apply_scene_selection(range_min, range_max, hi == BIN_COUNT - 1)
        if self._handle:
            self._handle.dirty_all()

    def _reset_marked_state(self, clear_scene: bool):
        self._marked_bin_start = None
        self._marked_bin_end = None
        self._marked_count = 0
        self._marked_range_text = _tr("histogram.no_marked_range", "No marked range")
        self._marked_count_text = _trf("histogram.gaussian_count", "{count} Gaussians", count="0")
        self._status_hint = _tr(
            "histogram.status_drag_delete",
            "Left-drag across the histogram to mark a range, then delete it.",
        )
        self._selection_left_style = "0%"
        self._selection_width_style = "0%"
        if clear_scene:
            self._clear_owned_scene_selection()
        else:
            self._applied_selection_values = None

    def _clear_marked_range(self, clear_scene: bool):
        self._reset_marked_state(clear_scene=clear_scene)
        self._update_bin_records()
        if self._handle:
            self._handle.dirty_all()

    def _selection_mask_for_range(
        self,
        range_min: float,
        range_max: float,
        include_upper_bound: bool,
    ) -> np.ndarray | None:
        if self._cached_values is None:
            return None

        values = self._cached_values
        mask = np.isfinite(values)
        if self._visible_mask is not None and self._visible_mask.size == values.size:
            mask &= self._visible_mask
        mask &= values >= range_min
        if include_upper_bound:
            mask &= values <= range_max
        else:
            mask &= values < range_max
        return mask

    def _apply_scene_selection(self, range_min: float, range_max: float, include_upper_bound: bool):
        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            self._applied_selection_values = None
            return

        mask = self._selection_mask_for_range(range_min, range_max, include_upper_bound)
        if mask is None or not np.any(mask):
            scene.clear_selection()
            self._applied_selection_values = None
            return

        group_id = 1
        try:
            group_id = max(int(scene.active_selection_group or 0), 1)
        except Exception:
            pass

        selection_values = np.zeros((mask.size,), dtype=np.uint8)
        selection_values[mask] = np.uint8(group_id)
        selection_tensor = lf.Tensor.from_numpy(selection_values, copy=True)
        scene.set_selection_mask(selection_tensor)
        self._applied_selection_values = selection_values

    def _clear_owned_scene_selection(self):
        scene = lf.get_scene()
        if scene is not None and scene.is_valid() and self._scene_selection_is_owned(scene):
            scene.clear_selection()
        self._applied_selection_values = None

    def _scene_selection_is_owned(self, scene) -> bool:
        if self._applied_selection_values is None:
            return False

        current_values = self._scene_selection_values(scene)
        if current_values is None:
            return False

        return (
            current_values.shape == self._applied_selection_values.shape and
            np.array_equal(current_values, self._applied_selection_values)
        )

    def _scene_selection_values(self, scene) -> np.ndarray | None:
        try:
            selection_mask = scene.selection_mask
        except Exception:
            selection_mask = None

        if selection_mask is None:
            return None

        try:
            return self._tensor_to_numpy(selection_mask, dtype=np.uint8).reshape(-1)
        except Exception:
            return None

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

    def _redo_tooltip(self) -> str:
        name = self._history_name(lf.undo.get_redo_name) if self._can_redo() else ""
        if name:
            return _trf("histogram.redo_named", "Redo: {name}", name=name)
        return _tr("histogram.redo", "Redo")

    def _on_clear_mark(self, _handle, _event, _args):
        self._clear_marked_range(clear_scene=True)

    def _on_close_click(self, event):
        self._clear_owned_scene_selection()
        try:
            lf.selection.clear_preview()
        except Exception:
            pass
        event.stop_propagation()
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
        if not self._has_marked_range() or self._marked_count <= 0:
            return

        lo, hi = self._marked_bounds()
        if lo is None or hi is None or self._hist_edges is None:
            return

        self._apply_scene_selection(float(self._hist_edges[lo]), float(self._hist_edges[hi + 1]), hi == BIN_COUNT - 1)
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

        self._clear_marked_range(clear_scene=False)
        self._scene_generation = -1
        self._refresh()

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

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
