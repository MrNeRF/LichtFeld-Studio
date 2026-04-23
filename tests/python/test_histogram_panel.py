# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for histogram metric extraction."""

from types import SimpleNamespace

import pytest

from lfs_plugins.rml_keys import KI_A, KI_DELETE, KI_I

RML_KM_CTRL = 1 << 0
RML_KM_SHIFT = 1 << 1
RML_KM_META = 1 << 3


class _ModelStub:
    def __init__(self, lf, means, scaling, opacity=None):
        self._means = lf.Tensor.from_numpy(means)
        self._scaling = lf.Tensor.from_numpy(scaling)
        self._opacity = None if opacity is None else lf.Tensor.from_numpy(opacity)

    def get_means(self):
        return self._means

    def get_scaling(self):
        return self._scaling

    def get_opacity(self):
        if self._opacity is None:
            raise AssertionError("Opacity should not be requested in this test")
        return self._opacity


class _SceneSelectionStub:
    def __init__(self, selection_mask=None):
        self.selection_mask = selection_mask
        self.clear_calls = 0

    def is_valid(self):
        return True

    def set_selection_mask(self, mask):
        self.selection_mask = (mask.reshape([-1]) != 0).contiguous()

    def clear_selection(self):
        self.selection_mask = None
        self.clear_calls += 1


class _KeyEventStub:
    def __init__(
        self,
        key_identifier: int,
        *,
        ctrl: bool = False,
        meta: bool = False,
        shift: bool = False,
        modifiers: int | None = None,
    ):
        self._params = {"key_identifier": str(key_identifier)}
        if modifiers is not None:
            self._params["modifiers"] = str(modifiers)
        self._bools = {"ctrl_key": ctrl, "meta_key": meta, "shift_key": shift}
        self.stopped = False

    def get_parameter(self, name, default=""):
        return self._params.get(name, default)

    def get_bool_parameter(self, name, default=False):
        return self._bools.get(name, default)

    def stop_propagation(self):
        self.stopped = True


class _MouseEventStub:
    def __init__(
        self,
        *,
        mouse_x: float,
        mouse_y: float = 0.0,
        button: int = 0,
        ctrl: bool = False,
        meta: bool = False,
        shift: bool = False,
        modifiers: int | None = None,
    ):
        self._params = {
            "mouse_x": str(mouse_x),
            "mouse_y": str(mouse_y),
            "button": str(button),
        }
        if modifiers is not None:
            self._params["modifiers"] = str(modifiers)
        self._bools = {"ctrl_key": ctrl, "meta_key": meta, "shift_key": shift}
        self.stopped = False

    def get_parameter(self, name, default=""):
        return self._params.get(name, default)

    def get_bool_parameter(self, name, default=False):
        return self._bools.get(name, default)

    def stop_propagation(self):
        self.stopped = True


def _translation_matrix(tx: float, ty: float, tz: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


@pytest.fixture
def histogram_panel_module():
    from lfs_plugins import histogram_panel

    return histogram_panel


def test_histogram_metrics_include_positions_volume_anisotropy_and_erank(histogram_panel_module):
    metric_ids = {metric.id for metric in histogram_panel_module.METRICS}

    assert {"position_x", "position_y", "position_z", "volume", "anisotropy", "erank", "world_distance"} <= metric_ids


def test_histogram_world_distance_metric_measures_from_origin(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    # Two Gaussians at local positions (1,0,0) and (0,3,4).
    # With a translation of (2,0,0) applied the world positions become (3,0,0) and (2,3,4).
    # Expected distances from origin: 3.0 and sqrt(4+9+16)=sqrt(29).
    model = _ModelStub(
        lf,
        numpy.array([[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=numpy.float32),
    )

    splat_type = getattr(getattr(lf, "NodeType", None), "SPLAT", None)
    if splat_type is None:
        splat_type = lf.scene.NodeType.SPLAT

    scene = SimpleNamespace(
        get_nodes=lambda: [
            SimpleNamespace(
                id=1,
                parent_id=-1,
                visible=True,
                type=splat_type,
                gaussian_count=2,
                world_transform=_translation_matrix(2.0, 0.0, 0.0),
            )
        ]
    )

    panel._metric_id = "world_distance"
    result = panel._extract_metric_values(scene, model).cpu().numpy()
    numpy.testing.assert_allclose(
        result,
        numpy.array([3.0, numpy.sqrt(29.0)], dtype=numpy.float32),
        rtol=1e-5,
    )


def test_histogram_position_metrics_use_world_space_means(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    model = _ModelStub(
        lf,
        numpy.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.5]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]], dtype=numpy.float32),
    )

    splat_type = getattr(getattr(lf, "NodeType", None), "SPLAT", None)
    if splat_type is None:
        splat_type = lf.scene.NodeType.SPLAT

    scene = SimpleNamespace(
        get_nodes=lambda: [
            SimpleNamespace(
                id=7,
                parent_id=-1,
                visible=True,
                type=splat_type,
                gaussian_count=2,
                world_transform=_translation_matrix(10.0, -3.0, 0.5),
            )
        ]
    )

    panel._metric_id = "position_x"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([11.0, 8.0], dtype=numpy.float32),
    )

    panel._metric_id = "position_y"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([-1.0, -2.5], dtype=numpy.float32),
    )

    panel._metric_id = "position_z"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([3.5, 5.0], dtype=numpy.float32),
    )


def test_histogram_volume_anisotropy_and_erank_metrics_match_gaussian_scales(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]], dtype=numpy.float32),
    )
    scene = SimpleNamespace(get_nodes=lambda: [])

    panel._metric_id = "volume"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([4.0 * numpy.pi / 3.0, 32.0 * numpy.pi / 3.0], dtype=numpy.float32),
        rtol=1e-6,
    )

    panel._metric_id = "anisotropy"
    anisotropy = panel._extract_metric_values(scene, model).cpu().numpy()
    numpy.testing.assert_allclose(anisotropy, numpy.array([1.0, 4.0], dtype=numpy.float32), rtol=1e-6)
    assert panel._histogram_bounds(lf.Tensor.from_numpy(anisotropy)) == (1.0, 4.0)

    panel._metric_id = "erank"
    erank = panel._extract_metric_values(scene, model).cpu().numpy()
    numpy.testing.assert_allclose(erank, numpy.array([3.0, 1.9503675], dtype=numpy.float32), rtol=1e-6)
    assert panel._histogram_bounds(lf.Tensor.from_numpy(erank)) == (1.0, 3.0)


def test_compare_heatmap_reuses_primary_metric_and_selects_joint_cells(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id = "volume"
    panel._compare_metric_id = "opacity"

    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [1.5, 1.5, 1.5], [3.0, 3.0, 3.0]], dtype=numpy.float32),
        opacity=numpy.array([0.05, 0.55, 0.95], dtype=numpy.float32),
    )
    scene = SimpleNamespace(get_nodes=lambda: [])

    primary_values = panel._extract_metric_values(scene, model, panel._metric_id)
    panel._refresh_compare(scene, model, primary_values, None)

    assert panel._show_compare_card is True
    assert panel._show_compare_chart is True
    assert panel._compare_x_metric_label == "Volume"
    assert panel._compare_y_metric_label == "Opacity"
    assert panel._compare_counts is not None
    assert sum(panel._compare_counts) == 3
    assert len(panel._compare_counts) == histogram_panel_module.DEFAULT_COMPARE_X_BIN_COUNT ** 2

    x_bins = panel._compare_x_bin_indices.cpu().tolist()
    y_bins = panel._compare_y_bin_indices.cpu().tolist()
    mask = panel._selection_mask_for_compare_value_bounds(
        panel._compare_x_edges[x_bins[0]],
        panel._compare_x_edges[x_bins[0] + 1],
        panel._compare_y_edges[y_bins[0]],
        panel._compare_y_edges[y_bins[0] + 1],
    )
    numpy.testing.assert_array_equal(mask.cpu().numpy(), numpy.array([True, False, False]))


def test_histogram_bin_slider_rebins_and_preserves_marked_range(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35, 0.65, 0.85], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    panel._marked_bin_start = 4
    panel._marked_bin_end = 7
    panel._sync_marked_range(apply_scene=False)

    panel._set_histogram_bin_count(32)

    assert len(panel._hist_counts) == 32
    assert panel._marked_bounds() == (8, 15)
    assert panel._marked_count == 1
    assert panel._peak_text == "1"


def test_histogram_rebin_does_not_expand_selected_samples(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"

    values = lf.Tensor.from_numpy(numpy.array([0.07, 0.14, 0.40], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    panel._marked_bin_start = 1
    panel._marked_bin_end = 1
    panel._sync_marked_range(apply_scene=False)

    assert panel._marked_count == 1
    assert panel._marked_range_text == "0.0625 to 0.125"

    panel._set_histogram_bin_count(17)

    assert panel._marked_count == 1
    assert panel._marked_range_text == "0.0625 to 0.125"


def test_histogram_drag_can_expand_across_multiple_bins(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35, 0.65, 0.85], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    panel._dragging_mark = True
    panel._marked_bin_start = 1
    panel._marked_bin_end = 5
    panel._sync_marked_range(apply_scene=False)

    assert panel._marked_bounds() == (1, 5)
    assert panel._marked_count == 2


def test_histogram_selection_geometry_accounts_for_bar_gaps(histogram_panel_module):
    panel = histogram_panel_module.HistogramPanel()
    panel._histogram_bin_count = 4
    panel._chart_el = SimpleNamespace(absolute_left=10.0, absolute_width=100.0)

    left, width = panel._histogram_selection_geometry(1, 2)

    assert left == pytest.approx(25.5)
    assert width == pytest.approx(49.0)
    assert panel._bin_index_for_mouse_x(40.0) == 1
    assert panel._bin_index_for_mouse_x(65.0) == 2


def test_compare_bin_sliders_support_rectangular_grids(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id = "volume"
    panel._compare_metric_id = "opacity"

    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [1.5, 1.5, 1.5], [3.0, 3.0, 3.0]], dtype=numpy.float32),
        opacity=numpy.array([0.05, 0.55, 0.95], dtype=numpy.float32),
    )
    scene = SimpleNamespace(get_nodes=lambda: [])

    primary_values = panel._extract_metric_values(scene, model, panel._metric_id)
    panel._refresh_compare(scene, model, primary_values, None)
    panel._set_compare_x_bin_count(12)
    panel._set_compare_y_bin_count(9)

    records = list(panel._build_compare_bin_records())

    assert len(panel._compare_counts) == 12 * 9
    assert len(records) == 12 * 9
    assert len(panel._compare_x_edges) == 13
    assert len(panel._compare_y_edges) == 10
    assert "width: 8.3333%;" in records[0]["style_attr"]
    assert "height: 11.1111%;" in records[0]["style_attr"]
    assert panel._format_compare_bin_count_text() == "12 x 9 bins"


def test_compare_rebin_does_not_expand_selected_samples(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id = "scale_x"
    panel._compare_metric_id = "opacity"

    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=numpy.float32),
        numpy.array([[0.06, 1.0, 1.0], [0.12, 1.0, 1.0], [0.40, 1.0, 1.0]], dtype=numpy.float32),
        opacity=numpy.array([0.06, 0.12, 0.40], dtype=numpy.float32),
    )
    scene = SimpleNamespace(get_nodes=lambda: [])

    primary_values = panel._extract_metric_values(scene, model, panel._metric_id)
    panel._refresh_compare(scene, model, primary_values, None)

    x_bin = int(panel._compare_x_bin_indices.cpu().tolist()[0])
    y_bin = int(panel._compare_y_bin_indices.cpu().tolist()[0])
    panel._compare_mark_start = (x_bin, y_bin)
    panel._compare_mark_end = (x_bin, y_bin)
    panel._sync_compare_mark(apply_scene=False)

    assert panel._marked_count == 1

    panel._set_compare_x_bin_count(19)
    panel._set_compare_y_bin_count(19)

    assert panel._marked_count == 1


def test_compare_drag_can_expand_across_multiple_bins(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id = "scale_x"
    panel._compare_metric_id = "opacity"

    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=numpy.float32),
        numpy.array([[0.06, 1.0, 1.0], [0.12, 1.0, 1.0], [0.40, 1.0, 1.0]], dtype=numpy.float32),
        opacity=numpy.array([0.06, 0.12, 0.40], dtype=numpy.float32),
    )
    scene = SimpleNamespace(get_nodes=lambda: [])

    primary_values = panel._extract_metric_values(scene, model, panel._metric_id)
    panel._refresh_compare(scene, model, primary_values, None)

    x_bins = panel._compare_x_bin_indices.cpu().tolist()
    y_bins = panel._compare_y_bin_indices.cpu().tolist()
    panel._dragging_compare_mark = True
    panel._compare_mark_start = (min(x_bins[0], x_bins[1]), min(y_bins[0], y_bins[1]))
    panel._compare_mark_end = (max(x_bins[0], x_bins[1]), max(y_bins[0], y_bins[1]))
    panel._sync_compare_mark(apply_scene=False)

    assert panel._marked_count == 2


def test_histogram_ctrl_a_selects_full_domain(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35, 0.65, 0.85], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    event = _KeyEventStub(KI_A, ctrl=True)
    panel._on_keydown(event)

    assert event.stopped is True
    assert panel._marked_bounds() == (0, 15)
    assert panel._marked_count == 5
    assert panel._selected_histogram_bins == set(range(16))
    assert panel._histogram_overlay_bounds == (0, 15)
    panel._rebuild_histogram_from_cache()
    assert panel._histogram_overlay_bounds == (0, 15)
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.ones(5, dtype=bool))


def test_histogram_ctrl_i_inverts_current_panel_selection(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35, 0.65, 0.85], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._marked_bin_start = 0
    panel._marked_bin_end = 0
    panel._sync_marked_range(apply_scene=True)

    event = _KeyEventStub(KI_I, ctrl=True)
    panel._on_keydown(event)

    assert event.stopped is True
    assert panel._marked_count == 4
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._has_marked_range() is False
    numpy.testing.assert_array_equal(
        scene.selection_mask.cpu().numpy(),
        numpy.array([False, True, True, True, True], dtype=bool),
    )


def test_histogram_ctrl_i_clears_full_selection_immediately(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35, 0.65, 0.85], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._on_keydown(_KeyEventStub(KI_A, modifiers=RML_KM_CTRL))
    panel._on_keydown(_KeyEventStub(KI_I, modifiers=RML_KM_CTRL))

    assert panel._marked_count == 0
    assert panel._panel_selection_mask is None
    assert panel._selected_histogram_bins == set()
    assert panel._histogram_overlay_bounds is None
    assert panel._has_marked_range() is False
    assert all(not record["selected"] for record in panel._build_bin_records(panel._hist_counts, panel._hist_edges))
    assert scene.selection_mask is None
    assert scene.clear_calls == 1


def test_compare_ctrl_a_selects_full_grid(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id = "scale_x"
    panel._compare_metric_id = "opacity"

    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=numpy.float32),
        numpy.array([[0.06, 1.0, 1.0], [0.12, 1.0, 1.0], [0.40, 1.0, 1.0]], dtype=numpy.float32),
        opacity=numpy.array([0.06, 0.12, 0.40], dtype=numpy.float32),
    )
    scene_data = SimpleNamespace(get_nodes=lambda: [])
    primary_values = panel._extract_metric_values(scene_data, model, panel._metric_id)
    panel._refresh_compare(scene_data, model, primary_values, None)

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    event = _KeyEventStub(KI_A, ctrl=True)
    panel._on_keydown(event)

    assert event.stopped is True
    assert panel._has_compare_marked_range() is True
    assert panel._compare_marked_bounds() == (
        0,
        panel._compare_x_bin_count - 1,
        0,
        panel._compare_y_bin_count - 1,
    )
    assert len(panel._selected_compare_cells) == panel._compare_x_bin_count * panel._compare_y_bin_count
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.ones(3, dtype=bool))


def test_histogram_delete_shortcut_deletes_panel_selection(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)
    monkeypatch.setattr(panel, "_execute_delete_pipeline", lambda: None)
    monkeypatch.setattr(panel, "_refresh", lambda: None)

    panel._marked_bin_start = 0
    panel._marked_bin_end = 0
    panel._sync_marked_range(apply_scene=False)

    event = _KeyEventStub(KI_DELETE)
    panel._on_keydown(event)

    assert event.stopped is True
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, False], dtype=bool))
    assert panel._has_any_mark() is False


def test_histogram_shift_drag_adds_to_existing_selection(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._marked_bin_start = 0
    panel._marked_bin_end = 0
    panel._sync_marked_range(apply_scene=True)

    start_x = 5 * 10.0 + 1.0
    end_x = 5 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=start_x, shift=True))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=end_x, shift=True))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=end_x, shift=True))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._marked_count == 2
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (5, 5)


def test_histogram_shift_drag_adds_with_modifier_bitmask(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._marked_bin_start = 0
    panel._marked_bin_end = 0
    panel._sync_marked_range(apply_scene=True)

    x = 5 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, modifiers=RML_KM_SHIFT))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x, modifiers=RML_KM_SHIFT))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, modifiers=RML_KM_SHIFT))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (5, 5)


def test_histogram_shift_drag_selection_survives_rebuild(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._marked_bin_start = 0
    panel._marked_bin_end = 0
    panel._sync_marked_range(apply_scene=True)

    x = 5 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, shift=True))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x, shift=True))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, shift=True))

    panel._rebuild_histogram_from_cache()

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (5, 5)


def test_histogram_ctrl_drag_subtracts_from_existing_selection(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    select_all = _KeyEventStub(KI_A, ctrl=True)
    panel._on_keydown(select_all)

    x = 2 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, ctrl=True))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x, ctrl=True))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, ctrl=True))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._marked_count == 2
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (2, 2)


def test_histogram_ctrl_drag_selection_survives_rebuild(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._on_keydown(_KeyEventStub(KI_A, ctrl=True))

    x = 2 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, ctrl=True))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x, ctrl=True))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, ctrl=True))

    panel._rebuild_histogram_from_cache()

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (2, 2)


def test_histogram_ctrl_drag_subtracts_with_modifier_bitmask(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._on_keydown(_KeyEventStub(KI_A, modifiers=RML_KM_CTRL))

    x = 2 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, modifiers=RML_KM_CTRL))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x, modifiers=RML_KM_CTRL))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, modifiers=RML_KM_CTRL))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (2, 2)


def test_histogram_shift_drag_selection_uses_modifier_seen_on_mouseup(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._marked_bin_start = 0
    panel._marked_bin_end = 0
    panel._sync_marked_range(apply_scene=True)

    x = 5 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, shift=True))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (5, 5)


def test_histogram_ctrl_drag_subtracts_when_modifier_is_seen_on_mouseup(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._on_keydown(_KeyEventStub(KI_A, ctrl=True))

    x = 2 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, ctrl=True))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (2, 2)


def test_histogram_click_selected_bar_deselects_only_that_bar(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub()
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._marked_bin_start = 0
    panel._marked_bin_end = 0
    panel._sync_marked_range(apply_scene=True)

    x = 5 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, shift=True))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x, shift=True))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, shift=True))

    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, False], dtype=bool))
    assert panel._selected_histogram_bins == {0}
    assert panel._histogram_overlay_bounds == (0, 0)


def test_histogram_owned_modifier_selection_survives_two_follow_up_updates(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)
    panel._scene_generation = 0
    panel._history_generation = 0
    panel._last_lang = "en"
    panel._trainer_state = ""

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    scene = _SceneSelectionStub(lf.Tensor.from_numpy(numpy.array([True, False, False], dtype=bool)))
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._commit_histogram_mask_selection(
        lf.Tensor.from_numpy(numpy.array([True, False, True], dtype=bool)),
        apply_scene=False,
        overlay_bounds=(5, 5),
    )
    panel._selection_owned = True
    panel._pending_selection_commit = 2

    scene_generations = iter([1, 1])
    history_generations = iter([0, 1])
    monkeypatch.setattr(lf, "get_scene_generation", lambda: next(scene_generations))
    monkeypatch.setattr(panel, "_history_generation_value", lambda: next(history_generations))
    monkeypatch.setattr(panel, "_sync_panel_space_state", lambda: False)
    monkeypatch.setattr(lf.ui, "get_current_language", lambda: "en")

    panel.on_update(None)
    panel.on_update(None)

    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (5, 5)


def test_histogram_panel_can_toggle_between_bottom_dock_and_floating(histogram_panel_module, lf):
    panel = histogram_panel_module.HistogramPanel()
    state = {"space": lf.ui.PanelSpace.BOTTOM_DOCK}

    def _get_panel(panel_id):
        assert panel_id == panel.id
        return SimpleNamespace(space=state["space"])

    def _set_panel_space(panel_id, space):
        assert panel_id == panel.id
        state["space"] = space
        return True

    original_get_panel = lf.ui.get_panel
    original_set_panel_space = lf.ui.set_panel_space
    try:
        lf.ui.get_panel = _get_panel
        lf.ui.set_panel_space = _set_panel_space

        assert panel._sync_panel_space_state() is False
        assert panel._is_floating is False
        assert panel._dock_toggle_label() == "Undock"

        panel._on_toggle_dock_mode(None, None, None)

        assert state["space"] == lf.ui.PanelSpace.FLOATING
        assert panel._is_floating is True
        assert panel._dock_toggle_label() == "Dock"

        panel._on_toggle_dock_mode(None, None, None)

        assert state["space"] == lf.ui.PanelSpace.BOTTOM_DOCK
        assert panel._is_floating is False
    finally:
        lf.ui.get_panel = original_get_panel
        lf.ui.set_panel_space = original_set_panel_space
