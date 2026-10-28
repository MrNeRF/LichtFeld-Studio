# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for histogram metric extraction."""

from pathlib import Path
from types import SimpleNamespace

import pytest


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
        self.selection_history = []
        self._preview_before = None
        self._preview_active = False
        self.clear_calls = 0
        self.preview_mask_calls = 0
        self.commit_preview_calls = 0
        self.cancel_preview_calls = 0

    def is_valid(self):
        return True

    def set_selection_mask(self, mask):
        self.selection_mask = (mask.reshape([-1]) != 0).contiguous()

    def preview_selection_mask(self, mask):
        if not self._preview_active:
            self._preview_before = self.selection_mask
            self._preview_active = True
        self.preview_mask_calls += 1
        self.set_selection_mask(mask)

    def commit_selection_preview(self):
        self.commit_preview_calls += 1
        self.selection_history.append(self._preview_before)
        self._preview_before = None
        self._preview_active = False

    def cancel_selection_preview(self):
        self.cancel_preview_calls += 1
        self.selection_mask = self._preview_before
        self._preview_before = None
        self._preview_active = False

    def clear_selection(self):
        self.selection_mask = None
        self.clear_calls += 1


class _KeyEventStub:
    def __init__(
        self,
        key_identifier: int,
        *,
        ctrl: bool = False,
        alt: bool = False,
        meta: bool = False,
        shift: bool = False,
        modifiers: int | None = None,
    ):
        self._params = {"key_identifier": str(key_identifier)}
        if modifiers is not None:
            self._params["modifiers"] = str(modifiers)
        self._bools = {"ctrl_key": ctrl, "alt_key": alt, "meta_key": meta, "shift_key": shift}
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
        alt: bool = False,
        meta: bool = False,
        shift: bool = False,
        modifiers: int | None = None,
        wheel_delta: float | None = None,
        wheel_delta_y: float | None = None,
    ):
        self._params = {
            "mouse_x": str(mouse_x),
            "mouse_y": str(mouse_y),
            "button": str(button),
        }
        if modifiers is not None:
            self._params["modifiers"] = str(modifiers)
        if wheel_delta is not None:
            self._params["wheel_delta"] = str(wheel_delta)
        if wheel_delta_y is not None:
            self._params["wheel_delta_y"] = str(wheel_delta_y)
        self._bools = {"ctrl_key": ctrl, "alt_key": alt, "meta_key": meta, "shift_key": shift}
        self.stopped = False

    def get_parameter(self, name, default=""):
        return self._params.get(name, default)

    def get_bool_parameter(self, name, default=False):
        return self._bools.get(name, default)

    def stop_propagation(self):
        self.stopped = True


class _SignalStub:
    def __init__(self):
        self._callbacks = []

    def subscribe(self, callback):
        self._callbacks.append(callback)

        def unsubscribe():
            if callback in self._callbacks:
                self._callbacks.remove(callback)

        return unsubscribe

    def emit(self, value):
        for callback in list(self._callbacks):
            callback(value)


class _UpdateHandleStub:
    def __init__(self):
        self.request_update_count = 0
        self.dirty_all_count = 0
        self.records = {}

    def request_update(self):
        self.request_update_count += 1

    def dirty_all(self):
        self.dirty_all_count += 1

    def update_record_list(self, name, items):
        self.records[name] = list(items)


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


def test_histogram_panel_uses_dirty_update_policy(histogram_panel_module):
    assert histogram_panel_module.HistogramPanel.update_policy == "dirty"
    assert histogram_panel_module.HistogramPanel.update_interval_ms is None


@pytest.mark.parametrize("trainer_state", ["idle", "preparing"])
def test_histogram_mode_available_hides_when_paused(histogram_panel_module, monkeypatch, trainer_state):
    from lfs_plugins import histogram_support as support

    monkeypatch.setattr(
        support.lf.ui, "get_content_type", lambda: "splat_files", raising=False
    )
    monkeypatch.setattr(
        support.lf.ui, "is_point_cloud_forced", lambda: False, raising=False
    )
    monkeypatch.setattr(
        support,
        "RuntimeState",
        SimpleNamespace(trainer_state=SimpleNamespace(value=trainer_state)),
    )

    paused = SimpleNamespace(
        has_scene=True,
        num_gaussians=8,
        is_training=False,
        is_paused=True,
    )
    reset = SimpleNamespace(
        has_scene=True,
        num_gaussians=8,
        is_training=False,
        is_paused=False,
    )

    assert histogram_panel_module.histogram_mode_available(paused) is False
    assert histogram_panel_module.histogram_mode_available(reset) is (trainer_state == "idle")


def test_histogram_panel_requests_update_from_reactive_store(histogram_panel_module, monkeypatch):
    module = histogram_panel_module
    signals = SimpleNamespace(
        scene_generation=_SignalStub(),
        selection_generation=_SignalStub(),
        training_state=_SignalStub(),
        language_generation=_SignalStub(),
    )
    monkeypatch.setattr(module, "RuntimeState", signals)

    panel = module.HistogramPanel()
    panel._handle = _UpdateHandleStub()

    panel._subscribe_reactive_state()
    signals.scene_generation.emit(1)
    signals.selection_generation.emit(2)
    signals.training_state.emit("running")
    signals.language_generation.emit(1)

    assert panel._handle.request_update_count == 4

    panel._unsubscribe_reactive_state()
    signals.scene_generation.emit(3)

    assert panel._handle.request_update_count == 4


def test_histogram_async_result_waits_for_ui_scheduler(histogram_panel_module, monkeypatch):
    module = histogram_panel_module
    panel = module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    panel._histogram_compute_token = 7
    scheduled = []
    panel._ui_scheduler = scheduled.append
    monkeypatch.setattr(module.lf, "get_scene_generation", lambda: 3)
    monkeypatch.setattr(module.RuntimeState, "selection_generation", SimpleNamespace(value=11))
    cache_key = panel._histogram_cache_key(3, ())

    panel._schedule_histogram_result(7, cache_key, {"kind": "empty", "scope_active": False})

    assert len(scheduled) == 1
    assert panel._handle.dirty_all_count == 0
    scheduled.pop()()
    assert panel._empty_title == "No visible values"
    assert panel._handle.dirty_all_count > 0


def test_histogram_cache_key_tracks_scene_attribute_bins_and_node_scope(histogram_panel_module):
    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id = "scale_x"
    panel._histogram_bin_count = 64

    key = panel._histogram_cache_key(19, (23,))

    assert key[:4] == (19, "scale_x", 64, (23,))
    panel._histogram_bin_count = 65
    assert panel._histogram_cache_key(19, (23,)) != key


def test_worker_histogram_matches_numpy_for_random_tensor(histogram_panel_module, lf, numpy):
    rng = numpy.random.default_rng(42)
    values = rng.uniform(-2.0, 3.0, size=257).astype(numpy.float32)
    tensor = lf.Tensor.from_numpy(values)

    class Model:
        num_points = len(values)

        def get_opacity(self):
            return tensor

    result = histogram_panel_module.HistogramPanel._compute_histogram_result(
        SimpleNamespace(get_nodes=lambda: []),
        Model(),
        "opacity",
        32,
        "",
        20,
        20,
        (None, None),
        (None, None),
        set(),
    )

    assert result["kind"] == "ok"
    expected, _ = numpy.histogram(values, bins=numpy.asarray(result["primary"]["edges"]))
    assert result["primary"]["counts"] == expected.tolist()

def test_worker_job_never_touches_the_live_scene(histogram_panel_module, lf, numpy, monkeypatch):
    """The scene and combined-model bindings wrap live C++ objects that the UI and render threads keep
    using, so the worker must not call into them once _refresh has queued the job. Fails if _refresh
    hands the live objects to the worker instead of a snapshot."""
    module = histogram_panel_module
    panel = module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    panel._metric_id = "opacity"
    panel._compare_metric_id = "scale_x"
    values = numpy.array([0.1, 0.4, 0.6, 0.9], dtype=numpy.float32)
    opacity = lf.Tensor.from_numpy(values)
    scaling = lf.Tensor.from_numpy(numpy.repeat(values[:, None], 3, axis=1))
    deleted = lf.Tensor.from_numpy(numpy.array([False, True, False, False]))
    released = {"value": False}

    def live(fn):
        def call(*args, **kwargs):
            assert not released["value"], "worker used the live scene after the job was queued"
            return fn(*args, **kwargs)

        return call

    def crop(mask):
        mask[3] = False

    model = SimpleNamespace(num_points=len(values), get_opacity=live(lambda: opacity),
                            get_scaling=live(lambda: scaling), has_deleted_mask=live(lambda: True),
                            deleted=deleted)
    scene = SimpleNamespace(is_valid=lambda: True, combined_model=lambda: model,
                            get_nodes=live(lambda: []), apply_crop_filter=live(crop))
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene)
    monkeypatch.setattr(panel, "_scene_data_generation_value", lambda: 1)
    monkeypatch.setattr(panel, "_scene_node_selection_signature", lambda: ())
    queued = []
    monkeypatch.setattr(panel, "_queue_histogram_compute", lambda *args, **_kwargs: queued.append(args))

    panel._refresh()
    released["value"] = True
    job_scene, job_model, _cache_key, scope_ids = queued[0]
    result = panel._compute_histogram_result(
        job_scene, job_model, "opacity", 16, "scale_x", 8, 8, (None, None), (None, None), scope_ids,
    )

    assert result["kind"] == "ok"
    numpy.testing.assert_array_equal(result["primary"]["finite_mask"].numpy(), [True, False, True, False])
    assert sum(result["compare"]["counts"]) == 2


@pytest.mark.parametrize("crop_enabled", [False, True])
def test_histogram_applies_render_crop_to_both_axes(histogram_panel_module, lf, numpy, crop_enabled):
    values = lf.Tensor.from_numpy(numpy.array([0.1, 0.3, 0.6, 0.9], dtype=numpy.float32))
    deleted = lf.Tensor.from_numpy(numpy.array([False, True, False, False]))
    model = SimpleNamespace(get_opacity=lambda: values, has_deleted_mask=lambda: True, deleted=deleted)
    calls = []

    def apply_crop_filter(mask):
        calls.append(True)
        if crop_enabled:
            mask[3] = False

    scene = SimpleNamespace(get_nodes=lambda: [], apply_crop_filter=apply_crop_filter)
    result = histogram_panel_module.HistogramPanel._compute_histogram_result(
        scene, model, "opacity", 16, "opacity", 8, 8, (None, None), (None, None), set(),
    )
    expected = [True, False, True, not crop_enabled]
    assert sum(result["primary"]["counts"]) == sum(expected)
    assert sum(result["compare"]["counts"]) == sum(expected)
    numpy.testing.assert_array_equal(result["primary"]["finite_mask"].numpy(), expected)
    numpy.testing.assert_array_equal(result["primary"]["bin_indices"].numpy() >= 0, expected)
    numpy.testing.assert_array_equal(result["compare"]["x_bin_indices"].numpy() >= 0, expected)
    numpy.testing.assert_array_equal(result["compare"]["y_bin_indices"].numpy() >= 0, expected)
    numpy.testing.assert_array_equal(values.numpy(), [numpy.float32(x) for x in (0.1, 0.3, 0.6, 0.9)])
    if crop_enabled:
        assert calls == [True]

    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id = "opacity"
    panel._compare_metric_id = "opacity"
    panel._refresh_compare(scene, model, values, ~deleted)
    assert sum(panel._compare_counts) == sum(expected)
    numpy.testing.assert_array_equal(panel._compare_x_bin_indices.numpy() >= 0, expected)
    numpy.testing.assert_array_equal(panel._compare_y_bin_indices.numpy() >= 0, expected)


def test_histogram_metrics_include_positions_volume_anisotropy_and_erank(histogram_panel_module):
    metric_ids = {metric.id for metric in histogram_panel_module.METRICS}

    assert {"position_x", "position_y", "position_z", "volume", "anisotropy", "erank", "world_distance"} <= metric_ids


@pytest.mark.parametrize("include_nonfinite", [False, True])
def test_erank_auto_range_keeps_rounded_boundary_samples(histogram_panel_module, lf, numpy, include_nonfinite):
    # Float32 entropy can put finite effective ranks just outside the ideal [1, 3] domain.
    values = numpy.array([0.9999997615814209, 1.0, 2.0, 3.0, 3.000000238418579], dtype=numpy.float32)
    if include_nonfinite:
        values = numpy.concatenate((values, numpy.array([numpy.nan, numpy.inf, -numpy.inf], dtype=numpy.float32)))
    finite = numpy.isfinite(values)
    tensor = lf.Tensor.from_numpy(values)
    panel = histogram_panel_module.HistogramPanel()
    result = panel._build_series_result(tensor, tensor.isfinite(), "erank", 56, (None, None), None)

    assert sum(result["counts"]) == int(finite.sum())
    assert result["auto_min"] == float(values[finite].min())
    assert result["auto_max"] == float(values[finite].max())
    numpy.testing.assert_array_equal(result["bin_indices"].numpy() >= 0, finite)
    numpy.testing.assert_array_equal(result["values"].numpy(), values)

    # Both axes and selection indices must use the same automatic bounds as the 1D chart.
    compare = panel._build_compare_result(
        tensor, tensor, tensor.isfinite(), "erank", "erank", 20, 20,
        (None, None), (None, None), None,
    )
    assert numpy.asarray(compare["counts"]).sum() == int(finite.sum())
    numpy.testing.assert_array_equal(compare["x_bin_indices"].numpy() >= 0, finite)
    numpy.testing.assert_array_equal(compare["y_bin_indices"].numpy() >= 0, finite)


@pytest.mark.parametrize("metric,values,ideal", [
    ("opacity", [-1e-7, 0.0, 0.25, 0.75, 1.0, 1.0000001192092896], (0.0, 1.0)),
    ("anisotropy", [0.0028483483, 0.99999994, 1.0, 2.0, 4.0], (1.0, 4.0)),
    ("erank", [0.99999976, 1.0, 2.0, 3.0, 3.00000024], (1.0, 3.0)),
])
@pytest.mark.parametrize("compare", [False, True])
def test_bounded_auto_ranges_keep_all_finite_values(histogram_panel_module, lf, numpy, metric, values, ideal, compare):
    values = numpy.array(values + [numpy.nan, numpy.inf, -numpy.inf], dtype=numpy.float32)
    tensor = lf.Tensor.from_numpy(values)
    finite = numpy.isfinite(values)
    panel = histogram_panel_module.HistogramPanel()
    if compare:
        result = panel._build_compare_result(
            tensor, tensor, tensor.isfinite(), metric, metric, 8, 8, (None, None), (None, None))
        for axis in ("x", "y"):
            numpy.testing.assert_array_equal(result[f"{axis}_bin_indices"].numpy() >= 0, finite)
    else:
        result = panel._build_series_result(tensor, tensor.isfinite(), metric, 8, (None, None))
        numpy.testing.assert_array_equal(result["bin_indices"].numpy() >= 0, finite)
        numpy.testing.assert_array_equal(result["values"].numpy(), values)
    assert numpy.asarray(result["counts"]).sum() == finite.sum()

    # An explicit ideal-domain range must continue to exclude the same samples.
    expected = finite & (values >= ideal[0]) & (values <= ideal[1])
    if compare:
        custom = panel._build_compare_result(tensor, tensor, tensor.isfinite(), metric, metric, 8, 8, ideal, ideal)
        for axis in ("x", "y"):
            numpy.testing.assert_array_equal(custom[f"{axis}_bin_indices"].numpy() >= 0, expected)
    else:
        custom = panel._build_series_result(tensor, tensor.isfinite(), metric, 8, ideal)
        numpy.testing.assert_array_equal(custom["bin_indices"].numpy() >= 0, expected)
    assert numpy.asarray(custom["counts"]).sum() == expected.sum()


def test_anisotropy_small_scales_keep_metric_values(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    scaling = numpy.array([[1e-15, 2e-15, 3e-15], [1e-12, 1e-12, 1e-12], [1.0, 2.0, 4.0]], dtype=numpy.float32)
    model = _ModelStub(lf, numpy.zeros((3, 3), dtype=numpy.float32), scaling)
    panel._metric_id = "anisotropy"
    values = panel._extract_metric_values(_SceneSelectionStub(None), model)
    expected = scaling.max(axis=1) / (scaling.min(axis=1) + 1e-12)
    numpy.testing.assert_array_equal(values.numpy(), expected)
    result = panel._build_series_result(values, values.isfinite(), "anisotropy", 8, (None, None))
    assert sum(result["counts"]) == 3


@pytest.mark.parametrize("reverse", [False, True])
def test_bounded_auto_ranges_survive_compare_cache_rebind(histogram_panel_module, lf, numpy, reverse):
    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id, panel._compare_metric_id = ("anisotropy", "opacity") if reverse else ("opacity", "anisotropy")
    model = _ModelStub(
        lf, numpy.zeros((3, 3), dtype=numpy.float32),
        numpy.array([[1e-15, 2e-15, 3e-15], [1e-12, 1e-12, 1e-12], [1.0, 2.0, 4.0]], dtype=numpy.float32),
        opacity=numpy.array([-1e-7, 0.5, 1.00000012], dtype=numpy.float32),
    )
    scene = _SceneSelectionStub()
    primary = panel._extract_metric_values(scene, model, panel._metric_id)
    panel._refresh_compare(scene, model, primary, None)
    assert sum(panel._compare_counts) == 3
    expected = list(panel._compare_counts)
    panel._rebind_compare_from_cache()
    assert panel._compare_counts == expected


@pytest.mark.parametrize("metric,values", [
    ("anisotropy", [0.25]), ("anisotropy", [0.25, 0.5]), ("anisotropy", [1.0, 1.0]),
    ("opacity", [-1e-7]), ("opacity", [1.00000012]), ("erank", [0.99999976]),
])
def test_bounded_auto_ranges_keep_constant_and_outside_only_values(histogram_panel_module, lf, numpy, metric, values):
    panel = histogram_panel_module.HistogramPanel()
    tensor = lf.Tensor.from_numpy(numpy.array(values, dtype=numpy.float32))
    result = panel._build_series_result(tensor, tensor.isfinite(), metric, 8, (None, None))
    assert sum(result["counts"]) == len(values)
    compare = panel._build_compare_result(tensor, tensor, tensor.isfinite(), metric, metric, 8, 8, (None, None), (None, None))
    assert sum(compare["counts"]) == len(values)


@pytest.mark.parametrize("metric", [
    "opacity", "position_x", "position_y", "position_z", "scale_x", "scale_y", "scale_z",
    "scale_max", "volume", "anisotropy", "erank", "distance", "world_distance",
])
def test_all_metric_auto_ranges_preserve_in_domain_bins(histogram_panel_module, lf, numpy, metric):
    panel = histogram_panel_module.HistogramPanel()
    values = numpy.array([0.0, 0.13, 0.39, 0.61, 0.87, 1.0], dtype=numpy.float32)
    if metric in ("anisotropy", "erank"):
        values = 1.0 + 2.0 * values
    tensor = lf.Tensor.from_numpy(values)
    result = panel._build_series_result(tensor, tensor.isfinite(), metric, 8, (None, None))
    expected, _ = numpy.histogram(values, bins=8, range=(float(values.min()), float(values.max())))
    assert result["counts"] == expected.tolist()
    numpy.testing.assert_array_equal(result["values"].numpy(), values)


def test_erank_in_domain_and_explicit_ranges_keep_existing_bins(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    values = numpy.array([1.0, 1.125, 1.5, 2.0, 2.5, 2.875, 3.0], dtype=numpy.float32)
    tensor = lf.Tensor.from_numpy(values)
    assert panel._histogram_bounds(tensor, "erank") == (1.0, 3.0)
    result = panel._build_series_result(tensor, tensor.isfinite(), "erank", 8, (None, None), None)
    expected, _ = numpy.histogram(values, bins=8, range=(1.0, 3.0))
    assert result["counts"] == expected.tolist()

    # Explicit user limits continue to exclude values outside the requested range.
    boundary_values = numpy.array([0.9999997615814209, 1.0, 2.0, 3.0, 3.000000238418579], dtype=numpy.float32)
    boundary = lf.Tensor.from_numpy(boundary_values)
    explicit = panel._build_series_result(boundary, boundary.isfinite(), "erank", 8, (1.0, 3.0), None)
    assert sum(explicit["counts"]) == 3
    assert explicit["histogram_min"] == 1.0
    assert explicit["histogram_max"] == 3.0
    compare = panel._build_compare_result(
        boundary, boundary, boundary.isfinite(), "erank", "erank", 8, 8,
        (1.0, 3.0), (1.0, 3.0), None,
    )
    assert numpy.asarray(compare["counts"]).sum() == 3


@pytest.mark.parametrize("values", [[1.0], [2.0, 2.0], [3.0], [1.125, 1.5, 2.875]])
@pytest.mark.parametrize("bounds", [(1.0, 3.0), (1.5, 2.0), (2.9, 3.0), (2.0, 1.0)])
def test_cached_erank_extent_preserves_range_snapping(histogram_panel_module, lf, numpy, values, bounds):
    tensor = lf.Tensor.from_numpy(numpy.array(values, dtype=numpy.float32))
    snap = histogram_panel_module.HistogramPanel._snap_bounds_to_data
    # The uncached path retains the previous mask/gather/reduce implementation.
    assert snap(tensor, *bounds, (min(values), max(values))) == snap(tensor, *bounds)


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
    assert panel._marked_range_text == "6.25e-2 to 1.25e-1"

    panel._set_histogram_bin_count(17)

    assert panel._marked_count == 1
    assert panel._marked_range_text == "6.25e-2 to 1.25e-1"


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


def test_histogram_drag_live_updates_scene_selection_before_mouseup(histogram_panel_module, lf, numpy, monkeypatch):
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

    panel._on_chart_mousedown(_MouseEventStub(mouse_x=1.0))

    assert scene.preview_mask_calls == 1
    assert scene.commit_preview_calls == 0
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, False]))

    panel._on_document_mousemove(_MouseEventStub(mouse_x=51.0))

    assert scene.preview_mask_calls == 2
    assert scene.commit_preview_calls == 0
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, True, True]))

    panel._on_document_mouseup(_MouseEventStub(mouse_x=51.0))

    assert scene.commit_preview_calls == 1
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, True, True]))


def test_histogram_range_uses_scene_selection_and_standard_edit_actions(histogram_panel_module, lf, numpy, monkeypatch):
    from lfs_plugins.selection_controls import SelectionControlsController

    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)
    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35], dtype=numpy.float32))
    panel._primary_values = values
    panel._primary_finite_mask = values.isfinite()
    panel._primary_valid_values = values[panel._primary_finite_mask]
    panel._primary_histogram_min, panel._primary_histogram_max = 0.0, 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()
    scene = _SceneSelectionStub(lf.Tensor.from_numpy(numpy.array([False, False, True])))
    monkeypatch.setattr(lf, "get_scene", lambda: scene)
    monkeypatch.setattr(panel, "_any_true", lambda *_args: pytest.fail("Drag must use cached bin counts"))

    panel._on_chart_mousedown(_MouseEventStub(mouse_x=1.0))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=21.0))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=21.0))
    expected = numpy.array([True, True, False])
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), expected)
    assert scene.commit_preview_calls == 1

    deleted_masks = []

    def delete_selected():
        deleted_masks.append(scene.selection_mask.cpu().numpy().copy())
        scene.selection_history.append(scene.selection_mask)
        return {"ok": True}

    monkeypatch.setattr(
        lf,
        "pipeline",
        SimpleNamespace(edit=SimpleNamespace(delete_=lambda: SimpleNamespace(
            execute=delete_selected
        ))),
        raising=False,
    )

    def undo_selection():
        scene.selection_mask = scene.selection_history.pop()

    monkeypatch.setattr(
        lf,
        "undo",
        SimpleNamespace(can_undo=lambda: bool(scene.selection_history), undo=undo_selection),
        raising=False,
    )
    controls = SelectionControlsController()
    controls._refresh_state = lambda: None
    controls._dirty_all = lambda: None
    controls._on_action(None, None, ["delete"])
    controls._on_action(None, None, ["undo"])
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), expected)
    controls._on_action(None, None, ["undo"])

    numpy.testing.assert_array_equal(deleted_masks[0], expected)
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([False, False, True]))


def test_histogram_panel_has_no_action_footer_and_keeps_mark_stats_as_cards():
    resources = Path(__file__).resolve().parents[2] / "src/visualizer/gui/rmlui/resources"
    rml = (resources / "histogram_panel.rml").read_text(encoding="utf-8")
    rcss = (resources / "histogram_panel.rcss").read_text(encoding="utf-8")

    assert "histogram-footer" not in rml + rcss
    assert "delete_marked()" not in rml
    assert "undo_history()" not in rml
    assert rml.count("{{ marked_label }}") == 1
    assert rml.count("{{ count_label }}") == 1


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










def test_histogram_ctrl_scroll_zooms_at_cursor_through_keymap(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._handle = _UpdateHandleStub()
    panel._primary_valid_values = lf.Tensor.from_numpy(
        numpy.array([0.05, 0.15, 0.25, 0.35, 0.45], dtype=numpy.float32)
    )
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._auto_histogram_min = 0.0
    panel._auto_histogram_max = 1.0
    panel._histogram_bin_count = 10
    # Chart geometry so a mouse-x maps to a value; cursor sits at the chart centre (value 0.5).
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=100.0)
    # Isolate the cursor-zoom math from data snapping (snapping is exercised separately).
    monkeypatch.setattr(panel, "_snap_histogram_zoom_bounds_to_data", lambda lo, hi: (lo, hi))

    zoom_action = object()
    seen_scrolls = []

    def get_action_for_scroll(mode, modifiers):
        seen_scrolls.append((mode, modifiers))
        return zoom_action if modifiers == 2 else object()

    fake_keymap = SimpleNamespace(
        Modifier=SimpleNamespace(
            SHIFT=SimpleNamespace(value=1),
            CTRL=SimpleNamespace(value=2),
            ALT=SimpleNamespace(value=4),
            SUPER=SimpleNamespace(value=8),
        ),
        ToolMode=SimpleNamespace(GLOBAL="global"),
        Action=SimpleNamespace(HISTOGRAM_ZOOM_MARKED=zoom_action),
        get_action_for_scroll=get_action_for_scroll,
    )
    monkeypatch.setattr(histogram_panel_module.lf, "keymap", fake_keymap, raising=False)

    # A refresh would re-bin against the new custom range; mirror that so the view the
    # next scroll reads from tracks the committed range.
    def sync_view_from_custom_range(view_only=False):
        panel._primary_histogram_min = (
            panel._auto_histogram_min
            if panel._custom_range_min_value is None
            else panel._custom_range_min_value
        )
        panel._primary_histogram_max = (
            panel._auto_histogram_max
            if panel._custom_range_max_value is None
            else panel._custom_range_max_value
        )

    monkeypatch.setattr(panel, "_refresh", sync_view_from_custom_range)

    # Zoom in: the value under the cursor (0.5) stays pinned while the span shrinks 20%.
    zoom_event = _MouseEventStub(mouse_x=50.0, modifiers=RML_KM_CTRL, wheel_delta=-1.0)
    panel._on_chart_mousescroll(zoom_event)
    assert zoom_event.stopped is True
    assert seen_scrolls == [("global", 2)]
    assert panel._custom_range_min_value == pytest.approx(0.1)
    assert panel._custom_range_max_value == pytest.approx(0.9)

    # Zoom in again around the same cursor value.
    second_zoom_event = _MouseEventStub(mouse_x=50.0, modifiers=RML_KM_CTRL, wheel_delta=-1.0)
    panel._on_chart_mousescroll(second_zoom_event)
    assert second_zoom_event.stopped is True
    assert panel._custom_range_min_value == pytest.approx(0.18)
    assert panel._custom_range_max_value == pytest.approx(0.82)

    # Zoom back out widens the window around the cursor.
    zoom_out_event = _MouseEventStub(mouse_x=50.0, modifiers=RML_KM_CTRL, wheel_delta=1.0)
    panel._on_chart_mousescroll(zoom_out_event)
    assert zoom_out_event.stopped is True
    assert panel._custom_range_min_value == pytest.approx(0.1)
    assert panel._custom_range_max_value == pytest.approx(0.9)

    # Zooming all the way out drops the custom range entirely (back to the full extent).
    for _ in range(10):
        panel._on_chart_mousescroll(_MouseEventStub(mouse_x=50.0, modifiers=RML_KM_CTRL, wheel_delta=1.0))
    assert panel._custom_range_min_value is None
    assert panel._custom_range_max_value is None




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

    scene.selection_mask = lf.Tensor.from_numpy(numpy.ones(3, dtype=bool))

    x = 2 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, ctrl=True))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x, ctrl=True))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, ctrl=True))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._marked_count == 2
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (2, 2)


def test_histogram_alt_drag_intersects_existing_selection(histogram_panel_module, lf, numpy, monkeypatch):
    panel, scene = _three_bin_panel(histogram_panel_module, lf, numpy, monkeypatch)
    scene.selection_mask = lf.Tensor.from_numpy(numpy.array([True, False, True], dtype=bool))

    x = 5 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, alt=True))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x, alt=True))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, alt=True))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([False, False, True]))


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

    scene.selection_mask = lf.Tensor.from_numpy(numpy.ones(3, dtype=bool))

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

    scene.selection_mask = lf.Tensor.from_numpy(numpy.ones(3, dtype=bool))

    x = 2 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, modifiers=RML_KM_CTRL))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x, modifiers=RML_KM_CTRL))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, modifiers=RML_KM_CTRL))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (2, 2)


def test_histogram_modifier_pressed_after_mousedown_does_not_change_mode(histogram_panel_module, lf, numpy, monkeypatch):
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

    # Like the viewport selection tools, the modifiers at press time decide the mode.
    x = 5 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x))
    panel._on_document_mousemove(_MouseEventStub(mouse_x=x + 1.0, shift=True))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x, shift=True))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([False, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {5}
    assert panel._histogram_overlay_bounds == (5, 5)


def test_histogram_shift_click_adds_a_bin_like_selection_tools(histogram_panel_module, lf, numpy, monkeypatch):
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
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (5, 5)


def _three_bin_panel(histogram_panel_module, lf, numpy, monkeypatch):
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
    return panel, scene


def _click_bin(panel, bin_index, **modifiers):
    x = bin_index * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, **modifiers))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x))


def test_histogram_consecutive_shift_clicks_keep_every_added_bin(histogram_panel_module, lf, numpy, monkeypatch):
    panel, scene = _three_bin_panel(histogram_panel_module, lf, numpy, monkeypatch)

    _click_bin(panel, 0)
    _click_bin(panel, 2, shift=True)
    _click_bin(panel, 5, shift=True)

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, True, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 2, 5}


def test_histogram_consecutive_ctrl_clicks_keep_every_removed_bin(histogram_panel_module, lf, numpy, monkeypatch):
    panel, scene = _three_bin_panel(histogram_panel_module, lf, numpy, monkeypatch)

    panel._marked_bin_start = 0
    panel._marked_bin_end = 5
    panel._sync_marked_range(apply_scene=True)
    _click_bin(panel, 0, ctrl=True)
    _click_bin(panel, 2, ctrl=True)

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([False, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {5}


def _settle_on_update(panel, module, lf, monkeypatch, selection_generation):
    panel._scene_generation = 0
    panel._scene_data_generation = 0
    panel._history_generation = 0
    panel._selection_generation = selection_generation - 1
    panel._last_lang = "en"
    panel._trainer_state = module.RuntimeState.trainer_state.value
    panel._selected_nodes_signature = ()
    monkeypatch.setattr(lf, "get_scene_generation", lambda: 0)
    monkeypatch.setattr(panel, "_scene_data_generation_value", lambda: 0)
    monkeypatch.setattr(panel, "_history_generation_value", lambda: 0)
    monkeypatch.setattr(panel, "_selection_generation_value", lambda: selection_generation)
    monkeypatch.setattr(panel, "_scene_node_selection_signature", lambda: ())
    monkeypatch.setattr(panel, "_sync_panel_space_state", lambda: False)
    monkeypatch.setattr(lf.ui, "get_current_language", lambda: "en")
    panel.on_update(None)


def test_histogram_ctrl_click_unhighlights_the_removed_bin_at_once(histogram_panel_module, lf, numpy, monkeypatch):
    panel, scene = _three_bin_panel(histogram_panel_module, lf, numpy, monkeypatch)
    panel._handle = _UpdateHandleStub()

    _click_bin(panel, 0)
    _click_bin(panel, 5, shift=True)
    _click_bin(panel, 0, ctrl=True)

    # The bars the panel last pushed, not a fresh rebuild.
    highlighted = {index for index, record in enumerate(panel._handle.records["bins"]) if record["selected"]}
    assert panel._selected_histogram_bins == {5}
    assert highlighted == {5}


def test_histogram_resets_when_the_selection_changes_outside_the_panel(histogram_panel_module, lf, numpy, monkeypatch):
    panel, scene = _three_bin_panel(histogram_panel_module, lf, numpy, monkeypatch)
    _click_bin(panel, 0)
    assert panel._selected_histogram_bins == {0}

    # A viewport tool selects the second Gaussian instead: the panel's mark resets
    # and the tool's selection stays in the scene.
    scene.selection_mask = lf.Tensor.from_numpy(numpy.array([False, True, False], dtype=bool))
    _settle_on_update(panel, histogram_panel_module, lf, monkeypatch, selection_generation=5)
    assert panel._selected_histogram_bins == set()
    assert panel._panel_selection_mask is None
    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([False, True, False], dtype=bool))

    # The panel's own next selection is kept across updates.
    _click_bin(panel, 0)
    _settle_on_update(panel, histogram_panel_module, lf, monkeypatch, selection_generation=6)
    assert panel._selected_histogram_bins == {0}


def test_histogram_shift_click_adds_to_a_selection_made_outside_the_panel(histogram_panel_module, lf, numpy, monkeypatch):
    panel, scene = _three_bin_panel(histogram_panel_module, lf, numpy, monkeypatch)
    _click_bin(panel, 0)

    scene.selection_mask = lf.Tensor.from_numpy(numpy.array([False, True, False], dtype=bool))
    _settle_on_update(panel, histogram_panel_module, lf, monkeypatch, selection_generation=5)
    _click_bin(panel, 5, shift=True)

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([False, True, True], dtype=bool))
    assert panel._selected_histogram_bins == {2, 5}


def test_histogram_ctrl_click_removes_a_bin_like_selection_tools(histogram_panel_module, lf, numpy, monkeypatch):
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

    scene.selection_mask = lf.Tensor.from_numpy(numpy.ones(3, dtype=bool))

    x = 2 * 10.0 + 1.0
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x, ctrl=True))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([True, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {0, 5}
    assert panel._marked_range_text == "Multiple ranges"
    assert panel._histogram_overlay_bounds == (2, 2)


def test_histogram_plain_click_on_selected_bar_replaces_the_selection(histogram_panel_module, lf, numpy, monkeypatch):
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

    assert panel._selected_histogram_bins == {0, 5}

    # A plain click never toggles: it replaces the selection with the clicked bin.
    panel._on_chart_mousedown(_MouseEventStub(mouse_x=x))
    panel._on_document_mouseup(_MouseEventStub(mouse_x=x))

    numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), numpy.array([False, False, True], dtype=bool))
    assert panel._selected_histogram_bins == {5}
    assert panel._histogram_overlay_bounds == (5, 5)


def test_histogram_owned_modifier_selection_survives_two_follow_up_updates(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"
    panel._chart_el = SimpleNamespace(absolute_left=0.0, absolute_width=160.0)
    panel._scene_generation = 0
    panel._scene_data_generation = 0
    monkeypatch.setattr(panel, "_scene_data_generation_value", lambda: 0)
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

    scene = _SceneSelectionStub(lf.Tensor.from_numpy(numpy.array([True, False, True], dtype=bool)))
    monkeypatch.setattr(lf, "get_scene", lambda: scene)

    panel._commit_histogram_mask_selection(
        lf.Tensor.from_numpy(numpy.array([True, False, True], dtype=bool)),
        apply_scene=False,
        overlay_bounds=(5, 5),
    )
    panel._selection_owned = True

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


@pytest.mark.parametrize("saved_space", ["FLOATING", "BOTTOM_DOCK"])
def test_histogram_panel_preserves_and_toggles_saved_space(histogram_panel_module, lf, monkeypatch, saved_space):
    panel = histogram_panel_module.HistogramPanel()
    state = {"space": getattr(lf.ui.PanelSpace, saved_space)}
    moves = []
    monkeypatch.setattr(lf.ui, "get_panel", lambda panel_id: SimpleNamespace(space=state["space"]))

    def set_space(panel_id, space):
        assert panel_id == panel.id
        moves.append(space)
        state["space"] = space
        return True

    monkeypatch.setattr(lf.ui, "set_panel_space", set_space)
    assert panel._sync_panel_space_state() is (saved_space == "FLOATING")
    assert state["space"] == getattr(lf.ui.PanelSpace, saved_space)
    assert panel._sync_panel_space_state() is False
    assert moves == []
    assert panel._dock_toggle_label() == ("Dock" if saved_space == "FLOATING" else "Undock")

    for _ in range(2):
        previous = state["space"]
        panel._on_toggle_dock_mode(None, None, None)
        assert state["space"] != previous
        assert panel._is_floating == (state["space"] == lf.ui.PanelSpace.FLOATING)
        assert panel._sync_panel_space_state() is False
    assert state["space"] == getattr(lf.ui.PanelSpace, saved_space)
    assert len(moves) == 2


def test_histogram_failed_dock_toggle_preserves_state(histogram_panel_module, lf, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    monkeypatch.setattr(lf.ui, "get_panel", lambda _: SimpleNamespace(space=lf.ui.PanelSpace.BOTTOM_DOCK))
    monkeypatch.setattr(lf.ui, "set_panel_space", lambda *_: False)
    panel._on_toggle_dock_mode(None, None, None)
    assert panel._dock_toggle_label() == "Undock"
    assert not panel._is_floating



@pytest.mark.parametrize("prop,minimum,maximum", [
    ("histogram_bin_count", 16, 128),
    ("compare_x_bin_count", 8, 48),
    ("compare_y_bin_count", 8, 48),
])
def test_bin_scrub_drag_cancel_and_typed_value(histogram_panel_module, prop, minimum, maximum):
    from unittest.mock import MagicMock
    from lfs_plugins.scrub_fields import _ScrubFieldState
    from lfs_plugins.rml_keys import KI_ESCAPE

    panel = histogram_panel_module.HistogramPanel()
    controller = panel._scrub_fields
    attrs = {}
    input_el = MagicMock()
    input_el.get_attribute.side_effect = lambda name, default="": attrs.get(name, default)
    input_el.set_attribute.side_effect = attrs.__setitem__
    state = _ScrubFieldState(prop, controller._specs[prop], MagicMock(absolute_width=100.0),
                             MagicMock(), MagicMock(), input_el)
    controller._fields[prop] = state
    initial = panel._get_bin_scrub_value(prop)

    def event(**params):
        return SimpleNamespace(get_parameter=lambda name, default: params.get(name, default),
                               get_bool_parameter=lambda name, default: params.get(name, default),
                               stop_propagation=lambda: None)

    controller._on_field_mousedown(prop, event(mouse_x=0))
    controller._on_body_mousemove(event(mouse_x=10))
    assert panel._get_bin_scrub_value(prop) > initial
    controller._on_keydown(event(key_identifier=KI_ESCAPE))
    assert panel._get_bin_scrub_value(prop) == initial
    controller._enter_edit(state)
    attrs["value"] = "32"
    controller._on_input_change(prop, event(linebreak=True))
    controller._on_input_blur(prop, event())
    assert panel._get_bin_scrub_value(prop) == 32
    for value, expected in [(999, maximum), (-5, minimum)]:
        controller._apply_value(state, value)
        assert panel._get_bin_scrub_value(prop) == expected


@pytest.mark.parametrize("value,expected", [
    (0.0, "0"), (1e-4, "1e-4"), (-0.0001234, "-1.234e-4"),
    (0.003531, "3.531e-3"), (0.9821, "9.821e-1"), (1234000.0, "1.234e6"),
])
def test_range_scientific_text_is_compact_and_parseable(histogram_panel_module, value, expected):
    panel = histogram_panel_module.HistogramPanel()
    text = panel._format_range_input(value)
    assert text == expected
    assert panel._parse_range_input(text) == pytest.approx(value, rel=5e-4)
    assert len(text) <= 10


# --- Snappiness / elite-UX regressions -------------------------------------------------

def test_wheel_zoom_magnitude_scales_with_delta(histogram_panel_module):
    f = histogram_panel_module.HistogramPanel._wheel_zoom_magnitude
    assert f(1.0) == 1.0
    assert f(-1.0) == 1.0
    assert f(120.0) == 1.0           # one HID notch
    assert f(5.0) == 1.0             # small per-notch systems stay single-step
    assert f(600.0) == pytest.approx(5.0)   # 5-notch flick zooms 5x further
    assert f(-100000.0) == 8.0       # clamped


def test_approx_equal_uses_relative_tolerance(histogram_panel_module):
    f = histogram_panel_module.HistogramPanel._approx_equal
    assert f(1.0, 1.0)
    assert f(1_000_000.0, 1_000_000.5)        # within 1e-6 relative
    assert not f(1_000_000.0, 1_000_100.0)    # outside relative tolerance
    assert f(0.0, 0.0)
    assert not f(0.0, 1e-3)


def test_cursor_zoom_magnitude_zooms_further(histogram_panel_module):
    f = histogram_panel_module.HistogramPanel._cursor_zoom_bounds
    one = f(0.0, 1.0, 0.5, 0.0, 1.0, zoom_in=True, magnitude=1.0)
    five = f(0.0, 1.0, 0.5, 0.0, 1.0, zoom_in=True, magnitude=5.0)
    assert one is not None and five is not None
    assert (one[1] - one[0]) == pytest.approx(0.8)
    assert (five[1] - five[0]) == pytest.approx(0.8 ** 5)
    assert (five[1] - five[0]) < (one[1] - one[0])


def test_refresh_view_only_reuses_cache_without_extracting(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    data = numpy.array([0.1, 0.5, 0.9], dtype=numpy.float32)
    panel._primary_valid_values = lf.Tensor.from_numpy(data)
    panel._primary_finite_values_cpu = lf.Tensor.from_numpy(data)

    calls = {"extract": 0, "rebind": 0}
    monkeypatch.setattr(panel, "_extract_metric_values",
                        lambda *a, **k: calls.__setitem__("extract", calls["extract"] + 1))
    monkeypatch.setattr(panel, "_rebind_view_from_cache",
                        lambda: calls.__setitem__("rebind", calls["rebind"] + 1))

    panel._refresh(view_only=True)

    assert calls["rebind"] == 1           # re-binned from cache
    assert calls["extract"] == 0          # never re-extracted the scene
    assert panel._handle.dirty_all_count == 1   # exactly one repaint


def test_refresh_view_only_falls_back_when_cache_empty(histogram_panel_module, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    panel._primary_finite_values_cpu = None
    rebind = []
    monkeypatch.setattr(panel, "_rebind_view_from_cache", lambda: rebind.append(1))
    monkeypatch.setattr(histogram_panel_module.lf, "get_scene", lambda: None, raising=False)

    panel._refresh(view_only=True)

    assert rebind == []   # no cache -> falls through to the full (here: no-scene) path


@pytest.mark.parametrize("compare", [False, True])
@pytest.mark.parametrize("mode", ["replace", "add", "remove", "intersect"])
def test_chart_mousedown_allows_native_doubleclick(histogram_panel_module, monkeypatch, compare, mode):
    panel = histogram_panel_module.HistogramPanel()
    focused, previews, refreshed = [], [], []
    chart = SimpleNamespace(focus=lambda: focused.append(True))
    panel._chart_el = panel._compare_chart_el = chart
    panel._show_chart = panel._show_compare_chart = True
    panel._hist_edges = [0.0, 1.0]
    panel._custom_range_min_value, panel._custom_range_max_value = 0.2, 0.8
    panel._compare_y_custom_range_min_value, panel._compare_y_custom_range_max_value = 0.3, 0.7
    mask, bins = object(), {2}
    monkeypatch.setattr(panel, "_current_selection_mask_for_source", lambda source: mask)
    monkeypatch.setattr(panel, "_selected_histogram_bins_from_mask", lambda value: bins)
    monkeypatch.setattr(panel, "_selected_compare_cells_from_mask", lambda value: bins)
    monkeypatch.setattr(panel, "_bin_index_for_mouse_x", lambda x: 2)
    monkeypatch.setattr(panel, "_compare_bin_indices_for_mouse", lambda x, y: (2, 3))
    monkeypatch.setattr(panel, "_clear_compare_mark", lambda **kwargs: None)
    monkeypatch.setattr(panel, "_clear_histogram_mark", lambda **kwargs: None)
    monkeypatch.setattr(panel, "_sync_marked_range", lambda **kwargs: previews.append(kwargs))
    monkeypatch.setattr(panel, "_sync_compare_mark", lambda **kwargs: previews.append(kwargs))
    monkeypatch.setattr(panel, "_refresh_range_preserving_mark", lambda: refreshed.append(True))
    down = panel._on_compare_chart_mousedown if compare else panel._on_chart_mousedown
    doubleclick = panel._on_compare_chart_dblclick if compare else panel._on_chart_dblclick

    event = _MouseEventStub(
        mouse_x=20.0,
        mouse_y=30.0,
        shift=mode == "add",
        ctrl=mode == "remove",
        alt=mode == "intersect",
    )
    down(event)

    assert focused == [True]
    assert previews == [{"apply_scene": False, "preview_scene": True}]
    if compare:
        assert panel._dragging_compare_mark
        assert panel._compare_mark_start == panel._compare_mark_end == (2, 3)
        assert panel._drag_compare_selection_mode == mode
        assert panel._drag_compare_selection_base_mask is (None if mode == "replace" else mask)
    else:
        assert panel._dragging_mark
        assert panel._marked_bin_start == panel._marked_bin_end == 2
        assert panel._drag_selection_mode == mode
        assert panel._drag_selection_base_mask is (None if mode == "replace" else mask)

    # RmlUi Context::ProcessMouseButtonDown only detects dblclick when the
    # mousedown dispatch propagates. Stopping it prevents the callback entirely.
    assert not event.stopped
    doubleclick(event)
    assert panel._custom_range_min_value is None
    assert panel._custom_range_max_value is None
    if compare:
        assert panel._compare_y_custom_range_min_value is None
        assert panel._compare_y_custom_range_max_value is None
    else:
        assert panel._compare_y_custom_range_min_value == 0.3
        assert panel._compare_y_custom_range_max_value == 0.7
    assert refreshed == [True]
    assert event.stopped


def test_chart_dblclick_fits_and_clears_custom_range(histogram_panel_module, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    panel._show_chart = True
    panel._custom_range_min_value = 0.2
    panel._custom_range_max_value = 0.8
    refreshed = []
    monkeypatch.setattr(panel, "_refresh_range_preserving_mark", lambda: refreshed.append(1))

    event = _MouseEventStub(mouse_x=0.0)
    panel._on_chart_dblclick(event)

    assert panel._custom_range_min_value is None
    assert panel._custom_range_max_value is None
    assert refreshed == [1]
    assert event.stopped is True


def test_mouseup_aborts_drag_when_scene_invalid(histogram_panel_module, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    panel._dragging_mark = True
    monkeypatch.setattr(histogram_panel_module.lf, "get_scene",
                        lambda: SimpleNamespace(is_valid=lambda: False), raising=False)
    reset = []
    monkeypatch.setattr(panel, "_reset_mark_fields", lambda clear_scene=False: reset.append(clear_scene))

    event = _MouseEventStub(mouse_x=0.0)
    panel._on_document_mouseup(event)

    assert panel._dragging_mark is False
    assert reset == [False]
    assert event.stopped is True


def test_histogram_worker_runs_without_numpy(lf):
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        sys.path[:] = PATHS
        import importlib.abc
        class NoNumpy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'numpy' or fullname.startswith('numpy.'):
                    raise ModuleNotFoundError("No module named 'numpy'")
        sys.meta_path.insert(0, NoNumpy())
        import lichtfeld as lf
        from lfs_plugins.histogram_panel import HistogramPanel
        panel = HistogramPanel()
        values = lf.Tensor.linspace(0.0, 1.0, 5, device='cpu')
        mask = lf.Tensor.ones([5], dtype='bool', device='cpu')
        result = panel._build_series_result(values, mask, 'opacity', 4, (None, None))
        assert result['counts'] == [1, 1, 1, 2], result['counts']
        assert result['mean_value'] == 0.5
        assert result['median_value'] == 0.5
        assert abs(result['p95_value'] - 0.95) < 1e-6
        compare = panel._build_compare_result(values, values, mask, 'opacity', 'opacity',
                                             4, 4, (None, None), (None, None))
        assert compare['counts'] == [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,2], compare['counts']
    """).replace('PATHS', repr(sys.path))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_selection_change_updates_highlights_without_recomputing(histogram_panel_module, monkeypatch):
    module = histogram_panel_module
    panel = module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    panel._scene_generation = 3
    panel._scene_data_generation = 8
    panel._history_generation = 4
    panel._selection_generation = 2
    panel._last_lang = 'en'
    panel._trainer_state = module.RuntimeState.trainer_state.value
    panel._selected_nodes_signature = ()
    monkeypatch.setattr(module.lf, 'get_scene_generation', lambda: 5)
    monkeypatch.setattr(module.lf.ui, 'get_current_language', lambda: 'en')
    monkeypatch.setattr(panel, '_scene_data_generation_value', lambda: 8, raising=False)
    monkeypatch.setattr(panel, '_history_generation_value', lambda: 5)
    monkeypatch.setattr(panel, '_selection_generation_value', lambda: 3)
    monkeypatch.setattr(panel, '_scene_node_selection_signature', lambda: ())
    monkeypatch.setattr(panel, '_sync_panel_space_state', lambda: False)
    calls = []
    monkeypatch.setattr(panel, '_refresh', lambda: calls.append('recompute'))
    monkeypatch.setattr(panel, '_reset_marks_after_external_selection', lambda: calls.append('highlight'))
    panel.on_scene_changed(None)
    panel.on_update(None)
    assert calls == ['highlight']


@pytest.mark.parametrize('setter', ['_set_histogram_bin_count', '_set_compare_x_bin_count', '_set_compare_y_bin_count'])
def test_last_bin_slider_value_is_computed_while_worker_is_pending(histogram_panel_module, setter):
    panel = histogram_panel_module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    panel._show_chart = False
    panel._show_compare_card = False
    panel._computing = True
    requested = []
    panel._refresh = lambda: requested.append((panel._histogram_bin_count, panel._compare_x_bin_count, panel._compare_y_bin_count))
    getattr(panel, setter)(32)
    getattr(panel, setter)(48)
    assert len(requested) == 2
    assert 48 in requested[-1]






@pytest.fixture
def pending_range_panel(histogram_panel_module, lf, numpy, monkeypatch):
    """Use real snapshot computation, but deliver it after input events deterministically."""
    module = histogram_panel_module
    panel = module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    panel._metric_id = "opacity"
    panel._compare_metric_id = "scale_x"
    values = numpy.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=numpy.float32)
    opacity = lf.Tensor.from_numpy(values)
    scaling = lf.Tensor.from_numpy(numpy.repeat(values[:, None], 3, axis=1))
    means = lf.Tensor.from_numpy(numpy.array([
        [-40, -10, 0], [-2, -5, 0], [0, 0, 0], [2, 5, 0], [3, 10, 0], [58, 20, 0],
    ], dtype=numpy.float32))
    model = SimpleNamespace(num_points=len(values), get_opacity=lambda: opacity,
                            get_scaling=lambda: scaling, get_means=lambda: means)
    scene = SimpleNamespace(is_valid=lambda: True, combined_model=lambda: model, get_nodes=lambda: [])
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene)
    monkeypatch.setattr(panel, "_scene_data_generation_value", lambda: 1)
    monkeypatch.setattr(panel, "_scene_node_selection_signature", lambda: ())
    monkeypatch.setattr(panel, "_reset_marks_after_external_selection", lambda: None)
    queued = []
    scheduled = []
    panel._ui_scheduler = scheduled.append

    def queue(scene, model, cache_key, scope_ids, **_kwargs):
        panel._cancel_histogram_compute()
        result = panel._compute_histogram_result(
            scene, model, panel._metric_id, panel._histogram_bin_count,
            panel._compare_metric_id, panel._compare_x_bin_count, panel._compare_y_bin_count,
            (panel._custom_range_min_value, panel._custom_range_max_value),
            (panel._compare_y_custom_range_min_value, panel._compare_y_custom_range_max_value),
            scope_ids,
        )
        assert result["kind"] == "ok"
        queued.append((panel._histogram_compute_token, cache_key, result))

    monkeypatch.setattr(panel, "_queue_histogram_compute", queue)

    def deliver():
        panel._schedule_histogram_result(*queued.pop(0))
        scheduled.pop(0)()

    panel._refresh()
    deliver()
    panel._refresh_range_input_strings()
    panel._refresh_compare_y_input_strings()
    return panel, deliver


@pytest.mark.parametrize("axis", ["primary", "compare_y"])
@pytest.mark.parametrize("side,commits", [("min", ["0.35", "0.15"]), ("max", ["0.65", "0.85"])])
@pytest.mark.parametrize("blur_before_result", [False, True])
def test_range_commit_updates_field_after_delayed_result(pending_range_panel, axis, side, commits, blur_before_result):
    panel, deliver = pending_range_panel
    compare = axis == "compare_y"
    setter = getattr(panel, f"_set_compare_y_range_{side}" if compare else f"_set_custom_range_{side}")
    change = panel._on_compare_y_range_input_change if compare else panel._on_range_input_change
    blur = panel._on_compare_y_range_input_blur if compare else panel._on_range_input_blur
    string_attr = f"_compare_y_custom_range_{side}_str" if compare else f"_custom_range_{side}_str"
    bound_attr = f"_compare_y_{side}" if compare else f"_primary_histogram_{side}"
    constraint_attr = f"_compare_y_custom_range_{side}_value" if compare else f"_custom_range_{side}_value"
    for text in commits:
        old_bound = getattr(panel, bound_attr)
        setter(text)
        change(SimpleNamespace(get_bool_parameter=lambda name, default: False))
        assert getattr(panel, bound_attr) == old_bound
        change(SimpleNamespace(get_bool_parameter=lambda name, default: name == "linebreak"))
        # RmlUi sends Return/change then blur before the worker result arrives.
        if blur_before_result:
            blur(None)
        assert getattr(panel, constraint_attr) == float(text)
        deliver()
        assert getattr(panel, bound_attr) != old_bound
        assert getattr(panel, string_attr) == panel._format_range_input(getattr(panel, bound_attr))
        assert getattr(panel, constraint_attr) == float(text)  # Snapping only changes the display.


@pytest.mark.parametrize("axis", ["primary", "compare_y"])
@pytest.mark.parametrize("side", ["min", "max"])
def test_pending_result_does_not_replace_uncommitted_range_text(pending_range_panel, axis, side):
    panel, deliver = pending_range_panel
    panel._histogram_cache.clear()
    panel._refresh()
    compare = axis == "compare_y"
    setter = getattr(panel, f"_set_compare_y_range_{side}" if compare else f"_set_custom_range_{side}")
    string_attr = f"_compare_y_custom_range_{side}_str" if compare else f"_custom_range_{side}_str"
    setter("0.")
    deliver()
    assert getattr(panel, string_attr) == "0."


@pytest.mark.parametrize("axis", ["primary", "compare_y"])
@pytest.mark.parametrize("text", ["invalid", "nan", "0.95"])
def test_invalid_range_commit_preserves_bounds(pending_range_panel, axis, text):
    panel, _deliver = pending_range_panel
    compare = axis == "compare_y"
    prefix = "_compare_y_custom_range" if compare else "_custom_range"
    old_strings = tuple(getattr(panel, f"{prefix}_{side}_str") for side in ("min", "max"))
    # Primary opacity has natural bounds [0, 1]; compare uses the sample extent.
    if not compare:
        panel._custom_range_min_value = 0.1
        panel._custom_range_max_value = 0.9
    old_constraints = tuple(getattr(panel, f"{prefix}_{side}_value") for side in ("min", "max"))
    setter = panel._set_compare_y_range_min if compare else panel._set_custom_range_min
    commit = panel._commit_compare_y_range if compare else panel._commit_custom_range
    setter(text)
    commit()
    assert tuple(getattr(panel, f"{prefix}_{side}_str") for side in ("min", "max")) == old_strings
    assert tuple(getattr(panel, f"{prefix}_{side}_value") for side in ("min", "max")) == old_constraints
    assert not panel._computing


@pytest.mark.parametrize("axis", ["primary", "compare_y"])
def test_range_reset_and_cached_commit_update_fields(pending_range_panel, axis):
    panel, deliver = pending_range_panel
    compare = axis == "compare_y"
    prefix = "_compare_y_custom_range" if compare else "_custom_range"
    setter = panel._set_compare_y_range_min if compare else panel._set_custom_range_min
    commit = panel._commit_compare_y_range if compare else panel._commit_custom_range
    reset = panel._on_reset_compare_range if compare else panel._on_reset_range
    setter("0.35")
    commit()
    deliver()
    setter("0.")  # Reset intentionally discards pending text.
    reset(None, None, None)
    assert getattr(panel, f"{prefix}_min_value") is None
    assert getattr(panel, f"{prefix}_min_str") == "1e-1"
    setter("0.35")
    commit()  # The same range is now in the result cache.
    assert not panel._computing
    assert getattr(panel, f"{prefix}_min_str") == "4e-1"
    assert getattr(panel, f"{prefix}_min_value") == 0.35


@pytest.mark.parametrize("compare", [False, True])
def test_metric_switch_publishes_current_range_before_commit(pending_range_panel, compare):
    panel, deliver = pending_range_panel
    panel._set_compare_metric_id("position_y" if compare else "")
    deliver()
    for metric in ("erank", "position_x", "opacity"):
        panel._set_metric_id(metric)
        if panel._computing:
            deliver()
        assert panel._custom_range_min_str == panel._format_range_input(panel._primary_histogram_min)
        assert panel._custom_range_max_str == panel._format_range_input(panel._primary_histogram_max)
        if compare:
            assert panel._compare_y_custom_range_min_str == panel._format_range_input(panel._compare_y_min)
            assert panel._compare_y_custom_range_max_str == panel._format_range_input(panel._compare_y_max)
        counts = list(panel._hist_counts)
        # RmlUi can send the current value again on Return, without a text edit.
        panel._set_custom_range_min(panel._custom_range_min_str)
        panel._on_range_input_change(SimpleNamespace(get_bool_parameter=lambda name, default: name == "linebreak"))
        panel._on_range_input_blur(None)
        assert panel._custom_range_min_value is None
        assert panel._custom_range_max_value is None
        assert not panel._computing
        assert panel._hist_counts == counts


@pytest.mark.parametrize("side", ["min", "max"])
@pytest.mark.parametrize("axis", ["primary", "compare_y"])
def test_pending_metric_switch_cannot_commit_previous_metric(pending_range_panel, side, axis):
    panel, deliver = pending_range_panel
    compare = axis == "compare_y"
    switch = panel._set_compare_metric_id if compare else panel._set_metric_id
    commit = panel._commit_compare_y_range if compare else panel._commit_custom_range
    prefix = "_compare_y_custom_range" if compare else "_custom_range"
    setter = getattr(panel, f"_set_compare_y_range_{side}" if compare else f"_set_custom_range_{side}")
    switch("position_x")
    # The old snapshot stays visible while the different metric computes.
    setter("0.5")
    commit()
    assert getattr(panel, f"{prefix}_min_value") is None
    assert getattr(panel, f"{prefix}_max_value") is None
    deliver()
    assert getattr(panel, f"{prefix}_min_str") == "-4e1"
    assert getattr(panel, f"{prefix}_max_str") == "5.8e1"
    assert not panel._computing


def test_compare_metric_switch_publishes_both_y_bounds(pending_range_panel):
    panel, deliver = pending_range_panel
    for metric in ("position_x", "erank", "scale_x"):
        panel._set_compare_metric_id(metric)
        if panel._computing:
            deliver()
        assert panel._compare_y_custom_range_min_str == panel._format_range_input(panel._compare_y_min)
        assert panel._compare_y_custom_range_max_str == panel._format_range_input(panel._compare_y_max)
        panel._set_compare_y_range_max(panel._compare_y_custom_range_max_str)
        panel._on_compare_y_range_input_change(SimpleNamespace(get_bool_parameter=lambda name, default: name == "linebreak"))
        panel._on_compare_y_range_input_blur(None)
        assert panel._compare_y_custom_range_min_value is None
        assert panel._compare_y_custom_range_max_value is None
        assert not panel._computing


@pytest.mark.parametrize("axis", ["primary", "compare_y"])
def test_editing_one_bound_preserves_other_typed_constraint(pending_range_panel, axis):
    panel, deliver = pending_range_panel
    compare = axis == "compare_y"
    prefix = "_compare_y_custom_range" if compare else "_custom_range"
    set_min = panel._set_compare_y_range_min if compare else panel._set_custom_range_min
    set_max = panel._set_compare_y_range_max if compare else panel._set_custom_range_max
    commit = panel._commit_compare_y_range if compare else panel._commit_custom_range
    set_min("0.35")
    commit()
    deliver()
    assert getattr(panel, f"{prefix}_min_str") == "4e-1"
    set_max("0.65")
    commit()
    deliver()
    assert getattr(panel, f"{prefix}_min_value") == 0.35
    assert getattr(panel, f"{prefix}_max_value") == 0.65
    assert getattr(panel, f"{prefix}_min_str") == "4e-1"
    assert getattr(panel, f"{prefix}_max_str") == "6e-1"


@pytest.mark.parametrize("axis", ["primary", "compare_y"])
def test_outdated_metric_result_cannot_replace_current_range(pending_range_panel, axis):
    panel, deliver = pending_range_panel
    compare = axis == "compare_y"
    switch = panel._set_compare_metric_id if compare else panel._set_metric_id
    prefix = "_compare_y_custom_range" if compare else "_custom_range"
    switch("position_x")
    switch("position_y")
    deliver()  # Older metric result is rejected by the existing token/key check.
    assert panel._computing
    deliver()
    assert not panel._computing
    assert getattr(panel, f"{prefix}_min_str") == "-1e1"
    assert getattr(panel, f"{prefix}_max_str") == "2e1"


def test_bin_count_change_rebins_loaded_snapshot_without_the_worker(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    panel._handle = _UpdateHandleStub()
    panel._show_chart = True
    panel._metric_id = "opacity"
    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35, 0.65, 0.95], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()
    # A bin-count change keeps the extracted values; re-running the worker re-reads and
    # re-sorts every sample, which took ~0.5 s per change on a 3M-Gaussian scene.
    monkeypatch.setattr(panel, "_queue_histogram_compute", lambda *a, **k: pytest.fail("worker must not run"))
    monkeypatch.setattr(panel, "_extract_metric_values", lambda *a, **k: pytest.fail("values must not be re-extracted"))

    panel._set_histogram_bin_count(32)

    expected = [0] * 32
    for index in (1, 4, 11, 20, 30):
        expected[index] = 1
    assert panel._hist_counts == expected
    assert panel._handle.dirty_all_count >= 1


def test_zoom_snapping_uses_sorted_values_without_a_full_scan(histogram_panel_module, lf, numpy, monkeypatch):
    panel = histogram_panel_module.HistogramPanel()
    data = numpy.array([0.1, 0.25, 0.4, 0.55, 0.9], dtype=numpy.float32)
    panel._primary_valid_values = lf.Tensor.from_numpy(data)
    panel._primary_sorted_values = lf.Tensor.from_numpy(data)
    monkeypatch.setattr(
        histogram_panel_module.HistogramPanel, "_snap_bounds_to_data",
        staticmethod(lambda *a, **k: pytest.fail("zoom must not scan every sample")),
    )

    assert panel._snap_histogram_zoom_bounds_to_data(0.2, 0.6) == pytest.approx((0.25, 0.55))
    assert panel._snap_histogram_zoom_bounds_to_data(0.95, 1.0) == pytest.approx((0.95, 1.0))
    assert panel._snap_histogram_zoom_bounds_to_data(0.0, 1.0) == pytest.approx((0.1, 0.9))


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.2, 0.6), (0.4, 0.4), (0.41, 0.54), (0.9, 2.0), (-1.0, 0.1)])
def test_sorted_snapping_matches_the_full_scan(histogram_panel_module, lf, numpy, bounds):
    data = numpy.array([0.1, 0.25, 0.25, 0.4, 0.55, 0.9], dtype=numpy.float32)
    tensor = lf.Tensor.from_numpy(data)
    panel_type = histogram_panel_module.HistogramPanel
    assert panel_type._snap_sorted_bounds_to_data(tensor, *bounds) == panel_type._snap_bounds_to_data(tensor, *bounds)


def test_bin_counts_match_the_compacted_reference(histogram_panel_module, lf, numpy):
    rng = numpy.random.default_rng(7)
    data = rng.normal(0.5, 0.3, 2000).astype(numpy.float32)
    data[::97] = numpy.nan
    values = lf.Tensor.from_numpy(data)
    finite = values.isfinite()
    panel_type = histogram_panel_module.HistogramPanel
    bins = panel_type._bin_indices_for_values(values, 0.1, 0.9, 32, finite)
    counts = panel_type._bin_counts(bins, 32)
    valid = data[numpy.isfinite(data)]
    valid = valid[(valid >= 0.1) & (valid <= 0.9)]
    expected = numpy.clip(numpy.floor(((valid - numpy.float32(0.1)) / numpy.float32(0.8)) * 32), 0, 31).astype(int)
    assert counts == numpy.bincount(expected, minlength=32).tolist()
    assert bins.cpu().numpy()[~numpy.isfinite(data)].tolist() == [-1] * int((~numpy.isfinite(data)).sum())


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.1, 0.2), (0.33, 0.34), (0.5, 0.7), (0.95, 1.5), (-1.0, 0.004)])
def test_sampled_sorted_snapping_matches_the_full_scan_on_many_values(histogram_panel_module, lf, numpy, bounds, monkeypatch):
    monkeypatch.setattr(histogram_panel_module, "SORTED_SAMPLE_POINTS", 64)
    data = numpy.sort(numpy.random.default_rng(3).random(10_000).astype(numpy.float32))
    data[5000:5100] = data[5000]
    tensor = lf.Tensor.from_numpy(data)
    panel_type = histogram_panel_module.HistogramPanel
    sample = panel_type._sorted_sample(tensor)
    assert sample[0] > 1
    expected = panel_type._snap_bounds_to_data(tensor, *bounds)
    assert panel_type._snap_sorted_bounds_to_data(tensor, *bounds, sample=sample) == expected


@pytest.mark.parametrize("grid", [(8, 8), (8, 48), (48, 8), (24, 32), (48, 48)])
@pytest.mark.parametrize("mode", ["replace", "add", "remove", "intersect"])
@pytest.mark.parametrize("seed", [5, 23, 42, 99])
def test_compare_selection_seeded_drag_fuzz(histogram_panel_module, lf, numpy, monkeypatch, grid, mode, seed):
    """Exercise real press/move/release handlers against a per-point oracle."""
    rng = numpy.random.default_rng(seed)
    x_count, y_count = grid
    size = 521
    domain = rng.random(size) > 0.15
    xs = rng.integers(0, x_count, size, dtype=numpy.int32)
    ys = rng.integers(0, y_count, size, dtype=numpy.int32)
    # Dense cells and partially selected cells catch count shortcuts that only
    # work when the previous selection covers every point in each touched cell.
    xs[:200], ys[:200] = 0, 0
    xs[~domain], ys[~domain] = -1, -1
    xs[200:220], ys[200:220] = -1, -1  # Finite points outside the zoomed view.
    in_view = domain & (xs >= 0) & (ys >= 0)
    base = (rng.random(size) < 0.35) & domain
    scene = _SceneSelectionStub(lf.Tensor.from_numpy(base))
    monkeypatch.setattr(lf, "get_scene", lambda: scene)
    panel = histogram_panel_module.HistogramPanel()
    values = lf.Tensor.from_numpy(numpy.zeros(size, dtype=numpy.float32))
    panel._primary_values = panel._compare_values = values
    panel._primary_finite_mask = panel._compare_finite_mask = lf.Tensor.from_numpy(domain)
    panel._compare_x_bin_indices = lf.Tensor.from_numpy(xs)
    panel._compare_y_bin_indices = lf.Tensor.from_numpy(ys)
    panel._compare_x_bin_count, panel._compare_y_bin_count = grid
    panel._compare_x_edges = numpy.linspace(0, 1, x_count + 1).tolist()
    panel._compare_y_edges = numpy.linspace(0, 1, y_count + 1).tolist()
    panel._compare_counts = numpy.bincount((ys[in_view] * x_count + xs[in_view]), minlength=x_count*y_count).tolist()
    panel._show_chart = panel._show_compare_chart = True
    panel._compare_chart_el = SimpleNamespace(absolute_left=0, absolute_top=0,
                                              absolute_width=x_count*10, absolute_height=y_count*10,
                                              focus=lambda: None)
    panel._handle = _UpdateHandleStub()
    start = (0, 0)
    modifiers = {"shift": mode == "add", "ctrl": mode == "remove", "alt": mode == "intersect"}

    def event(cell):
        return _MouseEventStub(mouse_x=(cell[0]+0.5)*10,
                               mouse_y=(y_count-cell[1]-0.5)*10, **modifiers)

    panel._on_compare_chart_mousedown(event(start))
    endpoints = [(x_count-1, y_count-1), (0, 0)] + [tuple(rng.integers([0, 0], grid)) for _ in range(10)]
    for end in endpoints:
        panel._on_document_mousemove(event(end))
        swept = in_view & (xs <= end[0]) & (ys <= end[1])
        expected = {"replace": swept, "add": base | swept,
                    "remove": base & ~swept, "intersect": base & swept}[mode]
        numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), expected)
        assert panel._marked_count == int(expected.sum()), (grid, mode, seed, end)
        assert panel._marked_count == int(panel._panel_selection_mask.count_nonzero())
    panel._on_document_mouseup(event(endpoints[-1]))
    assert scene.commit_preview_calls == 1
    assert not panel._dragging_compare_mark


@pytest.mark.parametrize("seed", [5, 23, 42, 99])
@pytest.mark.parametrize("mode", ["replace", "add", "remove", "intersect"])
def test_histogram_partial_selection_count_fuzz(histogram_panel_module, lf, numpy, monkeypatch, seed, mode):
    rng = numpy.random.default_rng(seed)
    panel = histogram_panel_module.HistogramPanel()
    size, bins = 521, 16
    domain = rng.random(size) > 0.15
    indices = rng.integers(0, bins, size, dtype=numpy.int32)
    indices[:200] = 0
    indices[~domain] = -1
    indices[200:220] = -1  # Finite points outside the zoomed view.
    in_view = domain & (indices >= 0)
    base = (rng.random(size) < 0.35) & domain
    scene = _SceneSelectionStub(lf.Tensor.from_numpy(base))
    monkeypatch.setattr(lf, "get_scene", lambda: scene)
    panel._primary_values = lf.Tensor.from_numpy(numpy.zeros(size, dtype=numpy.float32))
    panel._primary_finite_mask = lf.Tensor.from_numpy(domain)
    panel._selection_bin_indices = lf.Tensor.from_numpy(indices)
    panel._histogram_bin_count = bins
    panel._hist_counts = numpy.bincount(indices[in_view], minlength=bins).tolist()
    panel._hist_prefix_counts = panel._prefix_counts(panel._hist_counts)
    panel._hist_edges = numpy.linspace(0, 1, bins+1).tolist()
    panel._show_chart = True
    panel._chart_el = SimpleNamespace(absolute_left=0, absolute_width=160, focus=lambda: None)
    modifiers = {"shift": mode == "add", "ctrl": mode == "remove", "alt": mode == "intersect"}
    def event(index):
        left, width = panel._histogram_selection_geometry(index, index)
        return _MouseEventStub(mouse_x=left+width/2, **modifiers)
    panel._on_chart_mousedown(event(0))
    for end in [15, 0] + rng.integers(0, bins, 10).tolist():
        panel._on_document_mousemove(event(end))
        swept = in_view & (indices <= end)
        expected = {"replace": swept, "add": base | swept,
                    "remove": base & ~swept, "intersect": base & swept}[mode]
        numpy.testing.assert_array_equal(scene.selection_mask.cpu().numpy(), expected)
        assert panel._marked_count == int(expected.sum())
    panel._on_document_mouseup(event(end))
    assert scene.commit_preview_calls == 1


def test_compare_records_cache_selection_and_invalidation(histogram_panel_module, monkeypatch):
    module = histogram_panel_module
    panel = module.HistogramPanel()
    panel._compare_x_bin_count, panel._compare_y_bin_count = 8, 9
    panel._compare_counts = list(range(72))
    panel._compare_x_edges = list(range(9))
    panel._compare_y_edges = list(range(10))
    calls = []
    original = panel._format_range_text
    monkeypatch.setattr(panel, "_format_range_text", lambda lo, hi: (calls.append((lo, hi)), original(lo, hi))[1])
    first = panel._build_compare_bin_records()
    assert len(calls) == 17  # Each axis range is formatted once, not per cell.
    for selected in [{(0, 0)}, {(7, 8)}, {(0, 0), (7, 8)}]:
        panel._selected_compare_cells = selected
        records = panel._build_compare_bin_records()
        assert len(calls) == 17
        assert [record["selected"] for record in records] == [
            (index % 8, 8-index//8) in selected for index in range(72)]
        assert [{k: v for k, v in record.items() if k != "selected"} for record in records] == [
            {k: v for k, v in record.items() if k != "selected"} for record in first]
    assert not any(record["selected"] for record in first)
    for change in [lambda: panel._compare_counts.__setitem__(0, 300),
                   lambda: panel._compare_x_edges.__setitem__(0, -1),
                   lambda: panel._compare_y_edges.__setitem__(0, -2),
                   lambda: setattr(panel, "_log_scale_enabled", True),
                   lambda: setattr(panel, "_last_lang", "de")]:
        previous = panel._static_compare_records_value
        change()
        panel._build_compare_bin_records()
        assert panel._static_compare_records_value is not previous
    panel._clear_compare_cache()
    assert panel._build_compare_bin_records() == []
    assert panel._static_compare_records_value is None


def test_histogram_bar_color_is_constant_while_heatmap_encodes_density(histogram_panel_module):
    panel = histogram_panel_module.HistogramPanel()
    for log in [False, True]:
        panel._log_scale_enabled = log
        bars = list(panel._build_bin_records([0, 100, 500, 1000], [0, 1, 2, 3, 4]))
        assert [bar["opacity_style"] for bar in bars] == ["0.000", "1.000", "1.000", "1.000"]
        assert len({bar["height_style"] for bar in bars}) == 4
        panel._compare_x_bin_count, panel._compare_y_bin_count = 2, 2
        panel._compare_counts = [0, 1, 10, 1000]
        panel._compare_x_edges = panel._compare_y_edges = [0, 1, 2]
        heatmap = panel._build_compare_bin_records()
        assert len({cell["opacity_style"] for cell in heatmap}) == 4
