# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera Selection popup and ROI toolbar regressions."""

from pathlib import Path


def test_distinct_and_roi_controls_use_editable_scrub_fields():
    root = Path(__file__).parents[2]
    panel = (root / "src/python/lfs_plugins/camera_selection_panel.py").read_text(encoding="utf-8")
    popup = (root / "src/visualizer/gui/rmlui/resources/camera_selection.rml").read_text(encoding="utf-8")

    assert "ScrubFieldController" in panel
    assert '"percentage": ScrubFieldSpec(0, 100, 1, "%d", data_type=int)' in panel
    assert '"overlap": ScrubFieldSpec(0, 100, 1, "%d", data_type=int)' in panel
    assert '"coverage": ScrubFieldSpec(0, 100, 1, "%d", data_type=int)' in panel
    assert "self._scrub_fields.mount(doc)" in panel
    assert "self._scrub_fields.sync_all()" in panel
    assert 'data-value="percentage" min="0" max="100" step="1"' in popup
    assert 'data-value="overlap" min="0" max="100" step="1"' in popup
    assert 'data-value="coverage" min="0" max="100" step="1"' in popup


def test_roi_popup_keeps_coverage_retained_without_a_cropbox_picker():
    root = Path(__file__).parents[2]
    panel = (root / "src/python/lfs_plugins/camera_selection_panel.py").read_text(encoding="utf-8")
    popup = (root / "src/visualizer/gui/rmlui/resources/camera_selection.rml").read_text(encoding="utf-8")

    assert 'data-class-hidden="!is_distinct"' in popup
    assert 'model.bind_record_list("cropboxes")' not in panel
    assert '<div data-if="is_distinct">' not in popup
    assert 'data-if="is_roi"' not in popup
    assert 'data-class-hidden="!is_roi"' in popup


def test_popup_title_tracks_the_selected_operation_with_i18n_keys():
    root = Path(__file__).parents[2]
    panel = (root / "src/python/lfs_plugins/camera_selection_panel.py").read_text(encoding="utf-8")

    assert 'model.bind_func("panel_label", self._panel_label)' in panel
    assert '"random": "scene.select_random_cameras"' in panel
    assert '"every_n": "scene.select_every_n_cameras"' in panel
    assert '"distinct": "scene.select_distinct_cameras"' in panel
    assert '"roi_visibility": "scene.select_roi_cameras"' in panel
    assert '"panel_label"' in panel


def test_panel_rebind_uses_roi_state_and_preserves_local_validation_errors():
    root = Path(__file__).parents[2]
    panel = (root / "src/python/lfs_plugins/camera_selection_panel.py").read_text(encoding="utf-8")

    assert "_cropbox_records" not in panel
    assert "self._local_error = \"\"" in panel
    assert "self._task_error = \"\"" in panel
    assert "lambda: self._local_error or self._task_error" in panel
    assert "self._local_error = str(error)" in panel


def test_camera_selection_uses_viewer_thread_and_safe_crop_projection():
    root = Path(__file__).parents[2]
    module = (root / "src/python/lfs/module.cpp").read_text(encoding="utf-8")
    tasks = (root / "src/visualizer/gui/async_task_manager.cpp").read_text(encoding="utf-8")
    algorithms = (root / "src/visualizer/gui/camera_selection_algorithms.hpp").read_text(encoding="utf-8")
    scene_graph = (root / "src/visualizer/gui/rmlui/elements/scene_graph_element.cpp").read_text(encoding="utf-8")
    interaction = (root / "src/visualizer/rendering/camera_interaction_service.cpp").read_text(encoding="utf-8")

    assert '"python.camera_selection_start"' in module
    assert '"python.camera_selection_status"' in module
    assert '"python.camera_selection_cancel"' in module
    assert "post_guarded_and_wait" in module
    assert "camera.center_x" in algorithms and "camera.center_y" in algorithms
    assert "camera.world_to_camera" in algorithms
    assert "camera_selection_roi_pinhole_required" in tasks
    assert "camera_selection_stale" in tasks
    assert "same_roi_points" in tasks
    assert "panel->on_content_changed()" in scene_graph
    assert "enabled_cameras" in interaction


def test_camera_selection_is_group_scoped_and_reports_validation_errors():
    root = Path(__file__).parents[2]
    module = (root / "src/python/lfs/module.cpp").read_text(encoding="utf-8")
    tasks = (root / "src/visualizer/gui/async_task_manager.cpp").read_text(encoding="utf-8")
    header = (root / "src/visualizer/gui/async_task_manager.hpp").read_text(encoding="utf-8")
    scene = (root / "src/core/scene.cpp").read_text(encoding="utf-8")
    graph = (root / "src/visualizer/gui/rmlui/elements/scene_graph_element.cpp").read_text(encoding="utf-8")
    mcp = (root / "src/app/mcp_gui_tools.cpp").read_text(encoding="utf-8")

    assert "camera_group_id" in module
    assert "camera_group_id" in header
    assert "setCameraGroupTrainingSelection" in scene
    assert "cropbox_world_to_local" in tasks
    assert "camera_group" in mcp
    assert '"enum", json::array' in mcp
    assert "set_camera_selection_group_id" in graph
    assert '"select_cameras"' not in module


def test_roi_activation_materializes_a_visible_target_owned_roi_and_toolbar_state():
    root = Path(__file__).parents[2]
    ui = (root / "src/python/lfs/py_ui.cpp").read_text(encoding="utf-8")
    gizmo = (root / "src/visualizer/gui/gizmo_manager.cpp").read_text(encoding="utf-8")
    graph = (root / "src/visualizer/gui/rmlui/elements/scene_graph_element.cpp").read_text(encoding="utf-8")
    toolbar = (root / "src/python/lfs_plugins/toolbar.py").read_text(encoding="utf-8")
    rml = (root / "src/visualizer/gui/rmlui/resources/viewport_overlay.rml").read_text(encoding="utf-8")

    assert 'id == "builtin.roi"' in ui
    assert "ensureCropToolStateForRestore" in ui
    assert "world_min = {-1.0F, -1.0F, -1.0F}" in gizmo
    assert 'snapshot.label += "  [ROI]"' not in graph
    assert 'node->name + "_ROI"' in graph
    assert "def _resolve_selected_roi()" in toolbar
    assert "ROI Settings" in toolbar
    assert "roi_crop_point_cloud_before_training" in rml
    assert "roi_select_relevant_cameras" in rml
    assert "roi_camera_coverage_percent" in rml
    assert (
        '"crop_fit",\n'
        '            "crop_trim",\n'
        '            "crop_reset",\n'
        '            "crop_toggle_inverse",\n'
        '            "crop_delete",'
    ) in toolbar


def test_roi_toolbar_runtime_records_and_param_sync_do_not_raise():
    import sys
    import types

    root = Path(__file__).parents[2]
    plugins_parent = str(root / "src/python")
    if plugins_parent not in sys.path:
        sys.path.insert(0, plugins_parent)

    fake_ui = types.SimpleNamespace(
        tr=lambda key: key,
        get_active_tool=lambda: "builtin.roi",
        get_gizmo_type=lambda: "translate",
        get_pivot_mode=lambda: 0,
        get_crop_tool_shape=lambda: "box",
        get_roi_training_settings=lambda _name=None: {
            "exists": True,
            "enabled": True,
            "locked": False,
            "target_name": "Model",
            "roi_node_name": "Model_ROI",
            "shape": "box",
            "inverse": True,
            "crop_point_cloud_before_training": True,
            "crop_model_after_training": False,
            "select_relevant_cameras": True,
            "camera_coverage_percent": 50.0,
        },
        get_selected_node_cropbox=lambda: None,
        get_selected_node_ellipsoid=lambda: None,
        get_selected_node_crop_polygon=lambda: None,
        get_context=lambda: None,
    )
    fake_node = types.SimpleNamespace(id=1, name="Model", type="NodeType.SPLAT", children=[])
    fake_scene = types.SimpleNamespace(
        get_node=lambda name: fake_node if name == "Model" else None,
        get_node_by_id=lambda _id: fake_node,
        get_nodes=lambda: [fake_node],
        is_empty=lambda: False,
    )
    fake_lf = types.ModuleType("lichtfeld")
    fake_lf.ui = fake_ui
    fake_lf.get_scene = lambda: fake_scene
    fake_lf.get_selected_node_names = lambda: ["Model"]
    fake_lf.get_selected_node_name = lambda: "Model"
    fake_lf.optimization_params = lambda: types.SimpleNamespace(
        has_params=lambda: True,
        cropbox_lr_scale=0.25,
        cropbox_loss_weight=0.5,
        cropbox_growth_inside=True,
        crop_to_cropbox_at_end=False,
    )

    previous_lf = sys.modules.get("lichtfeld")
    sys.modules["lichtfeld"] = fake_lf
    try:
        from lfs_plugins import toolbar as tb

        controller = tb._ViewportToolbarController()
        action_records = controller._gizmo._build_crop_action_records("builtin.roi")
        button_ids = [record["button_id"] for record in action_records]
        assert button_ids == [
            "crop-fit",
            "crop-trim",
            "crop-reset",
            "crop-invert",
            "crop-delete",
        ]

        sig = controller._gizmo.cropbox_toolbar_signature()
        assert sig[1] == "roi"
        controller._sync_crop_roi_params(sig)
    finally:
        if previous_lf is None:
            sys.modules.pop("lichtfeld", None)
        else:
            sys.modules["lichtfeld"] = previous_lf


if __name__ == "__main__":
    test_distinct_and_roi_controls_use_editable_scrub_fields()
    test_roi_popup_keeps_coverage_retained_without_a_cropbox_picker()
    test_popup_title_tracks_the_selected_operation_with_i18n_keys()
    test_panel_rebind_uses_roi_state_and_preserves_local_validation_errors()
    test_camera_selection_uses_viewer_thread_and_safe_crop_projection()
    test_camera_selection_is_group_scoped_and_reports_validation_errors()
    test_roi_activation_materializes_a_visible_target_owned_roi_and_toolbar_state()
    test_roi_toolbar_runtime_records_and_param_sync_do_not_raise()
