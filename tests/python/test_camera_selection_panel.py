# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera Selection popup control regressions."""

from pathlib import Path


def test_distinct_and_cropbox_controls_use_editable_scrub_fields():
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


def test_cropbox_popup_registers_its_record_list_and_keeps_coverage_retained():
    root = Path(__file__).parents[2]
    panel = (root / "src/python/lfs_plugins/camera_selection_panel.py").read_text(encoding="utf-8")
    popup = (root / "src/visualizer/gui/rmlui/resources/camera_selection.rml").read_text(encoding="utf-8")

    assert 'data-class-hidden="!is_distinct"' in popup
    assert 'model.bind_record_list("cropboxes")' in panel
    assert '<div data-if="is_distinct">' not in popup
    assert 'data-if="is_cropbox"' in popup
    assert 'data-class-hidden="!is_cropbox"' in popup
    assert "if len(records) == 1 and not self._cropbox_id:" in panel
    assert 'self._cropbox_id = records[0]["id"]' in panel


def test_popup_title_tracks_the_selected_operation_with_i18n_keys():
    root = Path(__file__).parents[2]
    panel = (root / "src/python/lfs_plugins/camera_selection_panel.py").read_text(encoding="utf-8")

    assert 'model.bind_func("panel_label", self._panel_label)' in panel
    assert '"random": "scene.select_random_cameras"' in panel
    assert '"every_n": "scene.select_every_n_cameras"' in panel
    assert '"distinct": "scene.select_distinct_cameras"' in panel
    assert '"cropbox_visibility": "scene.select_cropbox_cameras"' in panel
    assert '"panel_label"' in panel


def test_panel_rebind_refreshes_cropboxes_and_preserves_local_validation_errors():
    root = Path(__file__).parents[2]
    panel = (root / "src/python/lfs_plugins/camera_selection_panel.py").read_text(encoding="utf-8")

    assert "self._cropbox_records = None" in panel
    assert "self._local_error = \"\"" in panel
    assert "self._task_error = \"\"" in panel
    assert "lambda: self._local_error or self._task_error" in panel
    assert "self._local_error = str(error)" in panel


def test_camera_selection_uses_viewer_thread_and_safe_crop_projection():
    root = Path(__file__).parents[2]
    module = (root / "src/python/lfs/module.cpp").read_text(encoding="utf-8")
    tasks = (root / "src/visualizer/gui/async_task_manager.cpp").read_text(encoding="utf-8")
    scene_graph = (root / "src/visualizer/gui/rmlui/elements/scene_graph_element.cpp").read_text(encoding="utf-8")
    interaction = (root / "src/visualizer/rendering/camera_interaction_service.cpp").read_text(encoding="utf-8")

    assert '"python.select_cameras"' in module
    assert '"python.camera_selection_status"' in module
    assert '"python.camera_selection_cancel"' in module
    assert "post_guarded_and_wait" in module
    assert "camera.center_x" in tasks and "camera.center_y" in tasks
    assert "camera.world_to_camera" in tasks
    assert "camera_selection_pinhole_required" in tasks
    assert "camera_selection_stale" in tasks
    assert "same_cropbox_points" in tasks
    assert "panel->on_content_changed()" in scene_graph
    assert "enabled_cameras" in interaction


if __name__ == "__main__":
    test_distinct_and_cropbox_controls_use_editable_scrub_fields()
    test_cropbox_popup_registers_its_record_list_and_keeps_coverage_retained()
    test_popup_title_tracks_the_selected_operation_with_i18n_keys()
    test_panel_rebind_refreshes_cropboxes_and_preserves_local_validation_errors()
    test_camera_selection_uses_viewer_thread_and_safe_crop_projection()
