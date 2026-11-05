# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Draft/revision regressions independent of model availability or GPU timing."""

from test_rmlui_image_sources import panel_modules
import pytest
from types import SimpleNamespace
from test_preview_workflow import Element, event
from test_rmlui_image_sources import _DocumentStub


def editor(module):
    value = module.FrameMaskEditor()
    value.enabled = True
    value.bind("frame.png", "revision-a")
    return value


def test_prompts_follow_their_image_and_undo_is_local(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    value.point(0.2, 0.4, 1)
    value.set_box((0.8, 0.9), (0.1, 0.2))
    value.bind("other.png", "revision-b")
    value.point(0.6, 0.7, 0)
    value.bind("frame.png", "revision-a")
    assert value.draft.points == [(0.2, 0.4, 1)]
    assert value.draft.box == (0.1, 0.2, 0.8, 0.9)
    value.undo()
    assert value.draft.box is None
    assert value.draft.points == [(0.2, 0.4, 1)]
    value.bind("other.png", "revision-b")
    assert value.draft.points == [(0.6, 0.7, 0)]


def test_replacement_invalidates_prompts_and_pending_results(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    value.point(0.2, 0.4, 1)
    value.request = 8
    value.bind("frame.png", "replacement")
    assert not value.draft.points
    assert not value.accept({"request": 8, "revision": "revision-a", "phase": "ready",
                             "overlays": ["stale.png"], "scores": [0.9]})
    assert value.overlay is None


def test_newer_request_wins_and_refinement_survives_prompt_correction(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    value.request = 9
    assert not value.accept({"request": 8, "revision": "revision-a", "phase": "error", "error": "old"})
    assert value.accept({"request": 9, "revision": "revision-a", "phase": "ready",
                         "overlays": ["one.png", "two.png"], "scores": [0.8, 0.9]})
    assert value.overlay == "one.png"
    value.point(0.4, 0.6, 0)
    assert value.overlay is None
    assert value.refinement == (9, 0)
    assert not value.accept({"request": 9, "revision": "revision-a", "phase": "ready",
                             "overlays": ["old.png"], "scores": [0.8]})
    value.clear()
    assert value.refinement is None
    value.undo()
    assert value.draft.points == [(0.4, 0.6, 0)]


def test_prompt_memory_is_bounded_and_points_do_not_become_multiple_objects(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    for index in range(400):
        value.point(index / 400, 0.5, 1)
    assert len(value.draft.points) == 256
    assert len(value.draft.history) == 64
    assert len(value.drafts) == 1


def test_cancel_preserves_prompts_and_rejects_late_results(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(0.3, 0.4, 1)
    value.request = 12
    closed = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_close", lambda: closed.append(True), raising=False)
    panel._sam_cancel()
    assert closed == [True]
    assert value.draft.points == [(0.3, 0.4, 1)]
    assert not value.accept({"request": 12, "revision": "revision-a", "phase": "ready"})


@pytest.mark.parametrize("changed_source", [False, True])
def test_apply_targets_only_the_reviewed_image_and_ignores_late_results(panel_modules, monkeypatch, changed_source):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._mask_editor = value = editor(module)
    value.request, value.phase = 42, "rendering"
    panel._sam_apply_target = (42, "camera-uuid", value.key, module.RuntimeState.scene_generation.value)
    panel._path_revision_cache["frame.png"] = "changed" if changed_source else "revision-a"
    applied = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"request": 42, "phase": "mask_ready"}, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_apply", lambda *args: applied.append(args), raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_close", lambda: None, raising=False)
    monkeypatch.setattr(panel, "_refresh_sources", lambda: None)
    panel._service_sam()
    panel._service_sam()
    assert applied == ([] if changed_source else [("camera-uuid", 42)])
    assert panel._sam_apply_target is None
    if not changed_source:
        assert panel._show_overlay and not value.enabled


@pytest.mark.parametrize("exclude,inverted", [(False, False), (False, True), (True, False), (True, True)])
def test_apply_keeps_selected_polarity_with_inverted_training_masks(panel_modules, monkeypatch, exclude, inverted):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(0.5, 0.5, 1)
    value.request, value.phase, value.overlays = 5, "ready", ["proposal.png"]
    panel._path_stat_cache["frame.png"] = (True, 100, 200, 0, 0, 0)
    monkeypatch.setattr(panel, "_current_camera_node", lambda **kw: SimpleNamespace(uuid="camera-uuid"))
    monkeypatch.setattr(module.lf, "optimization_params", lambda: SimpleNamespace(invert_masks=inverted), raising=False)
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *args: submitted.append(args) or 6, raising=False)
    panel._sam_apply(exclude)
    assert submitted[0][-1] == (exclude != inverted)
    assert panel._sam_apply_target == (6, "camera-uuid", value.key, module.RuntimeState.scene_generation.value)


def test_delete_move_and_change_label_have_local_undo_and_discard_old_mask_prior(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    value.point(0.2, 0.4, 1)
    value.point(0.7, 0.8, 0)
    value.refinement = (9, 0)
    value.selected = 0
    value.move_selected(0.3, 0.5)
    assert value.draft.points == [(0.3, 0.5, 1), (0.7, 0.8, 0)]
    assert value.refinement is None
    value.flip_selected()
    assert value.draft.points[0] == (0.3, 0.5, 0)
    value.remove_selected()
    assert value.draft.points == [(0.7, 0.8, 0)]
    assert value.selected is None
    value.undo()
    assert value.draft.points == [(0.3, 0.5, 0), (0.7, 0.8, 0)]
    value.undo()
    value.undo()
    assert value.draft.points == [(0.2, 0.4, 1), (0.7, 0.8, 0)]


def test_box_can_be_removed_without_deleting_the_points(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    value.point(0.2, 0.4, 1)
    value.set_box((0.1, 0.1), (0.8, 0.9))
    assert value.selected == "box"
    value.remove_selected()
    assert value.draft.points == [(0.2, 0.4, 1)]
    assert value.draft.box is None
    value.undo()
    assert value.draft.box == (0.1, 0.1, 0.8, 0.9)


def test_refinement_only_uses_a_prior_after_prompt_changes(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    value.point(0.3, 0.5, 1)
    value.request = 9
    value.submitted_signature = value.signature
    value.accept({"request": 9, "revision": "revision-a", "phase": "ready",
                  "overlays": ["one.png"], "scores": [0.9]})
    assert not value.can_refine
    value.point(0.8, 0.8, 0)
    assert value.can_refine
    value.remove_selected()
    assert not value.can_refine, "deletion must not retain a mask prior containing the deleted constraint"


def test_delete_key_does_not_remove_a_prompt_while_typing(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(0.2, 0.4, 1)
    panel._on_keydown(event(Element("input"), key_identifier=module.KI_DELETE))
    assert len(value.draft.points) == 1
    panel._on_keydown(event(key_identifier=module.KI_DELETE))
    assert not value.draft.points


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_select_drag_and_right_click_remove_existing_point(panel_modules, monkeypatch, scale, rotation):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._image_info_cache["frame.png"] = (400, 200, 3)
    panel._displayed_index = panel._current_index = 0
    panel._rotation_quadrants = rotation
    panel._mask_editor = value = editor(module)
    value.point(0.25, 0.40, 1)
    value.selected = None
    viewport = Element()
    viewport.client_width, viewport.client_height = 900, 700
    viewport.absolute_left, viewport.absolute_top = 130, 75
    panel._doc = _DocumentStub({"image-viewport": viewport})
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: scale)
    layout = panel._get_image_layout(panel._image_paths[0], viewport)

    def mouse(x, y, button=0):
        vx, vy = module.source_to_view(layout, rotation, x * 399, y * 199)
        return event(mouse_x=viewport.absolute_left + vx * scale,
                     mouse_y=viewport.absolute_top + vy * scale, button=button)

    panel._on_img_mousedown(mouse(0.25, 0.40))
    assert value.selected == 0
    assert len(value.draft.points) == 1, "clicking a marker must not add another point"
    panel._sam_tool("edit")
    panel._on_img_mousedown(mouse(0.25, 0.40))
    panel._on_img_mousemove(mouse(0.6, 0.55))
    panel._on_img_mousemove(mouse(0.65, 0.6))
    assert len(value.draft.history) == 1, "a drag must create only one undo entry"
    panel._on_img_mouseup(mouse(0.65, 0.6))
    assert value.draft.points[0][:2] == pytest.approx((0.65, 0.6))
    assert len(value.draft.history) == 2
    panel._on_img_mousedown(mouse(0.65, 0.6, button=1))
    assert not value.draft.points
    value.undo()
    assert value.draft.points[0][:2] == pytest.approx((0.65, 0.6))


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_sam_pointer_uses_original_pixels_after_fit_rotation_and_hidpi(panel_modules, monkeypatch, scale, rotation):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._image_info_cache["frame.png"] = (400, 200, 3)
    panel._displayed_index = panel._current_index = 0
    panel._rotation_quadrants = rotation
    viewport = Element()
    viewport.client_width, viewport.client_height = 900, 700
    viewport.absolute_left, viewport.absolute_top = 130, 75
    panel._doc = _DocumentStub({"image-viewport": viewport})
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: scale)
    layout = panel._get_image_layout(panel._image_paths[0], viewport)
    x, y = module.source_to_view(layout, rotation, 100.5, 75.25)
    mouse = event(mouse_x=viewport.absolute_left + x * scale, mouse_y=viewport.absolute_top + y * scale)
    assert panel._sam_source_point(mouse) == pytest.approx((100.5 / 399, 75.25 / 199))
    assert panel._sam_source_point(event(mouse_x=-10000, mouse_y=-10000)) is None
