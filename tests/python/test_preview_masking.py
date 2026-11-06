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
    value.edits = []
    value.on_edit = lambda key, before, after: value.edits.append((key, before, after))
    value.enabled = True
    value.bind("frame.png", "revision-a")
    return value



def undo(value):
    key, before, _ = value.edits.pop()
    value.restore(key, before)


def test_edits_restore_their_source_draft_without_changing_the_active_image(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    value.point(0.2, 0.4, 1)
    value.set_box((0.8, 0.9), (0.1, 0.2))
    value.bind("other.png", "revision-b")
    value.point(0.6, 0.7, 0)
    value.bind("frame.png", "revision-a")
    assert value.draft.points == [(0.2, 0.4, 1)]
    assert value.draft.box == (0.1, 0.2, 0.8, 0.9)
    undo(value)  # Shared chronology undoes the most recent edit on the other image.
    assert value.draft.box == (0.1, 0.2, 0.8, 0.9)
    assert not value.drafts["other.png"].points
    undo(value)
    assert value.draft.box is None
    assert value.draft.points == [(0.2, 0.4, 1)]


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
    undo(value)
    assert value.draft.points == [(0.4, 0.6, 0)]


def test_prompt_memory_is_bounded_and_points_do_not_become_multiple_objects(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    for index in range(400):
        value.point(index / 400, 0.5, 1)
    assert len(value.draft.points) == 256
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


@pytest.mark.parametrize("changed_source,changed_scene", [(False, False), (True, False), (False, True)])
def test_apply_targets_only_the_reviewed_image_and_ignores_late_results(panel_modules, monkeypatch, changed_source, changed_scene):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._mask_editor = value = editor(module)
    value.request, value.phase = 41, "rendering"
    panel._sam_apply_target = module.MaskApply(42, 41, "camera-uuid", value.key, module.RuntimeState.scene_generation.value)
    if changed_scene:
        module.RuntimeState.scene_generation.value += 1
    panel._path_revision_cache["frame.png"] = "changed" if changed_source else "revision-a"
    applied = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"request": 42, "phase": "mask_ready"}, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_apply", lambda *args: applied.append(args), raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_close", lambda: None, raising=False)
    monkeypatch.setattr(panel, "_refresh_sources", lambda: None)
    panel._service_sam()
    panel._service_sam()
    assert applied == ([] if changed_source or changed_scene else [("camera-uuid", 42)])
    assert panel._sam_apply_target is None
    if changed_source or changed_scene:
        assert value.phase == "error", "a stale Apply must not leave the editor busy"
    else:
        assert panel._show_overlay and not value.enabled


@pytest.mark.parametrize("exclude,inverted", [(False, False), (False, True), (True, False), (True, True)])
def test_apply_keeps_selected_polarity_with_inverted_training_masks(panel_modules, monkeypatch, exclude, inverted):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(0.5, 0.5, 1)
    class Proposal:
        def __bool__(self):
            raise AssertionError("An image tensor is not a scalar boolean")
    value.request, value.phase, value.overlays = 5, "ready", [Proposal()]
    panel._path_stat_cache["frame.png"] = (True, 100, 200, 0, 0, 0)
    monkeypatch.setattr(panel, "_current_camera_node", lambda **kw: SimpleNamespace(uuid="camera-uuid"))
    monkeypatch.setattr(module.lf, "optimization_params", lambda: SimpleNamespace(invert_masks=inverted), raising=False)
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *args: submitted.append(args) or 6, raising=False)
    panel._sam_apply(exclude)
    assert submitted[0][-1] == exclude
    assert value.request == 5
    assert panel._sam_apply_target == module.MaskApply(6, 5, "camera-uuid", value.key, module.RuntimeState.scene_generation.value)


@pytest.mark.parametrize("export_error", [False, True])
def test_apply_failure_preserves_prediction_for_retry_and_refinement(panel_modules, monkeypatch, export_error):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._path_revision_cache["frame.png"] = "revision-a"
    panel._path_stat_cache["frame.png"] = (True, 100, 200, 0, 0, 0)
    panel._preview_state = "ready"
    panel._mask_editor = value = editor(module)
    value.point(0.5, 0.5, 1)
    value.request, value.phase = 5, "ready"
    value.overlays, value.scores = [object(), object()], [0.9, 0.8]
    value.submitted_signature = value.signature
    monkeypatch.setattr(panel, "_current_camera_node", lambda **kw: SimpleNamespace(uuid="camera-uuid"))
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *args: submitted.append(args) or (5 + len(submitted)), raising=False)
    panel._sam_apply(False)
    panel._sam_candidate(1)
    panel._sam_generate()
    panel._sam_apply(True)
    panel._sam_edit("clear")
    assert len(submitted) == 1
    assert value.candidate == 0 and value.draft.points
    assert value.request == 5

    def reject(*args):
        raise RuntimeError("project busy")

    monkeypatch.setattr(module.lf.ui, "sam_preview_apply", reject, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {
        "request": 6, "phase": "error" if export_error else "mask_ready", "error": "render failed"}, raising=False)
    panel._service_sam()
    assert panel._sam_apply_target is None
    assert value.phase == "ready" and value.error
    assert value.request == 5 and len(value.overlays) == 2
    panel._sam_candidate(1)
    assert value.refinement == (5, 1)
    panel._sam_apply(True)
    assert submitted[-1][6] == 5  # Export always references the prediction, never the failed export.
    assert submitted[-1][-2:] == (1, True)


def test_delete_move_and_change_label_restore_and_discard_old_mask_prior(panel_modules):
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
    undo(value)
    assert value.draft.points == [(0.3, 0.5, 0), (0.7, 0.8, 0)]
    undo(value)
    undo(value)
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
    undo(value)
    assert value.draft.box == (0.1, 0.1, 0.8, 0.9)


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("rotation", range(4))
def test_box_corner_resize_and_move_are_bounded_single_undo_edits(panel_modules, monkeypatch, scale, rotation):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._image_info_cache["frame.png"] = (400, 200, 3)
    panel._displayed_index = 0
    panel._rotation_quadrants = rotation
    panel._mask_editor = value = editor(module)
    value.set_box((0.1, 0.1), (0.8, 0.9))
    value.tool = "box"
    viewport = Element()
    viewport.client_width, viewport.client_height = 900, 700
    panel._doc = _DocumentStub({"image-viewport": viewport})
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: scale)
    layout = panel._get_image_layout(panel._image_paths[0])
    def mouse(x, y):
        vx, vy = module.source_to_view(layout, rotation, x * 399, y * 199)
        return event(mouse_x=vx * scale, mouse_y=vy * scale)
    panel._on_img_mousedown(mouse(0.1, 0.1))
    panel._on_img_mousemove(mouse(0.2, 0.3))
    assert len(value.edits) == 1
    panel._on_img_mouseup(mouse(0.2, 0.3))
    assert value.draft.box == pytest.approx((0.2, 0.3, 0.8, 0.9))
    assert len(value.edits) == 2
    panel._on_img_mousedown(mouse(0.5, 0.6))
    panel._on_img_mouseup(mouse(0.9, 0.95))
    assert value.draft.box == pytest.approx((0.4, 0.4, 1.0, 1.0))
    assert len(value.edits) == 3
    undo(value)
    assert value.draft.box == pytest.approx((0.2, 0.3, 0.8, 0.9))
    undo(value)
    assert value.draft.box == (0.1, 0.1, 0.8, 0.9)


def test_selecting_an_unchanged_box_does_not_discard_its_proposal(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    value.set_box((0.1, 0.2), (0.8, 0.9))
    value.phase, value.overlays = "ready", [object()]
    value.set_box((0.1, 0.2), (0.8, 0.9))
    assert value.phase == "ready" and value.overlay is not None


def test_control_selector_supports_keyboard_selection_and_shared_edits(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(0.2, 0.4, 1)
    value.point(0.7, 0.8, 0)
    value.set_box((0.1, 0.1), (0.8, 0.9))
    panel._sam_select_control(event(value="1"))
    value.flip_selected()
    assert value.draft.points[1][2] == 1
    panel._sam_select_control(event(value="box"))
    value.remove_selected()
    assert value.draft.box is None
    panel._sam_select_control(event(value="256"))
    assert value.selected is None


def test_native_select_change_feedback_preserves_selection_and_does_not_redraw_forever(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(0.2, 0.4, 1)
    class Select(Element):
        def get_attribute(self, name, default):
            return self.attributes.get(name, default)

        def set_inner_rml(self, text):
            self.attributes["value"] = ""
            panel._sam_select_control(event(value=""))

        def set_attribute(self, name, text):
            super().set_attribute(name, text)
            if name == "value":
                panel._sam_select_control(event(value=text))

    select = Select()
    doc = _DocumentStub({"sam-selection": select})
    panel._dirty = False
    panel._update_sam_selection(doc)
    assert value.selected == 0
    assert not panel._dirty
    panel._update_sam_selection(doc)
    assert not panel._dirty
    panel._sam_apply_target = object()
    panel._update_sam_selection(doc)
    assert select.has_attribute("disabled")
    panel._sam_apply_target = None
    panel._update_sam_selection(doc)
    assert not select.has_attribute("disabled")
    select.set_attribute("value", "")
    assert value.selected is None and panel._dirty


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("rotation", range(4))
def test_settled_chrome_reflow_moves_image_proposal_and_markers_together(panel_modules, monkeypatch, scale, rotation):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._image_info_cache["frame.png"] = (400, 200, 3)
    panel._rotation_quadrants = rotation
    panel._mask_editor = value = editor(module)
    value.point(0.5, 0.5, 1)
    value.request, value.phase, value.overlays = 9, "ready", [object()]
    panel._sam_texture_key, panel._sam_texture_source = (9, 0), "existing-texture"
    viewport, image, overlay, markers = Element(), Element(), Element(), Element()
    panel._doc = doc = _DocumentStub({"image-viewport": viewport, "main-image-a": image,
                                      "sam-overlay": overlay, "sam-markers": markers})
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: scale)
    panel._service_metadata = panel._service_sam = lambda: None
    panel._update_filmstrip = lambda *args: None
    panel._preview_state, panel._dirty = "ready", False
    panel._sam_controls_state = (0, False)
    panel._layout_pending = True
    viewport.client_height = 400  # Help/sidebar layout has now settled.
    assert panel.on_update(doc)
    for property in ("left", "top", "width", "height", "transform"):
        assert image.properties[property] == overlay.properties[property]
    layout = panel._get_image_layout(panel._image_paths[0])
    x, y = module.source_to_view(layout, rotation, 0.5 * 399, 0.5 * 199)
    assert f"left: {x:.3f}dp; top: {y:.3f}dp;" in markers.inner_rml


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
    panel._on_img_mousedown(mouse(0.25, 0.40))
    panel._on_img_mousemove(mouse(0.6, 0.55))
    panel._on_img_mousemove(mouse(0.65, 0.6))
    assert len(value.edits) == 1, "a drag must create only one undo entry"
    panel._on_img_mouseup(mouse(0.65, 0.6))
    assert value.draft.points[0][:2] == pytest.approx((0.65, 0.6))
    assert len(value.edits) == 2
    panel._on_img_mousedown(mouse(0.65, 0.6, button=1))
    assert not value.draft.points
    undo(value)
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


def test_prompt_edits_use_shared_history_and_expired_drafts_are_not_revived(panel_modules, monkeypatch):
    module, _ = panel_modules
    entries = []
    monkeypatch.setattr(module.lf.undo, "push", lambda name, undo, redo, **kw: entries.append((undo, redo)))
    panel = module.ImagePreviewPanel()
    value = panel._mask_editor
    value.bind("one.png", "r1")
    value.point(0.2, 0.4, 1)
    value.move_selected(0.6, 0.7)
    assert len(entries) == 2
    value.bind("two.png", "r2")
    value.point(0.5, 0.5, 0)
    module.RuntimeState.scene_generation.value += 1  # Ordinary scene changes keep edits replayable.
    entries[1][0]()
    assert value.drafts["one.png"].points == [(0.2, 0.4, 1)]
    assert value.draft.points == [(0.5, 0.5, 0)]
    entries[1][1]()
    assert value.drafts["one.png"].points == [(0.6, 0.7, 1)]
    value.bind("one.png", "replacement")
    entries[1][0]()
    assert not value.draft.points


def test_reopening_preview_does_not_replay_old_edits_into_a_new_draft(panel_modules, monkeypatch):
    module, _ = panel_modules
    entries = []
    monkeypatch.setattr(module.lf.undo, "push", lambda name, undo, redo, **kw: entries.append((undo, redo)))
    panel = module.ImagePreviewPanel()
    panel._mask_editor.bind("frame.png", "revision-a")
    panel._mask_editor.point(0.2, 0.4, 1)
    panel._mask_editor.point(0.8, 0.6, 0)
    panel.open([module.Path("frame.png")], [], 0)
    panel._mask_editor.bind("frame.png", "revision-a")
    entries[-1][0]()
    assert not panel._mask_editor.draft.points
    panel._metadata.close()


def test_mask_keymap_respects_rebinding_unbinding_modifiers_and_capture(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(0.4, 0.5, 1)
    keymap = module.lf.keymap
    monkeypatch.setattr(keymap, "get_action_for_key", lambda scope, key, mods:
                        keymap.Action.IMAGE_PREVIEW_DELETE if (scope, key, mods) == ("image_preview", 66, 2)
                        else keymap.Action.NONE)
    panel._on_keydown(event(key_identifier=module.KI_DELETE))
    panel._on_keydown(event(key_identifier=13))
    assert len(value.draft.points) == 1
    monkeypatch.setattr(keymap, "is_capturing", lambda: True)
    panel._on_keydown(event(key_identifier=13, ctrl_key=True))
    assert len(value.draft.points) == 1
    monkeypatch.setattr(keymap, "is_capturing", lambda: False)
    panel._on_keydown(event(Element("textarea"), key_identifier=13, ctrl_key=True))
    assert len(value.draft.points) == 1
    panel._on_keydown(event(key_identifier=13, ctrl_key=True))
    assert not value.draft.points


def test_candidate_texture_is_reused_and_released_before_its_owner(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._image_info_cache["frame.png"] = (400, 200, 3)
    panel._preview_state = "ready"
    panel._mask_editor = value = editor(module)
    value.overlays, value.scores, value.phase, value.request = [object(), object()], [0.8, 0.9], "ready", 9
    overlay = Element()
    doc = panel._doc = _DocumentStub({"sam-overlay": overlay})
    calls = []

    class Texture:
        def __init__(self, tensor):
            calls.append(("create", tensor))

        def update(self, tensor):
            calls.append(("update", tensor))

        def rml_src(self, width, height):
            assert (width, height) == (400, 200)
            return "dynamic-texture-source"

        def destroy(self):
            calls.append(("destroy", overlay.properties["decorator"]))

    monkeypatch.setattr(module.lf.ui, "DynamicTexture", Texture, raising=False)
    monkeypatch.setattr(module.lf.ui, "release_rml_texture", lambda source: calls.append(("release", source)), raising=False)
    monkeypatch.setattr(panel, "_current_camera_node", lambda **kw: None)
    panel._update_sam(doc)
    panel._update_sam(doc)
    assert calls == [("create", value.overlays[0])]
    panel._sam_candidate(1)
    panel._update_sam(doc)
    assert calls[1] == ("release", "dynamic-texture-source")
    assert calls[2] == ("update", value.overlays[1])
    panel._sam_cancel()
    assert calls[-2:] == [("release", "dynamic-texture-source"), ("destroy", "none")]


def test_failed_shared_history_push_rolls_back_prompt_edit(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor.bind("frame.png", "revision-a")
    def reject(*args, **kwargs):
        raise RuntimeError("history unavailable")
    monkeypatch.setattr(module.lf.undo, "push", reject)
    with pytest.raises(RuntimeError, match="history unavailable"):
        panel._mask_editor.point(0.4, 0.5, 1)
    assert not panel._mask_editor.draft.points


def test_mouse_removal_respects_rebound_double_click_and_modifiers(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(0.4, 0.5, 1)
    keymap = module.lf.keymap
    monkeypatch.setattr(panel, "_sam_hit", lambda ev: 0)
    monkeypatch.setattr(keymap, "get_action_for_mouse_button", lambda mode, button, mods, double:
                        keymap.Action.IMAGE_PREVIEW_REMOVE_POINT if (button, mods, double) == (2, 2, True)
                        else keymap.Action.NONE)
    panel._on_img_mousedown(event(button=1))
    panel._on_img_mousedown(event(button=2, ctrl_key=True))
    assert len(value.draft.points) == 1
    panel._on_img_mousedown(event(button=2, ctrl_key=True), double_click=True)
    assert not value.draft.points
