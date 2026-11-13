# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Draft/revision regressions independent of model availability or GPU timing."""

from test_rmlui_image_sources import panel_modules
import pytest
from types import SimpleNamespace
from test_preview_workflow import Element, event, browsing_panel
from test_rmlui_image_sources import _DocumentStub


def editor(module):
    value = module.FrameMaskEditor()
    value.edits = []
    value.on_edit = lambda key, before, after: value.edits.append((key, before, after))
    value.enabled = True
    value.bind("frame.png", "revision-a")
    return value


@pytest.fixture
def tracked_sequence(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._sam_sequence = {"run": "clip-a"}
    panel._image_paths, panel._mask_paths = [module.Path("frame.png")], [None]
    panel._path_revision_cache["frame.png"] = "revision-a"
    panel._path_stat_cache["frame.png"] = (True, 11, 22)
    yield module, panel
    panel._metadata.close()


@pytest.mark.parametrize("has_dataset", [False, True])
@pytest.mark.parametrize("invert,ignore,expected", [(False, False, False), (True, False, True), (True, True, False)])
def test_update_reads_optimization_once_and_preserves_mask_polarity(panel_modules, monkeypatch, has_dataset, invert, ignore, expected):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._preview_state = "ready"
    reads = []
    monkeypatch.setattr(module.lf, "MaskMode", SimpleNamespace(SEGMENT_AND_IGNORE=3), raising=False)
    params = SimpleNamespace(has_params=lambda: True, undistort=True, invert_masks=invert,
                             mask_mode=module.lf.MaskMode.SEGMENT_AND_IGNORE if ignore else object())
    monkeypatch.setattr(module.lf, "optimization_params", lambda: reads.append(True) or params)
    monkeypatch.setattr(module.lf, "dataset_params", lambda: SimpleNamespace(
        has_params=lambda: has_dataset, resize_factor=2, max_width=3840), raising=False)
    monkeypatch.setattr(panel, "_current_camera_node", lambda: None)
    for name in ("_service_sample", "_service_metadata", "_service_sam", "_update_filmstrip", "_refresh_ui"):
        monkeypatch.setattr(panel, name, lambda *args: None)
    panel.on_update(None)
    assert len(reads) == 1
    assert panel._mask_inverted is expected
    assert panel._last_training_params == ((2, 3840, True, expected) if has_dataset else (1, 0, False, expected))
    panel._metadata.close()



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


def test_repeated_ready_poll_preserves_local_validation_feedback(panel_modules):
    module, _ = panel_modules
    value = editor(module)
    value.request = 9
    snapshot = {"request": 9, "revision": "revision-a", "phase": "ready",
                "overlays": ["mask"], "scores": [.9]}
    assert value.accept(snapshot)
    value.error = "Enter a valid frame range."
    assert not value.accept(snapshot)
    assert value.error == "Enter a valid frame range."
    value.phase = "queued"
    assert value.accept(snapshot)
    assert not value.error


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
    panel._sam_apply_target = module.MaskApply(42, 41, "camera-uuid", value.key)
    # A transient preview also advances the UI generation. Only the native
    # document revision expires a mask prepared for a changed scene.
    module.RuntimeState.scene_generation.value += 1
    panel._path_revision_cache["frame.png"] = "changed" if changed_source else "revision-a"
    applied = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"request": 42, "phase": "expired" if changed_scene else "mask_ready"}, raising=False)
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
@pytest.mark.parametrize("sequence", [False, True])
def test_apply_keeps_selected_polarity_with_inverted_training_masks(panel_modules, monkeypatch, exclude, inverted, sequence):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(0.5, 0.5, 1)
    class Proposal:
        def __bool__(self):
            raise AssertionError("An image tensor is not a scalar boolean")
    value.request, value.phase, value.overlays = 5, "ready", [Proposal()]
    panel._path_stat_cache["frame.png"] = (True, 100, 200, 0, 0, 0)
    monkeypatch.setattr(panel, "_current_camera_node", lambda **kw: SimpleNamespace(uuid="camera-uuid", read_only=False))
    monkeypatch.setattr(module.lf, "optimization_params", lambda: SimpleNamespace(invert_masks=inverted), raising=False)
    submitted = []
    if sequence:
        panel._sam_sequence = {"run": "clip-a"}
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *args, **kw: submitted.append(args) or 6, raising=False)
    panel._sam_apply(exclude)
    assert submitted[0][-1] == exclude
    assert value.request == 5
    assert panel._sam_apply_target == module.MaskApply(6, 5, "camera-uuid", value.key)


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
    monkeypatch.setattr(panel, "_current_camera_node", lambda **kw: SimpleNamespace(uuid="camera-uuid", read_only=False))
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


@pytest.mark.parametrize("matched_objects", (-1, 1, 2))
def test_refinement_only_uses_a_prior_after_prompt_changes(panel_modules, matched_objects):
    module, _ = panel_modules
    value = editor(module)
    value.point(0.3, 0.5, 1)
    value.request = 9
    value.submitted_signature = value.signature
    value.accept({"request": 9, "revision": "revision-a", "phase": "ready",
                  "overlays": ["one.png"], "scores": [0.9], "matched_objects": matched_objects})
    assert not value.can_refine
    value.point(0.8, 0.8, 0)
    assert value.can_refine == (matched_objects <= 1)
    value.remove_selected()
    assert value.refinement is None
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
    generated = []
    monkeypatch.setattr(keymap, "get_action_for_key", lambda *args: keymap.Action.IMAGE_PREVIEW_GENERATE)
    monkeypatch.setattr(panel, "_sam_generate", lambda **kwargs: generated.append(kwargs))
    panel._set_sam_method("semantic")
    panel._on_keydown(event(Element("textarea"), key_identifier=13))
    assert not generated
    panel._on_keydown(event(key_identifier=13))
    assert generated == [{"semantic": True}]
    panel._set_sam_method("manual")
    panel._on_keydown(event(key_identifier=13))
    assert generated[-1] == {"semantic": False}


def test_candidate_texture_is_reused_and_released_before_its_owner(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._image_info_cache["frame.png"] = (400, 200, 3)
    panel._preview_state = "ready"
    panel._mask_editor = value = editor(module)
    value.overlays, value.scores, value.phase, value.request = [object(), object()], [0.8, 0.9], "ready", 9
    overlay = Element()
    legend = Element()
    panel._sam_frame_summaries = {0: {"mask_path": "previously-saved.png"}}
    doc = panel._doc = _DocumentStub({"sam-overlay": overlay, "mask-legend": legend})
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
    assert legend.attributes["title"] == "image_preview.sam_proposal_legend"
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


def test_tracking_submits_the_selected_seed_and_rejects_a_range_without_it(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._mask_editor = value = editor(module)
    value.point(.4, .6, 1)
    value.phase, value.request, value.candidate = "ready", 7, 2
    panel._current_index, panel._sam_track_first, panel._sam_track_last = 4, 2, 8
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: submitted.append((a, kw)) or 8, raising=False)
    panel._sam_track_first = 5
    panel._sam_track()
    assert not submitted
    assert value.error == "image_preview.sam_range_seed_required"
    panel._sam_track_first = 2
    panel._sam_track()
    args, kwargs = submitted[0]
    assert args[6:8] == (7, 2)
    assert kwargs == {"sequence": {"run": "clip-a", "first": 2, "last": 8, "seed": 4, "direction": "both"}}
    assert panel._sam_tracking_request == value.request == 8


def test_tracking_range_keeps_partial_edits_and_blocks_invalid_submission(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.phase, value.request = "ready", 7
    panel._image_paths = [module.Path(f"frame-{i}.png") for i in range(7)]
    panel._sam_sequence = {"run": "clip-a", "output": "frames", "frames": [{}] * 7}
    panel._sam_track_last = 6
    panel._sam_progress = (0, 0, 1)

    class Control(Element):
        option_updates = 0

        def get_attribute(self, name, default):
            return self.attributes.get(name, default)

        def get_inner_rml(self):
            return ""  # RmlUI selects store their options outside inner RML.

        def set_inner_rml(self, value):
            self.option_updates += 1
            super().set_inner_rml(value)

        def set_class(self, name, enabled):
            self.attributes[name] = enabled

    controls = {name: Control() for name in (
        "sam-controls", "sam-pages", "sam-page-frame", "sam-page-sequence", "sam-frame-controls",
        "sam-tracking", "sam-track-first", "sam-track-last", "sam-track-direction",
        "sam-approve", "sam-track", "sam-save", "sam-save-exclude", "sam-apply-heading", "sam-hint", "sam-generate-row", "sam-semantic-find-row")}
    panel._doc = doc = _DocumentStub(controls)
    panel._show_help = True
    controls["sam-hint"].set_text("image_preview.sam_tracking_hint")
    panel._update_sam_sequence(doc, False)
    direction = controls["sam-track-direction"]
    panel._sam_track_direction = "reverse"
    panel._update_sam_sequence(doc, False)
    assert direction.option_updates == 1
    assert direction.get_attribute("value", "") == "reverse"
    assert controls["sam-hint"].text == "image_preview.sam_tracking_hint"
    last = controls["sam-track-last"]
    assert last.get_attribute("value", "") == "7"
    assert controls["sam-apply-heading"].text.endswith("1 / 7")
    last.set_attribute("value", "")
    assert not panel._sam_track_options_changed(event(last))
    # Switching between frame editing and sequence review must not reset an
    # unfinished range edit, the current proposal or the shared draft.
    value.draft.points = [(.3, .5, 1)]
    controls["sam-controls"].scroll_top = 200
    panel._set_sam_page("sequence")
    panel._update_sam_page(doc)
    assert controls["sam-controls"].scroll_top == 0
    assert controls["sam-frame-controls"].attributes["hidden"]
    assert controls["sam-generate-row"].attributes["hidden"]
    assert not controls["sam-tracking"].attributes["hidden"]
    assert controls["sam-page-sequence"].attributes["aria-pressed"] == "true"
    panel._set_sam_page("frame")
    panel._update_sam_page(doc)
    assert not controls["sam-frame-controls"].attributes["hidden"]
    assert not controls["sam-generate-row"].attributes["hidden"]
    assert controls["sam-tracking"].attributes["hidden"]
    assert controls["sam-semantic-find-row"].attributes["hidden"]
    panel._sam_method = "semantic"
    panel._update_sam_page(doc)
    assert controls["sam-generate-row"].attributes["hidden"]
    assert not controls["sam-semantic-find-row"].attributes["hidden"]
    panel._set_sam_page("sequence")
    panel._update_sam_page(doc)
    assert controls["sam-semantic-find-row"].attributes["hidden"]
    panel._sam_method = "manual"
    panel._set_sam_page("frame")
    panel._update_sam_page(doc)
    assert value.request == 7 and value.draft.points == [(.3, .5, 1)]
    panel._update_sam_sequence(doc, False)
    assert last.get_attribute("value", "") == ""
    panel._path_stat_cache["frame.png"] = (True, 11, 22)
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: submitted.append(kw), raising=False)
    value.matched_objects = 0
    panel._update_sam_sequence(doc, False)
    assert controls["sam-track"].has_attribute("disabled")
    panel._sam_track()
    assert not submitted
    assert value.error == "image_preview.sam_semantic_no_matches"
    panel._sam_tracking_request = 9
    panel._update_sam_sequence(doc, True)
    assert not controls["sam-track"].has_attribute("disabled")
    panel._sam_tracking_request = 0
    value.matched_objects = -1
    panel._sam_track()
    assert not submitted
    assert value.error == "image_preview.sam_range_invalid"
    last.set_attribute("value", "6")
    assert panel._sam_track_options_changed(event(last))
    panel._update_sam_sequence(doc, False)
    assert panel._sam_track_last == 5
    assert last.get_attribute("value", "") == "6"
    assert direction.option_updates == 1
    translate = module.lf.ui.tr
    monkeypatch.setattr(module.lf.ui, "tr", lambda key: translate(key) + " translated")
    panel._update_sam_sequence(doc, False)
    assert direction.option_updates == 2
    assert direction.get_attribute("value", "") == "reverse"


def test_retaining_another_candidate_preserves_its_native_refinement_identity(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._mask_editor = value = editor(module)
    value.request, value.phase, value.overlays = 7, "ready", [object(), object()]
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: 8, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_sequence_frames", lambda *a: [{"revision": 8}], raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"request": 8, "phase": "mask_ready", "sequence_revision": 8}, raising=False)
    panel._sam_candidate(1)
    assert value.phase == "storing"
    module.RuntimeState.scene_generation.value += 1  # Transient previews do not expire retained frames.
    panel._service_sam()
    assert (value.phase, value.request, value.refinement) == ("ready", 7, (7, 1))
    assert panel._sam_apply_target is None


def test_failed_sequence_retain_keeps_prediction_and_blocks_export_edits(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._mask_editor = value = editor(module)
    panel._preview_state = "ready"
    value.request, value.phase, value.overlays = 7, "ready", [object(), object()]
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: submitted.append(a) or 8, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_sequence_frames", lambda *a: [], raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"request": 8, "phase": "error", "error": "encode failed"}, raising=False)
    panel._sam_candidate(1)
    assert value.busy and value.request == 7
    panel._sam_candidate(0)
    panel._sam_generate()
    panel._sam_edit("clear")
    assert value.candidate == 1 and len(submitted) == 1
    panel._service_sam()
    assert value.phase == "ready" and value.error
    assert value.request == 7 and value.refinement == (7, 1)
    panel._sam_retain_candidate()
    assert submitted[-1][6] == 7 and panel._sam_apply_target.proposal_request == 7


def test_saving_requires_confirmation_of_the_same_review_revision(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    panel._sam_sequence = {"run": "clip-a", "output": "frames<&>"}
    panel._sam_progress = (3, 3, 2)
    panel._path_stat_cache["frame.png"] = (True, 11, 22)
    dialogs, requests = [], []
    revision = {"sequence_revision": 10, "approved": 3}
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: revision, raising=False)
    monkeypatch.setattr(module.lf.ui, "form_dialog", lambda *a: dialogs.append(a), raising=False)
    monkeypatch.setattr(module.lf.ui, "tr", lambda key: "Approved: {approved}\n\n{output}\n\nReplace masks. Saving cannot be undone."
                        if key.endswith("sam_save_hint") else key)
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: requests.append(kw) or 20, raising=False)
    revision["phase"] = "tracking"
    panel._sam_save(True)
    assert not dialogs and not requests
    revision["phase"] = "ready"
    panel._sam_save(True)
    assert not requests
    revision["sequence_revision"] = 11
    assert dialogs[0][1] == "image_preview.sam_save_excluded"
    assert module.escape(str(module.Path("frames<&>") / "masks")) in dialogs[0][2]
    assert '<strong>Approved: 3</strong>' in dialogs[0][2]
    assert dialogs[0][2].count('class="modal-note"') == 3
    dialogs[0][4](dialogs[0][3][1]["label"], {})
    assert not requests
    panel._sam_save(True)
    dialogs[1][4](dialogs[1][3][1]["label"], {})
    dialogs[1][4](dialogs[1][3][1]["label"], {})
    assert requests == [{"exclude": True, "sequence": {"run": "clip-a", "publish": True}}]
    assert value.phase == "saving"


def test_range_approval_pages_summaries_and_uses_one_reversible_group(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._mask_editor = editor(module)
    panel._mask_editor.phase = "ready"
    panel._image_paths = [module.Path(f"frame-{i}.png") for i in range(257)]
    panel._sam_track_last = 256
    frames = [{"revision": i + 1, "approved": i == 0, "reliable": i != 1} for i in range(257)]
    frames[2]["revision"] = 0
    pages, dialogs, writes, history = [], [], [], []
    def summaries(run, first, count):
        pages.append((run, first, count))
        return frames[first:first + count]
    monkeypatch.setattr(module.lf.ui, "sam_preview_sequence_frames", summaries, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"sequence_revision": 300}, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_approve_frames", lambda run, updates: writes.append((run, updates)) or True, raising=False)
    monkeypatch.setattr(module.lf.ui, "form_dialog", lambda *a: dialogs.append(a), raising=False)
    monkeypatch.setattr(module.lf.undo, "push", lambda *a, **kw: history.append((a, kw)))
    monkeypatch.setattr(module.lf.ui, "tr", lambda key: "{first}-{last}:{count}:{uncertain}:{missing}" if key.endswith("sam_approve_range_hint") else key)
    panel._sam_approve_range()
    assert pages == [("clip-a", 0, 256), ("clip-a", 256, 1)]
    assert not writes and not history
    assert "1-257:255:1:1" in dialogs[0][2]
    dialogs[0][4](dialogs[0][3][1]["label"], {})
    dialogs[0][4](dialogs[0][3][1]["label"], {})
    expected = [(i, i + 1, True) for i in range(1, 257) if i != 2]
    assert writes == [("clip-a", expected)]
    assert len(history) == 1
    panel._mask_editor.phase = "saving"
    with pytest.raises(RuntimeError, match="projects.gallery.error.busy"):
        history[0][0][1]()
    assert len(writes) == 1, "busy playback must refuse before mutating any approval"
    panel._mask_editor.phase = "ready"
    history[0][0][1]()
    assert writes[-1] == ("clip-a", [(i, i + 1, False) for i in range(1, 257) if i != 2])
    history[0][0][2]()
    assert writes[-1] == ("clip-a", expected)
    panel._sam_sequence = {"run": "clip-b"}
    history[0][0][1]()
    assert len(writes) == 3, "history from an old extraction cannot change the new run"


@pytest.mark.parametrize("changed", ["revision", "run", "busy", "missing", "cancel"])
def test_range_approval_rejects_missing_or_superseded_review(tracked_sequence, monkeypatch, changed):
    module, panel = tracked_sequence
    panel._mask_editor = editor(module)
    panel._mask_editor.phase = "ready"
    panel._image_paths = [module.Path("frame.png")]
    state = {"sequence_revision": 3}
    dialogs, writes = [], []
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: state, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_sequence_frames", lambda *a: [
        {"revision": 0 if changed == "missing" else 3, "approved": False, "reliable": True}], raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_approve_frames", lambda *a: writes.append(a) or True, raising=False)
    monkeypatch.setattr(module.lf.ui, "form_dialog", lambda *a: dialogs.append(a), raising=False)
    panel._sam_approve_range()
    if changed == "missing":
        assert not dialogs
        assert panel._mask_editor.error == "image_preview.sam_range_missing"
    else:
        if changed == "revision":
            state["sequence_revision"] = 4
        elif changed == "run":
            panel._sam_sequence = {"run": "clip-b"}
        elif changed == "busy":
            panel._mask_editor.phase = "tracking"
        dialogs[0][4](dialogs[0][3][0 if changed == "cancel" else 1]["label"], {})
    assert not writes


def test_rejected_history_rolls_back_group_approval(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    writes = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_approve_frames", lambda *a: writes.append(a) or True, raising=False)
    def reject(*a, **kw):
        raise RuntimeError("history unavailable")
    monkeypatch.setattr(module.lf.undo, "push", reject)
    previous = [(0, 7, False), (1, 9, False)]
    with pytest.raises(RuntimeError, match="history unavailable"):
        panel._sam_commit_approvals("clip-a", previous, True, "approve")
    assert writes == [("clip-a", [(0, 7, True), (1, 9, True)]), ("clip-a", previous)]


@pytest.mark.parametrize("refusal", ["busy", "native_busy"])
def test_approval_history_refusal_raises_and_can_be_retried(tracked_sequence, monkeypatch, refusal):
    module, panel = tracked_sequence
    accepted, history, writes, phase = [True], [], [], ["ready"]
    def approve(run, values):
        writes.append(values)
        return accepted[0]
    monkeypatch.setattr(module.lf.ui, "sam_preview_approve_frames", approve, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"phase": phase[0]}, raising=False)
    monkeypatch.setattr(module.lf.undo, "push", lambda label, undo, redo, **kw: history.append((undo, redo)))
    previous = [(0, 7, False)]
    panel._sam_commit_approvals("clip-a", previous, True, "approve")
    assert len(history) == 1
    panel._sam_worker_busy = refusal == "busy"
    phase[0] = "tracking" if refusal == "native_busy" else "ready"
    accepted[0] = False
    for callback in history[0]:
        with pytest.raises(RuntimeError):
            callback()
    panel._sam_worker_busy = False
    phase[0] = "ready"
    accepted[0] = True
    history[0][0]()
    assert writes[-1] == previous
    history[0][1]()
    assert writes[-1] == [(0, 7, True)]
    accepted[0] = False
    history[0][0]()  # A replaced proposal is reported, but cannot trap older history.
    assert panel._sam_sequence_error == "image_preview.sam_proposal_expired"
    panel._sam_sequence = {"run": "clip-b"}
    count = len(writes)
    history[0][0]()
    assert len(writes) == count  # Closed/replaced documents remain harmless.


def test_undo_of_another_frame_preserves_current_proposal(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._image_paths = [module.Path("a.png"), module.Path("b.png")]
    invalidated, history = [], []
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"phase": "ready"}, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_invalidate_frame", lambda *a: invalidated.append(a), raising=False)
    monkeypatch.setattr(module.lf.undo, "push", lambda label, undo, redo, **kw: history.append((undo, redo)))
    value = panel._mask_editor
    value.bind("a.png", "ra")
    value.point(.2, .2, 1)
    panel._current_index = 1
    value.bind("b.png", "rb")
    overlay = object()
    value.request, value.phase, value.overlays = 44, "ready", [overlay]
    for callback in history[0]:
        callback()
        assert invalidated[-1] == ("clip-a", 0)
        assert value.key == ("b.png", "rb")
        assert value.request == 44 and value.phase == "ready"
        assert value.overlays == [overlay]
        assert not panel._sam_inspect_pending


def test_source_revision_waits_for_native_work_then_revokes_old_proposal(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._image_paths, panel._mask_paths = [module.Path("a.png")], [None]
    value = panel._mask_editor
    value.enabled = True
    value.bind("a.png", "old")
    value.request, value.phase, value.overlays = 22, "ready", [object()]
    panel._path_revision_cache["a.png"] = "new"
    phase, invalidated, frames, updates = ["tracking"], [], [{"revision": 7, "approved": True}], []
    def invalidate(*args):
        assert phase[0] == "ready"
        invalidated.append(args)
        frames[0] = {"revision": 0, "approved": False}
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"phase": phase[0]}, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_invalidate_frame", invalidate, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_sequence_frames", lambda *a: frames, raising=False)
    monkeypatch.setattr(panel, "_request_async_update", lambda: updates.append(True))
    panel._path_revision_cache.pop("a.png")
    phase[0] = "ready"
    panel._service_sam()
    assert value.key == ("a.png", "old") and value.request == 22
    assert not invalidated  # Missing metadata is not a confirmed source change.
    panel._path_revision_cache["a.png"] = "new"
    phase[0] = "tracking"
    for _ in range(2):
        panel._service_sam()
        assert value.key == ("a.png", "old")
        assert not invalidated
        assert not value.overlays and not value.request
        assert panel._sam_sequence_error == "image_preview.sam_source_changed"
    assert updates
    phase[0] = "ready"
    panel._service_sam()
    assert value.key == ("a.png", "new")
    assert invalidated == [("clip-a", 0)]
    assert not panel._sam_frame_summaries[0]["approved"]
    assert not panel._sam_sequence_error
    panel._service_sam()
    assert len(invalidated) == 1


def test_thumbnail_overlays_use_only_visible_proposals_and_release_old_revisions(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel, film, doc, records = browsing_panel(module)
    panel._sam_sequence = {"run": "clip-a"}
    panel._mask_editor.enabled = True
    panel._sam_frame_summaries = {i: {"revision": 8, "reliable": True} for i in range(30)}
    for path in panel._image_paths[:30]:
        panel._image_info_cache[str(path)] = (1280, 534, 3)
    released = []
    monkeypatch.setattr(module.lf.ui, "release_rml_texture", released.append, raising=False)
    panel._update_filmstrip(doc, True)
    assert len(records[-1]) <= 20
    old = "preview://memory=sam:clip-a:0:8&thumb=256&w=1280&h=534&mode=mask"
    assert records[-1][0]["mask_decorator"] == f"image({old} contain)"
    assert len(panel._sam_thumbnail_sources) == len(records[-1])
    panel._sam_frame_summaries[0]["revision"] = 9
    panel._sam_sequence_revision = 2
    panel._update_filmstrip(doc, True)
    assert released == [old]
    assert ":0:9&" in records[-1][0]["mask_decorator"]
    previous_sources = set(panel._sam_thumbnail_sources)
    panel._sam_show()
    panel._update_filmstrip(doc, True)
    assert all(row["mask_decorator"] == "none" for row in records[-1])
    assert previous_sources <= set(released)
    assert not panel._sam_thumbnail_sources
    panel._metadata.close()


def test_scrolling_or_hiding_filmstrip_retires_unseen_mask_textures(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel, film, doc, records = browsing_panel(module)
    panel._sam_sequence = {"run": "clip-a"}
    panel._mask_editor.enabled = True
    panel._sam_frame_summaries = {i: {"revision": 8, "reliable": True} for i in range(30)}
    released = []
    monkeypatch.setattr(module.lf.ui, "release_rml_texture", released.append, raising=False)
    panel._update_filmstrip(doc, True)
    old = set(panel._sam_thumbnail_sources)
    panel._ensure_selection_visible = False
    panel._filmstrip_follow_selection = False
    film.scroll_top = 320000
    panel._update_filmstrip(doc, True)
    assert old <= set(released)
    assert all(row["mask_decorator"] == "none" for row in records[-1])
    panel._sam_thumbnail_sources = {"preview://owned-test-source"}
    panel._show_filmstrip = False
    panel._update_filmstrip(doc, True)
    assert released[-1] == "preview://owned-test-source"
    assert not panel._sam_thumbnail_sources
    panel._metadata.close()


def test_prompt_edits_invalidate_only_their_sequence_frame_after_history_accepts(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._image_paths = [module.Path("one.png"), module.Path("two.png")]
    panel._mask_editor.bind("two.png", "r1")
    invalidated = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_invalidate_frame", lambda *a: invalidated.append(a), raising=False)
    panel._mask_editor.point(.4, .6, 1)
    assert invalidated == [("clip-a", 1)]
    def reject(*a, **kw):
        raise RuntimeError("history unavailable")
    monkeypatch.setattr(module.lf.undo, "push", reject)
    with pytest.raises(RuntimeError, match="history unavailable"):
        panel._mask_editor.point(.8, .6, 0)
    assert invalidated == [("clip-a", 1)]
    assert panel._mask_editor.draft.points == [(.4, .6, 1)]


def test_prompt_edit_during_native_work_restores_draft_without_history(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._image_paths = [module.Path("frame.png")]
    panel._mask_editor.bind("frame.png", "r1")
    panel._sam_tracking_request = 42
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"phase": "saving"}, raising=False)
    history = []
    monkeypatch.setattr(module.lf.undo, "push", lambda *a, **kw: history.append(a))
    with pytest.raises(RuntimeError, match="busy"):
        panel._mask_editor.point(.4, .6, 1)
    assert panel._mask_editor.draft.points == []
    assert not history
    assert panel._sam_tracking_request == 42


def test_sequence_summary_pages_preserve_saved_masks_after_proposal_revocation(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._image_paths = [module.Path(f"{i}.png") for i in range(300)]
    panel._mask_paths = [None] * 300
    panel._filmstrip_range = (0, 300)
    calls = []
    def summaries(run, first, count):
        calls.append((first, count))
        return [{"revision": 0, "mask_path": f"masks/{i}.png"} for i in range(first, first + count)]
    monkeypatch.setattr(module.lf.ui, "sam_preview_sequence_frames", summaries, raising=False)
    panel._sam_sync_sequence({"sequence_revision": 2})
    assert calls == [(0, 256), (256, 44)]
    assert len(panel._sam_frame_summaries) == 300
    assert panel._mask_paths[-1] == module.Path("masks/299.png")


def test_failed_inspection_submit_preserves_tracking_handle(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._mask_editor = value = editor(module)
    panel._sam_tracking_request = 42
    def reject(*a, **kw):
        raise RuntimeError("admission failed")
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", reject, raising=False)
    panel._sam_inspect_current()
    assert panel._sam_tracking_request == 42
    assert value.error == "admission failed"
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"request": 42, "phase": "tracking"}, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_sequence_frames", lambda *a: [], raising=False)
    updates = []
    monkeypatch.setattr(panel, "_request_async_update", lambda: updates.append(True))
    panel._service_sam()
    assert updates  # A rejected replacement must not stop polling the live job.
    assert panel._sam_tracking_request == 42
    assert panel._sam_worker_busy
    value.draft.points = [(.4, .6, 1)]
    panel._sam_edit("clear")
    panel._sam_pointer(None)
    assert value.draft.points == [(.4, .6, 1)]
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"phase": "idle"}, raising=False)
    panel._service_sam()
    assert panel._sam_tracking_request == 0
    assert not panel._sam_worker_busy


def test_approval_binding_failure_is_reported_without_creating_history(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    def reject(*a):
        raise RuntimeError("native unavailable")
    monkeypatch.setattr(module.lf.ui, "sam_preview_approve_frames", reject, raising=False)
    history = []
    monkeypatch.setattr(module.lf.undo, "push", lambda *a, **kw: history.append(a))
    panel._sam_commit_approvals("clip-a", [(0, 7, False)], True, "approve")
    assert panel._sam_sequence_error == "native unavailable"
    assert not history


def test_reopening_the_same_extraction_keeps_the_active_draft_and_approval_context(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    sequence = {"run": "clip-a", "output": "frames", "frames": [{"time": .25, "source_frame": 9}]}
    monkeypatch.setattr(module.lf.ui, "sam_preview_close", lambda: None, raising=False)
    ended = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_end_sequence", ended.append, raising=False)
    panel.open([module.Path("frame.png")], [None], 0, assisted=True, weights="video.lfw", sequence=sequence)
    value = panel._mask_editor
    value.bind("frame.png", "r1")
    monkeypatch.setattr(module.lf.ui, "sam_preview_invalidate_frame", lambda *a: None, raising=False)
    value.point(.3, .5, 1)
    panel.open([module.Path("frame.png")], [None], 0, assisted=True, weights="video.lfw", sequence=dict(sequence))
    assert panel._mask_editor is value
    assert value.draft.points == [(.3, .5, 1)]
    assert panel._sam_sequence == sequence
    assert not ended
    panel.open([module.Path("other.png")], [None], 0, assisted=True,
               sequence={"run": "clip-b", "output": "other", "frames": [{"time": 0, "source_frame": 0}]})
    assert ended == ["clip-a"]
    panel.on_unmount(SimpleNamespace(remove_data_model=lambda name: None))
    assert ended == ["clip-a", "clip-b"]
    assert panel._sam_sequence is None


def test_sequence_failure_survives_navigation_to_another_frame(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._mask_editor = value = editor(module)
    panel._preview_state = "ready"
    panel._sam_inspect_pending = True
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {
        "request": 20, "phase": "error", "error": "disk full", "completed": 1, "total": 2}, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_sequence_frames", lambda *a: [{"revision": 10}], raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: 21, raising=False)
    panel._service_sam()
    assert value.phase == "queued"
    assert panel._sam_sequence_error == "disk full · 1 / 2"


def test_frame_navigation_continues_after_approving_but_text_fields_keep_the_arrows(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    steps = []
    monkeypatch.setattr(panel, "_navigate", steps.append)
    checkbox = Element("input")
    checkbox.id = "sam-approve"
    panel._on_keydown(event(checkbox, key_identifier=module.KI_RIGHT))
    panel._on_keydown(event(checkbox, key_identifier=module.KI_LEFT))
    field = Element("input")
    field.id = "sam-track-first"
    panel._on_keydown(event(field, key_identifier=module.KI_RIGHT))
    assert steps == [1, -1]


def test_saving_after_releasing_sam_does_not_reopen_gpu_inspection(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._mask_editor = value = editor(module)
    value.request, value.phase = 20, "saving"
    closed, submitted = [], []
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {
        "request": 20, "revision": "revision-a", "phase": "saved", "approved": 1}, raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_sequence_frames", lambda *a: [{"revision": 10, "mask_path": "masks/frame.png.png"}], raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_close", lambda: closed.append(True), raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: submitted.append(a), raising=False)
    panel._service_sam()
    assert closed == [True]
    assert not submitted
    assert panel._mask_paths == [module.Path("masks/frame.png.png")]


@pytest.mark.parametrize("all_categories", (False, True))
def test_semantic_search_reuses_submit_with_owned_tokens_and_no_manual_geometry(panel_modules, monkeypatch, all_categories):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(.25, .5, 1)
    value.set_box((.1, .2), (.8, .9))
    original = value.signature
    value.semantic_categories = tuple(module.SEMANTIC_CATEGORIES) if all_categories else ("person",)
    value.semantic_text = "PERSON\nred car"
    value.request, value.phase, value.overlays = 42, "ready", [object()]
    panel._preview_state = "ready"
    panel._path_stat_cache["frame.png"] = (True, 100, 200, 0, 0, 0)
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: submitted.append((a, kw)) or 77, raising=False)
    panel._sam_generate(semantic=True)
    assert len(submitted) == 1
    args, kwargs = submitted[0]
    assert args[4:6] == ([], None)
    assert "weights" not in kwargs["semantic"]
    assert kwargs["semantic"]["confidence"] == .4
    descriptions = kwargs["semantic"]["descriptions"]
    assert len(descriptions) == len(value.semantic_categories) + 1  # PERSON shares the preset embedding.
    assert descriptions[0] == [101, 2711, 1012, 102]
    assert all(isinstance(row, list) for row in descriptions)
    assert value.signature == original
    assert value.request == 77 and value.phase == "queued" and not value.overlays
    panel._metadata.close()


def test_semantic_settings_preserve_manual_edits_and_sequence_approval(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(.5, .5, 1)
    original, edits = value.signature, list(value.edits)
    proposal = object()
    value.request, value.phase, value.overlays, value.scores = 42, "ready", [proposal], [.9]
    value.refinement, value.refinement_object_count = (42, 0), 2
    panel._sam_add_category(event(value="person"))
    panel._sam_add_category(event(value="person"))  # Programmatic select updates are idempotent.
    panel._sam_semantic_edit("semantic_text", "red car")
    panel._sam_set_semantic_confidence("65")
    assert value.semantic_confidence == 65
    assert panel._sam_semantic_payload()["confidence"] == .65
    panel._sam_set_semantic_confidence("nan")
    assert value.semantic_confidence == 65
    assert value.signature == original and value.edits == edits
    assert value.semantic_categories == ("person",)
    assert value.phase == "ready" and value.request == 42 and value.overlay is proposal
    assert value.scores == [.9] and value.refinement == (42, 0) and value.refinement_object_count == 2
    value.bind("next-frame.png", "revision-b")
    assert value.semantic_categories == ("person",)
    assert value.semantic_text == "red car"
    assert value.semantic_confidence == 65
    assert panel._sam_apply_reset("semantic_confidence", panel._sam_reset_default("semantic_confidence"))
    assert value.semantic_confidence == 40
    assert not panel._sam_apply_reset("unrelated", 1)
    value.phase = "predicting"
    assert not panel._sam_apply_reset("semantic_confidence", 65)
    assert value.semantic_confidence == 40
    value.phase = "idle"
    panel._sam_remove_category("person")
    assert value.semantic_categories == ()
    assert value.edits == edits
    panel._metadata.close()



@pytest.mark.parametrize("target", ["dataset", "unbound", "read_only"])
def test_semantic_proposal_apply_availability_explains_missing_dataset_target(panel_modules, monkeypatch, target):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    panel._sam_method = "semantic"
    value.request, value.phase, value.overlays = 42, "ready", [object()]
    class Button(Element):
        def set_class(self, name, enabled):
            self.attributes["class"] = name if enabled else ""
    buttons = {name: Button("button") for name in ("sam-apply", "sam-exclude")}
    doc = _DocumentStub(buttons)
    node = None if target == "unbound" else SimpleNamespace(read_only=target == "read_only")
    monkeypatch.setattr(panel, "_current_camera_node", lambda **kwargs: node)
    panel._update_sam(doc)
    for button in buttons.values():
        assert button.has_attribute("disabled") == (target != "dataset")
        if target != "dataset":
            assert button.attributes["title"] == "image_preview.sam_apply_dataset"
    panel._sam_semantic_edit("semantic_text", "chair")
    panel._update_sam(doc)
    assert value.request == 42 and value.phase == "ready" and value.overlays
    assert buttons["sam-apply"].has_attribute("disabled") == (target != "dataset")
    panel._metadata.close()


def test_semantic_category_menu_tracks_repeated_additions_and_removals(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = editor(module)
    panel._sam_method = "semantic"
    panel._sam_semantic_ui_state = None
    category = Element("select")
    doc = _DocumentStub({"sam-semantic-category": category})
    panel._doc = doc
    records = []
    panel._handle = SimpleNamespace(update_record_list=lambda name, rows: records.append(rows),
                                    dirty=lambda name: None, dirty_all=lambda: None)
    for selected in (("person",), (), ("chair",), ("chair", "person"), ("person",), ()):
        current = panel._mask_editor.semantic_categories
        for key in current:
            if key not in selected:
                panel._sam_remove_category(key)
        for key in selected:
            panel._sam_add_category(event(value=key))
        panel._update_sam_semantic(doc)
        assert [row["value"] for row in records[-1]] == list(selected)
        for key in module.SEMANTIC_CATEGORIES:
            assert (f'value="{key}"' in category.inner_rml) == (key not in selected)
        assert category.attributes["value"] == ""
    panel._metadata.close()


@pytest.mark.parametrize("description", ("cat " * 254, "cat " * 130 + "\n" + "dog " * 130))
def test_semantic_validation_does_not_queue_or_drop_tracking(panel_modules, monkeypatch, description):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.semantic_text = description
    value.request, value.phase, value.overlays = 42, "ready", [object()]
    panel._preview_state = "ready"
    panel._path_stat_cache["frame.png"] = (True, 100, 200, 0, 0, 0)
    calls = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: calls.append(kw), raising=False)
    panel._sam_generate(semantic=True)
    assert not calls
    assert value.phase == "ready" and value.request == 42 and value.overlays
    assert value.error == "image_preview.sam_semantic_text_error"
    panel._sam_semantic_edit("semantic_text", "person")
    assert not value.error and value.phase == "ready" and value.request == 42
    def fail_submit(*args, **kwargs):
        raise RuntimeError("submission failed")
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", fail_submit)
    panel._sam_generate(semantic=True)
    assert value.error == "submission failed"
    assert value.phase == "ready" and value.request == 42 and value.overlays
    value.semantic_text = "person " * 76
    value.phase = "tracking"
    panel._sam_tracking_request = 27
    panel._sam_semantic_edit("semantic_text", "cat")
    panel._sam_generate(semantic=True)
    assert panel._sam_tracking_request == 27 and not calls
    assert value.semantic_text == "person " * 76
    panel._metadata.close()


def test_semantic_metadata_and_lexical_resource_retire_with_the_editor(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.semantic_categories = ("person",)
    value.request = 7
    assert value.accept({"request": 7, "revision": "revision-a", "phase": "ready",
                         "overlays": [object()], "scores": [0], "matched_objects": 0})
    assert value.matched_objects == 0
    panel._sam_tokenizer = object()
    closed = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_close", lambda: closed.append(True))
    panel._sam_cancel()
    assert closed == [True] and panel._sam_tokenizer is None
    assert value.matched_objects == -1 and value.request == 0
    assert value.semantic_categories == ("person",)
    assert not value.accept({"request": 7, "revision": "revision-a", "phase": "ready", "overlays": [object()]})
    panel._metadata.close()


def test_selection_methods_preserve_draft_and_proposal_and_do_not_edit_hidden_controls(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.point(.3, .5, 1)
    value.request, value.phase, value.overlays = 7, "ready", [object()]
    value.semantic_text = "person"
    original = (value.draft, value.request, value.overlays)
    panel._set_sam_method("semantic")
    panel._sam_edit("remove_selected")
    panel._sam_pointer(None)  # Hidden manual controls must not receive canvas input.
    assert value.draft.points == [(.3, .5, 1)]
    assert (value.draft, value.request, value.overlays) == original
    panel._set_sam_method("manual")
    assert value.semantic_text == "person"
    assert (value.draft, value.request, value.overlays) == original
    panel._metadata.close()


def test_new_review_resets_semantic_fields_in_the_bound_view(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor.semantic_text = "person"
    panel._mask_editor.semantic_confidence = 70
    panel._sam_semantic_ui_state = (("person",), 0, False)
    dirty = []
    panel._handle = SimpleNamespace(dirty=dirty.append, request_update=lambda: None)
    panel.open([module.Path("other.png")], [None], 0, assisted=True)
    assert not panel._mask_editor.semantic_text
    assert "semantic_text" in dirty
    assert "semantic_confidence" in dirty
    assert panel._mask_editor.semantic_confidence == 40
    panel._sam_semantic_edit("semantic_text", "person")
    assert panel._sam_semantic_payload()["confidence"] == .4
    assert panel._sam_semantic_ui_state is None
    panel._metadata.close()


def test_semantic_no_match_keeps_exportable_mask_without_uploading_an_empty_overlay(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.request = 9
    mask = object()
    value.accept({"request": 9, "revision": "revision-a", "phase": "ready",
                  "overlays": [mask], "scores": [0], "matched_objects": 0})
    panel._preview_state = "ready"
    def unexpected_upload(*args, **kwargs):
        raise AssertionError("An empty semantic result has no visible selection")
    monkeypatch.setattr(module.lf.ui, "DynamicTexture", unexpected_upload, raising=False)
    released = []
    monkeypatch.setattr(module.lf.ui, "release_rml_texture", released.append, raising=False)
    panel._sam_texture = object()
    panel._sam_texture_source = "previous-selection"
    overlay = Element()
    panel._update_sam_geometry(_DocumentStub({"sam-overlay": overlay}))
    assert "hidden" in overlay.attributes["class"]
    assert released == ["previous-selection"]
    assert value.overlay is mask
    panel._metadata.close()


def test_empty_semantic_selection_cannot_keep_all_pixels_hidden_but_can_exclude_none(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.phase, value.overlays, value.matched_objects = "ready", [object()], 0
    panel._path_stat_cache["frame.png"] = (True, 100, 200)
    monkeypatch.setattr(panel, "_current_camera_node", lambda **kw: SimpleNamespace(uuid="camera", read_only=False))
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: submitted.append((a, kw)) or 10, raising=False)
    panel._sam_apply(False)
    assert not submitted and value.phase == "ready"
    panel._sam_apply(True)
    assert len(submitted) == 1 and submitted[0][0][-1] is True
    panel._metadata.close()


def test_empty_semantic_refinement_uses_only_the_new_visual_controls(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.request = 9
    value.submitted_signature = value.signature
    value.accept({"request": 9, "revision": "revision-a", "phase": "ready",
                  "overlays": [object()], "scores": [0], "matched_objects": 0})
    value.point(.5, .5, 1)
    assert not value.can_refine
    panel._preview_state = "ready"
    panel._path_stat_cache["frame.png"] = (True, 100, 200)
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: submitted.append(a) or 10, raising=False)
    panel._sam_generate(True)
    assert len(submitted) == 1 and submitted[0][6:8] == (0, -1)
    assert submitted[0][4] == [(.5, .5, 1)]
    panel._metadata.close()


def test_sequence_with_an_approved_empty_selection_refuses_keep_before_confirmation(tracked_sequence, monkeypatch):
    module, panel = tracked_sequence
    panel._mask_editor = value = editor(module)
    panel._sam_progress = (1, 1, 1)
    panel._sam_sequence["output"] = "frames"
    dialogs, submitted = [], []
    monkeypatch.setattr(module.lf.ui, "sam_preview_poll", lambda: {"approved": 1, "empty_approved": 1, "phase": "ready"}, raising=False)
    monkeypatch.setattr(module.lf.ui, "form_dialog", lambda *a: dialogs.append(a), raising=False)
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: submitted.append(a), raising=False)
    panel._sam_save(False)
    assert not dialogs and not submitted
    assert panel._sam_sequence_error == "image_preview.sam_empty_keep"
    panel._sam_save(True)
    assert len(dialogs) == 1 and not submitted


@pytest.mark.parametrize("sequence", [False, True])
def test_read_only_camera_cannot_submit_mask_apply(tracked_sequence, monkeypatch, sequence):
    module, panel = tracked_sequence
    if not sequence:
        panel._sam_sequence = None
    panel._mask_editor = value = editor(module)
    value.phase, value.overlays = "ready", [object()]
    monkeypatch.setattr(panel, "_current_camera_node", lambda **kw: SimpleNamespace(uuid="external", read_only=True))
    submitted = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: submitted.append(a), raising=False)
    panel._sam_apply(False)
    assert not submitted
    assert panel._sam_apply_target is None
    assert value.phase == "ready"


def test_combined_semantic_point_refinement_does_not_queue_a_replacement(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._mask_editor = value = editor(module)
    value.request = 9
    value.submitted_signature = value.signature
    value.accept({"request": 9, "revision": "revision-a", "phase": "ready",
                  "overlays": [object()], "scores": [.9], "matched_objects": 2})
    value.point(.7, .6, 1)
    panel._preview_state = "ready"
    panel._path_stat_cache["frame.png"] = (True, 100, 200, 0, 0, 0)
    calls = []
    monkeypatch.setattr(module.lf.ui, "sam_preview_submit", lambda *a, **kw: calls.append(a), raising=False)
    panel._sam_generate(True)
    assert not calls
    assert value.error == "image_preview.sam_semantic_refine_combined"
    assert value.draft.points == [(.7, .6, 1)]
    panel._metadata.close()
