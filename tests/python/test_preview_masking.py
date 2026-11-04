# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Draft/revision regressions independent of model availability or GPU timing."""

from test_rmlui_image_sources import panel_modules
import pytest
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
