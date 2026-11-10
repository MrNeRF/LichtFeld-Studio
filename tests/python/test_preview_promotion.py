# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ready-image continuity and source isolation during asynchronous promotion."""
import pytest

from test_rmlui_image_sources import panel_modules, _DocumentStub
from test_preview_workflow import Element


@pytest.fixture
def inspection(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    path = module.Path("large.jpg")
    panel._image_paths = [path]
    panel._image_info_cache[str(path)] = (5187, 3361, 3)
    panel._path_revision_cache[str(path)] = "revision-1"
    doc = _DocumentStub({key: Element() for key in
                         ("main-image-a", "main-image-b", "mask-overlay", "no-image-text",
                          "image-viewport", "btn-picker", "preview-message")})
    panel._doc = doc
    states, redraws = {}, []
    monkeypatch.setattr(module.lf.ui, "preview_texture_status", lambda url, document: states.get(url, "pending"))
    monkeypatch.setattr(module.lf.ui, "request_redraw", redraws.append)
    monkeypatch.setattr(module.lf, "optimization_params", lambda: None, raising=False)
    panel._service_metadata = lambda: None
    panel._update_main_image(doc, True)
    states[panel._requested_preview] = "ready"
    panel._update_main_image(doc, True)
    assert panel._preview_state == "ready"
    return panel, path, doc, states, redraws


def test_zoom_keeps_ready_pixels_and_pick_until_atomic_promotion(inspection):
    panel, _, doc, states, redraws = inspection
    old = panel._ready_preview
    panel._actual_size()
    panel._update_main_image(doc, True)
    requested = panel._requested_preview
    assert requested != old and "&pmw=5187&" in requested
    assert panel._ready_preview == old and panel._preview_pending
    assert doc.get_element_by_id("main-image-a").properties["decorator"] == f"image({old})"
    assert doc.get_element_by_id("main-image-b").properties["decorator"] == f"image({requested})"
    assert "hidden" not in doc.get_element_by_id("main-image-b").attributes["class"]
    assert panel._preview_state == "ready" and panel._displayed_index == 0
    panel._update_toolbar(doc, True)
    assert not doc.get_element_by_id("btn-picker").has_attribute("disabled")
    assert doc.get_element_by_id("no-image-text").attributes["class"] == "hidden"
    panel._dirty = False
    count = len(redraws)
    panel.on_update(doc)
    assert len(redraws) > count, "promotion must keep polling while the displayed image is ready"
    states[requested] = "ready"
    panel.on_update(doc)
    assert panel._ready_preview == requested and not panel._preview_pending
    assert panel._active_image_layer == "main-image-b"
    assert doc.get_element_by_id("main-image-a").properties["decorator"] == "none"


def test_zoom_and_fit_reuse_full_size_without_demoting_or_polling(inspection):
    panel, _, doc, states, redraws = inspection
    panel._actual_size()
    panel._update_main_image(doc, True)
    full = panel._requested_preview
    states[full] = "ready"
    panel._update_main_image(doc, True)
    count = len(redraws)
    for zoom, fit in [(0.5, False), (0.1, False), (0.1, True), (2, False)]:
        panel._zoom, panel._fit_to_window = zoom, fit
        panel._update_main_image(doc, True)
        assert panel._requested_preview == full and panel._ready_preview == full
    assert len(redraws) == count


def test_failed_promotion_keeps_ready_image_and_stops_polling(inspection):
    panel, _, doc, states, redraws = inspection
    old = panel._ready_preview
    panel._actual_size()
    panel._update_main_image(doc, True)
    states[panel._requested_preview] = "error"
    panel._update_main_image(doc, True)
    panel._update_toolbar(doc, True)
    assert panel._preview_state == "ready" and panel._ready_preview == old
    assert panel._preview_load_failed and not panel._preview_pending
    assert doc.get_element_by_id("main-image-b").properties["decorator"] == "none"
    assert doc.get_element_by_id("preview-message").text
    count = len(redraws)
    panel._update_main_image(doc, True)
    assert len(redraws) == count


@pytest.mark.parametrize("change", ["path", "revision", "camera", "training", "refresh", "dimensions"])
def test_changed_source_never_reuses_old_ready_pixels_or_sample(inspection, change):
    panel, path, doc, _, _ = inspection
    old = panel._ready_preview
    panel._sampled_color = (1, 0, 0)
    if change == "path":
        other = path.with_name("other.jpg")
        panel._image_paths[0] = other
        panel._image_info_cache[str(other)] = (5187, 3361, 3)
    elif change == "revision":
        panel._path_revision_cache[str(path)] = "replacement"
    elif change == "camera":
        panel._camera_uids = [7]
    elif change == "training":
        panel._last_training_params = (2, 2048, True)
    elif change == "refresh":
        panel._refresh_revision += 1
    else:
        panel._image_info_cache[str(path)] = (4000, 3000, 3)
    panel._update_main_image(doc, True)
    assert not panel._ready_preview and panel._displayed_index == -1
    assert panel._preview_state == "loading" and panel._sampled_color is None
    assert all(old not in doc.get_element_by_id(layer).properties["decorator"]
               for layer in ("main-image-a", "main-image-b"))


def test_missing_required_status_api_is_not_treated_as_ready(inspection, panel_modules, monkeypatch):
    panel, _, doc, _, _ = inspection
    module, _ = panel_modules
    monkeypatch.delattr(module.lf.ui, "preview_texture_status")
    with pytest.raises(AttributeError):
        panel._update_main_image(doc, True)
