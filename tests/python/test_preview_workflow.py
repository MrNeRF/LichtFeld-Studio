# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""User-facing preview regressions: browsing, input, readiness and explicit color changes."""

import os
from concurrent.futures import Future
from threading import get_ident
from types import SimpleNamespace

import pytest

from test_rmlui_image_sources import panel_modules, _DocumentStub, _ElementStub
from test_preview_file_revision import settle_metadata


def test_extracted_review_hides_extractor_only_after_success_and_can_retry(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    monkeypatch.setattr(module, "_instance", panel)
    changes = []
    monkeypatch.setattr(module.lf.ui, "set_panel_enabled", lambda *args: changes.append(args))
    with monkeypatch.context() as patch:
        def fail():
            raise RuntimeError("preview initialization failed")
        patch.setattr(panel, "_reset_preview", fail)
        with pytest.raises(RuntimeError, match="preview initialization failed"):
            module.open_extracted_frame_preview(["frame.png"])
    assert ("native.video_extractor", False) not in changes
    module.open_extracted_frame_preview(["frame.png"])
    assert panel._image_paths == [module.Path("frame.png")]
    assert panel._mask_editor.enabled
    assert changes[-1] == ("native.video_extractor", False)
    panel._metadata.close()


def test_pending_or_empty_review_keeps_extractor_available(panel_modules, monkeypatch):
    module, _ = panel_modules
    changes = []
    monkeypatch.setattr(module.lf.ui, "set_panel_enabled", lambda *args: changes.append(args))
    empty = module.ImagePreviewPanel()
    empty.open([], [], 0, assisted=True)
    assert not changes
    empty._metadata.close()
    monkeypatch.setattr(module, "_instance", None)
    module.open_extracted_frame_preview(["frame.png"])
    assert changes == [("lfs.image_preview", True)]
    panel = module.ImagePreviewPanel()
    assert module._pending_open is None
    assert panel._image_paths == [module.Path("frame.png")]
    assert changes[-1] == ("native.video_extractor", False)
    panel._metadata.close()


class Element(_ElementStub):
    def __init__(self, tag="div"):
        super().__init__()
        self.tag_name = tag
        self.scroll_top = 0.0
        self.scroll_height = 640000.0
        self.absolute_left = self.absolute_top = 0

    def has_attribute(self, key):
        return key in self.attributes

    def remove_attribute(self, key):
        self.attributes.pop(key, None)

    def parent(self):
        return None


def event(target=None, **values):
    return SimpleNamespace(target=lambda: target, current_target=lambda: target,
                           get_parameter=lambda key, default: str(values.get(key, default)),
                           get_bool_parameter=lambda key, default: bool(values.get(key, default)),
                           stop_propagation=lambda: None)


def browsing_panel(module, count=10000):
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path(f"frame_{i}.png") for i in range(count)]
    records = []
    panel._handle = SimpleNamespace(update_record_list=lambda _, rows: records.append(rows),
                                    dirty=lambda _: None, dirty_all=lambda: None)
    film = Element()
    film.client_height = 640
    doc = _DocumentStub({"filmstrip": film})
    panel._doc = doc
    return panel, film, doc, records


def test_scroll_renders_the_viewport_without_changing_selection(panel_modules):
    module, _ = panel_modules
    panel, film, doc, records = browsing_panel(module)
    panel._update_filmstrip(doc, True)
    film.scroll_top = 320000
    panel._on_filmstrip_scroll(event(film))
    assert records[-1][0]["index"] <= 5000 <= records[-1][-1]["index"]
    assert len(records[-1]) <= 20
    assert panel._current_index == 0
    panel._update_filmstrip(doc, True)
    assert film.scroll_top == 320000
    assert len(records) == 2, "unchanged frames must not replace the record list"


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_navigation_scrolls_to_selection_in_physical_pixels(panel_modules, monkeypatch, scale):
    module, _ = panel_modules
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: scale)
    panel, film, doc, records = browsing_panel(module)
    panel._current_index = 9000
    panel._update_filmstrip(doc, True)
    panel._update_filmstrip(doc, True)  # after the new virtual extent is laid out
    assert 9000 in [row["index"] for row in records[-1]]
    assert film.scroll_top <= 9000 * 64 * scale < film.scroll_top + film.client_height


def test_remount_populates_the_new_filmstrip_model_without_navigation(panel_modules):
    module, _ = panel_modules
    panel, film, doc, records = browsing_panel(module)
    panel._update_filmstrip(doc, True)
    expected = records[-1]
    panel.on_unmount(doc)
    new_records = []
    handle = SimpleNamespace(update_record_list=lambda _, rows: new_records.append(rows), dirty=lambda _: None)
    model = SimpleNamespace(bind_func=lambda *args: None, bind_record_list=lambda *args: None, get_handle=lambda: handle)
    panel.on_bind_model(SimpleNamespace(create_data_model=lambda _: model))
    panel._update_filmstrip(doc, True)
    assert new_records == [expected]
    panel._metadata.close()


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_navigation_retries_clamped_scroll_after_layout(panel_modules, monkeypatch, scale):
    module, _ = panel_modules
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: scale)
    panel, _, doc, records = browsing_panel(module)

    class ClampedFilm(Element):
        ready = False
        top = 0.0
        attempts = 0

        @property
        def scroll_top(self):
            return self.top

        @scroll_top.setter
        def scroll_top(self, value):
            self.attempts += 1
            self.top = min(value, 640000 * scale - self.client_height) if self.ready else 0.0
            panel._on_filmstrip_scroll(event(self))

    film = ClampedFilm.__new__(ClampedFilm)
    film.client_height = 640
    film.attributes = {}
    doc._elements["filmstrip"] = film
    panel._current_index = 9000
    panel._update_filmstrip(doc, True)
    assert film.attempts == 0, "publish the extent before requesting its scroll"
    assert 9000 in [row["index"] for row in records[-1]]
    panel._on_filmstrip_scroll(event(film))  # layout can emit scroll before the new extent is ready
    assert panel._ensure_selection_visible and film.scroll_top == 0
    film.ready = True
    panel._update_filmstrip(doc, True)
    assert not panel._ensure_selection_visible
    assert film.scroll_top <= 9000 * 64 * scale < film.scroll_top + film.client_height


@pytest.mark.parametrize("input_kind", ["pointer", "wheel"])
def test_user_input_cancels_pending_navigation_scroll(panel_modules, input_kind):
    module, _ = panel_modules
    panel, film, doc, records = browsing_panel(module)
    panel._current_index = 9000
    panel._update_filmstrip(doc, True)
    assert panel._ensure_selection_visible
    if input_kind == "wheel":
        film.scroll_height = film.client_height  # extent has not been laid out yet
        panel._on_precise_scroll(event(film, wheel_delta_y=1))
    else:
        panel._on_filmstrip_interaction(event(film))
    panel._on_filmstrip_scroll(event(film))
    assert not panel._ensure_selection_visible
    assert film.scroll_top == 0
    assert records[-1][0]["index"] == 0


def test_navigation_remains_visible_when_localized_layout_shrinks_the_viewport(panel_modules):
    module, _ = panel_modules
    panel, film, doc, records = browsing_panel(module)
    panel._current_index = 9000
    panel._update_filmstrip(doc, True)
    panel._update_filmstrip(doc, True)
    film.client_height -= 120
    panel._update_filmstrip(doc, True)
    panel._update_filmstrip(doc, True)
    assert film.scroll_top <= 9000 * 64 and 9001 * 64 <= film.scroll_top + film.client_height
    assert not panel._ensure_selection_visible
    panel._on_filmstrip_interaction(event(film))
    film.scroll_top = 0
    film.client_height -= 100
    panel._on_filmstrip_scroll(event(film))
    assert film.scroll_top == 0, "layout must not override the user's scroll position"
    assert not panel._ensure_selection_visible


def test_disabled_toolbar_action_keeps_hover_but_cannot_activate(panel_modules):
    module, _ = panel_modules
    button = Element()
    actions = []
    button.set_attribute("disabled", "")
    module.ImagePreviewPanel._activate_if_enabled(event(button), lambda _: actions.append(1))
    assert actions == []
    button.remove_attribute("disabled")
    module.ImagePreviewPanel._activate_if_enabled(event(button), lambda _: actions.append(1))
    assert actions == [1]


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0])
def test_pointer_zoom_preserves_source_pixel(panel_modules, monkeypatch, rotation, ratio):
    module, _ = panel_modules
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: ratio)
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._rotation_quadrants = rotation
    panel._get_image_info = lambda _: (1200, 800, 3)
    viewport = Element()
    panel._doc = _DocumentStub({"image-viewport": viewport})
    before = panel._get_image_layout(panel._image_paths[0])
    expected = module.view_to_source(before, rotation, 260 / ratio, 240 / ratio)
    panel._zoom_at_pointer(1.15, event(mouse_x=260, mouse_y=240))
    after = panel._get_image_layout(panel._image_paths[0])
    actual = module.view_to_source(after, rotation, 260 / ratio, 240 / ratio)
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("tag", ["input", "textarea", "select"])
def test_shortcuts_do_not_steal_keys_from_fields(panel_modules, tag):
    module, _ = panel_modules
    panel, _, _, _ = browsing_panel(module, 3)
    panel._on_keydown(event(Element(tag), key_identifier=module.KI_RIGHT))
    assert panel._current_index == 0


def test_modified_shortcut_is_left_to_the_application(panel_modules):
    module, _ = panel_modules
    panel, _, _, _ = browsing_panel(module, 3)
    panel._on_keydown(event(key_identifier=module.KI_RIGHT, ctrl_key=True))
    assert panel._current_index == 0


def test_scroll_callback_handles_distinct_wrappers_for_the_same_dom_element(panel_modules):
    module, _ = panel_modules
    panel, film, doc, records = browsing_panel(module)
    panel._update_filmstrip(doc, True)

    class Wrapper:
        id = "filmstrip"

        def __getattr__(self, name):
            return getattr(film, name)

        def __setattr__(self, name, value):
            setattr(film, name, value)

    panel._on_precise_scroll(event(Wrapper(), wheel_delta_y=10000))
    assert records[-1][0]["index"] > 4000
    assert panel._current_index == 0


def test_refresh_retries_the_same_file_and_thumbnail_revision(panel_modules):
    module, _ = panel_modules
    panel, film, doc, records = browsing_panel(module, 20)
    panel._update_filmstrip(doc, True)
    before = records[-1][0]["decorator"]
    panel._refresh_sources()
    panel._update_filmstrip(doc, True)
    assert records[-1][0]["decorator"] != before
    assert "refresh=1" in records[-1][0]["decorator"]


def test_leaving_fit_requests_source_resolution_at_every_zoom(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._get_image_info = lambda _: (8000, 6000, 3)
    panel._fit_to_window = False
    doc = _DocumentStub({"image-viewport": Element()})
    for zoom in (0.1, 0.5, 1.0, 2.0):
        panel._zoom = zoom
        assert panel._get_main_preview_max_width(doc, module.Path("large.png")) == 8000


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0])
def test_actual_size_maps_one_source_pixel_to_one_screen_pixel(panel_modules, monkeypatch, ratio):
    module, _ = panel_modules
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: ratio)
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._get_image_info = lambda _: (640, 480, 3)
    panel._doc = _DocumentStub({"image-viewport": Element()})
    panel._actual_size()
    layout = panel._get_image_layout(panel._image_paths[0])
    assert layout["draw_width"] * ratio == pytest.approx(640)
    assert layout["draw_height"] * ratio == pytest.approx(480)
    assert panel._get_zoom_display() == "100%"


@pytest.mark.parametrize("initial_info", [(0, 0, 0), (640, 480, 3)])
def test_refresh_rereads_the_current_header_even_when_stat_is_unchanged(panel_modules, monkeypatch, tmp_path, initial_info):
    module, _ = panel_modules
    path = tmp_path / "locked-temporarily.png"
    path.write_bytes(b"fixture")
    panel = module.ImagePreviewPanel()
    reads = []
    def read_header(_):
        reads.append(1)
        return initial_info if len(reads) == 1 else (320, 240, 3)
    monkeypatch.setattr(module.lf.ui, "get_image_info", read_header, raising=False)
    panel.open([path], [], 0)
    settle_metadata(panel)
    assert panel._get_image_info(path) == initial_info
    panel._refresh_sources()
    settle_metadata(panel)
    assert panel._get_image_info(path) == (320, 240, 3)
    assert len(reads) == 2
    panel._metadata.close()


def test_open_has_no_filesystem_reads_and_metadata_runs_off_ui_thread(panel_modules, monkeypatch, tmp_path):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    path = tmp_path / "frame.png"
    path.write_bytes(b"fixture")
    ui_thread = get_ident()
    threads = []
    monkeypatch.setattr(module.lf.ui, "get_image_info",
                        lambda _: threads.append(get_ident()) or (320, 240, 3), raising=False)
    with monkeypatch.context() as patch:
        patch.setattr(os, "stat", lambda *_: (_ for _ in ()).throw(AssertionError("UI filesystem read")))
        panel.open([path] * 10000, [], 0)
        assert panel._get_image_info(path) == (0, 0, 0)
    settle_metadata(panel)
    assert threads and all(thread != ui_thread for thread in threads)
    assert panel._get_image_info(path) == (320, 240, 3)
    panel._metadata.close()


def test_saved_view_keeps_the_requested_image_on_first_open(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel.apply_chrome({"camera_index": 0, "zoom": 2.0, "fit_to_window": False, "show_filmstrip": False})
    panel.open([module.Path(f"frame_{i}.png") for i in range(8)], [], 5)
    assert panel._current_index == 5
    assert (panel._zoom, panel._fit_to_window, panel._show_filmstrip) == (2.0, False, False)

    panel.apply_chrome({"camera_index": 2})
    assert panel._current_index == 2
    panel._metadata.close()


def test_overlay_waits_for_current_image_texture(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    path, mask = module.Path("frame.png"), module.Path("mask.png")
    panel._image_paths, panel._mask_paths = [path], [mask]
    panel._image_info_cache[str(path)] = (640, 480, 3)
    panel._path_stat_cache[str(mask)] = (True, 1, 10, 1, 1, 1)
    panel._show_overlay = True
    doc = _DocumentStub({eid: Element() for eid in
                         ("main-image-a", "main-image-b", "mask-overlay", "no-image-text", "image-viewport")})
    panel._doc = doc
    queries = []
    monkeypatch.setattr(module.lf.ui, "preview_texture_status",
                        lambda source, document: queries.append((source, document)) or "pending", raising=False)
    panel._update_main_image(doc, True)
    assert panel._displayed_index == -1
    assert doc.get_element_by_id("mask-overlay").properties["decorator"] == "none"
    assert queries[-1][1] is doc
    monkeypatch.setattr(module.lf.ui, "preview_texture_status", lambda source, document: "ready")
    panel._update_main_image(doc, True)
    assert panel._displayed_index == 0
    assert "&w=640&h=480" in doc.get_element_by_id("main-image-a").properties["decorator"]
    assert "preview://" in doc.get_element_by_id("mask-overlay").properties["decorator"]
    assert "&mode=mask&" in doc.get_element_by_id("mask-overlay").properties["decorator"]
    assert "&mode=mask&" not in doc.get_element_by_id("main-image-a").properties["decorator"]
    panel._mask_editor.enabled = True
    panel._update_main_image(doc, True)
    assert doc.get_element_by_id("mask-overlay").properties["decorator"] == "none"
    panel._mask_editor.enabled = False
    panel._update_main_image(doc, True)
    assert "&mode=mask&" in doc.get_element_by_id("mask-overlay").properties["decorator"]
    monkeypatch.setattr(module.lf.ui, "preview_texture_status", lambda source, document: "error")
    panel._update_main_image(doc, True)
    assert panel._preview_state == "error" and panel._displayed_index == -1


def test_remount_restarts_readiness_timeout_for_the_new_document(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    path = module.Path("frame.png")
    panel._image_paths = [path]
    panel._image_info_cache[str(path)] = (640, 480, 3)
    doc = _DocumentStub({eid: Element() for eid in
                         ("main-image-a", "mask-overlay", "no-image-text", "image-viewport")})
    doc.add_event_listener = lambda *args: None
    panel._requested_preview = panel._make_preview_url(path, -1, preview_mw=panel._get_main_preview_max_width(doc, path))
    panel._requested_at = module.time.monotonic() - 100
    panel._preview_state = "ready"
    monkeypatch.setattr(module.lf.ui, "preview_texture_status", lambda source, document: "unknown", raising=False)
    panel.on_mount(doc)
    panel._update_main_image(doc, True)
    assert panel._preview_state == "loading"
    assert module.time.monotonic() - panel._requested_at < 5
    panel.on_unmount(doc)


@pytest.mark.parametrize("element_id", ["sam-controls", "metadata-info", "sam-status"])
def test_nested_sidebar_content_receives_precise_scroll(panel_modules, element_id):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    content = Element()
    content.id, content.client_height, content.scroll_height = element_id, 200, 300
    listeners = {}
    content.add_event_listener = lambda name, callback: listeners.setdefault(name, callback)
    doc = _DocumentStub({element_id: content})
    doc.add_event_listener = lambda *args: None
    panel.on_mount(doc)
    listeners["mousescroll"](event(content, wheel_delta_y=1))
    assert 0 < content.scroll_top <= 100
    listeners["mousescroll"](event(content, wheel_delta_y=10000))
    assert content.scroll_top == 100
    content.scroll_height = content.client_height
    stopped = []
    wheel = event(content, wheel_delta_y=1)
    wheel.stop_propagation = lambda: stopped.append(True)
    listeners["mousescroll"](wheel)
    assert not stopped, "content without overflow must let its parent sidebar scroll"
    panel.on_unmount(doc)


def test_sample_does_not_mutate_backgrounds_and_apply_is_targeted_and_undoable(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._displayed_index = 0
    panel._image_info_cache["frame.png"] = (640, 480, 3)
    panel._doc = _DocumentStub({"image-viewport": Element()})
    colors = {"background_color": (0.1, 0.2, 0.3), "exposure": 1.0}
    params = SimpleNamespace(bg_color=(0.8, 0.8, 0.8), has_params=lambda: True)
    history = []
    monkeypatch.setattr(module.lf.ui, "sample_image_color", lambda *_: (0.4, 0.5, 0.6), raising=False)
    class RenderSnapshot:
        def __init__(self):
            self.values = dict(colors)

        def get(self, key):
            return self.values[key]

        def set(self, key, value):
            self.values[key] = value
            colors.update(self.values)

    monkeypatch.setattr(module.lf, "get_render_settings", RenderSnapshot, raising=False)
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params, raising=False)
    monkeypatch.setattr(module.lf, "undo", SimpleNamespace(push=lambda name, undo, redo, **kw: history.append((undo, redo))), raising=False)
    panel._perform_color_pick(event(mouse_x=400, mouse_y=300))
    panel._sample_request[-1].result(timeout=5)
    panel._service_sample()
    assert panel._sampled_color == (0.4, 0.5, 0.6)
    assert colors["background_color"] == (0.1, 0.2, 0.3) and params.bg_color == (0.8, 0.8, 0.8)
    panel._apply_sampled_color("render")
    assert colors["background_color"] == (0.4, 0.5, 0.6) and params.bg_color == (0.8, 0.8, 0.8)
    colors["exposure"] = 2.0
    history[-1][0]()
    assert colors["background_color"] == (0.1, 0.2, 0.3)
    assert colors["exposure"] == 2.0
    history[-1][1]()
    assert colors["background_color"] == (0.4, 0.5, 0.6)
    assert colors["exposure"] == 2.0


def test_failed_history_registration_restores_the_previous_color(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._sampled_color = (0.4, 0.5, 0.6)
    colors = {"background_color": (0.1, 0.2, 0.3)}
    monkeypatch.setattr(module.lf, "get_render_settings", lambda: SimpleNamespace(
        get=lambda key: colors[key], set=lambda key, value: colors.__setitem__(key, value)), raising=False)
    def reject(*args, **kwargs):
        raise RuntimeError("History is unavailable")
    monkeypatch.setattr(module.lf, "undo", SimpleNamespace(push=reject), raising=False)
    panel._apply_sampled_color("render")
    assert colors["background_color"] == (0.1, 0.2, 0.3)
    assert panel._action_error == "History is unavailable"


@pytest.mark.parametrize("destination", ["render", "training"])
def test_color_history_survives_rename_and_undo_rename(panel_modules, monkeypatch, destination):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._sampled_color = (0.4, 0.5, 0.6)
    colors = {"background_color": (0.1, 0.2, 0.3), "exposure": 1.0}
    params = SimpleNamespace(bg_color=(0.7, 0.8, 0.9), has_params=lambda: True)
    monkeypatch.setattr(module.lf, "get_render_settings", lambda: SimpleNamespace(
        get=lambda key: colors[key], set=lambda key, value: colors.__setitem__(key, value)), raising=False)
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params, raising=False)
    history = []
    monkeypatch.setattr(module.lf, "undo", SimpleNamespace(
        push=lambda name, undo, redo, **kwargs: history.append((undo, redo))), raising=False)
    signal = module.RuntimeState.scene_generation
    generation = signal.value
    try:
        previous = colors["background_color"] if destination == "render" else params.bg_color
        panel._apply_sampled_color(destination)
        assert not panel._action_error
        signal.value = generation + 2  # scene mutations for rename and undo rename
        colors["exposure"] = 2.0
        undo, redo = history[-1]
        undo()
        assert (colors["background_color"] if destination == "render" else params.bg_color) == previous
        redo()
        assert (colors["background_color"] if destination == "render" else params.bg_color) == panel._sampled_color
        assert colors["exposure"] == 2.0
    finally:
        signal.value = generation


def test_late_color_sample_does_not_follow_navigation(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("other.png")]
    future = Future()
    future.set_result((0.4, 0.5, 0.6))
    panel._sample_request = ("previous.png", None, future)
    panel._dirty = False
    panel._service_sample()
    assert panel._sample_request is None
    assert panel._sampled_color is None
    assert panel._dirty, "discarding a late result must re-enable the picker"


@pytest.mark.parametrize("during_sample", [False, True])
def test_file_replacement_rejects_a_color_sample(panel_modules, tmp_path, during_sample):
    module, _ = panel_modules
    path = tmp_path / "frame.png"
    path.write_bytes(b"original")
    stat = path.stat()
    revision = (True, stat.st_mtime_ns, stat.st_size, stat.st_dev, stat.st_ino, stat.st_ctime_ns)
    sampled = []

    def sample(*_):
        sampled.append(get_ident())
        path.write_bytes(b"different image data")
        return (0.4, 0.5, 0.6)

    if not during_sample:
        path.write_bytes(b"different image data")
    metadata = module.PreviewMetadata()
    try:
        future = metadata.sample(path, revision, 0, 0, sample)
        with pytest.raises(module.SourceChangedError):
            future.result(timeout=5)
        assert bool(sampled) == during_sample
        assert all(thread != get_ident() for thread in sampled)
    finally:
        metadata.close()


@pytest.mark.parametrize("source_changed", [False, True])
def test_sampling_errors_are_translated_when_published_on_ui(panel_modules, monkeypatch, source_changed):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    future = Future()
    future.set_exception(module.SourceChangedError() if source_changed else RuntimeError("decode failed"))
    panel._sample_request = ("frame.png", None, future)
    ui_thread = get_ident()

    def translate(key):
        assert get_ident() == ui_thread
        return {"image_preview.source_changed": "Aggiorna l’immagine",
                "image_preview.color_sample_failed": "Prelievo non riuscito: {}"}[key]

    monkeypatch.setattr(module.lf.ui, "tr", translate)
    panel._service_sample()
    assert panel._sampled_color is None
    assert panel._sample_request is None
    assert panel._action_error == ("Aggiorna l’immagine" if source_changed else "Prelievo non riuscito: decode failed")


def test_texture_completion_updates_the_available_commands(panel_modules, monkeypatch):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._preview_state = "loading"
    panel._dirty = False
    panel._service_metadata = lambda: None
    panel._update_filmstrip = lambda *args: None
    def complete(*args):
        panel._preview_state = "ready"
    panel._update_main_image = complete
    refresh = []
    panel._refresh_ui = lambda *args: refresh.append(panel._preview_state)
    panel.on_update(_DocumentStub({}))
    assert refresh == ["ready"]


@pytest.mark.parametrize("height, expected", [(300, "38%"), (428, "54%")])
def test_fit_percentage_follows_the_final_viewport_after_chrome_reflows(panel_modules, monkeypatch, height, expected):
    module, _ = panel_modules
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: 1.5)
    panel = module.ImagePreviewPanel()
    panel._image_paths = [module.Path("frame.png")]
    panel._image_info_cache["frame.png"] = (1200, 800, 3)
    viewport, zoom = Element(), Element()
    viewport.client_width, viewport.client_height = 800, height
    doc = _DocumentStub({"image-viewport": viewport, "st-zoom": zoom})
    panel._doc = doc
    panel._preview_state = "ready"
    panel._dirty = False
    panel._layout_pending = True
    panel._service_metadata = lambda: None
    panel._update_filmstrip = lambda *args: None
    assert panel.on_update(doc)
    assert expected in zoom.text
