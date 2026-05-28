# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for RmlUI image source escaping in Python panels."""

from importlib import import_module
from pathlib import Path
from threading import Lock
from types import ModuleType, SimpleNamespace
from urllib.parse import quote
import sys

import pytest


_RML_PATH_SAFE_CHARS = "/:._-~"


def _install_lf_stub(monkeypatch):
    class _Panel:
        def on_mount(self, _doc):
            pass

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        Panel=_Panel,
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(CONTENT="CONTENT"),
        tr=lambda key: key,
        open_url=lambda _url: None,
        get_ui_scale=lambda: 1.0,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)


@pytest.fixture
def panel_modules(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    for module_name in [
        "lfs_plugins.image_preview_panel",
        "lfs_plugins.getting_started_panel",
        "lfs_plugins",
    ]:
        sys.modules.pop(module_name, None)

    _install_lf_stub(monkeypatch)

    image_preview = import_module("lfs_plugins.image_preview_panel")
    getting_started = import_module("lfs_plugins.getting_started_panel")
    return image_preview, getting_started


def test_preview_url_percent_encodes_image_paths(panel_modules, tmp_path):
    image_preview, _ = panel_modules
    panel = image_preview.ImagePreviewPanel()
    panel._last_training_params = (2, 3840, True)

    image_path = tmp_path / "Photogrammetry Sekal pipes" / "frame & sample(1).jpg"
    preview_url = panel._make_preview_url(image_path, cam_uid=9, thumb=256)

    expected_path = quote(str(image_path), safe=_RML_PATH_SAFE_CHARS)
    assert preview_url.endswith(f"&path={expected_path}")
    assert str(image_path) not in preview_url
    assert "%20" in preview_url
    assert "%26" in preview_url
    assert "%28" in preview_url
    assert "%29" in preview_url


class _ElementStub:
    def __init__(self):
        self.properties = {}
        self.attributes = {}
        self.client_width = 800
        self.client_height = 600
        self.offset_top = 0
        self.offset_height = 0

    def set_property(self, name, value):
        self.properties[name] = value

    def remove_property(self, name):
        self.properties.pop(name, None)

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def set_text(self, value):
        self.text = value

    def query_selector(self, selector):
        if selector == ".card-body":
            return getattr(self, "body", None)
        return None


class _DocumentStub:
    def __init__(self, elements):
        self._elements = elements

    def get_element_by_id(self, element_id):
        return self._elements.get(element_id)


class _HandleStub:
    def __init__(self):
        self.request_update_count = 0

    def request_update(self):
        self.request_update_count += 1


def test_image_preview_mask_decorator_escapes_paths(panel_modules, tmp_path):
    image_preview, _ = panel_modules
    panel = image_preview.ImagePreviewPanel()

    image_path = tmp_path / "images" / "scene image.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"img")
    mask_path = tmp_path / "masks" / "mask overlay 01.png"
    mask_path.parent.mkdir(parents=True)
    mask_path.write_bytes(b"mask")

    panel._image_paths = [image_path]
    panel._mask_paths = [mask_path]
    panel._camera_uids = [7]
    panel._show_overlay = True
    panel._prev_image_index = -1
    panel._doc = _DocumentStub({
        "main-image-a": _ElementStub(),
        "main-image-b": _ElementStub(),
        "mask-overlay": _ElementStub(),
        "no-image-text": _ElementStub(),
        "image-viewport": _ElementStub(),
    })
    panel._get_image_info = lambda _path: (400, 200, 4)

    panel._update_main_image(panel._doc, has_images=True)

    expected_mask = quote(str(mask_path), safe=_RML_PATH_SAFE_CHARS)
    mask_img = panel._doc.get_element_by_id("mask-overlay")
    assert mask_img.properties["decorator"] == f"image({expected_mask})"


def test_getting_started_panel_escapes_thumbnail_paths(panel_modules, tmp_path):
    _, getting_started = panel_modules
    panel = getting_started.GettingStartedPanel()
    thumb_path = tmp_path / "thumb cache" / "video & thumb(1).jpg"

    body = _ElementStub()
    card = _ElementStub()
    card.body = body

    panel._ready_lock = Lock()
    panel._ready_queue = [("intro", str(thumb_path))]
    panel._thumb_card_map = {"intro": "card-intro"}

    doc = _DocumentStub({"card-intro": card})
    panel.on_update(doc)

    expected_thumb = quote(str(thumb_path), safe=_RML_PATH_SAFE_CHARS)
    assert body.properties["decorator"] == f"image({expected_thumb})"


def test_getting_started_panel_uses_dirty_update_policy(panel_modules):
    _, getting_started = panel_modules
    assert getting_started.GettingStartedPanel.update_policy == "dirty"
    assert "update_interval_ms" not in getting_started.GettingStartedPanel.__dict__


def test_image_preview_uses_dirty_update_policy(panel_modules):
    image_preview, _ = panel_modules
    assert image_preview.ImagePreviewPanel.update_policy == "dirty"


def test_image_preview_dirty_request_schedules_update(panel_modules):
    image_preview, _ = panel_modules
    panel = image_preview.ImagePreviewPanel()
    panel._handle = _HandleStub()
    panel._dirty = False

    panel._mark_dirty()

    assert panel._dirty is True
    assert panel._handle.request_update_count == 1


def test_getting_started_panel_requests_update_when_thumbnail_ready(panel_modules):
    _, getting_started = panel_modules
    callbacks = []
    sys.modules["lichtfeld"].ui.schedule_on_ui_thread = callbacks.append

    panel = getting_started.GettingStartedPanel()
    panel._handle = _HandleStub()
    panel._ready_lock = Lock()
    panel._ready_queue = []
    panel._thumb_update_scheduled = False

    panel._on_thumb_ready("intro", "/tmp/intro.jpg")
    panel._on_thumb_ready("next", "/tmp/next.jpg")

    assert panel._ready_queue == [
        ("intro", "/tmp/intro.jpg"),
        ("next", "/tmp/next.jpg"),
    ]
    assert len(callbacks) == 1
    assert panel._handle.request_update_count == 0

    callbacks.pop()()

    assert panel._handle.request_update_count == 1
    assert panel._thumb_update_scheduled is False
