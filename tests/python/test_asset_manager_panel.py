# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for Asset Manager panel record formatting and selection."""

import asyncio
from importlib import import_module
from pathlib import Path, PureWindowsPath
from types import ModuleType, SimpleNamespace
from urllib.parse import quote
import json
import re
import sys
import time

import pytest


RML_PATH_SAFE_CHARS = "/:._-~"


def _install_lf_stub(monkeypatch):
    class _LogStub:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(("info", message))

        def warn(self, message):
            self.messages.append(("warn", message))

        def error(self, message):
            self.messages.append(("error", message))

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(FILL="FILL", CONTENT="CONTENT"),
        tr=lambda key: key,
    )
    lf_stub.log = _LogStub()
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)


@pytest.fixture
def asset_manager_panel_module(monkeypatch):
    folder_root = Path(__file__).parent.parent.parent
    source_python = folder_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.asset_manager_panel", None)
    sys.modules.pop("lfs_plugins", None)
    _install_lf_stub(monkeypatch)
    return import_module("lfs_plugins.asset_manager_panel")


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_fields.append("__all__")

    def request_update(self):
        self.dirty_fields.append("__update__")


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


class _ElementStub:
    def __init__(self, attrs=None, parent=None, tag_name="div"):
        self._attrs = attrs or {}
        self._parent = parent
        self.tag_name = tag_name
        self.listeners = {}
        self.scroll_height = 0.0
        self.client_height = 0.0
        self.scroll_top = 0.0

    def get_attribute(self, name, default=""):
        return self._attrs.get(name, default)

    def has_attribute(self, name):
        return name in self._attrs

    def parent(self):
        return self._parent

    def add_event_listener(self, event, callback):
        self.listeners[event] = callback


class _EventStub:
    def __init__(
        self,
        current_target=None,
        target=None,
        bool_params=None,
        params=None,
    ):
        self._current_target = current_target
        self._target = target or current_target
        self._bool_params = bool_params or {}
        self._params = params or {}
        self.stopped = False

    def current_target(self):
        return self._current_target

    def target(self):
        return self._target

    def get_bool_parameter(self, key, default=False):
        return self._bool_params.get(key, default)

    def get_parameter(self, key, default=""):
        return self._params.get(key, default)

    def stop_propagation(self):
        self.stopped = True


class _DocumentStub:
    def __init__(self, elements=None):
        self._elements = elements or {}
        self.listeners = {}

    def get_element_by_id(self, element_id):
        return self._elements.get(element_id)

    def add_event_listener(self, event, callback):
        self.listeners[event] = callback


def _make_asset():
    return {
        "id": "a1",
        "name": "bicycle",
        "type": "dataset",
        "role": "source_dataset",
        "absolute_path": "/tmp/bicycle",
        "path": "/tmp/bicycle",
        "file_size_bytes": 4206437268,
        "tags": ["outdoor", "benchmark"],
        "exists": True,
        "dataset_metadata": {
            "image_count": 194,
            "has_masks": False,
            "mask_count": 0,
            "sparse_model": True,
            "camera_count": 1,
            "database_present": False,
            "image_root": "images",
        },
        "geometry_metadata": {},
        "folder_id": "p1",
        "scene_id": "s1",
        "created_at": "2026-02-15T21:52:45.881056",
        "modified_at": "2026-04-28T14:48:57.606369",
        "is_favorite": False,
    }


def test_asset_manager_uses_dirty_update_policy(asset_manager_panel_module):
    assert asset_manager_panel_module.AssetManagerPanel.update_policy == "dirty"
    assert "update_interval_ms" not in asset_manager_panel_module.AssetManagerPanel.__dict__


def test_asset_manager_requests_update_from_reactive_store(asset_manager_panel_module, monkeypatch):
    module = asset_manager_panel_module
    signals = SimpleNamespace(
        scene_generation=_SignalStub(),
        language_generation=_SignalStub(),
    )
    monkeypatch.setattr(module, "RuntimeState", signals)

    panel = module.AssetManagerPanel()
    panel._handle = _HandleStub()

    panel._subscribe_reactive_state()
    signals.scene_generation.emit(1)
    signals.language_generation.emit(1)

    assert panel._handle.dirty_fields == ["__update__", "__update__"]

    panel._unsubscribe_reactive_state()
    signals.scene_generation.emit(2)

    assert panel._handle.dirty_fields == ["__update__", "__update__"]


def test_scan_status_transition_requests_model_update(asset_manager_panel_module, monkeypatch):
    panel = asset_manager_panel_module.AssetManagerPanel()
    requested = []
    monkeypatch.setattr(panel, "_request_model_update", lambda: requested.append(True))

    panel._set_scan_status("scanning")

    assert panel.get_scan_status_visible() is True
    assert panel.get_scan_status_label() == "asset_manager.status.scanning"
    assert panel.get_scan_status_class() == "scan-status-scanning"
    assert requested == [True]

    panel._set_scan_status("idle")
    assert panel.get_scan_status_class() == "scan-status-idle"
    assert panel.get_scan_status_visible() is True
    assert panel.get_scan_status_label() == "asset_manager.status.scanning"
    assert requested == [True, True]


def test_asset_rows_expose_scalar_tag_label(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    row = panel._format_asset_for_ui(_make_asset())

    assert row["tags_label"] == "outdoor, benchmark"
    assert "tags" not in row


def test_asset_rows_expose_thumbnail_decorator(asset_manager_panel_module, tmp_path):
    panel = asset_manager_panel_module.AssetManagerPanel()
    thumb_path = tmp_path / "asset preview.png"
    thumb_path.write_bytes(b"not a real png")
    asset = _make_asset()
    asset["thumbnail_path"] = str(thumb_path)

    row = panel._format_asset_for_ui(asset)

    assert row["thumbnail_decorator"] == (
        f"image({quote(thumb_path.as_posix(), safe=RML_PATH_SAFE_CHARS)})"
    )


def test_thumbnail_decorator_normalizes_windows_path_separators(
    asset_manager_panel_module, monkeypatch
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    windows_path = PureWindowsPath("C:/Users/paja/asset preview.png")

    class _ExistingWindowsPath:
        def __init__(self, raw_path):
            self.raw_path = raw_path

        def expanduser(self):
            return self

        def exists(self):
            return True

        def as_posix(self):
            return windows_path.as_posix()

        def __str__(self):
            return str(windows_path)

    monkeypatch.setattr(asset_manager_panel_module, "Path", _ExistingWindowsPath)

    decorator = panel._thumbnail_decorator({"thumbnail_path": str(windows_path)})

    assert decorator == "image(C:/Users/paja/asset%20preview.png)"
    assert "%5C" not in decorator


def test_asset_card_title_uses_asset_path_leaf(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    asset = _make_asset()
    asset["name"] = ""
    asset["absolute_path"] = "/data/tandt/truck/train"
    fields = panel._get_asset_display_fields(
        asset,
        folder_name="tandt",
        scene_name="truck",
    )

    assert fields["display_name"] == "train"
    assert fields["display_subtitle"] == "truck"


def test_asset_manager_rml_uses_text_interpolation_for_display_values():
    folder_root = Path(__file__).parent.parent.parent
    rml_path = (
        folder_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "asset_manager.rml"
    )
    rml = rml_path.read_text(encoding="utf-8")

    display_data_value_lines = [
        line.strip()
        for line in rml.splitlines()
        if ("<span" in line or "<div" in line) and "data-value=" in line
    ]

    assert display_data_value_lines == []
    assert "{{asset.display_name}}" in rml
    assert "{{selected_asset_name}}" in rml
    assert "{{selected_asset_dataset_image_count}}" in rml
    assert "data-if=\"selected_asset_has_sh_degree\"" in rml
    assert 'data-style-decorator="asset.thumbnail_decorator"' in rml


def test_asset_manager_card_thumbs_do_not_use_gradient_placeholders():
    folder_root = Path(__file__).parent.parent.parent
    rcss_path = (
        folder_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "asset_manager.rcss"
    )
    rcss = rcss_path.read_text(encoding="utf-8")

    assert "vertical-gradient" not in rcss


def test_dataset_thumbnail_uses_first_dataset_image(asset_manager_panel_module, tmp_path):
    dataset_dir = tmp_path / "dataset"
    image_dir = dataset_dir / "images"
    mask_dir = image_dir / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    first = image_dir / "0001.jpg"
    second = image_dir / "0002.jpg"
    ignored_mask = mask_dir / "0000.jpg"
    first.write_bytes(b"not a real jpeg")
    second.write_bytes(b"not a real jpeg")
    ignored_mask.write_bytes(b"not a real jpeg")

    thumbnails = asset_manager_panel_module.AssetThumbnails(tmp_path / "thumbs")
    thumb_path = asyncio.run(
        thumbnails.generate_dataset_preview(
            "dataset",
            "dataset_asset",
            dataset_dir,
            {"image_root": "images"},
        )
    )

    assert thumb_path is not None
    assert thumb_path.exists()
    assert thumb_path in {
        first,
        thumbnails.get_dataset_thumbnail_path("dataset_asset"),
    }


def test_dataset_thumbnail_cache_uses_gallery_aspect(
    asset_manager_panel_module,
    tmp_path,
):
    pytest.importorskip("PIL")
    from PIL import Image

    dataset_dir = tmp_path / "dataset"
    image_dir = dataset_dir / "images"
    image_dir.mkdir(parents=True)
    first = image_dir / "0001.png"
    Image.new("RGB", (320, 240), "#cc3333").save(first)

    thumbnails = asset_manager_panel_module.AssetThumbnails(tmp_path / "thumbs")
    thumb_path = asyncio.run(
        thumbnails.generate_dataset_preview(
            "dataset",
            "dataset_asset",
            dataset_dir,
            {"image_root": "images"},
        )
    )

    # generate_dataset_preview returns the source image path directly
    assert thumb_path == first
    with Image.open(thumb_path) as img:
        assert img.size == (320, 240)


def test_stale_managed_thumbnail_requests_refresh(
    asset_manager_panel_module,
    tmp_path,
):
    thumb_path = tmp_path / "dataset_asset.dataset.png"
    thumb_path.write_bytes(b"not a current thumbnail")

    class _Thumbnails:
        thumbnails_dir = tmp_path

        def get_dataset_thumbnail_path(self, asset_id):
            return tmp_path / f"{asset_id}.dataset.png"

        def get_thumbnail_path(self, asset_id):
            return tmp_path / f"{asset_id}.png"

        def thumbnail_matches_expected_size(self, path):
            return False

    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._asset_thumbnails = _Thumbnails()

    asset = _make_asset()
    asset["id"] = "dataset_asset"
    asset["thumbnail_path"] = str(thumb_path)

    assert panel._asset_needs_thumbnail_refresh(asset) is True


def test_rendered_thumbnails_cover_geometry_and_checkpoints(
    asset_manager_panel_module,
    tmp_path,
):
    calls = []

    def render_asset_preview(path, width, height):
        calls.append((path, width, height))
        return object()

    def save_image(path, image):
        Path(path).write_bytes(b"preview")

    asset_manager_panel_module.lf.render_asset_preview = render_asset_preview
    asset_manager_panel_module.lf.io = SimpleNamespace(save_image=save_image)

    thumbnails = asset_manager_panel_module.AssetThumbnails(tmp_path / "thumbs")
    for asset_type in ("checkpoint", "mesh", "ply_pcl"):
        asset_path = tmp_path / f"asset.{asset_type}"
        asset_path.write_bytes(b"asset")
        thumb_path = asyncio.run(
            thumbnails.generate_rendered_preview(
                asset_type,
                f"{asset_type}_asset",
                asset_path,
            )
        )

        assert re.fullmatch(
            rf"{asset_type}_asset\.render\.\d+\.png",
            thumb_path.name,
        )
        assert thumb_path.exists()

    assert [Path(call[0]).name for call in calls] == [
        "asset.checkpoint",
        "asset.mesh",
        "asset.ply_pcl",
    ]


def test_cleanup_orphans_preserves_timestamped_rendered_thumbnails(
    asset_manager_panel_module,
    tmp_path,
):
    thumbnails = asset_manager_panel_module.AssetThumbnails(tmp_path / "thumbs")
    thumbs_dir = thumbnails.thumbnails_dir

    keep_rendered = thumbs_dir / "alive.render.1700000000.png"
    keep_dataset = thumbs_dir / "alive.dataset.png"
    keep_placeholder = thumbs_dir / "alive.png"
    orphan_rendered = thumbs_dir / "gone.render.1700000001.png"
    orphan_dataset = thumbs_dir / "gone.dataset.png"
    for path in (keep_rendered, keep_dataset, keep_placeholder, orphan_rendered, orphan_dataset):
        path.write_bytes(b"thumb")

    removed = asyncio.run(thumbnails.cleanup_orphans({"alive"}))

    assert keep_rendered.exists()
    assert keep_dataset.exists()
    assert keep_placeholder.exists()
    assert not orphan_rendered.exists()
    assert not orphan_dataset.exists()
    assert set(removed) == {orphan_rendered, orphan_dataset}


def test_rendered_thumbnail_from_camera_passes_dimensions_by_keyword(
    asset_manager_panel_module,
    tmp_path,
):
    calls = []

    def render_asset_preview_from_camera(
        path,
        eye,
        target,
        width,
        height,
        focal_length_mm=35.0,
        up=(0.0, 1.0, 0.0),
    ):
        calls.append((path, eye, target, width, height, focal_length_mm, up))
        return object()

    def save_image(path, image):
        Path(path).write_bytes(b"preview")

    asset_manager_panel_module.lf.render_asset_preview_from_camera = (
        render_asset_preview_from_camera
    )
    asset_manager_panel_module.lf.io = SimpleNamespace(save_image=save_image)

    asset_path = tmp_path / "asset.mesh"
    asset_path.write_bytes(b"asset")

    thumbnails = asset_manager_panel_module.AssetThumbnails(tmp_path / "thumbs")
    thumb_path = asyncio.run(
        thumbnails.generate_rendered_preview_from_camera(
            "mesh",
            "mesh_asset",
            asset_path,
            eye=(1.0, 2.0, 3.0),
            target=(4.0, 5.0, 6.0),
            up=(0.0, 0.0, 1.0),
        )
    )

    assert thumb_path is not None
    assert re.fullmatch(r"mesh_asset\.render\.\d+\.png", thumb_path.name)
    assert thumb_path.exists()
    assert calls == [
        (
            str(asset_path),
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            512,
            224,
            35.0,
            (0.0, 0.0, 1.0),
        )
    ]


def test_mesh_extension_detects_as_mesh(asset_manager_panel_module, tmp_path):
    mesh_path = tmp_path / "scan.mesh"
    mesh_path.write_bytes(b"mesh")

    scanner = asset_manager_panel_module.AssetScanner()

    assert scanner.detect_type(str(mesh_path)) == "mesh"


def test_asset_manager_load_context_actions_are_localized():
    folder_root = Path(__file__).parent.parent.parent
    locale_dir = folder_root / "src" / "visualizer" / "gui" / "resources" / "locales"
    required_keys = (
        "action.load_new",
        "action.add_to_scene",
        "action.refresh",
        "action.clean_missing",
        "tooltip.refresh",
        "tooltip.clean_missing",
        "import_from_url",
        "import_button",
        "import_button_downloading",
        "panel_title",
        "property.assets",
        "status_connecting",
        "status_extracting",
        "status_complete",
    )

    for locale_path in sorted(locale_dir.glob("*.json")):
        data = json.loads(locale_path.read_text(encoding="utf-8"))
        asset_manager = data["asset_manager"]
        for key in required_keys:
            assert asset_manager.get(key), f"{locale_path.name} missing asset_manager.{key}"


def test_asset_selection_dirties_info_fields(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        folders={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
        tags={},
        collections={},
    )

    panel.toggle_asset_selection(None, None, ["a1"])

    assert panel.get_selected_asset_name() == "bicycle"
    assert panel.get_selected_asset_path() == "/tmp/bicycle"
    assert panel.get_selected_asset_dataset_image_count() == "194"
    dirty = panel._handle.dirty_fields
    assert "selected_asset_path" in dirty or "__all__" in dirty
    assert "show_selection_asset" in dirty or "__all__" in dirty


def test_selected_asset_sh_degree_is_visible_for_any_geometry_asset_type(
    asset_manager_panel_module,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    asset = _make_asset()
    asset["type"] = "mesh"
    asset["geometry_metadata"] = {"sh_degree": 0}
    panel._asset_index = SimpleNamespace(
        assets={"a1": asset},
        folders={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
        tags={},
        collections={},
    )
    panel._selected_asset_ids = {"a1"}
    panel._selection_type = "asset"

    assert panel.get_selected_asset_has_sh_degree() is True
    assert panel.get_selected_asset_sh_degree() == "0"


def test_asset_selection_resolves_asset_id_from_clicked_element(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        folders={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
        tags={},
        collections={},
    )

    event = _EventStub(current_target=_ElementStub({"data-asset-id": "a1"}))
    panel.toggle_asset_selection(None, event, [])

    assert panel.get_selected_asset_name() == "bicycle"
    assert panel.get_selected_count() == 1


def test_generate_asset_thumbnail_prefers_dataset_preview(asset_manager_panel_module):
    class _Thumbnails:
        def __init__(self):
            self.placeholder_calls = []

        def generate_dataset_preview(
            self,
            asset_type,
            asset_id,
            asset_path,
            dataset_metadata,
        ):
            assert asset_type == "dataset"
            assert asset_id == "a1"
            assert asset_path == "/tmp/bicycle"
            assert dataset_metadata == {"image_root": "images"}
            return Path("/tmp/rendered-dataset.png")

        def generate_placeholder(self, asset_type, asset_id):
            self.placeholder_calls.append((asset_type, asset_id))
            return Path("/tmp/placeholder.png")

    updates = []
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._asset_thumbnails = _Thumbnails()
    panel._asset_index = SimpleNamespace(
        update_asset=lambda asset_id, **kwargs: updates.append((asset_id, kwargs))
    )

    panel._generate_asset_thumbnail_for_values(
        "a1",
        "dataset",
        "/tmp/bicycle",
        {"image_root": "images"},
    )
    time.sleep(0.1)

    assert panel._asset_thumbnails.placeholder_calls == []
    assert updates == [("a1", {"thumbnail_path": "/tmp/rendered-dataset.png"})]


def test_generate_asset_thumbnail_falls_back_to_placeholder(asset_manager_panel_module):
    class _Thumbnails:
        def generate_rendered_preview(self, asset_type, asset_id, asset_path):
            return None

        def generate_placeholder(self, asset_type, asset_id):
            return Path("/tmp/placeholder.png")

    updates = []
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._asset_thumbnails = _Thumbnails()
    panel._asset_index = SimpleNamespace(
        update_asset=lambda asset_id, **kwargs: updates.append((asset_id, kwargs))
    )

    panel._generate_asset_thumbnail_for_values("a1", "ply", "/tmp/model.ply", {})
    time.sleep(0.1)

    assert updates == [("a1", {"thumbnail_path": "/tmp/placeholder.png"})]


def test_dom_card_click_selects_asset_from_stable_parent(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        folders={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
        tags={},
        collections={},
    )

    container = _ElementStub({"id": "asset-popup-content"})
    card = _ElementStub(
        {"data-asset-id": "a1", "data-asset-action": "select"},
        parent=container,
    )
    child = _ElementStub(parent=card)
    event = _EventStub(current_target=container, target=child)

    panel._on_asset_manager_click(event)

    assert panel.get_selected_asset_name() == "bicycle"
    assert panel.get_selected_count() == 1
    assert event.stopped is True


def test_dom_card_ctrl_click_adds_to_multi_selection(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    first = _make_asset()
    second = _make_asset()
    second["id"] = "a2"
    second["name"] = "garden"
    second["absolute_path"] = "/tmp/garden"
    panel._asset_index = SimpleNamespace(
        assets={"a1": first, "a2": second},
        folders={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
        tags={},
        collections={},
    )

    container = _ElementStub({"id": "asset-popup-content"})
    card_a1 = _ElementStub(
        {"data-asset-id": "a1", "data-asset-action": "select"},
        parent=container,
    )
    card_a2 = _ElementStub(
        {"data-asset-id": "a2", "data-asset-action": "select"},
        parent=container,
    )

    panel._on_asset_manager_click(_EventStub(current_target=container, target=card_a1))
    panel._on_asset_manager_click(
        _EventStub(
            current_target=container,
            target=card_a2,
            bool_params={"ctrl_key": True},
        )
    )

    assert panel.get_selected_count() == 2
    assert panel.get_selection_type() == "multiple"


def test_dom_card_double_click_opens_dataset_import_panel(
    asset_manager_panel_module,
    monkeypatch,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        folders={
            "p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}
        },
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
        tags={},
        collections={},
    )
    calls = []
    monkeypatch.setattr(
        asset_manager_panel_module.os.path, "exists", lambda _path: True
    )
    monkeypatch.setattr(
        asset_manager_panel_module,
        "open_dataset_import_panel",
        lambda path, *, clear_scene_on_load=False: calls.append(
            (path, clear_scene_on_load)
        )
        or True,
    )

    container = _ElementStub({"id": "asset-popup-content"})
    card = _ElementStub(
        {"data-asset-id": "a1", "data-asset-action": "select"},
        parent=container,
    )
    child = _ElementStub(parent=card)
    event = _EventStub(current_target=container, target=child)

    panel._on_asset_manager_double_click(event)

    assert calls == [("/tmp/bicycle", False)]
    assert panel.get_selected_asset_name() == "bicycle"
    assert event.stopped is True


def test_dataset_load_new_opens_import_panel_with_scene_clear(
    asset_manager_panel_module,
    monkeypatch,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        folders={
            "p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}
        },
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
        tags={},
        collections={},
    )
    calls = []

    monkeypatch.setattr(
        asset_manager_panel_module.os.path,
        "exists",
        lambda _path: True,
    )
    monkeypatch.setattr(
        asset_manager_panel_module,
        "open_dataset_import_panel",
        lambda path, *, clear_scene_on_load=False: calls.append(
            (path, clear_scene_on_load)
        )
        or True,
    )

    panel._load_asset("a1", replace_scene=True)

    assert calls == [("/tmp/bicycle", True)]
    assert panel.get_selected_asset_name() == "bicycle"


def test_dom_card_double_click_ignored_during_input_capture(
    asset_manager_panel_module,
    monkeypatch,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        folders={
            "p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}
        },
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
        tags={},
        collections={},
    )
    asset_manager_panel_module.lf.keymap = SimpleNamespace(is_capturing=lambda: True)
    calls = []
    monkeypatch.setattr(
        asset_manager_panel_module.lf,
        "load_file",
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )

    container = _ElementStub({"id": "asset-popup-content"})
    card = _ElementStub(
        {"data-asset-id": "a1", "data-asset-action": "select"},
        parent=container,
    )
    event = _EventStub(current_target=container, target=card)

    panel._on_asset_manager_double_click(event)

    assert calls == []
    assert panel.get_selected_count() == 0
    assert event.stopped is False


def test_dataset_remove_deletes_catalog_json_entry(asset_manager_panel_module, tmp_path):
    index = asset_manager_panel_module.AssetIndex(
        library_path=tmp_path / "library.json"
    )
    folder = index.create_folder("truck")
    scene = index.create_scene(folder.id, "train")
    asset = index.create_asset(
        folder_id=folder.id,
        scene_id=scene.id,
        name="train",
        type="dataset",
        role="source_dataset",
        path="/tmp/train",
        absolute_path="/tmp/train",
        exists=True,
    )
    index.update_scene(scene.id, dataset_asset_id=asset.id)

    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = index
    panel._selected_asset_ids = {asset.id}
    panel._selection_type = "asset"

    panel.on_remove_from_catalog(None, None, [])

    data = json.loads((tmp_path / "library.json").read_text(encoding="utf-8"))
    assert asset.id not in data["assets"]
    assert scene.id not in data["scenes"]
    assert folder.id not in data["folders"]
    assert panel.get_selected_count() == 0


def test_edit_watch_dirs_uses_clicked_folder_without_selecting_it(
    asset_manager_panel_module,
    monkeypatch,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={},
        folders={
            "default": {"id": "default", "name": "Default", "scene_ids": []},
            "target": {"id": "target", "name": "Target", "scene_ids": []},
        },
        scenes={},
        tags={},
        collections={},
    )
    panel._selected_folder_id = "default"
    opened = []

    monkeypatch.setattr(
        asset_manager_panel_module,
        "open_watch_dirs_dialog",
        lambda folder_id: opened.append(folder_id) or True,
    )

    panel.on_edit_watch_dirs(None, None, ["target"])

    assert panel._selected_folder_id == "default"
    assert opened == ["target"]


def test_repair_selected_folder_prefers_default_name_when_selection_is_stale(
    asset_manager_panel_module,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._asset_index = SimpleNamespace(
        assets={},
        folders={
            "zeta": {"id": "zeta", "name": "Zeta", "scene_ids": []},
            "default": {"id": "default", "name": "Default", "scene_ids": []},
        },
        scenes={},
        tags={},
        collections={},
    )
    panel._selected_folder_id = "missing"
    panel._selection_type = "folder"

    assert panel._repair_selected_folder() == "default"
    assert panel._selected_folder_id == "default"


def test_bind_dom_event_listeners_registers_gallery_wheel_handler(
    asset_manager_panel_module,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    content = _ElementStub({"id": "asset-popup-content"})
    gallery_scroll = _ElementStub({"id": "asset-gallery-scroll"})
    doc = _DocumentStub(
        {
            "asset-popup-content": content,
            "asset-gallery-scroll": gallery_scroll,
        }
    )

    panel._bind_dom_event_listeners(doc)

    assert "mousescroll" in gallery_scroll.listeners
    assert "click" in content.listeners
    assert "mousemove" in doc.listeners


def test_gallery_precise_scroll_moves_scroll_container(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    gallery_scroll = _ElementStub({"id": "asset-gallery-scroll"})
    gallery_scroll.scroll_height = 900.0
    gallery_scroll.client_height = 300.0
    gallery_scroll.scroll_top = 120.0
    event = _EventStub(
        current_target=gallery_scroll,
        params={"wheel_delta_y": "1"},
    )

    panel._on_gallery_precise_scroll(event)

    assert gallery_scroll.scroll_top == 152.0
    assert event.stopped is True


def test_folder_count_matches_visible_list(asset_manager_panel_module, tmp_path):
    present_file = tmp_path / "present.ply"
    present_file.write_bytes(b"ply")

    def _asset(asset_id, path):
        asset = dict(_make_asset())
        asset["id"] = asset_id
        asset["absolute_path"] = str(path)
        asset["path"] = str(path)
        asset["folder_id"] = "p1"
        asset["scene_id"] = "s1"
        return asset

    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={
            "present": _asset("present", present_file),
            "missing": _asset("missing", tmp_path / "deleted.ply"),
        },
        folders={"p1": {"id": "p1", "name": "Default", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "scene", "folder_id": "p1"}},
        tags={},
        collections={},
    )

    visible = panel.get_filtered_assets()

    assert len(panel._asset_index.assets) == 2
    assert [asset["id"] for asset in visible] == ["present"]
    assert panel._folder_asset_count("p1") == len(visible)
    assert panel._scene_asset_count("s1") == 1
