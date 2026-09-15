# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the `.licht`-only Asset Manager panel."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.parse import quote
import json
import re
import sys
import threading
import time
import uuid

import pytest

def _install_lf_stub(monkeypatch):
    class _Panel:
        def __init__(self):
            pass

        def on_mount(self, _doc):
            pass

    context_menus = []
    state = SimpleNamespace(
        context_menus=context_menus,
        confirm_dialogs=[],
        message_dialogs=[],
        opened=[],
        revealed=[],
        enabled=[],
        drag_begins=[],
        drag_ends=[],
        drag_cancels=[],
        released_textures=[],
        dialog_path="",
        folder_dialog_path="",
        project_dirty=False,
    )

    def show_context_menu(items, x, y, on_action=None):
        context_menus.append(
            {"items": items, "position": (x, y), "on_action": on_action}
        )

    def begin_drag_payload(payload_type, data, label=""):
        token = len(state.drag_begins) + 1
        state.drag_begins.append((token, payload_type, data, label))
        return token

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        Panel=_Panel,
        PanelSpace=SimpleNamespace(
            FLOATING="FLOATING",
            LEFT_DOCK="LEFT_DOCK",
        ),
        PanelHeightMode=SimpleNamespace(FILL="FILL", CONTENT="CONTENT"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        tr=lambda key: {
            "asset_manager.dialog.remove_folder_message": 'Remove "{name}" with {count} projects?',
            "asset_manager.status.scanning": "Scanning {name}: {folders} folders, {projects} projects found",
            "asset_manager.action.stop_scan": "Stop scan",
            "asset_manager.status.scan_stopped": "Scan stopped",
        }.get(key, key),
        get_current_language=lambda: "en",
        get_mouse_screen_pos=lambda: (120.0, 220.0),
        show_context_menu=show_context_menu,
        confirm_dialog=lambda title, message, buttons, callback=None: state.confirm_dialogs.append(
            (title, message, buttons, callback)
        ),
        message_dialog=lambda title, message, style=None: state.message_dialogs.append(
            (title, message, style)
        ),
        reveal_in_file_manager=lambda path: state.revealed.append(path) or True,
        open_project_file_dialog=lambda _start="": state.dialog_path,
        open_folder_dialog=lambda _title, _start="": state.folder_dialog_path,
        get_project_location=lambda: "/home/tester/.lichtfeld/projects",
        input_dialog=lambda *_args: None,
        set_panel_enabled=lambda panel_id, enabled: state.enabled.append(
            (panel_id, enabled)
        ),
        schedule_on_ui_thread=lambda callback: callback(),
        begin_drag_payload=begin_drag_payload,
        end_drag_payload=lambda token: state.drag_ends.append(token),
        cancel_drag_payload=lambda token: state.drag_cancels.append(token),
        release_rml_texture=lambda source: state.released_textures.append(source) or True,
    )
    lf_stub.log = SimpleNamespace(info=lambda _msg: None, warn=lambda _msg: None, error=lambda _msg: None)
    lf_stub.project_is_dirty = lambda: state.project_dirty
    lf_stub.project_has_path = lambda: False
    lf_stub.is_training_active = lambda: False
    lf_stub.project_open = (
        lambda path, discard_changes=False, stop_training=False, keep_asset_manager_open=False: state.opened.append(
            (path, discard_changes, stop_training, keep_asset_manager_open)
        )
    )
    lf_stub.is_dataset_path = lambda _path: True
    lf_stub.read_checkpoint_header = lambda _path: object()
    lf_stub.read_checkpoint_params = lambda _path: object()
    lf_stub.load_file = lambda *_args, **_kwargs: None
    lf_stub.load_config_file = lambda *_args, **_kwargs: None
    lf_stub._test_state = state
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

@pytest.fixture
def panel_module(monkeypatch):
    source_python = Path(__file__).resolve().parents[2] / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    for name in list(sys.modules):
        if name == "lfs_plugins" or name.startswith("lfs_plugins."):
            sys.modules.pop(name, None)
    _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.asset_manager_panel")
    yield module
    controller_module = sys.modules.get("lfs_plugins.gallery_controller")
    controller = getattr(controller_module, "_controller", None)
    if controller is not None:
        _stop_gallery_controller(controller)

def _stop_gallery_controller(controller):
    """Stop fixture-owned timers after the singleton's test lifetime."""
    controller._poll = lambda: None
    if controller._timer:
        controller._timer.cancel()
        controller._timer = None
    controller._subscribers.clear()


class _Handle:
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

class _BindingModel:
    def __init__(self):
        self.func_bindings = {}
        self.handle = _Handle()

    def bind(self, name, getter, setter=None):
        return None

    def bind_func(self, name, getter):
        self.func_bindings[name] = getter

    def bind_event(self, name, handler):
        return None

    def bind_record_list(self, name):
        return None

    def get_handle(self):
        return self.handle

class _BindingContext:
    def __init__(self, model):
        self._model = model

    def create_data_model(self, _name):
        return self._model

class _Element:
    def __init__(self, attrs=None, parent=None):
        self.attrs = attrs or {}
        self._parent = parent
        self.children = []
        self.classes = set(str(self.attrs.get("class", "")).split())
        self.listeners = {}
        self.scroll_top = 0.0
        self.scroll_height = 900.0
        self.client_height = 300.0
        self.client_width = 800.0
        self.focused = False
        if parent is not None:
            parent.children.append(self)

    def get_attribute(self, name, default=""):
        return self.attrs.get(name, default)

    def has_attribute(self, name):
        return name in self.attrs

    def parent(self):
        return self._parent

    def add_event_listener(self, event, callback):
        self.listeners[event] = callback

    def query_selector_all(self, selectors):
        wanted = {
            selector.strip().removeprefix(".")
            for selector in selectors.split(",")
        }
        rows = []

        def visit(node):
            for child in node.children:
                if child.classes.intersection(wanted):
                    rows.append(child)
                visit(child)

        visit(self)
        return rows

    def set_class(self, name, enabled):
        if enabled:
            self.classes.add(name)
        else:
            self.classes.discard(name)

    def is_class_set(self, name):
        return name in self.classes

    def focus(self):
        self.focused = True

class _Event:
    def __init__(self, current_target=None, target=None, params=None, bool_params=None):
        self._current_target = current_target
        self._target = target or current_target
        self.params = params or {}
        self.bool_params = bool_params or {}
        self.stopped = False

    def current_target(self):
        return self._current_target

    def target(self):
        return self._target

    def get_parameter(self, name, default=""):
        return self.params.get(name, default)

    def get_bool_parameter(self, name, default=False):
        return self.bool_params.get(name, default)

    def stop_propagation(self):
        self.stopped = True

class _Document:
    def __init__(self, elements=None):
        self.elements = elements or {}
        self.listeners = {}

    def get_element_by_id(self, element_id):
        return self.elements.get(element_id)

    def add_event_listener(self, event, callback):
        self.listeners[event] = callback

_MIN_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)

def _write_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_MIN_PNG)
    return path

def _project(project_id="11111111-1111-4111-8111-111111111111", **overrides):
    value = {
        "id": project_id,
        "project_uuid": project_id,
        "name": "Bicycle",
        "path": "/tmp/bicycle project.licht",
        "folder_id": "default",
        "file_uuid": "22222222-2222-4222-8222-222222222222",
        "commit_uuid": "33333333-3333-4333-8333-333333333333",
        "generation": 4,
        "created_at_unix_ns": 1_700_000_000_000_000_000,
        "saved_at_unix_ns": 1_710_000_000_000_000_000,
        "file_size_bytes": 4_206_437_268,
        "role": "MASTER",
        "open_state": "OPEN",
        "has_preview": True,
        "fallback_preview_path": "",
        "exists": True,
        "available": True,
        "status": "AVAILABLE",
        "error": "",
    }
    value.update(overrides)
    return value

def _index(assets=None, folders=None, **methods):
    payload = {"load_issues": []}
    payload.update(methods)
    return SimpleNamespace(
        assets=assets or {},
        folders=folders
        or {
            "default": {
                "id": "default",
                "name": "assets",
                "path": "/home/tester/.lichtfeld/assets",
            }
        },
        **payload,
    )

def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False

def _scan_result(**overrides):
    value = SimpleNamespace(
        discovered=0, added=0, already_cataloged=0, failed=0, cancelled=False
    )
    for key, item in overrides.items():
        setattr(value, key, item)
    return value

def test_panel_contract_polls_preference_and_remains_left_dock(panel_module):
    panel_type = panel_module.AssetManagerPanel
    assert panel_type.update_policy == "dirty"
    assert panel_type.update_interval_ms == 100
    assert panel_type.space == panel_module.lf.ui.PanelSpace.LEFT_DOCK
    assert panel_type.order == 20

def test_rml_and_panel_have_no_scene_or_disk_thumbnail_model():
    root = Path(__file__).resolve().parents[2]
    rml = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rml").read_text()
    rcss = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rcss").read_text()
    source = (root / "src/python/lfs_plugins/asset_manager_panel.py").read_text()

    assert "scene" not in rml.casefold()
    assert "scene" not in source.casefold()
    assert "scene-asset" not in rcss
    assert "absolute_path" not in source
    assert "fingerprint" not in source
    assert 'bind_record_list("thumbnails")' not in source
    assert "LICHT" not in rml
    assert "asset-col-type" not in rml
    assert "asset-pill-licht" not in rcss
    assert "asset-card-overlay" not in rcss
    assert "col_type_label" not in source
    assert "folder_pill_label" not in source
    assert "asset-pill-folder" not in rml
    assert ".asset-pill {" not in rcss
    assert ".asset-pill-folder" not in rcss
    assert 'data-style-decorator="asset.thumbnail_decorator"' in rml

def test_results_header_uses_icon_views_and_nondestructive_refresh_action():
    root = Path(__file__).resolve().parents[2]
    rml = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rml").read_text()
    rcss = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rcss").read_text()

    assert "{{gallery_label}}" not in rml
    assert "{{list_label}}" not in rml
    assert "asset-icon-grid" in rml
    assert "asset-icon-list" in rml
    assert 'data-event-click="refresh_and_clean"' not in rml
    assert rml.count('data-event-click="refresh_catalog"') == 1
    assert 'data-if="has_scan_status"' in rml
    assert "{{scan_status}}" in rml
    assert 'data-class-is-stop="scan_active"' in rml
    assert 'data-attr-data-tooltip="refresh_action_tooltip"' in rml
    assert "{{stop_scan_label}}" in rml
    assert 'data-event-click="clean_missing"' not in rml
    assert "asset-list-secondary" in rml
    assert "display_subtitle" not in rml
    assert ".asset-refresh-button img" in rcss
    assert ".asset-refresh-clean-button" not in rcss
    assert 'data-if="has_catalog_notice"' in rml
    assert "{{catalog_notice}}" in rml
    assert "width: 18dp;" in rcss

def test_all_asset_manager_buttons_use_strict_size_variants():
    root = Path(__file__).resolve().parents[2]
    rml = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rml").read_text()
    rcss = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rcss").read_text()

    button_tags = [part.split(">", 1)[0] for part in rml.split("<button")[1:]]
    assert button_tags
    assert all("asset-button" in tag or "btn btn--" in tag for tag in button_tags)
    assert all(
        "btn btn--" in tag
        or "asset-button--text" in tag
        or "asset-button--icon" in tag
        or "asset-button--small-icon" in tag
        for tag in button_tags
    )
    assert ".asset-button {" in rcss
    assert "max-height: 28dp;" in rcss
    assert ".asset-button--icon {" in rcss
    assert "max-width: 28dp;" in rcss
    assert ".asset-button--small-icon {" in rcss
    assert "max-height: 24dp;" in rcss
    assert '<span class="asset-button-text">{{import_project_label}}</span>' in rml
    assert '<span class="asset-button-glyph">&#215;</span>' in rml
    assert ".asset-button-text {" in rcss
    assert ".asset-button-glyph {" in rcss
    assert "margin-top: 4dp;" in rcss
    assert "padding: 0 12dp 32dp 12dp;" in rcss
    assert "padding: 12dp 0 0 0;" in rcss

def test_embedded_preview_url_encodes_path_and_keys_cache_by_commit(panel_module):
    panel = panel_module.AssetManagerPanel()
    asset = _project(path="/tmp/a folder/project & one.licht")

    decorator = panel._thumbnail_decorator(asset)

    encoded = quote(asset["path"], safe="/:._-~")
    assert " " not in encoded
    assert decorator == (
        "image(preview://kind=licht&thumb=256"
        f"&rev={asset['commit_uuid']}&path={encoded} cover center)"
    )
    assert asset["path"] not in decorator
    assert panel._thumbnail_decorator({**asset, "has_preview": False}) == "none"

def test_thumbnail_decorator_embedded_fallback_and_none(panel_module, tmp_path):
    panel = panel_module.AssetManagerPanel()
    fallback_image = _write_png(tmp_path / "dataset" / "frame 1.png")
    stat = fallback_image.stat()
    encoded_fallback = quote(str(fallback_image), safe="/:._-~")
    fallback_rev = quote(f"{stat.st_size}-{stat.st_mtime_ns}", safe="-._~")
    assert " " not in encoded_fallback
    expected_fallback = (
        "image(preview://kind=image&thumb=256"
        f"&rev={fallback_rev}&path={encoded_fallback} cover center)"
    )

    embedded = _project()
    embedded_decorator = panel._thumbnail_decorator(embedded)
    encoded_project = quote(embedded["path"], safe="/:._-~")
    assert " " not in encoded_project
    assert embedded_decorator == (
        "image(preview://kind=licht&thumb=256"
        f"&rev={embedded['commit_uuid']}&path={encoded_project} cover center)"
    )
    embedded_row = panel._format_asset_for_ui(embedded)
    assert embedded_row["shows_placeholder"] is False
    assert embedded_row["has_preview"] is True

    fallback_asset = _project(
        has_preview=False,
        fallback_preview_path=str(fallback_image),
    )
    fallback_decorator = panel._thumbnail_decorator(fallback_asset)
    assert fallback_decorator == expected_fallback
    assert "kind=image" in fallback_decorator
    assert "kind=licht" not in fallback_decorator
    assert "frame 1.png" not in fallback_decorator
    fallback_row = panel._format_asset_for_ui(fallback_asset)
    assert fallback_row["shows_placeholder"] is False
    assert fallback_row["has_preview"] is False
    assert fallback_row["thumbnail_decorator"] == expected_fallback

    none_asset = _project(has_preview=False)
    assert panel._thumbnail_decorator(none_asset) == "none"
    none_row = panel._format_asset_for_ui(none_asset)
    assert none_row["shows_placeholder"] is True
    assert none_row["has_preview"] is False

    missing_fallback = _project(
        has_preview=False,
        fallback_preview_path=str(tmp_path / "missing.png"),
    )
    assert panel._thumbnail_decorator(missing_fallback) == "none"

    both = _project(fallback_preview_path=str(fallback_image))
    assert "kind=licht" in panel._thumbnail_decorator(both)

def test_fallback_thumbnail_source_is_tracked_and_released(panel_module, tmp_path):
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index()
    fallback_image = _write_png(tmp_path / "dataset" / "preview.png")
    asset = _project(has_preview=False, fallback_preview_path=str(fallback_image))

    first = panel._format_asset_for_ui(asset)["thumbnail_decorator"]
    first_source = panel._thumbnail_source_from_decorator(first)
    assert first.startswith("image(preview://kind=image")
    assert first.endswith(" cover center)")
    assert first_source.startswith("preview://kind=image")
    assert " cover center" not in first_source
    assert panel._thumbnail_sources_by_asset[asset["id"]] == first_source

    fallback_image.write_bytes(_MIN_PNG + b"\x00")
    second = panel._format_asset_for_ui(asset)["thumbnail_decorator"]
    second_source = panel._thumbnail_source_from_decorator(second)
    assert first != second
    assert panel._thumbnail_sources_by_asset[asset["id"]] == second_source
    assert panel_module.lf._test_state.released_textures == [first_source]

    asset["fallback_preview_path"] = ""
    none_row = panel._format_asset_for_ui(asset)
    assert none_row["thumbnail_decorator"] == "none"
    assert none_row["shows_placeholder"] is True
    assert asset["id"] not in panel._thumbnail_sources_by_asset
    assert panel_module.lf._test_state.released_textures == [first_source, second_source]

def test_thumbnail_source_strips_cover_center_suffix(panel_module, tmp_path):
    panel = panel_module.AssetManagerPanel()
    fallback_image = _write_png(tmp_path / "dataset" / "frame 1.png")
    embedded = _project()
    fallback = _project(
        has_preview=False,
        fallback_preview_path=str(fallback_image),
    )

    for asset in (embedded, fallback):
        decorator = panel._format_asset_for_ui(asset)["thumbnail_decorator"]
        source = panel._thumbnail_source_from_decorator(decorator)
        assert decorator == f"image({source} cover center)"
        assert " " not in source
        assert source.startswith("preview://")
        assert "cover" not in source
        assert "center" not in source
        assert panel._thumbnail_sources_by_asset[asset["id"]] == source

    none_source = panel._thumbnail_source_from_decorator("none")
    assert none_source == ""

def test_asset_rows_use_custom_name_and_runtime_metadata(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(assets={_project()["id"]: _project()})
    panel._selected_folder_id = "default"

    row = panel.get_filtered_assets()[0]

    assert row["display_name"] == "Bicycle"
    assert "display_subtitle" not in row
    assert row["status_label"] == "asset_manager.status.available"
    assert row["saved_label"]
    assert row["thumbnail_decorator"].startswith("image(preview://kind=licht")

def test_selecting_project_updates_info_without_rebuilding_rows(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    asset = _project()
    panel._asset_index = _index(assets={asset["id"]: asset})

    panel.toggle_asset_selection(None, None, [asset["id"]])

    assert panel.get_selection_type() == "asset"
    assert panel.get_selected_asset_name() == "Bicycle"
    assert panel.get_selected_asset_path() == asset["path"]
    assert panel.get_selected_asset_has_folder() is True
    assert "selected_asset_path" in panel._handle.dirty_fields
    assert "selected_asset_has_folder" in panel._handle.dirty_fields
    assert "selected_asset_has_relocation_candidate" in panel._handle.dirty_fields
    assert "has_catalog_notice" in panel._handle.dirty_fields
    assert "assets" not in panel._handle.records

def test_dom_right_click_uses_shared_app_context_menu(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    asset = _project()
    panel._asset_index = _index(
        assets={asset["id"]: asset},
        folders={
            "default": {"id": "default", "name": "assets", "path": "/tmp"},
            "archive": {"id": "archive", "name": "Archive", "path": "/archive"},
        },
    )
    shell = _Element()
    row = _Element(
        {"class": "asset-list-row", "data-asset-id": asset["id"], "data-asset-action": "select"},
        shell,
    )
    event = _Event(shell, row, params={"button": "1"})

    panel._on_asset_manager_mousedown(event)

    menu = panel_module.lf._test_state.context_menus[-1]
    assert menu["position"] == (120.0, 220.0)
    assert [item["action"] for item in menu["items"]] == [
        "load",
        "gallery:publish",
        "rename",
        "show_in_folder",
        "remove",
    ]
    assert event.stopped is True

def test_gallery_more_button_uses_same_shared_menu(panel_module):
    panel = panel_module.AssetManagerPanel()
    asset = _project()
    panel._asset_index = _index(assets={asset["id"]: asset})
    shell = _Element()
    button = _Element(
        {"data-asset-id": asset["id"], "data-asset-action": "menu"}, shell
    )
    event = _Event(shell, button)

    panel._on_asset_manager_click(event)

    assert len(panel_module.lf._test_state.context_menus) == 1
    assert event.stopped is True

def test_real_folder_menu_reveals_or_removes_mapping(panel_module, monkeypatch):
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(
        folders={"projects": {"id": "projects", "name": "Projects", "path": "/tmp/projects"}}
    )
    calls = []
    monkeypatch.setattr(
        panel,
        "on_delete_folder",
        lambda _handle, _event, args: calls.append(tuple(args)),
    )

    assert panel._show_folder_context_menu("projects") is True
    menu = panel_module.lf._test_state.context_menus[-1]
    assert [item["action"] for item in menu["items"]] == [
        "show",
        "remove",
    ]
    menu["on_action"]("show")
    assert panel_module.lf._test_state.revealed == ["/tmp/projects"]
    menu["on_action"]("remove")
    assert calls == [("projects",)]

def test_open_project_verifies_then_uses_project_lifecycle(panel_module):
    panel = panel_module.AssetManagerPanel()
    asset = _project()
    project = SimpleNamespace(to_dict=lambda: asset)
    panel._handle = _Handle()
    panel._asset_index = _index(
        assets={asset["id"]: asset},
        verify_asset=lambda project_id: project if project_id == asset["id"] else None,
    )

    panel._load_asset(asset["id"])

    assert panel_module.lf._test_state.opened == [
        (asset["path"], True, False, True)
    ]
    assert panel.get_selected_asset_id() == asset["id"]

def test_gallery_toolbar_opens_transfers_without_opening_project(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel.on_open_gallery()
    assert panel_module.lf._test_state.opened == []
    assert ("lfs.gallery_transfer", True) in panel_module.lf._test_state.enabled

def test_open_project_confirms_before_discarding_unsaved_changes(panel_module):
    panel = panel_module.AssetManagerPanel()
    asset = _project()
    panel._handle = _Handle()
    panel._asset_index = _index(
        assets={asset["id"]: asset},
        verify_asset=lambda _project_id: SimpleNamespace(
            to_dict=lambda: asset
        ),
    )
    state = panel_module.lf._test_state
    state.project_dirty = True

    panel._load_asset(asset["id"])

    assert state.opened == []
    assert len(state.confirm_dialogs) == 1
    title, _message, buttons, callback = state.confirm_dialogs[0]
    assert title == "menu.file.open_project"
    assert buttons == [
        "menu.file.save_project_as",
        "unsaved_work.continue_without_saving",
        "common.cancel",
    ]

    callback("common.cancel")
    assert state.opened == []

    callback("unsaved_work.continue_without_saving")
    assert state.opened == [(asset["path"], True, False, True)]

def test_import_registers_only_selected_licht_project(panel_module):
    panel = panel_module.AssetManagerPanel()
    asset = _project()
    calls = []
    panel_module.lf._test_state.dialog_path = asset["path"]
    panel._selected_folder_id = "default"
    panel._asset_index = _index(
        register_licht_asset=lambda path, folder_id=None: (
            calls.append((path, folder_id)) or SimpleNamespace(id=asset["id"]),
            True,
        ),
        verify_projects=lambda: (0, 1),
    )
    panel.refresh_catalog = lambda **_kwargs: None

    panel.on_import_project()

    assert calls == [(asset["path"], None)]
    assert panel.get_selected_asset_id() == asset["id"]

def test_add_folder_uses_real_directory_picker(panel_module):
    panel = panel_module.AssetManagerPanel()
    selected = "/tmp/assets"
    panel_module.lf._test_state.folder_dialog_path = selected
    calls = []
    panel._asset_index = _index(
        add_folder=lambda path: calls.append(path)
        or SimpleNamespace(id="selected-folder"),
        verify_projects=lambda: (0, 0),
    )
    panel.refresh_catalog = lambda **_kwargs: None
    panel._scan_asset_folders = lambda **_kwargs: None

    panel.on_add_folder()

    assert calls == [selected]
    assert panel._selected_folder_id == "selected-folder"

def test_folder_counts_match_search_results(panel_module):
    first = _project(name="Bicycle")
    second = _project(
        "44444444-4444-4444-8444-444444444444",
        name="Garden",
        path="/tmp/garden.licht",
    )
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(assets={first["id"]: first, second["id"]: second})
    panel._selected_folder_id = "default"
    panel._search_query = "garden"

    assert len(panel.get_filtered_assets()) == 1
    assert panel.get_folder_list()[0]["project_count"] == 1

def test_search_matches_path_and_type(panel_module):
    asset = _project(type="capture")
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(assets={asset["id"]: asset})

    for query in ("bicycle project.licht", "capture", "licht project"):
        panel._search_query = query
        assert [row["id"] for row in panel.get_filtered_assets()] == [asset["id"]]

def test_all_assets_navigation_and_folder_scopes_filter_catalog(panel_module):
    first = _project(name="Bicycle")
    second = _project(
        "44444444-4444-4444-8444-444444444444",
        name="Garden",
        path="/tmp/garden.licht",
        folder_id="archive",
    )
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(
        assets={first["id"]: first, second["id"]: second},
        folders={
            "default": {"id": "default", "name": "Default", "path": "/tmp/default"},
            "archive": {"id": "archive", "name": "Archive", "path": "/tmp/archive"},
        },
    )

    assert panel._selected_folder_id == panel_module.SCOPE_ALL
    assert [row["id"] for row in panel.get_filtered_assets()] == [first["id"], second["id"]]
    assert panel._select_folder_id("archive") is True
    assert [row["id"] for row in panel.get_filtered_assets()] == [second["id"]]

    folders = panel.get_folder_list()
    assert {row["id"] for row in folders} == {"default", "archive"}
    assert all(row["can_manage"] for row in folders)
    assert panel.get_all_assets_count() == 2

def test_folder_tree_is_expanded_by_default(panel_module):
    panel = panel_module.AssetManagerPanel()
    assert panel._folders_collapsed is False

def test_precise_scroll_moves_gallery_container(panel_module):
    panel = panel_module.AssetManagerPanel()
    scroll = _Element()
    scroll.scroll_top = 120.0
    event = _Event(scroll, params={"wheel_delta_y": "1"})

    panel._on_gallery_precise_scroll(event)

    assert scroll.scroll_top == 152.0
    assert event.stopped is True

def test_mount_binds_stable_delegated_handlers(panel_module):
    panel = panel_module.AssetManagerPanel()
    shell = _Element()
    scroll = _Element()
    doc = _Document({"asset-shell": shell, "asset-gallery-scroll": scroll})

    panel._bind_dom_event_listeners(doc)

    assert {"mousedown", "click", "dblclick", "dragstart", "dragend"}.issubset(shell.listeners)
    assert {"scroll", "mousescroll", "keydown"}.issubset(scroll.listeners)
    assert {"mousemove", "mouseup"}.issubset(doc.listeners)

def test_keyboard_navigation_enter_delete_and_typeahead(panel_module, monkeypatch):
    first = _project(name="Alpha", path="/tmp/alpha.licht")
    second = _project(
        "44444444-4444-4444-8444-444444444444",
        name="Beta",
        path="/tmp/beta.licht",
    )
    third = _project(
        "55555555-5555-4555-8555-555555555555",
        name="Gamma",
        path="/tmp/gamma.licht",
    )
    assets = {first["id"]: first, second["id"]: second, third["id"]: third}
    deleted = []

    def delete_assets(asset_ids):
        deleted.append(list(asset_ids))
        for asset_id in asset_ids:
            assets.pop(asset_id, None)
        return len(asset_ids)

    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(assets=assets, delete_assets=delete_assets)
    scroll = _Element()
    search = _Element()
    panel._doc = _Document({"asset-gallery-scroll": scroll, "asset-search-input": search})

    panel._on_asset_results_keydown(
        _Event(scroll, params={"key_identifier": str(panel_module.KI_DOWN)})
    )
    assert panel.get_selected_asset_id() == first["id"]
    panel._on_asset_results_keydown(
        _Event(scroll, params={"key_identifier": str(panel_module.KI_DOWN)})
    )
    assert panel.get_selected_asset_id() == second["id"]

    opened = []
    monkeypatch.setattr(panel, "_load_asset", lambda asset_id: opened.append(asset_id))
    panel._on_asset_results_keydown(
        _Event(scroll, params={"key_identifier": str(panel_module.KI_RETURN)})
    )
    assert opened == [second["id"]]

    panel._on_asset_results_keydown(
        _Event(scroll, params={"key_identifier": str(panel_module.KI_DELETE)})
    )
    assert deleted == [[second["id"]]]
    assert panel.get_selected_asset_id() == third["id"]

    panel._on_asset_results_keydown(_Event(scroll, params={"key_identifier": "12"}))
    assert search.focused is True
    assert panel.get_search_query() == "a"

def test_filter_clamps_hidden_selection_for_enter_and_delete(panel_module, monkeypatch):
    alpha = _project(name="Alpha", path="/tmp/alpha.licht")
    beta = _project(
        "44444444-4444-4444-8444-444444444444",
        name="Beta",
        path="/tmp/beta.licht",
    )
    deleted = []
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(
        assets={alpha["id"]: alpha, beta["id"]: beta},
        delete_assets=lambda ids: deleted.extend(ids) or len(ids),
    )
    panel._selected_asset_ids = {alpha["id"]}
    panel._selection_cursor_id = alpha["id"]

    panel.set_search_query("beta")

    assert panel._selected_asset_ids == set()
    assert panel._selection_cursor_id is None
    panel._selected_asset_ids = {alpha["id"]}
    panel._selection_cursor_id = alpha["id"]
    assert panel._delete_selected_assets() is False
    opened = []
    monkeypatch.setattr(panel, "_load_asset", opened.append)
    panel._on_asset_results_keydown(
        _Event(params={"key_identifier": str(panel_module.KI_RETURN)})
    )
    assert deleted == []
    assert opened == []

def test_gallery_keyboard_navigation_uses_visual_columns(panel_module):
    assets = {}
    for index, name in enumerate(("Alpha", "Beta", "Gamma", "Omega"), start=1):
        asset = _project(
            f"{index:08d}-1111-4111-8111-111111111111",
            name=name,
            path=f"/tmp/{name.casefold()}.licht",
        )
        assets[asset["id"]] = asset
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(assets=assets)
    panel._view_mode = "gallery"
    panel._asset_window_client_width = 500.0

    assert panel._navigate_selection(panel_module.KI_RIGHT) is True
    assert panel.get_selected_asset_id() == list(assets)[0]
    panel._navigate_selection(panel_module.KI_DOWN)
    assert panel.get_selected_asset_id() == list(assets)[2]
    panel._navigate_selection(panel_module.KI_LEFT)
    assert panel.get_selected_asset_id() == list(assets)[1]

def test_toolbar_refresh_does_not_verify_on_ui_thread_then_scans(
    panel_module, monkeypatch
):
    missing = _project(exists=False, available=False, status="MISSING")
    present = _project(
        "44444444-4444-4444-8444-444444444444",
        name="Garden",
        path="/tmp/garden.licht",
    )
    assets = {missing["id"]: missing, present["id"]: present}
    calls = []

    def verify_projects():
        calls.append("verify")
        return 1, 2

    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(
        assets=assets,
        verify_projects=verify_projects,
    )
    monkeypatch.setattr(panel, "_scan_asset_folders", lambda: calls.append("scan"))
    monkeypatch.setattr(panel, "_start_catalog_verify", lambda: calls.append("verify_bg"))

    panel.refresh_catalog()

    assert calls == ["verify_bg", "scan"]
    assert set(assets) == {missing["id"], present["id"]}

def test_delete_folder_requires_confirmation_with_project_count(panel_module):
    first = _project(folder_id="projects")
    second = _project(
        "44444444-4444-4444-8444-444444444444",
        folder_id="projects",
    )
    deleted = []
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(
        assets={first["id"]: first, second["id"]: second},
        folders={"projects": {"id": "projects", "name": "Work"}},
        delete_folder=lambda folder_id: deleted.append(folder_id) or True,
        verify_projects=lambda: (0, 0),
    )
    panel.refresh_catalog = lambda **_kwargs: None

    panel.on_delete_folder(None, None, ["projects"])

    assert deleted == []
    title, message, buttons, callback = panel_module.lf._test_state.confirm_dialogs[-1]
    assert title == "asset_manager.dialog.remove_folder"
    assert message == 'Remove "Work" with 2 projects?'
    assert buttons[-1] == "asset_manager.action.remove_folder"
    callback("common.cancel")
    assert deleted == []
    callback("asset_manager.action.remove_folder")
    assert deleted == ["projects"]

def test_identity_mismatch_has_distinct_status(panel_module):
    panel = panel_module.AssetManagerPanel()

    assert panel._project_status_label({"status": "IDENTITY_MISMATCH"}) == (
        "asset_manager.status.identity_mismatch"
    )


@pytest.mark.parametrize("status", ["IDENTITY_MISMATCH", "UNREADABLE"])
def test_gallery_check_rescans_registered_folder_for_identity_mismatch(panel_module, status):
    asset = _project(status=status, exists=True, available=False)
    calls = []
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(assets={asset["id"]: asset})
    panel._selected_asset_ids = {asset["id"]}
    panel._selection_cursor_id = asset["id"]
    panel._update_selection_type()
    panel._gallery_controller = SimpleNamespace(
        refresh=lambda: calls.append("portal_refresh")
    )
    panel.refresh_catalog = lambda **kwargs: calls.append(("catalog_refresh", kwargs))
    panel._scan_asset_folders = lambda **kwargs: calls.append(("folder_scan", kwargs))

    panel._gallery_command("check")

    assert calls == [
        ("catalog_refresh", {"scan_folders": False}),
        (
            "folder_scan",
            {
                "folder_id": "default",
                "directory": "/home/tester/.lichtfeld/assets",
            },
        ),
    ]


def test_thumbnail_revision_falls_back_and_releases_stale_source(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index()
    asset = _project(commit_uuid="", generation=4)

    first = panel._format_asset_for_ui(asset)["thumbnail_decorator"]
    asset["generation"] = 5
    second = panel._format_asset_for_ui(asset)["thumbnail_decorator"]

    assert "rev=4-" in first
    assert "rev=5-" in second
    assert first.endswith(" cover center)")
    assert second.endswith(" cover center)")
    assert first != second
    assert panel_module.lf._test_state.released_textures == [
        panel._thumbnail_source_from_decorator(first)
    ]

def test_completed_project_save_reverifies_catalog_thumbnail(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    asset = _project(commit_uuid="old", generation=4)
    project = SimpleNamespace(id=asset["id"])
    verified = []

    def verify_asset(asset_id):
        verified.append(asset_id)
        asset["commit_uuid"] = "new"
        asset["generation"] = 5
        return project

    panel._asset_index = _index(
        assets={asset["id"]: asset},
        find_asset_by_path=lambda path: project if path == asset["path"] else None,
        verify_asset=verify_asset,
    )
    polls = [
        {"running": False, "generation": 4, "path": asset["path"], "error": ""},
        {"running": False, "generation": 5, "path": asset["path"], "error": ""},
    ]
    panel_module.lf.project_poll_write = lambda: polls.pop(0)

    assert panel._refresh_after_project_write() is False
    assert panel._refresh_after_project_write() is True
    assert verified == [asset["id"]]
    assert "rev=new" in panel._handle.records["assets"][0]["thumbnail_decorator"]

def test_drag_available_project_publishes_typed_payload(panel_module):
    panel = panel_module.AssetManagerPanel()
    asset = _project()
    panel._handle = _Handle()
    panel._asset_index = _index(
        assets={asset["id"]: asset},
        verify_asset=lambda _asset_id: SimpleNamespace(to_dict=lambda: asset),
    )
    shell = _Element()
    row = _Element(
        {
            "class": "asset-list-row is-draggable",
            "data-asset-id": asset["id"],
            "data-asset-action": "select",
        },
        shell,
    )

    start = _Event(shell, row)
    panel._on_asset_drag_start(start)

    state = panel_module.lf._test_state
    assert state.drag_begins == [
        (1, panel_module.PROJECT_DRAG_PAYLOAD_TYPE, asset["path"], "Bicycle")
    ]
    assert start.stopped is True
    assert panel.get_selected_asset_id() == asset["id"]

    end = _Event(shell, row)
    panel._on_asset_drag_end(end)
    assert state.drag_ends == [1]
    assert end.stopped is True

def test_drag_missing_project_is_rejected(panel_module, monkeypatch):
    panel = panel_module.AssetManagerPanel()
    asset = _project()
    panel._asset_index = _index(
        assets={asset["id"]: asset},
        verify_asset=lambda _asset_id: None,
    )
    monkeypatch.setattr(panel, "refresh_catalog", lambda **_kwargs: None)
    shell = _Element()
    row = _Element(
        {"data-asset-id": asset["id"], "data-asset-action": "select"}, shell
    )

    panel._on_asset_drag_start(_Event(shell, row))

    assert panel_module.lf._test_state.drag_begins == []

def test_default_folder_links_to_settings_instead_of_removal(panel_module):
    panel = panel_module.AssetManagerPanel()
    assert panel._folder_context_menu_items("default") == [
        {
            "label": "asset_manager.action.show_in_folder",
            "action": "show",
        },
        {
            "label": "asset_manager.action.settings",
            "action": "settings",
            "separator_before": True,
        },
    ]

def test_add_folder_starts_scan(panel_module, monkeypatch):
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    scans = []
    panel._asset_index = _index(
        add_folder=lambda _path: SimpleNamespace(id="selected-folder", path="/tmp/assets"),
        verify_projects=lambda: (0, 0),
    )
    monkeypatch.setattr(
        panel,
        "_scan_asset_folders",
        lambda folder_id=None, directory=None: scans.append((folder_id, directory)),
    )

    assert panel._add_folder_from_path("/tmp/assets") == "selected-folder"
    assert scans == [("selected-folder", "/tmp/assets")]

def test_catalog_notice_for_skipped_entries_and_clean_load(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(load_issues=["bad uuid", "duplicate path"])

    assert panel.get_catalog_notice() == "asset_manager.status.skipped_entries"
    assert panel.get_has_catalog_notice() is True

    panel._asset_index.load_issues = []
    assert panel.get_catalog_notice() == ""
    assert panel.get_has_catalog_notice() is False

def test_catalog_notice_for_failed_load_and_on_mount_warning(panel_module, monkeypatch):
    panel = panel_module.AssetManagerPanel()
    panel._catalog_load_failed = True
    assert panel.get_catalog_notice() == "asset_manager.status.load_failed"
    assert panel.get_has_catalog_notice() is True

    monkeypatch.setattr(
        panel,
        "_start_backend_initialization",
        lambda: setattr(panel, "_catalog_load_failed", True),
    )
    monkeypatch.setattr(panel, "_bind_dom_event_listeners", lambda _doc: None)
    monkeypatch.setattr(panel, "_scan_asset_folders", lambda **_kwargs: None)
    panel.on_mount(_Document())
    assert panel.get_has_catalog_notice()
    panel.on_unmount(_Document())

def test_backend_initialization_completes_without_ui_scheduler(panel_module, monkeypatch, tmp_path):
    loaded = threading.Event()

    class _LoadedIndex:
        load_issues = []

        def load(self):
            loaded.set()
            return True

    monkeypatch.setattr(panel_module, "AssetIndex", _LoadedIndex)
    monkeypatch.setattr(
        panel_module,
        "resolve_asset_manager_storage_path",
        lambda: tmp_path / "catalog",
    )
    monkeypatch.setattr(
        panel_module,
        "resolve_default_asset_directory",
        lambda: tmp_path / "assets",
    )
    monkeypatch.setattr(panel_module, "BACKEND_AVAILABLE", True)
    monkeypatch.delattr(panel_module.lf.ui, "schedule_on_ui_thread", raising=False)

    panel = panel_module.AssetManagerPanel()
    monkeypatch.setattr(panel, "_repair_selection", lambda: None)
    monkeypatch.setattr(panel, "_refresh_records", lambda **_kwargs: None)
    monkeypatch.setattr(panel, "_start_catalog_verify", lambda: None)
    monkeypatch.setattr(panel, "_scan_asset_folders", lambda: None)
    updates = []
    monkeypatch.setattr(panel, "_request_model_update", lambda: updates.append(True))

    panel._start_backend_initialization()

    assert loaded.wait(timeout=2.0)
    assert _wait_until(lambda: not panel._backend_load_active)
    assert isinstance(panel._asset_index, _LoadedIndex)
    assert panel._catalog_load_failed is False
    assert updates == [True]

def test_unmount_cancels_running_folder_scan(panel_module, monkeypatch):
    started = threading.Event()
    observed = {}

    def fake_scan(_index, cancel_event=None, progress=None):
        observed["event"] = cancel_event
        started.set()
        if cancel_event is not None:
            observed["waited"] = cancel_event.wait(timeout=2.0)
        return _scan_result()

    monkeypatch.setattr(panel_module, "scan_all_asset_folders", fake_scan)
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(verify_projects=lambda: (0, 0))
    panel._scan_asset_folders()
    assert started.wait(timeout=2.0)

    cancel = panel._folder_scan_cancel
    thread = panel._folder_scan_thread
    panel.on_unmount(_Document())

    assert cancel is not None and cancel.is_set()
    assert observed["event"] is cancel
    assert thread is not None
    thread.join(timeout=2.0)
    assert observed.get("waited") is True
    assert not thread.is_alive()

def test_unmount_does_not_join_scan_worker(panel_module):
    class NoJoin:
        ident = 1
        def join(self, *args, **kwargs):
            raise AssertionError("unmount must not join workers")

        def is_alive(self):
            return True

    panel = panel_module.AssetManagerPanel()
    panel._folder_scan_thread = NoJoin()
    panel._folder_scan_cancel = threading.Event()
    panel.on_unmount(_Document())

def test_refresh_during_scan_schedules_exactly_one_rerun(panel_module, monkeypatch):
    started = threading.Event()
    gate = threading.Event()
    calls = []

    def fake_scan(_index, cancel_event=None, progress=None):
        calls.append(1)
        started.set()
        gate.wait(timeout=2.0)
        return _scan_result()

    monkeypatch.setattr(panel_module, "scan_all_asset_folders", fake_scan)
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(verify_projects=lambda: (0, 0))
    panel._scan_asset_folders()
    assert started.wait(timeout=2.0)

    panel._scan_asset_folders()
    panel._scan_asset_folders()
    assert len(calls) == 1

    started.clear()
    gate.set()
    assert _wait_until(lambda: len(calls) == 2)
    assert len(calls) == 2
    assert _wait_until(lambda: not panel._folder_scan_active)
    panel.on_unmount(_Document())

def test_add_folder_scans_only_the_added_folder(panel_module, monkeypatch):
    started = threading.Event()
    one_calls = []
    all_calls = []

    def fake_one(_index, folder_id, directory, cancel_event=None, progress=None):
        one_calls.append((folder_id, directory))
        started.set()
        if cancel_event is not None:
            cancel_event.wait(timeout=2.0)
        return _scan_result()

    def fake_all(_index, cancel_event=None, progress=None):
        all_calls.append(1)
        return _scan_result()

    monkeypatch.setattr(panel_module, "scan_asset_folder", fake_one)
    monkeypatch.setattr(panel_module, "scan_all_asset_folders", fake_all)
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(
        add_folder=lambda path: SimpleNamespace(id="selected-folder", path=path),
        verify_projects=lambda: (0, 0),
    )
    panel._add_folder_from_path("/tmp/mrnf_local")
    assert started.wait(timeout=2.0)
    assert one_calls == [("selected-folder", "/tmp/mrnf_local")]
    assert all_calls == []
    panel.on_unmount(_Document())

def test_on_mount_scans_all_folders_only_before_first_completed_scan(
    panel_module, monkeypatch
):
    started = threading.Event()
    gate = threading.Event()
    calls = []

    def fake_all(_index, cancel_event=None, progress=None):
        calls.append("all")
        started.set()
        gate.wait(timeout=2.0)
        return _scan_result()

    monkeypatch.setattr(panel_module, "scan_all_asset_folders", fake_all)
    monkeypatch.setattr(
        panel_module,
        "scan_asset_folder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mount must scan all folders")
        ),
    )
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(verify_projects=lambda: (0, 0))
    monkeypatch.setattr(panel, "_bind_dom_event_listeners", lambda _doc: None)
    panel_module._folder_scan_completed_in_process = False

    panel.on_mount(_Document())
    assert started.wait(timeout=2.0)
    assert calls == ["all"]
    gate.set()
    assert _wait_until(lambda: not panel._folder_scan_active)
    assert panel_module._folder_scan_completed_in_process is True

    started.clear()
    panel.on_unmount(_Document())
    panel._panel_mounted = True
    panel.on_mount(_Document())
    assert calls == ["all"]
    assert not started.is_set()
    panel.on_unmount(_Document())

def test_refresh_during_scan_cancels_and_shows_stopped_status(panel_module, monkeypatch):
    started = threading.Event()
    calls = []

    def fake_scan(_index, cancel_event=None, progress=None):
        calls.append(1)
        started.set()
        if cancel_event is not None:
            cancel_event.wait(timeout=2.0)
        return _scan_result(cancelled=True)

    monkeypatch.setattr(panel_module, "scan_all_asset_folders", fake_scan)
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(verify_projects=lambda: (0, 0))
    panel._scan_asset_folders()
    assert started.wait(timeout=2.0)
    assert panel.get_scan_active() is True

    panel.refresh_catalog()
    assert panel._folder_scan_cancel is not None and panel._folder_scan_cancel.is_set()
    assert _wait_until(lambda: not panel._folder_scan_active)
    assert len(calls) == 1
    assert panel.get_scan_active() is False
    assert panel.get_scan_status() == "Scan stopped"
    assert panel.get_has_scan_status() is True
    assert panel.get_refresh_action_tooltip() == "asset_manager.tooltip.refresh"
    panel.on_unmount(_Document())

def test_scan_status_reads_worker_progress_counters(panel_module, monkeypatch):
    started = threading.Event()
    gate = threading.Event()

    def fake_scan(_index, cancel_event=None, progress=None):
        if progress is not None:
            progress.report(
                directories=12, projects=3, current_root="/tmp/mrnf_local"
            )
        started.set()
        gate.wait(timeout=2.0)
        return _scan_result()

    monkeypatch.setattr(panel_module, "scan_all_asset_folders", fake_scan)
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(verify_projects=lambda: (0, 0))
    panel._scan_asset_folders()
    assert started.wait(timeout=2.0)
    assert panel.get_scan_active() is True
    assert panel.get_scan_status() == (
        "Scanning mrnf_local: 12 folders, 3 projects found"
    )
    assert panel.get_has_scan_status() is True
    assert panel.get_refresh_action_tooltip() == "asset_manager.action.stop_scan"
    panel._published_scan_status = ""
    panel._published_scan_active = False
    assert panel.on_update(_Document()) is True
    assert "scan_status" in panel._handle.dirty_fields
    gate.set()
    assert _wait_until(lambda: not panel._folder_scan_active)
    panel.on_unmount(_Document())

def test_use_found_location_relinks_selected_asset(panel_module):
    asset = _project(
        exists=False,
        available=False,
        status="MISSING",
        relocation_candidate="/tmp/found.licht",
    )
    relinked = []
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(
        assets={asset["id"]: asset},
        relink_asset=lambda asset_id, path: relinked.append((asset_id, path)) or True,
        verify_projects=lambda: (0, 0),
    )
    panel._selected_asset_ids = {asset["id"]}
    panel._selection_cursor_id = asset["id"]
    panel._update_selection_type()
    panel.refresh_catalog = lambda **_kwargs: relinked.append("refresh")

    assert panel.get_selected_asset_has_relocation_candidate() is True
    assert panel.get_selected_asset_relocation_candidate() == "/tmp/found.licht"
    assert panel.get_selected_asset_has_folder() is True

    panel.on_use_found_location()

    assert relinked == [(asset["id"], "/tmp/found.licht"), "refresh"]

def test_context_menu_shows_use_found_location_only_with_candidate(panel_module):
    panel = panel_module.AssetManagerPanel()
    asset = _project()
    assert [item["action"] for item in panel._asset_context_menu_items(asset)] == [
        "load",
        "gallery:publish",
        "rename",
        "show_in_folder",
        "remove",
    ]

    asset["relocation_candidate"] = "/tmp/found.licht"
    assert "use_found_location" in [
        item["action"] for item in panel._asset_context_menu_items(asset)
    ]

def test_identity_mismatch_exposes_locate_and_relinks(panel_module):
    asset = _project(status="IDENTITY_MISMATCH", exists=True, available=False)
    relinked = []
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(
        assets={asset["id"]: asset},
        relink_asset=lambda asset_id, path: relinked.append((asset_id, path)) or True,
        verify_projects=lambda: (0, 0),
    )
    panel._selected_asset_ids = {asset["id"]}
    panel._selection_cursor_id = asset["id"]
    panel._update_selection_type()
    panel.refresh_catalog = lambda **_kwargs: None
    panel_module.lf._test_state.dialog_path = "/tmp/correct.licht"

    assert panel.get_selected_asset_can_locate() is True
    assert panel.get_selected_asset_file_missing() is False
    assert panel.get_locate_section_title() == "asset_manager.status.identity_mismatch"

    root = Path(__file__).resolve().parents[2]
    rml = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rml").read_text()
    assert 'data-if="selected_asset_can_locate"' in rml
    assert 'data-if="selected_asset_file_missing"' not in rml

    panel.on_locate_file()
    assert relinked == [(asset["id"], "/tmp/correct.licht")]

def test_repair_only_and_newer_version_status_labels(panel_module):
    panel = panel_module.AssetManagerPanel()
    assert panel._project_status_label({"status": "REPAIR_ONLY"}) == (
        "asset_manager.status.needs_repair"
    )
    assert panel._project_status_label({"status": "UNSUPPORTED_NEWER"}) == (
        "asset_manager.status.newer_version"
    )

def test_completed_save_registers_new_project_inside_project_location(panel_module):
    registered = []
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(
        find_asset_by_path=lambda _path: None,
        folder_id_for_path=lambda path: "default" if path.startswith("/tmp/projects") else None,
        register_licht_asset=lambda path: registered.append(path) or SimpleNamespace(id="new"),
        verify_projects=lambda: (0, 0),
    )
    polls = [
        {"running": False, "generation": 4, "path": "", "error": ""},
        {"running": False, "generation": 5, "path": "/tmp/projects/new.licht", "error": ""},
    ]
    panel_module.lf.project_poll_write = lambda: polls.pop(0)

    assert panel._refresh_after_project_write() is False
    assert panel._refresh_after_project_write() is True
    assert registered == ["/tmp/projects/new.licht"]

def test_completed_save_outside_folder_is_ignored(panel_module):
    registered = []
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(
        find_asset_by_path=lambda _path: None,
        folder_id_for_path=lambda _path: None,
        register_licht_asset=lambda path: registered.append(path),
        verify_projects=lambda: (0, 0),
    )
    polls = [
        {"running": False, "generation": 4, "path": "", "error": ""},
        {"running": False, "generation": 5, "path": "/tmp/outside/new.licht", "error": ""},
    ]
    panel_module.lf.project_poll_write = lambda: polls.pop(0)

    assert panel._refresh_after_project_write() is False
    assert panel._refresh_after_project_write() is False
    assert registered == []

def test_placeholder_label_uses_first_two_words(panel_module):
    panel = panel_module.AssetManagerPanel()
    row = panel._format_asset_for_ui(
        _project(name="Bicycle Scene Extra Words", has_preview=False)
    )
    assert row["placeholder_label"] == "Bicycle Scene"
    assert row["has_preview"] is False
    assert row["shows_placeholder"] is True

    truncated = panel._format_asset_for_ui(
        _project(name="Supercalifragilisticexpialidocious Wonderful", has_preview=False)
    )
    assert truncated["placeholder_label"] == "Supercalifragilisticexpi"
    assert len(truncated["placeholder_label"]) == 24

    root = Path(__file__).resolve().parents[2]
    rml = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rml").read_text()
    rcss = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rcss").read_text()
    assert 'data-if="asset.shows_placeholder"' in rml
    assert 'data-if="!asset.has_preview"' not in rml
    assert "{{asset.placeholder_label}}" in rml
    assert "display: block;" in rcss.split(".asset-card-placeholder {", 1)[1].split("}", 1)[0]
    assert "width: 100%;" in rcss.split(".asset-card-placeholder {", 1)[1].split("}", 1)[0]

def test_data_if_model_fields_are_boolean_bindings(panel_module):
    root = Path(__file__).resolve().parents[2]
    rml = (root / "src/visualizer/gui/rmlui/resources/asset_manager.rml").read_text()
    expressions = re.findall(r'\bdata-if="([^"]+)"', rml)
    assert expressions

    asset = _project(
        has_preview=False,
        relocation_candidate="/tmp/found.licht",
    )
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(assets={asset["id"]: asset})
    panel._selected_asset_ids = {asset["id"]}
    panel._selection_cursor_id = asset["id"]
    panel._update_selection_type()

    model = _BindingModel()
    panel.on_bind_model(_BindingContext(model))
    formatted = panel._format_asset_for_ui(asset)
    folders = panel.get_folder_list()
    assert folders

    plain = re.compile(r"^!?[A-Za-z_][A-Za-z0-9_]*$")
    dotted = re.compile(r"^!?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$")
    for expr in expressions:
        if plain.fullmatch(expr):
            name = expr[1:] if expr.startswith("!") else expr
            assert name in model.func_bindings, expr
            result = model.func_bindings[name]()
            assert isinstance(result, bool), (expr, type(result), result)
            continue
        match = dotted.fullmatch(expr)
        assert match, f"unsupported data-if expression: {expr}"
        scope, field = match.group(1), match.group(2)
        if scope == "asset":
            assert field in formatted, expr
            assert isinstance(formatted[field], bool), (expr, type(formatted[field]))
        elif scope == "folder":
            assert field in folders[0], expr
            assert isinstance(folders[0][field], bool), (expr, type(folders[0][field]))
        else:
            raise AssertionError(f"unsupported data-if scope: {expr}")

def test_on_mount_shows_cached_rows_without_inspecting(
    panel_module, monkeypatch, tmp_path
):
    from lfs_plugins.asset_index import AssetIndex

    inspect_calls = []
    projects = {}
    inspections = {}
    for index in range(34):
        project_uuid = str(uuid.uuid4())
        path = tmp_path / f"{project_uuid}.licht"
        path.write_bytes(b"container")
        projects[project_uuid] = {
            "name": f"P{index:02d}",
            "path": str(path),
            "folder_id": "default",
        }
        inspections[path.name] = SimpleNamespace(
            project_uuid=project_uuid,
            file_uuid=str(uuid.uuid4()),
            commit_uuid=str(uuid.uuid4()),
            generation=1,
            created_at_unix_ns=100,
            saved_at_unix_ns=200,
            physical_file_size=1234,
            role=SimpleNamespace(name="MASTER"),
            open_state=SimpleNamespace(name="OPEN"),
            has_preview=False,
            fallback_preview_path="",
        )

    def inspect(path):
        inspect_calls.append(path)
        time.sleep(0.002)
        return inspections[Path(path).name]

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))
    verify_done = threading.Event()
    original_verify_catalog = panel_module.verify_catalog_projects

    def verify_catalog(*args, **kwargs):
        try:
            return original_verify_catalog(*args, **kwargs)
        finally:
            verify_done.set()

    monkeypatch.setattr(panel_module, "verify_catalog_projects", verify_catalog)
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "folders": {"default": {"path": str(tmp_path)}},
                "projects": projects,
            }
        ),
        encoding="utf-8",
    )
    index = AssetIndex(library_path=library_path, default_folder_path=tmp_path)
    loaded_at = time.perf_counter()
    assert index.load() is True
    load_ms = (time.perf_counter() - loaded_at) * 1000
    assert inspect_calls == []
    assert {project.status for project in index.list_projects()} == {"UNVERIFIED"}

    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = index
    panel_module._folder_scan_completed_in_process = True
    monkeypatch.setattr(panel, "_bind_dom_event_listeners", lambda _doc: None)
    monkeypatch.setattr(panel, "_scan_asset_folders", lambda **_kwargs: None)
    refresh_at = {}
    original_refresh = panel._refresh_records

    def timed_refresh(*, assets=False, folders=False):
        refresh_at["inspects"] = len(inspect_calls)
        refresh_at["time"] = time.perf_counter()
        return original_refresh(assets=assets, folders=folders)

    panel._refresh_records = timed_refresh
    ui_callbacks = []
    monkeypatch.setattr(panel_module.lf.ui, "schedule_on_ui_thread", ui_callbacks.append)
    started = time.perf_counter()
    panel.on_mount(_Document())
    mount_to_refresh_ms = (refresh_at["time"] - started) * 1000

    assert refresh_at["inspects"] == 0
    assert panel._last_asset_match_count == 34
    assert {row["status"] for row in panel._handle.records["assets"]} == {"UNVERIFIED"}
    assert mount_to_refresh_ms < 100.0
    panel._mount_timing = {
        "load_ms": load_ms,
        "mount_to_refresh_ms": mount_to_refresh_ms,
    }
    assert verify_done.wait(timeout=2.0)
    assert {project.status for project in index.list_projects()} == {"AVAILABLE"}
    if panel._catalog_verify_thread:
        panel._catalog_verify_thread.join(timeout=2.0)
    for callback in ui_callbacks:
        callback()
    panel.on_update(_Document())
    assert {row["status"] for row in panel._handle.records["assets"]} <= {
        "AVAILABLE",
        "UNVERIFIED",
    }
    panel.on_unmount(_Document())

def test_on_update_publishes_new_catalog_rows(panel_module):
    first = _project()
    second = _project(
        "44444444-4444-4444-8444-444444444444",
        name="Garden",
        path="/tmp/garden.licht",
    )
    epoch = {"value": 1}
    assets = {first["id"]: first}
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    panel._asset_index = _index(
        assets=assets,
        catalog_epoch=lambda: epoch["value"],
    )
    panel._catalog_epoch_seen = 1
    panel._refresh_records(assets=True, folders=True)
    assert {row["id"] for row in panel._handle.records["assets"]} == {first["id"]}

    assets[second["id"]] = second
    epoch["value"] = 2
    assert panel.on_update(_Document()) is True
    assert {row["id"] for row in panel._handle.records["assets"]} == {
        first["id"],
        second["id"],
    }

def _gallery_fixture(panel_module):
    from lfs_plugins.gallery_sync import shared_fields
    panel = panel_module.AssetManagerPanel()
    local = _project(commit_uuid='saved')
    remote = dict(id='scene',title='Published project',description='',visibility='private',revision='r1',status='ready',sourceFormat='licht',contentLength=42,viewerSettings={}, contentRevision='r1', metadataRevision='r1')
    panel._asset_index = _index(assets={local['id']:local})
    panel._gallery_state = dict(identity='account',signed_in=True,connected=True,checkedAt=1,
        links={local['id']:dict(sceneId='scene',revision='r1',commitUuid='saved',sharedFields=shared_fields(remote), contentRevision='r1', metadataRevision='r1')},
        scenes=[remote,dict(remote,id='remote-only',title='Remote only')],jobs=[])
    return panel,local,remote

def test_gallery_union_has_one_linked_pair_and_remote_projection(panel_module):
    panel, local, remote = _gallery_fixture(panel_module)
    panel.select_gallery_scope()
    rows = panel._filtered_assets()
    assert {r['id'] for r in rows} == {local['id'],'remote:remote-only'}
    assert panel.get_all_assets_count() == 1
    assert set(panel._asset_index.assets) == {local['id']}
    assert panel._select_asset_id('remote:remote-only')
    panel._repair_selection()
    assert panel.get_selected_asset_id() == 'remote:remote-only'
    badge = panel._gallery_badge(panel._asset_dict('remote:remote-only'))
    assert badge['gallery_state'] == 'remote_only' and badge['gallery_action'] == 'pull'
    panel._select_folder_id('__all__')
    assert [r['id'] for r in panel._filtered_assets()] == [local['id']]

def test_gallery_attention_scope_and_state_specific_context_menu(panel_module):
    panel, local, remote = _gallery_fixture(panel_module)
    assert panel._gallery_facts(local)['state'] == 'equal'
    local['commit_uuid'] = 'new'
    remote['title'] = 'Portal edit'
    remote['metadataRevision'] = 'metadata-edited'
    assert panel._gallery_facts(local)['state'] == 'diverged'
    panel._select_folder_id('__gallery_attention__')
    assert [r['id'] for r in panel._filtered_assets()] == [local['id']]
    actions = [i['action'] for i in panel._asset_context_menu_items(local)]
    assert actions[0:2] == ['load','gallery:resolve']
    assert 'gallery:update' not in actions and 'gallery:publish' not in actions
    remote_actions=[i['action'] for i in panel._asset_context_menu_items(panel._asset_dict('remote:remote-only'))]
    assert remote_actions == ['gallery:pull','gallery:pull_open','gallery:open','gallery:copy','gallery:remove']

def test_multi_selection_publish_and_update_are_disjoint(panel_module):
    panel,local,remote = _gallery_fixture(panel_module)
    ready=_project(id='ready',project_uuid='ready')
    missing=_project(id='missing',project_uuid='missing',exists=False,available=False)
    panel._asset_index.assets.update(ready=ready,missing=missing)
    panel._selected_asset_ids={local['id'],'ready','missing'}
    assert panel._gallery_counts()==dict(count=3,ready=1,linked=1,missing=1)

@pytest.mark.parametrize('elapsed,key,count', [(0,'just_now',None),(59,'just_now',None),(60,'minutes',1),
    (3599,'minutes',59),(3600,'hours',1),(86399,'hours',23),(86400,'yesterday',None),(172800,'days',2)])
def test_gallery_relative_time_shared_boundaries(panel_module, monkeypatch, elapsed, key, count):
    from lfs_plugins import asset_gallery_ui as ui
    monkeypatch.setattr(ui, 'tr', lambda key, **values: (key, values))
    assert ui.relative_time(1000, now=1000+elapsed) == ('time.'+key, {} if count is None else {'count': count})

def test_signed_out_gallery_never_claims_offline_or_checked(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel._gallery_state = {'signed_in': False, 'offline': True, 'checkedAt': 123}
    assert panel._gallery_checked_label().endswith('sidebar.sign_in_hint')
    panel._gallery_state['signed_in'] = True
    assert panel._gallery_checked_label().endswith('sidebar.offline')

def test_sidebar_restores_old_heights_with_room_for_gallery(panel_module):
    panel = panel_module.AssetManagerPanel()
    assert panel._sidebar_height == 280
    panel.apply_chrome({'sidebar_height': 176})
    assert panel._sidebar_height == 280

def test_remote_double_click_routes_to_explicit_pull_open(panel_module, monkeypatch):
    panel, local, remote = _gallery_fixture(panel_module)
    actions = []
    monkeypatch.setattr(panel, '_gallery_command', lambda action: actions.append(action))
    panel._load_asset('remote:remote-only')
    assert actions == ['pull_open']
    assert panel.get_selected_asset_id() == 'remote:remote-only'

def test_pull_completion_reloads_catalog_and_selects_new_card(panel_module, monkeypatch):
    panel, local, remote = _gallery_fixture(panel_module)
    panel._gallery_state['identity'] = 'account'
    new = _project(id='pulled', project_uuid='pulled')
    def load():
        panel._asset_index.assets['pulled'] = new
        return True
    panel._asset_index.load = load
    panel._gallery_changed(dict(panel._gallery_state, pulledProject={'id':'pulled','jobId':'job','path':new['path']},
        links={**panel._gallery_state['links'], 'pulled':{'sceneId':'remote-only'}}))
    assert panel.get_selected_asset_id() == 'pulled'
    assert panel._selected_folder_id == '__gallery__'
    assert panel._gallery_notice.endswith('info.pulled')

def test_removed_scene_keeps_unlink_action(panel_module):
    panel, local, remote = _gallery_fixture(panel_module)
    panel._select_asset_id(local['id'])
    panel._gallery_state['scenes'] = []
    panel._gallery_state['links'][local['id']]['remoteDeleted'] = True
    assert panel._has_gallery_link()
    assert panel._gallery_scene(local) is None
    actions = panel._gallery_context_items(local)
    assert any(item['action'] == 'gallery:unlink' for item in actions)

def test_D1_undo_window_rearms_after_worker_failure(panel_module, monkeypatch):
    panel, local, remote = _gallery_fixture(panel_module)
    clock = [100.0]
    from lfs_plugins import asset_gallery_ui as ui
    monkeypatch.setattr(ui.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(ui.threading, 'Timer', lambda *args: SimpleNamespace(start=lambda:None, cancel=lambda:None))
    calls = []
    panel._gallery_controller = SimpleNamespace(undo_pull=lambda: calls.append('restore'))
    state = dict(panel._gallery_state, undoPull={'backup':'/backup', 'attempt':0})
    panel._gallery_changed(state)
    panel._gallery_command('undo')
    assert panel._gallery_undo and calls == ['restore']
    clock[0] = 120.0
    panel._gallery_changed(dict(state, undoPull={'backup':'/backup', 'attempt':1, 'error':'localized reason'}))
    assert panel._gallery_undo[0] == 128.0
    assert panel._gallery_notice == 'localized reason'
    panel._gallery_changed(dict(state, undoPull={'backup':'/backup', 'attempt':2, 'backupMissing':True, 'error':'gone'}))
    assert panel._gallery_undo is None and panel._gallery_notice == 'gone'

# Moved from test_gallery_controller: the Asset Manager owns the editor/open flow.
@pytest.mark.parametrize('height', [600, 720, 1000])
@pytest.mark.parametrize('folder_count', [2, 40])
@pytest.mark.parametrize('restored_info', [220, 1000])
def test_A4_small_panel_reserves_results_and_all_gallery_rows(panel_module, monkeypatch, height, folder_count, restored_info):
    from lfs_plugins.asset_layout import panel_layout
    panel = panel_module.AssetManagerPanel()
    scale = 1.5
    monkeypatch.setattr(panel_module.lf.ui, 'get_ui_scale', lambda: scale, raising=False)
    monkeypatch.setattr(panel, 'get_folder_list', lambda: [{}] * folder_count)
    content = 16 + 77 + 34 * folder_count + 140
    panel._doc = _Document({
        'asset-popup': SimpleNamespace(client_height=height * scale),
        'asset-popup-toolbar': SimpleNamespace(client_height=114 * scale),
        'asset-results-header': SimpleNamespace(client_height=48 * scale),
        'asset-sidebar-local-content': SimpleNamespace(scroll_height=(77 + 34 * folder_count) * scale),
        'asset-sidebar-gallery': SimpleNamespace(client_height=132 * scale),
    })
    panel.apply_chrome({'sidebar_height': 1000, 'bottom_panel_height': restored_info, 'folders_collapsed': False})
    layout = panel_layout(height, folder_count=folder_count, info_height=min(500, restored_info))
    assert panel._sidebar_height == layout['sidebar'] == min(content, height * .4)
    assert panel._bottom_panel_height == layout['info']
    # Subtract the actual bound chrome; this must be usable results height, not
    # an artificial minimum on an overflowing/clipped child.
    results = height - 115 - 49 - 20 - panel._sidebar_height - panel._bottom_panel_height
    assert results >= 160
    assert panel._main_min_height + panel._bottom_panel_height + 115 + 10 <= height
    # Gallery's fixed 140 dp block is below the shrinking local scroll area.
    gallery_top = panel._sidebar_height - 8 - 140
    for row in range(3):
        top = gallery_top + 34 + 20 + row * 26
        assert top >= 8 and top + 24 <= panel._sidebar_height - 8
    assert not panel._sync_panel_layout()  # Stable geometry does not dirty every tick.

def test_A4_short_panel_starts_folders_collapsed_and_keeps_info_preference(panel_module):
    panel = panel_module.AssetManagerPanel()
    popup = SimpleNamespace(client_height=600)
    panel._doc = _Document({'asset-popup': popup})
    panel._sync_panel_layout()
    assert panel._folders_collapsed
    assert panel._bottom_panel_height < 220
    assert panel.capture_chrome()['bottom_panel_height'] == 220
    popup.client_height = 1000
    panel._sync_panel_layout()
    assert panel._bottom_panel_height == 220

def test_A4_gallery_scopes_are_outside_the_scrolling_folder_content():
    import xml.etree.ElementTree as ET
    resources = Path(__file__).resolve().parents[2] / 'src/visualizer/gui/rmlui/resources'
    root = ET.fromstring((resources / 'asset_manager.rml').read_text())
    local = root.find('.//*[@id="asset-sidebar-local-scroll"]')
    gallery = root.find('.//*[@id="asset-sidebar-gallery"]')
    assert gallery not in list(local.iter())
    assert {e.get('data-folder-id') for e in gallery.iter() if e.get('data-folder-id')} == {
        '__gallery__', '__gallery_attention__', '__gallery_transfers__'}
    rcss = (resources / 'asset_manager.rcss').read_text()
    assert '#asset-sidebar-local-scroll { min-height: 0; overflow-y: auto;' in rcss
    assert '#asset-sidebar-gallery { flex-shrink: 1; min-height: 140dp; max-height: 100%; overflow-y: auto; }' in rcss

@pytest.mark.parametrize('size,expected', [(0, '0.0 B'), (9, '9.0 B'), (10, '10 B'), (1024, '1.0 KB'),
    (137114, '134 KB'), (10 * 1024, '10 KB'), (1024**2, '1.0 MB'), (42 * 1024**2, '42 MB'), (1024**3, '1.0 GB')])
def test_A4_adaptive_sizes_match_tray_cards_and_info(panel_module, monkeypatch, size, expected):
    from lfs_plugins.asset_format import format_size
    from lfs_plugins.gallery_transfer_panel import transfer_rows
    locale = json.loads((Path(__file__).resolve().parents[2] / 'src/visualizer/gui/resources/locales/en.json').read_text())
    # Use the real localized templates/units instead of checking untranslated keys.
    flattened = dict(locale)
    flattened.update({'asset_manager.' + key: value for key, value in locale['asset_manager'].items()})
    monkeypatch.setattr(panel_module.lf.ui, 'tr', lambda key: flattened.get(key, key))
    assert format_size(size) == expected
    panel, local, remote = _gallery_fixture(panel_module)
    local['file_size_bytes'] = size
    remote['contentLength'] = size
    panel._select_asset_id(local['id'])
    assert panel._format_asset_for_ui(local)['size_label'] == expected
    assert expected in panel._gallery_published_summary()
    job = dict(id='job', status='running', completed=size, total=size)
    assert transfer_rows({'jobs':[job]})[0]['bytes'] == f'{expected} / {expected}'

@pytest.mark.parametrize('width,modified', [(320, False), (380, True), (420, True)])
@pytest.mark.parametrize('scale', [1.0, 1.5])
def test_A4_list_gallery_header_fits_before_modified(panel_module, monkeypatch, width, modified, scale):
    from lfs_plugins.asset_layout import list_columns
    panel = panel_module.AssetManagerPanel()
    panel._asset_window_client_width = width * scale
    monkeypatch.setattr(panel_module.lf.ui, 'get_ui_scale', lambda: scale, raising=False)
    model = _BindingModel()
    panel.on_bind_model(_BindingContext(model))
    assert model.func_bindings['asset_list_wide']() == modified
    assert not model.func_bindings['asset_list_show_folder']()
    assert model.func_bindings['col_gallery_label']().endswith('gallery.sidebar.title')
    columns = list_columns(width)
    assert columns['name'] >= 64 and columns['gallery'] == 96
    resources = Path(__file__).resolve().parents[2] / 'src/visualizer/gui/rmlui/resources'
    assert '.asset-col-gallery { width: 96dp; min-width: 96dp; flex-shrink: 0; }' in (resources / 'asset_manager.rcss').read_text()

def test_portal_posters_obey_scope_and_release_on_scroll(panel_module, tmp_path):
    panel, local, remote = _gallery_fixture(panel_module)
    poster = _write_png(tmp_path / "portal poster.png")
    panel._gallery_state["posters"] = {"scene": str(poster), "remote-only": str(poster)}
    panel._selected_folder_id = "__all__"
    assert "kind=licht" in panel._format_asset_for_ui(local)["thumbnail_decorator"]
    panel._selected_folder_id = panel_module.SCOPE_PUBLISHED
    assert "kind=image" in panel._format_asset_for_ui(local)["thumbnail_decorator"]
    row = panel._gallery_remote_assets()["remote:remote-only"]
    card = panel._format_asset_for_ui(row)
    assert "kind=image" in card["thumbnail_decorator"] and not card["shows_placeholder"]
    source = panel._thumbnail_sources_by_asset[row["id"]]
    panel._release_obsolete_thumbnail_sources()
    assert panel._thumbnail_sources_by_asset[row["id"]] == source
    panel._window_assets = lambda assets: []
    panel._release_thumbnails_outside_window()
    assert source in panel_module.lf._test_state.released_textures
    assert panel._thumbnail_sources_by_asset == {}
    panel._gallery_state["posters"] = {}
    assert panel._format_asset_for_ui(row)["thumbnail_decorator"] == "none"

@pytest.mark.parametrize("domain,expected", [("presentationRevision", "equal"), ("contentRevision", "remote"), ("metadataRevision", "remote")])
def test_freshness_uses_domain_tokens_not_presentation(panel_module, domain, expected):
    from lfs_plugins.gallery_controller import asset_sync_state
    panel, local, remote = _gallery_fixture(panel_module)
    link = panel._gallery_state["links"][local["id"]]
    link.update(contentRevision="c", metadataRevision="m")
    remote.update(contentRevision="c", metadataRevision="m", presentationRevision="p")
    remote[domain] = "changed"
    remote["revision"] = "broad-change"
    assert asset_sync_state(local, link, remote)["freshness"] == expected

def test_update_review_explains_cover_preservation(panel_module):
    panel, local, remote = _gallery_fixture(panel_module)
    panel._select_asset_id(local["id"])
    text = panel._gallery_review_includes()
    assert "asset_manager.gallery.review.cover_kept" in text

@pytest.mark.parametrize('visibility', ['private', 'public'])
def test_open_in_portal_uses_the_scene_login_destination(panel_module, visibility):
    from lfs_plugins.gallery_controller import GalleryController
    urls = []
    panel_module.lf.ui.open_url = urls.append
    controller = SimpleNamespace(service=SimpleNamespace(account=SimpleNamespace(base_url='https://portal.example')))
    scene = {'id': str(uuid.uuid4()), 'visibility': visibility}
    GalleryController.open_portal(controller, scene)
    assert urls == [f'https://portal.example/gallery/scenes/{scene["id"]}/open/']

def test_info_poster_is_inserted_updated_and_released(panel_module, tmp_path):
    panel, local, remote = _gallery_fixture(panel_module)
    poster = _write_png(tmp_path / "poster.png")
    panel._gallery_state["posters"] = {"remote-only": str(poster)}
    panel._selected_folder_id = panel_module.SCOPE_PUBLISHED
    panel._select_asset_id("remote:remote-only")
    properties, elements = {}, {}
    class Element:
        def set_id(self, value):
            elements[value] = self
        def set_property(self, key, value):
            properties[key] = value
    header = SimpleNamespace(parent=lambda: SimpleNamespace(insert_before=lambda *args: Element()))
    doc = SimpleNamespace(query_selector=lambda selector: header, get_element_by_id=elements.get)
    assert panel._sync_info_thumbnail(doc)
    assert properties["display"] == "block" and "kind=image" in properties["decorator"]
    source = panel._info_thumbnail_source
    panel._selected_asset_ids.clear()
    assert panel._sync_info_thumbnail(doc)
    assert properties["display"] == "none"
    assert source in panel_module.lf._test_state.released_textures

def test_translated_message_has_no_english_append(panel_module):
    from lfs_plugins.gallery_messages import localize_message
    panel_module.lf.ui.tr = lambda key: "Téléversement terminé."
    assert localize_message("Upload complete.") == "Téléversement terminé."
