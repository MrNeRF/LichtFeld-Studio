# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the `.licht`-only Asset Manager panel."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.parse import quote
import sys

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
        opened=[],
        revealed=[],
        enabled=[],
        dialog_path="",
    )

    def show_context_menu(items, x, y, on_action=None):
        context_menus.append(
            {"items": items, "position": (x, y), "on_action": on_action}
        )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        Panel=_Panel,
        PanelSpace=SimpleNamespace(
            FLOATING="FLOATING",
            LEFT_DOCK="LEFT_DOCK",
        ),
        PanelHeightMode=SimpleNamespace(FILL="FILL", CONTENT="CONTENT"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        tr=lambda key: key,
        get_current_language=lambda: "en",
        get_mouse_screen_pos=lambda: (120.0, 220.0),
        show_context_menu=show_context_menu,
        reveal_in_file_manager=lambda path: state.revealed.append(path) or True,
        open_project_file_dialog=lambda _start="": state.dialog_path,
        input_dialog=lambda *_args: None,
        set_panel_enabled=lambda panel_id, enabled: state.enabled.append(
            (panel_id, enabled)
        ),
        schedule_on_ui_thread=lambda callback: callback(),
    )
    lf_stub.log = SimpleNamespace(info=lambda _msg: None, warn=lambda _msg: None, error=lambda _msg: None)
    lf_stub.project_open = lambda path, discard_changes=False: state.opened.append(
        (path, discard_changes)
    )
    lf_stub._test_state = state
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)


@pytest.fixture
def panel_module(monkeypatch):
    source_python = Path(__file__).resolve().parents[2] / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    for name in ("lfs_plugins.asset_manager_panel", "lfs_plugins"):
        sys.modules.pop(name, None)
    _install_lf_stub(monkeypatch)
    return import_module("lfs_plugins.asset_manager_panel")


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
        "exists": True,
        "available": True,
        "status": "AVAILABLE",
        "error": "",
    }
    value.update(overrides)
    return value


def _index(assets=None, folders=None, **methods):
    return SimpleNamespace(
        assets=assets or {},
        folders=folders
        or {"default": {"id": "default", "name": "Default", "watch_directories": []}},
        **methods,
    )


def test_panel_contract_remains_dirty_left_dock(panel_module):
    panel_type = panel_module.AssetManagerPanel
    assert panel_type.update_policy == "dirty"
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
    assert "thumbnails" not in source
    assert "LICHT" not in rml
    assert "asset-col-type" not in rml
    assert "asset-pill-licht" not in rcss
    assert "asset-card-overlay" not in rcss
    assert "col_type_label" not in source
    assert 'data-style-decorator="asset.thumbnail_decorator"' in rml


def test_embedded_preview_url_encodes_path_and_keys_cache_by_commit(panel_module):
    panel = panel_module.AssetManagerPanel()
    asset = _project(path="/tmp/a folder/project & one.licht")

    decorator = panel._thumbnail_decorator(asset)

    encoded = quote(asset["path"], safe="/:._-~")
    assert decorator == (
        "image(preview://kind=licht&thumb=256"
        f"&rev={asset['commit_uuid']}&path={encoded})"
    )
    assert asset["path"] not in decorator
    assert panel._thumbnail_decorator({**asset, "has_preview": False}) == "none"


def test_asset_rows_use_custom_name_and_runtime_metadata(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(assets={_project()["id"]: _project()})
    panel._selected_folder_id = "default"

    row = panel.get_filtered_assets()[0]

    assert row["display_name"] == "Bicycle"
    assert row["display_subtitle"] == "bicycle project.licht"
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
    assert "selected_asset_path" in panel._handle.dirty_fields
    assert "assets" not in panel._handle.records


def test_dom_right_click_uses_shared_app_context_menu(panel_module):
    panel = panel_module.AssetManagerPanel()
    panel._handle = _Handle()
    asset = _project()
    panel._asset_index = _index(
        assets={asset["id"]: asset},
        folders={
            "default": {"id": "default", "name": "Default", "watch_directories": []},
            "archive": {"id": "archive", "name": "Archive", "watch_directories": []},
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
        "rename",
        "",
        "create_folder",
        "move_to_folder:archive",
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


def test_folder_menu_keeps_watched_directory_action(panel_module, monkeypatch):
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index(
        folders={"projects": {"id": "projects", "name": "Projects", "watch_directories": []}}
    )
    calls = []
    monkeypatch.setattr(
        panel,
        "on_edit_watch_dirs",
        lambda _handle, _event, args: calls.append(tuple(args)),
    )

    assert panel._show_folder_context_menu("projects") is True
    menu = panel_module.lf._test_state.context_menus[-1]
    assert [item["action"] for item in menu["items"]] == [
        "watch_dirs",
        "rename",
        "delete",
    ]
    menu["on_action"]("watch_dirs")
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

    assert panel_module.lf._test_state.opened == [(asset["path"], False)]
    assert panel.get_selected_asset_id() == asset["id"]


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

    assert calls == [(asset["path"], "default")]
    assert panel.get_selected_asset_id() == asset["id"]


def test_edit_watch_directories_targets_clicked_folder(panel_module, monkeypatch):
    panel = panel_module.AssetManagerPanel()
    panel._asset_index = _index()
    calls = []
    monkeypatch.setattr(
        panel_module,
        "open_watch_dirs_dialog",
        lambda index, folder_id, callback: calls.append((index, folder_id, callback)) or True,
    )
    event = _Event()

    panel.on_edit_watch_dirs(None, event, ["default"])

    assert calls[0][0] is panel._asset_index
    assert calls[0][1] == "default"
    assert event.stopped is True


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

    assert {"mousedown", "click", "dblclick"}.issubset(shell.listeners)
    assert {"scroll", "mousescroll"}.issubset(scroll.listeners)
    assert {"mousemove", "mouseup"}.issubset(doc.listeners)


def test_default_folder_cannot_be_renamed_or_deleted_from_menu(panel_module):
    panel = panel_module.AssetManagerPanel()
    assert panel._folder_context_menu_items("default") == [
        {
            "label": "asset_manager.action.edit_watch_dirs",
            "action": "watch_dirs",
        }
    ]
