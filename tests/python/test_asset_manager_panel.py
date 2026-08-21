# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for Asset Manager panel record formatting and selection."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import sys

import pytest


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
        PanelSpace=SimpleNamespace(
            FLOATING="FLOATING",
            BOTTOM_DOCK="BOTTOM_DOCK",
            LEFT_DOCK="LEFT_DOCK",
        ),
        PanelHeightMode=SimpleNamespace(FILL="FILL", CONTENT="CONTENT"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
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
        self._classes = set(str(self._attrs.get("class", "")).split())
        self._children = []
        self.scroll_height = 0.0
        self.client_height = 0.0
        self.scroll_top = 0.0
        if parent is not None and hasattr(parent, "_children"):
            parent._children.append(self)

    def get_attribute(self, name, default=""):
        return self._attrs.get(name, default)

    def has_attribute(self, name):
        return name in self._attrs

    def parent(self):
        return self._parent

    def query_selector_all(self, selector):
        result = []
        selectors = [part.strip() for part in str(selector).split(",") if part.strip()]

        def _matches(element, item):
            if not item.startswith("."):
                return False
            return item[1:] in element._classes

        def _visit(element):
            for child in element._children:
                if any(_matches(child, item) for item in selectors):
                    result.append(child)
                _visit(child)

        _visit(self)
        return result

    def set_class(self, name, active):
        if active:
            self._classes.add(name)
        else:
            self._classes.discard(name)

    def is_class_set(self, name):
        return name in self._classes

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
        "name": "bicycle-project",
        "absolute_path": "/tmp/bicycle.licht",
        "path": "/tmp/bicycle.licht",
        "file_size_bytes": 4206437268,
        "exists": True,
        "verification_disposition": "MATCH_FAST_PATH",
        "folder_id": "p1",
        "scene_id": "s1",
        "created_at": "2026-02-15T21:52:45.881056",
        "modified_at": "2026-04-28T14:48:57.606369",
    }


def test_asset_manager_uses_dirty_update_policy(asset_manager_panel_module):
    assert asset_manager_panel_module.AssetManagerPanel.update_policy == "dirty"
    assert "update_interval_ms" not in asset_manager_panel_module.AssetManagerPanel.__dict__


def test_asset_manager_remains_left_dock_panel(asset_manager_panel_module):
    assert (
        asset_manager_panel_module.AssetManagerPanel.space
        == asset_manager_panel_module.lf.ui.PanelSpace.LEFT_DOCK
    )
    assert asset_manager_panel_module.AssetManagerPanel.order == 20


def test_builtin_registration_keeps_asset_manager_closed_by_default():
    panels_source = (
        Path(__file__).resolve().parents[2] / "src" / "python" / "lfs_plugins" / "panels.py"
    ).read_text(encoding="utf-8")

    assert 'set_panel_enabled("lfs.asset_manager", False)' in panels_source


def test_asset_manager_requests_update_from_reactive_store(asset_manager_panel_module, monkeypatch):
    module = asset_manager_panel_module
    signals = SimpleNamespace(
        language_generation=_SignalStub(),
    )
    monkeypatch.setattr(module, "RuntimeState", signals)

    panel = module.AssetManagerPanel()
    panel._handle = _HandleStub()

    panel._subscribe_reactive_state()
    signals.language_generation.emit(1)

    assert panel._handle.dirty_fields == ["__update__"]

    panel._unsubscribe_reactive_state()
    signals.language_generation.emit(2)

    assert panel._handle.dirty_fields == ["__update__"]


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
    assert 'data-event-click="on_import_project"' in rml
    assert "asset-thumb-project" in rml
    assert "LICHT" in rml


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


def test_asset_manager_has_visible_viewport_edge():
    folder_root = Path(__file__).parent.parent.parent
    resources_dir = (
        folder_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
    )
    rcss = (resources_dir / "asset_manager.rcss").read_text(encoding="utf-8")
    rml = (resources_dir / "asset_manager.rml").read_text(encoding="utf-8")
    theme_rcss = (resources_dir / "asset_manager.theme.rcss").read_text(encoding="utf-8")

    assert 'id="asset-viewport-edge"' in rml
    assert "position: relative;" in rcss
    assert "#asset-viewport-edge" in rcss
    assert "position: absolute;" in rcss
    assert "right: 0;" in rcss
    assert "width: 1dp;" in rcss
    assert "background-color: rgba(88, 91, 112, 153);" in rcss
    assert "#asset-shell.is-floating #asset-viewport-edge" in rcss
    assert "background-color: @{right_panel.border};" in theme_rcss


def test_asset_manager_project_actions_are_localized():
    folder_root = Path(__file__).parent.parent.parent
    locale_dir = folder_root / "src" / "visualizer" / "gui" / "resources" / "locales"
    required_keys = (
        "action.refresh",
        "action.clean_missing",
        "tooltip.refresh",
        "tooltip.clean_missing",
        "panel_title",
        "property.assets",
        "status.content_changed",
        "status.type_changed",
        "status.unverified",
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
        folders={"p1": {"id": "p1", "name": "Projects", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
    )

    panel.toggle_asset_selection(None, None, ["a1"])

    assert panel.get_selected_asset_name() == "bicycle-project"
    assert panel.get_selected_asset_path() == "/tmp/bicycle.licht"
    dirty = panel._handle.dirty_fields
    assert "selected_asset_path" in dirty or "__all__" in dirty
    assert "show_selection_asset" in dirty or "__all__" in dirty


def test_asset_selection_resolves_asset_id_from_clicked_element(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        folders={"p1": {"id": "p1", "name": "Projects", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
    )

    event = _EventStub(current_target=_ElementStub({"data-asset-id": "a1"}))
    panel.toggle_asset_selection(None, event, [])

    assert panel.get_selected_asset_name() == "bicycle-project"
    assert panel.get_selected_count() == 1



def test_dom_card_click_selects_asset_from_stable_parent(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        folders={"p1": {"id": "p1", "name": "Projects", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
    )

    container = _ElementStub({"id": "asset-main-row"})
    card = _ElementStub(
        {"data-asset-id": "a1", "data-asset-action": "select"},
        parent=container,
    )
    child = _ElementStub(parent=card)
    event = _EventStub(current_target=container, target=child)

    panel._on_asset_manager_click(event)

    assert panel.get_selected_asset_name() == "bicycle-project"
    assert panel.get_selected_count() == 1
    assert event.stopped is True


def test_dom_card_click_updates_visible_row_class(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    first = _make_asset()
    second = _make_asset()
    second["id"] = "a2"
    second["name"] = "garden"
    panel._asset_index = SimpleNamespace(
        assets={"a1": first, "a2": second},
        folders={"p1": {"id": "p1", "name": "Projects", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
    )

    container = _ElementStub({"id": "asset-popup-content"})
    card_a1 = _ElementStub(
        {
            "class": "asset-list-row",
            "data-asset-id": "a1",
            "data-asset-action": "select",
        },
        parent=container,
    )
    card_a2 = _ElementStub(
        {
            "class": "asset-list-row",
            "data-asset-id": "a2",
            "data-asset-action": "select",
        },
        parent=container,
    )

    panel._on_asset_manager_click(_EventStub(current_target=container, target=card_a1))
    panel._on_asset_manager_click(_EventStub(current_target=container, target=card_a2))

    assert card_a1.is_class_set("is-selected") is False
    assert card_a2.is_class_set("is-selected") is True
    assert "assets" not in panel._handle.records


def test_dom_card_ctrl_click_adds_to_multi_selection(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    first = _make_asset()
    second = _make_asset()
    second["id"] = "a2"
    second["name"] = "garden"
    second["absolute_path"] = "/tmp/garden.licht"
    second["path"] = "/tmp/garden.licht"
    panel._asset_index = SimpleNamespace(
        assets={"a1": first, "a2": second},
        folders={"p1": {"id": "p1", "name": "Projects", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
    )

    container = _ElementStub({"id": "asset-main-row"})
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


def test_dom_card_double_click_opens_licht_project(
    asset_manager_panel_module,
    monkeypatch,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    project = _make_asset()
    panel._asset_index = SimpleNamespace(
        assets={"a1": project},
        verify_asset=lambda _asset_id: SimpleNamespace(to_dict=lambda: project),
        folders={
            "p1": {"id": "p1", "name": "Projects", "scene_ids": ["s1"]}
        },
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
    )
    calls = []
    monkeypatch.setattr(
        asset_manager_panel_module.lf,
        "project_open",
        lambda path, discard_changes=False: calls.append((path, discard_changes)),
        raising=False,
    )

    container = _ElementStub({"id": "asset-popup-content"})
    card = _ElementStub(
        {"data-asset-id": "a1", "data-asset-action": "select"},
        parent=container,
    )
    child = _ElementStub(parent=card)
    event = _EventStub(current_target=container, target=child)

    panel._on_asset_manager_double_click(event)

    assert calls == [("/tmp/bicycle.licht", False)]
    assert panel.get_selected_asset_name() == "bicycle-project"
    assert event.stopped is True


def test_project_load_uses_lifecycle_without_discarding_changes(
    asset_manager_panel_module,
    monkeypatch,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    project = _make_asset()
    panel._asset_index = SimpleNamespace(
        assets={"a1": project},
        verify_asset=lambda _asset_id: SimpleNamespace(to_dict=lambda: project),
        folders={
            "p1": {"id": "p1", "name": "Projects", "scene_ids": ["s1"]}
        },
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
    )
    calls = []

    monkeypatch.setattr(
        asset_manager_panel_module.lf,
        "project_open",
        lambda path, discard_changes=False: calls.append((path, discard_changes)),
        raising=False,
    )

    panel._load_asset("a1")

    assert calls == [("/tmp/bicycle.licht", False)]
    assert panel.get_selected_asset_name() == "bicycle-project"


def test_import_project_uses_licht_dialog_and_content_registration(
    asset_manager_panel_module,
    monkeypatch,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    project = _make_asset()
    project_record = SimpleNamespace(id="a1", name=project["name"])
    calls = []
    panel._asset_index = SimpleNamespace(
        register_licht_asset=lambda path, folder_id=None: (
            calls.append((path, folder_id)) or project_record,
            True,
        )
    )
    panel._with_import_folder = lambda continuation: continuation("default")
    panel.refresh_catalog = lambda: None
    monkeypatch.setattr(
        asset_manager_panel_module.lf.ui,
        "open_project_file_dialog",
        lambda _start_dir="": "/tmp/bicycle.licht",
        raising=False,
    )

    panel.on_import_project(None, None, [])

    assert calls == [("/tmp/bicycle.licht", "default")]
    assert panel._selected_asset_ids == {"a1"}
    assert panel.get_selection_type() == "asset"


def test_dom_card_double_click_ignored_during_input_capture(
    asset_manager_panel_module,
    monkeypatch,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        folders={
            "p1": {"id": "p1", "name": "Projects", "scene_ids": ["s1"]}
        },
        scenes={"s1": {"id": "s1", "name": "bicycle", "folder_id": "p1"}},
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
    )
    panel._selected_folder_id = "missing"
    panel._selection_type = "folder"

    assert panel._repair_selected_folder() == "default"
    assert panel._selected_folder_id == "default"


def test_bind_dom_event_listeners_registers_gallery_wheel_handler(
    asset_manager_panel_module,
):
    panel = asset_manager_panel_module.AssetManagerPanel()
    content = _ElementStub({"id": "asset-main-row"})
    gallery_scroll = _ElementStub({"id": "asset-gallery-scroll"})
    doc = _DocumentStub(
        {
            "asset-main-row": content,
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


def test_asset_scroll_schedules_coalesced_window_refresh(asset_manager_panel_module):
    panel = object.__new__(asset_manager_panel_module.AssetManagerPanel)
    panel._asset_scroll_event_suppressed = False
    panel._asset_scroll_suppressed_top = -1.0
    panel._asset_window_refresh_pending = False
    panel._asset_window_update_requested = False
    scheduled = []
    panel._request_model_update = lambda: scheduled.append("update")
    scroll_el = _ElementStub({"id": "asset-gallery-scroll"})
    event = _EventStub(current_target=scroll_el)

    panel._on_asset_scroll(event)
    panel._on_asset_scroll(event)

    assert panel._asset_window_refresh_pending is True
    assert panel._asset_window_update_requested is True
    assert scheduled == ["update"]


def test_on_update_applies_pending_window_refresh(asset_manager_panel_module):
    panel = object.__new__(asset_manager_panel_module.AssetManagerPanel)
    panel._handle = _HandleStub()
    panel._asset_window_refresh_pending = True
    panel._asset_window_update_requested = True
    panel._view_mode = "list"
    panel._sync_asset_window_viewport = lambda doc=None: False
    panel._sync_gallery_card_width = lambda doc=None: False
    panel._sync_panel_space_state = lambda: False
    panel.get_filtered_assets = lambda: ["row-1", "row-2"]
    panel._asset_window_dirty_fields = lambda: (
        "assets",
        "asset_list_top_spacer_height",
        "asset_list_bottom_spacer_height",
        "asset_gallery_top_spacer_height",
        "asset_gallery_bottom_spacer_height",
    )
    runtime_state = SimpleNamespace(
        scene_generation=SimpleNamespace(value=0),
        language_generation=SimpleNamespace(value=0),
    )
    asset_manager_panel_module.RuntimeState = runtime_state
    panel._last_language_generation = 0
    panel._last_scene_generation = 0
    panel._asset_index = None

    changed = panel.on_update(None)

    assert changed is True
    assert panel._handle.records["assets"] == ["row-1", "row-2"]
    assert "assets" in panel._handle.dirty_fields
    assert panel._asset_window_refresh_pending is False
    assert panel._asset_window_update_requested is False


def test_dirty_model_assets_refresh_does_not_invalidate_catalog_cache(
    asset_manager_panel_module,
):
    panel = object.__new__(asset_manager_panel_module.AssetManagerPanel)
    panel._handle = _HandleStub()
    invalidations = []
    panel._invalidate_catalog_cache = lambda: invalidations.append("invalidate")
    panel._selection_count_fields = lambda: ()
    panel._selection_visibility_fields = lambda: ()
    panel._selected_asset_detail_fields = lambda: ()
    panel._selected_scene_detail_fields = lambda: ()
    panel._selected_folder_detail_fields = lambda: ()
    panel._update_selection_details = lambda update_scene_assets=True: {
        "counts": {},
        "timings_ms": {},
    }
    panel._updating_selection_details = False
    panel.get_filtered_assets = lambda: []
    panel.get_folder_list = lambda: []
    panel.get_scene_list = lambda: []
    panel._request_model_update = lambda: None
    panel._elapsed_ms = lambda start: 0.0
    panel._log_perf = lambda *args, **kwargs: None
    panel._last_dirty_model_timing = {}
    panel._last_asset_rows_update_count = 0
    panel._last_asset_rows_update_ms = 0.0

    panel._dirty_model("assets")

    assert invalidations == []
    assert panel._handle.records["assets"] == []


def test_folder_count_matches_visible_list(asset_manager_panel_module, tmp_path):
    present_file = tmp_path / "present.licht"
    present_file.write_bytes(b"licht")

    def _asset(asset_id, path, *, exists=True):
        asset = dict(_make_asset())
        asset["id"] = asset_id
        asset["name"] = asset_id
        asset["absolute_path"] = str(path)
        asset["path"] = str(path)
        asset["exists"] = exists
        asset["folder_id"] = "p1"
        asset["scene_id"] = "s1"
        return asset

    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={
            "present": _asset("present", present_file),
            "missing": _asset("missing", tmp_path / "deleted.licht", exists=False),
        },
        folders={"p1": {"id": "p1", "name": "Default", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "scene", "folder_id": "p1"}},
    )

    visible = panel.get_filtered_assets()

    assert len(panel._asset_index.assets) == 2
    assert [asset["id"] for asset in visible] == ["missing", "present"]
    assert panel.get_folder_list()[0]["scene_count"] == len(visible)
    assert panel.get_scene_list()[0]["asset_count"] == len(visible)
