# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the retained export panel data model."""

from enum import IntEnum
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _make_node(node_type, name, gaussian_count):
    return SimpleNamespace(type=node_type, name=name, gaussian_count=gaussian_count)


def _install_lf_stub(monkeypatch):
    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")
    node_type = IntEnum("NodeType", {"SPLAT": 1, "MESH": 2})
    state = SimpleNamespace(
        language=["en"],
        nodes=[],
        content_type="splat_files",
        active_camera_count=0,
        export_state={"active": False},
        set_panel_enabled_calls=[],
        export_calls=[],
        folder_dialog_calls=[],
        folder_dialog_result="/tmp/colmap_sparse",
        confirm_calls=[],
        confirm_response="Overwrite",
        colmap_source_path="/datasets/source/sparse/0",
        cancel_calls=0,
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.scene = SimpleNamespace(NodeType=node_type)
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        get_current_language=lambda: state.language[0],
        get_export_state=lambda: dict(state.export_state),
        set_panel_enabled=lambda panel_id, enabled: state.set_panel_enabled_calls.append((panel_id, enabled)),
        cancel_export=lambda: setattr(state, "cancel_calls", state.cancel_calls + 1),
        save_ply_file_dialog=lambda default_name: f"/tmp/{default_name}.ply",
        save_sog_file_dialog=lambda default_name: f"/tmp/{default_name}.sog",
        save_spz_file_dialog=lambda default_name: f"/tmp/{default_name}.spz",
        save_usd_file_dialog=lambda default_name: f"/tmp/{default_name}.usd",
        save_usdz_file_dialog=lambda default_name: f"/tmp/{default_name}.usdz",
        save_html_file_dialog=lambda default_name: f"/tmp/{default_name}.html",
        save_rad_file_dialog=lambda default_name: f"/tmp/{default_name}.rad",
        open_dataset_folder_dialog=lambda default_path="": (
            state.folder_dialog_calls.append(default_path) or state.folder_dialog_result
        ),
        select_colmap_sparse_folder_dialog=lambda default_path="": (
            state.folder_dialog_calls.append(default_path) or state.folder_dialog_result
        ),
        confirm_dialog=lambda title, message, buttons, callback=None: (
            state.confirm_calls.append((title, message, tuple(buttons)))
            or (callback(state.confirm_response) if callback else None)
        ),
        get_content_type=lambda: state.content_type,
    )
    lf_stub.get_colmap_sparse_source_path = lambda: state.colmap_source_path
    lf_stub.get_scene = lambda: SimpleNamespace(
        get_nodes=lambda: list(state.nodes),
        active_camera_count=state.active_camera_count,
    )
    lf_stub.export_scene = (
        lambda fmt, path, nodes, sh_degree, **_kwargs:
        state.export_calls.append((fmt, path, tuple(nodes), sh_degree))
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


@pytest.fixture
def export_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.export_panel", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.export_panel")
    return module, state


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
        self.callbacks = []

    def subscribe(self, callback):
        self.callbacks.append(callback)

        def unsubscribe():
            if callback in self.callbacks:
                self.callbacks.remove(callback)

        return unsubscribe

    def emit(self, value):
        for callback in list(self.callbacks):
            callback(value)


def test_export_panel_builds_format_and_model_records(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._format = module.ExportFormat.SPZ
    panel._selected_nodes = {"Tree"}
    state.nodes = [
        _make_node(module.lf.scene.NodeType.SPLAT, "Tree", 128),
        _make_node(module.lf.scene.NodeType.SPLAT, "House", 64),
    ]

    panel._rebuild_format_records()
    panel._rebuild_model_records(state.nodes)

    assert panel._handle.records["formats"] == [
        {"index": "0", "label": "export.format.ply_standard", "selected": False},
        {"index": "1", "label": "export.format.sog_supersplat", "selected": False},
        {"index": "2", "label": "export.format.spz_niantic", "selected": True},
        {"index": "6", "label": "export.format.rad_random_access", "selected": False},
        {"index": "4", "label": "export.format.usd_openusd", "selected": False},
        {"index": "5", "label": "export.format.usdz_nurec", "selected": False},
        {"index": "3", "label": "export.format.html_viewer", "selected": False},
        {"index": "7", "label": "export.format.colmap_sparse", "selected": False},
    ]
    assert panel._handle.records["models"] == [
        {"name": "Tree", "selected": True, "count_text": "(128)"},
        {"name": "House", "selected": False, "count_text": "(64)"},
    ]
    assert panel._has_models is True
    assert panel.update_policy == "dirty"
    assert "update_interval_ms" not in module.ExportPanel.__dict__


def test_export_panel_seeds_selection_from_scene_nodes(export_panel_module):
    module, _state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._export_sh_degree = 1
    nodes = [
        _make_node(module.lf.scene.NodeType.SPLAT, "Tree", 128),
        _make_node(module.lf.scene.NodeType.SPLAT, "House", 64),
    ]

    assert panel._sync_selection(nodes) is True
    assert panel._selected_nodes == {"Tree", "House"}
    assert panel._selection_seeded is True
    assert panel._export_sh_degree == 3
    assert panel._handle.dirty_fields == ["sh_degree"]


def test_export_panel_progress_updates_bound_value(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    state.export_state = {
        "active": True,
        "progress": 0.5,
        "stage": "writing",
        "format": "ply",
    }

    assert panel._update_export_progress() is True
    assert panel._progress_value == "0.5"
    assert panel._cached_export_state["stage"] == "writing"
    assert panel._handle.dirty_fields == [
        "progress_value",
        "progress_title",
        "progress_pct",
        "progress_stage",
    ]


def test_export_panel_progress_prefers_native_store(export_panel_module, monkeypatch):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    state.export_state = {
        "active": True,
        "progress": 0.1,
        "stage": "legacy",
        "format": "legacy",
    }
    monkeypatch.setattr(
        module,
        "_native_store_value",
        lambda field, fallback: {
            "active": True,
            "progress": 0.75,
            "stage": "native",
            "format": "SPZ",
        }
        if field == "export_progress_state"
        else fallback,
    )

    assert panel._update_export_progress() is True
    assert panel._progress_value == "0.75"
    assert panel._cached_export_state["stage"] == "native"
    assert panel._cached_export_state["format"] == "SPZ"


def test_export_panel_closes_when_export_finishes(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._exporting = True
    panel._selection_seeded = True
    state.export_state = {"active": False}

    assert panel._update_export_progress() is True
    assert panel._exporting is False
    assert panel._selection_seeded is False
    assert state.set_panel_enabled_calls == [("lfs.export", False)]


def test_export_panel_does_not_register_failed_export(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._exporting = True
    panel._last_export_path = "/tmp/failed.ply"
    panel._last_export_format = module.ExportFormat.PLY
    registered = []
    panel._register_export = lambda path, fmt: registered.append((path, fmt))
    state.export_state = {
        "active": False,
        "stage": "Failed",
        "error": "disk full",
        "format": "PLY",
    }

    assert panel._update_export_progress() is True
    assert registered == []
    assert panel._last_export_path is None
    assert panel._last_export_format is None


def test_export_panel_store_subscriptions_mark_panel_dirty(export_panel_module, monkeypatch):
    module, _state = export_panel_module
    scene_signal = _SignalStub()
    export_signal = _SignalStub()
    language_signal = _SignalStub()
    monkeypatch.setattr(
        module,
        "RuntimeState",
        SimpleNamespace(
            scene_generation=scene_signal,
            export_progress_state=export_signal,
            language_generation=language_signal,
        ),
    )
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._last_node_key = ("stale",)
    panel._last_colmap_key = ("stale",)
    panel._last_colmap_source_path = "/old/sparse"

    panel._subscribe_reactive_state()
    export_signal.emit({"active": True})

    assert panel._handle.dirty_fields == ["__update__"]

    scene_signal.emit(1)

    assert panel._last_node_key is None
    assert panel._last_colmap_key is None
    assert panel._last_colmap_source_path == ""
    assert panel._handle.dirty_fields == ["__update__", "__update__"]

    language_signal.emit(1)

    assert panel._handle.dirty_fields == ["__update__", "__update__", "__update__"]

    panel._unsubscribe_reactive_state()
    assert scene_signal.callbacks == []
    assert export_signal.callbacks == []
    assert language_signal.callbacks == []


def test_export_panel_uses_usd_dialog_and_format_id(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._format = module.ExportFormat.USD
    panel._selected_nodes = {"Tree"}
    state.nodes = [_make_node(module.lf.scene.NodeType.SPLAT, "Tree", 128)]

    panel._do_export()

    assert state.export_calls == [
        (int(module.ExportFormat.USD), "/tmp/Tree.usd", ("Tree",), 3),
    ]


def test_export_panel_uses_nurec_usdz_dialog_and_format_id(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._format = module.ExportFormat.NUREC_USDZ
    panel._selected_nodes = {"Tree"}
    state.nodes = [_make_node(module.lf.scene.NodeType.SPLAT, "Tree", 128)]

    panel._do_export()

    assert state.export_calls == [
        (int(module.ExportFormat.NUREC_USDZ), "/tmp/Tree.usdz", ("Tree",), 3),
    ]


def test_export_panel_uses_colmap_folder_picker_without_selected_models(export_panel_module, tmp_path):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._format = module.ExportFormat.COLMAP
    panel._selected_nodes = set()
    state.content_type = "dataset"
    state.active_camera_count = 3
    state.folder_dialog_result = str(tmp_path)

    panel._do_export()

    assert state.export_calls == [
        (int(module.ExportFormat.COLMAP), str(tmp_path), (), 3),
    ]
    assert state.folder_dialog_calls == ["/datasets/source/sparse"]
    assert state.confirm_calls == []


def test_export_panel_exports_to_selected_sparse_root_not_child_model(export_panel_module, tmp_path):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._format = module.ExportFormat.COLMAP
    panel._selected_nodes = set()
    state.content_type = "dataset"
    state.active_camera_count = 3
    sparse_root = tmp_path / "sparse"
    child_model = sparse_root / "0"
    child_model.mkdir(parents=True)
    (child_model / "cameras.bin").write_text("existing child data\n")
    state.folder_dialog_result = str(sparse_root)

    panel._do_export()

    assert state.export_calls == [
        (int(module.ExportFormat.COLMAP), str(sparse_root), (), 3),
    ]
    assert state.confirm_calls == []


def test_export_panel_uses_exact_folder_returned_by_picker(export_panel_module, tmp_path):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._format = module.ExportFormat.COLMAP
    panel._selected_nodes = set()
    state.content_type = "dataset"
    state.active_camera_count = 3
    sparse_root = tmp_path / "sparse"
    child_model = sparse_root / "0"
    child_model.mkdir(parents=True)
    (child_model / "cameras.bin").write_text("existing cameras\n")
    (child_model / "images.bin").write_text("existing images\n")
    state.colmap_source_path = str(child_model)
    state.folder_dialog_result = str(child_model)

    panel._do_export()

    assert state.folder_dialog_calls == [str(sparse_root)]
    assert len(state.confirm_calls) == 1
    title, message, buttons = state.confirm_calls[0]
    assert title == "Export COLMAP sparse"
    assert str(child_model) in message
    assert buttons == ("Overwrite", "Cancel")
    assert state.export_calls == [
        (int(module.ExportFormat.COLMAP), str(child_model), (), 3),
    ]


def test_export_panel_colmap_available_has_no_error(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._format = module.ExportFormat.COLMAP
    state.content_type = "dataset"
    state.active_camera_count = 3

    assert panel._can_export() is True
    assert panel._get_export_error_text() == ""


def test_export_panel_confirms_colmap_overwrite(export_panel_module, tmp_path):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._format = module.ExportFormat.COLMAP
    state.content_type = "dataset"
    state.active_camera_count = 3
    state.folder_dialog_result = str(tmp_path)
    (tmp_path / "cameras.txt").write_text("# existing\n")

    panel._do_export()

    assert len(state.confirm_calls) == 1
    title, message, buttons = state.confirm_calls[0]
    assert title == "Export COLMAP sparse"
    assert str(tmp_path) in message
    assert buttons == ("Overwrite", "Cancel")
    assert state.export_calls == [
        (int(module.ExportFormat.COLMAP), str(tmp_path), (), 3),
    ]


def test_export_panel_cancel_colmap_overwrite(export_panel_module, tmp_path):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._format = module.ExportFormat.COLMAP
    state.content_type = "dataset"
    state.active_camera_count = 3
    state.folder_dialog_result = str(tmp_path)
    state.confirm_response = "Cancel"
    (tmp_path / "images.txt").write_text("# existing\n")

    panel._do_export()

    assert len(state.confirm_calls) == 1
    assert state.export_calls == []
