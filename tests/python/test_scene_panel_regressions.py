# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for scene panel virtualization and mutation handling."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


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
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        get_current_language=lambda: "en",
        get_invert_masks=lambda: False,
        is_ctrl_down=lambda: False,
        request_redraw=lambda: None,
        is_shift_down=lambda: False,
    )
    lf_stub.get_ui_scale = lambda: 1.0
    lf_stub.get_scene = lambda: None
    lf_stub.get_selected_node_names = lambda: []
    lf_stub.undo = SimpleNamespace(
        stack=lambda: {
            "undo": [],
            "redo": [],
            "total_bytes": 0,
            "transaction_active": False,
            "transaction_depth": 0,
            "transaction_name": "",
        },
        can_undo=lambda: False,
        can_redo=lambda: False,
        undo=lambda: None,
        redo=lambda: None,
        clear=lambda: None,
        jump=lambda stack, count: None,
        subscribe=lambda callback: 0,
        unsubscribe=lambda subscription_id: None,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return lf_stub


@pytest.fixture
def scene_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    sys.modules.pop("lfs_plugins.scene_panel", None)
    sys.modules.pop("lfs_plugins", None)
    _install_lf_stub(monkeypatch)
    return import_module("lfs_plugins.scene_panel")


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)


class _ContainerStub:
    def __init__(self, height=120.0):
        self.client_height = height
        self.offset_height = height
        self.scroll_top = 0.0


class _DocStub:
    def __init__(self, elements):
        self._elements = elements

    def get_element_by_id(self, element_id):
        return self._elements.get(element_id)


def _make_row(index):
    return {
        "name": f"node-{index}",
        "id": index,
        "node_type": "SPLAT",
        "has_children": False,
        "collapsed": False,
        "visible": True,
        "label": f"Node {index}",
        "depth": 0,
        "draggable": True,
        "training_enabled": True,
        "has_mask": False,
        "deletable": True,
    }


def test_scene_panel_prefers_consuming_mutation_flags(scene_panel_module, monkeypatch):
    panel = scene_panel_module.ScenePanel()
    lf_stub = SimpleNamespace(consume_scene_mutation_flags=lambda: 17)

    monkeypatch.setattr(scene_panel_module, "lf", lf_stub)

    assert panel._scene_mutation_flags() == 17


def test_collapsed_models_replace_virtual_rows_with_placeholders(scene_panel_module, monkeypatch):
    panel = scene_panel_module.ScenePanel()
    handle = _HandleStub()
    container = _ContainerStub()

    monkeypatch.setattr(
        scene_panel_module,
        "lf",
        SimpleNamespace(get_ui_scale=lambda: 1.0),
    )

    panel._handle = handle
    panel.container = container
    panel._scene_has_nodes = True
    panel._flat_rows = [_make_row(i) for i in range(6)]

    assert panel._render_tree_window(force=True)
    assert len(handle.records["visible_rows"]) == 6
    assert panel._visible_row_capacity == 6

    panel._models_collapsed = True

    assert panel._render_tree_window(force=True)
    assert len(handle.records["visible_rows"]) == 6
    assert all(not row["present"] for row in handle.records["visible_rows"])
    assert panel._top_spacer_height == "0dp"
    assert panel._bottom_spacer_height == "0dp"
    assert panel._visible_row_capacity == 6


def test_scene_panel_context_menu_uses_record_list(scene_panel_module, monkeypatch):
    panel = scene_panel_module.ScenePanel()
    panel._handle = _HandleStub()
    panel._context_menu = object()
    panel.doc = _DocStub({
        "body": SimpleNamespace(scroll_height=60),
    })
    panel._selected_nodes = {"CameraA"}

    camera = SimpleNamespace(
        type="CAMERA",
        name="CameraA",
        parent_id=-1,
        camera_uid=7,
        training_enabled=True,
    )
    scene = SimpleNamespace(
        get_node=lambda name: camera if name == "CameraA" else None,
    )

    monkeypatch.setattr(
        scene_panel_module,
        "lf",
        SimpleNamespace(
            get_scene=lambda: scene,
            ui=SimpleNamespace(tr=lambda key: key),
        ),
    )

    panel._show_context_menu("CameraA", "14", "50")

    assert panel._context_menu_visible is True
    assert panel._context_menu_left == "14px"
    assert panel._context_menu_top == "0px"
    assert panel._context_menu_node == "CameraA"
    assert panel._handle.records["context_menu_entries"] == [
        {
            "label": "scene.go_to_camera_view",
            "action": "go_to_camera:7",
            "is_label": False,
            "separator_before": False,
            "is_submenu_item": False,
            "is_active": False,
        },
        {
            "label": "scene.disable_for_training",
            "action": "disable_train:CameraA",
            "is_label": False,
            "separator_before": True,
            "is_submenu_item": False,
            "is_active": False,
        },
    ]
    assert panel._handle.dirty_fields == [
        "context_menu_entries",
        "context_menu_visible",
        "context_menu_left",
        "context_menu_top",
    ]


def test_scene_panel_move_to_submenu_builds_records(scene_panel_module):
    panel = scene_panel_module.ScenePanel()

    scene = SimpleNamespace(
        get_nodes=lambda: [
            SimpleNamespace(type="SPLAT", name="Cloud"),
            SimpleNamespace(type="GROUP", name="GroupA"),
            SimpleNamespace(type="GROUP", name="GroupB"),
        ],
    )

    assert panel._build_move_to_items(scene, "Cloud") == [
        {
            "label": "scene.move_to",
            "action": "",
            "is_label": True,
            "separator_before": True,
            "is_submenu_item": False,
            "is_active": False,
        },
        {
            "label": "scene.move_to_root",
            "action": "reparent:Cloud:",
            "is_label": False,
            "separator_before": False,
            "is_submenu_item": True,
            "is_active": False,
        },
        {
            "label": "GroupA",
            "action": "reparent:Cloud:GroupA",
            "is_label": False,
            "separator_before": False,
            "is_submenu_item": True,
            "is_active": False,
        },
        {
            "label": "GroupB",
            "action": "reparent:Cloud:GroupB",
            "is_label": False,
            "separator_before": False,
            "is_submenu_item": True,
            "is_active": False,
        },
    ]


def test_scene_panel_history_tab_builds_embedded_rows(scene_panel_module, monkeypatch):
    panel = scene_panel_module.ScenePanel()
    panel._handle = _HandleStub()
    redraw_requests = {"count": 0}
    undo_state = {
        "undo": [
            {
                "id": "scene_graph.patch",
                "label": "Rename Node",
                "source": "core",
                "scope": "scene_graph",
                "estimated_bytes": 4096,
            }
        ],
        "redo": [],
        "total_bytes": 4096,
        "transaction_active": False,
        "transaction_depth": 0,
        "transaction_name": "",
    }

    monkeypatch.setattr(
        scene_panel_module,
        "lf",
        SimpleNamespace(
            ui=SimpleNamespace(
                get_invert_masks=lambda: False,
                request_redraw=lambda: redraw_requests.__setitem__("count", redraw_requests["count"] + 1),
            ),
            undo=SimpleNamespace(
                stack=lambda: undo_state,
                can_undo=lambda: True,
                can_redo=lambda: False,
                undo=lambda: None,
                redo=lambda: None,
                clear=lambda: None,
                jump=lambda stack, count: None,
            ),
            get_ui_scale=lambda: 1.0,
        ),
    )

    assert panel._refresh_history(force=True) is True
    assert panel._history_undo_label == "Undo: Rename Node"
    assert panel._history_summary_text == "1 undo / 0 redo · 4.0 KB"
    assert panel._handle.records["history_undo_items"] == [
        {
            "label": "Rename Node",
            "title_line": "● Rename Node",
            "stack_line": "NEXT UNDO · Top of stack",
            "detail_line": "scene graph · core · Size: 4.0 KB",
            "scope": "scene graph",
            "source": "core",
            "size": "4.0 KB",
            "is_next": True,
            "kind": "undo",
            "steps": 1,
        }
    ]
    assert panel._history_can_clear is True
    assert redraw_requests["count"] == 1


def test_scene_panel_switching_to_history_tab_hides_context_menu(scene_panel_module, monkeypatch):
    panel = scene_panel_module.ScenePanel()
    panel._handle = _HandleStub()
    panel._context_menu_visible = True
    panel._context_menu_node = "bike"
    panel._context_menu_left = "12px"
    panel._context_menu_top = "8px"
    redraw_requests = {"count": 0}

    monkeypatch.setattr(
        scene_panel_module,
        "lf",
        SimpleNamespace(
            ui=SimpleNamespace(
                request_redraw=lambda: redraw_requests.__setitem__("count", redraw_requests["count"] + 1),
            ),
            undo=SimpleNamespace(
                stack=lambda: {
                    "undo": [],
                    "redo": [],
                    "total_bytes": 0,
                    "transaction_active": False,
                    "transaction_depth": 0,
                    "transaction_name": "",
                },
                clear=lambda: None,
            ),
        ),
    )

    panel._on_switch_tab(args=["history"])

    assert panel._active_tab == scene_panel_module.SCENE_TAB_HISTORY
    assert panel._context_menu_visible is False
    assert panel._context_menu_node is None
    assert "active_tab" in panel._handle.dirty_fields
    assert redraw_requests["count"] >= 1


def test_scene_panel_history_clear_calls_undo_clear(scene_panel_module, monkeypatch):
    panel = scene_panel_module.ScenePanel()
    panel._handle = _HandleStub()
    clear_calls = {"count": 0}
    undo_state = {
        "undo": [{"id": "scene_graph.patch", "label": "Delete Node"}],
        "redo": [],
        "total_bytes": 128,
        "transaction_active": False,
        "transaction_depth": 0,
        "transaction_name": "",
    }

    monkeypatch.setattr(
        scene_panel_module,
        "lf",
        SimpleNamespace(
            ui=SimpleNamespace(
                get_invert_masks=lambda: False,
                request_redraw=lambda: None,
            ),
            undo=SimpleNamespace(
                stack=lambda: undo_state,
                can_undo=lambda: True,
                can_redo=lambda: False,
                undo=lambda: None,
                redo=lambda: None,
                clear=lambda: clear_calls.__setitem__("count", clear_calls["count"] + 1),
                jump=lambda stack, count: None,
            ),
            get_ui_scale=lambda: 1.0,
        ),
    )

    assert panel._refresh_history(force=True) is True

    panel._on_history_clear()

    assert clear_calls["count"] == 1
    assert panel._history_last_state_key is None


def test_scene_panel_scene_change_refreshes_cleared_history(scene_panel_module, monkeypatch):
    panel = scene_panel_module.ScenePanel()
    panel._handle = _HandleStub()
    undo_state = {
        "undo": [{"id": "scene_graph.patch", "label": "Delete Node"}],
        "redo": [],
        "total_bytes": 128,
        "transaction_active": False,
        "transaction_depth": 0,
        "transaction_name": "",
    }

    monkeypatch.setattr(
        scene_panel_module,
        "lf",
        SimpleNamespace(
            ui=SimpleNamespace(
                get_current_language=lambda: "en",
                get_invert_masks=lambda: False,
                request_redraw=lambda: None,
            ),
            undo=SimpleNamespace(
                stack=lambda: undo_state,
                can_undo=lambda: bool(undo_state["undo"]),
                can_redo=lambda: False,
                undo=lambda: None,
                redo=lambda: None,
                clear=lambda: None,
                jump=lambda stack, count: None,
            ),
            get_ui_scale=lambda: 1.0,
            get_selected_node_names=lambda: [],
        ),
    )
    monkeypatch.setattr(panel, "_scene_mutation_flags", lambda: 0)
    monkeypatch.setattr(panel, "_handle_scene_changed", lambda mutation_flags: True)

    assert panel._refresh_history(force=True) is True

    undo_state["undo"] = []
    undo_state["redo"] = []
    undo_state["total_bytes"] = 0

    panel.on_scene_changed(None)

    assert panel._history_summary_text == "No history yet"
    assert panel._handle.records["history_undo_items"] == []
    assert panel._handle.records["history_redo_items"] == []
