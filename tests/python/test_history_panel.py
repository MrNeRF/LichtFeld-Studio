# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the history panel data model."""

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
    undo_state = {
        "undo": [],
        "redo": [],
        "undo_bytes": 0,
        "redo_bytes": 0,
        "total_bytes": 0,
        "transaction_active": False,
        "transaction_depth": 0,
        "transaction_name": "",
    }
    redraw_requests = {"count": 0}
    clear_calls = {"count": 0}
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        request_redraw=lambda: redraw_requests.__setitem__("count", redraw_requests["count"] + 1),
    )
    lf_stub.undo = SimpleNamespace(
        stack=lambda: undo_state,
        can_undo=lambda: bool(undo_state["undo"]),
        can_redo=lambda: bool(undo_state["redo"]),
        undo=lambda: None,
        redo=lambda: None,
        clear=lambda: clear_calls.__setitem__("count", clear_calls["count"] + 1),
        jump=lambda stack, count: {"success": True, "changed": True, "steps_performed": count, "error": ""},
        subscribe=lambda callback: 1,
        unsubscribe=lambda subscription_id: None,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return lf_stub, undo_state, redraw_requests, clear_calls


@pytest.fixture
def history_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    sys.modules.pop("lfs_plugins.history_panel", None)
    sys.modules.pop("lfs_plugins", None)
    lf_stub, undo_state, redraw_requests, clear_calls = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.history_panel")
    module.lf = lf_stub
    return module, undo_state, redraw_requests, clear_calls


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)


def test_history_panel_builds_rows_from_structured_stack(history_panel_module):
    module, undo_state, redraw_requests, _clear_calls = history_panel_module
    panel = module.HistoryPanel()
    panel._handle = _HandleStub()

    undo_state.update(
        {
            "undo": [
                {
                    "id": "scene_graph.patch",
                    "label": "Rename Node",
                    "source": "core",
                    "scope": "scene_graph",
                    "estimated_bytes": 4096,
                }
            ],
            "redo": [
                {
                    "id": "selection.grow",
                    "label": "Grow Selection",
                    "source": "core",
                    "scope": "selection",
                    "estimated_bytes": 512,
                }
            ],
            "total_bytes": 4608,
            "transaction_active": True,
            "transaction_depth": 2,
            "transaction_name": "Grouped Move",
        }
    )

    assert panel._refresh(force=True) is True

    assert panel._undo_label == "Undo: Rename Node"
    assert panel._redo_label == "Redo: Grow Selection"
    assert panel._summary_text == "1 undo / 1 redo · 4.5 KB"
    assert panel._transaction_label == "Transaction active: Grouped Move (depth 2)"
    assert panel._handle.records["undo_items"] == [
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
    assert panel._handle.records["redo_items"] == [
        {
            "label": "Grow Selection",
            "title_line": "● Grow Selection",
            "stack_line": "NEXT REDO · Top of stack",
            "detail_line": "selection · core · Size: 512 B",
            "scope": "selection",
            "source": "core",
            "size": "512 B",
            "is_next": True,
            "kind": "redo",
            "steps": 1,
        }
    ]
    assert panel._can_clear is True
    assert redraw_requests["count"] == 1


def test_history_panel_empty_state(history_panel_module):
    module, _undo_state, redraw_requests, _clear_calls = history_panel_module
    panel = module.HistoryPanel()
    panel._handle = _HandleStub()

    assert panel._refresh(force=True) is True
    assert panel._summary_text == "No history yet"
    assert panel._empty_text == "Nothing recorded yet"
    assert panel._can_clear is False
    assert panel._handle.records["undo_items"] == []
    assert panel._handle.records["redo_items"] == []
    assert redraw_requests["count"] == 1


def test_history_panel_on_update_polls_even_with_subscription(history_panel_module):
    module, undo_state, redraw_requests, _clear_calls = history_panel_module
    panel = module.HistoryPanel()
    panel._handle = _HandleStub()
    panel._subscription_id = 1

    assert panel._refresh(force=True) is True

    undo_state.update(
        {
            "undo": [
                {
                    "id": "scene_graph.patch",
                    "label": "Duplicate Node",
                    "source": "core",
                    "scope": "scene_graph",
                    "estimated_bytes": 2048,
                    "gpu_bytes": 0,
                }
            ],
            "redo": [],
            "total_bytes": 2048,
            "total_cpu_bytes": 2048,
            "total_gpu_bytes": 0,
            "transaction_active": False,
            "transaction_depth": 0,
            "transaction_name": "",
        }
    )

    assert panel.on_update(None) is True
    assert panel._summary_text == "1 undo / 0 redo · 2.0 KB total · 0 B GPU"
    assert redraw_requests["count"] == 2


def test_history_panel_subscription_refresh_requests_redraw(history_panel_module):
    module, undo_state, redraw_requests, _clear_calls = history_panel_module
    panel = module.HistoryPanel()
    panel._handle = _HandleStub()

    assert panel._refresh(force=True) is True

    undo_state.update(
        {
            "undo": [
                {
                    "id": "history.transaction.translate",
                    "label": "Translate",
                    "source": "history",
                    "scope": "grouped",
                    "estimated_bytes": 136,
                    "cpu_bytes": 136,
                    "gpu_bytes": 0,
                }
            ],
            "redo": [],
            "total_bytes": 136,
            "total_cpu_bytes": 136,
            "total_gpu_bytes": 0,
            "transaction_active": False,
            "transaction_depth": 0,
            "transaction_name": "",
        }
    )

    panel._on_history_changed()

    assert redraw_requests["count"] == 2
    assert panel._last_state_key is None

    assert panel.on_update(None) is True
    assert panel._summary_text == "1 undo / 0 redo · 136 B total · 0 B GPU"
    assert redraw_requests["count"] == 3


def test_history_panel_clear_invokes_undo_clear(history_panel_module):
    module, undo_state, _redraw_requests, clear_calls = history_panel_module
    panel = module.HistoryPanel()
    panel._handle = _HandleStub()

    undo_state["undo"] = [{"id": "scene_graph.patch", "label": "Delete Node"}]
    assert panel._refresh(force=True) is True

    panel._on_clear()

    assert clear_calls["count"] == 1
    assert panel._last_state_key is None


def test_history_panel_scene_change_forces_empty_refresh(history_panel_module):
    module, undo_state, redraw_requests, _clear_calls = history_panel_module
    panel = module.HistoryPanel()
    panel._handle = _HandleStub()

    undo_state["undo"] = [{"id": "scene_graph.patch", "label": "Delete Node"}]
    assert panel._refresh(force=True) is True

    undo_state["undo"] = []
    undo_state["redo"] = []
    undo_state["total_bytes"] = 0

    panel.on_scene_changed(None)

    assert panel._summary_text == "No history yet"
    assert panel._handle.records["undo_items"] == []
    assert panel._handle.records["redo_items"] == []
    assert redraw_requests["count"] >= 2
