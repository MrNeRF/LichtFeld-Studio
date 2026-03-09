# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for hook-driven viewport toolbar updates."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_stub_modules(monkeypatch):
    hook_calls = []
    remove_calls = []

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        add_hook=lambda panel, section, callback, position="append": hook_calls.append(
            (panel, section, callback, position)
        ),
        remove_hook=lambda panel, section, callback: remove_calls.append(
            (panel, section, callback)
        ),
        rml=SimpleNamespace(get_document=lambda _name: None),
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    tools_mod = ModuleType("lfs_plugins.tools")

    class _ToolRegistryStub:
        @staticmethod
        def get_all():
            return []

        @staticmethod
        def get(_tool_id):
            return None

        @staticmethod
        def clear_active():
            return None

        @staticmethod
        def set_active(_tool_id):
            return None

    tools_mod.ToolRegistry = _ToolRegistryStub
    monkeypatch.setitem(sys.modules, "lfs_plugins.tools", tools_mod)

    op_context_mod = ModuleType("lfs_plugins.op_context")
    op_context_mod.get_context = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "lfs_plugins.op_context", op_context_mod)

    ui_pkg = ModuleType("lfs_plugins.ui")
    ui_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "lfs_plugins.ui", ui_pkg)

    state_mod = ModuleType("lfs_plugins.ui.state")
    state_mod.AppState = SimpleNamespace(trainer_state=SimpleNamespace(value="idle"))
    monkeypatch.setitem(sys.modules, "lfs_plugins.ui.state", state_mod)

    return hook_calls, remove_calls


@pytest.fixture
def toolbar_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) in sys.path:
        sys.path.remove(str(source_python))
    sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins", None)
    sys.modules.pop("lfs_plugins.toolbar", None)
    hook_calls, remove_calls = _install_stub_modules(monkeypatch)
    module = import_module("lfs_plugins.toolbar")
    return module, hook_calls, remove_calls


def test_toolbar_register_uses_viewport_overlay_hook(toolbar_module):
    module, hook_calls, _remove_calls = toolbar_module

    module.register()

    assert len(hook_calls) == 1
    panel, section, callback, position = hook_calls[0]
    assert (panel, section, position) == ("viewport_overlay", "document", "append")

    class _DocumentStub:
        def get_element_by_id(self, _element_id):
            return None

    callback(_DocumentStub())


def test_toolbar_unregister_removes_same_hook(toolbar_module):
    module, hook_calls, remove_calls = toolbar_module

    module.register()
    module.unregister()

    assert len(hook_calls) == 1
    assert remove_calls == [("viewport_overlay", "document", hook_calls[0][2])]


def test_gizmo_toolbar_does_not_rebuild_for_new_document_wrappers(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module

    class _ElementStub:
        def __init__(self, tag="div"):
            self.tag = tag
            self.attrs = {}
            self.classes = set()
            self.listeners = []
            self.child_nodes = []

        def set_id(self, value):
            self.attrs["id"] = value

        def set_class(self, name, active):
            if active:
                self.classes.add(name)
            else:
                self.classes.discard(name)

        def set_class_names(self, names):
            self.classes = set(names.split())

        def add_event_listener(self, name, callback):
            self.listeners.append((name, callback))

        def set_attribute(self, name, value):
            self.attrs[name] = value

        def remove_attribute(self, name):
            self.attrs.pop(name, None)

        def append_child(self, tag):
            child = _ElementStub(tag)
            self.child_nodes.append(child)
            return child

        def query_selector(self, _selector):
            return None

    class _ContainerStub(_ElementStub):
        def __init__(self):
            super().__init__("div")
            self.removed = 0

        def num_children(self):
            return len(self.child_nodes)

        def children(self):
            return list(self.child_nodes)

        def remove_child(self, child):
            self.removed += 1
            self.child_nodes.remove(child)

    gizmo_container = _ContainerStub()
    submode_container = _ContainerStub()
    pivot_container = _ContainerStub()

    class _DocumentStub:
        def __init__(self, gizmo, submode, pivot):
            self._elements = {
                "gizmo-toolbar": gizmo,
                "submode-toolbar": submode,
                "pivot-toolbar": pivot,
            }

        def get_element_by_id(self, element_id):
            return self._elements.get(element_id)

    docs = [
        _DocumentStub(gizmo_container, submode_container, pivot_container),
        _DocumentStub(gizmo_container, submode_container, pivot_container),
    ]
    monkeypatch.setattr(
        sys.modules["lichtfeld"].ui.rml,
        "get_document",
        lambda _name: docs.pop(0),
    )
    monkeypatch.setattr(
        sys.modules["lichtfeld"].ui,
        "get_active_tool",
        lambda: "",
        raising=False,
    )

    tool_stub = SimpleNamespace(
        id="builtin.select",
        icon="selection",
        label="Select",
        shortcut="1",
        submodes=(),
        pivot_modes=(),
        can_activate=lambda _context: True,
    )
    monkeypatch.setattr(
        sys.modules["lfs_plugins.tools"].ToolRegistry,
        "get_all",
        staticmethod(lambda: [tool_stub]),
    )
    monkeypatch.setattr(
        sys.modules["lfs_plugins.tools"].ToolRegistry,
        "get",
        staticmethod(lambda _tool_id: tool_stub),
    )
    monkeypatch.setattr(
        sys.modules["lfs_plugins.op_context"],
        "get_context",
        lambda: SimpleNamespace(has_scene=True, num_gaussians=1),
    )

    rml_widgets_mod = ModuleType("lfs_plugins.rml_widgets")
    rml_widgets_mod.icon_button = lambda container, button_id, icon_src, **_kwargs: _make_icon_button(
        container, button_id, icon_src
    )
    monkeypatch.setitem(sys.modules, "lfs_plugins.rml_widgets", rml_widgets_mod)

    controller = module._GizmoToolbarController()
    controller.update(docs[0])
    controller.update(docs[1])

    assert gizmo_container.num_children() == 1
    assert gizmo_container.removed == 0


def _make_icon_button(container, button_id, icon_src):
    button = container.append_child("div")
    button.set_id(button_id)
    button.set_attribute("data-icon-src", icon_src)
    return button
