# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the final plugin API surface."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import re
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _source_python_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "src" / "python"))


def test_plugin_package_exports_unified_panel_types(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins", raising=False)

    module = import_module("lfs_plugins")

    assert module.Panel.__name__ == "Panel"
    assert not hasattr(module, "RmlPanel")
    assert module.Menu.__name__ == "Menu"
    assert module.ScrubFieldController.__name__ == "ScrubFieldController"
    assert module.ScrubFieldSpec.__name__ == "ScrubFieldSpec"


def test_plugin_package_imports_without_v1_panel_runtime(monkeypatch):
    monkeypatch.delitem(sys.modules, "lichtfeld", raising=False)
    monkeypatch.delitem(sys.modules, "lfs_plugins", raising=False)
    monkeypatch.delitem(sys.modules, "lfs_plugins.panels", raising=False)
    monkeypatch.delitem(sys.modules, "lfs_plugins.plugin_marketplace_panel", raising=False)

    fake_lf = ModuleType("lichtfeld")
    fake_lf.ui = SimpleNamespace(
        Panel=type("Panel", (), {}),
        PanelSpace=SimpleNamespace(MAIN_PANEL_TAB="MAIN_PANEL_TAB"),
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", fake_lf)

    module = import_module("lfs_plugins")
    templates = import_module("lfs_plugins.templates")
    signals = import_module("lfs_plugins.ui.signals")
    tool_defs = import_module("lfs_plugins.tool_defs")

    assert module.Panel.__name__ == "Panel"
    assert callable(module.register_builtin_panels)
    assert callable(templates.create_plugin)
    assert signals.Signal.__name__ == "Signal"
    assert tool_defs.ToolDef.__name__ == "ToolDef"


def test_menu_base_exposes_schema_fallback(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins.types", raising=False)

    types_mod = import_module("lfs_plugins.types")

    assert types_mod.Menu().menu_items() == []


def test_rml_widgets_collapsible_uses_text_for_arrow(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins.rml_widgets", raising=False)

    widgets = import_module("lfs_plugins.rml_widgets")

    class _ElementStub:
        def __init__(self, tag="div"):
            self.tag = tag
            self.attrs = {}
            self.classes = ""
            self.text = None
            self.inner_rml = None
            self.children_list = []

        def append_child(self, tag):
            child = _ElementStub(tag)
            self.children_list.append(child)
            return child

        def set_id(self, value):
            self.attrs["id"] = value

        def set_class_names(self, value):
            self.classes = value

        def set_attribute(self, name, value):
            self.attrs[name] = value

        def set_text(self, value):
            self.text = value

        def set_inner_rml(self, value):
            self.inner_rml = value

        def set_class(self, _name, _enabled):
            return None

        def remove_property(self, _name):
            return None

        def set_property(self, _name, _value):
            return None

        @property
        def client_height(self):
            return 0

        @property
        def scroll_height(self):
            return 0

    container = _ElementStub()
    header, _content = widgets.collapsible(container, "advanced", title="Advanced", open=True)

    arrow = header.children_list[0]
    assert arrow.text == chr(0x25B6)
    assert arrow.inner_rml is None


def test_ui_stub_surface_matches_panel_api_names():
    stub_path = PROJECT_ROOT / "src" / "python" / "stubs" / "lichtfeld" / "ui" / "__init__.pyi"
    stub_text = stub_path.read_text()

    assert "class PanelInfo:" in stub_text
    assert "class PanelSummary:" in stub_text
    assert "poll_dependencies" in stub_text
    assert "idname:" not in stub_text
    assert "poll_deps" not in stub_text
    assert "space: PanelSpace = PanelSpace.MAIN_PANEL_TAB" in stub_text
    assert "PanelSpace | str" not in stub_text
    assert "def get_panel_names(space: PanelSpace = PanelSpace.FLOATING)" in stub_text
    assert "def set_panel_space(panel_id: str, space: PanelSpace)" in stub_text
    assert "    NONE: int = ..." in stub_text
    assert "    None: int = ..." not in stub_text
    assert "class ThemeVignette:" in stub_text
    assert "def vignette(self) -> ThemeVignette: ..." in stub_text
    assert "def set_theme_vignette_enabled(arg: bool, /) -> None:" in stub_text
    assert "def set_theme_vignette_intensity(arg: float, /) -> None:" in stub_text
    assert "def set_theme_vignette_style(arg0: float, arg1: float, arg2: float, /) -> None:" in stub_text


def test_training_session_stub_surface():
    stub = (PROJECT_ROOT / "src" / "python" / "stubs" / "lichtfeld" / "__init__.pyi").read_text()
    assert "def project_training_session_state() -> dict:" in stub
    assert "def restore_training_session(then_start: bool = False) -> None:" in stub
    assert "def training_get_state() -> dict:" in stub


def test_plugin_stub_surface_exposes_v1_compatibility_contract():
    root_stub = (PROJECT_ROOT / "src" / "python" / "stubs" / "lichtfeld" / "__init__.pyi").read_text()
    plugins_stub = (PROJECT_ROOT / "src" / "python" / "stubs" / "lichtfeld" / "plugins.pyi").read_text()

    assert "PLUGIN_API_VERSION: str" in root_stub
    assert "API_VERSION: str" in plugins_stub
    assert "FEATURES: list = ..." in plugins_stub


def test_plugin_api_version_matches_across_cpp_and_stub_copies(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins.compat", raising=False)
    from lfs_plugins.compat import PLUGIN_API_VERSION

    py_plugins_text = (PROJECT_ROOT / "src" / "python" / "lfs" / "py_plugins.cpp").read_text()
    module_text = (PROJECT_ROOT / "src" / "python" / "lfs" / "module.cpp").read_text()
    stub_text = (PROJECT_ROOT / "src" / "python" / "stubs" / "lichtfeld" / "__init__.pyi").read_text()

    py_plugins_match = re.search(r'plugins\.attr\("API_VERSION"\)\s*=\s*"([^"]+)"', py_plugins_text)
    assert py_plugins_match, 'py_plugins.cpp: could not find plugins.attr("API_VERSION") assignment'
    assert py_plugins_match.group(1) == PLUGIN_API_VERSION, (
        f"py_plugins.cpp API_VERSION ({py_plugins_match.group(1)!r}) is out of sync with "
        f"compat.PLUGIN_API_VERSION ({PLUGIN_API_VERSION!r})"
    )

    module_match = re.search(r'm\.attr\("PLUGIN_API_VERSION"\)\s*=\s*"([^"]+)"', module_text)
    assert module_match, 'module.cpp: could not find m.attr("PLUGIN_API_VERSION") assignment'
    assert module_match.group(1) == PLUGIN_API_VERSION, (
        f"module.cpp PLUGIN_API_VERSION ({module_match.group(1)!r}) is out of sync with "
        f"compat.PLUGIN_API_VERSION ({PLUGIN_API_VERSION!r})"
    )

    stub_match = re.search(r"PLUGIN_API_VERSION:\s*str\s*=\s*'([^']+)'", stub_text)
    assert stub_match, "lichtfeld/__init__.pyi: could not find PLUGIN_API_VERSION literal"
    assert stub_match.group(1) == PLUGIN_API_VERSION, (
        f"stub PLUGIN_API_VERSION ({stub_match.group(1)!r}) is out of sync with "
        f"compat.PLUGIN_API_VERSION ({PLUGIN_API_VERSION!r})"
    )


def test_plugin_supported_features_match_cpp_features_list(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins.compat", raising=False)
    from lfs_plugins.compat import SUPPORTED_PLUGIN_FEATURES

    py_plugins_text = (PROJECT_ROOT / "src" / "python" / "lfs" / "py_plugins.cpp").read_text()
    cpp_features = re.findall(r'features\.append\("([^"]+)"\);', py_plugins_text)
    assert cpp_features, "py_plugins.cpp: could not find any features.append(...) calls"

    assert len(cpp_features) == len(SUPPORTED_PLUGIN_FEATURES), (
        f"py_plugins.cpp declares {len(cpp_features)} features, "
        f"compat.SUPPORTED_PLUGIN_FEATURES has {len(SUPPORTED_PLUGIN_FEATURES)}"
    )
    assert set(cpp_features) == set(SUPPORTED_PLUGIN_FEATURES), (
        f"py_plugins.cpp features {sorted(cpp_features)} do not match "
        f"compat.SUPPORTED_PLUGIN_FEATURES {sorted(SUPPORTED_PLUGIN_FEATURES)}"
    )


def test_lichtfeld_version_matches_cmake_project_version(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins.compat", raising=False)
    from lfs_plugins.compat import LICHTFELD_VERSION

    cmake_text = (PROJECT_ROOT / "CMakeLists.txt").read_text()
    cmake_match = re.search(r"project\([^)]*\bVERSION\s+(\d+\.\d+\.\d+)", cmake_text)
    assert cmake_match, "CMakeLists.txt: could not find project() VERSION"
    assert cmake_match.group(1) == LICHTFELD_VERSION, (
        f"CMakeLists.txt project VERSION ({cmake_match.group(1)!r}) is out of sync with "
        f"compat.LICHTFELD_VERSION ({LICHTFELD_VERSION!r})"
    )


def test_python_stub_workflow_uses_explicit_sync_and_check_targets():
    cmake_path = PROJECT_ROOT / "src" / "python" / "CMakeLists.txt"
    cmake_text = cmake_path.read_text()

    assert "refresh_python_stubs" in cmake_text
    assert "check_python_stubs" in cmake_text
    assert "Syncing stubs to source tree" not in cmake_text
    assert "lfs_ui_panel.py" not in cmake_text
    assert "_lfs_panel_contract.py" in cmake_text
    assert not (PROJECT_ROOT / "src" / "python" / "lfs_ui_panel.py").exists()


def test_panel_core_sources_use_v1_internal_names():
    panel_core_files = [
        "src/visualizer/gui/panel_registry.hpp",
        "src/visualizer/gui/panel_registry.cpp",
        "src/visualizer/gui/panel_layout.hpp",
        "src/visualizer/gui/panel_layout.cpp",
        "src/visualizer/gui/rml_right_panel.hpp",
        "src/visualizer/gui/rml_right_panel.cpp",
        "src/visualizer/gui/rmlui/resources/right_panel.rml",
    ]

    for rel_path in panel_core_files:
        text = (PROJECT_ROOT / rel_path).read_text()
        assert "idname" not in text
        assert "parent_idname" not in text
        assert "poll_deps" not in text

    gui_manager_text = (PROJECT_ROOT / "src" / "visualizer" / "gui" / "gui_manager.cpp").read_text()
    assert ".id = t.id" in gui_manager_text
    assert "makeRmlTabDomId(t.id)" in gui_manager_text
    assert ".idname = t.idname" not in gui_manager_text


def test_python_undo_surface_has_single_entry_point(lf):
    assert hasattr(lf, "undo")
    assert not hasattr(lf.ops, "undo")
    assert not hasattr(lf.ops, "redo")
    assert not hasattr(lf.ops, "can_undo")
    assert not hasattr(lf.ops, "can_redo")
    assert not hasattr(lf.pipeline, "undo")


def test_floating_panels_accept_typed_space_enums(lf):
    if not hasattr(lf, "ui") or not hasattr(lf.ui, "Panel"):
        pytest.skip("panel API not available")

    assert not hasattr(lf.ui.PanelSpace, "DOCKABLE")

    panel_id = "tests.typed_floating_panel"

    class TypedFloatingPanel(lf.ui.Panel):
        id = panel_id
        label = "Typed Floating"
        space = lf.ui.PanelSpace.FLOATING
        options = {lf.ui.PanelOption.DEFAULT_CLOSED}

        def draw(self, ui):
            del ui

    try:
        lf.register_class(TypedFloatingPanel)
    except ValueError as exc:
        if "retained UI manager" in str(exc):
            pytest.skip(
                "floating window registration requires an active retained UI manager"
            )
        raise
    try:
        floating_names = set(lf.ui.get_panel_names(lf.ui.PanelSpace.FLOATING))
        panel_info = lf.ui.get_panel(panel_id)
        assert panel_info is not None
        assert panel_info.id == panel_id
        assert panel_info.space == lf.ui.PanelSpace.FLOATING
        assert lf.ui.set_panel_space(panel_id, lf.ui.PanelSpace.FLOATING) is True
        assert panel_id in floating_names
    finally:
        lf.unregister_class(TypedFloatingPanel)
