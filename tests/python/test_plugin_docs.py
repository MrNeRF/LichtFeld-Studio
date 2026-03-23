# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for plugin documentation examples.

Verifies that all example scripts compile, use correct imports,
follow proper class hierarchies, and match the actual API surface.
"""

import ast
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_EXAMPLES = PROJECT_ROOT / "docs" / "plugins" / "examples"
SRC_PYTHON = PROJECT_ROOT / "src" / "python"

# Ensure lfs_plugins is importable from source
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compile_file(path: Path):
    """Compile a Python file and return the AST, raising on syntax errors."""
    source = path.read_text()
    return compile(source, str(path), "exec", flags=ast.PyCF_ONLY_AST)


def _parse_file(path: Path) -> ast.Module:
    """Parse a Python file into an AST module."""
    return ast.parse(path.read_text(), filename=str(path))


def _find_classes(tree: ast.Module) -> dict[str, list[str]]:
    """Return {class_name: [base_name, ...]} for all classes in the AST."""
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            result[node.name] = bases
    return result


def _find_imports(tree: ast.Module) -> list[str]:
    """Return all imported module names."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _find_functions(tree: ast.Module) -> list[str]:
    """Return all top-level function names."""
    return [
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef)
    ]


EXAMPLE_FILES = sorted(DOCS_EXAMPLES.glob("*.py")) if DOCS_EXAMPLES.exists() else []
FULL_PLUGIN_FILES = (
    sorted((DOCS_EXAMPLES / "full_plugin").rglob("*.py"))
    if (DOCS_EXAMPLES / "full_plugin").exists()
    else []
)
SCRUB_DEMO_FILES = (
    sorted((DOCS_EXAMPLES / "scrub_controls_demo").rglob("*.py"))
    if (DOCS_EXAMPLES / "scrub_controls_demo").exists()
    else []
)
ALL_PY_FILES = EXAMPLE_FILES + FULL_PLUGIN_FILES


# ===========================================================================
# 1. Syntax validation — every .py file must compile
# ===========================================================================

class TestExamplesSyntax:
    """All example scripts must be valid Python."""

    @pytest.mark.parametrize(
        "path",
        ALL_PY_FILES,
        ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_PY_FILES],
    )
    def test_compiles(self, path):
        _compile_file(path)

    @pytest.mark.parametrize(
        "path",
        ALL_PY_FILES,
        ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_PY_FILES],
    )
    def test_parses_to_ast(self, path):
        tree = _parse_file(path)
        assert isinstance(tree, ast.Module)


# ===========================================================================
# 2. Import validation — examples use the correct import paths
# ===========================================================================

ALLOWED_LFS_IMPORTS = {
    "lfs_plugins.types",
    "lfs_plugins.props",
    "lfs_plugins.tools",
    "lfs_plugins.tool_defs.definition",
    "lfs_plugins.capabilities",
    "lfs_plugins.context",
    "lfs_plugins.ui.signals",
    "lfs_plugins.ui.state",
    "lfs_plugins.ui.layout",
    "lfs_plugins.manager",
    "lfs_plugins.icon_manager",
    "lfs_plugins.layouts.menus",
    "lfs_plugins.ui.subscription_registry",
    "lichtfeld",
}


class TestExamplesImports:
    """Examples must use correct import paths."""

    @pytest.mark.parametrize(
        "path",
        EXAMPLE_FILES,
        ids=[p.stem for p in EXAMPLE_FILES],
    )
    def test_imports_are_valid(self, path):
        tree = _parse_file(path)
        imports = _find_imports(tree)
        for imp in imports:
            if imp.startswith(("lfs_plugins", "lichtfeld")):
                assert imp in ALLOWED_LFS_IMPORTS or imp.startswith(
                    "lfs_plugins."
                ), f"Unexpected import: {imp}"

    @pytest.mark.parametrize(
        "path",
        EXAMPLE_FILES,
        ids=[p.stem for p in EXAMPLE_FILES],
    )
    def test_no_wrong_import_patterns(self, path):
        """Examples must use the canonical panel API and avoid retired helpers."""
        source = path.read_text()
        bad_patterns = [
            "lf.ui.Operator",
            "lf.ui.register_panel",
            "lf.ui.register_operator",
            "lf.ui.text(",       # Should be layout.label() or layout.text_wrapped()
        ]
        for pattern in bad_patterns:
            assert pattern not in source, f"Bad pattern '{pattern}' found in {path.name}"


# ===========================================================================
# 3. Class hierarchy — Panels extend Panel, Operators extend Operator
# ===========================================================================

class TestClassHierarchy:
    """Panels must extend Panel, Operators must extend Operator."""

    @pytest.mark.parametrize(
        "path",
        EXAMPLE_FILES,
        ids=[p.stem for p in EXAMPLE_FILES],
    )
    def test_panels_extend_panel(self, path):
        tree = _parse_file(path)
        classes = _find_classes(tree)
        for name, bases in classes.items():
            if "Panel" in bases and name != "Panel":
                assert "Panel" in bases, f"{name} should extend Panel"

    @pytest.mark.parametrize(
        "path",
        EXAMPLE_FILES,
        ids=[p.stem for p in EXAMPLE_FILES],
    )
    def test_operators_extend_operator(self, path):
        tree = _parse_file(path)
        classes = _find_classes(tree)
        for name, bases in classes.items():
            if "Operator" in name and name != "Operator":
                assert "Operator" in bases or "PropertyGroup" in bases, (
                    f"{name} should extend Operator"
                )


# ===========================================================================
# 4. Plugin lifecycle — on_load / on_unload defined
# ===========================================================================

class TestPluginLifecycle:
    """Standalone examples must define on_load() and on_unload()."""

    @pytest.mark.parametrize(
        "path",
        EXAMPLE_FILES,
        ids=[p.stem for p in EXAMPLE_FILES],
    )
    def test_has_on_load(self, path):
        tree = _parse_file(path)
        fns = _find_functions(tree)
        assert "on_load" in fns, f"{path.name} missing on_load()"

    @pytest.mark.parametrize(
        "path",
        EXAMPLE_FILES,
        ids=[p.stem for p in EXAMPLE_FILES],
    )
    def test_has_on_unload(self, path):
        tree = _parse_file(path)
        fns = _find_functions(tree)
        assert "on_unload" in fns, f"{path.name} missing on_unload()"

    def test_full_plugin_init_has_lifecycle(self):
        init = DOCS_EXAMPLES / "full_plugin" / "__init__.py"
        assert init.exists()
        tree = _parse_file(init)
        fns = _find_functions(tree)
        assert "on_load" in fns
        assert "on_unload" in fns


# ===========================================================================
# 5. Full plugin structure
# ===========================================================================

class TestFullPluginStructure:
    """The full_plugin example must have correct directory structure."""

    def test_has_pyproject_toml(self):
        toml = DOCS_EXAMPLES / "full_plugin" / "pyproject.toml"
        assert toml.exists()
        content = toml.read_text()
        assert "[project]" in content
        assert "[tool.lichtfeld]" in content
        assert "name" in content
        assert "version" in content

    def test_has_init(self):
        assert (DOCS_EXAMPLES / "full_plugin" / "__init__.py").exists()

    def test_has_panels_package(self):
        assert (DOCS_EXAMPLES / "full_plugin" / "panels" / "__init__.py").exists()

    def test_has_operators(self):
        assert (DOCS_EXAMPLES / "full_plugin" / "operators.py").exists()

    def test_panel_files_compile(self):
        for py in (DOCS_EXAMPLES / "full_plugin" / "panels").glob("*.py"):
            _compile_file(py)


# ===========================================================================
# 6. Scrub demo structure
# ===========================================================================

class TestScrubControlsDemo:
    """The packaged scrub-controls demo must stay runnable."""

    def test_has_package_files(self):
        root = DOCS_EXAMPLES / "scrub_controls_demo"
        assert (root / "pyproject.toml").exists()
        assert (root / "__init__.py").exists()
        assert (root / "panels" / "__init__.py").exists()
        assert (root / "panels" / "main_panel.py").exists()
        assert (root / "panels" / "main_panel.rml").exists()
        assert (root / "panels" / "main_panel.rcss").exists()

    @pytest.mark.parametrize(
        "path",
        SCRUB_DEMO_FILES,
        ids=[str(p.relative_to(PROJECT_ROOT)) for p in SCRUB_DEMO_FILES],
    )
    def test_python_files_compile(self, path):
        _compile_file(path)

    def test_panel_stays_docked_and_uses_scrub_helpers(self):
        source = (DOCS_EXAMPLES / "scrub_controls_demo" / "panels" / "main_panel.py").read_text()
        assert "ScrubFieldController" in source
        assert "ScrubFieldSpec" in source
        assert "PanelSpace.MAIN_PANEL_TAB" in source


# ===========================================================================
# 7. Pure-Python API verification — test against real source
# ===========================================================================

class TestPanelAPI:
    """Panel base class matches documented API."""

    def test_panel_has_label(self):
        from lfs_plugins.types import Panel
        assert hasattr(Panel, "label")

    def test_panel_has_draw(self):
        from lfs_plugins.types import Panel
        assert callable(getattr(Panel, "draw", None))

    def test_panel_has_poll(self):
        from lfs_plugins.types import Panel
        assert callable(getattr(Panel, "poll", None))

    def test_panel_poll_is_classmethod(self):
        from lfs_plugins.types import Panel
        assert isinstance(Panel.__dict__["poll"], classmethod)

    def test_panel_subclass_works(self):
        from lfs_plugins.types import Panel

        class TestPanel(Panel):
            label = "Test"
            space = "MAIN_PANEL_TAB"
            order = 10

            def draw(self, layout):
                pass

        assert TestPanel.label == "Test"
        assert TestPanel.poll(None) is True
        p = TestPanel()
        p.draw(None)


class TestOperatorAPI:
    """Operator base class matches documented API."""

    def test_operator_has_label(self):
        from lfs_plugins.types import Operator
        assert hasattr(Operator, "label")

    def test_operator_has_description(self):
        from lfs_plugins.types import Operator
        assert hasattr(Operator, "description")

    def test_operator_has_options(self):
        from lfs_plugins.types import Operator
        assert hasattr(Operator, "options")

    def test_operator_has_execute(self):
        from lfs_plugins.types import Operator
        assert callable(getattr(Operator, "execute", None))

    def test_operator_has_invoke(self):
        from lfs_plugins.types import Operator
        assert callable(getattr(Operator, "invoke", None))

    def test_operator_has_modal(self):
        from lfs_plugins.types import Operator
        assert callable(getattr(Operator, "modal", None))

    def test_operator_has_cancel(self):
        from lfs_plugins.types import Operator
        assert callable(getattr(Operator, "cancel", None))

    def test_operator_has_poll(self):
        from lfs_plugins.types import Operator
        assert isinstance(Operator.__dict__["poll"], classmethod)

    def test_operator_extends_property_group(self):
        from lfs_plugins.types import Operator
        from lfs_plugins.props import PropertyGroup
        assert issubclass(Operator, PropertyGroup)

    def test_operator_execute_returns_finished(self):
        from lfs_plugins.types import Operator

        class TestOp(Operator):
            label = "Test"

            def execute(self, context) -> set:
                return {"FINISHED"}

        op = TestOp()
        result = op.execute(None)
        assert result == {"FINISHED"}

    def test_operator_invoke_delegates_to_execute(self):
        from lfs_plugins.types import Operator

        class TestOp(Operator):
            label = "Test"

            def execute(self, context) -> set:
                return {"FINISHED"}

        op = TestOp()
        result = op.invoke(None, None)
        assert result == {"FINISHED"}


class TestEventAPI:
    """Event class matches documented attributes."""

    def test_event_has_documented_attributes(self):
        from lfs_plugins.types import Event
        expected_attrs = [
            "type", "value", "mouse_x", "mouse_y",
            "mouse_region_x", "mouse_region_y",
            "delta_x", "delta_y", "scroll_x", "scroll_y",
            "shift", "ctrl", "alt", "pressure", "over_gui", "key_code",
        ]
        for attr in expected_attrs:
            assert attr in Event.__annotations__, f"Event missing attribute: {attr}"


# ===========================================================================
# 7. Property system verification
# ===========================================================================

class TestPropertySystem:
    """Property types and PropertyGroup match documented API."""

    def test_all_property_types_importable(self):
        from lfs_plugins.props import (
            Property, FloatProperty, IntProperty, BoolProperty,
            StringProperty, EnumProperty, FloatVectorProperty,
            IntVectorProperty, TensorProperty, CollectionProperty,
            PointerProperty, PropertyGroup, PropSubtype,
        )
        assert all([
            Property, FloatProperty, IntProperty, BoolProperty,
            StringProperty, EnumProperty, FloatVectorProperty,
            IntVectorProperty, TensorProperty, CollectionProperty,
            PointerProperty, PropertyGroup, PropSubtype,
        ])

    def test_float_property_validates(self):
        from lfs_plugins.props import FloatProperty
        prop = FloatProperty(default=0.5, min=0.0, max=1.0)
        assert prop.validate(0.5) == 0.5
        assert prop.validate(-1.0) == 0.0
        assert prop.validate(2.0) == 1.0

    def test_int_property_validates(self):
        from lfs_plugins.props import IntProperty
        prop = IntProperty(default=5, min=0, max=10)
        assert prop.validate(5) == 5
        assert prop.validate(-1) == 0
        assert prop.validate(20) == 10

    def test_bool_property_validates(self):
        from lfs_plugins.props import BoolProperty
        prop = BoolProperty(default=False)
        assert prop.validate(True) is True
        assert prop.validate(0) is False

    def test_string_property_validates(self):
        from lfs_plugins.props import StringProperty
        prop = StringProperty(default="", maxlen=5)
        assert prop.validate("hello") == "hello"
        assert prop.validate("hello world") == "hello"

    def test_enum_property_validates(self):
        from lfs_plugins.props import EnumProperty
        prop = EnumProperty(items=[
            ("a", "A", "First"),
            ("b", "B", "Second"),
        ])
        assert prop.validate("a") == "a"
        assert prop.validate("b") == "b"
        assert prop.validate("c") == "a"  # falls back to default

    def test_float_vector_property_validates(self):
        from lfs_plugins.props import FloatVectorProperty
        prop = FloatVectorProperty(default=(0.0, 0.0, 0.0), size=3, min=0.0, max=1.0)
        result = prop.validate((0.5, 0.5, 0.5))
        assert result == (0.5, 0.5, 0.5)
        result = prop.validate((-1.0, 2.0, 0.5))
        assert result == (0.0, 1.0, 0.5)

    def test_int_vector_property_validates(self):
        from lfs_plugins.props import IntVectorProperty
        prop = IntVectorProperty(default=(0, 0), size=2, min=0, max=10)
        result = prop.validate((5, 15))
        assert result == (5, 10)

    def test_property_group_get_set(self):
        from lfs_plugins.props import PropertyGroup, FloatProperty, IntProperty

        class TestGroup(PropertyGroup):
            value = FloatProperty(default=1.0, min=0.0, max=10.0)
            count = IntProperty(default=5, min=0, max=100)

        group = TestGroup()
        assert group.value == 1.0
        assert group.count == 5

        group.value = 3.0
        assert group.value == 3.0

        group.count = 50
        assert group.count == 50

        # Clamping
        group.value = 20.0
        assert group.value == 10.0

    def test_property_group_get_instance_singleton(self):
        from lfs_plugins.props import PropertyGroup, FloatProperty

        class SingletonGroup(PropertyGroup):
            val = FloatProperty(default=0.0)

        a = SingletonGroup.get_instance()
        b = SingletonGroup.get_instance()
        assert a is b

    def test_property_group_runtime_properties(self):
        from lfs_plugins.props import PropertyGroup, FloatProperty

        class DynGroup(PropertyGroup):
            pass

        group = DynGroup()
        group.add_property("dynamic", FloatProperty(default=42.0))
        assert group.dynamic == 42.0

        group.dynamic = 99.0
        assert group.dynamic == 99.0

        group.remove_property("dynamic")
        with pytest.raises(AttributeError):
            _ = group.dynamic

    def test_collection_property(self):
        from lfs_plugins.props import PropertyGroup, CollectionProperty, StringProperty

        class Item(PropertyGroup):
            name = StringProperty(default="untitled")

        class Container(PropertyGroup):
            items = CollectionProperty(type=Item)

        coll = Container.__dict__["items"]
        item = coll.add()
        assert item.name == "untitled"
        assert len(coll) == 1

        item.name = "first"
        assert coll[0].name == "first"

        coll.remove(0)
        assert len(coll) == 0

    def test_prop_subtype_constants(self):
        from lfs_plugins.props import PropSubtype
        assert PropSubtype.COLOR == "COLOR"
        assert PropSubtype.FILE_PATH == "FILE_PATH"
        assert PropSubtype.FACTOR == "FACTOR"
        assert PropSubtype.PERCENTAGE == "PERCENTAGE"
        assert PropSubtype.EULER == "EULER"
        assert PropSubtype.QUATERNION == "QUATERNION"
        assert PropSubtype.TRANSLATION == "TRANSLATION"

    def test_factor_subtype_auto_clamps(self):
        from lfs_plugins.props import FloatProperty
        prop = FloatProperty(default=0.5, subtype="FACTOR")
        assert prop.min == 0.0
        assert prop.max == 1.0

    def test_percentage_subtype_auto_clamps(self):
        from lfs_plugins.props import FloatProperty
        prop = FloatProperty(default=50.0, subtype="PERCENTAGE")
        assert prop.min == 0.0
        assert prop.max == 100.0

    def test_color_subtype_auto_clamps(self):
        from lfs_plugins.props import FloatVectorProperty
        prop = FloatVectorProperty(default=(1.0, 1.0, 1.0), size=3, subtype="COLOR")
        assert prop.min == 0.0
        assert prop.max == 1.0


# ===========================================================================
# 8. Signal system verification
# ===========================================================================

class TestSignalSystem:
    """Reactive signal system matches documented API."""

    def test_signal_get_set(self):
        from lfs_plugins.ui.signals import Signal
        s = Signal(0, name="test")
        assert s.value == 0
        s.value = 42
        assert s.value == 42

    def test_signal_subscribe(self):
        from lfs_plugins.ui.signals import Signal
        s = Signal(0)
        values = []
        unsub = s.subscribe(lambda v: values.append(v))
        s.value = 1
        s.value = 2
        assert values == [1, 2]
        unsub()
        s.value = 3
        assert values == [1, 2]

    def test_signal_no_notify_on_same_value(self):
        from lfs_plugins.ui.signals import Signal
        s = Signal(5)
        values = []
        s.subscribe(lambda v: values.append(v))
        s.value = 5  # same value
        assert values == []

    def test_signal_peek(self):
        from lfs_plugins.ui.signals import Signal
        s = Signal(10)
        assert s.peek() == 10

    def test_computed_signal(self):
        from lfs_plugins.ui.signals import Signal, ComputedSignal
        a = Signal(2)
        b = Signal(3)
        product = ComputedSignal(lambda: a.value * b.value, [a, b])
        assert product.value == 6
        a.value = 4
        assert product.value == 12

    def test_computed_signal_subscribe(self):
        from lfs_plugins.ui.signals import Signal, ComputedSignal
        a = Signal(1)
        b = Signal(2)
        total = ComputedSignal(lambda: a.value + b.value, [a, b])
        values = []
        total.subscribe(lambda v: values.append(v))
        a.value = 10
        assert 12 in values

    def test_throttled_signal(self):
        from lfs_plugins.ui.signals import ThrottledSignal
        s = ThrottledSignal(0, max_rate_hz=1000)
        s.value = 1
        assert s.value == 1

    def test_throttled_signal_flush(self):
        from lfs_plugins.ui.signals import ThrottledSignal
        s = ThrottledSignal(0, max_rate_hz=0.001)  # Very slow rate
        values = []
        s.subscribe(lambda v: values.append(v))
        s.value = 1  # This should go through (first update)
        s.value = 2  # This should be pending (too fast)
        s.flush()
        assert 2 in values

    def test_batch_defers_notifications(self):
        from lfs_plugins.ui.signals import Signal, Batch
        s = Signal(0)
        values = []
        s.subscribe(lambda v: values.append(v))

        with Batch():
            s.value = 1
            s.value = 2
            s.value = 3
            assert values == []  # No notifications yet

        # After batch, should notify with final value
        assert len(values) == 1
        assert values[0] == 3

    def test_batch_context_manager(self):
        from lfs_plugins.ui.signals import Signal, batch
        s = Signal(0)
        values = []
        s.subscribe(lambda v: values.append(v))

        with batch():
            s.value = 10
            assert values == []

        assert values == [10]


# ===========================================================================
# 9. AppState verification
# ===========================================================================

class TestAppState:
    """AppState signals match documented names."""

    def test_training_signals_exist(self):
        from lfs_plugins.ui.state import AppState
        from lfs_plugins.ui.signals import Signal
        assert isinstance(AppState.is_training, Signal)
        assert isinstance(AppState.trainer_state, Signal)
        assert isinstance(AppState.has_trainer, Signal)
        assert isinstance(AppState.iteration, Signal)
        assert isinstance(AppState.max_iterations, Signal)
        assert isinstance(AppState.loss, Signal)
        assert isinstance(AppState.psnr, Signal)
        assert isinstance(AppState.num_gaussians, Signal)

    def test_scene_signals_exist(self):
        from lfs_plugins.ui.state import AppState
        from lfs_plugins.ui.signals import Signal
        assert isinstance(AppState.has_scene, Signal)
        assert isinstance(AppState.scene_generation, Signal)
        assert isinstance(AppState.scene_path, Signal)

    def test_selection_signals_exist(self):
        from lfs_plugins.ui.state import AppState
        from lfs_plugins.ui.signals import Signal
        assert isinstance(AppState.has_selection, Signal)
        assert isinstance(AppState.selection_count, Signal)
        assert isinstance(AppState.selection_generation, Signal)

    def test_computed_signals_exist(self):
        from lfs_plugins.ui.state import AppState
        from lfs_plugins.ui.signals import ComputedSignal
        assert isinstance(AppState.training_progress, ComputedSignal)
        assert isinstance(AppState.can_start_training, ComputedSignal)

    def test_training_progress_computed(self):
        from lfs_plugins.ui.state import AppState
        AppState.iteration.value = 15000
        AppState.max_iterations.value = 30000
        assert abs(AppState.training_progress.value - 0.5) < 1e-6
        # Restore
        AppState.iteration.value = 0
        AppState.max_iterations.value = 30000

    def test_reset(self):
        from lfs_plugins.ui.state import AppState
        AppState.iteration.value = 999
        AppState.reset()
        assert AppState.iteration.value == 0
        assert AppState.is_training.value is False


# ===========================================================================
# 10. Tool system verification
# ===========================================================================

class TestToolSystem:
    """ToolDef and ToolRegistry match documented API."""

    def test_tooldef_creation(self):
        from lfs_plugins.tool_defs.definition import ToolDef
        tool = ToolDef(id="test.tool", label="Test", icon="star")
        assert tool.id == "test.tool"
        assert tool.label == "Test"
        assert tool.icon == "star"
        assert tool.group == "default"
        assert tool.order == 100

    def test_tooldef_is_frozen(self):
        from lfs_plugins.tool_defs.definition import ToolDef
        tool = ToolDef(id="test", label="Test", icon="star")
        with pytest.raises(AttributeError):
            tool.id = "changed"

    def test_submode_def(self):
        from lfs_plugins.tool_defs.definition import SubmodeDef
        sub = SubmodeDef(id="local", label="Local", icon="local")
        assert sub.id == "local"
        assert sub.shortcut == ""

    def test_pivot_mode_def(self):
        from lfs_plugins.tool_defs.definition import PivotModeDef
        pivot = PivotModeDef(id="center", label="Center", icon="dot")
        assert pivot.id == "center"

    def test_tooldef_with_submodes(self):
        from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef, PivotModeDef
        tool = ToolDef(
            id="test.transform",
            label="Transform",
            icon="transform",
            group="transform",
            submodes=(
                SubmodeDef("local", "Local", "local"),
                SubmodeDef("world", "World", "world"),
            ),
            pivot_modes=(
                PivotModeDef("origin", "Origin", "circle"),
            ),
        )
        assert len(tool.submodes) == 2
        assert len(tool.pivot_modes) == 1
        assert tool.submodes[0].id == "local"

    def test_tooldef_to_dict(self):
        from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef
        tool = ToolDef(
            id="test.tool",
            label="Test",
            icon="star",
            submodes=(SubmodeDef("a", "A", "icon_a"),),
        )
        d = tool.to_dict()
        assert d["id"] == "test.tool"
        assert d["label"] == "Test"
        assert len(d["submodes"]) == 1
        assert d["submodes"][0]["id"] == "a"

    def test_tooldef_can_activate_no_poll(self):
        from lfs_plugins.tool_defs.definition import ToolDef
        tool = ToolDef(id="test", label="Test", icon="star")
        assert tool.can_activate(None) is True

    def test_tooldef_can_activate_with_poll(self):
        from lfs_plugins.tool_defs.definition import ToolDef
        tool = ToolDef(
            id="test", label="Test", icon="star",
            poll=lambda ctx: ctx is not None,
        )
        assert tool.can_activate(None) is False
        assert tool.can_activate("something") is True

    def test_tool_registry_register_unregister(self):
        from lfs_plugins.tool_defs.definition import ToolDef
        from lfs_plugins.tools import ToolRegistry

        tool = ToolDef(id="test.reg", label="Reg", icon="star")
        ToolRegistry.register_tool(tool)
        assert ToolRegistry.get("test.reg") is not None
        assert ToolRegistry.get("test.reg").label == "Reg"

        ToolRegistry.unregister_tool("test.reg")
        assert ToolRegistry.get("test.reg") is None


# ===========================================================================
# 11. Capability system verification
# ===========================================================================

class TestCapabilitySystem:
    """CapabilityRegistry, CapabilitySchema match documented API."""

    def test_registry_singleton(self):
        from lfs_plugins.capabilities import CapabilityRegistry
        a = CapabilityRegistry.instance()
        b = CapabilityRegistry.instance()
        assert a is b

    def test_register_and_invoke(self):
        from lfs_plugins.capabilities import CapabilityRegistry, CapabilitySchema

        registry = CapabilityRegistry.instance()

        def handler(args, ctx):
            return {"success": True, "doubled": args.get("value", 0) * 2}

        registry.register(
            name="test.double",
            handler=handler,
            description="Double a value",
            schema=CapabilitySchema(
                properties={"value": {"type": "number"}},
                required=["value"],
            ),
            plugin_name="test_plugin",
            requires_gui=False,
        )

        assert registry.has("test.double")
        cap = registry.get("test.double")
        assert cap is not None
        assert cap.description == "Double a value"
        assert cap.plugin_name == "test_plugin"

        # Cleanup
        registry.unregister("test.double")
        assert not registry.has("test.double")

    def test_unregister_all_for_plugin(self):
        from lfs_plugins.capabilities import CapabilityRegistry

        registry = CapabilityRegistry.instance()
        registry.register("test.a", lambda a, c: {}, plugin_name="bulk_test", requires_gui=False)
        registry.register("test.b", lambda a, c: {}, plugin_name="bulk_test", requires_gui=False)
        registry.register("test.c", lambda a, c: {}, plugin_name="other", requires_gui=False)

        removed = registry.unregister_all_for_plugin("bulk_test")
        assert removed == 2
        assert not registry.has("test.a")
        assert not registry.has("test.b")
        assert registry.has("test.c")

        registry.unregister("test.c")

    def test_invoke_unknown_capability(self):
        from lfs_plugins.capabilities import CapabilityRegistry
        registry = CapabilityRegistry.instance()
        result = registry.invoke("nonexistent.capability", {})
        assert result["success"] is False
        assert "Unknown capability" in result["error"]

    def test_list_all(self):
        from lfs_plugins.capabilities import CapabilityRegistry
        registry = CapabilityRegistry.instance()
        registry.register("test.list", lambda a, c: {}, plugin_name="lister", requires_gui=False)
        caps = registry.list_all()
        names = [c.name for c in caps]
        assert "test.list" in names
        registry.unregister("test.list")


# ===========================================================================
# 12. Context types verification
# ===========================================================================

class TestContextTypes:
    """PluginContext, SceneContext, ViewContext match documented API."""

    def test_scene_context_dataclass(self):
        from lfs_plugins.context import SceneContext
        ctx = SceneContext(scene=None)
        assert ctx.scene is None

    def test_view_context_dataclass(self):
        from lfs_plugins.context import ViewContext
        ctx = ViewContext(
            image=None,
            screen_positions=None,
            width=1920,
            height=1080,
            fov=60.0,
            rotation=None,
            translation=None,
        )
        assert ctx.width == 1920
        assert ctx.height == 1080

    def test_plugin_context_dataclass(self):
        from lfs_plugins.context import PluginContext, CapabilityBroker
        from lfs_plugins.capabilities import CapabilityRegistry
        broker = CapabilityBroker(CapabilityRegistry.instance())
        ctx = PluginContext(scene=None, view=None, capabilities=broker)
        assert ctx.scene is None
        assert ctx.view is None
        assert ctx.capabilities is broker

    def test_capability_broker_has_and_list(self):
        from lfs_plugins.context import CapabilityBroker
        from lfs_plugins.capabilities import CapabilityRegistry
        registry = CapabilityRegistry.instance()
        registry.register("test.broker", lambda a, c: {}, requires_gui=False)

        broker = CapabilityBroker(registry)
        assert broker.has("test.broker")
        assert "test.broker" in broker.list_all()

        registry.unregister("test.broker")


# ===========================================================================
# 13. Icon manager verification
# ===========================================================================

class TestIconManager:
    """Icon manager functions are importable."""

    def test_functions_importable(self):
        from lfs_plugins.icon_manager import (
            get_icon, get_ui_icon, get_scene_icon, get_plugin_icon,
        )
        assert callable(get_icon)
        assert callable(get_ui_icon)
        assert callable(get_scene_icon)
        assert callable(get_plugin_icon)


# ===========================================================================
# 14. Markdown docs verification
# ===========================================================================

class TestMarkdownDocs:
    """Documentation markdown files exist and reference correct APIs."""

    def test_getting_started_exists(self):
        path = PROJECT_ROOT / "docs" / "plugins" / "getting-started.md"
        assert path.exists()
        content = path.read_text()
        assert len(content) > 1000

    def test_api_reference_exists(self):
        path = PROJECT_ROOT / "docs" / "plugins" / "api-reference.md"
        assert path.exists()
        content = path.read_text()
        assert len(content) > 1000

    def test_getting_started_references_correct_imports(self):
        content = (PROJECT_ROOT / "docs" / "plugins" / "getting-started.md").read_text()
        assert "lf.ui.Panel" in content
        assert "from lfs_plugins.types import Operator" in content
        assert "lf.register_class" in content
        assert "lf.unregister_class" in content

    def test_api_reference_covers_all_property_types(self):
        content = (PROJECT_ROOT / "docs" / "plugins" / "api-reference.md").read_text()
        for prop_type in [
            "FloatProperty", "IntProperty", "BoolProperty",
            "StringProperty", "EnumProperty", "FloatVectorProperty",
            "IntVectorProperty", "TensorProperty", "CollectionProperty",
            "PointerProperty",
        ]:
            assert prop_type in content, f"API reference missing {prop_type}"

    def test_api_reference_covers_signals(self):
        content = (PROJECT_ROOT / "docs" / "plugins" / "api-reference.md").read_text()
        assert "Signal[T]" in content or "Signal" in content
        assert "ComputedSignal" in content
        assert "ThrottledSignal" in content
        assert "Batch" in content

    def test_api_reference_covers_layout_api(self):
        content = (PROJECT_ROOT / "docs" / "plugins" / "api-reference.md").read_text()
        for widget in [
            "label(", "button(", "checkbox(", "slider_float(",
            "input_text(", "combo(", "collapsing_header(",
            "separator()", "begin_table(", "progress_bar(",
        ]:
            assert widget in content, f"API reference missing layout widget: {widget}"

    def test_no_wrong_api_patterns_in_docs(self):
        """Docs must not reference wrong API patterns."""
        for md in (PROJECT_ROOT / "docs" / "plugins").glob("*.md"):
            content = md.read_text()
            bad_patterns = [
                "lf.ui.register_panel",
                "bare `draw(self)`",
            ]
            for pattern in bad_patterns:
                assert pattern not in content, (
                    f"Bad pattern '{pattern}' in {md.name}"
                )
