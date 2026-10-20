# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for plugin structure validation."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _source_python_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "src" / "python"))


def _write_plugin(plugin_dir: Path, panel_source: str) -> None:
    (plugin_dir / "panels").mkdir(parents=True)
    (plugin_dir / "pyproject.toml").write_text(
        """[project]
name = "example_plugin"
version = "0.1.0"
description = "Example"

[tool.lichtfeld]
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
"""
    )
    (plugin_dir / "__init__.py").write_text(
        """def on_load():
    pass


def on_unload():
    pass
"""
    )
    (plugin_dir / "panels" / "__init__.py").write_text("")
    (plugin_dir / "panels" / "main_panel.py").write_text(panel_source)


def test_validator_accepts_pure_immediate_panel(tmp_path):
    from lfs_plugins.validator import validate_plugin

    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(
        plugin_dir,
        """import lichtfeld as lf


class MainPanel(lf.ui.Panel):
    id = "example.main_panel"
    label = "Example"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB

    def draw(self, ui):
        ui.label("hello")
""",
    )

    assert validate_plugin(plugin_dir) == []


def test_validator_ignores_panel_sources_inside_venv(tmp_path):
    from lfs_plugins.validator import validate_plugin

    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(
        plugin_dir,
        """import lichtfeld as lf


class MainPanel(lf.ui.Panel):
    id = "example.main_panel"
    label = "Example"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
""",
    )

    venv = plugin_dir / ".venv"
    python_name = "python.exe" if sys.platform == "win32" else "python"
    python_path = venv / ("Scripts" if sys.platform == "win32" else "bin") / python_name
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    dependency = venv / "Lib" / "site-packages" / "dependency.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text(
        "import lichtfeld as lf\n"
        "from pathlib import Path\n\n"
        "class DependencyPanel(lf.ui.Panel):\n"
        "    template = str(Path(__file__).with_name('missing.rml'))\n"
        "    label = '日本語'\n",
        encoding="utf-8",
    )

    assert validate_plugin(plugin_dir) == []


def test_validator_reports_missing_local_template_file(tmp_path):
    from lfs_plugins.validator import validate_plugin

    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(
        plugin_dir,
        """import lichtfeld as lf
from pathlib import Path


class MainPanel(lf.ui.Panel):
    id = "example.main_panel"
    label = "Example"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
""",
    )

    errors = validate_plugin(plugin_dir)

    assert "Missing template file: panels/main_panel.rml" in errors


def test_validator_reports_missing_linked_rcss_file(tmp_path):
    from lfs_plugins.validator import validate_plugin

    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(
        plugin_dir,
        """import lichtfeld as lf
from pathlib import Path


class MainPanel(lf.ui.Panel):
    id = "example.main_panel"
    label = "Example"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
""",
    )
    (plugin_dir / "panels" / "main_panel.rml").write_text(
        """<rml>
<head>
  <link type="text/rcss" href="main_panel.rcss"/>
</head>
<body>
  <div id="im-root"></div>
</body>
</rml>
"""
    )

    errors = validate_plugin(plugin_dir)

    assert "Missing stylesheet file: panels/main_panel.rcss" in errors


def test_validator_accepts_local_template_with_stylesheet(tmp_path):
    from lfs_plugins.validator import validate_plugin

    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(
        plugin_dir,
        """import lichtfeld as lf
from pathlib import Path


class MainPanel(lf.ui.Panel):
    id = "example.main_panel"
    label = "Example"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
""",
    )
    (plugin_dir / "panels" / "main_panel.rml").write_text(
        """<rml>
<head>
  <link type="text/rcss" href="main_panel.rcss"/>
</head>
<body>
  <div id="im-root"></div>
</body>
</rml>
"""
    )
    (plugin_dir / "panels" / "main_panel.rcss").write_text("body { padding: 0; }\n")

    assert validate_plugin(plugin_dir) == []


def test_validator_reports_missing_local_template_from_parent_join(tmp_path):
    from lfs_plugins.validator import validate_plugin

    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(
        plugin_dir,
        """import lichtfeld as lf
from pathlib import Path


class MainPanel(lf.ui.Panel):
    id = "example.main_panel"
    label = "Example"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    template = str(Path(__file__).resolve().parent / "main_panel.rml")
""",
    )

    errors = validate_plugin(plugin_dir)

    assert "Missing template file: panels/main_panel.rml" in errors


def test_validator_reports_missing_linked_rcss_from_parent_join(tmp_path):
    from lfs_plugins.validator import validate_plugin

    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(
        plugin_dir,
        """import lichtfeld as lf
from pathlib import Path


class MainPanel(lf.ui.Panel):
    id = "example.main_panel"
    label = "Example"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    template = str(Path(__file__).resolve().parent / "main_panel.rml")
""",
    )
    (plugin_dir / "panels" / "main_panel.rml").write_text(
        """<rml>
<head>
  <link type="text/rcss" href="main_panel.rcss"/>
</head>
<body>
  <div id="im-root"></div>
</body>
</rml>
"""
    )

    errors = validate_plugin(plugin_dir)

    assert "Missing stylesheet file: panels/main_panel.rcss" in errors


def test_validator_requires_plugin_compatibility_fields(tmp_path):
    from lfs_plugins.validator import validate_plugin

    plugin_dir = tmp_path / "example_plugin"
    (plugin_dir / "panels").mkdir(parents=True)
    (plugin_dir / "pyproject.toml").write_text(
        """[project]
name = "example_plugin"
version = "0.1.0"
description = "Example"

[tool.lichtfeld]
hot_reload = true
"""
    )
    (plugin_dir / "__init__.py").write_text(
        """def on_load():
    pass


def on_unload():
    pass
"""
    )

    errors = validate_plugin(plugin_dir)

    assert any("missing tool.lichtfeld.plugin_api" in error for error in errors)
    assert any("missing tool.lichtfeld.lichtfeld_version" in error for error in errors)
    assert any("missing tool.lichtfeld.required_features" in error for error in errors)
    assert any("v1 manifest requires" in error for error in errors)


@pytest.mark.parametrize("declaration, message", [
    ('EnumProperty(items=[("a", "A", "")], default="missing")', "is not in items"),
    ('EnumProperty(items=[("a", "A", ""), ("a", "Again", "")])', "duplicate enum identifier"),
    ('EnumProperty(items=[("a", "A", "")], default=0)', "string identifier"),
    ('EnumProperty(items=[("", "Empty", "")])', "non-empty string identifier"),
    ('IntProperty(default=True)', "incompatible type"),
    ('FloatProperty(default="0.5")', "incompatible type"),
    ('BoolProperty(default=1)', "incompatible type"),
    ('StringProperty(default=12)', "incompatible type"),
    ('IntProperty(default=0, min=1)', "below the declared minimum"),
    ('IntProperty(min=1)', "below the declared minimum"),
    ('FloatProperty(default=2.0, max=1.0)', "above the declared maximum"),
])
def test_checker_reports_literal_default_issues_without_executing_plugin(tmp_path, declaration, message):
    from lfs_plugins.validator import validate_plugin
    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(plugin_dir, f'''from lfs_plugins.props import EnumProperty, IntProperty, FloatProperty, BoolProperty, StringProperty
raise RuntimeError("Plugin code must not execute during preflight")
class Settings:
    value = {declaration}
''')
    findings = validate_plugin(plugin_dir)
    assert len(findings) == 1
    assert message in findings[0]
    assert "panels/main_panel.py:4:" in findings[0]
    assert "plugin loading is unaffected" in findings[0]


@pytest.mark.parametrize("imports, constructor", [
    ("from lfs_plugins.props import EnumProperty as Choice", "Choice"),
    ("import lfs_plugins.props as p", "p.EnumProperty"),
    ("from lfs_plugins import props", "props.EnumProperty"),
    ("import lichtfeld as lf", "lf.props.EnumProperty"),
    ("from lichtfeld import props as p", "p.EnumProperty"),
])
def test_checker_resolves_supported_property_import_aliases(tmp_path, imports, constructor):
    from lfs_plugins.validator import validate_plugin
    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(plugin_dir, f'{imports}\nvalue = {constructor}(items=[("a", "A", "")], default="missing")\n')
    assert any("is not in items" in finding for finding in validate_plugin(plugin_dir))


def test_checker_includes_init_and_ignores_virtualenv(tmp_path):
    from lfs_plugins.validator import validate_plugin
    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(plugin_dir, "")
    source = 'from lfs_plugins.props import EnumProperty\nx = EnumProperty(items=[], default="missing")\n'
    with (plugin_dir / "__init__.py").open("a") as stream:
        stream.write(source)
    ignored = plugin_dir / "venv" / "bad.py"
    ignored.parent.mkdir()
    ignored.write_text(source)
    findings = validate_plugin(plugin_dir)
    assert len(findings) == 1 and "__init__.py:" in findings[0]


def test_checker_does_not_guess_dynamic_expressions_or_custom_constructors(tmp_path):
    from lfs_plugins.validator import validate_plugin
    plugin_dir = tmp_path / "example_plugin"
    _write_plugin(plugin_dir, '''from lfs_plugins.props import EnumProperty as Choice
from other_package import EnumProperty
a = Choice(items=make_items(), default=get_default())
b = EnumProperty(items=[], default="custom semantics")
''')
    assert validate_plugin(plugin_dir) == []


@pytest.mark.parametrize("kind, kwargs", [
    ("BoolProperty", {"default": False}), ("IntProperty", {"default": 0}),
    ("FloatProperty", {"default": 0.0}), ("StringProperty", {"default": ""}),
    ("EnumProperty", {"items": [("a", "A", "")], "default": None}),
    ("EnumProperty", {"items": [("a", "A", "")], "default": "a"}),
])
def test_shared_contract_accepts_valid_falsey_and_implicit_defaults(kind, kwargs):
    from lfs_plugins.property_contract import default_metadata_issues
    assert default_metadata_issues(kind, **kwargs) == []


@pytest.mark.parametrize("kind, value", [
    ("FloatProperty", float("nan")), ("FloatProperty", float("inf")),
    ("FloatProperty", 10**1000), ("IntProperty", 2**40),
])
def test_shared_contract_reports_unrepresentable_defaults_without_throwing(kind, value):
    from lfs_plugins.property_contract import default_metadata_issues
    assert default_metadata_issues(kind, default=value)
