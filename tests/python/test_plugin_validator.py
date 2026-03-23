# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for plugin structure validation."""

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
