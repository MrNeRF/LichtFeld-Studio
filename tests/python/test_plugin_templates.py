# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for generated plugin templates."""

from importlib import import_module
from pathlib import Path
from types import ModuleType
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PYTHON = PROJECT_ROOT / "src" / "python"
if str(SOURCE_PYTHON) in sys.path:
    sys.path.remove(str(SOURCE_PYTHON))
sys.path.insert(0, str(SOURCE_PYTHON))

from lfs_plugins.templates import create_plugin


def test_create_plugin_generates_rml_panel_template(tmp_path):
    plugin_dir = create_plugin("example_plugin", tmp_path)

    panel_py = plugin_dir / "panels" / "main_panel.py"
    panel_rml = plugin_dir / "panels" / "main_panel.rml"
    panel_rcss = plugin_dir / "panels" / "main_panel.rcss"

    assert panel_py.exists()
    assert panel_rml.exists()
    assert panel_rcss.exists()

    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.setdefault("lichtfeld", ModuleType("lichtfeld"))
        sys.modules.pop("example_plugin", None)
        sys.modules.pop("example_plugin.panels", None)
        sys.modules.pop("example_plugin.panels.main_panel", None)

        module = import_module("example_plugin.panels.main_panel")
        panel_cls = module.MainPanel

        assert panel_cls.__mro__[1].__name__ == "RmlPanel"
        assert Path(panel_cls.rml_template).is_absolute()
        assert Path(panel_cls.rml_template) == panel_rml.resolve()
        assert "data-model=\"example_plugin_main_panel\"" in panel_rml.read_text()
        assert "plugin-main-panel__card" in panel_rcss.read_text()
    finally:
        sys.path.remove(str(tmp_path))
