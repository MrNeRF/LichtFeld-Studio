# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin template generator for scaffolding new plugins."""

import logging
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

PYPROJECT_TOML = '''[project]
name = "{name}"
version = "0.1.0"
description = "A new LichtFeld plugin"

[tool.lichtfeld]
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
'''

INIT_PY = '''"""
{name} - A LichtFeld Studio plugin.
"""

import lichtfeld as lf
from .panels.main_panel import MainPanel

_classes = [MainPanel]


def on_load():
    """Called when plugin is loaded."""
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("{name} plugin loaded")


def on_unload():
    """Called when plugin is unloaded."""
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("{name} plugin unloaded")
'''

MAIN_PANEL_PY = '''"""Main panel for {name} plugin."""

import lichtfeld as lf
from pathlib import Path


class MainPanel(lf.ui.Panel):
    """Example plugin panel using the unified Panel API."""

    id = "{name}.main_panel"
    label = "{title}"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 100
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))

    def __init__(self):
        self._click_count = 0
        self._enabled = True
        self._strength = 0.35

    def draw(self, ui):
        ui.heading("{title}")
        ui.text_disabled("Immediate widgets rendered inside the RML shell below.")

        if ui.button(f"Click Me ({{self._click_count}})"):
            self._click_count += 1
            lf.log.info("{name}: Button clicked")
        ui.same_line()
        if ui.button_styled("Reset", "secondary"):
            self._click_count = 0
            self._enabled = True
            self._strength = 0.35

        _, self._enabled = ui.checkbox("Enabled", self._enabled)
        _, self._strength = ui.slider_float("Strength", self._strength, 0.0, 1.0)

        ui.separator()
        ui.text_wrapped(
            f"enabled={{self._enabled}} | strength={{self._strength:.2f}} | clicks={{self._click_count}}"
        )
'''

MAIN_PANEL_RML = '''<rml>
<head>
  <link type="text/rcss" href="main_panel.rcss"/>
</head>
<body id="body" class="plugin-panel {name}-panel">
  <div class="panel-shell">
    <div class="panel-header">
      <span class="panel-kicker">Plugin Scaffold</span>
      <span class="panel-title">{title}</span>
      <span class="panel-note">RML and RCSS are set up already. Immediate widgets mount into the slot below.</span>
    </div>
    <div id="im-root"></div>
  </div>
</body>
</rml>
'''

MAIN_PANEL_RCSS = '''body.plugin-panel {
    font-family: Inter;
    padding: 0;
}

.panel-shell {
    display: flex;
    flex-direction: column;
    gap: 12dp;
    padding: 16dp;
}

.panel-header {
    display: flex;
    flex-direction: column;
    gap: 4dp;
}

.panel-kicker {
    color: #7fb8ff;
    font-size: 12dp;
    font-weight: bold;
    letter-spacing: 0.8dp;
    text-transform: uppercase;
}

.panel-title {
    font-size: 18dp;
    font-weight: bold;
}

.panel-note {
    color: #a0a8b7;
}

#im-root {
    width: 100%;
}
'''


def create_plugin(name: str, target_dir: Optional[Path] = None) -> Path:
    """Create a new plugin from template.

    Args:
        name: Plugin name (used for directory and module)
        target_dir: Optional target directory (defaults to ~/.lichtfeld/plugins)

    Returns:
        Path to created plugin directory

    Raises:
        FileExistsError: If plugin directory already exists
    """
    if target_dir is None:
        target_dir = Path.home() / ".lichtfeld" / "plugins"

    plugin_dir = target_dir / name
    if plugin_dir.exists():
        raise FileExistsError(f"Plugin directory already exists: {plugin_dir}")

    title = name.replace("_", " ").title()

    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "panels").mkdir(exist_ok=True)

    (plugin_dir / "pyproject.toml").write_text(PYPROJECT_TOML.format(name=name))
    (plugin_dir / "__init__.py").write_text(INIT_PY.format(name=name))
    (plugin_dir / "panels" / "__init__.py").write_text("")
    (plugin_dir / "panels" / "main_panel.py").write_text(
        MAIN_PANEL_PY.format(name=name, title=title)
    )
    (plugin_dir / "panels" / "main_panel.rml").write_text(
        MAIN_PANEL_RML.format(name=name, title=title)
    )
    (plugin_dir / "panels" / "main_panel.rcss").write_text(MAIN_PANEL_RCSS)

    _log.info("Created plugin template at %s", plugin_dir)
    return plugin_dir
