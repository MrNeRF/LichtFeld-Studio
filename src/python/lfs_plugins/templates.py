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

from lfs_plugins.types import RmlPanel


class MainPanel(RmlPanel):
    """Example plugin panel using a plugin-local RmlUI template."""

    idname = "{name}.main_panel"
    label = "{title}"
    space = "MAIN_PANEL_TAB"
    order = 100
    rml_template = str(Path(__file__).resolve().with_name("main_panel.rml"))

    def __init__(self):
        self._click_count = 0
        self._model_handle = None

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("{name}_main_panel")
        if model is None:
            return

        model.bind_func("greeting", lambda: "Hello from {name}!")
        model.bind_func("button_label", lambda: "Click Me")
        model.bind_func("click_summary", lambda: f"Clicked {{self._click_count}} times")
        model.bind_event("click_me", self._on_click_me)
        self._model_handle = model.get_handle()

    def _on_click_me(self, _handle, _ev, _args):
        self._click_count += 1
        if self._model_handle:
            self._model_handle.dirty("click_summary")
            lf.log.info("{name}: Button clicked!")
'''

MAIN_PANEL_RML = '''<rml>
<head>
  <link type="text/rcss" href="main_panel.rcss"/>
</head>
<body data-model="{name}_main_panel" class="plugin-main-panel">
  <div class="plugin-main-panel__card">
    <span class="plugin-main-panel__title">{{greeting}}</span>
    <span class="plugin-main-panel__summary">{{click_summary}}</span>
    <button class="btn btn--primary" data-event-click="click_me">{{button_label}}</button>
  </div>
</body>
</rml>
'''

MAIN_PANEL_RCSS = '''body.plugin-main-panel {
    padding: 16dp;
}

.plugin-main-panel__card {
    display: flex;
    flex-direction: column;
    gap: 10dp;
    max-width: 320dp;
}

.plugin-main-panel__title {
    font-size: 15dp;
    font-weight: bold;
}

.plugin-main-panel__summary {
    color: #a0a8b7;
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
    (plugin_dir / "panels" / "main_panel.rml").write_text(MAIN_PANEL_RML.format(name=name))
    (plugin_dir / "panels" / "main_panel.rcss").write_text(MAIN_PANEL_RCSS)

    _log.info("Created plugin template at %s", plugin_dir)
    return plugin_dir
