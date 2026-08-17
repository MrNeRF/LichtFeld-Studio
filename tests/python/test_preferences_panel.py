# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for application preferences behavior."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def preferences_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    state = SimpleNamespace(language="it", set_language_calls=[])
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(FILL="fill"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        get_current_language=lambda: state.language,
        set_language=lambda language: state.set_language_calls.append(language),
    )

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    sys.modules.pop("lfs_plugins.preferences_panel", None)
    sys.modules.pop("lfs_plugins", None)
    module = import_module("lfs_plugins.preferences_panel")
    return module, state


def test_language_selection_does_not_reload_active_language(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._language_catalog = [("en", "English"), ("it", "Italiano")]

    panel._set_language_index("1")

    assert state.set_language_calls == []


def test_language_selection_applies_a_different_language(
    preferences_panel_module, monkeypatch
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._language_catalog = [("en", "English"), ("it", "Italiano")]
    monkeypatch.setattr(panel, "_refresh_selection", lambda: None)

    panel._set_language_index("0")

    assert state.set_language_calls == ["en"]
