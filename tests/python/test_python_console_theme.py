# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression contracts for the Python console theme boundary."""

from pathlib import Path
import re

from rcss_test_support import rule_body as _rule, rule_bodies as _rules_with


ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "src" / "visualizer" / "gui" / "rmlui" / "resources"


def _resource(name: str) -> str:
    return (RESOURCES / name).read_text(encoding="utf-8")


def test_python_console_theme_overrides_toolbar_specificity():
    theme = _resource("python_console_panel.theme.rcss")

    assert "#content #python-console-toolbar" in theme
    assert "#python-console-toolbar button" in theme
    assert "#python-console-toolbar button:hover" in theme
    assert "#python-console-toolbar button.disabled" in theme


def test_python_console_keeps_original_semantic_controls_outside_theme_overrides():
    base = _resource("python_console_panel.rcss")
    theme = _resource("python_console_panel.theme.rcss")

    active_rule = _rule(base, "#python-console-toolbar button:active")
    run_rule = _rule(base, "#python-console-toolbar .run-button")
    stop_rule = _rule(base, "#python-console-toolbar .stop-button")

    assert "rgb(31, 74, 111)" in active_rule
    assert "rgb(35, 197, 82)" in run_rule
    assert "rgb(166, 63, 67)" in stop_rule
    for selector, role in (
        ("#python-console-toolbar button:active", "primary"),
        ("#python-console-toolbar button.active", "primary"),
        ("#python-console-toolbar .run-button", "success"),
        ("#python-console-toolbar .stop-button", "error"),
    ):
        rules = _rules_with(theme, selector)
        assert f"border-color: @{{{role}}};" in "\n".join(rules)
        assert all("background-color:" not in rule for rule in rules)


def test_python_console_terminals_take_their_colors_from_the_theme():
    base = _resource("python_console_panel.rcss")
    theme = _resource("python_console_panel.theme.rcss")

    for terminal in ("#python-output-terminal", "#python-repl-terminal"):
        base_rules = _rules_with(base, terminal)
        theme_rules = _rules_with(theme, terminal)
        assert any("font-family" in rule for rule in base_rules)
        assert not any(re.search(r"(^|[\s;])(background-)?color:", rule) for rule in base_rules)
        assert any("background-color: @{surface}" in rule and "color: @{text}" in rule for rule in theme_rules)


def test_python_console_tabs_and_splitter_stay_compact():
    base = _resource("python_console_panel.rcss")
    theme = _resource("python_console_panel.theme.rcss")

    tab_rule = _rule(base, "#tabbar button")
    assert "height: 24dp" in tab_rule
    assert "line-height: 24dp" in tab_rule
    assert "text-align: center" in tab_rule

    splitter_hover = _rule(theme, "#python-splitter:hover")
    assert "border-top-color: @{primary}" in splitter_hover
    assert "background-color: @{primary_dim}" not in splitter_hover
