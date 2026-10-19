# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression contracts for the Python console theme boundary."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "src" / "visualizer" / "gui" / "rmlui" / "resources"


def _resource(name: str) -> str:
    return (RESOURCES / name).read_text(encoding="utf-8")


def _rule(css: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(.*?)\n\}}", css, re.DOTALL)
    assert match is not None
    return match.group(1)


def _rules_with(css: str, selector: str) -> list[str]:
    return [
        body
        for selectors, body in re.findall(r"([^{}]+)\{(.*?)\n\}", css, re.DOTALL)
        if selector in (s.strip() for s in selectors.split(","))
    ]


def test_python_console_theme_overrides_toolbar_specificity():
    theme = _resource("python_console_panel.theme.rcss")

    assert "#content #python-console-toolbar" in theme
    assert "#python-console-toolbar button" in theme
    assert "#python-console-toolbar button:hover" in theme
    assert "#python-console-toolbar button.disabled" in theme


def test_python_console_keeps_original_semantic_controls_outside_theme_overrides():
    base = _resource("python_console_panel.rcss")
    theme = _resource("python_console_panel.theme.rcss")

    active_rule = _rule(base, "#python-console-toolbar button:active,\n#python-console-toolbar button.active")
    run_rule = _rule(base, "#python-console-toolbar .run-button")
    stop_rule = _rule(base, "#python-console-toolbar .stop-button")

    assert "rgb(31, 74, 111)" in active_rule
    assert "rgb(35, 197, 82)" in run_rule
    assert "rgb(166, 63, 67)" in stop_rule
    assert "#python-console-toolbar button:active" not in theme
    assert "#python-console-toolbar .run-button" not in theme
    assert "#python-console-toolbar .stop-button" not in theme


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
