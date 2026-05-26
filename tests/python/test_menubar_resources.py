# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for retained RmlUI menu bar resources."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_menubar_submenus_are_stacked_above_overlay_and_hit_testable():
    rml = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "menubar.rml"
    ).read_text(encoding="utf-8")
    rcss = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "menubar.rcss"
    ).read_text(encoding="utf-8")
    theme_rcss = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "menubar.theme.rcss"
    ).read_text(encoding="utf-8")

    assert 'data-class-open="item.submenu_open"' in rml
    assert 'data-attr-data-root-index="item.index"' in rml
    assert 'data-class-active="item.submenu_open"' in rml
    assert 'id="dropdown-popup"' in rml
    assert "#dropdown-overlay" in rcss
    assert "#dropdown-container" in rcss
    assert "#dropdown-popup" in rcss
    assert ".submenu-popup" in rcss
    assert "z-index: 1;" in rcss
    assert "z-index: 2;" in rcss
    assert "z-index: 3;" in rcss
    assert "#dropdown-container {\n    display: none;\n    position: absolute;\n    top: 0;" in rcss
    assert "width: 100%;" in rcss and "height: 100%;" in rcss
    assert ".dropdown-popup" in rcss and "overflow: visible;" in rcss
    assert ".submenu-container:hover" not in rcss
    assert ".submenu-popup:hover" not in rcss
    assert ".submenu-container.open > .submenu-popup" in rcss
    assert "pointer-events: auto;" in rcss
    assert ".menu-item.active" in theme_rcss


def test_rml_tooltips_request_only_pending_animation_frames():
    tooltip_header = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "rml_tooltip.hpp"
    ).read_text(encoding="utf-8")
    viewport_header = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rml_viewport_overlay.hpp"
    ).read_text(encoding="utf-8")
    viewport_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rml_viewport_overlay.cpp"
    ).read_text(encoding="utf-8")
    gui_manager_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "gui_manager.cpp"
    ).read_text(encoding="utf-8")

    assert "bool needsFrame() const" in tooltip_header
    assert "&& !visible_" in tooltip_header
    assert "tooltip_.needsFrame()" in viewport_header
    assert "tooltip_.hasActiveState() && applyFrameTooltip()" in viewport_cpp
    assert "setContextNeedsPassiveMouseMoveFrames(rml_context_, tooltip_.needsFrame())" in viewport_cpp
    assert "rml_viewport_overlay_.needsAnimationFrame()" in gui_manager_cpp


def test_menu_bar_uses_retained_bounds_for_submenu_hover():
    menu_bar_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rml_menu_bar.cpp"
    ).read_text(encoding="utf-8")
    menu_bar_header = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rml_menu_bar.hpp"
    ).read_text(encoding="utf-8")

    assert "Rml::Element* dropdownElementAtPoint(float x, float y) const" in menu_bar_header
    assert "RmlMenuBar::dropdownElementAtPoint" in menu_bar_cpp
    assert "GetAbsoluteOffset(Rml::BoxArea::Border)" in menu_bar_cpp
    assert "setOpenSubmenu(submenuIndexForElement(hit_element))" in menu_bar_cpp
    assert "rml_context_->GetElementAtPoint" not in menu_bar_cpp


def test_open_menu_requests_passive_mouse_render_and_blocks_viewport_hit_testing():
    gui_manager_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "gui_manager.cpp"
    ).read_text(encoding="utf-8")

    assert "if (rml_menu_bar_.isOpen())\n            return true;" in gui_manager_cpp
    assert "if (!ui_hidden_ && rml_menu_bar_.isOpen())" in gui_manager_cpp
