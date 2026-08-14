# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for retained RmlUI menu bar resources."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _rule_body(rcss: str, selector: str) -> str:
    return rcss.split(f"{selector} {{", 1)[1].split("}", 1)[0]


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
    dropdown_rule = _rule_body(rcss, ".dropdown-popup")
    assert "min-width: 200dp;" in dropdown_rule
    assert "max-width: 360dp;" in dropdown_rule
    assert "overflow: visible;" in dropdown_rule
    submenu_rule = _rule_body(rcss, ".submenu-popup")
    assert "min-width: 200dp;" in submenu_rule
    assert "max-width: 420dp;" in submenu_rule
    assert "overflow: hidden;" in submenu_rule
    menu_item_rule = _rule_body(rcss, ".menu-item")
    assert "flex-shrink: 0;" in menu_item_rule
    assert "white-space: nowrap;" in menu_item_rule
    label_rule = _rule_body(rcss, ".menu-item .label")
    assert "min-width: 0;" in label_rule
    assert "white-space: nowrap;" in label_rule
    assert "overflow: hidden;" in label_rule
    assert "text-overflow: ellipsis;" in label_rule
    assert ".submenu-container:hover" not in rcss
    assert ".submenu-popup:hover" not in rcss
    assert ".submenu-container.open > .submenu-popup" in rcss
    assert "pointer-events: auto;" in rcss
    assert 'data-attr-title="item.tooltip"' in rml
    assert 'data-attr-title="child.tooltip"' in rml
    assert ".menu-item.active" in theme_rcss
    assert 'id="menu-window-fullscreen"' not in rml
    assert 'data-action="window_toggle_fullscreen"' not in rml
    assert 'id="menu-window-toggle-ui"' in rml
    assert 'data-action="window_toggle_ui"' in rml
    assert rml.count('data-for="button : menu_camera_buttons"') == 1
    assert rml.count('data-for="button : menu_render_buttons"') == 1
    assert rml.index('data-for="button : menu_camera_buttons"') < rml.index(
        'data-for="button : menu_render_buttons"'
    )
    assert rml.index('data-for="button : menu_render_buttons"') < rml.index(
        'data-for="button : menu_projection_buttons"'
    )
    toolbar_button_rule = _rule_body(rcss, ".menu-toolbar-btn")
    assert "transition: none;" in toolbar_button_rule


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
    assert "tooltip_.revealDue()" in viewport_header
    assert "tooltip_.hasActiveState()" in viewport_cpp
    assert "applyFrameTooltip()" in viewport_cpp
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
    assert "tooltip_.setHover(resolveRmlTooltip(hit_element), hit_element)" in menu_bar_cpp
    assert "void sizeOpenDropdowns();" in menu_bar_header
    assert "RmlMenuBar::sizeOpenDropdowns" in menu_bar_cpp
    assert "GetScrollWidth()" in menu_bar_cpp
    assert 'GetElementsByClassName(submenus, "submenu-popup")' in menu_bar_cpp
    assert "element->GetDisplay() == Rml::Style::Display::None" in menu_bar_cpp
    assert "tested against the previous submenu geometry" in menu_bar_cpp
    assert "a fast following click cannot hit the previous menu's rows" in menu_bar_cpp
    assert "rml_context_->GetElementAtPoint" not in menu_bar_cpp
    assert 'action == "window_toggle_fullscreen"' not in menu_bar_cpp
    assert 'ctor.Bind("menu_camera_buttons", &camera_buttons_)' in menu_bar_cpp
    assert 'action == "set_camera_navigation_mode"' in menu_bar_cpp
    assert "setCameraNavigationMode" in menu_bar_cpp
    assert "std::vector<MenuToolbarButtonView> camera_buttons_" in menu_bar_header


def test_open_menu_requests_passive_mouse_render_and_blocks_viewport_hit_testing():
    gui_manager_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "gui_manager.cpp"
    ).read_text(encoding="utf-8")

    assert "if (rml_menu_bar_.isOpen())\n            return true;" in gui_manager_cpp
    assert (
        "if (rml_menu_bar_.isOpen()) {\n"
        "            return {.blocks_pointer = true, .takes_keyboard_focus = true};\n"
        "        }"
    ) in gui_manager_cpp
    assert "if (!ui_hidden_ && rml_menu_bar_.isOpen())" not in gui_manager_cpp


def test_viewport_overlay_toolbar_origin_tracks_viewport_content_offset():
    gui_manager_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "gui_manager.cpp"
    ).read_text(encoding="utf-8")

    assert "const float viewport_content_offset = viewport_layout_.pos.x - screen.work_pos.x;" in gui_manager_cpp
    assert "float primary_toolbar_x = viewport_content_offset;" in gui_manager_cpp
    assert "rml_viewport_overlay_.setViewportContentOffset(viewport_content_offset);" in gui_manager_cpp


def test_scene_header_hosts_asset_manager_launcher():
    scene_rml = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "scene_tree.rml"
    ).read_text(encoding="utf-8")
    scene_rcss = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "scene_tree.rcss"
    ).read_text(encoding="utf-8")
    scene_cpp = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "scene_panel_native.cpp"
    ).read_text(encoding="utf-8")

    assert 'id="asset-manager-button"' in scene_rml
    assert 'data-tooltip="toolbar.asset_manager"' in scene_rml
    assert ".scene-header-icon-button" in scene_rcss
    header_rule = _rule_body(scene_rcss, ".scene-header-row")
    assert "min-width: 0;" in header_rule
    tab_bar_rule = _rule_body(scene_rcss, ".scene-tab-bar")
    assert "min-width: 0;" in tab_bar_rule
    assert "flex-grow: 1;" in tab_bar_rule
    assert "flex-shrink: 1;" in tab_bar_rule
    tab_rule = _rule_body(scene_rcss, ".scene-tab")
    assert "min-width: 0;" in tab_rule
    assert "flex-grow: 1;" in tab_rule
    assert "flex-shrink: 1;" in tab_rule
    assert "text-overflow: ellipsis;" in tab_rule
    asset_button_rule = _rule_body(scene_rcss, ".scene-header-icon-button")
    assert "flex-shrink: 0;" in asset_button_rule
    assert "width: 30dp;" in scene_rcss
    assert "height: 30dp;" in scene_rcss
    assert "width: 20dp;" in scene_rcss
    assert "height: 20dp;" in scene_rcss
    assert 'resolveRmlImageSource("icon/archive.png")' in scene_cpp
    assert 'panel_registry.is_panel_enabled("lfs.asset_manager")' in scene_cpp
    assert 'panel_registry.set_panel_enabled("lfs.asset_manager", !currently_open);' in scene_cpp
