# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for retained RmlUI menu bar resources."""

import json
import re
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
    assert 'data-class-open="child.submenu_open"' in rml
    assert 'data-attr-data-child-index="child.index"' in rml
    assert 'data-for="leaf : child.children"' in rml
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
    assert "overflow: visible;" in submenu_rule
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


def test_toolbar_icons_use_contrast_aware_theme_tokens():
    resources = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "rmlui" / "resources"
    )
    menubar_theme = (resources / "menubar.theme.rcss").read_text(encoding="utf-8")
    viewport_theme = (resources / "viewport_overlay.theme.rcss").read_text(
        encoding="utf-8"
    )
    viewport_layout = (resources / "viewport_overlay.rcss").read_text(
        encoding="utf-8"
    )
    rml_theme_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "rml_theme.cpp"
    ).read_text(encoding="utf-8")

    assert "readableIconColor" in rml_theme_cpp
    assert "MIN_ICON_CONTRAST = 4.5f" in rml_theme_cpp
    assert "const std::array backgrounds" in rml_theme_cpp
    assert "VIEWPORT_GIZMO_OPACITY = 0.64f" in rml_theme_cpp
    assert 'colorToRmlAlpha(viewport_toolbar_icon, 0.70f)' in rml_theme_cpp
    for token in (
        "menu.toolbar_icon",
        "menu.toolbar_selected_decor",
        "menu.toolbar_selected_icon",
        "viewport.toolbar_glass_decor",
        "viewport.toolbar_text",
        "viewport.toolbar_selected_decor",
        "viewport.toolbar_selected_icon",
        "viewport.gizmo_decor",
        "viewport.gizmo_icon",
        "viewport.gizmo_hover_decor",
        "viewport.gizmo_hover_icon",
        "viewport.gizmo_disabled_icon",
        "viewport.gizmo_selected_decor",
        "viewport.gizmo_selected_icon",
        "viewport.gizmo_selected_hover_decor",
        "viewport.gizmo_selected_hover_icon",
    ):
        assert f'{{"{token}"' in rml_theme_cpp

    assert "image-color: @{menu.toolbar_icon};" in menubar_theme
    assert "@{menu.toolbar_selected_decor};" in menubar_theme
    assert "image-color: @{menu.toolbar_selected_icon};" in menubar_theme
    assert "@{viewport.toolbar_glass_decor};" in viewport_theme
    assert "@{viewport.toolbar_selected_decor};" in viewport_theme
    assert "image-color: @{viewport.toolbar_selected_icon};" in viewport_theme
    assert "@{viewport.gizmo_decor};" in viewport_theme
    assert "image-color: @{viewport.gizmo_icon};" in viewport_theme
    assert "@{viewport.gizmo_hover_decor};" in viewport_theme
    assert "image-color: @{viewport.gizmo_hover_icon};" in viewport_theme
    assert "@{viewport.gizmo_selected_decor};" in viewport_theme
    assert "image-color: @{viewport.gizmo_selected_icon};" in viewport_theme
    assert "width: 27dp;" in viewport_layout
    assert "height: 27dp;" in viewport_layout
    assert "width: 18dp;" in viewport_layout
    assert "height: 18dp;" in viewport_layout


def test_optional_gradients_style_the_primary_application_chrome():
    resources = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "rmlui" / "resources"
    )
    rml_theme_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "rml_theme.cpp"
    ).read_text(encoding="utf-8")
    consumers = {
        "chrome.menu_decor": ("shell.theme.rcss", "menubar.theme.rcss"),
        "chrome.status_decor": ("shell.theme.rcss", "statusbar.theme.rcss"),
        "chrome.right_panel_decor": ("right_panel.theme.rcss",),
        "chrome.right_panel_edge_shape": ("right_panel.theme.rcss",),
        "chrome.right_panel_separator_decor": ("right_panel.theme.rcss",),
        "chrome.toolbar_decor": ("viewport_overlay.theme.rcss",),
        "controls.selected_decor": ("viewport_overlay.theme.rcss",),
        "controls.selected_icon": ("viewport_overlay.theme.rcss",),
    }
    for token, filenames in consumers.items():
        assert f'{{"{token}"' in rml_theme_cpp
        for filename in filenames:
            resource = (resources / filename).read_text(encoding="utf-8")
            assert f"@{{{token}}}" in resource

    panel_host_theme = (resources / "panel_host.theme.rcss").read_text(
        encoding="utf-8"
    )
    assert '{"panel.host_body_decor"' in rml_theme_cpp
    assert '"decorator: none; background-color: transparent"' in rml_theme_cpp
    assert "@{panel.host_body_decor}" in panel_host_theme
    assert '"width: 1dp; height: 100%' in rml_theme_cpp


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
    assert "setOpenSubmenu(" in menu_bar_cpp
    assert "submenuIndexForElement(hit_element)," in menu_bar_cpp
    assert "childSubmenuIndexForElement(hit_element)" in menu_bar_cpp
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


def test_theme_auto_visibility_and_variant_button_width_are_capability_driven():
    preferences_panel = (
        PROJECT_ROOT / "src" / "python" / "lfs_plugins" / "preferences_panel.py"
    ).read_text(encoding="utf-8")
    preferences_rcss = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "preferences.rcss"
    ).read_text(encoding="utf-8")
    theme_cpp = (
        PROJECT_ROOT / "src" / "visualizer" / "theme" / "theme.cpp"
    ).read_text(encoding="utf-8")

    assert '{"dark", "light"}.issubset(modes) and lf.ui.supports_system_theme()' in preferences_panel
    variants_rule = _rule_body(preferences_rcss, ".preferences-theme-variants")
    assert "display: flex;" in variants_rule
    assert "flex: 0 0 200dp;" in variants_rule
    variant_rule = _rule_body(preferences_rcss, ".preferences-theme-variant")
    assert "flex: 1 1 0;" in variant_rule
    assert "min-width: 0;" in variant_rule

    assert "#elif defined(__linux__)" in theme_cpp
    assert "SDL_GetSystemTheme()" in theme_cpp
    assert "SDL_SYSTEM_THEME_LIGHT" in theme_cpp
    assert "SDL_SYSTEM_THEME_DARK" in theme_cpp


def test_all_optional_theme_gradients_reach_rml_consumers():
    theme_cpp = (
        PROJECT_ROOT / "src" / "visualizer" / "theme" / "theme.cpp"
    ).read_text(encoding="utf-8")
    rml_theme_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "rml_theme.cpp"
    ).read_text(encoding="utf-8")
    resources = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "rmlui" / "resources"
    )
    signal = json.loads(
        (
            PROJECT_ROOT
            / "src"
            / "visualizer"
            / "gui"
            / "assets"
            / "themes"
            / "signal.json"
        ).read_text(encoding="utf-8")
    )

    gradient_tokens = {
        "window_body": ("window.body_decor", "components.theme.rcss"),
        "panel_body": ("panel.body_decor", "shell.theme.rcss"),
        "window_title": ("window.title_decor", "components.theme.rcss"),
        "section_header": ("components.header_decor", "components.theme.rcss"),
        "section_header_hover": (
            "components.header_hover_decor",
            "components.theme.rcss",
        ),
        "progress": ("components.progress_fill_decor", "components.theme.rcss"),
        "scrubber_track": ("components.scrub_bg_decor", "components.theme.rcss"),
        "scrubber_fill": ("components.scrub_fill_decor", "components.theme.rcss"),
        "histogram_header": (
            "panel.histogram_hero_decor",
            "histogram_panel.theme.rcss",
        ),
        "histogram_fill": (
            "panel.histogram_fill_decor",
            "histogram_panel.theme.rcss",
        ),
        "histogram_selection": (
            "panel.histogram_fill_selected_decor",
            "histogram_panel.theme.rcss",
        ),
    }

    for mode in ("dark", "light"):
        defined = {
            key
            for key in signal["variants"][mode]["gradients"]
            if not key.startswith("_")
        }
        assert defined == set(gradient_tokens)

    for gradient, (token, resource_name) in gradient_tokens.items():
        assert f'apply_gradient("{gradient}", t.gradients.{gradient})' in theme_cpp
        assert f"hashGradient(seed, gradients.{gradient})" in rml_theme_cpp
        assert f"t.gradients.{gradient}" in rml_theme_cpp
        resource = (resources / resource_name).read_text(encoding="utf-8")
        assert f"@{{{token}}}" in resource


def test_signal_family_keeps_a_distinct_visual_language():
    themes = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "assets" / "themes"
    )
    signal = json.loads((themes / "signal.json").read_text(encoding="utf-8"))
    lichtfeld = json.loads((themes / "lichtfeld.json").read_text(encoding="utf-8"))

    for mode in ("dark", "light"):
        variant = signal["variants"][mode]
        palette = variant["palette"]
        progress = variant["gradients"]["progress"]
        assert progress["start"] == palette["primary"]
        assert progress["end"] == palette["secondary"]
        assert variant["sizes"]["window_rounding"] >= 12.0
        assert variant["sizes"]["frame_rounding"] >= 7.0

    signal_day = signal["variants"]["light"]
    lichtfeld_light = lichtfeld["variants"]["light"]
    canvas_distance = sum(
        abs(a - b)
        for a, b in zip(
            signal_day["palette"]["background"][:3],
            lichtfeld_light["palette"]["background"][:3],
        )
    )
    assert canvas_distance >= 0.10
    assert signal_day["sizes"]["window_rounding"] >= (
        lichtfeld_light["sizes"]["window_rounding"] + 4.0
    )


def test_sequencer_quality_scrubber_uses_shared_theme_gradients():
    sequencer_rml = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "sequencer.rml"
    ).read_text(encoding="utf-8")
    sequencer_rcss = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "sequencer.rcss"
    ).read_text(encoding="utf-8")
    sequencer_theme = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "sequencer.theme.rcss"
    ).read_text(encoding="utf-8")

    assert 'id="quality-scrub" class="scrub-field"' in sequencer_rml
    assert 'id="quality-fill" class="scrub-field-fill"' in sequencer_rml
    quality_scrubber = sequencer_rcss.split("/* ── Quality Scrub Field", 1)[1].split(
        "#btn-equirect", 1
    )[0]
    assert "decorator: vertical-gradient" not in quality_scrubber
    assert "decorator: horizontal-gradient" not in quality_scrubber
    assert "color: #cdd6f4;" not in quality_scrubber
    assert sequencer_theme.count("@{panel.body_decor}") == 2
    assert "@{components.header_decor}" in sequencer_theme

    assert "@media (max-width: 1620dp)" not in sequencer_rcss
    responsive_rules = sequencer_rcss.split("@media", 1)[1]
    assert ".quality-group" not in responsive_rules
    assert "#quality-scrub" not in responsive_rules
    assert ".speed-text" not in responsive_rules
    assert "#btn-speed" not in responsive_rules
    assert "#sequence-fps-field" not in responsive_rules
    assert "#resolution-field" not in responsive_rules


def test_asset_manager_palette_is_fully_theme_driven():
    resources = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
    )
    base_rcss = (resources / "asset_manager.rcss").read_text(encoding="utf-8")
    theme_rcss = (resources / "asset_manager.theme.rcss").read_text(encoding="utf-8")
    rml = (resources / "asset_manager.rml").read_text(encoding="utf-8")

    # Layout resources must not smuggle in a dark fallback palette. Entity
    # references such as &#215; are not color literals.
    assert re.search(r"(?<!&)#[0-9a-fA-F]{3,8}\b", base_rcss) is None
    assert re.search(r"\brgba?\s*\(", base_rcss) is None
    assert re.search(r"(?<!&)#[0-9a-fA-F]{3,8}\b", rml) is None

    required_theme_rules = {
        "#asset-popup": ("@{panel.body_decor}", "@{text}", "@{border}"),
        ".asset-button": ("@{surface_bright}", "@{border}", "@{text}"),
        ".asset-import-button": ("@{blend(surface,primary,button.tint_normal)}",),
        ".asset-icon-grid > span,\n.asset-icon-list > span": ("@{text}",),
        ".asset-refresh-button img,\n.asset-folder-menu img,\n.asset-card-menu img": (
            "@{alpha(text,0.90)}",
        ),
        "#asset-sidebar": ("@{alpha(background,0.32)}", "@{border}"),
        ".asset-card": ("@{surface_bright}", "@{border}"),
        ".asset-list-row": ("@{surface_bright}", "@{border}", "@{text}"),
        "#asset-info-panel": ("@{alpha(background,0.32)}", "@{border}"),
        ".asset-info-warning": ("@{alpha(error,0.10)}", "@{error}"),
    }
    for selector, expected_tokens in required_theme_rules.items():
        body = theme_rcss.split(f"{selector} {{", 1)[1].split("\n}", 1)[0]
        for token in expected_tokens:
            assert token in body

    assert 'class="asset-add-folder-glyph"' in rml
    assert "stroke=" not in rml


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


def test_asset_manager_launcher_is_not_duplicated_in_scene_header():
    scene_rml = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "scene_tree.rml"
    ).read_text(encoding="utf-8")
    scene_cpp = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "scene_panel_native.cpp"
    ).read_text(encoding="utf-8")

    assert 'id="asset-manager-button"' not in scene_rml
    assert 'id="asset-manager-icon"' not in scene_rml
    assert '"lfs.asset_manager"' not in scene_cpp


def test_scene_filter_clear_resets_the_live_input_control():
    scene_cpp = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "scene_panel_native.cpp"
    ).read_text(encoding="utf-8")

    assert 'id == "filter-clear" || current_id == "filter-clear"' in scene_cpp
    assert 'input->SetValue("")' in scene_cpp
    assert "input ? input->GetValue()" in scene_cpp
    assert 'filter_input_el_->SetAttribute("value", "")' not in scene_cpp


def test_scene_tree_aligns_hierarchy_and_keeps_row_actions_trailing():
    scene_rcss = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "scene_tree.rcss"
    ).read_text(encoding="utf-8")
    scene_graph_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "elements"
        / "scene_graph_element.cpp"
    ).read_text(encoding="utf-8")

    scene_view_rule = _rule_body(scene_rcss, "#scene-view")
    assert "margin-left: 6dp;" in scene_view_rule
    assert "margin-right: 6dp;" in scene_view_rule

    tree_rule = _rule_body(scene_rcss, "#tree-container")
    assert "overflow-y: scroll;" in tree_rule

    node_name_rule = _rule_body(scene_rcss, ".node-name")
    assert "flex-grow: 1;" in node_name_rule
    assert "flex-shrink: 1;" in node_name_rule
    trash_rule = _rule_body(scene_rcss, ".row-icon.trash-icon")
    assert "margin-left: 2dp;" in trash_rule
    assert "flex-shrink: 0;" in trash_rule
    training_toggle_rule = _rule_body(scene_rcss, ".row-icon.training-toggle-icon")
    assert "flex-shrink: 0;" in training_toggle_rule

    checkbox = scene_graph_cpp.index('selection_checkbox->SetClass("tree-checkbox", true)')
    expand = scene_graph_cpp.index('expand->SetClass("expand-toggle", true)')
    visibility = scene_graph_cpp.index('vis_icon->SetAttribute("data-action", "toggle-vis")')
    name = scene_graph_cpp.index('node_name->SetClass("node-name", true)')
    training_toggle = scene_graph_cpp.index(
        'training_toggle_icon->SetAttribute("data-action", "toggle-training")'
    )
    delete = scene_graph_cpp.index('trash_icon->SetAttribute("data-action", "delete")')
    assert checkbox < expand < name < visibility < training_toggle < delete
    assert 'actions->SetClass("row-actions", true)' in scene_graph_cpp
    assert scene_graph_cpp.count('SetClass("row-action-slot", true)') == 3
    row_actions_rule = _rule_body(scene_rcss, ".row-actions")
    assert "margin-right: -11dp;" in row_actions_rule
    assert "padding-right: 3dp;" in row_actions_rule
    assert "width: 23dp;" in _rule_body(scene_rcss, ".row-action-slot")


def test_scene_tree_inline_actions_cannot_trigger_row_double_click_activation():
    scene_graph_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "elements"
        / "scene_graph_element.cpp"
    ).read_text(encoding="utf-8")

    dblclick = scene_graph_cpp.split('} else if (type == "dblclick") {', 1)[1].split(
        '} else if (type == "mousedown") {', 1
    )[0]
    action_guard = dblclick.index('GetAttribute<Rml::String>("data-action", "")')
    activation = dblclick.index("activateNode(node_id)")
    assert action_guard < activation
    assert "event.StopPropagation();" in dblclick


def test_scene_tree_removes_models_header_space_and_confirms_all_node_deletes():
    scene_graph_hpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "elements"
        / "scene_graph_element.hpp"
    ).read_text(encoding="utf-8")
    scene_graph_cpp = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "elements"
        / "scene_graph_element.cpp"
    ).read_text(encoding="utf-8")

    assert "kHeaderHeightDpInt" not in scene_graph_hpp
    assert "kHeaderHeightDp" not in scene_graph_hpp
    assert 'header_el_->SetProperty("display", "none")' in scene_graph_cpp
    assert "void SceneGraphElement::requestDeleteNodes" in scene_graph_cpp
    assert "gui->enqueueModal(std::move(request))" in scene_graph_cpp
    assert "getNodeIdByUuid(uuid)" in scene_graph_cpp
    assert "std::vector<core::NodeId> live_ids" in scene_graph_cpp
    assert "live_ids.size() != node_uuids.size()" in scene_graph_cpp
    assert "removeNodesByIdsWithResult(live_ids" in scene_graph_cpp
    assert "live_names" not in scene_graph_cpp
    request_body = scene_graph_cpp.split("void SceneGraphElement::requestDeleteNodes", 1)[1].split(
        "void SceneGraphElement::deleteSelectedNodes", 1
    )[0]
    assert "the complete selection is no longer removable" in request_body
    assert "Scene node deletion request aborted" in request_body
    assert "continue;" not in request_body
    assert scene_graph_cpp.count("requestDeleteNodes({node_id})") == 2
    assert "requestDeleteNodes(ids);" in scene_graph_cpp
    assert "deleteEnabledSelectedNodeIds" not in scene_graph_cpp
    bulk_delete = scene_graph_cpp.split('kind == "delete_selected"', 1)[1].split(
        '} else if (kind == "set_easing"', 1
    )[0]
    assert "requestDeleteSelection();" in bulk_delete
    delete_key = scene_graph_cpp.split("case Rml::Input::KI_DELETE:", 1)[1].split(
        "case Rml::Input::KI_ESCAPE:", 1
    )[0]
    assert "requestDeleteSelection();" in delete_key
    assert 'data-action", "toggle-training"' in scene_graph_cpp


def test_scene_tree_delete_key_uses_selection_gated_confirmation_path():
    input_controller = (
        PROJECT_ROOT / "src" / "visualizer" / "input" / "input_controller.cpp"
    ).read_text(encoding="utf-8")
    gui_manager = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "gui_manager.cpp"
    ).read_text(encoding="utf-8")
    scene_panel = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "scene_panel_native.cpp"
    ).read_text(encoding="utf-8")
    scene_graph = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "elements"
        / "scene_graph_element.cpp"
    ).read_text(encoding="utf-8")

    delete_case = input_controller.index("case input::Action::DELETE_NODE:")
    confirmation_route = input_controller.index(
        "gui->requestDeleteSceneSelectionIfAvailable()", delete_case
    )
    legacy_remove = input_controller.index("cmd::RemovePLY", delete_case)
    assert delete_case < confirmation_route < legacy_remove
    assert "requestDeleteSceneSelectionIfAvailable" in gui_manager
    assert "requestDeleteSelectionIfAvailable" in scene_panel
    assert "tree_el_->selectedCount() == 0" in scene_panel
    assert "if (active_tab_ == Tab::Scene)" in scene_panel
    assert "tree_el_->requestDeleteSelection();" in scene_panel
    assert "return true;" in scene_panel.split(
        "bool NativeScenePanel::requestDeleteSelectionIfAvailable()", 1
    )[1].split("void NativeScenePanel::applyPendingTreeChrome", 1)[0]
    assert "requestDeleteSelectionIfAvailable" not in scene_graph


def test_viewport_ctrl_click_toggles_nodes_without_breaking_modified_drags():
    input_controller = (
        PROJECT_ROOT / "src" / "visualizer" / "input" / "input_controller.cpp"
    ).read_text(encoding="utf-8")

    release_pick = input_controller.split(
        "selectCameraByUid(pressed_camera_frustum_id,", 1
    )[1].split(";", 1)[0]
    assert "pressed_camera_frustum_modifiers & input::KEYMOD_CTRL" in release_pick

    camera_press = input_controller.split(
        "const bool camera_selection_modifier =", 1
    )[1].split("last_click_time_ = now;", 1)[0]
    assert "input::KEYMOD_CTRL | input::KEYMOD_SHIFT" in camera_press
    assert "!camera_selection_modifier && is_double_click" in camera_press

    helper = input_controller.split(
        "void InputController::selectCameraByUid", 1
    )[1].split("bool InputController::handleFocusSelection", 1)[0]
    assert "toggle_selection" in helper
    assert "sm->getSelectedNodeIds()" in helper
    assert "sm->removeFromSelection(node->id)" in helper
    assert "sm->addToSelection(node->id)" in helper
    assert "sm->selectNode(node->id)" in helper

    rect_release = input_controller.split(
        "const int node_rect_modifiers = node_rect_modifiers_;", 1
    )[1].split("// Camera click selection", 1)[0]
    assert "node_rect_modifiers & input::KEYMOD_CTRL" in rect_release
    assert "node_rect_modifiers & input::KEYMOD_SHIFT" in rect_release
    normalized_rect_release = " ".join(rect_release.split())
    assert (
        "node_rect_modifiers & (input::KEYMOD_CTRL | input::KEYMOD_SHIFT)"
        in normalized_rect_release
    )
    assert "picked_node && ctrl_held" in rect_release
    assert "scene_manager->removeFromSelection(picked_node->id)" in rect_release
    assert "scene_manager->addToSelection(picked_node->id)" in rect_release
    assert "picked_node && shift_held" in rect_release


def test_scene_tree_duplicate_rename_opens_a_retry_or_cancel_modal():
    scene_graph = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "elements"
        / "scene_graph_element.cpp"
    ).read_text(encoding="utf-8")

    confirm_rename = scene_graph.split(
        "void SceneGraphElement::confirmRename()", 1
    )[1].split("void SceneGraphElement::cancelRename()", 1)[0]
    assert "rename_conflict_modal_pending_" in confirm_rename
    assert "RENAME_CONFLICT_TITLE" in confirm_rename
    assert "RENAME_CONFLICT_MESSAGE" in confirm_rename
    assert "Scene node rename rejected" in confirm_rename
    assert "gui->enqueueModal(std::move(request))" in confirm_rename
    assert "result.button_label == rename_button" in confirm_rename
    assert "rename_focus_pending_ = true" in confirm_rename
    assert "cancelRename()" in confirm_rename


def test_scene_tree_filter_always_reads_the_live_form_control_value():
    scene_panel = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "scene_panel_native.cpp"
    ).read_text(encoding="utf-8")

    assert 'filter_input_el_->AddEventListener("input", &listener_)' in scene_panel
    assert 'type == "input" && current == filter_input_el_' in scene_panel
    assert scene_panel.count("input->GetValue()") >= 3
    assert 'filter_input_el_->GetAttribute<Rml::String>("value"' not in scene_panel


def test_scene_tree_multi_selection_actions_are_fixed_below_the_scroll_view():
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
    scene_panel = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "scene_panel_native.cpp"
    ).read_text(encoding="utf-8")
    scene_graph = (
        PROJECT_ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "elements"
        / "scene_graph_element.cpp"
    ).read_text(encoding="utf-8")

    tree_pos = scene_rml.index('<scene-graph id="tree-container"')
    actions_pos = scene_rml.index('<div id="selection-action-bar"')
    assert tree_pos < actions_pos
    assert '<div id="scene-chip-row" class="scene-chip-row">' in scene_rml
    assert 'id="summary-model-chip"' in scene_rml
    assert 'id="summary-node-chip"' in scene_rml
    assert 'id="summary-selection-chip"' not in scene_rml
    assert scene_rml.count('id="selection-visibility"') == 1
    assert scene_rml.count('id="selection-training"') == 1
    assert "selection-action-separator" not in scene_rml
    count_pos = scene_rml.index('id="selection-action-count"')
    clear_pos = scene_rml.index('id="selection-clear"')
    spacer_pos = scene_rml.index('class="selection-action-spacer"')
    visibility_pos = scene_rml.index('id="selection-visibility"')
    assert count_pos < clear_pos < spacer_pos < visibility_pos
    action_rule = _rule_body(scene_rcss, ".selection-action-bar")
    assert "flex-shrink: 0;" in action_rule
    assert "height: 27dp;" in action_rule
    count_rule = _rule_body(scene_rcss, ".selection-action-count")
    assert "height: 22dp;" in count_rule
    assert "line-height: 22dp;" in count_rule
    button_rule = _rule_body(scene_rcss, ".selection-action-button")
    assert "display: flex;" in button_rule
    assert "align-items: center;" in button_rule
    assert "justify-content: center;" in button_rule
    assert "box-sizing: border-box;" in button_rule
    assert "padding: 0;" in button_rule
    tree_scrollbar_rule = _rule_body(scene_rcss, "scene-graph scrollbarvertical")
    assert "width: 4dp;" in tree_scrollbar_rule
    tree_slider_rule = _rule_body(scene_rcss, "scene-graph scrollbarvertical sliderbar")
    assert "min-height: 12dp;" in tree_slider_rule
    assert "state.count > 0" in scene_panel
    assert "!state.all_training_compatible" in scene_panel
    assert "!state.all_delete_enabled" in scene_panel
    assert "requestDeleteSelection()" in scene_panel
    assert "training_mixed" in scene_graph
    assert 'std::format("{} / {}"' in scene_graph
    assert 'SetAttribute("data-action", "toggle-select")' in scene_graph
    assert "collectCheckboxSelectionIds(child_id, ids)" in scene_graph
    assert "checkboxState(child_id)" in scene_graph
    checkbox_rule = _rule_body(scene_rcss, ".tree-checkbox")
    assert "position: absolute;" in checkbox_rule
    assert "left: 5dp;" in checkbox_rule
    assert "width: 9dp;" in checkbox_rule
    assert "height: 9dp;" in checkbox_rule
    assert "opacity:" not in checkbox_rule
    assert ".tree-row:hover .tree-checkbox" not in scene_rcss
    assert "(selection_markers_visible_ ? 17 : 4) + depth * 16" in scene_graph
    assert 'selection_markers_visible_ ? "inline-block" : "none"' in scene_graph
    assert "border-radius: 5dp;" in checkbox_rule
    assert 'prefixedAction("select_hierarchy")' in scene_graph
    assert "selectHierarchyFromSelection()" in scene_graph
    assert "if (ctrl && shift)" in scene_graph
    assert "selectHierarchy(node_id);" in scene_graph
    assert "case Rml::Input::KI_SPACE:" not in scene_graph
    assert "previewCameraNode" not in scene_graph
    input_bindings = (
        PROJECT_ROOT / "src" / "visualizer" / "input" / "input_bindings.cpp"
    ).read_text(encoding="utf-8")
    assert "Action::SELECT_ALL_SCENE_NODES" in input_bindings
    assert "MODIFIER_CTRL | MODIFIER_SHIFT" in input_bindings
    assert 'id="selection-clear"' in scene_rml
    assert "clearSelectedNodes()" in scene_panel
    assert "const core::NodeId toggled = current != core::NULL_NODE ? current : target" in scene_graph
    assert 'is_panel_enabled("lfs.image_preview")' in scene_graph


def test_scene_graph_selection_markers_are_optional_user_preferences():
    preferences_hpp = (PROJECT_ROOT / "src" / "visualizer" / "preferences.hpp").read_text(encoding="utf-8")
    preferences_cpp = (PROJECT_ROOT / "src" / "visualizer" / "preferences.cpp").read_text(encoding="utf-8")
    preferences_rml = (
        PROJECT_ROOT / "src" / "visualizer" / "gui" / "rmlui" / "resources" / "preferences.rml"
    ).read_text(encoding="utf-8")
    panel = (PROJECT_ROOT / "src" / "python" / "lfs_plugins" / "preferences_panel.py").read_text(encoding="utf-8")

    assert "setSceneGraphSelectionMarkers(bool enabled)" in preferences_hpp
    assert 'values["scene_graph_selection_markers"] = enabled' in preferences_cpp
    assert 'value("scene_graph_selection_markers", false)' in preferences_cpp
    assert 'data-checked="scene_graph_selection_markers"' in preferences_rml
    assert 'toggle_section(\'scene_graph\')' in preferences_rml
    assert '@tr:preferences.scene_graph' in preferences_rml
    assert "lf.ui.set_scene_graph_selection_markers" in panel
