/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"

#include <RmlUi/Core/DataModelHandle.h>
#include <cstddef>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    class RmlUIManager;

    struct MenuItemDesc {
        enum class Type { Operator,
                          Separator,
                          SubMenuBegin,
                          SubMenuEnd,
                          Toggle,
                          ShortcutItem,
                          Item };
        Type type;
        std::string label;
        std::string operator_id;
        std::string shortcut;
        bool enabled = true;
        bool selected = false;
        int callback_index = -1;
    };

    struct MenuDropdownContent {
        std::string menu_idname;
        std::vector<MenuItemDesc> items;
    };

    struct MenuLabelView {
        int index = -1;
        std::string label;
        bool active = false;
    };

    struct MenuToolbarButtonView {
        std::string button_id;
        std::string action;
        std::string value;
        std::string icon_src;
        std::string tooltip_key;
        std::string tooltip_text;
        bool selected = false;
        bool operator==(const MenuToolbarButtonView&) const = default;
    };

    struct MenuDropdownLeafView {
        std::string label;
        std::string action;
        std::string operator_id;
        std::string shortcut;
        std::string checkmark;
        bool enabled = true;
        bool separator_before = false;
        bool has_shortcut = false;
        bool show_checkmark = false;
        bool is_label = false;
        int callback_index = -1;
    };

    struct MenuDropdownRootView {
        int index = -1;
        std::string label;
        std::string action;
        std::string operator_id;
        std::string shortcut;
        std::string checkmark;
        bool enabled = true;
        bool separator_before = false;
        bool has_shortcut = false;
        bool show_checkmark = false;
        bool has_children = false;
        bool submenu_open = false;
        int callback_index = -1;
        std::vector<MenuDropdownLeafView> children;
    };

    class RmlMenuBar {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void draw(int screen_w, int screen_h);
        void updateLabels(const std::vector<std::string>& labels,
                          const std::vector<std::string>& idnames);
        void reloadResources();
        void processInput(const PanelInputState& input);
        void setViewportRightEdge(float x) { viewport_right_edge_ = x; }
        void setUiHidden(bool hidden);
        void suspend();
        bool wantsInput() const { return wants_input_; }
        bool isOpen() const { return open_menu_index_ >= 0; }
        float barHeight() const;

        // Keeps the render-on-demand loop ticking while a tooltip is counting
        // down so it reveals on time without needing a mouse jiggle.
        [[nodiscard]] bool needsAnimationFrame() const { return tooltip_.revealDue(); }

    private:
        bool updateTheme();
        void rebuildLabels();
        void syncActiveLabelState();
        void openDropdown(int index);
        void closeDropdown();
        void rebuildDropdownDOM();
        void setOpenSubmenu(int index);
        Rml::Element* dropdownElementAtPoint(float x, float y) const;
        int submenuIndexForElement(Rml::Element* element) const;
        void rebuildToolbarButtons();
        void dispatchToolbarAction(const std::string& action, const std::string& value);
        Rml::Element* toolbarButtonAtPoint(float x, float y) const;
        void updateTitlebarDragRegion(int bar_height_px);
        void clearTitlebarDragRegion();

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::DataModelHandle menu_model_;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;

        std::vector<std::string> current_labels_;
        std::vector<std::string> current_idnames_;
        std::vector<MenuLabelView> menu_labels_;
        std::vector<MenuDropdownRootView> dropdown_items_;
        std::vector<MenuToolbarButtonView> camera_buttons_;
        std::vector<MenuToolbarButtonView> render_buttons_;
        std::vector<MenuToolbarButtonView> projection_buttons_;
        int active_index_ = -1;

        Rml::Element* menu_items_ = nullptr;
        Rml::Element* dropdown_container_ = nullptr;
        Rml::Element* dropdown_popup_ = nullptr;
        Rml::Element* dropdown_overlay_ = nullptr;
        Rml::Element* brand_logo_ = nullptr;
        Rml::Element* menu_toolbar_ = nullptr;
        Rml::Element* menu_window_controls_ = nullptr;
        Rml::Element* menu_window_split_view_ = nullptr;
        Rml::Element* menu_window_toggle_ui_ = nullptr;
        Rml::Element* menu_window_maximize_ = nullptr;
        Rml::Element* body_el_ = nullptr;
        RmlTooltipController tooltip_;
        float viewport_right_edge_ = 0.0f;
        float applied_toolbar_right_ = -1.0f;
        bool ui_hidden_ = false;
        bool last_window_split_view_ = false;
        bool last_ui_hidden_ = false;
        bool last_window_maximized_ = false;

        int open_menu_index_ = -1;
        int open_submenu_index_ = -1;
        std::string open_menu_idname_;
        bool wants_input_ = false;
        bool render_needed_ = true;
        bool mouse_pos_valid_ = false;
        int last_mouse_x_ = 0;
        int last_mouse_y_ = 0;
        int last_hovered_label_ = -1;
        bool last_toolbar_hovered_ = false;
        int last_ctx_w_ = 0;
        int last_ctx_h_ = 0;
        int last_document_h_ = 0;
        CachedVulkanContextRender direct_cache_;

        float bar_height_ = 30.0f;
    };

} // namespace lfs::vis::gui
