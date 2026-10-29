/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/gui_input.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"

#include <RmlUi/Core/DataModelHandle.h>
#include <SDL3/SDL_events.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis {
    class InputControllerFocusTest_WalkRejectionDoesNotEscapeNativeDispatch_Test;
    struct Theme;
    struct ProjectDisplayInfo;
} // namespace lfs::vis
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
        std::string tooltip;
        bool enabled = true;
        bool selected = false;
        int callback_index = -1;
    };

    struct MenuDropdownContent {
        std::string menu_idname;
        std::vector<MenuItemDesc> items;
    };

    struct AccessKeyLabel {
        std::string label_before;
        std::string access_mark;
        std::string label_after;
        char access_key = 0;
    };

    struct MenuLabelView {
        int index = -1;
        std::string label;
        bool active = false;
        AccessKeyLabel access;
        bool logo = false;
    };

    struct MenuToolbarButtonView {
        bool enabled = true;
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
        int index = -1;
        std::string label;
        std::string action;
        std::string operator_id;
        std::string shortcut;
        std::string checkmark;
        std::string tooltip;
        bool enabled = true;
        bool separator_before = false;
        bool has_shortcut = false;
        bool show_checkmark = false;
        bool is_label = false;
        bool keyboard_active = false;
        int callback_index = -1;
        AccessKeyLabel access;
    };

    struct MenuDropdownGrandchildView {
        int index = -1;
        std::string label;
        std::string action;
        std::string operator_id;
        std::string shortcut;
        std::string checkmark;
        std::string tooltip;
        bool enabled = true;
        bool separator_before = false;
        bool has_shortcut = false;
        bool show_checkmark = false;
        bool has_children = false;
        bool submenu_open = false;
        bool keyboard_active = false;
        int callback_index = -1;
        AccessKeyLabel access;
        std::vector<MenuDropdownLeafView> children = {};
    };

    struct MenuDropdownChildView {
        int index = -1;
        std::string label;
        std::string action;
        std::string operator_id;
        std::string shortcut;
        std::string checkmark;
        std::string tooltip;
        bool enabled = true;
        bool separator_before = false;
        bool has_shortcut = false;
        bool show_checkmark = false;
        bool has_children = false;
        bool submenu_open = false;
        bool keyboard_active = false;
        int callback_index = -1;
        AccessKeyLabel access;
        std::vector<MenuDropdownGrandchildView> children = {};
    };

    struct MenuDropdownRootView {
        int index = -1;
        std::string label;
        std::string action;
        std::string operator_id;
        std::string shortcut;
        std::string checkmark;
        std::string tooltip;
        bool enabled = true;
        bool separator_before = false;
        bool has_shortcut = false;
        bool show_checkmark = false;
        bool has_children = false;
        bool submenu_open = false;
        bool keyboard_active = false;
        int callback_index = -1;
        AccessKeyLabel access;
        std::vector<MenuDropdownChildView> children;
    };

    class LFS_VIS_API RmlMenuBar {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void draw(int screen_w, int screen_h);
        void updateLabels(const std::vector<std::string>& labels,
                          const std::vector<std::string>& idnames);
        void updateProjectDisplay(const ProjectDisplayInfo& project_display);
        void updateProjectDisplay(std::string title, std::string tooltip, bool dirty);
        void reloadResources();
        void processInput(const PanelInputState& input);
        void onPointerLeave();
        void onPointerMotion();
        void onWindowPointerEvent(const SDL_Event& event, std::uint32_t main_window_id);
        void closeDropdown();
        void setViewportRightEdge(float x) { viewport_right_edge_ = x; }
        void setUiHidden(bool hidden);
        void suspend();
        bool wantsInput() const { return wants_input_; }
        bool isOpen() const { return open_menu_index_ >= 0; }
        float barHeight() const;

        // Keeps the render-on-demand loop ticking while a tooltip is counting
        // down so it reveals on time without needing a mouse jiggle.
        [[nodiscard]] bool needsAnimationFrame() const {
            return tooltip_.revealDue() || portal_transfer_animation_active_;
        }

    private:
        friend class RmlMenuBarTestAccess;
        friend class lfs::vis::InputControllerFocusTest_WalkRejectionDoesNotEscapeNativeDispatch_Test;
        void bindModel();
        bool updateTheme();
        void rebuildLabels();
        void syncActiveLabelState();
        void openDropdown(int index, bool from_keyboard = false);
        void syncAccessKeyVisibility();
        enum class AccessKeyResult { Ignore,
                                     Handled,
                                     Decline };
        AccessKeyResult handleAccessKey(const FrameInputEvent& event);
        void rebuildDropdownDOM();
        void sizeOpenDropdowns();
        void setOpenSubmenu(int root_index, int child_index, int grandchild_index = -1);
        static std::vector<MenuDropdownRootView> buildDropdownItems(const std::vector<MenuItemDesc>& items);
        Rml::Element* dropdownElementAtPoint(float x, float y) const;
        int submenuIndexForElement(Rml::Element* element) const;
        int childSubmenuIndexForElement(Rml::Element* element) const;
        int grandchildSubmenuIndexForElement(Rml::Element* element) const;
        void rebuildToolbarButtons();
        void rebuildPortalStatus();
        void dispatchToolbarAction(const std::string& action, const std::string& value);
        Rml::Element* toolbarButtonAtPoint(float x, float y) const;
        bool projectTitleAtPoint(float x, float y) const;
        void updateCompactLayout(int screen_w, float dp_ratio);
        void updateToolbarLayout(int screen_w, float dp_ratio);
        void updateProjectTitleLayout(int screen_w, float dp_ratio);
        void updateTitlebarDragRegion(int bar_height_px);
        void clearTitlebarDragRegion();
        bool handleMenuKey(const FrameInputEvent& event, bool mouse_button_active,
                           bool window_has_keyboard_focus = true);
        void onWindowFocusLost();
        void updateKeyboardHighlight();
        void moveKeyboardHighlight(int direction, bool to_edge = false);
        bool activateKeyboardHighlight(bool submenu_only = false);

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
        std::vector<MenuToolbarButtonView> snap_buttons_;
        std::array<std::string, 5> navigation_tooltips_;
        std::uint64_t navigation_tooltip_language_generation_ = 0;
        bool has_navigation_tooltip_language_generation_ = false;
        int active_index_ = -1;
        std::string brand_logo_src_;
        std::string portal_connection_label_;
        std::string portal_connection_tooltip_;
        std::string portal_connection_icon_;
        std::string portal_connection_tone_;
        bool portal_transfer_animation_active_ = false;
        std::string gallery_progress_label_;
        std::string gallery_progress_detail_;
        std::string gallery_progress_tooltip_;
        std::string gallery_progress_width_{"0%"};
        bool gallery_has_progress_ = false;
        bool gallery_progress_indeterminate_ = false;
        std::string project_title_;
        std::string project_tooltip_;
        bool project_dirty_ = false;

        Rml::Element* menu_items_ = nullptr;
        Rml::Element* dropdown_container_ = nullptr;
        Rml::Element* dropdown_popup_ = nullptr;
        Rml::Element* dropdown_overlay_ = nullptr;
        Rml::Element* project_title_container_ = nullptr;
        Rml::Element* project_title_el_ = nullptr;
        Rml::Element* menu_toolbar_ = nullptr;
        Rml::Element* menu_window_controls_ = nullptr;
        Rml::Element* menu_window_toggle_ui_ = nullptr;
        Rml::Element* menu_window_maximize_ = nullptr;
        Rml::Element* body_el_ = nullptr;
        RmlTooltipController tooltip_;
        float viewport_right_edge_ = 0.0f;
        float applied_toolbar_right_ = -1.0f;
        float applied_project_title_left_ = -1.0f;
        float applied_project_title_width_ = -1.0f;
        bool project_title_has_room_ = false;
        bool toolbar_fits_ = true;
        bool toolbar_squeezed_ = false;
        int toolbar_squeeze_screen_w_ = -1;
        float toolbar_squeeze_menus_right_ = -1.0f;
        bool ui_hidden_ = false;
        bool last_ui_hidden_ = false;
        bool last_window_maximized_ = false;

        int open_menu_index_ = -1;
        int open_submenu_index_ = -1;
        int open_child_submenu_index_ = -1;
        int open_grandchild_submenu_index_ = -1;
        std::array<int, 4> keyboard_path_{{-1, -1, -1, -1}};
        int keyboard_depth_ = 0;
        int keyboard_focus_menu_index_ = -1;
        bool alt_pressed_ = false;
        bool alt_chord_ = false;
        bool access_alt_held_ = false;
        bool keyboard_opened_menu_ = false;
        bool access_keys_visible_ = false;
        // Keys this bar already handled through SDL dispatch since the last frame pass.
        struct DispatchKeyReceipt {
            uint64_t timestamp = 0;
            SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;
            FrameInputEventKind kind = FrameInputEventKind::KeyDown;
            bool repeat = false;
            bool matched = false;
        };
        std::vector<DispatchKeyReceipt> dispatch_key_receipts_;
        std::size_t window_focus_lost_handler_id_ = 0;
        std::string open_menu_idname_;
        bool wants_input_ = false;
        bool render_needed_ = true;
        bool mouse_pos_valid_ = false;
        bool pointer_suspended_ = false;
        int last_mouse_x_ = 0;
        int last_mouse_y_ = 0;
        int last_hovered_label_ = -1;
        bool last_toolbar_hovered_ = false;
        float last_dp_ratio_ = 0.0f;
        int last_ctx_w_ = 0;
        int last_ctx_h_ = 0;
        int last_document_h_ = 0;
        CachedUiContextRender direct_cache_;

        float bar_height_ = 30.0f;
    };

} // namespace lfs::vis::gui
