/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rml_right_panel.hpp"

#include <RmlUi/Core/DataModelHandle.h>
#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis::gui {

    struct BottomDockLayout {
        glm::vec2 pos{0, 0};
        glm::vec2 size{0, 0};
        float grip_h = 0.0f;
        float tab_bar_h = 0.0f;
        float separator_h = 0.0f;
        bool grip_hovered = false;
        bool grip_active = false;
    };

    class RmlBottomDock {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void setVisible(bool visible);
        void processInput(const BottomDockLayout& layout, const PanelInputState& input);
        void reloadResources();
        void render(const BottomDockLayout& layout,
                    const std::vector<TabSnapshot>& tabs,
                    const std::string& active_tab,
                    float screen_x, float screen_y, int screen_w, int screen_h);
        void blurFocus();

        bool wantsInput() const { return wants_input_; }
        bool wantsKeyboard() const { return wants_keyboard_; }
        bool needsAnimationFrame() const;
        [[nodiscard]] std::string animationDemandDescription() const;
        CursorRequest getCursorRequest() const;

        std::function<void(const std::string&)> on_tab_changed;
        std::function<void(const std::string&)> on_tab_closed;

    private:
        bool updateTheme();
        bool syncTabData(const std::vector<TabSnapshot>& tabs, const std::string& active_tab);
        bool syncTabScrollState();
        void syncTabNavigation();
        void scrollTabs(float delta);

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* dock_grip_el_ = nullptr;
        Rml::Element* tab_bar_el_ = nullptr;
        Rml::Element* tab_strip_viewport_el_ = nullptr;
        Rml::Element* tab_separator_el_ = nullptr;

        Rml::DataModelHandle tab_model_;
        std::vector<TabSnapshot> tabs_;
        Rml::String active_tab_;
        float tab_scroll_left_ = 0.0f;
        bool tabs_overflow_ = false;
        bool can_scroll_tabs_left_ = false;
        bool can_scroll_tabs_right_ = false;
        bool last_grip_hovered_ = false;
        bool last_grip_active_ = false;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;
        bool wants_input_ = false;
        bool wants_keyboard_ = false;
        Rml::Element* last_blurred_focus_ = nullptr;
        Rml::Element* last_hover_element_ = nullptr;
        bool last_over_interactive_ = false;
        bool rml_pointer_inside_ = false;
        CursorRequest cursor_request_{};
        float prev_mouse_x_ = 0.0f;
        float prev_mouse_y_ = 0.0f;
        bool render_needed_ = true;
        bool input_dirty_ = false;
        bool visible_ = false;
        int last_fbo_w_ = 0;
        int last_fbo_h_ = 0;
        CachedVulkanContextRender direct_cache_;
    };

} // namespace lfs::vis::gui
