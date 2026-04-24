/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rml_fbo.hpp"
#include <RmlUi/Core/DataModelHandle.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
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

    struct TabSnapshot {
        std::string id;
        std::string label;
        std::string dom_id;
        bool operator==(const TabSnapshot&) const = default;
    };

    enum class CursorRequest : uint8_t;
    struct PanelInputState;

    struct RightPanelLayout {
        glm::vec2 pos{0, 0};
        glm::vec2 size{0, 0};
        float scene_h = 0;
        float splitter_h = 6.0f;
    };

    class RmlRightPanel {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();

        void processInput(const RightPanelLayout& layout, const PanelInputState& input);
        void reloadResources();
        void render(const RightPanelLayout& layout,
                    const std::vector<TabSnapshot>& tabs,
                    const std::string& active_tab,
                    float screen_x, float screen_y,
                    int screen_w, int screen_h);
        void blurFocus();

        bool wantsInput() const { return wants_input_; }
        bool wantsKeyboard() const { return wants_keyboard_; }
        bool needsAnimationFrame() const;
        CursorRequest getCursorRequest() const;

        std::function<void(const std::string&)> on_tab_changed;
        std::function<void(float)> on_splitter_delta;
        std::function<void()> on_splitter_end;
        std::function<void(float)> on_resize_delta;
        std::function<void()> on_resize_end;

    private:
        bool updateTheme();
        bool syncTabData(const std::vector<TabSnapshot>& tabs, const std::string& active_tab);
        bool syncTabScrollState();
        void syncTabNavigation();
        void scrollTabs(float delta);

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        Rml::Element* resize_handle_el_ = nullptr;
        Rml::Element* left_border_el_ = nullptr;
        Rml::Element* splitter_el_ = nullptr;
        Rml::Element* tab_bar_el_ = nullptr;
        Rml::Element* tab_strip_viewport_el_ = nullptr;
        Rml::Element* tab_separator_el_ = nullptr;

        RmlFBO fbo_;
        Rml::DataModelHandle tab_model_;
        std::vector<TabSnapshot> tabs_;
        Rml::String active_tab_;
        float tab_scroll_left_ = 0.0f;
        bool tabs_overflow_ = false;
        bool can_scroll_tabs_left_ = false;
        bool can_scroll_tabs_right_ = false;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;
        bool wants_input_ = false;
        bool wants_keyboard_ = false;

        bool splitter_dragging_ = false;

        bool resize_dragging_ = false;

        CursorRequest cursor_request_{};
        float prev_mouse_x_ = 0;
        float prev_mouse_y_ = 0;

        bool render_needed_ = true;
        int last_fbo_w_ = 0;
        int last_fbo_h_ = 0;
        float last_scene_h_ = -1.0f;
        float last_splitter_h_ = -1.0f;
        bool input_dirty_ = false;
        bool last_over_interactive_ = false;
    };

} // namespace lfs::vis::gui
