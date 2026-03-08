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
        std::string idname;
        std::string label;
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
        void render(const RightPanelLayout& layout,
                    const std::vector<TabSnapshot>& tabs,
                    const std::string& active_tab);

        bool wantsInput() const { return wants_input_; }
        bool needsAnimationFrame() const;
        CursorRequest getCursorRequest() const;

        std::function<void(const std::string&)> on_tab_changed;
        std::function<void(float)> on_splitter_delta;
        std::function<void()> on_splitter_end;
        std::function<void(float)> on_resize_delta;
        std::function<void()> on_resize_end;

    private:
        bool updateTheme();
        std::string generateThemeRCSS(const lfs::vis::Theme& t) const;
        bool syncTabData(const std::vector<TabSnapshot>& tabs, const std::string& active_tab);

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        Rml::Element* resize_handle_el_ = nullptr;
        Rml::Element* left_border_el_ = nullptr;
        Rml::Element* splitter_el_ = nullptr;
        Rml::Element* tab_bar_el_ = nullptr;
        Rml::Element* tab_separator_el_ = nullptr;

        RmlFBO fbo_;
        Rml::DataModelHandle tab_model_;
        std::vector<TabSnapshot> tabs_;
        Rml::String active_tab_;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;
        bool wants_input_ = false;

        bool splitter_dragging_ = false;
        float drag_start_y_ = 0;

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
