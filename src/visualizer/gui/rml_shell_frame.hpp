/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rmlui_manager.hpp"

#include <cstddef>
#include <string>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    struct ShellRect {
        float x = 0;
        float y = 0;
        float w = 0;
        float h = 0;
    };

    struct ShellRegions {
        ShellRect screen;
        ShellRect menu;
        ShellRect right_panel;
        ShellRect status;
    };

    class RmlShellFrame {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void reloadResources();
        void render(const ShellRegions& regions);

    private:
        struct LayoutSignature {
            int width = 0;
            int height = 0;
            int menu_top = 0;
            int menu_height = 0;
            int work_top = 0;
            int right_width = 0;
            int right_height = 0;
            int status_height = 0;

            bool operator==(const LayoutSignature&) const = default;
        };

        bool updateTheme();

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        Rml::Element* menu_region_ = nullptr;
        Rml::Element* right_panel_region_ = nullptr;
        Rml::Element* status_region_ = nullptr;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;

        LayoutSignature last_layout_signature_;
        bool has_layout_signature_ = false;
        bool render_needed_ = true;
        CachedVulkanContextRender direct_cache_;
    };

} // namespace lfs::vis::gui
