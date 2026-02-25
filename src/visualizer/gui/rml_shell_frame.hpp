/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rml_fbo.hpp"
#include <string>
#include <imgui.h>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis::gui {

    class RmlUIManager;

    struct ShellRegions {
        ImVec2 menu_pos{0, 0};
        ImVec2 menu_size{0, 0};
        ImVec2 right_panel_pos{0, 0};
        ImVec2 right_panel_size{0, 0};
        ImVec2 status_pos{0, 0};
        ImVec2 status_size{0, 0};
    };

    class RmlShellFrame {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void render(const ShellRegions& regions);

    private:
        void updateTheme();
        std::string generateThemeRCSS() const;

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        Rml::Element* menu_region_ = nullptr;
        Rml::Element* right_panel_region_ = nullptr;
        Rml::Element* status_region_ = nullptr;

        RmlFBO fbo_;

        std::string last_theme_;
        std::string base_rcss_;
    };

} // namespace lfs::vis::gui
