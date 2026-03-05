/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rml_fbo.hpp"
#include <glm/glm.hpp>
#include <string>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis::gui {

    class RmlUIManager;

    class RmlViewportOverlay {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void setViewportBounds(glm::vec2 pos, glm::vec2 size);
        void render();
        void processInput();
        bool wantsInput() const { return wants_input_; }

    private:
        void updateTheme();
        std::string generateThemeRCSS() const;

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        RmlFBO fbo_;

        glm::vec2 vp_pos_{0, 0};
        glm::vec2 vp_size_{0, 0};
        std::string last_theme_;
        std::string base_rcss_;
        bool wants_input_ = false;
        bool doc_registered_ = false;
    };

} // namespace lfs::vis::gui
