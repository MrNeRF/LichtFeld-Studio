/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glad/glad.h>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis::gui {

    class RmlUIManager;

    class RmlMenuBar {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void draw();
        void updateLabels(const std::vector<std::string>& labels);
        void setActiveIndex(int index);

    private:
        void initFBO(int w, int h);
        void destroyFBO();
        void updateTheme();
        void rebuildLabels();
        std::string generateThemeRCSS() const;

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        GLuint fbo_ = 0;
        GLuint fbo_texture_ = 0;
        GLuint fbo_depth_stencil_ = 0;
        int fbo_w_ = 0, fbo_h_ = 0;

        std::string last_theme_;
        std::string base_rcss_;

        std::vector<std::string> current_labels_;
        int active_index_ = -1;

        Rml::Element* menu_items_ = nullptr;
        Rml::Element* bottom_border_ = nullptr;
    };

} // namespace lfs::vis::gui
