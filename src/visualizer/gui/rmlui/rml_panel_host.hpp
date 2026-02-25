/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"
#include <core/export.hpp>
#include <glad/glad.h>
#include <mutex>
#include <string>
#include <vector>
#include <imgui.h>

namespace Rml {
    class Context;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis::gui {

    class RmlUIManager;

    class LFS_VIS_API RmlPanelHost {
    public:
        RmlPanelHost(RmlUIManager* manager, std::string context_name, std::string rml_path);
        ~RmlPanelHost();

        RmlPanelHost(const RmlPanelHost&) = delete;
        RmlPanelHost& operator=(const RmlPanelHost&) = delete;

        void draw(const PanelDrawContext& ctx);

        Rml::ElementDocument* getDocument() { return document_; }
        Rml::Context* getContext() { return rml_context_; }
        bool isDocumentLoaded() const { return document_ != nullptr; }

        static void pushTextInput(const std::string& text);

    private:
        static std::vector<uint32_t> drainTextInput();
        void initFBO(int width, int height);
        void destroyFBO();
        void forwardInput(float panel_x, float panel_y);
        void syncThemeProperties();
        std::string generateThemeRCSS() const;

        RmlUIManager* manager_;
        std::string context_name_;
        std::string rml_path_;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        std::string base_rcss_;
        ImVec4 last_synced_text_{};
        bool has_text_focus_ = false;

        GLuint fbo_ = 0;
        GLuint fbo_texture_ = 0;
        GLuint fbo_depth_stencil_ = 0;
        int fbo_width_ = 0;
        int fbo_height_ = 0;
    };

} // namespace lfs::vis::gui
