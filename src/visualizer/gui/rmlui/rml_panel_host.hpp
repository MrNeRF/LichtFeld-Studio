/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_fbo.hpp"
#include <core/export.hpp>
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

    enum class HeightMode { Fill,
                            Content };

    class LFS_VIS_API RmlPanelHost {
    public:
        RmlPanelHost(RmlUIManager* manager, std::string context_name, std::string rml_path);
        ~RmlPanelHost();

        RmlPanelHost(const RmlPanelHost&) = delete;
        RmlPanelHost& operator=(const RmlPanelHost&) = delete;

        void draw(const PanelDrawContext& ctx);
        void drawDirect(float x, float y, float w, float h);
        bool ensureContext();

        void setHeightMode(HeightMode mode) { height_mode_ = mode; }
        HeightMode getHeightMode() const { return height_mode_; }
        float getContentHeight() const { return last_content_height_; }
        void markContentDirty() { content_dirty_ = true; }
        void setForeground(bool fg) { foreground_ = fg; }

        Rml::ElementDocument* getDocument() { return document_; }
        Rml::Context* getContext() { return rml_context_; }
        bool isDocumentLoaded() const { return document_ != nullptr; }

        static void pushTextInput(const std::string& text);

    private:
        static std::vector<uint32_t> drainTextInput();
        void forwardInput(float panel_x, float panel_y);
        void syncThemeProperties();
        std::string generateThemeRCSS() const;

        RmlUIManager* manager_;
        std::string context_name_;
        std::string rml_path_;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        HeightMode height_mode_ = HeightMode::Fill;
        float last_content_height_ = 0.0f;
        int last_measure_w_ = 0;
        bool content_dirty_ = true;

        std::string base_rcss_;
        ImVec4 last_synced_text_{};
        bool has_text_focus_ = false;

        bool foreground_ = false;
        RmlFBO fbo_;
    };

} // namespace lfs::vis::gui
