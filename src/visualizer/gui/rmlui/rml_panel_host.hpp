/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_fbo.hpp"
#include <core/export.hpp>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

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
        void draw(const PanelDrawContext& ctx, float avail_w, float avail_h,
                  float pos_x, float pos_y);
        void drawDirect(float x, float y, float w, float h);
        bool ensureContext();

        void setInput(const PanelInputState* input) { input_ = input; }
        bool hasInput() const { return input_ != nullptr; }
        bool wantsKeyboard() const { return wants_keyboard_; }

        static std::string consumeFrameTooltip();
        static bool consumeFrameWantsKeyboard();

        void setHeightMode(HeightMode mode) { height_mode_ = mode; }
        HeightMode getHeightMode() const { return height_mode_; }
        float getContentHeight() const { return last_content_height_; }
        void markContentDirty() { content_dirty_ = true; }
        void setForeground(bool fg) { foreground_ = fg; }
        void setInputClipY(float y_min, float y_max) {
            clip_y_min_ = y_min;
            clip_y_max_ = y_max;
        }

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
        float last_synced_text_[4]{};
        bool has_text_focus_ = false;
        bool wants_keyboard_ = false;

        bool foreground_ = false;
        float clip_y_min_ = -1.0f;
        float clip_y_max_ = -1.0f;
        const PanelInputState* input_ = nullptr;
        RmlFBO fbo_;
    };

} // namespace lfs::vis::gui
