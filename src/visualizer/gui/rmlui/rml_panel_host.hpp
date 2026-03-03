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
    class Element;
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
        bool forwardInput(float panel_x, float panel_y);
        bool syncThemeProperties();
        std::string generateThemeRCSS() const;
        bool loadDocument();
        void cacheContentElements();
        void renderIfDirty(int pw, int ph, float& display_h);

        RmlUIManager* manager_;
        std::string context_name_;
        std::string rml_path_;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* content_wrap_el_ = nullptr;

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

        bool render_needed_ = true;
        bool animation_active_ = false;
        int last_fbo_w_ = 0;
        int last_fbo_h_ = 0;
        int last_forwarded_mx_ = -1;
        int last_forwarded_my_ = -1;
        bool last_hovered_ = false;
    };

} // namespace lfs::vis::gui
