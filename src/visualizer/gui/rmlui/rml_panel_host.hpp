/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_height_mode.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include <core/export.hpp>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    class RmlUIManager;

    class LFS_VIS_API RmlPanelHost {
    public:
        RmlPanelHost(RmlUIManager* manager, std::string context_name, std::string rml_path,
                     std::string inline_rcss = {});
        ~RmlPanelHost();

        RmlPanelHost(const RmlPanelHost&) = delete;
        RmlPanelHost& operator=(const RmlPanelHost&) = delete;

        void draw(const PanelDrawContext& ctx);
        void draw(const PanelDrawContext& ctx, float avail_w, float avail_h,
                  float pos_x, float pos_y);
        void drawDirect(float x, float y, float w, float h);
        bool drawDirectCached(float x, float y, float w, float h);
        void prepareDirect(float w, float h);
        void syncDirectLayout(float w, float h);
        bool ensureContext();
        bool ensureDocumentLoaded();
        bool reloadDocument();
        void releaseRendererResources();

        void setInput(const PanelInputState* input) { input_ = input; }
        bool hasInput() const { return input_ != nullptr; }
        bool wantsKeyboard() const { return wants_keyboard_; }

        static bool consumeFrameWantsKeyboard();
        static bool consumeFrameWantsTextInput();

        void setHeightMode(PanelHeightMode mode) { height_mode_ = mode; }
        float getContentHeight() const { return last_content_height_; }
        void setForcedHeight(float h) { forced_height_ = h; }
        void markContentDirty() {
            content_dirty_ = true;
            direct_cache_dirty_ = true;
        }
        void setForeground(bool fg) { foreground_ = fg; }
        void setFloating(bool floating);
        void setInputClipY(float y_min, float y_max) {
            clip_y_min_ = y_min;
            clip_y_max_ = y_max;
        }
        bool needsAnimationFrame() const {
            return render_needed_ || content_dirty_ || animation_active_ || tooltip_.revealDue();
        }

        Rml::ElementDocument* getDocument() { return document_; }
        Rml::Context* getContext() { return rml_context_; }
        bool isDocumentLoaded() const { return document_ != nullptr; }

    private:
        std::optional<RmlRect> openDropdownBounds() const;
        bool openDropdownContainsPoint(float local_x, float local_y) const;
        Rml::Element* openDropdownOptionAtPoint(float local_x, float local_y) const;
        void setManualDropdownHover(Rml::Element* option);
        void trackFrame(float panel_x, float panel_y);
        void applyHoverTooltip(int pw, float panel_y, float display_h);
        bool hitTestPanelShape(float local_x, float local_y, float logical_w, float logical_h) const;
        bool forwardInput(float panel_x, float panel_y);
        bool syncThemeProperties();
        bool loadDocument();
        void cacheContentElements();
        float computeScrollHeightCap() const;
        float computeContentHeight() const;
        float clampScrollTop(float scroll_top) const;
        void restoreScrollTop(float scroll_top);
        void resolveDirectRenderHeight(float requested_h, int& ph, float& display_h) const;
        bool updateContextLayout(int pw, int ph);
        int renderPadding() const;
        void applyPanelSpaceClass();
        void renderIfDirty(int pw, int ph, float& display_h);
        void compositeDirectToScreen(float x, float y, float w, float h);

        RmlUIManager* manager_;
        std::string context_name_;
        std::string rml_path_;
        std::string inline_rcss_;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* frame_el_ = nullptr;
        Rml::Element* content_wrap_el_ = nullptr;
        Rml::Element* content_el_ = nullptr;
        Rml::Element* scroll_el_ = nullptr;

        PanelHeightMode height_mode_ = PanelHeightMode::Fill;
        float forced_height_ = 0.0f;
        float last_content_height_ = 0.0f;
        float last_content_el_height_ = 0.0f;
        int last_measure_w_ = 0;
        bool content_dirty_ = true;

        std::string base_rcss_;
        bool base_rcss_loaded_ = false;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        bool has_text_focus_ = false;
        bool wants_keyboard_ = false;

        bool foreground_ = false;
        bool floating_ = false;
        float clip_y_min_ = -1.0f;
        float clip_y_max_ = -1.0f;
        const PanelInputState* input_ = nullptr;

        bool render_needed_ = true;
        bool animation_active_ = false;
        bool direct_cache_dirty_ = true;
        CachedVulkanContextRender direct_cache_;
        int last_fbo_w_ = 0;
        int last_fbo_h_ = 0;
        int last_fbo_padding_ = 0;
        int last_layout_w_ = 0;
        int last_layout_h_ = 0;
        int last_layout_padding_ = -1;
        int last_forwarded_mx_ = -1;
        int last_forwarded_my_ = -1;
        bool last_hovered_ = false;
        Rml::Element* manual_dropdown_hover_ = nullptr;
        bool manual_dropdown_mouse_captured_ = false;
        // Per-button capture so scrollbar drags continue when the cursor
        // leaves the panel and the matching Up always reaches RmlUI.
        bool mouse_captured_[3] = {false, false, false};
        RmlTooltipController tooltip_;
    };

} // namespace lfs::vis::gui
