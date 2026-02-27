/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_overlay_context.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <cassert>
#include <cstring>
#include <format>

namespace lfs::vis::gui {

    RmlOverlayContext::RmlOverlayContext(RmlUIManager* mgr, const std::string& name, const std::string& rml_path)
        : mgr_(mgr),
          context_name_(name),
          rml_path_(rml_path) {
        assert(mgr_);
    }

    RmlOverlayContext::~RmlOverlayContext() {
        fbo_.destroy();
        if (ctx_ && mgr_)
            mgr_->destroyContext(context_name_);
    }

    void RmlOverlayContext::initContext() {
        if (ctx_)
            return;

        ctx_ = mgr_->createContext(context_name_, width_ > 0 ? width_ : 800, height_ > 0 ? height_ : 600);
        if (!ctx_) {
            LOG_ERROR("RmlOverlayContext: failed to create context '{}'", context_name_);
            return;
        }

        try {
            const auto full_path = lfs::vis::getAssetPath(rml_path_);
            doc_ = ctx_->LoadDocument(full_path.string());
            if (doc_) {
                doc_->Show();
            } else {
                LOG_ERROR("RmlOverlayContext: failed to load {}", rml_path_);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("RmlOverlayContext: resource not found: {}", e.what());
        }
    }

    void RmlOverlayContext::resize(const int w, const int h) {
        width_ = w;
        height_ = h;
        if (ctx_)
            ctx_->SetDimensions(Rml::Vector2i(w, h));
    }

    void RmlOverlayContext::syncTheme() {
        if (!doc_)
            return;

        const auto& p = lfs::vis::theme().palette;
        if (std::memcmp(last_synced_text_, &p.text, sizeof(last_synced_text_)) == 0)
            return;
        std::memcpy(last_synced_text_, &p.text, sizeof(last_synced_text_));

        if (base_rcss_.empty()) {
            const std::string rcss_path = rml_path_.substr(0, rml_path_.rfind('.')) + ".rcss";
            base_rcss_ = rml_theme::loadBaseRCSS(rcss_path);
        }

        rml_theme::applyTheme(doc_, base_rcss_, generateThemeRCSS());
    }

    std::string RmlOverlayContext::generateThemeRCSS() const {
        const auto& p = lfs::vis::theme().palette;
        const auto& t = lfs::vis::theme();

        using rml_theme::colorToRml;
        using rml_theme::colorToRmlAlpha;

        const auto surface = colorToRmlAlpha(p.surface, 0.95f);
        const auto border = colorToRmlAlpha(p.border, 0.4f);
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto primary = colorToRml(p.primary);
        const int rounding = static_cast<int>(t.sizes.window_rounding);

        return std::format(
            ".overlay-panel {{ background-color: {}; border-width: 1dp; border-color: {}; "
            "border-radius: {}dp; }}\n"
            ".overlay-text {{ color: {}; }}\n"
            ".overlay-text-dim {{ color: {}; }}\n"
            ".overlay-primary {{ color: {}; }}\n",
            surface, border, rounding,
            text, text_dim, primary);
    }

    void RmlOverlayContext::update() {
        if (!ctx_) {
            initContext();
            if (!ctx_)
                return;
        }
        syncTheme();
        ctx_->Update();
    }

    void RmlOverlayContext::render(const float x, const float y, const float w, const float h,
                                   const int screen_w, const int screen_h) {
        if (!ctx_ || !doc_)
            return;

        const float dp_ratio = mgr_->getDpRatio();
        const int px_w = static_cast<int>(w * dp_ratio);
        const int px_h = static_cast<int>(h * dp_ratio);

        if (px_w <= 0 || px_h <= 0)
            return;

        if (px_w != width_ || px_h != height_)
            resize(px_w, px_h);

        fbo_.ensure(px_w, px_h);
        if (!fbo_.valid())
            return;

        auto* render_iface = mgr_->getRenderInterface();
        assert(render_iface);
        render_iface->SetViewport(px_w, px_h);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);

        render_iface->BeginFrame();
        ctx_->Render();
        render_iface->EndFrame();

        fbo_.unbind(prev_fbo);

        fbo_.blitToScreen(x, y, w, h, screen_w, screen_h);
    }

    void RmlOverlayContext::forwardMouseInput(const PanelInputState& input,
                                              const float overlay_x, const float overlay_y) {
        if (!ctx_)
            return;

        const float dp_ratio = mgr_->getDpRatio();
        const float local_x = (input.mouse_x - overlay_x) * dp_ratio;
        const float local_y = (input.mouse_y - overlay_y) * dp_ratio;

        ctx_->ProcessMouseMove(static_cast<int>(local_x), static_cast<int>(local_y), 0);

        if (input.mouse_clicked[0])
            ctx_->ProcessMouseButtonDown(0, 0);
        if (!input.mouse_down[0])
            ctx_->ProcessMouseButtonUp(0, 0);
        if (input.mouse_clicked[1])
            ctx_->ProcessMouseButtonDown(1, 0);
        if (!input.mouse_down[1])
            ctx_->ProcessMouseButtonUp(1, 0);
    }

    Rml::Element* RmlOverlayContext::getElementById(const std::string& id) {
        if (!doc_)
            return nullptr;
        return doc_->GetElementById(id);
    }

    void RmlOverlayContext::showElement(const std::string& id) {
        auto* el = getElementById(id);
        if (el)
            el->SetProperty("display", "block");
    }

    void RmlOverlayContext::hideElement(const std::string& id) {
        auto* el = getElementById(id);
        if (el)
            el->SetProperty("display", "none");
    }

    void RmlOverlayContext::showContextMenu(const std::string& element_id, const float x, const float y,
                                            const std::string& inner_rml) {
        auto* el = getElementById(element_id);
        if (!el)
            return;
        el->SetInnerRML(inner_rml);
        el->SetProperty("left", std::format("{:.0f}dp", x));
        el->SetProperty("top", std::format("{:.0f}dp", y));
        el->SetClass("visible", true);
    }

    void RmlOverlayContext::hideContextMenu(const std::string& element_id) {
        auto* el = getElementById(element_id);
        if (el)
            el->SetClass("visible", false);
    }

    void RmlOverlayContext::destroyGLResources() {
        fbo_.destroy();
    }

} // namespace lfs::vis::gui
