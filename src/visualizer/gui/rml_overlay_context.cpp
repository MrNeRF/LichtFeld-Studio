/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_overlay_context.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <cassert>
#include <format>

namespace lfs::vis::gui {

    RmlOverlayContext::RmlOverlayContext(RmlUIManager* mgr, const std::string& name, const std::string& rml_path)
        : mgr_(mgr),
          context_name_(name),
          rml_path_(rml_path) {
        assert(mgr_);
    }

    RmlOverlayContext::~RmlOverlayContext() {
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
            doc_ = rml_documents::loadDocument(ctx_, full_path);
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

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty()) {
            const std::string rcss_path = rml_path_.substr(0, rml_path_.rfind('.')) + ".rcss";
            base_rcss_ = rml_theme::loadBaseRCSS(rcss_path);
        }

        rml_theme::applyTheme(doc_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/overlay_context.theme.rcss"));
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
        (void)screen_w;
        (void)screen_h;
        if (!ctx_ || !doc_)
            return;

        const int px_w = static_cast<int>(w);
        const int px_h = static_cast<int>(h);

        if (px_w <= 0 || px_h <= 0)
            return;

        if (!mgr_ || !mgr_->getVulkanRenderInterface())
            return;

        if (px_w != width_ || px_h != height_)
            resize(px_w, px_h);
        ctx_->Update();
        mgr_->queueVulkanContext(ctx_, x, y, true, true, x, y, x + w, y + h);
    }

    void RmlOverlayContext::forwardMouseInput(const PanelInputState& input,
                                              const float overlay_x, const float overlay_y) {
        if (!ctx_)
            return;

        const float local_x = input.mouse_x - overlay_x;
        const float local_y = input.mouse_y - overlay_y;

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);

        ctx_->ProcessMouseMove(static_cast<int>(local_x), static_cast<int>(local_y), mods);

        if (input.mouse_clicked[0])
            ctx_->ProcessMouseButtonDown(0, mods);
        if (!input.mouse_down[0])
            ctx_->ProcessMouseButtonUp(0, mods);
        if (input.mouse_clicked[1])
            ctx_->ProcessMouseButtonDown(1, mods);
        if (!input.mouse_down[1])
            ctx_->ProcessMouseButtonUp(1, mods);
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

    void RmlOverlayContext::releaseRendererResources() {
    }

} // namespace lfs::vis::gui
