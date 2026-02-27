/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_viewport_overlay.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "python/python_runtime.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Input.h>
#include <cassert>
#include <format>
#include <imgui.h>

namespace lfs::vis::gui {

    void RmlViewportOverlay::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("viewport_overlay", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlViewportOverlay: failed to create RML context");
            return;
        }

        rml_context_->EnableMouseCursor(false);

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/viewport_overlay.rml");
            document_ = rml_context_->LoadDocument(rml_path.string());
            if (!document_) {
                LOG_ERROR("RmlViewportOverlay: failed to load viewport_overlay.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlViewportOverlay: resource not found: {}", e.what());
            return;
        }

        updateTheme();
    }

    void RmlViewportOverlay::shutdown() {
        if (doc_registered_)
            lfs::python::unregister_rml_document("viewport_overlay");
        doc_registered_ = false;

        fbo_.destroy();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("viewport_overlay");
        rml_context_ = nullptr;
        document_ = nullptr;
    }

    std::string RmlViewportOverlay::generateThemeRCSS() const {
        using rml_theme::colorToRml;
        using rml_theme::colorToRmlAlpha;
        const auto& t = lfs::vis::theme();

        const auto toolbar_bg = colorToRml(t.toolbar_background());
        const auto subtoolbar_bg = colorToRml(t.subtoolbar_background());
        const auto icon_dim = colorToRmlAlpha(t.palette.text, 0.9f);
        const auto selected_bg = colorToRml(t.palette.primary);
        const auto selected_bg_hover = colorToRml(ImVec4(
            std::min(1.0f, t.palette.primary.x + 0.1f),
            std::min(1.0f, t.palette.primary.y + 0.1f),
            std::min(1.0f, t.palette.primary.z + 0.1f),
            t.palette.primary.w));
        const auto selected_icon = colorToRml(t.palette.background);
        const auto hover_bg = colorToRmlAlpha(t.palette.surface_bright, 0.3f);

        return std::format(
            ".toolbar-container {{ background-color: {}; border-radius: {:.0f}dp; }}\n"
            ".subtoolbar-container {{ background-color: {}; border-radius: {:.0f}dp; }}\n"
            ".icon-btn img {{ image-color: {}; }}\n"
            ".icon-btn:hover {{ background-color: {}; }}\n"
            ".icon-btn.selected {{ background-color: {}; }}\n"
            ".icon-btn.selected:hover {{ background-color: {}; }}\n"
            ".icon-btn.selected img {{ image-color: {}; }}\n",
            toolbar_bg, t.sizes.window_rounding,
            subtoolbar_bg, t.sizes.window_rounding,
            icon_dim,
            hover_bg,
            selected_bg, selected_bg_hover, selected_icon);
    }

    void RmlViewportOverlay::updateTheme() {
        if (!document_)
            return;

        const auto& t = lfs::vis::theme();
        if (t.name == last_theme_)
            return;
        last_theme_ = t.name;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/viewport_overlay.rcss");

        rml_theme::applyTheme(document_, base_rcss_, generateThemeRCSS());
    }

    void RmlViewportOverlay::setViewportBounds(glm::vec2 pos, glm::vec2 size) {
        vp_pos_ = pos;
        vp_size_ = size;
    }

    void RmlViewportOverlay::processInput() {
        wants_input_ = false;
        if (!rml_context_ || !document_)
            return;
        if (vp_size_.x <= 0 || vp_size_.y <= 0)
            return;

        ImGuiIO& io = ImGui::GetIO();
        const float dp_ratio = rml_manager_->getDpRatio();

        float mx = (io.MousePos.x - vp_pos_.x) * dp_ratio;
        float my = (io.MousePos.y - vp_pos_.y) * dp_ratio;

        rml_context_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), 0);

        auto* hover = rml_context_->GetHoverElement();
        bool over_interactive = hover && hover->GetTagName() != "body" &&
                                hover->GetId() != "overlay-body";

        if (over_interactive) {
            wants_input_ = true;
            io.WantCaptureMouse = true;

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                rml_context_->ProcessMouseButtonDown(0, 0);
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
                rml_context_->ProcessMouseButtonUp(0, 0);
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
                rml_context_->ProcessMouseButtonDown(1, 0);
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Right))
                rml_context_->ProcessMouseButtonUp(1, 0);
        }
    }

    void RmlViewportOverlay::render() {
        if (!rml_context_ || !document_)
            return;
        if (vp_size_.x <= 0 || vp_size_.y <= 0)
            return;

        if (!doc_registered_) {
            lfs::python::register_rml_document("viewport_overlay", document_);
            doc_registered_ = true;
        }

        updateTheme();

        const float dp_ratio = rml_manager_->getDpRatio();
        const int w = static_cast<int>(vp_size_.x * dp_ratio);
        const int h = static_cast<int>(vp_size_.y * dp_ratio);

        auto* body = document_->GetElementById("overlay-body");
        if (body) {
            body->SetAttribute("data-vp-w", std::to_string(static_cast<int>(vp_size_.x)));
            body->SetAttribute("data-vp-h", std::to_string(static_cast<int>(vp_size_.y)));
        }

        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        rml_context_->Update();

        fbo_.ensure(w, h);
        if (!fbo_.valid())
            return;

        auto* render = rml_manager_->getRenderInterface();
        assert(render);
        render->SetViewport(w, h);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        fbo_.unbind(prev_fbo);

        auto* vp = ImGui::GetMainViewport();
        const ImVec2 blit_pos(vp_pos_.x, vp_pos_.y);
        const ImVec2 blit_size(vp_size_.x, vp_size_.y);
        fbo_.blitToDrawList(ImGui::GetForegroundDrawList(vp), blit_pos, blit_size);
    }

} // namespace lfs::vis::gui
