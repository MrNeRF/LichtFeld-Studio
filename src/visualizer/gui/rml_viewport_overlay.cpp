/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_viewport_overlay.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "python/python_runtime.hpp"
#include "python/ui_hooks.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Input.h>
#include <cassert>
#include <format>
#include <vector>
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

        render_needed_ = true;
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

    std::string RmlViewportOverlay::generateThemeRCSS(const lfs::vis::Theme& t) const {
        using rml_theme::colorToRml;
        using rml_theme::colorToRmlAlpha;

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
        const auto overlay_backdrop = colorToRmlAlpha(t.palette.background, 0.55f);
        const auto overlay_panel_bg = colorToRmlAlpha(t.palette.surface, 0.97f);
        const auto overlay_panel_border = colorToRmlAlpha(t.palette.border, 0.45f);
        const auto overlay_text = colorToRml(t.palette.text);
        const auto overlay_text_dim = colorToRml(t.palette.text_dim);

        return std::format(
            ".toolbar-container {{ background-color: {}; border-radius: {:.0f}dp; }}\n"
            ".subtoolbar-container {{ background-color: {}; border-radius: {:.0f}dp; }}\n"
            ".icon-btn img {{ image-color: {}; }}\n"
            ".icon-btn:hover {{ background-color: {}; }}\n"
            ".icon-btn.selected {{ background-color: {}; }}\n"
            ".icon-btn.selected:hover {{ background-color: {}; }}\n"
            ".icon-btn.selected img {{ image-color: {}; }}\n"
            ".viewport-status-backdrop {{ background-color: {}; }}\n"
            ".viewport-status-panel {{ background-color: {}; border-color: {}; border-radius: {:.0f}dp; }}\n"
            ".viewport-status-title {{ color: {}; }}\n"
            ".viewport-status-path {{ color: {}; }}\n"
            ".viewport-status-stage {{ color: {}; }}\n",
            toolbar_bg, t.sizes.window_rounding,
            subtoolbar_bg, t.sizes.window_rounding,
            icon_dim,
            hover_bg,
            selected_bg, selected_bg_hover, selected_icon,
            overlay_backdrop,
            overlay_panel_bg, overlay_panel_border, t.sizes.window_rounding,
            overlay_text, overlay_text_dim, overlay_text_dim);
    }

    bool RmlViewportOverlay::updateTheme() {
        if (!document_)
            return false;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return false;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/viewport_overlay.rcss");

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::generateAllThemeMedia([this](const auto& th) { return generateThemeRCSS(th); }));
        return true;
    }

    bool RmlViewportOverlay::shouldRunDocumentHooks(const bool force) const {
        if (!lfs::python::has_python_hooks("viewport_overlay", "document"))
            return false;
        if (force || last_document_hook_run_ == std::chrono::steady_clock::time_point{})
            return true;
        return (std::chrono::steady_clock::now() - last_document_hook_run_) >=
               kDocumentHookPollInterval;
    }

    void RmlViewportOverlay::setViewportBounds(glm::vec2 pos, glm::vec2 size,
                                               glm::vec2 screen_origin) {
        if (vp_pos_ != pos || screen_origin_ != screen_origin)
            mouse_pos_valid_ = false;
        if (vp_size_ != size)
            render_needed_ = true;
        vp_pos_ = pos;
        vp_size_ = size;
        screen_origin_ = screen_origin;
    }

    void RmlViewportOverlay::processInput() {
        wants_input_ = false;
        if (!rml_context_ || !document_)
            return;
        if (vp_size_.x <= 0 || vp_size_.y <= 0)
            return;

        ImGuiIO& io = ImGui::GetIO();
        if (guiFocusState().want_capture_mouse)
            return;
        float mx = io.MousePos.x - vp_pos_.x;
        float my = io.MousePos.y - vp_pos_.y;
        const int rml_mx = static_cast<int>(mx);
        const int rml_my = static_cast<int>(my);
        const bool was_inside = mouse_pos_valid_ &&
                                last_mouse_x_ >= 0 && last_mouse_x_ < static_cast<int>(vp_size_.x) &&
                                last_mouse_y_ >= 0 && last_mouse_y_ < static_cast<int>(vp_size_.y);
        const bool is_inside = rml_mx >= 0 && rml_mx < static_cast<int>(vp_size_.x) &&
                               rml_my >= 0 && rml_my < static_cast<int>(vp_size_.y);
        if ((!mouse_pos_valid_ || rml_mx != last_mouse_x_ || rml_my != last_mouse_y_) &&
            (was_inside || is_inside)) {
            mouse_pos_valid_ = true;
            last_mouse_x_ = rml_mx;
            last_mouse_y_ = rml_my;
            render_needed_ = true;
            rml_context_->ProcessMouseMove(rml_mx, rml_my, 0);
        }

        auto* hover = rml_context_->GetHoverElement();
        bool over_interactive = hover && hover->GetTagName() != "body" &&
                                hover->GetId() != "overlay-body" &&
                                hover->GetId() != "dm-root";

        if (over_interactive) {
            wants_input_ = true;
            guiFocusState().want_capture_mouse = true;
            io.WantCaptureMouse = true;

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                render_needed_ = true;
                rml_context_->ProcessMouseButtonDown(0, 0);
            }
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                render_needed_ = true;
                rml_context_->ProcessMouseButtonUp(0, 0);
            }
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                render_needed_ = true;
                rml_context_->ProcessMouseButtonDown(1, 0);
            }
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                render_needed_ = true;
                rml_context_->ProcessMouseButtonUp(1, 0);
            }

            RmlPanelHost::setFrameTooltip(resolveRmlTooltip(hover));
        }
    }

    void RmlViewportOverlay::ensureBodyDataModelBound(Rml::Element* body) {
        if (!body)
            return;

        const auto data_model = body->GetAttribute<Rml::String>("data-model", "");
        if (data_model.empty() || body->GetDataModel())
            return;

        auto* existing = document_->GetElementById("dm-root");
        if (existing) {
            if (existing->GetDataModel())
                return;

            // Wrapper exists but binding is stale (data model was rebuilt).
            // Tear it down so we can recreate with a fresh binding.
            std::vector<Rml::Element*> children;
            children.reserve(existing->GetNumChildren());
            for (int i = 0; i < existing->GetNumChildren(); ++i)
                children.push_back(existing->GetChild(i));
            for (auto* child : children)
                body->AppendChild(existing->RemoveChild(child));
            body->RemoveChild(existing);
        }

        // RmlUI does not rebind data-model when the attribute is set after
        // document load. Reattaching the subtree through a wrapper element
        // forces the binding pass.
        auto wrapper_ptr = document_->CreateElement("div");
        wrapper_ptr->SetId("dm-root");
        wrapper_ptr->SetAttribute("data-model", data_model);
        wrapper_ptr->SetProperty("position", "relative");
        wrapper_ptr->SetProperty("width", "100%");
        wrapper_ptr->SetProperty("height", "100%");
        auto* wrapper = body->AppendChild(std::move(wrapper_ptr));

        std::vector<Rml::Element*> children_to_move;
        children_to_move.reserve(body->GetNumChildren());
        for (int i = 0; i < body->GetNumChildren(); ++i) {
            auto* child = body->GetChild(i);
            if (child != wrapper)
                children_to_move.push_back(child);
        }
        for (auto* child : children_to_move)
            wrapper->AppendChild(body->RemoveChild(child));
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

        if (rml_manager_->shouldDeferFboUpdate(fbo_))
            return;

        const bool theme_changed = updateTheme();
        const int w = static_cast<int>(vp_size_.x);
        const int h = static_cast<int>(vp_size_.y);
        const bool size_changed = (w != last_render_w_ || h != last_render_h_);
        const bool run_document_hooks = shouldRunDocumentHooks(
            theme_changed || size_changed || render_needed_ || animation_active_);
        if (run_document_hooks) {
            lfs::python::invoke_python_document_hooks("viewport_overlay", "document", document_, true);
            lfs::python::invoke_python_document_hooks("viewport_overlay", "document", document_, false);
            last_document_hook_run_ = std::chrono::steady_clock::now();
        }

        auto* body = document_->GetElementById("overlay-body");
        ensureBodyDataModelBound(body);

        const bool needs_render = render_needed_ || animation_active_ || run_document_hooks ||
                                  theme_changed || size_changed;
        if (!needs_render)
            return;

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
        render->SetTargetFramebuffer(fbo_.fbo());

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        render->SetTargetFramebuffer(0);
        fbo_.unbind(prev_fbo);

        animation_active_ = (rml_context_->GetNextUpdateDelay() == 0);
        render_needed_ = false;
        last_render_w_ = w;
        last_render_h_ = h;
    }

    void RmlViewportOverlay::compositeToScreen(const int screen_w, const int screen_h) const {
        if (!fbo_.valid() || screen_w <= 0 || screen_h <= 0)
            return;
        fbo_.blitToScreen(vp_pos_.x - screen_origin_.x,
                          vp_pos_.y - screen_origin_.y,
                          vp_size_.x, vp_size_.y,
                          screen_w, screen_h);
    }

} // namespace lfs::vis::gui
