/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_viewport_overlay.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
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

namespace lfs::vis::gui {

    void RmlViewportOverlay::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("viewport_overlay", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlViewportOverlay: failed to create RML context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/viewport_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlViewportOverlay: failed to load viewport_overlay.rml");
                return;
            }
            cacheBodyTemplate();
            document_->Show();
            applyGTMetricsOverlay();
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

        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("viewport_overlay");
        rml_context_ = nullptr;
        document_ = nullptr;
        body_template_rml_.clear();
    }

    void RmlViewportOverlay::reloadResources() {
        if (!rml_context_)
            return;

        if (doc_registered_)
            lfs::python::unregister_rml_document("viewport_overlay");
        doc_registered_ = false;

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        base_rcss_.clear();
        has_theme_signature_ = false;
        wants_input_ = false;
        render_needed_ = true;
        animation_active_ = true;
        mouse_pos_valid_ = false;
        last_render_w_ = 0;
        last_render_h_ = 0;
        last_document_hook_run_ = {};

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/viewport_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlViewportOverlay: failed to reload viewport_overlay.rml");
                return;
            }
            cacheBodyTemplate();
            document_->Show();
            applyGTMetricsOverlay();
            updateToolbarRoots();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlViewportOverlay: resource not found during reload: {}", e.what());
            return;
        }

        updateTheme();
    }

    void RmlViewportOverlay::cacheBodyTemplate() {
        body_template_rml_.clear();
        if (!document_)
            return;

        if (auto* const body = document_->GetElementById("overlay-body"))
            body_template_rml_ = body->GetInnerRML();
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

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/viewport_overlay.theme.rcss"));
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

    void RmlViewportOverlay::setToolbarPanels(const float primary_x,
                                              const float primary_width,
                                              const bool show_secondary,
                                              const float secondary_x,
                                              const float secondary_width) {
        const bool changed =
            std::abs(primary_toolbar_x_ - primary_x) > 0.5f ||
            std::abs(primary_toolbar_width_ - primary_width) > 0.5f ||
            show_secondary_toolbar_ != show_secondary ||
            std::abs(secondary_toolbar_x_ - secondary_x) > 0.5f ||
            std::abs(secondary_toolbar_width_ - secondary_width) > 0.5f;
        if (!changed) {
            return;
        }

        primary_toolbar_x_ = primary_x;
        primary_toolbar_width_ = primary_width;
        show_secondary_toolbar_ = show_secondary;
        secondary_toolbar_x_ = secondary_x;
        secondary_toolbar_width_ = secondary_width;
        render_needed_ = true;
        updateToolbarRoots();
    }

    void RmlViewportOverlay::setGTMetricsOverlay(GTMetricsOverlayState state) {
        const bool changed =
            gt_metrics_overlay_.visible != state.visible ||
            std::abs(gt_metrics_overlay_.x - state.x) > 0.5f ||
            std::abs(gt_metrics_overlay_.y - state.y) > 0.5f ||
            gt_metrics_overlay_.psnr_text != state.psnr_text ||
            gt_metrics_overlay_.show_ssim != state.show_ssim ||
            gt_metrics_overlay_.ssim_text != state.ssim_text;
        if (!changed) {
            return;
        }

        gt_metrics_overlay_ = std::move(state);
        applyGTMetricsOverlay();
        render_needed_ = true;
    }

    void RmlViewportOverlay::updateToolbarRoots() {
        if (!document_) {
            return;
        }

        const auto apply_root = [&](const char* element_id,
                                    const float x,
                                    const float width,
                                    const bool visible) {
            if (auto* const element = document_->GetElementById(element_id)) {
                element->SetProperty("left", std::format("{:.1f}px", x));
                element->SetProperty("width", std::format("{:.1f}px", std::max(width, 0.0f)));
                element->SetClass("hidden", !visible);
            }
        };

        apply_root("primary-toolbar-root", primary_toolbar_x_, primary_toolbar_width_, primary_toolbar_width_ > 0.0f);
        apply_root("secondary-toolbar-root",
                   secondary_toolbar_x_,
                   secondary_toolbar_width_,
                   show_secondary_toolbar_ && secondary_toolbar_width_ > 0.0f);
    }

    void RmlViewportOverlay::applyGTMetricsOverlay() {
        if (!document_) {
            return;
        }

        if (auto* const overlay = document_->GetElementById("gt-metrics-overlay")) {
            overlay->SetClass("hidden", !gt_metrics_overlay_.visible);
            overlay->SetProperty("left", std::format("{:.1f}px", gt_metrics_overlay_.x));
            overlay->SetProperty("top", std::format("{:.1f}px", gt_metrics_overlay_.y));
        }
        if (auto* const psnr = document_->GetElementById("gt-metrics-psnr")) {
            psnr->SetInnerRML(Rml::String(gt_metrics_overlay_.psnr_text));
        }
        if (auto* const ssim_row = document_->GetElementById("gt-metrics-ssim-row")) {
            ssim_row->SetClass("hidden", !gt_metrics_overlay_.show_ssim);
        }
        if (auto* const ssim = document_->GetElementById("gt-metrics-ssim")) {
            ssim->SetInnerRML(Rml::String(gt_metrics_overlay_.ssim_text));
        }
    }

    void RmlViewportOverlay::processInput(const PanelInputState& input) {
        wants_input_ = false;
        if (!rml_context_ || !document_)
            return;
        if (vp_size_.x <= 0 || vp_size_.y <= 0)
            return;
        updateToolbarRoots();
        if (rml_manager_) {
            rml_manager_->trackContextFrame(rml_context_,
                                            static_cast<int>(vp_pos_.x - screen_origin_.x),
                                            static_cast<int>(vp_pos_.y - screen_origin_.y));
        }

        if (guiFocusState().want_capture_mouse)
            return;
        const float mx = input.mouse_x - vp_pos_.x;
        const float my = input.mouse_y - vp_pos_.y;
        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);
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
            rml_context_->ProcessMouseMove(rml_mx, rml_my, mods);
        }

        auto* hover = rml_context_->GetHoverElement();
        bool over_interactive = hover && hover->GetTagName() != "body" &&
                                hover->GetId() != "overlay-body" &&
                                hover->GetId() != "dm-root";

        if (over_interactive) {
            wants_input_ = true;
            guiFocusState().want_capture_mouse = true;

            if (input.mouse_clicked[0]) {
                render_needed_ = true;
                rml_context_->ProcessMouseButtonDown(0, mods);
            }
            if (input.mouse_released[0]) {
                render_needed_ = true;
                rml_context_->ProcessMouseButtonUp(0, mods);
            }
            if (input.mouse_clicked[1]) {
                render_needed_ = true;
                rml_context_->ProcessMouseButtonDown(1, mods);
            }
            if (input.mouse_released[1]) {
                render_needed_ = true;
                rml_context_->ProcessMouseButtonUp(1, mods);
            }
            if (input.mouse_wheel != 0.0f) {
                render_needed_ = true;
                rml_context_->ProcessMouseWheel(Rml::Vector2f(0.0f, -input.mouse_wheel), mods);
            }

            RmlPanelHost::setFrameTooltip(resolveRmlTooltip(hover), hover);
        }
    }

    bool RmlViewportOverlay::blocksPointer(const double screen_x, const double screen_y) const {
        if (!rml_context_ || !document_ || vp_size_.x <= 0 || vp_size_.y <= 0) {
            return false;
        }

        const float local_x = static_cast<float>(screen_x) - vp_pos_.x;
        const float local_y = static_cast<float>(screen_y) - vp_pos_.y;
        if (local_x < 0.0f || local_x >= vp_size_.x || local_y < 0.0f || local_y >= vp_size_.y) {
            return false;
        }

        auto* const hover = rml_context_->GetElementAtPoint(Rml::Vector2f(local_x, local_y));
        return hover && hover->GetTagName() != "body" &&
               hover->GetId() != "overlay-body" &&
               hover->GetId() != "dm-root";
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
            // Restore the original unbound body markup instead of recycling
            // the live subtree, which may already contain expanded data-for views.
            if (!body_template_rml_.empty()) {
                body->SetInnerRML(Rml::String(body_template_rml_));
            } else {
                LOG_WARN("RmlViewportOverlay: missing body template for stale data-model rebind");
                body->RemoveChild(existing);
            }
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

        applyGTMetricsOverlay();
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

        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
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
        updateToolbarRoots();

        const bool needs_render = render_needed_ || animation_active_ || run_document_hooks ||
                                  theme_changed || size_changed;
        if (!needs_render) {
            rml_manager_->queueVulkanContext(rml_context_,
                                             vp_pos_.x - screen_origin_.x,
                                             vp_pos_.y - screen_origin_.y,
                                             true,
                                             true,
                                             vp_pos_.x - screen_origin_.x,
                                             vp_pos_.y - screen_origin_.y,
                                             vp_pos_.x - screen_origin_.x + vp_size_.x,
                                             vp_pos_.y - screen_origin_.y + vp_size_.y);
            return;
        }

        rml_manager_->trackContextFrame(rml_context_,
                                        static_cast<int>(vp_pos_.x - screen_origin_.x),
                                        static_cast<int>(vp_pos_.y - screen_origin_.y));
        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        rml_context_->Update();

        rml_manager_->queueVulkanContext(rml_context_,
                                         vp_pos_.x - screen_origin_.x,
                                         vp_pos_.y - screen_origin_.y,
                                         true,
                                         true,
                                         vp_pos_.x - screen_origin_.x,
                                         vp_pos_.y - screen_origin_.y,
                                         vp_pos_.x - screen_origin_.x + vp_size_.x,
                                         vp_pos_.y - screen_origin_.y + vp_size_.y);
        animation_active_ = (rml_context_->GetNextUpdateDelay() == 0);
        render_needed_ = false;
        last_render_w_ = w;
        last_render_h_ = h;
    }

} // namespace lfs::vis::gui
