/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_viewport_overlay.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
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
#include <algorithm>
#include <cassert>
#include <format>
#include <vector>

namespace lfs::vis::gui {
    namespace {
        [[nodiscard]] bool isInteractiveViewportOverlayElement(const Rml::Element* const element) {
            return element && element->GetTagName() != "body" &&
                   element->GetTagName() != "#root" &&
                   element->GetId() != "overlay-body" &&
                   element->GetId() != "dm-root" &&
                   !element->IsClassSet("viewport-split-divider");
        }

        [[nodiscard]] bool isViewportOverlayHoverRoot(const Rml::Element* const element) {
            if (!isInteractiveViewportOverlayElement(element))
                return false;

            const auto tag = element->GetTagName();
            return tag == "button" || tag == "input" || tag == "textarea" || tag == "select" ||
                   element->IsClassSet("icon-btn") ||
                   element->IsClassSet("toolbar-group-container") ||
                   element->IsClassSet("viewport-transform-option") ||
                   element->IsClassSet("viewport-transform-action") ||
                   element->IsClassSet("vram-hud-tree-row") ||
                   element->IsClassSet("vram-hud-expand-toggle") ||
                   element->IsClassSet("vram-hud-tab") ||
                   element->IsClassSet("vram-hud-filter-clear") ||
                   element->IsClassSet("vram-hud-resize");
        }

        [[nodiscard]] const Rml::Element* viewportOverlayHoverRoot(
            const Rml::Element* const element) {
            for (auto* node = element; isInteractiveViewportOverlayElement(node);
                 node = node->GetParentNode()) {
                if (isViewportOverlayHoverRoot(node))
                    return node;
            }
            return isInteractiveViewportOverlayElement(element) ? element : nullptr;
        }

    } // namespace

    RmlViewportOverlay::RmlViewportOverlay()
        : vram_hud_(std::make_unique<VramHudOverlay>()) {}

    RmlViewportOverlay::~RmlViewportOverlay() = default;

    void RmlViewportOverlay::markRenderNeeded(const RenderReason reason) {
        render_needed_ = true;
        render_reason_bits_ |= static_cast<std::uint32_t>(reason);
    }

    std::string RmlViewportOverlay::renderReasonSources() const {
        if (render_reason_bits_ == 0)
            return "none";

        std::string sources;
        const auto append = [&](const RenderReason reason, const char* name) {
            if ((render_reason_bits_ & static_cast<std::uint32_t>(reason)) == 0)
                return;
            if (!sources.empty())
                sources.push_back('|');
            sources.append(name);
        };
        append(RenderReason::Initial, "initial");
        append(RenderReason::Reload, "reload");
        append(RenderReason::DocumentSync, "document_sync");
        append(RenderReason::DocumentHook, "document_hook");
        append(RenderReason::ViewportResize, "viewport_resize");
        append(RenderReason::ToolbarLayout, "toolbar_layout");
        append(RenderReason::SplitDivider, "split_divider");
        append(RenderReason::GTMetrics, "gt_metrics");
        append(RenderReason::VramHud, "vram_hud");
        append(RenderReason::DataModelBinding, "data_model_binding");
        append(RenderReason::PointerHover, "pointer_hover");
        append(RenderReason::PointerButton, "pointer_button");
        append(RenderReason::PointerWheel, "pointer_wheel");
        append(RenderReason::PointerDrag, "pointer_drag");
        append(RenderReason::Keyboard, "keyboard");
        append(RenderReason::LodStats, "lod_stats");
        return sources.empty() ? "unknown" : sources;
    }

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
            bindReactiveStore();
            refreshGTMetricsOverlayFromStore();
            applyLodStatsOverlay();
            if (vram_hud_)
                vram_hud_->onDocumentLoaded(document_);
        } catch (const std::exception& e) {
            LOG_ERROR("RmlViewportOverlay: resource not found: {}", e.what());
            return;
        }

        markRenderNeeded(RenderReason::Initial);
        updateTheme();
    }

    void RmlViewportOverlay::shutdown() {
        if (doc_registered_)
            lfs::python::unregister_rml_document("viewport_overlay");
        doc_registered_ = false;
        gt_metrics_config_subscription_.reset();
        camera_metrics_subscription_.reset();
        vram_hud_subscription_.reset();
        document_sync_subscriptions_.clear();

        if (vram_hud_)
            vram_hud_->onDocumentDestroyed();
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("viewport_overlay");
        rml_context_ = nullptr;
        document_ = nullptr;
        body_el_ = nullptr;
        body_template_rml_.clear();
        hovered_interactive_ = false;
        last_hover_element_ = nullptr;
        mouse_pos_valid_ = false;
    }

    void RmlViewportOverlay::reloadResources() {
        if (!rml_context_)
            return;

        if (doc_registered_)
            lfs::python::unregister_rml_document("viewport_overlay");
        doc_registered_ = false;

        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        if (vram_hud_)
            vram_hud_->onDocumentDestroyed();
        document_ = nullptr;
        body_el_ = nullptr;
        base_rcss_.clear();
        has_theme_signature_ = false;
        wants_input_ = false;
        markRenderNeeded(RenderReason::Reload);
        document_sync_dirty_ = true;
        data_model_binding_dirty_ = true;
        toolbar_roots_dirty_ = true;
        animation_active_ = true;
        hovered_interactive_ = false;
        last_hover_element_ = nullptr;
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
            applySplitDividerOverlay();
            applyLodStatsOverlay();
            if (vram_hud_)
                vram_hud_->onDocumentLoaded(document_);
            updateToolbarRoots();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlViewportOverlay: resource not found during reload: {}", e.what());
            return;
        }

        updateTheme();
    }

    void RmlViewportOverlay::cacheBodyTemplate() {
        body_template_rml_.clear();
        body_el_ = nullptr;
        if (!document_)
            return;

        if (auto* const body = document_->GetElementById("overlay-body")) {
            body_el_ = body;
            body_template_rml_ = body->GetInnerRML();
        }
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

    // Python document hooks are allowed to mutate the Rml document, so a due
    // hook must break the cached-render path even if all native overlay state is
    // unchanged. Plugins that register passive hooks therefore opt into this
    // polling cadence; hooks that dirty every poll intentionally repaint.
    bool RmlViewportOverlay::shouldRunDocumentHooks(const bool force, const bool prepend) const {
        if (!lfs::python::has_python_hooks("viewport_overlay", "document", prepend))
            return false;
        if (force || last_document_hook_run_ == std::chrono::steady_clock::time_point{})
            return true;
        return (std::chrono::steady_clock::now() - last_document_hook_run_) >=
               kDocumentHookPollInterval;
    }

    bool RmlViewportOverlay::shouldRunAnyDocumentHooks(const bool force) const {
        return shouldRunDocumentHooks(force, true) || shouldRunDocumentHooks(force, false);
    }

    void RmlViewportOverlay::markDocumentSyncDirty() {
        document_sync_dirty_ = true;
    }

    bool RmlViewportOverlay::syncBuiltinDocument(const bool force) {
        if (!document_ || (!force && !document_sync_dirty_))
            return false;

        LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render.builtin_document_sync", 0.25);
        const bool dirty = lfs::python::sync_viewport_overlay_document(document_);
        document_sync_dirty_ = false;
        if (dirty)
            markRenderNeeded(RenderReason::DocumentSync);
        return dirty;
    }

    void RmlViewportOverlay::setViewportBounds(glm::vec2 pos, glm::vec2 size,
                                               glm::vec2 screen_origin) {
        const bool context_size_changed =
            static_cast<int>(vp_size_.x) != static_cast<int>(size.x) ||
            static_cast<int>(vp_size_.y) != static_cast<int>(size.y);
        if (vp_pos_ != pos || screen_origin_ != screen_origin)
            mouse_pos_valid_ = false;
        if (vp_pos_ != pos || screen_origin_ != screen_origin || vp_size_ != size)
            last_hover_element_ = nullptr;
        if (context_size_changed)
            markRenderNeeded(RenderReason::ViewportResize);
        vp_pos_ = pos;
        vp_size_ = size;
        screen_origin_ = screen_origin;
    }

    void RmlViewportOverlay::setViewportContentOffset(const float x) {
        if (std::abs(viewport_content_offset_ - x) > 0.5f) {
            viewport_content_offset_ = x;
            viewport_content_offset_dirty_ = true;
            markRenderNeeded(RenderReason::ViewportResize);
        }
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
        markRenderNeeded(RenderReason::ToolbarLayout);
        toolbar_roots_dirty_ = true;
        updateToolbarRoots();
    }

    void RmlViewportOverlay::setSplitDividerOverlay(SplitDividerOverlayState state) {
        const bool changed =
            split_divider_overlay_.visible != state.visible ||
            std::abs(split_divider_overlay_.x - state.x) > 0.5f ||
            std::abs(split_divider_overlay_.y - state.y) > 0.5f ||
            std::abs(split_divider_overlay_.width - state.width) > 0.5f ||
            std::abs(split_divider_overlay_.height - state.height) > 0.5f;
        if (!changed) {
            return;
        }

        split_divider_overlay_ = state;
        applySplitDividerOverlay();
        markRenderNeeded(RenderReason::SplitDivider);
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
        markRenderNeeded(RenderReason::GTMetrics);
    }

    void RmlViewportOverlay::setLodStatsOverlay(LodStatsOverlayState state) {
        const bool changed =
            lod_stats_overlay_.visible != state.visible ||
            std::abs(lod_stats_overlay_.x - state.x) > 0.5f ||
            std::abs(lod_stats_overlay_.y - state.y) > 0.5f ||
            lod_stats_overlay_.status_text != state.status_text ||
            lod_stats_overlay_.selected_text != state.selected_text ||
            lod_stats_overlay_.budget_text != state.budget_text ||
            lod_stats_overlay_.model_text != state.model_text ||
            lod_stats_overlay_.tree_text != state.tree_text ||
            lod_stats_overlay_.traversal_text != state.traversal_text ||
            lod_stats_overlay_.stop_text != state.stop_text ||
            lod_stats_overlay_.chunks_text != state.chunks_text ||
            lod_stats_overlay_.cache_text != state.cache_text ||
            lod_stats_overlay_.selector_text != state.selector_text ||
            lod_stats_overlay_.pixel_text != state.pixel_text ||
            lod_stats_overlay_.render_text != state.render_text ||
            lod_stats_overlay_.foveation_text != state.foveation_text ||
            lod_stats_overlay_.hash_text != state.hash_text;
        if (!changed) {
            return;
        }

        lod_stats_overlay_ = std::move(state);
        applyLodStatsOverlay();
        markRenderNeeded(RenderReason::LodStats);
    }

    void RmlViewportOverlay::setVramHudOverlay(VramHudOverlayState state) {
        if (!vram_hud_)
            return;
        const bool was_visible = vram_hud_->isVisible();
        vram_hud_->setState(std::move(state));
        if (was_visible || vram_hud_->isVisible())
            markRenderNeeded(RenderReason::VramHud);
    }

    bool RmlViewportOverlay::isDueForVramProcessSample(std::chrono::milliseconds interval) {
        return vram_hud_ ? vram_hud_->isDueForProcessSample(interval) : false;
    }

    void RmlViewportOverlay::bindReactiveStore() {
        auto& store = lfs::vis::app_store();
        gt_metrics_config_ = store.gt_metrics_overlay_config.get();
        camera_metrics_ = store.camera_metrics.get();

        const auto vram_hud_state = store.vram_hud.get();
        RmlViewportOverlay::VramHudOverlayState overlay_state;
        if (vram_hud_state.visible && vram_hud_state.snapshot) {
            overlay_state.visible = true;
            overlay_state.snapshot = *vram_hud_state.snapshot;
        }
        setVramHudOverlay(std::move(overlay_state));

        gt_metrics_config_subscription_ = store.gt_metrics_overlay_config.subscribe(
            [this](const lfs::vis::AppStore::GTMetricsOverlayConfig& config) {
                gt_metrics_config_ = config;
                refreshGTMetricsOverlayFromStore();
            });
        camera_metrics_subscription_ = store.camera_metrics.subscribe(
            [this](const std::optional<lfs::vis::AppStore::CameraMetrics>& metrics) {
                camera_metrics_ = metrics;
                refreshGTMetricsOverlayFromStore();
            });
        vram_hud_subscription_ = store.vram_hud.subscribe(
            [this](const lfs::vis::AppStore::VramHud& state) {
                RmlViewportOverlay::VramHudOverlayState overlay;
                if (state.visible && state.snapshot) {
                    overlay.visible = true;
                    overlay.snapshot = *state.snapshot;
                }
                setVramHudOverlay(std::move(overlay));
            });

        auto mark_document_dirty = [this](const auto&) {
            markDocumentSyncDirty();
        };
        document_sync_subscriptions_.push_back(store.training_state.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.scene_generation.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.selection_generation.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.active_tool.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.active_submode.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.transform_space.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.pivot_mode.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.render_settings_generation.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.viewport_toolbar_generation.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.import_overlay_state.subscribe(mark_document_dirty));
        document_sync_subscriptions_.push_back(store.video_export_overlay_state.subscribe(mark_document_dirty));
    }

    void RmlViewportOverlay::refreshGTMetricsOverlayFromStore() {
        GTMetricsOverlayState state;
        state.visible = gt_metrics_config_.visible;
        state.x = gt_metrics_config_.x;
        state.y = gt_metrics_config_.y;
        state.show_ssim = gt_metrics_config_.show_ssim;
        state.psnr_text = "--";
        state.ssim_text = "--";

        if (state.visible && camera_metrics_ &&
            camera_metrics_->camera_id == gt_metrics_config_.current_camera_id) {
            state.psnr_text = std::format("{:.2f}", camera_metrics_->psnr);
            if (state.show_ssim && camera_metrics_->ssim.has_value())
                state.ssim_text = std::format("{:.4f}", *camera_metrics_->ssim);
        }

        setGTMetricsOverlay(std::move(state));
    }

    bool RmlViewportOverlay::updateToolbarRoots() {
        if (!document_) {
            return false;
        }

        const bool changed =
            toolbar_roots_dirty_ ||
            std::abs(applied_primary_toolbar_x_ - primary_toolbar_x_) > 0.5f ||
            std::abs(applied_primary_toolbar_width_ - primary_toolbar_width_) > 0.5f ||
            applied_show_secondary_toolbar_ != show_secondary_toolbar_ ||
            std::abs(applied_secondary_toolbar_x_ - secondary_toolbar_x_) > 0.5f ||
            std::abs(applied_secondary_toolbar_width_ - secondary_toolbar_width_) > 0.5f;
        if (!changed)
            return false;

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
        const auto apply_left_toolbar_offset = [&](const char* element_id, const float x) {
            if (auto* const element = document_->GetElementById(element_id)) {
                element->SetProperty("left", std::format("{:.1f}px", x));
            }
        };
        apply_left_toolbar_offset("primary-shelf-background", -primary_toolbar_x_);
        apply_left_toolbar_offset("primary-utility-toolbar", -primary_toolbar_x_);
        applied_primary_toolbar_x_ = primary_toolbar_x_;
        applied_primary_toolbar_width_ = primary_toolbar_width_;
        applied_show_secondary_toolbar_ = show_secondary_toolbar_;
        applied_secondary_toolbar_x_ = secondary_toolbar_x_;
        applied_secondary_toolbar_width_ = secondary_toolbar_width_;
        toolbar_roots_dirty_ = false;
        return true;
    }

    void RmlViewportOverlay::updateViewportContentOffset() {
        if (!document_ || !viewport_content_offset_dirty_)
            return;
        if (auto* const element = document_->GetElementById("viewport-content")) {
            element->SetProperty("left", std::format("{:.1f}px", viewport_content_offset_));
        }
        viewport_content_offset_dirty_ = false;
    }

    void RmlViewportOverlay::applySplitDividerOverlay() {
        if (!document_) {
            return;
        }

        if (auto* const overlay = document_->GetElementById("split-divider-overlay")) {
            overlay->SetClass("hidden", !split_divider_overlay_.visible);
            overlay->SetProperty("left", std::format("{:.1f}px", split_divider_overlay_.x));
            overlay->SetProperty("top", std::format("{:.1f}px", split_divider_overlay_.y));
            overlay->SetProperty("width", std::format("{:.1f}px", std::max(split_divider_overlay_.width, 0.0f)));
            overlay->SetProperty("height", std::format("{:.1f}px", std::max(split_divider_overlay_.height, 0.0f)));
            markRenderNeeded(RenderReason::SplitDivider);
        }
    }

    void RmlViewportOverlay::applyGTMetricsOverlay() {
        if (!document_) {
            return;
        }

        const auto metric_text = [](const std::string& text) -> Rml::String {
            return text.empty() ? Rml::String("--") : Rml::String(text);
        };
        bool touched = false;
        if (auto* const overlay = document_->GetElementById("gt-metrics-overlay")) {
            overlay->SetClass("hidden", !gt_metrics_overlay_.visible);
            overlay->SetProperty("left", std::format("{:.1f}px", gt_metrics_overlay_.x));
            overlay->SetProperty("top", std::format("{:.1f}px", gt_metrics_overlay_.y));
            touched = true;
        }
        if (auto* const psnr = document_->GetElementById("gt-metrics-psnr")) {
            psnr->SetInnerRML(metric_text(gt_metrics_overlay_.psnr_text));
            touched = true;
        }
        if (auto* const ssim_row = document_->GetElementById("gt-metrics-ssim-row")) {
            ssim_row->SetClass("hidden", !gt_metrics_overlay_.show_ssim);
            touched = true;
        }
        if (auto* const ssim = document_->GetElementById("gt-metrics-ssim")) {
            ssim->SetInnerRML(metric_text(gt_metrics_overlay_.ssim_text));
            touched = true;
        }

        if (touched)
            markRenderNeeded(RenderReason::GTMetrics);
    }

    void RmlViewportOverlay::applyLodStatsOverlay() {
        if (!document_) {
            return;
        }

        const auto value_text = [](const std::string& text) -> Rml::String {
            return text.empty() ? Rml::String("--") : Rml::String(text);
        };
        bool touched = false;
        if (auto* const overlay = document_->GetElementById("lod-stats-overlay")) {
            overlay->SetClass("hidden", !lod_stats_overlay_.visible);
            overlay->SetProperty("left", std::format("{:.1f}px", lod_stats_overlay_.x));
            overlay->SetProperty("top", std::format("{:.1f}px", lod_stats_overlay_.y));
            touched = true;
        }
        const auto set_text = [&](const char* id, const std::string& text) {
            if (auto* const element = document_->GetElementById(id)) {
                element->SetInnerRML(value_text(text));
                touched = true;
            }
        };
        set_text("lod-stats-status", lod_stats_overlay_.status_text);
        set_text("lod-stats-selected", lod_stats_overlay_.selected_text);
        set_text("lod-stats-budget", lod_stats_overlay_.budget_text);
        set_text("lod-stats-model", lod_stats_overlay_.model_text);
        set_text("lod-stats-tree", lod_stats_overlay_.tree_text);
        set_text("lod-stats-traversal", lod_stats_overlay_.traversal_text);
        set_text("lod-stats-stop", lod_stats_overlay_.stop_text);
        set_text("lod-stats-chunks", lod_stats_overlay_.chunks_text);
        set_text("lod-stats-cache", lod_stats_overlay_.cache_text);
        set_text("lod-stats-selector", lod_stats_overlay_.selector_text);
        set_text("lod-stats-pixel", lod_stats_overlay_.pixel_text);
        set_text("lod-stats-render", lod_stats_overlay_.render_text);
        set_text("lod-stats-foveation", lod_stats_overlay_.foveation_text);
        set_text("lod-stats-hash", lod_stats_overlay_.hash_text);

        if (touched)
            markRenderNeeded(RenderReason::LodStats);
    }

    void RmlViewportOverlay::processInput(const PanelInputState& input) {
        wants_input_ = false;
        if (!rml_context_ || !document_)
            return;
        if (vp_size_.x <= 0 || vp_size_.y <= 0)
            return;
        if (rml_manager_) {
            rml_manager_->trackContextFrame(rml_context_,
                                            static_cast<int>(vp_pos_.x - screen_origin_.x),
                                            static_cast<int>(vp_pos_.y - screen_origin_.y));
        }
        const bool external_mouse_capture = guiFocusState().want_capture_mouse;

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
        const bool mouse_moved =
            !mouse_pos_valid_ || rml_mx != last_mouse_x_ || rml_my != last_mouse_y_;
        const bool pointer_event =
            input.mouse_clicked[0] || input.mouse_released[0] ||
            input.mouse_clicked[1] || input.mouse_released[1] ||
            input.mouse_wheel != 0.0f;
        const bool pointer_drag =
            input.mouse_down[0] || input.mouse_down[1] || input.mouse_down[2];
        const bool keyboard_event =
            !input.keys_pressed.empty() || !input.keys_released.empty() ||
            !input.keys_repeated.empty() || !input.text_codepoints.empty() ||
            !input.text_inputs.empty() || input.has_text_editing;
        const bool vram_drag_capture = vram_hud_ && vram_hud_->isCapturingPointer();
        auto* const focused_before = rml_context_->GetFocusElement();
        const bool focused_text_target =
            focused_before &&
            (focused_before->GetTagName() == "input" ||
             focused_before->GetTagName() == "textarea");
        if (mouse_pos_valid_ && !mouse_moved && !pointer_event && !pointer_drag &&
            !keyboard_event && !vram_drag_capture) {
            wants_input_ = hovered_interactive_ || focused_text_target;
            auto* const hover = hovered_interactive_ ? rml_context_->GetHoverElement() : nullptr;
            const auto* const hover_root = viewportOverlayHoverRoot(hover);
            if (hovered_interactive_) {
                guiFocusState().want_capture_mouse = true;
                if (hover)
                    tooltip_.setHover(resolveRmlTooltip(hover), hover_root ? hover_root : hover);
            } else {
                tooltip_.setHover({}, nullptr);
            }
            if (focused_text_target)
                guiFocusState().want_capture_keyboard = true;
            return;
        }
        auto* const point_element = is_inside
                                        ? rml_context_->GetElementAtPoint(Rml::Vector2f(
                                              static_cast<float>(rml_mx),
                                              static_cast<float>(rml_my)))
                                        : nullptr;
        const bool point_interactive = viewportOverlayHoverRoot(point_element) != nullptr;
        const bool hover_target_changed = point_element != last_hover_element_;
        if (external_mouse_capture && !point_interactive && !hovered_interactive_ &&
            !vram_drag_capture) {
            tooltip_.setHover({}, nullptr);
            return;
        }
        const bool should_process_mouse_move =
            (mouse_moved || pointer_event) &&
            (was_inside || is_inside || vram_drag_capture) &&
            (pointer_event || pointer_drag || hover_target_changed ||
             hovered_interactive_ || was_inside != is_inside || vram_drag_capture) &&
            (is_inside || hovered_interactive_ || was_inside != is_inside ||
             vram_drag_capture);
        if (should_process_mouse_move) {
            mouse_pos_valid_ = true;
            last_mouse_x_ = rml_mx;
            last_mouse_y_ = rml_my;
            auto* const prev_hover = rml_context_->GetHoverElement();
            const auto* const prev_hover_root = viewportOverlayHoverRoot(prev_hover);
            rml_context_->ProcessMouseMove(rml_mx, rml_my, mods);
            auto* const next_hover = rml_context_->GetHoverElement();
            const auto* const next_hover_root = viewportOverlayHoverRoot(next_hover);
            if (pointer_event || pointer_drag || vram_drag_capture ||
                was_inside != is_inside || next_hover_root != prev_hover_root) {
                if (input.mouse_wheel != 0.0f)
                    markRenderNeeded(RenderReason::PointerWheel);
                if (input.mouse_clicked[0] || input.mouse_released[0] ||
                    input.mouse_clicked[1] || input.mouse_released[1])
                    markRenderNeeded(RenderReason::PointerButton);
                if (pointer_drag || vram_drag_capture)
                    markRenderNeeded(RenderReason::PointerDrag);
                if (was_inside != is_inside || next_hover_root != prev_hover_root)
                    markRenderNeeded(RenderReason::PointerHover);
            }
        } else if (mouse_moved && (was_inside || is_inside || vram_drag_capture)) {
            mouse_pos_valid_ = true;
            last_mouse_x_ = rml_mx;
            last_mouse_y_ = rml_my;
        }
        last_hover_element_ = point_element;

        auto* const hover = should_process_mouse_move ? rml_context_->GetHoverElement()
                                                      : point_element;
        const auto* const hover_root = viewportOverlayHoverRoot(hover);
        const bool over_interactive = is_inside && hover_root != nullptr;
        hovered_interactive_ = over_interactive;

        if (over_interactive || vram_drag_capture) {
            wants_input_ = true;
            guiFocusState().want_capture_mouse = true;

            if (input.mouse_clicked[0]) {
                markRenderNeeded(RenderReason::PointerButton);
                rml_context_->ProcessMouseButtonDown(0, mods);
            }
            if (input.mouse_released[0]) {
                markRenderNeeded(RenderReason::PointerButton);
                rml_context_->ProcessMouseButtonUp(0, mods);
            }
            if (input.mouse_clicked[1]) {
                markRenderNeeded(RenderReason::PointerButton);
                rml_context_->ProcessMouseButtonDown(1, mods);
            }
            if (input.mouse_released[1]) {
                markRenderNeeded(RenderReason::PointerButton);
                rml_context_->ProcessMouseButtonUp(1, mods);
            }
            if (input.mouse_wheel != 0.0f) {
                markRenderNeeded(RenderReason::PointerWheel);
                rml_context_->ProcessMouseWheel(Rml::Vector2f(0.0f, -input.mouse_wheel), mods);
            }

            if (hover)
                tooltip_.setHover(resolveRmlTooltip(hover), hover_root ? hover_root : hover);
        } else {
            tooltip_.setHover({}, nullptr);
        }

        // Forward keyboard + text input whenever an RmlUi element on this context owns focus
        // (e.g. the Annotations / Drill-down filter <input>). This must run regardless of
        // over_interactive, because a text input keeps focus even when the mouse roams away.
        if (auto* focused = rml_context_->GetFocusElement()) {
            const auto tag = focused->GetTagName();
            const bool is_text_target = tag == "input" || tag == "textarea";
            if (is_text_target) {
                wants_input_ = true;
                guiFocusState().want_capture_keyboard = true;
                // Numpad digit and period scancodes must be suppressed from
                // ProcessKeyDown / ProcessKeyUp when a text input is focused,
                // otherwise RmlUi treats them as navigation keys (Home, End,
                // arrows, etc.). The actual digit text arrives via
                // ProcessTextInput below. This mirrors the fix in
                // rml_panel_host.cpp for the sidebar text inputs.
                auto isNumpadTextKey = [](int sc) {
                    return (sc >= SDL_SCANCODE_KP_1 && sc <= SDL_SCANCODE_KP_0) ||
                           sc == SDL_SCANCODE_KP_PERIOD;
                };
                for (const int sc : input.keys_pressed) {
                    if (isNumpadTextKey(sc))
                        continue;
                    const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                    if (rml_key != Rml::Input::KI_UNKNOWN) {
                        markRenderNeeded(RenderReason::Keyboard);
                        rml_context_->ProcessKeyDown(rml_key, mods);
                    }
                }
                for (const int sc : input.keys_released) {
                    if (isNumpadTextKey(sc))
                        continue;
                    const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                    if (rml_key != Rml::Input::KI_UNKNOWN) {
                        markRenderNeeded(RenderReason::Keyboard);
                        rml_context_->ProcessKeyUp(rml_key, mods);
                    }
                }
                for (uint32_t cp : input.text_codepoints) {
                    markRenderNeeded(RenderReason::Keyboard);
                    rml_context_->ProcessTextInput(static_cast<Rml::Character>(cp));
                }
            }
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
        return isInteractiveViewportOverlayElement(hover);
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
                toolbar_roots_dirty_ = true;
            } else {
                LOG_WARN("RmlViewportOverlay: missing body template for stale data-model rebind");
                body->RemoveChild(existing);
            }
            markRenderNeeded(RenderReason::DataModelBinding);
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
        markRenderNeeded(RenderReason::DataModelBinding);

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
        applySplitDividerOverlay();
        applyLodStatsOverlay();
        if (vram_hud_)
            vram_hud_->onDocumentLoaded(document_);
    }

    bool RmlViewportOverlay::applyFrameTooltip() {
        if (!document_)
            return false;
        if (!body_el_)
            body_el_ = document_->GetElementById("overlay-body");
        return tooltip_.apply(body_el_,
                              last_mouse_x_, last_mouse_y_,
                              static_cast<int>(vp_size_.x),
                              static_cast<int>(vp_size_.y));
    }

    void RmlViewportOverlay::queueCachedVulkanContext(const bool refresh_cache) {
        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;
        const float x = vp_pos_.x - screen_origin_.x;
        const float y = vp_pos_.y - screen_origin_.y;
        const int w = static_cast<int>(vp_size_.x);
        const int h = static_cast<int>(vp_size_.y);
        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = w,
            .cache_height = h,
            .offset_x = x,
            .offset_y = y,
            .draw_width = vp_size_.x,
            .draw_height = vp_size_.y,
            .refresh = refresh_cache,
            .foreground = false,
            .clip_enabled = true,
            .clip = {
                .x1 = x,
                .y1 = y,
                .x2 = x + vp_size_.x,
                .y2 = y + vp_size_.y,
            },
        });
    }

    void RmlViewportOverlay::renderCached() {
        if (!rml_context_ || !document_)
            return;
        if (vp_size_.x <= 0 || vp_size_.y <= 0)
            return;

        const int w = static_cast<int>(vp_size_.x);
        const int h = static_cast<int>(vp_size_.y);
        const bool theme_current =
            has_theme_signature_ && rml_theme::currentThemeSignature() == last_theme_signature_;
        const bool document_hooks_due = shouldRunAnyDocumentHooks(false);
        const bool builtin_document_sync_due = document_sync_dirty_;
        bool tooltip_changed = false;
        if (tooltip_.hasActiveState()) {
            LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render.tooltip", 0.25);
            tooltip_changed = applyFrameTooltip();
        }
        if (rml_manager_)
            rml_manager_->setContextNeedsPassiveMouseMoveFrames(rml_context_, tooltip_.needsFrame());
        const bool can_update_tooltip_only =
            rml_manager_ && tooltip_changed && theme_current && !document_hooks_due &&
            !builtin_document_sync_due && !render_needed_ && !animation_active_ &&
            !data_model_binding_dirty_ && !toolbar_roots_dirty_ &&
            w == last_render_w_ && h == last_render_h_;
        if (can_update_tooltip_only) {
            rml_manager_->trackContextFrame(rml_context_,
                                            static_cast<int>(vp_pos_.x - screen_origin_.x),
                                            static_cast<int>(vp_pos_.y - screen_origin_.y));
            rml_context_->SetDimensions(Rml::Vector2i(w, h));
            rml_context_->Update();
            queueCachedVulkanContext(true);
            animation_active_ = (rml_context_->GetNextUpdateDelay() == 0);
            return;
        }
        const bool can_reuse = theme_current && !render_needed_ && !animation_active_ &&
                               !document_hooks_due && !builtin_document_sync_due &&
                               !data_model_binding_dirty_ && !toolbar_roots_dirty_ &&
                               !tooltip_changed && w == last_render_w_ && h == last_render_h_;
        if (!can_reuse) {
            render();
            return;
        }

        queueCachedVulkanContext(direct_cache_.texture == 0 ||
                                 direct_cache_.width != w ||
                                 direct_cache_.height != h);
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
        const bool toolbar_changed = updateToolbarRoots();
        updateViewportContentOffset();
        const bool document_force = theme_changed || size_changed || toolbar_changed;
        bool document_dirty = syncBuiltinDocument(document_force);
        const bool run_prepend_document_hooks = shouldRunDocumentHooks(document_force, true);
        const bool run_append_document_hooks = shouldRunDocumentHooks(document_force, false);
        if (run_prepend_document_hooks || run_append_document_hooks) {
            LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render.document_hooks", 0.25);
            bool hook_dirty = false;
            if (run_prepend_document_hooks)
                hook_dirty |= lfs::python::invoke_python_document_hooks(
                    "viewport_overlay", "document", document_, true);
            if (run_append_document_hooks)
                hook_dirty |= lfs::python::invoke_python_document_hooks(
                    "viewport_overlay", "document", document_, false);
            if (hook_dirty)
                markRenderNeeded(RenderReason::DocumentHook);
            document_dirty |= hook_dirty;
            last_document_hook_run_ = std::chrono::steady_clock::now();
        }

        if (!body_el_)
            body_el_ = document_->GetElementById("overlay-body");
        const bool had_data_model_binding_dirty = data_model_binding_dirty_;
        if (had_data_model_binding_dirty) {
            LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render.bind_data_model", 0.25);
            markRenderNeeded(RenderReason::DataModelBinding);
            ensureBodyDataModelBound(body_el_);
            data_model_binding_dirty_ = false;
            document_dirty |= syncBuiltinDocument(true);
        }
        bool tooltip_changed = false;
        if (tooltip_.hasActiveState()) {
            LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render.tooltip", 0.25);
            tooltip_changed = applyFrameTooltip();
        }
        if (rml_manager_)
            rml_manager_->setContextNeedsPassiveMouseMoveFrames(rml_context_, tooltip_.needsFrame());

        const bool needs_render = render_needed_ || animation_active_ || document_dirty ||
                                  theme_changed || size_changed || toolbar_changed ||
                                  tooltip_changed || had_data_model_binding_dirty;
        if (!needs_render) {
            queueCachedVulkanContext(direct_cache_.texture == 0 ||
                                     direct_cache_.width != w ||
                                     direct_cache_.height != h);
            return;
        }

        LOG_PERF("gui_render.rml_viewport_overlay.render_reasons render_needed={} sources={} animation={} document_dirty={} theme={} size={} toolbar={} tooltip={} data_model={}",
                 render_needed_,
                 renderReasonSources(),
                 animation_active_,
                 document_dirty,
                 theme_changed,
                 size_changed,
                 toolbar_changed,
                 tooltip_changed,
                 had_data_model_binding_dirty);
        {
            LOG_TIMER("gui_render.rml_viewport_overlay.render.update");
            {
                LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render.update.track_context", 0.25);
                rml_manager_->trackContextFrame(rml_context_,
                                                static_cast<int>(vp_pos_.x - screen_origin_.x),
                                                static_cast<int>(vp_pos_.y - screen_origin_.y));
            }
            {
                LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render.update.set_dimensions", 0.25);
                rml_context_->SetDimensions(Rml::Vector2i(w, h));
            }
            {
                LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render.update.context_update", 0.25);
                rml_context_->Update();
            }
        }

        queueCachedVulkanContext(true);
        {
            LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render.update.next_delay", 0.25);
            animation_active_ = (rml_context_->GetNextUpdateDelay() == 0);
        }
        render_needed_ = false;
        render_reason_bits_ = 0;
        last_render_w_ = w;
        last_render_h_ = h;
    }

} // namespace lfs::vis::gui
