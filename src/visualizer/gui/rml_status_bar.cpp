/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_status_bar.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "gui/gpu_memory_query.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_context.hpp"
#include "internal/resource_paths.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/training_manager.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer_impl.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <SDL3/SDL_clipboard.h>
#include <SDL3/SDL_video.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <optional>
#include <vector>

#include "git_version.h"

namespace lfs::vis::gui {

    using rml_theme::colorToRml;
    using rml_theme::colorToRmlAlpha;

    namespace {
        class GitCommitClickListener final : public Rml::EventListener {
        public:
            explicit GitCommitClickListener(const std::string* commit) : commit_(commit) {}

            void ProcessEvent(Rml::Event& /*event*/) override {
                if (commit_->empty())
                    return;
                SDL_SetClipboardText(commit_->c_str());
                LOG_INFO("Copied commit {} to clipboard", *commit_);
            }

        private:
            const std::string* commit_;
        };

        // Clicking the GPU icon toggles the performance HUD visibility. Profiler
        // collection remains an independent deep-diagnostics concern.
        class VramHudToggleListener final : public Rml::EventListener {
        public:
            void ProcessEvent(Rml::Event& /*event*/) override {
                lfs::core::events::ui::ToggleVramHud{}.emit();
            }
        };

        class AccountPanelOpenListener final : public Rml::EventListener {
        public:
            void ProcessEvent(Rml::Event& /*event*/) override {
                PanelRegistry::instance().set_panel_enabled("lfs.account", true);
            }
        };

        class CallbackListener final : public Rml::EventListener {
        public:
            explicit CallbackListener(std::function<void()> callback)
                : callback_(std::move(callback)) {}

            void ProcessEvent(Rml::Event& event) override {
                event.StopPropagation();
                callback_();
            }

        private:
            std::function<void()> callback_;
        };

        std::string fmtCount(int64_t n) {
            if (n >= 1'000'000)
                return std::format("{:.2f}M", n / 1e6);
            if (n >= 1'000)
                return std::format("{:.0f}K", n / 1e3);
            return std::to_string(n);
        }

        std::string formatLocalizedValue(const std::string& pattern,
                                         const std::string& value) noexcept {
            try {
                return std::vformat(pattern, std::make_format_args(value));
            } catch (const std::format_error&) {
                return pattern;
            }
        }

        std::string fmtTime(float secs) {
            if (secs < 0)
                return "--:--";
            int total = static_cast<int>(secs);
            int h = total / 3600;
            int m = (total % 3600) / 60;
            int s = total % 60;
            if (h > 0)
                return std::format("{}:{:02d}:{:02d}", h, m, s);
            return std::format("{}:{:02d}", m, s);
        }

        std::string stripColon(const std::string& s) {
            auto end = s.find_last_not_of(": ");
            if (end == std::string::npos)
                return s;
            return s.substr(0, end + 1);
        }

        std::string formatStepLabel(const size_t step) {
            return std::format("{} {}", stripColon(LOC(lichtfeld::Strings::Status::STEP)), step);
        }

        // Width the element's content box would need to show everything on one line.
        // RmlUi clamps a flex item's base size to the containing block, so a child that is
        // already too narrow under-reports its own size; recursing past it recovers the real
        // requirement. Overflow beyond the padding box covers text that outgrew its own box.
        float neededContentWidth(Rml::Element* element) {
            assert(element);

            float children_width = 0.0f;
            for (int i = 0; i < element->GetNumChildren(); ++i) {
                auto* const child = element->GetChild(i);
                // A display:none element keeps its last laid-out box, so its width must not be
                // read back; only the computed display tells us it is gone.
                if (!child || child->GetDisplay() == Rml::Style::Display::None)
                    continue;

                const auto position = child->GetComputedValues().position();
                if (position == Rml::Style::Position::Absolute ||
                    position == Rml::Style::Position::Fixed)
                    continue;

                const auto& child_box = child->GetBox();
                const float outer_width = child_box.GetSize(Rml::BoxArea::Margin).x;
                const float frame_width = outer_width - child_box.GetSize().x;
                children_width += std::max(outer_width, neededContentWidth(child) + frame_width);
            }

            // RmlUi only maintains a scrollable overflow rectangle on elements that clip, and it
            // is where a descendant whose box was clamped shows up as extra width.
            const bool clips =
                element->GetComputedValues().overflow_x() != Rml::Style::Overflow::Visible;
            const float overflow_width =
                clips ? std::max(0.0f, element->GetScrollWidth() - element->GetClientWidth()) : 0.0f;
            const float own_width =
                children_width > 0.0f ? children_width : element->GetBox().GetSize().x;
            return own_width + overflow_width;
        }

        struct ProgressMarkerRenderState {
            bool dragging = false;
            bool adding = false;
            size_t original_step = 0;
            size_t preview_step = 0;
            size_t hover_step = 0;
        };

        void appendProgressMarkerRml(std::string& markers,
                                     const size_t step,
                                     const int total_iterations,
                                     const bool past,
                                     const bool hovered,
                                     const bool preview) {
            const float left_pct = std::clamp(
                100.0f * static_cast<float>(step) / static_cast<float>(total_iterations),
                0.5f,
                99.5f);

            std::string classes = "progress-marker";
            if (past)
                classes += " is-past";
            if (hovered)
                classes += " is-hovered";
            if (preview)
                classes += " is-preview";

            markers += std::format(
                "<div class=\"{}\" style=\"left:{:.3f}%;\"></div>",
                classes,
                left_pct);
        }

        std::string buildProgressMarkersRml(std::vector<size_t> save_steps,
                                            const int total_iterations,
                                            const int current_iteration,
                                            const ProgressMarkerRenderState& state) {
            if (total_iterations <= 0)
                return {};

            std::ranges::sort(save_steps);
            save_steps.erase(std::ranges::unique(save_steps).begin(), save_steps.end());

            std::string markers;
            markers.reserve((save_steps.size() + 1) * 136);
            for (const size_t save_step : save_steps) {
                if (save_step == 0 || save_step > static_cast<size_t>(total_iterations))
                    continue;
                if (state.dragging && !state.adding && save_step == state.original_step)
                    continue;

                appendProgressMarkerRml(markers,
                                        save_step,
                                        total_iterations,
                                        save_step <= static_cast<size_t>(std::max(0, current_iteration)),
                                        !state.dragging && save_step == state.hover_step,
                                        false);
            }

            if (state.dragging && state.preview_step > 0 &&
                state.preview_step <= static_cast<size_t>(total_iterations)) {
                appendProgressMarkerRml(markers,
                                        state.preview_step,
                                        total_iterations,
                                        false,
                                        true,
                                        true);
            }
            return markers;
        }

        struct FramebufferBlitRect {
            float x = 0.0f;
            float y = 0.0f;
            float w = 0.0f;
            float h = 0.0f;
            int screen_w = 0;
            int screen_h = 0;
        };

        FramebufferBlitRect toFramebufferBlitRect(SDL_Window* window,
                                                  float x, float y, float w, float h,
                                                  int screen_w, int screen_h) {
            FramebufferBlitRect rect{x, y, w, h, screen_w, screen_h};
            if (!window || screen_w <= 0 || screen_h <= 0)
                return rect;

            int framebuffer_w = 0;
            int framebuffer_h = 0;
            SDL_GetWindowSizeInPixels(window, &framebuffer_w, &framebuffer_h);
            if (framebuffer_w <= 0 || framebuffer_h <= 0)
                return rect;

            const float scale_x = static_cast<float>(framebuffer_w) / static_cast<float>(screen_w);
            const float scale_y = static_cast<float>(framebuffer_h) / static_cast<float>(screen_h);
            rect.x = std::round(x * scale_x);
            rect.y = std::round(y * scale_y);
            rect.w = std::max(1.0f, std::round(w * scale_x));
            rect.h = std::max(1.0f, std::round(h * scale_y));
            rect.screen_w = framebuffer_w;
            rect.screen_h = framebuffer_h;
            return rect;
        }
    } // namespace

    // StatusMessageState

    void StatusMessageState::post(std::string text, const ErrorNoticeLevel level) {
        std::lock_guard lock(mutex_);
        text_ = std::move(text);
        level_ = level;
        posted_at_ = std::chrono::steady_clock::now();
        has_message_ = true;
    }

    StatusMessageState::Snapshot
    StatusMessageState::snapshot(const std::chrono::steady_clock::time_point now) {
        std::lock_guard lock(mutex_);
        Snapshot snap;
        if (!has_message_)
            return snap;

        const auto elapsed = now - posted_at_;
        if (elapsed >= kDuration) {
            has_message_ = false;
            return snap;
        }

        snap.visible = true;
        snap.text = text_;
        snap.level = level_;
        const auto remaining =
            kDuration - std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        const float remaining_ms = static_cast<float>(remaining.count());
        snap.alpha = remaining_ms < kFadeMs ? remaining_ms / kFadeMs : 1.0f;
        return snap;
    }

    // SpeedOverlayState

    void RmlStatusBar::SpeedOverlayState::showWasd(float speed) {
        wasd_speed = speed;
        wasd_visible = true;
        wasd_start = std::chrono::steady_clock::now();
    }

    void RmlStatusBar::SpeedOverlayState::showZoom(float speed) {
        zoom_speed = speed;
        zoom_visible = true;
        zoom_start = std::chrono::steady_clock::now();
    }

    std::pair<float, float> RmlStatusBar::SpeedOverlayState::getWasd() const {
        if (!wasd_visible)
            return {0.0f, 0.0f};
        auto now = std::chrono::steady_clock::now();
        if (now - wasd_start >= DURATION)
            return {0.0f, 0.0f};
        auto remaining = DURATION - std::chrono::duration_cast<std::chrono::milliseconds>(now - wasd_start);
        float alpha = (remaining.count() < FADE_MS) ? remaining.count() / FADE_MS : 1.0f;
        return {wasd_speed, alpha};
    }

    std::pair<float, float> RmlStatusBar::SpeedOverlayState::getZoom() const {
        if (!zoom_visible)
            return {0.0f, 0.0f};
        auto now = std::chrono::steady_clock::now();
        if (now - zoom_start >= DURATION)
            return {0.0f, 0.0f};
        auto remaining = DURATION - std::chrono::duration_cast<std::chrono::milliseconds>(now - zoom_start);
        float alpha = (remaining.count() < FADE_MS) ? remaining.count() / FADE_MS : 1.0f;
        return {zoom_speed, alpha};
    }

    // RmlStatusBar

    void RmlStatusBar::init(RmlUIManager* mgr, const bool safe_mode,
                            std::function<RuntimeServiceStatus()> mcp_status_provider) {
        assert(mgr);
        rml_manager_ = mgr;
        mcp_status_provider_ = std::move(mcp_status_provider);
        model_.safe_mode = safe_mode;

        const auto& palette = lfs::vis::theme().palette;
        model_.mode_color = colorToRml(palette.text_dim);
        model_.splat_color = colorToRml(palette.text);
        model_.split_mode_color = colorToRml(palette.warning);
        model_.wasd_color = colorToRml(palette.info);
        model_.wasd_sep_color = colorToRml(palette.text_dim);
        model_.zoom_color = colorToRml(palette.info);
        model_.zoom_sep_color = colorToRml(palette.text_dim);
        model_.account_color = colorToRml(palette.text_dim);
        model_.mcp_color = colorToRml(palette.text_dim);
        model_.lfs_mem_color = colorToRml(palette.info);
        model_.gpu_mem_color = colorToRml(palette.text);
        model_.fps_color = colorToRml(palette.success);
        model_.status_message_color = colorToRml(palette.info);

        rml_context_ = rml_manager_->createContext("status_bar", 800, 22);
        if (!rml_context_) {
            LOG_ERROR("RmlStatusBar: failed to create RML context");
            return;
        }

        auto ctor = rml_context_->CreateDataModel("status_bar");
        assert(ctor);
        ctor.Bind("safe_mode", &model_.safe_mode);
        ctor.Bind("safe_mode_text", &model_.safe_mode_text);
        ctor.Bind("mode_text", &model_.mode_text);
        ctor.Bind("mode_color", &model_.mode_color);
        ctor.Bind("show_training", &model_.show_training);
        ctor.Bind("progress_width", &model_.progress_width);
        ctor.Bind("progress_text", &model_.progress_text);
        ctor.Bind("step_label", &model_.step_label);
        ctor.Bind("step_value", &model_.step_value);
        ctor.Bind("loss_label", &model_.loss_label);
        ctor.Bind("loss_value", &model_.loss_value);
        ctor.Bind("show_eval_metrics", &model_.show_eval_metrics);
        ctor.Bind("eval_metrics_value", &model_.eval_metrics_value);
        ctor.Bind("gaussians_label", &model_.gaussians_label);
        ctor.Bind("gaussians_value", &model_.gaussians_value);
        ctor.Bind("time_value", &model_.time_value);
        ctor.Bind("eta_label", &model_.eta_label);
        ctor.Bind("eta_value", &model_.eta_value);
        ctor.Bind("show_splats", &model_.show_splats);
        ctor.Bind("splat_text", &model_.splat_text);
        ctor.Bind("splat_color", &model_.splat_color);
        ctor.Bind("show_split", &model_.show_split);
        ctor.Bind("split_mode", &model_.split_mode);
        ctor.Bind("split_mode_color", &model_.split_mode_color);
        ctor.Bind("split_detail", &model_.split_detail);
        ctor.Bind("show_wasd", &model_.show_wasd);
        ctor.Bind("wasd_text", &model_.wasd_text);
        ctor.Bind("wasd_color", &model_.wasd_color);
        ctor.Bind("wasd_sep_color", &model_.wasd_sep_color);
        ctor.Bind("show_zoom", &model_.show_zoom);
        ctor.Bind("zoom_text", &model_.zoom_text);
        ctor.Bind("zoom_color", &model_.zoom_color);
        ctor.Bind("zoom_sep_color", &model_.zoom_sep_color);
        ctor.Bind("account_label", &model_.account_label);
        ctor.Bind("account_tier", &model_.account_tier);
        ctor.Bind("account_tooltip", &model_.account_tooltip);
        ctor.Bind("account_color", &model_.account_color);
        ctor.Bind("account_show_tier", &model_.account_show_tier);
        ctor.Bind("account_membership_required", &model_.account_membership_required);
        ctor.Bind("lfs_mem_text", &model_.lfs_mem_text);
        ctor.Bind("lfs_mem_color", &model_.lfs_mem_color);
        ctor.Bind("show_gpu_model", &model_.show_gpu_model);
        ctor.Bind("gpu_panel_active", &model_.gpu_panel_active);
        ctor.Bind("gpu_model_text", &model_.gpu_model_text);
        ctor.Bind("gpu_mem_text", &model_.gpu_mem_text);
        ctor.Bind("gpu_mem_color", &model_.gpu_mem_color);
        ctor.Bind("fps_value", &model_.fps_value);
        ctor.Bind("fps_color", &model_.fps_color);
        ctor.Bind("fps_label", &model_.fps_label);
        ctor.Bind("git_commit", &model_.git_commit);
        ctor.Bind("mcp_details_expanded", &model_.mcp_details_expanded);
        ctor.Bind("mcp_summary", &model_.mcp_summary);
        ctor.Bind("mcp_details", &model_.mcp_details);
        ctor.Bind("mcp_tooltip", &model_.mcp_tooltip);
        ctor.Bind("mcp_color", &model_.mcp_color);
        ctor.Bind("mcp_preferences_label", &model_.mcp_preferences_label);
        ctor.Bind("mcp_server_enabled", &model_.mcp_server_enabled);
        ctor.Bind("mcp_toggle_label", &model_.mcp_toggle_label);
        ctor.Bind("mcp_total_text", &model_.mcp_total_text);
        ctor.Bind("mcp_success_text", &model_.mcp_success_text);
        ctor.Bind("mcp_error_text", &model_.mcp_error_text);
        ctor.Bind("show_status_message", &model_.show_status_message);
        ctor.Bind("status_message_text", &model_.status_message_text);
        ctor.Bind("status_message_color", &model_.status_message_color);
        model_handle_ = ctor.GetModelHandle();

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/statusbar.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlStatusBar: failed to load statusbar.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlStatusBar: resource not found: {}", e.what());
            return;
        }

        attachElementListeners();
        bindReactiveStore();

        if (!speed_events_initialized_) {
            lfs::core::events::ui::SpeedChanged::when([this](const auto& e) {
                speed_state_.showWasd(e.current_speed);
                animation_active_ = true;
                next_refresh_at_ = {};
                markModelDirty();
            });
            lfs::core::events::ui::ZoomSpeedChanged::when([this](const auto& e) {
                speed_state_.showZoom(e.zoom_speed);
                animation_active_ = true;
                next_refresh_at_ = {};
                markModelDirty();
            });
            speed_events_initialized_ = true;
        }

        updateTheme();
    }

    void RmlStatusBar::shutdown() {
        if (pending_gpu_mem_.valid()) {
            pending_gpu_mem_.wait();
            try {
                cached_gpu_mem_ = pending_gpu_mem_.get();
            } catch (const std::exception& e) {
                LOG_WARN("RmlStatusBar: GPU memory query failed during shutdown: {}", e.what());
            }
        }

        subscriptions_.clear();
        model_handle_ = {};
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("status_bar");
        rml_context_ = nullptr;
        document_ = nullptr;
        delete git_commit_listener_;
        git_commit_listener_ = nullptr;
        delete gpu_icon_listener_;
        gpu_icon_listener_ = nullptr;
        delete account_listener_;
        account_listener_ = nullptr;
        delete mcp_toggle_listener_;
        mcp_toggle_listener_ = nullptr;
        delete mcp_power_listener_;
        mcp_power_listener_ = nullptr;
        delete mcp_preferences_listener_;
        mcp_preferences_listener_ = nullptr;
    }

    void RmlStatusBar::reloadResources() {
        if (!rml_context_)
            return;

        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        base_rcss_.clear();
        has_theme_signature_ = false;
        model_.progress_markers_rml.clear();
        model_dirty_ = true;
        animation_active_ = true;
        fit_level_ = 0;
        last_dp_ratio_ = 0.0f;
        last_section_signature_ = 0;
        last_render_w_ = 0;
        last_render_h_ = 0;
        last_document_h_ = 0;
        next_refresh_at_ = {};

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/statusbar.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlStatusBar: failed to reload statusbar.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlStatusBar: resource not found during reload: {}", e.what());
            return;
        }

        attachElementListeners();
        bindReactiveStore();

        updateTheme();
    }

    void RmlStatusBar::bindReactiveStore() {
        subscriptions_.clear();
        auto& store = lfs::vis::app_store();
        const auto bind = [this](auto& observable) {
            subscriptions_.push_back(observable.subscribe([this](const auto&) {
                markModelDirty();
            }));
        };

        bind(store.iteration);
        bind(store.total_iterations);
        bind(store.loss);
        bind(store.num_gaussians);
        bind(store.max_gaussians);
        bind(store.training_running);
        bind(store.training_state);
        bind(store.trainer_loaded);
        bind(store.eval_psnr);
        bind(store.eval_ssim);
        bind(store.scene_generation);
        bind(store.selection_generation);
        subscriptions_.push_back(store.fps.subscribe([this](const float& fps) {
            reactive_fps_available_ = true;
            reactive_fps_value_ = fps;
            markModelDirty();
        }));
        bind(store.mode_text);
        bind(store.account_state);
        subscriptions_.push_back(store.perf_hud.subscribe([this](const lfs::vis::AppStore::PerfHud& state) {
            setModelBool("gpu_panel_active", model_.gpu_panel_active, state.visible);
            markModelDirty();
        }));
    }

    void RmlStatusBar::markModelDirty() {
        model_dirty_ = true;
        next_refresh_at_ = {};
    }

    void RmlStatusBar::postStatusMessage(std::string text, const ErrorNoticeLevel level) {
        status_message_.post(std::move(text), level);
    }

    bool RmlStatusBar::updateTheme() {
        if (!document_)
            return false;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return false;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/statusbar.rcss");

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/statusbar.theme.rcss"));
        model_dirty_ = true;
        return true;
    }

    bool RmlStatusBar::layoutFits(const float reserve_px) const {
        assert(reserve_px >= 0.0f);
        if (!document_)
            return false;

        auto* const body = document_->GetElementById("body");
        auto* const status_left = document_->GetElementById("status-left");
        auto* const status_right = document_->GetElementById("status-right");
        assert(body);
        assert(status_left);
        assert(status_right);
        if (!body || !status_left || !status_right)
            return false;

        constexpr float tolerance_px = 0.5f;
        const auto height_fits = [](Rml::Element* element) {
            return element->GetScrollHeight() <= element->GetClientHeight() + tolerance_px;
        };

        return height_fits(body) && height_fits(status_left) && height_fits(status_right) &&
               neededContentWidth(status_left) + reserve_px <=
                   status_left->GetBox().GetSize().x + tolerance_px &&
               neededContentWidth(status_right) <=
                   status_right->GetBox().GetSize().x + tolerance_px;
    }

    void RmlStatusBar::applyFitLevel(const int level) {
        assert(level >= 0 && level <= kMaxFitLevel);
        if (!document_)
            return;

        auto* const body = document_->GetElementById("body");
        assert(body);
        if (!body)
            return;

        static constexpr const char* fit_classes[kMaxFitLevel] = {
            "fit-1", "fit-2", "fit-3", "fit-4", "fit-5",
            "fit-6", "fit-7", "fit-8", "fit-9"};
        for (int fit_class = 1; fit_class <= kMaxFitLevel; ++fit_class)
            body->SetClass(fit_classes[fit_class - 1], level >= fit_class);
    }

    void RmlStatusBar::fitToAvailableWidth(const bool allow_expand) {
        if (!document_ || !rml_context_ || save_step_interaction_.dragging)
            return;

        const int entry_level = fit_level_;
        while (!layoutFits(0.0f) && fit_level_ < kMaxFitLevel) {
            ++fit_level_;
            applyFitLevel(fit_level_);
            rml_context_->Update();
        }

        if (allow_expand) {
            const float reserve_px = 8.0f * rml_context_->GetDensityIndependentPixelRatio();
            while (fit_level_ > 0) {
                const int previous_level = fit_level_;
                --fit_level_;
                applyFitLevel(fit_level_);
                rml_context_->Update();
                if (layoutFits(reserve_px))
                    continue;

                fit_level_ = previous_level;
                applyFitLevel(fit_level_);
                rml_context_->Update();
                break;
            }
        }

        if (fit_level_ != entry_level)
            LOG_DEBUG("Status bar fit level {} -> {} at {} px", entry_level, fit_level_,
                      rml_context_->GetDimensions().x);

#ifndef NDEBUG
        for (const char* const id : {"body", "status-left", "status-right"}) {
            auto* const element = document_->GetElementById(id);
            assert(element);
            assert(element->GetScrollHeight() <= element->GetClientHeight() + 0.5f);
        }
#endif
    }

    void RmlStatusBar::pollGpuMemoryQuery(const std::chrono::steady_clock::time_point now) {
        if (pending_gpu_mem_.valid() &&
            pending_gpu_mem_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            try {
                cached_gpu_mem_ = pending_gpu_mem_.get();
            } catch (const std::exception& e) {
                LOG_WARN("RmlStatusBar: GPU memory query failed: {}", e.what());
            }
        }

        if (pending_gpu_mem_.valid())
            return;

        if (next_gpu_refresh_at_ != std::chrono::steady_clock::time_point{} &&
            now < next_gpu_refresh_at_) {
            return;
        }

        next_gpu_refresh_at_ = now + kGpuRefreshInterval;
        pending_gpu_mem_ = std::async(std::launch::async, [] {
            return queryGpuMemory();
        });
    }

    void RmlStatusBar::attachElementListeners() {
        if (!document_)
            return;

        if (!git_commit_listener_)
            git_commit_listener_ = new GitCommitClickListener(&model_.git_commit);
        if (auto* el = document_->GetElementById("git-commit"))
            el->AddEventListener(Rml::EventId::Click, git_commit_listener_);

        if (!gpu_icon_listener_)
            gpu_icon_listener_ = new VramHudToggleListener();
        if (auto* el = document_->GetElementById("gpu-icon"))
            el->AddEventListener(Rml::EventId::Click, gpu_icon_listener_);

        if (!account_listener_)
            account_listener_ = new AccountPanelOpenListener();
        if (auto* el = document_->GetElementById("account-chip"))
            el->AddEventListener(Rml::EventId::Click, account_listener_);

        if (!mcp_toggle_listener_) {
            mcp_toggle_listener_ = new CallbackListener([this] {
                model_.mcp_details_expanded = !model_.mcp_details_expanded;
                model_handle_.DirtyVariable("mcp_details_expanded");
                markModelDirty();
            });
        }
        if (auto* el = document_->GetElementById("mcp-chip"))
            el->AddEventListener(Rml::EventId::Click, mcp_toggle_listener_);

        if (!mcp_preferences_listener_) {
            mcp_preferences_listener_ = new CallbackListener([this] {
                setModelBool("mcp_details_expanded", model_.mcp_details_expanded, false);
                openPreferencesPanel("mcp");
            });
        }
        if (auto* el = document_->GetElementById("mcp-preferences"))
            el->AddEventListener(Rml::EventId::Click, mcp_preferences_listener_);

        if (!mcp_power_listener_) {
            mcp_power_listener_ = new CallbackListener([] {
                lfs::vis::toggleMcpRuntimeEnabled();
            });
        }
        if (auto* el = document_->GetElementById("mcp-toggle"))
            el->AddEventListener(Rml::EventId::Click, mcp_power_listener_);
    }

    void RmlStatusBar::setModelString(const char* name, std::string& field, std::string value) {
        if (field == value)
            return;
        field = std::move(value);
        model_handle_.DirtyVariable(name);
        model_dirty_ = true;
    }

    void RmlStatusBar::setModelBool(const char* name, bool& field, bool value) {
        if (field == value)
            return;
        field = value;
        model_handle_.DirtyVariable(name);
        model_dirty_ = true;
    }

    void RmlStatusBar::setProgressMarkersRml(std::string value) {
        if (model_.progress_markers_rml == value)
            return;

        model_.progress_markers_rml = std::move(value);
        if (document_) {
            if (auto* el = document_->GetElementById("progress-markers"))
                el->SetInnerRML(model_.progress_markers_rml);
        }
        model_dirty_ = true;
    }

    std::optional<RmlStatusBar::ProgressBarGeometry> RmlStatusBar::progressBarGeometry() const {
        if (!document_)
            return std::nullopt;

        auto* el = document_->GetElementById("progress-container");
        if (!el)
            return std::nullopt;

        const auto offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
        const float width = el->GetOffsetWidth();
        const float height = el->GetOffsetHeight();
        if (width <= 0.0f || height <= 0.0f)
            return std::nullopt;

        return ProgressBarGeometry{offset.x, offset.y, width, height};
    }

    void RmlStatusBar::resetSaveStepInteraction() {
        if (!save_step_interaction_.dragging &&
            !save_step_interaction_.adding &&
            save_step_interaction_.original_step == 0 &&
            save_step_interaction_.preview_step == 0 &&
            save_step_interaction_.hover_step == 0) {
            return;
        }

        save_step_interaction_ = {};
        markModelDirty();
    }

    std::optional<size_t> RmlStatusBar::hitSaveStep(const float local_x, const float local_y,
                                                    const ProgressBarGeometry& geom,
                                                    const std::vector<size_t>& save_steps,
                                                    const int total_iterations) const {
        if (total_iterations <= 0 || save_steps.empty())
            return std::nullopt;
        if (local_y < geom.y - 4.0f || local_y > geom.y + geom.h + 4.0f)
            return std::nullopt;

        constexpr float kHitRadius = 8.0f;
        float best_distance = kHitRadius + 1.0f;
        std::optional<size_t> best_step;
        for (const size_t step : save_steps) {
            if (step == 0 || step > static_cast<size_t>(total_iterations))
                continue;
            const float x = geom.x + (static_cast<float>(step) / static_cast<float>(total_iterations)) * geom.w;
            const float distance = std::abs(local_x - x);
            if (distance <= kHitRadius && distance < best_distance) {
                best_distance = distance;
                best_step = step;
            }
        }
        return best_step;
    }

    size_t RmlStatusBar::saveStepFromProgressX(const float local_x,
                                               const ProgressBarGeometry& geom,
                                               const int current_iteration,
                                               const int total_iterations) const {
        if (total_iterations <= 0)
            return 0;

        const float t = std::clamp((local_x - geom.x) / std::max(1.0f, geom.w), 0.0f, 1.0f);
        size_t step = static_cast<size_t>(std::lround(t * static_cast<float>(total_iterations)));
        const size_t min_step = current_iteration > 0
                                    ? std::min(static_cast<size_t>(total_iterations),
                                               static_cast<size_t>(current_iteration + 1))
                                    : size_t{1};
        step = std::clamp(step, min_step, static_cast<size_t>(total_iterations));
        return step;
    }

    void RmlStatusBar::commitSaveStepEdit() {
        auto* tm = lfs::vis::services().trainerOrNull();
        if (!tm || !tm->canEditSaveSteps() || save_step_interaction_.preview_step == 0)
            return;

        auto steps = tm->getSaveSteps();
        if (!save_step_interaction_.adding) {
            const auto original = save_step_interaction_.original_step;
            steps.erase(std::remove(steps.begin(), steps.end(), original), steps.end());
        }
        if (std::find(steps.begin(), steps.end(), save_step_interaction_.preview_step) == steps.end())
            steps.push_back(save_step_interaction_.preview_step);
        tm->setSaveSteps(std::move(steps));
        markModelDirty();
    }

    void RmlStatusBar::removeSaveStep(const size_t step) {
        auto* tm = lfs::vis::services().trainerOrNull();
        if (!tm || !tm->canEditSaveSteps() || step == 0)
            return;

        auto steps = tm->getSaveSteps();
        const auto old_size = steps.size();
        steps.erase(std::remove(steps.begin(), steps.end(), step), steps.end());
        if (steps.size() == old_size)
            return;
        tm->setSaveSteps(std::move(steps));
        if (save_step_interaction_.hover_step == step)
            save_step_interaction_.hover_step = 0;
        markModelDirty();
    }

    void RmlStatusBar::clearSaveStepHover() {
        if (save_step_interaction_.hover_step == 0)
            return;
        save_step_interaction_.hover_step = 0;
        markModelDirty();
    }

    void RmlStatusBar::handleSaveStepInteraction(const PanelInputState& input,
                                                 const float local_x,
                                                 const float local_y) {
        auto* tm = lfs::vis::services().trainerOrNull();
        if (!tm || !tm->canEditSaveSteps() || tm->getTotalIterations() <= 0) {
            resetSaveStepInteraction();
            return;
        }

        const auto geom = progressBarGeometry();
        if (!geom) {
            resetSaveStepInteraction();
            return;
        }

        const int total = tm->getTotalIterations();
        const int current = tm->getCurrentIteration();
        const auto save_steps = tm->getSaveSteps();
        const bool inside_progress =
            local_x >= geom->x && local_x <= geom->x + geom->w &&
            local_y >= geom->y && local_y <= geom->y + geom->h;

        if (save_step_interaction_.dragging) {
            const size_t preview = saveStepFromProgressX(local_x, *geom, current, total);
            if (preview != save_step_interaction_.preview_step) {
                save_step_interaction_.preview_step = preview;
                markModelDirty();
            }
            if (input.mouse_released[0] || !input.mouse_down[0]) {
                commitSaveStepEdit();
                save_step_interaction_.dragging = false;
                save_step_interaction_.adding = false;
                save_step_interaction_.original_step = 0;
                save_step_interaction_.preview_step = 0;
                markModelDirty();
            }
            return;
        }

        const auto hit_step = hitSaveStep(local_x, local_y, *geom, save_steps, total);
        const bool hit_future_step = hit_step && static_cast<int>(*hit_step) > current;
        if (input.mouse_clicked[1] && hit_step) {
            removeSaveStep(*hit_step);
            return;
        }
        if (input.mouse_clicked[0] && input.key_ctrl && hit_step) {
            removeSaveStep(*hit_step);
            return;
        }

        if (input.mouse_clicked[0]) {
            if (hit_future_step) {
                save_step_interaction_.dragging = true;
                save_step_interaction_.adding = false;
                save_step_interaction_.original_step = *hit_step;
                save_step_interaction_.preview_step = *hit_step;
                save_step_interaction_.hover_step = *hit_step;
                markModelDirty();
                return;
            }
            if (hit_step)
                return;

            if (inside_progress) {
                const size_t step = saveStepFromProgressX(local_x, *geom, current, total);
                save_step_interaction_.dragging = true;
                save_step_interaction_.adding = true;
                save_step_interaction_.original_step = 0;
                save_step_interaction_.preview_step = step;
                save_step_interaction_.hover_step = step;
                markModelDirty();
                return;
            }
        }

        const size_t next_hover = hit_step.value_or(0);
        if (next_hover != save_step_interaction_.hover_step) {
            save_step_interaction_.hover_step = next_hover;
            markModelDirty();
        }
    }

    bool RmlStatusBar::updateContent(const PanelDrawContext& ctx, const bool force_refresh) {
        if (!document_)
            return false;

        const auto now = std::chrono::steady_clock::now();
        if (!force_refresh && next_refresh_at_ != std::chrono::steady_clock::time_point{} &&
            now < next_refresh_at_) {
            return false;
        }

        model_dirty_ = false;

        const auto& p = lfs::vis::theme().palette;

        setModelString("safe_mode_text", model_.safe_mode_text, LOC("status_bar.safe_mode"));
        setModelString("mcp_preferences_label", model_.mcp_preferences_label,
                       LOC("status_bar.mcp_preferences"));
        if (mcp_status_provider_) {
            const auto status = mcp_status_provider_();
            std::string summary;
            std::string details;
            std::string tooltip;
            std::string color;
            if (status.phase == RuntimeServicePhase::Disabled || !status.enabled) {
                summary = std::format("{} · {}", LOC("status_bar.mcp_off"),
                                      LOC(status.network_exposed
                                              ? "status_bar.mcp_scope_network"
                                              : "status_bar.mcp_scope_local"));
                tooltip = LOC(status.network_exposed
                                  ? "status_bar.mcp_disabled_network_detail"
                                  : "status_bar.mcp_disabled_detail");
                color = colorToRml(p.text_dim);
            } else if (status.phase == RuntimeServicePhase::Starting) {
                summary = LOC("status_bar.mcp_starting");
                const auto endpoint = std::format("{}:{}",
                                                  status.network_exposed ? "0.0.0.0"
                                                                         : "127.0.0.1",
                                                  status.port);
                details = formatLocalizedValue(LOC("status_bar.mcp_starting_detail"),
                                               endpoint);
                tooltip = details;
                color = colorToRml(p.info);
            } else if (status.phase == RuntimeServicePhase::Stopping) {
                summary = LOC("status_bar.mcp_stopping");
                tooltip = LOC("status_bar.mcp_stopping_detail");
                color = colorToRml(p.text_dim);
            } else if (status.phase == RuntimeServicePhase::Failed || !status.running) {
                summary = LOC("status_bar.mcp_error");
                if (status.error_kind == RuntimeServiceErrorKind::BindFailed) {
                    const auto endpoint = std::format("{}:{}", status.error_address,
                                                      status.error_port);
                    details = formatLocalizedValue(LOC("status_bar.mcp_bind_failed"),
                                                   endpoint);
                } else {
                    details = LOC("status_bar.mcp_error_detail");
                }
                tooltip = details;
                color = colorToRml(p.error);
            } else {
                summary = status.network_exposed ? LOC("status_bar.mcp_network")
                                                 : LOC("status_bar.mcp_local");
                for (const auto& endpoint : status.endpoints) {
                    if (!details.empty())
                        details += '\n';
                    details += endpoint;
                }
                tooltip = status.network_exposed
                              ? LOC("status_bar.mcp_network_tooltip")
                              : LOC("status_bar.mcp_local_tooltip");
                color = colorToRml(status.network_exposed ? p.warning : p.success);
            }
            setModelString("mcp_summary", model_.mcp_summary, std::move(summary));
            setModelString("mcp_details", model_.mcp_details, std::move(details));
            setModelString("mcp_tooltip", model_.mcp_tooltip, std::move(tooltip));
            setModelString("mcp_color", model_.mcp_color, std::move(color));
            setModelString("mcp_toggle_label", model_.mcp_toggle_label,
                           LOC(status.enabled ? "status_bar.mcp_turn_off"
                                              : "status_bar.mcp_turn_on"));
            setModelBool("mcp_server_enabled", model_.mcp_server_enabled, status.enabled);
            setModelString("mcp_total_text", model_.mcp_total_text,
                           std::format("{} {}", status.request_count,
                                       LOC("status_bar.mcp_requests")));
            setModelString("mcp_success_text", model_.mcp_success_text,
                           std::format("{} {}", status.success_count,
                                       LOC("status_bar.mcp_successes")));
            setModelString("mcp_error_text", model_.mcp_error_text,
                           std::format("{} {}", status.error_count,
                                       LOC("status_bar.mcp_errors")));
        }

        // Get managers
        auto* viewer = ctx.ui ? ctx.ui->viewer : nullptr;
        auto* sm = viewer ? viewer->getSceneManager() : nullptr;
        auto* rm = viewer ? viewer->getRenderingManager() : nullptr;
        auto* tm = viewer ? viewer->getTrainerManager() : nullptr;

        // Mode text
        auto content_type = sm ? sm->getContentType() : SceneManager::ContentType::Empty;
        auto training_state = tm ? tm->getState() : TrainingState::Idle;

        std::string mode_rml;
        std::string mode_color;

        if (content_type == SceneManager::ContentType::Empty) {
            mode_rml = LOC("mode.empty");
            mode_color = colorToRml(p.text_dim);
        } else if (content_type == SceneManager::ContentType::SplatFiles) {
            mode_rml = LOC("mode.viewer");
            mode_color = colorToRml(p.info);
        } else {
            const char* strategy_raw = tm ? tm->getStrategyType() : "default";
            bool gut = tm && tm->isGutEnabled();
            std::string method = gut ? "GUT" : "3DGS";
            std::string strat_name;
            const std::string_view strategy = strategy_raw ? std::string_view(strategy_raw) : std::string_view{};
            if (strategy == "mcmc") {
                strat_name = LOC("training.options.strategy.mcmc");
            } else if (lfs::core::param::is_mrnf_strategy(strategy)) {
                strat_name = LOC("training.options.strategy.mrnf");
            } else if (strategy == "igs+") {
                strat_name = LOC("training.options.strategy.igs_plus");
            } else {
                strat_name = LOC("status_bar.strategy_default");
            }

            auto suffix = std::format(" ({}/{})", strat_name, method);

            switch (training_state) {
            case TrainingState::Running:
                mode_rml = LOC(lichtfeld::Strings::Status::TRAINING) + suffix;
                mode_color = colorToRml(p.warning);
                break;
            case TrainingState::Paused:
                mode_rml = LOC(lichtfeld::Strings::Status::PAUSED) + suffix;
                mode_color = colorToRml(p.text_dim);
                break;
            case TrainingState::Ready: {
                int cur_iter = tm ? tm->getCurrentIteration() : 0;
                const char* label_key = cur_iter > 0
                                            ? lichtfeld::Strings::TrainingPanel::RESUME
                                            : lichtfeld::Strings::Status::READY;
                mode_rml = LOC(label_key) + suffix;
                mode_color = colorToRml(p.success);
                break;
            }
            case TrainingState::Finished:
                mode_rml = LOC(lichtfeld::Strings::Status::COMPLETE) + suffix;
                mode_color = colorToRml(p.success);
                break;
            case TrainingState::Stopping:
                mode_rml = LOC(lichtfeld::Strings::Status::STOPPING) + suffix;
                mode_color = colorToRml(p.text_dim);
                break;
            default:
                mode_rml = LOC("mode.dataset");
                mode_color = colorToRml(p.text_dim);
                break;
            }
        }
        setModelString("mode_text", model_.mode_text, std::move(mode_rml));
        setModelString("mode_color", model_.mode_color, std::move(mode_color));

        // Training section
        bool show_training = content_type == SceneManager::ContentType::Dataset &&
                             (training_state == TrainingState::Running ||
                              training_state == TrainingState::Paused);
        setModelBool("show_training", model_.show_training, show_training);

        setModelString("step_label", model_.step_label, LOC(lichtfeld::Strings::Status::STEP));
        setModelString("loss_label", model_.loss_label, LOC(lichtfeld::Strings::Status::LOSS));
        setModelString("gaussians_label", model_.gaussians_label,
                       stripColon(LOC(lichtfeld::Strings::Status::GAUSSIANS)));
        setModelString("eta_label", model_.eta_label, LOC(lichtfeld::Strings::Status::ETA));

        if (show_training && tm) {
            int cur = tm->getCurrentIteration();
            int total = tm->getTotalIterations();
            float loss = tm->getCurrentLoss();
            int num_splats = tm->getNumSplats();
            int max_g = tm->getMaxGaussians();
            float elapsed = tm->getElapsedSeconds();
            float eta = tm->getEstimatedRemainingSeconds();
            float progress = total > 0 ? static_cast<float>(cur) / static_cast<float>(total) : 0.0f;
            auto progress_pct = std::format("{:.0f}%", progress * 100.0f);
            auto progress_text = progress_pct;
            if (save_step_interaction_.dragging && save_step_interaction_.preview_step > 0) {
                progress_text = formatStepLabel(save_step_interaction_.preview_step);
            } else if (save_step_interaction_.hover_step > 0) {
                progress_text = formatStepLabel(save_step_interaction_.hover_step);
            }

            setModelString("progress_width", model_.progress_width, progress_pct);
            setModelString("progress_text", model_.progress_text, std::move(progress_text));
            const ProgressMarkerRenderState marker_state{
                .dragging = save_step_interaction_.dragging,
                .adding = save_step_interaction_.adding,
                .original_step = save_step_interaction_.original_step,
                .preview_step = save_step_interaction_.preview_step,
                .hover_step = save_step_interaction_.hover_step,
            };
            setProgressMarkersRml(buildProgressMarkersRml(tm->getSaveSteps(), total, cur, marker_state));
            setModelString("step_value", model_.step_value, std::format("{}/{}", cur, total));
            setModelString("loss_value", model_.loss_value, std::format("{:.4f}", loss));
            setModelString("gaussians_value", model_.gaussians_value,
                           std::format("{}/{}", fmtCount(num_splats), fmtCount(max_g)));
            setModelString("time_value", model_.time_value, fmtTime(elapsed));
            setModelString("eta_value", model_.eta_value, fmtTime(eta));

            const auto eval_metrics = tm->getLastEvaluationMetrics();
            setModelBool("show_eval_metrics", model_.show_eval_metrics, eval_metrics.has_value());
            if (eval_metrics) {
                setModelString("eval_metrics_value", model_.eval_metrics_value,
                               std::format("{} {:.2f} / {} {:.4f}",
                                           LOC(lichtfeld::Strings::Status::PSNR),
                                           eval_metrics->psnr,
                                           LOC(lichtfeld::Strings::Status::SSIM),
                                           eval_metrics->ssim));
            } else {
                setModelString("eval_metrics_value", model_.eval_metrics_value, "");
            }
        } else {
            resetSaveStepInteraction();
            setModelString("progress_width", model_.progress_width, "0%");
            setModelString("progress_text", model_.progress_text, "");
            setProgressMarkersRml("");
            setModelString("step_value", model_.step_value, "");
            setModelString("loss_value", model_.loss_value, "");
            setModelBool("show_eval_metrics", model_.show_eval_metrics, false);
            setModelString("eval_metrics_value", model_.eval_metrics_value, "");
            setModelString("gaussians_value", model_.gaussians_value, "");
            setModelString("time_value", model_.time_value, "");
            setModelString("eta_value", model_.eta_value, "");
        }

        // Splat section (non-training)
        bool show_splats = !show_training && content_type != SceneManager::ContentType::Empty;
        size_t total_gaussians = 0;
        if (show_splats && sm) {
            total_gaussians = sm->getScene().getVisibleGaussianCount();
            if (total_gaussians == 0)
                show_splats = false;
        }
        setModelBool("show_splats", model_.show_splats, show_splats);

        if (show_splats) {
            auto splat_rml = std::format("{} {}",
                                         fmtCount(static_cast<int64_t>(total_gaussians)),
                                         stripColon(LOC(lichtfeld::Strings::Status::GAUSSIANS)));
            setModelString("splat_text", model_.splat_text, std::move(splat_rml));
            setModelString("splat_color", model_.splat_color, colorToRml(p.text));
        } else {
            setModelString("splat_text", model_.splat_text, "");
        }

        // Split view
        bool split_enabled = false;
        std::string split_mode_rml;
        std::string split_detail_rml;

        if (rm) {
            auto split_info = rm->getSplitViewInfo();
            split_enabled = split_info.enabled;
            if (split_enabled) {
                split_mode_rml = split_info.mode_label;
                split_detail_rml = split_info.detail_label;
            }
        }
        setModelBool("show_split", model_.show_split, split_enabled);

        if (split_enabled) {
            setModelString("split_mode", model_.split_mode, std::move(split_mode_rml));
            setModelString("split_mode_color", model_.split_mode_color, colorToRml(p.warning));
            setModelString("split_detail", model_.split_detail, std::move(split_detail_rml));
        } else {
            setModelString("split_mode", model_.split_mode, "");
            setModelString("split_detail", model_.split_detail, "");
        }

        // Speed overlays
        auto [wasd_speed, wasd_alpha] = speed_state_.getWasd();
        bool wasd_visible = wasd_alpha > 0.0f;

        if (wasd_visible) {
            auto wasd_rml = std::format("{}: {:.0f}",
                                        stripColon(LOC(lichtfeld::Strings::Controls::WASD)),
                                        wasd_speed);
            setModelBool("show_wasd", model_.show_wasd, true);
            setModelString("wasd_text", model_.wasd_text, std::move(wasd_rml));
            setModelString("wasd_color", model_.wasd_color, colorToRmlAlpha(p.info, wasd_alpha));
            setModelString("wasd_sep_color", model_.wasd_sep_color,
                           colorToRmlAlpha(p.text_dim, wasd_alpha));
        } else {
            setModelBool("show_wasd", model_.show_wasd, false);
        }

        auto [zoom_speed, zoom_alpha] = speed_state_.getZoom();
        bool zoom_visible = zoom_alpha > 0.0f;

        if (zoom_visible) {
            auto zoom_rml = std::format("{}: {:.0f}",
                                        stripColon(LOC(lichtfeld::Strings::Controls::ZOOM)),
                                        zoom_speed);
            setModelBool("show_zoom", model_.show_zoom, true);
            setModelString("zoom_text", model_.zoom_text, std::move(zoom_rml));
            setModelString("zoom_color", model_.zoom_color, colorToRmlAlpha(p.info, zoom_alpha));
            setModelString("zoom_sep_color", model_.zoom_sep_color,
                           colorToRmlAlpha(p.text_dim, zoom_alpha));
        } else {
            setModelBool("show_zoom", model_.show_zoom, false);
        }

        // Transient StatusOnly message (ErrorBus)
        const auto status_msg = status_message_.snapshot(now);
        setModelBool("show_status_message", model_.show_status_message, status_msg.visible);
        if (status_msg.visible) {
            setModelString("status_message_text", model_.status_message_text, status_msg.text);
            const ThemeColor& status_col = status_msg.level == ErrorNoticeLevel::Error     ? p.error
                                           : status_msg.level == ErrorNoticeLevel::Warning ? p.warning
                                                                                           : p.info;
            setModelString("status_message_color", model_.status_message_color,
                           colorToRmlAlpha(status_col, status_msg.alpha));
        }

        const auto account = lfs::vis::app_store().account_state.get();
        std::string account_label = account.label;
        if (account.linking) {
            account_label = LOC("account.status.linking");
        } else if (!account.signed_in) {
            account_label = LOC("account.status.sign_in");
        } else if (account_label.empty()) {
            account_label = "LF";
        }
        setModelString("account_label", model_.account_label, std::move(account_label));
        setModelString("account_tier", model_.account_tier, account.tier);
        std::string account_tooltip;
        if (account.membership_required) {
            account_tooltip = LOC("account.status.membership_required");
        } else if (account.linking) {
            account_tooltip = LOC("account.status.linking");
        } else if (!account.signed_in) {
            account_tooltip = LOC("account.status.tooltip");
        }
        if (!account.tooltip.empty()) {
            if (!account_tooltip.empty())
                account_tooltip += " — ";
            account_tooltip += account.tooltip;
        }
        if (account_tooltip.empty())
            account_tooltip = LOC("account.status.tooltip");
        setModelString("account_tooltip", model_.account_tooltip, std::move(account_tooltip));
        setModelBool("account_show_tier", model_.account_show_tier,
                     account.signed_in && !account.tier.empty());
        setModelBool("account_membership_required", model_.account_membership_required,
                     account.membership_required);
        const ThemeColor& account_color = account.membership_required ? p.warning
                                          : account.linking           ? p.info
                                          : account.signed_in         ? p.text
                                                                      : p.text_dim;
        setModelString("account_color", model_.account_color, colorToRml(account_color));

        // Right section: GPU memory
        pollGpuMemoryQuery(now);
        const auto mem = cached_gpu_mem_;
        constexpr float gib = 1024.0f * 1024.0f * 1024.0f;
        float app_gib = mem.process_used / gib;
        float used_gib = mem.total_used / gib;
        float total_gib = mem.total / gib;
        float pct = total_gib > 0.0f ? (used_gib / total_gib) * 100.0f : 0.0f;

        ThemeColor mem_color = pct < 50.0f ? p.success : (pct < 75.0f ? p.warning : p.error);
        setModelBool("gpu_panel_active", model_.gpu_panel_active,
                     lfs::vis::app_store().perf_hud.get().visible);
        setModelString("lfs_mem_text", model_.lfs_mem_text, std::format("LFS {:.2f} GiB", app_gib));
        setModelString("lfs_mem_color", model_.lfs_mem_color, colorToRml(p.info));
        setModelBool("show_gpu_model", model_.show_gpu_model, !mem.device_name.empty());
        setModelString("gpu_model_text", model_.gpu_model_text, mem.device_name);
        setModelString("gpu_mem_text", model_.gpu_mem_text,
                       std::format("{} {:.2f}/{:.2f} GiB", LOC("status_bar.gpu"), used_gib, total_gib));
        setModelString("gpu_mem_color", model_.gpu_mem_color, colorToRml(mem_color));

        // FPS: prefer scene-render rate when scene frames are in the measurement
        // window; when only GUI frames are presented, show that rate as ui-fps
        // so a GUI-only spin is not invisible. True idle (no samples) stays 0.
        const float scene_fps = reactive_fps_available_ ? reactive_fps_value_
                                                        : (rm ? rm->getAverageFPS() : 0.0f);
        const float presented_fps = rm ? rm->getPresentedAverageFPS() : 0.0f;
        const bool ui_only_fps = scene_fps <= 0.0f && presented_fps > 0.0f;
        const float fps = ui_only_fps ? presented_fps : scene_fps;
        ThemeColor fps_col = ui_only_fps
                                 ? p.text_dim
                                 : (fps >= 30.0f ? p.success : (fps >= 15.0f ? p.warning : p.error));
        setModelString("fps_value", model_.fps_value, std::format("{:.0f}", fps));
        setModelString("fps_color", model_.fps_color, colorToRml(fps_col));
        setModelString("fps_label", model_.fps_label,
                       ui_only_fps ? std::format(" {}", LOC("status_bar.ui_fps"))
                                   : std::format(" {}", LOC(lichtfeld::Strings::Status::FPS)));
        setModelString("git_commit", model_.git_commit, GIT_COMMIT_HASH_SHORT);

        section_signature_ =
            (model_.show_training ? uint32_t{1} << 0 : 0) |
            (model_.show_eval_metrics ? uint32_t{1} << 1 : 0) |
            (model_.show_splats ? uint32_t{1} << 2 : 0) |
            (model_.show_split ? uint32_t{1} << 3 : 0) |
            (model_.show_wasd ? uint32_t{1} << 4 : 0) |
            (model_.show_zoom ? uint32_t{1} << 5 : 0) |
            (model_.show_status_message ? uint32_t{1} << 6 : 0) |
            (model_.show_gpu_model ? uint32_t{1} << 7 : 0) |
            (model_.account_show_tier ? uint32_t{1} << 8 : 0);

        animation_active_ = wasd_visible || zoom_visible || status_msg.visible;
        next_refresh_at_ = now + (animation_active_ ? kAnimatedRefreshInterval
                                                    : (ctx.is_training ? kBusyRefreshInterval
                                                                       : kIdleRefreshInterval));
        return model_dirty_;
    }

    void RmlStatusBar::processInput(const PanelInputState& input, const float bar_x, const float bar_y,
                                    const float bar_w, const float bar_h) {
        if (!rml_context_ || !document_)
            return;

        const float overlay_height = overlayHeight();
        trackContextFrame(bar_x - input.screen_x,
                          bar_y - overlay_height - input.screen_y);
        const float local_x = input.mouse_x - bar_x;
        const float local_y = input.mouse_y - (bar_y - overlay_height);
        const bool is_inside = local_x >= 0.0f && local_x < bar_w &&
                               local_y >= 0.0f && local_y < bar_h + overlay_height;
        if (!is_inside && !input.mouse_released[0] && !input.mouse_released[1] &&
            !save_step_interaction_.dragging) {
            clearSaveStepHover();
            return;
        }

        handleSaveStepInteraction(input, local_x, local_y);

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);
        rml_context_->ProcessMouseMove(static_cast<int>(local_x), static_cast<int>(local_y), mods);

        if (is_inside && input.mouse_clicked[0])
            rml_context_->ProcessMouseButtonDown(0, mods);
        if (input.mouse_released[0])
            rml_context_->ProcessMouseButtonUp(0, mods);
        if (is_inside && input.mouse_clicked[1])
            rml_context_->ProcessMouseButtonDown(1, mods);
        if (input.mouse_released[1])
            rml_context_->ProcessMouseButtonUp(1, mods);
    }

    float RmlStatusBar::overlayHeight() const {
        if (!model_.mcp_details_expanded)
            return 0.0f;
        const float dp_ratio = rml_context_
                                   ? rml_context_->GetDensityIndependentPixelRatio()
                                   : 1.0f;
        const float fallback_height = 150.0f * dp_ratio;
        if (!document_)
            return fallback_height;

        auto* const popup = document_->GetElementById("mcp-popup");
        if (!popup || !popup->IsVisible())
            return fallback_height;

        // The popup sits 20dp above the status row. Keep the first-frame fallback,
        // but grow the render/input surface when localized text, an error, or
        // multiple network endpoints make the actual popup taller.
        const float measured_height = popup->GetOffsetHeight() + 20.0f * dp_ratio;
        return std::max(fallback_height, measured_height);
    }

    bool RmlStatusBar::isOverlayPoint(const float local_x, const float local_y,
                                      const float bar_w) const {
        (void)bar_w;
        if (overlayHeight() <= 0.0f || !document_)
            return false;
        auto* const popup = document_->GetElementById("mcp-popup");
        if (!popup || !popup->IsVisible())
            return false;
        const auto offset = popup->GetAbsoluteOffset(Rml::BoxArea::Border);
        const float width = popup->GetOffsetWidth();
        const float height = popup->GetOffsetHeight();
        const float context_y = local_y + overlayHeight();
        return local_x >= offset.x && local_x < offset.x + width &&
               context_y >= offset.y && context_y < offset.y + height;
    }

    void RmlStatusBar::trackContextFrame(const float window_x, const float window_y) {
        if (!rml_manager_ || !rml_context_)
            return;
        std::optional<RmlRect> popup_rect;
        if (model_.mcp_details_expanded && document_) {
            if (auto* const popup = document_->GetElementById("mcp-popup");
                popup && popup->IsVisible()) {
                const auto offset = popup->GetAbsoluteOffset(Rml::BoxArea::Border);
                popup_rect = RmlRect{
                    .x1 = offset.x,
                    .y1 = offset.y,
                    .x2 = offset.x + popup->GetOffsetWidth(),
                    .y2 = offset.y + popup->GetOffsetHeight(),
                };
            }
        }
        rml_manager_->trackContextFrame(rml_context_,
                                        static_cast<int>(std::lround(window_x)),
                                        static_cast<int>(std::lround(window_y)),
                                        popup_rect);
    }

    void RmlStatusBar::queueCachedVulkanContext(const float x, const float y,
                                                const float w_px, const float h_px,
                                                const int screen_w, const int screen_h,
                                                const int render_w, const int render_h,
                                                const bool refresh_cache) {
        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;

        const auto blit_rect = toFramebufferBlitRect(rml_manager_->getWindow(),
                                                     x, y, w_px, h_px, screen_w, screen_h);
        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = render_w,
            .cache_height = render_h,
            .offset_x = blit_rect.x,
            .offset_y = blit_rect.y,
            .draw_width = blit_rect.w,
            .draw_height = blit_rect.h,
            .refresh = refresh_cache,
            .foreground = false,
            .clip_enabled = true,
            .clip = {
                .x1 = blit_rect.x,
                .y1 = blit_rect.y,
                .x2 = blit_rect.x + blit_rect.w,
                .y2 = blit_rect.y + blit_rect.h,
            },
        });
    }

    void RmlStatusBar::renderCached(const PanelDrawContext& ctx, const float x, const float y,
                                    const float w_px, const float h_px,
                                    const int screen_w, const int screen_h) {
        if (!rml_context_ || !document_)
            return;
        if (w_px <= 0.0f || h_px <= 0.0f || screen_w <= 0 || screen_h <= 0)
            return;

        const float overlay_height = overlayHeight();
        const int render_w = static_cast<int>(w_px);
        const int render_h = static_cast<int>(h_px + overlay_height);
        const float dp_ratio = rml_context_->GetDensityIndependentPixelRatio();
        const bool dp_changed = dp_ratio != last_dp_ratio_;
        const bool theme_current =
            has_theme_signature_ && rml_theme::currentThemeSignature() == last_theme_signature_;
        const auto now = std::chrono::steady_clock::now();
        const auto runtime_revision = lfs::vis::runtimeServiceRevision();
        if (runtime_revision != last_runtime_service_revision_) {
            last_runtime_service_revision_ = runtime_revision;
            model_dirty_ = true;
        }
        const bool refresh_due =
            next_refresh_at_ == std::chrono::steady_clock::time_point{} ||
            now >= next_refresh_at_;
        const bool can_reuse = theme_current && !dp_changed && !model_dirty_ && !animation_active_ &&
                               !refresh_due && render_w == last_render_w_ &&
                               render_h == last_render_h_;
        if (!can_reuse) {
            render(ctx, x, y, w_px, h_px, screen_w, screen_h);
            return;
        }

        trackRenderedContextFrame(x, y);

        queueCachedVulkanContext(x, y - overlay_height, w_px, h_px + overlay_height,
                                 screen_w, screen_h,
                                 render_w, render_h, direct_cache_.texture == 0);
    }

    void RmlStatusBar::render(const PanelDrawContext& ctx, const float x, const float y,
                              const float w_px, const float h_px,
                              const int screen_w, const int screen_h) {
        if (!rml_context_ || !document_)
            return;

        if (w_px <= 0.0f || h_px <= 0.0f || screen_w <= 0 || screen_h <= 0)
            return;

        const float overlay_height = overlayHeight();
        const int render_w = static_cast<int>(w_px);
        const int render_h = static_cast<int>(h_px + overlay_height);
        const bool size_changed = (render_w != last_render_w_ || render_h != last_render_h_);
        const float dp_ratio = rml_context_->GetDensityIndependentPixelRatio();
        const bool dp_changed = dp_ratio != last_dp_ratio_;
        const bool had_pending_model_dirty = model_dirty_;
        const bool theme_changed = updateTheme();
        const auto now = std::chrono::steady_clock::now();
        const bool refresh_due =
            size_changed || dp_changed || theme_changed || had_pending_model_dirty || animation_active_ ||
            next_refresh_at_ == std::chrono::steady_clock::time_point{} ||
            now >= next_refresh_at_;
        const bool content_changed = updateContent(ctx, refresh_due);
        const bool section_signature_changed = section_signature_ != last_section_signature_;
        const bool needs_render = size_changed || dp_changed || theme_changed || had_pending_model_dirty ||
                                  content_changed ||
                                  (animation_active_ && refresh_due);
        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;

        if (needs_render) {
            rml_context_->SetDimensions(Rml::Vector2i(render_w, render_h));
            if (render_h != last_document_h_) {
                document_->SetProperty("height", std::format("{}px", render_h));
                last_document_h_ = render_h;
            }
            rml_context_->Update();
            fitToAvailableWidth(size_changed || dp_changed || theme_changed || section_signature_changed);

            animation_active_ = animation_active_ || (rml_context_->GetNextUpdateDelay() == 0);
            last_dp_ratio_ = dp_ratio;
            last_section_signature_ = section_signature_;
            last_render_w_ = render_w;
            last_render_h_ = render_h;
        }

        trackRenderedContextFrame(x, y);

        queueCachedVulkanContext(x, y - overlay_height, w_px, h_px + overlay_height,
                                 screen_w, screen_h,
                                 render_w, render_h, true);
    }

} // namespace lfs::vis::gui
