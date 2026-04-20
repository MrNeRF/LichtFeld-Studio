/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_status_bar.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/gpu_memory_query.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_context.hpp"
#include "internal/resource_paths.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/training_manager.hpp"
#include "visualizer_impl.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <SDL3/SDL_video.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>

#include "git_version.h"

namespace lfs::vis::gui {

    using rml_theme::colorToRml;
    using rml_theme::colorToRmlAlpha;

    namespace {
        std::string fmtCount(int64_t n) {
            if (n >= 1'000'000)
                return std::format("{:.2f}M", n / 1e6);
            if (n >= 1'000)
                return std::format("{:.0f}K", n / 1e3);
            return std::to_string(n);
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

    void RmlStatusBar::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        const auto& palette = lfs::vis::theme().palette;
        model_.mode_color = colorToRml(palette.text_dim);
        model_.splat_color = colorToRml(palette.text);
        model_.split_mode_color = colorToRml(palette.warning);
        model_.wasd_color = colorToRml(palette.info);
        model_.wasd_sep_color = colorToRml(palette.text_dim);
        model_.zoom_color = colorToRml(palette.info);
        model_.zoom_sep_color = colorToRml(palette.text_dim);
        model_.lfs_mem_color = colorToRml(palette.info);
        model_.gpu_mem_color = colorToRml(palette.text);
        model_.fps_color = colorToRml(palette.success);

        rml_context_ = rml_manager_->createContext("status_bar", 800, 22);
        if (!rml_context_) {
            LOG_ERROR("RmlStatusBar: failed to create RML context");
            return;
        }

        auto ctor = rml_context_->CreateDataModel("status_bar");
        assert(ctor);
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
        ctor.Bind("lfs_mem_text", &model_.lfs_mem_text);
        ctor.Bind("lfs_mem_color", &model_.lfs_mem_color);
        ctor.Bind("show_gpu_model", &model_.show_gpu_model);
        ctor.Bind("gpu_model_text", &model_.gpu_model_text);
        ctor.Bind("gpu_mem_text", &model_.gpu_mem_text);
        ctor.Bind("gpu_mem_color", &model_.gpu_mem_color);
        ctor.Bind("fps_value", &model_.fps_value);
        ctor.Bind("fps_color", &model_.fps_color);
        ctor.Bind("fps_label", &model_.fps_label);
        ctor.Bind("git_commit", &model_.git_commit);
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

        if (!speed_events_initialized_) {
            lfs::core::events::ui::SpeedChanged::when([this](const auto& e) {
                speed_state_.showWasd(e.current_speed);
                animation_active_ = true;
                next_refresh_at_ = {};
            });
            lfs::core::events::ui::ZoomSpeedChanged::when([this](const auto& e) {
                speed_state_.showZoom(e.zoom_speed);
                animation_active_ = true;
                next_refresh_at_ = {};
            });
            speed_events_initialized_ = true;
        }

        updateTheme();
    }

    void RmlStatusBar::shutdown() {
        model_handle_ = {};
        fbo_.destroy();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("status_bar");
        rml_context_ = nullptr;
        document_ = nullptr;
    }

    std::string RmlStatusBar::generateThemeRCSS(const lfs::vis::Theme& t) const {
        const auto& p = t.palette;

        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto surface_bright = colorToRml(p.surface_bright);
        const auto primary = colorToRml(p.primary);
        const auto success = colorToRml(p.success);
        const auto warning = colorToRml(p.warning);
        const auto error = colorToRml(p.error);
        const auto info = colorToRml(p.info);

        auto surface_bright_half = colorToRmlAlpha(p.surface_bright, 0.5f);

        return std::format(
            "body {{ color: {0}; }}\n"
            ".dim {{ color: {1}; }}\n"
            ".separator {{ color: {1}; }}\n"
            "#progress-container {{ background-color: {2}; }}\n"
            "#progress-fill {{ background-color: {3}; }}\n"
            "#progress-text {{ color: {0}; }}\n"
            "#gpu-icon {{ image-color: {1}; }}\n",
            text, text_dim, surface_bright_half, primary);
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

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::generateAllThemeMedia([this](const auto& th) { return generateThemeRCSS(th); }));
        model_dirty_ = true;
        return true;
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

            setModelString("progress_width", model_.progress_width, progress_pct);
            setModelString("progress_text", model_.progress_text, progress_pct);
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
            setModelString("progress_width", model_.progress_width, "0%");
            setModelString("progress_text", model_.progress_text, "");
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
                                        zoom_speed * 10.0f);
            setModelBool("show_zoom", model_.show_zoom, true);
            setModelString("zoom_text", model_.zoom_text, std::move(zoom_rml));
            setModelString("zoom_color", model_.zoom_color, colorToRmlAlpha(p.info, zoom_alpha));
            setModelString("zoom_sep_color", model_.zoom_sep_color,
                           colorToRmlAlpha(p.text_dim, zoom_alpha));
        } else {
            setModelBool("show_zoom", model_.show_zoom, false);
        }

        // Right section: GPU memory
        if (next_gpu_refresh_at_ == std::chrono::steady_clock::time_point{} ||
            now >= next_gpu_refresh_at_) {
            cached_gpu_mem_ = queryGpuMemory();
            next_gpu_refresh_at_ = now + kGpuRefreshInterval;
        }
        const auto mem = cached_gpu_mem_;
        float app_gb = mem.process_used / 1e9f;
        float used_gb = mem.total_used / 1e9f;
        float total_gb = mem.total / 1e9f;
        float pct = total_gb > 0.0f ? (used_gb / total_gb) * 100.0f : 0.0f;

        ImVec4 mem_color = pct < 50.0f ? p.success : (pct < 75.0f ? p.warning : p.error);
        setModelString("lfs_mem_text", model_.lfs_mem_text, std::format("LFS {:.1f}GB", app_gb));
        setModelString("lfs_mem_color", model_.lfs_mem_color, colorToRml(p.info));
        setModelBool("show_gpu_model", model_.show_gpu_model, !mem.device_name.empty());
        setModelString("gpu_model_text", model_.gpu_model_text, mem.device_name);
        setModelString("gpu_mem_text", model_.gpu_mem_text,
                       std::format("{} {:.1f}/{:.1f}GB", LOC("status_bar.gpu"), used_gb, total_gb));
        setModelString("gpu_mem_color", model_.gpu_mem_color, colorToRml(mem_color));

        // FPS
        float fps = rm ? rm->getAverageFPS() : 0.0f;
        ImVec4 fps_col = fps >= 30.0f ? p.success : (fps >= 15.0f ? p.warning : p.error);
        setModelString("fps_value", model_.fps_value, std::format("{:.0f}", fps));
        setModelString("fps_color", model_.fps_color, colorToRml(fps_col));
        setModelString("fps_label", model_.fps_label,
                       std::format(" {}", LOC(lichtfeld::Strings::Status::FPS)));
        setModelString("git_commit", model_.git_commit, GIT_COMMIT_HASH_SHORT);

        animation_active_ = wasd_visible || zoom_visible;
        next_refresh_at_ = now + (animation_active_ ? kAnimatedRefreshInterval
                                                    : (ctx.is_training ? kBusyRefreshInterval
                                                                       : kIdleRefreshInterval));
        return model_dirty_;
    }

    void RmlStatusBar::render(const PanelDrawContext& ctx, const float x, const float y,
                              const float w_px, const float h_px,
                              const int screen_w, const int screen_h) {
        if (!rml_context_ || !document_)
            return;

        if (w_px <= 0.0f || h_px <= 0.0f || screen_w <= 0 || screen_h <= 0)
            return;

        const auto blit_rect = toFramebufferBlitRect(rml_manager_ ? rml_manager_->getWindow() : nullptr,
                                                     x, y, w_px, h_px, screen_w, screen_h);
        const int render_w = static_cast<int>(w_px);
        const int render_h = static_cast<int>(h_px);
        const bool size_changed = (render_w != last_render_w_ || render_h != last_render_h_);
        const bool had_pending_model_dirty = model_dirty_;
        const bool theme_changed = updateTheme();
        const auto now = std::chrono::steady_clock::now();
        const bool refresh_due =
            size_changed || theme_changed || had_pending_model_dirty || animation_active_ ||
            next_refresh_at_ == std::chrono::steady_clock::time_point{} ||
            now >= next_refresh_at_;
        const bool content_changed = updateContent(ctx, refresh_due);
        const bool needs_render = size_changed || theme_changed || had_pending_model_dirty ||
                                  content_changed ||
                                  (animation_active_ && refresh_due);
        if (rml_manager_->shouldDeferFboUpdate(fbo_)) {
            if (needs_render)
                model_dirty_ = true;
            if (fbo_.valid())
                fbo_.blitToScreen(blit_rect.x, blit_rect.y, blit_rect.w, blit_rect.h,
                                  blit_rect.screen_w, blit_rect.screen_h);
            return;
        }

        if (needs_render) {
            rml_context_->SetDimensions(Rml::Vector2i(render_w, render_h));
            if (render_h != last_document_h_) {
                document_->SetProperty("height", std::format("{}px", render_h));
                last_document_h_ = render_h;
            }
            rml_context_->Update();

            fbo_.ensure(render_w, render_h);
            if (!fbo_.valid())
                return;

            auto* render = rml_manager_->getRenderInterface();
            assert(render);
            render->SetViewport(render_w, render_h);

            GLint prev_fbo = 0;
            fbo_.bind(&prev_fbo);
            render->SetTargetFramebuffer(fbo_.fbo());

            render->BeginFrame();
            rml_context_->Render();
            render->EndFrame();

            render->SetTargetFramebuffer(0);
            fbo_.unbind(prev_fbo);

            animation_active_ = animation_active_ || (rml_context_->GetNextUpdateDelay() == 0);
            last_render_w_ = render_w;
            last_render_h_ = render_h;
        }

        if (fbo_.valid())
            fbo_.blitToScreen(blit_rect.x, blit_rect.y, blit_rect.w, blit_rect.h,
                              blit_rect.screen_w, blit_rect.screen_h);
    }

} // namespace lfs::vis::gui
