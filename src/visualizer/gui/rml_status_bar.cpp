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
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <imgui.h>

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

        rml_context_ = rml_manager_->createContext("status_bar", 800, 22);
        if (!rml_context_) {
            LOG_ERROR("RmlStatusBar: failed to create RML context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/statusbar.rml");
            document_ = rml_context_->LoadDocument(rml_path.string());
            if (!document_) {
                LOG_ERROR("RmlStatusBar: failed to load statusbar.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlStatusBar: resource not found: {}", e.what());
            return;
        }

        cacheElements();

        try {
            const auto icon_path = lfs::vis::getAssetPath("icon/gpu.png");
            if (gpu_icon_)
                gpu_icon_->SetAttribute("src", icon_path.string());
        } catch (...) {
        }

        if (!speed_events_initialized_) {
            lfs::core::events::ui::SpeedChanged::when([this](const auto& e) {
                speed_state_.showWasd(e.current_speed);
            });
            lfs::core::events::ui::ZoomSpeedChanged::when([this](const auto& e) {
                speed_state_.showZoom(e.zoom_speed);
            });
            speed_events_initialized_ = true;
        }

        updateTheme();
    }

    void RmlStatusBar::shutdown() {
        fbo_.destroy();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("status_bar");
        rml_context_ = nullptr;
        document_ = nullptr;
    }

    void RmlStatusBar::cacheElements() {
        assert(document_);
        auto get = [this](const char* id) -> Rml::Element* {
            return document_->GetElementById(id);
        };

        mode_text_ = get("mode-text");
        training_section_ = get("training-section");
        progress_fill_ = get("progress-fill");
        progress_text_ = get("progress-text");
        step_label_ = get("step-label");
        step_value_ = get("step-value");
        loss_label_ = get("loss-label");
        loss_value_ = get("loss-value");
        gaussians_label_ = get("gaussians-label");
        gaussians_value_ = get("gaussians-value");
        time_value_ = get("time-value");
        eta_label_ = get("eta-label");
        eta_value_ = get("eta-value");
        splat_section_ = get("splat-section");
        splat_text_ = get("splat-text");
        split_section_ = get("split-section");
        split_mode_ = get("split-mode");
        split_detail_ = get("split-detail");
        wasd_section_ = get("wasd-section");
        wasd_text_ = get("wasd-text");
        zoom_section_ = get("zoom-section");
        zoom_text_ = get("zoom-text");
        gpu_icon_ = get("gpu-icon");
        lfs_mem_ = get("lfs-mem");
        gpu_mem_ = get("gpu-mem");
        fps_value_ = get("fps-value");
        fps_label_ = get("fps-label");
        git_commit_ = get("git-commit");
    }

    std::string RmlStatusBar::generateThemeRCSS() const {
        const auto& p = lfs::vis::theme().palette;

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

    void RmlStatusBar::updateTheme() {
        if (!document_)
            return;

        const auto& t = lfs::vis::theme();
        if (t.name == last_theme_)
            return;
        last_theme_ = t.name;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/statusbar.rcss");

        rml_theme::applyTheme(document_, base_rcss_, generateThemeRCSS());

        cache_ = CachedState{};
    }

    void RmlStatusBar::updateContent(const PanelDrawContext& ctx) {
        if (!document_)
            return;

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
            std::string strat_name = (std::string_view(strategy_raw) == "mcmc")
                                         ? LOC("training.options.strategy.mcmc")
                                         : LOC("status_bar.strategy_default");

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

        if (mode_rml != cache_.mode_rml || mode_color != cache_.mode_color) {
            cache_.mode_rml = mode_rml;
            cache_.mode_color = mode_color;
            if (mode_text_) {
                mode_text_->SetInnerRML(mode_rml);
                mode_text_->SetProperty("color", mode_color);
            }
        }

        // Training section
        bool show_training = content_type == SceneManager::ContentType::Dataset &&
                             (training_state == TrainingState::Running ||
                              training_state == TrainingState::Paused);

        if (show_training != cache_.show_training) {
            cache_.show_training = show_training;
            if (training_section_)
                training_section_->SetClass("hidden", !show_training);
        }

        if (show_training && tm) {
            int cur = tm->getCurrentIteration();
            int total = tm->getTotalIterations();
            float loss = tm->getCurrentLoss();
            int num_splats = tm->getNumSplats();
            int max_g = tm->getMaxGaussians();
            float elapsed = tm->getElapsedSeconds();
            float eta = tm->getEstimatedRemainingSeconds();

            if (cur != cache_.current_iter || total != cache_.total_iter) {
                cache_.current_iter = cur;
                cache_.total_iter = total;

                float progress = total > 0 ? static_cast<float>(cur) / static_cast<float>(total) : 0.0f;
                if (progress_fill_)
                    progress_fill_->SetProperty("width", std::format("{:.0f}%", progress * 100.0f));
                if (progress_text_)
                    progress_text_->SetInnerRML(std::format("{:.0f}%", progress * 100.0f));
                if (step_value_)
                    step_value_->SetInnerRML(std::format("{}/{}", cur, total));
            }

            if (loss != cache_.loss) {
                cache_.loss = loss;
                if (loss_value_)
                    loss_value_->SetInnerRML(std::format("{:.4f}", loss));
            }

            if (num_splats != cache_.num_splats || max_g != cache_.max_gaussians) {
                cache_.num_splats = num_splats;
                cache_.max_gaussians = max_g;
                if (gaussians_value_)
                    gaussians_value_->SetInnerRML(
                        std::format("{}/{}", fmtCount(num_splats), fmtCount(max_g)));
            }

            if (elapsed != cache_.elapsed) {
                cache_.elapsed = elapsed;
                if (time_value_)
                    time_value_->SetInnerRML(fmtTime(elapsed));
            }

            if (eta != cache_.eta) {
                cache_.eta = eta;
                if (eta_value_)
                    eta_value_->SetInnerRML(fmtTime(eta));
            }

            // Update labels (only on first pass or theme change)
            if (!cache_.git_set) {
                if (step_label_)
                    step_label_->SetInnerRML(LOC(lichtfeld::Strings::Status::STEP));
                if (loss_label_)
                    loss_label_->SetInnerRML(LOC(lichtfeld::Strings::Status::LOSS));
                if (gaussians_label_)
                    gaussians_label_->SetInnerRML(
                        stripColon(LOC(lichtfeld::Strings::Status::GAUSSIANS)));
                if (eta_label_)
                    eta_label_->SetInnerRML(LOC(lichtfeld::Strings::Status::ETA));
            }
        }

        // Splat section (non-training)
        bool show_splats = !show_training && content_type != SceneManager::ContentType::Empty;
        size_t total_gaussians = 0;
        if (show_splats && sm) {
            const auto* model = sm->getScene().getCombinedModel();
            total_gaussians = model ? model->size() : 0;
            if (total_gaussians == 0)
                show_splats = false;
        }

        if (show_splats != cache_.show_splats) {
            cache_.show_splats = show_splats;
            if (splat_section_)
                splat_section_->SetClass("hidden", !show_splats);
        }

        if (show_splats) {
            auto splat_rml = std::format("{} {}",
                                         fmtCount(static_cast<int64_t>(total_gaussians)),
                                         stripColon(LOC(lichtfeld::Strings::Status::GAUSSIANS)));
            if (splat_rml != cache_.splat_rml) {
                cache_.splat_rml = splat_rml;
                if (splat_text_) {
                    splat_text_->SetInnerRML(splat_rml);
                    splat_text_->SetProperty("color", colorToRml(p.text));
                }
            }
        }

        // Split view
        bool split_enabled = false;
        std::string split_mode_rml;
        std::string split_detail_rml;

        if (rm) {
            auto split_info = rm->getSplitViewInfo();
            split_enabled = split_info.enabled;
            if (split_enabled) {
                auto settings = rm->getSettings();
                if (settings.split_view_mode == SplitViewMode::GTComparison) {
                    int cam_id = rm->getCurrentCameraId();
                    split_mode_rml = LOC("status_bar.gt_compare");
                    std::string cam_template = LOC("status_bar.camera");
                    auto pos = cam_template.find("{cam_id}");
                    if (pos != std::string::npos)
                        cam_template.replace(pos, 8, std::to_string(cam_id));
                    split_detail_rml = cam_template;
                } else if (settings.split_view_mode == SplitViewMode::PLYComparison) {
                    split_mode_rml = LOC("status_bar.split");
                    split_detail_rml = std::format("{} | {}", split_info.left_name, split_info.right_name);
                }
            }
        }

        if (split_enabled != cache_.split_enabled) {
            cache_.split_enabled = split_enabled;
            if (split_section_)
                split_section_->SetClass("hidden", !split_enabled);
        }

        if (split_enabled) {
            if (split_mode_rml != cache_.split_mode_rml) {
                cache_.split_mode_rml = split_mode_rml;
                if (split_mode_) {
                    split_mode_->SetInnerRML(split_mode_rml);
                    split_mode_->SetProperty("color", colorToRml(p.warning));
                }
            }
            if (split_detail_rml != cache_.split_detail_rml) {
                cache_.split_detail_rml = split_detail_rml;
                if (split_detail_)
                    split_detail_->SetInnerRML(split_detail_rml);
            }
        }

        // Speed overlays
        auto [wasd_speed, wasd_alpha] = speed_state_.getWasd();
        bool wasd_visible = wasd_alpha > 0.0f;

        if (wasd_visible) {
            auto wasd_rml = std::format("{}: {:.0f}",
                                        stripColon(LOC(lichtfeld::Strings::Controls::WASD)),
                                        wasd_speed);
            if (wasd_rml != cache_.wasd_rml || wasd_alpha != cache_.wasd_alpha) {
                cache_.wasd_rml = wasd_rml;
                cache_.wasd_alpha = wasd_alpha;
                if (wasd_section_)
                    wasd_section_->SetClass("hidden", false);
                if (wasd_text_) {
                    wasd_text_->SetInnerRML(wasd_rml);
                    wasd_text_->SetProperty("color", colorToRmlAlpha(p.info, wasd_alpha));
                }
                auto* sep = wasd_section_ ? wasd_section_->GetElementById("wasd-sep") : nullptr;
                if (!sep && wasd_section_ && wasd_section_->GetNumChildren() > 0)
                    sep = wasd_section_->GetChild(0);
                if (sep)
                    sep->SetProperty("color", colorToRmlAlpha(p.text_dim, wasd_alpha));
            }
        } else if (cache_.wasd_alpha > 0.0f) {
            cache_.wasd_alpha = 0.0f;
            cache_.wasd_rml.clear();
            if (wasd_section_)
                wasd_section_->SetClass("hidden", true);
        }

        auto [zoom_speed, zoom_alpha] = speed_state_.getZoom();
        bool zoom_visible = zoom_alpha > 0.0f;

        if (zoom_visible) {
            auto zoom_rml = std::format("{}: {:.0f}",
                                        stripColon(LOC(lichtfeld::Strings::Controls::ZOOM)),
                                        zoom_speed * 10.0f);
            if (zoom_rml != cache_.zoom_rml || zoom_alpha != cache_.zoom_alpha) {
                cache_.zoom_rml = zoom_rml;
                cache_.zoom_alpha = zoom_alpha;
                if (zoom_section_)
                    zoom_section_->SetClass("hidden", false);
                if (zoom_text_) {
                    zoom_text_->SetInnerRML(zoom_rml);
                    zoom_text_->SetProperty("color", colorToRmlAlpha(p.info, zoom_alpha));
                }
                auto* sep = zoom_section_ ? zoom_section_->GetElementById("zoom-sep") : nullptr;
                if (!sep && zoom_section_ && zoom_section_->GetNumChildren() > 0)
                    sep = zoom_section_->GetChild(0);
                if (sep)
                    sep->SetProperty("color", colorToRmlAlpha(p.text_dim, zoom_alpha));
            }
        } else if (cache_.zoom_alpha > 0.0f) {
            cache_.zoom_alpha = 0.0f;
            cache_.zoom_rml.clear();
            if (zoom_section_)
                zoom_section_->SetClass("hidden", true);
        }

        // Right section: GPU memory
        auto mem = queryGpuMemory();
        float app_gb = mem.process_used / 1e9f;
        float used_gb = mem.total_used / 1e9f;
        float total_gb = mem.total / 1e9f;
        float pct = total_gb > 0.0f ? (used_gb / total_gb) * 100.0f : 0.0f;

        ImVec4 mem_color = pct < 50.0f ? p.success : (pct < 75.0f ? p.warning : p.error);

        auto lfs_mem_rml = std::format("LFS {:.1f}GB", app_gb);
        if (lfs_mem_rml != cache_.lfs_mem_rml) {
            cache_.lfs_mem_rml = lfs_mem_rml;
            if (lfs_mem_) {
                lfs_mem_->SetInnerRML(lfs_mem_rml);
                lfs_mem_->SetProperty("color", colorToRml(p.info));
            }
        }

        auto gpu_mem_rml = std::format("{} {:.1f}/{:.1f}GB",
                                       LOC("status_bar.gpu"), used_gb, total_gb);
        auto gpu_mem_color = colorToRml(mem_color);
        if (gpu_mem_rml != cache_.gpu_mem_rml || gpu_mem_color != cache_.gpu_mem_color) {
            cache_.gpu_mem_rml = gpu_mem_rml;
            cache_.gpu_mem_color = gpu_mem_color;
            if (gpu_mem_) {
                gpu_mem_->SetInnerRML(gpu_mem_rml);
                gpu_mem_->SetProperty("color", gpu_mem_color);
            }
        }

        // FPS
        float fps = rm ? rm->getAverageFPS() : 0.0f;
        ImVec4 fps_col = fps >= 30.0f ? p.success : (fps >= 15.0f ? p.warning : p.error);

        auto fps_rml = std::format("{:.0f}", fps);
        auto fps_color = colorToRml(fps_col);
        if (fps_rml != cache_.fps_rml || fps_color != cache_.fps_color) {
            cache_.fps_rml = fps_rml;
            cache_.fps_color = fps_color;
            if (fps_value_) {
                fps_value_->SetInnerRML(fps_rml);
                fps_value_->SetProperty("color", fps_color);
            }
        }

        // FPS label and git commit (set once)
        if (!cache_.git_set) {
            cache_.git_set = true;
            if (fps_label_)
                fps_label_->SetInnerRML(std::format(" {}", LOC(lichtfeld::Strings::Status::FPS)));
            if (git_commit_)
                git_commit_->SetInnerRML(GIT_COMMIT_HASH_SHORT);
        }
    }

    void RmlStatusBar::draw(const PanelDrawContext& ctx) {
        if (!rml_context_ || !document_)
            return;

        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float avail_h = ImGui::GetContentRegionAvail().y;
        if (avail_w <= 0 || avail_h <= 0)
            return;

        if (!rml_manager_->shouldDeferFboUpdate(fbo_)) {
            updateTheme();
            updateContent(ctx);

            const int w = static_cast<int>(avail_w);
            const int h = static_cast<int>(avail_h);

            rml_context_->SetDimensions(Rml::Vector2i(w, h));
            document_->SetProperty("height", std::format("{}px", h));
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
        }

        if (fbo_.valid())
            fbo_.blitAsImage(avail_w, avail_h);
    }

} // namespace lfs::vis::gui
