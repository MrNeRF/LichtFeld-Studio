/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/gpu_memory_query.hpp"
#include "gui/panel_registry.hpp"
#include <RmlUi/Core/DataModelHandle.h>
#include <chrono>
#include <cstddef>
#include <string>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
    class EventListener;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    class RmlUIManager;

    class RmlStatusBar {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void reloadResources();
        void render(const PanelDrawContext& ctx, float x, float y, float w, float h,
                    int screen_w, int screen_h);
        void processInput(const PanelInputState& input, float bar_x, float bar_y,
                          float bar_w, float bar_h);

    private:
        bool updateContent(const PanelDrawContext& ctx, bool force_refresh);
        bool updateTheme();
        void setModelString(const char* name, std::string& field, std::string value);
        void setModelBool(const char* name, bool& field, bool value);
        void attachGitCommitListener();

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::DataModelHandle model_handle_;
        Rml::EventListener* git_commit_listener_ = nullptr;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;

        struct SpeedOverlayState {
            float wasd_speed = 0.0f;
            float zoom_speed = 0.0f;
            std::chrono::steady_clock::time_point wasd_start;
            std::chrono::steady_clock::time_point zoom_start;
            bool wasd_visible = false;
            bool zoom_visible = false;
            static constexpr auto DURATION = std::chrono::milliseconds(3000);
            static constexpr float FADE_MS = 500.0f;

            void showWasd(float speed);
            void showZoom(float speed);
            std::pair<float, float> getWasd() const;
            std::pair<float, float> getZoom() const;
        };

        SpeedOverlayState speed_state_;
        bool speed_events_initialized_ = false;

        struct ModelState {
            std::string mode_text;
            std::string mode_color;
            bool show_training = false;
            std::string progress_width = "0%";
            std::string progress_text;
            std::string step_label;
            std::string step_value;
            std::string loss_label;
            std::string loss_value;
            bool show_eval_metrics = false;
            std::string eval_metrics_value;
            std::string gaussians_label;
            std::string gaussians_value;
            std::string time_value;
            std::string eta_label;
            std::string eta_value;
            bool show_splats = false;
            std::string splat_text;
            std::string splat_color;
            bool show_split = false;
            std::string split_mode;
            std::string split_mode_color;
            std::string split_detail;
            bool show_wasd = false;
            std::string wasd_text;
            std::string wasd_color;
            std::string wasd_sep_color;
            bool show_zoom = false;
            std::string zoom_text;
            std::string zoom_color;
            std::string zoom_sep_color;
            std::string lfs_mem_text;
            std::string lfs_mem_color;
            bool show_gpu_model = false;
            std::string gpu_model_text;
            std::string gpu_mem_text;
            std::string gpu_mem_color;
            std::string fps_value;
            std::string fps_color;
            std::string fps_label;
            std::string git_commit;
        };

        ModelState model_;
        GpuMemoryInfo cached_gpu_mem_;
        std::chrono::steady_clock::time_point next_refresh_at_{};
        std::chrono::steady_clock::time_point next_gpu_refresh_at_{};
        bool model_dirty_ = true;
        bool animation_active_ = false;
        int last_render_w_ = 0;
        int last_render_h_ = 0;
        int last_document_h_ = 0;
        static constexpr auto kIdleRefreshInterval = std::chrono::milliseconds(200);
        static constexpr auto kBusyRefreshInterval = std::chrono::milliseconds(100);
        static constexpr auto kAnimatedRefreshInterval = std::chrono::milliseconds(16);
        static constexpr auto kGpuRefreshInterval = std::chrono::milliseconds(250);
    };

} // namespace lfs::vis::gui
