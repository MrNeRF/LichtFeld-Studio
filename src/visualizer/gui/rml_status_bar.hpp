/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/reactive/store.hpp"
#include "gui/error_surface_types.hpp"
#include "gui/gpu_memory_query.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include <RmlUi/Core/DataModelHandle.h>
#include <chrono>
#include <cstddef>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

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

    // Thread-safe one-line transient status (ErrorBus StatusOnly surface).
    // post() from any thread overwrites and restarts the timer; snapshot() copies
    // under the mutex for the UI-thread refresh and self-clears after kDuration.
    struct LFS_VIS_API StatusMessageState {
        static constexpr std::chrono::milliseconds kDuration{5000};
        static constexpr float kFadeMs = 500.0f;

        void post(std::string text, ErrorNoticeLevel level);

        struct Snapshot {
            bool visible = false;
            std::string text;
            ErrorNoticeLevel level = ErrorNoticeLevel::Info;
            float alpha = 0.0f;
        };
        [[nodiscard]] Snapshot snapshot(std::chrono::steady_clock::time_point now);

    private:
        std::mutex mutex_;
        std::string text_;
        ErrorNoticeLevel level_ = ErrorNoticeLevel::Info;
        std::chrono::steady_clock::time_point posted_at_{};
        bool has_message_ = false;
    };

    class RmlStatusBar {
    public:
        void init(RmlUIManager* mgr);
        // Thread-safe one-line transient status text (ErrorBus StatusOnly
        // surface). Picked up by the next periodic content refresh; auto-clears.
        void postStatusMessage(std::string text, ErrorNoticeLevel level);
        void shutdown();
        void reloadResources();
        void render(const PanelDrawContext& ctx, float x, float y, float w, float h,
                    int screen_w, int screen_h);
        void renderCached(const PanelDrawContext& ctx, float x, float y, float w, float h,
                          int screen_w, int screen_h);
        void processInput(const PanelInputState& input, float bar_x, float bar_y,
                          float bar_w, float bar_h);

    private:
        struct ProgressBarGeometry {
            float x = 0.0f;
            float y = 0.0f;
            float w = 0.0f;
            float h = 0.0f;
        };

        bool updateContent(const PanelDrawContext& ctx, bool force_refresh);
        bool updateTheme();
        void queueCachedVulkanContext(float x, float y, float w_px, float h_px,
                                      int screen_w, int screen_h,
                                      int render_w, int render_h,
                                      bool refresh_cache);
        void pollGpuMemoryQuery(std::chrono::steady_clock::time_point now);
        void setModelString(const char* name, std::string& field, std::string value);
        void setModelBool(const char* name, bool& field, bool value);
        void setProgressMarkersRml(std::string value);
        std::optional<ProgressBarGeometry> progressBarGeometry() const;
        void resetSaveStepInteraction();
        std::optional<size_t> hitSaveStep(float local_x, float local_y,
                                          const ProgressBarGeometry& geom,
                                          const std::vector<size_t>& save_steps,
                                          int total_iterations) const;
        size_t saveStepFromProgressX(float local_x, const ProgressBarGeometry& geom,
                                     int current_iteration, int total_iterations) const;
        void handleSaveStepInteraction(const PanelInputState& input, float local_x, float local_y);
        void commitSaveStepEdit();
        void removeSaveStep(size_t step);
        void clearSaveStepHover();
        void attachElementListeners();
        void bindReactiveStore();
        void markModelDirty();

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::DataModelHandle model_handle_;
        Rml::EventListener* git_commit_listener_ = nullptr;
        Rml::EventListener* gpu_icon_listener_ = nullptr;

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

        struct SaveStepInteractionState {
            bool dragging = false;
            bool adding = false;
            size_t original_step = 0;
            size_t preview_step = 0;
            size_t hover_step = 0;
        };

        SaveStepInteractionState save_step_interaction_;

        struct ModelState {
            std::string mode_text;
            std::string mode_color;
            bool show_training = false;
            std::string progress_width = "0%";
            std::string progress_text;
            std::string progress_markers_rml;
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
            bool gpu_panel_active = false;
            std::string gpu_model_text;
            std::string gpu_mem_text;
            std::string gpu_mem_color;
            std::string fps_value;
            std::string fps_color;
            std::string fps_label;
            std::string git_commit;
            bool show_status_message = false;
            std::string status_message_text;
            std::string status_message_color;
        };

        ModelState model_;
        StatusMessageState status_message_;
        GpuMemoryInfo cached_gpu_mem_;
        std::future<GpuMemoryInfo> pending_gpu_mem_;
        std::chrono::steady_clock::time_point next_refresh_at_{};
        std::chrono::steady_clock::time_point next_gpu_refresh_at_{};
        bool model_dirty_ = true;
        bool animation_active_ = false;
        bool reactive_fps_available_ = false;
        float reactive_fps_value_ = 0.0f;
        std::vector<lfs::core::reactive::SubscriptionToken> subscriptions_;
        int last_render_w_ = 0;
        int last_render_h_ = 0;
        int last_document_h_ = 0;
        CachedVulkanContextRender direct_cache_;
        static constexpr auto kIdleRefreshInterval = std::chrono::milliseconds(200);
        static constexpr auto kBusyRefreshInterval = std::chrono::milliseconds(100);
        static constexpr auto kAnimatedRefreshInterval = std::chrono::milliseconds(16);
        static constexpr auto kGpuRefreshInterval = std::chrono::milliseconds(500);
    };

} // namespace lfs::vis::gui
