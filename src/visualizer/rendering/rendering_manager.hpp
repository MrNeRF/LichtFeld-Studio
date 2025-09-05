/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "framerate_controller.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>

namespace gs {
    class SceneManager;
} // namespace gs

namespace gs::visualizer {

    struct RenderSettings {
        // Core rendering settings
        float fov = 60.0f;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;

        // Crop box
        bool show_crop_box = false;
        bool use_crop_box = false;
        glm::vec3 crop_min = glm::vec3(-1.0f, -1.0f, -1.0f);
        glm::vec3 crop_max = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 crop_color = glm::vec3(1.0f, 1.0f, 0.0f);
        float crop_line_width = 2.0f;
        geometry::EuclideanTransform crop_transform;

        // Background
        glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);

        // Coordinate axes
        bool show_coord_axes = false;
        float axes_size = 2.0f;
        std::array<bool, 3> axes_visibility = {true, true, true};

        // World transform
        geometry::EuclideanTransform world_transform;

        // Grid
        bool show_grid = true;
        int grid_plane = 1;
        float grid_opacity = 0.5f;

        // Point cloud
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;

        // Translation gizmo
        bool show_translation_gizmo = false;
        float gizmo_scale = 1.0f;

        // Camera frustums
        bool show_camera_frustums = false;
        float camera_frustum_scale = 0.25f;
        glm::vec3 train_camera_color = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 eval_camera_color = glm::vec3(1.0f, 0.0f, 0.0f);

        // Split view
        bool split_view_enabled = false;
        float split_position = 0.5f;
        size_t split_view_offset = 0;

        bool gut = false;
    };

    struct SplitViewInfo {
        bool enabled = false;
        std::string left_name;
        std::string right_name;
    };

    struct ViewportRegion {
        float x, y, width, height;
    };

    class RenderingManager {
    public:
        struct RenderContext {
            const Viewport& viewport;
            const RenderSettings& settings;
            const ViewportRegion* viewport_region = nullptr;
            bool has_focus = false;
            SceneManager* scene_manager = nullptr;
        };

        RenderingManager();
        ~RenderingManager();

        // Initialize rendering resources
        void initialize();
        bool isInitialized() const { return initialized_; }

        // Set initial viewport size (must be called before initialize())
        void setInitialViewportSize(const glm::ivec2& size) {
            initial_viewport_size_ = size;
        }

        // Main render function
        void renderFrame(const RenderContext& context, SceneManager* scene_manager);

        // Mark that rendering is needed
        void markDirty();

        // Settings management
        void updateSettings(const RenderSettings& settings);
        RenderSettings getSettings() const;

        // Direct accessors
        float getFovDegrees() const;
        float getScalingModifier() const;
        void setFov(float f);
        void setScalingModifier(float s);

        // Split view control
        void advanceSplitOffset();
        SplitViewInfo getSplitViewInfo() const;

        // FPS monitoring
        float getCurrentFPS() const { return framerate_controller_.getCurrentFPS(); }
        float getAverageFPS() const { return framerate_controller_.getAverageFPS(); }

        // Access to rendering engine (for initialization only)
        gs::rendering::RenderingEngine* getRenderingEngine();

        // Camera frustum picking
        int pickCameraFrustum(const glm::vec2& mouse_pos);
        void setHoveredCameraId(int cam_id) { hovered_camera_id_ = cam_id; }
        int getHoveredCameraId() const { return hovered_camera_id_; }

    private:
        void doFullRender(const RenderContext& context, SceneManager* scene_manager, const SplatData* model);
        void renderOverlays(const RenderContext& context);
        void setupEventHandlers();

        std::optional<gs::rendering::SplitViewRequest> createSplitViewRequest(
            const RenderContext& context,
            SceneManager* scene_manager);

        // Core components
        std::unique_ptr<gs::rendering::RenderingEngine> engine_;
        FramerateController framerate_controller_;

        // State tracking
        std::atomic<bool> needs_render_{true};
        gs::rendering::RenderResult cached_result_;
        size_t last_model_ptr_ = 0;
        glm::ivec2 last_render_size_{0, 0};
        std::chrono::steady_clock::time_point last_training_render_;

        // Split view state
        mutable std::mutex split_info_mutex_;
        SplitViewInfo current_split_info_;

        // Settings
        RenderSettings settings_;
        mutable std::mutex settings_mutex_;

        bool initialized_ = false;
        glm::ivec2 initial_viewport_size_{1280, 720}; // Default fallback

        // Camera picking state
        int hovered_camera_id_ = -1;
        int highlighted_camera_index_ = -1;
        glm::vec2 pending_pick_pos_{-1, -1};
        bool pick_requested_ = false;
        std::chrono::steady_clock::time_point last_pick_time_;
        static constexpr auto pick_throttle_interval_ = std::chrono::milliseconds(50);

        // Debug tracking
        uint64_t render_count_ = 0;
        uint64_t pick_count_ = 0;
    };

} // namespace gs::visualizer