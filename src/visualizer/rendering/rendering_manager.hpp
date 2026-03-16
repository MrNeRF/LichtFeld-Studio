/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "dirty_flags.hpp"
#include "framerate_controller.hpp"
#include "internal/viewport.hpp"
#include "gt_texture_cache.hpp"
#include "rendering_manager_state.hpp"
#include "rendering/cuda_gl_interop.hpp"
#include "rendering/rendering.hpp"
#include "rendering_types.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::core {
    class Tensor;
}

namespace lfs::core::events::ui {
    struct GridSettingsChanged;
    struct PointCloudModeChanged;
    struct RenderSettingsChanged;
}

namespace lfs::vis {
    class RenderPass;
    class SceneManager;
    class SplatRasterPass;
    class PointCloudPass;

    class LFS_VIS_API RenderingManager {
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

        // Render preview to external texture (for PiP preview)
        bool renderPreviewFrame(SceneManager* scene_manager,
                                const glm::mat3& camera_rotation,
                                const glm::vec3& camera_position,
                                float focal_length_mm,
                                unsigned int target_fbo,
                                unsigned int target_texture,
                                int width, int height);

        void markDirty();
        void markDirty(DirtyMask flags);

        [[nodiscard]] bool pollDirtyState() {
            if (const DirtyMask animation_dirty = animation_state_.pollDirtyState(); animation_dirty) {
                dirty_mask_.fetch_or(animation_dirty, std::memory_order_relaxed);
                return true;
            }
            return dirty_mask_.load(std::memory_order_relaxed) != 0;
        }

        void setPivotAnimationEndTime(const std::chrono::steady_clock::time_point end_time) {
            animation_state_.setPivotAnimationEndTime(end_time);
        }

        void triggerSelectionFlash() {
            markDirty(animation_state_.triggerSelectionFlash());
        }

        void setOverlayAnimationActive(const bool active) { animation_state_.setOverlayAnimationActive(active); }

        [[nodiscard]] float getSelectionFlashIntensity() const {
            return animation_state_.selectionFlashIntensity();
        }

        // Settings management
        void updateSettings(const RenderSettings& settings);
        RenderSettings getSettings() const;

        // Toggle orthographic mode, calculating ortho_scale to preserve size at pivot
        void setOrthographic(bool enabled, float viewport_height, float distance_to_pivot);

        float getFovDegrees() const;
        float getScalingModifier() const;
        void setScalingModifier(float s);
        float getFocalLengthMm() const;
        void setFocalLength(float focal_mm);

        void advanceSplitOffset();
        SplitViewInfo getSplitViewInfo() const;

        struct ContentBounds {
            float x, y, width, height;
            bool letterboxed = false;
        };
        ContentBounds getContentBounds(const glm::ivec2& viewport_size) const;

        // Current camera tracking for GT comparison
        void setCurrentCameraId(int cam_id) {
            camera_interaction_state_.current_camera_id = cam_id;
            markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::PPISP);
        }
        int getCurrentCameraId() const { return camera_interaction_state_.current_camera_id; }

        // FPS monitoring
        float getCurrentFPS() const { return framerate_controller_.getCurrentFPS(); }
        float getAverageFPS() const { return framerate_controller_.getAverageFPS(); }

        // Access to rendering engine (for initialization only)
        lfs::rendering::RenderingEngine* getRenderingEngine();

        // Camera frustum picking
        int pickCameraFrustum(const glm::vec2& mouse_pos);
        void setHoveredCameraId(int cam_id) { camera_interaction_state_.hovered_camera_id = cam_id; }
        int getHoveredCameraId() const { return camera_interaction_state_.hovered_camera_id; }

        // Depth buffer access for tools (returns camera-space depth at pixel, or -1 if invalid)
        float getDepthAtPixel(int x, int y) const;
        glm::ivec2 getRenderedSize() const { return viewport_artifacts_.rendered_size; }
        std::shared_ptr<lfs::core::Tensor> getViewportImageIfAvailable() const;
        std::shared_ptr<lfs::core::Tensor> captureViewportImage();
        [[nodiscard]] uint64_t getViewportArtifactGeneration() const { return viewport_artifacts_.artifact_generation; }

        void setBrushState(bool active, float x, float y, float radius, bool add_mode = true,
                           lfs::core::Tensor* selection_tensor = nullptr,
                           bool saturation_mode = false, float saturation_amount = 0.0f);
        void clearBrushState();
        [[nodiscard]] bool isBrushActive() const { return interaction_state_.brush.active; }
        void getBrushState(float& x, float& y, float& radius, bool& add_mode) const {
            x = interaction_state_.brush.x;
            y = interaction_state_.brush.y;
            radius = interaction_state_.brush.radius;
            add_mode = interaction_state_.brush.add_mode;
        }

        // Rectangle preview
        void setRectPreview(float x0, float y0, float x1, float y1, bool add_mode = true);
        void clearRectPreview();
        [[nodiscard]] bool isRectPreviewActive() const { return interaction_state_.rect.active; }
        void getRectPreview(float& x0, float& y0, float& x1, float& y1, bool& add_mode) const {
            x0 = interaction_state_.rect.x0;
            y0 = interaction_state_.rect.y0;
            x1 = interaction_state_.rect.x1;
            y1 = interaction_state_.rect.y1;
            add_mode = interaction_state_.rect.add_mode;
        }

        // Polygon preview (render-space points, same coordinate system as screen_positions output)
        void setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed, bool add_mode = true);
        // Interactive polygon preview in world-space coordinates.
        void setPolygonPreviewWorldSpace(const std::vector<glm::vec3>& world_points, bool closed,
                                         bool add_mode = true);
        void clearPolygonPreview();
        [[nodiscard]] bool isPolygonPreviewActive() const { return interaction_state_.polygon.active; }
        [[nodiscard]] const std::vector<std::pair<float, float>>& getPolygonPoints() const { return interaction_state_.polygon.points; }
        [[nodiscard]] const std::vector<glm::vec3>& getPolygonWorldPoints() const { return interaction_state_.polygon.world_points; }
        [[nodiscard]] bool isPolygonClosed() const { return interaction_state_.polygon.closed; }
        [[nodiscard]] bool isPolygonAddMode() const { return interaction_state_.polygon.add_mode; }
        [[nodiscard]] bool isPolygonPreviewWorldSpace() const { return interaction_state_.polygon.world_space; }

        // Lasso preview
        void setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode = true);
        void clearLassoPreview();
        [[nodiscard]] bool isLassoPreviewActive() const { return interaction_state_.lasso.active; }
        [[nodiscard]] const std::vector<std::pair<float, float>>& getLassoPoints() const { return interaction_state_.lasso.points; }
        [[nodiscard]] bool isLassoAddMode() const { return interaction_state_.lasso.add_mode; }

        // Preview selection
        void setPreviewSelection(lfs::core::Tensor* preview, bool add_mode = true) {
            interaction_state_.setPreviewSelection(preview, add_mode);
            markDirty(DirtyFlag::SELECTION);
        }
        void clearPreviewSelection() {
            interaction_state_.clearPreviewSelection();
            markDirty(DirtyFlag::SELECTION);
        }
        void clearSelectionPreviews();

        // Selection mode for brush tool
        void setSelectionMode(lfs::rendering::SelectionMode mode) { interaction_state_.setSelectionMode(mode); }
        [[nodiscard]] lfs::rendering::SelectionMode getSelectionMode() const { return interaction_state_.getSelectionMode(); }
        [[nodiscard]] int getHoveredGaussianId() const { return interaction_state_.hovered_gaussian_id; }

        // Sync selection group colors to GPU constant memory
        void syncSelectionGroupColor(int group_id, const glm::vec3& color);

        // Gizmo state for wireframe sync during manipulation
        void setCropboxGizmoState(bool active, const glm::vec3& min, const glm::vec3& max,
                                  const glm::mat4& world_transform) {
            gizmo_state_.setCropbox(active, min, max, world_transform);
        }
        void setEllipsoidGizmoState(bool active, const glm::vec3& radii,
                                    const glm::mat4& world_transform) {
            gizmo_state_.setEllipsoid(active, radii, world_transform);
        }
        void setCropboxGizmoActive(bool active) { gizmo_state_.cropbox_active = active; }
        void setEllipsoidGizmoActive(bool active) { gizmo_state_.ellipsoid_active = active; }

        void setViewportResizeActive(bool active);
        [[nodiscard]] bool isViewportResizeDeferring() const {
            return frame_lifecycle_state_.isResizeDeferring();
        }
        bool consumeResizeCompleted() { return frame_lifecycle_state_.consumeResizeCompleted(); }

    private:
        void doFullRender(const RenderContext& context, SceneManager* scene_manager,
                          const lfs::core::SplatData* model);
        void setupEventHandlers();
        void handleToggleSplitView();
        void handleToggleGTComparison();
        void handleGoToCamView(int cam_id);
        void handleSplitPositionChanged(float position);
        void handleRenderSettingsChanged(const lfs::core::events::ui::RenderSettingsChanged& event);
        void handleWindowResized();
        void handleGridSettingsChanged(const lfs::core::events::ui::GridSettingsChanged& event);
        void handleSceneLoaded();
        void handleSceneChanged();
        void handleSceneCleared();
        void handlePLYVisibilityChanged();
        void handlePLYAdded();
        void handlePLYRemoved();
        void handleCropBoxChanged(bool enabled);
        void handleEllipsoidChanged(bool enabled);
        void handlePointCloudModeChanged(const lfs::core::events::ui::PointCloudModeChanged& event);

        // Core components
        std::unique_ptr<lfs::rendering::RenderingEngine> engine_;
        std::vector<std::unique_ptr<RenderPass>> passes_;
        SplatRasterPass* splat_raster_pass_ = nullptr;
        PointCloudPass* point_cloud_pass_ = nullptr;
        mutable FramerateController framerate_controller_;

        // GT texture cache
        GTTextureCache gt_texture_cache_;

        // Granular dirty tracking
        std::atomic<uint32_t> dirty_mask_{DirtyFlag::ALL};

        AnimationState animation_state_;
        ViewportArtifactState viewport_artifacts_;

        CameraInteractionState camera_interaction_state_;
        SplitViewState split_view_state_;
        ViewportFrameLifecycleState frame_lifecycle_state_;

        // Settings
        RenderSettings settings_;
        mutable std::mutex settings_mutex_;

        bool initialized_ = false;
        glm::ivec2 initial_viewport_size_{1280, 720}; // Default fallback

        ViewportInteractionContextState viewport_interaction_context_;

        // Debug tracking
        uint64_t render_count_ = 0;

        InteractionPreviewState interaction_state_;
        RenderGizmoState gizmo_state_;
        GpuDepthReadbackState gpu_depth_readback_state_;
    };

} // namespace lfs::vis
