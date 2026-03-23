/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "camera_interaction_service.hpp"
#include "core/export.hpp"
#include "dirty_flags.hpp"
#include "framerate_controller.hpp"
#include "gt_texture_cache.hpp"
#include "internal/viewport.hpp"
#include "render_animation_state.hpp"
#include "render_pass_graph.hpp"
#include "rendering/cuda_gl_interop.hpp"
#include "rendering/rendering.hpp"
#include "rendering_types.hpp"
#include "split_view_service.hpp"
#include "viewport_artifact_service.hpp"
#include "viewport_frame_lifecycle_service.hpp"
#include "viewport_interaction_context.hpp"
#include "viewport_overlay_service.hpp"
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
} // namespace lfs::core::events::ui

namespace lfs::vis {
    class SceneManager;

    class LFS_VIS_API RenderingManager {
    public:
        struct RenderContext {
            const Viewport& viewport;
            const RenderSettings& settings;
            const ViewportRegion* viewport_region = nullptr;
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
        void renderFrame(const RenderContext& context);

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
            camera_interaction_service_.setCurrentCameraId(cam_id);
            markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::PPISP);
        }
        int getCurrentCameraId() const { return camera_interaction_service_.currentCameraId(); }

        // FPS monitoring
        float getCurrentFPS() const { return framerate_controller_.getCurrentFPS(); }
        float getAverageFPS() const { return framerate_controller_.getAverageFPS(); }

        // Access to rendering engine (for initialization only)
        lfs::rendering::RenderingEngine* getRenderingEngine();

        // Camera frustum picking
        int pickCameraFrustum(const glm::vec2& mouse_pos);

        // Depth buffer access for tools (returns camera-space depth at pixel, or -1 if invalid)
        float getDepthAtPixel(int x, int y) const;
        glm::ivec2 getRenderedSize() const { return viewport_artifact_service_.renderedSize(); }
        std::shared_ptr<lfs::core::Tensor> getViewportImageIfAvailable() const;
        std::shared_ptr<lfs::core::Tensor> captureViewportImage();
        [[nodiscard]] uint64_t getViewportArtifactGeneration() const {
            return viewport_artifact_service_.artifactGeneration();
        }

        void setCursorPreviewState(bool active, float x, float y, float radius, bool add_mode = true,
                                   lfs::core::Tensor* selection_tensor = nullptr,
                                   bool saturation_mode = false, float saturation_amount = 0.0f);
        void clearCursorPreviewState();
        [[nodiscard]] bool isCursorPreviewActive() const { return viewport_overlay_service_.isCursorPreviewActive(); }
        void getCursorPreviewState(float& x, float& y, float& radius, bool& add_mode) const {
            const auto& cursor = viewport_overlay_service_.cursorPreview();
            x = cursor.x;
            y = cursor.y;
            radius = cursor.radius;
            add_mode = cursor.add_mode;
        }

        // Rectangle preview
        void setRectPreview(float x0, float y0, float x1, float y1, bool add_mode = true);
        void clearRectPreview();
        [[nodiscard]] bool isRectPreviewActive() const { return viewport_overlay_service_.isRectPreviewActive(); }
        void getRectPreview(float& x0, float& y0, float& x1, float& y1, bool& add_mode) const {
            x0 = viewport_overlay_service_.rectX0();
            y0 = viewport_overlay_service_.rectY0();
            x1 = viewport_overlay_service_.rectX1();
            y1 = viewport_overlay_service_.rectY1();
            add_mode = viewport_overlay_service_.rectAddMode();
        }

        // Polygon preview (render-space points, same coordinate system as screen_positions output)
        void setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed, bool add_mode = true);
        // Interactive polygon preview in world-space coordinates.
        void setPolygonPreviewWorldSpace(const std::vector<glm::vec3>& world_points, bool closed,
                                         bool add_mode = true);
        void clearPolygonPreview();
        [[nodiscard]] bool isPolygonPreviewActive() const { return viewport_overlay_service_.isPolygonPreviewActive(); }
        [[nodiscard]] const std::vector<std::pair<float, float>>& getPolygonPoints() const {
            return viewport_overlay_service_.polygonPoints();
        }
        [[nodiscard]] const std::vector<glm::vec3>& getPolygonWorldPoints() const {
            return viewport_overlay_service_.polygonWorldPoints();
        }
        [[nodiscard]] bool isPolygonClosed() const { return viewport_overlay_service_.polygonClosed(); }
        [[nodiscard]] bool isPolygonAddMode() const { return viewport_overlay_service_.polygonAddMode(); }
        [[nodiscard]] bool isPolygonPreviewWorldSpace() const {
            return viewport_overlay_service_.polygonWorldSpace();
        }

        // Lasso preview
        void setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode = true);
        void clearLassoPreview();
        [[nodiscard]] bool isLassoPreviewActive() const { return viewport_overlay_service_.isLassoPreviewActive(); }
        [[nodiscard]] const std::vector<std::pair<float, float>>& getLassoPoints() const {
            return viewport_overlay_service_.lassoPoints();
        }
        [[nodiscard]] bool isLassoAddMode() const { return viewport_overlay_service_.lassoAddMode(); }

        // Preview selection
        void setPreviewSelection(lfs::core::Tensor* preview, bool add_mode = true) {
            viewport_overlay_service_.setPreviewSelection(preview, add_mode);
            markDirty(DirtyFlag::SELECTION);
        }
        void clearPreviewSelection() {
            viewport_overlay_service_.clearPreviewSelection();
            markDirty(DirtyFlag::SELECTION);
        }
        void clearSelectionPreviews();

        // Selection preview mode for viewport interaction overlays
        void setSelectionPreviewMode(SelectionPreviewMode mode) {
            viewport_overlay_service_.setSelectionPreviewMode(mode);
        }
        [[nodiscard]] SelectionPreviewMode getSelectionPreviewMode() const {
            return viewport_overlay_service_.selectionPreviewMode();
        }
        [[nodiscard]] int getHoveredGaussianId() const { return viewport_overlay_service_.hoveredGaussianId(); }

        // Sync selection group colors to GPU constant memory
        void syncSelectionGroupColor(int group_id, const glm::vec3& color);

        // Gizmo state for wireframe sync during manipulation
        void setCropboxGizmoState(bool active, const glm::vec3& min, const glm::vec3& max,
                                  const glm::mat4& world_transform) {
            viewport_overlay_service_.setCropbox(active, min, max, world_transform);
        }
        void setEllipsoidGizmoState(bool active, const glm::vec3& radii,
                                    const glm::mat4& world_transform) {
            viewport_overlay_service_.setEllipsoid(active, radii, world_transform);
        }
        void setCropboxGizmoActive(bool active) { viewport_overlay_service_.setCropboxActive(active); }
        void setEllipsoidGizmoActive(bool active) { viewport_overlay_service_.setEllipsoidActive(active); }

        void setViewportResizeActive(bool active);
        [[nodiscard]] bool isViewportResizeDeferring() const {
            return frame_lifecycle_service_.isResizeDeferring();
        }
        bool consumeResizeCompleted() { return frame_lifecycle_service_.consumeResizeCompleted(); }

    private:
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
        RenderPassGraph pass_graph_;
        mutable FramerateController framerate_controller_;

        // GT texture cache
        GTTextureCache gt_texture_cache_;

        // Granular dirty tracking
        std::atomic<uint32_t> dirty_mask_{DirtyFlag::ALL};

        RenderAnimationState animation_state_;
        ViewportArtifactService viewport_artifact_service_;

        CameraInteractionService camera_interaction_service_;
        SplitViewService split_view_service_;
        ViewportFrameLifecycleService frame_lifecycle_service_;

        // Settings
        RenderSettings settings_;
        mutable std::mutex settings_mutex_;

        bool initialized_ = false;
        glm::ivec2 initial_viewport_size_{1280, 720}; // Default fallback

        ViewportInteractionContext viewport_interaction_context_;

        // Debug tracking
        uint64_t render_count_ = 0;

        ViewportOverlayService viewport_overlay_service_;
    };

} // namespace lfs::vis
