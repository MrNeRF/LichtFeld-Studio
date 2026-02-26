/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "dirty_flags.hpp"
#include "framerate_controller.hpp"
#include "internal/viewport.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "rendering/cuda_gl_interop.hpp"
#include "rendering/rendering.hpp"
#include "rendering_types.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lfs::vis {
    class RenderPass;
    class SceneManager;
    class SplatRasterPass;
    class OverlayPass;
    class PointCloudPass;

    // GT Image Cache for efficient GPU-resident texture management
    class GTTextureCache {
    public:
        static constexpr int MAX_TEXTURE_DIM = 2048;

        struct TextureInfo {
            unsigned int texture_id = 0;
            int width = 0;
            int height = 0;
            bool needs_flip = false;
            glm::vec2 texcoord_scale{1.0f};
        };

        GTTextureCache();
        ~GTTextureCache();

        TextureInfo getGTTexture(int cam_id, const std::filesystem::path& image_path);
        void clear();

    private:
        struct CacheEntry {
            std::unique_ptr<lfs::rendering::CudaGLInteropTexture> interop_texture;
            unsigned int texture_id = 0;
            int width = 0;
            int height = 0;
            bool needs_flip = false;
            std::chrono::steady_clock::time_point last_access;
        };

        std::unordered_map<int, CacheEntry> texture_cache_;
        std::unique_ptr<lfs::io::NvCodecImageLoader> nvcodec_loader_;
        static constexpr size_t MAX_CACHE_SIZE = 20;

        void evictOldest();
        TextureInfo loadTexture(const std::filesystem::path& path);
        TextureInfo loadTextureGPU(const std::filesystem::path& path, CacheEntry& entry);
    };

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
            if (pivot_animation_active_.load() &&
                std::chrono::steady_clock::now() < from_ns(pivot_animation_end_ns_.load(std::memory_order_acquire))) {
                dirty_mask_.fetch_or(DirtyFlag::CAMERA | DirtyFlag::OVERLAY, std::memory_order_relaxed);
                return true;
            }
            pivot_animation_active_.store(false);

            if (selection_flash_active_.load()) {
                const auto elapsed = std::chrono::steady_clock::now() -
                                     from_ns(selection_flash_start_ns_.load(std::memory_order_acquire));
                if (std::chrono::duration<float>(elapsed).count() < SELECTION_FLASH_DURATION_SEC) {
                    dirty_mask_.fetch_or(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY, std::memory_order_relaxed);
                    return true;
                }
                selection_flash_active_.store(false);
            }

            if (overlay_animation_active_.load()) {
                dirty_mask_.fetch_or(DirtyFlag::OVERLAY, std::memory_order_relaxed);
                return true;
            }
            return dirty_mask_.load(std::memory_order_relaxed) != 0;
        }

        void setPivotAnimationEndTime(const std::chrono::steady_clock::time_point end_time) {
            pivot_animation_end_ns_.store(to_ns(end_time), std::memory_order_release);
            pivot_animation_active_.store(true);
        }

        void triggerSelectionFlash() {
            selection_flash_start_ns_.store(to_ns(std::chrono::steady_clock::now()), std::memory_order_release);
            selection_flash_active_.store(true);
            markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH);
        }

        void setOverlayAnimationActive(const bool active) { overlay_animation_active_.store(active); }

        [[nodiscard]] float getSelectionFlashIntensity() const {
            if (!selection_flash_active_.load())
                return 0.0f;
            const float t = std::chrono::duration<float>(
                                std::chrono::steady_clock::now() -
                                from_ns(selection_flash_start_ns_.load(std::memory_order_acquire)))
                                .count() /
                            SELECTION_FLASH_DURATION_SEC;
            if (t >= 1.0f)
                return 0.0f;
            return 1.0f - t * t; // Ease-out
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
            current_camera_id_ = cam_id;
            markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::PPISP);
        }
        int getCurrentCameraId() const { return current_camera_id_; }

        // FPS monitoring
        float getCurrentFPS() const { return framerate_controller_.getCurrentFPS(); }
        float getAverageFPS() const { return framerate_controller_.getAverageFPS(); }

        // Access to rendering engine (for initialization only)
        lfs::rendering::RenderingEngine* getRenderingEngine();

        // Camera frustum picking
        int pickCameraFrustum(const glm::vec2& mouse_pos);
        void setHoveredCameraId(int cam_id) { hovered_camera_id_ = cam_id; }
        int getHoveredCameraId() const { return hovered_camera_id_; }

        // Depth buffer access for tools (returns camera-space depth at pixel, or -1 if invalid)
        float getDepthAtPixel(int x, int y) const;
        const lfs::rendering::RenderResult& getCachedResult() const { return cached_result_; }
        glm::ivec2 getRenderedSize() const { return cached_result_size_; }

        // Screen positions output for brush tool
        void setOutputScreenPositions(bool enable) { output_screen_positions_ = enable; }
        bool getOutputScreenPositions() const { return output_screen_positions_; }
        std::shared_ptr<lfs::core::Tensor> getScreenPositions() const { return cached_result_.screen_positions; }

        // Brush selection on GPU - mouse_x/y in image coords (not window coords!)
        void brushSelect(float mouse_x, float mouse_y, float radius, lfs::core::Tensor& selection_out);

        // Apply crop filter to selection - filters out selections outside crop box/ellipsoid
        void applyCropFilter(lfs::core::Tensor& selection);

        void setBrushState(bool active, float x, float y, float radius, bool add_mode = true,
                           lfs::core::Tensor* selection_tensor = nullptr,
                           bool saturation_mode = false, float saturation_amount = 0.0f);
        void clearBrushState();
        [[nodiscard]] bool isBrushActive() const { return brush_active_; }
        void getBrushState(float& x, float& y, float& radius, bool& add_mode) const {
            x = brush_x_;
            y = brush_y_;
            radius = brush_radius_;
            add_mode = brush_add_mode_;
        }

        // Rectangle preview
        void setRectPreview(float x0, float y0, float x1, float y1, bool add_mode = true);
        void clearRectPreview();
        [[nodiscard]] bool isRectPreviewActive() const { return rect_preview_active_; }
        void getRectPreview(float& x0, float& y0, float& x1, float& y1, bool& add_mode) const {
            x0 = rect_x0_;
            y0 = rect_y0_;
            x1 = rect_x1_;
            y1 = rect_y1_;
            add_mode = rect_add_mode_;
        }

        // Polygon preview
        void setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed, bool add_mode = true);
        void clearPolygonPreview();
        [[nodiscard]] bool isPolygonPreviewActive() const { return polygon_preview_active_; }
        [[nodiscard]] const std::vector<std::pair<float, float>>& getPolygonPoints() const { return polygon_points_; }
        [[nodiscard]] bool isPolygonClosed() const { return polygon_closed_; }
        [[nodiscard]] bool isPolygonAddMode() const { return polygon_add_mode_; }

        // Lasso preview
        void setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode = true);
        void clearLassoPreview();
        [[nodiscard]] bool isLassoPreviewActive() const { return lasso_preview_active_; }
        [[nodiscard]] const std::vector<std::pair<float, float>>& getLassoPoints() const { return lasso_points_; }
        [[nodiscard]] bool isLassoAddMode() const { return lasso_add_mode_; }

        // Preview selection
        void setPreviewSelection(lfs::core::Tensor* preview, bool add_mode = true) {
            preview_selection_ = preview;
            brush_add_mode_ = add_mode;
            markDirty(DirtyFlag::SELECTION);
        }
        void clearPreviewSelection() {
            preview_selection_ = nullptr;
            markDirty(DirtyFlag::SELECTION);
        }

        // Selection mode for brush tool
        void setSelectionMode(lfs::rendering::SelectionMode mode) { selection_mode_ = mode; }
        [[nodiscard]] lfs::rendering::SelectionMode getSelectionMode() const { return selection_mode_; }
        [[nodiscard]] int getHoveredGaussianId() const { return hovered_gaussian_id_; }
        void adjustSaturation(float mouse_x, float mouse_y, float radius, float saturation_delta,
                              lfs::core::Tensor& sh0_tensor);

        // Sync selection group colors to GPU constant memory
        void syncSelectionGroupColor(int group_id, const glm::vec3& color);

        // Gizmo state for wireframe sync during manipulation
        void setCropboxGizmoState(bool active, const glm::vec3& min, const glm::vec3& max,
                                  const glm::mat4& world_transform) {
            cropbox_gizmo_active_ = active;
            if (active) {
                pending_cropbox_min_ = min;
                pending_cropbox_max_ = max;
                pending_cropbox_transform_ = world_transform;
            }
        }
        void setEllipsoidGizmoState(bool active, const glm::vec3& radii,
                                    const glm::mat4& world_transform) {
            ellipsoid_gizmo_active_ = active;
            if (active) {
                pending_ellipsoid_radii_ = radii;
                pending_ellipsoid_transform_ = world_transform;
            }
        }
        void setCropboxGizmoActive(bool active) { cropbox_gizmo_active_ = active; }
        void setEllipsoidGizmoActive(bool active) { ellipsoid_gizmo_active_ = active; }

    private:
        static int64_t to_ns(std::chrono::steady_clock::time_point tp) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
        }
        static std::chrono::steady_clock::time_point from_ns(int64_t ns) {
            return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
        }

        void doFullRender(const RenderContext& context, SceneManager* scene_manager, const lfs::core::SplatData* model);
        void setupEventHandlers();

        // Core components
        std::unique_ptr<lfs::rendering::RenderingEngine> engine_;
        std::vector<std::unique_ptr<RenderPass>> passes_;
        SplatRasterPass* splat_raster_pass_ = nullptr;
        OverlayPass* overlay_pass_ = nullptr;
        PointCloudPass* point_cloud_pass_ = nullptr;
        FramerateController framerate_controller_;

        // GT texture cache
        GTTextureCache gt_texture_cache_;

        // Cached render texture for reuse in split view
        unsigned int cached_render_texture_ = 0;
        std::atomic<bool> render_texture_valid_{false};

        // Granular dirty tracking
        std::atomic<uint32_t> dirty_mask_{DirtyFlag::ALL};

        std::atomic<bool> pivot_animation_active_{false};
        std::atomic<int64_t> pivot_animation_end_ns_{0};
        lfs::rendering::RenderResult cached_result_;

        // Selection flash animation
        mutable std::atomic<bool> selection_flash_active_{false};
        std::atomic<int64_t> selection_flash_start_ns_{0};
        static constexpr float SELECTION_FLASH_DURATION_SEC = 0.5f;

        std::atomic<bool> overlay_animation_active_{false};

        size_t last_model_ptr_ = 0;
        std::chrono::steady_clock::time_point last_training_render_;

        // Split view state
        mutable std::mutex split_info_mutex_;
        SplitViewInfo current_split_info_;

        int current_camera_id_ = -1;
        bool pre_gt_equirectangular_ = false;

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

        // Screen positions output flag
        bool output_screen_positions_ = false;

        // Brush state
        bool brush_active_ = false;
        float brush_x_ = 0.0f;
        float brush_y_ = 0.0f;
        float brush_radius_ = 0.0f;
        bool brush_add_mode_ = true;
        lfs::core::Tensor* brush_selection_tensor_ = nullptr;
        lfs::core::Tensor* preview_selection_ = nullptr;
        bool brush_saturation_mode_ = false;
        float brush_saturation_amount_ = 0.0f;
        lfs::rendering::SelectionMode selection_mode_ = lfs::rendering::SelectionMode::Centers;

        // Selection shape preview state (for rectangle, polygon, lasso)
        bool rect_preview_active_ = false;
        float rect_x0_ = 0.0f, rect_y0_ = 0.0f, rect_x1_ = 0.0f, rect_y1_ = 0.0f;
        bool rect_add_mode_ = true;

        bool polygon_preview_active_ = false;
        std::vector<std::pair<float, float>> polygon_points_;
        bool polygon_closed_ = false;
        bool polygon_add_mode_ = true;

        bool lasso_preview_active_ = false;
        std::vector<std::pair<float, float>> lasso_points_;
        bool lasso_add_mode_ = true;

        // Ring mode hover preview (gaussian ID extracted from packed depth+id atomicMin)
        int hovered_gaussian_id_ = -1;

        // Viewport state
        glm::ivec2 last_viewport_size_{0, 0}; // Last requested viewport size
        glm::ivec2 cached_result_size_{0, 0}; // Size at which cached_result_ was actually rendered

        std::optional<GTComparisonContext> gt_context_;
        int gt_context_camera_id_ = -1;

        // Gizmo state for wireframe sync
        bool cropbox_gizmo_active_ = false;
        bool ellipsoid_gizmo_active_ = false;
        glm::vec3 pending_cropbox_min_{0.0f};
        glm::vec3 pending_cropbox_max_{0.0f};
        glm::mat4 pending_cropbox_transform_{1.0f};
        glm::vec3 pending_ellipsoid_radii_{1.0f};
        glm::mat4 pending_ellipsoid_transform_{1.0f};
    };

} // namespace lfs::vis
