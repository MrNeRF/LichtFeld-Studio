/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "framerate_controller.hpp"
#include "internal/viewport.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "rendering/cuda_gl_interop.hpp"
#include "rendering/rendering.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace lfs::vis {
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis {

    enum class SplitViewMode {
        Disabled,
        PLYComparison,
        GTComparison
    };

    struct PPISPOverrides {
        // Exposure (Section 4.1)
        float exposure_offset = 0.0f; // EV stops (-3 to +3)

        // Vignetting (Section 4.2)
        bool vignette_enabled = true;
        float vignette_strength = 1.0f; // 0.0 to 2.0

        // Color Correction (Section 4.3) - 4 chromaticity control points
        // White point (neutral) - intuitive temperature/tint controls
        float wb_temperature = 0.0f; // -1.0 to +1.0 (cool to warm)
        float wb_tint = 0.0f;        // -1.0 to +1.0 (green to magenta)
        // RGB primary offsets - direct chromaticity manipulation
        float color_red_x = 0.0f;   // -0.5 to +0.5
        float color_red_y = 0.0f;   // -0.5 to +0.5
        float color_green_x = 0.0f; // -0.5 to +0.5
        float color_green_y = 0.0f; // -0.5 to +0.5
        float color_blue_x = 0.0f;  // -0.5 to +0.5
        float color_blue_y = 0.0f;  // -0.5 to +0.5

        // CRF (Section 4.4) - piecewise power curve per channel
        float gamma_multiplier = 1.0f; // 0.5 to 2.5 (overall gamma)
        float gamma_red = 0.0f;        // -0.5 to +0.5 (per-channel offset)
        float gamma_green = 0.0f;      // -0.5 to +0.5
        float gamma_blue = 0.0f;       // -0.5 to +0.5
        float crf_toe = 0.0f;          // -1.0 to +1.0 (shadow compression)
        float crf_shoulder = 0.0f;     // -1.0 to +1.0 (highlight roll-off)

        [[nodiscard]] bool isIdentity() const {
            return exposure_offset == 0.0f && vignette_enabled && vignette_strength == 1.0f &&
                   wb_temperature == 0.0f && wb_tint == 0.0f && color_red_x == 0.0f && color_red_y == 0.0f &&
                   color_green_x == 0.0f && color_green_y == 0.0f && color_blue_x == 0.0f && color_blue_y == 0.0f &&
                   gamma_multiplier == 1.0f && gamma_red == 0.0f && gamma_green == 0.0f && gamma_blue == 0.0f &&
                   crf_toe == 0.0f && crf_shoulder == 0.0f;
        }
    };

    struct RenderSettings {
        // Core rendering settings
        float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool mip_filter = false;
        int sh_degree = 3;
        float render_scale = 1.0f; // Viewer resolution scale (0.25-1.0), does not affect training

        // Crop box (data stored in scene graph CropBoxData, these are UI toggles only)
        bool show_crop_box = false;
        bool use_crop_box = false;
        // Ellipsoid (data stored in scene graph EllipsoidData, these are UI toggles only)
        bool show_ellipsoid = false;
        bool use_ellipsoid = false;
        bool desaturate_unselected = false;     // Desaturate unselected PLYs when one is selected
        bool desaturate_cropping = true;        // Desaturate outside crop box/ellipsoid instead of hiding
        bool crop_filter_for_selection = false; // Use crop box/ellipsoid as selection filter

        // Appearance correction (PPISP)
        bool apply_appearance_correction = false;
        enum class PPISPMode { MANUAL = 0,
                               AUTO = 1 };
        PPISPMode ppisp_mode = PPISPMode::AUTO;
        PPISPOverrides ppisp_overrides;

        // Background
        glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);

        // Coordinate axes
        bool show_coord_axes = false;
        float axes_size = 2.0f;
        std::array<bool, 3> axes_visibility = {true, true, true};

        // Grid
        bool show_grid = true;
        int grid_plane = 1;
        float grid_opacity = 0.5f;

        // Point cloud
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;

        // Ring mode (only active in splat mode)
        bool show_rings = false;
        float ring_width = 0.01f;
        bool show_center_markers = false;

        // Camera frustums
        bool show_camera_frustums = true; // Master toggle for camera frustum rendering
        float camera_frustum_scale = 0.25f;
        glm::vec3 train_camera_color = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 eval_camera_color = glm::vec3(1.0f, 0.0f, 0.0f);

        // Pivot point visualization
        bool show_pivot = false;

        // Split view
        SplitViewMode split_view_mode = SplitViewMode::Disabled;
        float split_position = 0.5f;
        size_t split_view_offset = 0;

        bool gut = false;
        bool equirectangular = false;
        bool orthographic = false;
        float ortho_scale = 100.0f; // Pixels per world unit (larger = more zoomed in)

        // Selection colors (RGB: committed=219,83,83 preview=0,222,76 center=0,154,187)
        glm::vec3 selection_color_committed{0.859f, 0.325f, 0.325f};
        glm::vec3 selection_color_preview{0.0f, 0.871f, 0.298f};
        glm::vec3 selection_color_center_marker{0.0f, 0.604f, 0.733f};

        // Depth clipping
        bool depth_clip_enabled = false;
        float depth_clip_far = 100.0f;

        bool mesh_wireframe = false;
        glm::vec3 mesh_wireframe_color{0.2f};
        float mesh_wireframe_width = 1.0f;
        glm::vec3 mesh_light_dir{0.3f, 1.0f, 0.5f};
        float mesh_light_intensity = 0.7f;
        float mesh_ambient = 0.4f;
        bool mesh_backface_culling = true;
        bool mesh_shadow_enabled = false;
        int mesh_shadow_resolution = 2048;

        // Depth filter (Selection tool only - separate from crop box)
        bool depth_filter_enabled = false;
        glm::vec3 depth_filter_min = glm::vec3(-50.0f, -10000.0f, 0.0f);
        glm::vec3 depth_filter_max = glm::vec3(50.0f, 10000.0f, 100.0f);
        lfs::geometry::EuclideanTransform depth_filter_transform;
    };

    struct SplitViewInfo {
        bool enabled = false;
        std::string left_name;
        std::string right_name;
    };

    struct ViewportRegion {
        float x, y, width, height;
    };

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

    struct GTComparisonContext {
        unsigned int gt_texture_id = 0;
        glm::ivec2 dimensions{0, 0};
        glm::ivec2 gpu_aligned_dims{0, 0};
        glm::vec2 render_texcoord_scale{1.0f, 1.0f};
        glm::vec2 gt_texcoord_scale{1.0f, 1.0f};
        bool gt_needs_flip = false;

        [[nodiscard]] bool valid() const { return gt_texture_id != 0 && dimensions.x > 0 && dimensions.y > 0; }
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

        [[nodiscard]] bool needsRender() const {
            if (pivot_animation_active_.load() &&
                std::chrono::steady_clock::now() < pivot_animation_end_time_) {
                return true;
            }
            pivot_animation_active_.store(false);

            // Selection flash: continuous rendering while active
            if (selection_flash_active_.load()) {
                const auto elapsed = std::chrono::steady_clock::now() - selection_flash_start_time_;
                if (std::chrono::duration<float>(elapsed).count() < SELECTION_FLASH_DURATION_SEC) {
                    needs_render_.store(true);
                    mesh_dirty_.store(true);
                    return true;
                }
                selection_flash_active_.store(false);
            }

            if (overlay_animation_active_.load())
                return true;
            return needs_render_.load();
        }

        void setPivotAnimationEndTime(const std::chrono::steady_clock::time_point end_time) {
            pivot_animation_end_time_ = end_time;
            pivot_animation_active_.store(true);
        }

        void triggerSelectionFlash() {
            selection_flash_start_time_ = std::chrono::steady_clock::now();
            selection_flash_active_.store(true);
            markDirty();
        }

        void setOverlayAnimationActive(const bool active) { overlay_animation_active_.store(active); }

        [[nodiscard]] float getSelectionFlashIntensity() const {
            if (!selection_flash_active_.load())
                return 0.0f;
            const float t = std::chrono::duration<float>(
                                std::chrono::steady_clock::now() - selection_flash_start_time_)
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
            markDirty();
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
            markDirty();
        }
        void clearPreviewSelection() {
            preview_selection_ = nullptr;
            markDirty();
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
        void doFullRender(const RenderContext& context, SceneManager* scene_manager, const lfs::core::SplatData* model);
        void renderOverlays(const RenderContext& context);
        void setupEventHandlers();
        void renderToTexture(const RenderContext& context, SceneManager* scene_manager, const lfs::core::SplatData* model);

        std::optional<lfs::rendering::SplitViewRequest> createSplitViewRequest(
            const RenderContext& context,
            SceneManager* scene_manager);

        // Core components
        std::unique_ptr<lfs::rendering::RenderingEngine> engine_;
        FramerateController framerate_controller_;

        // GT texture cache
        GTTextureCache gt_texture_cache_;

        // Cached render texture for reuse in split view
        unsigned int cached_render_texture_ = 0;
        bool render_texture_valid_ = false;

        // State tracking
        mutable std::atomic<bool> needs_render_{true};
        mutable std::atomic<bool> pivot_animation_active_{false};
        std::chrono::steady_clock::time_point pivot_animation_end_time_;
        lfs::rendering::RenderResult cached_result_;

        // Selection flash animation
        mutable std::atomic<bool> selection_flash_active_{false};
        std::chrono::steady_clock::time_point selection_flash_start_time_;
        static constexpr float SELECTION_FLASH_DURATION_SEC = 0.5f;

        mutable std::atomic<bool> overlay_animation_active_{false};

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

        // Ring mode hover preview (packed depth+id from atomicMin)
        unsigned long long hovered_depth_id_ = 0xFFFFFFFFFFFFFFFFULL;
        unsigned long long* d_hovered_depth_id_ = nullptr;
        int hovered_gaussian_id_ = -1; // Extracted from lower 32 bits

        // Cached filtered point cloud for cropbox preview (avoid CPU filtering every frame)
        mutable std::unique_ptr<lfs::core::PointCloud> cached_filtered_point_cloud_;
        mutable glm::mat4 cached_cropbox_transform_{1.0f};
        mutable glm::vec3 cached_cropbox_min_{0.0f};
        mutable glm::vec3 cached_cropbox_max_{0.0f};
        mutable bool cached_cropbox_inverse_ = false;
        mutable const lfs::core::PointCloud* cached_source_point_cloud_ = nullptr;

        // Viewport state
        glm::ivec2 last_viewport_size_{0, 0}; // Last requested viewport size
        glm::ivec2 cached_result_size_{0, 0}; // Size at which cached_result_ was actually rendered

        std::optional<GTComparisonContext> gt_context_;
        int gt_context_camera_id_ = -1;

        mutable std::atomic<bool> mesh_dirty_{false};

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
