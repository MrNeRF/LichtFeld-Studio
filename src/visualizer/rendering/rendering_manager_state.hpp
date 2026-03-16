/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include "render_pass.hpp"
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace lfs::vis {
    class GTTextureCache;
    class SceneManager;

    struct RectPreviewState {
        bool active = false;
        float x0 = 0.0f;
        float y0 = 0.0f;
        float x1 = 0.0f;
        float y1 = 0.0f;
        bool add_mode = true;
    };

    struct PolygonPreviewState {
        bool active = false;
        std::vector<std::pair<float, float>> points;
        std::vector<glm::vec3> world_points;
        bool closed = false;
        bool add_mode = true;
        bool world_space = false;
    };

    struct LassoPreviewState {
        bool active = false;
        std::vector<std::pair<float, float>> points;
        bool add_mode = true;
    };

    struct InteractionPreviewState {
        BrushState brush;
        RectPreviewState rect;
        PolygonPreviewState polygon;
        LassoPreviewState lasso;
        int hovered_gaussian_id = -1;

        void setBrush(bool active, float x, float y, float radius, bool add_mode,
                      lfs::core::Tensor* selection_tensor,
                      bool saturation_mode, float saturation_amount) {
            brush.active = active;
            brush.x = x;
            brush.y = y;
            brush.radius = radius;
            brush.add_mode = add_mode;
            brush.selection_tensor = selection_tensor;
            brush.saturation_mode = saturation_mode;
            brush.saturation_amount = saturation_amount;
        }

        void clearBrush() {
            brush.active = false;
            brush.x = 0.0f;
            brush.y = 0.0f;
            brush.radius = 0.0f;
            brush.selection_tensor = nullptr;
            brush.preview_selection = nullptr;
            brush.saturation_mode = false;
            brush.saturation_amount = 0.0f;
            hovered_gaussian_id = -1;
        }

        void setPreviewSelection(lfs::core::Tensor* preview, bool add_mode) {
            brush.preview_selection = preview;
            brush.add_mode = add_mode;
        }

        void clearPreviewSelection() {
            brush.preview_selection = nullptr;
        }

        void clearSelectionPreviews() {
            clearPreviewSelection();
            clearBrush();
            clearRect();
            clearPolygon();
            clearLasso();
        }

        void setRect(float x0, float y0, float x1, float y1, bool add_mode) {
            rect.active = true;
            rect.x0 = x0;
            rect.y0 = y0;
            rect.x1 = x1;
            rect.y1 = y1;
            rect.add_mode = add_mode;
        }

        void clearRect() {
            rect.active = false;
        }

        void setPolygon(const std::vector<std::pair<float, float>>& points, bool closed, bool add_mode) {
            polygon.active = true;
            polygon.points = points;
            polygon.world_points.clear();
            polygon.closed = closed;
            polygon.add_mode = add_mode;
            polygon.world_space = false;
        }

        void setPolygonWorldSpace(const std::vector<glm::vec3>& world_points, bool closed, bool add_mode) {
            polygon.active = true;
            polygon.points.clear();
            polygon.world_points = world_points;
            polygon.closed = closed;
            polygon.add_mode = add_mode;
            polygon.world_space = true;
        }

        void clearPolygon() {
            polygon.active = false;
            polygon.points.clear();
            polygon.world_points.clear();
            polygon.closed = false;
            polygon.world_space = false;
        }

        void setLasso(const std::vector<std::pair<float, float>>& points, bool add_mode) {
            lasso.active = true;
            lasso.points = points;
            lasso.add_mode = add_mode;
        }

        void clearLasso() {
            lasso.active = false;
            lasso.points.clear();
        }

        void setSelectionMode(lfs::rendering::SelectionMode mode) {
            brush.selection_mode = mode;
        }

        [[nodiscard]] lfs::rendering::SelectionMode getSelectionMode() const {
            return brush.selection_mode;
        }
    };

    struct RenderGizmoState {
        bool cropbox_active = false;
        bool ellipsoid_active = false;
        glm::vec3 cropbox_min{0.0f};
        glm::vec3 cropbox_max{0.0f};
        glm::mat4 cropbox_transform{1.0f};
        glm::vec3 ellipsoid_radii{1.0f};
        glm::mat4 ellipsoid_transform{1.0f};

        void setCropbox(bool active, const glm::vec3& min, const glm::vec3& max,
                        const glm::mat4& world_transform) {
            cropbox_active = active;
            if (active) {
                cropbox_min = min;
                cropbox_max = max;
                cropbox_transform = world_transform;
            }
        }

        void setEllipsoid(bool active, const glm::vec3& radii, const glm::mat4& world_transform) {
            ellipsoid_active = active;
            if (active) {
                ellipsoid_radii = radii;
                ellipsoid_transform = world_transform;
            }
        }

        [[nodiscard]] GizmoState makeFrameState() const {
            return GizmoState{
                .cropbox_active = cropbox_active,
                .cropbox_min = cropbox_min,
                .cropbox_max = cropbox_max,
                .cropbox_transform = cropbox_transform,
                .ellipsoid_active = ellipsoid_active,
                .ellipsoid_radii = ellipsoid_radii,
                .ellipsoid_transform = ellipsoid_transform,
            };
        }
    };

    struct ViewportArtifactState {
        CachedRenderMetadata metadata;
        std::optional<lfs::rendering::GpuFrame> gpu_frame;
        glm::ivec2 rendered_size{0};
        std::shared_ptr<lfs::core::Tensor> captured_image;
        uint64_t artifact_generation = 1;
        uint64_t captured_artifact_generation = 0;

        [[nodiscard]] bool hasGpuFrame() const {
            return gpu_frame && gpu_frame->valid();
        }

        [[nodiscard]] bool hasViewportOutput() const {
            return hasGpuFrame();
        }

        [[nodiscard]] bool hasOutputArtifacts() const {
            return (metadata.depth && metadata.depth->is_valid()) ||
                   (metadata.depth_right && metadata.depth_right->is_valid()) ||
                   hasGpuFrame() ||
                   rendered_size.x > 0 ||
                   rendered_size.y > 0;
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> getCapturedImageIfCurrent() const {
            if (captured_image && captured_artifact_generation == artifact_generation) {
                return captured_image;
            }
            return {};
        }

        void invalidateCapture() {
            captured_image.reset();
            captured_artifact_generation = 0;
            ++artifact_generation;
            if (artifact_generation == 0) {
                artifact_generation = 1;
            }
        }

        void clearViewportOutput() {
            metadata = {};
            gpu_frame.reset();
            rendered_size = {0, 0};
            invalidateCapture();
        }

        void updateFromFrameResources(const FrameResources& resources, bool viewport_output_updated) {
            metadata = resources.cached_metadata;
            gpu_frame = resources.cached_gpu_frame;
            rendered_size = resources.cached_result_size;
            if (viewport_output_updated) {
                invalidateCapture();
            }
        }

        void storeCapturedImage(std::shared_ptr<lfs::core::Tensor> image) {
            captured_image = std::move(image);
            captured_artifact_generation = artifact_generation;
        }
    };

    struct ViewportInteractionContextState {
        SceneManager* scene_manager = nullptr;
        lfs::rendering::ViewportData viewport_data{};
        ViewportRegion viewport_region{};
        bool pick_context_valid = false;

        void updatePickContext(const ViewportRegion* region,
                               const lfs::rendering::ViewportData& data) {
            if (region) {
                viewport_region = *region;
                viewport_data = data;
                pick_context_valid = true;
            } else {
                pick_context_valid = false;
            }
        }
    };

    struct CameraInteractionState {
        static constexpr auto PICK_THROTTLE_INTERVAL = std::chrono::milliseconds(50);

        int current_camera_id = -1;
        int hovered_camera_id = -1;
        std::chrono::steady_clock::time_point last_pick_time{};

        [[nodiscard]] bool shouldThrottlePick(std::chrono::steady_clock::time_point now) const {
            return now - last_pick_time < PICK_THROTTLE_INTERVAL;
        }

        void notePick(std::chrono::steady_clock::time_point now) {
            last_pick_time = now;
        }

        [[nodiscard]] bool updateHoveredCamera(int cam_id) {
            if (cam_id == hovered_camera_id) {
                return false;
            }
            hovered_camera_id = cam_id;
            return true;
        }
    };

    struct AnimationState {
        static constexpr float SELECTION_FLASH_DURATION_SEC = 0.5f;

        std::atomic<bool> pivot_active{false};
        std::atomic<int64_t> pivot_end_ns{0};
        std::atomic<bool> selection_flash_active{false};
        std::atomic<int64_t> selection_flash_start_ns{0};
        std::atomic<bool> overlay_active{false};

        static int64_t toNs(std::chrono::steady_clock::time_point tp) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
        }

        static std::chrono::steady_clock::time_point fromNs(int64_t ns) {
            return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
        }

        [[nodiscard]] DirtyMask pollDirtyState() {
            if (pivot_active.load() &&
                std::chrono::steady_clock::now() < fromNs(pivot_end_ns.load(std::memory_order_acquire))) {
                return DirtyFlag::CAMERA | DirtyFlag::OVERLAY;
            }
            pivot_active.store(false);

            if (selection_flash_active.load()) {
                const auto elapsed = std::chrono::steady_clock::now() -
                                     fromNs(selection_flash_start_ns.load(std::memory_order_acquire));
                if (std::chrono::duration<float>(elapsed).count() < SELECTION_FLASH_DURATION_SEC) {
                    return DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY;
                }
                selection_flash_active.store(false);
            }

            if (overlay_active.load()) {
                return DirtyFlag::OVERLAY;
            }

            return 0;
        }

        void setPivotAnimationEndTime(std::chrono::steady_clock::time_point end_time) {
            pivot_end_ns.store(toNs(end_time), std::memory_order_release);
            pivot_active.store(true);
        }

        [[nodiscard]] DirtyMask triggerSelectionFlash() {
            selection_flash_start_ns.store(toNs(std::chrono::steady_clock::now()), std::memory_order_release);
            selection_flash_active.store(true);
            return DirtyFlag::SPLATS | DirtyFlag::MESH;
        }

        void setOverlayAnimationActive(bool active) { overlay_active.store(active); }

        [[nodiscard]] float selectionFlashIntensity() const {
            if (!selection_flash_active.load())
                return 0.0f;
            const float t = std::chrono::duration<float>(
                                std::chrono::steady_clock::now() -
                                fromNs(selection_flash_start_ns.load(std::memory_order_acquire)))
                                .count() /
                            SELECTION_FLASH_DURATION_SEC;
            if (t >= 1.0f)
                return 0.0f;
            return 1.0f - t * t;
        }
    };

    struct GpuDepthReadbackState {
        GpuDepthReadbackState() = default;
        ~GpuDepthReadbackState();

        GpuDepthReadbackState(const GpuDepthReadbackState&) = delete;
        GpuDepthReadbackState& operator=(const GpuDepthReadbackState&) = delete;

        [[nodiscard]] float readLinearDepth(const lfs::rendering::GpuFrame& frame,
                                            int x,
                                            int y,
                                            int viewport_height) const;

    private:
        mutable unsigned int depth_readback_fbo = 0;
    };

    struct SplitViewState {
        struct GTToggleResult {
            bool enabled = false;
            std::optional<bool> restore_equirectangular;
        };

        mutable std::mutex info_mutex;
        SplitViewInfo current_info;
        std::optional<GTComparisonContext> gt_context;
        bool pre_gt_equirectangular = false;

        [[nodiscard]] bool hasValidGTContext() const {
            return gt_context && gt_context->valid();
        }

        [[nodiscard]] std::optional<glm::ivec2> gtContentDimensions() const {
            if (!hasValidGTContext()) {
                return std::nullopt;
            }
            return gt_context->dimensions;
        }

        void clear();
        void clearGTContext();
        [[nodiscard]] bool togglePLYComparison(RenderSettings& settings);
        [[nodiscard]] GTToggleResult toggleGTComparison(RenderSettings& settings);
        void handleSceneLoaded(RenderSettings& settings);
        void handleSceneCleared();
        [[nodiscard]] bool handlePLYRemoved(RenderSettings& settings, SceneManager* scene_manager);
        void advanceSplitOffset(RenderSettings& settings);
        [[nodiscard]] SplitViewInfo getInfo() const;
        void updateInfo(const FrameResources& resources);
        void prepareGTComparisonContext(SceneManager* scene_manager,
                                        const RenderSettings& settings,
                                        int current_camera_id,
                                        bool has_renderable_content,
                                        bool has_viewport_output,
                                        GTTextureCache& texture_cache,
                                        bool& request_viewport_prerender);
    };

    struct ViewportFrameLifecycleState {
        struct ResizeResult {
            DirtyMask dirty = 0;
            bool completed = false;
        };

        struct ModelChangeResult {
            bool changed = false;
            size_t previous_model_ptr = 0;
        };

        glm::ivec2 last_viewport_size{0, 0};
        size_t last_model_ptr = 0;
        std::chrono::steady_clock::time_point last_training_render;
        std::atomic<bool> resize_active{false};
        int resize_debounce = 0;
        bool resize_completed = false;

        [[nodiscard]] ResizeResult handleViewportResize(const glm::ivec2& current_size);
        [[nodiscard]] ModelChangeResult handleModelChange(size_t model_ptr, ViewportArtifactState& viewport_artifacts);
        [[nodiscard]] DirtyMask handleTrainingRefresh(bool is_training, float refresh_interval_sec);
        [[nodiscard]] DirtyMask requiredDirtyMask(bool has_viewport_output,
                                                  bool has_renderable_content,
                                                  SplitViewMode split_view_mode) const;
        [[nodiscard]] DirtyMask setViewportResizeActive(bool active);
        [[nodiscard]] bool isResizeDeferring() const {
            return resize_active.load(std::memory_order_relaxed) || resize_debounce > 0;
        }
        bool consumeResizeCompleted() { return std::exchange(resize_completed, false); }
        void noteResizeCompleted() { resize_completed = true; }
        void resetViewportSize() { last_viewport_size = glm::ivec2(0, 0); }
        void resetModelTracking() { last_model_ptr = 0; }
    };

} // namespace lfs::vis
