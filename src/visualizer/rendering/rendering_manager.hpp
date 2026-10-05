/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "render_target_id.hpp"

#include "view_render_state.hpp"
#include "view_source.hpp"

#include "camera_interaction_service.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/export.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"
#include "depth_window_state.hpp"
#include "dirty_flags.hpp"
#include "frame_demand.hpp"
#include "framerate_controller.hpp"
#include "internal/viewport.hpp"
#include "io/loader.hpp"
#include "render_animation_state.hpp"
#include "rendering/rendering.hpp"
#include "rendering/scene_temporal_resolve.hpp"
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "rendering/temporal_frame_tracker.hpp"
#include "rendering_types.hpp"
#include "spark_lod_controller.hpp"
#include "split_view_cpu_desc.hpp"
#include "split_view_service.hpp"
#include "stale_frame_guard.hpp"
#include "viewport_appearance_correction.hpp"
#include "viewport_artifact_service.hpp"
#include "viewport_frame_lifecycle_service.hpp"
#include "viewport_frame_result.hpp"
#include "viewport_interaction_context.hpp"
#include "viewport_overlay_service.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <expected>
#include <filesystem>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

class PythonIntegrationTest_ParkedGtPanelWindowRequestsAreAtomicallyRefused_Test;

namespace lfs::core {
    class Camera;
    class Scene;
    class SplatData;
    class Tensor;
} // namespace lfs::core

namespace lfs::io {
    class PipelinedImageLoader;
}

namespace lfs::core::events::ui {
    struct GridSettingsChanged;
    struct PointCloudModeChanged;
    struct RenderSettingsChanged;
} // namespace lfs::core::events::ui

namespace lfs::core::events::cmd {} // namespace lfs::core::events::cmd

namespace lfs::vis::op {
    struct DepthWindowModeSnapshot;
}

namespace lfs::vis {
    class GraphicsContext;
    class SceneRenderer;
    class PointSceneRenderer;

    class SceneManager;
    struct SceneRenderState;
    class TrainerManager;

    enum class SceneUpscalerPresetUpdate : std::uint8_t {
        UseRequested,
        RestoreRememberedForBackend,
    };

    class LFS_VIS_API RenderingManager {
    public:
        struct RenderContext {
            ViewId view;
            const Viewport& viewport;
            const RenderSettings& settings;
            glm::ivec2 logical_screen_size{0, 0};
            const ViewportRegion* viewport_region = nullptr;
            SceneManager* scene_manager = nullptr;
            GraphicsContext* graphics_context = nullptr;
            bool preparing_import = false;
            core::Uuid provisional_import_node;
        };

        explicit RenderingManager(ViewSource& views);
        ~RenderingManager();
        void setWakeCallback(std::function<void()> callback);
        [[nodiscard]] FrameDemandLedger& frameDemandLedger() { return frame_demand_ledger_; }
        [[nodiscard]] const FrameDemandLedger& frameDemandLedger() const { return frame_demand_ledger_; }

        // Initialize rendering resources
        void initialize();
        bool isInitialized() const { return initialized_; }
        void releaseSceneModelResources();
        void beginImportRenderCheck(uint64_t scene_generation);
        bool importUsesCombinedModel() const;
        std::optional<std::string> pollImportRenderCheck(const RenderContext& context, const std::function<void()>& prepare_viewport = {});
        void cancelImportRenderCheck();

        // Main render function
        ViewportFrameResult renderFrame(const RenderContext& context);
        void publishFrameToInterop(ViewId view, const ViewportFrameResult& frame);
        [[nodiscard]] std::expected<void, std::string> ensureVksplatTrainingSharedScratchReady(
            GraphicsContext& context,
            const lfs::core::SplatData& model,
            glm::ivec2 viewport_size);
        // Called by the viewer loop at idle cadence. Releases private viewer
        // scratch after a hysteresis window, or immediately under pressure.
        void releaseIdleVksplatScratch(bool training_active);

        enum class VksplatSelectionMaskShape : std::uint32_t {
            Brush = 0,
            Rectangle = 1,
            Polygon = 2,
            Ring = 3,
        };
        [[nodiscard]] std::expected<lfs::core::Tensor, std::string> buildVksplatSelectionMask(
            SceneManager& scene_manager,
            const lfs::rendering::FrameView& frame_view,
            bool equirectangular,
            VksplatSelectionMaskShape shape,
            const std::vector<glm::vec4>& primitives,
            const std::vector<glm::vec2>& polygon_vertices = {},
            std::uint32_t* picked_ring_id_out = nullptr);

        // Render preview image without touching the shared viewport presentation textures.
        std::shared_ptr<lfs::core::Tensor> renderPreviewImage(SceneManager* scene_manager,
                                                              const glm::mat3& camera_rotation,
                                                              const glm::vec3& camera_position,
                                                              float focal_length_mm,
                                                              int width, int height,
                                                              std::optional<glm::vec3> background_color_override = std::nullopt,
                                                              std::optional<bool> orthographic_override = std::nullopt,
                                                              std::optional<float> ortho_scale_override = std::nullopt);
        // Renders the scene from a dataset camera's pose and intrinsics, as the GT comparison
        // view does, at max(image, camera) size: the size selection projects that camera at.
        // Returns a CPU float [H,W,3] image without touching the viewport presentation, or
        // null for an equirectangular camera or a failed render. Viewer thread only.
        std::shared_ptr<lfs::core::Tensor> renderDatasetCameraImage(SceneManager* scene_manager,
                                                                    const lfs::core::Camera& camera);
        std::shared_ptr<lfs::core::Tensor> renderPreviewImageRgb8(SceneManager* scene_manager,
                                                                  const glm::mat3& camera_rotation,
                                                                  const glm::vec3& camera_position,
                                                                  float focal_length_mm,
                                                                  int width, int height,
                                                                  std::optional<glm::vec3> background_color_override = std::nullopt,
                                                                  std::optional<bool> orthographic_override = std::nullopt,
                                                                  std::optional<float> ortho_scale_override = std::nullopt,
                                                                  int reference_height = 0);

        // Image + per-pixel linear depth from the same viewport render. When
        // expected_depth is true, depth is alpha-weighted expected depth instead
        // of median depth. image is [H,W,3] and depth is [H,W], both CPU float32.
        struct PreviewRgbd {
            std::shared_ptr<lfs::core::Tensor> image;
            std::shared_ptr<lfs::core::Tensor> depth;
        };
        PreviewRgbd renderPreviewImageAndDepth(SceneManager* scene_manager,
                                               const glm::mat3& camera_rotation,
                                               const glm::vec3& camera_position,
                                               float focal_length_mm,
                                               int width, int height,
                                               bool expected_depth = false,
                                               std::optional<glm::vec3> background_color_override = std::nullopt);
        std::shared_ptr<lfs::core::Tensor> renderPreviewImageRgba8(SceneManager* scene_manager,
                                                                   const glm::mat3& camera_rotation,
                                                                   const glm::vec3& camera_position,
                                                                   float focal_length_mm,
                                                                   int width, int height,
                                                                   std::optional<bool> orthographic_override = std::nullopt,
                                                                   std::optional<float> ortho_scale_override = std::nullopt,
                                                                   int reference_height = 0);
        std::shared_ptr<lfs::core::Tensor> renderPreviewImage(const lfs::core::SplatData& model,
                                                              SceneRenderState scene_state,
                                                              const glm::mat3& camera_rotation,
                                                              const glm::vec3& camera_position,
                                                              float focal_length_mm,
                                                              int width, int height,
                                                              std::optional<glm::vec3> background_color_override = std::nullopt,
                                                              std::optional<bool> orthographic_override = std::nullopt,
                                                              std::optional<float> ortho_scale_override = std::nullopt);
        std::shared_ptr<lfs::core::Tensor> renderPreviewImageRgb8(const lfs::core::SplatData& model,
                                                                  SceneRenderState scene_state,
                                                                  const glm::mat3& camera_rotation,
                                                                  const glm::vec3& camera_position,
                                                                  float focal_length_mm,
                                                                  int width, int height,
                                                                  std::optional<glm::vec3> background_color_override = std::nullopt,
                                                                  std::optional<bool> orthographic_override = std::nullopt,
                                                                  std::optional<float> ortho_scale_override = std::nullopt);
        std::shared_ptr<lfs::core::Tensor> renderPreviewImageRgba8(const lfs::core::SplatData& model,
                                                                   SceneRenderState scene_state,
                                                                   const glm::mat3& camera_rotation,
                                                                   const glm::vec3& camera_position,
                                                                   float focal_length_mm,
                                                                   int width, int height,
                                                                   std::optional<bool> orthographic_override = std::nullopt,
                                                                   std::optional<float> ortho_scale_override = std::nullopt);
        void releasePreviewImageResources();

        // One-shot export: (tiled) preview render followed by the streamed GPU
        // post-process (PPISP correction and, for EnvironmentComposite, HDRI
        // background compositing). Returns the final CPU u8 HWC image. Must run
        // on the viewer thread.
        struct ExportImageRequest {
            glm::mat3 rotation{1.0f};
            glm::vec3 translation{0.0f};
            float focal_length_mm = 0.0f;
            int width = 0;
            int height = 0;
            // Positive for viewport exports: ortho_scale_override is the source
            // viewport scale. Zero keeps native-resolution rasterization and scale.
            int reference_height = 0;
            std::optional<bool> orthographic_override;
            std::optional<float> ortho_scale_override;
            ExportPostProcessMode mode = ExportPostProcessMode::Opaque;
        };
        [[nodiscard]] std::expected<lfs::core::Tensor, std::string> renderExportImage(
            SceneManager* scene_manager, const ExportImageRequest& request);

        [[nodiscard]] lfs::io::SplatTensorAllocator makeSplatTensorAllocator() const;

        void markDirty(DirtyMask flags, FrameReason reason, std::string detail = {});
        void markViewDirty(ViewId view, DirtyMask flags, FrameReason reason, std::string detail = {});
        void markCameraPoseChanged(ViewId view);
        [[nodiscard]] ViewMask viewMask(ViewId view) const;
        [[nodiscard]] ViewMask visibleViewMask() const;
        [[nodiscard]] std::vector<ViewId> ledgerViews() const {
            std::lock_guard lock(views_mutex_);
            return ledger_views_;
        }
        // Marks a discontinuous camera jump. Unlike interactive camera motion,
        // the next successfully published temporal frame must not reproject
        // history across this boundary.
        void markCameraCut(ViewId view);

        [[nodiscard]] bool pollDirtyState();
        [[nodiscard]] DirtyMask pendingDirtyMask() const;
        // The training preview refreshes on its own cadence, not only when an
        // unrelated redraw happens to notice it is due.
        void pollTrainingRefresh(bool is_training, int current_iteration);
        void requestViewportResize(ViewId view, glm::ivec2 size);
        [[nodiscard]] std::uint64_t viewInputFingerprint(const Viewport& viewport,
                                                         const SceneManager* scene_manager, ViewId view) const;
        [[nodiscard]] double secondsUntilTrainingRefresh() const;
        // Seconds until an over-budget navigation render may run (camera at
        // rest); +inf when no settle is pending.
        [[nodiscard]] double secondsUntilCameraSettle() const;
        // Re-arms a parked passive training refresh once its render can claim the arena.
        void pollParkedArenaRetry();
        [[nodiscard]] bool hasParkedArenaRetry() const {
            std::lock_guard lock(views_mutex_);
            for (const auto& [id, view] : view_states_)
                if (view->parked_arena_retry_ != 0)
                    return true;
            return false;
        }
        void retainVksplatScratch();
        [[nodiscard]] double secondsUntilVksplatScratchRelease() const;

        void setPivotAnimationEndTime(ViewId view, const std::chrono::steady_clock::time_point end_time) {
            viewState(view).animation_state_.setPivotAnimationEndTime(end_time);
        }

        void triggerSelectionFlash() {
            markDirty(this->state().animation_state_.triggerSelectionFlash(), lfs::vis::FrameReason::Selection);
        }

        void setOverlayAnimationActive(const bool active) {
            this->state().animation_state_.setOverlayAnimationActive(active);
        }

        // Settings management
        void updateSettings(const RenderSettings& settings);
        void updateSettings(
            const RenderSettings& settings,
            DirtyMask dirty_flags,
            SceneUpscalerPresetUpdate preset_update = SceneUpscalerPresetUpdate::UseRequested);
        RenderSettings getSettings() const;
        RenderSettings settingsForView(ViewId view) const;
        void editViewSettings(ViewId view, const std::function<void(ViewSettings&)>& edit);
        [[nodiscard]] ViewId activeViewId() const { return view_source_.activeView(); }
        // The presentation pass reports its actual runtime choice after pipeline
        // preparation. Rendering uses this feedback on the next frame so a failed
        // reconstruction pipeline never receives a reduced-resolution image.
        void reportSceneUpscalerRuntimeSelection(ViewId view, SceneUpscalerSelection selection);
        [[nodiscard]] SceneUpscalerSelection sceneUpscalerRuntimeSelection(ViewId view = kNoView) const;
        [[nodiscard]] bool sceneUpscalerModeUnsupported(ViewId view) const;

        // Entering computes ortho_scale so the view at the pivot matches the current
        // lens. Leaving ortho keeps the focal length the user set.
        void setOrthographic(bool enabled, float viewport_height, float distance_to_pivot);

        float getFovDegrees() const;
        float getFocalLengthMm() const;
        void setFocalLength(float focal_mm);

        void advanceSplitOffset();
        SplitViewInfo getSplitViewInfo() const;
        [[nodiscard]] std::optional<SplitViewInfo> getSplitViewInfoIfChanged(std::uint64_t& generation) const;
        [[nodiscard]] bool isSplitViewActive() const;
        [[nodiscard]] bool isGTComparisonActive() const;
        [[nodiscard]] bool isPLYComparisonActive() const;
        [[nodiscard]] bool depthWindowDragPreview(ViewId view = kNoView) const;
        void beginDepthWindowDrag(ViewId view, uint64_t& out_drag_token);
        void endDepthWindowDrag(ViewId view, uint64_t drag_token);
        void beginDepthWindowPreview(ViewId view);
        void endDepthWindowPreview(ViewId view);
        [[nodiscard]] GTComparisonMode getGTComparisonMode() const;
        [[nodiscard]] SplitViewMode getSplitViewMode() const;
        void restoreSplitViewMode(SplitViewMode mode);
        [[nodiscard]] float getSplitPosition() const;
        [[nodiscard]] std::optional<float> getSplitDividerScreenX(ViewId view, const glm::vec2& viewport_pos,
                                                                  const glm::vec2& viewport_size) const;
        [[nodiscard]] DepthWindowState getDepthWindow() const;
        void setDepthWindow(const DepthWindowState& state);
        bool applyDepthWindowIfEpoch(ViewId view, const DepthWindowState& state,
                                     uint64_t expected_epoch,
                                     uint64_t drag_token);
        bool restorePinnedDepthWindow(ViewId view, const DepthWindowState& state,
                                      uint64_t expected_epoch,
                                      uint64_t drag_token);
        bool commitDepthWindowIfEpoch(ViewId view, const DepthWindowState& state,
                                      uint64_t expected_epoch,
                                      uint64_t drag_token,
                                      op::DepthWindowModeSnapshot& out_snapshot);
        [[nodiscard]] std::unique_lock<std::mutex> acquireDepthWindowTransitionLock(ViewId view) {
            return std::unique_lock<std::mutex>(viewState(view).depth_window_transition_mutex_);
        }
        [[nodiscard]] op::DepthWindowModeSnapshot depthWindowSnapshot(ViewId view) const;
        [[nodiscard]] op::DepthWindowModeSnapshot
        depthWindowBaselineSnapshotForDrag(ViewId view, uint64_t drag_token) const;
        void restoreDepthWindowStateFromProject();
        bool restoreDepthWindowSnapshotIfEpoch(const op::DepthWindowModeSnapshot& snapshot,
                                               uint64_t expected_epoch);
        [[nodiscard]] uint64_t depthWindowProjectionGeneration() const;

        struct ViewerPanelInfo {
            SplitViewPanelId panel = SplitViewPanelId::Left;
            const Viewport* viewport = nullptr;
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;
            int render_width = 0;
            int render_height = 0;

            [[nodiscard]] bool valid() const {
                return viewport != nullptr &&
                       width > 0.0f &&
                       height > 0.0f &&
                       render_width > 0 &&
                       render_height > 0;
            }
        };
        struct MutableViewerPanelInfo {
            SplitViewPanelId panel = SplitViewPanelId::Left;
            Viewport* viewport = nullptr;
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;
            int render_width = 0;
            int render_height = 0;

            [[nodiscard]] bool valid() const {
                return viewport != nullptr &&
                       width > 0.0f &&
                       height > 0.0f &&
                       render_width > 0 &&
                       render_height > 0;
            }
        };
        [[nodiscard]] std::optional<MutableViewerPanelInfo> resolveViewerPanel(
            ViewId view, Viewport& viewport,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size,
            std::optional<glm::vec2> screen_point = std::nullopt,
            std::optional<SplitViewPanelId> panel_override = std::nullopt);
        [[nodiscard]] std::optional<ViewerPanelInfo> resolveViewerPanel(
            ViewId view, const Viewport& viewport,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size,
            std::optional<glm::vec2> screen_point = std::nullopt,
            std::optional<SplitViewPanelId> panel_override = std::nullopt) const;

        struct ContentBounds {
            float x, y, width, height;
            bool letterboxed = false;
        };
        ContentBounds getContentBounds(ViewId view, const glm::ivec2& viewport_size) const;

        struct GTSelectionContext {
            GTRenderCamera camera;
            glm::ivec2 size{0, 0};
        };
        [[nodiscard]] std::optional<GTSelectionContext> gtComparisonSelectionContext(ViewId view = kNoView) const;

        // Current camera tracking for GT comparison
        void setCurrentCameraId(int cam_id) {
            const bool changed = camera_interaction_service_.currentCameraId() != cam_id;
            camera_interaction_service_.setCurrentCameraId(cam_id);
            if (changed) {
                invalidateCameraMetricsRequests(true);
            }
            markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::PPISP, lfs::vis::FrameReason::SettingsChange);
        }
        int getCurrentCameraId() const { return camera_interaction_service_.currentCameraId(); }
        int getHoveredCameraId() const { return camera_interaction_service_.hoveredCameraId(); }

        struct CameraMetricsOverlayState {
            int camera_id = -1;
            int iteration = -1;
            float psnr = 0.0f;
            std::optional<float> ssim;
            bool used_mask = false;
        };

        void clearLatestCameraMetrics();

        // FPS monitoring (scene renders vs. swapchain-presented GUI frames)
        [[nodiscard]] std::optional<lfs::rendering::ViewerBackend> activeViewerBackend() const {
            return this->state().viewport_artifact_service_.viewerBackend();
        }
        FrameRates getFrameRates() const { return frame_rates_.sample(); }
        float getAverageFPS() const { return getFrameRates().view; }
        float getPresentedAverageFPS() const { return getFrameRates().ui; }
        void sampleFrameRates(const FramePlan& plan) {
            gui_frame_rates_ = getFrameRates();
            auto activity = plan.reasons;
            activity.reset(static_cast<std::size_t>(FrameReason::FpsIdle));
            fps_idle_frame_ = activity.none();
        }
        [[nodiscard]] std::uint32_t temporalConvergenceRemaining() const {
            return this->state().temporal_convergence_.remaining();
        }
        // Measurement only — does not affect scene render pacing/limiting.
        FrameRates guiFrameRates() const { return gui_frame_rates_; }
        bool isFpsIdleFrame() const { return fps_idle_frame_; }
        void countPresentedFrame(const FramePlan& plan) {
            frame_rates_.countPresented(plan);
            frame_demand_ledger_.countPresented(plan);
        }
        void countViewRendered(const FramePlan& plan) {
            frame_rates_.countView();
            frame_demand_ledger_.countViewRendered(plan.render_views, plan);
        }
        std::optional<FrameClock::time_point> fpsIdleDeadline() const { return frame_rates_.idleDeadline(); }
        void refreshIdleFps(const FrameClock::time_point now) {
            if (frame_rates_.idleDue(now)) {
                frame_demand_ledger_.request({.reason = FrameReason::FpsIdle,
                                              .scope = FrameScope::Gui,
                                              .views = 0,
                                              .detail = "fps_window_expired"});
            }
        }

        // Access to the auxiliary rendering engine used by point-cloud, mesh, and readback paths.
        lfs::rendering::RenderingEngine* getRenderingEngine();
        [[nodiscard]] lfs::rendering::ScreenOverlayRenderer* getScreenOverlayRenderer() {
            return &this->state().screen_overlay_renderer_;
        }

        // Camera frustum picking
        int pickCameraFrustum(ViewId view, const glm::vec2& mouse_pos);

        // Depth access for tools (returns camera-space depth at pixel, or -1 if invalid).
        float getDepthAtPixel(ViewId view, int x, int y, std::optional<SplitViewPanelId> panel = std::nullopt) const;
        struct ExpectedDepthSampleRequest {
            ViewId view = kNoView;
            SceneManager* scene_manager = nullptr;
            const Viewport* viewport = nullptr;
            glm::ivec2 render_size{0, 0};
            glm::ivec2 pixel{0, 0};
            float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
            bool orthographic = false;
            float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
            std::optional<SplitViewPanelId> panel;
        };
        // Renders a fresh expected-depth preview for precise picking on sparse or low-opacity splats.
        float renderExpectedDepthAtPixel(const ExpectedDepthSampleRequest& request);
        glm::ivec2 getRenderedSize() const { return this->state().viewport_artifact_service_.renderedSize(); }
        std::shared_ptr<lfs::core::Tensor> getViewportImageIfAvailable() const;
        std::shared_ptr<lfs::core::Tensor> captureViewportImage();
        [[nodiscard]] static std::shared_ptr<lfs::core::Tensor> composeSplitViewCpu(
            const SplitViewCpuDesc& params, const glm::ivec2& output_size);

        // Where the 3D viewport sat inside the window framebuffer on the last frame,
        // top-left origin. Lets callers crop a full-window readback down to the viewport
        // when no render path published an offscreen image to capture.

        [[nodiscard]] FramebufferViewportRect framebufferViewportRect() const {
            return this->state().framebuffer_viewport_rect_;
        }
        [[nodiscard]] uint64_t getViewportProjectionGeneration() const {
            return this->state().viewport_projection_generation_;
        }

        void setCursorPreviewState(bool active, float x, float y, float radius, bool add_mode = true,
                                   lfs::core::Tensor* selection_tensor = nullptr,
                                   bool saturation_mode = false, float saturation_amount = 0.0f,
                                   std::optional<SplitViewPanelId> panel = std::nullopt,
                                   int focused_gaussian_id = -1, bool request_render = true);
        void clearCursorPreviewState();
        [[nodiscard]] bool isCursorPreviewActive() const { return this->state().viewport_overlay_service_.isCursorPreviewActive(); }
        [[nodiscard]] std::optional<SplitViewPanelId> getCursorPreviewPanel() const {
            return this->state().viewport_overlay_service_.cursorPreview().panel;
        }
        void getCursorPreviewState(float& x, float& y, float& radius, bool& add_mode) const {
            const auto& cursor = this->state().viewport_overlay_service_.cursorPreview();
            x = cursor.x;
            y = cursor.y;
            radius = cursor.radius;
            add_mode = cursor.add_mode;
        }

        // Rectangle preview
        void setRectPreview(float x0, float y0, float x1, float y1, bool add_mode = true,
                            std::optional<SplitViewPanelId> panel = std::nullopt,
                            bool track_cursor = false);
        void clearRectPreview();
        [[nodiscard]] bool isRectPreviewActive() const { return this->state().viewport_overlay_service_.isRectPreviewActive(); }
        void getRectPreview(float& x0, float& y0, float& x1, float& y1, bool& add_mode) const {
            x0 = this->state().viewport_overlay_service_.rectX0();
            y0 = this->state().viewport_overlay_service_.rectY0();
            x1 = this->state().viewport_overlay_service_.rectX1();
            y1 = this->state().viewport_overlay_service_.rectY1();
            add_mode = this->state().viewport_overlay_service_.rectAddMode();
        }
        [[nodiscard]] bool rectPreviewTracksCursor() const {
            return this->state().viewport_overlay_service_.rectTracksCursor();
        }

        // Polygon preview (render-space points, same coordinate system as screen_positions output)
        void setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed,
                               bool add_mode = true, std::optional<SplitViewPanelId> panel = std::nullopt);
        // Interactive polygon preview in world-space coordinates.
        void setPolygonPreviewWorldSpace(const std::vector<glm::vec3>& world_points, bool closed,
                                         bool add_mode = true,
                                         std::optional<SplitViewPanelId> panel = std::nullopt);
        void clearPolygonPreview();
        [[nodiscard]] bool isPolygonPreviewActive() const { return this->state().viewport_overlay_service_.isPolygonPreviewActive(); }
        [[nodiscard]] const std::vector<std::pair<float, float>>& getPolygonPoints() const {
            return this->state().viewport_overlay_service_.polygonPoints();
        }
        [[nodiscard]] const std::vector<glm::vec3>& getPolygonWorldPoints() const {
            return this->state().viewport_overlay_service_.polygonWorldPoints();
        }
        [[nodiscard]] bool isPolygonClosed() const { return this->state().viewport_overlay_service_.polygonClosed(); }
        [[nodiscard]] bool isPolygonAddMode() const { return this->state().viewport_overlay_service_.polygonAddMode(); }
        [[nodiscard]] bool isPolygonPreviewWorldSpace() const {
            return this->state().viewport_overlay_service_.polygonWorldSpace();
        }

        // Lasso preview
        void setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode = true,
                             std::optional<SplitViewPanelId> panel = std::nullopt,
                             bool track_cursor = false);
        void clearLassoPreview();
        [[nodiscard]] bool isLassoPreviewActive() const { return this->state().viewport_overlay_service_.isLassoPreviewActive(); }
        [[nodiscard]] const std::vector<std::pair<float, float>>& getLassoPoints() const {
            return this->state().viewport_overlay_service_.lassoPoints();
        }
        [[nodiscard]] bool isLassoAddMode() const { return this->state().viewport_overlay_service_.lassoAddMode(); }
        [[nodiscard]] bool lassoPreviewTracksCursor() const {
            return this->state().viewport_overlay_service_.lassoTracksCursor();
        }

        // Preview selection
        void setPreviewSelection(lfs::core::Tensor* preview, bool add_mode = true) {
            this->state().viewport_overlay_service_.setPreviewSelection(preview, add_mode);
            markDirty(DirtyFlag::SELECTION, lfs::vis::FrameReason::Selection);
        }
        void clearPreviewSelection() {
            this->state().viewport_overlay_service_.clearPreviewSelection();
            markDirty(DirtyFlag::SELECTION, lfs::vis::FrameReason::Selection);
        }
        void clearSelectionPreviews();

        // Selection preview mode for viewport interaction overlays
        void setSelectionPreviewMode(SelectionPreviewMode mode) {
            this->state().viewport_overlay_service_.setSelectionPreviewMode(mode);
        }
        [[nodiscard]] SelectionPreviewMode getSelectionPreviewMode() const {
            return this->state().viewport_overlay_service_.selectionPreviewMode();
        }
        [[nodiscard]] int getHoveredGaussianId() const { return this->state().viewport_overlay_service_.hoveredGaussianId(); }

        void setCropboxGizmoState(bool active, const glm::vec3& min, const glm::vec3& max,
                                  const glm::mat4& transform, bool affects_render, int parent_node_index) {
            gizmo_state_.cropbox_active = active;
            gizmo_state_.cropbox_min = min;
            gizmo_state_.cropbox_max = max;
            gizmo_state_.cropbox_transform = transform;
            gizmo_state_.cropbox_affects_render = affects_render;
            gizmo_state_.cropbox_parent_node_index = parent_node_index;
        }
        void setEllipsoidGizmoState(bool active, const glm::vec3& radii,
                                    const glm::mat4& transform, bool affects_render, int parent_node_index) {
            gizmo_state_.ellipsoid_active = active;
            gizmo_state_.ellipsoid_radii = radii;
            gizmo_state_.ellipsoid_transform = transform;
            gizmo_state_.ellipsoid_affects_render = affects_render;
            gizmo_state_.ellipsoid_parent_node_index = parent_node_index;
        }
        void setCropboxGizmoActive(bool active) { gizmo_state_.cropbox_active = active; }
        void setEllipsoidGizmoActive(bool active) { gizmo_state_.ellipsoid_active = active; }
        // Return whether the drawn state changed, so callers redraw only then.
        bool setNodeBoxGizmoState(bool active, const glm::mat4& transform,
                                  const glm::mat4& falloff_transform, bool has_falloff) {
            const bool changed = gizmo_state_.node_box_active != active ||
                                 (active && (gizmo_state_.node_box_transform != transform ||
                                             gizmo_state_.node_box_falloff_transform != falloff_transform ||
                                             gizmo_state_.node_box_has_falloff != has_falloff));
            gizmo_state_.node_box_active = active;
            gizmo_state_.node_box_transform = transform;
            gizmo_state_.node_box_falloff_transform = falloff_transform;
            gizmo_state_.node_box_has_falloff = has_falloff;
            return changed;
        }
        bool setNodeEllipsoidGizmoState(bool active, const glm::mat4& transform,
                                        const glm::mat4& falloff_transform, bool has_falloff) {
            const bool changed = gizmo_state_.node_ellipsoid_active != active ||
                                 (active && (gizmo_state_.node_ellipsoid_transform != transform ||
                                             gizmo_state_.node_ellipsoid_falloff_transform != falloff_transform ||
                                             gizmo_state_.node_ellipsoid_has_falloff != has_falloff));
            gizmo_state_.node_ellipsoid_active = active;
            gizmo_state_.node_ellipsoid_transform = transform;
            gizmo_state_.node_ellipsoid_falloff_transform = falloff_transform;
            gizmo_state_.node_ellipsoid_has_falloff = has_falloff;
            return changed;
        }
        [[nodiscard]] GizmoState getGizmoState() const { return gizmo_state_; }

        void setViewportResizeActive(
            bool active,
            ViewportResizeRenderPolicy render_policy = ViewportResizeRenderPolicy::InteractivePreview);
        [[nodiscard]] bool isViewportResizeDeferring() const {
            std::lock_guard lock(views_mutex_);
            for (const auto& [id, view] : view_states_)
                if (view->frame_lifecycle_service_.isResizeDeferring())
                    return true;
            return false;
        }

        void clearViewportSceneImage();
        void shutdownViewportInterop(GraphicsContext* context = nullptr);
        [[nodiscard]] bool hasPendingViewportResizeSettle() const {
            std::lock_guard lock(views_mutex_);
            for (const auto& [id, view] : view_states_)
                if (view->frame_lifecycle_service_.hasPendingResizeSettle())
                    return true;
            return false;
        }
        [[nodiscard]] bool viewportResizeSettleReady() const {
            std::lock_guard lock(views_mutex_);
            for (const auto& [id, view] : view_states_)
                if (view->frame_lifecycle_service_.resizeSettleReady())
                    return true;
            return false;
        }
        [[nodiscard]] double secondsUntilViewportResizeSettleReady() const {
            std::lock_guard lock(views_mutex_);
            double wait = 1.0;
            for (const auto& [id, view] : view_states_)
                if (view->frame_lifecycle_service_.hasPendingResizeSettle())
                    wait = std::min(wait, view->frame_lifecycle_service_.secondsUntilResizeSettleReady());
            return wait;
        }
        // LOD management
        void setLodAvailable(bool available);
        void setLodEnabled(bool enabled);
        [[nodiscard]] SparkLodController::Stats getLodStats() const;

        ViewRenderState& viewState(ViewId view) const;
        void retainVisibleViews(const std::vector<ViewId>& views);
        void dropViewStates();
        bool depthWindowSnapshotCurrent(const op::DepthWindowModeSnapshot& snapshot) const;
        bool hasViewState(ViewId view) const {
            std::lock_guard lock(views_mutex_);
            return view_states_.contains(view);
        }

    private:
        ViewRenderState& state() const;
        bool releaseViewTargets(ViewRenderState& view);
        mutable std::recursive_mutex views_mutex_;
        mutable std::unordered_map<ViewId, std::unique_ptr<ViewRenderState>> view_states_;
        std::vector<std::unique_ptr<ViewRenderState>> retired_view_states_;
        std::uint64_t screen_epoch_ = 0;
        std::uint64_t view_lifetime_epoch_ = 0;
        mutable std::unordered_map<ViewId, std::pair<uint64_t, uint64_t>> depth_window_epochs_;
        enum class PreviewImageReadback {
            FloatRgb,
            UInt8Rgb,
            UInt8Rgba,
        };

        struct PreviewImageReadbackConfig {
            lfs::core::DataType dtype = lfs::core::DataType::Float32;
            int channels = 3;
            std::optional<bool> transparent_background_override;
        };

        [[nodiscard]] static PreviewImageReadbackConfig previewImageReadbackConfig(
            PreviewImageReadback readback,
            bool has_background_color_override);
        void clearViewportImageState(ViewRenderState& view, glm::ivec2 size = {0, 0},
                                     bool flip_y = false,
                                     glm::ivec2 alloc_size = {0, 0});
        [[nodiscard]] float exportRasterizationScale(int target_height, int reference_height) const;
        [[nodiscard]] std::optional<float> exportOrthoScale(std::optional<float> scale, int target_height, int reference_height) const;

        std::shared_ptr<lfs::core::Tensor> renderPreviewImageWithState(
            SceneManager* scene_manager,
            const lfs::core::SplatData& model,
            SceneRenderState scene_state,
            const glm::mat3& camera_rotation,
            const glm::vec3& camera_position,
            float focal_length_mm,
            int width,
            int height,
            bool render_lock_held,
            std::optional<lfs::rendering::CameraIntrinsics> intrinsics_override,
            std::optional<bool> orthographic_override,
            std::optional<float> ortho_scale_override,
            std::optional<glm::vec3> background_color_override,
            PreviewImageReadback readback,
            float rasterization_scale = 1.0f);
        [[nodiscard]] std::expected<void, std::string> renderPreviewImageToPreviewSlotWithState(
            const RenderSettings& settings,
            SceneManager* scene_manager,
            const lfs::core::SplatData& model,
            SceneRenderState scene_state,
            const glm::mat3& camera_rotation,
            const glm::vec3& camera_position,
            float focal_length_mm,
            int width,
            int height,
            bool render_lock_held,
            std::optional<lfs::rendering::CameraIntrinsics> intrinsics_override,
            glm::ivec2 subregion_origin,
            glm::ivec2 subregion_full_size,
            std::optional<bool> orthographic_override,
            std::optional<float> ortho_scale_override,
            std::optional<glm::vec3> background_color_override,
            std::optional<bool> transparent_background_override,
            float rasterization_scale = 1.0f,
            bool deterministic_export = false);
        [[nodiscard]] std::expected<void, std::string> renderDepthCaptureToPreviewSlotWithState(
            const RenderSettings& settings,
            SceneManager* scene_manager,
            const lfs::core::SplatData& model,
            SceneRenderState scene_state,
            const glm::mat3& camera_rotation,
            const glm::vec3& camera_position,
            float focal_length_mm,
            int width,
            int height,
            bool render_lock_held,
            bool expected_depth,
            std::optional<glm::vec3> background_color_override,
            std::optional<bool> orthographic_override,
            std::optional<float> ortho_scale_override);
        std::shared_ptr<lfs::core::Tensor> renderPreviewImageTiledWithState(
            SceneManager* scene_manager,
            const lfs::core::SplatData& model,
            SceneRenderState scene_state,
            const glm::mat3& camera_rotation,
            const glm::vec3& camera_position,
            float focal_length_mm,
            int width,
            int height,
            bool render_lock_held,
            std::optional<glm::vec3> background_color_override,
            std::optional<bool> orthographic_override,
            std::optional<float> ortho_scale_override,
            PreviewImageReadback readback,
            float rasterization_scale = 1.0f);

        struct CameraMetricsJobRequest {
            uint64_t generation = 0;
            TrainerManager* trainer_manager = nullptr;
            int camera_id = -1;
            int iteration = -1;
            RenderSettings settings{};
        };

        struct GTComparisonImageJobRequest {
            uint64_t generation = 0;
            int camera_uid = -1;
            GTComparisonMode mode = GTComparisonMode::RGB;
            std::filesystem::path image_path;
            int preview_max_dimension = 0;
            glm::ivec2 image_size{0, 0};
            bool undistort_requested = false;
            lfs::core::UndistortParams undistort_params{};
            lfs::rendering::DepthVisualizationMode depth_visualization_mode =
                lfs::rendering::DepthVisualizationMode::Palette;
            glm::vec3 background_color{0.0f};
            std::shared_ptr<lfs::core::Camera> camera;
            std::chrono::steady_clock::time_point queued_at{};
        };

        enum class GTComparisonImageStatus {
            Loading,
            Ready,
            Failed,
        };

        struct GTComparisonImageLookup {
            GTComparisonImageStatus status = GTComparisonImageStatus::Loading;
            std::shared_ptr<lfs::core::Tensor> image;
            std::string error;
            std::shared_ptr<lfs::core::Tensor> stale_image;
            bool grace_elapsed = true;
        };

        static constexpr auto CAMERA_METRICS_REFRESH_INTERVAL = std::chrono::milliseconds(500);
        static constexpr auto GT_COMPARISON_IMAGE_GRACE_PERIOD = std::chrono::milliseconds(300);
        static constexpr auto GT_COMPARISON_IMAGE_RETRY_COOLDOWN = std::chrono::seconds(2);

        void applySplitModeChange(const SplitViewService::ModeChangeResult& result);
        void queueCameraMetricsRefreshIfStale(ViewId view, SceneManager* scene_manager);
        void invalidateCameraMetricsRequests(bool clear_latest = false);
        void requestViewFollowUp(ViewRenderState& view, DirtyMask flags);
        void queueSharedScratchRetry(ViewRenderState& view, DirtyMask retry_dirty);
        void notifyAsyncLodResultsReady();
        // Fills the request's level-of-detail cut for `model` (rendering_manager_lod.cpp).
        void prepareLodRequest(const RenderSettings& settings, const lfs::core::SplatData* model,
                               lfs::rendering::ViewportRenderRequest& request,
                               std::vector<std::uint32_t>& lod_touched_chunks);
        void noteLodPageGeneration(std::uint64_t generation);
        void cameraMetricsWorkerLoop(std::stop_token stop_token);
        [[nodiscard]] GTComparisonImageLookup getOrQueueGTComparisonImage(
            GTComparisonImageJobRequest request);
        void queueGTComparisonImagePrefetch(GTComparisonImageJobRequest request);
        void invalidateGTComparisonImageCache(ViewRenderState& view);
        void insertGTComparisonImageCacheEntry(
            const GTComparisonImageJobRequest& request,
            std::shared_ptr<lfs::core::Tensor> image,
            std::string error,
            std::chrono::steady_clock::time_point now);
        void gtComparisonImageWorkerLoop(std::stop_token stop_token);
        void releaseSceneRenderResources();
        void setupEventHandlers();
        void handleToggleSplitView();
        void handleToggleGTComparison();
        void handleGoToCamView(int cam_id);
        void handleSplitPositionChanged(float position);
        void handleRenderSettingsChanged(const lfs::core::events::ui::RenderSettingsChanged& event);
        void handleWindowResized();
        void handleGridSettingsChanged(const lfs::core::events::ui::GridSettingsChanged& event);
        void handleTrainingStarted();
        void handleTrainingCompleted();
        void handleSceneLoaded();
        void handleSceneChanged(uint32_t mutation_flags);
        void handleSceneCleared();
        void handlePLYVisibilityChanged();
        void handlePLYAdded();
        void handlePLYRemoved();
        void handleCropBoxChanged(bool enabled);
        void handleEllipsoidChanged(bool enabled);
        void handlePointCloudModeChanged(const lfs::core::events::ui::PointCloudModeChanged& event);
        [[nodiscard]] static int clampGridPlane(int plane);
        [[nodiscard]] op::DepthWindowModeSnapshot depthWindowSnapshotLocked(ViewId view) const;
        void applyDepthWindowProjectionLocked(ViewId view, const DepthWindowState& state);
        void applyDepthWindowModeTransitionLocked(SplitViewMode previous_mode,
                                                  SplitViewMode new_mode);

        // Core components
        std::unique_ptr<lfs::rendering::RenderingEngine> engine_;
        FrameRateTracker frame_rates_;
        FrameRates gui_frame_rates_;
        bool fps_idle_frame_ = false;

        RenderTargetRegistry render_targets_;
        RenderTargetId preview_render_target_ = render_targets_.allocate();
        std::optional<std::string> prepareImportRenderCheck(const RenderContext& context, const std::function<void()>& prepare_viewport);
        void noteImportRenderFrame(uint64_t scene_generation, std::string error = {});
        bool import_render_check_ = false;
        bool import_render_preparing_ = false;
        uint64_t import_render_generation_ = 0;
        unsigned import_render_frames_ = 0;
        std::optional<std::string> import_render_result_;
        [[nodiscard]] float trainingRefreshIntervalSec(const ViewRenderState& view) const;
        std::unique_ptr<SceneRenderer> scene_renderer_;
        std::unique_ptr<PointSceneRenderer> point_scene_renderer_;
        std::unique_ptr<SparkLodController> lod_controller_;
        const lfs::core::SplatData* lod_controller_model_ = nullptr;
        bool lod_controller_needs_sync_traversal_ = false;
        std::uint64_t lod_controller_page_map_generation_ = 0;
        // Cached SH0→RGB derivation for the point-cloud Vulkan path. Refreshed
        // only when the source sh0_raw() pointer/size changes so the Vulkan
        // renderer's per-tensor upload cache stays warm across frames.
        lfs::core::Tensor point_cloud_colors_cache_;
        const void* point_cloud_colors_cache_key_ = nullptr;
        std::size_t point_cloud_colors_cache_size_ = 0;
        // Submit serial of the last frame that drew the point cloud; its buffers
        // are released only after that frame has retired on the GPU.
        std::uint64_t point_cloud_last_frame_serial_ = 0;
        std::uint64_t point_cloud_data_revision_ = 0;
        std::uint64_t point_cloud_preview_selection_revision_ = 0;
        GraphicsContext* last_graphics_context_ = nullptr;
        std::atomic<bool> vksplat_terminal_release_pending_{false};
        ViewportFrameLifecycleService::ModelSource renderer_model_source_ = ViewportFrameLifecycleService::ModelSource::Scene;

        std::chrono::steady_clock::time_point vksplat_idle_since_{};
        static constexpr std::uint64_t SPLIT_LEFT_GENERATION_BIT = 1ULL << 63;
        const lfs::core::Scene* gt_camera_index_scene_ = nullptr;
        std::uint64_t gt_camera_index_generation_ = 0;
        std::vector<std::shared_ptr<lfs::core::Camera>> gt_camera_index_cameras_;
        std::unordered_map<int, std::size_t> gt_camera_index_by_uid_;
        std::shared_ptr<lfs::core::Tensor> gt_comparison_cuda_image_;
        const lfs::core::Tensor* gt_comparison_cuda_source_ = nullptr;
        std::uint64_t gt_comparison_cuda_generation_ = 0;
        int gt_comparison_cuda_camera_uid_ = -1;
        glm::ivec2 gt_comparison_cuda_size_{0, 0};
        bool gt_comparison_cuda_undistorted_ = false;
        std::shared_ptr<lfs::core::Tensor> gt_comparison_loading_placeholder_;
        glm::ivec2 gt_comparison_loading_placeholder_size_{0, 0};
        std::shared_ptr<lfs::core::Tensor> gt_comparison_failed_placeholder_;
        glm::ivec2 gt_comparison_failed_placeholder_size_{0, 0};
        std::mutex wake_callback_mutex_;
        std::function<void()> wake_callback_;

        // GT compare-panel camera for the frame currently presented, for the
        // selection lane. Written and cleared at exactly the same sites as
        // this->state().vulkan_gt_comparison_content_size_.

        FrameDemandLedger frame_demand_ledger_;
        std::vector<ViewId> ledger_views_;
        struct GTComparisonImageCacheEntry {
            int camera_uid = -1;
            GTComparisonMode mode = GTComparisonMode::RGB;
            bool undistort_requested = false;
            std::filesystem::path image_path;
            glm::ivec2 image_size{0, 0};
            lfs::rendering::DepthVisualizationMode depth_visualization_mode =
                lfs::rendering::DepthVisualizationMode::Palette;
            glm::vec3 background_color{0.0f};
            std::shared_ptr<lfs::core::Tensor> image;
            std::string error;
            std::chrono::steady_clock::time_point failure_time{};
            std::chrono::steady_clock::time_point last_used{};
        };
        static bool gtRequestMatches(const GTComparisonImageJobRequest& lhs,
                                     const GTComparisonImageJobRequest& rhs);
        static bool gtCacheEntryMatches(const GTComparisonImageCacheEntry& entry,
                                        const GTComparisonImageJobRequest& request);
        static constexpr std::size_t GT_COMPARISON_IMAGE_CACHE_MAX_ENTRIES = 6;
        static constexpr std::size_t GT_COMPARISON_IMAGE_CACHE_MAX_BYTES = 128ULL * 1024ULL * 1024ULL;
        static constexpr std::size_t GT_COMPARISON_IMAGE_PREFETCH_MAX_ENTRIES = 4;
        std::list<GTComparisonImageCacheEntry> gt_comparison_image_cache_;
        std::size_t gt_comparison_image_cache_bytes_ = 0;
        mutable std::mutex gt_comparison_image_mutex_;
        std::optional<GTComparisonImageJobRequest> pending_gt_comparison_image_request_;
        std::optional<GTComparisonImageJobRequest> active_gt_comparison_image_request_;
        bool active_gt_comparison_image_is_prefetch_ = false;
        std::deque<GTComparisonImageJobRequest> prefetch_gt_comparison_image_requests_;
        uint64_t gt_comparison_image_request_generation_ = 0;
        std::condition_variable_any gt_comparison_image_cv_;
        std::jthread gt_comparison_image_worker_;
        // #1574 GT depth/normal async hold-then-swap: at most one outstanding ticket.

        // Granular dirty tracking

        CameraInteractionService camera_interaction_service_;

        RenderSettings activeSettingsLocked() const;
        void storeActiveSettingsLocked(const RenderSettings& settings);

        // Settings
        ViewSource& view_source_;
        SceneRenderSettings settings_;
        mutable std::mutex settings_mutex_;
        // Serializes release commit/undo/publication with mode transitions.
        // Acquire transition before settings; never while holding settings/history locks.
        // Release settings before pushing history. updateSettings releases settings,
        // acquires transition, then rechecks the mode. Equal-mode writes bypass this
        // nonrecursive lock for latch-release reentrancy.
        mutable std::mutex camera_metrics_mutex_;
        std::optional<CameraMetricsOverlayState> latest_camera_metrics_;
        std::optional<CameraMetricsJobRequest> pending_camera_metrics_request_;
        std::optional<CameraMetricsJobRequest> active_camera_metrics_request_;
        struct CameraMetricsCacheEntry {
            CameraMetricsJobRequest request;
            CameraMetricsOverlayState metrics;
        };
        std::list<CameraMetricsCacheEntry> camera_metrics_cache_;
        std::condition_variable_any camera_metrics_cv_;
        std::jthread camera_metrics_worker_;
        uint64_t camera_metrics_request_generation_ = 0;
        std::chrono::steady_clock::time_point last_camera_metrics_refresh_time_{};
        bool initialized_ = false;
        bool lod_available_ = false;

        GizmoState gizmo_state_;

        lfs::event::ScopedHandler event_handlers_;

        friend class RenderingManagerEventsTest_SceneClearedResetsFrustumLoaderSyncCache_Test;
        friend class SceneManager;
    };

} // namespace lfs::vis
