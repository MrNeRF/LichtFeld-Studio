/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "camera_interaction_service.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/export.hpp"
#include "core/tensor.hpp"
#include "depth_window_state.hpp"
#include "dirty_flags.hpp"
#include "framerate_controller.hpp"
#include "internal/viewport.hpp"
#include "io/loader.hpp"
#include "passes/vulkan_depth_blit_pass.hpp"
#include "passes/vulkan_environment_pass.hpp"
#include "passes/vulkan_mesh_pass.hpp"
#include "passes/vulkan_split_view_pass.hpp"
#include "render_animation_state.hpp"
#include "rendering/rendering.hpp"
#include "rendering/scene_temporal_resolve.hpp"
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "rendering/temporal_frame_tracker.hpp"
#include "rendering_types.hpp"
#include "spark_lod_controller.hpp"
#include "split_view_service.hpp"
#include "viewport_appearance_correction.hpp"
#include "viewport_artifact_service.hpp"
#include "viewport_frame_lifecycle_service.hpp"
#include "viewport_interaction_context.hpp"
#include "viewport_interop_service.hpp"
#include "viewport_overlay_service.hpp"
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
#include <vulkan/vulkan.h>

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

namespace lfs::core::events::cmd {
    struct ToggleIndependentSplitView;
} // namespace lfs::core::events::cmd

namespace lfs::vis::op {
    struct DepthWindowModeSnapshot;
}

namespace lfs::vis {
    class VulkanContext;
    class VksplatViewportRenderer;
    class PointCloudVulkanRenderer;
    struct VulkanViewportPassParams;

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
            const Viewport& viewport;
            const RenderSettings& settings;
            glm::ivec2 logical_screen_size{0, 0};
            const ViewportRegion* viewport_region = nullptr;
            SceneManager* scene_manager = nullptr;
            VulkanContext* vulkan_context = nullptr;
        };

        struct VulkanFrameResult {
            std::shared_ptr<const lfs::core::Tensor> image;
            VkImage external_image = VK_NULL_HANDLE;
            VkImageView external_image_view = VK_NULL_HANDLE;
            VkImageLayout external_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t external_image_generation = 0;
            VkSemaphore completion_semaphore = VK_NULL_HANDLE;
            std::uint64_t completion_value = 0;
            // Bumps only when the underlying image content changes (fresh render).
            // Cache-HIT frames keep the previous value so downstream consumers
            // (e.g. CUDA→Vulkan interop upload) can skip work by generation.
            std::uint64_t image_generation = 0;
            std::uint64_t split_left_image_generation = 0;
            glm::ivec2 size{0, 0};       // valid/logical viewport extent
            glm::ivec2 alloc_size{0, 0}; // bucketed image extent (0 = treat as size)
            bool flip_y = false;
            // True only when this output was rendered for the logical viewport
            // extent in the current request. Internal reconstruction resolution
            // may differ from that extent.
            bool matches_viewport_extent = false;

            // Split-view right panel. The left panel reuses the `image` slot above
            // (rideshares the existing scene-image interop). When this is set, the
            // gui-side split interop slot uploads it in parallel to the left panel.
            std::shared_ptr<const lfs::core::Tensor> split_right_image{};
            std::uint64_t split_right_image_generation = 0;
            glm::ivec2 split_right_size{0, 0};
            bool split_right_flip_y = false;
        };

        RenderingManager();
        ~RenderingManager();
        void setWakeCallback(std::function<void()> callback);

        // Initialize rendering resources
        void initialize();
        bool isInitialized() const { return initialized_; }
        void releaseSceneModelResources();

        // Main render function
        VulkanFrameResult renderVulkanFrame(const RenderContext& context);
        [[nodiscard]] std::expected<void, std::string> ensureVksplatTrainingSharedScratchReady(
            VulkanContext& context,
            const lfs::core::SplatData& model,
            glm::ivec2 viewport_size);
        // Called by the viewer loop at idle cadence. Releases private viewer
        // scratch after a hysteresis window, or immediately under pressure.
        void noteVksplatIdleFrame(bool training_active);

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
        std::shared_ptr<lfs::core::Tensor> renderPreviewImageRgb8(SceneManager* scene_manager,
                                                                  const glm::mat3& camera_rotation,
                                                                  const glm::vec3& camera_position,
                                                                  float focal_length_mm,
                                                                  int width, int height,
                                                                  std::optional<glm::vec3> background_color_override = std::nullopt,
                                                                  std::optional<bool> orthographic_override = std::nullopt,
                                                                  std::optional<float> ortho_scale_override = std::nullopt);

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
                                                                   std::optional<float> ortho_scale_override = std::nullopt);
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
            std::optional<bool> orthographic_override;
            std::optional<float> ortho_scale_override;
            ExportPostProcessMode mode = ExportPostProcessMode::Opaque;
        };
        [[nodiscard]] std::expected<lfs::core::Tensor, std::string> renderExportImage(
            SceneManager* scene_manager, const ExportImageRequest& request);

        [[nodiscard]] lfs::io::SplatTensorAllocator makeSplatTensorAllocator() const;

        void markDirty();
        void markDirty(DirtyMask flags);
        void markCameraPoseChanged();
        // Marks a discontinuous camera jump. Unlike interactive camera motion,
        // the next successfully published temporal frame must not reproject
        // history across this boundary.
        void markCameraCut();

        [[nodiscard]] bool pollDirtyState();

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
        void updateSettings(
            const RenderSettings& settings,
            DirtyMask dirty_flags,
            SceneUpscalerPresetUpdate preset_update = SceneUpscalerPresetUpdate::UseRequested);
        RenderSettings getSettings() const;
        // The presentation pass reports its actual runtime choice after pipeline
        // preparation. Rendering uses this feedback on the next frame so a failed
        // reconstruction pipeline never receives a reduced-resolution image.
        void reportSceneUpscalerRuntimeSelection(SceneUpscalerSelection selection);
        [[nodiscard]] SceneUpscalerSelection sceneUpscalerRuntimeSelection() const;

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
        // Internal drag-preview signal for the shader-approximate live reveal; never touches
        // RenderSettings.
        [[nodiscard]] bool depthWindowDragPreview() const;
        // Drag OWNERSHIP bracket, PER PANEL: opened at the drag operator's
        // INVOKE and closed at its DESTRUCTION. The registry invokes a
        // replacement modal BEFORE destroying the outgoing one, so this bracket
        // makes the panel continuously owned across a handoff - the count never
        // passes through zero mid-replacement, and the incoming drag inherits
        // the pre-drag backup the outgoing one recorded.
        // A panel's backup follows the CLAIM RULE (ownership is the authority,
        // not backup presence): a drag claiming an OWNED slot takes over and
        // RETAINS the recorded backup (first-wins across an unbroken ownership
        // chain - the replacement handoff); a drag claiming an UNOWNED slot
        // REFRESHES the backup from the live value, so an orphan left by a
        // cancelled chain can never become a later drag's false undo baseline.
        // A backup is released by a successful commit on that panel, by the
        // DEPTH-RELEVANT mode transition that consumes it, or by any
        // legitimate non-drag write to that slot - NEVER by this bracket's
        // close, which would strand an uncommitted preview with nothing left
        // to fold back to.
        // Returns TRUE when this drag also pinned the OTHER panel's backup
        // (the fan-out predicate), i.e. when its writes may reach both slots,
        // and reports through out_drag_token the OWNERSHIP token minted for
        // this drag. The drag hands that token back on every subsequent
        // manager call: it is the single identity that decides whether this
        // drag may still write, restore or release a given slot (see the
        // ownership note on depth_window_pin_owners_).
        bool beginDepthWindowDrag(SplitViewPanelId panel, uint64_t& out_drag_token);
        void endDepthWindowDrag(SplitViewPanelId panel, uint64_t drag_token);
        // PREVIEW gate, deliberately separate from the ownership bracket: it
        // spans only the drag's LATCH lifetime (startLatch..finishLatch), so a
        // subthreshold press that never starts a draw is not a live preview.
        // Still a counter, for the same replacement reason as above.
        void beginDepthWindowPreview(SplitViewPanelId panel);
        void endDepthWindowPreview(SplitViewPanelId panel);
        [[nodiscard]] bool isIndependentSplitViewActive() const;
        [[nodiscard]] GTComparisonMode getGTComparisonMode() const;
        [[nodiscard]] SplitViewMode getSplitViewMode() const;
        // Project restore may only enter/leave split modes through the service
        // transition path; it must never assign RenderSettings::split_view_mode.
        void restoreSplitViewMode(SplitViewMode mode,
                                  Viewport& primary_viewport);
        [[nodiscard]] float getSplitPosition() const;
        [[nodiscard]] std::optional<float> getSplitDividerScreenX(const glm::vec2& viewport_pos,
                                                                  const glm::vec2& viewport_size) const;
        void setFocusedSplitPanel(SplitViewPanelId panel);
        [[nodiscard]] SplitViewPanelId getFocusedSplitPanel() const { return split_view_service_.focusedPanel(); }
        [[nodiscard]] int getGridPlaneForPanel(SplitViewPanelId panel) const;
        void setGridPlaneForPanel(SplitViewPanelId panel, int plane);
        [[nodiscard]] DepthWindowState getDepthWindowForPanel(SplitViewPanelId panel) const;
        struct DepthWindowOverlaySnapshot {
            bool independent_dual_active = false;
            std::array<DepthWindowState, 2> panel_windows{};
        };
        [[nodiscard]] DepthWindowOverlaySnapshot getDepthWindowOverlaySnapshot() const;
        void setDepthWindowForPanel(SplitViewPanelId panel, const DepthWindowState& state);
        // DRAG-LANE preview write. Guarded by BOTH identities: the mode epoch
        // (the drag's lifetime against transitions) and the drag's OWNERSHIP
        // token against the pressed panel's slot. A legitimate non-drag write
        // clears that slot's owner, so a superseded drag's next write refuses
        // and can never overwrite the newer state.
        bool applyDepthWindowForPanelIfEpoch(SplitViewPanelId panel,
                                             const DepthWindowState& state,
                                             uint64_t expected_epoch,
                                             uint64_t drag_token);
        // RESTORE mode: safety machinery, not a user edit. Writes exactly the
        // one slot (plus the projection when that panel is focused) and NEVER
        // fans out, whatever the sync flag says - a teardown/cancel restore
        // must not push the restored panel's value over the other panel, whose
        // slot may legitimately carry a different, later window.
        bool restoreDepthWindowForPanelIfEpoch(SplitViewPanelId panel,
                                               const DepthWindowState& state,
                                               uint64_t expected_epoch);
        // TEARDOWN restore, per slot, decided UNDER ONE LOCK against the slot
        // OWNERS. A drag may only put back a slot it still OWNS: any legitimate
        // non-drag write cleared that slot's owner (the user superseded the
        // pre-drag value) and any later drag took the owner over, so the newer
        // intent wins.
        // own_state is the pressed panel's pre-drag value; other_state carries
        // the other slot's pre-drag value iff this drag pinned it (fan-out).
        // Each slot is written individually in RESTORE mode - never a fan-out.
        // Returns false only when the epoch moved (the caller is expired).
        bool restorePinnedDepthWindowSlots(SplitViewPanelId panel,
                                           const DepthWindowState& own_state,
                                           const std::optional<DepthWindowState>& other_state,
                                           uint64_t expected_epoch,
                                           uint64_t drag_token);
        // Release-time commit: the epoch check, the final panel write and the
        // snapshot the undo entry is built from all happen under ONE lock. A
        // refused commit therefore leaves NO trace - the caller skips both the
        // undo push and the draw-commit publication.
        bool commitDepthWindowForPanelIfEpoch(SplitViewPanelId panel,
                                              const DepthWindowState& state,
                                              uint64_t expected_epoch,
                                              uint64_t drag_token,
                                              op::DepthWindowModeSnapshot& out_snapshot);
        // Serializes a depth-window drag's RELEASE SEQUENCE against split-mode
        // transitions - see the lock-order note on
        // depth_window_transition_mutex_. The release holds this across
        // {commit + epoch re-check + undo push + draw-commit publication};
        // every mode-change site holds it around the mode change.
        [[nodiscard]] std::unique_lock<std::mutex> acquireDepthWindowTransitionLock() {
            return std::unique_lock<std::mutex>(depth_window_transition_mutex_);
        }
        void setDepthWindowSync(bool sync);
        [[nodiscard]] bool getDepthWindowSync() const;
        // WHICH kind of slot-invalidating write the last lineage stamp
        // described. A consumer that caches per-panel state derived from the
        // slots cannot replay that cache across any of these, but WHAT it
        // should do instead differs per kind, so the kind travels with the
        // record rather than being guessed from the endpoint.
        enum class DepthWindowLineageKind {
            // Leaving independent-dual folded the pre-transition focused
            // panel's window into the single remaining one.
            LeaveCollapse,
            // Enabling sync copied the focused panel's window over the other.
            SyncCopy,
            // A project restore seeded BOTH slots from the restored
            // projection; no previously cached slot survives it.
            ProjectRestore,
        };
        // One slot-invalidating write's provenance: which panel it took the
        // surviving window from, how many such writes had happened when it did,
        // and what kind of write it was.
        struct DepthWindowCollapseRecord {
            SplitViewPanelId source = SplitViewPanelId::Left;
            uint64_t generation = 0;
            DepthWindowLineageKind kind = DepthWindowLineageKind::LeaveCollapse;
        };
        // PROVENANCE of the last LINEAGE EVENT, not only of a collapse: the
        // panel the surviving window was taken from. For a leave collapse that
        // is the PRE-transition focused panel whose window
        // applyDepthWindowModeTransitionLocked folded into the single remaining
        // one, and the split service resets the observable focus to Left on the
        // way out (split_view_service.cpp:214), so this is the only surface a
        // poller can learn it from. A sync-ON copy and a project/sync-undo
        // restore overwrite this field too, so it names the source of whichever
        // write stamped LAST - use getDepthWindowCollapseRecord() to learn
        // which kind that was.
        [[nodiscard]] SplitViewPanelId getDepthWindowCollapseSource() const;
        // The REFERENCE-LINEAGE record: the same provenance, stamped with a
        // monotonically increasing generation and the kind of write that
        // stamped it. Endpoint identity alone cannot describe a
        // leave -> enter -> leave cycle that a 100ms poller slept through: the
        // final source names the last write only, while the poller's cached
        // per-panel state predates the FIRST one. Counting the writes is what
        // lets a consumer tell "the one transition I observed" from "boundaries
        // I missed", and the kind is what tells it how to recover. Source,
        // generation and kind are read under ONE lock, so a consumer can never
        // pair fields from different instants. The counter bumps at exactly the
        // FOUR writes that invalidate a slot-derived cache - the leave
        // collapse, the sync-ON copy, the project restore, and a SYNC undo/redo
        // restore - and nowhere else: not on mode-enter, not on GT boundaries,
        // not on an ordinary per-panel write. (depthWindowModeEpoch counts
        // mode-enter and GT boundaries too, which is why it cannot serve here.)
        // The first three stamp inside the critical section that does the write;
        // the fourth stamps from its call site through the public
        // stampDepthWindowSyncRestoreLineage, which takes settings_mutex_ a
        // SECOND time just after the restore released it (see that method).
        // Two of the four kinds COLLAPSE to one window (leave, sync-ON copy) and
        // two restore two possibly-DIFFERING absolute windows (project restore,
        // sync undo/redo - which is why the latter carries the ProjectRestore
        // kind), so a consumer holding per-panel references must re-baseline
        // each of them from its OWN slot rather than from the projection.
        [[nodiscard]] DepthWindowCollapseRecord getDepthWindowCollapseRecord() const;
        // One locked read of the whole depth-window state, so an absolute undo
        // snapshot can never mix slots, sync and epoch from different instants.
        [[nodiscard]] op::DepthWindowModeSnapshot depthWindowSnapshot() const;
        // The same locked read, composed for a DRAG's baseline capture: every
        // slot this drag OWNS reports the manager's RECORDED BACKUP instead of
        // the live value. That backup is the authoritative pre-drag value by
        // construction - first-wins on record, preserved across a take-over -
        // so a replacement drag's undo baseline and teardown target are the
        // clean state, never the abandoned live preview its predecessor left
        // in the slots. Slots this drag does not own report the live value
        // unchanged. A drag_token of 0 (never minted) composes nothing.
        [[nodiscard]] op::DepthWindowModeSnapshot
        depthWindowBaselineSnapshotForDrag(uint64_t drag_token) const;
        void restoreDepthWindowStateFromProject();
        // Epoch comparison and the absolute restore under ONE lock: an undo
        // entry can never observe a matching epoch and then write into a newer
        // one. `restore_sync` is false for drag entries — only the dedicated
        // sync entry owns the sync flag, and a drag undo recomputes the
        // projection from whichever panel is focused when it runs.
        bool restoreDepthWindowSnapshotIfEpoch(const op::DepthWindowModeSnapshot& snapshot,
                                               uint64_t expected_epoch,
                                               bool restore_sync);
        // The FOURTH stamp site, and the only one outside this class: a SYNC
        // undo/redo restore. It is stamped from the sync entry's call site
        // rather than from restoreDepthWindowSnapshotIfEpoch, because that
        // shared body also serves every DRAG undo - which invalidates nothing
        // and must keep stamping nothing. The kind is ProjectRestore because a
        // sync undo restores two possibly-differing ABSOLUTE window snapshots:
        // no cached per-panel reference survives it, which is exactly the
        // fresh-baseline rule that kind carries.
        void stampDepthWindowSyncRestoreLineage();
        [[nodiscard]] uint64_t depthWindowProjectionGeneration() const;
        [[nodiscard]] uint64_t depthWindowModeEpoch() const;
        [[nodiscard]] Viewport& resolvePanelViewport(Viewport& primary_viewport,
                                                     SplitViewPanelId panel = SplitViewPanelId::Left);
        [[nodiscard]] const Viewport& resolvePanelViewport(const Viewport& primary_viewport,
                                                           SplitViewPanelId panel = SplitViewPanelId::Left) const;
        // Project VIEW owns both panel cameras even while split view is
        // disabled; unlike resolvePanelViewport this never aliases primary.
        [[nodiscard]] Viewport& projectSecondaryViewport() {
            return split_view_service_.secondaryViewport();
        }
        [[nodiscard]] const Viewport& projectSecondaryViewport() const {
            return split_view_service_.secondaryViewport();
        }
        [[nodiscard]] Viewport& resolveFocusedViewport(Viewport& primary_viewport);
        [[nodiscard]] const Viewport& resolveFocusedViewport(const Viewport& primary_viewport) const;

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
            Viewport& primary_viewport,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size,
            std::optional<glm::vec2> screen_point = std::nullopt,
            std::optional<SplitViewPanelId> panel_override = std::nullopt);
        [[nodiscard]] std::optional<ViewerPanelInfo> resolveViewerPanel(
            const Viewport& primary_viewport,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size,
            std::optional<glm::vec2> screen_point = std::nullopt,
            std::optional<SplitViewPanelId> panel_override = std::nullopt) const;

        struct ContentBounds {
            float x, y, width, height;
            bool letterboxed = false;
        };
        ContentBounds getContentBounds(const glm::ivec2& viewport_size) const;

        struct GTSelectionContext {
            GTRenderCamera camera;
            glm::ivec2 size{0, 0};
        };
        [[nodiscard]] std::optional<GTSelectionContext> gtComparisonSelectionContext() const;

        // Current camera tracking for GT comparison
        void setCurrentCameraId(int cam_id) {
            const bool changed = camera_interaction_service_.currentCameraId() != cam_id;
            camera_interaction_service_.setCurrentCameraId(cam_id);
            if (changed) {
                invalidateCameraMetricsRequests(true);
            }
            markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::PPISP);
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
        float getAverageFPS() const { return framerate_controller_.getAverageFPS(); }
        float getPresentedAverageFPS() const {
            return presented_framerate_controller_.getAverageFPS();
        }
        [[nodiscard]] std::uint32_t temporalConvergenceRemaining() const {
            return temporal_convergence_.remaining();
        }
        // Measurement only — does not affect scene render pacing/limiting.
        void notePresentedFrame() { presented_framerate_controller_.beginFrame(); }

        // Access to the auxiliary rendering engine used by point-cloud, mesh, and readback paths.
        lfs::rendering::RenderingEngine* getRenderingEngine();
        [[nodiscard]] lfs::rendering::ScreenOverlayRenderer* getScreenOverlayRenderer() {
            return &screen_overlay_renderer_;
        }

        // Camera frustum picking
        int pickCameraFrustum(const glm::vec2& mouse_pos);

        // Depth access for tools (returns camera-space depth at pixel, or -1 if invalid).
        float getDepthAtPixel(int x, int y, std::optional<SplitViewPanelId> panel = std::nullopt) const;
        struct ExpectedDepthSampleRequest {
            SceneManager* scene_manager = nullptr;
            const Viewport* viewport = nullptr;
            glm::ivec2 render_size{0, 0};
            glm::ivec2 pixel{0, 0};
            float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
            bool orthographic = false;
            float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
        };
        // Renders a fresh expected-depth preview for precise picking on sparse or low-opacity splats.
        float renderExpectedDepthAtPixel(const ExpectedDepthSampleRequest& request);
        float renderDepthAtPixelForNodeMask(const SceneManager* scene_manager,
                                            const Viewport& viewport,
                                            const glm::ivec2& render_size,
                                            int x,
                                            int y,
                                            const std::vector<bool>& node_visibility_mask);
        glm::ivec2 getRenderedSize() const { return viewport_artifact_service_.renderedSize(); }
        std::shared_ptr<lfs::core::Tensor> getViewportImageIfAvailable() const;
        std::shared_ptr<lfs::core::Tensor> captureViewportImage();

        // Where the 3D viewport sat inside the window framebuffer on the last frame,
        // top-left origin. Lets callers crop a full-window readback down to the viewport
        // when no render path published an offscreen image to capture.
        struct FramebufferViewportRect {
            glm::ivec2 top_left{0, 0};
            glm::ivec2 size{0, 0};

            [[nodiscard]] bool valid() const { return size.x > 0 && size.y > 0; }
        };
        [[nodiscard]] FramebufferViewportRect framebufferViewportRect() const {
            return framebuffer_viewport_rect_;
        }
        [[nodiscard]] uint64_t getViewportProjectionGeneration() const {
            return viewport_projection_generation_;
        }

        void setCursorPreviewState(bool active, float x, float y, float radius, bool add_mode = true,
                                   lfs::core::Tensor* selection_tensor = nullptr,
                                   bool saturation_mode = false, float saturation_amount = 0.0f,
                                   std::optional<SplitViewPanelId> panel = std::nullopt,
                                   int focused_gaussian_id = -1, bool request_render = true);
        void clearCursorPreviewState();
        [[nodiscard]] bool isCursorPreviewActive() const { return viewport_overlay_service_.isCursorPreviewActive(); }
        [[nodiscard]] std::optional<SplitViewPanelId> getCursorPreviewPanel() const {
            return viewport_overlay_service_.cursorPreview().panel;
        }
        void getCursorPreviewState(float& x, float& y, float& radius, bool& add_mode) const {
            const auto& cursor = viewport_overlay_service_.cursorPreview();
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
        [[nodiscard]] bool isRectPreviewActive() const { return viewport_overlay_service_.isRectPreviewActive(); }
        [[nodiscard]] std::optional<SplitViewPanelId> getRectPreviewPanel() const {
            return viewport_overlay_service_.rectPanel();
        }
        void getRectPreview(float& x0, float& y0, float& x1, float& y1, bool& add_mode) const {
            x0 = viewport_overlay_service_.rectX0();
            y0 = viewport_overlay_service_.rectY0();
            x1 = viewport_overlay_service_.rectX1();
            y1 = viewport_overlay_service_.rectY1();
            add_mode = viewport_overlay_service_.rectAddMode();
        }
        [[nodiscard]] bool rectPreviewTracksCursor() const {
            return viewport_overlay_service_.rectTracksCursor();
        }

        // Polygon preview (render-space points, same coordinate system as screen_positions output)
        void setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed,
                               bool add_mode = true, std::optional<SplitViewPanelId> panel = std::nullopt);
        // Interactive polygon preview in world-space coordinates.
        void setPolygonPreviewWorldSpace(const std::vector<glm::vec3>& world_points, bool closed,
                                         bool add_mode = true,
                                         std::optional<SplitViewPanelId> panel = std::nullopt);
        void clearPolygonPreview();
        [[nodiscard]] bool isPolygonPreviewActive() const { return viewport_overlay_service_.isPolygonPreviewActive(); }
        [[nodiscard]] std::optional<SplitViewPanelId> getPolygonPreviewPanel() const {
            return viewport_overlay_service_.polygonPanel();
        }
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
        void setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode = true,
                             std::optional<SplitViewPanelId> panel = std::nullopt,
                             bool track_cursor = false);
        void clearLassoPreview();
        [[nodiscard]] bool isLassoPreviewActive() const { return viewport_overlay_service_.isLassoPreviewActive(); }
        [[nodiscard]] std::optional<SplitViewPanelId> getLassoPreviewPanel() const {
            return viewport_overlay_service_.lassoPanel();
        }
        [[nodiscard]] const std::vector<std::pair<float, float>>& getLassoPoints() const {
            return viewport_overlay_service_.lassoPoints();
        }
        [[nodiscard]] bool isLassoAddMode() const { return viewport_overlay_service_.lassoAddMode(); }
        [[nodiscard]] bool lassoPreviewTracksCursor() const {
            return viewport_overlay_service_.lassoTracksCursor();
        }

        // Vulkan mesh frame — populated by `renderVulkanFrame` when there are meshes in
        // the scene, consumed by gui_manager to feed `vulkan_viewport_pass.mesh_items`.
        struct VulkanMeshFrame {
            struct TemporalFrame {
                TemporalFrameInput input;
                SceneTemporalResolveSettings resolve_settings;
                SceneTemporalQuality quality = SceneTemporalQuality::Balanced;
            };

            glm::mat4 view_projection{1.0f};
            glm::vec3 camera_position{0.0f};
            std::vector<lfs::vis::VulkanMeshDrawItem> items;
            std::vector<lfs::vis::VulkanMeshViewportPanel> panels;
            lfs::vis::VulkanEnvironmentParams environment;
            lfs::vis::VulkanDepthBlitParams depth_blit;
            lfs::vis::VulkanSplitViewParams split_view;
            std::optional<TemporalFrame> temporal;
        };
        void setVulkanMeshFrame(VulkanMeshFrame frame) {
            std::lock_guard lock(vulkan_mesh_frame_mutex_);
            vulkan_mesh_frame_ = std::move(frame);
        }
        [[nodiscard]] VulkanMeshFrame getVulkanMeshFrame() const {
            std::lock_guard lock(vulkan_mesh_frame_mutex_);
            return vulkan_mesh_frame_;
        }
        void clearVulkanMeshFrame() {
            std::lock_guard lock(vulkan_mesh_frame_mutex_);
            vulkan_mesh_frame_ = {};
        }

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

        // Gizmo state for wireframe sync during manipulation
        void setCropboxGizmoState(bool active, const glm::vec3& min, const glm::vec3& max,
                                  const glm::mat4& world_transform, bool affects_render,
                                  int parent_node_index) {
            viewport_overlay_service_.setCropbox(active, min, max, world_transform, affects_render, parent_node_index);
        }
        void setEllipsoidGizmoState(bool active, const glm::vec3& radii,
                                    const glm::mat4& world_transform, bool affects_render,
                                    int parent_node_index) {
            viewport_overlay_service_.setEllipsoid(active, radii, world_transform, affects_render, parent_node_index);
        }
        void setCropboxGizmoActive(bool active) { viewport_overlay_service_.setCropboxActive(active); }
        void setEllipsoidGizmoActive(bool active) { viewport_overlay_service_.setEllipsoidActive(active); }
        [[nodiscard]] GizmoState getGizmoState() const { return viewport_overlay_service_.makeFrameGizmoState(); }

        void setViewportResizeActive(
            bool active,
            ViewportResizeRenderPolicy render_policy = ViewportResizeRenderPolicy::InteractivePreview);
        [[nodiscard]] bool isViewportResizeDeferring() const {
            return frame_lifecycle_service_.isResizeDeferring();
        }

        [[nodiscard]] ViewportInteropService& viewportInterop();
        [[nodiscard]] const ViewportInteropService& viewportInterop() const;
        void prepareViewportInterop(VulkanContext& context);
        void bindViewportInteropParams(VulkanViewportPassParams& params,
                                       std::size_t frame_slot,
                                       bool export_locked);
        void shutdownViewportInterop(VulkanContext* context = nullptr);
        [[nodiscard]] bool hasPendingViewportResizeSettle() const {
            return frame_lifecycle_service_.hasPendingResizeSettle();
        }
        [[nodiscard]] bool viewportResizeSettleReady() const {
            return frame_lifecycle_service_.resizeSettleReady();
        }
        [[nodiscard]] double secondsUntilViewportResizeSettleReady() const {
            return frame_lifecycle_service_.secondsUntilResizeSettleReady();
        }
        // LOD management
        void setLodAvailable(bool available);
        void setLodEnabled(bool enabled);
        [[nodiscard]] SparkLodController::Stats getLodStats() const;

    private:
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
        void clearVulkanViewportImageState(glm::ivec2 size = {0, 0},
                                           bool flip_y = false,
                                           glm::ivec2 alloc_size = {0, 0});

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
            PreviewImageReadback readback);
        [[nodiscard]] std::expected<void, std::string> renderPreviewImageToPreviewSlotWithState(
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
            std::optional<bool> transparent_background_override);
        [[nodiscard]] std::expected<void, std::string> renderDepthCaptureToPreviewSlotWithState(
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
            PreviewImageReadback readback);

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
            std::shared_ptr<lfs::io::PipelinedImageLoader> image_loader;
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
        void queueCameraMetricsRefreshIfStale(SceneManager* scene_manager);
        void invalidateCameraMetricsRequests(bool clear_latest = false);
        void requestRenderFollowUp();
        void requestTemporalFollowUp();
        void notifyAsyncLodResultsReady();
        void requestResizeTrainingPause(TrainerManager* trainer_manager);
        void releaseResizeTrainingPause();
        void cameraMetricsWorkerLoop(std::stop_token stop_token);
        [[nodiscard]] GTComparisonImageLookup getOrQueueGTComparisonImage(
            GTComparisonImageJobRequest request);
        void queueGTComparisonImagePrefetch(GTComparisonImageJobRequest request);
        void invalidateGTComparisonImageCache();
        void insertGTComparisonImageCacheEntry(
            const GTComparisonImageJobRequest& request,
            std::shared_ptr<lfs::core::Tensor> image,
            std::string error,
            std::chrono::steady_clock::time_point now);
        void gtComparisonImageWorkerLoop(std::stop_token stop_token);
        void releaseSceneRenderResources();
        void setupEventHandlers();
        void handleToggleSplitView();
        void handleToggleIndependentSplitView(const lfs::core::events::cmd::ToggleIndependentSplitView& event);
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
        void syncGridPlanesLocked(int plane);
        [[nodiscard]] op::DepthWindowModeSnapshot depthWindowSnapshotLocked() const;
        void restoreDepthWindowStateLocked(const std::array<DepthWindowState, 2>& panels,
                                           bool sync,
                                           const DepthWindowState& projection);
        // Returns whether the write fanned out to BOTH slots, so callers can
        // release the backups of exactly the slots they wrote.
        // restore_mode suppresses the sync fan-out unconditionally (see
        // restoreDepthWindowForPanelIfEpoch).
        bool applyDepthWindowForPanelLocked(SplitViewPanelId panel,
                                            const DepthWindowState& clamped,
                                            bool restore_mode = false);
        void releaseDepthWindowBackupsLocked(SplitViewPanelId panel, bool fan_out);
        // Uniform release for NON-DRAG full-slot writes: any slot no drag owns
        // loses its stale pre-drag backup, because the value now in that slot
        // is a legitimate write that a later transition must not roll back.
        void releaseIdleDepthWindowBackupsLocked();
        // The ONE writer of the reference-lineage record. Always called with
        // settings_mutex_ held, and it moves source, generation and kind
        // together, so the record itself is never torn.
        // It is NOT always the same lock hold as the write it describes: the
        // leave collapse, the sync-ON copy and the project restore stamp from
        // inside their own critical section, but the sync undo/redo restores
        // under one hold, releases it, and stamps through
        // stampDepthWindowSyncRestoreLineage under a second
        // (depth_window_undo_entry.cpp:100). A consumer that reads the slots
        // and the record separately can therefore observe one without the
        // other, and must revalidate the generation around its reads.
        void stampDepthWindowLineageLocked(SplitViewPanelId source,
                                           DepthWindowLineageKind kind);
        void applyDepthWindowModeTransitionLocked(SplitViewMode previous_mode,
                                                  SplitViewMode new_mode,
                                                  SplitViewPanelId pre_transition_focus);
        [[nodiscard]] bool depthWindowDragActiveLocked() const;
        // OWNERSHIP, not preview: true from the moment a drag is invoked until
        // its operator is destroyed, subthreshold presses included. The
        // sync-toggle gate reads THIS, so a drag's before_ capture can never
        // straddle a sync change.
        [[nodiscard]] bool depthWindowDragOwnedLocked() const;

        // Core components
        std::unique_ptr<lfs::rendering::RenderingEngine> engine_;
        lfs::rendering::ScreenOverlayRenderer screen_overlay_renderer_;
        mutable FramerateController framerate_controller_;
        // Parallel presented-frame counter (GUI-only frames included). Does not
        // drive pacing — scene path still uses framerate_controller_ alone.
        mutable FramerateController presented_framerate_controller_;

        std::shared_ptr<const lfs::core::Tensor> vulkan_viewport_image_;
        std::uint64_t vulkan_viewport_image_generation_ = 0;
        std::string last_logged_vksplat_render_error_;
        std::uint64_t viewport_projection_generation_ = 1;
        std::uint64_t temporal_scene_revision_ = 1;
        TemporalConvergenceController temporal_convergence_;
        std::atomic<std::uint64_t> temporal_camera_cut_generation_{0};
        std::uint64_t consumed_temporal_camera_cut_generation_ = 0;
        bool scene_reconstruction_request_logged_ = false;
        std::string last_scene_reconstruction_backend_;
        std::string last_scene_reconstruction_preset_;
        std::unique_ptr<VksplatViewportRenderer> vksplat_viewport_renderer_;
        std::unique_ptr<PointCloudVulkanRenderer> point_cloud_vulkan_renderer_;
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
        std::uint64_t point_cloud_data_revision_ = 0;
        std::uint64_t point_cloud_preview_selection_revision_ = 0;
        VulkanContext* last_vulkan_context_ = nullptr;
        std::atomic<bool> vksplat_terminal_release_pending_{false};
        std::uint32_t vksplat_idle_frame_count_ = 0;
        VkImage vulkan_external_viewport_image_ = VK_NULL_HANDLE;
        VkImageView vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
        VkImageLayout vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t vulkan_external_viewport_image_generation_ = 0;
        static constexpr std::uint64_t SPLIT_LEFT_GENERATION_BIT = 1ULL << 63;
        std::uint64_t split_view_image_generation_ = 0;
        std::uint64_t split_left_image_generation_ = 0;
        std::uint64_t split_right_image_generation_ = 0;
        const lfs::core::Tensor* split_left_source_ = nullptr;
        glm::ivec2 split_left_source_size_{0, 0};
        int split_left_source_camera_uid_ = -1;
        bool split_left_source_undistorted_ = false;
        glm::ivec2 split_right_source_size_{0, 0};
        cudaStream_t gt_comparison_worker_stream_ = nullptr;
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
        glm::ivec2 vulkan_viewport_image_size_{0, 0};
        glm::ivec2 vulkan_viewport_image_alloc_size_{0, 0};
        glm::ivec2 vulkan_viewport_coordinate_size_{0, 0};
        bool vulkan_viewport_image_flip_y_ = false;
        glm::ivec2 vulkan_gt_comparison_content_size_{0, 0};
        // GT compare-panel camera for the frame currently presented, for the selection lane.
        // Written and cleared at exactly the same sites as vulkan_gt_comparison_content_size_.
        struct GTPresentedView {
            GTRenderCamera camera;
            glm::ivec2 size{0, 0};
        };
        std::optional<GTPresentedView> vulkan_gt_comparison_selection_view_;
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
        // Panel keeps gt_async_held_display_ until the next ticket delivers (never blank).
        std::uint64_t gt_async_depth_ticket_ = 0;
        lfs::core::Tensor gt_async_depth_dest_{};
        GTComparisonMode gt_async_ticket_mode_ = GTComparisonMode::RGB;
        std::optional<lfs::rendering::CameraIntrinsics> gt_async_ticket_intrinsics_;
        bool gt_async_ticket_flip_y_ = false;
        lfs::rendering::FrameMetadata gt_async_ticket_metadata_{};
        // The GT camera and size that produced a given async depth/normal image. Carried with
        // the ticket and promoted with the held display, so the selection lane can only ever see
        // the camera of the image the panel is ACTUALLY showing (#1574 hold-then-swap).
        std::optional<GTPresentedView> gt_async_ticket_view_;
        std::shared_ptr<lfs::core::Tensor> gt_async_held_display_;
        bool gt_async_held_flip_y_ = false;
        lfs::rendering::FrameMetadata gt_async_held_metadata_{};
        std::optional<GTPresentedView> gt_async_held_view_;
        TrainerManager* resize_training_pause_trainer_ = nullptr;
        bool resize_training_pause_active_ = false;

        // Granular dirty tracking
        std::atomic<uint32_t> dirty_mask_{DirtyFlag::ALL};

        RenderAnimationState animation_state_;
        FramebufferViewportRect framebuffer_viewport_rect_;
        ViewportArtifactService viewport_artifact_service_;
        std::unique_ptr<ViewportInteropService> viewport_interop_;

        CameraInteractionService camera_interaction_service_;
        SplitViewService split_view_service_;
        ViewportFrameLifecycleService frame_lifecycle_service_;

        // Settings
        RenderSettings settings_;
        std::array<int, 2> depth_window_drag_counts_{};
        std::array<int, 2> depth_window_preview_counts_{};
        std::array<std::optional<DepthWindowState>, 2> depth_window_drag_backups_{};
        // PER-SLOT DRAG OWNERSHIP. A slot's pre-drag
        // backup is guarded by exactly ONE owning drag, named by the
        // monotonically increasing token beginDepthWindowDrag minted for it.
        // This single identity replaces the earlier {aggregate pin counts, pin
        // generation, mode epoch} triple, whose gaps were that the
        // three partial identities could each say "still mine" about a slot
        // some newer writer had legitimately taken over.
        // OWNERSHIP LIFECYCLE. An owner is SET by beginDepthWindowDrag on every
        // slot that drag needs (its own panel; the other iff its writes fan
        // out) - taking over from any earlier owner while KEEPING the recorded
        // backup value, since first-wins value semantics are the replacement
        // handoff's contract and the take-over is same-thread sequential. It is
        // CLEARED by (a) the owning drag's own endDepthWindowDrag, which
        // releases only slots it still owns; (b) the depth-relevant mode
        // transition and the project restore, which consume every backup; or
        // (c) any legitimate non-drag write to that slot
        // (releaseDepthWindowBackupsLocked / releaseIdleDepthWindowBackupsLocked),
        // which supersedes the pre-drag value outright.
        // Ownership is the ONLY licence to write a slot from the drag lane
        // (applyDepthWindowForPanelIfEpoch, commitDepthWindowForPanelIfEpoch),
        // to restore it at teardown (restorePinnedDepthWindowSlots), and to
        // release it at end - a superseded drag's very next call refuses.
        std::array<std::optional<uint64_t>, 2> depth_window_pin_owners_{};
        // Source of the tokens above. Monotonic and NEVER reused, so a token
        // that no longer matches a slot's owner is unambiguously stale; token
        // 0 is reserved as "no drag" and is never minted.
        uint64_t depth_window_last_drag_token_ = 0;
        SceneUpscalerSelection scene_upscaler_runtime_selection_{};
        std::array<int, 2> panel_grid_planes_{{1, 1}};
        std::array<DepthWindowState, 2> panel_depth_windows_{};
        bool depth_window_sync_ = false;
        // Written by stampDepthWindowLineageLocked at each of the four
        // slot-invalidating writes (settings_mutex_ held by every caller,
        // including the sync undo/redo's own re-acquisition in
        // stampDepthWindowSyncRestoreLineage), read under the same lock.
        SplitViewPanelId depth_window_collapse_source_ = SplitViewPanelId::Left;
        // Bumped at the same write site, under the same lock, so the triple is
        // always consistent. Starts at 0: "nothing has invalidated a slot yet".
        uint64_t depth_window_collapse_generation_ = 0;
        DepthWindowLineageKind depth_window_collapse_kind_ =
            DepthWindowLineageKind::LeaveCollapse;
        uint64_t depth_window_projection_generation_ = 0;
        uint64_t depth_window_mode_epoch_ = 0;
        mutable std::mutex settings_mutex_;
        // Serializes a depth-window drag's release sequence against split-mode
        // transitions, so a transition can never interleave between a
        // successful commit and the undo push / draw-commit publication that
        // belong to it (no born-expired undo entry, no stale rebase signal).
        //
        // LOCK ORDER (STRICT, outermost first):
        //     depth_window_transition_mutex_
        //         -> settings_mutex_
        //             -> the undo history's internal mutex (via push())
        //
        // Nothing may acquire depth_window_transition_mutex_ while holding
        // settings_mutex_ or the history mutex. Every acquisition site is
        // therefore OUTSIDE any settings_mutex_ scope: the drag operator's
        // release sequence, the seven pre-collapse cancel hook regions in
        // rendering_manager_events.cpp, and restoreDepthWindowStateFromProject.
        //
        // CONDITIONAL SITE - updateSettings. It acquires this mutex ONLY when
        // the incoming settings actually move split_view_mode, and it does so
        // by releasing settings_mutex_ first, taking this mutex, re-taking
        // settings_mutex_ and RE-CHECKING the mode (which may have moved in
        // the gap). The acquisition is unreachable on the re-entrant chains
        // that already hold this mutex on the SAME thread - the mode-change
        // event sites' cancel hook and the drag release both reach
        // updateSettings through finishLatch ->
        // SelectionTool::setDepthWindowDragInProgress ->
        // applySelectionFilterSettings, which builds its RenderSettings from a
        // fresh getSettings() and therefore always carries the CURRENT
        // split_view_mode. An equal-mode write never touches this mutex, so
        // those chains cannot self-deadlock on a non-recursive mutex.
        mutable std::mutex depth_window_transition_mutex_;
        mutable std::mutex camera_metrics_mutex_;
        mutable std::mutex vulkan_mesh_frame_mutex_;
        VulkanMeshFrame vulkan_mesh_frame_;
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

        ViewportInteractionContext viewport_interaction_context_;

        ViewportOverlayService viewport_overlay_service_;

        lfs::event::ScopedHandler event_handlers_;

        friend class RenderingManagerEventsTest_SceneClearedResetsFrustumLoaderSyncCache_Test;
        friend class SceneManager;
    };

} // namespace lfs::vis
