/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/exportable_storage.hpp"
#include "core/rad_pool_quant.hpp"
#include "core/splat_data.hpp"
#include "gpu_lod_target_feedback.hpp"
#include "lod_page_cache.hpp"
#include "lod_upload_engine.hpp"
#include "output_image_pool.hpp"
#include "output_slot_ring.hpp"
#include "readback_ticket_ring.hpp"
#include "scene_overlay_params.hpp"
#include "scene_renderer.hpp"
#if LFS_BUILD_TRAINER && LFS_HAS_CUDA
#include <cuda_runtime.h>
#endif
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include "rendering/rendering.hpp"
#include "vksplat_shared_scratch_install.hpp"
#include "window/vulkan_context.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis {

    // Starts with file-only work: shader blobs are cached for the first
    // renderer initialization without touching Vulkan.
    LFS_VIS_API void preloadVkSplatSpirvFiles();

    class VksplatViewportRenderer {
        friend struct VksplatScratchReleaseTestAccess;
        friend struct SplitOutputLifetimeTestAccess;

    public:
        using RenderResult = SceneRenderer::RenderResult;
        using ReadbackStats = SceneRenderer::ReadbackStats;
        using SelectionMaskShape = SceneRenderer::SelectionMaskShape;
        using SelectionMaskRequest = SceneRenderer::SelectionMaskRequest;
        using DepthSampleRequest = SceneRenderer::DepthSampleRequest;
        using ReadbackTicketStatus = SceneRenderer::ReadbackTicketStatus;
        using GpuLodSelectionStatus = SceneRenderer::GpuLodSelectionStatus;

        struct ModelInputSnapshot {
            const lfs::core::SplatData* model = nullptr;
            std::size_t count = 0;
            int max_sh_degree = -1;
            // Active training degree is part of the input contract: 0→1 is the
            // first frame the viewer ever samples pad-dropped q16 rest (M4).
            int active_sh_degree = -1;
            const void* means = nullptr;
            const void* scaling = nullptr;
            const void* rotation = nullptr;
            const void* opacity = nullptr;
            const void* sh0 = nullptr;
            const void* shn = nullptr;
            const void* shn_bounds = nullptr;
            // The pointer/size catches mask allocation or removal; the version
            // catches in-place content edits so every input ring opacity copy is
            // refreshed without treating the edit as a full model change.
            const void* deleted = nullptr;
            std::uint64_t deleted_version = 0;
            std::uint64_t exportable_generation = 0;
            bool shn_q16 = false;
            std::size_t means_bytes = 0;
            std::size_t scaling_bytes = 0;
            std::size_t rotation_bytes = 0;
            std::size_t opacity_bytes = 0;
            std::size_t sh0_bytes = 0;
            std::size_t shn_bytes = 0;
            std::size_t shn_bounds_bytes = 0;
            std::size_t deleted_bytes = 0;

            [[nodiscard]] bool valid() const { return model != nullptr && count > 0; }
            [[nodiscard]] friend bool operator==(const ModelInputSnapshot& a,
                                                 const ModelInputSnapshot& b) = default;
        };

        LFS_VIS_API VksplatViewportRenderer();
        LFS_VIS_API ~VksplatViewportRenderer();

        VksplatViewportRenderer(const VksplatViewportRenderer&) = delete;
        VksplatViewportRenderer& operator=(const VksplatViewportRenderer&) = delete;

        // Compiles the immutable raster pipelines. Independent of scene contents.
        [[nodiscard]] LFS_VIS_API std::expected<void, std::string> prepareDevice(VulkanContext& context);

        [[nodiscard]] LFS_VIS_API std::expected<RenderResult, std::string> render(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const lfs::rendering::ViewportRenderRequest& request,
            bool force_input_upload,
            RenderTargetId target,
            bool synchronize_input_upload = false,
            bool deterministic_export = false);
        [[nodiscard]] std::expected<RenderResult, std::string> rerenderSelectionOverlay(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const lfs::rendering::ViewportRenderRequest& request,
            RenderTargetId target,
            bool synchronize_input_read = false);
#if LFS_BUILD_TRAINER && LFS_HAS_CUDA
        [[nodiscard]] cudaExternalSemaphore_t renderCompleteFence() const {
            return static_cast<cudaExternalSemaphore_t>(training_completion_.get());
        }
#endif
        // True when the trainer can wait for this viewport's render completion.
        [[nodiscard]] bool hasLiveTrainerReleaseFence() const;
        [[nodiscard]] void* renderCompleteTimeline() const { return render_complete_timeline_; }
        [[nodiscard]] std::uint64_t renderCompleteValue() const { return last_submitted_render_value_; }

        [[nodiscard]] std::expected<void, std::string> ensureHandshakeReady(VulkanContext& context);
        [[nodiscard]] std::expected<void, std::string> ensureTrainingSharedScratchReady(
            VulkanContext& context,
            std::size_t num_splats,
            glm::ivec2 viewport_size);

        // Release viewer-owned scratch after an idle boundary. Shared training
        // scratch is released only when the caller explicitly permits it.
        void releaseScratchOnIdle(bool release_shared, bool allow_shared_reclaim = false);
        // Retain the next idle arena window after a bounded contention timeout.
        // The arena also expires the request, so an abandoned retry cannot stall
        // training indefinitely.
        void requestArenaHandoff();
        void cancelArenaHandoff();
        // Keeps a pending reservation alive and reports whether the next render
        // could claim the arena without waiting.
        [[nodiscard]] bool pollArenaHandoff();
        // Reserves the arena and waits, holding no lock, until the next render can
        // claim it or the timeout passes.
        [[nodiscard]] bool waitForArenaHandoff(std::chrono::milliseconds timeout);
        // While the camera moves during training, the viewer and training take
        // turns on the shared scratch (see kTrainingFramesPerNavigationRender).
        void setCameraNavigating(bool navigating);

        // Invoked with the completion value immediately after each live-model
        // submit, BEFORE the shared arena frame is released — the trainer's
        // borrow wait must cover the in-flight Vulkan batch before the trainer
        // can reacquire the arena (publishing at frame-scope exit is too late
        // and lets training kernels overwrite scratch the batch still reads).
        void setLiveSubmitCallback(std::function<void(std::uint64_t)> callback) {
            live_submit_callback_ = std::move(callback);
        }

        [[nodiscard]] bool nextOutputImagesNeedResize(
            glm::ivec2 size,
            RenderTargetId target) const;
        [[nodiscard]] LFS_VIS_API std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImage(
            VulkanContext& context,
            RenderTargetId target) const;
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImageRgba(
            VulkanContext& context,
            RenderTargetId target) const;
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImageRgb8(
            VulkanContext& context,
            RenderTargetId target) const;
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImageRgba8(
            VulkanContext& context,
            RenderTargetId target) const;
        // Reads the most recent render's raw per-pixel linear depth (the
        // final_pixel_depth buffer every chain writes) into an [H,W] CPU float32
        // tensor. Valid only directly after a render into this slot, before the
        // next render reuses the pixel_depth scratch.
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readPreviewDepth(
            VulkanContext& context,
            RenderTargetId target) const;
        // Forces the non-batched per-pixel rasterizer chain (not the macro-tile
        // HiGS chain, whose depth is one median per macro-tile, nor the batched
        // compose, which covers only a subset of pixels) so readPreviewDepth gets
        // full per-pixel depth. When `expected` is set, that rasterizer writes
        // alpha-weighted (expected) depth instead of the median — hole-free in
        // low-opacity regions. Set only around a depth-capture render.
        void setDepthCaptureMode(bool on, bool expected = false) {
            depth_capture_mode_ = on;
            depth_capture_expected_ = on && expected;
        }
        [[nodiscard]] std::expected<void, std::string> readOutputImageIntoCpuHwc(
            VulkanContext& context,
            RenderTargetId target,
            lfs::core::Tensor& destination,
            int destination_x,
            int destination_y) const;
        [[nodiscard]] std::expected<float, std::string> sampleDepthAtPixel(
            VulkanContext& context,
            const DepthSampleRequest& request) const;

        // Async readback tickets (#1574). Destination buffers must remain valid until
        // the ticket is Ready (poll/wait delivers) or Failed.

        [[nodiscard]] std::expected<std::uint64_t, std::string> submitReadOutputImageIntoCpuHwcTicket(
            VulkanContext& context,
            RenderTargetId target,
            lfs::core::Tensor& destination,
            int destination_x,
            int destination_y) const;
        [[nodiscard]] std::expected<std::uint64_t, std::string> submitReadOutputDepthImageTicket(
            VulkanContext& context,
            RenderTargetId target,
            lfs::core::Tensor& destination) const;
        [[nodiscard]] std::expected<ReadbackTicketStatus, std::string> pollReadbackTicket(
            std::uint64_t ticket) const;
        [[nodiscard]] std::expected<void, std::string> waitReadbackTicket(
            std::uint64_t ticket) const;
        // Mark ticket Failed (if Outstanding) and clear meta.dest so the host may drop
        // its storage without UAF. Pins stay until the timeline completes and freeCell runs.
        void abandonReadbackTicket(std::uint64_t ticket) const;
        // Observability counters for LOG_PERF / GT compare cycles.
        [[nodiscard]] ReadbackStats readbackStats() const;

        [[nodiscard]] std::expected<lfs::core::Tensor, std::string> buildSelectionMask(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const SelectionMaskRequest& request,
            bool force_input_upload);

        [[nodiscard]] bool hasRenderTarget(RenderTargetId target) const {
            std::lock_guard lock(target_mutex_);
            return ring_.contains(target);
        }
        [[nodiscard]] LFS_VIS_API bool releaseRenderTarget(RenderTargetId target);
        [[nodiscard]] LFS_VIS_API std::shared_ptr<void> retainOutputImage(VkImageView view);
        void releaseSceneResources();
        void reset();
        [[nodiscard]] std::optional<LodPageCache::Snapshot> ensureLodPageCacheSnapshot(
            const lfs::core::SplatData& splat_data);
        // VRAM page-pool budget in splats for RAD-backed LoD streaming; 0 = full residency.
        void setLodPagePoolBudget(std::size_t splats) {
            lod_pool_sizing_dirty_ = lod_pool_sizing_dirty_ || lod_page_pool_splats_ != splats;
            lod_page_pool_splats_ = splats;
        }
        // Fraction of free VRAM granted to the out-of-core page pool.
        void setLodPoolVramFraction(float fraction) {
            lod_pool_sizing_dirty_ = lod_pool_sizing_dirty_ || lod_pool_vram_fraction_ != fraction;
            lod_pool_vram_fraction_ = fraction;
        }
        // Frames a newly streamed page fades in over; 0 disables fading.
        void setLodFadeFrames(std::uint32_t frames) { lod_fade_frames_ = frames; }

        // Snapshot of the GPU LoD selector for stats overlays; counts are from
        // the deferred readback (one frame stale).

        [[nodiscard]] GpuLodSelectionStatus gpuLodSelectionStatus(RenderTargetId target) const;

    private:
        struct ResidentRasterScratchProvenance {
            RenderTargetId target{};
            glm::ivec2 size{0, 0};
            glm::ivec2 camera_size{0, 0};
            glm::ivec2 subregion_origin{0, 0};
            glm::mat3 rotation{1.0f};
            glm::vec3 translation{0.0f};
            float focal_length_mm = 0.0f;
            bool orthographic = false;
            float ortho_scale = 0.0f;
            lfs::rendering::CameraIntrinsics intrinsics{};
            float scaling_modifier = 1.0f;
            int sh_degree = 0;
            std::vector<glm::mat4> model_transforms;
            std::vector<bool> node_visibility;
            const void* transform_indices = nullptr;
            bool gut = false;
            bool equirectangular = false;
            bool mip_filter = false;
            bool antialiasing = false;
            std::size_t num_splats = 0;
            bool valid = false;
        };
        ResidentRasterScratchProvenance resident_raster_scratch_{};
        ModelInputSnapshot resident_model_snapshot_{};
        [[nodiscard]] LFS_VIS_API ResidentRasterScratchProvenance makeResidentRasterScratchProvenance(
            RenderTargetId target, const lfs::rendering::ViewportRenderRequest& request, std::size_t num_splats) const;
        [[nodiscard]] LFS_VIS_API bool residentRasterScratchCompatible(const ResidentRasterScratchProvenance& published,
                                                                       const ResidentRasterScratchProvenance& requested) const;
        struct ComposePipeline;
        struct InputBindingResult {
            bool model_snapshot_changed = false;
        };

        [[nodiscard]] std::expected<void, std::string> ensureInitialized(VulkanContext& context);
        // Returns the next candidate without mutating timeline state. The value
        // is committed only after VulkanGSPipeline confirms vkQueueSubmit
        // accepted its signal operation.
        [[nodiscard]] lfs::Result<std::uint64_t> nextRenderCompletionValue(
            std::string_view pass) const;
        [[nodiscard]] std::expected<InputBindingResult, std::string> prepareInputs(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            std::size_t ring_slot,
            bool force_upload,
            int upload_sh_degree);
        [[nodiscard]] std::expected<void, std::string> ensureLodPageInputStorage(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            int upload_sh_degree);
        [[nodiscard]] std::expected<void, std::string> ensureGpuLodTreeStorage(
            const lfs::core::SplatData& splat_data);
        [[nodiscard]] std::expected<void, std::string> uploadLodPageInputs(
            const lfs::core::SplatData& splat_data,
            std::span<const LodPageCache::PendingUpload> uploads);
        void configureLodUploadEngine(const lfs::core::SplatData& splat_data);
        void unconfigureLodUploadEngines(std::string_view reason);
        void stopLodStreaming(std::string_view reason);
        void discardLodEngineResults(std::vector<LodPageCache::PendingUpload>&& results,
                                     std::string_view reason);
        void logLodUploadProgress(std::size_t published_pages);
        struct OverlayBindingViews {
            _VulkanBuffer selection_mask{};
            _VulkanBuffer preview_mask{};
            _VulkanBuffer selection_colors{};
            _VulkanBuffer transform_indices{};
            _VulkanBuffer node_mask{};
            _VulkanBuffer overlay_params{};
            _VulkanBuffer model_transforms{};
            bool raster_overlays_active = true;
        };
        [[nodiscard]] std::expected<OverlayBindingViews, std::string> uploadOverlayBindings(
            VulkanContext& context,
            const lfs::rendering::ViewportRenderRequest& request,
            std::size_t num_splats,
            std::size_t ring_slot,
            RenderTargetId target);
        [[nodiscard]] lfs::Status ensureOutputImages(
            VulkanContext& context,
            glm::ivec2 size,
            RenderTargetId target,
            std::size_t ring_slot);
        [[nodiscard]] std::expected<void, std::string> ensureComposePipeline(VulkanContext& context);
        [[nodiscard]] lfs::Status composePixelState(
            VulkanContext& context,
            VkCommandBuffer cmd,
            const VulkanGSRendererUniforms& uniforms,
            const glm::vec3& background,
            RenderTargetId target,
            std::size_t output_ring_slot,
            bool transparent_background,
            bool depth_view,
            float depth_min,
            float depth_max,
            lfs::rendering::DepthVisualizationMode depth_visualization_mode);
        [[nodiscard]] lfs::Status waitForRingSlot(
            std::size_t ring_slot,
            std::string_view reason);
        [[nodiscard]] std::size_t acquireRingSlot(RenderTargetId target = {});

        static constexpr std::size_t kInputRegionCount = 7;
        static constexpr std::size_t kOverlayRegionCount = 7;
        static constexpr std::size_t kSelectionQueryRegionCount = 7;
        static constexpr std::size_t kRegionAlignment = 256; // VK minStorageBufferOffsetAlignment upper bound on common HW
        // A deleted mask whose byte size is not a multiple of 4 is bound from a
        // padded copy; other masks bind directly.
        struct DeletedMaskSlot {
            lfs::core::Tensor padded_mask;
        };
        struct OverlaySlot {
            lfs::core::Tensor storage;
            lfs::core::Tensor vulkan_selection_mask;
            lfs::core::Tensor vulkan_preview_mask;
            lfs::core::Tensor vulkan_selection_colors;
            lfs::core::Tensor vulkan_transform_indices;
            lfs::core::Tensor vulkan_node_mask;
            lfs::core::Tensor vulkan_overlay_params;
            lfs::core::Tensor vulkan_model_transforms;
            std::array<std::size_t, kOverlayRegionCount> region_offset{};
            std::array<std::size_t, kOverlayRegionCount> region_bytes{};
            lfs::core::Tensor selection_source;
            lfs::core::Tensor preview_source;
            std::vector<float> color_table_upload_cpu;
            // Fingerprint of the palette currently staged in the interop buffer.
            // Hits on drag frames where the theme/palette is unchanged.
            std::array<glm::vec4, lfs::rendering::kSelectionColorTableCount> cached_color_palette{};
            bool color_table_uploaded = false;
            lfs::core::Tensor transform_indices_source;
            const void* cached_transform_indices_ptr = nullptr;
            std::size_t cached_transform_indices_bytes = 0;
            bool transform_indices_uploaded = false;
            std::vector<std::uint8_t> node_mask_upload_cpu;
            // Fingerprint of the emphasized and culling visibility masks currently
            // staged in the interop buffer.
            std::vector<bool> cached_emphasized_node_mask;
            std::vector<bool> cached_culling_node_mask;
            RenderTargetId cached_node_mask_target{};
            bool node_mask_uploaded = false;
            std::vector<float> overlay_params_upload_cpu;
            // Output-byte fingerprint of the overlay-params table currently
            // staged in the interop buffer.
            std::vector<float> cached_overlay_params_cpu;
            bool overlay_params_uploaded = false;
            std::vector<float> model_transforms_upload_cpu;
            // Same output-byte fingerprint cache as overlay_params.
            std::vector<float> cached_model_transforms_cpu;
            bool model_transforms_uploaded = false;
        };
        struct SelectionQuerySlot {
            lfs::core::Tensor storage;
            lfs::core::Tensor vulkan_transform_indices;
            lfs::core::Tensor vulkan_node_mask;
            lfs::core::Tensor vulkan_primitives;
            lfs::core::Tensor vulkan_model_transforms;
            lfs::core::Tensor vulkan_polygon_vertices;
            lfs::core::Tensor vulkan_polygon_mask;
            lfs::core::Tensor vulkan_ring_pick;
            std::array<std::size_t, kSelectionQueryRegionCount> region_offset{};
            std::array<std::size_t, kSelectionQueryRegionCount> region_bytes{};
            std::array<std::size_t, kSelectionQueryRegionCount> region_capacity_bytes{};
            lfs::core::Tensor transform_indices_source;
            std::vector<std::uint8_t> node_mask_upload_cpu;
            std::vector<float> model_transforms_upload_cpu;
            std::vector<float> primitive_upload_cpu;
            std::vector<float> polygon_vertices_upload_cpu;
            lfs::core::Tensor output_tensor;
            const void* cached_transform_indices_ptr = nullptr;
            std::size_t cached_transform_indices_bytes = 0;
            bool transform_indices_uploaded = false;
            std::vector<bool> cached_node_visibility_mask;
            bool node_mask_uploaded = false;
            std::vector<float> cached_model_transforms_cpu;
            bool model_transforms_uploaded = false;
            std::vector<glm::vec2> cached_polygon_vertices;
            bool polygon_vertices_uploaded = false;
        };

        void detachManagedBuffers();
        void releaseDeletedMaskSlot(std::size_t ring_slot);
        void logVramBreakdownIfChanged(std::string_view reason);
        [[nodiscard]] std::expected<void, std::string> ensureSharedScratchArena(
            VulkanContext& context,
            std::size_t required_bytes);
        // Re-imports the shared block if training grew it in place. Must be called
        // while the render owns the arena frame (training excluded) so the block is
        // stable, avoiding a cross-thread grow/re-import race.
        [[nodiscard]] std::expected<void, std::string> reimportSharedScratchIfGrown(VulkanContext& context);
        // image_width/height are logical (valid) extents. Pixel/tile region
        // capacities inside are bucketed to ceil64 (issue #1565).
        [[nodiscard]] std::size_t estimateSharedScratchBytes(std::size_t num_splats,
                                                             std::size_t visible_capacity,
                                                             bool macro_chain,
                                                             std::size_t sort_capacity,
                                                             std::size_t image_width,
                                                             std::size_t image_height) const;
        void bindSharedScratchBuffers(std::size_t num_splats,
                                      std::size_t visible_capacity,
                                      bool macro_chain,
                                      std::size_t sort_capacity,
                                      std::size_t image_width,
                                      std::size_t image_height);
        LFS_VIS_API void releasePrivateScratchBuffers();
        void releaseGpuLodTreeStorage();
        void renewArenaHandoff();
        void detachSharedScratchBuffers();
        void releaseSharedScratchImportOnly();
        void releaseSharedScratchArena();
        // Called by the training arena after its CUDA/Vulkan release timeline
        // has drained, before exportable VMM chunks are unmapped.
        bool prepareSharedScratchForArenaShrink(
            const std::shared_ptr<lfs::core::ExportableBlock>& block);
        // evict=true: pool entries destroy on drain instead of free-list reuse.

        // Queues a no-longer-current shared-scratch import for destruction once
        // the GPU submission that last referenced it has retired. The old VkBuffer
        // may still be read by in-flight graphics/compute submissions (the resize
        // path only fences the graphics queue), so freeing it immediately is a
        // use-after-free that surfaces as VK_ERROR_DEVICE_LOST. The timeline value
        // the batch submit signals covers the async-compute work too.
        void retireSharedScratchBuffer(VulkanContext::ExternalBuffer&& old);
        // Destroys retired imports whose retirement timeline value has been
        // reached. force=true destroys all of them unconditionally and is only
        // safe after vkDeviceWaitIdle (reset/teardown).
        void drainRetiredScratchBuffers(bool force);
        // True when the render timeline has passed `value` (0 / no timeline =
        // trivially retired; a failed query holds the resource for retry).
        [[nodiscard]] bool renderTimelineValueRetired(std::uint64_t value);
        // Drain/trim the viewport output-image pool. Predicates mirror scratch
        // retirement (producer timeline) plus graphics-frame submit serials.
        // force=true only after device idle; never destroys live acquisitions.
        // When readback_mutex_held is true the caller already owns readback_mutex_
        // (release* paths); the pin predicate must not re-lock.
        void drainOutputImagePool(bool force, bool readback_mutex_held = false);
        // Free Failed ticket cells once the readback timeline reaches their ticket
        // (non-blocking). Caller must hold readback_mutex_.
        void reclaimCompletedFailedReadbackCells() const;
        void trimOutputImagePoolAged();
        void trimOutputImagePoolIdle();
        // Clamps input-storage retirements left keyed to a timeline value a
        // failed/early-exit frame never signalled (run on every render exit).
        void clampOrphanedInputRetirements();

        // 3-deep readback ticket ring (#1574): per-slot cmdbufs + staging, timeline
        // completion (replaces the single readback fence). Torn down in reset().
        [[nodiscard]] std::expected<void, std::string> ensureReadbackContext() const;
        [[nodiscard]] std::expected<void, std::string> ensureReadbackSlotStaging(
            VulkanContext& context,
            std::size_t cell,
            VkDeviceSize required_bytes) const;
        [[nodiscard]] std::expected<std::size_t, std::string> acquireReadbackCell() const;
        [[nodiscard]] std::expected<std::uint64_t, std::string> submitReadbackTicket(
            VulkanContext& context,
            std::size_t cell,
            VkCommandBuffer command_buffer,
            VkQueue submit_queue,
            std::uint64_t completion_value,
            VkPipelineStageFlags wait_stage,
            ReadbackTicketRing::TicketMeta meta,
            std::string_view validation_label,
            std::string_view operation_label,
            std::source_location location = std::source_location::current()) const;
        [[nodiscard]] std::expected<ReadbackTicketStatus, std::string> pollReadbackTicketLocked(
            std::uint64_t ticket) const;
        [[nodiscard]] std::expected<void, std::string> waitReadbackTicketLocked(
            std::uint64_t ticket) const;
        [[nodiscard]] std::expected<void, std::string> deliverReadbackTicket(
            std::size_t cell) const;
        [[nodiscard]] std::expected<void, std::string> waitReadbackTimelineValue(
            std::uint64_t value,
            std::string_view fingerprint) const;
        // Ring-cell pin: block OutputSlotRing reuse until readbacks sourcing the cell retire.
        [[nodiscard]] lfs::Status waitReadbackPinsForFrameRingCell(std::size_t ring_slot) const;
        [[nodiscard]] lfs::Result<glm::ivec2> latestOutputImageSize(RenderTargetId target) const;

        VulkanContext* context_ = nullptr;
        bool initialized_ = false;
        // Readback ticket ring resources (#1574). Mutable because public readbacks are const.
        mutable std::mutex readback_mutex_;
        mutable ReadbackTicketRing readback_ring_{};
        mutable VkCommandPool readback_graphics_pool_ = VK_NULL_HANDLE;
        mutable VkCommandPool readback_transfer_pool_ = VK_NULL_HANDLE;
        struct ReadbackSlotResources {
            VkCommandBuffer graphics_cmd = VK_NULL_HANDLE;
            VkCommandBuffer transfer_cmd = VK_NULL_HANDLE;
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
            VmaAllocationInfo staging_info{};
            VkDeviceSize staging_capacity = 0;
        };
        mutable std::array<ReadbackSlotResources, ReadbackTicketRing::kRingSize> readback_slots_{};
        mutable VkSemaphore readback_timeline_ = VK_NULL_HANDLE;
        mutable std::uint64_t next_readback_ticket_ = 0;
        VulkanGSRenderer renderer_;
        VulkanGSPipelineBuffers buffers_;
        struct LodUploadSignature {
            const lfs::core::SplatData* model = nullptr;
            std::size_t count = 0;
            std::uint64_t hash = 0;
            std::uint64_t generation = 0;
            bool valid = false;
        };
        LodUploadSignature uploaded_lod_indices_{};
        LodUploadSignature uploaded_lod_logical_indices_{};
        LodUploadSignature uploaded_lod_levels_{};
        LodUploadSignature uploaded_lod_weights_{};
        bool lod_indices_upload_pending_ = false;
        bool lod_logical_indices_upload_pending_ = false;
        bool lod_levels_upload_pending_ = false;
        bool lod_weights_upload_pending_ = false;
        GpuLodTargetFeedbackTable gpu_lod_feedback_;
        const lfs::core::SplatData* lod_feedback_model_ = nullptr;
        std::uint64_t lod_feedback_model_generation_ = 0;
        std::uint64_t lod_feedback_tree_generation_ = 0;
        const lfs::core::SplatData* lod_page_cache_model_ = nullptr;
        LodPageCache lod_page_cache_;
        std::size_t lod_page_pool_splats_ = 0;
        float lod_pool_vram_fraction_ = 0.15f;
        bool lod_pool_sizing_dirty_ = false;
        std::uint32_t lod_fade_frames_ = 12;
        std::uint64_t gpu_lod_last_page_generation_ = 0;
        std::uint64_t gpu_lod_last_publish_frame_ = 0;
        struct GpuLodTreeStorage {
            // Quantized sidecar records per physical-page node: RadMetaBoundsQ
            // (2 words) and RadMetaLinksQ (3 words); the selector dequantizes
            // against per-page frames and derives logical from page_to_chunk.
            Buffer<float> node_bounds;
            Buffer<std::uint32_t> node_links;
            // Per-page dequant frames for the non-pool (in-core) path; pool
            // models bind the page-input InputPageFrames region instead.
            Buffer<float> page_frames;
            Buffer<std::uint32_t> page_to_chunk;
            Buffer<std::uint32_t> chunk_to_page;
            // Per-page publish frame stamps driving selector fade-in.
            Buffer<std::uint32_t> page_age;
            const lfs::core::SplatData* model = nullptr;
            std::size_t node_count = 0;
            std::size_t physical_node_capacity = 0;
            std::size_t logical_chunks = 0;
            std::size_t physical_pages = 0;
            std::uint64_t tree_signature = 0;
            std::uint64_t page_map_generation = 0;
            std::vector<std::uint32_t> parent_indices;
            std::vector<std::uint32_t> page_to_chunk_cpu, chunk_to_page_cpu, page_age_cpu;
            lfs::core::Tensor cached_means_cpu, cached_scaling_cpu;
            bool valid = false;
        };
        GpuLodTreeStorage gpu_lod_tree_;
        struct LodTreeUpdate {
            _VulkanBuffer destination;
            std::vector<std::byte> bytes;
            size_t offset = 0;
        };
        std::vector<LodTreeUpdate> lod_tree_updates_;
        // The upload engine and renderer share tensor storage for tree metadata.
        struct LodTreeMetaStorage {
            lfs::core::Tensor tensor;
            _VulkanBuffer view{};
            std::size_t bounds_offset = 0;
            std::size_t links_offset = 0;
            std::size_t capacity_nodes = 0;
        };
        LodTreeMetaStorage lod_tree_meta_;
        // Pages published through the synchronous tensor path this frame
        // (pinned roots / in-core); for view-backed trees only these need
        // render-thread metadata writes — engine pages carry their own.
        std::vector<std::uint32_t> lod_sync_meta_pages_;
        const lfs::core::SplatData* lod_sink_model_ = nullptr;
        struct LodPageInputStorage {
            lfs::core::Tensor tensor;
            _VulkanBuffer view{};
            std::array<std::size_t, kInputRegionCount> region_offset{};
            std::array<std::size_t, kInputRegionCount> region_bytes{};
            const lfs::core::SplatData* model = nullptr;
            std::size_t physical_pages = 0;
            std::size_t splat_capacity = 0;
            int input_sh_degree = -1;
        };
        LodPageInputStorage lod_page_inputs_;
        std::unique_ptr<ComposePipeline> compose_;
        static constexpr std::size_t kFrameRingSize = OutputSlotRing::kFrameRingSize;
        OutputSlotRing ring_{};
        mutable std::recursive_mutex target_mutex_;
        RenderTargetId rendering_target_{};
        struct RetiredInputs {
            std::size_t base;
            std::uint64_t completion;
        };
        std::vector<RetiredInputs> retired_inputs_;
        OutputImagePool output_pool_{};
        std::shared_ptr<int> publication_lifetime_ = std::make_shared<int>(0);
        // Completion counter shared by tensor producers and Vulkan consumers.
        VkSemaphore render_complete_timeline_ = VK_NULL_HANDLE;
        // Last value whose signal operation was accepted by vkQueueSubmit.
        // Failed recording leaves it unchanged, so no consumer waits on an
        // unsignaled candidate.
        std::uint64_t last_submitted_render_value_ = 0;
        // When set, render() takes the legacy per-pixel chain so the depth
        // readback captures per-pixel depth (see setDepthCaptureMode).
        bool depth_capture_mode_ = false;
        // When set, the capture rasterizer writes expected (alpha-weighted) depth.
        bool depth_capture_expected_ = false;
        // Phase 7C-P3: owner latch for bounded ring/readback waits. Quarantine
        // never zeros ring completion watermarks (would manufacture a free slot).
        // Injected into wait context; policy stays on the renderer (not the ring).
        mutable std::atomic<bool> gpu_wait_quarantined_{false};
        // Whether the last main render used the macro-tile chain; the
        // selection-overlay re-render reuses its sorted buffers and must match.
        bool last_render_used_macro_chain_ = false;
        std::size_t resident_depth_wave_armed_ = 0;
        int resident_sort_bits_ = 0;
        // The first frame after an input reset remains on the legacy chain;
        // it uses the same fixed-K wave machinery as every other legacy frame.
        bool macro_chain_warmup_pending_ = true;

        std::vector<DeletedMaskSlot> deleted_mask_copies_{};
        std::vector<OverlaySlot> overlays_{};
        SelectionQuerySlot selection_query_{};
        std::vector<ModelInputSnapshot> ring_uploaded_{};
        int current_input_sh_degree_ = -1;
#if LFS_HAS_CUDA
        lfs::core::GpuBackend active_tensor_backend_ = lfs::core::GpuBackend::CUDA;
#else
        lfs::core::GpuBackend active_tensor_backend_ = lfs::core::GpuBackend::Vulkan;
#endif
        std::size_t last_vram_report_signature_ = 0;

        struct SharedScratchArena {
            std::shared_ptr<lfs::core::ExportableBlock> block;
            VulkanContext::ExternalBuffer imported_buffer{};
            std::size_t bytes = 0;
            // Viewer high-water is measured from offset zero. The trainer arena
            // deliberately reuses that same prefix during its exclusive epoch.
            std::atomic<std::size_t> viewer_high_water_bytes{0};
            // Set by the existing idle-release boundary. While set, the trainer
            // may reclaim the viewer-only prefix; the next viewer ensure clears it.
            std::atomic<bool> viewer_idle_reclaim_eligible{false};
            std::uint64_t generation = 0;
            bool installed_in_training_arena = false;
        };
        SharedScratchArena shared_scratch_{};
        std::uint64_t shared_scratch_attempt_serial_ = 0;
        // Last logged viewport scratch bucket (alloc extent); LOG_DEBUG only on change.
        std::uint32_t scratch_bucket_alloc_w_ = 0;
        std::uint32_t scratch_bucket_alloc_h_ = 0;

        // Old shared-scratch imports awaiting GPU retirement, keyed by the
        // render-complete timeline value at which they become safe to free.
        std::vector<std::pair<std::uint64_t, VulkanContext::ExternalBuffer>>
            retired_scratch_buffers_;
        // Private VMA scratch buffers awaiting the same timeline retirement.
        std::vector<std::pair<std::uint64_t, _VulkanBuffer>> retired_private_scratch_buffers_;

        std::uint64_t arena_handoff_token_ = 0;
        bool camera_navigating_ = false;

        std::function<void(std::uint64_t)> live_submit_callback_;

        // CUDA import of the render-complete timeline: the reverse edge of the
        // trainer↔viewer handshake. The trainer waits "render_complete >=
        // borrow value" GPU-side before its next in-place parameter writes.
        VulkanContext::ExternalSemaphore render_complete_external_{};
#if LFS_BUILD_TRAINER
        std::shared_ptr<void> training_completion_;
#endif
        VkSemaphore vulkan_query_complete_timeline_ = VK_NULL_HANDLE;
        std::uint64_t vulkan_query_complete_value_ = 0;

        // Zero-copy input storages bound to in-flight frames, keyed by the
        // completion value at which the GPU is done reading them. Keeps
        // VkBuffer + external memory + CUDA allocation alive across trainer
        // topology reallocations. Vulkan-backend tensors pin the same way
        // through TensorVulkanBuffer::keep_alive.
        std::vector<std::pair<std::uint64_t, std::vector<std::shared_ptr<void>>>>
            retired_input_storages_;
        // Async RAD page streaming: decoded pages are packed and copied on the
        // engine's own thread/stream; render frames only publish completions.
        LodUploadEngine lod_upload_engine_;
        std::vector<std::pair<lfs::core::TensorCompletion, lfs::core::RadPageSources>> resident_page_uploads_;
        // Last completion value whose frame read the LOD pool; pool reuse waits on it GPU-side.
        std::uint64_t last_lod_page_borrow_value_ = 0;
        std::uint64_t lod_upload_log_batches_ = 0;
        bool lod_upload_log_converged_ = false;
    };

} // namespace lfs::vis
