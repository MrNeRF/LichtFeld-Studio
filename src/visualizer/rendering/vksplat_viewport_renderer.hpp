/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/exportable_storage.hpp"
#include "core/splat_data.hpp"
#include "lod_page_cache.hpp"
#include "lod_pool_quant.hpp"
#include "lod_upload_engine.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include "rendering/rendering.hpp"
#include "window/vulkan_context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis {

    class VksplatViewportRenderer {
    public:
        struct RenderResult {
            VkImage image = VK_NULL_HANDLE;
            VkImageView image_view = VK_NULL_HANDLE;
            VkImageLayout image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t generation = 0;
            VkImage depth_image = VK_NULL_HANDLE;
            VkImageView depth_image_view = VK_NULL_HANDLE;
            VkImageLayout depth_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t depth_generation = 0;
            glm::ivec2 size{0, 0};
            bool flip_y = false;
            VkSemaphore completion_semaphore = VK_NULL_HANDLE;
            std::uint64_t completion_value = 0;
            std::uint64_t lod_page_generation = 0;
            // True while page decodes/uploads are still in flight, or a
            // deferred capacity readback found a clamped VkSplat frame.
            bool lod_streaming_active = false;
            // True when the deferred capacity readbacks from the previous pass
            // confirm an unclamped frame for the current steady-state path.
            bool capacity_readback_settled = false;
        };

        struct ModelInputSnapshot {
            const lfs::core::SplatData* model = nullptr;
            std::size_t count = 0;
            int max_sh_degree = -1;
            const void* means = nullptr;
            const void* scaling = nullptr;
            const void* rotation = nullptr;
            const void* opacity = nullptr;
            const void* sh0 = nullptr;
            const void* shn = nullptr;
            // The pointer/size catches mask allocation or removal; the version
            // catches in-place content edits so every input ring opacity copy is
            // refreshed without treating the edit as a full model change.
            const void* deleted = nullptr;
            std::uint64_t deleted_version = 0;
            std::size_t means_bytes = 0;
            std::size_t scaling_bytes = 0;
            std::size_t rotation_bytes = 0;
            std::size_t opacity_bytes = 0;
            std::size_t sh0_bytes = 0;
            std::size_t shn_bytes = 0;
            std::size_t deleted_bytes = 0;

            [[nodiscard]] bool valid() const { return model != nullptr && count > 0; }
            [[nodiscard]] friend bool operator==(const ModelInputSnapshot& a,
                                                 const ModelInputSnapshot& b) = default;
        };

        enum class SelectionMaskShape : std::uint32_t {
            Brush = 0,
            Rectangle = 1,
            Polygon = 2,
            Ring = 3,
        };

        enum class OutputSlot : std::size_t {
            Main = 0,
            SplitLeft = 1,
            SplitRight = 2,
            Preview = 3,
        };

        struct SelectionMaskRequest {
            lfs::rendering::FrameView frame_view;
            lfs::rendering::GaussianSceneState scene;
            SelectionMaskShape shape = SelectionMaskShape::Brush;
            std::vector<glm::vec4> primitives;
            std::vector<glm::vec2> polygon_vertices;
            bool gut = false;
            bool equirectangular = false;
            bool mip_filter = false;
            float ring_width = 0.01f;
            bool synchronize_input_upload = false;
            std::uint32_t* picked_ring_id_out = nullptr;
        };

        struct DepthSampleRequest {
            glm::ivec2 pixel{0, 0};
            // Coordinate space of `pixel`. When positive, the renderer maps the
            // sample into the actual output image size for the selected slot.
            glm::ivec2 source_size{0, 0};
            OutputSlot output_slot = OutputSlot::Main;
        };

        VksplatViewportRenderer();
        ~VksplatViewportRenderer();

        VksplatViewportRenderer(const VksplatViewportRenderer&) = delete;
        VksplatViewportRenderer& operator=(const VksplatViewportRenderer&) = delete;

        [[nodiscard]] std::expected<RenderResult, std::string> render(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const lfs::rendering::ViewportRenderRequest& request,
            bool force_input_upload,
            OutputSlot output_slot = OutputSlot::Main,
            bool synchronize_input_upload = false);
        [[nodiscard]] std::expected<RenderResult, std::string> rerenderSelectionOverlay(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const lfs::rendering::ViewportRenderRequest& request,
            OutputSlot output_slot = OutputSlot::Main,
            bool synchronize_input_read = false);
        // Dedicated non-blocking CUDA stream for the render path (input
        // packing, overlay staging, selection queries). Producer tensors
        // bridge in with event edges; upload-timeline signals are enqueued on
        // it so Vulkan's waits cover the packing.
        [[nodiscard]] cudaStream_t renderStream() const { return render_stream_; }

        // Reverse edge of the trainer↔viewer handshake: the render-complete
        // timeline imported into CUDA, and the latest completion value covering
        // submits that bound live training storage. The trainer enqueues
        // "wait fence >= value" on its stream before in-place writes.
        [[nodiscard]] cudaExternalSemaphore_t renderCompleteFence() const {
            return render_complete_cuda_.handle();
        }
        [[nodiscard]] std::uint64_t renderCompleteValue() const { return last_signaled_render_value_; }

        // Eagerly create the render stream + completion fence so the trainer↔viewer
        // handshake can be installed before the first live frame submits (covers
        // training start, scene switch, and post-reset() frames).
        [[nodiscard]] std::expected<void, std::string> ensureHandshakeReady(VulkanContext& context) {
            return ensureInitialized(context);
        }
        [[nodiscard]] std::expected<void, std::string> ensureTrainingSharedScratchReady(
            VulkanContext& context,
            std::size_t num_splats,
            glm::ivec2 viewport_size);

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
            OutputSlot output_slot = OutputSlot::Main) const;
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImage(
            VulkanContext& context,
            OutputSlot output_slot = OutputSlot::Main) const;
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImageRgba(
            VulkanContext& context,
            OutputSlot output_slot = OutputSlot::Main) const;
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImageRgb8(
            VulkanContext& context,
            OutputSlot output_slot = OutputSlot::Main) const;
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImageRgba8(
            VulkanContext& context,
            OutputSlot output_slot = OutputSlot::Main) const;
        // Reads the most recent render's raw per-pixel linear depth (the
        // final_pixel_depth buffer every chain writes) into an [H,W] CPU float32
        // tensor. Valid only directly after a render into this slot, before the
        // next render reuses the pixel_depth scratch.
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readPreviewDepth(
            VulkanContext& context,
            OutputSlot output_slot = OutputSlot::Preview) const;
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputDepthImage(
            VulkanContext& context,
            OutputSlot output_slot = OutputSlot::Preview) const;
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
            OutputSlot output_slot,
            lfs::core::Tensor& destination,
            int destination_x,
            int destination_y) const;
        [[nodiscard]] std::expected<float, std::string> sampleDepthAtPixel(
            VulkanContext& context,
            const DepthSampleRequest& request) const;
        [[nodiscard]] std::expected<lfs::core::Tensor, std::string> buildSelectionMask(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const SelectionMaskRequest& request,
            bool force_input_upload);

        void releasePreviewResources();
        void releaseSceneResources();
        void reset();
        [[nodiscard]] std::optional<LodPageCache::Snapshot> ensureLodPageCacheSnapshot(
            const lfs::core::SplatData& splat_data);
        [[nodiscard]] std::optional<LodPageCache::Snapshot> lodPageCacheSnapshot(
            const lfs::core::SplatData& splat_data) const;
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
        struct GpuLodSelectionStatus {
            bool active = false;
            std::size_t selected = 0;
            std::size_t capacity = 0;
            std::size_t overflow = 0;
            float pixel_scale_feedback = 1.0f;
            std::size_t resident_chunks = 0;
            std::size_t chunk_count = 0;
            std::size_t touched_chunks = 0;
            std::size_t miss_chunks = 0;
            std::size_t deferred_requests = 0;
            bool admission_frozen = false;
            std::size_t pool_pages = 0;
            std::size_t streaming_jobs = 0;
        };
        [[nodiscard]] GpuLodSelectionStatus gpuLodSelectionStatus() const;

        // True when the most recent render's start-of-frame deferred poll
        // confirmed the previously rendered frame produced complete, unclamped
        // content using the steady-state rasterizer chain. One-shot preview/
        // export captures poll this to avoid reading back a capacity-clamped
        // (partial) frame; see RenderingManager::renderPreviewImageToPreviewSlotWithState.
        [[nodiscard]] bool previewCaptureSettled() const { return last_preview_capture_settled_; }

    private:
        struct ComposePipeline;
        struct InputBindingResult {
            bool uses_temporary_upload_slot = false;
            bool model_snapshot_changed = false;
        };

        [[nodiscard]] std::expected<void, std::string> ensureInitialized(VulkanContext& context);
        // Returns true iff the timeline reached `value` within the bound (i.e. the
        // submit that would signal it actually completed); false on timeout.
        [[nodiscard]] bool waitCompletionValueBounded(std::uint64_t value) noexcept;
        [[nodiscard]] std::expected<InputBindingResult, std::string> prepareInputs(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            std::size_t ring_slot,
            bool force_upload,
            int upload_sh_degree,
            bool synchronize_upload = false);
        [[nodiscard]] std::expected<void, std::string> ensureLodPageInputStorage(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            int upload_sh_degree);
        [[nodiscard]] std::expected<void, std::string> ensureGpuLodTreeStorage(
            const lfs::core::SplatData& splat_data);
        [[nodiscard]] std::expected<void, std::string> uploadLodPageInputs(
            const lfs::core::SplatData& splat_data,
            std::span<const LodPageCache::PendingUpload> uploads,
            std::size_t ring_slot);
        void configureLodUploadEngine(const lfs::core::SplatData& splat_data);
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
            std::size_t ring_slot);
        [[nodiscard]] bool inputsResident(const lfs::core::SplatData& splat_data,
                                          std::size_t ring_slot) const;
        [[nodiscard]] std::expected<void, std::string> ensureOutputImages(
            VulkanContext& context,
            glm::ivec2 size,
            OutputSlot output_slot,
            std::size_t ring_slot);
        [[nodiscard]] std::expected<void, std::string> ensureComposePipeline(VulkanContext& context);
        [[nodiscard]] std::expected<void, std::string> composePixelState(
            VulkanContext& context,
            VkCommandBuffer cmd,
            const VulkanGSRendererUniforms& uniforms,
            const glm::vec3& background,
            OutputSlot output_slot,
            std::size_t output_ring_slot,
            bool transparent_background,
            bool depth_view,
            float depth_min,
            float depth_max,
            lfs::rendering::DepthVisualizationMode depth_visualization_mode);
        [[nodiscard]] std::expected<void, std::string> waitForRingSlot(
            std::size_t ring_slot,
            std::string_view reason);
        [[nodiscard]] std::size_t acquireRingSlot();
        [[nodiscard]] std::size_t latestOutputRingSlot(OutputSlot output_slot) const;

        // Fallback coalesced CUDA-imported VkBuffer per ring slot, holding raw
        // SplatData input regions back-to-back. Training tensors created as
        // Vulkan-external buffers bypass this allocation and are bound directly.
        static constexpr std::size_t kInputRegionCount = 7;
        static constexpr std::size_t kOverlayRegionCount = 7;
        static constexpr std::size_t kSelectionQueryRegionCount = 8;
        static constexpr std::size_t kRegionAlignment = 256; // VK minStorageBufferOffsetAlignment upper bound on common HW
        struct CudaInputSlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::array<std::size_t, kInputRegionCount> region_offset{};
            std::array<std::size_t, kInputRegionCount> region_bytes{};
        };
        struct CudaOpacityCopySlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::size_t bytes = 0;
        };
        struct CudaOverlaySlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
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
            // Fingerprint of emphasized_node_mask currently staged in the
            // interop buffer.
            std::vector<bool> cached_emphasized_node_mask;
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
        struct CudaSelectionQuerySlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
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
        void plugRingInputs(std::size_t ring_slot, std::size_t num_splats, bool reset_cached_raster_state);
        void aliasSortScratchToInputSlot(std::size_t ring_slot);
        void releaseInputSlot(VulkanContext& context, std::size_t ring_slot);
        void releaseOpacityCopySlot(VulkanContext& context, std::size_t ring_slot);
        void logVramBreakdownIfChanged(std::string_view reason);
        [[nodiscard]] std::expected<void, std::string> ensureSharedScratchArena(
            VulkanContext& context,
            std::size_t required_bytes);
        // Re-imports the shared block if training grew it in place. Must be called
        // while the render owns the arena frame (training excluded) so the block is
        // stable, avoiding a cross-thread grow/re-import race.
        [[nodiscard]] std::expected<void, std::string> reimportSharedScratchIfGrown(VulkanContext& context);
        [[nodiscard]] std::size_t estimateSharedScratchBytes(std::size_t num_splats,
                                                             std::size_t visible_capacity,
                                                             bool macro_chain,
                                                             std::size_t sort_capacity,
                                                             std::size_t num_pixels,
                                                             std::size_t num_tiles) const;
        void bindSharedScratchBuffers(std::size_t num_splats,
                                      std::size_t visible_capacity,
                                      bool macro_chain,
                                      std::size_t sort_capacity,
                                      std::size_t num_pixels,
                                      std::size_t num_tiles);
        void releasePrivateScratchBuffers();
        void releaseGpuLodTreeStorage();
        void detachSharedScratchBuffers();
        void releaseSharedScratchImportOnly();
        void releaseSharedScratchArena();
        void releaseOutputSlot(OutputSlot output_slot);
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
        // Clamps input-storage retirements left keyed to a timeline value a
        // failed/early-exit frame never signalled (run on every render exit).
        void clampOrphanedInputRetirements();

        // Lazily creates a persistent transfer command pool + buffer + fence reused by
        // readOutputImage / sampleDepthAtPixel instead of allocating a fresh pool/fence
        // per call. Torn down in reset() while the device is still valid.
        [[nodiscard]] std::expected<void, std::string> ensureReadbackContext() const;
        [[nodiscard]] std::expected<glm::ivec2, std::string> latestOutputImageSize(OutputSlot output_slot) const;

        VulkanContext* context_ = nullptr;
        bool initialized_ = false;
        // Persistent readback transfer resources (see ensureReadbackContext). Mutable
        // because the readback samplers are const but reuse these across calls.
        mutable std::mutex readback_mutex_;
        mutable VkCommandPool readback_pool_ = VK_NULL_HANDLE;
        mutable VkCommandBuffer readback_cmd_ = VK_NULL_HANDLE;
        mutable VkFence readback_fence_ = VK_NULL_HANDLE;
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
        float gpu_lod_pixel_scale_feedback_ = 1.0f;
        // Consecutive readback frames with deferred wants but zero
        // admissions and nothing in flight; gates the threshold descent and
        // the keep-rendering liveness once it crosses the frozen window.
        std::uint32_t gpu_lod_frozen_frames_ = 0;
        std::size_t gpu_lod_last_candidate_count_ = 0;
        std::size_t gpu_lod_last_overflow_count_ = 0;
        std::size_t gpu_lod_last_miss_count_ = 0;
        // GPU traversal misses from the deferred selector readback, sorted by
        // descending pixel-scale priority for LodPageCache decode scheduling.
        std::vector<LodPageCache::ChunkRequest> gpu_lod_prefetch_requests_;
        std::vector<std::uint32_t> gpu_lod_protected_chunks_;
        bool gpu_lod_prefetch_valid_ = false;
        bool gpu_lod_selection_active_ = false;
        std::size_t gpu_lod_render_capacity_last_ = 0;
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
            std::vector<std::uint32_t> page_to_chunk_cpu;
            bool valid = false;
        };
        GpuLodTreeStorage gpu_lod_tree_;
        // CUDA-importable backing for node_bounds/node_links so the upload
        // engine writes expanded tree metadata with page payloads; the
        // Buffer shells above hold region views into it.
        struct LodTreeMetaStorage {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::size_t bounds_offset = 0;
            std::size_t links_offset = 0;
            std::size_t capacity_nodes = 0;
        };
        LodTreeMetaStorage lod_tree_meta_;
        // Pages published through the synchronous tensor path this frame
        // (pinned roots / in-core); for view-backed trees only these need
        // render-thread metadata writes — engine pages carry their own.
        std::vector<std::uint32_t> lod_sync_meta_pages_;
        LodUploadEngine::DeviceLayout lod_engine_layout_{};
        const lfs::core::SplatData* lod_sink_model_ = nullptr;
        struct LodPageInputStorage {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::array<std::size_t, kInputRegionCount> region_offset{};
            std::array<std::size_t, kInputRegionCount> region_bytes{};
            const lfs::core::SplatData* model = nullptr;
            std::size_t physical_pages = 0;
            std::size_t splat_capacity = 0;
            int input_sh_degree = -1;
        };
        LodPageInputStorage lod_page_inputs_;
        std::unique_ptr<ComposePipeline> compose_;
        struct OutputImageSlot {
            VulkanContext::ExternalImage image{};
            VulkanContext::ExternalImage depth_image{};
            glm::ivec2 size{0, 0};
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t generation = 0;
        };
        static constexpr std::size_t kOutputSlotCount = 4;
        static constexpr std::size_t kFrameRingSize = 3;
        std::array<std::array<OutputImageSlot, kFrameRingSize>, kOutputSlotCount> output_slots_{};
        std::array<std::size_t, kOutputSlotCount> latest_output_ring_slot_{};
        std::array<std::uint64_t, kOutputSlotCount> output_generations_{};
        VkSemaphore render_complete_timeline_ = VK_NULL_HANDLE;
        std::uint64_t render_complete_value_ = 0;
        // The latest completion value a submit actually signaled (or is guaranteed
        // to signal). renderCompleteValue() returns this — never the reserved
        // counter — so the trainer/arena never wait a value a failed frame left
        // unsignaled.
        std::uint64_t last_signaled_render_value_ = 0;
        // When set, render() takes the legacy per-pixel chain so the depth
        // readback captures per-pixel depth (see setDepthCaptureMode).
        bool depth_capture_mode_ = false;
        // When set, the capture rasterizer writes expected (alpha-weighted) depth.
        bool depth_capture_expected_ = false;
        std::array<std::uint64_t, kFrameRingSize> ring_completion_values_{};
        std::size_t next_ring_slot_ = 0;
        // Whether the last main render used the macro-tile chain; the
        // selection-overlay re-render reuses its sorted buffers and must match.
        bool last_render_used_macro_chain_ = false;
        // The first frame after a model/input reset needs the synchronous
        // render-tile chain so the viewport never presents the macro chain's
        // zero-count warm-up frame.
        bool macro_chain_warmup_pending_ = true;
        // Visible-splat capacity high-water mark, fed by the deferred raw emit
        // count (decays /8 per poll, grows after a clamped frame). 0 until the
        // first readback lands; the first frames size at the render domain.
        std::size_t visible_high_water_ = 0;
        // A clamped frame was observed and a complete one has not landed yet;
        // keeps frames scheduled while idle so the capacity self-heal converges.
        bool visible_clamp_pending_ = false;
        bool instance_clamp_pending_ = false;
        // Set each render from the start-of-frame deferred poll: true once a
        // representative frame (one that used the same steady-state rasterizer
        // chain the next pass will use) is confirmed complete and unclamped.
        // Drives the synchronous capacity self-heal used by one-shot preview/
        // export captures, which cannot tolerate the interactive loop's
        // one-frame clamp transient.
        bool last_preview_capture_settled_ = false;

        // Fallback CUDA-backed input buffers for models that are not already
        // backed by Vulkan-external tensor storage. Direct Vulkan-external
        // training tensors bypass this ring and bind their VkBuffers directly.
        static constexpr std::size_t kInputRingSize = kFrameRingSize;
        std::array<CudaInputSlot, kInputRingSize> cuda_inputs_{};
        std::array<CudaOpacityCopySlot, kInputRingSize> cuda_opacity_copies_{};
        std::array<CudaOverlaySlot, kInputRingSize> cuda_overlays_{};
        CudaSelectionQuerySlot cuda_selection_query_{};
        std::array<ModelInputSnapshot, kInputRingSize> ring_uploaded_{};
        int current_input_sh_degree_ = -1;
        std::size_t last_vram_report_signature_ = 0;

        struct SharedScratchArena {
            std::shared_ptr<lfs::core::ExportableBlock> block;
            VulkanContext::ExternalBuffer imported_buffer{};
            std::size_t bytes = 0;
            std::uint64_t generation = 0;
            bool installed_in_training_arena = false;
        };
        SharedScratchArena shared_scratch_{};
        std::uint64_t shared_scratch_attempt_serial_ = 0;

        // Old shared-scratch imports awaiting GPU retirement, keyed by the
        // render-complete timeline value at which they become safe to free.
        std::vector<std::pair<std::uint64_t, VulkanContext::ExternalBuffer>>
            retired_scratch_buffers_;

        // Per-ring-slot timeline semaphore used to gate Vulkan compute on the
        // CUDA upload completing; eliminates the per-frame
        // cudaStreamSynchronize that previously blocked the CPU after every
        // upload (P15). Values are monotonic; on each upload we bump the slot's
        // counter, signal CUDA-side, and queue a Vulkan-side wait.
        struct UploadTimeline {
            VulkanContext::ExternalSemaphore vk_semaphore{};
            lfs::rendering::CudaTimelineSemaphore cuda_semaphore{};
            std::uint64_t value = 0;
        };
        std::array<UploadTimeline, kInputRingSize> upload_timelines_{};
        std::array<UploadTimeline, kInputRingSize> overlay_upload_timelines_{};
        UploadTimeline selection_query_timeline_{};

        cudaStream_t render_stream_ = nullptr;

        std::function<void(std::uint64_t)> live_submit_callback_;

        // CUDA import of the render-complete timeline: the reverse edge of the
        // trainer↔viewer handshake. The trainer waits "render_complete >=
        // borrow value" GPU-side before its next in-place parameter writes.
        VulkanContext::ExternalSemaphore render_complete_external_{};
        lfs::rendering::CudaTimelineSemaphore render_complete_cuda_{};

        // The last completion value whose frame read the persistent (non-ring)
        // lod_page_inputs_ buffer; next-frame page uploads wait on it GPU-side.
        std::uint64_t last_lod_page_borrow_value_ = 0;

        // Zero-copy input storages bound to in-flight frames, keyed by the
        // completion value at which the GPU is done reading them. Keeps
        // VkBuffer + external memory + CUDA allocation alive across trainer
        // topology reallocations.
        std::vector<std::pair<std::uint64_t, std::vector<std::shared_ptr<void>>>>
            retired_input_storages_;

        // Async RAD page streaming: decoded pages are packed and copied on the
        // engine's own thread/stream; render frames only publish completions.
        UploadTimeline lod_engine_timeline_{};
        LodUploadEngine lod_upload_engine_;
        std::uint64_t lod_upload_log_batches_ = 0;
        bool lod_upload_log_converged_ = false;
    };

} // namespace lfs::vis
