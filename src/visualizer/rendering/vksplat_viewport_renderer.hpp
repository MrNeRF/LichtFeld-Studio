/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/exportable_storage.hpp"
#include "core/splat_data.hpp"
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
            // Tracking the deleted mask pointer + byte count is enough to invalidate
            // the resident-input cache when the user soft-deletes (or undoes a
            // delete). The renderer then re-runs the copy path so the opacity
            // upload applies the latest mask.
            const void* deleted = nullptr;
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
            bool synchronize_input_upload = false;
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
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImage(
            VulkanContext& context,
            OutputSlot output_slot = OutputSlot::Main) const;
        [[nodiscard]] std::expected<float, std::string> sampleDepthAtPixel(
            VulkanContext& context,
            int x,
            int y,
            OutputSlot output_slot = OutputSlot::Main) const;
        [[nodiscard]] std::expected<lfs::core::Tensor, std::string> buildSelectionMask(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const SelectionMaskRequest& request,
            bool force_input_upload);

        void reset();

    private:
        struct ComposePipeline;
        struct InputBindingResult {
            bool uses_temporary_upload_slot = false;
        };

        [[nodiscard]] std::expected<void, std::string> ensureInitialized(VulkanContext& context);
        [[nodiscard]] std::expected<InputBindingResult, std::string> prepareInputs(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            std::size_t ring_slot,
            bool force_upload,
            int upload_sh_degree,
            bool synchronize_upload = false);
        struct OverlayBindingViews {
            _VulkanBuffer selection_mask{};
            _VulkanBuffer preview_mask{};
            _VulkanBuffer selection_colors{};
            _VulkanBuffer transform_indices{};
            _VulkanBuffer node_mask{};
            _VulkanBuffer overlay_params{};
            _VulkanBuffer model_transforms{};
            bool overlays_active = true;
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
            float depth_max);
        [[nodiscard]] std::expected<void, std::string> waitForRingSlot(
            std::size_t ring_slot,
            std::string_view reason);
        [[nodiscard]] std::size_t acquireRingSlot();
        [[nodiscard]] std::size_t latestOutputRingSlot(OutputSlot output_slot) const;

        // Fallback coalesced CUDA-imported VkBuffer per ring slot, holding raw
        // SplatData input regions back-to-back. Training tensors created as
        // Vulkan-external buffers bypass this allocation and are bound directly.
        static constexpr std::size_t kInputRegionCount = 6;
        static constexpr std::size_t kOverlayRegionCount = 7;
        static constexpr std::size_t kSelectionQueryRegionCount = 7;
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
                                                             std::size_t sort_capacity,
                                                             std::size_t num_pixels,
                                                             std::size_t num_tiles) const;
        void bindSharedScratchBuffers(std::size_t num_splats,
                                      std::size_t sort_capacity,
                                      std::size_t num_pixels,
                                      std::size_t num_tiles);
        void releasePrivateScratchBuffers();
        void detachSharedScratchBuffers();
        void releaseSharedScratchImportOnly();
        void releaseSharedScratchArena();
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

        // Lazily creates a persistent transfer command pool + buffer + fence reused by
        // readOutputImage / sampleDepthAtPixel instead of allocating a fresh pool/fence
        // per call. Torn down in reset() while the device is still valid.
        [[nodiscard]] std::expected<void, std::string> ensureReadbackContext() const;

        VulkanContext* context_ = nullptr;
        bool initialized_ = false;
        // Persistent readback transfer resources (see ensureReadbackContext). Mutable
        // because the readback samplers are const but reuse these across calls.
        mutable std::mutex readback_mutex_;
        mutable VkCommandPool readback_pool_ = VK_NULL_HANDLE;
        mutable VkCommandBuffer readback_cmd_ = VK_NULL_HANDLE;
        mutable VkFence readback_fence_ = VK_NULL_HANDLE;
        // Dedicated non-blocking CUDA stream for overlay-source H2D uploads.
        // Created with cudaStreamNonBlocking so it does NOT implicitly
        // serialize with the legacy default (NULL) stream where the rest of
        // the project's CUDA work runs — otherwise sub-KB uploads would still
        // wait for unrelated CUDA work to drain. Downstream Vulkan compute
        // observes the upload via the per-slot timeline semaphore signal, so
        // cross-API ordering is preserved without per-frame sync.
        cudaStream_t overlay_upload_stream_ = nullptr;
        VulkanGSRenderer renderer_;
        VulkanGSPipelineBuffers buffers_;
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
        std::array<std::uint64_t, kFrameRingSize> ring_completion_values_{};
        std::size_t next_ring_slot_ = 0;

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
    };

} // namespace lfs::vis
