/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/rasterizer/vksplat_fwd/src/gs_renderer.h"
#include "rendering/rendering.hpp"
#include "window/vulkan_context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <string>

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
            std::size_t means_bytes = 0;
            std::size_t scaling_bytes = 0;
            std::size_t rotation_bytes = 0;
            std::size_t opacity_bytes = 0;
            std::size_t sh0_bytes = 0;
            std::size_t shn_bytes = 0;

            [[nodiscard]] bool valid() const { return model != nullptr && count > 0; }
            [[nodiscard]] friend bool operator==(const ModelInputSnapshot& a,
                                                 const ModelInputSnapshot& b) = default;
        };

        enum class SelectionMaskShape : std::uint32_t {
            Brush = 0,
            Rectangle = 1,
        };

        enum class OutputSlot : std::size_t {
            Main = 0,
            SplitLeft = 1,
            SplitRight = 2,
        };

        struct SelectionMaskRequest {
            lfs::rendering::FrameView frame_view;
            lfs::rendering::GaussianSceneState scene;
            SelectionMaskShape shape = SelectionMaskShape::Brush;
            std::vector<glm::vec4> primitives;
            bool gut = false;
            bool equirectangular = false;
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
            bool synchronize_upload = false);
        struct OverlayBindingViews {
            _VulkanBuffer selection_mask{};
            _VulkanBuffer preview_mask{};
            _VulkanBuffer selection_colors{};
            _VulkanBuffer transform_indices{};
            _VulkanBuffer node_mask{};
            _VulkanBuffer overlay_params{};
            _VulkanBuffer model_transforms{};
        };
        [[nodiscard]] std::expected<OverlayBindingViews, std::string> uploadSelectionOverlay(
            VulkanContext& context,
            const lfs::rendering::ViewportRenderRequest& request,
            std::size_t num_splats,
            std::size_t ring_slot);
        [[nodiscard]] bool inputsResident(const lfs::core::SplatData& splat_data,
                                          std::size_t ring_slot) const;
        [[nodiscard]] std::expected<void, std::string> ensureOutputImages(
            VulkanContext& context,
            glm::ivec2 size,
            OutputSlot output_slot);
        [[nodiscard]] std::expected<void, std::string> ensureComposePipeline(VulkanContext& context);
        [[nodiscard]] std::expected<void, std::string> composePixelState(
            VulkanContext& context,
            VkCommandBuffer cmd,
            const VulkanGSRendererUniforms& uniforms,
            const glm::vec3& background,
            OutputSlot output_slot,
            bool transparent_background,
            bool depth_view,
            float depth_min,
            float depth_max);

        // Fallback coalesced CUDA-imported VkBuffer per ring slot, holding raw
        // SplatData input regions back-to-back. Training tensors created as
        // Vulkan-external buffers bypass this allocation and are bound directly.
        static constexpr std::size_t kInputRegionCount = 6;
        static constexpr std::size_t kOverlayRegionCount = 7;
        static constexpr std::size_t kSelectionQueryRegionCount = 5;
        static constexpr std::size_t kRegionAlignment = 256; // VK minStorageBufferOffsetAlignment upper bound on common HW
        struct CudaInputSlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::array<std::size_t, kInputRegionCount> region_offset{};
            std::array<std::size_t, kInputRegionCount> region_bytes{};
        };
        struct CudaOverlaySlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::array<std::size_t, kOverlayRegionCount> region_offset{};
            std::array<std::size_t, kOverlayRegionCount> region_bytes{};
            lfs::core::Tensor selection_source;
            lfs::core::Tensor preview_source;
            lfs::core::Tensor color_table_source;
            lfs::core::Tensor transform_indices_source;
            lfs::core::Tensor node_mask_source;
            lfs::core::Tensor overlay_params_source;
            lfs::core::Tensor model_transforms_source;
        };
        struct CudaSelectionQuerySlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::array<std::size_t, kSelectionQueryRegionCount> region_offset{};
            std::array<std::size_t, kSelectionQueryRegionCount> region_bytes{};
            lfs::core::Tensor transform_indices_source;
            lfs::core::Tensor node_mask_source;
            lfs::core::Tensor primitive_source;
            lfs::core::Tensor model_transforms_source;
            lfs::core::Tensor output_tensor;
        };

        void detachManagedBuffers();
        void plugRingInputs(std::size_t ring_slot, std::size_t num_splats);
        void aliasSortScratchToInputSlot(std::size_t ring_slot);
        void releaseInputSlot(VulkanContext& context, std::size_t ring_slot);

        VulkanContext* context_ = nullptr;
        bool initialized_ = false;
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
        static constexpr std::size_t kOutputSlotCount = 3;
        std::array<OutputImageSlot, kOutputSlotCount> output_slots_{};

        // Fallback CUDA-backed input buffers for models that are not already
        // backed by Vulkan-external tensor storage. Direct Vulkan-external
        // training tensors bypass this ring and bind their VkBuffers directly.
        static constexpr std::size_t kInputRingSize = 1;
        std::array<CudaInputSlot, kInputRingSize> cuda_inputs_{};
        std::array<CudaOverlaySlot, kInputRingSize> cuda_overlays_{};
        CudaSelectionQuerySlot cuda_selection_query_{};
        std::array<ModelInputSnapshot, kInputRingSize> ring_uploaded_{};

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
    };

} // namespace lfs::vis
