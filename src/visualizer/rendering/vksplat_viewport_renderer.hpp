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

        VksplatViewportRenderer();
        ~VksplatViewportRenderer();

        VksplatViewportRenderer(const VksplatViewportRenderer&) = delete;
        VksplatViewportRenderer& operator=(const VksplatViewportRenderer&) = delete;

        [[nodiscard]] std::expected<RenderResult, std::string> render(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const lfs::rendering::ViewportRenderRequest& request,
            bool force_input_upload);

        void reset();

    private:
        struct ComposePipeline;

        [[nodiscard]] std::expected<void, std::string> ensureInitialized(VulkanContext& context);
        [[nodiscard]] std::expected<void, std::string> uploadInputs(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            int active_sh_degree,
            std::size_t ring_slot);
        [[nodiscard]] bool inputsResident(const lfs::core::SplatData& splat_data,
                                          std::size_t ring_slot) const;
        [[nodiscard]] std::expected<void, std::string> ensureOutputImage(VulkanContext& context, glm::ivec2 size);
        [[nodiscard]] std::expected<void, std::string> ensureComposePipeline(VulkanContext& context);
        [[nodiscard]] std::expected<void, std::string> composePixelState(
            VulkanContext& context,
            VkCommandBuffer cmd,
            const VulkanGSRendererUniforms& uniforms,
            const glm::vec3& background);

        struct CudaInputSlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::size_t element_size = 0; // sizeof(float)
            std::size_t live_bytes = 0;
        };

        void detachManagedBuffers();
        void plugRingInputs(std::size_t ring_slot, std::size_t num_splats);
        [[nodiscard]] std::expected<void, std::string> ensureCudaInputSlot(
            VulkanContext& context,
            CudaInputSlot& slot,
            std::size_t required_bytes,
            const char* debug_name);

        VulkanContext* context_ = nullptr;
        bool initialized_ = false;
        VulkanGSRenderer renderer_;
        VulkanGSPipelineBuffers buffers_;
        std::unique_ptr<ComposePipeline> compose_;
        VulkanContext::ExternalImage output_image_{};
        glm::ivec2 output_size_{0, 0};
        VkImageLayout output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t output_generation_ = 0;

        // CUDA-backed input buffers (xyz_ws, rotations, scales_opacs, sh_coeffs),
        // ring-buffered per frame-in-flight. Each ring slot owns its own set of
        // CUDA-imported VkBuffers so frame N's CUDA upload cannot race frame
        // N-1's Vulkan compute reads. Ring index = currentFrameSlot %
        // framesInFlight; size matches VulkanContext::framesInFlight() (asserted
        // at runtime). The host fallback path bypasses the ring and uploads
        // through the renderer's staging buffer.
        static constexpr std::size_t kInputRingSize = 2; // matches VulkanContext::kFramesInFlight
        std::array<std::array<CudaInputSlot, 4>, kInputRingSize> cuda_inputs_{};
        std::array<ModelInputSnapshot, kInputRingSize> ring_uploaded_{};
    };

} // namespace lfs::vis
