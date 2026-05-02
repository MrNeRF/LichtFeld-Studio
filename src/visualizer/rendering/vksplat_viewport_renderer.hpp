/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rasterizer/vksplat_fwd/src/gs_renderer.h"
#include "window/vulkan_context.hpp"

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
            int active_sh_degree);
        [[nodiscard]] bool inputsResident(const lfs::core::SplatData& splat_data) const;
        [[nodiscard]] std::expected<void, std::string> ensureOutputImage(VulkanContext& context, glm::ivec2 size);
        [[nodiscard]] std::expected<void, std::string> ensureComposePipeline(VulkanContext& context);
        [[nodiscard]] std::expected<void, std::string> composePixelState(
            VulkanContext& context,
            const VulkanGSRendererUniforms& uniforms,
            const glm::vec3& background);

        VulkanContext* context_ = nullptr;
        bool initialized_ = false;
        VulkanGSRenderer renderer_;
        VulkanGSPipelineBuffers buffers_;
        ModelInputSnapshot uploaded_inputs_{};
        std::unique_ptr<ComposePipeline> compose_;
        VulkanContext::ExternalImage output_image_{};
        glm::ivec2 output_size_{0, 0};
        VkImageLayout output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t output_generation_ = 0;
    };

} // namespace lfs::vis
