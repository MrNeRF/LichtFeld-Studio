/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gs_renderer.h"

namespace {
    alignas(4) constexpr std::uint8_t stbn_data[] = {
#include "stbn_data.inc"
    };
    static_assert(sizeof(stbn_data) == 128u * 128u * 64u);
} // namespace

const _VulkanBuffer& VulkanGSRenderer::prepareStochasticNoise() {
    if (stochastic_noise_.buffer != VK_NULL_HANDLE)
        return stochastic_noise_;
    if (isCommandBatchInProgress())
        lfs::rendering::throw_renderer_contract(
            "Prepare STBN before recording the viewport batch", LFS_SOURCE_SITE_CURRENT());
    try {
        stochastic_noise_.label = "stochastic.stbn";
        createBuffer(sizeof(stbn_data), stochastic_noise_);
        {
            DEVICE_GUARD;
            const lfs::rendering::vulkan::DeclaredAccess access{
                .buffer = &stochastic_noise_,
                .use = lfs::rendering::vulkan::BufferUse::TransferWrite};
            planTransfer(std::span{&access, 1});
            constexpr VkDeviceSize chunk_size = 65536;
            for (VkDeviceSize offset = 0; offset < sizeof(stbn_data); offset += chunk_size)
                vkCmdUpdateBuffer(activeCommandBuffer(), stochastic_noise_.buffer, offset,
                                  chunk_size, stbn_data + offset);
        }
    } catch (...) {
        destroyBuffer(stochastic_noise_);
        throw;
    }
    return stochastic_noise_;
}

void VulkanGSRenderer::executeStochasticReuse(const VulkanGSRendererUniforms& uniforms,
                                              VulkanGSPipelineBuffers& buffers,
                                              VulkanGSStochasticBuffers& history) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::StochasticReuse> timer(this);
    using lfs::rendering::vulkan::BufferUse;
    const auto pixels = static_cast<std::size_t>(uniforms.image_width) * uniforms.image_height;
    history.occluders.deviceBuffer.label = "stochastic.history_occluders";
    history.outgoing_weights.deviceBuffer.label = "stochastic.outgoing_weights";
    resizeDeviceBuffer(history.occluders, pixels * VULKAN_GS_STOCHASTIC_HISTORY_SIZE * 2);
    resizeDeviceBuffer(history.outgoing_weights, pixels * 4);
    executeCompute({{pixels, 256}}, &uniforms, sizeof(uniforms), pipeline_stochastic_reuse,
                   std::vector<TaggedBinding>{
                       {buffers.n_contributors.deviceBuffer, BufferUse::ComputeRead},
                       {buffers.pixel_depth_weight.deviceBuffer, BufferUse::ComputeRead},
                       {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
                       {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
                       {history.occluders.deviceBuffer, BufferUse::ComputeReadWrite},
                       {history.outgoing_weights.deviceBuffer, BufferUse::ComputeWrite},
                   });
}

const _VulkanBuffer& VulkanGSRenderer::executeStochasticResolve(
    const VulkanGSStochasticResolveUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    VulkanGSStochasticBuffers& history,
    const std::size_t output_index) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::StochasticResolve> timer(this);
    using lfs::rendering::vulkan::BufferUse;
    const auto pixels = static_cast<std::size_t>(uniforms.width) * uniforms.height;
    for (std::size_t i = 0; i < history.color.size(); ++i) {
        history.color[i].deviceBuffer.label = i == 0 ? "stochastic.color_0" : "stochastic.color_1";
        resizeDeviceBuffer(history.color[i], pixels * 4);
    }
    if (uniforms.mode == 2u) {
        history.rest_average.deviceBuffer.label = "stochastic.rest_average";
        resizeDeviceBuffer(history.rest_average, pixels * 4);
    }
    const auto& weights = uniforms.mode == 2u ? history.outgoing_weights.deviceBuffer
                                              : buffers.pixel_state.deviceBuffer;
    const auto& rest_average = uniforms.mode == 2u ? history.rest_average.deviceBuffer
                                                   : buffers.pixel_state.deviceBuffer;
    executeCompute({{uniforms.width, 16}, {uniforms.height, 16}},
                   &uniforms, sizeof(uniforms), pipeline_stochastic_resolve,
                   std::vector<TaggedBinding>{
                       {buffers.pixel_state.deviceBuffer, BufferUse::ComputeRead},
                       {buffers.pixel_depth.deviceBuffer, BufferUse::ComputeRead},
                       {weights, BufferUse::ComputeRead},
                       {history.color[output_index ^ 1u].deviceBuffer, BufferUse::ComputeRead},
                       {history.color[output_index].deviceBuffer, BufferUse::ComputeWrite},
                       {rest_average, uniforms.mode == 2u ? BufferUse::ComputeReadWrite : BufferUse::ComputeRead},
                   });
    return history.color[output_index].deviceBuffer;
}
