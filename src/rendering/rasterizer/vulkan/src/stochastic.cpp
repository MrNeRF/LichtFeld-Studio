/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gs_renderer.h"

void VulkanGSRenderer::executeStochasticReuse(const VulkanGSRendererUniforms& uniforms,
                                              VulkanGSPipelineBuffers& buffers,
                                              VulkanGSStochasticBuffers& history) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::StochasticReuse> timer(this);
    using lfs::rendering::vulkan::BufferUse;
    const auto pixels = static_cast<std::size_t>(uniforms.image_width) * uniforms.image_height;
    history.ids_depths.deviceBuffer.label = "stochastic.history_ids_depths";
    history.occlusion.deviceBuffer.label = "stochastic.history_occlusion";
    history.outgoing_weights.deviceBuffer.label = "stochastic.outgoing_weights";
    resizeDeviceBuffer(history.ids_depths, pixels * VULKAN_GS_STOCHASTIC_HISTORY_SIZE * 2);
    resizeDeviceBuffer(history.occlusion, pixels * VULKAN_GS_STOCHASTIC_HISTORY_SIZE * 4);
    resizeDeviceBuffer(history.outgoing_weights, pixels * 4);
    executeCompute({{pixels, 256}}, &uniforms, sizeof(uniforms), pipeline_stochastic_reuse,
                   std::vector<TaggedBinding>{
                       {buffers.n_contributors.deviceBuffer, BufferUse::ComputeRead},
                       {buffers.pixel_depth_weight.deviceBuffer, BufferUse::ComputeRead},
                       {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
                       {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
                       {history.ids_depths.deviceBuffer, BufferUse::ComputeReadWrite},
                       {history.occlusion.deviceBuffer, BufferUse::ComputeReadWrite},
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
    const auto& weights = uniforms.mode == 2u ? history.outgoing_weights.deviceBuffer
                                              : buffers.pixel_state.deviceBuffer;
    executeCompute({{uniforms.width, 16}, {uniforms.height, 16}},
                   &uniforms, sizeof(uniforms), pipeline_stochastic_resolve,
                   std::vector<TaggedBinding>{
                       {buffers.pixel_state.deviceBuffer, BufferUse::ComputeRead},
                       {buffers.pixel_depth.deviceBuffer, BufferUse::ComputeRead},
                       {weights, BufferUse::ComputeRead},
                       {history.color[output_index ^ 1u].deviceBuffer, BufferUse::ComputeRead},
                       {history.color[output_index].deviceBuffer, BufferUse::ComputeWrite},
                   });
    return history.color[output_index].deviceBuffer;
}
