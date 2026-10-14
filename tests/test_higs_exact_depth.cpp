/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/executable_path.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include "rendering/vksplat_viewport_renderer.hpp"
#include "rendering/vulkan_external_tensor.hpp"
#include "window/vulkan_context.hpp"
#include <SDL3/SDL.h>
#include <array>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

TEST(HiGSExactDepth, MedianSurvivesHalfPrecisionBatchBoundary) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
        GTEST_SKIP() << SDL_GetError();
    struct VideoGuard {
        ~VideoGuard() { SDL_QuitSubSystem(SDL_INIT_VIDEO); }
    } video_guard;
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
        SDL_CreateWindow("HiGS exact depth regression", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN),
        SDL_DestroyWindow);
    ASSERT_TRUE(window) << SDL_GetError();
    lfs::vis::VulkanContext context;
    ASSERT_TRUE(context.init(window.get(), 64, 64)) << context.lastError();
    lfs::rendering::setExpectedVulkanDeviceUuid(context.deviceUUID());
    using lfs::core::Device;
    using lfs::core::Tensor;
    constexpr size_t count = 1025;
    for (const bool overlays : {false, true}) {
        SCOPED_TRACE(overlays);
        for (const glm::ivec2 size : {glm::ivec2(64, 64), glm::ivec2(65, 67)}) {
            SCOPED_TRACE(::testing::Message() << size.x << 'x' << size.y);
            lfs::rendering::ViewportRenderRequest request;
            request.frame_view.size = size;
            request.sh_degree = 0;
            request.overlay.cursor = {.enabled = overlays, .cursor = {-1000.0f, -1000.0f}, .radius = 1.0f};
            request.require_exact_depth = true;
            const auto k = request.frame_view.getCameraIntrinsics();
            std::vector<float> means(count * 3), rotations(count * 4, 0.0f), opacity(count);
            for (size_t i = 0; i < count; ++i) {
                // Only the first and last sources cover the queried pixel. The
                // first 1024-source batch leaves exact T=.50005, rounded to .5
                // in the FP16 color partial. The next batch must set depth to 6.
                const bool filler = i > 0 && i + 1 < count;
                const float z = i == 0 ? 4.0f : filler ? 4.1f
                                                       : 6.0f;
                const float pixel_x = static_cast<float>(size.x / 2) + (filler ? 16.5f : 0.5f);
                const float pixel_y = static_cast<float>(size.y / 2) + 0.5f;
                means[i * 3] = (pixel_x - k.center_x) * z / k.focal_x;
                means[i * 3 + 1] = (k.center_y - pixel_y) * z / k.focal_y;
                means[i * 3 + 2] = -z;
                rotations[i * 4] = 1.0f;
                const float alpha = i == 0 ? 0.49995f : 0.01f;
                opacity[i] = std::log(alpha / (1.0f - alpha));
            }
            const auto external = [&](Tensor values) {
                auto allocation = lfs::vis::makeVulkanExternalTensor(context, values.shape(), values.dtype(), values.numel(), "depth-fixture");
                if (!allocation)
                    throw std::runtime_error(allocation.error());
                allocation->copy_from(values);
                return std::move(*allocation);
            };
            lfs::core::SplatData model(
                0,
                external(Tensor::from_vector(means, {count, size_t{3}}, Device::CPU)),
                external(Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CPU)),
                Tensor::zeros({count, size_t{0}, size_t{3}}, Device::CUDA),
                external(Tensor::full({count, size_t{3}}, std::log(0.005f), Device::CPU)),
                external(Tensor::from_vector(rotations, {count, size_t{4}}, Device::CPU)),
                external(Tensor::from_vector(opacity, {count, size_t{1}}, Device::CPU)), 1.0f);
            lfs::vis::VksplatViewportRenderer renderer;
            request.require_exact_depth = false;
            for (int frame = 0; frame < 2; ++frame) {
                auto rendered = renderer.render(context, model, request, false,
                                                lfs::vis::VksplatViewportRenderer::OutputSlot::Main, false, true);
                ASSERT_TRUE(rendered) << rendered.error();
            }
            auto approximate = renderer.sampleDepthAtPixel(context, {.pixel = size / 2, .source_size = size});
            ASSERT_TRUE(approximate) << approximate.error();
            EXPECT_NEAR(*approximate, 4.0f, 1e-4f); // Confirms the HiGS approximate path is active.
            auto color = renderer.readOutputImageRgb8(context, lfs::vis::VksplatViewportRenderer::OutputSlot::Main);
            ASSERT_TRUE(color) << color.error();
            const auto* data = (*color)->ptr<uint8_t>();
            const std::vector<uint8_t> reference(data, data + (*color)->numel());
            request.require_exact_depth = true;
            for (const bool masked : {false, true}) {
                SCOPED_TRACE(masked);
                if (masked) {
                    const size_t columns = (size.x + 3) / 4;
                    const size_t tiles = columns * ((size.y + 3) / 4);
                    request.exact_depth_sample_mask.assign((tiles + 31) / 32, 0);
                    const size_t bit = size_t((size.y / 2) / 4) * columns + (size.x / 2) / 4;
                    request.exact_depth_sample_mask[bit / 32] |= 1u << (bit % 32);
                }
                // Deterministic export skips the cold legacy warmup but still uses
                // HiGS; depth capture would bypass HiGS and hide this regression.
                for (int frame = 0; frame < 2; ++frame) {
                    auto rendered = renderer.render(context, model, request, false,
                                                    lfs::vis::VksplatViewportRenderer::OutputSlot::Main, false, true);
                    ASSERT_TRUE(rendered) << rendered.error();
                    auto depth = renderer.sampleDepthAtPixel(context, {.pixel = size / 2, .source_size = size});
                    ASSERT_TRUE(depth) << depth.error();
                    EXPECT_NEAR(*depth, 6.0f, 1e-4f);
                    auto exact_color = renderer.readOutputImageRgb8(context, lfs::vis::VksplatViewportRenderer::OutputSlot::Main);
                    ASSERT_TRUE(exact_color) << exact_color.error();
                    const auto* pixels = (*exact_color)->ptr<uint8_t>();
                    EXPECT_EQ(std::vector<uint8_t>(pixels, pixels + (*exact_color)->numel()), reference);
                }
            }
        }
    }
}

// Exercise the production compose shader directly: two tiny waves suffice to
// test persisted FP32 state, without allocating millions of filler splats.
TEST(HiGSExactDepth, TransmittanceSurvivesDepthWaveBoundary) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
        GTEST_SKIP() << SDL_GetError();
    struct VideoGuard {
        ~VideoGuard() { SDL_QuitSubSystem(SDL_INIT_VIDEO); }
    } video_guard;
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
        SDL_CreateWindow("HiGS depth wave regression", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN),
        SDL_DestroyWindow);
    ASSERT_TRUE(window) << SDL_GetError();
    lfs::vis::VulkanContext context;
    ASSERT_TRUE(context.init(window.get(), 64, 64)) << context.lastError();
    const auto device = context.device();
    struct Resources {
        VkDevice device;
        std::array<VkBuffer, 13> buffers{};
        std::array<VkDeviceMemory, 13> memories{};
        std::array<void*, 13> mapped{};
        VkDescriptorSetLayout descriptors{};
        VkPipelineLayout layout{};
        VkDescriptorPool pool{};
        VkShaderModule shader{};
        VkPipeline pipeline{};
        VkCommandPool commands{};
        ~Resources() {
            vkDeviceWaitIdle(device);
            vkDestroyCommandPool(device, commands, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyShaderModule(device, shader, nullptr);
            vkDestroyDescriptorPool(device, pool, nullptr);
            vkDestroyPipelineLayout(device, layout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
            for (size_t i = 0; i < buffers.size(); ++i) {
                if (mapped[i])
                    vkUnmapMemory(device, memories[i]);
                vkDestroyBuffer(device, buffers[i], nullptr);
                vkFreeMemory(device, memories[i], nullptr);
            }
        }
    } owned{device};
    constexpr std::array<VkDeviceSize, 13> sizes{
        8, 8, 4, 16, 32, 24, 8, 32 * 64 * 8, 4, 64 * 16, 64 * 4, 64 * 4, 4};
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(context.physicalDevice(), &memory);
    std::array<VkDescriptorSetLayoutBinding, 13> bindings{};
    std::array<VkDescriptorBufferInfo, 13> infos{};
    for (uint32_t i = 0; i < sizes.size(); ++i) {
        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = sizes[i];
        buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        ASSERT_EQ(vkCreateBuffer(device, &buffer, nullptr, &owned.buffers[i]), VK_SUCCESS);
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, owned.buffers[i], &requirements);
        uint32_t type = memory.memoryTypeCount;
        for (uint32_t j = 0; j < memory.memoryTypeCount; ++j)
            if ((requirements.memoryTypeBits & (1u << j)) &&
                (memory.memoryTypes[j].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                type = j;
                break;
            }
        ASSERT_LT(type, memory.memoryTypeCount);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        ASSERT_EQ(vkAllocateMemory(device, &allocation, nullptr, &owned.memories[i]), VK_SUCCESS);
        ASSERT_EQ(vkBindBufferMemory(device, owned.buffers[i], owned.memories[i], 0), VK_SUCCESS);
        ASSERT_EQ(vkMapMemory(device, owned.memories[i], 0, VK_WHOLE_SIZE, 0, &owned.mapped[i]), VK_SUCCESS);
        std::memset(owned.mapped[i], 0, sizes[i]);
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        infos[i] = {owned.buffers[i], 0, sizes[i]};
    }
    VkDescriptorSetLayoutCreateInfo descriptor_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_layout.bindingCount = bindings.size();
    descriptor_layout.pBindings = bindings.data();
    ASSERT_EQ(vkCreateDescriptorSetLayout(device, &descriptor_layout, nullptr, &owned.descriptors), VK_SUCCESS);
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VulkanGSRendererUniforms)};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &owned.descriptors;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    ASSERT_EQ(vkCreatePipelineLayout(device, &layout, nullptr, &owned.layout), VK_SUCCESS);
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 13};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 1;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &pool_size;
    ASSERT_EQ(vkCreateDescriptorPool(device, &pool, nullptr, &owned.pool), VK_SUCCESS);
    VkDescriptorSetAllocateInfo sets{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sets.descriptorPool = owned.pool;
    sets.descriptorSetCount = 1;
    sets.pSetLayouts = &owned.descriptors;
    VkDescriptorSet set;
    ASSERT_EQ(vkAllocateDescriptorSets(device, &sets, &set), VK_SUCCESS);
    std::array<VkWriteDescriptorSet, 13> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
    auto path = lfs::core::getResourceBaseDir() /
                "shaders/vulkan_rasterizer/generated/macro_compose.spv";
#ifdef LFS_VULKAN_RASTERIZER_DEV_SPV_DIR
    if (!std::filesystem::exists(path))
        path = std::filesystem::path(LFS_VULKAN_RASTERIZER_DEV_SPV_DIR) / "generated/macro_compose.spv";
#endif
#ifdef PROJECT_ROOT_PATH
    if (!std::filesystem::exists(path))
        path = std::filesystem::path(PROJECT_ROOT_PATH) /
               "src/rendering/rasterizer/vulkan/shader/generated/macro_compose.spv";
#endif
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(input) << path;
    const auto bytes = static_cast<size_t>(input.tellg());
    ASSERT_EQ(bytes % sizeof(uint32_t), 0u);
    std::vector<uint32_t> code(bytes / sizeof(uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(code.data()), bytes);
    ASSERT_TRUE(input);
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = bytes;
    shader.pCode = code.data();
    ASSERT_EQ(vkCreateShaderModule(device, &shader, nullptr, &owned.shader), VK_SUCCESS);
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = owned.shader;
    pipeline.stage.pName = "main";
    pipeline.layout = owned.layout;
    ASSERT_EQ(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &owned.pipeline), VK_SUCCESS);
    VkCommandPoolCreateInfo command_pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool.queueFamilyIndex = context.graphicsQueueFamily();
    command_pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ASSERT_EQ(vkCreateCommandPool(device, &command_pool, nullptr, &owned.commands), VK_SUCCESS);
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = owned.commands;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command;
    ASSERT_EQ(vkAllocateCommandBuffers(device, &command_info, &command), VK_SUCCESS);
    const int32_t ids[]{0, 1};
    const float xy[]{4, 4, 4, 4};
    const float conics[]{1, 0, 1, 0.49995f, 1, 0, 1, 0.01f};
    const float depths[]{4, 6};
    std::memcpy(owned.mapped[0], ids, sizeof(ids));
    std::memcpy(owned.mapped[3], xy, sizeof(xy));
    std::memcpy(owned.mapped[4], conics, sizeof(conics));
    std::memcpy(owned.mapped[6], depths, sizeof(depths));
    *static_cast<int32_t*>(owned.mapped[2]) = 1;
    *static_cast<uint32_t*>(owned.mapped[8]) = 1;
    *static_cast<uint32_t*>(owned.mapped[12]) = 1u << 3; // 4x4 sample tile containing (4,4).
    VulkanGSRendererUniforms uniforms{};
    uniforms.image_width = uniforms.image_height = 8;
    uniforms.grid_width = uniforms.grid_height = 1;
    uniforms.num_splats = 2;
    for (const bool masked : {false, true}) {
        SCOPED_TRACE(masked);
        uniforms.mip_filter = masked ? 10u : 2u;
        for (uint32_t wave = 0; wave < 2; ++wave) {
            uniforms.depth_wave = wave;
            auto* ranges = static_cast<int32_t*>(owned.mapped[1]);
            ranges[0] = wave;
            ranges[1] = wave + 1;
            // Deliberately rounded colour state; depth must use its own scratch.
            auto* partials = static_cast<uint16_t*>(owned.mapped[7]);
            for (size_t pixel = 0; pixel < 64; ++pixel)
                partials[pixel * 4 + 3] = wave == 0 ? 0x3800 : 0x3bec; // half(.5), half(.99)
            ASSERT_EQ(vkResetCommandBuffer(command, 0), VK_SUCCESS);
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            ASSERT_EQ(vkBeginCommandBuffer(command, &begin), VK_SUCCESS);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, owned.pipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, owned.layout, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(command, owned.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uniforms), &uniforms);
            vkCmdDispatch(command, 1, 1, 1);
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &barrier, 0, nullptr, 0, nullptr);
            ASSERT_EQ(vkEndCommandBuffer(command), VK_SUCCESS);
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            ASSERT_EQ(vkQueueSubmit(context.graphicsQueue(), 1, &submit, VK_NULL_HANDLE), VK_SUCCESS);
            ASSERT_EQ(vkQueueWaitIdle(context.graphicsQueue()), VK_SUCCESS);
            const auto median = static_cast<float*>(owned.mapped[10])[36];
            const auto transmittance = static_cast<float*>(owned.mapped[11])[36];
            if (wave == 0) {
                EXPECT_GT(median, 1e9f);
                EXPECT_NEAR(transmittance, 0.50005f, 1e-6f);
            } else {
                EXPECT_NEAR(median, 6, 1e-4f);
                EXPECT_NEAR(transmittance, 0.50005f * 0.99f, 1e-6f);
            }
        }
    }
}
