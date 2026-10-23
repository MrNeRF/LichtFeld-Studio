/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/point_cloud_vulkan_renderer.hpp"
#include "rendering/viewport_interop_service.hpp"
#include "rendering/passes/vulkan_viewport_pass.hpp"
#include <SDL3/SDL.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <string_view>
#include <stdexcept>
#include <vector>

using namespace lfs::core;
using namespace lfs::vis;
static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
static void check(VkResult value) { require(value == VK_SUCCESS, "Vulkan contract operation failed"); }

int main(int argc, char** argv) try {
    Logger::get().init(LogLevel::Info);
    TensorBackendOptions options;
    options.rendering_vulkan_device = argc > 1 ? argv[1] : "0";
    require(bool(set_tensor_backend_options(options)), "Cannot select renderer GPU");
    require(bool(set_default_gpu_backend(GpuBackend::HIP)), "Cannot select HIP");
    require(SDL_InitSubSystem(SDL_INIT_VIDEO), SDL_GetError());
    struct Window {
        SDL_Window* value = SDL_CreateWindow("HIP secondary consumer contract", 96, 96, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
        ~Window() { if (value) SDL_DestroyWindow(value); SDL_QuitSubSystem(SDL_INIT_VIDEO); }
    } window;
    require(window.value != nullptr, SDL_GetError());
    {
        VulkanContext context;
        const bool initialized = context.init(window.value, 96, 96);
        require(initialized, context.lastError().c_str());
        auto& interop = context.tensorInterop();
        const bool shared = interop.shares_storage(GpuBackend::HIP);
        if (argc > 2) require(shared == (std::string_view(argv[2]) == "shared"), "Unexpected interop mode");
        {
            PointCloudVulkanRenderer renderer;
            auto allocate = interop.splat_allocator();
            auto positions = allocate({1, 3}, 1, DataType::Float32, "secondary viewer positions");
            auto colors = allocate({1, 3}, 1, DataType::Float32, "secondary viewer colors");
            positions.copy_from(Tensor::from_vector({0.f, 0.f, -2.f}, {1, 3}, Device::CPU));
            PointCloudVulkanRenderer::RenderRequest request;
            request.positions = &positions; request.colors = &colors;
            request.size = {64, 64}; request.voxel_size = .15f;
            request.focal_y = 32.f;
            glm::mat4 flip(1.f); flip[1][1] = -1.f;
            request.view_projection = flip * glm::perspective(glm::radians(90.f), 1.f, .1f, 10.f);
            for (unsigned frame = 0; frame < 8; ++frame) {
                const bool red = frame % 2 == 0;
                colors.copy_from(Tensor::from_vector(red ? std::vector<float>{1.f, 0.f, 0.f}
                                                        : std::vector<float>{0.f, 1.f, 0.f}, {1, 3}, Device::CPU));
                ++request.positions_revision; ++request.colors_revision;
                auto result = renderer.render(context, request, RenderTargetId{1});
                require(bool(result), result ? "" : result.error().c_str());
                auto image = renderer.readOutputImage(context, RenderTargetId{1}, true);
                require(bool(image), image ? "" : image.error().c_str());
                const auto bytes = (*image)->cpu().to_vector_uint8();
                require(bytes.size() == 64 * 64 * 4, "Point-cloud readback extent changed");
                unsigned pixels = 0;
                for (size_t i = 0; i < bytes.size(); i += 4) {
                    if (bytes[i + (red ? 0 : 1)] > 240) {
                        require(bytes[i + (red ? 1 : 0)] < 5, "Point-cloud read stale producer colors");
                        ++pixels;
                    }
                }
                require(pixels > 0, "Point-cloud consumer did not render HIP vertices");
            }
            if (shared)
                require(PointCloudOutputOwnershipTestAccess::hostVertexUploadBytes(renderer) == 0,
                        "Shared point-cloud fix introduced host vertex copies");
        }
        // Exercise the real reference-image service: mutate HIP after prepare,
        // before the later Vulkan frame submission. Its immutable publication
        // must still contain the completed generation rather than the new data.
        for (const auto size : {glm::ivec2{17, 13}, glm::ivec2{67, 35}}) {
            ViewportInteropService service;
            auto image = std::make_shared<Tensor>(Tensor::full({size_t(size.y), size_t(size.x), 3}, .25f, Device::GPU));
            service.setSceneImage(image, size, false, 1);
            service.prepareFrame(context, false);
            image->fill_(.75f);
            VulkanViewportPassParams params;
            service.bindViewportParams(params, context.currentFrameSlot(), false, false);
            require(params.external_scene_image != VK_NULL_HANDLE, "Reference image was not published");
            const auto device = context.device();
            VkCommandPool pool{};
            VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pi.queueFamilyIndex = context.graphicsQueueFamily(); check(vkCreateCommandPool(device, &pi, nullptr, &pool));
            VkCommandBuffer command{};
            VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ci.commandPool = pool; ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ci.commandBufferCount = 1;
            check(vkAllocateCommandBuffers(device, &ci, &command));
            const size_t count = size_t(size.x) * size.y * 4;
            VkBuffer buffer{}; VmaAllocation allocation{}; VmaAllocationInfo mapped{};
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = count; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VmaAllocationCreateInfo ai{}; ai.usage = VMA_MEMORY_USAGE_AUTO;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            check(vmaCreateBuffer(context.allocator(), &bi, &ai, &buffer, &allocation, &mapped));
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(command, &begin));
            service.recordFrameBarriers(command, context);
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = params.external_scene_image; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            VkBufferImageCopy copy{}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {uint32_t(size.x), uint32_t(size.y), 1};
            vkCmdCopyImageToBuffer(command, params.external_scene_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
            std::swap(barrier.oldLayout, barrier.newLayout);
            std::swap(barrier.srcAccessMask, barrier.dstAccessMask);
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
            check(vkEndCommandBuffer(command));
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
            {
                std::lock_guard lock(context.graphicsQueueMutex());
                check(vkQueueSubmit(context.graphicsQueue(), 1, &submit, VK_NULL_HANDLE));
                check(vkQueueWaitIdle(context.graphicsQueue()));
            }
            check(vmaInvalidateAllocation(context.allocator(), allocation, 0, count));
            const auto* bytes = static_cast<const uint8_t*>(mapped.pMappedData);
            for (size_t i = 0; i < count; ++i)
                require(i % 4 == 3 ? bytes[i] == 255 : bytes[i] >= 63 && bytes[i] <= 64,
                        "Reference image lost its immutable prepared generation");
            vmaDestroyBuffer(context.allocator(), buffer, allocation);
            vkDestroyCommandPool(device, pool, nullptr);
        }
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)), "Secondary viewer retained HIP Tensor owners");
    std::puts("PASS real HIP point-cloud and reference-image consumers: current pixels, immutable publication, resize and teardown");
} catch (const std::exception& error) { std::fprintf(stderr, "SECONDARY VIEWER FAILED: %s\n", error.what()); return 1; }
