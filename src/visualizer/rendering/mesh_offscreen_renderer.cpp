/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_offscreen_renderer.hpp"

#include "passes/vulkan_mesh_pass.hpp"
#include "rendering/vulkan_result.hpp"
#include "window/vulkan_context.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <limits>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>

namespace lfs::vis {

    namespace {

        constexpr VkFormat kMeshLayerColorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        constexpr VkFormat kMeshLayerDepthFormat = VK_FORMAT_D32_SFLOAT;

        [[nodiscard]] std::unexpected<std::string> vkUnexpected(
            const char* operation,
            const VkResult result) {
            return std::unexpected(std::format(
                "{} failed: {} ({})",
                operation,
                lfs::rendering::vkResultToString(result),
                static_cast<int>(result)));
        }

    } // namespace

    float linearizeMeshViewDepth(const float z_ndc, const glm::mat4& projection) noexcept {
        if (projection[2][3] == -1.0f) {
            return projection[3][2] / (z_ndc + projection[2][2]);
        }
        return -(z_ndc - projection[3][2]) / projection[2][2];
    }

    struct MeshOffscreenRenderer::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;

        VulkanMeshPass mesh_pass;
        bool initialized = false;

        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool submission_in_flight = false;

        VkImage color_image = VK_NULL_HANDLE;
        VmaAllocation color_allocation = VK_NULL_HANDLE;
        VkImageView color_view = VK_NULL_HANDLE;
        VkImage depth_image = VK_NULL_HANDLE;
        VmaAllocation depth_allocation = VK_NULL_HANDLE;
        VkImageView depth_view = VK_NULL_HANDLE;
        VkBuffer readback_buffer = VK_NULL_HANDLE;
        VmaAllocation readback_allocation = VK_NULL_HANDLE;
        VkDeviceSize readback_size = 0;
        int width = 0;
        int height = 0;

        ~Impl() { shutdown(); }

        [[nodiscard]] std::expected<void, std::string> initialize(VulkanContext& requested_context) {
            if (initialized && context == &requested_context) {
                return {};
            }
            if (context != nullptr && context != &requested_context) {
                shutdown();
            }

            context = &requested_context;
            device = requested_context.device();
            allocator = requested_context.allocator();
            graphics_queue = requested_context.graphicsQueue();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE) {
                shutdown();
                return std::unexpected(
                    "Mesh offscreen rendering requires a live Vulkan device, allocator, and graphics queue");
            }

            if (!mesh_pass.init(requested_context, kMeshLayerColorFormat, kMeshLayerDepthFormat)) {
                shutdown();
                return std::unexpected("VulkanMeshPass initialization failed for video export");
            }

            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = requested_context.graphicsQueueFamily();
            VkResult result = vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);
            if (result != VK_SUCCESS) {
                const auto error = vkUnexpected("vkCreateCommandPool(mesh offscreen)", result);
                shutdown();
                return error;
            }
            requested_context.setDebugObjectName(
                VK_OBJECT_TYPE_COMMAND_POOL, command_pool, "mesh.offscreen.command_pool");

            VkCommandBufferAllocateInfo command_info{};
            command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            command_info.commandPool = command_pool;
            command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command_info.commandBufferCount = 1;
            result = vkAllocateCommandBuffers(device, &command_info, &command_buffer);
            if (result != VK_SUCCESS) {
                const auto error = vkUnexpected("vkAllocateCommandBuffers(mesh offscreen)", result);
                shutdown();
                return error;
            }
            requested_context.setDebugObjectName(
                VK_OBJECT_TYPE_COMMAND_BUFFER, command_buffer, "mesh.offscreen.command_buffer");

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            result = vkCreateFence(device, &fence_info, nullptr, &fence);
            if (result != VK_SUCCESS) {
                const auto error = vkUnexpected("vkCreateFence(mesh offscreen)", result);
                shutdown();
                return error;
            }
            requested_context.setDebugObjectName(
                VK_OBJECT_TYPE_FENCE, fence, "mesh.offscreen.fence");

            initialized = true;
            return {};
        }

        void destroyTargets() {
            if (context != nullptr) {
                context->imageBarriers().forgetImage(color_image);
                context->imageBarriers().forgetImage(depth_image);
            }
            if (device != VK_NULL_HANDLE && color_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, color_view, nullptr);
            }
            if (device != VK_NULL_HANDLE && depth_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, depth_view, nullptr);
            }
            if (allocator != VK_NULL_HANDLE && color_image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator, color_image, color_allocation);
            }
            if (allocator != VK_NULL_HANDLE && depth_image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator, depth_image, depth_allocation);
            }
            if (allocator != VK_NULL_HANDLE && readback_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, readback_buffer, readback_allocation);
            }

            color_image = VK_NULL_HANDLE;
            color_allocation = VK_NULL_HANDLE;
            color_view = VK_NULL_HANDLE;
            depth_image = VK_NULL_HANDLE;
            depth_allocation = VK_NULL_HANDLE;
            depth_view = VK_NULL_HANDLE;
            readback_buffer = VK_NULL_HANDLE;
            readback_allocation = VK_NULL_HANDLE;
            readback_size = 0;
            width = 0;
            height = 0;
        }

        void shutdown() {
            if (submission_in_flight && device != VK_NULL_HANDLE && fence != VK_NULL_HANDLE) {
                const VkResult wait_result =
                    vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
                submission_in_flight = wait_result != VK_SUCCESS;
            }

            mesh_pass.shutdown();
            destroyTargets();

            if (device != VK_NULL_HANDLE && fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
            }
            if (device != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, command_pool, nullptr);
            }

            fence = VK_NULL_HANDLE;
            command_buffer = VK_NULL_HANDLE;
            command_pool = VK_NULL_HANDLE;
            submission_in_flight = false;
            initialized = false;
            graphics_queue = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            context = nullptr;
        }

        [[nodiscard]] std::expected<void, std::string> createImage(
            const VkFormat format,
            const VkImageUsageFlags usage,
            const VkImageAspectFlags aspect,
            const char* debug_name,
            VkImage& image,
            VmaAllocation& allocation,
            VkImageView& view) {
            VkImageCreateInfo image_info{};
            image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.format = format;
            image_info.extent = {
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height),
                1};
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.usage = usage;
            image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            const VkResult image_result = vmaCreateImage(
                allocator, &image_info, &allocation_info, &image, &allocation, nullptr);
            if (image_result != VK_SUCCESS) {
                return vkUnexpected("vmaCreateImage(mesh offscreen)", image_result);
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_IMAGE, image, debug_name);

            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = format;
            view_info.subresourceRange.aspectMask = aspect;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.layerCount = 1;
            const VkResult view_result = vkCreateImageView(device, &view_info, nullptr, &view);
            if (view_result != VK_SUCCESS) {
                return vkUnexpected("vkCreateImageView(mesh offscreen)", view_result);
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, view, debug_name);
            return {};
        }

        [[nodiscard]] std::expected<void, std::string> ensureTargets(
            const int requested_width,
            const int requested_height) {
            if (color_image != VK_NULL_HANDLE && depth_image != VK_NULL_HANDLE &&
                readback_buffer != VK_NULL_HANDLE && width == requested_width &&
                height == requested_height) {
                return {};
            }

            destroyTargets();
            width = requested_width;
            height = requested_height;

            const auto pixel_count = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
            constexpr std::uint64_t bytes_per_pixel = 5u * sizeof(float);
            if (pixel_count > std::numeric_limits<VkDeviceSize>::max() / bytes_per_pixel) {
                destroyTargets();
                return std::unexpected("Mesh offscreen readback size overflow");
            }

            auto color_result = createImage(
                kMeshLayerColorFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                "mesh.offscreen.color",
                color_image,
                color_allocation,
                color_view);
            if (!color_result) {
                const auto error = color_result.error();
                destroyTargets();
                return std::unexpected(error);
            }

            auto depth_result = createImage(
                kMeshLayerDepthFormat,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                "mesh.offscreen.depth",
                depth_image,
                depth_allocation,
                depth_view);
            if (!depth_result) {
                const auto error = depth_result.error();
                destroyTargets();
                return std::unexpected(error);
            }

            readback_size = static_cast<VkDeviceSize>(pixel_count * bytes_per_pixel);
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = readback_size;
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            const VkResult buffer_result = vmaCreateBuffer(
                allocator,
                &buffer_info,
                &allocation_info,
                &readback_buffer,
                &readback_allocation,
                nullptr);
            if (buffer_result != VK_SUCCESS) {
                const auto error = vkUnexpected("vmaCreateBuffer(mesh offscreen readback)", buffer_result);
                destroyTargets();
                return error;
            }
            context->setDebugObjectName(
                VK_OBJECT_TYPE_BUFFER, readback_buffer, "mesh.offscreen.readback");

            context->imageBarriers().registerImage(
                color_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, true);
            context->imageBarriers().registerImage(
                depth_image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, true);
            return {};
        }

        [[nodiscard]] std::expected<MeshLayer, std::string> render(
            VulkanContext& requested_context,
            const VulkanMeshPassParams& params,
            const glm::mat4& projection,
            const int requested_width,
            const int requested_height) {
            if (requested_width <= 0 || requested_height <= 0) {
                return std::unexpected("Mesh offscreen dimensions must be positive");
            }

            auto init_result = initialize(requested_context);
            if (!init_result) {
                return std::unexpected(init_result.error());
            }
            auto target_result = ensureTargets(requested_width, requested_height);
            if (!target_result) {
                return std::unexpected(target_result.error());
            }

            mesh_pass.prepare(requested_context, params);

            if (submission_in_flight) {
                const VkResult wait_result =
                    vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
                if (wait_result != VK_SUCCESS) {
                    return vkUnexpected("vkWaitForFences(mesh offscreen previous frame)", wait_result);
                }
                submission_in_flight = false;
            }

            VkResult result = vkResetCommandPool(device, command_pool, 0);
            if (result != VK_SUCCESS) {
                return vkUnexpected("vkResetCommandPool(mesh offscreen)", result);
            }

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(command_buffer, &begin_info);
            if (result != VK_SUCCESS) {
                return vkUnexpected("vkBeginCommandBuffer(mesh offscreen)", result);
            }

            const VkImageLayout previous_color_layout = context->imageBarriers().imageLayout(color_image);
            const VkImageLayout previous_depth_layout = context->imageBarriers().imageLayout(depth_image);
            const auto restoreTrackedLayouts = [&]() {
                context->imageBarriers().registerImage(
                    color_image, VK_IMAGE_ASPECT_COLOR_BIT, previous_color_layout, true);
                context->imageBarriers().registerImage(
                    depth_image, VK_IMAGE_ASPECT_DEPTH_BIT, previous_depth_layout, true);
            };

            context->imageBarriers().transitionImage(
                command_buffer,
                color_image,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            context->imageBarriers().transitionImage(
                command_buffer,
                depth_image,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

            VkClearValue color_clear{};
            color_clear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
            VkRenderingAttachmentInfo color_attachment{};
            color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachment.imageView = color_view;
            color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_attachment.clearValue = color_clear;

            VkClearValue depth_clear{};
            depth_clear.depthStencil = {1.0f, 0};
            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = depth_view;
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue = depth_clear;

            const VkRect2D render_area{
                .offset = {0, 0},
                .extent = {
                    static_cast<std::uint32_t>(requested_width),
                    static_cast<std::uint32_t>(requested_height)}};
            VkRenderingInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering_info.renderArea = render_area;
            rendering_info.layerCount = 1;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &color_attachment;
            rendering_info.pDepthAttachment = &depth_attachment;
            vkCmdBeginRendering(command_buffer, &rendering_info);
            mesh_pass.record(command_buffer, render_area, params);
            vkCmdEndRendering(command_buffer);

            context->imageBarriers().transitionImage(
                command_buffer,
                color_image,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            context->imageBarriers().transitionImage(
                command_buffer,
                depth_image,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

            const VkDeviceSize pixel_count =
                static_cast<VkDeviceSize>(requested_width) * static_cast<VkDeviceSize>(requested_height);
            const VkDeviceSize color_bytes = pixel_count * 4u * sizeof(float);
            VkBufferImageCopy color_copy{};
            color_copy.bufferOffset = 0;
            color_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            color_copy.imageSubresource.layerCount = 1;
            color_copy.imageExtent = {
                static_cast<std::uint32_t>(requested_width),
                static_cast<std::uint32_t>(requested_height),
                1};
            vkCmdCopyImageToBuffer(
                command_buffer,
                color_image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                readback_buffer,
                1,
                &color_copy);

            VkBufferImageCopy depth_copy{};
            depth_copy.bufferOffset = color_bytes;
            depth_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            depth_copy.imageSubresource.layerCount = 1;
            depth_copy.imageExtent = color_copy.imageExtent;
            vkCmdCopyImageToBuffer(
                command_buffer,
                depth_image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                readback_buffer,
                1,
                &depth_copy);

            result = vkEndCommandBuffer(command_buffer);
            if (result != VK_SUCCESS) {
                restoreTrackedLayouts();
                return vkUnexpected("vkEndCommandBuffer(mesh offscreen)", result);
            }

            result = vkResetFences(device, 1, &fence);
            if (result != VK_SUCCESS) {
                restoreTrackedLayouts();
                return vkUnexpected("vkResetFences(mesh offscreen)", result);
            }

            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            result = vkQueueSubmit(graphics_queue, 1, &submit_info, fence);
            if (result != VK_SUCCESS) {
                restoreTrackedLayouts();
                return vkUnexpected("vkQueueSubmit(mesh offscreen)", result);
            }
            submission_in_flight = true;

            result = vkWaitForFences(
                device, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
            if (result != VK_SUCCESS) {
                return vkUnexpected("vkWaitForFences(mesh offscreen)", result);
            }
            submission_in_flight = false;

            const auto count = static_cast<std::size_t>(pixel_count);
            std::vector<float> rgba(4u * count);
            std::vector<float> view_depth(count);
            void* mapped = nullptr;
            result = vmaMapMemory(allocator, readback_allocation, &mapped);
            if (result != VK_SUCCESS) {
                return vkUnexpected("vmaMapMemory(mesh offscreen readback)", result);
            }
            result = vmaInvalidateAllocation(allocator, readback_allocation, 0, readback_size);
            if (result != VK_SUCCESS) {
                vmaUnmapMemory(allocator, readback_allocation);
                return vkUnexpected("vmaInvalidateAllocation(mesh offscreen readback)", result);
            }

            const auto* color = static_cast<const float*>(mapped);
            const auto* depth = reinterpret_cast<const float*>(
                static_cast<const std::byte*>(mapped) + color_bytes);
            for (std::size_t pixel = 0; pixel < count; ++pixel) {
                const float z_ndc = depth[pixel];
                rgba[pixel] = color[4u * pixel];
                rgba[count + pixel] = color[4u * pixel + 1u];
                rgba[2u * count + pixel] = color[4u * pixel + 2u];
                if (z_ndc >= 1.0f) {
                    rgba[3u * count + pixel] = 0.0f;
                    view_depth[pixel] = std::numeric_limits<float>::infinity();
                } else {
                    rgba[3u * count + pixel] = 1.0f;
                    view_depth[pixel] = linearizeMeshViewDepth(z_ndc, projection);
                }
            }
            vmaUnmapMemory(allocator, readback_allocation);

            return MeshLayer{
                .rgba = lfs::core::Tensor::from_vector(
                    rgba,
                    {4u,
                     static_cast<std::size_t>(requested_height),
                     static_cast<std::size_t>(requested_width)},
                    lfs::core::Device::CPU),
                .view_depth = lfs::core::Tensor::from_vector(
                    view_depth,
                    {static_cast<std::size_t>(requested_height),
                     static_cast<std::size_t>(requested_width)},
                    lfs::core::Device::CPU),
            };
        }
    };

    MeshOffscreenRenderer::MeshOffscreenRenderer()
        : impl_(std::make_unique<Impl>()) {}

    MeshOffscreenRenderer::~MeshOffscreenRenderer() = default;
    MeshOffscreenRenderer::MeshOffscreenRenderer(MeshOffscreenRenderer&&) noexcept = default;
    MeshOffscreenRenderer& MeshOffscreenRenderer::operator=(MeshOffscreenRenderer&&) noexcept = default;

    std::expected<MeshLayer, std::string> MeshOffscreenRenderer::render(
        VulkanContext& context,
        const VulkanMeshPassParams& params,
        const glm::mat4& projection,
        const int width,
        const int height) {
        if (!impl_) {
            impl_ = std::make_unique<Impl>();
        }
        try {
            return impl_->render(context, params, projection, width, height);
        } catch (const std::exception& error) {
            return std::unexpected(std::format(
                "Mesh offscreen rendering threw an exception: {}", error.what()));
        } catch (...) {
            return std::unexpected("Mesh offscreen rendering threw an unknown exception");
        }
    }

    void MeshOffscreenRenderer::shutdown() {
        if (impl_) {
            impl_->shutdown();
        }
    }

} // namespace lfs::vis
