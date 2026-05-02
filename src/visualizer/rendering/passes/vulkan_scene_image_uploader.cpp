/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_image_uploader.hpp"

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "vulkan_viewport_pass.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_image_barrier_tracker.hpp"

#include <cstring>
#include <limits>

namespace lfs::vis {
    struct VulkanSceneImageUploader::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        std::uint32_t graphics_queue_family = 0;
        VkSampler scene_sampler = VK_NULL_HANDLE;
        VkCommandPool upload_command_pool = VK_NULL_HANDLE;

        VkImage scene_image = VK_NULL_HANDLE;
        VmaAllocation scene_image_allocation = VK_NULL_HANDLE;
        VkImageView scene_image_view = VK_NULL_HANDLE;
        VulkanImageBarrierTracker scene_image_barriers;
        glm::ivec2 scene_image_size{0, 0};
        const lfs::core::Tensor* uploaded_scene_tensor = nullptr;
        bool scene_image_external = false;
        std::uint64_t scene_image_external_generation = 0;

        [[nodiscard]] bool init(VulkanContext& vulkan_context, const VkSampler sampler) {
            if (device != VK_NULL_HANDLE) {
                return true;
            }
            context = &vulkan_context;
            device = vulkan_context.device();
            allocator = vulkan_context.allocator();
            graphics_queue = vulkan_context.graphicsQueue();
            graphics_queue_family = vulkan_context.graphicsQueueFamily();
            scene_sampler = sampler;
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE || scene_sampler == VK_NULL_HANDLE) {
                LOG_ERROR("Vulkan scene image uploader requires an initialized Vulkan context");
                return false;
            }

            VkCommandPoolCreateInfo command_pool_info{};
            command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            command_pool_info.queueFamilyIndex = graphics_queue_family;
            if (vkCreateCommandPool(device, &command_pool_info, nullptr, &upload_command_pool) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan scene image upload command pool");
                shutdown();
                return false;
            }
            return true;
        }

        void shutdown() {
            destroySceneImage();
            if (upload_command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, upload_command_pool, nullptr);
                upload_command_pool = VK_NULL_HANDLE;
            }
            scene_sampler = VK_NULL_HANDLE;
            graphics_queue = VK_NULL_HANDLE;
            graphics_queue_family = 0;
            allocator = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            context = nullptr;
        }

        [[nodiscard]] bool createBuffer(const VkDeviceSize size,
                                        const VkBufferUsageFlags usage,
                                        VkBuffer& buffer,
                                        VmaAllocation& allocation) const {
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = size;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

            if (vmaCreateBuffer(allocator, &buffer_info, &allocation_info, &buffer, &allocation, nullptr) != VK_SUCCESS) {
                buffer = VK_NULL_HANDLE;
                allocation = VK_NULL_HANDLE;
                return false;
            }
            return true;
        }

        [[nodiscard]] bool writeAllocation(const VmaAllocation allocation,
                                           const void* const source,
                                           const VkDeviceSize size) const {
            if (allocation == VK_NULL_HANDLE || !source || size == 0) {
                return false;
            }
            void* mapped = nullptr;
            if (vmaMapMemory(allocator, allocation, &mapped) != VK_SUCCESS || !mapped) {
                return false;
            }
            std::memcpy(mapped, source, static_cast<std::size_t>(size));
            vmaFlushAllocation(allocator, allocation, 0, size);
            vmaUnmapMemory(allocator, allocation);
            return true;
        }

        [[nodiscard]] VkCommandBuffer beginUploadCommands() const {
            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = upload_command_pool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc_info.commandBufferCount = 1;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
                return VK_NULL_HANDLE;
            }
            return command_buffer;
        }

        [[nodiscard]] bool endUploadCommands(const VkCommandBuffer command_buffer) const {
            if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
                return false;
            }

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence = VK_NULL_HANDLE;
            if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
                return false;
            }

            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            VkResult result = vkQueueSubmit(graphics_queue, 1, &submit_info, fence);
            if (result == VK_SUCCESS) {
                result = vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
            }
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
            return result == VK_SUCCESS;
        }

        void clearSceneImageBinding() {
            scene_image_barriers.forgetImage(scene_image);
            scene_image = VK_NULL_HANDLE;
            scene_image_allocation = VK_NULL_HANDLE;
            scene_image_view = VK_NULL_HANDLE;
            scene_image_size = {0, 0};
            uploaded_scene_tensor = nullptr;
            scene_image_external = false;
            scene_image_external_generation = 0;
        }

        void updateSceneDescriptor(const VkDescriptorSet scene_descriptor_set,
                                   const VkImageView image_view,
                                   const VkImageLayout image_layout) const {
            if (scene_descriptor_set == VK_NULL_HANDLE) {
                return;
            }
            VkDescriptorImageInfo descriptor_info{};
            descriptor_info.sampler = scene_sampler;
            descriptor_info.imageView = image_view;
            descriptor_info.imageLayout = image_layout;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = scene_descriptor_set;
            write.dstBinding = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &descriptor_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        void destroySceneImage() {
            if (scene_image_external) {
                clearSceneImageBinding();
                return;
            }
            if (scene_image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, scene_image_view, nullptr);
            }
            if (scene_image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator, scene_image, scene_image_allocation);
            }
            clearSceneImageBinding();
        }

        [[nodiscard]] bool ensureSceneImage(const glm::ivec2 size, const VkDescriptorSet scene_descriptor_set) {
            if (scene_image != VK_NULL_HANDLE && scene_image_size == size) {
                updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                return true;
            }
            destroySceneImage();

            VkImageCreateInfo image_info{};
            image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.extent = {static_cast<std::uint32_t>(size.x), static_cast<std::uint32_t>(size.y), 1};
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(allocator,
                               &image_info,
                               &allocation_info,
                               &scene_image,
                               &scene_image_allocation,
                               nullptr) != VK_SUCCESS) {
                destroySceneImage();
                return false;
            }
            vmaSetAllocationName(allocator, scene_image_allocation, "Viewport scene image");

            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = scene_image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &view_info, nullptr, &scene_image_view) != VK_SUCCESS) {
                destroySceneImage();
                return false;
            }

            scene_image_size = size;
            scene_image_barriers.registerImage(scene_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
            updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return true;
        }

        [[nodiscard]] bool bindExternalSceneImage(const VulkanViewportPassParams& params,
                                                  const VkDescriptorSet scene_descriptor_set) {
            if (params.external_scene_image == VK_NULL_HANDLE ||
                params.external_scene_image_view == VK_NULL_HANDLE ||
                params.scene_image_size.x <= 0 ||
                params.scene_image_size.y <= 0) {
                return false;
            }
            if (scene_image_external &&
                scene_image == params.external_scene_image &&
                scene_image_view == params.external_scene_image_view &&
                scene_image_size == params.scene_image_size &&
                scene_image_barriers.imageLayout(scene_image, VK_IMAGE_LAYOUT_UNDEFINED) == params.external_scene_image_layout &&
                scene_image_external_generation == params.external_scene_image_generation) {
                updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                return true;
            }

            destroySceneImage();
            scene_image = params.external_scene_image;
            scene_image_view = params.external_scene_image_view;
            scene_image_size = params.scene_image_size;
            uploaded_scene_tensor = params.scene_image.get();
            scene_image_external = true;
            scene_image_external_generation = params.external_scene_image_generation;
            scene_image_barriers.registerImage(scene_image, VK_IMAGE_ASPECT_COLOR_BIT, params.external_scene_image_layout);
            updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return true;
        }

        void upload(const VulkanViewportPassParams& params, const VkDescriptorSet scene_descriptor_set) {
            const bool has_external_image =
                params.external_scene_image != VK_NULL_HANDLE &&
                params.external_scene_image_view != VK_NULL_HANDLE;
            if ((!params.scene_image && !has_external_image) ||
                params.scene_image_size.x <= 0 || params.scene_image_size.y <= 0) {
                uploaded_scene_tensor = nullptr;
                return;
            }
            if (has_external_image) {
                if (!bindExternalSceneImage(params, scene_descriptor_set)) {
                    LOG_ERROR("Failed to bind external Vulkan viewport scene image");
                }
                return;
            }
            if (scene_image_external) {
                destroySceneImage();
            }
            if (uploaded_scene_tensor == params.scene_image.get() && scene_image_size == params.scene_image_size &&
                scene_image_view != VK_NULL_HANDLE) {
                updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                return;
            }
#ifdef LFS_VULKAN_NO_INTEROP_FALLBACK
            static bool logged_fallback_disabled = false;
            if (!logged_fallback_disabled) {
                LOG_ERROR("Vulkan viewport staging fallback is disabled at build time; no external scene image was supplied");
                logged_fallback_disabled = true;
            }
            uploaded_scene_tensor = nullptr;
#else
            const auto rgba = lfs::rendering::packTensorToRgba8Host(
                *params.scene_image,
                lfs::rendering::CudaVulkanExtent2D{
                    .width = static_cast<std::uint32_t>(params.scene_image_size.x),
                    .height = static_cast<std::uint32_t>(params.scene_image_size.y),
                });
            if (!rgba) {
                LOG_WARN("Vulkan viewport staging fallback skipped: {}", rgba.error);
                return;
            }
            if (!ensureSceneImage(params.scene_image_size, scene_descriptor_set)) {
                return;
            }

            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
            const VkDeviceSize upload_size = static_cast<VkDeviceSize>(rgba.pixels.size());
            if (!createBuffer(upload_size,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              staging_buffer,
                              staging_allocation)) {
                return;
            }

            if (!writeAllocation(staging_allocation, rgba.pixels.data(), upload_size)) {
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return;
            }

            VkCommandBuffer command_buffer = beginUploadCommands();
            if (command_buffer == VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return;
            }
            scene_image_barriers.transitionImage(command_buffer,
                                                 scene_image,
                                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy copy{};
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = 0;
            copy.imageSubresource.baseArrayLayer = 0;
            copy.imageSubresource.layerCount = 1;
            copy.imageExtent = {
                static_cast<std::uint32_t>(params.scene_image_size.x),
                static_cast<std::uint32_t>(params.scene_image_size.y),
                1,
            };
            vkCmdCopyBufferToImage(command_buffer,
                                   staging_buffer,
                                   scene_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1,
                                   &copy);
            scene_image_barriers.transitionImage(command_buffer,
                                                 scene_image,
                                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (endUploadCommands(command_buffer)) {
                uploaded_scene_tensor = params.scene_image.get();
            }
            vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
#endif
        }

        [[nodiscard]] bool hasImage() const {
            return scene_image_view != VK_NULL_HANDLE;
        }
    };

    VulkanSceneImageUploader::VulkanSceneImageUploader()
        : impl_(std::make_unique<Impl>()) {}

    VulkanSceneImageUploader::~VulkanSceneImageUploader() {
        if (impl_) {
            impl_->shutdown();
        }
    }

    VulkanSceneImageUploader::VulkanSceneImageUploader(VulkanSceneImageUploader&&) noexcept = default;

    VulkanSceneImageUploader& VulkanSceneImageUploader::operator=(VulkanSceneImageUploader&&) noexcept = default;

    bool VulkanSceneImageUploader::init(VulkanContext& context, const VkSampler scene_sampler) {
        return impl_->init(context, scene_sampler);
    }

    void VulkanSceneImageUploader::shutdown() {
        impl_->shutdown();
    }

    void VulkanSceneImageUploader::upload(const VulkanViewportPassParams& params,
                                          const VkDescriptorSet scene_descriptor_set) {
        impl_->upload(params, scene_descriptor_set);
    }

    bool VulkanSceneImageUploader::hasImage() const {
        return impl_->hasImage();
    }
} // namespace lfs::vis
#endif
