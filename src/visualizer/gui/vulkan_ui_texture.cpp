/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/vulkan_ui_texture.hpp"

#include "config.h"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "rendering/image_layout.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_image_barrier_tracker.hpp"

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <vulkan/vulkan.h>
#endif

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace lfs::vis::gui {

    namespace {
        VulkanContext* g_texture_context = nullptr;

#ifdef LFS_VULKAN_VIEWER_ENABLED
        [[nodiscard]] std::vector<std::uint8_t> toRgba(const std::uint8_t* pixels,
                                                       const int width,
                                                       const int height,
                                                       const int channels) {
            if (!pixels || width <= 0 || height <= 0 || channels <= 0) {
                return {};
            }

            std::vector<std::uint8_t> rgba(static_cast<std::size_t>(width) *
                                           static_cast<std::size_t>(height) * 4u);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t src = (static_cast<std::size_t>(y) *
                                                 static_cast<std::size_t>(width) +
                                             static_cast<std::size_t>(x)) *
                                            static_cast<std::size_t>(channels);
                    const std::size_t dst = (static_cast<std::size_t>(y) *
                                                 static_cast<std::size_t>(width) +
                                             static_cast<std::size_t>(x)) *
                                            4u;
                    if (channels == 1) {
                        rgba[dst + 0] = pixels[src];
                        rgba[dst + 1] = pixels[src];
                        rgba[dst + 2] = pixels[src];
                        rgba[dst + 3] = 255;
                    } else {
                        rgba[dst + 0] = pixels[src + 0];
                        rgba[dst + 1] = pixels[src + 1];
                        rgba[dst + 2] = pixels[src + 2];
                        rgba[dst + 3] = channels >= 4 ? pixels[src + 3] : 255;
                    }
                }
            }
            return rgba;
        }

        [[nodiscard]] std::vector<std::uint8_t> tensorToRgba(const lfs::core::Tensor& image,
                                                             const int expected_width,
                                                             const int expected_height) {
            if (!image.is_valid() || image.ndim() != 3 || expected_width <= 0 || expected_height <= 0) {
                return {};
            }

            const auto layout = lfs::rendering::detectImageLayout(image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                LOG_ERROR("Vulkan UI texture upload received unsupported tensor shape [{}, {}, {}]",
                          image.size(0), image.size(1), image.size(2));
                return {};
            }

            lfs::core::Tensor formatted = (layout == lfs::rendering::ImageLayout::HWC)
                                              ? image
                                              : image.permute({1, 2, 0}).contiguous();
            if (formatted.device() == lfs::core::Device::CUDA) {
                formatted = formatted.cpu();
            }
            if (formatted.dtype() != lfs::core::DataType::UInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
            }
            formatted = formatted.contiguous();

            const int height = static_cast<int>(formatted.size(0));
            const int width = static_cast<int>(formatted.size(1));
            const int channels = static_cast<int>(formatted.size(2));
            if (width != expected_width || height != expected_height || !formatted.ptr<std::uint8_t>()) {
                LOG_ERROR("Vulkan UI texture upload dimension mismatch: {}x{} vs {}x{}",
                          width, height, expected_width, expected_height);
                return {};
            }
            if (channels != 1 && channels != 3 && channels != 4) {
                LOG_ERROR("Vulkan UI texture upload received unsupported channel count {}", channels);
                return {};
            }
            return toRgba(formatted.ptr<std::uint8_t>(), width, height, channels);
        }
#endif
    } // namespace

    void setVulkanUiTextureContext(VulkanContext* const context) {
        g_texture_context = context;
    }

    VulkanContext* getVulkanUiTextureContext() {
        return g_texture_context;
    }

    struct VulkanUiTexture::Impl {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        VkDevice device = VK_NULL_HANDLE;
        VulkanContext* context = nullptr;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        std::uint32_t graphics_queue_family = 0;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation image_allocation = VK_NULL_HANDLE;
        VkImageView image_view = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkImageLayout image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VulkanImageBarrierTracker image_barriers;
        VkFence upload_fence = VK_NULL_HANDLE;
        VkBuffer pending_staging_buffer = VK_NULL_HANDLE;
        VmaAllocation pending_staging_allocation = VK_NULL_HANDLE;
        VkCommandBuffer pending_command_buffer = VK_NULL_HANDLE;
        int width = 0;
        int height = 0;

        void releasePendingUpload() {
            if (pending_command_buffer != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, command_pool, 1, &pending_command_buffer);
                pending_command_buffer = VK_NULL_HANDLE;
            }
            if (pending_staging_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, pending_staging_buffer, pending_staging_allocation);
                pending_staging_buffer = VK_NULL_HANDLE;
                pending_staging_allocation = VK_NULL_HANDLE;
            }
            if (upload_fence == VK_NULL_HANDLE) {
                return;
            }
            vkDestroyFence(device, upload_fence, nullptr);
            upload_fence = VK_NULL_HANDLE;
        }

        void tryReleasePendingUpload() {
            if (upload_fence == VK_NULL_HANDLE) {
                return;
            }
            if (vkGetFenceStatus(device, upload_fence) == VK_SUCCESS) {
                releasePendingUpload();
            }
        }

        void waitAndReleasePendingUpload() {
            if (upload_fence == VK_NULL_HANDLE) {
                return;
            }
            vkWaitForFences(device, 1, &upload_fence, VK_TRUE,
                            std::numeric_limits<std::uint64_t>::max());
            releasePendingUpload();
        }

        [[nodiscard]] bool init(VulkanContext& context) {
            if (device != VK_NULL_HANDLE) {
                return true;
            }
            this->context = &context;
            device = context.device();
            allocator = context.allocator();
            graphics_queue = context.graphicsQueue();
            graphics_queue_family = context.graphicsQueueFamily();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE) {
                LOG_ERROR("Vulkan UI texture requires an initialized Vulkan context");
                device = VK_NULL_HANDLE;
                return false;
            }

            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = graphics_queue_family;
            if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan UI texture command pool");
                device = VK_NULL_HANDLE;
                return false;
            }

            VkSamplerCreateInfo sampler_info{};
            sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler_info.magFilter = VK_FILTER_LINEAR;
            sampler_info.minFilter = VK_FILTER_LINEAR;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.maxLod = 1.0f;
            if (vkCreateSampler(device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan UI texture sampler");
                reset();
                return false;
            }

            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.bindingCount = 1;
            layout_info.pBindings = &binding;
            if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan UI texture descriptor set layout");
                reset();
                return false;
            }

            VkDescriptorPoolSize pool_size{};
            pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_size.descriptorCount = 1;

            VkDescriptorPoolCreateInfo pool_create{};
            pool_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_create.maxSets = 1;
            pool_create.poolSizeCount = 1;
            pool_create.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(device, &pool_create, nullptr, &descriptor_pool) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan UI texture descriptor pool");
                reset();
                return false;
            }
            return true;
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
                LOG_ERROR("Failed to create Vulkan UI texture staging buffer");
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

        [[nodiscard]] VkCommandBuffer beginSingleTimeCommands() const {
            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = command_pool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc_info.commandBufferCount = 1;

            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != VK_SUCCESS) {
                LOG_ERROR("Failed to allocate Vulkan UI texture command buffer");
                return VK_NULL_HANDLE;
            }

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
                LOG_ERROR("Failed to begin Vulkan UI texture command buffer");
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
                return VK_NULL_HANDLE;
            }
            return command_buffer;
        }

        [[nodiscard]] bool endSingleTimeCommands(const VkCommandBuffer command_buffer) const {
            if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
                LOG_ERROR("Failed to end Vulkan UI texture command buffer");
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
                return false;
            }

            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence submit_fence = VK_NULL_HANDLE;
            VkResult submit_status = vkCreateFence(device, &fence_info, nullptr, &submit_fence);
            if (submit_status == VK_SUCCESS) {
                submit_status = vkQueueSubmit(graphics_queue, 1, &submit_info, submit_fence);
            }
            if (submit_status == VK_SUCCESS) {
                submit_status = vkWaitForFences(device, 1, &submit_fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
            }
            if (submit_status != VK_SUCCESS) {
                LOG_ERROR("Failed to submit Vulkan UI texture upload: {}", static_cast<int>(submit_status));
            }
            if (submit_fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, submit_fence, nullptr);
            }
            vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
            return submit_status == VK_SUCCESS;
        }

        void transitionImageLayout(const VkCommandBuffer command_buffer,
                                   const VkImageLayout old_layout,
                                   const VkImageLayout new_layout) {
            if (command_buffer == VK_NULL_HANDLE || image == VK_NULL_HANDLE || old_layout == new_layout) {
                return;
            }
            image_barriers.registerImage(image, VK_IMAGE_ASPECT_COLOR_BIT, old_layout);
            image_barriers.transitionImage(command_buffer, image, VK_IMAGE_ASPECT_COLOR_BIT, new_layout);
        }

        [[nodiscard]] bool ensureImage(const int new_width, const int new_height) {
            if (image != VK_NULL_HANDLE && width == new_width && height == new_height) {
                return true;
            }

            destroyImage();
            width = new_width;
            height = new_height;

            VkImageCreateInfo image_info{};
            image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.extent.width = static_cast<std::uint32_t>(new_width);
            image_info.extent.height = static_cast<std::uint32_t>(new_height);
            image_info.extent.depth = 1;
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
            if (vmaCreateImage(allocator, &image_info, &allocation_info, &image, &image_allocation, nullptr) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan UI texture image");
                destroyImage();
                return false;
            }
            vmaSetAllocationName(allocator, image_allocation, "Vulkan UI texture");

            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &view_info, nullptr, &image_view) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan UI texture image view");
                destroyImage();
                return false;
            }
            image_barriers.registerImage(image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);

            if (descriptor_set == VK_NULL_HANDLE) {
                VkDescriptorSetAllocateInfo alloc_info{};
                alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                alloc_info.descriptorPool = descriptor_pool;
                alloc_info.descriptorSetCount = 1;
                alloc_info.pSetLayouts = &descriptor_set_layout;
                if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
                    LOG_ERROR("Failed to allocate Vulkan UI texture descriptor set");
                    destroyImage();
                    return false;
                }
            }

            VkDescriptorImageInfo image_info_write{};
            image_info_write.sampler = sampler;
            image_info_write.imageView = image_view;
            image_info_write.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptor_set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &image_info_write;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
            image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            return descriptor_set != VK_NULL_HANDLE;
        }

        [[nodiscard]] bool uploadRgbaRegion(const std::vector<std::uint8_t>& rgba,
                                            const int texture_width,
                                            const int texture_height,
                                            const int offset_x,
                                            const int offset_y,
                                            const int region_width,
                                            const int region_height) {
            if (rgba.empty() || texture_width <= 0 || texture_height <= 0 ||
                offset_x < 0 || offset_y < 0 || region_width <= 0 || region_height <= 0 ||
                offset_x + region_width > texture_width ||
                offset_y + region_height > texture_height ||
                rgba.size() != static_cast<std::size_t>(region_width) *
                                   static_cast<std::size_t>(region_height) * 4u) {
                return false;
            }
            VulkanContext* const context = getVulkanUiTextureContext();
            if (!context || !init(*context)) {
                return false;
            }

            if (!ensureImage(texture_width, texture_height)) {
                return false;
            }

            // Block here only if a previous upload to this texture is still in flight.
            waitAndReleasePendingUpload();

            const VkDeviceSize upload_size = static_cast<VkDeviceSize>(rgba.size());
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
            if (!createBuffer(upload_size,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              staging_buffer,
                              staging_allocation)) {
                return false;
            }

            if (!writeAllocation(staging_allocation, rgba.data(), upload_size)) {
                LOG_ERROR("Failed to map Vulkan UI texture staging memory");
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return false;
            }

            VkCommandBuffer command_buffer = beginSingleTimeCommands();
            if (command_buffer == VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return false;
            }

            transitionImageLayout(command_buffer, image_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            VkBufferImageCopy copy_region{};
            copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy_region.imageSubresource.mipLevel = 0;
            copy_region.imageSubresource.baseArrayLayer = 0;
            copy_region.imageSubresource.layerCount = 1;
            copy_region.imageOffset = {offset_x, offset_y, 0};
            copy_region.imageExtent = {static_cast<std::uint32_t>(region_width),
                                       static_cast<std::uint32_t>(region_height),
                                       1};
            vkCmdCopyBufferToImage(command_buffer,
                                   staging_buffer,
                                   image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1,
                                   &copy_region);

            transitionImageLayout(command_buffer,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
                LOG_ERROR("Failed to end Vulkan UI texture command buffer");
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return false;
            }

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence = VK_NULL_HANDLE;
            if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan UI texture upload fence");
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return false;
            }

            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            const VkResult submit_status = vkQueueSubmit(graphics_queue, 1, &submit_info, fence);
            if (submit_status != VK_SUCCESS) {
                LOG_ERROR("Failed to submit Vulkan UI texture upload: {}",
                          static_cast<int>(submit_status));
                vkDestroyFence(device, fence, nullptr);
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return false;
            }

            // Defer command-buffer + staging-buffer cleanup until the GPU finishes via the fence.
            // The next upload (or destruction) reaps them.
            image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            upload_fence = fence;
            pending_command_buffer = command_buffer;
            pending_staging_buffer = staging_buffer;
            pending_staging_allocation = staging_allocation;
            return true;
        }

        [[nodiscard]] bool uploadRgba(const std::vector<std::uint8_t>& rgba,
                                      const int new_width,
                                      const int new_height) {
            return uploadRgbaRegion(rgba, new_width, new_height, 0, 0, new_width, new_height);
        }

        [[nodiscard]] bool uploadRegion(const std::uint8_t* pixels,
                                        const int texture_width,
                                        const int texture_height,
                                        const int x,
                                        const int y,
                                        const int region_width,
                                        const int region_height,
                                        const int channels) {
            if (!pixels || region_width <= 0 || region_height <= 0 || channels <= 0 || channels > 4) {
                return false;
            }
            return uploadRgbaRegion(toRgba(pixels, region_width, region_height, channels),
                                    texture_width,
                                    texture_height,
                                    x,
                                    y,
                                    region_width,
                                    region_height);
        }

        [[nodiscard]] bool upload(const std::uint8_t* pixels,
                                  const int new_width,
                                  const int new_height,
                                  const int channels) {
            if (!pixels || new_width <= 0 || new_height <= 0 || channels <= 0 || channels > 4) {
                return false;
            }
            return uploadRgba(toRgba(pixels, new_width, new_height, channels), new_width, new_height);
        }

        void destroyImage() {
            waitAndReleasePendingUpload();
            if (image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, image_view, nullptr);
                image_view = VK_NULL_HANDLE;
            }
            if (image != VK_NULL_HANDLE) {
                image_barriers.forgetImage(image);
                vmaDestroyImage(allocator, image, image_allocation);
                image = VK_NULL_HANDLE;
                image_allocation = VK_NULL_HANDLE;
            }
            image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            width = 0;
            height = 0;
        }

        void reset() {
            if (device != VK_NULL_HANDLE) {
                if (context != nullptr && !context->waitForSubmittedFrames()) {
                    LOG_WARN("Vulkan UI texture shutdown could not wait for submitted frames: {}",
                             context->lastError());
                }
                destroyImage();
                if (sampler != VK_NULL_HANDLE) {
                    vkDestroySampler(device, sampler, nullptr);
                    sampler = VK_NULL_HANDLE;
                }
                if (descriptor_pool != VK_NULL_HANDLE) {
                    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
                    descriptor_pool = VK_NULL_HANDLE;
                    descriptor_set = VK_NULL_HANDLE;
                }
                if (descriptor_set_layout != VK_NULL_HANDLE) {
                    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
                    descriptor_set_layout = VK_NULL_HANDLE;
                }
                if (command_pool != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(device, command_pool, nullptr);
                    command_pool = VK_NULL_HANDLE;
                }
            }
            device = VK_NULL_HANDLE;
            context = nullptr;
            allocator = VK_NULL_HANDLE;
            graphics_queue = VK_NULL_HANDLE;
            graphics_queue_family = 0;
        }
#else
        [[nodiscard]] bool upload(const std::uint8_t*, int, int, int) { return false; }
        [[nodiscard]] bool uploadRegion(const std::uint8_t*, int, int, int, int, int, int, int) { return false; }
        void reset() {}
#endif
    };

    VulkanUiTexture::~VulkanUiTexture() {
        reset();
        delete impl_;
    }

    VulkanUiTexture::VulkanUiTexture(VulkanUiTexture&& other) noexcept
        : impl_(std::exchange(other.impl_, nullptr)) {}

    VulkanUiTexture& VulkanUiTexture::operator=(VulkanUiTexture&& other) noexcept {
        if (this != &other) {
            reset();
            delete impl_;
            impl_ = std::exchange(other.impl_, nullptr);
        }
        return *this;
    }

    bool VulkanUiTexture::upload(const std::uint8_t* const pixels,
                                    const int width,
                                    const int height,
                                    const int channels) {
        if (!impl_) {
            impl_ = new Impl();
        }
        return impl_->upload(pixels, width, height, channels);
    }

    bool VulkanUiTexture::uploadRegion(const std::uint8_t* const pixels,
                                       const int texture_width,
                                       const int texture_height,
                                       const int x,
                                       const int y,
                                       const int width,
                                       const int height,
                                       const int channels) {
        if (!impl_) {
            impl_ = new Impl();
        }
        return impl_->uploadRegion(pixels, texture_width, texture_height, x, y, width, height, channels);
    }

    bool VulkanUiTexture::upload(const lfs::core::Tensor& image,
                                    const int expected_width,
                                    const int expected_height) {
        if (!impl_) {
            impl_ = new Impl();
        }
#ifdef LFS_VULKAN_VIEWER_ENABLED
        const std::vector<std::uint8_t> rgba = tensorToRgba(image, expected_width, expected_height);
        return impl_->uploadRgba(rgba, expected_width, expected_height);
#else
        (void)image;
        (void)expected_width;
        (void)expected_height;
        return false;
#endif
    }

    std::uintptr_t VulkanUiTexture::textureId() const {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        if (!impl_) {
            return 0;
        }
        impl_->tryReleasePendingUpload();
        return reinterpret_cast<std::uintptr_t>(impl_->descriptor_set);
#else
        return 0;
#endif
    }

    bool VulkanUiTexture::valid() const {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        if (!impl_) {
            return false;
        }
        impl_->tryReleasePendingUpload();
        return impl_->descriptor_set != VK_NULL_HANDLE && impl_->image_view != VK_NULL_HANDLE;
#else
        return false;
#endif
    }

    void VulkanUiTexture::reset() {
        if (impl_) {
            impl_->reset();
        }
    }

} // namespace lfs::vis::gui
