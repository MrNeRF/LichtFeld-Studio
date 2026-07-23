/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/vulkan_ui_texture.hpp"

#include "config.h"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "gui/rmlui/rmlui_vk_backend.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/vulkan_wait.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_image_barrier_tracker.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis::gui {

    namespace {
        VulkanContext* g_texture_context = nullptr;
        lfs::rendering::CudaVulkanUploadStream g_texture_upload_stream;

        [[nodiscard]] const char* waitOutcomeLabel(const lfs::rendering::WaitOutcome outcome) noexcept {
            using lfs::rendering::WaitOutcome;
            switch (outcome) {
            case WaitOutcome::Ready: return "Ready";
            case WaitOutcome::Cancelled: return "Cancelled";
            case WaitOutcome::Shutdown: return "Shutdown";
            case WaitOutcome::Quarantined: return "Quarantined";
            }
            return "Unknown";
        }

        [[nodiscard]] std::string formatWaitFailure(
            const lfs::Result<lfs::rendering::WaitOutcome>& outcome) {
            if (outcome.has_value()) {
                return waitOutcomeLabel(*outcome);
            }
            return std::string(outcome.error().detail());
        }

        [[nodiscard]] std::vector<std::uint8_t> toRgba(const std::uint8_t* pixels,
                                                       const int width,
                                                       const int height,
                                                       const int channels,
                                                       const bool flip_y = false) {
            if (!pixels || width <= 0 || height <= 0 || channels <= 0) {
                return {};
            }

            const std::size_t row_in = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
            const std::size_t row_out = static_cast<std::size_t>(width) * 4u;
            std::vector<std::uint8_t> rgba(static_cast<std::size_t>(height) * row_out);
            for (int y = 0; y < height; ++y) {
                const int src_y = flip_y ? (height - 1 - y) : y;
                const std::uint8_t* src = pixels + static_cast<std::size_t>(src_y) * row_in;
                std::uint8_t* dst = rgba.data() + static_cast<std::size_t>(y) * row_out;
                if (channels == 4) {
                    std::memcpy(dst, src, row_out);
                    continue;
                }
                if (channels == 1) {
                    for (int x = 0; x < width; ++x, ++src, dst += 4) {
                        dst[0] = dst[1] = dst[2] = src[0];
                        dst[3] = 255;
                    }
                } else {
                    for (int x = 0; x < width; ++x, src += channels, dst += 4) {
                        dst[0] = src[0];
                        dst[1] = src[1];
                        dst[2] = src[2];
                        dst[3] = 255;
                    }
                }
            }
            return rgba;
        }

        [[nodiscard]] std::vector<std::uint8_t> tensorToRgba(const lfs::core::Tensor& image,
                                                             const int expected_width,
                                                             const int expected_height,
                                                             const bool flip_y = false) {
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
            return toRgba(formatted.ptr<std::uint8_t>(), width, height, channels, flip_y);
        }
    } // namespace

    void setVulkanUiTextureContext(VulkanContext* const context) {
        if (context == nullptr) {
            if (!g_texture_upload_stream.synchronize()) {
                LOG_WARN("CUDA/Vulkan UI texture stream synchronization failed during shutdown: {}",
                         g_texture_upload_stream.lastError());
            }
            g_texture_upload_stream.reset();
        } else if (!g_texture_upload_stream.valid() && !g_texture_upload_stream.init()) {
            LOG_ERROR("Could not create the non-blocking CUDA/Vulkan UI texture stream: {}",
                      g_texture_upload_stream.lastError());
        }
        g_texture_context = context;
    }

    VulkanContext* getVulkanUiTextureContext() {
        return g_texture_context;
    }

    struct VulkanUiTexture::Impl {
        enum class Mode : std::uint8_t {
            Uninitialized,
            Cpu,
            CudaInterop,
        };

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
        std::string image_vram_label;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkImageLayout image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VulkanImageBarrierTracker image_barriers;
        struct PendingUpload {
            VkFence fence = VK_NULL_HANDLE;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
        };
        // Bounded ring of in-flight uploads. Uploads to the same image serialize on the graphics
        // queue (no semaphores), so a depth > 1 only defers staging-buffer reclamation; it does not
        // race the GPU. The main thread blocks only when the ring is full.
        static constexpr std::size_t kMaxPendingUploads = 3;
        std::vector<PendingUpload> pending_uploads;

        int width = 0;
        int height = 0;

        // CUDA-Vulkan interop path: bypasses CPU readback by writing the rasterizer's
        // CUDA tensor directly into the VkImage via a shared external memory handle.
        Mode mode = Mode::Uninitialized;
        VulkanContext::ExternalImage interop_image{};
        VulkanContext::ExternalSemaphore interop_semaphore{};
        lfs::rendering::CudaVulkanInterop interop;
        std::uint64_t interop_timeline_value = 0;
        bool interop_disabled = false;

        void destroyPendingUpload(PendingUpload& upload) {
            if (upload.command_buffer != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, command_pool, 1, &upload.command_buffer);
            }
            if (upload.staging_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, upload.staging_buffer, upload.staging_allocation);
            }
            if (upload.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, upload.fence, nullptr);
            }
            upload = {};
        }

        // Reap entries whose GPU work has completed. Non-blocking.
        void tryReleasePendingUpload() {
            auto write = pending_uploads.begin();
            for (auto read = pending_uploads.begin(); read != pending_uploads.end(); ++read) {
                if (read->fence != VK_NULL_HANDLE && vkGetFenceStatus(device, read->fence) == VK_SUCCESS) {
                    destroyPendingUpload(*read);
                } else {
                    if (write != read) {
                        *write = *read;
                    }
                    ++write;
                }
            }
            pending_uploads.erase(write, pending_uploads.end());
        }

        // Wait for all in-flight uploads to finish, then release them. Used before destroying the image.
        // Non-Ready waits retain the entry (AMB-4); Ready (or null fence) entries are destroyed.
        void waitAndReleasePendingUpload() {
            auto write = pending_uploads.begin();
            for (auto read = pending_uploads.begin(); read != pending_uploads.end(); ++read) {
                bool ready = true;
                if (read->fence != VK_NULL_HANDLE) {
                    lfs::rendering::WaitContext wait_ctx;
                    wait_ctx.fingerprint = "ui_texture.pending_upload.wait";
                    auto wait_outcome = lfs::rendering::wait_fence_bounded(
                        device,
                        read->fence,
                        std::stop_token{},
                        lfs::rendering::VulkanWaitPolicy{},
                        wait_ctx);
                    ready = wait_outcome.has_value() &&
                            *wait_outcome == lfs::rendering::WaitOutcome::Ready;
                    if (!ready) {
                        LOG_ERROR(
                            "Vulkan UI texture pending-upload wait did not reach Ready "
                            "(fence={:#x}): {} — retaining upload resources",
                            lfs::rendering::vkHandleValue(read->fence),
                            formatWaitFailure(wait_outcome));
                    }
                }
                if (ready) {
                    destroyPendingUpload(*read);
                } else {
                    if (write != read) {
                        *write = *read;
                    }
                    ++write;
                }
            }
            pending_uploads.erase(write, pending_uploads.end());
        }

        // Block only when the ring is full: wait on the oldest entry to free a slot.
        // Returns false if the oldest entry cannot be drained (quarantine/cancel/error).
        [[nodiscard]] bool enforcePendingUploadBound() {
            tryReleasePendingUpload();
            while (pending_uploads.size() >= kMaxPendingUploads) {
                PendingUpload& oldest = pending_uploads.front();
                if (oldest.fence != VK_NULL_HANDLE) {
                    lfs::rendering::WaitContext wait_ctx;
                    wait_ctx.fingerprint = "ui_texture.pending_upload.bound";
                    auto wait_outcome = lfs::rendering::wait_fence_bounded(
                        device,
                        oldest.fence,
                        std::stop_token{},
                        lfs::rendering::VulkanWaitPolicy{},
                        wait_ctx);
                    if (!wait_outcome.has_value() ||
                        *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
                        LOG_ERROR(
                            "Vulkan UI texture pending-upload bound wait did not reach Ready "
                            "(fence={:#x}, pending={}): {} — retaining oldest slot",
                            lfs::rendering::vkHandleValue(oldest.fence),
                            pending_uploads.size(),
                            formatWaitFailure(wait_outcome));
                        return false;
                    }
                }
                destroyPendingUpload(oldest);
                pending_uploads.erase(pending_uploads.begin());
            }
            return true;
        }

        [[nodiscard]] bool init(VulkanContext& ctx) {
            if (device != VK_NULL_HANDLE) {
                return true;
            }
            this->context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            graphics_queue = ctx.graphicsQueue();
            graphics_queue_family = ctx.graphicsQueueFamily();
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
            const VkResult flush_result = vmaFlushAllocation(allocator, allocation, 0, size);
            vmaUnmapMemory(allocator, allocation);
            return flush_result == VK_SUCCESS;
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
            bool submitted = false;
            bool wait_ready = false;
            if (submit_status == VK_SUCCESS) {
                submit_status = vkQueueSubmit(graphics_queue, 1, &submit_info, submit_fence);
                if (submit_status == VK_SUCCESS) {
                    submitted = true;
                }
            }
            if (submit_status == VK_SUCCESS) {
                lfs::rendering::WaitContext wait_ctx;
                wait_ctx.fingerprint = "ui_texture.oneshot.wait";
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    submit_fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                if (wait_outcome.has_value() &&
                    *wait_outcome == lfs::rendering::WaitOutcome::Ready) {
                    wait_ready = true;
                } else {
                    submit_status = VK_TIMEOUT;
                    LOG_ERROR(
                        "Vulkan UI texture one-shot wait did not reach Ready (fence={:#x}): {}",
                        lfs::rendering::vkHandleValue(submit_fence),
                        formatWaitFailure(wait_outcome));
                }
            }
            if (submit_status != VK_SUCCESS && submit_status != VK_TIMEOUT) {
                LOG_ERROR("Failed to submit Vulkan UI texture upload: {}", static_cast<int>(submit_status));
            }
            // AMB-4: destroy fence/CB only when never submitted or wait Ready.
            if (submit_fence != VK_NULL_HANDLE) {
                if (!submitted || wait_ready) {
                    vkDestroyFence(device, submit_fence, nullptr);
                } else {
                    LOG_ERROR(
                        "Vulkan: retaining UI texture one-shot fence after non-Ready wait "
                        "(fence={:#x})",
                        lfs::rendering::vkHandleValue(submit_fence));
                }
            }
            if (!submitted || wait_ready) {
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
            } else {
                LOG_ERROR(
                    "Vulkan: retaining UI texture one-shot command buffer after non-Ready wait "
                    "(command_buffer={:#x})",
                    lfs::rendering::vkHandleValue(command_buffer));
            }
            return submit_status == VK_SUCCESS && wait_ready;
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
            if (mode == Mode::CudaInterop) {
                LOG_ERROR("Vulkan UI texture used CPU upload after CUDA-interop mode was engaged");
                return false;
            }
            if (image != VK_NULL_HANDLE && width == new_width && height == new_height) {
                return true;
            }

            if (image != VK_NULL_HANDLE) {
                waitAndReleasePendingUpload();
                // Descriptor updates and image destruction are only legal once
                // every submitted RmlUI draw that can reference the old view has
                // retired. Resize is already a cold path; pay the bounded fence
                // wait here instead of retaining every historical image forever.
                if (context != nullptr && !context->waitForSubmittedFrames()) {
                    LOG_ERROR("Vulkan UI texture resize could not drain submitted frames: {}",
                              context->lastError());
                    return false;
                }
                if (!image_vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.ui_texture.image",
                        image_vram_label,
                        0);
                }
                image_barriers.forgetImage(image);
                if (image_view != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, image_view, nullptr);
                }
                vmaDestroyImage(allocator, image, image_allocation);
                image = VK_NULL_HANDLE;
                image_view = VK_NULL_HANDLE;
                image_allocation = VK_NULL_HANDLE;
                image_vram_label.clear();
            }
            width = new_width;
            height = new_height;
            mode = Mode::Cpu;

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
            VmaAllocationInfo created_allocation_info{};
            if (vmaCreateImage(allocator, &image_info, &allocation_info, &image, &image_allocation, &created_allocation_info) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan UI texture image");
                destroyImage();
                return false;
            }
            vmaSetAllocationName(allocator, image_allocation, "Vulkan UI texture");
            image_vram_label = std::format("cpu_upload_rgba8:{}x{}", new_width, new_height);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.ui_texture.image",
                image_vram_label,
                static_cast<std::size_t>(created_allocation_info.size));

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
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                         image,
                                         "ui.texture.cpu[{}x{}]",
                                         new_width,
                                         new_height);
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                         image_view,
                                         "ui.texture.cpu[{}x{}].view",
                                         new_width,
                                         new_height);
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

        [[nodiscard]] bool ensureInteropImage(const int new_width, const int new_height) {
            if (interop_disabled || !context || !context->externalMemoryInteropEnabled() ||
                !context->externalSemaphoreInteropEnabled()) {
                return false;
            }
            if (mode == Mode::Cpu) {
                LOG_ERROR("Vulkan UI texture used CUDA-interop after CPU mode was engaged");
                return false;
            }
            if (interop.valid() && width == new_width && height == new_height) {
                return true;
            }

            destroyImage();
            width = new_width;
            height = new_height;

            const VkExtent2D extent{
                static_cast<std::uint32_t>(new_width),
                static_cast<std::uint32_t>(new_height),
            };
            if (!context->createExternalImage(extent,
                                              VK_FORMAT_R8G8B8A8_UNORM,
                                              interop_image,
                                              "vulkan.ui_texture.interop_image",
                                              "rgba8") ||
                !context->createExternalTimelineSemaphore(0, interop_semaphore)) {
                LOG_WARN("Vulkan UI texture interop setup failed: {}", context->lastError());
                destroyImage();
                interop_disabled = true;
                return false;
            }
            const std::uint64_t vulkan_ready_value = ++interop_timeline_value;
            if (!context->transitionImageLayoutImmediate(interop_image.image,
                                                         VK_IMAGE_LAYOUT_UNDEFINED,
                                                         VK_IMAGE_LAYOUT_GENERAL,
                                                         VulkanContext::ImmediateTransitionOptions::signalAt(
                                                             {interop_semaphore.semaphore, vulkan_ready_value}))) {
                LOG_WARN("Vulkan UI texture interop initial transition failed: {}", context->lastError());
                destroyImage();
                interop_disabled = true;
                return false;
            }
            // Retire the initial Vulkan signal before CUDA can advance the
            // exported timeline. Per-upload ownership transfers below stay
            // asynchronous; this is the one-time cross-API handoff boundary.
            if (!context->waitForImmediateSubmits()) {
                LOG_WARN("Vulkan UI texture interop initialization handoff failed: {}",
                         context->lastError());
                destroyImage();
                interop_disabled = true;
                return false;
            }

            const auto memory_handle = context->releaseExternalImageNativeHandle(interop_image);
            const auto semaphore_handle = context->releaseExternalSemaphoreNativeHandle(interop_semaphore);
            const lfs::rendering::CudaVulkanExternalImageImport image_import{
                .memory_handle = memory_handle,
                .allocation_size = static_cast<std::size_t>(interop_image.allocation_size),
                .extent = {.width = extent.width, .height = extent.height},
                .format = lfs::rendering::CudaVulkanImageFormat::Rgba8Unorm,
                .dedicated_allocation = context->externalMemoryDedicatedAllocationEnabled(),
            };
            const lfs::rendering::CudaVulkanExternalSemaphoreImport semaphore_import{
                .semaphore_handle = semaphore_handle,
                .initial_value = 0,
            };
            if (!interop.init(image_import, semaphore_import)) {
                LOG_WARN("Vulkan UI texture CUDA import failed: {}", interop.lastError());
                destroyImage();
                interop_disabled = true;
                return false;
            }

            image = interop_image.image;
            image_view = interop_image.view;
            image_layout = VK_IMAGE_LAYOUT_GENERAL;
            image_barriers.registerImage(image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            mode = Mode::CudaInterop;

            // Skip allocating an internal descriptor set: the interop path is consumed via RmlUi
            // (which allocates its own descriptor against its texture-set layout).
            descriptor_set = VK_NULL_HANDLE;
            return true;
        }

        [[nodiscard]] bool uploadCudaTensorImpl(const lfs::core::Tensor& tensor,
                                                const int expected_width,
                                                const int expected_height,
                                                const bool flip_y) {
            if (!tensor.is_valid() || tensor.device() != lfs::core::Device::CUDA) {
                return false;
            }
            VulkanContext* const ctx = getVulkanUiTextureContext();
            if (!ctx || !g_texture_upload_stream.valid() || !init(*ctx)) {
                return false;
            }
            if (!ensureInteropImage(expected_width, expected_height)) {
                return false;
            }

            if (image_layout != VK_IMAGE_LAYOUT_GENERAL) {
                const std::uint64_t vulkan_ready_value = ++interop_timeline_value;
                if (!ctx->transitionImageLayoutImmediate(image,
                                                         image_layout,
                                                         VK_IMAGE_LAYOUT_GENERAL,
                                                         VulkanContext::ImmediateTransitionOptions::signalAt(
                                                             {interop_semaphore.semaphore, vulkan_ready_value}))) {
                    LOG_ERROR("Vulkan UI texture interop transition to GENERAL failed: {}", ctx->lastError());
                    return false;
                }
                image_layout = VK_IMAGE_LAYOUT_GENERAL;
            }

            const cudaStream_t upload_stream = g_texture_upload_stream.stream();
            if (!interop.wait(interop_timeline_value, upload_stream)) {
                LOG_ERROR("Vulkan UI texture CUDA wait for Vulkan image release failed: {}",
                          interop.lastError());
                return false;
            }
            if (!interop.copyTensorToSurface(tensor, upload_stream, flip_y)) {
                LOG_ERROR("Vulkan UI texture CUDA copy failed: {}", interop.lastError());
                return false;
            }

            const std::uint64_t signal_value = ++interop_timeline_value;
            if (!interop.signal(signal_value, upload_stream)) {
                LOG_ERROR("Vulkan UI texture CUDA signal failed: {}", interop.lastError());
                return false;
            }
            if (!ctx->transitionImageLayoutImmediate(image,
                                                     VK_IMAGE_LAYOUT_GENERAL,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                     VulkanContext::ImmediateTransitionOptions::waitOn(
                                                         {interop_semaphore.semaphore, signal_value},
                                                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT))) {
                LOG_ERROR("Vulkan UI texture interop transition to read-only failed: {}", ctx->lastError());
                return false;
            }
            image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            return true;
        }

        [[nodiscard]] bool uploadRgbaRegion(const std::uint8_t* const rgba,
                                            const std::size_t rgba_size,
                                            const int texture_width,
                                            const int texture_height,
                                            const int offset_x,
                                            const int offset_y,
                                            const int region_width,
                                            const int region_height) {
            if (!rgba || rgba_size == 0 || texture_width <= 0 || texture_height <= 0 ||
                offset_x < 0 || offset_y < 0 || region_width <= 0 || region_height <= 0 ||
                offset_x + region_width > texture_width ||
                offset_y + region_height > texture_height ||
                rgba_size != static_cast<std::size_t>(region_width) *
                                   static_cast<std::size_t>(region_height) * 4u) {
                return false;
            }
            VulkanContext* const ctx = getVulkanUiTextureContext();
            if (!ctx || !init(*ctx)) {
                return false;
            }

            if (!ensureImage(texture_width, texture_height)) {
                return false;
            }

            // Reap completed uploads; block only if the in-flight ring is full.
            if (!enforcePendingUploadBound()) {
                return false;
            }

            const VkDeviceSize upload_size = static_cast<VkDeviceSize>(rgba_size);
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
            if (!createBuffer(upload_size,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              staging_buffer,
                              staging_allocation)) {
                return false;
            }

            if (!writeAllocation(staging_allocation, rgba, upload_size)) {
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
            pending_uploads.push_back(PendingUpload{fence, command_buffer, staging_buffer, staging_allocation});
            return true;
        }

        [[nodiscard]] bool uploadRgba(const std::vector<std::uint8_t>& rgba,
                                      const int new_width,
                                      const int new_height) {
            return uploadRgbaRegion(rgba.data(), rgba.size(), new_width, new_height,
                                    0, 0, new_width, new_height);
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
            const std::vector<std::uint8_t> rgba = toRgba(pixels, region_width, region_height, channels);
            return uploadRgbaRegion(rgba.data(), rgba.size(), texture_width,
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
            if (channels == 4) {
                const std::size_t rgba_size = static_cast<std::size_t>(new_width) *
                                              static_cast<std::size_t>(new_height) * 4u;
                return uploadRgbaRegion(pixels, rgba_size, new_width, new_height,
                                        0, 0, new_width, new_height);
            }
            return uploadRgba(toRgba(pixels, new_width, new_height, channels), new_width, new_height);
        }

        void destroyImage() {
            waitAndReleasePendingUpload();
            const bool has_interop_resources =
                mode == Mode::CudaInterop || interop.valid() ||
                interop_image.image != VK_NULL_HANDLE ||
                interop_semaphore.semaphore != VK_NULL_HANDLE;
            if (has_interop_resources) {
                if (g_texture_upload_stream.valid() && !g_texture_upload_stream.synchronize()) {
                    LOG_WARN("Vulkan UI texture CUDA upload drain failed during image destruction: {}",
                             g_texture_upload_stream.lastError());
                }
                if (context != nullptr) {
                    if (!context->waitForSubmittedFrames()) {
                        LOG_WARN("Vulkan UI texture could not drain submitted frames before image destruction: {}",
                                 context->lastError());
                    }
                    if (!context->waitForImmediateSubmits()) {
                        LOG_WARN("Vulkan UI texture could not drain immediate transitions before image destruction: {}",
                                 context->lastError());
                    }
                }
                if (image != VK_NULL_HANDLE) {
                    image_barriers.forgetImage(image);
                }
                interop.reset();
                if (context) {
                    context->destroyExternalSemaphore(interop_semaphore);
                    context->destroyExternalImage(interop_image);
                } else {
                    interop_image = {};
                    interop_semaphore = {};
                }
                image = VK_NULL_HANDLE;
                image_view = VK_NULL_HANDLE;
                image_allocation = VK_NULL_HANDLE;
                interop_timeline_value = 0;
            } else {
                if (image_view != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, image_view, nullptr);
                    image_view = VK_NULL_HANDLE;
                }
                if (image != VK_NULL_HANDLE) {
                    if (!image_vram_label.empty()) {
                        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                            "vulkan.ui_texture.image",
                            image_vram_label,
                            0);
                    }
                    image_barriers.forgetImage(image);
                    vmaDestroyImage(allocator, image, image_allocation);
                    image = VK_NULL_HANDLE;
                    image_allocation = VK_NULL_HANDLE;
                }
            }
            image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_vram_label.clear();
            mode = Mode::Uninitialized;
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
                                 const int expected_height,
                                 const bool flip_y) {
        if (!impl_) {
            impl_ = new Impl();
        }
        if (image.is_valid() && image.device() == lfs::core::Device::CUDA &&
            impl_->mode != Impl::Mode::Cpu) {
            if (impl_->uploadCudaTensorImpl(image, expected_width, expected_height, flip_y))
                return true;
        }
        const std::vector<std::uint8_t> rgba = tensorToRgba(image, expected_width, expected_height, flip_y);
        return impl_->uploadRgba(rgba, expected_width, expected_height);
    }

    std::uintptr_t VulkanUiTexture::textureId() const {
        if (!impl_) {
            return 0;
        }
        impl_->tryReleasePendingUpload();
        return reinterpret_cast<std::uintptr_t>(impl_->descriptor_set);
    }

    std::string VulkanUiTexture::rmlSrcUrl(const int width, const int height) const {
        if (!impl_) {
            return {};
        }
        impl_->tryReleasePendingUpload();
        return RenderInterface_VK::MakeExternalTextureSource(impl_->image_view, impl_->sampler, width, height);
    }

    bool VulkanUiTexture::valid() const {
        if (!impl_) {
            return false;
        }
        impl_->tryReleasePendingUpload();
        if (impl_->image_view == VK_NULL_HANDLE) {
            return false;
        }
        return impl_->mode == Impl::Mode::CudaInterop || impl_->descriptor_set != VK_NULL_HANDLE;
    }

    void VulkanUiTexture::reset() {
        if (impl_) {
            impl_->reset();
        }
    }

} // namespace lfs::vis::gui
