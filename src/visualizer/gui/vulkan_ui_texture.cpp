/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/vulkan_ui_texture.hpp"

#include "config.h"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "gui/rmlui/rmlui_vk_backend.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/image_tensor.hpp"
#include "rendering/vulkan_wait.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_image_barrier_tracker.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis::gui {

    namespace {
        VulkanContext* g_texture_context = nullptr;
        lfs::rendering::CudaVulkanUploadStream g_texture_upload_stream;
        // Process-wide, not per-VulkanUiTexture: image/view handles are recycled
        // across instances, so a per-object generation is not a unique Rml cache key.
        std::atomic<std::uint64_t> g_external_src_incarnation{1};

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

        [[nodiscard]] lfs::core::Tensor prepareUiRgba8(const lfs::core::Tensor& image,
                                                       const int expected_width,
                                                       const int expected_height,
                                                       const bool flip_y) {
            if (!image.is_valid() || expected_width <= 0 || expected_height <= 0) {
                return {};
            }

            lfs::core::Tensor formatted = lfs::rendering::prepareImageRgba8(image, flip_y);
            if (!formatted.is_valid()) {
                LOG_ERROR("Vulkan UI texture upload received unsupported tensor shape [{}, {}, {}]",
                          image.ndim() >= 1 ? image.size(0) : 0,
                          image.ndim() >= 2 ? image.size(1) : 0,
                          image.ndim() >= 3 ? image.size(2) : 0);
                return {};
            }

            const int height = static_cast<int>(formatted.size(0));
            const int width = static_cast<int>(formatted.size(1));
            if (width != expected_width || height != expected_height) {
                LOG_ERROR("Vulkan UI texture upload dimension mismatch: {}x{} vs {}x{}",
                          width, height, expected_width, expected_height);
                return {};
            }
            return formatted;
        }

        // Original CUDA surface copy: Float32/UInt8 1/3/4-channel tensors. Gray stays
        // 1-channel (the kernel expands it); Float16 is excluded so prepareImageRgba8
        // owns that conversion instead of locking a packed RGBA8 into the interop path.
        [[nodiscard]] bool cudaDirectUploadEligible(const lfs::core::Tensor& image,
                                                    const int expected_width,
                                                    const int expected_height) {
            if (!image.is_valid() || image.ndim() != 3 || expected_width <= 0 ||
                expected_height <= 0 || image.device() != lfs::core::Device::GPU ||
                lfs::core::gpu_backend_of(image) != lfs::core::GpuBackend::CUDA) {
                return false;
            }
            const auto dtype = image.dtype();
            if (dtype != lfs::core::DataType::Float32 && dtype != lfs::core::DataType::UInt8) {
                return false;
            }
            const auto layout = lfs::rendering::detectImageLayout(image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                return false;
            }
            const int channels = lfs::rendering::imageChannels(image, layout);
            if (channels != 1 && channels != 3 && channels != 4) {
                return false;
            }
            return lfs::rendering::imageWidth(image, layout) == expected_width &&
                   lfs::rendering::imageHeight(image, layout) == expected_height;
        }

        // When a timeline semaphore shares a submit with binary semaphores, every wait
        // and signal slot needs a value. Binary slots use 0 (ignored by the spec).
        struct MixedTimelineSubmit {
            VkTimelineSemaphoreSubmitInfo info{};
            std::array<std::uint64_t, 2> wait_values{};
            std::array<std::uint64_t, 2> signal_values{};

            void attach(VkSubmitInfo& submit,
                        const std::span<const std::uint64_t> waits,
                        const std::span<const std::uint64_t> signals) {
                info = {};
                info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
                const std::uint32_t wait_n = submit.waitSemaphoreCount;
                const std::uint32_t signal_n = submit.signalSemaphoreCount;
                for (std::uint32_t i = 0; i < wait_n && i < wait_values.size(); ++i) {
                    wait_values[i] = i < waits.size() ? waits[i] : 0;
                }
                for (std::uint32_t i = 0; i < signal_n && i < signal_values.size(); ++i) {
                    signal_values[i] = i < signals.size() ? signals[i] : 0;
                }
                info.waitSemaphoreValueCount = wait_n;
                info.pWaitSemaphoreValues = wait_n > 0 ? wait_values.data() : nullptr;
                info.signalSemaphoreValueCount = signal_n;
                info.pSignalSemaphoreValues = signal_n > 0 ? signal_values.data() : nullptr;
                submit.pNext = &info;
            }
        };

    } // namespace

    void setVulkanUiTextureContext(VulkanContext* const context) {
        if (context == nullptr) {
            if (g_texture_context != nullptr) {
                if (!g_texture_context->waitForSubmittedFrames()) {
                    LOG_WARN(
                        "Vulkan UI texture context shutdown could not wait for submitted frames: {}",
                        g_texture_context->lastError());
                }
                if (!g_texture_context->waitForImmediateSubmits()) {
                    LOG_WARN(
                        "Vulkan UI texture context shutdown could not drain immediate transitions: {}",
                        g_texture_context->lastError());
                }
            }
            if (const std::size_t remaining = VulkanUiTexture::serviceOrphanedImpls(true);
                remaining != 0) {
                LOG_ERROR(
                    "Vulkan UI texture retaining {} quarantined impl(s) after context shutdown",
                    remaining);
                // These objects deliberately retain GPU-live resources. Never
                // revisit their context after it has been destroyed.
                VulkanUiTexture::orphaned_impls_.clear();
            }
            if (g_texture_upload_stream.valid() && !g_texture_upload_stream.synchronize()) {
                LOG_WARN("CUDA/Vulkan UI texture stream synchronization failed during shutdown: {}",
                         g_texture_upload_stream.lastError());
            }
            g_texture_upload_stream.reset();
        } else if (lfs::core::gpu_backend_available(lfs::core::GpuBackend::CUDA) &&
                   !g_texture_upload_stream.valid() && !g_texture_upload_stream.init()) {
            LOG_WARN("Could not create the non-blocking CUDA/Vulkan UI texture stream: {}",
                     g_texture_upload_stream.lastError());
        }
        g_texture_context = context;
        if (context != nullptr)
            VulkanUiTexture::serviceOrphanedImpls(false);
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
        VkQueue compute_queue = VK_NULL_HANDLE;
        std::uint32_t compute_queue_family = 0;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandPool compute_command_pool = VK_NULL_HANDLE;
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
        std::uint64_t image_generation_ = 0;
        std::uint64_t src_incarnation_ = 0;
        struct PendingUpload {
            VkFence fence = VK_NULL_HANDLE;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
            std::shared_ptr<void> keep_alive;
            VkCommandBuffer compute_command_buffer = VK_NULL_HANDLE;
            VkSemaphore copy_semaphore = VK_NULL_HANDLE;
            VkCommandBuffer release_command_buffer = VK_NULL_HANDLE;
            VkSemaphore release_semaphore = VK_NULL_HANDLE;
            VkFence compute_fence = VK_NULL_HANDLE;
            VkFence release_fence = VK_NULL_HANDLE;
        };
        struct RgbaRegion {
            const std::uint8_t* pixels = nullptr;
            std::size_t size = 0;
            int texture_width = 0;
            int texture_height = 0;
            int x = 0;
            int y = 0;
            int width = 0;
            int height = 0;
        };
        // Bounded ring of in-flight uploads. Uploads to the same image serialize on the graphics
        // queue (no semaphores), so a depth > 1 only defers staging-buffer reclamation; it does not
        // race the GPU. The main thread blocks only when the ring is full.
        static constexpr std::size_t kMaxPendingUploads = 3;
        std::vector<PendingUpload> pending_uploads;

        struct RetiredImage {
            VkImage image = VK_NULL_HANDLE;
            VkImageView image_view = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            std::uint64_t retire_serial = 0;
            std::uint64_t generation = 0;
            std::string vram_label;
            std::vector<PendingUpload> pending_uploads;
        };
        static constexpr std::size_t kMaxRetiredImages = 8;
        std::vector<RetiredImage> retired_images;

        struct RetiredInteropImage {
            VulkanContext::ExternalImage image{};
            VulkanContext::ExternalSemaphore semaphore{};
            std::uint64_t retire_serial = 0;
            std::uint64_t generation = 0;
        };
        std::vector<RetiredInteropImage> retired_interop_images;

        int width = 0;
        int height = 0;

        // CUDA-Vulkan interop path: bypasses CPU readback by writing the rasterizer's
        // CUDA tensor directly into the VkImage via a shared external memory handle.
        Mode mode = Mode::Uninitialized;
        bool skip_host_fallback_ = false;
        bool graphics_sampling_quarantined_ = false;
        VulkanContext::ExternalImage interop_image{};
        VulkanContext::ExternalSemaphore interop_semaphore{};
        lfs::rendering::CudaVulkanInterop interop;
        std::uint64_t interop_timeline_value = 0;
        bool interop_disabled = false;

        void destroyPendingUpload(PendingUpload& upload) {
            if (upload.command_buffer != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, command_pool, 1, &upload.command_buffer);
            }
            if (upload.release_command_buffer != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, command_pool, 1, &upload.release_command_buffer);
            }
            if (upload.compute_command_buffer != VK_NULL_HANDLE &&
                compute_command_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, compute_command_pool, 1, &upload.compute_command_buffer);
            }
            if (upload.staging_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, upload.staging_buffer, upload.staging_allocation);
            }
            if (upload.copy_semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, upload.copy_semaphore, nullptr);
            }
            if (upload.release_semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, upload.release_semaphore, nullptr);
            }
            if (upload.compute_fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, upload.compute_fence, nullptr);
            }
            if (upload.release_fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, upload.release_fence, nullptr);
            }
            if (upload.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, upload.fence, nullptr);
            }
            upload = {};
        }

        // Cross-family PendingUpload may hold acquire, compute, and release fences.
        // Host visibility of the acquire fence does not imply host visibility of
        // the others; destroy only when every non-null fence is Ready.
        [[nodiscard]] bool pendingUploadFencesReady(const PendingUpload& upload,
                                                    const bool wait,
                                                    const std::string_view fingerprint) const {
            bool ready = true;
            const std::array<VkFence, 3> fences{
                upload.fence, upload.compute_fence, upload.release_fence};
            for (const VkFence fence : fences) {
                if (fence == VK_NULL_HANDLE) {
                    continue;
                }
                if (!wait) {
                    if (vkGetFenceStatus(device, fence) != VK_SUCCESS) {
                        return false;
                    }
                    continue;
                }
                lfs::rendering::WaitContext wait_ctx;
                wait_ctx.fingerprint = fingerprint;
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                if (!wait_outcome.has_value() ||
                    *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
                    LOG_ERROR(
                        "Vulkan UI texture pending-upload wait did not reach Ready "
                        "(fence={:#x}): {} — retaining upload resources",
                        lfs::rendering::vkHandleValue(fence),
                        formatWaitFailure(wait_outcome));
                    ready = false;
                }
            }
            return ready;
        }

        // Reap entries whose GPU work has completed. Non-blocking.
        void tryReleasePendingUpload() {
            releaseRetiredImages(false);
            auto write = pending_uploads.begin();
            for (auto read = pending_uploads.begin(); read != pending_uploads.end(); ++read) {
                if (pendingUploadFencesReady(*read, false, {})) {
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
                if (pendingUploadFencesReady(*read, true, "ui_texture.pending_upload.wait")) {
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

        [[nodiscard]] bool releaseRetiredUploads(RetiredImage& retired, const bool wait) {
            auto write = retired.pending_uploads.begin();
            bool all_ready = true;
            for (auto read = retired.pending_uploads.begin();
                 read != retired.pending_uploads.end(); ++read) {
                if (pendingUploadFencesReady(
                        *read, wait, wait ? "ui_texture.retired_upload.wait" : std::string_view{})) {
                    destroyPendingUpload(*read);
                } else {
                    all_ready = false;
                    if (write != read)
                        *write = *read;
                    ++write;
                }
            }
            retired.pending_uploads.erase(write, retired.pending_uploads.end());
            return all_ready;
        }

        [[nodiscard]] std::uint64_t displayedImageRetireSerial() const {
            if (!context)
                return 0;
            std::uint64_t serial = context->lastSuccessfulFrameSubmitSerial();
            if (context->hasActiveFrame()) {
                // Current command buffer is unsubmitted and may still sample this view.
                serial = std::max(serial, context->lastFrameSubmitSerial() + 1);
            }
            return serial;
        }

        void destroyRetiredInteropImage(RetiredInteropImage& retired) {
            if (retired.image.image != VK_NULL_HANDLE)
                image_barriers.forgetImage(retired.image.image, retired.generation);
            if (context) {
                context->destroyExternalSemaphore(retired.semaphore);
                context->destroyExternalImage(retired.image);
            } else {
                retired.image = {};
                retired.semaphore = {};
            }
            retired = {};
        }

        void releaseRetiredImages(const bool wait) {
            if (wait && context) {
                // Only wait serials that already have a submit. lastFrame+1 (current
                // unsubmitted GUI frame) has no fence yet; waitForRetiredFrameSubmitSerial
                // returns true on an empty fence list without the serial having retired.
                const std::uint64_t submitted = context->lastFrameSubmitSerial();
                std::uint64_t need = 0;
                for (const auto& retired : retired_images) {
                    if (retired.retire_serial > 0 && retired.retire_serial <= submitted)
                        need = std::max(need, retired.retire_serial);
                }
                for (const auto& retired : retired_interop_images) {
                    if (retired.retire_serial > 0 && retired.retire_serial <= submitted)
                        need = std::max(need, retired.retire_serial);
                }
                if (need != 0 && !context->waitForRetiredFrameSubmitSerial(need))
                    LOG_WARN("Vulkan UI texture could not retire frame serial {}: {}",
                             need,
                             context->lastError());
            }
            // No context: only serial-0 (no frame dependency) entries may free.
            // Do not treat missing context as "all frames retired".
            const auto retired_serial = context ? context->retiredFrameSubmitSerial() : 0;
            auto write = retired_images.begin();
            for (auto read = retired_images.begin(); read != retired_images.end(); ++read) {
                const bool uploads_ready = releaseRetiredUploads(*read, wait);
                if (read->retire_serial <= retired_serial && uploads_ready) {
                    image_barriers.forgetImage(read->image, read->generation);
                    if (read->image_view != VK_NULL_HANDLE)
                        vkDestroyImageView(device, read->image_view, nullptr);
                    if (read->image != VK_NULL_HANDLE)
                        vmaDestroyImage(allocator, read->image, read->allocation);
                    if (!read->vram_label.empty()) {
                        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                            "vulkan.ui_texture.image", read->vram_label, 0);
                    }
                    continue;
                }
                if (write != read)
                    *write = std::move(*read);
                ++write;
            }
            retired_images.erase(write, retired_images.end());

            auto interop_write = retired_interop_images.begin();
            for (auto read = retired_interop_images.begin(); read != retired_interop_images.end();
                 ++read) {
                if (read->retire_serial <= retired_serial) {
                    destroyRetiredInteropImage(*read);
                    continue;
                }
                if (interop_write != read)
                    *interop_write = std::move(*read);
                ++interop_write;
            }
            retired_interop_images.erase(interop_write, retired_interop_images.end());
        }

        void retireCurrentImage() {
            if (image == VK_NULL_HANDLE)
                return;
            tryReleasePendingUpload();
            // Serial 0 means no frame sampling dependency, not "no upload
            // dependency". Keep submitted pending uploads with the image until
            // their fences are Ready. An unsubmitted GUI frame is lastFrame+1.
            const std::uint64_t serial = displayedImageRetireSerial();
            if (retired_images.size() >= kMaxRetiredImages)
                releaseRetiredImages(true);
            retired_images.push_back({
                .image = image,
                .image_view = image_view,
                .allocation = image_allocation,
                .retire_serial = serial,
                .generation = image_generation_,
                .vram_label = std::move(image_vram_label),
                .pending_uploads = std::move(pending_uploads),
            });
            pending_uploads.clear();
            image = VK_NULL_HANDLE;
            image_view = VK_NULL_HANDLE;
            image_allocation = VK_NULL_HANDLE;
            image_vram_label.clear();
        }

        [[nodiscard]] bool retireCurrentInteropImage() {
            const bool has_interop =
                interop.valid() || interop_image.image != VK_NULL_HANDLE ||
                interop_semaphore.semaphore != VK_NULL_HANDLE;
            if (!has_interop)
                return true;
            if (g_texture_upload_stream.valid() && !g_texture_upload_stream.synchronize()) {
                LOG_WARN("Vulkan UI texture CUDA upload drain failed during interop retirement: {}",
                         g_texture_upload_stream.lastError());
                return false;
            }
            if (context != nullptr && !context->waitForImmediateSubmits()) {
                LOG_WARN(
                    "Vulkan UI texture could not drain immediate transitions before interop retirement: {}",
                    context->lastError());
                return false;
            }
            interop.reset();
            const std::uint64_t serial = displayedImageRetireSerial();
            releaseRetiredImages(false);
            if (retired_interop_images.size() >= kMaxRetiredImages)
                releaseRetiredImages(true);
            retired_interop_images.push_back({
                .image = interop_image,
                .semaphore = interop_semaphore,
                .retire_serial = serial,
                .generation = image_generation_,
            });
            interop_image = {};
            interop_semaphore = {};
            image = VK_NULL_HANDLE;
            image_view = VK_NULL_HANDLE;
            image_allocation = VK_NULL_HANDLE;
            interop_timeline_value = 0;
            image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            return true;
        }

        [[nodiscard]] bool hasGpuLifetime() const {
            return !pending_uploads.empty() || !retired_images.empty() ||
                   !retired_interop_images.empty() || image != VK_NULL_HANDLE || interop.valid() ||
                   interop_image.image != VK_NULL_HANDLE ||
                   interop_semaphore.semaphore != VK_NULL_HANDLE;
        }

        [[nodiscard]] bool tryFinishDestroy(const bool wait) {
            if (wait)
                waitAndReleasePendingUpload();
            else
                tryReleasePendingUpload();
            if (interop.valid() || interop_image.image != VK_NULL_HANDLE ||
                interop_semaphore.semaphore != VK_NULL_HANDLE) {
                if (!retireCurrentInteropImage())
                    return false;
            } else {
                retireCurrentImage();
            }
            releaseRetiredImages(wait);
            return !hasGpuLifetime();
        }

        void destroyDeviceObjects() {
            if (hasGpuLifetime())
                return;
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
            if (compute_command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, compute_command_pool, nullptr);
                compute_command_pool = VK_NULL_HANDLE;
            }
            device = VK_NULL_HANDLE;
            context = nullptr;
            allocator = VK_NULL_HANDLE;
            graphics_queue = VK_NULL_HANDLE;
            graphics_queue_family = 0;
            compute_queue = VK_NULL_HANDLE;
            compute_queue_family = 0;
            mode = Mode::Uninitialized;
            image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            src_incarnation_ = 0;
            skip_host_fallback_ = false;
            graphics_sampling_quarantined_ = false;
            width = 0;
            height = 0;
        }

        // Block only when the ring is full: wait on the oldest entry to free a slot.
        // Returns false if the oldest entry cannot be drained (quarantine/cancel/error).
        [[nodiscard]] bool enforcePendingUploadBound() {
            tryReleasePendingUpload();
            while (pending_uploads.size() >= kMaxPendingUploads) {
                PendingUpload& oldest = pending_uploads.front();
                if (!pendingUploadFencesReady(oldest, true, "ui_texture.pending_upload.bound")) {
                    LOG_ERROR(
                        "Vulkan UI texture pending-upload bound wait did not reach Ready "
                        "(pending={}) — retaining oldest slot",
                        pending_uploads.size());
                    return false;
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
            compute_queue = ctx.computeQueue();
            compute_queue_family = ctx.computeQueueFamily();
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
            if (ctx.hasDedicatedComputeQueue() &&
                compute_queue != VK_NULL_HANDLE &&
                compute_queue_family != graphics_queue_family) {
                VkCommandPoolCreateInfo compute_pool_info = pool_info;
                compute_pool_info.queueFamilyIndex = compute_queue_family;
                if (vkCreateCommandPool(device, &compute_pool_info, nullptr, &compute_command_pool) !=
                    VK_SUCCESS) {
                    LOG_WARN("Failed to create Vulkan UI texture compute command pool; "
                             "native Vulkan tensor uploads will fall back to host");
                    compute_command_pool = VK_NULL_HANDLE;
                }
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
                if (!reset(true)) {
                    LOG_ERROR("Vulkan UI texture retaining device objects after sampler init failure");
                }
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
                if (!reset(true)) {
                    LOG_ERROR("Vulkan UI texture retaining device objects after descriptor-layout init failure");
                }
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
                if (!reset(true)) {
                    LOG_ERROR("Vulkan UI texture retaining device objects after descriptor-pool init failure");
                }
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
                submit_status = lfs::rendering::vk_queue_submit_synced(graphics_queue, 1, &submit_info, submit_fence);
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
            image_barriers.registerImage(image, image_generation_, VK_IMAGE_ASPECT_COLOR_BIT, old_layout);
            image_barriers.transitionImage(command_buffer, image, image_generation_, VK_IMAGE_ASPECT_COLOR_BIT, new_layout);
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
                releaseRetiredImages(false);
                retireCurrentImage();
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
            std::array<std::uint32_t, 2> concurrent_families{};
            if (context->hasDedicatedComputeQueue() &&
                context->computeQueueFamily() != graphics_queue_family) {
                concurrent_families[0] = graphics_queue_family;
                concurrent_families[1] = context->computeQueueFamily();
                image_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
                image_info.queueFamilyIndexCount = 2;
                image_info.pQueueFamilyIndices = concurrent_families.data();
            }

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
            ++image_generation_;
            src_incarnation_ = g_external_src_incarnation.fetch_add(1, std::memory_order_relaxed);
            image_barriers.registerImage(image, image_generation_, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);

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

            releaseRetiredImages(false);
            if (!retireCurrentInteropImage())
                return false;
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
                !context->createExternalTimelineSemaphore(0, interop_semaphore, "vulkan.ui_texture.interop_semaphore")) {
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
            ++image_generation_;
            src_incarnation_ = g_external_src_incarnation.fetch_add(1, std::memory_order_relaxed);
            image_barriers.registerImage(image, image_generation_, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            mode = Mode::CudaInterop;

            // Skip allocating an internal descriptor set: the interop path is consumed via RmlUi
            // (which allocates its own descriptor against its texture-set layout).
            descriptor_set = VK_NULL_HANDLE;
            return true;
        }

        [[nodiscard]] VkCommandBuffer beginCommands(const VkCommandPool pool) const {
            if (pool == VK_NULL_HANDLE) {
                return VK_NULL_HANDLE;
            }
            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = pool;
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
                vkFreeCommandBuffers(device, pool, 1, &command_buffer);
                return VK_NULL_HANDLE;
            }
            return command_buffer;
        }

        using AccessScope = VulkanImageBarrierTracker::AccessScope;

        void restoreImageTracker(const VkImageLayout layout) {
            if (image != VK_NULL_HANDLE) {
                image_barriers.registerImage(
                    image, image_generation_, VK_IMAGE_ASPECT_COLOR_BIT, layout);
            }
        }

        void recordSameLayoutImageBarrier(const VkCommandBuffer command_buffer,
                                          const VkImageLayout layout,
                                          const AccessScope source,
                                          const AccessScope destination) const {
            if (command_buffer == VK_NULL_HANDLE || image == VK_NULL_HANDLE) {
                return;
            }
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = source.stage;
            barrier.srcAccessMask = source.access;
            barrier.dstStageMask = destination.stage;
            barrier.dstAccessMask = destination.access;
            barrier.oldLayout = layout;
            barrier.newLayout = layout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            VkDependencyInfo dependency{};
            dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency.imageMemoryBarrierCount = 1;
            dependency.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command_buffer, &dependency);
        }

        void recordBufferToImageCopy(const VkCommandBuffer command_buffer,
                                     const VkBuffer buffer,
                                     const VkDeviceSize buffer_offset,
                                     const int region_width,
                                     const int region_height,
                                     const bool copy_on_compute) {
            const AccessScope transfer_write{
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
            };
            const AccessScope transfer_read{
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT,
            };
            const AccessScope shader_write{
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
            };
            const AccessScope fragment_read{
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT,
            };
            // Compute-only queues cannot wait on fragment. The graphics release
            // semaphore covers prior samples; this dest makes the copy available
            // before the compute->graphics signal.
            const AccessScope copy_complete =
                copy_on_compute
                    ? AccessScope{VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE}
                    : fragment_read;

            VkBufferMemoryBarrier2 buffer_barrier{};
            buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            buffer_barrier.srcStageMask = shader_write.stage;
            buffer_barrier.srcAccessMask = shader_write.access;
            buffer_barrier.dstStageMask = transfer_read.stage;
            buffer_barrier.dstAccessMask = transfer_read.access;
            buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barrier.buffer = buffer;
            buffer_barrier.offset = buffer_offset;
            buffer_barrier.size = static_cast<VkDeviceSize>(region_width) *
                                  static_cast<VkDeviceSize>(region_height) * 4u;
            VkDependencyInfo buffer_dependency{};
            buffer_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            buffer_dependency.bufferMemoryBarrierCount = 1;
            buffer_dependency.pBufferMemoryBarriers = &buffer_barrier;
            vkCmdPipelineBarrier2(command_buffer, &buffer_dependency);

            if (copy_on_compute) {
                // Prior fragment samples are ordered by the graphics release
                // semaphore, not by this barrier (compute queues reject fragment).
                image_barriers.transitionImage(command_buffer,
                                               image,
                                               image_generation_,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                               AccessScope{},
                                               transfer_write);
            } else {
                image_barriers.transitionImage(command_buffer,
                                               image,
                                               image_generation_,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            }

            VkBufferImageCopy copy{};
            copy.bufferOffset = buffer_offset;
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.layerCount = 1;
            copy.imageExtent = {static_cast<std::uint32_t>(region_width),
                                static_cast<std::uint32_t>(region_height),
                                1};
            vkCmdCopyBufferToImage(command_buffer,
                                   buffer,
                                   image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1,
                                   &copy);

            if (copy_on_compute) {
                image_barriers.transitionImage(command_buffer,
                                               image,
                                               image_generation_,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                               transfer_write,
                                               copy_complete);
            } else {
                image_barriers.transitionImage(command_buffer,
                                               image,
                                               image_generation_,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }

        // Native Vulkan buffer->image. Same-family copies on graphics with tracked
        // fragment->transfer barriers. Cross-family exclusive tensor buffers copy on
        // compute after a graphics release semaphore; compute signals a binary
        // semaphore that graphics waits on. Packed RGBA8 that aliases the caller is
        // cloned so later mutation cannot race the GPU copy. Failed submits after
        // GPU work is accepted keep in-flight command buffers / semaphores / fences
        // / keep_alive (AMB-4) instead of freeing them.
        [[nodiscard]] bool uploadVulkanTensorImpl(const lfs::core::Tensor& rgba) {
            skip_host_fallback_ = graphics_sampling_quarantined_;
            if (graphics_sampling_quarantined_) {
                return false;
            }
            if (!rgba.is_valid() ||
                lfs::core::gpu_backend_of(rgba) != lfs::core::GpuBackend::Vulkan ||
                !lfs::core::vulkan_backend_adopted()) {
                return false;
            }
            VulkanContext* const ctx = getVulkanUiTextureContext();
            if (!ctx || !init(*ctx) || mode == Mode::CudaInterop) {
                return false;
            }

            // Flush producer writes before cloning so the snapshot copy is ordered
            // after them on the tensor recorder / timeline.
            if (!lfs::core::tensor_vulkan_buffer(rgba)) {
                return false;
            }
            const lfs::core::Tensor snapshot = rgba.clone();
            auto storage = lfs::core::tensor_vulkan_buffer(snapshot);
            if (!storage || storage->buffer == nullptr) {
                return false;
            }
            const int region_width = static_cast<int>(snapshot.size(1));
            const int region_height = static_cast<int>(snapshot.size(0));
            const VkDeviceSize packed_bytes = static_cast<VkDeviceSize>(region_width) *
                                              static_cast<VkDeviceSize>(region_height) * 4u;
            if (packed_bytes == 0 || storage->bytes < packed_bytes || (storage->offset % 4u) != 0) {
                return false;
            }
            if (!ensureImage(region_width, region_height)) {
                return false;
            }

            const std::uint32_t tensor_family = ctx->tensorBackendDevice().queue_family;
            const bool copy_on_compute = ctx->tensorBackendDevice().complete &&
                                         tensor_family != graphics_queue_family;
            if (copy_on_compute &&
                (compute_command_pool == VK_NULL_HANDLE || compute_queue == VK_NULL_HANDLE ||
                 tensor_family != compute_queue_family)) {
                return false;
            }
            if (copy_on_compute) {
                // Concurrent sharing does not order a previous graphics sample or
                // CPU upload against this compute write. Drain our pending ring,
                // then the graphics release semaphore covers submitted frames.
                waitAndReleasePendingUpload();
                if (!pending_uploads.empty()) {
                    return false;
                }
            } else if (!enforcePendingUploadBound()) {
                return false;
            }

            VkSemaphore timeline = VK_NULL_HANDLE;
            if (storage->pending_timeline_value != 0) {
                timeline = static_cast<VkSemaphore>(lfs::core::vulkan_backend_timeline());
                if (timeline == VK_NULL_HANDLE) {
                    return false;
                }
            }

            const VkImageLayout previous_layout = image_layout;
            PendingUpload pending{};
            const auto rollback_unsubmitted = [&]() {
                restoreImageTracker(previous_layout);
                destroyPendingUpload(pending);
            };

            const VkBuffer tensor_buffer = static_cast<VkBuffer>(storage->buffer);
            const VkCommandPool copy_pool = copy_on_compute ? compute_command_pool : command_pool;
            const VkCommandBuffer copy_cb = beginCommands(copy_pool);
            if (copy_cb == VK_NULL_HANDLE) {
                return false;
            }
            if (copy_on_compute) {
                pending.compute_command_buffer = copy_cb;
            } else {
                pending.command_buffer = copy_cb;
            }
            recordBufferToImageCopy(copy_cb,
                                    tensor_buffer,
                                    static_cast<VkDeviceSize>(storage->offset),
                                    region_width,
                                    region_height,
                                    copy_on_compute);
            if (vkEndCommandBuffer(copy_cb) != VK_SUCCESS) {
                LOG_ERROR("Failed to end Vulkan UI texture native copy command buffer");
                rollback_unsubmitted();
                return false;
            }

            const AccessScope graphics_all{
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            };
            const AccessScope graphics_release_dst{
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_NONE,
            };
            const AccessScope fragment_read{
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT,
            };
            const AccessScope transfer_write{
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
            };

            if (copy_on_compute) {
                VkSemaphoreCreateInfo semaphore_info{};
                semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
                if (vkCreateSemaphore(device, &semaphore_info, nullptr, &pending.release_semaphore) !=
                        VK_SUCCESS ||
                    vkCreateSemaphore(device, &semaphore_info, nullptr, &pending.copy_semaphore) !=
                        VK_SUCCESS) {
                    rollback_unsubmitted();
                    return false;
                }
                pending.release_command_buffer = beginCommands(command_pool);
                if (pending.release_command_buffer == VK_NULL_HANDLE) {
                    rollback_unsubmitted();
                    return false;
                }
                if (previous_layout != VK_IMAGE_LAYOUT_UNDEFINED) {
                    recordSameLayoutImageBarrier(pending.release_command_buffer,
                                                 previous_layout,
                                                 graphics_all,
                                                 graphics_release_dst);
                }
                if (vkEndCommandBuffer(pending.release_command_buffer) != VK_SUCCESS) {
                    LOG_ERROR("Failed to end Vulkan UI texture graphics release command buffer");
                    rollback_unsubmitted();
                    return false;
                }
                pending.command_buffer = beginCommands(command_pool);
                if (pending.command_buffer == VK_NULL_HANDLE) {
                    rollback_unsubmitted();
                    return false;
                }
                recordSameLayoutImageBarrier(pending.command_buffer,
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                             transfer_write,
                                             fragment_read);
                if (vkEndCommandBuffer(pending.command_buffer) != VK_SUCCESS) {
                    LOG_ERROR("Failed to end Vulkan UI texture graphics acquire command buffer");
                    rollback_unsubmitted();
                    return false;
                }
            }

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vkCreateFence(device, &fence_info, nullptr, &pending.fence) != VK_SUCCESS) {
                rollback_unsubmitted();
                return false;
            }
            if (copy_on_compute &&
                (vkCreateFence(device, &fence_info, nullptr, &pending.compute_fence) != VK_SUCCESS ||
                 vkCreateFence(device, &fence_info, nullptr, &pending.release_fence) != VK_SUCCESS)) {
                rollback_unsubmitted();
                return false;
            }

            const auto submit_cb = [&](const VkQueue queue,
                                       VkCommandBuffer command_buffer,
                                       const VkFence fence,
                                       const std::span<const VkSemaphore> waits,
                                       const std::span<const VkPipelineStageFlags> wait_stages,
                                       const std::span<const std::uint64_t> wait_values,
                                       const std::span<const VkSemaphore> signals,
                                       const bool includes_timeline) -> bool {
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                if (command_buffer != VK_NULL_HANDLE) {
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &command_buffer;
                }
                if (!waits.empty()) {
                    submit_info.waitSemaphoreCount = static_cast<std::uint32_t>(waits.size());
                    submit_info.pWaitSemaphores = waits.data();
                    submit_info.pWaitDstStageMask = wait_stages.data();
                }
                if (!signals.empty()) {
                    submit_info.signalSemaphoreCount = static_cast<std::uint32_t>(signals.size());
                    submit_info.pSignalSemaphores = signals.data();
                }
                MixedTimelineSubmit timeline_attach;
                if (includes_timeline) {
                    timeline_attach.attach(submit_info, wait_values, {});
                }
                const VkResult status =
                    lfs::rendering::vk_queue_submit_synced(queue, 1, &submit_info, fence);
                if (status != VK_SUCCESS) {
                    LOG_ERROR("Failed to submit Vulkan UI texture native copy: {}",
                              static_cast<int>(status));
                    return false;
                }
                return true;
            };

            if (copy_on_compute) {
                if (!submit_cb(graphics_queue,
                               pending.release_command_buffer,
                               pending.release_fence,
                               {},
                               {},
                               {},
                               std::span<const VkSemaphore>(&pending.release_semaphore, 1),
                               false)) {
                    rollback_unsubmitted();
                    return false;
                }

                std::array<VkSemaphore, 2> wait_semaphores{};
                std::array<VkPipelineStageFlags, 2> wait_stages{};
                std::array<std::uint64_t, 2> wait_values{};
                std::uint32_t wait_count = 0;
                if (timeline != VK_NULL_HANDLE) {
                    wait_semaphores[wait_count] = timeline;
                    wait_stages[wait_count] = VK_PIPELINE_STAGE_TRANSFER_BIT;
                    wait_values[wait_count] = storage->pending_timeline_value;
                    ++wait_count;
                }
                wait_semaphores[wait_count] = pending.release_semaphore;
                wait_stages[wait_count] = VK_PIPELINE_STAGE_TRANSFER_BIT;
                wait_values[wait_count] = 0;
                ++wait_count;
                if (!submit_cb(compute_queue,
                               pending.compute_command_buffer,
                               pending.compute_fence,
                               std::span<const VkSemaphore>(wait_semaphores.data(), wait_count),
                               std::span<const VkPipelineStageFlags>(wait_stages.data(), wait_count),
                               std::span<const std::uint64_t>(wait_values.data(), wait_count),
                               std::span<const VkSemaphore>(&pending.copy_semaphore, 1),
                               timeline != VK_NULL_HANDLE)) {
                    // Release is in flight. Keep its fence/CB/semaphore; free the
                    // unsubmitted compute and acquire resources. Image contents are
                    // unchanged, so host fallback is safe after tracker rollback.
                    LOG_ERROR(
                        "Vulkan: retaining UI texture graphics-release resources after compute "
                        "submit failure (command_buffer={:#x}, semaphore={:#x}, fence={:#x})",
                        lfs::rendering::vkHandleValue(pending.release_command_buffer),
                        lfs::rendering::vkHandleValue(pending.release_semaphore),
                        lfs::rendering::vkHandleValue(pending.release_fence));
                    vkFreeCommandBuffers(device, compute_command_pool, 1, &pending.compute_command_buffer);
                    pending.compute_command_buffer = VK_NULL_HANDLE;
                    vkFreeCommandBuffers(device, command_pool, 1, &pending.command_buffer);
                    pending.command_buffer = VK_NULL_HANDLE;
                    vkDestroySemaphore(device, pending.copy_semaphore, nullptr);
                    pending.copy_semaphore = VK_NULL_HANDLE;
                    vkDestroyFence(device, pending.compute_fence, nullptr);
                    pending.compute_fence = VK_NULL_HANDLE;
                    vkDestroyFence(device, pending.fence, nullptr);
                    pending.fence = pending.release_fence;
                    pending.release_fence = VK_NULL_HANDLE;
                    restoreImageTracker(previous_layout);
                    pending_uploads.push_back(std::move(pending));
                    return false;
                }

                VkPipelineStageFlags acquire_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                if (!submit_cb(graphics_queue,
                               pending.command_buffer,
                               pending.fence,
                               std::span<const VkSemaphore>(&pending.copy_semaphore, 1),
                               std::span<const VkPipelineStageFlags>(&acquire_stage, 1),
                               {},
                               {},
                               false)) {
                    // Compute is in flight and will finish the layout transition.
                    // Do not free its command buffer, semaphore, fence, or snapshot,
                    // and do not roll the tracker back. Skip host fallback.
                    LOG_ERROR(
                        "Vulkan: retaining UI texture compute-copy resources after graphics "
                        "handoff submit failure (command_buffer={:#x}, semaphore={:#x}, "
                        "fence={:#x})",
                        lfs::rendering::vkHandleValue(pending.compute_command_buffer),
                        lfs::rendering::vkHandleValue(pending.copy_semaphore),
                        lfs::rendering::vkHandleValue(pending.compute_fence));
                    vkFreeCommandBuffers(device, command_pool, 1, &pending.command_buffer);
                    pending.command_buffer = VK_NULL_HANDLE;
                    pending.keep_alive = std::move(storage->keep_alive);
                    image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                    // Acquire CB was not submitted. Retry a wait-only graphics submit
                    // so family 0 actually waits copy_semaphore (no CB required).
                    if (submit_cb(graphics_queue,
                                  VK_NULL_HANDLE,
                                  pending.fence,
                                  std::span<const VkSemaphore>(&pending.copy_semaphore, 1),
                                  std::span<const VkPipelineStageFlags>(&acquire_stage, 1),
                                  {},
                                  {},
                                  false)) {
                        pending_uploads.push_back(std::move(pending));
                        return true;
                    }

                    LOG_ERROR(
                        "Vulkan: graphics wait-only submit on copy semaphore also failed; "
                        "waiting compute fence before exposing the image");
                    vkDestroyFence(device, pending.fence, nullptr);
                    pending.fence = VK_NULL_HANDLE;

                    const auto latch_unsafe_copy =
                        [&](const lfs::Result<lfs::rendering::WaitOutcome>* wait_outcome) {
                            LOG_ERROR(
                                "Vulkan UI texture compute-copy wait did not reach Ready "
                                "(fence={:#x}): {} — retaining compute resources, skipping "
                                "host fallback",
                                lfs::rendering::vkHandleValue(pending.compute_fence),
                                wait_outcome != nullptr ? formatWaitFailure(*wait_outcome)
                                                        : "compute fence missing");
                            if (context != nullptr) {
                                if (wait_outcome != nullptr && !wait_outcome->has_value() &&
                                    wait_outcome->error().code() == lfs::ErrorCode::DeviceLost) {
                                    context->noteFailure(lfs::Exception(wait_outcome->error()));
                                } else {
                                    context->noteFailure(lfs::Exception(lfs::make_error(lfs::ErrorInit{
                                        .code = lfs::ErrorCode::DeadlineExceeded,
                                        .domain = lfs::ErrorDomain::Vulkan,
                                        .user_message = "Vulkan UI texture native copy is unsafe "
                                                        "to sample",
                                        .detail = "Compute copy did not complete after graphics "
                                                  "acquire submit failure",
                                        .detection = LFS_SOURCE_SITE_CURRENT(),
                                    })));
                                }
                            }
                            graphics_sampling_quarantined_ = true;
                            skip_host_fallback_ = true;
                        };

                    if (pending.compute_fence == VK_NULL_HANDLE) {
                        latch_unsafe_copy(nullptr);
                        pending_uploads.push_back(std::move(pending));
                        return false;
                    }

                    lfs::rendering::WaitContext wait_ctx;
                    wait_ctx.fingerprint = "ui_texture.native_copy.compute_fence";
                    if (context != nullptr) {
                        wait_ctx.is_quarantined = [ctx = context]() {
                            return ctx->rendererTerminalState() !=
                                   lfs::vis::RendererTerminalState::Running;
                        };
                    }
                    const auto wait_outcome = lfs::rendering::wait_fence_bounded(
                        device,
                        pending.compute_fence,
                        std::stop_token{},
                        lfs::rendering::VulkanWaitPolicy{},
                        wait_ctx);
                    if (!wait_outcome.has_value() ||
                        *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
                        latch_unsafe_copy(&wait_outcome);
                        pending_uploads.push_back(std::move(pending));
                        return false;
                    }

                    pending_uploads.push_back(std::move(pending));
                    return true;
                }
            } else {
                VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                const std::uint64_t wait_value = storage->pending_timeline_value;
                if (timeline != VK_NULL_HANDLE) {
                    if (!submit_cb(graphics_queue,
                                   pending.command_buffer,
                                   pending.fence,
                                   std::span<const VkSemaphore>(&timeline, 1),
                                   std::span<const VkPipelineStageFlags>(&wait_stage, 1),
                                   std::span<const std::uint64_t>(&wait_value, 1),
                                   {},
                                   true)) {
                        rollback_unsubmitted();
                        return false;
                    }
                } else if (!submit_cb(graphics_queue,
                                      pending.command_buffer,
                                      pending.fence,
                                      {},
                                      {},
                                      {},
                                      {},
                                      false)) {
                    rollback_unsubmitted();
                    return false;
                }
            }

            pending.keep_alive = std::move(storage->keep_alive);
            image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            pending_uploads.push_back(std::move(pending));
            return true;
        }

        [[nodiscard]] bool uploadCudaTensorImpl(const lfs::core::Tensor& tensor,
                                                const int expected_width,
                                                const int expected_height,
                                                const bool flip_y) {
            if (!tensor.is_valid() || tensor.device() != lfs::core::Device::GPU ||
                lfs::core::gpu_backend_of(tensor) != lfs::core::GpuBackend::CUDA) {
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

        [[nodiscard]] bool uploadRgbaRegions(const std::span<const RgbaRegion> regions) {
            if (regions.empty())
                return false;

            const int texture_width = regions.front().texture_width;
            const int texture_height = regions.front().texture_height;
            std::size_t total_size = 0;
            for (const RgbaRegion& region : regions) {
                if (!region.pixels || region.size == 0 || region.texture_width != texture_width ||
                    region.texture_height != texture_height || texture_width <= 0 || texture_height <= 0 ||
                    region.x < 0 || region.y < 0 || region.width <= 0 || region.height <= 0 ||
                    region.x + region.width > texture_width ||
                    region.y + region.height > texture_height ||
                    region.size != static_cast<std::size_t>(region.width) *
                                       static_cast<std::size_t>(region.height) * 4u ||
                    total_size > std::numeric_limits<std::size_t>::max() - region.size) {
                    return false;
                }
                total_size += region.size;
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

            const VkDeviceSize upload_size = static_cast<VkDeviceSize>(total_size);
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
            if (!createBuffer(upload_size,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              staging_buffer,
                              staging_allocation)) {
                return false;
            }

            void* mapped = nullptr;
            if (vmaMapMemory(allocator, staging_allocation, &mapped) != VK_SUCCESS || !mapped) {
                LOG_ERROR("Failed to map Vulkan UI texture staging memory");
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return false;
            }
            std::vector<VkBufferImageCopy> copy_regions;
            copy_regions.reserve(regions.size());
            VkDeviceSize buffer_offset = 0;
            for (const RgbaRegion& region : regions) {
                std::memcpy(static_cast<std::uint8_t*>(mapped) + buffer_offset,
                            region.pixels,
                            region.size);
                VkBufferImageCopy& copy = copy_regions.emplace_back();
                copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copy.imageSubresource.mipLevel = 0;
                copy.imageSubresource.baseArrayLayer = 0;
                copy.imageSubresource.layerCount = 1;
                copy.bufferOffset = buffer_offset;
                copy.imageOffset = {region.x, region.y, 0};
                copy.imageExtent = {static_cast<std::uint32_t>(region.width),
                                    static_cast<std::uint32_t>(region.height),
                                    1};
                buffer_offset += static_cast<VkDeviceSize>(region.size);
            }
            const VkResult flush_result = vmaFlushAllocation(allocator, staging_allocation, 0, upload_size);
            vmaUnmapMemory(allocator, staging_allocation);
            if (flush_result != VK_SUCCESS) {
                LOG_ERROR("Failed to flush Vulkan UI texture staging memory");
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return false;
            }

            VkCommandBuffer command_buffer = beginSingleTimeCommands();
            if (command_buffer == VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return false;
            }

            transitionImageLayout(command_buffer, image_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            vkCmdCopyBufferToImage(command_buffer,
                                   staging_buffer,
                                   image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   static_cast<std::uint32_t>(copy_regions.size()),
                                   copy_regions.data());

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
            const VkResult submit_status = lfs::rendering::vk_queue_submit_synced(graphics_queue, 1, &submit_info, fence);
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

        [[nodiscard]] bool uploadRgbaRegion(const std::uint8_t* const rgba,
                                            const std::size_t rgba_size,
                                            const int texture_width,
                                            const int texture_height,
                                            const int offset_x,
                                            const int offset_y,
                                            const int region_width,
                                            const int region_height) {
            const RgbaRegion region{
                .pixels = rgba,
                .size = rgba_size,
                .texture_width = texture_width,
                .texture_height = texture_height,
                .x = offset_x,
                .y = offset_y,
                .width = region_width,
                .height = region_height,
            };
            return uploadRgbaRegions(std::span<const RgbaRegion>(&region, 1));
        }

        [[nodiscard]] bool uploadRgba(const std::vector<std::uint8_t>& rgba,
                                      const int new_width,
                                      const int new_height) {
            return uploadRgbaRegion(rgba.data(), rgba.size(), new_width, new_height,
                                    0, 0, new_width, new_height);
        }

        [[nodiscard]] bool uploadRegions(const std::span<const VulkanUiTexture::Region> regions) {
            if (regions.empty())
                return false;

            std::vector<std::vector<std::uint8_t>> rgba_regions;
            rgba_regions.reserve(regions.size());
            std::vector<RgbaRegion> upload_regions;
            upload_regions.reserve(regions.size());
            for (const VulkanUiTexture::Region& region : regions) {
                if (!region.pixels || region.width <= 0 || region.height <= 0 ||
                    region.channels <= 0 || region.channels > 4) {
                    return false;
                }
                rgba_regions.push_back(toRgba(region.pixels,
                                              region.width,
                                              region.height,
                                              region.channels));
                if (rgba_regions.back().empty())
                    return false;
                upload_regions.push_back({
                    .pixels = rgba_regions.back().data(),
                    .size = rgba_regions.back().size(),
                    .texture_width = region.texture_width,
                    .texture_height = region.texture_height,
                    .x = region.x,
                    .y = region.y,
                    .width = region.width,
                    .height = region.height,
                });
            }
            return uploadRgbaRegions(upload_regions);
        }

        [[nodiscard]] bool uploadRegion(const std::uint8_t* const pixels,
                                        const int texture_width,
                                        const int texture_height,
                                        const int x,
                                        const int y,
                                        const int region_width,
                                        const int region_height,
                                        const int channels) {
            const VulkanUiTexture::Region region{
                .pixels = pixels,
                .texture_width = texture_width,
                .texture_height = texture_height,
                .x = x,
                .y = y,
                .width = region_width,
                .height = region_height,
                .channels = channels,
            };
            return uploadRegions(std::span<const VulkanUiTexture::Region>(&region, 1));
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
            // Retire current resources; never destroy leftover retired images whose
            // serial is the in-progress unsubmitted GUI frame.
            (void)tryFinishDestroy(true);
            image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_vram_label.clear();
            src_incarnation_ = 0;
            mode = Mode::Uninitialized;
            skip_host_fallback_ = false;
            if (!hasGpuLifetime())
                graphics_sampling_quarantined_ = false;
            width = 0;
            height = 0;
        }

        [[nodiscard]] bool reset(const bool wait) {
            if (device == VK_NULL_HANDLE && !hasGpuLifetime())
                return true;
            if (wait && context != nullptr && !context->waitForSubmittedFrames()) {
                LOG_WARN("Vulkan UI texture shutdown could not wait for submitted frames: {}",
                         context->lastError());
            }
            if (!tryFinishDestroy(wait))
                return false;
            destroyDeviceObjects();
            return true;
        }
    };

    std::vector<VulkanUiTexture::Impl*> VulkanUiTexture::orphaned_impls_;

    std::size_t VulkanUiTexture::serviceOrphanedImpls(const bool wait) {
        if (g_texture_context == nullptr)
            return orphaned_impls_.size();
        auto write = orphaned_impls_.begin();
        for (auto read = orphaned_impls_.begin(); read != orphaned_impls_.end(); ++read) {
            if (*read && (*read)->reset(wait)) {
                delete *read;
                continue;
            }
            if (write != read)
                *write = *read;
            ++write;
        }
        orphaned_impls_.erase(write, orphaned_impls_.end());
        return orphaned_impls_.size();
    }

    void VulkanUiTexture::orphanImpl(Impl* const impl) {
        if (impl)
            orphaned_impls_.push_back(impl);
    }

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
        serviceOrphanedImpls(false);
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
        serviceOrphanedImpls(false);
        if (!impl_) {
            impl_ = new Impl();
        }
        return impl_->uploadRegion(pixels, texture_width, texture_height, x, y, width, height, channels);
    }

    bool VulkanUiTexture::uploadRegions(const std::span<const Region> regions) {
        serviceOrphanedImpls(false);
        if (!impl_) {
            impl_ = new Impl();
        }
        return impl_->uploadRegions(regions);
    }

    bool VulkanUiTexture::upload(const lfs::core::Tensor& image,
                                 const int expected_width,
                                 const int expected_height,
                                 const bool flip_y) {
        serviceOrphanedImpls(false);
        if (!impl_) {
            impl_ = new Impl();
        }
        // Original CUDA surface path before RGBA8 packing. Eligible Float32/UInt8
        // 1/3/4-channel tensors keep toByte nearest (including gray); Float16 is
        // not locked into a packed RGBA8 interop upload. Vulkan sources never
        // enter this path.
        if (impl_->mode != Impl::Mode::Cpu &&
            cudaDirectUploadEligible(image, expected_width, expected_height) &&
            impl_->uploadCudaTensorImpl(image, expected_width, expected_height, flip_y)) {
            return true;
        }

        const lfs::core::Tensor prepared =
            prepareUiRgba8(image, expected_width, expected_height, flip_y);
        if (!prepared.is_valid()) {
            return false;
        }
        const auto backend = lfs::core::gpu_backend_of(prepared);
        if (prepared.device() == lfs::core::Device::GPU && impl_->mode != Impl::Mode::Cpu) {
            if (backend == lfs::core::GpuBackend::CUDA &&
                impl_->uploadCudaTensorImpl(prepared, expected_width, expected_height, false)) {
                return true;
            }
        }
        // Native Vulkan images use the same image allocation mode as host uploads.
        // Keep repeated updates on the device after ensureImage selects that mode.
        if (backend == lfs::core::GpuBackend::Vulkan) {
            if (impl_->uploadVulkanTensorImpl(prepared)) {
                return true;
            }
            if (impl_->skip_host_fallback_) {
                return false;
            }
        }
        const lfs::core::Tensor host = prepared.device() == lfs::core::Device::GPU
                                           ? prepared.cpu().contiguous()
                                           : prepared.contiguous();
        const auto* pixels = host.ptr<std::uint8_t>();
        if (!pixels) {
            return false;
        }
        return impl_->uploadRgba(std::vector<std::uint8_t>(pixels, pixels + host.bytes()),
                                 expected_width,
                                 expected_height);
    }

    std::uintptr_t VulkanUiTexture::textureId() const {
        serviceOrphanedImpls(false);
        if (!impl_ || impl_->graphics_sampling_quarantined_) {
            return 0;
        }
        impl_->tryReleasePendingUpload();
        return reinterpret_cast<std::uintptr_t>(impl_->descriptor_set);
    }

    std::string VulkanUiTexture::rmlSrcUrl(const int width, const int height) const {
        serviceOrphanedImpls(false);
        if (!impl_ || impl_->graphics_sampling_quarantined_) {
            return {};
        }
        impl_->tryReleasePendingUpload();
        return RenderInterface_VK::MakeExternalTextureSource(
            impl_->image_view, impl_->sampler, width, height, impl_->src_incarnation_);
    }

    bool VulkanUiTexture::valid() const {
        serviceOrphanedImpls(false);
        if (!impl_ || impl_->graphics_sampling_quarantined_) {
            return false;
        }
        impl_->tryReleasePendingUpload();
        if (impl_->image_view == VK_NULL_HANDLE) {
            return false;
        }
        return impl_->mode == Impl::Mode::CudaInterop || impl_->descriptor_set != VK_NULL_HANDLE;
    }

    void VulkanUiTexture::reset() {
        serviceOrphanedImpls(false);
        if (!impl_)
            return;
        if (!impl_->reset(true)) {
            orphanImpl(impl_);
            impl_ = nullptr;
        }
    }

} // namespace lfs::vis::gui
