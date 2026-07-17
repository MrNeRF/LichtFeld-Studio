/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_cloud_vulkan_renderer.hpp"

#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "rendering/vulkan_wait.hpp"
#include "window/vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <format>
#include <glm/gtc/type_ptr.hpp>
#include <mutex>
#include <source_location>
#include <stop_token>
#include <utility>
#include <vk_mem_alloc.h>

#include "viewport/point_cloud.frag.spv.h"
#include "viewport/point_cloud.vert.spv.h"

namespace lfs::vis {

    namespace {

        constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
        constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
        constexpr std::size_t kSlotCount = 3;
        constexpr std::size_t kPlaceholderSize = 16;
        constexpr std::uint32_t kBindingModelTransforms = 0;
        constexpr std::uint32_t kBindingTransformIndices = 1;
        constexpr std::uint32_t kBindingVisibility = 2;
        constexpr std::uint32_t kBindingSelectionMask = 3;
        constexpr std::uint32_t kBindingPreviewSelectionMask = 4;
        constexpr std::uint32_t kBindingSelectionColors = 5;
        constexpr std::uint32_t kBindingDeletedMask = 6;
        constexpr std::uint32_t kDescriptorBindingCount = 7;

        // Phase 7C-P2: point-cloud is the ResetPreWaitReplacement row only.
        constexpr lfs::rendering::SubmissionFencePolicy kPointCloudFencePolicy =
            lfs::rendering::SubmissionFencePolicy::ResetPreWaitReplacement;

        [[nodiscard]] const char* waitOutcomeLabel(
            const lfs::rendering::WaitOutcome outcome) noexcept {
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

        // Push constants exactly mirror the layout in point_cloud.vert.
        // Packed to 256 bytes (the Vulkan portable upper bound). Counts/flags
        // are bitpacked into a single ivec4 to save 16 bytes vs. an int-per-flag
        // layout that would push us to 272 bytes.
        enum FlagBits : int {
            kFlagHasCrop = 1 << 0,
            kFlagCropInverse = 1 << 1,
            kFlagCropDesaturate = 1 << 2,
            kFlagOrthographic = 1 << 3,
            kFlagHasIndices = 1 << 4,
            kFlagHasSelection = 1 << 5,
            kFlagHasPreviewSelection = 1 << 6,
            kFlagPreviewSelectionAdditive = 1 << 7,
            kFlagDepthViewGrayscale = 1 << 8,
            kFlagHasDeletedMask = 1 << 9,
        };
        struct PushConstants {
            float view_proj[16];        // 64
            float view[16];             // 64
            float crop_to_local[16];    // 64
            float crop_min[4];          // 16
            float crop_max[4];          // 16
            float voxel_focal_ortho[4]; // 16
            int counts[4]{};            // 16: x=n_transforms, y=n_visibility, z=flags, w=max_point_size
        };
        static_assert(sizeof(PushConstants) == 256,
                      "Push constants must fit Vulkan's portable upper bound");

        struct ManagedBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDeviceSize size = 0;
            VkBufferUsageFlags usage = 0;
            std::string vram_scope;
            std::string vram_label;
        };

        void destroyBuffer(VmaAllocator allocator, ManagedBuffer& buf) {
            if (!buf.vram_scope.empty() && !buf.vram_label.empty()) {
                lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(buf.vram_scope, buf.vram_label, 0);
            }
            if (buf.buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, buf.buffer, buf.allocation);
            }
            buf = {};
        }

        bool createDeviceBuffer(VmaAllocator allocator, VkDeviceSize size,
                                VkBufferUsageFlags usage, ManagedBuffer& out,
                                std::string_view vram_scope = "vulkan.point_cloud.buffer",
                                std::string_view vram_label = {}) {
            destroyBuffer(allocator, out);
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = std::max<VkDeviceSize>(size, kPlaceholderSize);
            bi.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

            VmaAllocationInfo allocation_info{};
            const VkResult r = vmaCreateBuffer(allocator, &bi, &ai,
                                               &out.buffer, &out.allocation, &allocation_info);
            if (r != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vmaCreateBuffer(allocator, &bi, &ai, &out.buffer, &out.allocation, &allocation_info)",
                              r,
                              std::format("Point-cloud device-buffer allocation failed (allocator={:#x}, requested_size={}, allocated_size={}, usage={:#x}, label='{}')",
                                          reinterpret_cast<std::uintptr_t>(allocator),
                                          size,
                                          bi.size,
                                          static_cast<std::uint32_t>(bi.usage),
                                          vram_label),
                              __FILE__,
                              __LINE__));
                out = {};
                return false;
            }
            out.size = bi.size;
            out.usage = bi.usage;
            out.vram_scope = std::string(vram_scope);
            out.vram_label = vram_label.empty()
                                 ? std::format("buffer.{}B", static_cast<std::size_t>(bi.size))
                                 : std::string(vram_label);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                out.vram_scope,
                out.vram_label,
                static_cast<std::size_t>(allocation_info.size));
            return true;
        }

        struct ScopedStagingBuffer {
            VmaAllocator allocator = VK_NULL_HANDLE;
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VmaAllocationInfo allocation_info{};
            std::string vram_scope;
            std::string vram_label;

            ~ScopedStagingBuffer() {
                if (allocator != VK_NULL_HANDLE && buffer != VK_NULL_HANDLE) {
                    if (!vram_scope.empty() && !vram_label.empty()) {
                        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                            vram_scope,
                            vram_label,
                            0);
                    }
                    vmaDestroyBuffer(allocator, buffer, allocation);
                }
            }
        };

        bool stageBufferUpload(VmaAllocator allocator,
                               VkCommandBuffer cb,
                               const void* src,
                               VkDeviceSize bytes,
                               ManagedBuffer& dst,
                               ManagedBuffer& staging_scratch) {
            if (allocator == VK_NULL_HANDLE || cb == VK_NULL_HANDLE || src == nullptr ||
                bytes == 0 || dst.buffer == VK_NULL_HANDLE || bytes > dst.size) {
                return logVkFailure(std::format(
                    "Point-cloud staging upload requires live handles and a copy range within the destination allocation (allocator={:#x}, command_buffer={:#x}, source={:#x}, copy_size={}, destination_buffer={:#x}, destination_size={}) ({}:{})",
                    reinterpret_cast<std::uintptr_t>(allocator),
                    vkHandleValue(cb),
                    reinterpret_cast<std::uintptr_t>(src),
                    bytes,
                    vkHandleValue(dst.buffer),
                    dst.size,
                    __FILE__,
                    __LINE__));
            }
            destroyBuffer(allocator, staging_scratch);
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo info{};
            const VkResult create_result =
                vmaCreateBuffer(allocator,
                                &bi,
                                &ai,
                                &staging_scratch.buffer,
                                &staging_scratch.allocation,
                                &info);
            if (create_result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vmaCreateBuffer(allocator, &bi, &ai, &staging_scratch.buffer, &staging_scratch.allocation, &info)",
                              create_result,
                              std::format("Point-cloud staging-buffer allocation failed (allocator={:#x}, requested_size={}, destination_buffer={:#x}, destination_size={})",
                                          reinterpret_cast<std::uintptr_t>(allocator),
                                          bytes,
                                          vkHandleValue(dst.buffer),
                                          dst.size),
                              __FILE__,
                              __LINE__));
                staging_scratch = {};
                return false;
            }
            staging_scratch.size = bytes;
            staging_scratch.usage = bi.usage;
            if (info.pMappedData == nullptr) {
                LOG_ERROR("Point-cloud staging allocation is not mapped after requesting mapped memory (allocator={:#x}, allocation={:#x}, buffer={:#x}, requested_size={}, mapped={:#x}) ({}:{})",
                          reinterpret_cast<std::uintptr_t>(allocator),
                          reinterpret_cast<std::uintptr_t>(staging_scratch.allocation),
                          vkHandleValue(staging_scratch.buffer),
                          bytes,
                          reinterpret_cast<std::uintptr_t>(info.pMappedData),
                          __FILE__,
                          __LINE__);
                destroyBuffer(allocator, staging_scratch);
                return false;
            }
            std::memcpy(info.pMappedData, src, bytes);
            const VkResult flush_result =
                vmaFlushAllocation(allocator, staging_scratch.allocation, 0, bytes);
            if (flush_result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vmaFlushAllocation(allocator, staging_scratch.allocation, 0, bytes)",
                              flush_result,
                              std::format("Point-cloud staging flush failed (allocator={:#x}, allocation={:#x}, buffer={:#x}, offset=0, flush_size={}, capacity={})",
                                          reinterpret_cast<std::uintptr_t>(allocator),
                                          reinterpret_cast<std::uintptr_t>(staging_scratch.allocation),
                                          vkHandleValue(staging_scratch.buffer),
                                          bytes,
                                          staging_scratch.size),
                              __FILE__,
                              __LINE__));
                destroyBuffer(allocator, staging_scratch);
                return false;
            }

            VkBufferCopy region{};
            region.size = bytes;
            vkCmdCopyBuffer(cb, staging_scratch.buffer, dst.buffer, 1, &region);
            return true;
        }

        void writePushConstants(PushConstants& pc, const PointCloudVulkanRenderer::RenderRequest& req,
                                int max_point_size, int n_transforms, int n_visibility,
                                bool has_transform_indices, bool has_selection,
                                bool has_preview_selection, bool has_deleted_mask) {
            std::memcpy(pc.view_proj, glm::value_ptr(req.view_projection), sizeof(pc.view_proj));
            std::memcpy(pc.view, glm::value_ptr(req.view), sizeof(pc.view));

            const glm::mat4 crop_to_local = req.crop ? req.crop->to_local : glm::mat4(1.0f);
            std::memcpy(pc.crop_to_local, glm::value_ptr(crop_to_local), sizeof(pc.crop_to_local));

            const glm::vec3 crop_min = req.crop ? req.crop->min : glm::vec3(0.0f);
            const glm::vec3 crop_max = req.crop ? req.crop->max : glm::vec3(0.0f);
            pc.crop_min[0] = crop_min.x;
            pc.crop_min[1] = crop_min.y;
            pc.crop_min[2] = crop_min.z;
            pc.crop_min[3] = req.depth_view_min;
            pc.crop_max[0] = crop_max.x;
            pc.crop_max[1] = crop_max.y;
            pc.crop_max[2] = crop_max.z;
            pc.crop_max[3] = req.depth_view_max;

            const float voxel = req.voxel_size * req.scaling_modifier;
            const float ortho_pixels_per_world = req.ortho_scale > 1e-5f
                                                     ? static_cast<float>(req.size.y) / req.ortho_scale
                                                     : static_cast<float>(req.size.y);
            pc.voxel_focal_ortho[0] = voxel;
            pc.voxel_focal_ortho[1] = req.focal_y;
            pc.voxel_focal_ortho[2] = ortho_pixels_per_world;
            pc.voxel_focal_ortho[3] = req.depth_view ? 1.0f : 0.0f;

            int flags = 0;
            if (req.crop) {
                flags |= kFlagHasCrop;
                if (req.crop->inverse)
                    flags |= kFlagCropInverse;
                if (req.crop->desaturate)
                    flags |= kFlagCropDesaturate;
            }
            if (req.orthographic) {
                flags |= kFlagOrthographic;
            }
            if (has_transform_indices) {
                flags |= kFlagHasIndices;
            }
            if (has_selection) {
                flags |= kFlagHasSelection;
            }
            if (has_preview_selection) {
                flags |= kFlagHasPreviewSelection;
                if (req.preview_selection_additive) {
                    flags |= kFlagPreviewSelectionAdditive;
                }
            }
            if (has_deleted_mask) {
                flags |= kFlagHasDeletedMask;
            }
            if (req.depth_visualization_mode == lfs::rendering::DepthVisualizationMode::Grayscale) {
                flags |= kFlagDepthViewGrayscale;
            }
            pc.counts[0] = n_transforms;
            pc.counts[1] = n_visibility;
            pc.counts[2] = flags;
            pc.counts[3] = max_point_size;
        }

        // Copy a [N, 3] float Tensor (CPU or CUDA) to a contiguous host buffer.
        bool tensorToHost(const lfs::core::Tensor& t, std::vector<float>& out) {
            if (!t.is_valid() || t.ndim() != 2 || t.size(1) != 3) {
                return false;
            }
            const auto host = t.to(lfs::core::Device::CPU).contiguous();
            const std::size_t n = static_cast<std::size_t>(host.size(0)) * 3;
            out.resize(n);
            std::memcpy(out.data(), host.ptr<float>(), n * sizeof(float));
            return true;
        }

        [[nodiscard]] bool hasPointMask(const lfs::core::Tensor* const mask,
                                        const std::size_t n_points) {
            return mask != nullptr && mask->is_valid() && mask->numel() == n_points;
        }

        [[nodiscard]] std::vector<std::uint32_t> maskTensorToUint32Host(
            const lfs::core::Tensor& mask) {
            const auto host = mask.to(lfs::core::DataType::UInt8)
                                  .to(lfs::core::Device::CPU)
                                  .contiguous();
            const auto* const src = host.ptr<std::uint8_t>();
            std::vector<std::uint32_t> out(static_cast<std::size_t>(host.numel()));
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = static_cast<std::uint32_t>(src[i]);
            }
            return out;
        }

    } // namespace

    struct PointCloudVulkanRenderer::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;

        // Pipeline
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        // Descriptor pool / set (transforms + indices + visibility + overlays)
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

        // Cached input buffers — point clouds rarely change, so we re-upload only
        // when the cache key (tensor pointer / size) changes.
        struct InputCache {
            ManagedBuffer positions;
            ManagedBuffer colors;
            ManagedBuffer transforms;
            ManagedBuffer transform_indices;
            ManagedBuffer visibility;
            ManagedBuffer deleted_mask;
            ManagedBuffer selection_mask;
            ManagedBuffer preview_selection_mask;
            ManagedBuffer selection_colors;

            const void* cached_positions_ptr = nullptr;
            std::size_t cached_positions_count = 0;
            std::uint64_t cached_positions_revision = 0;
            const void* cached_colors_ptr = nullptr;
            std::size_t cached_colors_count = 0;
            lfs::core::DataType cached_colors_dtype = lfs::core::DataType::Float32;
            std::uint64_t cached_colors_revision = 0;
            std::size_t cached_n_transforms = 0;
            std::size_t cached_n_visibility = 0;
            const void* cached_transform_indices_ptr = nullptr;
            std::size_t cached_transform_indices_count = 0;
            std::vector<glm::mat4> cached_transforms;
            std::vector<std::uint32_t> cached_visibility;

            const void* cached_deleted_mask_ptr = nullptr;
            std::size_t cached_deleted_mask_id = 0;
            std::size_t cached_deleted_mask_count = 0;
            std::uint64_t cached_deleted_mask_revision = 0;
            const void* cached_selection_mask_ptr = nullptr;
            std::size_t cached_selection_mask_id = 0;
            std::size_t cached_selection_mask_count = 0;
            std::uint64_t cached_selection_revision = 0;
            const void* cached_preview_selection_mask_ptr = nullptr;
            std::size_t cached_preview_selection_mask_id = 0;
            std::size_t cached_preview_selection_mask_count = 0;
            std::uint64_t cached_preview_selection_revision = 0;
            std::array<glm::vec4, lfs::rendering::kSelectionColorTableCount> cached_selection_colors{};
            bool cached_selection_colors_valid = false;
        };
        InputCache cache;

        // Placeholder bound to descriptors when an SSBO is unused — avoids
        // descriptor-indexing churn while keeping the bindings stable.
        ManagedBuffer placeholder;

        // One-time staging buffers, kept around so we don't reallocate per upload.
        std::vector<ManagedBuffer> pending_stagings;

        // Output slots: each owns its own color + depth attachment.
        struct OutputSlotResources {
            VkImage color_image = VK_NULL_HANDLE;
            VmaAllocation color_alloc = VK_NULL_HANDLE;
            VkImageView color_view = VK_NULL_HANDLE;

            VkImage depth_image = VK_NULL_HANDLE;
            VmaAllocation depth_alloc = VK_NULL_HANDLE;
            VkImageView depth_view = VK_NULL_HANDLE;
            std::string color_vram_label;
            std::string depth_vram_label;

            glm::ivec2 size{0, 0};
            VkImageLayout color_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t generation = 0;
        };
        std::array<OutputSlotResources, kSlotCount> slots{};

        // Transient command pool / fence reused across frames.
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        std::mutex command_mutex;

        // Phase 7C-P2: closed SubmissionState table for reset-before-submit.
        lfs::rendering::SubmissionState submission_state_{};
        std::atomic<bool> gpu_wait_quarantined_{false};

        bool initialized = false;

        ~Impl() { destroy(); }

        [[nodiscard]] lfs::rendering::WaitContext makeWaitContext(
            const std::string_view fingerprint) noexcept {
            lfs::rendering::WaitContext wait_ctx;
            wait_ctx.fingerprint = fingerprint;
            wait_ctx.owner_quarantine_flag = &gpu_wait_quarantined_;
            return wait_ctx;
        }

        [[nodiscard]] std::string formatFenceWaitUnexpected(
            const std::string_view what,
            const lfs::Result<lfs::rendering::WaitOutcome>& outcome) const {
            return std::format(
                "Point-cloud {} did not reach Ready (device={:#x}, fence={:#x}): {}",
                what,
                vkHandleValue(device),
                vkHandleValue(fence),
                formatWaitFailure(outcome));
        }

        // T3 helper: only replace when the transition table authorizes it
        // (ResetPreWaitReplacement + fence_reset). Ruling-1 reset failures
        // call replaceFenceSignaled directly and never enter here.
        void rejectSubmissionAndMaybeReplace(const char* const failed_operation,
                                             const VkResult failed_result) {
            using lfs::rendering::SubmissionTransition;
            using lfs::rendering::apply_submission_transition;
            auto effects = apply_submission_transition(
                submission_state_,
                SubmissionTransition::SubmitRejected,
                kPointCloudFencePolicy);
            if (!effects) {
                LOG_ERROR(
                    "Point-cloud SubmissionState T3 SubmitRejected illegal "
                    "(detail={}, op={}, result={})",
                    effects.error().detail(),
                    failed_operation,
                    static_cast<int>(failed_result));
                return;
            }
            if (effects->replace_fence_signaled) {
                (void)replaceFenceSignaled(failed_operation, failed_result);
            }
        }

        [[nodiscard]] std::string vkError(
            const std::string_view what,
            const VkResult r,
            const std::string_view details = {},
            const std::source_location location = std::source_location::current()) const {
            const std::string observed = details.empty()
                                             ? std::format(
                                                   "Point-cloud Vulkan operation failed (device={:#x}, queue={:#x}, command_pool={:#x}, command_buffer={:#x}, fence={:#x})",
                                                   vkHandleValue(device),
                                                   context != nullptr ? vkHandleValue(context->graphicsQueue()) : 0,
                                                   vkHandleValue(command_pool),
                                                   vkHandleValue(command_buffer),
                                                   vkHandleValue(fence))
                                             : std::string(details);
            return formatVkCheckFailure(
                what, r, observed, location.file_name(), static_cast<int>(location.line()));
        }

        bool replaceFenceSignaled(const char* const failed_operation,
                                  const VkResult failed_result) {
            LOG_ERROR("Vulkan: {}",
                      vkError(failed_operation,
                              failed_result,
                              std::format("Point-cloud command lifecycle failed (device={:#x}, queue={:#x}, command_pool={:#x}, command_buffer={:#x}, fence={:#x})",
                                          vkHandleValue(device),
                                          context != nullptr ? vkHandleValue(context->graphicsQueue()) : 0,
                                          vkHandleValue(command_pool),
                                          vkHandleValue(command_buffer),
                                          vkHandleValue(fence))));
            VkFenceCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VkFence replacement = VK_NULL_HANDLE;
            const VkResult replacement_result = vkCreateFence(device, &info, nullptr, &replacement);
            if (replacement_result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          vkError("vkCreateFence(device, &info, nullptr, &replacement)",
                                  replacement_result,
                                  std::format("Point-cloud renderer failed to replace a poisoned fence (device={:#x}, old_fence={:#x}, command_buffer={:#x}, flags={:#x})",
                                              vkHandleValue(device),
                                              vkHandleValue(fence),
                                              vkHandleValue(command_buffer),
                                              static_cast<std::uint32_t>(info.flags))));
                return false;
            }
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
            }
            fence = replacement;
            context->setDebugObjectName(VK_OBJECT_TYPE_FENCE,
                                        fence,
                                        "point_cloud.render.fence");
            return true;
        }

        // Allocate a fresh device-local buffer and stage `bytes` from `src` into
        // it. The transient host staging buffer is parked in pending_stagings
        // and released after the next fence-wait.
        std::expected<void, std::string> uploadInto(VkCommandBuffer cb,
                                                    VkBufferUsageFlags usage,
                                                    const void* src,
                                                    VkDeviceSize bytes,
                                                    ManagedBuffer& dst,
                                                    const char* what) {
            if (!createDeviceBuffer(allocator,
                                    bytes,
                                    usage,
                                    dst,
                                    "vulkan.point_cloud.buffer",
                                    what)) {
                return std::unexpected<std::string>(std::format(
                    "Point-cloud input buffer allocation failed (label='{}', requested_size={}, usage={:#x}, allocator={:#x}) ({}:{})",
                    what,
                    bytes,
                    static_cast<std::uint32_t>(usage),
                    reinterpret_cast<std::uintptr_t>(allocator),
                    __FILE__,
                    __LINE__));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         dst.buffer,
                                         "point_cloud.input.{}[{}]",
                                         what,
                                         dst.size);
            ManagedBuffer staging{};
            if (!stageBufferUpload(allocator, cb, src, bytes, dst, staging)) {
                destroyBuffer(allocator, staging);
                return std::unexpected<std::string>(std::format(
                    "Point-cloud input staging upload failed (label='{}', command_buffer={:#x}, source={:#x}, copy_size={}, destination_buffer={:#x}, destination_size={}) ({}:{})",
                    what,
                    vkHandleValue(cb),
                    reinterpret_cast<std::uintptr_t>(src),
                    bytes,
                    vkHandleValue(dst.buffer),
                    dst.size,
                    __FILE__,
                    __LINE__));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         staging.buffer,
                                         "point_cloud.input.{}.upload.staging[{}]",
                                         what,
                                         staging.size);
            pending_stagings.push_back(std::move(staging));
            return {};
        }

        std::expected<void, std::string> ensureInitialized(VulkanContext& ctx) {
            if (initialized) {
                return {};
            }
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE) {
                return std::unexpected<std::string>(std::format(
                    "Point-cloud renderer requires a live Vulkan device and allocator (device={:#x}, allocator={:#x}) ({}:{})",
                    vkHandleValue(device),
                    reinterpret_cast<std::uintptr_t>(allocator),
                    __FILE__,
                    __LINE__));
            }

            if (auto r = createPipeline(); !r) {
                return r;
            }
            if (auto r = createDescriptorPool(); !r) {
                return r;
            }
            if (auto r = createCommandResources(); !r) {
                return r;
            }
            if (!createDeviceBuffer(allocator,
                                    kPlaceholderSize,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    placeholder,
                                    "vulkan.point_cloud.buffer",
                                    "placeholder")) {
                return std::unexpected<std::string>(std::format(
                    "Point-cloud placeholder buffer allocation failed (allocator={:#x}, requested_size={}, usage={:#x}) ({}:{})",
                    reinterpret_cast<std::uintptr_t>(allocator),
                    kPlaceholderSize,
                    static_cast<std::uint32_t>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                    __FILE__,
                    __LINE__));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_BUFFER,
                                        placeholder.buffer,
                                        "point_cloud.input.placeholder");
            initialized = true;
            return {};
        }

        std::expected<void, std::string> createPipeline() {
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kPointCloudVertSpv, "Point-cloud");
            VkShaderModule frag = createShaderModule(device, kPointCloudFragSpv, "Point-cloud");
            if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
                if (vert)
                    vkDestroyShaderModule(device, vert, nullptr);
                if (frag)
                    vkDestroyShaderModule(device, frag, nullptr);
                return std::unexpected<std::string>("vkCreateShaderModule(point_cloud) failed");
            }

            std::array<VkDescriptorSetLayoutBinding, kDescriptorBindingCount> bindings{};
            for (std::uint32_t i = 0; i < bindings.size(); ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            }
            VkDescriptorSetLayoutCreateInfo dl{};
            dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dl.bindingCount = static_cast<std::uint32_t>(bindings.size());
            dl.pBindings = bindings.data();
            VkResult r = vkCreateDescriptorSetLayout(device, &dl, nullptr, &descriptor_set_layout);
            if (r != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return std::unexpected<std::string>(vkError(
                    "vkCreateDescriptorSetLayout(device, &dl, nullptr, &descriptor_set_layout)",
                    r,
                    std::format("Point-cloud descriptor-set layout creation failed (device={:#x}, binding_count={}, descriptor_type={})",
                                vkHandleValue(device),
                                dl.bindingCount,
                                static_cast<int>(bindings[0].descriptorType))));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        descriptor_set_layout,
                                        "point_cloud.descriptor.layout");

            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            push.size = sizeof(PushConstants);

            VkPipelineLayoutCreateInfo pli{};
            pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pli.setLayoutCount = 1;
            pli.pSetLayouts = &descriptor_set_layout;
            pli.pushConstantRangeCount = 1;
            pli.pPushConstantRanges = &push;
            r = vkCreatePipelineLayout(device, &pli, nullptr, &pipeline_layout);
            if (r != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return std::unexpected<std::string>(vkError(
                    "vkCreatePipelineLayout(device, &pli, nullptr, &pipeline_layout)",
                    r,
                    std::format("Point-cloud pipeline-layout creation failed (device={:#x}, descriptor_layout={:#x}, set_layout_count={}, push_constant_bytes={})",
                                vkHandleValue(device),
                                vkHandleValue(descriptor_set_layout),
                                pli.setLayoutCount,
                                push.size)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                        pipeline_layout,
                                        "point_cloud.pipeline.layout");

            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName = "main";

            std::array<VkVertexInputBindingDescription, 2> input_bindings{};
            input_bindings[0].binding = 0;
            input_bindings[0].stride = sizeof(float) * 3;
            input_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            input_bindings[1].binding = 1;
            input_bindings[1].stride = sizeof(float) * 3;
            input_bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            std::array<VkVertexInputAttributeDescription, 2> attrs{};
            attrs[0].location = 0;
            attrs[0].binding = 0;
            attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            attrs[0].offset = 0;
            attrs[1].location = 1;
            attrs[1].binding = 1;
            attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attrs[1].offset = 0;

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount = static_cast<std::uint32_t>(input_bindings.size());
            vi.pVertexBindingDescriptions = input_bindings.data();
            vi.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attrs.size());
            vi.pVertexAttributeDescriptions = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

            VkPipelineViewportStateCreateInfo vps{};
            vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vps.viewportCount = 1;
            vps.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp = VK_COMPARE_OP_LESS;

            VkPipelineColorBlendAttachmentState ba{};
            ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo bs{};
            bs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            bs.attachmentCount = 1;
            bs.pAttachments = &ba;

            std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dyn.size());
            dynamic.pDynamicStates = dyn.data();

            VkFormat color_format = kColorFormat;
            VkPipelineRenderingCreateInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &color_format;
            rendering.depthAttachmentFormat = kDepthFormat;

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.pNext = &rendering;
            pi.stageCount = static_cast<std::uint32_t>(stages.size());
            pi.pStages = stages.data();
            pi.pVertexInputState = &vi;
            pi.pInputAssemblyState = &ia;
            pi.pViewportState = &vps;
            pi.pRasterizationState = &rs;
            pi.pMultisampleState = &ms;
            pi.pDepthStencilState = &ds;
            pi.pColorBlendState = &bs;
            pi.pDynamicState = &dynamic;
            pi.layout = pipeline_layout;

            r = vkCreateGraphicsPipelines(device, context->pipelineCache(), 1, &pi,
                                          nullptr, &pipeline);
            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError(
                    "vkCreateGraphicsPipelines(device, context->pipelineCache(), 1, &pi, nullptr, &pipeline)",
                    r,
                    std::format("Point-cloud graphics-pipeline creation failed (device={:#x}, pipeline_cache={:#x}, pipeline_layout={:#x}, color_format={}, depth_format={})",
                                vkHandleValue(device),
                                vkHandleValue(context->pipelineCache()),
                                vkHandleValue(pipeline_layout),
                                static_cast<int>(color_format),
                                static_cast<int>(kDepthFormat))));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                                        pipeline,
                                        "point_cloud.pipeline");
            return {};
        }

        std::expected<void, std::string> createDescriptorPool() {
            VkDescriptorPoolSize ps{};
            ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ps.descriptorCount = kDescriptorBindingCount;
            VkDescriptorPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pi.maxSets = 1;
            pi.poolSizeCount = 1;
            pi.pPoolSizes = &ps;
            VkResult r = vkCreateDescriptorPool(device, &pi, nullptr, &descriptor_pool);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError(
                    "vkCreateDescriptorPool(device, &pi, nullptr, &descriptor_pool)",
                    r,
                    std::format("Point-cloud descriptor-pool creation failed (device={:#x}, max_sets={}, pool_size_count={}, descriptor_count={})",
                                vkHandleValue(device),
                                pi.maxSets,
                                pi.poolSizeCount,
                                ps.descriptorCount)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                        descriptor_pool,
                                        "point_cloud.descriptor.pool");
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptor_pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &descriptor_set_layout;
            r = vkAllocateDescriptorSets(device, &ai, &descriptor_set);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError(
                    "vkAllocateDescriptorSets(device, &ai, &descriptor_set)",
                    r,
                    std::format("Point-cloud descriptor-set allocation failed (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
                                vkHandleValue(device),
                                vkHandleValue(descriptor_pool),
                                vkHandleValue(descriptor_set_layout),
                                ai.descriptorSetCount)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                        descriptor_set,
                                        "point_cloud.descriptor");
            return {};
        }

        std::expected<void, std::string> createCommandResources() {
            VkCommandPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pi.queueFamilyIndex = context->graphicsQueueFamily();
            VkResult r = vkCreateCommandPool(device, &pi, nullptr, &command_pool);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError(
                    "vkCreateCommandPool(device, &pi, nullptr, &command_pool)",
                    r,
                    std::format("Point-cloud command-pool creation failed (device={:#x}, queue_family={}, flags={:#x})",
                                vkHandleValue(device),
                                pi.queueFamilyIndex,
                                static_cast<std::uint32_t>(pi.flags))));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                                        command_pool,
                                        "point_cloud.render.pool");
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = command_pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            r = vkAllocateCommandBuffers(device, &ai, &command_buffer);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError(
                    "vkAllocateCommandBuffers(device, &ai, &command_buffer)",
                    r,
                    std::format("Point-cloud command-buffer allocation failed (device={:#x}, command_pool={:#x}, requested_count={})",
                                vkHandleValue(device),
                                vkHandleValue(command_pool),
                                ai.commandBufferCount)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER,
                                        command_buffer,
                                        "point_cloud.render.command");
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            r = vkCreateFence(device, &fi, nullptr, &fence);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError(
                    "vkCreateFence(device, &fi, nullptr, &fence)",
                    r,
                    std::format("Point-cloud render fence creation failed (device={:#x}, command_buffer={:#x}, flags={:#x})",
                                vkHandleValue(device),
                                vkHandleValue(command_buffer),
                                static_cast<std::uint32_t>(fi.flags))));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_FENCE,
                                        fence,
                                        "point_cloud.render.fence");
            return {};
        }

        std::expected<void, std::string> ensureOutputImages(OutputSlotResources& slot,
                                                            glm::ivec2 size) {
            const std::size_t slot_index = static_cast<std::size_t>(&slot - slots.data());
            if (slot_index >= slots.size() || size.x <= 0 || size.y <= 0) {
                return std::unexpected<std::string>(std::format(
                    "Point-cloud output image request requires an in-range slot and positive dimensions (slot_index={}, slot_count={}, observed_width={}, observed_height={}) ({}:{})",
                    slot_index,
                    slots.size(),
                    size.x,
                    size.y,
                    __FILE__,
                    __LINE__));
            }
            if (slot.color_image != VK_NULL_HANDLE && slot.depth_image != VK_NULL_HANDLE &&
                slot.size == size) {
                return {};
            }
            destroySlot(slot);

            const VkExtent3D extent{static_cast<std::uint32_t>(size.x),
                                    static_cast<std::uint32_t>(size.y), 1u};

            // Color: R8G8B8A8_UNORM, used as render target + sampled by scene blit.
            VkImageCreateInfo color_info{};
            color_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            color_info.imageType = VK_IMAGE_TYPE_2D;
            color_info.format = kColorFormat;
            color_info.extent = extent;
            color_info.mipLevels = 1;
            color_info.arrayLayers = 1;
            color_info.samples = VK_SAMPLE_COUNT_1_BIT;
            color_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            color_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            color_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            VmaAllocationInfo color_allocation_info{};
            VkResult r = vmaCreateImage(allocator, &color_info, &ai, &slot.color_image,
                                        &slot.color_alloc, &color_allocation_info);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError(
                    "vmaCreateImage(allocator, &color_info, &ai, &slot.color_image, &slot.color_alloc, &color_allocation_info)",
                    r,
                    std::format("Point-cloud output color-image allocation failed (allocator={:#x}, slot_index={}, requested_extent={}x{}, format={}, usage={:#x})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                slot_index,
                                size.x,
                                size.y,
                                static_cast<int>(color_info.format),
                                static_cast<std::uint32_t>(color_info.usage))));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                         slot.color_image,
                                         "point_cloud.output[{}].color",
                                         slot_index);
            slot.color_vram_label = std::format("color.slot{}:{}x{}", slot_index, size.x, size.y);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.point_cloud.output_image",
                slot.color_vram_label,
                static_cast<std::size_t>(color_allocation_info.size));
            VkImageViewCreateInfo cv{};
            cv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            cv.image = slot.color_image;
            cv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            cv.format = kColorFormat;
            cv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            r = vkCreateImageView(device, &cv, nullptr, &slot.color_view);
            if (r != VK_SUCCESS) {
                const std::string error = vkError(
                    "vkCreateImageView(device, &cv, nullptr, &slot.color_view)",
                    r,
                    std::format("Point-cloud output color image-view creation failed (device={:#x}, slot_index={}, image={:#x}, extent={}x{}, format={}, aspect_mask={:#x})",
                                vkHandleValue(device),
                                slot_index,
                                vkHandleValue(cv.image),
                                size.x,
                                size.y,
                                static_cast<int>(cv.format),
                                static_cast<std::uint32_t>(cv.subresourceRange.aspectMask)));
                destroySlot(slot);
                return std::unexpected<std::string>(error);
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                         slot.color_view,
                                         "point_cloud.output[{}].color.view",
                                         slot_index);

            // Depth: D32_SFLOAT, used as depth attachment + sampled by depth-blit.
            VkImageCreateInfo depth_info = color_info;
            depth_info.format = kDepthFormat;
            depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT;
            VmaAllocationInfo depth_allocation_info{};
            r = vmaCreateImage(allocator, &depth_info, &ai, &slot.depth_image,
                               &slot.depth_alloc, &depth_allocation_info);
            if (r != VK_SUCCESS) {
                const std::string error = vkError(
                    "vmaCreateImage(allocator, &depth_info, &ai, &slot.depth_image, &slot.depth_alloc, &depth_allocation_info)",
                    r,
                    std::format("Point-cloud output depth-image allocation failed (allocator={:#x}, slot_index={}, requested_extent={}x{}, format={}, usage={:#x})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                slot_index,
                                size.x,
                                size.y,
                                static_cast<int>(depth_info.format),
                                static_cast<std::uint32_t>(depth_info.usage)));
                destroySlot(slot);
                return std::unexpected<std::string>(error);
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                         slot.depth_image,
                                         "point_cloud.output[{}].depth",
                                         slot_index);
            slot.depth_vram_label = std::format("depth.slot{}:{}x{}", slot_index, size.x, size.y);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.point_cloud.output_image",
                slot.depth_vram_label,
                static_cast<std::size_t>(depth_allocation_info.size));
            VkImageViewCreateInfo dv{};
            dv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            dv.image = slot.depth_image;
            dv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            dv.format = kDepthFormat;
            dv.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            r = vkCreateImageView(device, &dv, nullptr, &slot.depth_view);
            if (r != VK_SUCCESS) {
                const std::string error = vkError(
                    "vkCreateImageView(device, &dv, nullptr, &slot.depth_view)",
                    r,
                    std::format("Point-cloud output depth image-view creation failed (device={:#x}, slot_index={}, image={:#x}, extent={}x{}, format={}, aspect_mask={:#x})",
                                vkHandleValue(device),
                                slot_index,
                                vkHandleValue(dv.image),
                                size.x,
                                size.y,
                                static_cast<int>(dv.format),
                                static_cast<std::uint32_t>(dv.subresourceRange.aspectMask)));
                destroySlot(slot);
                return std::unexpected<std::string>(error);
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                         slot.depth_view,
                                         "point_cloud.output[{}].depth.view",
                                         slot_index);

            context->imageBarriers().registerImage(slot.color_image,
                                                   VK_IMAGE_ASPECT_COLOR_BIT,
                                                   VK_IMAGE_LAYOUT_UNDEFINED);
            context->imageBarriers().registerImage(slot.depth_image,
                                                   VK_IMAGE_ASPECT_DEPTH_BIT,
                                                   VK_IMAGE_LAYOUT_UNDEFINED);

            slot.size = size;
            slot.color_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            slot.depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ++slot.generation;
            return {};
        }

        void destroySlot(OutputSlotResources& slot) {
            if (slot.color_image != VK_NULL_HANDLE) {
                if (!slot.color_vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.point_cloud.output_image",
                        slot.color_vram_label,
                        0);
                }
                context->imageBarriers().forgetImage(slot.color_image);
                if (slot.color_view != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, slot.color_view, nullptr);
                }
                vmaDestroyImage(allocator, slot.color_image, slot.color_alloc);
            }
            if (slot.depth_image != VK_NULL_HANDLE) {
                if (!slot.depth_vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.point_cloud.output_image",
                        slot.depth_vram_label,
                        0);
                }
                context->imageBarriers().forgetImage(slot.depth_image);
                if (slot.depth_view != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, slot.depth_view, nullptr);
                }
                vmaDestroyImage(allocator, slot.depth_image, slot.depth_alloc);
            }
            slot = {};
        }

        void destroy() {
            if (device == VK_NULL_HANDLE) {
                return;
            }
            // C3: bounded teardown drain. Non-Ready retains fence + command pool
            // (AMB-4 / AMB-C5 — quarantine never authorizes free of in-flight work).
            bool command_resources_ready = true;
            if (fence != VK_NULL_HANDLE) {
                auto wait_ctx = makeWaitContext("point_cloud.destroy.wait_fence");
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                const bool ready = wait_outcome.has_value() &&
                                   *wait_outcome == lfs::rendering::WaitOutcome::Ready;
                if (ready) {
                    vkDestroyFence(device, fence, nullptr);
                    fence = VK_NULL_HANDLE;
                } else {
                    command_resources_ready = false;
                    LOG_ERROR(
                        "Vulkan: Point-cloud renderer destruction fence did not retire "
                        "(device={:#x}, fence={:#x}, command_pool={:#x}, command_buffer={:#x}): "
                        "{} — retaining fence and command pool until process exit",
                        vkHandleValue(device),
                        vkHandleValue(fence),
                        vkHandleValue(command_pool),
                        vkHandleValue(command_buffer),
                        formatWaitFailure(wait_outcome));
                }
            }
            if (command_resources_ready && command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, command_pool, nullptr);
                command_pool = VK_NULL_HANDLE;
                command_buffer = VK_NULL_HANDLE;
            }
            for (auto& s : slots) {
                destroySlot(s);
            }
            destroyBuffer(allocator, cache.positions);
            destroyBuffer(allocator, cache.colors);
            destroyBuffer(allocator, cache.transforms);
            destroyBuffer(allocator, cache.transform_indices);
            destroyBuffer(allocator, cache.visibility);
            destroyBuffer(allocator, cache.deleted_mask);
            destroyBuffer(allocator, cache.selection_mask);
            destroyBuffer(allocator, cache.preview_selection_mask);
            destroyBuffer(allocator, cache.selection_colors);
            destroyBuffer(allocator, placeholder);
            for (auto& s : pending_stagings) {
                destroyBuffer(allocator, s);
            }
            pending_stagings.clear();
            if (descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
                descriptor_pool = VK_NULL_HANDLE;
                descriptor_set = VK_NULL_HANDLE;
            }
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            }
            if (pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
                pipeline_layout = VK_NULL_HANDLE;
            }
            if (descriptor_set_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
                descriptor_set_layout = VK_NULL_HANDLE;
            }
            initialized = false;
            context = nullptr;
            device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
        }

        std::expected<void, std::string> uploadIfChanged(VkCommandBuffer cb,
                                                         const RenderRequest& req) {
            if (!req.positions || !req.colors) {
                return std::unexpected<std::string>("RenderRequest is missing positions/colors");
            }
            if (!req.positions->is_valid() || req.positions->ndim() != 2 ||
                req.positions->size(1) != 3) {
                return std::unexpected<std::string>("positions must be [N, 3] float");
            }
            if (!req.colors->is_valid() || req.colors->ndim() != 2 ||
                req.colors->size(1) != 3 ||
                req.colors->size(0) != req.positions->size(0)) {
                return std::unexpected<std::string>("colors must match positions [N, 3]");
            }

            const std::size_t n_points = static_cast<std::size_t>(req.positions->size(0));
            if (req.deleted_mask && req.deleted_mask->is_valid() &&
                req.deleted_mask->numel() != n_points) {
                return std::unexpected<std::string>("deleted mask must match positions");
            }

            // positions
            const void* pos_key = req.positions->ptr<float>();
            if (pos_key != cache.cached_positions_ptr || cache.cached_positions_count != n_points ||
                cache.cached_positions_revision != req.positions_revision ||
                cache.positions.buffer == VK_NULL_HANDLE) {
                std::vector<float> host;
                if (!tensorToHost(*req.positions, host)) {
                    return std::unexpected<std::string>("Failed to read positions to CPU");
                }
                if (auto r = uploadInto(cb, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                        host.data(), host.size() * sizeof(float),
                                        cache.positions, "positions");
                    !r) {
                    return r;
                }
                cache.cached_positions_ptr = pos_key;
                cache.cached_positions_count = n_points;
                cache.cached_positions_revision = req.positions_revision;
            }

            // colors (handle uint8 / float alike via Tensor::to). Key on the *source*
            // tensor like positions above — keying on the converted temporary's pointer
            // compares a fresh per-frame allocation each time, which both dangles and
            // defeats the cache (forcing a full re-upload every frame for uint8 colors).
            const void* col_key = req.colors->data_ptr();
            const lfs::core::DataType col_dtype = req.colors->dtype();
            if (col_key != cache.cached_colors_ptr || cache.cached_colors_count != n_points ||
                cache.cached_colors_revision != req.colors_revision ||
                cache.cached_colors_dtype != col_dtype || cache.colors.buffer == VK_NULL_HANDLE) {
                const lfs::core::Tensor colors_f32 =
                    col_dtype == lfs::core::DataType::Float32
                        ? *req.colors
                        : (col_dtype == lfs::core::DataType::UInt8
                               ? req.colors->to(lfs::core::DataType::Float32) / 255.0f
                               : req.colors->to(lfs::core::DataType::Float32));
                std::vector<float> host;
                if (!tensorToHost(colors_f32, host)) {
                    return std::unexpected<std::string>("Failed to read colors to CPU");
                }
                if (auto r = uploadInto(cb, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                        host.data(), host.size() * sizeof(float),
                                        cache.colors, "colors");
                    !r) {
                    return r;
                }
                cache.cached_colors_ptr = col_key;
                cache.cached_colors_count = n_points;
                cache.cached_colors_dtype = col_dtype;
                cache.cached_colors_revision = req.colors_revision;
            }

            // model_transforms (CPU vector of mat4)
            const std::vector<glm::mat4> empty_transforms;
            const auto& transforms = req.model_transforms ? *req.model_transforms : empty_transforms;
            const bool transforms_changed =
                cache.cached_n_transforms != transforms.size() ||
                cache.cached_transforms != transforms;
            if (transforms_changed || cache.transforms.buffer == VK_NULL_HANDLE) {
                if (!transforms.empty()) {
                    if (auto r = uploadInto(cb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            transforms.data(), transforms.size() * sizeof(glm::mat4),
                                            cache.transforms, "transforms");
                        !r) {
                        return r;
                    }
                } else if (cache.transforms.buffer == VK_NULL_HANDLE) {
                    if (!createDeviceBuffer(allocator,
                                            kPlaceholderSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            cache.transforms,
                                            "vulkan.point_cloud.buffer",
                                            "transforms.placeholder")) {
                        return std::unexpected<std::string>("Failed to allocate transforms placeholder");
                    }
                }
                cache.cached_transforms = transforms;
                cache.cached_n_transforms = transforms.size();
            }

            // transform_indices (Tensor)
            const std::size_t expected_indices_count =
                (cache.cached_n_transforms > 0 && req.transform_indices &&
                 req.transform_indices->is_valid() &&
                 req.transform_indices->numel() == n_points)
                    ? n_points
                    : 0;
            if (expected_indices_count > 0) {
                const lfs::core::Tensor indices_i32 =
                    (req.transform_indices->dtype() == lfs::core::DataType::Int32)
                        ? *req.transform_indices
                        : req.transform_indices->to(lfs::core::DataType::Int32);
                const void* idx_key = indices_i32.ptr<int>();
                if (idx_key != cache.cached_transform_indices_ptr ||
                    cache.cached_transform_indices_count != n_points ||
                    cache.transform_indices.buffer == VK_NULL_HANDLE) {
                    const auto host = indices_i32.to(lfs::core::Device::CPU).contiguous();
                    if (auto r = uploadInto(cb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            host.ptr<int>(), n_points * sizeof(std::int32_t),
                                            cache.transform_indices, "transform indices");
                        !r) {
                        return r;
                    }
                    cache.cached_transform_indices_ptr = idx_key;
                    cache.cached_transform_indices_count = n_points;
                }
            } else {
                if (cache.transform_indices.buffer == VK_NULL_HANDLE) {
                    if (!createDeviceBuffer(allocator, kPlaceholderSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            cache.transform_indices,
                                            "vulkan.point_cloud.buffer",
                                            "transform_indices.placeholder")) {
                        return std::unexpected<std::string>("Failed to allocate indices placeholder");
                    }
                }
                cache.cached_transform_indices_ptr = nullptr;
                cache.cached_transform_indices_count = 0;
            }

            // visibility_mask (vector<bool> → uint32 SSBO)
            std::vector<std::uint32_t> vis_host;
            if (req.node_visibility_mask && !req.node_visibility_mask->empty()) {
                vis_host.reserve(req.node_visibility_mask->size());
                for (bool b : *req.node_visibility_mask) {
                    vis_host.push_back(b ? 1u : 0u);
                }
            }
            const bool visibility_changed = vis_host != cache.cached_visibility;
            if (visibility_changed || cache.visibility.buffer == VK_NULL_HANDLE) {
                if (!vis_host.empty()) {
                    if (auto r = uploadInto(cb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            vis_host.data(), vis_host.size() * sizeof(std::uint32_t),
                                            cache.visibility, "visibility");
                        !r) {
                        return r;
                    }
                } else if (cache.visibility.buffer == VK_NULL_HANDLE) {
                    if (!createDeviceBuffer(allocator,
                                            kPlaceholderSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            cache.visibility,
                                            "vulkan.point_cloud.buffer",
                                            "visibility.placeholder")) {
                        return std::unexpected<std::string>("Failed to allocate visibility placeholder");
                    }
                }
                cache.cached_visibility = std::move(vis_host);
                cache.cached_n_visibility = cache.cached_visibility.size();
            }

            const auto upload_mask_if_changed =
                [&](const lfs::core::Tensor* const mask,
                    ManagedBuffer& buffer,
                    const char* const what,
                    const void*& cached_ptr,
                    std::size_t& cached_id,
                    std::size_t& cached_count,
                    std::uint64_t* cached_revision,
                    const std::uint64_t revision = 0) -> std::expected<std::size_t, std::string> {
                if (!hasPointMask(mask, n_points)) {
                    cached_ptr = nullptr;
                    cached_id = 0;
                    cached_count = 0;
                    if (cached_revision) {
                        *cached_revision = 0;
                    }
                    return std::size_t{0};
                }

                const void* const mask_ptr = mask;
                const std::size_t mask_id = mask->debug_id();
                const bool changed =
                    buffer.buffer == VK_NULL_HANDLE ||
                    cached_ptr != mask_ptr ||
                    cached_id != mask_id ||
                    cached_count != n_points ||
                    (cached_revision && *cached_revision != revision);
                if (changed) {
                    auto host = maskTensorToUint32Host(*mask);
                    if (host.size() != n_points) {
                        return std::unexpected<std::string>(
                            std::format("{} mask size mismatch", what));
                    }
                    if (auto r = uploadInto(cb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            host.data(), host.size() * sizeof(std::uint32_t),
                                            buffer, what);
                        !r) {
                        return std::unexpected(r.error());
                    }
                    cached_ptr = mask_ptr;
                    cached_id = mask_id;
                    cached_count = n_points;
                    if (cached_revision) {
                        *cached_revision = revision;
                    }
                }
                return n_points;
            };

            auto selection_count = upload_mask_if_changed(
                req.selection_mask,
                cache.selection_mask,
                "selection mask",
                cache.cached_selection_mask_ptr,
                cache.cached_selection_mask_id,
                cache.cached_selection_mask_count,
                &cache.cached_selection_revision,
                req.selection_revision);
            if (!selection_count) {
                return std::unexpected(selection_count.error());
            }
            auto deleted_count = upload_mask_if_changed(
                req.deleted_mask,
                cache.deleted_mask,
                "deleted mask",
                cache.cached_deleted_mask_ptr,
                cache.cached_deleted_mask_id,
                cache.cached_deleted_mask_count,
                &cache.cached_deleted_mask_revision,
                req.deleted_mask_revision);
            if (!deleted_count) {
                return std::unexpected(deleted_count.error());
            }
            auto preview_count = upload_mask_if_changed(
                req.preview_selection_mask,
                cache.preview_selection_mask,
                "preview selection mask",
                cache.cached_preview_selection_mask_ptr,
                cache.cached_preview_selection_mask_id,
                cache.cached_preview_selection_mask_count,
                &cache.cached_preview_selection_revision,
                req.preview_selection_revision);
            if (!preview_count) {
                return std::unexpected(preview_count.error());
            }

            const bool overlay_active = *selection_count > 0 || *preview_count > 0;
            if (overlay_active) {
                const auto default_colors = lfs::rendering::defaultSelectionColorTable();
                const auto& colors = req.selection_colors ? *req.selection_colors : default_colors;
                const bool colors_changed =
                    !cache.cached_selection_colors_valid ||
                    cache.selection_colors.buffer == VK_NULL_HANDLE ||
                    cache.cached_selection_colors != colors;
                if (colors_changed) {
                    static_assert(sizeof(glm::vec4) == sizeof(float) * 4);
                    if (auto r = uploadInto(cb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            colors.data(), colors.size() * sizeof(glm::vec4),
                                            cache.selection_colors, "selection colors");
                        !r) {
                        return r;
                    }
                    cache.cached_selection_colors = colors;
                    cache.cached_selection_colors_valid = true;
                }
            }

            return {};
        }

        void updateDescriptorSet() {
            VkBuffer transforms_buf = (cache.cached_n_transforms > 0)
                                          ? cache.transforms.buffer
                                          : placeholder.buffer;
            VkBuffer indices_buf = (cache.cached_transform_indices_count > 0)
                                       ? cache.transform_indices.buffer
                                       : placeholder.buffer;
            VkBuffer visibility_buf = (cache.cached_n_visibility > 0)
                                          ? cache.visibility.buffer
                                          : placeholder.buffer;
            VkBuffer deleted_buf = (cache.cached_deleted_mask_count > 0)
                                       ? cache.deleted_mask.buffer
                                       : placeholder.buffer;
            VkBuffer selection_buf = (cache.cached_selection_mask_count > 0)
                                         ? cache.selection_mask.buffer
                                         : placeholder.buffer;
            VkBuffer preview_selection_buf = (cache.cached_preview_selection_mask_count > 0)
                                                 ? cache.preview_selection_mask.buffer
                                                 : placeholder.buffer;
            const bool has_selection_colors =
                cache.cached_selection_colors_valid &&
                cache.selection_colors.buffer != VK_NULL_HANDLE;
            VkBuffer selection_colors_buf = has_selection_colors
                                                ? cache.selection_colors.buffer
                                                : placeholder.buffer;

            std::array<VkDescriptorBufferInfo, kDescriptorBindingCount> infos{};
            infos[kBindingModelTransforms].buffer = transforms_buf;
            infos[kBindingModelTransforms].range = VK_WHOLE_SIZE;
            infos[kBindingTransformIndices].buffer = indices_buf;
            infos[kBindingTransformIndices].range = VK_WHOLE_SIZE;
            infos[kBindingVisibility].buffer = visibility_buf;
            infos[kBindingVisibility].range = VK_WHOLE_SIZE;
            infos[kBindingSelectionMask].buffer = selection_buf;
            infos[kBindingSelectionMask].range = VK_WHOLE_SIZE;
            infos[kBindingPreviewSelectionMask].buffer = preview_selection_buf;
            infos[kBindingPreviewSelectionMask].range = VK_WHOLE_SIZE;
            infos[kBindingSelectionColors].buffer = selection_colors_buf;
            infos[kBindingSelectionColors].range = VK_WHOLE_SIZE;
            infos[kBindingDeletedMask].buffer = deleted_buf;
            infos[kBindingDeletedMask].range = VK_WHOLE_SIZE;

            std::array<VkWriteDescriptorSet, kDescriptorBindingCount> writes{};
            for (std::uint32_t i = 0; i < writes.size(); ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = descriptor_set;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &infos[i];
            }
            vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }

        std::expected<RenderResult, std::string> doRender(const RenderRequest& req,
                                                          OutputSlot output_slot) {
            std::lock_guard<std::mutex> command_lock(command_mutex);
            const std::size_t slot_idx = static_cast<std::size_t>(output_slot);
            if (slot_idx >= kSlotCount) {
                return std::unexpected<std::string>(std::format(
                    "Point-cloud render output slot is out of range (output_slot={}, slot_count={}) ({}:{})",
                    slot_idx,
                    kSlotCount,
                    __FILE__,
                    __LINE__));
            }
            if (req.size.x <= 0 || req.size.y <= 0) {
                return std::unexpected<std::string>(std::format(
                    "Point-cloud render size must be positive (observed_width={}, observed_height={}, output_slot={}) ({}:{})",
                    req.size.x,
                    req.size.y,
                    slot_idx,
                    __FILE__,
                    __LINE__));
            }

            // Wait for the previous frame on this renderer to finish before
            // touching slot resources — the cb is shared across slots and a
            // size change would otherwise destroy images the in-flight submit
            // is still sampling. (C4: bounded prewait.)
            {
                auto wait_ctx = makeWaitContext("point_cloud.render.prewait");
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                if (!wait_outcome.has_value() ||
                    *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
                    return std::unexpected<std::string>(
                        formatFenceWaitUnexpected("render prewait", wait_outcome));
                }
            }

            auto& slot = slots[slot_idx];
            const bool will_recreate = slot.color_image == VK_NULL_HANDLE ||
                                       slot.depth_image == VK_NULL_HANDLE ||
                                       slot.size != req.size;
            if (will_recreate) {
                // pcFence covers this renderer's CB only; the context's frame CB also
                // samples slot.color_image and must finish before destroySlot frees it.
                if (!context->waitForSubmittedFrames()) {
                    return std::unexpected<std::string>(
                        std::format("waitForSubmittedFrames failed before slot recreate: {}",
                                    context->lastError()));
                }
            }
            if (auto ensure = ensureOutputImages(slot, req.size); !ensure) {
                return std::unexpected<std::string>(ensure.error());
            }
            for (auto& s : pending_stagings) {
                destroyBuffer(allocator, s);
            }
            pending_stagings.clear();
            VkResult r = vkResetCommandBuffer(command_buffer, 0);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkResetCommandBuffer", r));
            }

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            r = vkBeginCommandBuffer(command_buffer, &bi);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkBeginCommandBuffer", r));
            }

            if (auto u = uploadIfChanged(command_buffer, req); !u) {
                const VkResult end_result = vkEndCommandBuffer(command_buffer);
                if (end_result != VK_SUCCESS) {
                    LOG_ERROR("Vulkan: {}",
                              formatVkCheckFailure(
                                  "vkEndCommandBuffer(command_buffer)",
                                  end_result,
                                  std::format("Point-cloud upload error cleanup could not end command-buffer recording (command_buffer={:#x}, upload_error={})",
                                              vkHandleValue(command_buffer),
                                              u.error()),
                                  __FILE__,
                                  __LINE__));
                }
                return std::unexpected<std::string>(u.error());
            }

            // Make the just-uploaded data visible to the vertex stage.
            VkMemoryBarrier2 xfer_to_vert{};
            xfer_to_vert.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            xfer_to_vert.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            xfer_to_vert.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            xfer_to_vert.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                                        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            xfer_to_vert.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                                         VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo xfer_dependency{};
            xfer_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            xfer_dependency.memoryBarrierCount = 1;
            xfer_dependency.pMemoryBarriers = &xfer_to_vert;
            vkCmdPipelineBarrier2(command_buffer, &xfer_dependency);

            updateDescriptorSet();

            VkDeviceSize zero_offsets[2] = {0, 0};
            VkBuffer vbufs[2] = {cache.positions.buffer, cache.colors.buffer};
            if (vbufs[0] == VK_NULL_HANDLE || vbufs[1] == VK_NULL_HANDLE ||
                cache.cached_positions_count > std::numeric_limits<std::uint32_t>::max() ||
                cache.positions.size < cache.cached_positions_count * sizeof(float) * 3u ||
                cache.colors.size < cache.cached_positions_count * sizeof(float) * 3u) {
                const std::string error = std::format(
                    "Point-cloud draw requires non-null vertex buffers, a 32-bit vertex count, and allocations large enough for all vertices (positions_buffer={:#x}, positions_size={}, colors_buffer={:#x}, colors_size={}, vertex_count={}, maximum_vertex_count={}, required_positions_bytes={}, required_colors_bytes={}) ({}:{})",
                    vkHandleValue(vbufs[0]),
                    cache.positions.size,
                    vkHandleValue(vbufs[1]),
                    cache.colors.size,
                    cache.cached_positions_count,
                    std::numeric_limits<std::uint32_t>::max(),
                    cache.cached_positions_count * sizeof(float) * 3u,
                    cache.cached_positions_count * sizeof(float) * 3u,
                    __FILE__,
                    __LINE__);
                const VkResult cleanup_result = vkEndCommandBuffer(command_buffer);
                if (cleanup_result != VK_SUCCESS) {
                    LOG_ERROR("Vulkan: {}",
                              vkError("vkEndCommandBuffer(command_buffer)",
                                      cleanup_result,
                                      std::format("Point-cloud invalid-draw cleanup could not end command-buffer recording (command_buffer={:#x})",
                                                  vkHandleValue(command_buffer))));
                }
                return std::unexpected<std::string>(error);
            }

            // Transition output images for rendering.
            context->imageBarriers().transitionImage(command_buffer, slot.color_image,
                                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            context->imageBarriers().transitionImage(command_buffer, slot.depth_image,
                                                     VK_IMAGE_ASPECT_DEPTH_BIT,
                                                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

            VkClearValue color_clear{};
            color_clear.color = {{req.background_color.r, req.background_color.g,
                                  req.background_color.b,
                                  req.transparent_background ? 0.0f : 1.0f}};
            VkClearValue depth_clear{};
            depth_clear.depthStencil = {1.0f, 0};

            VkRenderingAttachmentInfo color_attach{};
            color_attach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attach.imageView = slot.color_view;
            color_attach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_attach.clearValue = color_clear;

            VkRenderingAttachmentInfo depth_attach{};
            depth_attach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attach.imageView = slot.depth_view;
            depth_attach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attach.clearValue = depth_clear;

            VkRect2D area{};
            area.extent = {static_cast<std::uint32_t>(req.size.x),
                           static_cast<std::uint32_t>(req.size.y)};

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea = area;
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color_attach;
            rendering.pDepthAttachment = &depth_attach;

            vkCmdBeginRendering(command_buffer, &rendering);

            VkViewport vp{};
            vp.x = 0.0f;
            vp.y = 0.0f;
            vp.width = static_cast<float>(req.size.x);
            vp.height = static_cast<float>(req.size.y);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;
            vkCmdSetViewport(command_buffer, 0, 1, &vp);
            vkCmdSetScissor(command_buffer, 0, 1, &area);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

            vkCmdBindVertexBuffers(command_buffer, 0, 2, vbufs, zero_offsets);

            // 64 is the lower bound of Vulkan's guaranteed pointSizeRange; the
            // shader clamps gl_PointSize to this so very-near points don't
            // exceed device limits.
            constexpr int kMaxPointSize = 64;
            PushConstants push{};
            writePushConstants(push, req,
                               kMaxPointSize,
                               static_cast<int>(cache.cached_n_transforms),
                               static_cast<int>(cache.cached_n_visibility),
                               cache.cached_transform_indices_count > 0,
                               cache.cached_selection_mask_count > 0,
                               cache.cached_preview_selection_mask_count > 0,
                               cache.cached_deleted_mask_count > 0);
            vkCmdPushConstants(command_buffer, pipeline_layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(push), &push);

            const std::uint32_t vertex_count =
                static_cast<std::uint32_t>(cache.cached_positions_count);
            vkCmdDraw(command_buffer, vertex_count, 1, 0, 0);

            vkCmdEndRendering(command_buffer);

            // Transition both outputs to SHADER_READ_ONLY for downstream sampling.
            context->imageBarriers().transitionImage(command_buffer, slot.color_image,
                                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            context->imageBarriers().transitionImage(command_buffer, slot.depth_image,
                                                     VK_IMAGE_ASPECT_DEPTH_BIT,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            const VkImageLayout previous_color_layout = slot.color_layout;
            const VkImageLayout previous_depth_layout = slot.depth_layout;
            const auto restore_tracked_layouts = [&]() {
                context->imageBarriers().registerImage(slot.color_image,
                                                       VK_IMAGE_ASPECT_COLOR_BIT,
                                                       previous_color_layout);
                context->imageBarriers().registerImage(slot.depth_image,
                                                       VK_IMAGE_ASPECT_DEPTH_BIT,
                                                       previous_depth_layout);
            };

            r = vkEndCommandBuffer(command_buffer);
            if (r != VK_SUCCESS) {
                restore_tracked_layouts();
                return std::unexpected<std::string>(vkError("vkEndCommandBuffer", r));
            }

            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &command_buffer;

            // Phase 7C-P2 Appendix A.1: ResetPreWaitReplacement lifecycle.
            {
                using lfs::rendering::SubmissionTransition;
                using lfs::rendering::apply_submission_transition;
                auto begin = apply_submission_transition(
                    submission_state_,
                    SubmissionTransition::BeginLifecycle,
                    kPointCloudFencePolicy);
                if (!begin) {
                    restore_tracked_layouts();
                    return std::unexpected<std::string>(std::format(
                        "Point-cloud render SubmissionState T0 BeginLifecycle failed (detail={})",
                        begin.error().detail()));
                }
            }

            r = vkResetFences(device, 1, &fence);
            if (r != VK_SUCCESS) {
                // Binding Ruling 1: reset-failure replace is OUTSIDE SubmissionState
                // (no transition; lifecycle restarts at T0 with the fresh fence).
                restore_tracked_layouts();
                (void)replaceFenceSignaled("vkResetFences", r);
                return std::unexpected<std::string>(vkError("vkResetFences", r));
            }
            {
                using lfs::rendering::SubmissionTransition;
                using lfs::rendering::apply_submission_transition;
                auto fence_reset = apply_submission_transition(
                    submission_state_,
                    SubmissionTransition::FenceReset,
                    kPointCloudFencePolicy);
                if (!fence_reset) {
                    restore_tracked_layouts();
                    // Fence is unsignaled after successful reset but state machine
                    // rejected T1 — replace defensively so the next prewait sees
                    // a signaled fence (same recovery shape as a poisoned fence).
                    (void)replaceFenceSignaled("SubmissionState T1 FenceReset",
                                               VK_ERROR_INITIALIZATION_FAILED);
                    return std::unexpected<std::string>(std::format(
                        "Point-cloud render SubmissionState T1 FenceReset failed (detail={})",
                        fence_reset.error().detail()));
                }
            }
            const VkQueue submit_queue = context->graphicsQueue();
            if (submit_queue == VK_NULL_HANDLE || command_buffer == VK_NULL_HANDLE ||
                fence == VK_NULL_HANDLE || si.commandBufferCount != 1 ||
                si.pCommandBuffers == nullptr || si.pCommandBuffers[0] != command_buffer) {
                const std::string error = std::format(
                    "Point-cloud render submit requires a non-null queue, one expected command buffer, and a non-null fence (queue={:#x}, command_buffer={:#x}, fence={:#x}, command_buffer_count={}, command_buffer_array={:#x}, submitted_command_buffer={:#x}, wait_semaphore_count={}, signal_semaphore_count={}) ({}:{})",
                    vkHandleValue(submit_queue),
                    vkHandleValue(command_buffer),
                    vkHandleValue(fence),
                    si.commandBufferCount,
                    reinterpret_cast<std::uintptr_t>(si.pCommandBuffers),
                    si.pCommandBuffers != nullptr ? vkHandleValue(si.pCommandBuffers[0]) : 0,
                    si.waitSemaphoreCount,
                    si.signalSemaphoreCount,
                    __FILE__,
                    __LINE__);
                restore_tracked_layouts();
                rejectSubmissionAndMaybeReplace("point-cloud render submit integrity check",
                                                VK_ERROR_INITIALIZATION_FAILED);
                return std::unexpected<std::string>(error);
            }
            r = vkQueueSubmit(submit_queue, 1, &si, fence);
            if (r != VK_SUCCESS) {
                restore_tracked_layouts();
                rejectSubmissionAndMaybeReplace("vkQueueSubmit", r);
                return std::unexpected<std::string>(vkError("vkQueueSubmit", r));
            }
            {
                using lfs::rendering::SubmissionTransition;
                using lfs::rendering::apply_submission_transition;
                auto accepted = apply_submission_transition(
                    submission_state_,
                    SubmissionTransition::SubmitAccepted,
                    kPointCloudFencePolicy);
                if (!accepted) {
                    // Submit already queued — do not replace (would manufacture readiness).
                    LOG_ERROR(
                        "Point-cloud render SubmissionState T4 SubmitAccepted illegal after "
                        "successful vkQueueSubmit (detail={})",
                        accepted.error().detail());
                }
            }

            slot.color_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            slot.depth_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ++slot.generation;

            RenderResult result{};
            result.image = slot.color_image;
            result.image_view = slot.color_view;
            result.image_layout = slot.color_layout;
            result.generation = slot.generation;
            result.depth_image = slot.depth_image;
            result.depth_image_view = slot.depth_view;
            result.depth_image_layout = slot.depth_layout;
            result.depth_generation = slot.generation;
            result.size = slot.size;
            result.flip_y = false;
            return result;
        }

        std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImage(
            VulkanContext& ctx,
            OutputSlot output_slot) {
            std::lock_guard<std::mutex> command_lock(command_mutex);
            if (!initialized || context == nullptr) {
                return std::unexpected<std::string>("Point-cloud output readback requested before renderer initialization");
            }
            if (&ctx != context) {
                return std::unexpected<std::string>("Point-cloud output readback received a different Vulkan context");
            }

            const std::size_t slot_idx = static_cast<std::size_t>(output_slot);
            if (slot_idx >= kSlotCount) {
                return std::unexpected<std::string>(std::format(
                    "Point-cloud readback output slot is out of range (output_slot={}, slot_count={}) ({}:{})",
                    slot_idx,
                    kSlotCount,
                    __FILE__,
                    __LINE__));
            }
            auto& slot = slots[slot_idx];
            if (slot.color_image == VK_NULL_HANDLE || slot.size.x <= 0 || slot.size.y <= 0) {
                return std::unexpected<std::string>("Point-cloud output readback requested for an empty output slot");
            }

            if (!ctx.waitForSubmittedFrames()) {
                return std::unexpected<std::string>(ctx.lastError());
            }
            // C5: bounded readback prewait.
            {
                auto wait_ctx = makeWaitContext("point_cloud.readback.prewait");
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                if (!wait_outcome.has_value() ||
                    *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
                    return std::unexpected<std::string>(
                        formatFenceWaitUnexpected("readback prewait", wait_outcome));
                }
            }
            for (auto& s : pending_stagings) {
                destroyBuffer(allocator, s);
            }
            pending_stagings.clear();

            const VkDeviceSize byte_count =
                static_cast<VkDeviceSize>(slot.size.x) *
                static_cast<VkDeviceSize>(slot.size.y) *
                static_cast<VkDeviceSize>(4);
            if (byte_count == 0) {
                return std::unexpected<std::string>("Point-cloud output readback has zero bytes");
            }

            ScopedStagingBuffer staging{};
            staging.allocator = allocator;
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = byte_count;
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo alloc_info{};
            alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            alloc_info.flags =
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VkResult r = vmaCreateBuffer(
                allocator,
                &buffer_info,
                &alloc_info,
                &staging.buffer,
                &staging.allocation,
                &staging.allocation_info);
            if (r != VK_SUCCESS || staging.buffer == VK_NULL_HANDLE) {
                return std::unexpected<std::string>(vkError("vmaCreateBuffer(point-cloud readback)", r));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         staging.buffer,
                                         "point_cloud.output[{}].readback[{}]",
                                         slot_idx,
                                         byte_count);
            staging.vram_scope = "vulkan.point_cloud.readback_buffer";
            staging.vram_label = std::format("rgba:{}x{}", slot.size.x, slot.size.y);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                staging.vram_scope,
                staging.vram_label,
                static_cast<std::size_t>(staging.allocation_info.size));
            if (staging.allocation_info.pMappedData == nullptr ||
                staging.allocation_info.size < byte_count) {
                return std::unexpected<std::string>(std::format(
                    "Point-cloud readback staging allocation must be mapped and cover the copy (slot_index={}, buffer={:#x}, allocation={:#x}, mapped={:#x}, allocation_size={}, copy_size={}) ({}:{})",
                    slot_idx,
                    vkHandleValue(staging.buffer),
                    reinterpret_cast<std::uintptr_t>(staging.allocation),
                    reinterpret_cast<std::uintptr_t>(staging.allocation_info.pMappedData),
                    staging.allocation_info.size,
                    byte_count,
                    __FILE__,
                    __LINE__));
            }

            r = vkResetCommandBuffer(command_buffer, 0);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkResetCommandBuffer(point-cloud readback)", r));
            }

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            r = vkBeginCommandBuffer(command_buffer, &begin_info);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkBeginCommandBuffer(point-cloud readback)", r));
            }

            const VkImageLayout restore_layout =
                slot.color_layout != VK_IMAGE_LAYOUT_UNDEFINED
                    ? slot.color_layout
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ctx.imageBarriers().transitionImage(command_buffer,
                                                slot.color_image,
                                                VK_IMAGE_ASPECT_COLOR_BIT,
                                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

            VkBufferImageCopy copy_region{};
            copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy_region.imageSubresource.layerCount = 1;
            copy_region.imageExtent = {
                static_cast<std::uint32_t>(slot.size.x),
                static_cast<std::uint32_t>(slot.size.y),
                1};
            vkCmdCopyImageToBuffer(command_buffer,
                                   slot.color_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   staging.buffer,
                                   1,
                                   &copy_region);

            ctx.imageBarriers().transitionImage(command_buffer,
                                                slot.color_image,
                                                VK_IMAGE_ASPECT_COLOR_BIT,
                                                restore_layout);
            slot.color_layout = restore_layout;

            r = vkEndCommandBuffer(command_buffer);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkEndCommandBuffer(point-cloud readback)", r));
            }

            // Phase 7C-P2 Appendix A.1: ResetPreWaitReplacement lifecycle (readback).
            {
                using lfs::rendering::SubmissionTransition;
                using lfs::rendering::apply_submission_transition;
                auto begin = apply_submission_transition(
                    submission_state_,
                    SubmissionTransition::BeginLifecycle,
                    kPointCloudFencePolicy);
                if (!begin) {
                    return std::unexpected<std::string>(std::format(
                        "Point-cloud readback SubmissionState T0 BeginLifecycle failed (detail={})",
                        begin.error().detail()));
                }
            }

            r = vkResetFences(device, 1, &fence);
            if (r != VK_SUCCESS) {
                // Binding Ruling 1: reset-failure replace is OUTSIDE SubmissionState.
                (void)replaceFenceSignaled("vkResetFences(point-cloud readback)", r);
                return std::unexpected<std::string>(vkError("vkResetFences(point-cloud readback)", r));
            }
            {
                using lfs::rendering::SubmissionTransition;
                using lfs::rendering::apply_submission_transition;
                auto fence_reset = apply_submission_transition(
                    submission_state_,
                    SubmissionTransition::FenceReset,
                    kPointCloudFencePolicy);
                if (!fence_reset) {
                    (void)replaceFenceSignaled("SubmissionState T1 FenceReset(point-cloud readback)",
                                               VK_ERROR_INITIALIZATION_FAILED);
                    return std::unexpected<std::string>(std::format(
                        "Point-cloud readback SubmissionState T1 FenceReset failed (detail={})",
                        fence_reset.error().detail()));
                }
            }
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            const VkQueue submit_queue = ctx.graphicsQueue();
            if (submit_queue == VK_NULL_HANDLE || command_buffer == VK_NULL_HANDLE ||
                fence == VK_NULL_HANDLE || submit_info.commandBufferCount != 1 ||
                submit_info.pCommandBuffers == nullptr ||
                submit_info.pCommandBuffers[0] != command_buffer) {
                const std::string error = std::format(
                    "Point-cloud readback submit requires a non-null queue, one expected command buffer, and a non-null fence (slot_index={}, queue={:#x}, command_buffer={:#x}, fence={:#x}, command_buffer_count={}, command_buffer_array={:#x}, submitted_command_buffer={:#x}, wait_semaphore_count={}, signal_semaphore_count={}) ({}:{})",
                    slot_idx,
                    vkHandleValue(submit_queue),
                    vkHandleValue(command_buffer),
                    vkHandleValue(fence),
                    submit_info.commandBufferCount,
                    reinterpret_cast<std::uintptr_t>(submit_info.pCommandBuffers),
                    submit_info.pCommandBuffers != nullptr
                        ? vkHandleValue(submit_info.pCommandBuffers[0])
                        : 0,
                    submit_info.waitSemaphoreCount,
                    submit_info.signalSemaphoreCount,
                    __FILE__,
                    __LINE__);
                rejectSubmissionAndMaybeReplace("point-cloud readback submit integrity check",
                                                VK_ERROR_INITIALIZATION_FAILED);
                return std::unexpected<std::string>(error);
            }
            r = vkQueueSubmit(submit_queue, 1, &submit_info, fence);
            if (r != VK_SUCCESS) {
                rejectSubmissionAndMaybeReplace("vkQueueSubmit(point-cloud readback)", r);
                return std::unexpected<std::string>(vkError("vkQueueSubmit(point-cloud readback)", r));
            }
            {
                using lfs::rendering::SubmissionTransition;
                using lfs::rendering::apply_submission_transition;
                auto accepted = apply_submission_transition(
                    submission_state_,
                    SubmissionTransition::SubmitAccepted,
                    kPointCloudFencePolicy);
                if (!accepted) {
                    LOG_ERROR(
                        "Point-cloud readback SubmissionState T4 SubmitAccepted illegal after "
                        "successful vkQueueSubmit (detail={})",
                        accepted.error().detail());
                }
            }
            // C6: bounded post-submit wait.
            {
                auto wait_ctx = makeWaitContext("point_cloud.readback.wait_fence");
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                if (!wait_outcome.has_value() ||
                    *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
                    return std::unexpected<std::string>(
                        formatFenceWaitUnexpected("readback wait_fence", wait_outcome));
                }
            }

            r = vmaInvalidateAllocation(allocator, staging.allocation, 0, byte_count);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vmaInvalidateAllocation(point-cloud readback)", r));
            }

            const auto* const rgba = static_cast<const std::uint8_t*>(staging.allocation_info.pMappedData);
            const int width = slot.size.x;
            const int height = slot.size.y;
            const std::size_t pixel_count =
                static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
            std::vector<float> hwc(pixel_count * 3u);
            for (std::size_t i = 0; i < pixel_count; ++i) {
                const std::size_t src = i * 4u;
                const std::size_t dst = i * 3u;
                hwc[dst] = static_cast<float>(rgba[src]) / 255.0f;
                hwc[dst + 1u] = static_cast<float>(rgba[src + 1u]) / 255.0f;
                hwc[dst + 2u] = static_cast<float>(rgba[src + 2u]) / 255.0f;
            }

            auto tensor = lfs::core::Tensor::from_vector(
                hwc,
                {static_cast<std::size_t>(height), static_cast<std::size_t>(width), std::size_t{3}},
                lfs::core::Device::CPU);
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }
    };

    PointCloudVulkanRenderer::PointCloudVulkanRenderer()
        : impl_(std::make_unique<Impl>()) {}

    PointCloudVulkanRenderer::~PointCloudVulkanRenderer() = default;

    std::expected<PointCloudVulkanRenderer::RenderResult, std::string>
    PointCloudVulkanRenderer::render(VulkanContext& context, const RenderRequest& request,
                                     OutputSlot output_slot) {
        if (auto r = impl_->ensureInitialized(context); !r) {
            return std::unexpected<std::string>(r.error());
        }
        return impl_->doRender(request, output_slot);
    }

    std::expected<std::shared_ptr<lfs::core::Tensor>, std::string>
    PointCloudVulkanRenderer::readOutputImage(VulkanContext& context, OutputSlot output_slot) {
        return impl_->readOutputImage(context, output_slot);
    }

    void PointCloudVulkanRenderer::reset() {
        impl_->destroy();
    }

} // namespace lfs::vis
